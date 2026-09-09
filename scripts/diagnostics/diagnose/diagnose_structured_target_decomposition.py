from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(ROOT, "sagin_marl")):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        raise RuntimeError("Could not locate repository root.")
    ROOT = parent
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO
from sagin_marl.rl.structured_rollout_debug import copy_prefix_rollout_buffer
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


STAGE_NAMES = {0: "accel", 1: "sat", 2: "bw"}


def _load_bundle(cfg, run_dir: Path, update: int, device: torch.device):
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    load_checkpoint_forgiving(bundle.actor, str(run_dir / f"actor_u{update:04d}.pt"), map_location=device, strict=True)
    load_checkpoint_forgiving(bundle.critic, str(run_dir / f"critic_u{update:04d}.pt"), map_location=device, strict=True)
    bundle.actor.to(device).eval()
    bundle.critic.to(device).eval()
    return bundle


def _collect_full_episode(
    cfg,
    actor,
    critic,
    *,
    device: torch.device,
    deterministic: bool,
    seed: int,
) -> StructuredRolloutBuffer:
    env = make_structured_env(cfg, mode="script")
    try:
        env.reset(seed=int(seed))
        driver = as_structured_driver(env)
        buffer = StructuredRolloutBuffer()
        algo = StructuredMAPPO(
            actor,
            critic,
            gamma=float(cfg.gamma),
            gae_lambda=float(cfg.gae_lambda),
            clip_ratio=float(getattr(cfg, "clip_ratio", 0.2) or 0.2),
            value_coef=float(getattr(cfg, "value_coef", 0.5) or 0.5),
            entropy_coef=float(getattr(cfg, "entropy_coef", 0.0) or 0.0),
            max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5) or 0.5),
            ppo_epochs=int(getattr(cfg, "ppo_epochs", 1) or 1),
            num_mini_batch=int(getattr(cfg, "num_mini_batch", 1) or 1),
            device=device,
        )
        done = False
        while not done:
            step_result = algo.collect_env_step(driver, buffer, deterministic=deterministic)
            terminated = bool(next(iter(step_result.terminations.values())))
            truncated = bool(next(iter(step_result.truncations.values())))
            done = terminated or truncated
        return buffer
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def _prefix_buffer(full_buffer: StructuredRolloutBuffer, rollout_env_steps: int) -> StructuredRolloutBuffer:
    return copy_prefix_rollout_buffer(full_buffer, int(rollout_env_steps))


def _decompose_prefix_targets(
    buffer: StructuredRolloutBuffer,
    *,
    gamma_env: float,
    gae_lambda: float,
    final_bootstrap_value: float,
) -> Dict[str, np.ndarray]:
    return_view = buffer.build_return_view()
    n = int(return_view.transition_count)
    stage_ids = np.asarray(return_view.stage_ids, dtype=np.int64)
    values = np.asarray(return_view.values, dtype=np.float64)
    rewards = np.asarray(return_view.rewards, dtype=np.float64)
    terminated = np.asarray(np.logical_or(return_view.terminated, return_view.truncated), dtype=bool)

    reward_comp = np.zeros((n,), dtype=np.float64)
    same_stage_boot_comp = np.zeros((n,), dtype=np.float64)
    next_step_boot_comp = np.zeros((n,), dtype=np.float64)
    tail_boot_comp = np.zeros((n,), dtype=np.float64)
    train_target = np.zeros((n,), dtype=np.float64)

    for idx in range(n - 1, -1, -1):
        gamma = float(gamma_env) if int(stage_ids[idx]) == 2 else 1.0
        if terminated[idx]:
            reward_comp[idx] = rewards[idx]
            same_stage_boot_comp[idx] = 0.0
            next_step_boot_comp[idx] = 0.0
            tail_boot_comp[idx] = 0.0
            train_target[idx] = rewards[idx]
            continue

        is_last = idx == (n - 1)
        if is_last:
            reward_comp[idx] = rewards[idx]
            same_stage_boot_comp[idx] = 0.0
            next_step_boot_comp[idx] = 0.0
            tail_boot_comp[idx] = gamma * float(final_bootstrap_value)
            train_target[idx] = reward_comp[idx] + tail_boot_comp[idx]
            continue

        next_stage_id = int(stage_ids[idx + 1])
        next_value = float(values[idx + 1])
        same_step_direct = 0.0
        next_step_direct = 0.0
        if int(stage_ids[idx]) in (0, 1):
            same_step_direct = gamma * (1.0 - float(gae_lambda)) * next_value
        elif next_stage_id == 0:
            next_step_direct = gamma * (1.0 - float(gae_lambda)) * next_value

        reward_comp[idx] = rewards[idx] + gamma * float(gae_lambda) * reward_comp[idx + 1]
        same_stage_boot_comp[idx] = same_step_direct + gamma * float(gae_lambda) * same_stage_boot_comp[idx + 1]
        next_step_boot_comp[idx] = next_step_direct + gamma * float(gae_lambda) * next_step_boot_comp[idx + 1]
        tail_boot_comp[idx] = gamma * float(gae_lambda) * tail_boot_comp[idx + 1]
        train_target[idx] = (
            reward_comp[idx]
            + same_stage_boot_comp[idx]
            + next_step_boot_comp[idx]
            + tail_boot_comp[idx]
        )

    return {
        "stage_ids": stage_ids,
        "values": values,
        "rewards": rewards,
        "train_target": train_target,
        "reward_component": reward_comp,
        "same_stage_boot_component": same_stage_boot_comp,
        "next_step_boot_component": next_step_boot_comp,
        "tail_boot_component": tail_boot_comp,
    }


def _stage_component_summary(component: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    component = np.asarray(component, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    abs_target_mean = float(np.mean(np.abs(target))) if target.size else 0.0
    return {
        "mean": float(np.mean(component)) if component.size else 0.0,
        "abs_mean": float(np.mean(np.abs(component))) if component.size else 0.0,
        "share_of_abs_target": (
            float(np.mean(np.abs(component))) / abs_target_mean if abs_target_mean > 1.0e-12 else 0.0
        ),
    }


def diagnose_run(
    run_dir: Path,
    *,
    updates: Iterable[int],
    episodes: int,
    episode_seed_base: int,
    deterministic: bool,
    rollout_env_steps: int,
    device: torch.device,
) -> Dict[str, object]:
    cfg_path = run_dir / "config_source.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config_source.yaml in {run_dir}")
    cfg = load_config(str(cfg_path))
    results: Dict[str, object] = {
        "run_dir": str(run_dir),
        "episodes": int(episodes),
        "episode_seed_base": int(episode_seed_base),
        "deterministic": bool(deterministic),
        "rollout_env_steps": int(rollout_env_steps),
        "updates": {},
    }
    for update in updates:
        bundle = _load_bundle(cfg, run_dir, int(update), device)
        comp_by_stage: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        episode_rows: List[Dict[str, float]] = []
        for ep_idx in range(int(episodes)):
            seed = int(episode_seed_base) + ep_idx
            full_buffer = _collect_full_episode(
                cfg,
                bundle.actor,
                bundle.critic,
                device=device,
                deterministic=deterministic,
                seed=seed,
            )
            prefix_buffer = _prefix_buffer(full_buffer, rollout_env_steps=int(rollout_env_steps))
            prefix_return_view = prefix_buffer.build_return_view()
            final_bootstrap_value = 0.0
            bootstrap_worlds = prefix_buffer.build_bootstrap_world_state_dict()
            if bootstrap_worlds:
                algo = StructuredMAPPO(
                    bundle.actor,
                    bundle.critic,
                    gamma=float(cfg.gamma),
                    gae_lambda=float(cfg.gae_lambda),
                    clip_ratio=float(getattr(cfg, "clip_ratio", 0.2) or 0.2),
                    value_coef=float(getattr(cfg, "value_coef", 0.5) or 0.5),
                    entropy_coef=float(getattr(cfg, "entropy_coef", 0.0) or 0.0),
                    max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5) or 0.5),
                    ppo_epochs=int(getattr(cfg, "ppo_epochs", 1) or 1),
                    num_mini_batch=int(getattr(cfg, "num_mini_batch", 1) or 1),
                    device=device,
                )
                final_bootstrap_value = float(algo.bootstrap_value(next(iter(bootstrap_worlds.values()))))
            episode_rows.append(
                {
                    "seed": float(seed),
                    "full_episode_steps": float(full_buffer.build_return_view().transition_count // 3),
                    "prefix_steps": float(prefix_return_view.transition_count // 3),
                    "final_bootstrap_value": float(final_bootstrap_value),
                }
            )
            dec = _decompose_prefix_targets(
                prefix_buffer,
                gamma_env=float(cfg.gamma),
                gae_lambda=float(cfg.gae_lambda),
                final_bootstrap_value=float(final_bootstrap_value),
            )
            for stage_id in (0, 1, 2):
                mask = dec["stage_ids"] == stage_id
                for key in (
                    "train_target",
                    "reward_component",
                    "same_stage_boot_component",
                    "next_step_boot_component",
                    "tail_boot_component",
                ):
                    comp_by_stage[stage_id][key].extend(dec[key][mask].tolist())
        update_summary: Dict[str, object] = {"episode_rows": episode_rows}
        for stage_id, stage_name in STAGE_NAMES.items():
            target = np.asarray(comp_by_stage[stage_id]["train_target"], dtype=np.float64)
            reward_component = np.asarray(comp_by_stage[stage_id]["reward_component"], dtype=np.float64)
            same_stage_component = np.asarray(comp_by_stage[stage_id]["same_stage_boot_component"], dtype=np.float64)
            next_step_component = np.asarray(comp_by_stage[stage_id]["next_step_boot_component"], dtype=np.float64)
            tail_component = np.asarray(comp_by_stage[stage_id]["tail_boot_component"], dtype=np.float64)
            residual = target - reward_component - same_stage_component - next_step_component - tail_component
            update_summary[stage_name] = {
                "n": int(target.size),
                "target_mean": float(np.mean(target)) if target.size else 0.0,
                "target_abs_mean": float(np.mean(np.abs(target))) if target.size else 0.0,
                "reward_component": _stage_component_summary(reward_component, target),
                "same_stage_boot_component": _stage_component_summary(same_stage_component, target),
                "next_step_boot_component": _stage_component_summary(next_step_component, target),
                "tail_boot_component": _stage_component_summary(tail_component, target),
                "closure_residual_max_abs": float(np.max(np.abs(residual))) if residual.size else 0.0,
            }
        results["updates"][f"u{int(update):04d}"] = update_summary
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--updates", type=int, nargs="+", default=[100, 150, 200])
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--episode_seed_base", type=int, default=42000)
    parser.add_argument("--policy_mode", choices=["deterministic", "stochastic"], default="stochastic")
    parser.add_argument("--rollout_env_steps", type=int, default=50)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_path = Path(args.out) if args.out else run_dir / "structured_target_decomposition_diag.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = diagnose_run(
        run_dir,
        updates=args.updates,
        episodes=int(args.episodes),
        episode_seed_base=int(args.episode_seed_base),
        deterministic=(args.policy_mode == "deterministic"),
        rollout_env_steps=int(args.rollout_env_steps),
        device=device,
    )
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote structured target decomposition summary to {out_path}")
    for update_key, stage_map in summary["updates"].items():
        print(f"[{update_key}]")
        for stage_name in ("accel", "sat", "bw"):
            row = stage_map[stage_name]
            print(
                f"  {stage_name}: "
                f"reward={row['reward_component']['share_of_abs_target']:.3f} "
                f"same_stage={row['same_stage_boot_component']['share_of_abs_target']:.3f} "
                f"next_step={row['next_step_boot_component']['share_of_abs_target']:.3f} "
                f"tail={row['tail_boot_component']['share_of_abs_target']:.3f}"
            )


if __name__ == "__main__":
    main()
