from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = os.path.dirname(os.path.dirname(__file__))
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


def _rankdata(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return np.zeros((0,), dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_vals = a[order]
    start = 0
    n = int(a.size)
    while start < n:
        end = start + 1
        while end < n and sorted_vals[end] == sorted_vals[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size <= 1 or y.size <= 1:
        return 0.0
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError("x and y must have the same size")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 1e-12 or y_std <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size <= 1 or y.size <= 1:
        return 0.0
    return _safe_corr(_rankdata(np.asarray(x)), _rankdata(np.asarray(y)))


def _explained_variance(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size <= 1 or target.size <= 1:
        return 0.0
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    target_var = float(np.var(target))
    if target_var <= 1e-12:
        return 0.0
    return float(1.0 - np.var(target - pred) / target_var)


def _pair_summary(lhs: np.ndarray, rhs: np.ndarray) -> Dict[str, float]:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    diff = lhs - rhs
    return {
        "n": int(lhs.size),
        "lhs_mean": float(np.mean(lhs)) if lhs.size else 0.0,
        "rhs_mean": float(np.mean(rhs)) if rhs.size else 0.0,
        "gap_mean": float(np.mean(diff)) if diff.size else 0.0,
        "mae": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rmse": float(np.sqrt(np.mean(diff**2))) if diff.size else 0.0,
        "pearson": _safe_corr(lhs, rhs),
        "spearman": _safe_spearman(lhs, rhs),
        "explained_variance_lhs_as_pred": _explained_variance(lhs, rhs),
    }


def _component_summary(component: np.ndarray, target: np.ndarray) -> Dict[str, float]:
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


def _compute_step_targets(
    buffer: StructuredRolloutBuffer,
    *,
    gamma_env: float,
    gae_lambda: float,
    final_bootstrap_value: float,
) -> Dict[str, np.ndarray]:
    return_view = buffer.build_return_view()
    n = int(return_view.transition_count)
    if n % 3 != 0:
        raise ValueError(f"Expected transitions to be grouped into env steps of 3, got {n}")
    stage_ids = np.asarray(return_view.stage_ids, dtype=np.int64)
    values = np.asarray(return_view.values, dtype=np.float64)
    rewards = np.asarray(return_view.rewards, dtype=np.float64)
    terminals = np.asarray(np.logical_or(return_view.terminated, return_view.truncated), dtype=bool)

    num_env_steps = n // 3
    step_targets = np.zeros((n,), dtype=np.float64)
    reward_comp = np.zeros((n,), dtype=np.float64)
    next_step_boot_comp = np.zeros((n,), dtype=np.float64)
    tail_boot_comp = np.zeros((n,), dtype=np.float64)

    next_step_return = 0.0
    next_step_reward_comp = 0.0
    next_step_boot = 0.0
    next_step_tail = 0.0

    for step_idx in range(num_env_steps - 1, -1, -1):
        base = step_idx * 3
        idx_accel = base
        idx_bw = base + 2
        step_reward = float(rewards[idx_bw])
        terminal = bool(terminals[idx_bw])

        if terminal:
            g_step = step_reward
            reward_step = step_reward
            boot_step = 0.0
            tail_step = 0.0
        elif step_idx == (num_env_steps - 1):
            g_step = step_reward + float(gamma_env) * float(final_bootstrap_value)
            reward_step = step_reward
            boot_step = 0.0
            tail_step = float(gamma_env) * float(final_bootstrap_value)
        else:
            next_accel_value = float(values[idx_accel + 3])
            g_step = step_reward + float(gamma_env) * (
                (1.0 - float(gae_lambda)) * next_accel_value + float(gae_lambda) * next_step_return
            )
            reward_step = step_reward + float(gamma_env) * float(gae_lambda) * next_step_reward_comp
            boot_step = float(gamma_env) * (1.0 - float(gae_lambda)) * next_accel_value + float(gamma_env) * float(gae_lambda) * next_step_boot
            tail_step = float(gamma_env) * float(gae_lambda) * next_step_tail

        step_targets[base : base + 3] = g_step
        reward_comp[base : base + 3] = reward_step
        next_step_boot_comp[base : base + 3] = boot_step
        tail_boot_comp[base : base + 3] = tail_step

        next_step_return = g_step
        next_step_reward_comp = reward_step
        next_step_boot = boot_step
        next_step_tail = tail_step

    return {
        "stage_ids": stage_ids,
        "values": values,
        "rewards": rewards,
        "step_target": step_targets,
        "reward_component": reward_comp,
        "next_step_boot_component": next_step_boot_comp,
        "tail_boot_component": tail_boot_comp,
    }


def _collect_episode_compare(
    cfg,
    actor,
    critic,
    *,
    device: torch.device,
    deterministic: bool,
    seed: int,
    rollout_env_steps: int,
) -> Dict[str, np.ndarray]:
    full_buffer = _collect_full_episode(
        cfg,
        actor,
        critic,
        device=device,
        deterministic=deterministic,
        seed=seed,
    )
    prefix_buffer = _prefix_buffer(full_buffer, rollout_env_steps=int(rollout_env_steps))
    full_mc = full_buffer.compute_gae(gamma_env=float(cfg.gamma), gae_lambda=1.0, bootstrap_value=0.0)
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
    prefix_return_view = prefix_buffer.build_return_view()
    final_bootstrap_value = 0.0
    bootstrap_worlds = prefix_buffer.build_bootstrap_world_state_dict()
    if bootstrap_worlds:
        final_bootstrap_value = float(algo.bootstrap_value(next(iter(bootstrap_worlds.values()))))

    current_target = algo.compute_returns_and_advantages(
        prefix_buffer,
        prefix_buffer.build_bootstrap_view() if len(prefix_buffer) > 0 else None,
    )
    step_target = _compute_step_targets(
        prefix_buffer,
        gamma_env=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        final_bootstrap_value=float(final_bootstrap_value),
    )
    prefix_len = int(prefix_return_view.transition_count)
    full_mc_returns = np.asarray(full_mc["returns"][:prefix_len], dtype=np.float64)
    return {
        "stage_ids": step_target["stage_ids"],
        "values": step_target["values"],
        "rewards": step_target["rewards"],
        "current_train_target": np.asarray(current_target["returns"], dtype=np.float64),
        "step_target": np.asarray(step_target["step_target"], dtype=np.float64),
        "mc_returns": full_mc_returns,
        "step_reward_component": np.asarray(step_target["reward_component"], dtype=np.float64),
        "step_next_step_boot_component": np.asarray(step_target["next_step_boot_component"], dtype=np.float64),
        "step_tail_boot_component": np.asarray(step_target["tail_boot_component"], dtype=np.float64),
        "prefix_steps": np.asarray([prefix_len // 3], dtype=np.int64),
        "full_episode_steps": np.asarray([full_buffer.build_return_view().transition_count // 3], dtype=np.int64),
        "final_bootstrap_value": np.asarray([final_bootstrap_value], dtype=np.float64),
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
        values_by_stage: Dict[int, List[float]] = defaultdict(list)
        current_by_stage: Dict[int, List[float]] = defaultdict(list)
        step_by_stage: Dict[int, List[float]] = defaultdict(list)
        mc_by_stage: Dict[int, List[float]] = defaultdict(list)
        step_reward_by_stage: Dict[int, List[float]] = defaultdict(list)
        step_boot_by_stage: Dict[int, List[float]] = defaultdict(list)
        step_tail_by_stage: Dict[int, List[float]] = defaultdict(list)
        episode_rows: List[Dict[str, float]] = []
        for ep_idx in range(int(episodes)):
            seed = int(episode_seed_base) + ep_idx
            episode = _collect_episode_compare(
                cfg,
                bundle.actor,
                bundle.critic,
                device=device,
                deterministic=deterministic,
                seed=seed,
                rollout_env_steps=int(rollout_env_steps),
            )
            episode_rows.append(
                {
                    "seed": float(seed),
                    "prefix_steps": float(episode["prefix_steps"][0]),
                    "full_episode_steps": float(episode["full_episode_steps"][0]),
                    "final_bootstrap_value": float(episode["final_bootstrap_value"][0]),
                }
            )
            for stage_id in (0, 1, 2):
                mask = episode["stage_ids"] == stage_id
                values_by_stage[stage_id].extend(episode["values"][mask].tolist())
                current_by_stage[stage_id].extend(episode["current_train_target"][mask].tolist())
                step_by_stage[stage_id].extend(episode["step_target"][mask].tolist())
                mc_by_stage[stage_id].extend(episode["mc_returns"][mask].tolist())
                step_reward_by_stage[stage_id].extend(episode["step_reward_component"][mask].tolist())
                step_boot_by_stage[stage_id].extend(episode["step_next_step_boot_component"][mask].tolist())
                step_tail_by_stage[stage_id].extend(episode["step_tail_boot_component"][mask].tolist())
        update_summary: Dict[str, object] = {"episode_rows": episode_rows}
        for stage_id, stage_name in STAGE_NAMES.items():
            values = np.asarray(values_by_stage[stage_id], dtype=np.float64)
            current_target = np.asarray(current_by_stage[stage_id], dtype=np.float64)
            step_target = np.asarray(step_by_stage[stage_id], dtype=np.float64)
            mc_target = np.asarray(mc_by_stage[stage_id], dtype=np.float64)
            step_reward = np.asarray(step_reward_by_stage[stage_id], dtype=np.float64)
            step_boot = np.asarray(step_boot_by_stage[stage_id], dtype=np.float64)
            step_tail = np.asarray(step_tail_by_stage[stage_id], dtype=np.float64)
            residual = step_target - step_reward - step_boot - step_tail
            update_summary[stage_name] = {
                "current_train_target_vs_mc": _pair_summary(current_target, mc_target),
                "step_target_vs_mc": _pair_summary(step_target, mc_target),
                "current_vs_step_target": _pair_summary(current_target, step_target),
                "value_vs_current_train_target": _pair_summary(values, current_target),
                "value_vs_step_target": _pair_summary(values, step_target),
                "value_vs_mc": _pair_summary(values, mc_target),
                "step_target_decomposition": {
                    "target_mean": float(np.mean(step_target)) if step_target.size else 0.0,
                    "target_abs_mean": float(np.mean(np.abs(step_target))) if step_target.size else 0.0,
                    "reward_component": _component_summary(step_reward, step_target),
                    "next_step_boot_component": _component_summary(step_boot, step_target),
                    "tail_boot_component": _component_summary(step_tail, step_target),
                    "closure_residual_max_abs": float(np.max(np.abs(residual))) if residual.size else 0.0,
                },
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
    out_path = Path(args.out) if args.out else run_dir / "structured_step_target_diag.json"
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
    print(f"Wrote structured step-target summary to {out_path}")
    for update_key, stage_map in summary["updates"].items():
        print(f"[{update_key}]")
        for stage_name in ("accel", "sat", "bw"):
            stage_summary = stage_map[stage_name]
            current_vs_mc = stage_summary["current_train_target_vs_mc"]
            step_vs_mc = stage_summary["step_target_vs_mc"]
            step_decomp = stage_summary["step_target_decomposition"]
            print(
                f"  {stage_name}: "
                f"current_vs_mc_rho={current_vs_mc['pearson']:.3f} "
                f"step_vs_mc_rho={step_vs_mc['pearson']:.3f} "
                f"step_reward={step_decomp['reward_component']['share_of_abs_target']:.3f} "
                f"step_boot={step_decomp['next_step_boot_component']['share_of_abs_target']:.3f} "
                f"step_tail={step_decomp['tail_boot_component']['share_of_abs_target']:.3f}"
            )


if __name__ == "__main__":
    main()
