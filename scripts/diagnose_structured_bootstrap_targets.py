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
    prefix_return_view = prefix_buffer.build_return_view()
    full_return_view = full_buffer.build_return_view()
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
    train_target = algo.compute_returns_and_advantages(
        prefix_buffer,
        prefix_buffer.build_bootstrap_view() if len(prefix_buffer) > 0 else None,
    )
    prefix_len = int(prefix_return_view.transition_count)
    stage_ids = np.asarray(prefix_return_view.stage_ids, dtype=np.int64)
    values = np.asarray(prefix_return_view.values, dtype=np.float64)
    rewards = np.asarray(prefix_return_view.rewards, dtype=np.float64)
    terminals = np.asarray(
        np.logical_or(prefix_return_view.terminated, prefix_return_view.truncated),
        dtype=np.float64,
    )
    full_mc_returns = np.asarray(full_mc["returns"][:prefix_len], dtype=np.float64)
    train_returns = np.asarray(train_target["returns"], dtype=np.float64)
    next_bootstrap_values = np.zeros((prefix_len,), dtype=np.float64)
    bootstrap_worlds = prefix_buffer.build_bootstrap_world_state_dict()
    if prefix_len > 0 and bootstrap_worlds:
        bootstrap_next_value = float(algo.bootstrap_value(next(iter(bootstrap_worlds.values()))))
        next_bootstrap_values[-1] = bootstrap_next_value
    full_episode_steps = int(full_return_view.transition_count // 3)
    prefix_steps = int(prefix_len // 3)
    return {
        "stage_ids": stage_ids,
        "values": values,
        "rewards": rewards,
        "terminals": terminals,
        "train_returns": train_returns,
        "mc_returns": full_mc_returns,
        "prefix_steps": np.asarray([prefix_steps], dtype=np.int64),
        "full_episode_steps": np.asarray([full_episode_steps], dtype=np.int64),
        "bootstrap_next_value": next_bootstrap_values,
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
        train_returns_by_stage: Dict[int, List[float]] = defaultdict(list)
        mc_returns_by_stage: Dict[int, List[float]] = defaultdict(list)
        rewards_by_stage: Dict[int, List[float]] = defaultdict(list)
        episode_step_rows: List[Dict[str, float]] = []
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
            episode_step_rows.append(
                {
                    "seed": float(seed),
                    "prefix_steps": float(episode["prefix_steps"][0]),
                    "full_episode_steps": float(episode["full_episode_steps"][0]),
                    "last_bootstrap_value": float(episode["bootstrap_next_value"][-1]) if episode["bootstrap_next_value"].size else 0.0,
                }
            )
            for stage_id in (0, 1, 2):
                mask = episode["stage_ids"] == stage_id
                values_by_stage[stage_id].extend(episode["values"][mask].tolist())
                train_returns_by_stage[stage_id].extend(episode["train_returns"][mask].tolist())
                mc_returns_by_stage[stage_id].extend(episode["mc_returns"][mask].tolist())
                rewards_by_stage[stage_id].extend(episode["rewards"][mask].tolist())
        update_summary: Dict[str, object] = {"episode_step_rows": episode_step_rows}
        for stage_id, stage_name in STAGE_NAMES.items():
            values = np.asarray(values_by_stage[stage_id], dtype=np.float64)
            train_returns = np.asarray(train_returns_by_stage[stage_id], dtype=np.float64)
            mc_returns = np.asarray(mc_returns_by_stage[stage_id], dtype=np.float64)
            rewards = np.asarray(rewards_by_stage[stage_id], dtype=np.float64)
            update_summary[stage_name] = {
                "reward_mean": float(np.mean(rewards)) if rewards.size else 0.0,
                "train_target_vs_mc": _pair_summary(train_returns, mc_returns),
                "value_vs_train_target": _pair_summary(values, train_returns),
                "value_vs_mc": _pair_summary(values, mc_returns),
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
    out_path = Path(args.out) if args.out else run_dir / "structured_bootstrap_target_diag.json"
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
    print(f"Wrote bootstrap target summary to {out_path}")
    for update_key, stage_map in summary["updates"].items():
        print(f"[{update_key}]")
        for stage_name in ("accel", "sat", "bw"):
            target_cmp = stage_map[stage_name]["train_target_vs_mc"]
            value_cmp = stage_map[stage_name]["value_vs_mc"]
            print(
                f"  {stage_name}: "
                f"train_vs_mc pearson={target_cmp['pearson']:.3f} "
                f"ev={target_cmp['explained_variance_lhs_as_pred']:.3f} "
                f"gap={target_cmp['gap_mean']:.3f} | "
                f"value_vs_mc pearson={value_cmp['pearson']:.3f} "
                f"ev={value_cmp['explained_variance_lhs_as_pred']:.3f}"
            )


if __name__ == "__main__":
    main()
