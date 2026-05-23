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


def _quantile_bins(pred: np.ndarray, target: np.ndarray, bins: int = 5) -> List[Dict[str, float]]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n = int(pred.size)
    if n == 0:
        return []
    order = np.argsort(pred, kind="mergesort")
    rows: List[Dict[str, float]] = []
    for bin_idx in range(int(bins)):
        start = int(math.floor(bin_idx * n / bins))
        end = int(math.floor((bin_idx + 1) * n / bins))
        if end <= start:
            continue
        sl = order[start:end]
        rows.append(
            {
                "bin": float(bin_idx),
                "count": float(end - start),
                "pred_mean": float(np.mean(pred[sl])),
                "target_mean": float(np.mean(target[sl])),
                "pred_min": float(np.min(pred[sl])),
                "pred_max": float(np.max(pred[sl])),
            }
        )
    return rows


def _load_bundle(cfg, run_dir: Path, update: int, device: torch.device):
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    actor_ckpt = run_dir / f"actor_u{update:04d}.pt"
    critic_ckpt = run_dir / f"critic_u{update:04d}.pt"
    load_checkpoint_forgiving(bundle.actor, str(actor_ckpt), map_location=device, strict=True)
    load_checkpoint_forgiving(bundle.critic, str(critic_ckpt), map_location=device, strict=True)
    bundle.actor.to(device).eval()
    bundle.critic.to(device).eval()
    return bundle


def _collect_episode_transitions(
    cfg,
    actor,
    critic,
    *,
    device: torch.device,
    deterministic: bool,
    seed: int,
) -> Dict[str, np.ndarray]:
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
        mc = buffer.compute_gae(gamma_env=float(cfg.gamma), gae_lambda=1.0, bootstrap_value=0.0)
        return_view = buffer.build_return_view()
        stage_ids = np.asarray(return_view.stage_ids, dtype=np.int64)
        values = np.asarray(return_view.values, dtype=np.float64)
        rewards = np.asarray(return_view.rewards, dtype=np.float64)
        returns = np.asarray(mc["returns"], dtype=np.float64)
        return {
            "stage_ids": stage_ids,
            "values": values,
            "returns": returns,
            "rewards": rewards,
        }
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def _summarize_stage(pred: np.ndarray, target: np.ndarray) -> Dict[str, object]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    error = pred - target
    return {
        "n": int(pred.size),
        "pred_mean": float(np.mean(pred)) if pred.size else 0.0,
        "target_mean": float(np.mean(target)) if target.size else 0.0,
        "bias_mean": float(np.mean(error)) if error.size else 0.0,
        "mae": float(np.mean(np.abs(error))) if error.size else 0.0,
        "rmse": float(np.sqrt(np.mean(error**2))) if error.size else 0.0,
        "pearson": _safe_corr(pred, target),
        "spearman": _safe_spearman(pred, target),
        "explained_variance": _explained_variance(pred, target),
        "pred_qbins": _quantile_bins(pred, target, bins=5),
    }


def diagnose_run(
    run_dir: Path,
    *,
    updates: Iterable[int],
    episodes: int,
    episode_seed_base: int,
    deterministic: bool,
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
        "updates": {},
    }
    for update in updates:
        bundle = _load_bundle(cfg, run_dir, int(update), device)
        values_by_stage: Dict[int, List[float]] = defaultdict(list)
        returns_by_stage: Dict[int, List[float]] = defaultdict(list)
        rewards_by_stage: Dict[int, List[float]] = defaultdict(list)
        for ep_idx in range(int(episodes)):
            seed = int(episode_seed_base) + ep_idx
            episode = _collect_episode_transitions(
                cfg,
                bundle.actor,
                bundle.critic,
                device=device,
                deterministic=deterministic,
                seed=seed,
            )
            for stage_id in (0, 1, 2):
                mask = episode["stage_ids"] == stage_id
                values_by_stage[stage_id].extend(episode["values"][mask].tolist())
                returns_by_stage[stage_id].extend(episode["returns"][mask].tolist())
                rewards_by_stage[stage_id].extend(episode["rewards"][mask].tolist())
        update_summary: Dict[str, object] = {}
        for stage_id, stage_name in STAGE_NAMES.items():
            pred = np.asarray(values_by_stage[stage_id], dtype=np.float64)
            target = np.asarray(returns_by_stage[stage_id], dtype=np.float64)
            reward = np.asarray(rewards_by_stage[stage_id], dtype=np.float64)
            stage_summary = _summarize_stage(pred, target)
            stage_summary["reward_mean"] = float(np.mean(reward)) if reward.size else 0.0
            update_summary[stage_name] = stage_summary
        results["updates"][f"u{int(update):04d}"] = update_summary
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--updates", type=int, nargs="+", default=[100, 150, 200])
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--episode_seed_base", type=int, default=42000)
    parser.add_argument("--policy_mode", choices=["deterministic", "stochastic"], default="stochastic")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_path = Path(args.out) if args.out else run_dir / "structured_critic_alignment_diag.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = diagnose_run(
        run_dir,
        updates=args.updates,
        episodes=int(args.episodes),
        episode_seed_base=int(args.episode_seed_base),
        deterministic=(args.policy_mode == "deterministic"),
        device=device,
    )
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote critic alignment summary to {out_path}")
    for update_key, stage_map in summary["updates"].items():
        print(f"[{update_key}]")
        for stage_name in ("accel", "sat", "bw"):
            stage_summary = stage_map[stage_name]
            print(
                f"  {stage_name}: "
                f"pearson={stage_summary['pearson']:.3f} "
                f"spearman={stage_summary['spearman']:.3f} "
                f"ev={stage_summary['explained_variance']:.3f} "
                f"rmse={stage_summary['rmse']:.3f} "
                f"bias={stage_summary['bias_mean']:.3f}"
            )


if __name__ == "__main__":
    main()
