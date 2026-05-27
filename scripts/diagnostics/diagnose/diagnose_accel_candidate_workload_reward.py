from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_critic import ZeroStructuredCritic
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_env_group
from sagin_marl.utils.seeding import set_seed


def _as_np(tensor: torch.Tensor | None) -> np.ndarray:
    if tensor is None:
        return np.zeros((0,), dtype=np.float64)
    return tensor.detach().cpu().numpy().astype(np.float64, copy=False)


def _summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10.0)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    xa = np.asarray(x, dtype=np.float64).reshape(-1)
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if int(mask.sum()) <= 2:
        return 0.0
    xa = xa[mask]
    ya = ya[mask]
    sx = float(np.std(xa))
    sy = float(np.std(ya))
    if sx <= 1.0e-12 or sy <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def _source_spec(name: str) -> tuple[str, bool]:
    key = str(name).strip().lower()
    if key == "policy_random":
        return "policy", False
    if key == "policy_mean":
        return "policy", True
    return key, True


def _collect_source(
    cfg: Any,
    *,
    source_name: str,
    device: torch.device,
    num_envs: int,
    horizon: int,
    seed_base: int,
) -> dict[str, Any]:
    source, deterministic = _source_spec(source_name)
    run_cfg = copy.copy(cfg)
    run_cfg.train_accel = False
    run_cfg.train_sat = False
    run_cfg.train_bw = False
    run_cfg.exec_accel_source = source
    run_cfg.exec_sat_source = "queue_aware"
    run_cfg.exec_bw_source = "queue_aware"
    run_cfg.checkpoint_eval_enabled = False
    run_cfg.train_trace_enabled = False
    run_cfg.structured_env_tensor_backend = "cuda" if device.type == "cuda" else "cpu"
    run_cfg.structured_env_backend = "native"

    bundle = build_structured_modules_from_config(run_cfg, build_critic=False)
    actor = bundle.actor.to(device).eval()
    learner = StructuredMAPPO(
        actor=actor,
        critic=ZeroStructuredCritic(device),
        gamma=float(run_cfg.gamma),
        gae_lambda=float(run_cfg.gae_lambda),
        clip_ratio=float(run_cfg.clip_ratio),
        value_coef=float(run_cfg.value_coef),
        entropy_coef=float(run_cfg.entropy_coef),
        max_grad_norm=float(run_cfg.max_grad_norm),
        ppo_epochs=1,
        num_mini_batch=1,
        actor_optimizer=None,
        critic_optimizer=None,
        device=device,
        cfg=run_cfg,
        train_accel=False,
        train_sat=False,
        train_bw=False,
        exec_accel_source=run_cfg.exec_accel_source,
        exec_sat_source=run_cfg.exec_sat_source,
        exec_bw_source=run_cfg.exec_bw_source,
    )
    group = make_structured_env_group(run_cfg, num_envs=int(num_envs), backend="sync", mode="script")
    try:
        seeds = [int(seed_base) + i for i in range(int(num_envs))]
        group.reset_many(seeds=seeds)
        learner.begin_native_rollout(group, rollout_env_steps=int(horizon), num_envs=int(num_envs))
        buffer = StructuredRolloutBuffer()
        t0 = time.perf_counter()
        results = learner.collect_env_horizon_native_tensor_policy(
            group,
            buffer=buffer,
            horizon=int(horizon),
            deterministic=bool(deterministic),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        collect_sec = time.perf_counter() - t0

        parts: dict[str, list[np.ndarray]] = {}
        rewards: list[np.ndarray] = []
        terminated: list[np.ndarray] = []
        truncated: list[np.ndarray] = []
        for result in results:
            rewards.append(_as_np(result.team_rewards))
            terminated.append(_as_np(result.terminated))
            truncated.append(_as_np(result.truncated))
            for key, tensor in (result.reward_part_tensors or {}).items():
                parts.setdefault(str(key), []).append(_as_np(tensor))
        stacked = {key: np.stack(vals, axis=0) for key, vals in parts.items() if vals}
        reward = np.stack(rewards, axis=0)
        done = (np.stack(terminated, axis=0) > 0.5) | (np.stack(truncated, axis=0) > 0.5)
        if "bw_weighted_workload_delta_reward" not in stacked or "bw_weighted_workload_level_reward" not in stacked:
            raise RuntimeError("native rollout did not expose workload delta/level reward tensors.")
        delta = np.asarray(stacked["bw_weighted_workload_delta_reward"], dtype=np.float64)
        level = np.asarray(stacked["bw_weighted_workload_level_reward"], dtype=np.float64)
        j_post_drop = np.maximum(-level, 0.0)
        j_pre = np.maximum(delta + j_post_drop, 0.0)
        # Candidate signal discussed in chat: relative workload improvement.
        # Numerator is existing weighted_workload_delta = J_pre - J_post_drop.
        candidate = np.divide(delta, np.maximum(j_pre, 1.0), out=np.zeros_like(delta), where=j_pre > 0.0)
        positive_level = 1.0 / (1.0 + np.log1p(j_post_drop))

        outflow = np.asarray(stacked.get("outflow_sum", np.zeros_like(delta)), dtype=np.float64)
        arrival = np.asarray(stacked.get("arrival_sum", np.zeros_like(delta)), dtype=np.float64)
        drop = np.asarray(stacked.get("drop_sum", np.zeros_like(delta)), dtype=np.float64)
        access_ratio = np.divide(outflow, np.maximum(arrival, 1.0), out=np.zeros_like(outflow), where=arrival > 0.0)
        drop_ratio = np.divide(drop, np.maximum(arrival, 1.0), out=np.zeros_like(drop), where=arrival > 0.0)

        return {
            "source": str(source_name),
            "exec_accel_source": str(source),
            "deterministic": bool(deterministic),
            "seeds": seeds,
            "collect_sec": float(collect_sec),
            "steps_per_sec": float((int(horizon) * int(num_envs)) / max(collect_sec, 1.0e-9)),
            "reward": _summary(reward),
            "candidate_relative_delta": _summary(candidate),
            "weighted_delta": _summary(delta),
            "weighted_level": _summary(level),
            "positive_level_recomputed": _summary(positive_level),
            "j_pre": _summary(j_pre),
            "j_post_drop": _summary(j_post_drop),
            "access_ratio": _summary(access_ratio),
            "drop_ratio": _summary(drop_ratio),
            "done_rate": float(np.mean(done.astype(np.float64))),
            "candidate_corr_access_ratio": _corr(candidate, access_ratio),
            "candidate_corr_drop_ratio": _corr(candidate, drop_ratio),
        }
    finally:
        close_structured_env_group(group)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--seed_base", type=int, default=42000)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["cluster_center_queue_aware", "queue_aware", "zero", "policy_random"],
    )
    parser.add_argument("--torch_threads", type=int, default=1)
    args = parser.parse_args()

    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    cfg = load_config(args.config)
    set_seed(int(getattr(cfg, "seed", 0) or 0))

    rows = [
        _collect_source(
            cfg,
            source_name=str(source),
            device=device,
            num_envs=int(args.num_envs),
            horizon=int(args.horizon),
            seed_base=int(args.seed_base),
        )
        for source in args.sources
    ]
    ranking = sorted(
        (
            {
                "source": row["source"],
                "candidate_mean": row["candidate_relative_delta"]["mean"],
                "candidate_std": row["candidate_relative_delta"]["std"],
                "weighted_delta_mean": row["weighted_delta"]["mean"],
                "reward_mean": row["reward"]["mean"],
                "access_ratio_mean": row["access_ratio"]["mean"],
                "drop_ratio_mean": row["drop_ratio"]["mean"],
                "done_rate": row["done_rate"],
            }
            for row in rows
        ),
        key=lambda item: float(item["candidate_mean"]),
        reverse=True,
    )
    payload = {
        "config": str(args.config),
        "horizon": int(args.horizon),
        "num_envs": int(args.num_envs),
        "seed_base": int(args.seed_base),
        "candidate_formula": "bw_weighted_workload_delta_reward / max(J_pre, 1), J_pre = delta - level",
        "ranking_by_candidate_mean": ranking,
        "sources": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ranking_by_candidate_mean": ranking, "out": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
