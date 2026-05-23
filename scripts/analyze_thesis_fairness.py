from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from scripts.evaluate_thesis_native_methods import _actor_state_from_stage_checkpoints
from sagin_marl.env.config import load_config
from sagin_marl.rl import structured_critic_schema as critic_schema
from sagin_marl.rl.structured_eval import (
    _as_driver_list,
    _fixed_policy_exec_sources,
    ZeroStructuredCritic,
)
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO
from sagin_marl.rl.structured_parallel_eval import looks_like_driver_group, reset_many
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_env_group
from sagin_marl.utils.checkpoint import load_state_dict_forgiving


METHOD_NAMES = {
    "proposed": "本文方法",
    "cluster_center_queue_aware": "簇中心运动规则",
    "lyapunov": "李雅普诺夫队列方法",
}


def _jain(x: np.ndarray, eps: float = 1.0e-12) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    denom = float(arr.size) * float(np.sum(arr * arr))
    if denom <= eps:
        return float("nan")
    return float(np.sum(arr) ** 2 / denom)


def _cv(x: np.ndarray, eps: float = 1.0e-12) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    mean = float(np.mean(arr))
    if abs(mean) <= eps:
        return 0.0
    return float(np.std(arr) / abs(mean))


def _active_mask(done: np.ndarray) -> np.ndarray:
    done = np.asarray(done, dtype=bool)
    active = np.ones_like(done, dtype=bool)
    finished = np.zeros((done.shape[1],), dtype=bool)
    for t in range(done.shape[0]):
        active[t] = ~finished
        finished |= done[t]
    return active


def _reshape_history_tensor(tensor: torch.Tensor, *, steps: int, slots: int) -> np.ndarray:
    arr = tensor[: steps * slots].detach().cpu().numpy()
    return arr.reshape((steps, slots) + tuple(arr.shape[1:]))


def _mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _pct(values: list[float], q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _episode_metrics_from_history(history: Any, *, steps: int, slots: int) -> list[dict[str, float]]:
    done = _reshape_history_tensor(history.terminated | history.truncated, steps=steps, slots=slots)
    active = _active_mask(done)

    world = history.bw_stage.world_batch
    gu_nodes = _reshape_history_tensor(world.gu_nodes, steps=steps, slots=slots)
    uav_nodes = _reshape_history_tensor(world.uav_nodes, steps=steps, slots=slots)
    gu_arrival = np.expm1(gu_nodes[..., critic_schema.GU_LAST_ARRIVAL_STEPS])
    gu_outflow = np.expm1(gu_nodes[..., critic_schema.GU_LAST_OUTFLOW_STEPS])
    gu_drop = np.expm1(gu_nodes[..., critic_schema.GU_LAST_DROP_STEPS])
    gu_queue = np.expm1(gu_nodes[..., critic_schema.GU_QUEUE_STEPS])

    uav_inflow = np.expm1(uav_nodes[..., critic_schema.UAV_LAST_INFLOW_STEPS])
    uav_outflow = np.expm1(uav_nodes[..., critic_schema.UAV_LAST_OUTFLOW_STEPS])
    uav_queue = np.expm1(uav_nodes[..., critic_schema.UAV_QUEUE_STEPS])

    rows: list[dict[str, float]] = []
    eps = 1.0e-9
    for slot in range(slots):
        m = active[:, slot]
        if not np.any(m):
            continue

        arr = gu_arrival[m, slot, :].sum(axis=0)
        out = gu_outflow[m, slot, :].sum(axis=0)
        drop = gu_drop[m, slot, :].sum(axis=0)
        q_gu = gu_queue[m, slot, :].mean(axis=0)
        gu_has_arrival = arr > eps
        service_ratio = out[gu_has_arrival] / np.maximum(arr[gu_has_arrival], eps)
        drop_ratio = drop[gu_has_arrival] / np.maximum(arr[gu_has_arrival], eps)

        uav_in = uav_inflow[m, slot, :].sum(axis=0)
        uav_out = uav_outflow[m, slot, :].sum(axis=0)
        q_uav = uav_queue[m, slot, :].mean(axis=0)

        rows.append(
            {
                "active_steps": float(np.sum(m)),
                "gu_service_jain": _jain(service_ratio),
                "gu_service_p10": float(np.percentile(service_ratio, 10)) if service_ratio.size else float("nan"),
                "gu_service_min": float(np.min(service_ratio)) if service_ratio.size else float("nan"),
                "gu_service_cv": _cv(service_ratio),
                "gu_drop_jain": _jain(1.0 - np.clip(drop_ratio, 0.0, 1.0)),
                "gu_drop_p90": float(np.percentile(drop_ratio, 90)) if drop_ratio.size else float("nan"),
                "gu_mean_queue_cv": _cv(q_gu),
                "gu_mean_queue_p90_steps": float(np.percentile(q_gu, 90)),
                "gu_mean_queue_max_steps": float(np.max(q_gu)),
                "gu_starved_ratio_lt_0p8": float(np.mean(service_ratio < 0.8)) if service_ratio.size else float("nan"),
                "uav_inflow_jain": _jain(uav_in),
                "uav_outflow_jain": _jain(uav_out),
                "uav_mean_queue_cv": _cv(q_uav),
                "uav_mean_queue_avg_steps": float(np.mean(q_uav)),
                "uav_mean_queue_max_steps": float(np.max(q_uav)),
                "uav_queue_imbalance_steps": float(np.max(q_uav) - np.min(q_uav)),
            }
        )
    return rows


def _make_actor(cfg: Any, device: torch.device, args: argparse.Namespace, method: str) -> torch.nn.Module:
    if method != "proposed":
        return torch.nn.Linear(1, 1).to(device)
    bundle = build_structured_modules_from_config(cfg)
    actor_state = _actor_state_from_stage_checkpoints(
        accel_checkpoint=Path(args.accel_checkpoint),
        sat_checkpoint=Path(args.sat_checkpoint),
        bw_checkpoint=Path(args.bw_checkpoint),
    )
    info = load_state_dict_forgiving(bundle.actor, actor_state, strict=True)
    if info.get("missing_keys") or info.get("unexpected_keys"):
        raise RuntimeError(f"checkpoint load mismatch: {info}")
    return bundle.actor.to(device).eval()


def _collect_method(cfg: Any, args: argparse.Namespace, *, method: str, device: torch.device) -> list[dict[str, float]]:
    slots = max(min(int(args.num_envs), int(args.episodes)), 1)
    actor = _make_actor(cfg, device, args, method)
    if method == "proposed":
        exec_sources = ("policy", "policy", "policy")
        train_flags = (True, True, True)
    else:
        exec_sources = _fixed_policy_exec_sources(method)
        if exec_sources is None:
            raise ValueError(f"unsupported method {method!r}")
        train_flags = tuple(source == "policy" for source in exec_sources)
    learner = StructuredMAPPO(
        actor=actor,
        critic=ZeroStructuredCritic(device),
        gamma=float(getattr(cfg, "gamma", 0.99) or 0.99),
        gae_lambda=float(getattr(cfg, "gae_lambda", 0.95) or 0.95),
        clip_ratio=float(getattr(cfg, "clip_ratio", 0.2) or 0.2),
        value_coef=0.0,
        entropy_coef=0.0,
        max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5) or 0.5),
        ppo_epochs=1,
        num_mini_batch=1,
        actor_optimizer=None,
        critic_optimizer=None,
        device=device,
        target_mode="step_level",
        cfg=cfg,
        train_accel=bool(train_flags[0]),
        train_sat=bool(train_flags[1]),
        train_bw=bool(train_flags[2]),
        exec_accel_source=exec_sources[0],
        exec_sat_source=exec_sources[1],
        exec_bw_source=exec_sources[2],
    )
    env_group = make_structured_env_group(cfg, num_envs=slots, backend="sync", mode="eval")
    drivers = env_group if looks_like_driver_group(env_group) else _as_driver_list(env_group)
    rows: list[dict[str, float]] = []
    horizon = int(getattr(cfg, "T_steps", 250) or 250)
    try:
        cursor = 0
        while cursor < int(args.episodes):
            batch_episodes = min(slots, int(args.episodes) - cursor)
            seeds = [int(args.episode_seed_base) + cursor + slot for slot in range(slots)]
            reset_many(drivers, seeds)
            buffer = StructuredRolloutBuffer()
            learner.begin_native_rollout(drivers, rollout_env_steps=horizon, num_envs=slots)
            learner.collect_env_horizon_native_tensor_policy(
                drivers,
                buffer,
                horizon=horizon,
                deterministic=True,
            )
            runtime = getattr(drivers, "native_rollout_runtime", None)
            if runtime is None or getattr(runtime, "history", None) is None:
                raise RuntimeError("native runtime history is unavailable")
            batch_rows = _episode_metrics_from_history(runtime.history, steps=horizon, slots=slots)
            for local_idx, row in enumerate(batch_rows[:batch_episodes]):
                rows.append(
                    {
                        "episode": float(cursor + local_idx),
                        **row,
                    }
                )
            cursor += batch_episodes
    finally:
        close_structured_env_group(env_group)
        del learner
        del actor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def _summarize(method: str, rows: list[dict[str, float]]) -> dict[str, Any]:
    keys = [key for key in rows[0].keys() if key != "episode"] if rows else []
    summary: dict[str, Any] = {
        "method_id": method,
        "method_name": METHOD_NAMES.get(method, method),
        "episodes": len(rows),
    }
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row]
        summary[key] = _mean(vals)
        summary[f"{key}_p10"] = _pct(vals, 10)
        summary[f"{key}_p90"] = _pct(vals, 90)
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--accel_checkpoint", required=True)
    parser.add_argument("--sat_checkpoint", required=True)
    parser.add_argument("--bw_checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--methods", nargs="+", default=["proposed", "cluster_center_queue_aware", "lyapunov"])
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--episode_seed_base", type=int, default=903000)
    parser.add_argument("--access_bw_decision_interval", type=int, default=5)
    parser.add_argument("--sat_decision_interval", type=int, default=1)
    parser.add_argument("--lyapunov_params_json", default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.access_bw_decision_interval = max(int(args.access_bw_decision_interval), 1)
    cfg.sat_decision_interval = max(int(args.sat_decision_interval), 1)
    cfg.structured_native_history_snapshots_enabled = True
    if args.lyapunov_params_json:
        import json

        for key, value in json.loads(args.lyapunov_params_json).items():
            setattr(cfg, str(key), float(value))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for method in args.methods:
        print(f"[fairness] {method}")
        rows = _collect_method(cfg, args, method=method, device=device)
        for row in rows:
            all_rows.append({"method_id": method, "method_name": METHOD_NAMES.get(method, method), **row})
        summary_rows.append(_summarize(method, rows))

    _write_csv(out_dir / "fairness_episodes.csv", all_rows)
    _write_csv(out_dir / "fairness_summary.csv", summary_rows)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
