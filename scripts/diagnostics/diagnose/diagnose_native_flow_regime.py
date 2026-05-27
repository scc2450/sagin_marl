from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

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
from sagin_marl.rl.structured_critic import ZeroStructuredCritic
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_env_group
from sagin_marl.utils.seeding import set_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast native rollout diagnostic for arrival/outflow/queue regime under fixed exec sources.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default="native_flow")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--traffic_model", default=None)
    parser.add_argument("--reward_mode", default=None)
    parser.add_argument("--exec_accel_source", default="cluster_center_queue_aware")
    parser.add_argument("--exec_sat_source", default="queue_aware")
    parser.add_argument("--exec_bw_source", default="queue_aware")
    parser.add_argument("--torch_threads", type=int, default=1)
    return parser.parse_args()


def _device_from_arg(value: str) -> torch.device:
    requested = str(value).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(value)


def _as_np(tensor: torch.Tensor | None) -> np.ndarray | None:
    if tensor is None:
        return None
    return tensor.detach().cpu().numpy()


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    den = np.asarray(den, dtype=np.float64)
    num = np.asarray(num, dtype=np.float64)
    return np.divide(num, np.maximum(den, 1.0), out=np.zeros_like(num, dtype=np.float64), where=den > 0.0)


def _series_stat(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"mean": 0.0, "p50": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "max": float(np.max(arr)),
    }


def _summarize_current(name: str, cfg: Any, steps: int, seeds: list[int], reward_parts: dict[str, np.ndarray]) -> dict[str, Any]:
    required = [
        "arrival_sum",
        "outflow_sum",
        "backhaul_sum",
        "sat_processed_sum",
        "drop_sum",
        "gu_drop_sum",
        "uav_drop_sum",
        "sat_drop_sum",
        "gu_queue_sum",
        "uav_queue_sum",
        "sat_queue_sum",
        "collision_event",
    ]
    missing = [key for key in required if key not in reward_parts]
    if missing:
        raise RuntimeError(f"native reward_parts missing required fields: {missing}")

    # Shape is [T, E] after stacking.
    arr = {key: np.asarray(value, dtype=np.float64) for key, value in reward_parts.items()}
    arrival = arr["arrival_sum"]
    gu_out = arr["outflow_sum"]
    uav_out = arr["backhaul_sum"]
    sat_proc = arr["sat_processed_sum"]
    drop = arr["drop_sum"]
    gu_q = arr["gu_queue_sum"]
    uav_q = arr["uav_queue_sum"]
    sat_q = arr["sat_queue_sum"]
    collision = arr["collision_event"]
    reward = arr.get("reward", np.zeros_like(arrival))

    per_env: list[dict[str, float]] = []
    runs: list[dict[str, Any]] = []
    for env_idx, seed in enumerate(seeds):
        a_sum = float(np.sum(arrival[:, env_idx]))
        gu_sum = float(np.sum(gu_out[:, env_idx]))
        uav_sum = float(np.sum(uav_out[:, env_idx]))
        proc_sum = float(np.sum(sat_proc[:, env_idx]))
        drop_sum = float(np.sum(drop[:, env_idx]))
        env_summary = {
                "seed": int(seed),
                "steps": int(steps),
                "reward_mean": float(np.mean(reward[:, env_idx])),
                "gu_out_over_arr": float(gu_sum / max(a_sum, 1.0)),
                "uav_out_over_arr": float(uav_sum / max(a_sum, 1.0)),
                "uav_out_over_gu_out": float(uav_sum / max(gu_sum, 1.0)),
                "sat_proc_over_arr": float(proc_sum / max(a_sum, 1.0)),
                "sat_proc_over_uav_out": float(proc_sum / max(uav_sum, 1.0)),
                "drop_over_arr": float(drop_sum / max(a_sum, 1.0)),
                "gu_queue_first": float(gu_q[0, env_idx]),
                "gu_queue_last": float(gu_q[-1, env_idx]),
                "gu_queue_delta_per_step": float((gu_q[-1, env_idx] - gu_q[0, env_idx]) / max(steps - 1, 1)),
                "uav_queue_first": float(uav_q[0, env_idx]),
                "uav_queue_last": float(uav_q[-1, env_idx]),
                "uav_queue_delta_per_step": float((uav_q[-1, env_idx] - uav_q[0, env_idx]) / max(steps - 1, 1)),
                "sat_queue_first": float(sat_q[0, env_idx]),
                "sat_queue_last": float(sat_q[-1, env_idx]),
                "sat_queue_delta_per_step": float((sat_q[-1, env_idx] - sat_q[0, env_idx]) / max(steps - 1, 1)),
                "collision_count": int(np.sum(collision[:, env_idx] > 0.0)),
        }
        per_env.append(env_summary)
        rows: list[dict[str, float]] = []
        terminated_arr = arr.get("terminated", np.zeros_like(arrival))
        truncated_arr = arr.get("truncated", np.zeros_like(arrival))
        for t in range(steps):
            rows.append(
                {
                    "t": int(t),
                    "reward": float(reward[t, env_idx]),
                    "done": bool(terminated_arr[t, env_idx] > 0.0),
                    "trunc": bool(truncated_arr[t, env_idx] > 0.0),
                    "gu_queue_sum": float(gu_q[t, env_idx]),
                    "uav_queue_sum": float(uav_q[t, env_idx]),
                    "sat_queue_sum": float(sat_q[t, env_idx]),
                    "gu_arrival_sum": float(arrival[t, env_idx]),
                    "gu_outflow_sum": float(gu_out[t, env_idx]),
                    "uav_outflow_sum": float(uav_out[t, env_idx]),
                    "sat_processed_sum": float(sat_proc[t, env_idx]),
                    "gu_drop_sum": float(arr["gu_drop_sum"][t, env_idx]),
                    "uav_drop_sum": float(arr["uav_drop_sum"][t, env_idx]),
                    "sat_drop_sum": float(arr["sat_drop_sum"][t, env_idx]),
                    "x_acc": float(gu_out[t, env_idx] / max(arrival[t, env_idx], 1.0)),
                    "x_rel": float(uav_out[t, env_idx] / max(arrival[t, env_idx], 1.0)),
                    "processed_ratio_eval": float(sat_proc[t, env_idx] / max(arrival[t, env_idx], 1.0)),
                    "drop_ratio_eval": float(drop[t, env_idx] / max(arrival[t, env_idx], 1.0)),
                    "collision_event": float(collision[t, env_idx]),
                }
            )
        runs.append(
            {
                "seed": int(seed),
                "steps": int(steps),
                "terminated": bool(np.any(terminated_arr[:, env_idx] > 0.0)),
                "truncated": bool(np.any(truncated_arr[:, env_idx] > 0.0)),
                "rows": rows,
                "summary": env_summary,
            }
        )

    ratios = {
        "gu_out_over_arr": _safe_ratio(np.sum(gu_out, axis=0), np.sum(arrival, axis=0)),
        "uav_out_over_arr": _safe_ratio(np.sum(uav_out, axis=0), np.sum(arrival, axis=0)),
        "uav_out_over_gu_out": _safe_ratio(np.sum(uav_out, axis=0), np.sum(gu_out, axis=0)),
        "sat_proc_over_arr": _safe_ratio(np.sum(sat_proc, axis=0), np.sum(arrival, axis=0)),
        "sat_proc_over_uav_out": _safe_ratio(np.sum(sat_proc, axis=0), np.sum(uav_out, axis=0)),
        "drop_over_arr": _safe_ratio(np.sum(drop, axis=0), np.sum(arrival, axis=0)),
    }
    summary: dict[str, Any] = {
        "name": name,
        "steps": int(steps),
        "num_envs": int(len(seeds)),
        "reward_mean": float(np.mean(reward)),
        "arrival_sum_mean": float(np.mean(arrival)),
        "gu_outflow_sum_mean": float(np.mean(gu_out)),
        "uav_outflow_sum_mean": float(np.mean(uav_out)),
        "sat_processed_sum_mean": float(np.mean(sat_proc)),
        "drop_sum_mean": float(np.mean(drop)),
        "gu_drop_sum_mean": float(np.mean(arr["gu_drop_sum"])),
        "uav_drop_sum_mean": float(np.mean(arr["uav_drop_sum"])),
        "sat_drop_sum_mean": float(np.mean(arr["sat_drop_sum"])),
        "gu_queue_sum_mean": float(np.mean(gu_q)),
        "uav_queue_sum_mean": float(np.mean(uav_q)),
        "sat_queue_sum_mean": float(np.mean(sat_q)),
        "gu_queue_first_mean": float(np.mean(gu_q[0, :])),
        "gu_queue_last_mean": float(np.mean(gu_q[-1, :])),
        "gu_queue_delta_per_step_mean": float(np.mean((gu_q[-1, :] - gu_q[0, :]) / max(steps - 1, 1))),
        "uav_queue_first_mean": float(np.mean(uav_q[0, :])),
        "uav_queue_last_mean": float(np.mean(uav_q[-1, :])),
        "uav_queue_delta_per_step_mean": float(np.mean((uav_q[-1, :] - uav_q[0, :]) / max(steps - 1, 1))),
        "sat_queue_first_mean": float(np.mean(sat_q[0, :])),
        "sat_queue_last_mean": float(np.mean(sat_q[-1, :])),
        "sat_queue_delta_per_step_mean": float(np.mean((sat_q[-1, :] - sat_q[0, :]) / max(steps - 1, 1))),
        "collision_count_total": int(np.sum(collision > 0.0)),
        "terminated_count": int(np.sum(arr.get("terminated", np.zeros_like(arrival)) > 0.0)),
        "truncated_count": int(np.sum(arr.get("truncated", np.zeros_like(arrival)) > 0.0)),
        "cfg": {
            "reward_mode": str(getattr(cfg, "reward_mode", "")),
            "traffic_model": str(getattr(cfg, "traffic_model", "")),
            "task_arrival_rate": float(getattr(cfg, "task_arrival_rate", 0.0) or 0.0),
            "task_arrival_poisson": bool(getattr(cfg, "task_arrival_poisson", False)),
            "b_acc": float(getattr(cfg, "b_acc", 0.0) or 0.0),
            "b_backhaul_per_sat": float(
                getattr(cfg, "b_backhaul_per_sat", getattr(cfg, "b_sat_total", 0.0)) or 0.0
            ),
            "sat_cpu_freq": float(getattr(cfg, "sat_cpu_freq", 0.0) or 0.0),
            "uav_init_speed_frac": float(getattr(cfg, "uav_init_speed_frac", 0.0) or 0.0),
            "avoidance_enabled": bool(getattr(cfg, "avoidance_enabled", False)),
            "danger_imitation_enabled": bool(getattr(cfg, "danger_imitation_enabled", False)),
            "queue_init_gu_steps": float(getattr(cfg, "queue_init_gu_steps", 0.0) or 0.0),
            "queue_init_uav_steps": float(getattr(cfg, "queue_init_uav_steps", 0.0) or 0.0),
            "queue_init_sat_steps": float(getattr(cfg, "queue_init_sat_steps", 0.0) or 0.0),
        },
    }
    for key, values in ratios.items():
        stat = _series_stat(values)
        summary[f"{key}_mean"] = stat["mean"]
        summary[f"{key}_p50"] = stat["p50"]
        summary[f"{key}_max"] = stat["max"]

    return {"summary": summary, "per_env": per_env, "runs": runs}


def main() -> None:
    args = _parse_args()
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    cfg = load_config(args.config)
    cfg = copy.copy(cfg)
    if args.traffic_model is not None:
        cfg.traffic_model = str(args.traffic_model)
    if args.reward_mode is not None:
        cfg.reward_mode = str(args.reward_mode)
    cfg.exec_accel_source = str(args.exec_accel_source)
    cfg.exec_sat_source = str(args.exec_sat_source)
    cfg.exec_bw_source = str(args.exec_bw_source)
    cfg.train_accel = False
    cfg.train_sat = False
    cfg.train_bw = False
    cfg.checkpoint_eval_enabled = False
    cfg.train_trace_enabled = False
    cfg.structured_env_tensor_backend = "cuda"
    cfg.structured_env_backend = "native"

    device = _device_from_arg(args.device)
    set_seed(int(getattr(cfg, "seed", 0) or 0))
    steps = int(args.horizon) if args.horizon is not None else int(getattr(cfg, "T_steps", 0) or 0)
    if steps <= 0:
        raise ValueError("horizon must be positive.")
    num_envs = max(int(args.num_envs), 1)
    seeds = [int(args.seed_base) + i for i in range(num_envs)]

    t0 = time.perf_counter()
    bundle = build_structured_modules_from_config(cfg, build_critic=False)
    actor = bundle.actor.to(device).eval()
    learner = StructuredMAPPO(
        actor=actor,
        critic=ZeroStructuredCritic(device),
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_ratio=cfg.clip_ratio,
        value_coef=cfg.value_coef,
        entropy_coef=cfg.entropy_coef,
        max_grad_norm=cfg.max_grad_norm,
        ppo_epochs=1,
        num_mini_batch=1,
        actor_optimizer=None,
        critic_optimizer=None,
        device=device,
        cfg=cfg,
        train_accel=False,
        train_sat=False,
        train_bw=False,
        exec_accel_source=cfg.exec_accel_source,
        exec_sat_source=cfg.exec_sat_source,
        exec_bw_source=cfg.exec_bw_source,
    )
    env_group = make_structured_env_group(cfg, num_envs=num_envs, backend="sync", mode="script")
    prepare_sec = time.perf_counter() - t0
    try:
        env_group.reset_many(seeds=seeds)
        learner.begin_native_rollout(
            env_group,
            rollout_env_steps=steps,
            num_envs=num_envs,
        )
        buffer = StructuredRolloutBuffer()
        t1 = time.perf_counter()
        results = learner.collect_env_horizon_native_tensor_policy(
            env_group,
            buffer=buffer,
            horizon=steps,
            deterministic=True,
        )
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        collect_sec = time.perf_counter() - t1
        if len(results) != steps:
            raise RuntimeError(f"expected {steps} native step results, got {len(results)}")
        reward_parts: dict[str, list[np.ndarray]] = {}
        rewards: list[np.ndarray] = []
        terminated: list[np.ndarray] = []
        truncated: list[np.ndarray] = []
        for result in results:
            rewards.append(_as_np(result.team_rewards))
            terminated.append(_as_np(result.terminated).astype(np.float32))
            truncated.append(_as_np(result.truncated).astype(np.float32))
            parts = result.reward_part_tensors or {}
            for key, tensor in parts.items():
                reward_parts.setdefault(str(key), []).append(_as_np(tensor))
        stacked = {key: np.stack(values, axis=0) for key, values in reward_parts.items()}
        stacked["reward"] = np.stack(rewards, axis=0)
        stacked["terminated"] = np.stack(terminated, axis=0)
        stacked["truncated"] = np.stack(truncated, axis=0)
        payload = _summarize_current(str(args.name), cfg, steps, seeds, stacked)
        payload["summary"]["prepare_sec"] = float(prepare_sec)
        payload["summary"]["collect_sec"] = float(collect_sec)
        payload["summary"]["steps_per_sec"] = float((steps * num_envs) / max(collect_sec, 1e-9))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    finally:
        close_structured_env_group(env_group)


if __name__ == "__main__":
    main()
