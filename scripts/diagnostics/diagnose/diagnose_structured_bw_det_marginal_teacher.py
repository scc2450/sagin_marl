from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
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
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import _to_device_dataclass
from sagin_marl.rl.structured_stage_builders import build_local_bw_states_from_snapshot
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _done_from_step_result(step_result) -> bool:
    return bool(
        list(step_result.terminations.values())[0]
        or list(step_result.truncations.values())[0]
    )


def _zero_sat_action(cfg) -> np.ndarray:
    select_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 1)
    return np.full((cfg.num_uav, select_k), -1, dtype=np.int64)


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _safe_percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), float(q)))


def _safe_corr(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
        return 0.0
    if float(np.std(x_arr)) <= 1.0e-12 or float(np.std(y_arr)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _normalize_bw_action(action: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    valid_mask_arr = np.asarray(valid_mask, dtype=bool)
    action_arr = np.asarray(action, dtype=np.float32)
    out = np.zeros(valid_mask_arr.shape, dtype=np.float32)
    for u in range(int(valid_mask_arr.shape[0])):
        valid = np.flatnonzero(valid_mask_arr[u])
        if valid.size <= 0:
            continue
        row = np.clip(action_arr[u, valid], 0.0, None)
        row_sum = float(np.sum(row))
        if row_sum <= 1.0e-12:
            out[u, valid] = 1.0 / float(valid.size)
        else:
            out[u, valid] = row / row_sum
    return out


def _det_bw_action_from_snapshot(actor, snapshot, device: torch.device) -> np.ndarray:
    bw_states = build_local_bw_states_from_snapshot(snapshot)
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for local_state in bw_states:
            local_state_device = _to_device_dataclass(local_state, device)
            out = actor.act_bw(local_state_device, deterministic=True)
            row = np.asarray(out.action.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
            rows.append(row)
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def _reward_feature_snapshot(env: SaginParallelEnv) -> dict[str, dict[str, np.ndarray]]:
    return {
        "normalized": env._gu_reward_aligned_feature_dict(normalized=True),
        "raw": env._gu_reward_aligned_feature_dict(normalized=False),
    }


def _feature_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64) if values else np.asarray([], dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)) if arr.size > 0 else 0.0,
        "std": float(np.std(arr)) if arr.size > 0 else 0.0,
        "min": float(np.min(arr)) if arr.size > 0 else 0.0,
        "p05": _safe_percentile(values, 5.0),
        "p50": _safe_percentile(values, 50.0),
        "p95": _safe_percentile(values, 95.0),
        "max": float(np.max(arr)) if arr.size > 0 else 0.0,
    }


def _collect_actor_snapshots(
    *,
    cfg,
    actor,
    device: torch.device,
    num_samples: int,
    seed_base: int,
) -> tuple[list[dict[str, Any]], int]:
    env = make_structured_env(cfg, mode="script")
    driver = as_structured_driver(env)
    rows: list[dict[str, Any]] = []
    episode_count = 0
    try:
        while len(rows) < int(num_samples):
            env.reset(seed=int(seed_base) + int(episode_count))
            done = False
            while not done and len(rows) < int(num_samples):
                driver.begin_step()
                accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
                _ = driver.run_accel_stage(accel_zero)
                bw_world = driver.run_sat_stage(_zero_sat_action(cfg))
                snapshot = driver.build_bw_stage_snapshot(bw_world)
                snapshot_state = driver.export_bw_stage_state()
                det_action = _normalize_bw_action(
                    _det_bw_action_from_snapshot(actor, snapshot, device),
                    np.asarray(snapshot.bw_valid_mask, dtype=bool),
                )
                arrival_rates = np.asarray(
                    getattr(env, "last_gu_arrival_rate_vec", np.zeros((cfg.num_gu,), dtype=np.float32)),
                    dtype=np.float32,
                )
                feature_snapshot = _reward_feature_snapshot(env)
                rows.append(
                    {
                        "sample_index": int(len(rows)),
                        "episode": int(episode_count),
                        "t": int(env.t),
                        "snapshot_state": snapshot_state,
                        "det_action": det_action.tolist(),
                        "valid_mask": np.asarray(snapshot.bw_valid_mask, dtype=np.float32).tolist(),
                        "candidate_indices": np.asarray(snapshot.candidate_indices, dtype=np.int64).tolist(),
                        "arrival_rates": arrival_rates.tolist(),
                        "queue_raw": np.asarray(env.gu_queue, dtype=np.float32).tolist(),
                        "reward_features_raw": {
                            key: np.asarray(value, dtype=np.float32).tolist()
                            for key, value in feature_snapshot["raw"].items()
                        },
                        "reward_features_norm": {
                            key: np.asarray(value, dtype=np.float32).tolist()
                            for key, value in feature_snapshot["normalized"].items()
                        },
                    }
                )
                step_result = driver.execute_stage_bw_and_step(det_action)
                done = _done_from_step_result(step_result)
            episode_count += 1
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
    return rows, int(episode_count)


def _single_step_reward_from_snapshot(
    *,
    probe_driver: StructuredControlDriver,
    snapshot_state: dict[str, Any],
    bw_action: np.ndarray,
) -> float:
    probe_driver.load_bw_stage_state(snapshot_state)
    step_result = probe_driver.execute_stage_bw_and_step(np.asarray(bw_action, dtype=np.float32))
    return float(next(iter(step_result.rewards.values())))


def _perturb_transfer(
    base_action: np.ndarray,
    *,
    src_slot: int,
    dst_slot: int,
    transfer_mass: float,
    eps: float = 1.0e-8,
) -> tuple[np.ndarray | None, float]:
    row = np.asarray(base_action, dtype=np.float32).copy()
    if int(src_slot) == int(dst_slot):
        return None, 0.0
    movable = max(float(row[int(src_slot)]) - eps, 0.0)
    delta = min(max(float(transfer_mass), 0.0), movable)
    if delta <= eps:
        return None, 0.0
    row[int(src_slot)] -= float(delta)
    row[int(dst_slot)] += float(delta)
    row = np.clip(row, 0.0, None)
    row_sum = float(np.sum(row))
    if row_sum <= eps:
        return None, 0.0
    row = row / row_sum
    return row.astype(np.float32, copy=False), float(delta)


def _argmax_index(values: list[float]) -> int:
    if not values:
        return -1
    return int(np.argmax(np.asarray(values, dtype=np.float64)))


def _analyze_rows(
    *,
    cfg,
    rows: list[dict[str, Any]],
    transfer_mass: float,
) -> dict[str, Any]:
    probe_env = make_structured_env(cfg, mode="script")
    probe_driver = as_structured_driver(probe_env)
    try:
        user_level_rows: list[dict[str, Any]] = []
        representative_samples: list[dict[str, Any]] = []
        corr_best_receive_vs_weighted_queue: list[float] = []
        corr_best_receive_vs_queue: list[float] = []
        corr_best_receive_vs_local_cost: list[float] = []
        cold_top_receive_flags: list[float] = []
        top_receive_in_top_weighted_queue_flags: list[float] = []

        for row in rows:
            snapshot_state = dict(row["snapshot_state"])
            det_action = np.asarray(row["det_action"], dtype=np.float32)
            valid_mask = np.asarray(row["valid_mask"], dtype=bool)
            candidate_indices = np.asarray(row["candidate_indices"], dtype=np.int64)
            arrival_rates = np.asarray(row["arrival_rates"], dtype=np.float32).reshape(-1)
            queue_raw = np.asarray(row["queue_raw"], dtype=np.float32).reshape(-1)
            reward_features_raw = {
                key: np.asarray(value, dtype=np.float32).reshape(-1)
                for key, value in dict(row["reward_features_raw"]).items()
            }
            reward_features_norm = {
                key: np.asarray(value, dtype=np.float32).reshape(-1)
                for key, value in dict(row["reward_features_norm"]).items()
            }

            base_reward = _single_step_reward_from_snapshot(
                probe_driver=probe_driver,
                snapshot_state=snapshot_state,
                bw_action=det_action,
            )
            base_row = np.asarray(det_action[0], dtype=np.float32)
            valid_slots = np.flatnonzero(np.asarray(valid_mask[0], dtype=bool))
            slot_to_gu = np.asarray(candidate_indices[0], dtype=np.int64)
            hot_mask = arrival_rates > float(np.median(arrival_rates))

            per_user_best_receive = np.full((cfg.num_gu,), float("-inf"), dtype=np.float32)
            per_user_mean_receive: list[list[float]] = [[] for _ in range(cfg.num_gu)]
            pair_rows: list[dict[str, Any]] = []
            for src_slot in valid_slots.tolist():
                for dst_slot in valid_slots.tolist():
                    if int(src_slot) == int(dst_slot):
                        continue
                    perturbed_row, used_delta = _perturb_transfer(
                        base_row,
                        src_slot=int(src_slot),
                        dst_slot=int(dst_slot),
                        transfer_mass=float(transfer_mass),
                    )
                    if perturbed_row is None or used_delta <= 1.0e-8:
                        continue
                    perturbed_action = det_action.copy()
                    perturbed_action[0] = perturbed_row
                    reward = _single_step_reward_from_snapshot(
                        probe_driver=probe_driver,
                        snapshot_state=snapshot_state,
                        bw_action=perturbed_action,
                    )
                    delta = float(reward - base_reward)
                    src_gu = int(slot_to_gu[int(src_slot)])
                    dst_gu = int(slot_to_gu[int(dst_slot)])
                    pair_rows.append(
                        {
                            "src_slot": int(src_slot),
                            "dst_slot": int(dst_slot),
                            "src_gu": int(src_gu),
                            "dst_gu": int(dst_gu),
                            "delta": float(delta),
                            "used_delta": float(used_delta),
                        }
                    )
                    per_user_best_receive[dst_gu] = max(float(per_user_best_receive[dst_gu]), float(delta))
                    per_user_mean_receive[dst_gu].append(float(delta))

            per_user_best_receive[~np.isfinite(per_user_best_receive)] = 0.0
            per_user_mean_receive_vals = np.asarray(
                [_safe_mean(v) for v in per_user_mean_receive],
                dtype=np.float32,
            )
            top_receive_user = _argmax_index(per_user_best_receive.tolist())
            weighted_queue_raw = reward_features_raw["weighted_queue_cost"]
            top_weighted_queue_user = _argmax_index(weighted_queue_raw.tolist())
            cold_top_receive_flags.append(1.0 if top_receive_user >= 0 and not bool(hot_mask[top_receive_user]) else 0.0)
            top_receive_in_top_weighted_queue_flags.append(
                1.0 if top_receive_user >= 0 and int(top_receive_user) == int(top_weighted_queue_user) else 0.0
            )
            corr_best_receive_vs_weighted_queue.append(
                _safe_corr(weighted_queue_raw.tolist(), per_user_best_receive.tolist())
            )
            corr_best_receive_vs_queue.append(
                _safe_corr(queue_raw.tolist(), per_user_best_receive.tolist())
            )
            corr_best_receive_vs_local_cost.append(
                _safe_corr(
                    reward_features_raw["local_gu_service_cost"].tolist(),
                    per_user_best_receive.tolist(),
                )
            )

            for gu_idx in range(int(cfg.num_gu)):
                user_level_rows.append(
                    {
                        "sample_index": int(row["sample_index"]),
                        "episode": int(row["episode"]),
                        "t": int(row["t"]),
                        "gu": int(gu_idx),
                        "arrival_rate": float(arrival_rates[gu_idx]),
                        "is_hot": bool(hot_mask[gu_idx]),
                        "queue_raw": float(queue_raw[gu_idx]),
                        "det_weight": float(base_row[int(np.where(slot_to_gu == gu_idx)[0][0])]) if np.any(slot_to_gu == gu_idx) else 0.0,
                        "best_receive_delta": float(per_user_best_receive[gu_idx]),
                        "mean_receive_delta": float(per_user_mean_receive_vals[gu_idx]),
                        "local_gu_service_cost_raw": float(reward_features_raw["local_gu_service_cost"][gu_idx]),
                        "assoc_uav_cost_raw": float(reward_features_raw["assoc_uav_cost"][gu_idx]),
                        "assoc_sat_cost_mean_raw": float(reward_features_raw["assoc_sat_cost_mean"][gu_idx]),
                        "weighted_queue_cost_raw": float(weighted_queue_raw[gu_idx]),
                        "weighted_queue_cost_relative_raw": float(
                            reward_features_raw["weighted_queue_cost_relative"][gu_idx]
                        ),
                        "local_gu_service_cost_norm": float(reward_features_norm["local_gu_service_cost"][gu_idx]),
                        "assoc_uav_cost_norm": float(reward_features_norm["assoc_uav_cost"][gu_idx]),
                        "assoc_sat_cost_mean_norm": float(reward_features_norm["assoc_sat_cost_mean"][gu_idx]),
                        "weighted_queue_cost_norm": float(reward_features_norm["weighted_queue_cost"][gu_idx]),
                        "weighted_queue_cost_relative_norm": float(
                            reward_features_norm["weighted_queue_cost_relative"][gu_idx]
                        ),
                    }
                )

            if len(representative_samples) < 12:
                top_pairs = sorted(pair_rows, key=lambda item: float(item["delta"]), reverse=True)[:5]
                representative_samples.append(
                    {
                        "sample_index": int(row["sample_index"]),
                        "episode": int(row["episode"]),
                        "t": int(row["t"]),
                        "base_reward": float(base_reward),
                        "top_receive_user": int(top_receive_user),
                        "top_receive_is_hot": bool(top_receive_user >= 0 and hot_mask[top_receive_user]),
                        "top_weighted_queue_user": int(top_weighted_queue_user),
                        "arrival_rates": arrival_rates.tolist(),
                        "queue_raw": queue_raw.tolist(),
                        "best_receive_delta": per_user_best_receive.tolist(),
                        "weighted_queue_cost_raw": weighted_queue_raw.tolist(),
                        "det_action": base_row.tolist(),
                        "top_pairs": top_pairs,
                    }
                )

        def _feature_block(rows_all: list[dict[str, Any]], key: str) -> dict[str, Any]:
            vals = [float(row[key]) for row in rows_all]
            hot_vals = [float(row[key]) for row in rows_all if bool(row["is_hot"])]
            cold_vals = [float(row[key]) for row in rows_all if not bool(row["is_hot"])]
            payload = _feature_summary(vals)
            payload["hot_mean"] = _safe_mean(hot_vals)
            payload["cold_mean"] = _safe_mean(cold_vals)
            return payload

        summary = {
            "sample_count": int(len(rows)),
            "transfer_mass": float(transfer_mass),
            "mean_corr_best_receive_vs_weighted_queue": _safe_mean(corr_best_receive_vs_weighted_queue),
            "mean_corr_best_receive_vs_queue": _safe_mean(corr_best_receive_vs_queue),
            "mean_corr_best_receive_vs_local_service_cost": _safe_mean(corr_best_receive_vs_local_cost),
            "cold_top_receive_frac": _safe_mean(cold_top_receive_flags),
            "top_receive_matches_top_weighted_queue_frac": _safe_mean(top_receive_in_top_weighted_queue_flags),
            "best_receive_delta_hot_mean": _safe_mean(
                [float(row["best_receive_delta"]) for row in user_level_rows if bool(row["is_hot"])]
            ),
            "best_receive_delta_cold_mean": _safe_mean(
                [float(row["best_receive_delta"]) for row in user_level_rows if not bool(row["is_hot"])]
            ),
            "det_weight_hot_mean": _safe_mean(
                [float(row["det_weight"]) for row in user_level_rows if bool(row["is_hot"])]
            ),
            "det_weight_cold_mean": _safe_mean(
                [float(row["det_weight"]) for row in user_level_rows if not bool(row["is_hot"])]
            ),
            "feature_stats": {
                "local_gu_service_cost_raw": _feature_block(user_level_rows, "local_gu_service_cost_raw"),
                "assoc_uav_cost_raw": _feature_block(user_level_rows, "assoc_uav_cost_raw"),
                "assoc_sat_cost_mean_raw": _feature_block(user_level_rows, "assoc_sat_cost_mean_raw"),
                "weighted_queue_cost_raw": _feature_block(user_level_rows, "weighted_queue_cost_raw"),
                "weighted_queue_cost_relative_raw": _feature_block(user_level_rows, "weighted_queue_cost_relative_raw"),
                "local_gu_service_cost_norm": _feature_block(user_level_rows, "local_gu_service_cost_norm"),
                "assoc_uav_cost_norm": _feature_block(user_level_rows, "assoc_uav_cost_norm"),
                "assoc_sat_cost_mean_norm": _feature_block(user_level_rows, "assoc_sat_cost_mean_norm"),
                "weighted_queue_cost_norm": _feature_block(user_level_rows, "weighted_queue_cost_norm"),
                "weighted_queue_cost_relative_norm": _feature_block(user_level_rows, "weighted_queue_cost_relative_norm"),
            },
            "representative_samples": representative_samples,
            "user_level_rows": user_level_rows,
        }
        return summary
    finally:
        close_fn = getattr(probe_env, "close", None)
        if callable(close_fn):
            close_fn()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose det-action local marginal BW teacher and reward-aligned features.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=3042)
    parser.add_argument("--num_samples", type=int, default=160)
    parser.add_argument("--transfer_mass", type=float, default=0.02)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_all_seeds(int(args.seed))
    cfg = load_config(str(args.config))
    device = torch.device(str(args.device))

    bundle = build_structured_modules_from_config(cfg, build_critic=False)
    actor = bundle.actor.to(device)
    actor.eval()
    load_checkpoint_forgiving(actor, str(args.checkpoint), map_location=device, strict=True)

    rows, episodes = _collect_actor_snapshots(
        cfg=cfg,
        actor=actor,
        device=device,
        num_samples=int(args.num_samples),
        seed_base=int(args.seed),
    )
    summary = _analyze_rows(
        cfg=cfg,
        rows=rows,
        transfer_mass=float(args.transfer_mass),
    )
    payload = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "seed": int(args.seed),
        "num_samples": int(args.num_samples),
        "episodes": int(episodes),
        "transfer_mass": float(args.transfer_mass),
        "summary": summary,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"saved {out_path} | samples={int(args.num_samples)} | "
        f"corr(weighted_queue,best_receive)={float(summary['mean_corr_best_receive_vs_weighted_queue']):+.3f} | "
        f"cold_top_receive_frac={float(summary['cold_top_receive_frac']):.3f}"
    )


if __name__ == "__main__":
    main()
