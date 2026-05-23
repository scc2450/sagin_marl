from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_parallel_eval import (
    batched_policy_accel_actions,
    batched_policy_bw_outputs,
    batched_policy_sat_pair_indices,
)
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.utils.runtime_state_bank import save_runtime_state_bank_payload
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def _resolve_device(raw: str) -> torch.device:
    device_name = str(raw).strip().lower()
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def _load_actor(run_dir: Path, update: int, device: torch.device):
    cfg = load_config(str(run_dir / "config_source.yaml"))
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    actor_ckpt = run_dir / f"actor_u{int(update):04d}.pt"
    if not actor_ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {actor_ckpt}")
    load_checkpoint_forgiving(bundle.actor, str(actor_ckpt), map_location=device, strict=True)
    bundle.actor.to(device).eval()
    return cfg, bundle.actor


def _make_local_drivers(cfg, num_envs: int) -> list[StructuredControlDriver]:
    drivers: list[StructuredControlDriver] = []
    for env_index in range(max(int(num_envs), 1)):
        env = make_structured_env(cfg, mode="script")
        drivers.append(as_structured_driver(env))
    return drivers


def _close_local_drivers(drivers: list[StructuredControlDriver]) -> None:
    for driver in drivers:
        close_fn = getattr(driver.env, "close", None)
        if callable(close_fn):
            close_fn()


def _step_reward(driver: StructuredControlDriver, bw_action: np.ndarray) -> float:
    step_result = driver.execute_stage_bw_and_step(np.asarray(bw_action, dtype=np.float32))
    return float(next(iter(step_result.rewards.values())))


def _compute_state_local_opportunity(
    driver: StructuredControlDriver,
    snapshot_state: dict[str, Any],
    base_action: np.ndarray,
    valid_mask: np.ndarray,
    delta: float,
) -> dict[str, float]:
    driver.load_bw_stage_state(snapshot_state)
    base_reward = _step_reward(driver, base_action)
    per_u_best: list[float] = []
    per_u_positive: list[float] = []
    valid_uav_count = 0
    positive_target_count = 0
    total_target_count = 0
    valid_mask_arr = np.asarray(valid_mask, dtype=bool)
    base_action_arr = np.asarray(base_action, dtype=np.float32)
    for u in range(int(valid_mask_arr.shape[0])):
        valid = valid_mask_arr[u]
        valid_count = int(np.sum(valid))
        if valid_count <= 1:
            continue
        valid_uav_count += 1
        base_action_u = np.asarray(base_action_arr[u], dtype=np.float32)
        gains: list[float] = []
        for target_slot in np.flatnonzero(valid).tolist():
            cf_action_u, _ = StructuredControlDriver._reallocate_bw_toward_slot(
                base_action_u,
                valid,
                int(target_slot),
                float(delta),
            )
            if cf_action_u is None:
                continue
            cf_action = np.asarray(base_action_arr, dtype=np.float32).copy()
            cf_action[u] = cf_action_u
            driver.load_bw_stage_state(snapshot_state)
            cf_reward = _step_reward(driver, cf_action)
            gain = float(cf_reward - base_reward)
            gains.append(gain)
        if not gains:
            continue
        gains_arr = np.asarray(gains, dtype=np.float64)
        per_u_best.append(float(np.max(gains_arr)))
        positive_mask = gains_arr > 0.0
        per_u_positive.append(float(np.mean(positive_mask)))
        positive_target_count += int(np.sum(positive_mask))
        total_target_count += int(gains_arr.size)
    score_sum = float(np.sum(np.clip(np.asarray(per_u_best, dtype=np.float64), a_min=0.0, a_max=None))) if per_u_best else 0.0
    return {
        "state_score_sum": float(score_sum),
        "state_score_mean": float(np.mean(np.asarray(per_u_best, dtype=np.float64))) if per_u_best else 0.0,
        "state_score_max": float(np.max(np.asarray(per_u_best, dtype=np.float64))) if per_u_best else 0.0,
        "positive_uav_fraction": float(np.mean(np.asarray(np.asarray(per_u_best) > 0.0, dtype=np.float64))) if per_u_best else 0.0,
        "positive_target_fraction": float(positive_target_count / max(total_target_count, 1)),
        "valid_uav_count": float(valid_uav_count),
        "base_reward": float(base_reward),
    }


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def collect_bank(
    run_dir: Path,
    update: int,
    *,
    out_dir: Path,
    episodes: int,
    episode_seed_base: int,
    num_envs: int,
    candidate_states: int,
    bank_size: int,
    cf_delta: float,
    rollout_deterministic: bool,
    score_deterministic: bool,
    device: torch.device,
    selection_seed: int,
) -> dict[str, Any]:
    cfg, actor = _load_actor(run_dir, int(update), device)
    drivers = _make_local_drivers(cfg, num_envs=int(num_envs))
    selection_rng = np.random.default_rng(int(selection_seed))
    active_slots = max(min(int(num_envs), int(episodes)), 1)
    slot_active = [True for _ in range(active_slots)]
    next_episode = active_slots
    rows: list[dict[str, Any]] = []
    runtime_states: list[dict[str, Any]] = []
    candidate_row_indices: list[int] = []

    initial_seeds = [int(episode_seed_base) + slot for slot in range(active_slots)]
    for slot, seed in enumerate(initial_seeds):
        drivers[slot].env.reset(seed=int(seed))

    try:
        while any(slot_active) and len(rows) < int(candidate_states):
            active_indices = [slot for slot, is_active in enumerate(slot_active) if is_active]
            if not active_indices:
                break
            pre_states = [drivers[slot].env.export_runtime_state() for slot in active_indices]
            accel_world_states = [drivers[slot].begin_step() for slot in active_indices]
            accel_actions = batched_policy_accel_actions(actor, accel_world_states, device, deterministic=rollout_deterministic)
            sat_world_states = [
                drivers[slot].run_accel_stage(action)
                for slot, action in zip(active_indices, accel_actions)
            ]
            sat_snapshots = [
                drivers[slot].build_sat_stage_snapshot(world_state)
                for slot, world_state in zip(active_indices, sat_world_states)
            ]
            sat_pair_indices = batched_policy_sat_pair_indices(actor, sat_snapshots, device, deterministic=rollout_deterministic)
            sat_actions = [
                drivers[slot].decode_sat_pair_actions([], pair_idx)
                for slot, pair_idx in zip(active_indices, sat_pair_indices)
            ]
            bw_world_states = [
                drivers[slot].run_sat_stage(action)
                for slot, action in zip(active_indices, sat_actions)
            ]
            bw_snapshots = [
                drivers[slot].build_bw_stage_snapshot(world_state)
                for slot, world_state in zip(active_indices, bw_world_states)
            ]
            rollout_bw_eval = batched_policy_bw_outputs(
                actor,
                bw_snapshots,
                device,
                deterministic=rollout_deterministic,
            )
            if rollout_deterministic == score_deterministic:
                score_bw_eval = rollout_bw_eval
            else:
                score_bw_eval = batched_policy_bw_outputs(
                    actor,
                    bw_snapshots,
                    device,
                    deterministic=score_deterministic,
                )

            for local_slot, slot in enumerate(active_indices):
                if len(rows) >= int(candidate_states):
                    break
                driver = drivers[slot]
                runtime_state = pre_states[local_slot]
                snapshot_state = driver.export_bw_stage_state()
                rollout_bw_action = np.asarray(rollout_bw_eval.actions[local_slot], dtype=np.float32)
                score_bw_action = np.asarray(score_bw_eval.actions[local_slot], dtype=np.float32)
                score_metrics = _compute_state_local_opportunity(
                    driver,
                    snapshot_state,
                    score_bw_action,
                    np.asarray(bw_snapshots[local_slot].bw_valid_mask, dtype=bool),
                    cf_delta,
                )
                row_index = len(rows)
                row = {
                    "row_index": int(row_index),
                    "episode_seed": int(runtime_state.get("seed", -1) if runtime_state.get("seed", None) is not None else -1),
                    "t": int(runtime_state.get("t", 0) or 0),
                    "global_step": int(runtime_state.get("global_step", 0) or 0),
                    "score_rank_key": float(score_metrics["state_score_sum"]),
                    **score_metrics,
                }
                rows.append(row)
                runtime_states.append(runtime_state)
                candidate_row_indices.append(int(row_index))

                driver.load_bw_stage_state(snapshot_state)
                live_step = driver.execute_stage_bw_and_step(rollout_bw_action)
                done = bool(any(live_step.terminations.values()) or any(live_step.truncations.values()))
                if done:
                    if next_episode < int(episodes):
                        driver.env.reset(seed=int(episode_seed_base) + int(next_episode))
                        next_episode += 1
                    else:
                        slot_active[slot] = False
    finally:
        _close_local_drivers(drivers)

    if not rows:
        raise RuntimeError("No candidate runtime states were collected.")

    bank_count = min(max(int(bank_size), 1), len(rows))
    sorted_indices = sorted(
        range(len(rows)),
        key=lambda idx: (
            float(rows[idx]["score_rank_key"]),
            float(rows[idx]["state_score_mean"]),
            -int(rows[idx]["row_index"]),
        ),
        reverse=True,
    )
    high_indices = sorted_indices[:bank_count]
    available_random = [idx for idx in range(len(rows)) if idx not in set(high_indices)]
    if len(available_random) >= bank_count:
        random_indices = selection_rng.choice(np.asarray(available_random, dtype=np.int64), size=bank_count, replace=False).astype(np.int64).tolist()
    else:
        random_indices = selection_rng.choice(np.arange(len(rows), dtype=np.int64), size=bank_count, replace=False).astype(np.int64).tolist()

    high_entries = [runtime_states[idx] for idx in high_indices]
    random_entries = [runtime_states[idx] for idx in random_indices]

    out_dir.mkdir(parents=True, exist_ok=True)
    high_path = out_dir / "high_local_opportunity_bank.pkl.gz"
    random_path = out_dir / "random_bank.pkl.gz"
    save_runtime_state_bank_payload(
        high_path,
        {
            "format": "runtime_state_bank_v1",
            "bank_kind": "high_local_opportunity",
            "source_run_dir": str(run_dir),
            "source_update": int(update),
            "selection_seed": int(selection_seed),
            "selected_row_indices": [int(idx) for idx in high_indices],
            "entries": high_entries,
        },
    )
    save_runtime_state_bank_payload(
        random_path,
        {
            "format": "runtime_state_bank_v1",
            "bank_kind": "random",
            "source_run_dir": str(run_dir),
            "source_update": int(update),
            "selection_seed": int(selection_seed),
            "selected_row_indices": [int(idx) for idx in random_indices],
            "entries": random_entries,
        },
    )

    _write_rows_csv(out_dir / "state_rows.csv", rows)
    summary = {
        "run_dir": str(run_dir),
        "update": int(update),
        "episodes": int(episodes),
        "episode_seed_base": int(episode_seed_base),
        "num_envs": int(active_slots),
        "candidate_state_count": int(len(rows)),
        "bank_size": int(bank_count),
        "cf_delta": float(cf_delta),
        "rollout_deterministic": bool(rollout_deterministic),
        "score_deterministic": bool(score_deterministic),
        "selection_seed": int(selection_seed),
        "state_score_sum": _safe_stats([float(row["state_score_sum"]) for row in rows]),
        "state_score_mean": _safe_stats([float(row["state_score_mean"]) for row in rows]),
        "positive_uav_fraction": _safe_stats([float(row["positive_uav_fraction"]) for row in rows]),
        "positive_target_fraction": _safe_stats([float(row["positive_target_fraction"]) for row in rows]),
        "valid_uav_count": _safe_stats([float(row["valid_uav_count"]) for row in rows]),
        "t": _safe_stats([float(row["t"]) for row in rows]),
        "selected_high_row_indices": [int(idx) for idx in high_indices],
        "selected_random_row_indices": [int(idx) for idx in random_indices],
        "high_local_opportunity_bank": str(high_path),
        "random_bank": str(random_path),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--episode-seed-base", type=int, default=94000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--candidate-states", type=int, default=192)
    parser.add_argument("--bank-size", type=int, default=64)
    parser.add_argument("--cf-delta", type=float, default=0.05)
    parser.add_argument("--rollout-policy-mode", choices=["deterministic", "stochastic"], default="stochastic")
    parser.add_argument("--score-policy-mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--selection-seed", type=int, default=20260406)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    summary = collect_bank(
        Path(args.run_dir),
        int(args.update),
        out_dir=Path(args.out_dir),
        episodes=int(args.episodes),
        episode_seed_base=int(args.episode_seed_base),
        num_envs=int(args.num_envs),
        candidate_states=int(args.candidate_states),
        bank_size=int(args.bank_size),
        cf_delta=float(args.cf_delta),
        rollout_deterministic=(args.rollout_policy_mode == "deterministic"),
        score_deterministic=(args.score_policy_mode == "deterministic"),
        device=_resolve_device(args.device),
        selection_seed=int(args.selection_seed),
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
