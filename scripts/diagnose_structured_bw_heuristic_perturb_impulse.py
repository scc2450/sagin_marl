from __future__ import annotations

import argparse
import json
import os
import random
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
from sagin_marl.rl.baselines import queue_aware_bw_policy
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


def _heuristic_action_from_driver(driver: StructuredControlDriver, cfg) -> np.ndarray:
    obs = {agent: driver.env._get_obs(idx) for idx, agent in enumerate(driver.env.agents)}
    return np.asarray(queue_aware_bw_policy(list(obs.values()), cfg), dtype=np.float32)


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


def _heuristic_perturb_action(
    base_action: np.ndarray,
    valid_mask: np.ndarray,
    *,
    perturb_mode: str,
    transfer_mass: float,
    transfer_frac: float,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    base = _normalize_bw_action(base_action, valid_mask)
    out = np.asarray(base, dtype=np.float32).copy()
    metadata: list[dict[str, float | int]] = []
    eps = 1.0e-8
    mode = str(perturb_mode or "absolute").strip().lower()
    if mode not in {"absolute", "relative"}:
        raise ValueError(f"Unsupported perturb_mode: {perturb_mode}")
    for u in range(int(out.shape[0])):
        valid = np.flatnonzero(np.asarray(valid_mask[u], dtype=bool))
        if valid.size < 2:
            metadata.append({"uav": int(u), "src": -1, "dst": -1, "delta": 0.0, "src_weight": 0.0})
            continue
        row = np.asarray(out[u], dtype=np.float32)
        src = int(valid[np.argmax(row[valid])])
        other = valid[valid != src]
        if other.size <= 0:
            metadata.append({"uav": int(u), "src": src, "dst": -1, "delta": 0.0, "src_weight": float(row[src])})
            continue
        dst = int(other[np.argmin(row[other])])
        movable = max(float(row[src]) - eps, 0.0)
        if mode == "relative":
            proposed_delta = max(float(row[src]) * max(float(transfer_frac), 0.0), 0.0)
        else:
            proposed_delta = max(float(transfer_mass), 0.0)
        delta = min(float(proposed_delta), movable)
        if delta > 0.0:
            row[src] -= float(delta)
            row[dst] += float(delta)
        out[u] = row
        metadata.append(
            {
                "uav": int(u),
                "src": src,
                "dst": dst,
                "delta": float(delta),
                "src_weight": float(base[u, src]),
            }
        )
    return _normalize_bw_action(out, valid_mask), metadata


def _prepare_bw_snapshot(driver: StructuredControlDriver, cfg) -> tuple[dict[str, Any], Any, int]:
    driver.begin_step()
    accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
    _ = driver.run_accel_stage(accel_zero)
    bw_world = driver.run_sat_stage(_zero_sat_action(cfg))
    snapshot = driver.build_bw_stage_snapshot(bw_world)
    snapshot_state = driver.export_bw_stage_state()
    env_state = snapshot_state.get("env_state", {}) if isinstance(snapshot_state, dict) else {}
    t = int(env_state.get("t", 0) or 0) if isinstance(env_state, dict) else 0
    return snapshot_state, snapshot, t


def _collect_heuristic_bw_samples(
    *,
    cfg,
    num_samples: int,
    seed_base: int,
    perturb_mode: str,
    transfer_mass: float,
    transfer_frac: float,
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
                snapshot_state, snapshot, t = _prepare_bw_snapshot(driver, cfg)
                valid_mask = np.asarray(snapshot.bw_valid_mask, dtype=bool)
                heuristic_action = _heuristic_action_from_driver(driver, cfg)
                heuristic_action = _normalize_bw_action(heuristic_action, valid_mask)
                perturbed_action, perturb_meta = _heuristic_perturb_action(
                    heuristic_action,
                    valid_mask,
                    perturb_mode=str(perturb_mode),
                    transfer_mass=float(transfer_mass),
                    transfer_frac=float(transfer_frac),
                )
                rows.append(
                    {
                        "sample_index": int(len(rows)),
                        "episode": int(episode_count),
                        "t": int(t),
                        "snapshot_state": snapshot_state,
                        "heuristic_action": np.asarray(heuristic_action, dtype=np.float32).tolist(),
                        "perturbed_action": np.asarray(perturbed_action, dtype=np.float32).tolist(),
                        "perturb_meta": perturb_meta,
                        "first_action_l1": float(np.sum(np.abs(heuristic_action - perturbed_action))),
                    }
                )
                step_result = driver.execute_stage_bw_and_step(heuristic_action)
                done = _done_from_step_result(step_result)
            episode_count += 1
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
    return rows, int(episode_count)


def _timeline_step_record(env, k: int, step_result) -> dict[str, Any]:
    reward = float(next(iter(step_result.rewards.values())))
    return {
        "k": int(k),
        "reward": float(reward),
        "gu_queue": np.asarray(env.gu_queue, dtype=np.float32).tolist(),
        "last_arrival": np.asarray(
            getattr(env, "last_gu_arrival", np.zeros((env.cfg.num_gu,), dtype=np.float32)),
            dtype=np.float32,
        ).tolist(),
        "last_outflow": np.asarray(
            getattr(env, "last_gu_outflow", np.zeros((env.cfg.num_gu,), dtype=np.float32)),
            dtype=np.float32,
        ).tolist(),
    }


def _rollout_timeline_with_driver(
    *,
    probe_driver: StructuredControlDriver,
    snapshot_state: dict[str, Any],
    first_action: np.ndarray,
    cfg,
    max_k: int,
) -> list[dict[str, Any]]:
    probe_driver.load_bw_stage_state(snapshot_state)
    action = np.asarray(first_action, dtype=np.float32)
    timeline: list[dict[str, Any]] = []
    for step_idx in range(int(max_k)):
        step_result, _next_world = probe_driver.execute_stage_bw_and_prepare_next_accel(action)
        timeline.append(_timeline_step_record(probe_driver.env, step_idx + 1, step_result))
        if _done_from_step_result(step_result) or step_idx + 1 >= int(max_k):
            break
        probe_driver.begin_step()
        accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
        _ = probe_driver.run_accel_stage(accel_zero)
        _ = probe_driver.run_sat_stage(_zero_sat_action(cfg))
        action = _heuristic_action_from_driver(probe_driver, cfg)
    return timeline


def _discounted_prefix(values: list[float], gamma: float) -> list[float]:
    total = 0.0
    discount = 1.0
    out: list[float] = []
    for value in values:
        total += discount * float(value)
        out.append(float(total))
        discount *= float(gamma)
    return out


def _safe_percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), float(q)))


def _build_summary(rows: list[dict[str, Any]], *, max_k: int, gamma: float) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for k in range(1, int(max_k) + 1):
        eligible = [row for row in rows if len(row["queue_l1_by_k"]) >= int(k)]
        queue_vals = [float(row["queue_l1_by_k"][k - 1]) for row in eligible]
        reward_deltas = [float(row["reward_delta_by_k"][k - 1]) for row in eligible]
        cum_deltas = [
            float(_discounted_prefix(row["reward_delta_by_k"], gamma)[k - 1])
            for row in eligible
        ]
        summary.append(
            {
                "k": int(k),
                "count": int(len(eligible)),
                "mean_queue_l1": float(np.mean(np.asarray(queue_vals, dtype=np.float64))) if queue_vals else 0.0,
                "median_queue_l1": _safe_percentile(queue_vals, 50.0),
                "p90_queue_l1": _safe_percentile(queue_vals, 90.0),
                "frac_queue_l1_gt_1e5": float(np.mean((np.asarray(queue_vals, dtype=np.float64) > 1.0e5).astype(np.float64))) if queue_vals else 0.0,
                "frac_queue_l1_gt_5e5": float(np.mean((np.asarray(queue_vals, dtype=np.float64) > 5.0e5).astype(np.float64))) if queue_vals else 0.0,
                "mean_reward_delta": float(np.mean(np.asarray(reward_deltas, dtype=np.float64))) if reward_deltas else 0.0,
                "mean_abs_reward_delta": float(np.mean(np.abs(np.asarray(reward_deltas, dtype=np.float64)))) if reward_deltas else 0.0,
                "mean_discounted_cum_delta": float(np.mean(np.asarray(cum_deltas, dtype=np.float64))) if cum_deltas else 0.0,
                "mean_abs_discounted_cum_delta": float(np.mean(np.abs(np.asarray(cum_deltas, dtype=np.float64)))) if cum_deltas else 0.0,
            }
        )
    return summary


def _pick_representative_indices(rows: list[dict[str, Any]]) -> list[int]:
    if not rows:
        return []
    queue_mass = [float(np.sum(np.asarray(row["queue_l1_by_k"], dtype=np.float64))) for row in rows]
    zero_like = int(np.argmin(np.asarray(queue_mass, dtype=np.float64)))

    fade_candidates = [
        idx
        for idx, row in enumerate(rows)
        if row["queue_l1_by_k"]
        and float(row["queue_l1_by_k"][0]) > 0.0
        and any(float(value) <= 1.0e-6 for value in row["queue_l1_by_k"][1:3])
    ]
    if fade_candidates:
        fade_like = max(fade_candidates, key=lambda idx: float(rows[idx]["queue_l1_by_k"][0]))
    else:
        fade_like = max(
            range(len(rows)),
            key=lambda idx: float(rows[idx]["queue_l1_by_k"][0]) if rows[idx]["queue_l1_by_k"] else 0.0,
        )

    tail_candidates = [
        idx
        for idx, row in enumerate(rows)
        if len(row["queue_l1_by_k"]) >= 5 and all(float(value) > 1.0e-6 for value in row["queue_l1_by_k"][:5])
    ]
    if tail_candidates:
        tail_like = max(tail_candidates, key=lambda idx: float(rows[idx]["queue_l1_by_k"][4]))
    else:
        tail_like = max(
            range(len(rows)),
            key=lambda idx: float(rows[idx]["queue_l1_by_k"][-1]) if rows[idx]["queue_l1_by_k"] else 0.0,
        )

    chosen: list[int] = []
    for idx in (zero_like, fade_like, tail_like):
        if int(idx) not in chosen:
            chosen.append(int(idx))
    return chosen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--max_k", type=int, default=5)
    parser.add_argument("--seed_base", type=int, default=3042)
    parser.add_argument("--perturb_mode", choices=["absolute", "relative"], default="relative")
    parser.add_argument("--transfer_mass", type=float, default=0.1)
    parser.add_argument("--transfer_frac", type=float, default=0.2)
    parser.add_argument("--out_path", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_all_seeds(int(args.seed_base))
    cfg = load_config(str(args.config))
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sample_rows, episode_count = _collect_heuristic_bw_samples(
        cfg=cfg,
        num_samples=int(args.num_samples),
        seed_base=int(args.seed_base),
        perturb_mode=str(args.perturb_mode),
        transfer_mass=float(args.transfer_mass),
        transfer_frac=float(args.transfer_frac),
    )

    probe_env = make_structured_env(cfg, mode="script")
    probe_driver = as_structured_driver(probe_env)
    analyzed_rows: list[dict[str, Any]] = []
    try:
        for row in sample_rows:
            snapshot_state = row["snapshot_state"]
            heuristic_action = np.asarray(row["heuristic_action"], dtype=np.float32)
            perturbed_action = np.asarray(row["perturbed_action"], dtype=np.float32)
            timeline_base = _rollout_timeline_with_driver(
                probe_driver=probe_driver,
                snapshot_state=snapshot_state,
                first_action=heuristic_action,
                cfg=cfg,
                max_k=int(args.max_k),
            )
            timeline_perturbed = _rollout_timeline_with_driver(
                probe_driver=probe_driver,
                snapshot_state=snapshot_state,
                first_action=perturbed_action,
                cfg=cfg,
                max_k=int(args.max_k),
            )
            common_len = min(len(timeline_base), len(timeline_perturbed))
            queue_l1_by_k: list[float] = []
            reward_delta_by_k: list[float] = []
            for step_idx in range(common_len):
                queue_base = np.asarray(timeline_base[step_idx]["gu_queue"], dtype=np.float64)
                queue_perturbed = np.asarray(timeline_perturbed[step_idx]["gu_queue"], dtype=np.float64)
                queue_l1_by_k.append(float(np.sum(np.abs(queue_perturbed - queue_base))))
                reward_delta_by_k.append(
                    float(timeline_perturbed[step_idx]["reward"]) - float(timeline_base[step_idx]["reward"])
                )
            analyzed_rows.append(
                {
                    "rel_idx": int(row["sample_index"]),
                    "sample_index": int(row["sample_index"]),
                    "episode": int(row["episode"]),
                    "t": int(row["t"]),
                    "heuristic_action": row["heuristic_action"],
                    "perturbed_action": row["perturbed_action"],
                    "perturb_meta": row["perturb_meta"],
                    "first_action_l1": float(row["first_action_l1"]),
                    "timeline_base": timeline_base,
                    "timeline_perturbed": timeline_perturbed,
                    "queue_l1_by_k": queue_l1_by_k,
                    "reward_delta_by_k": reward_delta_by_k,
                }
            )
    finally:
        close_fn = getattr(probe_env, "close", None)
        if callable(close_fn):
            close_fn()

    summary = _build_summary(analyzed_rows, max_k=int(args.max_k), gamma=float(cfg.gamma))
    representative_indices = _pick_representative_indices(analyzed_rows)
    representative_samples = [analyzed_rows[idx] for idx in representative_indices]
    first_action_l1_values = [float(row["first_action_l1"]) for row in analyzed_rows]

    payload = {
        "config": str(Path(args.config)),
        "seed_base": int(args.seed_base),
        "num_bw_samples": int(len(analyzed_rows)),
        "episodes_used": int(episode_count),
        "follow_policy_mode": "heuristic_deterministic",
        "compare": "heuristic_action vs heuristic_action_plus_fixed_transfer",
        "perturb_mode": str(args.perturb_mode),
        "transfer_mass": float(args.transfer_mass),
        "transfer_frac": float(args.transfer_frac),
        "reward_mode": str(getattr(cfg, "reward_mode", "")),
        "gamma": float(cfg.gamma),
        "first_action_l1_mean": float(np.mean(np.asarray(first_action_l1_values, dtype=np.float64))) if first_action_l1_values else 0.0,
        "first_action_l1_median": _safe_percentile(first_action_l1_values, 50.0),
        "summary": summary,
        "representative_samples": representative_samples,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
