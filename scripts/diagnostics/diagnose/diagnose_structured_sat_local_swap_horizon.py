from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.structured_mappo import (
    _current_obs_list,
    _heuristic_accel,
    _heuristic_bw,
    _heuristic_sat,
    _refresh_stage_obs_cache,
    _sat_mask_to_ids,
)
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/clean_sat/structured_sat_clean_joint_beijing_hotspot_res200.yaml",
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--episode-seed-base", type=int, default=52000)
    parser.add_argument("--step-stride", type=int, default=25)
    parser.add_argument("--max-contexts", type=int, default=6)
    parser.add_argument("--horizons", type=str, default="1,2,5,10")
    parser.add_argument("--accel-source", type=str, default="cluster_center_queue_aware")
    parser.add_argument("--current-sat-source", type=str, default="queue_aware")
    parser.add_argument("--followup-sat-source", type=str, default="queue_aware")
    parser.add_argument("--bw-source", type=str, default="queue_aware")
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args()


def _parse_horizons(text: str) -> list[int]:
    values: list[int] = []
    for part in str(text).replace(";", ",").split(","):
        token = part.strip()
        if not token:
            continue
        values.append(max(int(token), 1))
    if not values:
        raise ValueError("horizons must be non-empty")
    return sorted(set(values))


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "p50": None, "p90": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _cluster_centers_and_counts(env: SaginParallelEnv) -> tuple[np.ndarray | None, np.ndarray | None]:
    centers = getattr(env, "gu_cluster_centers", None)
    counts = getattr(env, "gu_cluster_counts", None)
    centers_arr = None if centers is None else np.asarray(centers, dtype=np.float32)
    counts_arr = None if counts is None else np.asarray(counts, dtype=np.float32)
    return centers_arr, counts_arr


def _sat_action_ids_to_text(action_ids: np.ndarray) -> str:
    rows: list[str] = []
    for row in np.asarray(action_ids, dtype=np.int64):
        valid = [str(int(x)) for x in row.tolist() if int(x) >= 0]
        rows.append("-" if not valid else "|".join(valid))
    return ";".join(rows)


def _queue_snapshot(env: SaginParallelEnv) -> dict[str, Any]:
    gu_queue = np.asarray(getattr(env, "gu_queue", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    uav_queue = np.asarray(getattr(env, "uav_queue", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    sat_queue = np.asarray(getattr(env, "sat_queue", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    return {
        "gu_sum": float(np.sum(gu_queue, dtype=np.float64)),
        "uav_sum": float(np.sum(uav_queue, dtype=np.float64)),
        "sat_sum": float(np.sum(sat_queue, dtype=np.float64)),
        "total_sum": float(
            np.sum(gu_queue, dtype=np.float64)
            + np.sum(uav_queue, dtype=np.float64)
            + np.sum(sat_queue, dtype=np.float64)
        ),
        "gu_queue": [float(x) for x in gu_queue.tolist()],
        "uav_queue": [float(x) for x in uav_queue.tolist()],
        "sat_queue": [float(x) for x in sat_queue.tolist()],
    }


def _bw_candidate_ids(driver: StructuredControlDriver) -> list[list[int]]:
    cfg = driver.env.cfg
    candidates = driver._stage_candidates or [[] for _ in range(cfg.num_uav)]
    return [
        [int(x) for x in list(candidates[u])[: cfg.users_obs_max]]
        for u in range(cfg.num_uav)
    ]


def _bw_action_to_text(driver: StructuredControlDriver, bw_action: np.ndarray, *, eps: float = 1.0e-5) -> str:
    cfg = driver.env.cfg
    bw_arr = np.asarray(bw_action, dtype=np.float32)
    candidate_ids = _bw_candidate_ids(driver)
    rows: list[str] = []
    for u in range(cfg.num_uav):
        parts: list[str] = []
        valid_ids = candidate_ids[u]
        for slot, gu_id in enumerate(valid_ids):
            alloc = float(bw_arr[u, slot]) if slot < bw_arr.shape[1] else 0.0
            if abs(alloc) <= float(eps):
                continue
            parts.append(f"{int(gu_id)}:{alloc:.3f}")
        rows.append("-" if not parts else "|".join(parts))
    return ";".join(rows)


def _queue_sum_delta_path(
    best_steps: list[dict[str, Any]],
    baseline_steps: list[dict[str, Any]],
    horizon: int,
    key: str,
) -> list[float | None]:
    out: list[float | None] = []
    for t in range(int(horizon)):
        if t >= len(best_steps) or t >= len(baseline_steps):
            out.append(None)
            continue
        best_after = float(best_steps[t]["queue_after"][key])
        baseline_after = float(baseline_steps[t]["queue_after"][key])
        out.append(float(best_after - baseline_after))
    return out


def _prefix_discounted_returns(step_rewards: list[float], horizons: list[int], gamma: float) -> dict[int, float]:
    arr = np.asarray(step_rewards, dtype=np.float64)
    out: dict[int, float] = {}
    for horizon in horizons:
        total = 0.0
        for t in range(min(int(horizon), int(arr.size))):
            total += (float(gamma) ** float(t)) * float(arr[t])
        out[int(horizon)] = float(total)
    return out


def _path_abs_step0_share(step_delta: list[float], horizon: int) -> float | None:
    if horizon <= 0:
        return None
    prefix = np.asarray(step_delta[: int(horizon)], dtype=np.float64)
    if prefix.size <= 0:
        return None
    denom = float(np.sum(np.abs(prefix)))
    if denom <= 1.0e-12:
        return None
    return float(abs(prefix[0]) / denom)


def _canonicalize_sat_action_row(
    row_sat_ids: list[int],
    *,
    visible_ids: list[int],
    select_k: int,
) -> np.ndarray:
    visible_rank = {int(sat_id): idx for idx, sat_id in enumerate(visible_ids)}
    unique_valid = [int(sat_id) for sat_id in row_sat_ids if int(sat_id) >= 0]
    unique_valid = sorted(set(unique_valid), key=lambda sat_id: visible_rank.get(int(sat_id), 10**9))
    out = np.full((int(select_k),), -1, dtype=np.int64)
    keep = unique_valid[: int(select_k)]
    if keep:
        out[: len(keep)] = np.asarray(keep, dtype=np.int64)
    return out


def _enumerate_single_swap_candidates(
    *,
    current_sat_action_ids: np.ndarray,
    visible_per_uav: list[list[int]],
    select_k: int,
) -> list[dict[str, Any]]:
    current = np.asarray(current_sat_action_ids, dtype=np.int64)
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for u in range(int(current.shape[0])):
        visible_ids = [int(x) for x in list(visible_per_uav[u])[:]]
        current_row = [int(x) for x in current[u].tolist() if int(x) >= 0]
        if not current_row:
            continue
        alternatives = [sat_id for sat_id in visible_ids if sat_id not in set(current_row)]
        if not alternatives:
            continue
        for pos, old_sat in enumerate(current_row):
            for new_sat in alternatives:
                cand = np.asarray(current, dtype=np.int64).copy()
                swapped = list(current_row)
                swapped[pos] = int(new_sat)
                cand[u] = _canonicalize_sat_action_row(
                    swapped,
                    visible_ids=visible_ids,
                    select_k=int(select_k),
                )
                key = tuple(tuple(int(x) for x in row) for row in cand.tolist())
                if key in seen:
                    continue
                seen.add(key)
                if np.array_equal(cand, current):
                    continue
                candidates.append(
                    {
                        "action_ids": cand,
                        "uav": int(u),
                        "drop_sat_id": int(old_sat),
                        "add_sat_id": int(new_sat),
                    }
                )
    return candidates


def _rollout_single_sat_action_path(
    *,
    probe_driver: StructuredControlDriver,
    snapshot_state: dict[str, Any],
    first_sat_action_ids: np.ndarray,
    max_horizon: int,
    cfg,
    accel_source: str,
    followup_sat_source: str,
    bw_source: str,
) -> dict[str, Any]:
    probe_driver.load_sat_stage_state(snapshot_state)
    step_rewards: list[float] = []
    step_traces: list[dict[str, Any]] = []
    centers, counts = _cluster_centers_and_counts(probe_driver.env)

    queue_before = _queue_snapshot(probe_driver.env)
    t_before = int(getattr(probe_driver.env, "t", 0))
    probe_driver.run_sat_stage(np.asarray(first_sat_action_ids, dtype=np.int64))
    _refresh_stage_obs_cache(probe_driver)
    bw_obs = _current_obs_list(probe_driver)
    bw_action = _heuristic_bw(bw_obs, cfg, bw_source).astype(np.float32, copy=False)
    bw_candidate_ids = _bw_candidate_ids(probe_driver)
    bw_action_text = _bw_action_to_text(probe_driver, bw_action)
    step_result, _next_world = probe_driver.execute_stage_bw_and_prepare_next_accel(bw_action)
    reward = float(getattr(step_result, "bw_weighted_workload_level_reward", next(iter(step_result.rewards.values()))))
    step_rewards.append(reward)
    queue_after = _queue_snapshot(probe_driver.env)
    done = bool(next(iter(step_result.terminations.values())) or next(iter(step_result.truncations.values())))
    step_traces.append(
        {
            "step_index": 0,
            "t_before": t_before,
            "t_after": int(getattr(probe_driver.env, "t", t_before)),
            "sat_action_source": "injected",
            "sat_action_ids": np.asarray(first_sat_action_ids, dtype=np.int64).tolist(),
            "sat_action_ids_text": _sat_action_ids_to_text(np.asarray(first_sat_action_ids, dtype=np.int64)),
            "bw_action_source": str(bw_source),
            "bw_candidate_ids": bw_candidate_ids,
            "bw_action": np.asarray(bw_action, dtype=np.float32).tolist(),
            "bw_action_text": bw_action_text,
            "reward": reward,
            "done": bool(done),
            "queue_before": queue_before,
            "queue_after": queue_after,
        }
    )
    if done or int(max_horizon) <= 1:
        return {"step_rewards": step_rewards, "steps": step_traces}

    for _step in range(1, int(max_horizon)):
        queue_before = _queue_snapshot(probe_driver.env)
        t_before = int(getattr(probe_driver.env, "t", 0))
        obs_before = _current_obs_list(probe_driver)
        accel_action = _heuristic_accel(
            obs_before,
            cfg,
            accel_source,
            centers=centers,
            counts=counts,
        ).astype(np.float32, copy=False)
        sat_world = probe_driver.run_accel_stage(accel_action)
        _refresh_stage_obs_cache(probe_driver)
        sat_obs = _current_obs_list(probe_driver)
        sat_mask = _heuristic_sat(sat_obs, cfg, followup_sat_source).astype(np.float32, copy=False)
        sat_action_ids = _sat_mask_to_ids(probe_driver, sat_mask)
        probe_driver.run_sat_stage(sat_action_ids)
        _refresh_stage_obs_cache(probe_driver)
        bw_obs = _current_obs_list(probe_driver)
        bw_action = _heuristic_bw(bw_obs, cfg, bw_source).astype(np.float32, copy=False)
        bw_candidate_ids = _bw_candidate_ids(probe_driver)
        bw_action_text = _bw_action_to_text(probe_driver, bw_action)
        step_result, _next_world = probe_driver.execute_stage_bw_and_prepare_next_accel(bw_action)
        reward = float(getattr(step_result, "bw_weighted_workload_level_reward", next(iter(step_result.rewards.values()))))
        step_rewards.append(reward)
        queue_after = _queue_snapshot(probe_driver.env)
        done = bool(next(iter(step_result.terminations.values())) or next(iter(step_result.truncations.values())))
        step_traces.append(
            {
                "step_index": int(_step),
                "t_before": t_before,
                "t_after": int(getattr(probe_driver.env, "t", t_before)),
                "sat_action_source": str(followup_sat_source),
                "sat_action_ids": np.asarray(sat_action_ids, dtype=np.int64).tolist(),
                "sat_action_ids_text": _sat_action_ids_to_text(np.asarray(sat_action_ids, dtype=np.int64)),
                "bw_action_source": str(bw_source),
                "bw_candidate_ids": bw_candidate_ids,
                "bw_action": np.asarray(bw_action, dtype=np.float32).tolist(),
                "bw_action_text": bw_action_text,
                "reward": reward,
                "done": bool(done),
                "queue_before": queue_before,
                "queue_after": queue_after,
            }
        )
        if done:
            break
    return {"step_rewards": step_rewards, "steps": step_traces}


def _context_analysis(
    *,
    probe_driver: StructuredControlDriver,
    snapshot_state: dict[str, Any],
    current_sat_action_ids: np.ndarray,
    swap_candidates: list[dict[str, Any]],
    horizons: list[int],
    gamma: float,
    cfg,
    accel_source: str,
    followup_sat_source: str,
    bw_source: str,
) -> dict[str, Any]:
    max_horizon = max(int(h) for h in horizons)
    baseline_rollout = _rollout_single_sat_action_path(
        probe_driver=probe_driver,
        snapshot_state=snapshot_state,
        first_sat_action_ids=current_sat_action_ids,
        max_horizon=max_horizon,
        cfg=cfg,
        accel_source=accel_source,
        followup_sat_source=followup_sat_source,
        bw_source=bw_source,
    )
    baseline_path = [float(x) for x in baseline_rollout["step_rewards"]]
    baseline_steps = list(baseline_rollout["steps"])
    baseline_returns = _prefix_discounted_returns(baseline_path, horizons, gamma)

    candidate_rollouts: list[dict[str, Any]] = []
    candidate_paths: list[list[float]] = []
    candidate_returns: list[dict[int, float]] = []
    for candidate in swap_candidates:
        rollout = _rollout_single_sat_action_path(
            probe_driver=probe_driver,
            snapshot_state=snapshot_state,
            first_sat_action_ids=np.asarray(candidate["action_ids"], dtype=np.int64),
            max_horizon=max_horizon,
            cfg=cfg,
            accel_source=accel_source,
            followup_sat_source=followup_sat_source,
            bw_source=bw_source,
        )
        path = [float(x) for x in rollout["step_rewards"]]
        candidate_rollouts.append(rollout)
        candidate_paths.append(path)
        candidate_returns.append(_prefix_discounted_returns(path, horizons, gamma))

    horizon_data: dict[int, dict[str, Any]] = {}
    best_idx_h1: int | None = None
    for horizon in horizons:
        values = [float(item[int(horizon)]) for item in candidate_returns]
        best_idx = int(np.argmax(np.asarray(values, dtype=np.float64))) if values else -1
        if int(horizon) == int(min(horizons)):
            best_idx_h1 = best_idx
        best_return = float(values[best_idx]) if best_idx >= 0 else baseline_returns[int(horizon)]
        best_path = candidate_paths[best_idx] if best_idx >= 0 else baseline_path
        best_steps = (
            list(candidate_rollouts[best_idx]["steps"])
            if best_idx >= 0
            else baseline_steps
        )
        step_delta = [
            float((best_path[t] if t < len(best_path) else 0.0) - (baseline_path[t] if t < len(baseline_path) else 0.0))
            for t in range(int(horizon))
        ]
        best_meta = None if best_idx < 0 else swap_candidates[best_idx]
        horizon_data[int(horizon)] = {
            "baseline_return": float(baseline_returns[int(horizon)]),
            "best_return": best_return,
            "best_gap": float(best_return - baseline_returns[int(horizon)]),
            "best_idx": int(best_idx),
            "same_best_as_h1": None if best_idx_h1 is None or best_idx < 0 else int(best_idx == best_idx_h1),
            "step0_abs_share": _path_abs_step0_share(step_delta, int(horizon)),
            "best_swap_uav": None if best_meta is None else int(best_meta["uav"]),
            "best_swap_drop_sat_id": None if best_meta is None else int(best_meta["drop_sat_id"]),
            "best_swap_add_sat_id": None if best_meta is None else int(best_meta["add_sat_id"]),
            "best_sat_action_ids": None if best_meta is None else _sat_action_ids_to_text(best_meta["action_ids"]),
            "delta_path": [float(x) for x in step_delta],
            "baseline_trace_steps": baseline_steps[: int(horizon)],
            "best_trace_steps": best_steps[: int(horizon)],
            "gu_queue_sum_delta_path": _queue_sum_delta_path(best_steps, baseline_steps, int(horizon), "gu_sum"),
            "uav_queue_sum_delta_path": _queue_sum_delta_path(best_steps, baseline_steps, int(horizon), "uav_sum"),
            "sat_queue_sum_delta_path": _queue_sum_delta_path(best_steps, baseline_steps, int(horizon), "sat_sum"),
            "total_queue_sum_delta_path": _queue_sum_delta_path(best_steps, baseline_steps, int(horizon), "total_sum"),
        }
    return {
        "baseline_path": [float(x) for x in baseline_path],
        "baseline_returns": {str(k): float(v) for k, v in baseline_returns.items()},
        "horizons": horizon_data,
    }


def main() -> None:
    args = parse_args()
    horizons = _parse_horizons(args.horizons)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(args.config))
    gamma = float(getattr(cfg, "gamma", 1.0) or 1.0)
    select_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 1)

    live_env = make_structured_env(cfg, mode="script")
    probe_env = make_structured_env(cfg, mode="script")
    live_driver = as_structured_driver(live_env)
    probe_driver = as_structured_driver(probe_env)
    context_rows: list[dict[str, Any]] = []
    context_traces: list[dict[str, Any]] = []
    horizon_gap_values: dict[int, list[float]] = {int(h): [] for h in horizons}
    horizon_step0_share_values: dict[int, list[float]] = {int(h): [] for h in horizons}
    horizon_same_best_values: dict[int, list[float]] = {int(h): [] for h in horizons if int(h) != int(min(horizons))}
    horizon_growth_vs_h1_values: dict[int, list[float]] = {int(h): [] for h in horizons if int(h) != int(min(horizons))}

    contexts_done = 0
    try:
        for episode in range(int(args.episodes)):
            if contexts_done >= int(args.max_contexts):
                break
            seed = int(args.episode_seed_base) + int(episode)
            live_env.reset(seed=seed)
            done = False
            step = 0
            while not done and contexts_done < int(args.max_contexts):
                _ = live_driver.begin_step()
                obs_before = _current_obs_list(live_driver)
                centers, counts = _cluster_centers_and_counts(live_env)
                accel_action = _heuristic_accel(
                    obs_before,
                    cfg,
                    str(args.accel_source),
                    centers=centers,
                    counts=counts,
                ).astype(np.float32, copy=False)
                sat_world = live_driver.run_accel_stage(accel_action)
                _refresh_stage_obs_cache(live_driver)
                obs_after_accel = _current_obs_list(live_driver)
                current_sat_mask = _heuristic_sat(obs_after_accel, cfg, str(args.current_sat_source)).astype(np.float32, copy=False)
                current_sat_action_ids = _sat_mask_to_ids(live_driver, current_sat_mask)

                sampled = (step % max(int(args.step_stride), 1)) == 0
                if sampled:
                    visible_per_uav = [list(v) for v in (live_driver._stage_visible or [[] for _ in range(cfg.num_uav)])]
                    swap_candidates = _enumerate_single_swap_candidates(
                        current_sat_action_ids=current_sat_action_ids,
                        visible_per_uav=visible_per_uav,
                        select_k=int(select_k),
                    )
                    if swap_candidates:
                        snapshot_state = live_driver.export_sat_stage_state()
                        analysis = _context_analysis(
                            probe_driver=probe_driver,
                            snapshot_state=snapshot_state,
                            current_sat_action_ids=current_sat_action_ids,
                            swap_candidates=swap_candidates,
                            horizons=horizons,
                            gamma=gamma,
                            cfg=cfg,
                            accel_source=str(args.accel_source),
                            followup_sat_source=str(args.followup_sat_source),
                            bw_source=str(args.bw_source),
                        )
                        row: dict[str, Any] = {
                            "context_id": int(contexts_done),
                            "episode": int(episode),
                            "seed": int(seed),
                            "step": int(step),
                            "swap_candidate_count": int(len(swap_candidates)),
                            "baseline_sat_action_ids": _sat_action_ids_to_text(current_sat_action_ids),
                            "baseline_path": json.dumps([float(x) for x in analysis["baseline_path"]], ensure_ascii=False),
                        }
                        h1 = int(min(horizons))
                        best_gap_h1 = float(analysis["horizons"][h1]["best_gap"])
                        for horizon in horizons:
                            h = int(horizon)
                            info = analysis["horizons"][h]
                            row[f"baseline_return_h{h}"] = float(info["baseline_return"])
                            row[f"best_return_h{h}"] = float(info["best_return"])
                            row[f"best_gap_h{h}"] = float(info["best_gap"])
                            row[f"best_swap_uav_h{h}"] = info["best_swap_uav"]
                            row[f"best_swap_drop_sat_id_h{h}"] = info["best_swap_drop_sat_id"]
                            row[f"best_swap_add_sat_id_h{h}"] = info["best_swap_add_sat_id"]
                            row[f"best_sat_action_ids_h{h}"] = "" if info["best_sat_action_ids"] is None else str(info["best_sat_action_ids"])
                            row[f"same_best_as_h{h1}_h{h}"] = info["same_best_as_h1"]
                            row[f"step0_abs_share_h{h}"] = info["step0_abs_share"]
                            row[f"delta_path_h{h}"] = json.dumps([float(x) for x in info["delta_path"]], ensure_ascii=False)
                            row[f"gu_queue_sum_delta_path_h{h}"] = json.dumps(info["gu_queue_sum_delta_path"], ensure_ascii=False)
                            row[f"uav_queue_sum_delta_path_h{h}"] = json.dumps(info["uav_queue_sum_delta_path"], ensure_ascii=False)
                            row[f"sat_queue_sum_delta_path_h{h}"] = json.dumps(info["sat_queue_sum_delta_path"], ensure_ascii=False)
                            row[f"total_queue_sum_delta_path_h{h}"] = json.dumps(info["total_queue_sum_delta_path"], ensure_ascii=False)
                            horizon_gap_values[h].append(float(info["best_gap"]))
                            if info["step0_abs_share"] is not None:
                                horizon_step0_share_values[h].append(float(info["step0_abs_share"]))
                            if h != h1:
                                if info["same_best_as_h1"] is not None:
                                    horizon_same_best_values[h].append(float(info["same_best_as_h1"]))
                                horizon_growth_vs_h1_values[h].append(float(info["best_gap"]) - best_gap_h1)
                        context_rows.append(row)
                        trace_record: dict[str, Any] = {
                            "context_id": int(contexts_done),
                            "episode": int(episode),
                            "seed": int(seed),
                            "step": int(step),
                            "baseline_sat_action_ids": _sat_action_ids_to_text(current_sat_action_ids),
                            "swap_candidate_count": int(len(swap_candidates)),
                            "horizons": {},
                        }
                        for horizon in horizons:
                            h = int(horizon)
                            info = analysis["horizons"][h]
                            trace_record["horizons"][str(h)] = {
                                "best_gap": float(info["best_gap"]),
                                "best_swap_uav": info["best_swap_uav"],
                                "best_swap_drop_sat_id": info["best_swap_drop_sat_id"],
                                "best_swap_add_sat_id": info["best_swap_add_sat_id"],
                                "best_sat_action_ids": info["best_sat_action_ids"],
                                "delta_path": [float(x) for x in info["delta_path"]],
                                "gu_queue_sum_delta_path": info["gu_queue_sum_delta_path"],
                                "uav_queue_sum_delta_path": info["uav_queue_sum_delta_path"],
                                "sat_queue_sum_delta_path": info["sat_queue_sum_delta_path"],
                                "total_queue_sum_delta_path": info["total_queue_sum_delta_path"],
                                "baseline_trace_steps": info["baseline_trace_steps"],
                                "best_trace_steps": info["best_trace_steps"],
                            }
                        context_traces.append(trace_record)
                        contexts_done += 1

                live_driver.run_sat_stage(current_sat_action_ids)
                _refresh_stage_obs_cache(live_driver)
                bw_obs = _current_obs_list(live_driver)
                bw_action = _heuristic_bw(bw_obs, cfg, str(args.bw_source)).astype(np.float32, copy=False)
                step_result, _next_world = live_driver.execute_stage_bw_and_prepare_next_accel(bw_action)
                done = bool(next(iter(step_result.terminations.values())) or next(iter(step_result.truncations.values())))
                step += 1

        summary = {
            "config": str(args.config),
            "episodes": int(args.episodes),
            "episode_seed_base": int(args.episode_seed-base if False else args.episode_seed_base),
            "step_stride": int(args.step_stride),
            "max_contexts": int(args.max_contexts),
            "contexts_analyzed": int(len(context_rows)),
            "discount_gamma": float(gamma),
            "sources": {
                "accel_source": str(args.accel_source),
                "current_sat_source": str(args.current_sat_source),
                "followup_sat_source": str(args.followup_sat_source),
                "bw_source": str(args.bw_source),
            },
            "swap_candidate_count_mean": _safe_mean([float(row["swap_candidate_count"]) for row in context_rows]),
            "horizons": {},
            "context_csv": str(out_dir / "context_rows.csv"),
            "context_trace_json": str(out_dir / "context_traces.json"),
            "summary_json": str(out_dir / "summary.json"),
        }
        h1 = int(min(horizons))
        for horizon in horizons:
            h = int(horizon)
            summary["horizons"][str(h)] = {
                "best_gap": _summarize(horizon_gap_values[h]),
                "step0_abs_share": _summarize(horizon_step0_share_values[h]),
            }
            if h != h1:
                summary["horizons"][str(h)]["same_best_as_h1_frac"] = _summarize(horizon_same_best_values[h])
                summary["horizons"][str(h)]["best_gap_growth_vs_h1"] = _summarize(horizon_growth_vs_h1_values[h])

        fieldnames: list[str] = []
        for row in context_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with (out_dir / "context_rows.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames or ["context_id"])
            writer.writeheader()
            for row in context_rows:
                writer.writerow(row)
        with (out_dir / "context_traces.json").open("w", encoding="utf-8") as f:
            json.dump(context_traces, f, ensure_ascii=False, indent=2)
        with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        for env in (live_env, probe_env):
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()


if __name__ == "__main__":
    main()
