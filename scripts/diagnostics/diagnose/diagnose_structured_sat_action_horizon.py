from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
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
from sagin_marl.rl.baselines import _sat_heuristic_score
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
    parser.add_argument("--topm-per-uav", type=int, default=3)
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


def _pair_indices_to_text(pair_indices: list[int]) -> str:
    return "|".join(str(int(x)) for x in pair_indices)


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


def _subset_scores_for_local_state(
    *,
    sats_obs: np.ndarray,
    sat_valid_mask: np.ndarray,
    local_state,
    cfg,
) -> list[tuple[int, float]]:
    sat_scores = _sat_heuristic_score(np.asarray(sats_obs, dtype=np.float32), np.asarray(sat_valid_mask, dtype=bool), cfg)
    subset_mask = np.asarray(local_state.subset_mask[0].detach().cpu().numpy() > 0.5, dtype=bool)
    subset_members = np.asarray(local_state.subset_members[0].detach().cpu().numpy(), dtype=np.int64)
    ranked: list[tuple[int, float]] = []
    for subset_idx in np.flatnonzero(subset_mask).tolist():
        members = [int(x) for x in subset_members[int(subset_idx)].tolist() if int(x) >= 0]
        score = float(np.sum(sat_scores[np.asarray(members, dtype=np.int64)])) if members else -1.0e9
        ranked.append((int(subset_idx), score))
    ranked.sort(key=lambda item: float(item[1]), reverse=True)
    return ranked


def _topm_joint_pair_indices(
    *,
    obs_after_accel: list[dict[str, np.ndarray]],
    local_states: list[Any],
    cfg,
    topm_per_uav: int,
) -> list[list[int]]:
    per_uav: list[list[int]] = []
    for u, (obs_u, local_state) in enumerate(zip(obs_after_accel, local_states)):
        del u
        sats_obs = np.asarray(obs_u["sats"], dtype=np.float32)
        sat_valid_mask = np.asarray(obs_u.get("sat_valid_mask", obs_u["sats_mask"]) > 0.0, dtype=bool)
        ranked = _subset_scores_for_local_state(
            sats_obs=sats_obs,
            sat_valid_mask=sat_valid_mask,
            local_state=local_state,
            cfg=cfg,
        )
        chosen = [int(subset_idx) for subset_idx, _score in ranked[: max(int(topm_per_uav), 1)]]
        if not chosen:
            chosen = [-1]
        per_uav.append(chosen)
    return [[int(x) for x in combo] for combo in itertools.product(*per_uav)]


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
) -> list[float]:
    probe_driver.load_sat_stage_state(snapshot_state)
    step_rewards: list[float] = []
    centers, counts = _cluster_centers_and_counts(probe_driver.env)

    probe_driver.run_sat_stage(np.asarray(first_sat_action_ids, dtype=np.int64))
    _refresh_stage_obs_cache(probe_driver)
    bw_obs = _current_obs_list(probe_driver)
    bw_action = _heuristic_bw(bw_obs, cfg, bw_source).astype(np.float32, copy=False)
    step_result, _next_world = probe_driver.execute_stage_bw_and_prepare_next_accel(bw_action)
    step_rewards.append(float(getattr(step_result, "bw_weighted_workload_level_reward", next(iter(step_result.rewards.values())))))
    done = bool(next(iter(step_result.terminations.values())) or next(iter(step_result.truncations.values())))
    if done or int(max_horizon) <= 1:
        return step_rewards

    for _step in range(1, int(max_horizon)):
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
        step_result, _next_world = probe_driver.execute_stage_bw_and_prepare_next_accel(bw_action)
        step_rewards.append(float(getattr(step_result, "bw_weighted_workload_level_reward", next(iter(step_result.rewards.values())))))
        done = bool(next(iter(step_result.terminations.values())) or next(iter(step_result.truncations.values())))
        if done:
            break
    return step_rewards


def _context_analysis(
    *,
    probe_driver: StructuredControlDriver,
    snapshot_state: dict[str, Any],
    current_sat_action_ids: np.ndarray,
    candidate_pair_indices: list[list[int]],
    horizons: list[int],
    gamma: float,
    cfg,
    accel_source: str,
    followup_sat_source: str,
    bw_source: str,
) -> dict[str, Any]:
    max_horizon = max(int(h) for h in horizons)
    baseline_path = _rollout_single_sat_action_path(
        probe_driver=probe_driver,
        snapshot_state=snapshot_state,
        first_sat_action_ids=current_sat_action_ids,
        max_horizon=max_horizon,
        cfg=cfg,
        accel_source=accel_source,
        followup_sat_source=followup_sat_source,
        bw_source=bw_source,
    )
    baseline_returns = _prefix_discounted_returns(baseline_path, horizons, gamma)

    candidate_paths: list[list[float]] = []
    candidate_returns: list[dict[int, float]] = []
    candidate_actions: list[np.ndarray] = []
    for pair_indices in candidate_pair_indices:
        probe_driver.load_sat_stage_state(snapshot_state)
        sat_action_ids = probe_driver.decode_sat_pair_actions((), list(int(idx) for idx in pair_indices))
        candidate_actions.append(np.asarray(sat_action_ids, dtype=np.int64))
        path = _rollout_single_sat_action_path(
            probe_driver=probe_driver,
            snapshot_state=snapshot_state,
            first_sat_action_ids=sat_action_ids,
            max_horizon=max_horizon,
            cfg=cfg,
            accel_source=accel_source,
            followup_sat_source=followup_sat_source,
            bw_source=bw_source,
        )
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
        step_delta = [
            float((best_path[t] if t < len(best_path) else 0.0) - (baseline_path[t] if t < len(baseline_path) else 0.0))
            for t in range(int(horizon))
        ]
        horizon_data[int(horizon)] = {
            "baseline_return": float(baseline_returns[int(horizon)]),
            "best_return": best_return,
            "best_gap": float(best_return - baseline_returns[int(horizon)]),
            "best_idx": int(best_idx),
            "same_best_as_h1": None if best_idx_h1 is None or best_idx < 0 else int(best_idx == best_idx_h1),
            "step0_abs_share": _path_abs_step0_share(step_delta, int(horizon)),
            "best_pair_indices": (
                None
                if best_idx < 0
                else [int(x) for x in candidate_pair_indices[best_idx]]
            ),
            "best_sat_action_ids": (
                None
                if best_idx < 0
                else _sat_action_ids_to_text(candidate_actions[best_idx])
            ),
            "delta_path": [float(x) for x in step_delta],
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
    live_env = make_structured_env(cfg, mode="script")
    probe_env = make_structured_env(cfg, mode="script")
    live_driver = as_structured_driver(live_env)
    probe_driver = as_structured_driver(probe_env)
    context_rows: list[dict[str, Any]] = []
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
                local_states = live_driver.build_sat_pair_candidates(sat_world)
                current_sat_mask = _heuristic_sat(obs_after_accel, cfg, str(args.current_sat_source)).astype(np.float32, copy=False)
                current_sat_action_ids = _sat_mask_to_ids(live_driver, current_sat_mask)

                sampled = (step % max(int(args.step_stride), 1)) == 0
                if sampled:
                    snapshot_state = live_driver.export_sat_stage_state()
                    candidate_pair_indices = _topm_joint_pair_indices(
                        obs_after_accel=obs_after_accel,
                        local_states=local_states,
                        cfg=cfg,
                        topm_per_uav=int(args.topm_per_uav),
                    )
                    analysis = _context_analysis(
                        probe_driver=probe_driver,
                        snapshot_state=snapshot_state,
                        current_sat_action_ids=current_sat_action_ids,
                        candidate_pair_indices=candidate_pair_indices,
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
                        "panel_size": int(len(candidate_pair_indices)),
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
                        row[f"best_pair_indices_h{h}"] = (
                            "" if info["best_pair_indices"] is None else _pair_indices_to_text(list(info["best_pair_indices"]))
                        )
                        row[f"best_sat_action_ids_h{h}"] = "" if info["best_sat_action_ids"] is None else str(info["best_sat_action_ids"])
                        row[f"same_best_as_h{h1}_h{h}"] = info["same_best_as_h1"]
                        row[f"step0_abs_share_h{h}"] = info["step0_abs_share"]
                        row[f"delta_path_h{h}"] = json.dumps([float(x) for x in info["delta_path"]], ensure_ascii=False)
                        horizon_gap_values[h].append(float(info["best_gap"]))
                        if info["step0_abs_share"] is not None:
                            horizon_step0_share_values[h].append(float(info["step0_abs_share"]))
                        if h != h1:
                            if info["same_best_as_h1"] is not None:
                                horizon_same_best_values[h].append(float(info["same_best_as_h1"]))
                            horizon_growth_vs_h1_values[h].append(float(info["best_gap"]) - best_gap_h1)
                    context_rows.append(row)
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
            "episode_seed_base": int(args.episode_seed_base),
            "step_stride": int(args.step_stride),
            "max_contexts": int(args.max_contexts),
            "contexts_analyzed": int(len(context_rows)),
            "topm_per_uav": int(args.topm_per_uav),
            "candidate_panel_size_mean": _safe_mean([float(row["panel_size"]) for row in context_rows]),
            "discount_gamma": float(gamma),
            "sources": {
                "accel_source": str(args.accel_source),
                "current_sat_source": str(args.current_sat_source),
                "followup_sat_source": str(args.followup_sat_source),
                "bw_source": str(args.bw_source),
            },
            "horizons": {},
            "context_csv": str(out_dir / "context_rows.csv"),
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
