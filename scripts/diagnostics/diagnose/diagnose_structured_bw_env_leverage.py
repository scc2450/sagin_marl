from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.baselines import queue_aware_bw_policy
from sagin_marl.rl.structured_bw_update_direction import collect_bw_snapshot_panel
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for part in str(value).replace(";", ",").split(","):
        text = part.strip()
        if text:
            out.append(int(text))
    return out


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    xa = np.asarray(x, dtype=np.float64).reshape(-1)
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    if xa.size <= 1 or ya.size <= 1 or xa.size != ya.size:
        return 0.0
    finite = np.isfinite(xa) & np.isfinite(ya)
    if int(np.sum(finite)) <= 1:
        return 0.0
    xa = xa[finite]
    ya = ya[finite]
    if float(np.std(xa)) <= 1.0e-12 or float(np.std(ya)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def _safe_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        return 0.0
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return 0.0
    return float(np.mean(finite))


def _safe_abs_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        return 0.0
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return 0.0
    return float(np.mean(np.abs(finite)))


def _safe_std(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 1:
        return 0.0
    finite = arr[np.isfinite(arr)]
    if finite.size <= 1:
        return 0.0
    return float(np.std(finite))


def _safe_frac(mask: list[bool] | np.ndarray) -> float:
    arr = np.asarray(mask, dtype=bool).reshape(-1)
    if arr.size <= 0:
        return 0.0
    return float(np.mean(arr.astype(np.float64)))


def _safe_sign_agree(x: np.ndarray, y: np.ndarray) -> float:
    xa = np.asarray(x, dtype=np.float64).reshape(-1)
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    if xa.size <= 0 or ya.size <= 0 or xa.size != ya.size:
        return 0.0
    finite = np.isfinite(xa) & np.isfinite(ya) & (np.abs(xa) > 1.0e-12) & (np.abs(ya) > 1.0e-12)
    if int(np.sum(finite)) <= 0:
        return 0.0
    return float(np.mean((xa[finite] * ya[finite]) > 0.0))


def _zero_sat_action(cfg) -> np.ndarray:
    select_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 1)
    return np.full((cfg.num_uav, select_k), -1, dtype=np.int64)


def _heuristic_bw_action(driver: StructuredControlDriver, cfg) -> np.ndarray:
    env = driver.env
    obs = {agent: env._get_obs(idx) for idx, agent in enumerate(env.agents)}
    return np.asarray(queue_aware_bw_policy(list(obs.values()), cfg), dtype=np.float32)


def _uniform_valid_action(valid_mask: np.ndarray, width: int) -> np.ndarray:
    out = np.zeros((1, width), dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if int(np.sum(valid)) > 0:
        out[0, valid] = 1.0 / float(np.sum(valid))
    return out


def _rollout_from_snapshot_with_heuristic_tail(
    *,
    snapshot_state: dict[str, Any],
    first_action: np.ndarray,
    cfg,
    k_steps: int,
    gamma: float,
) -> dict[str, float]:
    env = make_structured_env(cfg, mode="script")
    driver = as_structured_driver(env)
    try:
        driver.load_bw_stage_state(snapshot_state)
        action = np.asarray(first_action, dtype=np.float32)
        discounted_reward = 0.0
        discounted_weighted = 0.0
        discount = 1.0
        for step in range(int(k_steps)):
            step_result, _ = driver.execute_stage_bw_and_prepare_next_accel(action)
            reward = float(next(iter(step_result.rewards.values())))
            weighted = float(getattr(step_result, "bw_weighted_workload_delta_reward", 0.0) or 0.0)
            discounted_reward += discount * reward
            discounted_weighted += discount * weighted
            done = bool(list(step_result.terminations.values())[0] or list(step_result.truncations.values())[0])
            if done or step == int(k_steps) - 1:
                break
            driver.begin_step()
            accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
            _ = driver.run_accel_stage(accel_zero)
            _ = driver.run_sat_stage(_zero_sat_action(cfg))
            action = _heuristic_bw_action(driver, cfg)
            discount *= float(gamma)
        return {"reward": float(discounted_reward), "weighted": float(discounted_weighted)}
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def _extract_slot_feature_arrays(entry: dict[str, Any], cfg: Any) -> dict[str, np.ndarray]:
    snapshot = entry["snapshot"]
    valid_mask = np.asarray(snapshot.bw_valid_mask[0], dtype=bool)
    candidate_indices = np.asarray(snapshot.candidate_indices[0], dtype=np.int64)
    world_state = snapshot.world_state
    gu_nodes = np.asarray(world_state.gu_nodes[0], dtype=np.float32)
    uav_gu_edges = np.asarray(world_state.uav_gu_edges[0, 0], dtype=np.float32)
    slot_queue: list[float] = []
    slot_eta: list[float] = []
    slot_arrival_rate: list[float] = []
    slot_recent_arrival: list[float] = []
    slot_recent_service: list[float] = []
    slot_headroom: list[float] = []
    slot_urgency_risk: list[float] = []
    slot_downstream_pressure: list[float] = []
    slot_service_gap: list[float] = []
    slot_service_gap_risk: list[float] = []
    slot_deadline_slack: list[float] = []
    slot_deadline_risk: list[float] = []
    feat_base = 3
    for slot, gu_idx in enumerate(candidate_indices.tolist()):
        if not bool(valid_mask[slot]) or int(gu_idx) < 0:
            continue
        gu_idx_i = int(gu_idx)
        slot_queue.append(float(gu_nodes[gu_idx_i, 2]))
        slot_eta.append(float(uav_gu_edges[gu_idx_i, 6]))
        feat_col = feat_base
        if bool(getattr(cfg, "obs_user_include_arrival_rate", False)):
            slot_arrival_rate.append(float(gu_nodes[gu_idx_i, feat_col]))
            feat_col += 1
        if bool(getattr(cfg, "obs_user_include_recent_arrival", False)):
            slot_recent_arrival.append(float(gu_nodes[gu_idx_i, feat_col]))
            feat_col += 1
        if bool(getattr(cfg, "obs_user_include_recent_service", False)):
            slot_recent_service.append(float(gu_nodes[gu_idx_i, feat_col]))
            feat_col += 1
        if bool(getattr(cfg, "obs_user_include_queue_headroom", False)):
            slot_headroom.append(float(gu_nodes[gu_idx_i, feat_col]))
            feat_col += 1
        if bool(getattr(cfg, "obs_user_include_urgency_risk", False)):
            slot_urgency_risk.append(float(gu_nodes[gu_idx_i, feat_col]))
            feat_col += 1
        if bool(getattr(cfg, "obs_user_include_downstream_pressure", False)):
            slot_downstream_pressure.append(float(gu_nodes[gu_idx_i, feat_col]))
            feat_col += 1
        if bool(getattr(cfg, "obs_user_include_service_gap", False)):
            slot_service_gap.append(float(gu_nodes[gu_idx_i, feat_col]))
            feat_col += 1
        if bool(getattr(cfg, "obs_user_include_service_gap_risk", False)):
            slot_service_gap_risk.append(float(gu_nodes[gu_idx_i, feat_col]))
            feat_col += 1
        if bool(getattr(cfg, "obs_user_include_deadline_slack", False)):
            slot_deadline_slack.append(float(gu_nodes[gu_idx_i, feat_col]))
            feat_col += 1
        if bool(getattr(cfg, "obs_user_include_deadline_risk", False)):
            slot_deadline_risk.append(float(gu_nodes[gu_idx_i, feat_col]))
            feat_col += 1
    return {
        "queue": np.asarray(slot_queue, dtype=np.float64),
        "eta": np.asarray(slot_eta, dtype=np.float64),
        "arrival_rate": np.asarray(slot_arrival_rate, dtype=np.float64),
        "recent_arrival": np.asarray(slot_recent_arrival, dtype=np.float64),
        "recent_service": np.asarray(slot_recent_service, dtype=np.float64),
        "queue_headroom": np.asarray(slot_headroom, dtype=np.float64),
        "urgency_risk": np.asarray(slot_urgency_risk, dtype=np.float64),
        "downstream_pressure": np.asarray(slot_downstream_pressure, dtype=np.float64),
        "service_gap": np.asarray(slot_service_gap, dtype=np.float64),
        "service_gap_risk": np.asarray(slot_service_gap_risk, dtype=np.float64),
        "deadline_slack": np.asarray(slot_deadline_slack, dtype=np.float64),
        "deadline_risk": np.asarray(slot_deadline_risk, dtype=np.float64),
    }


def _spread(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 1:
        return 0.0
    return float(np.max(arr) - np.min(arr))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _analyze_config(
    *,
    config_path: Path,
    horizons: list[int],
    shift_delta: float,
    episodes: int,
    states: int,
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    cfg = load_config(str(config_path))
    panel = collect_bw_snapshot_panel(cfg, episodes=int(episodes), states=int(states), seed=int(seed))
    if not panel:
        raise RuntimeError(f"No BW snapshots collected for {config_path}.")

    horizon_metrics: dict[int, dict[str, Any]] = {int(h): {"state_gap": [], "mean_abs_delta": [], "best_delta": [], "worst_delta": [], "heuristic_minus_uniform": [], "heuristic_beats_uniform": []} for h in horizons}
    perturb_by_horizon: dict[int, list[float]] = {int(h): [] for h in horizons}
    state_rows: list[dict[str, Any]] = []
    perturb_rows: list[dict[str, Any]] = []
    feature_spreads: dict[str, list[float]] = {
        "queue_spread": [],
        "eta_spread": [],
        "arrival_rate_spread": [],
        "recent_arrival_spread": [],
        "recent_service_spread": [],
        "queue_headroom_spread": [],
        "urgency_risk_spread": [],
        "downstream_pressure_spread": [],
        "service_gap_spread": [],
        "service_gap_risk_spread": [],
        "deadline_slack_spread": [],
        "deadline_risk_spread": [],
        "heuristic_weight_spread": [],
    }

    for state_idx, entry in enumerate(panel):
        snapshot = entry["snapshot"]
        snapshot_state = entry["snapshot_state"]
        heuristic_action = np.asarray(entry["heuristic_action"], dtype=np.float32)
        valid_mask = np.asarray(snapshot.bw_valid_mask[0], dtype=bool)
        candidate_indices = np.asarray(snapshot.candidate_indices[0], dtype=np.int64)
        valid_slots = [int(idx) for idx in np.flatnonzero(valid_mask).tolist()]
        if len(valid_slots) <= 1:
            continue

        slot_features = _extract_slot_feature_arrays(entry, cfg)
        feature_spreads["queue_spread"].append(_spread(slot_features["queue"]))
        feature_spreads["eta_spread"].append(_spread(slot_features["eta"]))
        feature_spreads["arrival_rate_spread"].append(_spread(slot_features["arrival_rate"]))
        feature_spreads["recent_arrival_spread"].append(_spread(slot_features["recent_arrival"]))
        feature_spreads["recent_service_spread"].append(_spread(slot_features["recent_service"]))
        feature_spreads["queue_headroom_spread"].append(_spread(slot_features["queue_headroom"]))
        feature_spreads["urgency_risk_spread"].append(_spread(slot_features["urgency_risk"]))
        feature_spreads["downstream_pressure_spread"].append(_spread(slot_features["downstream_pressure"]))
        feature_spreads["service_gap_spread"].append(_spread(slot_features["service_gap"]))
        feature_spreads["service_gap_risk_spread"].append(_spread(slot_features["service_gap_risk"]))
        feature_spreads["deadline_slack_spread"].append(_spread(slot_features["deadline_slack"]))
        feature_spreads["deadline_risk_spread"].append(_spread(slot_features["deadline_risk"]))
        feature_spreads["heuristic_weight_spread"].append(_spread(heuristic_action[0, valid_mask]))

        uniform_action = _uniform_valid_action(valid_mask, heuristic_action.shape[-1])
        per_horizon_base: dict[int, float] = {}
        per_horizon_uniform: dict[int, float] = {}
        per_horizon_target: dict[int, list[tuple[int, int, float, float]]] = {int(h): [] for h in horizons}

        for horizon in horizons:
            base_roll = _rollout_from_snapshot_with_heuristic_tail(
                snapshot_state=snapshot_state,
                first_action=heuristic_action,
                cfg=cfg,
                k_steps=int(horizon),
                gamma=float(cfg.gamma),
            )
            uniform_roll = _rollout_from_snapshot_with_heuristic_tail(
                snapshot_state=snapshot_state,
                first_action=uniform_action,
                cfg=cfg,
                k_steps=int(horizon),
                gamma=float(cfg.gamma),
            )
            per_horizon_base[int(horizon)] = float(base_roll["reward"])
            per_horizon_uniform[int(horizon)] = float(uniform_roll["reward"])
            for target_slot in valid_slots:
                realloc, used_delta = StructuredControlDriver._reallocate_bw_toward_slot(
                    heuristic_action[0],
                    valid_mask,
                    int(target_slot),
                    float(shift_delta),
                )
                if realloc is None or float(used_delta) <= 0.0:
                    continue
                action = heuristic_action.copy()
                action[0] = realloc
                roll = _rollout_from_snapshot_with_heuristic_tail(
                    snapshot_state=snapshot_state,
                    first_action=action,
                    cfg=cfg,
                    k_steps=int(horizon),
                    gamma=float(cfg.gamma),
                )
                reward = float(roll["reward"])
                delta = float(reward - per_horizon_base[int(horizon)])
                per_horizon_target[int(horizon)].append((int(target_slot), int(candidate_indices[target_slot]), reward, delta))
                perturb_by_horizon[int(horizon)].append(delta)
                perturb_rows.append(
                    {
                        "config_name": config_path.stem,
                        "state_index": int(state_idx),
                        "episode": int(entry["episode"]),
                        "t": int(entry["t"]),
                        "horizon": int(horizon),
                        "target_slot": int(target_slot),
                        "target_gu": int(candidate_indices[target_slot]),
                        "reward": float(reward),
                        "delta_vs_heuristic": float(delta),
                        "used_delta": float(used_delta),
                    }
                )

        state_row: dict[str, Any] = {
            "config_name": config_path.stem,
            "state_index": int(state_idx),
            "episode": int(entry["episode"]),
            "t": int(entry["t"]),
            "valid_count": int(len(valid_slots)),
            "queue_spread": float(feature_spreads["queue_spread"][-1]),
            "eta_spread": float(feature_spreads["eta_spread"][-1]),
            "arrival_rate_spread": float(feature_spreads["arrival_rate_spread"][-1]) if feature_spreads["arrival_rate_spread"] else 0.0,
            "recent_arrival_spread": float(feature_spreads["recent_arrival_spread"][-1]) if feature_spreads["recent_arrival_spread"] else 0.0,
            "recent_service_spread": float(feature_spreads["recent_service_spread"][-1]) if feature_spreads["recent_service_spread"] else 0.0,
            "queue_headroom_spread": float(feature_spreads["queue_headroom_spread"][-1]) if feature_spreads["queue_headroom_spread"] else 0.0,
            "urgency_risk_spread": float(feature_spreads["urgency_risk_spread"][-1]) if feature_spreads["urgency_risk_spread"] else 0.0,
            "downstream_pressure_spread": float(feature_spreads["downstream_pressure_spread"][-1]) if feature_spreads["downstream_pressure_spread"] else 0.0,
            "service_gap_spread": float(feature_spreads["service_gap_spread"][-1]) if feature_spreads["service_gap_spread"] else 0.0,
            "service_gap_risk_spread": float(feature_spreads["service_gap_risk_spread"][-1]) if feature_spreads["service_gap_risk_spread"] else 0.0,
            "deadline_slack_spread": float(feature_spreads["deadline_slack_spread"][-1]) if feature_spreads["deadline_slack_spread"] else 0.0,
            "deadline_risk_spread": float(feature_spreads["deadline_risk_spread"][-1]) if feature_spreads["deadline_risk_spread"] else 0.0,
            "heuristic_weight_spread": float(feature_spreads["heuristic_weight_spread"][-1]),
        }
        for horizon in horizons:
            target_rows = per_horizon_target[int(horizon)]
            target_rewards = [float(row[2]) for row in target_rows]
            target_deltas = [float(row[3]) for row in target_rows]
            base_reward = float(per_horizon_base[int(horizon)])
            uniform_reward = float(per_horizon_uniform[int(horizon)])
            gap = (max(target_rewards) - min(target_rewards)) if len(target_rewards) >= 2 else 0.0
            best_delta = max(target_deltas) if target_deltas else 0.0
            worst_delta = min(target_deltas) if target_deltas else 0.0
            mean_abs_delta = _safe_abs_mean(target_deltas)
            horizon_metrics[int(horizon)]["state_gap"].append(float(gap))
            horizon_metrics[int(horizon)]["mean_abs_delta"].append(float(mean_abs_delta))
            horizon_metrics[int(horizon)]["best_delta"].append(float(best_delta))
            horizon_metrics[int(horizon)]["worst_delta"].append(float(worst_delta))
            horizon_metrics[int(horizon)]["heuristic_minus_uniform"].append(float(base_reward - uniform_reward))
            horizon_metrics[int(horizon)]["heuristic_beats_uniform"].append(bool(base_reward > uniform_reward + 1.0e-9))
            state_row[f"h{int(horizon)}_heuristic_reward"] = base_reward
            state_row[f"h{int(horizon)}_uniform_reward"] = uniform_reward
            state_row[f"h{int(horizon)}_slot_gap"] = float(gap)
            state_row[f"h{int(horizon)}_mean_abs_delta"] = float(mean_abs_delta)
            state_row[f"h{int(horizon)}_best_delta"] = float(best_delta)
            state_row[f"h{int(horizon)}_worst_delta"] = float(worst_delta)
        state_rows.append(state_row)

    summary: dict[str, Any] = {
        "config_name": config_path.stem,
        "config_path": str(config_path),
        "states_requested": int(states),
        "states_collected": int(len(panel)),
        "states_analyzed": int(len(state_rows)),
        "episodes": int(episodes),
        "seed": int(seed),
        "shift_delta": float(shift_delta),
        "horizons": [int(h) for h in horizons],
        "feature_spreads": {name: {"mean": _safe_mean(values), "std": _safe_std(values)} for name, values in feature_spreads.items()},
        "horizon_metrics": {},
    }

    for horizon in horizons:
        metrics = horizon_metrics[int(horizon)]
        summary["horizon_metrics"][f"h{int(horizon)}"] = {
            "slot_gap_mean": _safe_mean(metrics["state_gap"]),
            "slot_gap_std": _safe_std(metrics["state_gap"]),
            "slot_gap_p_positive": _safe_frac(np.asarray(metrics["state_gap"], dtype=np.float64) > 1.0e-9),
            "mean_abs_delta_vs_heuristic": _safe_mean(metrics["mean_abs_delta"]),
            "best_delta_mean": _safe_mean(metrics["best_delta"]),
            "worst_delta_mean": _safe_mean(metrics["worst_delta"]),
            "heuristic_minus_uniform_mean": _safe_mean(metrics["heuristic_minus_uniform"]),
            "heuristic_beats_uniform_frac": _safe_frac(metrics["heuristic_beats_uniform"]),
            "perturb_delta_abs_mean": _safe_abs_mean(perturb_by_horizon[int(horizon)]),
            "perturb_delta_std": _safe_std(perturb_by_horizon[int(horizon)]),
        }

    if len(horizons) >= 2:
        for idx in range(len(horizons) - 1):
            h0 = int(horizons[idx])
            h1 = int(horizons[idx + 1])
            rows0 = [row for row in perturb_rows if int(row["horizon"]) == h0]
            rows1 = [row for row in perturb_rows if int(row["horizon"]) == h1]
            keyed0 = {(int(row["state_index"]), int(row["target_slot"])): float(row["delta_vs_heuristic"]) for row in rows0}
            keyed1 = {(int(row["state_index"]), int(row["target_slot"])): float(row["delta_vs_heuristic"]) for row in rows1}
            common_keys = sorted(set(keyed0).intersection(keyed1))
            x = np.asarray([keyed0[key] for key in common_keys], dtype=np.float64)
            y = np.asarray([keyed1[key] for key in common_keys], dtype=np.float64)
            summary[f"persistence_h{h0}_to_h{h1}"] = {
                "common_pairs": int(len(common_keys)),
                "corr": _safe_corr(x, y),
                "sign_agree": _safe_sign_agree(x, y),
                "abs_mean_h0": _safe_abs_mean(x),
                "abs_mean_h1": _safe_abs_mean(y),
            }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "state_metrics.csv", state_rows)
    _write_csv(out_dir / "perturb_metrics.csv", perturb_rows)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose BW leverage strength from snapshot perturbation rollouts.")
    parser.add_argument("--configs", type=str, required=True, help="Comma-separated config paths.")
    parser.add_argument("--states", type=int, default=96)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--horizons", type=str, default="2,5,10")
    parser.add_argument("--shift_delta", type=float, default=0.20)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    _set_all_seeds(int(args.seed))
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    config_paths = [Path(part.strip()) for part in str(args.configs).split(",") if part.strip()]
    horizons = _parse_int_list(args.horizons)
    combined_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for config_path in config_paths:
        cfg_out_dir = out_root / config_path.stem
        summary = _analyze_config(
            config_path=config_path,
            horizons=horizons,
            shift_delta=float(args.shift_delta),
            episodes=int(args.episodes),
            states=int(args.states),
            seed=int(args.seed),
            out_dir=cfg_out_dir,
        )
        summaries.append(summary)
        row: dict[str, Any] = {
            "config_name": str(summary["config_name"]),
            "states_analyzed": int(summary["states_analyzed"]),
        }
        for horizon_key, metrics in summary["horizon_metrics"].items():
            row[f"{horizon_key}_slot_gap_mean"] = float(metrics["slot_gap_mean"])
            row[f"{horizon_key}_mean_abs_delta_vs_heuristic"] = float(metrics["mean_abs_delta_vs_heuristic"])
            row[f"{horizon_key}_heuristic_minus_uniform_mean"] = float(metrics["heuristic_minus_uniform_mean"])
            row[f"{horizon_key}_heuristic_beats_uniform_frac"] = float(metrics["heuristic_beats_uniform_frac"])
        for key, value in summary.items():
            if key.startswith("persistence_h") and isinstance(value, dict):
                row[f"{key}_corr"] = float(value.get("corr", 0.0))
                row[f"{key}_sign_agree"] = float(value.get("sign_agree", 0.0))
        combined_rows.append(row)

    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"runs": summaries}, f, indent=2, ensure_ascii=False)
    _write_csv(out_root / "summary.csv", combined_rows)


if __name__ == "__main__":
    main()
