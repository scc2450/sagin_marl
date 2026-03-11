from __future__ import annotations

import argparse
import copy
import csv
import itertools
import os
import sys
from pathlib import Path
from typing import Dict

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from sagin_marl.env.config import load_config, update_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.rl.action_assembler import assemble_actions
from sagin_marl.rl.baselines import (
    centroid_accel_policy,
    queue_aware_policy,
    random_accel_policy,
    zero_accel_policy,
)
from sagin_marl.utils.progress import Progress

LAYERS = ("gu", "uav", "sat")


def _baseline_actions(
    baseline: str,
    obs_list,
    cfg,
    num_agents: int,
    rng: np.random.Generator | None = None,
):
    if baseline in ("fixed", "zero_accel"):
        return zero_accel_policy(num_agents), None, None
    if baseline == "random_accel":
        return random_accel_policy(num_agents, rng=rng), None, None
    if baseline == "centroid":
        gain = float(getattr(cfg, "baseline_centroid_gain", 2.0) or 2.0)
        queue_weighted = bool(getattr(cfg, "baseline_centroid_queue_weighted", True))
        return centroid_accel_policy(obs_list, gain=gain, queue_weighted=queue_weighted), None, None
    if baseline == "queue_aware":
        return queue_aware_policy(obs_list, cfg)
    raise ValueError(f"Unsupported baseline for queue diagnosis: {baseline}")


def _default_output_stem(config_path: str, baseline: str, episodes: int) -> str:
    config_stem = Path(config_path).stem
    out_dir = Path("runs") / "queue_diag"
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{config_stem}_{baseline}_n{episodes}")


def _cap_hit_step_ratio(queue: np.ndarray, queue_max: float, cap_frac: float) -> float:
    if queue.size == 0:
        return 0.0
    thresh = float(queue_max) * float(cap_frac)
    return float(np.mean(queue >= thresh))


def _cap_hit_any(queue: np.ndarray, queue_max: float, cap_frac: float) -> float:
    if queue.size == 0:
        return 0.0
    thresh = float(queue_max) * float(cap_frac)
    return 1.0 if bool(np.any(queue >= thresh)) else 0.0


def _nonzero_entity_frac(queue: np.ndarray, q_zero_eps: float) -> float:
    if queue.size == 0:
        return 0.0
    return float(np.mean(queue > q_zero_eps))


def _nonzero_entity_mean(queue: np.ndarray, q_zero_eps: float) -> float:
    if queue.size == 0:
        return 0.0
    mask = queue > q_zero_eps
    if not np.any(mask):
        return 0.0
    return float(np.mean(queue[mask]))


def _safe_percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _parse_overrides(items: list[str]) -> tuple[Dict[str, str], list[str]]:
    overrides: Dict[str, str] = {}
    labels: list[str] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected key=value.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid override '{item}'. Empty key.")
        overrides[key] = value
        labels.append(f"{key}={value}")
    return overrides, labels


def _parse_grid_specs(items: list[str]) -> list[tuple[str, list[str]]]:
    specs: list[tuple[str, list[str]]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid grid spec '{item}'. Expected key=v1,v2,...")
        key, raw_values = item.split("=", 1)
        key = key.strip()
        values = [value.strip() for value in raw_values.split(",") if value.strip()]
        if not key or not values:
            raise ValueError(f"Invalid grid spec '{item}'.")
        specs.append((key, values))
    return specs


def _episode_regime_row(
    episode: int,
    steps: int,
    collision: float,
    first_zero_step_q_active: float,
    fraction_steps_q_active_zero: float,
    fraction_steps_all_layers_nonzero: float,
    layer_queue_mean: Dict[str, float],
    layer_queue_sum_mean: Dict[str, float],
    layer_queue_p95: Dict[str, float],
    layer_queue_sum_p95: Dict[str, float],
    layer_first_zero_step: Dict[str, float],
    layer_fraction_steps_zero: Dict[str, float],
    layer_nonzero_entity_frac: Dict[str, float],
    layer_nonzero_mean: Dict[str, float],
    layer_cap_hit_ratio: Dict[str, float],
    layer_cap_hit_any_ratio: Dict[str, float],
    arrival_sum: float,
    gu_outflow_sum: float,
    sat_incoming_sum: float,
    sat_processed_sum: float,
    gu_drop_sum: float,
    uav_drop_sum: float,
    active_drop_sum: float,
    sat_drop_sum: float,
    layer_queue_start_sum: Dict[str, float],
    layer_queue_end_sum: Dict[str, float],
    layer_total_capacity: Dict[str, float],
) -> Dict[str, float]:
    steps_f = float(max(steps, 1))
    arrival_per_step = arrival_sum / steps_f
    arrival_denom = max(arrival_per_step, 1e-9)
    gu_outflow_per_step = gu_outflow_sum / steps_f
    sat_incoming_per_step = sat_incoming_sum / steps_f
    sat_processed_per_step = sat_processed_sum / steps_f
    outflow_arrival_ratio = gu_outflow_sum / max(arrival_sum, 1e-9)
    sat_incoming_arrival_ratio = sat_incoming_sum / max(arrival_sum, 1e-9)
    drop_sum = active_drop_sum + sat_drop_sum
    gu_drop_ratio = gu_drop_sum / max(arrival_sum, 1e-9)
    uav_drop_ratio = uav_drop_sum / max(arrival_sum, 1e-9)
    active_drop_ratio = active_drop_sum / max(arrival_sum, 1e-9)
    sat_drop_ratio = sat_drop_sum / max(arrival_sum, 1e-9)
    drop_ratio = drop_sum / max(arrival_sum, 1e-9)
    active_net_drift_sum = arrival_sum - sat_incoming_sum - active_drop_sum
    sat_net_drift_sum = sat_incoming_sum - sat_processed_sum - sat_drop_sum
    total_net_drift_sum = arrival_sum - sat_processed_sum - drop_sum
    queue_total_start_sum = sum(layer_queue_start_sum.values())
    queue_total_end_sum = sum(layer_queue_end_sum.values())
    row: Dict[str, float] = {
        "episode": float(episode),
        "steps": steps_f,
        "collision": float(collision),
        "first_zero_step_q_active": float(first_zero_step_q_active),
        "fraction_steps_q_active_zero": float(fraction_steps_q_active_zero),
        "fraction_steps_all_layers_nonzero": float(fraction_steps_all_layers_nonzero),
        "arrival_sum": float(arrival_sum),
        "gu_outflow_sum": float(gu_outflow_sum),
        "sat_incoming_sum": float(sat_incoming_sum),
        "sat_processed_sum": float(sat_processed_sum),
        "gu_drop_sum": float(gu_drop_sum),
        "uav_drop_sum": float(uav_drop_sum),
        "active_drop_sum": float(active_drop_sum),
        "sat_drop_sum": float(sat_drop_sum),
        "arrival_per_step": float(arrival_per_step),
        "gu_outflow_per_step": float(gu_outflow_per_step),
        "sat_incoming_per_step": float(sat_incoming_per_step),
        "sat_processed_per_step": float(sat_processed_per_step),
        "outflow_arrival_ratio": float(outflow_arrival_ratio),
        "sat_incoming_arrival_ratio": float(sat_incoming_arrival_ratio),
        "gu_drop_ratio": float(gu_drop_ratio),
        "uav_drop_ratio": float(uav_drop_ratio),
        "active_drop_ratio": float(active_drop_ratio),
        "sat_drop_ratio": float(sat_drop_ratio),
        "active_net_drift_sum": float(active_net_drift_sum),
        "active_net_drift_per_step": float(active_net_drift_sum / steps_f),
        "sat_net_drift_sum": float(sat_net_drift_sum),
        "sat_net_drift_per_step": float(sat_net_drift_sum / steps_f),
        "total_net_drift_sum": float(total_net_drift_sum),
        "total_net_drift_per_step": float(total_net_drift_sum / steps_f),
        "queue_total_start_sum": float(queue_total_start_sum),
        "queue_total_end_sum": float(queue_total_end_sum),
        "queue_total_delta_sum": float(queue_total_end_sum - queue_total_start_sum),
        "drop_sum": float(drop_sum),
        "drop_ratio": float(drop_ratio),
    }
    for layer in LAYERS:
        queue_sum_mean = float(layer_queue_sum_mean[layer])
        queue_sum_p95 = float(layer_queue_sum_p95[layer])
        queue_start = float(layer_queue_start_sum[layer])
        queue_end = float(layer_queue_end_sum[layer])
        capacity = max(float(layer_total_capacity[layer]), 1e-9)
        row[f"{layer}_queue_mean"] = float(layer_queue_mean[layer])
        row[f"{layer}_queue_sum_mean"] = queue_sum_mean
        row[f"{layer}_queue_arrival_steps_mean"] = queue_sum_mean / arrival_denom
        row[f"{layer}_queue_fill_fraction_mean"] = queue_sum_mean / capacity
        row[f"{layer}_queue_p95"] = float(layer_queue_p95[layer])
        row[f"{layer}_queue_sum_p95"] = queue_sum_p95
        row[f"{layer}_queue_arrival_steps_p95"] = queue_sum_p95 / arrival_denom
        row[f"{layer}_queue_fill_fraction_p95"] = queue_sum_p95 / capacity
        row[f"first_zero_step_{layer}"] = float(layer_first_zero_step[layer])
        row[f"fraction_steps_{layer}_zero"] = float(layer_fraction_steps_zero[layer])
        row[f"{layer}_queue_nonzero_entity_frac"] = float(layer_nonzero_entity_frac[layer])
        row[f"{layer}_queue_nonzero_mean"] = float(layer_nonzero_mean[layer])
        row[f"{layer}_queue_cap_hit_ratio"] = float(layer_cap_hit_ratio[layer])
        row[f"{layer}_queue_cap_hit_any_ratio"] = float(layer_cap_hit_any_ratio[layer])
        row[f"{layer}_queue_start_sum"] = queue_start
        row[f"{layer}_queue_end_sum"] = queue_end
        row[f"{layer}_queue_delta_sum"] = queue_end - queue_start
        row[f"{layer}_queue_delta_per_step"] = (queue_end - queue_start) / steps_f
    return row


def _summary_row(
    name: str,
    rows: list[Dict[str, float]],
    thresholds: Dict[str, float],
) -> Dict[str, float | str]:
    def arr(field: str) -> np.ndarray:
        return np.asarray([row[field] for row in rows], dtype=np.float64)

    first_zero = arr("first_zero_step_q_active")
    frac_zero = arr("fraction_steps_q_active_zero")
    all_layers_nonzero_frac = arr("fraction_steps_all_layers_nonzero")
    arrival_per_step = arr("arrival_per_step")
    gu_outflow_per_step = arr("gu_outflow_per_step")
    sat_incoming_per_step = arr("sat_incoming_per_step")
    sat_processed_per_step = arr("sat_processed_per_step")
    outflow_ratio = arr("outflow_arrival_ratio")
    sat_incoming_ratio = arr("sat_incoming_arrival_ratio")
    active_drop_ratio = arr("active_drop_ratio")
    sat_drop_ratio = arr("sat_drop_ratio")
    gu_drop_ratio = arr("gu_drop_ratio")
    uav_drop_ratio = arr("uav_drop_ratio")
    active_net_drift = arr("active_net_drift_per_step")
    sat_net_drift = arr("sat_net_drift_per_step")
    total_net_drift = arr("total_net_drift_per_step")
    drop_ratio = arr("drop_ratio")
    gu_drop_sum = arr("gu_drop_sum")
    uav_drop_sum = arr("uav_drop_sum")
    active_drop_sum = arr("active_drop_sum")
    sat_drop_sum = arr("sat_drop_sum")
    queue_total_start = arr("queue_total_start_sum")
    queue_total_end = arr("queue_total_end_sum")
    queue_total_delta = arr("queue_total_delta_sum")
    collisions = arr("collision")

    first_zero_valid = first_zero[first_zero >= 0.0]
    never_zero = np.asarray([1.0 if value < 0.0 else 0.0 for value in first_zero], dtype=np.float64)

    summary: Dict[str, float | str] = {
        "name": name,
        "episodes": len(rows),
        "collision_eps": int(np.sum(collisions >= 0.5)),
        "collision_rate": float(np.mean(collisions)),
        "first_zero_mean": float(np.mean(first_zero_valid)) if first_zero_valid.size else -1.0,
        "first_zero_median": float(np.median(first_zero_valid)) if first_zero_valid.size else -1.0,
        "first_zero_p90": float(np.percentile(first_zero_valid, 90)) if first_zero_valid.size else -1.0,
        "share_first_zero_le_1": float(np.mean((first_zero_valid >= 0.0) & (first_zero_valid <= 1.0)))
        if first_zero_valid.size
        else 0.0,
        "share_first_zero_le_5": float(np.mean((first_zero_valid >= 0.0) & (first_zero_valid <= 5.0)))
        if first_zero_valid.size
        else 0.0,
        "share_first_zero_le_10": float(np.mean((first_zero_valid >= 0.0) & (first_zero_valid <= 10.0)))
        if first_zero_valid.size
        else 0.0,
        "share_never_zero": float(np.mean(never_zero)),
        "fraction_steps_q_active_zero_mean": float(np.mean(frac_zero)),
        "fraction_steps_all_layers_nonzero_mean": float(np.mean(all_layers_nonzero_frac)),
        "arrival_per_step_mean": float(np.mean(arrival_per_step)),
        "gu_outflow_per_step_mean": float(np.mean(gu_outflow_per_step)),
        "sat_incoming_per_step_mean": float(np.mean(sat_incoming_per_step)),
        "sat_processed_per_step_mean": float(np.mean(sat_processed_per_step)),
        "outflow_arrival_ratio_mean": float(np.mean(outflow_ratio)),
        "outflow_arrival_ratio_p05": float(np.percentile(outflow_ratio, 5)),
        "sat_incoming_arrival_ratio_mean": float(np.mean(sat_incoming_ratio)),
        "sat_incoming_arrival_ratio_p05": float(np.percentile(sat_incoming_ratio, 5)),
        "gu_drop_ratio_mean": float(np.mean(gu_drop_ratio)),
        "uav_drop_ratio_mean": float(np.mean(uav_drop_ratio)),
        "active_drop_ratio_mean": float(np.mean(active_drop_ratio)),
        "sat_drop_ratio_mean": float(np.mean(sat_drop_ratio)),
        "active_net_drift_per_step_mean": float(np.mean(active_net_drift)),
        "sat_net_drift_per_step_mean": float(np.mean(sat_net_drift)),
        "total_net_drift_per_step_mean": float(np.mean(total_net_drift)),
        "drop_ratio_mean": float(np.mean(drop_ratio)),
        "gu_drop_sum_mean": float(np.mean(gu_drop_sum)),
        "uav_drop_sum_mean": float(np.mean(uav_drop_sum)),
        "active_drop_sum_mean": float(np.mean(active_drop_sum)),
        "sat_drop_sum_mean": float(np.mean(sat_drop_sum)),
        "queue_total_start_sum_mean": float(np.mean(queue_total_start)),
        "queue_total_end_sum_mean": float(np.mean(queue_total_end)),
        "queue_total_delta_sum_mean": float(np.mean(queue_total_delta)),
    }

    for layer in LAYERS:
        layer_first_zero = arr(f"first_zero_step_{layer}")
        layer_first_zero_valid = layer_first_zero[layer_first_zero >= 0.0]
        layer_never_zero = np.asarray([1.0 if value < 0.0 else 0.0 for value in layer_first_zero], dtype=np.float64)
        summary[f"{layer}_queue_mean_mean"] = float(np.mean(arr(f"{layer}_queue_mean")))
        summary[f"{layer}_queue_sum_mean_mean"] = float(np.mean(arr(f"{layer}_queue_sum_mean")))
        summary[f"{layer}_queue_arrival_steps_mean_mean"] = float(np.mean(arr(f"{layer}_queue_arrival_steps_mean")))
        summary[f"{layer}_queue_fill_fraction_mean_mean"] = float(np.mean(arr(f"{layer}_queue_fill_fraction_mean")))
        summary[f"{layer}_queue_p95_mean"] = float(np.mean(arr(f"{layer}_queue_p95")))
        summary[f"{layer}_queue_sum_p95_mean"] = float(np.mean(arr(f"{layer}_queue_sum_p95")))
        summary[f"{layer}_queue_arrival_steps_p95_mean"] = float(np.mean(arr(f"{layer}_queue_arrival_steps_p95")))
        summary[f"{layer}_queue_fill_fraction_p95_mean"] = float(np.mean(arr(f"{layer}_queue_fill_fraction_p95")))
        summary[f"first_zero_{layer}_mean"] = float(np.mean(layer_first_zero_valid)) if layer_first_zero_valid.size else -1.0
        summary[f"first_zero_{layer}_median"] = (
            float(np.median(layer_first_zero_valid)) if layer_first_zero_valid.size else -1.0
        )
        summary[f"share_{layer}_never_zero"] = float(np.mean(layer_never_zero))
        summary[f"fraction_steps_{layer}_zero_mean"] = float(np.mean(arr(f"fraction_steps_{layer}_zero")))
        summary[f"{layer}_queue_nonzero_entity_frac_mean"] = float(np.mean(arr(f"{layer}_queue_nonzero_entity_frac")))
        summary[f"{layer}_queue_nonzero_mean_mean"] = float(np.mean(arr(f"{layer}_queue_nonzero_mean")))
        summary[f"{layer}_queue_cap_hit_ratio_mean"] = float(np.mean(arr(f"{layer}_queue_cap_hit_ratio")))
        summary[f"{layer}_queue_cap_hit_any_ratio_mean"] = float(np.mean(arr(f"{layer}_queue_cap_hit_any_ratio")))
        summary[f"{layer}_queue_start_sum_mean"] = float(np.mean(arr(f"{layer}_queue_start_sum")))
        summary[f"{layer}_queue_end_sum_mean"] = float(np.mean(arr(f"{layer}_queue_end_sum")))
        summary[f"{layer}_queue_delta_sum_mean"] = float(np.mean(arr(f"{layer}_queue_delta_sum")))
        summary[f"{layer}_queue_delta_per_step_mean"] = float(np.mean(arr(f"{layer}_queue_delta_per_step")))

    summary["active_queue_first_empty_step_median"] = summary["first_zero_median"]
    summary["active_queue_nonempty_all_episode_share"] = summary["share_never_zero"]
    summary["active_queue_empty_step_fraction_mean"] = summary["fraction_steps_q_active_zero_mean"]
    summary["all_layers_nonempty_step_fraction_mean"] = summary["fraction_steps_all_layers_nonzero_mean"]
    summary["total_drop_fraction_mean"] = summary["drop_ratio_mean"]
    summary["gu_backlog_equiv_steps_mean"] = summary["gu_queue_arrival_steps_mean_mean"]
    summary["uav_backlog_equiv_steps_mean"] = summary["uav_queue_arrival_steps_mean_mean"]
    summary["sat_backlog_equiv_steps_mean"] = summary["sat_queue_arrival_steps_mean_mean"]
    summary["gu_backlog_equiv_steps_p95_mean"] = summary["gu_queue_arrival_steps_p95_mean"]
    summary["uav_backlog_equiv_steps_p95_mean"] = summary["uav_queue_arrival_steps_p95_mean"]
    summary["sat_backlog_equiv_steps_p95_mean"] = summary["sat_queue_arrival_steps_p95_mean"]
    summary["gu_buffer_fill_fraction_mean"] = summary["gu_queue_fill_fraction_mean_mean"]
    summary["uav_buffer_fill_fraction_mean"] = summary["uav_queue_fill_fraction_mean_mean"]
    summary["sat_buffer_fill_fraction_mean"] = summary["sat_queue_fill_fraction_mean_mean"]
    summary["gu_buffer_fill_fraction_p95_mean"] = summary["gu_queue_fill_fraction_p95_mean"]
    summary["uav_buffer_fill_fraction_p95_mean"] = summary["uav_queue_fill_fraction_p95_mean"]
    summary["sat_buffer_fill_fraction_p95_mean"] = summary["sat_queue_fill_fraction_p95_mean"]
    arrival_per_step_mean = max(float(summary["arrival_per_step_mean"]), 1e-9)
    for layer in LAYERS:
        drift_ratio = float(summary[f"{layer}_queue_delta_per_step_mean"]) / arrival_per_step_mean
        summary[f"{layer}_queue_drift_ratio_mean"] = drift_ratio
        summary[f"{layer}_queue_drift_abs_ratio_mean"] = abs(drift_ratio)

    summary["dominant_backlog_layer"] = _dominant_layer(
        summary,
        ("gu_backlog_equiv_steps_mean", "uav_backlog_equiv_steps_mean", "sat_backlog_equiv_steps_mean"),
    )
    summary["dominant_fill_layer"] = _dominant_layer(
        summary,
        ("gu_buffer_fill_fraction_mean", "uav_buffer_fill_fraction_mean", "sat_buffer_fill_fraction_mean"),
    )
    summary["dominant_drop_layer"] = _dominant_layer(
        summary,
        ("gu_drop_ratio_mean", "uav_drop_ratio_mean", "sat_drop_ratio_mean"),
    )
    summary["dominant_drift_layer"] = _dominant_layer(
        summary,
        ("gu_queue_drift_abs_ratio_mean", "uav_queue_drift_abs_ratio_mean", "sat_queue_drift_abs_ratio_mean"),
    )
    for layer in LAYERS:
        summary[f"{layer}_empty_step_fraction_mean"] = summary[f"fraction_steps_{layer}_zero_mean"]
        summary[f"{layer}_nonempty_all_episode_share"] = summary[f"share_{layer}_never_zero"]
    priority_layer, priority_type, priority_reason, issue_scores = _choose_tuning_priority(summary, thresholds)
    for key, value in issue_scores.items():
        summary[key] = float(value)
    summary["tuning_priority_layer"] = priority_layer
    summary["tuning_priority_type"] = priority_type
    summary["tuning_priority_reason"] = priority_reason
    summary["recommended_tuning_pair"] = _recommended_tuning_pair(priority_layer)
    summary["tuning_priority_hint"] = _tuning_hint(priority_layer, priority_type)
    passes_thresholds, failed_thresholds = _evaluate_thresholds(summary, thresholds)
    summary["passes_thresholds"] = 1 if passes_thresholds else 0
    summary["failed_threshold_count"] = len(failed_thresholds)
    summary["failed_thresholds"] = ",".join(failed_thresholds) if failed_thresholds else "none"
    score_parts = _compute_score(summary, thresholds)
    for key, value in score_parts.items():
        summary[key] = float(value)
    summary["regime"] = _judge_regime(summary)
    return summary


def _dominant_layer(summary: Dict[str, float | str], fields: tuple[str, str, str]) -> str:
    layer_values = {
        "gu": float(summary[fields[0]]),
        "uav": float(summary[fields[1]]),
        "sat": float(summary[fields[2]]),
    }
    return max(layer_values, key=layer_values.get)


def _positive_excess(value: float, threshold: float) -> float:
    return max(value - threshold, 0.0) / max(threshold, 1e-9)


def _positive_shortfall(value: float, threshold: float) -> float:
    return max(threshold - value, 0.0) / max(threshold, 1e-9)


def _choose_tuning_priority(
    summary: Dict[str, float | str], thresholds: Dict[str, float]
) -> tuple[str, str, str, Dict[str, float]]:
    arrival_per_step_mean = max(float(summary["arrival_per_step_mean"]), 1e-9)
    active_underflow = _positive_excess(
        float(summary["active_queue_empty_step_fraction_mean"]),
        thresholds["active_empty_max"],
    )
    all_layers_underflow = _positive_shortfall(
        float(summary["all_layers_nonempty_step_fraction_mean"]),
        thresholds["all_layers_nonempty_min"],
    )
    issue_scores: Dict[str, float] = {}
    best_layer = "uav"
    best_type = "drop"
    best_score = -1.0

    for layer in LAYERS:
        drift_ratio = float(summary[f"{layer}_queue_drift_ratio_mean"])
        drift_score = _positive_excess(
            abs(drift_ratio),
            thresholds[f"{layer}_drift_abs_max"],
        )
        underflow_score = (
            _positive_excess(
                float(summary[f"{layer}_empty_step_fraction_mean"]),
                thresholds[f"{layer}_empty_max"],
            )
            + _positive_shortfall(
                float(summary[f"{layer}_backlog_equiv_steps_mean"]),
                thresholds[f"{layer}_backlog_min"],
            )
            + 0.5 * active_underflow
            + 0.25 * all_layers_underflow
        )
        overflow_score = (
            _positive_excess(
                float(summary[f"{layer}_buffer_fill_fraction_p95_mean"]),
                thresholds[f"{layer}_fill_p95_max"],
            )
            + max(float(summary[f"{layer}_queue_delta_per_step_mean"]), 0.0) / arrival_per_step_mean
        )
        drop_score = _positive_excess(
            float(summary[f"{layer}_drop_ratio_mean"]),
            thresholds[f"{layer}_drop_max"],
        )
        layer_scores = {
            "underflow": underflow_score,
            "overflow": overflow_score,
            "drop": drop_score,
        }
        issue_scores[f"{layer}_underflow_score"] = underflow_score
        issue_scores[f"{layer}_overflow_score"] = overflow_score
        issue_scores[f"{layer}_drop_score"] = drop_score
        issue_scores[f"{layer}_drift_score"] = drift_score
        issue_scores[f"{layer}_drift_ratio"] = drift_ratio
        layer_type = max(layer_scores, key=layer_scores.get)
        layer_best_score = layer_scores[layer_type]
        if layer_best_score > best_score:
            best_layer = layer
            best_type = layer_type
            best_score = layer_best_score

    drift_layer = max(LAYERS, key=lambda layer: float(issue_scores[f"{layer}_drift_score"]))
    drift_score = float(issue_scores[f"{drift_layer}_drift_score"])
    if drift_score > 0.0:
        drift_ratio = float(summary[f"{drift_layer}_queue_drift_ratio_mean"])
        drift_type = "growth" if drift_ratio >= 0.0 else "decay"
        return drift_layer, drift_type, "largest_queue_drift_violation", issue_scores

    if best_score > 0.0:
        return best_layer, best_type, f"largest_{best_type}_violation", issue_scores

    fill_layer = _dominant_layer(
        summary,
        ("gu_buffer_fill_fraction_mean", "uav_buffer_fill_fraction_mean", "sat_buffer_fill_fraction_mean"),
    )
    fill_value = float(summary[f"{fill_layer}_buffer_fill_fraction_mean"])
    if fill_value > 0.30:
        return fill_layer, "overflow", "highest_buffer_fill_fraction", issue_scores

    drop_layer = _dominant_layer(
        summary,
        ("gu_drop_ratio_mean", "uav_drop_ratio_mean", "sat_drop_ratio_mean"),
    )
    drop_value = float(summary[f"{drop_layer}_drop_ratio_mean"])
    if drop_value > 0.001:
        return drop_layer, "drop", "highest_drop_fraction", issue_scores

    backlog_layer = _dominant_layer(
        summary,
        ("gu_backlog_equiv_steps_mean", "uav_backlog_equiv_steps_mean", "sat_backlog_equiv_steps_mean"),
    )
    return backlog_layer, "overflow", "largest_backlog_equiv_steps", issue_scores


def _tuning_hint(layer: str, issue_type: str) -> str:
    if layer == "gu":
        if issue_type == "decay":
            return "priority: increase task_arrival_rate or reduce b_acc drain; queue_max_gu and queue_init_gu are secondary"
        if issue_type == "growth":
            return "priority: reduce task_arrival_rate or increase b_acc; queue_max_gu and queue_init_gu are secondary"
        if issue_type == "underflow":
            return "priority: increase task_arrival_rate or ease GU drain only if it is overshooting; queue_max_gu and queue_init_gu are secondary"
        if issue_type == "drop":
            return "priority: reduce task_arrival_rate or increase b_acc; queue_max_gu only buffers burst, queue_init_gu only shapes startup"
        return "priority: increase b_acc or reduce task_arrival_rate; queue_max_gu only buffers burst, queue_init_gu only shapes startup"
    if layer == "uav":
        if issue_type == "decay":
            return "priority: increase b_acc or reduce b_sat_total drain; queue_max_uav and queue_init_uav are secondary"
        if issue_type == "growth":
            return "priority: reduce b_acc or increase b_sat_total; queue_max_uav and queue_init_uav are secondary"
        if issue_type == "underflow":
            return "priority: increase b_acc or soften UAV->SAT drain; queue_max_uav and queue_init_uav are secondary"
        if issue_type == "drop":
            return "priority: reduce upstream push or increase b_sat_total; queue_max_uav only buffers burst, queue_init_uav only shapes startup"
        return "priority: increase b_sat_total or reduce upstream push; queue_max_uav only buffers burst, queue_init_uav only shapes startup"
    if issue_type == "decay":
        return "priority: increase b_sat_total or reduce sat_cpu_freq drain; queue_max_sat and queue_init_sat are secondary"
    if issue_type == "growth":
        return "priority: reduce b_sat_total or increase sat_cpu_freq; queue_max_sat and queue_init_sat are secondary"
    if issue_type == "underflow":
        return "priority: increase b_sat_total or ease SAT drain only if it is over-clearing; queue_max_sat and queue_init_sat are secondary"
    if issue_type == "drop":
        return "priority: reduce UAV->SAT injection or increase sat_cpu_freq; queue_max_sat only buffers burst, queue_init_sat only shapes startup"
    return "priority: increase sat_cpu_freq or reduce UAV->SAT injection; queue_max_sat only buffers burst, queue_init_sat only shapes startup"


def _recommended_tuning_pair(layer: str) -> str:
    if layer == "gu":
        return "task_arrival_rate + b_acc"
    if layer == "uav":
        return "b_acc + b_sat_total"
    return "b_sat_total + sat_cpu_freq"


def _build_thresholds(args: argparse.Namespace) -> Dict[str, float]:
    thresholds: Dict[str, float] = {
        "active_empty_max": float(args.thr_active_empty_max),
        "all_layers_nonempty_min": float(args.thr_all_layers_nonempty_min),
        "total_drop_max": float(args.thr_total_drop_max),
        "outflow_ratio_min": float(args.thr_outflow_ratio_min),
        "outflow_ratio_max": float(args.thr_outflow_ratio_max),
    }
    for layer in LAYERS:
        empty_override = getattr(args, f"thr_{layer}_empty_max")
        backlog_override = getattr(args, f"thr_{layer}_backlog_min")
        fill_override = getattr(args, f"thr_{layer}_fill_p95_max")
        drop_override = getattr(args, f"thr_{layer}_drop_max")
        drift_override = getattr(args, f"thr_{layer}_drift_abs_max")
        thresholds[f"{layer}_empty_max"] = float(args.thr_layer_empty_max if empty_override is None else empty_override)
        thresholds[f"{layer}_backlog_min"] = float(
            args.thr_layer_backlog_min if backlog_override is None else backlog_override
        )
        thresholds[f"{layer}_fill_p95_max"] = float(
            args.thr_layer_fill_p95_max if fill_override is None else fill_override
        )
        thresholds[f"{layer}_drop_max"] = float(args.thr_layer_drop_max if drop_override is None else drop_override)
        thresholds[f"{layer}_drift_abs_max"] = float(
            args.thr_layer_drift_abs_max if drift_override is None else drift_override
        )
    return thresholds


def _evaluate_thresholds(summary: Dict[str, float | str], thresholds: Dict[str, float]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if float(summary["active_queue_empty_step_fraction_mean"]) > thresholds["active_empty_max"]:
        failures.append("active_queue_empty_step_fraction")
    if float(summary["all_layers_nonempty_step_fraction_mean"]) < thresholds["all_layers_nonempty_min"]:
        failures.append("all_layers_nonempty_step_fraction")
    if float(summary["total_drop_fraction_mean"]) > thresholds["total_drop_max"]:
        failures.append("total_drop_fraction")
    outflow_ratio = float(summary["outflow_arrival_ratio_mean"])
    if outflow_ratio < thresholds["outflow_ratio_min"] or outflow_ratio > thresholds["outflow_ratio_max"]:
        failures.append("outflow_arrival_ratio")
    for layer in LAYERS:
        if abs(float(summary[f"{layer}_queue_drift_ratio_mean"])) > thresholds[f"{layer}_drift_abs_max"]:
            failures.append(f"{layer}_queue_drift_abs_ratio")
        if float(summary[f"{layer}_empty_step_fraction_mean"]) > thresholds[f"{layer}_empty_max"]:
            failures.append(f"{layer}_empty_step_fraction")
        if float(summary[f"{layer}_backlog_equiv_steps_mean"]) < thresholds[f"{layer}_backlog_min"]:
            failures.append(f"{layer}_backlog_equiv_steps")
        if float(summary[f"{layer}_buffer_fill_fraction_p95_mean"]) > thresholds[f"{layer}_fill_p95_max"]:
            failures.append(f"{layer}_buffer_fill_fraction_p95")
        if float(summary[f"{layer}_drop_ratio_mean"]) > thresholds[f"{layer}_drop_max"]:
            failures.append(f"{layer}_drop_fraction")
    return len(failures) == 0, failures


def _outflow_ratio_penalty(summary: Dict[str, float | str], thresholds: Dict[str, float]) -> float:
    outflow_ratio = float(summary["outflow_arrival_ratio_mean"])
    if outflow_ratio < thresholds["outflow_ratio_min"]:
        return (thresholds["outflow_ratio_min"] - outflow_ratio) / max(thresholds["outflow_ratio_min"], 1e-9)
    if outflow_ratio > thresholds["outflow_ratio_max"]:
        return (outflow_ratio - thresholds["outflow_ratio_max"]) / max(thresholds["outflow_ratio_max"], 1e-9)
    return 0.0


def _compute_score(summary: Dict[str, float | str], thresholds: Dict[str, float]) -> Dict[str, float]:
    score_active_empty = float(summary["active_queue_empty_step_fraction_mean"]) / max(
        thresholds["active_empty_max"], 1e-9
    )
    score_all_layers_nonempty = max(
        thresholds["all_layers_nonempty_min"] - float(summary["all_layers_nonempty_step_fraction_mean"]),
        0.0,
    ) / max(thresholds["all_layers_nonempty_min"], 1e-9)
    score_total_drop = float(summary["total_drop_fraction_mean"]) / max(thresholds["total_drop_max"], 1e-9)
    score_outflow_ratio = _outflow_ratio_penalty(summary, thresholds)

    score_layer_drift = 0.0
    score_layer_empty = 0.0
    score_layer_backlog = 0.0
    score_layer_fill = 0.0
    score_layer_drop = 0.0
    for layer in LAYERS:
        score_layer_drift += abs(float(summary[f"{layer}_queue_drift_ratio_mean"])) / max(
            thresholds[f"{layer}_drift_abs_max"], 1e-9
        )
        score_layer_empty += float(summary[f"{layer}_empty_step_fraction_mean"]) / max(
            thresholds[f"{layer}_empty_max"], 1e-9
        )
        score_layer_backlog += max(
            thresholds[f"{layer}_backlog_min"] - float(summary[f"{layer}_backlog_equiv_steps_mean"]),
            0.0,
        ) / max(thresholds[f"{layer}_backlog_min"], 1e-9)
        score_layer_fill += float(summary[f"{layer}_buffer_fill_fraction_p95_mean"]) / max(
            thresholds[f"{layer}_fill_p95_max"], 1e-9
        )
        score_layer_drop += float(summary[f"{layer}_drop_ratio_mean"]) / max(
            thresholds[f"{layer}_drop_max"], 1e-9
        )

    total_score = (
        3.0 * score_layer_drift
        + 3.0 * score_active_empty
        + 1.5 * score_all_layers_nonempty
        + 3.0 * score_total_drop
        + 1.0 * score_outflow_ratio
        + 1.5 * score_layer_empty
        + 1.0 * score_layer_backlog
        + 1.0 * score_layer_fill
        + 2.0 * score_layer_drop
    )
    return {
        "score_active_empty": score_active_empty,
        "score_all_layers_nonempty": score_all_layers_nonempty,
        "score_total_drop": score_total_drop,
        "score_outflow_ratio": score_outflow_ratio,
        "score_layer_drift": score_layer_drift,
        "score_layer_empty": score_layer_empty,
        "score_layer_backlog": score_layer_backlog,
        "score_layer_fill": score_layer_fill,
        "score_layer_drop": score_layer_drop,
        "score_total": total_score,
    }


def _judge_regime(summary: Dict[str, float | str]) -> str:
    arrival_per_step_mean = float(summary["arrival_per_step_mean"])
    drift_ratio = float(summary["total_net_drift_per_step_mean"]) / max(arrival_per_step_mean, 1e-9)
    cap_hit_any_mean = max(
        float(summary["gu_queue_cap_hit_any_ratio_mean"]),
        float(summary["uav_queue_cap_hit_any_ratio_mean"]),
        float(summary["sat_queue_cap_hit_any_ratio_mean"]),
    )

    if float(summary["drop_ratio_mean"]) > 0.02 or cap_hit_any_mean >= 0.10 or drift_ratio > 0.15:
        return "heavy"

    if (
        float(summary["fraction_steps_q_active_zero_mean"]) >= 0.30
        or float(summary["fraction_steps_gu_zero_mean"]) >= 0.20
        or float(summary["fraction_steps_uav_zero_mean"]) >= 0.30
        or float(summary["fraction_steps_sat_zero_mean"]) >= 0.30
        or float(summary["uav_queue_arrival_steps_mean_mean"]) < 0.10
        or float(summary["sat_queue_arrival_steps_mean_mean"]) < 0.10
        or float(summary["outflow_arrival_ratio_mean"]) > 1.10
        or drift_ratio < -0.15
    ):
        return "light"

    if (
        float(summary["fraction_steps_q_active_zero_mean"]) < 0.05
        and float(summary["fraction_steps_gu_zero_mean"]) < 0.05
        and float(summary["fraction_steps_uav_zero_mean"]) < 0.10
        and float(summary["fraction_steps_sat_zero_mean"]) < 0.10
        and float(summary["uav_queue_arrival_steps_mean_mean"]) >= 0.10
        and float(summary["sat_queue_arrival_steps_mean_mean"]) >= 0.10
        and 0.90 <= float(summary["outflow_arrival_ratio_mean"]) <= 1.10
        and abs(drift_ratio) < 0.10
        and float(summary["drop_ratio_mean"]) < 0.01
        and cap_hit_any_mean < 0.05
    ):
        return "balanced"

    return "mixed"


def _write_csv(path: str, rows: list[Dict[str, float | str]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: str,
    config_path: str,
    baseline: str,
    q_zero_eps: float,
    cap_frac: float,
    override_labels: list[str],
    thresholds: Dict[str, float],
    summary: Dict[str, float | str],
) -> None:
    lines = [
        "# Queue Regime Diagnosis",
        "",
        f"- config: `{config_path}`",
        f"- baseline: `{baseline}`",
        f"- zero threshold: `{q_zero_eps}`",
        f"- cap-hit threshold: `{cap_frac}`",
        f"- overrides: `{', '.join(override_labels)}`" if override_labels else "- overrides: `none`",
        f"- episodes: `{summary['episodes']}`",
        f"- regime: `{summary['regime']}`",
        f"- passes_thresholds: `{int(summary['passes_thresholds'])}`",
        f"- failed_threshold_count: `{int(summary['failed_threshold_count'])}`",
        f"- failed_thresholds: `{summary['failed_thresholds']}`",
        f"- score_total: `{float(summary['score_total']):.4f}`",
        f"- collision_rate: `{float(summary['collision_rate']):.4f}`",
        f"- active_queue_first_empty_step_median: `{float(summary['active_queue_first_empty_step_median']):.4f}`",
        f"- active_queue_first_empty_step_p90: `{float(summary['first_zero_p90']):.4f}`",
        f"- active_queue_nonempty_all_episode_share: `{float(summary['active_queue_nonempty_all_episode_share']):.4f}`",
        f"- active_queue_empty_step_fraction_mean: `{float(summary['active_queue_empty_step_fraction_mean']):.4f}`",
        f"- all_layers_nonempty_step_fraction_mean: `{float(summary['all_layers_nonempty_step_fraction_mean']):.4f}`",
        f"- arrival_per_step_mean: `{float(summary['arrival_per_step_mean']):.4f}`",
        f"- gu_outflow_per_step_mean: `{float(summary['gu_outflow_per_step_mean']):.4f}`",
        f"- sat_incoming_per_step_mean: `{float(summary['sat_incoming_per_step_mean']):.4f}`",
        f"- sat_processed_per_step_mean: `{float(summary['sat_processed_per_step_mean']):.4f}`",
        f"- outflow_arrival_ratio_mean: `{float(summary['outflow_arrival_ratio_mean']):.4f}`",
        f"- sat_incoming_arrival_ratio_mean: `{float(summary['sat_incoming_arrival_ratio_mean']):.4f}`",
        f"- total_net_drift_per_step_mean: `{float(summary['total_net_drift_per_step_mean']):.4f}`",
        f"- total_drop_fraction_mean: `{float(summary['total_drop_fraction_mean']):.6f}`",
        f"- dominant_backlog_layer: `{summary['dominant_backlog_layer']}`",
        f"- dominant_fill_layer: `{summary['dominant_fill_layer']}`",
        f"- dominant_drop_layer: `{summary['dominant_drop_layer']}`",
        f"- dominant_drift_layer: `{summary['dominant_drift_layer']}`",
        f"- tuning_priority_layer: `{summary['tuning_priority_layer']}`",
        f"- tuning_priority_type: `{summary['tuning_priority_type']}`",
        f"- recommended_tuning_pair: `{summary['recommended_tuning_pair']}`",
        f"- tuning_priority_reason: `{summary['tuning_priority_reason']}`",
        f"- tuning_priority_hint: `{summary['tuning_priority_hint']}`",
        "",
        "## Thresholds",
        f"- active_queue_empty_step_fraction_max: `{thresholds['active_empty_max']:.4f}`",
        f"- all_layers_nonempty_step_fraction_min: `{thresholds['all_layers_nonempty_min']:.4f}`",
        f"- total_drop_fraction_max: `{thresholds['total_drop_max']:.4f}`",
        f"- outflow_arrival_ratio_range: `[{thresholds['outflow_ratio_min']:.4f}, {thresholds['outflow_ratio_max']:.4f}]`",
        "",
        "## Layer Breakdown",
    ]
    for layer in LAYERS:
        lines.extend(
            [
                "",
                f"### {layer.upper()}",
                f"- threshold_queue_drift_abs_ratio_max: `{thresholds[f'{layer}_drift_abs_max']:.4f}`",
                f"- threshold_empty_step_fraction_max: `{thresholds[f'{layer}_empty_max']:.4f}`",
                f"- threshold_backlog_equiv_steps_min: `{thresholds[f'{layer}_backlog_min']:.4f}`",
                f"- threshold_buffer_fill_fraction_p95_max: `{thresholds[f'{layer}_fill_p95_max']:.4f}`",
                f"- threshold_drop_fraction_max: `{thresholds[f'{layer}_drop_max']:.4f}`",
                f"- first_zero_median: `{float(summary[f'first_zero_{layer}_median']):.4f}`",
                f"- nonempty_all_episode_share: `{float(summary[f'{layer}_nonempty_all_episode_share']):.4f}`",
                f"- empty_step_fraction_mean: `{float(summary[f'{layer}_empty_step_fraction_mean']):.4f}`",
                f"- queue_mean_mean: `{float(summary[f'{layer}_queue_mean_mean']):.4f}`",
                f"- queue_sum_mean_mean: `{float(summary[f'{layer}_queue_sum_mean_mean']):.4f}`",
                f"- backlog_equiv_steps_mean: `{float(summary[f'{layer}_backlog_equiv_steps_mean']):.4f}`",
                f"- backlog_equiv_steps_p95_mean: `{float(summary[f'{layer}_backlog_equiv_steps_p95_mean']):.4f}`",
                f"- queue_drift_ratio_mean: `{float(summary[f'{layer}_queue_drift_ratio_mean']):.6f}`",
                f"- buffer_fill_fraction_mean: `{float(summary[f'{layer}_buffer_fill_fraction_mean']):.4f}`",
                f"- buffer_fill_fraction_p95_mean: `{float(summary[f'{layer}_buffer_fill_fraction_p95_mean']):.4f}`",
                f"- queue_nonzero_entity_frac_mean: `{float(summary[f'{layer}_queue_nonzero_entity_frac_mean']):.4f}`",
                f"- queue_nonzero_mean_mean: `{float(summary[f'{layer}_queue_nonzero_mean_mean']):.4f}`",
                f"- drop_fraction_mean: `{float(summary[f'{layer}_drop_ratio_mean']):.6f}`",
                f"- queue_cap_hit_ratio_mean: `{float(summary[f'{layer}_queue_cap_hit_ratio_mean']):.4f}`",
                f"- queue_cap_hit_any_ratio_mean: `{float(summary[f'{layer}_queue_cap_hit_any_ratio_mean']):.4f}`",
                f"- queue_start_sum_mean: `{float(summary[f'{layer}_queue_start_sum_mean']):.4f}`",
                f"- queue_end_sum_mean: `{float(summary[f'{layer}_queue_end_sum_mean']):.4f}`",
                f"- queue_delta_sum_mean: `{float(summary[f'{layer}_queue_delta_sum_mean']):.4f}`",
                f"- queue_delta_per_step_mean: `{float(summary[f'{layer}_queue_delta_per_step_mean']):.4f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Totals",
            f"- queue_total_start_sum_mean: `{float(summary['queue_total_start_sum_mean']):.4f}`",
            f"- queue_total_end_sum_mean: `{float(summary['queue_total_end_sum_mean']):.4f}`",
            f"- queue_total_delta_sum_mean: `{float(summary['queue_total_delta_sum_mean']):.4f}`",
            f"- gu_drop_sum_mean: `{float(summary['gu_drop_sum_mean']):.4f}`",
            f"- uav_drop_sum_mean: `{float(summary['uav_drop_sum_mean']):.4f}`",
            f"- active_drop_sum_mean: `{float(summary['active_drop_sum_mean']):.4f}`",
            f"- sat_drop_sum_mean: `{float(summary['sat_drop_sum_mean']):.4f}`",
            "",
            "## Score Components",
            f"- score_active_empty: `{float(summary['score_active_empty']):.4f}`",
            f"- score_all_layers_nonempty: `{float(summary['score_all_layers_nonempty']):.4f}`",
            f"- score_total_drop: `{float(summary['score_total_drop']):.4f}`",
            f"- score_outflow_ratio: `{float(summary['score_outflow_ratio']):.4f}`",
            f"- score_layer_drift: `{float(summary['score_layer_drift']):.4f}`",
            f"- score_layer_empty: `{float(summary['score_layer_empty']):.4f}`",
            f"- score_layer_backlog: `{float(summary['score_layer_backlog']):.4f}`",
            f"- score_layer_fill: `{float(summary['score_layer_fill']):.4f}`",
            f"- score_layer_drop: `{float(summary['score_layer_drop']):.4f}`",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compute_pareto_flags(
    rows: list[Dict[str, float | str]],
    metrics: tuple[str, str] = ("total_drop_fraction_mean", "active_queue_empty_step_fraction_mean"),
) -> list[int]:
    flags = [1] * len(rows)
    for i, row_i in enumerate(rows):
        xi = tuple(float(row_i[field]) for field in metrics)
        for j, row_j in enumerate(rows):
            if i == j:
                continue
            xj = tuple(float(row_j[field]) for field in metrics)
            if all(vj <= vi for vj, vi in zip(xj, xi)) and any(vj < vi for vj, vi in zip(xj, xi)):
                flags[i] = 0
                break
    return flags


def _write_grid_report(
    path: str,
    config_path: str,
    baseline: str,
    episodes: int,
    base_override_labels: list[str],
    thresholds: Dict[str, float],
    rows: list[Dict[str, float | str]],
    grid_specs: list[tuple[str, list[str]]],
) -> None:
    grid_axes_label = ", ".join(f"{key}={','.join(values)}" for key, values in grid_specs)
    lines = [
        "# Queue Regime Grid Search",
        "",
        f"- config: `{config_path}`",
        f"- baseline: `{baseline}`",
        f"- episodes_per_point: `{episodes}`",
        f"- base_overrides: `{', '.join(base_override_labels)}`" if base_override_labels else "- base_overrides: `none`",
        f"- grid_axes: `{grid_axes_label}`",
        f"- candidates: `{len(rows)}`",
        "",
        "## Thresholds",
        f"- active_queue_empty_step_fraction_max: `{thresholds['active_empty_max']:.4f}`",
        f"- all_layers_nonempty_step_fraction_min: `{thresholds['all_layers_nonempty_min']:.4f}`",
        f"- total_drop_fraction_max: `{thresholds['total_drop_max']:.4f}`",
        f"- outflow_arrival_ratio_range: `[{thresholds['outflow_ratio_min']:.4f}, {thresholds['outflow_ratio_max']:.4f}]`",
    ]
    for layer in LAYERS:
        lines.extend(
            [
                f"- {layer}_queue_drift_abs_ratio_max: `{thresholds[f'{layer}_drift_abs_max']:.4f}`",
                f"- {layer}_empty_step_fraction_max: `{thresholds[f'{layer}_empty_max']:.4f}`",
                f"- {layer}_backlog_equiv_steps_min: `{thresholds[f'{layer}_backlog_min']:.4f}`",
                f"- {layer}_buffer_fill_fraction_p95_max: `{thresholds[f'{layer}_fill_p95_max']:.4f}`",
                f"- {layer}_drop_fraction_max: `{thresholds[f'{layer}_drop_max']:.4f}`",
            ]
        )
    lines.append("")
    lines.append("## Top Candidates")
    for row in rows[: min(10, len(rows))]:
        combo_bits = [f"{key}={row[key]}" for key, _ in grid_specs]
        lines.extend(
            [
                "",
                f"### Rank {int(row['rank'])}",
                f"- combo: `{', '.join(combo_bits)}`",
                f"- passes_thresholds: `{int(row['passes_thresholds'])}`",
                f"- failed_thresholds: `{row['failed_thresholds']}`",
                f"- score_total: `{float(row['score_total']):.4f}`",
                f"- pareto_drop_empty: `{int(row['pareto_drop_empty'])}`",
                f"- active_queue_empty_step_fraction_mean: `{float(row['active_queue_empty_step_fraction_mean']):.4f}`",
                f"- all_layers_nonempty_step_fraction_mean: `{float(row['all_layers_nonempty_step_fraction_mean']):.4f}`",
                f"- total_drop_fraction_mean: `{float(row['total_drop_fraction_mean']):.6f}`",
                f"- dominant_backlog_layer: `{row['dominant_backlog_layer']}`",
                f"- dominant_fill_layer: `{row['dominant_fill_layer']}`",
                f"- dominant_drop_layer: `{row['dominant_drop_layer']}`",
                f"- dominant_drift_layer: `{row['dominant_drift_layer']}`",
                f"- tuning_priority_layer: `{row['tuning_priority_layer']}`",
                f"- tuning_priority_type: `{row['tuning_priority_type']}`",
                f"- recommended_tuning_pair: `{row['recommended_tuning_pair']}`",
            ]
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_diagnosis(
    cfg,
    baseline: str,
    episodes: int,
    episode_seed_base: int | None,
    q_zero_eps: float,
    cap_frac: float,
    thresholds: Dict[str, float],
    progress_desc: str,
) -> tuple[list[Dict[str, float]], Dict[str, float | str]]:
    env = SaginParallelEnv(cfg)
    progress = Progress(episodes, desc=progress_desc)
    rows: list[Dict[str, float]] = []

    layer_caps = {
        "gu": float(cfg.queue_max_gu),
        "uav": float(cfg.queue_max_uav),
        "sat": float(cfg.queue_max_sat),
    }
    layer_total_capacity = {
        "gu": float(cfg.num_gu) * float(cfg.queue_max_gu),
        "uav": float(cfg.num_uav) * float(cfg.queue_max_uav),
        "sat": float(cfg.num_sat) * float(cfg.queue_max_sat),
    }

    for ep in range(episodes):
        ep_seed = None if episode_seed_base is None else int(episode_seed_base) + ep
        obs, _ = env.reset(seed=ep_seed)
        done = False
        steps = 0
        first_zero_step_q_active = -1.0
        q_active_zero_count = 0.0
        all_layers_nonzero_count = 0.0
        layer_queue_steps = {layer: [] for layer in LAYERS}
        layer_queue_sum_steps = {layer: [] for layer in LAYERS}
        layer_first_zero_step = {layer: -1.0 for layer in LAYERS}
        layer_zero_count = {layer: 0.0 for layer in LAYERS}
        layer_nonzero_entity_frac_steps = {layer: [] for layer in LAYERS}
        layer_nonzero_mean_steps = {layer: [] for layer in LAYERS}
        layer_cap_hits = {layer: [] for layer in LAYERS}
        layer_cap_hit_any_steps = {layer: [] for layer in LAYERS}
        arrival_sum = 0.0
        gu_outflow_sum = 0.0
        sat_incoming_sum = 0.0
        sat_processed_sum = 0.0
        gu_drop_sum = 0.0
        uav_drop_sum = 0.0
        active_drop_sum = 0.0
        sat_drop_sum = 0.0
        collision = 0.0
        layer_queue_start_sum = {
            "gu": float(np.sum(env.gu_queue)),
            "uav": float(np.sum(env.uav_queue)),
            "sat": float(np.sum(env.sat_queue)),
        }

        while not done:
            obs_list = list(obs.values())
            accel_actions, bw_logits, sat_logits = _baseline_actions(
                baseline,
                obs_list,
                cfg,
                len(env.agents),
                rng=env.rng,
            )
            actions = assemble_actions(cfg, env.agents, accel_actions, bw_logits=bw_logits, sat_logits=sat_logits)
            obs, rewards, terms, truncs, _ = env.step(actions)
            done = bool(list(terms.values())[0] or list(truncs.values())[0])
            steps += 1

            parts = dict(getattr(env, "last_reward_parts", {}) or {})
            q_active = float(parts.get("queue_total_active", float(np.sum(env.gu_queue) + np.sum(env.uav_queue))))
            if q_active <= q_zero_eps:
                q_active_zero_count += 1.0
                if first_zero_step_q_active < 0.0:
                    first_zero_step_q_active = float(steps - 1)

            queues = {"gu": env.gu_queue, "uav": env.uav_queue, "sat": env.sat_queue}
            for layer, queue in queues.items():
                queue_mean = float(np.mean(queue)) if queue.size else 0.0
                queue_sum = float(np.sum(queue))
                layer_queue_steps[layer].append(queue_mean)
                layer_queue_sum_steps[layer].append(queue_sum)
                if queue_sum <= q_zero_eps:
                    layer_zero_count[layer] += 1.0
                    if layer_first_zero_step[layer] < 0.0:
                        layer_first_zero_step[layer] = float(steps - 1)
                layer_nonzero_entity_frac_steps[layer].append(_nonzero_entity_frac(queue, q_zero_eps))
                layer_nonzero_mean_steps[layer].append(_nonzero_entity_mean(queue, q_zero_eps))
                layer_cap_hits[layer].append(_cap_hit_step_ratio(queue, layer_caps[layer], cap_frac))
                layer_cap_hit_any_steps[layer].append(_cap_hit_any(queue, layer_caps[layer], cap_frac))
            if all(float(np.sum(queue)) > q_zero_eps for queue in queues.values()):
                all_layers_nonzero_count += 1.0

            arrival_sum += float(parts.get("arrival_sum", 0.0))
            gu_outflow_sum += float(parts.get("outflow_sum", 0.0))
            sat_incoming_sum += float(parts.get("backhaul_sum", 0.0))
            sat_processed_sum += float(np.sum(getattr(env, "last_sat_processed", 0.0)))
            gu_drop_sum += float(parts.get("gu_drop_sum", 0.0))
            uav_drop_sum += float(parts.get("uav_drop_sum", 0.0))
            active_drop_sum += float(parts.get("drop_sum_active", 0.0))
            sat_drop_sum += float(parts.get("sat_drop_sum", 0.0))
            collision = max(collision, float(parts.get("collision_event", 0.0)))

        layer_queue_end_sum = {
            "gu": float(np.sum(env.gu_queue)),
            "uav": float(np.sum(env.uav_queue)),
            "sat": float(np.sum(env.sat_queue)),
        }
        row = _episode_regime_row(
            episode=ep,
            steps=steps,
            collision=collision,
            first_zero_step_q_active=first_zero_step_q_active,
            fraction_steps_q_active_zero=(q_active_zero_count / max(steps, 1)),
            fraction_steps_all_layers_nonzero=(all_layers_nonzero_count / max(steps, 1)),
            layer_queue_mean={
                layer: float(np.mean(layer_queue_steps[layer])) if layer_queue_steps[layer] else 0.0
                for layer in LAYERS
            },
            layer_queue_sum_mean={
                layer: float(np.mean(layer_queue_sum_steps[layer])) if layer_queue_sum_steps[layer] else 0.0
                for layer in LAYERS
            },
            layer_queue_p95={layer: _safe_percentile(layer_queue_steps[layer], 95.0) for layer in LAYERS},
            layer_queue_sum_p95={layer: _safe_percentile(layer_queue_sum_steps[layer], 95.0) for layer in LAYERS},
            layer_first_zero_step=layer_first_zero_step,
            layer_fraction_steps_zero={layer: layer_zero_count[layer] / max(steps, 1) for layer in LAYERS},
            layer_nonzero_entity_frac={
                layer: float(np.mean(layer_nonzero_entity_frac_steps[layer]))
                if layer_nonzero_entity_frac_steps[layer]
                else 0.0
                for layer in LAYERS
            },
            layer_nonzero_mean={
                layer: float(np.mean(layer_nonzero_mean_steps[layer])) if layer_nonzero_mean_steps[layer] else 0.0
                for layer in LAYERS
            },
            layer_cap_hit_ratio={
                layer: float(np.mean(layer_cap_hits[layer])) if layer_cap_hits[layer] else 0.0
                for layer in LAYERS
            },
            layer_cap_hit_any_ratio={
                layer: float(np.mean(layer_cap_hit_any_steps[layer])) if layer_cap_hit_any_steps[layer] else 0.0
                for layer in LAYERS
            },
            arrival_sum=arrival_sum,
            gu_outflow_sum=gu_outflow_sum,
            sat_incoming_sum=sat_incoming_sum,
            sat_processed_sum=sat_processed_sum,
            gu_drop_sum=gu_drop_sum,
            uav_drop_sum=uav_drop_sum,
            active_drop_sum=active_drop_sum,
            sat_drop_sum=sat_drop_sum,
            layer_queue_start_sum=layer_queue_start_sum,
            layer_queue_end_sum=layer_queue_end_sum,
            layer_total_capacity=layer_total_capacity,
        )
        rows.append(row)
        progress.update(ep + 1)

    progress.close()
    summary = _summary_row(baseline, rows, thresholds)
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--baseline",
        type=str,
        default="fixed",
        choices=["fixed", "zero_accel", "random_accel", "centroid", "queue_aware"],
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--episode_seed_base", type=int, default=None)
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--summary_csv", type=str, default=None)
    parser.add_argument("--report_md", type=str, default=None)
    parser.add_argument("--q_zero_eps", type=float, default=1e-6)
    parser.add_argument("--cap_frac", type=float, default=0.98)
    parser.add_argument("--thr_active_empty_max", type=float, default=0.25)
    parser.add_argument("--thr_all_layers_nonempty_min", type=float, default=0.20)
    parser.add_argument("--thr_total_drop_max", type=float, default=0.05)
    parser.add_argument("--thr_outflow_ratio_min", type=float, default=0.90)
    parser.add_argument("--thr_outflow_ratio_max", type=float, default=1.10)
    parser.add_argument("--thr_layer_drift_abs_max", type=float, default=0.05)
    parser.add_argument("--thr_layer_empty_max", type=float, default=0.25)
    parser.add_argument("--thr_layer_backlog_min", type=float, default=0.25)
    parser.add_argument("--thr_layer_fill_p95_max", type=float, default=0.20)
    parser.add_argument("--thr_layer_drop_max", type=float, default=0.02)
    parser.add_argument("--thr_gu_drift_abs_max", type=float, default=None)
    parser.add_argument("--thr_uav_drift_abs_max", type=float, default=None)
    parser.add_argument("--thr_sat_drift_abs_max", type=float, default=None)
    parser.add_argument("--thr_gu_empty_max", type=float, default=None)
    parser.add_argument("--thr_uav_empty_max", type=float, default=None)
    parser.add_argument("--thr_sat_empty_max", type=float, default=None)
    parser.add_argument("--thr_gu_backlog_min", type=float, default=None)
    parser.add_argument("--thr_uav_backlog_min", type=float, default=None)
    parser.add_argument("--thr_sat_backlog_min", type=float, default=None)
    parser.add_argument("--thr_gu_fill_p95_max", type=float, default=None)
    parser.add_argument("--thr_uav_fill_p95_max", type=float, default=None)
    parser.add_argument("--thr_sat_fill_p95_max", type=float, default=None)
    parser.add_argument("--thr_gu_drop_max", type=float, default=None)
    parser.add_argument("--thr_uav_drop_max", type=float, default=None)
    parser.add_argument("--thr_sat_drop_max", type=float, default=None)
    parser.add_argument(
        "--grid",
        dest="grid_specs",
        action="append",
        default=[],
        help="Grid axis as key=v1,v2,... Can be repeated.",
    )
    parser.add_argument("--grid_summary_csv", type=str, default=None)
    parser.add_argument("--grid_report_md", type=str, default=None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config scalars with key=value. Can be repeated.",
    )
    args = parser.parse_args()

    stem = _default_output_stem(args.config, args.baseline, args.episodes)
    out_csv = args.out_csv or f"{stem}.csv"
    summary_csv = args.summary_csv or f"{stem}_summary.csv"
    report_md = args.report_md or f"{stem}_report.md"
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    overrides, override_labels = _parse_overrides(args.overrides)
    if overrides:
        update_config(cfg, overrides)
    thresholds = _build_thresholds(args)
    if args.grid_specs:
        grid_specs = _parse_grid_specs(args.grid_specs)
        grid_stem = _default_output_stem(args.config, f"{args.baseline}_grid", args.episodes)
        grid_summary_csv = args.grid_summary_csv or f"{grid_stem}_summary.csv"
        grid_report_md = args.grid_report_md or f"{grid_stem}_report.md"
        grid_rows: list[Dict[str, float | str]] = []
        keys = [key for key, _ in grid_specs]
        values_product = itertools.product(*(values for _, values in grid_specs))
        for idx, combo_values in enumerate(values_product, start=1):
            combo_overrides = dict(overrides)
            combo_labels = list(override_labels)
            for key, value in zip(keys, combo_values):
                combo_overrides[key] = value
                combo_labels.append(f"{key}={value}")
            cfg_combo = copy.deepcopy(cfg)
            combo_only = {key: value for key, value in zip(keys, combo_values)}
            update_config(cfg_combo, combo_only)
            _, summary = _run_diagnosis(
                cfg_combo,
                args.baseline,
                args.episodes,
                args.episode_seed_base,
                args.q_zero_eps,
                args.cap_frac,
                thresholds,
                progress_desc=f"Grid{idx}",
            )
            summary["grid_index"] = idx
            summary["overrides"] = ",".join(combo_labels) if combo_labels else "none"
            for key, value in zip(keys, combo_values):
                summary[key] = value
            grid_rows.append(summary)

        pareto_flags = _compute_pareto_flags(grid_rows)
        for row, pareto_flag in zip(grid_rows, pareto_flags):
            row["pareto_drop_empty"] = pareto_flag
        grid_rows.sort(
            key=lambda row: (
                -int(row["passes_thresholds"]),
                int(row["failed_threshold_count"]),
                float(row["score_total"]),
                -int(row["pareto_drop_empty"]),
            )
        )
        for rank, row in enumerate(grid_rows, start=1):
            row["rank"] = rank
        _write_csv(grid_summary_csv, grid_rows)
        _write_grid_report(
            grid_report_md,
            args.config,
            args.baseline,
            args.episodes,
            override_labels,
            thresholds,
            grid_rows,
            grid_specs,
        )
        best = grid_rows[0]
        print(f"Saved grid summary diagnosis to {grid_summary_csv}")
        print(f"Saved grid report to {grid_report_md}")
        print(
            "Grid best summary: "
            f"rank={int(best['rank'])} "
            f"passes_thresholds={int(best['passes_thresholds'])} "
            f"score_total={float(best['score_total']):.3f} "
            f"total_drop_fraction_mean={float(best['total_drop_fraction_mean']):.6f} "
            f"active_queue_empty_step_fraction_mean={float(best['active_queue_empty_step_fraction_mean']):.3f} "
            f"tuning_priority_layer={best['tuning_priority_layer']} "
            f"tuning_priority_type={best['tuning_priority_type']}"
        )
        return

    rows, summary = _run_diagnosis(
        cfg,
        args.baseline,
        args.episodes,
        args.episode_seed_base,
        args.q_zero_eps,
        args.cap_frac,
        thresholds,
        progress_desc="QueueDiag",
    )
    _write_csv(out_csv, rows)
    _write_csv(summary_csv, [summary])
    _write_report(
        report_md,
        args.config,
        args.baseline,
        args.q_zero_eps,
        args.cap_frac,
        override_labels,
        thresholds,
        summary,
    )

    print(f"Saved episode diagnosis to {out_csv}")
    print(f"Saved summary diagnosis to {summary_csv}")
    print(f"Saved diagnosis report to {report_md}")
    print(
        "Queue regime summary: "
        f"regime={summary['regime']} "
        f"active_queue_empty_step_fraction_mean={float(summary['active_queue_empty_step_fraction_mean']):.3f} "
        f"uav_backlog_equiv_steps_mean={float(summary['uav_backlog_equiv_steps_mean']):.3f} "
        f"sat_backlog_equiv_steps_mean={float(summary['sat_backlog_equiv_steps_mean']):.3f} "
        f"total_drop_fraction_mean={float(summary['total_drop_fraction_mean']):.6f} "
        f"passes_thresholds={int(summary['passes_thresholds'])} "
        f"score_total={float(summary['score_total']):.3f} "
        f"tuning_priority_layer={summary['tuning_priority_layer']} "
        f"tuning_priority_type={summary['tuning_priority_type']} "
        f"outflow_arrival_ratio_mean={float(summary['outflow_arrival_ratio_mean']):.3f}"
    )


if __name__ == "__main__":
    main()
