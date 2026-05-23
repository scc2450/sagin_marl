from __future__ import annotations

import argparse
import copy
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

from sagin_marl.env.config import load_config
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.baselines import cluster_center_accel_policy, queue_aware_policy
from sagin_marl.rl.structured_mappo import _refresh_stage_obs_cache
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver


DEFAULT_OLD_CONFIG = (
    "configs/"
    "phase1_actions_curriculum_joint_3heads_fading_interference_ka_vsat_joint_puremappo_"
    "criticdecoupled_diag_timeline_structured_step250_gamma0995_bwonly_learnedpartners_user0.yaml"
)
DEFAULT_FOCUS_CONFIG = (
    "configs/"
    "phase1_actions_curriculum_joint_3heads_fading_interference_ka_vsat_joint_puremappo_"
    "criticdecoupled_diag_timeline_structured_step250_gamma0995_bwfocusv1_bwonly_learnedpartners_user0.yaml"
)


def _current_obs_list(driver: StructuredControlDriver) -> list[dict[str, np.ndarray]]:
    env = driver.env
    return [env._get_obs(i) for i in range(len(env.agents))]


def _normalize_bw_action(action: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(np.asarray(action, dtype=np.float32))
    valid_arr = np.asarray(valid_mask, dtype=bool)
    for agent_idx in range(int(out.shape[0])):
        valid = valid_arr[agent_idx]
        valid_count = int(np.sum(valid))
        if valid_count <= 0:
            continue
        row = np.asarray(action[agent_idx], dtype=np.float32)[valid]
        row = np.clip(row, 0.0, None)
        row_sum = float(np.sum(row))
        if row_sum <= 1.0e-12:
            out[agent_idx, valid] = 1.0 / float(valid_count)
        else:
            out[agent_idx, valid] = row / row_sum
    return out


def _uniform_bw_action(valid_mask: np.ndarray) -> np.ndarray:
    valid_arr = np.asarray(valid_mask, dtype=bool)
    out = np.zeros(valid_arr.shape, dtype=np.float32)
    for agent_idx in range(int(valid_arr.shape[0])):
        valid = valid_arr[agent_idx]
        valid_count = int(np.sum(valid))
        if valid_count > 0:
            out[agent_idx, valid] = 1.0 / float(valid_count)
    return out


def _sat_mask_to_pair_action(
    driver: StructuredControlDriver,
    sat_select_mask: np.ndarray,
) -> np.ndarray:
    cfg = driver.env.cfg
    select_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 1)
    actions = np.full((cfg.num_uav, select_k), -1, dtype=np.int64)
    if driver._stage_visible is None:
        raise RuntimeError("run_accel_stage must be called before mapping sat selection masks.")
    sat_mask_arr = np.asarray(sat_select_mask, dtype=np.float32)
    for u in range(int(cfg.num_uav)):
        visible = list(driver._stage_visible[u][: cfg.sats_obs_max])
        chosen: list[int] = []
        for slot_idx, sat_idx in enumerate(visible):
            if slot_idx >= sat_mask_arr.shape[1]:
                break
            if sat_mask_arr[u, slot_idx] > 0.5 and int(sat_idx) not in chosen:
                chosen.append(int(sat_idx))
            if len(chosen) >= select_k:
                break
        actions[u, : len(chosen)] = np.asarray(chosen[:select_k], dtype=np.int64)
    return actions


def _step_metrics(parts: dict[str, Any]) -> dict[str, float]:
    return {
        "x_acc": float(parts.get("x_acc", 0.0) or 0.0),
        "d_pre": float(parts.get("d_pre", 0.0) or 0.0),
        "pre_backlog_steps_eval": float(parts.get("pre_backlog_steps_eval", 0.0) or 0.0),
    }


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "p10": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "min": 0.0,
            "max": 0.0,
            "positive_frac": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10.0)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "positive_frac": float(np.mean(arr > 0.0)),
    }


def _safe_ratio(numer: float, denom: float) -> float | None:
    denom_abs = abs(float(denom))
    if denom_abs <= 1.0e-12:
        return None
    return float(numer) / float(denom)


def _prepare_step_prefix(driver: StructuredControlDriver) -> dict[str, Any]:
    env = driver.env
    cfg = env.cfg
    obs_before = _current_obs_list(driver)
    centers = getattr(env, "gu_cluster_centers", None)
    counts = getattr(env, "gu_cluster_counts", None)
    accel_action = cluster_center_accel_policy(obs_before, cfg, centers, counts)
    driver.run_accel_stage(accel_action)
    _refresh_stage_obs_cache(driver)
    obs_after_accel = _current_obs_list(driver)
    _accel_unused, heuristic_bw_raw, sat_mask = queue_aware_policy(obs_after_accel, cfg)
    sat_pair_action = _sat_mask_to_pair_action(driver, sat_mask)
    driver.run_sat_stage(sat_pair_action)
    valid_mask = np.asarray(driver._stage_bw_valid_mask, dtype=np.float32) > 0.5
    heuristic_bw = _normalize_bw_action(heuristic_bw_raw, valid_mask)
    uniform_bw = _uniform_bw_action(valid_mask)
    return {
        "heuristic_bw": heuristic_bw,
        "uniform_bw": uniform_bw,
        "valid_mask": valid_mask.astype(np.float32, copy=False),
        "mean_valid_slots": float(np.mean(np.sum(valid_mask, axis=1))) if valid_mask.size > 0 else 0.0,
    }


def _execute_one_step(
    prefix_driver: StructuredControlDriver,
    bw_action: np.ndarray,
) -> tuple[dict[str, float], bool]:
    cf_driver = copy.deepcopy(prefix_driver)
    step_result = cf_driver.execute_stage_bw_and_step(np.asarray(bw_action, dtype=np.float32))
    done = bool(next(iter(step_result.terminations.values())) or next(iter(step_result.truncations.values())))
    metrics = _step_metrics(dict(getattr(cf_driver.env, "last_reward_parts", {}) or {}))
    return metrics, done


def _rollout_prefix_k(
    prefix_driver: StructuredControlDriver,
    *,
    k_steps: int,
    first_bw_action: np.ndarray,
    bw_source: str,
) -> dict[str, float]:
    cf_driver = copy.deepcopy(prefix_driver)
    cfg = cf_driver.env.cfg
    gamma = float(cfg.gamma)
    discount = 1.0
    totals = {
        "x_acc": 0.0,
        "d_pre": 0.0,
        "pre_backlog_steps_eval": 0.0,
        "steps_executed": 0.0,
    }
    current_bw = np.asarray(first_bw_action, dtype=np.float32)
    for _ in range(max(int(k_steps), 0)):
        step_result = cf_driver.execute_stage_bw_and_step(current_bw)
        parts = dict(getattr(cf_driver.env, "last_reward_parts", {}) or {})
        metrics = _step_metrics(parts)
        for key in ("x_acc", "d_pre", "pre_backlog_steps_eval"):
            totals[key] += discount * float(metrics[key])
        totals["steps_executed"] += 1.0
        done = bool(next(iter(step_result.terminations.values())) or next(iter(step_result.truncations.values())))
        if done:
            break
        discount *= gamma
        next_prefix = _prepare_step_prefix(cf_driver)
        current_bw = np.asarray(next_prefix[f"{bw_source}_bw"], dtype=np.float32)
    return totals


def audit_config(
    *,
    config_path: str,
    label: str,
    episodes: int,
    seed_base: int,
    max_state_samples: int,
    progress_every: int,
    rollout_bw_source: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = load_config(config_path)
    driver = make_structured_driver(cfg, backend="sync")
    env = driver.env
    rows: list[dict[str, Any]] = []
    state_counter = 0
    try:
        for episode_idx in range(max(int(episodes), 0)):
            if state_counter >= int(max_state_samples):
                break
            seed = int(seed_base) + int(episode_idx)
            env.reset(seed=seed)
            done = False
            step_idx = 0
            while not done and state_counter < int(max_state_samples):
                prefix = _prepare_step_prefix(driver)
                one_uniform, uniform_done = _execute_one_step(driver, prefix["uniform_bw"])
                one_heuristic, heuristic_done = _execute_one_step(driver, prefix["heuristic_bw"])
                k2_uniform = _rollout_prefix_k(
                    driver,
                    k_steps=2,
                    first_bw_action=prefix["uniform_bw"],
                    bw_source="uniform",
                )
                k2_heuristic = _rollout_prefix_k(
                    driver,
                    k_steps=2,
                    first_bw_action=prefix["heuristic_bw"],
                    bw_source="heuristic",
                )

                row = {
                    "config_label": str(label),
                    "config_path": str(config_path),
                    "episode_idx": int(episode_idx),
                    "seed": int(seed),
                    "step_idx": int(step_idx),
                    "state_index": int(state_counter),
                    "mean_valid_slots": float(prefix["mean_valid_slots"]),
                    "one_step_uniform_x_acc": float(one_uniform["x_acc"]),
                    "one_step_heuristic_x_acc": float(one_heuristic["x_acc"]),
                    "one_step_uniform_d_pre": float(one_uniform["d_pre"]),
                    "one_step_heuristic_d_pre": float(one_heuristic["d_pre"]),
                    "one_step_uniform_pre_backlog": float(one_uniform["pre_backlog_steps_eval"]),
                    "one_step_heuristic_pre_backlog": float(one_heuristic["pre_backlog_steps_eval"]),
                    "one_step_gap_x_acc": float(one_heuristic["x_acc"] - one_uniform["x_acc"]),
                    "one_step_gap_pre_drop": float(one_uniform["d_pre"] - one_heuristic["d_pre"]),
                    "one_step_gap_pre_backlog": float(
                        one_uniform["pre_backlog_steps_eval"] - one_heuristic["pre_backlog_steps_eval"]
                    ),
                    "k2_uniform_x_acc_discounted": float(k2_uniform["x_acc"]),
                    "k2_heuristic_x_acc_discounted": float(k2_heuristic["x_acc"]),
                    "k2_uniform_d_pre_discounted": float(k2_uniform["d_pre"]),
                    "k2_heuristic_d_pre_discounted": float(k2_heuristic["d_pre"]),
                    "k2_uniform_pre_backlog_discounted": float(k2_uniform["pre_backlog_steps_eval"]),
                    "k2_heuristic_pre_backlog_discounted": float(k2_heuristic["pre_backlog_steps_eval"]),
                    "k2_gap_x_acc": float(k2_heuristic["x_acc"] - k2_uniform["x_acc"]),
                    "k2_gap_pre_drop": float(k2_uniform["d_pre"] - k2_heuristic["d_pre"]),
                    "k2_gap_pre_backlog": float(k2_uniform["pre_backlog_steps_eval"] - k2_heuristic["pre_backlog_steps_eval"]),
                    "k2_uniform_steps_executed": float(k2_uniform["steps_executed"]),
                    "k2_heuristic_steps_executed": float(k2_heuristic["steps_executed"]),
                    "one_step_uniform_done": float(uniform_done),
                    "one_step_heuristic_done": float(heuristic_done),
                }
                rows.append(row)

                anchor_bw = prefix[f"{rollout_bw_source}_bw"]
                anchor_result = driver.execute_stage_bw_and_step(np.asarray(anchor_bw, dtype=np.float32))
                done = bool(next(iter(anchor_result.terminations.values())) or next(iter(anchor_result.truncations.values())))
                state_counter += 1
                step_idx += 1
                if progress_every > 0 and state_counter % int(progress_every) == 0:
                    print(
                        f"[{label}] audited_states={state_counter} "
                        f"episode={episode_idx} step={step_idx} done={int(done)}"
                    )
    finally:
        close_structured_env_group(driver)

    def rows_of(key: str) -> list[float]:
        return [float(row[key]) for row in rows]

    summary = {
        "label": str(label),
        "config_path": str(config_path),
        "episodes_requested": int(episodes),
        "episodes_sampled": int(len({int(row['episode_idx']) for row in rows})),
        "state_samples": int(len(rows)),
        "seed_base": int(seed_base),
        "rollout_bw_source": str(rollout_bw_source),
        "one_step": {
            "gap_x_acc": _summarize(rows_of("one_step_gap_x_acc")),
            "gap_pre_drop": _summarize(rows_of("one_step_gap_pre_drop")),
            "gap_pre_backlog": _summarize(rows_of("one_step_gap_pre_backlog")),
            "uniform_x_acc": _summarize(rows_of("one_step_uniform_x_acc")),
            "heuristic_x_acc": _summarize(rows_of("one_step_heuristic_x_acc")),
        },
        "k2_discounted": {
            "gap_x_acc": _summarize(rows_of("k2_gap_x_acc")),
            "gap_pre_drop": _summarize(rows_of("k2_gap_pre_drop")),
            "gap_pre_backlog": _summarize(rows_of("k2_gap_pre_backlog")),
            "uniform_x_acc": _summarize(rows_of("k2_uniform_x_acc_discounted")),
            "heuristic_x_acc": _summarize(rows_of("k2_heuristic_x_acc_discounted")),
        },
        "mean_valid_slots": _summarize(rows_of("mean_valid_slots")),
    }
    return summary, rows


def compare_summaries(old_summary: dict[str, Any], focus_summary: dict[str, Any]) -> dict[str, Any]:
    def ratio(metric_group: str, metric_name: str, stat_name: str) -> float | None:
        numer = float(focus_summary[metric_group][metric_name][stat_name])
        denom = float(old_summary[metric_group][metric_name][stat_name])
        return _safe_ratio(numer, denom)

    compare = {
        "one_step_gap_ratio_focus_over_old": {
            "x_acc_p50": ratio("one_step", "gap_x_acc", "p50"),
            "x_acc_p90": ratio("one_step", "gap_x_acc", "p90"),
            "pre_drop_p50": ratio("one_step", "gap_pre_drop", "p50"),
            "pre_drop_p90": ratio("one_step", "gap_pre_drop", "p90"),
            "pre_backlog_p50": ratio("one_step", "gap_pre_backlog", "p50"),
            "pre_backlog_p90": ratio("one_step", "gap_pre_backlog", "p90"),
        },
        "k2_gap_ratio_focus_over_old": {
            "x_acc_p50": ratio("k2_discounted", "gap_x_acc", "p50"),
            "x_acc_p90": ratio("k2_discounted", "gap_x_acc", "p90"),
            "pre_drop_p50": ratio("k2_discounted", "gap_pre_drop", "p50"),
            "pre_drop_p90": ratio("k2_discounted", "gap_pre_drop", "p90"),
            "pre_backlog_p50": ratio("k2_discounted", "gap_pre_backlog", "p50"),
            "pre_backlog_p90": ratio("k2_discounted", "gap_pre_backlog", "p90"),
        },
        "focus_threshold_checks": {
            "one_step_x_acc_gap_p50_ge_1e-2": float(focus_summary["one_step"]["gap_x_acc"]["p50"] >= 1.0e-2),
            "one_step_x_acc_gap_p90_ge_2e-2": float(focus_summary["one_step"]["gap_x_acc"]["p90"] >= 2.0e-2),
            "k2_x_acc_gap_p50_ge_1e-2": float(focus_summary["k2_discounted"]["gap_x_acc"]["p50"] >= 1.0e-2),
            "k2_x_acc_gap_p90_ge_2e-2": float(focus_summary["k2_discounted"]["gap_x_acc"]["p90"] >= 2.0e-2),
            "one_step_any_gap_ratio_p50_ge_3": float(
                any(
                    (compare_value or 0.0) >= 3.0
                    for compare_value in (
                        ratio("one_step", "gap_x_acc", "p50"),
                        ratio("one_step", "gap_pre_drop", "p50"),
                        ratio("one_step", "gap_pre_backlog", "p50"),
                    )
                )
            ),
            "k2_any_gap_ratio_p50_ge_3": float(
                any(
                    (compare_value or 0.0) >= 3.0
                    for compare_value in (
                        ratio("k2_discounted", "gap_x_acc", "p50"),
                        ratio("k2_discounted", "gap_pre_drop", "p50"),
                        ratio("k2_discounted", "gap_pre_backlog", "p50"),
                    )
                )
            ),
        },
    }
    return compare


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-config", type=str, default=DEFAULT_OLD_CONFIG)
    parser.add_argument("--focus-config", type=str, default=DEFAULT_FOCUS_CONFIG)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--max-state-samples", type=int, default=600)
    parser.add_argument("--seed-base", type=int, default=72000)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--rollout-bw-source", choices=["uniform", "heuristic"], default="heuristic")
    parser.add_argument("--out-dir", type=str, default="runs/bw_focus_leverage_audit_20260406")
    args = parser.parse_args()

    old_summary, old_rows = audit_config(
        config_path=str(args.old_config),
        label="old_env",
        episodes=int(args.episodes),
        seed_base=int(args.seed_base),
        max_state_samples=int(args.max_state_samples),
        progress_every=int(args.progress_every),
        rollout_bw_source=str(args.rollout_bw_source),
    )
    focus_summary, focus_rows = audit_config(
        config_path=str(args.focus_config),
        label="bw_focus_env_v1",
        episodes=int(args.episodes),
        seed_base=int(args.seed_base),
        max_state_samples=int(args.max_state_samples),
        progress_every=int(args.progress_every),
        rollout_bw_source=str(args.rollout_bw_source),
    )
    compare = compare_summaries(old_summary, focus_summary)

    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "old_env_rows.csv", old_rows)
    _write_csv(out_dir / "bw_focus_env_v1_rows.csv", focus_rows)
    payload = {
        "old_env": old_summary,
        "bw_focus_env_v1": focus_summary,
        "compare": compare,
    }
    _write_json(out_dir / "summary.json", payload)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
