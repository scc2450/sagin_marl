from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import pickle
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
from sagin_marl.rl.structured_mappo import (
    _current_obs_list,
    _heuristic_accel,
    _heuristic_bw,
    _heuristic_sat,
    _refresh_stage_obs_cache,
    _sat_mask_to_ids,
)
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


REWARD_GAP_THRESHOLDS = (0.01, 0.05, 0.10)


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _normalize_bw_action(action: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    valid_mask_arr = np.asarray(valid_mask, dtype=bool)
    out = np.zeros(valid_mask_arr.shape, dtype=np.float32)
    action_arr = np.asarray(action, dtype=np.float32)
    for u in range(int(valid_mask_arr.shape[0])):
        valid = valid_mask_arr[u]
        valid_count = int(np.sum(valid))
        if valid_count <= 0:
            continue
        row = np.clip(action_arr[u, valid], 0.0, None)
        row_sum = float(np.sum(row))
        if row_sum <= 1.0e-12:
            out[u, valid] = 1.0 / float(valid_count)
        else:
            out[u, valid] = row / row_sum
    return out


def _uniform_bw_action(valid_mask: np.ndarray) -> np.ndarray:
    valid_mask_arr = np.asarray(valid_mask, dtype=bool)
    out = np.zeros(valid_mask_arr.shape, dtype=np.float32)
    for u in range(int(valid_mask_arr.shape[0])):
        valid = valid_mask_arr[u]
        valid_count = int(np.sum(valid))
        if valid_count > 0:
            out[u, valid] = 1.0 / float(valid_count)
    return out


def _random_bw_action(valid_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    valid_mask_arr = np.asarray(valid_mask, dtype=bool)
    out = np.zeros(valid_mask_arr.shape, dtype=np.float32)
    for u in range(int(valid_mask_arr.shape[0])):
        valid = np.flatnonzero(valid_mask_arr[u])
        if valid.size <= 0:
            continue
        if valid.size == 1:
            out[u, valid[0]] = 1.0
            continue
        weights = rng.dirichlet(np.ones(valid.size, dtype=np.float64)).astype(np.float32)
        out[u, valid] = weights
    return out


def _topk_bw_action_from_scores(scores: np.ndarray, valid_mask: np.ndarray, k: int) -> np.ndarray:
    valid_mask_arr = np.asarray(valid_mask, dtype=bool)
    score_arr = np.asarray(scores, dtype=np.float32)
    out = np.zeros(valid_mask_arr.shape, dtype=np.float32)
    for u in range(int(valid_mask_arr.shape[0])):
        valid_idx = np.flatnonzero(valid_mask_arr[u])
        if valid_idx.size <= 0:
            continue
        row_scores = score_arr[u, valid_idx]
        order = valid_idx[np.argsort(row_scores)[::-1]]
        chosen = order[: max(1, min(int(k), order.size))]
        out[u, chosen] = 1.0 / float(chosen.size)
    return out


def _slot_score_matrix(
    obs_after_accel: list[dict[str, np.ndarray]],
    cfg,
    *,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    num_agents = len(obs_after_accel)
    scores = np.zeros((num_agents, cfg.users_obs_max), dtype=np.float32)
    valid_mask = np.zeros((num_agents, cfg.users_obs_max), dtype=bool)
    for u, obs in enumerate(obs_after_accel):
        users = np.asarray(obs["users"], dtype=np.float32)
        users_mask = np.asarray(obs["users_mask"] > 0.0, dtype=bool)
        bw_valid = np.asarray(obs.get("bw_valid_mask", obs["users_mask"]) > 0.0, dtype=bool)
        valid = users_mask & bw_valid
        valid_mask[u] = valid
        if not np.any(valid):
            continue
        q = np.asarray(users[:, 2], dtype=np.float32)
        eta = np.asarray(users[:, 3], dtype=np.float32)
        prev = np.asarray(users[:, 4], dtype=np.float32)
        if mode == "queue":
            slot_score = np.clip(q, 0.0, None)
        elif mode == "eta":
            slot_score = np.clip(eta, 0.0, None)
        elif mode == "qeta":
            slot_score = np.clip(q, 0.0, None) * (0.5 + np.clip(eta, 0.0, None))
        elif mode == "qeta_prev":
            slot_score = np.clip(q, 0.0, None) * (0.5 + np.clip(eta, 0.0, None)) * (1.0 + 0.3 * np.clip(prev, 0.0, None))
        else:
            raise ValueError(f"Unsupported slot score mode: {mode}")
        score_row = np.zeros((cfg.users_obs_max,), dtype=np.float32)
        score_row[valid] = slot_score[valid]
        scores[u] = score_row
    return scores, valid_mask


def _step_metrics(step_result, env) -> dict[str, float]:
    reward = float(next(iter(step_result.rewards.values())))
    parts = dict(getattr(env, "last_reward_parts", {}) or {})
    return {
        "reward": reward,
        "x_acc": float(parts.get("x_acc", 0.0) or 0.0),
        "x_rel": float(parts.get("x_rel", 0.0) or 0.0),
        "d_pre": float(parts.get("d_pre", 0.0) or 0.0),
        "pre_backlog_steps_eval": float(parts.get("pre_backlog_steps_eval", 0.0) or 0.0),
    }


def _prepare_bw_stage_snapshot(
    driver: StructuredControlDriver,
    cfg,
    *,
    accel_source: str,
    sat_source: str,
) -> tuple[dict[str, Any], Any, list[dict[str, np.ndarray]]]:
    obs_before = _current_obs_list(driver)
    centers = None
    counts = None
    if accel_source == "cluster_center_queue_aware":
        centers = None if getattr(driver.env, "gu_cluster_centers", None) is None else np.asarray(driver.env.gu_cluster_centers)
        counts = None if getattr(driver.env, "gu_cluster_counts", None) is None else np.asarray(driver.env.gu_cluster_counts)
    accel_action = _heuristic_accel(obs_before, cfg, accel_source, centers=centers, counts=counts)
    driver.run_accel_stage(accel_action)
    _refresh_stage_obs_cache(driver)
    obs_after_accel = _current_obs_list(driver)
    sat_mask = _heuristic_sat(obs_after_accel, cfg, sat_source)
    sat_action = _sat_mask_to_ids(driver, sat_mask)
    state_after_sat = driver.run_sat_stage(sat_action)
    snapshot = driver.build_bw_stage_snapshot(state_after_sat)
    exported = driver.export_bw_stage_state()
    return exported, snapshot, obs_after_accel


def _make_policy_independent_panel_actions(
    cfg,
    snapshot,
    obs_after_accel: list[dict[str, np.ndarray]],
    *,
    heuristic_bw_source: str,
    random_count: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    snapshot_valid_mask = np.asarray(snapshot.bw_valid_mask, dtype=bool)
    actions: dict[str, np.ndarray] = {
        "uniform": _uniform_bw_action(snapshot_valid_mask),
        "heuristic": _normalize_bw_action(_heuristic_bw(obs_after_accel, cfg, heuristic_bw_source), snapshot_valid_mask),
    }
    for mode in ("queue", "eta", "qeta", "qeta_prev"):
        scores, valid_mask = _slot_score_matrix(obs_after_accel, cfg, mode=mode)
        valid_mask = valid_mask & snapshot_valid_mask
        for top_k in (1, 2):
            actions[f"{mode}_top{top_k}"] = _topk_bw_action_from_scores(scores, valid_mask, top_k)
    for sample_idx in range(max(int(random_count), 0)):
        actions[f"random_{sample_idx}"] = _random_bw_action(snapshot_valid_mask, rng)
    return actions


def _execute_one_step_from_snapshot(
    driver: StructuredControlDriver,
    snapshot_state: dict[str, Any],
    bw_action: np.ndarray,
) -> tuple[dict[str, float], bool]:
    driver.load_bw_stage_state(snapshot_state)
    step_result = driver.execute_stage_bw_and_step(np.asarray(bw_action, dtype=np.float32))
    done = bool(next(iter(step_result.terminations.values())) or next(iter(step_result.truncations.values())))
    return _step_metrics(step_result, driver.env), done


def _default_followup_bw_action(
    driver: StructuredControlDriver,
    cfg,
    *,
    accel_source: str,
    sat_source: str,
    bw_source: str,
    heuristic_bw_source: str,
    rng: np.random.Generator,
) -> np.ndarray:
    obs_before = _current_obs_list(driver)
    centers = None
    counts = None
    if accel_source == "cluster_center_queue_aware":
        centers = None if getattr(driver.env, "gu_cluster_centers", None) is None else np.asarray(driver.env.gu_cluster_centers)
        counts = None if getattr(driver.env, "gu_cluster_counts", None) is None else np.asarray(driver.env.gu_cluster_counts)
    accel_action = _heuristic_accel(obs_before, cfg, accel_source, centers=centers, counts=counts)
    driver.run_accel_stage(accel_action)
    _refresh_stage_obs_cache(driver)
    obs_after_accel = _current_obs_list(driver)
    sat_mask = _heuristic_sat(obs_after_accel, cfg, sat_source)
    sat_action = _sat_mask_to_ids(driver, sat_mask)
    state_after_sat = driver.run_sat_stage(sat_action)
    snapshot = driver.build_bw_stage_snapshot(state_after_sat)
    valid_mask = np.asarray(snapshot.bw_valid_mask, dtype=bool)
    if bw_source == "uniform":
        return _uniform_bw_action(valid_mask)
    if bw_source == "random":
        return _random_bw_action(valid_mask, rng)
    if bw_source == "heuristic":
        raw = _heuristic_bw(obs_after_accel, cfg, heuristic_bw_source)
        return _normalize_bw_action(raw, valid_mask)
    raise ValueError(f"Unsupported rollout bw source: {bw_source}")


def _rollout_k_from_snapshot(
    driver: StructuredControlDriver,
    snapshot_state: dict[str, Any],
    *,
    first_bw_action: np.ndarray,
    k_steps: int,
    cfg,
    accel_source: str,
    sat_source: str,
    rollout_bw_source: str,
    heuristic_bw_source: str,
    rng: np.random.Generator,
) -> dict[str, float]:
    driver.load_bw_stage_state(snapshot_state)
    gamma = float(cfg.gamma)
    discount = 1.0
    totals = {
        "reward": 0.0,
        "x_acc": 0.0,
        "x_rel": 0.0,
        "d_pre": 0.0,
        "pre_backlog_steps_eval": 0.0,
        "steps_executed": 0.0,
    }
    current_bw_action = np.asarray(first_bw_action, dtype=np.float32)
    for step_idx in range(max(int(k_steps), 0)):
        step_result, _next_world_state = driver.execute_stage_bw_and_prepare_next_accel(current_bw_action)
        metrics = _step_metrics(step_result, driver.env)
        for key in ("reward", "x_acc", "x_rel", "d_pre", "pre_backlog_steps_eval"):
            totals[key] += discount * float(metrics[key])
        totals["steps_executed"] += 1.0
        done = bool(next(iter(step_result.terminations.values())) or next(iter(step_result.truncations.values())))
        if done or step_idx + 1 >= int(k_steps):
            break
        discount *= gamma
        current_bw_action = _default_followup_bw_action(
            driver,
            cfg,
            accel_source=accel_source,
            sat_source=sat_source,
            bw_source=rollout_bw_source,
            heuristic_bw_source=heuristic_bw_source,
            rng=rng,
        )
    return totals


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


def _write_bank(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _restore_consistency_summary(
    cfg,
    entries: list[dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
) -> dict[str, float]:
    if not entries:
        return {"sampled": 0.0, "max_abs_diff": 0.0, "mean_abs_diff": 0.0}
    rng = np.random.default_rng(int(seed))
    chosen_indices = rng.choice(len(entries), size=min(int(sample_size), len(entries)), replace=False)
    env = make_structured_env(cfg, mode="script")
    driver = as_structured_driver(env)
    diffs: list[float] = []
    try:
        for idx in chosen_indices.tolist():
            snapshot_state = entries[int(idx)]["snapshot"]
            snapshot = driver.load_bw_stage_state(snapshot_state)
            bw_action = _uniform_bw_action(np.asarray(snapshot.bw_valid_mask, dtype=bool))
            metrics_a, _ = _execute_one_step_from_snapshot(driver, snapshot_state, bw_action)
            metrics_b, _ = _execute_one_step_from_snapshot(driver, snapshot_state, bw_action)
            diffs.extend(
                [
                    abs(float(metrics_a["reward"]) - float(metrics_b["reward"])),
                    abs(float(metrics_a["x_acc"]) - float(metrics_b["x_acc"])),
                    abs(float(metrics_a["x_rel"]) - float(metrics_b["x_rel"])),
                    abs(float(metrics_a["d_pre"]) - float(metrics_b["d_pre"])),
                    abs(float(metrics_a["pre_backlog_steps_eval"]) - float(metrics_b["pre_backlog_steps_eval"])),
                ]
            )
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
    return {
        "sampled": float(len(chosen_indices)),
        "max_abs_diff": float(np.max(np.asarray(diffs, dtype=np.float64))) if diffs else 0.0,
        "mean_abs_diff": _safe_mean(diffs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-states", type=int, default=240)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed-base", type=int, default=93000)
    parser.add_argument("--bank-size", type=int, default=64)
    parser.add_argument("--restore-checks", type=int, default=16)
    parser.add_argument("--max-states-per-episode", type=int, default=6)
    parser.add_argument("--random-actions", type=int, default=2)
    parser.add_argument("--rollout-accel-source", choices=["queue_aware", "cluster_center_queue_aware", "zero"], default="cluster_center_queue_aware")
    parser.add_argument("--rollout-sat-source", choices=["queue_aware", "cluster_center_queue_aware", "zero"], default="cluster_center_queue_aware")
    parser.add_argument("--heuristic-bw-source", choices=["queue_aware", "cluster_center_queue_aware", "zero"], default="queue_aware")
    parser.add_argument("--rollout-bw-source", choices=["heuristic", "uniform", "random"], default="heuristic")
    parser.add_argument("--k-steps", type=int, default=2)
    parser.add_argument("--out-dir", type=str, default="runs/bw_snapshot_reward_gap_audit_20260406")
    parser.add_argument("--progress-every", type=int, default=40)
    args = parser.parse_args()

    _set_all_seeds(1234)
    cfg = load_config(str(args.config))
    env = make_structured_env(cfg, mode="script")
    sampling_driver = as_structured_driver(env)
    replay_env = make_structured_env(cfg, mode="script")
    replay_driver = as_structured_driver(replay_env)
    panel_rng = np.random.default_rng(20260406)
    rollout_rng = np.random.default_rng(20260407)

    rows: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    action_names_seen: list[str] = []
    state_counter = 0
    try:
        for episode_idx in range(max(int(args.episodes), 0)):
            if state_counter >= int(args.num_states):
                break
            seed = int(args.seed_base) + int(episode_idx)
            env.reset(seed=seed)
            done = False
            step_idx = 0
            states_this_episode = 0
            while not done and state_counter < int(args.num_states):
                if states_this_episode >= int(args.max_states_per_episode):
                    break
                sampling_driver.begin_step()
                snapshot_state, snapshot, obs_after_accel = _prepare_bw_stage_snapshot(
                    sampling_driver,
                    cfg,
                    accel_source=str(args.rollout_accel_source),
                    sat_source=str(args.rollout_sat_source),
                )
                panel_actions = _make_policy_independent_panel_actions(
                    cfg,
                    snapshot,
                    obs_after_accel,
                    heuristic_bw_source=str(args.heuristic_bw_source),
                    random_count=int(args.random_actions),
                    rng=panel_rng,
                )
                if not action_names_seen:
                    action_names_seen = list(panel_actions.keys())

                row: dict[str, Any] = {
                    "state_index": int(state_counter),
                    "episode_idx": int(episode_idx),
                    "seed": int(seed),
                    "step_idx": int(step_idx),
                    "mean_valid_slots": float(np.mean(np.sum(np.asarray(snapshot.bw_valid_mask, dtype=bool), axis=1))),
                }
                one_step_panel: dict[str, dict[str, float]] = {}
                k_panel: dict[str, dict[str, float]] = {}
                for action_name, bw_action in panel_actions.items():
                    one_step_metrics, one_step_done = _execute_one_step_from_snapshot(replay_driver, snapshot_state, bw_action)
                    k_metrics = _rollout_k_from_snapshot(
                        replay_driver,
                        snapshot_state,
                        first_bw_action=bw_action,
                        k_steps=int(args.k_steps),
                        cfg=cfg,
                        accel_source=str(args.rollout_accel_source),
                        sat_source=str(args.rollout_sat_source),
                        rollout_bw_source=str(args.rollout_bw_source),
                        heuristic_bw_source=str(args.heuristic_bw_source),
                        rng=rollout_rng,
                    )
                    one_step_panel[action_name] = one_step_metrics
                    k_panel[action_name] = k_metrics
                    row[f"one_step_{action_name}_reward"] = float(one_step_metrics["reward"])
                    row[f"one_step_{action_name}_x_acc"] = float(one_step_metrics["x_acc"])
                    row[f"one_step_{action_name}_x_rel"] = float(one_step_metrics["x_rel"])
                    row[f"one_step_{action_name}_d_pre"] = float(one_step_metrics["d_pre"])
                    row[f"one_step_{action_name}_pre_backlog"] = float(one_step_metrics["pre_backlog_steps_eval"])
                    row[f"one_step_{action_name}_done"] = float(one_step_done)
                    row[f"k{int(args.k_steps)}_{action_name}_reward"] = float(k_metrics["reward"])
                    row[f"k{int(args.k_steps)}_{action_name}_x_acc"] = float(k_metrics["x_acc"])
                    row[f"k{int(args.k_steps)}_{action_name}_x_rel"] = float(k_metrics["x_rel"])
                    row[f"k{int(args.k_steps)}_{action_name}_d_pre"] = float(k_metrics["d_pre"])
                    row[f"k{int(args.k_steps)}_{action_name}_pre_backlog"] = float(k_metrics["pre_backlog_steps_eval"])
                    row[f"k{int(args.k_steps)}_{action_name}_steps_executed"] = float(k_metrics["steps_executed"])

                for scope_name, panel in (("one_step", one_step_panel), (f"k{int(args.k_steps)}", k_panel)):
                    for metric_name in ("reward", "x_acc", "x_rel", "d_pre", "pre_backlog_steps_eval"):
                        values = [float(metrics[metric_name]) for metrics in panel.values()]
                        row[f"{scope_name}_range_{metric_name if metric_name != 'pre_backlog_steps_eval' else 'pre_backlog'}"] = float(max(values) - min(values))
                row["k2_range_reward"] = float(row[f"k{int(args.k_steps)}_range_reward"])
                row["k2_range_x_acc"] = float(row[f"k{int(args.k_steps)}_range_x_acc"])
                row["k2_range_x_rel"] = float(row[f"k{int(args.k_steps)}_range_x_rel"])
                row["k2_range_d_pre"] = float(row[f"k{int(args.k_steps)}_range_d_pre"])
                row["k2_range_pre_backlog"] = float(row[f"k{int(args.k_steps)}_range_pre_backlog"])
                heuristic_k_reward = float(row[f"k{int(args.k_steps)}_heuristic_reward"])
                row["k2_reward_gap_rel_to_heuristic_abs"] = float(
                    row["k2_range_reward"] / max(abs(heuristic_k_reward), 1.0e-6)
                )
                row["leverage_score"] = float(row["k2_range_reward"])
                rows.append(row)
                states.append(
                    {
                        "row": {
                            "state_index": int(row["state_index"]),
                            "episode_idx": int(row["episode_idx"]),
                            "seed": int(row["seed"]),
                            "step_idx": int(row["step_idx"]),
                            "leverage_score": float(row["leverage_score"]),
                        },
                        "snapshot": snapshot_state,
                    }
                )

                if str(args.rollout_bw_source) == "uniform":
                    anchor_action = panel_actions["uniform"]
                elif str(args.rollout_bw_source) == "random":
                    anchor_action = _random_bw_action(np.asarray(snapshot.bw_valid_mask, dtype=bool), rollout_rng)
                else:
                    anchor_action = panel_actions["heuristic"]
                anchor_result = sampling_driver.execute_stage_bw_and_step(anchor_action)
                done = bool(next(iter(anchor_result.terminations.values())) or next(iter(anchor_result.truncations.values())))
                state_counter += 1
                step_idx += 1
                states_this_episode += 1
                if int(args.progress_every) > 0 and state_counter % int(args.progress_every) == 0:
                    print(
                        f"[audit_bw_snapshot_reward_gap] sampled_states={state_counter} "
                        f"episode={episode_idx} step={step_idx} done={int(done)}"
                    )
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
        close_fn = getattr(replay_env, "close", None)
        if callable(close_fn):
            close_fn()

    if not rows:
        raise RuntimeError("No bw-stage states were collected")

    scores = np.asarray([float(row["leverage_score"]) for row in rows], dtype=np.float64)
    order = np.argsort(-scores)
    bank_size = min(int(args.bank_size), len(rows))
    high_indices = order[:bank_size].tolist()
    random_indices = panel_rng.choice(len(rows), size=bank_size, replace=False).tolist()
    high_bank_entries = [states[idx] for idx in high_indices]
    random_bank_entries = [states[idx] for idx in random_indices]
    k_label = f"k{int(args.k_steps)}"

    summary = {
        "config": str(args.config),
        "state_samples": int(len(rows)),
        "episodes_requested": int(args.episodes),
        "episodes_sampled": int(len({int(row['episode_idx']) for row in rows})),
        "rollout_sources": {
            "accel": str(args.rollout_accel_source),
            "sat": str(args.rollout_sat_source),
            "bw_followup": str(args.rollout_bw_source),
            "heuristic_bw": str(args.heuristic_bw_source),
        },
        "action_panel": list(action_names_seen),
        "one_step_ranges": {
            "reward": _summarize([float(row["one_step_range_reward"]) for row in rows]),
            "x_acc": _summarize([float(row["one_step_range_x_acc"]) for row in rows]),
            "x_rel": _summarize([float(row["one_step_range_x_rel"]) for row in rows]),
            "d_pre": _summarize([float(row["one_step_range_d_pre"]) for row in rows]),
            "pre_backlog": _summarize([float(row["one_step_range_pre_backlog"]) for row in rows]),
        },
        f"{k_label}_ranges": {
            "reward": _summarize([float(row[f"{k_label}_range_reward"]) for row in rows]),
            "x_acc": _summarize([float(row[f"{k_label}_range_x_acc"]) for row in rows]),
            "x_rel": _summarize([float(row[f"{k_label}_range_x_rel"]) for row in rows]),
            "d_pre": _summarize([float(row[f"{k_label}_range_d_pre"]) for row in rows]),
            "pre_backlog": _summarize([float(row[f"{k_label}_range_pre_backlog"]) for row in rows]),
        },
        "reward_gap_relative_to_heuristic_abs": _summarize([float(row["k2_reward_gap_rel_to_heuristic_abs"]) for row in rows]),
        "leverage_score": _summarize([float(row["leverage_score"]) for row in rows]),
        "occupancy": {
            **{
                f"one_step_reward_ge_{str(thresh).replace('.', 'p')}": float(
                    np.mean([float(row["one_step_range_reward"]) >= float(thresh) for row in rows])
                )
                for thresh in REWARD_GAP_THRESHOLDS
            },
            **{
                f"{k_label}_reward_ge_{str(thresh).replace('.', 'p')}": float(
                    np.mean([float(row[f"{k_label}_range_reward"]) >= float(thresh) for row in rows])
                )
                for thresh in REWARD_GAP_THRESHOLDS
            },
        },
        "bank_comparison": {
            "bank_size": int(bank_size),
            "high_reward_gap_mean": _safe_mean([float(states[idx]["row"]["leverage_score"]) for idx in high_indices]),
            "random_reward_gap_mean": _safe_mean([float(states[idx]["row"]["leverage_score"]) for idx in random_indices]),
            "high_reward_gap_p50": float(np.percentile(np.asarray([float(states[idx]["row"]["leverage_score"]) for idx in high_indices], dtype=np.float64), 50.0)) if high_indices else 0.0,
            "random_reward_gap_p50": float(np.percentile(np.asarray([float(states[idx]["row"]["leverage_score"]) for idx in random_indices], dtype=np.float64), 50.0)) if random_indices else 0.0,
        },
    }
    summary["restore_consistency"] = {
        "high_reward_gap_bank": _restore_consistency_summary(cfg, high_bank_entries, sample_size=int(args.restore_checks), seed=123),
        "random_bank": _restore_consistency_summary(cfg, random_bank_entries, sample_size=int(args.restore_checks), seed=456),
    }

    out_dir = Path(args.out_dir)
    rows_path = out_dir / "state_rows.csv"
    summary_path = out_dir / "summary.json"
    high_bank_path = out_dir / "high_reward_gap_bank.pkl.gz"
    random_bank_path = out_dir / "random_bank.pkl.gz"

    _write_csv(rows_path, rows)
    _write_json(summary_path, summary)
    _write_bank(
        high_bank_path,
        {
            "config": str(args.config),
            "rollout_sources": summary["rollout_sources"],
            "entries": high_bank_entries,
        },
    )
    _write_bank(
        random_bank_path,
        {
            "config": str(args.config),
            "rollout_sources": summary["rollout_sources"],
            "entries": random_bank_entries,
        },
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"rows_csv={rows_path}")
    print(f"summary_json={summary_path}")
    print(f"high_bank={high_bank_path}")
    print(f"random_bank={random_bank_path}")


if __name__ == "__main__":
    main()
