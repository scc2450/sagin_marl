from __future__ import annotations

import csv
import json
import math
import os
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
from sagin_marl.rl.action_assembler import assemble_actions
from sagin_marl.rl.baselines import cluster_center_accel_policy, queue_aware_policy
from sagin_marl.rl.policy import ActorNet, batch_flatten_obs
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


RUN_ROOT = Path("runs/phase1_actions/curriculum_formal_u1500_subproc12_t2")
ANALYSIS_DIR = RUN_ROOT / "requested_audit_20260324"

STAGE1_DIR = RUN_ROOT / "stage1_accel"
STAGE2_DIR = RUN_ROOT / "stage2_bw"
STAGE3_DIR = RUN_ROOT / "stage3_sat"

STAGE1_SEED_BASE = 62000
STAGE2_SEED_BASE = 72000
STAGE3_SEED_BASE = 82000
EVAL_EPISODES = 20


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size != y.size or x.size < 2:
        return None
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return None
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return None
    return float(np.dot(x, y) / denom)


def _cosine(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size != y.size or x.size == 0:
        return None
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return None
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return None
    return float(np.dot(x, y) / denom)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_actor(cfg_path: Path, checkpoint_path: Path) -> tuple[Any, ActorNet]:
    cfg = load_config(str(cfg_path))
    env = make_structured_env(cfg, mode="script")
    obs, _ = env.reset(seed=0)
    obs_dim = batch_flatten_obs(list(obs.values()), cfg).shape[1]
    actor = ActorNet(obs_dim, cfg).to(torch.device("cpu"))
    info = load_checkpoint_forgiving(actor, str(checkpoint_path), map_location="cpu", strict=True)
    adapted = int(len(info.get("adapted_keys", [])))
    if adapted:
        print(f"Loaded {checkpoint_path} with {adapted} adapted keys.")
    actor.eval()
    return cfg, actor


def _policy_outputs(cfg, actor: ActorNet, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    obs_list = list(obs.values())
    obs_batch = batch_flatten_obs(obs_list, cfg)
    obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=torch.device("cpu"))
    with torch.no_grad():
        dist_out = actor.forward(obs_tensor)
        dist = actor._build_hybrid_dist(dist_out)
        sample = dist.sample(deterministic=True)
        entropy_parts = dist.entropy()
    return {
        "accel": sample.accel.cpu().numpy(),
        "bw": sample.bw_action.cpu().numpy() if sample.bw_action is not None else None,
        "sat_mask": sample.sat_select_mask.cpu().numpy() if sample.sat_select_mask is not None else None,
        "entropy_accel": entropy_parts["accel"].cpu().numpy(),
        "entropy_bw": entropy_parts["bw"].cpu().numpy() if "bw" in entropy_parts else None,
        "entropy_sat": entropy_parts["sat"].cpu().numpy() if "sat" in entropy_parts else None,
    }


def _assemble_trained_actions(cfg, env: SaginParallelEnv, actor: ActorNet, obs: dict[str, dict[str, np.ndarray]]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    out = _policy_outputs(cfg, actor, obs)
    accel_actions = out["accel"] if str(getattr(cfg, "exec_accel_source", "policy")).strip().lower() == "policy" else np.zeros((len(env.agents), 2), dtype=np.float32)
    bw_values = out["bw"] if str(getattr(cfg, "exec_bw_source", "policy")).strip().lower() == "policy" else None
    sat_values = out["sat_mask"] if str(getattr(cfg, "exec_sat_source", "policy")).strip().lower() == "policy" else None
    actions = assemble_actions(cfg, env.agents, accel_actions, bw_logits=bw_values, sat_logits=sat_values)
    return actions, out


def _assemble_cluster_center_actions(cfg, env: SaginParallelEnv, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    obs_list = list(obs.values())
    accel_actions = cluster_center_accel_policy(
        obs_list,
        cfg,
        getattr(env, "gu_cluster_centers", None),
        getattr(env, "gu_cluster_counts", None),
    )
    return assemble_actions(cfg, env.agents, accel_actions, bw_logits=None, sat_logits=None)


def _stage1_step_dump() -> dict[str, Any]:
    cfg_path = STAGE1_DIR / "config_source.yaml"
    ckpt_path = STAGE1_DIR / "actor.pt"
    cfg, actor = _load_actor(cfg_path, ckpt_path)

    policy_modes = ("trained", "cluster_center")
    dump_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for mode in policy_modes:
        mode_cfg = load_config(str(cfg_path))
        if mode == "cluster_center":
            mode_cfg.avoidance_enabled = True
            mode_cfg.pairwise_hard_filter_enabled = True
        env = make_structured_env(mode_cfg, mode="script")
        for ep in range(EVAL_EPISODES):
            seed = STAGE1_SEED_BASE + ep
            obs, _ = env.reset(seed=seed)
            done = False
            reward_sum = 0.0
            steps = 0
            while not done:
                uav_pos_before = np.asarray(env.uav_pos, dtype=np.float64).copy()
                uav_queue_before = np.asarray(env.uav_queue, dtype=np.float64).copy()
                if mode == "trained":
                    actions, _ = _assemble_trained_actions(mode_cfg, env, actor, obs)
                else:
                    actions = _assemble_cluster_center_actions(mode_cfg, env, obs)
                obs, rewards, terms, truncs, _ = env.step(actions)
                reward = float(list(rewards.values())[0])
                terminated = bool(list(terms.values())[0])
                truncated = bool(list(truncs.values())[0])
                done = terminated or truncated
                parts = dict(getattr(env, "last_reward_parts", {}) or {})
                valid_assoc = np.asarray(env.last_association >= 0, dtype=bool)
                if np.any(valid_assoc):
                    assoc_count_vec = np.bincount(
                        np.asarray(env.last_association[valid_assoc], dtype=np.int32),
                        minlength=mode_cfg.num_uav,
                    ).astype(np.int64)
                    gu_outflow_vec = np.bincount(
                        np.asarray(env.last_association[valid_assoc], dtype=np.int32),
                        weights=np.asarray(env.last_gu_outflow[valid_assoc], dtype=np.float64),
                        minlength=mode_cfg.num_uav,
                    ).astype(np.float64)
                else:
                    assoc_count_vec = np.zeros((mode_cfg.num_uav,), dtype=np.int64)
                    gu_outflow_vec = np.zeros((mode_cfg.num_uav,), dtype=np.float64)
                backhaul_vec = (
                    uav_queue_before
                    + gu_outflow_vec
                    - np.asarray(env.uav_queue, dtype=np.float64)
                    - np.asarray(env.uav_drop, dtype=np.float64)
                )
                backhaul_vec = np.maximum(backhaul_vec, 0.0)
                dump_rows.append(
                    {
                        "policy": mode,
                        "episode": ep,
                        "seed": seed,
                        "step": steps,
                        "reward": reward,
                        "reward_raw": _to_float(parts.get("reward_raw")),
                        "x_acc": _to_float(parts.get("x_acc")),
                        "x_rel": _to_float(parts.get("x_rel")),
                        "g_pre": _to_float(parts.get("g_pre")),
                        "d_pre": _to_float(parts.get("d_pre")),
                        "throughput_access_norm": _to_float(parts.get("throughput_access_norm")),
                        "throughput_backhaul_norm": _to_float(parts.get("throughput_backhaul_norm")),
                        "queue_total_active": _to_float(parts.get("queue_total_active")),
                        "assoc_ratio": _to_float(parts.get("assoc_ratio")),
                        "gu_queue_sum": float(np.sum(env.gu_queue)),
                        "uav_queue_sum": float(np.sum(env.uav_queue)),
                        "sat_queue_sum": float(np.sum(env.sat_queue)),
                        "connected_sat_count": _to_float(getattr(env, "last_connected_sat_count", 0.0)),
                        "collision_event": _to_float(parts.get("collision_event")),
                        "terminated": float(terminated),
                        "truncated": float(truncated),
                    }
                )
                for u in range(mode_cfg.num_uav):
                    assoc_mask_u = np.asarray(env.last_association == u, dtype=bool)
                    if np.any(assoc_mask_u):
                        centroid = np.mean(np.asarray(env.gu_pos[assoc_mask_u], dtype=np.float64), axis=0)
                        centroid_x = float(centroid[0])
                        centroid_y = float(centroid[1])
                        centroid_valid = 1
                    else:
                        centroid_x = None
                        centroid_y = None
                        centroid_valid = 0
                    geometry_rows.append(
                        {
                            "policy": mode,
                            "episode": ep,
                            "seed": seed,
                            "step": steps,
                            "uav": u,
                            "uav_pos_x_before": float(uav_pos_before[u, 0]),
                            "uav_pos_y_before": float(uav_pos_before[u, 1]),
                            "uav_pos_x_after": float(env.uav_pos[u, 0]),
                            "uav_pos_y_after": float(env.uav_pos[u, 1]),
                            "assoc_count": int(assoc_count_vec[u]),
                            "gu_outflow_i_bits": float(gu_outflow_vec[u]),
                            "backhaul_i_bits": float(backhaul_vec[u]),
                            "assoc_centroid_valid": centroid_valid,
                            "assoc_centroid_x": centroid_x,
                            "assoc_centroid_y": centroid_y,
                        }
                    )
                reward_sum += reward
                steps += 1
            summary_rows.append(
                {
                    "policy": mode,
                    "episode": ep,
                    "seed": seed,
                    "reward_sum": reward_sum,
                    "steps": steps,
                    "final_queue_total_active": float(np.sum(env.gu_queue) + np.sum(env.uav_queue)),
                    "final_sat_queue_sum": float(np.sum(env.sat_queue)),
                }
            )

    _write_csv(
        ANALYSIS_DIR / "stage1_trained_vs_cluster_center_step_dump.csv",
        dump_rows,
        [
            "policy",
            "episode",
            "seed",
            "step",
            "reward",
            "reward_raw",
            "x_acc",
            "x_rel",
            "g_pre",
            "d_pre",
            "throughput_access_norm",
            "throughput_backhaul_norm",
            "queue_total_active",
            "assoc_ratio",
            "gu_queue_sum",
            "uav_queue_sum",
            "sat_queue_sum",
            "connected_sat_count",
            "collision_event",
            "terminated",
            "truncated",
        ],
    )
    _write_csv(
        ANALYSIS_DIR / "stage1_trained_vs_cluster_center_episode_summary.csv",
        summary_rows,
        [
            "policy",
            "episode",
            "seed",
            "reward_sum",
            "steps",
            "final_queue_total_active",
            "final_sat_queue_sum",
        ],
    )
    _write_csv(
        ANALYSIS_DIR / "stage1_trained_vs_cluster_center_geometry_dump.csv",
        geometry_rows,
        [
            "policy",
            "episode",
            "seed",
            "step",
            "uav",
            "uav_pos_x_before",
            "uav_pos_y_before",
            "uav_pos_x_after",
            "uav_pos_y_after",
            "assoc_count",
            "gu_outflow_i_bits",
            "backhaul_i_bits",
            "assoc_centroid_valid",
            "assoc_centroid_x",
            "assoc_centroid_y",
        ],
    )

    trained_rewards = [row["reward_sum"] for row in summary_rows if row["policy"] == "trained"]
    cluster_rewards = [row["reward_sum"] for row in summary_rows if row["policy"] == "cluster_center"]
    reward_gap = [a - b for a, b in zip(trained_rewards, cluster_rewards, strict=True)]
    return {
        "trained_reward_sum": _summary(trained_rewards),
        "cluster_center_reward_sum": _summary(cluster_rewards),
        "trained_minus_cluster_center_reward_sum": _summary(reward_gap),
        "output_step_dump": str(ANALYSIS_DIR / "stage1_trained_vs_cluster_center_step_dump.csv"),
        "output_geometry_dump": str(ANALYSIS_DIR / "stage1_trained_vs_cluster_center_geometry_dump.csv"),
        "output_episode_summary": str(ANALYSIS_DIR / "stage1_trained_vs_cluster_center_episode_summary.csv"),
    }


def _stage2_stats() -> dict[str, Any]:
    cfg_path = STAGE2_DIR / "config_source.yaml"
    ckpt_path = STAGE2_DIR / "actor.pt"
    cfg, actor = _load_actor(cfg_path, ckpt_path)
    env = make_structured_env(cfg, mode="script")
    sample_rows: list[dict[str, Any]] = []
    valid_counts: list[float] = []
    entropy_bw_values: list[float] = []
    cosine_values: list[float] = []
    l1_values: list[float] = []
    trained_flat: list[float] = []
    qaware_flat: list[float] = []

    for ep in range(EVAL_EPISODES):
        seed = STAGE2_SEED_BASE + ep
        obs, _ = env.reset(seed=seed)
        done = False
        step_idx = 0
        while not done:
            obs_list = list(obs.values())
            actions, out = _assemble_trained_actions(cfg, env, actor, obs)
            qaware_accel, qaware_bw, _ = queue_aware_policy(obs_list, cfg)
            del qaware_accel
            entropy_bw = out["entropy_bw"]
            trained_bw = out["bw"]

            for u, agent in enumerate(env.agents):
                valid_mask = np.asarray(obs[agent].get("bw_valid_mask", obs[agent]["users_mask"]) > 0.0, dtype=bool)
                valid_count = int(np.sum(valid_mask))
                valid_counts.append(float(valid_count))
                entropy_value = _to_float(entropy_bw[u]) if entropy_bw is not None else 0.0
                entropy_bw_values.append(entropy_value)

                trained_vec = np.asarray(trained_bw[u], dtype=np.float64) if trained_bw is not None else np.zeros((cfg.users_obs_max,), dtype=np.float64)
                qaware_vec = np.asarray(qaware_bw[u], dtype=np.float64)
                trained_valid = trained_vec[valid_mask]
                qaware_valid = qaware_vec[valid_mask]
                cosine_value = _cosine(trained_valid, qaware_valid)
                l1_value = float(np.sum(np.abs(trained_valid - qaware_valid))) if trained_valid.size else None
                pearson_value = _pearson(trained_valid, qaware_valid)

                if trained_valid.size:
                    trained_flat.extend(trained_valid.tolist())
                    qaware_flat.extend(qaware_valid.tolist())
                if cosine_value is not None:
                    cosine_values.append(cosine_value)
                if l1_value is not None:
                    l1_values.append(l1_value)

                sample_rows.append(
                    {
                        "episode": ep,
                        "seed": seed,
                        "step": step_idx,
                        "uav": u,
                        "bw_valid_count": valid_count,
                        "entropy_bw": entropy_value,
                        "bw_vs_queue_aware_pearson_valid": pearson_value,
                        "bw_vs_queue_aware_cosine_valid": cosine_value,
                        "bw_vs_queue_aware_l1_valid": l1_value,
                    }
                )

            obs, _, terms, truncs, _ = env.step(actions)
            done = bool(list(terms.values())[0] or list(truncs.values())[0])
            step_idx += 1

    entropy_curve_rows: list[dict[str, Any]] = []
    with (STAGE2_DIR / "metrics.csv").open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entropy_curve_rows.append(
                {
                    "step": int(row["step"]),
                    "entropy_bw": _to_float(row.get("entropy_bw")),
                }
            )

    _write_csv(
        ANALYSIS_DIR / "stage2_bw_eval_samples.csv",
        sample_rows,
        [
            "episode",
            "seed",
            "step",
            "uav",
            "bw_valid_count",
            "entropy_bw",
            "bw_vs_queue_aware_pearson_valid",
            "bw_vs_queue_aware_cosine_valid",
            "bw_vs_queue_aware_l1_valid",
        ],
    )
    _write_csv(
        ANALYSIS_DIR / "stage2_entropy_bw_curve.csv",
        entropy_curve_rows,
        ["step", "entropy_bw"],
    )

    overall_pearson = _pearson(np.asarray(trained_flat, dtype=np.float64), np.asarray(qaware_flat, dtype=np.float64))
    return {
        "bw_valid_count": _summary(valid_counts),
        "entropy_bw_eval": _summary(entropy_bw_values),
        "entropy_bw_train_curve_last": entropy_curve_rows[-1] if entropy_curve_rows else None,
        "bw_vs_queue_aware_overall_valid_slot_pearson": overall_pearson,
        "bw_vs_queue_aware_agent_step_cosine": _summary(cosine_values),
        "bw_vs_queue_aware_agent_step_l1": _summary(l1_values),
        "output_eval_samples": str(ANALYSIS_DIR / "stage2_bw_eval_samples.csv"),
        "output_entropy_curve": str(ANALYSIS_DIR / "stage2_entropy_bw_curve.csv"),
    }


def _select_stage3_satellite_ids(cfg, env: SaginParallelEnv, actions: dict[str, dict[str, np.ndarray]], sat_pos: np.ndarray, sat_vel: np.ndarray) -> list[tuple[int, int, int, float, bool]]:
    selected: list[tuple[int, int, int, float, bool]] = []
    sat_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 0)
    for u, agent in enumerate(env.agents):
        cand = list(getattr(env, "last_visible_candidates", [[] for _ in env.agents])[u][: cfg.sats_obs_max])
        if not cand:
            continue
        sat_raw = np.asarray(actions[agent].get("sat_select_mask", np.zeros((cfg.sats_obs_max,), dtype=np.float32)), dtype=np.float32)[: len(cand)]
        valid_flags = np.ones((len(cand),), dtype=bool)
        if cfg.doppler_enabled:
            for i, sat_id in enumerate(cand):
                valid_flags[i] = abs(env._doppler(u, sat_id, sat_pos, sat_vel)) <= cfg.nu_max
        valid_slots = np.flatnonzero(valid_flags)
        if valid_slots.size == 0:
            continue
        chosen_slots = np.flatnonzero((sat_raw > 0.5) & valid_flags)
        if chosen_slots.size > cfg.N_RF:
            order = np.argsort(-sat_raw[chosen_slots], kind="stable")
            chosen_slots = chosen_slots[order[: cfg.N_RF]]
        if chosen_slots.size == 0:
            best_slot = int(valid_slots[int(np.argmax(sat_raw[valid_slots]))])
            chosen_slots = np.array([best_slot], dtype=np.int64)
        chosen_slots = chosen_slots[: cfg.N_RF]
        if sat_k > 0:
            chosen_slots = chosen_slots[:sat_k]
        for slot in chosen_slots.tolist():
            sat_id = cand[int(slot)]
            doppler_hz = float(env._doppler(u, sat_id, sat_pos, sat_vel))
            overlimit = abs(doppler_hz) > float(cfg.nu_max)
            selected.append((u, int(slot), int(sat_id), doppler_hz, overlimit))
    return selected


def _stage3_stats() -> dict[str, Any]:
    cfg_path = STAGE3_DIR / "config_source.yaml"
    ckpt_path = STAGE3_DIR / "actor.pt"
    cfg, actor = _load_actor(cfg_path, ckpt_path)
    env = make_structured_env(cfg, mode="script")
    visible_rows: list[dict[str, Any]] = []
    sat_sample_rows: list[dict[str, Any]] = []
    raw_counts: list[float] = []
    kept_counts: list[float] = []
    selected_overlimit: list[float] = []
    sat_queue_selected: list[float] = []
    sat_processed_selected: list[float] = []
    sat_selection_counts = np.zeros((cfg.num_sat,), dtype=np.int64)

    for ep in range(EVAL_EPISODES):
        seed = STAGE3_SEED_BASE + ep
        obs, _ = env.reset(seed=seed)
        done = False
        step_idx = 0
        while not done:
            sat_pos, sat_vel = env._get_orbit_states()
            for u in range(cfg.num_uav):
                raw_count = int(getattr(env, "last_visible_raw_counts", np.zeros((cfg.num_uav,), dtype=np.int32))[u])
                kept_count = int(getattr(env, "last_visible_kept_counts", np.zeros((cfg.num_uav,), dtype=np.int32))[u])
                raw_counts.append(float(raw_count))
                kept_counts.append(float(kept_count))
                visible_rows.append(
                    {
                        "episode": ep,
                        "seed": seed,
                        "step": step_idx,
                        "uav": u,
                        "visible_raw_count": raw_count,
                        "visible_kept_count": kept_count,
                    }
                )

            actions, _ = _assemble_trained_actions(cfg, env, actor, obs)
            selected_links = _select_stage3_satellite_ids(cfg, env, actions, sat_pos, sat_vel)
            pre_step_sat_queue = np.asarray(env.sat_queue, dtype=np.float64).copy()
            obs, _, terms, truncs, _ = env.step(actions)
            post_step_processed = np.asarray(getattr(env, "last_sat_processed", np.zeros((cfg.num_sat,), dtype=np.float32)), dtype=np.float64)

            for u, slot, sat_id, doppler_hz, overlimit in selected_links:
                sat_selection_counts[sat_id] += 1
                selected_overlimit.append(1.0 if overlimit else 0.0)
                sat_queue_before = float(pre_step_sat_queue[sat_id])
                processed_bits_after = float(post_step_processed[sat_id])
                sat_queue_selected.append(sat_queue_before)
                sat_processed_selected.append(processed_bits_after)
                sat_sample_rows.append(
                    {
                        "episode": ep,
                        "seed": seed,
                        "step": step_idx,
                        "uav": u,
                        "slot": slot,
                        "sat_id": sat_id,
                        "doppler_hz": doppler_hz,
                        "doppler_overlimit": int(overlimit),
                        "sat_queue_before_bits": sat_queue_before,
                        "processed_bits_after": processed_bits_after,
                    }
                )

            done = bool(list(terms.values())[0] or list(truncs.values())[0])
            step_idx += 1

    sat_freq_rows = []
    total_selected = int(np.sum(sat_selection_counts))
    for sat_id, count in enumerate(sat_selection_counts.tolist()):
        if count <= 0:
            continue
        sat_freq_rows.append(
            {
                "sat_id": sat_id,
                "selected_count": count,
                "selected_fraction": float(count) / float(max(total_selected, 1)),
            }
        )
    sat_freq_rows.sort(key=lambda row: (-row["selected_count"], row["sat_id"]))

    _write_csv(
        ANALYSIS_DIR / "stage3_visible_sat_counts.csv",
        visible_rows,
        ["episode", "seed", "step", "uav", "visible_raw_count", "visible_kept_count"],
    )
    _write_csv(
        ANALYSIS_DIR / "stage3_selected_sat_samples.csv",
        sat_sample_rows,
        [
            "episode",
            "seed",
            "step",
            "uav",
            "slot",
            "sat_id",
            "doppler_hz",
            "doppler_overlimit",
            "sat_queue_before_bits",
            "processed_bits_after",
        ],
    )
    _write_csv(
        ANALYSIS_DIR / "stage3_sat_selection_frequency.csv",
        sat_freq_rows,
        ["sat_id", "selected_count", "selected_fraction"],
    )

    return {
        "visible_raw_count": _summary(raw_counts),
        "visible_kept_count": _summary(kept_counts),
        "selected_doppler_overlimit_ratio": float(np.mean(np.asarray(selected_overlimit, dtype=np.float64))) if selected_overlimit else None,
        "selected_sat_queue_before_bits": _summary(sat_queue_selected),
        "selected_processed_bits_after": _summary(sat_processed_selected),
        "top_selected_satellites": sat_freq_rows[:10],
        "output_visible_counts": str(ANALYSIS_DIR / "stage3_visible_sat_counts.csv"),
        "output_selected_sat_samples": str(ANALYSIS_DIR / "stage3_selected_sat_samples.csv"),
        "output_sat_frequency": str(ANALYSIS_DIR / "stage3_sat_selection_frequency.csv"),
    }


def _combine_checkpoint_eval_curves() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    stage_counts: dict[str, int] = {}
    for stage_name, stage_dir in (
        ("stage1_accel", STAGE1_DIR),
        ("stage2_bw", STAGE2_DIR),
        ("stage3_sat", STAGE3_DIR),
    ):
        path = stage_dir / "checkpoint_eval.csv"
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                out = {"stage": stage_name}
                out.update(row)
                rows.append(out)
                count += 1
            stage_counts[stage_name] = count
    fieldnames = list(rows[0].keys()) if rows else ["stage"]
    _write_csv(ANALYSIS_DIR / "checkpoint_eval_all_stages.csv", rows, fieldnames)
    return {
        "rows_per_stage": stage_counts,
        "output_curve": str(ANALYSIS_DIR / "checkpoint_eval_all_stages.csv"),
    }


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "run_root": str(RUN_ROOT),
        "analysis_dir": str(ANALYSIS_DIR),
        "stage1_same_seed_dump": _stage1_step_dump(),
        "stage2_stats": _stage2_stats(),
        "stage3_stats": _stage3_stats(),
        "checkpoint_eval_curves": _combine_checkpoint_eval_curves(),
    }
    _write_json(ANALYSIS_DIR / "requested_audit_summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
