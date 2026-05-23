from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.rl.baselines import (
    cluster_center_accel_policy,
    queue_aware_bw_policy,
    queue_aware_policy,
    queue_aware_sat_policy,
)
from sagin_marl.rl.structured_parallel_eval import (
    batched_policy_accel_actions,
    batched_policy_bw_outputs,
    batched_policy_sat_subset_indices,
    cluster_meta_many,
    current_obs_many,
    last_reward_parts_many,
    looks_like_driver_group,
    refresh_stage_obs_cache_many,
    reset_at,
    reset_many,
    sat_mask_to_ids_many,
)
from sagin_marl.rl.structured_train import (
    close_structured_env_group,
    make_structured_driver,
    make_structured_driver_group,
)
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import _collate_dataclass
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving


SOURCE_POLICY = "policy"
SOURCE_HEURISTIC = "heuristic"
SMALLER_IS_BETTER = {
    "drop_ratio_eval",
    "pre_backlog_steps_eval",
    "D_sys_report",
    "assoc_centroid_dist_norm_mean",
    "collision_episode_fraction",
}
PRIMARY_GLOBAL_METRICS = [
    "reward_sum",
    "processed_ratio_eval",
    "drop_ratio_eval",
    "pre_backlog_steps_eval",
    "x_acc_mean",
    "x_rel_mean",
    "throughput_access_norm_mean",
    "throughput_backhaul_norm_mean",
    "collision_episode_fraction",
]
HEAD_LOCAL_METRICS = {
    "accel": [
        "x_acc_mean",
        "processed_ratio_eval",
        "drop_ratio_eval",
        "pre_backlog_steps_eval",
        "assoc_centroid_dist_norm_mean",
        "collision_episode_fraction",
    ],
    "sat": [
        "x_rel_mean",
        "throughput_backhaul_norm_mean",
        "sat_processed_incoming_ratio_mean",
        "processed_ratio_eval",
        "drop_ratio_eval",
        "pre_backlog_steps_eval",
    ],
    "bw": [
        "x_acc_mean",
        "throughput_access_norm_mean",
        "processed_ratio_eval",
        "drop_ratio_eval",
        "pre_backlog_steps_eval",
    ],
}


def _resolve_checkpoint(run_dir: Path, checkpoint: str | None, update: int | None) -> Path:
    if checkpoint:
        return Path(checkpoint)
    if update is not None:
        return run_dir / f"actor_u{int(update):04d}.pt"
    final_ckpt = run_dir / "actor_final.pt"
    if final_ckpt.exists():
        return final_ckpt
    return run_dir / "actor.pt"


def _load_actor(cfg, checkpoint: Path, *, hidden_dim: int, embed_dim: int, device: torch.device):
    bundle = build_structured_modules_from_config(cfg, hidden_dim=hidden_dim, embed_dim=embed_dim)
    load_checkpoint_forgiving(bundle.actor, str(checkpoint), map_location=device, strict=False)
    bundle.actor.to(device).eval()
    return bundle.actor


def _combo_name(sources: Dict[str, str]) -> str:
    return f"a={sources['accel']}_s={sources['sat']}_b={sources['bw']}"


def _build_combos(mode: str) -> List[Dict[str, str]]:
    if mode == "minimal":
        return [
            {"accel": SOURCE_POLICY, "sat": SOURCE_POLICY, "bw": SOURCE_POLICY},
            {"accel": SOURCE_HEURISTIC, "sat": SOURCE_HEURISTIC, "bw": SOURCE_HEURISTIC},
            {"accel": SOURCE_POLICY, "sat": SOURCE_HEURISTIC, "bw": SOURCE_HEURISTIC},
            {"accel": SOURCE_HEURISTIC, "sat": SOURCE_POLICY, "bw": SOURCE_HEURISTIC},
            {"accel": SOURCE_HEURISTIC, "sat": SOURCE_HEURISTIC, "bw": SOURCE_POLICY},
        ]
    if mode == "bw_focus":
        return [
            {"accel": SOURCE_POLICY, "sat": SOURCE_POLICY, "bw": SOURCE_POLICY},
            {"accel": SOURCE_HEURISTIC, "sat": SOURCE_HEURISTIC, "bw": SOURCE_HEURISTIC},
            {"accel": SOURCE_HEURISTIC, "sat": SOURCE_HEURISTIC, "bw": SOURCE_POLICY},
        ]
    if mode == "all8":
        return [
            {"accel": accel, "sat": sat, "bw": bw}
            for accel, sat, bw in product((SOURCE_POLICY, SOURCE_HEURISTIC), repeat=3)
        ]
    raise ValueError(f"Unsupported combo_set: {mode}")


def _heuristic_accel(
    obs_list,
    cfg,
    heuristic_policy: str,
    *,
    centers: np.ndarray | None = None,
    counts: np.ndarray | None = None,
) -> np.ndarray:
    if heuristic_policy == "cluster_center_queue_aware":
        return cluster_center_accel_policy(obs_list, cfg, centers, counts)
    if heuristic_policy == "queue_aware":
        accel, _, _ = queue_aware_policy(obs_list, cfg)
        return accel
    raise ValueError(f"Unsupported heuristic policy: {heuristic_policy}")


def _cluster_accel_from_world_state(
    world_state: Any,
    cfg,
    *,
    centers: np.ndarray | None = None,
    counts: np.ndarray | None = None,
) -> np.ndarray:
    uav_nodes = np.asarray(world_state.uav_nodes, dtype=np.float32)
    if uav_nodes.ndim != 3 or uav_nodes.shape[0] <= 0:
        raise ValueError("world_state.uav_nodes must have shape (1, num_uav, feat_dim)")
    obs_list: list[dict[str, np.ndarray]] = []
    for u in range(int(cfg.num_uav)):
        own = np.zeros((10,), dtype=np.float32)
        width = min(int(uav_nodes.shape[-1]), own.size)
        own[:width] = uav_nodes[0, u, :width]
        obs_list.append({"own": own})
    return cluster_center_accel_policy(obs_list, cfg, centers, counts)


def _heuristic_sat(obs_list, cfg, heuristic_policy: str) -> np.ndarray:
    if heuristic_policy in {"cluster_center_queue_aware", "queue_aware"}:
        return queue_aware_sat_policy(obs_list, cfg)
    raise ValueError(f"Unsupported heuristic policy: {heuristic_policy}")


def _heuristic_bw(obs_list, cfg, heuristic_policy: str) -> np.ndarray:
    if heuristic_policy in {"cluster_center_queue_aware", "queue_aware"}:
        return queue_aware_bw_policy(obs_list, cfg)
    raise ValueError(f"Unsupported heuristic policy: {heuristic_policy}")


def _refresh_stage_obs_cache(driver) -> None:
    if driver._stage_assoc is None or driver._stage_candidates is None:
        raise RuntimeError("run_accel_stage must be called before refreshing stage obs cache")
    env = driver.env
    env._cached_assoc = driver._stage_assoc.copy()
    env._cached_candidates = [list(c) for c in driver._stage_candidates]
    if driver._stage_bw_valid_mask is not None:
        env._cached_bw_valid_mask = driver._stage_bw_valid_mask.copy()
    dummy_actions = env._dummy_actions()
    _, env._cached_eta = env._compute_access_rates(
        driver._stage_assoc,
        driver._stage_candidates,
        dummy_actions,
        record_exec=False,
    )
    if driver._stage_sat_pos is not None and driver._stage_sat_vel is not None and driver._stage_visible is not None:
        env._cache_sat_obs(driver._stage_sat_pos, driver._stage_sat_vel, driver._stage_visible)


def _current_obs_list(env) -> List[Dict[str, np.ndarray]]:
    return [env._get_obs(i) for i in range(len(env.agents))]


def _heuristic_sat_mask_to_ids(driver, sat_mask: np.ndarray) -> np.ndarray:
    cfg = driver.env.cfg
    select_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 1)
    out = np.full((cfg.num_uav, select_k), -1, dtype=np.int64)
    if driver._stage_visible is None:
        raise RuntimeError("run_accel_stage must be called before decoding heuristic sat mask")
    for u in range(cfg.num_uav):
        visible = driver._stage_visible[u][: cfg.sats_obs_max]
        active_slots = np.flatnonzero(np.asarray(sat_mask[u], dtype=np.float32) > 0.5)
        mapped: List[int] = []
        for slot in active_slots.tolist():
            if 0 <= int(slot) < len(visible):
                sat_idx = int(visible[int(slot)])
                if sat_idx not in mapped:
                    mapped.append(sat_idx)
            if len(mapped) >= select_k:
                break
        if mapped:
            out[u, : len(mapped)] = np.asarray(mapped[:select_k], dtype=np.int64)
    return out


def _run_hybrid_step(
    actor,
    driver,
    *,
    device: torch.device,
    deterministic: bool,
    sources: Dict[str, str],
    heuristic_policy: str,
):
    env = driver.env

    z_accel = driver.begin_step()
    if sources["accel"] == SOURCE_POLICY:
        accel_states = driver.build_local_accel_states(z_accel)
        accel_batch = _collate_dataclass(accel_states, device)
        accel_out = actor.act_accel(accel_batch, deterministic=deterministic)
        accel_action = accel_out.action.detach().cpu().numpy()
    else:
        if heuristic_policy == "cluster_center_queue_aware":
            accel_action = _cluster_accel_from_world_state(
                z_accel,
                env.cfg,
                centers=getattr(env, "gu_cluster_centers", None),
                counts=getattr(env, "gu_cluster_counts", None),
            )
        else:
            obs_start = _current_obs_list(env)
            accel_action = _heuristic_accel(obs_start, env.cfg, heuristic_policy)

    z_sat = driver.run_accel_stage(accel_action)
    _refresh_stage_obs_cache(driver)
    obs_after_accel = _current_obs_list(env)

    if sources["sat"] == SOURCE_POLICY:
        sat_states = driver.build_local_sat_states(z_sat)
        sat_batch = _collate_dataclass(sat_states, device)
        sat_out = actor.act_sat(sat_batch, deterministic=deterministic)
        sat_action = driver.decode_sat_subset_actions(sat_states, sat_out.subset_index.detach().cpu().tolist())
    else:
        sat_mask = _heuristic_sat(obs_after_accel, env.cfg, heuristic_policy)
        sat_action = _heuristic_sat_mask_to_ids(driver, sat_mask)

    z_bw = driver.run_sat_stage(sat_action)
    if sources["bw"] == SOURCE_POLICY:
        bw_states = driver.build_bw_valid_context(z_bw)
        bw_batch = _collate_dataclass(bw_states, device)
        bw_out = actor.act_bw(bw_batch, deterministic=deterministic)
        bw_action = bw_out.action.detach().cpu().numpy()
    else:
        bw_action = _heuristic_bw(obs_after_accel, env.cfg, heuristic_policy)

    return driver.execute_stage_bw_and_step(bw_action)


def _episode_metrics_template() -> Dict[str, float]:
    return {
        "reward_sum": 0.0,
        "processed_ratio_eval": 0.0,
        "drop_ratio_eval": 0.0,
        "pre_backlog_steps_eval": 0.0,
        "D_sys_report": 0.0,
        "x_acc_mean": 0.0,
        "x_rel_mean": 0.0,
        "g_pre_mean": 0.0,
        "d_pre_mean": 0.0,
        "throughput_access_norm_mean": 0.0,
        "throughput_backhaul_norm_mean": 0.0,
        "sat_processed_norm_mean": 0.0,
        "sat_processed_incoming_ratio_mean": 0.0,
        "assoc_centroid_dist_norm_mean": 0.0,
        "assoc_centroid_valid_uav_frac": 0.0,
        "collision_episode_fraction": 0.0,
    }


def _evaluate_combo(
    cfg,
    actor,
    *,
    device: torch.device,
    episodes: int,
    episode_seed_base: int | None,
    deterministic: bool,
    sources: Dict[str, str],
    heuristic_policy: str,
) -> tuple[Dict[str, float], List[Dict[str, float]]]:
    driver = make_structured_driver(cfg, backend="sync")
    rows: List[Dict[str, float]] = []
    totals = _episode_metrics_template()
    try:
        for ep in range(int(episodes)):
            seed = None if episode_seed_base is None else int(episode_seed_base) + ep
            driver.env.reset(seed=seed)
            done = False
            steps = 0
            reward_sum = 0.0
            processed_ratio_sum = 0.0
            drop_ratio_sum = 0.0
            pre_backlog_sum = 0.0
            d_sys_sum = 0.0
            x_acc_sum = 0.0
            x_rel_sum = 0.0
            g_pre_sum = 0.0
            d_pre_sum = 0.0
            throughput_access_sum = 0.0
            throughput_backhaul_sum = 0.0
            sat_processed_norm_sum = 0.0
            sat_processed_incoming_ratio_sum = 0.0
            assoc_centroid_sum = 0.0
            assoc_centroid_valid_frac_sum = 0.0
            collision_any = 0.0
            while not done:
                step_result = _run_hybrid_step(
                    actor,
                    driver,
                    device=device,
                    deterministic=deterministic,
                    sources=sources,
                    heuristic_policy=heuristic_policy,
                )
                reward_sum += float(list(step_result.rewards.values())[0])
                steps += 1
                env = driver.env
                parts = dict(getattr(env, "last_reward_parts", {}) or {})
                processed_ratio_sum += float(parts.get("processed_ratio_eval", 0.0))
                drop_ratio_sum += float(parts.get("drop_ratio_eval", 0.0))
                pre_backlog_sum += float(parts.get("pre_backlog_steps_eval", 0.0))
                d_sys_sum += float(parts.get("D_sys_report", 0.0))
                x_acc_sum += float(parts.get("x_acc", 0.0))
                x_rel_sum += float(parts.get("x_rel", 0.0))
                g_pre_sum += float(parts.get("g_pre", 0.0))
                d_pre_sum += float(parts.get("d_pre", 0.0))
                throughput_access_sum += float(parts.get("throughput_access_norm", 0.0))
                throughput_backhaul_sum += float(parts.get("throughput_backhaul_norm", 0.0))
                sat_processed_norm_sum += float(parts.get("sat_processed_norm", 0.0))
                sat_processed_incoming_ratio_sum += float(parts.get("sat_processed_incoming_ratio_step", 0.0))
                assoc_centroid_sum += float(parts.get("assoc_centroid_dist_norm_mean", 0.0))
                assoc_centroid_valid_frac_sum += float(parts.get("assoc_centroid_valid_uav_frac", 0.0))
                collision_any = max(collision_any, float(parts.get("collision_event", 0.0)))
                done = bool(list(step_result.terminations.values())[0] or list(step_result.truncations.values())[0])

            denom = float(max(steps, 1))
            row = {
                "episode": float(ep),
                "reward_sum": reward_sum,
                "processed_ratio_eval": processed_ratio_sum / denom,
                "drop_ratio_eval": drop_ratio_sum / denom,
                "pre_backlog_steps_eval": pre_backlog_sum / denom,
                "D_sys_report": d_sys_sum / denom,
                "x_acc_mean": x_acc_sum / denom,
                "x_rel_mean": x_rel_sum / denom,
                "g_pre_mean": g_pre_sum / denom,
                "d_pre_mean": d_pre_sum / denom,
                "throughput_access_norm_mean": throughput_access_sum / denom,
                "throughput_backhaul_norm_mean": throughput_backhaul_sum / denom,
                "sat_processed_norm_mean": sat_processed_norm_sum / denom,
                "sat_processed_incoming_ratio_mean": sat_processed_incoming_ratio_sum / denom,
                "assoc_centroid_dist_norm_mean": assoc_centroid_sum / denom,
                "assoc_centroid_valid_uav_frac": assoc_centroid_valid_frac_sum / denom,
                "collision_episode_fraction": collision_any,
            }
            rows.append(row)
            for key in totals:
                totals[key] += float(row[key])
        scale = 1.0 / float(max(len(rows), 1))
        summary = {key: float(val * scale) for key, val in totals.items()}
        return summary, rows
    finally:
        close_structured_env_group(driver)


def _new_episode_accumulator() -> Dict[str, float]:
    return {
        "reward_sum": 0.0,
        "processed_ratio_sum": 0.0,
        "drop_ratio_sum": 0.0,
        "pre_backlog_sum": 0.0,
        "d_sys_sum": 0.0,
        "x_acc_sum": 0.0,
        "x_rel_sum": 0.0,
        "g_pre_sum": 0.0,
        "d_pre_sum": 0.0,
        "throughput_access_sum": 0.0,
        "throughput_backhaul_sum": 0.0,
        "sat_processed_norm_sum": 0.0,
        "sat_processed_incoming_ratio_sum": 0.0,
        "assoc_centroid_sum": 0.0,
        "assoc_centroid_valid_frac_sum": 0.0,
        "collision_any": 0.0,
        "steps": 0.0,
    }


def _episode_row_from_accumulator(episode_index: int, acc: Dict[str, float]) -> Dict[str, float]:
    denom = float(max(int(acc["steps"]), 1))
    return {
        "episode": float(episode_index),
        "reward_sum": float(acc["reward_sum"]),
        "processed_ratio_eval": float(acc["processed_ratio_sum"] / denom),
        "drop_ratio_eval": float(acc["drop_ratio_sum"] / denom),
        "pre_backlog_steps_eval": float(acc["pre_backlog_sum"] / denom),
        "D_sys_report": float(acc["d_sys_sum"] / denom),
        "x_acc_mean": float(acc["x_acc_sum"] / denom),
        "x_rel_mean": float(acc["x_rel_sum"] / denom),
        "g_pre_mean": float(acc["g_pre_sum"] / denom),
        "d_pre_mean": float(acc["d_pre_sum"] / denom),
        "throughput_access_norm_mean": float(acc["throughput_access_sum"] / denom),
        "throughput_backhaul_norm_mean": float(acc["throughput_backhaul_sum"] / denom),
        "sat_processed_norm_mean": float(acc["sat_processed_norm_sum"] / denom),
        "sat_processed_incoming_ratio_mean": float(acc["sat_processed_incoming_ratio_sum"] / denom),
        "assoc_centroid_dist_norm_mean": float(acc["assoc_centroid_sum"] / denom),
        "assoc_centroid_valid_uav_frac": float(acc["assoc_centroid_valid_frac_sum"] / denom),
        "collision_episode_fraction": float(acc["collision_any"]),
    }


def _evaluate_combo_parallel(
    cfg,
    actor,
    *,
    device: torch.device,
    episodes: int,
    episode_seed_base: int | None,
    deterministic: bool,
    sources: Dict[str, str],
    heuristic_policy: str,
    num_envs: int,
    vec_backend: str,
) -> tuple[Dict[str, float], List[Dict[str, float]]]:
    active_slots = max(min(int(num_envs), int(episodes)), 1)

    drivers = make_structured_driver_group(cfg, num_envs=active_slots, backend=vec_backend)
    rows: List[Dict[str, float]] = []
    slot_episode = list(range(active_slots))
    slot_active = [True for _ in range(active_slots)]
    slot_acc = [_new_episode_accumulator() for _ in range(active_slots)]
    next_episode = active_slots
    initial_seeds = [
        None if episode_seed_base is None else int(episode_seed_base) + slot
        for slot in range(active_slots)
    ]
    reset_many(drivers, initial_seeds)
    try:
        while len(rows) < int(episodes):
            active_indices = [slot for slot, is_active in enumerate(slot_active) if is_active]
            if not active_indices:
                break
            if looks_like_driver_group(drivers):
                accel_world_states = drivers.prepare_accel_stage_many(indices=active_indices)
            else:
                accel_world_states = [drivers[slot].begin_step() for slot in active_indices]

            if sources["accel"] == SOURCE_POLICY:
                accel_actions = batched_policy_accel_actions(actor, accel_world_states, device, deterministic)
            else:
                if heuristic_policy == "cluster_center_queue_aware":
                    meta_many = cluster_meta_many(drivers, indices=active_indices)
                    accel_actions = [
                        _cluster_accel_from_world_state(
                            accel_world_states[local_slot],
                            cfg,
                            centers=meta_many[local_slot].get("centers"),
                            counts=meta_many[local_slot].get("counts"),
                        )
                        for local_slot in range(len(active_indices))
                    ]
                else:
                    obs_start_many = current_obs_many(drivers, indices=active_indices)
                    accel_actions = [
                        _heuristic_accel(obs_start_many[local_slot], cfg, heuristic_policy)
                        for local_slot in range(len(active_indices))
                    ]

            if looks_like_driver_group(drivers):
                sat_snapshots = drivers.run_accel_and_prepare_sat_many(accel_actions, indices=active_indices)
            else:
                sat_world_states = [
                    drivers[slot].run_accel_stage(action)
                    for slot, action in zip(active_indices, accel_actions)
                ]
                sat_snapshots = [
                    drivers[slot].build_sat_stage_snapshot(world_state)
                    for slot, world_state in zip(active_indices, sat_world_states)
                ]

            obs_after_accel_many = None
            if sources["sat"] == SOURCE_HEURISTIC or sources["bw"] == SOURCE_HEURISTIC:
                refresh_stage_obs_cache_many(drivers, indices=active_indices)
                obs_after_accel_many = current_obs_many(drivers, indices=active_indices)

            if sources["sat"] == SOURCE_POLICY:
                sat_pair_indices = batched_policy_sat_subset_indices(actor, sat_snapshots, device, deterministic)
                if looks_like_driver_group(drivers):
                    bw_snapshots = drivers.run_sat_and_prepare_bw_many(sat_pair_indices, indices=active_indices)
                else:
                    sat_actions = [
                        drivers[slot].decode_sat_subset_actions((), pair_indices)
                        for slot, pair_indices in zip(active_indices, sat_pair_indices)
                    ]
                    bw_world_states = [drivers[slot].run_sat_stage(action) for slot, action in zip(active_indices, sat_actions)]
                    bw_snapshots = [
                        drivers[slot].build_bw_stage_snapshot(world_state)
                        for slot, world_state in zip(active_indices, bw_world_states)
                    ]
            else:
                if obs_after_accel_many is None:
                    refresh_stage_obs_cache_many(drivers, indices=active_indices)
                    obs_after_accel_many = current_obs_many(drivers, indices=active_indices)
                sat_masks = [
                    _heuristic_sat(obs_after_accel_many[local_slot], cfg, heuristic_policy)
                    for local_slot in range(len(active_indices))
                ]
                sat_actions = sat_mask_to_ids_many(drivers, sat_masks, indices=active_indices)
                if looks_like_driver_group(drivers):
                    bw_world_states = drivers.run_sat_stage_many(sat_actions, indices=active_indices)
                    bw_snapshots = drivers.build_bw_stage_snapshot_many(bw_world_states, indices=active_indices)
                else:
                    bw_world_states = [drivers[slot].run_sat_stage(action) for slot, action in zip(active_indices, sat_actions)]
                    bw_snapshots = [
                        drivers[slot].build_bw_stage_snapshot(world_state)
                        for slot, world_state in zip(active_indices, bw_world_states)
                    ]

            if sources["bw"] == SOURCE_POLICY:
                bw_eval = batched_policy_bw_outputs(actor, bw_snapshots, device, deterministic)
                bw_actions = bw_eval.actions
            else:
                if obs_after_accel_many is None:
                    refresh_stage_obs_cache_many(drivers, indices=active_indices)
                    obs_after_accel_many = current_obs_many(drivers, indices=active_indices)
                bw_actions = [
                    _heuristic_bw(obs_after_accel_many[local_slot], cfg, heuristic_policy)
                    for local_slot in range(len(active_indices))
                ]

            if looks_like_driver_group(drivers):
                step_results = drivers.execute_stage_bw_and_step_many(bw_actions, indices=active_indices)
            else:
                step_results = [
                    drivers[slot].execute_stage_bw_and_step(action)
                    for slot, action in zip(active_indices, bw_actions)
                ]
            reward_parts_many = last_reward_parts_many(drivers, indices=active_indices)

            for local_slot, slot in enumerate(active_indices):
                step_result = step_results[local_slot]
                reward_parts = dict(reward_parts_many[local_slot] or {})
                acc = slot_acc[slot]
                acc["reward_sum"] += float(list(step_result.rewards.values())[0])
                acc["steps"] += 1.0
                acc["processed_ratio_sum"] += float(reward_parts.get("processed_ratio_eval", 0.0))
                acc["drop_ratio_sum"] += float(reward_parts.get("drop_ratio_eval", 0.0))
                acc["pre_backlog_sum"] += float(reward_parts.get("pre_backlog_steps_eval", 0.0))
                acc["d_sys_sum"] += float(reward_parts.get("D_sys_report", 0.0))
                acc["x_acc_sum"] += float(reward_parts.get("x_acc", 0.0))
                acc["x_rel_sum"] += float(reward_parts.get("x_rel", 0.0))
                acc["g_pre_sum"] += float(reward_parts.get("g_pre", 0.0))
                acc["d_pre_sum"] += float(reward_parts.get("d_pre", 0.0))
                acc["throughput_access_sum"] += float(reward_parts.get("throughput_access_norm", 0.0))
                acc["throughput_backhaul_sum"] += float(reward_parts.get("throughput_backhaul_norm", 0.0))
                acc["sat_processed_norm_sum"] += float(reward_parts.get("sat_processed_norm", 0.0))
                acc["sat_processed_incoming_ratio_sum"] += float(reward_parts.get("sat_processed_incoming_ratio_step", 0.0))
                acc["assoc_centroid_sum"] += float(reward_parts.get("assoc_centroid_dist_norm_mean", 0.0))
                acc["assoc_centroid_valid_frac_sum"] += float(reward_parts.get("assoc_centroid_valid_uav_frac", 0.0))
                acc["collision_any"] = max(acc["collision_any"], float(reward_parts.get("collision_event", 0.0)))

                done = bool(list(step_result.terminations.values())[0] or list(step_result.truncations.values())[0])
                if not done:
                    continue

                rows.append(_episode_row_from_accumulator(slot_episode[slot], acc))
                slot_acc[slot] = _new_episode_accumulator()
                if next_episode < int(episodes):
                    seed = None if episode_seed_base is None else int(episode_seed_base) + next_episode
                    reset_at(drivers, slot, seed)
                    slot_episode[slot] = int(next_episode)
                    next_episode += 1
                else:
                    slot_active[slot] = False
                    slot_episode[slot] = -1

        rows.sort(key=lambda row: int(row["episode"]))
        totals = _episode_metrics_template()
        for row in rows:
            for key in totals:
                totals[key] += float(row[key])
        scale = 1.0 / float(max(len(rows), 1))
        summary = {key: float(val * scale) for key, val in totals.items()}
        return summary, rows
    finally:
        close_structured_env_group(drivers)


def _delta_value(metric: str, combo_value: float, base_value: float) -> float:
    if metric in SMALLER_IS_BETTER:
        return float(base_value - combo_value)
    return float(combo_value - base_value)


def _build_delta_report(summary_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_name = {str(row["combo_name"]): row for row in summary_rows}
    heur_name = _combo_name({"accel": SOURCE_HEURISTIC, "sat": SOURCE_HEURISTIC, "bw": SOURCE_HEURISTIC})
    heur = by_name.get(heur_name)
    if heur is None:
        return {}

    report: Dict[str, Any] = {"baseline_combo": heur_name, "heads": {}}
    isolated = {
        "accel": _combo_name({"accel": SOURCE_POLICY, "sat": SOURCE_HEURISTIC, "bw": SOURCE_HEURISTIC}),
        "sat": _combo_name({"accel": SOURCE_HEURISTIC, "sat": SOURCE_POLICY, "bw": SOURCE_HEURISTIC}),
        "bw": _combo_name({"accel": SOURCE_HEURISTIC, "sat": SOURCE_HEURISTIC, "bw": SOURCE_POLICY}),
    }
    for head, combo_name in isolated.items():
        row = by_name.get(combo_name)
        if row is None:
            continue
        metrics = HEAD_LOCAL_METRICS[head]
        report["heads"][head] = {
            "combo_name": combo_name,
            "raw_delta_vs_heuristic": {
                metric: float(row[metric] - heur[metric]) for metric in metrics
            },
            "signed_gain_vs_heuristic": {
                metric: _delta_value(metric, float(row[metric]), float(heur[metric])) for metric in metrics
            },
        }
    return report


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--update", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode_seed_base", type=int, default=None)
    parser.add_argument("--policy_mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--combo_set", choices=["minimal", "bw_focus", "all8"], default="minimal")
    parser.add_argument("--accel_source", choices=[SOURCE_POLICY, SOURCE_HEURISTIC], default=None)
    parser.add_argument("--sat_source", choices=[SOURCE_POLICY, SOURCE_HEURISTIC], default=None)
    parser.add_argument("--bw_source", choices=[SOURCE_POLICY, SOURCE_HEURISTIC], default=None)
    parser.add_argument("--heuristic_policy", choices=["cluster_center_queue_aware", "queue_aware"], default="cluster_center_queue_aware")
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=20)
    parser.add_argument("--vec_backend", choices=["sync", "subproc"], default="sync")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--T_steps", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg_path = args.config or str(run_dir / "config_source.yaml")
    cfg = load_config(cfg_path)
    if args.T_steps is not None:
        cfg.T_steps = int(args.T_steps)
    checkpoint = _resolve_checkpoint(run_dir, args.checkpoint, args.update)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = _load_actor(
        cfg,
        checkpoint,
        hidden_dim=None if args.hidden_dim is None else int(args.hidden_dim),
        embed_dim=None if args.embed_dim is None else int(args.embed_dim),
        device=device,
    )
    explicit_sources = [args.accel_source, args.sat_source, args.bw_source]
    if any(source is not None for source in explicit_sources):
        if not all(source is not None for source in explicit_sources):
            raise ValueError("When specifying a single combo, accel_source, sat_source, and bw_source must all be provided.")
        combos = [
            {
                "accel": str(args.accel_source),
                "sat": str(args.sat_source),
                "bw": str(args.bw_source),
            }
        ]
        combo_set_label = "single"
    else:
        combos = _build_combos(args.combo_set)
        combo_set_label = args.combo_set
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "hybrid_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    per_episode_rows: List[Dict[str, Any]] = []
    deterministic = args.policy_mode != "stochastic"
    for combo in combos:
        summary, rows = _evaluate_combo_parallel(
            cfg,
            actor,
            device=device,
            episodes=int(args.episodes),
            episode_seed_base=args.episode_seed_base,
            deterministic=deterministic,
            sources=combo,
            heuristic_policy=args.heuristic_policy,
            num_envs=int(args.num_envs),
            vec_backend=str(args.vec_backend),
        )
        combo_name = _combo_name(combo)
        summary_row: Dict[str, Any] = {
            "combo_name": combo_name,
            "accel_source": combo["accel"],
            "sat_source": combo["sat"],
            "bw_source": combo["bw"],
        }
        summary_row.update(summary)
        summary_rows.append(summary_row)
        for row in rows:
            row_out: Dict[str, Any] = {
                "combo_name": combo_name,
                "accel_source": combo["accel"],
                "sat_source": combo["sat"],
                "bw_source": combo["bw"],
            }
            row_out.update(row)
            per_episode_rows.append(row_out)

    delta_report = _build_delta_report(summary_rows)
    _write_csv(out_dir / "summary.csv", summary_rows)
    _write_csv(out_dir / "per_episode.csv", per_episode_rows)
    payload = {
        "run_dir": str(run_dir),
        "config": str(Path(cfg_path)),
        "checkpoint": str(checkpoint),
        "episodes": int(args.episodes),
        "episode_seed_base": args.episode_seed_base,
        "num_envs": int(args.num_envs),
        "vec_backend": str(args.vec_backend),
        "policy_mode": args.policy_mode,
        "heuristic_policy": args.heuristic_policy,
        "combo_set": combo_set_label,
        "requested_single_combo": None if combo_set_label != "single" else combos[0],
        "summary": summary_rows,
        "isolated_head_deltas_vs_heuristic": delta_report,
        "global_metrics": PRIMARY_GLOBAL_METRICS,
        "head_local_metrics": HEAD_LOCAL_METRICS,
        "smaller_is_better_metrics": sorted(SMALLER_IS_BETTER),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    for row in summary_rows:
        print(
            f"{row['combo_name']}: "
            f"reward={row['reward_sum']:.4f} "
            f"processed={row['processed_ratio_eval']:.4f} "
            f"drop={row['drop_ratio_eval']:.4f} "
            f"pre_backlog={row['pre_backlog_steps_eval']:.4f} "
            f"x_acc={row['x_acc_mean']:.4f} "
            f"x_rel={row['x_rel_mean']:.4f}"
        )
    print(f"Wrote hybrid eval to {out_dir}")


if __name__ == "__main__":
    main()
