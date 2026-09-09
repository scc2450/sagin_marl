from __future__ import annotations

import argparse
import csv
from dataclasses import fields, is_dataclass
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.diagnostics.audit.audit_bw_broad2local_offline import (
    _as_driver_list,
    _batched_bw_actions_from_snapshots,
    _driver_capacity,
    _execute_bw_and_prepare_next_accel_many,
    _normalize_bw_action,
    _random_bw_action,
    _rollout_panel_action_scores,
)
from sagin_marl.env.config import load_config
from sagin_marl.rl.baselines import (
    cluster_center_accel_policy,
    queue_aware_bw_policy,
    queue_aware_sat_policy,
)
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import (
    _normalize_exec_source,
    _split_dataclass_by_counts,
    _to_device_dataclass,
)
from sagin_marl.rl.structured_parallel_eval import (
    batched_policy_accel_actions,
    batched_policy_sat_pair_indices,
    cluster_meta_many,
    current_obs_many,
    last_reward_parts_many,
    looks_like_driver_group,
    refresh_stage_obs_cache_many,
    reset_at,
    reset_many,
)
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_env_group
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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
    load_checkpoint_forgiving(bundle.actor, str(checkpoint), map_location=device, strict=True)
    bundle.actor.to(device).eval()
    return bundle.actor


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _cpu_dataclass(batch: Any) -> Any:
    if not is_dataclass(batch):
        raise TypeError("_cpu_dataclass expects a dataclass instance")
    kwargs = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if not torch.is_tensor(value):
            raise TypeError(f"Unsupported field type for {field.name}: {type(value)!r}")
        kwargs[field.name] = value.detach().cpu()
    return type(batch)(**kwargs)


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


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _heuristic_accel(
    obs_list,
    cfg,
    heuristic_policy: str,
    *,
    centers: np.ndarray | None = None,
    counts: np.ndarray | None = None,
) -> np.ndarray:
    if heuristic_policy != "cluster_center_queue_aware":
        raise ValueError(f"Unsupported heuristic policy: {heuristic_policy}")
    return cluster_center_accel_policy(obs_list, cfg, centers, counts)


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
    if heuristic_policy != "cluster_center_queue_aware":
        raise ValueError(f"Unsupported heuristic policy: {heuristic_policy}")
    return queue_aware_sat_policy(obs_list, cfg)


def _heuristic_bw(obs_list, cfg, heuristic_policy: str) -> np.ndarray:
    if heuristic_policy != "cluster_center_queue_aware":
        raise ValueError(f"Unsupported heuristic policy: {heuristic_policy}")
    return queue_aware_bw_policy(obs_list, cfg)


def _new_episode_accumulator() -> dict[str, float]:
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


def _episode_row_from_accumulator(episode_index: int, acc: dict[str, float]) -> dict[str, float]:
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


def _episode_metrics_template() -> dict[str, float]:
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


def _export_bw_stage_state_many(drivers, *, indices: list[int]) -> list[dict[str, Any]]:
    if looks_like_driver_group(drivers):
        return drivers.export_bw_stage_state_many(indices=indices)
    return [drivers[int(index)].export_bw_stage_state() for index in indices]


def _load_bw_stage_state_many(drivers, snapshots: list[dict[str, Any]], *, indices: list[int]) -> None:
    if looks_like_driver_group(drivers):
        drivers.load_bw_stage_state_many(snapshots, indices=indices)
        return
    for local_idx, driver_idx in enumerate(indices):
        drivers[int(driver_idx)].load_bw_stage_state(snapshots[int(local_idx)])


def _zero_accel_actions(cfg, count: int) -> list[np.ndarray]:
    return [np.zeros((cfg.num_uav, 2), dtype=np.float32) for _ in range(max(int(count), 0))]


def _zero_sat_actions(cfg, count: int) -> list[np.ndarray]:
    select_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 1)
    return [np.full((cfg.num_uav, select_k), -1, dtype=np.int64) for _ in range(max(int(count), 0))]


def _prepare_bw_stage_many(
    *,
    drivers,
    indices: list[int],
    accel_world_states: list[Any],
    actor,
    cfg,
    device: torch.device,
    heuristic_policy: str,
    exec_accel_source: str,
    exec_sat_source: str,
) -> tuple[list[Any], list[list[dict[str, np.ndarray]]], list[dict[str, Any]]]:
    slot_count = len(indices)
    if slot_count <= 0:
        return [], [], []

    accel_source = str(exec_accel_source).strip().lower()
    sat_source = str(exec_sat_source).strip().lower()

    if accel_source == "policy":
        accel_actions = batched_policy_accel_actions(actor, accel_world_states, device, True)
    elif accel_source == "zero":
        accel_actions = _zero_accel_actions(cfg, slot_count)
    elif accel_source == "cluster_center_queue_aware":
        centers_many, counts_many = cluster_meta_many(drivers, indices=indices)
        accel_actions = [
            _cluster_accel_from_world_state(
                world_state,
                cfg,
                centers=np.asarray(centers, dtype=np.float32),
                counts=np.asarray(counts, dtype=np.float32),
            )
            for world_state, centers, counts in zip(accel_world_states, centers_many, counts_many)
        ]
    else:
        raise ValueError(f"Unsupported exec_accel_source for BW select: {exec_accel_source}")

    if looks_like_driver_group(drivers):
        sat_snapshots = drivers.run_accel_and_prepare_sat_many(accel_actions, indices=indices)
    else:
        sat_world_states = [
            drivers[int(slot)].run_accel_stage(action)
            for slot, action in zip(indices, accel_actions)
        ]
        sat_snapshots = [
            drivers[int(slot)].build_sat_stage_snapshot(world_state)
            for slot, world_state in zip(indices, sat_world_states)
        ]

    refresh_stage_obs_cache_many(drivers, indices=indices)
    obs_after_accel_many = current_obs_many(drivers, indices=indices)

    if sat_source == "policy":
        sat_pair_indices = batched_policy_sat_pair_indices(actor, sat_snapshots, device, True)
        if looks_like_driver_group(drivers):
            _ = drivers.run_sat_and_prepare_bw_many(sat_pair_indices, indices=indices)
        else:
            sat_actions = [
                drivers[int(slot)].decode_sat_pair_actions((), pair_indices)
                for slot, pair_indices in zip(indices, sat_pair_indices)
            ]
            for slot, action in zip(indices, sat_actions):
                drivers[int(slot)].run_sat_stage(action)
    elif sat_source == "zero":
        sat_actions = _zero_sat_actions(cfg, slot_count)
        if looks_like_driver_group(drivers):
            _ = drivers.run_sat_stage_many(sat_actions, indices=indices)
        else:
            for slot, action in zip(indices, sat_actions):
                drivers[int(slot)].run_sat_stage(action)
    elif sat_source == "cluster_center_queue_aware":
        sat_masks = [_heuristic_sat(obs_after_accel, cfg, heuristic_policy) for obs_after_accel in obs_after_accel_many]
        if looks_like_driver_group(drivers):
            sat_actions = drivers.sat_mask_to_ids_many(sat_masks, indices=indices)
            _ = drivers.run_sat_stage_many(sat_actions, indices=indices)
        else:
            sat_actions = [
                _heuristic_sat_mask_to_ids(drivers[int(slot)], sat_mask)
                for slot, sat_mask in zip(indices, sat_masks)
            ]
            for slot, action in zip(indices, sat_actions):
                drivers[int(slot)].run_sat_stage(action)
    else:
        raise ValueError(f"Unsupported exec_sat_source for BW select: {exec_sat_source}")

    if looks_like_driver_group(drivers):
        bw_snapshots = drivers.build_bw_stage_snapshot_many(indices=indices)
    else:
        bw_snapshots = [drivers[int(slot)].build_bw_stage_snapshot() for slot in indices]
    snapshot_states = _export_bw_stage_state_many(drivers, indices=indices)
    return bw_snapshots, obs_after_accel_many, snapshot_states


def _followup_bw_actions_many(
    *,
    drivers,
    indices: list[int],
    actor,
    cfg,
    device: torch.device,
    heuristic_policy: str,
    accel_world_states: list[Any],
    exec_accel_source: str,
    exec_sat_source: str,
    follow_bw_source: str,
    bw_deterministic_opt_steps: int,
    bw_deterministic_step_size: float,
) -> list[np.ndarray]:
    bw_snapshots, obs_after_accel_many, _ = _prepare_bw_stage_many(
        drivers=drivers,
        indices=indices,
        accel_world_states=accel_world_states,
        actor=actor,
        cfg=cfg,
        device=device,
        heuristic_policy=heuristic_policy,
        exec_accel_source=exec_accel_source,
        exec_sat_source=exec_sat_source,
    )
    follow_source = str(follow_bw_source).strip().lower()
    if follow_source == "policy":
        bw_eval = _batched_bw_actions_from_snapshots(
            actor,
            bw_snapshots,
            device=device,
            deterministic=True,
            bw_deterministic_readout="latent_mean_pushforward",
            bw_deterministic_opt_steps=int(bw_deterministic_opt_steps),
            bw_deterministic_step_size=float(bw_deterministic_step_size),
        )
        return [np.asarray(action, dtype=np.float32) for action in bw_eval["actions"]]
    if follow_source in {"queue_aware", "cluster_center_queue_aware"}:
        actions: list[np.ndarray] = []
        for snapshot, obs_after_accel in zip(bw_snapshots, obs_after_accel_many):
            valid_mask = np.asarray(snapshot.bw_valid_mask, dtype=bool)
            raw = _heuristic_bw(obs_after_accel, cfg, heuristic_policy)
            actions.append(_normalize_bw_action(raw, valid_mask))
        return actions
    raise ValueError(f"Unsupported follow_bw_source for BW select: {follow_bw_source}")


def _rollout_panel_action_scores_fixed_exec(
    *,
    eval_drivers,
    snapshot_state: dict[str, Any],
    panel_actions: list[np.ndarray],
    actor,
    cfg,
    device: torch.device,
    heuristic_policy: str,
    exec_accel_source: str,
    exec_sat_source: str,
    follow_bw_source: str,
    bw_deterministic_opt_steps: int,
    bw_deterministic_step_size: float,
    gamma: float,
    k_steps: int,
) -> list[float]:
    if not panel_actions:
        return []
    chunk_size = max(1, _driver_capacity(eval_drivers))
    panel_scores: list[float] = []
    for start in range(0, len(panel_actions), chunk_size):
        chunk_actions = panel_actions[start : start + chunk_size]
        selected_indices = list(range(len(chunk_actions)))
        totals = [0.0 for _ in chunk_actions]
        active_local_indices = list(range(len(chunk_actions)))
        current_actions = [np.asarray(action, dtype=np.float32) for action in chunk_actions]
        discount = 1.0
        _load_bw_stage_state_many(
            eval_drivers,
            [snapshot_state for _ in chunk_actions],
            indices=selected_indices,
        )
        for step_idx in range(max(int(k_steps), 0)):
            if not active_local_indices:
                break
            active_driver_indices = [selected_indices[int(local_idx)] for local_idx in active_local_indices]
            active_actions = [current_actions[int(local_idx)] for local_idx in active_local_indices]
            step_results, next_world_states = _execute_bw_and_prepare_next_accel_many(
                eval_drivers,
                active_actions,
                indices=active_driver_indices,
            )
            next_active_local_indices: list[int] = []
            next_active_world_states: list[Any] = []
            for offset, local_idx in enumerate(active_local_indices):
                step_result = step_results[offset]
                reward = float(next(iter(step_result.rewards.values())))
                totals[int(local_idx)] += discount * reward
                done = bool(next(iter(step_result.terminations.values())) or next(iter(step_result.truncations.values())))
                if done or step_idx + 1 >= int(k_steps):
                    continue
                next_active_local_indices.append(int(local_idx))
                next_active_world_states.append(next_world_states[offset])
            if not next_active_local_indices or step_idx + 1 >= int(k_steps):
                break
            discount *= float(gamma)
            next_driver_indices = [selected_indices[int(local_idx)] for local_idx in next_active_local_indices]
            next_actions = _followup_bw_actions_many(
                drivers=eval_drivers,
                indices=next_driver_indices,
                actor=actor,
                cfg=cfg,
                device=device,
                heuristic_policy=heuristic_policy,
                accel_world_states=next_active_world_states,
                exec_accel_source=exec_accel_source,
                exec_sat_source=exec_sat_source,
                follow_bw_source=follow_bw_source,
                bw_deterministic_opt_steps=int(bw_deterministic_opt_steps),
                bw_deterministic_step_size=float(bw_deterministic_step_size),
            )
            for local_idx, action in zip(next_active_local_indices, next_actions):
                current_actions[int(local_idx)] = np.asarray(action, dtype=np.float32)
            active_local_indices = next_active_local_indices
        panel_scores.extend(float(score) for score in totals)
    return panel_scores


def _build_candidate_panels(
    *,
    actor,
    cfg,
    device: torch.device,
    bw_snapshots: list[Any],
    obs_after_accel_many: list[list[dict[str, np.ndarray]]] | None,
    heuristic_policy: str,
    include_heuristic: bool,
    include_simplex: bool,
    sample_count: int,
    random_count: int,
    bw_deterministic_opt_steps: int,
    bw_deterministic_step_size: float,
    rng: np.random.Generator,
) -> tuple[list[list[str]], list[list[np.ndarray]], list[Any]]:
    per_slot_names: list[list[str]] = [[] for _ in bw_snapshots]
    per_slot_actions: list[list[np.ndarray]] = [[] for _ in bw_snapshots]

    latent_eval = _batched_bw_actions_from_snapshots(
        actor,
        bw_snapshots,
        device=device,
        deterministic=True,
        bw_deterministic_readout="latent_mean_pushforward",
        bw_deterministic_opt_steps=int(bw_deterministic_opt_steps),
        bw_deterministic_step_size=float(bw_deterministic_step_size),
    )
    latent_actions = latent_eval["actions"]
    local_states_many = [
        _cpu_dataclass(_to_device_dataclass(piece, torch.device("cpu")))
        for piece in _split_dataclass_by_counts(latent_eval["local_state"], latent_eval["agent_counts"])
    ]
    for slot, action in enumerate(latent_actions):
        per_slot_names[slot].append("latent_det")
        per_slot_actions[slot].append(np.asarray(action, dtype=np.float32))

    if include_simplex:
        simplex_actions = _batched_bw_actions_from_snapshots(
            actor,
            bw_snapshots,
            device=device,
            deterministic=True,
            bw_deterministic_readout="simplex_argmax_logprob",
            bw_deterministic_opt_steps=int(bw_deterministic_opt_steps),
            bw_deterministic_step_size=float(bw_deterministic_step_size),
        )["actions"]
        for slot, action in enumerate(simplex_actions):
            per_slot_names[slot].append("simplex_det")
            per_slot_actions[slot].append(np.asarray(action, dtype=np.float32))

    if include_heuristic:
        if obs_after_accel_many is None:
            raise RuntimeError("Heuristic BW candidates require obs_after_accel_many.")
        for slot, snapshot in enumerate(bw_snapshots):
            valid_mask = np.asarray(snapshot.bw_valid_mask, dtype=bool)
            heur_raw = _heuristic_bw(obs_after_accel_many[slot], cfg, heuristic_policy)
            heur_action = _normalize_bw_action(heur_raw, valid_mask)
            per_slot_names[slot].append("heuristic")
            per_slot_actions[slot].append(np.asarray(heur_action, dtype=np.float32))

    for sample_idx in range(max(int(sample_count), 0)):
        sample_actions = _batched_bw_actions_from_snapshots(
            actor,
            bw_snapshots,
            device=device,
            deterministic=False,
            bw_deterministic_readout="latent_mean_pushforward",
            bw_deterministic_opt_steps=int(bw_deterministic_opt_steps),
            bw_deterministic_step_size=float(bw_deterministic_step_size),
        )["actions"]
        for slot, action in enumerate(sample_actions):
            per_slot_names[slot].append(f"sample_{sample_idx}")
            per_slot_actions[slot].append(np.asarray(action, dtype=np.float32))

    for random_idx in range(max(int(random_count), 0)):
        for slot, snapshot in enumerate(bw_snapshots):
            valid_mask = np.asarray(snapshot.bw_valid_mask, dtype=bool)
            action = _random_bw_action(valid_mask, rng)
            per_slot_names[slot].append(f"random_{random_idx}")
            per_slot_actions[slot].append(np.asarray(action, dtype=np.float32))

    return per_slot_names, per_slot_actions, local_states_many


def _evaluate_select_parallel(
    *,
    cfg,
    actor,
    device: torch.device,
    episodes: int,
    episode_seed_base: int | None,
    num_envs: int,
    vec_backend: str,
    heuristic_policy: str,
    include_heuristic: bool,
    include_simplex: bool,
    sample_count: int,
    random_count: int,
    k_steps: int,
    seed: int,
    bw_deterministic_opt_steps: int,
    bw_deterministic_step_size: float,
    export_bank: bool,
    exec_accel_source: str,
    exec_sat_source: str,
    follow_bw_source: str,
) -> tuple[dict[str, Any], list[dict[str, float]], list[dict[str, Any]], list[dict[str, Any]]]:
    active_slots = max(min(int(num_envs), int(episodes)), 1)
    env_group = make_structured_env_group(cfg, num_envs=active_slots, backend=vec_backend)
    drivers = env_group if looks_like_driver_group(env_group) else _as_driver_list(env_group)
    eval_slots = max(1, min(max(int(sample_count) + int(random_count) + 4, 1), max(int(num_envs), 1)))
    eval_group = make_structured_env_group(cfg, num_envs=eval_slots, backend=vec_backend)
    eval_drivers = eval_group if looks_like_driver_group(eval_group) else _as_driver_list(eval_group)
    rows: list[dict[str, float]] = []
    select_rows: list[dict[str, Any]] = []
    bank_entries: list[dict[str, Any]] = []
    selection_gain_vs_latent: list[float] = []
    selection_gain_vs_heuristic: list[float] = []
    selection_gain_vs_simplex: list[float] = []
    selection_source_counts: dict[str, int] = {}
    selected_beats_latent: list[float] = []
    selected_beats_heuristic: list[float] = []
    selected_beats_simplex: list[float] = []
    slot_episode = list(range(active_slots))
    slot_active = [True for _ in range(active_slots)]
    slot_acc = [_new_episode_accumulator() for _ in range(active_slots)]
    next_episode = active_slots
    initial_seeds = [
        None if episode_seed_base is None else int(episode_seed_base) + slot
        for slot in range(active_slots)
    ]
    rng = np.random.default_rng(int(seed))
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
            bw_snapshots, obs_after_accel_many, snapshot_states = _prepare_bw_stage_many(
                drivers=drivers,
                indices=active_indices,
                accel_world_states=accel_world_states,
                actor=actor,
                cfg=cfg,
                device=device,
                heuristic_policy=heuristic_policy,
                exec_accel_source=exec_accel_source,
                exec_sat_source=exec_sat_source,
            )

            candidate_names_many, candidate_actions_many, local_states_many = _build_candidate_panels(
                actor=actor,
                cfg=cfg,
                device=device,
                bw_snapshots=bw_snapshots,
                obs_after_accel_many=obs_after_accel_many,
                heuristic_policy=heuristic_policy,
                include_heuristic=include_heuristic,
                include_simplex=include_simplex,
                sample_count=int(sample_count),
                random_count=int(random_count),
                bw_deterministic_opt_steps=int(bw_deterministic_opt_steps),
                bw_deterministic_step_size=float(bw_deterministic_step_size),
                rng=rng,
            )

            selected_actions: list[np.ndarray] = []
            gamma = float(cfg.gamma)
            for local_slot, slot in enumerate(active_indices):
                candidate_names = candidate_names_many[local_slot]
                candidate_actions = candidate_actions_many[local_slot]
                candidate_scores = _rollout_panel_action_scores_fixed_exec(
                    eval_drivers=eval_drivers,
                    snapshot_state=snapshot_states[local_slot],
                    panel_actions=candidate_actions,
                    actor=actor,
                    cfg=cfg,
                    device=device,
                    heuristic_policy=heuristic_policy,
                    exec_accel_source=exec_accel_source,
                    exec_sat_source=exec_sat_source,
                    follow_bw_source=follow_bw_source,
                    bw_deterministic_opt_steps=int(bw_deterministic_opt_steps),
                    bw_deterministic_step_size=float(bw_deterministic_step_size),
                    gamma=gamma,
                    k_steps=int(k_steps),
                )
                best_idx = int(np.argmax(np.asarray(candidate_scores, dtype=np.float64)))
                best_name = str(candidate_names[best_idx])
                best_action = np.asarray(candidate_actions[best_idx], dtype=np.float32)
                selected_actions.append(best_action)
                selection_source_counts[best_name] = int(selection_source_counts.get(best_name, 0) + 1)

                score_map = {
                    str(name): float(score)
                    for name, score in zip(candidate_names, candidate_scores)
                }
                latent_score = float(score_map.get("latent_det", 0.0))
                heuristic_score = score_map.get("heuristic")
                simplex_score = score_map.get("simplex_det")
                best_score = float(candidate_scores[best_idx])
                selection_gain_vs_latent.append(float(best_score - latent_score))
                selected_beats_latent.append(float(best_score > latent_score + 1.0e-9))
                if heuristic_score is not None:
                    selection_gain_vs_heuristic.append(float(best_score - float(heuristic_score)))
                    selected_beats_heuristic.append(float(best_score > float(heuristic_score) + 1.0e-9))
                if simplex_score is not None:
                    selection_gain_vs_simplex.append(float(best_score - float(simplex_score)))
                    selected_beats_simplex.append(float(best_score > float(simplex_score) + 1.0e-9))

                select_rows.append(
                    {
                        "episode": int(slot_episode[slot]),
                        "slot": int(slot),
                        "t": int(snapshot_states[local_slot].get("env_state", {}).get("t", 0)),
                        "candidate_count": int(len(candidate_names)),
                        "selected_name": best_name,
                        "selected_score": best_score,
                        "latent_score": latent_score,
                        "heuristic_score": None if heuristic_score is None else float(heuristic_score),
                        "simplex_score": None if simplex_score is None else float(simplex_score),
                        "selected_minus_latent": float(best_score - latent_score),
                        "selected_minus_heuristic": None
                        if heuristic_score is None
                        else float(best_score - float(heuristic_score)),
                        "selected_minus_simplex": None
                        if simplex_score is None
                        else float(best_score - float(simplex_score)),
                    }
                )
                if export_bank:
                    bank_entries.append(
                        {
                            "episode": int(slot_episode[slot]),
                            "slot": int(slot),
                            "t": int(snapshot_states[local_slot].get("env_state", {}).get("t", 0)),
                            "local_state": local_states_many[local_slot],
                            "snapshot_state": snapshot_states[local_slot],
                            "candidate_names": list(candidate_names),
                            "candidate_actions": [np.asarray(action, dtype=np.float32) for action in candidate_actions],
                            "candidate_scores": [float(score) for score in candidate_scores],
                            "selected_idx": int(best_idx),
                            "selected_name": best_name,
                            "policy_det_idx": int(candidate_names.index("latent_det")),
                            "heuristic_idx": (None if "heuristic" not in candidate_names else int(candidate_names.index("heuristic"))),
                        }
                    )

            if looks_like_driver_group(drivers):
                step_results = drivers.execute_stage_bw_and_step_many(selected_actions, indices=active_indices)
            else:
                step_results = [
                    drivers[slot].execute_stage_bw_and_step(action)
                    for slot, action in zip(active_indices, selected_actions)
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
                    seed_next = None if episode_seed_base is None else int(episode_seed_base) + next_episode
                    reset_at(drivers, slot, seed_next)
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
        summary.update(
            {
                "selection_step_count": int(len(select_rows)),
                "selection_gain_vs_latent": _summarize(selection_gain_vs_latent),
                "selection_gain_vs_heuristic": _summarize(selection_gain_vs_heuristic),
                "selection_gain_vs_simplex": _summarize(selection_gain_vs_simplex),
                "selected_beats_latent_frac": _mean(selected_beats_latent),
                "selected_beats_heuristic_frac": _mean(selected_beats_heuristic),
                "selected_beats_simplex_frac": _mean(selected_beats_simplex),
                "selection_source_hist": {str(k): int(v) for k, v in sorted(selection_source_counts.items())},
            }
        )
        return summary, rows, select_rows, bank_entries
    finally:
        close_structured_env_group(env_group)
        close_structured_env_group(eval_group)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--update", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--episode_seed_base", type=int, default=61000)
    parser.add_argument("--heuristic_policy", choices=["cluster_center_queue_aware"], default="cluster_center_queue_aware")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--vec_backend", choices=["sync", "subproc"], default="sync")
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--sample_count", type=int, default=8)
    parser.add_argument("--random_count", type=int, default=1)
    parser.add_argument("--include_heuristic", action="store_true")
    parser.add_argument("--include_simplex", action="store_true")
    parser.add_argument("--k_steps", type=int, default=2)
    parser.add_argument("--bw_deterministic_opt_steps", type=int, default=8)
    parser.add_argument("--bw_deterministic_step_size", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--export_bank", action="store_true")
    parser.add_argument("--exec_accel_source", type=str, default=None)
    parser.add_argument("--exec_sat_source", type=str, default=None)
    parser.add_argument("--follow_bw_source", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_all_seeds(int(args.seed))
    run_dir = Path(args.run_dir)
    cfg_path = args.config or str(run_dir / "config_source.yaml")
    cfg = load_config(cfg_path)
    checkpoint = _resolve_checkpoint(run_dir, args.checkpoint, args.update)
    device = torch.device(args.device)
    exec_accel_source = _normalize_exec_source(args.exec_accel_source or getattr(cfg, "exec_accel_source", "policy"))
    exec_sat_source = _normalize_exec_source(args.exec_sat_source or getattr(cfg, "exec_sat_source", "policy"))
    follow_bw_source = _normalize_exec_source(args.follow_bw_source or getattr(cfg, "exec_bw_source", "policy"))
    actor = _load_actor(
        cfg,
        checkpoint,
        hidden_dim=None if args.hidden_dim is None else int(args.hidden_dim),
        embed_dim=None if args.embed_dim is None else int(args.embed_dim),
        device=device,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, episode_rows, select_rows, bank_entries = _evaluate_select_parallel(
        cfg=cfg,
        actor=actor,
        device=device,
        episodes=int(args.episodes),
        episode_seed_base=args.episode_seed_base,
        num_envs=int(args.num_envs),
        vec_backend=str(args.vec_backend),
        heuristic_policy=str(args.heuristic_policy),
        include_heuristic=bool(args.include_heuristic),
        include_simplex=bool(args.include_simplex),
        sample_count=int(args.sample_count),
        random_count=int(args.random_count),
        k_steps=int(args.k_steps),
        seed=int(args.seed),
        bw_deterministic_opt_steps=int(args.bw_deterministic_opt_steps),
        bw_deterministic_step_size=float(args.bw_deterministic_step_size),
        export_bank=bool(args.export_bank),
        exec_accel_source=exec_accel_source,
        exec_sat_source=exec_sat_source,
        follow_bw_source=follow_bw_source,
    )

    _write_csv(out_dir / "per_episode.csv", episode_rows)
    _write_csv(out_dir / "select_steps.csv", select_rows)
    payload = {
        "config": str(Path(cfg_path)),
        "checkpoint": str(checkpoint),
        "episodes": int(args.episodes),
        "episode_seed_base": int(args.episode_seed_base) if args.episode_seed_base is not None else None,
        "num_envs": int(args.num_envs),
        "vec_backend": str(args.vec_backend),
        "sample_count": int(args.sample_count),
        "random_count": int(args.random_count),
        "include_heuristic": bool(args.include_heuristic),
        "include_simplex": bool(args.include_simplex),
        "k_steps": int(args.k_steps),
        "bw_deterministic_opt_steps": int(args.bw_deterministic_opt_steps),
        "bw_deterministic_step_size": float(args.bw_deterministic_step_size),
        "exec_accel_source": str(exec_accel_source),
        "exec_sat_source": str(exec_sat_source),
        "follow_bw_source": str(follow_bw_source),
        "summary": summary,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    if args.export_bank:
        torch.save(
            {
                "meta": payload,
                "entries": bank_entries,
            },
            out_dir / "select_bank.pt",
        )

    print(
        f"Summary: reward={summary['reward_sum']:.4f} "
        f"processed={summary['processed_ratio_eval']:.4f} "
        f"drop={summary['drop_ratio_eval']:.4f} "
        f"pre_backlog={summary['pre_backlog_steps_eval']:.4f} "
        f"beats_latent={summary['selected_beats_latent_frac']:.4f} "
        f"beats_heuristic={summary['selected_beats_heuristic_frac']:.4f}"
    )


if __name__ == "__main__":
    main()
