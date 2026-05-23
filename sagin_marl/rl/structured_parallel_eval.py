from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.structured_mappo import (
    _collate_dataclass,
    _split_cpu_tensor_by_counts,
)
from sagin_marl.rl.structured_types import LocalBwState


@dataclass
class BatchedBwEvalOutputs:
    actions: list[np.ndarray]
    local_state: Any
    out: Any
    agent_counts: list[int]


def _current_obs_list(env: SaginParallelEnv) -> list[dict[str, np.ndarray]]:
    return [env._get_obs(i) for i in range(len(env.agents))]


def _refresh_stage_obs_cache(driver: StructuredControlDriver) -> None:
    if driver._stage_assoc is None or driver._stage_candidates is None:
        raise RuntimeError("run_accel_stage must be called before refreshing stage obs cache")
    env = driver.env
    _, eta_slots = env._compute_access_rates(
        driver._stage_assoc,
        driver._stage_candidates,
        driver._zero_bw_action_matrix(),
        record_exec=False,
    )
    env._store_cached_access_stage_context(
        driver._stage_assoc,
        driver._stage_candidates,
        eta=eta_slots,
        bw_valid_mask=driver._stage_bw_valid_mask,
        snapshot_step_t=int(env.t),
    )
    if driver._stage_sat_pos is not None and driver._stage_sat_vel is not None and driver._stage_visible is not None:
        env._cache_sat_obs(driver._stage_sat_pos, driver._stage_sat_vel, driver._stage_visible)


def _sat_mask_to_ids(driver: StructuredControlDriver, sat_mask: np.ndarray) -> np.ndarray:
    cfg = driver.env.cfg
    select_k = max(int(getattr(cfg, "sat_action_select_k", cfg.N_RF) or cfg.N_RF), 1)
    out = np.full((cfg.num_uav, select_k), -1, dtype=np.int64)
    if driver._stage_visible is None:
        raise RuntimeError("run_accel_stage must be called before decoding sat masks")
    sat_mask_arr = np.asarray(sat_mask, dtype=np.float32)
    visible_width = max(
        min(
            int(getattr(cfg, "per_uav_visible_sat_token_max", cfg.sats_obs_max) or cfg.sats_obs_max),
            int(cfg.num_sat),
        ),
        0,
    )
    for u in range(cfg.num_uav):
        visible = driver._stage_visible[u][:visible_width]
        active_slots = np.flatnonzero(sat_mask_arr[u] > 0.5)
        mapped: list[int] = []
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


def looks_like_driver_group(obj: Any) -> bool:
    required = (
        "reset_many",
        "reset_at",
        "native_rollout_program",
        "begin_native_main_kernel_rollout",
        "native_rollout_runtime",
    )
    return all(hasattr(obj, name) for name in required)


def current_obs_many(drivers: Any, indices: Sequence[int] | None = None) -> list[list[dict[str, np.ndarray]]]:
    if looks_like_driver_group(drivers):
        return drivers.current_obs_many(indices=indices)
    if indices is None:
        selected = drivers
    else:
        selected = [drivers[int(index)] for index in indices]
    return [_current_obs_list(driver.env) for driver in selected]


def current_obs_batch(
    drivers: Any,
    *,
    indices: Sequence[int] | None = None,
    device: torch.device | str | None = None,
) -> dict[str, np.ndarray | torch.Tensor]:
    if looks_like_driver_group(drivers):
        batch_getter = getattr(drivers, "current_obs_batch", None)
        if callable(batch_getter):
            return batch_getter(indices=indices, device=device)
    obs_many = current_obs_many(drivers, indices=indices)
    if not obs_many:
        return {}
    sample_keys = tuple(obs_many[0][0].keys())
    stacked = {
        key: np.stack(
            [
                np.stack(
                    [np.asarray(obs[key], dtype=np.float32) for obs in obs_list],
                    axis=0,
                )
                for obs_list in obs_many
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        for key in sample_keys
    }
    if device is None:
        return stacked
    out_device = torch.device(device)
    return {
        key: torch.as_tensor(value, dtype=torch.float32, device=out_device)
        for key, value in stacked.items()
    }


def cluster_meta_many(drivers: Any, indices: Sequence[int] | None = None) -> list[dict[str, Any]]:
    if looks_like_driver_group(drivers):
        return drivers.cluster_meta_many(indices=indices)
    if indices is None:
        selected = drivers
    else:
        selected = [drivers[int(index)] for index in indices]
    rows: list[dict[str, Any]] = []
    for driver in selected:
        env = driver.env
        rows.append(
            {
                "centers": None if getattr(env, "gu_cluster_centers", None) is None else np.asarray(env.gu_cluster_centers),
                "counts": None if getattr(env, "gu_cluster_counts", None) is None else np.asarray(env.gu_cluster_counts),
            }
        )
    return rows


def cluster_meta_batch(
    drivers: Any,
    *,
    indices: Sequence[int] | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    if looks_like_driver_group(drivers):
        batch_getter = getattr(drivers, "cluster_meta_batch", None)
        if callable(batch_getter):
            return batch_getter(indices=indices, device=device)
    rows = cluster_meta_many(drivers, indices=indices)
    if not rows:
        return {"centers": None, "counts": None}
    centers = [row.get("centers") for row in rows]
    counts = [row.get("counts") for row in rows]
    if any(center is None for center in centers) or any(count is None for count in counts):
        return {"centers": None, "counts": None}
    center_batch = np.stack([np.asarray(center, dtype=np.float32) for center in centers], axis=0).astype(
        np.float32,
        copy=False,
    )
    count_batch = np.stack([np.asarray(count, dtype=np.float32) for count in counts], axis=0).astype(
        np.float32,
        copy=False,
    )
    if device is None:
        return {"centers": center_batch, "counts": count_batch}
    out_device = torch.device(device)
    return {
        "centers": torch.as_tensor(center_batch, dtype=torch.float32, device=out_device),
        "counts": torch.as_tensor(count_batch, dtype=torch.float32, device=out_device),
    }


def refresh_stage_obs_cache_many(drivers: Any, indices: Sequence[int] | None = None) -> None:
    if looks_like_driver_group(drivers):
        drivers.refresh_stage_obs_cache_many(indices=indices)
        return
    if indices is None:
        selected = drivers
    else:
        selected = [drivers[int(index)] for index in indices]
    for driver in selected:
        _refresh_stage_obs_cache(driver)


def sat_mask_to_ids_many(
    drivers: Any,
    sat_masks: Sequence[np.ndarray],
    indices: Sequence[int] | None = None,
) -> list[np.ndarray]:
    if looks_like_driver_group(drivers):
        return drivers.sat_mask_to_ids_many(sat_masks, indices=indices)
    if indices is None:
        selected = drivers
    else:
        selected = [drivers[int(index)] for index in indices]
    return [_sat_mask_to_ids(driver, sat_mask) for driver, sat_mask in zip(selected, sat_masks)]


def last_reward_parts_many(drivers: Any, indices: Sequence[int] | None = None) -> list[dict[str, Any]]:
    if looks_like_driver_group(drivers):
        return drivers.last_reward_parts_many(indices=indices)
    if indices is None:
        selected = drivers
    else:
        selected = [drivers[int(index)] for index in indices]
    return [dict(getattr(driver.env, "last_reward_parts", {}) or {}) for driver in selected]


def reset_many(drivers_or_env_group: Any, seeds: Sequence[int | None]) -> None:
    if looks_like_driver_group(drivers_or_env_group):
        drivers_or_env_group.reset_many(seeds)
        return
    for driver, seed in zip(drivers_or_env_group, seeds):
        env = driver.env if hasattr(driver, "env") else driver
        if seed is None:
            env.reset()
        else:
            env.reset(seed=int(seed))


def reset_at(drivers_or_env_group: Any, index: int, seed: int | None) -> None:
    if looks_like_driver_group(drivers_or_env_group):
        drivers_or_env_group.reset_at(int(index), seed)
        return
    env_or_driver = drivers_or_env_group[int(index)]
    env = env_or_driver.env if hasattr(env_or_driver, "env") else env_or_driver
    if seed is None:
        env.reset()
    else:
        env.reset(seed=int(seed))


def batched_policy_accel_actions(actor, accel_world_states: Sequence[Any], device: torch.device, deterministic: bool) -> list[np.ndarray]:
    del actor, accel_world_states, device, deterministic
    raise RuntimeError("Accel actor evaluation must use StructuredControlDriver.build_local_accel_states(), not world-state slicing.")


def batched_policy_accel_actions_from_world_batch(
    actor,
    accel_world_batch: Any,
    deterministic: bool,
    *,
    agent_counts: Sequence[int] | None = None,
) -> list[np.ndarray]:
    del actor, accel_world_batch, deterministic, agent_counts
    raise RuntimeError("Accel actor evaluation must use StructuredControlDriver.build_local_accel_states(), not world-state slicing.")


def batched_policy_sat_subset_indices(actor, sat_snapshots: Sequence[Any], device: torch.device, deterministic: bool) -> list[list[int]]:
    if not sat_snapshots:
        return []
    if not all(getattr(snapshot, "local_state", None) is not None for snapshot in sat_snapshots):
        raise RuntimeError(
            "SAT actor evaluation requires SatStageSnapshot.local_state from "
            "StructuredControlDriver.build_sat_stage_snapshot()."
        )
    local_states = [snapshot.local_state for snapshot in sat_snapshots]
    agent_counts = [int(state.ego_features.shape[0]) for state in local_states]
    sat_batch = _collate_dataclass(local_states, device)
    with torch.inference_mode():
        sat_out = actor.act_sat(sat_batch, deterministic=deterministic)
    return [piece.to(dtype=torch.long).tolist() for piece in _split_cpu_tensor_by_counts(sat_out.subset_index, agent_counts)]


def batched_policy_sat_subset_indices_from_world_batch(
    actor,
    sat_world_batch: Any,
    *,
    sat_max_select: int,
    deterministic: bool,
    agent_counts: Sequence[int] | None = None,
) -> list[list[int]]:
    del actor, sat_world_batch, sat_max_select, deterministic, agent_counts
    raise RuntimeError(
        "SAT actor evaluation no longer rebuilds LocalSatState from StructuredWorldState. "
        "Use batched_policy_sat_subset_indices() with snapshots that carry local_state."
    )


def batched_policy_bw_outputs(
    actor,
    bw_snapshots: Sequence[Any],
    device: torch.device,
    deterministic: bool,
) -> BatchedBwEvalOutputs:
    if not bw_snapshots:
        return BatchedBwEvalOutputs(actions=[], local_state=None, out=None, agent_counts=[])
    if all(getattr(snapshot, "ego_features", None) is not None for snapshot in bw_snapshots):
        def _stack_field(name: str, *, dtype: torch.dtype) -> torch.Tensor:
            values = [
                torch.as_tensor(getattr(snapshot, name), dtype=dtype, device=device)
                for snapshot in bw_snapshots
            ]
            stacked = torch.stack(values, dim=0)
            return stacked.reshape((-1,) + tuple(stacked.shape[2:]))

        bw_batch = LocalBwState(
            ego_features=_stack_field("ego_features", dtype=torch.float32),
            selected_sat_tokens=_stack_field("selected_sat_tokens", dtype=torch.float32),
            selected_sat_mask=_stack_field("selected_sat_mask", dtype=torch.bool),
            gu_tokens=_stack_field("gu_tokens", dtype=torch.float32),
            gu_mask=_stack_field("gu_mask", dtype=torch.bool),
            bw_valid_mask=_stack_field("bw_valid_mask", dtype=torch.bool),
        )
        agent_counts = [int(torch.as_tensor(snapshot.ego_features).shape[0]) for snapshot in bw_snapshots]
        with torch.inference_mode():
            bw_out = actor.act_bw(bw_batch, deterministic=deterministic)
        actions = [piece.numpy() for piece in _split_cpu_tensor_by_counts(bw_out.action, agent_counts)]
        return BatchedBwEvalOutputs(
            actions=actions,
            local_state=bw_batch,
            out=bw_out,
            agent_counts=agent_counts,
        )
    raise RuntimeError(
        "BW actor evaluation requires BW snapshots carrying redesigned LocalBwState tensors. "
        "Create snapshots with StructuredControlDriver.build_bw_stage_snapshot()."
    )


def batched_policy_bw_outputs_from_batch(
    actor,
    bw_world_batch: Any,
    bw_candidate_indices: torch.Tensor,
    bw_valid_mask: torch.Tensor,
    *,
    deterministic: bool,
    agent_counts: Sequence[int] | None = None,
) -> BatchedBwEvalOutputs:
    del actor, bw_world_batch, bw_candidate_indices, bw_valid_mask, deterministic, agent_counts
    raise RuntimeError(
        "BW actor evaluation requires BW snapshots carrying redesigned LocalBwState tensors. "
        "Legacy candidate-slot world snapshots are no longer supported."
    )
