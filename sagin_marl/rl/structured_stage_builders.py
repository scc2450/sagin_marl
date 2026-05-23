from __future__ import annotations

from dataclasses import fields

import numpy as np
import torch

from sagin_marl.env import channel
from sagin_marl.env.config import backhaul_carrier_freq_from_config as _backhaul_carrier_freq_from_cfg
from sagin_marl.env.numeric_guards import LOG_RATIO_EPS, NORMALIZATION_DENOM_EPS, geometry_denominator, normalize_scale
from . import structured_accel_actor_schema as accel_schema
from . import structured_bw_actor_schema as bw_schema
from . import structured_sat_actor_schema as sat_schema
from .structured_actor import _subset_member_tensor
from .structured_types import (
    BwStageSnapshot,
    LocalAccelState,
    LocalBwState,
    LocalSatState,
    StructuredWorldState,
)

_LONG_ARANGE_CACHE: dict[tuple[int, str, int | None], torch.Tensor] = {}
_BOOL_EYE_CACHE: dict[tuple[int, str, int | None], torch.Tensor] = {}


def _cached_arange(size: int, device: torch.device) -> torch.Tensor:
    key = (int(size), str(device.type), int(device.index) if device.index is not None else None)
    cached = _LONG_ARANGE_CACHE.get(key)
    if cached is not None and cached.device == device:
        return cached
    value = torch.arange(int(size), device=device, dtype=torch.long)
    _LONG_ARANGE_CACHE[key] = value
    return value


def _cached_eye(size: int, device: torch.device) -> torch.Tensor:
    key = (int(size), str(device.type), int(device.index) if device.index is not None else None)
    cached = _BOOL_EYE_CACHE.get(key)
    if cached is not None and cached.device == device:
        return cached
    value = torch.eye(int(size), device=device, dtype=torch.bool)
    _BOOL_EYE_CACHE[key] = value
    return value


def _to_torch(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    raise TypeError(f"Unsupported structured snapshot field type: {type(value)!r}")


def _remaining_horizon_frac(env) -> float:
    cfg = env.cfg
    t_steps = max(float(getattr(cfg, "T_steps", 1) or 1), 1.0)
    denom = max(t_steps - 1.0, 1.0)
    t_now = float(getattr(env, "t", 0) or 0)
    return float(np.clip((t_steps - 1.0 - t_now) / denom, 0.0, 1.0))


def world_state_to_torch(ws: StructuredWorldState) -> StructuredWorldState:
    kwargs = {field.name: _to_torch(getattr(ws, field.name)) for field in fields(ws)}
    return StructuredWorldState(**kwargs)


def bw_stage_snapshot_to_torch(snapshot: BwStageSnapshot) -> BwStageSnapshot:
    def _optional_to_torch(value):
        return None if value is None else _to_torch(value)

    return BwStageSnapshot(
        world_state=world_state_to_torch(snapshot.world_state),
        assoc=_to_torch(snapshot.assoc),
        selected_sat_indices=_to_torch(snapshot.selected_sat_indices),
        selected_sat_mask=_to_torch(snapshot.selected_sat_mask),
        access_gain_matrix=_to_torch(snapshot.access_gain_matrix),
        bw_valid_mask_full=_to_torch(snapshot.bw_valid_mask_full),
        ego_features=_optional_to_torch(snapshot.ego_features),
        selected_sat_tokens=_optional_to_torch(snapshot.selected_sat_tokens),
        gu_tokens=_optional_to_torch(snapshot.gu_tokens),
        gu_mask=_optional_to_torch(snapshot.gu_mask),
        bw_valid_mask=_optional_to_torch(snapshot.bw_valid_mask),
    )


def _gather_visible_sat_batches(
    ws: StructuredWorldState,
    *,
    max_visible: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sat_mask = ws.uav_sat_mask.to(dtype=torch.bool)
    batch_size, num_uav, sat_slots = sat_mask.shape
    flat_sat_mask = sat_mask.reshape(batch_size * num_uav, sat_slots)
    if max_visible is None:
        max_vis = int(sat_mask.sum(dim=-1).max().item()) if sat_slots > 0 else 0
    else:
        max_vis = min(max(int(max_visible), 0), int(sat_slots))
    sat_nodes_dim = int(ws.sat_nodes.shape[-1])
    sat_edge_dim = int(ws.uav_sat_edges.shape[-1])
    if max_vis <= 0:
        row_count = batch_size * num_uav
        empty_nodes = ws.sat_nodes.new_zeros((row_count, 0, sat_nodes_dim))
        empty_edges = ws.uav_sat_edges.new_zeros((row_count, 0, sat_edge_dim))
        empty_mask = flat_sat_mask.new_zeros((row_count, 0))
        return empty_nodes, empty_edges, empty_mask

    index_grid = _cached_arange(sat_slots, flat_sat_mask.device).unsqueeze(0).expand(batch_size * num_uav, -1)
    masked_pos = torch.where(flat_sat_mask, index_grid, torch.full_like(index_grid, sat_slots))
    order = torch.argsort(masked_pos, dim=-1)
    gathered_idx = order[:, :max_vis]
    local_mask = flat_sat_mask.gather(1, gathered_idx)
    sat_nodes = ws.sat_nodes.unsqueeze(1).expand(-1, num_uav, -1, -1).reshape(batch_size * num_uav, sat_slots, -1)
    sat_nodes_gather = sat_nodes.gather(
        1, gathered_idx.unsqueeze(-1).expand(-1, -1, sat_nodes.shape[-1])
    )
    sat_edges = ws.uav_sat_edges.reshape(batch_size * num_uav, sat_slots, -1).gather(
        1, gathered_idx.unsqueeze(-1).expand(-1, -1, ws.uav_sat_edges.shape[-1])
    )
    sat_nodes_gather = sat_nodes_gather * local_mask.unsqueeze(-1).to(sat_nodes_gather.dtype)
    sat_edges = sat_edges * local_mask.unsqueeze(-1).to(sat_edges.dtype)
    return sat_nodes_gather, sat_edges, local_mask


def _split_local_state_batch(batch_state):
    count = int(getattr(batch_state, fields(batch_state)[0].name).shape[0])
    states = []
    for idx in range(count):
        kwargs = {}
        for field in fields(batch_state):
            value = getattr(batch_state, field.name)
            if isinstance(batch_state, LocalSatState) and field.name == "subset_members" and value.ndim == 2:
                kwargs[field.name] = value.unsqueeze(0)
            else:
                kwargs[field.name] = value[idx : idx + 1]
        states.append(type(batch_state)(**kwargs))
    return states


def _positive_float(value: float, name: str) -> float:
    value_f = float(value)
    if not np.isfinite(value_f) or value_f <= 0.0:
        raise ValueError(f"{name} must be positive and finite for accel actor builder, got {value_f!r}.")
    return value_f


def _log_ratio_np(value: np.ndarray | float, ref: float) -> np.ndarray:
    ref_f = max(float(ref), float(LOG_RATIO_EPS))
    return np.log(np.maximum(np.asarray(value, dtype=np.float32), float(LOG_RATIO_EPS)) / ref_f).astype(np.float32)


def _log1p_nonnegative_np(value: np.ndarray | float) -> np.ndarray:
    return np.log1p(np.maximum(np.asarray(value, dtype=np.float32), 0.0)).astype(np.float32)


def _access_se_from_gain(env, gain: np.ndarray, access_noise_ref: float) -> np.ndarray:
    cfg = env.cfg
    snr = np.asarray(float(cfg.gu_tx_power) * np.asarray(gain, dtype=np.float32) / access_noise_ref, dtype=np.float32)
    if bool(getattr(cfg, "fading_enabled", False)) and channel.access_fading_mode_from_config(cfg) == "ergodic_rician":
        return np.asarray(
            channel.rician_ergodic_spectral_efficiency(
                snr,
                channel.rician_k_linear_from_config(cfg),
                quadrature_points=int(getattr(cfg, "access_ergodic_rician_quadrature_points", 16) or 16),
            ),
            dtype=np.float32,
        )
    return np.asarray(channel.spectral_efficiency(snr), dtype=np.float32)


def build_batched_local_accel_states_from_spec(spec: dict, *, device: torch.device | str | None = None) -> LocalAccelState:
    env = spec["env"]
    cfg = env.cfg
    num_uav = int(cfg.num_uav)
    num_gu = int(cfg.num_gu)
    num_sat = int(cfg.num_sat)
    sat_width = int(getattr(cfg, "per_uav_visible_sat_token_max", 0) or 0)
    if num_uav <= 0:
        raise ValueError("num_uav must be positive for accel actor builder.")
    if num_gu <= 0:
        raise ValueError("num_gu must be positive for accel actor builder.")
    if num_sat <= 0:
        raise ValueError("num_sat must be positive for accel actor builder.")
    if sat_width <= 0:
        raise ValueError("per_uav_visible_sat_token_max must be positive for accel actor builder.")

    owner = np.asarray(spec["assoc"], dtype=np.int32).reshape(num_gu)
    if np.any((owner < 0) | (owner >= num_uav)):
        raise ValueError("accel stage owner/assoc must assign every GU to a UAV.")
    access_gain = np.asarray(env._coerce_access_gain_matrix(spec["access_gain_matrix"]), dtype=np.float32)
    if access_gain.shape != (num_gu, num_uav):
        raise ValueError(f"access_gain_matrix must have shape {(num_gu, num_uav)}, got {access_gain.shape}.")

    arrival_ref = _positive_float(env._arrival_ref(), "arrival_ref_bits_per_step")
    gu_flow_ref = _positive_float(arrival_ref / float(num_gu), "gu_flow_ref")
    uav_flow_ref = _positive_float(arrival_ref / float(num_uav), "uav_flow_ref")
    sat_flow_ref = _positive_float(arrival_ref / _positive_float(env._bw_weighted_workload_sat_active_ref_count(), "sat_active_ref_count"), "sat_flow_ref")
    map_ref = _positive_float(float(cfg.map_size), "map_size")
    vel_ref = _positive_float(float(cfg.v_max), "v_max")
    accel_ref = _positive_float(float(cfg.a_max), "a_max")
    energy_ref = _positive_float(float(cfg.uav_energy_init), "uav_energy_init")
    gu_queue_ref = _positive_float(float(cfg.queue_max_gu), "queue_max_gu")
    uav_queue_ref = _positive_float(float(cfg.queue_max_uav), "queue_max_uav")
    sat_queue_ref = _positive_float(float(cfg.queue_max_sat), "queue_max_sat")
    orbit_pos_ref = _positive_float(float(cfg.r_earth + cfg.sat_height), "orbit_pos_ref")
    sat_vel_ref = _positive_float(float(np.sqrt(3.986004418e14 / orbit_pos_ref)), "sat_vel_ref")
    service_floor = _positive_float(env._bw_weighted_workload_eps(), "service_floor_bits_per_step")
    access_noise_ref = _positive_float(
        float(cfg.noise_density) * float(cfg.b_acc) * channel.noise_figure_linear(float(getattr(cfg, "access_noise_figure_db", 0.0) or 0.0)),
        "access_noise_ref",
    )
    backhaul_bw_ref = _positive_float(env._effective_b_backhaul_per_sat(), "effective_b_backhaul_per_sat")
    backhaul_noise_ref = _positive_float(
        float(cfg.noise_density)
        * backhaul_bw_ref
        * channel.noise_figure_linear(float(getattr(cfg, "backhaul_noise_figure_db", 0.0) or 0.0)),
        "backhaul_noise_ref",
    )
    uav_flow_cost_ref = 1.0 / uav_flow_ref
    gu_flow_cost_ref = 1.0 / gu_flow_ref
    sat_cost_ref = 1.0 / sat_flow_ref
    uav_total_cost_ref = uav_flow_cost_ref + sat_cost_ref
    gu_total_cost_ref = gu_flow_cost_ref + uav_total_cost_ref

    gu_ema, uav_ema, sat_ema = env._bw_weighted_workload_device_ema_vectors()
    gu_local_cost = (1.0 / np.maximum(gu_ema, service_floor)).astype(np.float32, copy=False)
    uav_local_cost = (1.0 / np.maximum(uav_ema, service_floor)).astype(np.float32, copy=False)
    sat_cost = (1.0 / np.maximum(sat_ema, service_floor)).astype(np.float32, copy=False)
    last_assoc = np.asarray(getattr(env, "last_association", np.full((num_gu,), -1, dtype=np.int32)), dtype=np.int32).reshape(num_gu)
    last_gu_total_cost, last_uav_total_cost, _last_sat_cost = env._bw_weighted_workload_device_costs(
        assoc_override=last_assoc,
        sat_selection_override=getattr(env, "last_sat_selection", None),
    )

    gu_pos = np.asarray(env.gu_pos, dtype=np.float32)
    uav_pos = np.asarray(env.uav_pos, dtype=np.float32)
    uav_vel = np.asarray(env.uav_vel, dtype=np.float32)
    gu_queue = np.asarray(env.gu_queue, dtype=np.float32).reshape(num_gu)
    uav_queue = np.asarray(env.uav_queue, dtype=np.float32).reshape(num_uav)
    sat_queue = np.asarray(env.sat_queue, dtype=np.float32).reshape(num_sat)
    gu_expected = np.asarray(env._current_expected_gu_arrival_rates(), dtype=np.float32).reshape(num_gu) * float(cfg.tau0)
    gu_last_arrival = np.asarray(getattr(env, "last_gu_arrival", np.zeros((num_gu,), dtype=np.float32)), dtype=np.float32).reshape(num_gu)
    gu_last_outflow = np.asarray(getattr(env, "last_gu_outflow", np.zeros((num_gu,), dtype=np.float32)), dtype=np.float32).reshape(num_gu)
    gu_drop = np.asarray(getattr(env, "gu_drop", np.zeros((num_gu,), dtype=np.float32)), dtype=np.float32).reshape(num_gu)
    uav_last_inflow = np.asarray(getattr(env, "last_gu_to_uav_inflow_by_uav", np.zeros((num_uav,), dtype=np.float32)), dtype=np.float32).reshape(num_uav)
    uav_last_outflow = np.asarray(getattr(env, "last_uav_outflow", np.zeros((num_uav,), dtype=np.float32)), dtype=np.float32).reshape(num_uav)
    uav_drop = np.asarray(getattr(env, "uav_drop", np.zeros((num_uav,), dtype=np.float32)), dtype=np.float32).reshape(num_uav)
    sat_last_incoming = np.asarray(getattr(env, "last_sat_incoming", np.zeros((num_sat,), dtype=np.float32)), dtype=np.float32).reshape(num_sat)
    sat_last_processed = np.asarray(getattr(env, "last_sat_processed", np.zeros((num_sat,), dtype=np.float32)), dtype=np.float32).reshape(num_sat)
    sat_drop = np.asarray(getattr(env, "sat_drop", np.zeros((num_sat,), dtype=np.float32)), dtype=np.float32).reshape(num_sat)
    last_bw_fraction = np.asarray(getattr(env, "last_bw_fraction_by_uav_gu", np.zeros((num_uav, num_gu), dtype=np.float32)), dtype=np.float32).reshape(num_uav, num_gu)
    last_selected_mask = np.asarray(getattr(env, "last_selected_mask_by_uav_sat", np.zeros((num_uav, num_sat), dtype=np.float32)), dtype=np.float32).reshape(num_uav, num_sat)
    last_uav_sat_outflow = np.asarray(getattr(env, "last_uav_to_sat_outflow_matrix", np.zeros((num_uav, num_sat), dtype=np.float32)), dtype=np.float32).reshape(num_uav, num_sat)
    last_access_interf = np.asarray(getattr(env, "last_access_interference_by_uav", np.zeros((num_uav,), dtype=np.float32)), dtype=np.float32).reshape(num_uav)

    diff_gu_uav = gu_pos[:, None, :] - uav_pos[None, :, :]
    dist_gu_uav = np.linalg.norm(diff_gu_uav, axis=-1).astype(np.float32)
    d_owner = dist_gu_uav[np.arange(num_gu), owner]
    if num_uav > 1:
        masked = dist_gu_uav.copy()
        masked[np.arange(num_gu), owner] = np.inf
        d_second = np.min(masked, axis=1)
    else:
        d_second = d_owner + map_ref
    owner_stability_margin = (d_second - d_owner) / map_ref
    if num_uav == 1:
        owner_stability_margin.fill(1.0)
    partition_boundary_weight = np.exp(-owner_stability_margin).astype(np.float32) if num_uav > 1 else np.zeros((num_gu,), dtype=np.float32)

    gu_access_se_by_uav = _access_se_from_gain(env, access_gain, access_noise_ref)
    se_owner = gu_access_se_by_uav[np.arange(num_gu), owner]
    owner_link_weakness = (1.0 / (1.0 + se_owner)).astype(np.float32)
    gu_local_cost_log_ratio = _log_ratio_np(gu_local_cost, gu_flow_cost_ref)
    gu_last_total_cost_log_ratio = _log_ratio_np(last_gu_total_cost, gu_total_cost_ref)
    gu_last_workload_raw = np.maximum(last_gu_total_cost * gu_queue, 0.0).astype(np.float32)
    gu_last_workload_log1p = _log1p_nonnegative_np(gu_last_workload_raw)
    gu_queue_steps_raw = (gu_queue / gu_flow_ref).astype(np.float32)
    gu_expected_steps_raw = (gu_expected / gu_flow_ref).astype(np.float32)
    gu_last_arrival_steps_raw = (gu_last_arrival / gu_flow_ref).astype(np.float32)
    gu_last_outflow_steps_raw = (gu_last_outflow / gu_flow_ref).astype(np.float32)
    gu_last_drop_steps_raw = (gu_drop / gu_flow_ref).astype(np.float32)
    gu_queue_steps = _log1p_nonnegative_np(gu_queue_steps_raw)
    gu_expected_steps = _log1p_nonnegative_np(gu_expected_steps_raw)
    gu_last_arrival_steps = _log1p_nonnegative_np(gu_last_arrival_steps_raw)
    gu_last_outflow_steps = _log1p_nonnegative_np(gu_last_outflow_steps_raw)
    gu_last_drop_steps = _log1p_nonnegative_np(gu_last_drop_steps_raw)
    gu_service_ema_steps = _log1p_nonnegative_np(gu_ema / gu_flow_ref)
    gu_demand_steps_raw = (gu_queue_steps_raw + gu_expected_steps_raw).astype(np.float32)
    gu_unserved = np.maximum(gu_demand_steps_raw - gu_last_outflow_steps_raw, 0.0).astype(np.float32)
    gu_access_pressure = (gu_unserved + gu_last_drop_steps_raw).astype(np.float32)
    gu_last_bw_sum = np.where(
        (last_assoc >= 0) & (last_assoc < num_uav),
        np.maximum(np.sum(last_bw_fraction, axis=0), 0.0),
        0.0,
    ).astype(np.float32)

    cell_summary = np.zeros((num_uav, accel_schema.ACCEL_CELL_DIM), dtype=np.float32)
    cell_workload_raw_share = np.zeros((num_uav,), dtype=np.float32)
    rel_owner_gu = ((gu_pos - uav_pos[owner]) / map_ref).astype(np.float32)
    for u in range(num_uav):
        in_cell = owner == u
        if np.any(in_cell):
            cell_summary[u, accel_schema.CELL_GU_COUNT_FRAC] = float(np.sum(in_cell)) / float(num_gu)
            cell_summary[u, accel_schema.CELL_QUEUE_STEPS_SUM] = float(np.log1p(max(float(np.sum(gu_queue_steps_raw[in_cell])) / float(num_gu), 0.0)))
            cell_summary[u, accel_schema.CELL_EXPECTED_ARRIVAL_STEPS_SUM] = float(np.log1p(max(float(np.sum(gu_expected_steps_raw[in_cell])) / float(num_gu), 0.0)))
            cell_summary[u, accel_schema.CELL_LAST_ARRIVAL_STEPS_SUM] = float(np.log1p(max(float(np.sum(gu_last_arrival_steps_raw[in_cell])) / float(num_gu), 0.0)))
            cell_summary[u, accel_schema.CELL_LAST_OUTFLOW_STEPS_SUM] = float(np.log1p(max(float(np.sum(gu_last_outflow_steps_raw[in_cell])) / float(num_gu), 0.0)))
            cell_summary[u, accel_schema.CELL_LAST_DROP_STEPS_SUM] = float(np.log1p(max(float(np.sum(gu_last_drop_steps_raw[in_cell])) / float(num_gu), 0.0)))
            cell_workload_raw_share[u] = max(float(np.sum(gu_last_workload_raw[in_cell])) / float(num_gu), 0.0)
            cell_summary[u, accel_schema.CELL_LAST_WORKLOAD_LOG1P_SUM] = float(np.log1p(cell_workload_raw_share[u]))
            cell_summary[u, accel_schema.CELL_BOUNDARY_WORKLOAD_SUM] = float(np.log1p(max(float(np.sum(gu_last_workload_raw[in_cell] * partition_boundary_weight[in_cell])) / float(num_gu), 0.0)))
            cell_summary[u, accel_schema.CELL_WEAK_LINK_WORKLOAD_SUM] = float(np.log1p(max(float(np.sum(gu_last_workload_raw[in_cell] * owner_link_weakness[in_cell])) / float(num_gu), 0.0)))
            cell_summary[u, accel_schema.CELL_ACCESS_PRESSURE] = float(np.sum(gu_access_pressure[in_cell])) / float(num_gu)

            demand_denom = float(np.sum(gu_demand_steps_raw[in_cell]))
            if demand_denom > 0.0:
                cell_summary[u, accel_schema.CELL_DEMAND_MOMENT_X : accel_schema.CELL_DEMAND_MOMENT_Y + 1] = (
                    np.sum(gu_demand_steps_raw[in_cell, None] * rel_owner_gu[in_cell], axis=0) / demand_denom
                )
            boundary_weighted = gu_last_workload_raw[in_cell] * partition_boundary_weight[in_cell]
            boundary_denom = float(np.sum(boundary_weighted))
            if boundary_denom > 0.0:
                cell_summary[u, accel_schema.CELL_BOUNDARY_MOMENT_X : accel_schema.CELL_BOUNDARY_MOMENT_Y + 1] = (
                    np.sum(boundary_weighted[:, None] * rel_owner_gu[in_cell], axis=0) / boundary_denom
                )
            weak_weighted = gu_last_workload_raw[in_cell] * owner_link_weakness[in_cell]
            weak_denom = float(np.sum(weak_weighted))
            if weak_denom > 0.0:
                cell_summary[u, accel_schema.CELL_WEAK_LINK_MOMENT_X : accel_schema.CELL_WEAK_LINK_MOMENT_Y + 1] = (
                    np.sum(weak_weighted[:, None] * rel_owner_gu[in_cell], axis=0) / weak_denom
                )
        interf_power = float(np.sum(float(cfg.gu_tx_power) * access_gain[last_assoc != u, u] * gu_last_bw_sum[last_assoc != u]))
        cell_summary[u, accel_schema.CELL_INTERFERENCE_EXPOSURE] = float(np.log1p(max(interf_power / access_noise_ref, 0.0)))

    total_cell_workload = float(np.sum(cell_workload_raw_share))
    if total_cell_workload > 0.0:
        cell_summary[:, accel_schema.CELL_WORKLOAD_SHARE_GAP] = (
            cell_workload_raw_share / total_cell_workload - 1.0 / float(num_uav)
        )

    row_count = num_uav
    ego_features = np.zeros((row_count, accel_schema.ACCEL_EGO_DIM), dtype=np.float32)
    ego_cell = np.zeros((row_count, accel_schema.ACCEL_CELL_DIM), dtype=np.float32)
    gu_tokens = np.zeros((row_count, num_gu, accel_schema.ACCEL_GU_TOKEN_DIM), dtype=np.float32)
    gu_mask = np.ones((row_count, num_gu), dtype=bool)
    peer_tokens = np.zeros((row_count, max(num_uav - 1, 0), accel_schema.ACCEL_PEER_TOKEN_DIM), dtype=np.float32)
    peer_mask = np.ones((row_count, max(num_uav - 1, 0)), dtype=bool)
    sat_tokens = np.zeros((row_count, sat_width, accel_schema.ACCEL_SAT_TOKEN_DIM), dtype=np.float32)
    sat_mask = np.zeros((row_count, sat_width), dtype=bool)

    last_policy_accel = np.asarray(getattr(env, "last_policy_accel", np.zeros((num_uav, 2), dtype=np.float32)), dtype=np.float32).reshape(num_uav, 2)
    last_exec_accel = np.asarray(getattr(env, "last_exec_accel", np.zeros((num_uav, 2), dtype=np.float32)), dtype=np.float32).reshape(num_uav, 2)
    intervention = last_exec_accel - last_policy_accel
    remaining_horizon_frac = _remaining_horizon_frac(env)
    sat_select_ref_count = _sat_action_select_k_from_cfg(cfg)
    if sat_select_ref_count <= 0:
        raise ValueError("sat_select_ref_count must be positive for accel actor builder.")
    d_alert = float(getattr(cfg, "avoidance_alert_factor", 1.5) or 1.5) * float(cfg.d_safe)

    for ego in range(num_uav):
        ego_features[ego, accel_schema.EGO_X : accel_schema.EGO_Y + 1] = uav_pos[ego] / map_ref
        ego_features[ego, accel_schema.EGO_VX : accel_schema.EGO_VY + 1] = uav_vel[ego] / vel_ref
        ego_features[ego, accel_schema.EGO_SPEED] = float(np.linalg.norm(uav_vel[ego]) / vel_ref)
        ego_features[ego, accel_schema.EGO_ENERGY] = float(env.uav_energy[ego] / energy_ref)
        ego_features[ego, accel_schema.EGO_BOUNDARY_LEFT] = float(uav_pos[ego, 0] / map_ref)
        ego_features[ego, accel_schema.EGO_BOUNDARY_RIGHT] = float((map_ref - uav_pos[ego, 0]) / map_ref)
        ego_features[ego, accel_schema.EGO_BOUNDARY_BOTTOM] = float(uav_pos[ego, 1] / map_ref)
        ego_features[ego, accel_schema.EGO_BOUNDARY_TOP] = float((map_ref - uav_pos[ego, 1]) / map_ref)
        ego_features[ego, accel_schema.EGO_UAV_QUEUE_STEPS] = float(np.log1p(max(float(uav_queue[ego] / uav_flow_ref), 0.0)))
        ego_features[ego, accel_schema.EGO_UAV_QUEUE_FILL] = float(uav_queue[ego] / uav_queue_ref)
        ego_features[ego, accel_schema.EGO_UAV_LAST_INFLOW_STEPS] = float(np.log1p(max(float(uav_last_inflow[ego] / uav_flow_ref), 0.0)))
        ego_features[ego, accel_schema.EGO_UAV_LAST_OUTFLOW_STEPS] = float(np.log1p(max(float(uav_last_outflow[ego] / uav_flow_ref), 0.0)))
        ego_features[ego, accel_schema.EGO_UAV_LAST_DROP_STEPS] = float(np.log1p(max(float(uav_drop[ego] / uav_flow_ref), 0.0)))
        ego_features[ego, accel_schema.EGO_UAV_SERVICE_EMA_STEPS] = float(np.log1p(max(float(uav_ema[ego] / uav_flow_ref), 0.0)))
        ego_features[ego, accel_schema.EGO_UAV_LOCAL_COST_LOG_RATIO] = float(_log_ratio_np(uav_local_cost[ego], uav_flow_cost_ref))
        ego_features[ego, accel_schema.EGO_UAV_LAST_TOTAL_COST_LOG_RATIO] = float(_log_ratio_np(last_uav_total_cost[ego], uav_total_cost_ref))
        ego_features[ego, accel_schema.EGO_UAV_LAST_WORKLOAD_LOG1P] = float(np.log1p(max(last_uav_total_cost[ego] * uav_queue[ego], 0.0)))
        ego_features[ego, accel_schema.EGO_UAV_LAST_ACCESS_INTERFERENCE_LOG1P] = float(np.log1p(max(last_access_interf[ego] / access_noise_ref, 0.0)))
        ego_features[ego, accel_schema.EGO_LAST_POLICY_ACCEL_X : accel_schema.EGO_LAST_POLICY_ACCEL_Y + 1] = last_policy_accel[ego] / accel_ref
        ego_features[ego, accel_schema.EGO_LAST_EXEC_ACCEL_X : accel_schema.EGO_LAST_EXEC_ACCEL_Y + 1] = last_exec_accel[ego] / accel_ref
        ego_features[ego, accel_schema.EGO_LAST_INTERVENTION_DX : accel_schema.EGO_LAST_INTERVENTION_DY + 1] = intervention[ego] / accel_ref
        ego_features[ego, accel_schema.EGO_LAST_INTERVENTION_L2] = float(np.linalg.norm(intervention[ego]) / accel_ref)
        ego_features[ego, accel_schema.EGO_REMAINING_HORIZON_FRAC] = remaining_horizon_frac
        ego_cell[ego] = cell_summary[ego]

        gu_tokens[ego, :, accel_schema.GU_X : accel_schema.GU_Y + 1] = gu_pos / map_ref
        gu_tokens[ego, :, accel_schema.GU_QUEUE_STEPS] = gu_queue_steps
        gu_tokens[ego, :, accel_schema.GU_QUEUE_FILL] = gu_queue / gu_queue_ref
        gu_tokens[ego, :, accel_schema.GU_EXPECTED_ARRIVAL_STEPS] = gu_expected_steps
        gu_tokens[ego, :, accel_schema.GU_LAST_ARRIVAL_STEPS] = gu_last_arrival_steps
        gu_tokens[ego, :, accel_schema.GU_LAST_OUTFLOW_STEPS] = gu_last_outflow_steps
        gu_tokens[ego, :, accel_schema.GU_LAST_DROP_STEPS] = gu_last_drop_steps
        gu_tokens[ego, :, accel_schema.GU_SERVICE_EMA_STEPS] = gu_service_ema_steps
        gu_tokens[ego, :, accel_schema.GU_LOCAL_COST_LOG_RATIO] = gu_local_cost_log_ratio
        gu_tokens[ego, :, accel_schema.GU_LAST_TOTAL_COST_LOG_RATIO] = gu_last_total_cost_log_ratio
        gu_tokens[ego, :, accel_schema.GU_LAST_WORKLOAD_LOG1P] = gu_last_workload_log1p
        gu_tokens[ego, :, accel_schema.GU_REL_X : accel_schema.GU_REL_Y + 1] = (gu_pos - uav_pos[ego]) / map_ref
        gu_tokens[ego, :, accel_schema.GU_DIST] = dist_gu_uav[:, ego] / map_ref
        gu_tokens[ego, :, accel_schema.GU_ACCESS_SE_REF] = gu_access_se_by_uav[:, ego]
        last_assoc_to_ego = (last_assoc == ego)
        gu_tokens[ego, :, accel_schema.GU_LAST_ASSOC_TO_EGO] = last_assoc_to_ego.astype(np.float32)
        gu_tokens[ego, :, accel_schema.GU_LAST_BW_FRACTION_EGO] = last_bw_fraction[ego]
        gu_tokens[ego, :, accel_schema.GU_LAST_SERVED_BY_EGO] = last_bw_fraction[ego]
        gu_tokens[ego, :, accel_schema.GU_PRE_OWNER_IS_EGO] = (owner == ego).astype(np.float32)
        if num_uav > 1:
            best_other_ego = np.min(np.where(np.arange(num_uav)[None, :] == ego, np.inf, dist_gu_uav), axis=1)
            handoff_margin = (best_other_ego - dist_gu_uav[:, ego]) / map_ref
            takeover_gap = (dist_gu_uav[:, ego] - d_owner) / map_ref
        else:
            handoff_margin = np.ones((num_gu,), dtype=np.float32)
            takeover_gap = np.zeros((num_gu,), dtype=np.float32)
        gu_tokens[ego, :, accel_schema.GU_HANDOFF_MARGIN_EGO] = handoff_margin
        gu_tokens[ego, :, accel_schema.GU_OWNER_STABILITY_MARGIN] = owner_stability_margin
        gu_tokens[ego, :, accel_schema.GU_EGO_TAKEOVER_GAP] = takeover_gap
        gu_tokens[ego, :, accel_schema.GU_PARTITION_BOUNDARY_WEIGHT] = partition_boundary_weight
        gu_tokens[ego, :, accel_schema.GU_EGO_LINK_WEAKNESS] = 1.0 / (1.0 + gu_access_se_by_uav[:, ego])
        gu_tokens[ego, :, accel_schema.GU_LAST_BW_SUM] = gu_last_bw_sum
        nonself_power = float(cfg.gu_tx_power) * access_gain[:, ego] * gu_last_bw_sum * (last_assoc != ego).astype(np.float32)
        gu_tokens[ego, :, accel_schema.GU_LAST_NONSELF_INTERFERENCE_LOG1P] = np.log1p(np.maximum(nonself_power / access_noise_ref, 0.0))

        peer_slot = 0
        for peer in range(num_uav):
            if peer == ego:
                continue
            rel_pos = uav_pos[ego] - uav_pos[peer]
            rel_vel = uav_vel[ego] - uav_vel[peer]
            dist = float(np.linalg.norm(rel_pos))
            closing = max(0.0, -float(np.dot(rel_pos, rel_vel)) / (dist * vel_ref)) if dist > 0.0 else 0.0
            peer_tokens[ego, peer_slot, accel_schema.PEER_REL_X : accel_schema.PEER_REL_Y + 1] = rel_pos / map_ref
            peer_tokens[ego, peer_slot, accel_schema.PEER_REL_VX : accel_schema.PEER_REL_VY + 1] = rel_vel / vel_ref
            peer_tokens[ego, peer_slot, accel_schema.PEER_DIST] = dist / map_ref
            peer_tokens[ego, peer_slot, accel_schema.PEER_CLOSING_SPEED] = closing
            peer_tokens[ego, peer_slot, accel_schema.PEER_SAFE_DISTANCE_MARGIN] = (dist - float(cfg.d_safe)) / map_ref
            peer_tokens[ego, peer_slot, accel_schema.PEER_UNSAFE_FLAG] = 1.0 if dist < float(cfg.d_safe) else 0.0
            peer_tokens[ego, peer_slot, accel_schema.PEER_ALERT_FLAG] = 1.0 if dist < d_alert else 0.0
            peer_tokens[ego, peer_slot, accel_schema.PEER_LAST_SHARED_SAT_FRAC] = (
                float(np.sum(last_selected_mask[ego] * last_selected_mask[peer])) / float(sat_select_ref_count)
            )
            peer_tokens[
                ego,
                peer_slot,
                accel_schema.PEER_CELL_OFFSET : accel_schema.PEER_CELL_OFFSET + accel_schema.ACCEL_CELL_DIM,
            ] = cell_summary[peer]
            peer_slot += 1

    sat_pos = np.asarray(spec["sat_pos"], dtype=np.float32).reshape(num_sat, 3)
    sat_vel = np.asarray(spec["sat_vel"], dtype=np.float32).reshape(num_sat, 3)
    visible = [list(v) for v in spec["visible"]]
    elevation_matrix = env._get_elevation_matrix(sat_pos)
    loss_matrix = env._get_backhaul_loss_matrix(sat_pos)
    sat_proc_capacity_steps = float(np.log1p(max(
        float(env._effective_sat_cpu_freq())
        / _positive_float(float(cfg.task_cycles_per_bit), "task_cycles_per_bit")
        * float(cfg.tau0)
        / sat_flow_ref,
        0.0,
    )))
    sat_last_workload_log1p = np.log1p(np.maximum(sat_cost * sat_queue, 0.0)).astype(np.float32)
    sat_last_selected_load_frac = np.sum(last_selected_mask, axis=0) / float(num_uav)
    for ego in range(num_uav):
        sat_ids = [int(s) for s in visible[ego][:sat_width] if 0 <= int(s) < num_sat]
        if not sat_ids:
            continue
        ids = np.asarray(sat_ids, dtype=np.int64)
        n = int(ids.size)
        sat_mask[ego, :n] = True
        sat_tokens[ego, :n, accel_schema.SAT_X : accel_schema.SAT_Z + 1] = sat_pos[ids] / orbit_pos_ref
        sat_tokens[ego, :n, accel_schema.SAT_VX : accel_schema.SAT_VZ + 1] = sat_vel[ids] / sat_vel_ref
        sat_tokens[ego, :n, accel_schema.SAT_QUEUE_STEPS] = _log1p_nonnegative_np(sat_queue[ids] / sat_flow_ref)
        sat_tokens[ego, :n, accel_schema.SAT_QUEUE_FILL] = sat_queue[ids] / sat_queue_ref
        sat_tokens[ego, :n, accel_schema.SAT_LAST_INCOMING_STEPS] = _log1p_nonnegative_np(sat_last_incoming[ids] / sat_flow_ref)
        sat_tokens[ego, :n, accel_schema.SAT_LAST_PROCESSED_STEPS] = _log1p_nonnegative_np(sat_last_processed[ids] / sat_flow_ref)
        sat_tokens[ego, :n, accel_schema.SAT_LAST_DROP_STEPS] = _log1p_nonnegative_np(sat_drop[ids] / sat_flow_ref)
        sat_tokens[ego, :n, accel_schema.SAT_SERVICE_EMA_STEPS] = _log1p_nonnegative_np(sat_ema[ids] / sat_flow_ref)
        sat_tokens[ego, :n, accel_schema.SAT_COST_LOG_RATIO] = _log_ratio_np(sat_cost[ids], sat_cost_ref)
        sat_tokens[ego, :n, accel_schema.SAT_LAST_WORKLOAD_LOG1P] = sat_last_workload_log1p[ids]
        sat_tokens[ego, :n, accel_schema.SAT_LAST_SELECTED_LOAD_FRAC] = sat_last_selected_load_frac[ids]
        sat_tokens[ego, :n, accel_schema.SAT_PROC_CAPACITY_STEPS] = sat_proc_capacity_steps
        rel_pos = sat_pos[ids] - env._uav_ecef(ego)[None, :]
        rel_vel = sat_vel[ids] - env._uav_vel_ecef(ego)[None, :]
        range_m = np.linalg.norm(rel_pos, axis=1).astype(np.float32)
        if np.any(range_m <= 0.0):
            raise ValueError("UAV and SAT ECEF positions overlap; cannot build accel SAT token.")
        radial = np.sum(rel_pos * rel_vel, axis=1) / (range_m * sat_vel_ref)
        gain = (float(env._backhaul_gain_const) / (range_m * range_m)).astype(np.float32)
        if loss_matrix is not None:
            gain = gain * loss_matrix[ego, ids]
        snr = float(cfg.uav_tx_power) * gain / backhaul_noise_ref
        if bool(getattr(cfg, "doppler_enabled", False) or getattr(cfg, "doppler_atten_enabled", False) or getattr(cfg, "doppler_observed", False)):
            raw_nu = env._doppler_many(ego, ids, sat_pos, sat_vel)
            nu_eff, _ = env._effective_doppler_array(ego, ids, raw_nu)
            nu_max = _positive_float(float(cfg.nu_max), "nu_max")
            doppler_ratio = nu_eff / nu_max
        else:
            nu_eff = np.zeros((n,), dtype=np.float32)
            doppler_ratio = np.zeros((n,), dtype=np.float32)
        if bool(getattr(cfg, "doppler_atten_enabled", False)):
            snr = snr * channel.doppler_attenuation(nu_eff, float(cfg.subcarrier_spacing))
        backhaul_se = np.asarray(channel.spectral_efficiency(snr), dtype=np.float32)
        valid = elevation_matrix[ego, ids] >= float(cfg.theta_min_rad)
        if bool(getattr(cfg, "doppler_enabled", False)):
            valid = valid & (np.abs(nu_eff) <= float(cfg.nu_max))
        sat_tokens[ego, :n, accel_schema.SAT_REL_X : accel_schema.SAT_REL_Z + 1] = rel_pos / orbit_pos_ref
        sat_tokens[ego, :n, accel_schema.SAT_REL_VX : accel_schema.SAT_REL_VZ + 1] = rel_vel / sat_vel_ref
        sat_tokens[ego, :n, accel_schema.SAT_RANGE] = range_m / orbit_pos_ref
        sat_tokens[ego, :n, accel_schema.SAT_RADIAL_VELOCITY] = radial
        sat_tokens[ego, :n, accel_schema.SAT_ELEVATION] = elevation_matrix[ego, ids] / (np.pi * 0.5)
        sat_tokens[ego, :n, accel_schema.SAT_DOPPLER_RATIO] = doppler_ratio
        sat_tokens[ego, :n, accel_schema.SAT_DOPPLER_ABS_RATIO] = np.abs(doppler_ratio)
        sat_tokens[ego, :n, accel_schema.SAT_BACKHAUL_SE_REF] = backhaul_se
        sat_tokens[ego, :n, accel_schema.SAT_VISIBLE_FLAG] = 1.0
        sat_tokens[ego, :n, accel_schema.SAT_VALID_FLAG] = valid.astype(np.float32)
        sat_tokens[ego, :n, accel_schema.SAT_LAST_SELECTED_FLAG] = last_selected_mask[ego, ids]
        sat_tokens[ego, :n, accel_schema.SAT_LAST_OUTFLOW_STEPS] = _log1p_nonnegative_np(last_uav_sat_outflow[ego, ids] / uav_flow_ref)

    tensors = {
        "ego_features": torch.as_tensor(ego_features, dtype=torch.float32),
        "ego_cell": torch.as_tensor(ego_cell, dtype=torch.float32),
        "gu_tokens": torch.as_tensor(gu_tokens, dtype=torch.float32),
        "gu_mask": torch.as_tensor(gu_mask, dtype=torch.bool),
        "peer_tokens": torch.as_tensor(peer_tokens, dtype=torch.float32),
        "peer_mask": torch.as_tensor(peer_mask, dtype=torch.bool),
        "sat_tokens": torch.as_tensor(sat_tokens, dtype=torch.float32),
        "sat_mask": torch.as_tensor(sat_mask, dtype=torch.bool),
    }
    if device is not None:
        torch_device = torch.device(device)
        tensors = {name: tensor.to(device=torch_device) for name, tensor in tensors.items()}
    return LocalAccelState(**tensors)


def build_local_accel_states_from_spec(spec: dict, *, device: torch.device | str | None = None) -> list[LocalAccelState]:
    return _split_local_state_batch(build_batched_local_accel_states_from_spec(spec, device=device))


def _sat_action_select_k_from_cfg(cfg) -> int:
    raw = getattr(cfg, "sat_action_select_k", None)
    if raw is not None and int(raw) > 0:
        return int(raw)
    select_k = min(
        int(cfg.num_sat),
        int(cfg.N_RF),
        int(cfg.sat_num_select) if getattr(cfg, "sat_num_select", None) is not None and int(cfg.sat_num_select) > 0 else int(cfg.N_RF),
    )
    try:
        setattr(cfg, "sat_action_select_k", int(select_k))
    except Exception:
        pass
    return int(select_k)


def build_batched_local_sat_states_from_spec(
    spec: dict,
    *,
    device: torch.device | str | None = None,
) -> LocalSatState:
    env = spec["env"]
    cfg = env.cfg
    num_uav = int(cfg.num_uav)
    num_gu = int(cfg.num_gu)
    num_sat = int(cfg.num_sat)
    if num_uav <= 0 or num_sat <= 0:
        raise ValueError("num_uav and num_sat must be positive for SAT actor builder.")
    assoc = np.asarray(spec["assoc"], dtype=np.int32).reshape(num_gu)
    access_gain = np.asarray(env._coerce_access_gain_matrix(spec["access_gain_matrix"]), dtype=np.float32)
    if access_gain.shape != (num_gu, num_uav):
        raise ValueError(f"access_gain_matrix must have shape {(num_gu, num_uav)}, got {access_gain.shape}.")
    sat_pos = np.asarray(spec["sat_pos"], dtype=np.float32).reshape(num_sat, 3)
    sat_vel = np.asarray(spec["sat_vel"], dtype=np.float32).reshape(num_sat, 3)
    visible = [list(v) for v in spec["visible"]]
    sat_width = int(getattr(cfg, "per_uav_visible_sat_token_max", 0) or 0)
    if sat_width <= 0:
        raise ValueError("per_uav_visible_sat_token_max must be positive for SAT actor builder.")
    sat_width = min(sat_width, num_sat)
    select_k = int(getattr(cfg, "sat_action_select_k", 0) or 0)
    if select_k <= 0:
        select_k = _sat_action_select_k_from_cfg(cfg)
    if select_k <= 0:
        raise ValueError("sat_action_select_k must be positive for SAT actor builder.")

    arrival_ref = _positive_float(env._arrival_ref(), "arrival_ref_bits_per_step")
    gu_flow_ref = _positive_float(arrival_ref / float(num_gu), "gu_flow_ref")
    uav_flow_ref = _positive_float(arrival_ref / float(num_uav), "uav_flow_ref")
    sat_flow_ref = _positive_float(arrival_ref / _positive_float(env._bw_weighted_workload_sat_active_ref_count(), "sat_active_ref_count"), "sat_flow_ref")
    gu_queue_ref = _positive_float(float(cfg.queue_max_gu), "queue_max_gu")
    uav_queue_ref = _positive_float(float(cfg.queue_max_uav), "queue_max_uav")
    sat_queue_ref = _positive_float(float(cfg.queue_max_sat), "queue_max_sat")
    service_floor = _positive_float(env._bw_weighted_workload_eps(), "service_floor_bits_per_step")
    map_ref = _positive_float(float(cfg.map_size), "map_size")
    orbit_pos_ref = _positive_float(float(cfg.r_earth + cfg.sat_height), "sat_orbit_radius")
    sat_vel_ref = _positive_float(float(np.sqrt(3.986004418e14 / orbit_pos_ref)), "sat_vel_ref")
    access_noise_ref = _positive_float(
        float(cfg.noise_density) * float(cfg.b_acc) * channel.noise_figure_linear(float(getattr(cfg, "access_noise_figure_db", 0.0) or 0.0)),
        "access_noise_ref",
    )
    backhaul_bw_ref = _positive_float(env._effective_b_backhaul_per_sat(), "effective_b_backhaul_per_sat")
    backhaul_noise_ref = _positive_float(
        float(cfg.noise_density)
        * backhaul_bw_ref
        * channel.noise_figure_linear(float(getattr(cfg, "backhaul_noise_figure_db", 0.0) or 0.0)),
        "backhaul_noise_ref",
    )
    doppler_ref = 1.0
    if bool(getattr(cfg, "doppler_enabled", False) or getattr(cfg, "doppler_atten_enabled", False) or getattr(cfg, "doppler_observed", False)):
        doppler_ref = _positive_float(
            float(_backhaul_carrier_freq_from_cfg(cfg)) * sat_vel_ref / float(cfg.speed_of_light),
            "doppler_ref",
        )
        _positive_float(float(cfg.nu_max), "nu_max")

    gu_flow_cost_ref = 1.0 / gu_flow_ref
    uav_flow_cost_ref = 1.0 / uav_flow_ref
    sat_cost_ref = 1.0 / sat_flow_ref
    uav_total_cost_ref = uav_flow_cost_ref + sat_cost_ref
    gu_total_cost_ref = gu_flow_cost_ref + uav_total_cost_ref

    gu_ema, uav_ema, sat_ema = env._bw_weighted_workload_device_ema_vectors()
    gu_local_cost = (1.0 / np.maximum(gu_ema, service_floor)).astype(np.float32, copy=False)
    uav_local_cost = (1.0 / np.maximum(uav_ema, service_floor)).astype(np.float32, copy=False)
    sat_cost = (1.0 / np.maximum(sat_ema, service_floor)).astype(np.float32, copy=False)
    last_assoc = np.asarray(getattr(env, "last_association", np.full((num_gu,), -1, dtype=np.int32)), dtype=np.int32).reshape(num_gu)
    last_gu_total_cost, last_uav_total_cost, _last_sat_cost = env._bw_weighted_workload_device_costs(
        assoc_override=last_assoc,
        sat_selection_override=getattr(env, "last_sat_selection", None),
    )

    gu_queue = np.asarray(env.gu_queue, dtype=np.float32).reshape(num_gu)
    uav_queue = np.asarray(env.uav_queue, dtype=np.float32).reshape(num_uav)
    sat_queue = np.asarray(env.sat_queue, dtype=np.float32).reshape(num_sat)
    gu_expected = np.asarray(env._current_expected_gu_arrival_rates(), dtype=np.float32).reshape(num_gu) * float(cfg.tau0)
    gu_last_arrival = np.asarray(getattr(env, "last_gu_arrival", np.zeros((num_gu,), dtype=np.float32)), dtype=np.float32).reshape(num_gu)
    gu_last_outflow = np.asarray(getattr(env, "last_gu_outflow", np.zeros((num_gu,), dtype=np.float32)), dtype=np.float32).reshape(num_gu)
    gu_drop = np.asarray(getattr(env, "gu_drop", np.zeros((num_gu,), dtype=np.float32)), dtype=np.float32).reshape(num_gu)
    uav_last_inflow = np.asarray(getattr(env, "last_gu_to_uav_inflow_by_uav", np.zeros((num_uav,), dtype=np.float32)), dtype=np.float32).reshape(num_uav)
    uav_last_outflow = np.asarray(getattr(env, "last_uav_outflow", np.zeros((num_uav,), dtype=np.float32)), dtype=np.float32).reshape(num_uav)
    uav_drop = np.asarray(getattr(env, "uav_drop", np.zeros((num_uav,), dtype=np.float32)), dtype=np.float32).reshape(num_uav)
    sat_last_incoming = np.asarray(getattr(env, "last_sat_incoming", np.zeros((num_sat,), dtype=np.float32)), dtype=np.float32).reshape(num_sat)
    sat_last_processed = np.asarray(getattr(env, "last_sat_processed", np.zeros((num_sat,), dtype=np.float32)), dtype=np.float32).reshape(num_sat)
    sat_drop = np.asarray(getattr(env, "sat_drop", np.zeros((num_sat,), dtype=np.float32)), dtype=np.float32).reshape(num_sat)
    last_selected_mask = np.asarray(getattr(env, "last_selected_mask_by_uav_sat", np.zeros((num_uav, num_sat), dtype=np.float32)), dtype=np.float32).reshape(num_uav, num_sat)
    last_uav_sat_outflow = np.asarray(getattr(env, "last_uav_to_sat_outflow_matrix", np.zeros((num_uav, num_sat), dtype=np.float32)), dtype=np.float32).reshape(num_uav, num_sat)
    last_access_interf = np.asarray(getattr(env, "last_access_interference_by_uav", np.zeros((num_uav,), dtype=np.float32)), dtype=np.float32).reshape(num_uav)

    gu_last_total_cost_log_ratio = _log_ratio_np(last_gu_total_cost, gu_total_cost_ref)
    uav_local_cost_log_ratio = _log_ratio_np(uav_local_cost, uav_flow_cost_ref)
    uav_last_total_cost_log_ratio = _log_ratio_np(last_uav_total_cost, uav_total_cost_ref)
    sat_cost_log_ratio = _log_ratio_np(sat_cost, sat_cost_ref)
    gu_last_workload_raw = np.maximum(last_gu_total_cost * gu_queue, 0.0).astype(np.float32)
    gu_last_workload_log1p = _log1p_nonnegative_np(gu_last_workload_raw)
    uav_last_workload_log1p = _log1p_nonnegative_np(last_uav_total_cost * uav_queue)
    sat_last_workload_log1p = _log1p_nonnegative_np(sat_cost * sat_queue)

    access_se_full_ref = _access_se_from_gain(env, access_gain, access_noise_ref)
    access_rate_full_bw_ref_raw = (float(cfg.b_acc) * access_se_full_ref * float(cfg.tau0) / gu_flow_ref).astype(np.float32)
    access_rate_full_bw_ref = _log1p_nonnegative_np(access_rate_full_bw_ref_raw)

    gu_queue_steps_raw = (gu_queue / gu_flow_ref).astype(np.float32)
    gu_expected_steps_raw = (gu_expected / gu_flow_ref).astype(np.float32)
    gu_last_arrival_steps_raw = (gu_last_arrival / gu_flow_ref).astype(np.float32)
    gu_last_outflow_steps_raw = (gu_last_outflow / gu_flow_ref).astype(np.float32)
    gu_last_drop_steps_raw = (gu_drop / gu_flow_ref).astype(np.float32)
    gu_queue_steps = _log1p_nonnegative_np(gu_queue_steps_raw)
    gu_expected_steps = _log1p_nonnegative_np(gu_expected_steps_raw)
    gu_last_arrival_steps = _log1p_nonnegative_np(gu_last_arrival_steps_raw)
    gu_last_outflow_steps = _log1p_nonnegative_np(gu_last_outflow_steps_raw)
    gu_last_drop_steps = _log1p_nonnegative_np(gu_last_drop_steps_raw)
    uav_queue_steps = _log1p_nonnegative_np(uav_queue / uav_flow_ref)
    uav_last_inflow_steps = _log1p_nonnegative_np(uav_last_inflow / uav_flow_ref)
    uav_last_outflow_steps = _log1p_nonnegative_np(uav_last_outflow / uav_flow_ref)
    uav_last_drop_steps = _log1p_nonnegative_np(uav_drop / uav_flow_ref)
    uav_service_ema_steps = _log1p_nonnegative_np(uav_ema / uav_flow_ref)
    sat_queue_steps = _log1p_nonnegative_np(sat_queue / sat_flow_ref)
    sat_last_incoming_steps = _log1p_nonnegative_np(sat_last_incoming / sat_flow_ref)
    sat_last_processed_steps = _log1p_nonnegative_np(sat_last_processed / sat_flow_ref)
    sat_last_drop_steps = _log1p_nonnegative_np(sat_drop / sat_flow_ref)
    sat_service_ema_steps = _log1p_nonnegative_np(sat_ema / sat_flow_ref)

    ego_features = np.zeros((num_uav, sat_schema.SAT_EGO_DIM), dtype=np.float32)
    demand_features = np.zeros((num_uav, sat_schema.SAT_DEMAND_DIM), dtype=np.float32)
    role_features = np.zeros((num_uav, sat_schema.SAT_ROLE_DIM), dtype=np.float32)
    sat_tokens = np.zeros((num_uav, sat_width, sat_schema.SAT_TOKEN_DIM), dtype=np.float32)
    sat_mask = np.zeros((num_uav, sat_width), dtype=bool)
    sat_valid_mask = np.zeros((num_uav, sat_width), dtype=bool)
    candidate_sat_ids = np.full((num_uav, sat_width), -1, dtype=np.int64)

    sat_proc_capacity_steps = float(np.log1p(max(
        float(env._effective_sat_cpu_freq())
        / _positive_float(float(cfg.task_cycles_per_bit), "task_cycles_per_bit")
        * float(cfg.tau0)
        / sat_flow_ref,
        0.0,
    )))
    sat_last_selected_load_frac = np.sum(last_selected_mask, axis=0) / float(num_uav)
    remaining_horizon_frac = _remaining_horizon_frac(env)
    elevation_matrix = env._get_elevation_matrix(sat_pos)
    loss_matrix = env._get_backhaul_loss_matrix(sat_pos)

    for ego in range(num_uav):
        ego_features[ego, sat_schema.EGO_UAV_QUEUE_STEPS] = uav_queue_steps[ego]
        ego_features[ego, sat_schema.EGO_UAV_QUEUE_FILL] = uav_queue[ego] / uav_queue_ref
        ego_features[ego, sat_schema.EGO_UAV_LAST_INFLOW_STEPS] = uav_last_inflow_steps[ego]
        ego_features[ego, sat_schema.EGO_UAV_LAST_OUTFLOW_STEPS] = uav_last_outflow_steps[ego]
        ego_features[ego, sat_schema.EGO_UAV_LAST_DROP_STEPS] = uav_last_drop_steps[ego]
        ego_features[ego, sat_schema.EGO_UAV_SERVICE_EMA_STEPS] = uav_service_ema_steps[ego]
        ego_features[ego, sat_schema.EGO_UAV_LOCAL_COST_LOG_RATIO] = uav_local_cost_log_ratio[ego]
        ego_features[ego, sat_schema.EGO_UAV_LAST_TOTAL_COST_LOG_RATIO] = uav_last_total_cost_log_ratio[ego]
        ego_features[ego, sat_schema.EGO_UAV_LAST_WORKLOAD_LOG1P] = uav_last_workload_log1p[ego]
        ego_features[ego, sat_schema.EGO_UAV_LAST_ACCESS_INTERFERENCE_LOG1P] = np.log1p(max(float(last_access_interf[ego]) / access_noise_ref, 0.0))
        ego_features[ego, sat_schema.EGO_LAST_SELECTED_COUNT_FRAC] = float(np.sum(last_selected_mask[ego])) / float(select_k)
        ego_features[ego, sat_schema.EGO_LAST_BACKHAUL_OUTFLOW_STEPS] = float(np.log1p(max(float(np.sum(last_uav_sat_outflow[ego]) / uav_flow_ref), 0.0)))
        ego_features[ego, sat_schema.EGO_REMAINING_HORIZON_FRAC] = remaining_horizon_frac
        role_features[ego, 0] = float(ego) / float(max(num_uav - 1, 1))

        in_cell = assoc == ego
        demand_features[ego, sat_schema.DEMAND_CELL_GU_COUNT_FRAC] = float(np.sum(in_cell)) / float(num_gu)
        if np.any(in_cell):
            demand_features[ego, sat_schema.DEMAND_CELL_QUEUE_STEPS_SUM] = float(np.log1p(max(float(np.sum(gu_queue_steps_raw[in_cell])) / float(num_gu), 0.0)))
            demand_features[ego, sat_schema.DEMAND_CELL_EXPECTED_ARRIVAL_STEPS_SUM] = float(np.log1p(max(float(np.sum(gu_expected_steps_raw[in_cell])) / float(num_gu), 0.0)))
            demand_features[ego, sat_schema.DEMAND_CELL_LAST_ARRIVAL_STEPS_SUM] = float(np.log1p(max(float(np.sum(gu_last_arrival_steps_raw[in_cell])) / float(num_gu), 0.0)))
            demand_features[ego, sat_schema.DEMAND_CELL_LAST_OUTFLOW_STEPS_SUM] = float(np.log1p(max(float(np.sum(gu_last_outflow_steps_raw[in_cell])) / float(num_gu), 0.0)))
            demand_features[ego, sat_schema.DEMAND_CELL_LAST_DROP_STEPS_SUM] = float(np.log1p(max(float(np.sum(gu_last_drop_steps_raw[in_cell])) / float(num_gu), 0.0)))
            demand_features[ego, sat_schema.DEMAND_CELL_LAST_WORKLOAD_LOG1P_SUM] = float(np.log1p(max(float(np.sum(gu_last_workload_raw[in_cell])) / float(num_gu), 0.0)))
            demand_features[ego, sat_schema.DEMAND_CELL_ACCESS_RATE_FULL_BW_REF_SUM] = float(np.log1p(max(float(np.sum(access_rate_full_bw_ref_raw[in_cell, ego])) / float(num_gu), 0.0)))

        sat_ids = [int(s) for s in visible[ego][:sat_width] if 0 <= int(s) < num_sat]
        if not sat_ids:
            continue
        ids = np.asarray(sat_ids, dtype=np.int64)
        n = int(ids.size)
        candidate_sat_ids[ego, :n] = ids
        sat_mask[ego, :n] = True
        sat_tokens[ego, :n, sat_schema.SAT_QUEUE_STEPS] = sat_queue_steps[ids]
        sat_tokens[ego, :n, sat_schema.SAT_QUEUE_FILL] = sat_queue[ids] / sat_queue_ref
        sat_tokens[ego, :n, sat_schema.SAT_LAST_INCOMING_STEPS] = sat_last_incoming_steps[ids]
        sat_tokens[ego, :n, sat_schema.SAT_LAST_PROCESSED_STEPS] = sat_last_processed_steps[ids]
        sat_tokens[ego, :n, sat_schema.SAT_LAST_DROP_STEPS] = sat_last_drop_steps[ids]
        sat_tokens[ego, :n, sat_schema.SAT_SERVICE_EMA_STEPS] = sat_service_ema_steps[ids]
        sat_tokens[ego, :n, sat_schema.SAT_COST_LOG_RATIO] = sat_cost_log_ratio[ids]
        sat_tokens[ego, :n, sat_schema.SAT_LAST_WORKLOAD_LOG1P] = sat_last_workload_log1p[ids]
        sat_tokens[ego, :n, sat_schema.SAT_LAST_SELECTED_LOAD_FRAC] = sat_last_selected_load_frac[ids]
        sat_tokens[ego, :n, sat_schema.SAT_PROC_CAPACITY_STEPS] = sat_proc_capacity_steps
        rel_pos = sat_pos[ids] - env._uav_ecef(ego)[None, :]
        rel_vel = sat_vel[ids] - env._uav_vel_ecef(ego)[None, :]
        range_m = np.linalg.norm(rel_pos, axis=1).astype(np.float32)
        if np.any(range_m <= 0.0):
            raise ValueError("UAV and SAT ECEF positions overlap; cannot build SAT token.")
        radial = np.sum(rel_pos * rel_vel, axis=1) / (range_m * sat_vel_ref)
        gain = (float(env._backhaul_gain_const) / (range_m * range_m)).astype(np.float32)
        if loss_matrix is not None:
            gain = gain * loss_matrix[ego, ids]
        snr = float(cfg.uav_tx_power) * gain / backhaul_noise_ref
        if bool(getattr(cfg, "doppler_enabled", False) or getattr(cfg, "doppler_atten_enabled", False) or getattr(cfg, "doppler_observed", False)):
            raw_nu = env._doppler_many(ego, ids, sat_pos, sat_vel)
            nu_eff, _ = env._effective_doppler_array(ego, ids, raw_nu)
            nu_max = _positive_float(float(cfg.nu_max), "nu_max")
            doppler_norm = nu_eff / doppler_ref
            doppler_margin = np.abs(nu_eff) / nu_max
        else:
            nu_eff = np.zeros((n,), dtype=np.float32)
            doppler_norm = np.zeros((n,), dtype=np.float32)
            doppler_margin = np.zeros((n,), dtype=np.float32)
        if bool(getattr(cfg, "doppler_atten_enabled", False)):
            snr = snr * channel.doppler_attenuation(nu_eff, float(cfg.subcarrier_spacing))
        backhaul_se = np.asarray(channel.spectral_efficiency(snr), dtype=np.float32)
        valid = elevation_matrix[ego, ids] >= float(cfg.theta_min_rad)
        if bool(getattr(cfg, "doppler_enabled", False)):
            valid = valid & (np.abs(nu_eff) <= float(cfg.nu_max))
        sat_valid_mask[ego, :n] = valid.astype(bool)
        sat_tokens[ego, :n, sat_schema.US_REL_X_NORM : sat_schema.US_REL_Z_NORM + 1] = rel_pos / orbit_pos_ref
        sat_tokens[ego, :n, sat_schema.US_REL_VX_NORM : sat_schema.US_REL_VZ_NORM + 1] = rel_vel / sat_vel_ref
        sat_tokens[ego, :n, sat_schema.US_RANGE_NORM] = range_m / orbit_pos_ref
        sat_tokens[ego, :n, sat_schema.US_RADIAL_VELOCITY_NORM] = radial
        sat_tokens[ego, :n, sat_schema.US_ELEVATION_NORM] = elevation_matrix[ego, ids] / (np.pi * 0.5)
        sat_tokens[ego, :n, sat_schema.US_DOPPLER_NORM] = doppler_norm
        sat_tokens[ego, :n, sat_schema.US_DOPPLER_MARGIN] = doppler_margin
        sat_tokens[ego, :n, sat_schema.US_BACKHAUL_SE_REF] = backhaul_se
        sat_tokens[ego, :n, sat_schema.US_VISIBLE_FLAG] = 1.0
        sat_tokens[ego, :n, sat_schema.US_VALID_FLAG] = valid.astype(np.float32)
        sat_tokens[ego, :n, sat_schema.US_LAST_SELECTED_FLAG] = last_selected_mask[ego, ids]
        sat_tokens[ego, :n, sat_schema.US_LAST_OUTFLOW_STEPS] = _log1p_nonnegative_np(last_uav_sat_outflow[ego, ids] / uav_flow_ref)

    subset_members_t, subset_sizes_t = _subset_member_tensor(sat_width, select_k, torch.device("cpu"))
    subset_members = subset_members_t.cpu().numpy().astype(np.int64, copy=False)
    subset_sizes = subset_sizes_t.cpu().numpy().astype(np.int64, copy=False)
    subset_mask = np.zeros((num_uav, subset_members.shape[0]), dtype=bool)
    valid_slots = sat_mask & sat_valid_mask
    for ego in range(num_uav):
        valid_count = int(np.sum(valid_slots[ego]))
        if valid_count == 0:
            subset_mask[ego, 0] = True
            continue
        for m, members in enumerate(subset_members):
            size = int(subset_sizes[m])
            if size == 0:
                continue
            if size > min(select_k, valid_count):
                continue
            slots = members[:size]
            if np.all((slots >= 0) & (slots < sat_width) & valid_slots[ego, slots]):
                subset_mask[ego, m] = True

    tensors = {
        "ego_features": torch.as_tensor(ego_features, dtype=torch.float32),
        "demand_features": torch.as_tensor(demand_features, dtype=torch.float32),
        "role_features": torch.as_tensor(role_features, dtype=torch.float32),
        "sat_tokens": torch.as_tensor(sat_tokens, dtype=torch.float32),
        "sat_mask": torch.as_tensor(sat_mask, dtype=torch.bool),
        "sat_valid_mask": torch.as_tensor(sat_valid_mask, dtype=torch.bool),
        "subset_members": torch.as_tensor(subset_members, dtype=torch.long),
        "subset_mask": torch.as_tensor(subset_mask, dtype=torch.bool),
        "candidate_sat_ids": torch.as_tensor(candidate_sat_ids, dtype=torch.long),
    }
    if device is not None:
        torch_device = torch.device(device)
        tensors = {name: tensor.to(device=torch_device) for name, tensor in tensors.items()}
    return LocalSatState(**tensors)


def build_local_sat_states_from_spec(spec: dict, *, device: torch.device | str | None = None) -> list[LocalSatState]:
    return _split_local_state_batch(build_batched_local_sat_states_from_spec(spec, device=device))


def build_batched_local_sat_states_from_world(
    ws: StructuredWorldState,
    max_select: int,
    *,
    max_visible: int | None = None,
) -> LocalSatState:
    del ws, max_select, max_visible
    raise RuntimeError(
        "build_batched_local_sat_states_from_world is retired for the redesigned SAT actor. "
        "Use StructuredControlDriver.build_local_sat_states() / build_batched_local_sat_states_from_spec()."
    )


def build_local_sat_states_from_world(ws: StructuredWorldState, max_select: int) -> list[LocalSatState]:
    return _split_local_state_batch(build_batched_local_sat_states_from_world(ws, max_select=max_select))


def build_batched_local_bw_states_from_spec(spec: dict, *, device: torch.device | str | None = None) -> LocalBwState:
    env = spec["env"]
    cfg = env.cfg
    num_uav = int(cfg.num_uav)
    num_gu = int(cfg.num_gu)
    num_sat = int(cfg.num_sat)
    if num_uav <= 0 or num_gu <= 0:
        raise ValueError("num_uav and num_gu must be positive for BW actor builder.")

    assoc = np.asarray(spec["assoc"], dtype=np.int32).reshape(num_gu)
    access_gain = np.asarray(env._coerce_access_gain_matrix(spec["access_gain_matrix"]), dtype=np.float32)
    if access_gain.shape != (num_gu, num_uav):
        raise ValueError(f"access_gain_matrix must have shape {(num_gu, num_uav)}, got {access_gain.shape}.")

    sat_select_k = _sat_action_select_k_from_cfg(cfg)
    selected_matrix = env._sat_selection_matrix(spec.get("sat_selection", np.full((num_uav, sat_select_k), -1, dtype=np.int64)))
    if selected_matrix.shape[1] != sat_select_k:
        fixed = np.full((num_uav, sat_select_k), -1, dtype=np.int64)
        width = min(sat_select_k, int(selected_matrix.shape[1]))
        fixed[:, :width] = selected_matrix[:, :width]
        selected_matrix = fixed
    sat_loads = np.asarray(spec.get("sat_loads", np.ones((num_sat,), dtype=np.float32)), dtype=np.float32).reshape(num_sat)
    sat_pos = np.asarray(spec["sat_pos"], dtype=np.float32).reshape(num_sat, 3)
    sat_vel = np.asarray(spec["sat_vel"], dtype=np.float32).reshape(num_sat, 3)

    arrival_ref = _positive_float(env._arrival_ref(), "arrival_ref_bits_per_step")
    gu_flow_ref = _positive_float(arrival_ref / float(num_gu), "gu_flow_ref")
    uav_flow_ref = _positive_float(arrival_ref / float(num_uav), "uav_flow_ref")
    sat_flow_ref = _positive_float(arrival_ref / _positive_float(env._bw_weighted_workload_sat_active_ref_count(), "sat_active_ref_count"), "sat_flow_ref")
    gu_queue_ref = _positive_float(float(cfg.queue_max_gu), "queue_max_gu")
    uav_queue_ref = _positive_float(float(cfg.queue_max_uav), "queue_max_uav")
    sat_queue_ref = _positive_float(float(cfg.queue_max_sat), "queue_max_sat")
    service_floor = _positive_float(env._bw_weighted_workload_eps(), "service_floor_bits_per_step")
    access_noise_ref = _positive_float(
        float(cfg.noise_density) * float(cfg.b_acc) * channel.noise_figure_linear(float(getattr(cfg, "access_noise_figure_db", 0.0) or 0.0)),
        "access_noise_ref",
    )
    gu_flow_cost_ref = 1.0 / gu_flow_ref
    uav_flow_cost_ref = 1.0 / uav_flow_ref
    sat_cost_ref = 1.0 / sat_flow_ref
    uav_total_cost_ref = uav_flow_cost_ref + sat_cost_ref
    gu_total_cost_ref = gu_flow_cost_ref + uav_total_cost_ref

    gu_ema, uav_ema, sat_ema = env._bw_weighted_workload_device_ema_vectors()
    gu_local_cost = (1.0 / np.maximum(gu_ema, service_floor)).astype(np.float32, copy=False)
    uav_local_cost = (1.0 / np.maximum(uav_ema, service_floor)).astype(np.float32, copy=False)
    sat_cost = (1.0 / np.maximum(sat_ema, service_floor)).astype(np.float32, copy=False)
    last_assoc = np.asarray(getattr(env, "last_association", np.full((num_gu,), -1, dtype=np.int32)), dtype=np.int32).reshape(num_gu)
    last_gu_total_cost, last_uav_total_cost, _last_sat_cost = env._bw_weighted_workload_device_costs(
        assoc_override=last_assoc,
        sat_selection_override=getattr(env, "last_sat_selection", None),
    )

    gu_queue = np.asarray(env.gu_queue, dtype=np.float32).reshape(num_gu)
    uav_queue = np.asarray(env.uav_queue, dtype=np.float32).reshape(num_uav)
    sat_queue = np.asarray(env.sat_queue, dtype=np.float32).reshape(num_sat)
    gu_expected = np.asarray(env._current_expected_gu_arrival_rates(), dtype=np.float32).reshape(num_gu) * float(cfg.tau0)
    gu_last_arrival = np.asarray(getattr(env, "last_gu_arrival", np.zeros((num_gu,), dtype=np.float32)), dtype=np.float32).reshape(num_gu)
    gu_last_outflow = np.asarray(getattr(env, "last_gu_outflow", np.zeros((num_gu,), dtype=np.float32)), dtype=np.float32).reshape(num_gu)
    gu_drop = np.asarray(getattr(env, "gu_drop", np.zeros((num_gu,), dtype=np.float32)), dtype=np.float32).reshape(num_gu)
    uav_last_inflow = np.asarray(getattr(env, "last_gu_to_uav_inflow_by_uav", np.zeros((num_uav,), dtype=np.float32)), dtype=np.float32).reshape(num_uav)
    uav_last_outflow = np.asarray(getattr(env, "last_uav_outflow", np.zeros((num_uav,), dtype=np.float32)), dtype=np.float32).reshape(num_uav)
    uav_drop = np.asarray(getattr(env, "uav_drop", np.zeros((num_uav,), dtype=np.float32)), dtype=np.float32).reshape(num_uav)
    sat_last_incoming = np.asarray(getattr(env, "last_sat_incoming", np.zeros((num_sat,), dtype=np.float32)), dtype=np.float32).reshape(num_sat)
    sat_last_processed = np.asarray(getattr(env, "last_sat_processed", np.zeros((num_sat,), dtype=np.float32)), dtype=np.float32).reshape(num_sat)
    sat_drop = np.asarray(getattr(env, "sat_drop", np.zeros((num_sat,), dtype=np.float32)), dtype=np.float32).reshape(num_sat)
    last_access_interf = np.asarray(getattr(env, "last_access_interference_by_uav", np.zeros((num_uav,), dtype=np.float32)), dtype=np.float32).reshape(num_uav)

    gu_local_cost_log_ratio = _log_ratio_np(gu_local_cost, gu_flow_cost_ref)
    gu_last_total_cost_log_ratio = _log_ratio_np(last_gu_total_cost, gu_total_cost_ref)
    uav_local_cost_log_ratio = _log_ratio_np(uav_local_cost, uav_flow_cost_ref)
    uav_last_total_cost_log_ratio = _log_ratio_np(last_uav_total_cost, uav_total_cost_ref)
    sat_cost_log_ratio = _log_ratio_np(sat_cost, sat_cost_ref)
    gu_last_workload_log1p = _log1p_nonnegative_np(last_gu_total_cost * gu_queue)
    uav_last_workload_log1p = _log1p_nonnegative_np(last_uav_total_cost * uav_queue)
    sat_last_workload_log1p = _log1p_nonnegative_np(sat_cost * sat_queue)
    access_se_full_ref = _access_se_from_gain(env, access_gain, access_noise_ref)
    access_rate_full_bw_ref = _log1p_nonnegative_np(float(cfg.b_acc) * access_se_full_ref * float(cfg.tau0) / gu_flow_ref)

    gu_queue_steps = _log1p_nonnegative_np(gu_queue / gu_flow_ref)
    gu_expected_steps = _log1p_nonnegative_np(gu_expected / gu_flow_ref)
    gu_last_arrival_steps = _log1p_nonnegative_np(gu_last_arrival / gu_flow_ref)
    gu_last_outflow_steps = _log1p_nonnegative_np(gu_last_outflow / gu_flow_ref)
    gu_last_drop_steps = _log1p_nonnegative_np(gu_drop / gu_flow_ref)
    gu_service_ema_steps = _log1p_nonnegative_np(gu_ema / gu_flow_ref)
    uav_queue_steps = _log1p_nonnegative_np(uav_queue / uav_flow_ref)
    uav_last_inflow_steps = _log1p_nonnegative_np(uav_last_inflow / uav_flow_ref)
    uav_last_outflow_steps = _log1p_nonnegative_np(uav_last_outflow / uav_flow_ref)
    uav_last_drop_steps = _log1p_nonnegative_np(uav_drop / uav_flow_ref)
    uav_service_ema_steps = _log1p_nonnegative_np(uav_ema / uav_flow_ref)
    sat_queue_steps = _log1p_nonnegative_np(sat_queue / sat_flow_ref)
    sat_last_incoming_steps = _log1p_nonnegative_np(sat_last_incoming / sat_flow_ref)
    sat_last_processed_steps = _log1p_nonnegative_np(sat_last_processed / sat_flow_ref)
    sat_last_drop_steps = _log1p_nonnegative_np(sat_drop / sat_flow_ref)
    sat_service_ema_steps = _log1p_nonnegative_np(sat_ema / sat_flow_ref)

    cross_log = np.zeros((num_uav, num_gu, 2), dtype=np.float32)
    if num_uav > 1:
        for ego in range(num_uav):
            other = [v for v in range(num_uav) if v != ego]
            unit = float(cfg.gu_tx_power) * access_gain[:, other]
            logs = np.log1p(np.maximum(unit / access_noise_ref, 0.0)).astype(np.float32)
            cross_log[ego, :, 0] = np.mean(logs, axis=1)
            cross_log[ego, :, 1] = np.max(logs, axis=1)

    row_count = num_uav
    ego_features = np.zeros((row_count, bw_schema.BW_EGO_DIM), dtype=np.float32)
    selected_sat_tokens = np.zeros((row_count, sat_select_k, bw_schema.BW_SAT_TOKEN_DIM), dtype=np.float32)
    selected_sat_mask = np.zeros((row_count, sat_select_k), dtype=bool)
    gu_tokens = np.zeros((row_count, num_gu, bw_schema.BW_GU_TOKEN_DIM), dtype=np.float32)
    gu_mask = np.ones((row_count, num_gu), dtype=bool)
    bw_valid_mask = np.zeros((row_count, num_gu), dtype=bool)

    backhaul_bw_per_sat = _positive_float(env._effective_b_backhaul_per_sat(), "effective_b_backhaul_per_sat")
    backhaul_nf = channel.noise_figure_linear(float(getattr(cfg, "backhaul_noise_figure_db", 0.0) or 0.0))
    elevation_matrix = env._get_elevation_matrix(sat_pos)
    loss_matrix = env._get_backhaul_loss_matrix(sat_pos)
    remaining_horizon_frac = _remaining_horizon_frac(env)
    for ego in range(num_uav):
        ego_features[ego, 0] = uav_queue_steps[ego]
        ego_features[ego, 1] = uav_queue[ego] / uav_queue_ref
        ego_features[ego, 2] = uav_last_inflow_steps[ego]
        ego_features[ego, 3] = uav_last_outflow_steps[ego]
        ego_features[ego, 4] = uav_last_drop_steps[ego]
        ego_features[ego, 5] = uav_service_ema_steps[ego]
        ego_features[ego, 6] = uav_local_cost_log_ratio[ego]
        ego_features[ego, 7] = uav_last_total_cost_log_ratio[ego]
        ego_features[ego, 8] = uav_last_workload_log1p[ego]
        ego_features[ego, 9] = np.log1p(max(float(last_access_interf[ego]) / access_noise_ref, 0.0))
        ego_features[ego, bw_schema.BW_EGO_REMAINING_HORIZON_FRAC] = remaining_horizon_frac

        bw_valid_mask[ego] = assoc == ego
        gu_tokens[ego, :, 0] = gu_queue_steps
        gu_tokens[ego, :, 1] = gu_queue / gu_queue_ref
        gu_tokens[ego, :, 2] = gu_expected_steps
        gu_tokens[ego, :, 3] = gu_last_arrival_steps
        gu_tokens[ego, :, 4] = gu_last_outflow_steps
        gu_tokens[ego, :, 5] = gu_last_drop_steps
        gu_tokens[ego, :, 6] = gu_service_ema_steps
        gu_tokens[ego, :, 7] = gu_local_cost_log_ratio
        gu_tokens[ego, :, 8] = gu_last_total_cost_log_ratio
        gu_tokens[ego, :, 9] = gu_last_workload_log1p
        gu_tokens[ego, :, 10] = access_rate_full_bw_ref[:, ego]
        gu_tokens[ego, :, 11] = cross_log[ego, :, 0]
        gu_tokens[ego, :, 12] = cross_log[ego, :, 1]
        gu_tokens[ego, ~bw_valid_mask[ego], :] = 0.0

        for slot, sat_raw in enumerate(selected_matrix[ego, :sat_select_k].tolist()):
            sat_idx = int(sat_raw)
            if sat_idx < 0 or sat_idx >= num_sat:
                continue
            valid = elevation_matrix[ego, sat_idx] >= float(cfg.theta_min_rad)
            if bool(getattr(cfg, "doppler_enabled", False)):
                raw_nu = env._doppler_many(ego, np.asarray([sat_idx], dtype=np.int64), sat_pos, sat_vel)
                nu_eff, _ = env._effective_doppler_array(ego, np.asarray([sat_idx], dtype=np.int64), raw_nu)
                valid = bool(valid) and abs(float(nu_eff[0])) <= float(cfg.nu_max)
            else:
                nu_eff = np.zeros((1,), dtype=np.float32)
            if not valid:
                continue
            selected_sat_mask[ego, slot] = True
            load = max(float(sat_loads[sat_idx]), 1.0)
            b_share = backhaul_bw_per_sat / load
            rel = sat_pos[sat_idx] - env._uav_ecef(ego)
            dist = float(np.linalg.norm(rel))
            gain = float(env._backhaul_gain_const) / float(geometry_denominator(dist * dist))
            if loss_matrix is not None:
                gain *= float(loss_matrix[ego, sat_idx])
            noise = float(cfg.noise_density) * b_share * backhaul_nf
            snr = float(cfg.uav_tx_power) * gain / max(noise, float(LOG_RATIO_EPS))
            if bool(getattr(cfg, "doppler_atten_enabled", False)):
                snr *= float(channel.doppler_attenuation(nu_eff, float(cfg.subcarrier_spacing))[0])
            se = float(channel.spectral_efficiency(np.asarray([snr], dtype=np.float32))[0])
            selected_sat_tokens[ego, slot, 0] = float(np.log1p(max(float(se * b_share * float(cfg.tau0) / uav_flow_ref), 0.0)))
            selected_sat_tokens[ego, slot, 1] = sat_queue_steps[sat_idx]
            selected_sat_tokens[ego, slot, 2] = sat_queue[sat_idx] / sat_queue_ref
            selected_sat_tokens[ego, slot, 3] = sat_last_incoming_steps[sat_idx]
            selected_sat_tokens[ego, slot, 4] = sat_last_processed_steps[sat_idx]
            selected_sat_tokens[ego, slot, 5] = sat_last_drop_steps[sat_idx]
            selected_sat_tokens[ego, slot, 6] = sat_service_ema_steps[sat_idx]
            selected_sat_tokens[ego, slot, 7] = sat_cost_log_ratio[sat_idx]
            selected_sat_tokens[ego, slot, 8] = sat_last_workload_log1p[sat_idx]

    tensors = {
        "ego_features": torch.as_tensor(ego_features, dtype=torch.float32),
        "selected_sat_tokens": torch.as_tensor(selected_sat_tokens, dtype=torch.float32),
        "selected_sat_mask": torch.as_tensor(selected_sat_mask, dtype=torch.bool),
        "gu_tokens": torch.as_tensor(gu_tokens, dtype=torch.float32),
        "gu_mask": torch.as_tensor(gu_mask, dtype=torch.bool),
        "bw_valid_mask": torch.as_tensor(bw_valid_mask, dtype=torch.bool),
    }
    if device is not None:
        torch_device = torch.device(device)
        tensors = {name: tensor.to(device=torch_device) for name, tensor in tensors.items()}
    return LocalBwState(**tensors)


def build_local_bw_states_from_spec(spec: dict, *, device: torch.device | str | None = None) -> list[LocalBwState]:
    return _split_local_state_batch(build_batched_local_bw_states_from_spec(spec, device=device))


def build_batched_local_bw_states_from_snapshot(
    ws: StructuredWorldState,
    candidate_indices: torch.Tensor,
    bw_valid_mask: torch.Tensor,
    *,
    max_visible: int | None = None,
) -> LocalBwState:
    del ws, candidate_indices, bw_valid_mask, max_visible
    raise RuntimeError(
        "BW actor local state is built from BW stage spec, not legacy candidate-slot snapshots. "
        "Use build_batched_local_bw_states_from_spec(...)."
    )


def _tensor_from_snapshot_field(
    snapshot: BwStageSnapshot,
    field_name: str,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = getattr(snapshot, field_name, None)
    if value is None:
        raise RuntimeError(
            "BwStageSnapshot does not carry redesigned LocalBwState tensors. "
            "Create it through StructuredControlDriver.build_bw_stage_snapshot()."
        )
    return torch.as_tensor(value, dtype=dtype)


def build_local_bw_states_from_snapshot(snapshot: BwStageSnapshot) -> list[LocalBwState]:
    local_state = LocalBwState(
        ego_features=_tensor_from_snapshot_field(snapshot, "ego_features", dtype=torch.float32),
        selected_sat_tokens=_tensor_from_snapshot_field(snapshot, "selected_sat_tokens", dtype=torch.float32),
        selected_sat_mask=_tensor_from_snapshot_field(snapshot, "selected_sat_mask", dtype=torch.bool),
        gu_tokens=_tensor_from_snapshot_field(snapshot, "gu_tokens", dtype=torch.float32),
        gu_mask=_tensor_from_snapshot_field(snapshot, "gu_mask", dtype=torch.bool),
        bw_valid_mask=_tensor_from_snapshot_field(snapshot, "bw_valid_mask", dtype=torch.bool),
    )
    return _split_local_state_batch(local_state)
