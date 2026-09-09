from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sagin_marl.env.config import SaginConfig
from sagin_marl.env.structured_batch_env_core import (
    _build_fast_bw_step_metrics_tensor_impl,
    _build_global_state_tensor_impl,
    _build_world_from_packed_specs_tensor_impl,
    _compute_access_link_gain_matrix_tensor_impl,
    _compute_uav_cache_arrays,
    _native_typed_domains_from_cfg,
    _visible_sats_batch_tensor_impl,
)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _clone_cfg(cfg: SaginConfig, *, operator_mode: str) -> SaginConfig:
    cloned = copy.deepcopy(cfg)
    cloned.structured_kernel_operator_mode = operator_mode
    return cloned


def _random_unit_vectors(rng: np.random.Generator, count: int) -> np.ndarray:
    vec = rng.normal(size=(count, 3))
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1.0e-9)
    return (vec / norm).astype(np.float32, copy=False)


def _make_synthetic_inputs(cfg: SaginConfig, *, batch_size: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    batch = int(batch_size)
    num_gu = int(cfg.num_gu)
    num_uav = int(cfg.num_uav)
    num_sat = int(cfg.num_sat)
    users_obs_max = int(cfg.users_obs_max)
    sat_num_select = int(cfg.sat_num_select or 1)

    gu_queue = rng.uniform(0.0, float(cfg.queue_max_gu), size=(batch, num_gu)).astype(np.float32)
    uav_queue = rng.uniform(0.0, float(cfg.queue_max_uav), size=(batch, num_uav)).astype(np.float32)
    sat_queue = rng.uniform(0.0, float(cfg.queue_max_sat), size=(batch, num_sat)).astype(np.float32)
    gu_pos = rng.uniform(0.0, float(cfg.map_size), size=(batch, num_gu, 2)).astype(np.float32)
    uav_pos = rng.uniform(0.0, float(cfg.map_size), size=(batch, num_uav, 2)).astype(np.float32)
    arrivals = rng.uniform(0.0, float(cfg.task_arrival_rate) * 1.5, size=(batch, num_gu)).astype(np.float32)
    access_rates = rng.uniform(0.0, float(cfg.b_acc), size=(batch, num_gu)).astype(np.float32)
    prev_service_gap = rng.uniform(0.0, 4.0, size=(batch, num_gu)).astype(np.float32)
    prev_deadline_age = rng.uniform(0.0, 2.0, size=(batch, num_gu)).astype(np.float32)
    deadline_steps = rng.uniform(2.0, 6.0, size=(batch, num_gu)).astype(np.float32)
    base_arrival_steps = rng.uniform(1.0e4, 3.0e5, size=(batch,)).astype(np.float32)

    associations = rng.integers(low=0, high=max(num_uav, 1), size=(batch, num_gu), dtype=np.int32)
    candidate_indices = rng.integers(low=0, high=max(num_gu, 1), size=(batch, num_uav, users_obs_max), dtype=np.int64)
    candidate_mask = (rng.random(size=(batch, num_uav, users_obs_max)) > 0.2)
    bw_action_matrix = rng.uniform(0.0, 1.0, size=(batch, num_uav, users_obs_max)).astype(np.float32)
    prev_association = rng.integers(low=-1, high=max(num_uav, 1), size=(batch, num_gu), dtype=np.int32)

    gain_matrix = rng.uniform(1.0e-10, 1.0e-6, size=(batch, num_gu, num_uav)).astype(np.float32)
    gu_outflow = rng.uniform(0.0, float(cfg.b_acc) * float(cfg.tau0), size=(batch, num_gu)).astype(np.float32)
    rate_matrix = rng.uniform(0.0, float(cfg.b_backhaul_per_sat), size=(batch, num_uav, num_sat)).astype(np.float32)
    sat_compute_rates = rng.uniform(1.0e5, 5.0e6, size=(batch,)).astype(np.float32)

    sat_orbit_radius = float(cfg.r_earth + cfg.sat_height)
    uav_orbit_radius = float(cfg.r_earth + cfg.uav_height)
    sat_pos = (_random_unit_vectors(rng, batch * num_sat).reshape(batch, num_sat, 3) * sat_orbit_radius).astype(np.float32)
    sat_vel = rng.normal(0.0, 3000.0, size=(batch, num_sat, 3)).astype(np.float32)
    uav_ecef = (_random_unit_vectors(rng, batch * num_uav).reshape(batch, num_uav, 3) * uav_orbit_radius).astype(np.float32)
    uav_vel_ecef = rng.normal(0.0, 50.0, size=(batch, num_uav, 3)).astype(np.float32)
    uav_vel_xy = rng.normal(0.0, min(float(cfg.v_max), 20.0), size=(batch, num_uav, 2)).astype(np.float32)
    uav_energy_before = rng.uniform(0.5, 1.0, size=(batch, num_uav)).astype(np.float32) * float(getattr(cfg, "uav_energy_init", 1.0) or 1.0)
    doppler_residual = rng.normal(0.0, 25.0, size=(batch, num_uav, num_sat)).astype(np.float32)
    loss_matrix = rng.uniform(0.85, 1.0, size=(batch, num_uav, num_sat)).astype(np.float32)

    sat_selection_matrix = rng.integers(low=0, high=max(num_sat, 1), size=(batch, num_uav, sat_num_select), dtype=np.int64)
    invalid_mask = rng.random(size=sat_selection_matrix.shape) < 0.15
    sat_selection_matrix[invalid_mask] = -1
    last_sat_connection_counts = rng.uniform(0.0, 3.0, size=(batch, num_sat)).astype(np.float32)
    next_arrival_rates = rng.uniform(0.0, float(cfg.task_arrival_rate) * 1.5, size=(batch, num_gu)).astype(np.float32)
    arrival_ref = rng.uniform(8.0e5, 1.2e6, size=(batch,)).astype(np.float32)
    gu_ema = rng.uniform(1.0e4, 3.0e5, size=(batch, num_gu)).astype(np.float32)
    uav_ema = rng.uniform(1.0e4, 3.0e5, size=(batch, num_uav)).astype(np.float32)
    sat_ema = rng.uniform(1.0e4, 3.0e5, size=(batch, num_sat)).astype(np.float32)
    urgency_risk = rng.uniform(0.0, 1.0, size=(batch, num_gu)).astype(np.float32)
    downstream_pressure = rng.uniform(0.0, 1.0, size=(batch, num_gu)).astype(np.float32)
    service_gap_risk = rng.uniform(0.0, 1.0, size=(batch, num_gu)).astype(np.float32)
    deadline_slack = rng.uniform(-2.0, 2.0, size=(batch, num_gu)).astype(np.float32)
    deadline_risk = rng.uniform(0.0, 1.0, size=(batch, num_gu)).astype(np.float32)
    gu_proxy_features = rng.uniform(-1.0, 1.0, size=(batch, num_gu, 6)).astype(np.float32)
    candidate_flag = rng.integers(0, 2, size=(batch, num_uav, num_gu), dtype=np.int32).astype(np.float32)
    bw_valid_flag = rng.integers(0, 2, size=(batch, num_uav, num_gu), dtype=np.int32).astype(np.float32)
    prev_assoc_flag = rng.integers(0, 2, size=(batch, num_uav, num_gu), dtype=np.int32).astype(np.float32)
    eta_ref_feature = rng.uniform(0.0, 1.0, size=(batch, num_uav, num_gu)).astype(np.float32)
    sat_cost_norm_active = rng.uniform(-1.0, 1.0, size=(batch, num_sat)).astype(np.float32)
    rel_pos_active = rng.normal(0.0, 1000.0, size=(batch, num_uav, num_sat, 3)).astype(np.float32)
    rel_vel_active = rng.normal(0.0, 100.0, size=(batch, num_uav, num_sat, 3)).astype(np.float32)
    gain_active = rng.uniform(1.0e-10, 1.0e-6, size=(batch, num_uav, num_sat)).astype(np.float32)
    nu_eff_active = rng.normal(0.0, 10.0, size=(batch, num_uav, num_sat)).astype(np.float32)
    visible_flag_active = rng.integers(0, 2, size=(batch, num_uav, num_sat), dtype=np.int32).astype(np.float32)
    valid_flag_active = rng.integers(0, 2, size=(batch, num_uav, num_sat), dtype=np.int32).astype(np.float32)
    current_sel_flag_active = np.zeros((batch, num_uav, num_sat), dtype=np.float32)
    for slot in range(sat_num_select):
        sat_idx = sat_selection_matrix[:, :, slot]
        valid = (sat_idx >= 0) & (sat_idx < num_sat)
        batch_coords, uav_coords = np.where(valid)
        current_sel_flag_active[batch_coords, uav_coords, sat_idx[valid]] = 1.0
    intervention_norm_uav = rng.uniform(0.0, 1.0, size=(batch, num_uav)).astype(np.float32)
    close_risk_uav = rng.uniform(0.0, 1.0, size=(batch, num_uav)).astype(np.float32)
    danger_imitation_mask = rng.integers(0, 2, size=(batch, num_uav), dtype=np.int32).astype(np.float32)
    intervention_norm = intervention_norm_uav.mean(axis=1).astype(np.float32)
    intervention_rate = (intervention_norm_uav > 0.05).mean(axis=1).astype(np.float32)
    intervention_norm_top1 = intervention_norm_uav.max(axis=1).astype(np.float32)
    close_risk = close_risk_uav.mean(axis=1).astype(np.float32)
    danger_imitation_active_rate = danger_imitation_mask.mean(axis=1).astype(np.float32)
    collision = (rng.random(size=(batch,)) > 0.8)
    t = np.linspace(0.0, float(max(cfg.T_steps - 1, 1)), num=batch, dtype=np.float32)

    return {
        "gu_queue": gu_queue,
        "uav_queue": uav_queue,
        "sat_queue": sat_queue,
        "gu_pos": gu_pos,
        "uav_pos": uav_pos,
        "arrivals": arrivals,
        "access_rates": access_rates,
        "prev_service_gap": prev_service_gap,
        "prev_deadline_age": prev_deadline_age,
        "deadline_steps": deadline_steps,
        "base_arrival_steps": base_arrival_steps,
        "associations": associations,
        "candidate_indices": candidate_indices,
        "candidate_mask": candidate_mask,
        "bw_action_matrix": bw_action_matrix,
        "prev_association": prev_association,
        "gain_matrix": gain_matrix,
        "gu_outflow": gu_outflow,
        "rate_matrix": rate_matrix,
        "sat_compute_rates": sat_compute_rates,
        "sat_pos": sat_pos,
        "sat_vel": sat_vel,
        "uav_ecef": uav_ecef,
        "uav_vel_ecef": uav_vel_ecef,
        "uav_vel_xy": uav_vel_xy,
        "uav_energy_before": uav_energy_before,
        "doppler_residual": doppler_residual,
        "loss_matrix": loss_matrix,
        "sat_selection_matrix": sat_selection_matrix,
        "last_sat_connection_counts": last_sat_connection_counts,
        "next_arrival_rates": next_arrival_rates,
        "arrival_ref": arrival_ref,
        "gu_ema": gu_ema,
        "uav_ema": uav_ema,
        "sat_ema": sat_ema,
        "urgency_risk": urgency_risk,
        "downstream_pressure": downstream_pressure,
        "service_gap_risk": service_gap_risk,
        "deadline_slack": deadline_slack,
        "deadline_risk": deadline_risk,
        "gu_proxy_features": gu_proxy_features,
        "candidate_flag": candidate_flag,
        "bw_valid_flag": bw_valid_flag,
        "prev_assoc_flag": prev_assoc_flag,
        "eta_ref_feature": eta_ref_feature,
        "sat_cost_norm_active": sat_cost_norm_active,
        "rel_pos_active": rel_pos_active,
        "rel_vel_active": rel_vel_active,
        "gain_active": gain_active,
        "nu_eff_active": nu_eff_active,
        "visible_flag_active": visible_flag_active,
        "valid_flag_active": valid_flag_active,
        "current_sel_flag_active": current_sel_flag_active,
        "intervention_norm_uav": intervention_norm_uav,
        "close_risk_uav": close_risk_uav,
        "danger_imitation_mask": danger_imitation_mask,
        "intervention_norm": intervention_norm,
        "intervention_rate": intervention_rate,
        "intervention_norm_top1": intervention_norm_top1,
        "close_risk": close_risk,
        "danger_imitation_active_rate": danger_imitation_active_rate,
        "collision": collision,
        "t": t,
        "uav_orbit_radius": uav_orbit_radius,
        "uav_orbit_radius_sq": uav_orbit_radius * uav_orbit_radius,
        "sat_orbit_radius_sq": sat_orbit_radius * sat_orbit_radius,
    }


def _make_kernel_runner(name: str, cfg: SaginConfig, device: torch.device, payload: dict[str, Any]) -> Callable[[], Any]:
    typed_domains = _native_typed_domains_from_cfg(cfg, num_envs=int(payload["uav_pos"].shape[0]))
    if name == "visible_sats":
        uav_ecef, _uav_vel_ecef = _compute_uav_cache_arrays(
            cfg,
            np.asarray(payload["uav_pos"], dtype=np.float32),
            np.asarray(payload["uav_vel_xy"], dtype=np.float32),
        )
        backhaul_gain_const = float(
            ((float(cfg.speed_of_light) / (4.0 * math.pi * float(cfg.carrier_freq))) ** 2)
            * float(cfg.uav_tx_gain)
            * float(cfg.sat_rx_gain)
        )
        effective_b_backhaul_per_sat = float(cfg.b_backhaul_per_sat) * max(float(getattr(cfg, "b_backhaul_per_sat_scale", 1.0) or 1.0), 0.0)
        return lambda: _visible_sats_batch_tensor_impl(
            sat_pos_t=torch.as_tensor(payload["sat_pos"], dtype=torch.float32, device=device),
            uav_ecef_t=torch.as_tensor(uav_ecef, dtype=torch.float32, device=device),
            sat_queue_t=torch.as_tensor(payload["sat_queue"], dtype=torch.float32, device=device),
            sat_load_t=torch.as_tensor(payload["last_sat_connection_counts"], dtype=torch.float32, device=device),
            current_sel_mask_t=torch.as_tensor(payload["current_sel_flag_active"] > 0.5, dtype=torch.bool, device=device),
            channel_params=typed_domains.channel,
            sat_geometry_params=typed_domains.sat_geometry,
            sat_orbit_radius_sq=payload["sat_orbit_radius_sq"],
            uav_orbit_radius_sq=payload["uav_orbit_radius_sq"],
            uav_orbit_radius=payload["uav_orbit_radius"],
            backhaul_gain_const=backhaul_gain_const,
            effective_b_backhaul_per_sat=effective_b_backhaul_per_sat,
        )
    if name == "access_gain":
        return lambda: _compute_access_link_gain_matrix_tensor_impl(
            gu_pos_t=torch.as_tensor(payload["gu_pos"], dtype=torch.float32, device=device),
            uav_pos_t=torch.as_tensor(payload["uav_pos"], dtype=torch.float32, device=device),
            channel_params=typed_domains.channel,
            candidate_params=typed_domains.candidate,
        )
    if name == "global_state":
        return lambda: _build_global_state_tensor_impl(
            global_state_params=typed_domains.global_state,
            sat_geometry_params=typed_domains.sat_geometry,
            bw_workload_static_params=typed_domains.bw_workload,
            local_obs_params=typed_domains.local_obs,
            uav_pos_t=torch.as_tensor(payload["uav_pos"], dtype=torch.float32, device=device),
            uav_vel_t=torch.as_tensor(payload["uav_vel_xy"], dtype=torch.float32, device=device),
            uav_queue_t=torch.as_tensor(payload["uav_queue"], dtype=torch.float32, device=device),
            uav_energy_t=torch.as_tensor(payload["uav_energy_before"], dtype=torch.float32, device=device),
            gu_pos_t=torch.as_tensor(payload["gu_pos"], dtype=torch.float32, device=device),
            gu_queue_t=torch.as_tensor(payload["gu_queue"], dtype=torch.float32, device=device),
            sat_pos_t=torch.as_tensor(payload["sat_pos"], dtype=torch.float32, device=device),
            sat_vel_t=torch.as_tensor(payload["sat_vel"], dtype=torch.float32, device=device),
            sat_queue_t=torch.as_tensor(payload["sat_queue"], dtype=torch.float32, device=device),
            t_t=torch.as_tensor(payload["t"], dtype=torch.float32, device=device),
            arrival_ref_t=torch.as_tensor(payload["arrival_ref"], dtype=torch.float32, device=device),
            gu_ema_t=torch.as_tensor(payload["gu_ema"], dtype=torch.float32, device=device),
            uav_ema_t=torch.as_tensor(payload["uav_ema"], dtype=torch.float32, device=device),
            sat_ema_t=torch.as_tensor(payload["sat_ema"], dtype=torch.float32, device=device),
            assoc_t=torch.as_tensor(payload["associations"], dtype=torch.long, device=device),
            sat_selection_matrix_t=torch.as_tensor(payload["sat_selection_matrix"], dtype=torch.long, device=device),
            arrival_rate_vec_t=torch.as_tensor(payload["next_arrival_rates"], dtype=torch.float32, device=device),
            recent_arrival_t=torch.as_tensor(payload["arrivals"], dtype=torch.float32, device=device),
            recent_service_t=torch.as_tensor(payload["gu_outflow"], dtype=torch.float32, device=device),
            urgency_risk_t=torch.as_tensor(payload["urgency_risk"], dtype=torch.float32, device=device),
            downstream_pressure_t=torch.as_tensor(payload["downstream_pressure"], dtype=torch.float32, device=device),
            service_gap_t=torch.as_tensor(payload["prev_service_gap"], dtype=torch.float32, device=device),
            service_gap_risk_t=torch.as_tensor(payload["service_gap_risk"], dtype=torch.float32, device=device),
            deadline_steps_t=torch.as_tensor(payload["deadline_steps"], dtype=torch.float32, device=device),
            deadline_slack_t=torch.as_tensor(payload["deadline_slack"], dtype=torch.float32, device=device),
            deadline_risk_t=torch.as_tensor(payload["deadline_risk"], dtype=torch.float32, device=device),
            sat_state_max=int(cfg.sat_state_max),
        )
    if name == "world":
        return lambda: _build_world_from_packed_specs_tensor_impl(
            cfg=cfg,
            stage_ids_t=torch.arange(payload["uav_pos"].shape[0], dtype=torch.long, device=device),
            effective_b_backhaul_per_sat_t=torch.full((payload["uav_pos"].shape[0],), float(cfg.b_backhaul_per_sat), dtype=torch.float32, device=device),
            uav_pos_t=torch.as_tensor(payload["uav_pos"], dtype=torch.float32, device=device),
            uav_vel_t=torch.as_tensor(payload["uav_vel_xy"], dtype=torch.float32, device=device),
            uav_energy_t=torch.as_tensor(payload["uav_energy_before"], dtype=torch.float32, device=device),
            uav_queue_t=torch.as_tensor(payload["uav_queue"], dtype=torch.float32, device=device),
            uav_assoc_uav_cost_t=torch.zeros((payload["uav_pos"].shape[0], cfg.num_uav), dtype=torch.float32, device=device),
            gu_pos_t=torch.as_tensor(payload["gu_pos"], dtype=torch.float32, device=device),
            gu_queue_t=torch.as_tensor(payload["gu_queue"], dtype=torch.float32, device=device),
            gu_proxy_features_t=torch.as_tensor(payload["gu_proxy_features"], dtype=torch.float32, device=device),
            assoc_t=torch.as_tensor(payload["associations"], dtype=torch.long, device=device),
            candidate_flag_t=torch.as_tensor(payload["candidate_flag"], dtype=torch.float32, device=device),
            bw_valid_flag_t=torch.as_tensor(payload["bw_valid_flag"], dtype=torch.float32, device=device),
            prev_assoc_flag_t=torch.as_tensor(payload["prev_assoc_flag"], dtype=torch.float32, device=device),
            eta_ref_feature_t=torch.as_tensor(payload["eta_ref_feature"], dtype=torch.float32, device=device),
            sat_pos_active_t=torch.as_tensor(payload["sat_pos"], dtype=torch.float32, device=device),
            sat_vel_active_t=torch.as_tensor(payload["sat_vel"], dtype=torch.float32, device=device),
            sat_queue_active_t=torch.as_tensor(payload["sat_queue"], dtype=torch.float32, device=device),
            sat_load_active_t=torch.as_tensor(payload["last_sat_connection_counts"], dtype=torch.float32, device=device),
            sat_cost_norm_active_t=torch.as_tensor(payload["sat_cost_norm_active"], dtype=torch.float32, device=device),
            sat_active_mask_t=torch.ones((payload["uav_pos"].shape[0], cfg.num_sat), dtype=torch.bool, device=device),
            rel_pos_active_t=torch.as_tensor(payload["rel_pos_active"], dtype=torch.float32, device=device),
            rel_vel_active_t=torch.as_tensor(payload["rel_vel_active"], dtype=torch.float32, device=device),
            gain_active_t=torch.as_tensor(payload["gain_active"], dtype=torch.float32, device=device),
            nu_eff_active_t=torch.as_tensor(payload["nu_eff_active"], dtype=torch.float32, device=device),
            visible_flag_active_t=torch.as_tensor(payload["visible_flag_active"], dtype=torch.float32, device=device),
            valid_flag_active_t=torch.as_tensor(payload["valid_flag_active"], dtype=torch.float32, device=device),
            current_sel_flag_active_t=torch.as_tensor(payload["current_sel_flag_active"], dtype=torch.float32, device=device),
        )
    if name == "bw_finalize":
        return lambda: _build_fast_bw_step_metrics_tensor_impl(
            cfg=cfg,
            reward_mode=str(cfg.reward_mode),
            gu_ema_prev_t=torch.as_tensor(payload["gu_ema"], dtype=torch.float32, device=device),
            uav_ema_prev_t=torch.as_tensor(payload["uav_ema"], dtype=torch.float32, device=device),
            sat_ema_prev_t=torch.as_tensor(payload["sat_ema"], dtype=torch.float32, device=device),
            gu_outflow_t=torch.as_tensor(payload["gu_outflow"], dtype=torch.float32, device=device),
            uav_outflow_t=torch.as_tensor(payload["gu_outflow"][:, : cfg.num_uav], dtype=torch.float32, device=device),
            sat_processed_t=torch.as_tensor(payload["sat_compute_rates"][:, None].repeat(cfg.num_sat, axis=1), dtype=torch.float32, device=device),
            assoc_t=torch.as_tensor(payload["associations"], dtype=torch.long, device=device),
            sat_selection_matrix_t=torch.as_tensor(payload["sat_selection_matrix"], dtype=torch.long, device=device),
            gu_queue_before_t=torch.as_tensor(payload["gu_queue"], dtype=torch.float32, device=device),
            uav_queue_before_t=torch.as_tensor(payload["uav_queue"], dtype=torch.float32, device=device),
            sat_queue_before_t=torch.as_tensor(payload["sat_queue"], dtype=torch.float32, device=device),
            arrivals_t=torch.as_tensor(payload["arrivals"], dtype=torch.float32, device=device),
            gu_queue_after_t=torch.as_tensor(payload["gu_queue"], dtype=torch.float32, device=device),
            uav_queue_after_t=torch.as_tensor(payload["uav_queue"], dtype=torch.float32, device=device),
            sat_queue_after_t=torch.as_tensor(payload["sat_queue"], dtype=torch.float32, device=device),
            gu_drop_t=torch.zeros_like(torch.as_tensor(payload["gu_queue"], dtype=torch.float32, device=device)),
            uav_drop_t=torch.zeros_like(torch.as_tensor(payload["uav_queue"], dtype=torch.float32, device=device)),
            sat_drop_t=torch.zeros_like(torch.as_tensor(payload["sat_queue"], dtype=torch.float32, device=device)),
            last_sat_incoming_t=torch.as_tensor(payload["rate_matrix"].sum(axis=1), dtype=torch.float32, device=device),
            arrival_ref_t=torch.as_tensor(payload["arrival_ref"], dtype=torch.float32, device=device),
            prev_queue_sum_gu_t=torch.as_tensor(payload["arrival_ref"] * 0.25, dtype=torch.float32, device=device),
            prev_queue_sum_uav_t=torch.as_tensor(payload["arrival_ref"] * 0.15, dtype=torch.float32, device=device),
            gu_urgency_risk_t=torch.as_tensor(payload["urgency_risk"], dtype=torch.float32, device=device),
            downstream_pressure_t=torch.as_tensor(payload["downstream_pressure"], dtype=torch.float32, device=device),
            service_gap_t=torch.as_tensor(payload["prev_service_gap"], dtype=torch.float32, device=device),
            service_gap_risk_t=torch.as_tensor(payload["service_gap_risk"], dtype=torch.float32, device=device),
            intervention_norm_uav_t=torch.as_tensor(payload["intervention_norm_uav"], dtype=torch.float32, device=device),
            close_risk_uav_t=torch.as_tensor(payload["close_risk_uav"], dtype=torch.float32, device=device),
            danger_imitation_mask_t=torch.as_tensor(payload["danger_imitation_mask"], dtype=torch.float32, device=device),
            intervention_norm_t=torch.as_tensor(payload["intervention_norm"], dtype=torch.float32, device=device),
            intervention_rate_t=torch.as_tensor(payload["intervention_rate"], dtype=torch.float32, device=device),
            intervention_norm_top1_t=torch.as_tensor(payload["intervention_norm_top1"], dtype=torch.float32, device=device),
            close_risk_t=torch.as_tensor(payload["close_risk"], dtype=torch.float32, device=device),
            danger_imitation_active_rate_t=torch.as_tensor(payload["danger_imitation_active_rate"], dtype=torch.float32, device=device),
            collision_t=torch.as_tensor(payload["collision"], dtype=torch.bool, device=device),
            t_t=torch.as_tensor(payload["t"], dtype=torch.float32, device=device),
            uav_energy_after_t=torch.as_tensor(payload["uav_energy_before"], dtype=torch.float32, device=device),
        )
    raise ValueError(f"Unsupported kernel {name!r}")


def _to_numpy_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: _to_numpy_tree(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_numpy_tree(item) for item in value]
    return np.asarray(value)


def _assert_close(a: Any, b: Any, *, atol: float = 1.0e-5, rtol: float = 1.0e-5) -> None:
    if isinstance(a, dict):
        for key in a:
            _assert_close(a[key], b[key], atol=atol, rtol=rtol)
        return
    if isinstance(a, list):
        for left, right in zip(a, b):
            _assert_close(left, right, atol=atol, rtol=rtol)
        return
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol)


def _benchmark_callable(fn: Callable[[], Any], *, device: torch.device, warmup: int, iters: int) -> tuple[float, Any]:
    result = None
    for _ in range(max(int(warmup), 0)):
        result = fn()
    _sync_device(device)
    start = time.perf_counter()
    for _ in range(max(int(iters), 1)):
        result = fn()
    _sync_device(device)
    elapsed = time.perf_counter() - start
    return elapsed / max(int(iters), 1), result


def _profile_callable(fn: Callable[[], Any], *, device: torch.device, top_k: int) -> list[str]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
        result = fn()
        if torch.is_tensor(result):
            result.sum().item()
        _sync_device(device)
    table = profiler.key_averages().table(sort_by="self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total", row_limit=max(int(top_k), 1))
    return [line for line in table.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark structured native torch kernels in eager vs compile mode.")
    parser.add_argument("--kernel", choices=["all", "visible_sats", "global_state", "world", "bw_finalize", "access_gain"], default="all")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260417)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-top-k", type=int, default=12)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = _resolve_device(str(args.device))
    cfg = SaginConfig(
        num_uav=6,
        num_gu=24,
        num_sat=12,
        users_obs_max=8,
        sat_num_select=3,
    )
    cfg.enable_bw_action = True
    cfg.interference_enabled = True
    cfg.deadline_enabled = True
    cfg.energy_enabled = True
    cfg.doppler_enabled = True
    cfg.doppler_atten_enabled = True
    cfg.sat_candidate_mode = "score"
    cfg.sat_state_max = 6
    cfg.reward_mode = "controllable_flow"
    cfg.obs_user_include_arrival_rate = True
    cfg.obs_user_include_recent_arrival = True
    cfg.obs_user_include_recent_service = True
    cfg.obs_user_include_queue_headroom = True
    cfg.obs_user_include_local_gu_service_cost = True
    cfg.obs_user_include_assoc_uav_cost = True
    cfg.obs_user_include_assoc_sat_cost_mean = True
    cfg.obs_user_include_weighted_queue_cost = True
    cfg.obs_user_include_weighted_queue_cost_relative = True
    cfg.obs_user_include_urgency_risk = True
    cfg.obs_user_include_downstream_pressure = True
    cfg.obs_user_include_service_gap = True
    cfg.obs_user_include_service_gap_risk = True
    cfg.obs_user_include_deadline_slack = True
    cfg.obs_user_include_deadline_risk = True
    cfg.obs_own_include_assoc_uav_cost = True
    cfg.obs_own_include_uav_id_norm = True
    cfg.obs_sat_include_sat_cost = True
    cfg.close_risk_enabled = True
    cfg.danger_imitation_enabled = True
    cfg.structured_env_tensor_backend = device.type

    payload = _make_synthetic_inputs(cfg, batch_size=int(args.batch_size), seed=int(args.seed))
    kernels = ["visible_sats", "global_state", "world", "bw_finalize", "access_gain"] if args.kernel == "all" else [str(args.kernel)]
    summary: dict[str, Any] = {
        "device": str(device),
        "batch_size": int(args.batch_size),
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "kernels": {},
    }

    eager_cfg = _clone_cfg(cfg, operator_mode="eager")
    compile_cfg = _clone_cfg(cfg, operator_mode="compile")

    for kernel_name in kernels:
        eager_runner = _make_kernel_runner(kernel_name, eager_cfg, device, payload)
        compiled_runner = _make_kernel_runner(kernel_name, compile_cfg, device, payload)
        eager_ms, eager_out = _benchmark_callable(
            eager_runner,
            device=device,
            warmup=int(args.warmup),
            iters=int(args.iters),
        )
        compiled_ms, compiled_out = _benchmark_callable(
            compiled_runner,
            device=device,
            warmup=max(int(args.warmup), 4),
            iters=int(args.iters),
        )
        eager_np = _to_numpy_tree(eager_out)
        compiled_np = _to_numpy_tree(compiled_out)
        _assert_close(eager_np, compiled_np)
        kernel_summary: dict[str, Any] = {
            "eager_ms": float(eager_ms * 1000.0),
            "compiled_ms": float(compiled_ms * 1000.0),
            "speedup": float(eager_ms / max(compiled_ms, 1.0e-12)),
        }
        if args.profile:
            kernel_summary["profile_top"] = _profile_callable(
                compiled_runner,
                device=device,
                top_k=int(args.profile_top_k),
            )
        summary["kernels"][kernel_name] = kernel_summary
        print(
            f"{kernel_name:>12s} | eager={kernel_summary['eager_ms']:.3f} ms | "
            f"compile={kernel_summary['compiled_ms']:.3f} ms | speedup={kernel_summary['speedup']:.3f}x"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
