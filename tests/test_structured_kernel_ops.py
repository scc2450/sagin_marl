from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from sagin_marl.env.config import SaginConfig
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_batch_env_core import (
    _apply_batched_access_rate_static_tensor_impl,
    _apply_batched_close_risk_and_danger_tensor_impl,
    _apply_batched_bw_queue_transition_tensor_impl,
    _build_fast_bw_step_metrics_tensor_impl,
    _build_global_state_tensor_impl,
    _build_world_from_packed_specs_tensor_impl,
    _compute_access_link_gain_matrix_tensor_impl,
    _native_access_rate_static_params_from_cfg,
    _native_candidate_static_params_from_cfg,
    _native_channel_static_params_from_cfg,
    _native_typed_domains_from_cfg,
    _resolve_kernel_callable,
    _visible_sats_batch_tensor_impl,
)
from sagin_marl.env.structured_kernel_runtime import get_structured_kernel_runtime


def _apply_batched_access_rate_torch(
    *,
    gain_matrix,
    associations,
    candidate_indices,
    candidate_mask,
    bw_action_matrix,
    gu_queue_before,
    prev_association,
    cfg,
    device,
    compile_cfg=None,
    return_tensors: bool = False,
):
    del compile_cfg
    kernel_device = torch.device("cpu") if device is None else torch.device(device)
    rates_t, exec_bw_t, bw_align_t, eta_slots_t = _apply_batched_access_rate_static_tensor_impl(
        params=_native_access_rate_static_params_from_cfg(cfg),
        gain_matrix_t=torch.as_tensor(gain_matrix, dtype=torch.float32, device=kernel_device),
        assoc_t=torch.as_tensor(associations, dtype=torch.long, device=kernel_device),
        candidate_indices_t=torch.as_tensor(candidate_indices, dtype=torch.long, device=kernel_device),
        candidate_mask_t=torch.as_tensor(candidate_mask, dtype=torch.bool, device=kernel_device),
        bw_action_matrix_t=torch.as_tensor(bw_action_matrix, dtype=torch.float32, device=kernel_device),
        gu_queue_before_t=torch.as_tensor(gu_queue_before, dtype=torch.float32, device=kernel_device),
        prev_association_t=torch.as_tensor(prev_association, dtype=torch.long, device=kernel_device),
        candidate_uav_ids_t=torch.arange(int(cfg.num_uav), dtype=torch.long, device=kernel_device).view(1, int(cfg.num_uav), 1),
    )
    result_t = {
        "rates": rates_t,
        "last_exec_bw_alloc": exec_bw_t,
        "last_bw_align": bw_align_t,
        "eta_slots": eta_slots_t,
    }
    if return_tensors:
        return result_t
    return {key: value.detach().cpu().numpy() for key, value in result_t.items()}


def _apply_batched_bw_queue_transition_torch(
    *,
    uav_queue_before,
    sat_queue_before,
    gu_outflow,
    associations,
    rate_matrix,
    tau0,
    queue_max_uav,
    queue_max_sat,
    sat_compute_rates,
    device,
    compile_cfg=None,
    return_tensors: bool = False,
):
    kernel_device = torch.device("cpu") if device is None else torch.device(device)
    result = _apply_batched_bw_queue_transition_tensor_impl(
        uav_queue_before_t=torch.as_tensor(uav_queue_before, dtype=torch.float32, device=kernel_device),
        sat_queue_before_t=torch.as_tensor(sat_queue_before, dtype=torch.float32, device=kernel_device),
        gu_outflow_t=torch.as_tensor(gu_outflow, dtype=torch.float32, device=kernel_device),
        assoc_t=torch.as_tensor(associations, dtype=torch.long, device=kernel_device),
        rate_matrix_t=torch.as_tensor(rate_matrix, dtype=torch.float32, device=kernel_device),
        tau0=tau0,
        queue_max_uav=queue_max_uav,
        queue_max_sat=queue_max_sat,
        sat_compute_rates_t=torch.as_tensor(sat_compute_rates, dtype=torch.float32, device=kernel_device),
        fast_float32=False if compile_cfg is None else bool(getattr(compile_cfg, "structured_env_tensor_float32", False)),
    )
    keys = (
        "uav_queue",
        "uav_drop",
        "last_uav_outflow",
        "sat_queue",
        "sat_drop",
        "last_sat_incoming",
        "last_sat_processed",
    )
    result_t = {key: value for key, value in zip(keys, result)}
    if return_tensors:
        return result_t
    return {key: value.detach().cpu().numpy() for key, value in result_t.items()}


def _apply_batched_close_risk_and_danger_torch(
    *,
    uav_pos,
    uav_vel,
    last_exec_accel,
    last_policy_accel,
    cfg,
    device,
    compile_cfg=None,
    return_tensors: bool = False,
):
    del compile_cfg
    kernel_device = torch.device("cpu") if device is None else torch.device(device)
    num_uav = int(cfg.num_uav)
    result_t = _apply_batched_close_risk_and_danger_tensor_impl(
        pos_t=torch.as_tensor(uav_pos, dtype=torch.float32, device=kernel_device),
        vel_t=torch.as_tensor(uav_vel, dtype=torch.float32, device=kernel_device),
        exec_accel_t=torch.as_tensor(last_exec_accel, dtype=torch.float32, device=kernel_device),
        policy_accel_t=torch.as_tensor(last_policy_accel, dtype=torch.float32, device=kernel_device),
        params=_native_typed_domains_from_cfg(cfg, num_envs=int(np.asarray(uav_pos).shape[0])).post_stats_safety,
        upper_pair_mask_t=torch.triu(
            torch.ones((num_uav, num_uav), dtype=torch.bool, device=kernel_device),
            diagonal=1,
        ).view(1, num_uav, num_uav),
    )
    result_t = {field_name: getattr(result_t, field_name) for field_name in result_t._fields}
    if return_tensors:
        return result_t
    return {key: value.detach().cpu().numpy() for key, value in result_t.items()}


def _access_kernel_inputs():
    gain_matrix = np.array(
        [
            [
                [2.0e-7, 0.5e-7],
                [1.8e-7, 0.8e-7],
                [0.4e-7, 2.2e-7],
                [0.3e-7, 2.1e-7],
            ]
        ],
        dtype=np.float32,
    )
    associations = np.array([[0, 0, 1, 1]], dtype=np.int32)
    candidate_indices = np.array([[[0, 1, -1], [2, 3, -1]]], dtype=np.int64)
    candidate_mask = candidate_indices >= 0
    candidate_indices = np.clip(candidate_indices, a_min=0, a_max=None)
    bw_action_matrix = np.array([[[0.7, 0.3, 0.0, 0.0], [0.0, 0.0, 0.4, 0.6]]], dtype=np.float32)
    gu_queue_before = np.array([[1.0e6, 2.0e6, 3.0e6, 1.5e6]], dtype=np.float32)
    prev_association = np.array([[0, 0, 1, 1]], dtype=np.int32)
    return {
        "gain_matrix": gain_matrix,
        "associations": associations,
        "candidate_indices": candidate_indices,
        "candidate_mask": candidate_mask,
        "bw_action_matrix": bw_action_matrix,
        "gu_queue_before": gu_queue_before,
        "prev_association": prev_association,
    }


def _bw_queue_inputs():
    return {
        "uav_queue_before": np.array([[2.5e6, 1.5e6]], dtype=np.float32),
        "sat_queue_before": np.array([[4.0e6, 2.0e6, 1.0e6]], dtype=np.float32),
        "gu_outflow": np.array([[1.0e5, 2.0e5, 1.5e5, 0.5e5]], dtype=np.float32),
        "associations": np.array([[0, 0, 1, 1]], dtype=np.int32),
        "rate_matrix": np.array(
            [
                [
                    [2.5e5, 1.0e5, 0.0],
                    [0.0, 3.0e5, 2.0e5],
                ]
            ],
            dtype=np.float32,
        ),
        "sat_compute_rates": np.array([3.0e5], dtype=np.float32),
    }


def _access_gain_inputs():
    cfg = SaginConfig(num_uav=2, num_gu=4)
    cfg.fading_enabled = False
    envs = [SaginParallelEnv(cfg)]
    gu_pos = np.array(
        [[[100.0, 100.0], [150.0, 110.0], [200.0, 220.0], [90.0, 260.0]]],
        dtype=np.float32,
    )
    uav_pos = np.array(
        [[[110.0, 120.0], [230.0, 210.0]]],
        dtype=np.float32,
    )
    return cfg, envs, gu_pos, uav_pos


def _close_risk_inputs():
    return {
        "uav_pos": np.array(
            [[[100.0, 100.0], [170.0, 100.0], [400.0, 400.0]]],
            dtype=np.float32,
        ),
        "uav_vel": np.array(
            [[[10.0, 0.0], [-8.0, 0.0], [0.0, 0.0]]],
            dtype=np.float32,
        ),
        "last_exec_accel": np.array(
            [[[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]],
            dtype=np.float32,
        ),
        "last_policy_accel": np.zeros((1, 3, 2), dtype=np.float32),
    }


def _structured_hotpath_cfg():
    cfg = SaginConfig(num_uav=2, num_gu=4, num_sat=3, users_obs_max=3, sat_num_select=2)
    cfg.fading_enabled = False
    cfg.sat_candidate_mode = "score"
    cfg.sat_state_max = 2
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
    return cfg


def _selection_presence(matrix: np.ndarray, *, num_sat: int) -> np.ndarray:
    matrix_arr = np.asarray(matrix, dtype=np.int64)
    batch, num_uav, select_k = matrix_arr.shape
    mask = np.zeros((batch, num_uav, int(num_sat)), dtype=bool)
    for slot in range(select_k):
        sat_idx = matrix_arr[:, :, slot]
        valid = (sat_idx >= 0) & (sat_idx < int(num_sat))
        batch_coords, uav_coords = np.where(valid)
        mask[batch_coords, uav_coords, sat_idx[valid]] = True
    return mask


def _mixed_hotpath_inputs():
    cfg = _structured_hotpath_cfg()
    rng = np.random.default_rng(20260419)
    batch = 2
    num_uav = int(cfg.num_uav)
    num_gu = int(cfg.num_gu)
    num_sat = int(cfg.num_sat)
    envs = [SaginParallelEnv(cfg) for _ in range(batch)]
    uav_pos = rng.uniform(100.0, 400.0, size=(batch, num_uav, 2)).astype(np.float32)
    uav_vel = rng.uniform(-15.0, 15.0, size=(batch, num_uav, 2)).astype(np.float32)
    sat_pos_batch = np.zeros((batch, num_sat, 3), dtype=np.float32)
    uav_ecef_batch = np.zeros((batch, num_uav, 3), dtype=np.float32)
    for env_index, env in enumerate(envs):
        env.uav_pos = np.asarray(uav_pos[env_index], dtype=np.float32)
        env.uav_vel = np.asarray(uav_vel[env_index], dtype=np.float32)
        env._refresh_uav_cache()
        sat_pos, _sat_vel = env._get_orbit_states()
        sat_pos_batch[env_index] = np.asarray(sat_pos, dtype=np.float32)
        uav_ecef_batch[env_index] = np.asarray(env._cached_uav_ecef, dtype=np.float32)
    sat_vel = rng.normal(0.0, 1500.0, size=(batch, num_sat, 3)).astype(np.float32)
    sat_queue = rng.uniform(0.0, float(cfg.queue_max_sat) * 0.7, size=(batch, num_sat)).astype(np.float32)
    sat_load = rng.uniform(1.0, 3.0, size=(batch, num_sat)).astype(np.float32)
    sat_selection_matrix = np.array(
        [
            [[0, 1], [1, 2]],
            [[1, 2], [0, -1]],
        ],
        dtype=np.int64,
    )
    current_sel_mask = _selection_presence(sat_selection_matrix, num_sat=num_sat)
    gu_pos = rng.uniform(0.0, float(cfg.map_size), size=(batch, num_gu, 2)).astype(np.float32)
    gu_queue = rng.uniform(0.0, float(cfg.queue_max_gu) * 0.7, size=(batch, num_gu)).astype(np.float32)
    uav_queue = rng.uniform(0.0, float(cfg.queue_max_uav) * 0.6, size=(batch, num_uav)).astype(np.float32)
    uav_energy = rng.uniform(0.5, 1.0, size=(batch, num_uav)).astype(np.float32) * float(cfg.uav_energy_init)
    arrival_ref = rng.uniform(8.0e5, 1.2e6, size=(batch,)).astype(np.float32)
    gu_ema = rng.uniform(1.0e4, 3.0e5, size=(batch, num_gu)).astype(np.float32)
    uav_ema = rng.uniform(1.0e4, 3.0e5, size=(batch, num_uav)).astype(np.float32)
    sat_ema = rng.uniform(1.0e4, 3.0e5, size=(batch, num_sat)).astype(np.float32)
    assoc = np.array([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=np.int32)
    arrival_rate_vec = rng.uniform(0.0, float(cfg.task_arrival_rate) * 1.5, size=(batch, num_gu)).astype(np.float32)
    recent_arrival = rng.uniform(0.0, 3.0e5, size=(batch, num_gu)).astype(np.float32)
    recent_service = rng.uniform(0.0, 3.0e5, size=(batch, num_gu)).astype(np.float32)
    urgency_risk = rng.uniform(0.0, 1.0, size=(batch, num_gu)).astype(np.float32)
    downstream_pressure = rng.uniform(0.0, 1.0, size=(batch, num_gu)).astype(np.float32)
    service_gap = rng.uniform(0.0, 3.0, size=(batch, num_gu)).astype(np.float32)
    service_gap_risk = rng.uniform(0.0, 1.0, size=(batch, num_gu)).astype(np.float32)
    deadline_steps = rng.uniform(2.0, 6.0, size=(batch, num_gu)).astype(np.float32)
    deadline_slack = rng.uniform(-2.0, 2.0, size=(batch, num_gu)).astype(np.float32)
    deadline_risk = rng.uniform(0.0, 1.0, size=(batch, num_gu)).astype(np.float32)
    gu_feature_dim = 6
    gu_proxy_features = rng.uniform(-1.0, 1.0, size=(batch, num_gu, gu_feature_dim)).astype(np.float32)
    candidate_flag = rng.integers(0, 2, size=(batch, num_uav, num_gu), dtype=np.int32).astype(np.float32)
    bw_valid_flag = rng.integers(0, 2, size=(batch, num_uav, num_gu), dtype=np.int32).astype(np.float32)
    prev_assoc_flag = rng.integers(0, 2, size=(batch, num_uav, num_gu), dtype=np.int32).astype(np.float32)
    eta_ref_feature = rng.uniform(0.0, 1.0, size=(batch, num_uav, num_gu)).astype(np.float32)
    active_sat_mask = np.ones((batch, num_sat), dtype=bool)
    sat_cost_norm_active = rng.uniform(-1.0, 1.0, size=(batch, num_sat)).astype(np.float32)
    rel_pos_active = rng.normal(0.0, 1000.0, size=(batch, num_uav, num_sat, 3)).astype(np.float32)
    rel_vel_active = rng.normal(0.0, 100.0, size=(batch, num_uav, num_sat, 3)).astype(np.float32)
    gain_active = rng.uniform(1.0e-10, 1.0e-6, size=(batch, num_uav, num_sat)).astype(np.float32)
    nu_eff_active = rng.normal(0.0, 10.0, size=(batch, num_uav, num_sat)).astype(np.float32)
    visible_flag_active = rng.integers(0, 2, size=(batch, num_uav, num_sat), dtype=np.int32).astype(np.float32)
    valid_flag_active = rng.integers(0, 2, size=(batch, num_uav, num_sat), dtype=np.int32).astype(np.float32)
    current_sel_flag_active = _selection_presence(sat_selection_matrix, num_sat=num_sat).astype(np.float32)
    gu_queue_before = rng.uniform(0.0, 3.0e6, size=(batch, num_gu)).astype(np.float32)
    uav_queue_before = rng.uniform(0.0, 2.5e6, size=(batch, num_uav)).astype(np.float32)
    sat_queue_before = rng.uniform(0.0, 4.0e6, size=(batch, num_sat)).astype(np.float32)
    arrivals = rng.uniform(0.0, 3.0e5, size=(batch, num_gu)).astype(np.float32)
    gu_queue_after = rng.uniform(0.0, 3.0e6, size=(batch, num_gu)).astype(np.float32)
    uav_queue_after = rng.uniform(0.0, 2.5e6, size=(batch, num_uav)).astype(np.float32)
    sat_queue_after = rng.uniform(0.0, 4.0e6, size=(batch, num_sat)).astype(np.float32)
    gu_drop = rng.uniform(0.0, 2.0e5, size=(batch, num_gu)).astype(np.float32)
    uav_drop = rng.uniform(0.0, 1.0e5, size=(batch, num_uav)).astype(np.float32)
    sat_drop = rng.uniform(0.0, 1.0e5, size=(batch, num_sat)).astype(np.float32)
    gu_outflow = rng.uniform(0.0, 3.0e5, size=(batch, num_gu)).astype(np.float32)
    uav_outflow = rng.uniform(0.0, 3.0e5, size=(batch, num_uav)).astype(np.float32)
    sat_processed = rng.uniform(0.0, 3.0e5, size=(batch, num_sat)).astype(np.float32)
    last_sat_incoming = rng.uniform(0.0, 3.0e5, size=(batch, num_sat)).astype(np.float32)
    prev_queue_sum_gu = rng.uniform(0.0, 1.0e6, size=(batch,)).astype(np.float32)
    prev_queue_sum_uav = rng.uniform(0.0, 1.0e6, size=(batch,)).astype(np.float32)
    intervention_norm_uav = rng.uniform(0.0, 1.0, size=(batch, num_uav)).astype(np.float32)
    close_risk_uav = rng.uniform(0.0, 1.0, size=(batch, num_uav)).astype(np.float32)
    danger_imitation_mask = rng.integers(0, 2, size=(batch, num_uav), dtype=np.int32).astype(np.float32)
    intervention_norm = intervention_norm_uav.mean(axis=1).astype(np.float32)
    intervention_rate = (intervention_norm_uav > 0.05).mean(axis=1).astype(np.float32)
    intervention_norm_top1 = intervention_norm_uav.max(axis=1).astype(np.float32)
    close_risk = close_risk_uav.mean(axis=1).astype(np.float32)
    danger_imitation_active_rate = danger_imitation_mask.mean(axis=1).astype(np.float32)
    collision = np.array([False, True], dtype=bool)
    t = np.array([0.0, float(cfg.T_steps - 1)], dtype=np.float32)
    return {
        "cfg": cfg,
        "envs": envs,
        "sat_pos_batch": sat_pos_batch,
        "uav_ecef_batch": uav_ecef_batch,
        "sat_queue": sat_queue,
        "sat_load": sat_load,
        "current_sel_mask": current_sel_mask,
        "uav_pos": uav_pos,
        "uav_vel": uav_vel,
        "uav_energy": uav_energy,
        "uav_queue": uav_queue,
        "gu_pos": gu_pos,
        "gu_queue": gu_queue,
        "sat_vel": sat_vel,
        "arrival_ref": arrival_ref,
        "gu_ema": gu_ema,
        "uav_ema": uav_ema,
        "sat_ema": sat_ema,
        "assoc": assoc,
        "sat_selection_matrix": sat_selection_matrix,
        "arrival_rate_vec": arrival_rate_vec,
        "recent_arrival": recent_arrival,
        "recent_service": recent_service,
        "urgency_risk": urgency_risk,
        "downstream_pressure": downstream_pressure,
        "service_gap": service_gap,
        "service_gap_risk": service_gap_risk,
        "deadline_steps": deadline_steps,
        "deadline_slack": deadline_slack,
        "deadline_risk": deadline_risk,
        "gu_proxy_features": gu_proxy_features,
        "candidate_flag": candidate_flag,
        "bw_valid_flag": bw_valid_flag,
        "prev_assoc_flag": prev_assoc_flag,
        "eta_ref_feature": eta_ref_feature,
        "active_sat_mask": active_sat_mask,
        "sat_cost_norm_active": sat_cost_norm_active,
        "rel_pos_active": rel_pos_active,
        "rel_vel_active": rel_vel_active,
        "gain_active": gain_active,
        "nu_eff_active": nu_eff_active,
        "visible_flag_active": visible_flag_active,
        "valid_flag_active": valid_flag_active,
        "current_sel_flag_active": current_sel_flag_active,
        "gu_queue_before": gu_queue_before,
        "uav_queue_before": uav_queue_before,
        "sat_queue_before": sat_queue_before,
        "arrivals": arrivals,
        "gu_queue_after": gu_queue_after,
        "uav_queue_after": uav_queue_after,
        "sat_queue_after": sat_queue_after,
        "gu_drop": gu_drop,
        "uav_drop": uav_drop,
        "sat_drop": sat_drop,
        "gu_outflow": gu_outflow,
        "uav_outflow": uav_outflow,
        "sat_processed": sat_processed,
        "last_sat_incoming": last_sat_incoming,
        "prev_queue_sum_gu": prev_queue_sum_gu,
        "prev_queue_sum_uav": prev_queue_sum_uav,
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
    }


def _tensor_mapping(value) -> dict[str, torch.Tensor]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "_fields"):
        return {field_name: getattr(value, field_name) for field_name in value._fields}
    raise TypeError(f"Expected dict or NamedTuple tensor result, got {type(value)!r}")


def _assert_tensor_dict_close(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    left_map = _tensor_mapping(left)
    right_map = _tensor_mapping(right)
    assert left_map.keys() == right_map.keys()
    for key in left_map:
        if left_map[key].dtype == torch.bool:
            np.testing.assert_array_equal(
                left_map[key].detach().cpu().numpy(),
                right_map[key].detach().cpu().numpy(),
            )
        else:
            np.testing.assert_allclose(
                left_map[key].detach().cpu().numpy(),
                right_map[key].detach().cpu().numpy(),
                rtol=1.0e-6,
                atol=1.0e-6,
            )


def test_access_rate_kernel_cpu_path_matches_eager_even_with_compile_mode(monkeypatch):
    cfg = SaginConfig(num_uav=2, num_gu=4, users_obs_max=3)
    cfg.enable_bw_action = True
    cfg.interference_enabled = True
    compile_cfg = copy.deepcopy(cfg)
    compile_cfg.structured_kernel_operator_mode = "compile"
    eager_cfg = copy.deepcopy(cfg)
    eager_cfg.structured_kernel_operator_mode = "eager"

    compile_calls: list[tuple[str, bool, bool]] = []

    def _fake_compile(fn, *, mode=None, fullgraph=None, dynamic=None):
        compile_calls.append((str(mode), bool(fullgraph), bool(dynamic)))
        return fn

    monkeypatch.setattr(torch, "compile", _fake_compile)
    inputs = _access_kernel_inputs()
    eager = _apply_batched_access_rate_torch(
        **inputs,
        cfg=eager_cfg,
        device=torch.device("cpu"),
        compile_cfg=eager_cfg,
        return_tensors=True,
    )
    compiled = _apply_batched_access_rate_torch(
        **inputs,
        cfg=compile_cfg,
        device=torch.device("cpu"),
        compile_cfg=compile_cfg,
        return_tensors=True,
    )

    assert not compile_calls
    np.testing.assert_allclose(
        eager["rates"].detach().cpu().numpy(),
        compiled["rates"].detach().cpu().numpy(),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        eager["last_exec_bw_alloc"].detach().cpu().numpy(),
        compiled["last_exec_bw_alloc"].detach().cpu().numpy(),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        eager["last_bw_align"].detach().cpu().numpy(),
        compiled["last_bw_align"].detach().cpu().numpy(),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_bw_queue_kernel_cpu_path_stays_eager_fastpath(monkeypatch):
    cfg = SaginConfig(num_uav=2, num_gu=4, num_sat=3)
    cfg.structured_kernel_operator_mode = "compile"
    inputs = _bw_queue_inputs()

    compile_invocations = {"count": 0}

    def _fake_compile(fn, *, mode=None, fullgraph=None, dynamic=None):
        def _wrapped(*args, **kwargs):
            compile_invocations["count"] += 1
            raise RuntimeError("forced compile failure")

        return _wrapped

    monkeypatch.setattr(torch, "compile", _fake_compile)
    fallback = _apply_batched_bw_queue_transition_torch(
        **inputs,
        tau0=float(cfg.tau0),
        queue_max_uav=float(cfg.queue_max_uav),
        queue_max_sat=float(cfg.queue_max_sat),
        device=torch.device("cpu"),
        compile_cfg=cfg,
        return_tensors=True,
    )
    eager_cfg = copy.deepcopy(cfg)
    eager_cfg.structured_kernel_operator_mode = "eager"
    eager = _apply_batched_bw_queue_transition_torch(
        **inputs,
        tau0=float(cfg.tau0),
        queue_max_uav=float(cfg.queue_max_uav),
        queue_max_sat=float(cfg.queue_max_sat),
        device=torch.device("cpu"),
        compile_cfg=eager_cfg,
        return_tensors=True,
    )

    runtime = get_structured_kernel_runtime(cfg, device=torch.device("cpu"))
    assert compile_invocations["count"] == 0
    assert "bw_queue_transition" not in runtime.stats.fallback
    assert "bw_queue_transition" not in runtime.stats.enabled
    np.testing.assert_allclose(
        fallback["uav_queue"].detach().cpu().numpy(),
        eager["uav_queue"].detach().cpu().numpy(),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        fallback["sat_queue"].detach().cpu().numpy(),
        eager["sat_queue"].detach().cpu().numpy(),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_access_gain_kernel_compile_path_matches_eager(monkeypatch):
    cfg, envs, gu_pos, uav_pos = _access_gain_inputs()
    del envs
    compile_cfg = copy.deepcopy(cfg)
    compile_cfg.structured_kernel_operator_mode = "compile"
    eager_cfg = copy.deepcopy(cfg)
    eager_cfg.structured_kernel_operator_mode = "eager"
    device = torch.device("cpu")

    compile_calls: list[str] = []

    def _fake_compile(fn, *, mode=None, fullgraph=None, dynamic=None):
        del mode, fullgraph, dynamic
        compile_calls.append("compiled")
        return fn

    monkeypatch.setattr(torch, "compile", _fake_compile)
    eager_kernel = _resolve_kernel_callable(
        name="access_link_gain",
        eager_fn=_compute_access_link_gain_matrix_tensor_impl,
        compile_cfg=eager_cfg,
        device=device,
    )
    compiled_kernel = _resolve_kernel_callable(
        name="access_link_gain",
        eager_fn=_compute_access_link_gain_matrix_tensor_impl,
        compile_cfg=compile_cfg,
        device=device,
    )
    eager = eager_kernel(
        gu_pos_t=torch.as_tensor(gu_pos, dtype=torch.float32, device=device),
        uav_pos_t=torch.as_tensor(uav_pos, dtype=torch.float32, device=device),
        channel_params=_native_channel_static_params_from_cfg(eager_cfg),
        candidate_params=_native_candidate_static_params_from_cfg(eager_cfg),
    )
    compiled = compiled_kernel(
        gu_pos_t=torch.as_tensor(gu_pos, dtype=torch.float32, device=device),
        uav_pos_t=torch.as_tensor(uav_pos, dtype=torch.float32, device=device),
        channel_params=_native_channel_static_params_from_cfg(compile_cfg),
        candidate_params=_native_candidate_static_params_from_cfg(compile_cfg),
    )

    assert compile_calls
    np.testing.assert_allclose(
        eager.detach().cpu().numpy(),
        compiled.detach().cpu().numpy(),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_close_risk_kernel_compile_runtime_falls_back_to_eager(monkeypatch):
    cfg = SaginConfig(num_uav=3)
    cfg.close_risk_enabled = True
    cfg.danger_imitation_enabled = True
    cfg.structured_kernel_operator_mode = "compile"
    inputs = _close_risk_inputs()

    compile_invocations = {"count": 0}

    def _fake_compile(fn, *, mode=None, fullgraph=None, dynamic=None):
        del fn, mode, fullgraph, dynamic

        def _wrapped(*args, **kwargs):
            del args, kwargs
            compile_invocations["count"] += 1
            raise RuntimeError("forced compile failure")

        return _wrapped

    monkeypatch.setattr(torch, "compile", _fake_compile)
    fallback = _apply_batched_close_risk_and_danger_torch(
        **inputs,
        cfg=cfg,
        device=torch.device("cpu"),
        compile_cfg=cfg,
        return_tensors=True,
    )
    eager_cfg = copy.deepcopy(cfg)
    eager_cfg.structured_kernel_operator_mode = "eager"
    eager = _apply_batched_close_risk_and_danger_torch(
        **inputs,
        cfg=eager_cfg,
        device=torch.device("cpu"),
        compile_cfg=eager_cfg,
        return_tensors=True,
    )

    runtime = get_structured_kernel_runtime(cfg, device=torch.device("cpu"))
    assert compile_invocations["count"] == 0
    assert "close_risk_and_danger" not in runtime.stats.fallback
    for key in ("intervention_norm_uav", "intervention_norm", "intervention_rate", "intervention_norm_top1", "close_risk", "close_risk_uav", "danger_imitation_mask", "danger_imitation_active_rate"):
        np.testing.assert_allclose(
            fallback[key].detach().cpu().numpy(),
            eager[key].detach().cpu().numpy(),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
    np.testing.assert_array_equal(
        fallback["collision"].detach().cpu().numpy(),
        eager["collision"].detach().cpu().numpy(),
    )


def test_cuda_required_compile_kernel_raises_instead_of_fallback(monkeypatch):
    cfg = SaginConfig()
    cfg.structured_kernel_operator_mode = "compile"
    cfg.structured_kernel_compile_required = "strict_kernel"

    def _fake_compile(fn, *, mode=None, fullgraph=None, dynamic=None):
        del fn, mode, fullgraph, dynamic
        raise RuntimeError("forced compile init failure")

    monkeypatch.setattr(torch, "compile", _fake_compile)
    runtime = get_structured_kernel_runtime(cfg, device=torch.device("cuda"))

    with pytest.raises(RuntimeError, match="compile-required"):
        runtime.compile_kernel("strict_kernel", lambda x: x)
    assert "strict_kernel" not in runtime.stats.fallback


def test_cuda_required_compile_runtime_failure_raises_instead_of_fallback(monkeypatch):
    cfg = SaginConfig()
    cfg.structured_kernel_operator_mode = "compile"
    cfg.structured_kernel_compile_required = "strict_kernel"

    def _fake_compile(fn, *, mode=None, fullgraph=None, dynamic=None):
        del fn, mode, fullgraph, dynamic

        def _compiled(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("forced compiled runtime failure")

        return _compiled

    monkeypatch.setattr(torch, "compile", _fake_compile)
    runtime = get_structured_kernel_runtime(cfg, device=torch.device("cuda"))
    compiled = runtime.compile_kernel("strict_kernel", lambda x: x)

    with pytest.raises(RuntimeError, match="compile-required"):
        compiled(torch.zeros((1,), dtype=torch.float32))
    assert "strict_kernel" not in runtime.stats.fallback


def test_cuda_required_cudagraph_rejects_non_cuda_inputs_instead_of_fallback():
    cfg = SaginConfig()
    cfg.structured_kernel_operator_mode = "compile"
    cfg.structured_kernel_compile_backend = "cudagraphs"
    cfg.structured_kernel_compile_required = "strict_kernel"
    runtime = get_structured_kernel_runtime(cfg, device=torch.device("cuda"))
    compiled = runtime.compile_kernel("strict_kernel", lambda x: x)

    with pytest.raises(RuntimeError, match="manual cudagraph capture initialization failed"):
        compiled(torch.zeros((1,), dtype=torch.float32))
    assert "strict_kernel" not in runtime.stats.fallback


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for cudagraph output-buffer validation")
def test_cuda_required_cudagraph_outputs_are_persistent_buffers():
    cfg = SaginConfig()
    cfg.structured_kernel_operator_mode = "compile"
    cfg.structured_kernel_compile_backend = "cudagraphs"
    cfg.structured_kernel_compile_required = "strict_kernel"
    cfg.structured_kernel_compile_cudagraphs = True
    cfg.structured_kernel_cudagraph_mark_step_begin = True

    runtime = get_structured_kernel_runtime(cfg, device=torch.device("cuda"))
    compiled = runtime.compile_kernel("strict_kernel", lambda x: x + 1.0)
    first = compiled(torch.ones((8,), dtype=torch.float32, device="cuda"))
    first_ptr = int(first.data_ptr())
    torch.testing.assert_close(first, torch.full_like(first, 2.0))

    second = compiled(torch.full((8,), 3.0, dtype=torch.float32, device="cuda"))
    assert int(second.data_ptr()) == first_ptr
    torch.testing.assert_close(second, torch.full_like(second, 4.0))


def test_visible_sats_kernel_compile_path_matches_eager(monkeypatch):
    inputs = _mixed_hotpath_inputs()
    cfg = inputs["cfg"]
    compile_cfg = copy.deepcopy(cfg)
    compile_cfg.structured_kernel_operator_mode = "compile"
    eager_cfg = copy.deepcopy(cfg)
    eager_cfg.structured_kernel_operator_mode = "eager"
    compile_calls: list[str] = []

    def _fake_compile(fn, *, mode=None, fullgraph=None, dynamic=None):
        del mode, fullgraph, dynamic
        compile_calls.append("compiled")
        return fn

    monkeypatch.setattr(torch, "compile", _fake_compile)
    device = torch.device("cpu")
    eager_domains = _native_typed_domains_from_cfg(eager_cfg, num_envs=int(inputs["sat_pos_batch"].shape[0]))
    eager = _visible_sats_batch_tensor_impl(
        sat_pos_t=torch.as_tensor(inputs["sat_pos_batch"], dtype=torch.float32, device=device),
        uav_ecef_t=torch.as_tensor(inputs["uav_ecef_batch"], dtype=torch.float32, device=device),
        sat_queue_t=torch.as_tensor(inputs["sat_queue"], dtype=torch.float32, device=device),
        sat_load_t=torch.as_tensor(inputs["sat_load"], dtype=torch.float32, device=device),
        current_sel_mask_t=torch.as_tensor(inputs["current_sel_mask"], dtype=torch.bool, device=device),
        channel_params=eager_domains.channel,
        sat_geometry_params=eager_domains.sat_geometry,
        sat_orbit_radius_sq=float(inputs["envs"][0]._sat_orbit_radius_sq),
        uav_orbit_radius_sq=float(inputs["envs"][0]._uav_orbit_radius_sq),
        uav_orbit_radius=float(inputs["envs"][0]._uav_orbit_radius),
        backhaul_gain_const=float(inputs["envs"][0]._backhaul_gain_const),
        effective_b_backhaul_per_sat=float(inputs["envs"][0]._effective_b_backhaul_per_sat()),
    )
    compiled_fn = get_structured_kernel_runtime(compile_cfg, device=device).compile_kernel(
        "visible_sats",
        _visible_sats_batch_tensor_impl,
    )
    compiled_domains = _native_typed_domains_from_cfg(compile_cfg, num_envs=int(inputs["sat_pos_batch"].shape[0]))
    compiled = compiled_fn(
        sat_pos_t=torch.as_tensor(inputs["sat_pos_batch"], dtype=torch.float32, device=device),
        uav_ecef_t=torch.as_tensor(inputs["uav_ecef_batch"], dtype=torch.float32, device=device),
        sat_queue_t=torch.as_tensor(inputs["sat_queue"], dtype=torch.float32, device=device),
        sat_load_t=torch.as_tensor(inputs["sat_load"], dtype=torch.float32, device=device),
        current_sel_mask_t=torch.as_tensor(inputs["current_sel_mask"], dtype=torch.bool, device=device),
        channel_params=compiled_domains.channel,
        sat_geometry_params=compiled_domains.sat_geometry,
        sat_orbit_radius_sq=float(inputs["envs"][0]._sat_orbit_radius_sq),
        uav_orbit_radius_sq=float(inputs["envs"][0]._uav_orbit_radius_sq),
        uav_orbit_radius=float(inputs["envs"][0]._uav_orbit_radius),
        backhaul_gain_const=float(inputs["envs"][0]._backhaul_gain_const),
        effective_b_backhaul_per_sat=float(inputs["envs"][0]._effective_b_backhaul_per_sat()),
    )
    assert compile_calls
    _assert_tensor_dict_close(eager, compiled)


def test_global_state_kernel_compile_path_matches_eager(monkeypatch):
    inputs = _mixed_hotpath_inputs()
    cfg = inputs["cfg"]
    compile_cfg = copy.deepcopy(cfg)
    compile_cfg.structured_kernel_operator_mode = "compile"
    eager_cfg = copy.deepcopy(cfg)
    eager_cfg.structured_kernel_operator_mode = "eager"
    compile_calls: list[str] = []

    def _fake_compile(fn, *, mode=None, fullgraph=None, dynamic=None):
        del mode, fullgraph, dynamic
        compile_calls.append("compiled")
        return fn

    monkeypatch.setattr(torch, "compile", _fake_compile)
    device = torch.device("cpu")
    eager_domains = _native_typed_domains_from_cfg(eager_cfg, num_envs=int(inputs["uav_pos"].shape[0]))
    compiled_domains = _native_typed_domains_from_cfg(compile_cfg, num_envs=int(inputs["uav_pos"].shape[0]))
    eager = _build_global_state_tensor_impl(
        global_state_params=eager_domains.global_state,
        sat_geometry_params=eager_domains.sat_geometry,
        bw_workload_static_params=eager_domains.bw_workload,
        local_obs_params=eager_domains.local_obs,
        uav_pos_t=torch.as_tensor(inputs["uav_pos"], dtype=torch.float32, device=device),
        uav_vel_t=torch.as_tensor(inputs["uav_vel"], dtype=torch.float32, device=device),
        uav_queue_t=torch.as_tensor(inputs["uav_queue"], dtype=torch.float32, device=device),
        uav_energy_t=torch.as_tensor(inputs["uav_energy"], dtype=torch.float32, device=device),
        gu_pos_t=torch.as_tensor(inputs["gu_pos"], dtype=torch.float32, device=device),
        gu_queue_t=torch.as_tensor(inputs["gu_queue"], dtype=torch.float32, device=device),
        sat_pos_t=torch.as_tensor(inputs["sat_pos_batch"], dtype=torch.float32, device=device),
        sat_vel_t=torch.as_tensor(inputs["sat_vel"], dtype=torch.float32, device=device),
        sat_queue_t=torch.as_tensor(inputs["sat_queue"], dtype=torch.float32, device=device),
        t_t=torch.as_tensor(inputs["t"], dtype=torch.float32, device=device),
        arrival_ref_t=torch.as_tensor(inputs["arrival_ref"], dtype=torch.float32, device=device),
        gu_ema_t=torch.as_tensor(inputs["gu_ema"], dtype=torch.float32, device=device),
        uav_ema_t=torch.as_tensor(inputs["uav_ema"], dtype=torch.float32, device=device),
        sat_ema_t=torch.as_tensor(inputs["sat_ema"], dtype=torch.float32, device=device),
        assoc_t=torch.as_tensor(inputs["assoc"], dtype=torch.long, device=device),
        sat_selection_matrix_t=torch.as_tensor(inputs["sat_selection_matrix"], dtype=torch.long, device=device),
        arrival_rate_vec_t=torch.as_tensor(inputs["arrival_rate_vec"], dtype=torch.float32, device=device),
        recent_arrival_t=torch.as_tensor(inputs["recent_arrival"], dtype=torch.float32, device=device),
        recent_service_t=torch.as_tensor(inputs["recent_service"], dtype=torch.float32, device=device),
        urgency_risk_t=torch.as_tensor(inputs["urgency_risk"], dtype=torch.float32, device=device),
        downstream_pressure_t=torch.as_tensor(inputs["downstream_pressure"], dtype=torch.float32, device=device),
        service_gap_t=torch.as_tensor(inputs["service_gap"], dtype=torch.float32, device=device),
        service_gap_risk_t=torch.as_tensor(inputs["service_gap_risk"], dtype=torch.float32, device=device),
        deadline_steps_t=torch.as_tensor(inputs["deadline_steps"], dtype=torch.float32, device=device),
        deadline_slack_t=torch.as_tensor(inputs["deadline_slack"], dtype=torch.float32, device=device),
        deadline_risk_t=torch.as_tensor(inputs["deadline_risk"], dtype=torch.float32, device=device),
        sat_state_max=int(cfg.sat_state_max),
    )
    compiled_fn = get_structured_kernel_runtime(compile_cfg, device=device).compile_kernel(
        "global_state_batch",
        _build_global_state_tensor_impl,
    )
    compiled = compiled_fn(
        global_state_params=compiled_domains.global_state,
        sat_geometry_params=compiled_domains.sat_geometry,
        bw_workload_static_params=compiled_domains.bw_workload,
        local_obs_params=compiled_domains.local_obs,
        uav_pos_t=torch.as_tensor(inputs["uav_pos"], dtype=torch.float32, device=device),
        uav_vel_t=torch.as_tensor(inputs["uav_vel"], dtype=torch.float32, device=device),
        uav_queue_t=torch.as_tensor(inputs["uav_queue"], dtype=torch.float32, device=device),
        uav_energy_t=torch.as_tensor(inputs["uav_energy"], dtype=torch.float32, device=device),
        gu_pos_t=torch.as_tensor(inputs["gu_pos"], dtype=torch.float32, device=device),
        gu_queue_t=torch.as_tensor(inputs["gu_queue"], dtype=torch.float32, device=device),
        sat_pos_t=torch.as_tensor(inputs["sat_pos_batch"], dtype=torch.float32, device=device),
        sat_vel_t=torch.as_tensor(inputs["sat_vel"], dtype=torch.float32, device=device),
        sat_queue_t=torch.as_tensor(inputs["sat_queue"], dtype=torch.float32, device=device),
        t_t=torch.as_tensor(inputs["t"], dtype=torch.float32, device=device),
        arrival_ref_t=torch.as_tensor(inputs["arrival_ref"], dtype=torch.float32, device=device),
        gu_ema_t=torch.as_tensor(inputs["gu_ema"], dtype=torch.float32, device=device),
        uav_ema_t=torch.as_tensor(inputs["uav_ema"], dtype=torch.float32, device=device),
        sat_ema_t=torch.as_tensor(inputs["sat_ema"], dtype=torch.float32, device=device),
        assoc_t=torch.as_tensor(inputs["assoc"], dtype=torch.long, device=device),
        sat_selection_matrix_t=torch.as_tensor(inputs["sat_selection_matrix"], dtype=torch.long, device=device),
        arrival_rate_vec_t=torch.as_tensor(inputs["arrival_rate_vec"], dtype=torch.float32, device=device),
        recent_arrival_t=torch.as_tensor(inputs["recent_arrival"], dtype=torch.float32, device=device),
        recent_service_t=torch.as_tensor(inputs["recent_service"], dtype=torch.float32, device=device),
        urgency_risk_t=torch.as_tensor(inputs["urgency_risk"], dtype=torch.float32, device=device),
        downstream_pressure_t=torch.as_tensor(inputs["downstream_pressure"], dtype=torch.float32, device=device),
        service_gap_t=torch.as_tensor(inputs["service_gap"], dtype=torch.float32, device=device),
        service_gap_risk_t=torch.as_tensor(inputs["service_gap_risk"], dtype=torch.float32, device=device),
        deadline_steps_t=torch.as_tensor(inputs["deadline_steps"], dtype=torch.float32, device=device),
        deadline_slack_t=torch.as_tensor(inputs["deadline_slack"], dtype=torch.float32, device=device),
        deadline_risk_t=torch.as_tensor(inputs["deadline_risk"], dtype=torch.float32, device=device),
        sat_state_max=int(cfg.sat_state_max),
    )
    assert compile_calls
    np.testing.assert_allclose(
        eager.detach().cpu().numpy(),
        compiled.detach().cpu().numpy(),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_world_kernel_compile_path_matches_eager(monkeypatch):
    inputs = _mixed_hotpath_inputs()
    cfg = inputs["cfg"]
    compile_cfg = copy.deepcopy(cfg)
    compile_cfg.structured_kernel_operator_mode = "compile"
    eager_cfg = copy.deepcopy(cfg)
    eager_cfg.structured_kernel_operator_mode = "eager"
    compile_calls: list[str] = []

    def _fake_compile(fn, *, mode=None, fullgraph=None, dynamic=None):
        del mode, fullgraph, dynamic
        compile_calls.append("compiled")
        return fn

    monkeypatch.setattr(torch, "compile", _fake_compile)
    device = torch.device("cpu")
    eager_domains = _native_typed_domains_from_cfg(eager_cfg, num_envs=2)
    compile_domains = _native_typed_domains_from_cfg(compile_cfg, num_envs=2)
    eager = _build_world_from_packed_specs_tensor_impl(
        local_obs_params=eager_domains.local_obs,
        stage_ids_t=torch.tensor([1, 2], dtype=torch.long, device=device),
        effective_b_backhaul_per_sat_t=torch.full((2,), float(cfg.b_backhaul_per_sat), dtype=torch.float32, device=device),
        uav_pos_t=torch.as_tensor(inputs["uav_pos"], dtype=torch.float32, device=device),
        uav_vel_t=torch.as_tensor(inputs["uav_vel"], dtype=torch.float32, device=device),
        uav_energy_t=torch.as_tensor(inputs["uav_energy"], dtype=torch.float32, device=device),
        uav_queue_t=torch.as_tensor(inputs["uav_queue"], dtype=torch.float32, device=device),
        uav_assoc_uav_cost_t=torch.zeros((2, cfg.num_uav), dtype=torch.float32, device=device),
        gu_pos_t=torch.as_tensor(inputs["gu_pos"], dtype=torch.float32, device=device),
        gu_queue_t=torch.as_tensor(inputs["gu_queue"], dtype=torch.float32, device=device),
        gu_proxy_features_t=torch.as_tensor(inputs["gu_proxy_features"], dtype=torch.float32, device=device),
        assoc_t=torch.as_tensor(inputs["assoc"], dtype=torch.long, device=device),
        candidate_flag_t=torch.as_tensor(inputs["candidate_flag"], dtype=torch.float32, device=device),
        bw_valid_flag_t=torch.as_tensor(inputs["bw_valid_flag"], dtype=torch.float32, device=device),
        prev_assoc_flag_t=torch.as_tensor(inputs["prev_assoc_flag"], dtype=torch.float32, device=device),
        eta_ref_feature_t=torch.as_tensor(inputs["eta_ref_feature"], dtype=torch.float32, device=device),
        sat_pos_active_t=torch.as_tensor(inputs["sat_pos_batch"], dtype=torch.float32, device=device),
        sat_vel_active_t=torch.as_tensor(inputs["sat_vel"], dtype=torch.float32, device=device),
        sat_queue_active_t=torch.as_tensor(inputs["sat_queue"], dtype=torch.float32, device=device),
        sat_load_active_t=torch.as_tensor(inputs["sat_load"], dtype=torch.float32, device=device),
        sat_cost_norm_active_t=torch.as_tensor(inputs["sat_cost_norm_active"], dtype=torch.float32, device=device),
        sat_active_mask_t=torch.as_tensor(inputs["active_sat_mask"], dtype=torch.bool, device=device),
        rel_pos_active_t=torch.as_tensor(inputs["rel_pos_active"], dtype=torch.float32, device=device),
        rel_vel_active_t=torch.as_tensor(inputs["rel_vel_active"], dtype=torch.float32, device=device),
        gain_active_t=torch.as_tensor(inputs["gain_active"], dtype=torch.float32, device=device),
        nu_eff_active_t=torch.as_tensor(inputs["nu_eff_active"], dtype=torch.float32, device=device),
        visible_flag_active_t=torch.as_tensor(inputs["visible_flag_active"], dtype=torch.float32, device=device),
        valid_flag_active_t=torch.as_tensor(inputs["valid_flag_active"], dtype=torch.float32, device=device),
        current_sel_flag_active_t=torch.as_tensor(inputs["current_sel_flag_active"], dtype=torch.float32, device=device),
    )
    compiled_fn = get_structured_kernel_runtime(compile_cfg, device=device).compile_kernel(
        "world_from_specs",
        _build_world_from_packed_specs_tensor_impl,
    )
    compiled = compiled_fn(
        local_obs_params=compile_domains.local_obs,
        stage_ids_t=torch.tensor([1, 2], dtype=torch.long, device=device),
        effective_b_backhaul_per_sat_t=torch.full((2,), float(cfg.b_backhaul_per_sat), dtype=torch.float32, device=device),
        uav_pos_t=torch.as_tensor(inputs["uav_pos"], dtype=torch.float32, device=device),
        uav_vel_t=torch.as_tensor(inputs["uav_vel"], dtype=torch.float32, device=device),
        uav_energy_t=torch.as_tensor(inputs["uav_energy"], dtype=torch.float32, device=device),
        uav_queue_t=torch.as_tensor(inputs["uav_queue"], dtype=torch.float32, device=device),
        uav_assoc_uav_cost_t=torch.zeros((2, cfg.num_uav), dtype=torch.float32, device=device),
        gu_pos_t=torch.as_tensor(inputs["gu_pos"], dtype=torch.float32, device=device),
        gu_queue_t=torch.as_tensor(inputs["gu_queue"], dtype=torch.float32, device=device),
        gu_proxy_features_t=torch.as_tensor(inputs["gu_proxy_features"], dtype=torch.float32, device=device),
        assoc_t=torch.as_tensor(inputs["assoc"], dtype=torch.long, device=device),
        candidate_flag_t=torch.as_tensor(inputs["candidate_flag"], dtype=torch.float32, device=device),
        bw_valid_flag_t=torch.as_tensor(inputs["bw_valid_flag"], dtype=torch.float32, device=device),
        prev_assoc_flag_t=torch.as_tensor(inputs["prev_assoc_flag"], dtype=torch.float32, device=device),
        eta_ref_feature_t=torch.as_tensor(inputs["eta_ref_feature"], dtype=torch.float32, device=device),
        sat_pos_active_t=torch.as_tensor(inputs["sat_pos_batch"], dtype=torch.float32, device=device),
        sat_vel_active_t=torch.as_tensor(inputs["sat_vel"], dtype=torch.float32, device=device),
        sat_queue_active_t=torch.as_tensor(inputs["sat_queue"], dtype=torch.float32, device=device),
        sat_load_active_t=torch.as_tensor(inputs["sat_load"], dtype=torch.float32, device=device),
        sat_cost_norm_active_t=torch.as_tensor(inputs["sat_cost_norm_active"], dtype=torch.float32, device=device),
        sat_active_mask_t=torch.as_tensor(inputs["active_sat_mask"], dtype=torch.bool, device=device),
        rel_pos_active_t=torch.as_tensor(inputs["rel_pos_active"], dtype=torch.float32, device=device),
        rel_vel_active_t=torch.as_tensor(inputs["rel_vel_active"], dtype=torch.float32, device=device),
        gain_active_t=torch.as_tensor(inputs["gain_active"], dtype=torch.float32, device=device),
        nu_eff_active_t=torch.as_tensor(inputs["nu_eff_active"], dtype=torch.float32, device=device),
        visible_flag_active_t=torch.as_tensor(inputs["visible_flag_active"], dtype=torch.float32, device=device),
        valid_flag_active_t=torch.as_tensor(inputs["valid_flag_active"], dtype=torch.float32, device=device),
        current_sel_flag_active_t=torch.as_tensor(inputs["current_sel_flag_active"], dtype=torch.float32, device=device),
    )
    assert compile_calls
    _assert_tensor_dict_close(eager, compiled)


def test_bw_step_metrics_kernel_compile_path_matches_eager(monkeypatch):
    inputs = _mixed_hotpath_inputs()
    cfg = inputs["cfg"]
    compile_cfg = copy.deepcopy(cfg)
    compile_cfg.structured_kernel_operator_mode = "compile"
    eager_cfg = copy.deepcopy(cfg)
    eager_cfg.structured_kernel_operator_mode = "eager"
    compile_calls: list[str] = []

    def _fake_compile(fn, *, mode=None, fullgraph=None, dynamic=None):
        del mode, fullgraph, dynamic
        compile_calls.append("compiled")
        return fn

    monkeypatch.setattr(torch, "compile", _fake_compile)
    device = torch.device("cpu")
    eager_domains = _native_typed_domains_from_cfg(eager_cfg, num_envs=int(inputs["gu_ema"].shape[0]))
    eager = _build_fast_bw_step_metrics_tensor_impl(
        reward_metrics_params=eager_domains.reward_metrics,
        bw_workload_static_params=eager_domains.bw_workload,
        reward_mode=str(cfg.reward_mode),
        gu_ema_prev_t=torch.as_tensor(inputs["gu_ema"], dtype=torch.float32, device=device),
        uav_ema_prev_t=torch.as_tensor(inputs["uav_ema"], dtype=torch.float32, device=device),
        sat_ema_prev_t=torch.as_tensor(inputs["sat_ema"], dtype=torch.float32, device=device),
        gu_outflow_t=torch.as_tensor(inputs["gu_outflow"], dtype=torch.float32, device=device),
        uav_outflow_t=torch.as_tensor(inputs["uav_outflow"], dtype=torch.float32, device=device),
        sat_processed_t=torch.as_tensor(inputs["sat_processed"], dtype=torch.float32, device=device),
        assoc_t=torch.as_tensor(inputs["assoc"], dtype=torch.long, device=device),
        sat_selection_matrix_t=torch.as_tensor(inputs["sat_selection_matrix"], dtype=torch.long, device=device),
        gu_queue_before_t=torch.as_tensor(inputs["gu_queue_before"], dtype=torch.float32, device=device),
        uav_queue_before_t=torch.as_tensor(inputs["uav_queue_before"], dtype=torch.float32, device=device),
        sat_queue_before_t=torch.as_tensor(inputs["sat_queue_before"], dtype=torch.float32, device=device),
        arrivals_t=torch.as_tensor(inputs["arrivals"], dtype=torch.float32, device=device),
        gu_queue_after_t=torch.as_tensor(inputs["gu_queue_after"], dtype=torch.float32, device=device),
        uav_queue_after_t=torch.as_tensor(inputs["uav_queue_after"], dtype=torch.float32, device=device),
        sat_queue_after_t=torch.as_tensor(inputs["sat_queue_after"], dtype=torch.float32, device=device),
        gu_drop_t=torch.as_tensor(inputs["gu_drop"], dtype=torch.float32, device=device),
        gu_expired_t=torch.zeros_like(torch.as_tensor(inputs["gu_drop"], dtype=torch.float32, device=device)),
        uav_drop_t=torch.as_tensor(inputs["uav_drop"], dtype=torch.float32, device=device),
        sat_drop_t=torch.as_tensor(inputs["sat_drop"], dtype=torch.float32, device=device),
        last_sat_incoming_t=torch.as_tensor(inputs["last_sat_incoming"], dtype=torch.float32, device=device),
        arrival_ref_t=torch.as_tensor(inputs["arrival_ref"], dtype=torch.float32, device=device),
        prev_queue_sum_gu_t=torch.as_tensor(inputs["prev_queue_sum_gu"], dtype=torch.float32, device=device),
        prev_queue_sum_uav_t=torch.as_tensor(inputs["prev_queue_sum_uav"], dtype=torch.float32, device=device),
        prev_queue_sum_sat_t=torch.as_tensor(inputs["sat_queue_before"], dtype=torch.float32, device=device).sum(dim=1),
        effective_task_arrival_rate_t=torch.as_tensor(inputs["arrival_ref"], dtype=torch.float32, device=device)
        / max(float(cfg.num_gu) * float(cfg.tau0), 1.0e-9),
        gu_pos_t=torch.as_tensor(inputs["gu_pos"], dtype=torch.float32, device=device),
        uav_pos_t=torch.as_tensor(inputs["uav_pos"], dtype=torch.float32, device=device),
        last_exec_accel_t=torch.zeros_like(torch.as_tensor(inputs["uav_vel"], dtype=torch.float32, device=device)),
        last_energy_cost_t=torch.zeros((2, int(cfg.num_uav)), dtype=torch.float32, device=device),
        global_step_t=torch.as_tensor(inputs["t"], dtype=torch.float32, device=device),
        prev_q_norm_active_t=torch.zeros((2,), dtype=torch.float32, device=device),
        gu_urgency_risk_t=torch.as_tensor(inputs["urgency_risk"], dtype=torch.float32, device=device),
        downstream_pressure_t=torch.as_tensor(inputs["downstream_pressure"], dtype=torch.float32, device=device),
        service_gap_t=torch.as_tensor(inputs["service_gap"], dtype=torch.float32, device=device),
        service_gap_risk_t=torch.as_tensor(inputs["service_gap_risk"], dtype=torch.float32, device=device),
        intervention_norm_uav_t=torch.as_tensor(inputs["intervention_norm_uav"], dtype=torch.float32, device=device),
        close_risk_uav_t=torch.as_tensor(inputs["close_risk_uav"], dtype=torch.float32, device=device),
        danger_imitation_mask_t=torch.as_tensor(inputs["danger_imitation_mask"], dtype=torch.float32, device=device),
        intervention_norm_t=torch.as_tensor(inputs["intervention_norm"], dtype=torch.float32, device=device),
        intervention_rate_t=torch.as_tensor(inputs["intervention_rate"], dtype=torch.float32, device=device),
        intervention_norm_top1_t=torch.as_tensor(inputs["intervention_norm_top1"], dtype=torch.float32, device=device),
        close_risk_t=torch.as_tensor(inputs["close_risk"], dtype=torch.float32, device=device),
        danger_imitation_active_rate_t=torch.as_tensor(inputs["danger_imitation_active_rate"], dtype=torch.float32, device=device),
        collision_t=torch.as_tensor(inputs["collision"], dtype=torch.bool, device=device),
        t_t=torch.as_tensor(inputs["t"], dtype=torch.float32, device=device),
        uav_energy_after_t=torch.as_tensor(inputs["uav_energy"], dtype=torch.float32, device=device),
    )
    compiled_fn = get_structured_kernel_runtime(compile_cfg, device=device).compile_kernel(
        "bw_step_metrics",
        _build_fast_bw_step_metrics_tensor_impl,
    )
    compiled_domains = _native_typed_domains_from_cfg(compile_cfg, num_envs=int(inputs["gu_ema"].shape[0]))
    compiled = compiled_fn(
        reward_metrics_params=compiled_domains.reward_metrics,
        bw_workload_static_params=compiled_domains.bw_workload,
        reward_mode=str(cfg.reward_mode),
        gu_ema_prev_t=torch.as_tensor(inputs["gu_ema"], dtype=torch.float32, device=device),
        uav_ema_prev_t=torch.as_tensor(inputs["uav_ema"], dtype=torch.float32, device=device),
        sat_ema_prev_t=torch.as_tensor(inputs["sat_ema"], dtype=torch.float32, device=device),
        gu_outflow_t=torch.as_tensor(inputs["gu_outflow"], dtype=torch.float32, device=device),
        uav_outflow_t=torch.as_tensor(inputs["uav_outflow"], dtype=torch.float32, device=device),
        sat_processed_t=torch.as_tensor(inputs["sat_processed"], dtype=torch.float32, device=device),
        assoc_t=torch.as_tensor(inputs["assoc"], dtype=torch.long, device=device),
        sat_selection_matrix_t=torch.as_tensor(inputs["sat_selection_matrix"], dtype=torch.long, device=device),
        gu_queue_before_t=torch.as_tensor(inputs["gu_queue_before"], dtype=torch.float32, device=device),
        uav_queue_before_t=torch.as_tensor(inputs["uav_queue_before"], dtype=torch.float32, device=device),
        sat_queue_before_t=torch.as_tensor(inputs["sat_queue_before"], dtype=torch.float32, device=device),
        arrivals_t=torch.as_tensor(inputs["arrivals"], dtype=torch.float32, device=device),
        gu_queue_after_t=torch.as_tensor(inputs["gu_queue_after"], dtype=torch.float32, device=device),
        uav_queue_after_t=torch.as_tensor(inputs["uav_queue_after"], dtype=torch.float32, device=device),
        sat_queue_after_t=torch.as_tensor(inputs["sat_queue_after"], dtype=torch.float32, device=device),
        gu_drop_t=torch.as_tensor(inputs["gu_drop"], dtype=torch.float32, device=device),
        gu_expired_t=torch.zeros_like(torch.as_tensor(inputs["gu_drop"], dtype=torch.float32, device=device)),
        uav_drop_t=torch.as_tensor(inputs["uav_drop"], dtype=torch.float32, device=device),
        sat_drop_t=torch.as_tensor(inputs["sat_drop"], dtype=torch.float32, device=device),
        last_sat_incoming_t=torch.as_tensor(inputs["last_sat_incoming"], dtype=torch.float32, device=device),
        arrival_ref_t=torch.as_tensor(inputs["arrival_ref"], dtype=torch.float32, device=device),
        prev_queue_sum_gu_t=torch.as_tensor(inputs["prev_queue_sum_gu"], dtype=torch.float32, device=device),
        prev_queue_sum_uav_t=torch.as_tensor(inputs["prev_queue_sum_uav"], dtype=torch.float32, device=device),
        prev_queue_sum_sat_t=torch.as_tensor(inputs["sat_queue_before"], dtype=torch.float32, device=device).sum(dim=1),
        effective_task_arrival_rate_t=torch.as_tensor(inputs["arrival_ref"], dtype=torch.float32, device=device)
        / max(float(cfg.num_gu) * float(cfg.tau0), 1.0e-9),
        gu_pos_t=torch.as_tensor(inputs["gu_pos"], dtype=torch.float32, device=device),
        uav_pos_t=torch.as_tensor(inputs["uav_pos"], dtype=torch.float32, device=device),
        last_exec_accel_t=torch.zeros_like(torch.as_tensor(inputs["uav_vel"], dtype=torch.float32, device=device)),
        last_energy_cost_t=torch.zeros((2, int(cfg.num_uav)), dtype=torch.float32, device=device),
        global_step_t=torch.as_tensor(inputs["t"], dtype=torch.float32, device=device),
        prev_q_norm_active_t=torch.zeros((2,), dtype=torch.float32, device=device),
        gu_urgency_risk_t=torch.as_tensor(inputs["urgency_risk"], dtype=torch.float32, device=device),
        downstream_pressure_t=torch.as_tensor(inputs["downstream_pressure"], dtype=torch.float32, device=device),
        service_gap_t=torch.as_tensor(inputs["service_gap"], dtype=torch.float32, device=device),
        service_gap_risk_t=torch.as_tensor(inputs["service_gap_risk"], dtype=torch.float32, device=device),
        intervention_norm_uav_t=torch.as_tensor(inputs["intervention_norm_uav"], dtype=torch.float32, device=device),
        close_risk_uav_t=torch.as_tensor(inputs["close_risk_uav"], dtype=torch.float32, device=device),
        danger_imitation_mask_t=torch.as_tensor(inputs["danger_imitation_mask"], dtype=torch.float32, device=device),
        intervention_norm_t=torch.as_tensor(inputs["intervention_norm"], dtype=torch.float32, device=device),
        intervention_rate_t=torch.as_tensor(inputs["intervention_rate"], dtype=torch.float32, device=device),
        intervention_norm_top1_t=torch.as_tensor(inputs["intervention_norm_top1"], dtype=torch.float32, device=device),
        close_risk_t=torch.as_tensor(inputs["close_risk"], dtype=torch.float32, device=device),
        danger_imitation_active_rate_t=torch.as_tensor(inputs["danger_imitation_active_rate"], dtype=torch.float32, device=device),
        collision_t=torch.as_tensor(inputs["collision"], dtype=torch.bool, device=device),
        t_t=torch.as_tensor(inputs["t"], dtype=torch.float32, device=device),
        uav_energy_after_t=torch.as_tensor(inputs["uav_energy"], dtype=torch.float32, device=device),
    )
    assert compile_calls
    _assert_tensor_dict_close(eager, compiled)
