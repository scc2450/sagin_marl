from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from sagin_marl.env import channel
from sagin_marl.env.config import SaginConfig
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_batch_env_core import _backhaul_loss_factor_batch
from sagin_marl.env.structured_batch_env_core import (
    _apply_batched_access_rate_static_tensor_impl,
    _apply_batched_bw_post_stats_tensor_impl,
    _apply_batched_close_risk_and_danger_tensor_impl,
    _bw_weighted_workload_device_costs_batch,
    _native_access_rate_static_params_from_cfg,
    _native_typed_domains_from_cfg,
    _sat_overlap_eval_batch,
    _semantic_quantum,
)


def _access_gain(cfg: SaginConfig, gu_pos: np.ndarray, uav_pos: np.ndarray) -> float:
    d2d = float(np.linalg.norm(np.asarray(gu_pos, dtype=np.float64) - np.asarray(uav_pos, dtype=np.float64)))
    d3d = math.sqrt(d2d * d2d + float(cfg.uav_height) ** 2)
    phi = math.asin(float(cfg.uav_height) / (d3d + 1e-9))
    pl = channel.pathloss_db(np.array([d3d], dtype=np.float64), np.array([phi], dtype=np.float64), cfg)[0]
    return float(10 ** (-pl / 10.0))


def _backhaul_gain(cfg: SaginConfig, sat_pos: np.ndarray, uav_ecef: np.ndarray) -> float:
    dist = float(np.linalg.norm(np.asarray(sat_pos, dtype=np.float64) - np.asarray(uav_ecef, dtype=np.float64))) + 1e-9
    gain = (float(cfg.speed_of_light) / (4.0 * math.pi * float(cfg.carrier_freq) * dist)) ** 2
    gain *= float(cfg.uav_tx_gain) * float(cfg.sat_rx_gain)
    return float(gain)


def _access_spectral_efficiency_for_cfg(cfg: SaginConfig, snr: np.ndarray) -> np.ndarray:
    if bool(cfg.fading_enabled) and channel.access_fading_mode_from_config(cfg) == "ergodic_rician":
        return channel.rician_ergodic_spectral_efficiency(
            snr,
            channel.rician_k_linear_from_config(cfg),
            quadrature_points=int(getattr(cfg, "access_ergodic_rician_quadrature_points", 16) or 16),
        )
    return channel.spectral_efficiency(snr)


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
    return_eta_slots: bool = False,
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
    }
    if return_eta_slots:
        result_t["eta_slots"] = eta_slots_t
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
):
    kernel_device = torch.device("cpu") if device is None else torch.device(device)
    outputs = _apply_batched_close_risk_and_danger_tensor_impl(
        pos_t=torch.as_tensor(uav_pos, dtype=torch.float32, device=kernel_device),
        vel_t=torch.as_tensor(uav_vel, dtype=torch.float32, device=kernel_device),
        exec_accel_t=torch.as_tensor(last_exec_accel, dtype=torch.float32, device=kernel_device),
        policy_accel_t=torch.as_tensor(last_policy_accel, dtype=torch.float32, device=kernel_device),
        params=_native_typed_domains_from_cfg(cfg, num_envs=int(np.asarray(uav_pos).shape[0])).post_stats_safety,
        upper_pair_mask_t=torch.triu(
            torch.ones((1, int(cfg.num_uav), int(cfg.num_uav)), dtype=torch.bool, device=kernel_device),
            diagonal=1,
        ),
    )
    return {key: getattr(outputs, key).detach().cpu().numpy() for key in outputs._fields}


def _apply_batched_bw_post_stats_torch(
    *,
    sat_pos,
    sat_selection_matrix,
    uav_ecef,
    gu_queue,
    last_gu_outflow,
    next_arrival_rates,
    uav_queue,
    sat_queue,
    last_sat_connection_counts,
    last_gu_service_gap,
    cfg,
    uav_orbit_radius,
    uav_orbit_radius_sq,
    sat_orbit_radius_sq,
    device,
):
    kernel_device = torch.device("cpu") if device is None else torch.device(device)
    result = _apply_batched_bw_post_stats_tensor_impl(
        sat_pos_t=torch.as_tensor(sat_pos, dtype=torch.float64, device=kernel_device),
        selection_t=torch.as_tensor(sat_selection_matrix, dtype=torch.long, device=kernel_device),
        uav_ecef_t=torch.as_tensor(uav_ecef, dtype=torch.float64, device=kernel_device),
        gu_queue_t=torch.as_tensor(gu_queue, dtype=torch.float32, device=kernel_device),
        last_gu_outflow_t=torch.as_tensor(last_gu_outflow, dtype=torch.float32, device=kernel_device),
        next_arrival_rates_t=torch.as_tensor(next_arrival_rates, dtype=torch.float32, device=kernel_device),
        uav_queue_t=torch.as_tensor(uav_queue, dtype=torch.float32, device=kernel_device),
        sat_queue_t=torch.as_tensor(sat_queue, dtype=torch.float32, device=kernel_device),
        sat_connection_counts_t=torch.as_tensor(last_sat_connection_counts, dtype=torch.float32, device=kernel_device),
        last_gu_service_gap_t=torch.as_tensor(last_gu_service_gap, dtype=torch.float32, device=kernel_device),
        params=_native_typed_domains_from_cfg(cfg, num_envs=int(np.asarray(sat_selection_matrix).shape[0])).post_stats_safety,
        uav_orbit_radius=uav_orbit_radius,
        uav_orbit_radius_sq=uav_orbit_radius_sq,
        sat_orbit_radius_sq=sat_orbit_radius_sq,
    )
    keys = (
        "last_connected_sat_count",
        "last_connected_sat_dist_mean",
        "last_connected_sat_dist_p95",
        "last_connected_sat_elevation_deg_mean",
        "last_connected_sat_elevation_deg_min",
        "last_gu_urgency_risk",
        "last_gu_downstream_pressure",
        "last_gu_service_gap_risk",
    )
    return {key: value.detach().cpu().numpy() for key, value in zip(keys, result)}


def test_access_rate_uses_beta_scaled_noise_bandwidth():
    cfg = SaginConfig(num_uav=1, num_gu=2, num_sat=1, users_obs_max=2, sats_obs_max=1)
    cfg.enable_bw_action = True
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed)
    env.uav_pos[0] = np.array([0.0, 0.0], dtype=np.float32)
    env.gu_pos[0] = np.array([0.0, 0.0], dtype=np.float32)
    env.gu_pos[1] = np.array([0.0, 0.0], dtype=np.float32)

    assoc = np.array([0, 0], dtype=np.int32)
    candidates = [[0, 1]]
    actions = {
        env.agents[0]: {
            "bw_alloc": np.array([0.25, 0.75], dtype=np.float32),
        }
    }

    gain_matrix = env._compute_access_link_gain_matrix()
    rates, eta = env._compute_access_rates(assoc, candidates, actions, record_exec=False, access_snapshot=gain_matrix)

    gain = float(gain_matrix[0, 0])
    beta = np.array([0.25, 0.75], dtype=np.float64)
    eff_bw = beta * float(cfg.b_acc)
    expected_eta_values = channel.quantize_array(
        _access_spectral_efficiency_for_cfg(
            cfg,
            channel.snr_linear(
                float(cfg.gu_tx_power),
                np.full((2,), gain, dtype=np.float64),
                float(cfg.noise_density),
                eff_bw,
                noise_figure_db=float(cfg.access_noise_figure_db),
            ),
        ),
        _semantic_quantum(cfg, "structured_access_eta_quantum", 1.0e-6),
        dtype=np.float32,
    )
    expected_rates = channel.quantize_array(
        eff_bw * expected_eta_values,
        _semantic_quantum(cfg, "structured_access_rate_quantum", 32.0),
        dtype=np.float32,
    )
    expected_eta = float(
        channel.quantize_array(
            _access_spectral_efficiency_for_cfg(
                cfg,
                channel.snr_linear(
                    float(cfg.gu_tx_power),
                    np.array([gain], dtype=np.float64),
                    float(cfg.noise_density),
                    float(cfg.b_acc),
                    noise_figure_db=float(cfg.access_noise_figure_db),
                ),
            ),
            _semantic_quantum(cfg, "structured_access_eta_quantum", 1.0e-6),
            dtype=np.float32,
        )[0]
    )

    np.testing.assert_allclose(rates, expected_rates.astype(np.float32), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(eta[0, :2], np.full((2,), expected_eta, dtype=np.float32), rtol=1e-5, atol=1e-5)


def test_access_rate_matrix_fast_path_matches_dict_path():
    cfg = SaginConfig(num_uav=2, num_gu=3, num_sat=1, users_obs_max=2, sats_obs_max=1)
    cfg.enable_bw_action = True
    cfg.interference_enabled = True
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed)
    env.uav_pos[0] = np.array([0.0, 0.0], dtype=np.float32)
    env.uav_pos[1] = np.array([400.0, 0.0], dtype=np.float32)
    env.gu_pos[0] = np.array([0.0, 0.0], dtype=np.float32)
    env.gu_pos[1] = np.array([0.0, 0.0], dtype=np.float32)
    env.gu_pos[2] = np.array([400.0, 0.0], dtype=np.float32)

    assoc = np.array([0, 0, 1], dtype=np.int32)
    candidates = [[0, 1], [2]]
    bw_alloc = np.array(
        [
            [0.25, 0.75, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    actions = {
        env.agents[0]: {"bw_alloc": bw_alloc[0]},
        env.agents[1]: {"bw_alloc": bw_alloc[1]},
    }

    rates_dict, eta_dict = env._compute_access_rates(assoc, candidates, actions, record_exec=False)
    rates_matrix, eta_matrix = env._compute_access_rates(assoc, candidates, bw_alloc, record_exec=False)

    np.testing.assert_allclose(rates_matrix, rates_dict, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(eta_matrix, eta_dict, rtol=1e-6, atol=1e-6)


def test_apply_uav_dynamics_matrix_fast_path_matches_dict_path():
    cfg = SaginConfig(num_uav=2, num_gu=2, num_sat=2, users_obs_max=2, sats_obs_max=2, nbrs_obs_max=1)
    cfg.avoidance_enabled = False
    cfg.energy_enabled = False
    env_dict = SaginParallelEnv(cfg)
    env_matrix = SaginParallelEnv(cfg)
    env_dict.reset(seed=cfg.seed)
    env_matrix.reset(seed=cfg.seed)

    for env in (env_dict, env_matrix):
        env.uav_pos[:] = np.array([[120.0, 80.0], [220.0, 140.0]], dtype=np.float32)
        env.uav_vel[:] = np.array([[3.0, -2.0], [-1.0, 4.0]], dtype=np.float32)
        env._refresh_uav_cache()

    accel_matrix = np.array([[0.25, -0.5], [-0.75, 0.5]], dtype=np.float32)
    action_dict = {
        env_dict.agents[0]: {"accel": accel_matrix[0]},
        env_dict.agents[1]: {"accel": accel_matrix[1]},
    }

    env_dict._apply_uav_dynamics(action_dict)
    env_matrix._apply_uav_dynamics(accel_matrix)

    np.testing.assert_allclose(env_matrix.uav_pos, env_dict.uav_pos, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(env_matrix.uav_vel, env_dict.uav_vel, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(env_matrix.last_policy_accel, env_dict.last_policy_accel, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(env_matrix.last_exec_accel, env_dict.last_exec_accel, rtol=1e-6, atol=1e-6)


def test_access_interference_uses_average_spectrum_overlap_approximation():
    cfg = SaginConfig(num_uav=2, num_gu=3, num_sat=1, users_obs_max=2, sats_obs_max=1)
    cfg.enable_bw_action = True
    cfg.interference_enabled = True
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed)
    env.uav_pos[0] = np.array([0.0, 0.0], dtype=np.float32)
    env.uav_pos[1] = np.array([400.0, 0.0], dtype=np.float32)
    env.gu_pos[0] = np.array([0.0, 0.0], dtype=np.float32)
    env.gu_pos[1] = np.array([0.0, 0.0], dtype=np.float32)
    env.gu_pos[2] = np.array([400.0, 0.0], dtype=np.float32)

    assoc = np.array([0, 0, 1], dtype=np.int32)
    candidates = [[0, 1], [2]]
    actions = {
        env.agents[0]: {
            "bw_alloc": np.array([0.25, 0.75, 0.0], dtype=np.float32),
        },
        env.agents[1]: {
            "bw_alloc": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        },
    }

    gain_matrix = env._compute_access_link_gain_matrix()
    rates, eta = env._compute_access_rates(assoc, candidates, actions, record_exec=False, access_snapshot=gain_matrix)

    gain_sig = float(gain_matrix[0, 0])
    gain_int = float(gain_matrix[2, 0])
    beta = np.array([0.25, 0.75], dtype=np.float64)
    eff_bw = beta * float(cfg.b_acc)
    interference = float(cfg.gu_tx_power) * gain_int
    expected_eta_values = channel.quantize_array(
        _access_spectral_efficiency_for_cfg(
            cfg,
            channel.snr_linear(
                float(cfg.gu_tx_power),
                np.full((2,), gain_sig, dtype=np.float64),
                float(cfg.noise_density),
                eff_bw,
                beta * interference,
                noise_figure_db=float(cfg.access_noise_figure_db),
            )
        ),
        _semantic_quantum(cfg, "structured_access_eta_quantum", 1.0e-6),
        dtype=np.float32,
    )
    expected_rates = channel.quantize_array(
        eff_bw * expected_eta_values,
        _semantic_quantum(cfg, "structured_access_rate_quantum", 32.0),
        dtype=np.float32,
    )
    expected_eta = float(
        channel.quantize_array(
            _access_spectral_efficiency_for_cfg(
                cfg,
                channel.snr_linear(
                    float(cfg.gu_tx_power),
                    np.array([gain_sig], dtype=np.float64),
                    float(cfg.noise_density),
                    float(cfg.b_acc),
                    interference,
                    noise_figure_db=float(cfg.access_noise_figure_db),
                ),
            ),
            _semantic_quantum(cfg, "structured_access_eta_quantum", 1.0e-6),
            dtype=np.float32,
        )[0]
    )

    np.testing.assert_allclose(rates[:2], expected_rates.astype(np.float32), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(eta[0, :2], np.full((2,), expected_eta, dtype=np.float32), rtol=1e-5, atol=1e-5)


def test_access_interference_power_scales_with_interferer_beta():
    cfg = SaginConfig(num_uav=2, num_gu=3, num_sat=1, users_obs_max=3, sats_obs_max=1)
    cfg.enable_bw_action = True
    cfg.interference_enabled = True
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed)
    assoc = np.array([0, 1, 1], dtype=np.int32)
    candidates = [[0], [1, 2]]
    gain_matrix = np.array(
        [
            [10.0, 0.0],
            [2.0, 3.0],
            [0.0, 4.0],
        ],
        dtype=np.float32,
    )

    def recorded_interference(interferer_beta: float) -> np.ndarray:
        actions = {
            env.agents[0]: {"bw_alloc": np.array([1.0, 0.0, 0.0], dtype=np.float32)},
            env.agents[1]: {
                "bw_alloc": np.array([0.0, interferer_beta, 1.0 - interferer_beta], dtype=np.float32),
            },
        }
        env._compute_access_rates(assoc, candidates, actions, record_exec=True, access_snapshot=gain_matrix)
        return env.last_access_interference_by_uav.copy()

    zero = recorded_interference(0.0)
    small = recorded_interference(0.01)
    full = recorded_interference(1.0)

    assert zero[0] == 0.0
    assert small[0] > 0.0
    np.testing.assert_allclose(small[0], 0.01 * full[0], rtol=1.0e-4, atol=1.0e-8)


def test_batched_access_rate_torch_matches_env_reference():
    cfg = SaginConfig(num_uav=2, num_gu=4, num_sat=1, users_obs_max=3, sats_obs_max=1)
    cfg.enable_bw_action = True
    cfg.interference_enabled = True

    env_a = SaginParallelEnv(cfg)
    env_b = SaginParallelEnv(cfg)
    env_a.reset(seed=cfg.seed)
    env_b.reset(seed=cfg.seed + 1)

    for env in (env_a, env_b):
        env.uav_pos[0] = np.array([0.0, 0.0], dtype=np.float32)
        env.uav_pos[1] = np.array([300.0, 0.0], dtype=np.float32)
        env.gu_pos[0] = np.array([0.0, 0.0], dtype=np.float32)
        env.gu_pos[1] = np.array([40.0, 0.0], dtype=np.float32)
        env.gu_pos[2] = np.array([300.0, 0.0], dtype=np.float32)
        env.gu_pos[3] = np.array([340.0, 0.0], dtype=np.float32)

    env_a.gu_queue = np.array([1.2e6, 2.4e6, 1.0e6, 1.8e6], dtype=np.float32)
    env_b.gu_queue = np.array([2.0e6, 0.8e6, 1.6e6, 2.2e6], dtype=np.float32)
    env_a.prev_association = np.array([0, 0, 1, 1], dtype=np.int32)
    env_b.prev_association = np.array([0, 1, 1, 1], dtype=np.int32)

    assoc_a = np.array([0, 0, 1, 1], dtype=np.int32)
    assoc_b = np.array([0, 0, 1, 1], dtype=np.int32)
    candidates_a = [[0, 1, 2], [1, 2, 3]]
    candidates_b = [[0, 1, 3], [1, 2, 3]]
    bw_alloc_a = np.array([[0.7, 0.3, 0.0, 0.0], [0.0, 0.0, 0.375, 0.625]], dtype=np.float32)
    bw_alloc_b = np.array([[0.8, 0.2, 0.0, 0.0], [0.0, 0.0, 0.6, 0.4]], dtype=np.float32)

    rates_a, _ = env_a._compute_access_rates(assoc_a, candidates_a, bw_alloc_a, record_exec=True)
    rates_b, _ = env_b._compute_access_rates(assoc_b, candidates_b, bw_alloc_b, record_exec=True)

    candidate_indices = np.array(
        [
            [[0, 1, 2], [1, 2, 3]],
            [[0, 1, 3], [1, 2, 3]],
        ],
        dtype=np.int64,
    )
    candidate_mask = np.ones((2, cfg.num_uav, cfg.users_obs_max), dtype=bool)
    result = _apply_batched_access_rate_torch(
        gain_matrix=np.stack(
            [
                env_a._compute_access_link_gain_matrix(),
                env_b._compute_access_link_gain_matrix(),
            ],
            axis=0,
        ),
        associations=np.stack([assoc_a, assoc_b], axis=0),
        candidate_indices=candidate_indices,
        candidate_mask=candidate_mask,
        bw_action_matrix=np.stack([bw_alloc_a, bw_alloc_b], axis=0),
        gu_queue_before=np.stack([env_a.gu_queue, env_b.gu_queue], axis=0),
        prev_association=np.stack([env_a.prev_association, env_b.prev_association], axis=0),
        cfg=cfg,
        device=torch.device("cpu"),
    )

    np.testing.assert_allclose(result["rates"][0], rates_a, rtol=5e-6, atol=64.0)
    np.testing.assert_allclose(result["rates"][1], rates_b, rtol=5e-6, atol=64.0)
    np.testing.assert_allclose(result["last_exec_bw_alloc"][0], env_a.last_exec_bw_alloc, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result["last_exec_bw_alloc"][1], env_b.last_exec_bw_alloc, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        result["last_bw_align"],
        np.array([env_a.last_bw_align, env_b.last_bw_align], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_batched_access_rate_rejects_invalid_full_g_bw_action():
    cfg = SaginConfig(num_uav=2, num_gu=4, num_sat=1, users_obs_max=3, sats_obs_max=1)
    cfg.enable_bw_action = True
    cfg.interference_enabled = True
    assoc = np.array([[0, 0, 1, 1]], dtype=np.int64)
    gain = np.ones((1, cfg.num_gu, cfg.num_uav), dtype=np.float32)
    candidate_indices = np.array([[[0, 1, 2], [1, 2, 3]]], dtype=np.int64)
    candidate_mask = np.ones((1, cfg.num_uav, cfg.users_obs_max), dtype=bool)
    valid = np.array([[[0.6, 0.4, 0.0, 0.0], [0.0, 0.0, 0.25, 0.75]]], dtype=np.float32)
    invalid_mass = valid.copy()
    invalid_mass[0, 0, 2] = 1.0e-3
    with pytest.raises((RuntimeError, AssertionError), match="invalid GU"):
        _apply_batched_access_rate_torch(
            gain_matrix=gain,
            associations=assoc,
            candidate_indices=candidate_indices,
            candidate_mask=candidate_mask,
            bw_action_matrix=invalid_mass,
            gu_queue_before=np.ones((1, cfg.num_gu), dtype=np.float32),
            prev_association=assoc,
            cfg=cfg,
            device=torch.device("cpu"),
        )

    bad_sum = valid.copy()
    bad_sum[0, 0, 0] = 0.5
    with pytest.raises((RuntimeError, AssertionError), match="simplex sum"):
        _apply_batched_access_rate_torch(
            gain_matrix=gain,
            associations=assoc,
            candidate_indices=candidate_indices,
            candidate_mask=candidate_mask,
            bw_action_matrix=bad_sum,
            gu_queue_before=np.ones((1, cfg.num_gu), dtype=np.float32),
            prev_association=assoc,
            cfg=cfg,
            device=torch.device("cpu"),
        )


def test_batched_close_risk_and_danger_torch_matches_env_reference():
    cfg = SaginConfig(
        num_uav=2,
        num_gu=2,
        num_sat=3,
        users_obs_max=2,
        sats_obs_max=3,
        nbrs_obs_max=1,
        avoidance_enabled=False,
        avoidance_alert_factor=2.0,
        avoidance_prealert_factor=6.0,
        avoidance_prealert_closing_speed=5.0,
        close_risk_enabled=True,
        danger_imitation_enabled=True,
        danger_imitation_trigger_mode="risk_or_intervention",
        danger_imitation_close_risk_thresh=0.05,
        danger_imitation_intervention_thresh=0.05,
    )
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed)
    env.uav_pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    env.uav_pos[1] = np.array([200.0, 100.0], dtype=np.float32)
    env.uav_vel[0] = np.array([15.0, 0.0], dtype=np.float32)
    env.uav_vel[1] = np.array([-15.0, 0.0], dtype=np.float32)
    env.last_policy_accel[:] = 0.0
    env.last_exec_accel[:] = 0.0
    env.last_exec_accel[0, 0] = 1.0
    env._compute_reward()

    outputs = _apply_batched_close_risk_and_danger_torch(
        uav_pos=np.asarray(env.uav_pos, dtype=np.float32)[None, ...],
        uav_vel=np.asarray(env.uav_vel, dtype=np.float32)[None, ...],
        last_exec_accel=np.asarray(env.last_exec_accel, dtype=np.float32)[None, ...],
        last_policy_accel=np.asarray(env.last_policy_accel, dtype=np.float32)[None, ...],
        cfg=cfg,
        device=torch.device("cpu"),
    )

    np.testing.assert_allclose(outputs["intervention_norm_uav"][0], env.last_intervention_norm_uav, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(outputs["close_risk_uav"][0], env.last_close_risk_uav, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(outputs["danger_imitation_mask"][0], env.last_danger_imitation_mask, rtol=1e-6, atol=1e-6)
    assert abs(float(outputs["intervention_norm"][0]) - float(env.last_reward_parts["intervention_norm"])) < 1e-6
    assert abs(float(outputs["intervention_rate"][0]) - float(env.last_reward_parts["intervention_rate"])) < 1e-6
    assert abs(float(outputs["intervention_norm_top1"][0]) - float(env.last_reward_parts["intervention_norm_top1"])) < 1e-6
    assert abs(float(outputs["danger_imitation_active_rate"][0]) - float(env.last_reward_parts["danger_imitation_active_rate"])) < 1e-6
    assert abs(float(outputs["close_risk"][0]) - float(env.last_reward_parts["close_risk"])) < 1e-6
    assert bool(outputs["collision"][0]) is False


def test_sat_observation_and_rank_use_projected_backhaul_bandwidth():
    cfg = SaginConfig(num_uav=1, num_gu=1, num_sat=2, users_obs_max=1, sats_obs_max=2)
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed)

    sat_pos, sat_vel = env._get_orbit_states()
    env.last_sat_selection = [[]]
    env.last_sat_connection_counts = np.array([2.0, 0.0], dtype=np.float32)

    visible = [[0]]
    env._cache_sat_obs(sat_pos, sat_vel, visible)
    rank = env._sat_candidate_rank_data(0, np.array([0], dtype=np.int32), sat_pos)

    projected_bw = float(cfg.b_backhaul_per_sat) / 3.0
    gain = _backhaul_gain(cfg, sat_pos[0], env._uav_ecef(0))
    expected_se = float(
        channel.spectral_efficiency(
        channel.snr_linear(
            float(cfg.uav_tx_power),
            np.array([gain], dtype=np.float64),
            float(cfg.noise_density),
            projected_bw,
            noise_figure_db=float(cfg.backhaul_noise_figure_db),
        )
        )[0]
    )

    assert env._cached_sat_obs[0, 0, 10] == np.float32(1.0 / 3.0)
    np.testing.assert_allclose(env._cached_sat_obs[0, 0, 7], np.float32(expected_se), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(rank["spectral_efficiency"][0], np.float32(expected_se), rtol=1e-5, atol=1e-5)


def test_backhaul_matrix_fast_path_matches_list_path_and_side_effects():
    cfg = SaginConfig(num_uav=2, num_gu=1, num_sat=4, users_obs_max=1, sats_obs_max=3, sat_num_select=2)
    cfg.energy_enabled = True

    list_env = SaginParallelEnv(cfg)
    matrix_env = SaginParallelEnv(cfg)
    list_env.reset(seed=cfg.seed)
    matrix_env.reset(seed=cfg.seed)

    sat_pos, sat_vel = list_env._get_orbit_states()
    matrix_env.uav_pos = list_env.uav_pos.copy()
    matrix_env.uav_vel = list_env.uav_vel.copy()
    matrix_env.uav_energy = list_env.uav_energy.copy()
    matrix_env.sat_queue = list_env.sat_queue.copy()

    selections = [[0, 1], [1, 2]]
    selection_matrix = np.array([[0, 1], [1, 2]], dtype=np.int64)

    list_rate, list_counts = list_env._compute_backhaul_rates(sat_pos, sat_vel, selections)
    matrix_rate, matrix_counts = matrix_env._compute_backhaul_rates(sat_pos, sat_vel, selection_matrix)

    np.testing.assert_allclose(matrix_rate, list_rate, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(matrix_counts, list_counts)

    list_env._update_energy(selections)
    matrix_env._update_energy(selection_matrix)
    np.testing.assert_allclose(matrix_env.last_energy_cost, list_env.last_energy_cost, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(matrix_env.uav_energy, list_env.uav_energy, rtol=1e-6, atol=1e-6)

    list_env._update_connected_sat_link_stats(sat_pos, selections)
    matrix_env._update_connected_sat_link_stats(sat_pos, selection_matrix)
    assert matrix_env.last_connected_sat_count == list_env.last_connected_sat_count
    np.testing.assert_allclose(
        np.array(
            [
                matrix_env.last_connected_sat_dist_mean,
                matrix_env.last_connected_sat_dist_p95,
                matrix_env.last_connected_sat_elevation_deg_mean,
                matrix_env.last_connected_sat_elevation_deg_min,
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                list_env.last_connected_sat_dist_mean,
                list_env.last_connected_sat_dist_p95,
                list_env.last_connected_sat_elevation_deg_mean,
                list_env.last_connected_sat_elevation_deg_min,
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_backhaul_loss_factor_batch_matches_env_helper():
    cfg = SaginConfig(num_uav=2, num_gu=1, num_sat=4, users_obs_max=1, sats_obs_max=3, sat_num_select=2)
    cfg.atm_loss_enabled = True
    cfg.rain_loss_enabled = True

    env_a = SaginParallelEnv(cfg)
    env_b = SaginParallelEnv(cfg)
    env_a.reset(seed=cfg.seed)
    env_b.reset(seed=cfg.seed + 1)

    sat_pos_a, _ = env_a._get_orbit_states()
    sat_pos_b, _ = env_b._get_orbit_states()
    elev_a = env_a._get_elevation_matrix(sat_pos_a)
    elev_b = env_b._get_elevation_matrix(sat_pos_b)

    batch = _backhaul_loss_factor_batch(
        cfg=cfg,
        elevation_batch=np.stack([elev_a, elev_b], axis=0),
    )

    np.testing.assert_allclose(batch[0], env_a._backhaul_loss_factor(elev_a), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(batch[1], env_b._backhaul_loss_factor(elev_b), rtol=1e-6, atol=1e-6)


def test_weighted_workload_costs_batch_and_sat_overlap_match_env_helpers():
    cfg = SaginConfig(num_uav=2, num_gu=4, num_sat=5, users_obs_max=3, sats_obs_max=3, sat_num_select=2)

    env_a = SaginParallelEnv(cfg)
    env_b = SaginParallelEnv(cfg)
    env_a.reset(seed=cfg.seed)
    env_b.reset(seed=cfg.seed + 1)

    env_a.last_association = np.array([0, 0, 1, 1], dtype=np.int32)
    env_b.last_association = np.array([0, 1, 1, -1], dtype=np.int32)
    env_a.last_sat_selection = [[0, 2], [2, 4]]
    env_b.last_sat_selection = [[1, 1], [0, -1]]
    env_a.bw_weighted_workload_acc_ema_vec = np.array([1.2, 2.0, 1.6, 1.1], dtype=np.float32)
    env_b.bw_weighted_workload_acc_ema_vec = np.array([0.9, 1.4, 2.3, 1.7], dtype=np.float32)
    env_a.bw_weighted_workload_rel_ema_vec = np.array([3.2, 2.1], dtype=np.float32)
    env_b.bw_weighted_workload_rel_ema_vec = np.array([2.7, 1.9], dtype=np.float32)
    env_a.bw_weighted_workload_sat_ema_vec = np.array([1.0, 1.3, 2.4, 1.7, 1.1], dtype=np.float32)
    env_b.bw_weighted_workload_sat_ema_vec = np.array([1.5, 2.1, 1.8, 1.2, 2.6], dtype=np.float32)

    associations = np.stack([env_a.last_association, env_b.last_association], axis=0)
    sat_selection_matrix = np.stack(
        [
            env_a._sat_selection_matrix(env_a.last_sat_selection),
            env_b._sat_selection_matrix(env_b.last_sat_selection),
        ],
        axis=0,
    )
    gu_costs, uav_costs, sat_costs = _bw_weighted_workload_device_costs_batch(
        [env_a, env_b],
        associations=associations,
        sat_selection_matrix=sat_selection_matrix,
    )
    overlap = _sat_overlap_eval_batch(sat_selection_matrix, num_sat=cfg.num_sat)

    ref_costs_a = env_a._bw_weighted_workload_device_costs()
    ref_costs_b = env_b._bw_weighted_workload_device_costs()
    np.testing.assert_allclose(gu_costs[0], ref_costs_a[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(uav_costs[0], ref_costs_a[1], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(sat_costs[0], ref_costs_a[2], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gu_costs[1], ref_costs_b[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(uav_costs[1], ref_costs_b[1], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(sat_costs[1], ref_costs_b[2], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        overlap,
        np.array([env_a._compute_sat_overlap_eval(), env_b._compute_sat_overlap_eval()], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_batched_bw_post_stats_torch_matches_env_reference():
    cfg = SaginConfig(num_uav=2, num_gu=4, num_sat=5, users_obs_max=3, sats_obs_max=3, sat_num_select=2)

    env_a = SaginParallelEnv(cfg)
    env_b = SaginParallelEnv(cfg)
    env_a.reset(seed=cfg.seed)
    env_b.reset(seed=cfg.seed + 1)

    for env in (env_a, env_b):
        env._refresh_uav_cache()

    sat_pos_a, _ = env_a._get_orbit_states()
    sat_pos_b, _ = env_b._get_orbit_states()
    sat_selection_a = np.array([[0, 2], [1, -1]], dtype=np.int64)
    sat_selection_b = np.array([[2, 4], [0, 3]], dtype=np.int64)

    env_a.gu_queue = np.array([1.1e6, 2.5e6, 0.9e6, 1.4e6], dtype=np.float32)
    env_b.gu_queue = np.array([2.0e6, 1.2e6, 1.8e6, 0.6e6], dtype=np.float32)
    env_a.last_gu_outflow = np.array([1.2e5, 2.8e5, 1.0e5, 1.5e5], dtype=np.float32)
    env_b.last_gu_outflow = np.array([2.1e5, 1.3e5, 2.4e5, 0.8e5], dtype=np.float32)
    env_a.uav_queue = np.array([3.0e6, 1.5e6], dtype=np.float32)
    env_b.uav_queue = np.array([1.0e6, 3.8e6], dtype=np.float32)
    env_a.sat_queue = np.array([2.5e6, 1.2e6, 3.4e6, 0.8e6, 0.3e6], dtype=np.float32)
    env_b.sat_queue = np.array([0.9e6, 2.6e6, 1.7e6, 3.2e6, 1.1e6], dtype=np.float32)
    env_a.last_sat_connection_counts = np.array([1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    env_b.last_sat_connection_counts = np.array([1.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
    env_a.last_gu_service_gap = np.array([0.5, 2.2, 3.8, 1.0], dtype=np.float32)
    env_b.last_gu_service_gap = np.array([2.8, 0.4, 4.1, 1.6], dtype=np.float32)

    env_a.last_arrival_rate = float(cfg.task_arrival_rate) * 1.1
    env_b.last_arrival_rate = float(cfg.task_arrival_rate) * 0.8
    next_arrival_rates_a = np.asarray(env_a._current_task_arrival_rates(env_a.last_arrival_rate), dtype=np.float32)
    next_arrival_rates_b = np.asarray(env_b._current_task_arrival_rates(env_b.last_arrival_rate), dtype=np.float32)

    env_a._update_connected_sat_link_stats(sat_pos_a, sat_selection_a)
    env_b._update_connected_sat_link_stats(sat_pos_b, sat_selection_b)
    env_a._refresh_bw_proxy_features()
    env_b._refresh_bw_proxy_features()

    result = _apply_batched_bw_post_stats_torch(
        sat_pos=np.stack([sat_pos_a, sat_pos_b], axis=0),
        sat_selection_matrix=np.stack(
            [
                env_a._sat_selection_matrix(sat_selection_a),
                env_b._sat_selection_matrix(sat_selection_b),
            ],
            axis=0,
        ),
        uav_ecef=np.stack(
            [
                np.asarray(env_a._cached_uav_ecef, dtype=np.float32),
                np.asarray(env_b._cached_uav_ecef, dtype=np.float32),
            ],
            axis=0,
        ),
        gu_queue=np.stack([env_a.gu_queue, env_b.gu_queue], axis=0),
        last_gu_outflow=np.stack([env_a.last_gu_outflow, env_b.last_gu_outflow], axis=0),
        next_arrival_rates=np.stack([next_arrival_rates_a, next_arrival_rates_b], axis=0),
        uav_queue=np.stack([env_a.uav_queue, env_b.uav_queue], axis=0),
        sat_queue=np.stack([env_a.sat_queue, env_b.sat_queue], axis=0),
        last_sat_connection_counts=np.stack([env_a.last_sat_connection_counts, env_b.last_sat_connection_counts], axis=0),
        last_gu_service_gap=np.stack([env_a.last_gu_service_gap, env_b.last_gu_service_gap], axis=0),
        cfg=cfg,
        uav_orbit_radius=float(env_a._uav_orbit_radius),
        uav_orbit_radius_sq=float(env_a._uav_orbit_radius_sq),
        sat_orbit_radius_sq=float(env_a._sat_orbit_radius_sq),
        device=torch.device("cpu"),
    )

    np.testing.assert_allclose(
        result["last_connected_sat_count"],
        np.array([env_a.last_connected_sat_count, env_b.last_connected_sat_count], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result["last_connected_sat_dist_mean"],
        np.array([env_a.last_connected_sat_dist_mean, env_b.last_connected_sat_dist_mean], dtype=np.float32),
        rtol=1e-6,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        result["last_connected_sat_dist_p95"],
        np.array([env_a.last_connected_sat_dist_p95, env_b.last_connected_sat_dist_p95], dtype=np.float32),
        rtol=1e-6,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        result["last_connected_sat_elevation_deg_mean"],
        np.array([env_a.last_connected_sat_elevation_deg_mean, env_b.last_connected_sat_elevation_deg_mean], dtype=np.float32),
        rtol=1e-6,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        result["last_connected_sat_elevation_deg_min"],
        np.array([env_a.last_connected_sat_elevation_deg_min, env_b.last_connected_sat_elevation_deg_min], dtype=np.float32),
        rtol=1e-6,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        result["last_gu_urgency_risk"],
        np.stack([env_a.last_gu_urgency_risk, env_b.last_gu_urgency_risk], axis=0),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result["last_gu_downstream_pressure"],
        np.stack([env_a.last_gu_downstream_pressure, env_b.last_gu_downstream_pressure], axis=0),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result["last_gu_service_gap_risk"],
        np.stack([env_a.last_gu_service_gap_risk, env_b.last_gu_service_gap_risk], axis=0),
        rtol=1e-6,
        atol=1e-6,
    )


def test_bw_transition_core_matrix_fast_path_matches_dict_list_path():
    cfg = SaginConfig(num_uav=2, num_gu=4, num_sat=6, users_obs_max=3, sats_obs_max=4, sat_num_select=2)
    cfg.enable_bw_action = True
    cfg.interference_enabled = True

    dict_env = SaginParallelEnv(cfg)
    matrix_env = SaginParallelEnv(cfg)
    dict_env.reset(seed=cfg.seed)
    matrix_env.reset(seed=cfg.seed)

    dict_env.uav_pos[0] = np.array([0.0, 0.0], dtype=np.float32)
    dict_env.uav_pos[1] = np.array([300.0, 0.0], dtype=np.float32)
    dict_env.gu_pos[0] = np.array([0.0, 0.0], dtype=np.float32)
    dict_env.gu_pos[1] = np.array([40.0, 0.0], dtype=np.float32)
    dict_env.gu_pos[2] = np.array([300.0, 0.0], dtype=np.float32)
    dict_env.gu_pos[3] = np.array([340.0, 0.0], dtype=np.float32)
    matrix_env.uav_pos = dict_env.uav_pos.copy()
    matrix_env.gu_pos = dict_env.gu_pos.copy()
    matrix_env.uav_vel = dict_env.uav_vel.copy()
    matrix_env.uav_energy = dict_env.uav_energy.copy()
    matrix_env.gu_queue = dict_env.gu_queue.copy()
    matrix_env.uav_queue = dict_env.uav_queue.copy()
    matrix_env.sat_queue = dict_env.sat_queue.copy()
    matrix_env.last_association = dict_env.last_association.copy()
    matrix_env.prev_association = dict_env.prev_association.copy()

    assoc = np.array([0, 0, 1, 1], dtype=np.int32)
    candidates = [[0, 1, 2], [1, 2, 3]]
    bw_alloc = np.array(
        [
            [0.7, 0.3, 0.0, 0.0],
            [0.0, 0.0, 0.375, 0.625],
        ],
        dtype=np.float32,
    )
    actions = {
        dict_env.agents[0]: {"bw_alloc": bw_alloc[0]},
        dict_env.agents[1]: {"bw_alloc": bw_alloc[1]},
    }
    sat_pos, sat_vel = dict_env._get_orbit_states()
    selection_list = [[0, 1], [1, 2]]
    selection_matrix = np.array([[0, 1], [1, 2]], dtype=np.int64)

    dict_result = dict_env._apply_bw_transition_core(
        assoc,
        candidates,
        actions,
        selection_list,
        sat_pos,
        sat_vel,
    )
    matrix_result = matrix_env._apply_bw_transition_core(
        assoc,
        candidates,
        bw_alloc,
        selection_matrix,
        sat_pos,
        sat_vel,
    )

    np.testing.assert_allclose(matrix_result.gu_queue_before, dict_result.gu_queue_before, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(matrix_result.uav_queue_before, dict_result.uav_queue_before, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(matrix_result.sat_queue_before, dict_result.sat_queue_before, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(matrix_result.realized_arrival, dict_result.realized_arrival, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(matrix_result.rate_matrix, dict_result.rate_matrix, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(matrix_result.sat_loads, dict_result.sat_loads, rtol=1e-6, atol=1e-6)

    np.testing.assert_allclose(matrix_env.gu_queue, dict_env.gu_queue, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(matrix_env.uav_queue, dict_env.uav_queue, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(matrix_env.sat_queue, dict_env.sat_queue, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(matrix_env.last_exec_bw_alloc, dict_env.last_exec_bw_alloc, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(matrix_env.last_sat_connection_counts, dict_env.last_sat_connection_counts, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(matrix_env.last_sat_incoming, dict_env.last_sat_incoming, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(matrix_env.last_sat_processed, dict_env.last_sat_processed, rtol=1e-6, atol=1e-6)
    assert matrix_env.last_sat_selection == dict_env.last_sat_selection
