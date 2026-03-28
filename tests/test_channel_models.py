from __future__ import annotations

import math

import numpy as np

from sagin_marl.env import channel
from sagin_marl.env.config import SaginConfig
from sagin_marl.env.sagin_env import SaginParallelEnv


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
            "bw_alloc": np.array([1.0, 3.0], dtype=np.float32),
        }
    }

    rates, eta = env._compute_access_rates(assoc, candidates, actions, record_exec=False)

    gain = _access_gain(cfg, env.gu_pos[0], env.uav_pos[0])
    beta = np.array([0.25, 0.75], dtype=np.float64)
    eff_bw = beta * float(cfg.b_acc)
    expected_rates = eff_bw * channel.spectral_efficiency(
        channel.snr_linear(float(cfg.gu_tx_power), np.full((2,), gain, dtype=np.float64), float(cfg.noise_density), eff_bw)
    )
    expected_eta = float(
        channel.spectral_efficiency(
            channel.snr_linear(float(cfg.gu_tx_power), np.array([gain], dtype=np.float64), float(cfg.noise_density), float(cfg.b_acc))
        )[0]
    )

    np.testing.assert_allclose(rates, expected_rates.astype(np.float32), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(eta[0, :2], np.full((2,), expected_eta, dtype=np.float32), rtol=1e-5, atol=1e-5)


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
            "bw_alloc": np.array([1.0, 3.0], dtype=np.float32),
        },
        env.agents[1]: {
            "bw_alloc": np.array([1.0, 0.0], dtype=np.float32),
        },
    }

    rates, eta = env._compute_access_rates(assoc, candidates, actions, record_exec=False)

    gain_sig = _access_gain(cfg, env.gu_pos[0], env.uav_pos[0])
    gain_int = _access_gain(cfg, env.gu_pos[2], env.uav_pos[0])
    beta = np.array([0.25, 0.75], dtype=np.float64)
    eff_bw = beta * float(cfg.b_acc)
    interference = float(cfg.gu_tx_power) * gain_int
    expected_rates = eff_bw * channel.spectral_efficiency(
        channel.snr_linear(
            float(cfg.gu_tx_power),
            np.full((2,), gain_sig, dtype=np.float64),
            float(cfg.noise_density),
            eff_bw,
            beta * interference,
        )
    )
    expected_eta = float(
        channel.spectral_efficiency(
            channel.snr_linear(
                float(cfg.gu_tx_power),
                np.array([gain_sig], dtype=np.float64),
                float(cfg.noise_density),
                float(cfg.b_acc),
                interference,
            )
        )[0]
    )

    np.testing.assert_allclose(rates[:2], expected_rates.astype(np.float32), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(eta[0, :2], np.full((2,), expected_eta, dtype=np.float32), rtol=1e-5, atol=1e-5)


def test_sat_observation_and_rank_use_projected_backhaul_bandwidth():
    cfg = SaginConfig(num_uav=1, num_gu=0, num_sat=2, users_obs_max=1, sats_obs_max=2)
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed)

    sat_pos, sat_vel = env._get_orbit_states()
    env.last_sat_selection = [[]]
    env.last_sat_connection_counts = np.array([2.0, 0.0], dtype=np.float32)

    visible = [[0]]
    env._cache_sat_obs(sat_pos, sat_vel, visible)
    rank = env._sat_candidate_rank_data(0, np.array([0], dtype=np.int32), sat_pos)

    projected_bw = float(cfg.b_sat_total) / 3.0
    gain = _backhaul_gain(cfg, sat_pos[0], env._uav_ecef(0))
    expected_se = float(
        channel.spectral_efficiency(
            channel.snr_linear(
                float(cfg.uav_tx_power),
                np.array([gain], dtype=np.float64),
                float(cfg.noise_density),
                projected_bw,
            )
        )[0]
    )

    assert env._cached_sat_obs[0, 0, 10] == np.float32(1.0 / 3.0)
    np.testing.assert_allclose(env._cached_sat_obs[0, 0, 7], np.float32(expected_se), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(rank["spectral_efficiency"][0], np.float32(expected_se), rtol=1e-5, atol=1e-5)
