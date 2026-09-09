from __future__ import annotations

import numpy as np
import torch

from sagin_marl.env.config import SaginConfig
from sagin_marl.rl.baselines import (
    cluster_center_accel_policy_batch,
    cluster_center_accel_policy,
    cluster_center_queue_aware_policy_batch,
    cluster_center_queue_aware_policy,
    centroid_accel_policy,
    lyapunov_queue_aware_policy_step,
    queue_aware_policy_batch,
    queue_aware_policy,
    random_accel_policy,
    topology_dpp_policy_step,
    zero_accel_policy,
)
from sagin_marl.rl.structured_eval import _fixed_policy_exec_sources
from sagin_marl.rl.policy import OWN_OBS_DIM, SAT_OBS_DIM


def test_fixed_policy_exec_sources_include_current_dpp_aliases():
    assert _fixed_policy_exec_sources("lyapunov") == ("lyapunov", "lyapunov", "lyapunov")
    assert _fixed_policy_exec_sources("maxweight_lyapunov") == ("lyapunov", "lyapunov", "lyapunov")
    assert _fixed_policy_exec_sources("lyapunov_maxweight") == ("lyapunov", "lyapunov", "lyapunov")
    assert _fixed_policy_exec_sources("dpp_no_mobility") == ("zero", "lyapunov", "lyapunov")
    assert _fixed_policy_exec_sources("dpp_equal_bw") == ("lyapunov", "lyapunov", "uniform")
    assert _fixed_policy_exec_sources("dpp_greedy_sat") == ("lyapunov", "queue_aware", "lyapunov")
    assert _fixed_policy_exec_sources("topology_dpp") is None
    assert _fixed_policy_exec_sources("dpp_resource_hybrid") is None
    assert _fixed_policy_exec_sources("topology_dpp_resource") is None


def _pack_obs_many(obs_many):
    sample_keys = tuple(obs_many[0][0].keys())
    return {
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


def test_zero_accel_policy_shape_dtype():
    actions = zero_accel_policy(3)
    assert actions.shape == (3, 2)
    assert actions.dtype == np.float32
    assert np.all(actions == 0.0)


def test_random_accel_policy_shape_dtype_and_range():
    actions = random_accel_policy(4, rng=np.random.default_rng(123))
    assert actions.shape == (4, 2)
    assert actions.dtype == np.float32
    assert np.max(actions) <= 1.0 + 1e-6
    assert np.min(actions) >= -1.0 - 1e-6


def test_centroid_accel_policy_points_to_users():
    obs = {
        "own": np.zeros((OWN_OBS_DIM,), dtype=np.float32),
        "users": np.zeros((3, 5), dtype=np.float32),
        "users_mask": np.zeros((3,), dtype=np.float32),
        "sats": np.zeros((1, SAT_OBS_DIM), dtype=np.float32),
        "sats_mask": np.zeros((1,), dtype=np.float32),
        "nbrs": np.zeros((1, 4), dtype=np.float32),
        "nbrs_mask": np.zeros((1,), dtype=np.float32),
    }
    obs["users_mask"][0] = 1.0
    obs["users"][0, 0:2] = np.array([0.4, -0.2], dtype=np.float32)
    obs["users"][0, 2] = 1.0

    accel = centroid_accel_policy([obs], gain=2.0, queue_weighted=True)
    assert accel.shape == (1, 2)
    assert accel.dtype == np.float32
    assert accel[0, 0] > 0.0
    assert accel[0, 1] < 0.0
    assert np.max(np.abs(accel)) <= 1.0 + 1e-6


def test_cluster_center_accel_policy_assigns_by_cluster_priority_and_uav_position():
    cfg = SaginConfig(num_uav=2, map_size=1000.0)
    cfg.baseline_cluster_cruise_speed = 10.0
    cfg.baseline_cluster_slow_radius = 120.0
    cfg.baseline_cluster_stop_radius = 10.0
    cfg.baseline_cluster_speed_tol = 0.5

    base_obs = {
        "users": np.zeros((cfg.users_obs_max, 5), dtype=np.float32),
        "users_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "sats": np.zeros((cfg.sats_obs_max, SAT_OBS_DIM), dtype=np.float32),
        "sats_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "nbrs": np.zeros((cfg.nbrs_obs_max, 4), dtype=np.float32),
        "nbrs_mask": np.zeros((cfg.nbrs_obs_max,), dtype=np.float32),
    }
    obs_left = {
        **base_obs,
        "own": np.array([0.05, 0.10, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    }
    obs_right = {
        **base_obs,
        "own": np.array([0.95, 0.10, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    }

    centers = np.array(
        [
            [900.0, 100.0],
            [100.0, 100.0],
            [500.0, 900.0],
        ],
        dtype=np.float32,
    )
    counts = np.array([9, 8, 1], dtype=np.int32)

    accel = cluster_center_accel_policy([obs_left, obs_right], cfg, centers, counts)
    assert accel.shape == (2, 2)
    assert accel.dtype == np.float32
    assert accel[0, 0] > 0.0
    assert accel[1, 0] < 0.0
    assert np.max(np.abs(accel)) <= 1.0 + 1e-6


def test_cluster_center_accel_policy_brakes_at_cluster_center():
    cfg = SaginConfig(num_uav=1, map_size=1000.0)
    cfg.baseline_cluster_stop_radius = 20.0
    cfg.baseline_cluster_speed_tol = 0.5

    obs = {
        "own": np.array([0.50, 0.50, 0.40, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "users": np.zeros((cfg.users_obs_max, 5), dtype=np.float32),
        "users_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "sats": np.zeros((cfg.sats_obs_max, SAT_OBS_DIM), dtype=np.float32),
        "sats_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "nbrs": np.zeros((cfg.nbrs_obs_max, 4), dtype=np.float32),
        "nbrs_mask": np.zeros((cfg.nbrs_obs_max,), dtype=np.float32),
    }

    centers = np.array([[500.0, 500.0]], dtype=np.float32)
    counts = np.array([5], dtype=np.int32)

    accel = cluster_center_accel_policy([obs], cfg, centers, counts)
    assert accel.shape == (1, 2)
    assert accel[0, 0] < 0.0
    assert abs(float(accel[0, 1])) <= 1e-6


def test_queue_aware_policy_shapes():
    cfg = SaginConfig()
    cfg.enable_bw_action = True
    cfg.fixed_satellite_strategy = False
    obs = {
        "own": np.zeros((OWN_OBS_DIM,), dtype=np.float32),
        "users": np.zeros((cfg.users_obs_max, 5), dtype=np.float32),
        "users_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "bw_valid_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "sats": np.zeros((cfg.sats_obs_max, SAT_OBS_DIM), dtype=np.float32),
        "sats_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "sat_valid_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "nbrs": np.zeros((cfg.nbrs_obs_max, 4), dtype=np.float32),
        "nbrs_mask": np.zeros((cfg.nbrs_obs_max,), dtype=np.float32),
    }
    obs["users_mask"][0] = 1.0
    obs["bw_valid_mask"][0] = 1.0
    obs["users"][0, 0:2] = np.array([0.2, -0.1], dtype=np.float32)
    obs["users"][0, 2] = 0.5
    obs["users"][0, 3] = 1.0
    obs["users"][0, 4] = 1.0
    obs["sats_mask"][0] = 1.0
    obs["sat_valid_mask"][0] = 1.0
    obs["sats"][0, 7] = 2.0
    obs["sats"][0, 8] = 0.1

    accel, bw_alloc, sat_select_mask = queue_aware_policy([obs, obs], cfg)
    assert accel.shape == (2, 2)
    assert bw_alloc.shape == (2, cfg.num_gu)
    assert sat_select_mask.shape == (2, cfg.sats_obs_max)
    assert accel.dtype == np.float32
    assert bw_alloc.dtype == np.float32
    assert sat_select_mask.dtype == np.float32
    assert np.all(np.isfinite(accel))
    assert np.all(np.isfinite(bw_alloc))
    assert np.all(np.isfinite(sat_select_mask))
    assert np.max(np.abs(accel)) <= 1.0 + 1e-6
    assert np.all(bw_alloc >= -1e-6)
    assert np.all(sat_select_mask >= -1e-6)
    assert np.allclose(np.sum(bw_alloc, axis=1), 1.0, atol=1e-6)
    assert np.all(np.sum(sat_select_mask, axis=1) <= cfg.N_RF + 1e-6)


def test_queue_aware_policy_batch_matches_per_env_numpy_outputs():
    cfg = SaginConfig(num_uav=1, users_obs_max=3, sats_obs_max=2, nbrs_obs_max=1)
    cfg.enable_bw_action = True
    cfg.fixed_satellite_strategy = False

    obs_a = {
        "own": np.array([0.10, 0.20, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "users": np.zeros((cfg.users_obs_max, 5), dtype=np.float32),
        "users_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "bw_valid_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "sats": np.zeros((cfg.sats_obs_max, SAT_OBS_DIM), dtype=np.float32),
        "sats_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "sat_valid_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "nbrs": np.zeros((cfg.nbrs_obs_max, 4), dtype=np.float32),
        "nbrs_mask": np.zeros((cfg.nbrs_obs_max,), dtype=np.float32),
    }
    obs_a["users_mask"][0] = 1.0
    obs_a["bw_valid_mask"][0] = 1.0
    obs_a["users"][0, 0:2] = np.array([0.3, -0.2], dtype=np.float32)
    obs_a["users"][0, 2] = 0.6
    obs_a["users"][0, 3] = 1.0
    obs_a["users"][0, 4] = 1.0
    obs_a["sats_mask"][:] = 1.0
    obs_a["sat_valid_mask"][:] = 1.0
    obs_a["sats"][0, 7] = 4.0
    obs_a["sats"][0, 8] = 0.2
    obs_a["sats"][0, 9] = 0.1
    obs_a["sats"][0, 10] = 0.9

    obs_b = {
        "own": np.array([0.70, 0.40, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "users": np.zeros((cfg.users_obs_max, 5), dtype=np.float32),
        "users_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "bw_valid_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "sats": np.zeros((cfg.sats_obs_max, SAT_OBS_DIM), dtype=np.float32),
        "sats_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "sat_valid_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "nbrs": np.zeros((cfg.nbrs_obs_max, 4), dtype=np.float32),
        "nbrs_mask": np.zeros((cfg.nbrs_obs_max,), dtype=np.float32),
    }
    obs_b["users_mask"][1] = 1.0
    obs_b["bw_valid_mask"][1] = 1.0
    obs_b["users"][1, 0:2] = np.array([-0.4, 0.1], dtype=np.float32)
    obs_b["users"][1, 2] = 0.9
    obs_b["users"][1, 3] = 0.7
    obs_b["users"][1, 4] = 0.0
    obs_b["sats_mask"][:] = 1.0
    obs_b["sat_valid_mask"][:] = 1.0
    obs_b["sats"][0, 7] = 1.0
    obs_b["sats"][0, 8] = 0.8
    obs_b["sats"][0, 9] = 0.8
    obs_b["sats"][0, 10] = 0.2
    obs_b["sats"][1, 7] = 4.5
    obs_b["sats"][1, 8] = 0.1
    obs_b["sats"][1, 9] = 0.1
    obs_b["sats"][1, 10] = 0.95

    obs_many = [[obs_a], [obs_b]]
    obs_batch = _pack_obs_many(obs_many)

    accel_ref = np.stack([queue_aware_policy(obs_list, cfg)[0] for obs_list in obs_many], axis=0)
    bw_ref = np.stack([queue_aware_policy(obs_list, cfg)[1] for obs_list in obs_many], axis=0)
    sat_ref = np.stack([queue_aware_policy(obs_list, cfg)[2] for obs_list in obs_many], axis=0)

    accel_batch, bw_batch, sat_batch = queue_aware_policy_batch(obs_batch, cfg)
    np.testing.assert_allclose(accel_batch, accel_ref, atol=1e-6)
    np.testing.assert_allclose(bw_batch, bw_ref, atol=1e-6)
    np.testing.assert_allclose(sat_batch, sat_ref, atol=1e-6)


def test_queue_aware_policy_batch_preserves_torch_outputs():
    cfg = SaginConfig(num_uav=1, users_obs_max=2, sats_obs_max=2, nbrs_obs_max=1)
    cfg.enable_bw_action = True
    cfg.fixed_satellite_strategy = False
    obs = {
        "own": np.array([0.20, 0.10, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "users": np.zeros((cfg.users_obs_max, 5), dtype=np.float32),
        "users_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "bw_valid_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "sats": np.zeros((cfg.sats_obs_max, SAT_OBS_DIM), dtype=np.float32),
        "sats_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "sat_valid_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "nbrs": np.zeros((cfg.nbrs_obs_max, 4), dtype=np.float32),
        "nbrs_mask": np.zeros((cfg.nbrs_obs_max,), dtype=np.float32),
    }
    obs["users_mask"][0] = 1.0
    obs["bw_valid_mask"][0] = 1.0
    obs["users"][0, 0:2] = np.array([0.2, 0.0], dtype=np.float32)
    obs["users"][0, 2] = 0.8
    obs["users"][0, 3] = 1.1
    obs["users"][0, 4] = 1.0
    obs["sats_mask"][:] = 1.0
    obs["sat_valid_mask"][:] = 1.0
    obs["sats"][1, 7] = 5.0
    obs["sats"][1, 8] = 0.0
    obs["sats"][1, 9] = 0.0
    obs["sats"][1, 10] = 1.0

    obs_batch = {
        key: torch.as_tensor(value[None, None, ...], dtype=torch.float32)
        for key, value in obs.items()
    }
    accel_batch, bw_batch, sat_batch = queue_aware_policy_batch(obs_batch, cfg)

    assert torch.is_tensor(accel_batch)
    assert torch.is_tensor(bw_batch)
    assert torch.is_tensor(sat_batch)
    assert accel_batch.shape == (1, 1, 2)
    assert bw_batch.shape == (1, 1, cfg.num_gu)
    assert sat_batch.shape == (1, 1, cfg.sats_obs_max)


def test_queue_aware_sat_uses_load_bw_and_stay_features():
    cfg = SaginConfig(num_uav=3, sats_obs_max=2)
    cfg.fixed_satellite_strategy = False
    obs = {
        "own": np.zeros((OWN_OBS_DIM,), dtype=np.float32),
        "users": np.zeros((cfg.users_obs_max, 5), dtype=np.float32),
        "users_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "bw_valid_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "sats": np.zeros((cfg.sats_obs_max, SAT_OBS_DIM), dtype=np.float32),
        "sats_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "sat_valid_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "nbrs": np.zeros((cfg.nbrs_obs_max, 4), dtype=np.float32),
        "nbrs_mask": np.zeros((cfg.nbrs_obs_max,), dtype=np.float32),
    }
    obs["sats_mask"][:] = 1.0
    obs["sat_valid_mask"][:] = 1.0
    obs["sats"][0, 7] = 5.0
    obs["sats"][0, 8] = 0.5
    obs["sats"][0, 9] = 2.0 / 3.0
    obs["sats"][0, 10] = 0.5
    obs["sats"][0, 11] = 0.0

    obs["sats"][1, 7] = 4.4
    obs["sats"][1, 8] = 0.0
    obs["sats"][1, 9] = 0.0
    obs["sats"][1, 10] = 1.0
    obs["sats"][1, 11] = 1.0

    _, _, sat_select_mask = queue_aware_policy([obs], cfg)
    assert sat_select_mask.shape == (1, cfg.sats_obs_max)
    assert sat_select_mask[0, 1] > sat_select_mask[0, 0]


def test_queue_aware_sat_prefers_current_sat_within_switch_margin():
    cfg = SaginConfig(num_uav=3, sats_obs_max=2, N_RF=1)
    cfg.fixed_satellite_strategy = False
    cfg.baseline_sat_switch_margin = 0.2
    obs = {
        "own": np.zeros((OWN_OBS_DIM,), dtype=np.float32),
        "users": np.zeros((cfg.users_obs_max, 5), dtype=np.float32),
        "users_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "bw_valid_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "sats": np.zeros((cfg.sats_obs_max, SAT_OBS_DIM), dtype=np.float32),
        "sats_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "sat_valid_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "nbrs": np.zeros((cfg.nbrs_obs_max, 4), dtype=np.float32),
        "nbrs_mask": np.zeros((cfg.nbrs_obs_max,), dtype=np.float32),
    }
    obs["sats_mask"][:] = 1.0
    obs["sat_valid_mask"][:] = 1.0
    obs["sats"][0, 7] = 4.8
    obs["sats"][0, 8] = 0.2
    obs["sats"][0, 9] = 1.0 / 3.0
    obs["sats"][0, 10] = 1.0
    obs["sats"][0, 11] = 1.0

    obs["sats"][1, 7] = 4.9
    obs["sats"][1, 8] = 0.15
    obs["sats"][1, 9] = 1.0 / 3.0
    obs["sats"][1, 10] = 0.5
    obs["sats"][1, 11] = 0.0

    _, _, sat_select_mask = queue_aware_policy([obs], cfg)
    assert sat_select_mask[0, 0] > sat_select_mask[0, 1]


def test_lyapunov_sat_penalizes_high_doppler_when_links_otherwise_match():
    cfg = SaginConfig(num_uav=1, sats_obs_max=2, N_RF=1)
    cfg.enable_bw_action = False
    cfg.fixed_satellite_strategy = False
    obs = {
        "own": np.zeros((OWN_OBS_DIM,), dtype=np.float32),
        "users": np.zeros((cfg.users_obs_max, 5), dtype=np.float32),
        "users_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "bw_valid_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "sats": np.zeros((cfg.sats_obs_max, SAT_OBS_DIM), dtype=np.float32),
        "sats_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "sat_valid_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "nbrs": np.zeros((cfg.nbrs_obs_max, 4), dtype=np.float32),
        "nbrs_mask": np.zeros((cfg.nbrs_obs_max,), dtype=np.float32),
    }
    # Lyapunov SAT scoring is a backpressure term. Give the UAV positive queue
    # pressure so better relay support increases the selected score.
    obs["own"][5] = 1.0
    obs["users_mask"][0] = 1.0
    obs["users"][0, 0:2] = np.array([0.2, 0.0], dtype=np.float32)
    obs["users"][0, 2] = 0.5
    obs["users"][0, 3] = 1.0
    obs["sats_mask"][:] = 1.0
    obs["sat_valid_mask"][:] = 1.0
    obs["sats"][0, 6] = 0.95
    obs["sats"][0, 7] = 4.0
    obs["sats"][0, 8] = 0.2
    obs["sats"][0, 9] = 0.1
    obs["sats"][0, 10] = 0.5
    obs["sats"][1, 6] = 0.05
    obs["sats"][1, 7] = 4.0
    obs["sats"][1, 8] = 0.2
    obs["sats"][1, 9] = 0.1
    obs["sats"][1, 10] = 0.5

    _, _, sat_select_mask, _ = lyapunov_queue_aware_policy_step([obs], cfg, state=None)
    assert sat_select_mask[0, 1] > sat_select_mask[0, 0]


def test_topology_dpp_policy_step_outputs_full_gu_bw_and_state():
    cfg = SaginConfig(num_uav=1, num_gu=5, users_obs_max=3, sats_obs_max=2, nbrs_obs_max=1, N_RF=1)
    cfg.enable_bw_action = True
    cfg.fixed_satellite_strategy = False
    cfg.topology_dpp_accel_num_candidates = 3
    cfg.topology_dpp_gu_max_select = 2
    obs = {
        "own": np.zeros((OWN_OBS_DIM,), dtype=np.float32),
        "users": np.zeros((cfg.users_obs_max, 5), dtype=np.float32),
        "users_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "bw_valid_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "candidate_indices": np.array([1, 3, 4], dtype=np.int64),
        "sats": np.zeros((cfg.sats_obs_max, SAT_OBS_DIM), dtype=np.float32),
        "sats_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "sat_valid_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "nbrs": np.zeros((cfg.nbrs_obs_max, 4), dtype=np.float32),
        "nbrs_mask": np.zeros((cfg.nbrs_obs_max,), dtype=np.float32),
    }
    obs["users_mask"][:2] = 1.0
    obs["bw_valid_mask"][:2] = 1.0
    obs["users"][0, 0:4] = np.array([0.2, 0.0, 1.0, 1.2], dtype=np.float32)
    obs["users"][1, 0:4] = np.array([-0.1, 0.1, 0.8, 0.9], dtype=np.float32)
    obs["sats_mask"][:] = 1.0
    obs["sat_valid_mask"][:] = 1.0
    obs["sats"][0, 7] = 4.0
    obs["sats"][0, 8] = 0.1
    obs["sats"][0, 9] = 0.1
    obs["sats"][0, 10] = 1.0
    obs["sats"][1, 7] = 2.0
    obs["sats"][1, 8] = 0.3
    obs["sats"][1, 9] = 0.2
    obs["sats"][1, 10] = 0.5

    accel, bw_alloc, sat_select_mask, state = topology_dpp_policy_step([obs], cfg, state=None)

    assert accel.shape == (1, 2)
    assert bw_alloc.shape == (1, cfg.num_gu)
    assert sat_select_mask.shape == (1, cfg.sats_obs_max)
    assert state["pressure_ema"].shape == (1, cfg.num_gu)
    assert state["service_est"].shape == (1, cfg.num_gu)
    assert np.all(np.isfinite(accel))
    assert np.linalg.norm(accel[0]) > 0.1
    assert np.all(bw_alloc >= -1e-6)
    assert np.allclose(np.sum(bw_alloc, axis=1), 1.0, atol=1e-5)
    assert bw_alloc[0, 0] == 0.0
    assert bw_alloc[0, 2] == 0.0
    assert np.sum(sat_select_mask[0]) <= cfg.N_RF + 1e-6


def test_cluster_center_queue_aware_policy_combines_accel_bw_and_sat_heads():
    cfg = SaginConfig(num_uav=1, map_size=1000.0, sats_obs_max=2)
    cfg.enable_bw_action = True
    cfg.fixed_satellite_strategy = False

    obs = {
        "own": np.array([0.10, 0.10, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "users": np.zeros((cfg.users_obs_max, 5), dtype=np.float32),
        "users_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "bw_valid_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "sats": np.zeros((cfg.sats_obs_max, SAT_OBS_DIM), dtype=np.float32),
        "sats_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "sat_valid_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "nbrs": np.zeros((cfg.nbrs_obs_max, 4), dtype=np.float32),
        "nbrs_mask": np.zeros((cfg.nbrs_obs_max,), dtype=np.float32),
    }
    obs["users_mask"][0] = 1.0
    obs["bw_valid_mask"][0] = 1.0
    obs["users"][0, 0:2] = np.array([0.4, 0.0], dtype=np.float32)
    obs["users"][0, 2] = 1.0
    obs["users"][0, 3] = 1.5
    obs["users"][0, 4] = 1.0
    obs["sats_mask"][:] = 1.0
    obs["sat_valid_mask"][:] = 1.0
    obs["sats"][0, 7] = 3.0
    obs["sats"][0, 8] = 0.2
    obs["sats"][0, 9] = 0.3
    obs["sats"][0, 10] = 0.8
    obs["sats"][1, 7] = 1.0
    obs["sats"][1, 8] = 0.8
    obs["sats"][1, 9] = 0.9
    obs["sats"][1, 10] = 0.2

    centers = np.array([[900.0, 100.0]], dtype=np.float32)
    counts = np.array([5], dtype=np.int32)

    accel_ref = cluster_center_accel_policy([obs], cfg, centers, counts)
    _, bw_ref, sat_ref = queue_aware_policy([obs], cfg)
    accel, bw_alloc, sat_select_mask = cluster_center_queue_aware_policy([obs], cfg, centers, counts)

    np.testing.assert_allclose(accel, accel_ref, atol=1e-6)
    np.testing.assert_allclose(bw_alloc, bw_ref, atol=1e-6)
    np.testing.assert_allclose(sat_select_mask, sat_ref, atol=1e-6)


def test_cluster_center_batch_policies_match_per_env_outputs():
    cfg = SaginConfig(num_uav=1, map_size=1000.0, users_obs_max=2, sats_obs_max=2, nbrs_obs_max=1)
    cfg.enable_bw_action = True
    cfg.fixed_satellite_strategy = False

    obs_left = {
        "own": np.array([0.10, 0.10, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "users": np.zeros((cfg.users_obs_max, 5), dtype=np.float32),
        "users_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "bw_valid_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "sats": np.zeros((cfg.sats_obs_max, SAT_OBS_DIM), dtype=np.float32),
        "sats_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "sat_valid_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "nbrs": np.zeros((cfg.nbrs_obs_max, 4), dtype=np.float32),
        "nbrs_mask": np.zeros((cfg.nbrs_obs_max,), dtype=np.float32),
    }
    obs_left["users_mask"][0] = 1.0
    obs_left["bw_valid_mask"][0] = 1.0
    obs_left["users"][0, 0:2] = np.array([0.4, 0.0], dtype=np.float32)
    obs_left["users"][0, 2] = 0.8
    obs_left["users"][0, 3] = 1.0
    obs_left["users"][0, 4] = 1.0
    obs_left["sats_mask"][:] = 1.0
    obs_left["sat_valid_mask"][:] = 1.0
    obs_left["sats"][0, 7] = 3.0
    obs_left["sats"][0, 8] = 0.2
    obs_left["sats"][0, 9] = 0.2
    obs_left["sats"][0, 10] = 0.8

    obs_right = {
        "own": np.array([0.90, 0.10, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "users": np.zeros((cfg.users_obs_max, 5), dtype=np.float32),
        "users_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "bw_valid_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "sats": np.zeros((cfg.sats_obs_max, SAT_OBS_DIM), dtype=np.float32),
        "sats_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "sat_valid_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "nbrs": np.zeros((cfg.nbrs_obs_max, 4), dtype=np.float32),
        "nbrs_mask": np.zeros((cfg.nbrs_obs_max,), dtype=np.float32),
    }
    obs_right["users_mask"][1] = 1.0
    obs_right["bw_valid_mask"][1] = 1.0
    obs_right["users"][1, 0:2] = np.array([-0.3, 0.1], dtype=np.float32)
    obs_right["users"][1, 2] = 0.7
    obs_right["users"][1, 3] = 1.2
    obs_right["users"][1, 4] = 0.0
    obs_right["sats_mask"][:] = 1.0
    obs_right["sat_valid_mask"][:] = 1.0
    obs_right["sats"][1, 7] = 4.0
    obs_right["sats"][1, 8] = 0.1
    obs_right["sats"][1, 9] = 0.1
    obs_right["sats"][1, 10] = 0.9

    obs_many = [[obs_left], [obs_right]]
    obs_batch = _pack_obs_many(obs_many)
    centers = np.array([[[100.0, 100.0]], [[900.0, 100.0]]], dtype=np.float32)
    counts = np.array([[5], [7]], dtype=np.float32)

    accel_ref = np.stack(
        [
            cluster_center_accel_policy(obs_list, cfg, centers[idx], counts[idx])
            for idx, obs_list in enumerate(obs_many)
        ],
        axis=0,
    )
    accel_batch = cluster_center_accel_policy_batch(obs_batch, cfg, centers, counts)
    np.testing.assert_allclose(accel_batch, accel_ref, atol=1e-6)

    triple_ref = [
        cluster_center_queue_aware_policy(obs_list, cfg, centers[idx], counts[idx])
        for idx, obs_list in enumerate(obs_many)
    ]
    accel_ref = np.stack([piece[0] for piece in triple_ref], axis=0)
    bw_ref = np.stack([piece[1] for piece in triple_ref], axis=0)
    sat_ref = np.stack([piece[2] for piece in triple_ref], axis=0)
    accel_batch, bw_batch, sat_batch = cluster_center_queue_aware_policy_batch(
        obs_batch,
        cfg,
        centers,
        counts,
    )
    np.testing.assert_allclose(accel_batch, accel_ref, atol=1e-6)
    np.testing.assert_allclose(bw_batch, bw_ref, atol=1e-6)
    np.testing.assert_allclose(sat_batch, sat_ref, atol=1e-6)
