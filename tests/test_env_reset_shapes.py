from __future__ import annotations

import numpy as np

from sagin_marl.env.config import SaginConfig
from sagin_marl.env.sagin_env import SaginParallelEnv


def test_env_reset_shapes():
    cfg = SaginConfig(num_uav=2, num_gu=5, num_sat=3, users_obs_max=5, sats_obs_max=3, nbrs_obs_max=1)
    env = SaginParallelEnv(cfg)
    obs, _ = env.reset()
    assert len(obs) == cfg.num_uav
    sample = next(iter(obs.values()))
    assert sample["own"].shape == (env.own_dim,)
    assert sample["users"].shape == (cfg.users_obs_max, env.user_dim)
    assert sample["users_mask"].shape == (cfg.users_obs_max,)
    assert sample["sats"].shape == (cfg.sats_obs_max, env.sat_dim)
    assert sample["sats_mask"].shape == (cfg.sats_obs_max,)
    assert sample["nbrs"].shape == (cfg.nbrs_obs_max, env.nbr_dim)
    assert sample["nbrs_mask"].shape == (cfg.nbrs_obs_max,)


def test_env_reset_shapes_with_danger_neighbor_obs():
    cfg = SaginConfig(
        num_uav=2,
        num_gu=5,
        num_sat=3,
        users_obs_max=5,
        sats_obs_max=3,
        nbrs_obs_max=1,
        danger_nbr_enabled=True,
    )
    env = SaginParallelEnv(cfg)
    obs, _ = env.reset()
    sample = next(iter(obs.values()))
    assert sample["danger_nbr"].shape == (env.danger_nbr_dim,)


def test_env_reset_shapes_with_reward_aligned_obs_features():
    cfg = SaginConfig(
        num_uav=1,
        num_gu=5,
        num_sat=3,
        users_obs_max=5,
        sats_obs_max=3,
        nbrs_obs_max=0,
        enable_bw_action=True,
        obs_own_include_assoc_uav_cost=True,
        obs_user_include_local_gu_service_cost=True,
        obs_user_include_weighted_queue_cost=True,
        obs_user_include_weighted_queue_cost_relative=True,
        obs_sat_include_sat_cost=True,
    )
    env = SaginParallelEnv(cfg)
    obs, _ = env.reset(seed=cfg.seed)
    sample = next(iter(obs.values()))
    assert sample["own"].shape == (env.own_dim,)
    assert sample["users"].shape == (cfg.users_obs_max, env.user_dim)
    assert sample["sats"].shape == (cfg.sats_obs_max, env.sat_dim)
    assert env.own_dim == 11
    assert env.user_dim == 8
    assert env.gu_node_dim == 6
    assert env.sat_dim == 13

    reward_features = env._gu_reward_aligned_feature_dict(normalized=True)
    for key, values in reward_features.items():
        assert values.shape == (cfg.num_gu,)
        assert np.all(np.isfinite(values)), key
        assert np.all(np.abs(values) < 32.0), key
    uav_features = env._uav_reward_aligned_feature_dict(normalized=True)
    sat_features = env._sat_reward_aligned_feature_dict(normalized=True)
    assert uav_features["assoc_uav_cost"].shape == (cfg.num_uav,)
    assert sat_features["sat_cost"].shape == (cfg.num_sat,)


def test_env_reset_safe_random_uav_init():
    cfg = SaginConfig(
        seed=7,
        map_size=1000.0,
        tau0=1.0,
        num_uav=3,
        num_gu=5,
        num_sat=3,
        users_obs_max=5,
        sats_obs_max=3,
        nbrs_obs_max=2,
        v_max=30.0,
        d_safe=20.0,
        uav_spawn_curriculum_enabled=False,
        uav_safe_random_init_enabled=True,
        uav_init_boundary_margin_steps=3.0,
        uav_init_speed_frac=0.2,
        uav_init_min_spacing=20.0,
    )
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed)

    margin = min(cfg.uav_init_boundary_margin_steps * cfg.v_max * cfg.tau0, 0.5 * cfg.map_size - 1e-6)
    assert np.all(env.uav_pos >= margin - 1e-6)
    assert np.all(env.uav_pos <= cfg.map_size - margin + 1e-6)

    speeds = np.linalg.norm(env.uav_vel, axis=1)
    assert np.allclose(speeds, cfg.uav_init_speed_frac * cfg.v_max, atol=1e-5)

    for i in range(cfg.num_uav):
        for j in range(i + 1, cfg.num_uav):
            dist = float(np.linalg.norm(env.uav_pos[i] - env.uav_pos[j]))
            assert dist >= cfg.uav_init_min_spacing - 1e-6
