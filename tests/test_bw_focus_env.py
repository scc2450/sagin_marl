from __future__ import annotations

import numpy as np

from sagin_marl.env.config import SaginConfig
from sagin_marl.env.sagin_env import SaginParallelEnv


def _make_focus_env(**overrides) -> SaginParallelEnv:
    cfg = SaginConfig(
        seed=7,
        map_size=200.0,
        tau0=1.0,
        T_steps=16,
        num_uav=2,
        num_gu=6,
        num_sat=3,
        users_obs_max=6,
        sats_obs_max=3,
        nbrs_obs_max=1,
        uav_height=10.0,
        task_arrival_rate=10.0,
        task_arrival_poisson=False,
        traffic_model="sticky_subset_hotspot",
        arrival_base_hetero=0.0,
        hotspot_num_subsets=4,
        hotspot_subset_size=2,
        hotspot_rho=4.0,
        hotspot_on_mean_steps=15.0,
        hotspot_off_mean_steps=8.0,
        arrival_mean_preserve=True,
        queue_max_gu=100.0,
        queue_max_uav=100.0,
        queue_max_sat=100.0,
        queue_init_frac=0.0,
        queue_init_uav_frac=0.0,
        queue_init_sat_frac=0.0,
        queue_ref_gu_per_step=60.0,
        queue_ref_uav_per_step=40.0,
        queue_ref_sat_per_step=30.0,
        queue_ref_sat_active_count=1.0,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    env = SaginParallelEnv(cfg)
    env.gu_pos = np.asarray(
        [
            [10.0, 10.0],
            [12.0, 10.0],
            [14.0, 10.0],
            [150.0, 10.0],
            [152.0, 10.0],
            [154.0, 10.0],
        ],
        dtype=np.float32,
    )
    env.uav_pos = np.asarray([[12.0, 12.0], [152.0, 12.0]], dtype=np.float32)
    env.uav_vel = np.zeros((cfg.num_uav, 2), dtype=np.float32)
    env.gu_queue.fill(0.0)
    env.uav_queue.fill(0.0)
    env.sat_queue.fill(0.0)
    env._init_traffic_model_state()
    return env


def test_sticky_subset_hotspot_arrivals_preserve_mean_and_localize_pressure():
    env = _make_focus_env()
    assert len(env._hotspot_subsets) >= 1
    env._hotspot_active_idx = 0

    rates = env._current_task_arrival_rates(10.0)
    hot_subset = np.asarray(env._hotspot_subsets[0], dtype=np.int32)
    cold_mask = np.ones((env.cfg.num_gu,), dtype=bool)
    cold_mask[hot_subset] = False

    np.testing.assert_allclose(float(np.mean(rates)), 10.0, atol=1e-6)
    assert np.all(rates[hot_subset] > 10.0)
    assert np.all(rates[cold_mask] < 10.0)


def test_focus_preload_localizes_gu_and_uav_queues():
    env = _make_focus_env(
        preload_enabled=True,
        preload_prob=1.0,
        preload_hot_gu_steps=3.0,
        preload_bg_gu_steps=0.2,
        preload_hot_uav_steps=2.0,
        preload_sat_steps=0.0,
    )
    env._apply_focus_preload()

    hot_idx = int(env.last_hotspot_index)
    assert hot_idx >= 0
    hot_mask = np.asarray(env._hotspot_member_mask[hot_idx], dtype=bool)
    cold_mask = ~hot_mask

    per_gu_ref = env._queue_init_entity_ref("gu")
    per_uav_ref = env._queue_init_entity_ref("uav")
    np.testing.assert_allclose(env.gu_queue[hot_mask], 3.0 * per_gu_ref, atol=1e-6)
    np.testing.assert_allclose(env.gu_queue[cold_mask], 0.2 * per_gu_ref, atol=1e-6)

    assoc = env._associate_users()
    hot_uavs = np.unique(assoc[hot_mask][assoc[hot_mask] >= 0])
    assert hot_uavs.size >= 1
    np.testing.assert_allclose(env.uav_queue[hot_uavs], 2.0 * per_uav_ref, atol=1e-6)
    cold_uavs = np.setdiff1d(np.arange(env.cfg.num_uav), hot_uavs)
    if cold_uavs.size > 0:
        np.testing.assert_allclose(env.uav_queue[cold_uavs], 0.0, atol=1e-6)
    np.testing.assert_allclose(env.sat_queue, 0.0, atol=1e-6)


def test_effective_sat_resource_scaling_helpers():
    env = _make_focus_env(
        b_backhaul_per_sat=100.0,
        b_backhaul_per_sat_scale=1.5,
        sat_cpu_freq=1000.0,
        sat_cpu_freq_scale=2.0,
    )
    assert env._effective_b_backhaul_per_sat() == 150.0
    assert env._effective_sat_cpu_freq() == 2000.0
