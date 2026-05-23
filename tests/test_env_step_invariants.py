from __future__ import annotations

import numpy as np
from unittest import mock

from sagin_marl.env.config import SaginConfig
from sagin_marl.env.sagin_env import SaginParallelEnv


def test_env_step_invariants():
    cfg = SaginConfig(num_uav=2, num_gu=5, num_sat=3, users_obs_max=5, sats_obs_max=3, nbrs_obs_max=1)
    env = SaginParallelEnv(cfg)
    obs, _ = env.reset()
    actions = {agent: {"accel": np.zeros(2, dtype=np.float32),
                       "bw_logits": np.zeros(cfg.users_obs_max, dtype=np.float32),
                       "sat_logits": np.zeros(cfg.sats_obs_max, dtype=np.float32)}
               for agent in env.agents}
    for _ in range(3):
        obs, rewards, terms, truncs, _ = env.step(actions)
        assert np.all(env.gu_queue >= 0)
        assert np.all(env.uav_queue >= 0)
        assert np.all(env.sat_queue >= 0)
        assert np.isfinite(env.gu_queue).all()
        assert np.isfinite(env.uav_queue).all()
        assert np.isfinite(env.sat_queue).all()


def test_queue_init_steps_use_arrival_reference():
    cfg = SaginConfig(
        num_uav=2,
        num_gu=4,
        num_sat=3,
        users_obs_max=4,
        sats_obs_max=3,
        nbrs_obs_max=1,
        task_arrival_rate=10.0,
        traffic_level=2,
        queue_init_gu_steps=3.0,
        queue_init_uav_steps=1.0,
        queue_init_sat_steps=2.0,
    )
    env = SaginParallelEnv(cfg)
    env._init_state()

    arrival_ref = cfg.task_arrival_rate * cfg.num_gu * cfg.tau0
    np.testing.assert_allclose(float(np.sum(env.gu_queue)), 3.0 * arrival_ref, atol=1e-6)
    np.testing.assert_allclose(float(np.sum(env.uav_queue)), 1.0 * arrival_ref, atol=1e-6)
    np.testing.assert_allclose(float(np.sum(env.sat_queue)), 2.0 * arrival_ref, atol=1e-6)


def test_queue_init_steps_use_layer_inflow_references_when_provided():
    cfg = SaginConfig(
        num_uav=2,
        num_gu=4,
        num_sat=3,
        users_obs_max=4,
        sats_obs_max=3,
        nbrs_obs_max=1,
        task_arrival_rate=10.0,
        queue_init_gu_steps=3.0,
        queue_init_uav_steps=1.0,
        queue_init_sat_steps=2.0,
        queue_ref_gu_per_step=100.0,
        queue_ref_uav_per_step=250.0,
        queue_ref_sat_per_step=400.0,
    )
    env = SaginParallelEnv(cfg)
    env._init_state()

    np.testing.assert_allclose(float(np.sum(env.gu_queue)), 300.0, atol=1e-6)
    np.testing.assert_allclose(float(np.sum(env.uav_queue)), 250.0, atol=1e-6)
    np.testing.assert_allclose(float(np.sum(env.sat_queue)), 800.0, atol=1e-6)


def test_queue_init_abs_overrides_fraction_and_clips_to_cap():
    cfg = SaginConfig(
        num_uav=2,
        num_gu=4,
        num_sat=3,
        users_obs_max=4,
        sats_obs_max=3,
        nbrs_obs_max=1,
        queue_max_gu=100.0,
        queue_max_uav=50.0,
        queue_init_frac=0.1,
        queue_init_uav_frac=0.1,
        queue_init_gu_abs=1000.0,
        queue_init_uav_abs=500.0,
    )
    env = SaginParallelEnv(cfg)
    env._init_state()

    np.testing.assert_allclose(float(np.sum(env.gu_queue)), 400.0, atol=1e-6)
    np.testing.assert_allclose(float(np.sum(env.uav_queue)), 100.0, atol=1e-6)


def test_sat_queue_clips_to_capacity_and_records_drop():
    cfg = SaginConfig(
        num_uav=1,
        num_gu=1,
        num_sat=1,
        users_obs_max=1,
        sats_obs_max=1,
        nbrs_obs_max=1,
        queue_max_sat=100.0,
        sat_cpu_freq=0.0,
    )
    env = SaginParallelEnv(cfg)
    env.reset()

    outflow_matrix = np.array([[250.0]], dtype=np.float32)
    env._update_sat_queues(outflow_matrix)

    np.testing.assert_allclose(float(np.sum(env.last_sat_incoming)), 250.0, atol=1e-6)
    np.testing.assert_allclose(float(np.sum(env.last_sat_processed)), 0.0, atol=1e-6)
    np.testing.assert_allclose(float(np.sum(env.sat_drop)), 150.0, atol=1e-6)
    np.testing.assert_allclose(float(np.sum(env.sat_queue)), 100.0, atol=1e-6)


def test_reward_drop_sum_includes_sat_drop():
    cfg = SaginConfig(
        num_uav=1,
        num_gu=1,
        num_sat=1,
        users_obs_max=1,
        sats_obs_max=1,
        nbrs_obs_max=1,
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.last_gu_arrival = np.array([200.0], dtype=np.float32)
    env.last_gu_outflow = np.zeros((1,), dtype=np.float32)
    env.gu_drop = np.zeros((1,), dtype=np.float32)
    env.uav_drop = np.zeros((1,), dtype=np.float32)
    env.sat_drop = np.array([20.0], dtype=np.float32)
    env.last_sat_incoming = np.array([40.0], dtype=np.float32)
    env.last_sat_processed = np.array([10.0], dtype=np.float32)

    env._compute_reward()
    parts = env.last_reward_parts

    assert abs(float(parts["drop_sum_active"])) < 1e-9
    assert abs(float(parts["sat_drop_sum"]) - 20.0) < 1e-9
    assert abs(float(parts["drop_sum"]) - 20.0) < 1e-9
    assert abs(float(parts["drop_ratio"]) - 0.1) < 1e-9


def test_reward_drop_terms_can_be_weighted_per_layer():
    cfg = SaginConfig(
        num_uav=1,
        num_gu=1,
        num_sat=1,
        users_obs_max=1,
        sats_obs_max=1,
        nbrs_obs_max=1,
        task_arrival_rate=200.0,
        omega_q=0.0,
        eta_drop=1.3,
        eta_drop_gu=1.0,
        eta_drop_uav=2.0,
        eta_drop_sat=3.0,
        eta_drop_step=0.0,
        eta_service=0.0,
        eta_q_delta=0.0,
        eta_accel=0.0,
        eta_centroid=0.0,
        eta_crash=0.0,
        close_risk_enabled=False,
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.last_gu_arrival = np.array([200.0], dtype=np.float32)
    env.gu_drop = np.array([10.0], dtype=np.float32)
    env.uav_drop = np.array([20.0], dtype=np.float32)
    env.sat_drop = np.array([30.0], dtype=np.float32)

    env._compute_reward()
    parts = env.last_reward_parts

    assert abs(float(parts["gu_drop_norm"]) - 0.05) < 1e-9
    assert abs(float(parts["uav_drop_norm"]) - 0.10) < 1e-9
    assert abs(float(parts["sat_drop_norm"]) - 0.15) < 1e-9
    assert abs(float(parts["term_drop_gu"]) + 0.05) < 1e-9
    assert abs(float(parts["term_drop_uav"]) + 0.20) < 1e-9
    assert abs(float(parts["term_drop_sat"]) + 0.45) < 1e-9
    assert abs(float(parts["term_drop"]) + 0.70) < 1e-9


def test_tail_queue_penalty_active_branch():
    cfg = SaginConfig(
        num_uav=2,
        num_gu=5,
        num_sat=3,
        users_obs_max=5,
        sats_obs_max=3,
        nbrs_obs_max=1,
        queue_delta_use_active=True,
        q_norm_tail_q0=0.005,
        omega_q=1.0,
        omega_q_tail=10.0,
    )
    env = SaginParallelEnv(cfg)
    obs, _ = env.reset()
    actions = {
        agent: {
            "accel": np.zeros(2, dtype=np.float32),
            "bw_logits": np.zeros(cfg.users_obs_max, dtype=np.float32),
            "sat_logits": np.zeros(cfg.sats_obs_max, dtype=np.float32),
        }
        for agent in env.agents
    }
    env.step(actions)
    parts = env.last_reward_parts
    q0 = float(parts["q_norm_tail_q0"])
    q_norm = float(parts["q_norm_active"])
    x = max(q_norm - q0, 0.0)
    assert abs(float(parts["q_norm_tail_excess"]) - x) < 1e-8
    assert abs(float(parts["queue_pen"]) - x * x) < 1e-8
    assert abs(float(parts["queue_weight"]) - 10.0) < 1e-8
    assert abs(float(parts["term_queue"]) + float(parts["queue_weight"]) * float(parts["queue_pen"])) < 1e-8


def test_reward_throughput_terms_are_applied():
    cfg = SaginConfig(
        num_uav=1,
        num_gu=1,
        num_sat=1,
        users_obs_max=1,
        sats_obs_max=1,
        nbrs_obs_max=1,
        task_arrival_rate=200.0,
        omega_q=0.0,
        eta_drop=0.0,
        eta_drop_step=0.0,
        eta_service=0.0,
        eta_throughput_access=0.4,
        eta_throughput_backhaul=0.6,
        eta_q_delta=0.0,
        eta_accel=0.0,
        eta_centroid=0.0,
        eta_crash=0.0,
        close_risk_enabled=False,
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.last_gu_arrival = np.array([200.0], dtype=np.float32)
    env.last_gu_outflow = np.array([50.0], dtype=np.float32)
    env.last_sat_incoming = np.array([80.0], dtype=np.float32)
    env.last_sat_processed = np.array([30.0], dtype=np.float32)

    reward = env._compute_reward()
    parts = env.last_reward_parts

    assert abs(float(parts["throughput_access_norm"]) - 0.25) < 1e-9
    assert abs(float(parts["throughput_backhaul_norm"]) - 0.4) < 1e-9
    assert abs(float(parts["sat_processed_norm"]) - 0.15) < 1e-9
    assert abs(float(parts["term_throughput_access"]) - 0.1) < 1e-9
    assert abs(float(parts["term_throughput_backhaul"]) - 0.24) < 1e-9
    assert abs(float(parts["reward_raw"]) - 0.34) < 1e-9
    assert abs(float(reward) - 0.34) < 1e-9


def test_env_step_uses_shared_post_bw_helpers():
    cfg = SaginConfig(num_uav=2, num_gu=4, num_sat=3, users_obs_max=4, sats_obs_max=3, nbrs_obs_max=1)
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed)
    actions = {
        agent: {
            "accel": np.zeros(2, dtype=np.float32),
            "bw_logits": np.zeros(cfg.users_obs_max, dtype=np.float32),
            "sat_logits": np.zeros(cfg.sats_obs_max, dtype=np.float32),
        }
        for agent in env.agents
    }

    with (
        mock.patch.object(
            env,
            "_prepare_next_step_observation_cache",
            wraps=env._prepare_next_step_observation_cache,
        ) as cache_mock,
        mock.patch.object(
            env,
            "_finalize_post_bw_step",
            wraps=env._finalize_post_bw_step,
        ) as finalize_mock,
        mock.patch.object(
            env,
            "_materialize_post_step_outputs",
            wraps=env._materialize_post_step_outputs,
        ) as materialize_mock,
    ):
        obs, rewards, terms, truncs, infos = env.step(actions)

    assert set(obs.keys()) == set(env.agents)
    assert set(rewards.keys()) == set(env.agents)
    assert set(terms.keys()) == set(env.agents)
    assert set(truncs.keys()) == set(env.agents)
    assert set(infos.keys()) == set(env.agents)
    cache_mock.assert_called_once()
    assert cache_mock.call_args.kwargs["advance_doppler"] is True
    finalize_mock.assert_called_once()
    materialize_mock.assert_called_once()
    assert materialize_mock.call_args.kwargs["materialize_step_outputs"] is True
    assert materialize_mock.call_args.kwargs["materialize_agent_dicts"] is True


def test_env_reset_uses_shared_obs_builder_without_per_uav_obs_helper():
    cfg = SaginConfig(
        seed=17,
        num_uav=3,
        num_gu=5,
        num_sat=6,
        users_obs_max=4,
        sats_obs_max=4,
        nbrs_obs_max=2,
        danger_nbr_enabled=True,
        obs_own_include_assoc_uav_cost=True,
        obs_user_include_assoc_uav_cost=True,
        obs_user_include_assoc_sat_cost_mean=True,
        obs_user_include_weighted_queue_cost=True,
    )
    baseline_env = SaginParallelEnv(cfg)
    env = SaginParallelEnv(cfg)
    try:
        baseline_obs, baseline_infos = baseline_env.reset(seed=cfg.seed)
        baseline_obs_old = {agent: baseline_env._get_obs(idx) for idx, agent in enumerate(baseline_env.agents)}
        baseline_global = np.asarray(baseline_env._build_global_state(), dtype=np.float32)

        with (
            mock.patch.object(env, "_get_obs", side_effect=AssertionError("unexpected per-uav obs helper during reset")),
            mock.patch.object(env, "_danger_neighbor_obs", side_effect=AssertionError("unexpected scalar danger helper during reset")),
            mock.patch.object(env, "_gu_proxy_feature_arrays", wraps=env._gu_proxy_feature_arrays) as gu_proxy_mock,
            mock.patch.object(env, "_uav_reward_aligned_feature_dict", wraps=env._uav_reward_aligned_feature_dict) as uav_reward_mock,
        ):
            obs, infos = env.reset(seed=cfg.seed)

        assert set(obs.keys()) == set(baseline_obs.keys())
        assert infos == baseline_infos
        for agent in env.agents:
            for key, baseline_value in baseline_obs_old[agent].items():
                np.testing.assert_allclose(
                    np.asarray(obs[agent][key]),
                    np.asarray(baseline_value),
                    rtol=1e-6,
                    atol=1e-6,
                )
        np.testing.assert_allclose(np.asarray(env.get_global_state(), dtype=np.float32), baseline_global, rtol=1e-6, atol=1e-6)
        assert gu_proxy_mock.call_count == 1
        assert uav_reward_mock.call_count == 1
    finally:
        close_fn = getattr(baseline_env, "close", None)
        if callable(close_fn):
            close_fn()
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def test_env_step_materialize_uses_shared_obs_builder_without_per_uav_obs_helper():
    cfg = SaginConfig(
        seed=18,
        num_uav=3,
        num_gu=5,
        num_sat=6,
        users_obs_max=4,
        sats_obs_max=4,
        nbrs_obs_max=2,
        danger_nbr_enabled=True,
        obs_own_include_assoc_uav_cost=True,
        obs_user_include_assoc_uav_cost=True,
        obs_user_include_assoc_sat_cost_mean=True,
        obs_user_include_weighted_queue_cost=True,
    )
    baseline_env = SaginParallelEnv(cfg)
    env = SaginParallelEnv(cfg)
    actions = {
        agent: {
            "accel": np.zeros(2, dtype=np.float32),
            "bw_logits": np.zeros(cfg.users_obs_max, dtype=np.float32),
            "sat_logits": np.zeros(cfg.sats_obs_max, dtype=np.float32),
        }
        for agent in env.agents
    }
    try:
        baseline_env.reset(seed=cfg.seed)
        baseline_obs, baseline_rewards, baseline_terms, baseline_truncs, baseline_infos = baseline_env.step(actions)
        baseline_global = np.asarray(baseline_env.get_global_state(), dtype=np.float32)

        env.reset(seed=cfg.seed)
        with (
            mock.patch.object(env, "_get_obs", side_effect=AssertionError("unexpected per-uav obs helper during step materialize")),
            mock.patch.object(env, "_danger_neighbor_obs", side_effect=AssertionError("unexpected scalar danger helper during step materialize")),
            mock.patch.object(env, "_gu_proxy_feature_arrays", wraps=env._gu_proxy_feature_arrays) as gu_proxy_mock,
            mock.patch.object(env, "_uav_reward_aligned_feature_dict", wraps=env._uav_reward_aligned_feature_dict) as uav_reward_mock,
        ):
            obs, rewards, terms, truncs, infos = env.step(actions)

        for agent in env.agents:
            for key, baseline_value in baseline_obs[agent].items():
                np.testing.assert_allclose(
                    np.asarray(obs[agent][key]),
                    np.asarray(baseline_value),
                    rtol=1e-6,
                    atol=1e-6,
                )
        assert rewards == baseline_rewards
        assert terms == baseline_terms
        assert truncs == baseline_truncs
        assert infos == baseline_infos
        np.testing.assert_allclose(np.asarray(env.get_global_state(), dtype=np.float32), baseline_global, rtol=1e-6, atol=1e-6)
        assert gu_proxy_mock.call_count == 1
        assert uav_reward_mock.call_count == 1
    finally:
        close_fn = getattr(baseline_env, "close", None)
        if callable(close_fn):
            close_fn()
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def test_get_obs_uses_shared_runtime_context_without_scalar_helpers():
    cfg = SaginConfig(
        seed=19,
        num_uav=3,
        num_gu=5,
        num_sat=6,
        users_obs_max=4,
        sats_obs_max=4,
        nbrs_obs_max=2,
        danger_nbr_enabled=True,
        obs_own_include_assoc_uav_cost=True,
        obs_user_include_assoc_uav_cost=True,
        obs_user_include_assoc_sat_cost_mean=True,
        obs_user_include_weighted_queue_cost=True,
    )
    env = SaginParallelEnv(cfg)
    try:
        env.reset(seed=cfg.seed)
        baseline = env._get_obs(0)
        with (
            mock.patch.object(env, "_assoc_centroid_summary", side_effect=AssertionError("unexpected assoc centroid helper")),
            mock.patch.object(env, "_uav_reward_aligned_feature_dict", side_effect=AssertionError("unexpected uav reward helper")),
            mock.patch.object(env, "_gu_proxy_feature_arrays", side_effect=AssertionError("unexpected gu proxy helper")),
            mock.patch.object(env, "_ensure_neighbor_cache", side_effect=AssertionError("unexpected neighbor helper")),
            mock.patch.object(env, "_danger_neighbor_obs", side_effect=AssertionError("unexpected danger helper")),
        ):
            obs = env._get_obs(0)
        for key, baseline_value in baseline.items():
            np.testing.assert_allclose(
                np.asarray(obs[key]),
                np.asarray(baseline_value),
                rtol=1e-6,
                atol=1e-6,
            )
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def test_reward_mode_throughput_only_ignores_dense_shaping_terms():
    cfg = SaginConfig(
        num_uav=1,
        num_gu=1,
        num_sat=1,
        users_obs_max=1,
        sats_obs_max=1,
        nbrs_obs_max=1,
        task_arrival_rate=200.0,
        reward_mode="throughput_only",
        omega_q=7.0,
        eta_drop=9.0,
        eta_drop_step=3.0,
        eta_service=2.0,
        eta_throughput_access=0.4,
        eta_throughput_backhaul=0.6,
        eta_q_delta=5.0,
        eta_accel=4.0,
        eta_centroid=1.5,
        eta_crash=8.0,
        close_risk_enabled=False,
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.gu_queue[:] = 100.0
    env.uav_queue[:] = 40.0
    env.sat_queue[:] = 20.0
    env.gu_drop[:] = 10.0
    env.uav_drop[:] = 5.0
    env.sat_drop[:] = 2.0
    env.prev_queue_sum = 240.0
    env.prev_queue_sum_gu = 120.0
    env.prev_queue_sum_uav = 60.0
    env.prev_queue_sum_sat = 60.0
    env.last_exec_accel = np.array([[3.0, 4.0]], dtype=np.float32)
    env.last_gu_arrival = np.array([200.0], dtype=np.float32)
    env.last_gu_outflow = np.array([50.0], dtype=np.float32)
    env.last_sat_incoming = np.array([80.0], dtype=np.float32)
    env.last_sat_processed = np.array([30.0], dtype=np.float32)

    reward = env._compute_reward()
    parts = env.last_reward_parts

    assert abs(float(parts["term_throughput_access"]) - 0.25) < 1e-9
    assert abs(float(parts["term_throughput_backhaul"]) - 0.4) < 1e-9
    assert abs(float(parts["term_drop"])) < 1e-9
    assert abs(float(parts["term_queue"])) < 1e-9
    assert abs(float(parts["term_q_delta"])) < 1e-9
    assert abs(float(parts["term_accel"])) < 1e-9
    assert abs(float(parts["queue_weight"])) < 1e-9
    assert abs(float(parts["q_delta_weight"])) < 1e-9
    assert abs(float(parts["crash_weight"])) < 1e-9
    assert abs(float(parts["reward_raw"]) - 0.65) < 1e-9
    assert abs(float(reward) - 0.65) < 1e-9


def test_controllable_flow_uses_log_pre_backlog_penalty_and_alias_terms():
    cfg = SaginConfig(
        num_uav=1,
        num_gu=1,
        num_sat=1,
        users_obs_max=1,
        sats_obs_max=1,
        nbrs_obs_max=1,
        task_arrival_rate=100.0,
        reward_mode="controllable_flow",
        reward_w_access=0.5,
        reward_w_relay=0.5,
        reward_w_pre_backlog=0.08,
        reward_w_pre_drop=1.0,
        reward_w_pre_growth=0.0,
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.gu_queue[:] = 60.0
    env.uav_queue[:] = 40.0
    env.gu_drop[:] = 10.0
    env.uav_drop[:] = 5.0
    env.sat_drop[:] = 0.0
    env.last_gu_arrival = np.array([100.0], dtype=np.float32)
    env.last_gu_outflow = np.array([50.0], dtype=np.float32)
    env.last_sat_incoming = np.array([40.0], dtype=np.float32)
    env.last_sat_processed = np.array([20.0], dtype=np.float32)

    reward = env._compute_reward()
    parts = env.last_reward_parts
    expected = 0.25 + 0.20 - 0.15 - 0.08 * np.log1p(1.0)

    assert abs(float(parts["b_pre_steps"]) - 1.0) < 1e-9
    assert abs(float(parts["term_access"]) - 0.25) < 1e-9
    assert abs(float(parts["term_relay"]) - 0.20) < 1e-9
    assert abs(float(parts["term_pre_drop"]) + 0.15) < 1e-9
    assert abs(float(parts["term_pre_backlog"]) + 0.08 * np.log1p(1.0)) < 1e-9
    assert abs(float(parts["term_queue"]) - float(parts["term_pre_backlog"])) < 1e-9
    assert abs(float(parts["term_drop"]) - float(parts["term_pre_drop"])) < 1e-9
    assert abs(float(parts["reward_raw"]) - expected) < 1e-9
    assert abs(float(reward) - expected) < 1e-9


def test_reward_mode_weighted_workload_level_uses_nested_per_entity_cost():
    cfg = SaginConfig(
        num_uav=1,
        num_gu=1,
        num_sat=1,
        users_obs_max=1,
        sats_obs_max=1,
        nbrs_obs_max=1,
        reward_mode="weighted_workload_level",
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.gu_queue[:] = 10.0
    env.uav_queue[:] = 20.0
    env.sat_queue[:] = 30.0
    env.gu_drop[:] = 1.0
    env.uav_drop[:] = 2.0
    env.sat_drop[:] = 3.0
    env.last_association = np.asarray([0], dtype=np.int32)
    env.last_sat_selection = [[0]]
    env.bw_weighted_workload_acc_ema_vec = np.asarray([2.0], dtype=np.float32)
    env.bw_weighted_workload_rel_ema_vec = np.asarray([4.0], dtype=np.float32)
    env.bw_weighted_workload_sat_ema_vec = np.asarray([8.0], dtype=np.float32)

    reward = env._compute_reward()
    parts = env.last_reward_parts

    sat_cost = 1.0 / 8.0
    uav_cost = 1.0 / 4.0 + sat_cost
    gu_cost = 1.0 / 2.0 + uav_cost
    expected = -(
        gu_cost * 10.0
        + uav_cost * 20.0
        + sat_cost * 30.0
        + gu_cost * 1.0
        + uav_cost * 2.0
        + sat_cost * 3.0
    )

    assert abs(float(parts["bw_weighted_workload_level_reward"]) - expected) < 1e-9
    assert abs(float(parts["reward_raw"]) - expected) < 1e-9
    assert abs(float(reward) - expected) < 1e-9


def test_own_obs_includes_assoc_centroid_summary_features():
    cfg = SaginConfig(
        num_uav=2,
        num_gu=3,
        num_sat=1,
        users_obs_max=3,
        sats_obs_max=1,
        nbrs_obs_max=1,
        map_size=100.0,
    )
    env = SaginParallelEnv(cfg)
    env.reset(seed=0)
    env.uav_pos[:] = np.array([[10.0, 10.0], [90.0, 10.0]], dtype=np.float32)
    env.gu_pos[:] = np.array([[20.0, 10.0], [40.0, 10.0], [80.0, 20.0]], dtype=np.float32)
    env._cached_assoc = np.array([0, 0, 1], dtype=np.int32)

    obs0 = env._get_obs(0)
    obs1 = env._get_obs(1)

    assert obs0["own"].shape == (10,)
    assert abs(float(obs0["own"][7]) - (2.0 / 3.0)) < 1e-6
    assert abs(float(obs0["own"][8]) - 0.20) < 1e-6
    assert abs(float(obs0["own"][9]) - 0.0) < 1e-6
    assert abs(float(obs1["own"][7]) - (1.0 / 3.0)) < 1e-6
    assert abs(float(obs1["own"][8]) + 0.10) < 1e-6
    assert abs(float(obs1["own"][9]) - 0.10) < 1e-6


def test_sat_overlap_eval_matches_selected_satellite_pattern():
    cfg = SaginConfig(
        num_uav=3,
        num_gu=0,
        num_sat=3,
        users_obs_max=1,
        sats_obs_max=2,
        nbrs_obs_max=1,
        fixed_satellite_strategy=False,
    )
    env = SaginParallelEnv(cfg)
    env.reset(seed=0)
    env.last_sat_selection = [[0, 1], [0], [2]]

    assert abs(env._compute_sat_overlap_eval() - 0.25) < 1e-9


def test_weighted_queue_delta_uses_layer_weights():
    cfg = SaginConfig(
        num_uav=1,
        num_gu=1,
        num_sat=1,
        users_obs_max=1,
        sats_obs_max=1,
        nbrs_obs_max=1,
        queue_max_gu=100.0,
        queue_max_uav=200.0,
        queue_max_sat=400.0,
        omega_q=0.0,
        omega_q_gu=1.0,
        omega_q_uav=0.5,
        omega_q_sat=0.25,
        eta_drop=0.0,
        eta_drop_step=0.0,
        eta_service=0.0,
        eta_q_delta=1.0,
        eta_accel=0.0,
        eta_crash=0.0,
        queue_delta_use_active=False,
        queue_delta_mode="weighted",
        close_risk_enabled=False,
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.prev_queue_sum = 380.0
    env.prev_queue_sum_gu = 80.0
    env.prev_queue_sum_uav = 100.0
    env.prev_queue_sum_sat = 200.0
    env.gu_queue[:] = 60.0
    env.uav_queue[:] = 120.0
    env.sat_queue[:] = 160.0

    env._compute_reward()
    parts = env.last_reward_parts

    assert abs(float(parts["queue_delta_gu"]) - 0.2) < 1e-9
    assert abs(float(parts["queue_delta_uav"]) + 0.1) < 1e-9
    assert abs(float(parts["queue_delta_sat"]) - 0.1) < 1e-9
    assert parts["queue_delta_mode"] == "weighted"
    assert abs(float(parts["queue_delta"]) - 0.1) < 1e-9
    assert abs(float(parts["term_q_delta"]) - 0.1) < 1e-9


def test_queue_reward_can_use_arrival_normalization():
    cfg = SaginConfig(
        num_uav=1,
        num_gu=1,
        num_sat=1,
        users_obs_max=1,
        sats_obs_max=1,
        nbrs_obs_max=1,
        task_arrival_rate=10.0,
        tau0=1.0,
        queue_max_gu=100.0,
        queue_max_uav=100.0,
        queue_max_sat=100.0,
        omega_q=1.0,
        omega_q_gu=1.0,
        omega_q_uav=0.5,
        omega_q_sat=0.25,
        eta_drop=0.0,
        eta_drop_step=0.0,
        eta_service=0.0,
        eta_q_delta=1.0,
        eta_accel=0.0,
        eta_crash=0.0,
        queue_penalty_mode="linear",
        queue_delta_use_active=False,
        queue_delta_mode="weighted",
        queue_reward_use_arrival_norm=True,
        close_risk_enabled=False,
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.prev_queue_sum = 90.0
    env.prev_queue_sum_gu = 30.0
    env.prev_queue_sum_uav = 40.0
    env.prev_queue_sum_sat = 20.0
    env.gu_queue[:] = 20.0
    env.uav_queue[:] = 50.0
    env.sat_queue[:] = 10.0

    env._compute_reward()
    parts = env.last_reward_parts

    assert abs(float(parts["gu_queue_arrival_steps"]) - 2.0) < 1e-9
    assert abs(float(parts["uav_queue_arrival_steps"]) - 5.0) < 1e-9
    assert abs(float(parts["sat_queue_arrival_steps"]) - 1.0) < 1e-9
    assert abs(float(parts["queue_pen_gu"]) - 2.0) < 1e-9
    assert abs(float(parts["queue_pen_uav"]) - 5.0) < 1e-9
    assert abs(float(parts["queue_pen_sat"]) - 1.0) < 1e-9
    assert abs(float(parts["queue_pen"]) - (4.75 / 1.75)) < 1e-9
    assert abs(float(parts["queue_delta_gu"]) - 1.0) < 1e-9
    assert abs(float(parts["queue_delta_uav"]) + 1.0) < 1e-9
    assert abs(float(parts["queue_delta_sat"]) - 1.0) < 1e-9
    assert parts["queue_delta_mode"] == "weighted"
    assert abs(float(parts["queue_delta"]) - (0.75 / 1.75)) < 1e-9
    assert abs(float(parts["term_queue"]) + (4.75 / 1.75)) < 1e-9
    assert abs(float(parts["term_q_delta"]) - (0.75 / 1.75)) < 1e-9


def test_avoidance_linear_repulsion_and_clip():
    cfg = SaginConfig(
        num_uav=2,
        num_gu=2,
        num_sat=3,
        users_obs_max=2,
        sats_obs_max=3,
        nbrs_obs_max=1,
        avoidance_enabled=True,
        avoidance_eta=3.0,
        avoidance_alert_factor=1.5,
        avoidance_repulse_mode="linear",
        avoidance_repulse_clip=True,
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.uav_pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    env.uav_pos[1] = np.array([110.0, 100.0], dtype=np.float32)
    env.uav_vel[:] = 0.0
    actions = {
        agent: {
            "accel": np.zeros(2, dtype=np.float32),
            "bw_logits": np.zeros(cfg.users_obs_max, dtype=np.float32),
            "sat_logits": np.zeros(cfg.sats_obs_max, dtype=np.float32),
        }
        for agent in env.agents
    }
    env._apply_uav_dynamics(actions)
    accel = env.last_exec_accel
    assert np.max(np.abs(accel)) <= cfg.a_max + 1e-6
    assert accel[0, 0] < 0.0
    assert accel[1, 0] > 0.0


def test_avoidance_prealert_triggers_for_fast_closing_pair():
    cfg = SaginConfig(
        num_uav=2,
        num_gu=2,
        num_sat=3,
        users_obs_max=2,
        sats_obs_max=3,
        nbrs_obs_max=1,
        avoidance_enabled=True,
        avoidance_eta=3.0,
        avoidance_alert_factor=2.0,
        avoidance_prealert_factor=6.0,
        avoidance_prealert_closing_speed=5.0,
        avoidance_repulse_mode="linear",
        avoidance_repulse_clip=True,
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.uav_pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    env.uav_pos[1] = np.array([200.0, 100.0], dtype=np.float32)
    env.uav_vel[0] = np.array([15.0, 0.0], dtype=np.float32)
    env.uav_vel[1] = np.array([-15.0, 0.0], dtype=np.float32)
    actions = {
        agent: {
            "accel": np.zeros(2, dtype=np.float32),
            "bw_logits": np.zeros(cfg.users_obs_max, dtype=np.float32),
            "sat_logits": np.zeros(cfg.sats_obs_max, dtype=np.float32),
        }
        for agent in env.agents
    }
    env._apply_uav_dynamics(actions)
    accel = env.last_exec_accel
    assert accel[0, 0] < 0.0
    assert accel[1, 0] > 0.0


def test_avoidance_prealert_does_not_trigger_for_non_closing_pair():
    cfg = SaginConfig(
        num_uav=2,
        num_gu=2,
        num_sat=3,
        users_obs_max=2,
        sats_obs_max=3,
        nbrs_obs_max=1,
        avoidance_enabled=True,
        avoidance_eta=3.0,
        avoidance_alert_factor=2.0,
        avoidance_prealert_factor=6.0,
        avoidance_prealert_closing_speed=5.0,
        avoidance_repulse_mode="linear",
        avoidance_repulse_clip=True,
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.uav_pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    env.uav_pos[1] = np.array([200.0, 100.0], dtype=np.float32)
    env.uav_vel[0] = np.array([-10.0, 0.0], dtype=np.float32)
    env.uav_vel[1] = np.array([10.0, 0.0], dtype=np.float32)
    actions = {
        agent: {
            "accel": np.zeros(2, dtype=np.float32),
            "bw_logits": np.zeros(cfg.users_obs_max, dtype=np.float32),
            "sat_logits": np.zeros(cfg.sats_obs_max, dtype=np.float32),
        }
        for agent in env.agents
    }
    env._apply_uav_dynamics(actions)
    accel = env.last_exec_accel
    assert np.allclose(accel, 0.0, atol=1e-6)


def test_danger_imitation_mask_tracks_exec_policy_delta():
    cfg = SaginConfig(
        num_uav=2,
        num_gu=2,
        num_sat=3,
        users_obs_max=2,
        sats_obs_max=3,
        nbrs_obs_max=1,
        avoidance_enabled=True,
        avoidance_eta=3.0,
        avoidance_alert_factor=1.5,
        avoidance_repulse_mode="linear",
        avoidance_repulse_clip=True,
        danger_imitation_enabled=True,
        danger_imitation_intervention_thresh=0.05,
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.uav_pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    env.uav_pos[1] = np.array([110.0, 100.0], dtype=np.float32)
    env.uav_vel[:] = 0.0
    actions = {
        agent: {
            "accel": np.zeros(2, dtype=np.float32),
            "bw_logits": np.zeros(cfg.users_obs_max, dtype=np.float32),
            "sat_logits": np.zeros(cfg.sats_obs_max, dtype=np.float32),
        }
        for agent in env.agents
    }

    env._apply_uav_dynamics(actions)
    env._compute_reward()
    parts = env.last_reward_parts
    delta = env.last_exec_accel - env.last_policy_accel
    delta_norms = np.linalg.norm(delta, axis=1)
    expected_mean = float(np.mean(delta_norms)) / (cfg.a_max + 1e-9)
    expected_top1 = float(np.max(delta_norms)) / (cfg.a_max + 1e-9)
    expected_rate = float(np.mean(delta_norms > 1e-6))

    assert abs(float(parts["intervention_norm"]) - expected_mean) < 1e-6
    assert abs(float(parts["intervention_norm_top1"]) - expected_top1) < 1e-6
    assert abs(float(parts["intervention_rate"]) - expected_rate) < 1e-6
    assert abs(float(parts["danger_imitation_active_rate"]) - 1.0) < 1e-6
    assert np.allclose(env.last_danger_imitation_mask, 1.0, atol=1e-6)


def test_danger_imitation_mask_can_trigger_from_close_risk_without_intervention():
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
        danger_imitation_enabled=True,
        danger_imitation_close_risk_thresh=0.05,
        danger_imitation_intervention_thresh=0.05,
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.uav_pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    env.uav_pos[1] = np.array([200.0, 100.0], dtype=np.float32)
    env.uav_vel[0] = np.array([15.0, 0.0], dtype=np.float32)
    env.uav_vel[1] = np.array([-15.0, 0.0], dtype=np.float32)
    actions = {
        agent: {
            "accel": np.zeros(2, dtype=np.float32),
            "bw_logits": np.zeros(cfg.users_obs_max, dtype=np.float32),
            "sat_logits": np.zeros(cfg.sats_obs_max, dtype=np.float32),
        }
        for agent in env.agents
    }

    env._apply_uav_dynamics(actions)
    env._compute_reward()
    parts = env.last_reward_parts

    assert abs(float(parts["intervention_rate"])) < 1e-9
    assert abs(float(parts["danger_imitation_active_rate"]) - 1.0) < 1e-6
    assert np.allclose(env.last_danger_imitation_mask, 1.0, atol=1e-6)


def test_danger_imitation_intervention_any_ignores_close_risk_only_case():
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
        danger_imitation_enabled=True,
        danger_imitation_trigger_mode="intervention_any",
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.uav_pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    env.uav_pos[1] = np.array([200.0, 100.0], dtype=np.float32)
    env.uav_vel[0] = np.array([15.0, 0.0], dtype=np.float32)
    env.uav_vel[1] = np.array([-15.0, 0.0], dtype=np.float32)

    env._compute_reward()
    parts = env.last_reward_parts

    assert abs(float(parts["intervention_rate"])) < 1e-9
    assert abs(float(parts["danger_imitation_active_rate"])) < 1e-9
    assert np.allclose(env.last_danger_imitation_mask, 0.0, atol=1e-6)


def test_danger_imitation_intervention_threshold_respects_eps():
    cfg = SaginConfig(
        num_uav=2,
        num_gu=2,
        num_sat=3,
        users_obs_max=2,
        sats_obs_max=3,
        nbrs_obs_max=1,
        danger_imitation_enabled=True,
        danger_imitation_trigger_mode="intervention_threshold",
        danger_imitation_intervention_thresh=0.25,
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.last_policy_accel[:] = 0.0
    env.last_exec_accel[:] = 0.0
    env.last_exec_accel[0, 0] = 1.0  # normalized intervention = 0.2 when a_max=5

    env._compute_reward()
    parts = env.last_reward_parts

    assert abs(float(parts["intervention_norm_top1"]) - 0.2) < 1e-6
    assert abs(float(parts["danger_imitation_active_rate"])) < 1e-9
    assert np.allclose(env.last_danger_imitation_mask, 0.0, atol=1e-6)

    env.cfg.danger_imitation_intervention_thresh = 0.15
    env._compute_reward()
    parts = env.last_reward_parts
    assert abs(float(parts["danger_imitation_active_rate"]) - 0.5) < 1e-6
    assert np.allclose(env.last_danger_imitation_mask, np.array([1.0, 0.0], dtype=np.float32), atol=1e-6)


def test_close_risk_reward_penalizes_fast_closing_pair():
    cfg = SaginConfig(
        num_uav=2,
        num_gu=2,
        num_sat=3,
        users_obs_max=2,
        sats_obs_max=3,
        nbrs_obs_max=1,
        avoidance_alert_factor=2.0,
        avoidance_prealert_factor=6.0,
        avoidance_prealert_closing_speed=5.0,
        avoidance_prealert_mode="distance",
        close_risk_enabled=True,
        eta_close_risk=0.02,
        close_risk_cap=2.0,
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.uav_pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    env.uav_pos[1] = np.array([200.0, 100.0], dtype=np.float32)
    env.uav_vel[0] = np.array([15.0, 0.0], dtype=np.float32)
    env.uav_vel[1] = np.array([-15.0, 0.0], dtype=np.float32)

    env._compute_reward()
    parts = env.last_reward_parts
    assert abs(float(parts["close_risk"]) - 0.5) < 1e-6
    assert abs(float(parts["term_close_risk"]) + 0.01) < 1e-6


def test_close_risk_reward_requires_closing_motion():
    cfg = SaginConfig(
        num_uav=2,
        num_gu=2,
        num_sat=3,
        users_obs_max=2,
        sats_obs_max=3,
        nbrs_obs_max=1,
        avoidance_alert_factor=2.0,
        avoidance_prealert_factor=6.0,
        avoidance_prealert_closing_speed=5.0,
        avoidance_prealert_mode="distance",
        close_risk_enabled=True,
        eta_close_risk=0.02,
        close_risk_cap=2.0,
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.uav_pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    env.uav_pos[1] = np.array([200.0, 100.0], dtype=np.float32)
    env.uav_vel[0] = np.array([-15.0, 0.0], dtype=np.float32)
    env.uav_vel[1] = np.array([15.0, 0.0], dtype=np.float32)

    env._compute_reward()
    parts = env.last_reward_parts
    assert abs(float(parts["close_risk"])) < 1e-9
    assert abs(float(parts["term_close_risk"])) < 1e-9


def test_danger_neighbor_obs_prefers_fast_closing_prealert_pair():
    cfg = SaginConfig(
        num_uav=3,
        num_gu=2,
        num_sat=3,
        users_obs_max=2,
        sats_obs_max=3,
        nbrs_obs_max=2,
        danger_nbr_enabled=True,
        avoidance_enabled=True,
        avoidance_alert_factor=2.0,
        avoidance_prealert_factor=6.0,
        avoidance_prealert_closing_speed=5.0,
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.uav_pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    env.uav_pos[1] = np.array([200.0, 100.0], dtype=np.float32)
    env.uav_pos[2] = np.array([130.0, 180.0], dtype=np.float32)
    env.uav_vel[0] = np.array([15.0, 0.0], dtype=np.float32)
    env.uav_vel[1] = np.array([-15.0, 0.0], dtype=np.float32)
    env.uav_vel[2] = np.array([0.0, 0.0], dtype=np.float32)

    feat = env._danger_neighbor_obs(0)

    expected_dist = 100.0 / cfg.map_size
    expected_closing = 30.0 / cfg.v_max
    np.testing.assert_allclose(feat[0], expected_dist, atol=1e-6)
    np.testing.assert_allclose(feat[1], expected_closing, atol=1e-6)
    np.testing.assert_allclose(feat[2:4], np.array([1.0, 0.0], dtype=np.float32), atol=1e-6)
    assert feat[4] == 1.0


def test_avoidance_closing_gain_strengthens_faster_prealert_repulsion():
    cfg = SaginConfig(
        num_uav=2,
        num_gu=2,
        num_sat=3,
        users_obs_max=2,
        sats_obs_max=3,
        nbrs_obs_max=1,
        avoidance_enabled=True,
        avoidance_eta=3.0,
        avoidance_alert_factor=2.0,
        avoidance_prealert_factor=6.0,
        avoidance_prealert_closing_speed=5.0,
        avoidance_repulse_mode="linear",
        avoidance_repulse_clip=True,
        avoidance_closing_gain_enabled=True,
        avoidance_closing_gain_cap=3.0,
    )
    actions = {
        f"uav_{idx}": {
            "accel": np.zeros(2, dtype=np.float32),
            "bw_logits": np.zeros(cfg.users_obs_max, dtype=np.float32),
            "sat_logits": np.zeros(cfg.sats_obs_max, dtype=np.float32),
        }
        for idx in range(cfg.num_uav)
    }

    env_slow = SaginParallelEnv(cfg)
    env_slow.reset()
    env_slow.uav_pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    env_slow.uav_pos[1] = np.array([200.0, 100.0], dtype=np.float32)
    env_slow.uav_vel[0] = np.array([4.0, 0.0], dtype=np.float32)
    env_slow.uav_vel[1] = np.array([-4.0, 0.0], dtype=np.float32)
    env_slow._apply_uav_dynamics(actions)

    env_fast = SaginParallelEnv(cfg)
    env_fast.reset()
    env_fast.uav_pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    env_fast.uav_pos[1] = np.array([200.0, 100.0], dtype=np.float32)
    env_fast.uav_vel[0] = np.array([6.0, 0.0], dtype=np.float32)
    env_fast.uav_vel[1] = np.array([-6.0, 0.0], dtype=np.float32)
    env_fast._apply_uav_dynamics(actions)

    slow_mag = abs(float(env_slow.last_exec_accel[0, 0]))
    fast_mag = abs(float(env_fast.last_exec_accel[0, 0]))
    assert abs(slow_mag - 0.96) < 1e-5
    assert abs(fast_mag - 1.44) < 1e-5
    assert fast_mag > slow_mag


def test_avoidance_closing_gain_top1_only_limits_extra_boost_to_single_neighbor():
    cfg_all = SaginConfig(
        num_uav=3,
        num_gu=3,
        num_sat=3,
        users_obs_max=3,
        sats_obs_max=3,
        nbrs_obs_max=2,
        avoidance_enabled=True,
        avoidance_eta=3.0,
        avoidance_alert_factor=2.0,
        avoidance_prealert_factor=6.0,
        avoidance_prealert_closing_speed=5.0,
        avoidance_repulse_mode="linear",
        avoidance_repulse_clip=True,
        avoidance_closing_gain_enabled=True,
        avoidance_closing_gain_cap=3.0,
        avoidance_closing_gain_top1_only=False,
    )
    cfg_top1 = SaginConfig(**{**cfg_all.__dict__, "avoidance_closing_gain_top1_only": True})
    actions = {
        f"uav_{idx}": {
            "accel": np.zeros(2, dtype=np.float32),
            "bw_logits": np.zeros(cfg_all.users_obs_max, dtype=np.float32),
            "sat_logits": np.zeros(cfg_all.sats_obs_max, dtype=np.float32),
        }
        for idx in range(cfg_all.num_uav)
    }

    env_all = SaginParallelEnv(cfg_all)
    env_all.reset()
    env_all.uav_pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    env_all.uav_pos[1] = np.array([200.0, 100.0], dtype=np.float32)
    env_all.uav_pos[2] = np.array([180.0, 100.0], dtype=np.float32)
    env_all.uav_vel[0] = np.array([6.0, 0.0], dtype=np.float32)
    env_all.uav_vel[1] = np.array([-6.0, 0.0], dtype=np.float32)
    env_all.uav_vel[2] = np.array([-2.0, 0.0], dtype=np.float32)
    env_all._apply_uav_dynamics(actions)

    env_top1 = SaginParallelEnv(cfg_top1)
    env_top1.reset()
    env_top1.uav_pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    env_top1.uav_pos[1] = np.array([200.0, 100.0], dtype=np.float32)
    env_top1.uav_pos[2] = np.array([180.0, 100.0], dtype=np.float32)
    env_top1.uav_vel[0] = np.array([6.0, 0.0], dtype=np.float32)
    env_top1.uav_vel[1] = np.array([-6.0, 0.0], dtype=np.float32)
    env_top1.uav_vel[2] = np.array([-2.0, 0.0], dtype=np.float32)
    env_top1._apply_uav_dynamics(actions)

    assert abs(float(env_all.last_exec_accel[0, 0]) + 3.36) < 1e-5
    assert abs(float(env_top1.last_exec_accel[0, 0]) + 2.64) < 1e-5
    assert abs(float(env_top1.last_exec_accel[0, 0])) < abs(float(env_all.last_exec_accel[0, 0]))
    assert abs(float(env_top1.last_exec_accel[0, 0])) > 1.8


def test_avoidance_ttc_prealert_triggers_before_legacy_distance_threshold():
    cfg = SaginConfig(
        num_uav=2,
        num_gu=2,
        num_sat=3,
        users_obs_max=2,
        sats_obs_max=3,
        nbrs_obs_max=1,
        avoidance_enabled=True,
        avoidance_eta=3.0,
        avoidance_alert_factor=2.0,
        avoidance_prealert_factor=6.0,
        avoidance_prealert_closing_speed=0.0,
        avoidance_prealert_mode="ttc",
        avoidance_prealert_ttc=3.0,
        avoidance_prealert_dist_cap=200.0,
        avoidance_repulse_mode="linear",
        avoidance_repulse_clip=True,
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.uav_pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    env.uav_pos[1] = np.array([280.0, 100.0], dtype=np.float32)
    env.uav_vel[0] = np.array([30.0, 0.0], dtype=np.float32)
    env.uav_vel[1] = np.array([-30.0, 0.0], dtype=np.float32)
    actions = {
        agent: {
            "accel": np.zeros(2, dtype=np.float32),
            "bw_logits": np.zeros(cfg.users_obs_max, dtype=np.float32),
            "sat_logits": np.zeros(cfg.sats_obs_max, dtype=np.float32),
        }
        for agent in env.agents
    }
    env._apply_uav_dynamics(actions)
    accel = env.last_exec_accel
    assert accel[0, 0] < 0.0
    assert accel[1, 0] > 0.0


def test_avoidance_ttc_prealert_respects_ttc_horizon():
    cfg = SaginConfig(
        num_uav=2,
        num_gu=2,
        num_sat=3,
        users_obs_max=2,
        sats_obs_max=3,
        nbrs_obs_max=1,
        avoidance_enabled=True,
        avoidance_eta=3.0,
        avoidance_alert_factor=2.0,
        avoidance_prealert_closing_speed=0.0,
        avoidance_prealert_mode="ttc",
        avoidance_prealert_ttc=3.0,
        avoidance_prealert_dist_cap=200.0,
        avoidance_repulse_mode="linear",
        avoidance_repulse_clip=True,
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.uav_pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    env.uav_pos[1] = np.array([280.0, 100.0], dtype=np.float32)
    env.uav_vel[0] = np.array([10.0, 0.0], dtype=np.float32)
    env.uav_vel[1] = np.array([-10.0, 0.0], dtype=np.float32)
    actions = {
        agent: {
            "accel": np.zeros(2, dtype=np.float32),
            "bw_logits": np.zeros(cfg.users_obs_max, dtype=np.float32),
            "sat_logits": np.zeros(cfg.sats_obs_max, dtype=np.float32),
        }
        for agent in env.agents
    }
    env._apply_uav_dynamics(actions)
    accel = env.last_exec_accel
    assert np.allclose(accel, 0.0, atol=1e-6)


def test_avoidance_ttc_prealert_respects_dist_cap():
    cfg = SaginConfig(
        num_uav=2,
        num_gu=2,
        num_sat=3,
        users_obs_max=2,
        sats_obs_max=3,
        nbrs_obs_max=1,
        avoidance_enabled=True,
        avoidance_eta=3.0,
        avoidance_alert_factor=2.0,
        avoidance_prealert_closing_speed=0.0,
        avoidance_prealert_mode="ttc",
        avoidance_prealert_ttc=3.0,
        avoidance_prealert_dist_cap=200.0,
        avoidance_repulse_mode="linear",
        avoidance_repulse_clip=True,
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.uav_pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    env.uav_pos[1] = np.array([340.0, 100.0], dtype=np.float32)
    env.uav_vel[0] = np.array([40.0, 0.0], dtype=np.float32)
    env.uav_vel[1] = np.array([-40.0, 0.0], dtype=np.float32)
    actions = {
        agent: {
            "accel": np.zeros(2, dtype=np.float32),
            "bw_logits": np.zeros(cfg.users_obs_max, dtype=np.float32),
            "sat_logits": np.zeros(cfg.sats_obs_max, dtype=np.float32),
        }
        for agent in env.agents
    }
    env._apply_uav_dynamics(actions)
    accel = env.last_exec_accel
    assert np.allclose(accel, 0.0, atol=1e-6)


def test_boundary_hard_filter_projects_next_step_inside_margin():
    cfg = SaginConfig(
        num_uav=1,
        num_gu=2,
        num_sat=3,
        users_obs_max=2,
        sats_obs_max=3,
        nbrs_obs_max=1,
        boundary_mode="reflect",
        boundary_hard_filter_enabled=True,
        boundary_margin=30.0,
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.uav_pos[0] = np.array([34.0, 100.0], dtype=np.float32)
    env.uav_vel[0] = np.array([-5.0, 0.0], dtype=np.float32)
    actions = {
        agent: {
            "accel": np.zeros(2, dtype=np.float32),
            "bw_logits": np.zeros(cfg.users_obs_max, dtype=np.float32),
            "sat_logits": np.zeros(cfg.sats_obs_max, dtype=np.float32),
        }
        for agent in env.agents
    }
    env._apply_uav_dynamics(actions)
    assert abs(float(env.last_exec_accel[0, 0]) - 1.0) < 1e-6
    assert abs(float(env.uav_pos[0, 0]) - 30.0) < 1e-6
    assert abs(float(env.uav_vel[0, 0]) + 4.0) < 1e-6
    assert abs(float(env.last_filter_active_ratio) - 1.0) < 1e-6
    assert abs(float(env.last_boundary_filter_count) - 1.0) < 1e-6
    assert abs(float(env.last_fallback_count) - 0.0) < 1e-6


def test_boundary_hard_filter_uses_inward_fallback_when_margin_is_infeasible():
    cfg = SaginConfig(
        num_uav=1,
        num_gu=2,
        num_sat=3,
        users_obs_max=2,
        sats_obs_max=3,
        nbrs_obs_max=1,
        boundary_mode="reflect",
        boundary_hard_filter_enabled=True,
        boundary_margin=30.0,
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.uav_pos[0] = np.array([31.0, 100.0], dtype=np.float32)
    env.uav_vel[0] = np.array([-10.0, 0.0], dtype=np.float32)
    actions = {
        agent: {
            "accel": np.zeros(2, dtype=np.float32),
            "bw_logits": np.zeros(cfg.users_obs_max, dtype=np.float32),
            "sat_logits": np.zeros(cfg.sats_obs_max, dtype=np.float32),
        }
        for agent in env.agents
    }
    env._apply_uav_dynamics(actions)
    assert abs(float(env.last_exec_accel[0, 0]) - cfg.a_max) < 1e-6
    assert abs(float(env.uav_vel[0, 0]) + 5.0) < 1e-6
    assert abs(float(env.uav_pos[0, 0]) - 26.0) < 1e-6
    assert abs(float(env.last_filter_active_ratio) - 1.0) < 1e-6
    assert abs(float(env.last_boundary_filter_count) - 1.0) < 1e-6
    assert abs(float(env.last_fallback_count) - 1.0) < 1e-6


def test_pairwise_hard_filter_projects_most_dangerous_pair_to_d_hard():
    cfg = SaginConfig(
        num_uav=2,
        num_gu=2,
        num_sat=3,
        users_obs_max=2,
        sats_obs_max=3,
        nbrs_obs_max=1,
        pairwise_hard_filter_enabled=True,
        pairwise_hard_distance=25.0,
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.uav_pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    env.uav_pos[1] = np.array([124.0, 100.0], dtype=np.float32)
    env.uav_vel[:] = 0.0
    actions = {
        agent: {
            "accel": np.zeros(2, dtype=np.float32),
            "bw_logits": np.zeros(cfg.users_obs_max, dtype=np.float32),
            "sat_logits": np.zeros(cfg.sats_obs_max, dtype=np.float32),
        }
        for agent in env.agents
    }
    env._apply_uav_dynamics(actions)
    assert abs(float(env.uav_pos[0, 0]) - 99.5) < 1e-6
    assert abs(float(env.uav_pos[1, 0]) - 124.5) < 1e-6
    assert abs(float(np.linalg.norm(env.uav_pos[0] - env.uav_pos[1])) - 25.0) < 1e-6
    assert env.last_exec_accel[0, 0] < 0.0
    assert env.last_exec_accel[1, 0] > 0.0
    assert abs(float(env.last_pairwise_filter_count) - 1.0) < 1e-6
    assert abs(float(env.last_pairwise_filter_active_ratio) - 1.0) < 1e-6
    assert abs(float(env.last_pairwise_fallback_count) - 0.0) < 1e-6
    assert abs(float(env.last_pairwise_candidate_infeasible_count) - 0.0) < 1e-6
    assert float(env.last_pairwise_projected_delta_norm) > 0.4


def test_pairwise_hard_filter_uses_fallback_when_one_step_separation_is_infeasible():
    cfg = SaginConfig(
        num_uav=2,
        num_gu=2,
        num_sat=3,
        users_obs_max=2,
        sats_obs_max=3,
        nbrs_obs_max=1,
        pairwise_hard_filter_enabled=True,
        pairwise_hard_distance=25.0,
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.uav_pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    env.uav_pos[1] = np.array([122.0, 100.0], dtype=np.float32)
    env.uav_vel[0] = np.array([10.0, 0.0], dtype=np.float32)
    env.uav_vel[1] = np.array([-10.0, 0.0], dtype=np.float32)
    actions = {
        agent: {
            "accel": np.zeros(2, dtype=np.float32),
            "bw_logits": np.zeros(cfg.users_obs_max, dtype=np.float32),
            "sat_logits": np.zeros(cfg.sats_obs_max, dtype=np.float32),
        }
        for agent in env.agents
    }
    env._apply_uav_dynamics(actions)
    assert abs(float(env.last_exec_accel[0, 0]) + cfg.a_max) < 1e-6
    assert abs(float(env.last_exec_accel[1, 0]) - cfg.a_max) < 1e-6
    assert abs(float(np.linalg.norm(env.uav_pos[0] - env.uav_pos[1])) - 12.0) < 1e-6
    assert abs(float(env.last_pairwise_filter_count) - 1.0) < 1e-6
    assert abs(float(env.last_pairwise_filter_active_ratio) - 1.0) < 1e-6
    assert abs(float(env.last_pairwise_fallback_count) - 1.0) < 1e-6
    assert abs(float(env.last_pairwise_candidate_infeasible_count) - 1.0) < 1e-6
    assert abs(float(env.last_fallback_count) - 1.0) < 1e-6


def test_pairwise_ttc_filter_triggers_before_next_step_distance_violation():
    cfg = SaginConfig(
        num_uav=2,
        num_gu=2,
        num_sat=3,
        users_obs_max=2,
        sats_obs_max=3,
        nbrs_obs_max=1,
        pairwise_hard_filter_enabled=True,
        pairwise_hard_distance=25.0,
        pairwise_hard_trigger_mode="ttc",
        pairwise_hard_trigger_ttc=2.0,
        pairwise_hard_trigger_distance=80.0,
        pairwise_hard_closing_speed=0.0,
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.uav_pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    env.uav_pos[1] = np.array([150.0, 100.0], dtype=np.float32)
    env.uav_vel[0] = np.array([8.0, 0.0], dtype=np.float32)
    env.uav_vel[1] = np.array([-8.0, 0.0], dtype=np.float32)
    actions = {
        agent: {
            "accel": np.zeros(2, dtype=np.float32),
            "bw_logits": np.zeros(cfg.users_obs_max, dtype=np.float32),
            "sat_logits": np.zeros(cfg.sats_obs_max, dtype=np.float32),
        }
        for agent in env.agents
    }
    env._apply_uav_dynamics(actions)
    assert abs(float(env.last_exec_accel[0, 0]) + 1.75) < 1e-5
    assert abs(float(env.last_exec_accel[1, 0]) - 1.75) < 1e-5
    assert abs(float(env.uav_pos[0, 0]) - 106.25) < 1e-5
    assert abs(float(env.uav_pos[1, 0]) - 143.75) < 1e-5
    assert abs(float(np.linalg.norm(env.uav_pos[0] - env.uav_pos[1])) - 37.5) < 1e-5
    assert abs(float(env.last_pairwise_filter_count) - 1.0) < 1e-6
    assert abs(float(env.last_pairwise_fallback_count) - 0.0) < 1e-6
    assert abs(float(env.last_pairwise_candidate_infeasible_count) - 0.0) < 1e-6


def test_pairwise_ttc_filter_ignores_opening_pair():
    cfg = SaginConfig(
        num_uav=2,
        num_gu=2,
        num_sat=3,
        users_obs_max=2,
        sats_obs_max=3,
        nbrs_obs_max=1,
        pairwise_hard_filter_enabled=True,
        pairwise_hard_distance=25.0,
        pairwise_hard_trigger_mode="ttc",
        pairwise_hard_trigger_ttc=2.0,
        pairwise_hard_trigger_distance=80.0,
        pairwise_hard_closing_speed=0.0,
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.uav_pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    env.uav_pos[1] = np.array([150.0, 100.0], dtype=np.float32)
    env.uav_vel[0] = np.array([-8.0, 0.0], dtype=np.float32)
    env.uav_vel[1] = np.array([8.0, 0.0], dtype=np.float32)
    actions = {
        agent: {
            "accel": np.zeros(2, dtype=np.float32),
            "bw_logits": np.zeros(cfg.users_obs_max, dtype=np.float32),
            "sat_logits": np.zeros(cfg.sats_obs_max, dtype=np.float32),
        }
        for agent in env.agents
    }
    env._apply_uav_dynamics(actions)
    assert np.allclose(env.last_exec_accel, 0.0, atol=1e-6)
    assert abs(float(env.last_pairwise_filter_count) - 0.0) < 1e-6
    assert abs(float(env.last_pairwise_fallback_count) - 0.0) < 1e-6
    assert abs(float(env.last_pairwise_candidate_infeasible_count) - 0.0) < 1e-6


def test_pairwise_ttc_filter_only_adjusts_most_dangerous_pair_for_three_uavs():
    cfg = SaginConfig(
        num_uav=3,
        num_gu=3,
        num_sat=3,
        users_obs_max=3,
        sats_obs_max=3,
        nbrs_obs_max=2,
        pairwise_hard_filter_enabled=True,
        pairwise_hard_distance=25.0,
        pairwise_hard_trigger_mode="ttc",
        pairwise_hard_trigger_ttc=2.0,
        pairwise_hard_trigger_distance=80.0,
        pairwise_hard_closing_speed=0.0,
        pairwise_hard_single_pair_only=True,
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    env.uav_pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    env.uav_pos[1] = np.array([132.0, 100.0], dtype=np.float32)
    env.uav_pos[2] = np.array([170.0, 100.0], dtype=np.float32)
    env.uav_vel[0] = np.array([8.0, 0.0], dtype=np.float32)
    env.uav_vel[1] = np.array([0.0, 0.0], dtype=np.float32)
    env.uav_vel[2] = np.array([-8.0, 0.0], dtype=np.float32)
    actions = {
        agent: {
            "accel": np.zeros(2, dtype=np.float32),
            "bw_logits": np.zeros(cfg.users_obs_max, dtype=np.float32),
            "sat_logits": np.zeros(cfg.sats_obs_max, dtype=np.float32),
        }
        for agent in env.agents
    }
    env._apply_uav_dynamics(actions)
    assert abs(float(env.last_exec_accel[0, 0]) + 2.25) < 1e-5
    assert abs(float(env.last_exec_accel[1, 0]) - 2.25) < 1e-5
    assert np.allclose(env.last_exec_accel[2], 0.0, atol=1e-6)
    assert abs(float(env.last_pairwise_filter_count) - 1.0) < 1e-6
    assert abs(float(env.last_pairwise_filter_active_ratio) - (1.0 / 3.0)) < 1e-6


def test_centroid_cross_anneal_transfers_weights():
    cfg = SaginConfig(
        num_uav=2,
        num_gu=4,
        num_sat=3,
        users_obs_max=4,
        sats_obs_max=3,
        nbrs_obs_max=1,
        queue_delta_use_active=True,
        omega_q=2.0,
        eta_q_delta=1.5,
        eta_crash=4.0,
        eta_centroid=0.4,
        eta_centroid_final=0.0,
        eta_centroid_decay_steps=2,
        centroid_cross_anneal_enabled=True,
        centroid_cross_queue_gain=1.0,
        centroid_cross_q_delta_gain=1.0,
        centroid_cross_crash_gain=1.0,
        centroid_cross_avoidance_gain=1.0,
        avoidance_enabled=True,
        avoidance_eta=2.0,
        avoidance_eta_min=0.0,
        avoidance_eta_max=5.0,
    )
    env = SaginParallelEnv(cfg)
    env.reset()
    actions = {
        agent: {
            "accel": np.zeros(2, dtype=np.float32),
            "bw_logits": np.zeros(cfg.users_obs_max, dtype=np.float32),
            "sat_logits": np.zeros(cfg.sats_obs_max, dtype=np.float32),
        }
        for agent in env.agents
    }
    env.step(actions)
    parts = env.last_reward_parts
    # global_step=1 with decay_steps=2 => centroid transfer ratio = 0.5
    assert abs(float(parts["centroid_transfer_ratio"]) - 0.5) < 1e-6
    assert abs(float(parts["queue_weight"]) - 3.0) < 1e-6
    assert abs(float(parts["q_delta_weight"]) - 2.25) < 1e-6
    assert abs(float(parts["crash_weight"]) - 6.0) < 1e-6
    assert abs(float(parts["avoidance_eta_exec"]) - 3.0) < 1e-6
