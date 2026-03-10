from __future__ import annotations

import numpy as np

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
