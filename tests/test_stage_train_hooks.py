from __future__ import annotations

import numpy as np

from sagin_marl.env.config import SaginConfig
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.vec_env import SyncVecSaginEnv, _collect_step_stats
from sagin_marl.rl.mappo import (
    _compute_train_reward_adjustment,
    _configure_actor_trainability,
    _estimate_aux_schedule_total_updates,
)
from sagin_marl.rl.policy import ActorNet, OWN_OBS_DIM, SAT_OBS_DIM, batch_flatten_obs


def _make_obs(cfg: SaginConfig) -> dict[str, np.ndarray]:
    return {
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


def test_stage1_assoc_centroid_reward_adjustment_anneals_to_zero():
    cfg = SaginConfig()
    cfg.reward_stage1_assoc_centroid_enabled = True
    cfg.reward_stage1_assoc_centroid_weight_init = 0.15
    cfg.reward_stage1_assoc_centroid_weight_mid = 0.05
    cfg.reward_stage1_assoc_centroid_weight_floor = 0.0
    cfg.reward_stage1_assoc_centroid_hold_ratio = 0.30
    cfg.reward_stage1_assoc_centroid_mid_ratio = 0.70

    stats = {
        "reward_parts": {"assoc_centroid_dist_norm_mean": 0.4},
        "assoc_centroid_dist_norms": np.array([0.3, 0.4, 0.5], dtype=np.float32),
    }
    reward0, info0 = _compute_train_reward_adjustment(stats, cfg, update_idx=0, total_updates=100)
    reward50, info50 = _compute_train_reward_adjustment(stats, cfg, update_idx=50, total_updates=100)
    reward99, info99 = _compute_train_reward_adjustment(stats, cfg, update_idx=99, total_updates=100)

    assert abs(info0["stage1_assoc_centroid_weight"] - 0.15) < 1e-9
    np.testing.assert_allclose(info0["stage1_assoc_centroid_dist_norm_uav"], np.array([0.3, 0.4, 0.5], dtype=np.float32))
    assert abs(reward0 + 0.06) < 1e-9
    assert abs(info50["stage1_assoc_centroid_weight"] - 0.0975) < 1e-9
    assert abs(reward50 + 0.039) < 1e-9
    assert abs(info99["stage1_assoc_centroid_weight"]) < 1e-9
    assert abs(reward99) < 1e-9


def test_stage3_sat_overlap_reward_adjustment_uses_overlap_metric():
    cfg = SaginConfig()
    cfg.reward_stage3_sat_overlap_enabled = True
    cfg.reward_stage3_sat_overlap_weight_init = 0.10
    cfg.reward_stage3_sat_overlap_weight_mid = 0.03
    cfg.reward_stage3_sat_overlap_weight_floor = 0.0
    cfg.reward_stage3_sat_overlap_hold_ratio = 0.40
    cfg.reward_stage3_sat_overlap_mid_ratio = 0.80

    stats = {
        "reward_parts": {"sat_overlap_eval": 0.25},
        "sat_overlap_uav": np.array([0.5, 0.25, 0.0], dtype=np.float32),
    }
    reward0, info0 = _compute_train_reward_adjustment(stats, cfg, update_idx=0, total_updates=100)
    reward99, info99 = _compute_train_reward_adjustment(stats, cfg, update_idx=99, total_updates=100)

    assert abs(info0["stage3_sat_overlap_weight"] - 0.10) < 1e-9
    np.testing.assert_allclose(info0["stage3_sat_overlap_uav"], np.array([0.5, 0.25, 0.0], dtype=np.float32))
    assert abs(reward0 + 0.025) < 1e-9
    assert abs(info99["stage3_sat_overlap_weight"]) < 1e-9
    assert abs(reward99) < 1e-9


def test_estimate_aux_schedule_total_updates_uses_checkpoint_eval_stop_horizon():
    cfg = SaginConfig(
        checkpoint_eval_enabled=True,
        checkpoint_eval_interval_updates=50,
        checkpoint_eval_start_update=200,
        checkpoint_eval_early_stop_enabled=True,
        checkpoint_eval_reward_early_stop_enabled=True,
        checkpoint_eval_reward_patience=5,
        checkpoint_eval_sat_drop_early_stop_enabled=False,
        early_stop_enabled=False,
    )

    assert _estimate_aux_schedule_total_updates(cfg, planned_total_updates=1500) == 450


def test_estimate_aux_schedule_total_updates_falls_back_to_planned_when_no_stop_rule_is_active():
    cfg = SaginConfig(
        checkpoint_eval_enabled=False,
        early_stop_enabled=False,
    )

    assert _estimate_aux_schedule_total_updates(cfg, planned_total_updates=1500) == 1500


def test_configure_actor_trainability_unfreezes_only_fusion_last_layer_for_stage2_style_setup():
    cfg = SaginConfig(users_obs_max=2, sats_obs_max=2, nbrs_obs_max=1)
    cfg.enable_bw_action = True
    cfg.fixed_satellite_strategy = True
    cfg.actor_encoder_type = "set_pool"
    cfg.actor_set_embed_dim = 8
    cfg.train_shared_backbone = False
    cfg.train_fusion_last_layer = True

    obs_dim = batch_flatten_obs([_make_obs(cfg)], cfg).shape[1]
    actor = ActorNet(obs_dim, cfg)
    _configure_actor_trainability(actor, cfg, {"accel": False, "bw": True, "sat": False})

    assert actor.log_std.requires_grad is False
    assert actor.mu_head.weight.requires_grad is False
    assert next(actor.own_encoder.parameters()).requires_grad is False
    assert actor.fusion_fc1.weight.requires_grad is False
    assert actor.fusion_fc2.weight.requires_grad is True
    assert next(actor.bw_user_encoder.parameters()).requires_grad is True
    assert next(actor.bw_scorer.parameters()).requires_grad is True


def test_vec_env_collect_step_stats_preserves_stage_aux_arrays():
    cfg = SaginConfig(num_uav=3, num_gu=6, num_sat=4, users_obs_max=6, sats_obs_max=4, nbrs_obs_max=2)
    env = SaginParallelEnv(cfg)
    try:
        obs, _ = env.reset(seed=123)
        zero_accel = np.zeros((cfg.num_uav, 2), dtype=np.float32)
        actions = {
            agent: {"accel": zero_accel[idx]}
            for idx, agent in enumerate(env.agents)
        }
        env.step(actions)
        stats = _collect_step_stats(env)

        np.testing.assert_allclose(
            stats["assoc_centroid_dist_norms"],
            env.last_assoc_centroid_dist_norms,
        )
        np.testing.assert_allclose(
            stats["sat_overlap_uav"],
            env.last_sat_overlap_uav,
        )
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def test_sync_vec_env_auto_reset_preserves_post_step_obs_for_bootstrap():
    cfg = SaginConfig(
        seed=7,
        T_steps=1,
        num_uav=1,
        num_gu=3,
        num_sat=4,
        users_obs_max=3,
        sats_obs_max=4,
        nbrs_obs_max=1,
        sat_num_select=1,
        enable_bw_action=True,
        fixed_satellite_strategy=True,
    )
    vec_env = SyncVecSaginEnv(cfg, num_envs=1)
    try:
        vec_env.reset(seeds=[cfg.seed])
        action = vec_env.envs[0]._dummy_actions()
        next_obs_batch, _, _, truncs_batch, _ = vec_env.step([action], auto_reset=True)
        assert bool(next(iter(truncs_batch[0].values())))
        post_step_obs = vec_env.last_step_stats[0]["post_step_obs"]
        assert isinstance(post_step_obs, dict)
        assert set(post_step_obs) == set(next_obs_batch[0])
        assert batch_flatten_obs(list(post_step_obs.values()), cfg).shape == batch_flatten_obs(
            list(next_obs_batch[0].values()),
            cfg,
        ).shape
    finally:
        vec_env.close()
