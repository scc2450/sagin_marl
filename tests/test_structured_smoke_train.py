from __future__ import annotations
import types
import numpy as np
import pytest
import torch

from sagin_marl.env.config import SaginConfig
from sagin_marl.rl.structured_critic import ZeroStructuredCritic
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_mappo import StructuredMAPPO, bw_actor_only_signal_critic_free_enabled
from sagin_marl.rl.structured_train import _make_episode_stats_state, make_structured_env_group, run_structured_training


def test_structured_smoke_train_preserves_episode_stats_across_short_rollouts():
    cfg = SaginConfig(
        seed=3,
        num_uav=1,
        num_gu=3,
        num_sat=6,
        users_obs_max=3,
        sats_obs_max=4,
        nbrs_obs_max=1,
        sat_num_select=1,
        T_steps=4,
    )
    bundle = build_structured_modules_from_config(cfg, hidden_dim=32, embed_dim=16)
    actor = bundle.actor
    critic = bundle.critic
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)
    learner = StructuredMAPPO(
        actor=actor,
        critic=critic,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=1,
        num_mini_batch=1,
        actor_optimizer=actor_optim,
        critic_optimizer=critic_optim,
    )

    env = make_structured_env_group(cfg, num_envs=1, backend="sync")
    episode_stats_state = _make_episode_stats_state(num_envs=1, episode_stat_window=8)
    try:
        first = run_structured_training(
            env,
            learner,
            num_updates=1,
            rollout_env_steps=2,
            reset_seed=int(cfg.seed),
            reset_on_start=True,
            episode_stats_state=episode_stats_state,
        )
        second = run_structured_training(
            env,
            learner,
            num_updates=1,
            rollout_env_steps=2,
            reset_seed=int(cfg.seed) + 1000,
            reset_on_start=False,
            episode_stats_state=episode_stats_state,
        )
        assert len(first) == 1
        assert len(second) == 1
        assert first[0].completed_episode_count == 0
        assert first[0].episodes_finished == 0
        assert second[0].completed_episode_count >= 1
        assert second[0].episodes_finished >= 1
        assert second[0].episode_length_mean == float(cfg.T_steps)
        assert np.isfinite(second[0].episode_reward)
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def test_structured_smoke_train_runs_sync_group_batch_backend():
    cfg = SaginConfig(
        seed=23,
        num_uav=2,
        num_gu=4,
        num_sat=6,
        users_obs_max=4,
        sats_obs_max=4,
        nbrs_obs_max=1,
        sat_num_select=2,
        T_steps=6,
        structured_env_tensor_backend="cuda",
    )
    bundle = build_structured_modules_from_config(cfg, hidden_dim=32, embed_dim=16)
    actor = bundle.actor
    critic = bundle.critic
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)
    learner = StructuredMAPPO(
        actor=actor,
        critic=critic,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=1,
        num_mini_batch=1,
        actor_optimizer=actor_optim,
        critic_optimizer=critic_optim,
        cfg=cfg,
    )
    env_group = make_structured_env_group(cfg, num_envs=2, backend="sync")
    try:
        history = run_structured_training(
            env_group,
            learner,
            num_updates=1,
            rollout_env_steps=2,
            reset_seed=int(cfg.seed),
        )
        assert len(history) == 1
        assert np.isfinite(history[0].env_reward_mean)
        assert np.isfinite(history[0].policy_loss)
        assert np.isfinite(history[0].value_loss)
    finally:
        close_fn = getattr(env_group, "close", None)
        if callable(close_fn):
            close_fn()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="final native CUDA rollout requires CUDA")
def test_structured_marginal_teacher_sample_bw_only_skips_critic_training():
    device = torch.device("cuda")
    cfg = SaginConfig(
        seed=18,
        num_uav=1,
        num_gu=4,
        num_sat=8,
        users_obs_max=4,
        sats_obs_max=4,
        nbrs_obs_max=1,
        sat_num_select=2,
        T_steps=6,
        b_acc=1.0e6,
        task_arrival_rate=1.0e6,
        traffic_model="sticky_subset_hotspot",
        hotspot_num_subsets=1,
        hotspot_subset_size=2,
        preload_enabled=True,
        preload_prob=1.0,
        preload_hot_gu_steps=10.0,
        preload_bg_gu_steps=2.0,
        train_accel=False,
        train_sat=False,
        train_bw=True,
        exec_accel_source="zero",
        exec_sat_source="zero",
        exec_bw_source="policy",
        reward_mode="weighted_workload_level",
        bw_train_target_mode="weighted_workload_level",
        bw_return_mode="bw_gae",
        bw_actor_advantage_override_mode="gae",
        bw_marginal_teacher_sample_enabled=True,
        bw_marginal_teacher_sample_weight=1.0,
        bw_flow_proxy_base_action_mode="deterministic",
        structured_env_tensor_backend="cuda",
    )
    assert bw_actor_only_signal_critic_free_enabled(
        cfg,
        train_accel=False,
        train_sat=False,
        train_bw=True,
    )
    bundle = build_structured_modules_from_config(cfg, hidden_dim=32, embed_dim=16, build_critic=False)
    actor = bundle.actor
    assert bundle.critic is None
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
    learner = StructuredMAPPO(
        actor=actor,
        critic=ZeroStructuredCritic(device),
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        max_grad_norm=0.5,
        ppo_epochs=1,
        num_mini_batch=1,
        actor_optimizer=actor_optim,
        critic_optimizer=None,
        device=device,
        target_mode="step_level",
        cfg=cfg,
        train_accel=False,
        train_sat=False,
        train_bw=True,
        exec_accel_source="zero",
        exec_sat_source="zero",
        exec_bw_source="policy",
    )
    env = make_structured_env_group(cfg, num_envs=1, backend="sync")
    try:
        rows = run_structured_training(
            env,
            learner,
            num_updates=1,
            rollout_env_steps=3,
            reset_seed=int(cfg.seed),
        )
        assert len(rows) == 1
        assert rows[0].value_loss == 0.0
        assert rows[0].value_loss_bw == 0.0
        assert rows[0].entropy_bw != 0.0
        assert learner.critic_optimizer is None
        assert isinstance(learner.critic, ZeroStructuredCritic)
        assert learner.disable_critic_training_for_bw_actor_only_signal
        assert learner.actor_optimizer is not None
        assert learner.actor_optimizer.state_dict()["state"]
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="final native CUDA rollout requires CUDA")
def test_structured_marginal_teacher_perslot_zero_base_bw_only_skips_critic_training():
    device = torch.device("cuda")
    cfg = SaginConfig(
        seed=19,
        num_uav=1,
        num_gu=4,
        num_sat=8,
        users_obs_max=4,
        sats_obs_max=4,
        nbrs_obs_max=1,
        sat_num_select=2,
        T_steps=6,
        b_acc=1.0e6,
        task_arrival_rate=1.0e6,
        traffic_model="sticky_subset_hotspot",
        hotspot_num_subsets=1,
        hotspot_subset_size=2,
        preload_enabled=True,
        preload_prob=1.0,
        preload_hot_gu_steps=10.0,
        preload_bg_gu_steps=2.0,
        train_accel=False,
        train_sat=False,
        train_bw=True,
        exec_accel_source="zero",
        exec_sat_source="zero",
        exec_bw_source="policy",
        reward_mode="weighted_workload_level",
        bw_train_target_mode="weighted_workload_level",
        bw_return_mode="bw_gae",
        bw_actor_advantage_override_mode="gae",
        structured_bw_per_slot_surrogate_enabled=True,
        bw_slot_advantage_base_mode="zero",
        bw_flow_proxy_base_action_mode="deterministic",
        structured_env_tensor_backend="cuda",
    )
    assert bw_actor_only_signal_critic_free_enabled(
        cfg,
        train_accel=False,
        train_sat=False,
        train_bw=True,
    )
    bundle = build_structured_modules_from_config(cfg, hidden_dim=32, embed_dim=16, build_critic=False)
    actor = bundle.actor
    assert bundle.critic is None
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
    learner = StructuredMAPPO(
        actor=actor,
        critic=ZeroStructuredCritic(device),
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        max_grad_norm=0.5,
        ppo_epochs=1,
        num_mini_batch=1,
        actor_optimizer=actor_optim,
        critic_optimizer=None,
        device=device,
        target_mode="step_level",
        cfg=cfg,
        train_accel=False,
        train_sat=False,
        train_bw=True,
        exec_accel_source="zero",
        exec_sat_source="zero",
        exec_bw_source="policy",
    )
    env = make_structured_env_group(cfg, num_envs=1, backend="sync")
    try:
        rows = run_structured_training(
            env,
            learner,
            num_updates=1,
            rollout_env_steps=3,
            reset_seed=int(cfg.seed),
        )
        assert len(rows) == 1
        assert rows[0].value_loss == 0.0
        assert rows[0].value_loss_bw == 0.0
        assert rows[0].entropy_bw != 0.0
        assert learner.critic_optimizer is None
        assert isinstance(learner.critic, ZeroStructuredCritic)
        assert learner.disable_critic_training_for_bw_actor_only_signal
        assert learner.actor_optimizer is not None
        assert learner.actor_optimizer.state_dict()["state"]
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def _fake_clean_targets_shift_mass(self, *, snapshot_states, ref_actions, valid_masks):
    del self, snapshot_states
    targets = np.asarray(ref_actions, dtype=np.float32).copy()
    rho_values = np.zeros((targets.shape[0],), dtype=np.float32)
    utility_l1 = np.zeros((targets.shape[0],), dtype=np.float32)
    for row_idx in range(targets.shape[0]):
        ref = np.asarray(targets[row_idx], dtype=np.float32).reshape(-1).copy()
        valid = np.asarray(valid_masks[row_idx], dtype=bool).reshape(-1)
        ref[~valid] = 0.0
        valid_slots = np.flatnonzero(valid)
        if valid_slots.size <= 1:
            targets[row_idx] = ref
            continue
        ref_sum = float(np.sum(ref[valid]))
        if ref_sum <= 1.0e-8:
            ref[valid] = 1.0 / float(valid_slots.size)
        else:
            ref[valid] = ref[valid] / ref_sum
        donor = int(valid_slots[np.argmax(ref[valid_slots])])
        receiver = int(valid_slots[np.argmin(ref[valid_slots])])
        if donor == receiver:
            receiver = int(valid_slots[-1])
            donor = int(valid_slots[0])
        delta = min(0.05, float(ref[donor]) * 0.5)
        if delta <= 1.0e-8:
            targets[row_idx] = ref
            continue
        target = ref.copy()
        target[donor] -= float(delta)
        target[receiver] += float(delta)
        targets[row_idx] = target.astype(np.float32, copy=False)
        rho_values[row_idx] = float(delta)
        utility_l1[row_idx] = float(delta)
    return targets.astype(np.float32, copy=False), rho_values, utility_l1


def _fake_clean_target_rollouts_positive(self, *, snapshot_states, first_actions):
    del self, snapshot_states
    return np.ones((len(first_actions),), dtype=np.float32)


def _fake_clean_target_rollouts_zero(self, *, snapshot_states, first_actions):
    del self, snapshot_states
    return np.zeros((len(first_actions),), dtype=np.float32)


@pytest.mark.parametrize("clean_loss", ["huber", "masked_kl"])
def test_structured_clean_per_user_bw_only_score_softmax_skips_critic_and_updates_actor(clean_loss: str):
    cfg = SaginConfig(
        seed=23,
        num_uav=1,
        num_gu=4,
        num_sat=8,
        users_obs_max=4,
        sats_obs_max=4,
        nbrs_obs_max=1,
        sat_num_select=2,
        T_steps=6,
        b_acc=1.0e6,
        task_arrival_rate=1.0e6,
        traffic_model="sticky_subset_hotspot",
        hotspot_num_subsets=1,
        hotspot_subset_size=2,
        preload_enabled=True,
        preload_prob=1.0,
        preload_hot_gu_steps=10.0,
        preload_bg_gu_steps=2.0,
        train_accel=False,
        train_sat=False,
        train_bw=True,
        exec_accel_source="zero",
        exec_sat_source="zero",
        exec_bw_source="policy",
        reward_mode="weighted_workload_level",
        bw_train_target_mode="weighted_workload_level",
        bw_return_mode="bw_gae",
        enable_bw_action=True,
        bw_clean_per_user_enabled=True,
        bw_clean_per_user_horizon=2,
        bw_clean_per_user_delta_probe=0.02,
        bw_clean_per_user_beta=0.5,
        bw_clean_per_user_loss=str(clean_loss),
        bw_clean_trust_region_enabled=True,
        bw_clean_trust_region_target_kl=0.01,
        bw_clean_trust_region_kl_coef_init=0.1,
        bw_clean_trust_region_kl_coef_min=1.0e-4,
        bw_clean_trust_region_kl_coef_max=1.0e3,
        bw_clean_trust_region_backtrack_factor=0.5,
        bw_clean_trust_region_max_backtracks=4,
    )
    assert bw_actor_only_signal_critic_free_enabled(
        cfg,
        train_accel=False,
        train_sat=False,
        train_bw=True,
    )
    bundle = build_structured_modules_from_config(cfg, hidden_dim=32, embed_dim=16, build_critic=False)
    actor = bundle.actor
    assert bundle.critic is None
    actor_before = {
        name: tensor.detach().cpu().clone()
        for name, tensor in actor.state_dict().items()
        if tensor.dtype.is_floating_point
    }
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
    learner = StructuredMAPPO(
        actor=actor,
        critic=ZeroStructuredCritic(torch.device("cpu")),
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        max_grad_norm=0.5,
        ppo_epochs=1,
        num_mini_batch=1,
        actor_optimizer=actor_optim,
        critic_optimizer=None,
        target_mode="step_level",
        cfg=cfg,
        train_accel=False,
        train_sat=False,
        train_bw=True,
        exec_accel_source="zero",
        exec_sat_source="zero",
        exec_bw_source="policy",
    )
    def _fake_clean_targets_positive(self, *, snapshot_states, ref_actions, valid_masks):
        targets, rho_values, utility_l1 = _fake_clean_targets_shift_mass(
            self,
            snapshot_states=snapshot_states,
            ref_actions=ref_actions,
            valid_masks=valid_masks,
        )
        self._bw_clean_last_ref_returns = np.zeros((targets.shape[0],), dtype=np.float32)
        return targets, rho_values, utility_l1

    learner._bw_clean_target_actions_parallel = types.MethodType(_fake_clean_targets_positive, learner)
    learner._bw_clean_rollout_returns_parallel = types.MethodType(_fake_clean_target_rollouts_positive, learner)
    env = make_structured_env_group(cfg, num_envs=1, backend="sync")
    try:
        rows = run_structured_training(
            env,
            learner,
            num_updates=1,
            rollout_env_steps=2,
            reset_seed=int(cfg.seed),
        )
        assert len(rows) == 1
        assert rows[0].value_loss == 0.0
        assert rows[0].value_loss_bw == 0.0
        assert rows[0].entropy == 0.0
        assert rows[0].entropy_bw == 0.0
        assert np.isfinite(rows[0].clean_mean_target_gap)
        assert float(rows[0].clean_mean_target_gap) >= 0.0
        assert np.isfinite(rows[0].clean_mean_update_shift)
        assert float(rows[0].clean_mean_update_shift) >= 0.0
        assert np.isfinite(rows[0].clean_update_to_target_ratio)
        assert 0.0 <= float(rows[0].clean_target_beats_ref_frac) <= 1.0
        assert np.isfinite(rows[0].clean_measured_kl)
        assert float(rows[0].clean_measured_kl) >= 0.0
        assert float(rows[0].clean_measured_kl) <= float(cfg.bw_clean_trust_region_target_kl) + 1.0e-6
        assert np.isfinite(rows[0].clean_kl_coef)
        assert float(rows[0].clean_kl_coef) > 0.0
        assert learner.critic_optimizer is None
        assert isinstance(learner.critic, ZeroStructuredCritic)
        assert learner.disable_critic_training_for_bw_actor_only_signal
        assert learner.bw_clean_per_user_enabled
        assert learner.actor_optimizer is not None
        assert learner.actor_optimizer.state_dict()["state"]
        actor_after = {name: tensor.detach().cpu() for name, tensor in actor.state_dict().items()}
        changed = any(
            tensor.dtype.is_floating_point and not torch.allclose(actor_before[name], actor_after[name])
            for name, tensor in actor_after.items()
            if name in actor_before
        )
        assert changed
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def test_structured_clean_per_user_trust_region_backtracks_under_large_lr():
    cfg = SaginConfig(
        seed=37,
        num_uav=1,
        num_gu=4,
        num_sat=8,
        users_obs_max=4,
        sats_obs_max=4,
        nbrs_obs_max=1,
        sat_num_select=2,
        T_steps=6,
        b_acc=1.0e6,
        task_arrival_rate=1.0e6,
        traffic_model="sticky_subset_hotspot",
        hotspot_num_subsets=1,
        hotspot_subset_size=2,
        preload_enabled=True,
        preload_prob=1.0,
        preload_hot_gu_steps=10.0,
        preload_bg_gu_steps=2.0,
        train_accel=False,
        train_sat=False,
        train_bw=True,
        exec_accel_source="zero",
        exec_sat_source="zero",
        exec_bw_source="policy",
        reward_mode="weighted_workload_level",
        bw_train_target_mode="weighted_workload_level",
        bw_return_mode="bw_gae",
        enable_bw_action=True,
        bw_clean_per_user_enabled=True,
        bw_clean_per_user_horizon=2,
        bw_clean_per_user_delta_probe=0.02,
        bw_clean_per_user_beta=0.5,
        bw_clean_per_user_loss="huber",
        bw_clean_trust_region_enabled=True,
        bw_clean_trust_region_target_kl=1.0e-4,
        bw_clean_trust_region_kl_coef_init=0.1,
        bw_clean_trust_region_kl_coef_min=1.0e-4,
        bw_clean_trust_region_kl_coef_max=1.0e3,
        bw_clean_trust_region_backtrack_factor=0.25,
        bw_clean_trust_region_max_backtracks=6,
    )
    bundle = build_structured_modules_from_config(cfg, hidden_dim=32, embed_dim=16, build_critic=False)
    actor = bundle.actor
    actor_optim = torch.optim.Adam(actor.parameters(), lr=3.0e-2)
    learner = StructuredMAPPO(
        actor=actor,
        critic=ZeroStructuredCritic(torch.device("cpu")),
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        max_grad_norm=10.0,
        ppo_epochs=1,
        num_mini_batch=1,
        actor_optimizer=actor_optim,
        critic_optimizer=None,
        target_mode="step_level",
        cfg=cfg,
        train_accel=False,
        train_sat=False,
        train_bw=True,
        exec_accel_source="zero",
        exec_sat_source="zero",
        exec_bw_source="policy",
    )
    def _fake_clean_targets_positive(self, *, snapshot_states, ref_actions, valid_masks):
        targets, rho_values, utility_l1 = _fake_clean_targets_shift_mass(
            self,
            snapshot_states=snapshot_states,
            ref_actions=ref_actions,
            valid_masks=valid_masks,
        )
        self._bw_clean_last_ref_returns = np.zeros((targets.shape[0],), dtype=np.float32)
        return targets, rho_values, utility_l1

    learner._bw_clean_target_actions_parallel = types.MethodType(_fake_clean_targets_positive, learner)
    learner._bw_clean_rollout_returns_parallel = types.MethodType(_fake_clean_target_rollouts_positive, learner)
    env = make_structured_env_group(cfg, num_envs=1, backend="sync")
    try:
        rows = run_structured_training(
            env,
            learner,
            num_updates=1,
            rollout_env_steps=2,
            reset_seed=int(cfg.seed),
        )
        assert len(rows) == 1
        assert float(rows[0].clean_measured_kl) <= float(cfg.bw_clean_trust_region_target_kl) + 1.0e-6
        assert float(rows[0].clean_kl_coef) > 0.0
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def test_structured_clean_per_user_uses_native_tensor_teacher_without_legacy_worker_gate():
    cfg = SaginConfig(
        seed=41,
        num_uav=1,
        num_gu=4,
        num_sat=8,
        users_obs_max=4,
        sats_obs_max=4,
        nbrs_obs_max=1,
        sat_num_select=2,
        T_steps=6,
        b_acc=1.0e6,
        task_arrival_rate=1.0e6,
        traffic_model="sticky_subset_hotspot",
        hotspot_num_subsets=1,
        hotspot_subset_size=2,
        preload_enabled=True,
        preload_prob=1.0,
        preload_hot_gu_steps=10.0,
        preload_bg_gu_steps=2.0,
        train_accel=False,
        train_sat=False,
        train_bw=True,
        exec_accel_source="zero",
        exec_sat_source="zero",
        exec_bw_source="policy",
        reward_mode="weighted_workload_level",
        bw_train_target_mode="weighted_workload_level",
        bw_return_mode="bw_gae",
        enable_bw_action=True,
        bw_clean_per_user_enabled=True,
        bw_clean_per_user_horizon=2,
        bw_clean_per_user_delta_probe=0.02,
        bw_clean_per_user_beta=0.5,
        bw_clean_per_user_loss="huber",
        bw_clean_trust_region_enabled=True,
        bw_clean_trust_region_target_kl=0.01,
        bw_clean_trust_region_kl_coef_init=0.1,
        bw_clean_trust_region_kl_coef_min=1.0e-4,
        bw_clean_trust_region_kl_coef_max=1.0e3,
        bw_clean_trust_region_backtrack_factor=0.5,
        bw_clean_trust_region_max_backtracks=4,
    )
    bundle = build_structured_modules_from_config(cfg, hidden_dim=32, embed_dim=16, build_critic=False)
    actor = bundle.actor
    actor_before = {
        name: tensor.detach().cpu().clone()
        for name, tensor in actor.state_dict().items()
        if tensor.dtype.is_floating_point
    }
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
    learner = StructuredMAPPO(
        actor=actor,
        critic=ZeroStructuredCritic(torch.device("cpu")),
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        max_grad_norm=0.5,
        ppo_epochs=1,
        num_mini_batch=1,
        actor_optimizer=actor_optim,
        critic_optimizer=None,
        target_mode="step_level",
        cfg=cfg,
        train_accel=False,
        train_sat=False,
        train_bw=True,
        exec_accel_source="zero",
        exec_sat_source="zero",
        exec_bw_source="policy",
    )
    def _legacy_worker_should_not_be_used(self, *args, **kwargs):
        raise AssertionError("BW clean per-user update must not call legacy worker/NumPy teacher hooks.")

    def _fake_clean_targets_non_improving(self, *, snapshot_states, ref_actions, valid_masks):
        targets, rho_values, utility_l1 = _fake_clean_targets_shift_mass(
            self,
            snapshot_states=snapshot_states,
            ref_actions=ref_actions,
            valid_masks=valid_masks,
        )
        self._bw_clean_last_ref_returns = np.ones((targets.shape[0],), dtype=np.float32)
        return targets, rho_values, utility_l1

    del _fake_clean_targets_non_improving
    learner._bw_clean_target_actions_parallel = types.MethodType(_legacy_worker_should_not_be_used, learner)
    learner._bw_clean_rollout_returns_parallel = types.MethodType(_legacy_worker_should_not_be_used, learner)
    env = make_structured_env_group(cfg, num_envs=1, backend="sync")
    try:
        rows = run_structured_training(
            env,
            learner,
            num_updates=1,
            rollout_env_steps=2,
            reset_seed=int(cfg.seed),
        )
        assert len(rows) == 1
        assert float(rows[0].bw_actor_update_skipped) == 0.0
        assert float(rows[0].policy_loss) > 0.0
        assert float(rows[0].clean_target_beats_ref_frac) > 0.0
        assert float(rows[0].clean_mean_target_gap) > 0.0
        assert float(rows[0].clean_mean_update_shift) >= 0.0
        assert float(rows[0].clean_update_to_target_ratio) >= 0.0
        actor_after = {name: tensor.detach().cpu() for name, tensor in actor.state_dict().items()}
        changed = any(
            tensor.dtype.is_floating_point and not torch.allclose(actor_before[name], actor_after[name])
            for name, tensor in actor_after.items()
            if name in actor_before
        )
        assert changed
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
