from __future__ import annotations

import csv

import pytest

from sagin_marl.env.config import SaginConfig
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.rl.mappo import train


@pytest.mark.parametrize("actor_encoder_type", ["flat_mlp", "set_pool"])
def test_smoke_train(tmp_path, actor_encoder_type):
    cfg = SaginConfig(num_uav=2, num_gu=5, num_sat=3, users_obs_max=5, sats_obs_max=3, nbrs_obs_max=1)
    cfg.buffer_size = 5
    cfg.num_mini_batch = 1
    cfg.ppo_epochs = 1
    cfg.actor_encoder_type = actor_encoder_type
    cfg.actor_set_embed_dim = 16
    env = SaginParallelEnv(cfg)
    train(env, cfg, str(tmp_path), total_updates=1)
    with (tmp_path / "metrics.csv").open("r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    assert "episode_reward" in header
    assert "episode_length_mean" in header
    assert "completed_episode_count" in header
    assert "rollout_reward_per_step" in header
    assert "episode_term_throughput_access" in header
    assert "episode_term_throughput_backhaul" in header
    assert "throughput_access_norm" in header
    assert "throughput_backhaul_norm" in header
    assert "gu_queue_mean" in header
    assert "uav_queue_mean" in header
    assert "queue_total_active" in header
    assert "collision_rate" in header
    assert "approx_kl" in header
    assert "clip_frac" in header
    assert "policy_loss" in header
    assert "value_loss" in header
    assert "explained_variance" in header
    assert "entropy" in header
    assert "danger_imitation_loss" in header
    assert "danger_imitation_coef" in header
    assert "danger_imitation_active_rate" in header
    assert "actor_lr" in header
    assert "critic_lr" in header
    assert "env_steps_per_sec" in header
    assert "update_steps_per_sec" in header
    assert "total_env_steps" in header
