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
    assert "approx_kl" in header
    assert "clip_frac" in header
    assert "adv_raw_std" in header
    assert "adv_preclip_mean" in header
    assert "adv_postclip_std" in header
    assert "adv_clip_frac" in header
    assert "log_std_mean" in header
    assert "action_std_mean" in header
    assert "reward_rms_sigma" in header
    assert "reward_clip_frac" in header
    assert "drop_sum" in header
    assert "r_close_risk" in header
    assert "r_term_accel" in header
    assert "r_term_close_risk" in header
    assert "r_collision_penalty" in header
    assert "r_battery_penalty" in header
    assert "q_norm_tail_hit_rate" in header
