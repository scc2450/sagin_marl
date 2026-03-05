from __future__ import annotations

import csv

from sagin_marl.env.config import SaginConfig
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.rl.mappo import train


def test_smoke_train(tmp_path):
    cfg = SaginConfig(num_uav=2, num_gu=5, num_sat=3, users_obs_max=5, sats_obs_max=3, nbrs_obs_max=1)
    cfg.buffer_size = 5
    cfg.num_mini_batch = 1
    cfg.ppo_epochs = 1
    env = SaginParallelEnv(cfg)
    train(env, cfg, str(tmp_path), total_updates=1)
    with (tmp_path / "metrics.csv").open("r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    assert "approx_kl" in header
    assert "clip_frac" in header
    assert "adv_raw_std" in header
    assert "reward_rms_sigma" in header
    assert "reward_clip_frac" in header
    assert "drop_sum" in header
    assert "r_term_accel" in header
    assert "r_collision_penalty" in header
    assert "r_battery_penalty" in header
