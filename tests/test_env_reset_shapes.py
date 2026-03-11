from __future__ import annotations

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
