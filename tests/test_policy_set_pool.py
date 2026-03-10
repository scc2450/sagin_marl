from __future__ import annotations

import numpy as np
import torch

from sagin_marl.env.config import SaginConfig
from sagin_marl.rl.policy import ActorNet, batch_flatten_obs


def _make_obs(cfg: SaginConfig) -> dict[str, np.ndarray]:
    obs = {
        "own": np.array([0.1, -0.2, 0.05, -0.05, 0.8, 0.3, 0.4], dtype=np.float32),
        "users": np.zeros((cfg.users_obs_max, 5), dtype=np.float32),
        "users_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "sats": np.zeros((cfg.sats_obs_max, 9), dtype=np.float32),
        "sats_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "nbrs": np.zeros((cfg.nbrs_obs_max, 4), dtype=np.float32),
        "nbrs_mask": np.zeros((cfg.nbrs_obs_max,), dtype=np.float32),
    }
    obs["users"][0] = np.array([0.2, -0.1, 0.6, 1.2, 1.0], dtype=np.float32)
    obs["users_mask"][0] = 1.0
    obs["sats"][0] = np.array([0.1, 0.2, 0.3, 0.0, 0.1, -0.2, 0.4, 3.0, 0.5], dtype=np.float32)
    obs["sats_mask"][0] = 1.0
    obs["nbrs"][0] = np.array([-0.3, 0.2, 0.1, -0.1], dtype=np.float32)
    obs["nbrs_mask"][0] = 1.0
    return obs


def _make_actor(cfg: SaginConfig) -> ActorNet:
    obs_dim = batch_flatten_obs([_make_obs(cfg)], cfg).shape[1]
    actor = ActorNet(obs_dim, cfg)
    actor.eval()
    return actor


def test_set_pool_actor_ignores_masked_slots():
    torch.manual_seed(0)
    cfg = SaginConfig(users_obs_max=3, sats_obs_max=2, nbrs_obs_max=2)
    cfg.actor_encoder_type = "set_pool"
    cfg.actor_set_embed_dim = 16

    actor = _make_actor(cfg)

    obs_a = _make_obs(cfg)
    obs_b = _make_obs(cfg)
    obs_b["users"][1] = np.array([99.0, -88.0, 77.0, -66.0, 55.0], dtype=np.float32)
    obs_b["sats"][1] = np.array([11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype=np.float32)
    obs_b["nbrs"][1] = np.array([-9.0, -8.0, -7.0, -6.0], dtype=np.float32)

    batch = batch_flatten_obs([obs_a, obs_b], cfg)
    with torch.no_grad():
        mu = actor.forward(torch.tensor(batch, dtype=torch.float32))["mu"]

    torch.testing.assert_close(mu[0], mu[1], rtol=0.0, atol=1e-6)


def test_set_pool_actor_handles_empty_masks():
    torch.manual_seed(0)
    cfg = SaginConfig(users_obs_max=3, sats_obs_max=2, nbrs_obs_max=2)
    cfg.actor_encoder_type = "set_pool"
    cfg.actor_set_embed_dim = 16

    actor = _make_actor(cfg)
    obs = _make_obs(cfg)
    obs["users_mask"][:] = 0.0
    obs["sats_mask"][:] = 0.0
    obs["nbrs_mask"][:] = 0.0
    obs["users"][:] = 123.0
    obs["sats"][:] = -456.0
    obs["nbrs"][:] = 789.0
    obs_tensor = torch.tensor(batch_flatten_obs([obs], cfg), dtype=torch.float32)

    with torch.no_grad():
        forward_out = actor.forward(obs_tensor)
        det_out = actor.act(obs_tensor, deterministic=True)
        sample_out = actor.act(obs_tensor, deterministic=False)

    assert torch.isfinite(forward_out["mu"]).all()
    assert torch.isfinite(det_out.action).all()
    assert torch.isfinite(det_out.logprob).all()
    assert torch.isfinite(sample_out.action).all()
    assert torch.isfinite(sample_out.logprob).all()
    assert det_out.action.shape == (1, 2)
    assert det_out.logprob.shape == (1,)
    assert sample_out.action.shape == (1, 2)
    assert sample_out.logprob.shape == (1,)


def test_set_pool_evaluate_actions_parts_returns_finite_logprobs():
    torch.manual_seed(0)
    cfg = SaginConfig(users_obs_max=3, sats_obs_max=2, nbrs_obs_max=2)
    cfg.actor_encoder_type = "set_pool"
    cfg.actor_set_embed_dim = 16

    actor = _make_actor(cfg)
    obs_tensor = torch.tensor(batch_flatten_obs([_make_obs(cfg), _make_obs(cfg)], cfg), dtype=torch.float32)

    with torch.no_grad():
        policy_out = actor.act(obs_tensor, deterministic=False)
        logprob_parts, entropy_parts = actor.evaluate_actions_parts(
            obs_tensor, policy_out.action, out=policy_out.dist_out
        )

    assert set(logprob_parts.keys()) == {"accel"}
    assert set(entropy_parts.keys()) == {"accel"}
    assert logprob_parts["accel"].shape == (2,)
    assert entropy_parts["accel"].shape == (2,)
    assert torch.isfinite(logprob_parts["accel"]).all()
    assert torch.isfinite(entropy_parts["accel"]).all()
    torch.testing.assert_close(logprob_parts["accel"], policy_out.logprob, rtol=1e-5, atol=1e-5)


def test_actor_encoder_types_have_compatible_output_shapes():
    torch.manual_seed(0)
    cfg_flat = SaginConfig(users_obs_max=3, sats_obs_max=2, nbrs_obs_max=2)
    cfg_flat.actor_encoder_type = "flat_mlp"

    cfg_set = SaginConfig(users_obs_max=3, sats_obs_max=2, nbrs_obs_max=2)
    cfg_set.actor_encoder_type = "set_pool"
    cfg_set.actor_set_embed_dim = 16

    obs_batch = batch_flatten_obs([_make_obs(cfg_flat), _make_obs(cfg_flat), _make_obs(cfg_flat)], cfg_flat)
    obs_tensor = torch.tensor(obs_batch, dtype=torch.float32)

    actor_flat = _make_actor(cfg_flat)
    actor_set = _make_actor(cfg_set)

    with torch.no_grad():
        mu_flat = actor_flat.forward(obs_tensor)["mu"]
        mu_set = actor_set.forward(obs_tensor)["mu"]
        act_flat = actor_flat.act(obs_tensor, deterministic=True)
        act_set = actor_set.act(obs_tensor, deterministic=True)

    assert mu_flat.shape == mu_set.shape == (3, 2)
    assert act_flat.action.shape == act_set.action.shape == (3, 2)
    assert act_flat.accel.shape == act_set.accel.shape == (3, 2)
    assert act_flat.logprob.shape == act_set.logprob.shape == (3,)
