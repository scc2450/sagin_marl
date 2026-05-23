from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import sagin_marl.env.vec_env as vec_env
from sagin_marl.env.config import SaginConfig
from sagin_marl.env.structured_sync_group import (
    GpuStructuredDriverGroup,
    GpuStructuredEnvGroup,
)
from sagin_marl.rl.structured_checkpoint import (
    load_structured_train_state,
    save_structured_train_state,
)
from sagin_marl.rl.structured_parallel_eval import (
    batched_policy_accel_actions,
    batched_policy_bw_outputs,
    batched_policy_sat_subset_indices,
)
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_train import (
    close_structured_env_group,
    make_structured_driver_group,
    make_structured_env_group,
)
from tests.structured_test_utils import (
    make_structured_test_cfg as _make_cfg,
    make_structured_test_learner as _make_learner,
)


def test_structured_factory_group_aliases_route_sync_backend_to_gpu_groups():
    cfg = _make_cfg()
    env_group = make_structured_env_group(cfg, num_envs=2, backend="sync")
    driver_group = make_structured_driver_group(cfg, num_envs=2, backend="sync")
    try:
        assert isinstance(env_group, GpuStructuredEnvGroup)
        assert isinstance(driver_group, GpuStructuredDriverGroup)
    finally:
        close_structured_env_group(env_group)
        close_structured_env_group(driver_group)


def test_structured_parallel_config_defaults_prefer_sync():
    cfg = SaginConfig()
    assert cfg.bw_actor_branch_parallel_backend == "sync"
    assert cfg.sat_clean_parallel_backend == "sync"


def test_make_vec_env_defaults_to_sync_backend(monkeypatch):
    cfg = _make_cfg()
    called = {}

    def _sync_factory(factory_cfg, num_envs):
        called["sync"] = (factory_cfg, int(num_envs))
        return ("sync", int(num_envs))

    def _unexpected_subproc(*_args, **_kwargs):
        raise AssertionError("make_vec_env() should default to sync backend.")

    monkeypatch.setattr(vec_env, "SyncVecSaginEnv", _sync_factory)
    monkeypatch.setattr(vec_env, "SubprocVecSaginEnv", _unexpected_subproc)

    out = vec_env.make_vec_env(cfg, num_envs=2)
    assert out == ("sync", 2)
    assert called["sync"][0] is cfg
    assert called["sync"][1] == 2


def test_batched_policy_accel_actions_rejects_world_state_slicing():
    world_states = [
        SimpleNamespace(uav_nodes=torch.zeros((1, 1, 3), dtype=torch.float32)),
        SimpleNamespace(uav_nodes=torch.zeros((1, 1, 3), dtype=torch.float32)),
    ]
    with pytest.raises(RuntimeError, match="StructuredControlDriver.build_local_accel_states"):
        batched_policy_accel_actions(object(), world_states, torch.device("cuda"), True)


def test_batched_policy_sat_subset_indices_collates_on_requested_device(monkeypatch):
    import sagin_marl.rl.structured_parallel_eval as parallel_eval

    collate_devices: list[torch.device] = []
    target_device = torch.device("cuda")

    def _fake_collate(items, device):
        collate_devices.append(device)
        assert len(items) == 2
        return "sat_batch"

    class _Actor:
        def act_sat(self, batch, deterministic):
            assert batch == "sat_batch"
            assert deterministic is False
            return SimpleNamespace(subset_index=torch.tensor([1, 0], dtype=torch.long))

    monkeypatch.setattr(parallel_eval, "_collate_dataclass", _fake_collate)
    sat_snapshots = [
        SimpleNamespace(local_state=SimpleNamespace(ego_features=torch.zeros((1, 3), dtype=torch.float32))),
        SimpleNamespace(local_state=SimpleNamespace(ego_features=torch.zeros((1, 3), dtype=torch.float32))),
    ]
    subset_indices = batched_policy_sat_subset_indices(_Actor(), sat_snapshots, target_device, False)
    assert collate_devices == [target_device]
    assert subset_indices == [[1], [0]]


def test_batched_policy_bw_outputs_materializes_snapshot_tensors_on_requested_device(monkeypatch):
    snapshot_tensor_devices: list[torch.device | None] = []
    target_device = torch.device("cuda")
    real_as_tensor = torch.as_tensor

    def _fake_as_tensor(data, dtype=None, device=None):
        snapshot_tensor_devices.append(device)
        return real_as_tensor(data, dtype=dtype)

    class _Actor:
        def act_bw(self, batch, deterministic):
            assert tuple(batch.ego_features.shape) == (2, 10)
            assert tuple(batch.selected_sat_tokens.shape) == (2, 2, 9)
            assert tuple(batch.gu_tokens.shape) == (2, 2, 13)
            assert tuple(batch.bw_valid_mask.shape) == (2, 2)
            assert deterministic is True
            return SimpleNamespace(action=torch.tensor([[0.25, 0.75], [0.6, 0.4]], dtype=torch.float32))

    monkeypatch.setattr(torch, "as_tensor", _fake_as_tensor)
    bw_snapshots = [
        SimpleNamespace(
            ego_features=torch.zeros((1, 10), dtype=torch.float32),
            selected_sat_tokens=torch.zeros((1, 2, 9), dtype=torch.float32),
            selected_sat_mask=torch.tensor([[True, False]], dtype=torch.bool),
            gu_tokens=torch.zeros((1, 2, 13), dtype=torch.float32),
            gu_mask=torch.tensor([[True, True]], dtype=torch.bool),
            bw_valid_mask=torch.tensor([[True, False]], dtype=torch.bool),
        ),
        SimpleNamespace(
            ego_features=torch.ones((1, 10), dtype=torch.float32),
            selected_sat_tokens=torch.ones((1, 2, 9), dtype=torch.float32),
            selected_sat_mask=torch.tensor([[True, True]], dtype=torch.bool),
            gu_tokens=torch.ones((1, 2, 13), dtype=torch.float32),
            gu_mask=torch.tensor([[True, True]], dtype=torch.bool),
            bw_valid_mask=torch.tensor([[True, True]], dtype=torch.bool),
        ),
    ]
    outputs = batched_policy_bw_outputs(_Actor(), bw_snapshots, target_device, True)
    assert snapshot_tensor_devices.count(target_device) == 12
    assert len(outputs.actions) == 2
    assert outputs.actions[1].shape == (1, 2)


def test_structured_train_state_roundtrip(tmp_path):
    cfg = _make_cfg()
    learner, actor, critic = _make_learner(cfg)
    actor_optim = learner.actor_optimizer
    critic_optim = learner.critic_optimizer
    actor_loss = sum(p.sum() for p in actor.parameters())
    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()
    critic_loss = sum(p.sum() for p in critic.parameters())
    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()
    history_rows = [
        {
            "update": 1,
            "env_reward_mean": 1.0,
            "episodes_finished": 0,
            "policy_loss": 0.1,
            "value_loss": 0.2,
            "entropy": 0.3,
            "approx_kl": 0.4,
            "clip_frac": 0.5,
        }
    ]
    save_structured_train_state(
        str(tmp_path),
        actor,
        critic,
        actor_optim,
        critic_optim,
        update=3,
        planned_total_updates=10,
        total_env_steps=24,
        history_rows=history_rows,
        total_time_sec=12.5,
    )

    learner2, actor2, critic2 = _make_learner(cfg)
    meta = load_structured_train_state(
        str(tmp_path / "train_state.pt"),
        actor2,
        critic2,
        learner2.actor_optimizer,
        learner2.critic_optimizer,
        device=learner2.device,
    )
    assert meta["update"] == 3
    assert meta["planned_total_updates"] == 10
    assert meta["total_env_steps"] == 24
    assert meta["history_rows"] == history_rows
    for name, value in actor.state_dict().items():
        assert torch.allclose(value, actor2.state_dict()[name])
    for name, value in critic.state_dict().items():
        assert torch.allclose(value, critic2.state_dict()[name])


def test_structured_actor_only_train_state_roundtrip(tmp_path):
    cfg = SaginConfig(
        seed=13,
        num_uav=1,
        num_gu=4,
        num_sat=6,
        users_obs_max=4,
        sats_obs_max=4,
        nbrs_obs_max=1,
        sat_num_select=2,
        T_steps=6,
        train_accel=False,
        train_sat=False,
        train_bw=True,
        exec_accel_source="zero",
        exec_sat_source="zero",
        exec_bw_source="policy",
        bw_actor_advantage_override_mode="branch_delta",
    )
    bundle = build_structured_modules_from_config(cfg, hidden_dim=32, embed_dim=16, build_critic=False)
    actor = bundle.actor
    assert bundle.critic is None
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
    actor_loss = sum(p.sum() for p in actor.parameters())
    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()

    save_structured_train_state(
        str(tmp_path),
        actor,
        None,
        actor_optim,
        None,
        update=2,
        planned_total_updates=10,
        total_env_steps=12,
        history_rows=[{"update": 1, "env_reward_mean": 0.5}],
        total_time_sec=3.0,
    )

    bundle2 = build_structured_modules_from_config(cfg, hidden_dim=32, embed_dim=16, build_critic=False)
    actor2 = bundle2.actor
    actor_optim2 = torch.optim.Adam(actor2.parameters(), lr=1e-3)
    meta = load_structured_train_state(
        str(tmp_path / "train_state.pt"),
        actor2,
        None,
        actor_optim2,
        None,
        device=torch.device("cpu"),
    )
    assert meta["update"] == 2
    assert meta["critic_enabled"] is False
    assert meta["critic_info"] is None
    for name, value in actor.state_dict().items():
        assert torch.allclose(value, actor2.state_dict()[name])




