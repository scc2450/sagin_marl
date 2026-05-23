from __future__ import annotations

import numpy as np
import torch

from sagin_marl.env.config import SaginConfig
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.env.structured_sync_group import GpuStructuredDriverGroup
from sagin_marl.rl.structured_actor import AccelPolicy, BwPolicy, SatSubsetPolicy, StructuredActor
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_critic import StructuredCritic
from sagin_marl.rl.structured_mappo import StructuredMAPPO
from sagin_marl.rl.structured_types import LocalBwState
from tests.structured_test_utils import (
    assert_local_state_group_close as _assert_local_state_group_close,
    assert_runtime_state_matches_env as _assert_runtime_state_matches_env,
    assert_world_state_close as _assert_world_state_close,
    build_structured_modules_from_probe_driver as _build_structured_modules_from_probe_driver,
    default_sat_subset_action as _default_sat_subset_action,
    materialize_stage_obs_baseline as _materialize_stage_obs_baseline,
)

def test_structured_driver_can_materialize_tensor_world_states_and_snapshots():
    import torch

    cfg = SaginConfig(
        seed=11,
        num_uav=2,
        num_gu=4,
        num_sat=6,
        users_obs_max=4,
        sats_obs_max=4,
        nbrs_obs_max=1,
        sat_num_select=2,
    )
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed)
    driver = StructuredControlDriver(env, tensor_device=torch.device("cpu"))

    z0 = driver.begin_step()
    assert torch.is_tensor(z0.uav_nodes)
    assert z0.uav_nodes.device.type == "cpu"
    accel_states = driver.build_local_accel_states(z0)
    assert accel_states[0].ego_features.device.type == "cpu"

    z1 = driver.run_accel_stage(np.zeros((cfg.num_uav, 2), dtype=np.float32))
    assert torch.is_tensor(z1.uav_nodes)
    sat_states = driver.build_local_sat_states(z1)
    assert sat_states[0].ego_features.device.type == "cpu"

    sat_action = _default_sat_subset_action(driver, sat_states)
    z2 = driver.run_sat_stage(sat_action)
    snapshot = driver.build_bw_stage_snapshot(z2)
    assert torch.is_tensor(snapshot.bw_valid_mask)
    bw_states = driver.build_bw_valid_context(z2)
    assert bw_states[0].gu_tokens.device.type == "cpu"


def test_structured_batch_core_sync_runtime_state_uses_core_orbit_lookup():
    import torch

    cfg = SaginConfig(
        seed=126,
        num_uav=2,
        num_gu=4,
        num_sat=6,
        users_obs_max=4,
        sats_obs_max=4,
        nbrs_obs_max=1,
        sat_num_select=2,
        T_steps=8,
    )

    sync_group = GpuStructuredDriverGroup(cfg, num_envs=2, tensor_device=torch.device("cpu"))

    try:
        seeds = [int(cfg.seed) + idx for idx in range(2)]
        sync_group.reset_many(seeds)

        sync_group.batch_core._sync_runtime_state()

        runtime_state = sync_group.batch_core.runtime_state
        expected_sat_pos = np.stack(
            [np.asarray(sync_group.batch_core._orbit_pos_table[int(env.t)], dtype=np.float32) for env in sync_group.envs],
            axis=0,
        )
        expected_sat_vel = np.stack(
            [np.asarray(sync_group.batch_core._orbit_vel_table[int(env.t)], dtype=np.float32) for env in sync_group.envs],
            axis=0,
        )
        np.testing.assert_allclose(runtime_state.sat_pos, expected_sat_pos, atol=1e-6)
        np.testing.assert_allclose(runtime_state.sat_vel, expected_sat_vel, atol=1e-6)
    finally:
        sync_group.close()


def test_structured_batch_core_runtime_state_is_tensor_backed_proxy():
    import torch

    cfg = SaginConfig(
        seed=127,
        num_uav=2,
        num_gu=4,
        num_sat=6,
        users_obs_max=4,
        sats_obs_max=4,
        nbrs_obs_max=1,
        sat_num_select=2,
        T_steps=8,
    )

    sync_group = GpuStructuredDriverGroup(cfg, num_envs=2, tensor_device=torch.device("cpu"))

    try:
        seeds = [int(cfg.seed) + idx for idx in range(2)]
        sync_group.reset_many(seeds)

        runtime_state = sync_group.batch_core.runtime_state
        runtime_tensor_state = sync_group.batch_core.runtime_tensor_state

        assert not isinstance(runtime_state.uav_pos, np.ndarray)
        assert not isinstance(runtime_state.global_step, np.ndarray)

        updated_uav_pos = np.full((1, int(cfg.num_uav), 2), 0.125, dtype=np.float32)
        runtime_state.uav_pos[[0]] = torch.as_tensor(
            updated_uav_pos,
            dtype=runtime_state.uav_pos.dtype,
            device=runtime_state.uav_pos.device,
        )
        runtime_state.global_step[[1]] = torch.as_tensor(
            [7],
            dtype=runtime_state.global_step.dtype,
            device=runtime_state.global_step.device,
        )

        np.testing.assert_allclose(
            runtime_tensor_state.uav_pos[0:1].cpu().numpy(),
            updated_uav_pos,
            atol=0,
        )
        np.testing.assert_allclose(
            runtime_tensor_state.global_step[1:2].cpu().numpy(),
            np.asarray([7], dtype=np.int32),
            atol=0,
        )
        np.testing.assert_allclose(
            np.asarray(runtime_state.uav_pos[0:1], dtype=np.float32),
            updated_uav_pos,
            atol=0,
        )
    finally:
        sync_group.close()


def test_structured_sync_group_batch_env_global_state_batch_skips_env_get_global_state():
    import torch

    cfg = SaginConfig(
        seed=5011,
        num_uav=2,
        num_gu=4,
        num_sat=6,
        users_obs_max=4,
        sats_obs_max=4,
        nbrs_obs_max=1,
        sat_num_select=2,
        T_steps=8,
    )
    sync_group = GpuStructuredDriverGroup(cfg, num_envs=2, tensor_device=torch.device("cpu"))
    try:
        seeds = [int(cfg.seed) + idx for idx in range(2)]
        sync_group.reset_many(seeds)
        baseline_global = torch.as_tensor(
            np.stack([np.asarray(env.get_global_state(), dtype=np.float32) for env in sync_group.envs], axis=0),
            dtype=torch.float32,
        )
        batch_global = sync_group.get_global_state_batch(device=torch.device("cpu"))
        torch.testing.assert_close(batch_global, baseline_global)
    finally:
        sync_group.close()


def test_structured_sync_group_batch_env_global_state_batch_reuses_obs_context_when_global_cache_missing():
    import torch

    cfg = SaginConfig(
        seed=5012,
        num_uav=2,
        num_gu=4,
        num_sat=6,
        users_obs_max=4,
        sats_obs_max=4,
        nbrs_obs_max=1,
        sat_num_select=2,
        T_steps=8,
        danger_nbr_enabled=True,
        obs_user_include_assoc_uav_cost=True,
        obs_user_include_assoc_sat_cost_mean=True,
        obs_user_include_weighted_queue_cost=True,
    )
    sync_group = GpuStructuredDriverGroup(cfg, num_envs=2, tensor_device=torch.device("cpu"))
    try:
        seeds = [int(cfg.seed) + idx for idx in range(2)]
        sync_group.reset_many(seeds)
        for env in sync_group.envs:
            _ = env._get_obs(0)
            env._cached_global_state = None
        baseline_global = torch.as_tensor(
            np.stack([np.asarray(env.get_global_state(), dtype=np.float32) for env in sync_group.envs], axis=0),
            dtype=torch.float32,
        )
        batch_global = sync_group.get_global_state_batch(device=torch.device("cpu"))
        torch.testing.assert_close(batch_global, baseline_global)
    finally:
        sync_group.close()






