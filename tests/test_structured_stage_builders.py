from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest
import torch

from sagin_marl.env.config import SaginConfig
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.structured_stage_builders import (
    build_batched_local_accel_states_from_spec,
    build_batched_local_sat_states_from_spec,
    build_local_accel_states_from_spec,
    build_local_bw_states_from_snapshot,
    build_local_sat_states_from_spec,
)
from sagin_marl.rl import structured_stage_builders
from sagin_marl.rl.structured_types import StructuredWorldState


def _default_sat_subset_action(driver: StructuredControlDriver, sat_states):
    subset_indices = []
    for state in sat_states:
        valid = state.subset_mask[0].nonzero(as_tuple=False).flatten()
        subset_indices.append(int(valid[0].item()) if valid.numel() > 0 else -1)
    return driver.decode_sat_subset_actions(sat_states, subset_indices)


def _stack_world_states(world_states) -> StructuredWorldState:
    kwargs = {}
    for field in fields(StructuredWorldState):
        kwargs[field.name] = torch.cat(
            [torch.as_tensor(getattr(world_state, field.name)) for world_state in world_states],
            dim=0,
        )
    return StructuredWorldState(**kwargs)


def _prepare_structured_states(seed: int):
    cfg = SaginConfig(
        seed=seed,
        num_uav=3,
        num_gu=6,
        num_sat=8,
        users_obs_max=6,
        sats_obs_max=5,
        nbrs_obs_max=2,
        sat_num_select=2,
    )
    env = SaginParallelEnv(cfg)
    env.reset(seed=seed)
    driver = StructuredControlDriver(env)
    accel_world = driver.begin_step()
    sat_world = driver.run_accel_stage(np.zeros((cfg.num_uav, 2), dtype=np.float32))
    sat_states = driver.build_local_sat_states(sat_world)
    sat_action = _default_sat_subset_action(driver, sat_states)
    bw_world = driver.run_sat_stage(sat_action)
    bw_snapshot = driver.build_bw_stage_snapshot(bw_world)
    return accel_world, sat_world, bw_snapshot


def _prepare_sat_stage_spec(seed: int):
    cfg = SaginConfig(
        seed=seed,
        num_uav=3,
        num_gu=6,
        num_sat=8,
        users_obs_max=6,
        sats_obs_max=5,
        visible_sats_max=5,
        per_uav_visible_sat_token_max=5,
        nbrs_obs_max=2,
        sat_num_select=2,
    )
    env = SaginParallelEnv(cfg)
    env.reset(seed=seed)
    driver = StructuredControlDriver(env)
    driver.begin_step()
    driver.run_accel_stage(np.zeros((cfg.num_uav, 2), dtype=np.float32))
    return driver._prepare_sat_stage_spec()


def _prepare_accel_stage_spec(seed: int):
    cfg = SaginConfig(
        seed=seed,
        num_uav=3,
        num_gu=6,
        num_sat=8,
        users_obs_max=6,
        sats_obs_max=5,
        nbrs_obs_max=2,
        sat_num_select=2,
    )
    env = SaginParallelEnv(cfg)
    env.reset(seed=seed)
    driver = StructuredControlDriver(env)
    accel_world = driver.begin_step()
    assert driver._accel_stage_spec_cache is not None
    return driver._accel_stage_spec_cache, accel_world


def _concat_field(samples, field_name: str) -> torch.Tensor:
    return torch.cat([getattr(sample, field_name) for sample in samples], dim=0)


def test_accel_builder_uses_stage_spec_not_world_slicing():
    spec, accel_world = _prepare_accel_stage_spec(11)
    expected = build_local_accel_states_from_spec(spec)
    batch = build_batched_local_accel_states_from_spec(spec)
    for field in fields(type(batch)):
        torch.testing.assert_close(getattr(batch, field.name), _concat_field(expected, field.name))
    with pytest.raises(RuntimeError, match="retired"):
        structured_stage_builders.build_local_sat_states_from_world(accel_world, max_select=2)


def test_batched_sat_builder_matches_per_row_builder():
    spec = _prepare_sat_stage_spec(21)
    expected = build_local_sat_states_from_spec(spec)
    batch = build_batched_local_sat_states_from_spec(spec)
    for field in fields(type(batch)):
        if field.name == "subset_members":
            torch.testing.assert_close(getattr(batch, field.name), getattr(expected[0], field.name).squeeze(0))
            continue
        torch.testing.assert_close(getattr(batch, field.name), _concat_field(expected, field.name))


def test_bw_snapshot_builder_rehydrates_full_g_state():
    bw_snapshots = [_prepare_structured_states(31)[2], _prepare_structured_states(32)[2]]
    for snapshot in bw_snapshots:
        rows = build_local_bw_states_from_snapshot(snapshot)
        for field in fields(type(rows[0])):
            rebuilt = _concat_field(rows, field.name)
            original = torch.as_tensor(getattr(snapshot, field.name))
            torch.testing.assert_close(rebuilt, original)


