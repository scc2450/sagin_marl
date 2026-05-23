from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch

from sagin_marl.env.config import SaginConfig
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.structured_types import BwStageSnapshot, StructuredWorldState
from sagin_marl.utils.runtime_state_bank import save_runtime_state_bank_payload


def _make_cfg() -> SaginConfig:
    return SaginConfig(
        seed=19,
        num_uav=3,
        num_gu=8,
        num_sat=10,
        users_obs_max=8,
        sats_obs_max=5,
        nbrs_obs_max=2,
        sat_num_select=2,
        T_steps=10,
        candidate_mode="nearest",
        candidate_k=8,
    )


def _decode_first_valid_sat_actions(driver: StructuredControlDriver, world_state: StructuredWorldState) -> np.ndarray:
    sat_states = driver.build_local_sat_states(world_state)
    subset_indices: list[int] = []
    for state in sat_states:
        valid = torch.nonzero(state.subset_mask[0], as_tuple=False).flatten()
        subset_indices.append(int(valid[0].item()) if valid.numel() > 0 else -1)
    return driver.decode_sat_subset_actions(sat_states, subset_indices)


def _build_test_bw_action_from_valid_mask(valid_mask: np.ndarray) -> np.ndarray:
    valid_mask_arr = np.asarray(valid_mask, dtype=bool)
    action = np.zeros(valid_mask_arr.shape, dtype=np.float32)
    for u in range(int(valid_mask_arr.shape[0])):
        valid = valid_mask_arr[u]
        valid_slots = np.flatnonzero(valid)
        if valid_slots.size <= 0:
            continue
        if valid_slots.size == 1:
            action[u, valid_slots[0]] = 1.0
            continue
        action[u, valid_slots[0]] = 0.65
        residual = 0.35 / float(valid_slots.size - 1)
        action[u, valid_slots[1:]] = residual
    return action


def _prepare_bw_stage(driver: StructuredControlDriver) -> tuple[StructuredWorldState, BwStageSnapshot, np.ndarray]:
    z0 = driver.begin_step()
    accel_action = np.zeros((driver.env.cfg.num_uav, 2), dtype=np.float32)
    z1 = driver.run_accel_stage(accel_action)
    sat_action = _decode_first_valid_sat_actions(driver, z1)
    z2 = driver.run_sat_stage(sat_action)
    snapshot = driver.build_bw_stage_snapshot(z2)
    bw_action = _build_test_bw_action_from_valid_mask(snapshot.bw_valid_mask)
    return z2, snapshot, bw_action


def _assert_world_state_close(left: StructuredWorldState, right: StructuredWorldState) -> None:
    for field in fields(StructuredWorldState):
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        if torch.is_tensor(left_value):
            assert torch.is_tensor(right_value)
            assert left_value.shape == right_value.shape
            assert torch.allclose(left_value, right_value, atol=1.0e-6, rtol=1.0e-6)
        else:
            np.testing.assert_allclose(
                np.asarray(left_value),
                np.asarray(right_value),
                atol=1.0e-6,
                rtol=1.0e-6,
            )


def _assert_snapshot_close(left: BwStageSnapshot, right: BwStageSnapshot) -> None:
    _assert_world_state_close(left.world_state, right.world_state)
    for name in (
        "assoc",
        "selected_sat_indices",
        "selected_sat_mask",
        "access_gain_matrix",
        "bw_valid_mask_full",
        "ego_features",
        "selected_sat_tokens",
        "gu_tokens",
        "gu_mask",
        "bw_valid_mask",
    ):
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        if torch.is_tensor(left_value):
            assert torch.is_tensor(right_value)
            assert left_value.shape == right_value.shape
            if left_value.dtype.is_floating_point:
                assert torch.allclose(left_value, right_value, atol=1.0e-6, rtol=1.0e-6)
            else:
                assert torch.equal(left_value, right_value)
        else:
            left_arr = np.asarray(left_value)
            right_arr = np.asarray(right_value)
            if left_arr.dtype.kind in {"b", "i", "u"}:
                np.testing.assert_array_equal(left_arr, right_arr)
            else:
                np.testing.assert_allclose(left_arr, right_arr, atol=1.0e-6, rtol=1.0e-6)


def _assert_reward_parts_close(left: dict[str, float], right: dict[str, float]) -> None:
    assert set(left.keys()) == set(right.keys())
    for key in left:
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, str) or isinstance(right_value, str):
            assert left_value == right_value
            continue
        np.testing.assert_allclose(float(left_value), float(right_value), atol=1.0e-6, rtol=1.0e-6)


def _assert_env_runtime_close(left: SaginParallelEnv, right: SaginParallelEnv) -> None:
    np.testing.assert_allclose(left.gu_pos, right.gu_pos, atol=1.0e-6, rtol=1.0e-6)
    np.testing.assert_allclose(left.uav_pos, right.uav_pos, atol=1.0e-6, rtol=1.0e-6)
    np.testing.assert_allclose(left.uav_vel, right.uav_vel, atol=1.0e-6, rtol=1.0e-6)
    np.testing.assert_allclose(left.gu_queue, right.gu_queue, atol=1.0e-6, rtol=1.0e-6)
    np.testing.assert_allclose(left.uav_queue, right.uav_queue, atol=1.0e-6, rtol=1.0e-6)
    np.testing.assert_allclose(left.sat_queue, right.sat_queue, atol=1.0e-6, rtol=1.0e-6)
    np.testing.assert_array_equal(left.last_association, right.last_association)
    np.testing.assert_array_equal(left.prev_association, right.prev_association)
    assert int(left.t) == int(right.t)
    assert int(getattr(left, "global_step", 0)) == int(getattr(right, "global_step", 0))


def _assert_runtime_state_payload_close(left: dict, right: dict) -> None:
    for key in ("gu_pos", "uav_pos", "uav_vel", "gu_queue", "uav_queue", "sat_queue", "last_association", "prev_association"):
        np.testing.assert_allclose(np.asarray(left[key]), np.asarray(right[key]), atol=1.0e-6, rtol=1.0e-6)
    assert int(left["t"]) == int(right["t"])
    assert int(left.get("global_step", 0)) == int(right.get("global_step", 0))


def test_bw_stage_state_roundtrip_reproduces_local_step_outcome():
    cfg = _make_cfg()
    base_env = SaginParallelEnv(cfg)
    replay_env = SaginParallelEnv(cfg)
    try:
        base_env.reset(seed=cfg.seed)
        replay_env.reset(seed=cfg.seed + 97)
        base_driver = StructuredControlDriver(base_env)
        replay_driver = StructuredControlDriver(replay_env)

        bw_world, base_snapshot, bw_action = _prepare_bw_stage(base_driver)
        exported = base_driver.export_bw_stage_state()
        restored_snapshot = replay_driver.load_bw_stage_state(exported)
        _assert_snapshot_close(base_snapshot, restored_snapshot)

        base_step, base_next = base_driver.execute_stage_bw_and_prepare_next_accel(bw_action)
        base_reward_parts = dict(base_env.last_reward_parts)

        replay_step, replay_next = replay_driver.execute_stage_bw_and_prepare_next_accel(bw_action)
        replay_reward_parts = dict(replay_env.last_reward_parts)

        np.testing.assert_allclose(
            np.asarray(list(base_step.rewards.values()), dtype=np.float64),
            np.asarray(list(replay_step.rewards.values()), dtype=np.float64),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        assert base_step.terminations == replay_step.terminations
        assert base_step.truncations == replay_step.truncations
        _assert_reward_parts_close(base_reward_parts, replay_reward_parts)
        _assert_world_state_close(base_next, replay_next)
        _assert_env_runtime_close(base_env, replay_env)
        assert bw_world.stage_id.shape == base_snapshot.world_state.stage_id.shape
    finally:
        for env in (base_env, replay_env):
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()




