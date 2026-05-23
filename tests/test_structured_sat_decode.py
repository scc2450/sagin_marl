from __future__ import annotations

import numpy as np
import pytest
import torch

from sagin_marl.env.config import SaginConfig
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl import structured_sat_actor_schema as sat_schema
from sagin_marl.rl.structured_types import LocalSatState


def _first_legal_sat_action(driver: StructuredControlDriver, sat_states: list[LocalSatState]) -> np.ndarray:
    subset_indices: list[int] = []
    for state in sat_states:
        legal = torch.nonzero(state.subset_mask[0], as_tuple=False).flatten()
        assert int(legal.numel()) > 0
        subset_indices.append(int(legal[0].item()))
    return driver.decode_sat_subset_actions(sat_states, subset_indices)


def test_decode_sat_subset_actions_keeps_empty_subset_when_no_visible_sat():
    cfg = SaginConfig(seed=0, num_uav=1, num_gu=1, num_sat=2, sats_obs_max=2, sat_num_select=2, N_RF=2)
    env = SaginParallelEnv(cfg)
    try:
        env.reset(seed=cfg.seed)
        driver = StructuredControlDriver(env)
        driver._stage_visible = [[]]
        state = LocalSatState(
            ego_features=torch.zeros((1, sat_schema.SAT_EGO_DIM), dtype=torch.float32),
            demand_features=torch.zeros((1, sat_schema.SAT_DEMAND_DIM), dtype=torch.float32),
            role_features=torch.zeros((1, sat_schema.SAT_ROLE_DIM), dtype=torch.float32),
            sat_tokens=torch.zeros((1, 0, sat_schema.SAT_TOKEN_DIM), dtype=torch.float32),
            sat_mask=torch.zeros((1, 0), dtype=torch.bool),
            sat_valid_mask=torch.zeros((1, 0), dtype=torch.bool),
            subset_members=torch.full((1, 2), -1, dtype=torch.long),
            subset_mask=torch.ones((1, 1), dtype=torch.bool),
            candidate_sat_ids=torch.full((1, 0), -1, dtype=torch.long),
        )
        actions = driver.decode_sat_subset_actions([state], [0])
        np.testing.assert_array_equal(actions, np.asarray([[-1, -1]], dtype=np.int64))
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def test_run_sat_stage_requires_sat_action_select_k_width_not_raw_sat_num_select():
    cfg = SaginConfig(
        seed=1,
        num_uav=1,
        num_gu=1,
        num_sat=2,
        sats_obs_max=2,
        sat_num_select=2,
        N_RF=1,
        fixed_satellite_strategy=False,
    )
    assert cfg.sat_action_select_k == 1
    env = SaginParallelEnv(cfg)
    try:
        env.reset(seed=cfg.seed)
        driver = StructuredControlDriver(env)
        driver.begin_step()
        z1 = driver.run_accel_stage(np.zeros((cfg.num_uav, 2), dtype=np.float32))
        sat_states = driver.build_local_sat_states(z1)
        valid_action = _first_legal_sat_action(driver, sat_states)
        assert valid_action.shape == (cfg.num_uav, cfg.sat_action_select_k)
        driver.run_sat_stage(valid_action)

        env.reset(seed=cfg.seed)
        driver = StructuredControlDriver(env)
        driver.begin_step()
        driver.run_accel_stage(np.zeros((cfg.num_uav, 2), dtype=np.float32))
        raw_sat_num_select_width = np.full((cfg.num_uav, int(cfg.sat_num_select)), -1, dtype=np.int64)
        with pytest.raises(ValueError, match="sat_action must have shape"):
            driver.run_sat_stage(raw_sat_num_select_width)
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def test_run_sat_stage_rejects_duplicate_and_fractional_selected_sat():
    cfg = SaginConfig(
        seed=2,
        num_uav=1,
        num_gu=1,
        num_sat=6,
        sats_obs_max=4,
        sat_num_select=2,
        N_RF=2,
        fixed_satellite_strategy=False,
    )
    env = SaginParallelEnv(cfg)
    try:
        env.reset(seed=cfg.seed)
        driver = StructuredControlDriver(env)
        driver.begin_step()
        z1 = driver.run_accel_stage(np.zeros((cfg.num_uav, 2), dtype=np.float32))
        sat_states = driver.build_local_sat_states(z1)
        valid_action = _first_legal_sat_action(driver, sat_states)
        valid_ids = valid_action[0][valid_action[0] >= 0]
        if valid_ids.size == 0:
            pytest.skip("No legal SAT candidate in this seeded geometry.")

        duplicate = np.full_like(valid_action, -1)
        duplicate[0, :] = int(valid_ids[0])
        with pytest.raises(ValueError, match="duplicate"):
            driver.run_sat_stage(duplicate)

        env.reset(seed=cfg.seed)
        driver = StructuredControlDriver(env)
        driver.begin_step()
        driver.run_accel_stage(np.zeros((cfg.num_uav, 2), dtype=np.float32))
        fractional = valid_action.astype(np.float32)
        fractional[0, 0] = float(valid_ids[0]) + 0.5
        with pytest.raises(ValueError, match="finite integer"):
            driver.run_sat_stage(fractional)
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

