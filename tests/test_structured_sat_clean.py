from __future__ import annotations

import types
from dataclasses import fields

import numpy as np
import pytest
import torch

from sagin_marl.env.config import SaginConfig, load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.structured_critic import ZeroStructuredCritic
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import (
    StructuredMAPPO,
    _current_obs_list,
    _heuristic_bw,
    _refresh_stage_obs_cache,
    sat_clean_joint_critic_free_enabled,
)
from sagin_marl.rl.structured_train import run_structured_training


def _make_sat_clean_cfg(**overrides) -> SaginConfig:
    cfg = SaginConfig(
        seed=41,
        num_uav=3,
        num_gu=8,
        num_sat=10,
        users_obs_max=8,
        sats_obs_max=5,
        nbrs_obs_max=2,
        sat_num_select=2,
        T_steps=8,
        candidate_mode="nearest",
        candidate_k=8,
        reward_mode="weighted_workload_level",
        train_accel=False,
        train_sat=True,
        train_bw=False,
        exec_accel_source="cluster_center_queue_aware",
        exec_sat_source="policy",
        exec_bw_source="queue_aware",
        sat_clean_joint_enabled=True,
        sat_clean_contexts_per_update=4,
        sat_clean_entropy_topk_per_env=1,
        sat_clean_uniform_contexts_per_env=1,
        sat_clean_topm_per_uav=3,
        sat_clean_parallel_envs=0,
        obs_own_include_uav_id_norm=True,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_sat_clean_reference_config_loads_with_expected_recipe():
    cfg = load_config("configs/clean_sat/structured_sat_clean_joint_beijing_hotspot_res200.yaml")
    assert int(cfg.num_uav) == 3
    assert int(cfg.num_gu) == 20
    assert str(cfg.traffic_model) == "sticky_subset_hotspot"
    assert float(cfg.ref_lat_deg) == pytest.approx(39.9042)
    assert float(cfg.rain_lat_deg) == pytest.approx(39.9042)
    assert bool(cfg.resource_scale_enabled)
    assert float(cfg.resource_scale_ref_num_uav) == pytest.approx(3.0)
    assert float(cfg.resource_scale_ref_num_gu) == pytest.approx(20.0)
    assert int(cfg.hotspot_num_subsets) == 20
    assert int(cfg.hotspot_subset_size) == 10
    assert bool(cfg.sat_clean_joint_enabled)
    assert not bool(cfg.train_accel)
    assert bool(cfg.train_sat)
    assert not bool(cfg.train_bw)
    assert str(cfg.exec_accel_source) == "cluster_center_queue_aware"
    assert str(cfg.exec_bw_source) == "queue_aware"
    assert str(cfg.reward_mode) == "weighted_workload_level"
    assert not bool(cfg.reward_stage3_sat_overlap_enabled)
    assert not bool(cfg.sat_counterfactual_credit_enabled)
    assert not bool(cfg.sat_supervision_enabled)
    assert bool(cfg.obs_own_include_uav_id_norm)


def _decode_first_valid_sat_actions(driver: StructuredControlDriver, world_state) -> np.ndarray:
    sat_states = driver.build_local_sat_states(world_state)
    subset_indices: list[int] = []
    for state in sat_states:
        valid = torch.nonzero(state.subset_mask[0], as_tuple=False).flatten()
        subset_indices.append(int(valid[0].item()) if valid.numel() > 0 else -1)
    return driver.decode_sat_subset_actions(sat_states, subset_indices)


def _queue_aware_bw_action(driver: StructuredControlDriver, cfg: SaginConfig) -> np.ndarray:
    _refresh_stage_obs_cache(driver)
    return _heuristic_bw(_current_obs_list(driver), cfg, "queue_aware").astype(np.float32, copy=False)


def _assert_world_state_close(left, right) -> None:
    for field in fields(type(left)):
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        if torch.is_tensor(left_value):
            assert torch.is_tensor(right_value)
            assert left_value.shape == right_value.shape
            assert torch.allclose(left_value, right_value, atol=1.0e-6, rtol=1.0e-6)
        else:
            np.testing.assert_allclose(np.asarray(left_value), np.asarray(right_value), atol=1.0e-6, rtol=1.0e-6)


def test_sat_stage_state_roundtrip_reproduces_queue_aware_bw_step():
    cfg = _make_sat_clean_cfg()
    base_env = SaginParallelEnv(cfg)
    replay_env = SaginParallelEnv(cfg)
    try:
        base_env.reset(seed=cfg.seed)
        replay_env.reset(seed=cfg.seed + 97)
        base_driver = StructuredControlDriver(base_env)
        replay_driver = StructuredControlDriver(replay_env)

        accel_action = np.zeros((cfg.num_uav, 2), dtype=np.float32)
        sat_world = base_driver.run_accel_stage(accel_action) if base_driver.begin_step() is not None else None
        assert sat_world is not None
        exported = base_driver.export_sat_stage_state()
        restored_snapshot = replay_driver.load_sat_stage_state(exported)
        _assert_world_state_close(base_driver.build_sat_stage_snapshot().world_state, restored_snapshot.world_state)

        sat_action = _decode_first_valid_sat_actions(base_driver, sat_world)
        base_driver.run_sat_stage(sat_action)
        replay_driver.run_sat_stage(sat_action)
        bw_action_base = _queue_aware_bw_action(base_driver, cfg)
        bw_action_replay = _queue_aware_bw_action(replay_driver, cfg)
        np.testing.assert_allclose(bw_action_base, bw_action_replay, atol=1.0e-6, rtol=1.0e-6)

        base_step, base_next = base_driver.execute_stage_bw_and_prepare_next_accel(bw_action_base)
        replay_step, replay_next = replay_driver.execute_stage_bw_and_prepare_next_accel(bw_action_replay)

        np.testing.assert_allclose(
            np.asarray(list(base_step.rewards.values()), dtype=np.float64),
            np.asarray(list(replay_step.rewards.values()), dtype=np.float64),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        np.testing.assert_allclose(
            float(base_step.bw_weighted_workload_level_reward),
            float(replay_step.bw_weighted_workload_level_reward),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        _assert_world_state_close(base_next, replay_next)
    finally:
        for env in (base_env, replay_env):
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()




