from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from sagin_marl.env import channel
from sagin_marl.env.config import SaginConfig
from sagin_marl.env.numeric_guards import NORMALIZATION_DENOM_EPS
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl import structured_critic_schema as critic_schema
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_rollout_debug import append_single_env_step
from sagin_marl.rl.structured_types import StructuredWorldState


def _dummy_world_state(stage_id: int) -> StructuredWorldState:
    return StructuredWorldState(
        uav_nodes=torch.zeros((1, 2, critic_schema.CRITIC_UAV_NODE_DIM), dtype=torch.float32),
        gu_nodes=torch.zeros((1, 3, critic_schema.CRITIC_GU_NODE_DIM), dtype=torch.float32),
        sat_nodes=torch.zeros((1, 2, critic_schema.CRITIC_SAT_NODE_DIM), dtype=torch.float32),
        sat_ids=torch.tensor([[0, 1]], dtype=torch.long),
        uav_gu_edges=torch.zeros((1, 2, 3, critic_schema.CRITIC_UAV_GU_EDGE_DIM), dtype=torch.float32),
        uav_sat_edges=torch.zeros((1, 2, 2, critic_schema.CRITIC_UAV_SAT_EDGE_DIM), dtype=torch.float32),
        uav_uav_edges=torch.zeros((1, 2, 2, critic_schema.CRITIC_UAV_UAV_EDGE_DIM), dtype=torch.float32),
        global_scalars=torch.zeros((1, critic_schema.CRITIC_GLOBAL_SCALAR_DIM), dtype=torch.float32),
        gu_mask=torch.ones((1, 3), dtype=torch.bool),
        sat_mask=torch.ones((1, 2), dtype=torch.bool),
        uav_gu_mask=torch.ones((1, 2, 3), dtype=torch.bool),
        uav_sat_mask=torch.ones((1, 2, 2), dtype=torch.bool),
        uav_uav_mask=torch.ones((1, 2, 2), dtype=torch.bool),
        stage_id=torch.tensor([stage_id], dtype=torch.long),
    )


def _digest_rng_state(state) -> str:
    payload = json.dumps(state, sort_keys=True, default=lambda x: x.tolist() if hasattr(x, "tolist") else x)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _uniform_full_g_bw_action(cfg: SaginConfig, bw_states) -> np.ndarray:
    action = np.zeros((cfg.num_uav, cfg.num_gu), dtype=np.float32)
    for u, state in enumerate(bw_states):
        valid = state.bw_valid_mask[0].cpu().numpy().astype(bool)
        if np.any(valid):
            action[u, valid] = 1.0 / float(np.sum(valid))
    return action


def _default_sat_action(driver: StructuredControlDriver) -> np.ndarray:
    sat_states = driver.build_local_sat_states()
    subset_indices = []
    for state in sat_states:
        valid = torch.nonzero(state.subset_mask[0], as_tuple=False).flatten()
        subset_indices.append(int(valid[0].item()) if int(valid.numel()) > 0 else 0)
    return driver.decode_sat_subset_actions(sat_states, subset_indices)


def _append_debug_env_step(
    buffer: StructuredRolloutBuffer,
    *,
    accel_world_state: StructuredWorldState,
    sat_world_state: StructuredWorldState,
    bw_world_state: StructuredWorldState,
    next_world_state: StructuredWorldState,
    accel_value: float,
    sat_value: float,
    bw_value: float,
    reward: float,
    terminated: bool,
    truncated: bool,
    bw_weighted_workload_delta_reward: float = 0.0,
) -> None:
    append_single_env_step(
        buffer,
        env_index=0,
        accel_world_state=accel_world_state,
        sat_world_state=sat_world_state,
        bw_world_state=bw_world_state,
        next_world_state=next_world_state,
        accel_local_actor_state=None,
        sat_local_actor_state=None,
        bw_local_actor_state=None,
        accel_action=torch.zeros(1),
        sat_action=torch.zeros(1, dtype=torch.long),
        bw_action=torch.zeros(1),
        accel_old_logprob=torch.tensor(0.0),
        sat_old_logprob=torch.tensor(0.0),
        bw_old_logprob=torch.tensor(0.0),
        accel_value=torch.tensor(accel_value, dtype=torch.float32),
        sat_value=torch.tensor(sat_value, dtype=torch.float32),
        bw_value=torch.tensor(bw_value, dtype=torch.float32),
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        bw_weighted_workload_delta_reward=bw_weighted_workload_delta_reward,
    )


def test_structured_rollout_buffer_three_stage_gae():
    buffer = StructuredRolloutBuffer()
    ws0 = _dummy_world_state(0)
    ws1 = _dummy_world_state(1)
    ws2 = _dummy_world_state(2)
    ws3 = _dummy_world_state(0)
    _append_debug_env_step(
        buffer,
        accel_world_state=ws0,
        sat_world_state=ws1,
        bw_world_state=ws2,
        next_world_state=ws3,
        accel_value=1.0,
        sat_value=2.0,
        bw_value=3.0,
        reward=10.0,
        terminated=False,
        truncated=False,
    )
    out = buffer.compute_gae(gamma_env=0.9, gae_lambda=1.0, bootstrap_value=0.0)
    np.testing.assert_allclose(out["advantages"], np.array([9.0, 8.0, 7.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(out["returns"], np.array([10.0, 10.0, 10.0], dtype=np.float32), atol=1e-6)



def test_structured_rollout_buffer_step_level_single_step_differs_from_stage_chained():
    buffer = StructuredRolloutBuffer()
    ws0 = _dummy_world_state(0)
    ws1 = _dummy_world_state(1)
    ws2 = _dummy_world_state(2)
    ws3 = _dummy_world_state(0)
    _append_debug_env_step(
        buffer,
        accel_world_state=ws0,
        sat_world_state=ws1,
        bw_world_state=ws2,
        next_world_state=ws3,
        accel_value=1.0,
        sat_value=2.0,
        bw_value=3.0,
        reward=10.0,
        terminated=False,
        truncated=False,
    )
    stage_chained = buffer.compute_gae(gamma_env=0.9, gae_lambda=0.5, bootstrap_value=0.0, mode="stage_chained")
    step_level = buffer.compute_gae(gamma_env=0.9, gae_lambda=0.5, bootstrap_value=0.0, mode="step_level")
    np.testing.assert_allclose(stage_chained["returns"], np.array([4.25, 6.5, 10.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(step_level["returns"], np.array([10.0, 10.0, 10.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(step_level["advantages"], np.array([9.0, 8.0, 7.0], dtype=np.float32), atol=1e-6)


def test_structured_rollout_buffer_step_level_two_step_recursion():
    buffer = StructuredRolloutBuffer()
    ws0 = _dummy_world_state(0)
    ws1 = _dummy_world_state(1)
    ws2 = _dummy_world_state(2)
    ws3 = _dummy_world_state(0)
    ws4 = _dummy_world_state(1)
    ws5 = _dummy_world_state(2)
    ws6 = _dummy_world_state(0)
    _append_debug_env_step(
        buffer,
        accel_world_state=ws0,
        sat_world_state=ws1,
        bw_world_state=ws2,
        next_world_state=ws3,
        accel_value=1.0,
        sat_value=2.0,
        bw_value=3.0,
        reward=1.0,
        terminated=False,
        truncated=False,
    )
    _append_debug_env_step(
        buffer,
        accel_world_state=ws3,
        sat_world_state=ws4,
        bw_world_state=ws5,
        next_world_state=ws6,
        accel_value=4.0,
        sat_value=5.0,
        bw_value=6.0,
        reward=10.0,
        terminated=True,
        truncated=False,
    )
    out = buffer.compute_gae(gamma_env=0.9, gae_lambda=0.5, bootstrap_value=0.0, mode="step_level")
    np.testing.assert_allclose(out["returns"], np.array([7.3, 7.3, 7.3, 10.0, 10.0, 10.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(out["advantages"], np.array([6.3, 5.3, 4.3, 6.0, 5.0, 4.0], dtype=np.float32), atol=1e-6)


def test_structured_rollout_buffer_step_level_bw_gae_uses_shared_step_bootstrap():
    buffer = StructuredRolloutBuffer()
    ws0 = _dummy_world_state(0)
    ws1 = _dummy_world_state(1)
    ws2 = _dummy_world_state(2)
    ws3 = _dummy_world_state(0)
    _append_debug_env_step(
        buffer,
        accel_world_state=ws0,
        sat_world_state=ws1,
        bw_world_state=ws2,
        next_world_state=ws3,
        accel_value=1.0,
        sat_value=2.0,
        bw_value=3.0,
        reward=1.0,
        terminated=False,
        truncated=False,
        bw_weighted_workload_delta_reward=10.0,
    )
    out = buffer.compute_gae(
        gamma_env=0.9,
        gae_lambda=0.0,
        bootstrap_values={0: 5.0},
        mode="step_level",
        bw_target_mode="weighted_workload_delta",
        bw_return_mode="gae",
    )
    np.testing.assert_allclose(out["returns"], np.array([5.5, 5.5, 14.5], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(out["advantages"], np.array([4.5, 3.5, 11.5], dtype=np.float32), atol=1e-6)


def test_structured_rollout_buffer_step_level_bw_step_lambda_return_skips_immediate_bootstrap():
    buffer = StructuredRolloutBuffer()
    ws0 = _dummy_world_state(0)
    ws1 = _dummy_world_state(1)
    ws2 = _dummy_world_state(2)
    ws3 = _dummy_world_state(0)
    _append_debug_env_step(
        buffer,
        accel_world_state=ws0,
        sat_world_state=ws1,
        bw_world_state=ws2,
        next_world_state=ws3,
        accel_value=1.0,
        sat_value=2.0,
        bw_value=3.0,
        reward=1.0,
        terminated=False,
        truncated=False,
        bw_weighted_workload_delta_reward=10.0,
    )
    out = buffer.compute_gae(
        gamma_env=0.9,
        gae_lambda=0.0,
        bootstrap_values={0: 5.0},
        mode="step_level",
        bw_target_mode="weighted_workload_delta",
        bw_return_mode="step_lambda_return",
    )
    np.testing.assert_allclose(out["returns"], np.array([5.5, 5.5, 10.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(out["advantages"], np.array([4.5, 3.5, 7.0], dtype=np.float32), atol=1e-6)

    alias_out = buffer.compute_gae(
        gamma_env=0.9,
        gae_lambda=0.0,
        bootstrap_values={0: 5.0},
        mode="step_level",
        bw_target_mode="weighted_workload_delta",
        bw_return_mode="monte_carlo",
    )
    np.testing.assert_allclose(alias_out["returns"], out["returns"], atol=1e-6)


def test_structured_rollout_buffer_step_level_time_limit_is_finite_horizon_terminal():
    buffer = StructuredRolloutBuffer()
    ws0 = _dummy_world_state(0)
    ws1 = _dummy_world_state(1)
    ws2 = _dummy_world_state(2)
    ws3 = _dummy_world_state(0)
    _append_debug_env_step(
        buffer,
        accel_world_state=ws0,
        sat_world_state=ws1,
        bw_world_state=ws2,
        next_world_state=ws3,
        accel_value=1.0,
        sat_value=2.0,
        bw_value=3.0,
        reward=1.0,
        terminated=False,
        truncated=True,
    )
    for bw_return_mode in ("gae", "step_lambda_return", "monte_carlo", "bw_episode_mc"):
        out = buffer.compute_gae(
            gamma_env=0.9,
            gae_lambda=0.5,
            bootstrap_value=0.0,
            truncated_bootstrap_values={2: 5.0},
            mode="step_level",
            bw_return_mode=bw_return_mode,
        )
        np.testing.assert_allclose(out["returns"], np.array([1.0, 1.0, 1.0], dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(out["advantages"], np.array([0.0, -1.0, -2.0], dtype=np.float32), atol=1e-6)


def test_structured_rollout_buffer_step_level_time_limit_bootstrap_is_opt_in():
    buffer = StructuredRolloutBuffer()
    ws0 = _dummy_world_state(0)
    ws1 = _dummy_world_state(1)
    ws2 = _dummy_world_state(2)
    ws3 = _dummy_world_state(0)
    _append_debug_env_step(
        buffer,
        accel_world_state=ws0,
        sat_world_state=ws1,
        bw_world_state=ws2,
        next_world_state=ws3,
        accel_value=1.0,
        sat_value=2.0,
        bw_value=3.0,
        reward=1.0,
        terminated=False,
        truncated=True,
    )
    out = buffer.compute_gae(
        gamma_env=0.9,
        gae_lambda=0.5,
        bootstrap_value=0.0,
        truncated_bootstrap_values={2: 5.0},
        bootstrap_truncated=True,
        mode="step_level",
        bw_return_mode="bw_episode_mc",
    )
    np.testing.assert_allclose(out["returns"], np.array([5.5, 5.5, 5.5], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(out["advantages"], np.array([4.5, 3.5, 2.5], dtype=np.float32), atol=1e-6)


def test_structured_rollout_buffer_step_level_bw_episode_mc_uses_bw_reward_to_episode_end():
    buffer = StructuredRolloutBuffer()
    ws0 = _dummy_world_state(0)
    ws1 = _dummy_world_state(1)
    ws2 = _dummy_world_state(2)
    ws3 = _dummy_world_state(0)
    ws4 = _dummy_world_state(1)
    ws5 = _dummy_world_state(2)
    ws6 = _dummy_world_state(0)
    _append_debug_env_step(
        buffer,
        accel_world_state=ws0,
        sat_world_state=ws1,
        bw_world_state=ws2,
        next_world_state=ws3,
        accel_value=1.0,
        sat_value=2.0,
        bw_value=3.0,
        reward=1.0,
        terminated=False,
        truncated=False,
        bw_weighted_workload_delta_reward=1.0,
    )
    _append_debug_env_step(
        buffer,
        accel_world_state=ws3,
        sat_world_state=ws4,
        bw_world_state=ws5,
        next_world_state=ws6,
        accel_value=4.0,
        sat_value=5.0,
        bw_value=6.0,
        reward=10.0,
        terminated=True,
        truncated=False,
        bw_weighted_workload_delta_reward=10.0,
    )
    out = buffer.compute_gae(
        gamma_env=0.9,
        gae_lambda=0.0,
        bootstrap_values={0: 5.0},
        mode="step_level",
        bw_target_mode="weighted_workload_delta",
        bw_return_mode="bw_episode_mc",
    )
    np.testing.assert_allclose(out["returns"], np.array([4.6, 4.6, 10.0, 10.0, 10.0, 10.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(out["advantages"], np.array([3.6, 2.6, 7.0, 6.0, 5.0, 4.0], dtype=np.float32), atol=1e-6)


def test_structured_rollout_buffer_step_level_bw_gae_uses_next_bw_value_not_accel_value():
    buffer = StructuredRolloutBuffer()
    ws0 = _dummy_world_state(0)
    ws1 = _dummy_world_state(1)
    ws2 = _dummy_world_state(2)
    ws3 = _dummy_world_state(0)
    ws4 = _dummy_world_state(1)
    ws5 = _dummy_world_state(2)
    ws6 = _dummy_world_state(0)
    _append_debug_env_step(
        buffer,
        accel_world_state=ws0,
        sat_world_state=ws1,
        bw_world_state=ws2,
        next_world_state=ws3,
        accel_value=100.0,
        sat_value=2.0,
        bw_value=3.0,
        reward=0.0,
        terminated=False,
        truncated=False,
        bw_weighted_workload_delta_reward=10.0,
    )
    _append_debug_env_step(
        buffer,
        accel_world_state=ws3,
        sat_world_state=ws4,
        bw_world_state=ws5,
        next_world_state=ws6,
        accel_value=50.0,
        sat_value=5.0,
        bw_value=7.0,
        reward=0.0,
        terminated=True,
        truncated=False,
        bw_weighted_workload_delta_reward=1.0,
    )
    out = buffer.compute_gae(
        gamma_env=0.9,
        gae_lambda=0.5,
        bootstrap_values={0: 999.0},
        mode="step_level",
        bw_target_mode="weighted_workload_delta",
        bw_return_mode="bw_gae",
    )
    np.testing.assert_allclose(out["returns"][[2, 5]], np.array([13.6, 1.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(out["advantages"][[2, 5]], np.array([10.6, -6.0], dtype=np.float32), atol=1e-6)


def test_structured_rollout_buffer_preserves_bw_old_logprob_per_agent():
    buffer = StructuredRolloutBuffer()
    ws0 = _dummy_world_state(2)
    ws1 = _dummy_world_state(0)
    per_agent = torch.tensor([0.1, -0.2], dtype=torch.float32)
    append_single_env_step(
        buffer,
        env_index=0,
        accel_world_state=_dummy_world_state(0),
        sat_world_state=_dummy_world_state(1),
        bw_world_state=ws0,
        next_world_state=ws1,
        accel_local_actor_state=None,
        sat_local_actor_state=None,
        bw_local_actor_state=None,
        accel_action=torch.zeros((2, 3), dtype=torch.float32),
        sat_action=torch.zeros((2,), dtype=torch.long),
        bw_action=torch.zeros((2, 3), dtype=torch.float32),
        accel_old_logprob=torch.tensor(0.0),
        sat_old_logprob=torch.tensor(0.0),
        bw_old_logprob=torch.tensor(0.0),
        accel_value=torch.tensor(0.0),
        sat_value=torch.tensor(0.0),
        bw_value=torch.tensor(1.0),
        reward=0.5,
        terminated=False,
        truncated=False,
        bw_old_logprob_per_agent=per_agent,
    )
    assert len(buffer._records) == 1
    record = buffer._records[0]
    assert record.bw_old_logprobs_per_agent is not None
    torch.testing.assert_close(record.bw_old_logprobs_per_agent.reshape(1, -1)[0], per_agent)

def test_structured_control_driver_three_stage_step():
    cfg = SaginConfig(
        seed=0,
        num_uav=2,
        num_gu=4,
        num_sat=12,
        users_obs_max=4,
        sats_obs_max=4,
        nbrs_obs_max=1,
        sat_num_select=2,
        danger_imitation_enabled=True,
        danger_imitation_coef=0.1,
    )
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed)
    driver = StructuredControlDriver(env)

    z0 = driver.begin_step()
    assert z0.stage_id.item() == driver.STAGE_ACCEL
    assert z0.uav_nodes.shape[1] == cfg.num_uav
    assert z0.gu_nodes.shape[1] == cfg.num_gu

    accel = np.zeros((cfg.num_uav, 2), dtype=np.float32)
    z1 = driver.run_accel_stage(accel)
    assert z1.stage_id.item() == driver.STAGE_SAT

    sat_states = driver.build_local_sat_states(z1)
    subset_indices = []
    for state in sat_states:
        valid = torch.nonzero(state.subset_mask[0], as_tuple=False).flatten()
        subset_indices.append(int(valid[0].item()) if valid.numel() > 0 else -1)
    sat_actions = driver.decode_sat_subset_actions(sat_states, subset_indices)
    z2 = driver.run_sat_stage(sat_actions)
    assert z2.stage_id.item() == driver.STAGE_BW

    bw_states = driver.build_bw_valid_context(z2)
    assert bw_states[0].selected_sat_tokens.shape[-2] == sat_actions.shape[-1]
    assert bw_states[0].gu_tokens.shape[-2] == cfg.num_gu
    bw_action = _uniform_full_g_bw_action(cfg, bw_states)
    result = driver.execute_stage_bw_and_step(bw_action)
    assert env.t == 1
    assert set(result.obs.keys()) == set(env.agents)
    assert set(result.rewards.keys()) == set(env.agents)
    assert all(np.isfinite(v) for v in result.rewards.values())
    assert result.danger_imitation_target.shape == (cfg.num_uav, 2)
    assert result.danger_imitation_mask.shape == (cfg.num_uav, 2)
    assert np.all(np.isfinite(result.danger_imitation_target))
    assert np.all((result.danger_imitation_mask == 0.0) | (result.danger_imitation_mask == 1.0))
    sample_obs = next(iter(result.obs.values()))
    assert sample_obs["own"].shape == (env.own_dim,)


def test_structured_control_driver_bw_rewards_match_post_reward_parts():
    cfg = SaginConfig(
        seed=9,
        num_uav=1,
        num_gu=4,
        num_sat=6,
        users_obs_max=4,
        sats_obs_max=4,
        nbrs_obs_max=1,
        sat_num_select=2,
        T_steps=6,
        reward_mode="weighted_workload_level",
    )
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed)
    driver = StructuredControlDriver(env)

    driver.begin_step()
    accel = np.zeros((cfg.num_uav, 2), dtype=np.float32)
    z1 = driver.run_accel_stage(accel)
    sat_states = driver.build_local_sat_states(z1)
    subset_indices = []
    for state in sat_states:
        valid = torch.nonzero(state.subset_mask[0], as_tuple=False).flatten()
        subset_indices.append(int(valid[0].item()) if valid.numel() > 0 else -1)
    sat_actions = driver.decode_sat_subset_actions(sat_states, subset_indices)
    z2 = driver.run_sat_stage(sat_actions)
    bw_states = driver.build_bw_valid_context(z2)
    bw_action = _uniform_full_g_bw_action(cfg, bw_states)

    result = driver.execute_stage_bw_and_step(bw_action)
    parts = dict(env.last_reward_parts)
    reward_value = float(next(iter(result.rewards.values())))

    assert abs(result.bw_weighted_workload_delta_reward - float(parts["bw_weighted_workload_delta_reward"])) < 1e-9
    assert abs(result.bw_weighted_workload_level_reward - float(parts["bw_weighted_workload_level_reward"])) < 1e-9
    assert abs(result.bw_gu_queue_level_reward - float(parts["bw_gu_queue_level_reward"])) < 1e-9
    assert abs(result.bw_system_queue_level_reward - float(parts["bw_system_queue_level_reward"])) < 1e-9
    assert abs(result.bw_gu_service_queue_reward - float(parts["bw_gu_service_queue_reward"])) < 1e-9
    assert abs(reward_value - float(parts["bw_weighted_workload_level_reward"])) < 1e-9


def _eta_feature_from_slots(cfg: SaginConfig, candidates, eta_slots: np.ndarray) -> np.ndarray:
    eta_feature = np.zeros((cfg.num_uav, cfg.num_gu), dtype=np.float32)
    for u in range(cfg.num_uav):
        cand = candidates[u][: cfg.users_obs_max]
        for slot, gu_idx in enumerate(cand):
            eta_feature[u, int(gu_idx)] = float(eta_slots[u, slot])
    return eta_feature


def _as_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _access_se_ref_from_env(env: SaginParallelEnv, access_gain_matrix: np.ndarray | None) -> np.ndarray:
    cfg = env.cfg
    if access_gain_matrix is None:
        access_gain_matrix = env._compute_access_link_gain_matrix()
    access_gain = np.asarray(access_gain_matrix, dtype=np.float32).T
    access_snr = channel.snr_linear(
        cfg.gu_tx_power,
        access_gain,
        cfg.noise_density,
        cfg.b_acc,
        noise_figure_db=float(getattr(cfg, "access_noise_figure_db", 0.0) or 0.0),
    )
    if bool(cfg.fading_enabled) and channel.access_fading_mode_from_config(cfg) == "ergodic_rician":
        return np.asarray(
            channel.rician_ergodic_spectral_efficiency(
                access_snr,
                channel.rician_k_linear_from_config(cfg),
                quadrature_points=int(getattr(cfg, "access_ergodic_rician_quadrature_points", 16) or 16),
            ),
            dtype=np.float32,
        )
    return np.asarray(channel.spectral_efficiency(access_snr), dtype=np.float32)


def _log_ratio_np(value: np.ndarray, ref: float) -> np.ndarray:
    return np.log(
        np.maximum(np.asarray(value, dtype=np.float32), float(NORMALIZATION_DENOM_EPS))
        / max(float(ref), float(NORMALIZATION_DENOM_EPS))
    ).astype(np.float32)


def _assert_world_state_close(left: StructuredWorldState, right: StructuredWorldState) -> None:
    for field_name in StructuredWorldState.__dataclass_fields__:
        left_value = getattr(left, field_name)
        right_value = getattr(right, field_name)
        torch.testing.assert_close(
            torch.as_tensor(left_value),
            torch.as_tensor(right_value),
        )


def test_structured_world_state_carries_access_reference_se_feature():
    cfg = SaginConfig(
        seed=3,
        num_uav=2,
        num_gu=5,
        num_sat=6,
        users_obs_max=5,
        sats_obs_max=4,
        nbrs_obs_max=1,
        sat_num_select=2,
    )
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed)
    driver = StructuredControlDriver(env)

    z0 = driver.begin_step()
    expected_begin = _access_se_ref_from_env(env, driver._stage_access_gain_matrix)
    np.testing.assert_allclose(
        z0.uav_gu_edges[0, :, :, critic_schema.UG_ACCESS_SE_REF],
        expected_begin,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        z0.uav_gu_edges[0, :, :, critic_schema.UG_PREFIX_BW_VALID_KNOWN],
        np.zeros((cfg.num_uav, cfg.num_gu), dtype=np.float32),
        atol=1e-6,
    )

    accel = np.zeros((cfg.num_uav, 2), dtype=np.float32)
    z1 = driver.run_accel_stage(accel)
    expected_after_accel = _access_se_ref_from_env(env, driver._stage_access_gain_matrix)
    np.testing.assert_allclose(
        z1.uav_gu_edges[0, :, :, critic_schema.UG_ACCESS_SE_REF],
        expected_after_accel,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        z1.uav_gu_edges[0, :, :, critic_schema.UG_PREFIX_BW_VALID_KNOWN],
        np.ones((cfg.num_uav, cfg.num_gu), dtype=np.float32),
        atol=1e-6,
    )


def test_structured_world_state_uses_visible_sat_union_not_stage_active_cache():
    cfg = SaginConfig(
        seed=4,
        num_uav=2,
        num_gu=5,
        num_sat=6,
        users_obs_max=5,
        sats_obs_max=4,
        nbrs_obs_max=1,
        sat_num_select=2,
    )
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed)
    driver = StructuredControlDriver(env)

    accel = np.zeros((cfg.num_uav, 2), dtype=np.float32)
    driver.run_accel_stage(accel)
    expected_sat_ids = driver._critic_sat_ids_from_visible(driver._stage_visible, None, driver.STAGE_SAT)

    driver._stage_active_sat_ids = np.asarray([-999], dtype=np.int32)
    rebuilt = driver._build_world_state(
        stage_id=driver.STAGE_SAT,
        assoc=driver._stage_assoc,
        candidates=driver._stage_candidates,
        sat_pos=driver._stage_sat_pos,
        sat_vel=driver._stage_sat_vel,
        visible=driver._stage_visible,
        sat_selection=None,
    )
    np.testing.assert_array_equal(_as_numpy(rebuilt.sat_ids[0]), expected_sat_ids)


def test_structured_world_state_rejects_bw_selection_outside_visible_union():
    cfg = SaginConfig(
        seed=4,
        num_uav=2,
        num_gu=5,
        num_sat=12,
        users_obs_max=5,
        sats_obs_max=4,
        visible_sats_max=2,
        nbrs_obs_max=1,
        sat_num_select=1,
    )
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed)
    driver = StructuredControlDriver(env)

    accel = np.zeros((cfg.num_uav, 2), dtype=np.float32)
    driver.run_accel_stage(accel)
    visible_union = {int(sat_idx) for visible in driver._stage_visible for sat_idx in visible}
    missing = next(int(sat_idx) for sat_idx in range(cfg.num_sat) if sat_idx not in visible_union)
    bad_selection = [list(driver._stage_visible[u][:1]) for u in range(cfg.num_uav)]
    bad_selection[0] = [missing]

    with pytest.raises(RuntimeError, match="BW prefix selected SATs are not present"):
        driver._build_world_state(
            stage_id=driver.STAGE_BW,
            assoc=driver._stage_assoc,
            candidates=driver._stage_candidates,
            sat_pos=driver._stage_sat_pos,
            sat_vel=driver._stage_sat_vel,
            visible=driver._stage_visible,
            sat_selection=bad_selection,
        )


def test_load_bw_stage_state_restores_cached_eta_without_recompute():
    cfg = SaginConfig(
        seed=41,
        num_uav=1,
        num_gu=4,
        num_sat=5,
        users_obs_max=4,
        sats_obs_max=4,
        nbrs_obs_max=1,
        sat_num_select=1,
        enable_bw_action=True,
        fading_enabled=True,
    )
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed)
    driver = StructuredControlDriver(env)

    z1 = driver.run_accel_stage(np.zeros((cfg.num_uav, 2), dtype=np.float32))
    sat_states = driver.build_local_sat_states(z1)
    subset_indices = []
    for state in sat_states:
        valid = torch.nonzero(state.subset_mask[0], as_tuple=False).flatten()
        subset_indices.append(int(valid[0].item()) if valid.numel() > 0 else -1)
    driver.run_sat_stage(driver.decode_sat_subset_actions(sat_states, subset_indices))

    snapshot = driver.export_bw_stage_state()
    assert snapshot.get("stage_cached_eta") is not None
    expected_eta = np.asarray(snapshot["stage_cached_eta"], dtype=np.float32)

    def _unexpected_recompute(*_args, **_kwargs):
        raise AssertionError("load_bw_stage_state should restore cached eta without recompute when snapshot provides it")

    env._compute_access_rates = _unexpected_recompute  # type: ignore[method-assign]
    driver.load_bw_stage_state(snapshot)

    np.testing.assert_allclose(env._cached_eta, expected_eta, atol=1e-6)


def test_bw_replay_branches_share_exogenous_randomness_with_ref_branch():
    cfg = SaginConfig(
        seed=43,
        num_uav=1,
        num_gu=4,
        num_sat=5,
        users_obs_max=4,
        sats_obs_max=4,
        nbrs_obs_max=1,
        sat_num_select=1,
        enable_bw_action=True,
        fading_enabled=True,
        task_arrival_poisson=True,
    )
    env = SaginParallelEnv(cfg)
    ref_env = SaginParallelEnv(cfg)
    pert_env = SaginParallelEnv(cfg)
    try:
        env.reset(seed=cfg.seed)
        ref_env.reset(seed=cfg.seed)
        pert_env.reset(seed=cfg.seed)
        driver = StructuredControlDriver(env)
        ref_driver = StructuredControlDriver(ref_env)
        pert_driver = StructuredControlDriver(pert_env)

        driver.begin_step()
        driver.run_accel_stage(np.zeros((cfg.num_uav, 2), dtype=np.float32))
        driver.run_sat_stage(_default_sat_action(driver))
        snapshot = driver.export_bw_stage_state()

        valid_mask = np.asarray(driver._stage_bw_valid_mask, dtype=bool)
        valid_slots = np.flatnonzero(valid_mask[0])
        assert valid_slots.size >= 2

        ref_action = np.zeros((cfg.num_uav, cfg.num_gu), dtype=np.float32)
        ref_action[0, valid_slots] = 1.0 / float(valid_slots.size)
        pert_action = ref_action.copy()
        donor = int(valid_slots[0])
        receiver = int(valid_slots[1])
        delta = min(0.1, float(ref_action[0, donor]) * 0.5)
        assert delta > 1.0e-8
        pert_action[0, donor] -= float(delta)
        pert_action[0, receiver] += float(delta)

        ref_driver.load_bw_stage_state(snapshot)
        pert_driver.load_bw_stage_state(snapshot)
        ref_rng_before = _digest_rng_state(copy.deepcopy(ref_env.rng.bit_generator.state))
        pert_rng_before = _digest_rng_state(copy.deepcopy(pert_env.rng.bit_generator.state))
        assert ref_rng_before == pert_rng_before

        ref_driver.execute_stage_bw_and_prepare_next_accel(ref_action)
        pert_driver.execute_stage_bw_and_prepare_next_accel(pert_action)

        ref_rng_after_bw = _digest_rng_state(copy.deepcopy(ref_env.rng.bit_generator.state))
        pert_rng_after_bw = _digest_rng_state(copy.deepcopy(pert_env.rng.bit_generator.state))
        assert ref_rng_after_bw == pert_rng_after_bw
        np.testing.assert_array_equal(ref_env.last_gu_arrival, pert_env.last_gu_arrival)
        np.testing.assert_array_equal(ref_env.last_gu_arrival_rate_vec, pert_env.last_gu_arrival_rate_vec)
        np.testing.assert_array_equal(
            np.asarray(getattr(ref_env, "last_hotspot_mask", np.zeros((cfg.num_gu,), dtype=np.float32))),
            np.asarray(getattr(pert_env, "last_hotspot_mask", np.zeros((cfg.num_gu,), dtype=np.float32))),
        )

        zero_accel = np.zeros((cfg.num_uav, 2), dtype=np.float32)
        ref_driver.run_accel_stage(zero_accel)
        ref_driver.run_sat_stage(_default_sat_action(ref_driver))
        ref_next_snapshot = ref_driver.export_bw_stage_state()

        pert_driver.run_accel_stage(zero_accel)
        pert_driver.run_sat_stage(_default_sat_action(pert_driver))
        pert_next_snapshot = pert_driver.export_bw_stage_state()

        ref_rng_after_follow = _digest_rng_state(copy.deepcopy(ref_env.rng.bit_generator.state))
        pert_rng_after_follow = _digest_rng_state(copy.deepcopy(pert_env.rng.bit_generator.state))
        assert ref_rng_after_follow == pert_rng_after_follow
        np.testing.assert_allclose(
            np.asarray(ref_next_snapshot["stage_cached_eta"], dtype=np.float32),
            np.asarray(pert_next_snapshot["stage_cached_eta"], dtype=np.float32),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(ref_next_snapshot["stage_world_cache"]["stage_access_gain_matrix"], dtype=np.float32),
            np.asarray(pert_next_snapshot["stage_world_cache"]["stage_access_gain_matrix"], dtype=np.float32),
            atol=1e-6,
        )
    finally:
        for close_env in (env, ref_env, pert_env):
            close_fn = getattr(close_env, "close", None)
            if callable(close_fn):
                close_fn()


def test_bw_weighted_workload_device_costs_use_per_entity_nested_ema():
    cfg = SaginConfig(
        seed=5,
        num_uav=2,
        num_gu=2,
        num_sat=3,
        users_obs_max=2,
        sats_obs_max=3,
        nbrs_obs_max=1,
        sat_num_select=2,
    )
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed)
    driver = StructuredControlDriver(env)

    env.bw_weighted_workload_acc_ema_vec = np.asarray([2.0, 4.0], dtype=np.float32)
    env.bw_weighted_workload_rel_ema_vec = np.asarray([5.0, 10.0], dtype=np.float32)
    env.bw_weighted_workload_sat_ema_vec = np.asarray([20.0, 40.0, 80.0], dtype=np.float32)
    env.last_association = np.asarray([0, 1], dtype=np.int32)
    env.last_sat_selection = [[0, 2], [1]]

    gu_cost, uav_cost, sat_cost = driver._bw_weighted_workload_device_costs()

    expected_sat_cost = np.asarray([0.05, 0.025, 0.0125], dtype=np.float32)
    expected_uav_cost = np.asarray(
        [
            1.0 / 5.0 + np.mean(expected_sat_cost[[0, 2]]),
            1.0 / 10.0 + np.mean(expected_sat_cost[[1]]),
        ],
        dtype=np.float32,
    )
    expected_gu_cost = np.asarray(
        [
            1.0 / 2.0 + expected_uav_cost[0],
            1.0 / 4.0 + expected_uav_cost[1],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(sat_cost, expected_sat_cost, atol=1e-6)
    np.testing.assert_allclose(uav_cost, expected_uav_cost, atol=1e-6)
    np.testing.assert_allclose(gu_cost, expected_gu_cost, atol=1e-6)


def test_bw_weighted_workload_ema_updates_per_entity_and_keeps_scalar_sums():
    cfg = SaginConfig(
        seed=6,
        num_uav=2,
        num_gu=2,
        num_sat=3,
        users_obs_max=2,
        sats_obs_max=3,
        nbrs_obs_max=1,
        sat_num_select=2,
        bw_weighted_workload_ema_decay=0.5,
    )
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed)
    driver = StructuredControlDriver(env)

    env.bw_weighted_workload_acc_ema_vec = np.asarray([10.0, 20.0], dtype=np.float32)
    env.bw_weighted_workload_rel_ema_vec = np.asarray([30.0, 40.0], dtype=np.float32)
    env.bw_weighted_workload_sat_ema_vec = np.asarray([50.0, 60.0, 70.0], dtype=np.float32)
    env.last_gu_outflow = np.asarray([1.0, 2.0], dtype=np.float32)
    env.last_uav_outflow = np.asarray([3.0, 4.0], dtype=np.float32)
    env.last_sat_processed = np.asarray([5.0, 6.0, 7.0], dtype=np.float32)

    driver._update_bw_weighted_workload_ema()

    expected_gu = np.asarray([5.5, 11.0], dtype=np.float32)
    expected_uav = np.asarray([16.5, 22.0], dtype=np.float32)
    expected_sat = np.asarray([27.5, 33.0, 38.5], dtype=np.float32)

    np.testing.assert_allclose(env.bw_weighted_workload_acc_ema_vec, expected_gu, atol=1e-6)
    np.testing.assert_allclose(env.bw_weighted_workload_rel_ema_vec, expected_uav, atol=1e-6)
    np.testing.assert_allclose(env.bw_weighted_workload_sat_ema_vec, expected_sat, atol=1e-6)
    np.testing.assert_allclose(env.bw_weighted_workload_acc_ema, float(np.sum(expected_gu)), atol=1e-6)
    np.testing.assert_allclose(env.bw_weighted_workload_rel_ema, float(np.sum(expected_uav)), atol=1e-6)
    np.testing.assert_allclose(env.bw_weighted_workload_sat_ema, float(np.sum(expected_sat)), atol=1e-6)


def test_structured_bw_reward_aligned_obs_use_stage_cache_instead_of_env_last_state():
    cfg = SaginConfig(
        seed=17,
        num_uav=2,
        num_gu=6,
        num_sat=5,
        users_obs_max=6,
        sats_obs_max=4,
        nbrs_obs_max=1,
        sat_num_select=1,
        enable_bw_action=True,
        obs_own_include_assoc_uav_cost=True,
        obs_user_include_local_gu_service_cost=True,
        obs_user_include_weighted_queue_cost=True,
        obs_user_include_weighted_queue_cost_relative=True,
        obs_sat_include_sat_cost=True,
    )
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed)
    driver = StructuredControlDriver(env)

    accel = np.zeros((cfg.num_uav, 2), dtype=np.float32)
    z1 = driver.run_accel_stage(accel)
    sat_states = driver.build_local_sat_states(z1)
    subset_indices = []
    for state in sat_states:
        valid = torch.nonzero(state.subset_mask[0], as_tuple=False).flatten()
        subset_indices.append(int(valid[0].item()) if valid.numel() > 0 else -1)
    driver.run_sat_stage(driver.decode_sat_subset_actions(sat_states, subset_indices))

    env.bw_weighted_workload_acc_ema_vec = np.asarray([2.0, 4.0, 8.0, 16.0, 32.0, 64.0], dtype=np.float32)
    env.bw_weighted_workload_rel_ema_vec = np.asarray([10.0, 20.0], dtype=np.float32)
    env.bw_weighted_workload_sat_ema_vec = np.asarray([20.0, 40.0, 80.0, 160.0, 320.0], dtype=np.float32)
    env.gu_queue = np.asarray([100.0, 220.0, 340.0, 460.0, 580.0, 700.0], dtype=np.float32)

    stage_assoc = np.asarray(driver._stage_assoc, dtype=np.int32).copy()
    stage_sat_selection = [list(sel) for sel in driver._stage_sat_selection]

    stale_assoc = stage_assoc.copy()
    valid_assoc = stale_assoc >= 0
    stale_assoc[valid_assoc] = (stale_assoc[valid_assoc] + 1) % cfg.num_uav
    env.last_association = stale_assoc
    stale_sat_selection: list[list[int]] = []
    for u in range(cfg.num_uav):
        current = list(stage_sat_selection[u])
        if current:
            stale_sat_selection.append([int((current[0] + 1) % cfg.num_sat)])
        else:
            stale_sat_selection.append([int(u % cfg.num_sat)])
    env.last_sat_selection = stale_sat_selection

    rebuilt = driver._build_world_state(
        stage_id=driver.STAGE_BW,
        assoc=driver._stage_assoc,
        candidates=driver._stage_candidates,
        sat_pos=driver._stage_sat_pos,
        sat_vel=driver._stage_sat_vel,
        visible=driver._stage_visible,
        sat_selection=driver._stage_sat_selection,
    )

    prefix_gu_cost, prefix_uav_cost, _prefix_sat_cost = env._bw_weighted_workload_device_costs(
        assoc_override=stage_assoc,
        sat_selection_override=stage_sat_selection,
    )
    last_gu_cost, last_uav_cost, sat_cost = env._bw_weighted_workload_device_costs()
    arrival_ref = env._arrival_ref()
    gu_flow_ref = arrival_ref / float(cfg.num_gu)
    uav_flow_ref = arrival_ref / float(cfg.num_uav)
    sat_flow_ref = arrival_ref / float(env._bw_weighted_workload_sat_active_ref_count())
    gu_total_cost_ref = 1.0 / gu_flow_ref + 1.0 / uav_flow_ref + 1.0 / sat_flow_ref
    uav_total_cost_ref = 1.0 / uav_flow_ref + 1.0 / sat_flow_ref
    sat_cost_ref = 1.0 / sat_flow_ref

    np.testing.assert_allclose(
        rebuilt.gu_nodes[0, :, critic_schema.GU_PREFIX_TOTAL_COST_LOG_RATIO],
        _log_ratio_np(prefix_gu_cost, gu_total_cost_ref),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        rebuilt.gu_nodes[0, :, critic_schema.GU_LAST_TOTAL_COST_LOG_RATIO],
        _log_ratio_np(last_gu_cost, gu_total_cost_ref),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        rebuilt.uav_nodes[0, :, critic_schema.UAV_PREFIX_TOTAL_COST_LOG_RATIO],
        _log_ratio_np(prefix_uav_cost, uav_total_cost_ref),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        rebuilt.uav_nodes[0, :, critic_schema.UAV_LAST_TOTAL_COST_LOG_RATIO],
        _log_ratio_np(last_uav_cost, uav_total_cost_ref),
        atol=1e-6,
    )
    valid_sat_ids = _as_numpy(rebuilt.sat_ids[0])
    valid_sat_ids = valid_sat_ids[valid_sat_ids >= 0]
    np.testing.assert_allclose(
        rebuilt.sat_nodes[0, : valid_sat_ids.size, critic_schema.SAT_COST_LOG_RATIO],
        _log_ratio_np(sat_cost[valid_sat_ids], sat_cost_ref),
        atol=1e-6,
    )

    assert np.any(
        np.abs(
            _as_numpy(rebuilt.gu_nodes[0, :, critic_schema.GU_PREFIX_TOTAL_COST_LOG_RATIO])
            - _as_numpy(rebuilt.gu_nodes[0, :, critic_schema.GU_LAST_TOTAL_COST_LOG_RATIO])
        )
        > 1e-6
    )
    assert np.all(_as_numpy(rebuilt.gu_nodes[0, :, critic_schema.GU_PREFIX_COST_KNOWN]) == 1.0)
    assert np.all(_as_numpy(rebuilt.uav_nodes[0, :, critic_schema.UAV_PREFIX_COST_KNOWN]) == 1.0)


def test_structured_driver_bw_core_returns_materialized_step_outputs():
    cfg = SaginConfig(
        seed=91,
        num_uav=2,
        num_gu=6,
        num_sat=5,
        users_obs_max=6,
        sats_obs_max=4,
        nbrs_obs_max=1,
        sat_num_select=1,
    )
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed)
    driver = StructuredControlDriver(env)

    accel_action = np.zeros((cfg.num_uav, 2), dtype=np.float32)
    sat_world = driver.run_accel_stage(accel_action)
    sat_states = driver.build_local_sat_states(sat_world)
    subset_indices = [
        int(state.subset_mask[0].nonzero(as_tuple=False).flatten()[0].item())
        if bool(state.subset_mask[0].any().item())
        else -1
        for state in sat_states
    ]
    sat_action = driver.decode_sat_subset_actions(sat_states, subset_indices)
    driver.run_sat_stage(sat_action)
    bw_action = _uniform_full_g_bw_action(cfg, driver.build_bw_valid_context())

    step_result = driver.execute_stage_bw_and_step(bw_action)

    assert isinstance(step_result.rewards, dict)
    assert isinstance(step_result.obs, dict)
    assert set(step_result.rewards) == set(env.agents)
    assert set(step_result.terminations) == set(env.agents)
    assert set(step_result.truncations) == set(env.agents)
