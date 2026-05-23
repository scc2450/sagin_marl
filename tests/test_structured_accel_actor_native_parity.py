from __future__ import annotations

import numpy as np
import pytest
import torch

from sagin_marl.env import native_cuda
from sagin_marl.env.config import SaginConfig
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.env.structured_gpu_rollout_runtime import StructuredGpuAccelObsView
from sagin_marl.rl.native_actor_cuda import build_native_actor_cuda_binding
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_stage_builders import build_batched_local_accel_states_from_spec
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_env_group
from sagin_marl.rl.structured_types import LocalAccelState


def _native_cfg() -> SaginConfig:
    return SaginConfig(
        seed=321,
        num_uav=3,
        num_gu=5,
        num_sat=7,
        users_obs_max=5,
        sats_obs_max=4,
        visible_sats_max=4,
        per_uav_visible_sat_token_max=4,
        nbrs_obs_max=2,
        sat_num_select=2,
        candidate_mode="nearest",
        candidate_k=5,
        access_fading_mode="large_scale",
        fading_enabled=False,
        fixed_satellite_strategy=False,
        structured_env_backend="native",
        structured_env_tensor_backend="cuda",
        T_steps=4,
        accel_gu_query_count=4,
        accel_peer_query_count=2,
        accel_sat_query_count=2,
    )


def _install_nontrivial_slot_state(slot_env) -> None:
    cfg = slot_env.cfg
    slot_env.uav_pos = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]], dtype=np.float32)
    slot_env.uav_vel = np.array([[1.0, 0.5], [-1.0, 2.0], [0.25, -1.5]], dtype=np.float32)
    slot_env.gu_pos = np.array([[8.0, 0.0], [96.0, 4.0], [152.0, 0.0], [194.0, 6.0], [102.0, 44.0]], dtype=np.float32)
    slot_env.gu_queue = np.array([5.0, 7.0, 11.0, 13.0, 17.0], dtype=np.float32)
    slot_env.uav_queue = np.array([19.0, 23.0, 29.0], dtype=np.float32)
    slot_env.sat_queue = np.linspace(3.0, 15.0, int(cfg.num_sat), dtype=np.float32)
    slot_env.gu_drop = np.array([0.0, 1.0, 0.5, 2.0, 0.0], dtype=np.float32)
    slot_env.uav_drop = np.array([0.25, 0.0, 0.75], dtype=np.float32)
    slot_env.sat_drop = np.linspace(0.0, 1.2, int(cfg.num_sat), dtype=np.float32)
    slot_env.last_association = np.array([0, 1, 2, 1, -1], dtype=np.int32)
    slot_env.last_gu_arrival = np.array([1.0, 0.5, 2.0, 0.25, 1.5], dtype=np.float32)
    slot_env.last_gu_outflow = np.array([0.4, 0.7, 0.2, 1.0, 0.0], dtype=np.float32)
    slot_env.last_bw_fraction_by_uav_gu = np.array(
        [
            [0.6, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.4, 0.0, 0.3, 0.0],
            [0.0, 0.0, 0.8, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    slot_env.last_gu_to_uav_inflow_by_uav = np.array([0.5, 1.5, 0.25], dtype=np.float32)
    slot_env.last_uav_to_sat_outflow_matrix = np.full((int(cfg.num_uav), int(cfg.num_sat)), 0.05, dtype=np.float32)
    slot_env.last_selected_mask_by_uav_sat = np.zeros((int(cfg.num_uav), int(cfg.num_sat)), dtype=np.float32)
    slot_env.last_selected_mask_by_uav_sat[0, [0, 1]] = 1.0
    slot_env.last_selected_mask_by_uav_sat[1, [1, 2]] = 1.0
    slot_env.last_selected_mask_by_uav_sat[2, [2, 3]] = 1.0
    slot_env.last_sat_processed = np.linspace(0.1, 0.7, int(cfg.num_sat), dtype=np.float32)
    slot_env.last_policy_accel = np.array([[0.1, -0.2], [0.3, 0.0], [-0.1, 0.2]], dtype=np.float32)
    slot_env.last_exec_accel = np.array([[0.0, -0.1], [0.25, 0.05], [-0.05, 0.15]], dtype=np.float32)
    slot_env._invalidate_step_caches()


def _expected_accel_state_from_slot(slot_env, *, device: str) -> LocalAccelState:
    driver = StructuredControlDriver(slot_env)
    driver.begin_step()
    assert driver._accel_stage_spec_cache is not None
    return build_batched_local_accel_states_from_spec(driver._accel_stage_spec_cache, device=device)


def _assert_accel_obs_close(actual: StructuredGpuAccelObsView, expected: LocalAccelState) -> None:
    for field_name in ("ego_features", "ego_cell", "gu_tokens", "peer_tokens", "sat_tokens"):
        torch.testing.assert_close(
            getattr(actual, field_name),
            getattr(expected, field_name),
            rtol=8.0e-4,
            atol=8.0e-5,
        )
    for field_name in ("gu_mask", "peer_mask", "sat_mask"):
        torch.testing.assert_close(getattr(actual, field_name), getattr(expected, field_name))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native accel obs parity")
def test_native_accel_obs_matches_python_stage_spec_builder() -> None:
    cfg = _native_cfg()
    env_group = make_structured_env_group(cfg, num_envs=1, backend="sync", mode="eval")
    try:
        env_group.reset_many([int(cfg.seed)])
        slot_env = env_group.batch_core.envs[0]
        _install_nontrivial_slot_state(slot_env)
        env_group.set_tensor_device(torch.device("cuda"))
        env_group.begin_native_main_kernel_rollout(capacity=2, num_envs=1)
        runtime = env_group.native_rollout_runtime
        abi = runtime.main.native_cuda_abi
        native_cuda.prepare_initial_accel_live(
            abi,
            slot=0,
            active_idx=0,
            accel_source_mode=native_cuda.SOURCE_POLICY,
            sat_source_mode=native_cuda.SOURCE_POLICY,
            bw_source_mode=native_cuda.SOURCE_POLICY,
        )

        expected = _expected_accel_state_from_slot(slot_env, device="cuda")
        _assert_accel_obs_close(runtime.main.accel_live_obs_buffers[0], expected)
    finally:
        close_structured_env_group(env_group)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native accel actor parity")
def test_native_accel_actor_deterministic_matches_torch_policy() -> None:
    cfg = _native_cfg()
    env_group = make_structured_env_group(cfg, num_envs=1, backend="sync", mode="eval")
    try:
        env_group.reset_many([int(cfg.seed) + 1])
        slot_env = env_group.batch_core.envs[0]
        _install_nontrivial_slot_state(slot_env)
        env_group.set_tensor_device(torch.device("cuda"))
        env_group.begin_native_main_kernel_rollout(capacity=2, num_envs=1)
        runtime = env_group.native_rollout_runtime
        abi = runtime.main.native_cuda_abi
        native_cuda.prepare_initial_accel_live(
            abi,
            slot=0,
            active_idx=0,
            accel_source_mode=native_cuda.SOURCE_POLICY,
            sat_source_mode=native_cuda.SOURCE_POLICY,
            bw_source_mode=native_cuda.SOURCE_POLICY,
        )

        torch.manual_seed(1234)
        actor = build_structured_modules_from_config(cfg, hidden_dim=32, embed_dim=16, build_critic=False).actor.to("cuda")
        actor.eval()
        binding = build_native_actor_cuda_binding(actor, device="cuda")
        with torch.no_grad():
            expected = actor.accel_policy(runtime.main.accel_live_obs_buffers[0], deterministic=True)
        native_cuda.actor_accel_live(abi, binding.abi, active_idx=0, deterministic=True, rng_step=0)

        torch.testing.assert_close(
            runtime.main.live_accel_action.reshape_as(expected.action),
            expected.action,
            rtol=2.0e-5,
            atol=2.0e-5,
        )
        torch.testing.assert_close(
            runtime.main.live_accel_old_logprob.reshape_as(expected.logprob),
            expected.logprob,
            rtol=2.0e-5,
            atol=2.0e-5,
        )

        native_cuda.actor_accel_live_fused(abi, binding.abi, active_idx=0, deterministic=True, rng_step=0)
        torch.testing.assert_close(
            runtime.main.live_accel_action.reshape_as(expected.action),
            expected.action,
            rtol=2.0e-5,
            atol=2.0e-5,
        )
        torch.testing.assert_close(
            runtime.main.live_accel_old_logprob.reshape_as(expected.logprob),
            expected.logprob,
            rtol=2.0e-5,
            atol=2.0e-5,
        )
    finally:
        close_structured_env_group(env_group)
