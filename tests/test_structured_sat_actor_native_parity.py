from __future__ import annotations

import pytest
import torch

from sagin_marl.env import native_cuda
from sagin_marl.env.config import SaginConfig
from sagin_marl.rl.native_actor_cuda import build_native_actor_cuda_binding
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_env_group


def _sat_acceptance_cfg() -> SaginConfig:
    return SaginConfig(
        seed=17,
        num_uav=2,
        num_gu=4,
        num_sat=6,
        users_obs_max=4,
        sats_obs_max=4,
        nbrs_obs_max=1,
        sat_num_select=2,
        T_steps=8,
        fixed_satellite_strategy=False,
        structured_env_backend="native",
        structured_env_tensor_backend="cuda",
        accel_gu_query_count=4,
        accel_peer_query_count=2,
        accel_sat_query_count=2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native SAT actor parity")
def test_native_sat_actor_matches_python_and_writes_live_metadata():
    cfg = _sat_acceptance_cfg()
    cfg.sat_candidate_mode = "elevation"
    cfg.structured_kernel_operator_mode = "auto"
    cfg.structured_kernel_compile_backend = "auto"
    cfg.structured_kernel_compile_cudagraphs = False
    cfg.structured_native_main_kernel_require_compiled_segments = True
    env_group = make_structured_env_group(cfg, num_envs=1, backend="sync", mode="eval")
    try:
        env_group.reset_many([int(cfg.seed) * 61])
        env_group.set_tensor_device(torch.device("cuda"))
        env_group.begin_native_main_kernel_rollout(capacity=2, num_envs=1)
        runtime = env_group.native_rollout_runtime
        abi = runtime.main.native_cuda_abi

        actor = build_structured_modules_from_config(cfg, hidden_dim=32, embed_dim=16, build_critic=False).actor.to("cuda")
        actor.eval()
        actor.native_cuda_rng_seed = 0x4567000000000000
        binding = build_native_actor_cuda_binding(actor, device="cuda")

        native_cuda.prepare_initial_accel_live(
            abi,
            slot=0,
            active_idx=0,
            accel_source_mode=native_cuda.SOURCE_POLICY,
            sat_source_mode=native_cuda.SOURCE_POLICY,
            bw_source_mode=native_cuda.SOURCE_POLICY,
        )
        native_cuda.actor_accel_live(abi, binding.abi, active_idx=0, deterministic=True, rng_step=3)
        native_cuda.accel_to_sat_live(
            abi,
            slot=0,
            active_idx=0,
            accel_source_mode=native_cuda.SOURCE_POLICY,
            sat_source_mode=native_cuda.SOURCE_POLICY,
            bw_source_mode=native_cuda.SOURCE_POLICY,
        )

        visible_ids = runtime.main.sat_stage_fields.visible_ids.reshape_as(runtime.main.live_sat_obs.candidate_sat_ids)
        torch.testing.assert_close(runtime.main.live_sat_obs.candidate_sat_ids, visible_ids)

        with torch.no_grad():
            expected = actor.sat_subset_policy(runtime.main.live_sat_obs, deterministic=True)
            expected_subset_mask = actor.sat_subset_policy._legal_subset_mask(runtime.main.live_sat_obs, expected.logits)
        assert bool(expected_subset_mask.any(dim=1).all().item())
        torch.testing.assert_close(runtime.main.live_sat_obs.subset_members[0], runtime.main.sat_subset_members_base)

        native_cuda.actor_sat_live(abi, binding.abi, deterministic=True, rng_step=4)
        torch.testing.assert_close(runtime.main.live_sat_subset_index.reshape_as(expected.subset_index), expected.subset_index)
        torch.testing.assert_close(runtime.main.live_sat_action_indices.reshape_as(expected.selected_sat_indices), expected.selected_sat_indices)
        torch.testing.assert_close(
            runtime.main.live_sat_old_logprobs_per_agent.reshape_as(expected.logprob),
            expected.logprob,
            atol=2.0e-5,
            rtol=2.0e-5,
        )
        torch.testing.assert_close(
            runtime.main.live_sat_entropy_per_agent.reshape_as(expected.entropy),
            expected.entropy,
            atol=2.0e-5,
            rtol=2.0e-5,
        )

        native_cuda.actor_sat_live_fused(abi, binding.abi, deterministic=True, rng_step=4)
        torch.testing.assert_close(runtime.main.live_sat_subset_index.reshape_as(expected.subset_index), expected.subset_index)
        torch.testing.assert_close(runtime.main.live_sat_action_indices.reshape_as(expected.selected_sat_indices), expected.selected_sat_indices)
        torch.testing.assert_close(
            runtime.main.live_sat_old_logprobs_per_agent.reshape_as(expected.logprob),
            expected.logprob,
            atol=2.0e-5,
            rtol=2.0e-5,
        )
        torch.testing.assert_close(
            runtime.main.live_sat_entropy_per_agent.reshape_as(expected.entropy),
            expected.entropy,
            atol=2.0e-5,
            rtol=2.0e-5,
        )
    finally:
        close_structured_env_group(env_group)

