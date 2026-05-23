from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pytest
import torch

from sagin_marl.env.config import SaginConfig
import sagin_marl.env.structured_batch_env_core as structured_batch_env_core
import sagin_marl.env.structured_gpu_rollout_runtime as structured_runtime_module
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
import sagin_marl.rl.structured_actor as structured_actor_module
import sagin_marl.rl.structured_buffer as structured_buffer_module
import sagin_marl.rl.structured_mappo as structured_mappo_module
from sagin_marl.rl.structured_mappo import StructuredMAPPO


def _single_bw_return_view(*, reward: float, value: float, terminated: bool, truncated: bool):
    one_float = np.zeros((1,), dtype=np.float32)
    return structured_buffer_module.StructuredReturnBatchView(
        transition_count=1,
        stage_ids=np.asarray([2], dtype=np.int64),
        env_indices=np.asarray([0], dtype=np.int64),
        values=np.asarray([float(value)], dtype=np.float32),
        rewards=np.asarray([float(reward)], dtype=np.float32),
        terminated=np.asarray([bool(terminated)], dtype=bool),
        truncated=np.asarray([bool(truncated)], dtype=bool),
        bw_access_rewards=one_float.copy(),
        bw_weighted_workload_delta_rewards=one_float.copy(),
        bw_weighted_workload_level_rewards=one_float.copy(),
        bw_gu_queue_level_rewards=one_float.copy(),
        bw_system_queue_level_rewards=one_float.copy(),
        bw_gu_service_queue_rewards=one_float.copy(),
        stage_batches={},
    )


def test_structured_mappo_rejects_training_nonpolicy_exec_source():
    actor = torch.nn.Linear(1, 1)
    critic = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match="train_accel=True requires exec_accel_source=policy"):
        StructuredMAPPO(
            actor=actor,
            critic=critic,
            exec_accel_source="queue_aware",
            train_accel=True,
        )


def test_structured_gae_rollout_tail_non_done_uses_bootstrap_value():
    buffer = structured_buffer_module.StructuredRolloutBuffer()
    result = buffer.compute_gae(
        gamma_env=0.9,
        gae_lambda=1.0,
        return_view=_single_bw_return_view(reward=2.0, value=0.0, terminated=False, truncated=False),
        bootstrap_values={0: 5.0},
    )
    assert result["returns"][0] == pytest.approx(2.0 + 0.9 * 5.0)


def test_structured_gae_timeout_is_finite_horizon_terminal_by_default():
    buffer = structured_buffer_module.StructuredRolloutBuffer()
    result = buffer.compute_gae(
        gamma_env=0.9,
        gae_lambda=1.0,
        return_view=_single_bw_return_view(reward=2.0, value=0.0, terminated=False, truncated=True),
        bootstrap_values={0: 5.0},
        truncated_bootstrap_values={0: 7.0},
    )
    assert result["returns"][0] == pytest.approx(2.0)


def test_structured_gae_timeout_bootstrap_is_opt_in():
    buffer = structured_buffer_module.StructuredRolloutBuffer()
    result = buffer.compute_gae(
        gamma_env=0.9,
        gae_lambda=1.0,
        return_view=_single_bw_return_view(reward=2.0, value=0.0, terminated=False, truncated=True),
        bootstrap_values={0: 5.0},
        truncated_bootstrap_values={0: 7.0},
        bootstrap_truncated=True,
    )
    assert result["returns"][0] == pytest.approx(2.0 + 0.9 * 7.0)


def test_structured_gae_terminated_episode_tail_does_not_bootstrap():
    buffer = structured_buffer_module.StructuredRolloutBuffer()
    result = buffer.compute_gae(
        gamma_env=0.9,
        gae_lambda=1.0,
        return_view=_single_bw_return_view(reward=2.0, value=0.0, terminated=True, truncated=False),
        bootstrap_values={0: 5.0},
        truncated_bootstrap_values={0: 7.0},
    )
    assert result["returns"][0] == pytest.approx(2.0)


class _DummyStructuredActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((1,), dtype=torch.float32))

    def act_accel(self, local_state=None, deterministic: bool = False):
        del local_state, deterministic
        return self.weight

    def act_sat(self, local_state=None, deterministic: bool = False):
        del local_state, deterministic
        return self.weight

    def act_bw(self, local_state=None, deterministic: bool = False):
        del local_state, deterministic
        return self.weight

    def act_accel_into(
        self,
        local_state=None,
        *,
        action_out,
        logprob_out=None,
        history_action_out=None,
        history_logprob_out=None,
        history_slot_t=None,
        history_env_row_ids_t=None,
        deterministic: bool = False,
        num_envs: int,
        num_agents: int,
    ):
        del local_state, deterministic, num_envs, num_agents, history_slot_t, history_env_row_ids_t
        action_out.zero_()
        if logprob_out is not None:
            logprob_out.zero_()
        if history_action_out is not None:
            history_action_out.zero_()
        if history_logprob_out is not None:
            history_logprob_out.zero_()

    def act_sat_into(
        self,
        local_state=None,
        *,
        subset_index_out,
        logprob_out=None,
        history_action_out=None,
        history_logprob_out=None,
        history_slot_t=None,
        history_env_row_ids_t=None,
        deterministic: bool = False,
        num_envs: int,
        num_agents: int,
    ):
        del local_state, deterministic, num_envs, num_agents, history_slot_t, history_env_row_ids_t
        subset_index_out.zero_()
        if logprob_out is not None:
            logprob_out.zero_()
        if history_action_out is not None:
            history_action_out.zero_()
        if history_logprob_out is not None:
            history_logprob_out.zero_()

    def act_bw_into(
        self,
        local_state=None,
        *,
        action_out,
        ref_action_out,
        logprob_out=None,
        logprob_per_agent_out=None,
        entropy_per_agent_out=None,
        logprob_raw_per_agent_out=None,
        entropy_raw_per_agent_out=None,
        tau_out=None,
        kappa_out=None,
        valid_count_out=None,
        latent_count_out=None,
        history_action_out=None,
        history_ref_action_out=None,
        history_logprob_out=None,
        history_logprob_per_agent_out=None,
        history_slot_t=None,
        history_env_row_ids_t=None,
        deterministic: bool = False,
        num_envs: int,
        num_agents: int,
    ):
        del local_state, deterministic, num_envs, num_agents, history_slot_t, history_env_row_ids_t
        action_out.zero_()
        ref_action_out.zero_()
        if logprob_out is not None:
            logprob_out.zero_()
        if logprob_per_agent_out is not None:
            logprob_per_agent_out.zero_()
        if entropy_per_agent_out is not None:
            entropy_per_agent_out.zero_()
        if logprob_raw_per_agent_out is not None:
            logprob_raw_per_agent_out.zero_()
        if entropy_raw_per_agent_out is not None:
            entropy_raw_per_agent_out.zero_()
        if tau_out is not None:
            tau_out.zero_()
        if kappa_out is not None:
            kappa_out.zero_()
        if valid_count_out is not None:
            valid_count_out.zero_()
        if latent_count_out is not None:
            latent_count_out.zero_()
        if history_action_out is not None:
            history_action_out.zero_()
        if history_ref_action_out is not None:
            history_ref_action_out.zero_()
        if history_logprob_out is not None:
            history_logprob_out.zero_()
        if history_logprob_per_agent_out is not None:
            history_logprob_per_agent_out.zero_()


class _DummyStructuredCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros((1,), dtype=torch.float32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native actor strict contract validation")
def test_native_cuda_rollout_actor_contract_uses_native_binding_not_torch_compile(monkeypatch):
    cfg = SaginConfig()
    cfg.structured_env_backend = "native"
    cfg.structured_env_tensor_backend = "cuda"
    cfg.structured_rollout_actor_compile_enabled = False
    cfg.structured_native_main_kernel_actor_compile_enabled = False
    cfg.structured_rollout_actor_compile_mode = "default"
    cfg.structured_kernel_compile_backend = "auto"
    cfg.structured_kernel_compile_cudagraphs = False
    cfg.structured_kernel_cudagraph_mark_step_begin = False

    compile_calls: list[tuple[str, dict[str, object]]] = []

    def _fake_compile(fn, **kwargs):
        compile_calls.append((getattr(fn, "__name__", "<unknown>"), dict(kwargs)))
        return fn

    monkeypatch.setattr(torch, "compile", _fake_compile)

    actor = build_structured_modules_from_config(cfg, hidden_dim=8, embed_dim=4, build_critic=False).actor.to("cuda")
    critic = _DummyStructuredCritic()
    learner = StructuredMAPPO(
        actor=actor,
        critic=critic,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=1.0e-3),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=1.0e-3),
        device=torch.device("cuda"),
        cfg=cfg,
        train_accel=False,
        train_sat=False,
        train_bw=True,
    )

    assert learner.device.type == "cuda"
    assert learner._native_actor_policy_binding is not None
    assert cfg.structured_rollout_actor_compile_enabled is False
    assert cfg.structured_native_main_kernel_actor_compile_enabled is False
    assert cfg.structured_rollout_actor_compile_mode == "default"
    assert compile_calls == []


def _called_function_names(fn) -> set[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                calls.add(str(target.id))
            elif isinstance(target, ast.Attribute):
                calls.add(str(target.attr))
    return calls


def _attribute_chains(fn) -> set[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    attrs: set[str] = set()

    def _chain(node: ast.AST) -> str | None:
        parts: list[str] = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(str(current.attr))
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(str(current.id))
            return ".".join(reversed(parts))
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            chain = _chain(node)
            if chain is not None:
                attrs.add(chain)
    return attrs


def test_actor_live_into_methods_do_not_use_group_copy_helpers():
    helper_names = {"_copy_group_agent_tensor_into_", "_sum_group_agent_tensor_into_"}
    for fn in (
        structured_actor_module.StructuredActor.act_accel_into,
        structured_actor_module.StructuredActor.act_sat_into,
        structured_actor_module.StructuredActor.act_bw_into,
    ):
        assert helper_names.isdisjoint(_called_function_names(fn))


def test_native_rollout_buffer_no_longer_uses_native_history_flatten_builder():
    assert not hasattr(structured_buffer_module, "_build_rollout_views_from_native_history")
    for fn in (
        structured_buffer_module.StructuredRolloutBuffer.build_rollout_views,
        structured_buffer_module.StructuredRolloutBuffer.build_bootstrap_view,
        structured_buffer_module.StructuredRolloutBuffer.build_return_view,
        structured_buffer_module.StructuredRolloutBuffer.build_training_view,
    ):
        calls = _called_function_names(fn)
        assert "_build_rollout_views_from_native_history" not in calls


def test_bw_direct_live_path_no_longer_calls_legacy_history_materialization():
    for name in (
        "_execute_bw_native_fused_step_next_accel_obs_out_tensor_impl",
        "_execute_bw_native_fused_step_out_tensor_impl",
        "_write_native_main_kernel_history_slot_tensor_impl",
        "_write_native_main_kernel_training_slot_tensor_impl",
        "_finalize_native_main_kernel_training_step_result_view",
    ):
        assert not hasattr(structured_batch_env_core, name)
        assert not hasattr(structured_batch_env_core.StructuredBatchEnvCore, name)


def test_native_training_ring_is_stage_scoped_not_full_history_bag():
    history = structured_runtime_module.StructuredGpuRolloutTrainingRingBuffers()
    forbidden_top_level = {
        "accel_world_batch",
        "sat_world_batch",
        "bw_world_batch",
        "next_world_batch",
        "accel_local_batch",
        "sat_local_batch",
        "bw_local_batch",
        "local_obs_params",
    }
    assert all(not hasattr(history, name) for name in forbidden_top_level)
    assert hasattr(history, "accel_stage")
    assert hasattr(history, "sat_stage")
    assert hasattr(history, "bw_stage")
    assert not hasattr(history, "bw_next_world_batch")
    assert hasattr(history, "terminal_next_world")
    assert hasattr(history, "terminal_next_world_mask")


def test_official_live_path_no_longer_exposes_generic_training_slot_writer_helpers():
    assert not hasattr(structured_batch_env_core, "_write_native_main_kernel_training_slot_tensor_impl")
    assert not hasattr(structured_batch_env_core, "_copy_history_dataclass_slot_out_")
    assert not hasattr(structured_batch_env_core, "_copy_history_tensor_slot_out_")


def test_native_rollout_views_no_longer_rebuild_worlds_from_stage_fields():
    calls = _called_function_names(structured_buffer_module._build_rollout_views_from_native_training_ring)
    assert "_build_world_from_stage_fields_direct_tensor_impl" not in calls


def test_official_live_path_no_longer_calls_world_refresh_chain():
    forbidden = {
        "_refresh_native_training_world_from_stage_fields",
        "_write_training_world_from_stage_fields_direct_tensor_impl",
        "_build_world_from_stage_fields_direct_tensor_impl",
    }
    for removed_name in (
        "_prepare_native_stage_and_accel_obs_tensor_impl",
        "_prepare_native_stage_and_sat_obs_tensor_impl",
        "_apply_sat_selection_to_bw_stage_and_obs_tensor_impl",
        "_apply_sat_subset_to_bw_stage_and_obs_tensor_impl",
        "_execute_bw_native_fused_step_next_accel_obs_out_tensor_impl",
        "_runtime_tensor_prepare_stage_and_obs",
        "_runtime_tensor_apply_accel_publish_sat_obs_impl",
        "_runtime_tensor_apply_sat_publish_bw_obs_impl",
    ):
        assert not hasattr(structured_batch_env_core, removed_name)
        assert not hasattr(structured_batch_env_core.StructuredBatchEnvCore, removed_name)
    live_fns = (
        structured_batch_env_core.StructuredBatchEnvCore._runtime_step_begin_accel_obs,
        structured_batch_env_core.StructuredBatchEnvCore._runtime_step_publish_sat_obs,
        structured_batch_env_core.StructuredBatchEnvCore._runtime_step_publish_bw_obs,
        structured_batch_env_core.StructuredBatchEnvCore._runtime_step_finish_bw,
    )
    for fn in live_fns:
        assert forbidden.isdisjoint(_called_function_names(fn))


def test_training_ring_world_batches_are_not_stage_fields():
    src = textwrap.dedent(
        inspect.getsource(structured_runtime_module.StructuredGpuRolloutRuntime.preallocate_native_main_kernel_training_ring_buffers)
    )
    assert "_NativeStageTensorFields" not in src
    assert "stage_fields" not in src


def test_training_ring_stage_buffers_no_longer_store_generic_local_batch_bag():
    for cls in (
        structured_runtime_module.StructuredGpuRolloutTrainingStageBuffers,
        structured_runtime_module.StructuredGpuRolloutAccelTrainingStageBuffers,
        structured_runtime_module.StructuredGpuRolloutSatTrainingStageBuffers,
        structured_runtime_module.StructuredGpuRolloutBwTrainingStageBuffers,
    ):
        annotations = getattr(cls, "__annotations__", {})
        assert "local_batch" not in annotations
        assert not hasattr(cls, "local_batch")


def test_native_rollout_views_no_longer_read_ring_local_batch_field():
    src = textwrap.dedent(
        inspect.getsource(structured_buffer_module._build_rollout_views_from_native_training_ring)
    )
    assert ".local_batch" not in src
    assert "history.accel_stage.local_batch" not in src
    assert "history.sat_stage.local_batch" not in src
    assert "history.bw_stage.local_batch" not in src


def test_actor_live_into_methods_do_not_route_through_policy_output_objects():
    for fn in (
        structured_actor_module.AccelPolicy.act_into,
        structured_actor_module.SatSubsetPolicy.act_into,
        structured_actor_module.BwPolicy.act_into,
    ):
        src = textwrap.dedent(inspect.getsource(fn))
        assert "self(" not in src
        assert "PolicyOutput" not in src
    assert ".forward(" not in textwrap.dedent(inspect.getsource(structured_actor_module.AccelPolicy.act_into))


def test_official_live_actor_bridge_no_longer_uses_legacy_training_input_buffers():
    for fn in (
        structured_mappo_module._StructuredMAPPOGpuActorBridge.write_accel_action,
        structured_mappo_module._StructuredMAPPOGpuActorBridge.write_sat_action,
        structured_mappo_module._StructuredMAPPOGpuActorBridge.write_bw_action,
    ):
        src = textwrap.dedent(inspect.getsource(fn))
        assert "training_accel_values_input" not in src
        assert "training_sat_values_input" not in src
        assert "training_bw_values_input" not in src
        assert "training_bw_old_logprobs_per_agent_input" not in src
        assert "training_bw_old_logprobs_per_slot_input" not in src
        assert "training_bw_support_masks_input" not in src


def test_official_live_bw_next_direct_path_writes_next_world_history_ring():
    assert not hasattr(structured_batch_env_core.StructuredBatchEnvCore, "_execute_bw_stage_native_main_direct_out")
    src = textwrap.dedent(
        inspect.getsource(structured_batch_env_core.StructuredBatchEnvCore._bind_native_main_kernel_history_outputs)
    )
    assert "next_accel_history_world_out" in src
    assert "terminal_next_history_world_out" in src
    finish_src = textwrap.dedent(
        inspect.getsource(structured_batch_env_core.StructuredBatchEnvCore._runtime_step_finish_bw)
    )
    assert "native_cuda.finish_commit_prepare_live" in finish_src
