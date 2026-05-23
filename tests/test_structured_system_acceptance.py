from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sagin_marl.env.config import SaginConfig
from sagin_marl.env import native_cuda
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredBatchStepResult
from sagin_marl.env.structured_gpu_rollout_runtime import (
    StructuredGpuRolloutRuntime,
)
from sagin_marl.env.structured_batch_env_core import StructuredBatchEnvCore
from sagin_marl.rl import structured_accel_actor_schema as accel_schema
from sagin_marl.rl import structured_bw_actor_schema as bw_schema
from sagin_marl.rl import structured_critic_schema as critic_schema
from sagin_marl.rl import structured_sat_actor_schema as sat_schema
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredGpuRolloutProgram
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_env_group


def test_native_main_kernel_call_sites_do_not_pass_cfg_runtime_arg():
    import sagin_marl.env.structured_batch_env_core as core_module

    source = Path(core_module.__file__).read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    strict_kernel_variables = {
        "access_kernel",
        "kernel",
        "fused_accel_kernel",
        "fused_sat_kernel",
        "fused_step_kernel",
        "workload_costs_kernel",
    }
    strict_host_functions = {
        "_runtime_step_begin_accel_obs",
        "_runtime_step_publish_sat_obs",
        "_runtime_step_publish_bw_obs",
        "_runtime_step_finish_bw",
        "_runtime_begin_horizon",
    }
    strict_ranges = [
        (int(node.lineno), int(node.end_lineno))
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in strict_host_functions
    ]
    offenders: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if not any(start <= int(node.lineno) <= end for start, end in strict_ranges):
            continue
        if node.func.id not in strict_kernel_variables:
            continue
        if any(keyword.arg == "cfg" for keyword in node.keywords):
            offenders.append(int(node.lineno))
    assert offenders == []


def test_native_main_kernel_strict_hosts_do_not_dispatch_kernels_at_step_time():
    import sagin_marl.env.structured_batch_env_core as core_module

    source = Path(core_module.__file__).read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    strict_host_functions = {
        "_runtime_step_begin_accel_obs",
        "_runtime_step_publish_sat_obs",
        "_runtime_step_publish_bw_obs",
        "_runtime_step_finish_bw",
        "_runtime_begin_horizon",
    }
    offenders: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in strict_host_functions:
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "_resolve_kernel_callable"
            ):
                offenders.append((node.name, int(child.lineno)))
    assert offenders == []


def test_native_main_kernel_live_hosts_do_not_read_runtime_cfg():
    import sagin_marl.env.structured_batch_env_core as core_module

    source = Path(core_module.__file__).read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    hot_host_functions = {
        "_runtime_step_begin_accel_obs",
        "_runtime_step_publish_sat_obs",
        "_runtime_step_publish_bw_obs",
        "_runtime_step_finish_bw",
        "_runtime_begin_horizon",
    }
    offenders: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in hot_host_functions:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute):
                if (
                    isinstance(child.value, ast.Name)
                    and child.value.id == "self"
                    and child.attr == "_cfg"
                ):
                    offenders.append((node.name, int(child.lineno)))
            elif (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "getattr"
                and child.args
                and isinstance(child.args[0], ast.Attribute)
                and isinstance(child.args[0].value, ast.Name)
                and child.args[0].value.id == "self"
                and child.args[0].attr == "_cfg"
            ):
                offenders.append((node.name, int(child.lineno)))
    assert offenders == []


def test_native_main_kernel_actor_bridge_hot_methods_do_not_read_cfg():
    import sagin_marl.rl.structured_mappo as mappo_module

    source = Path(mappo_module.__file__).read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    target_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "_StructuredMAPPOGpuActorBridge":
            target_class = node
            break
    assert target_class is not None
    hot_methods = {"write_accel_action", "write_sat_action", "write_bw_action"}
    offenders: list[tuple[str, int]] = []
    for node in target_class.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in hot_methods:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute):
                if (
                    isinstance(child.value, ast.Attribute)
                    and isinstance(child.value.value, ast.Name)
                    and child.value.value.id == "self"
                    and child.value.attr == "learner"
                    and child.attr == "cfg"
                ):
                    offenders.append((node.name, int(child.lineno)))
            elif (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "getattr"
                and child.args
                and isinstance(child.args[0], ast.Attribute)
                and isinstance(child.args[0].value, ast.Attribute)
                and isinstance(child.args[0].value.value, ast.Name)
                and child.args[0].value.value.id == "self"
                and child.args[0].value.attr == "learner"
                and child.args[0].attr == "cfg"
            ):
                offenders.append((node.name, int(child.lineno)))
    assert offenders == []


def test_native_main_kernel_actor_bridge_source_modes_are_bound_before_hot_step():
    import sagin_marl.rl.structured_mappo as mappo_module

    source = Path(mappo_module.__file__).read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    target_class = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "_StructuredMAPPOGpuActorBridge"
    )
    unbound_methods = {"write_accel_action", "write_sat_action", "write_bw_action"}
    offenders: list[tuple[str, int]] = []
    for node in target_class.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in unbound_methods:
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Attribute)
                and child.attr.endswith("_actor_source_mode_code")
            ):
                offenders.append((node.name, int(child.lineno)))
            if isinstance(child, ast.If):
                offenders.append((node.name, int(child.lineno)))
    assert offenders == []


def test_official_native_rollout_code_does_not_call_debug_hot_replay_or_cudagraph_runtime():
    import sagin_marl.rl.structured_eval as eval_module
    import sagin_marl.rl.structured_mappo as mappo_module

    offenders: list[tuple[str, str, int]] = []
    for module in (mappo_module, eval_module):
        source = Path(module.__file__).read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "native_hot_replay_program":
                    offenders.append((module.__name__, "native_hot_replay_program", int(node.lineno)))
                if isinstance(func, ast.Attribute) and func.attr == "torch_compile":
                    offenders.append((module.__name__, "torch_compile", int(node.lineno)))
                if isinstance(func, ast.Attribute) and func.attr in {"compiled", "compile"}:
                    owner = func.value
                    if isinstance(owner, ast.Name) and owner.id == "torch":
                        offenders.append((module.__name__, "torch.compile", int(node.lineno)))
            if isinstance(node, ast.Name) and node.id == "StructuredKernelRuntime":
                offenders.append((module.__name__, "StructuredKernelRuntime", int(node.lineno)))
    assert offenders == []


def test_native_main_kernel_step_program_does_not_rebind_current_stage_fields():
    import sagin_marl.env.structured_gpu_rollout_runtime as runtime_module

    source = Path(runtime_module.__file__).read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    target_class = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "StructuredGpuNativeRuntimeStepProgram"
    )
    offenders: list[int] = []
    for node in target_class.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "_current_or_begin_accel_obs":
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Assign):
                continue
            for target in child.targets:
                if isinstance(target, ast.Attribute) and target.attr == "accel_stage_fields":
                    offenders.append(int(child.lineno))
    assert offenders == []


def test_native_main_kernel_finish_api_has_no_flow_proxy_string_argument():
    import sagin_marl.env.structured_batch_env_core as core_module
    import sagin_marl.env.structured_gpu_rollout_runtime as runtime_module

    for module, function_name in (
        (core_module, "_runtime_step_finish_bw"),
        (runtime_module, "_after_bw_action"),
    ):
        source = Path(module.__file__).read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        matches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        ]
        assert matches
        for node in matches:
            arg_names = {arg.arg for arg in [*node.args.args, *node.args.kwonlyargs]}
            assert "bw_proxy_base_action_mode" not in arg_names


def test_native_main_kernel_direct_hosts_are_not_compat_publishers():
    import sagin_marl.env.structured_batch_env_core as core_module

    source = Path(core_module.__file__).read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    strict_host_functions = {
        "_runtime_step_begin_accel_obs",
        "_runtime_step_publish_sat_obs",
        "_runtime_step_publish_bw_obs",
        "_runtime_step_finish_bw",
        "_runtime_begin_horizon",
    }
    forbidden_calls = {
        "_stage_value",
        "_runtime_stage_view_from_fields",
        "_as_grouped_tensor_batch",
        "write_accel_obs_tensors",
        "write_sat_obs_tensors",
        "write_bw_obs_tensors",
        "bind_accel_obs",
        "bind_sat_obs",
        "bind_bw_obs",
        "ensure_step_result_buffers",
        "_ensure_native_stage_fields_like",
        "_native_zero_float_tensor",
        "ensure_fixed_rollout_history_buffers",
        "record_fixed_rollout_history_slot",
        "_native_world_from_stage_fields",
        "_native_world_ref_from_stage_fields",
        "StructuredBatchStepResult",
        "_NativeBwStepTensorFields",
        "_NativeBwStepNextAccelTensorFields",
        "_NativeBwResultOutBuffers",
        "_NativeBwRuntimeStateOutBuffers",
        "as_tensor",
        "stack",
        "clone",
        "dict",
    }
    offenders: list[tuple[str, str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in strict_host_functions:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Dict):
                offenders.append((node.name, "dict_literal", int(child.lineno)))
            if isinstance(child, ast.Name) and child.id == "cfg":
                offenders.append((node.name, "cfg", int(child.lineno)))
            if isinstance(child, ast.Attribute) and child.attr == "cfg":
                offenders.append((node.name, "cfg", int(child.lineno)))
            if isinstance(child, ast.Call):
                func = child.func
                call_name = ""
                if isinstance(func, ast.Name):
                    call_name = func.id
                elif isinstance(func, ast.Attribute):
                    call_name = func.attr
                if call_name in forbidden_calls:
                    offenders.append((node.name, call_name, int(child.lineno)))
    assert offenders == []


def test_native_main_kernel_strict_segments_do_not_allocate_or_read_cfg():
    import sagin_marl.env.structured_batch_env_core as core_module

    source = Path(core_module.__file__).read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    strict_segments = {
        "_build_candidate_index_mask_tensor_impl",
        "_candidate_flag_bundle_from_index_mask_tensor_impl",
        "_prepare_native_stage_batch_tensor_impl",
        "_prepare_full_sat_geometry_tensor_impl",
        "_prepare_native_stage_and_accel_obs_tensor_impl",
        "_prepare_native_stage_and_sat_obs_tensor_impl",
        "_apply_accel_stage_prepare_sat_obs_tensor_impl",
        "_apply_sat_subset_to_bw_stage_and_obs_tensor_impl",
        "_apply_sat_selection_to_bw_stage_and_obs_tensor_impl",
        "_execute_bw_native_fused_step_tensor_impl",
        "_execute_bw_native_fused_step_next_accel_obs_out_tensor_impl",
        "_build_local_accel_obs_from_stage_tensor_impl",
        "_build_local_sat_obs_from_stage_tensor_impl",
        "_build_local_bw_obs_from_stage_tensor_impl",
    }
    forbidden_calls = {
        "zeros",
        "zeros_like",
        "empty",
        "empty_like",
        "full",
        "full_like",
        "arange",
        "eye",
        "stack",
        "as_tensor",
        "tensor",
        "ones",
        "ones_like",
        "clone",
        "_RuntimeStageTensorView",
        "StructuredBatchStepResult",
    }
    offenders: list[tuple[str, str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in strict_segments:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id == "cfg":
                offenders.append((node.name, "cfg", int(child.lineno)))
            if isinstance(child, ast.Attribute) and child.attr == "cfg":
                offenders.append((node.name, "cfg", int(child.lineno)))
            if isinstance(child, ast.Call):
                func = child.func
                call_name = ""
                if isinstance(func, ast.Name):
                    call_name = func.id
                elif isinstance(func, ast.Attribute):
                    call_name = func.attr
                if call_name in forbidden_calls:
                    offenders.append((node.name, call_name, int(child.lineno)))
    assert offenders == []


def test_native_main_kernel_scalar_static_segments_do_not_read_cfg():
    import sagin_marl.env.structured_batch_env_core as core_module

    source = Path(core_module.__file__).read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    checked_functions = {
        "_apply_batched_access_rate_static_tensor_impl",
        "_bw_weighted_workload_device_costs_static_tensor_impl",
    }
    offenders: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in checked_functions:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id == "cfg":
                offenders.append((node.name, int(child.lineno)))
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "getattr"
                and child.args
                and isinstance(child.args[0], ast.Name)
                and child.args[0].id in {"cfg", "params"}
            ):
                offenders.append((node.name, int(child.lineno)))
    assert offenders == []


def _acceptance_cfg() -> SaginConfig:
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
        structured_env_tensor_backend="cpu",
    )


class _FixedNativeActionBridge:
    def __init__(
        self,
        *,
        cfg: SaginConfig,
        device: torch.device | str,
        num_envs: int,
        accel_action,
        sat_action,
        bw_action,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.accel_action = torch.as_tensor(accel_action, dtype=torch.float32, device=self.device)
        self.sat_action = torch.as_tensor(sat_action, dtype=torch.long, device=self.device)
        self.bw_action = torch.as_tensor(bw_action, dtype=torch.float32, device=self.device)
        self._scalar_logprob = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._bw_old_logprobs_per_agent = torch.zeros(
            (self.num_envs, int(self.cfg.num_uav)),
            dtype=torch.float32,
            device=self.device,
        )

    def begin_step(self, *, deterministic: bool) -> None:
        del deterministic

    def write_accel_action(self, accel_obs, *, runtime, num_envs: int, deterministic: bool) -> None:
        del accel_obs, deterministic
        action = self.accel_action
        if action.ndim == 2:
            action = action.unsqueeze(0).expand(int(num_envs), -1, -1).contiguous()
        runtime.main.live_accel_action[: int(num_envs)].copy_(
            action.to(device=runtime.main.live_accel_action.device, dtype=torch.float32)
        )
        runtime.main.live_accel_old_logprob[: int(num_envs)].zero_()

    def env_phase_b_kwargs(self) -> dict:
        return {"indices": list(range(self.num_envs))}

    def write_sat_action(
        self,
        sat_obs,
        *,
        runtime,
        num_envs: int,
        sat_max_select: int,
        deterministic: bool,
    ) -> None:
        del sat_obs, sat_max_select, deterministic
        action = self.sat_action
        if action.ndim == 1:
            action = action.unsqueeze(0).expand(int(num_envs), -1).contiguous()
        runtime.main.live_sat_subset_index[: int(num_envs)].copy_(
            action.to(device=runtime.main.live_sat_subset_index.device, dtype=torch.long)
        )
        runtime.main.live_sat_old_logprobs_per_agent[: int(num_envs)].zero_()

    def env_phase_c_kwargs(self) -> dict:
        return {"indices": list(range(self.num_envs))}

    def write_bw_action(self, bw_obs, *, runtime, num_envs: int, deterministic: bool) -> None:
        del bw_obs, deterministic
        action = self.bw_action
        if action.ndim == 2:
            action = action.unsqueeze(0).expand(int(num_envs), -1, -1).contiguous()
        runtime.main.live_bw_action[: int(num_envs)].copy_(
            action.to(device=runtime.main.live_bw_action.device, dtype=torch.float32)
        )
        runtime.main.live_bw_ref_action[: int(num_envs)].copy_(
            action.to(device=runtime.main.live_bw_ref_action.device, dtype=torch.float32)
        )
        runtime.main.live_bw_old_logprob[: int(num_envs)].zero_()
        runtime.main.live_bw_old_logprobs_per_agent[: int(num_envs)].zero_()

    def env_phase_d_kwargs(self) -> dict:
        return {"indices": list(range(self.num_envs))}


def test_cpu_native_rollout_begin_rejects_final_cuda_contract():
    cfg = _acceptance_cfg()
    cfg.structured_env_tensor_backend = "cpu"
    env_group = make_structured_env_group(cfg, num_envs=1, backend="sync", mode="eval")
    try:
        env_group.reset_many([int(cfg.seed) * 7])
        env_group.set_tensor_device(torch.device("cpu"))
        with pytest.raises(RuntimeError, match="final native CUDA"):
            env_group.begin_native_main_kernel_rollout(capacity=2, num_envs=1)
    finally:
        close_structured_env_group(env_group)
def _zero_native_action_bridge(
    cfg: SaginConfig,
    *,
    device: torch.device | str,
    num_envs: int,
) -> _FixedNativeActionBridge:
    device_t = torch.device(device)
    return _FixedNativeActionBridge(
        cfg=cfg,
        device=device_t,
        num_envs=int(num_envs),
        accel_action=torch.zeros((int(num_envs), int(cfg.num_uav), 2), dtype=torch.float32, device=device_t),
        sat_action=torch.zeros((int(num_envs), int(cfg.num_uav)), dtype=torch.long, device=device_t),
        bw_action=torch.zeros(
            (int(num_envs), int(cfg.num_uav), int(cfg.num_gu)),
            dtype=torch.float32,
            device=device_t,
        ),
    )


def _assert_finite_and_bounded(name: str, value: torch.Tensor, *, max_abs: float = 1.0e5) -> None:
    assert torch.is_tensor(value), name
    assert bool(torch.isfinite(value).all().item()), name
    if int(value.numel()) > 0:
        assert float(value.detach().abs().max().cpu().item()) <= float(max_abs), name


def _assert_bool_tensor(name: str, value: torch.Tensor) -> None:
    assert torch.is_tensor(value), name
    assert value.dtype == torch.bool, name


def _assert_unit_interval(name: str, value: torch.Tensor) -> None:
    assert torch.is_tensor(value), name
    if int(value.numel()) == 0:
        return
    value_f = value.detach().float()
    assert bool(torch.isfinite(value_f).all().item()), name
    assert float(value_f.min().cpu().item()) >= -1.0e-5, name
    assert float(value_f.max().cpu().item()) <= 1.0 + 1.0e-5, name


def _poison_native_active_sat_derived_buffers(runtime, *, sentinel: float = 1.0e20) -> None:
    stage_objects = []
    stage_objects.extend(getattr(runtime.main, "accel_stage_field_buffers", None) or ())
    for attr in ("sat_stage_fields", "bw_stage_fields"):
        value = getattr(runtime.main, attr, None)
        if value is not None:
            stage_objects.append(value)
    float_field_names = (
        "sat_pos_active",
        "sat_vel_active",
        "sat_queue_active",
        "sat_load_active",
        "sat_cost_norm_active",
        "us_rel_pos_active",
        "us_rel_vel_active",
        "us_gain_active",
        "us_nu_eff_active",
        "visible_flag_active",
        "us_valid_flag_active",
    )
    for stage_fields in stage_objects:
        for field_name in float_field_names:
            value = getattr(stage_fields, field_name, None)
            if torch.is_tensor(value) and value.is_floating_point():
                value.fill_(float(sentinel))
        active_ids = getattr(stage_fields, "active_sat_ids", None)
        if torch.is_tensor(active_ids):
            active_ids.fill_(-777)


def _assert_accel_actor_inputs(stage) -> None:
    for name in ("ego_features", "ego_cell", "gu_tokens", "peer_tokens", "sat_tokens"):
        _assert_finite_and_bounded(f"accel.{name}", getattr(stage, name))
    for name in ("gu_mask", "peer_mask", "sat_mask"):
        _assert_bool_tensor(f"accel.{name}", getattr(stage, name))
    _assert_unit_interval("accel.ego_x", stage.ego_features[..., accel_schema.EGO_X])
    _assert_unit_interval("accel.ego_y", stage.ego_features[..., accel_schema.EGO_Y])
    _assert_unit_interval("accel.ego_energy", stage.ego_features[..., accel_schema.EGO_ENERGY])
    _assert_unit_interval("accel.ego_queue_fill", stage.ego_features[..., accel_schema.EGO_UAV_QUEUE_FILL])
    _assert_unit_interval("accel.cell_gu_count_frac", stage.ego_cell[..., accel_schema.CELL_GU_COUNT_FRAC])
    _assert_unit_interval("accel.gu_x", stage.gu_tokens[..., accel_schema.GU_X])
    _assert_unit_interval("accel.gu_y", stage.gu_tokens[..., accel_schema.GU_Y])
    _assert_unit_interval("accel.gu_queue_fill", stage.gu_tokens[..., accel_schema.GU_QUEUE_FILL])
    _assert_unit_interval("accel.gu_last_assoc_to_ego", stage.gu_tokens[..., accel_schema.GU_LAST_ASSOC_TO_EGO])
    _assert_unit_interval("accel.gu_last_bw_fraction_ego", stage.gu_tokens[..., accel_schema.GU_LAST_BW_FRACTION_EGO])
    _assert_unit_interval("accel.gu_pre_owner_is_ego", stage.gu_tokens[..., accel_schema.GU_PRE_OWNER_IS_EGO])
    _assert_unit_interval("accel.gu_last_bw_sum", stage.gu_tokens[..., accel_schema.GU_LAST_BW_SUM])
    _assert_unit_interval("accel.peer_unsafe_flag", stage.peer_tokens[..., accel_schema.PEER_UNSAFE_FLAG])
    _assert_unit_interval("accel.peer_alert_flag", stage.peer_tokens[..., accel_schema.PEER_ALERT_FLAG])
    _assert_unit_interval("accel.peer_last_shared_sat_frac", stage.peer_tokens[..., accel_schema.PEER_LAST_SHARED_SAT_FRAC])
    _assert_unit_interval("accel.sat_queue_fill", stage.sat_tokens[..., accel_schema.SAT_QUEUE_FILL])
    _assert_unit_interval("accel.sat_last_selected_load_frac", stage.sat_tokens[..., accel_schema.SAT_LAST_SELECTED_LOAD_FRAC])
    _assert_unit_interval("accel.sat_visible_flag", stage.sat_tokens[..., accel_schema.SAT_VISIBLE_FLAG])
    _assert_unit_interval("accel.sat_valid_flag", stage.sat_tokens[..., accel_schema.SAT_VALID_FLAG])
    _assert_unit_interval("accel.sat_last_selected_flag", stage.sat_tokens[..., accel_schema.SAT_LAST_SELECTED_FLAG])


def _assert_sat_actor_inputs(stage, *, sats_obs_max: int) -> None:
    for name in ("ego_features", "demand_features", "role_features", "sat_tokens"):
        _assert_finite_and_bounded(f"sat.{name}", getattr(stage, name))
    for name in ("sat_mask", "sat_valid_mask", "subset_mask"):
        _assert_bool_tensor(f"sat.{name}", getattr(stage, name))
    assert bool(((stage.candidate_sat_ids >= -1) & (stage.candidate_sat_ids < int(sats_obs_max))).all().item())
    assert bool(((stage.subset_members >= -1) & (stage.subset_members < int(sats_obs_max))).all().item())
    _assert_unit_interval("sat.ego_queue_fill", stage.ego_features[..., sat_schema.EGO_UAV_QUEUE_FILL])
    _assert_unit_interval("sat.ego_last_selected_count_frac", stage.ego_features[..., sat_schema.EGO_LAST_SELECTED_COUNT_FRAC])
    _assert_unit_interval("sat.demand_gu_count_frac", stage.demand_features[..., sat_schema.DEMAND_CELL_GU_COUNT_FRAC])
    _assert_unit_interval("sat.token_queue_fill", stage.sat_tokens[..., sat_schema.SAT_QUEUE_FILL])
    _assert_unit_interval("sat.token_last_selected_load_frac", stage.sat_tokens[..., sat_schema.SAT_LAST_SELECTED_LOAD_FRAC])
    _assert_unit_interval("sat.token_visible_flag", stage.sat_tokens[..., sat_schema.US_VISIBLE_FLAG])
    _assert_unit_interval("sat.token_valid_flag", stage.sat_tokens[..., sat_schema.US_VALID_FLAG])
    _assert_unit_interval("sat.token_last_selected_flag", stage.sat_tokens[..., sat_schema.US_LAST_SELECTED_FLAG])


def _assert_bw_actor_inputs(stage) -> None:
    for name in ("ego_features", "selected_sat_tokens", "gu_tokens"):
        _assert_finite_and_bounded(f"bw.{name}", getattr(stage, name))
    for name in ("selected_sat_mask", "gu_mask", "bw_valid_mask"):
        _assert_bool_tensor(f"bw.{name}", getattr(stage, name))
    _assert_unit_interval("bw.ego_queue_fill", stage.ego_features[..., bw_schema.BW_EGO_FIELDS.index("uav_queue_fill")])
    _assert_unit_interval("bw.sat_queue_fill", stage.selected_sat_tokens[..., bw_schema.BW_SAT_TOKEN_FIELDS.index("sat_queue_fill")])
    _assert_unit_interval("bw.gu_queue_fill", stage.gu_tokens[..., bw_schema.BW_GU_TOKEN_FIELDS.index("gu_queue_fill")])


def _assert_critic_world_inputs(stage) -> None:
    world = stage.world_batch
    for name in ("uav_nodes", "gu_nodes", "sat_nodes", "uav_gu_edges", "uav_sat_edges", "uav_uav_edges", "global_scalars"):
        _assert_finite_and_bounded(f"critic.{name}", getattr(world, name))
    for name in ("gu_mask", "sat_mask", "uav_gu_mask", "uav_sat_mask", "uav_uav_mask"):
        _assert_bool_tensor(f"critic.{name}", getattr(world, name))
    _assert_unit_interval("critic.uav_x", world.uav_nodes[..., critic_schema.UAV_X])
    _assert_unit_interval("critic.uav_y", world.uav_nodes[..., critic_schema.UAV_Y])
    _assert_unit_interval("critic.uav_energy", world.uav_nodes[..., critic_schema.UAV_ENERGY])
    _assert_unit_interval("critic.gu_queue_fill", world.gu_nodes[..., critic_schema.GU_QUEUE_FILL])
    _assert_unit_interval("critic.gu_x", world.gu_nodes[..., critic_schema.GU_X])
    _assert_unit_interval("critic.gu_y", world.gu_nodes[..., critic_schema.GU_Y])
    _assert_unit_interval("critic.uav_queue_fill", world.uav_nodes[..., critic_schema.UAV_QUEUE_FILL])
    _assert_unit_interval("critic.uav_prefix_cost_known", world.uav_nodes[..., critic_schema.UAV_PREFIX_COST_KNOWN])
    _assert_unit_interval("critic.uav_prefix_bw_valid_count_frac", world.uav_nodes[..., critic_schema.UAV_PREFIX_BW_VALID_COUNT_FRAC])
    _assert_unit_interval("critic.sat_queue_fill", world.sat_nodes[..., critic_schema.SAT_QUEUE_FILL])
    _assert_unit_interval("critic.sat_prefix_load_known", world.sat_nodes[..., critic_schema.SAT_PREFIX_LOAD_KNOWN])
    _assert_unit_interval("critic.sat_prefix_selected_load_frac", world.sat_nodes[..., critic_schema.SAT_PREFIX_SELECTED_LOAD_FRAC])
    _assert_unit_interval("critic.sat_last_selected_load_frac", world.sat_nodes[..., critic_schema.SAT_LAST_SELECTED_LOAD_FRAC])
    _assert_unit_interval("critic.ug_last_bw_fraction", world.uav_gu_edges[..., critic_schema.UG_LAST_BW_FRACTION])
    _assert_unit_interval("critic.ug_prefix_bw_valid", world.uav_gu_edges[..., critic_schema.UG_PREFIX_BW_VALID_FLAG])
    _assert_unit_interval("critic.ug_prefix_bw_known", world.uav_gu_edges[..., critic_schema.UG_PREFIX_BW_VALID_KNOWN])
    _assert_unit_interval("critic.us_visible", world.uav_sat_edges[..., critic_schema.US_VISIBLE_FLAG])
    _assert_unit_interval("critic.us_valid", world.uav_sat_edges[..., critic_schema.US_VALID_FLAG])
    _assert_unit_interval("critic.us_last_selected", world.uav_sat_edges[..., critic_schema.US_LAST_SELECTED_FLAG])
    _assert_unit_interval("critic.us_prefix_selected", world.uav_sat_edges[..., critic_schema.US_PREFIX_SELECTED_FLAG])
    _assert_unit_interval("critic.us_prefix_known", world.uav_sat_edges[..., critic_schema.US_PREFIX_SELECTED_KNOWN])
    _assert_unit_interval("critic.uu_alert", world.uav_uav_edges[..., critic_schema.UU_ALERT_FLAG])
    _assert_unit_interval("critic.uu_unsafe", world.uav_uav_edges[..., critic_schema.UU_UNSAFE_FLAG])
    _assert_unit_interval("critic.uu_last_shared_sat", world.uav_uav_edges[..., critic_schema.UU_LAST_SHARED_SAT_FRAC])
    _assert_unit_interval("critic.uu_prefix_shared_sat", world.uav_uav_edges[..., critic_schema.UU_PREFIX_SHARED_SAT_FRAC])
    _assert_unit_interval("critic.uu_prefix_shared_known", world.uav_uav_edges[..., critic_schema.UU_PREFIX_SHARED_SAT_KNOWN])
    _assert_unit_interval("critic.global_prefix_workload_known", world.global_scalars[..., critic_schema.GLOBAL_PREFIX_WORKLOAD_KNOWN])
    _assert_unit_interval("critic.global_selected_sat_load_known", world.global_scalars[..., critic_schema.GLOBAL_SELECTED_SAT_LOAD_KNOWN])


def _assert_native_inputs_are_consumable_by_networks(cfg: SaginConfig, history, *, device: torch.device) -> None:
    cfg.critic_edge_embed_dim = 32
    cfg.critic_global_embed_dim = 32
    cfg.critic_system_token_dim = 32
    cfg.critic_value_head_hidden = 32
    cfg.critic_message_layers = 1
    bundle = build_structured_modules_from_config(
        cfg,
        hidden_dim=32,
        embed_dim=32,
        critic_hidden_dim=32,
        critic_embed_dim=32,
    )
    actor = bundle.actor.to(device=device).eval()
    critic = bundle.critic.to(device=device).eval()
    assert critic is not None
    with torch.no_grad():
        accel_out = actor.accel_policy(history.accel_stage, deterministic=True)
        for name in ("action", "logprob", "entropy", "mean", "std"):
            _assert_finite_and_bounded(f"accel_network.{name}", getattr(accel_out, name))
        _assert_finite_and_bounded(
            "accel_network.eval_logprob",
            actor.accel_policy.evaluate_actions(history.accel_stage, accel_out.action).logprob,
        )

        sat_out = actor.sat_subset_policy(history.sat_stage, deterministic=True)
        for name in ("selected_sat_indices", "subset_index", "subset_members", "logprob", "entropy", "logits"):
            _assert_finite_and_bounded(f"sat_network.{name}", getattr(sat_out, name))
        assert bool(((sat_out.selected_sat_indices >= -1) & (sat_out.selected_sat_indices < int(cfg.num_sat))).all().item())
        _assert_finite_and_bounded(
            "sat_network.eval_logprob",
            actor.sat_subset_policy.evaluate_actions(history.sat_stage, sat_out.subset_index).logprob,
        )

        bw_out = actor.bw_policy(history.bw_stage, deterministic=True)
        for name in (
            "action",
            "logprob",
            "entropy",
            "logprob_raw",
            "entropy_raw",
            "det_mean",
            "alpha",
            "kappa",
            "tau",
        ):
            _assert_finite_and_bounded(f"bw_network.{name}", getattr(bw_out, name))
        valid = (history.bw_stage.gu_mask > 0.5) & (history.bw_stage.bw_valid_mask > 0.5)
        _assert_finite_and_bounded("bw_network.valid_score", bw_out.score[valid])
        if bool((~valid).any().item()):
            assert bool((bw_out.score[~valid] <= -1.0e8).all().item())
        active_rows = valid.sum(dim=-1) > 0
        if bool(active_rows.any().item()):
            torch.testing.assert_close(
                bw_out.action.masked_fill(~valid, 0.0).sum(dim=-1)[active_rows],
                torch.ones_like(bw_out.valid_count.to(dtype=bw_out.action.dtype)[active_rows]),
                atol=1.0e-4,
                rtol=1.0e-4,
            )
        _assert_finite_and_bounded(
            "bw_network.eval_logprob",
            actor.bw_policy.evaluate_actions(history.bw_stage, bw_out.action).logprob,
        )

        for stage_name, stage in (("accel", history.accel_stage), ("sat", history.sat_stage), ("bw", history.bw_stage)):
            values = critic(stage.world_batch)
            for head_name, value in values.items():
                _assert_finite_and_bounded(f"critic_network.{stage_name}.{head_name}", value)
                assert tuple(value.shape) == (int(stage.world_batch.uav_nodes.shape[0]),)


def _run_native_done_reset_applies_preload_queue_tape_without_python_reset(device: str) -> None:
    cfg = _acceptance_cfg()
    cfg.structured_env_tensor_backend = str(device)
    cfg.d_safe = 2000.0
    cfg.uav_init_min_spacing = 0.0
    cfg.pl_threshold_db = 1.0e9
    cfg.task_arrival_rate = 10.0
    cfg.tau0 = 1.0
    cfg.traffic_model = "sticky_subset_hotspot"
    cfg.hotspot_num_subsets = 2
    cfg.hotspot_subset_size = 1
    cfg.preload_enabled = True
    cfg.preload_prob = 1.0
    cfg.preload_hot_gu_steps = 3.0
    cfg.preload_bg_gu_steps = 0.2
    cfg.preload_hot_uav_steps = 2.0
    cfg.preload_sat_steps = 1.5
    cfg.queue_max_gu = 1000.0
    cfg.queue_max_uav = 1000.0
    cfg.queue_max_sat = 1000.0
    env_group = make_structured_env_group(cfg, num_envs=1, backend="sync", mode="eval")
    try:
        env_group.reset_many([int(cfg.seed) * 97])
        env_group.set_tensor_device(torch.device(device))
        env_group.begin_native_main_kernel_rollout(capacity=2, num_envs=1)
        result = env_group.native_rollout_program().replay_step(
            actor_bridge=_zero_native_action_bridge(cfg, device=device, num_envs=1),
            deterministic=True,
            rollout_tail=False,
        )
        tensor_state = env_group.batch_core._runtime_tensor_state
        gu_queue = tensor_state.gu_queue.reshape(int(cfg.num_gu))
        uav_queue = tensor_state.uav_queue.reshape(int(cfg.num_uav))
        sat_queue = tensor_state.sat_queue.reshape(int(cfg.num_sat))
        arrival_rate_vec = tensor_state.last_gu_arrival_rate_vec.reshape(int(cfg.num_gu))
        assert bool(result.terminated.reshape(-1)[0].item())
        assert int(tensor_state.t.reshape(-1)[0].item()) == 0
        assert torch.max(gu_queue) > torch.min(gu_queue)
        assert torch.max(gu_queue) > 0.0
        assert torch.max(uav_queue) > 0.0
        assert torch.all(sat_queue > 0.0)
        assert torch.max(arrival_rate_vec) > torch.min(arrival_rate_vec)
        assert env_group.native_rollout_runtime.main.accel_stage_field_buffers is not None
        assert env_group.native_rollout_runtime.main.accel_live_obs_buffers is not None
    finally:
        close_structured_env_group(env_group)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native GPU reset validation")
def test_cuda_native_done_reset_applies_preload_queue_tape_without_python_reset():
    _run_native_done_reset_applies_preload_queue_tape_without_python_reset("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native 14.3 acceptance matrix")
def test_cuda_native_long_rollout_acceptance_matrix_covers_doc_14_3(tmp_path):
    from sagin_marl.rl.structured_eval import validate_structured_fixed_seed_long_rollout_acceptance_matrix
    from sagin_marl.rl.structured_factory import build_structured_modules_from_config

    cfg = _acceptance_cfg()
    cfg.structured_env_tensor_backend = "cuda"
    cfg.T_steps = 2
    cfg.reward_mode = "weighted_workload_level"
    teacher_actor = build_structured_modules_from_config(cfg, build_critic=False).actor
    teacher_path = tmp_path / "teacher_actor.pt"
    torch.save(teacher_actor.state_dict(), teacher_path)
    cfg.exec_teacher_actor_path = str(teacher_path)
    cfg.exec_teacher_deterministic = True
    report = validate_structured_fixed_seed_long_rollout_acceptance_matrix(
        cfg,
        baseline_policy="cluster_center_queue_aware",
        episodes=1,
        episode_seed_base=170,
        num_envs=1,
        native_tensor_backend="cuda",
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert bool(report["passed"]), report
    assert bool(report["exact_passed"]), report
    expected_source_cases = {
        "policy_policy_policy",
        "queue_aware_queue_aware_queue_aware",
        "cluster_center_queue_aware_cluster_center_queue_aware_cluster_center_queue_aware",
        "zero_policy_policy",
        "policy_zero_policy",
        "policy_policy_zero",
        "queue_aware_policy_queue_aware",
        "cluster_center_queue_aware_policy_queue_aware",
        "teacher_teacher_teacher",
        "teacher_policy_queue_aware",
        "zero_zero_zero",
    }
    assert len(report["cases"]) == len(expected_source_cases) + 3 + 2
    assert set(report["source_cases"]) == expected_source_cases
    assert set(report["flow_cases"]) == {
        "flow_disabled",
        "flow_enabled_executed",
        "flow_enabled_deterministic",
        "flow_enabled_external_live_override",
    }
    assert all(bool(value) for value in dict(report["coverage"]).values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native flow proxy override validation")
def test_cuda_native_flow_proxy_external_live_override_uses_fixed_zero_constant():
    cfg = _acceptance_cfg()
    cfg.structured_env_tensor_backend = "cuda"
    cfg.bw_flow_proxy_aux_enabled = True
    cfg.bw_flow_proxy_base_action_mode = "external_live_override"
    env_group = make_structured_env_group(cfg, num_envs=1, backend="sync", mode="eval")
    try:
        env_group.reset_many([int(cfg.seed) * 101])
        env_group.set_tensor_device(torch.device("cuda"))
        env_group.begin_native_main_kernel_rollout(capacity=2, num_envs=1)
        runtime = env_group.native_rollout_runtime
        assert runtime.main.flow_proxy_base_action_mode_code == native_cuda.FLOW_BASE_EXTERNAL_LIVE_OVERRIDE
        assert torch.is_tensor(runtime.main.live_bw_flow_proxy_override_action)
        torch.testing.assert_close(
            runtime.main.live_bw_flow_proxy_override_action,
            torch.zeros_like(runtime.main.live_bw_flow_proxy_override_action),
        )
    finally:
        close_structured_env_group(env_group)


def test_native_ppo_collect_horizon_only_submits_official_runtime_history():
    class _FakeLearner:
        cfg = _acceptance_cfg()
        exec_source_by_stage = {0: "policy", 1: "policy", 2: "policy"}

        @staticmethod
        def _structured_env_tensor_device():
            return torch.device("cpu")

        @staticmethod
        def _ensure_native_actor_cuda_bindings(*_args, **_kwargs):
            return None

        @staticmethod
        def _require_native_actor_policy_binding():
            return SimpleNamespace(abi=SimpleNamespace())

    class _FakeProgram:
        def __init__(self, history):
            self.runtime = SimpleNamespace(
                history=history,
                main=SimpleNamespace(
                    accel_actor_source_mode_code=0,
                    sat_actor_source_mode_code=0,
                    bw_actor_source_mode_code=0,
                    flow_proxy_base_action_mode_code=0,
                    native_cuda_abi=native_cuda.NativeCudaRuntimeABI(
                        float_tensors=(),
                        long_tensors=(),
                        bool_tensors=(),
                        int_tensors=(),
                        int_params=(),
                        float_params=(),
                    ),
                ),
            )
            self.replay_calls = 0

        def replay_horizon(self, **_kwargs):
            self.replay_calls += 1
            return [
                StructuredBatchStepResult(
                    team_rewards=torch.zeros((1,), dtype=torch.float32),
                    terminated=torch.zeros((1,), dtype=torch.bool),
                    truncated=torch.zeros((1,), dtype=torch.bool),
                )
            ]

    class _FakeDrivers:
        def __init__(self, replay_program):
            self.replay_program = replay_program
            self.sub_batch_calls = 0
            self.hot_calls = 0

        def __len__(self):
            return 1

        def native_hot_replay_program(self, *_args, **_kwargs):
            self.hot_calls += 1
            raise AssertionError("official native non-record rollout must not call native_hot_replay_program")

        def native_sub_batch_rollout_program(self, *, capacity: int, selected_indices):
            assert capacity == 1
            assert tuple(selected_indices) == (0,)
            self.sub_batch_calls += 1
            return self.replay_program

    class _FakeBuffer:
        def __init__(self):
            self.added_history = None

        def add_native_rollout_training_view(self, *, history, num_steps: int, num_envs: int):
            self.added_history = history
            assert num_steps == 1
            assert num_envs == 1

    official_history = object()
    replay_history = object()
    official_program = _FakeProgram(official_history)
    replay_program = _FakeProgram(replay_history)
    drivers = _FakeDrivers(replay_program)
    collector = StructuredGpuRolloutProgram.__new__(StructuredGpuRolloutProgram)
    collector.learner = _FakeLearner()
    collector.drivers = drivers
    collector.program = official_program
    collector.runtime = official_program.runtime

    buffer = _FakeBuffer()
    collector.collect_horizon(horizon=1, buffer=buffer, deterministic=True)
    assert buffer.added_history is official_history
    assert buffer.added_history is not replay_history
    assert drivers.hot_calls == 0
    assert drivers.sub_batch_calls == 0
    assert official_program.replay_calls == 1
    assert replay_program.replay_calls == 0

    collector.collect_horizon(horizon=1, buffer=None, deterministic=True)
    assert drivers.hot_calls == 0
    assert drivers.sub_batch_calls == 1
    assert replay_program.replay_calls == 1
    assert buffer.added_history is official_history


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native direct main-kernel validation")
def test_cuda_native_main_kernel_direct_path_does_not_call_legacy_stage_helpers(monkeypatch):
    import sagin_marl.env.structured_batch_env_core as core_module

    cfg = _acceptance_cfg()
    cfg.structured_env_tensor_backend = "cuda"
    cfg.avoidance_enabled = True
    cfg.avoidance_eta = 3.5
    cfg.avoidance_adaptive_enabled = False
    cfg.sat_candidate_mode = "elevation"
    cfg.structured_kernel_operator_mode = "compile"
    cfg.structured_kernel_compile_required = "bw_workload_costs"
    cfg.structured_native_main_kernel_require_compiled_segments = True
    cfg.structured_native_main_kernel_required_compile_names = cfg.structured_kernel_compile_required

    env_group = make_structured_env_group(cfg, num_envs=1, backend="sync", mode="eval")
    try:
        env_group.reset_many([int(cfg.seed) * 19])
        env_group.set_tensor_device(torch.device("cuda"))
        env_group.begin_native_main_kernel_rollout(capacity=4, num_envs=1)
        assert env_group.native_rollout_runtime.main.native_cuda_abi is not None
        assert env_group.native_rollout_runtime.main.native_cuda_marker.device.type == "cuda"
        program = env_group.native_rollout_program()
        assert not hasattr(program, "step_program")
        for name in ("publish_accel_obs", "publish_sat_obs_from_accel_action", "replay_sat_bw_tail"):
            assert not hasattr(program, name)

        def _unexpected(*_args, **_kwargs):
            raise AssertionError("CUDA native main-kernel direct path must not call legacy stage helpers")

        removed_core_names = (
            "_prepare_accel_specs_native",
            "_build_bw_specs_from_selection_matrices",
            "_bw_snapshot_batch_native",
            "_execute_bw_stage_batch_core",
            "_native_main_kernel_stage_world_ref",
            "build_local_accel_from_stage_register",
            "build_local_sat_from_stage_register",
            "build_local_bw_from_stage_register",
        )
        for name in removed_core_names:
            assert not hasattr(env_group.batch_core, name)
        removed_runtime_names = (
            "write_stage_register",
            "write_stage_batch_history",
            "write_kernel_cache",
            "write_kernel_cache_view",
            "write_kernel_cache_world_history",
            "write_fixed_stage_cache_from_stage_fields",
            "write_fixed_stage_cache_from_base",
            "write_fixed_stage_cache_view",
            "fixed_stage_cache_view",
        )
        for name in removed_runtime_names:
            assert not hasattr(env_group.native_rollout_runtime, name)
        legacy_helper_names = (
            "_decode_sat_subset_indices_to_selection_matrices",
            "_decode_sat_actions_to_selection_matrices",
        )
        for name in legacy_helper_names:
            assert not hasattr(env_group.batch_core, name)
        assert not hasattr(env_group.native_rollout_runtime, "write_current_rollout_step_history")
        for name in ("LocalAccelState", "LocalSatState", "LocalBwState"):
            monkeypatch.setattr(core_module, name, _unexpected)
        assert not hasattr(core_module, "StructuredNativeStageRegister")
        assert not hasattr(core_module, "StructuredGpuTypedKernelCache")
        assert not hasattr(core_module, "_NativeBwStepTensorFields")
        assert not hasattr(core_module, "_NativeBwStepNextAccelTensorFields")
        assert not hasattr(core_module, "_execute_bw_native_fused_step_next_accel_obs_tensor_impl")
        source = Path(core_module.__file__).read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        hot_functions = {
            "_runtime_tensor_prepare_stage_and_obs",
            "_runtime_tensor_apply_accel_publish_sat_obs_impl",
            "_runtime_tensor_apply_sat_publish_bw_obs_impl",
            "_execute_bw_native_fused_step_out_tensor_impl",
            "_execute_bw_native_fused_step_next_accel_obs_out_tensor_impl",
            "_execute_bw_stage_native_main_direct_out",
        }
        offenders: list[tuple[str, int]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name not in hot_functions:
                continue
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Name)
                    and child.func.id in {"_as_grouped_tensor_batch", "_resolve_kernel_callable"}
                ):
                    offenders.append((node.name, int(child.lineno)))
        assert offenders == []

        accel_action = np.zeros((cfg.num_uav, 2), dtype=np.float32)
        sat_action = -np.ones((cfg.num_uav, cfg.sat_action_select_k), dtype=np.int64)
        bw_action = np.zeros((cfg.num_uav, cfg.num_gu), dtype=np.float32)

        runtime = env_group.native_rollout_runtime
        assert runtime.random.reset_gu_pos is not None
        assert runtime.random.reset_uav_pos is not None
        assert runtime.random.reset_uav_vel is not None
        assert runtime.random.reset_deadline_steps is not None
        assert runtime.random.reset_doppler_residual is not None
        assert runtime.random.arrival_rollout_tape is not None
        assert runtime.random.arrival_rate_rollout_tape is not None
        assert runtime.random.reset_gu_pos.device.type == "cuda"
        assert runtime.random.arrival_rollout_tape.device.type == "cuda"
        program.replay_step(
            actor_bridge=_FixedNativeActionBridge(
                cfg=cfg,
                device="cuda",
                num_envs=1,
                accel_action=accel_action,
                sat_action=sat_action[:, 0],
                bw_action=bw_action,
            ),
            deterministic=True,
            rollout_tail=False,
        )
        for name in ("cpu", "numpy", "item", "tolist"):
            monkeypatch.setattr(torch.Tensor, name, _unexpected)
        for name in ("write_history_world_fields_slot", "write_history_world_buffer_slot", "history_world_buffer_view"):
            assert not hasattr(type(env_group.native_rollout_runtime), name)
        assert not hasattr(env_group.batch_core, "_execute_bw_stage_native_main_tensor_only")
        assert not hasattr(env_group.batch_core, "_native_world_from_stage_fields")
        assert not hasattr(env_group.batch_core, "_native_world_ref_from_stage_fields")
        result = program.replay_step(
            actor_bridge=_FixedNativeActionBridge(
                cfg=cfg,
                device="cuda",
                num_envs=1,
                accel_action=accel_action,
                sat_action=sat_action[:, 0],
                bw_action=bw_action,
            ),
            deterministic=True,
            rollout_tail=False,
        )
        assert env_group.native_rollout_runtime.main.accel_stage_fields is not None
        assert env_group.native_rollout_runtime.main.sat_stage_fields is not None
        assert env_group.native_rollout_runtime.main.bw_stage_fields is not None
        assert env_group.native_rollout_runtime.main.accel_stage_field_buffers is not None
        assert env_group.native_rollout_runtime.main.accel_live_obs_buffers is not None
        assert not hasattr(env_group.native_rollout_runtime.main, "accel_world_ref")
        assert not hasattr(env_group.native_rollout_runtime.main, "sat_world_ref")
        assert not hasattr(env_group.native_rollout_runtime.main, "bw_world_ref")
        assert not hasattr(env_group.native_rollout_runtime.main, "next_world_ref")
        assert env_group.native_rollout_runtime.main.step_result_view is not None
        assert not hasattr(env_group.native_rollout_runtime.history, "world_value_buffers")
        assert env_group.native_rollout_runtime.history.accel_stage.world_batch is not None
        assert env_group.native_rollout_runtime.history.sat_stage.world_batch is not None
        assert env_group.native_rollout_runtime.history.bw_stage.world_batch is not None
        assert not hasattr(env_group.native_rollout_runtime.history, "bw_next_world_batch")
        assert env_group.native_rollout_runtime.history.terminal_next_world is not None
        assert env_group.native_rollout_runtime.history.terminal_next_world_mask is not None
        assert env_group.native_rollout_runtime.random.arrivals is not None
        assert env_group.native_rollout_runtime.random.arrival_rates is not None
        assert env_group.native_rollout_runtime.random.arrivals.device.type == "cuda"
        assert torch.is_tensor(result.team_rewards)
        assert result.team_rewards.device.type == "cuda"
    finally:
        close_structured_env_group(env_group)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native direct main-kernel validation")
def test_cuda_native_main_kernel_no_history_uses_field_buffers_not_stage_group(monkeypatch):
    cfg = _acceptance_cfg()
    cfg.structured_env_tensor_backend = "cuda"
    cfg.avoidance_enabled = True
    cfg.avoidance_eta = 3.5
    cfg.avoidance_adaptive_enabled = False
    cfg.sat_candidate_mode = "elevation"
    cfg.structured_kernel_operator_mode = "compile"
    cfg.structured_kernel_compile_required = "bw_workload_costs"
    cfg.structured_native_main_kernel_require_compiled_segments = True
    cfg.structured_native_main_kernel_required_compile_names = cfg.structured_kernel_compile_required

    env_group = make_structured_env_group(cfg, num_envs=1, backend="sync", mode="eval")
    try:
        env_group.reset_many([int(cfg.seed) * 31])
        env_group.set_tensor_device(torch.device("cuda"))
        env_group.begin_native_main_kernel_rollout(capacity=2, num_envs=1)
        program = env_group.native_rollout_program()

        def _stage_group_forbidden(*_args, **_kwargs):
            raise AssertionError("CUDA no-history direct live path must not use stage group dynamic access")

        assert not hasattr(StructuredGpuRolloutRuntime, "write_main_stage_fields")
        assert not hasattr(StructuredGpuRolloutRuntime, "copy_main_stage")

        result = program.replay_step(
            actor_bridge=_FixedNativeActionBridge(
                cfg=cfg,
                device="cuda",
                num_envs=1,
                accel_action=torch.zeros((1, int(cfg.num_uav), 2), dtype=torch.float32, device="cuda"),
                sat_action=torch.zeros((1, int(cfg.num_uav)), dtype=torch.long, device="cuda"),
                bw_action=torch.zeros((1, int(cfg.num_uav), int(cfg.num_gu)), dtype=torch.float32, device="cuda"),
            ),
            deterministic=True,
            rollout_tail=False,
        )
        runtime = program.runtime
        assert runtime.main.accel_stage_fields is not None
        assert runtime.main.sat_stage_fields is not None
        assert runtime.main.bw_stage_fields is not None
        assert runtime.main.accel_stage_field_buffers is not None
        assert runtime.main.accel_live_obs_buffers is not None
        assert torch.is_tensor(result.team_rewards)
        assert result.team_rewards.device.type == "cuda"
    finally:
        close_structured_env_group(env_group)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native hot replay validation")
@pytest.mark.parametrize("prepare_next_accel", [False, True])
def test_cuda_native_main_kernel_hot_replay_does_not_read_static_spec(prepare_next_accel):
    import sagin_marl.env.structured_batch_env_core as core_module

    cfg = _acceptance_cfg()
    cfg.structured_env_tensor_backend = "cuda"
    cfg.avoidance_enabled = True
    cfg.avoidance_eta = 3.5
    cfg.avoidance_adaptive_enabled = False
    cfg.sat_candidate_mode = "elevation"
    cfg.structured_kernel_operator_mode = "compile"
    cfg.structured_kernel_compile_backend = "cudagraphs"
    cfg.structured_kernel_compile_cudagraphs = True
    cfg.structured_kernel_cudagraph_direct_inputs = True
    cfg.structured_kernel_compile_required = "bw_workload_costs"
    cfg.structured_native_main_kernel_require_compiled_segments = True
    cfg.structured_native_main_kernel_required_compile_names = cfg.structured_kernel_compile_required

    env_group = make_structured_env_group(cfg, num_envs=1, backend="sync", mode="eval")
    try:
        env_group.reset_many([int(cfg.seed) * (37 if prepare_next_accel else 41)])
        env_group.set_tensor_device(torch.device("cuda"))
        env_group.begin_native_main_kernel_rollout(capacity=5, num_envs=1)
        assert not hasattr(env_group, "native_hot_replay_program")
        program = env_group.debug_native_hot_replay_program(capacity=5, num_envs=1)
        bridge = _FixedNativeActionBridge(
            cfg=cfg,
            device="cuda",
            num_envs=1,
            accel_action=torch.zeros((1, int(cfg.num_uav), 2), dtype=torch.float32, device="cuda"),
            sat_action=torch.zeros((1, int(cfg.num_uav)), dtype=torch.long, device="cuda"),
            bw_action=torch.zeros((1, int(cfg.num_uav), int(cfg.num_gu)), dtype=torch.float32, device="cuda"),
        )

        # First replay is graph build/capture. The second replay is the hot path
        # and must not fall back into Python config/spec reads.
        # `rollout_tail=False` uses a bounded ping-pong of persistent
        # current/next accel buffers, so the direct-input graph owns two
        # stable pointer variants (A->B and B->A). Only after both variants
        # have been captured does hot replay become allocation-free.
        program.replay_step(
            actor_bridge=bridge,
            deterministic=True,
            rollout_tail=not prepare_next_accel,
        )
        if prepare_next_accel:
            program.replay_step(actor_bridge=bridge, deterministic=True, rollout_tail=False)
        if prepare_next_accel:
            program.replay_step(actor_bridge=bridge, deterministic=True, rollout_tail=False)
        with core_module.NativeMainKernelHotReplayGuard():
            result = program.replay_step(
                actor_bridge=bridge,
                deterministic=True,
                rollout_tail=not prepare_next_accel,
            )
        assert torch.is_tensor(result.team_rewards)
        assert result.team_rewards.device.type == "cuda"
    finally:
        close_structured_env_group(env_group)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native strict graph validation")
def test_cuda_native_main_kernel_rollout_begin_enforces_strict_graph_contract():
    cfg = _acceptance_cfg()
    cfg.structured_env_tensor_backend = "cuda"
    cfg.avoidance_enabled = True
    cfg.avoidance_eta = 3.5
    cfg.avoidance_adaptive_enabled = False
    cfg.sat_candidate_mode = "elevation"
    cfg.structured_kernel_operator_mode = "auto"
    cfg.structured_kernel_compile_backend = "auto"
    cfg.structured_kernel_compile_cudagraphs = False
    cfg.structured_kernel_cudagraph_direct_inputs = False
    cfg.structured_kernel_auto_compile_warmup_calls = 8
    cfg.structured_kernel_compile_required = ""
    cfg.structured_native_main_kernel_require_compiled_segments = True
    cfg.structured_native_main_kernel_required_compile_names = ""

    env_group = make_structured_env_group(cfg, num_envs=1, backend="sync", mode="eval")
    try:
        env_group.reset_many([int(cfg.seed) * 39])
        env_group.set_tensor_device(torch.device("cuda"))
        env_group.begin_native_main_kernel_rollout(capacity=2, num_envs=1)

        expected = set(env_group.batch_core._native_main_kernel_kernel_specs().keys())
        assert "global_state_batch" not in expected
        assert expected == set()
        assert env_group.native_rollout_runtime.main.native_cuda_abi is not None

        program = env_group.native_rollout_program()
        result = program.replay_step(
            actor_bridge=_FixedNativeActionBridge(
                cfg=cfg,
                device="cuda",
                num_envs=1,
                accel_action=torch.zeros((1, int(cfg.num_uav), 2), dtype=torch.float32, device="cuda"),
                sat_action=torch.zeros((1, int(cfg.num_uav)), dtype=torch.long, device="cuda"),
                bw_action=torch.zeros((1, int(cfg.num_uav), int(cfg.num_gu)), dtype=torch.float32, device="cuda"),
            ),
            deterministic=True,
            rollout_tail=False,
        )
        assert torch.is_tensor(result.team_rewards)
        assert env_group.native_rollout_runtime.main.native_cuda_marker.device.type == "cuda"
    finally:
        close_structured_env_group(env_group)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native partial hot replay validation")
def test_cuda_native_hot_replay_supports_persistent_partial_batch_workspace():
    cfg = _acceptance_cfg()
    cfg.structured_env_tensor_backend = "cuda"
    cfg.sat_candidate_mode = "elevation"
    cfg.structured_kernel_operator_mode = "auto"
    cfg.structured_kernel_compile_backend = "auto"
    cfg.structured_kernel_compile_cudagraphs = False
    cfg.structured_kernel_cudagraph_direct_inputs = False
    cfg.structured_kernel_compile_required = ""
    cfg.structured_native_main_kernel_require_compiled_segments = True
    cfg.structured_native_main_kernel_required_compile_names = ""

    env_group = make_structured_env_group(cfg, num_envs=2, backend="sync", mode="eval")
    try:
        env_group.reset_many([int(cfg.seed) * 53, int(cfg.seed) * 59])
        env_group.set_tensor_device(torch.device("cuda"))
        env_group.begin_native_main_kernel_rollout(capacity=3, num_envs=2)

        selected_uav_pos = env_group.batch_core.runtime_tensor_state.uav_pos[1].detach().clone()
        assert not hasattr(env_group, "native_hot_replay_program")
        program = env_group.native_sub_batch_rollout_program(capacity=3, selected_indices=[1])
        sub_state_initial = env_group.batch_core._native_sub_batch_tensor_state
        assert sub_state_initial is not None
        assert torch.allclose(sub_state_initial.uav_pos[0], selected_uav_pos)
        assert tuple(program.runtime.main.selected_env_mapping.detach().cpu().tolist()) == (1,)
        result = program.replay_step(
            actor_bridge=_zero_native_action_bridge(cfg, device="cuda", num_envs=1),
            deterministic=True,
            rollout_tail=False,
        )
        sub_runtime = program.runtime
        sub_state = env_group.batch_core._native_sub_batch_tensor_state

        assert sub_runtime.main.num_envs == 1
        assert sub_runtime.history.num_envs == 1
        assert sub_state is not None
        assert int(sub_state.uav_pos.shape[0]) == 1
        assert tuple(sub_runtime.main.selected_env_mapping.detach().cpu().tolist()) == (1,)
        assert torch.is_tensor(result.team_rewards)
        assert tuple(result.team_rewards.shape) == (1,)
        assert env_group.native_rollout_runtime.main.num_envs == 2
    finally:
        close_structured_env_group(env_group)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native actor branch matrix")
def test_cuda_native_actor_branch_matrix_uses_philox_and_fixed_live_abi():
    from sagin_marl.rl.native_actor_cuda import build_native_actor_cuda_binding
    from sagin_marl.rl.structured_factory import build_structured_modules_from_config

    base_cfg = _acceptance_cfg()
    base_cfg.structured_env_tensor_backend = "cuda"
    base_cfg.sat_candidate_mode = "elevation"
    base_cfg.structured_kernel_operator_mode = "auto"
    base_cfg.structured_kernel_compile_backend = "auto"
    base_cfg.structured_kernel_compile_cudagraphs = False
    base_cfg.structured_native_main_kernel_require_compiled_segments = True
    env_group = make_structured_env_group(base_cfg, num_envs=1, backend="sync", mode="eval")
    try:
        env_group.reset_many([int(base_cfg.seed) * 61])
        env_group.set_tensor_device(torch.device("cuda"))
        env_group.begin_native_main_kernel_rollout(capacity=2, num_envs=1)
        runtime = env_group.native_rollout_runtime
        abi = runtime.main.native_cuda_abi
        branch_updates = [
            {},
            {"bw_down_query_count": 1},
            {"bw_competition_layers": 1, "bw_attention_heads": 1},
            {"bw_tau_min": 0.4, "bw_tau_max": 1.6},
            {"bw_kappa_min": 0.75, "bw_kappa_max": 24.0},
        ]
        for case_index, updates in enumerate(branch_updates):
            cfg = _acceptance_cfg()
            cfg.structured_env_tensor_backend = "cuda"
            cfg.sat_candidate_mode = "elevation"
            cfg.bw_attention_heads = 1
            for key, value in updates.items():
                setattr(cfg, key, value)
            torch.manual_seed(10_000 + int(case_index))
            actor = build_structured_modules_from_config(cfg, hidden_dim=32, embed_dim=16, build_critic=False).actor.to("cuda")
            actor.eval()
            actor.native_cuda_rng_seed = 0x1234000000000000 + int(case_index)
            binding = build_native_actor_cuda_binding(actor, device="cuda")
            runtime.main.accel_active_idx = 0
            native_cuda.prepare_initial_accel_live(
                abi,
                slot=0,
                active_idx=0,
                accel_source_mode=native_cuda.SOURCE_POLICY,
                sat_source_mode=native_cuda.SOURCE_POLICY,
                bw_source_mode=native_cuda.SOURCE_POLICY,
            )
            with torch.no_grad():
                expected_accel = actor.accel_policy(runtime.main.accel_live_obs_buffers[0], deterministic=True)
            native_cuda.actor_accel_live(abi, binding.abi, active_idx=0, deterministic=True, rng_step=3)
            torch.testing.assert_close(
                runtime.main.live_accel_action.reshape_as(expected_accel.action),
                expected_accel.action,
                atol=2.0e-5,
                rtol=2.0e-5,
            )
            torch.testing.assert_close(
                runtime.main.live_accel_old_logprob.reshape_as(expected_accel.logprob),
                expected_accel.logprob,
                atol=2.0e-5,
                rtol=2.0e-5,
            )
            native_cuda.accel_to_sat_live(
                abi,
                slot=0,
                active_idx=0,
                accel_source_mode=native_cuda.SOURCE_POLICY,
                sat_source_mode=native_cuda.SOURCE_POLICY,
                bw_source_mode=native_cuda.SOURCE_POLICY,
            )
            with torch.no_grad():
                expected_sat = actor.sat_subset_policy(runtime.main.live_sat_obs, deterministic=True)
            native_cuda.actor_sat_live(abi, binding.abi, deterministic=True, rng_step=4)
            torch.testing.assert_close(runtime.main.live_sat_subset_index.reshape_as(expected_sat.subset_index), expected_sat.subset_index)
            torch.testing.assert_close(runtime.main.live_sat_action_indices.reshape_as(expected_sat.selected_sat_indices), expected_sat.selected_sat_indices)
            torch.testing.assert_close(
                runtime.main.live_sat_old_logprobs_per_agent.reshape_as(expected_sat.logprob),
                expected_sat.logprob,
                atol=2.0e-5,
                rtol=2.0e-5,
            )
            native_cuda.sat_to_bw_live(
                abi,
                slot=0,
                active_idx=0,
                accel_source_mode=native_cuda.SOURCE_POLICY,
                sat_source_mode=native_cuda.SOURCE_POLICY,
                bw_source_mode=native_cuda.SOURCE_POLICY,
            )
            with torch.no_grad():
                expected_bw = actor.bw_policy(runtime.main.live_bw_obs, deterministic=True)
            native_cuda.actor_bw_live(abi, binding.abi, deterministic=True, rng_step=5)
            # With raw physical input LayerNorm disabled, the manual native
            # attention path and PyTorch MHA can differ at ~1e-3 in downstream
            # simplex probabilities. Keep this parity check tight enough for
            # wiring/index bugs, but do not make it a bitwise attention test.
            torch.testing.assert_close(
                runtime.main.live_bw_action.reshape_as(expected_bw.action),
                expected_bw.action,
                atol=5.0e-3,
                rtol=5.0e-3,
            )
            expected_ref = expected_bw.det_mean if expected_bw.det_mean is not None else expected_bw.action
            torch.testing.assert_close(
                runtime.main.live_bw_ref_action.reshape_as(expected_ref),
                expected_ref,
                atol=5.0e-3,
                rtol=5.0e-3,
            )
            torch.testing.assert_close(
                runtime.main.live_bw_old_logprobs_per_agent.reshape_as(expected_bw.logprob),
                expected_bw.logprob,
                atol=5.0e-4,
                rtol=5.0e-4,
            )
            runtime.main.accel_active_idx = 0
            native_cuda.prepare_initial_accel_live(
                abi,
                slot=0,
                active_idx=0,
                accel_source_mode=native_cuda.SOURCE_POLICY,
                sat_source_mode=native_cuda.SOURCE_POLICY,
                bw_source_mode=native_cuda.SOURCE_POLICY,
            )
            native_cuda.actor_accel_live(
                abi,
                binding.abi,
                active_idx=0,
                deterministic=False,
                rng_step=5,
            )
            accel_a = runtime.main.live_accel_action.detach().clone()
            accel_lp_a = runtime.main.live_accel_old_logprob.detach().clone()
            native_cuda.actor_accel_live(
                abi,
                binding.abi,
                active_idx=0,
                deterministic=False,
                rng_step=5,
            )
            accel_b = runtime.main.live_accel_action.detach().clone()
            native_cuda.actor_accel_live(
                abi,
                binding.abi,
                active_idx=0,
                deterministic=False,
                rng_step=6,
            )
            accel_c = runtime.main.live_accel_action.detach().clone()
            torch.testing.assert_close(accel_a, accel_b)
            assert torch.isfinite(accel_c).all()
            assert not torch.equal(accel_a, accel_c)
            with torch.no_grad():
                accel_eval = actor.accel_policy.evaluate_actions(runtime.main.accel_live_obs_buffers[0], accel_a.reshape(-1, 2))
            torch.testing.assert_close(
                accel_lp_a.reshape_as(accel_eval.logprob),
                accel_eval.logprob,
                atol=2.0e-5,
                rtol=2.0e-5,
            )

            native_cuda.accel_to_sat_live(
                abi,
                slot=0,
                active_idx=0,
                accel_source_mode=native_cuda.SOURCE_POLICY,
                sat_source_mode=native_cuda.SOURCE_POLICY,
                bw_source_mode=native_cuda.SOURCE_POLICY,
            )
            native_cuda.actor_sat_live(abi, binding.abi, deterministic=False, rng_step=7)
            assert torch.isfinite(runtime.main.live_sat_old_logprobs_per_agent).all()
            sat_action = runtime.main.live_sat_subset_index.detach().clone().reshape(-1)
            sat_logprob = runtime.main.live_sat_old_logprobs_per_agent.detach().clone()
            native_cuda.actor_sat_live(abi, binding.abi, deterministic=False, rng_step=7)
            torch.testing.assert_close(runtime.main.live_sat_subset_index.detach().clone().reshape(-1), sat_action)
            torch.testing.assert_close(runtime.main.live_sat_old_logprobs_per_agent.detach().clone(), sat_logprob)
            sat_variant_changed = False
            for rng_step in range(8, 24):
                native_cuda.actor_sat_live(abi, binding.abi, deterministic=False, rng_step=rng_step)
                sat_variant_changed = sat_variant_changed or not torch.equal(
                    runtime.main.live_sat_subset_index.detach().clone().reshape(-1),
                    sat_action,
                )
            with torch.no_grad():
                sat_mask_logits = actor.sat_subset_policy._compute_logits(runtime.main.live_sat_obs)
                sat_subset_mask = actor.sat_subset_policy._legal_subset_mask(
                    runtime.main.live_sat_obs,
                    sat_mask_logits,
                )
            if bool((sat_subset_mask.to(dtype=torch.int32).sum(dim=1) > 1).any().item()):
                assert sat_variant_changed
            native_cuda.actor_sat_live(abi, binding.abi, deterministic=False, rng_step=7)
            with torch.no_grad():
                sat_eval = actor.sat_subset_policy.evaluate_actions(runtime.main.live_sat_obs, sat_action)
            torch.testing.assert_close(
                sat_logprob.reshape_as(sat_eval.logprob),
                sat_eval.logprob,
                atol=2.0e-5,
                rtol=2.0e-5,
            )
            native_cuda.sat_to_bw_live(
                abi,
                slot=0,
                active_idx=0,
                accel_source_mode=native_cuda.SOURCE_POLICY,
                sat_source_mode=native_cuda.SOURCE_POLICY,
                bw_source_mode=native_cuda.SOURCE_POLICY,
            )
            native_cuda.actor_bw_live(abi, binding.abi, deterministic=False, rng_step=11)
            assert torch.isfinite(runtime.main.live_bw_action).all()
            assert torch.isfinite(runtime.main.live_bw_ref_action).all()
            assert torch.isfinite(runtime.main.live_bw_old_logprobs_per_agent).all()
            bw_action = runtime.main.live_bw_action.detach().clone().reshape(-1, int(cfg.num_gu))
            bw_logprob = runtime.main.live_bw_old_logprobs_per_agent.detach().clone()
            native_cuda.actor_bw_live(abi, binding.abi, deterministic=False, rng_step=11)
            torch.testing.assert_close(runtime.main.live_bw_action.detach().clone().reshape_as(bw_action), bw_action)
            torch.testing.assert_close(runtime.main.live_bw_old_logprobs_per_agent.detach().clone(), bw_logprob)
            bw_variant_changed = False
            for rng_step in range(12, 28):
                native_cuda.actor_bw_live(abi, binding.abi, deterministic=False, rng_step=rng_step)
                bw_variant_changed = bw_variant_changed or not torch.equal(
                    runtime.main.live_bw_action.detach().clone().reshape_as(bw_action),
                    bw_action,
                )
            assert bw_variant_changed
            native_cuda.actor_bw_live(abi, binding.abi, deterministic=False, rng_step=11)
            with torch.no_grad():
                bw_eval = actor.bw_policy.evaluate_actions(
                    runtime.main.live_bw_obs,
                    bw_action,
                )
            torch.testing.assert_close(
                runtime.main.live_bw_old_logprobs_per_agent.reshape_as(bw_eval.logprob),
                bw_eval.logprob,
                atol=3.0e-2,
                rtol=3.0e-2,
            )
    finally:
        close_structured_env_group(env_group)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native hot replay validation")
@pytest.mark.parametrize("prepare_next_accel", [False, True])
@pytest.mark.parametrize("doppler_noise_enabled", [False, True])
def test_cuda_native_main_kernel_hot_replay_does_not_allocate_live_tensors(
    monkeypatch,
    prepare_next_accel,
    doppler_noise_enabled,
):
    import sagin_marl.env.structured_batch_env_core as core_module

    cfg = _acceptance_cfg()
    cfg.structured_env_tensor_backend = "cuda"
    cfg.avoidance_enabled = True
    cfg.avoidance_eta = 3.5
    cfg.avoidance_adaptive_enabled = False
    cfg.sat_candidate_mode = "elevation"
    cfg.structured_kernel_operator_mode = "compile"
    cfg.structured_kernel_compile_backend = "cudagraphs"
    cfg.structured_kernel_compile_cudagraphs = True
    cfg.structured_kernel_cudagraph_direct_inputs = True
    cfg.structured_kernel_compile_required = "bw_workload_costs"
    cfg.structured_native_main_kernel_require_compiled_segments = True
    cfg.structured_native_main_kernel_required_compile_names = cfg.structured_kernel_compile_required
    if doppler_noise_enabled:
        cfg.doppler_precomp_mode = "residual_hz"
        cfg.doppler_residual_hz = 50.0
        cfg.doppler_residual_sigma_hz = 1.0

    env_group = make_structured_env_group(cfg, num_envs=1, backend="sync", mode="eval")
    try:
        seed_multiplier = 43 if prepare_next_accel else 47
        if doppler_noise_enabled:
            seed_multiplier += 2
        env_group.reset_many([int(cfg.seed) * seed_multiplier])
        env_group.set_tensor_device(torch.device("cuda"))
        env_group.begin_native_main_kernel_rollout(capacity=4, num_envs=1)
        assert not hasattr(env_group, "native_hot_replay_program")
        program = env_group.debug_native_hot_replay_program(capacity=4, num_envs=1)
        bridge = _FixedNativeActionBridge(
            cfg=cfg,
            device="cuda",
            num_envs=1,
            accel_action=torch.zeros((1, int(cfg.num_uav), 2), dtype=torch.float32, device="cuda"),
            sat_action=torch.zeros((1, int(cfg.num_uav)), dtype=torch.long, device="cuda"),
            bw_action=torch.zeros((1, int(cfg.num_uav), int(cfg.num_gu)), dtype=torch.float32, device="cuda"),
        )

        program.replay_step(
            actor_bridge=bridge,
            deterministic=True,
            rollout_tail=not prepare_next_accel,
        )
        if prepare_next_accel:
            program.replay_step(actor_bridge=bridge, deterministic=True, rollout_tail=False)

        def _factory_forbidden(*_args, **_kwargs):
            raise AssertionError("CUDA native hot replay must not allocate live tensors via torch factory functions")

        for name in (
            "empty",
            "empty_like",
            "zeros",
            "zeros_like",
            "ones",
            "ones_like",
            "full",
            "full_like",
            "arange",
            "eye",
            "tensor",
            "as_tensor",
            "stack",
        ):
            monkeypatch.setattr(torch, name, _factory_forbidden)
        monkeypatch.setattr(torch.Tensor, "clone", _factory_forbidden)

        def _step_meta_forbidden(*_args, **_kwargs):
            raise AssertionError("CUDA native hot replay must advance step metadata inside fused tensor segments")

        monkeypatch.setattr(env_group.batch_core, "_ensure_step_started_native", _step_meta_forbidden)
        if prepare_next_accel:
            def _begin_forbidden(*_args, **_kwargs):
                raise AssertionError("CUDA native hot replay must consume prepared next accel obs without begin-stage dispatch")

            monkeypatch.setattr(env_group.batch_core, "_runtime_step_begin_accel_obs", _begin_forbidden)

        with core_module.NativeMainKernelHotReplayGuard():
            result = program.replay_step(
                actor_bridge=bridge,
                deterministic=True,
                rollout_tail=not prepare_next_accel,
            )
        assert torch.is_tensor(result.team_rewards)
        assert result.team_rewards.device.type == "cuda"
    finally:
        close_structured_env_group(env_group)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native direct main-kernel validation")
def test_cuda_native_main_kernel_uses_persistent_sub_batch_program_for_partial_replay():
    cfg = _acceptance_cfg()
    cfg.structured_env_tensor_backend = "cuda"
    cfg.avoidance_enabled = True
    cfg.avoidance_eta = 3.5
    cfg.avoidance_adaptive_enabled = False
    cfg.sat_candidate_mode = "elevation"
    cfg.structured_kernel_operator_mode = "auto"
    cfg.structured_kernel_compile_backend = "auto"
    cfg.structured_kernel_compile_cudagraphs = False
    cfg.structured_kernel_cudagraph_direct_inputs = False
    cfg.structured_kernel_compile_required = ""
    cfg.structured_native_main_kernel_require_compiled_segments = True
    cfg.structured_native_main_kernel_required_compile_names = ""

    env_group = make_structured_env_group(cfg, num_envs=3, backend="sync", mode="eval")
    reference_group = make_structured_env_group(cfg, num_envs=3, backend="sync", mode="eval")
    try:
        seeds = [int(cfg.seed) * 53 + offset for offset in range(3)]
        env_group.reset_many(seeds)
        reference_group.reset_many(seeds)
        env_group.set_tensor_device(torch.device("cuda"))
        reference_group.set_tensor_device(torch.device("cuda"))
        env_group.begin_native_main_kernel_rollout(capacity=3, num_envs=3)
        reference_group.begin_native_main_kernel_rollout(capacity=3, num_envs=3)
        reference_group.batch_core._copy_selected_random_rollout_tapes(
            source_runtime=env_group.native_rollout_runtime,
            target_runtime=reference_group.native_rollout_runtime,
            selected_indices=(0, 1, 2),
        )
        for api_call in (
            lambda: env_group.current_obs_batch(),
            lambda: env_group.current_obs_many(),
            lambda: env_group.get_global_state_batch(),
            lambda: env_group.cluster_meta_batch(),
            lambda: env_group.cluster_meta_many(),
            lambda: env_group.refresh_stage_obs_cache_many(),
            lambda: env_group.export_runtime_state_batch(),
            lambda: env_group.load_runtime_state_batch([], indices=[]),
            lambda: env_group.sat_mask_to_ids_many([]),
        ):
            with pytest.raises(RuntimeError, match="legacy adapter API"):
                api_call()
        selected = [2, 0]
        program = env_group.native_sub_batch_rollout_program(capacity=3, selected_indices=selected)
        assert env_group.batch_core._native_sub_batch_runtime is program.runtime
        assert env_group.batch_core._native_hot_replay_runtime is None
        assert tuple(program.runtime.main.selected_env_mapping.detach().cpu().tolist()) == tuple(selected)
        official_random = env_group.native_rollout_runtime.random
        selected_t = torch.as_tensor(selected, dtype=torch.long, device="cuda")
        if torch.is_tensor(official_random.arrival_rollout_tape):
            torch.testing.assert_close(
                program.runtime.random.arrival_rollout_tape,
                official_random.arrival_rollout_tape.index_select(1, selected_t),
            )
        if torch.is_tensor(official_random.fading_gain_rollout_tape):
            torch.testing.assert_close(
                program.runtime.random.fading_gain_rollout_tape,
                official_random.fading_gain_rollout_tape.index_select(1, selected_t),
            )
        reference_program = reference_group.native_rollout_program()
        reference_bridge = _zero_native_action_bridge(cfg, device="cuda", num_envs=3)
        sub_bridge = _zero_native_action_bridge(cfg, device="cuda", num_envs=2)
        result = None
        for _ in range(2):
            reference_program.replay_step(
                actor_bridge=reference_bridge,
                deterministic=True,
                rollout_tail=False,
            )
            result = program.replay_step(
                actor_bridge=sub_bridge,
                deterministic=True,
                rollout_tail=False,
            )
        assert torch.is_tensor(result.team_rewards)
        assert tuple(result.team_rewards.shape) == (2,)
        sub_state = env_group.batch_core._native_sub_batch_tensor_state
        ref_state = reference_group.batch_core.runtime_tensor_state
        assert sub_state is not None
        for field_name in (
            "uav_pos",
            "uav_vel",
            "uav_queue",
            "gu_queue",
            "sat_queue",
            "last_gu_arrival",
            "last_gu_arrival_rate_vec",
            "sat_pos",
            "sat_vel",
        ):
            try:
                torch.testing.assert_close(
                    getattr(sub_state, field_name),
                    getattr(ref_state, field_name).index_select(0, selected_t),
                    atol=1.0e-5,
                    rtol=1.0e-5,
                )
            except AssertionError as exc:
                raise AssertionError(f"selected-env sub-batch mismatch for {field_name}") from exc
        torch.testing.assert_close(sub_state.t, ref_state.t.index_select(0, selected_t))
    finally:
        close_structured_env_group(env_group)
        close_structured_env_group(reference_group)


def test_native_batch_core_rejects_materialized_legacy_slots_at_construction():
    cfg = _acceptance_cfg()
    with pytest.raises(TypeError):
        StructuredBatchEnvCore(cfg, envs=[object()])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native direct main-kernel validation")
def test_cuda_native_main_kernel_strict_score_candidate_mode_stays_direct():
    cfg = _acceptance_cfg()
    cfg.structured_env_tensor_backend = "cuda"
    cfg.avoidance_enabled = True
    cfg.avoidance_eta = 3.5
    cfg.avoidance_adaptive_enabled = False
    cfg.sat_candidate_mode = "score"
    cfg.structured_kernel_operator_mode = "compile"
    cfg.structured_kernel_compile_backend = "cudagraphs"
    cfg.structured_kernel_compile_cudagraphs = True
    cfg.structured_kernel_compile_required = "bw_workload_costs"
    cfg.structured_native_main_kernel_require_compiled_segments = True
    cfg.structured_native_main_kernel_required_compile_names = cfg.structured_kernel_compile_required

    env_group = make_structured_env_group(cfg, num_envs=1, backend="sync", mode="eval")
    try:
        env_group.reset_many([int(cfg.seed) * 57])
        env_group.set_tensor_device(torch.device("cuda"))
        env_group.begin_native_main_kernel_rollout(capacity=2, num_envs=1)
        program = env_group.native_rollout_program()
        result = program.replay_step(
            actor_bridge=_FixedNativeActionBridge(
                cfg=cfg,
                device="cuda",
                num_envs=1,
                accel_action=torch.zeros((1, int(cfg.num_uav), 2), dtype=torch.float32, device="cuda"),
                sat_action=torch.zeros((1, int(cfg.num_uav)), dtype=torch.long, device="cuda"),
                bw_action=torch.zeros((1, int(cfg.num_uav), int(cfg.num_gu)), dtype=torch.float32, device="cuda"),
            ),
            deterministic=True,
            rollout_tail=False,
        )
        runtime = env_group.native_rollout_runtime
        assert runtime.main.sat_stage_fields is not None
        assert runtime.main.bw_stage_fields is not None
        assert torch.is_tensor(result.team_rewards)
        assert result.team_rewards.device.type == "cuda"
    finally:
        close_structured_env_group(env_group)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native direct main-kernel validation")
@pytest.mark.parametrize(
    "flag_updates",
    [
        {"energy_enabled": True, "energy_safety_enabled": True, "use_energy_safety_layer": True},
        {"boundary_hard_filter_enabled": True, "boundary_margin": 10.0},
        {"pairwise_hard_filter_enabled": True, "pairwise_hard_distance": 75.0},
        {
            "centroid_cross_anneal_enabled": True,
            "eta_centroid": 1.0,
            "eta_centroid_final": 0.0,
            "eta_centroid_decay_steps": 4,
            "centroid_cross_avoidance_gain": 0.5,
            "avoidance_adaptive_enabled": True,
        },
    ],
)
def test_cuda_native_main_kernel_accel_semantic_flags_stay_on_direct_path(monkeypatch, flag_updates):
    import sagin_marl.env.structured_batch_env_core as core_module

    cfg = _acceptance_cfg()
    cfg.structured_env_tensor_backend = "cuda"
    cfg.avoidance_enabled = True
    cfg.avoidance_eta = 3.5
    cfg.sat_candidate_mode = "elevation"
    for key, value in flag_updates.items():
        setattr(cfg, key, value)

    env_group = make_structured_env_group(cfg, num_envs=1, backend="sync", mode="eval")
    try:
        env_group.reset_many([int(cfg.seed) * 29])
        env_group.set_tensor_device(torch.device("cuda"))
        env_group.begin_native_main_kernel_rollout(capacity=2, num_envs=1)
        program = env_group.native_rollout_program()

        def _unexpected(*_args, **_kwargs):
            raise AssertionError("CUDA native main-kernel direct path must not leave the fused tensor path")

        legacy_helper_names = (
            "_decode_sat_subset_indices_to_selection_matrices",
            "_decode_sat_actions_to_selection_matrices",
        )
        for name in legacy_helper_names:
            assert not hasattr(env_group.batch_core, name)
        for name in ("LocalAccelState", "LocalSatState", "LocalBwState"):
            monkeypatch.setattr(core_module, name, _unexpected)
        for name in ("cpu", "numpy", "item", "tolist"):
            monkeypatch.setattr(torch.Tensor, name, _unexpected)

        result = program.replay_step(
            actor_bridge=_FixedNativeActionBridge(
                cfg=cfg,
                device="cuda",
                num_envs=1,
                accel_action=torch.zeros((1, int(cfg.num_uav), 2), dtype=torch.float32, device="cuda"),
                sat_action=torch.zeros((1, int(cfg.num_uav)), dtype=torch.long, device="cuda"),
                bw_action=torch.zeros((1, int(cfg.num_uav), int(cfg.num_gu)), dtype=torch.float32, device="cuda"),
            ),
            deterministic=True,
            rollout_tail=False,
        )
        assert env_group.native_rollout_runtime.main.sat_stage_fields is not None
        assert torch.is_tensor(result.team_rewards)
        assert result.team_rewards.device.type == "cuda"
    finally:
        close_structured_env_group(env_group)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native direct main-kernel validation")
def test_cuda_native_main_kernel_full_batch_sat_subset_uses_visible_width():
    cfg = _acceptance_cfg()
    cfg.structured_env_tensor_backend = "cuda"
    cfg.avoidance_enabled = True
    cfg.avoidance_eta = 3.5
    cfg.avoidance_adaptive_enabled = False
    cfg.sat_candidate_mode = "elevation"
    cfg.visible_sats_max = int(cfg.sats_obs_max)

    num_envs = 8
    env_group = make_structured_env_group(cfg, num_envs=num_envs, backend="sync", mode="eval")
    try:
        env_group.reset_many([int(cfg.seed) * 23 + idx for idx in range(num_envs)])
        env_group.set_tensor_device(torch.device("cuda"))
        env_group.begin_native_main_kernel_rollout(capacity=2, num_envs=num_envs)
        program = env_group.native_rollout_program()

        accel_actions = [np.zeros((cfg.num_uav, 2), dtype=np.float32) for _ in range(num_envs)]
        sat_subset_indices = [np.zeros((cfg.num_uav,), dtype=np.int64) for _ in range(num_envs)]
        bw_actions = [np.zeros((cfg.num_uav, cfg.num_gu), dtype=np.float32) for _ in range(num_envs)]

        result = program.replay_step(
            actor_bridge=_FixedNativeActionBridge(
                cfg=cfg,
                device="cuda",
                num_envs=num_envs,
                accel_action=torch.as_tensor(np.stack(accel_actions, axis=0), dtype=torch.float32, device="cuda"),
                sat_action=torch.as_tensor(np.stack(sat_subset_indices, axis=0), dtype=torch.long, device="cuda"),
                bw_action=torch.as_tensor(np.stack(bw_actions, axis=0), dtype=torch.float32, device="cuda"),
            ),
            deterministic=True,
            rollout_tail=False,
        )
        sat_obs = env_group.native_rollout_runtime.main.live_sat_obs
        sat_max_select = env_group.native_rollout_runtime.main.sat_max_select
        expected_subset_count = sum(
            math.comb(int(cfg.sats_obs_max), k)
            for k in range(0, min(int(sat_max_select), int(cfg.sats_obs_max)) + 1)
        )
        assert sat_obs.sat_tokens.shape[0] == num_envs * int(cfg.num_uav)
        assert sat_obs.sat_tokens.shape[1] == int(cfg.sats_obs_max)
        assert sat_obs.subset_mask.shape[1] == expected_subset_count
        assert sat_obs.subset_members.shape[1] == expected_subset_count

        bw_obs = env_group.native_rollout_runtime.main.live_bw_obs
        assert bw_obs.bw_valid_mask.shape[0] == num_envs * int(cfg.num_uav)
        assert result.team_rewards.shape == (num_envs,)
        assert result.team_rewards.device.type == "cuda"
    finally:
        close_structured_env_group(env_group)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native network-input contract validation")
def test_cuda_native_multi_env_actor_and_critic_inputs_are_semantic_after_rollout():
    cfg = _acceptance_cfg()
    cfg.structured_env_tensor_backend = "cuda"
    cfg.structured_env_backend = "native"
    cfg.num_uav = 3
    cfg.num_gu = 5
    cfg.num_sat = 8
    cfg.users_obs_max = 5
    cfg.sats_obs_max = 4
    cfg.visible_sats_max = 4
    cfg.sat_num_select = 2
    cfg.T_steps = 6
    cfg.sat_candidate_mode = "elevation"

    num_envs = 4
    horizon = 3
    device = torch.device("cuda")
    env_group = make_structured_env_group(cfg, num_envs=num_envs, backend="sync", mode="eval")
    try:
        env_group.reset_many([int(cfg.seed) * 101 + idx for idx in range(num_envs)])
        env_group.set_tensor_device(device)
        env_group.begin_native_main_kernel_rollout(capacity=horizon, num_envs=num_envs)
        runtime = env_group.native_rollout_runtime
        _poison_native_active_sat_derived_buffers(runtime)

        program = env_group.native_rollout_program()
        bridge = _FixedNativeActionBridge(
            cfg=cfg,
            device=device,
            num_envs=num_envs,
            accel_action=torch.zeros((num_envs, int(cfg.num_uav), 2), dtype=torch.float32, device=device),
            # Subset index 0 is the empty subset; index 1 selects the first visible SAT
            # and exercises the BW selected-SAT actor inputs.
            sat_action=torch.ones((num_envs, int(cfg.num_uav)), dtype=torch.long, device=device),
            bw_action=torch.zeros((num_envs, int(cfg.num_uav), int(cfg.num_gu)), dtype=torch.float32, device=device),
        )
        results = program.replay_horizon(
            actor_bridge_factory=bridge,
            horizon=horizon,
            deterministic=True,
        )
        assert len(results) == horizon

        history = runtime.history
        assert int(history.cursor) == horizon
        _assert_accel_actor_inputs(history.accel_stage)
        _assert_sat_actor_inputs(history.sat_stage, sats_obs_max=int(cfg.sats_obs_max))
        _assert_bw_actor_inputs(history.bw_stage)
        _assert_critic_world_inputs(history.accel_stage)
        _assert_critic_world_inputs(history.sat_stage)
        _assert_critic_world_inputs(history.bw_stage)
        # This zero-action scenario should never create UAV relay backlog.  It
        # is intentionally stride-sensitive: if the native writer uses an actor
        # feature width for critic nodes, neighboring fields leak into these
        # queue slots even though the source stage/runtime queues are zero.
        for stage_name, stage in (("accel", history.accel_stage), ("sat", history.sat_stage), ("bw", history.bw_stage)):
            torch.testing.assert_close(
                stage.world_batch.uav_nodes[..., critic_schema.UAV_QUEUE_STEPS],
                torch.zeros_like(stage.world_batch.uav_nodes[..., critic_schema.UAV_QUEUE_STEPS]),
                msg=f"{stage_name} critic UAV queue steps must stay zero",
            )
            torch.testing.assert_close(
                stage.world_batch.uav_nodes[..., critic_schema.UAV_QUEUE_FILL],
                torch.zeros_like(stage.world_batch.uav_nodes[..., critic_schema.UAV_QUEUE_FILL]),
                msg=f"{stage_name} critic UAV queue fill must stay zero",
            )
        _assert_native_inputs_are_consumable_by_networks(cfg, history, device=device)
        if history.terminal_next_world is not None:
            _assert_finite_and_bounded("critic.terminal_next.uav_nodes", history.terminal_next_world.uav_nodes)
            _assert_finite_and_bounded("critic.terminal_next.uav_sat_edges", history.terminal_next_world.uav_sat_edges)
            _assert_bool_tensor("critic.terminal_next.uav_sat_mask", history.terminal_next_world.uav_sat_mask)

        cache = history.bw_runtime_cache
        assert cache is not None
        for name in ("gain_active", "nu_eff_active", "valid_flag_active"):
            value = getattr(cache, name)
            _assert_finite_and_bounded(f"bw_runtime_cache.{name}", value)
        _assert_unit_interval("bw_runtime_cache.valid_flag_active", cache.valid_flag_active)
        assert bool(((cache.active_sat_ids >= -1) & (cache.active_sat_ids < int(cfg.num_sat))).all().item())
    finally:
        close_structured_env_group(env_group)


def test_native_current_obs_many_reuses_reset_stage_eta_without_advancing_rng(monkeypatch):
    cfg = _acceptance_cfg()
    cfg.fading_enabled = True
    cfg.rician_K = 5.0

    env_group = make_structured_env_group(cfg, num_envs=1, backend="sync", mode="eval")
    try:
        env_group.reset_many([int(cfg.seed) * 17])
        state_before = env_group.batch_core.export_runtime_state_batch(indices=[0])[0]["rng_bit_generator_state"]

        def _unexpected(*_args, **_kwargs):
            raise AssertionError("native current_obs_many() should reuse reset-stage eta cache without recomputing access rates")

        monkeypatch.setattr(SaginParallelEnv, "_compute_access_rates", _unexpected)

        obs_many = env_group.current_obs_many(indices=[0])
        assert len(obs_many) == 1
        assert len(obs_many[0]) == cfg.num_uav

        state_after = env_group.batch_core.export_runtime_state_batch(indices=[0])[0]["rng_bit_generator_state"]
        assert state_after == state_before
    finally:
        close_structured_env_group(env_group)
