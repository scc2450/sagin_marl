from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Mapping, Sequence

import torch

from sagin_marl.env.structured_driver import StructuredBatchStepResult


def _tensor_field_names(value: Any) -> tuple[str, ...]:
    explicit = getattr(value, "_tensor_fields", None)
    if explicit is not None:
        return tuple(str(name) for name in explicit)
    namedtuple_fields = getattr(value, "_fields", None)
    if namedtuple_fields is not None:
        return tuple(str(name) for name in namedtuple_fields)
    if is_dataclass(value):
        return tuple(str(field.name) for field in fields(value))
    raise TypeError("GPU rollout buffers expect a fixed tensor-field object.")


def clone_dataclass_tensors(value: Any) -> Any:
    try:
        field_names = _tensor_field_names(value)
    except TypeError:
        return value
    kwargs = {}
    for field_name in field_names:
        tensor = getattr(value, field_name)
        kwargs[field_name] = tensor.detach().clone() if torch.is_tensor(tensor) else tensor
    return type(value)(**kwargs)


def _materialize_tensor_tree(value: Any, *, device: str | torch.device) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().to(device=device).clone()
    if isinstance(value, dict):
        return {str(key): _materialize_tensor_tree(item, device=device) for key, item in value.items()}
    try:
        field_names = _tensor_field_names(value)
    except TypeError:
        return value
    kwargs = {}
    for field_name in field_names:
        kwargs[field_name] = _materialize_tensor_tree(getattr(value, field_name), device=device)
    return type(value)(**kwargs)


def _is_tensor_dataclass(value: Any) -> bool:
    try:
        field_names = _tensor_field_names(value)
    except TypeError:
        return False
    tensor_count = 0
    for field_name in field_names:
        field_value = getattr(value, field_name)
        if field_value is None:
            continue
        if not torch.is_tensor(field_value):
            return False
        tensor_count += 1
    return tensor_count > 0


class StructuredGpuAccelObsView:
    _tensor_fields = (
        "ego_features",
        "ego_cell",
        "gu_tokens",
        "gu_mask",
        "peer_tokens",
        "peer_mask",
        "sat_tokens",
        "sat_mask",
    )
    __slots__ = _tensor_fields

    def __init__(
        self,
        *,
        ego_features: torch.Tensor,
        ego_cell: torch.Tensor,
        gu_tokens: torch.Tensor,
        gu_mask: torch.Tensor,
        peer_tokens: torch.Tensor,
        peer_mask: torch.Tensor,
        sat_tokens: torch.Tensor,
        sat_mask: torch.Tensor,
    ) -> None:
        self.ego_features = ego_features
        self.ego_cell = ego_cell
        self.gu_tokens = gu_tokens
        self.gu_mask = gu_mask
        self.peer_tokens = peer_tokens
        self.peer_mask = peer_mask
        self.sat_tokens = sat_tokens
        self.sat_mask = sat_mask


class StructuredGpuSatObsView:
    _tensor_fields = (
        "ego_features",
        "demand_features",
        "role_features",
        "sat_tokens",
        "sat_mask",
        "sat_valid_mask",
        "candidate_sat_ids",
        "subset_mask",
        "subset_members",
    )
    __slots__ = _tensor_fields

    def __init__(
        self,
        *,
        ego_features: torch.Tensor,
        demand_features: torch.Tensor,
        role_features: torch.Tensor,
        sat_tokens: torch.Tensor,
        sat_mask: torch.Tensor,
        sat_valid_mask: torch.Tensor,
        candidate_sat_ids: torch.Tensor,
        subset_mask: torch.Tensor,
        subset_members: torch.Tensor,
    ) -> None:
        self.ego_features = ego_features
        self.demand_features = demand_features
        self.role_features = role_features
        self.sat_tokens = sat_tokens
        self.sat_mask = sat_mask
        self.sat_valid_mask = sat_valid_mask
        self.candidate_sat_ids = candidate_sat_ids
        self.subset_mask = subset_mask
        self.subset_members = subset_members


class StructuredGpuBwObsView:
    _tensor_fields = (
        "ego_features",
        "selected_sat_tokens",
        "selected_sat_mask",
        "gu_tokens",
        "gu_mask",
        "bw_valid_mask",
    )
    __slots__ = _tensor_fields

    def __init__(
        self,
        *,
        ego_features: torch.Tensor,
        selected_sat_tokens: torch.Tensor,
        selected_sat_mask: torch.Tensor,
        gu_tokens: torch.Tensor,
        gu_mask: torch.Tensor,
        bw_valid_mask: torch.Tensor,
    ) -> None:
        self.ego_features = ego_features
        self.selected_sat_tokens = selected_sat_tokens
        self.selected_sat_mask = selected_sat_mask
        self.gu_tokens = gu_tokens
        self.gu_mask = gu_mask
        self.bw_valid_mask = bw_valid_mask


_STAGE_REGISTER_VALUE_KEYS: tuple[str, ...] = (
    "stage_id",
    "effective_b_backhaul_per_sat",
    "uav_pos",
    "uav_vel",
    "uav_energy",
    "uav_queue",
    "gu_pos",
    "gu_queue",
    "sat_queue",
    "sat_loads",
    "sat_pos",
    "sat_vel",
    "assoc",
    "prev_association",
    "candidate_indices",
    "candidate_mask",
    "bw_valid_mask",
    "sat_selection_matrix",
    "candidate_flag",
    "bw_valid_flag",
    "prev_assoc_flag",
    "eta_ref_feature",
    "eta_slots",
    "gu_proxy_features",
    "uav_assoc_uav_cost",
    "sat_cost_norm",
    "access_gain_matrix",
    "visible_ids",
    "visible_mask",
    "visible_flag_all",
    "elevation_matrix",
    "uav_ecef_all",
    "uav_vel_ecef_all",
    "active_sat_ids",
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
    "us_rel_pos_all",
    "us_rel_vel_all",
    "us_gain_all",
    "us_nu_eff_all",
    "us_valid_flag_all",
    "us_sat_queue_all",
)
_STAGE_REGISTER_STORAGE_KEYS: tuple[str, ...] = tuple(
    key for key in _STAGE_REGISTER_VALUE_KEYS if key != "stage_id"
)


@dataclass
class StructuredGpuStepResultBuffers:
    batch_result: StructuredBatchStepResult | None = None
    step_result_views: list[StructuredBatchStepResult] = field(default_factory=list)
    horizon_num_steps: int = 0
    team_rewards: torch.Tensor | None = None
    terminated: torch.Tensor | None = None
    truncated: torch.Tensor | None = None
    danger_imitation_target: torch.Tensor | None = None
    danger_imitation_mask: torch.Tensor | None = None
    bw_access_rewards: torch.Tensor | None = None
    bw_weighted_workload_delta_rewards: torch.Tensor | None = None
    bw_weighted_workload_level_rewards: torch.Tensor | None = None
    bw_gu_queue_level_rewards: torch.Tensor | None = None
    bw_system_queue_level_rewards: torch.Tensor | None = None
    bw_gu_service_queue_rewards: torch.Tensor | None = None
    bw_flow_proxy_scores: torch.Tensor | None = None
    bw_flow_proxy_mask: torch.Tensor | None = None
    bw_flow_proxy_deltas: torch.Tensor | None = None
    reward_parts: "StructuredGpuRewardPartBuffers | None" = None
    reward_part_tensors: dict[str, torch.Tensor] | None = None
    reward_mode_active: str | None = None


@dataclass
class StructuredGpuRewardPartBuffers:
    service_ratio: torch.Tensor
    drop_ratio: torch.Tensor
    arrival_ref: torch.Tensor
    b_pre_steps: torch.Tensor
    x_acc: torch.Tensor
    x_rel: torch.Tensor
    g_pre: torch.Tensor
    d_pre: torch.Tensor
    processed_ratio_eval: torch.Tensor
    drop_ratio_eval: torch.Tensor
    pre_backlog_steps_eval: torch.Tensor
    sat_overlap_eval: torch.Tensor
    D_sys_report: torch.Tensor
    drop_sum: torch.Tensor
    gu_queue_sum: torch.Tensor
    uav_queue_sum: torch.Tensor
    sat_queue_sum: torch.Tensor
    queue_total_sum: torch.Tensor
    drop_sum_active: torch.Tensor
    expire_sum: torch.Tensor
    gu_drop_sum: torch.Tensor
    uav_drop_sum: torch.Tensor
    sat_drop_sum: torch.Tensor
    arrival_sum: torch.Tensor
    outflow_sum: torch.Tensor
    backhaul_sum: torch.Tensor
    sat_processed_sum: torch.Tensor
    collision_event: torch.Tensor
    overflow_risk_mean: torch.Tensor
    downstream_pressure_mean: torch.Tensor
    service_gap_mean: torch.Tensor
    service_gap_risk_mean: torch.Tensor
    bw_weighted_workload_delta_reward: torch.Tensor
    bw_weighted_workload_level_reward: torch.Tensor
    bw_gu_queue_level_reward: torch.Tensor
    bw_system_queue_level_reward: torch.Tensor
    bw_gu_service_queue_reward: torch.Tensor
    intervention_norm: torch.Tensor
    intervention_rate: torch.Tensor
    intervention_norm_top1: torch.Tensor
    danger_imitation_active_rate: torch.Tensor
    close_risk: torch.Tensor
    term_close_risk: torch.Tensor
    reward_raw: torch.Tensor


_BW_REWARD_PART_KEYS: tuple[str, ...] = (
    "service_ratio",
    "drop_ratio",
    "arrival_ref",
    "b_pre_steps",
    "x_acc",
    "x_rel",
    "g_pre",
    "d_pre",
    "processed_ratio_eval",
    "drop_ratio_eval",
    "pre_backlog_steps_eval",
    "sat_overlap_eval",
    "D_sys_report",
    "drop_sum",
    "gu_queue_sum",
    "uav_queue_sum",
    "sat_queue_sum",
    "queue_total_sum",
    "drop_sum_active",
    "expire_sum",
    "gu_drop_sum",
    "uav_drop_sum",
    "sat_drop_sum",
    "arrival_sum",
    "outflow_sum",
    "backhaul_sum",
    "sat_processed_sum",
    "collision_event",
    "overflow_risk_mean",
    "downstream_pressure_mean",
    "service_gap_mean",
    "service_gap_risk_mean",
    "bw_weighted_workload_delta_reward",
    "bw_weighted_workload_level_reward",
    "bw_gu_queue_level_reward",
    "bw_system_queue_level_reward",
    "bw_gu_service_queue_reward",
    "intervention_norm",
    "intervention_rate",
    "intervention_norm_top1",
    "danger_imitation_active_rate",
    "close_risk",
    "term_close_risk",
    "reward_raw",
)


@dataclass
class StructuredGpuRandomTapeBuffers:
    """Runtime-owned tensor random inputs consumed by native live kernels."""

    reset_gu_pos: torch.Tensor | None = None
    reset_uav_pos: torch.Tensor | None = None
    reset_uav_vel: torch.Tensor | None = None
    reset_gu_cluster_centers: torch.Tensor | None = None
    reset_gu_cluster_counts: torch.Tensor | None = None
    reset_arrival_base_scale: torch.Tensor | None = None
    reset_deadline_steps: torch.Tensor | None = None
    reset_doppler_residual: torch.Tensor | None = None
    reset_effective_arrival_rate: torch.Tensor | None = None
    reset_gu_pos_rollout_tape: torch.Tensor | None = None
    reset_uav_pos_rollout_tape: torch.Tensor | None = None
    reset_uav_vel_rollout_tape: torch.Tensor | None = None
    reset_gu_cluster_centers_rollout_tape: torch.Tensor | None = None
    reset_gu_cluster_counts_rollout_tape: torch.Tensor | None = None
    reset_gu_queue_rollout_tape: torch.Tensor | None = None
    reset_uav_queue_rollout_tape: torch.Tensor | None = None
    reset_sat_queue_rollout_tape: torch.Tensor | None = None
    reset_arrival_base_scale_rollout_tape: torch.Tensor | None = None
    reset_deadline_steps_rollout_tape: torch.Tensor | None = None
    reset_doppler_residual_rollout_tape: torch.Tensor | None = None
    reset_effective_arrival_rate_rollout_tape: torch.Tensor | None = None
    reset_arrival_rate_vec_rollout_tape: torch.Tensor | None = None
    reset_arrival_ref_rollout_tape: torch.Tensor | None = None
    reset_episode_idx_rollout_tape: torch.Tensor | None = None
    reset_hotspot_active_idx_rollout_tape: torch.Tensor | None = None
    reset_hotspot_subset_count_rollout_tape: torch.Tensor | None = None
    reset_hotspot_member_mask_rollout_tape: torch.Tensor | None = None
    arrivals: torch.Tensor | None = None
    arrival_rates: torch.Tensor | None = None
    fading_gain: torch.Tensor | None = None
    doppler_noise: torch.Tensor | None = None
    arrival_rollout_tape: torch.Tensor | None = None
    arrival_rate_rollout_tape: torch.Tensor | None = None
    hotspot_active_after_rollout_tape: torch.Tensor | None = None
    reset_followup_arrival_rollout_tape: torch.Tensor | None = None
    reset_followup_arrival_rate_rollout_tape: torch.Tensor | None = None
    reset_followup_hotspot_active_after_rollout_tape: torch.Tensor | None = None
    hotspot_active_rollout_tape: torch.Tensor | None = None
    hotspot_mask_rollout_tape: torch.Tensor | None = None
    fading_gain_rollout_tape: torch.Tensor | None = None
    doppler_noise_rollout_tape: torch.Tensor | None = None
    step_tensor: torch.Tensor | None = None
    reset_count: torch.Tensor | None = None
    step: int = 0


class StructuredGpuNativeRuntimeStepProgram:
    """Single object ABI for native GPU live rollout execution.

    The env core owns the actual fused tensor kernels. This program is the only
    object allowed to drive them from the rollout loop, so callers cannot stitch
    phase callables or fall back to per-env stage objects.
    """

    REQUIRED_EXECUTOR_METHODS: tuple[str, ...] = (
        "_runtime_step_begin_accel_obs",
        "_runtime_step_publish_sat_obs",
        "_runtime_step_publish_bw_obs",
        "_runtime_step_finish_bw",
    )

    def __init__(
        self,
        *,
        runtime: "StructuredGpuRolloutRuntime",
        executor: Any,
        num_envs: int,
        fixed_visible_sat_width: int | None = None,
    ) -> None:
        if runtime is None:
            raise RuntimeError("native GPU runtime step program requires a persistent runtime.")
        if executor is None:
            raise RuntimeError("native GPU runtime step program requires an env tensor executor.")
        missing = [name for name in self.REQUIRED_EXECUTOR_METHODS if not callable(getattr(executor, name, None))]
        if missing:
            raise RuntimeError(
                "native GPU runtime step program executor is missing required tensor methods: "
                + ", ".join(missing)
            )
        self.runtime = runtime
        self.executor = executor
        self.num_envs = max(int(num_envs), 0)
        self.fixed_visible_sat_width = (
            None if fixed_visible_sat_width is None else max(int(fixed_visible_sat_width), 0)
        )
        self._horizon_started = False

    def _ensure_horizon_started(self) -> None:
        if self._horizon_started:
            return
        begin_horizon = getattr(self.executor, "_runtime_begin_horizon", None)
        if not callable(begin_horizon):
            raise RuntimeError("native GPU rollout program executor cannot begin a native horizon.")
        capacity = int(getattr(self.runtime.history, "capacity", 0) or 0)
        if capacity <= 0:
            raise RuntimeError("native GPU rollout program requires preallocated rollout history capacity.")
        begin_horizon(num_steps=capacity)
        self._horizon_started = True

    def _begin(self) -> Any:
        select_storage = getattr(self.executor, "_runtime_step_select_rollout_storage", None)
        if callable(select_storage):
            select_storage(num_envs=int(self.num_envs))
        return self.executor._runtime_step_begin_accel_obs()

    def _current_or_begin_accel_obs(self) -> Any:
        select_storage = getattr(self.executor, "_runtime_step_select_rollout_storage", None)
        if callable(select_storage):
            select_storage(num_envs=int(self.num_envs))
        main = self.runtime.main
        live_buffers = main.accel_live_obs_buffers
        stage_buffers = main.accel_stage_field_buffers
        if live_buffers is None or stage_buffers is None:
            raise RuntimeError("native GPU rollout program requires initialized accel live/stage buffers.")
        active_idx = int(main.accel_active_idx)
        if active_idx not in {0, 1}:
            raise RuntimeError("native accel active index must be 0 or 1.")
        return live_buffers[active_idx]

    def _after_accel_action(
        self,
        *,
        max_visible: int | None,
        **kwargs: Any,
    ) -> tuple[Any, int]:
        return self.executor._runtime_step_publish_sat_obs(
            max_visible=max_visible,
            **kwargs,
        )

    def _after_sat_action(
        self,
        *,
        max_visible: int | None,
        **kwargs: Any,
    ) -> Any:
        return self.executor._runtime_step_publish_bw_obs(
            max_visible=max_visible,
            **kwargs,
        )

    def _after_bw_action(self, **kwargs: Any) -> StructuredBatchStepResult:
        return self.executor._runtime_step_finish_bw(**kwargs)

    def _apply_bw_macro_live(self, **kwargs: Any) -> None:
        apply_fn = getattr(self.executor, "_runtime_step_apply_bw_macro_live", None)
        if callable(apply_fn):
            apply_fn(**kwargs)

    def replay_step(
        self,
        *,
        actor_bridge: Any,
        deterministic: bool = False,
        rollout_tail: bool,
    ) -> StructuredBatchStepResult:
        """Replay one actor/env step through the fixed native live program.

        Actor decisions remain explicit dependency boundaries. The env side is
        driven only by this fixed program; callers cannot manually stitch the
        stage callables in the official live path.
        """
        if self.num_envs <= 0:
            raise RuntimeError("native GPU rollout program cannot replay a step for zero envs.")
        self._ensure_horizon_started()
        if hasattr(actor_bridge, "begin_step"):
            actor_bridge.begin_step(deterministic=deterministic)

        accel_obs = self._current_or_begin_accel_obs()
        if accel_obs is None:
            raise RuntimeError("native GPU rollout program did not publish accel obs.")
        actor_bridge.write_accel_action(
            accel_obs,
            runtime=self.runtime,
            num_envs=self.num_envs,
            deterministic=deterministic,
        )

        sat_obs, sat_max_select = self._after_accel_action(
            max_visible=self.fixed_visible_sat_width,
        )
        if sat_obs is None:
            raise RuntimeError("native GPU rollout program did not publish SAT obs.")
        actor_bridge.write_sat_action(
            sat_obs,
            runtime=self.runtime,
            num_envs=self.num_envs,
            sat_max_select=int(sat_max_select),
            deterministic=deterministic,
        )

        bw_obs = self._after_sat_action(
            max_visible=self.fixed_visible_sat_width,
        )
        if bw_obs is None:
            raise RuntimeError("native GPU rollout program did not publish BW obs.")
        actor_bridge.write_bw_action(
            bw_obs,
            runtime=self.runtime,
            num_envs=self.num_envs,
            deterministic=deterministic,
        )
        self._apply_bw_macro_live(
            max_visible=self.fixed_visible_sat_width,
            rollout_tail=bool(rollout_tail),
        )

        finish_kwargs = {
            "max_visible": self.fixed_visible_sat_width,
            "rollout_tail": bool(rollout_tail),
        }
        step_result = self._after_bw_action(**finish_kwargs)
        if not isinstance(step_result, StructuredBatchStepResult):
            raise RuntimeError("native GPU rollout program requires StructuredBatchStepResult.")
        return step_result


class StructuredGpuNativeRolloutProgram:
    """Fixed GPU rollout program boundary for native live execution.

    The program owns the ordering between env kernels and actor decision points.
    Callers provide an actor bridge that reads obs buffers and writes action
    buffers; callers must not manually stitch env stage callables.
    """

    def __init__(
        self,
        *,
        step_program: StructuredGpuNativeRuntimeStepProgram,
    ) -> None:
        runtime = getattr(step_program, "runtime", None)
        if runtime is None:
            raise RuntimeError("native GPU rollout program requires a persistent runtime.")
        if not isinstance(step_program, StructuredGpuNativeRuntimeStepProgram):
            raise RuntimeError("native GPU rollout program requires a StructuredGpuNativeRuntimeStepProgram.")
        self.runtime = runtime
        self._step_program = step_program
        self.num_envs = int(step_program.num_envs)
        self.fixed_visible_sat_width = step_program.fixed_visible_sat_width

    def replay_step(
        self,
        *,
        actor_bridge: Any,
        deterministic: bool = False,
        rollout_tail: bool,
    ) -> StructuredBatchStepResult:
        return self._step_program.replay_step(
            actor_bridge=actor_bridge,
            deterministic=deterministic,
            rollout_tail=rollout_tail,
        )

    def replay_horizon(
        self,
        *,
        actor_bridge_factory: Any,
        horizon: int,
        deterministic: bool = False,
        step_callback: Any | None = None,
    ) -> list[StructuredBatchStepResult]:
        steps = max(int(horizon), 0)
        results: list[StructuredBatchStepResult] = []
        begin_horizon = getattr(self._step_program.executor, "_runtime_begin_horizon", None)
        if callable(begin_horizon):
            begin_horizon(num_steps=steps)
            self._step_program._horizon_started = True
        if hasattr(actor_bridge_factory, "begin_horizon"):
            actor_bridge_factory.begin_horizon(
                horizon=steps,
                runtime=self.runtime,
                deterministic=deterministic,
            )
        for step_index in range(steps):
            actor_bridge = (
                actor_bridge_factory(step_index=step_index, runtime=self.runtime)
                if callable(actor_bridge_factory)
                else actor_bridge_factory
            )
            step_result = self.replay_step(
                actor_bridge=actor_bridge,
                deterministic=deterministic,
                rollout_tail=bool(step_index + 1 >= steps),
            )
            step_result_view = (
                self.runtime.result.step_result_views[step_index]
                if step_index < len(self.runtime.result.step_result_views)
                else getattr(self.runtime.main, "step_result_view", None)
            )
            if not isinstance(step_result_view, StructuredBatchStepResult):
                raise RuntimeError("native GPU rollout horizon requires a runtime-history-backed step result view.")
            results.append(step_result_view)
            if callable(step_callback):
                step_callback(
                    step_index=step_index,
                    step_result=step_result,
                    actor_bridge=actor_bridge,
                    runtime=self.runtime,
                )
        if hasattr(actor_bridge_factory, "end_horizon"):
            actor_bridge_factory.end_horizon(results=results, runtime=self.runtime)
        return results

@dataclass
class StructuredGpuNativeMainKernelBuffers:
    """Long-lived per-step views for the native GPU executor.

    MAPPO/eval/formal consume the local obs/action/result tensors published in
    this runtime. Stage buffers are fixed-layout tensor snapshots owned by this
    runtime; legacy stage registers are not part of the live runtime contract.
    """

    indices: tuple[int, ...] = ()
    selected_env_mapping: torch.Tensor | None = None
    accel_stage_fields: Any | None = None
    accel_stage_field_buffers: tuple[Any, Any] | None = None
    accel_active_idx: int = 0
    accel_live_obs_buffers: tuple[StructuredGpuAccelObsView, StructuredGpuAccelObsView] | None = None
    live_sat_obs: StructuredGpuSatObsView | None = None
    live_bw_obs: StructuredGpuBwObsView | None = None
    live_accel_action: torch.Tensor | None = None
    live_accel_latent_action: torch.Tensor | None = None
    live_sat_subset_index: torch.Tensor | None = None
    live_sat_action_indices: torch.Tensor | None = None
    live_bw_action: torch.Tensor | None = None
    live_bw_ref_action: torch.Tensor | None = None
    live_bw_flow_proxy_override_action: torch.Tensor | None = None
    live_accel_old_logprob: torch.Tensor | None = None
    live_sat_old_logprobs_per_agent: torch.Tensor | None = None
    live_sat_entropy_per_agent: torch.Tensor | None = None
    live_bw_old_logprob: torch.Tensor | None = None
    live_bw_old_logprobs_per_agent: torch.Tensor | None = None
    live_bw_entropy_per_agent: torch.Tensor | None = None
    live_bw_logprob_raw_per_agent: torch.Tensor | None = None
    live_bw_entropy_raw_per_agent: torch.Tensor | None = None
    live_bw_tau: torch.Tensor | None = None
    live_bw_kappa: torch.Tensor | None = None
    live_bw_valid_count: torch.Tensor | None = None
    live_bw_latent_count: torch.Tensor | None = None
    accel_actor_source_mode_code: int = 0
    sat_actor_source_mode_code: int = 0
    bw_actor_source_mode_code: int = 0
    flow_proxy_base_action_mode_code: int = 0
    native_cuda_abi: Any | None = None
    native_cuda_marker: torch.Tensor | None = None
    native_cuda_empty_float: torch.Tensor | None = None
    native_cuda_empty_long: torch.Tensor | None = None
    native_cuda_empty_bool: torch.Tensor | None = None
    native_cuda_empty_int: torch.Tensor | None = None
    current_history_slot: int = 0
    sat_stage_fields: Any | None = None
    bw_stage_fields: Any | None = None
    accel_stage: Any | None = None
    sat_stage: Any | None = None
    bw_stage: Any | None = None
    bw_candidate_indices: torch.Tensor | None = None
    bw_valid_mask: torch.Tensor | None = None
    bw_assoc: torch.Tensor | None = None
    bw_prev_association: torch.Tensor | None = None
    bw_candidate_mask: torch.Tensor | None = None
    bw_access_gain_matrix: torch.Tensor | None = None
    bw_sat_selection_matrix: torch.Tensor | None = None
    bw_active_sat_ids: torch.Tensor | None = None
    bw_gain_active: torch.Tensor | None = None
    bw_nu_eff_active: torch.Tensor | None = None
    bw_valid_flag_active: torch.Tensor | None = None
    bw_sat_pos: torch.Tensor | None = None
    bw_uav_ecef: torch.Tensor | None = None
    bw_uav_pos: torch.Tensor | None = None
    bw_uav_vel: torch.Tensor | None = None
    bw_gu_pos: torch.Tensor | None = None
    bw_sat_compute_rates: torch.Tensor | None = None
    bw_direct_input_fields: Any | None = None
    bw_reward_part_out: Any | None = None
    bw_state_out: Any | None = None
    accel_history_out: Any | None = None
    sat_history_out: Any | None = None
    bw_history_out: Any | None = None
    next_accel_history_world_out: Any | None = None
    terminal_next_history_world_out: Any | None = None
    terminal_next_history_mask_out: torch.Tensor | None = None
    sat_subset_members_base: torch.Tensor | None = None
    sat_subset_sizes: torch.Tensor | None = None
    sat_visible_width: int = 0
    bw_action_shape: tuple[int, ...] = ()
    bw_reward_mode_active: str = "dense"
    bw_flow_proxy_enabled: bool = False
    bw_flow_proxy_reward_mode_code: int = 0
    bw_fading_enabled: bool = False
    bw_uav_orbit_radius: float = 0.0
    bw_uav_orbit_radius_sq: float = 0.0
    bw_sat_orbit_radius_sq: float = 0.0
    bw_backhaul_gain_const: float = 0.0
    bw_effective_b_backhaul_per_sat: float = 0.0
    bw_inv_a_max: float = 1.0
    bw_doppler_cap: float = 0.0
    bw_doppler_rho: float = 0.0
    bw_doppler_sigma: float = 0.0
    bw_doppler_noise_zero: torch.Tensor | None = None
    bw_fading_gain_unity: torch.Tensor | None = None
    bw_link_transition: Any | None = None
    bw_link_transition_override: Any | None = None
    bw_link_transition_override_active: torch.Tensor | None = None
    sat_max_select: int = 0
    step_result_view: StructuredBatchStepResult | None = None
    copy_graph_outputs: bool = False
    num_envs: int = 0
    num_uav: int = 0
    num_gu: int = 0
    num_sat: int = 0
    users_obs_max: int = 0
    sats_obs_max: int = 0
    visible_sats_max: int = 0
    sat_num_select: int = 0
    accel_direct_supported: bool = False
    accel_offdiag_mask: torch.Tensor | None = None
    uav_pair_upper_mask: torch.Tensor | None = None
    accel_uav_index_order: torch.Tensor | None = None
    accel_neighbor_indices: torch.Tensor | None = None
    candidate_slot_ids: torch.Tensor | None = None
    candidate_env_ids: torch.Tensor | None = None
    candidate_gu_ids: torch.Tensor | None = None
    candidate_uav_ids: torch.Tensor | None = None
    sat_all_ids: torch.Tensor | None = None
    sat_bw_empty_gu_proxy: torch.Tensor | None = None
    base_result: "StructuredGpuStepResultBuffers | None" = None


@dataclass
class StructuredGpuRolloutTrainingStageBuffers:
    world_batch: Any | None = None
    actions: torch.Tensor | None = None
    latent_actions: torch.Tensor | None = None
    old_logprobs: torch.Tensor | None = None
    values: torch.Tensor | None = None


@dataclass
class StructuredGpuRolloutAccelTrainingStageBuffers(StructuredGpuRolloutTrainingStageBuffers):
    ego_features: torch.Tensor | None = None
    ego_cell: torch.Tensor | None = None
    gu_tokens: torch.Tensor | None = None
    gu_mask: torch.Tensor | None = None
    peer_tokens: torch.Tensor | None = None
    peer_mask: torch.Tensor | None = None
    sat_tokens: torch.Tensor | None = None
    sat_mask: torch.Tensor | None = None
    danger_imitation_targets: torch.Tensor | None = None
    danger_imitation_masks: torch.Tensor | None = None


@dataclass
class StructuredGpuRolloutSatTrainingStageBuffers(StructuredGpuRolloutTrainingStageBuffers):
    ego_features: torch.Tensor | None = None
    demand_features: torch.Tensor | None = None
    role_features: torch.Tensor | None = None
    sat_tokens: torch.Tensor | None = None
    sat_mask: torch.Tensor | None = None
    sat_valid_mask: torch.Tensor | None = None
    candidate_sat_ids: torch.Tensor | None = None
    subset_mask: torch.Tensor | None = None
    subset_members: torch.Tensor | None = None
    action_indices: torch.Tensor | None = None
    old_logprobs_per_agent: torch.Tensor | None = None


@dataclass
class StructuredGpuRolloutBwTrainingStageBuffers(StructuredGpuRolloutTrainingStageBuffers):
    ego_features: torch.Tensor | None = None
    selected_sat_tokens: torch.Tensor | None = None
    selected_sat_mask: torch.Tensor | None = None
    gu_tokens: torch.Tensor | None = None
    gu_mask: torch.Tensor | None = None
    bw_valid_mask: torch.Tensor | None = None
    rewards: torch.Tensor | None = None
    bw_access_rewards: torch.Tensor | None = None
    bw_weighted_workload_delta_rewards: torch.Tensor | None = None
    bw_weighted_workload_level_rewards: torch.Tensor | None = None
    bw_gu_queue_level_rewards: torch.Tensor | None = None
    bw_system_queue_level_rewards: torch.Tensor | None = None
    bw_gu_service_queue_rewards: torch.Tensor | None = None
    bw_flow_proxy_scores: torch.Tensor | None = None
    bw_flow_proxy_masks: torch.Tensor | None = None
    bw_flow_proxy_deltas: torch.Tensor | None = None
    bw_ref_actions: torch.Tensor | None = None
    bw_old_logprobs_per_agent: torch.Tensor | None = None
    bw_entropy_per_agent: torch.Tensor | None = None
    bw_logprob_raw_per_agent: torch.Tensor | None = None
    bw_entropy_raw_per_agent: torch.Tensor | None = None
    bw_tau: torch.Tensor | None = None
    bw_kappa: torch.Tensor | None = None
    bw_valid_count: torch.Tensor | None = None
    bw_latent_count: torch.Tensor | None = None
    reward_parts: StructuredGpuRewardPartBuffers | None = None
    reward_part_tensors: dict[str, torch.Tensor] | None = None


@dataclass
class StructuredGpuBwRuntimeCacheBuffers:
    candidate_indices: torch.Tensor | None = None
    valid_mask: torch.Tensor | None = None
    assoc: torch.Tensor | None = None
    prev_association: torch.Tensor | None = None
    candidate_mask: torch.Tensor | None = None
    access_gain_matrix: torch.Tensor | None = None
    sat_selection_matrix: torch.Tensor | None = None
    active_sat_ids: torch.Tensor | None = None
    gain_active: torch.Tensor | None = None
    nu_eff_active: torch.Tensor | None = None
    valid_flag_active: torch.Tensor | None = None
    sat_pos: torch.Tensor | None = None
    uav_ecef: torch.Tensor | None = None
    uav_pos: torch.Tensor | None = None
    uav_vel: torch.Tensor | None = None
    gu_pos: torch.Tensor | None = None


@dataclass
class StructuredGpuRolloutTrainingRingBuffers:
    capacity: int = 0
    cursor: int = 0
    num_envs: int = 0
    accel_stage: StructuredGpuRolloutAccelTrainingStageBuffers = field(
        default_factory=StructuredGpuRolloutAccelTrainingStageBuffers
    )
    sat_stage: StructuredGpuRolloutSatTrainingStageBuffers = field(
        default_factory=StructuredGpuRolloutSatTrainingStageBuffers
    )
    bw_stage: StructuredGpuRolloutBwTrainingStageBuffers = field(
        default_factory=StructuredGpuRolloutBwTrainingStageBuffers
    )
    terminal_next_world: Any | None = None
    accel_runtime_state: Any | None = None
    accel_runtime_stage: Any | None = None
    sat_runtime_state: Any | None = None
    sat_runtime_stage: Any | None = None
    bw_runtime_state: Any | None = None
    bw_runtime_cache: StructuredGpuBwRuntimeCacheBuffers | None = None
    bw_runtime_stage: Any | None = None
    terminal_next_world_mask: torch.Tensor | None = None
    terminated: torch.Tensor | None = None
    truncated: torch.Tensor | None = None


def _empty_training_like_dataclass(value: Any, capacity: int) -> Any:
    kwargs = {}
    for field_name in _tensor_field_names(value):
        tensor = getattr(value, field_name)
        if tensor is None:
            kwargs[field_name] = None
            continue
        if not torch.is_tensor(tensor):
            raise TypeError(f"GPU rollout history field {field_name!r} is not a tensor.")
        kwargs[field_name] = tensor.new_empty((int(capacity) * int(tensor.shape[0]),) + tuple(tensor.shape[1:]))
    return type(value)(**kwargs)


def _training_dataclass_matches(target: Any, source: Any, capacity: int) -> bool:
    if target is None or type(target) is not type(source):
        return False
    for field_name in _tensor_field_names(source):
        src = getattr(source, field_name)
        dst = getattr(target, field_name)
        if src is None:
            if dst is not None:
                return False
            continue
        if (
            not torch.is_tensor(src)
            or not torch.is_tensor(dst)
            or tuple(dst.shape) != (int(capacity) * int(src.shape[0]),) + tuple(src.shape[1:])
            or dst.dtype != src.dtype
            or dst.device != src.device
        ):
            return False
    return True


def _require_training_dataclass_buffer(target: Any | None, source: Any, *, capacity: int, field_name: str) -> Any:
    if not _training_dataclass_matches(target, source, capacity):
        raise RuntimeError(
            f"{field_name} training buffer was not preallocated for the native main-kernel ABI."
        )
    return target


class StructuredGpuRolloutRuntime:
    """Persistent tensor rollout buffers owned by the native batch environment."""

    def __init__(self, *, device: torch.device) -> None:
        self.device = torch.device(device)
        self.result = StructuredGpuStepResultBuffers()
        self.random = StructuredGpuRandomTapeBuffers()
        self.main = StructuredGpuNativeMainKernelBuffers()
        self.rollout_history = StructuredGpuRolloutTrainingRingBuffers()
        self.history = self.rollout_history
        self.history_storage_kind = "rollout"

    def activate_rollout_history(self) -> bool:
        changed = self.history is not self.rollout_history
        self.history = self.rollout_history
        self.history_storage_kind = "rollout"
        return bool(changed)

    def clear_main_kernel(self) -> None:
        self.main = StructuredGpuNativeMainKernelBuffers()
        self.random.step = 0
        if torch.is_tensor(self.random.step_tensor):
            self.random.step_tensor.zero_()
        if torch.is_tensor(self.random.reset_count):
            self.random.reset_count.zero_()

    def begin_main_kernel_rollout(self) -> None:
        self.random.step = 0
        if torch.is_tensor(self.random.step_tensor):
            self.random.step_tensor.zero_()
        if torch.is_tensor(self.random.reset_count):
            self.random.reset_count.zero_()
        self.main.step_result_view = None
        self.restore_native_main_kernel_base_views()

    def advance_main_kernel_rollout_step(self) -> None:
        self.random.step = int(self.random.step) + 1
        if torch.is_tensor(self.random.step_tensor):
            self.random.step_tensor.fill_(int(self.random.step))

    def _copy_graph_outputs_enabled(self, cfg: Any | None = None) -> bool:
        if cfg is not None and bool(getattr(cfg, "structured_kernel_compile_cudagraphs", False)):
            self.main.copy_graph_outputs = True
        return bool(self.main.copy_graph_outputs)

    def _require_runtime_tensor(
        self,
        value: torch.Tensor,
        *,
        field_name: str,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if not torch.is_tensor(value):
            raise TypeError(f"native GPU runtime field {field_name!r} must be a tensor.")
        value_device = torch.device(value.device)
        runtime_device = torch.device(self.device)
        same_device = value_device == runtime_device
        if not same_device and value_device.type == runtime_device.type == "cuda":
            same_device = runtime_device.index is None or value_device.index in {None, runtime_device.index}
        if not same_device:
            raise RuntimeError(
                f"native GPU runtime field {field_name!r} is on {value.device}, expected {self.device}; "
                "official native main-kernel live path must write device-local tensors instead of relying on runtime migration."
            )
        if dtype is not None and value.dtype != dtype:
            raise RuntimeError(
                f"native GPU runtime field {field_name!r} has dtype {value.dtype}, expected {dtype}; "
                "official native main-kernel live path must normalize dtype before publishing the buffer."
            )
        return value

    @staticmethod
    def _copy_tensor_buffer(current: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
        if (
            current is None
            or tuple(current.shape) != tuple(value.shape)
            or current.dtype != value.dtype
            or current.device != value.device
        ):
            current = torch.empty_like(value)
        current.copy_(value)
        return current

    def write_random_arrival_tape(
        self,
        *,
        arrivals: torch.Tensor,
        rates: torch.Tensor,
    ) -> StructuredGpuRandomTapeBuffers:
        arrivals = self._require_runtime_tensor(arrivals.detach(), field_name="random.arrivals", dtype=torch.float32)
        rates = self._require_runtime_tensor(rates.detach(), field_name="random.arrival_rates", dtype=torch.float32)
        if self._copy_graph_outputs_enabled():
            self.random.arrivals = arrivals
            self.random.arrival_rates = rates
        else:
            self.random.arrivals = self._copy_tensor_buffer(self.random.arrivals, arrivals)
            self.random.arrival_rates = self._copy_tensor_buffer(self.random.arrival_rates, rates)
        return self.random

    def write_random_reset_tape(
        self,
        *,
        gu_pos: torch.Tensor,
        uav_pos: torch.Tensor,
        uav_vel: torch.Tensor,
        gu_cluster_centers: torch.Tensor,
        gu_cluster_counts: torch.Tensor,
        arrival_base_scale: torch.Tensor,
        deadline_steps: torch.Tensor,
        doppler_residual: torch.Tensor,
        effective_arrival_rate: torch.Tensor,
    ) -> StructuredGpuRandomTapeBuffers:
        self.random.reset_gu_pos = self._copy_tensor_buffer(
            self.random.reset_gu_pos,
            self._require_runtime_tensor(gu_pos.detach(), field_name="random.reset_gu_pos", dtype=torch.float32),
        )
        self.random.reset_uav_pos = self._copy_tensor_buffer(
            self.random.reset_uav_pos,
            self._require_runtime_tensor(uav_pos.detach(), field_name="random.reset_uav_pos", dtype=torch.float32),
        )
        self.random.reset_uav_vel = self._copy_tensor_buffer(
            self.random.reset_uav_vel,
            self._require_runtime_tensor(uav_vel.detach(), field_name="random.reset_uav_vel", dtype=torch.float32),
        )
        self.random.reset_gu_cluster_centers = self._copy_tensor_buffer(
            self.random.reset_gu_cluster_centers,
            self._require_runtime_tensor(
                gu_cluster_centers.detach(),
                field_name="random.reset_gu_cluster_centers",
                dtype=torch.float32,
            ),
        )
        self.random.reset_gu_cluster_counts = self._copy_tensor_buffer(
            self.random.reset_gu_cluster_counts,
            self._require_runtime_tensor(
                gu_cluster_counts.detach(),
                field_name="random.reset_gu_cluster_counts",
                dtype=torch.float32,
            ),
        )
        self.random.reset_arrival_base_scale = self._copy_tensor_buffer(
            self.random.reset_arrival_base_scale,
            self._require_runtime_tensor(
                arrival_base_scale.detach(),
                field_name="random.reset_arrival_base_scale",
                dtype=torch.float32,
            ),
        )
        self.random.reset_deadline_steps = self._copy_tensor_buffer(
            self.random.reset_deadline_steps,
            self._require_runtime_tensor(
                deadline_steps.detach(),
                field_name="random.reset_deadline_steps",
                dtype=torch.float32,
            ),
        )
        self.random.reset_doppler_residual = self._copy_tensor_buffer(
            self.random.reset_doppler_residual,
            self._require_runtime_tensor(
                doppler_residual.detach(),
                field_name="random.reset_doppler_residual",
                dtype=torch.float32,
            ),
        )
        self.random.reset_effective_arrival_rate = self._copy_tensor_buffer(
            self.random.reset_effective_arrival_rate,
            self._require_runtime_tensor(
                effective_arrival_rate.detach(),
                field_name="random.reset_effective_arrival_rate",
                dtype=torch.float32,
            ),
        )
        return self.random

    def write_random_rollout_tapes(
        self,
        *,
        arrival_tape: torch.Tensor | None = None,
        arrival_rate_tape: torch.Tensor | None = None,
        hotspot_active_after_tape: torch.Tensor | None = None,
        reset_followup_arrival_tape: torch.Tensor | None = None,
        reset_followup_arrival_rate_tape: torch.Tensor | None = None,
        reset_followup_hotspot_active_after_tape: torch.Tensor | None = None,
        hotspot_active_tape: torch.Tensor | None = None,
        hotspot_mask_tape: torch.Tensor | None = None,
        fading_gain_tape: torch.Tensor | None = None,
        doppler_noise_tape: torch.Tensor | None = None,
    ) -> StructuredGpuRandomTapeBuffers:
        if arrival_tape is not None:
            self.random.arrival_rollout_tape = self._copy_tensor_buffer(
                self.random.arrival_rollout_tape,
                self._require_runtime_tensor(
                    arrival_tape.detach(),
                    field_name="random.arrival_rollout_tape",
                    dtype=torch.float32,
                ),
            )
        else:
            self.random.arrival_rollout_tape = None
        if arrival_rate_tape is not None:
            self.random.arrival_rate_rollout_tape = self._copy_tensor_buffer(
                self.random.arrival_rate_rollout_tape,
                self._require_runtime_tensor(
                    arrival_rate_tape.detach(),
                    field_name="random.arrival_rate_rollout_tape",
                    dtype=torch.float32,
                ),
            )
        else:
            self.random.arrival_rate_rollout_tape = None
        if hotspot_active_after_tape is not None:
            self.random.hotspot_active_after_rollout_tape = self._copy_tensor_buffer(
                self.random.hotspot_active_after_rollout_tape,
                self._require_runtime_tensor(
                    hotspot_active_after_tape.detach(),
                    field_name="random.hotspot_active_after_rollout_tape",
                    dtype=torch.int32,
                ),
            )
        else:
            self.random.hotspot_active_after_rollout_tape = None
        if reset_followup_arrival_tape is not None:
            self.random.reset_followup_arrival_rollout_tape = self._copy_tensor_buffer(
                self.random.reset_followup_arrival_rollout_tape,
                self._require_runtime_tensor(
                    reset_followup_arrival_tape.detach(),
                    field_name="random.reset_followup_arrival_rollout_tape",
                    dtype=torch.float32,
                ),
            )
        else:
            self.random.reset_followup_arrival_rollout_tape = None
        if reset_followup_arrival_rate_tape is not None:
            self.random.reset_followup_arrival_rate_rollout_tape = self._copy_tensor_buffer(
                self.random.reset_followup_arrival_rate_rollout_tape,
                self._require_runtime_tensor(
                    reset_followup_arrival_rate_tape.detach(),
                    field_name="random.reset_followup_arrival_rate_rollout_tape",
                    dtype=torch.float32,
                ),
            )
        else:
            self.random.reset_followup_arrival_rate_rollout_tape = None
        if reset_followup_hotspot_active_after_tape is not None:
            self.random.reset_followup_hotspot_active_after_rollout_tape = self._copy_tensor_buffer(
                self.random.reset_followup_hotspot_active_after_rollout_tape,
                self._require_runtime_tensor(
                    reset_followup_hotspot_active_after_tape.detach(),
                    field_name="random.reset_followup_hotspot_active_after_rollout_tape",
                    dtype=torch.int32,
                ),
            )
        else:
            self.random.reset_followup_hotspot_active_after_rollout_tape = None
        if hotspot_active_tape is not None:
            self.random.hotspot_active_rollout_tape = self._copy_tensor_buffer(
                self.random.hotspot_active_rollout_tape,
                self._require_runtime_tensor(
                    hotspot_active_tape.detach(),
                    field_name="random.hotspot_active_rollout_tape",
                    dtype=torch.long,
                ),
            )
        else:
            self.random.hotspot_active_rollout_tape = None
        if hotspot_mask_tape is not None:
            self.random.hotspot_mask_rollout_tape = self._copy_tensor_buffer(
                self.random.hotspot_mask_rollout_tape,
                self._require_runtime_tensor(
                    hotspot_mask_tape.detach(),
                    field_name="random.hotspot_mask_rollout_tape",
                    dtype=torch.float32,
                ),
            )
        else:
            self.random.hotspot_mask_rollout_tape = None
        if fading_gain_tape is not None:
            self.random.fading_gain_rollout_tape = self._copy_tensor_buffer(
                self.random.fading_gain_rollout_tape,
                self._require_runtime_tensor(
                    fading_gain_tape.detach(),
                    field_name="random.fading_gain_rollout_tape",
                    dtype=torch.float32,
                ),
            )
        else:
            self.random.fading_gain_rollout_tape = None
        if doppler_noise_tape is not None:
            self.random.doppler_noise_rollout_tape = self._copy_tensor_buffer(
                self.random.doppler_noise_rollout_tape,
                self._require_runtime_tensor(
                    doppler_noise_tape.detach(),
                    field_name="random.doppler_noise_rollout_tape",
                    dtype=torch.float32,
                ),
            )
        else:
            self.random.doppler_noise_rollout_tape = None
        return self.random

    def write_random_reset_rollout_tapes(
        self,
        *,
        gu_pos_tape: torch.Tensor,
        uav_pos_tape: torch.Tensor,
        uav_vel_tape: torch.Tensor,
        gu_cluster_centers_tape: torch.Tensor,
        gu_cluster_counts_tape: torch.Tensor,
        arrival_base_scale_tape: torch.Tensor,
        deadline_steps_tape: torch.Tensor,
        doppler_residual_tape: torch.Tensor,
        effective_arrival_rate_tape: torch.Tensor,
        episode_idx_tape: torch.Tensor,
        gu_queue_tape: torch.Tensor | None = None,
        uav_queue_tape: torch.Tensor | None = None,
        sat_queue_tape: torch.Tensor | None = None,
        arrival_rate_vec_tape: torch.Tensor | None = None,
        arrival_ref_tape: torch.Tensor | None = None,
        hotspot_active_idx_tape: torch.Tensor | None = None,
        hotspot_subset_count_tape: torch.Tensor | None = None,
        hotspot_member_mask_tape: torch.Tensor | None = None,
    ) -> StructuredGpuRandomTapeBuffers:
        self.random.reset_gu_pos_rollout_tape = self._copy_tensor_buffer(
            self.random.reset_gu_pos_rollout_tape,
            self._require_runtime_tensor(
                gu_pos_tape.detach(),
                field_name="random.reset_gu_pos_rollout_tape",
                dtype=torch.float32,
            ),
        )
        self.random.reset_uav_pos_rollout_tape = self._copy_tensor_buffer(
            self.random.reset_uav_pos_rollout_tape,
            self._require_runtime_tensor(
                uav_pos_tape.detach(),
                field_name="random.reset_uav_pos_rollout_tape",
                dtype=torch.float32,
            ),
        )
        self.random.reset_uav_vel_rollout_tape = self._copy_tensor_buffer(
            self.random.reset_uav_vel_rollout_tape,
            self._require_runtime_tensor(
                uav_vel_tape.detach(),
                field_name="random.reset_uav_vel_rollout_tape",
                dtype=torch.float32,
            ),
        )
        self.random.reset_gu_cluster_centers_rollout_tape = self._copy_tensor_buffer(
            self.random.reset_gu_cluster_centers_rollout_tape,
            self._require_runtime_tensor(
                gu_cluster_centers_tape.detach(),
                field_name="random.reset_gu_cluster_centers_rollout_tape",
                dtype=torch.float32,
            ),
        )
        self.random.reset_gu_cluster_counts_rollout_tape = self._copy_tensor_buffer(
            self.random.reset_gu_cluster_counts_rollout_tape,
            self._require_runtime_tensor(
                gu_cluster_counts_tape.detach(),
                field_name="random.reset_gu_cluster_counts_rollout_tape",
                dtype=torch.float32,
            ),
        )
        if gu_queue_tape is not None:
            self.random.reset_gu_queue_rollout_tape = self._copy_tensor_buffer(
                self.random.reset_gu_queue_rollout_tape,
                self._require_runtime_tensor(
                    gu_queue_tape.detach(),
                    field_name="random.reset_gu_queue_rollout_tape",
                    dtype=torch.float32,
                ),
            )
        else:
            self.random.reset_gu_queue_rollout_tape = None
        if uav_queue_tape is not None:
            self.random.reset_uav_queue_rollout_tape = self._copy_tensor_buffer(
                self.random.reset_uav_queue_rollout_tape,
                self._require_runtime_tensor(
                    uav_queue_tape.detach(),
                    field_name="random.reset_uav_queue_rollout_tape",
                    dtype=torch.float32,
                ),
            )
        else:
            self.random.reset_uav_queue_rollout_tape = None
        if sat_queue_tape is not None:
            self.random.reset_sat_queue_rollout_tape = self._copy_tensor_buffer(
                self.random.reset_sat_queue_rollout_tape,
                self._require_runtime_tensor(
                    sat_queue_tape.detach(),
                    field_name="random.reset_sat_queue_rollout_tape",
                    dtype=torch.float32,
                ),
            )
        else:
            self.random.reset_sat_queue_rollout_tape = None
        self.random.reset_arrival_base_scale_rollout_tape = self._copy_tensor_buffer(
            self.random.reset_arrival_base_scale_rollout_tape,
            self._require_runtime_tensor(
                arrival_base_scale_tape.detach(),
                field_name="random.reset_arrival_base_scale_rollout_tape",
                dtype=torch.float32,
            ),
        )
        self.random.reset_deadline_steps_rollout_tape = self._copy_tensor_buffer(
            self.random.reset_deadline_steps_rollout_tape,
            self._require_runtime_tensor(
                deadline_steps_tape.detach(),
                field_name="random.reset_deadline_steps_rollout_tape",
                dtype=torch.float32,
            ),
        )
        self.random.reset_doppler_residual_rollout_tape = self._copy_tensor_buffer(
            self.random.reset_doppler_residual_rollout_tape,
            self._require_runtime_tensor(
                doppler_residual_tape.detach(),
                field_name="random.reset_doppler_residual_rollout_tape",
                dtype=torch.float32,
            ),
        )
        self.random.reset_effective_arrival_rate_rollout_tape = self._copy_tensor_buffer(
            self.random.reset_effective_arrival_rate_rollout_tape,
            self._require_runtime_tensor(
                effective_arrival_rate_tape.detach(),
                field_name="random.reset_effective_arrival_rate_rollout_tape",
                dtype=torch.float32,
            ),
        )
        if arrival_rate_vec_tape is not None:
            self.random.reset_arrival_rate_vec_rollout_tape = self._copy_tensor_buffer(
                self.random.reset_arrival_rate_vec_rollout_tape,
                self._require_runtime_tensor(
                    arrival_rate_vec_tape.detach(),
                    field_name="random.reset_arrival_rate_vec_rollout_tape",
                    dtype=torch.float32,
                ),
            )
        else:
            self.random.reset_arrival_rate_vec_rollout_tape = None
        if arrival_ref_tape is not None:
            self.random.reset_arrival_ref_rollout_tape = self._copy_tensor_buffer(
                self.random.reset_arrival_ref_rollout_tape,
                self._require_runtime_tensor(
                    arrival_ref_tape.detach(),
                    field_name="random.reset_arrival_ref_rollout_tape",
                    dtype=torch.float32,
                ),
            )
        else:
            self.random.reset_arrival_ref_rollout_tape = None
        self.random.reset_episode_idx_rollout_tape = self._copy_tensor_buffer(
            self.random.reset_episode_idx_rollout_tape,
            self._require_runtime_tensor(
                episode_idx_tape.detach(),
                field_name="random.reset_episode_idx_rollout_tape",
                dtype=torch.int32,
            ),
        )
        if hotspot_active_idx_tape is not None:
            self.random.reset_hotspot_active_idx_rollout_tape = self._copy_tensor_buffer(
                self.random.reset_hotspot_active_idx_rollout_tape,
                self._require_runtime_tensor(
                    hotspot_active_idx_tape.detach(),
                    field_name="random.reset_hotspot_active_idx_rollout_tape",
                    dtype=torch.int32,
                ),
            )
        else:
            self.random.reset_hotspot_active_idx_rollout_tape = None
        if hotspot_subset_count_tape is not None:
            self.random.reset_hotspot_subset_count_rollout_tape = self._copy_tensor_buffer(
                self.random.reset_hotspot_subset_count_rollout_tape,
                self._require_runtime_tensor(
                    hotspot_subset_count_tape.detach(),
                    field_name="random.reset_hotspot_subset_count_rollout_tape",
                    dtype=torch.int32,
                ),
            )
        else:
            self.random.reset_hotspot_subset_count_rollout_tape = None
        if hotspot_member_mask_tape is not None:
            self.random.reset_hotspot_member_mask_rollout_tape = self._copy_tensor_buffer(
                self.random.reset_hotspot_member_mask_rollout_tape,
                self._require_runtime_tensor(
                    hotspot_member_mask_tape.detach(),
                    field_name="random.reset_hotspot_member_mask_rollout_tape",
                    dtype=torch.float32,
                ),
            )
        else:
            self.random.reset_hotspot_member_mask_rollout_tape = None
        return self.random

    def write_random_fading_tape(self, fading_gain: torch.Tensor) -> StructuredGpuRandomTapeBuffers:
        self.random.fading_gain = self._copy_tensor_buffer(
            self.random.fading_gain,
            self._require_runtime_tensor(fading_gain.detach(), field_name="random.fading_gain", dtype=torch.float32),
        )
        return self.random

    def write_random_doppler_noise_tape(self, doppler_noise: torch.Tensor) -> StructuredGpuRandomTapeBuffers:
        self.random.doppler_noise = self._copy_tensor_buffer(
            self.random.doppler_noise,
            self._require_runtime_tensor(doppler_noise.detach(), field_name="random.doppler_noise", dtype=torch.float32),
        )
        return self.random

    @staticmethod
    def _ensure_tensor_buffer(
        current: torch.Tensor | None,
        *,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if (
            current is None
            or tuple(current.shape) != tuple(shape)
            or current.dtype != dtype
            or current.device != device
        ):
            return torch.empty(tuple(int(dim) for dim in shape), dtype=dtype, device=device)
        return current

    def preallocate_step_result_buffers(
        self,
        *,
        num_envs: int,
        num_uav: int,
        bw_action_shape: tuple[int, ...],
        device: torch.device,
        reward_mode_active: str | None,
        expose_auxiliary_outputs: bool,
        expose_reward_parts: bool,
        expose_flow_proxy: bool,
    ) -> StructuredBatchStepResult:
        """Create stable result buffers before the BW segment writes them.

        This does not publish a temporary step result. The BW segment receives
        these tensors as output buffers and writes them directly.
        """

        buffers = self.result
        n = max(int(num_envs), 0)
        u = max(int(num_uav), 0)
        action_shape = tuple(int(dim) for dim in bw_action_shape)
        buffers.team_rewards = self._ensure_tensor_buffer(
            buffers.team_rewards,
            shape=(n,),
            dtype=torch.float32,
            device=device,
        )
        buffers.terminated = self._ensure_tensor_buffer(
            buffers.terminated,
            shape=(n,),
            dtype=torch.bool,
            device=device,
        )
        buffers.truncated = self._ensure_tensor_buffer(
            buffers.truncated,
            shape=(n,),
            dtype=torch.bool,
            device=device,
        )
        if expose_auxiliary_outputs:
            buffers.danger_imitation_target = self._ensure_tensor_buffer(
                buffers.danger_imitation_target,
                shape=(n, u, 2),
                dtype=torch.float32,
                device=device,
            )
            buffers.danger_imitation_mask = self._ensure_tensor_buffer(
                buffers.danger_imitation_mask,
                shape=(n, u, 2),
                dtype=torch.float32,
                device=device,
            )
        else:
            buffers.danger_imitation_target = None
            buffers.danger_imitation_mask = None
        for field_name in (
            "bw_access_rewards",
            "bw_weighted_workload_delta_rewards",
            "bw_weighted_workload_level_rewards",
            "bw_gu_queue_level_rewards",
            "bw_system_queue_level_rewards",
            "bw_gu_service_queue_rewards",
        ):
            setattr(
                buffers,
                field_name,
                self._ensure_tensor_buffer(
                    getattr(buffers, field_name),
                    shape=(n,),
                    dtype=torch.float32,
                    device=device,
                ),
            )
        if expose_flow_proxy:
            buffers.bw_flow_proxy_scores = self._ensure_tensor_buffer(
                buffers.bw_flow_proxy_scores,
                shape=action_shape,
                dtype=torch.float32,
                device=device,
            )
            buffers.bw_flow_proxy_mask = self._ensure_tensor_buffer(
                buffers.bw_flow_proxy_mask,
                shape=action_shape,
                dtype=torch.float32,
                device=device,
            )
            buffers.bw_flow_proxy_deltas = self._ensure_tensor_buffer(
                buffers.bw_flow_proxy_deltas,
                shape=action_shape,
                dtype=torch.float32,
                device=device,
            )
        else:
            buffers.bw_flow_proxy_scores = None
            buffers.bw_flow_proxy_mask = None
            buffers.bw_flow_proxy_deltas = None

        existing_reward_parts = buffers.reward_parts
        reward_part_kwargs: dict[str, torch.Tensor] = {}
        for field_info in fields(StructuredGpuRewardPartBuffers):
            field_name = str(field_info.name)
            current_tensor = None if existing_reward_parts is None else getattr(existing_reward_parts, field_name)
            reward_part_kwargs[field_name] = self._ensure_tensor_buffer(
                current_tensor,
                shape=(n,),
                dtype=torch.float32,
                device=device,
            )
        reward_parts = StructuredGpuRewardPartBuffers(**reward_part_kwargs)
        reward_part_tensors: dict[str, torch.Tensor] = {
            field_name: getattr(reward_parts, field_name) for field_name in _BW_REWARD_PART_KEYS
        }
        buffers.reward_parts = reward_parts
        buffers.reward_part_tensors = reward_part_tensors
        buffers.reward_mode_active = reward_mode_active
        buffers.batch_result = StructuredBatchStepResult(
            team_rewards=buffers.team_rewards,
            terminated=buffers.terminated,
            truncated=buffers.truncated,
            danger_imitation_target=buffers.danger_imitation_target,
            danger_imitation_mask=buffers.danger_imitation_mask,
            bw_access_rewards=buffers.bw_access_rewards,
            bw_weighted_workload_delta_rewards=buffers.bw_weighted_workload_delta_rewards,
            bw_weighted_workload_level_rewards=buffers.bw_weighted_workload_level_rewards,
            bw_gu_queue_level_rewards=buffers.bw_gu_queue_level_rewards,
            bw_system_queue_level_rewards=buffers.bw_system_queue_level_rewards,
            bw_gu_service_queue_rewards=buffers.bw_gu_service_queue_rewards,
            bw_flow_proxy_scores=buffers.bw_flow_proxy_scores,
            bw_flow_proxy_mask=buffers.bw_flow_proxy_mask,
            bw_flow_proxy_deltas=buffers.bw_flow_proxy_deltas,
            reward_part_tensors=reward_part_tensors if expose_reward_parts else None,
            reward_mode_active=reward_mode_active,
        )
        return buffers.batch_result

    def bind_horizon_step_result_views(
        self,
        *,
        num_steps: int,
        num_envs: int,
        expose_reward_parts: bool = True,
    ) -> list[StructuredBatchStepResult]:
        history = self.history
        capacity = int(getattr(history, "capacity", 0) or 0)
        steps = max(int(num_steps), 0)
        n = max(int(num_envs), 0)
        if steps <= 0 or n <= 0:
            self.result.step_result_views = []
            self.result.horizon_num_steps = steps
            self.main.step_result_view = None
            return []
        if capacity < steps:
            raise RuntimeError("native rollout history capacity is smaller than the requested horizon.")
        if getattr(history.accel_stage, "world_batch", None) is not None:
            sample_world = getattr(history.accel_stage.world_batch, "uav_nodes", None)
            if torch.is_tensor(sample_world) and int(sample_world.shape[0]) < (steps + 1) * n:
                raise RuntimeError("native rollout accel world history must have K+1 slots for horizon bootstrap.")

        def _slice(value: torch.Tensor | None, *, step_index: int) -> torch.Tensor | None:
            if value is None:
                return None
            if not torch.is_tensor(value):
                raise RuntimeError("native rollout result view field must be tensor or None.")
            return value.narrow(0, int(step_index) * n, n)

        views: list[StructuredBatchStepResult] = []
        reward_part_tensors = getattr(history.bw_stage, "reward_part_tensors", None)
        for step_index in range(steps):
            step_reward_parts = None
            if expose_reward_parts and isinstance(reward_part_tensors, dict):
                step_reward_parts = {
                    str(key): tensor.narrow(0, int(step_index) * n, n)
                    for key, tensor in reward_part_tensors.items()
                    if torch.is_tensor(tensor)
                }
            views.append(
                StructuredBatchStepResult(
                    team_rewards=_slice(history.bw_stage.rewards, step_index=step_index),
                    terminated=_slice(history.terminated, step_index=step_index),
                    truncated=_slice(history.truncated, step_index=step_index),
                    danger_imitation_target=_slice(history.accel_stage.danger_imitation_targets, step_index=step_index),
                    danger_imitation_mask=_slice(history.accel_stage.danger_imitation_masks, step_index=step_index),
                    bw_access_rewards=_slice(history.bw_stage.bw_access_rewards, step_index=step_index),
                    bw_weighted_workload_delta_rewards=_slice(
                        history.bw_stage.bw_weighted_workload_delta_rewards,
                        step_index=step_index,
                    ),
                    bw_weighted_workload_level_rewards=_slice(
                        history.bw_stage.bw_weighted_workload_level_rewards,
                        step_index=step_index,
                    ),
                    bw_gu_queue_level_rewards=_slice(history.bw_stage.bw_gu_queue_level_rewards, step_index=step_index),
                    bw_system_queue_level_rewards=_slice(
                        history.bw_stage.bw_system_queue_level_rewards,
                        step_index=step_index,
                    ),
                    bw_gu_service_queue_rewards=_slice(
                        history.bw_stage.bw_gu_service_queue_rewards,
                        step_index=step_index,
                    ),
                    bw_flow_proxy_scores=_slice(history.bw_stage.bw_flow_proxy_scores, step_index=step_index),
                    bw_flow_proxy_mask=_slice(history.bw_stage.bw_flow_proxy_masks, step_index=step_index),
                    bw_flow_proxy_deltas=_slice(history.bw_stage.bw_flow_proxy_deltas, step_index=step_index),
                    reward_part_tensors=step_reward_parts,
                    reward_mode_active=self.main.bw_reward_mode_active,
                )
            )
        self.result.step_result_views = views
        self.result.horizon_num_steps = steps
        self.main.step_result_view = views[0] if views else None
        return views

    def materialize_step_result_view(
        self,
        view: StructuredBatchStepResult,
        *,
        device: str | torch.device = "cpu",
    ) -> StructuredBatchStepResult:
        if not isinstance(view, StructuredBatchStepResult):
            raise TypeError("materialize_step_result_view expects StructuredBatchStepResult.")
        return _materialize_tensor_tree(view, device=device)

    def materialize_horizon_step_results(
        self,
        views: Sequence[StructuredBatchStepResult] | None = None,
        *,
        device: str | torch.device = "cpu",
    ) -> list[StructuredBatchStepResult]:
        source = self.result.step_result_views if views is None else list(views)
        return [self.materialize_step_result_view(view, device=device) for view in source]

    def begin_rollout_training_ring(self, *, capacity: int, num_envs: int) -> None:
        self.activate_rollout_history()
        capacity_i = max(int(capacity), 1)
        num_envs_i = max(int(num_envs), 0)
        history = self.rollout_history
        history.capacity = capacity_i
        history.cursor = 0
        history.num_envs = num_envs_i
        if torch.is_tensor(history.terminal_next_world_mask):
            history.terminal_next_world_mask.zero_()

    def begin_horizon(self, *, num_steps: int, num_envs: int) -> list[StructuredBatchStepResult]:
        self.activate_rollout_history()
        steps = max(int(num_steps), 0)
        n = max(int(num_envs), 0)
        history = self.history
        if int(history.capacity) < max(steps, 1):
            raise RuntimeError("native rollout history capacity is smaller than the requested horizon.")
        history.cursor = 0
        history.num_envs = n
        self.main.current_history_slot = 0
        self.main.accel_active_idx = 0
        return self.bind_horizon_step_result_views(
            num_steps=steps,
            num_envs=n,
            expose_reward_parts=getattr(history.bw_stage, "reward_part_tensors", None) is not None,
        )

    def _ensure_history_slot(self, *, num_envs: int) -> int:
        history = self.history
        if int(history.capacity) <= 0:
            raise RuntimeError("native rollout history capacity was not initialized at rollout begin.")
        if int(history.num_envs) not in {0, int(num_envs)}:
            raise RuntimeError(
                f"rollout history num_envs changed: got {int(num_envs)}, expected {int(history.num_envs)}."
            )
        history.num_envs = int(num_envs)
        if int(history.cursor) >= int(history.capacity):
            raise RuntimeError("rollout training ring capacity exhausted; call begin_rollout_training_ring with rollout length.")
        return int(history.cursor)

    def _ensure_history_dataclass_buffer(
        self,
        current: Any | None,
        source: Any,
        *,
        capacity: int,
        field_name: str,
    ) -> Any | None:
        if not _is_tensor_dataclass(source):
            return current
        if not _training_dataclass_matches(current, source, capacity):
            if int(self.history.cursor) != 0:
                raise RuntimeError(f"{field_name} shape changed after rollout training ring started.")
            return _empty_training_like_dataclass(source, capacity)
        return current

    def _ensure_history_tensor_buffer(
        self,
        current: torch.Tensor | None,
        value: torch.Tensor,
        *,
        capacity: int,
        field_name: str,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        src = self._require_runtime_tensor(value.detach(), field_name=f"history.{field_name}")
        target_dtype = src.dtype if dtype is None else dtype
        expected_shape = (int(capacity) * int(src.shape[0]),) + tuple(src.shape[1:])
        if (
            current is None
            or tuple(current.shape) != expected_shape
            or current.dtype != target_dtype
            or current.device != src.device
        ):
            if int(self.history.cursor) != 0:
                raise RuntimeError(f"{field_name} shape changed after rollout training ring started.")
            current = torch.empty(expected_shape, dtype=target_dtype, device=src.device)
        return current

    def _ensure_optional_history_tensor_buffer(
        self,
        current: torch.Tensor | None,
        value: torch.Tensor | None,
        *,
        capacity: int,
        field_name: str,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        return self._ensure_history_tensor_buffer(
            current,
            value,
            capacity=capacity,
            field_name=field_name,
            dtype=dtype,
        )

    def capture_native_main_kernel_base_views(self) -> None:
        self.main.base_result = StructuredGpuStepResultBuffers(
            batch_result=self.result.batch_result,
            team_rewards=self.result.team_rewards,
            terminated=self.result.terminated,
            truncated=self.result.truncated,
            danger_imitation_target=self.result.danger_imitation_target,
            danger_imitation_mask=self.result.danger_imitation_mask,
            bw_access_rewards=self.result.bw_access_rewards,
            bw_weighted_workload_delta_rewards=self.result.bw_weighted_workload_delta_rewards,
            bw_weighted_workload_level_rewards=self.result.bw_weighted_workload_level_rewards,
            bw_gu_queue_level_rewards=self.result.bw_gu_queue_level_rewards,
            bw_system_queue_level_rewards=self.result.bw_system_queue_level_rewards,
            bw_gu_service_queue_rewards=self.result.bw_gu_service_queue_rewards,
            bw_flow_proxy_scores=self.result.bw_flow_proxy_scores,
            bw_flow_proxy_mask=self.result.bw_flow_proxy_mask,
            bw_flow_proxy_deltas=self.result.bw_flow_proxy_deltas,
            reward_parts=self.result.reward_parts,
            reward_part_tensors=self.result.reward_part_tensors,
            reward_mode_active=self.result.reward_mode_active,
        )

    def restore_native_main_kernel_base_views(self) -> None:
        base_result = self.main.base_result
        if isinstance(base_result, StructuredGpuStepResultBuffers):
            self.result.batch_result = base_result.batch_result
            self.result.team_rewards = base_result.team_rewards
            self.result.terminated = base_result.terminated
            self.result.truncated = base_result.truncated
            self.result.danger_imitation_target = base_result.danger_imitation_target
            self.result.danger_imitation_mask = base_result.danger_imitation_mask
            self.result.bw_access_rewards = base_result.bw_access_rewards
            self.result.bw_weighted_workload_delta_rewards = base_result.bw_weighted_workload_delta_rewards
            self.result.bw_weighted_workload_level_rewards = base_result.bw_weighted_workload_level_rewards
            self.result.bw_gu_queue_level_rewards = base_result.bw_gu_queue_level_rewards
            self.result.bw_system_queue_level_rewards = base_result.bw_system_queue_level_rewards
            self.result.bw_gu_service_queue_rewards = base_result.bw_gu_service_queue_rewards
            self.result.bw_flow_proxy_scores = base_result.bw_flow_proxy_scores
            self.result.bw_flow_proxy_mask = base_result.bw_flow_proxy_mask
            self.result.bw_flow_proxy_deltas = base_result.bw_flow_proxy_deltas
            self.result.reward_parts = base_result.reward_parts
            self.result.reward_part_tensors = base_result.reward_part_tensors
            self.result.reward_mode_active = base_result.reward_mode_active

    def _bind_step_result_views(
        self,
        *,
        team_rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        danger_imitation_target: torch.Tensor | None,
        danger_imitation_mask: torch.Tensor | None,
        bw_access_rewards: torch.Tensor | None,
        bw_weighted_workload_delta_rewards: torch.Tensor | None,
        bw_weighted_workload_level_rewards: torch.Tensor | None,
        bw_gu_queue_level_rewards: torch.Tensor | None,
        bw_system_queue_level_rewards: torch.Tensor | None,
        bw_gu_service_queue_rewards: torch.Tensor | None,
        bw_flow_proxy_scores: torch.Tensor | None,
        bw_flow_proxy_mask: torch.Tensor | None,
        bw_flow_proxy_deltas: torch.Tensor | None,
    ) -> None:
        self.result.team_rewards = team_rewards
        self.result.terminated = terminated
        self.result.truncated = truncated
        self.result.danger_imitation_target = danger_imitation_target
        self.result.danger_imitation_mask = danger_imitation_mask
        self.result.bw_access_rewards = bw_access_rewards
        self.result.bw_weighted_workload_delta_rewards = bw_weighted_workload_delta_rewards
        self.result.bw_weighted_workload_level_rewards = bw_weighted_workload_level_rewards
        self.result.bw_gu_queue_level_rewards = bw_gu_queue_level_rewards
        self.result.bw_system_queue_level_rewards = bw_system_queue_level_rewards
        self.result.bw_gu_service_queue_rewards = bw_gu_service_queue_rewards
        self.result.bw_flow_proxy_scores = bw_flow_proxy_scores
        self.result.bw_flow_proxy_mask = bw_flow_proxy_mask
        self.result.bw_flow_proxy_deltas = bw_flow_proxy_deltas
        self.result.batch_result = StructuredBatchStepResult(
            team_rewards=team_rewards,
            terminated=terminated,
            truncated=truncated,
            danger_imitation_target=danger_imitation_target,
            danger_imitation_mask=danger_imitation_mask,
            bw_access_rewards=bw_access_rewards,
            bw_weighted_workload_delta_rewards=bw_weighted_workload_delta_rewards,
            bw_weighted_workload_level_rewards=bw_weighted_workload_level_rewards,
            bw_gu_queue_level_rewards=bw_gu_queue_level_rewards,
            bw_system_queue_level_rewards=bw_system_queue_level_rewards,
            bw_gu_service_queue_rewards=bw_gu_service_queue_rewards,
            bw_flow_proxy_scores=bw_flow_proxy_scores,
            bw_flow_proxy_mask=bw_flow_proxy_mask,
            bw_flow_proxy_deltas=bw_flow_proxy_deltas,
            reward_part_tensors=self.result.reward_part_tensors,
            reward_mode_active=self.result.reward_mode_active,
        )

    def preallocate_native_main_kernel_training_ring_buffers(
        self,
        *,
        accel_world_batch: Any,
        sat_world_batch: Any,
        bw_world_batch: Any,
        next_world_batch: Any,
        accel_runtime_state: Any | None = None,
        accel_runtime_stage: Any | None = None,
        sat_runtime_state: Any | None = None,
        sat_runtime_stage: Any | None = None,
        bw_runtime_state: Any | None = None,
        bw_runtime_cache: StructuredGpuBwRuntimeCacheBuffers | None = None,
        bw_runtime_stage: Any | None = None,
        accel_local_batch: Any,
        sat_local_batch: Any,
        bw_local_batch: Any,
        accel_actions: torch.Tensor,
        sat_actions: torch.Tensor,
        sat_action_indices: torch.Tensor,
        bw_actions: torch.Tensor,
        accel_old_logprobs: torch.Tensor,
        sat_old_logprobs: torch.Tensor,
        sat_old_logprobs_per_agent: torch.Tensor | None,
        bw_old_logprobs: torch.Tensor,
        accel_values: torch.Tensor,
        sat_values: torch.Tensor,
        bw_values: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        accel_latent_actions: torch.Tensor | None = None,
        accel_danger_imitation_targets: torch.Tensor | None = None,
        accel_danger_imitation_masks: torch.Tensor | None = None,
        bw_access_rewards: torch.Tensor | None = None,
        bw_weighted_workload_delta_rewards: torch.Tensor | None = None,
        bw_weighted_workload_level_rewards: torch.Tensor | None = None,
        bw_gu_queue_level_rewards: torch.Tensor | None = None,
        bw_system_queue_level_rewards: torch.Tensor | None = None,
        bw_gu_service_queue_rewards: torch.Tensor | None = None,
        bw_flow_proxy_scores: torch.Tensor | None = None,
        bw_flow_proxy_masks: torch.Tensor | None = None,
        bw_flow_proxy_deltas: torch.Tensor | None = None,
        bw_ref_actions: torch.Tensor | None = None,
        bw_old_logprobs_per_agent: torch.Tensor | None = None,
        bw_entropy_per_agent: torch.Tensor | None = None,
        bw_logprob_raw_per_agent: torch.Tensor | None = None,
        bw_entropy_raw_per_agent: torch.Tensor | None = None,
        bw_tau: torch.Tensor | None = None,
        bw_kappa: torch.Tensor | None = None,
        bw_valid_count: torch.Tensor | None = None,
        bw_latent_count: torch.Tensor | None = None,
    ) -> None:
        self.activate_rollout_history()
        history = self.rollout_history
        accel_stage = history.accel_stage
        sat_stage = history.sat_stage
        bw_stage = history.bw_stage
        capacity = int(history.capacity)
        if capacity <= 0:
            raise RuntimeError("native main-kernel history capacity must be set before buffer preallocation.")
        accel_stage.world_batch = self._ensure_history_dataclass_buffer(
            accel_stage.world_batch,
            accel_world_batch,
            capacity=capacity + 1,
            field_name="accel_stage.world_batch",
        )
        sat_stage.world_batch = self._ensure_history_dataclass_buffer(
            sat_stage.world_batch,
            sat_world_batch,
            capacity=capacity,
            field_name="sat_stage.world_batch",
        )
        bw_stage.world_batch = self._ensure_history_dataclass_buffer(
            bw_stage.world_batch,
            bw_world_batch,
            capacity=capacity,
            field_name="bw_stage.world_batch",
        )
        history.terminal_next_world = self._ensure_history_dataclass_buffer(
            history.terminal_next_world,
            next_world_batch,
            capacity=capacity,
            field_name="terminal_next_world",
        )
        if accel_runtime_state is not None:
            history.accel_runtime_state = self._ensure_history_dataclass_buffer(
                history.accel_runtime_state,
                accel_runtime_state,
                capacity=capacity,
                field_name="accel_runtime_state",
            )
        if accel_runtime_stage is not None:
            history.accel_runtime_stage = self._ensure_history_dataclass_buffer(
                history.accel_runtime_stage,
                accel_runtime_stage,
                capacity=capacity,
                field_name="accel_runtime_stage",
            )
        if sat_runtime_state is not None:
            history.sat_runtime_state = self._ensure_history_dataclass_buffer(
                history.sat_runtime_state,
                sat_runtime_state,
                capacity=capacity,
                field_name="sat_runtime_state",
            )
        if sat_runtime_stage is not None:
            history.sat_runtime_stage = self._ensure_history_dataclass_buffer(
                history.sat_runtime_stage,
                sat_runtime_stage,
                capacity=capacity,
                field_name="sat_runtime_stage",
            )
        if bw_runtime_state is not None:
            history.bw_runtime_state = self._ensure_history_dataclass_buffer(
                history.bw_runtime_state,
                bw_runtime_state,
                capacity=capacity,
                field_name="bw_runtime_state",
            )
        if bw_runtime_cache is not None:
            history.bw_runtime_cache = self._ensure_history_dataclass_buffer(
                history.bw_runtime_cache,
                bw_runtime_cache,
                capacity=capacity,
                field_name="bw_runtime_cache",
            )
        if bw_runtime_stage is not None:
            history.bw_runtime_stage = self._ensure_history_dataclass_buffer(
                history.bw_runtime_stage,
                bw_runtime_stage,
                capacity=capacity,
                field_name="bw_runtime_stage",
            )
        history.terminal_next_world_mask = self._ensure_history_tensor_buffer(
            history.terminal_next_world_mask,
            terminated.reshape(-1),
            capacity=capacity,
            field_name="terminal_next_world_mask",
            dtype=torch.bool,
        )
        for field_name in StructuredGpuAccelObsView._tensor_fields:
            setattr(
                accel_stage,
                field_name,
                self._ensure_history_tensor_buffer(
                    getattr(accel_stage, field_name),
                    getattr(accel_local_batch, field_name),
                    capacity=capacity,
                    field_name=f"accel_stage.{field_name}",
                    dtype=getattr(accel_local_batch, field_name).dtype,
                ),
            )
        for field_name in StructuredGpuSatObsView._tensor_fields:
            setattr(
                sat_stage,
                field_name,
                self._ensure_history_tensor_buffer(
                    getattr(sat_stage, field_name),
                    getattr(sat_local_batch, field_name),
                    capacity=capacity,
                    field_name=f"sat_stage.{field_name}",
                    dtype=getattr(sat_local_batch, field_name).dtype,
                ),
            )
        for field_name in StructuredGpuBwObsView._tensor_fields:
            setattr(
                bw_stage,
                field_name,
                self._ensure_history_tensor_buffer(
                    getattr(bw_stage, field_name),
                    getattr(bw_local_batch, field_name),
                    capacity=capacity,
                    field_name=f"bw_stage.{field_name}",
                    dtype=getattr(bw_local_batch, field_name).dtype,
                ),
            )
        accel_stage.actions = self._ensure_history_tensor_buffer(
            accel_stage.actions,
            accel_actions,
            capacity=capacity,
            field_name="accel_stage.actions",
        )
        accel_stage.latent_actions = self._ensure_optional_history_tensor_buffer(
            accel_stage.latent_actions,
            accel_latent_actions,
            capacity=capacity,
            field_name="accel_stage.latent_actions",
            dtype=torch.float32,
        )
        sat_stage.actions = self._ensure_history_tensor_buffer(
            sat_stage.actions,
            sat_actions,
            capacity=capacity,
            field_name="sat_stage.actions",
        )
        sat_stage.action_indices = self._ensure_history_tensor_buffer(
            sat_stage.action_indices,
            sat_action_indices,
            capacity=capacity,
            field_name="sat_stage.action_indices",
            dtype=torch.long,
        )
        bw_stage.actions = self._ensure_history_tensor_buffer(
            bw_stage.actions,
            bw_actions,
            capacity=capacity,
            field_name="bw_stage.actions",
        )
        accel_stage.old_logprobs = self._ensure_history_tensor_buffer(
            accel_stage.old_logprobs,
            accel_old_logprobs.reshape(-1),
            capacity=capacity,
            field_name="accel_stage.old_logprobs",
            dtype=torch.float32,
        )
        sat_stage.old_logprobs = self._ensure_history_tensor_buffer(
            sat_stage.old_logprobs,
            sat_old_logprobs.reshape(-1),
            capacity=capacity,
            field_name="sat_stage.old_logprobs",
            dtype=torch.float32,
        )
        sat_stage.old_logprobs_per_agent = self._ensure_optional_history_tensor_buffer(
            sat_stage.old_logprobs_per_agent,
            sat_old_logprobs_per_agent,
            capacity=capacity,
            field_name="sat_stage.old_logprobs_per_agent",
            dtype=torch.float32,
        )
        bw_stage.old_logprobs = self._ensure_history_tensor_buffer(
            bw_stage.old_logprobs,
            bw_old_logprobs.reshape(-1),
            capacity=capacity,
            field_name="bw_stage.old_logprobs",
            dtype=torch.float32,
        )
        accel_stage.values = self._ensure_history_tensor_buffer(
            accel_stage.values,
            accel_values.reshape(-1),
            capacity=capacity,
            field_name="accel_stage.values",
            dtype=torch.float32,
        )
        sat_stage.values = self._ensure_history_tensor_buffer(
            sat_stage.values,
            sat_values.reshape(-1),
            capacity=capacity,
            field_name="sat_stage.values",
            dtype=torch.float32,
        )
        bw_stage.values = self._ensure_history_tensor_buffer(
            bw_stage.values,
            bw_values.reshape(-1),
            capacity=capacity,
            field_name="bw_stage.values",
            dtype=torch.float32,
        )
        bw_stage.rewards = self._ensure_history_tensor_buffer(
            bw_stage.rewards,
            rewards.reshape(-1),
            capacity=capacity,
            field_name="bw_stage.rewards",
            dtype=torch.float32,
        )
        history.terminated = self._ensure_history_tensor_buffer(
            history.terminated,
            terminated.reshape(-1),
            capacity=capacity,
            field_name="terminated",
            dtype=torch.bool,
        )
        history.truncated = self._ensure_history_tensor_buffer(
            history.truncated,
            truncated.reshape(-1),
            capacity=capacity,
            field_name="truncated",
            dtype=torch.bool,
        )
        accel_stage.danger_imitation_targets = self._ensure_optional_history_tensor_buffer(
            accel_stage.danger_imitation_targets,
            accel_danger_imitation_targets,
            capacity=capacity,
            field_name="accel_stage.danger_imitation_targets",
            dtype=torch.float32,
        )
        accel_stage.danger_imitation_masks = self._ensure_optional_history_tensor_buffer(
            accel_stage.danger_imitation_masks,
            accel_danger_imitation_masks,
            capacity=capacity,
            field_name="accel_stage.danger_imitation_masks",
            dtype=torch.float32,
        )
        bw_stage.bw_access_rewards = self._ensure_optional_history_tensor_buffer(
            bw_stage.bw_access_rewards,
            bw_access_rewards,
            capacity=capacity,
            field_name="bw_stage.bw_access_rewards",
            dtype=torch.float32,
        )
        bw_stage.bw_weighted_workload_delta_rewards = self._ensure_optional_history_tensor_buffer(
            bw_stage.bw_weighted_workload_delta_rewards,
            bw_weighted_workload_delta_rewards,
            capacity=capacity,
            field_name="bw_stage.bw_weighted_workload_delta_rewards",
            dtype=torch.float32,
        )
        bw_stage.bw_weighted_workload_level_rewards = self._ensure_optional_history_tensor_buffer(
            bw_stage.bw_weighted_workload_level_rewards,
            bw_weighted_workload_level_rewards,
            capacity=capacity,
            field_name="bw_stage.bw_weighted_workload_level_rewards",
            dtype=torch.float32,
        )
        bw_stage.bw_gu_queue_level_rewards = self._ensure_optional_history_tensor_buffer(
            bw_stage.bw_gu_queue_level_rewards,
            bw_gu_queue_level_rewards,
            capacity=capacity,
            field_name="bw_stage.bw_gu_queue_level_rewards",
            dtype=torch.float32,
        )
        bw_stage.bw_system_queue_level_rewards = self._ensure_optional_history_tensor_buffer(
            bw_stage.bw_system_queue_level_rewards,
            bw_system_queue_level_rewards,
            capacity=capacity,
            field_name="bw_stage.bw_system_queue_level_rewards",
            dtype=torch.float32,
        )
        bw_stage.bw_gu_service_queue_rewards = self._ensure_optional_history_tensor_buffer(
            bw_stage.bw_gu_service_queue_rewards,
            bw_gu_service_queue_rewards,
            capacity=capacity,
            field_name="bw_stage.bw_gu_service_queue_rewards",
            dtype=torch.float32,
        )
        bw_stage.bw_flow_proxy_scores = self._ensure_optional_history_tensor_buffer(
            bw_stage.bw_flow_proxy_scores,
            bw_flow_proxy_scores,
            capacity=capacity,
            field_name="bw_stage.bw_flow_proxy_scores",
            dtype=torch.float32,
        )
        bw_stage.bw_flow_proxy_masks = self._ensure_optional_history_tensor_buffer(
            bw_stage.bw_flow_proxy_masks,
            bw_flow_proxy_masks,
            capacity=capacity,
            field_name="bw_stage.bw_flow_proxy_masks",
            dtype=torch.float32,
        )
        bw_stage.bw_flow_proxy_deltas = self._ensure_optional_history_tensor_buffer(
            bw_stage.bw_flow_proxy_deltas,
            bw_flow_proxy_deltas,
            capacity=capacity,
            field_name="bw_stage.bw_flow_proxy_deltas",
            dtype=torch.float32,
        )
        bw_stage.bw_ref_actions = self._ensure_optional_history_tensor_buffer(
            bw_stage.bw_ref_actions,
            bw_ref_actions,
            capacity=capacity,
            field_name="bw_stage.bw_ref_actions",
            dtype=torch.float32,
        )
        bw_stage.bw_old_logprobs_per_agent = self._ensure_optional_history_tensor_buffer(
            bw_stage.bw_old_logprobs_per_agent,
            bw_old_logprobs_per_agent,
            capacity=capacity,
            field_name="bw_stage.bw_old_logprobs_per_agent",
            dtype=torch.float32,
        )
        bw_stage.bw_entropy_per_agent = self._ensure_optional_history_tensor_buffer(
            bw_stage.bw_entropy_per_agent,
            bw_entropy_per_agent,
            capacity=capacity,
            field_name="bw_stage.bw_entropy_per_agent",
            dtype=torch.float32,
        )
        bw_stage.bw_logprob_raw_per_agent = self._ensure_optional_history_tensor_buffer(
            bw_stage.bw_logprob_raw_per_agent,
            bw_logprob_raw_per_agent,
            capacity=capacity,
            field_name="bw_stage.bw_logprob_raw_per_agent",
            dtype=torch.float32,
        )
        bw_stage.bw_entropy_raw_per_agent = self._ensure_optional_history_tensor_buffer(
            bw_stage.bw_entropy_raw_per_agent,
            bw_entropy_raw_per_agent,
            capacity=capacity,
            field_name="bw_stage.bw_entropy_raw_per_agent",
            dtype=torch.float32,
        )
        bw_stage.bw_tau = self._ensure_optional_history_tensor_buffer(
            bw_stage.bw_tau,
            bw_tau,
            capacity=capacity,
            field_name="bw_stage.bw_tau",
            dtype=torch.float32,
        )
        bw_stage.bw_kappa = self._ensure_optional_history_tensor_buffer(
            bw_stage.bw_kappa,
            bw_kappa,
            capacity=capacity,
            field_name="bw_stage.bw_kappa",
            dtype=torch.float32,
        )
        bw_stage.bw_valid_count = self._ensure_optional_history_tensor_buffer(
            bw_stage.bw_valid_count,
            bw_valid_count,
            capacity=capacity,
            field_name="bw_stage.bw_valid_count",
            dtype=torch.long,
        )
        bw_stage.bw_latent_count = self._ensure_optional_history_tensor_buffer(
            bw_stage.bw_latent_count,
            bw_latent_count,
            capacity=capacity,
            field_name="bw_stage.bw_latent_count",
            dtype=torch.long,
        )
        existing_reward_parts = bw_stage.reward_parts
        result_reward_parts = self.result.reward_parts
        if isinstance(result_reward_parts, StructuredGpuRewardPartBuffers):
            reward_part_kwargs: dict[str, torch.Tensor] = {}
            for field_info in fields(StructuredGpuRewardPartBuffers):
                field_name = str(field_info.name)
                reward_part_kwargs[field_name] = self._ensure_history_tensor_buffer(
                    None if existing_reward_parts is None else getattr(existing_reward_parts, field_name),
                    getattr(result_reward_parts, field_name),
                    capacity=capacity,
                    field_name=f"bw_stage.reward_parts.{field_name}",
                    dtype=torch.float32,
                )
            bw_stage.reward_parts = StructuredGpuRewardPartBuffers(**reward_part_kwargs)
            bw_stage.reward_part_tensors = {
                field_name: getattr(bw_stage.reward_parts, field_name) for field_name in _BW_REWARD_PART_KEYS
            }
        else:
            bw_stage.reward_parts = None
            bw_stage.reward_part_tensors = None
        self.bind_horizon_step_result_views(
            num_steps=capacity,
            num_envs=int(history.num_envs),
            expose_reward_parts=bw_stage.reward_part_tensors is not None,
        )

    def begin_native_main_kernel_training_write(self, *, num_envs: int) -> int:
        slot = self._ensure_history_slot(num_envs=int(num_envs))
        if 0 <= int(slot) < len(self.result.step_result_views):
            self.main.step_result_view = self.result.step_result_views[int(slot)]
        return int(slot)

    def finish_native_main_kernel_training_write(self, *, slot: int) -> None:
        history = self.history
        slot_i = int(slot)
        history.cursor = slot_i + 1

