from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from .structured_types import LocalAccelState, LocalBwState, LocalSatState, StructuredWorldState


def _clone_rollout_value(value: Any) -> Any:
    """Own tensor/array rollout data when appending it to a buffer.

    Native and GPU rollout views can point into reusable runtime storage.  If we
    keep those references and only collate later, later rollout steps may mutate
    the inputs while scalar targets/logprobs remain unchanged.
    """
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if is_dataclass(value):
        field_names = getattr(value, "_tensor_fields", None)
        if field_names is None:
            field_names = tuple(field_info.name for field_info in fields(value))
        return type(value)(**{str(name): _clone_rollout_value(getattr(value, str(name))) for name in field_names})
    if isinstance(value, dict):
        return {key: _clone_rollout_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_rollout_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_rollout_value(item) for item in value)
    return copy.deepcopy(value)

@dataclass
class _EnvStepBatchRecord:
    env_indices: np.ndarray
    num_agents: int
    accel_world_batch: StructuredWorldState
    sat_world_batch: StructuredWorldState
    bw_world_batch: StructuredWorldState
    next_world_batch: StructuredWorldState
    accel_local_batch: Any
    sat_local_batch: Any
    bw_local_batch: Any
    accel_actions: torch.Tensor
    accel_latent_actions: torch.Tensor | None
    sat_actions: torch.Tensor
    bw_actions: torch.Tensor
    accel_old_logprobs: torch.Tensor
    sat_old_logprobs: torch.Tensor
    sat_old_logprobs_per_agent: torch.Tensor | None
    bw_old_logprobs: torch.Tensor
    accel_values: torch.Tensor
    sat_values: torch.Tensor
    bw_values: torch.Tensor
    rewards: torch.Tensor | np.ndarray
    terminated: torch.Tensor | np.ndarray
    truncated: torch.Tensor | np.ndarray
    accel_danger_targets: torch.Tensor | None
    accel_danger_masks: torch.Tensor | None
    sat_stage_states: list[Dict[str, Any] | None]
    bw_stage_states: list[Dict[str, Any] | None]
    bw_access_rewards: torch.Tensor | np.ndarray
    bw_weighted_workload_delta_rewards: torch.Tensor | np.ndarray
    bw_weighted_workload_level_rewards: torch.Tensor | np.ndarray
    bw_gu_queue_level_rewards: torch.Tensor | np.ndarray
    bw_system_queue_level_rewards: torch.Tensor | np.ndarray
    bw_gu_service_queue_rewards: torch.Tensor | np.ndarray
    bw_flow_proxy_scores: torch.Tensor | None
    bw_flow_proxy_masks: torch.Tensor | None
    bw_flow_proxy_deltas: torch.Tensor | None
    bw_ref_actions: torch.Tensor | None
    bw_old_logprobs_per_agent: torch.Tensor | None

    @property
    def num_envs(self) -> int:
        return int(self.env_indices.shape[0])

    @property
    def transition_count(self) -> int:
        return 3 * self.num_envs


@dataclass
class StructuredStageTrainingBatch:
    stage_id: int
    transition_indices: np.ndarray
    env_indices: np.ndarray
    world_batch: StructuredWorldState
    local_batch: Any
    actions: torch.Tensor
    old_logprobs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    latent_actions: torch.Tensor | None = None
    danger_imitation_targets: torch.Tensor | None = None
    danger_imitation_masks: torch.Tensor | None = None
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
    bw_tau: torch.Tensor | None = None
    bw_kappa: torch.Tensor | None = None
    bw_valid_count: torch.Tensor | None = None
    bw_latent_count: torch.Tensor | None = None
    bw_logprob_raw_per_agent: torch.Tensor | None = None
    old_logprobs_per_agent: torch.Tensor | None = None
    sat_action_indices: torch.Tensor | None = None
    sat_stage_states: list[Dict[str, Any] | None] | None = None
    bw_stage_states: list[Dict[str, Any] | None] | None = None
    duration: torch.Tensor | None = None

    @property
    def num_samples(self) -> int:
        return int(self.actions.shape[0])

    @property
    def num_agents(self) -> int:
        return int(self.actions.shape[1]) if self.actions.ndim >= 2 else 1


@dataclass
class StructuredTrainingBatchView:
    transition_count: int
    stage_ids: np.ndarray
    old_logprobs: torch.Tensor
    values: torch.Tensor
    stage_batches: dict[int, StructuredStageTrainingBatch]


@dataclass
class StructuredStageReturnBatch:
    stage_id: int
    transition_indices: np.ndarray
    env_indices: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    bw_access_rewards: np.ndarray
    bw_weighted_workload_delta_rewards: np.ndarray
    bw_weighted_workload_level_rewards: np.ndarray
    bw_gu_queue_level_rewards: np.ndarray
    bw_system_queue_level_rewards: np.ndarray
    bw_gu_service_queue_rewards: np.ndarray
    reward_part_arrays: dict[str, np.ndarray]
    next_world_batch: StructuredWorldState | None
    duration: np.ndarray | None = None

    @property
    def num_samples(self) -> int:
        return int(self.transition_indices.shape[0])


@dataclass
class StructuredReturnBatchView:
    transition_count: int
    stage_ids: np.ndarray
    env_indices: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    bw_access_rewards: np.ndarray
    bw_weighted_workload_delta_rewards: np.ndarray
    bw_weighted_workload_level_rewards: np.ndarray
    bw_gu_queue_level_rewards: np.ndarray
    bw_system_queue_level_rewards: np.ndarray
    bw_gu_service_queue_rewards: np.ndarray
    reward_part_arrays: dict[str, np.ndarray] = field(default_factory=dict)
    stage_batches: dict[int, StructuredStageReturnBatch] = field(default_factory=dict)


@dataclass
class StructuredBootstrapBatchView:
    env_indices: np.ndarray
    next_world_batch: StructuredWorldState

    @property
    def num_envs(self) -> int:
        return int(self.env_indices.shape[0])


@dataclass
class StructuredRolloutViews:
    training_view: StructuredTrainingBatchView
    return_view: StructuredReturnBatchView
    bootstrap_view: StructuredBootstrapBatchView | None


@dataclass
class StructuredNativeRolloutTrainingBatchView:
    history: Any
    num_steps: int
    num_envs: int
    access_bw_decision_interval: int = 1
    sat_decision_interval: int = 1
    gamma_env: float = 1.0

    @property
    def transition_count(self) -> int:
        return 3 * int(self.num_steps) * int(self.num_envs)


def _padcat_tensors(values: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
    if not values:
        raise ValueError("values must be non-empty")
    ndim = values[0].ndim
    if any(v.ndim != ndim for v in values):
        raise ValueError("all tensors must share ndim")
    ref_shape = values[0].shape[1:]
    same_shape = all(tuple(v.shape[1:]) == tuple(ref_shape) for v in values)
    if same_shape:
        converted = [v if v.device == device else v.to(device) for v in values]
        return torch.cat(converted, dim=0)
    total_batch = int(sum(int(v.shape[0]) for v in values))
    max_shape = [max(int(v.shape[d]) for v in values) for d in range(1, ndim)]
    out_shape = [total_batch] + max_shape
    out = torch.zeros(out_shape, dtype=values[0].dtype, device=device)
    cursor = 0
    for value in values:
        value = value.to(device)
        batch = int(value.shape[0])
        slices = [slice(cursor, cursor + batch)] + [slice(0, int(size)) for size in value.shape[1:]]
        out[tuple(slices)] = value
        cursor += batch
    return out


def _field_to_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    raise TypeError(f"Unsupported field type for structured collation: {type(value)!r}")


def _as_record_float_tensor(
    value: Sequence[float] | np.ndarray | torch.Tensor | None,
    *,
    num_envs: int,
    device: torch.device,
    clone: bool = True,
) -> torch.Tensor:
    if value is None:
        return torch.zeros((int(num_envs),), dtype=torch.float32, device=device)
    if torch.is_tensor(value):
        out = value.detach().to(device=device, dtype=torch.float32).reshape(int(num_envs))
        return out.clone() if clone else out
    out = torch.as_tensor(value, dtype=torch.float32, device=device).reshape(int(num_envs))
    return out.clone() if clone else out


def _as_record_bool_tensor(
    value: Sequence[bool] | np.ndarray | torch.Tensor,
    *,
    num_envs: int,
    device: torch.device,
    clone: bool = True,
) -> torch.Tensor:
    if torch.is_tensor(value):
        out = value.detach().to(device=device, dtype=torch.bool).reshape(int(num_envs))
        return out.clone() if clone else out
    out = torch.as_tensor(value, dtype=torch.bool, device=device).reshape(int(num_envs))
    return out.clone() if clone else out


def _record_float_tensor(value: torch.Tensor | np.ndarray, *, num_envs: int, device: torch.device) -> torch.Tensor:
    return _field_to_tensor(value).to(device=device, dtype=torch.float32).reshape(int(num_envs))


def _record_bool_tensor(value: torch.Tensor | np.ndarray, *, num_envs: int, device: torch.device) -> torch.Tensor:
    return _field_to_tensor(value).to(device=device, dtype=torch.bool).reshape(int(num_envs))


def _record_float_numpy(value: torch.Tensor | np.ndarray, *, num_envs: int) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(np.float32, copy=False).reshape(int(num_envs))
    return np.asarray(value, dtype=np.float32).reshape(int(num_envs))


def _record_bool_numpy(value: torch.Tensor | np.ndarray, *, num_envs: int) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(bool, copy=False).reshape(int(num_envs))
    return np.asarray(value, dtype=bool).reshape(int(num_envs))


def _infer_dataclass_device(items: Sequence[Any]) -> torch.device:
    for item in items:
        item_device = getattr(item, "device", None)
        if item_device is not None:
            return torch.device(item_device)
        field_names = _tensor_field_names(item)
        if not field_names:
            continue
        for field_name in field_names:
            value = getattr(item, field_name)
            if torch.is_tensor(value):
                return value.device
            if isinstance(value, np.ndarray):
                return torch.device("cpu")
    return torch.device("cpu")


def _tensor_field_names(value: Any) -> tuple[str, ...]:
    explicit = getattr(value, "_tensor_fields", None)
    if explicit is not None:
        return tuple(str(name) for name in explicit)
    namedtuple_fields = getattr(value, "_fields", None)
    if namedtuple_fields is not None:
        return tuple(str(name) for name in namedtuple_fields)
    if is_dataclass(value):
        return tuple(str(field.name) for field in fields(value))
    return ()


def _collate_dataclass(items: Sequence[Any], device: torch.device) -> Any:
    if not items:
        raise ValueError("items must be non-empty")
    sample = items[0]
    materialize_many = getattr(type(sample), "materialize_many", None)
    if callable(materialize_many):
        return materialize_many(items, device=device)
    materialize = getattr(sample, "materialize", None)
    if callable(materialize):
        items = [item.materialize(device=device) for item in items]
        sample = items[0]
    field_names = _tensor_field_names(sample)
    if not field_names:
        raise TypeError("_collate_dataclass expects fixed tensor-field objects")
    kwargs: dict[str, Any] = {}
    for field_name in field_names:
        values = [getattr(item, field_name) for item in items]
        tensor_values = [_field_to_tensor(value) for value in values]
        kwargs[field_name] = _padcat_tensors(tensor_values, device)
    return type(sample)(**kwargs)


def _index_dataclass_items(batch: Any, indices: Sequence[int] | np.ndarray | torch.Tensor) -> Any:
    field_names = _tensor_field_names(batch)
    if not field_names:
        raise TypeError("_index_dataclass_items expects a fixed tensor-field object")
    if torch.is_tensor(indices):
        indices_arr = indices.to(dtype=torch.long)
    else:
        indices_arr = np.asarray(indices, dtype=np.int64).reshape(-1)
    kwargs: dict[str, Any] = {}
    for field_name in field_names:
        value = getattr(batch, field_name)
        if value is None:
            kwargs[field_name] = None
            continue
        if torch.is_tensor(value):
            if torch.is_tensor(indices_arr):
                idx_t = indices_arr.to(device=value.device, dtype=torch.long)
            else:
                idx_t = torch.as_tensor(indices_arr, device=value.device, dtype=torch.long)
            kwargs[field_name] = value.index_select(0, idx_t)
            continue
        if isinstance(value, np.ndarray):
            idx_np = indices_arr.detach().cpu().numpy() if torch.is_tensor(indices_arr) else indices_arr
            kwargs[field_name] = value[idx_np]
            continue
        raise TypeError(f"Unsupported field type for dataclass indexing: {type(value)!r}")
    return type(batch)(**kwargs)


def _concat_tensor_batches(values: Sequence[torch.Tensor | np.ndarray], device: torch.device) -> torch.Tensor:
    if not values:
        raise ValueError("values must be non-empty")
    tensor_values = [_field_to_tensor(value) for value in values]
    return _padcat_tensors(tensor_values, device)


def _concat_optional_tensor_batches(
    values: Sequence[torch.Tensor | np.ndarray | None],
    *,
    device: torch.device,
    field_name: str,
) -> torch.Tensor | None:
    if not values:
        return None
    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(present) != len(values):
        raise RuntimeError(f"{field_name} entries must be all-present or all-absent.")
    return _concat_tensor_batches(present, device)


def _flatten_history_tensor(
    value: torch.Tensor | None,
    *,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    if value is None:
        return None
    if not torch.is_tensor(value):
        raise TypeError(f"native rollout history value must be a tensor, got {type(value)!r}.")
    prefix = value[: int(num_steps)]
    if prefix.ndim >= 2:
        flat = prefix.reshape(int(prefix.shape[0]) * int(prefix.shape[1]), *prefix.shape[2:])
    else:
        flat = prefix.reshape(-1)
    if dtype is not None:
        flat = flat.to(dtype=dtype)
    return flat if flat.device == device else flat.to(device=device)


def _slice_flat_history_dataclass(
    value: Any,
    *,
    num_steps: int,
    capacity: int,
    device: torch.device,
    start_slot: int = 0,
) -> Any:
    if value is None:
        return None
    field_names = _tensor_field_names(value)
    if not field_names:
        raise TypeError("_slice_flat_history_dataclass expects a fixed tensor-field object.")
    kwargs: dict[str, Any] = {}
    for field_name in field_names:
        tensor = getattr(value, field_name)
        if tensor is None:
            kwargs[field_name] = None
            continue
        per_step = int(tensor.shape[0]) // max(int(capacity), 1)
        start = max(int(start_slot), 0) * int(per_step)
        end = start + int(num_steps) * int(per_step)
        prefix = tensor[start:end]
        kwargs[field_name] = prefix if prefix.device == device else prefix.to(device=device)
    return type(value)(**kwargs)


def _world_state_from_flat_history_ring(
    value: Any,
    *,
    num_steps: int,
    capacity: int,
    stage_id: int,
    device: torch.device,
    start_slot: int = 0,
) -> StructuredWorldState:
    world_fields = _slice_flat_history_dataclass(
        value,
        num_steps=num_steps,
        capacity=capacity,
        device=device,
        start_slot=start_slot,
    )
    if world_fields is None:
        raise RuntimeError("native rollout training ring is missing world tensors.")
    total_rows = int(world_fields.uav_nodes.shape[0])
    stage_id_t = torch.full((total_rows,), int(stage_id), dtype=torch.long, device=device)
    return StructuredWorldState(
        uav_nodes=world_fields.uav_nodes,
        gu_nodes=world_fields.gu_nodes,
        sat_nodes=world_fields.sat_nodes,
        sat_ids=world_fields.sat_ids,
        uav_gu_edges=world_fields.uav_gu_edges,
        uav_sat_edges=world_fields.uav_sat_edges,
        uav_uav_edges=world_fields.uav_uav_edges,
        global_scalars=world_fields.global_scalars,
        gu_mask=world_fields.gu_mask,
        sat_mask=world_fields.sat_mask,
        uav_gu_mask=world_fields.uav_gu_mask,
        uav_sat_mask=world_fields.uav_sat_mask,
        uav_uav_mask=world_fields.uav_uav_mask,
        stage_id=stage_id_t,
    )


def _where_world_state(mask: torch.Tensor, true_world: StructuredWorldState, false_world: StructuredWorldState) -> StructuredWorldState:
    if mask.ndim != 1:
        mask = mask.reshape(-1)

    def _select(field_name: str) -> torch.Tensor:
        true_value = getattr(true_world, field_name)
        false_value = getattr(false_world, field_name)
        if not torch.is_tensor(true_value) or not torch.is_tensor(false_value):
            raise RuntimeError("native transition next-world resolution requires tensor world fields.")
        view_shape = (int(mask.shape[0]),) + (1,) * (true_value.ndim - 1)
        return torch.where(mask.to(device=false_value.device, dtype=torch.bool).view(view_shape), true_value, false_value)

    return StructuredWorldState(
        uav_nodes=_select("uav_nodes"),
        gu_nodes=_select("gu_nodes"),
        sat_nodes=_select("sat_nodes"),
        sat_ids=_select("sat_ids"),
        uav_gu_edges=_select("uav_gu_edges"),
        uav_sat_edges=_select("uav_sat_edges"),
        uav_uav_edges=_select("uav_uav_edges"),
        global_scalars=_select("global_scalars"),
        gu_mask=_select("gu_mask"),
        sat_mask=_select("sat_mask"),
        uav_gu_mask=_select("uav_gu_mask"),
        uav_sat_mask=_select("uav_sat_mask"),
        uav_uav_mask=_select("uav_uav_mask"),
        stage_id=false_world.stage_id,
    )


def _flat_history_field(
    value: torch.Tensor | None,
    *,
    num_steps: int,
    capacity: int,
    device: torch.device,
) -> torch.Tensor | None:
    if value is None:
        return None
    per_step = int(value.shape[0]) // max(int(capacity), 1)
    prefix = value[: int(num_steps) * int(per_step)]
    return prefix if prefix.device == device else prefix.to(device=device)


def _accel_local_from_flat_history_ring(
    stage: Any,
    *,
    num_steps: int,
    capacity: int,
    device: torch.device,
) -> LocalAccelState:
    return LocalAccelState(
        ego_features=_flat_history_field(stage.ego_features, num_steps=num_steps, capacity=capacity, device=device),
        ego_cell=_flat_history_field(stage.ego_cell, num_steps=num_steps, capacity=capacity, device=device),
        gu_tokens=_flat_history_field(stage.gu_tokens, num_steps=num_steps, capacity=capacity, device=device),
        gu_mask=_flat_history_field(stage.gu_mask, num_steps=num_steps, capacity=capacity, device=device),
        peer_tokens=_flat_history_field(stage.peer_tokens, num_steps=num_steps, capacity=capacity, device=device),
        peer_mask=_flat_history_field(stage.peer_mask, num_steps=num_steps, capacity=capacity, device=device),
        sat_tokens=_flat_history_field(stage.sat_tokens, num_steps=num_steps, capacity=capacity, device=device),
        sat_mask=_flat_history_field(stage.sat_mask, num_steps=num_steps, capacity=capacity, device=device),
    )


def _sat_local_from_flat_history_ring(
    stage: Any,
    *,
    num_steps: int,
    capacity: int,
    device: torch.device,
) -> LocalSatState:
    return LocalSatState(
        ego_features=_flat_history_field(stage.ego_features, num_steps=num_steps, capacity=capacity, device=device),
        demand_features=_flat_history_field(stage.demand_features, num_steps=num_steps, capacity=capacity, device=device),
        role_features=_flat_history_field(stage.role_features, num_steps=num_steps, capacity=capacity, device=device),
        sat_tokens=_flat_history_field(stage.sat_tokens, num_steps=num_steps, capacity=capacity, device=device),
        sat_mask=_flat_history_field(stage.sat_mask, num_steps=num_steps, capacity=capacity, device=device),
        sat_valid_mask=_flat_history_field(stage.sat_valid_mask, num_steps=num_steps, capacity=capacity, device=device),
        candidate_sat_ids=_flat_history_field(stage.candidate_sat_ids, num_steps=num_steps, capacity=capacity, device=device),
        subset_mask=_flat_history_field(stage.subset_mask, num_steps=num_steps, capacity=capacity, device=device),
        subset_members=_flat_history_field(stage.subset_members, num_steps=num_steps, capacity=capacity, device=device),
    )


def _bw_local_from_flat_history_ring(
    stage: Any,
    *,
    num_steps: int,
    capacity: int,
    device: torch.device,
) -> LocalBwState:
    return LocalBwState(
        ego_features=_flat_history_field(stage.ego_features, num_steps=num_steps, capacity=capacity, device=device),
        selected_sat_tokens=_flat_history_field(stage.selected_sat_tokens, num_steps=num_steps, capacity=capacity, device=device),
        selected_sat_mask=_flat_history_field(stage.selected_sat_mask, num_steps=num_steps, capacity=capacity, device=device),
        gu_tokens=_flat_history_field(stage.gu_tokens, num_steps=num_steps, capacity=capacity, device=device),
        gu_mask=_flat_history_field(stage.gu_mask, num_steps=num_steps, capacity=capacity, device=device),
        bw_valid_mask=_flat_history_field(stage.bw_valid_mask, num_steps=num_steps, capacity=capacity, device=device),
    )


def _history_float_numpy(value: torch.Tensor | None, *, num_steps: int, num_envs: int) -> np.ndarray:
    if value is None:
        return np.zeros((int(num_steps) * int(num_envs),), dtype=np.float32)
    flat = value[: int(num_steps)].reshape(int(num_steps) * int(num_envs), *value.shape[2:])
    return flat.detach().cpu().numpy().astype(np.float32, copy=False).reshape(int(num_steps) * int(num_envs))


def _history_bool_numpy(value: torch.Tensor | None, *, num_steps: int, num_envs: int) -> np.ndarray:
    if value is None:
        return np.zeros((int(num_steps) * int(num_envs),), dtype=bool)
    flat = value[: int(num_steps)].reshape(int(num_steps) * int(num_envs), *value.shape[2:])
    return flat.detach().cpu().numpy().astype(bool, copy=False).reshape(int(num_steps) * int(num_envs))


def _interleave_stage_tensors(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.stack((a, b, c), dim=2).reshape(-1, *a.shape[2:])


def _interleave_stage_numpy(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.stack((a, b, c), axis=2).reshape(-1)


def _empty_training_batch_view(device: torch.device) -> StructuredTrainingBatchView:
    empty = torch.zeros((0,), dtype=torch.float32, device=device)
    return StructuredTrainingBatchView(
        transition_count=0,
        stage_ids=np.zeros((0,), dtype=np.int64),
        old_logprobs=empty,
        values=empty.clone(),
        stage_batches={},
    )


def _empty_return_batch_view() -> StructuredReturnBatchView:
    empty_float = np.zeros((0,), dtype=np.float32)
    empty_int = np.zeros((0,), dtype=np.int64)
    empty_bool = np.zeros((0,), dtype=bool)
    return StructuredReturnBatchView(
        transition_count=0,
        stage_ids=empty_int.copy(),
        env_indices=empty_int.copy(),
        values=empty_float.copy(),
        rewards=empty_float.copy(),
        terminated=empty_bool.copy(),
        truncated=empty_bool.copy(),
        bw_access_rewards=empty_float.copy(),
        bw_weighted_workload_delta_rewards=empty_float.copy(),
        bw_weighted_workload_level_rewards=empty_float.copy(),
        bw_gu_queue_level_rewards=empty_float.copy(),
        bw_system_queue_level_rewards=empty_float.copy(),
        bw_gu_service_queue_rewards=empty_float.copy(),
        reward_part_arrays={},
        stage_batches={},
    )


def _new_stage_training_parts() -> dict[str, Any]:
    return {
        "transition_indices": [],
        "env_indices": [],
        "world_batches": [],
        "local_batches": [],
        "actions": [],
        "latent_actions": [],
        "old_logprobs": [],
        "values": [],
        "rewards": [],
        "terminated": [],
        "truncated": [],
        "danger_imitation_targets": [],
        "danger_imitation_masks": [],
        "bw_access_rewards": [],
        "bw_weighted_workload_delta_rewards": [],
        "bw_weighted_workload_level_rewards": [],
        "bw_gu_queue_level_rewards": [],
        "bw_system_queue_level_rewards": [],
        "bw_gu_service_queue_rewards": [],
        "bw_flow_proxy_scores": [],
        "bw_flow_proxy_masks": [],
        "bw_flow_proxy_deltas": [],
        "bw_ref_actions": [],
        "old_logprobs_per_agent": [],
        "sat_stage_states": [],
        "bw_stage_states": [],
    }


def _new_stage_return_parts() -> dict[str, Any]:
    return {
        "transition_indices": [],
        "env_indices": [],
        "values": [],
        "rewards": [],
        "terminated": [],
        "truncated": [],
        "bw_access_rewards": [],
        "bw_weighted_workload_delta_rewards": [],
        "bw_weighted_workload_level_rewards": [],
        "bw_gu_queue_level_rewards": [],
        "bw_system_queue_level_rewards": [],
        "bw_gu_service_queue_rewards": [],
        "reward_part_arrays": {},
        "next_world_batches": [],
    }


def _append_stage_training_part(
    parts: dict[str, Any],
    *,
    transition_indices: np.ndarray,
    env_indices: np.ndarray,
    world_batch: StructuredWorldState,
    local_batch: Any,
    actions: torch.Tensor,
    old_logprobs: torch.Tensor | np.ndarray,
    values: torch.Tensor | np.ndarray,
    rewards: torch.Tensor | np.ndarray,
    terminated: torch.Tensor | np.ndarray,
    truncated: torch.Tensor | np.ndarray,
    latent_actions: torch.Tensor | np.ndarray | None = None,
    danger_imitation_targets: torch.Tensor | np.ndarray | None = None,
    danger_imitation_masks: torch.Tensor | np.ndarray | None = None,
    bw_access_rewards: torch.Tensor | np.ndarray | None = None,
    bw_weighted_workload_delta_rewards: torch.Tensor | np.ndarray | None = None,
    bw_weighted_workload_level_rewards: torch.Tensor | np.ndarray | None = None,
    bw_gu_queue_level_rewards: torch.Tensor | np.ndarray | None = None,
    bw_system_queue_level_rewards: torch.Tensor | np.ndarray | None = None,
    bw_gu_service_queue_rewards: torch.Tensor | np.ndarray | None = None,
    bw_flow_proxy_scores: torch.Tensor | np.ndarray | None = None,
    bw_flow_proxy_masks: torch.Tensor | np.ndarray | None = None,
    bw_flow_proxy_deltas: torch.Tensor | np.ndarray | None = None,
    bw_ref_actions: torch.Tensor | np.ndarray | None = None,
    old_logprobs_per_agent: torch.Tensor | np.ndarray | None = None,
    sat_stage_states: Sequence[Dict[str, Any] | None] | None = None,
    bw_stage_states: Sequence[Dict[str, Any] | None] | None = None,
) -> None:
    parts["transition_indices"].append(np.asarray(transition_indices, dtype=np.int64).reshape(-1))
    parts["env_indices"].append(np.asarray(env_indices, dtype=np.int64).reshape(-1))
    parts["world_batches"].append(world_batch)
    parts["local_batches"].append(local_batch)
    parts["actions"].append(actions)
    parts["latent_actions"].append(latent_actions)
    parts["old_logprobs"].append(old_logprobs)
    parts["values"].append(values)
    parts["rewards"].append(rewards)
    parts["terminated"].append(terminated)
    parts["truncated"].append(truncated)
    parts["danger_imitation_targets"].append(danger_imitation_targets)
    parts["danger_imitation_masks"].append(danger_imitation_masks)
    parts["bw_access_rewards"].append(bw_access_rewards)
    parts["bw_weighted_workload_delta_rewards"].append(bw_weighted_workload_delta_rewards)
    parts["bw_weighted_workload_level_rewards"].append(bw_weighted_workload_level_rewards)
    parts["bw_gu_queue_level_rewards"].append(bw_gu_queue_level_rewards)
    parts["bw_system_queue_level_rewards"].append(bw_system_queue_level_rewards)
    parts["bw_gu_service_queue_rewards"].append(bw_gu_service_queue_rewards)
    parts["bw_flow_proxy_scores"].append(bw_flow_proxy_scores)
    parts["bw_flow_proxy_masks"].append(bw_flow_proxy_masks)
    parts["bw_flow_proxy_deltas"].append(bw_flow_proxy_deltas)
    parts["bw_ref_actions"].append(bw_ref_actions)
    parts["old_logprobs_per_agent"].append(old_logprobs_per_agent)
    if sat_stage_states is not None:
        parts["sat_stage_states"].extend(list(sat_stage_states))
    if bw_stage_states is not None:
        parts["bw_stage_states"].extend(list(bw_stage_states))


def _append_stage_return_part(
    parts: dict[str, Any],
    *,
    transition_indices: np.ndarray,
    env_indices: np.ndarray,
    values: np.ndarray,
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    bw_access_rewards: np.ndarray,
    bw_weighted_workload_delta_rewards: np.ndarray,
    bw_weighted_workload_level_rewards: np.ndarray,
    bw_gu_queue_level_rewards: np.ndarray,
    bw_system_queue_level_rewards: np.ndarray,
    bw_gu_service_queue_rewards: np.ndarray,
    reward_part_arrays: dict[str, np.ndarray] | None = None,
    next_world_batch: StructuredWorldState,
) -> None:
    parts["transition_indices"].append(np.asarray(transition_indices, dtype=np.int64).reshape(-1))
    parts["env_indices"].append(np.asarray(env_indices, dtype=np.int64).reshape(-1))
    parts["values"].append(np.asarray(values, dtype=np.float32).reshape(-1))
    parts["rewards"].append(np.asarray(rewards, dtype=np.float32).reshape(-1))
    parts["terminated"].append(np.asarray(terminated, dtype=bool).reshape(-1))
    parts["truncated"].append(np.asarray(truncated, dtype=bool).reshape(-1))
    parts["bw_access_rewards"].append(np.asarray(bw_access_rewards, dtype=np.float32).reshape(-1))
    parts["bw_weighted_workload_delta_rewards"].append(
        np.asarray(bw_weighted_workload_delta_rewards, dtype=np.float32).reshape(-1)
    )
    parts["bw_weighted_workload_level_rewards"].append(
        np.asarray(bw_weighted_workload_level_rewards, dtype=np.float32).reshape(-1)
    )
    parts["bw_gu_queue_level_rewards"].append(np.asarray(bw_gu_queue_level_rewards, dtype=np.float32).reshape(-1))
    parts["bw_system_queue_level_rewards"].append(
        np.asarray(bw_system_queue_level_rewards, dtype=np.float32).reshape(-1)
    )
    parts["bw_gu_service_queue_rewards"].append(
        np.asarray(bw_gu_service_queue_rewards, dtype=np.float32).reshape(-1)
    )
    if reward_part_arrays:
        storage = parts.setdefault("reward_part_arrays", {})
        for key, value in reward_part_arrays.items():
            storage.setdefault(str(key), []).append(np.asarray(value, dtype=np.float32).reshape(-1))
    parts["next_world_batches"].append(next_world_batch)


def _finalize_stage_training_batch(
    stage_id: int,
    parts: dict[str, Any],
    *,
    device: torch.device,
) -> StructuredStageTrainingBatch | None:
    if not parts["transition_indices"]:
        return None
    return StructuredStageTrainingBatch(
        stage_id=int(stage_id),
        transition_indices=np.concatenate(parts["transition_indices"]).astype(np.int64, copy=False),
        env_indices=np.concatenate(parts["env_indices"]).astype(np.int64, copy=False),
        world_batch=_collate_dataclass(parts["world_batches"], device),
        local_batch=_collate_dataclass(parts["local_batches"], device),
        actions=_concat_tensor_batches(parts["actions"], device),
        latent_actions=_concat_optional_tensor_batches(
            parts["latent_actions"],
            device=device,
            field_name="latent_action",
        ),
        old_logprobs=_concat_tensor_batches(parts["old_logprobs"], device).reshape(-1),
        values=_concat_tensor_batches(parts["values"], device).reshape(-1),
        rewards=_concat_tensor_batches(parts["rewards"], device).reshape(-1),
        terminated=_concat_tensor_batches(parts["terminated"], device).reshape(-1).to(dtype=torch.bool),
        truncated=_concat_tensor_batches(parts["truncated"], device).reshape(-1).to(dtype=torch.bool),
        danger_imitation_targets=_concat_optional_tensor_batches(
            parts["danger_imitation_targets"],
            device=device,
            field_name="danger_imitation_target",
        ),
        danger_imitation_masks=_concat_optional_tensor_batches(
            parts["danger_imitation_masks"],
            device=device,
            field_name="danger_imitation_mask",
        ),
        bw_access_rewards=_concat_optional_tensor_batches(
            parts["bw_access_rewards"],
            device=device,
            field_name="bw_access_reward",
        ),
        bw_weighted_workload_delta_rewards=_concat_optional_tensor_batches(
            parts["bw_weighted_workload_delta_rewards"],
            device=device,
            field_name="bw_weighted_workload_delta_reward",
        ),
        bw_weighted_workload_level_rewards=_concat_optional_tensor_batches(
            parts["bw_weighted_workload_level_rewards"],
            device=device,
            field_name="bw_weighted_workload_level_reward",
        ),
        bw_gu_queue_level_rewards=_concat_optional_tensor_batches(
            parts["bw_gu_queue_level_rewards"],
            device=device,
            field_name="bw_gu_queue_level_reward",
        ),
        bw_system_queue_level_rewards=_concat_optional_tensor_batches(
            parts["bw_system_queue_level_rewards"],
            device=device,
            field_name="bw_system_queue_level_reward",
        ),
        bw_gu_service_queue_rewards=_concat_optional_tensor_batches(
            parts["bw_gu_service_queue_rewards"],
            device=device,
            field_name="bw_gu_service_queue_reward",
        ),
        bw_flow_proxy_scores=_concat_optional_tensor_batches(
            parts["bw_flow_proxy_scores"],
            device=device,
            field_name="bw_flow_proxy_scores",
        ),
        bw_flow_proxy_masks=_concat_optional_tensor_batches(
            parts["bw_flow_proxy_masks"],
            device=device,
            field_name="bw_flow_proxy_masks",
        ),
        bw_flow_proxy_deltas=_concat_optional_tensor_batches(
            parts["bw_flow_proxy_deltas"],
            device=device,
            field_name="bw_flow_proxy_deltas",
        ),
        bw_ref_actions=_concat_optional_tensor_batches(
            parts["bw_ref_actions"],
            device=device,
            field_name="bw_ref_action",
        ),
        old_logprobs_per_agent=_concat_optional_tensor_batches(
            parts["old_logprobs_per_agent"],
            device=device,
            field_name="old_logprob_per_agent",
        ),
        sat_stage_states=(list(parts["sat_stage_states"]) if parts["sat_stage_states"] else None),
        bw_stage_states=(list(parts["bw_stage_states"]) if parts["bw_stage_states"] else None),
    )


def _finalize_stage_return_batch(
    stage_id: int,
    parts: dict[str, Any],
) -> StructuredStageReturnBatch | None:
    if not parts["transition_indices"]:
        return None
    next_world_items = parts["next_world_batches"]
    next_world_batch = _collate_dataclass(next_world_items, _infer_dataclass_device(next_world_items))
    reward_part_arrays = {
        str(key): np.concatenate(values).astype(np.float32, copy=False)
        for key, values in parts.get("reward_part_arrays", {}).items()
        if values
    }
    return StructuredStageReturnBatch(
        stage_id=int(stage_id),
        transition_indices=np.concatenate(parts["transition_indices"]).astype(np.int64, copy=False),
        env_indices=np.concatenate(parts["env_indices"]).astype(np.int64, copy=False),
        values=np.concatenate(parts["values"]).astype(np.float32, copy=False),
        rewards=np.concatenate(parts["rewards"]).astype(np.float32, copy=False),
        terminated=np.concatenate(parts["terminated"]).astype(bool, copy=False),
        truncated=np.concatenate(parts["truncated"]).astype(bool, copy=False),
        bw_access_rewards=np.concatenate(parts["bw_access_rewards"]).astype(np.float32, copy=False),
        bw_weighted_workload_delta_rewards=np.concatenate(
            parts["bw_weighted_workload_delta_rewards"]
        ).astype(np.float32, copy=False),
        bw_weighted_workload_level_rewards=np.concatenate(
            parts["bw_weighted_workload_level_rewards"]
        ).astype(np.float32, copy=False),
        bw_gu_queue_level_rewards=np.concatenate(parts["bw_gu_queue_level_rewards"]).astype(np.float32, copy=False),
        bw_system_queue_level_rewards=np.concatenate(
            parts["bw_system_queue_level_rewards"]
        ).astype(np.float32, copy=False),
        bw_gu_service_queue_rewards=np.concatenate(
            parts["bw_gu_service_queue_rewards"]
        ).astype(np.float32, copy=False),
        reward_part_arrays=reward_part_arrays,
        next_world_batch=next_world_batch,
    )

def _build_training_batch_view_from_records(
    records: Sequence[_EnvStepBatchRecord],
    *,
    device: torch.device,
) -> StructuredTrainingBatchView:
    if not records:
        return _empty_training_batch_view(device)
    stage_parts = {0: _new_stage_training_parts(), 1: _new_stage_training_parts(), 2: _new_stage_training_parts()}
    stage_ids_parts: list[np.ndarray] = []
    old_logprob_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    transition_cursor = 0
    for record in records:
        num_envs = int(record.num_envs)
        stage_ids_parts.append(np.tile(np.asarray([0, 1, 2], dtype=np.int64), num_envs))
        old_logprob_parts.append(
            torch.stack(
                (
                    record.accel_old_logprobs.reshape(num_envs),
                    record.sat_old_logprobs.reshape(num_envs),
                    record.bw_old_logprobs.reshape(num_envs),
                ),
                dim=1,
            ).reshape(-1)
        )
        value_parts.append(
            torch.stack(
                (
                    record.accel_values.reshape(num_envs),
                    record.sat_values.reshape(num_envs),
                    record.bw_values.reshape(num_envs),
                ),
                dim=1,
            ).reshape(-1)
        )
        base_indices = transition_cursor + 3 * np.arange(num_envs, dtype=np.int64)
        env_indices = record.env_indices.astype(np.int64, copy=False)
        record_device = record.accel_values.device
        rewards = _record_float_tensor(record.rewards, num_envs=num_envs, device=record_device)
        terminated = _record_bool_tensor(record.terminated, num_envs=num_envs, device=record_device)
        truncated = _record_bool_tensor(record.truncated, num_envs=num_envs, device=record_device)
        zeros = torch.zeros((num_envs,), dtype=torch.float32, device=record_device)
        _append_stage_training_part(
            stage_parts[0],
            transition_indices=base_indices,
            env_indices=env_indices,
            world_batch=record.accel_world_batch,
            local_batch=record.accel_local_batch,
            actions=record.accel_actions,
            latent_actions=record.accel_latent_actions,
            old_logprobs=record.accel_old_logprobs.reshape(num_envs),
            values=record.accel_values.reshape(num_envs),
            rewards=zeros,
            terminated=terminated,
            truncated=truncated,
            danger_imitation_targets=record.accel_danger_targets,
            danger_imitation_masks=record.accel_danger_masks,
        )
        _append_stage_training_part(
            stage_parts[1],
            transition_indices=base_indices + 1,
            env_indices=env_indices,
            world_batch=record.sat_world_batch,
            local_batch=record.sat_local_batch,
            actions=record.sat_actions,
            old_logprobs=record.sat_old_logprobs.reshape(num_envs),
            values=record.sat_values.reshape(num_envs),
            rewards=zeros,
            terminated=terminated,
            truncated=truncated,
            old_logprobs_per_agent=record.sat_old_logprobs_per_agent,
            sat_stage_states=record.sat_stage_states,
        )
        _append_stage_training_part(
            stage_parts[2],
            transition_indices=base_indices + 2,
            env_indices=env_indices,
            world_batch=record.bw_world_batch,
            local_batch=record.bw_local_batch,
            actions=record.bw_actions,
            old_logprobs=record.bw_old_logprobs.reshape(num_envs),
            values=record.bw_values.reshape(num_envs),
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            bw_access_rewards=_record_float_tensor(record.bw_access_rewards, num_envs=num_envs, device=record_device),
            bw_weighted_workload_delta_rewards=_record_float_tensor(record.bw_weighted_workload_delta_rewards, num_envs=num_envs, device=record_device),
            bw_weighted_workload_level_rewards=_record_float_tensor(record.bw_weighted_workload_level_rewards, num_envs=num_envs, device=record_device),
            bw_gu_queue_level_rewards=_record_float_tensor(record.bw_gu_queue_level_rewards, num_envs=num_envs, device=record_device),
            bw_system_queue_level_rewards=_record_float_tensor(record.bw_system_queue_level_rewards, num_envs=num_envs, device=record_device),
            bw_gu_service_queue_rewards=_record_float_tensor(record.bw_gu_service_queue_rewards, num_envs=num_envs, device=record_device),
            bw_flow_proxy_scores=record.bw_flow_proxy_scores,
            bw_flow_proxy_masks=record.bw_flow_proxy_masks,
            bw_flow_proxy_deltas=record.bw_flow_proxy_deltas,
            bw_ref_actions=record.bw_ref_actions,
            old_logprobs_per_agent=record.bw_old_logprobs_per_agent,
            bw_stage_states=record.bw_stage_states,
        )
        transition_cursor += int(record.transition_count)
    finalized = {
        stage_id: batch
        for stage_id in (0, 1, 2)
        if (batch := _finalize_stage_training_batch(stage_id, stage_parts[stage_id], device=device)) is not None
    }
    return StructuredTrainingBatchView(
        transition_count=int(transition_cursor),
        stage_ids=np.concatenate(stage_ids_parts).astype(np.int64, copy=False),
        old_logprobs=_concat_tensor_batches(old_logprob_parts, device).reshape(-1),
        values=_concat_tensor_batches(value_parts, device).reshape(-1),
        stage_batches=finalized,
    )


def _build_return_batch_view_from_records(
    records: Sequence[_EnvStepBatchRecord],
) -> StructuredReturnBatchView:
    if not records:
        return _empty_return_batch_view()
    stage_parts = {0: _new_stage_return_parts(), 1: _new_stage_return_parts(), 2: _new_stage_return_parts()}
    stage_ids_parts: list[np.ndarray] = []
    env_indices_parts: list[np.ndarray] = []
    values_parts: list[np.ndarray] = []
    rewards_parts: list[np.ndarray] = []
    terminated_parts: list[np.ndarray] = []
    truncated_parts: list[np.ndarray] = []
    bw_access_parts: list[np.ndarray] = []
    bw_weighted_delta_parts: list[np.ndarray] = []
    bw_weighted_level_parts: list[np.ndarray] = []
    bw_gu_queue_parts: list[np.ndarray] = []
    bw_system_queue_parts: list[np.ndarray] = []
    bw_gu_service_parts: list[np.ndarray] = []
    transition_cursor = 0
    for record in records:
        num_envs = int(record.num_envs)
        env_indices = record.env_indices.astype(np.int64, copy=False)
        rewards = _record_float_numpy(record.rewards, num_envs=num_envs)
        terminated = _record_bool_numpy(record.terminated, num_envs=num_envs)
        truncated = _record_bool_numpy(record.truncated, num_envs=num_envs)
        bw_access = _record_float_numpy(record.bw_access_rewards, num_envs=num_envs)
        bw_weighted_delta = _record_float_numpy(record.bw_weighted_workload_delta_rewards, num_envs=num_envs)
        bw_weighted_level = _record_float_numpy(record.bw_weighted_workload_level_rewards, num_envs=num_envs)
        bw_gu_queue = _record_float_numpy(record.bw_gu_queue_level_rewards, num_envs=num_envs)
        bw_system_queue = _record_float_numpy(record.bw_system_queue_level_rewards, num_envs=num_envs)
        bw_gu_service = _record_float_numpy(record.bw_gu_service_queue_rewards, num_envs=num_envs)
        stage_ids_parts.append(np.tile(np.asarray([0, 1, 2], dtype=np.int64), num_envs))
        env_indices_parts.append(np.repeat(env_indices, 3))
        values_parts.append(
            torch.stack(
                (
                    record.accel_values.reshape(num_envs),
                    record.sat_values.reshape(num_envs),
                    record.bw_values.reshape(num_envs),
                ),
                dim=1,
            ).detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
        )
        rewards_parts.append(
            np.stack(
                (
                    np.zeros((num_envs,), dtype=np.float32),
                    np.zeros((num_envs,), dtype=np.float32),
                    rewards,
                ),
                axis=1,
            ).reshape(-1).astype(np.float32, copy=False)
        )
        terminated_repeat = np.repeat(terminated, 3)
        truncated_repeat = np.repeat(truncated, 3)
        terminated_parts.append(terminated_repeat)
        truncated_parts.append(truncated_repeat)
        zeros = np.zeros((num_envs,), dtype=np.float32)
        bw_access_parts.append(
            np.stack((zeros, zeros, bw_access), axis=1)
            .reshape(-1)
            .astype(np.float32, copy=False)
        )
        bw_weighted_delta_parts.append(
            np.stack((zeros, zeros, bw_weighted_delta), axis=1)
            .reshape(-1)
            .astype(np.float32, copy=False)
        )
        bw_weighted_level_parts.append(
            np.stack((zeros, zeros, bw_weighted_level), axis=1)
            .reshape(-1)
            .astype(np.float32, copy=False)
        )
        bw_gu_queue_parts.append(
            np.stack((zeros, zeros, bw_gu_queue), axis=1)
            .reshape(-1)
            .astype(np.float32, copy=False)
        )
        bw_system_queue_parts.append(
            np.stack((zeros, zeros, bw_system_queue), axis=1)
            .reshape(-1)
            .astype(np.float32, copy=False)
        )
        bw_gu_service_parts.append(
            np.stack((zeros, zeros, bw_gu_service), axis=1)
            .reshape(-1)
            .astype(np.float32, copy=False)
        )
        base_indices = transition_cursor + 3 * np.arange(num_envs, dtype=np.int64)
        _append_stage_return_part(
            stage_parts[0],
            transition_indices=base_indices,
            env_indices=env_indices,
            values=record.accel_values.detach().cpu().numpy().reshape(num_envs).astype(np.float32, copy=False),
            rewards=zeros,
            terminated=terminated,
            truncated=truncated,
            bw_access_rewards=zeros,
            bw_weighted_workload_delta_rewards=zeros,
            bw_weighted_workload_level_rewards=zeros,
            bw_gu_queue_level_rewards=zeros,
            bw_system_queue_level_rewards=zeros,
            bw_gu_service_queue_rewards=zeros,
            next_world_batch=record.sat_world_batch,
        )
        _append_stage_return_part(
            stage_parts[1],
            transition_indices=base_indices + 1,
            env_indices=env_indices,
            values=record.sat_values.detach().cpu().numpy().reshape(num_envs).astype(np.float32, copy=False),
            rewards=zeros,
            terminated=terminated,
            truncated=truncated,
            bw_access_rewards=zeros,
            bw_weighted_workload_delta_rewards=zeros,
            bw_weighted_workload_level_rewards=zeros,
            bw_gu_queue_level_rewards=zeros,
            bw_system_queue_level_rewards=zeros,
            bw_gu_service_queue_rewards=zeros,
            next_world_batch=record.bw_world_batch,
        )
        _append_stage_return_part(
            stage_parts[2],
            transition_indices=base_indices + 2,
            env_indices=env_indices,
            values=record.bw_values.detach().cpu().numpy().reshape(num_envs).astype(np.float32, copy=False),
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            bw_access_rewards=bw_access,
            bw_weighted_workload_delta_rewards=bw_weighted_delta,
            bw_weighted_workload_level_rewards=bw_weighted_level,
            bw_gu_queue_level_rewards=bw_gu_queue,
            bw_system_queue_level_rewards=bw_system_queue,
            bw_gu_service_queue_rewards=bw_gu_service,
            next_world_batch=record.next_world_batch,
        )
        transition_cursor += int(record.transition_count)
    stage_batches = {
        stage_id: batch
        for stage_id in (0, 1, 2)
        if (batch := _finalize_stage_return_batch(stage_id, stage_parts[stage_id])) is not None
    }
    return StructuredReturnBatchView(
        transition_count=int(transition_cursor),
        stage_ids=np.concatenate(stage_ids_parts).astype(np.int64, copy=False),
        env_indices=np.concatenate(env_indices_parts).astype(np.int64, copy=False),
        values=np.concatenate(values_parts).astype(np.float32, copy=False),
        rewards=np.concatenate(rewards_parts).astype(np.float32, copy=False),
        terminated=np.concatenate(terminated_parts).astype(bool, copy=False),
        truncated=np.concatenate(truncated_parts).astype(bool, copy=False),
        bw_access_rewards=np.concatenate(bw_access_parts).astype(np.float32, copy=False),
        bw_weighted_workload_delta_rewards=np.concatenate(bw_weighted_delta_parts).astype(np.float32, copy=False),
        bw_weighted_workload_level_rewards=np.concatenate(bw_weighted_level_parts).astype(np.float32, copy=False),
        bw_gu_queue_level_rewards=np.concatenate(bw_gu_queue_parts).astype(np.float32, copy=False),
        bw_system_queue_level_rewards=np.concatenate(bw_system_queue_parts).astype(np.float32, copy=False),
        bw_gu_service_queue_rewards=np.concatenate(bw_gu_service_parts).astype(np.float32, copy=False),
        reward_part_arrays={},
        stage_batches=stage_batches,
    )


def _build_bootstrap_batch_view_from_return_view(
    return_view: StructuredReturnBatchView,
) -> StructuredBootstrapBatchView | None:
    bw_stage_batch = return_view.stage_batches.get(2)
    if bw_stage_batch is None or int(bw_stage_batch.num_samples) <= 0:
        return None
    env_indices = np.asarray(bw_stage_batch.env_indices, dtype=np.int64).reshape(-1)
    if env_indices.size <= 0:
        return None
    latest_index_by_env: dict[int, int] = {}
    for rev_pos in range(env_indices.size - 1, -1, -1):
        env_index = int(env_indices[rev_pos])
        if env_index not in latest_index_by_env:
            latest_index_by_env[env_index] = int(rev_pos)
    ordered_env_indices = np.asarray(sorted(latest_index_by_env.keys()), dtype=np.int64)
    selected_positions = np.asarray(
        [latest_index_by_env[int(env_index)] for env_index in ordered_env_indices.tolist()],
        dtype=np.int64,
    )
    next_world_batch = _index_dataclass_items(bw_stage_batch.next_world_batch, selected_positions)
    return StructuredBootstrapBatchView(
        env_indices=ordered_env_indices,
        next_world_batch=next_world_batch,
    )


def _build_rollout_views_from_native_training_ring(
    native_view: StructuredNativeRolloutTrainingBatchView,
    *,
    device: torch.device,
) -> StructuredRolloutViews:
    history = native_view.history
    accel_stage = history.accel_stage
    sat_stage = history.sat_stage
    bw_stage = history.bw_stage
    num_steps = max(int(native_view.num_steps), 0)
    num_envs = max(int(native_view.num_envs), 0)
    num_samples = int(num_steps) * int(num_envs)
    capacity = max(int(getattr(history, "capacity", 0) or 0), 1)
    if num_steps <= 0 or num_envs <= 0:
        return StructuredRolloutViews(
            training_view=_empty_training_batch_view(device),
            return_view=_empty_return_batch_view(),
            bootstrap_view=None,
        )
    if int(getattr(history, "cursor", 0) or 0) < num_steps:
        raise RuntimeError("native rollout history cursor is behind the recorded horizon.")
    env_indices_stage = np.tile(np.arange(num_envs, dtype=np.int64), num_steps)
    step_offsets = np.arange(num_steps, dtype=np.int64).reshape(num_steps, 1) * (3 * num_envs)
    env_offsets = 3 * np.arange(num_envs, dtype=np.int64).reshape(1, num_envs)
    base_indices = (step_offsets + env_offsets).reshape(-1)
    stage_ids = np.tile(np.tile(np.asarray([0, 1, 2], dtype=np.int64), num_envs), num_steps)
    env_indices_all = np.tile(np.repeat(np.arange(num_envs, dtype=np.int64), 3), num_steps)

    def _hist2_tensor(value: torch.Tensor | None, *, field_name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if not torch.is_tensor(value):
            raise RuntimeError(f"native rollout training ring is missing {field_name}.")
        per_step = int(value.shape[0]) // capacity
        prefix = value[: int(num_steps) * int(per_step)].to(device=device, dtype=dtype)
        return prefix.reshape(num_steps, per_step, *value.shape[1:])

    def _flat_tensor(value: torch.Tensor | None, *, field_name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if value is None:
            raise RuntimeError(f"native rollout training ring is missing {field_name}.")
        if not torch.is_tensor(value):
            raise RuntimeError(f"native rollout training ring field {field_name} must be a tensor.")
        per_step = int(value.shape[0]) // capacity
        prefix = value[: int(num_steps) * int(per_step)]
        prefix = prefix if prefix.device == device else prefix.to(device=device)
        if dtype is not None and prefix.dtype != dtype:
            prefix = prefix.to(dtype=dtype)
        return prefix

    def _optional_flat_tensor(value: torch.Tensor | None, *, dtype: torch.dtype = torch.float32) -> torch.Tensor | None:
        if value is None:
            return None
        if not torch.is_tensor(value):
            raise RuntimeError("native rollout training ring optional field must be a tensor.")
        per_step = int(value.shape[0]) // capacity
        prefix = value[: int(num_steps) * int(per_step)]
        prefix = prefix if prefix.device == device else prefix.to(device=device)
        if dtype is not None and prefix.dtype != dtype:
            prefix = prefix.to(dtype=dtype)
        return prefix

    accel_values_2 = _hist2_tensor(accel_stage.values, field_name="accel_stage.values")
    sat_values_2 = _hist2_tensor(sat_stage.values, field_name="sat_stage.values")
    bw_values_2 = _hist2_tensor(bw_stage.values, field_name="bw_stage.values")
    accel_logprob_2 = _hist2_tensor(accel_stage.old_logprobs, field_name="accel_stage.old_logprobs")
    sat_logprob_2 = _hist2_tensor(sat_stage.old_logprobs, field_name="sat_stage.old_logprobs")
    bw_logprob_2 = _hist2_tensor(bw_stage.old_logprobs, field_name="bw_stage.old_logprobs")
    rewards_2 = _hist2_tensor(bw_stage.rewards, field_name="bw_stage.rewards")
    terminated_2 = _hist2_tensor(history.terminated, field_name="terminated", dtype=torch.bool)
    truncated_2 = _hist2_tensor(history.truncated, field_name="truncated", dtype=torch.bool)
    zeros_2 = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    zeros_flat = zeros_2.reshape(-1)

    accel_values = accel_values_2.reshape(-1)
    sat_values = sat_values_2.reshape(-1)
    bw_values = bw_values_2.reshape(-1)
    rewards = rewards_2.reshape(-1)
    terminated = terminated_2.reshape(-1)
    truncated = truncated_2.reshape(-1)
    bw_access_rewards = _optional_flat_tensor(bw_stage.bw_access_rewards)
    if bw_access_rewards is None:
        bw_access_rewards = torch.zeros_like(rewards)
    bw_weighted_delta = _optional_flat_tensor(bw_stage.bw_weighted_workload_delta_rewards)
    if bw_weighted_delta is None:
        bw_weighted_delta = torch.zeros_like(rewards)
    bw_weighted_level = _optional_flat_tensor(bw_stage.bw_weighted_workload_level_rewards)
    if bw_weighted_level is None:
        bw_weighted_level = torch.zeros_like(rewards)
    bw_gu_queue = _optional_flat_tensor(bw_stage.bw_gu_queue_level_rewards)
    if bw_gu_queue is None:
        bw_gu_queue = torch.zeros_like(rewards)
    bw_system_queue = _optional_flat_tensor(bw_stage.bw_system_queue_level_rewards)
    if bw_system_queue is None:
        bw_system_queue = torch.zeros_like(rewards)
    bw_gu_service = _optional_flat_tensor(bw_stage.bw_gu_service_queue_rewards)
    if bw_gu_service is None:
        bw_gu_service = torch.zeros_like(rewards)

    accel_world = _world_state_from_flat_history_ring(
        accel_stage.world_batch,
        num_steps=num_steps,
        capacity=capacity + 1,
        stage_id=0,
        device=device,
    )
    sat_world = _world_state_from_flat_history_ring(
        sat_stage.world_batch,
        num_steps=num_steps,
        capacity=capacity,
        stage_id=1,
        device=device,
    )
    bw_world = _world_state_from_flat_history_ring(
        bw_stage.world_batch,
        num_steps=num_steps,
        capacity=capacity,
        stage_id=2,
        device=device,
    )
    next_actor_world = _world_state_from_flat_history_ring(
        accel_stage.world_batch,
        num_steps=num_steps,
        capacity=capacity + 1,
        stage_id=0,
        device=device,
        start_slot=1,
    )
    terminal_next_world = _world_state_from_flat_history_ring(
        history.terminal_next_world,
        num_steps=num_steps,
        capacity=capacity,
        stage_id=0,
        device=device,
    )
    terminal_next_world_mask = _hist2_tensor(
        history.terminal_next_world_mask,
        field_name="terminal_next_world_mask",
        dtype=torch.bool,
    ).reshape(-1)
    next_world = _where_world_state(terminal_next_world_mask, terminal_next_world, next_actor_world)
    accel_local = _accel_local_from_flat_history_ring(accel_stage, num_steps=num_steps, capacity=capacity, device=device)
    sat_local = _sat_local_from_flat_history_ring(sat_stage, num_steps=num_steps, capacity=capacity, device=device)
    bw_local = _bw_local_from_flat_history_ring(bw_stage, num_steps=num_steps, capacity=capacity, device=device)

    training_stage_batches = {
        0: StructuredStageTrainingBatch(
            stage_id=0,
            transition_indices=base_indices,
            env_indices=env_indices_stage,
            world_batch=accel_world,
            local_batch=accel_local,
            actions=_flat_tensor(accel_stage.actions, field_name="accel_stage.actions"),
            latent_actions=_optional_flat_tensor(getattr(accel_stage, "latent_actions", None)),
            old_logprobs=accel_logprob_2.reshape(-1),
            values=accel_values,
            rewards=zeros_flat,
            terminated=terminated,
            truncated=truncated,
            danger_imitation_targets=_optional_flat_tensor(accel_stage.danger_imitation_targets),
            danger_imitation_masks=_optional_flat_tensor(accel_stage.danger_imitation_masks),
        ),
        1: StructuredStageTrainingBatch(
            stage_id=1,
            transition_indices=base_indices + 1,
            env_indices=env_indices_stage,
            world_batch=sat_world,
            local_batch=sat_local,
            actions=_flat_tensor(sat_stage.actions, field_name="sat_stage.actions", dtype=torch.long),
            old_logprobs=sat_logprob_2.reshape(-1),
            values=sat_values,
            rewards=zeros_flat,
            terminated=terminated,
            truncated=truncated,
            old_logprobs_per_agent=_optional_flat_tensor(getattr(sat_stage, "old_logprobs_per_agent", None)),
            sat_action_indices=_optional_flat_tensor(getattr(sat_stage, "action_indices", None), dtype=torch.long),
        ),
        2: StructuredStageTrainingBatch(
            stage_id=2,
            transition_indices=base_indices + 2,
            env_indices=env_indices_stage,
            world_batch=bw_world,
            local_batch=bw_local,
            actions=_flat_tensor(bw_stage.actions, field_name="bw_stage.actions"),
            old_logprobs=bw_logprob_2.reshape(-1),
            values=bw_values,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            bw_access_rewards=bw_access_rewards,
            bw_weighted_workload_delta_rewards=bw_weighted_delta,
            bw_weighted_workload_level_rewards=bw_weighted_level,
            bw_gu_queue_level_rewards=bw_gu_queue,
            bw_system_queue_level_rewards=bw_system_queue,
            bw_gu_service_queue_rewards=bw_gu_service,
            bw_flow_proxy_scores=_optional_flat_tensor(bw_stage.bw_flow_proxy_scores),
            bw_flow_proxy_masks=_optional_flat_tensor(bw_stage.bw_flow_proxy_masks),
            bw_flow_proxy_deltas=_optional_flat_tensor(bw_stage.bw_flow_proxy_deltas),
            bw_ref_actions=_optional_flat_tensor(bw_stage.bw_ref_actions),
            bw_tau=_optional_flat_tensor(getattr(bw_stage, "bw_tau", None)),
            bw_kappa=_optional_flat_tensor(getattr(bw_stage, "bw_kappa", None)),
            bw_valid_count=_optional_flat_tensor(getattr(bw_stage, "bw_valid_count", None), dtype=torch.long),
            bw_latent_count=_optional_flat_tensor(getattr(bw_stage, "bw_latent_count", None), dtype=torch.long),
            bw_logprob_raw_per_agent=_optional_flat_tensor(getattr(bw_stage, "bw_logprob_raw_per_agent", None)),
            old_logprobs_per_agent=_optional_flat_tensor(bw_stage.bw_old_logprobs_per_agent),
        ),
    }

    accel_np = accel_values.detach().cpu().numpy().astype(np.float32, copy=False)
    sat_np = sat_values.detach().cpu().numpy().astype(np.float32, copy=False)
    bw_np = bw_values.detach().cpu().numpy().astype(np.float32, copy=False)
    zeros_np = np.zeros((num_steps * num_envs,), dtype=np.float32)
    rewards_np = rewards.detach().cpu().numpy().astype(np.float32, copy=False)
    terminated_np = terminated.detach().cpu().numpy().astype(bool, copy=False)
    truncated_np = truncated.detach().cpu().numpy().astype(bool, copy=False)
    bw_access_np = bw_access_rewards.detach().cpu().numpy().astype(np.float32, copy=False)
    bw_weighted_delta_np = bw_weighted_delta.detach().cpu().numpy().astype(np.float32, copy=False)
    bw_weighted_level_np = bw_weighted_level.detach().cpu().numpy().astype(np.float32, copy=False)
    bw_gu_queue_np = bw_gu_queue.detach().cpu().numpy().astype(np.float32, copy=False)
    bw_system_queue_np = bw_system_queue.detach().cpu().numpy().astype(np.float32, copy=False)
    bw_gu_service_np = bw_gu_service.detach().cpu().numpy().astype(np.float32, copy=False)
    bw_stage_reward_part_arrays: dict[str, np.ndarray] = {}
    return_reward_part_arrays: dict[str, np.ndarray] = {}
    reward_part_tensors = getattr(bw_stage, "reward_part_tensors", None)
    if isinstance(reward_part_tensors, dict):
        for key, tensor in reward_part_tensors.items():
            flat_tensor = _optional_flat_tensor(tensor)
            if flat_tensor is None:
                continue
            key_s = str(key)
            bw_array = flat_tensor.detach().cpu().numpy().astype(np.float32, copy=False).reshape(num_steps * num_envs)
            bw_stage_reward_part_arrays[key_s] = bw_array
            return_reward_part_arrays[key_s] = _interleave_stage_numpy(
                zeros_np.reshape(num_steps, num_envs),
                zeros_np.reshape(num_steps, num_envs),
                bw_array.reshape(num_steps, num_envs),
            ).astype(np.float32, copy=False)
    return_stage_batches = {
        0: StructuredStageReturnBatch(
            stage_id=0,
            transition_indices=base_indices,
            env_indices=env_indices_stage,
            values=accel_np,
            rewards=zeros_np,
            terminated=terminated_np,
            truncated=truncated_np,
            bw_access_rewards=zeros_np,
            bw_weighted_workload_delta_rewards=zeros_np,
            bw_weighted_workload_level_rewards=zeros_np,
            bw_gu_queue_level_rewards=zeros_np,
            bw_system_queue_level_rewards=zeros_np,
            bw_gu_service_queue_rewards=zeros_np,
            reward_part_arrays={},
            next_world_batch=None,
        ),
        1: StructuredStageReturnBatch(
            stage_id=1,
            transition_indices=base_indices + 1,
            env_indices=env_indices_stage,
            values=sat_np,
            rewards=zeros_np,
            terminated=terminated_np,
            truncated=truncated_np,
            bw_access_rewards=zeros_np,
            bw_weighted_workload_delta_rewards=zeros_np,
            bw_weighted_workload_level_rewards=zeros_np,
            bw_gu_queue_level_rewards=zeros_np,
            bw_system_queue_level_rewards=zeros_np,
            bw_gu_service_queue_rewards=zeros_np,
            reward_part_arrays={},
            next_world_batch=None,
        ),
        2: StructuredStageReturnBatch(
            stage_id=2,
            transition_indices=base_indices + 2,
            env_indices=env_indices_stage,
            values=bw_np,
            rewards=rewards_np,
            terminated=terminated_np,
            truncated=truncated_np,
            bw_access_rewards=bw_access_np,
            bw_weighted_workload_delta_rewards=bw_weighted_delta_np,
            bw_weighted_workload_level_rewards=bw_weighted_level_np,
            bw_gu_queue_level_rewards=bw_gu_queue_np,
            bw_system_queue_level_rewards=bw_system_queue_np,
            bw_gu_service_queue_rewards=bw_gu_service_np,
            reward_part_arrays=bw_stage_reward_part_arrays,
            next_world_batch=next_world,
        ),
    }
    sat_macro_interval = max(int(getattr(native_view, "sat_decision_interval", 1) or 1), 1)
    sat_macro_starts, sat_macro_ends, sat_macro_durations = _build_access_bw_macro_plan(
        num_steps=num_steps,
        num_envs=num_envs,
        interval=sat_macro_interval,
        terminated_2=terminated_np.reshape(num_steps, num_envs),
        truncated_2=truncated_np.reshape(num_steps, num_envs),
    )
    sat_start_t = torch.as_tensor(sat_macro_starts, dtype=torch.long, device=device)
    sat_duration_t = torch.as_tensor(sat_macro_durations, dtype=torch.long, device=device)
    num_sat_agents_i = int(_flat_tensor(sat_stage.actions, field_name="sat_stage.actions", dtype=torch.long).shape[1])
    sat_local_start_t = (
        sat_start_t.reshape(-1, 1) * max(num_sat_agents_i, 1)
        + torch.arange(max(num_sat_agents_i, 1), dtype=torch.long, device=device).reshape(1, -1)
    ).reshape(-1)
    sat_start_env_np = env_indices_stage[sat_macro_starts]
    sat_term_start = terminated.index_select(0, sat_start_t)
    sat_trunc_start = truncated.index_select(0, sat_start_t)
    sat_macro_batch = StructuredStageTrainingBatch(
        stage_id=1,
        transition_indices=base_indices[sat_macro_starts] + 1,
        env_indices=sat_start_env_np,
        world_batch=_index_dataclass_items(sat_world, sat_start_t),
        local_batch=_index_dataclass_items(sat_local, sat_local_start_t),
        actions=_flat_tensor(sat_stage.actions, field_name="sat_stage.actions", dtype=torch.long).index_select(0, sat_start_t),
        old_logprobs=sat_logprob_2.reshape(-1).index_select(0, sat_start_t),
        values=sat_values.index_select(0, sat_start_t),
        rewards=zeros_flat.index_select(0, sat_start_t),
        terminated=sat_term_start,
        truncated=sat_trunc_start,
        old_logprobs_per_agent=_index_optional_tensor(
            _optional_flat_tensor(getattr(sat_stage, "old_logprobs_per_agent", None)),
            sat_start_t,
        ),
        sat_action_indices=_index_optional_tensor(
            _optional_flat_tensor(getattr(sat_stage, "action_indices", None), dtype=torch.long),
            sat_start_t,
        ),
        duration=sat_duration_t,
    )
    training_stage_batches[1] = sat_macro_batch

    bw_macro_interval = max(int(getattr(native_view, "access_bw_decision_interval", 1) or 1), 1)
    bw_macro_starts, bw_macro_ends, bw_macro_durations = _build_access_bw_macro_plan(
        num_steps=num_steps,
        num_envs=num_envs,
        interval=bw_macro_interval,
        terminated_2=terminated_np.reshape(num_steps, num_envs),
        truncated_2=truncated_np.reshape(num_steps, num_envs),
    )
    start_t = torch.as_tensor(bw_macro_starts, dtype=torch.long, device=device)
    duration_t = torch.as_tensor(bw_macro_durations, dtype=torch.long, device=device)
    num_uav_i = int(_flat_tensor(bw_stage.actions, field_name="bw_stage.actions").shape[1])
    local_start_t = (
        start_t.reshape(-1, 1) * max(num_uav_i, 1)
        + torch.arange(max(num_uav_i, 1), dtype=torch.long, device=device).reshape(1, -1)
    ).reshape(-1)
    bw_start_env_np = env_indices_stage[bw_macro_starts]
    # Macro K changes which BW rows are actor decision points, not the
    # primitive-step reward/return chain.  Keep these reward fields as
    # macro-start diagnostics only; MC/GAE targets are gathered from the full
    # primitive return view by transition index.
    bw_term_start = terminated.index_select(0, start_t)
    bw_trunc_start = truncated.index_select(0, start_t)
    bw_macro_batch = StructuredStageTrainingBatch(
        stage_id=2,
        transition_indices=base_indices[bw_macro_starts] + 2,
        env_indices=bw_start_env_np,
        world_batch=_index_dataclass_items(bw_world, start_t),
        local_batch=_index_dataclass_items(bw_local, local_start_t),
        actions=_flat_tensor(bw_stage.actions, field_name="bw_stage.actions").index_select(0, start_t),
        old_logprobs=bw_logprob_2.reshape(-1).index_select(0, start_t),
        values=bw_values.index_select(0, start_t),
        rewards=rewards.index_select(0, start_t),
        terminated=bw_term_start,
        truncated=bw_trunc_start,
        bw_access_rewards=bw_access_rewards.index_select(0, start_t),
        bw_weighted_workload_delta_rewards=bw_weighted_delta.index_select(0, start_t),
        bw_weighted_workload_level_rewards=bw_weighted_level.index_select(0, start_t),
        bw_gu_queue_level_rewards=bw_gu_queue.index_select(0, start_t),
        bw_system_queue_level_rewards=bw_system_queue.index_select(0, start_t),
        bw_gu_service_queue_rewards=bw_gu_service.index_select(0, start_t),
        bw_flow_proxy_scores=_index_optional_tensor(_optional_flat_tensor(bw_stage.bw_flow_proxy_scores), start_t),
        bw_flow_proxy_masks=_index_optional_tensor(_optional_flat_tensor(bw_stage.bw_flow_proxy_masks), start_t),
        bw_flow_proxy_deltas=_index_optional_tensor(_optional_flat_tensor(bw_stage.bw_flow_proxy_deltas), start_t),
        bw_ref_actions=_index_optional_tensor(_optional_flat_tensor(bw_stage.bw_ref_actions), start_t),
        bw_tau=_index_optional_tensor(_optional_flat_tensor(getattr(bw_stage, "bw_tau", None)), start_t),
        bw_kappa=_index_optional_tensor(_optional_flat_tensor(getattr(bw_stage, "bw_kappa", None)), start_t),
        bw_valid_count=_index_optional_tensor(
            _optional_flat_tensor(getattr(bw_stage, "bw_valid_count", None), dtype=torch.long),
            start_t,
        ),
        bw_latent_count=_index_optional_tensor(
            _optional_flat_tensor(getattr(bw_stage, "bw_latent_count", None), dtype=torch.long),
            start_t,
        ),
        bw_logprob_raw_per_agent=_index_optional_tensor(
            _optional_flat_tensor(getattr(bw_stage, "bw_logprob_raw_per_agent", None)),
            start_t,
        ),
        old_logprobs_per_agent=_index_optional_tensor(_optional_flat_tensor(bw_stage.bw_old_logprobs_per_agent), start_t),
        duration=duration_t,
    )
    training_stage_batches[2] = bw_macro_batch
    return_view = StructuredReturnBatchView(
        transition_count=3 * num_steps * num_envs,
        stage_ids=stage_ids,
        env_indices=env_indices_all,
        values=_interleave_stage_numpy(
            accel_np.reshape(num_steps, num_envs),
            sat_np.reshape(num_steps, num_envs),
            bw_np.reshape(num_steps, num_envs),
        ).astype(np.float32, copy=False),
        rewards=_interleave_stage_numpy(
            zeros_np.reshape(num_steps, num_envs),
            zeros_np.reshape(num_steps, num_envs),
            rewards_np.reshape(num_steps, num_envs),
        ).astype(np.float32, copy=False),
        terminated=np.stack(
            (
                terminated_np.reshape(num_steps, num_envs),
                terminated_np.reshape(num_steps, num_envs),
                terminated_np.reshape(num_steps, num_envs),
            ),
            axis=2,
        ).reshape(-1).astype(bool, copy=False),
        truncated=np.stack(
            (
                truncated_np.reshape(num_steps, num_envs),
                truncated_np.reshape(num_steps, num_envs),
                truncated_np.reshape(num_steps, num_envs),
            ),
            axis=2,
        ).reshape(-1).astype(bool, copy=False),
        bw_access_rewards=_interleave_stage_numpy(
            zeros_np.reshape(num_steps, num_envs),
            zeros_np.reshape(num_steps, num_envs),
            bw_access_np.reshape(num_steps, num_envs),
        ).astype(np.float32, copy=False),
        bw_weighted_workload_delta_rewards=_interleave_stage_numpy(
            zeros_np.reshape(num_steps, num_envs),
            zeros_np.reshape(num_steps, num_envs),
            bw_weighted_delta_np.reshape(num_steps, num_envs),
        ).astype(np.float32, copy=False),
        bw_weighted_workload_level_rewards=_interleave_stage_numpy(
            zeros_np.reshape(num_steps, num_envs),
            zeros_np.reshape(num_steps, num_envs),
            bw_weighted_level_np.reshape(num_steps, num_envs),
        ).astype(np.float32, copy=False),
        bw_gu_queue_level_rewards=_interleave_stage_numpy(
            zeros_np.reshape(num_steps, num_envs),
            zeros_np.reshape(num_steps, num_envs),
            bw_gu_queue_np.reshape(num_steps, num_envs),
        ).astype(np.float32, copy=False),
        bw_system_queue_level_rewards=_interleave_stage_numpy(
            zeros_np.reshape(num_steps, num_envs),
            zeros_np.reshape(num_steps, num_envs),
            bw_system_queue_np.reshape(num_steps, num_envs),
        ).astype(np.float32, copy=False),
        bw_gu_service_queue_rewards=_interleave_stage_numpy(
            zeros_np.reshape(num_steps, num_envs),
            zeros_np.reshape(num_steps, num_envs),
            bw_gu_service_np.reshape(num_steps, num_envs),
        ).astype(np.float32, copy=False),
        reward_part_arrays=return_reward_part_arrays,
        stage_batches=return_stage_batches,
    )
    training_view = StructuredTrainingBatchView(
        transition_count=3 * num_steps * num_envs,
        stage_ids=stage_ids,
        old_logprobs=_interleave_stage_tensors(accel_logprob_2, sat_logprob_2, bw_logprob_2).reshape(-1),
        values=_interleave_stage_tensors(accel_values_2, sat_values_2, bw_values_2).reshape(-1),
        stage_batches=training_stage_batches,
    )
    return StructuredRolloutViews(
        training_view=training_view,
        return_view=return_view,
        bootstrap_view=_build_bootstrap_batch_view_from_return_view(return_view),
    )


def _build_rollout_views_from_records(
    records: Sequence[_EnvStepBatchRecord],
    *,
    device: torch.device,
) -> StructuredRolloutViews:
    if not records:
        return StructuredRolloutViews(
            training_view=_empty_training_batch_view(device),
            return_view=_empty_return_batch_view(),
            bootstrap_view=None,
        )
    stage_training_parts = {
        0: _new_stage_training_parts(),
        1: _new_stage_training_parts(),
        2: _new_stage_training_parts(),
    }
    stage_return_parts = {
        0: _new_stage_return_parts(),
        1: _new_stage_return_parts(),
        2: _new_stage_return_parts(),
    }
    stage_ids_parts: list[np.ndarray] = []
    training_old_logprob_parts: list[torch.Tensor] = []
    training_value_parts: list[torch.Tensor] = []
    return_env_indices_parts: list[np.ndarray] = []
    return_values_parts: list[np.ndarray] = []
    return_rewards_parts: list[np.ndarray] = []
    return_terminated_parts: list[np.ndarray] = []
    return_truncated_parts: list[np.ndarray] = []
    return_bw_access_parts: list[np.ndarray] = []
    return_bw_weighted_delta_parts: list[np.ndarray] = []
    return_bw_weighted_level_parts: list[np.ndarray] = []
    return_bw_gu_queue_parts: list[np.ndarray] = []
    return_bw_system_queue_parts: list[np.ndarray] = []
    return_bw_gu_service_parts: list[np.ndarray] = []
    transition_cursor = 0
    for record in records:
        num_envs = int(record.num_envs)
        env_indices = record.env_indices.astype(np.int64, copy=False)
        record_device = record.accel_values.device
        terminated = _record_bool_tensor(record.terminated, num_envs=num_envs, device=record_device)
        truncated = _record_bool_tensor(record.truncated, num_envs=num_envs, device=record_device)
        rewards = _record_float_tensor(record.rewards, num_envs=num_envs, device=record_device)
        terminated_np = _record_bool_numpy(record.terminated, num_envs=num_envs)
        truncated_np = _record_bool_numpy(record.truncated, num_envs=num_envs)
        rewards_np = _record_float_numpy(record.rewards, num_envs=num_envs)
        bw_access_np = _record_float_numpy(record.bw_access_rewards, num_envs=num_envs)
        bw_weighted_delta_np = _record_float_numpy(record.bw_weighted_workload_delta_rewards, num_envs=num_envs)
        bw_weighted_level_np = _record_float_numpy(record.bw_weighted_workload_level_rewards, num_envs=num_envs)
        bw_gu_queue_np = _record_float_numpy(record.bw_gu_queue_level_rewards, num_envs=num_envs)
        bw_system_queue_np = _record_float_numpy(record.bw_system_queue_level_rewards, num_envs=num_envs)
        bw_gu_service_np = _record_float_numpy(record.bw_gu_service_queue_rewards, num_envs=num_envs)
        zeros = torch.zeros((num_envs,), dtype=torch.float32, device=record_device)
        zeros_np = np.zeros((num_envs,), dtype=np.float32)
        stage_ids_parts.append(np.tile(np.asarray([0, 1, 2], dtype=np.int64), num_envs))
        training_old_logprob_parts.append(
            torch.stack(
                (
                    record.accel_old_logprobs.reshape(num_envs),
                    record.sat_old_logprobs.reshape(num_envs),
                    record.bw_old_logprobs.reshape(num_envs),
                ),
                dim=1,
            ).reshape(-1)
        )
        training_value_parts.append(
            torch.stack(
                (
                    record.accel_values.reshape(num_envs),
                    record.sat_values.reshape(num_envs),
                    record.bw_values.reshape(num_envs),
                ),
                dim=1,
            ).reshape(-1)
        )
        return_env_indices_parts.append(np.repeat(env_indices, 3))
        return_values_parts.append(
            torch.stack(
                (
                    record.accel_values.reshape(num_envs),
                    record.sat_values.reshape(num_envs),
                    record.bw_values.reshape(num_envs),
                ),
                dim=1,
            ).detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
        )
        return_rewards_parts.append(
            np.stack(
                (
                    zeros_np,
                    zeros_np,
                    rewards_np,
                ),
                axis=1,
            ).reshape(-1).astype(np.float32, copy=False)
        )
        return_terminated_parts.append(np.repeat(terminated_np, 3))
        return_truncated_parts.append(np.repeat(truncated_np, 3))
        return_bw_access_parts.append(
            np.stack((zeros_np, zeros_np, bw_access_np), axis=1)
            .reshape(-1)
            .astype(np.float32, copy=False)
        )
        return_bw_weighted_delta_parts.append(
            np.stack((zeros_np, zeros_np, bw_weighted_delta_np), axis=1)
            .reshape(-1)
            .astype(np.float32, copy=False)
        )
        return_bw_weighted_level_parts.append(
            np.stack((zeros_np, zeros_np, bw_weighted_level_np), axis=1)
            .reshape(-1)
            .astype(np.float32, copy=False)
        )
        return_bw_gu_queue_parts.append(
            np.stack((zeros_np, zeros_np, bw_gu_queue_np), axis=1)
            .reshape(-1)
            .astype(np.float32, copy=False)
        )
        return_bw_system_queue_parts.append(
            np.stack((zeros_np, zeros_np, bw_system_queue_np), axis=1)
            .reshape(-1)
            .astype(np.float32, copy=False)
        )
        return_bw_gu_service_parts.append(
            np.stack((zeros_np, zeros_np, bw_gu_service_np), axis=1)
            .reshape(-1)
            .astype(np.float32, copy=False)
        )
        base_indices = transition_cursor + 3 * np.arange(num_envs, dtype=np.int64)
        _append_stage_training_part(
            stage_training_parts[0],
            transition_indices=base_indices,
            env_indices=env_indices,
            world_batch=record.accel_world_batch,
            local_batch=record.accel_local_batch,
            actions=record.accel_actions,
            latent_actions=record.accel_latent_actions,
            old_logprobs=record.accel_old_logprobs.reshape(num_envs),
            values=record.accel_values.reshape(num_envs),
            rewards=zeros,
            terminated=terminated,
            truncated=truncated,
            danger_imitation_targets=record.accel_danger_targets,
            danger_imitation_masks=record.accel_danger_masks,
        )
        _append_stage_training_part(
            stage_training_parts[1],
            transition_indices=base_indices + 1,
            env_indices=env_indices,
            world_batch=record.sat_world_batch,
            local_batch=record.sat_local_batch,
            actions=record.sat_actions,
            old_logprobs=record.sat_old_logprobs.reshape(num_envs),
            values=record.sat_values.reshape(num_envs),
            rewards=zeros,
            terminated=terminated,
            truncated=truncated,
            old_logprobs_per_agent=record.sat_old_logprobs_per_agent,
            sat_stage_states=record.sat_stage_states,
        )
        _append_stage_training_part(
            stage_training_parts[2],
            transition_indices=base_indices + 2,
            env_indices=env_indices,
            world_batch=record.bw_world_batch,
            local_batch=record.bw_local_batch,
            actions=record.bw_actions,
            old_logprobs=record.bw_old_logprobs.reshape(num_envs),
            values=record.bw_values.reshape(num_envs),
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            bw_access_rewards=_record_float_tensor(record.bw_access_rewards, num_envs=num_envs, device=record_device),
            bw_weighted_workload_delta_rewards=_record_float_tensor(record.bw_weighted_workload_delta_rewards, num_envs=num_envs, device=record_device),
            bw_weighted_workload_level_rewards=_record_float_tensor(record.bw_weighted_workload_level_rewards, num_envs=num_envs, device=record_device),
            bw_gu_queue_level_rewards=_record_float_tensor(record.bw_gu_queue_level_rewards, num_envs=num_envs, device=record_device),
            bw_system_queue_level_rewards=_record_float_tensor(record.bw_system_queue_level_rewards, num_envs=num_envs, device=record_device),
            bw_gu_service_queue_rewards=_record_float_tensor(record.bw_gu_service_queue_rewards, num_envs=num_envs, device=record_device),
            bw_flow_proxy_scores=record.bw_flow_proxy_scores,
            bw_flow_proxy_masks=record.bw_flow_proxy_masks,
            bw_flow_proxy_deltas=record.bw_flow_proxy_deltas,
            bw_ref_actions=record.bw_ref_actions,
            old_logprobs_per_agent=record.bw_old_logprobs_per_agent,
            bw_stage_states=record.bw_stage_states,
        )
        _append_stage_return_part(
            stage_return_parts[0],
            transition_indices=base_indices,
            env_indices=env_indices,
            values=record.accel_values.detach().cpu().numpy().reshape(num_envs).astype(np.float32, copy=False),
            rewards=zeros_np,
            terminated=terminated_np,
            truncated=truncated_np,
            bw_access_rewards=zeros_np,
            bw_weighted_workload_delta_rewards=zeros_np,
            bw_weighted_workload_level_rewards=zeros_np,
            bw_gu_queue_level_rewards=zeros_np,
            bw_system_queue_level_rewards=zeros_np,
            bw_gu_service_queue_rewards=zeros_np,
            next_world_batch=record.sat_world_batch,
        )
        _append_stage_return_part(
            stage_return_parts[1],
            transition_indices=base_indices + 1,
            env_indices=env_indices,
            values=record.sat_values.detach().cpu().numpy().reshape(num_envs).astype(np.float32, copy=False),
            rewards=zeros_np,
            terminated=terminated_np,
            truncated=truncated_np,
            bw_access_rewards=zeros_np,
            bw_weighted_workload_delta_rewards=zeros_np,
            bw_weighted_workload_level_rewards=zeros_np,
            bw_gu_queue_level_rewards=zeros_np,
            bw_system_queue_level_rewards=zeros_np,
            bw_gu_service_queue_rewards=zeros_np,
            next_world_batch=record.bw_world_batch,
        )
        _append_stage_return_part(
            stage_return_parts[2],
            transition_indices=base_indices + 2,
            env_indices=env_indices,
            values=record.bw_values.detach().cpu().numpy().reshape(num_envs).astype(np.float32, copy=False),
            rewards=rewards_np,
            terminated=terminated_np,
            truncated=truncated_np,
            bw_access_rewards=bw_access_np,
            bw_weighted_workload_delta_rewards=bw_weighted_delta_np,
            bw_weighted_workload_level_rewards=bw_weighted_level_np,
            bw_gu_queue_level_rewards=bw_gu_queue_np,
            bw_system_queue_level_rewards=bw_system_queue_np,
            bw_gu_service_queue_rewards=bw_gu_service_np,
            next_world_batch=record.next_world_batch,
        )
        transition_cursor += int(record.transition_count)
    training_stage_batches = {
        stage_id: batch
        for stage_id in (0, 1, 2)
        if (batch := _finalize_stage_training_batch(stage_id, stage_training_parts[stage_id], device=device)) is not None
    }
    return_stage_batches = {
        stage_id: batch
        for stage_id in (0, 1, 2)
        if (batch := _finalize_stage_return_batch(stage_id, stage_return_parts[stage_id])) is not None
    }
    training_view = StructuredTrainingBatchView(
            transition_count=int(transition_cursor),
            stage_ids=np.concatenate(stage_ids_parts).astype(np.int64, copy=False),
            old_logprobs=_concat_tensor_batches(training_old_logprob_parts, device).reshape(-1),
            values=_concat_tensor_batches(training_value_parts, device).reshape(-1),
            stage_batches=training_stage_batches,
        )
    return_view = StructuredReturnBatchView(
            transition_count=int(transition_cursor),
            stage_ids=np.concatenate(stage_ids_parts).astype(np.int64, copy=False),
            env_indices=np.concatenate(return_env_indices_parts).astype(np.int64, copy=False),
            values=np.concatenate(return_values_parts).astype(np.float32, copy=False),
            rewards=np.concatenate(return_rewards_parts).astype(np.float32, copy=False),
            terminated=np.concatenate(return_terminated_parts).astype(bool, copy=False),
            truncated=np.concatenate(return_truncated_parts).astype(bool, copy=False),
            bw_access_rewards=np.concatenate(return_bw_access_parts).astype(np.float32, copy=False),
            bw_weighted_workload_delta_rewards=np.concatenate(return_bw_weighted_delta_parts).astype(np.float32, copy=False),
            bw_weighted_workload_level_rewards=np.concatenate(return_bw_weighted_level_parts).astype(np.float32, copy=False),
            bw_gu_queue_level_rewards=np.concatenate(return_bw_gu_queue_parts).astype(np.float32, copy=False),
        bw_system_queue_level_rewards=np.concatenate(return_bw_system_queue_parts).astype(np.float32, copy=False),
        bw_gu_service_queue_rewards=np.concatenate(return_bw_gu_service_parts).astype(np.float32, copy=False),
        reward_part_arrays={},
        stage_batches=return_stage_batches,
    )
    return StructuredRolloutViews(
        training_view=training_view,
        return_view=return_view,
        bootstrap_view=_build_bootstrap_batch_view_from_return_view(return_view),
    )


def _slice_dataclass_range(value: Any, start: int, end: int) -> Any:
    field_names = _tensor_field_names(value)
    if not field_names:
        raise TypeError(f"Expected fixed tensor-field value, got {type(value)!r}.")
    kwargs: dict[str, Any] = {}
    for field_name in field_names:
        kwargs[field_name] = getattr(value, field_name)[start:end]
    return type(value)(**kwargs)


def _index_optional_tensor(value: torch.Tensor | None, indices: torch.Tensor) -> torch.Tensor | None:
    if value is None:
        return None
    if not torch.is_tensor(value):
        raise RuntimeError("Expected optional rollout tensor to be a torch.Tensor.")
    return value.index_select(0, indices.to(device=value.device, dtype=torch.long))


def _build_access_bw_macro_plan(
    *,
    num_steps: int,
    num_envs: int,
    interval: int,
    terminated_2: np.ndarray,
    truncated_2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened [step, env] start/end rows and primitive durations.

    The function is intentionally used for interval=1 too.  That gives one
    macro row per primitive row with duration=1, so the new path is always
    exercised and can be parity-checked against the old behavior.
    """

    steps_i = max(int(num_steps), 0)
    envs_i = max(int(num_envs), 0)
    k = max(int(interval), 1)
    term = np.asarray(terminated_2, dtype=bool).reshape(steps_i, envs_i)
    trunc = np.asarray(truncated_2, dtype=bool).reshape(steps_i, envs_i)
    starts: list[int] = []
    ends: list[int] = []
    durations: list[int] = []
    for e in range(envs_i):
        t = 0
        while t < steps_i:
            end = t
            while (end + 1) < steps_i and (end - t + 1) < k and not bool(term[end, e] or trunc[end, e]):
                end += 1
            starts.append(t * envs_i + e)
            ends.append(end * envs_i + e)
            durations.append(end - t + 1)
            t = end + 1
    starts_arr = np.asarray(starts, dtype=np.int64)
    order = np.argsort(starts_arr, kind="stable")
    return (
        starts_arr[order],
        np.asarray(ends, dtype=np.int64)[order],
        np.asarray(durations, dtype=np.int64)[order],
    )


class StructuredRolloutBuffer:
    def __init__(self) -> None:
        self._records: List[_EnvStepBatchRecord] = []
        self._native_training_view: StructuredNativeRolloutTrainingBatchView | None = None
        self._transition_count = 0
        self._env_step_boundaries: List[int] = []
        self._stage_batches_cache: Dict[str, Any] | None = None

    def __len__(self) -> int:
        return int(self._transition_count)

    def _invalidate_stage_batch_cache(self) -> None:
        self._stage_batches_cache = None

    def _append_record(self, record: _EnvStepBatchRecord) -> None:
        if self._native_training_view is not None:
            raise RuntimeError("Cannot mix legacy step records with native rollout training ring.")
        self._records.append(_clone_rollout_value(record))
        self._transition_count += int(record.transition_count)
        self._invalidate_stage_batch_cache()

    def reset(self) -> None:
        self._records.clear()
        self._native_training_view = None
        self._transition_count = 0
        self._env_step_boundaries.clear()
        self._invalidate_stage_batch_cache()

    def add_native_rollout_training_view(
        self,
        *,
        history: Any,
        num_steps: int,
        num_envs: int,
        access_bw_decision_interval: int = 1,
        sat_decision_interval: int = 1,
        gamma_env: float = 1.0,
    ) -> None:
        if self._records:
            raise RuntimeError("Cannot mix native rollout training ring with legacy step records.")
        steps_i = max(int(num_steps), 0)
        envs_i = max(int(num_envs), 0)
        self._native_training_view = StructuredNativeRolloutTrainingBatchView(
            history=history,
            num_steps=steps_i,
            num_envs=envs_i,
            access_bw_decision_interval=max(int(access_bw_decision_interval), 1),
            sat_decision_interval=max(int(sat_decision_interval), 1),
            gamma_env=float(gamma_env),
        )
        self._transition_count = 3 * steps_i * envs_i
        self._env_step_boundaries = [3 * (idx + 1) for idx in range(steps_i * envs_i)]
        self._invalidate_stage_batch_cache()

    def add_env_step_batch(
        self,
        *,
        accel_world_batch: StructuredWorldState,
        sat_world_batch: StructuredWorldState,
        bw_world_batch: StructuredWorldState,
        next_world_batch: StructuredWorldState,
        accel_local_batch: Any,
        sat_local_batch: Any,
        bw_local_batch: Any,
        accel_actions: torch.Tensor,
        sat_actions: torch.Tensor,
        bw_actions: torch.Tensor,
        accel_old_logprobs: torch.Tensor,
        sat_old_logprobs: torch.Tensor,
        bw_old_logprobs: torch.Tensor,
        accel_values: torch.Tensor,
        sat_values: torch.Tensor,
        bw_values: torch.Tensor,
        rewards: Sequence[float] | np.ndarray | torch.Tensor,
        terminated: Sequence[bool] | np.ndarray | torch.Tensor,
        truncated: Sequence[bool] | np.ndarray | torch.Tensor,
        accel_latent_actions: torch.Tensor | None = None,
        accel_danger_imitation_targets: torch.Tensor | None = None,
        accel_danger_imitation_masks: torch.Tensor | None = None,
        sat_stage_states: Sequence[Dict[str, Any] | None] | None = None,
        bw_stage_states: Sequence[Dict[str, Any] | None] | None = None,
        bw_access_rewards: Sequence[float] | np.ndarray | torch.Tensor | None = None,
        bw_weighted_workload_delta_rewards: Sequence[float] | np.ndarray | torch.Tensor | None = None,
        bw_weighted_workload_level_rewards: Sequence[float] | np.ndarray | torch.Tensor | None = None,
        bw_gu_queue_level_rewards: Sequence[float] | np.ndarray | torch.Tensor | None = None,
        bw_system_queue_level_rewards: Sequence[float] | np.ndarray | torch.Tensor | None = None,
        bw_gu_service_queue_rewards: Sequence[float] | np.ndarray | torch.Tensor | None = None,
        bw_flow_proxy_scores: torch.Tensor | None = None,
        bw_flow_proxy_masks: torch.Tensor | None = None,
        bw_flow_proxy_deltas: torch.Tensor | None = None,
        bw_ref_actions: torch.Tensor | None = None,
        sat_old_logprobs_per_agent: torch.Tensor | None = None,
        bw_old_logprobs_per_agent: torch.Tensor | None = None,
        env_indices: Sequence[int] | np.ndarray | None = None,
        finalize_env_steps: bool = True,
        clone_tensors: bool = True,
    ) -> None:
        num_envs = int(accel_actions.shape[0])
        num_agents = int(accel_actions.shape[1]) if accel_actions.ndim >= 2 else 1
        env_index_arr = np.arange(num_envs, dtype=np.int64) if env_indices is None else np.asarray(env_indices, dtype=np.int64).reshape(num_envs)
        reward_device = accel_values.device if torch.is_tensor(accel_values) else torch.device("cpu")
        start_count = int(self._transition_count)
        record = _EnvStepBatchRecord(
            env_indices=env_index_arr.copy(),
            num_agents=num_agents,
            accel_world_batch=accel_world_batch,
            sat_world_batch=sat_world_batch,
            bw_world_batch=bw_world_batch,
            next_world_batch=next_world_batch,
            accel_local_batch=accel_local_batch,
            sat_local_batch=sat_local_batch,
            bw_local_batch=bw_local_batch,
            accel_actions=accel_actions.detach().clone() if clone_tensors else accel_actions.detach(),
            accel_latent_actions=(
                None
                if accel_latent_actions is None
                else (accel_latent_actions.detach().clone() if clone_tensors else accel_latent_actions.detach())
            ),
            sat_actions=sat_actions.detach().clone() if clone_tensors else sat_actions.detach(),
            bw_actions=bw_actions.detach().clone() if clone_tensors else bw_actions.detach(),
            accel_old_logprobs=(
                accel_old_logprobs.detach().reshape(num_envs).clone()
                if clone_tensors
                else accel_old_logprobs.detach().reshape(num_envs)
            ),
            sat_old_logprobs=(
                sat_old_logprobs.detach().reshape(num_envs).clone()
                if clone_tensors
                else sat_old_logprobs.detach().reshape(num_envs)
            ),
            sat_old_logprobs_per_agent=None if sat_old_logprobs_per_agent is None else (
                sat_old_logprobs_per_agent.detach().clone()
                if clone_tensors
                else sat_old_logprobs_per_agent.detach()
            ),
            bw_old_logprobs=(
                bw_old_logprobs.detach().reshape(num_envs).clone()
                if clone_tensors
                else bw_old_logprobs.detach().reshape(num_envs)
            ),
            accel_values=(
                accel_values.detach().reshape(num_envs).clone()
                if clone_tensors
                else accel_values.detach().reshape(num_envs)
            ),
            sat_values=(
                sat_values.detach().reshape(num_envs).clone()
                if clone_tensors
                else sat_values.detach().reshape(num_envs)
            ),
            bw_values=(
                bw_values.detach().reshape(num_envs).clone()
                if clone_tensors
                else bw_values.detach().reshape(num_envs)
            ),
            rewards=_as_record_float_tensor(rewards, num_envs=num_envs, device=reward_device, clone=clone_tensors),
            terminated=_as_record_bool_tensor(terminated, num_envs=num_envs, device=reward_device, clone=clone_tensors),
            truncated=_as_record_bool_tensor(truncated, num_envs=num_envs, device=reward_device, clone=clone_tensors),
            accel_danger_targets=(
                None
                if accel_danger_imitation_targets is None
                else (
                    accel_danger_imitation_targets.detach().clone()
                    if clone_tensors
                    else accel_danger_imitation_targets.detach()
                )
            ),
            accel_danger_masks=(
                None
                if accel_danger_imitation_masks is None
                else (
                    accel_danger_imitation_masks.detach().clone()
                    if clone_tensors
                    else accel_danger_imitation_masks.detach()
                )
            ),
            sat_stage_states=[None for _ in range(num_envs)] if sat_stage_states is None else [None if state is None else dict(state) for state in sat_stage_states],
            bw_stage_states=[None for _ in range(num_envs)] if bw_stage_states is None else [None if state is None else dict(state) for state in bw_stage_states],
            bw_access_rewards=_as_record_float_tensor(bw_access_rewards, num_envs=num_envs, device=reward_device, clone=clone_tensors),
            bw_weighted_workload_delta_rewards=_as_record_float_tensor(bw_weighted_workload_delta_rewards, num_envs=num_envs, device=reward_device, clone=clone_tensors),
            bw_weighted_workload_level_rewards=_as_record_float_tensor(bw_weighted_workload_level_rewards, num_envs=num_envs, device=reward_device, clone=clone_tensors),
            bw_gu_queue_level_rewards=_as_record_float_tensor(bw_gu_queue_level_rewards, num_envs=num_envs, device=reward_device, clone=clone_tensors),
            bw_system_queue_level_rewards=_as_record_float_tensor(bw_system_queue_level_rewards, num_envs=num_envs, device=reward_device, clone=clone_tensors),
            bw_gu_service_queue_rewards=_as_record_float_tensor(bw_gu_service_queue_rewards, num_envs=num_envs, device=reward_device, clone=clone_tensors),
            bw_flow_proxy_scores=None if bw_flow_proxy_scores is None else (bw_flow_proxy_scores.detach().clone() if clone_tensors else bw_flow_proxy_scores.detach()),
            bw_flow_proxy_masks=None if bw_flow_proxy_masks is None else (bw_flow_proxy_masks.detach().clone() if clone_tensors else bw_flow_proxy_masks.detach()),
            bw_flow_proxy_deltas=None if bw_flow_proxy_deltas is None else (bw_flow_proxy_deltas.detach().clone() if clone_tensors else bw_flow_proxy_deltas.detach()),
            bw_ref_actions=None if bw_ref_actions is None else (bw_ref_actions.detach().clone() if clone_tensors else bw_ref_actions.detach()),
            bw_old_logprobs_per_agent=None if bw_old_logprobs_per_agent is None else (bw_old_logprobs_per_agent.detach().clone() if clone_tensors else bw_old_logprobs_per_agent.detach()),
        )
        self._append_record(record)
        if finalize_env_steps:
            self._env_step_boundaries.extend(start_count + 3 * (env_offset + 1) for env_offset in range(num_envs))

    def build_rollout_views(self, device: torch.device) -> StructuredRolloutViews:
        if self._native_training_view is not None:
            return _build_rollout_views_from_native_training_ring(self._native_training_view, device=device)
        return _build_rollout_views_from_records(self._records, device=device)

    def build_bootstrap_view(self) -> StructuredBootstrapBatchView | None:
        if self._native_training_view is not None:
            return _build_rollout_views_from_native_training_ring(
                self._native_training_view,
                device=torch.device("cpu"),
            ).bootstrap_view
        return _build_bootstrap_batch_view_from_return_view(self.build_return_view())

    def build_bootstrap_world_state_dict(
        self,
        bootstrap_view: StructuredBootstrapBatchView | None = None,
    ) -> Dict[int, StructuredWorldState]:
        if bootstrap_view is None:
            bootstrap_view = self.build_bootstrap_view()
        if bootstrap_view is None or int(bootstrap_view.num_envs) <= 0:
            return {}
        latest: Dict[int, StructuredWorldState] = {}
        for env_offset, env_index in enumerate(np.asarray(bootstrap_view.env_indices, dtype=np.int64).tolist()):
            latest[int(env_index)] = _index_dataclass_items(bootstrap_view.next_world_batch, [env_offset])
        return latest

    def build_return_view(self) -> StructuredReturnBatchView:
        if self._native_training_view is not None:
            return _build_rollout_views_from_native_training_ring(
                self._native_training_view,
                device=torch.device("cpu"),
            ).return_view
        return _build_return_batch_view_from_records(self._records)

    def build_training_view(self, device: torch.device) -> StructuredTrainingBatchView:
        if self._native_training_view is not None:
            return _build_rollout_views_from_native_training_ring(self._native_training_view, device=device).training_view
        return _build_training_batch_view_from_records(self._records, device=device)

    def compute_gae(
        self,
        gamma_env: float,
        gae_lambda: float,
        bootstrap_value: float = 0.0,
        bootstrap_values: Dict[int, float] | None = None,
        truncated_bootstrap_values: Dict[int, float] | None = None,
        bootstrap_truncated: bool = False,
        mode: str = "stage_chained",
        step_target_mode: str = "env_reward",
        bw_target_mode: str = "env_reward",
        bw_return_mode: str = "gae",
        bw_reward_w_access: float = 1.0,
        bw_nstep_horizon: int = 3,
        return_view: StructuredReturnBatchView | None = None,
        value_override: np.ndarray | torch.Tensor | None = None,
    ) -> Dict[str, np.ndarray]:
        view = self.build_return_view() if return_view is None else return_view
        mode_l = str(mode).strip().lower()
        if mode_l not in {"stage_chained", "step_level"}:
            raise ValueError(f"Unsupported structured GAE mode: {mode}")
        step_target_mode_l = str(step_target_mode).strip().lower()
        if step_target_mode_l not in {"env_reward", "access_term", "access_raw"}:
            raise ValueError(f"Unsupported step target mode: {step_target_mode}")
        bw_target_mode_l = str(bw_target_mode).strip().lower()
        if bw_target_mode_l not in {
            "env_reward",
            "access_term",
            "access_raw",
            "weighted_workload_delta",
            "weighted_workload_level",
            "gu_queue_level",
            "system_queue_level",
            "gu_service_queue",
        }:
            raise ValueError(f"Unsupported bw target mode: {bw_target_mode}")
        bw_return_mode_l = str(bw_return_mode).strip().lower()
        if bw_return_mode_l not in {
            "gae",
            "bw_gae",
            "step_lambda_return",
            "monte_carlo",
            "mc",
            "bw_episode_mc",
            "bw_nstep",
        }:
            raise ValueError(f"Unsupported bw return mode: {bw_return_mode}")
        if bw_return_mode_l in {"mc", "monte_carlo"}:
            bw_return_mode_l = "step_lambda_return"
        bw_nstep_horizon_eff = max(int(bw_nstep_horizon), 1)
        num_steps = int(view.transition_count)
        advantages = np.zeros((num_steps,), dtype=np.float32)
        returns = np.zeros((num_steps,), dtype=np.float32)
        if bootstrap_values is None:
            bootstrap_values = {0: float(bootstrap_value)}
        if truncated_bootstrap_values is None:
            truncated_bootstrap_values = {}
        if value_override is None:
            values = np.asarray(view.values, dtype=np.float32).reshape(num_steps)
        else:
            values = np.asarray(
                value_override.detach().cpu().numpy() if torch.is_tensor(value_override) else value_override,
                dtype=np.float32,
            ).reshape(num_steps)
            if int(values.shape[0]) != num_steps:
                raise ValueError(
                    f"value_override length mismatch: got {int(values.shape[0])}, expected {num_steps}."
                )
        env_to_indices: Dict[int, List[int]] = {}
        for idx, env_index in enumerate(np.asarray(view.env_indices, dtype=np.int64)):
            env_to_indices.setdefault(int(env_index), []).append(idx)
        if mode_l == "stage_chained":
            for env_index, indices in env_to_indices.items():
                next_adv = 0.0
                next_value = float(bootstrap_values.get(int(env_index), 0.0))
                for idx in reversed(indices):
                    stage_id = int(view.stage_ids[idx])
                    gamma = float(gamma_env) if stage_id == 2 else 1.0
                    value = float(values[idx])
                    env_step_boundary = stage_id == 2
                    if env_step_boundary and bool(view.terminated[idx]):
                        next_value_eff = 0.0
                        next_adv = 0.0
                    elif env_step_boundary and bool(view.truncated[idx]):
                        next_value_eff = (
                            float(truncated_bootstrap_values.get(int(idx), 0.0))
                            if bool(bootstrap_truncated)
                            else 0.0
                        )
                        next_adv = 0.0
                    else:
                        next_value_eff = next_value
                    delta = float(view.rewards[idx]) + gamma * next_value_eff - value
                    advantages[idx] = delta + gamma * float(gae_lambda) * next_adv
                    returns[idx] = advantages[idx] + value
                    next_adv = float(advantages[idx])
                    next_value = value
            return {"advantages": advantages, "returns": returns}

        for env_index, indices in env_to_indices.items():
            if len(indices) % 3 != 0:
                raise ValueError(
                    f"step_level target expects groups of 3 transitions per env-step; "
                    f"env {env_index} has {len(indices)} transitions"
                )
            step_groups = [indices[offset : offset + 3] for offset in range(0, len(indices), 3)]
            next_step_return = 0.0
            next_step_accel_value = float(bootstrap_values.get(int(env_index), 0.0))
            next_bw_episode_return = 0.0
            next_bw_adv = 0.0
            next_bw_value = float(bootstrap_values.get(int(env_index), 0.0))
            if bw_return_mode_l == "bw_nstep":
                bw_rewards_by_step: list[float] = []
                bw_values_by_step: list[float] = []
                bw_terminated_by_step: list[bool] = []
                bw_truncated_by_step: list[bool] = []
                bw_timeout_bootstrap_by_step: list[float] = []
                for group in step_groups:
                    _, _, bw_idx_local = group
                    step_reward_local = float(
                        view.rewards[group[0]]
                        + view.rewards[group[1]]
                        + view.rewards[group[2]]
                    )
                    if step_target_mode_l != "env_reward":
                        bw_step_reward_local = (
                            float(bw_reward_w_access) * float(view.bw_access_rewards[bw_idx_local])
                            if step_target_mode_l == "access_term"
                            else float(view.bw_access_rewards[bw_idx_local])
                        )
                    elif bw_target_mode_l == "env_reward":
                        bw_step_reward_local = step_reward_local
                    elif bw_target_mode_l == "access_term":
                        bw_step_reward_local = float(bw_reward_w_access) * float(view.bw_access_rewards[bw_idx_local])
                    elif bw_target_mode_l == "weighted_workload_delta":
                        bw_step_reward_local = float(view.bw_weighted_workload_delta_rewards[bw_idx_local])
                    elif bw_target_mode_l == "weighted_workload_level":
                        bw_step_reward_local = float(view.bw_weighted_workload_level_rewards[bw_idx_local])
                    elif bw_target_mode_l == "gu_queue_level":
                        bw_step_reward_local = float(view.bw_gu_queue_level_rewards[bw_idx_local])
                    elif bw_target_mode_l == "system_queue_level":
                        bw_step_reward_local = float(view.bw_system_queue_level_rewards[bw_idx_local])
                    elif bw_target_mode_l == "gu_service_queue":
                        bw_step_reward_local = float(view.bw_gu_service_queue_rewards[bw_idx_local])
                    else:
                        bw_step_reward_local = float(view.bw_access_rewards[bw_idx_local])
                    bw_rewards_by_step.append(float(bw_step_reward_local))
                    bw_values_by_step.append(float(values[bw_idx_local]))
                    bw_terminated_by_step.append(bool(view.terminated[bw_idx_local]))
                    bw_truncated_by_step.append(bool(view.truncated[bw_idx_local]))
                    bw_timeout_bootstrap_by_step.append(float(truncated_bootstrap_values.get(int(bw_idx_local), 0.0)))
            for group_offset in range(len(step_groups) - 1, -1, -1):
                group = step_groups[group_offset]
                stage_ids = [int(view.stage_ids[idx]) for idx in group]
                if stage_ids != [0, 1, 2]:
                    raise ValueError(
                        f"step_level target expects [0, 1, 2] stage ordering per env-step; "
                        f"env {env_index} group {group_offset} saw {stage_ids}"
                    )
                accel_idx, sat_idx, bw_idx = group
                step_reward = float(
                    view.rewards[accel_idx]
                    + view.rewards[sat_idx]
                    + view.rewards[bw_idx]
                )
                terminated = bool(view.terminated[bw_idx])
                truncated = bool(view.truncated[bw_idx])
                if step_target_mode_l == "env_reward":
                    step_train_reward = step_reward
                elif step_target_mode_l == "access_term":
                    step_train_reward = float(bw_reward_w_access) * float(view.bw_access_rewards[bw_idx])
                else:
                    step_train_reward = float(view.bw_access_rewards[bw_idx])
                if step_target_mode_l != "env_reward":
                    bw_step_reward = step_train_reward
                elif bw_target_mode_l == "env_reward":
                    bw_step_reward = step_reward
                elif bw_target_mode_l == "access_term":
                    bw_step_reward = float(bw_reward_w_access) * float(view.bw_access_rewards[bw_idx])
                elif bw_target_mode_l == "weighted_workload_delta":
                    bw_step_reward = float(view.bw_weighted_workload_delta_rewards[bw_idx])
                elif bw_target_mode_l == "weighted_workload_level":
                    bw_step_reward = float(view.bw_weighted_workload_level_rewards[bw_idx])
                elif bw_target_mode_l == "gu_queue_level":
                    bw_step_reward = float(view.bw_gu_queue_level_rewards[bw_idx])
                elif bw_target_mode_l == "system_queue_level":
                    bw_step_reward = float(view.bw_system_queue_level_rewards[bw_idx])
                elif bw_target_mode_l == "gu_service_queue":
                    bw_step_reward = float(view.bw_gu_service_queue_rewards[bw_idx])
                else:
                    bw_step_reward = float(view.bw_access_rewards[bw_idx])
                value_bw = float(values[bw_idx])
                if terminated:
                    step_return = step_train_reward
                    bw_step_return = bw_step_reward
                    bw_advantage = bw_step_return - value_bw
                    next_step_return = 0.0
                    next_step_accel_value = 0.0
                    next_bw_episode_return = 0.0
                    next_bw_adv = 0.0
                    next_bw_value = 0.0
                elif truncated:
                    timeout_bootstrap = (
                        float(truncated_bootstrap_values.get(int(bw_idx), 0.0))
                        if bool(bootstrap_truncated)
                        else 0.0
                    )
                    step_return = step_train_reward + float(gamma_env) * timeout_bootstrap
                    bw_step_return = bw_step_reward + float(gamma_env) * timeout_bootstrap
                    bw_advantage = bw_step_return - value_bw
                    next_step_return = 0.0
                    next_step_accel_value = 0.0
                    next_bw_episode_return = 0.0
                    next_bw_adv = 0.0
                    next_bw_value = 0.0
                else:
                    step_return = step_train_reward + float(gamma_env) * (
                        (1.0 - float(gae_lambda)) * next_step_accel_value
                        + float(gae_lambda) * next_step_return
                    )
                    if bw_return_mode_l == "gae":
                        bw_step_return = bw_step_reward + float(gamma_env) * (
                            (1.0 - float(gae_lambda)) * next_step_accel_value
                            + float(gae_lambda) * next_step_return
                        )
                        bw_advantage = bw_step_return - value_bw
                    elif bw_return_mode_l == "bw_episode_mc":
                        bw_step_return = bw_step_reward + float(gamma_env) * next_bw_episode_return
                        bw_advantage = bw_step_return - value_bw
                    elif bw_return_mode_l == "bw_gae":
                        delta_bw = bw_step_reward + float(gamma_env) * next_bw_value - value_bw
                        bw_advantage = delta_bw + float(gamma_env) * float(gae_lambda) * next_bw_adv
                        bw_step_return = bw_advantage + value_bw
                    elif bw_return_mode_l == "bw_nstep":
                        bw_step_return = 0.0
                        discount = 1.0
                        bootstrap_after_h = float(bootstrap_values.get(int(env_index), 0.0))
                        terminated_early = False
                        truncated_early = False
                        for future_offset in range(group_offset, group_offset + bw_nstep_horizon_eff):
                            if future_offset >= len(step_groups):
                                break
                            bw_step_return += discount * float(bw_rewards_by_step[future_offset])
                            if bw_terminated_by_step[future_offset]:
                                bootstrap_after_h = 0.0
                                terminated_early = True
                                break
                            if bw_truncated_by_step[future_offset]:
                                if bool(bootstrap_truncated):
                                    bw_step_return += discount * float(gamma_env) * float(
                                        bw_timeout_bootstrap_by_step[future_offset]
                                    )
                                bootstrap_after_h = 0.0
                                truncated_early = True
                                break
                            discount *= float(gamma_env)
                        if not terminated_early and not truncated_early:
                            bootstrap_step = group_offset + bw_nstep_horizon_eff
                            if bootstrap_step < len(step_groups):
                                bootstrap_after_h = float(bw_values_by_step[bootstrap_step])
                            bw_step_return += discount * float(bootstrap_after_h)
                        bw_advantage = bw_step_return - value_bw
                    else:
                        bw_step_return = bw_step_reward + float(gamma_env) * next_step_return
                        bw_advantage = bw_step_return - value_bw
                for idx in (accel_idx, sat_idx):
                    value = float(values[idx])
                    returns[idx] = float(step_return)
                    advantages[idx] = float(step_return - value)
                returns[bw_idx] = float(bw_step_return)
                advantages[bw_idx] = float(bw_advantage)
                next_step_return = float(step_return)
                next_step_accel_value = float(values[accel_idx])
                next_bw_episode_return = float(bw_step_return)
                next_bw_adv = float(bw_advantage)
                next_bw_value = float(value_bw)
        return {"advantages": advantages, "returns": returns}
