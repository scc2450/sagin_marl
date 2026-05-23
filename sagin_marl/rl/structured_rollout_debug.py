from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from .structured_buffer import StructuredRolloutBuffer
from .structured_types import StructuredWorldState


def _clone_tensor(value: torch.Tensor | np.ndarray | float | int | bool, *, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().clone().to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def _batch_tensor(
    value: torch.Tensor | np.ndarray | float | int | bool,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = _clone_tensor(value, dtype=dtype)
    if tensor.ndim == 0:
        return tensor.reshape(1)
    return tensor.unsqueeze(0)


def _batch_optional_tensor(
    value: torch.Tensor | np.ndarray | None,
    *,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if value is None:
        return None
    return _batch_tensor(value, dtype=dtype)


def append_single_env_step(
    buffer: StructuredRolloutBuffer,
    *,
    env_index: int,
    accel_world_state: StructuredWorldState,
    sat_world_state: StructuredWorldState,
    bw_world_state: StructuredWorldState,
    next_world_state: StructuredWorldState,
    accel_local_actor_state: Any,
    sat_local_actor_state: Any,
    bw_local_actor_state: Any,
    accel_action: torch.Tensor | np.ndarray,
    sat_action: torch.Tensor | np.ndarray,
    bw_action: torch.Tensor | np.ndarray,
    accel_old_logprob: torch.Tensor | np.ndarray | float,
    sat_old_logprob: torch.Tensor | np.ndarray | float,
    bw_old_logprob: torch.Tensor | np.ndarray | float,
    accel_value: torch.Tensor | np.ndarray | float,
    sat_value: torch.Tensor | np.ndarray | float,
    bw_value: torch.Tensor | np.ndarray | float,
    reward: float,
    terminated: bool,
    truncated: bool,
    accel_danger_imitation_target: torch.Tensor | np.ndarray | None = None,
    accel_danger_imitation_mask: torch.Tensor | np.ndarray | None = None,
    sat_stage_state: dict[str, Any] | None = None,
    bw_stage_state: dict[str, Any] | None = None,
    bw_access_reward: float = 0.0,
    bw_weighted_workload_delta_reward: float = 0.0,
    bw_weighted_workload_level_reward: float = 0.0,
    bw_gu_queue_level_reward: float = 0.0,
    bw_system_queue_level_reward: float = 0.0,
    bw_gu_service_queue_reward: float = 0.0,
    bw_flow_proxy_scores: torch.Tensor | np.ndarray | None = None,
    bw_flow_proxy_mask: torch.Tensor | np.ndarray | None = None,
    bw_flow_proxy_deltas: torch.Tensor | np.ndarray | None = None,
    bw_ref_action: torch.Tensor | np.ndarray | None = None,
    bw_old_logprob_per_agent: torch.Tensor | np.ndarray | None = None,
) -> None:
    buffer.add_env_step_batch(
        accel_world_batch=accel_world_state,
        sat_world_batch=sat_world_state,
        bw_world_batch=bw_world_state,
        next_world_batch=next_world_state,
        accel_local_batch=accel_local_actor_state,
        sat_local_batch=sat_local_actor_state,
        bw_local_batch=bw_local_actor_state,
        accel_actions=_batch_tensor(accel_action, dtype=torch.float32),
        sat_actions=_batch_tensor(sat_action, dtype=torch.long),
        bw_actions=_batch_tensor(bw_action, dtype=torch.float32),
        accel_old_logprobs=_batch_tensor(accel_old_logprob, dtype=torch.float32),
        sat_old_logprobs=_batch_tensor(sat_old_logprob, dtype=torch.float32),
        bw_old_logprobs=_batch_tensor(bw_old_logprob, dtype=torch.float32),
        accel_values=_batch_tensor(accel_value, dtype=torch.float32),
        sat_values=_batch_tensor(sat_value, dtype=torch.float32),
        bw_values=_batch_tensor(bw_value, dtype=torch.float32),
        rewards=np.asarray([float(reward)], dtype=np.float32),
        terminated=np.asarray([bool(terminated)], dtype=bool),
        truncated=np.asarray([bool(truncated)], dtype=bool),
        accel_danger_imitation_targets=_batch_optional_tensor(
            accel_danger_imitation_target,
            dtype=torch.float32,
        ),
        accel_danger_imitation_masks=_batch_optional_tensor(
            accel_danger_imitation_mask,
            dtype=torch.float32,
        ),
        sat_stage_states=[None if sat_stage_state is None else dict(sat_stage_state)],
        bw_stage_states=[None if bw_stage_state is None else dict(bw_stage_state)],
        bw_access_rewards=np.asarray([float(bw_access_reward)], dtype=np.float32),
        bw_weighted_workload_delta_rewards=np.asarray(
            [float(bw_weighted_workload_delta_reward)],
            dtype=np.float32,
        ),
        bw_weighted_workload_level_rewards=np.asarray(
            [float(bw_weighted_workload_level_reward)],
            dtype=np.float32,
        ),
        bw_gu_queue_level_rewards=np.asarray([float(bw_gu_queue_level_reward)], dtype=np.float32),
        bw_system_queue_level_rewards=np.asarray([float(bw_system_queue_level_reward)], dtype=np.float32),
        bw_gu_service_queue_rewards=np.asarray([float(bw_gu_service_queue_reward)], dtype=np.float32),
        bw_flow_proxy_scores=_batch_optional_tensor(bw_flow_proxy_scores, dtype=torch.float32),
        bw_flow_proxy_masks=_batch_optional_tensor(bw_flow_proxy_mask, dtype=torch.float32),
        bw_flow_proxy_deltas=_batch_optional_tensor(bw_flow_proxy_deltas, dtype=torch.float32),
        bw_ref_actions=_batch_optional_tensor(bw_ref_action, dtype=torch.float32),
        bw_old_logprobs_per_agent=_batch_optional_tensor(
            bw_old_logprob_per_agent,
            dtype=torch.float32,
        ),
        env_indices=np.asarray([int(env_index)], dtype=np.int64),
        finalize_env_steps=True,
    )


def copy_prefix_rollout_buffer(
    full_buffer: StructuredRolloutBuffer,
    rollout_env_steps: int,
) -> StructuredRolloutBuffer:
    prefix = StructuredRolloutBuffer()
    max_env_steps = max(int(rollout_env_steps), 0)
    if max_env_steps <= 0:
        return prefix
    for env_step_index, record in enumerate(full_buffer._records):
        if env_step_index >= max_env_steps:
            break
        prefix._records.append(copy.deepcopy(record))
        prefix._transition_count += int(record.transition_count)
        prefix._env_step_boundaries.append(prefix._transition_count)
    return prefix
