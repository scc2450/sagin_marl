from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from sagin_marl.env.structured_batch_env_core import StructuredBatchEnvCore


class StructuredTorchEnvBatch:
    def __init__(self, core: StructuredBatchEnvCore) -> None:
        self._core = core

    @property
    def tensor_device(self) -> torch.device | None:
        return self._core.tensor_device

    @property
    def native_rollout_runtime(self):
        return self._core.native_rollout_runtime

    def set_tensor_device(self, device: torch.device | str | None) -> None:
        self._core.set_tensor_device(device)

    def set_native_rollout_fast_random(self, enabled: bool) -> None:
        self._core.set_native_rollout_fast_random(enabled)

    def clear_native_main_kernel(self) -> None:
        self._core.clear_native_main_kernel()

    def begin_native_main_kernel_rollout(self, *, capacity: int, num_envs: int) -> None:
        self._core.begin_native_main_kernel_rollout(capacity=capacity, num_envs=num_envs)

    def ensure_native_rollout_reset_tape_capacity(self, *, chunk_rows: int | None = None) -> bool:
        return self._core.ensure_native_rollout_reset_tape_capacity(chunk_rows=chunk_rows)

    @property
    def drivers(self):
        return self._core.drivers

    @property
    def envs(self):
        return self._core.envs

    def reset(self, indices: Sequence[int] | None = None, seeds: Sequence[int | None] | None = None) -> None:
        self._core.reset_many(seeds=seeds, indices=indices)

    def reset_at(self, index: int, seed: int | None = None) -> None:
        self._core.reset_at(index=index, seed=seed)

    def native_rollout_program(self):
        return self._core.native_rollout_program()

    def debug_native_hot_replay_program(
        self,
        *,
        capacity: int = 3,
        num_envs: int | None = None,
        selected_indices: Sequence[int] | None = None,
    ):
        return self._core.debug_native_hot_replay_program(
            capacity=capacity,
            num_envs=num_envs,
            selected_indices=selected_indices,
        )

    def native_sub_batch_rollout_program(
        self,
        *,
        capacity: int = 3,
        selected_indices: Sequence[int],
        allow_duplicate_indices: bool = False,
    ):
        return self._core.native_sub_batch_rollout_program(
            capacity=capacity,
            selected_indices=selected_indices,
            allow_duplicate_indices=allow_duplicate_indices,
        )

    def prepare_native_branch_replay_from_history(
        self,
        *,
        history_rows: Sequence[int],
        horizon: int,
        stage_id: int = 2,
        future_random_mode: str = "copy",
        future_random_seed: int | None = None,
    ) -> None:
        self._core.prepare_native_branch_replay_from_history(
            history_rows=history_rows,
            horizon=horizon,
            stage_id=stage_id,
            future_random_mode=future_random_mode,
            future_random_seed=future_random_seed,
        )

    def get_global_state_batch(
        self,
        *,
        indices: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ):
        return self._core.get_global_state_batch(indices=indices, device=device)

    def export_runtime_state_batch(self, *, indices: Sequence[int] | None = None) -> list[dict[str, Any]]:
        return self._core.export_runtime_state_batch(indices=indices)

    def load_runtime_state_batch(
        self,
        states: Sequence[dict[str, Any]],
        *,
        indices: Sequence[int] | None = None,
        refresh_observation_cache: bool = True,
        refresh_global_state_cache: bool = True,
    ) -> None:
        self._core.load_runtime_state_batch(
            states,
            indices=indices,
            refresh_observation_cache=refresh_observation_cache,
            refresh_global_state_cache=refresh_global_state_cache,
        )

    def refresh_stage_obs_cache_many(self, indices: Sequence[int] | None = None) -> None:
        self._core.refresh_stage_obs_cache_many(indices=indices)

    def current_obs_batch(
        self,
        *,
        indices: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ):
        return self._core.current_obs_batch(indices=indices, device=device)

    def current_obs_many(self, indices: Sequence[int] | None = None):
        return self._core.current_obs_many(indices=indices)

    def cluster_meta_batch(
        self,
        *,
        indices: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ):
        return self._core.cluster_meta_batch(indices=indices, device=device)

    def cluster_meta_many(self, indices: Sequence[int] | None = None):
        return self._core.cluster_meta_many(indices=indices)

    def sat_mask_to_ids_many(
        self,
        sat_masks: Sequence[np.ndarray],
        indices: Sequence[int] | None = None,
    ):
        return self._core.sat_mask_to_ids_many(sat_masks, indices=indices)

    def close(self) -> None:
        self._core.close()
