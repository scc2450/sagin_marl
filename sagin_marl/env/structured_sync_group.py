from __future__ import annotations

import copy
from typing import Any, Sequence

import numpy as np
import torch

from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_batch_env_core import StructuredBatchEnvCore
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.env.structured_torch_env_batch import StructuredTorchEnvBatch


class _StructuredNativeBatchDriverGroup:
    """Primary structured batch group used by training/eval hot paths."""

    def __init__(
        self,
        cfg,
        num_envs: int | None = None,
        *,
        tensor_device: torch.device | str | None = None,
    ) -> None:
        if num_envs is None:
            raise ValueError("num_envs is required for native structured batch groups")
        self._batch_core = StructuredBatchEnvCore(
            cfg,
            num_envs=int(num_envs),
            tensor_device=tensor_device,
        )
        self._drivers = self._batch_core.drivers
        self._envs = self._batch_core.envs
        self._batch_env = StructuredTorchEnvBatch(self._batch_core)

    def __len__(self) -> int:
        return len(self._drivers)

    def __getitem__(self, index: int) -> StructuredControlDriver:
        return self._drivers[int(index)]

    @property
    def drivers(self) -> list[StructuredControlDriver]:
        return self._drivers

    @property
    def envs(self) -> list[Any]:
        return self._envs

    @property
    def batch_core(self) -> StructuredBatchEnvCore:
        return self._batch_core

    @property
    def batch_env(self) -> StructuredTorchEnvBatch:
        return self._batch_env

    @property
    def cfg(self):
        return self._batch_core.cfg

    @property
    def native_rollout_runtime(self):
        return self._batch_core.native_rollout_runtime

    @property
    def tensor_device(self) -> torch.device | None:
        return self._batch_env.tensor_device

    def set_tensor_device(self, device: torch.device | str | None) -> None:
        self._batch_env.set_tensor_device(device)

    def set_native_rollout_fast_random(self, enabled: bool) -> None:
        self._batch_core.set_native_rollout_fast_random(enabled)

    def clear_native_main_kernel(self) -> None:
        self._batch_env.clear_native_main_kernel()

    def begin_native_main_kernel_rollout(self, *, capacity: int, num_envs: int) -> None:
        self._batch_env.begin_native_main_kernel_rollout(capacity=capacity, num_envs=num_envs)

    def ensure_native_rollout_reset_tape_capacity(self, *, chunk_rows: int | None = None) -> bool:
        return self._batch_env.ensure_native_rollout_reset_tape_capacity(chunk_rows=chunk_rows)

    def _resolve_indices(self, indices: Sequence[int] | None = None) -> list[int]:
        return self._batch_core.resolve_indices(indices)

    def _select_drivers(self, indices: Sequence[int] | None = None) -> tuple[list[int], list[StructuredControlDriver]]:
        return self._batch_core.select_drivers(indices)

    def _output_device(self, device: torch.device | str | None = None) -> torch.device | None:
        return self._batch_core.output_device(device)

    def native_rollout_program(self):
        return self._batch_env.native_rollout_program()

    def debug_native_hot_replay_program(
        self,
        *,
        capacity: int = 3,
        num_envs: int | None = None,
        selected_indices: Sequence[int] | None = None,
    ):
        return self._batch_env.debug_native_hot_replay_program(
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
        return self._batch_env.native_sub_batch_rollout_program(
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
        self._batch_env.prepare_native_branch_replay_from_history(
            history_rows=history_rows,
            horizon=horizon,
            stage_id=stage_id,
            future_random_mode=future_random_mode,
            future_random_seed=future_random_seed,
        )

    def reset_many(self, seeds: Sequence[int | None] | None = None, indices: Sequence[int] | None = None) -> None:
        self._batch_env.reset(indices=indices, seeds=seeds)

    def reset_at(self, index: int, seed: int | None = None) -> None:
        self._batch_env.reset_at(index, seed)

    def get_global_state_batch(
        self,
        *,
        indices: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ):
        return self._batch_env.get_global_state_batch(indices=indices, device=device)

    def export_runtime_state_batch(self, *, indices: Sequence[int] | None = None) -> list[dict[str, Any]]:
        return self._batch_env.export_runtime_state_batch(indices=indices)

    def load_runtime_state_batch(
        self,
        states: Sequence[dict[str, Any]],
        *,
        indices: Sequence[int] | None = None,
        refresh_observation_cache: bool = True,
        refresh_global_state_cache: bool = True,
    ) -> None:
        self._batch_env.load_runtime_state_batch(
            states,
            indices=indices,
            refresh_observation_cache=refresh_observation_cache,
            refresh_global_state_cache=refresh_global_state_cache,
        )

    def refresh_stage_obs_cache_many(self, indices: Sequence[int] | None = None) -> None:
        self._batch_env.refresh_stage_obs_cache_many(indices=indices)

    def current_obs_batch(
        self,
        *,
        indices: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ):
        return self._batch_env.current_obs_batch(indices=indices, device=device)

    def current_obs_many(self, indices: Sequence[int] | None = None):
        return self._batch_env.current_obs_many(indices=indices)

    def cluster_meta_batch(
        self,
        *,
        indices: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ):
        return self._batch_env.cluster_meta_batch(indices=indices, device=device)

    def cluster_meta_many(self, indices: Sequence[int] | None = None):
        return self._batch_env.cluster_meta_many(indices=indices)

    def sat_mask_to_ids_many(self, sat_masks: Sequence[np.ndarray], indices: Sequence[int] | None = None):
        return self._batch_env.sat_mask_to_ids_many(sat_masks, indices=indices)

    def last_reward_parts_many(self, indices: Sequence[int] | None = None):
        selected = self._drivers if indices is None else [self._drivers[int(index)] for index in indices]
        return [dict(getattr(driver.env, "last_reward_parts", {}) or {}) for driver in selected]

    def close(self) -> None:
        self._batch_env.close()


class GpuStructuredEnvGroup(_StructuredNativeBatchDriverGroup):
    pass


class GpuStructuredDriverGroup(_StructuredNativeBatchDriverGroup):
    pass


class PythonStructuredDriverGroup:
    """Small, portable structured driver group for CPU/Mac smoke runs.

    The native group intentionally exposes CUDA rollout hooks.  This group does
    not, so learners can distinguish it and fall back to the Python stage driver.
    It is meant for correctness/debugging on machines without NVIDIA CUDA, not
    for reproducing the fused native GPU throughput.
    """

    is_python_structured_driver_group = True

    def __init__(
        self,
        cfg,
        num_envs: int | None = None,
        *,
        tensor_device: torch.device | str | None = "cpu",
    ) -> None:
        if num_envs is None:
            raise ValueError("num_envs is required for Python structured driver groups")
        self._cfg = cfg
        self._num_envs = max(int(num_envs), 1)
        self._tensor_device = None if tensor_device is None else torch.device(tensor_device)
        self._envs: list[Any] = []
        self._drivers: list[StructuredControlDriver] = []
        for env_index in range(self._num_envs):
            env_cfg = copy.copy(cfg)
            if hasattr(env_cfg, "seed"):
                env_cfg.seed = int(getattr(cfg, "seed", 0) or 0) + int(env_index)
            env = SaginParallelEnv(env_cfg)
            env.reset(seed=int(getattr(env_cfg, "seed", 0) or 0))
            self._envs.append(env)
            self._drivers.append(StructuredControlDriver(env, tensor_device=self._tensor_device))

    def __len__(self) -> int:
        return len(self._drivers)

    def __getitem__(self, index: int) -> StructuredControlDriver:
        return self._drivers[int(index)]

    @property
    def drivers(self) -> list[StructuredControlDriver]:
        return self._drivers

    @property
    def envs(self) -> list[Any]:
        return self._envs

    @property
    def cfg(self):
        return self._cfg

    @property
    def tensor_device(self) -> torch.device | None:
        return self._tensor_device

    def set_tensor_device(self, device: torch.device | str | None) -> None:
        self._tensor_device = None if device is None else torch.device(device)
        for driver in self._drivers:
            driver.set_tensor_device(self._tensor_device)

    def reset_many(self, seeds: Sequence[int | None] | None = None, indices: Sequence[int] | None = None) -> None:
        selected = list(range(len(self._envs))) if indices is None else [int(index) for index in indices]
        if seeds is None:
            seeds_l = [None for _ in selected]
        else:
            seeds_l = [None if seed is None else int(seed) for seed in seeds]
        if len(seeds_l) != len(selected):
            raise ValueError("seeds length must match selected indices")
        for env_index, seed in zip(selected, seeds_l):
            env = self._envs[int(env_index)]
            if seed is None:
                env.reset()
            else:
                env.reset(seed=int(seed))
            self._drivers[int(env_index)] = StructuredControlDriver(env, tensor_device=self._tensor_device)

    def reset_at(self, index: int, seed: int | None = None) -> None:
        self.reset_many([seed], indices=[int(index)])

    def close(self) -> None:
        for env in self._envs:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()


class PythonStructuredEnvGroup(PythonStructuredDriverGroup):
    pass
