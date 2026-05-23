from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any, Callable

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch.profiler import ProfilerActivity

from sagin_marl.env.config import load_config
from sagin_marl.env.native_cuda import bindings as native_cuda_bindings
from sagin_marl.env.structured_driver import StructuredBatchStepResult
from sagin_marl.env.structured_gpu_rollout_runtime import StructuredGpuNativeRuntimeStepProgram
from sagin_marl.env.structured_kernel_runtime import StructuredKernelRuntime
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_mappo import StructuredMAPPO
from sagin_marl.rl.structured_system_benchmark import (
    summarize_structured_benchmark,
)
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_env_group


SEGMENT_ORDER = (
    "begin_step",
    "begin_accel_obs",
    "actor_accel",
    "accel_to_sat",
    "actor_sat",
    "sat_to_bw",
    "actor_bw",
    "finish_bw",
    "finish_step",
)

NATIVE_ENV_SEGMENTS = {"begin_accel_obs", "accel_to_sat", "sat_to_bw", "finish_bw"}
ENV_FORBIDDEN_TORCH_OPS = {
    "aten::copy_",
    "aten::gather",
    "aten::index",
    "aten::index_put_",
    "aten::index_select",
    "aten::scatter",
    "aten::index_copy_",
    "aten::index_fill_",
}
ACTOR_FORBIDDEN_TORCH_OPS = {
    "aten::copy_",
    "aten::gather",
    "aten::index",
    "aten::index_put_",
    "aten::index_select",
    "aten::scatter",
    "aten::index_copy_",
    "aten::index_fill_",
}

DEFAULT_OUTPUT_DIR = os.path.join("runs", "diagnostics", "native_rollout_profile")
FINISH_BASE_INTERNAL_SEGMENTS = (
    "arrival_and_flow_proxy",
    "access",
    "backhaul_setup",
    "gu_queue",
    "uav_queue",
    "sat_queue",
    "refresh_queue_derived",
    "reward_stats",
    "history_and_danger",
    "terminal_world",
    "commit_or_reset",
    "next_random_tape",
    "next_prepare_stage",
    "next_world",
    "next_snapshots",
    "next_accel_obs",
)
ACCEL_OBS_INTERNAL_SEGMENTS = (
    "accel_obs_ego_cell",
    "accel_obs_gu_tokens",
    "accel_obs_peer_tokens",
    "accel_obs_sat_tokens",
)
REFRESH_INTERNAL_SEGMENTS = (
    "refresh_mean_cost",
    "refresh_assoc",
    "refresh_candidate_access",
    "refresh_full_access_gain",
    "refresh_proxy_and_cost",
    "refresh_full_us_geometry",
    "refresh_visible_topk",
    "refresh_active_sat",
    "refresh_active_us",
    "refresh_last_selection",
)
QUEUE_REFRESH_INTERNAL_SEGMENTS = (
    "queue_refresh_mean_cost",
    "queue_refresh_proxy_and_cost",
    "queue_refresh_active_sat",
    "queue_refresh_full_us_sat_queue",
)
FINISH_INTERNAL_SEGMENTS = (
    FINISH_BASE_INTERNAL_SEGMENTS
    + ACCEL_OBS_INTERNAL_SEGMENTS
    + REFRESH_INTERNAL_SEGMENTS
    + QUEUE_REFRESH_INTERNAL_SEGMENTS
)


class StepSegmentProfiler:
    def __init__(self, *, profile_every: int, skip_steps: int, use_cuda_events: bool) -> None:
        self.profile_every = max(int(profile_every), 1)
        self.skip_steps = max(int(skip_steps), 0)
        self.use_cuda_events = bool(use_cuda_events)
        self.enabled = False
        self.reset()

    def reset(self) -> None:
        self.seen_steps = 0
        self.profiled_steps = 0
        self.segment_total_ms = {name: 0.0 for name in SEGMENT_ORDER}
        self.segment_counts = {name: 0 for name in SEGMENT_ORDER}
        self.total_step_ms = 0.0
        self.total_step_count = 0

    def should_profile_next_step(self) -> bool:
        self.seen_steps += 1
        if not self.enabled:
            return False
        if self.seen_steps <= self.skip_steps:
            return False
        ordinal = self.seen_steps - self.skip_steps
        return (ordinal - 1) % self.profile_every == 0

    def add_step(self, segment_ms: dict[str, float], total_ms: float) -> None:
        self.profiled_steps += 1
        self.total_step_count += 1
        self.total_step_ms += float(total_ms)
        for name in SEGMENT_ORDER:
            value = float(segment_ms.get(name, 0.0))
            self.segment_total_ms[name] += value
            self.segment_counts[name] += 1

    def rows(self) -> list[dict[str, float | int | str]]:
        total = max(float(self.total_step_ms), 1.0e-12)
        rows: list[dict[str, float | int | str]] = []
        for name in SEGMENT_ORDER:
            count = int(self.segment_counts.get(name, 0))
            total_ms = float(self.segment_total_ms.get(name, 0.0))
            rows.append(
                {
                    "segment": name,
                    "count": count,
                    "total_ms": total_ms,
                    "avg_ms_per_profiled_step": total_ms / max(count, 1),
                    "pct_of_total_step": 100.0 * total_ms / total,
                }
            )
        rows.append(
            {
                "segment": "total_step",
                "count": int(self.total_step_count),
                "total_ms": float(self.total_step_ms),
                "avg_ms_per_profiled_step": float(self.total_step_ms) / max(int(self.total_step_count), 1),
                "pct_of_total_step": 100.0,
            }
        )
        return rows

    def payload(self) -> dict[str, Any]:
        return {
            "profile_every": int(self.profile_every),
            "skip_steps": int(self.skip_steps),
            "use_cuda_events": bool(self.use_cuda_events),
            "seen_steps": int(self.seen_steps),
            "profiled_steps": int(self.profiled_steps),
            "rows": self.rows(),
        }


class KernelRuntimeProfiler:
    def __init__(self, *, use_cuda_events: bool) -> None:
        self.use_cuda_events = bool(use_cuda_events)
        self.enabled = False
        self.patched_native_binding_names: tuple[str, ...] = ()
        self.reset()

    def reset(self) -> None:
        self.profiled_steps = 0
        self.total_step_ms = 0.0
        self.kernel_total_ms: dict[str, float] = {}
        self.kernel_counts: dict[str, int] = {}
        self.finish_internal_tensor: torch.Tensor | None = None
        self._step_active = False
        self._pending_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._pending_cpu_ms: dict[str, float] = {}

    def begin_step(self, *, active: bool) -> None:
        self._step_active = bool(self.enabled and active)
        self._pending_events = []
        self._pending_cpu_ms = {}

    def abort_step(self) -> None:
        self._step_active = False
        self._pending_events = []
        self._pending_cpu_ms = {}

    def call(self, name: str, fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        force_out_of_step = bool(self.enabled and str(name) == "prepare_initial_accel_live")
        if not self._step_active and not force_out_of_step:
            return fn(*args, **kwargs)
        if force_out_of_step:
            if self.use_cuda_events and torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                result = fn(*args, **kwargs)
                end.record()
                end.synchronize()
                self.kernel_total_ms[str(name)] = self.kernel_total_ms.get(str(name), 0.0) + float(start.elapsed_time(end))
                self.kernel_counts[str(name)] = self.kernel_counts.get(str(name), 0) + 1
                return result
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            self.kernel_total_ms[str(name)] = self.kernel_total_ms.get(str(name), 0.0) + (
                time.perf_counter() - t0
            ) * 1000.0
            self.kernel_counts[str(name)] = self.kernel_counts.get(str(name), 0) + 1
            return result
        if self.use_cuda_events and torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn(*args, **kwargs)
            end.record()
            self._pending_events.append((str(name), start, end))
            return result
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        self._pending_cpu_ms[str(name)] = self._pending_cpu_ms.get(str(name), 0.0) + (
            time.perf_counter() - t0
        ) * 1000.0
        return result

    def finish_step_after_sync(self, *, total_step_ms: float) -> None:
        if not self._step_active:
            return
        self.profiled_steps += 1
        self.total_step_ms += float(total_step_ms)
        if self.use_cuda_events and torch.cuda.is_available():
            for name, start, end in self._pending_events:
                self.kernel_total_ms[name] = self.kernel_total_ms.get(name, 0.0) + float(start.elapsed_time(end))
                self.kernel_counts[name] = self.kernel_counts.get(name, 0) + 1
        else:
            for name, ms in self._pending_cpu_ms.items():
                self.kernel_total_ms[name] = self.kernel_total_ms.get(name, 0.0) + float(ms)
                self.kernel_counts[name] = self.kernel_counts.get(name, 0) + 1
        self.abort_step()

    def rows(self) -> list[dict[str, float | int | str]]:
        total_step_ms = max(float(self.total_step_ms), 1.0e-12)
        steps = max(int(self.profiled_steps), 1)
        names = sorted(
            self.kernel_counts,
            key=lambda item: float(self.kernel_total_ms.get(item, 0.0)),
            reverse=True,
        )
        rows: list[dict[str, float | int | str]] = []
        for name in names:
            count = int(self.kernel_counts.get(name, 0))
            total_ms = float(self.kernel_total_ms.get(name, 0.0))
            rows.append(
                {
                    "kernel": name,
                    "count": count,
                    "calls_per_profiled_step": float(count) / float(steps),
                    "total_ms": total_ms,
                    "avg_ms_per_call": total_ms / max(count, 1),
                    "avg_ms_per_profiled_step": total_ms / float(steps),
                    "pct_of_total_step": 100.0 * total_ms / total_step_ms,
                }
            )
        return rows

    def finish_profile_tensor_for(self, abi: Any) -> torch.Tensor:
        if self.finish_internal_tensor is not None:
            return self.finish_internal_tensor
        float_tensors = getattr(abi, "float_tensors", None)
        if not float_tensors:
            raise RuntimeError("finish internal profile requires a native ABI with float tensors.")
        device = float_tensors[0].device
        num_envs = int(getattr(abi, "int_params")[0])
        self.finish_internal_tensor = torch.zeros(
            (num_envs, len(FINISH_INTERNAL_SEGMENTS)),
            device=device,
            dtype=torch.float32,
        )
        return self.finish_internal_tensor

    def finish_internal_rows(self) -> list[dict[str, float | int | str]]:
        tensor = self.finish_internal_tensor
        if tensor is None:
            return []
        values = tensor.detach().sum(dim=0).cpu().tolist()[: len(FINISH_BASE_INTERNAL_SEGMENTS)]
        total = max(float(sum(values)), 1.0e-12)
        steps = max(int(self.profiled_steps), 1)
        env_steps = max(steps * int(tensor.size(0)), 1)
        rows: list[dict[str, float | int | str]] = []
        for idx, name in enumerate(FINISH_BASE_INTERNAL_SEGMENTS):
            total_cycles = float(values[idx]) if idx < len(values) else 0.0
            rows.append(
                {
                    "segment": str(name),
                    "index": int(idx),
                    "total_cycles": total_cycles,
                    "avg_cycles_per_profiled_step": total_cycles / float(steps),
                    "avg_cycles_per_env_step": total_cycles / float(env_steps),
                    "pct_of_finish_profiled_cycles": 100.0 * total_cycles / total,
                }
            )
        rows.sort(key=lambda row: float(row["total_cycles"]), reverse=True)
        return rows

    def accel_obs_internal_rows(self) -> list[dict[str, float | int | str]]:
        tensor = self.finish_internal_tensor
        if tensor is None:
            return []
        values = tensor.detach().sum(dim=0).cpu().tolist()
        offset = len(FINISH_BASE_INTERNAL_SEGMENTS)
        sub_values = values[offset : offset + len(ACCEL_OBS_INTERNAL_SEGMENTS)]
        total = max(float(sum(sub_values)), 1.0e-12)
        steps = max(int(self.profiled_steps), 1)
        env_steps = max(steps * int(tensor.size(0)), 1)
        rows: list[dict[str, float | int | str]] = []
        for local_idx, name in enumerate(ACCEL_OBS_INTERNAL_SEGMENTS):
            idx = offset + local_idx
            total_cycles = float(values[idx]) if idx < len(values) else 0.0
            rows.append(
                {
                    "segment": str(name),
                    "index": int(idx),
                    "total_cycles": total_cycles,
                    "avg_cycles_per_profiled_step": total_cycles / float(steps),
                    "avg_cycles_per_env_step": total_cycles / float(env_steps),
                    "pct_of_accel_obs_profiled_cycles": 100.0 * total_cycles / total,
                }
            )
        rows.sort(key=lambda row: float(row["total_cycles"]), reverse=True)
        return rows

    def _sub_profile_rows(
        self,
        *,
        offset: int,
        names: tuple[str, ...],
        pct_key: str,
    ) -> list[dict[str, float | int | str]]:
        tensor = self.finish_internal_tensor
        if tensor is None:
            return []
        values = tensor.detach().sum(dim=0).cpu().tolist()
        sub_values = values[offset : offset + len(names)]
        total = max(float(sum(sub_values)), 1.0e-12)
        steps = max(int(self.profiled_steps), 1)
        env_steps = max(steps * int(tensor.size(0)), 1)
        rows: list[dict[str, float | int | str]] = []
        for local_idx, name in enumerate(names):
            idx = offset + local_idx
            total_cycles = float(values[idx]) if idx < len(values) else 0.0
            rows.append(
                {
                    "segment": str(name),
                    "index": int(idx),
                    "total_cycles": total_cycles,
                    "avg_cycles_per_profiled_step": total_cycles / float(steps),
                    "avg_cycles_per_env_step": total_cycles / float(env_steps),
                    pct_key: 100.0 * total_cycles / total,
                }
            )
        rows.sort(key=lambda row: float(row["total_cycles"]), reverse=True)
        return rows

    def refresh_internal_rows(self) -> list[dict[str, float | int | str]]:
        return self._sub_profile_rows(
            offset=len(FINISH_BASE_INTERNAL_SEGMENTS) + len(ACCEL_OBS_INTERNAL_SEGMENTS),
            names=REFRESH_INTERNAL_SEGMENTS,
            pct_key="pct_of_refresh_profiled_cycles",
        )

    def queue_refresh_internal_rows(self) -> list[dict[str, float | int | str]]:
        return self._sub_profile_rows(
            offset=len(FINISH_BASE_INTERNAL_SEGMENTS)
            + len(ACCEL_OBS_INTERNAL_SEGMENTS)
            + len(REFRESH_INTERNAL_SEGMENTS),
            names=QUEUE_REFRESH_INTERNAL_SEGMENTS,
            pct_key="pct_of_queue_refresh_profiled_cycles",
        )

    def payload(self) -> dict[str, Any]:
        finish_rows = self.finish_internal_rows()
        accel_obs_rows = self.accel_obs_internal_rows()
        refresh_rows = self.refresh_internal_rows()
        queue_refresh_rows = self.queue_refresh_internal_rows()
        return {
            "use_cuda_events": bool(self.use_cuda_events),
            "profiled_steps": int(self.profiled_steps),
            "total_step_ms": float(self.total_step_ms),
            "patched_native_binding_names": list(self.patched_native_binding_names),
            "rows": self.rows(),
            "finish_internal_segments": list(FINISH_INTERNAL_SEGMENTS),
            "finish_internal_rows": finish_rows,
            "accel_obs_internal_segments": list(ACCEL_OBS_INTERNAL_SEGMENTS),
            "accel_obs_internal_rows": accel_obs_rows,
            "refresh_internal_segments": list(REFRESH_INTERNAL_SEGMENTS),
            "refresh_internal_rows": refresh_rows,
            "queue_refresh_internal_segments": list(QUEUE_REFRESH_INTERNAL_SEGMENTS),
            "queue_refresh_internal_rows": queue_refresh_rows,
        }


class TorchOpProfilerController:
    def __init__(self) -> None:
        self.enabled = False
        self.profiler: Any | None = None
        self.steps = 0

    def bind(self, profiler: Any | None) -> None:
        self.profiler = profiler

    def step(self) -> None:
        if not self.enabled or self.profiler is None:
            return
        self.steps += 1
        self.profiler.step()


def _run_with_timing(
    profiler: StepSegmentProfiler,
    name: str,
    fn: Callable[[], Any],
    events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] | None,
    cpu_ms: dict[str, float] | None,
) -> Any:
    if events is not None:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        events.append((name, start, end))
        return result
    if cpu_ms is not None:
        t0 = time.perf_counter()
        result = fn()
        cpu_ms[name] = cpu_ms.get(name, 0.0) + (time.perf_counter() - t0) * 1000.0
        return result
    return fn()


def _make_profiled_replay_step(
    profiler: StepSegmentProfiler,
    kernel_profiler: KernelRuntimeProfiler | None,
    torch_op_profiler: TorchOpProfilerController | None = None,
):
    def profiled_replay_step(
        self,
        *,
        actor_bridge: Any,
        deterministic: bool = False,
        rollout_tail: bool,
    ) -> StructuredBatchStepResult:
        if self.num_envs <= 0:
            raise RuntimeError("native GPU rollout program cannot replay a step for zero envs.")

        should_profile = profiler.should_profile_next_step()
        use_cuda = bool(should_profile and profiler.use_cuda_events and torch.cuda.is_available())
        events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] | None = [] if use_cuda else None
        cpu_ms: dict[str, float] | None = {} if should_profile and not use_cuda else None
        total_start = torch.cuda.Event(enable_timing=True) if use_cuda else None
        total_end = torch.cuda.Event(enable_timing=True) if use_cuda else None
        total_cpu_start = time.perf_counter() if should_profile and not use_cuda else 0.0
        if total_start is not None:
            total_start.record()
        if kernel_profiler is not None:
            kernel_profiler.begin_step(active=bool(should_profile))

        def timed(name: str, fn: Callable[[], Any]) -> Any:
            if torch_op_profiler is not None and torch_op_profiler.enabled:
                with torch.profiler.record_function(f"native_segment::{name}"):
                    return _run_with_timing(profiler, name, fn, events, cpu_ms)
            return _run_with_timing(profiler, name, fn, events, cpu_ms)

        try:
            if hasattr(actor_bridge, "begin_step"):
                timed(
                    "begin_step",
                    lambda: actor_bridge.begin_step(
                        deterministic=deterministic,
                    ),
                )

            accel_obs = timed(
                "begin_accel_obs",
                lambda: self._current_or_begin_accel_obs(),
            )
            if accel_obs is None:
                raise RuntimeError("native GPU rollout program did not publish accel obs.")

            timed(
                "actor_accel",
                lambda: actor_bridge.write_accel_action(
                    accel_obs,
                    runtime=self.runtime,
                    num_envs=self.num_envs,
                    deterministic=deterministic,
                ),
            )

            sat_publish_kwargs = (
                dict(actor_bridge.env_phase_b_kwargs())
                if hasattr(actor_bridge, "env_phase_b_kwargs")
                else {}
            )
            sat_obs, sat_max_select = timed(
                "accel_to_sat",
                lambda: self._after_accel_action(
                    max_visible=self.fixed_visible_sat_width,
                    **sat_publish_kwargs,
                ),
            )
            if sat_obs is None:
                raise RuntimeError("native GPU rollout program did not publish SAT obs.")

            timed(
                "actor_sat",
                lambda: actor_bridge.write_sat_action(
                    sat_obs,
                    runtime=self.runtime,
                    num_envs=self.num_envs,
                    sat_max_select=int(sat_max_select),
                    deterministic=deterministic,
                ),
            )

            bw_publish_kwargs = (
                dict(actor_bridge.env_phase_c_kwargs())
                if hasattr(actor_bridge, "env_phase_c_kwargs")
                else {}
            )
            bw_obs = timed(
                "sat_to_bw",
                lambda: self._after_sat_action(
                    max_visible=self.fixed_visible_sat_width,
                    **bw_publish_kwargs,
                ),
            )
            if bw_obs is None:
                raise RuntimeError("native GPU rollout program did not publish BW obs.")

            timed(
                "actor_bw",
                lambda: actor_bridge.write_bw_action(
                    bw_obs,
                    runtime=self.runtime,
                    num_envs=self.num_envs,
                    deterministic=deterministic,
                ),
            )

            finish_kwargs = {
                "max_visible": self.fixed_visible_sat_width,
                "rollout_tail": bool(rollout_tail),
            }
            if hasattr(actor_bridge, "env_phase_d_kwargs"):
                bridge_finish_kwargs = dict(actor_bridge.env_phase_d_kwargs())
                for key in ("record_rollout", "rollout_tail", "has_next_actor_slot", "prepare_next_accel"):
                    if key in bridge_finish_kwargs:
                        raise RuntimeError(f"native actor bridge must not override rollout scheduling key {key!r}.")
                finish_kwargs.update(bridge_finish_kwargs)
            step_result = timed("finish_bw", lambda: self._after_bw_action(**finish_kwargs))
            if not isinstance(step_result, StructuredBatchStepResult):
                raise RuntimeError("native GPU rollout program requires StructuredBatchStepResult.")

            if hasattr(actor_bridge, "finish_step"):
                timed(
                    "finish_step",
                    lambda: actor_bridge.finish_step(
                        step_result=step_result,
                        runtime=self.runtime,
                    ),
                )
        except Exception:
            if kernel_profiler is not None:
                kernel_profiler.abort_step()
            raise

        if should_profile:
            if total_end is not None and total_start is not None and events is not None:
                total_end.record()
                torch.cuda.synchronize()
                segment_ms = {segment_name: start.elapsed_time(end) for segment_name, start, end in events}
                total_ms = float(total_start.elapsed_time(total_end))
                profiler.add_step(segment_ms, total_ms)
                if kernel_profiler is not None:
                    kernel_profiler.finish_step_after_sync(total_step_ms=total_ms)
            elif cpu_ms is not None:
                total_ms = (time.perf_counter() - total_cpu_start) * 1000.0
                profiler.add_step(cpu_ms, total_ms)
                if kernel_profiler is not None:
                    kernel_profiler.finish_step_after_sync(total_step_ms=total_ms)
        if torch_op_profiler is not None:
            torch_op_profiler.step()
        return step_result

    return profiled_replay_step


@contextmanager
def patch_replay_step(
    profiler: StepSegmentProfiler,
    kernel_profiler: KernelRuntimeProfiler | None = None,
    torch_op_profiler: TorchOpProfilerController | None = None,
):
    original = StructuredGpuNativeRuntimeStepProgram.replay_step
    StructuredGpuNativeRuntimeStepProgram.replay_step = _make_profiled_replay_step(
        profiler,
        kernel_profiler,
        torch_op_profiler,
    )
    try:
        yield
    finally:
        StructuredGpuNativeRuntimeStepProgram.replay_step = original


@contextmanager
def patch_kernel_runtime(kernel_profiler: KernelRuntimeProfiler):
    original = StructuredKernelRuntime.compile_kernel

    def patched_compile_kernel(self, name: str, eager_fn: Callable[..., Any]) -> Callable[..., Any]:
        runner = original(self, name, eager_fn)
        if bool(getattr(runner, "_native_kernel_profile_wrapped", False)):
            return runner

        def profiled_runner(*args, **kwargs):
            return kernel_profiler.call(str(name), runner, args, kwargs)

        setattr(profiled_runner, "_native_kernel_profile_wrapped", True)
        setattr(profiled_runner, "_native_kernel_profile_name", str(name))
        cache = getattr(self, "_cache", None)
        if isinstance(cache, dict) and cache.get(name) is runner:
            cache[name] = profiled_runner
        return profiled_runner

    StructuredKernelRuntime.compile_kernel = patched_compile_kernel
    try:
        yield
    finally:
        StructuredKernelRuntime.compile_kernel = original


@contextmanager
def patch_native_cuda_bindings(kernel_profiler: KernelRuntimeProfiler):
    requested_names = (
        "prepare_initial_accel_live",
        "actor_accel_live",
        "accel_to_sat_live",
        "actor_sat_live",
        "actor_sat_pair_live",
        "sat_to_bw_live",
        "actor_bw_live",
        "finish_commit_prepare_live",
        "queue_aware_accel_live",
        "cluster_center_accel_live",
        "queue_aware_sat_live",
        "queue_aware_bw_live",
    )
    discovered_live_names = tuple(
        name
        for name in dir(native_cuda_bindings)
        if name.endswith("_live") and callable(getattr(native_cuda_bindings, name, None))
    )
    names = tuple(dict.fromkeys((*requested_names, *discovered_live_names)))
    originals = {name: getattr(native_cuda_bindings, name) for name in names if hasattr(native_cuda_bindings, name)}
    profiled_finish_fn = getattr(native_cuda_bindings, "finish_commit_prepare_live_profiled", None)
    kernel_profiler.patched_native_binding_names = tuple(sorted(originals))

    def make_profiled(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def profiled(*args, **kwargs):
            call_fn = fn
            call_kwargs = kwargs
            if (
                str(name) == "finish_commit_prepare_live"
                and kernel_profiler._step_active
                and callable(profiled_finish_fn)
            ):
                abi = args[0] if args else kwargs.get("abi")
                if abi is None:
                    raise RuntimeError("finish_commit_prepare_live profiler could not find ABI argument.")
                call_kwargs = dict(kwargs)
                call_kwargs["profile_out"] = kernel_profiler.finish_profile_tensor_for(abi)
                call_fn = profiled_finish_fn
            return kernel_profiler.call(str(name), call_fn, args, call_kwargs)

        return profiled

    for _name, _fn in originals.items():
        setattr(native_cuda_bindings, _name, make_profiled(_name, _fn))
    try:
        yield
    finally:
        for _name, _fn in originals.items():
            setattr(native_cuda_bindings, _name, _fn)


def _resolve_torch_device(device_arg: str) -> torch.device:
    requested = str(device_arg).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def _build_learner(cfg, *, device: torch.device, hidden_dim: int, embed_dim: int) -> StructuredMAPPO:
    bundle = build_structured_modules_from_config(cfg, hidden_dim=hidden_dim, embed_dim=embed_dim)
    actor = bundle.actor
    critic = bundle.critic
    actor_optim = torch.optim.Adam(actor.parameters(), lr=float(getattr(cfg, "actor_lr", 3.0e-4) or 3.0e-4))
    critic_optim = torch.optim.Adam(critic.parameters(), lr=float(getattr(cfg, "critic_lr", 1.0e-3) or 1.0e-3))
    return StructuredMAPPO(
        actor=actor,
        critic=critic,
        gamma=float(getattr(cfg, "gamma", 0.99) or 0.99),
        gae_lambda=float(getattr(cfg, "gae_lambda", 0.95) or 0.95),
        clip_ratio=float(getattr(cfg, "clip_ratio", 0.2) or 0.2),
        value_coef=float(getattr(cfg, "value_coef", 0.5) or 0.5),
        entropy_coef=float(getattr(cfg, "entropy_coef", 0.01) or 0.01),
        max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5) or 0.5),
        ppo_epochs=max(int(getattr(cfg, "ppo_epochs", 1) or 1), 1),
        num_mini_batch=max(int(getattr(cfg, "num_mini_batch", 1) or 1), 1),
        target_mode="step_level",
        actor_optimizer=actor_optim,
        critic_optimizer=critic_optim,
        device=device,
        cfg=cfg,
        train_accel=bool(getattr(cfg, "train_accel", True)),
        train_sat=bool(getattr(cfg, "train_sat", True)),
        train_bw=bool(getattr(cfg, "train_bw", True)),
        exec_accel_source=str(getattr(cfg, "exec_accel_source", "policy") or "policy"),
        exec_sat_source=str(getattr(cfg, "exec_sat_source", "policy") or "policy"),
        exec_bw_source=str(getattr(cfg, "exec_bw_source", "policy") or "policy"),
    )


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_csv_with_fields(path: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ensure_parent_dir(path: str | None) -> None:
    if not path:
        return
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _print_segment_table(rows: list[dict[str, Any]]) -> None:
    print("native_step_segment_profile")
    print(f"{'segment':<18} {'count':>8} {'avg_ms/step':>14} {'total_ms':>14} {'pct':>8}")
    for row in rows:
        print(
            f"{str(row['segment']):<18} "
            f"{int(row['count']):>8d} "
            f"{float(row['avg_ms_per_profiled_step']):>14.4f} "
            f"{float(row['total_ms']):>14.4f} "
            f"{float(row['pct_of_total_step']):>7.2f}%"
        )


def _print_kernel_table(rows: list[dict[str, Any]], *, limit: int) -> None:
    print("native_kernel_profile")
    print(
        f"{'kernel':<42} {'count':>8} {'calls/step':>10} "
        f"{'avg_ms/call':>12} {'avg_ms/step':>13} {'total_ms':>14} {'pct':>8}"
    )
    for row in rows[: max(int(limit), 0)]:
        print(
            f"{str(row['kernel']):<42} "
            f"{int(row['count']):>8d} "
            f"{float(row['calls_per_profiled_step']):>10.3f} "
            f"{float(row['avg_ms_per_call']):>12.4f} "
            f"{float(row['avg_ms_per_profiled_step']):>13.4f} "
            f"{float(row['total_ms']):>14.4f} "
            f"{float(row['pct_of_total_step']):>7.2f}%"
        )


def _print_finish_internal_table(rows: list[dict[str, Any]]) -> None:
    print("finish_internal_profile")
    print(
        f"{'segment':<34} {'idx':>4} {'avg_cycles/step':>18} "
        f"{'avg_cycles/env-step':>20} {'total_cycles':>18} {'pct':>8}"
    )
    for row in rows:
        print(
            f"{str(row['segment']):<34} "
            f"{int(row['index']):>4d} "
            f"{float(row['avg_cycles_per_profiled_step']):>18.1f} "
            f"{float(row['avg_cycles_per_env_step']):>20.1f} "
            f"{float(row['total_cycles']):>18.1f} "
            f"{float(row['pct_of_finish_profiled_cycles']):>7.2f}%"
        )


def _print_accel_obs_internal_table(rows: list[dict[str, Any]]) -> None:
    print("accel_obs_internal_profile")
    print(
        f"{'segment':<34} {'idx':>4} {'avg_cycles/step':>18} "
        f"{'avg_cycles/env-step':>20} {'total_cycles':>18} {'pct':>8}"
    )
    for row in rows:
        print(
            f"{str(row['segment']):<34} "
            f"{int(row['index']):>4d} "
            f"{float(row['avg_cycles_per_profiled_step']):>18.1f} "
            f"{float(row['avg_cycles_per_env_step']):>20.1f} "
            f"{float(row['total_cycles']):>18.1f} "
            f"{float(row['pct_of_accel_obs_profiled_cycles']):>7.2f}%"
        )


def _print_refresh_internal_table(rows: list[dict[str, Any]]) -> None:
    print("refresh_internal_profile")
    print(
        f"{'segment':<34} {'idx':>4} {'avg_cycles/step':>18} "
        f"{'avg_cycles/env-step':>20} {'total_cycles':>18} {'pct':>8}"
    )
    for row in rows:
        print(
            f"{str(row['segment']):<34} "
            f"{int(row['index']):>4d} "
            f"{float(row['avg_cycles_per_profiled_step']):>18.1f} "
            f"{float(row['avg_cycles_per_env_step']):>20.1f} "
            f"{float(row['total_cycles']):>18.1f} "
            f"{float(row['pct_of_refresh_profiled_cycles']):>7.2f}%"
        )


def _print_queue_refresh_internal_table(rows: list[dict[str, Any]]) -> None:
    print("queue_refresh_internal_profile")
    print(
        f"{'segment':<34} {'idx':>4} {'avg_cycles/step':>18} "
        f"{'avg_cycles/env-step':>20} {'total_cycles':>18} {'pct':>8}"
    )
    for row in rows:
        print(
            f"{str(row['segment']):<34} "
            f"{int(row['index']):>4d} "
            f"{float(row['avg_cycles_per_profiled_step']):>18.1f} "
            f"{float(row['avg_cycles_per_env_step']):>20.1f} "
            f"{float(row['total_cycles']):>18.1f} "
            f"{float(row['pct_of_queue_refresh_profiled_cycles']):>7.2f}%"
        )


def _event_cuda_time_us(event: Any, *, self_time: bool) -> float:
    names = (
        ("self_cuda_time_total", "self_device_time_total")
        if self_time
        else ("cuda_time_total", "device_time_total")
    )
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 0.0


def _event_cpu_time_us(event: Any, *, self_time: bool) -> float:
    names = ("self_cpu_time_total",) if self_time else ("cpu_time_total",)
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 0.0


def _segment_parent_name(event: Any) -> str:
    parent = getattr(event, "cpu_parent", None)
    while parent is not None:
        parent_name = str(getattr(parent, "name", "") or "")
        prefix = "native_segment::"
        if parent_name.startswith(prefix):
            return parent_name[len(prefix) :]
        parent = getattr(parent, "cpu_parent", None)
    return "unattributed"


def _torch_profiler_op_rows(profiler: Any, *, include_cpu: bool = False) -> list[dict[str, Any]]:
    aggregate: dict[str, dict[str, Any]] = {}
    try:
        events = profiler.events()
    except Exception:
        return []
    for event in events:
        op_name = str(getattr(event, "name", "") or "")
        if op_name.startswith("native_segment::"):
            continue
        if _segment_parent_name(event) not in NATIVE_ENV_SEGMENTS:
            continue
        self_cuda_us = _event_cuda_time_us(event, self_time=True)
        cuda_us = _event_cuda_time_us(event, self_time=False)
        self_cpu_us = _event_cpu_time_us(event, self_time=True)
        cpu_us = _event_cpu_time_us(event, self_time=False)
        if not include_cpu and max(self_cuda_us, cuda_us) <= 0.0:
            continue
        row = aggregate.get(op_name)
        if row is None:
            row = {
                "op": op_name,
                "count": 0,
                "self_cuda_ms": 0.0,
                "cuda_ms": 0.0,
                "self_cpu_ms": 0.0,
                "cpu_ms": 0.0,
            }
            aggregate[op_name] = row
        row["count"] = int(row["count"]) + 1
        row["self_cuda_ms"] = float(row["self_cuda_ms"]) + self_cuda_us / 1000.0
        row["cuda_ms"] = float(row["cuda_ms"]) + cuda_us / 1000.0
        row["self_cpu_ms"] = float(row["self_cpu_ms"]) + self_cpu_us / 1000.0
        row["cpu_ms"] = float(row["cpu_ms"]) + cpu_us / 1000.0
    rows = list(aggregate.values())
    for row in rows:
        count = max(int(row["count"]), 1)
        row["avg_self_cuda_us"] = float(row["self_cuda_ms"]) * 1000.0 / float(count)
        row["avg_cuda_us"] = float(row["cuda_ms"]) * 1000.0 / float(count)
        row["avg_self_cpu_us"] = float(row["self_cpu_ms"]) * 1000.0 / float(count)
        row["avg_cpu_us"] = float(row["cpu_ms"]) * 1000.0 / float(count)
    rows.sort(key=lambda row: (float(row["self_cuda_ms"]), float(row["cuda_ms"])), reverse=True)
    return rows


def _torch_profiler_segment_op_rows(profiler: Any, *, include_cpu: bool = False) -> list[dict[str, Any]]:
    aggregate: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        events = profiler.events()
    except Exception:
        return []
    for event in events:
        op_name = str(getattr(event, "name", "") or "")
        if op_name.startswith("native_segment::"):
            continue
        self_cuda_us = _event_cuda_time_us(event, self_time=True)
        cuda_us = _event_cuda_time_us(event, self_time=False)
        self_cpu_us = _event_cpu_time_us(event, self_time=True)
        cpu_us = _event_cpu_time_us(event, self_time=False)
        if not include_cpu and max(self_cuda_us, cuda_us) <= 0.0:
            continue
        segment = _segment_parent_name(event)
        key = (segment, op_name)
        row = aggregate.get(key)
        if row is None:
            row = {
                "segment": segment,
                "op": op_name,
                "count": 0,
                "self_cuda_ms": 0.0,
                "cuda_ms": 0.0,
                "self_cpu_ms": 0.0,
                "cpu_ms": 0.0,
            }
            aggregate[key] = row
        row["count"] = int(row["count"]) + 1
        row["self_cuda_ms"] = float(row["self_cuda_ms"]) + self_cuda_us / 1000.0
        row["cuda_ms"] = float(row["cuda_ms"]) + cuda_us / 1000.0
        row["self_cpu_ms"] = float(row["self_cpu_ms"]) + self_cpu_us / 1000.0
        row["cpu_ms"] = float(row["cpu_ms"]) + cpu_us / 1000.0
    rows = list(aggregate.values())
    for row in rows:
        count = max(int(row["count"]), 1)
        row["avg_self_cuda_us"] = float(row["self_cuda_ms"]) * 1000.0 / float(count)
        row["avg_cuda_us"] = float(row["cuda_ms"]) * 1000.0 / float(count)
        row["avg_self_cpu_us"] = float(row["self_cpu_ms"]) * 1000.0 / float(count)
        row["avg_cpu_us"] = float(row["cpu_ms"]) * 1000.0 / float(count)
    rows.sort(
        key=lambda row: (
            str(row["segment"]),
            -float(row["self_cuda_ms"]),
            -float(row["cuda_ms"]),
        )
    )
    return rows


def _print_op_table(rows: list[dict[str, Any]], *, limit: int) -> None:
    print("torch_profiler_env_cuda_op_profile")
    print(
        f"{'op':<42} {'count':>8} {'self_cuda_ms':>14} "
        f"{'cuda_ms':>12} {'avg_cuda_us':>13} {'self_cpu_ms':>13}"
    )
    for row in rows[: max(int(limit), 0)]:
        print(
            f"{str(row['op']):<42} "
            f"{int(row['count']):>8d} "
            f"{float(row['self_cuda_ms']):>14.4f} "
            f"{float(row['cuda_ms']):>12.4f} "
            f"{float(row['avg_cuda_us']):>13.4f} "
            f"{float(row['self_cpu_ms']):>13.4f}"
        )


def _print_segment_op_table(rows: list[dict[str, Any]], *, limit: int) -> None:
    print("torch_profiler_segment_cuda_op_profile")
    print(
        f"{'segment':<18} {'op':<34} {'count':>8} "
        f"{'self_cuda_ms':>14} {'cuda_ms':>12} {'avg_cuda_us':>13}"
    )
    for row in rows[: max(int(limit), 0)]:
        print(
            f"{str(row['segment']):<18} "
            f"{str(row['op']):<34} "
            f"{int(row['count']):>8d} "
            f"{float(row['self_cuda_ms']):>14.4f} "
            f"{float(row['cuda_ms']):>12.4f} "
            f"{float(row['avg_cuda_us']):>13.4f}"
        )


def _forbidden_native_op_rows(segment_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for row in segment_rows:
        segment = str(row.get("segment", ""))
        op = str(row.get("op", ""))
        if segment in NATIVE_ENV_SEGMENTS and op in ENV_FORBIDDEN_TORCH_OPS:
            violations.append({**row, "contract": "env_native_segment"})
        elif segment in {"actor_accel", "actor_sat", "actor_bw"} and op in ACTOR_FORBIDDEN_TORCH_OPS:
            violations.append({**row, "contract": "actor_history_bridge"})
    return violations


def _print_forbidden_op_contract(violations: list[dict[str, Any]]) -> None:
    print("native_forbidden_op_contract")
    if not violations:
        print("status=passed violations=0")
        return
    print(f"status=failed violations={len(violations)}")
    for row in violations[:20]:
        print(
            f"{str(row.get('contract', '')):<24} "
            f"{str(row.get('segment', '')):<18} "
            f"{str(row.get('op', '')):<30} "
            f"count={int(row.get('count', 0) or 0)}"
        )


def _run_native_rollout_only(
    env_group: Any,
    learner: StructuredMAPPO,
    *,
    rollout_env_steps: int,
    reset_seed: int,
    device: torch.device,
) -> list[StructuredBatchStepResult]:
    num_envs = len(env_group) if hasattr(env_group, "__len__") else 1
    reset_many = getattr(env_group, "reset_many", None)
    if callable(reset_many):
        reset_many([int(reset_seed) + i for i in range(int(num_envs))])
    set_tensor_device = getattr(env_group, "set_tensor_device", None)
    if callable(set_tensor_device):
        set_tensor_device(device)
    learner.begin_native_rollout(
        env_group,
        rollout_env_steps=int(rollout_env_steps),
        num_envs=int(num_envs),
    )
    buffer = StructuredRolloutBuffer()
    return learner.collect_env_horizon_native_tensor_policy(
        env_group,
        buffer,
        horizon=int(rollout_env_steps),
        deterministic=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile native rollout replay_step segment timing.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-updates", type=int, default=1)
    parser.add_argument("--warmup-updates", type=int, default=1)
    parser.add_argument("--rollout-env-steps", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--env-tensor-backend",
        choices=["device", "config", "auto", "cpu", "cuda"],
        default="device",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--sample-interval-sec", type=float, default=0.05)
    parser.add_argument("--profile-every", type=int, default=1)
    parser.add_argument("--skip-profile-steps", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bench-json-path", type=str, default=None)
    parser.add_argument("--segment-json-path", type=str, default=None)
    parser.add_argument("--segment-csv-path", type=str, default=None)
    parser.add_argument("--kernel-json-path", type=str, default=None)
    parser.add_argument("--kernel-csv-path", type=str, default=None)
    parser.add_argument("--kernel-table-limit", type=int, default=32)
    parser.add_argument("--torch-profiler", action="store_true")
    parser.add_argument("--torch-profiler-active-steps", type=int, default=20)
    parser.add_argument("--torch-profiler-warmup-steps", type=int, default=1)
    parser.add_argument("--torch-profiler-record-shapes", action="store_true")
    parser.add_argument("--torch-profiler-profile-memory", action="store_true")
    parser.add_argument("--op-json-path", type=str, default=None)
    parser.add_argument("--op-csv-path", type=str, default=None)
    parser.add_argument("--segment-op-csv-path", type=str, default=None)
    parser.add_argument("--trace-path", type=str, default=None)
    parser.add_argument("--op-table-limit", type=int, default=40)
    parser.add_argument("--op-include-cpu", action="store_true")
    parser.add_argument("--exec-accel-source", type=str, default=None)
    parser.add_argument("--exec-sat-source", type=str, default=None)
    parser.add_argument("--exec-bw-source", type=str, default=None)
    parser.add_argument(
        "--avoidance-enabled",
        choices=["config", "true", "false"],
        default="config",
        help="Diagnostic override for the regular avoidance safety layer.",
    )
    parser.add_argument(
        "--native-safety-shield",
        choices=["config", "true", "false"],
        default="config",
        help="Diagnostic override for safety_shield_enabled + safety_shield_solver=NATIVE_CUDA.",
    )
    args = parser.parse_args()

    output_dir = str(args.output_dir or DEFAULT_OUTPUT_DIR)
    args.bench_json_path = args.bench_json_path or os.path.join(output_dir, "bench.json")
    args.segment_json_path = args.segment_json_path or os.path.join(output_dir, "step_segments.json")
    args.segment_csv_path = args.segment_csv_path or os.path.join(output_dir, "step_segments.csv")
    args.kernel_json_path = args.kernel_json_path or os.path.join(output_dir, "kernel_segments.json")
    args.kernel_csv_path = args.kernel_csv_path or os.path.join(output_dir, "kernel_segments.csv")
    args.op_json_path = args.op_json_path or os.path.join(output_dir, "torch_profiler_ops.json")
    args.op_csv_path = args.op_csv_path or os.path.join(output_dir, "torch_profiler_ops.csv")
    args.segment_op_csv_path = args.segment_op_csv_path or os.path.join(
        output_dir,
        "torch_profiler_segment_ops.csv",
    )
    args.forbidden_op_csv_path = os.path.join(output_dir, "native_forbidden_ops.csv")

    cfg = load_config(args.config)
    if str(args.avoidance_enabled) != "config":
        cfg.avoidance_enabled = str(args.avoidance_enabled) == "true"
    if str(args.native_safety_shield) != "config":
        native_shield_enabled = str(args.native_safety_shield) == "true"
        cfg.safety_shield_enabled = bool(native_shield_enabled)
        if native_shield_enabled:
            cfg.safety_shield_solver = "NATIVE_CUDA"
            # Native shield replaces the regular avoidance branch in the CUDA
            # accel step kernel; keep the diagnostic modes mutually exclusive.
            cfg.avoidance_enabled = False
    if args.exec_accel_source is not None:
        cfg.exec_accel_source = str(args.exec_accel_source)
        if str(cfg.exec_accel_source).strip().lower() != "policy":
            cfg.train_accel = False
    if args.exec_sat_source is not None:
        cfg.exec_sat_source = str(args.exec_sat_source)
        if str(cfg.exec_sat_source).strip().lower() != "policy":
            cfg.train_sat = False
    if args.exec_bw_source is not None:
        cfg.exec_bw_source = str(args.exec_bw_source)
        if str(cfg.exec_bw_source).strip().lower() not in {"policy", "policy_single_uav_queue_aware"}:
            cfg.train_bw = False
    device = _resolve_torch_device(str(args.device))
    env_tensor_backend = str(args.env_tensor_backend).strip().lower()
    if env_tensor_backend == "device":
        cfg.structured_env_tensor_backend = "cuda" if device.type == "cuda" else "cpu"
    elif env_tensor_backend != "config":
        cfg.structured_env_tensor_backend = env_tensor_backend
    rollout_env_steps = (
        int(args.rollout_env_steps)
        if args.rollout_env_steps is not None
        else int(getattr(cfg, "buffer_size", 32) or 32)
    )

    profiler = StepSegmentProfiler(
        profile_every=max(int(args.profile_every), 1),
        skip_steps=max(int(args.skip_profile_steps), 0),
        use_cuda_events=bool(device.type == "cuda"),
    )
    kernel_profiler = KernelRuntimeProfiler(use_cuda_events=bool(device.type == "cuda"))
    torch_op_profiler = TorchOpProfilerController()
    torch_profiler_obj = None
    rows = []
    profiled_env_steps = 0
    rollout_collect_time_sec = 0.0

    env_group = None
    try:
        with patch_kernel_runtime(kernel_profiler), patch_native_cuda_bindings(kernel_profiler), patch_replay_step(
            profiler,
            kernel_profiler,
            torch_op_profiler,
        ):
            env_group = make_structured_env_group(cfg, num_envs=max(int(args.num_envs), 1), backend="sync", mode="train")
            learner = _build_learner(
                cfg,
                device=device,
                hidden_dim=max(int(args.hidden_dim), 1),
                embed_dim=max(int(args.embed_dim), 1),
            )
            if int(args.warmup_updates) > 0:
                profiler.enabled = False
                kernel_profiler.enabled = False
                _run_native_rollout_only(
                    env_group,
                    learner,
                    rollout_env_steps=max(int(rollout_env_steps), 1),
                    reset_seed=int(cfg.seed) + 1_000_000,
                    device=device,
                )
            profiler.reset()
            kernel_profiler.reset()
            torch_op_profiler.bind(None)
            profiler.enabled = True
            kernel_profiler.enabled = True
            if bool(args.torch_profiler):
                activities = [ProfilerActivity.CPU]
                if device.type == "cuda" and torch.cuda.is_available():
                    activities.append(ProfilerActivity.CUDA)
                schedule = torch.profiler.schedule(
                    wait=0,
                    warmup=max(int(args.torch_profiler_warmup_steps), 0),
                    active=max(int(args.torch_profiler_active_steps), 1),
                    repeat=1,
                )
                with torch.profiler.profile(
                    activities=activities,
                    schedule=schedule,
                    record_shapes=bool(args.torch_profiler_record_shapes),
                    profile_memory=bool(args.torch_profiler_profile_memory),
                    with_stack=False,
                ) as active_profiler:
                    torch_profiler_obj = active_profiler
                    torch_op_profiler.bind(active_profiler)
                    torch_op_profiler.enabled = True
                    for update_index in range(max(int(args.num_updates), 1)):
                        collect_start = time.perf_counter()
                        _run_native_rollout_only(
                            env_group,
                            learner,
                            rollout_env_steps=max(int(rollout_env_steps), 1),
                            reset_seed=int(cfg.seed) + int(update_index) * max(int(args.num_envs), 1) * 1000,
                            device=device,
                        )
                        rollout_collect_time_sec += max(time.perf_counter() - collect_start, 0.0)
                        profiled_env_steps += max(int(rollout_env_steps), 1) * max(int(args.num_envs), 1)
                    torch_op_profiler.enabled = False
                    torch_op_profiler.bind(None)
            else:
                for update_index in range(max(int(args.num_updates), 1)):
                    collect_start = time.perf_counter()
                    _run_native_rollout_only(
                        env_group,
                        learner,
                        rollout_env_steps=max(int(rollout_env_steps), 1),
                        reset_seed=int(cfg.seed) + int(update_index) * max(int(args.num_envs), 1) * 1000,
                        device=device,
                    )
                    rollout_collect_time_sec += max(time.perf_counter() - collect_start, 0.0)
                    profiled_env_steps += max(int(rollout_env_steps), 1) * max(int(args.num_envs), 1)
    finally:
        if env_group is not None:
            close_structured_env_group(env_group)

    row_payloads = [asdict(row) for row in rows]
    benchmark_summary = summarize_structured_benchmark(rows)
    benchmark_summary.update(
        {
            "updates": float(max(int(args.num_updates), 1)),
            "env_steps": float(profiled_env_steps),
            "transition_samples": float(profiled_env_steps),
            "env_steps_per_sec": float(profiled_env_steps) / max(float(rollout_collect_time_sec), 1.0e-12),
            "samples_per_sec": float(profiled_env_steps) / max(float(rollout_collect_time_sec), 1.0e-12),
            "rollout_collect_time_sec": float(rollout_collect_time_sec),
            "rollout_total_time_sec": float(rollout_collect_time_sec),
            "update_total_time_sec": 0.0,
            "iteration_time_sec": float(rollout_collect_time_sec),
        }
    )
    segment_payload = profiler.payload()
    kernel_payload = kernel_profiler.payload()
    op_rows: list[dict[str, Any]] = []
    segment_op_rows: list[dict[str, Any]] = []
    if torch_profiler_obj is not None:
        op_rows = _torch_profiler_op_rows(torch_profiler_obj, include_cpu=bool(args.op_include_cpu))
        segment_op_rows = _torch_profiler_segment_op_rows(
            torch_profiler_obj,
            include_cpu=bool(args.op_include_cpu),
        )
        if args.trace_path:
            _ensure_parent_dir(args.trace_path)
            torch_profiler_obj.export_chrome_trace(args.trace_path)
    forbidden_op_rows = _forbidden_native_op_rows(segment_op_rows)
    segment_rows = segment_payload["rows"]
    kernel_rows = kernel_payload["rows"]

    if args.bench_json_path:
        _ensure_parent_dir(args.bench_json_path)
        with open(args.bench_json_path, "w", encoding="utf-8") as f:
            json.dump({"rows": row_payloads, "summary": benchmark_summary}, f, indent=2, ensure_ascii=False)
    if args.segment_json_path:
        _ensure_parent_dir(args.segment_json_path)
        with open(args.segment_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "args": vars(args),
                    "benchmark_summary": benchmark_summary,
                    "segment_profile": segment_payload,
                    "kernel_profile": kernel_payload,
                    "torch_profiler": {
                        "enabled": bool(args.torch_profiler),
                        "steps": int(torch_op_profiler.steps),
                        "op_rows": op_rows,
                        "segment_op_rows": segment_op_rows,
                        "forbidden_op_rows": forbidden_op_rows,
                    },
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    if args.segment_csv_path:
        _write_csv(args.segment_csv_path, segment_rows)
    if args.kernel_json_path:
        _ensure_parent_dir(args.kernel_json_path)
        with open(args.kernel_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "args": vars(args),
                    "benchmark_summary": benchmark_summary,
                    "kernel_profile": kernel_payload,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    if args.kernel_csv_path:
        if kernel_rows:
            _write_csv(args.kernel_csv_path, kernel_rows)
        else:
            with open(args.kernel_csv_path, "w", encoding="utf-8", newline="") as f:
                f.write("kernel,count,calls_per_profiled_step,total_ms,avg_ms_per_call,avg_ms_per_profiled_step,pct_of_total_step\n")
    if bool(args.torch_profiler):
        op_fields = [
            "op",
            "count",
            "self_cuda_ms",
            "cuda_ms",
            "avg_self_cuda_us",
            "avg_cuda_us",
            "self_cpu_ms",
            "cpu_ms",
            "avg_self_cpu_us",
            "avg_cpu_us",
        ]
        segment_op_fields = ["segment", *op_fields]
        _write_csv_with_fields(args.op_csv_path, op_rows, op_fields)
        _write_csv_with_fields(args.segment_op_csv_path, segment_op_rows, segment_op_fields)
        _write_csv_with_fields(args.forbidden_op_csv_path, forbidden_op_rows, ["contract", *segment_op_fields])
        if args.op_json_path:
            _ensure_parent_dir(args.op_json_path)
            with open(args.op_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "args": vars(args),
                        "benchmark_summary": benchmark_summary,
                        "torch_profiler": {
                            "enabled": True,
                            "steps": int(torch_op_profiler.steps),
                            "op_rows": op_rows,
                            "segment_op_rows": segment_op_rows,
                            "forbidden_op_rows": forbidden_op_rows,
                        },
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

    print(
        "structured_benchmark_profiled "
        f"updates={int(benchmark_summary['updates'])} "
        f"device={device.type} "
        f"env_tensor_backend={str(getattr(cfg, 'structured_env_tensor_backend', 'unknown'))} "
        f"env_steps_per_sec={float(benchmark_summary['env_steps_per_sec']):.2f} "
        f"samples_per_sec={float(benchmark_summary['samples_per_sec']):.2f}"
    )
    print(
        "structured_benchmark_profiled_timing "
        f"rollout={float(benchmark_summary['rollout_total_time_sec']):.4f}s "
        f"update={float(benchmark_summary['update_total_time_sec']):.4f}s "
        f"iter={float(benchmark_summary['iteration_time_sec']):.4f}s"
    )
    print(
        "segment_profile_meta "
        f"seen_steps={int(segment_payload['seen_steps'])} "
        f"profiled_steps={int(segment_payload['profiled_steps'])} "
        f"profile_every={int(segment_payload['profile_every'])} "
        f"use_cuda_events={bool(segment_payload['use_cuda_events'])}"
    )
    _print_segment_table(segment_rows)
    print(
        "kernel_profile_meta "
        f"profiled_steps={int(kernel_payload['profiled_steps'])} "
        f"use_cuda_events={bool(kernel_payload['use_cuda_events'])} "
        f"kernels={len(kernel_rows)}"
    )
    _print_kernel_table(kernel_rows, limit=int(args.kernel_table_limit))
    finish_internal_rows = kernel_payload.get("finish_internal_rows", [])
    if finish_internal_rows:
        _print_finish_internal_table(finish_internal_rows)
    accel_obs_internal_rows = kernel_payload.get("accel_obs_internal_rows", [])
    if accel_obs_internal_rows:
        _print_accel_obs_internal_table(accel_obs_internal_rows)
    refresh_internal_rows = kernel_payload.get("refresh_internal_rows", [])
    if refresh_internal_rows:
        _print_refresh_internal_table(refresh_internal_rows)
    queue_refresh_internal_rows = kernel_payload.get("queue_refresh_internal_rows", [])
    if queue_refresh_internal_rows:
        _print_queue_refresh_internal_table(queue_refresh_internal_rows)
    if bool(args.torch_profiler):
        print(
            "torch_profiler_meta "
            f"steps={int(torch_op_profiler.steps)} "
            f"active_steps={max(int(args.torch_profiler_active_steps), 1)} "
            f"warmup_steps={max(int(args.torch_profiler_warmup_steps), 0)} "
            f"ops={len(op_rows)} "
            f"segment_ops={len(segment_op_rows)}"
        )
        _print_op_table(op_rows, limit=int(args.op_table_limit))
        _print_segment_op_table(segment_op_rows, limit=int(args.op_table_limit))
        _print_forbidden_op_contract(forbidden_op_rows)
        if forbidden_op_rows:
            raise RuntimeError("native rollout torch profiler forbidden op contract failed.")


if __name__ == "__main__":
    main()
