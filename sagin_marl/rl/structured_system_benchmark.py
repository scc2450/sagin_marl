from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
import subprocess
import threading
import time
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from sagin_marl.rl.structured_train import (
    StructuredTrainStepMetrics,
    _make_episode_stats_state,
    run_structured_training,
)


def _safe_gpu_utilization(device: torch.device | None) -> tuple[float, float]:
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return float("nan"), float("nan")
    index = int(device.index if device.index is not None else torch.cuda.current_device())
    try:
        gpu_util = float(torch.cuda.utilization(index))
        gpu_mem_util = float(torch.cuda.memory_usage(index))
        return gpu_util, gpu_mem_util
    except Exception:
        pass
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        used_percent = 100.0 * (1.0 - float(free_bytes) / max(float(total_bytes), 1.0))
        return float("nan"), float(used_percent)
    except Exception:
        return float("nan"), float("nan")


def _safe_gpu_utilization_from_nvidia_smi(device: torch.device | None) -> tuple[float, float]:
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return float("nan"), float("nan")
    index = int(device.index if device.index is not None else torch.cuda.current_device())
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        first_line = completed.stdout.strip().splitlines()[0]
        gpu_util_raw, gpu_mem_util_raw = [part.strip() for part in first_line.split(",", maxsplit=1)]
        return float(gpu_util_raw), float(gpu_mem_util_raw)
    except Exception:
        return float("nan"), float("nan")


class _ProcessUtilizationSampler:
    def __init__(self, *, device: torch.device | None, sample_interval_sec: float = 0.05) -> None:
        self._device = device
        self._sample_interval_sec = max(float(sample_interval_sec), 0.01)
        self._cpu_count = max(int(os.cpu_count() or 1), 1)
        self._samples_cpu: list[float] = []
        self._samples_gpu: list[float] = []
        self._samples_gpu_mem: list[float] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._prev_wall: float | None = None
        self._prev_cpu_time: float | None = None

    def _sample_once(self) -> None:
        now_wall = time.perf_counter()
        now_cpu = time.process_time()
        if self._prev_wall is not None and self._prev_cpu_time is not None:
            wall_delta = max(now_wall - self._prev_wall, 1.0e-9)
            cpu_delta = max(now_cpu - self._prev_cpu_time, 0.0)
            cpu_util = 100.0 * cpu_delta / wall_delta / float(self._cpu_count)
            gpu_util, gpu_mem_util = _safe_gpu_utilization(self._device)
            with self._lock:
                self._samples_cpu.append(float(cpu_util))
                self._samples_gpu.append(float(gpu_util))
                self._samples_gpu_mem.append(float(gpu_mem_util))
        self._prev_wall = now_wall
        self._prev_cpu_time = now_cpu

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self._sample_interval_sec)
        self._sample_once()

    def start(self) -> None:
        self._prev_wall = None
        self._prev_cpu_time = None
        self._samples_cpu.clear()
        self._samples_gpu.clear()
        self._samples_gpu_mem.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="structured-benchmark-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        with self._lock:
            cpu = np.asarray(self._samples_cpu, dtype=np.float64)
            gpu = np.asarray(self._samples_gpu, dtype=np.float64)
            gpu_mem = np.asarray(self._samples_gpu_mem, dtype=np.float64)
        gpu_util = float(np.nanmean(gpu)) if np.isfinite(gpu).any() else float("nan")
        gpu_mem_util = float(np.nanmean(gpu_mem)) if np.isfinite(gpu_mem).any() else float("nan")
        if not np.isfinite(gpu_util):
            fallback_gpu_util, fallback_gpu_mem_util = _safe_gpu_utilization_from_nvidia_smi(self._device)
            gpu_util = fallback_gpu_util
            if not np.isfinite(gpu_mem_util):
                gpu_mem_util = fallback_gpu_mem_util
        return {
            "cpu_util_percent": float(np.mean(cpu)) if cpu.size > 0 else 0.0,
            "gpu_util_percent": gpu_util,
            "gpu_memory_util_percent": gpu_mem_util,
        }


@dataclass
class StructuredSystemBenchmarkRow:
    update: int
    env_reward_mean: float
    policy_loss: float
    value_loss: float
    env_steps: int
    transition_samples: int
    env_steps_per_sec: float
    samples_per_sec: float
    rollout_reset_time_sec: float
    rollout_collect_time_sec: float
    rollout_total_time_sec: float
    rollout_view_build_time_sec: float
    update_prepare_time_sec: float
    update_optimize_time_sec: float
    update_total_time_sec: float
    iteration_time_sec: float
    cpu_util_percent: float
    gpu_util_percent: float
    gpu_memory_util_percent: float


def benchmark_structured_training(
    env_group,
    learner,
    *,
    num_updates: int,
    warmup_updates: int = 0,
    rollout_env_steps: int,
    reset_seed: int | None = None,
    reset_controller=None,
    episode_stat_window: int = 100,
    trace_fn=None,
    trace_interval: int = 0,
    before_update_callback=None,
    sample_interval_sec: float = 0.05,
) -> List[StructuredSystemBenchmarkRow]:
    if num_updates <= 0:
        return []
    env_count = len(env_group) if hasattr(env_group, "__len__") else 1
    episode_stat_window = max(int(episode_stat_window), 1)

    def _new_episode_stats_state():
        return _make_episode_stats_state(
            num_envs=max(int(env_count), 1),
            episode_stat_window=episode_stat_window,
        )

    if warmup_updates > 0:
        warmup_stats_state = _new_episode_stats_state()
        for warmup_idx in range(int(warmup_updates)):
            run_structured_training(
                env_group,
                learner,
                num_updates=1,
                rollout_env_steps=rollout_env_steps,
                reset_seed=(
                    None
                    if reset_seed is None
                    else int(reset_seed) + 1_000_000 + int(warmup_idx) * max(int(env_count), 1) * 1000
                ),
                reset_on_start=True,
                reset_controller=reset_controller,
                episode_stats_state=warmup_stats_state,
                episode_stat_window=episode_stat_window,
                trace_fn=None,
                trace_interval=0,
                before_update_callback=before_update_callback,
            )

    episode_stats_state = _new_episode_stats_state()
    monitor = _ProcessUtilizationSampler(
        device=getattr(learner, "device", None),
        sample_interval_sec=sample_interval_sec,
    )
    rows: List[StructuredSystemBenchmarkRow] = []
    for update_idx in range(int(num_updates)):
        monitor.start()
        try:
            history = run_structured_training(
                env_group,
                learner,
                num_updates=1,
                rollout_env_steps=rollout_env_steps,
                reset_seed=(
                    None
                    if reset_seed is None
                    else int(reset_seed) + int(update_idx) * max(int(env_count), 1) * 1000
                ),
                reset_on_start=bool(update_idx == 0),
                reset_controller=reset_controller,
                episode_stats_state=episode_stats_state,
                episode_stat_window=max(int(episode_stat_window), 1),
                trace_fn=trace_fn,
                trace_interval=max(int(trace_interval), 0),
                before_update_callback=before_update_callback,
            )
        finally:
            util = monitor.stop()
        row: StructuredTrainStepMetrics = history[-1]
        rows.append(
            StructuredSystemBenchmarkRow(
                update=int(update_idx + 1),
                env_reward_mean=float(row.env_reward_mean),
                policy_loss=float(row.policy_loss),
                value_loss=float(row.value_loss),
                env_steps=int(row.env_steps),
                transition_samples=int(row.transition_samples),
                env_steps_per_sec=float(row.env_steps_per_sec),
                samples_per_sec=float(row.samples_per_sec),
                rollout_reset_time_sec=float(row.rollout_reset_time_sec),
                rollout_collect_time_sec=float(row.rollout_collect_time_sec),
                rollout_total_time_sec=float(row.rollout_total_time_sec),
                rollout_view_build_time_sec=float(row.rollout_view_build_time_sec),
                update_prepare_time_sec=float(row.update_prepare_time_sec),
                update_optimize_time_sec=float(row.update_optimize_time_sec),
                update_total_time_sec=float(row.update_total_time_sec),
                iteration_time_sec=float(row.iteration_time_sec),
                cpu_util_percent=float(util["cpu_util_percent"]),
                gpu_util_percent=float(util["gpu_util_percent"]),
                gpu_memory_util_percent=float(util["gpu_memory_util_percent"]),
            )
        )
    return rows


def summarize_structured_benchmark(rows: Sequence[StructuredSystemBenchmarkRow]) -> Dict[str, float]:
    if not rows:
        return {
            "updates": 0.0,
            "env_steps_per_sec": 0.0,
            "samples_per_sec": 0.0,
            "rollout_total_time_sec": 0.0,
            "update_total_time_sec": 0.0,
            "cpu_util_percent": 0.0,
            "gpu_util_percent": float("nan"),
            "gpu_memory_util_percent": float("nan"),
        }
    payloads = [asdict(row) for row in rows]
    keys = [key for key in payloads[0] if key != "update"]
    summary: Dict[str, float] = {"updates": float(len(rows))}
    for key in keys:
        values = np.asarray([float(payload[key]) for payload in payloads], dtype=np.float64)
        summary[key] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    return summary
