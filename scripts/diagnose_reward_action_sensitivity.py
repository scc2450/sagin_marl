from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import (
    StructuredMAPPO,
    _StructuredMAPPOGpuActorBridge,
    _VsRefFirstActionOverrideBridge,
    _index_dataclass,
)
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver_group


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _summ(x: np.ndarray | torch.Tensor) -> dict[str, float]:
    arr = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10.0)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    xa = np.asarray(x, dtype=np.float64).reshape(-1)
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if int(mask.sum()) <= 2:
        return 0.0
    xa = xa[mask]
    ya = ya[mask]
    sx = float(np.std(xa))
    sy = float(np.std(ya))
    if sx <= 1.0e-12 or sy <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def _as_reward_vector(result: Any, mode: str, *, cfg: Any, device: torch.device) -> torch.Tensor:
    key = str(mode or "env_reward").strip().lower()

    def _field(name: str) -> torch.Tensor:
        value = getattr(result, name, None)
        if not torch.is_tensor(value):
            raise RuntimeError(f"branch result missing {name!r} for reward mode {mode!r}.")
        return value.to(device=device, dtype=torch.float32).reshape(-1)

    def _part(name: str) -> torch.Tensor:
        parts = getattr(result, "reward_part_tensors", None)
        if not isinstance(parts, dict) or name not in parts or not torch.is_tensor(parts[name]):
            raise RuntimeError(f"branch result missing reward_part_tensors[{name!r}] for reward mode {mode!r}.")
        return parts[name].to(device=device, dtype=torch.float32).reshape(-1)

    if key == "env_reward":
        return _field("team_rewards")
    if key in {"access_raw", "bw_access"}:
        return _field("bw_access_rewards")
    if key == "weighted_workload_delta":
        return _field("bw_weighted_workload_delta_rewards")
    if key == "weighted_workload_level":
        return _field("bw_weighted_workload_level_rewards")
    if key == "positive_weighted_workload_level":
        level = _field("bw_weighted_workload_level_rewards")
        workload = torch.clamp(-level, min=0.0)
        return 1.0 / (1.0 + torch.log1p(workload))
    if key == "relative_weighted_workload_delta":
        delta = _field("bw_weighted_workload_delta_rewards")
        level = _field("bw_weighted_workload_level_rewards")
        workload_before = torch.clamp(delta - level, min=0.0)
        return delta / torch.clamp(workload_before, min=1.0)
    if key == "controllable_flow":
        x_acc = _part("x_acc")
        x_rel = _part("x_rel")
        d_pre = _part("d_pre")
        backlog = _part("pre_backlog_steps_eval")
        overflow = _part("overflow_risk_mean")
        service_gap = _part("service_gap_risk_mean")
        return (
            float(getattr(cfg, "reward_w_access", 0.5) or 0.0) * x_acc
            + float(getattr(cfg, "reward_w_relay", 0.5) or 0.0) * x_rel
            - float(getattr(cfg, "reward_w_pre_drop", 1.0) or 0.0) * d_pre
            - float(getattr(cfg, "reward_w_pre_backlog", 0.08) or 0.0) * torch.log1p(torch.clamp(backlog, min=0.0))
            - float(getattr(cfg, "reward_w_pre_overflow_risk", 0.0) or 0.0) * overflow
            - float(getattr(cfg, "reward_w_pre_service_gap", 0.0) or 0.0) * service_gap
        ).to(dtype=torch.float32)
    if key == "accel_access_safe":
        x_acc = _part("x_acc")
        d_pre = _part("d_pre")
        backlog = _part("pre_backlog_steps_eval")
        close_risk = _part("close_risk")
        collision = _part("collision_event")
        return (
            x_acc
            - 1.0 * d_pre
            - 0.05 * torch.log1p(torch.clamp(backlog, min=0.0))
            - 0.2 * close_risk
            - 2.0 * collision
        ).to(dtype=torch.float32)
    if key == "accel_delta_safe":
        delta = _field("bw_weighted_workload_delta_rewards")
        close_risk = _part("close_risk")
        collision = _part("collision_event")
        return (delta - 10.0 * close_risk - 100.0 * collision).to(dtype=torch.float32)
    if key == "accel_growth_safe":
        x_acc = _part("x_acc")
        g_pre = _part("g_pre")
        d_pre = _part("d_pre")
        close_risk = _part("close_risk")
        collision = _part("collision_event")
        return (
            x_acc
            - torch.relu(g_pre)
            - 1.0 * d_pre
            - 0.5 * close_risk
            - 2.0 * collision
        ).to(dtype=torch.float32)
    if key == "accel_pressure_safe":
        x_acc = _part("x_acc")
        d_pre = _part("d_pre")
        overflow = _part("overflow_risk_mean")
        downstream = _part("downstream_pressure_mean")
        close_risk = _part("close_risk")
        collision = _part("collision_event")
        return (
            x_acc
            - 1.0 * d_pre
            - 0.25 * overflow
            - 0.25 * downstream
            - 0.5 * close_risk
            - 2.0 * collision
        ).to(dtype=torch.float32)
    if key == "accel_queue_relief":
        g_pre = _part("g_pre")
        d_pre = _part("d_pre")
        close_risk = _part("close_risk")
        collision = _part("collision_event")
        return (-g_pre - 1.0 * d_pre - 0.5 * close_risk - 2.0 * collision).to(dtype=torch.float32)
    if key == "sat_relay_processed":
        x_rel = _part("x_rel")
        processed = _part("processed_ratio_eval")
        drop_eval = _part("drop_ratio_eval")
        overlap = _part("sat_overlap_eval")
        return (0.5 * x_rel + 0.5 * processed - 1.0 * drop_eval - 0.05 * overlap).to(dtype=torch.float32)
    if key == "sat_backhaul_drop":
        x_rel = _part("x_rel")
        d_pre = _part("d_pre")
        overlap = _part("sat_overlap_eval")
        return (x_rel - 1.0 * d_pre - 0.05 * overlap).to(dtype=torch.float32)
    if key == "bw_access_drop":
        x_acc = _part("x_acc")
        d_pre = _part("d_pre")
        service_gap = _part("service_gap_risk_mean")
        return (x_acc - 1.0 * d_pre - 0.2 * service_gap).to(dtype=torch.float32)
    if key == "bw_relative_access":
        x_acc = _part("x_acc")
        backlog = torch.clamp(_part("pre_backlog_steps_eval"), min=0.0)
        d_pre = _part("d_pre")
        return (x_acc / torch.clamp(backlog + 1.0, min=1.0) - 0.5 * d_pre).to(dtype=torch.float32)
    if key == "bw_growth_drop":
        x_acc = _part("x_acc")
        g_pre = _part("g_pre")
        d_pre = _part("d_pre")
        service_gap = _part("service_gap_risk_mean")
        return (x_acc - torch.relu(g_pre) - 2.0 * d_pre - 0.5 * service_gap).to(dtype=torch.float32)
    if key == "bw_queue_relief":
        g_pre = _part("g_pre")
        d_pre = _part("d_pre")
        service_gap = _part("service_gap_risk_mean")
        return (-g_pre - 1.0 * d_pre - 0.5 * service_gap).to(dtype=torch.float32)
    if key == "bw_access_pressure":
        x_acc = _part("x_acc")
        d_pre = _part("d_pre")
        overflow = _part("overflow_risk_mean")
        service_gap = _part("service_gap_risk_mean")
        return (x_acc - 2.0 * d_pre - 0.5 * overflow - 0.5 * service_gap).to(dtype=torch.float32)
    raise RuntimeError(f"unsupported reward mode {mode!r}")


def _discounted_returns_for_modes(
    results: Sequence[Any],
    *,
    reward_modes: Sequence[str],
    cfg: Any,
    gamma: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not results:
        return {str(mode): torch.zeros((0,), dtype=torch.float32, device=device) for mode in reward_modes}
    first = _as_reward_vector(results[0], str(reward_modes[0]), cfg=cfg, device=device)
    returns = {str(mode): torch.zeros_like(first) for mode in reward_modes}
    alive = torch.ones_like(first, dtype=torch.float32)
    discount = 1.0
    for result in results:
        done_t = result.terminated.to(device=device, dtype=torch.bool).reshape(-1)
        done_t = done_t | result.truncated.to(device=device, dtype=torch.bool).reshape(-1)
        for mode in reward_modes:
            reward_t = _as_reward_vector(result, str(mode), cfg=cfg, device=device)
            returns[str(mode)] = returns[str(mode)] + alive * float(discount) * reward_t
        alive = alive * (~done_t).to(dtype=torch.float32)
        discount *= float(gamma)
    return returns


@dataclass
class _FirstAccelActionBridge:
    base_bridge: _StructuredMAPPOGpuActorBridge
    first_actions: torch.Tensor
    step_index: int = 0

    def bind_source_modes(self, runtime: Any) -> None:
        self.base_bridge.bind_source_modes(runtime)

    def begin_horizon(self, *, horizon: int, runtime: Any, deterministic: bool) -> None:
        self.bind_source_modes(runtime)
        self.base_bridge.begin_horizon(horizon=horizon, runtime=runtime, deterministic=deterministic)

    def end_horizon(self, *, results: Sequence[Any], runtime: Any) -> None:
        del results, runtime

    def __call__(self, *, step_index: int, runtime: Any):
        del runtime
        self.step_index = int(step_index)
        return self

    def begin_step(self, *, deterministic: bool) -> None:
        self.base_bridge.begin_step(deterministic=deterministic)

    def write_accel_action(self, accel_obs: Any, *, runtime: Any, num_envs: int, deterministic: bool) -> None:
        if int(self.step_index) != 0:
            self.base_bridge.write_accel_action(
                accel_obs,
                runtime=runtime,
                num_envs=num_envs,
                deterministic=True,
            )
            return
        del accel_obs, deterministic
        action_dst = getattr(runtime.main, "live_accel_action", None)
        if not torch.is_tensor(action_dst):
            raise RuntimeError("branch replay requires live_accel_action buffer.")
        action_dst.copy_(self.first_actions.to(device=action_dst.device, dtype=action_dst.dtype).reshape_as(action_dst))
        logprob_dst = getattr(runtime.main, "live_accel_old_logprob", None)
        if torch.is_tensor(logprob_dst):
            logprob_dst.zero_()

    def write_sat_action(self, *args, **kwargs) -> None:
        self.base_bridge.write_sat_action(*args, **kwargs)

    def write_bw_action(self, *args, **kwargs) -> None:
        self.base_bridge.write_bw_action(*args, **kwargs)


def _branch_returns_from_history_for_modes(
    learner: StructuredMAPPO,
    *,
    stage_id: int,
    history_rows: Sequence[int],
    first_actions: torch.Tensor,
    horizons: Sequence[int],
    reward_modes: Sequence[str],
) -> dict[str, torch.Tensor]:
    rollout_program = getattr(learner, "_native_rollout_program", None)
    drivers = None if rollout_program is None else getattr(rollout_program, "drivers", None)
    runtime = None if rollout_program is None else getattr(rollout_program, "runtime", None)
    history = None if runtime is None else getattr(runtime, "history", None)
    if drivers is None or history is None:
        raise RuntimeError("native rollout program/history is required for branch replay.")
    source_num_envs = int(getattr(history, "num_envs", 0) or 0)
    if source_num_envs <= 0:
        raise RuntimeError("cannot infer source rollout env count.")
    sub_batch_program_factory = getattr(drivers, "native_sub_batch_rollout_program", None)
    prepare_branch = getattr(drivers, "prepare_native_branch_replay_from_history", None)
    if not callable(sub_batch_program_factory) or not callable(prepare_branch):
        raise RuntimeError("native branch replay support is unavailable.")

    rows_tuple = tuple(int(x) for x in history_rows)
    horizons_t = torch.as_tensor(list(horizons), dtype=torch.long, device=learner.device).reshape(-1)
    if len(rows_tuple) != int(first_actions.shape[0]) or len(rows_tuple) != int(horizons_t.numel()):
        raise ValueError("history_rows, first_actions, and horizons must have the same length.")
    first_actions = first_actions.detach().to(device=learner.device, dtype=torch.float32)
    out = {
        str(mode): torch.empty((len(rows_tuple),), dtype=torch.float32, device=learner.device)
        for mode in reward_modes
    }
    stage_id_i = int(stage_id)
    if stage_id_i not in {0, 1, 2}:
        raise RuntimeError(f"stage_id must be 0, 1, or 2, got {stage_id_i}.")
    learner._ensure_native_actor_cuda_bindings(sync=False)
    for horizon_i in torch.unique(horizons_t).detach().cpu().tolist():
        horizon_i = max(int(horizon_i), 1)
        group_idx_t = torch.nonzero(horizons_t == horizon_i, as_tuple=False).reshape(-1)
        group_idx = group_idx_t.detach().cpu().numpy().astype(np.int64).tolist()
        chunk_rows = tuple(rows_tuple[i] for i in group_idx)
        chunk_actions = first_actions.index_select(0, group_idx_t.to(device=learner.device)).contiguous()
        selected_envs = tuple(int(row % source_num_envs) for row in chunk_rows)
        program = sub_batch_program_factory(
            capacity=horizon_i,
            selected_indices=selected_envs,
            allow_duplicate_indices=True,
        )
        sub_runtime = program.runtime
        begin_horizon = getattr(program._step_program.executor, "_runtime_begin_horizon", None)
        if not callable(begin_horizon):
            raise RuntimeError("sub-batch program cannot begin a horizon.")
        begin_horizon(num_steps=horizon_i)
        program._step_program._horizon_started = True
        prepare_branch(history_rows=chunk_rows, horizon=horizon_i, stage_id=stage_id_i)
        bridge = _VsRefFirstActionOverrideBridge(
            base_bridge=_StructuredMAPPOGpuActorBridge(learner),
            stage_id=stage_id_i,
            first_actions=chunk_actions,
        )
        bridge.begin_horizon(horizon=horizon_i, runtime=sub_runtime, deterministic=True)
        results: list[Any] = []
        if stage_id_i == 0:
            step_result = program.replay_step(
                actor_bridge=bridge(step_index=0, runtime=sub_runtime),
                deterministic=True,
                rollout_tail=bool(horizon_i <= 1),
            )
            step_view = sub_runtime.result.step_result_views[0] if sub_runtime.result.step_result_views else step_result
            results.append(step_view)
        elif stage_id_i == 1:
            step_bridge = bridge(step_index=0, runtime=sub_runtime)
            step_bridge.begin_step(deterministic=True)
            sat_obs = sub_runtime.main.live_sat_obs
            if sat_obs is None:
                raise RuntimeError("SAT branch replay missing SAT live obs at snapshot.")
            step_bridge.write_sat_action(
                sat_obs,
                runtime=sub_runtime,
                num_envs=int(len(chunk_rows)),
                sat_max_select=int(getattr(sub_runtime.main, "sat_max_select", 1) or 1),
                deterministic=True,
            )
            bw_obs = program._step_program.executor._runtime_step_publish_bw_obs(
                max_visible=program.fixed_visible_sat_width
            )
            step_bridge.write_bw_action(
                bw_obs,
                runtime=sub_runtime,
                num_envs=int(len(chunk_rows)),
                deterministic=True,
            )
            first_result = program._step_program.executor._runtime_step_finish_bw(
                max_visible=program.fixed_visible_sat_width,
                rollout_tail=bool(horizon_i <= 1),
            )
            step_view = sub_runtime.result.step_result_views[0] if sub_runtime.result.step_result_views else first_result
            results.append(step_view)
        else:
            step_bridge = bridge(step_index=0, runtime=sub_runtime)
            step_bridge.begin_step(deterministic=True)
            bw_obs = sub_runtime.main.live_bw_obs
            if bw_obs is None:
                raise RuntimeError("BW branch replay missing BW live obs at snapshot.")
            step_bridge.write_bw_action(
                bw_obs,
                runtime=sub_runtime,
                num_envs=int(len(chunk_rows)),
                deterministic=True,
            )
            first_result = program._step_program.executor._runtime_step_finish_bw(
                max_visible=program.fixed_visible_sat_width,
                rollout_tail=bool(horizon_i <= 1),
            )
            step_view = sub_runtime.result.step_result_views[0] if sub_runtime.result.step_result_views else first_result
            results.append(step_view)
        for step_index in range(1, horizon_i):
            step_result = program.replay_step(
                actor_bridge=bridge(step_index=step_index, runtime=sub_runtime),
                deterministic=True,
                rollout_tail=bool(step_index + 1 >= horizon_i),
            )
            step_view = (
                sub_runtime.result.step_result_views[step_index]
                if step_index < len(sub_runtime.result.step_result_views)
                else step_result
            )
            results.append(step_view)
        bridge.end_horizon(results=results, runtime=sub_runtime)
        returns_by_mode = _discounted_returns_for_modes(
            results,
            reward_modes=reward_modes,
            cfg=learner.cfg,
            gamma=float(learner.gamma),
            device=learner.device,
        )
        for mode, values in returns_by_mode.items():
            if not bool(torch.isfinite(values).all().detach().cpu().item()):
                raise RuntimeError(f"non-finite branch returns for mode {mode!r}.")
            out[str(mode)].index_copy_(0, group_idx_t.to(device=learner.device), values)
    return out


def _reward_diagnostics(matrix: np.ndarray, labels: Sequence[str]) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    row_mean = np.mean(values, axis=1)
    state_scale = float(np.std(row_mean))
    action_var = np.var(values, axis=1)
    action_scale = float(math.sqrt(max(float(np.mean(action_var)), 0.0)))
    total_scale = float(np.std(values.reshape(-1)))
    action_snr = action_scale / max(state_scale, 1.0e-12)
    action_total_frac = action_scale / max(total_scale, 1.0e-12)
    ref = values[:, 0]
    sample = values[:, 1] if values.shape[1] > 1 else ref
    best = np.max(values, axis=1)
    best_idx = np.argmax(values, axis=1)
    label_counts: dict[str, int] = {}
    for idx in best_idx.tolist():
        label = str(labels[int(idx)])
        label_counts[label] = int(label_counts.get(label, 0) + 1)
    row_centered_abs = np.abs(values - row_mean[:, None])
    return {
        "return": _summ(values),
        "state_scale": state_scale,
        "action_scale": action_scale,
        "total_scale": total_scale,
        "action_snr": action_snr,
        "action_total_frac": action_total_frac,
        "pairwise_gap_mean_abs_centered": float(np.mean(row_centered_abs)),
        "best_minus_ref": _summ(best - ref),
        "sample_minus_ref": _summ(sample - ref),
        "best_positive_frac": float(np.mean((best - ref) > 0.0)),
        "sample_positive_frac": float(np.mean((sample - ref) > 0.0)),
        "best_label_counts": label_counts,
        "corr_ref_vs_best_gap": _corr(ref, best - ref),
    }


def _atanh_scaled_action(action: torch.Tensor, scale: float) -> torch.Tensor:
    scale_f = max(float(scale), 1.0e-12)
    t = torch.clamp(action.to(dtype=torch.float32) / scale_f, -1.0 + 1.0e-4, 1.0 - 1.0e-4)
    return 0.5 * (torch.log1p(t) - torch.log1p(-t))


def _ppo_score_proxy_diagnostics(
    matrix: np.ndarray,
    *,
    score_proxy: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(matrix, dtype=np.float64)
    scores = np.asarray(score_proxy, dtype=np.float64)
    if values.ndim != 2 or scores.ndim != 3:
        raise ValueError("values must be [rows, actions], score_proxy must be [rows, actions, score_dim].")
    if values.shape[:2] != scores.shape[:2]:
        raise ValueError("values and score_proxy row/action dimensions must match.")
    if values.shape[0] <= 0 or values.shape[1] <= 1:
        return {
            "oracle_grad_signal_norm": 0.0,
            "oracle_grad_noise_norm": 0.0,
            "oracle_grad_snr": 0.0,
            "required_batch_multiplier_for_snr3": 0.0,
            "cancellation_ratio": 0.0,
            "mean_state_grad_norm": 0.0,
            "score_cov_norm": 0.0,
        }
    oracle_adv = values - np.mean(values, axis=1, keepdims=True)
    grad_samples = oracle_adv[..., None] * scores
    flat = grad_samples.reshape(-1, grad_samples.shape[-1])
    signal = np.mean(flat, axis=0)
    signal_norm = float(np.linalg.norm(signal))
    centered = flat - signal[None, :]
    sample_count = max(int(flat.shape[0]), 1)
    noise_norm = float(math.sqrt(max(float(np.sum(np.var(centered, axis=0)) / sample_count), 0.0)))
    snr = signal_norm / max(noise_norm, 1.0e-12)
    state_grad = np.mean(grad_samples, axis=1)
    mean_state_grad_norm = float(np.mean(np.linalg.norm(state_grad, axis=1)))
    cancellation_ratio = signal_norm / max(mean_state_grad_norm, 1.0e-12)
    # This is the norm of Cov(A_oracle, score) in the squashed-Gaussian latent mean space.
    score_cov = np.mean(oracle_adv[..., None] * scores, axis=(0, 1))
    score_cov_norm = float(np.linalg.norm(score_cov))
    required_multiplier = float((3.0 / max(snr, 1.0e-12)) ** 2) if snr > 0.0 else float("inf")
    return {
        "oracle_grad_signal_norm": signal_norm,
        "oracle_grad_noise_norm": noise_norm,
        "oracle_grad_snr": snr,
        "required_batch_multiplier_for_snr3": required_multiplier,
        "cancellation_ratio": cancellation_ratio,
        "mean_state_grad_norm": mean_state_grad_norm,
        "score_cov_norm": score_cov_norm,
    }


def _stage_module(actor: torch.nn.Module, stage_id: int) -> torch.nn.Module:
    if int(stage_id) == 0:
        return actor.accel_policy
    if int(stage_id) == 1:
        return actor.sat_subset_policy
    if int(stage_id) == 2:
        return actor.bw_policy
    raise ValueError(f"unsupported stage_id={stage_id!r}")


def _flat_grad_vector(
    loss: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
) -> torch.Tensor:
    grads = torch.autograd.grad(
        loss,
        list(params),
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    pieces: list[torch.Tensor] = []
    for param, grad in zip(params, grads):
        if grad is None:
            pieces.append(torch.zeros(param.numel(), dtype=param.dtype, device=param.device))
        else:
            pieces.append(grad.detach().reshape(-1))
    if not pieces:
        return torch.zeros((0,), dtype=torch.float32, device=loss.device)
    return torch.cat([piece.to(dtype=torch.float32) for piece in pieces], dim=0)


def _stage_policy_gradient_vector(
    learner: StructuredMAPPO,
    *,
    stage_id: int,
    local_batch: Any,
    joint_actions: torch.Tensor,
    advantages: torch.Tensor,
    num_agents: int,
    params: Sequence[torch.nn.Parameter],
) -> torch.Tensor:
    if int(joint_actions.shape[0]) != int(advantages.numel()):
        raise ValueError("joint_actions first dimension must match advantages.")
    logprob, _entropy, _out = learner._stage_actor_eval_from_batch(
        int(stage_id),
        local_batch,
        joint_actions,
        int(num_agents),
    )
    adv = advantages.to(device=logprob.device, dtype=logprob.dtype).reshape_as(logprob).detach()
    loss = -(adv * logprob).mean()
    return _flat_grad_vector(loss, params)


def _true_autograd_gradient_diagnostics(
    learner: StructuredMAPPO,
    *,
    stage_id: int,
    selected_local: Any,
    stochastic_joint_actions: torch.Tensor,
    oracle_advantage: np.ndarray,
    num_agents: int,
    chunk_count: int,
) -> dict[str, float]:
    actions = stochastic_joint_actions.detach().to(device=learner.device)
    sample_count = int(actions.shape[0])
    stochastic_count = int(actions.shape[1])
    if sample_count <= 0 or stochastic_count <= 1:
        return {
            "true_grad_signal_norm": 0.0,
            "true_grad_noise_norm": 0.0,
            "true_grad_snr": 0.0,
            "true_required_batch_multiplier_for_snr3": 0.0,
            "true_cancellation_ratio": 0.0,
            "true_mean_state_grad_norm": 0.0,
            "true_chunk_count": 0.0,
            "true_state_count": float(sample_count),
        }
    module = _stage_module(learner.actor, int(stage_id))
    params = [param for param in module.parameters() if param.requires_grad]
    if not params:
        return {
            "true_grad_signal_norm": 0.0,
            "true_grad_noise_norm": 0.0,
            "true_grad_snr": 0.0,
            "true_required_batch_multiplier_for_snr3": 0.0,
            "true_cancellation_ratio": 0.0,
            "true_mean_state_grad_norm": 0.0,
            "true_chunk_count": 0.0,
            "true_state_count": float(sample_count),
        }
    adv_np = np.asarray(oracle_advantage, dtype=np.float32)
    if tuple(adv_np.shape) != (sample_count, stochastic_count):
        raise ValueError(
            f"oracle_advantage must be [{sample_count},{stochastic_count}], got {tuple(adv_np.shape)}."
        )
    adv_t = torch.as_tensor(adv_np, dtype=torch.float32, device=learner.device)
    base_flat = torch.arange(sample_count * int(num_agents), dtype=torch.long, device=learner.device).reshape(
        sample_count, int(num_agents)
    )
    repeat_flat = base_flat[:, None, :].expand(sample_count, stochastic_count, int(num_agents)).reshape(-1)
    repeated_local = _index_dataclass(selected_local, repeat_flat)
    flat_actions = actions.reshape(sample_count * stochastic_count, int(num_agents), *actions.shape[3:]).contiguous()
    flat_adv = adv_t.reshape(-1)
    signal_vec = _stage_policy_gradient_vector(
        learner,
        stage_id=int(stage_id),
        local_batch=repeated_local,
        joint_actions=flat_actions,
        advantages=flat_adv,
        num_agents=int(num_agents),
        params=params,
    )
    signal_norm = float(torch.linalg.vector_norm(signal_vec).detach().cpu().item())

    total = sample_count * stochastic_count
    chunks = max(min(int(chunk_count), total), 1)
    chunk_vectors: list[torch.Tensor] = []
    flat_indices = torch.arange(total, dtype=torch.long, device=learner.device)
    for idx_t in torch.chunk(flat_indices, chunks):
        if int(idx_t.numel()) <= 0:
            continue
        sample_ids = torch.div(idx_t, stochastic_count, rounding_mode="floor")
        action_ids = idx_t - sample_ids * stochastic_count
        local_idx = (sample_ids[:, None] * int(num_agents) + torch.arange(int(num_agents), device=learner.device)[None, :]).reshape(-1)
        local_chunk = _index_dataclass(selected_local, local_idx)
        action_chunk = actions[sample_ids, action_ids].reshape(int(idx_t.numel()), int(num_agents), *actions.shape[3:]).contiguous()
        adv_chunk = adv_t[sample_ids, action_ids]
        chunk_vectors.append(
            _stage_policy_gradient_vector(
                learner,
                stage_id=int(stage_id),
                local_batch=local_chunk,
                joint_actions=action_chunk,
                advantages=adv_chunk,
                num_agents=int(num_agents),
                params=params,
            ).cpu()
        )
    if len(chunk_vectors) > 1:
        chunk_stack = torch.stack(chunk_vectors, dim=0)
        noise_norm = float(
            torch.sqrt(torch.var(chunk_stack, dim=0, unbiased=False).sum().clamp_min(0.0) / float(len(chunk_vectors)))
            .detach()
            .item()
        )
    else:
        noise_norm = 0.0

    state_vectors: list[torch.Tensor] = []
    agent_offsets = torch.arange(int(num_agents), dtype=torch.long, device=learner.device)
    for state_idx in range(sample_count):
        local_idx = (
            torch.full((stochastic_count, 1), int(state_idx), dtype=torch.long, device=learner.device)
            * int(num_agents)
            + agent_offsets.view(1, int(num_agents))
        ).reshape(-1)
        local_i = _index_dataclass(selected_local, local_idx)
        actions_i = actions[state_idx].reshape(stochastic_count, int(num_agents), *actions.shape[3:]).contiguous()
        adv_i = adv_t[state_idx]
        state_vectors.append(
            _stage_policy_gradient_vector(
                learner,
                stage_id=int(stage_id),
                local_batch=local_i,
                joint_actions=actions_i,
                advantages=adv_i,
                num_agents=int(num_agents),
                params=params,
            ).cpu()
        )
    if state_vectors:
        state_stack = torch.stack(state_vectors, dim=0)
        mean_state_grad_norm = float(torch.linalg.vector_norm(state_stack, dim=1).mean().detach().item())
        cancellation_ratio = signal_norm / max(mean_state_grad_norm, 1.0e-12)
    else:
        mean_state_grad_norm = 0.0
        cancellation_ratio = 0.0
    snr = signal_norm / max(noise_norm, 1.0e-12)
    required_multiplier = float((3.0 / max(snr, 1.0e-12)) ** 2) if snr > 0.0 else float("inf")
    return {
        "true_grad_signal_norm": signal_norm,
        "true_grad_noise_norm": noise_norm,
        "true_grad_snr": snr,
        "true_required_batch_multiplier_for_snr3": required_multiplier,
        "true_cancellation_ratio": float(cancellation_ratio),
        "true_mean_state_grad_norm": mean_state_grad_norm,
        "true_chunk_count": float(len(chunk_vectors)),
        "true_state_count": float(sample_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--rollout_env_steps", type=int, default=250)
    parser.add_argument("--sample_rows", type=int, default=12)
    parser.add_argument("--policy_action_samples", type=int, default=4)
    parser.add_argument("--min_horizon", type=int, default=10)
    parser.add_argument("--branch_horizon_cap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42000)
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--exec_sat_source", default="queue_aware")
    parser.add_argument("--exec_bw_source", default="queue_aware")
    parser.add_argument("--accel_log_std_init", type=float, default=None)
    parser.add_argument("--stage", choices=["accel", "sat", "bw"], default="accel")
    parser.add_argument("--autograd_chunk_count", type=int, default=8)
    parser.add_argument(
        "--reward_modes",
        nargs="+",
        default=[
            "weighted_workload_level",
            "positive_weighted_workload_level",
            "weighted_workload_delta",
            "relative_weighted_workload_delta",
            "controllable_flow",
            "access_raw",
            "accel_access_safe",
            "accel_delta_safe",
            "sat_relay_processed",
            "sat_backhaul_drop",
            "bw_access_drop",
            "bw_relative_access",
            "accel_growth_safe",
            "accel_pressure_safe",
            "accel_queue_relief",
            "bw_growth_drop",
            "bw_queue_relief",
            "bw_access_pressure",
        ],
    )
    args = parser.parse_args()

    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    _set_seed(int(args.seed))
    cfg = load_config(args.config)
    stage_name = str(args.stage)
    stage_id = {"accel": 0, "sat": 1, "bw": 2}[stage_name]
    cfg.exec_accel_source = "policy" if stage_id == 0 else "cluster_center_queue_aware"
    cfg.exec_sat_source = "policy" if stage_id == 1 else str(args.exec_sat_source)
    cfg.exec_bw_source = "policy" if stage_id == 2 else str(args.exec_bw_source)
    cfg.train_accel = bool(stage_id == 0)
    cfg.train_sat = bool(stage_id == 1)
    cfg.train_bw = bool(stage_id == 2)
    cfg.checkpoint_eval_enabled = False
    cfg.train_trace_enabled = False
    cfg.structured_env_tensor_backend = "cuda" if device.type == "cuda" else "cpu"
    cfg.structured_env_backend = "native"
    if args.accel_log_std_init is not None:
        cfg.accel_log_std_init = float(args.accel_log_std_init)

    bundle = build_structured_modules_from_config(cfg)
    actor = bundle.actor.to(device)
    critic = bundle.critic.to(device) if bundle.critic is not None else None
    actor_optimizer = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=float(cfg.actor_lr))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(cfg.critic_lr)) if critic is not None else None
    learner = StructuredMAPPO(
        actor=actor,
        critic=critic,
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_ratio=float(cfg.clip_ratio),
        value_coef=float(cfg.value_coef),
        entropy_coef=float(cfg.entropy_coef),
        max_grad_norm=float(cfg.max_grad_norm),
        ppo_epochs=1,
        num_mini_batch=1,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        device=device,
        cfg=cfg,
        train_accel=bool(stage_id == 0),
        train_sat=bool(stage_id == 1),
        train_bw=bool(stage_id == 2),
        exec_accel_source="policy",
        exec_sat_source="policy" if stage_id == 1 else str(args.exec_sat_source),
        exec_bw_source="policy" if stage_id == 2 else str(args.exec_bw_source),
    )
    group = make_structured_driver_group(cfg, num_envs=int(args.num_envs), backend="sync", mode="train")
    try:
        if hasattr(group, "reset_many"):
            group.reset_many([int(args.seed) + i for i in range(int(args.num_envs))])
        learner.bind_native_runtime_contract(group)
        learner.begin_native_rollout(group, rollout_env_steps=int(args.rollout_env_steps), num_envs=int(args.num_envs))
        buffer = StructuredRolloutBuffer()
        t0 = time.perf_counter()
        learner.collect_env_horizon_native_tensor_policy(
            group,
            buffer=buffer,
            horizon=int(args.rollout_env_steps),
            deterministic=False,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        rollout_sec = time.perf_counter() - t0
        rollout_views = buffer.build_rollout_views(device)
        batch_view = rollout_views.training_view
        stage_batch = batch_view.stage_batches[stage_id]
        num_samples = int(stage_batch.num_samples)
        num_agents = int(stage_batch.num_agents)
        transition_indices = np.asarray(stage_batch.transition_indices, dtype=np.int64)
        history_rows_all = ((transition_indices - stage_id) // 3).astype(np.int64)
        step_indices_all = history_rows_all // int(args.num_envs)
        horizons_all = np.maximum(int(cfg.T_steps) - step_indices_all, 1).astype(np.int64)
        eligible = np.flatnonzero(horizons_all >= max(int(args.min_horizon), 1))
        if eligible.size == 0:
            raise RuntimeError(f"no eligible {stage_name} rows after min_horizon filtering.")
        rng = np.random.default_rng(int(args.seed) + 17)
        sample_count = min(max(int(args.sample_rows), 1), int(eligible.size))
        selected_np = np.sort(rng.choice(eligible, size=sample_count, replace=False))
        selected_t = torch.as_tensor(selected_np, dtype=torch.long, device=device)

        flat_rows = (
            selected_t.view(-1, 1) * int(num_agents)
            + torch.arange(num_agents, dtype=torch.long, device=device).view(1, num_agents)
        ).reshape(-1)
        selected_local = _index_dataclass(stage_batch.local_batch, flat_rows)
        sampled_actions = stage_batch.actions.index_select(0, selected_t)
        labels = ["ref", "rollout_sample"]
        actions = []
        with torch.no_grad():
            if stage_id == 0:
                ref_out = actor.act_accel(selected_local, deterministic=True)
                ref_actions = ref_out.action.reshape(sample_count, num_agents, -1).to(device=device, dtype=torch.float32)
                mean_actions = ref_out.mean.reshape(sample_count, num_agents, -1).to(device=device, dtype=torch.float32)
                std_actions = ref_out.std.reshape(sample_count, num_agents, -1).to(device=device, dtype=torch.float32)
                actions.append(ref_actions)
                actions.append(sampled_actions.to(device=device, dtype=torch.float32))
                for idx in range(max(int(args.policy_action_samples), 0)):
                    alt_out = actor.act_accel(selected_local, deterministic=False)
                    actions.append(alt_out.action.reshape(sample_count, num_agents, -1).to(device=device, dtype=torch.float32))
                    labels.append(f"policy_sample_{idx}")
                action_tensor = torch.stack(actions, dim=1)
                stochastic_action_tensor = action_tensor[:, 1:, :, :]
                latent_action = _atanh_scaled_action(
                    stochastic_action_tensor,
                    scale=float(getattr(getattr(actor, "accel_policy", None), "action_scale", 1.0) or 1.0),
                )
                score_proxy_t = (latent_action - mean_actions[:, None, :, :]) / torch.clamp(std_actions[:, None, :, :] ** 2, min=1.0e-12)
            elif stage_id == 1:
                ref_out = actor.act_sat(selected_local, deterministic=True)
                logits = ref_out.logits.reshape(sample_count, num_agents, -1)
                probs = torch.softmax(logits, dim=-1)
                ref_actions = ref_out.subset_index.reshape(sample_count, num_agents).to(device=device, dtype=torch.long)
                actions.append(ref_actions)
                actions.append(sampled_actions.to(device=device, dtype=torch.long).reshape(sample_count, num_agents))
                for idx in range(max(int(args.policy_action_samples), 0)):
                    alt_out = actor.act_sat(selected_local, deterministic=False)
                    actions.append(alt_out.subset_index.reshape(sample_count, num_agents).to(device=device, dtype=torch.long))
                    labels.append(f"policy_sample_{idx}")
                action_tensor = torch.stack(actions, dim=1)
                stochastic_action_tensor = action_tensor[:, 1:, :]
                score_parts = []
                for k in range(int(stochastic_action_tensor.shape[1])):
                    one_hot = torch.nn.functional.one_hot(
                        stochastic_action_tensor[:, k, :].clamp(min=0, max=max(int(logits.shape[-1]) - 1, 0)),
                        num_classes=int(logits.shape[-1]),
                    ).to(dtype=probs.dtype, device=device)
                    score_parts.append(one_hot - probs)
                score_proxy_t = torch.stack(score_parts, dim=1)
            else:
                ref_out = actor.act_bw(selected_local, deterministic=True)
                det = ref_out.det_mean.reshape(sample_count, num_agents, -1).to(device=device, dtype=torch.float32)
                kappa = ref_out.kappa.reshape(sample_count, num_agents, 1).to(device=device, dtype=torch.float32)
                alpha = torch.clamp(det * kappa, min=1.0e-6)
                actions.append(ref_out.action.reshape(sample_count, num_agents, -1).to(device=device, dtype=torch.float32))
                actions.append(sampled_actions.to(device=device, dtype=torch.float32).reshape(sample_count, num_agents, -1))
                for idx in range(max(int(args.policy_action_samples), 0)):
                    alt_out = actor.act_bw(selected_local, deterministic=False)
                    actions.append(alt_out.action.reshape(sample_count, num_agents, -1).to(device=device, dtype=torch.float32))
                    labels.append(f"policy_sample_{idx}")
                action_tensor = torch.stack(actions, dim=1)
                stochastic_action_tensor = action_tensor[:, 1:, :, :]
                valid = (det > 0.0).to(dtype=torch.float32)
                # Bounded local simplex direction. This is not the exact Dirichlet score, but avoids
                # invalid-slot log explosions and is enough to compare reward credit across modes.
                score_proxy_t = (stochastic_action_tensor - det[:, None, :, :]) * valid[:, None, :, :]
        action_count = int(action_tensor.shape[1])
        score_proxy = score_proxy_t.detach().cpu().numpy().reshape(sample_count, max(action_count - 1, 0), -1)
        history_rows = history_rows_all[selected_np]
        horizons = horizons_all[selected_np]
        if int(args.branch_horizon_cap) > 0:
            horizons = np.minimum(horizons, int(args.branch_horizon_cap)).astype(np.int64, copy=False)
        branch_rows = np.repeat(history_rows, action_count)
        branch_horizons = np.repeat(horizons, action_count)
        branch_actions = action_tensor.reshape(sample_count * action_count, num_agents, -1)

        t1 = time.perf_counter()
        returns_by_mode = _branch_returns_from_history_for_modes(
            learner,
            stage_id=stage_id,
            history_rows=branch_rows.tolist(),
            first_actions=branch_actions,
            horizons=branch_horizons.tolist(),
            reward_modes=[str(x) for x in args.reward_modes],
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        branch_sec = time.perf_counter() - t1

        modes_payload: dict[str, Any] = {}
        ppo_score_payload: dict[str, Any] = {}
        true_grad_payload: dict[str, Any] = {}
        for mode, values in returns_by_mode.items():
            matrix = values.detach().cpu().numpy().reshape(sample_count, action_count)
            modes_payload[str(mode)] = _reward_diagnostics(matrix, labels)
            if action_count > 1:
                ppo_score_payload[str(mode)] = _ppo_score_proxy_diagnostics(
                    matrix[:, 1:],
                    score_proxy=score_proxy,
                )
                oracle_adv = matrix[:, 1:] - np.mean(matrix[:, 1:], axis=1, keepdims=True)
                true_grad_payload[str(mode)] = _true_autograd_gradient_diagnostics(
                    learner,
                    stage_id=int(stage_id),
                    selected_local=selected_local,
                    stochastic_joint_actions=action_tensor[:, 1:],
                    oracle_advantage=oracle_adv,
                    num_agents=int(num_agents),
                    chunk_count=max(int(args.autograd_chunk_count), 1),
                )
        ranking = sorted(
            (
                {
                    "reward_mode": mode,
                    "action_snr": float(payload["action_snr"]),
                    "action_scale": float(payload["action_scale"]),
                    "state_scale": float(payload["state_scale"]),
                    "best_minus_ref_mean": float(payload["best_minus_ref"]["mean"]),
                    "pairwise_gap": float(payload["pairwise_gap_mean_abs_centered"]),
                }
                for mode, payload in modes_payload.items()
            ),
            key=lambda x: float(x["action_snr"]),
            reverse=True,
        )
        ppo_ranking = sorted(
            (
                {
                    "reward_mode": mode,
                    "oracle_grad_snr": float(payload["oracle_grad_snr"]),
                    "signal_norm": float(payload["oracle_grad_signal_norm"]),
                    "noise_norm": float(payload["oracle_grad_noise_norm"]),
                    "cancellation_ratio": float(payload["cancellation_ratio"]),
                    "required_batch_multiplier_for_snr3": float(payload["required_batch_multiplier_for_snr3"]),
                }
                for mode, payload in ppo_score_payload.items()
            ),
            key=lambda x: float(x["oracle_grad_snr"]),
            reverse=True,
        )
        true_grad_ranking = sorted(
            (
                {
                    "reward_mode": mode,
                    "true_grad_snr": float(payload["true_grad_snr"]),
                    "signal_norm": float(payload["true_grad_signal_norm"]),
                    "noise_norm": float(payload["true_grad_noise_norm"]),
                    "cancellation_ratio": float(payload["true_cancellation_ratio"]),
                    "required_batch_multiplier_for_snr3": float(payload["true_required_batch_multiplier_for_snr3"]),
                }
                for mode, payload in true_grad_payload.items()
            ),
            key=lambda x: float(x["true_grad_snr"]),
            reverse=True,
        )
        payload = {
            "config": str(args.config),
            "device": str(device),
            "num_envs": int(args.num_envs),
            "rollout_env_steps": int(args.rollout_env_steps),
            "sample_rows": int(sample_count),
            "policy_action_samples": int(args.policy_action_samples),
            "action_labels": labels,
            "exec_sources": {
                "accel": str(cfg.exec_accel_source),
                "sat": str(cfg.exec_sat_source),
                "bw": str(cfg.exec_bw_source),
            },
            "stage": stage_name,
            "stage_id": int(stage_id),
            "reward_modes": [str(x) for x in args.reward_modes],
            "row_summary": {
                "history_rows": [int(x) for x in history_rows.tolist()],
                "steps": [int(x) for x in step_indices_all[selected_np].tolist()],
                "horizons": [int(x) for x in horizons.tolist()],
                "min_horizon": int(args.min_horizon),
                "branch_horizon_cap": int(args.branch_horizon_cap),
            },
            "timing": {
                "rollout_sec": float(rollout_sec),
                "branch_sec": float(branch_sec),
            },
            "ranking_by_action_snr": ranking,
            "ranking_by_oracle_policy_gradient_snr": ppo_ranking,
            "ranking_by_true_autograd_gradient_snr": true_grad_ranking,
            "modes": modes_payload,
            "ppo_score_proxy": {
                "description": "Uses same-state oracle-centered returns over stochastic policy samples and squashed-Gaussian latent mean score (atanh(action/scale)-mean)/std^2. Ref deterministic action is excluded.",
                "score_dim": int(score_proxy.shape[-1]) if score_proxy.ndim == 3 else 0,
                "modes": ppo_score_payload,
            },
            "true_autograd_gradient": {
                "description": "Uses actual actor.evaluate_* logprob and torch.autograd over the active stage actor parameters. Noise is estimated from gradient chunks; cancellation uses per-state gradients.",
                "chunk_count": int(args.autograd_chunk_count),
                "modes": true_grad_payload,
            },
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "ranking_by_action_snr": ranking,
                    "ranking_by_oracle_policy_gradient_snr": ppo_ranking,
                    "ranking_by_true_autograd_gradient_snr": true_grad_ranking,
                    "out": str(out),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        close_structured_env_group(group)


if __name__ == "__main__":
    main()
