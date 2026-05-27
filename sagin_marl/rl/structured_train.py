from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, replace
import time
from typing import Any, Callable, List, Sequence

import numpy as np
import torch

from sagin_marl.env.structured_driver import StructuredBatchStepResult
from sagin_marl.env.structured_sync_group import (
    GpuStructuredDriverGroup,
    GpuStructuredEnvGroup,
    PythonStructuredDriverGroup,
    PythonStructuredEnvGroup,
)
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer


def _step_team_reward(step_result) -> float:
    if hasattr(step_result, "team_reward"):
        return float(step_result.team_reward)
    if not step_result.rewards:
        return 0.0
    first_agent = next(iter(step_result.rewards))
    return float(step_result.rewards[first_agent])


def _is_done(step_result) -> bool:
    if hasattr(step_result, "terminated") or hasattr(step_result, "truncated"):
        return bool(getattr(step_result, "terminated", False) or getattr(step_result, "truncated", False))
    if not step_result.terminations:
        return False
    first_agent = next(iter(step_result.terminations))
    return bool(step_result.terminations[first_agent] or step_result.truncations[first_agent])


@dataclass
class StructuredTrainStepMetrics:
    env_reward_mean: float
    rollout_reward_per_step: float
    episode_reward: float
    episode_reward_std: float
    episode_reward_p25: float
    episode_reward_p75: float
    episode_length_mean: float
    completed_episode_count: int
    episodes_finished: int
    policy_loss: float
    value_loss: float
    env_steps: int = 0
    transition_samples: int = 0
    rollout_reset_time_sec: float = 0.0
    native_rollout_prepare_time_sec: float = 0.0
    rollout_collect_time_sec: float = 0.0
    rollout_total_time_sec: float = 0.0
    rollout_view_build_time_sec: float = 0.0
    update_prepare_time_sec: float = 0.0
    update_optimize_time_sec: float = 0.0
    update_total_time_sec: float = 0.0
    update_profile_old_logprob_sec: float = 0.0
    update_profile_value_override_sec: float = 0.0
    update_profile_returns_sec: float = 0.0
    update_profile_stage_cache_sec: float = 0.0
    update_profile_critic_train_sec: float = 0.0
    update_profile_actor_train_sec: float = 0.0
    update_profile_explained_variance_sec: float = 0.0
    iteration_time_sec: float = 0.0
    env_steps_per_sec: float = 0.0
    samples_per_sec: float = 0.0
    bw_access_reward_mean: float = 0.0
    bw_train_reward_mean: float = 0.0
    reward_part_x_acc_mean: float = 0.0
    reward_part_x_rel_mean: float = 0.0
    reward_part_processed_ratio_eval_mean: float = 0.0
    reward_part_drop_ratio_eval_mean: float = 0.0
    reward_part_d_pre_mean: float = 0.0
    reward_part_sat_overlap_eval_mean: float = 0.0
    reward_part_g_pre_mean: float = 0.0
    reward_part_service_gap_risk_mean: float = 0.0
    value_loss_accel: float = 0.0
    value_loss_sat: float = 0.0
    value_loss_bw: float = 0.0
    critic_popart_mean_bw: float = 0.0
    critic_popart_std_bw: float = 1.0
    explained_variance_accel: float = 0.0
    explained_variance_sat: float = 0.0
    explained_variance_bw: float = 0.0
    entropy: float = 0.0
    entropy_accel: float = 0.0
    entropy_sat: float = 0.0
    entropy_bw: float = 0.0
    approx_kl: float = 0.0
    approx_kl_accel: float = 0.0
    approx_kl_sat: float = 0.0
    approx_kl_bw: float = 0.0
    clip_frac: float = 0.0
    clip_frac_accel: float = 0.0
    clip_frac_sat: float = 0.0
    clip_frac_bw: float = 0.0
    vs_ref_rows: float = 0.0
    vs_ref_adv_mean: float = 0.0
    vs_ref_adv_std: float = 0.0
    vs_ref_positive_frac: float = 0.0
    vs_ref_rows_accel: float = 0.0
    vs_ref_adv_mean_accel: float = 0.0
    vs_ref_adv_std_accel: float = 0.0
    vs_ref_positive_frac_accel: float = 0.0
    vs_ref_rows_sat: float = 0.0
    vs_ref_adv_mean_sat: float = 0.0
    vs_ref_adv_std_sat: float = 0.0
    vs_ref_positive_frac_sat: float = 0.0
    vs_ref_rows_bw: float = 0.0
    vs_ref_adv_mean_bw: float = 0.0
    vs_ref_adv_std_bw: float = 0.0
    vs_ref_positive_frac_bw: float = 0.0
    grad_norm_accel: float = 0.0
    grad_norm_sat: float = 0.0
    grad_norm_bw: float = 0.0
    vs_ref_sampling_random_count_accel: float = 0.0
    vs_ref_sampling_time_count_accel: float = 0.0
    vs_ref_sampling_leverage_count_accel: float = 0.0
    vs_ref_sampling_uncertainty_count_accel: float = 0.0
    vs_ref_sampling_horizon_mean_accel: float = 0.0
    vs_ref_sampling_horizon_min_accel: float = 0.0
    vs_ref_sampling_horizon_max_accel: float = 0.0
    vs_ref_sampling_priority_mean_accel: float = 0.0
    vs_ref_sampling_priority_max_accel: float = 0.0
    vs_ref_sampling_random_count_sat: float = 0.0
    vs_ref_sampling_time_count_sat: float = 0.0
    vs_ref_sampling_leverage_count_sat: float = 0.0
    vs_ref_sampling_uncertainty_count_sat: float = 0.0
    vs_ref_sampling_horizon_mean_sat: float = 0.0
    vs_ref_sampling_horizon_min_sat: float = 0.0
    vs_ref_sampling_horizon_max_sat: float = 0.0
    vs_ref_sampling_priority_mean_sat: float = 0.0
    vs_ref_sampling_priority_max_sat: float = 0.0
    vs_ref_sampling_random_count_bw: float = 0.0
    vs_ref_sampling_time_count_bw: float = 0.0
    vs_ref_sampling_leverage_count_bw: float = 0.0
    vs_ref_sampling_uncertainty_count_bw: float = 0.0
    vs_ref_sampling_horizon_mean_bw: float = 0.0
    vs_ref_sampling_horizon_min_bw: float = 0.0
    vs_ref_sampling_horizon_max_bw: float = 0.0
    vs_ref_sampling_priority_mean_bw: float = 0.0
    vs_ref_sampling_priority_max_bw: float = 0.0
    danger_imitation_loss: float = 0.0
    danger_imitation_active_rate: float = 0.0
    bw_counterfactual_credit_active_rate: float = 0.0
    bw_counterfactual_credit_agent_active_rate: float = 0.0
    bw_counterfactual_credit_mean: float = 0.0
    bw_counterfactual_credit_abs_mean: float = 0.0
    bw_counterfactual_credit_positive_frac: float = 0.0
    bw_flow_proxy_aux_loss: float = 0.0
    bw_flow_proxy_regression_loss: float = 0.0
    bw_flow_proxy_pairwise_acc: float = 0.0
    bw_flow_proxy_pair_count: float = 0.0
    bw_grad_norm_policy: float = 0.0
    bw_grad_norm_aux_scaled: float = 0.0
    bw_grad_ratio_aux_to_policy: float = 0.0
    bw_score_head_grad_norm_policy: float = 0.0
    bw_score_head_grad_norm_aux_scaled: float = 0.0
    bw_score_head_grad_ratio_aux_to_policy: float = 0.0
    bw_abs_log_ratio_corr_valid_count: float = 0.0
    bw_abs_log_ratio_corr_latent_count: float = 0.0
    bw_kappa_mean: float = 0.0
    bw_kappa_p10: float = 0.0
    bw_kappa_p90: float = 0.0
    bw_kappa_hi_frac: float = 0.0
    bw_delta_student_teacher_corr: float = 0.0
    bw_delta_student_true_corr: float = 0.0
    bw_delta_actor_mix_alpha: float = 0.0
    bw_delta_teacher_used: float = 0.0
    bw_delta_teacher_observed: float = 0.0
    bw_delta_teacher_probe_only: float = 0.0
    bw_branch_gate_snr: float = 0.0
    bw_branch_gate_triggered: float = 0.0
    bw_actor_update_skipped: float = 0.0
    clean_mean_target_gap: float = 0.0
    clean_mean_update_shift: float = 0.0
    clean_update_to_target_ratio: float = 0.0
    clean_target_beats_ref_frac: float = 0.0
    clean_target_beats_sampled_frac: float = 0.0
    clean_measured_kl: float = 0.0
    clean_kl_coef: float = 0.0
    clean_row_sample_eligible: float = 0.0
    clean_row_sample_count: float = 0.0
    clean_row_sample_frac: float = 0.0
    clean_ref_branch_count: float = 0.0
    clean_perturb_branch_count: float = 0.0
    clean_gate_branch_count: float = 0.0
    clean_row_filter_count: float = 0.0
    clean_row_filter_frac: float = 0.0
    clean_candidate_positive_branch_count: float = 0.0
    clean_candidate_selected_rows: float = 0.0


@dataclass
class StructuredEpisodeStatsState:
    recent_episode_returns: deque[float]
    recent_episode_lengths: deque[float]
    current_episode_returns: list[float]
    current_episode_lengths: list[int]


def _make_episode_stats_state(num_envs: int, episode_stat_window: int) -> StructuredEpisodeStatsState:
    return StructuredEpisodeStatsState(
        recent_episode_returns=deque(maxlen=max(int(episode_stat_window), 1)),
        recent_episode_lengths=deque(maxlen=max(int(episode_stat_window), 1)),
        current_episode_returns=[0.0 for _ in range(int(num_envs))],
        current_episode_lengths=[0 for _ in range(int(num_envs))],
    )


def _record_episode_transition(
    episode_stats_state: StructuredEpisodeStatsState,
    *,
    env_index: int,
    reward: float,
    done: bool,
) -> bool:
    episode_stats_state.current_episode_returns[int(env_index)] += float(reward)
    episode_stats_state.current_episode_lengths[int(env_index)] += 1
    if not bool(done):
        return False
    episode_stats_state.recent_episode_returns.append(
        float(episode_stats_state.current_episode_returns[int(env_index)])
    )
    episode_stats_state.recent_episode_lengths.append(
        float(episode_stats_state.current_episode_lengths[int(env_index)])
    )
    episode_stats_state.current_episode_returns[int(env_index)] = 0.0
    episode_stats_state.current_episode_lengths[int(env_index)] = 0
    return True


def _rolling_window_summary(values) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "p25": 0.0,
            "p75": 0.0,
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p25": float(np.percentile(arr, 25.0)),
        "p75": float(np.percentile(arr, 75.0)),
    }


def _maybe_empty_native_cuda_cache_after_update(learner: Any) -> None:
    cfg = getattr(learner, "cfg", None)
    if cfg is None:
        return
    if not bool(getattr(cfg, "structured_native_cuda_empty_cache_after_update", True)):
        return
    try:
        device = torch.device(getattr(learner, "device", "cpu"))
    except (TypeError, RuntimeError):
        return
    env_backend = str(getattr(cfg, "structured_env_backend", "auto") or "auto").strip().lower()
    env_tensor_backend = str(getattr(cfg, "structured_env_tensor_backend", "auto") or "auto").strip().lower()
    if device.type != "cuda" or env_backend != "native" or env_tensor_backend != "cuda":
        return
    if torch.cuda.is_available():
        # PPO update leaves large cached allocator blocks in this path; releasing
        # them avoids a large slowdown in the next native fused rollout kernel.
        torch.cuda.empty_cache()


def _normalize_env_group(env_or_envs) -> List:
    if isinstance(env_or_envs, Sequence) and not isinstance(env_or_envs, (str, bytes, bytearray)):
        return list(env_or_envs)
    return [env_or_envs]


def _reset_env_at(env, seed: int | None) -> None:
    if seed is None:
        env.reset()
    else:
        env.reset(seed=int(seed))


def _reset_with_optional_controller(
    *,
    structured_group,
    envs,
    drivers,
    reset_controller,
    indices: Sequence[int],
    seeds: Sequence[int | None],
) -> None:
    if reset_controller is None:
        if structured_group is not None:
            structured_group.reset_many(seeds, indices=indices)
            return
        for env_index, seed in zip(indices, seeds):
            _reset_env_at(envs[int(env_index)], seed)
        return
    reset_many = getattr(reset_controller, "reset_many", None)
    if not callable(reset_many):
        raise TypeError("reset_controller must define a callable reset_many method.")
    reset_many(
        structured_group=structured_group,
        envs=envs,
        drivers=drivers,
        indices=[int(index) for index in indices],
        seeds=[None if seed is None else int(seed) for seed in seeds],
    )


def _looks_like_structured_driver_group(obj) -> bool:
    native_required = (
        "reset_many",
        "reset_at",
        "native_rollout_program",
        "begin_native_main_kernel_rollout",
        "native_rollout_runtime",
    )
    return all(hasattr(obj, name) for name in native_required)


def _looks_like_structured_driver(obj) -> bool:
    """Backward-compatible singular name used by legacy diagnostics."""
    return _looks_like_structured_driver_group(obj)


def _looks_like_python_structured_driver_group(obj) -> bool:
    return bool(getattr(obj, "is_python_structured_driver_group", False))


def _looks_like_any_structured_driver_group(obj) -> bool:
    return _looks_like_structured_driver_group(obj) or _looks_like_python_structured_driver_group(obj)


def as_structured_drivers(env_or_envs):
    if _looks_like_any_structured_driver_group(env_or_envs):
        return env_or_envs
    if isinstance(env_or_envs, Sequence) and not isinstance(env_or_envs, (str, bytes, bytearray)):
        drivers = list(env_or_envs)
        required = ("begin_step", "run_accel_stage", "run_sat_stage", "execute_stage_bw_and_prepare_next_accel")
        if drivers and all(all(hasattr(driver, name) for name in required) for driver in drivers):
            return drivers
    raise RuntimeError(
        "legacy per-env structured driver adaptation has been removed; "
        "pass a native structured batch driver group instead."
    )


def make_structured_env_group(
    cfg,
    num_envs: int,
    backend: str = "sync",
    *,
    mode: str = "script",
):
    num_envs_i = max(int(num_envs), 1)
    mode_l = str(mode or "script").strip().lower()
    if mode_l not in {"train", "eval", "script"}:
        raise ValueError(f"Unsupported structured env mode: {mode}")
    env_tensor_backend = str(getattr(cfg, "structured_env_tensor_backend", "cuda") or "cuda").strip().lower()
    env_backend = str(getattr(cfg, "structured_env_backend", "auto") or "auto").strip().lower()
    prefer_native = env_backend in {"native", "gpu", "auto"}
    group_cfg = copy.copy(cfg)
    setattr(group_cfg, "_structured_kernel_runtime_cache", {})
    if env_backend in {"python", "cpu", "mac", "legacy"} or env_tensor_backend == "cpu":
        return PythonStructuredEnvGroup(group_cfg, num_envs=num_envs_i, tensor_device="cpu")
    if env_tensor_backend in {"cuda", "auto"} and prefer_native:
        return GpuStructuredEnvGroup(group_cfg, num_envs=num_envs_i)
    backend_l = str(backend).lower()
    if mode_l in {"train", "eval"} and backend_l == "subproc":
        backend_l = "sync"
    if backend_l == "sync" and prefer_native:
        return GpuStructuredEnvGroup(group_cfg, num_envs=num_envs_i)
    if backend_l == "sync":
        return GpuStructuredEnvGroup(group_cfg, num_envs=num_envs_i)
    if backend_l == "subproc":
        raise RuntimeError("structured subproc backend has been removed; use the native single-process GPU batch group.")
    raise ValueError(f"Unknown structured vec backend: {backend}")


def make_structured_driver_group(
    cfg,
    num_envs: int,
    backend: str = "sync",
    *,
    mode: str = "script",
):
    num_envs_i = max(int(num_envs), 1)
    env_tensor_backend = str(getattr(cfg, "structured_env_tensor_backend", "cuda") or "cuda").strip().lower()
    backend_l = str(backend).lower()
    env_backend = str(getattr(cfg, "structured_env_backend", "auto") or "auto").strip().lower()
    prefer_native = env_backend in {"native", "gpu", "auto"}
    group_cfg = copy.copy(cfg)
    setattr(group_cfg, "_structured_kernel_runtime_cache", {})
    if env_backend in {"python", "cpu", "mac", "legacy"} or env_tensor_backend == "cpu":
        return PythonStructuredDriverGroup(group_cfg, num_envs=num_envs_i, tensor_device="cpu")
    if str(mode or "script").strip().lower() in {"train", "eval"} and backend_l == "subproc":
        backend_l = "sync"
    if env_tensor_backend in {"cuda", "auto"} and prefer_native:
        return GpuStructuredDriverGroup(group_cfg, num_envs=num_envs_i)
    if backend_l == "sync" and prefer_native:
        return GpuStructuredDriverGroup(group_cfg, num_envs=num_envs_i)
    if backend_l == "sync":
        return GpuStructuredDriverGroup(group_cfg, num_envs=num_envs_i)
    return as_structured_drivers(make_structured_env_group(cfg, num_envs=num_envs, backend=backend, mode=mode))


def as_structured_driver(env_or_driver):
    del env_or_driver
    raise RuntimeError("legacy single structured driver API has been removed; use a native batch driver group.")


def make_structured_env(cfg, backend: str = "sync", *, mode: str = "script"):
    del cfg, backend, mode
    raise RuntimeError("legacy single structured env API has been removed; use make_structured_env_group.")


def make_structured_driver(cfg, backend: str = "sync", *, mode: str = "script"):
    del cfg, backend, mode
    raise RuntimeError("legacy single structured driver API has been removed; use make_structured_driver_group.")


def _close_one_structured_target(target) -> None:
    if target is None:
        return
    close_fn = getattr(target, "close", None)
    if callable(close_fn):
        close_fn()


def close_structured_env_group(env_or_envs) -> None:
    if env_or_envs is None:
        return
    if _looks_like_any_structured_driver_group(env_or_envs):
        _close_one_structured_target(env_or_envs)
        return
    close_fn = getattr(env_or_envs, "close", None)
    if callable(close_fn):
        close_fn()
        return
    for env in _normalize_env_group(env_or_envs):
        _close_one_structured_target(env)


def run_structured_training(
    env,
    learner,
    *,
    num_updates: int,
    rollout_env_steps: int,
    reset_seed: int | None = None,
    reset_on_start: bool = True,
    reset_controller=None,
    episode_stats_state: StructuredEpisodeStatsState | None = None,
    episode_stat_window: int = 100,
    trace_fn: Callable[[str], None] | None = None,
    trace_interval: int = 0,
    before_update_callback: Callable[[StructuredRolloutBuffer, Any], None] | None = None,
) -> List[StructuredTrainStepMetrics]:
    if num_updates <= 0:
        return []
    if rollout_env_steps <= 0:
        raise ValueError("rollout_env_steps must be positive")
    structured_group = env if _looks_like_any_structured_driver_group(env) else None
    if structured_group is not None:
        num_envs = len(structured_group)
        initial_seeds = [None if reset_seed is None else int(reset_seed) + env_index for env_index in range(num_envs)]
        envs = None
        drivers = structured_group
    else:
        drivers_or_group = as_structured_drivers(env)
        if _looks_like_any_structured_driver_group(drivers_or_group):
            structured_group = drivers_or_group
            num_envs = len(structured_group)
            initial_seeds = [None if reset_seed is None else int(reset_seed) + env_index for env_index in range(num_envs)]
            envs = None
            drivers = structured_group
        else:
            envs = _normalize_env_group(env)
            num_envs = len(envs)
            drivers = list(drivers_or_group)
            initial_seeds = [None if reset_seed is None else int(reset_seed) + env_index for env_index in range(num_envs)]
    bind_native_contract = getattr(learner, "bind_native_runtime_contract", None)
    if callable(bind_native_contract) and _looks_like_structured_driver_group(drivers):
        bind_native_contract(drivers)
    if bool(reset_on_start):
        reset_start = time.perf_counter()
        _reset_with_optional_controller(
            structured_group=structured_group,
            envs=envs,
            drivers=drivers,
            reset_controller=reset_controller,
            indices=list(range(num_envs)),
            seeds=initial_seeds,
        )
        initial_reset_time_sec = max(time.perf_counter() - reset_start, 0.0)
    else:
        initial_reset_time_sec = 0.0
    if episode_stats_state is None or len(episode_stats_state.current_episode_returns) != int(num_envs):
        episode_stats_state = _make_episode_stats_state(num_envs=int(num_envs), episode_stat_window=int(episode_stat_window))
    history: List[StructuredTrainStepMetrics] = []
    reset_counters = [0 for _ in range(num_envs)]
    rollout_deterministic = bool(getattr(learner, "bw_clean_per_user_enabled", False))

    if trace_fn is not None:
        trace_fn(
            f"run_structured_training:start num_updates={int(num_updates)} "
            f"rollout_env_steps={int(rollout_env_steps)} num_envs={int(num_envs)}"
        )

    for update_local_idx in range(int(num_updates)):
        iteration_start = time.perf_counter()
        rollout_reset_time_sec = float(initial_reset_time_sec) if update_local_idx == 0 else 0.0
        native_rollout_prepare_time_sec = 0.0
        rollout_collect_time_sec = 0.0
        rollout_view_build_time_sec = 0.0
        update_prepare_time_sec = 0.0
        update_optimize_time_sec = 0.0
        if trace_fn is not None:
            trace_fn(f"run_structured_training:update_start local_update={update_local_idx + 1}")
        buffer = StructuredRolloutBuffer()
        begin_native_rollout = getattr(learner, "begin_native_rollout", None)
        if callable(begin_native_rollout) and _looks_like_structured_driver_group(drivers):
            native_prepare_start = time.perf_counter()
            begin_native_rollout(
                drivers,
                rollout_env_steps=int(rollout_env_steps),
                num_envs=int(num_envs),
            )
            native_rollout_prepare_time_sec = max(time.perf_counter() - native_prepare_start, 0.0)
        reward_sum = 0.0
        reward_sum_tensor: torch.Tensor | None = None
        pending_episode_reward_delta: torch.Tensor | None = None
        native_episode_reward_tensors: list[torch.Tensor] = []
        native_episode_done_tensors: list[torch.Tensor] = []
        episodes_finished = 0
        completed_episode_count = 0
        native_horizon_results = None
        collect_native_horizon = getattr(learner, "collect_env_horizon_native_tensor_policy", None)
        can_collect_native = getattr(learner, "_can_use_native_tensor_policy_rollout", None)
        if (
            callable(collect_native_horizon)
            and (not callable(can_collect_native) or bool(can_collect_native(drivers)))
        ):
            collect_start = time.perf_counter()
            native_horizon_results = collect_native_horizon(
                drivers,
                buffer,
                horizon=int(rollout_env_steps),
                deterministic=rollout_deterministic,
            )
            rollout_collect_time_sec += max(time.perf_counter() - collect_start, 0.0)
            if len(native_horizon_results) != int(rollout_env_steps):
                raise RuntimeError("native GPU horizon program returned an unexpected number of rollout steps.")
        for rollout_step_idx in range(int(rollout_env_steps)):
            if native_horizon_results is None:
                collect_start = time.perf_counter()
                results = learner.collect_env_steps(drivers, buffer, deterministic=rollout_deterministic)
                rollout_collect_time_sec += max(time.perf_counter() - collect_start, 0.0)
            else:
                results = native_horizon_results[int(rollout_step_idx)]
            if isinstance(results, StructuredBatchStepResult):
                step_reward_tensor = results.team_rewards.detach().reshape(int(num_envs))
                reward_sum_tensor = (
                    step_reward_tensor.sum()
                    if reward_sum_tensor is None
                    else reward_sum_tensor + step_reward_tensor.sum()
                )
                done_tensor = (results.terminated.reshape(int(num_envs)) | results.truncated.reshape(int(num_envs))).detach()
                if native_horizon_results is not None:
                    native_episode_reward_tensors.append(step_reward_tensor.detach())
                    native_episode_done_tensors.append(done_tensor.detach())
                elif bool(done_tensor.any().cpu().item()):
                    pending_episode_reward_delta = (
                        step_reward_tensor.clone()
                        if pending_episode_reward_delta is None
                        else pending_episode_reward_delta + step_reward_tensor
                    )
                    pending_rewards = pending_episode_reward_delta.detach().cpu().numpy().astype(np.float64, copy=False)
                    done_flags = done_tensor.cpu().numpy().astype(bool, copy=False)
                    pending_episode_reward_delta = torch.zeros_like(pending_episode_reward_delta)
                    for env_index in range(int(num_envs)):
                        episode_stats_state.current_episode_lengths[env_index] += 1
                        episode_stats_state.current_episode_returns[env_index] += float(pending_rewards[env_index])
                        if not bool(done_flags[env_index]):
                            continue
                        episodes_finished += 1
                        completed_episode_count += 1
                        episode_stats_state.recent_episode_returns.append(
                            float(episode_stats_state.current_episode_returns[env_index])
                        )
                        episode_stats_state.recent_episode_lengths.append(
                            float(episode_stats_state.current_episode_lengths[env_index])
                        )
                        episode_stats_state.current_episode_returns[env_index] = 0.0
                        episode_stats_state.current_episode_lengths[env_index] = 0
                        reset_counters[env_index] += 1
                        seed = None
                        if reset_seed is not None:
                            seed = int(reset_seed) + reset_counters[env_index] * num_envs + env_index
                        reset_start = time.perf_counter()
                        _reset_with_optional_controller(
                            structured_group=structured_group,
                            envs=envs,
                            drivers=drivers,
                            reset_controller=reset_controller,
                            indices=[env_index],
                            seeds=[seed],
                        )
                        rollout_reset_time_sec += max(time.perf_counter() - reset_start, 0.0)
                else:
                    pending_episode_reward_delta = (
                        step_reward_tensor.clone()
                        if pending_episode_reward_delta is None
                        else pending_episode_reward_delta + step_reward_tensor
                    )
                    for env_index in range(int(num_envs)):
                        episode_stats_state.current_episode_lengths[env_index] += 1
            else:
                step_rewards = [_step_team_reward(result) for result in results]
                reward_sum += sum(step_rewards)
                for env_index, (result, step_reward) in enumerate(zip(results, step_rewards)):
                    episode_stats_state.current_episode_returns[env_index] += float(step_reward)
                    episode_stats_state.current_episode_lengths[env_index] += 1
                    if _is_done(result):
                        episodes_finished += 1
                        completed_episode_count += 1
                        episode_stats_state.recent_episode_returns.append(
                            float(episode_stats_state.current_episode_returns[env_index])
                        )
                        episode_stats_state.recent_episode_lengths.append(
                            float(episode_stats_state.current_episode_lengths[env_index])
                        )
                        episode_stats_state.current_episode_returns[env_index] = 0.0
                        episode_stats_state.current_episode_lengths[env_index] = 0
                        reset_counters[env_index] += 1
                        seed = None
                        if reset_seed is not None:
                            seed = int(reset_seed) + reset_counters[env_index] * num_envs + env_index
                        reset_start = time.perf_counter()
                        _reset_with_optional_controller(
                            structured_group=structured_group,
                            envs=envs,
                            drivers=drivers,
                            reset_controller=reset_controller,
                            indices=[env_index],
                            seeds=[seed],
                        )
                        rollout_reset_time_sec += max(time.perf_counter() - reset_start, 0.0)
            if trace_fn is not None and trace_interval > 0:
                step_ordinal = rollout_step_idx + 1
                if step_ordinal == 1 or step_ordinal % int(trace_interval) == 0 or step_ordinal == int(rollout_env_steps):
                    trace_fn(
                        f"run_structured_training:rollout_progress "
                        f"local_update={update_local_idx + 1} step={step_ordinal}/{int(rollout_env_steps)} "
                        f"buffer_len={len(buffer)} episodes_finished={int(episodes_finished)}"
                    )
        if native_episode_reward_tensors:
            reward_matrix = torch.stack(native_episode_reward_tensors, dim=0).detach().cpu().numpy().astype(np.float64, copy=False)
            done_matrix = torch.stack(native_episode_done_tensors, dim=0).detach().cpu().numpy().astype(bool, copy=False)
            for step_index in range(int(reward_matrix.shape[0])):
                for env_index in range(int(num_envs)):
                    if _record_episode_transition(
                        episode_stats_state,
                        env_index=env_index,
                        reward=float(reward_matrix[step_index, env_index]),
                        done=bool(done_matrix[step_index, env_index]),
                    ):
                        episodes_finished += 1
                        completed_episode_count += 1
                        reset_counters[env_index] += 1
        elif pending_episode_reward_delta is not None:
            pending_rewards = pending_episode_reward_delta.detach().cpu().numpy().astype(np.float64, copy=False)
            for env_index in range(int(num_envs)):
                episode_stats_state.current_episode_returns[env_index] += float(pending_rewards[env_index])
        if reward_sum_tensor is not None:
            reward_sum += float(reward_sum_tensor.detach().cpu().item())
        rollout_total_time_sec = float(
            rollout_reset_time_sec
            + native_rollout_prepare_time_sec
            + rollout_collect_time_sec
        )
        view_start = time.perf_counter()
        rollout_views = buffer.build_rollout_views(learner.device) if len(buffer) > 0 else None
        rollout_view_build_time_sec = max(time.perf_counter() - view_start, 0.0)
        bootstrap_world_state = None
        if before_update_callback is not None:
            bootstrap_world_state = (
                buffer.build_bootstrap_world_state_dict(
                    None if rollout_views is None else rollout_views.bootstrap_view
                )
                if len(buffer) > 0
                else None
            )
            before_update_callback(buffer, bootstrap_world_state)
        update_prepare_time_sec = float(rollout_view_build_time_sec)
        if trace_fn is not None:
            trace_fn(
                f"run_structured_training:before_update local_update={update_local_idx + 1} "
                f"buffer_len={len(buffer)}"
            )
        update_start = time.perf_counter()
        update_metrics = learner.update(
            buffer,
            bootstrap_world_state,
            rollout_views=rollout_views,
        )
        _maybe_empty_native_cuda_cache_after_update(learner)
        update_optimize_time_sec = max(time.perf_counter() - update_start, 0.0)
        update_total_time_sec = float(update_prepare_time_sec + update_optimize_time_sec)
        if trace_fn is not None:
            trace_fn(
                f"run_structured_training:after_update local_update={update_local_idx + 1} "
                f"policy_loss={float(update_metrics['policy_loss']):.6f} "
                f"value_loss={float(update_metrics['value_loss']):.6f}"
            )
        episode_reward_stats = _rolling_window_summary(episode_stats_state.recent_episode_returns)
        episode_length_mean = (
            float(np.mean(episode_stats_state.recent_episode_lengths))
            if episode_stats_state.recent_episode_lengths
            else 0.0
        )
        bw_stage_batch = None if rollout_views is None else rollout_views.return_view.stage_batches.get(2)
        if bw_stage_batch is not None and int(bw_stage_batch.num_samples) > 0:
            bw_access_reward_mean = float(np.mean(bw_stage_batch.bw_access_rewards, dtype=np.float64))
            bw_weighted_reward_mean = float(
                np.mean(bw_stage_batch.bw_weighted_workload_delta_rewards, dtype=np.float64)
            )
            bw_weighted_level_reward_mean = float(
                np.mean(bw_stage_batch.bw_weighted_workload_level_rewards, dtype=np.float64)
            )
            reward_part_arrays = getattr(bw_stage_batch, "reward_part_arrays", {}) or {}

            def _reward_part_mean(name: str) -> float:
                arr = reward_part_arrays.get(name)
                if arr is None:
                    return 0.0
                arr_np = np.asarray(arr, dtype=np.float64).reshape(-1)
                arr_np = arr_np[np.isfinite(arr_np)]
                return float(np.mean(arr_np)) if arr_np.size else 0.0

            reward_part_x_acc_mean = _reward_part_mean("x_acc")
            reward_part_x_rel_mean = _reward_part_mean("x_rel")
            reward_part_processed_ratio_eval_mean = _reward_part_mean("processed_ratio_eval")
            reward_part_drop_ratio_eval_mean = _reward_part_mean("drop_ratio_eval")
            reward_part_d_pre_mean = _reward_part_mean("d_pre")
            reward_part_sat_overlap_eval_mean = _reward_part_mean("sat_overlap_eval")
            reward_part_g_pre_mean = _reward_part_mean("g_pre")
            reward_part_service_gap_risk_mean = _reward_part_mean("service_gap_risk_mean")
        else:
            bw_access_reward_mean = 0.0
            bw_weighted_reward_mean = 0.0
            bw_weighted_level_reward_mean = 0.0
            reward_part_x_acc_mean = 0.0
            reward_part_x_rel_mean = 0.0
            reward_part_processed_ratio_eval_mean = 0.0
            reward_part_drop_ratio_eval_mean = 0.0
            reward_part_d_pre_mean = 0.0
            reward_part_sat_overlap_eval_mean = 0.0
            reward_part_g_pre_mean = 0.0
            reward_part_service_gap_risk_mean = 0.0
        rollout_reward_per_step = reward_sum / max(int(rollout_env_steps) * num_envs, 1)
        step_target_mode = str(getattr(learner, "step_train_target_mode", "env_reward") or "env_reward").strip().lower()
        bw_target_mode = str(getattr(learner, "bw_train_target_mode", "env_reward") or "env_reward").strip().lower()
        effective_bw_target_mode = step_target_mode if step_target_mode != "env_reward" else bw_target_mode
        if effective_bw_target_mode == "access_raw":
            bw_train_reward_mean = bw_access_reward_mean
        elif effective_bw_target_mode == "access_term":
            bw_train_reward_mean = float(getattr(learner, "bw_reward_w_access", 0.0) or 0.0) * bw_access_reward_mean
        elif effective_bw_target_mode == "weighted_workload_delta":
            bw_train_reward_mean = bw_weighted_reward_mean
        elif effective_bw_target_mode == "weighted_workload_level":
            bw_train_reward_mean = bw_weighted_level_reward_mean
        else:
            bw_train_reward_mean = rollout_reward_per_step
        env_steps = int(rollout_env_steps) * int(num_envs)
        transition_samples = int(len(buffer))
        measured_iteration_time_sec = max(time.perf_counter() - iteration_start, 0.0)
        accounted_iteration_time_sec = float(rollout_total_time_sec + update_total_time_sec)
        iteration_time_sec = max(float(measured_iteration_time_sec), accounted_iteration_time_sec)
        env_steps_per_sec = float(env_steps / max(iteration_time_sec, 1.0e-9))
        samples_per_sec = float(transition_samples / max(iteration_time_sec, 1.0e-9))
        history.append(
            StructuredTrainStepMetrics(
                env_reward_mean=rollout_reward_per_step,
                rollout_reward_per_step=rollout_reward_per_step,
                bw_access_reward_mean=bw_access_reward_mean,
                bw_train_reward_mean=bw_train_reward_mean,
                reward_part_x_acc_mean=reward_part_x_acc_mean,
                reward_part_x_rel_mean=reward_part_x_rel_mean,
                reward_part_processed_ratio_eval_mean=reward_part_processed_ratio_eval_mean,
                reward_part_drop_ratio_eval_mean=reward_part_drop_ratio_eval_mean,
                reward_part_d_pre_mean=reward_part_d_pre_mean,
                reward_part_sat_overlap_eval_mean=reward_part_sat_overlap_eval_mean,
                reward_part_g_pre_mean=reward_part_g_pre_mean,
                reward_part_service_gap_risk_mean=reward_part_service_gap_risk_mean,
                episode_reward=episode_reward_stats["mean"],
                episode_reward_std=episode_reward_stats["std"],
                episode_reward_p25=episode_reward_stats["p25"],
                episode_reward_p75=episode_reward_stats["p75"],
                episode_length_mean=episode_length_mean,
                completed_episode_count=int(completed_episode_count),
                episodes_finished=int(episodes_finished),
                env_steps=int(env_steps),
                transition_samples=int(transition_samples),
                rollout_reset_time_sec=float(rollout_reset_time_sec),
                native_rollout_prepare_time_sec=float(native_rollout_prepare_time_sec),
                rollout_collect_time_sec=float(rollout_collect_time_sec),
                rollout_total_time_sec=float(rollout_total_time_sec),
                rollout_view_build_time_sec=float(rollout_view_build_time_sec),
                update_prepare_time_sec=float(update_prepare_time_sec),
                update_optimize_time_sec=float(update_optimize_time_sec),
                update_total_time_sec=float(update_total_time_sec),
                update_profile_old_logprob_sec=float(update_metrics.get("update_profile_old_logprob_sec", 0.0)),
                update_profile_value_override_sec=float(update_metrics.get("update_profile_value_override_sec", 0.0)),
                update_profile_returns_sec=float(update_metrics.get("update_profile_returns_sec", 0.0)),
                update_profile_stage_cache_sec=float(update_metrics.get("update_profile_stage_cache_sec", 0.0)),
                update_profile_critic_train_sec=float(update_metrics.get("update_profile_critic_train_sec", 0.0)),
                update_profile_actor_train_sec=float(update_metrics.get("update_profile_actor_train_sec", 0.0)),
                update_profile_explained_variance_sec=float(
                    update_metrics.get("update_profile_explained_variance_sec", 0.0)
                ),
                iteration_time_sec=float(iteration_time_sec),
                env_steps_per_sec=float(env_steps_per_sec),
                samples_per_sec=float(samples_per_sec),
                policy_loss=float(update_metrics["policy_loss"]),
                value_loss=float(update_metrics["value_loss"]),
                value_loss_accel=float(update_metrics.get("value_loss_accel", 0.0)),
                value_loss_sat=float(update_metrics.get("value_loss_sat", 0.0)),
                value_loss_bw=float(update_metrics.get("value_loss_bw", 0.0)),
                critic_popart_mean_bw=float(update_metrics.get("critic_popart_mean_bw", 0.0)),
                critic_popart_std_bw=float(update_metrics.get("critic_popart_std_bw", 1.0)),
                explained_variance_accel=float(update_metrics.get("explained_variance_accel", 0.0)),
                explained_variance_sat=float(update_metrics.get("explained_variance_sat", 0.0)),
                explained_variance_bw=float(update_metrics.get("explained_variance_bw", 0.0)),
                entropy=float(update_metrics["entropy"]),
                entropy_accel=float(update_metrics.get("entropy_accel", 0.0)),
                entropy_sat=float(update_metrics.get("entropy_sat", 0.0)),
                entropy_bw=float(update_metrics.get("entropy_bw", 0.0)),
                approx_kl=float(update_metrics["approx_kl"]),
                approx_kl_accel=float(update_metrics.get("approx_kl_accel", 0.0)),
                approx_kl_sat=float(update_metrics.get("approx_kl_sat", 0.0)),
                approx_kl_bw=float(update_metrics.get("approx_kl_bw", 0.0)),
                clip_frac=float(update_metrics["clip_frac"]),
                clip_frac_accel=float(update_metrics.get("clip_frac_accel", 0.0)),
                clip_frac_sat=float(update_metrics.get("clip_frac_sat", 0.0)),
                clip_frac_bw=float(update_metrics.get("clip_frac_bw", 0.0)),
                vs_ref_rows=float(update_metrics.get("vs_ref_rows", 0.0)),
                vs_ref_adv_mean=float(update_metrics.get("vs_ref_adv_mean", 0.0)),
                vs_ref_adv_std=float(update_metrics.get("vs_ref_adv_std", 0.0)),
                vs_ref_positive_frac=float(update_metrics.get("vs_ref_positive_frac", 0.0)),
                vs_ref_rows_accel=float(update_metrics.get("vs_ref_rows_accel", 0.0)),
                vs_ref_adv_mean_accel=float(update_metrics.get("vs_ref_adv_mean_accel", 0.0)),
                vs_ref_adv_std_accel=float(update_metrics.get("vs_ref_adv_std_accel", 0.0)),
                vs_ref_positive_frac_accel=float(update_metrics.get("vs_ref_positive_frac_accel", 0.0)),
                vs_ref_rows_sat=float(update_metrics.get("vs_ref_rows_sat", 0.0)),
                vs_ref_adv_mean_sat=float(update_metrics.get("vs_ref_adv_mean_sat", 0.0)),
                vs_ref_adv_std_sat=float(update_metrics.get("vs_ref_adv_std_sat", 0.0)),
                vs_ref_positive_frac_sat=float(update_metrics.get("vs_ref_positive_frac_sat", 0.0)),
                vs_ref_rows_bw=float(update_metrics.get("vs_ref_rows_bw", 0.0)),
                vs_ref_adv_mean_bw=float(update_metrics.get("vs_ref_adv_mean_bw", 0.0)),
                vs_ref_adv_std_bw=float(update_metrics.get("vs_ref_adv_std_bw", 0.0)),
                vs_ref_positive_frac_bw=float(update_metrics.get("vs_ref_positive_frac_bw", 0.0)),
                grad_norm_accel=float(update_metrics.get("grad_norm_accel", 0.0)),
                grad_norm_sat=float(update_metrics.get("grad_norm_sat", 0.0)),
                grad_norm_bw=float(update_metrics.get("grad_norm_bw", 0.0)),
                **{
                    f"vs_ref_sampling_{kind}_{stage}": float(
                        update_metrics.get(f"vs_ref_sampling_{kind}_{stage}", 0.0)
                    )
                    for stage in ("accel", "sat", "bw")
                    for kind in (
                        "random_count",
                        "time_count",
                        "leverage_count",
                        "uncertainty_count",
                        "horizon_mean",
                        "horizon_min",
                        "horizon_max",
                        "priority_mean",
                        "priority_max",
                    )
                },
                danger_imitation_loss=float(update_metrics.get("danger_imitation_loss", 0.0)),
                danger_imitation_active_rate=float(update_metrics.get("danger_imitation_active_rate", 0.0)),
                bw_counterfactual_credit_active_rate=float(
                    update_metrics.get("bw_counterfactual_credit_active_rate", 0.0)
                ),
                bw_counterfactual_credit_agent_active_rate=float(
                    update_metrics.get("bw_counterfactual_credit_agent_active_rate", 0.0)
                ),
                bw_counterfactual_credit_mean=float(update_metrics.get("bw_counterfactual_credit_mean", 0.0)),
                bw_counterfactual_credit_abs_mean=float(
                    update_metrics.get("bw_counterfactual_credit_abs_mean", 0.0)
                ),
                bw_counterfactual_credit_positive_frac=float(
                    update_metrics.get("bw_counterfactual_credit_positive_frac", 0.0)
                ),
                bw_flow_proxy_aux_loss=float(update_metrics.get("bw_flow_proxy_aux_loss", 0.0)),
                bw_flow_proxy_regression_loss=float(update_metrics.get("bw_flow_proxy_regression_loss", 0.0)),
                bw_flow_proxy_pairwise_acc=float(update_metrics.get("bw_flow_proxy_pairwise_acc", 0.0)),
                bw_flow_proxy_pair_count=float(update_metrics.get("bw_flow_proxy_pair_count", 0.0)),
                bw_grad_norm_policy=float(update_metrics.get("bw_grad_norm_policy", 0.0)),
                bw_grad_norm_aux_scaled=float(update_metrics.get("bw_grad_norm_aux_scaled", 0.0)),
                bw_grad_ratio_aux_to_policy=float(update_metrics.get("bw_grad_ratio_aux_to_policy", 0.0)),
                bw_score_head_grad_norm_policy=float(update_metrics.get("bw_score_head_grad_norm_policy", 0.0)),
                bw_score_head_grad_norm_aux_scaled=float(
                    update_metrics.get("bw_score_head_grad_norm_aux_scaled", 0.0)
                ),
                bw_score_head_grad_ratio_aux_to_policy=float(
                    update_metrics.get("bw_score_head_grad_ratio_aux_to_policy", 0.0)
                ),
                bw_abs_log_ratio_corr_valid_count=float(
                    update_metrics.get("bw_abs_log_ratio_corr_valid_count", 0.0)
                ),
                bw_abs_log_ratio_corr_latent_count=float(
                    update_metrics.get("bw_abs_log_ratio_corr_latent_count", 0.0)
                ),
                bw_kappa_mean=float(update_metrics.get("bw_kappa_mean", 0.0)),
                bw_kappa_p10=float(update_metrics.get("bw_kappa_p10", 0.0)),
                bw_kappa_p90=float(update_metrics.get("bw_kappa_p90", 0.0)),
                bw_kappa_hi_frac=float(update_metrics.get("bw_kappa_hi_frac", 0.0)),
                bw_delta_student_teacher_corr=float(
                    update_metrics.get("bw_delta_student_teacher_corr", 0.0)
                ),
                bw_delta_student_true_corr=float(update_metrics.get("bw_delta_student_true_corr", 0.0)),
                bw_delta_actor_mix_alpha=float(update_metrics.get("bw_delta_actor_mix_alpha", 0.0)),
                bw_delta_teacher_used=float(update_metrics.get("bw_delta_teacher_used", 0.0)),
                bw_delta_teacher_observed=float(update_metrics.get("bw_delta_teacher_observed", 0.0)),
                bw_delta_teacher_probe_only=float(update_metrics.get("bw_delta_teacher_probe_only", 0.0)),
                bw_branch_gate_snr=float(update_metrics.get("bw_branch_gate_snr", 0.0)),
                bw_branch_gate_triggered=float(update_metrics.get("bw_branch_gate_triggered", 0.0)),
                bw_actor_update_skipped=float(update_metrics.get("bw_actor_update_skipped", 0.0)),
                clean_mean_target_gap=float(update_metrics.get("clean_mean_target_gap", 0.0)),
                clean_mean_update_shift=float(update_metrics.get("clean_mean_update_shift", 0.0)),
                clean_update_to_target_ratio=float(update_metrics.get("clean_update_to_target_ratio", 0.0)),
                clean_target_beats_ref_frac=float(update_metrics.get("clean_target_beats_ref_frac", 0.0)),
                clean_target_beats_sampled_frac=float(update_metrics.get("clean_target_beats_sampled_frac", 0.0)),
                clean_measured_kl=float(update_metrics.get("clean_measured_kl", 0.0)),
                clean_kl_coef=float(update_metrics.get("clean_kl_coef", 0.0)),
                clean_row_sample_eligible=float(update_metrics.get("clean_row_sample_eligible", 0.0)),
                clean_row_sample_count=float(update_metrics.get("clean_row_sample_count", 0.0)),
                clean_row_sample_frac=float(update_metrics.get("clean_row_sample_frac", 0.0)),
                clean_ref_branch_count=float(update_metrics.get("clean_ref_branch_count", 0.0)),
                clean_perturb_branch_count=float(update_metrics.get("clean_perturb_branch_count", 0.0)),
                clean_gate_branch_count=float(update_metrics.get("clean_gate_branch_count", 0.0)),
                clean_row_filter_count=float(update_metrics.get("clean_row_filter_count", 0.0)),
                clean_row_filter_frac=float(update_metrics.get("clean_row_filter_frac", 0.0)),
                clean_candidate_positive_branch_count=float(update_metrics.get("clean_candidate_positive_branch_count", 0.0)),
                clean_candidate_selected_rows=float(update_metrics.get("clean_candidate_selected_rows", 0.0)),
            )
        )
    if trace_fn is not None:
        trace_fn("run_structured_training:done")
    return history
