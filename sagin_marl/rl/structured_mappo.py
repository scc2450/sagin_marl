from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass, replace
from itertools import product
import math
import time
from typing import Any, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from sagin_marl.env.structured_driver import StructuredBatchStepResult
from sagin_marl.env.structured_gpu_rollout_runtime import (
    StructuredGpuAccelObsView,
    StructuredGpuBwObsView,
    StructuredGpuNativeRolloutProgram,
    StructuredGpuSatObsView,
)
from sagin_marl.env.native_cuda import bindings as native_cuda

from .baselines import (
    cluster_center_accel_policy,
    observable_cluster_accel_policy,
    queue_aware_bw_policy,
    queue_aware_policy,
    queue_aware_sat_policy,
)
from .distributions import squash_action
from .structured_buffer import (
    StructuredBootstrapBatchView,
    StructuredRolloutBuffer,
    StructuredRolloutViews,
    StructuredStageTrainingBatch,
)
from .structured_types import LocalBwState
from .native_actor_cuda import NativeActorCudaBinding, build_native_actor_cuda_binding

def _explained_variance(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_flat = pred.reshape(-1).detach()
    target_flat = target.reshape(-1).detach()
    if target_flat.numel() <= 1:
        return 0.0
    target_var = torch.var(target_flat, unbiased=False)
    target_var_value = float(target_var.item())
    if target_var_value <= 1e-8:
        return 0.0
    residual = target_flat - pred_flat
    residual_var = float(torch.var(residual, unbiased=False).item())
    return 1.0 - residual_var / target_var_value


def _padcat_tensors(values: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
    if not values:
        raise ValueError("values must be non-empty")
    ndim = values[0].ndim
    if any(v.ndim != ndim for v in values):
        raise ValueError("all tensors must share ndim")
    ref_shape = values[0].shape[1:]
    same_shape = all(tuple(v.shape[1:]) == tuple(ref_shape) for v in values)
    if same_shape:
        converted = [v if (v.device == device) else v.to(device) for v in values]
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


def _tensor_field_names(value: Any) -> tuple[str, ...]:
    explicit = getattr(value, "_tensor_fields", None)
    if explicit is not None:
        return tuple(str(name) for name in explicit)
    if is_dataclass(value):
        return tuple(str(field.name) for field in fields(value))
    return ()


def _collate_dataclass(items: Sequence[Any], device: torch.device) -> Any:
    if not items:
        raise ValueError("items must be non-empty")
    sample = items[0]
    field_names = _tensor_field_names(sample)
    if not field_names:
        raise TypeError("_collate_dataclass expects fixed tensor-field objects")
    kwargs = {}
    for field_name in field_names:
        values = [getattr(item, field_name) for item in items]
        tensor_values = [_field_to_tensor(value) for value in values]
        kwargs[field_name] = _padcat_tensors(tensor_values, device)
    return type(sample)(**kwargs)


def _index_dataclass(batch: Any, indices: torch.Tensor) -> Any:
    field_names = _tensor_field_names(batch)
    if not field_names:
        raise TypeError("_index_dataclass expects a fixed tensor-field object")
    kwargs = {}
    for field_name in field_names:
        value = getattr(batch, field_name)
        if not torch.is_tensor(value):
            raise TypeError(f"Unsupported field type for {field_name}: {type(value)!r}")
        kwargs[field_name] = value.index_select(0, indices.to(device=value.device, dtype=torch.long))
    return type(batch)(**kwargs)


def _slice_dataclass(batch: Any, start: int, end: int) -> Any:
    field_names = _tensor_field_names(batch)
    if not field_names:
        raise TypeError("_slice_dataclass expects a fixed tensor-field object")
    kwargs = {}
    for field_name in field_names:
        value = getattr(batch, field_name)
        if not torch.is_tensor(value):
            raise TypeError(f"Unsupported field type for {field_name}: {type(value)!r}")
        kwargs[field_name] = value[int(start) : int(end)]
    return type(batch)(**kwargs)


def _split_dataclass_by_counts(batch: Any, counts: Sequence[int]) -> list[Any]:
    pieces: list[Any] = []
    cursor = 0
    for count in counts:
        pieces.append(_slice_dataclass(batch, cursor, cursor + int(count)))
        cursor += int(count)
    return pieces


def _to_device_dataclass(batch: Any, device: torch.device) -> Any:
    field_names = _tensor_field_names(batch)
    if not field_names:
        raise TypeError("_to_device_dataclass expects a fixed tensor-field object")
    kwargs = {}
    for field_name in field_names:
        value = getattr(batch, field_name)
        if torch.is_tensor(value):
            kwargs[field_name] = value if value.device == device else value.to(device)
            continue
        if isinstance(value, np.ndarray):
            kwargs[field_name] = torch.as_tensor(value, device=device)
            continue
        raise TypeError(f"Unsupported field type for {field_name}: {type(value)!r}")
    return type(batch)(**kwargs)


def _split_tensor_by_counts(value: torch.Tensor, counts: Sequence[int]) -> list[torch.Tensor]:
    pieces: list[torch.Tensor] = []
    cursor = 0
    for count in counts:
        pieces.append(value[cursor : cursor + int(count)])
        cursor += int(count)
    return pieces


def _split_sum_by_counts(value: torch.Tensor, counts: Sequence[int]) -> list[torch.Tensor]:
    return [piece.sum() for piece in _split_tensor_by_counts(value, counts)]


def _split_tensor_and_sums_by_counts(
    value: torch.Tensor,
    counts: Sequence[int],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    pieces = _split_tensor_by_counts(value, counts)
    return pieces, [piece.sum() for piece in pieces]


def _split_tensor_and_numpy_by_counts(
    value: torch.Tensor,
    counts: Sequence[int],
) -> tuple[list[torch.Tensor], list[np.ndarray]]:
    pieces = _split_tensor_by_counts(value, counts)
    return pieces, [piece.detach().cpu().numpy() for piece in pieces]


def _to_cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value if value.device.type == "cpu" else value.cpu()


def _split_cpu_tensor_by_counts(value: torch.Tensor, counts: Sequence[int]) -> list[torch.Tensor]:
    return _split_tensor_by_counts(_to_cpu_tensor(value), counts)


def _split_cpu_sum_by_counts(value: torch.Tensor, counts: Sequence[int]) -> list[torch.Tensor]:
    return [piece.sum() for piece in _split_cpu_tensor_by_counts(value, counts)]


def _split_cpu_tensor_and_sums_by_counts(
    value: torch.Tensor,
    counts: Sequence[int],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    pieces = _split_cpu_tensor_by_counts(value, counts)
    return pieces, [piece.sum() for piece in pieces]


def _split_cpu_tensor_and_numpy_by_counts(
    value: torch.Tensor,
    counts: Sequence[int],
) -> tuple[list[torch.Tensor], list[np.ndarray]]:
    pieces = _split_cpu_tensor_by_counts(value, counts)
    return pieces, [piece.numpy() for piece in pieces]


def _bw_valid_and_effective_loc(local_state: Any, loc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid_mask = (local_state.gu_mask > 0.5) & (local_state.bw_valid_mask > 0.5)
    latent_mask = valid_mask.clone()
    if valid_mask.shape[-1] > 0:
        ref_idx = torch.where(
            valid_mask,
            torch.arange(valid_mask.shape[-1], device=valid_mask.device, dtype=torch.long).view(1, -1).expand_as(valid_mask),
            torch.full_like(valid_mask, -1, dtype=torch.long),
        ).amax(dim=-1)
        active_rows = torch.nonzero(valid_mask.sum(dim=-1) > 1, as_tuple=False).flatten()
        if active_rows.numel() > 0:
            latent_mask[active_rows, ref_idx[active_rows]] = False
    effective_loc = torch.where(latent_mask, loc, torch.zeros_like(loc))
    return valid_mask, effective_loc


def _step_team_reward(step_result: Any) -> float:
    rewards = getattr(step_result, "rewards", {}) or {}
    if rewards:
        return float(next(iter(rewards.values())))
    return float(getattr(step_result, "team_reward", 0.0) or 0.0)


def _step_terminated(step_result: Any) -> bool:
    terminations = getattr(step_result, "terminations", {}) or {}
    if terminations:
        return bool(next(iter(terminations.values())))
    return bool(getattr(step_result, "terminated", False))


def _step_truncated(step_result: Any) -> bool:
    truncations = getattr(step_result, "truncations", {}) or {}
    if truncations:
        return bool(next(iter(truncations.values())))
    return bool(getattr(step_result, "truncated", False))


def _structured_zero_update_metrics(**overrides: float) -> dict[str, float]:
    metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_frac": 0.0,
        "value_loss_accel": 0.0,
        "value_loss_sat": 0.0,
        "value_loss_bw": 0.0,
        "critic_popart_mean_bw": 0.0,
        "critic_popart_std_bw": 1.0,
        "explained_variance_accel": 0.0,
        "explained_variance_sat": 0.0,
        "explained_variance_bw": 0.0,
        "entropy_accel": 0.0,
        "entropy_sat": 0.0,
        "entropy_bw": 0.0,
        "approx_kl_accel": 0.0,
        "approx_kl_sat": 0.0,
        "approx_kl_bw": 0.0,
        "clip_frac_accel": 0.0,
        "clip_frac_sat": 0.0,
        "clip_frac_bw": 0.0,
        "danger_imitation_loss": 0.0,
        "danger_imitation_active_rate": 0.0,
        "bw_counterfactual_credit_active_rate": 0.0,
        "bw_counterfactual_credit_agent_active_rate": 0.0,
        "bw_counterfactual_credit_mean": 0.0,
        "bw_counterfactual_credit_abs_mean": 0.0,
        "bw_counterfactual_credit_positive_frac": 0.0,
        "bw_flow_proxy_aux_loss": 0.0,
        "bw_flow_proxy_regression_loss": 0.0,
        "bw_flow_proxy_pairwise_acc": 0.0,
        "bw_flow_proxy_pair_count": 0.0,
        "bw_grad_norm_policy": 0.0,
        "bw_grad_norm_aux_scaled": 0.0,
        "bw_grad_ratio_aux_to_policy": 0.0,
        "bw_score_head_grad_norm_policy": 0.0,
        "bw_score_head_grad_norm_aux_scaled": 0.0,
        "bw_score_head_grad_ratio_aux_to_policy": 0.0,
        "bw_abs_log_ratio_corr_valid_count": 0.0,
        "bw_abs_log_ratio_corr_latent_count": 0.0,
        "bw_kappa_mean": 0.0,
        "bw_kappa_p10": 0.0,
        "bw_kappa_p90": 0.0,
        "bw_kappa_hi_frac": 0.0,
        "bw_delta_student_teacher_corr": 0.0,
        "bw_delta_student_true_corr": 0.0,
        "bw_delta_actor_mix_alpha": 0.0,
        "bw_delta_teacher_used": 0.0,
        "bw_delta_teacher_observed": 0.0,
        "bw_delta_teacher_probe_only": 0.0,
        "bw_branch_gate_snr": 0.0,
        "bw_branch_gate_triggered": 0.0,
        "bw_actor_update_skipped": 0.0,
        "clean_mean_target_gap": 0.0,
        "clean_mean_update_shift": 0.0,
        "clean_update_to_target_ratio": 0.0,
        "clean_target_beats_ref_frac": 0.0,
        "clean_target_beats_sampled_frac": 0.0,
        "clean_measured_kl": 0.0,
        "clean_kl_coef": 0.0,
        "clean_row_sample_eligible": 0.0,
        "clean_row_sample_count": 0.0,
        "clean_row_sample_frac": 0.0,
        "clean_ref_branch_count": 0.0,
        "clean_perturb_branch_count": 0.0,
        "clean_gate_branch_count": 0.0,
        "clean_row_filter_count": 0.0,
        "clean_row_filter_frac": 0.0,
        "clean_candidate_positive_branch_count": 0.0,
        "clean_candidate_selected_rows": 0.0,
        "vs_ref_rows": 0.0,
        "vs_ref_adv_mean": 0.0,
        "vs_ref_adv_std": 0.0,
        "vs_ref_positive_frac": 0.0,
    }
    for _stage_name in ("accel", "sat", "bw"):
        metrics[f"vs_ref_rows_{_stage_name}"] = 0.0
        metrics[f"vs_ref_adv_mean_{_stage_name}"] = 0.0
        metrics[f"vs_ref_adv_std_{_stage_name}"] = 0.0
        metrics[f"vs_ref_positive_frac_{_stage_name}"] = 0.0
        for _kind in ("random", "time", "leverage", "uncertainty"):
            metrics[f"vs_ref_sampling_{_kind}_count_{_stage_name}"] = 0.0
        for _stat in ("mean", "min", "max"):
            metrics[f"vs_ref_sampling_horizon_{_stat}_{_stage_name}"] = 0.0
        for _stat in ("mean", "max"):
            metrics[f"vs_ref_sampling_priority_{_stat}_{_stage_name}"] = 0.0
    metrics.update({key: float(value) for key, value in overrides.items()})
    return metrics


def _masked_simplex_probs(logits: torch.Tensor, mask: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    safe_logits = logits.masked_fill(~mask, float("-inf"))
    has_valid = mask.any(dim=-1, keepdim=True)
    safe_logits = torch.where(has_valid, safe_logits, torch.zeros_like(safe_logits))
    probs = torch.softmax(safe_logits, dim=-1)
    probs = probs * mask.to(probs.dtype)
    norm = probs.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    return probs / norm


def _grad_l2_norm(loss: torch.Tensor, params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    if not torch.is_tensor(loss) or not loss.requires_grad:
        device = loss.device if torch.is_tensor(loss) else torch.device("cpu")
        return torch.zeros((), dtype=torch.float32, device=device)
    trainable = [param for param in params if param.requires_grad]
    if not trainable:
        return torch.zeros((), dtype=torch.float32, device=loss.device)
    grads = torch.autograd.grad(
        loss,
        trainable,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    total = torch.zeros((), dtype=torch.float32, device=loss.device)
    for grad in grads:
        if grad is None:
            continue
        total = total + grad.detach().to(dtype=torch.float32).pow(2).sum()
    return total.sqrt()


def _safe_corrcoef(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(x) or not torch.is_tensor(y):
        raise TypeError("_safe_corrcoef expects tensors")
    x = x.reshape(-1).detach().to(dtype=torch.float32)
    y = y.reshape(-1).detach().to(dtype=torch.float32, device=x.device)
    finite = torch.isfinite(x) & torch.isfinite(y)
    if int(finite.sum().item()) <= 1:
        return torch.zeros((), dtype=torch.float32, device=x.device)
    x = x[finite]
    y = y[finite]
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    x_std = x_centered.pow(2).mean().sqrt()
    y_std = y_centered.pow(2).mean().sqrt()
    if float(x_std.item()) <= 1.0e-8 or float(y_std.item()) <= 1.0e-8:
        return torch.zeros((), dtype=torch.float32, device=x.device)
    return (x_centered * y_centered).mean() / (x_std * y_std).clamp_min(1.0e-8)


def _normalize_exec_source(raw: str | None) -> str:
    source = str("policy" if raw is None else raw).strip().lower()
    allowed = {
        "policy",
        "policy_single_uav_queue_aware",
        "queue_aware",
        "cluster_center_queue_aware",
        "observable_cluster_queue_aware",
        "zero",
        "teacher",
        "uniform",
        "random",
        "link_priority",
        "demand_priority",
        "lyapunov",
        "dpp_resource_bw",
        "topology_dpp_accel",
        "topology_dpp_bw",
        "topology_dpp_sat",
    }
    if source not in allowed:
        raise ValueError(f"Unsupported structured exec source: {raw!r}")
    return source


_NATIVE_ZERO_ACTION_SOURCES = {"zero"}
_NATIVE_POLICY_ACTION_SOURCES = {"policy", "policy_single_uav_queue_aware"}
_NATIVE_BASELINE_ACTION_SOURCES = {
    "uniform",
    "random",
    "link_priority",
    "demand_priority",
    "queue_aware",
    "cluster_center_queue_aware",
    "observable_cluster_queue_aware",
    "lyapunov",
    "dpp_resource_bw",
    "topology_dpp_accel",
    "topology_dpp_bw",
    "topology_dpp_sat",
}
_NATIVE_LIVE_ACTION_SOURCES = _NATIVE_POLICY_ACTION_SOURCES | _NATIVE_BASELINE_ACTION_SOURCES | {"teacher"}
_NATIVE_EXEC_SOURCES = _NATIVE_ZERO_ACTION_SOURCES | _NATIVE_LIVE_ACTION_SOURCES


def _native_source_uses_live_action(source: str) -> bool:
    return str(source).strip().lower() in _NATIVE_LIVE_ACTION_SOURCES


def _native_exec_source_mode_code(source: object) -> int:
    source_s = str(source or "policy").strip().lower()
    table = {
        "policy": native_cuda.SOURCE_POLICY,
        "policy_single_uav_queue_aware": native_cuda.SOURCE_POLICY,
        "teacher": native_cuda.SOURCE_POLICY,
        "zero": native_cuda.SOURCE_ZERO,
        "uniform": native_cuda.SOURCE_UNIFORM,
        "random": native_cuda.SOURCE_RANDOM,
        "link_priority": native_cuda.SOURCE_LINK_PRIORITY,
        "demand_priority": native_cuda.SOURCE_DEMAND_PRIORITY,
        "queue_aware": native_cuda.SOURCE_QUEUE_AWARE,
        "cluster_center_queue_aware": native_cuda.SOURCE_CLUSTER_CENTER_QUEUE_AWARE,
        "observable_cluster_queue_aware": native_cuda.SOURCE_OBSERVABLE_CLUSTER_QUEUE_AWARE,
        "lyapunov": native_cuda.SOURCE_LYAPUNOV,
        "dpp_resource_bw": native_cuda.SOURCE_DPP_RESOURCE_BW,
        "topology_dpp_accel": native_cuda.SOURCE_TOPOLOGY_DPP_ACCEL,
        "topology_dpp_bw": native_cuda.SOURCE_TOPOLOGY_DPP_BW,
        "topology_dpp_sat": native_cuda.SOURCE_TOPOLOGY_DPP_SAT,
    }
    if source_s not in table:
        raise RuntimeError(f"native rollout source {source_s!r} is not supported.")
    return int(table[source_s])


def direct_branch_delta_actor_only_enabled(
    cfg: Any | None,
    *,
    train_accel: bool,
    train_sat: bool,
    train_bw: bool,
) -> bool:
    if cfg is None:
        return False
    override_mode = str(getattr(cfg, "bw_actor_advantage_override_mode", "") or "").strip().lower()
    return (
        override_mode == "branch_delta"
        and bool(train_bw)
        and not bool(train_accel)
        and not bool(train_sat)
    )


def bw_actor_only_signal_critic_free_enabled(
    cfg: Any | None,
    *,
    train_accel: bool,
    train_sat: bool,
    train_bw: bool,
) -> bool:
    if cfg is None:
        return False
    bw_only = bool(train_bw) and not bool(train_accel) and not bool(train_sat)
    if not bw_only:
        return False
    if direct_branch_delta_actor_only_enabled(
        cfg,
        train_accel=train_accel,
        train_sat=train_sat,
        train_bw=train_bw,
    ):
        return True
    if bool(getattr(cfg, "bw_clean_per_user_enabled", False)):
        return True
    if bool(getattr(cfg, "bw_marginal_teacher_sample_enabled", False)):
        return True
    slot_base_mode = str(getattr(cfg, "bw_slot_advantage_base_mode", "sample") or "sample").strip().lower()
    if bool(getattr(cfg, "structured_bw_per_slot_surrogate_enabled", False)) and slot_base_mode == "zero":
        return True
    return False


def sat_clean_joint_critic_free_enabled(
    cfg: Any | None,
    *,
    train_accel: bool,
    train_sat: bool,
    train_bw: bool,
) -> bool:
    if cfg is None:
        return False
    return (
        bool(getattr(cfg, "sat_clean_joint_enabled", False))
        and bool(train_sat)
        and not bool(train_accel)
        and not bool(train_bw)
    )


def _current_obs_list(driver: Any) -> list[dict[str, np.ndarray]]:
    env = driver.env
    return [env._get_obs(i) for i in range(len(env.agents))]


def _heuristic_accel(
    obs_list: Sequence[dict[str, np.ndarray]],
    cfg: Any,
    heuristic_policy: str,
    *,
    centers: np.ndarray | None = None,
    counts: np.ndarray | None = None,
) -> np.ndarray:
    """Backward-compatible heuristic helper used by legacy diagnostics."""
    policy = str(heuristic_policy or "queue_aware").strip().lower()
    if policy == "cluster_center_queue_aware":
        return np.asarray(cluster_center_accel_policy(list(obs_list), cfg, centers, counts), dtype=np.float32)
    if policy == "observable_cluster_queue_aware":
        return np.asarray(observable_cluster_accel_policy(list(obs_list), cfg), dtype=np.float32)
    if policy in {"queue_aware", "queue_aware_accel"}:
        accel, _, _ = queue_aware_policy(list(obs_list), cfg)
        return np.asarray(accel, dtype=np.float32)
    raise ValueError(f"Unsupported accel heuristic policy: {heuristic_policy}")


def _heuristic_sat(obs_list: Sequence[dict[str, np.ndarray]], cfg: Any, heuristic_policy: str) -> np.ndarray:
    """Backward-compatible SAT heuristic helper used by legacy diagnostics."""
    policy = str(heuristic_policy or "queue_aware").strip().lower()
    if policy in {"queue_aware", "queue_aware_sat", "cluster_center_queue_aware", "observable_cluster_queue_aware"}:
        return np.asarray(queue_aware_sat_policy(list(obs_list), cfg), dtype=np.float32)
    raise ValueError(f"Unsupported SAT heuristic policy: {heuristic_policy}")


def _heuristic_bw(obs_list: Sequence[dict[str, np.ndarray]], cfg: Any, heuristic_policy: str) -> np.ndarray:
    policy = str(heuristic_policy or "queue_aware").strip().lower()
    if policy in {"queue_aware", "queue_aware_bw", "cluster_center_queue_aware", "observable_cluster_queue_aware"}:
        return np.asarray(queue_aware_bw_policy(list(obs_list), cfg), dtype=np.float32)
    raise ValueError(f"Unsupported BW heuristic policy: {heuristic_policy}")


def _sat_action_select_k_from_cfg(cfg: Any) -> int:
    raw = getattr(cfg, "sat_action_select_k", None)
    if raw is not None and int(raw) > 0:
        return int(raw)
    n_rf = max(int(getattr(cfg, "N_RF", 0) or 0), 1)
    num_sat = max(int(getattr(cfg, "num_sat", 0) or 0), 0)
    sat_select_cfg = getattr(cfg, "sat_num_select", None)
    sat_select = int(sat_select_cfg) if sat_select_cfg is not None and int(sat_select_cfg) > 0 else n_rf
    upper_sat = num_sat if num_sat > 0 else sat_select
    return max(min(int(upper_sat), int(n_rf), int(sat_select)), 1)


def _refresh_stage_obs_cache(driver: Any) -> None:
    if driver._stage_assoc is None or driver._stage_candidates is None:
        raise RuntimeError("run_accel_stage must be called before refreshing stage obs cache")
    env = driver.env
    _, eta_slots = env._compute_access_rates(
        driver._stage_assoc,
        driver._stage_candidates,
        driver._zero_bw_action_matrix(),
        record_exec=False,
    )
    env._store_cached_access_stage_context(
        driver._stage_assoc,
        driver._stage_candidates,
        eta=eta_slots,
        bw_valid_mask=driver._stage_bw_valid_mask,
        snapshot_step_t=int(env.t),
    )
    if driver._stage_sat_pos is not None and driver._stage_sat_vel is not None and driver._stage_visible is not None:
        env._cache_sat_obs(driver._stage_sat_pos, driver._stage_sat_vel, driver._stage_visible)


def _sat_mask_to_ids(driver: Any, sat_mask: np.ndarray) -> np.ndarray:
    cfg = driver.env.cfg
    select_k = _sat_action_select_k_from_cfg(cfg)
    out = np.full((cfg.num_uav, select_k), -1, dtype=np.int64)
    if driver._stage_visible is None:
        raise RuntimeError("run_accel_stage must be called before decoding sat masks")
    sat_mask_arr = np.asarray(sat_mask, dtype=np.float32)
    visible_width = max(
        min(
            int(getattr(cfg, "per_uav_visible_sat_token_max", cfg.sats_obs_max) or cfg.sats_obs_max),
            int(cfg.num_sat),
        ),
        0,
    )
    for u in range(cfg.num_uav):
        visible = driver._stage_visible[u][:visible_width]
        active_slots = np.flatnonzero(sat_mask_arr[u] > 0.5)
        mapped: list[int] = []
        for slot in active_slots.tolist():
            if 0 <= int(slot) < len(visible):
                sat_idx = int(visible[int(slot)])
                if sat_idx not in mapped:
                    mapped.append(sat_idx)
            if len(mapped) >= select_k:
                break
        if mapped:
            out[u, : len(mapped)] = np.asarray(mapped[:select_k], dtype=np.int64)
    return out


class StructuredNativeRolloutExecutor:
    """Executor boundary for the persistent native tensor rollout runtime.

    The executor owns the env-side runtime-view API calls. MAPPO policy code only
    reads obs/result tensors that have already been published into the runtime.
    """

    def __init__(self, drivers: Any, *, tensor_device: torch.device, cfg: Any | None = None) -> None:
        set_group_tensor_device = getattr(drivers, "set_tensor_device", None)
        if callable(set_group_tensor_device):
            set_group_tensor_device(tensor_device)
        set_fast_random = getattr(drivers, "set_native_rollout_fast_random", None)
        if callable(set_fast_random):
            set_fast_random(True)
        runtime = getattr(drivers, "native_rollout_runtime", None)
        if runtime is None:
            raise RuntimeError("native tensor rollout executor requires a persistent native rollout runtime.")
        self.drivers = drivers
        self.runtime = runtime
        program_factory = getattr(self.drivers, "native_rollout_program", None)
        if not callable(program_factory):
            raise RuntimeError("native rollout executor requires native_rollout_program.")
        self.program = program_factory()
        if not isinstance(self.program, StructuredGpuNativeRolloutProgram):
            raise RuntimeError("native_rollout_program must return StructuredGpuNativeRolloutProgram.")


class _StructuredMAPPOGpuActorBridge:
    """MAPPO actor adapter for the env-owned native GPU rollout program."""

    def __init__(self, learner: "StructuredMAPPO") -> None:
        self.learner = learner
        self.source_by_stage = {
            0: str(learner.exec_source_by_stage.get(0, "policy")),
            1: str(learner.exec_source_by_stage.get(1, "policy")),
            2: str(learner.exec_source_by_stage.get(2, "policy")),
        }
        self.accel_batch = None
        self.sat_batch = None
        self.bw_batch = None
        self.num_agents = 0
        self.v_step_bootstrap = None
        self.v_sat = None
        self.v_bw = None
        self.accel_action_batch = None
        self.sat_action_batch = None
        self.bw_action_batch = None
        self.bw_ref_action_batch = None
        self.accel_logprob_batch = None
        self.sat_logprob_batch = None
        self.bw_logprob_batch = None
        self.bw_logprobs_per_agent_batch = None
        self._policy_actor_binding: NativeActorCudaBinding | None = None
        self._teacher_actor_binding: NativeActorCudaBinding | None = None
        self._runtime_abi: native_cuda.NativeCudaRuntimeABI | None = None
        self._teacher_deterministic_override: bool | None = None

    @staticmethod
    def _same_runtime_device(value_device: torch.device, runtime_device: torch.device | str) -> bool:
        target_device = torch.device(runtime_device)
        value_device = torch.device(value_device)
        return value_device == target_device or (
            value_device.type == target_device.type == "cuda"
            and (target_device.index is None or value_device.index in {None, target_device.index})
        )

    def begin_step(self, *, deterministic: bool) -> None:
        self.accel_batch = None
        self.sat_batch = None
        self.bw_batch = None
        self.num_agents = 0
        self.v_step_bootstrap = None
        self.v_sat = None
        self.v_bw = None
        self.accel_action_batch = None
        self.sat_action_batch = None
        self.bw_action_batch = None
        self.bw_ref_action_batch = None
        self.accel_logprob_batch = None
        self.sat_logprob_batch = None
        self.bw_logprob_batch = None
        self.bw_logprobs_per_agent_batch = None

    def _num_agents_from_accel_obs(self, accel_obs: Any, num_envs: int) -> int:
        return max(int(accel_obs.ego_features.shape[0]) // max(int(num_envs), 1), 1)

    @staticmethod
    def _num_agents_from_flat_obs_tensor(value: Any, num_envs: int) -> int:
        if torch.is_tensor(value) and int(value.ndim) >= 1:
            return max(int(value.shape[0]) // max(int(num_envs), 1), 1)
        return 1

    def _single_uav_policy_uav_id(self, num_agents: int) -> int:
        raw = 0 if self.learner.cfg is None else int(getattr(self.learner.cfg, "bw_single_uav_policy_uav_id", 0) or 0)
        num_agents_i = max(int(num_agents), 1)
        if raw < 0 or raw >= num_agents_i:
            raise ValueError(
                f"bw_single_uav_policy_uav_id={raw} is out of range for num_agents={num_agents_i}."
            )
        return int(raw)

    def bind_source_modes(self, runtime: Any) -> None:
        accel_mode = int(runtime.main.accel_actor_source_mode_code)
        sat_mode = int(runtime.main.sat_actor_source_mode_code)
        bw_mode = int(runtime.main.bw_actor_source_mode_code)
        accel_source = self.source_by_stage[0]
        sat_source = self.source_by_stage[1]
        bw_source = self.source_by_stage[2]
        expected_accel_mode = _native_exec_source_mode_code(accel_source)
        expected_sat_mode = _native_exec_source_mode_code(sat_source)
        expected_bw_mode = _native_exec_source_mode_code(bw_source)
        if accel_mode != expected_accel_mode:
            raise RuntimeError("native accel source mode code does not match the bound action producer.")
        if sat_mode != expected_sat_mode:
            raise RuntimeError("native SAT source mode code does not match the bound action producer.")
        if bw_mode != expected_bw_mode:
            raise RuntimeError("native BW source mode code does not match the bound action producer.")

        self._runtime_abi = self._runtime_native_abi(runtime)
        flat_policy_actor = self.learner._flat_policy_actor_native_module_enabled()
        needs_policy = any(
            self.learner._native_policy_source_requires_cuda_binding(source)
            for source in (accel_source, sat_source, bw_source)
        )
        needs_teacher = any(source == "teacher" for source in (accel_source, sat_source, bw_source))
        self._policy_actor_binding = self.learner._require_native_actor_policy_binding() if needs_policy else None
        self._teacher_actor_binding = self.learner._require_native_actor_teacher_binding() if needs_teacher else None
        if needs_teacher:
            raw_teacher_det = self.learner.teacher_deterministic
            self._teacher_deterministic_override = None if raw_teacher_det is None else bool(raw_teacher_det)
        else:
            self._teacher_deterministic_override = None

        if accel_source == "policy":
            self.write_accel_action = (
                self._write_accel_action_policy_module
                if flat_policy_actor
                else self._write_accel_action_policy
            )  # type: ignore[method-assign]
        elif accel_source == "zero":
            self.write_accel_action = self._write_accel_action_zero  # type: ignore[method-assign]
        elif accel_source == "queue_aware":
            self.write_accel_action = self._write_accel_action_queue_aware  # type: ignore[method-assign]
        elif accel_source == "cluster_center_queue_aware":
            self.write_accel_action = self._write_accel_action_cluster_center_queue_aware  # type: ignore[method-assign]
        elif accel_source in {
            "uniform",
            "random",
            "link_priority",
            "demand_priority",
            "lyapunov",
            "topology_dpp_accel",
            "observable_cluster_queue_aware",
        }:
            self.write_accel_action = self._write_accel_action_baseline  # type: ignore[method-assign]
        elif accel_source == "teacher":
            self.write_accel_action = self._write_accel_action_teacher  # type: ignore[method-assign]
        else:
            raise RuntimeError(f"native main-kernel accel exec source {accel_source!r} is not tensor-native.")

        if sat_source == "policy":
            self.write_sat_action = (
                self._write_sat_action_policy_module
                if flat_policy_actor
                else self._write_sat_action_policy
            )  # type: ignore[method-assign]
        elif sat_source == "zero":
            self.write_sat_action = self._write_sat_action_zero  # type: ignore[method-assign]
        elif sat_source in {"queue_aware", "cluster_center_queue_aware", "observable_cluster_queue_aware"}:
            self.write_sat_action = self._write_sat_action_queue_aware  # type: ignore[method-assign]
        elif sat_source in {"uniform", "random", "link_priority", "demand_priority", "lyapunov", "topology_dpp_sat"}:
            self.write_sat_action = self._write_sat_action_baseline  # type: ignore[method-assign]
        elif sat_source == "teacher":
            self.write_sat_action = self._write_sat_action_teacher  # type: ignore[method-assign]
        else:
            raise RuntimeError(f"native main-kernel SAT exec source {sat_source!r} is not tensor-native.")

        if bw_source == "policy":
            self.write_bw_action = (
                self._write_bw_action_policy_module
                if flat_policy_actor
                else self._write_bw_action_policy
            )  # type: ignore[method-assign]
        elif bw_source == "policy_single_uav_queue_aware":
            self.write_bw_action = self._write_bw_action_policy_single_uav_queue_aware  # type: ignore[method-assign]
        elif bw_source == "zero":
            self.write_bw_action = self._write_bw_action_zero  # type: ignore[method-assign]
        elif bw_source in {"queue_aware", "cluster_center_queue_aware", "observable_cluster_queue_aware"}:
            self.write_bw_action = self._write_bw_action_queue_aware  # type: ignore[method-assign]
        elif bw_source in {
            "uniform",
            "random",
            "link_priority",
            "demand_priority",
            "lyapunov",
            "dpp_resource_bw",
            "topology_dpp_bw",
        }:
            self.write_bw_action = self._write_bw_action_baseline  # type: ignore[method-assign]
        elif bw_source == "teacher":
            self.write_bw_action = self._write_bw_action_teacher  # type: ignore[method-assign]
        else:
            raise RuntimeError(f"native main-kernel BW exec source {bw_source!r} is not tensor-native.")

    def begin_horizon(self, *, horizon: int, runtime: Any, deterministic: bool) -> None:
        del horizon, deterministic
        self._runtime_abi = self._runtime_native_abi(runtime)

    def _write_accel_action_zero(
        self,
        accel_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        deterministic: bool,
    ) -> None:
        del runtime, deterministic
        self.num_agents = self._num_agents_from_accel_obs(accel_obs, num_envs)

    def _write_accel_action_policy(
        self,
        accel_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        deterministic: bool,
    ) -> None:
        self.num_agents = self._num_agents_from_accel_obs(accel_obs, num_envs)
        self.accel_batch = accel_obs
        accel_batch = self.accel_batch
        if accel_batch is None:
            raise RuntimeError("native accel actor requires current local state.")
        binding = self._policy_actor_binding
        if binding is None:
            raise RuntimeError("native accel policy source was not bound to a CUDA actor ABI.")
        accel_actor_live = (
            native_cuda.actor_accel_live_fused
            if bool(getattr(self.learner.cfg, "structured_native_accel_actor_fused", False))
            else native_cuda.actor_accel_live
        )
        accel_actor_live(
            self._bound_runtime_abi(),
            binding.abi,
            active_idx=int(runtime.main.accel_active_idx),
            deterministic=deterministic,
            rng_step=int(runtime.random.step),
        )
        self.accel_action_batch = None
        self.accel_logprob_batch = None

    def _write_accel_action_policy_module(
        self,
        accel_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        deterministic: bool,
    ) -> None:
        self._write_accel_action_with_module(
            self.learner.actor,
            accel_obs,
            runtime=runtime,
            num_envs=num_envs,
            deterministic=deterministic,
        )

    def _write_accel_action_with_module(
        self,
        actor_module: Any,
        accel_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        deterministic: bool,
    ) -> None:
        self.num_agents = self._num_agents_from_accel_obs(accel_obs, num_envs)
        self.accel_batch = accel_obs
        main = runtime.main
        with torch.no_grad():
            out = actor_module.act_accel(accel_obs, deterministic=deterministic)
            action = out.action.reshape(int(num_envs), int(self.num_agents), -1)
            if not torch.is_tensor(getattr(main, "live_accel_action", None)):
                raise RuntimeError("flat native accel policy requires live_accel_action buffer.")
            main.live_accel_action.copy_(action.to(device=main.live_accel_action.device, dtype=main.live_accel_action.dtype))
            latent_dst = getattr(main, "live_accel_latent_action", None)
            if torch.is_tensor(latent_dst):
                latent = out.latent_action if out.latent_action is not None else out.action
                latent = latent.reshape(int(num_envs), int(self.num_agents), -1)
                latent_dst.copy_(latent.to(device=latent_dst.device, dtype=latent_dst.dtype))
            logprob_dst = getattr(main, "live_accel_old_logprob", None)
            if torch.is_tensor(logprob_dst):
                logprob = out.logprob.reshape(int(num_envs), int(self.num_agents))
                if tuple(logprob_dst.shape) == tuple(logprob.shape):
                    logprob_dst.copy_(logprob.to(device=logprob_dst.device, dtype=logprob_dst.dtype))
                elif tuple(logprob_dst.shape) == (int(num_envs),):
                    logprob_dst.copy_(logprob.sum(dim=1).to(device=logprob_dst.device, dtype=logprob_dst.dtype))
                else:
                    raise RuntimeError(
                        "flat native accel policy cannot write logprob shape "
                        f"{tuple(logprob.shape)} into {tuple(logprob_dst.shape)}."
                    )
        self.accel_action_batch = None
        self.accel_logprob_batch = None

    @staticmethod
    def _runtime_native_abi(runtime: Any) -> native_cuda.NativeCudaRuntimeABI:
        abi = getattr(getattr(runtime, "main", None), "native_cuda_abi", None)
        if not isinstance(abi, native_cuda.NativeCudaRuntimeABI):
            raise RuntimeError("native source producer requires the prebuilt native CUDA runtime ABI.")
        return abi

    def _bound_runtime_abi(self) -> native_cuda.NativeCudaRuntimeABI:
        abi = self._runtime_abi
        if not isinstance(abi, native_cuda.NativeCudaRuntimeABI):
            raise RuntimeError("native source producer was not bound to a CUDA runtime ABI.")
        return abi

    def _teacher_deterministic(self, fallback: bool) -> bool:
        override = self._teacher_deterministic_override
        return bool(fallback if override is None else override)

    def _write_accel_action_queue_aware(
        self,
        accel_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        deterministic: bool,
    ) -> None:
        del deterministic
        self.num_agents = self._num_agents_from_accel_obs(accel_obs, num_envs)
        native_cuda.queue_aware_accel_live(
            self._bound_runtime_abi(),
            active_idx=int(runtime.main.accel_active_idx),
            accel_source_mode=int(runtime.main.accel_actor_source_mode_code),
            sat_source_mode=int(runtime.main.sat_actor_source_mode_code),
            bw_source_mode=int(runtime.main.bw_actor_source_mode_code),
        )

    def _write_accel_action_cluster_center_queue_aware(
        self,
        accel_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        deterministic: bool,
    ) -> None:
        del deterministic
        self.num_agents = self._num_agents_from_accel_obs(accel_obs, num_envs)
        native_cuda.cluster_center_accel_live(
            self._bound_runtime_abi(),
            active_idx=int(runtime.main.accel_active_idx),
            accel_source_mode=int(runtime.main.accel_actor_source_mode_code),
            sat_source_mode=int(runtime.main.sat_actor_source_mode_code),
            bw_source_mode=int(runtime.main.bw_actor_source_mode_code),
        )

    def _write_accel_action_baseline(
        self,
        accel_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        deterministic: bool,
    ) -> None:
        del deterministic
        self.num_agents = self._num_agents_from_accel_obs(accel_obs, num_envs)
        native_cuda.baseline_accel_live(
            self._bound_runtime_abi(),
            active_idx=int(runtime.main.accel_active_idx),
            accel_source_mode=int(runtime.main.accel_actor_source_mode_code),
            sat_source_mode=int(runtime.main.sat_actor_source_mode_code),
            bw_source_mode=int(runtime.main.bw_actor_source_mode_code),
        )

    def _write_accel_action_teacher(
        self,
        accel_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        deterministic: bool,
    ) -> None:
        self.num_agents = self._num_agents_from_accel_obs(accel_obs, num_envs)
        binding = self._teacher_actor_binding
        if binding is None:
            raise RuntimeError("native accel teacher source was not bound to a CUDA actor ABI.")
        accel_actor_live = (
            native_cuda.actor_accel_live_fused
            if bool(getattr(self.learner.cfg, "structured_native_accel_actor_fused", False))
            else native_cuda.actor_accel_live
        )
        accel_actor_live(
            self._bound_runtime_abi(),
            binding.abi,
            active_idx=int(runtime.main.accel_active_idx),
            deterministic=self._teacher_deterministic(deterministic),
            rng_step=int(runtime.random.step),
        )

    def _write_sat_action_zero(
        self,
        sat_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        sat_max_select: int,
        deterministic: bool,
    ) -> None:
        del runtime, sat_max_select, deterministic
        if self.num_agents <= 0:
            self.num_agents = self._num_agents_from_flat_obs_tensor(
                getattr(sat_obs, "ego_features", None),
                num_envs,
            )

    def _write_sat_action_policy(
        self,
        sat_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        sat_max_select: int,
        deterministic: bool,
    ) -> None:
        del sat_max_select
        self.sat_batch = sat_obs
        sat_batch = self.sat_batch
        if sat_batch is None:
            raise RuntimeError("native SAT actor requires current local state.")
        binding = self._policy_actor_binding
        if binding is None:
            raise RuntimeError("native SAT policy source was not bound to a CUDA actor ABI.")
        sat_actor_live = (
            native_cuda.actor_sat_live_fused
            if bool(getattr(self.learner.cfg, "structured_native_sat_actor_fused", False))
            else native_cuda.actor_sat_live
        )
        sat_actor_live(
            self._bound_runtime_abi(),
            binding.abi,
            deterministic=deterministic,
            rng_step=int(runtime.random.step),
        )
        self.sat_action_batch = None
        self.sat_logprob_batch = None

    def _write_sat_action_policy_module(
        self,
        sat_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        sat_max_select: int,
        deterministic: bool,
    ) -> None:
        self._write_sat_action_with_module(
            self.learner.actor,
            sat_obs,
            runtime=runtime,
            num_envs=num_envs,
            sat_max_select=sat_max_select,
            deterministic=deterministic,
        )

    def _write_sat_action_with_module(
        self,
        actor_module: Any,
        sat_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        sat_max_select: int,
        deterministic: bool,
    ) -> None:
        del sat_max_select
        if self.num_agents <= 0:
            self.num_agents = self._num_agents_from_flat_obs_tensor(
                getattr(sat_obs, "ego_features", None),
                num_envs,
            )
        self.sat_batch = sat_obs
        main = runtime.main
        with torch.no_grad():
            out = actor_module.act_sat(sat_obs, deterministic=deterministic)
            sat_policy = getattr(actor_module, "sat_subset_policy", None)
            if sat_policy is not None and hasattr(sat_policy, "_compute_logits") and hasattr(sat_policy, "_legal_subset_mask"):
                logits = sat_policy._compute_logits(sat_obs)
                legal_mask = sat_policy._legal_subset_mask(sat_obs, logits)
                chosen = out.subset_index.reshape(-1).to(device=logits.device, dtype=torch.long)
                row_ids = torch.arange(int(chosen.shape[0]), device=logits.device, dtype=torch.long)
                chosen_in_range = (chosen >= 0) & (chosen < int(legal_mask.shape[1]))
                chosen_safe = chosen.clamp(min=0, max=max(int(legal_mask.shape[1]) - 1, 0))
                legal_count = legal_mask.sum(dim=-1)
                illegal_with_fallback = (legal_count > 0) & (~chosen_in_range | ~legal_mask[row_ids, chosen_safe])
                if bool(illegal_with_fallback.any().detach().cpu().item()):
                    fallback = legal_mask.to(dtype=torch.long).argmax(dim=-1)
                    fixed_chosen = torch.where(illegal_with_fallback, fallback, chosen_safe)
                    out = sat_policy.evaluate_actions(sat_obs, fixed_chosen, compute_entropy=True)
            subset_index = out.subset_index.reshape(int(num_envs), int(self.num_agents))
            subset_dst = getattr(main, "live_sat_subset_index", None)
            if not torch.is_tensor(subset_dst):
                raise RuntimeError("flat native SAT policy requires live_sat_subset_index buffer.")
            subset_dst.copy_(subset_index.to(device=subset_dst.device, dtype=subset_dst.dtype))

            action_dst = getattr(main, "live_sat_action_indices", None)
            if torch.is_tensor(action_dst):
                selected = out.selected_sat_indices.reshape(int(num_envs), int(self.num_agents), -1)
                selected_t = selected.to(device=action_dst.device, dtype=action_dst.dtype)
                if tuple(selected_t.shape) == tuple(action_dst.shape):
                    action_dst.copy_(selected_t)
                else:
                    action_dst.fill_(-1)
                    keep = min(int(selected_t.shape[-1]), int(action_dst.shape[-1]))
                    if keep > 0:
                        action_dst[..., :keep].copy_(selected_t[..., :keep])

            logprob = out.logprob.reshape(int(num_envs), int(self.num_agents))
            logprob_dst = getattr(main, "live_sat_old_logprobs_per_agent", None)
            if torch.is_tensor(logprob_dst):
                if tuple(logprob_dst.shape) != tuple(logprob.shape):
                    raise RuntimeError(
                        "flat native SAT policy cannot write logprob shape "
                        f"{tuple(logprob.shape)} into {tuple(logprob_dst.shape)}."
                    )
                logprob_dst.copy_(logprob.to(device=logprob_dst.device, dtype=logprob_dst.dtype))
            entropy_dst = getattr(main, "live_sat_entropy_per_agent", None)
            if torch.is_tensor(entropy_dst):
                entropy = out.entropy.reshape(int(num_envs), int(self.num_agents))
                if tuple(entropy_dst.shape) == tuple(entropy.shape):
                    entropy_dst.copy_(entropy.to(device=entropy_dst.device, dtype=entropy_dst.dtype))
                else:
                    entropy_dst.zero_()
        self.sat_action_batch = None
        self.sat_logprob_batch = None

    def _write_sat_action_queue_aware(
        self,
        sat_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        sat_max_select: int,
        deterministic: bool,
    ) -> None:
        del sat_obs, num_envs, sat_max_select, deterministic
        native_cuda.queue_aware_sat_live(
            self._bound_runtime_abi(),
            accel_source_mode=int(runtime.main.accel_actor_source_mode_code),
            sat_source_mode=int(runtime.main.sat_actor_source_mode_code),
            bw_source_mode=int(runtime.main.bw_actor_source_mode_code),
        )

    def _write_sat_action_baseline(
        self,
        sat_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        sat_max_select: int,
        deterministic: bool,
    ) -> None:
        del sat_max_select, deterministic
        if self.num_agents <= 0:
            self.num_agents = self._num_agents_from_flat_obs_tensor(
                getattr(sat_obs, "ego_features", None),
                num_envs,
            )
        native_cuda.baseline_sat_live(
            self._bound_runtime_abi(),
            accel_source_mode=int(runtime.main.accel_actor_source_mode_code),
            sat_source_mode=int(runtime.main.sat_actor_source_mode_code),
            bw_source_mode=int(runtime.main.bw_actor_source_mode_code),
        )

    def _write_sat_action_teacher(
        self,
        sat_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        sat_max_select: int,
        deterministic: bool,
    ) -> None:
        del sat_max_select
        if self.num_agents <= 0:
            self.num_agents = self._num_agents_from_flat_obs_tensor(
                getattr(sat_obs, "ego_features", None),
                num_envs,
            )
        binding = self._teacher_actor_binding
        if binding is None:
            raise RuntimeError("native SAT teacher source was not bound to a CUDA actor ABI.")
        sat_actor_live = (
            native_cuda.actor_sat_live_fused
            if bool(getattr(self.learner.cfg, "structured_native_sat_actor_fused", False))
            else native_cuda.actor_sat_live
        )
        sat_actor_live(
            self._bound_runtime_abi(),
            binding.abi,
            deterministic=self._teacher_deterministic(deterministic),
            rng_step=int(runtime.random.step),
        )

    def _write_bw_action_zero(
        self,
        bw_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        deterministic: bool,
    ) -> None:
        del deterministic
        if self.num_agents <= 0:
            self.num_agents = self._num_agents_from_flat_obs_tensor(
                getattr(bw_obs, "ego_features", None),
                num_envs,
            )
        main = runtime.main
        for name in (
            "live_bw_action",
            "live_bw_ref_action",
            "live_bw_flow_proxy_override_action",
            "live_bw_old_logprob",
            "live_bw_old_logprobs_per_agent",
            "live_bw_entropy_per_agent",
            "live_bw_logprob_raw_per_agent",
            "live_bw_entropy_raw_per_agent",
            "live_bw_tau",
            "live_bw_kappa",
        ):
            value = getattr(main, name, None)
            if torch.is_tensor(value):
                value.zero_()
        for name in ("live_bw_valid_count", "live_bw_latent_count"):
            value = getattr(main, name, None)
            if torch.is_tensor(value):
                value.zero_()

    def _write_bw_action_with_module(
        self,
        actor_module: Any,
        bw_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        deterministic: bool,
    ) -> None:
        self.bw_batch = bw_obs
        bw_batch = self.bw_batch
        if bw_batch is None:
            raise RuntimeError("BW actor requires current local state.")
        if self.num_agents <= 0:
            self.num_agents = self._num_agents_from_flat_obs_tensor(
                getattr(bw_batch, "ego_features", None),
                num_envs,
            )
        main = runtime.main
        with torch.no_grad():
            actor_module.act_bw_into(
                bw_batch,
                action_out=main.live_bw_action,
                ref_action_out=main.live_bw_ref_action,
                logprob_out=main.live_bw_old_logprob,
                logprob_per_agent_out=main.live_bw_old_logprobs_per_agent,
                entropy_per_agent_out=getattr(main, "live_bw_entropy_per_agent", None),
                logprob_raw_per_agent_out=getattr(main, "live_bw_logprob_raw_per_agent", None),
                entropy_raw_per_agent_out=getattr(main, "live_bw_entropy_raw_per_agent", None),
                tau_out=getattr(main, "live_bw_tau", None),
                kappa_out=getattr(main, "live_bw_kappa", None),
                valid_count_out=getattr(main, "live_bw_valid_count", None),
                latent_count_out=getattr(main, "live_bw_latent_count", None),
                deterministic=deterministic,
                num_envs=int(num_envs),
                num_agents=int(self.num_agents),
            )
        flow_proxy_action = getattr(main, "live_bw_flow_proxy_override_action", None)
        if torch.is_tensor(flow_proxy_action):
            flow_proxy_action.copy_(main.live_bw_action.to(device=flow_proxy_action.device, dtype=flow_proxy_action.dtype))
        self.bw_action_batch = None
        self.bw_ref_action_batch = None
        self.bw_logprob_batch = None
        self.bw_logprobs_per_agent_batch = None

    def _write_bw_action_policy(
        self,
        bw_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        deterministic: bool,
    ) -> None:
        self.bw_batch = bw_obs
        bw_batch = self.bw_batch
        if bw_batch is None:
            raise RuntimeError("native BW actor requires current local state.")
        if self.num_agents <= 0:
            self.num_agents = self._num_agents_from_flat_obs_tensor(
                getattr(bw_batch, "ego_features", None),
                num_envs,
            )
        binding = self._policy_actor_binding
        if binding is None:
            raise RuntimeError("native BW policy source was not bound to a CUDA actor ABI.")
        for name in (
            "live_bw_old_logprob",
            "live_bw_old_logprobs_per_agent",
            "live_bw_entropy_per_agent",
            "live_bw_logprob_raw_per_agent",
            "live_bw_entropy_raw_per_agent",
            "live_bw_tau",
            "live_bw_kappa",
        ):
            value = getattr(runtime.main, name, None)
            if torch.is_tensor(value):
                value.zero_()
        for name in ("live_bw_valid_count", "live_bw_latent_count"):
            value = getattr(runtime.main, name, None)
            if torch.is_tensor(value):
                value.zero_()
        macro_interval = max(int(getattr(self.learner.cfg, "access_bw_decision_interval", 1) or 1), 1)
        if macro_interval > 1:
            # Macro-BW must keep the same start-step distribution as the normal
            # fused actor.  The older hand-written BW kernel is not parity-clean
            # enough for PPO old-logprob checks, so use the fused path and let
            # its write kernel restore continuation rows from history.
            native_cuda.actor_bw_live_fused(
                self._bound_runtime_abi(),
                binding.abi,
                deterministic=deterministic,
                rng_step=int(runtime.random.step),
                history_slot=int(getattr(runtime.history, "cursor", 0)),
            )
        else:
            bw_actor_live = (
                native_cuda.actor_bw_live_fused
                if bool(getattr(self.learner.cfg, "structured_native_bw_actor_fused", False))
                else native_cuda.actor_bw_live
            )
            bw_actor_live(
                self._bound_runtime_abi(),
                binding.abi,
                deterministic=deterministic,
                rng_step=int(runtime.random.step),
            )
        self.bw_action_batch = None
        self.bw_ref_action_batch = None
        self.bw_logprob_batch = None

    def _write_bw_action_policy_module(
        self,
        bw_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        deterministic: bool,
    ) -> None:
        self._write_bw_action_with_module(
            self.learner.actor,
            bw_obs,
            runtime=runtime,
            num_envs=num_envs,
            deterministic=deterministic,
        )

    def _write_bw_action_policy_single_uav_queue_aware(
        self,
        bw_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        deterministic: bool,
    ) -> None:
        self.bw_batch = bw_obs
        bw_batch = self.bw_batch
        if bw_batch is None:
            raise RuntimeError("native BW hybrid policy source requires current local state.")
        if self.num_agents <= 0:
            self.num_agents = self._num_agents_from_flat_obs_tensor(
                getattr(bw_batch, "ego_features", None),
                num_envs,
            )
        target_uav = self._single_uav_policy_uav_id(self.num_agents)

        main = runtime.main
        native_cuda.queue_aware_bw_live(
            self._bound_runtime_abi(),
            accel_source_mode=int(main.accel_actor_source_mode_code),
            sat_source_mode=int(main.sat_actor_source_mode_code),
            bw_source_mode=int(main.bw_actor_source_mode_code),
        )
        queue_action = main.live_bw_action.detach().clone()
        queue_ref_action = main.live_bw_ref_action.detach().clone()
        queue_flow_proxy_action = main.live_bw_flow_proxy_override_action.detach().clone()

        binding = self._policy_actor_binding
        if binding is None:
            raise RuntimeError("native BW hybrid policy source was not bound to a CUDA actor ABI.")
        for name in (
            "live_bw_old_logprob",
            "live_bw_old_logprobs_per_agent",
            "live_bw_entropy_per_agent",
            "live_bw_logprob_raw_per_agent",
            "live_bw_entropy_raw_per_agent",
            "live_bw_tau",
            "live_bw_kappa",
        ):
            value = getattr(main, name, None)
            if torch.is_tensor(value):
                value.zero_()
        for name in ("live_bw_valid_count", "live_bw_latent_count"):
            value = getattr(main, name, None)
            if torch.is_tensor(value):
                value.zero_()
        bw_actor_live = (
            native_cuda.actor_bw_live_fused
            if bool(getattr(self.learner.cfg, "structured_native_bw_actor_fused", False))
            else native_cuda.actor_bw_live
        )
        bw_actor_live(
            self._bound_runtime_abi(),
            binding.abi,
            deterministic=deterministic,
            rng_step=int(runtime.random.step),
        )

        keep_fixed = torch.ones((int(self.num_agents),), dtype=torch.bool, device=main.live_bw_action.device)
        keep_fixed[int(target_uav)] = False
        fixed_idx = torch.nonzero(keep_fixed, as_tuple=False).reshape(-1).to(dtype=torch.long)
        if int(fixed_idx.numel()) > 0:
            main.live_bw_action.index_copy_(1, fixed_idx, queue_action.index_select(1, fixed_idx))
            main.live_bw_ref_action.index_copy_(1, fixed_idx, queue_ref_action.index_select(1, fixed_idx))
            main.live_bw_flow_proxy_override_action.index_copy_(
                1,
                fixed_idx,
                queue_flow_proxy_action.index_select(1, fixed_idx),
            )
            main.live_bw_old_logprobs_per_agent.index_fill_(1, fixed_idx, 0.0)
        if torch.is_tensor(main.live_bw_old_logprob):
            main.live_bw_old_logprob.copy_(main.live_bw_old_logprobs_per_agent[:, int(target_uav)])
        self.bw_action_batch = None
        self.bw_ref_action_batch = None
        self.bw_logprob_batch = None

    def _write_bw_action_queue_aware(
        self,
        bw_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        deterministic: bool,
    ) -> None:
        del deterministic
        if self.num_agents <= 0:
            self.num_agents = self._num_agents_from_flat_obs_tensor(
                getattr(bw_obs, "ego_features", None),
                num_envs,
            )
        native_cuda.queue_aware_bw_live(
            self._bound_runtime_abi(),
            accel_source_mode=int(runtime.main.accel_actor_source_mode_code),
            sat_source_mode=int(runtime.main.sat_actor_source_mode_code),
            bw_source_mode=int(runtime.main.bw_actor_source_mode_code),
        )

    def _write_bw_action_baseline(
        self,
        bw_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        deterministic: bool,
    ) -> None:
        del deterministic
        if self.num_agents <= 0:
            self.num_agents = self._num_agents_from_flat_obs_tensor(
                getattr(bw_obs, "ego_features", None),
                num_envs,
            )
        native_cuda.baseline_bw_live(
            self._bound_runtime_abi(),
            accel_source_mode=int(runtime.main.accel_actor_source_mode_code),
            sat_source_mode=int(runtime.main.sat_actor_source_mode_code),
            bw_source_mode=int(runtime.main.bw_actor_source_mode_code),
        )

    def _write_bw_action_teacher(
        self,
        bw_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        deterministic: bool,
    ) -> None:
        if self.num_agents <= 0:
            self.num_agents = self._num_agents_from_flat_obs_tensor(
                getattr(bw_obs, "ego_features", None),
                num_envs,
            )
        binding = self._teacher_actor_binding
        if binding is None:
            raise RuntimeError("native BW teacher source was not bound to a CUDA actor ABI.")
        macro_interval = max(int(getattr(self.learner.cfg, "access_bw_decision_interval", 1) or 1), 1)
        if macro_interval > 1:
            native_cuda.actor_bw_live_fused(
                self._bound_runtime_abi(),
                binding.abi,
                deterministic=self._teacher_deterministic(deterministic),
                rng_step=int(runtime.random.step),
                history_slot=int(getattr(runtime.history, "cursor", 0)),
            )
        else:
            bw_actor_live = (
                native_cuda.actor_bw_live_fused
                if bool(getattr(self.learner.cfg, "structured_native_bw_actor_fused", False))
                else native_cuda.actor_bw_live
            )
            bw_actor_live(
                self._bound_runtime_abi(),
                binding.abi,
                deterministic=self._teacher_deterministic(deterministic),
                rng_step=int(runtime.random.step),
            )

    def write_accel_action(
        self,
        accel_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        deterministic: bool,
    ) -> None:
        del accel_obs, runtime, num_envs, deterministic
        raise RuntimeError("native MAPPO actor bridge must be bound with bind_source_modes() before accel action.")

    def write_sat_action(
        self,
        sat_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        sat_max_select: int,
        deterministic: bool,
    ) -> None:
        del sat_obs, runtime, num_envs, sat_max_select, deterministic
        raise RuntimeError("native MAPPO actor bridge must be bound with bind_source_modes() before SAT action.")

    def write_bw_action(
        self,
        bw_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        deterministic: bool,
    ) -> None:
        del bw_obs, runtime, num_envs, deterministic
        raise RuntimeError("native MAPPO actor bridge must be bound with bind_source_modes() before BW action.")


class _BwCleanFirstActionOverrideBridge:
    """Actor bridge that replaces only the first BW action of a branch rollout."""

    def __init__(self, *, base_bridge: _StructuredMAPPOGpuActorBridge, first_bw_actions: torch.Tensor) -> None:
        self.base_bridge = base_bridge
        self.first_bw_actions = first_bw_actions.detach()
        self.step_index = 0

    def bind_source_modes(self, runtime: Any) -> None:
        self.base_bridge.bind_source_modes(runtime)

    def begin_horizon(self, *, horizon: int, runtime: Any, deterministic: bool) -> None:
        del horizon
        self.bind_source_modes(runtime)
        self.base_bridge.begin_horizon(horizon=0, runtime=runtime, deterministic=deterministic)

    def end_horizon(self, *, results: Sequence[Any], runtime: Any) -> None:
        del results, runtime

    def __call__(self, *, step_index: int, runtime: Any):
        del runtime
        self.step_index = int(step_index)
        return self

    def begin_step(self, *, deterministic: bool) -> None:
        self.base_bridge.begin_step(deterministic=deterministic)

    def write_accel_action(
        self,
        accel_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        deterministic: bool,
    ) -> None:
        self.base_bridge.write_accel_action(
            accel_obs,
            runtime=runtime,
            num_envs=num_envs,
            deterministic=deterministic,
        )

    def write_sat_action(
        self,
        sat_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        sat_max_select: int,
        deterministic: bool,
    ) -> None:
        self.base_bridge.write_sat_action(
            sat_obs,
            runtime=runtime,
            num_envs=num_envs,
            sat_max_select=sat_max_select,
            deterministic=deterministic,
        )

    def write_bw_action(
        self,
        bw_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        deterministic: bool,
    ) -> None:
        if int(self.step_index) != 0:
            self.base_bridge.write_bw_action(
                bw_obs,
                runtime=runtime,
                num_envs=num_envs,
                deterministic=True,
            )
            return
        action_dst = getattr(runtime.main, "live_bw_action", None)
        if not torch.is_tensor(action_dst):
            raise RuntimeError("BW clean branch replay requires live_bw_action buffer.")
        action_t = self.first_bw_actions.to(device=action_dst.device, dtype=action_dst.dtype)
        action_t = action_t.reshape_as(action_dst)
        action_dst.copy_(action_t)
        ref_dst = getattr(runtime.main, "live_bw_ref_action", None)
        if torch.is_tensor(ref_dst):
            ref_dst.copy_(action_t.to(device=ref_dst.device, dtype=ref_dst.dtype))
        for name in (
            "live_bw_old_logprob",
            "live_bw_old_logprobs_per_agent",
        ):
            value = getattr(runtime.main, name, None)
            if torch.is_tensor(value):
                value.zero_()


class _VsRefFirstActionOverrideBridge:
    """Replace only the first action for one stage; follow policy stays deterministic."""

    def __init__(
        self,
        *,
        base_bridge: _StructuredMAPPOGpuActorBridge,
        stage_id: int,
        first_actions: torch.Tensor,
    ) -> None:
        self.base_bridge = base_bridge
        self.stage_id = int(stage_id)
        self.first_actions = first_actions.detach()
        self.step_index = 0

    def bind_source_modes(self, runtime: Any) -> None:
        self.base_bridge.source_by_stage = {0: "policy", 1: "policy", 2: "policy"}
        self.base_bridge.bind_source_modes(runtime)

    def begin_horizon(self, *, horizon: int, runtime: Any, deterministic: bool) -> None:
        del horizon
        self.bind_source_modes(runtime)
        self.base_bridge.begin_horizon(horizon=0, runtime=runtime, deterministic=deterministic)

    def end_horizon(self, *, results: Sequence[Any], runtime: Any) -> None:
        del results, runtime

    def __call__(self, *, step_index: int, runtime: Any):
        del runtime
        self.step_index = int(step_index)
        return self

    def begin_step(self, *, deterministic: bool) -> None:
        self.base_bridge.begin_step(deterministic=deterministic)

    def write_accel_action(
        self,
        accel_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        deterministic: bool,
    ) -> None:
        if int(self.step_index) != 0 or int(self.stage_id) != 0:
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
            raise RuntimeError("sample-vs-ref accel branch replay requires live_accel_action buffer.")
        action_t = self.first_actions.to(device=action_dst.device, dtype=action_dst.dtype).reshape_as(action_dst)
        action_dst.copy_(action_t)
        latent_dst = getattr(runtime.main, "live_accel_latent_action", None)
        if torch.is_tensor(latent_dst):
            latent_dst.zero_()
        logprob_dst = getattr(runtime.main, "live_accel_old_logprob", None)
        if torch.is_tensor(logprob_dst):
            logprob_dst.zero_()

    def write_sat_action(
        self,
        sat_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        sat_max_select: int,
        deterministic: bool,
    ) -> None:
        if int(self.step_index) != 0 or int(self.stage_id) != 1:
            self.base_bridge.write_sat_action(
                sat_obs,
                runtime=runtime,
                num_envs=num_envs,
                sat_max_select=sat_max_select,
                deterministic=True,
            )
            return
        del sat_obs, sat_max_select, deterministic
        action_dst = getattr(runtime.main, "live_sat_subset_index", None)
        if not torch.is_tensor(action_dst):
            raise RuntimeError("sample-vs-ref SAT branch replay requires live_sat_subset_index buffer.")
        action_t = self.first_actions.to(device=action_dst.device, dtype=action_dst.dtype).reshape_as(action_dst)
        action_dst.copy_(action_t)
        for name in ("live_sat_old_logprobs_per_agent", "live_sat_entropy_per_agent"):
            value = getattr(runtime.main, name, None)
            if torch.is_tensor(value):
                value.zero_()

    def write_bw_action(
        self,
        bw_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        deterministic: bool,
    ) -> None:
        if int(self.step_index) != 0 or int(self.stage_id) != 2:
            self.base_bridge.write_bw_action(
                bw_obs,
                runtime=runtime,
                num_envs=num_envs,
                deterministic=True,
            )
            return
        del bw_obs, deterministic
        action_dst = getattr(runtime.main, "live_bw_action", None)
        if not torch.is_tensor(action_dst):
            raise RuntimeError("sample-vs-ref BW branch replay requires live_bw_action buffer.")
        action_t = self.first_actions.to(device=action_dst.device, dtype=action_dst.dtype).reshape_as(action_dst)
        action_dst.copy_(action_t)
        ref_dst = getattr(runtime.main, "live_bw_ref_action", None)
        if torch.is_tensor(ref_dst):
            ref_dst.copy_(action_t.to(device=ref_dst.device, dtype=ref_dst.dtype))
        flow_proxy_action = getattr(runtime.main, "live_bw_flow_proxy_override_action", None)
        if torch.is_tensor(flow_proxy_action):
            flow_proxy_action.copy_(action_t.to(device=flow_proxy_action.device, dtype=flow_proxy_action.dtype))
        for name in (
            "live_bw_old_logprob",
            "live_bw_old_logprobs_per_agent",
            "live_bw_entropy_per_agent",
            "live_bw_logprob_raw_per_agent",
            "live_bw_entropy_raw_per_agent",
            "live_bw_tau",
            "live_bw_kappa",
        ):
            value = getattr(runtime.main, name, None)
            if torch.is_tensor(value):
                value.zero_()
        for name in ("live_bw_valid_count", "live_bw_latent_count"):
            value = getattr(runtime.main, name, None)
            if torch.is_tensor(value):
                value.zero_()


class StructuredGpuRolloutProgram:
    """Single-entry native GPU rollout step program.

    The program owns the fixed ordering between actor decisions and env kernels.
    MAPPO supplies actor/buffer policy objects; it no longer directly strings
    together the native env stage API.
    """

    def __init__(self, *, learner: "StructuredMAPPO", drivers: Any, tensor_device: torch.device) -> None:
        self.learner = learner
        self.drivers = drivers
        self.executor = StructuredNativeRolloutExecutor(drivers, tensor_device=tensor_device, cfg=learner.cfg)
        self.runtime = self.executor.runtime
        self.program = self.executor.program
        self._actor_bridge = _StructuredMAPPOGpuActorBridge(learner)

    def _actor_source_key(self) -> tuple[str, str, str]:
        sources = self.learner.exec_source_by_stage
        return (
            str(sources.get(0, "policy")),
            str(sources.get(1, "policy")),
            str(sources.get(2, "policy")),
        )

    def _actor_bridge_for_current_sources(self) -> _StructuredMAPPOGpuActorBridge:
        bridge = getattr(self, "_actor_bridge", None)
        source_key = self._actor_source_key()
        if (
            bridge is None
            or not isinstance(bridge, _StructuredMAPPOGpuActorBridge)
            or bridge.source_by_stage.get(0) != source_key[0]
            or bridge.source_by_stage.get(1) != source_key[1]
            or bridge.source_by_stage.get(2) != source_key[2]
        ):
            bridge = _StructuredMAPPOGpuActorBridge(self.learner)
            self._actor_bridge = bridge
        return bridge

    @staticmethod
    def _source_mode_code(source: object) -> int:
        return _native_exec_source_mode_code(source)

    def _bind_actor_source_modes(self, runtime: Any) -> None:
        main = runtime.main
        main.accel_actor_source_mode_code = self._source_mode_code(self.learner.exec_source_by_stage[0])
        main.sat_actor_source_mode_code = self._source_mode_code(self.learner.exec_source_by_stage[1])
        main.bw_actor_source_mode_code = self._source_mode_code(self.learner.exec_source_by_stage[2])

    def collect_horizon(
        self,
        *,
        horizon: int,
        buffer: StructuredRolloutBuffer | None,
        deterministic: bool = False,
    ) -> list[StructuredBatchStepResult]:
        steps = max(int(horizon), 0)
        learner = self.learner
        submit_to_ppo = buffer is not None
        structured_env_tensor_device = learner._structured_env_tensor_device()
        if structured_env_tensor_device is None:
            raise RuntimeError("native tensor rollout executor requires a structured env tensor device.")
        num_envs = len(self.drivers)
        if num_envs <= 0 or steps <= 0:
            return []
        program = self.program
        runtime = self.runtime
        if not submit_to_ppo:
            sub_batch_program_factory = getattr(self.drivers, "native_sub_batch_rollout_program", None)
            if not callable(sub_batch_program_factory):
                raise RuntimeError("native non-record rollout requires native_sub_batch_rollout_program().")
            program = sub_batch_program_factory(
                capacity=steps,
                selected_indices=tuple(range(num_envs)),
            )
            runtime = program.runtime
        learner._ensure_native_actor_cuda_bindings(sync=False)
        self._bind_actor_source_modes(runtime)

        bridge = self._actor_bridge_for_current_sources()
        bridge.bind_source_modes(runtime)
        ensure_reset_tape_capacity = getattr(self.drivers, "ensure_native_rollout_reset_tape_capacity", None)
        reset_tape_chunk_rows = max(
            min(
                steps,
                int(getattr(learner.cfg, "structured_native_reset_tape_chunk_rows", 8) or 8),
            ),
            1,
        )

        def _after_native_rollout_step(**callback_kwargs: Any) -> None:
            if not callable(ensure_reset_tape_capacity):
                return
            step_index = int(callback_kwargs.get("step_index", -1))
            if step_index + 1 >= steps:
                return
            if (step_index + 1) % int(reset_tape_chunk_rows) != 0:
                return
            extended = bool(ensure_reset_tape_capacity(chunk_rows=int(reset_tape_chunk_rows)))
            if extended:
                callback_runtime = callback_kwargs.get("runtime", runtime)
                callback_bridge = callback_kwargs.get("actor_bridge", bridge)
                if hasattr(callback_bridge, "bind_source_modes"):
                    callback_bridge.bind_source_modes(callback_runtime)

        with torch.no_grad():
            results = program.replay_horizon(
                actor_bridge_factory=bridge,
                horizon=steps,
                deterministic=deterministic,
                step_callback=_after_native_rollout_step if callable(ensure_reset_tape_capacity) else None,
            )
        if any(not isinstance(result, StructuredBatchStepResult) for result in results):
            raise RuntimeError("native GPU rollout program must return StructuredBatchStepResult for every step.")
        if submit_to_ppo:
            if buffer is None:
                raise RuntimeError("native rollout recording requires rollout buffer.")
            buffer.add_native_rollout_training_view(
                history=runtime.history,
                num_steps=len(results),
                num_envs=num_envs,
                access_bw_decision_interval=int(getattr(getattr(learner, "cfg", None), "access_bw_decision_interval", 1) or 1),
                sat_decision_interval=int(getattr(getattr(learner, "cfg", None), "sat_decision_interval", 1) or 1),
                gamma_env=float(getattr(learner, "gamma", 1.0)),
            )
        return results

    def collect_step(self, *, buffer: StructuredRolloutBuffer | None, deterministic: bool = False):
        results = self.collect_horizon(horizon=1, buffer=buffer, deterministic=deterministic)
        return results[0] if results else []

    def _run_fixed_step_program(self, *, buffer: StructuredRolloutBuffer | None, deterministic: bool = False):
        results = self.collect_horizon(horizon=1, buffer=buffer, deterministic=deterministic)
        return results[0] if results else []

class StructuredMAPPO:
    def __init__(
        self,
        actor,
        critic,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        num_mini_batch: int = 1,
        danger_imitation_enabled: bool = False,
        danger_imitation_coef: float = 0.0,
        actor_optimizer=None,
        actor_stage_optimizers: dict[int, Any] | None = None,
        critic_optimizer=None,
        device: torch.device | str | None = None,
        target_mode: str = "stage_chained",
        joint_stage_updates: bool = False,
        cfg: Any | None = None,
        train_accel: bool = True,
        train_sat: bool = True,
        train_bw: bool = True,
        exec_accel_source: str = "policy",
        exec_sat_source: str = "policy",
        exec_bw_source: str = "policy",
        teacher_actor: Any | None = None,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.teacher_actor = teacher_actor
        self.cfg = cfg
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_ratio = float(clip_ratio)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.ppo_epochs = int(ppo_epochs)
        self.num_mini_batch = int(num_mini_batch)
        critic_epochs_cfg = None if cfg is None else getattr(cfg, "critic_epochs", None)
        self.critic_epochs = (
            self.ppo_epochs
            if critic_epochs_cfg is None
            else max(int(critic_epochs_cfg), self.ppo_epochs)
        )
        self.critic_warmup_before_actor_epochs = (
            0
            if cfg is None
            else max(int(getattr(cfg, "critic_warmup_before_actor_epochs", 0) or 0), 0)
        )
        self.critic_warmup_recompute_advantages = (
            cfg is not None
            and bool(getattr(cfg, "critic_warmup_recompute_advantages", False))
            and self.critic_warmup_before_actor_epochs > 0
        )
        self.critic_warmup_recompute_mode = (
            "advantage_only"
            if cfg is None
            else str(getattr(cfg, "critic_warmup_recompute_mode", "advantage_only") or "advantage_only").strip().lower()
        )
        if self.critic_warmup_recompute_mode == "full_gae":
            self.critic_warmup_recompute_mode = "gae"
        if self.critic_warmup_recompute_mode not in {"advantage_only", "gae"}:
            raise ValueError("critic_warmup_recompute_mode must be one of {'advantage_only', 'gae'}.")
        self.actor_update_microbatch_size = (
            0
            if cfg is None
            else max(int(getattr(cfg, "actor_update_microbatch_size", 1024) or 0), 0)
        )
        self.rollout_value_eval_microbatch_size = (
            0
            if cfg is None
            else max(
                int(
                    getattr(
                        cfg,
                        "rollout_value_eval_microbatch_size",
                        getattr(cfg, "actor_update_microbatch_size", 1024),
                    )
                    or 0
                ),
                0,
            )
        )
        self.critic_loss_target_standardize = bool(
            False if cfg is None else getattr(cfg, "critic_loss_target_standardize", False)
        )
        self.critic_loss_running_standardize = bool(
            False if cfg is None else getattr(cfg, "critic_loss_running_standardize", False)
        )
        self.critic_loss_running_standardize_decay = (
            0.99
            if cfg is None
            else float(np.clip(float(getattr(cfg, "critic_loss_running_standardize_decay", 0.99) or 0.99), 0.0, 0.9999))
        )
        self.critic_replay_bank_enabled = bool(
            False if cfg is None else getattr(cfg, "critic_replay_bank_enabled", False)
        )
        self.critic_replay_bank_capacity = (
            0 if cfg is None else max(int(getattr(cfg, "critic_replay_bank_capacity", 0) or 0), 0)
        )
        self.critic_value_mode = (
            "relational"
            if cfg is None
            else str(getattr(cfg, "critic_value_mode", "relational") or "relational").strip().lower()
        )
        self.critic_compile_enabled = bool(cfg is not None and getattr(cfg, "critic_compile_enabled", True))
        self.critic_compile_fullgraph = bool(cfg is not None and getattr(cfg, "critic_compile_fullgraph", True))
        self._compiled_stage_value_fns: dict[int, Any] = {}
        if self.critic_value_mode in {"global-linear", "linear_global", "linear-global"}:
            self.critic_value_mode = "global_linear"
        if self.critic_value_mode in {"global", "global_scalars", "global-scalar", "global-only"}:
            self.critic_value_mode = "global_only"
        self.critic_global_linear_fit_ridge = (
            1.0e-6 if cfg is None else max(float(getattr(cfg, "critic_global_linear_fit_ridge", 1.0e-6) or 0.0), 0.0)
        )
        self.critic_global_linear_fit_decay = (
            0.90
            if cfg is None
            else float(np.clip(float(getattr(cfg, "critic_global_linear_fit_decay", 0.90) or 0.0), 0.0, 0.999999))
        )
        self.critic_global_linear_fit_recompute_advantages = bool(
            cfg is not None and bool(getattr(cfg, "critic_global_linear_fit_recompute_advantages", True))
        )
        self.critic_popart_enabled = bool(
            cfg is not None
            and bool(getattr(cfg, "critic_popart_enabled", False))
            and bool(getattr(self.critic, "popart_enabled", False))
        )
        self._critic_return_stats: dict[int, dict[str, float]] = {
            sid: {"initialized": 0.0, "mean": 0.0, "var": 1.0}
            for sid in (0, 1, 2)
        }
        self._critic_replay_bank: dict[int, list[tuple[Any, float]]] = {sid: [] for sid in (0, 1, 2)}
        self._actor_advantage_override_by_stage: dict[int, np.ndarray] = {}
        self.last_update_actor_advantages_np: np.ndarray | None = None
        self.last_update_actor_raw_advantages_np: np.ndarray | None = None
        self.last_update_actor_values_np: np.ndarray | None = None
        self.last_update_returns_np: np.ndarray | None = None
        self._bw_delta_teacher_ready_streak: int = 0
        self._bw_delta_dense_teacher_disabled: bool = False
        self._bw_delta_last_teacher_corr: float = 0.0
        self._skip_actor_update_stage_ids_once: set[int] = set()
        self._bw_branch_gate_snr_once: float = 0.0
        self._bw_branch_gate_triggered_once: float = 0.0
        self.current_update_index: int = 0
        self.danger_imitation_enabled = bool(danger_imitation_enabled) and float(danger_imitation_coef) > 0.0
        self.danger_imitation_coef = max(float(danger_imitation_coef), 0.0)
        self.actor_optimizer = actor_optimizer
        self.actor_stage_optimizers = {
            int(stage_id): optimizer
            for stage_id, optimizer in dict(actor_stage_optimizers or {}).items()
            if optimizer is not None
        }
        self.critic_optimizer = critic_optimizer
        target_mode_l = str(target_mode).strip().lower()
        if target_mode_l not in {"stage_chained", "step_level"}:
            raise ValueError(f"Unsupported structured target mode: {target_mode}")
        self.target_mode = target_mode_l
        self.joint_stage_updates = bool(joint_stage_updates)
        self.train_actor_stage = {
            0: bool(train_accel),
            1: bool(train_sat),
            2: bool(train_bw),
        }
        self.any_train_actor_stage = any(bool(enabled) for enabled in self.train_actor_stage.values())
        self.exec_source_by_stage = {
            0: _normalize_exec_source(exec_accel_source),
            1: _normalize_exec_source(exec_sat_source),
            2: _normalize_exec_source(exec_bw_source),
        }
        self.bw_single_uav_policy_enabled = self.exec_source_by_stage[2] == "policy_single_uav_queue_aware"
        self.bw_single_uav_policy_uav_id = (
            0 if cfg is None else int(getattr(cfg, "bw_single_uav_policy_uav_id", 0) or 0)
        )
        self.bw_clean_per_user_enabled = (
            cfg is not None
            and bool(getattr(cfg, "bw_clean_per_user_enabled", False))
            and bool(self.train_actor_stage[2])
        )
        self._native_rollout_program: StructuredGpuRolloutProgram | None = None
        self._native_rollout_program_drivers_id: int | None = None
        self._native_rollout_program_device: torch.device | None = None
        self._native_actor_policy_binding: NativeActorCudaBinding | None = None
        self._native_actor_teacher_binding: NativeActorCudaBinding | None = None
        self.bw_clean_per_user_horizon = (
            10 if cfg is None else max(int(getattr(cfg, "bw_clean_per_user_horizon", 10) or 1), 1)
        )
        self.bw_clean_per_user_delta_probe = (
            0.02 if cfg is None else max(float(getattr(cfg, "bw_clean_per_user_delta_probe", 0.02) or 0.0), 1.0e-6)
        )
        self.bw_clean_per_user_beta = (
            0.5 if cfg is None else float(np.clip(float(getattr(cfg, "bw_clean_per_user_beta", 0.5) or 0.5), 1.0e-6, 1.0))
        )
        self.bw_clean_per_user_loss = (
            "huber"
            if cfg is None
            else str(getattr(cfg, "bw_clean_per_user_loss", "huber") or "huber").strip().lower()
        )
        self.bw_clean_candidate_select_enabled = bool(
            False if cfg is None else getattr(cfg, "bw_clean_candidate_select_enabled", False)
        )

        def _parse_clean_candidate_deltas(raw: Any) -> tuple[float, ...]:
            if raw is None:
                raw_values: list[Any] = [0.02, 0.05, 0.1, 0.2, 0.5]
            elif isinstance(raw, str):
                raw_values = [part for part in raw.replace(",", " ").split() if part]
            elif isinstance(raw, (list, tuple)):
                raw_values = list(raw)
            else:
                raw_values = [raw]
            parsed: list[float] = []
            for value in raw_values:
                delta = float(value)
                if delta <= 0.0:
                    continue
                if delta not in parsed:
                    parsed.append(delta)
            return tuple(parsed or [0.02])

        self.bw_clean_candidate_select_deltas = _parse_clean_candidate_deltas(
            None if cfg is None else getattr(cfg, "bw_clean_candidate_select_deltas", None)
        )
        self.bw_clean_candidate_select_include_onehot = bool(
            True if cfg is None else getattr(cfg, "bw_clean_candidate_select_include_onehot", True)
        )
        self.bw_clean_candidate_select_include_uniform = bool(
            True if cfg is None else getattr(cfg, "bw_clean_candidate_select_include_uniform", True)
        )
        self.bw_clean_candidate_select_row_filter_mode = (
            "ref_negative"
            if cfg is None
            else str(getattr(cfg, "bw_clean_candidate_select_row_filter_mode", "ref_negative") or "ref_negative")
            .strip()
            .lower()
        )
        if self.bw_clean_candidate_select_row_filter_mode in {"off", "all"}:
            self.bw_clean_candidate_select_row_filter_mode = "none"
        if self.bw_clean_candidate_select_row_filter_mode not in {"none", "ref_negative"}:
            raise ValueError(
                "bw_clean_candidate_select_row_filter_mode must be one of {'ref_negative', 'none'}."
            )
        self.bw_clean_candidate_select_ref_return_eps = (
            1.0e-6
            if cfg is None
            else max(float(getattr(cfg, "bw_clean_candidate_select_ref_return_eps", 1.0e-6) or 0.0), 0.0)
        )
        self.bw_clean_candidate_select_gate_eps = (
            1.0e-6
            if cfg is None
            else max(float(getattr(cfg, "bw_clean_candidate_select_gate_eps", 1.0e-6) or 0.0), 0.0)
        )
        self.bw_clean_grad_aggregation = (
            "mean"
            if cfg is None
            else str(getattr(cfg, "bw_clean_grad_aggregation", "mean") or "mean").strip().lower()
        )
        if self.bw_clean_grad_aggregation not in {"mean", "pcgrad"}:
            raise ValueError(
                f"Unsupported bw_clean_grad_aggregation={self.bw_clean_grad_aggregation!r}; expected 'mean' or 'pcgrad'."
            )
        self.bw_clean_pcgrad_task_group_size = (
            32 if cfg is None else max(int(getattr(cfg, "bw_clean_pcgrad_task_group_size", 32) or 1), 1)
        )
        self.bw_clean_pcgrad_group_mode = (
            "random"
            if cfg is None
            else str(getattr(cfg, "bw_clean_pcgrad_group_mode", "random") or "random").strip().lower()
        )
        if self.bw_clean_pcgrad_group_mode not in {"random", "target_stats"}:
            raise ValueError(
                "bw_clean_pcgrad_group_mode must be one of {'random', 'target_stats'}."
            )
        self.bw_clean_row_sample_enabled = bool(
            False if cfg is None else getattr(cfg, "bw_clean_row_sample_enabled", False)
        )
        self.bw_clean_row_sample_budget = (
            0 if cfg is None else max(int(getattr(cfg, "bw_clean_row_sample_budget", 0) or 0), 0)
        )
        clean_row_sample_seed_raw = None if cfg is None else getattr(cfg, "bw_clean_row_sample_seed", None)
        if clean_row_sample_seed_raw is None:
            clean_row_sample_seed_raw = (0 if cfg is None else int(getattr(cfg, "seed", 0) or 0)) + 104729
        self._bw_clean_row_sample_generator = torch.Generator(device="cpu")
        self._bw_clean_row_sample_generator.manual_seed(int(clean_row_sample_seed_raw))
        self.sat_clean_joint_enabled = (
            cfg is not None
            and bool(getattr(cfg, "sat_clean_joint_enabled", False))
            and bool(self.train_actor_stage[1])
        )
        self.sat_clean_topm_per_uav = (
            4 if cfg is None else max(int(getattr(cfg, "sat_clean_topm_per_uav", 4) or 1), 1)
        )
        self.sat_clean_contexts_per_update = (
            16 if cfg is None else max(int(getattr(cfg, "sat_clean_contexts_per_update", 16) or 1), 1)
        )
        self.sat_clean_entropy_topk_per_env = (
            2 if cfg is None else max(int(getattr(cfg, "sat_clean_entropy_topk_per_env", 2) or 0), 0)
        )
        self.sat_clean_uniform_contexts_per_env = (
            1 if cfg is None else max(int(getattr(cfg, "sat_clean_uniform_contexts_per_env", 1) or 0), 0)
        )
        self.sat_clean_parallel_envs = (
            0 if cfg is None else max(int(getattr(cfg, "sat_clean_parallel_envs", 0) or 0), 0)
        )
        self.sat_clean_parallel_backend = (
            "sync"
            if cfg is None
            else str(getattr(cfg, "sat_clean_parallel_backend", "sync") or "sync").strip().lower()
        )
        self.sat_clean_tasks_per_worker = (
            8 if cfg is None else max(int(getattr(cfg, "sat_clean_tasks_per_worker", 8) or 1), 1)
        )
        self.sat_clean_positive_gap_eps = (
            1.0e-6
            if cfg is None
            else max(float(getattr(cfg, "sat_clean_positive_gap_eps", 1.0e-6) or 0.0), 0.0)
        )
        self.bw_clean_trust_region_enabled = bool(
            False if cfg is None else getattr(cfg, "bw_clean_trust_region_enabled", False)
        )
        self.bw_clean_trust_region_target_kl = (
            0.01
            if cfg is None
            else max(float(getattr(cfg, "bw_clean_trust_region_target_kl", 0.01) or 0.0), 1.0e-8)
        )
        self.bw_clean_trust_region_kl_coef_init = (
            0.1
            if cfg is None
            else max(float(getattr(cfg, "bw_clean_trust_region_kl_coef_init", 0.1) or 0.0), 0.0)
        )
        self.bw_clean_trust_region_kl_coef_min = (
            1.0e-4
            if cfg is None
            else max(float(getattr(cfg, "bw_clean_trust_region_kl_coef_min", 1.0e-4) or 0.0), 0.0)
        )
        self.bw_clean_trust_region_kl_coef_max = (
            1.0e3
            if cfg is None
            else max(float(getattr(cfg, "bw_clean_trust_region_kl_coef_max", 1.0e3) or 0.0), 0.0)
        )
        if self.bw_clean_trust_region_kl_coef_max < self.bw_clean_trust_region_kl_coef_min:
            self.bw_clean_trust_region_kl_coef_max = self.bw_clean_trust_region_kl_coef_min
        self.bw_clean_trust_region_backtrack_factor = (
            0.5
            if cfg is None
            else float(getattr(cfg, "bw_clean_trust_region_backtrack_factor", 0.5) or 0.5)
        )
        self.bw_clean_trust_region_backtrack_factor = min(
            max(float(self.bw_clean_trust_region_backtrack_factor), 1.0e-3),
            0.999,
        )
        self.bw_clean_trust_region_max_backtracks = (
            4
            if cfg is None
            else max(int(getattr(cfg, "bw_clean_trust_region_max_backtracks", 4) or 0), 0)
        )
        if self.bw_clean_per_user_enabled:
            if bool(self.train_actor_stage[0]) or bool(self.train_actor_stage[1]) or not bool(self.train_actor_stage[2]):
                raise ValueError("bw_clean_per_user_enabled requires train_accel=False, train_sat=False, train_bw=True.")
            if self.exec_source_by_stage[2] != "policy":
                raise ValueError("bw_clean_per_user_enabled requires exec_bw_source='policy'.")
            if self.bw_clean_per_user_loss not in {"huber", "masked_kl"}:
                raise ValueError("bw_clean_per_user_loss currently only supports 'huber' or 'masked_kl'.")
        if self.sat_clean_joint_enabled:
            if bool(self.train_actor_stage[0]) or bool(self.train_actor_stage[2]) or not bool(self.train_actor_stage[1]):
                raise ValueError("sat_clean_joint_enabled requires train_accel=False, train_sat=True, train_bw=False.")
            if self.cfg is not None and int(getattr(self.cfg, "num_uav", 0) or 0) <= 1:
                raise ValueError("sat_clean_joint_enabled requires num_uav > 1.")
            if self.exec_source_by_stage[0] != "cluster_center_queue_aware":
                raise ValueError("sat_clean_joint_enabled requires exec_accel_source='cluster_center_queue_aware'.")
            if self.exec_source_by_stage[1] != "policy":
                raise ValueError("sat_clean_joint_enabled requires exec_sat_source='policy'.")
            if self.exec_source_by_stage[2] != "queue_aware":
                raise ValueError("sat_clean_joint_enabled requires exec_bw_source='queue_aware'.")
            reward_mode_l = str(getattr(self.cfg, "reward_mode", "dense") or "dense").strip().lower()
            if reward_mode_l != "weighted_workload_level":
                raise ValueError("sat_clean_joint_enabled requires reward_mode='weighted_workload_level'.")
            if bool(getattr(self.cfg, "reward_stage3_sat_overlap_enabled", False)):
                raise ValueError("sat_clean_joint_enabled cannot be mixed with reward_stage3_sat_overlap_enabled.")
            if bool(getattr(self.cfg, "sat_counterfactual_credit_enabled", False)):
                raise ValueError("sat_clean_joint_enabled cannot be mixed with sat_counterfactual_credit_enabled.")
            if bool(getattr(self.cfg, "sat_supervision_enabled", False)):
                raise ValueError("sat_clean_joint_enabled cannot be mixed with sat_supervision_enabled.")
        self._bw_clean_trust_region_kl_coef = float(self.bw_clean_trust_region_kl_coef_init)
        bw_counterfactual_credit_weight = (
            0.0 if cfg is None else float(getattr(cfg, "bw_counterfactual_credit_weight", 1.0) or 0.0)
        )
        self.bw_counterfactual_credit_weight = float(np.clip(bw_counterfactual_credit_weight, 0.0, 1.0))
        self.bw_counterfactual_credit_enabled = (
            cfg is not None
            and bool(getattr(cfg, "bw_counterfactual_credit_enabled", False))
            and self.bw_counterfactual_credit_weight > 0.0
            and bool(self.train_actor_stage[2])
        )
        self.bw_policy_update_mode = (
            "ppo"
            if cfg is None
            else str(getattr(cfg, "structured_bw_policy_update_mode", "ppo") or "ppo").strip().lower()
        )
        if self.bw_policy_update_mode not in {"ppo", "awr", "det_awr", "vmpo_lite"}:
            raise ValueError("structured_bw_policy_update_mode must be one of {'ppo', 'awr', 'det_awr', 'vmpo_lite'}.")
        self.structured_actor_update_mode = (
            "ppo"
            if cfg is None
            else str(getattr(cfg, "structured_actor_update_mode", "ppo") or "ppo").strip().lower()
        )
        if self.structured_actor_update_mode not in {"ppo", "vs_ref"}:
            raise ValueError("structured_actor_update_mode must be one of {'ppo', 'vs_ref'}.")
        self.stage_actor_update_mode = {}
        for _stage_id, _attr in ((0, "accel_update_mode"), (1, "sat_update_mode"), (2, "bw_update_mode")):
            _mode = getattr(cfg, _attr, None) if cfg is not None else None
            _mode_s = self.structured_actor_update_mode if _mode is None or str(_mode).strip() == "" else str(_mode).strip().lower()
            if _mode_s not in {"ppo", "vs_ref"}:
                raise ValueError(f"{_attr} must be one of {{'ppo', 'vs_ref'}} when set.")
            self.stage_actor_update_mode[int(_stage_id)] = _mode_s
        self.vs_ref_rows_per_update = 32 if cfg is None else max(int(getattr(cfg, "vs_ref_rows_per_update", 32) or 32), 1)
        self.vs_ref_samples_per_row = 1 if cfg is None else max(int(getattr(cfg, "vs_ref_samples_per_row", 1) or 1), 1)
        self.vs_ref_horizon_mode = (
            "episode_remaining"
            if cfg is None
            else str(getattr(cfg, "vs_ref_horizon_mode", "episode_remaining") or "episode_remaining").strip().lower()
        )
        if self.vs_ref_horizon_mode not in {"episode_remaining"}:
            raise ValueError("vs_ref_horizon_mode currently supports only 'episode_remaining'.")
        self.vs_ref_advantage_normalize = (
            "stage"
            if cfg is None
            else str(getattr(cfg, "vs_ref_advantage_normalize", "stage") or "stage").strip().lower()
        )
        if self.vs_ref_advantage_normalize not in {"none", "stage"}:
            raise ValueError("vs_ref_advantage_normalize must be one of {'none', 'stage'}.")
        self.vs_ref_disable_critic = bool(True if cfg is None else getattr(cfg, "vs_ref_disable_critic", True))
        self.vs_ref_sampling_mode = (
            "uniform"
            if cfg is None
            else str(getattr(cfg, "vs_ref_sampling_mode", "uniform") or "uniform").strip().lower()
        )
        if self.vs_ref_sampling_mode not in {"uniform", "active_mixture"}:
            raise ValueError("vs_ref_sampling_mode must be one of {'uniform', 'active_mixture'}.")
        self.vs_ref_sampling_alpha = (
            0.6
            if cfg is None
            else float(np.clip(float(getattr(cfg, "vs_ref_sampling_alpha", 0.6) or 0.0), 0.0, 4.0))
        )
        self.vs_ref_sampling_random_frac = (
            0.25 if cfg is None else max(float(getattr(cfg, "vs_ref_sampling_random_frac", 0.25) or 0.0), 0.0)
        )
        self.vs_ref_sampling_time_frac = (
            0.25 if cfg is None else max(float(getattr(cfg, "vs_ref_sampling_time_frac", 0.25) or 0.0), 0.0)
        )
        self.vs_ref_sampling_leverage_frac = (
            0.25 if cfg is None else max(float(getattr(cfg, "vs_ref_sampling_leverage_frac", 0.25) or 0.0), 0.0)
        )
        self.vs_ref_sampling_uncertainty_frac = (
            0.25 if cfg is None else max(float(getattr(cfg, "vs_ref_sampling_uncertainty_frac", 0.25) or 0.0), 0.0)
        )
        self.vs_ref_sampling_cost_power = (
            0.5 if cfg is None else max(float(getattr(cfg, "vs_ref_sampling_cost_power", 0.5) or 0.0), 0.0)
        )
        self.any_vs_ref_train_stage = any(
            self.stage_actor_update_mode.get(i) == "vs_ref" and bool(self.train_actor_stage[i])
            for i in (0, 1, 2)
        )
        self.any_ppo_train_stage = any(
            self.stage_actor_update_mode.get(i, "ppo") == "ppo" and bool(self.train_actor_stage[i])
            for i in (0, 1, 2)
        )
        if self.any_vs_ref_train_stage:
            _backend = "cuda" if cfg is None else str(getattr(cfg, "structured_env_tensor_backend", "cuda") or "cuda").strip().lower()
            if _backend != "cuda":
                raise ValueError("sample-vs-ref actor update requires structured_env_tensor_backend='cuda'.")
        self.bw_per_slot_surrogate_enabled = (
            cfg is not None
            and bool(getattr(cfg, "structured_bw_per_slot_surrogate_enabled", False))
            and bool(self.train_actor_stage[2])
        )
        self.bw_slot_advantage_weight = (
            1.0 if cfg is None else float(getattr(cfg, "bw_slot_advantage_weight", 1.0) or 0.0)
        )
        self.bw_slot_advantage_base_mode = (
            "sample"
            if cfg is None
            else str(getattr(cfg, "bw_slot_advantage_base_mode", "sample") or "sample").strip().lower()
        )
        if self.bw_slot_advantage_base_mode not in {"sample", "zero"}:
            raise ValueError("bw_slot_advantage_base_mode must be one of {'sample', 'zero'}.")
        if self.bw_per_slot_surrogate_enabled and self.bw_policy_update_mode != "ppo":
            raise ValueError("structured_bw_per_slot_surrogate_enabled requires structured_bw_policy_update_mode='ppo'.")
        self.bw_awr_temperature = (
            0.5
            if cfg is None
            else max(float(getattr(cfg, "structured_bw_awr_temperature", 0.5) or 0.5), 1.0e-6)
        )
        self.bw_awr_max_weight = (
            20.0
            if cfg is None
            else max(float(getattr(cfg, "structured_bw_awr_max_weight", 20.0) or 20.0), 1.0)
        )
        self.bw_awr_normalize_weights = bool(
            True if cfg is None else getattr(cfg, "structured_bw_awr_normalize_weights", True)
        )
        self.bw_awr_kl_coef = (
            0.0
            if cfg is None
            else max(float(getattr(cfg, "structured_bw_awr_kl_coef", 0.0) or 0.0), 0.0)
        )
        self.bw_vmpo_temperature = (
            0.5
            if cfg is None
            else max(float(getattr(cfg, "structured_bw_vmpo_temperature", 0.5) or 0.5), 1.0e-6)
        )
        self.bw_vmpo_top_frac = (
            0.5
            if cfg is None
            else float(np.clip(float(getattr(cfg, "structured_bw_vmpo_top_frac", 0.5) or 0.5), 1.0e-3, 1.0))
        )
        self.bw_vmpo_kl_coef = (
            0.01
            if cfg is None
            else max(float(getattr(cfg, "structured_bw_vmpo_kl_coef", 0.01) or 0.0), 0.0)
        )
        bw_flow_proxy_aux_coef = 0.0 if cfg is None else float(getattr(cfg, "bw_flow_proxy_aux_coef", 0.0) or 0.0)
        self.bw_flow_proxy_aux_enabled = (
            cfg is not None
            and bool(getattr(cfg, "bw_flow_proxy_aux_enabled", False))
            and bw_flow_proxy_aux_coef > 0.0
            and bool(self.train_actor_stage[2])
        )
        bw_marginal_teacher_sample_weight = (
            0.0 if cfg is None else float(getattr(cfg, "bw_marginal_teacher_sample_weight", 1.0) or 0.0)
        )
        self.bw_marginal_teacher_sample_weight = float(np.clip(bw_marginal_teacher_sample_weight, 0.0, 1.0))
        self.bw_marginal_teacher_sample_enabled = (
            cfg is not None
            and bool(getattr(cfg, "bw_marginal_teacher_sample_enabled", False))
            and self.bw_marginal_teacher_sample_weight > 0.0
            and bool(self.train_actor_stage[2])
        )
        self.bw_flow_proxy_signal_enabled = (
            self.bw_counterfactual_credit_enabled
            or self.bw_flow_proxy_aux_enabled
            or self.bw_marginal_teacher_sample_enabled
            or self.bw_per_slot_surrogate_enabled
        )
        self.bw_flow_proxy_aux_coef = max(float(bw_flow_proxy_aux_coef), 0.0)
        self.bw_flow_proxy_aux_min_gap = (
            0.0 if cfg is None else max(float(getattr(cfg, "bw_flow_proxy_aux_min_gap", 1.0e-4) or 0.0), 0.0)
        )
        self.bw_flow_proxy_aux_regression_coef = (
            0.0
            if cfg is None
            else max(float(getattr(cfg, "bw_flow_proxy_aux_regression_coef", 0.0) or 0.0), 0.0)
        )
        self.bw_flow_proxy_grad_diagnostics_enabled = bool(
            False if cfg is None else getattr(cfg, "bw_flow_proxy_grad_diagnostics_enabled", False)
        )
        self.step_train_target_mode = (
            "env_reward"
            if cfg is None
            else str(getattr(cfg, "step_train_target_mode", "env_reward") or "env_reward").strip().lower()
        )
        if self.step_train_target_mode not in {"env_reward", "access_term", "access_raw"}:
            raise ValueError(
                "step_train_target_mode must be one of {'env_reward', 'access_term', 'access_raw'}."
            )
        self.bw_train_target_mode = (
            "env_reward" if cfg is None else str(getattr(cfg, "bw_train_target_mode", "env_reward") or "env_reward").strip().lower()
        )
        if self.bw_train_target_mode not in {
            "env_reward",
            "access_term",
            "access_raw",
            "weighted_workload_delta",
            "weighted_workload_level",
            "gu_queue_level",
            "system_queue_level",
            "gu_service_queue",
        }:
            raise ValueError(
                "bw_train_target_mode must be one of {'env_reward', 'access_term', 'access_raw', 'weighted_workload_delta', 'weighted_workload_level', 'gu_queue_level', 'system_queue_level', 'gu_service_queue'}."
            )
        self.bw_return_mode = (
            "gae" if cfg is None else str(getattr(cfg, "bw_return_mode", "gae") or "gae").strip().lower()
        )
        if self.bw_return_mode not in {
            "gae",
            "bw_gae",
            "step_lambda_return",
            "monte_carlo",
            "mc",
            "bw_episode_mc",
            "bw_nstep",
        }:
            raise ValueError(
                "bw_return_mode must be one of {'gae', 'bw_gae', 'step_lambda_return', 'bw_episode_mc', 'bw_nstep'}."
            )
        if self.bw_return_mode in {"mc", "monte_carlo"}:
            # Backward-compatible alias for the historical name. It is not
            # strict MC when gae_lambda < 1; future next_step_return terms
            # still include lambda-return value bootstrap.
            self.bw_return_mode = "step_lambda_return"
        self.bw_nstep_horizon = 3 if cfg is None else max(int(getattr(cfg, "bw_nstep_horizon", 3) or 3), 1)
        self.time_limit_bootstrap_enabled = bool(
            False if cfg is None else getattr(cfg, "time_limit_bootstrap_enabled", False)
        )
        self.step_bootstrap_stage = (
            "accel"
            if cfg is None
            else str(getattr(cfg, "structured_step_bootstrap_stage", "accel") or "accel").strip().lower()
        )
        if self.step_bootstrap_stage not in {"accel", "bw"}:
            raise ValueError("structured_step_bootstrap_stage must be one of {'accel', 'bw'}.")
        if self.step_bootstrap_stage == "bw" and self.any_ppo_train_stage and (
            bool(self.train_actor_stage[0]) or bool(self.train_actor_stage[1]) or not bool(self.train_actor_stage[2])
        ):
            raise ValueError(
                "structured_step_bootstrap_stage='bw' requires train_accel=False, train_sat=False, train_bw=True."
            )
        if self.bw_return_mode == "bw_gae":
            if str(self.target_mode).strip().lower() != "step_level":
                raise ValueError("bw_return_mode='bw_gae' requires target_mode='step_level'.")
            if self.any_ppo_train_stage and (
                bool(self.train_actor_stage[0]) or bool(self.train_actor_stage[1]) or not bool(self.train_actor_stage[2])
            ):
                raise ValueError(
                    "bw_return_mode='bw_gae' requires train_accel=False, train_sat=False, train_bw=True."
                )
            self.step_bootstrap_stage = "bw"
        self.bw_reward_w_access = 0.0 if cfg is None else float(getattr(cfg, "reward_w_access", 0.0) or 0.0)
        reward_mode = "dense" if cfg is None else str(getattr(cfg, "reward_mode", "dense") or "dense").strip().lower()
        self.stagewise_advantage_norm_enabled = bool(
            True if cfg is None else getattr(cfg, "stagewise_advantage_norm_enabled", True)
        )
        self.actor_advantage_normalize_enabled = bool(
            True if cfg is None else getattr(cfg, "actor_advantage_normalize_enabled", True)
        )
        self.bw_actor_advantage_override_mode = (
            "gae"
            if cfg is None
            else str(getattr(cfg, "bw_actor_advantage_override_mode", "gae") or "gae").strip().lower()
        )
        if self.bw_actor_advantage_override_mode not in {"gae", "branch_delta", "true_adv_mc", "delta_critic", "delta_teacher_student"}:
            raise ValueError(
                "bw_actor_advantage_override_mode must be one of {'gae', 'branch_delta', 'true_adv_mc', 'delta_critic', 'delta_teacher_student'}."
            )
        self.bw_delta_critic_enabled = bool(
            False if cfg is None else getattr(cfg, "bw_delta_critic_enabled", False)
        ) or self.bw_actor_advantage_override_mode in {"delta_critic", "delta_teacher_student"}
        self.bw_delta_critic_horizon = (
            3 if cfg is None else max(int(getattr(cfg, "bw_delta_critic_horizon", 3) or 3), 1)
        )
        self.bw_delta_critic_samples = (
            1 if cfg is None else max(int(getattr(cfg, "bw_delta_critic_samples", 1) or 1), 1)
        )
        self.bw_delta_critic_ref_mode = (
            "deterministic"
            if cfg is None
            else str(getattr(cfg, "bw_delta_critic_ref_mode", "deterministic") or "deterministic").strip().lower()
        )
        self.bw_delta_critic_follow_policy_mode = (
            "deterministic"
            if cfg is None
            else str(
                getattr(cfg, "bw_delta_critic_follow_policy_mode", "deterministic") or "deterministic"
            ).strip().lower()
        )
        self.bw_delta_critic_loss_coef = (
            1.0 if cfg is None else max(float(getattr(cfg, "bw_delta_critic_loss_coef", 1.0) or 0.0), 0.0)
        )
        self.bw_delta_critic_warmup_epochs = (
            1 if cfg is None else max(int(getattr(cfg, "bw_delta_critic_warmup_epochs", 1) or 0), 0)
        )
        self.bw_delta_teacher_student_enabled = self.bw_actor_advantage_override_mode == "delta_teacher_student"
        self.bw_delta_teacher_student_corr_low = (
            0.6
            if cfg is None
            else float(getattr(cfg, "bw_delta_teacher_student_corr_low", 0.6) or 0.6)
        )
        self.bw_delta_teacher_student_corr_high = (
            0.9
            if cfg is None
            else float(getattr(cfg, "bw_delta_teacher_student_corr_high", 0.9) or 0.9)
        )
        if self.bw_delta_teacher_student_corr_high <= self.bw_delta_teacher_student_corr_low:
            self.bw_delta_teacher_student_corr_high = self.bw_delta_teacher_student_corr_low + 1.0e-6
        self.bw_delta_teacher_student_mix_power = (
            1.0
            if cfg is None
            else max(float(getattr(cfg, "bw_delta_teacher_student_mix_power", 1.0) or 1.0), 1.0e-6)
        )
        self.bw_delta_teacher_student_disable_dense_teacher_when_ready = bool(
            False
            if cfg is None
            else getattr(cfg, "bw_delta_teacher_student_disable_dense_teacher_when_ready", False)
        )
        self.bw_delta_teacher_student_ready_patience = (
            2
            if cfg is None
            else max(int(getattr(cfg, "bw_delta_teacher_student_ready_patience", 2) or 1), 1)
        )
        self.bw_delta_teacher_student_probe_interval_updates = (
            0
            if cfg is None
            else max(int(getattr(cfg, "bw_delta_teacher_student_probe_interval_updates", 0) or 0), 0)
        )
        self.bw_delta_teacher_student_probe_reenable_corr = (
            0.85
            if cfg is None
            else float(getattr(cfg, "bw_delta_teacher_student_probe_reenable_corr", 0.85) or 0.85)
        )
        self.bw_delta_only_training_enabled = (
            self.bw_delta_critic_enabled
            and self.bw_actor_advantage_override_mode in {"delta_critic", "delta_teacher_student"}
            and bool(self.train_actor_stage[2])
            and not bool(self.train_actor_stage[0])
            and not bool(self.train_actor_stage[1])
        )
        self.disable_critic_training_for_direct_branch_delta = direct_branch_delta_actor_only_enabled(
            self.cfg,
            train_accel=bool(self.train_actor_stage[0]),
            train_sat=bool(self.train_actor_stage[1]),
            train_bw=bool(self.train_actor_stage[2]),
        )
        self.disable_critic_training_for_bw_actor_only_signal = bw_actor_only_signal_critic_free_enabled(
            self.cfg,
            train_accel=bool(self.train_actor_stage[0]),
            train_sat=bool(self.train_actor_stage[1]),
            train_bw=bool(self.train_actor_stage[2]),
        )
        self.disable_critic_training_for_sat_clean_joint = sat_clean_joint_critic_free_enabled(
            self.cfg,
            train_accel=bool(self.train_actor_stage[0]),
            train_sat=bool(self.train_actor_stage[1]),
            train_bw=bool(self.train_actor_stage[2]),
        )
        self.update_direction_probe_enabled = bool(
            False
            if cfg is None
            else (
                bool(getattr(cfg, "update_direction_probe_enabled", False))
                or self.bw_actor_advantage_override_mode in {"branch_delta", "true_adv_mc"}
            )
        )
        if self.bw_clean_per_user_enabled:
            if self.bw_actor_advantage_override_mode != "gae":
                raise ValueError("bw_clean_per_user_enabled requires bw_actor_advantage_override_mode='gae'.")
            if self.bw_counterfactual_credit_enabled:
                raise ValueError("bw_clean_per_user_enabled cannot be mixed with bw_counterfactual_credit_enabled.")
            if self.bw_flow_proxy_aux_enabled:
                raise ValueError("bw_clean_per_user_enabled cannot be mixed with bw_flow_proxy_aux_enabled.")
            if self.bw_marginal_teacher_sample_enabled:
                raise ValueError("bw_clean_per_user_enabled cannot be mixed with bw_marginal_teacher_sample_enabled.")
            if self.bw_per_slot_surrogate_enabled:
                raise ValueError("bw_clean_per_user_enabled cannot be mixed with structured_bw_per_slot_surrogate_enabled.")
        self.entropy_coef_by_stage = {
            0: float(
                self.entropy_coef
                if cfg is None or getattr(cfg, "entropy_coef_accel", None) is None
                else getattr(cfg, "entropy_coef_accel")
            ),
            1: float(
                self.entropy_coef
                if cfg is None or getattr(cfg, "entropy_coef_sat", None) is None
                else getattr(cfg, "entropy_coef_sat")
            ),
            2: float(
                self.entropy_coef
                if cfg is None or getattr(cfg, "entropy_coef_bw", None) is None
                else getattr(cfg, "entropy_coef_bw")
            ),
        }
        self.bw_per_uav_surrogate_enabled = bool(
            False if cfg is None else getattr(cfg, "structured_bw_per_uav_surrogate_enabled", False)
        )
        self.bw_entropy_norm_mode = (
            "none"
            if cfg is None
            else str(getattr(cfg, "structured_bw_entropy_norm_mode", "none") or "none").strip().lower()
        )
        if self.bw_entropy_norm_mode not in {"none", "per_latent_count", "per_simplex_dim"}:
            raise ValueError(
                "structured_bw_entropy_norm_mode must be one of {'none', 'per_latent_count', 'per_simplex_dim'}."
            )
        stage_name_by_id = {0: "accel", 1: "sat", 2: "bw"}
        for stage_id, stage_name in stage_name_by_id.items():
            source = self.exec_source_by_stage[stage_id]
            if self.train_actor_stage[stage_id] and source not in _NATIVE_POLICY_ACTION_SOURCES:
                raise ValueError(
                    f"train_{stage_name}=True requires exec_{stage_name}_source=policy or another policy source in structured training."
                )
        self.device = torch.device(device if device is not None else "cpu")
        self.structured_env_tensor_backend = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if cfg is None
            else str(getattr(cfg, "structured_env_tensor_backend", "cuda") or "cuda").strip().lower()
        )
        if self.structured_env_tensor_backend not in {"cpu", "cuda", "auto"}:
            raise ValueError(
                "structured_env_tensor_backend must be one of {'cpu', 'cuda', 'auto'}."
            )
        self._bw_clean_probe_group = None
        self._bw_clean_probe_group_signature = None
        self._sat_clean_probe_group = None
        self._sat_clean_probe_group_signature = None
        self._bw_clean_last_ref_returns: np.ndarray | None = None
        self._bw_clean_last_row_sample_stats: dict[str, float] | None = None
        self._bw_clean_last_group_debug: dict[str, Any] | None = None
        self.teacher_deterministic = bool(True if cfg is None else getattr(cfg, "exec_teacher_deterministic", True))
        self.actor.to(self.device)
        self.critic.to(self.device)
        self._initialize_exec_teacher_actor()
        self._ensure_native_actor_cuda_bindings(sync=True)

    def _initialize_exec_teacher_actor(self) -> None:
        if "teacher" not in set(self.exec_source_by_stage.values()):
            return
        if self.teacher_actor is None:
            teacher_path = None if self.cfg is None else getattr(self.cfg, "exec_teacher_actor_path", None)
            if not teacher_path:
                raise RuntimeError("teacher exec source requires cfg.exec_teacher_actor_path or a teacher_actor argument.")
            self.teacher_actor = copy.deepcopy(self.actor)
            from sagin_marl.utils.checkpoint import load_checkpoint_forgiving

            load_checkpoint_forgiving(self.teacher_actor, str(teacher_path), map_location=self.device)
        self.teacher_actor.to(self.device)
        self.teacher_actor.eval()
        for param in self.teacher_actor.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _native_driver_cfg(drivers: Any) -> Any | None:
        for attr_name in ("cfg", "_cfg"):
            value = getattr(drivers, attr_name, None)
            if value is not None:
                return value
        for attr_name in ("batch_core", "_batch_core"):
            core = getattr(drivers, attr_name, None)
            value = getattr(core, "cfg", None)
            if value is not None:
                return value
        batch_env = getattr(drivers, "batch_env", None)
        core = getattr(batch_env, "_core", None)
        value = getattr(core, "cfg", None)
        if value is not None:
            return value
        return None

    @staticmethod
    def _native_driver_tensor_device(drivers: Any) -> torch.device | None:
        for attr_name in ("tensor_device", "_tensor_device"):
            value = getattr(drivers, attr_name, None)
            if value is not None:
                return torch.device(value)
        runtime = getattr(drivers, "native_rollout_runtime", None)
        value = getattr(runtime, "device", None)
        if value is not None:
            return torch.device(value)
        return None

    @staticmethod
    def _move_optimizer_state_to_device(optimizer: Any, device: torch.device) -> None:
        if optimizer is None:
            return
        for state in getattr(optimizer, "state", {}).values():
            if not isinstance(state, dict):
                continue
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    def bind_native_runtime_contract(self, drivers: Any) -> None:
        """Bind a learner created from modules to the env-owned native tensor runtime."""

        cfg = self._native_driver_cfg(drivers)
        if cfg is not None and self.cfg is None:
            self.cfg = cfg

        requested_tensor_backend = (
            "cuda"
            if cfg is None
            else str(getattr(cfg, "structured_env_tensor_backend", "cuda") or "cuda").strip().lower()
        )
        if requested_tensor_backend not in {"cuda", "auto"}:
            raise RuntimeError("final native structured rollout requires structured_env_tensor_backend='cuda'.")
        if not torch.cuda.is_available():
            raise RuntimeError("final native structured rollout requires CUDA.")
        tensor_device = torch.device("cuda")

        if tensor_device is not None:
            tensor_device = torch.device(tensor_device)
            self.structured_env_tensor_backend = str(tensor_device.type)
            if self.device != tensor_device:
                self.device = tensor_device
                self.actor.to(self.device)
                self.critic.to(self.device)
                if self.teacher_actor is not None:
                    self.teacher_actor.to(self.device)
                self._move_optimizer_state_to_device(self.actor_optimizer, self.device)
                for optimizer in self.actor_stage_optimizers.values():
                    self._move_optimizer_state_to_device(optimizer, self.device)
                self._move_optimizer_state_to_device(self.critic_optimizer, self.device)
                self._native_rollout_program = None
                self._native_rollout_program_drivers_id = None
                self._native_rollout_program_device = None
            set_group_tensor_device = getattr(drivers, "set_tensor_device", None)
            if callable(set_group_tensor_device):
                set_group_tensor_device(self.device)

        self._ensure_native_actor_cuda_bindings(sync=True)

    def _native_rollout_policy_actor_required(self) -> bool:
        if self.cfg is None or self.device.type != "cuda":
            return False
        backend = str(getattr(self.cfg, "structured_env_backend", "") or "").strip().lower()
        tensor_backend = str(self.structured_env_tensor_backend or "").strip().lower()
        return bool(
            backend == "native"
            and tensor_backend == "cuda"
            and any(
                self._native_policy_source_requires_cuda_binding(self.exec_source_by_stage.get(stage_id))
                for stage_id in (0, 1, 2)
            )
        )

    def _flat_policy_actor_native_module_enabled(self) -> bool:
        if self.cfg is None:
            return False
        return str(getattr(self.cfg, "structured_actor_backbone", "") or "").strip().lower() == "flat_mlp"

    def _native_policy_source_requires_cuda_binding(self, source: Any) -> bool:
        source_s = str(source or "")
        if source_s not in _NATIVE_POLICY_ACTION_SOURCES:
            return False
        if source_s == "policy" and self._flat_policy_actor_native_module_enabled():
            return False
        return True

    def _native_rollout_teacher_actor_required(self) -> bool:
        if self.cfg is None or self.device.type != "cuda":
            return False
        backend = str(getattr(self.cfg, "structured_env_backend", "") or "").strip().lower()
        tensor_backend = str(self.structured_env_tensor_backend or "").strip().lower()
        return bool(
            backend == "native"
            and tensor_backend == "cuda"
            and any(self.exec_source_by_stage.get(stage_id) == "teacher" for stage_id in (0, 1, 2))
        )

    def _ensure_native_actor_cuda_bindings(self, *, sync: bool) -> None:
        if self.cfg is None or self.device.type != "cuda":
            return
        if self._native_rollout_policy_actor_required():
            if self._native_actor_policy_binding is None or self._native_actor_policy_binding.actor is not self.actor:
                self._native_actor_policy_binding = build_native_actor_cuda_binding(self.actor, device=self.device)
            elif sync:
                self._native_actor_policy_binding.sync_from_module()
        if self._native_rollout_teacher_actor_required():
            teacher = self.teacher_actor
            if teacher is None:
                raise RuntimeError("teacher exec source requires a loaded teacher actor.")
            if self._native_actor_teacher_binding is None or self._native_actor_teacher_binding.actor is not teacher:
                self._native_actor_teacher_binding = build_native_actor_cuda_binding(teacher, device=self.device)
            elif sync:
                self._native_actor_teacher_binding.sync_from_module()

    def _sync_native_actor_cuda_bindings_after_update(self) -> None:
        if self.device.type != "cuda":
            return
        if self._native_actor_policy_binding is not None:
            self._native_actor_policy_binding.sync_from_module()
        if self._native_actor_teacher_binding is not None:
            self._native_actor_teacher_binding.sync_from_module()

    def _require_native_actor_policy_binding(self) -> NativeActorCudaBinding:
        self._ensure_native_actor_cuda_bindings(sync=False)
        binding = self._native_actor_policy_binding
        if binding is None:
            raise RuntimeError("official CUDA native policy rollout requires a native actor CUDA binding.")
        return binding

    def _require_native_actor_teacher_binding(self) -> NativeActorCudaBinding:
        self._ensure_native_actor_cuda_bindings(sync=False)
        binding = self._native_actor_teacher_binding
        if binding is None:
            raise RuntimeError("official CUDA native teacher rollout requires a native actor CUDA binding.")
        return binding

    def _structured_env_tensor_device(self) -> torch.device | None:
        mode = self.structured_env_tensor_backend
        if mode == "cpu":
            return torch.device("cpu") if self.device.type == "cpu" else None
        if mode == "auto":
            return self.device
        if mode == "cuda" and self.device.type != "cuda":
            return None
        return self.device

    def set_actor_advantage_override(self, stage_id: int, advantages: np.ndarray | torch.Tensor) -> None:
        array = np.asarray(
            advantages.detach().cpu().numpy() if torch.is_tensor(advantages) else advantages,
            dtype=np.float32,
        ).reshape(-1)
        self._actor_advantage_override_by_stage[int(stage_id)] = array.copy()

    def clear_actor_advantage_override(self) -> None:
        self._actor_advantage_override_by_stage.clear()

    def request_skip_actor_update_once(self, stage_id: int) -> None:
        self._skip_actor_update_stage_ids_once.add(int(stage_id))

    def set_bw_branch_gate_state(self, *, snr: float, triggered: bool) -> None:
        self._bw_branch_gate_snr_once = float(snr)
        self._bw_branch_gate_triggered_once = 1.0 if bool(triggered) else 0.0

    @staticmethod
    def _bw_is_dirichlet_actor_out(actor_out: Any) -> bool:
        return actor_out is not None and getattr(actor_out, "kappa", None) is not None

    @staticmethod
    def _bw_simplex_dim(
        valid_count: torch.Tensor | None,
        latent_count: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if valid_count is not None:
            return torch.where(valid_count > 1, valid_count - 1, torch.zeros_like(valid_count))
        return latent_count

    def _bw_agent_mask(
        self,
        actor_out: Any,
        valid_count: torch.Tensor | None,
        latent_count: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if self._bw_is_dirichlet_actor_out(actor_out) and valid_count is not None:
            agent_mask = valid_count > 1
        elif latent_count is not None:
            agent_mask = latent_count > 0
        elif valid_count is not None:
            agent_mask = valid_count > 1
        else:
            agent_mask = torch.ones_like(reference, dtype=torch.bool)
        agent_mask = agent_mask.to(device=reference.device, dtype=torch.bool)
        if agent_mask.shape != reference.shape:
            if agent_mask.numel() == reference.numel():
                agent_mask = agent_mask.reshape_as(reference)
            elif agent_mask.ndim == 1 and reference.ndim == 2 and agent_mask.numel() == reference.shape[0]:
                agent_mask = agent_mask[:, None].expand_as(reference)
            elif agent_mask.ndim == 2 and reference.ndim == 2 and agent_mask.shape[0] == reference.shape[0] and agent_mask.shape[1] == 1:
                agent_mask = agent_mask.expand_as(reference)
            else:
                agent_mask = torch.ones_like(reference, dtype=torch.bool)
        if not bool(torch.any(agent_mask).item()):
            agent_mask = torch.ones_like(reference, dtype=torch.bool)
        return agent_mask

    def _bw_awr_weights(self, advantage: torch.Tensor) -> torch.Tensor:
        scaled = advantage / float(self.bw_awr_temperature)
        max_log_weight = float(np.log(max(self.bw_awr_max_weight, 1.0)))
        scaled = torch.clamp(scaled, min=-20.0, max=max_log_weight)
        weights = torch.exp(scaled)
        if self.bw_awr_normalize_weights:
            weights = weights / weights.mean().clamp_min(1.0e-8)
        return weights.detach()

    def _bw_vmpo_selection(self, advantage: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        adv_flat = advantage.reshape(-1)
        positive_idx = torch.nonzero(adv_flat > 0.0, as_tuple=False).flatten()
        if positive_idx.numel() > 0:
            candidate_idx = positive_idx
            candidate_adv = adv_flat.index_select(0, positive_idx)
        else:
            candidate_idx = torch.arange(adv_flat.numel(), device=adv_flat.device)
            candidate_adv = adv_flat
        top_k = max(1, int(math.ceil(float(self.bw_vmpo_top_frac) * float(candidate_adv.numel()))))
        top_k = min(top_k, int(candidate_adv.numel()))
        if top_k < int(candidate_adv.numel()):
            _, local_idx = torch.topk(candidate_adv, k=top_k, largest=True, sorted=False)
            selected_idx = candidate_idx.index_select(0, local_idx)
            selected_adv = candidate_adv.index_select(0, local_idx)
        else:
            selected_idx = candidate_idx
            selected_adv = candidate_adv
        scaled = selected_adv / float(self.bw_vmpo_temperature)
        scaled = scaled - scaled.max()
        weights = torch.softmax(scaled, dim=0).detach()
        return selected_idx, weights

    def _current_value_stage_id(self, stage_id: int) -> int:
        stage_id_i = int(stage_id)
        if stage_id_i not in {0, 1, 2}:
            raise ValueError(f"Unsupported stage_id={stage_id_i}")
        return stage_id_i

    def _stage_value_callable(self, stage_id: int):
        """Return a stage-specific value function without a traced stage branch.

        `torch.compile(fullgraph=True)` is sensitive to Python stage dispatch
        inside the compiled function.  Build one direct callable per stage so
        Dynamo never has to trace `_current_value_stage_id()` or the
        accel/sat/BW branch during value evaluation.
        """

        stage_id_i = self._current_value_stage_id(int(stage_id))
        if stage_id_i == 0 and hasattr(self.critic, "value_accel"):
            value_fn = self.critic.value_accel

            def _value_accel(batch: Any) -> torch.Tensor:
                return value_fn(batch).reshape(-1)

            return _value_accel
        if stage_id_i == 1 and hasattr(self.critic, "value_sat"):
            value_fn = self.critic.value_sat

            def _value_sat(batch: Any) -> torch.Tensor:
                return value_fn(batch).reshape(-1)

            return _value_sat
        if stage_id_i == 2 and hasattr(self.critic, "value_bw"):
            value_fn = self.critic.value_bw

            def _value_bw(batch: Any) -> torch.Tensor:
                return value_fn(batch).reshape(-1)

            return _value_bw

        stage_name = {0: "accel", 1: "sat", 2: "bw"}[stage_id_i]

        def _value_from_full_forward(batch: Any) -> torch.Tensor:
            outputs = self.critic(batch)
            if not isinstance(outputs, dict) or stage_name not in outputs:
                raise RuntimeError(f"critic did not return value head {stage_name!r}.")
            return outputs[stage_name].reshape(-1)

        return _value_from_full_forward

    def _evaluate_world_batch_for_stage(
        self,
        stage_id: int,
        world_batch: Any,
    ) -> torch.Tensor:
        stage_id_i = self._current_value_stage_id(int(stage_id))
        if bool(getattr(self, "critic_compile_enabled", False)) and self.device.type == "cuda":
            compiled = self._compiled_stage_value_fns.get(stage_id_i)
            if compiled is None:
                if not hasattr(torch, "compile"):
                    raise RuntimeError("critic_compile_enabled=True requires torch.compile, but this torch build has none.")
                eager_value_fn = self._stage_value_callable(stage_id_i)

                compiled = torch.compile(
                    eager_value_fn,
                    fullgraph=bool(getattr(self, "critic_compile_fullgraph", True)),
                    options={
                        # The critic is evaluated before and after optimizer
                        # steps in the same training loop. Inductor's CUDA
                        # graph replay can keep stale graph output/input
                        # storage across those calls, so keep fullgraph
                        # compilation but disable cudagraph replay here.
                        "triton.cudagraphs": False,
                        "triton.cudagraph_trees": False,
                    },
                )
                self._compiled_stage_value_fns[stage_id_i] = compiled
            outer_grad_enabled = bool(torch.is_grad_enabled())
            with torch.enable_grad():
                value = compiled(world_batch).to(self.device).reshape(-1)
            return value if outer_grad_enabled else value.detach()
        return self._evaluate_world_batch_for_stage_eager(stage_id_i, world_batch)

    def _evaluate_world_batch_for_stage_eager(
        self,
        stage_id: int,
        world_batch: Any,
    ) -> torch.Tensor:
        stage_id_i = self._current_value_stage_id(int(stage_id))
        if stage_id_i == 0 and hasattr(self.critic, "value_accel"):
            return self.critic.value_accel(world_batch).reshape(-1)
        if stage_id_i == 1 and hasattr(self.critic, "value_sat"):
            return self.critic.value_sat(world_batch).reshape(-1)
        if stage_id_i == 2 and hasattr(self.critic, "value_bw"):
            return self.critic.value_bw(world_batch).reshape(-1)
        outputs = self.critic(world_batch)
        stage_name = {0: "accel", 1: "sat", 2: "bw"}[stage_id_i]
        if not isinstance(outputs, dict) or stage_name not in outputs:
            raise RuntimeError(f"critic did not return value head {stage_name!r}.")
        return outputs[stage_name].reshape(-1)

    def _stage_value_eval_from_batch(
        self,
        stage_id: int,
        world_batch: Any,
    ) -> torch.Tensor:
        return self._evaluate_world_batch_for_stage(stage_id, world_batch).to(self.device).reshape(-1)

    def _rollout_value_override_from_training_view(
        self,
        batch_view: Any,
    ) -> torch.Tensor:
        value_vector = torch.zeros(
            (int(batch_view.transition_count),),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            for stage_id in (0, 1, 2):
                stage_batch = batch_view.stage_batches.get(stage_id)
                if stage_batch is None:
                    continue
                stage_idx = torch.as_tensor(
                    np.asarray(stage_batch.transition_indices, dtype=np.int64),
                    dtype=torch.long,
                    device=self.device,
                )
                stage_values = self._stage_value_eval_microbatched_from_stage_batch(stage_id, stage_batch)
                value_vector.index_copy_(0, stage_idx, stage_values.to(self.device, dtype=torch.float32))
        return value_vector

    def _stage_value_eval_microbatched_from_stage_batch(
        self,
        stage_id: int,
        stage_batch: Any,
    ) -> torch.Tensor:
        sample_count = int(getattr(stage_batch, "num_samples", 0) or 0)
        if sample_count <= 0:
            return torch.empty((0,), dtype=torch.float32, device=self.device)
        micro_size = int(getattr(self, "rollout_value_eval_microbatch_size", 0) or 0)
        if micro_size <= 0 or sample_count <= micro_size:
            return self._stage_value_eval_from_batch(stage_id, stage_batch.world_batch)

        values = torch.empty((sample_count,), dtype=torch.float32, device=self.device)
        for start in range(0, sample_count, micro_size):
            end = min(start + micro_size, sample_count)
            rel_idx = torch.arange(start, end, dtype=torch.long, device=self.device)
            world_batch_mb = _index_dataclass(stage_batch.world_batch, rel_idx)
            if (end - start) == micro_size:
                value_mb = self._stage_value_eval_from_batch(stage_id, world_batch_mb)
            else:
                # Strict torch.compile rejects the final shorter remainder batch
                # after seeing the fixed-size microbatches. Eager eval keeps the
                # same value semantics without triggering a shape recompile.
                value_mb = self._evaluate_world_batch_for_stage_eager(stage_id, world_batch_mb)
            values[start:end].copy_(value_mb.to(device=self.device, dtype=torch.float32).reshape(-1))
        return values

    def _refresh_actor_old_logprobs_from_training_view(self, batch_view: Any) -> None:
        """Recompute PPO old log-probs with the PyTorch actor used for update.

        Native rollout stores old log-probs for speed, but the PPO ratio is
        differentiated through the PyTorch actor. Keeping both sides in the
        same numerical parameterization avoids small native-vs-PyTorch ratio
        offsets before the first optimizer step.
        """
        if batch_view is None or not hasattr(batch_view, "stage_batches"):
            return
        old_vector = torch.zeros(
            (int(batch_view.transition_count),),
            dtype=torch.float32,
            device=self.device,
        )
        if torch.is_tensor(getattr(batch_view, "old_logprobs", None)):
            old_vector = batch_view.old_logprobs.to(device=self.device, dtype=torch.float32).clone()
        with torch.no_grad():
            for stage_id in (0, 1, 2):
                if not self.train_actor_stage.get(int(stage_id), True):
                    continue
                stage_batch = batch_view.stage_batches.get(stage_id)
                if stage_batch is None or int(stage_batch.num_samples) <= 0:
                    continue
                stage_id_i = int(stage_id)
                num_samples = int(stage_batch.num_samples)
                num_agents = max(int(stage_batch.num_agents), 1)
                stage_idx = torch.as_tensor(
                    np.asarray(stage_batch.transition_indices, dtype=np.int64),
                    dtype=torch.long,
                    device=self.device,
                )
                sample_idx_all = torch.arange(num_samples, device=self.device, dtype=torch.long)
                stage_old = torch.empty((num_samples,), dtype=torch.float32, device=self.device)
                per_agent_old = (
                    torch.empty((num_samples, num_agents), dtype=torch.float32, device=self.device)
                    if (
                        stage_id_i in {1, 2}
                        and getattr(stage_batch, "old_logprobs_per_agent", None) is not None
                    )
                    else None
                )
                for micro_idx in self._iter_actor_microbatches(sample_idx_all, stage_id=stage_id_i):
                    flat_idx = self._stage_flat_local_indices(micro_idx, num_agents)
                    local_batch_mb = _index_dataclass(stage_batch.local_batch, flat_idx)
                    actions_mb = stage_batch.actions.index_select(0, micro_idx)
                    latent_mb = None
                    if stage_id_i == 0:
                        latent_all = getattr(stage_batch, "latent_actions", None)
                        if latent_all is None:
                            raise RuntimeError(
                                "accel PPO old-logprob refresh requires latent_actions; "
                                "recollect rollout with accel latent history enabled."
                            )
                        latent_mb = latent_all.index_select(0, micro_idx)
                    logprob, _entropy, actor_out = self._stage_actor_eval_from_batch(
                        stage_id_i,
                        local_batch_mb,
                        actions_mb,
                        num_agents,
                        latent_actions=latent_mb,
                    )
                    logprob = logprob.detach().to(device=self.device, dtype=torch.float32).reshape(-1)
                    stage_old.index_copy_(0, micro_idx, logprob)
                    if per_agent_old is not None and getattr(actor_out, "logprob", None) is not None:
                        per_agent_old.index_copy_(
                            0,
                            micro_idx,
                            actor_out.logprob.detach()
                            .reshape(int(micro_idx.numel()), num_agents)
                            .to(device=self.device, dtype=torch.float32),
                        )
                parity_enabled = bool(getattr(self.cfg, "stage_actor_logprob_parity_check_enabled", True))
                if parity_enabled:
                    stage_name = {0: "accel", 1: "sat", 2: "bw"}[stage_id_i]
                    abs_tol = max(float(getattr(self.cfg, "stage_actor_logprob_parity_abs_tol", 1.0e-3) or 0.0), 0.0)
                    rel_tol = max(float(getattr(self.cfg, "stage_actor_logprob_parity_rel_tol", 1.0e-4) or 0.0), 0.0)

                    def _assert_logprob_parity(label: str, stored: torch.Tensor, replay: torch.Tensor) -> None:
                        stored_t = stored.to(device=self.device, dtype=torch.float32).reshape(-1)
                        replay_t = replay.to(device=self.device, dtype=torch.float32).reshape(-1)
                        if int(stored_t.numel()) != int(replay_t.numel()):
                            raise RuntimeError(
                                f"{stage_name} old-logprob parity shape mismatch for {label}: "
                                f"stored={tuple(stored_t.shape)} replay={tuple(replay_t.shape)}."
                            )
                        diff = replay_t.detach() - stored_t.detach()
                        abs_diff = diff.abs()
                        scale = torch.maximum(replay_t.detach().abs(), stored_t.detach().abs())
                        allowed = float(abs_tol) + float(rel_tol) * scale
                        bad = abs_diff > allowed
                        if bool(bad.any().detach().cpu().item()):
                            worst = int(abs_diff.argmax().detach().cpu().item())
                            stored_v = float(stored_t[worst].detach().cpu().item())
                            replay_v = float(replay_t[worst].detach().cpu().item())
                            diff_v = float(diff[worst].detach().cpu().item())
                            allowed_v = float(allowed[worst].detach().cpu().item())
                            abs_mean = float(abs_diff.mean().detach().cpu().item()) if int(abs_diff.numel()) else 0.0
                            abs_max = float(abs_diff.max().detach().cpu().item()) if int(abs_diff.numel()) else 0.0
                            bad_frac = float(bad.to(dtype=torch.float32).mean().detach().cpu().item()) if int(bad.numel()) else 0.0
                            raise RuntimeError(
                                f"{stage_name} old-logprob parity check failed for {label}: "
                                f"sample={worst}, stored/native={stored_v:.9g}, replay/torch={replay_v:.9g}, "
                                f"diff={diff_v:.9g}, allowed={allowed_v:.9g}, "
                                f"abs_mean={abs_mean:.9g}, abs_max={abs_max:.9g}, bad_frac={bad_frac:.9g}, "
                                f"abs_tol={abs_tol:.3g}, rel_tol={rel_tol:.3g}. "
                                "Do not bypass this by choosing one logprob source; fix the stage action/mask/logprob parity."
                            )

                    _assert_logprob_parity("stage", stage_batch.old_logprobs, stage_old)
                    if per_agent_old is not None and getattr(stage_batch, "old_logprobs_per_agent", None) is not None:
                        _assert_logprob_parity("per_agent", stage_batch.old_logprobs_per_agent, per_agent_old)
                old_vector.index_copy_(0, stage_idx, stage_old)
                stage_batch.old_logprobs = stage_old
                if per_agent_old is not None:
                    stage_batch.old_logprobs_per_agent = per_agent_old
        batch_view.old_logprobs = old_vector

    def _apply_rollout_value_override_to_views(
        self,
        *,
        batch_view: Any,
        return_view: Any,
        value_override: torch.Tensor | None,
    ) -> None:
        """Expose the update-time critic values through the rollout views.

        Native rollout does not run the PyTorch critic in the hot path. When
        update recomputes values for GAE, also write them into the view objects
        so diagnostics and metrics do not observe stale zero placeholders.
        """
        if value_override is None:
            return
        values = value_override.detach().to(device=self.device, dtype=torch.float32).reshape(-1)
        if batch_view is not None and hasattr(batch_view, "values"):
            batch_view.values = values
            for stage_id in (0, 1, 2):
                stage_batch = batch_view.stage_batches.get(stage_id) if hasattr(batch_view, "stage_batches") else None
                if stage_batch is None:
                    continue
                stage_idx = torch.as_tensor(
                    np.asarray(stage_batch.transition_indices, dtype=np.int64),
                    dtype=torch.long,
                    device=self.device,
                )
                stage_batch.values = values.index_select(0, stage_idx)
        if return_view is not None and hasattr(return_view, "values"):
            values_np = values.detach().cpu().numpy().astype(np.float32, copy=False)
            return_view.values = values_np
            for stage_id in (0, 1, 2):
                stage_batch = return_view.stage_batches.get(stage_id) if hasattr(return_view, "stage_batches") else None
                if stage_batch is None:
                    continue
                stage_idx_np = np.asarray(stage_batch.transition_indices, dtype=np.int64)
                stage_batch.values = values_np[stage_idx_np]

    def compute_returns_and_advantages(
        self,
        buffer: StructuredRolloutBuffer,
        bootstrap_world_state: Any = None,
        *,
        return_view: Any | None = None,
        value_override: np.ndarray | torch.Tensor | None = None,
    ) -> dict[str, np.ndarray]:
        view = buffer.build_return_view() if return_view is None else return_view
        bootstrap_stage_id = 2 if self.step_bootstrap_stage == "bw" else 0
        bootstrap_values: dict[int, float] = {}
        if bootstrap_world_state is not None:
            if hasattr(bootstrap_world_state, "next_world_batch") and hasattr(bootstrap_world_state, "env_indices"):
                env_indices = np.asarray(bootstrap_world_state.env_indices, dtype=np.int64).reshape(-1)
                if env_indices.size > 0:
                    with torch.no_grad():
                        values = self._evaluate_world_batch_for_stage(
                            bootstrap_stage_id,
                            bootstrap_world_state.next_world_batch,
                        ).detach().to("cpu", dtype=torch.float32).numpy().reshape(-1)
                    bootstrap_values = {
                        int(env_idx): float(values[pos])
                        for pos, env_idx in enumerate(env_indices.tolist())
                        if pos < int(values.shape[0])
                    }
            elif isinstance(bootstrap_world_state, dict) and bootstrap_world_state:
                env_items = sorted((int(env_idx), state) for env_idx, state in bootstrap_world_state.items())
                env_indices = np.asarray([item[0] for item in env_items], dtype=np.int64)
                world_batch = _collate_dataclass([item[1] for item in env_items], self.device)
                with torch.no_grad():
                    values = self._evaluate_world_batch_for_stage(
                        bootstrap_stage_id,
                        world_batch,
                    ).detach().to("cpu", dtype=torch.float32).numpy().reshape(-1)
                bootstrap_values = {
                    int(env_idx): float(values[pos])
                    for pos, env_idx in enumerate(env_indices.tolist())
                    if pos < int(values.shape[0])
                }
        truncated_bootstrap_values: dict[int, float] = {}
        bw_stage_batch = view.stage_batches.get(2) if hasattr(view, "stage_batches") else None
        if (
            bool(getattr(self, "time_limit_bootstrap_enabled", False))
            and bw_stage_batch is not None
            and int(bw_stage_batch.num_samples) > 0
        ):
            truncated = np.asarray(bw_stage_batch.truncated, dtype=bool).reshape(-1)
            terminated = np.asarray(bw_stage_batch.terminated, dtype=bool).reshape(-1)
            timeout_mask = truncated & ~terminated
            if bool(np.any(timeout_mask)):
                with torch.no_grad():
                    values = self._evaluate_world_batch_for_stage(
                        bootstrap_stage_id,
                        bw_stage_batch.next_world_batch,
                    ).detach().to("cpu", dtype=torch.float32).numpy().reshape(-1)
                transition_indices = np.asarray(bw_stage_batch.transition_indices, dtype=np.int64).reshape(-1)
                for pos in np.flatnonzero(timeout_mask).tolist():
                    if pos < int(values.shape[0]):
                        truncated_bootstrap_values[int(transition_indices[pos])] = float(values[pos])
        return buffer.compute_gae(
            gamma_env=float(self.gamma),
            gae_lambda=float(self.gae_lambda),
            bootstrap_value=0.0,
            bootstrap_values=bootstrap_values,
            truncated_bootstrap_values=truncated_bootstrap_values,
            mode=str(self.target_mode),
            step_target_mode=str(self.step_train_target_mode),
            bw_target_mode=str(self.bw_train_target_mode),
            bw_return_mode=str(self.bw_return_mode),
            bw_reward_w_access=float(self.bw_reward_w_access),
            bw_nstep_horizon=int(self.bw_nstep_horizon),
            return_view=view,
            value_override=value_override,
            bootstrap_truncated=bool(getattr(self, "time_limit_bootstrap_enabled", False)),
        )

    def _stage_actor_eval_from_batch(
        self,
        stage_id: int,
        local_batch: Any,
        joint_actions: torch.Tensor,
        num_agents: int,
        *,
        compute_entropy: bool = True,
        latent_actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        stage_id_i = int(stage_id)
        num_agents_i = max(int(num_agents), 1)
        sample_count = int(joint_actions.shape[0])
        if stage_id_i == 0:
            flat_action = joint_actions.reshape(sample_count * num_agents_i, -1).to(self.device)
            actor_out = self.actor.evaluate_accel(
                local_batch,
                flat_action,
                compute_entropy=bool(compute_entropy),
                latent_action=(
                    None
                    if latent_actions is None
                    else latent_actions.reshape(sample_count * num_agents_i, -1).to(self.device)
                ),
            )
            logprob = actor_out.logprob.reshape(sample_count, num_agents_i).sum(dim=1)
            entropy = actor_out.entropy.reshape(sample_count, num_agents_i).sum(dim=1)
            return logprob, entropy, actor_out
        if stage_id_i == 1:
            flat_action = joint_actions.reshape(sample_count * num_agents_i).to(self.device, dtype=torch.long)
            actor_out = self.actor.evaluate_sat(
                local_batch,
                flat_action,
                compute_entropy=bool(compute_entropy),
            )
            logprob = actor_out.logprob.reshape(sample_count, num_agents_i).sum(dim=1)
            entropy = actor_out.entropy.reshape(sample_count, num_agents_i).sum(dim=1)
            return logprob, entropy, actor_out
        if stage_id_i == 2:
            flat_action = joint_actions.reshape(sample_count * num_agents_i, -1).to(self.device)
            actor_out = self.actor.evaluate_bw(
                local_batch,
                flat_action,
                compute_entropy=bool(compute_entropy),
            )
            logprob_per_agent = actor_out.logprob.reshape(sample_count, num_agents_i)
            entropy_per_agent = actor_out.entropy.reshape(sample_count, num_agents_i)
            valid_count = None if actor_out.valid_count is None else actor_out.valid_count.reshape(sample_count, num_agents_i)
            latent_count = None if actor_out.latent_count is None else actor_out.latent_count.reshape(sample_count, num_agents_i)
            entropy_per_agent = self._normalize_bw_entropy(
                entropy_per_agent,
                actor_out=actor_out,
                valid_count=valid_count,
                latent_count=latent_count,
            )
            if self.bw_single_uav_policy_enabled:
                target_uav = int(self.bw_single_uav_policy_uav_id)
                if target_uav < 0 or target_uav >= num_agents_i:
                    raise ValueError(
                        f"bw_single_uav_policy_uav_id={target_uav} is out of range for num_agents={num_agents_i}."
                    )
                return logprob_per_agent[:, target_uav], entropy_per_agent[:, target_uav], actor_out
            return logprob_per_agent.sum(dim=1), entropy_per_agent.sum(dim=1), actor_out
        raise ValueError(f"Unsupported stage_id={stage_id_i}")

    def _actor_eval_microbatch_size_for_stage(self, stage_id: int, sample_count: int) -> int:
        del stage_id
        configured = int(getattr(self, "actor_update_microbatch_size", 0) or 0)
        if configured <= 0:
            return max(int(sample_count), 1)
        return max(min(configured, max(int(sample_count), 1)), 1)

    def _stage_flat_local_indices(self, sample_idx: torch.Tensor, num_agents: int) -> torch.Tensor:
        sample_idx = sample_idx.to(device=self.device, dtype=torch.long).reshape(-1)
        num_agents_i = max(int(num_agents), 1)
        agent_offsets = torch.arange(num_agents_i, device=self.device, dtype=torch.long)
        return (sample_idx[:, None] * num_agents_i + agent_offsets[None, :]).reshape(-1)

    def _iter_actor_microbatches(self, mb_rel: torch.Tensor, *, stage_id: int) -> list[torch.Tensor]:
        mb_rel = mb_rel.to(device=self.device, dtype=torch.long).reshape(-1)
        total = int(mb_rel.numel())
        if total <= 0:
            return []
        micro_size = self._actor_eval_microbatch_size_for_stage(stage_id, total)
        if micro_size >= total:
            return [mb_rel]
        return [mb_rel[start : start + micro_size] for start in range(0, total, micro_size)]

    @staticmethod
    def _zero_metric_tensor(device: torch.device) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32, device=device)

    def _run_stage_actor_update_minibatch(
        self,
        *,
        stage_id: int,
        cache: dict[str, Any],
        mb_rel: torch.Tensor,
    ) -> dict[str, float] | None:
        """Run one PPO actor minibatch with memory microbatches.

        This preserves PPO minibatch semantics: policy/entropy terms are
        weighted by sample count and all microbatch gradients are accumulated
        before a single optimizer step.
        """
        stage_id_i = int(stage_id)
        mb_rel = mb_rel.to(device=self.device, dtype=torch.long).reshape(-1)
        total_samples = int(mb_rel.numel())
        if total_samples <= 0:
            return None
        if stage_id_i == 2 and self.bw_policy_update_mode == "vmpo_lite":
            # VMPO-lite selects samples within the minibatch, so chunking would
            # change the selected set. Keep its original whole-minibatch path.
            microbatches = [mb_rel]
        else:
            microbatches = self._iter_actor_microbatches(mb_rel, stage_id=stage_id_i)
        if not microbatches:
            return None

        total_active_danger = 0.0
        if self.danger_imitation_enabled and stage_id_i == 0 and "danger_masks" in cache:
            danger_masks_total = cache["danger_masks"].index_select(0, mb_rel).to(device=self.device, dtype=torch.float32)
            active_total = torch.sum(danger_masks_total.reshape(total_samples, max(int(cache["num_agents"]), 1), -1), dim=-1) > 0.0
            total_active_danger = float(active_total.to(dtype=torch.float32).sum().item())

        total_proxy_mask = 0.0
        if self.bw_flow_proxy_aux_enabled and stage_id_i == 2 and torch.is_tensor(cache.get("bw_flow_proxy_masks", None)):
            proxy_masks_total = cache["bw_flow_proxy_masks"].index_select(0, mb_rel).to(device=self.device)
            total_proxy_mask = float((proxy_masks_total > 0.5).to(dtype=torch.float32).sum().item())

        metric_sums: dict[str, float] = {
            "policy_loss": 0.0,
            "entropy_mean": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
            "danger_imitation_loss": 0.0,
            "bw_flow_proxy_aux_loss": 0.0,
            "bw_flow_proxy_regression_loss": 0.0,
            "bw_flow_proxy_pairwise_acc": 0.0,
            "bw_flow_proxy_pair_count": 0.0,
            "bw_abs_log_ratio_corr_valid_count": 0.0,
            "bw_abs_log_ratio_corr_latent_count": 0.0,
            "bw_kappa_mean": 0.0,
            "bw_kappa_p10": 0.0,
            "bw_kappa_p90": 0.0,
            "bw_kappa_hi_frac": 0.0,
        }
        did_backward = False
        self.actor_optimizer.zero_grad()
        for micro_rel in microbatches:
            micro_count = int(micro_rel.numel())
            if micro_count <= 0:
                continue
            sample_weight = float(micro_count) / float(total_samples)
            flat_mb = cache["flat_indices"].index_select(0, micro_rel).reshape(-1)
            local_batch_mb = _index_dataclass(cache["local_batch"], flat_mb)
            joint_actions_mb = cache["joint_actions"].index_select(0, micro_rel)
            latent_actions_mb = None
            if stage_id_i == 0:
                latent_all = cache.get("latent_actions")
                if latent_all is None:
                    raise RuntimeError(
                        "accel PPO actor update requires latent_actions so the ratio uses pre-squash z samples."
                    )
                latent_actions_mb = latent_all.index_select(0, micro_rel)
            new_logprob, entropy, actor_out = self._stage_actor_eval_from_batch(
                stage_id_i,
                local_batch_mb,
                joint_actions_mb,
                cache["num_agents"],
                compute_entropy=abs(float(self.entropy_coef_by_stage[stage_id_i])) > 0.0,
                latent_actions=latent_actions_mb,
            )
            (
                policy_loss,
                entropy_mean,
                approx_kl,
                clip_frac,
                bw_abs_log_ratio_corr_valid_count,
                bw_abs_log_ratio_corr_latent_count,
                bw_kappa_mean,
                bw_kappa_p10,
                bw_kappa_p90,
                bw_kappa_hi_frac,
            ) = self._stage_policy_terms(
                stage_id_i,
                actor_out=actor_out,
                cache=cache,
                mb_rel=micro_rel,
                joint_actions_mb=joint_actions_mb,
                new_logprob=new_logprob,
                entropy=entropy,
            )

            danger_imitation_loss = self._zero_metric_tensor(self.device)
            danger_loss_weight = 0.0
            if self.danger_imitation_enabled and stage_id_i == 0 and actor_out is not None:
                target_accel = cache["danger_targets"].index_select(0, micro_rel)
                danger_mask = cache["danger_masks"].index_select(0, micro_rel)
                target_shape = (micro_count, max(int(cache["num_agents"]), 1), 2)
                danger_mask_reshaped = danger_mask.reshape(target_shape).to(device=self.device, dtype=torch.float32)
                active = torch.sum(danger_mask_reshaped, dim=-1) > 0.0
                active_count = float(active.to(dtype=torch.float32).sum().item())
                if active_count > 0.0:
                    target_accel = target_accel.reshape(target_shape).to(device=self.device, dtype=torch.float32)
                    pred_accel = squash_action(actor_out.mean).reshape_as(target_accel)
                    diff = (pred_accel - target_accel) * danger_mask_reshaped
                    denom = torch.sum(danger_mask_reshaped, dim=-1).clamp_min(1.0)
                    per_agent = diff.pow(2).sum(dim=-1) / denom
                    danger_imitation_loss = per_agent[active].mean()
                    danger_loss_weight = active_count / max(total_active_danger, 1.0)

            bw_flow_proxy_aux_loss = self._zero_metric_tensor(self.device)
            bw_flow_proxy_regression_loss = self._zero_metric_tensor(self.device)
            bw_flow_proxy_pairwise_acc = self._zero_metric_tensor(self.device)
            bw_flow_proxy_pair_count = self._zero_metric_tensor(self.device)
            bw_aux_weight = sample_weight
            if self.bw_flow_proxy_aux_enabled and stage_id_i == 2 and actor_out is not None:
                proxy_scores_mb = cache["bw_flow_proxy_scores"].index_select(0, micro_rel).reshape(
                    -1,
                    joint_actions_mb.shape[-1],
                )
                proxy_masks_mb = cache["bw_flow_proxy_masks"].index_select(0, micro_rel).reshape(
                    -1,
                    joint_actions_mb.shape[-1],
                )
                (
                    bw_flow_proxy_aux_loss,
                    bw_flow_proxy_pairwise_acc,
                    bw_flow_proxy_pair_count,
                    bw_flow_proxy_regression_loss,
                ) = self._bw_flow_proxy_aux_loss(
                    local_batch_mb,
                    actor_out,
                    proxy_scores_mb,
                    proxy_masks_mb,
                )
                if total_proxy_mask > 0.0:
                    micro_proxy_mask = float((proxy_masks_mb > 0.5).to(dtype=torch.float32).sum().item())
                    bw_aux_weight = micro_proxy_mask / max(total_proxy_mask, 1.0)

            loss = (
                sample_weight * policy_loss
                - sample_weight * self.entropy_coef_by_stage[stage_id_i] * entropy_mean
                + float(self.danger_imitation_coef) * danger_loss_weight * danger_imitation_loss
                + float(self.bw_flow_proxy_aux_coef) * bw_aux_weight * bw_flow_proxy_aux_loss
            )
            if loss.requires_grad:
                loss.backward()
                did_backward = True

            metric_sums["policy_loss"] += sample_weight * float(policy_loss.detach().item())
            metric_sums["entropy_mean"] += sample_weight * float(entropy_mean.detach().item())
            metric_sums["approx_kl"] += sample_weight * float(approx_kl.detach().item())
            metric_sums["clip_frac"] += sample_weight * float(clip_frac.detach().item())
            metric_sums["danger_imitation_loss"] += danger_loss_weight * float(danger_imitation_loss.detach().item())
            metric_sums["bw_flow_proxy_aux_loss"] += bw_aux_weight * float(bw_flow_proxy_aux_loss.detach().item())
            metric_sums["bw_flow_proxy_regression_loss"] += bw_aux_weight * float(bw_flow_proxy_regression_loss.detach().item())
            metric_sums["bw_flow_proxy_pairwise_acc"] += sample_weight * float(bw_flow_proxy_pairwise_acc.detach().item())
            metric_sums["bw_flow_proxy_pair_count"] += float(bw_flow_proxy_pair_count.detach().item())
            metric_sums["bw_abs_log_ratio_corr_valid_count"] += sample_weight * float(bw_abs_log_ratio_corr_valid_count.detach().item())
            metric_sums["bw_abs_log_ratio_corr_latent_count"] += sample_weight * float(bw_abs_log_ratio_corr_latent_count.detach().item())
            metric_sums["bw_kappa_mean"] += sample_weight * float(bw_kappa_mean.detach().item())
            metric_sums["bw_kappa_p10"] += sample_weight * float(bw_kappa_p10.detach().item())
            metric_sums["bw_kappa_p90"] += sample_weight * float(bw_kappa_p90.detach().item())
            metric_sums["bw_kappa_hi_frac"] += sample_weight * float(bw_kappa_hi_frac.detach().item())

        if not did_backward:
            return None
        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        self._sync_native_actor_cuda_bindings_after_update()
        metric_sums["grad_norm"] = float(grad_norm.detach().item()) if torch.is_tensor(grad_norm) else float(grad_norm)
        return metric_sums

    def _stage_actor_eval_per_agent_from_batch(
        self,
        stage_id: int,
        local_batch: Any,
        joint_actions: torch.Tensor,
        num_agents: int,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """Evaluate per-UAV policy terms without summing the joint action.

        The sample-vs-ref objective trains one sampled UAV action against its
        paired deterministic ref, so its loss must gather the target UAV's
        log-prob rather than using the old PPO joint log-prob sum.
        """
        stage_id_i = int(stage_id)
        num_agents_i = max(int(num_agents), 1)
        sample_count = int(joint_actions.shape[0])
        if stage_id_i == 0:
            flat_action = joint_actions.reshape(sample_count * num_agents_i, -1).to(self.device)
            actor_out = self.actor.evaluate_accel(local_batch, flat_action)
            return (
                actor_out.logprob.reshape(sample_count, num_agents_i),
                actor_out.entropy.reshape(sample_count, num_agents_i),
                actor_out,
            )
        if stage_id_i == 1:
            flat_action = joint_actions.reshape(sample_count * num_agents_i).to(self.device, dtype=torch.long)
            actor_out = self.actor.evaluate_sat(local_batch, flat_action)
            return (
                actor_out.logprob.reshape(sample_count, num_agents_i),
                actor_out.entropy.reshape(sample_count, num_agents_i),
                actor_out,
            )
        if stage_id_i == 2:
            flat_action = joint_actions.reshape(sample_count * num_agents_i, -1).to(self.device)
            actor_out = self.actor.evaluate_bw(local_batch, flat_action)
            logprob_per_agent = actor_out.logprob.reshape(sample_count, num_agents_i)
            entropy_per_agent = actor_out.entropy.reshape(sample_count, num_agents_i)
            valid_count = None if actor_out.valid_count is None else actor_out.valid_count.reshape(sample_count, num_agents_i)
            latent_count = None if actor_out.latent_count is None else actor_out.latent_count.reshape(sample_count, num_agents_i)
            entropy_per_agent = self._normalize_bw_entropy(
                entropy_per_agent,
                actor_out=actor_out,
                valid_count=valid_count,
                latent_count=latent_count,
            )
            return logprob_per_agent, entropy_per_agent, actor_out
        raise ValueError(f"Unsupported stage_id={stage_id_i}")

    def _actor_stage_module(self, stage_id: int) -> torch.nn.Module:
        stage_id_i = int(stage_id)
        if stage_id_i == 0:
            return self.actor.accel_policy
        if stage_id_i == 1:
            return self.actor.sat_subset_policy
        if stage_id_i == 2:
            return self.actor.bw_policy
        raise ValueError(f"Unsupported stage_id={stage_id_i}")

    def _actor_stage_optimizer(self, stage_id: int):
        optimizer = self.actor_stage_optimizers.get(int(stage_id))
        return self.actor_optimizer if optimizer is None else optimizer

    def _accel_danger_imitation_loss(
        self,
        *,
        actor_out: Any,
        targets: torch.Tensor | None,
        masks: torch.Tensor | None,
        sample_idx: torch.Tensor,
        selected_count: int,
        num_agents: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero = torch.zeros((), dtype=torch.float32, device=self.device)
        if (
            not self.danger_imitation_enabled
            or actor_out is None
            or not torch.is_tensor(targets)
            or not torch.is_tensor(masks)
            or int(selected_count) <= 0
            or int(num_agents) <= 0
        ):
            return zero, zero
        target_accel = targets.index_select(0, sample_idx).to(device=self.device, dtype=torch.float32)
        danger_mask = masks.index_select(0, sample_idx).to(device=self.device, dtype=torch.float32)
        target_shape = (int(selected_count), int(num_agents), 2)
        target_accel = target_accel.reshape(target_shape)
        danger_mask = danger_mask.reshape(target_shape)
        pred_accel = squash_action(actor_out.mean).reshape_as(target_accel)
        active = torch.sum(danger_mask, dim=-1) > 0.0
        active_rate = active.to(dtype=torch.float32).mean() if int(active.numel()) > 0 else zero
        if not torch.any(active):
            return zero, active_rate
        diff = (pred_accel - target_accel) * danger_mask
        denom = torch.sum(danger_mask, dim=-1).clamp_min(1.0)
        per_agent = diff.pow(2).sum(dim=-1) / denom
        return per_agent[active].mean(), active_rate

    @staticmethod
    def _vs_ref_stage_name(stage_id: int) -> str:
        return ("accel", "sat", "bw")[int(stage_id)]

    def _vs_ref_sampling_quotas(self, budget: int) -> dict[str, int]:
        budget_i = max(int(budget), 0)
        kinds = ("random", "time", "leverage", "uncertainty")
        if budget_i <= 0:
            return {kind: 0 for kind in kinds}
        fracs = np.asarray(
            [
                float(self.vs_ref_sampling_random_frac),
                float(self.vs_ref_sampling_time_frac),
                float(self.vs_ref_sampling_leverage_frac),
                float(self.vs_ref_sampling_uncertainty_frac),
            ],
            dtype=np.float64,
        )
        if not np.isfinite(fracs).all() or float(np.sum(fracs)) <= 0.0:
            return {"random": budget_i, "time": 0, "leverage": 0, "uncertainty": 0}
        scaled = fracs / float(np.sum(fracs)) * float(budget_i)
        quotas = np.floor(scaled).astype(np.int64)
        remainder = int(budget_i - int(np.sum(quotas)))
        if remainder > 0:
            order = np.argsort(-(scaled - quotas))
            for idx in order[:remainder]:
                quotas[int(idx)] += 1
        return {kind: int(value) for kind, value in zip(kinds, quotas.tolist())}

    def _sample_vs_ref_rows_uniform(
        self,
        *,
        flat_total: int,
        budget: int,
        num_agents: int,
        rng_device: torch.device,
        remaining_by_candidate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        row_budget = min(max(int(budget), 0), max(int(flat_total), 0))
        if row_budget <= 0:
            empty = torch.empty((0,), dtype=torch.long, device=self.device)
            return empty, empty, {}
        perm = torch.randperm(int(flat_total), device=rng_device)[:row_budget].to(device=self.device, dtype=torch.long)
        stats: dict[str, float] = {
            "random_count": float(row_budget),
            "time_count": 0.0,
            "leverage_count": 0.0,
            "uncertainty_count": 0.0,
            "priority_mean": 0.0,
            "priority_max": 0.0,
        }
        if torch.is_tensor(remaining_by_candidate) and int(remaining_by_candidate.numel()) == int(flat_total):
            horizon = remaining_by_candidate.index_select(0, perm).to(dtype=torch.float32)
            stats.update(
                {
                    "horizon_mean": float(horizon.mean().detach().item()) if horizon.numel() > 0 else 0.0,
                    "horizon_min": float(horizon.min().detach().item()) if horizon.numel() > 0 else 0.0,
                    "horizon_max": float(horizon.max().detach().item()) if horizon.numel() > 0 else 0.0,
                }
            )
        sample_idx_t = torch.div(perm, int(num_agents), rounding_mode="floor").to(dtype=torch.long)
        target_uav_t = torch.remainder(perm, int(num_agents)).to(dtype=torch.long)
        return sample_idx_t, target_uav_t, stats

    def _sample_vs_ref_rows_active_mixture(
        self,
        *,
        stage_id: int,
        stage_batch: StructuredStageTrainingBatch,
        flat_total: int,
        budget: int,
        num_samples: int,
        num_agents: int,
        remaining_by_candidate: torch.Tensor,
        source_step_by_candidate: torch.Tensor,
        env_reward_by_sample: torch.Tensor,
        rng_device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        row_budget = min(max(int(budget), 0), max(int(flat_total), 0))
        if row_budget <= 0:
            empty = torch.empty((0,), dtype=torch.long, device=self.device)
            return empty, empty, {}

        quotas = self._vs_ref_sampling_quotas(row_budget)
        selected_mask = torch.zeros((int(flat_total),), dtype=torch.bool, device=self.device)
        selected_parts: list[torch.Tensor] = []
        counts = {key: 0 for key in ("random", "time", "leverage", "uncertainty")}

        def _available_indices() -> torch.Tensor:
            return torch.nonzero(~selected_mask, as_tuple=False).reshape(-1)

        def _add(chosen: torch.Tensor, kind: str) -> None:
            if not torch.is_tensor(chosen) or int(chosen.numel()) <= 0:
                return
            chosen = chosen.to(device=self.device, dtype=torch.long).reshape(-1)
            chosen = chosen[(chosen >= 0) & (chosen < int(flat_total))]
            if int(chosen.numel()) <= 0:
                return
            chosen = chosen[~selected_mask.index_select(0, chosen)]
            if int(chosen.numel()) <= 0:
                return
            selected_mask.index_fill_(0, chosen, True)
            selected_parts.append(chosen)
            counts[kind] = counts.get(kind, 0) + int(chosen.numel())

        def _uniform(k: int) -> torch.Tensor:
            k_i = max(int(k), 0)
            if k_i <= 0:
                return torch.empty((0,), dtype=torch.long, device=self.device)
            avail = _available_indices()
            if int(avail.numel()) <= 0:
                return torch.empty((0,), dtype=torch.long, device=self.device)
            k_i = min(k_i, int(avail.numel()))
            order = torch.randperm(int(avail.numel()), device=rng_device)[:k_i].to(device=self.device, dtype=torch.long)
            return avail.index_select(0, order)

        def _priority(k: int, score: torch.Tensor) -> torch.Tensor:
            k_i = max(int(k), 0)
            if k_i <= 0:
                return torch.empty((0,), dtype=torch.long, device=self.device)
            avail = _available_indices()
            if int(avail.numel()) <= 0:
                return torch.empty((0,), dtype=torch.long, device=self.device)
            k_i = min(k_i, int(avail.numel()))
            score_avail = score.index_select(0, avail).to(dtype=torch.float32).clamp_min(0.0)
            if float(score_avail.sum().detach().item()) <= 0.0:
                return _uniform(k_i)
            probs = (score_avail + 1.0e-12).pow(float(self.vs_ref_sampling_alpha))
            probs_sum = probs.sum()
            if not torch.isfinite(probs_sum) or float(probs_sum.detach().item()) <= 0.0:
                return _uniform(k_i)
            probs = probs / probs_sum
            rel = torch.multinomial(probs, num_samples=k_i, replacement=False)
            return avail.index_select(0, rel.to(device=self.device, dtype=torch.long))

        _add(_uniform(quotas.get("random", 0)), "random")

        time_quota = max(int(quotas.get("time", 0)), 0)
        if time_quota > 0:
            capacity = max(int(getattr(self.cfg, "buffer_size", 0) or 0), int(source_step_by_candidate.max().detach().item()) + 1, 1)
            bucket = torch.clamp((source_step_by_candidate.to(dtype=torch.long) * 4) // int(capacity), min=0, max=3)
            for offset in range(time_quota):
                b = int(offset % 4)
                avail = _available_indices()
                if int(avail.numel()) <= 0:
                    break
                bucket_avail = avail.index_select(0, torch.nonzero(bucket.index_select(0, avail) == b, as_tuple=False).reshape(-1))
                if int(bucket_avail.numel()) <= 0:
                    _add(_uniform(1), "time")
                    continue
                rel = torch.randint(int(bucket_avail.numel()), (1,), device=rng_device).to(device=self.device)
                _add(bucket_avail.index_select(0, rel.to(dtype=torch.long)), "time")

        reward_by_candidate = env_reward_by_sample.repeat_interleave(int(num_agents)).to(device=self.device, dtype=torch.float32)
        reward_gap = torch.clamp(1.0 - reward_by_candidate, min=0.0, max=1.0)
        horizon_cost = remaining_by_candidate.to(dtype=torch.float32).clamp_min(1.0).pow(float(self.vs_ref_sampling_cost_power))
        leverage_priority = reward_gap / horizon_cost.clamp_min(1.0e-12)

        with torch.no_grad():
            if int(stage_id) == 0:
                ref = self.actor.act_accel(stage_batch.local_batch, deterministic=True).action.reshape(
                    int(num_samples),
                    int(num_agents),
                    -1,
                )
                action = stage_batch.actions.to(device=self.device, dtype=torch.float32).reshape_as(ref)
                uncertainty = torch.linalg.vector_norm(action - ref, dim=-1).reshape(-1)
            elif int(stage_id) == 1:
                ref = self.actor.act_sat(stage_batch.local_batch, deterministic=True).subset_index.reshape(
                    int(num_samples),
                    int(num_agents),
                )
                action = stage_batch.actions.to(device=self.device, dtype=torch.long).reshape_as(ref)
                uncertainty = (action != ref).to(dtype=torch.float32).reshape(-1)
            else:
                ref = self.actor.act_bw(stage_batch.local_batch, deterministic=True).action.reshape(
                    int(num_samples),
                    int(num_agents),
                    -1,
                )
                action = stage_batch.actions.to(device=self.device, dtype=torch.float32).reshape_as(ref)
                uncertainty = torch.sum(torch.abs(action - ref), dim=-1).reshape(-1)
        uncertainty_priority = uncertainty.to(device=self.device, dtype=torch.float32).clamp_min(0.0) / horizon_cost.clamp_min(1.0e-12)

        _add(_priority(quotas.get("leverage", 0), leverage_priority), "leverage")
        _add(_priority(quotas.get("uncertainty", 0), uncertainty_priority), "uncertainty")

        remaining = row_budget - int(torch.count_nonzero(selected_mask).detach().item())
        if remaining > 0:
            _add(_uniform(remaining), "random")

        if selected_parts:
            selected_flat = torch.cat(selected_parts, dim=0)
        else:
            selected_flat = _uniform(row_budget)
            counts["random"] += int(selected_flat.numel())
        selected_flat = selected_flat[:row_budget].to(device=self.device, dtype=torch.long)
        priority_for_stats = torch.maximum(leverage_priority, uncertainty_priority)
        selected_horizon = remaining_by_candidate.index_select(0, selected_flat).to(dtype=torch.float32)
        selected_priority = priority_for_stats.index_select(0, selected_flat).to(dtype=torch.float32)
        stats: dict[str, float] = {
            "random_count": float(counts.get("random", 0)),
            "time_count": float(counts.get("time", 0)),
            "leverage_count": float(counts.get("leverage", 0)),
            "uncertainty_count": float(counts.get("uncertainty", 0)),
            "horizon_mean": float(selected_horizon.mean().detach().item()) if selected_horizon.numel() > 0 else 0.0,
            "horizon_min": float(selected_horizon.min().detach().item()) if selected_horizon.numel() > 0 else 0.0,
            "horizon_max": float(selected_horizon.max().detach().item()) if selected_horizon.numel() > 0 else 0.0,
            "priority_mean": float(selected_priority.mean().detach().item()) if selected_priority.numel() > 0 else 0.0,
            "priority_max": float(selected_priority.max().detach().item()) if selected_priority.numel() > 0 else 0.0,
        }
        sample_idx_t = torch.div(selected_flat, int(num_agents), rounding_mode="floor").to(dtype=torch.long)
        target_uav_t = torch.remainder(selected_flat, int(num_agents)).to(dtype=torch.long)
        return sample_idx_t, target_uav_t, stats

    def _sample_vs_ref_rows(
        self,
        *,
        stage_id: int,
        stage_batch: StructuredStageTrainingBatch,
        history: Any,
        history_rows_np: np.ndarray,
        source_num_envs: int,
        env_reward_by_sample: torch.Tensor | None,
        rng_device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        num_samples = int(stage_batch.num_samples)
        num_agents = max(int(stage_batch.num_agents), 1)
        flat_total = int(num_samples) * int(num_agents)
        row_budget = min(int(self.vs_ref_rows_per_update), flat_total)
        if flat_total <= 0 or row_budget <= 0:
            empty = torch.empty((0,), dtype=torch.long, device=self.device)
            return empty, empty, empty, {}

        history_rows_t = torch.as_tensor(history_rows_np, dtype=torch.long, device=self.device)
        history_state = getattr(history, f"{self._vs_ref_stage_name(int(stage_id))}_runtime_state")
        hist_t = getattr(history_state, "t").index_select(
            0,
            history_rows_t.to(device=getattr(history_state, "t").device, dtype=torch.long),
        ).to(device=self.device, dtype=torch.long)
        source_step_t = torch.div(history_rows_t, int(source_num_envs), rounding_mode="floor")
        t_steps = max(int(getattr(self.cfg, "T_steps", 1) or 1), 1)
        history_capacity = max(int(getattr(history, "capacity", 0) or 0), 1)
        remaining_episode_t = torch.clamp(
            torch.as_tensor(t_steps, device=self.device, dtype=torch.long) - hist_t,
            min=1,
        )
        remaining_tape_t = torch.clamp(
            torch.as_tensor(history_capacity, device=self.device, dtype=torch.long) - source_step_t,
            min=1,
        )
        remaining_t = torch.minimum(remaining_episode_t, remaining_tape_t)
        remaining_by_candidate = remaining_t.repeat_interleave(int(num_agents))
        source_step_by_candidate = source_step_t.repeat_interleave(int(num_agents)).to(device=self.device, dtype=torch.long)

        reward_by_sample = env_reward_by_sample
        if not torch.is_tensor(reward_by_sample) or int(reward_by_sample.numel()) != int(num_samples):
            reward_by_sample = stage_batch.rewards
        reward_by_sample = reward_by_sample.to(device=self.device, dtype=torch.float32).reshape(int(num_samples))

        if self.vs_ref_sampling_mode == "active_mixture":
            sample_idx_t, target_uav_t, stats = self._sample_vs_ref_rows_active_mixture(
                stage_id=int(stage_id),
                stage_batch=stage_batch,
                flat_total=flat_total,
                budget=row_budget,
                num_samples=num_samples,
                num_agents=num_agents,
                remaining_by_candidate=remaining_by_candidate,
                source_step_by_candidate=source_step_by_candidate,
                env_reward_by_sample=reward_by_sample,
                rng_device=rng_device,
            )
        else:
            sample_idx_t, target_uav_t, stats = self._sample_vs_ref_rows_uniform(
                flat_total=flat_total,
                budget=row_budget,
                num_agents=num_agents,
                rng_device=rng_device,
                remaining_by_candidate=remaining_by_candidate,
            )

        if int(self.vs_ref_samples_per_row) > 1 and int(sample_idx_t.numel()) > 0:
            repeat = int(self.vs_ref_samples_per_row)
            sample_idx_t = sample_idx_t.repeat_interleave(repeat)
            target_uav_t = target_uav_t.repeat_interleave(repeat)
        selected_remaining_t = remaining_t.index_select(0, sample_idx_t) if int(sample_idx_t.numel()) > 0 else sample_idx_t
        return sample_idx_t, target_uav_t, selected_remaining_t, stats

    def _update_vs_ref_native(
        self,
        buffer: StructuredRolloutBuffer,
        rollout_views: StructuredRolloutViews,
    ) -> dict[str, float]:
        del buffer
        active_stages = [
            int(stage_id)
            for stage_id in (0, 1, 2)
            if bool(self.train_actor_stage.get(int(stage_id), True))
            and self.stage_actor_update_mode.get(int(stage_id), "ppo") == "vs_ref"
        ]
        if not active_stages:
            return _structured_zero_update_metrics()
        rollout_program = getattr(self, "_native_rollout_program", None)
        runtime = None if rollout_program is None else getattr(rollout_program, "runtime", None)
        history = None if runtime is None else getattr(runtime, "history", None)
        if history is None:
            raise RuntimeError("sample-vs-ref update requires native rollout history snapshots.")
        missing: list[str] = []
        for stage_id in active_stages:
            stage_name = ("accel", "sat", "bw")[int(stage_id)]
            if getattr(history, f"{stage_name}_runtime_state", None) is None:
                missing.append(f"{stage_name}_runtime_state")
            if getattr(history, f"{stage_name}_runtime_stage", None) is None:
                missing.append(f"{stage_name}_runtime_stage")
        if 2 in active_stages and getattr(history, "bw_runtime_cache", None) is None:
            missing.append("bw_runtime_cache")
        if missing:
            raise RuntimeError(
                "sample-vs-ref update requires complete stage runtime snapshots; missing "
                + ", ".join(sorted(set(missing)))
                + "."
            )
        stage_losses: list[tuple[int, torch.Tensor]] = []
        stage_metric_values: dict[str, float] = {}
        branch_rows_total = 0
        adv_values_all: list[torch.Tensor] = []
        entropy_values_all: list[torch.Tensor] = []
        approx_kl_values_all: list[torch.Tensor] = []
        clip_values_all: list[torch.Tensor] = []
        danger_imitation_losses: list[torch.Tensor] = []
        danger_imitation_active_rates: list[torch.Tensor] = []

        source_num_envs = int(getattr(history, "num_envs", 0) or 0)
        if source_num_envs <= 0:
            raise RuntimeError("sample-vs-ref update cannot infer source rollout env count.")
        rng_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        reward_stage_batch = rollout_views.training_view.stage_batches.get(2)
        env_reward_by_sample = (
            reward_stage_batch.rewards
            if reward_stage_batch is not None and torch.is_tensor(reward_stage_batch.rewards)
            else None
        )
        for stage_id in active_stages:
            stage_batch = rollout_views.training_view.stage_batches.get(int(stage_id))
            if stage_batch is None or int(stage_batch.num_samples) <= 0:
                continue
            num_samples = int(stage_batch.num_samples)
            num_agents = max(int(stage_batch.num_agents), 1)
            transition_indices = np.asarray(stage_batch.transition_indices, dtype=np.int64).reshape(-1)
            if int(transition_indices.size) != num_samples:
                raise RuntimeError("sample-vs-ref requires one transition index per stage sample.")
            if np.any((transition_indices - int(stage_id)) % 3 != 0):
                raise RuntimeError("sample-vs-ref stage batch has transition indices inconsistent with stage_id.")
            history_rows_np = ((transition_indices - int(stage_id)) // 3).astype(np.int64, copy=False)
            sample_idx_t, target_uav_t, selected_remaining_t, sampling_stats = self._sample_vs_ref_rows(
                stage_id=int(stage_id),
                stage_batch=stage_batch,
                history=history,
                history_rows_np=history_rows_np,
                source_num_envs=int(source_num_envs),
                env_reward_by_sample=env_reward_by_sample,
                rng_device=rng_device,
            )
            selected_count = int(sample_idx_t.numel())
            if selected_count <= 0:
                continue
            agent_offsets = torch.arange(int(num_agents), dtype=torch.long, device=self.device).view(1, int(num_agents))
            flat_local_rows = (sample_idx_t.view(-1, 1) * int(num_agents) + agent_offsets).reshape(-1)
            selected_local = _index_dataclass(stage_batch.local_batch, flat_local_rows)
            with torch.no_grad():
                if int(stage_id) == 0:
                    ref_out = self.actor.act_accel(selected_local, deterministic=True)
                    sample_out = self.actor.act_accel(selected_local, deterministic=False)
                    ref_joint = ref_out.action.reshape(selected_count, int(num_agents), -1)
                    sample_all = sample_out.action.reshape(selected_count, int(num_agents), -1)
                    old_logprob_all = sample_out.logprob.reshape(selected_count, int(num_agents))
                elif int(stage_id) == 1:
                    ref_out = self.actor.act_sat(selected_local, deterministic=True)
                    sample_out = self.actor.act_sat(selected_local, deterministic=False)
                    ref_joint = ref_out.subset_index.reshape(selected_count, int(num_agents))
                    sample_all = sample_out.subset_index.reshape(selected_count, int(num_agents))
                    old_logprob_all = sample_out.logprob.reshape(selected_count, int(num_agents))
                else:
                    ref_out = self.actor.act_bw(selected_local, deterministic=True)
                    sample_out = self.actor.act_bw(selected_local, deterministic=False)
                    ref_joint = ref_out.action.reshape(selected_count, int(num_agents), -1)
                    sample_all = sample_out.action.reshape(selected_count, int(num_agents), -1)
                    old_logprob_all = sample_out.logprob.reshape(selected_count, int(num_agents))
            if int(stage_id) == 1:
                sample_joint = ref_joint.clone()
                sample_joint[torch.arange(selected_count, device=self.device), target_uav_t] = sample_all[
                    torch.arange(selected_count, device=self.device), target_uav_t
                ]
                paired_actions = torch.empty(
                    (selected_count * 2, int(num_agents)),
                    dtype=sample_joint.dtype,
                    device=self.device,
                )
            else:
                sample_joint = ref_joint.clone()
                sample_joint[torch.arange(selected_count, device=self.device), target_uav_t, :] = sample_all[
                    torch.arange(selected_count, device=self.device), target_uav_t, :
                ]
                paired_actions = torch.empty(
                    (selected_count * 2,) + tuple(sample_joint.shape[1:]),
                    dtype=sample_joint.dtype,
                    device=self.device,
                )
            paired_actions[0::2] = ref_joint
            paired_actions[1::2] = sample_joint
            history_rows_selected = torch.as_tensor(
                history_rows_np,
                dtype=torch.long,
                device=self.device,
            ).index_select(0, sample_idx_t)
            paired_history_rows = history_rows_selected.repeat_interleave(2)
            paired_remaining_t = selected_remaining_t.to(device=self.device, dtype=torch.long).repeat_interleave(2)
            branch_returns = torch.empty((int(paired_history_rows.numel()),), dtype=torch.float32, device=self.device)
            for horizon_t in torch.unique(paired_remaining_t.detach()).to(dtype=torch.long).tolist():
                horizon_i = max(int(horizon_t), 1)
                group_idx = torch.nonzero(paired_remaining_t == horizon_i, as_tuple=False).reshape(-1)
                if int(group_idx.numel()) <= 0:
                    continue
                group_returns = self._native_vs_ref_rollout_returns_from_history(
                    stage_id=int(stage_id),
                    history_rows=paired_history_rows.index_select(0, group_idx).detach().cpu().tolist(),
                    first_actions=paired_actions.index_select(0, group_idx),
                    horizon=horizon_i,
                )
                branch_returns.index_copy_(0, group_idx, group_returns.to(device=self.device, dtype=torch.float32))
            ref_returns = branch_returns[0::2]
            sample_returns = branch_returns[1::2]
            advantage = (sample_returns - ref_returns).detach()
            if str(self.vs_ref_advantage_normalize) == "stage" and int(advantage.numel()) > 1:
                advantage_for_loss = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1.0e-8)
            else:
                advantage_for_loss = advantage
            logprob_per_agent, entropy_per_agent, actor_out = self._stage_actor_eval_per_agent_from_batch(
                int(stage_id),
                selected_local,
                sample_joint,
                int(num_agents),
            )
            gather_idx = target_uav_t.view(-1, 1)
            new_logprob = logprob_per_agent.gather(1, gather_idx).reshape(-1)
            entropy_target = entropy_per_agent.gather(1, gather_idx).reshape(-1)
            old_logprob = old_logprob_all.gather(1, gather_idx).reshape(-1).detach()
            log_ratio = new_logprob - old_logprob
            ratio = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))
            clipped_ratio = torch.clamp(ratio, 1.0 - float(self.clip_ratio), 1.0 + float(self.clip_ratio))
            policy_loss = -torch.minimum(ratio * advantage_for_loss, clipped_ratio * advantage_for_loss).mean()
            entropy_mean = entropy_target.mean() if int(entropy_target.numel()) > 0 else torch.zeros((), device=self.device)
            danger_imitation_loss = torch.zeros((), dtype=torch.float32, device=self.device)
            danger_imitation_active_rate = torch.zeros((), dtype=torch.float32, device=self.device)
            if int(stage_id) == 0 and self.danger_imitation_enabled:
                danger_imitation_loss, danger_imitation_active_rate = self._accel_danger_imitation_loss(
                    actor_out=actor_out,
                    targets=stage_batch.danger_imitation_targets,
                    masks=stage_batch.danger_imitation_masks,
                    sample_idx=sample_idx_t,
                    selected_count=int(selected_count),
                    num_agents=int(num_agents),
                )
                danger_imitation_losses.append(danger_imitation_loss.detach())
                danger_imitation_active_rates.append(danger_imitation_active_rate.detach())
            stage_loss = (
                policy_loss
                - float(self.entropy_coef_by_stage[int(stage_id)]) * entropy_mean
                + float(self.danger_imitation_coef) * danger_imitation_loss
            )
            stage_losses.append((int(stage_id), stage_loss))
            branch_rows_total += int(selected_count)
            adv_values_all.append(advantage.detach())
            entropy_values_all.append(entropy_target.detach())
            approx_kl_t = (old_logprob - new_logprob).detach()
            clip_t = ((ratio.detach() - 1.0).abs() > float(self.clip_ratio)).to(torch.float32)
            approx_kl_values_all.append(approx_kl_t)
            clip_values_all.append(clip_t)
            stage_name = ("accel", "sat", "bw")[int(stage_id)]
            stage_metric_values[f"entropy_{stage_name}"] = float(entropy_mean.detach().item())
            stage_metric_values[f"approx_kl_{stage_name}"] = float(approx_kl_t.mean().detach().item())
            stage_metric_values[f"clip_frac_{stage_name}"] = float(clip_t.mean().detach().item())
            stage_metric_values[f"vs_ref_adv_mean_{stage_name}"] = float(advantage.mean().detach().item())
            stage_metric_values[f"vs_ref_adv_std_{stage_name}"] = float(advantage.std(unbiased=False).detach().item()) if advantage.numel() > 1 else 0.0
            stage_metric_values[f"vs_ref_positive_frac_{stage_name}"] = float((advantage > 0.0).to(torch.float32).mean().detach().item())
            stage_metric_values[f"vs_ref_rows_{stage_name}"] = float(selected_count)
            for key, value in sampling_stats.items():
                stage_metric_values[f"vs_ref_sampling_{key}_{stage_name}"] = float(value)
            if int(stage_id) == 0 and self.danger_imitation_enabled:
                stage_metric_values["danger_imitation_loss"] = float(danger_imitation_loss.detach().item())
                stage_metric_values["danger_imitation_active_rate"] = float(danger_imitation_active_rate.detach().item())

        if not stage_losses:
            return _structured_zero_update_metrics()
        total_loss = torch.stack([loss.reshape(()) for _stage_id, loss in stage_losses]).sum()
        stage_grad_norms: list[torch.Tensor] = []
        for stage_id, stage_loss in stage_losses:
            optimizer = self._actor_stage_optimizer(int(stage_id))
            if optimizer is None:
                raise RuntimeError(
                    f"sample-vs-ref update for stage {int(stage_id)} requires an actor optimizer."
                )
            stage_module = self._actor_stage_module(int(stage_id))
            stage_params = [p for p in stage_module.parameters() if p.requires_grad]
            if not stage_params:
                continue
            optimizer.zero_grad()
            stage_loss.backward()
            grad_norm_t = torch.nn.utils.clip_grad_norm_(stage_params, float(self.max_grad_norm))
            optimizer.step()
            grad_norm_t = grad_norm_t if torch.is_tensor(grad_norm_t) else torch.as_tensor(float(grad_norm_t), device=self.device)
            stage_grad_norms.append(grad_norm_t.detach())
            stage_name = ("accel", "sat", "bw")[int(stage_id)]
            stage_metric_values[f"grad_norm_{stage_name}"] = float(grad_norm_t.detach().item())
        entropy_all = torch.cat([v.reshape(-1) for v in entropy_values_all]) if entropy_values_all else torch.zeros((0,), device=self.device)
        adv_all = torch.cat([v.reshape(-1) for v in adv_values_all]) if adv_values_all else torch.zeros((0,), device=self.device)
        kl_all = torch.cat([v.reshape(-1) for v in approx_kl_values_all]) if approx_kl_values_all else torch.zeros((0,), device=self.device)
        clip_all = torch.cat([v.reshape(-1) for v in clip_values_all]) if clip_values_all else torch.zeros((0,), device=self.device)
        grad_norm = (
            torch.stack([v.reshape(()) for v in stage_grad_norms]).mean()
            if stage_grad_norms
            else torch.zeros((), device=self.device)
        )
        metrics = _structured_zero_update_metrics(
            policy_loss=float(total_loss.detach().item()),
            value_loss=0.0,
            entropy=float(entropy_all.mean().detach().item()) if entropy_all.numel() > 0 else 0.0,
            approx_kl=float(kl_all.mean().detach().item()) if kl_all.numel() > 0 else 0.0,
            clip_frac=float(clip_all.mean().detach().item()) if clip_all.numel() > 0 else 0.0,
            bw_grad_norm_policy=float(grad_norm.detach().item()) if torch.is_tensor(grad_norm) else float(grad_norm),
            vs_ref_rows=float(branch_rows_total),
            vs_ref_adv_mean=float(adv_all.mean().detach().item()) if adv_all.numel() > 0 else 0.0,
            vs_ref_adv_std=float(adv_all.std(unbiased=False).detach().item()) if adv_all.numel() > 1 else 0.0,
            vs_ref_positive_frac=float((adv_all > 0.0).to(torch.float32).mean().detach().item()) if adv_all.numel() > 0 else 0.0,
            danger_imitation_loss=(
                float(torch.stack([v.reshape(()) for v in danger_imitation_losses]).mean().detach().item())
                if danger_imitation_losses
                else 0.0
            ),
            danger_imitation_active_rate=(
                float(torch.stack([v.reshape(()) for v in danger_imitation_active_rates]).mean().detach().item())
                if danger_imitation_active_rates
                else 0.0
            ),
        )
        metrics.update(stage_metric_values)
        return metrics

    def _stage_policy_terms(
        self,
        stage_id: int,
        *,
        actor_out: Any,
        cache: dict[str, Any],
        mb_rel: torch.Tensor,
        joint_actions_mb: torch.Tensor,
        new_logprob: torch.Tensor,
        entropy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del joint_actions_mb
        stage_id_i = int(stage_id)
        old_logprob = cache["old_logprobs"].index_select(0, mb_rel).to(self.device).reshape_as(new_logprob)
        advantage = cache["advantages"].index_select(0, mb_rel).to(self.device).detach().reshape_as(new_logprob)
        log_ratio = new_logprob - old_logprob
        ratio = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))
        entropy_mean = entropy.mean() if entropy.numel() > 0 else torch.zeros((), dtype=torch.float32, device=self.device)
        approx_kl = (old_logprob - new_logprob).mean()
        clip_frac = ((ratio - 1.0).abs() > float(self.clip_ratio)).to(torch.float32).mean()
        if stage_id_i == 2 and self.bw_policy_update_mode in {"awr", "det_awr"}:
            weights = self._bw_awr_weights(advantage)
            policy_loss = -(weights * new_logprob).mean()
            if self.bw_awr_kl_coef > 0.0:
                policy_loss = policy_loss + float(self.bw_awr_kl_coef) * approx_kl.clamp_min(0.0)
        elif stage_id_i == 2 and self.bw_policy_update_mode == "vmpo_lite":
            selected_idx, weights = self._bw_vmpo_selection(advantage)
            policy_loss = -(weights * new_logprob.index_select(0, selected_idx)).sum()
            if self.bw_vmpo_kl_coef > 0.0:
                policy_loss = policy_loss + float(self.bw_vmpo_kl_coef) * approx_kl.clamp_min(0.0)
        else:
            clipped_ratio = torch.clamp(ratio, 1.0 - float(self.clip_ratio), 1.0 + float(self.clip_ratio))
            policy_loss = -torch.minimum(ratio * advantage, clipped_ratio * advantage).mean()
        zero = torch.zeros((), dtype=torch.float32, device=self.device)
        bw_abs_log_ratio_corr_valid_count = zero
        bw_abs_log_ratio_corr_latent_count = zero
        bw_kappa_mean = zero
        bw_kappa_p10 = zero
        bw_kappa_p90 = zero
        bw_kappa_hi_frac = zero
        if stage_id_i == 2 and actor_out is not None:
            abs_log_ratio = log_ratio.detach().abs()
            if getattr(actor_out, "valid_count", None) is not None:
                valid_count = actor_out.valid_count.reshape(-1).to(dtype=abs_log_ratio.dtype)
                if valid_count.numel() >= abs_log_ratio.numel():
                    valid_count_sample = valid_count.reshape(abs_log_ratio.numel(), -1).mean(dim=1)
                    bw_abs_log_ratio_corr_valid_count = _safe_corrcoef(abs_log_ratio, valid_count_sample)
            if getattr(actor_out, "latent_count", None) is not None:
                latent_count = actor_out.latent_count.reshape(-1).to(dtype=abs_log_ratio.dtype)
                if latent_count.numel() >= abs_log_ratio.numel():
                    latent_count_sample = latent_count.reshape(abs_log_ratio.numel(), -1).mean(dim=1)
                    bw_abs_log_ratio_corr_latent_count = _safe_corrcoef(abs_log_ratio, latent_count_sample)
            (
                bw_kappa_mean,
                bw_kappa_p10,
                bw_kappa_p90,
                bw_kappa_hi_frac,
            ) = self._bw_dirichlet_kappa_stats(
                actor_out=actor_out,
                valid_count=getattr(actor_out, "valid_count", None),
                latent_count=getattr(actor_out, "latent_count", None),
                num_samples=int(new_logprob.numel()),
                num_agents=max(int(cache.get("num_agents", 1)), 1),
            )
        return (
            policy_loss,
            entropy_mean,
            approx_kl,
            clip_frac,
            bw_abs_log_ratio_corr_valid_count,
            bw_abs_log_ratio_corr_latent_count,
            bw_kappa_mean,
            bw_kappa_p10,
            bw_kappa_p90,
            bw_kappa_hi_frac,
        )

    def _bw_delta_eval_from_batch(
        self,
        local_batch: Any,
        joint_actions: torch.Tensor,
        ref_actions: torch.Tensor,
        num_agents: int,
    ) -> torch.Tensor:
        if not hasattr(self.critic, "delta_bw"):
            raise RuntimeError("critic does not provide delta_bw.")
        sample_count = int(joint_actions.shape[0])
        num_agents_i = max(int(num_agents), 1)
        sampled = joint_actions.reshape(sample_count * num_agents_i, -1).to(self.device)
        ref = ref_actions.reshape(sample_count * num_agents_i, -1).to(self.device)
        return self.critic.delta_bw(local_batch, sampled, ref, num_agents=num_agents_i).reshape(-1)

    def _bw_counterfactual_credit_from_proxy(
        self,
        proxy_scores: torch.Tensor,
        proxy_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        scores = proxy_scores.to(self.device, dtype=torch.float32)
        masks = proxy_masks.to(self.device) > 0.5
        masked_scores = torch.where(masks, scores, torch.zeros_like(scores))
        denom = masks.to(scores.dtype).sum(dim=-1).clamp_min(1.0)
        mean_score = masked_scores.sum(dim=-1) / denom
        max_score = torch.where(masks, scores, torch.full_like(scores, -1.0e9)).amax(dim=-1)
        valid_agent = masks.any(dim=-1)
        credit_agent = torch.where(valid_agent, max_score - mean_score, torch.zeros_like(mean_score))
        if credit_agent.ndim > 1:
            valid_sample = valid_agent.any(dim=1)
            denom_sample = valid_agent.to(scores.dtype).sum(dim=1).clamp_min(1.0)
            credit = credit_agent.sum(dim=1) / denom_sample
        else:
            valid_sample = valid_agent
            credit = credit_agent
        active = valid_sample & torch.isfinite(credit)
        metrics = {
            "active_rate": float(active.to(torch.float32).mean().item()) if active.numel() else 0.0,
            "agent_active_rate": float(valid_agent.to(torch.float32).mean().item()) if valid_agent.numel() else 0.0,
            "mean": float(credit[active].mean().item()) if torch.any(active) else 0.0,
            "abs_mean": float(credit[active].abs().mean().item()) if torch.any(active) else 0.0,
            "positive_frac": float((credit[active] > 0.0).to(torch.float32).mean().item()) if torch.any(active) else 0.0,
        }
        return credit, active, metrics

    def _bw_marginal_teacher_sample_credit(
        self,
        proxy_scores: torch.Tensor,
        proxy_masks: torch.Tensor,
        joint_actions: torch.Tensor,
        ref_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        scores = proxy_scores.to(self.device, dtype=torch.float32)
        masks = proxy_masks.to(self.device, dtype=torch.float32)
        action_delta = (joint_actions.to(self.device, dtype=torch.float32) - ref_actions.to(self.device, dtype=torch.float32))
        marginal = (action_delta * scores * masks).sum(dim=-1)
        valid_agent = masks.sum(dim=-1) > 0.5
        if marginal.ndim > 1:
            valid_sample = valid_agent.any(dim=1)
            denom = valid_agent.to(marginal.dtype).sum(dim=1).clamp_min(1.0)
            credit = marginal.sum(dim=1) / denom
        else:
            valid_sample = valid_agent
            credit = marginal
        active = valid_sample & torch.isfinite(credit)
        metrics = {
            "active_rate": float(active.to(torch.float32).mean().item()) if active.numel() else 0.0,
            "agent_active_rate": float(valid_agent.to(torch.float32).mean().item()) if valid_agent.numel() else 0.0,
            "mean": float(credit[active].mean().item()) if torch.any(active) else 0.0,
            "abs_mean": float(credit[active].abs().mean().item()) if torch.any(active) else 0.0,
            "positive_frac": float((credit[active] > 0.0).to(torch.float32).mean().item()) if torch.any(active) else 0.0,
        }
        return credit, active, metrics

    def _bw_slot_advantages_from_proxy(
        self,
        proxy_scores: torch.Tensor,
        proxy_masks: torch.Tensor,
        sample_advantages: torch.Tensor,
        *,
        slot_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = proxy_scores.to(self.device, dtype=torch.float32)
        masks = (proxy_masks.to(self.device) > 0.5) & slot_mask.to(self.device)
        denom = masks.to(scores.dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean_score = torch.where(masks, scores, torch.zeros_like(scores)).sum(dim=-1, keepdim=True) / denom
        centered = torch.where(masks, scores - mean_score, torch.zeros_like(scores))
        base = sample_advantages.to(self.device, dtype=torch.float32).view(-1, 1, 1)
        if self.bw_slot_advantage_base_mode == "zero":
            slot_adv = centered
        else:
            slot_adv = base + centered
        return slot_adv, masks

    def _bw_flow_proxy_aux_loss(
        self,
        local_batch: Any,
        actor_out: Any,
        proxy_scores: torch.Tensor,
        proxy_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del local_batch
        score = getattr(actor_out, "score", None)
        if score is None:
            zero = torch.zeros((), dtype=torch.float32, device=self.device)
            return zero, zero, zero, zero
        pred = score.to(self.device, dtype=torch.float32)
        target = proxy_scores.to(self.device, dtype=torch.float32)
        mask = proxy_masks.to(self.device) > 0.5
        if pred.shape != target.shape:
            pred = pred.reshape_as(target)
        if not torch.any(mask):
            zero = torch.zeros((), dtype=torch.float32, device=self.device)
            return zero, zero, zero, zero
        regression_loss = F.mse_loss(pred[mask], target[mask])
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        target_gap = target.unsqueeze(-1) - target.unsqueeze(-2)
        pred_gap = pred.unsqueeze(-1) - pred.unsqueeze(-2)
        pair_mask = pair_mask & (target_gap.abs() > float(self.bw_flow_proxy_aux_min_gap))
        if torch.any(pair_mask):
            pairwise_acc = ((pred_gap[pair_mask] * target_gap[pair_mask]) > 0.0).to(torch.float32).mean()
            pair_count = pair_mask.to(torch.float32).sum()
        else:
            pairwise_acc = torch.zeros((), dtype=torch.float32, device=self.device)
            pair_count = torch.zeros((), dtype=torch.float32, device=self.device)
        aux_loss = regression_loss
        return aux_loss, pairwise_acc, pair_count, regression_loss

    def _bw_grad_diagnostics(
        self,
        policy_loss: torch.Tensor,
        aux_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        params = [p for p in self.actor.parameters() if p.requires_grad]
        score_params = [
            p
            for p in getattr(getattr(self.actor, "bw_policy", None), "score_head", torch.nn.Module()).parameters()
            if p.requires_grad
        ]

        def _norm(loss: torch.Tensor, selected: list[torch.nn.Parameter]) -> torch.Tensor:
            if not loss.requires_grad or not selected:
                return torch.zeros((), dtype=torch.float32, device=self.device)
            grads = torch.autograd.grad(loss, selected, retain_graph=True, allow_unused=True)
            terms = [g.detach().pow(2).sum() for g in grads if g is not None]
            if not terms:
                return torch.zeros((), dtype=torch.float32, device=self.device)
            return torch.stack(terms).sum().sqrt()

        policy_norm = _norm(policy_loss, params)
        aux_scaled = float(self.bw_flow_proxy_aux_coef) * aux_loss
        aux_norm = _norm(aux_scaled, params)
        ratio = aux_norm / policy_norm.clamp_min(1.0e-12)
        score_policy_norm = _norm(policy_loss, score_params)
        score_aux_norm = _norm(aux_scaled, score_params)
        score_ratio = score_aux_norm / score_policy_norm.clamp_min(1.0e-12)
        return policy_norm, aux_norm, ratio, score_policy_norm, score_aux_norm, score_ratio

    @staticmethod
    def _native_bw_target_reward_tensor(
        result: StructuredBatchStepResult,
        *,
        bw_target_mode: str,
        bw_reward_w_access: float,
        device: torch.device,
    ) -> torch.Tensor:
        mode = str(bw_target_mode or "env_reward").strip().lower()

        def _required(name: str) -> torch.Tensor:
            value = getattr(result, name, None)
            if not torch.is_tensor(value):
                raise RuntimeError(f"native clean teacher branch replay returned no {name} tensor.")
            return value.to(device=device, dtype=torch.float32).reshape(-1)

        if mode == "env_reward":
            return _required("team_rewards")
        if mode == "access_term":
            return float(bw_reward_w_access) * _required("bw_access_rewards")
        if mode == "access_raw":
            return _required("bw_access_rewards")
        if mode == "weighted_workload_delta":
            return _required("bw_weighted_workload_delta_rewards")
        if mode == "weighted_workload_level":
            return _required("bw_weighted_workload_level_rewards")
        if mode == "gu_queue_level":
            return _required("bw_gu_queue_level_rewards")
        if mode == "system_queue_level":
            return _required("bw_system_queue_level_rewards")
        if mode == "gu_service_queue":
            return _required("bw_gu_service_queue_rewards")
        raise RuntimeError(f"Unsupported native BW clean teacher target mode: {bw_target_mode!r}")

    @staticmethod
    def _discounted_native_returns(
        results: Sequence[StructuredBatchStepResult],
        *,
        device: torch.device,
        gamma: float,
        bw_target_mode: str = "env_reward",
        bw_reward_w_access: float = 1.0,
    ) -> torch.Tensor:
        if not results:
            return torch.zeros((0,), dtype=torch.float32, device=device)
        first_reward = StructuredMAPPO._native_bw_target_reward_tensor(
            results[0],
            bw_target_mode=bw_target_mode,
            bw_reward_w_access=bw_reward_w_access,
            device=device,
        )
        returns = torch.zeros_like(first_reward)
        alive = torch.ones_like(returns, dtype=torch.float32)
        discount = 1.0
        for result in results:
            reward_t = StructuredMAPPO._native_bw_target_reward_tensor(
                result,
                bw_target_mode=bw_target_mode,
                bw_reward_w_access=bw_reward_w_access,
                device=device,
            )
            returns = returns + alive * float(discount) * reward_t
            done_t = result.terminated.to(device=device, dtype=torch.bool).reshape(-1)
            done_t = done_t | result.truncated.to(device=device, dtype=torch.bool).reshape(-1)
            alive = alive * (~done_t).to(dtype=torch.float32)
            discount *= float(gamma)
        return returns

    def _native_bw_clean_rollout_returns_from_history(
        self,
        *,
        history_rows: Sequence[int],
        first_actions: torch.Tensor,
        horizon: int | None = None,
    ) -> torch.Tensor:
        rows_tuple = tuple(int(row) for row in history_rows)
        branch_count = len(rows_tuple)
        if branch_count <= 0:
            return torch.zeros((0,), dtype=torch.float32, device=self.device)
        horizon_i = max(int(self.bw_clean_per_user_horizon if horizon is None else horizon), 1)
        rollout_program = getattr(self, "_native_rollout_program", None)
        drivers = None if rollout_program is None else getattr(rollout_program, "drivers", None)
        if drivers is None:
            raise RuntimeError("native BW clean teacher requires an active native rollout program.")
        runtime = getattr(rollout_program, "runtime", None)
        history = None if runtime is None else getattr(runtime, "history", None)
        source_num_envs = int(getattr(history, "num_envs", 0) or 0)
        if source_num_envs <= 0:
            raise RuntimeError("native BW clean teacher cannot infer source rollout env count.")
        sub_batch_program_factory = getattr(drivers, "native_sub_batch_rollout_program", None)
        prepare_branch = getattr(drivers, "prepare_native_branch_replay_from_history", None)
        if not callable(sub_batch_program_factory) or not callable(prepare_branch):
            raise RuntimeError("native BW clean teacher requires native branch replay support.")
        self._ensure_native_actor_cuda_bindings(sync=False)
        action_t = first_actions.detach().to(device=self.device, dtype=torch.float32)
        if int(action_t.shape[0]) != branch_count:
            raise RuntimeError("native BW clean teacher action count does not match history row count.")
        cfg_num_uav = max(int(getattr(self.cfg, "num_uav", 1) or 1), 1)
        if int(action_t.ndim) == 2:
            if int(action_t.shape[1]) % cfg_num_uav != 0:
                raise RuntimeError(
                    "native BW clean teacher received a flat first-action tensor whose width is not divisible "
                    f"by num_uav={cfg_num_uav}: shape={tuple(action_t.shape)}."
                )
            action_t = action_t.reshape(branch_count, cfg_num_uav, -1)
        elif int(action_t.ndim) == 3:
            if int(action_t.shape[1]) != cfg_num_uav:
                raise RuntimeError(
                    "native BW clean teacher first actions must contain the full joint UAV action: "
                    f"shape={tuple(action_t.shape)}, num_uav={cfg_num_uav}."
                )
            action_t = action_t.contiguous()
        else:
            raise RuntimeError(
                "native BW clean teacher first actions must be [branch, uav, user] or [branch, uav*user], "
                f"got shape={tuple(action_t.shape)}."
            )
        # Clean exact teacher calls this helper once per phase: ref, perturb, and target-gate.
        # Batch the whole phase by default; bw_actor_branch_parallel_envs is for legacy/sampled
        # branch probes and would otherwise turn one native phase into hundreds of tiny replays.
        clean_chunk_limit = max(int(getattr(self.cfg, "bw_clean_native_branch_chunk_size", 0) or 0), 0)
        chunk_size = branch_count if clean_chunk_limit <= 0 else min(branch_count, clean_chunk_limit)
        returns_out = torch.empty((branch_count,), dtype=torch.float32, device=self.device)
        for start in range(0, branch_count, max(int(chunk_size), 1)):
            end = min(start + max(int(chunk_size), 1), branch_count)
            chunk_rows = rows_tuple[start:end]
            chunk_actions = action_t[start:end]
            selected_envs = tuple(int(row % source_num_envs) for row in chunk_rows)
            program = sub_batch_program_factory(
                capacity=horizon_i,
                selected_indices=selected_envs,
                allow_duplicate_indices=True,
            )
            sub_runtime = program.runtime
            sub_runtime.main.accel_actor_source_mode_code = StructuredGpuRolloutProgram._source_mode_code(
                self.exec_source_by_stage[0]
            )
            sub_runtime.main.sat_actor_source_mode_code = StructuredGpuRolloutProgram._source_mode_code(
                self.exec_source_by_stage[1]
            )
            sub_runtime.main.bw_actor_source_mode_code = StructuredGpuRolloutProgram._source_mode_code(
                self.exec_source_by_stage[2]
            )
            begin_horizon = getattr(program._step_program.executor, "_runtime_begin_horizon", None)
            if not callable(begin_horizon):
                raise RuntimeError("native BW clean teacher sub-batch program cannot begin a horizon.")
            begin_horizon(num_steps=horizon_i)
            program._step_program._horizon_started = True
            prepare_branch(history_rows=chunk_rows, horizon=horizon_i)
            bridge = _BwCleanFirstActionOverrideBridge(
                base_bridge=_StructuredMAPPOGpuActorBridge(self),
                first_bw_actions=chunk_actions,
            )
            bridge.begin_horizon(horizon=horizon_i, runtime=sub_runtime, deterministic=True)
            results: list[StructuredBatchStepResult] = []
            step_bridge = bridge(step_index=0, runtime=sub_runtime)
            step_bridge.begin_step(deterministic=True)
            bw_obs = sub_runtime.main.live_bw_obs
            if bw_obs is None:
                raise RuntimeError("native BW clean teacher branch replay missing BW live obs at snapshot.")
            step_bridge.write_bw_action(
                bw_obs,
                runtime=sub_runtime,
                num_envs=int(end - start),
                deterministic=True,
            )
            first_result = program._step_program.executor._runtime_step_finish_bw(
                max_visible=program.fixed_visible_sat_width,
                rollout_tail=bool(horizon_i <= 1),
            )
            first_view = (
                sub_runtime.result.step_result_views[0]
                if sub_runtime.result.step_result_views
                else first_result
            )
            if not isinstance(first_view, StructuredBatchStepResult):
                raise RuntimeError("native BW clean teacher first branch step did not produce a result view.")
            results.append(first_view)
            for step_index in range(1, horizon_i):
                follow_bridge = bridge(step_index=step_index, runtime=sub_runtime)
                step_result = program.replay_step(
                    actor_bridge=follow_bridge,
                    deterministic=True,
                    rollout_tail=bool(step_index + 1 >= horizon_i),
                )
                step_view = (
                    sub_runtime.result.step_result_views[step_index]
                    if step_index < len(sub_runtime.result.step_result_views)
                    else step_result
                )
                if not isinstance(step_view, StructuredBatchStepResult):
                    raise RuntimeError("native BW clean teacher branch step did not produce a result view.")
                results.append(step_view)
            bridge.end_horizon(results=results, runtime=sub_runtime)
            returns_out[start:end] = self._discounted_native_returns(
                results,
                device=self.device,
                gamma=float(self.gamma),
                bw_target_mode=str(self.bw_train_target_mode),
                bw_reward_w_access=float(self.bw_reward_w_access),
            )
        return returns_out

    def _native_vs_ref_rollout_returns_from_history(
        self,
        *,
        stage_id: int,
        history_rows: Sequence[int],
        first_actions: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        rows_tuple = tuple(int(row) for row in history_rows)
        branch_count = len(rows_tuple)
        if branch_count <= 0:
            return torch.zeros((0,), dtype=torch.float32, device=self.device)
        stage_id_i = int(stage_id)
        if stage_id_i not in {0, 1, 2}:
            raise RuntimeError(f"sample-vs-ref stage_id must be 0, 1, or 2, got {stage_id_i}.")
        horizon_i = max(int(horizon), 1)
        rollout_program = getattr(self, "_native_rollout_program", None)
        drivers = None if rollout_program is None else getattr(rollout_program, "drivers", None)
        if drivers is None:
            raise RuntimeError("sample-vs-ref requires an active native rollout program.")
        runtime = getattr(rollout_program, "runtime", None)
        history = None if runtime is None else getattr(runtime, "history", None)
        source_num_envs = int(getattr(history, "num_envs", 0) or 0)
        if source_num_envs <= 0:
            raise RuntimeError("sample-vs-ref cannot infer source rollout env count.")
        sub_batch_program_factory = getattr(drivers, "native_sub_batch_rollout_program", None)
        prepare_branch = getattr(drivers, "prepare_native_branch_replay_from_history", None)
        if not callable(sub_batch_program_factory) or not callable(prepare_branch):
            raise RuntimeError("sample-vs-ref requires native branch replay support.")
        self._ensure_native_actor_cuda_bindings(sync=False)
        action_t = first_actions.detach().to(device=self.device)
        if int(action_t.shape[0]) != branch_count:
            raise RuntimeError("sample-vs-ref first-action count does not match history row count.")

        chunk_limit = max(int(getattr(self.cfg, "vs_ref_native_branch_chunk_size", 0) or 0), 0)
        chunk_size = branch_count if chunk_limit <= 0 else min(branch_count, chunk_limit)
        returns_out = torch.empty((branch_count,), dtype=torch.float32, device=self.device)
        for start in range(0, branch_count, max(int(chunk_size), 1)):
            end = min(start + max(int(chunk_size), 1), branch_count)
            chunk_rows = rows_tuple[start:end]
            chunk_actions = action_t[start:end].contiguous()
            selected_envs = tuple(int(row % source_num_envs) for row in chunk_rows)
            program = sub_batch_program_factory(
                capacity=horizon_i,
                selected_indices=selected_envs,
                allow_duplicate_indices=True,
            )
            sub_runtime = program.runtime
            # sample-vs-ref definition uses current actor deterministic as both
            # ref and follow policy, independent of normal exec_* rule sources.
            sub_runtime.main.accel_actor_source_mode_code = native_cuda.SOURCE_POLICY
            sub_runtime.main.sat_actor_source_mode_code = native_cuda.SOURCE_POLICY
            sub_runtime.main.bw_actor_source_mode_code = native_cuda.SOURCE_POLICY
            begin_horizon = getattr(program._step_program.executor, "_runtime_begin_horizon", None)
            if not callable(begin_horizon):
                raise RuntimeError("sample-vs-ref sub-batch program cannot begin a horizon.")
            begin_horizon(num_steps=horizon_i)
            program._step_program._horizon_started = True
            prepare_branch(history_rows=chunk_rows, horizon=horizon_i, stage_id=stage_id_i)
            bridge = _VsRefFirstActionOverrideBridge(
                base_bridge=_StructuredMAPPOGpuActorBridge(self),
                stage_id=stage_id_i,
                first_actions=chunk_actions,
            )
            bridge.begin_horizon(horizon=horizon_i, runtime=sub_runtime, deterministic=True)
            results: list[StructuredBatchStepResult] = []
            if stage_id_i == 0:
                step_result = program.replay_step(
                    actor_bridge=bridge(step_index=0, runtime=sub_runtime),
                    deterministic=True,
                    rollout_tail=bool(horizon_i <= 1),
                )
                step_view = sub_runtime.result.step_result_views[0] if sub_runtime.result.step_result_views else step_result
                if not isinstance(step_view, StructuredBatchStepResult):
                    raise RuntimeError("sample-vs-ref accel first branch step did not produce a result view.")
                results.append(step_view)
            elif stage_id_i == 1:
                step_bridge = bridge(step_index=0, runtime=sub_runtime)
                step_bridge.begin_step(deterministic=True)
                sat_obs = sub_runtime.main.live_sat_obs
                if sat_obs is None:
                    raise RuntimeError("sample-vs-ref SAT branch replay missing SAT live obs at snapshot.")
                step_bridge.write_sat_action(
                    sat_obs,
                    runtime=sub_runtime,
                    num_envs=int(end - start),
                    sat_max_select=int(getattr(sub_runtime.main, "sat_max_select", 1) or 1),
                    deterministic=True,
                )
                bw_obs = program._step_program.executor._runtime_step_publish_bw_obs(
                    max_visible=program.fixed_visible_sat_width
                )
                step_bridge.write_bw_action(
                    bw_obs,
                    runtime=sub_runtime,
                    num_envs=int(end - start),
                    deterministic=True,
                )
                first_result = program._step_program.executor._runtime_step_finish_bw(
                    max_visible=program.fixed_visible_sat_width,
                    rollout_tail=bool(horizon_i <= 1),
                )
                first_view = sub_runtime.result.step_result_views[0] if sub_runtime.result.step_result_views else first_result
                if not isinstance(first_view, StructuredBatchStepResult):
                    raise RuntimeError("sample-vs-ref SAT first branch step did not produce a result view.")
                results.append(first_view)
            else:
                step_bridge = bridge(step_index=0, runtime=sub_runtime)
                step_bridge.begin_step(deterministic=True)
                bw_obs = sub_runtime.main.live_bw_obs
                if bw_obs is None:
                    raise RuntimeError("sample-vs-ref BW branch replay missing BW live obs at snapshot.")
                step_bridge.write_bw_action(
                    bw_obs,
                    runtime=sub_runtime,
                    num_envs=int(end - start),
                    deterministic=True,
                )
                first_result = program._step_program.executor._runtime_step_finish_bw(
                    max_visible=program.fixed_visible_sat_width,
                    rollout_tail=bool(horizon_i <= 1),
                )
                first_view = sub_runtime.result.step_result_views[0] if sub_runtime.result.step_result_views else first_result
                if not isinstance(first_view, StructuredBatchStepResult):
                    raise RuntimeError("sample-vs-ref BW first branch step did not produce a result view.")
                results.append(first_view)
            for step_index in range(1, horizon_i):
                follow_bridge = bridge(step_index=step_index, runtime=sub_runtime)
                step_result = program.replay_step(
                    actor_bridge=follow_bridge,
                    deterministic=True,
                    rollout_tail=bool(step_index + 1 >= horizon_i),
                )
                step_view = (
                    sub_runtime.result.step_result_views[step_index]
                    if step_index < len(sub_runtime.result.step_result_views)
                    else step_result
                )
                if not isinstance(step_view, StructuredBatchStepResult):
                    raise RuntimeError("sample-vs-ref branch step did not produce a result view.")
                results.append(step_view)
            bridge.end_horizon(results=results, runtime=sub_runtime)
            returns_out[start:end] = self._discounted_native_returns(
                results,
                device=self.device,
                gamma=float(self.gamma),
                bw_target_mode="env_reward",
                bw_reward_w_access=float(self.bw_reward_w_access),
            )
        return returns_out

    def _build_native_bw_clean_per_user_targets(
        self,
        *,
        bw_stage_batch: Any,
        ref_probs: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        transition_indices = np.asarray(getattr(bw_stage_batch, "transition_indices", []), dtype=np.int64).reshape(-1)
        num_samples = int(getattr(bw_stage_batch, "num_samples", 0) or 0)
        num_agents = int(getattr(bw_stage_batch, "num_agents", 1) or 1)
        user_dim = int(ref_probs.shape[1])
        row_count = int(ref_probs.shape[0])
        if row_count != int(num_samples) * int(num_agents):
            raise RuntimeError(
                "native BW clean teacher expected flattened rows to equal num_samples * num_agents "
                f"(rows={row_count}, samples={num_samples}, agents={num_agents})."
            )
        if int(transition_indices.size) != int(num_samples):
            raise RuntimeError(
                "native BW clean teacher requires one BW transition index per BW snapshot "
                f"(got indices={int(transition_indices.size)}, samples={num_samples})."
            )
        if np.any((transition_indices - 2) % 3 != 0):
            raise RuntimeError("native BW clean teacher received non-BW transition indices.")
        history_rows = [int(row) for row in (((transition_indices - 2) // 3).astype(np.int64, copy=False)).tolist()]
        ref_action_3d = ref_probs.detach().reshape(int(num_samples), int(num_agents), int(user_dim))
        valid_f = valid_mask.to(dtype=torch.float32)
        valid_count_t = valid_f.sum(dim=1, keepdim=True)
        row_has_choice_t = valid_count_t.reshape(-1) > 1.0
        target_t = ref_probs.detach().clone()
        utility_t = torch.zeros_like(ref_probs)
        ref_returns_t = torch.zeros((row_count,), dtype=torch.float32, device=self.device)
        potential_rows_t = row_has_choice_t & torch.isfinite(ref_probs).all(dim=1)
        eligible_rows = torch.nonzero(potential_rows_t, as_tuple=False).reshape(-1).detach().cpu().tolist()
        eligible_count = int(len(eligible_rows))
        sample_budget = int(self.bw_clean_row_sample_budget)

        def _sample_rows(rows: list[int]) -> list[int]:
            rows = [int(row) for row in rows]
            row_total = int(len(rows))
            if bool(self.bw_clean_row_sample_enabled) and sample_budget > 0 and row_total > sample_budget:
                perm = torch.randperm(
                    row_total,
                    generator=self._bw_clean_row_sample_generator,
                    device=torch.device("cpu"),
                )[:sample_budget].tolist()
                return [rows[int(idx)] for idx in perm]
            return rows

        if bool(self.bw_clean_candidate_select_enabled):
            row_sample_stats = {
                "eligible_rows": float(eligible_count),
                "filtered_rows": 0.0,
                "sampled_rows": 0.0,
                "sample_frac": 0.0,
                "filter_frac": 0.0,
                "ref_branch_count": 0.0,
                "perturb_branch_count": 0.0,
                "gate_branch_count": 0.0,
                "candidate_positive_branch_count": 0.0,
                "candidate_selected_rows": 0.0,
            }
            if eligible_rows:
                ref_samples = sorted({int(row) // int(num_agents) for row in eligible_rows})
                row_sample_stats["ref_branch_count"] = float(len(ref_samples))
                ref_samples_t = torch.as_tensor(ref_samples, dtype=torch.long, device=self.device)
                ref_returns_active_t = self._native_bw_clean_rollout_returns_from_history(
                    history_rows=[history_rows[int(sample)] for sample in ref_samples],
                    first_actions=ref_action_3d.index_select(0, ref_samples_t),
                )
                ref_return_by_sample = {
                    int(sample): ref_returns_active_t[idx]
                    for idx, sample in enumerate(ref_samples)
                }
                for row in eligible_rows:
                    ref_returns_t[int(row)] = ref_return_by_sample[int(row) // int(num_agents)]

            if self.bw_clean_candidate_select_row_filter_mode == "ref_negative":
                ref_eps = float(self.bw_clean_candidate_select_ref_return_eps)
                filtered_rows = [
                    int(row)
                    for row in eligible_rows
                    if float(ref_returns_t[int(row)].detach().item()) < -float(ref_eps)
                ]
            else:
                filtered_rows = [int(row) for row in eligible_rows]
            row_sample_stats["filtered_rows"] = float(len(filtered_rows))
            row_sample_stats["filter_frac"] = float(len(filtered_rows)) / float(max(eligible_count, 1))

            potential_rows = _sample_rows(filtered_rows)
            sampled_count = int(len(potential_rows))
            row_sample_stats["sampled_rows"] = float(sampled_count)
            row_sample_stats["sample_frac"] = float(sampled_count) / float(max(eligible_count, 1))

            masked_ref_for_donor = torch.where(valid_mask, ref_probs, torch.full_like(ref_probs, -1.0))
            donor_idx_t = masked_ref_for_donor.argmax(dim=1)
            branch_rows: list[int] = []
            branch_agents: list[int] = []
            branch_actions: list[torch.Tensor] = []
            gate_eps = float(self.bw_clean_candidate_select_gate_eps)
            for row in potential_rows:
                row_i = int(row)
                sample = row_i // int(num_agents)
                agent = row_i % int(num_agents)
                donor = int(donor_idx_t[row_i].detach().item())
                donor_mass = float(ref_probs[row_i, donor].detach().item())
                valid_users = torch.nonzero(valid_mask[row_i], as_tuple=False).reshape(-1).detach().cpu().tolist()
                if len(valid_users) <= 1:
                    continue
                seen_deltas: set[float] = set()
                for delta_raw in self.bw_clean_candidate_select_deltas:
                    delta = min(float(delta_raw), donor_mass)
                    if delta <= 1.0e-8:
                        continue
                    delta_key = round(float(delta), 12)
                    if delta_key in seen_deltas:
                        continue
                    seen_deltas.add(delta_key)
                    for user_idx in valid_users:
                        user = int(user_idx)
                        if user == donor:
                            continue
                        action_row = ref_action_3d[int(sample)].detach().clone()
                        action_row[agent, user] = action_row[agent, user] + float(delta)
                        action_row[agent, donor] = action_row[agent, donor] - float(delta)
                        branch_rows.append(row_i)
                        branch_agents.append(agent)
                        branch_actions.append(action_row)
                if bool(self.bw_clean_candidate_select_include_onehot):
                    for user_idx in valid_users:
                        user = int(user_idx)
                        action_row = ref_action_3d[int(sample)].detach().clone()
                        action_row[agent].zero_()
                        action_row[agent, user] = 1.0
                        branch_rows.append(row_i)
                        branch_agents.append(agent)
                        branch_actions.append(action_row)
                if bool(self.bw_clean_candidate_select_include_uniform):
                    action_row = ref_action_3d[int(sample)].detach().clone()
                    action_row[agent].zero_()
                    uniform_value = 1.0 / float(len(valid_users))
                    for user_idx in valid_users:
                        action_row[agent, int(user_idx)] = float(uniform_value)
                    branch_rows.append(row_i)
                    branch_agents.append(agent)
                    branch_actions.append(action_row)

            row_sample_stats["perturb_branch_count"] = float(len(branch_actions))
            active_rows_t = torch.zeros((row_count,), dtype=torch.bool, device=self.device)
            target_returns_t = torch.full_like(ref_returns_t, float("nan"))
            utility_l1_t = torch.zeros((row_count,), dtype=torch.float32, device=self.device)
            if branch_actions:
                branch_returns_t = self._native_bw_clean_rollout_returns_from_history(
                    history_rows=[history_rows[int(row) // int(num_agents)] for row in branch_rows],
                    first_actions=torch.stack(branch_actions, dim=0),
                )
                branch_rows_t = torch.as_tensor(branch_rows, dtype=torch.long, device=self.device)
                branch_ref_returns_t = ref_returns_t.index_select(0, branch_rows_t)
                positive_branch_t = branch_returns_t > (branch_ref_returns_t + float(gate_eps))
                row_sample_stats["candidate_positive_branch_count"] = float(
                    positive_branch_t.to(dtype=torch.float32).sum().detach().item()
                )
                best_return_by_row: dict[int, float] = {}
                best_target_by_row: dict[int, torch.Tensor] = {}
                for branch_idx, row in enumerate(branch_rows):
                    row_i = int(row)
                    branch_return = float(branch_returns_t[int(branch_idx)].detach().item())
                    if row_i not in best_return_by_row or branch_return > best_return_by_row[row_i]:
                        best_return_by_row[row_i] = branch_return
                        best_target_by_row[row_i] = branch_actions[int(branch_idx)][int(branch_agents[int(branch_idx)])].detach().clone()
                for row_i, best_return in best_return_by_row.items():
                    ref_return = float(ref_returns_t[int(row_i)].detach().item())
                    if best_return > ref_return + float(gate_eps):
                        active_rows_t[int(row_i)] = True
                        target_t[int(row_i)] = best_target_by_row[int(row_i)].to(device=self.device, dtype=torch.float32)
                        target_returns_t[int(row_i)] = float(best_return)
                        utility_l1_t[int(row_i)] = max(float(best_return - ref_return), 0.0)

            target_t = torch.where(valid_mask, target_t.clamp_min(0.0), torch.zeros_like(target_t))
            target_sum_t = target_t.sum(dim=1, keepdim=True)
            target_t = torch.where(target_sum_t > 1.0e-8, target_t / target_sum_t.clamp_min(1.0e-8), ref_probs)
            row_sample_stats["candidate_selected_rows"] = float(active_rows_t.to(dtype=torch.float32).sum().detach().item())
            row_sample_stats["gate_branch_count"] = row_sample_stats["candidate_selected_rows"]
            self._bw_clean_last_ref_returns = ref_returns_t.detach().cpu().numpy().astype(np.float32, copy=False)
            self._bw_clean_last_target_returns = target_returns_t.detach().cpu().numpy().astype(np.float32, copy=False)
            self._bw_clean_last_utility_l1 = utility_l1_t.detach().cpu().numpy().astype(np.float32, copy=False)
            self._bw_clean_last_row_sample_stats = row_sample_stats
            target_gap_t = (target_t - ref_probs).abs()
            return target_t.detach(), active_rows_t.detach(), target_gap_t.detach(), utility_l1_t.detach()

        potential_rows = eligible_rows
        if bool(self.bw_clean_row_sample_enabled) and sample_budget > 0 and eligible_count > sample_budget:
            perm = torch.randperm(
                eligible_count,
                generator=self._bw_clean_row_sample_generator,
                device=torch.device("cpu"),
            )[:sample_budget].tolist()
            potential_rows = [eligible_rows[int(idx)] for idx in perm]
        sampled_count = int(len(potential_rows))
        sample_frac = float(sampled_count) / float(max(eligible_count, 1))
        row_sample_stats = {
            "eligible_rows": float(eligible_count),
            "sampled_rows": float(sampled_count),
            "sample_frac": float(sample_frac),
            "ref_branch_count": 0.0,
            "perturb_branch_count": 0.0,
            "gate_branch_count": 0.0,
        }
        if potential_rows:
            ref_samples = sorted({int(row) // int(num_agents) for row in potential_rows})
            row_sample_stats["ref_branch_count"] = float(len(ref_samples))
            ref_samples_t = torch.as_tensor(ref_samples, dtype=torch.long, device=self.device)
            ref_returns_active_t = self._native_bw_clean_rollout_returns_from_history(
                history_rows=[history_rows[int(sample)] for sample in ref_samples],
                first_actions=ref_action_3d.index_select(0, ref_samples_t),
            )
            ref_return_by_sample = {
                int(sample): ref_returns_active_t[idx]
                for idx, sample in enumerate(ref_samples)
            }
            for row in potential_rows:
                ref_returns_t[int(row)] = ref_return_by_sample[int(row) // int(num_agents)]

        masked_ref_for_donor = torch.where(valid_mask, ref_probs, torch.full_like(ref_probs, -1.0))
        donor_idx_t = masked_ref_for_donor.argmax(dim=1)
        branch_rows: list[int] = []
        branch_users: list[int] = []
        branch_deltas: list[float] = []
        branch_actions: list[torch.Tensor] = []
        delta_probe = float(self.bw_clean_per_user_delta_probe)
        for row in potential_rows:
            donor = int(donor_idx_t[int(row)].detach().item())
            donor_mass = float(ref_probs[int(row), donor].detach().item())
            delta = min(float(delta_probe), donor_mass)
            if delta <= 1.0e-8:
                continue
            valid_users = torch.nonzero(valid_mask[int(row)], as_tuple=False).reshape(-1).detach().cpu().tolist()
            for user_idx in valid_users:
                user = int(user_idx)
                if user == donor:
                    continue
                sample = int(row) // int(num_agents)
                agent = int(row) % int(num_agents)
                action_row = ref_action_3d[int(sample)].detach().clone()
                action_row[agent, user] = action_row[agent, user] + float(delta)
                action_row[agent, donor] = action_row[agent, donor] - float(delta)
                branch_rows.append(int(row))
                branch_users.append(user)
                branch_deltas.append(float(delta))
                branch_actions.append(action_row)
        row_sample_stats["perturb_branch_count"] = float(len(branch_actions))
        if branch_actions:
            branch_returns_t = self._native_bw_clean_rollout_returns_from_history(
                history_rows=[history_rows[int(row) // int(num_agents)] for row in branch_rows],
                first_actions=torch.stack(branch_actions, dim=0),
            )
            branch_rows_t = torch.as_tensor(branch_rows, dtype=torch.long, device=self.device)
            branch_users_t = torch.as_tensor(branch_users, dtype=torch.long, device=self.device)
            branch_delta_t = torch.as_tensor(branch_deltas, dtype=torch.float32, device=self.device)
            branch_gain_t = (branch_returns_t - ref_returns_t.index_select(0, branch_rows_t)) / branch_delta_t.clamp_min(1.0e-8)
            utility_t[branch_rows_t, branch_users_t] = branch_gain_t

        centered_t = torch.where(
            valid_mask,
            utility_t - utility_t.sum(dim=1, keepdim=True) / valid_count_t.clamp_min(1.0),
            torch.zeros_like(utility_t),
        )
        utility_l1_t = centered_t.abs().sum(dim=1)
        direction_t = centered_t / utility_l1_t.view(-1, 1).clamp_min(1.0e-8)

        neg_direction_t = direction_t < -1.0e-8
        rho_limit_t = torch.where(
            neg_direction_t,
            ref_probs / (-direction_t).clamp_min(1.0e-8),
            torch.full_like(ref_probs, float("inf")),
        )
        rho_max_t = rho_limit_t.min(dim=1).values
        active_rows_t = row_has_choice_t & torch.isfinite(rho_max_t) & (rho_max_t > 1.0e-8) & (utility_l1_t > 1.0e-8)
        rho_t = torch.where(
            active_rows_t,
            float(self.bw_clean_per_user_beta) * rho_max_t,
            torch.zeros_like(rho_max_t),
        )
        target_t = ref_probs + rho_t.view(-1, 1) * direction_t
        target_t = torch.where(valid_mask, target_t.clamp_min(0.0), torch.zeros_like(target_t))
        target_sum_t = target_t.sum(dim=1, keepdim=True)
        target_t = torch.where(target_sum_t > 1.0e-8, target_t / target_sum_t.clamp_min(1.0e-8), ref_probs)

        target_returns_t = torch.full_like(ref_returns_t, float("nan"))
        gate_rows = torch.nonzero(active_rows_t, as_tuple=False).reshape(-1).detach().cpu().tolist()
        row_sample_stats["gate_branch_count"] = float(len(gate_rows))
        if gate_rows:
            gate_rows_t = torch.as_tensor(gate_rows, dtype=torch.long, device=self.device)
            gate_actions: list[torch.Tensor] = []
            for row in gate_rows:
                sample = int(row) // int(num_agents)
                agent = int(row) % int(num_agents)
                action_row = ref_action_3d[int(sample)].detach().clone()
                action_row[int(agent)] = target_t[int(row)].detach()
                gate_actions.append(action_row)
            target_returns_active_t = self._native_bw_clean_rollout_returns_from_history(
                history_rows=[history_rows[int(row) // int(num_agents)] for row in gate_rows],
                first_actions=torch.stack(gate_actions, dim=0),
            )
            target_returns_t[gate_rows_t] = target_returns_active_t
            gated_active_t = torch.zeros_like(active_rows_t)
            gated_active_t[gate_rows_t] = target_returns_active_t > (
                ref_returns_t.index_select(0, gate_rows_t) + 1.0e-6
            )
            active_rows_t = active_rows_t & gated_active_t
            target_t = torch.where(active_rows_t.view(-1, 1), target_t, ref_probs)
        self._bw_clean_last_ref_returns = ref_returns_t.detach().cpu().numpy().astype(np.float32, copy=False)
        self._bw_clean_last_target_returns = target_returns_t.detach().cpu().numpy().astype(np.float32, copy=False)
        self._bw_clean_last_utility_l1 = utility_l1_t.detach().cpu().numpy().astype(np.float32, copy=False)
        self._bw_clean_last_row_sample_stats = row_sample_stats
        target_gap_t = (target_t - ref_probs).abs()
        return target_t.detach(), active_rows_t.detach(), target_gap_t.detach(), utility_l1_t.detach()

    def _update_bw_clean_per_user(self, bw_stage_batch: Any) -> dict[str, float]:
        if bw_stage_batch is None or int(getattr(bw_stage_batch, "num_samples", 0) or 0) <= 0:
            return _structured_zero_update_metrics(bw_actor_update_skipped=1.0)
        if self.actor_optimizer is None:
            raise RuntimeError("bw_clean_per_user_enabled requires an actor optimizer.")

        local_batch = bw_stage_batch.local_batch
        num_samples = int(bw_stage_batch.num_samples)
        num_agents = int(bw_stage_batch.num_agents)
        user_mask = getattr(local_batch, "gu_mask", None)
        bw_valid_mask = getattr(local_batch, "bw_valid_mask", None)
        if user_mask is None or bw_valid_mask is None:
            raise RuntimeError("bw_clean_per_user_enabled requires BW local tensor masks.")
        valid_t = ((user_mask.to(self.device) > 0.5) & (bw_valid_mask.to(self.device) > 0.5)).reshape(
            num_samples * num_agents,
            -1,
        )
        row_has_choice_t = valid_t.sum(dim=1) > 1
        if not bool(row_has_choice_t.any().item()):
            return _structured_zero_update_metrics(bw_actor_update_skipped=1.0)

        def _masked_probs(raw: torch.Tensor) -> torch.Tensor:
            masked = torch.where(valid_t, raw.clamp_min(0.0), torch.zeros_like(raw))
            denom = masked.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
            fallback = valid_t.to(dtype=torch.float32) / valid_t.to(dtype=torch.float32).sum(dim=1, keepdim=True).clamp_min(1.0)
            return torch.where(denom > 1.0e-8, masked / denom, fallback)

        def _det_bw_action() -> torch.Tensor:
            actor_out = self.actor.act_bw(local_batch, deterministic=True)
            pred = getattr(actor_out, "det_mean", None)
            if pred is None:
                pred = actor_out.action
            return pred.to(self.device, dtype=torch.float32).reshape(num_samples * num_agents, -1)

        with torch.no_grad():
            ref_norm_t = _masked_probs(_det_bw_action()).detach()
        target_t, active_rows_t, target_gap_t, utility_l1_t = self._build_native_bw_clean_per_user_targets(
            bw_stage_batch=bw_stage_batch,
            ref_probs=ref_norm_t,
            valid_mask=valid_t,
        )
        row_gap_t = (target_gap_t * valid_t.to(dtype=torch.float32)).sum(dim=1)
        active_rows_t = active_rows_t & row_has_choice_t & (row_gap_t > 1.0e-8)
        row_sample_stats = self._bw_clean_last_row_sample_stats or {}
        sampled_row_count = max(float(row_sample_stats.get("sampled_rows", float(valid_t.shape[0]))), 1.0)
        target_beats_frac = float(active_rows_t.to(dtype=torch.float32).mean().detach().item())
        target_beats_sampled_frac = float(active_rows_t.to(dtype=torch.float32).sum().detach().item()) / sampled_row_count
        clean_row_sample_metrics = {
            "clean_target_beats_sampled_frac": float(target_beats_sampled_frac),
            "clean_row_sample_eligible": float(row_sample_stats.get("eligible_rows", 0.0)),
            "clean_row_sample_count": float(row_sample_stats.get("sampled_rows", 0.0)),
            "clean_row_sample_frac": float(row_sample_stats.get("sample_frac", 0.0)),
            "clean_ref_branch_count": float(row_sample_stats.get("ref_branch_count", 0.0)),
            "clean_perturb_branch_count": float(row_sample_stats.get("perturb_branch_count", 0.0)),
            "clean_gate_branch_count": float(row_sample_stats.get("gate_branch_count", 0.0)),
            "clean_row_filter_count": float(row_sample_stats.get("filtered_rows", row_sample_stats.get("sampled_rows", 0.0))),
            "clean_row_filter_frac": float(row_sample_stats.get("filter_frac", 0.0)),
            "clean_candidate_positive_branch_count": float(row_sample_stats.get("candidate_positive_branch_count", 0.0)),
            "clean_candidate_selected_rows": float(row_sample_stats.get("candidate_selected_rows", 0.0)),
        }
        if not bool(active_rows_t.any().item()):
            return _structured_zero_update_metrics(
                bw_actor_update_skipped=1.0,
                clean_target_beats_ref_frac=target_beats_frac,
                **clean_row_sample_metrics,
            )

        def _clean_loss(pred_action: torch.Tensor, old_probs: torch.Tensor | None = None) -> torch.Tensor:
            pred_probs = _masked_probs(pred_action)
            active_mask = active_rows_t[:, None] & valid_t
            if self.bw_clean_per_user_loss == "masked_kl":
                target_safe = target_t.clamp_min(1.0e-8)
                pred_safe = pred_probs.clamp_min(1.0e-8)
                per_row = torch.where(valid_t, target_t * (torch.log(target_safe) - torch.log(pred_safe)), torch.zeros_like(target_t)).sum(dim=1)
                base_loss = per_row[active_rows_t].mean()
            else:
                base_loss = F.smooth_l1_loss(pred_probs[active_mask], target_t[active_mask])
            if self.bw_clean_trust_region_enabled and old_probs is not None and self._bw_clean_trust_region_kl_coef > 0.0:
                old_safe = old_probs.detach().clamp_min(1.0e-8)
                pred_safe = pred_probs.clamp_min(1.0e-8)
                kl_row = torch.where(valid_t, old_probs.detach() * (torch.log(old_safe) - torch.log(pred_safe)), torch.zeros_like(pred_probs)).sum(dim=1)
                base_loss = base_loss + float(self._bw_clean_trust_region_kl_coef) * kl_row[active_rows_t].mean()
            return base_loss

        with torch.no_grad():
            old_probs_t = ref_norm_t.detach()
        actor_state = copy.deepcopy(self.actor.state_dict())
        optim_state = copy.deepcopy(self.actor_optimizer.state_dict())
        original_lrs = [float(group.get("lr", 0.0)) for group in self.actor_optimizer.param_groups]
        measured_kl = 0.0
        accepted = False
        final_loss_value = 0.0
        attempts = 1 + (int(self.bw_clean_trust_region_max_backtracks) if self.bw_clean_trust_region_enabled else 0)
        scale = 1.0
        for _attempt in range(max(attempts, 1)):
            self.actor.load_state_dict(actor_state)
            self.actor_optimizer.load_state_dict(optim_state)
            for group, lr in zip(self.actor_optimizer.param_groups, original_lrs):
                group["lr"] = float(lr) * float(scale)
            self.actor_optimizer.zero_grad()
            loss = _clean_loss(_det_bw_action(), old_probs_t)
            final_loss_value = float(loss.detach().item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            self._sync_native_actor_cuda_bindings_after_update()
            with torch.no_grad():
                new_probs_t = _masked_probs(_det_bw_action()).detach()
                old_safe = old_probs_t.clamp_min(1.0e-8)
                new_safe = new_probs_t.clamp_min(1.0e-8)
                kl_row = torch.where(valid_t, old_probs_t * (torch.log(old_safe) - torch.log(new_safe)), torch.zeros_like(new_probs_t)).sum(dim=1)
                measured_kl = max(float(kl_row[active_rows_t].mean().detach().item()), 0.0)
            if (not self.bw_clean_trust_region_enabled) or measured_kl <= float(self.bw_clean_trust_region_target_kl) + 1.0e-12:
                accepted = True
                break
            scale *= float(self.bw_clean_trust_region_backtrack_factor)
        for group, lr in zip(self.actor_optimizer.param_groups, original_lrs):
            group["lr"] = float(lr)
        if not accepted:
            self.actor.load_state_dict(actor_state)
            self.actor_optimizer.load_state_dict(optim_state)
            self._sync_native_actor_cuda_bindings_after_update()
            return _structured_zero_update_metrics(
                bw_actor_update_skipped=1.0,
                clean_target_beats_ref_frac=target_beats_frac,
                clean_kl_coef=float(self._bw_clean_trust_region_kl_coef),
                **clean_row_sample_metrics,
            )

        with torch.no_grad():
            updated_probs_t = _masked_probs(_det_bw_action()).detach()
            update_shift_t = (updated_probs_t - ref_norm_t).abs()
            active_valid_t = active_rows_t[:, None] & valid_t
            mean_target_gap = float(target_gap_t[active_valid_t].mean().detach().item())
            mean_update_shift = float(update_shift_t[active_valid_t].mean().detach().item())
            update_to_target_ratio = mean_update_shift / max(mean_target_gap, 1.0e-8)
        return _structured_zero_update_metrics(
            policy_loss=final_loss_value,
            bw_actor_update_skipped=0.0,
            clean_mean_target_gap=mean_target_gap,
            clean_mean_update_shift=mean_update_shift,
            clean_update_to_target_ratio=update_to_target_ratio,
            clean_target_beats_ref_frac=target_beats_frac,
            clean_measured_kl=measured_kl,
            clean_kl_coef=float(self._bw_clean_trust_region_kl_coef),
            **clean_row_sample_metrics,
        )

    def _update_sat_clean_joint(self, sat_stage_batch: Any) -> dict[str, float]:
        del sat_stage_batch
        raise RuntimeError(
            "sat_clean_joint_enabled depends on the removed clean-task legacy trainer; "
            "it is not part of the official single-GPU native main-kernel path."
        )

    def _normalize_bw_entropy(
        self,
        entropy_per_agent: torch.Tensor,
        *,
        actor_out: Any,
        valid_count: torch.Tensor | None,
        latent_count: torch.Tensor | None,
    ) -> torch.Tensor:
        if getattr(actor_out, "entropy_raw", None) is not None:
            return entropy_per_agent
        if self.bw_entropy_norm_mode == "none":
            return entropy_per_agent
        norm_count = None
        if self.bw_entropy_norm_mode == "per_simplex_dim":
            norm_count = self._bw_simplex_dim(valid_count, latent_count)
        elif self.bw_entropy_norm_mode == "per_latent_count":
            norm_count = latent_count
            if norm_count is None and self._bw_is_dirichlet_actor_out(actor_out):
                norm_count = self._bw_simplex_dim(valid_count, latent_count)
        if norm_count is None:
            return entropy_per_agent
        return entropy_per_agent / norm_count.to(dtype=entropy_per_agent.dtype).clamp_min(1.0)

    def _bw_dirichlet_kappa_stats(
        self,
        *,
        actor_out: Any,
        valid_count: torch.Tensor | None,
        latent_count: torch.Tensor | None,
        num_samples: int,
        num_agents: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = torch.zeros((), dtype=torch.float32, device=self.device)
        kappa = None if actor_out is None else getattr(actor_out, "kappa", None)
        if kappa is None:
            return zero, zero, zero, zero
        kappa_per_agent = kappa.reshape(num_samples, num_agents).to(self.device)
        agent_mask = self._bw_agent_mask(actor_out, valid_count, latent_count, kappa_per_agent)
        active = kappa_per_agent[agent_mask]
        if active.numel() == 0:
            active = kappa_per_agent.reshape(-1)
        if active.numel() == 0:
            return zero, zero, zero, zero
        kappa_mean = active.mean()
        if active.numel() == 1:
            kappa_p10 = active[0]
            kappa_p90 = active[0]
        else:
            quantiles = torch.quantile(active, active.new_tensor([0.1, 0.9]))
            kappa_p10 = quantiles[0]
            kappa_p90 = quantiles[1]
        bw_policy = getattr(self.actor, "bw_policy", None)
        kappa_max = float(getattr(bw_policy, "kappa_max", 0.0) or 0.0)
        if kappa_max <= 0.0:
            return kappa_mean, kappa_p10, kappa_p90, zero
        kappa_hi_frac = (active >= (0.95 * kappa_max)).to(dtype=active.dtype).mean()
        return kappa_mean, kappa_p10, kappa_p90, kappa_hi_frac

    @staticmethod
    def _looks_like_driver_group(obj: Any) -> bool:
        required = (
            "reset_many",
            "reset_at",
            "native_rollout_program",
            "begin_native_main_kernel_rollout",
            "native_rollout_runtime",
        )
        return all(hasattr(obj, name) for name in required)

    def _can_use_native_tensor_policy_rollout(self, drivers: Any) -> bool:
        structured_env_tensor_device = self._structured_env_tensor_device()
        if structured_env_tensor_device is None:
            return False
        if str(getattr(self.cfg, "structured_env_backend", "") or "").strip().lower() != "native":
            return False
        if not self._looks_like_driver_group(drivers):
            return False
        if any(self.exec_source_by_stage[int(stage_id)] not in _NATIVE_EXEC_SOURCES for stage_id in (0, 1, 2)):
            return False
        program_factory = getattr(drivers, "native_rollout_program", None)
        if not callable(program_factory):
            return False
        try:
            program = program_factory()
        except Exception:
            return False
        return isinstance(program, StructuredGpuNativeRolloutProgram)

    def begin_native_rollout(self, drivers: Sequence[Any], *, rollout_env_steps: int, num_envs: int) -> None:
        if not self._can_use_native_tensor_policy_rollout(drivers):
            self._native_rollout_program = None
            self._native_rollout_program_drivers_id = None
            self._native_rollout_program_device = None
            return
        structured_env_tensor_device = self._structured_env_tensor_device()
        if structured_env_tensor_device is None:
            self._native_rollout_program = None
            self._native_rollout_program_drivers_id = None
            self._native_rollout_program_device = None
            return
        runtime = getattr(drivers, "native_rollout_runtime", None)
        if runtime is None:
            self._native_rollout_program = None
            self._native_rollout_program_drivers_id = None
            self._native_rollout_program_device = None
            return
        begin_main_kernel = getattr(drivers, "begin_native_main_kernel_rollout", None)
        rollout_capacity = int(rollout_env_steps)
        if self.bw_clean_per_user_enabled:
            rollout_capacity += max(int(self.bw_clean_per_user_horizon) - 1, 0)
        if callable(begin_main_kernel):
            begin_main_kernel(capacity=int(rollout_capacity), num_envs=int(num_envs))
        else:
            runtime.begin_rollout_training_ring(capacity=int(rollout_capacity), num_envs=int(num_envs))
            clear_main = getattr(runtime, "clear_main_kernel", None)
            if callable(clear_main):
                clear_main()
        self._native_rollout_program = StructuredGpuRolloutProgram(
            learner=self,
            drivers=drivers,
            tensor_device=structured_env_tensor_device,
        )
        self._native_rollout_program_drivers_id = id(drivers)
        self._native_rollout_program_device = torch.device(structured_env_tensor_device)

    def _native_rollout_program_for(
        self,
        drivers: Sequence[Any],
        *,
        tensor_device: torch.device,
    ) -> StructuredGpuRolloutProgram:
        device = torch.device(tensor_device)
        if (
            self._native_rollout_program is None
            or self._native_rollout_program_drivers_id != id(drivers)
            or self._native_rollout_program_device != device
        ):
            self._native_rollout_program = StructuredGpuRolloutProgram(
                learner=self,
                drivers=drivers,
                tensor_device=device,
            )
            self._native_rollout_program_drivers_id = id(drivers)
            self._native_rollout_program_device = device
        return self._native_rollout_program

    def _collect_env_steps_native_tensor_policy(
        self,
        drivers: Sequence[Any],
        buffer: StructuredRolloutBuffer | None,
        deterministic: bool = False,
    ):
        structured_env_tensor_device = self._structured_env_tensor_device()
        if structured_env_tensor_device is None:
            raise RuntimeError("native tensor rollout executor requires a structured env tensor device.")
        return self._native_rollout_program_for(
            drivers,
            tensor_device=structured_env_tensor_device,
        ).collect_step(buffer=buffer, deterministic=deterministic)

    def collect_env_horizon_native_tensor_policy(
        self,
        drivers: Sequence[Any],
        buffer: StructuredRolloutBuffer | None,
        *,
        horizon: int,
        deterministic: bool = False,
    ) -> list[Any]:
        if not self._can_use_native_tensor_policy_rollout(drivers):
            return [
                self.collect_env_steps(
                    drivers,
                    buffer,
                    deterministic=deterministic,
                )
                for _ in range(int(horizon))
            ]
        structured_env_tensor_device = self._structured_env_tensor_device()
        if structured_env_tensor_device is None:
            raise RuntimeError("native tensor rollout executor requires a structured env tensor device.")
        return self._native_rollout_program_for(
            drivers,
            tensor_device=structured_env_tensor_device,
        ).collect_horizon(
            horizon=int(horizon),
            buffer=buffer,
            deterministic=deterministic,
        )

    @staticmethod
    def _python_structured_driver_list(drivers: Any) -> list[Any]:
        source = getattr(drivers, "drivers", None)
        if source is None:
            if isinstance(drivers, Sequence) and not isinstance(drivers, (str, bytes, bytearray)):
                source = drivers
            elif hasattr(drivers, "__len__") and hasattr(drivers, "__getitem__"):
                source = [drivers[index] for index in range(int(len(drivers)))]
            else:
                source = [drivers]
        driver_list = list(source)
        required = (
            "begin_step",
            "build_local_accel_states",
            "run_accel_stage",
            "build_sat_stage_snapshot",
            "decode_sat_subset_actions",
            "run_sat_stage",
            "build_bw_stage_snapshot",
            "execute_stage_bw_and_prepare_next_accel",
        )
        if not driver_list or not all(all(hasattr(driver, name) for name in required) for driver in driver_list):
            raise RuntimeError("Python structured rollout requires StructuredControlDriver-like objects.")
        return driver_list

    @staticmethod
    def _env_batch_tensor(
        value: Any,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
        return tensor.to(device=device, dtype=dtype).unsqueeze(0)

    @staticmethod
    def _sat_local_with_row_subset_members(local_state: Any, row_count: int) -> Any:
        members = getattr(local_state, "subset_members", None)
        if not torch.is_tensor(members) or members.ndim != 2:
            return local_state
        expanded = members.unsqueeze(0).expand(max(int(row_count), 1), -1, -1).contiguous()
        return replace(local_state, subset_members=expanded)

    def _collect_env_steps_python_policy(
        self,
        drivers: Sequence[Any],
        buffer: StructuredRolloutBuffer | None,
        deterministic: bool = False,
    ) -> list[Any]:
        """Collect one structured env step through the portable Python driver.

        This path intentionally mirrors the three policy stages used by the
        native rollout, but executes one environment at a time.  It is a Mac/CPU
        compatibility path for smoke tests and debugging.
        """

        driver_list = self._python_structured_driver_list(drivers)
        if self.actor is None:
            raise RuntimeError("Python structured rollout requires an actor.")
        if self.critic is None:
            raise RuntimeError("Python structured rollout requires a critic.")
        results: list[Any] = []
        for env_index, driver in enumerate(driver_list):
            set_tensor_device = getattr(driver, "set_tensor_device", None)
            if callable(set_tensor_device):
                set_tensor_device(self.device)

            accel_world = driver.begin_step()
            accel_local = _collate_dataclass(driver.build_local_accel_states(accel_world), self.device)
            accel_value = self._stage_value_eval_from_batch(0, accel_world).detach().reshape(1)
            accel_out = self.actor.act_accel(accel_local, deterministic=deterministic)
            num_agents = int(accel_out.action.shape[0])
            accel_action = accel_out.action.detach()
            accel_logprob_per_agent = accel_out.logprob.detach().reshape(1, num_agents)
            accel_logprob = accel_logprob_per_agent.sum(dim=1)

            sat_world = driver.run_accel_stage(accel_action.detach().cpu().numpy())
            sat_stage_state = None
            export_sat_stage_state = getattr(driver, "export_sat_stage_state", None)
            if callable(export_sat_stage_state):
                try:
                    sat_stage_state = export_sat_stage_state()
                except Exception:
                    sat_stage_state = None
            sat_snapshot = driver.build_sat_stage_snapshot(sat_world)
            if sat_snapshot.local_state is None:
                raise RuntimeError("Python structured rollout could not build SAT local state.")
            sat_local = self._sat_local_with_row_subset_members(sat_snapshot.local_state, num_agents)
            sat_value = self._stage_value_eval_from_batch(1, sat_world).detach().reshape(1)
            sat_out = self.actor.act_sat(sat_local, deterministic=deterministic)
            sat_logprob_per_agent = sat_out.logprob.detach().reshape(1, num_agents)
            sat_logprob = sat_logprob_per_agent.sum(dim=1)
            sat_subset_index = sat_out.subset_index.detach().to(dtype=torch.long)
            sat_action = driver.decode_sat_subset_actions(
                [],
                sat_subset_index.detach().cpu().numpy().reshape(-1),
            )

            bw_world = driver.run_sat_stage(sat_action)
            bw_stage_state = None
            export_bw_stage_state = getattr(driver, "export_bw_stage_state", None)
            if callable(export_bw_stage_state):
                try:
                    bw_stage_state = export_bw_stage_state()
                except Exception:
                    bw_stage_state = None
            bw_snapshot = driver.build_bw_stage_snapshot(bw_world)
            bw_local = LocalBwState(
                ego_features=bw_snapshot.ego_features,
                selected_sat_tokens=bw_snapshot.selected_sat_tokens,
                selected_sat_mask=bw_snapshot.selected_sat_mask,
                gu_tokens=bw_snapshot.gu_tokens,
                gu_mask=bw_snapshot.gu_mask,
                bw_valid_mask=bw_snapshot.bw_valid_mask,
            )
            bw_value = self._stage_value_eval_from_batch(2, bw_world).detach().reshape(1)
            bw_out = self.actor.act_bw(bw_local, deterministic=deterministic)
            bw_action = bw_out.action.detach()
            bw_logprob_per_agent = bw_out.logprob.detach().reshape(1, num_agents)
            bw_logprob = bw_logprob_per_agent.sum(dim=1)
            bw_ref_action = bw_out.det_mean.detach()

            step_result, next_world = driver.execute_stage_bw_and_prepare_next_accel(
                bw_action.detach().cpu().numpy(),
                bw_proxy_base_action=bw_ref_action.detach().cpu().numpy(),
            )
            results.append(step_result)

            if buffer is None:
                continue
            reward = torch.as_tensor([float(getattr(step_result, "team_reward", 0.0))], dtype=torch.float32, device=self.device)
            terminated = torch.as_tensor([bool(getattr(step_result, "terminated", False))], dtype=torch.bool, device=self.device)
            truncated = torch.as_tensor([bool(getattr(step_result, "truncated", False))], dtype=torch.bool, device=self.device)
            buffer.add_env_step_batch(
                accel_world_batch=accel_world,
                sat_world_batch=sat_world,
                bw_world_batch=bw_world,
                next_world_batch=next_world,
                accel_local_batch=accel_local,
                sat_local_batch=sat_local,
                bw_local_batch=bw_local,
                accel_actions=accel_action.reshape(1, num_agents, -1),
                accel_latent_actions=(
                    None
                    if accel_out.latent_action is None
                    else accel_out.latent_action.detach().reshape(1, num_agents, -1)
                ),
                sat_actions=sat_subset_index.reshape(1, num_agents),
                bw_actions=bw_action.reshape(1, num_agents, -1),
                accel_old_logprobs=accel_logprob,
                sat_old_logprobs=sat_logprob,
                sat_old_logprobs_per_agent=sat_logprob_per_agent,
                bw_old_logprobs=bw_logprob,
                bw_old_logprobs_per_agent=bw_logprob_per_agent,
                accel_values=accel_value,
                sat_values=sat_value,
                bw_values=bw_value,
                rewards=reward,
                terminated=terminated,
                truncated=truncated,
                accel_danger_imitation_targets=self._env_batch_tensor(
                    getattr(step_result, "danger_imitation_target", None),
                    device=self.device,
                ),
                accel_danger_imitation_masks=self._env_batch_tensor(
                    getattr(step_result, "danger_imitation_mask", None),
                    device=self.device,
                ),
                sat_stage_states=[sat_stage_state],
                bw_stage_states=[bw_stage_state],
                bw_access_rewards=torch.as_tensor(
                    [float(getattr(step_result, "bw_access_reward", 0.0))],
                    dtype=torch.float32,
                    device=self.device,
                ),
                bw_weighted_workload_delta_rewards=torch.as_tensor(
                    [float(getattr(step_result, "bw_weighted_workload_delta_reward", 0.0))],
                    dtype=torch.float32,
                    device=self.device,
                ),
                bw_weighted_workload_level_rewards=torch.as_tensor(
                    [float(getattr(step_result, "bw_weighted_workload_level_reward", 0.0))],
                    dtype=torch.float32,
                    device=self.device,
                ),
                bw_gu_queue_level_rewards=torch.as_tensor(
                    [float(getattr(step_result, "bw_gu_queue_level_reward", 0.0))],
                    dtype=torch.float32,
                    device=self.device,
                ),
                bw_system_queue_level_rewards=torch.as_tensor(
                    [float(getattr(step_result, "bw_system_queue_level_reward", 0.0))],
                    dtype=torch.float32,
                    device=self.device,
                ),
                bw_gu_service_queue_rewards=torch.as_tensor(
                    [float(getattr(step_result, "bw_gu_service_queue_reward", 0.0))],
                    dtype=torch.float32,
                    device=self.device,
                ),
                bw_flow_proxy_scores=self._env_batch_tensor(
                    getattr(step_result, "bw_flow_proxy_scores", None),
                    device=self.device,
                ),
                bw_flow_proxy_masks=self._env_batch_tensor(
                    getattr(step_result, "bw_flow_proxy_mask", None),
                    device=self.device,
                ),
                bw_flow_proxy_deltas=self._env_batch_tensor(
                    getattr(step_result, "bw_flow_proxy_deltas", None),
                    device=self.device,
                ),
                bw_ref_actions=bw_ref_action.reshape(1, num_agents, -1),
                env_indices=[int(env_index)],
            )
        return results

    @torch.no_grad()
    def collect_env_steps(
        self,
        drivers: Sequence[Any],
        buffer: StructuredRolloutBuffer | None,
        deterministic: bool = False,
    ):
        if self._can_use_native_tensor_policy_rollout(drivers):
            return self._collect_env_steps_native_tensor_policy(
                drivers,
                buffer,
                deterministic=deterministic,
            )
        try:
            return self._collect_env_steps_python_policy(
                drivers,
                buffer,
                deterministic=deterministic,
            )
        except RuntimeError:
            raise
        raise RuntimeError(
            "StructuredMAPPO collect_env_steps requires the persistent native main-kernel rollout API; "
            "legacy stage-batch rollout has been removed."
        )

    def update(
        self,
        buffer: StructuredRolloutBuffer,
        bootstrap_world_state=None,
        *,
        rollout_views: StructuredRolloutViews | None = None,
    ) -> dict[str, float]:
        critic_training_disabled = bool(
            getattr(self, "disable_critic_training_for_bw_actor_only_signal", False)
            or getattr(self, "disable_critic_training_for_sat_clean_joint", False)
        )
        if self.actor_optimizer is None:
            raise RuntimeError("actor_optimizer must be provided before calling update()")
        if not critic_training_disabled and self.critic_optimizer is None:
            raise RuntimeError(
                "critic_optimizer must be provided before calling update() when critic training is enabled"
            )
        rollout_views = buffer.build_rollout_views(self.device) if rollout_views is None else rollout_views
        batch_view = rollout_views.training_view
        return_view = rollout_views.return_view
        effective_bootstrap_world_state = (
            rollout_views.bootstrap_view if bootstrap_world_state is None else bootstrap_world_state
        )
        stage_name_by_id = {0: "accel", 1: "sat", 2: "bw"}
        if int(batch_view.transition_count) == 0:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_frac": 0.0,
                "value_loss_accel": 0.0,
                "value_loss_sat": 0.0,
                "value_loss_bw": 0.0,
                "explained_variance_accel": 0.0,
                "explained_variance_sat": 0.0,
                "explained_variance_bw": 0.0,
                "entropy_accel": 0.0,
                "entropy_sat": 0.0,
                "entropy_bw": 0.0,
                "approx_kl_accel": 0.0,
                "approx_kl_sat": 0.0,
                "approx_kl_bw": 0.0,
                "clip_frac_accel": 0.0,
                "clip_frac_sat": 0.0,
                "clip_frac_bw": 0.0,
                "danger_imitation_loss": 0.0,
                "danger_imitation_active_rate": 0.0,
                "bw_counterfactual_credit_active_rate": 0.0,
                "bw_counterfactual_credit_agent_active_rate": 0.0,
                "bw_counterfactual_credit_mean": 0.0,
                "bw_counterfactual_credit_abs_mean": 0.0,
                "bw_counterfactual_credit_positive_frac": 0.0,
                "bw_flow_proxy_aux_loss": 0.0,
                "bw_flow_proxy_regression_loss": 0.0,
                "bw_flow_proxy_pairwise_acc": 0.0,
                "bw_flow_proxy_pair_count": 0.0,
                "bw_grad_norm_policy": 0.0,
                "bw_grad_norm_aux_scaled": 0.0,
                "bw_grad_ratio_aux_to_policy": 0.0,
                "bw_score_head_grad_norm_policy": 0.0,
                "bw_score_head_grad_norm_aux_scaled": 0.0,
                "bw_score_head_grad_ratio_aux_to_policy": 0.0,
                "bw_abs_log_ratio_corr_valid_count": 0.0,
                "bw_abs_log_ratio_corr_latent_count": 0.0,
                "bw_kappa_mean": 0.0,
                "bw_kappa_p10": 0.0,
                "bw_kappa_p90": 0.0,
                "bw_kappa_hi_frac": 0.0,
                "bw_delta_student_teacher_corr": 0.0,
                "bw_delta_student_true_corr": 0.0,
                "bw_delta_actor_mix_alpha": 0.0,
                "bw_delta_teacher_used": 0.0,
                "bw_delta_teacher_observed": 0.0,
                "bw_delta_teacher_probe_only": 0.0,
                "bw_branch_gate_snr": 0.0,
                "bw_branch_gate_triggered": 0.0,
                "bw_actor_update_skipped": 0.0,
            }
        if self.bw_clean_per_user_enabled:
            bw_stage_batch = batch_view.stage_batches.get(2)
            if bw_stage_batch is None:
                return _structured_zero_update_metrics()
            return self._update_bw_clean_per_user(bw_stage_batch)
        if self.sat_clean_joint_enabled:
            sat_stage_batch = batch_view.stage_batches.get(1)
            if sat_stage_batch is None:
                return _structured_zero_update_metrics()
            return self._update_sat_clean_joint(sat_stage_batch)
        vs_ref_metrics: dict[str, float] | None = None
        vs_ref_active_this_update = any(
            bool(self.train_actor_stage.get(int(stage_id), True))
            and self.stage_actor_update_mode.get(int(stage_id), "ppo") == "vs_ref"
            for stage_id in (0, 1, 2)
        )
        ppo_active_this_update = any(
            bool(self.train_actor_stage.get(int(stage_id), True))
            and self.stage_actor_update_mode.get(int(stage_id), "ppo") == "ppo"
            for stage_id in (0, 1, 2)
        )
        if vs_ref_active_this_update:
            vs_ref_metrics = self._update_vs_ref_native(buffer, rollout_views)
            if not ppo_active_this_update:
                return vs_ref_metrics
        update_profile_times: dict[str, float] = {}

        def _profile_add(name: str, elapsed: float) -> None:
            update_profile_times[name] = float(update_profile_times.get(name, 0.0) + max(float(elapsed), 0.0))

        _profile_t0 = time.perf_counter()
        self._refresh_actor_old_logprobs_from_training_view(batch_view)
        _profile_add("old_logprob_sec", time.perf_counter() - _profile_t0)

        def _critic_training_enabled_for_stage(stage_id: int) -> bool:
            del stage_id
            if critic_training_disabled:
                return False
            if getattr(self.critic, "value_mode", None) == "global_linear":
                return False
            # Critic heads estimate stage-conditioned values for the executed PPO
            # rollout, not only for the actor head being optimized this update.
            # Keep all three heads trained whenever any PPO actor stage is active.
            return bool(ppo_active_this_update)

        def _actor_training_enabled_for_stage(stage_id: int) -> bool:
            return bool(self.train_actor_stage.get(int(stage_id), True)) and (
                self.stage_actor_update_mode.get(int(stage_id), "ppo") == "ppo"
            )
        stage_ids = batch_view.stage_ids
        bw_delta_only_training = bool(
            self.bw_delta_only_training_enabled and int(np.count_nonzero(np.asarray(stage_ids) == 2)) > 0
        )
        if bw_delta_only_training:
            transition_count = int(len(stage_ids))
            advantages = torch.zeros((transition_count,), dtype=torch.float32, device=self.device)
            returns = torch.zeros_like(advantages)
            values_for_advantage = torch.zeros_like(advantages)
        else:
            _profile_t0 = time.perf_counter()
            rollout_value_override = (
                self._rollout_value_override_from_training_view(batch_view)
                if self.device.type == "cuda"
                and str(getattr(self, "structured_env_tensor_backend", "cpu") or "cpu").strip().lower() == "cuda"
                else None
            )
            _profile_add("value_override_sec", time.perf_counter() - _profile_t0)
            _profile_t0 = time.perf_counter()
            self._apply_rollout_value_override_to_views(
                batch_view=batch_view,
                return_view=return_view,
                value_override=rollout_value_override,
            )
            gae = self.compute_returns_and_advantages(
                buffer,
                effective_bootstrap_world_state,
                return_view=return_view,
                value_override=rollout_value_override,
            )
            _profile_add("returns_sec", time.perf_counter() - _profile_t0)
            advantages = torch.from_numpy(gae["advantages"]).to(self.device)
            returns = torch.from_numpy(gae["returns"]).to(self.device)
            values_for_advantage = batch_view.values if rollout_value_override is None else rollout_value_override
        self.last_update_actor_advantages_np = None
        self.last_update_actor_raw_advantages_np = None
        self.last_update_actor_values_np = None
        self.last_update_returns_np = None
        skip_actor_stage_ids_once = set(self._skip_actor_update_stage_ids_once)
        self._skip_actor_update_stage_ids_once.clear()
        bw_branch_gate_snr = float(self._bw_branch_gate_snr_once)
        bw_branch_gate_triggered = float(self._bw_branch_gate_triggered_once)
        self._bw_branch_gate_snr_once = 0.0
        self._bw_branch_gate_triggered_once = 0.0
        if not bw_delta_only_training and self.actor_advantage_normalize_enabled and advantages.numel() > 1:
            if self.stagewise_advantage_norm_enabled:
                advantages = advantages.clone()
                for stage_id in (0, 1, 2):
                    stage_idx_np = np.flatnonzero(stage_ids == stage_id)
                    if stage_idx_np.size <= 1:
                        continue
                    stage_idx = torch.as_tensor(stage_idx_np, device=self.device, dtype=torch.long)
                    stage_adv = advantages.index_select(0, stage_idx)
                    stage_adv = (
                        stage_adv - stage_adv.mean()
                    ) / stage_adv.std(unbiased=False).clamp_min(1e-8)
                    advantages.index_copy_(0, stage_idx, stage_adv)
            else:
                advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)

        bw_counterfactual_credit_metrics = {
            "active_rate": 0.0,
            "agent_active_rate": 0.0,
            "mean": 0.0,
            "abs_mean": 0.0,
            "positive_frac": 0.0,
        }
        bw_marginal_teacher_sample_metrics = {
            "active_rate": 0.0,
            "agent_active_rate": 0.0,
            "mean": 0.0,
            "abs_mean": 0.0,
            "positive_frac": 0.0,
        }
        stage_cache: dict[int, dict[str, Any]] = {}
        _profile_t0 = time.perf_counter()
        for stage_id in (0, 1, 2):
            if bw_delta_only_training and stage_id != 2:
                continue
            stage_batch = batch_view.stage_batches.get(stage_id)
            if stage_batch is None:
                continue
            stage_idx_np = np.asarray(stage_batch.transition_indices, dtype=np.int64)
            stage_idx = torch.as_tensor(stage_idx_np, device=self.device, dtype=torch.long)
            joint_actions = stage_batch.actions
            num_samples = int(stage_batch.num_samples)
            num_agents = int(stage_batch.num_agents)
            flat_indices = torch.arange(num_samples * num_agents, device=self.device, dtype=torch.long).reshape(num_samples, num_agents)
            cache: dict[str, Any] = {
                "stage_idx_np": stage_idx_np,
                "stage_idx": stage_idx,
                "num_samples": num_samples,
                "num_agents": num_agents,
                "world_batch": stage_batch.world_batch,
                "local_batch": stage_batch.local_batch,
                "joint_actions": joint_actions,
                "latent_actions": getattr(stage_batch, "latent_actions", None),
                "flat_indices": flat_indices,
                "stage_id": int(stage_id),
                "old_logprobs": stage_batch.old_logprobs,
                "returns": returns.index_select(0, stage_idx),
                "values_for_advantage": values_for_advantage.index_select(0, stage_idx),
                "advantages": advantages.index_select(0, stage_idx),
            }
            if stage_id == 2:
                if stage_batch.old_logprobs_per_agent is not None:
                    cache["old_logprobs_per_agent"] = stage_batch.old_logprobs_per_agent
                if stage_batch.bw_ref_actions is not None:
                    cache["bw_ref_actions"] = stage_batch.bw_ref_actions
            if self.danger_imitation_enabled and stage_id == 0:
                cache["danger_targets"] = (
                    stage_batch.danger_imitation_targets
                    if stage_batch.danger_imitation_targets is not None
                    else torch.zeros((num_samples, num_agents, 2), dtype=torch.float32, device=self.device)
                )
                cache["danger_masks"] = (
                    stage_batch.danger_imitation_masks
                    if stage_batch.danger_imitation_masks is not None
                    else torch.zeros((num_samples, num_agents, 2), dtype=torch.float32, device=self.device)
                )
            if self.bw_flow_proxy_signal_enabled and stage_id == 2:
                cache["bw_flow_proxy_scores"] = (
                    stage_batch.bw_flow_proxy_scores
                    if stage_batch.bw_flow_proxy_scores is not None
                    else torch.zeros_like(stage_batch.actions, dtype=torch.float32)
                )
                cache["bw_flow_proxy_masks"] = (
                    stage_batch.bw_flow_proxy_masks
                    if stage_batch.bw_flow_proxy_masks is not None
                    else torch.zeros_like(stage_batch.actions, dtype=torch.float32)
                )
            if self.bw_marginal_teacher_sample_enabled and stage_id == 2:
                ref_actions = cache.get("bw_ref_actions")
                if ref_actions is None:
                    raise RuntimeError("bw_marginal_teacher_sample_enabled requires cached BW ref actions.")
                mt_credit, mt_mask, bw_marginal_teacher_sample_metrics = self._bw_marginal_teacher_sample_credit(
                    cache["bw_flow_proxy_scores"],
                    cache["bw_flow_proxy_masks"],
                    cache["joint_actions"],
                    ref_actions,
                )
                cache["bw_marginal_teacher_sample_credit"] = mt_credit
                cache["bw_marginal_teacher_sample_mask"] = mt_mask
                if torch.any(mt_mask):
                    adv_stage = cache["advantages"].clone()
                    active_idx = torch.nonzero(mt_mask, as_tuple=False).flatten()
                    mt_active = mt_credit.index_select(0, active_idx)
                    if mt_active.numel() > 1:
                        mt_norm = (mt_active - mt_active.mean()) / mt_active.std(unbiased=False).clamp_min(1e-8)
                    else:
                        mt_norm = mt_active
                    base_adv = adv_stage.index_select(0, active_idx)
                    mixed_adv = (
                        (1.0 - self.bw_marginal_teacher_sample_weight) * base_adv
                        + self.bw_marginal_teacher_sample_weight * mt_norm
                    )
                    adv_stage.index_copy_(0, active_idx, mixed_adv)
                    cache["advantages"] = adv_stage
            if self.bw_per_slot_surrogate_enabled and stage_id == 2:
                local_state = cache["local_batch"]
                slot_mask = ((local_state.gu_mask > 0.5) & (local_state.bw_valid_mask > 0.5)).reshape(
                    num_samples,
                    num_agents,
                    -1,
                )
                if slot_mask.shape[-1] > 0:
                    ref_idx = torch.where(
                        slot_mask,
                        torch.arange(slot_mask.shape[-1], device=slot_mask.device, dtype=torch.long)
                        .view(1, 1, -1)
                        .expand_as(slot_mask),
                        torch.full_like(slot_mask, -1, dtype=torch.long),
                    ).amax(dim=-1)
                    active_rows = torch.nonzero(slot_mask.sum(dim=-1) > 1, as_tuple=False)
                    if active_rows.numel() > 0:
                        slot_mask = slot_mask.clone()
                        slot_mask[
                            active_rows[:, 0],
                            active_rows[:, 1],
                            ref_idx[active_rows[:, 0], active_rows[:, 1]],
                        ] = False
                slot_adv, slot_adv_mask = self._bw_slot_advantages_from_proxy(
                    cache["bw_flow_proxy_scores"],
                    cache["bw_flow_proxy_masks"],
                    cache["advantages"],
                    slot_mask=slot_mask,
                )
                cache["bw_slot_advantages"] = slot_adv
                cache["bw_slot_advantage_mask"] = slot_adv_mask
            if self.bw_counterfactual_credit_enabled and stage_id == 2:
                cf_credit, cf_mask, bw_counterfactual_credit_metrics = self._bw_counterfactual_credit_from_proxy(
                    cache["bw_flow_proxy_scores"],
                    cache["bw_flow_proxy_masks"],
                )
                cache["bw_counterfactual_credit"] = cf_credit
                cache["bw_counterfactual_credit_mask"] = cf_mask
                if torch.any(cf_mask):
                    adv_stage = cache["advantages"].clone()
                    active_idx = torch.nonzero(cf_mask, as_tuple=False).flatten()
                    cf_active = cf_credit.index_select(0, active_idx)
                    if cf_active.numel() > 1:
                        cf_norm = (cf_active - cf_active.mean()) / cf_active.std(unbiased=False).clamp_min(1e-8)
                    else:
                        cf_norm = cf_active
                    base_adv = adv_stage.index_select(0, active_idx)
                    mixed_adv = (
                        (1.0 - self.bw_counterfactual_credit_weight) * base_adv
                        + self.bw_counterfactual_credit_weight * cf_norm
                    )
                    adv_stage.index_copy_(0, active_idx, mixed_adv)
                    cache["advantages"] = adv_stage
            stage_cache[stage_id] = cache
        _profile_add("stage_cache_sec", time.perf_counter() - _profile_t0)

        def _prepare_bw_delta_critic_cache() -> None:
            nonlocal bw_delta_teacher_used, bw_delta_teacher_observed, bw_delta_teacher_probe_only
            if not self.bw_delta_critic_enabled:
                return
            cache_local = stage_cache.get(2)
            if cache_local is None or int(cache_local["num_samples"]) <= 0:
                return
            with torch.no_grad():
                det_actor_out = self.actor.act_bw(cache_local["local_batch"], deterministic=True)
                cache_local["bw_delta_ref_actions"] = det_actor_out.action.reshape(
                    int(cache_local["num_samples"]),
                    int(cache_local["num_agents"]),
                    -1,
                ).detach()
            cache_local.pop("bw_delta_targets", None)
            cache_local["bw_delta_teacher_probe_only"] = False
            use_dense_teacher = True
            probe_only_teacher = False
            if (
                self.bw_delta_teacher_student_enabled
                and self.bw_delta_teacher_student_disable_dense_teacher_when_ready
                and self._bw_delta_dense_teacher_disabled
            ):
                probe_interval = int(self.bw_delta_teacher_student_probe_interval_updates)
                update_idx_local = int(getattr(self, "current_update_index", 0) or 0)
                if probe_interval > 0 and update_idx_local > 0 and update_idx_local % probe_interval == 0:
                    use_dense_teacher = True
                    probe_only_teacher = True
                else:
                    use_dense_teacher = False
            cache_local["bw_delta_teacher_probe_only"] = bool(probe_only_teacher)
            cache_local["bw_delta_teacher_used"] = bool(use_dense_teacher and not probe_only_teacher)
            bw_delta_teacher_used = 1.0 if (use_dense_teacher and not probe_only_teacher) else 0.0
            bw_delta_teacher_observed = 1.0 if use_dense_teacher else 0.0
            bw_delta_teacher_probe_only = 1.0 if probe_only_teacher else 0.0
            if not use_dense_teacher:
                return
            from .structured_bw_update_direction import compute_bw_branch_advantage_override

            override_summary = compute_bw_branch_advantage_override(
                learner=self,
                buffer=buffer,
                bootstrap_world_state=effective_bootstrap_world_state,
                cfg=self.cfg,
                device=self.device,
                horizon=int(self.bw_delta_critic_horizon),
                branch_samples=int(self.bw_delta_critic_samples),
                branch_seed=int(getattr(self.cfg, "seed", 0) or 0) + 5_000_000,
                ref_mode=str(self.bw_delta_critic_ref_mode),
                follow_policy_mode=str(self.bw_delta_critic_follow_policy_mode),
                normalize=False,
                include_default_advantage_stats=not bw_delta_only_training,
            )
            if override_summary is None:
                raise RuntimeError("BW delta critic target generation produced no BW samples.")
            target_tensor = torch.as_tensor(
                np.asarray(override_summary["raw_branch_advantages"], dtype=np.float32).reshape(-1),
                dtype=cache_local["advantages"].dtype,
                device=cache_local["advantages"].device,
            )
            if int(target_tensor.numel()) != int(cache_local["advantages"].numel()):
                raise RuntimeError(
                    "BW delta critic target count mismatch: "
                    f"got {int(target_tensor.numel())}, expected {int(cache_local['advantages'].numel())}."
                )
            cache_local["bw_delta_targets"] = target_tensor

        _prepare_bw_delta_critic_cache()

        def _update_return_stats() -> None:
            if bw_delta_only_training:
                return
            if critic_training_disabled:
                return
            if not self.critic_loss_running_standardize:
                return
            decay = float(self.critic_loss_running_standardize_decay)
            for sid, cache_local in stage_cache.items():
                target = cache_local["returns"].detach()
                if target.numel() <= 0:
                    continue
                batch_mean = float(target.mean().item())
                batch_var = float(target.var(unbiased=False).clamp_min(1.0e-6).item())
                stats = self._critic_return_stats[int(sid)]
                if stats["initialized"] <= 0.5:
                    stats["mean"] = batch_mean
                    stats["var"] = batch_var
                    stats["initialized"] = 1.0
                else:
                    stats["mean"] = decay * float(stats["mean"]) + (1.0 - decay) * batch_mean
                    stats["var"] = decay * float(stats["var"]) + (1.0 - decay) * batch_var

        _update_return_stats()

        def _update_popart_stats() -> None:
            if bw_delta_only_training:
                return
            if critic_training_disabled:
                return
            if not self.critic_popart_enabled:
                return
            update_fn = getattr(self.critic, "update_popart_stats", None)
            if not callable(update_fn):
                return
            for sid, cache_local in stage_cache.items():
                if not self.train_actor_stage.get(sid, True):
                    continue
                update_fn(int(sid), cache_local["returns"].detach())

        _update_popart_stats()

        def _normalize_advantages_tensor(advantages_raw: torch.Tensor) -> torch.Tensor:
            normalized = advantages_raw
            if normalized.numel() <= 1:
                return normalized
            if not self.actor_advantage_normalize_enabled:
                return normalized
            if self.stagewise_advantage_norm_enabled:
                normalized = normalized.clone()
                for sid in (0, 1, 2):
                    stage_idx_np_local = np.flatnonzero(stage_ids == sid)
                    if stage_idx_np_local.size <= 1:
                        continue
                    stage_idx_local = torch.as_tensor(stage_idx_np_local, device=self.device, dtype=torch.long)
                    stage_adv = normalized.index_select(0, stage_idx_local)
                    stage_adv = (
                        stage_adv - stage_adv.mean()
                    ) / stage_adv.std(unbiased=False).clamp_min(1.0e-8)
                    normalized.index_copy_(0, stage_idx_local, stage_adv)
                return normalized
            return (normalized - normalized.mean()) / normalized.std(unbiased=False).clamp_min(1.0e-8)

        def _bw_delta_loss_pair(
            cache_local: dict[str, Any],
            local_batch_mb: Any,
            joint_actions_mb: torch.Tensor,
            ref_actions_mb: torch.Tensor,
            target_mb: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            pred_mb = self._bw_delta_eval_from_batch(
                local_batch_mb,
                joint_actions_mb,
                ref_actions_mb,
                int(cache_local["num_agents"]),
            )
            raw_loss = F.mse_loss(pred_mb, target_mb)
            return raw_loss, raw_loss

        def _refresh_bw_delta_advantages_from_critic() -> None:
            nonlocal bw_delta_student_teacher_corr
            nonlocal bw_delta_student_true_corr
            nonlocal bw_delta_actor_mix_alpha
            nonlocal bw_delta_teacher_used
            nonlocal bw_delta_teacher_observed
            nonlocal bw_delta_teacher_probe_only
            nonlocal bw_delta_teacher_ready_accounted
            if self.bw_actor_advantage_override_mode not in {"delta_critic", "delta_teacher_student"}:
                return
            cache_local = stage_cache.get(2)
            if cache_local is None or "bw_delta_ref_actions" not in cache_local:
                return
            with torch.no_grad():
                pred = self._bw_delta_eval_from_batch(
                    cache_local["local_batch"],
                    cache_local["joint_actions"],
                    cache_local["bw_delta_ref_actions"],
                    int(cache_local["num_agents"]),
                ).detach()
            teacher = cache_local.get("bw_delta_targets")
            teacher_tensor = None if teacher is None else teacher.detach()
            probe_only_teacher = bool(cache_local.get("bw_delta_teacher_probe_only", False))
            if teacher_tensor is not None and teacher_tensor.numel() == pred.numel():
                bw_delta_student_teacher_corr = float(_safe_corrcoef(pred, teacher_tensor).item()) if pred.numel() > 1 else 0.0
                if not np.isfinite(bw_delta_student_teacher_corr):
                    bw_delta_student_teacher_corr = 0.0
                self._bw_delta_last_teacher_corr = float(bw_delta_student_teacher_corr)
                bw_delta_teacher_observed = 1.0
                bw_delta_teacher_probe_only = 1.0 if probe_only_teacher else 0.0
                bw_delta_teacher_used = 0.0 if probe_only_teacher else 1.0
            else:
                bw_delta_student_teacher_corr = float(self._bw_delta_last_teacher_corr)
                bw_delta_teacher_observed = 0.0
                bw_delta_teacher_probe_only = 0.0
                bw_delta_teacher_used = 0.0
            true_mc = cache_local.get("bw_true_advantages")
            if true_mc is not None and true_mc.numel() == pred.numel():
                bw_delta_student_true_corr = float(_safe_corrcoef(pred, true_mc.detach()).item()) if pred.numel() > 1 else 0.0
            else:
                bw_delta_student_true_corr = 0.0
            if self.bw_actor_advantage_override_mode == "delta_teacher_student":
                if teacher_tensor is not None and teacher_tensor.numel() == pred.numel() and not probe_only_teacher:
                    corr_value = bw_delta_student_teacher_corr
                    if corr_value <= float(self.bw_delta_teacher_student_corr_low):
                        alpha = 0.0
                    elif corr_value >= float(self.bw_delta_teacher_student_corr_high):
                        alpha = 1.0
                    else:
                        ratio = (corr_value - float(self.bw_delta_teacher_student_corr_low)) / max(
                            float(self.bw_delta_teacher_student_corr_high - self.bw_delta_teacher_student_corr_low),
                            1.0e-6,
                        )
                        alpha = float(np.clip(ratio, 0.0, 1.0) ** float(self.bw_delta_teacher_student_mix_power))
                    bw_delta_actor_mix_alpha = float(alpha)
                    cache_local["advantages"] = teacher_tensor.lerp(pred, float(alpha))
                    if self.bw_delta_teacher_student_disable_dense_teacher_when_ready and not bw_delta_teacher_ready_accounted:
                        if corr_value >= float(self.bw_delta_teacher_student_corr_high):
                            self._bw_delta_teacher_ready_streak += 1
                        else:
                            self._bw_delta_teacher_ready_streak = 0
                        if self._bw_delta_teacher_ready_streak >= int(self.bw_delta_teacher_student_ready_patience):
                            self._bw_delta_dense_teacher_disabled = True
                        bw_delta_teacher_ready_accounted = True
                else:
                    if probe_only_teacher and bw_delta_student_teacher_corr < float(self.bw_delta_teacher_student_probe_reenable_corr):
                        self._bw_delta_dense_teacher_disabled = False
                        self._bw_delta_teacher_ready_streak = 0
                    bw_delta_actor_mix_alpha = 1.0
                    cache_local["advantages"] = pred
            else:
                bw_delta_actor_mix_alpha = 1.0
                cache_local["advantages"] = pred

        def _critic_loss_pair(
            cache_local: dict[str, Any],
            value_pred: torch.Tensor,
            value_target: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            raw_loss = F.mse_loss(value_pred, value_target)
            if self.critic_popart_enabled:
                sid = int(cache_local.get("stage_id", 0))
                normalize_fn = getattr(self.critic, "popart_normalize", None)
                if callable(normalize_fn):
                    return (
                        F.mse_loss(
                            normalize_fn(sid, value_pred),
                            normalize_fn(sid, value_target),
                        ),
                        raw_loss,
                    )
            if not (self.critic_loss_target_standardize or self.critic_loss_running_standardize):
                return raw_loss, raw_loss
            if self.critic_loss_running_standardize:
                sid = int(cache_local.get("stage_id", 0))
                stats = self._critic_return_stats.get(sid, {"mean": 0.0, "var": 1.0})
                target_mean = torch.as_tensor(float(stats["mean"]), dtype=value_target.dtype, device=value_target.device)
                target_std = torch.as_tensor(float(stats["var"]) ** 0.5, dtype=value_target.dtype, device=value_target.device).clamp_min(1.0e-6)
            else:
                full_target = cache_local["returns"].detach()
                target_std = full_target.std(unbiased=False).clamp_min(1.0e-6)
                target_mean = full_target.mean()
            scaled_loss = F.mse_loss(
                (value_pred - target_mean) / target_std,
                (value_target - target_mean) / target_std,
            )
            return scaled_loss, raw_loss

        def _run_critic_replay_epoch() -> bool:
            if critic_training_disabled:
                return False
            if bw_delta_only_training:
                return False
            if not self.critic_replay_bank_enabled or self.critic_replay_bank_capacity <= 0:
                return False
            used_any = False
            for sid in (0, 1, 2):
                cache_local = stage_cache.get(sid)
                if cache_local is None or not _critic_training_enabled_for_stage(sid):
                    continue
                current_targets = cache_local["returns"].detach().cpu().reshape(-1).tolist()
                bank_items = list(self._critic_replay_bank.get(sid, []))
                current_world_batch = cache_local["world_batch"]
                targets = torch.as_tensor(
                    current_targets + [float(item[1]) for item in bank_items],
                    dtype=torch.float32,
                    device=self.device,
                )
                if bank_items:
                    replay_world_batch = _collate_dataclass([item[0] for item in bank_items], self.device)
                    world_batch = _collate_dataclass([current_world_batch, replay_world_batch], self.device)
                else:
                    world_batch = current_world_batch
                rel_idx = np.arange(int(targets.numel()), dtype=np.int64)
                np.random.shuffle(rel_idx)
                minibatch_size = max(1, int(np.ceil(rel_idx.size / max(self.num_mini_batch, 1))))
                replay_cache = {"stage_id": int(sid), "returns": targets}
                stage_name_local = stage_name_by_id[sid]
                for start in range(0, rel_idx.size, minibatch_size):
                    mb_rel = torch.as_tensor(
                        rel_idx[start : start + minibatch_size],
                        device=self.device,
                        dtype=torch.long,
                    )
                    world_batch_mb = _index_dataclass(world_batch, mb_rel)
                    value_pred = self._stage_value_eval_from_batch(sid, world_batch_mb)
                    value_target = targets.index_select(0, mb_rel)
                    value_loss, raw_value_loss = _critic_loss_pair(replay_cache, value_pred, value_target)
                    self.critic_optimizer.zero_grad()
                    (self.value_coef * value_loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optimizer.step()
                    value_losses.append(float(raw_value_loss.item()))
                    stage_value_losses[stage_name_local].append(float(raw_value_loss.item()))
                    used_any = True
            return used_any

        def _append_critic_replay_bank() -> None:
            if bw_delta_only_training:
                return
            if critic_training_disabled:
                return
            if not self.critic_replay_bank_enabled or self.critic_replay_bank_capacity <= 0:
                return
            capacity = int(self.critic_replay_bank_capacity)
            for sid, cache_local in stage_cache.items():
                if not self.train_actor_stage.get(sid, True):
                    continue
                target_values = cache_local["returns"].detach().cpu().reshape(-1).tolist()
                bank = self._critic_replay_bank.setdefault(int(sid), [])
                for sample_idx, target_value in enumerate(target_values):
                    bank.append((_slice_dataclass(cache_local["world_batch"], sample_idx, sample_idx + 1), float(target_value)))
                if len(bank) > capacity:
                    del bank[: len(bank) - capacity]

        def _recompute_value_vector_from_critic() -> torch.Tensor:
            value_vector = batch_view.values.clone()
            with torch.no_grad():
                for sid, cache_local in stage_cache.items():
                    value_tensor = self._evaluate_world_batch_for_stage(
                        self._current_value_stage_id(int(sid)),
                        cache_local["world_batch"],
                    ).reshape(-1)
                    value_vector.index_copy_(0, cache_local["stage_idx"], value_tensor.to(self.device))
            return value_vector

        def _replace_stage_cache_from_gae(
            gae_local: dict[str, np.ndarray],
            value_override_local: torch.Tensor,
        ) -> None:
            nonlocal advantages, returns, values_for_advantage
            returns = torch.from_numpy(gae_local["returns"]).to(self.device)
            advantages = torch.from_numpy(gae_local["advantages"]).to(self.device)
            values_for_advantage = value_override_local.to(self.device)
            advantages = _normalize_advantages_tensor(advantages)
            for sid in (0, 1, 2):
                cache_local = stage_cache.get(sid)
                if cache_local is None:
                    continue
                stage_idx_local = cache_local["stage_idx"]
                cache_local["returns"] = returns.index_select(0, stage_idx_local)
                cache_local["values_for_advantage"] = values_for_advantage.index_select(0, stage_idx_local)
                cache_local["advantages"] = advantages.index_select(0, stage_idx_local)
            _update_return_stats()
            _update_popart_stats()

        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropy_values: List[float] = []
        approx_kl_values: List[float] = []
        clip_frac_values: List[float] = []
        danger_imitation_losses: List[float] = []
        bw_flow_proxy_aux_losses: List[float] = []
        bw_flow_proxy_regression_losses: List[float] = []
        bw_flow_proxy_pairwise_accs: List[float] = []
        bw_flow_proxy_pair_counts: List[float] = []
        bw_grad_norm_policy_values: List[float] = []
        bw_grad_norm_aux_scaled_values: List[float] = []
        bw_grad_ratio_aux_to_policy_values: List[float] = []
        bw_score_head_grad_norm_policy_values: List[float] = []
        bw_score_head_grad_norm_aux_scaled_values: List[float] = []
        bw_score_head_grad_ratio_aux_to_policy_values: List[float] = []
        bw_abs_log_ratio_corr_valid_count_values: List[float] = []
        bw_abs_log_ratio_corr_latent_count_values: List[float] = []
        bw_kappa_mean_values: List[float] = []
        bw_kappa_p10_values: List[float] = []
        bw_kappa_p90_values: List[float] = []
        bw_kappa_hi_frac_values: List[float] = []
        bw_delta_student_teacher_corr = 0.0
        bw_delta_student_true_corr = 0.0
        bw_delta_actor_mix_alpha = 0.0
        bw_delta_teacher_used = 0.0
        bw_delta_teacher_observed = 0.0
        bw_delta_teacher_probe_only = 0.0
        bw_delta_teacher_ready_accounted = False
        bw_actor_update_skipped = 1.0 if 2 in skip_actor_stage_ids_once else 0.0
        stage_value_losses: dict[str, List[float]] = {name: [] for name in stage_name_by_id.values()}
        stage_entropy_values: dict[str, List[float]] = {name: [] for name in stage_name_by_id.values()}
        stage_approx_kl_values: dict[str, List[float]] = {name: [] for name in stage_name_by_id.values()}
        stage_clip_frac_values: dict[str, List[float]] = {name: [] for name in stage_name_by_id.values()}
        stage_grad_norm_values: dict[str, List[float]] = {name: [] for name in stage_name_by_id.values()}

        danger_imitation_active_rate = 0.0
        if self.danger_imitation_enabled:
            stage0_cache = stage_cache.get(0)
            if stage0_cache is not None:
                stage0_masks = stage0_cache["danger_masks"]
                danger_imitation_active_rate = float((stage0_masks.sum(dim=-1) > 0.0).float().mean().item())

        def _fit_global_linear_critic_from_cache() -> None:
            if critic_training_disabled:
                return
            if getattr(self.critic, "value_mode", None) != "global_linear":
                return
            fit_fn = getattr(self.critic, "fit_global_linear", None)
            if not callable(fit_fn):
                raise RuntimeError("critic_value_mode='global_linear' requires StructuredCritic.fit_global_linear().")
            for sid in (0, 1, 2):
                cache_local = stage_cache.get(sid)
                if cache_local is None:
                    continue
                if not self.train_actor_stage.get(int(sid), True):
                    continue
                fit_fn(
                    int(sid),
                    cache_local["world_batch"],
                    cache_local["returns"],
                    ridge=float(self.critic_global_linear_fit_ridge),
                    decay=float(self.critic_global_linear_fit_decay),
                )
            if self.critic_global_linear_fit_recompute_advantages:
                _refresh_values_for_advantage_from_critic()
                _normalize_actor_advantages_from_values()
            with torch.no_grad():
                for sid in (0, 1, 2):
                    cache_local = stage_cache.get(sid)
                    if cache_local is None:
                        continue
                    if not self.train_actor_stage.get(int(sid), True):
                        continue
                    stage_name_local = stage_name_by_id[sid]
                    value_pred = self._stage_value_eval_from_batch(sid, cache_local["world_batch"])
                    raw_value_loss = F.mse_loss(value_pred, cache_local["returns"])
                    value_losses.append(float(raw_value_loss.item()))
                    stage_value_losses[stage_name_local].append(float(raw_value_loss.item()))

        def _make_stage_minibatches() -> tuple[dict[int, list[torch.Tensor]], int]:
            stage_minibatches_local: dict[int, list[torch.Tensor]] = {}
            max_stage_minibatches_local = 0
            for sid in (0, 1, 2):
                cache_local = stage_cache.get(sid)
                if cache_local is None:
                    continue
                rel_idx = np.arange(int(cache_local["num_samples"]), dtype=np.int64)
                np.random.shuffle(rel_idx)
                minibatch_size = max(1, int(np.ceil(rel_idx.size / max(self.num_mini_batch, 1))))
                minibatches = [
                    torch.as_tensor(rel_idx[start : start + minibatch_size], device=self.device, dtype=torch.long)
                    for start in range(0, rel_idx.size, minibatch_size)
                ]
                stage_minibatches_local[sid] = minibatches
                max_stage_minibatches_local = max(max_stage_minibatches_local, len(minibatches))
            return stage_minibatches_local, max_stage_minibatches_local

        def _run_critic_epoch(stage_minibatches_local: dict[int, list[torch.Tensor]], max_stage_minibatches_local: int) -> None:
            if bw_delta_only_training:
                cache_local = stage_cache.get(2)
                if (
                    cache_local is None
                    or "bw_delta_targets" not in cache_local
                    or bool(cache_local.get("bw_delta_teacher_probe_only", False))
                    or self.bw_delta_critic_loss_coef <= 0.0
                ):
                    return
                for mb_rel in stage_minibatches_local.get(2, []):
                    flat_mb = cache_local["flat_indices"].index_select(0, mb_rel).reshape(-1)
                    local_batch_mb = _index_dataclass(cache_local["local_batch"], flat_mb)
                    joint_actions_mb = cache_local["joint_actions"].index_select(0, mb_rel)
                    ref_actions_mb = cache_local["bw_delta_ref_actions"].index_select(0, mb_rel)
                    delta_target_mb = cache_local["bw_delta_targets"].index_select(0, mb_rel)
                    delta_loss, _raw_delta_loss = _bw_delta_loss_pair(
                        cache_local,
                        local_batch_mb,
                        joint_actions_mb,
                        ref_actions_mb,
                        delta_target_mb,
                    )
                    self.critic_optimizer.zero_grad()
                    (self.bw_delta_critic_loss_coef * delta_loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optimizer.step()
                return
            if self.joint_stage_updates:
                for minibatch_idx in range(max_stage_minibatches_local):
                    critic_loss_total = None
                    critic_used = False
                    for sid in (0, 1, 2):
                        cache_local = stage_cache.get(sid)
                        mb_list = stage_minibatches_local.get(sid)
                        if cache_local is None or mb_list is None or minibatch_idx >= len(mb_list):
                            continue
                        if not _critic_training_enabled_for_stage(sid):
                            continue
                        stage_name_local = stage_name_by_id[sid]
                        mb_rel = mb_list[minibatch_idx]
                        world_batch_mb = _index_dataclass(cache_local["world_batch"], mb_rel)
                        value_pred = self._stage_value_eval_from_batch(sid, world_batch_mb)
                        value_target = cache_local["returns"].index_select(0, mb_rel)
                        value_loss, raw_value_loss = _critic_loss_pair(cache_local, value_pred, value_target)
                        critic_loss_total = (
                            self.value_coef * value_loss
                            if critic_loss_total is None
                            else critic_loss_total + self.value_coef * value_loss
                        )
                        critic_used = True
                        value_losses.append(float(raw_value_loss.item()))
                        stage_value_losses[stage_name_local].append(float(raw_value_loss.item()))
                    if critic_used and critic_loss_total is not None:
                        self.critic_optimizer.zero_grad()
                        critic_loss_total.backward()
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                        self.critic_optimizer.step()
                return

            for sid in (0, 1, 2):
                cache_local = stage_cache.get(sid)
                if cache_local is None:
                    continue
                if not _critic_training_enabled_for_stage(sid):
                    continue
                stage_name_local = stage_name_by_id[sid]
                for mb_rel in stage_minibatches_local.get(sid, []):
                    world_batch_mb = _index_dataclass(cache_local["world_batch"], mb_rel)
                    value_pred = self._stage_value_eval_from_batch(sid, world_batch_mb)
                    value_target = cache_local["returns"].index_select(0, mb_rel)
                    value_loss, raw_value_loss = _critic_loss_pair(cache_local, value_pred, value_target)
                    critic_loss_total = self.value_coef * value_loss
                    if sid == 2 and "bw_delta_targets" in cache_local and self.bw_delta_critic_loss_coef > 0.0:
                        flat_mb = cache_local["flat_indices"].index_select(0, mb_rel).reshape(-1)
                        local_batch_mb = _index_dataclass(cache_local["local_batch"], flat_mb)
                        joint_actions_mb = cache_local["joint_actions"].index_select(0, mb_rel)
                        ref_actions_mb = cache_local["bw_delta_ref_actions"].index_select(0, mb_rel)
                        delta_target_mb = cache_local["bw_delta_targets"].index_select(0, mb_rel)
                        delta_loss, _raw_delta_loss = _bw_delta_loss_pair(
                            cache_local,
                            local_batch_mb,
                            joint_actions_mb,
                            ref_actions_mb,
                            delta_target_mb,
                        )
                        critic_loss_total = critic_loss_total + self.bw_delta_critic_loss_coef * delta_loss
                    self.critic_optimizer.zero_grad()
                    critic_loss_total.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optimizer.step()
                    value_losses.append(float(raw_value_loss.item()))
                    stage_value_losses[stage_name_local].append(float(raw_value_loss.item()))

        def _normalize_actor_advantages_from_values() -> None:
            if not self.actor_advantage_normalize_enabled:
                for sid in (0, 1, 2):
                    cache_local = stage_cache.get(sid)
                    if cache_local is None:
                        continue
                    cache_local["advantages"] = cache_local["returns"] - cache_local["values_for_advantage"]
                return
            if self.stagewise_advantage_norm_enabled:
                for sid in (0, 1, 2):
                    cache_local = stage_cache.get(sid)
                    if cache_local is None:
                        continue
                    raw_adv = cache_local["returns"] - cache_local["values_for_advantage"]
                    adv = raw_adv.clone()
                    if adv.numel() > 1:
                        adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
                    cache_local["advantages"] = adv
                return
            raw_adv_by_stage: dict[int, torch.Tensor] = {}
            all_raw_adv: list[torch.Tensor] = []
            for sid in (0, 1, 2):
                cache_local = stage_cache.get(sid)
                if cache_local is None:
                    continue
                raw_adv = cache_local["returns"] - cache_local["values_for_advantage"]
                raw_adv_by_stage[sid] = raw_adv
                all_raw_adv.append(raw_adv)
            if not all_raw_adv:
                return
            stacked = torch.cat(all_raw_adv, dim=0)
            if stacked.numel() > 1:
                mean = stacked.mean()
                std = stacked.std(unbiased=False).clamp_min(1e-8)
                for sid, raw_adv in raw_adv_by_stage.items():
                    stage_cache[sid]["advantages"] = (raw_adv - mean) / std
            else:
                for sid, raw_adv in raw_adv_by_stage.items():
                    stage_cache[sid]["advantages"] = raw_adv

        def _refresh_values_for_advantage_from_critic() -> None:
            with torch.no_grad():
                for sid in (0, 1, 2):
                    cache_local = stage_cache.get(sid)
                    if cache_local is None:
                        continue
                    if not self.train_actor_stage.get(sid, True):
                        continue
                    cache_local["values_for_advantage"] = self._stage_value_eval_from_batch(
                        sid,
                        cache_local["world_batch"],
                    ).detach()

        def _export_actor_advantage_snapshot() -> None:
            full_adv = torch.zeros_like(returns)
            full_raw_adv = torch.zeros_like(returns)
            full_values = torch.zeros_like(returns)
            full_returns = returns.detach().clone()
            for sid in (0, 1, 2):
                cache_local = stage_cache.get(sid)
                if cache_local is None:
                    continue
                stage_idx_local = cache_local["stage_idx"]
                if bw_delta_only_training and sid == 2 and "bw_delta_targets" in cache_local:
                    delta_target = cache_local["bw_delta_targets"].detach()
                    full_adv.index_copy_(0, stage_idx_local, cache_local["advantages"].detach())
                    full_raw_adv.index_copy_(0, stage_idx_local, delta_target)
                    full_returns.index_copy_(0, stage_idx_local, delta_target)
                    continue
                full_adv.index_copy_(0, stage_idx_local, cache_local["advantages"].detach())
                raw_adv = (cache_local["returns"] - cache_local["values_for_advantage"]).detach()
                full_raw_adv.index_copy_(0, stage_idx_local, raw_adv)
                full_values.index_copy_(0, stage_idx_local, cache_local["values_for_advantage"].detach())
            self.last_update_actor_advantages_np = full_adv.detach().cpu().numpy().astype(np.float32)
            self.last_update_actor_raw_advantages_np = full_raw_adv.detach().cpu().numpy().astype(np.float32)
            self.last_update_actor_values_np = full_values.detach().cpu().numpy().astype(np.float32)
            self.last_update_returns_np = full_returns.detach().cpu().numpy().astype(np.float32)

        def _apply_actor_advantage_override() -> None:
            if not self._actor_advantage_override_by_stage:
                return
            for sid, override_values in list(self._actor_advantage_override_by_stage.items()):
                cache_local = stage_cache.get(int(sid))
                if cache_local is None:
                    continue
                override_tensor = torch.as_tensor(
                    np.asarray(override_values, dtype=np.float32).reshape(-1),
                    dtype=cache_local["advantages"].dtype,
                    device=cache_local["advantages"].device,
                )
                if int(override_tensor.numel()) != int(cache_local["advantages"].numel()):
                    raise ValueError(
                        f"Actor advantage override for stage {int(sid)} has {int(override_tensor.numel())} entries, "
                        f"expected {int(cache_local['advantages'].numel())}."
                    )
                cache_local["advantages"] = override_tensor.reshape_as(cache_local["advantages"])
            self.clear_actor_advantage_override()

        if self.critic_warmup_before_actor_epochs > 0 and not bw_delta_only_training:
            if self.critic_warmup_recompute_advantages and (
                self.bw_counterfactual_credit_enabled or self.bw_per_slot_surrogate_enabled
            ):
                raise ValueError(
                    "critic_warmup_recompute_advantages is not supported together with "
                    "BW counterfactual/per-slot advantage overrides."
                )
            for _ in range(int(self.critic_warmup_before_actor_epochs)):
                used_replay = _run_critic_replay_epoch()
                if not used_replay:
                    stage_minibatches, max_stage_minibatches = _make_stage_minibatches()
                    _run_critic_epoch(stage_minibatches, max_stage_minibatches)
            if self.critic_warmup_recompute_advantages:
                if self.critic_warmup_recompute_mode == "gae":
                    value_override = _recompute_value_vector_from_critic()
                    _replace_stage_cache_from_gae(
                        self.compute_returns_and_advantages(
                            buffer,
                            effective_bootstrap_world_state,
                            return_view=return_view,
                            value_override=value_override,
                        ),
                        value_override,
                    )
                else:
                    _refresh_values_for_advantage_from_critic()
                    _normalize_actor_advantages_from_values()

        if self.bw_delta_critic_enabled and self.bw_delta_critic_warmup_epochs > 0:
            for _ in range(int(self.bw_delta_critic_warmup_epochs)):
                stage_minibatches, max_stage_minibatches = _make_stage_minibatches()
                _run_critic_epoch(stage_minibatches, max_stage_minibatches)
            _refresh_bw_delta_advantages_from_critic()

        _fit_global_linear_critic_from_cache()
        _apply_actor_advantage_override()
        _refresh_bw_delta_advantages_from_critic()
        _export_actor_advantage_snapshot()

        for _ in range(self.ppo_epochs):
            stage_minibatches: dict[int, list[torch.Tensor]] = {}
            max_stage_minibatches = 0
            for stage_id in (0, 1, 2):
                cache = stage_cache.get(stage_id)
                if cache is None:
                    continue
                rel_idx = np.arange(int(cache["num_samples"]), dtype=np.int64)
                np.random.shuffle(rel_idx)
                minibatch_size = max(1, int(np.ceil(rel_idx.size / max(self.num_mini_batch, 1))))
                minibatches = [
                    torch.as_tensor(rel_idx[start : start + minibatch_size], device=self.device, dtype=torch.long)
                    for start in range(0, rel_idx.size, minibatch_size)
                ]
                stage_minibatches[stage_id] = minibatches
                max_stage_minibatches = max(max_stage_minibatches, len(minibatches))

            if bw_delta_only_training:
                cache = stage_cache.get(2)
                if (
                    cache is not None
                    and self.train_actor_stage.get(2, True)
                    and "bw_delta_targets" in cache
                    and not bool(cache.get("bw_delta_teacher_probe_only", False))
                    and self.bw_delta_critic_loss_coef > 0.0
                ):
                    for mb_rel in stage_minibatches.get(2, []):
                        flat_mb = cache["flat_indices"].index_select(0, mb_rel).reshape(-1)
                        local_batch_mb = _index_dataclass(cache["local_batch"], flat_mb)
                        joint_actions_mb = cache["joint_actions"].index_select(0, mb_rel)
                        ref_actions_mb = cache["bw_delta_ref_actions"].index_select(0, mb_rel)
                        delta_target_mb = cache["bw_delta_targets"].index_select(0, mb_rel)
                        delta_loss, _raw_delta_loss = _bw_delta_loss_pair(
                            cache,
                            local_batch_mb,
                            joint_actions_mb,
                            ref_actions_mb,
                            delta_target_mb,
                        )
                        self.critic_optimizer.zero_grad()
                        (self.bw_delta_critic_loss_coef * delta_loss).backward()
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                        self.critic_optimizer.step()
                if cache is not None and self.train_actor_stage.get(2, True):
                    stage_name = stage_name_by_id[2]
                    if 2 in skip_actor_stage_ids_once:
                        continue
                    for mb_rel in stage_minibatches.get(2, []):
                        flat_mb = cache["flat_indices"].index_select(0, mb_rel).reshape(-1)
                        local_batch_mb = _index_dataclass(cache["local_batch"], flat_mb)
                        joint_actions_mb = cache["joint_actions"].index_select(0, mb_rel)
                        new_logprob, entropy, actor_out = self._stage_actor_eval_from_batch(
                            2,
                            local_batch_mb,
                            joint_actions_mb,
                            cache["num_agents"],
                        )
                        (
                            policy_loss,
                            entropy_mean,
                            approx_kl,
                            clip_frac,
                            bw_abs_log_ratio_corr_valid_count,
                            bw_abs_log_ratio_corr_latent_count,
                            bw_kappa_mean,
                            bw_kappa_p10,
                            bw_kappa_p90,
                            bw_kappa_hi_frac,
                        ) = self._stage_policy_terms(
                            2,
                            actor_out=actor_out,
                            cache=cache,
                            mb_rel=mb_rel,
                            joint_actions_mb=joint_actions_mb,
                            new_logprob=new_logprob,
                            entropy=entropy,
                        )
                        bw_flow_proxy_aux_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                        bw_flow_proxy_regression_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                        bw_flow_proxy_pairwise_acc = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                        bw_flow_proxy_pair_count = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                        bw_grad_norm_policy = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                        bw_grad_norm_aux_scaled = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                        bw_grad_ratio_aux_to_policy = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                        bw_score_head_grad_norm_policy = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                        bw_score_head_grad_norm_aux_scaled = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                        bw_score_head_grad_ratio_aux_to_policy = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                        if self.bw_flow_proxy_aux_enabled and actor_out is not None:
                            proxy_scores_mb = cache["bw_flow_proxy_scores"].index_select(0, mb_rel).reshape(
                                -1,
                                joint_actions_mb.shape[-1],
                            )
                            proxy_masks_mb = cache["bw_flow_proxy_masks"].index_select(0, mb_rel).reshape(
                                -1,
                                joint_actions_mb.shape[-1],
                            )
                            (
                                bw_flow_proxy_aux_loss,
                                bw_flow_proxy_pairwise_acc,
                                bw_flow_proxy_pair_count,
                                bw_flow_proxy_regression_loss,
                            ) = self._bw_flow_proxy_aux_loss(
                                local_batch_mb,
                                actor_out,
                                proxy_scores_mb,
                                proxy_masks_mb,
                            )
                            if self.bw_flow_proxy_grad_diagnostics_enabled:
                                (
                                    bw_grad_norm_policy,
                                    bw_grad_norm_aux_scaled,
                                    bw_grad_ratio_aux_to_policy,
                                    bw_score_head_grad_norm_policy,
                                    bw_score_head_grad_norm_aux_scaled,
                                    bw_score_head_grad_ratio_aux_to_policy,
                                ) = self._bw_grad_diagnostics(policy_loss, bw_flow_proxy_aux_loss)
                        loss = policy_loss + self.bw_flow_proxy_aux_coef * bw_flow_proxy_aux_loss - self.entropy_coef_by_stage[2] * entropy_mean
                        self.actor_optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                        self.actor_optimizer.step()
                        self._sync_native_actor_cuda_bindings_after_update()
                        policy_losses.append(float(policy_loss.item()))
                        entropy_values.append(float(entropy_mean.item()))
                        approx_kl_values.append(float(approx_kl.item()))
                        clip_frac_values.append(float(clip_frac.item()))
                        stage_entropy_values[stage_name].append(float(entropy_mean.item()))
                        stage_approx_kl_values[stage_name].append(float(approx_kl.item()))
                        stage_clip_frac_values[stage_name].append(float(clip_frac.item()))
                        bw_flow_proxy_aux_losses.append(float(bw_flow_proxy_aux_loss.item()))
                        bw_flow_proxy_regression_losses.append(float(bw_flow_proxy_regression_loss.item()))
                        bw_flow_proxy_pairwise_accs.append(float(bw_flow_proxy_pairwise_acc.item()))
                        bw_flow_proxy_pair_counts.append(float(bw_flow_proxy_pair_count.item()))
                        bw_grad_norm_policy_values.append(float(bw_grad_norm_policy.item()))
                        bw_grad_norm_aux_scaled_values.append(float(bw_grad_norm_aux_scaled.item()))
                        bw_grad_ratio_aux_to_policy_values.append(float(bw_grad_ratio_aux_to_policy.item()))
                        bw_score_head_grad_norm_policy_values.append(float(bw_score_head_grad_norm_policy.item()))
                        bw_score_head_grad_norm_aux_scaled_values.append(float(bw_score_head_grad_norm_aux_scaled.item()))
                        bw_score_head_grad_ratio_aux_to_policy_values.append(
                            float(bw_score_head_grad_ratio_aux_to_policy.item())
                        )
                        bw_abs_log_ratio_corr_valid_count_values.append(
                            float(bw_abs_log_ratio_corr_valid_count.item())
                        )
                        bw_abs_log_ratio_corr_latent_count_values.append(
                            float(bw_abs_log_ratio_corr_latent_count.item())
                        )
                        bw_kappa_mean_values.append(float(bw_kappa_mean.item()))
                        bw_kappa_p10_values.append(float(bw_kappa_p10.item()))
                        bw_kappa_p90_values.append(float(bw_kappa_p90.item()))
                        bw_kappa_hi_frac_values.append(float(bw_kappa_hi_frac.item()))
            elif self.joint_stage_updates:
                for minibatch_idx in range(max_stage_minibatches):
                    critic_loss_total = None
                    critic_used = False
                    for stage_id in (0, 1, 2):
                        cache = stage_cache.get(stage_id)
                        mb_list = stage_minibatches.get(stage_id)
                        if cache is None or mb_list is None or minibatch_idx >= len(mb_list):
                            continue
                        if not _critic_training_enabled_for_stage(stage_id):
                            continue
                        if stage_id in skip_actor_stage_ids_once:
                            continue
                        stage_name = stage_name_by_id[stage_id]
                        mb_rel = mb_list[minibatch_idx]
                        world_batch_mb = _index_dataclass(cache["world_batch"], mb_rel)
                        value_pred = self._stage_value_eval_from_batch(stage_id, world_batch_mb)
                        value_target = cache["returns"].index_select(0, mb_rel)
                        value_loss, raw_value_loss = _critic_loss_pair(cache, value_pred, value_target)
                        critic_loss_total = (
                            self.value_coef * value_loss
                            if critic_loss_total is None
                            else critic_loss_total + self.value_coef * value_loss
                        )
                        if stage_id == 2 and "bw_delta_targets" in cache and self.bw_delta_critic_loss_coef > 0.0:
                            flat_mb = cache["flat_indices"].index_select(0, mb_rel).reshape(-1)
                            local_batch_mb = _index_dataclass(cache["local_batch"], flat_mb)
                            joint_actions_mb = cache["joint_actions"].index_select(0, mb_rel)
                            ref_actions_mb = cache["bw_delta_ref_actions"].index_select(0, mb_rel)
                            delta_target_mb = cache["bw_delta_targets"].index_select(0, mb_rel)
                            delta_loss, _raw_delta_loss = _bw_delta_loss_pair(
                                cache,
                                local_batch_mb,
                                joint_actions_mb,
                                ref_actions_mb,
                                delta_target_mb,
                            )
                            critic_loss_total = critic_loss_total + self.bw_delta_critic_loss_coef * delta_loss
                        critic_used = True
                        value_losses.append(float(raw_value_loss.item()))
                        stage_value_losses[stage_name].append(float(raw_value_loss.item()))
                    if critic_used and critic_loss_total is not None:
                        self.critic_optimizer.zero_grad()
                        critic_loss_total.backward()
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                        self.critic_optimizer.step()

                for minibatch_idx in range(max_stage_minibatches):
                    actor_loss_total = None
                    actor_used = False
                    for stage_id in (0, 1, 2):
                        cache = stage_cache.get(stage_id)
                        mb_list = stage_minibatches.get(stage_id)
                        if cache is None or mb_list is None or minibatch_idx >= len(mb_list):
                            continue
                        if not _actor_training_enabled_for_stage(stage_id):
                            continue
                        stage_name = stage_name_by_id[stage_id]
                        mb_rel = mb_list[minibatch_idx]
                        flat_mb = cache["flat_indices"].index_select(0, mb_rel).reshape(-1)
                        local_batch_mb = _index_dataclass(cache["local_batch"], flat_mb)
                        joint_actions_mb = cache["joint_actions"].index_select(0, mb_rel)
                        latent_actions_mb = None
                        if int(stage_id) == 0:
                            latent_all = cache.get("latent_actions")
                            if latent_all is None:
                                raise RuntimeError(
                                    "accel joint PPO actor update requires latent_actions so the ratio uses pre-squash z samples."
                                )
                            latent_actions_mb = latent_all.index_select(0, mb_rel)
                        new_logprob, entropy, actor_out = self._stage_actor_eval_from_batch(
                            stage_id,
                            local_batch_mb,
                            joint_actions_mb,
                            cache["num_agents"],
                            latent_actions=latent_actions_mb,
                        )
                        (
                            policy_loss,
                            entropy_mean,
                            approx_kl,
                            clip_frac,
                            bw_abs_log_ratio_corr_valid_count,
                            bw_abs_log_ratio_corr_latent_count,
                            bw_kappa_mean,
                            bw_kappa_p10,
                            bw_kappa_p90,
                            bw_kappa_hi_frac,
                        ) = self._stage_policy_terms(
                            stage_id,
                            actor_out=actor_out,
                            cache=cache,
                            mb_rel=mb_rel,
                            joint_actions_mb=joint_actions_mb,
                            new_logprob=new_logprob,
                            entropy=entropy,
                        )
                        danger_imitation_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                        bw_flow_proxy_aux_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                        bw_flow_proxy_regression_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                        bw_flow_proxy_pairwise_acc = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                        bw_flow_proxy_pair_count = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                        bw_grad_norm_policy = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                        bw_grad_norm_aux_scaled = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                        bw_grad_ratio_aux_to_policy = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                        bw_score_head_grad_norm_policy = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                        bw_score_head_grad_norm_aux_scaled = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                        bw_score_head_grad_ratio_aux_to_policy = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                        if self.danger_imitation_enabled and stage_id == 0 and actor_out is not None:
                            target_accel = cache["danger_targets"].index_select(0, mb_rel)
                            danger_mask = cache["danger_masks"].index_select(0, mb_rel)
                            pred_accel = squash_action(actor_out.mean).reshape_as(target_accel)
                            active = torch.sum(danger_mask, dim=-1) > 0.0
                            if torch.any(active):
                                diff = (pred_accel - target_accel) * danger_mask
                                denom = torch.sum(danger_mask, dim=-1) + 1e-9
                                per_agent = diff.pow(2).sum(dim=-1) / denom
                                danger_imitation_loss = per_agent[active].mean()
                        if self.bw_flow_proxy_aux_enabled and stage_id == 2 and actor_out is not None:
                            proxy_scores_mb = cache["bw_flow_proxy_scores"].index_select(0, mb_rel).reshape(
                                -1,
                                joint_actions_mb.shape[-1],
                            )
                            proxy_masks_mb = cache["bw_flow_proxy_masks"].index_select(0, mb_rel).reshape(
                                -1,
                                joint_actions_mb.shape[-1],
                            )
                            (
                                bw_flow_proxy_aux_loss,
                                bw_flow_proxy_pairwise_acc,
                                bw_flow_proxy_pair_count,
                                bw_flow_proxy_regression_loss,
                            ) = self._bw_flow_proxy_aux_loss(
                                local_batch_mb,
                                actor_out,
                                proxy_scores_mb,
                                proxy_masks_mb,
                            )
                            if self.bw_flow_proxy_grad_diagnostics_enabled:
                                (
                                    bw_grad_norm_policy,
                                    bw_grad_norm_aux_scaled,
                                    bw_grad_ratio_aux_to_policy,
                                    bw_score_head_grad_norm_policy,
                                    bw_score_head_grad_norm_aux_scaled,
                                    bw_score_head_grad_ratio_aux_to_policy,
                                ) = self._bw_grad_diagnostics(policy_loss, bw_flow_proxy_aux_loss)
                        loss = (
                            policy_loss
                            + self.danger_imitation_coef * danger_imitation_loss
                            + self.bw_flow_proxy_aux_coef * bw_flow_proxy_aux_loss
                            - self.entropy_coef_by_stage[stage_id] * entropy_mean
                        )
                        actor_loss_total = loss if actor_loss_total is None else actor_loss_total + loss
                        actor_used = True
                        policy_losses.append(float(policy_loss.item()))
                        entropy_values.append(float(entropy_mean.item()))
                        approx_kl_values.append(float(approx_kl.item()))
                        clip_frac_values.append(float(clip_frac.item()))
                        stage_entropy_values[stage_name].append(float(entropy_mean.item()))
                        stage_approx_kl_values[stage_name].append(float(approx_kl.item()))
                        stage_clip_frac_values[stage_name].append(float(clip_frac.item()))
                        danger_imitation_losses.append(float(danger_imitation_loss.item()))
                        if stage_id == 2:
                            bw_flow_proxy_aux_losses.append(float(bw_flow_proxy_aux_loss.item()))
                            bw_flow_proxy_regression_losses.append(float(bw_flow_proxy_regression_loss.item()))
                            bw_flow_proxy_pairwise_accs.append(float(bw_flow_proxy_pairwise_acc.item()))
                            bw_flow_proxy_pair_counts.append(float(bw_flow_proxy_pair_count.item()))
                            bw_grad_norm_policy_values.append(float(bw_grad_norm_policy.item()))
                            bw_grad_norm_aux_scaled_values.append(float(bw_grad_norm_aux_scaled.item()))
                            bw_grad_ratio_aux_to_policy_values.append(float(bw_grad_ratio_aux_to_policy.item()))
                            bw_score_head_grad_norm_policy_values.append(float(bw_score_head_grad_norm_policy.item()))
                            bw_score_head_grad_norm_aux_scaled_values.append(float(bw_score_head_grad_norm_aux_scaled.item()))
                            bw_score_head_grad_ratio_aux_to_policy_values.append(
                                float(bw_score_head_grad_ratio_aux_to_policy.item())
                            )
                            bw_abs_log_ratio_corr_valid_count_values.append(
                                float(bw_abs_log_ratio_corr_valid_count.item())
                            )
                            bw_abs_log_ratio_corr_latent_count_values.append(
                                float(bw_abs_log_ratio_corr_latent_count.item())
                            )
                            bw_kappa_mean_values.append(float(bw_kappa_mean.item()))
                            bw_kappa_p10_values.append(float(bw_kappa_p10.item()))
                            bw_kappa_p90_values.append(float(bw_kappa_p90.item()))
                            bw_kappa_hi_frac_values.append(float(bw_kappa_hi_frac.item()))
                    if actor_used and actor_loss_total is not None and actor_loss_total.requires_grad:
                        self.actor_optimizer.zero_grad()
                        actor_loss_total.backward()
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                        self.actor_optimizer.step()
                        self._sync_native_actor_cuda_bindings_after_update()
            else:
                # Critic update first, stage-specific
                _profile_t0 = time.perf_counter()
                for stage_id in (0, 1, 2):
                    cache = stage_cache.get(stage_id)
                    if cache is None:
                        continue
                    if not _critic_training_enabled_for_stage(stage_id):
                        continue
                    stage_name = stage_name_by_id[stage_id]
                    for mb_rel in stage_minibatches.get(stage_id, []):
                        world_batch_mb = _index_dataclass(cache["world_batch"], mb_rel)
                        value_pred = self._stage_value_eval_from_batch(stage_id, world_batch_mb)
                        value_target = cache["returns"].index_select(0, mb_rel)
                        value_loss, raw_value_loss = _critic_loss_pair(cache, value_pred, value_target)
                        self.critic_optimizer.zero_grad()
                        (self.value_coef * value_loss).backward()
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                        self.critic_optimizer.step()
                        value_losses.append(float(raw_value_loss.item()))
                        stage_value_losses[stage_name].append(float(raw_value_loss.item()))
                _profile_add("critic_train_sec", time.perf_counter() - _profile_t0)

                # Actor update second, stage-specific
                _profile_t0 = time.perf_counter()
                for stage_id in (0, 1, 2):
                    cache = stage_cache.get(stage_id)
                    if cache is None:
                        continue
                    if not _actor_training_enabled_for_stage(stage_id):
                        continue
                    if stage_id in skip_actor_stage_ids_once:
                        continue
                    stage_name = stage_name_by_id[stage_id]
                    for mb_rel in stage_minibatches.get(stage_id, []):
                        metrics = self._run_stage_actor_update_minibatch(
                            stage_id=stage_id,
                            cache=cache,
                            mb_rel=mb_rel,
                        )
                        if metrics is None:
                            continue
                        policy_losses.append(float(metrics.get("policy_loss", 0.0)))
                        entropy_values.append(float(metrics.get("entropy_mean", 0.0)))
                        approx_kl_values.append(float(metrics.get("approx_kl", 0.0)))
                        clip_frac_values.append(float(metrics.get("clip_frac", 0.0)))
                        stage_entropy_values[stage_name].append(float(metrics.get("entropy_mean", 0.0)))
                        stage_approx_kl_values[stage_name].append(float(metrics.get("approx_kl", 0.0)))
                        stage_clip_frac_values[stage_name].append(float(metrics.get("clip_frac", 0.0)))
                        stage_grad_norm_values[stage_name].append(float(metrics.get("grad_norm", 0.0)))
                        danger_imitation_losses.append(float(metrics.get("danger_imitation_loss", 0.0)))
                        if stage_id == 2:
                            bw_flow_proxy_aux_losses.append(float(metrics.get("bw_flow_proxy_aux_loss", 0.0)))
                            bw_flow_proxy_regression_losses.append(float(metrics.get("bw_flow_proxy_regression_loss", 0.0)))
                            bw_flow_proxy_pairwise_accs.append(float(metrics.get("bw_flow_proxy_pairwise_acc", 0.0)))
                            bw_flow_proxy_pair_counts.append(float(metrics.get("bw_flow_proxy_pair_count", 0.0)))
                            bw_grad_norm_policy_values.append(float(metrics.get("bw_grad_norm_policy", 0.0)))
                            bw_grad_norm_aux_scaled_values.append(float(metrics.get("bw_grad_norm_aux_scaled", 0.0)))
                            bw_grad_ratio_aux_to_policy_values.append(float(metrics.get("bw_grad_ratio_aux_to_policy", 0.0)))
                            bw_score_head_grad_norm_policy_values.append(float(metrics.get("bw_score_head_grad_norm_policy", 0.0)))
                            bw_score_head_grad_norm_aux_scaled_values.append(float(metrics.get("bw_score_head_grad_norm_aux_scaled", 0.0)))
                            bw_score_head_grad_ratio_aux_to_policy_values.append(
                                float(metrics.get("bw_score_head_grad_ratio_aux_to_policy", 0.0))
                            )
                            bw_abs_log_ratio_corr_valid_count_values.append(
                                float(metrics.get("bw_abs_log_ratio_corr_valid_count", 0.0))
                            )
                            bw_abs_log_ratio_corr_latent_count_values.append(
                                float(metrics.get("bw_abs_log_ratio_corr_latent_count", 0.0))
                            )
                            bw_kappa_mean_values.append(float(metrics.get("bw_kappa_mean", 0.0)))
                            bw_kappa_p10_values.append(float(metrics.get("bw_kappa_p10", 0.0)))
                            bw_kappa_p90_values.append(float(metrics.get("bw_kappa_p90", 0.0)))
                            bw_kappa_hi_frac_values.append(float(metrics.get("bw_kappa_hi_frac", 0.0)))
                _profile_add("actor_train_sec", time.perf_counter() - _profile_t0)

        extra_critic_epochs = max(int(self.critic_epochs) - int(self.ppo_epochs), 0)
        for _ in range(extra_critic_epochs):
            stage_minibatches = {}
            max_stage_minibatches = 0
            for stage_id in (0, 1, 2):
                cache = stage_cache.get(stage_id)
                if cache is None:
                    continue
                rel_idx = np.arange(int(cache["num_samples"]), dtype=np.int64)
                np.random.shuffle(rel_idx)
                minibatch_size = max(1, int(np.ceil(rel_idx.size / max(self.num_mini_batch, 1))))
                minibatches = [
                    torch.as_tensor(rel_idx[start : start + minibatch_size], device=self.device, dtype=torch.long)
                    for start in range(0, rel_idx.size, minibatch_size)
                ]
                stage_minibatches[stage_id] = minibatches
                max_stage_minibatches = max(max_stage_minibatches, len(minibatches))

            if self.joint_stage_updates:
                for minibatch_idx in range(max_stage_minibatches):
                    critic_loss_total = None
                    critic_used = False
                    for stage_id in (0, 1, 2):
                        cache = stage_cache.get(stage_id)
                        mb_list = stage_minibatches.get(stage_id)
                        if cache is None or mb_list is None or minibatch_idx >= len(mb_list):
                            continue
                        if not _critic_training_enabled_for_stage(stage_id):
                            continue
                        stage_name = stage_name_by_id[stage_id]
                        mb_rel = mb_list[minibatch_idx]
                        world_batch_mb = _index_dataclass(cache["world_batch"], mb_rel)
                        value_pred = self._stage_value_eval_from_batch(stage_id, world_batch_mb)
                        value_target = cache["returns"].index_select(0, mb_rel)
                        value_loss, raw_value_loss = _critic_loss_pair(cache, value_pred, value_target)
                        critic_loss_total = (
                            self.value_coef * value_loss
                            if critic_loss_total is None
                            else critic_loss_total + self.value_coef * value_loss
                        )
                        critic_used = True
                        value_losses.append(float(raw_value_loss.item()))
                        stage_value_losses[stage_name].append(float(raw_value_loss.item()))
                    if critic_used and critic_loss_total is not None:
                        self.critic_optimizer.zero_grad()
                        critic_loss_total.backward()
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                        self.critic_optimizer.step()
            else:
                for stage_id in (0, 1, 2):
                    cache = stage_cache.get(stage_id)
                    if cache is None:
                        continue
                    if not _critic_training_enabled_for_stage(stage_id):
                        continue
                    stage_name = stage_name_by_id[stage_id]
                    for mb_rel in stage_minibatches.get(stage_id, []):
                        world_batch_mb = _index_dataclass(cache["world_batch"], mb_rel)
                        value_pred = self._stage_value_eval_from_batch(stage_id, world_batch_mb)
                        value_target = cache["returns"].index_select(0, mb_rel)
                        value_loss, raw_value_loss = _critic_loss_pair(cache, value_pred, value_target)
                        self.critic_optimizer.zero_grad()
                        (self.value_coef * value_loss).backward()
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                        self.critic_optimizer.step()
                        value_losses.append(float(raw_value_loss.item()))
                        stage_value_losses[stage_name].append(float(raw_value_loss.item()))

        stage_explained_variance: dict[str, float] = {name: 0.0 for name in stage_name_by_id.values()}
        _profile_t0 = time.perf_counter()
        with torch.no_grad():
            for stage_id, stage_name in stage_name_by_id.items():
                cache = stage_cache.get(stage_id)
                if cache is None:
                    continue
                if bw_delta_only_training and stage_id == 2:
                    continue
                value_pred = self._stage_value_eval_from_batch(stage_id, cache["world_batch"])
                value_target = cache["returns"]
                stage_explained_variance[stage_name] = _explained_variance(value_pred, value_target)
        _profile_add("explained_variance_sec", time.perf_counter() - _profile_t0)

        _append_critic_replay_bank()

        metrics = {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
            "approx_kl": float(np.mean(approx_kl_values)) if approx_kl_values else 0.0,
            "clip_frac": float(np.mean(clip_frac_values)) if clip_frac_values else 0.0,
            "value_loss_accel": float(np.mean(stage_value_losses["accel"])) if stage_value_losses["accel"] else 0.0,
            "value_loss_sat": float(np.mean(stage_value_losses["sat"])) if stage_value_losses["sat"] else 0.0,
            "value_loss_bw": float(np.mean(stage_value_losses["bw"])) if stage_value_losses["bw"] else 0.0,
            "critic_popart_mean_bw": (
                float(self.critic.popart_stats(2)[0].detach().cpu().item())
                if self.critic_popart_enabled and hasattr(self.critic, "popart_stats")
                else 0.0
            ),
            "critic_popart_std_bw": (
                float(self.critic.popart_stats(2)[1].detach().cpu().item())
                if self.critic_popart_enabled and hasattr(self.critic, "popart_stats")
                else 1.0
            ),
            "explained_variance_accel": float(stage_explained_variance["accel"]),
            "explained_variance_sat": float(stage_explained_variance["sat"]),
            "explained_variance_bw": float(stage_explained_variance["bw"]),
            "entropy_accel": float(np.mean(stage_entropy_values["accel"])) if stage_entropy_values["accel"] else 0.0,
            "entropy_sat": float(np.mean(stage_entropy_values["sat"])) if stage_entropy_values["sat"] else 0.0,
            "entropy_bw": float(np.mean(stage_entropy_values["bw"])) if stage_entropy_values["bw"] else 0.0,
            "approx_kl_accel": float(np.mean(stage_approx_kl_values["accel"])) if stage_approx_kl_values["accel"] else 0.0,
            "approx_kl_sat": float(np.mean(stage_approx_kl_values["sat"])) if stage_approx_kl_values["sat"] else 0.0,
            "approx_kl_bw": float(np.mean(stage_approx_kl_values["bw"])) if stage_approx_kl_values["bw"] else 0.0,
            "clip_frac_accel": float(np.mean(stage_clip_frac_values["accel"])) if stage_clip_frac_values["accel"] else 0.0,
            "clip_frac_sat": float(np.mean(stage_clip_frac_values["sat"])) if stage_clip_frac_values["sat"] else 0.0,
            "clip_frac_bw": float(np.mean(stage_clip_frac_values["bw"])) if stage_clip_frac_values["bw"] else 0.0,
            "grad_norm_accel": float(np.mean(stage_grad_norm_values["accel"])) if stage_grad_norm_values["accel"] else 0.0,
            "grad_norm_sat": float(np.mean(stage_grad_norm_values["sat"])) if stage_grad_norm_values["sat"] else 0.0,
            "grad_norm_bw": float(np.mean(stage_grad_norm_values["bw"])) if stage_grad_norm_values["bw"] else 0.0,
            "danger_imitation_loss": float(np.mean(danger_imitation_losses)) if danger_imitation_losses else 0.0,
            "danger_imitation_active_rate": float(danger_imitation_active_rate),
            "bw_counterfactual_credit_active_rate": float(bw_counterfactual_credit_metrics["active_rate"]),
            "bw_counterfactual_credit_agent_active_rate": float(
                bw_counterfactual_credit_metrics["agent_active_rate"]
            ),
            "bw_counterfactual_credit_mean": float(bw_counterfactual_credit_metrics["mean"]),
            "bw_counterfactual_credit_abs_mean": float(bw_counterfactual_credit_metrics["abs_mean"]),
            "bw_counterfactual_credit_positive_frac": float(bw_counterfactual_credit_metrics["positive_frac"]),
            "bw_flow_proxy_aux_loss": (
                float(np.mean(bw_flow_proxy_aux_losses)) if bw_flow_proxy_aux_losses else 0.0
            ),
            "bw_flow_proxy_regression_loss": (
                float(np.mean(bw_flow_proxy_regression_losses)) if bw_flow_proxy_regression_losses else 0.0
            ),
            "bw_flow_proxy_pairwise_acc": (
                float(np.mean(bw_flow_proxy_pairwise_accs)) if bw_flow_proxy_pairwise_accs else 0.0
            ),
            "bw_flow_proxy_pair_count": (
                float(np.mean(bw_flow_proxy_pair_counts)) if bw_flow_proxy_pair_counts else 0.0
            ),
            "bw_grad_norm_policy": (
                float(np.mean(bw_grad_norm_policy_values)) if bw_grad_norm_policy_values else 0.0
            ),
            "bw_grad_norm_aux_scaled": (
                float(np.mean(bw_grad_norm_aux_scaled_values)) if bw_grad_norm_aux_scaled_values else 0.0
            ),
            "bw_grad_ratio_aux_to_policy": (
                float(np.mean(bw_grad_ratio_aux_to_policy_values)) if bw_grad_ratio_aux_to_policy_values else 0.0
            ),
            "bw_score_head_grad_norm_policy": (
                float(np.mean(bw_score_head_grad_norm_policy_values))
                if bw_score_head_grad_norm_policy_values
                else 0.0
            ),
            "bw_score_head_grad_norm_aux_scaled": (
                float(np.mean(bw_score_head_grad_norm_aux_scaled_values))
                if bw_score_head_grad_norm_aux_scaled_values
                else 0.0
            ),
            "bw_score_head_grad_ratio_aux_to_policy": (
                float(np.mean(bw_score_head_grad_ratio_aux_to_policy_values))
                if bw_score_head_grad_ratio_aux_to_policy_values
                else 0.0
            ),
            "bw_abs_log_ratio_corr_valid_count": (
                float(np.mean(bw_abs_log_ratio_corr_valid_count_values))
                if bw_abs_log_ratio_corr_valid_count_values
                else 0.0
            ),
            "bw_abs_log_ratio_corr_latent_count": (
                float(np.mean(bw_abs_log_ratio_corr_latent_count_values))
                if bw_abs_log_ratio_corr_latent_count_values
                else 0.0
            ),
            "bw_kappa_mean": (
                float(np.mean(bw_kappa_mean_values))
                if bw_kappa_mean_values
                else 0.0
            ),
            "bw_kappa_p10": (
                float(np.mean(bw_kappa_p10_values))
                if bw_kappa_p10_values
                else 0.0
            ),
            "bw_kappa_p90": (
                float(np.mean(bw_kappa_p90_values))
                if bw_kappa_p90_values
                else 0.0
            ),
            "bw_kappa_hi_frac": (
                float(np.mean(bw_kappa_hi_frac_values))
                if bw_kappa_hi_frac_values
                else 0.0
            ),
            "bw_delta_student_teacher_corr": float(bw_delta_student_teacher_corr),
            "bw_delta_student_true_corr": float(bw_delta_student_true_corr),
            "bw_delta_actor_mix_alpha": float(bw_delta_actor_mix_alpha),
            "bw_delta_teacher_used": float(bw_delta_teacher_used),
            "bw_delta_teacher_observed": float(bw_delta_teacher_observed),
            "bw_delta_teacher_probe_only": float(bw_delta_teacher_probe_only),
            "bw_branch_gate_snr": float(bw_branch_gate_snr),
            "bw_branch_gate_triggered": float(bw_branch_gate_triggered),
            "bw_actor_update_skipped": float(bw_actor_update_skipped),
            "update_profile_old_logprob_sec": float(update_profile_times.get("old_logprob_sec", 0.0)),
            "update_profile_value_override_sec": float(update_profile_times.get("value_override_sec", 0.0)),
            "update_profile_returns_sec": float(update_profile_times.get("returns_sec", 0.0)),
            "update_profile_stage_cache_sec": float(update_profile_times.get("stage_cache_sec", 0.0)),
            "update_profile_critic_train_sec": float(update_profile_times.get("critic_train_sec", 0.0)),
            "update_profile_actor_train_sec": float(update_profile_times.get("actor_train_sec", 0.0)),
            "update_profile_explained_variance_sec": float(update_profile_times.get("explained_variance_sec", 0.0)),
        }
        if vs_ref_metrics:
            aggregate_keys = {"policy_loss", "entropy", "approx_kl", "clip_frac"}
            for key, value in vs_ref_metrics.items():
                value_f = float(value)
                if key in aggregate_keys:
                    metrics[key] = float(metrics.get(key, 0.0)) + value_f
                elif key.startswith("vs_ref_"):
                    metrics[key] = value_f
                elif key.endswith("_accel") or key.endswith("_sat") or key.endswith("_bw"):
                    if abs(value_f) > 0.0:
                        metrics[key] = value_f
                elif key not in metrics:
                    metrics[key] = value_f
        return metrics
