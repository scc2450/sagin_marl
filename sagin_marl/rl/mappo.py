from __future__ import annotations

from collections import deque
from dataclasses import replace
import csv
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .policy import ActorNet, batch_flatten_obs
from .critic import CriticNet, VALUE_HEAD_NAMES
from .buffer import RolloutBuffer
from .action_assembler import assemble_actions
from .baselines import cluster_center_queue_aware_policy, queue_aware_policy
from .distributions import squash_action
from ..utils.checkpoint import load_checkpoint_forgiving, load_state_dict_forgiving
from ..utils.normalization import RunningMeanStd
from ..env.config import ablation_flag

BW_PRIVATE_TRUNK_KEY_PREFIXES = (
    "bw_obs_norm.",
    "bw_fc1.",
    "bw_fc2.",
    "bw_own_encoder.",
    "bw_danger_nbr_encoder.",
    "bw_users_encoder.",
    "bw_sats_encoder.",
    "bw_nbrs_encoder.",
    "bw_fusion_fc1.",
    "bw_fusion_fc2.",
)


def _project_normalized_accel_np(accel: np.ndarray) -> np.ndarray:
    arr = np.asarray(accel, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    scale = np.minimum(1.0, 1.0 / np.maximum(norms, 1.0e-8))
    return (arr * scale).astype(np.float32, copy=False)


def compute_gae(rewards, values, bootstrap_values, episode_boundaries, gamma, lam):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    next_adv = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * bootstrap_values[t] - values[t]
        continue_mask = 1.0 if (t < T - 1 and not bool(episode_boundaries[t])) else 0.0
        next_adv = delta + gamma * lam * continue_mask * next_adv
        advantages[t] = next_adv
    returns = advantages + values
    return advantages, returns


def _stack_value_head_dict(value_heads: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([value_heads[head_name] for head_name in VALUE_HEAD_NAMES], dim=-1)


def _compute_per_head_gae_targets(
    rewards: np.ndarray,
    values_by_head: np.ndarray,
    bootstrap_by_head: np.ndarray,
    episode_boundaries: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    values_arr = np.asarray(values_by_head, dtype=np.float32)
    bootstrap_arr = np.asarray(bootstrap_by_head, dtype=np.float32)
    adv_by_head: Dict[str, np.ndarray] = {}
    ret_by_head: Dict[str, np.ndarray] = {}
    for head_idx, head_name in enumerate(VALUE_HEAD_NAMES):
        adv_h, ret_h = compute_gae(
            rewards,
            values_arr[:, head_idx],
            bootstrap_arr[:, head_idx],
            episode_boundaries,
            gamma,
            lam,
        )
        adv_by_head[head_name] = adv_h
        ret_by_head[head_name] = ret_h
    return adv_by_head, ret_by_head


def _normalize_advantages(advantages: np.ndarray, adv_clip: float) -> tuple[np.ndarray, Dict[str, float]]:
    adv_raw = np.asarray(advantages, dtype=np.float32)
    adv_raw_mean = float(np.mean(adv_raw))
    adv_raw_std = float(np.std(adv_raw))
    adv_preclip = (adv_raw - adv_raw_mean) / (adv_raw_std + 1e-8)
    if adv_clip > 0.0:
        adv_final = np.clip(adv_preclip, -adv_clip, adv_clip)
        clip_frac = float(np.count_nonzero(np.abs(adv_preclip) > adv_clip)) / float(max(adv_preclip.size, 1))
    else:
        adv_final = adv_preclip
        clip_frac = 0.0
    stats = {
        "raw_mean": adv_raw_mean,
        "raw_std": adv_raw_std,
        "preclip_mean": float(np.mean(adv_preclip)),
        "preclip_std": float(np.std(adv_preclip)),
        "postclip_mean": float(np.mean(adv_final)),
        "postclip_std": float(np.std(adv_final)),
        "clip_frac": clip_frac,
    }
    return adv_final.astype(np.float32, copy=False), stats


def _zscore_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size <= 0:
        return arr.astype(np.float32, copy=False)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mean) / std).astype(np.float32, copy=False)


def _compute_future_badness_signals(
    gu_queue_arrival_steps: np.ndarray,
    gu_drop_ratio_step: np.ndarray,
    episode_boundaries: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    queue_arr = np.asarray(gu_queue_arrival_steps, dtype=np.float32).reshape(-1)
    drop_arr = np.asarray(gu_drop_ratio_step, dtype=np.float32).reshape(-1)
    boundaries = np.asarray(episode_boundaries, dtype=bool).reshape(-1)
    n = queue_arr.shape[0]
    if n == 0 or horizon <= 0:
        zeros = np.zeros_like(queue_arr, dtype=np.float32)
        return zeros, zeros

    bad_queue = np.zeros(n, dtype=np.float32)
    bad_drop = np.zeros(n, dtype=np.float32)
    for t in range(n):
        last_idx = t
        drop_sum = 0.0
        idx = t
        for _ in range(horizon):
            if idx >= n - 1 or boundaries[idx]:
                break
            idx += 1
            last_idx = idx
            drop_sum += float(drop_arr[idx])
        if last_idx > t:
            bad_queue[t] = max(float(queue_arr[last_idx] - queue_arr[t]), 0.0)
            bad_drop[t] = float(drop_sum)
    return bad_queue, bad_drop


def _rolling_window_summary(values) -> Dict[str, float]:
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


def _single_env_step_stats(env) -> Dict[str, object]:
    def _safe_mean(arr) -> float:
        a = np.asarray(arr, dtype=np.float32)
        return float(np.mean(a)) if a.size > 0 else 0.0

    def _safe_max(arr) -> float:
        a = np.asarray(arr, dtype=np.float32)
        return float(np.max(a)) if a.size > 0 else 0.0

    def _safe_sum(arr) -> float:
        a = np.asarray(arr, dtype=np.float32)
        return float(np.sum(a)) if a.size > 0 else 0.0

    num_agents = len(getattr(env, "agents", []))
    default_accel = np.zeros((num_agents, 2), dtype=np.float32)
    default_bw = np.zeros((num_agents, getattr(env.cfg, "num_gu", 0)), dtype=np.float32)
    sat_k = max(int(getattr(env.cfg, "sat_action_select_k", getattr(env.cfg, "sat_num_select", env.cfg.N_RF)) or env.cfg.N_RF), 0)
    default_sat_mask = np.zeros((num_agents, getattr(env.cfg, "sats_obs_max", 0)), dtype=np.float32)
    default_sat_indices = np.full((num_agents, sat_k), -1, dtype=np.int64)
    parts = getattr(env, "last_reward_parts", None) or {}
    profile = getattr(env, "last_step_profile", None) or {}
    sat_processed = getattr(env, "last_sat_processed", None)
    sat_incoming = getattr(env, "last_sat_incoming", None)

    return {
        "last_exec_accel": np.asarray(getattr(env, "last_exec_accel", default_accel), dtype=np.float32),
        "last_exec_bw_alloc": np.asarray(getattr(env, "last_exec_bw_alloc", default_bw), dtype=np.float32),
        "last_exec_sat_select_mask": np.asarray(
            getattr(env, "last_exec_sat_select_mask", default_sat_mask),
            dtype=np.float32,
        ),
        "last_exec_sat_indices": np.asarray(
            getattr(env, "last_exec_sat_indices", default_sat_indices),
            dtype=np.int64,
        ),
        "danger_imitation_mask": np.asarray(
            getattr(env, "last_danger_imitation_mask", np.zeros((num_agents,), dtype=np.float32)),
            dtype=np.float32,
        ),
        "visible_raw_counts": np.asarray(
            getattr(env, "last_visible_raw_counts", np.zeros((num_agents,), dtype=np.float32)),
            dtype=np.float32,
        ),
        "visible_kept_counts": np.asarray(
            getattr(env, "last_visible_kept_counts", np.zeros((num_agents,), dtype=np.float32)),
            dtype=np.float32,
        ),
        "visible_stats": dict(getattr(env, "last_visible_stats", {}) or {}),
        "gu_queue_mean": _safe_mean(getattr(env, "gu_queue", 0.0)),
        "uav_queue_mean": _safe_mean(getattr(env, "uav_queue", 0.0)),
        "sat_queue_mean": _safe_mean(getattr(env, "sat_queue", 0.0)),
        "gu_queue_max": _safe_max(getattr(env, "gu_queue", 0.0)),
        "uav_queue_max": _safe_max(getattr(env, "uav_queue", 0.0)),
        "sat_queue_max": _safe_max(getattr(env, "sat_queue", 0.0)),
        "gu_drop_sum": _safe_sum(getattr(env, "gu_drop", 0.0)),
        "uav_drop_sum": _safe_sum(getattr(env, "uav_drop", 0.0)),
        "sat_drop_sum": _safe_sum(getattr(env, "sat_drop", 0.0)),
        "sat_processed_sum": _safe_sum(sat_processed) if sat_processed is not None else 0.0,
        "sat_incoming_sum": _safe_sum(sat_incoming) if sat_incoming is not None else 0.0,
        "connected_sat_count": float(getattr(env, "last_connected_sat_count", 0.0)),
        "connected_sat_dist_mean": float(getattr(env, "last_connected_sat_dist_mean", 0.0)),
        "connected_sat_dist_p95": float(getattr(env, "last_connected_sat_dist_p95", 0.0)),
        "connected_sat_elevation_deg_mean": float(
            getattr(env, "last_connected_sat_elevation_deg_mean", 0.0)
        ),
        "connected_sat_elevation_deg_min": float(
            getattr(env, "last_connected_sat_elevation_deg_min", 0.0)
        ),
        "assoc_centroid_dist_norms": np.asarray(
            getattr(env, "last_assoc_centroid_dist_norms", np.zeros((num_agents,), dtype=np.float32)),
            dtype=np.float32,
        ),
        "sat_overlap_uav": np.asarray(
            getattr(env, "last_sat_overlap_uav", np.zeros((num_agents,), dtype=np.float32)),
            dtype=np.float32,
        ),
        "energy_mean": _safe_mean(getattr(env, "uav_energy", 0.0)),
        "dynamics_time_sec": float(profile.get("dynamics_time_sec", 0.0)),
        "orbit_visible_time_sec": float(profile.get("orbit_visible_time_sec", 0.0)),
        "assoc_access_time_sec": float(profile.get("assoc_access_time_sec", 0.0)),
        "backhaul_queue_time_sec": float(profile.get("backhaul_queue_time_sec", 0.0)),
        "reward_time_sec": float(profile.get("reward_time_sec", 0.0)),
        "obs_time_sec": float(profile.get("obs_time_sec", 0.0)),
        "state_time_sec": float(profile.get("state_time_sec", 0.0)),
        "step_total_time_sec": float(profile.get("step_total_time_sec", 0.0)),
        "reward_parts": dict(parts),
    }


def _get_state_batch(env) -> np.ndarray:
    cached_states = getattr(env, "last_state_batch", None)
    if cached_states is not None:
        return np.asarray(cached_states, dtype=np.float32)
    if hasattr(env, "get_global_state_batch"):
        states = env.get_global_state_batch()
        return np.asarray(states, dtype=np.float32)
    return np.expand_dims(np.asarray(env.get_global_state(), dtype=np.float32), axis=0)


def _flatten_obs_env_batch(obs_env, cfg) -> Tuple[List[list], np.ndarray]:
    obs_lists_env = [list(obs_e.values()) for obs_e in obs_env]
    obs_step_batch = np.stack([batch_flatten_obs(obs_list, cfg) for obs_list in obs_lists_env], axis=0)
    return obs_lists_env, obs_step_batch.astype(np.float32, copy=False)


def _set_module_requires_grad(module, enabled: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = bool(enabled)


def _scale_module_grads(module, scale: float) -> None:
    if module is None:
        return
    scale_f = float(scale)
    if np.isclose(scale_f, 1.0):
        return
    for param in module.parameters():
        if param.grad is not None:
            param.grad.mul_(scale_f)


def _apply_bw_head_grad_scale(actor, scale: float) -> None:
    scale_f = float(scale)
    if np.isclose(scale_f, 1.0):
        return
    for module in getattr(actor, "bw_private_trunk_modules", lambda: tuple())():
        _scale_module_grads(module, scale_f)
    _scale_module_grads(getattr(actor, "bw_user_encoder", None), scale_f)
    _scale_module_grads(getattr(actor, "bw_scorer", None), scale_f)


def _configure_actor_trainability(actor, cfg, train_heads: Dict[str, bool]) -> None:
    _set_module_requires_grad(actor.mu_head, train_heads["accel"])
    actor.log_std.requires_grad = train_heads["accel"]
    for module in getattr(actor, "bw_private_trunk_modules", lambda: tuple())():
        _set_module_requires_grad(module, train_heads["bw"])
    _set_module_requires_grad(getattr(actor, "bw_user_encoder", None), train_heads["bw"])
    _set_module_requires_grad(getattr(actor, "bw_scorer", None), train_heads["bw"])
    _set_module_requires_grad(getattr(actor, "sat_action_encoder", None), train_heads["sat"])
    _set_module_requires_grad(getattr(actor, "sat_scorer", None), train_heads["sat"])

    train_shared_backbone = bool(getattr(cfg, "train_shared_backbone", True))
    train_fusion = bool(getattr(cfg, "train_fusion", False))
    train_fusion_last_layer = bool(getattr(cfg, "train_fusion_last_layer", False))
    if train_shared_backbone:
        return

    for module in actor.backbone_modules():
        _set_module_requires_grad(module, False)

    if train_fusion:
        if getattr(actor, "encoder_type", "flat_mlp") == "set_pool":
            _set_module_requires_grad(getattr(actor, "fusion_fc1", None), True)
            _set_module_requires_grad(getattr(actor, "fusion_fc2", None), True)
        else:
            _set_module_requires_grad(getattr(actor, "fc1", None), True)
            _set_module_requires_grad(getattr(actor, "fc2", None), True)
        return

    if train_fusion_last_layer:
        if getattr(actor, "encoder_type", "flat_mlp") == "set_pool":
            _set_module_requires_grad(getattr(actor, "fusion_fc2", None), True)
        else:
            _set_module_requires_grad(getattr(actor, "fc2", None), True)


def _module_states_match(target_module, source_module) -> bool:
    if target_module is None or source_module is None:
        return target_module is None and source_module is None
    target_state = target_module.state_dict()
    source_state = source_module.state_dict()
    if target_state.keys() != source_state.keys():
        return False
    for key, target_value in target_state.items():
        source_value = source_state[key]
        if tuple(target_value.shape) != tuple(source_value.shape):
            return False
        if not torch.equal(
            target_value.detach().cpu(),
            source_value.detach().to(device="cpu", dtype=target_value.dtype),
        ):
            return False
    return True


def _actor_backbone_matches(actor, source_actor) -> bool:
    target_modules = actor.backbone_modules()
    source_modules = source_actor.backbone_modules()
    if len(target_modules) != len(source_modules):
        return False
    for target_module, source_module in zip(target_modules, source_modules):
        if not _module_states_match(target_module, source_module):
            return False
    return True


def _actor_head_is_frozen(actor, head: str) -> bool:
    modules: List[torch.nn.Module] = []
    if head == "accel":
        modules.append(actor.mu_head)
        if bool(actor.log_std.requires_grad):
            return False
    elif head == "bw":
        modules.extend(list(getattr(actor, "bw_private_trunk_modules", lambda: tuple())()))
        if getattr(actor, "bw_user_encoder", None) is not None:
            modules.append(actor.bw_user_encoder)
        if getattr(actor, "bw_scorer", None) is not None:
            modules.append(actor.bw_scorer)
    elif head == "sat":
        if getattr(actor, "sat_action_encoder", None) is not None:
            modules.append(actor.sat_action_encoder)
        if getattr(actor, "sat_scorer", None) is not None:
            modules.append(actor.sat_scorer)
    else:
        raise ValueError(f"Unsupported actor head '{head}'")
    for module in modules:
        for param in module.parameters():
            if param.requires_grad:
                return False
    return True


def _copy_module_state_(target_module, source_module) -> None:
    if target_module is None or source_module is None:
        return
    target_module.load_state_dict(source_module.state_dict(), strict=True)


def _copy_actor_head_state_(actor, source_actor, head: str) -> None:
    if head == "accel":
        _copy_module_state_(actor.mu_head, source_actor.mu_head)
        actor.log_std.data.copy_(source_actor.log_std.detach())
        return
    if head == "bw":
        target_private_modules = tuple(getattr(actor, "bw_private_trunk_modules", lambda: tuple())())
        source_private_modules = tuple(getattr(source_actor, "bw_private_trunk_modules", lambda: tuple())())
        if target_private_modules and source_private_modules and len(target_private_modules) == len(source_private_modules):
            for target_module, source_module in zip(target_private_modules, source_private_modules):
                _copy_module_state_(target_module, source_module)
        _copy_module_state_(getattr(actor, "bw_user_encoder", None), getattr(source_actor, "bw_user_encoder", None))
        _copy_module_state_(getattr(actor, "bw_scorer", None), getattr(source_actor, "bw_scorer", None))
        return
    if head == "sat":
        _copy_module_state_(
            getattr(actor, "sat_action_encoder", None),
            getattr(source_actor, "sat_action_encoder", None),
        )
        _copy_module_state_(getattr(actor, "sat_scorer", None), getattr(source_actor, "sat_scorer", None))
        return
    raise ValueError(f"Unsupported actor head '{head}'")


def _maybe_init_bw_private_trunk_from_shared(actor, load_info: Dict[str, object] | None, source_name: str) -> None:
    if not bool(getattr(actor, "bw_private_trunk_enabled", False)):
        return
    if not bool(getattr(actor, "bw_private_trunk_init_from_shared", True)):
        return
    if not isinstance(load_info, dict):
        return
    missing_keys = [str(key) for key in (load_info.get("missing_keys", []) or [])]
    if not any(any(key.startswith(prefix) for prefix in BW_PRIVATE_TRUNK_KEY_PREFIXES) for key in missing_keys):
        return
    actor.initialize_bw_private_trunk_from_shared()
    print(
        "Initialized bw private trunk from shared trunk after loading "
        f"{source_name} because the checkpoint did not contain private bw trunk weights."
    )


def _resolve_frozen_teacher_exec_sources(
    actor,
    teacher_actor,
    train_heads: Dict[str, bool],
    exec_accel_source: str,
    exec_bw_source: str,
    exec_sat_source: str,
) -> Tuple[str, str, str, List[str], List[str]]:
    if teacher_actor is None:
        return exec_accel_source, exec_bw_source, exec_sat_source, [], []

    sources = {
        "accel": exec_accel_source,
        "bw": exec_bw_source,
        "sat": exec_sat_source,
    }
    resolved: List[str] = []
    kept: List[str] = []
    backbone_frozen = actor.backbone_is_frozen()
    backbone_matches = _actor_backbone_matches(actor, teacher_actor) if backbone_frozen else False

    for head in ("accel", "bw", "sat"):
        if sources[head] != "teacher":
            continue
        if bool(train_heads.get(head, False)):
            kept.append(f"{head}(trainable)")
            continue
        if not _actor_head_is_frozen(actor, head):
            kept.append(f"{head}(head_not_frozen)")
            continue
        if not backbone_frozen:
            kept.append(f"{head}(backbone_not_frozen)")
            continue
        if not backbone_matches:
            kept.append(f"{head}(backbone_mismatch)")
            continue
        _copy_actor_head_state_(actor, teacher_actor, head)
        sources[head] = "policy"
        resolved.append(head)

    return sources["accel"], sources["bw"], sources["sat"], resolved, kept


def _module_param_counts(module) -> Tuple[int, int]:
    if module is None:
        return 0, 0
    total = 0
    trainable = 0
    for param in module.parameters():
        total += int(param.numel())
        if param.requires_grad:
            trainable += int(param.numel())
    return total, trainable


def _format_actor_trainability_report(actor) -> str:
    module_names: List[Tuple[str, object]] = []
    if getattr(actor, "encoder_type", "flat_mlp") == "flat_mlp":
        module_names.extend(
            [
                ("obs_norm", getattr(actor, "obs_norm", None)),
                ("fc1", getattr(actor, "fc1", None)),
                ("fc2", getattr(actor, "fc2", None)),
            ]
        )
        if bool(getattr(actor, "bw_private_trunk_enabled", False)):
            module_names.extend(
                [
                    ("bw_obs_norm", getattr(actor, "bw_obs_norm", None)),
                    ("bw_fc1", getattr(actor, "bw_fc1", None)),
                    ("bw_fc2", getattr(actor, "bw_fc2", None)),
                ]
            )
    else:
        module_names.extend(
            [
                ("own_encoder", getattr(actor, "own_encoder", None)),
                ("danger_nbr_encoder", getattr(actor, "danger_nbr_encoder", None)),
                ("users_encoder", getattr(actor, "users_encoder", None)),
                ("sats_encoder", getattr(actor, "sats_encoder", None)),
                ("nbrs_encoder", getattr(actor, "nbrs_encoder", None)),
                ("fusion_fc1", getattr(actor, "fusion_fc1", None)),
                ("fusion_fc2", getattr(actor, "fusion_fc2", None)),
            ]
        )
        if bool(getattr(actor, "bw_private_trunk_enabled", False)):
            module_names.extend(
                [
                    ("bw_own_encoder", getattr(actor, "bw_own_encoder", None)),
                    ("bw_danger_nbr_encoder", getattr(actor, "bw_danger_nbr_encoder", None)),
                    ("bw_users_encoder", getattr(actor, "bw_users_encoder", None)),
                    ("bw_sats_encoder", getattr(actor, "bw_sats_encoder", None)),
                    ("bw_nbrs_encoder", getattr(actor, "bw_nbrs_encoder", None)),
                    ("bw_fusion_fc1", getattr(actor, "bw_fusion_fc1", None)),
                    ("bw_fusion_fc2", getattr(actor, "bw_fusion_fc2", None)),
                ]
            )
    module_names.extend(
        [
            ("mu_head", getattr(actor, "mu_head", None)),
            ("log_std", None),
            ("bw_user_encoder", getattr(actor, "bw_user_encoder", None)),
            ("bw_scorer", getattr(actor, "bw_scorer", None)),
            ("sat_action_encoder", getattr(actor, "sat_action_encoder", None)),
            ("sat_scorer", getattr(actor, "sat_scorer", None)),
        ]
    )

    lines: List[str] = []
    for name, module in module_names:
        if name == "log_std":
            total = int(actor.log_std.numel())
            trainable = total if bool(actor.log_std.requires_grad) else 0
        else:
            total, trainable = _module_param_counts(module)
        if total <= 0:
            continue
        if trainable <= 0:
            state = "frozen"
        elif trainable >= total:
            state = "trainable"
        else:
            state = "partial"
        lines.append(f"{name}={state}({trainable}/{total})")

    trainable_names = [name for name, param in actor.named_parameters() if param.requires_grad]
    return (
        "Actor module freeze map: "
        + ", ".join(lines)
        + "\nActor trainable parameters:\n  - "
        + "\n  - ".join(trainable_names)
    )


def _scheduled_three_phase_weight(
    update_idx: int,
    total_updates: int,
    weight_init: float,
    weight_mid: float,
    weight_floor: float,
    hold_ratio: float,
    mid_ratio: float,
) -> float:
    if weight_init <= 0.0 and weight_mid <= 0.0 and weight_floor <= 0.0:
        return 0.0
    hold = float(np.clip(hold_ratio, 0.0, 1.0))
    mid = float(np.clip(mid_ratio, hold, 1.0))
    floor = min(max(float(weight_floor), 0.0), max(float(weight_mid), 0.0))
    progress = 1.0
    if total_updates > 0:
        progress = float(np.clip((update_idx + 1) / max(total_updates, 1), 0.0, 1.0))
    if progress <= hold:
        return float(weight_init)
    if progress <= mid:
        alpha = (progress - hold) / max(mid - hold, 1e-9)
        return float(weight_init + (weight_mid - weight_init) * alpha)
    alpha = (progress - mid) / max(1.0 - mid, 1e-9)
    return float(weight_mid + (floor - weight_mid) * alpha)


def _estimate_aux_schedule_total_updates(cfg, planned_total_updates: int) -> int:
    if bool(getattr(cfg, "aux_schedule_use_planned_total_updates", False)):
        return max(int(planned_total_updates or 0), 1)
    estimates: List[int] = []
    checkpoint_eval_enabled = bool(getattr(cfg, "checkpoint_eval_enabled", False))
    checkpoint_eval_interval = max(int(getattr(cfg, "checkpoint_eval_interval_updates", 0) or 0), 0)
    checkpoint_eval_start = max(int(getattr(cfg, "checkpoint_eval_start_update", 0) or 0), 0)
    if checkpoint_eval_enabled and checkpoint_eval_interval > 0 and bool(
        getattr(cfg, "checkpoint_eval_early_stop_enabled", True)
    ):
        if checkpoint_eval_start <= 0:
            checkpoint_eval_start = checkpoint_eval_interval
        if bool(getattr(cfg, "checkpoint_eval_reward_early_stop_enabled", False)):
            reward_patience = max(int(getattr(cfg, "checkpoint_eval_reward_patience", 5) or 0), 1)
            estimates.append(checkpoint_eval_start + checkpoint_eval_interval * reward_patience)
        if bool(getattr(cfg, "checkpoint_eval_sat_drop_early_stop_enabled", True)):
            quality_patience = max(int(getattr(cfg, "checkpoint_eval_worsen_patience", 2) or 0), 1)
            estimates.append(checkpoint_eval_start + checkpoint_eval_interval * quality_patience)

    if bool(getattr(cfg, "early_stop_enabled", False)):
        early_stop_min_updates = max(int(getattr(cfg, "early_stop_min_updates", 0) or 0), 0)
        early_stop_window = max(int(getattr(cfg, "early_stop_window", 1) or 1), 1)
        early_stop_patience = max(int(getattr(cfg, "early_stop_patience", 1) or 1), 1)
        estimates.append(max(early_stop_min_updates, early_stop_window) + early_stop_patience - 1)

    if not estimates:
        return max(int(planned_total_updates or 0), 1)
    effective_total = min(max(int(planned_total_updates or 0), 1), min(estimates))
    return max(effective_total, 1)


def _compute_train_reward_adjustment(
    stats: Dict[str, object],
    cfg,
    update_idx: int,
    total_updates: int,
) -> Tuple[float, Dict[str, object]]:
    parts = dict(stats.get("reward_parts", {}) or {})
    num_uav = max(int(getattr(cfg, "num_uav", 0) or 0), 0)
    stage1_metric_uav = np.asarray(
        stats.get("assoc_centroid_dist_norms", np.zeros((num_uav,), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    stage3_metric_uav = np.asarray(
        stats.get("sat_overlap_uav", np.zeros((num_uav,), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    stage1_weight = 0.0
    stage1_metric = float(parts.get("assoc_centroid_dist_norm_mean", 0.0))
    stage1_term = 0.0
    if bool(getattr(cfg, "reward_stage1_assoc_centroid_enabled", False)):
        stage1_weight = _scheduled_three_phase_weight(
            update_idx,
            total_updates,
            max(float(getattr(cfg, "reward_stage1_assoc_centroid_weight_init", 0.0) or 0.0), 0.0),
            max(float(getattr(cfg, "reward_stage1_assoc_centroid_weight_mid", 0.0) or 0.0), 0.0),
            max(float(getattr(cfg, "reward_stage1_assoc_centroid_weight_floor", 0.0) or 0.0), 0.0),
            float(getattr(cfg, "reward_stage1_assoc_centroid_hold_ratio", 0.30) or 0.30),
            float(getattr(cfg, "reward_stage1_assoc_centroid_mid_ratio", 0.70) or 0.70),
        )
        stage1_term = -stage1_weight * stage1_metric

    stage3_weight = 0.0
    stage3_metric = float(parts.get("sat_overlap_eval", 0.0))
    stage3_term = 0.0
    if bool(getattr(cfg, "reward_stage3_sat_overlap_enabled", False)):
        stage3_weight = _scheduled_three_phase_weight(
            update_idx,
            total_updates,
            max(float(getattr(cfg, "reward_stage3_sat_overlap_weight_init", 0.0) or 0.0), 0.0),
            max(float(getattr(cfg, "reward_stage3_sat_overlap_weight_mid", 0.0) or 0.0), 0.0),
            max(float(getattr(cfg, "reward_stage3_sat_overlap_weight_floor", 0.0) or 0.0), 0.0),
            float(getattr(cfg, "reward_stage3_sat_overlap_hold_ratio", 0.40) or 0.40),
            float(getattr(cfg, "reward_stage3_sat_overlap_mid_ratio", 0.80) or 0.80),
        )
        stage3_term = -stage3_weight * stage3_metric

    reward_aux = stage1_term + stage3_term
    return reward_aux, {
        "reward_aux": reward_aux,
        "stage1_assoc_centroid_dist_norm_mean": stage1_metric,
        "stage1_assoc_centroid_dist_norm_uav": stage1_metric_uav,
        "stage1_assoc_centroid_weight": stage1_weight,
        "stage1_assoc_centroid_term": stage1_term,
        "stage3_sat_overlap_eval": stage3_metric,
        "stage3_sat_overlap_uav": stage3_metric_uav,
        "stage3_sat_overlap_weight": stage3_weight,
        "stage3_sat_overlap_term": stage3_term,
    }


def _compute_sat_counterfactual_credit(
    cf_rows: List[Dict[str, object]],
    actual_reward: float,
    actual_next_state: np.ndarray,
    actual_next_obs_step: np.ndarray,
    actual_done: bool,
    critic,
    device: torch.device,
    cfg,
    update_idx: int,
    total_updates: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    num_uav = max(int(getattr(cfg, "num_uav", 0) or 0), 0)
    credit = np.zeros((num_uav,), dtype=np.float32)
    mask = np.zeros((num_uav,), dtype=np.float32)
    metrics = {
        "sampled_contexts": 0.0,
        "selected_q_mean": 0.0,
        "baseline_q_mean": 0.0,
        "credit_mean": 0.0,
        "credit_abs_mean": 0.0,
    }
    if not cf_rows or num_uav <= 0:
        return credit, mask, metrics

    gamma = float(getattr(cfg, "gamma", 0.0) or 0.0)
    with torch.inference_mode():
        if actual_done:
            actual_bootstrap = 0.0
        else:
            next_state_t = torch.from_numpy(np.asarray(actual_next_state, dtype=np.float32)[None, :]).to(device)
            next_obs_t = torch.from_numpy(np.asarray(actual_next_obs_step, dtype=np.float32)[None, :, :]).to(device)
            actual_bootstrap = float(critic(next_state_t, next_obs_t).detach().cpu().numpy().reshape(-1)[0])
    actual_q = float(actual_reward) + (0.0 if actual_done else gamma * actual_bootstrap)

    alt_reward_totals: List[float] = []
    alt_done_flags: List[bool] = []
    alt_states: List[np.ndarray] = []
    alt_obs_steps: List[np.ndarray] = []
    row_slices: List[Tuple[int, int]] = []
    for row in cf_rows:
        alternatives = list(row.get("alternatives", []) or [])
        start_idx = len(alt_reward_totals)
        for alt in alternatives:
            reward_aux, _ = _compute_train_reward_adjustment(
                dict(alt.get("train_reward_stats", {}) or {}),
                cfg,
                update_idx,
                total_updates,
            )
            alt_reward_totals.append(float(alt.get("reward", 0.0)) + float(reward_aux))
            alt_done_flags.append(bool(alt.get("terminated", False)))
            alt_states.append(np.asarray(alt.get("next_state"), dtype=np.float32))
            alt_obs_steps.append(np.asarray(alt.get("next_obs_step"), dtype=np.float32))
        row_slices.append((start_idx, len(alt_reward_totals)))

    alt_bootstrap = np.zeros((len(alt_reward_totals),), dtype=np.float32)
    bootstrap_indices = [idx for idx, done in enumerate(alt_done_flags) if not done]
    if bootstrap_indices:
        state_batch = np.stack([alt_states[idx] for idx in bootstrap_indices], axis=0).astype(np.float32, copy=False)
        obs_batch = np.stack([alt_obs_steps[idx] for idx in bootstrap_indices], axis=0).astype(np.float32, copy=False)
        with torch.inference_mode():
            state_t = torch.from_numpy(state_batch).to(device)
            obs_t = torch.from_numpy(obs_batch).to(device)
            value_np = critic(state_t, obs_t).detach().cpu().numpy().reshape(-1)
        alt_bootstrap[np.asarray(bootstrap_indices, dtype=np.int64)] = value_np.astype(np.float32, copy=False)

    selected_q_values: List[float] = []
    baseline_q_values: List[float] = []
    credit_values: List[float] = []
    for row, (start_idx, end_idx) in zip(cf_rows, row_slices):
        uav_idx = int(row.get("uav", -1))
        if uav_idx < 0 or uav_idx >= num_uav:
            continue
        if end_idx <= start_idx:
            continue
        q_values = [actual_q]
        for idx in range(start_idx, end_idx):
            q_values.append(float(alt_reward_totals[idx]) + gamma * float(alt_bootstrap[idx]))
        baseline_q = float(np.mean(np.asarray(q_values, dtype=np.float32)))
        local_credit = float(actual_q - baseline_q)
        credit[uav_idx] = local_credit
        mask[uav_idx] = 1.0
        selected_q_values.append(actual_q)
        baseline_q_values.append(baseline_q)
        credit_values.append(local_credit)

    if credit_values:
        metrics["sampled_contexts"] = float(len(credit_values))
        metrics["selected_q_mean"] = float(np.mean(np.asarray(selected_q_values, dtype=np.float32)))
        metrics["baseline_q_mean"] = float(np.mean(np.asarray(baseline_q_values, dtype=np.float32)))
        metrics["credit_mean"] = float(np.mean(np.asarray(credit_values, dtype=np.float32)))
        metrics["credit_abs_mean"] = float(np.mean(np.abs(np.asarray(credit_values, dtype=np.float32))))
    return credit, mask, metrics


def _checkpoint_eval_best_summary_from_state(state: Dict[str, float]) -> Dict[str, float] | None:
    pre_backlog = float(state.get("best_pre_backlog_steps_eval", float("inf")))
    if not np.isfinite(pre_backlog):
        return None
    return {
        "reward_sum": float(state.get("best_model_reward_sum", -float("inf"))),
        "processed_ratio_eval": float(state.get("best_processed_ratio_eval", -float("inf"))),
        "drop_ratio_eval": float(state.get("best_drop_ratio_eval", float("inf"))),
        "pre_backlog_steps_eval": pre_backlog,
        "collision_episode_fraction": float(state.get("best_collision_episode_fraction", float("inf"))),
        "sat_overlap_eval": float(state.get("best_sat_overlap_eval", float("inf"))),
    }


def _checkpoint_eval_model_better(
    summary: Dict[str, float],
    best_summary: Dict[str, float] | None,
    cfg,
) -> bool:
    if best_summary is None:
        return True

    model_collision_threshold = max(
        float(getattr(cfg, "checkpoint_eval_model_collision_threshold", 0.0) or 0.0),
        0.0,
    )
    reward_tie_rel_tol = max(
        float(getattr(cfg, "checkpoint_eval_reward_tie_rel_tol", 0.05) or 0.0),
        0.0,
    )
    use_sat_overlap = bool(getattr(cfg, "checkpoint_eval_use_sat_overlap", False))

    current_collision = float(summary["collision_episode_fraction"])
    best_collision = float(best_summary["collision_episode_fraction"])
    current_collision_pass = current_collision <= model_collision_threshold
    best_collision_pass = best_collision <= model_collision_threshold
    if current_collision_pass != best_collision_pass:
        return current_collision_pass
    if not current_collision_pass:
        if current_collision < best_collision - 1e-12:
            return True
        if current_collision > best_collision + 1e-12:
            return False

    current_pre = float(summary["pre_backlog_steps_eval"])
    best_pre = float(best_summary["pre_backlog_steps_eval"])
    tie_band = reward_tie_rel_tol * max(abs(best_pre), 1.0)
    if current_pre < best_pre - tie_band:
        return True
    if current_pre > best_pre + tie_band:
        return False

    current_processed = float(summary["processed_ratio_eval"])
    best_processed = float(best_summary["processed_ratio_eval"])
    current_reward = float(summary["reward_sum"])
    best_reward = float(best_summary["reward_sum"])
    if use_sat_overlap:
        if current_processed > best_processed + 1e-12:
            return True
        if current_processed < best_processed - 1e-12:
            return False
        current_overlap = float(summary.get("sat_overlap_eval", 0.0))
        best_overlap = float(best_summary.get("sat_overlap_eval", float("inf")))
        if current_overlap < best_overlap - 1e-12:
            return True
        if current_overlap > best_overlap + 1e-12:
            return False
        return current_reward > best_reward + 1e-12

    if current_reward > best_reward + 1e-12:
        return True
    if current_reward < best_reward - 1e-12:
        return False
    return current_processed > best_processed + 1e-12


def _sum_selected_parts(parts: Dict[str, torch.Tensor], train_heads: Dict[str, bool]) -> torch.Tensor:
    order = ("accel", "bw", "sat")
    selected: List[torch.Tensor] = []
    for key in order:
        if train_heads.get(key, False):
            if key not in parts:
                raise ValueError(f"Missing action head '{key}' while it is marked trainable.")
            selected.append(parts[key])
    if not selected:
        raise ValueError("No trainable action heads selected. Check train_accel/train_bw/train_sat.")
    out = selected[0]
    for item in selected[1:]:
        out = out + item
    return out


def compute_sat_supervision_loss(
    sat_logits: torch.Tensor,
    sat_projected_rate: torch.Tensor,
    sat_valid_mask: torch.Tensor,
    k: int = 2,
    top1_coef: float = 1.0,
    topk_coef: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    valid = sat_valid_mask > 0.5
    active_rows = torch.any(valid, dim=-1)
    zero = sat_logits.new_zeros(())
    aux = {
        "sat_sup_loss_top1": 0.0,
        "sat_sup_loss_topk": 0.0,
        "sat_sup_top1_acc": 0.0,
        "sat_sup_exact_topk": 0.0,
    }
    if not torch.any(active_rows):
        return zero, aux

    logits_active = sat_logits[active_rows]
    rate_active = sat_projected_rate[active_rows]
    valid_active = valid[active_rows]
    neg_large = torch.tensor(-1.0e9, dtype=logits_active.dtype, device=logits_active.device)
    masked_logits = torch.where(valid_active, logits_active, neg_large)
    masked_rate = torch.where(valid_active, rate_active, neg_large)

    top1_idx = masked_rate.argmax(dim=-1)
    loss_top1 = F.cross_entropy(masked_logits, top1_idx)

    max_k = min(int(k), int(masked_logits.shape[-1]))
    valid_counts = valid_active.sum(dim=-1)
    use_counts = torch.clamp(valid_counts, min=1, max=max_k)
    topk_idx = masked_rate.topk(k=max_k, dim=-1).indices
    rank_idx = torch.arange(max_k, device=masked_logits.device).unsqueeze(0)
    take_mask = rank_idx < use_counts.unsqueeze(-1)
    safe_topk_idx = torch.where(take_mask, topk_idx, torch.zeros_like(topk_idx))

    target = torch.zeros_like(logits_active)
    target.scatter_(1, safe_topk_idx, take_mask.to(logits_active.dtype))
    bce_raw = F.binary_cross_entropy_with_logits(logits_active, target, reduction="none")
    valid_float = valid_active.to(logits_active.dtype)
    loss_topk = (bce_raw * valid_float).sum() / valid_float.sum().clamp_min(1.0)

    pred_top1 = masked_logits.argmax(dim=-1)
    pred_topk_idx = masked_logits.topk(k=max_k, dim=-1).indices
    safe_pred_idx = torch.where(take_mask, pred_topk_idx, torch.zeros_like(pred_topk_idx))
    pred_set = torch.zeros_like(logits_active)
    pred_set.scatter_(1, safe_pred_idx, take_mask.to(logits_active.dtype))
    exact_topk = (pred_set == target).all(dim=-1).to(logits_active.dtype).mean()
    top1_acc = (pred_top1 == top1_idx).to(logits_active.dtype).mean()

    loss = float(top1_coef) * loss_top1 + float(topk_coef) * loss_topk
    aux = {
        "sat_sup_loss_top1": float(loss_top1.detach().cpu().item()),
        "sat_sup_loss_topk": float(loss_topk.detach().cpu().item()),
        "sat_sup_top1_acc": float(top1_acc.detach().cpu().item()),
        "sat_sup_exact_topk": float(exact_topk.detach().cpu().item()),
    }
    return loss, aux


def _tensor_quantile(values: torch.Tensor, quantile: float) -> torch.Tensor:
    flat = values.reshape(-1)
    if flat.numel() <= 0:
        return torch.zeros((), dtype=values.dtype, device=values.device)
    q = float(np.clip(quantile, 0.0, 1.0))
    if flat.numel() == 1:
        return flat[0]
    return torch.quantile(flat, q)


def _compute_bw_danger_mask(
    part_adv_t: torch.Tensor,
    part_log_ratio: torch.Tensor,
    bw_valid_count: torch.Tensor | None,
    bw_queue_max: torch.Tensor | None,
    cfg,
) -> tuple[torch.Tensor, Dict[str, float]]:
    default_mask = torch.zeros_like(part_adv_t, dtype=torch.bool)
    default_metrics = {
        "sample_rate": 0.0,
        "tightclip_hit_rate": 0.0,
        "adv_threshold": 0.0,
        "valid_count_threshold": 0.0,
        "queue_max_threshold": 0.0,
    }
    if (
        not bool(getattr(cfg, "bw_danger_tighter_clip_enabled", False))
        or bw_valid_count is None
        or bw_queue_max is None
        or part_adv_t.numel() <= 0
    ):
        return default_mask, default_metrics

    adv_det = part_adv_t.detach()
    log_ratio_det = part_log_ratio.detach()
    valid_count_det = bw_valid_count.detach().to(dtype=adv_det.dtype)
    queue_max_det = bw_queue_max.detach().to(dtype=adv_det.dtype)
    adv_threshold = _tensor_quantile(torch.abs(adv_det), float(getattr(cfg, "bw_danger_adv_quantile", 0.5) or 0.5))
    valid_threshold = _tensor_quantile(
        valid_count_det,
        float(getattr(cfg, "bw_danger_valid_count_quantile", 0.5) or 0.5),
    )
    queue_threshold = _tensor_quantile(
        queue_max_det,
        float(getattr(cfg, "bw_danger_queue_max_quantile", 0.5) or 0.5),
    )
    positive_push = torch.logical_and(adv_det > adv_threshold, log_ratio_det > 0.0)
    high_competition = torch.logical_and(valid_count_det >= valid_threshold, queue_max_det >= queue_threshold)
    danger_mask = torch.logical_and(positive_push, high_competition)
    metrics = {
        "sample_rate": float(danger_mask.to(dtype=adv_det.dtype).mean().item()),
        "tightclip_hit_rate": 0.0,
        "adv_threshold": float(adv_threshold.item()),
        "valid_count_threshold": float(valid_threshold.item()),
        "queue_max_threshold": float(queue_threshold.item()),
    }
    if torch.any(danger_mask):
        danger_clip = min(
            float(getattr(cfg, "clip_ratio", 0.2) or 0.2),
            float(getattr(cfg, "bw_danger_clip_ratio", 0.1) or 0.1),
        )
        tight_hits = torch.logical_and(danger_mask, torch.exp(log_ratio_det) > (1.0 + danger_clip))
        metrics["tightclip_hit_rate"] = float(
            tight_hits.to(dtype=adv_det.dtype).sum().item() / max(float(danger_mask.sum().item()), 1.0)
        )
    return danger_mask, metrics


def _compute_part_policy_term(
    part_name: str,
    new_logprob: torch.Tensor,
    old_logprob: torch.Tensor,
    part_adv_t: torch.Tensor,
    clip_ratio: float,
    cfg,
    out: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    part_log_ratio = torch.clamp(new_logprob - old_logprob, -8.0, 8.0)
    part_ratio = torch.exp(part_log_ratio)
    lower = 1.0 - float(clip_ratio)
    upper_clip = torch.full_like(part_ratio, 1.0 + float(clip_ratio))
    bw_metrics = {
        "sample_rate": 0.0,
        "tightclip_hit_rate": 0.0,
        "adv_threshold": 0.0,
        "valid_count_threshold": 0.0,
        "queue_max_threshold": 0.0,
    }
    if part_name == "bw":
        danger_mask, bw_metrics = _compute_bw_danger_mask(
            part_adv_t,
            part_log_ratio,
            out.get("bw_valid_count"),
            out.get("bw_queue_max"),
            cfg,
        )
        if torch.any(danger_mask):
            tighter = min(float(clip_ratio), float(getattr(cfg, "bw_danger_clip_ratio", 0.1) or 0.1))
            upper_clip = torch.where(danger_mask, torch.full_like(upper_clip, 1.0 + tighter), upper_clip)
    clipped_ratio = torch.clamp(part_ratio, min=lower)
    clipped_ratio = torch.minimum(clipped_ratio, upper_clip)
    surr1 = part_ratio * part_adv_t
    surr2 = clipped_ratio * part_adv_t
    policy_term = -torch.min(surr1, surr2).mean()
    approx_part = (old_logprob - new_logprob).mean()
    return policy_term, part_log_ratio, part_ratio, approx_part, bw_metrics


def _selected_train_head_names(train_heads: Dict[str, bool]) -> Tuple[str, ...]:
    return tuple(head for head in ("accel", "bw", "sat") if bool(train_heads.get(head, False)))


def _exec_source_head_names(
    cfg,
    exec_accel_source: str,
    exec_bw_source: str,
    exec_sat_source: str,
    target_source: str,
) -> Tuple[str, ...]:
    heads: list[str] = []
    if exec_accel_source == target_source:
        heads.append("accel")
    if bool(cfg.enable_bw_action) and exec_bw_source == target_source:
        heads.append("bw")
    if not bool(cfg.fixed_satellite_strategy) and exec_sat_source == target_source:
        heads.append("sat")
    return tuple(heads)


def _merge_head_names(*head_groups: Tuple[str, ...]) -> Tuple[str, ...]:
    order = ("accel", "bw", "sat")
    merged: list[str] = []
    for head in order:
        if any(head in group for group in head_groups):
            merged.append(head)
    return tuple(merged)


def _normalize_exec_source(raw: str | None) -> str:
    src = str("policy" if raw is None else raw).strip().lower()
    if src == "heuristic_residual":
        src = "policy"
    allowed = {"policy", "teacher", "heuristic", "cluster_center_queue_aware", "zero"}
    if src not in allowed:
        raise ValueError(f"Invalid exec source '{raw}'. Allowed: {sorted(allowed)}")
    return src


def _normalize_checkpoint_eval_fixed_policy(raw: str | None) -> str:
    policy = str("zero" if raw is None else raw).strip().lower()
    allowed = {
        "zero",
        "queue_aware",
        "cluster_center_queue_aware",
        "teacher_accel_queue_aware",
        "stage2_exec_fixed_sat",
    }
    if policy not in allowed:
        raise ValueError(f"Invalid checkpoint-eval fixed policy '{raw}'. Allowed: {sorted(allowed)}")
    return policy


def _select_exec_values(
    source: str,
    policy_values: np.ndarray | None,
    teacher_values: np.ndarray | None,
    heuristic_values: np.ndarray | None,
    shape: Tuple[int, int],
) -> np.ndarray:
    if source == "policy" and policy_values is not None:
        return np.asarray(policy_values, dtype=np.float32)
    if source == "teacher" and teacher_values is not None:
        return np.asarray(teacher_values, dtype=np.float32)
    if source in {"heuristic", "cluster_center_queue_aware"} and heuristic_values is not None:
        return np.asarray(heuristic_values, dtype=np.float32)
    return np.zeros(shape, dtype=np.float32)


def _source_needs_heuristic(source: str) -> bool:
    return source in {"heuristic", "cluster_center_queue_aware"}


def _resolve_exec_heuristic_triplet(
    obs_list,
    cfg,
    env,
    accel_source: str,
    bw_source: str,
    sat_source: str,
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    queue_bundle = (None, None, None)
    cluster_bundle = (None, None, None)
    sources = {accel_source, bw_source, sat_source}

    if "heuristic" in sources:
        queue_bundle = queue_aware_policy(obs_list, cfg)

    if "cluster_center_queue_aware" in sources:
        centers = getattr(env, "gu_cluster_centers", None)
        counts = getattr(env, "gu_cluster_counts", None)
        cluster_bundle = cluster_center_queue_aware_policy(obs_list, cfg, centers, counts)

    def pick(source: str):
        if source == "cluster_center_queue_aware":
            return cluster_bundle
        if source == "heuristic":
            return queue_bundle
        return (None, None, None)

    accel_triplet = pick(accel_source)
    bw_triplet = pick(bw_source)
    sat_triplet = pick(sat_source)
    return accel_triplet[0], bw_triplet[1], sat_triplet[2]


def _compose_bw_exec_values(
    source: str,
    policy_values: np.ndarray | None,
    teacher_values: np.ndarray | None,
    heuristic_values: np.ndarray | None,
    shape: Tuple[int, int],
    cfg,
) -> np.ndarray:
    del cfg
    return _select_exec_values(source, policy_values, teacher_values, heuristic_values, shape)


def _compose_bw_train_action(
    source: str,
    policy_values: np.ndarray | None,
    shape: Tuple[int, int],
    cfg,
) -> np.ndarray:
    del source, cfg
    if policy_values is None:
        return np.zeros(shape, dtype=np.float32)
    return np.asarray(policy_values, dtype=np.float32)


def _bw_action_to_full_gu_from_obs(
    obs_list,
    bw_values: np.ndarray | None,
    cfg,
) -> np.ndarray:
    num_agents = len(obs_list)
    num_gu = int(cfg.num_gu)
    if bw_values is None:
        return np.zeros((num_agents, num_gu), dtype=np.float32)
    arr = np.asarray(bw_values, dtype=np.float32)
    if arr.shape == (num_agents, num_gu):
        return arr.astype(np.float32, copy=False)
    if arr.ndim != 2 or int(arr.shape[0]) != num_agents:
        raise ValueError(f"BW action must have shape ({num_agents}, {num_gu}) or candidate slots, got {tuple(arr.shape)}.")
    out = np.zeros((num_agents, num_gu), dtype=np.float32)
    for agent_idx, obs in enumerate(obs_list):
        candidate_raw = obs.get("candidate_indices") if isinstance(obs, dict) else None
        if candidate_raw is None:
            candidate = np.arange(int(arr.shape[1]), dtype=np.int64)
        else:
            candidate = np.asarray(candidate_raw, dtype=np.int64).reshape(-1)
        valid_slot = np.asarray(
            obs.get("bw_valid_mask", obs.get("users_mask", np.ones((arr.shape[1],), dtype=np.float32)))
            if isinstance(obs, dict)
            else np.ones((arr.shape[1],), dtype=np.float32),
            dtype=np.float32,
        ).reshape(-1) > 0.5
        width = min(int(arr.shape[1]), int(candidate.shape[0]), int(valid_slot.shape[0]))
        for slot in range(width):
            gu_idx = int(candidate[slot])
            if valid_slot[slot] and 0 <= gu_idx < num_gu:
                out[agent_idx, gu_idx] += max(float(arr[agent_idx, slot]), 0.0)
    return out.astype(np.float32, copy=False)


def _build_danger_imitation_step_data(stats: Dict[str, object], cfg, num_agents: int) -> Tuple[np.ndarray, np.ndarray]:
    accel_exec = np.asarray(
        stats.get("last_exec_accel", np.zeros((num_agents, 2), dtype=np.float32)),
        dtype=np.float32,
    )
    accel_exec_norm = _project_normalized_accel_np(accel_exec / max(float(cfg.a_max), 1e-6))
    mask_uav = np.asarray(
        stats.get("danger_imitation_mask", np.zeros((num_agents,), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(num_agents)
    mask_uav = (mask_uav > 0.5).astype(np.float32, copy=False)
    mask = np.repeat(mask_uav[:, None], 2, axis=1).astype(np.float32, copy=False)
    return accel_exec_norm, mask


def _format_checkpoint_load(prefix: str, info: Dict[str, object]) -> str:
    adapted = info.get("adapted_keys", [])
    skipped = info.get("skipped_keys", [])
    missing = info.get("missing_keys", [])
    unexpected = info.get("unexpected_keys", [])
    parts = [prefix]
    if adapted:
        parts.append(f"adapted={len(adapted)}")
    if skipped:
        parts.append(f"skipped={len(skipped)}")
    if missing:
        parts.append(f"missing={len(missing)}")
    if unexpected:
        parts.append(f"unexpected={len(unexpected)}")
    return ", ".join(parts)


def _save_checkpoints(log_dir: str, actor, critic, suffix: str | None = None) -> None:
    os.makedirs(log_dir, exist_ok=True)
    actor_name = "actor.pt" if suffix is None else f"actor_{suffix}.pt"
    critic_name = "critic.pt" if suffix is None else f"critic_{suffix}.pt"
    torch.save(actor.state_dict(), os.path.join(log_dir, actor_name))
    torch.save(critic.state_dict(), os.path.join(log_dir, critic_name))


def _reward_rms_state(reward_rms: RunningMeanStd | None) -> Dict[str, float] | None:
    if reward_rms is None:
        return None
    return {
        "mean": float(reward_rms.mean),
        "var": float(reward_rms.var),
        "count": float(reward_rms.count),
    }


def _restore_reward_rms(reward_rms: RunningMeanStd | None, state: Dict[str, object] | None) -> None:
    if reward_rms is None or not isinstance(state, dict):
        return
    if "mean" in state:
        reward_rms.mean = float(state["mean"])
    if "var" in state:
        reward_rms.var = float(state["var"])
    if "count" in state:
        reward_rms.count = float(state["count"])


def _save_train_state(
    log_dir: str,
    actor,
    critic,
    actor_optim,
    critic_optim,
    actor_sched,
    critic_sched,
    reward_rms: RunningMeanStd | None,
    update: int,
    planned_total_updates: int,
    total_env_steps: int,
    reward_history: List[float],
    best_ma: float,
    no_improve: int,
    checkpoint_eval_state: Dict[str, float],
    checkpoint_eval_fixed_summary: Dict[str, float] | None,
    total_time_sec: float,
    suffix: str | None = None,
) -> None:
    os.makedirs(log_dir, exist_ok=True)
    state_name = "train_state.pt" if suffix is None else f"train_state_{suffix}.pt"
    payload = {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer_state_dict": actor_optim.state_dict(),
        "critic_optimizer_state_dict": critic_optim.state_dict(),
        "actor_scheduler_state_dict": actor_sched.state_dict() if actor_sched is not None else None,
        "critic_scheduler_state_dict": critic_sched.state_dict() if critic_sched is not None else None,
        "reward_rms": _reward_rms_state(reward_rms),
        "update": int(update),
        "planned_total_updates": int(planned_total_updates),
        "total_env_steps": int(total_env_steps),
        "reward_history": [float(x) for x in reward_history],
        "best_ma": float(best_ma),
        "no_improve": int(no_improve),
        "checkpoint_eval_state": {str(k): float(v) for k, v in checkpoint_eval_state.items()},
        "checkpoint_eval_fixed_summary": checkpoint_eval_fixed_summary,
        "total_time_sec": float(total_time_sec),
    }
    torch.save(payload, os.path.join(log_dir, state_name))


def _load_train_state(
    path: str,
    actor,
    critic,
    actor_optim,
    critic_optim,
    actor_sched,
    critic_sched,
    reward_rms: RunningMeanStd | None,
    device: torch.device,
) -> Dict[str, object]:
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise TypeError(f"Train state '{path}' did not contain a dictionary payload.")
    actor_state = payload.get("actor_state_dict")
    critic_state = payload.get("critic_state_dict")
    if not isinstance(actor_state, dict) or not isinstance(critic_state, dict):
        raise KeyError(
            f"Train state '{path}' is missing 'actor_state_dict' or 'critic_state_dict'."
        )
    actor_info = load_state_dict_forgiving(actor, actor_state, strict=False)
    critic_info = load_state_dict_forgiving(critic, critic_state, strict=False)

    actor_optim_state = payload.get("actor_optimizer_state_dict")
    critic_optim_state = payload.get("critic_optimizer_state_dict")
    if actor_optim_state is not None:
        try:
            actor_optim.load_state_dict(actor_optim_state)
        except ValueError as exc:
            print(f"Warning: failed to load actor optimizer state from {path}: {exc}")
    if critic_optim_state is not None:
        try:
            critic_optim.load_state_dict(critic_optim_state)
        except ValueError as exc:
            print(f"Warning: failed to load critic optimizer state from {path}: {exc}")

    actor_sched_state = payload.get("actor_scheduler_state_dict")
    critic_sched_state = payload.get("critic_scheduler_state_dict")
    if actor_sched is not None and actor_sched_state is not None:
        actor_sched.load_state_dict(actor_sched_state)
    if critic_sched is not None and critic_sched_state is not None:
        critic_sched.load_state_dict(critic_sched_state)

    _restore_reward_rms(reward_rms, payload.get("reward_rms"))

    return {
        "path": path,
        "actor_info": actor_info,
        "critic_info": critic_info,
        "update": int(payload.get("update", 0) or 0),
        "planned_total_updates": int(payload.get("planned_total_updates", 0) or 0),
        "total_env_steps": int(payload.get("total_env_steps", 0) or 0),
        "reward_history": [float(x) for x in (payload.get("reward_history", []) or [])],
        "best_ma": float(payload.get("best_ma", -float("inf"))),
        "no_improve": int(payload.get("no_improve", 0) or 0),
        "checkpoint_eval_state": {
            str(k): float(v) for k, v in dict(payload.get("checkpoint_eval_state", {}) or {}).items()
        },
        "checkpoint_eval_fixed_summary": (
            dict(payload["checkpoint_eval_fixed_summary"])
            if payload.get("checkpoint_eval_fixed_summary") is not None
            else None
        ),
        "total_time_sec": float(payload.get("total_time_sec", 0.0) or 0.0),
    }


def _checkpoint_eval_summary(
    cfg,
    actor,
    device: torch.device,
    episodes: int,
    episode_seed_base: int | None,
    exec_accel_source: str,
    exec_bw_source: str,
    exec_sat_source: str,
    teacher_actor,
    teacher_deterministic: bool,
    need_heuristic_exec: bool,
    fixed_baseline: bool = False,
    fixed_baseline_policy: str = "zero",
) -> Dict[str, float]:
    from ..env.sagin_env import SaginParallelEnv

    eval_cfg = cfg
    if fixed_baseline and fixed_baseline_policy == "stage2_exec_fixed_sat":
        eval_cfg = replace(cfg, fixed_satellite_strategy=True, train_sat=False)

    eval_env = SaginParallelEnv(eval_cfg)
    policy_exec_heads = _exec_source_head_names(
        eval_cfg,
        exec_accel_source,
        exec_bw_source,
        exec_sat_source,
        "policy",
    )
    teacher_exec_heads = _exec_source_head_names(
        eval_cfg,
        exec_accel_source,
        exec_bw_source,
        exec_sat_source,
        "teacher",
    )
    try:
        reward_sum_total = 0.0
        processed_ratio_total = 0.0
        drop_ratio_total = 0.0
        pre_backlog_total = 0.0
        d_sys_total = 0.0
        x_acc_total = 0.0
        x_rel_total = 0.0
        g_pre_total = 0.0
        d_pre_total = 0.0
        sat_overlap_total = 0.0
        collision_total = 0.0

        for ep in range(max(int(episodes), 1)):
            ep_seed = None if episode_seed_base is None else int(episode_seed_base) + ep
            obs, _ = eval_env.reset(seed=ep_seed)
            done = False
            steps = 0
            reward_sum = 0.0
            processed_ratio_sum = 0.0
            drop_ratio_sum = 0.0
            pre_backlog_sum = 0.0
            x_acc_sum = 0.0
            x_rel_sum = 0.0
            g_pre_sum = 0.0
            d_pre_sum = 0.0
            sat_overlap_sum = 0.0
            sat_processed_sum_ep = 0.0
            collision_any = 0.0

            while not done:
                obs_list = list(obs.values())
                if fixed_baseline:
                    if fixed_baseline_policy == "queue_aware":
                        heur_accel, heur_bw, heur_sat = queue_aware_policy(obs_list, eval_cfg)
                        actions = assemble_actions(
                            eval_cfg,
                            eval_env.agents,
                            heur_accel,
                            bw_logits=heur_bw,
                            sat_logits=heur_sat,
                        )
                    elif fixed_baseline_policy == "cluster_center_queue_aware":
                        centers = getattr(eval_env, "gu_cluster_centers", None)
                        counts = getattr(eval_env, "gu_cluster_counts", None)
                        heur_accel, heur_bw, heur_sat = cluster_center_queue_aware_policy(
                            obs_list,
                            eval_cfg,
                            centers,
                            counts,
                        )
                        actions = assemble_actions(
                            eval_cfg,
                            eval_env.agents,
                            heur_accel,
                            bw_logits=heur_bw,
                            sat_logits=heur_sat,
                        )
                    elif fixed_baseline_policy == "teacher_accel_queue_aware":
                        if teacher_actor is None:
                            raise ValueError(
                                "checkpoint_eval_fixed_policy='teacher_accel_queue_aware' requires a teacher actor."
                            )
                        obs_batch = batch_flatten_obs(obs_list, cfg)
                        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
                        with torch.inference_mode():
                            teacher_out = teacher_actor.act(
                                obs_tensor,
                                deterministic=teacher_deterministic,
                                required_heads=("accel",),
                                compute_logprob=False,
                            )
                        _, heur_bw, heur_sat = queue_aware_policy(obs_list, eval_cfg)
                        actions = assemble_actions(
                            eval_cfg,
                            eval_env.agents,
                            teacher_out.accel.cpu().numpy(),
                            bw_alloc=heur_bw,
                            sat_select_mask=heur_sat,
                        )
                    elif fixed_baseline_policy == "stage2_exec_fixed_sat":
                        if actor is None:
                            raise ValueError(
                                "checkpoint_eval_fixed_policy='stage2_exec_fixed_sat' requires the current actor."
                            )
                        obs_batch = batch_flatten_obs(obs_list, eval_cfg)
                        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
                        with torch.inference_mode():
                            policy_out = actor.act(
                                obs_tensor,
                                deterministic=True,
                                required_heads=policy_exec_heads,
                                compute_logprob=False,
                            )
                        teacher_accel = None
                        teacher_bw = None
                        if teacher_actor is not None and teacher_exec_heads:
                            with torch.inference_mode():
                                teacher_out = teacher_actor.act(
                                    obs_tensor,
                                    deterministic=teacher_deterministic,
                                    required_heads=teacher_exec_heads,
                                    compute_logprob=False,
                                )
                            teacher_accel = (
                                teacher_out.accel.cpu().numpy()
                                if teacher_out.accel is not None
                                else None
                            )
                            teacher_bw = (
                                teacher_out.bw_action.cpu().numpy()
                                if teacher_out.bw_action is not None
                                else None
                            )
                        heur_accel = None
                        heur_bw = None
                        if need_heuristic_exec:
                            heur_accel, heur_bw, _ = _resolve_exec_heuristic_triplet(
                                obs_list,
                                eval_cfg,
                                eval_env,
                                exec_accel_source,
                                exec_bw_source,
                                "zero",
                            )
                        accel_actions = _select_exec_values(
                            exec_accel_source,
                            policy_out.accel.cpu().numpy() if policy_out.accel is not None else None,
                            teacher_accel,
                            heur_accel,
                            (len(eval_env.agents), 2),
                        )
                        bw_logits = None
                        if eval_cfg.enable_bw_action:
                            policy_bw = (
                                policy_out.bw_action.cpu().numpy() if policy_out.bw_action is not None else None
                            )
                            bw_logits = _compose_bw_exec_values(
                                exec_bw_source,
                                policy_bw,
                                teacher_bw,
                                heur_bw,
                                (len(eval_env.agents), eval_cfg.num_gu),
                                eval_cfg,
                            )
                            bw_logits = _bw_action_to_full_gu_from_obs(obs_list, bw_logits, eval_cfg)
                        actions = assemble_actions(
                            eval_cfg,
                            eval_env.agents,
                            accel_actions,
                            bw_alloc=bw_logits,
                            sat_select_mask=None,
                        )
                    else:
                        accel_actions = np.zeros((len(eval_env.agents), 2), dtype=np.float32)
                        actions = assemble_actions(eval_cfg, eval_env.agents, accel_actions)
                else:
                    obs_batch = batch_flatten_obs(obs_list, cfg)
                    obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
                    with torch.inference_mode():
                        policy_out = actor.act(
                            obs_tensor,
                            deterministic=True,
                            required_heads=policy_exec_heads,
                            compute_logprob=False,
                        )
                    teacher_accel = None
                    teacher_bw = None
                    teacher_sat = None
                    if teacher_actor is not None and teacher_exec_heads:
                        with torch.inference_mode():
                            teacher_out = teacher_actor.act(
                                obs_tensor,
                                deterministic=teacher_deterministic,
                                required_heads=teacher_exec_heads,
                                compute_logprob=False,
                            )
                        teacher_accel = (
                            teacher_out.accel.cpu().numpy()
                            if teacher_out.accel is not None
                            else None
                        )
                        teacher_bw = (
                            teacher_out.bw_action.cpu().numpy() if teacher_out.bw_action is not None else None
                        )
                        teacher_sat = (
                            teacher_out.sat_select_mask.cpu().numpy()
                            if teacher_out.sat_select_mask is not None
                            else None
                        )
                    heur_accel = None
                    heur_bw = None
                    heur_sat = None
                    if need_heuristic_exec:
                        heur_accel, heur_bw, heur_sat = _resolve_exec_heuristic_triplet(
                            obs_list,
                            cfg,
                            eval_env,
                            exec_accel_source,
                            exec_bw_source,
                            exec_sat_source,
                        )
                    accel_actions = _select_exec_values(
                        exec_accel_source,
                        policy_out.accel.cpu().numpy() if policy_out.accel is not None else None,
                        teacher_accel,
                        heur_accel,
                        (len(eval_env.agents), 2),
                    )
                    bw_logits = None
                    sat_logits = None
                    if cfg.enable_bw_action:
                        policy_bw = (
                            policy_out.bw_action.cpu().numpy() if policy_out.bw_action is not None else None
                        )
                        bw_logits = _compose_bw_exec_values(
                            exec_bw_source,
                            policy_bw,
                            teacher_bw,
                            heur_bw,
                            (len(eval_env.agents), cfg.num_gu),
                            cfg,
                        )
                        bw_logits = _bw_action_to_full_gu_from_obs(obs_list, bw_logits, cfg)
                    if not cfg.fixed_satellite_strategy:
                        policy_sat = (
                            policy_out.sat_select_mask.cpu().numpy()
                            if policy_out.sat_select_mask is not None
                            else None
                        )
                        sat_logits = _select_exec_values(
                            exec_sat_source,
                            policy_sat,
                            teacher_sat,
                            heur_sat,
                            (len(eval_env.agents), cfg.sats_obs_max),
                        )
                    actions = assemble_actions(
                        cfg,
                        eval_env.agents,
                        accel_actions,
                        bw_alloc=bw_logits,
                        sat_select_mask=sat_logits,
                    )

                obs, rewards, terms, truncs, _ = eval_env.step(actions)
                reward_sum += float(list(rewards.values())[0])
                done = bool(list(terms.values())[0] or list(truncs.values())[0])
                steps += 1
                parts = dict(getattr(eval_env, 'last_reward_parts', {}) or {})
                processed_ratio_sum += float(parts.get('processed_ratio_eval', 0.0))
                drop_ratio_sum += float(parts.get('drop_ratio_eval', 0.0))
                pre_backlog_sum += float(parts.get('pre_backlog_steps_eval', 0.0))
                x_acc_sum += float(parts.get('x_acc', 0.0))
                x_rel_sum += float(parts.get('x_rel', 0.0))
                g_pre_sum += float(parts.get('g_pre', 0.0))
                d_pre_sum += float(parts.get('d_pre', 0.0))
                sat_overlap_sum += float(parts.get('sat_overlap_eval', 0.0))
                sat_processed_sum_ep += float(parts.get('sat_processed_sum', 0.0))
                collision_any = max(collision_any, float(parts.get('collision_event', 0.0)))

            steps = max(steps, 1)
            reward_sum_total += reward_sum
            processed_ratio_total += processed_ratio_sum / float(steps)
            drop_ratio_total += drop_ratio_sum / float(steps)
            pre_backlog_total += pre_backlog_sum / float(steps)
            x_acc_total += x_acc_sum / float(steps)
            x_rel_total += x_rel_sum / float(steps)
            g_pre_total += g_pre_sum / float(steps)
            d_pre_total += d_pre_sum / float(steps)
            sat_overlap_total += sat_overlap_sum / float(steps)
            d_sys_total += (
                float(np.sum(eval_env.gu_queue) + np.sum(eval_env.uav_queue) + np.sum(eval_env.sat_queue))
                / max(sat_processed_sum_ep, 1e-9)
            )
            collision_total += collision_any

        denom = float(max(int(episodes), 1))
        return {
            'episodes': denom,
            'reward_sum': reward_sum_total / denom,
            'processed_ratio_eval': processed_ratio_total / denom,
            'drop_ratio_eval': drop_ratio_total / denom,
            'pre_backlog_steps_eval': pre_backlog_total / denom,
            'D_sys_report': d_sys_total / denom,
            'x_acc_mean': x_acc_total / denom,
            'x_rel_mean': x_rel_total / denom,
            'g_pre_mean': g_pre_total / denom,
            'd_pre_mean': d_pre_total / denom,
            'sat_overlap_eval': sat_overlap_total / denom,
            'collision_episode_fraction': collision_total / denom,
        }
    finally:
        close_fn = getattr(eval_env, 'close', None)
        if callable(close_fn):
            close_fn()


def _checkpoint_eval_fieldnames() -> List[str]:
    return [
        'update',
        'checkpoint_suffix',
        'episodes',
        'reward_sum',
        'processed_ratio_eval',
        'drop_ratio_eval',
        'pre_backlog_steps_eval',
        'D_sys_report',
        'x_acc_mean',
        'x_rel_mean',
        'g_pre_mean',
        'd_pre_mean',
        'sat_overlap_eval',
        'collision_episode_fraction',
        'fixed_reward_sum',
        'fixed_processed_ratio_eval',
        'fixed_drop_ratio_eval',
        'fixed_pre_backlog_steps_eval',
        'fixed_D_sys_report',
        'fixed_x_acc_mean',
        'fixed_x_rel_mean',
        'fixed_g_pre_mean',
        'fixed_d_pre_mean',
        'fixed_sat_overlap_eval',
        'fixed_collision_episode_fraction',
        'processed_improved',
        'drop_improved',
        'pre_backlog_improved',
        'model_improved',
        'quality_worsened',
        'quality_worse_streak',
        'reward_improved',
        'reward_plateau_streak',
        'collision_gate_passed',
        'quality_early_stop_triggered',
        'reward_early_stop_triggered',
        'early_stop_triggered',
    ]


def _append_checkpoint_eval_row(path: str, row: Dict[str, object]) -> None:
    fieldnames = _checkpoint_eval_fieldnames()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def _update_checkpoint_eval_state(
    state: Dict[str, float],
    summary: Dict[str, float],
    cfg,
) -> Dict[str, object]:
    reward_sum = float(summary["reward_sum"])
    processed = float(summary["processed_ratio_eval"])
    drop_ratio = float(summary["drop_ratio_eval"])
    pre_backlog = float(summary["pre_backlog_steps_eval"])
    collision_frac = float(summary["collision_episode_fraction"])
    rel_tol = max(float(getattr(cfg, "checkpoint_eval_front_queue_rel_improve_tol", 0.05) or 0.0), 0.0)
    worsen_delta = max(float(getattr(cfg, "checkpoint_eval_sat_drop_worsen_delta", 0.0) or 0.0), 0.0)
    patience = max(int(getattr(cfg, "checkpoint_eval_worsen_patience", 2) or 0), 1)
    sat_drop_early_stop_enabled = bool(getattr(cfg, "checkpoint_eval_sat_drop_early_stop_enabled", True))
    reward_early_stop_enabled = bool(getattr(cfg, "checkpoint_eval_reward_early_stop_enabled", False))
    reward_patience = max(int(getattr(cfg, "checkpoint_eval_reward_patience", 5) or 0), 1)
    reward_min_delta_rel = max(
        float(getattr(cfg, "checkpoint_eval_reward_min_delta_rel", 0.0) or 0.0),
        0.0,
    )
    reward_collision_threshold = max(
        float(getattr(cfg, "checkpoint_eval_reward_collision_threshold", 1.0) or 0.0),
        0.0,
    )

    best_processed_prev = float(state.get("best_processed_ratio_eval", -float("inf")))
    best_drop_prev = float(state.get("best_drop_ratio_eval", float("inf")))
    best_pre_backlog_prev = float(state.get("best_pre_backlog_steps_eval", float("inf")))
    prev_processed = state.get("prev_processed_ratio_eval", None)
    prev_drop = state.get("prev_drop_ratio_eval", None)
    prev_pre_backlog = state.get("prev_pre_backlog_steps_eval", None)
    best_reward_prev = float(state.get("best_reward_sum", -float("inf")))

    processed_improved = processed > best_processed_prev + worsen_delta
    drop_improved = drop_ratio < best_drop_prev - worsen_delta
    pre_backlog_improved = (
        (not np.isfinite(best_pre_backlog_prev))
        or pre_backlog < best_pre_backlog_prev * (1.0 - rel_tol)
    )
    best_summary_prev = _checkpoint_eval_best_summary_from_state(state)
    model_improved = _checkpoint_eval_model_better(summary, best_summary_prev, cfg)
    quality_worsened = (
        prev_processed is not None
        and prev_drop is not None
        and prev_pre_backlog is not None
        and processed < float(prev_processed) - worsen_delta
        and drop_ratio > float(prev_drop) + worsen_delta
        and pre_backlog > float(prev_pre_backlog) * (1.0 + rel_tol)
    )
    reward_margin = reward_min_delta_rel * max(abs(best_reward_prev), 1.0) if np.isfinite(best_reward_prev) else 0.0
    reward_improved = (not np.isfinite(best_reward_prev)) or reward_sum > (best_reward_prev + reward_margin)

    if model_improved:
        state["best_processed_ratio_eval"] = processed
        state["best_drop_ratio_eval"] = drop_ratio
        state["best_pre_backlog_steps_eval"] = pre_backlog
        state["best_model_reward_sum"] = reward_sum
        state["best_collision_episode_fraction"] = collision_frac
        state["best_sat_overlap_eval"] = float(summary.get("sat_overlap_eval", 0.0))
    if (not np.isfinite(best_reward_prev)) or reward_sum > best_reward_prev:
        state["best_reward_sum"] = reward_sum

    prev_streak = int(state.get("quality_worse_streak", 0))
    if quality_worsened and not model_improved:
        quality_worse_streak = prev_streak + 1
    else:
        quality_worse_streak = 0
    state["quality_worse_streak"] = float(quality_worse_streak)
    state["prev_processed_ratio_eval"] = processed
    state["prev_drop_ratio_eval"] = drop_ratio
    state["prev_pre_backlog_steps_eval"] = pre_backlog

    reward_prev_streak = int(state.get("reward_plateau_streak", 0))
    if reward_improved:
        reward_plateau_streak = 0
    else:
        reward_plateau_streak = reward_prev_streak + 1
    state["reward_plateau_streak"] = float(reward_plateau_streak)
    collision_gate_passed = collision_frac <= reward_collision_threshold

    quality_should_stop = sat_drop_early_stop_enabled and quality_worse_streak >= patience
    reward_should_stop = (
        reward_early_stop_enabled
        and reward_plateau_streak >= reward_patience
        and collision_gate_passed
    )
    should_stop = quality_should_stop or reward_should_stop
    return {
        "processed_improved": float(processed_improved),
        "drop_improved": float(drop_improved),
        "pre_backlog_improved": float(pre_backlog_improved),
        "model_improved": float(model_improved),
        "quality_worsened": float(quality_worsened),
        "quality_worse_streak": float(quality_worse_streak),
        "reward_improved": float(reward_improved),
        "reward_plateau_streak": float(reward_plateau_streak),
        "collision_gate_passed": float(collision_gate_passed),
        "quality_early_stop_triggered": float(quality_should_stop),
        "reward_early_stop_triggered": float(reward_should_stop),
        "early_stop_triggered": float(should_stop),
    }


def train(
    env,
    cfg,
    log_dir: str,
    total_updates: int = 50,
    save_interval_updates: int = 0,
    init_actor_path: str | None = None,
    init_critic_path: str | None = None,
    resume_state_path: str | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_interval_updates = max(0, int(save_interval_updates or 0))
    total_updates = max(0, int(total_updates or 0))

    num_envs = int(getattr(env, "num_envs", 1))

    # Determine observation and state dimensions
    obs_sample_raw, _ = env.reset()
    if num_envs > 1:
        if not isinstance(obs_sample_raw, list) or len(obs_sample_raw) != num_envs:
            raise ValueError("Vectorized env reset() must return a list of per-env observations.")
        obs_sample_env = obs_sample_raw
    else:
        obs_sample_env = [obs_sample_raw]
    obs_sample_lists_env, obs_sample_step_batch_np = _flatten_obs_env_batch(obs_sample_env, cfg)

    num_agents = len(env.agents)
    obs_dim = obs_sample_step_batch_np.shape[2]
    state_batch_np = _get_state_batch(env)
    global_state = state_batch_np[0]
    state_dim = global_state.shape[0]

    actor = ActorNet(obs_dim, cfg).to(device)
    critic = CriticNet(state_dim, obs_dim, num_agents, cfg).to(device)
    if resume_state_path and (init_actor_path or init_critic_path):
        raise ValueError("--resume_state cannot be combined with --init_actor/--init_critic.")
    if init_actor_path:
        try:
            info = load_checkpoint_forgiving(actor, init_actor_path, map_location=device)
            _maybe_init_bw_private_trunk_from_shared(actor, info, init_actor_path)
            print(_format_checkpoint_load(f"Loaded actor init from {init_actor_path}", info))
        except Exception as exc:
            print(f"Warning: failed to load actor init from {init_actor_path}: {exc}")
    if init_critic_path:
        try:
            info = load_checkpoint_forgiving(critic, init_critic_path, map_location=device)
            print(_format_checkpoint_load(f"Loaded critic init from {init_critic_path}", info))
        except Exception as exc:
            print(f"Warning: failed to load critic init from {init_critic_path}: {exc}")

    raw_train_accel = getattr(cfg, "train_accel", None)
    raw_train_bw = getattr(cfg, "train_bw", None)
    raw_train_sat = getattr(cfg, "train_sat", None)
    train_accel = True if raw_train_accel is None else bool(raw_train_accel)
    train_bw = bool(cfg.enable_bw_action) if raw_train_bw is None else bool(raw_train_bw)
    train_sat = (not bool(cfg.fixed_satellite_strategy)) if raw_train_sat is None else bool(raw_train_sat)
    if raw_train_bw is not None and train_bw and not cfg.enable_bw_action:
        raise ValueError("train_bw=true requires enable_bw_action=true")
    if raw_train_sat is not None and train_sat and cfg.fixed_satellite_strategy:
        raise ValueError("train_sat=true requires fixed_satellite_strategy=false")
    if not cfg.enable_bw_action:
        train_bw = False
    if cfg.fixed_satellite_strategy:
        train_sat = False
    train_heads = {
        "accel": train_accel,
        "bw": train_bw,
        "sat": train_sat,
    }
    if not any(train_heads.values()):
        raise ValueError("At least one of train_accel/train_bw/train_sat must be true.")
    train_head_names = _selected_train_head_names(train_heads)
    critic_multihead_value_enabled = bool(getattr(cfg, "critic_multihead_value_enabled", False))
    ppo_per_head_advantage_enabled = bool(getattr(cfg, "ppo_per_head_advantage_enabled", False))
    sat_counterfactual_credit_enabled = bool(getattr(cfg, "sat_counterfactual_credit_enabled", False))
    sat_counterfactual_credit_samples_per_update = max(
        int(getattr(cfg, "sat_counterfactual_credit_samples_per_update", 0) or 0),
        0,
    )
    sat_counterfactual_credit_weight = float(getattr(cfg, "sat_counterfactual_credit_weight", 1.0) or 1.0)
    sat_counterfactual_credit_weight = float(np.clip(sat_counterfactual_credit_weight, 0.0, 1.0))
    sat_counterfactual_credit_enabled = (
        sat_counterfactual_credit_enabled
        and bool(train_heads.get("sat", False))
        and not bool(getattr(cfg, "fixed_satellite_strategy", False))
        and sat_counterfactual_credit_samples_per_update > 0
    )
    if ppo_per_head_advantage_enabled and not critic_multihead_value_enabled:
        raise ValueError("ppo_per_head_advantage_enabled=true requires critic_multihead_value_enabled=true")
    bw_head_grad_scale_raw = getattr(cfg, "bw_head_grad_scale", 1.0)
    bw_head_grad_scale = 1.0 if bw_head_grad_scale_raw is None else float(bw_head_grad_scale_raw)
    if bw_head_grad_scale < 0.0:
        raise ValueError("bw_head_grad_scale must be >= 0")
    bw_continuous_credit_enabled = bool(getattr(cfg, "bw_continuous_credit_enabled", False))
    bw_continuous_credit_alpha_raw = getattr(cfg, "bw_continuous_credit_alpha", 0.0)
    bw_continuous_credit_alpha = 0.0 if bw_continuous_credit_alpha_raw is None else float(bw_continuous_credit_alpha_raw)
    if bw_continuous_credit_alpha < 0.0:
        raise ValueError("bw_continuous_credit_alpha must be >= 0")
    bw_false_positive_horizon = int(getattr(cfg, "bw_false_positive_horizon", 5) or 5)
    bw_continuous_credit_horizon = int(getattr(cfg, "bw_continuous_credit_horizon", 5) or 5)
    if bool(getattr(cfg, "bw_false_positive_mask_enabled", False)) and bw_false_positive_horizon <= 0:
        raise ValueError("bw_false_positive_horizon must be positive when bw_false_positive_mask_enabled=true")
    if bw_continuous_credit_enabled and bw_continuous_credit_horizon <= 0:
        raise ValueError("bw_continuous_credit_horizon must be positive when bw_continuous_credit_enabled=true")
    if (
        bw_continuous_credit_enabled
        and bool(getattr(cfg, "bw_false_positive_mask_enabled", False))
        and bw_continuous_credit_horizon != bw_false_positive_horizon
    ):
        raise ValueError(
            "bw_continuous_credit_horizon must match bw_false_positive_horizon when both bw credit modes are enabled"
        )
    bw_badness_horizon = (
        bw_continuous_credit_horizon
        if bw_continuous_credit_enabled
        else max(bw_false_positive_horizon, 1)
    )
    bw_continuous_credit_renorm_enabled = bool(getattr(cfg, "bw_continuous_credit_renorm_enabled", True))

    train_shared_backbone = bool(getattr(cfg, "train_shared_backbone", True))
    _configure_actor_trainability(actor, cfg, train_heads)
    cache_actor_ctx = actor.backbone_is_frozen()

    actor_params = [p for p in actor.parameters() if p.requires_grad]
    if not actor_params:
        raise ValueError("Actor has no trainable parameters after stage freeze settings.")

    actor_optim = torch.optim.Adam(actor_params, lr=cfg.actor_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    reward_rms = RunningMeanStd() if getattr(cfg, "reward_norm_enabled", False) else None
    resume_meta: Dict[str, object] | None = None
    resume_update = 0
    total_env_steps = 0
    best_ma = -float("inf")
    no_improve = 0
    reward_history: List[float] = []
    resumed_total_time_sec = 0.0
    planned_total_updates = total_updates
    if resume_state_path:
        resume_payload = torch.load(resume_state_path, map_location=device)
        if not isinstance(resume_payload, dict):
            raise TypeError(f"Train state '{resume_state_path}' did not contain a dictionary payload.")
        resume_update = int(resume_payload.get("update", 0) or 0)
        saved_planned_total_updates = int(resume_payload.get("planned_total_updates", 0) or 0)
        if saved_planned_total_updates > 0:
            planned_total_updates = saved_planned_total_updates
    aux_schedule_total_updates = _estimate_aux_schedule_total_updates(cfg, planned_total_updates)

    lr_decay_enabled = bool(getattr(cfg, "lr_decay_enabled", False))
    lr_final_factor = float(getattr(cfg, "lr_final_factor", 0.1) or 0.1)
    lr_final_factor = float(np.clip(lr_final_factor, 0.0, 1.0))
    actor_sched = None
    critic_sched = None
    if lr_decay_enabled and planned_total_updates > 1 and lr_final_factor < 1.0:
        def _linear_lr(step_idx: int) -> float:
            progress = min(max(step_idx, 0), planned_total_updates - 1) / max(planned_total_updates - 1, 1)
            return 1.0 - (1.0 - lr_final_factor) * progress

        actor_sched = torch.optim.lr_scheduler.LambdaLR(actor_optim, lr_lambda=_linear_lr)
        critic_sched = torch.optim.lr_scheduler.LambdaLR(critic_optim, lr_lambda=_linear_lr)
    if resume_state_path:
        resume_meta = _load_train_state(
            resume_state_path,
            actor=actor,
            critic=critic,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            actor_sched=actor_sched,
            critic_sched=critic_sched,
            reward_rms=reward_rms,
            device=device,
        )
        _maybe_init_bw_private_trunk_from_shared(actor, resume_meta.get("actor_info"), resume_state_path)
        resume_update = int(resume_meta.get("update", 0) or 0)
        total_env_steps = int(resume_meta.get("total_env_steps", 0) or 0)
        reward_history = [float(x) for x in (resume_meta.get("reward_history", []) or [])]
        best_ma = float(resume_meta.get("best_ma", -float("inf")))
        no_improve = int(resume_meta.get("no_improve", 0) or 0)
        resumed_total_time_sec = float(resume_meta.get("total_time_sec", 0.0) or 0.0)
        print(
            f"Resumed train state from {resume_state_path}: "
            f"completed_updates={resume_update}, total_env_steps={total_env_steps}"
        )
        print(_format_checkpoint_load(f"Resumed actor from {resume_state_path}", resume_meta["actor_info"]))
        print(_format_checkpoint_load(f"Resumed critic from {resume_state_path}", resume_meta["critic_info"]))

    from ..utils.logging import MetricLogger
    from ..utils.progress import Progress

    metric_fields = [
        "episode_reward",
        "episode_reward_std",
        "episode_reward_p25",
        "episode_reward_p75",
        "episode_length_mean",
        "completed_episode_count",
        "rollout_reward_per_step",
        "reward_raw",
        "reward_aux",
        "x_acc",
        "x_rel",
        "g_pre",
        "d_pre",
        "processed_ratio_eval",
        "drop_ratio_eval",
        "pre_backlog_steps_eval",
        "assoc_centroid_dist_norm_mean",
        "stage1_assoc_centroid_weight",
        "stage1_assoc_centroid_term",
        "sat_overlap_eval",
        "stage3_sat_overlap_weight",
        "stage3_sat_overlap_term",
        "D_sys_report",
        "episode_term_throughput_access",
        "episode_term_throughput_backhaul",
        "episode_term_queue_gu_arrival",
        "throughput_access_norm",
        "throughput_backhaul_norm",
        "gu_queue_arrival_steps",
        "gu_queue_mean",
        "uav_queue_mean",
        "queue_total_active",
        "collision_rate",
        "policy_loss",
        "value_loss",
        "ret_mean",
        "ret_std",
        "ret_p95",
        "explained_variance",
        "entropy",
        "entropy_accel",
        "entropy_bw",
        "entropy_sat",
        "accel_std_mean",
        "bw_std_mean",
        "bw_concentration_mean",
        "approx_kl",
        "approx_kl_accel",
        "approx_kl_bw",
        "approx_kl_sat",
        "clip_frac",
        "clip_frac_accel",
        "clip_frac_bw",
        "clip_frac_sat",
        "ratio_p50",
        "ratio_p90",
        "ratio_p99",
        "log_ratio_abs_mean",
        "log_ratio_mean_accel",
        "log_ratio_mean_bw",
        "log_ratio_mean_sat",
        "log_ratio_var_accel",
        "log_ratio_var_bw",
        "log_ratio_var_sat",
        "bw_false_positive_mask_rate",
        "bw_false_positive_bad_score_mean",
        "bw_false_positive_bad_tau",
        "bw_danger_sample_rate",
        "bw_danger_tightclip_hit_rate",
        "bw_danger_adv_threshold",
        "bw_danger_valid_count_threshold",
        "bw_danger_queue_max_threshold",
        "reward_rms_mean",
        "reward_rms_var",
        "danger_imitation_loss",
        "residual_reg_loss",
        "sat_sup_loss_top1",
        "sat_sup_loss_topk",
        "sat_sup_top1_acc",
        "sat_sup_exact_topk",
        "sat_cf_credit_sampled_contexts",
        "sat_cf_credit_selected_q_mean",
        "sat_cf_credit_baseline_q_mean",
        "sat_cf_credit_mean",
        "sat_cf_credit_abs_mean",
        "actor_epochs_executed",
        "actor_minibatches_executed",
        "actor_minibatches_executed_accel",
        "actor_minibatches_executed_bw",
        "actor_minibatches_executed_sat",
        "critic_updates_executed",
        "kl_stop_triggered",
        "kl_stop_triggered_accel",
        "kl_stop_triggered_bw",
        "kl_stop_triggered_sat",
        "danger_imitation_coef",
        "danger_imitation_active_rate",
        "log_std_mean",
        "actor_lr",
        "critic_lr",
        "update_time_sec",
        "rollout_time_sec",
        "optim_time_sec",
        "dynamics_time_sec",
        "orbit_visible_time_sec",
        "assoc_access_time_sec",
        "backhaul_queue_time_sec",
        "reward_time_sec",
        "obs_time_sec",
        "state_time_sec",
        "global_state_time_sec",
        "step_total_time_sec",
        "env_steps_per_sec",
        "update_steps_per_sec",
        "total_env_steps",
        "total_time_sec",
    ]
    tb_fields = [
        "episode_reward",
        "episode_reward_std",
        "episode_reward_p25",
        "episode_reward_p75",
        "rollout_reward_per_step",
        "reward_raw",
        "reward_aux",
        "x_acc",
        "x_rel",
        "g_pre",
        "d_pre",
        "processed_ratio_eval",
        "drop_ratio_eval",
        "pre_backlog_steps_eval",
        "assoc_centroid_dist_norm_mean",
        "sat_overlap_eval",
        "episode_term_throughput_access",
        "episode_term_throughput_backhaul",
        "episode_term_queue_gu_arrival",
        "throughput_access_norm",
        "throughput_backhaul_norm",
        "gu_queue_arrival_steps",
        "gu_queue_mean",
        "uav_queue_mean",
        "queue_total_active",
        "collision_rate",
        "policy_loss",
        "value_loss",
        "explained_variance",
        "entropy",
        "entropy_accel",
        "entropy_bw",
        "entropy_sat",
        "accel_std_mean",
        "bw_std_mean",
        "bw_concentration_mean",
        "approx_kl",
        "approx_kl_accel",
        "approx_kl_bw",
        "approx_kl_sat",
        "clip_frac",
        "clip_frac_accel",
        "clip_frac_bw",
        "clip_frac_sat",
        "log_ratio_mean_accel",
        "log_ratio_mean_bw",
        "log_ratio_mean_sat",
        "log_ratio_var_accel",
        "log_ratio_var_bw",
        "log_ratio_var_sat",
        "bw_false_positive_mask_rate",
        "bw_false_positive_bad_score_mean",
        "bw_false_positive_bad_tau",
        "bw_danger_sample_rate",
        "bw_danger_tightclip_hit_rate",
        "bw_danger_adv_threshold",
        "bw_danger_valid_count_threshold",
        "bw_danger_queue_max_threshold",
        "danger_imitation_loss",
        "residual_reg_loss",
        "sat_sup_loss_top1",
        "sat_sup_loss_topk",
        "sat_sup_top1_acc",
        "sat_sup_exact_topk",
        "sat_cf_credit_sampled_contexts",
        "sat_cf_credit_selected_q_mean",
        "sat_cf_credit_baseline_q_mean",
        "sat_cf_credit_mean",
        "sat_cf_credit_abs_mean",
        "actor_epochs_executed",
        "actor_minibatches_executed",
        "actor_minibatches_executed_accel",
        "actor_minibatches_executed_bw",
        "actor_minibatches_executed_sat",
        "critic_updates_executed",
        "kl_stop_triggered",
        "kl_stop_triggered_accel",
        "kl_stop_triggered_bw",
        "kl_stop_triggered_sat",
        "danger_imitation_coef",
        "danger_imitation_active_rate",
    ]
    env_step_fields = [
        "episode_reward",
        "episode_reward_std",
        "episode_reward_p25",
        "episode_reward_p75",
        "rollout_reward_per_step",
        "reward_raw",
        "reward_aux",
        "x_acc",
        "x_rel",
        "g_pre",
        "d_pre",
        "processed_ratio_eval",
        "drop_ratio_eval",
        "pre_backlog_steps_eval",
        "assoc_centroid_dist_norm_mean",
        "sat_overlap_eval",
        "episode_term_throughput_access",
        "episode_term_throughput_backhaul",
        "episode_term_queue_gu_arrival",
        "throughput_access_norm",
        "throughput_backhaul_norm",
        "gu_queue_arrival_steps",
        "gu_queue_mean",
        "uav_queue_mean",
        "queue_total_active",
        "collision_rate",
    ]
    stage1_agent_metric_names = [f"stage1_aux_agent{i}" for i in range(num_agents)]
    stage3_agent_metric_names = [f"stage3_overlap_agent{i}" for i in range(num_agents)]
    metric_fields.extend(stage1_agent_metric_names + stage3_agent_metric_names)
    tb_fields.extend(stage1_agent_metric_names + stage3_agent_metric_names)
    env_step_fields.extend(stage1_agent_metric_names + stage3_agent_metric_names)
    logger = MetricLogger(
        log_dir,
        fieldnames=metric_fields,
        tb_fields=tb_fields,
        env_step_fields=env_step_fields,
    )
    progress = Progress(total_updates, desc="Train")
    training_start = time.perf_counter() - resumed_total_time_sec
    imitation_coef_start = float(getattr(cfg, "imitation_coef", 0.0) or 0.0)
    imitation_coef_final_cfg = getattr(cfg, "imitation_coef_final", None)
    imitation_coef_final = (
        imitation_coef_start
        if imitation_coef_final_cfg is None
        else float(imitation_coef_final_cfg)
    )
    imitation_coef_decay_start_update = int(getattr(cfg, "imitation_coef_decay_start_update", 0) or 0)
    imitation_coef_decay_updates = int(getattr(cfg, "imitation_coef_decay_updates", 0) or 0)

    def _imitation_coef_at(update_idx: int) -> float:
        if imitation_coef_decay_updates <= 0:
            return imitation_coef_start
        if update_idx < imitation_coef_decay_start_update:
            return imitation_coef_start
        progress = min(
            1.0,
            float(max(update_idx - imitation_coef_decay_start_update, 0))
            / float(max(imitation_coef_decay_updates - 1, 1)),
        )
        return imitation_coef_start + (imitation_coef_final - imitation_coef_start) * progress

    use_imitation_loss = ablation_flag(
        cfg,
        "use_imitation_loss",
        fallback_attr="imitation_enabled",
        default=False,
    )
    use_heuristic_mask = ablation_flag(cfg, "use_heuristic_mask", default=False)
    imitation_enabled = use_imitation_loss and max(imitation_coef_start, imitation_coef_final) > 0.0
    imitation_accel = bool(getattr(cfg, "imitation_accel", True)) and train_heads["accel"]
    imitation_bw = bool(getattr(cfg, "imitation_bw", True)) and train_heads["bw"]
    imitation_sat = bool(getattr(cfg, "imitation_sat", False)) and train_heads["sat"]
    bw_residual_l2_coef = max(float(getattr(cfg, "bw_residual_l2_coef", 0.0) or 0.0), 0.0)
    imitation_coef_curr = _imitation_coef_at(resume_update)
    imitation_enabled_curr = imitation_enabled and imitation_coef_curr > 0.0
    danger_imitation_coef = max(float(getattr(cfg, "danger_imitation_coef", 0.0) or 0.0), 0.0)
    danger_imitation_enabled = bool(getattr(cfg, "danger_imitation_enabled", False)) and danger_imitation_coef > 0.0
    if danger_imitation_enabled and not train_heads["accel"]:
        raise ValueError("danger_imitation_enabled=true requires train_accel=true")
    teacher_deterministic = bool(getattr(cfg, "exec_teacher_deterministic", True))
    exec_accel_source = _normalize_exec_source(getattr(cfg, "exec_accel_source", "policy"))
    exec_bw_source = _normalize_exec_source(getattr(cfg, "exec_bw_source", "policy"))
    exec_sat_source = _normalize_exec_source(getattr(cfg, "exec_sat_source", "policy"))
    need_teacher_exec = "teacher" in {exec_accel_source, exec_bw_source, exec_sat_source}
    need_heuristic_exec = any(
        _source_needs_heuristic(src) for src in (exec_accel_source, exec_bw_source, exec_sat_source)
    )

    teacher_actor = None
    if need_teacher_exec:
        teacher_path = getattr(cfg, "exec_teacher_actor_path", None) or init_actor_path
        if not teacher_path:
            raise ValueError(
                "Execution override uses 'teacher' but no teacher checkpoint is provided. "
                "Set exec_teacher_actor_path or pass --init_actor."
            )
        teacher_actor = ActorNet(obs_dim, cfg).to(device)
        info = load_checkpoint_forgiving(teacher_actor, teacher_path, map_location=device)
        print(_format_checkpoint_load(f"Loaded teacher actor from {teacher_path}", info))
        teacher_actor.eval()
        for p in teacher_actor.parameters():
            p.requires_grad = False
        (
            exec_accel_source,
            exec_bw_source,
            exec_sat_source,
            resolved_teacher_heads,
            kept_teacher_heads,
        ) = _resolve_frozen_teacher_exec_sources(
            actor,
            teacher_actor,
            train_heads,
            exec_accel_source,
            exec_bw_source,
            exec_sat_source,
        )
        if resolved_teacher_heads:
            print(
                "Resolved frozen teacher exec heads to current actor: "
                + ", ".join(resolved_teacher_heads)
            )
        if kept_teacher_heads:
            print(
                "Keeping runtime teacher exec for heads: "
                + ", ".join(kept_teacher_heads)
            )
    policy_exec_heads = _exec_source_head_names(
        cfg,
        exec_accel_source,
        exec_bw_source,
        exec_sat_source,
        "policy",
    )
    teacher_exec_heads = _exec_source_head_names(
        cfg,
        exec_accel_source,
        exec_bw_source,
        exec_sat_source,
        "teacher",
    )
    rollout_policy_heads = _merge_head_names(policy_exec_heads, train_head_names)
    print(
        "Train heads: "
        f"accel={train_heads['accel']}, bw={train_heads['bw']}, sat={train_heads['sat']} | "
        f"train_shared_backbone={train_shared_backbone}, "
        f"train_fusion={bool(getattr(cfg, 'train_fusion', False))}, "
        f"train_fusion_last_layer={bool(getattr(cfg, 'train_fusion_last_layer', False))} | "
        f"backbone_ctx_cache={cache_actor_ctx} | "
        f"aux_schedule_total_updates={aux_schedule_total_updates} | "
        f"danger_imitation={danger_imitation_enabled} | "
        "Exec sources: "
        f"accel={exec_accel_source}, bw={exec_bw_source}, sat={exec_sat_source}"
    )
    print(_format_actor_trainability_report(actor))

    obs_lists_env = [list(obs_list) for obs_list in obs_sample_lists_env]
    obs_step_batch_np = np.asarray(obs_sample_step_batch_np, dtype=np.float32)
    checkpoint_eval_interval = max(int(getattr(cfg, "checkpoint_eval_interval_updates", 0) or 0), 0)
    checkpoint_eval_enabled = (
        bool(getattr(cfg, "checkpoint_eval_enabled", False))
        and checkpoint_eval_interval > 0
        and int(getattr(cfg, "checkpoint_eval_episodes", 0) or 0) > 0
    )
    checkpoint_eval_start_update = max(int(getattr(cfg, "checkpoint_eval_start_update", 0) or 0), 0)
    if checkpoint_eval_enabled and checkpoint_eval_start_update <= 0:
        checkpoint_eval_start_update = checkpoint_eval_interval
    checkpoint_eval_episodes = max(int(getattr(cfg, "checkpoint_eval_episodes", 20) or 0), 1)
    checkpoint_eval_episode_seed_base = getattr(cfg, "checkpoint_eval_episode_seed_base", None)
    checkpoint_eval_fixed_policy = _normalize_checkpoint_eval_fixed_policy(
        getattr(cfg, "checkpoint_eval_fixed_policy", "zero")
    )
    checkpoint_eval_csv_path = os.path.join(log_dir, "checkpoint_eval.csv")
    checkpoint_eval_state: Dict[str, float] = (
        dict(resume_meta.get("checkpoint_eval_state", {}) or {}) if resume_meta is not None else {}
    )
    checkpoint_eval_fixed_summary: Dict[str, float] | None = (
        dict(resume_meta.get("checkpoint_eval_fixed_summary", {}) or {})
        if resume_meta is not None and resume_meta.get("checkpoint_eval_fixed_summary") is not None
        else None
    )
    checkpoint_eval_early_stop_enabled = bool(getattr(cfg, "checkpoint_eval_early_stop_enabled", True))
    best_checkpoint_saved = (
        os.path.exists(os.path.join(log_dir, "actor_best.pt"))
        and os.path.exists(os.path.join(log_dir, "critic_best.pt"))
    )
    if checkpoint_eval_enabled:
        if resume_meta is None and os.path.exists(checkpoint_eval_csv_path):
            os.remove(checkpoint_eval_csv_path)
        if checkpoint_eval_fixed_summary is None:
            checkpoint_eval_fixed_summary = _checkpoint_eval_summary(
                cfg,
                actor=actor,
                device=device,
                episodes=checkpoint_eval_episodes,
                episode_seed_base=checkpoint_eval_episode_seed_base,
                exec_accel_source=exec_accel_source,
                exec_bw_source=exec_bw_source,
                exec_sat_source=exec_sat_source,
                teacher_actor=teacher_actor,
                teacher_deterministic=teacher_deterministic,
                need_heuristic_exec=need_heuristic_exec,
                fixed_baseline=True,
                fixed_baseline_policy=checkpoint_eval_fixed_policy,
            )
            print(
                f"Checkpoint eval {checkpoint_eval_fixed_policy} reference: "
                f"reward={checkpoint_eval_fixed_summary['reward_sum']:.4f}, "
                f"processed={checkpoint_eval_fixed_summary['processed_ratio_eval']:.4f}, "
                f"drop={checkpoint_eval_fixed_summary['drop_ratio_eval']:.4f}, "
                f"pre_backlog={checkpoint_eval_fixed_summary['pre_backlog_steps_eval']:.4f}, "
                f"sat_overlap={checkpoint_eval_fixed_summary['sat_overlap_eval']:.4f}, "
                f"collision={checkpoint_eval_fixed_summary['collision_episode_fraction']:.4f}"
            )
        else:
            print(
                f"Reused checkpoint eval {checkpoint_eval_fixed_policy} reference from {resume_state_path}: "
                f"reward={checkpoint_eval_fixed_summary['reward_sum']:.4f}, "
                f"processed={checkpoint_eval_fixed_summary['processed_ratio_eval']:.4f}, "
                f"drop={checkpoint_eval_fixed_summary['drop_ratio_eval']:.4f}, "
                f"pre_backlog={checkpoint_eval_fixed_summary['pre_backlog_steps_eval']:.4f}, "
                f"sat_overlap={checkpoint_eval_fixed_summary['sat_overlap_eval']:.4f}, "
                f"collision={checkpoint_eval_fixed_summary['collision_episode_fraction']:.4f}"
            )

    def _build_imitation_target(obs_list):
        if not imitation_enabled_curr:
            return None
        base_accel, base_bw, base_sat = queue_aware_policy(obs_list, cfg)
        parts = []
        if imitation_accel:
            parts.append(base_accel)
        else:
            parts.append(np.zeros_like(base_accel))
        if cfg.enable_bw_action:
            if imitation_bw:
                parts.append(base_bw)
            else:
                parts.append(np.zeros_like(base_bw))
        if not cfg.fixed_satellite_strategy:
            if imitation_sat:
                parts.append(base_sat)
            else:
                parts.append(np.zeros_like(base_sat))
        return np.concatenate(parts, axis=1).astype(np.float32, copy=False)

    completed_updates = resume_update
    episode_step_counts_env = np.zeros((num_envs,), dtype=np.int64)
    episode_assoc_unfair_step_counts_env = np.zeros((num_envs,), dtype=np.int64)
    episode_return_env = np.zeros((num_envs,), dtype=np.float64)
    episode_term_throughput_access_env = np.zeros((num_envs,), dtype=np.float64)
    episode_term_throughput_backhaul_env = np.zeros((num_envs,), dtype=np.float64)
    episode_term_queue_gu_arrival_env = np.zeros((num_envs,), dtype=np.float64)
    episode_collision_env = np.zeros((num_envs,), dtype=np.float32)
    episode_stat_window = max(1, int(getattr(cfg, "train_episode_stat_window", 100) or 100))
    recent_episode_returns: deque[float] = deque(maxlen=episode_stat_window)
    recent_episode_lengths: deque[float] = deque(maxlen=episode_stat_window)
    recent_episode_term_throughput_access: deque[float] = deque(maxlen=episode_stat_window)
    recent_episode_term_throughput_backhaul: deque[float] = deque(maxlen=episode_stat_window)
    recent_episode_term_queue_gu_arrival: deque[float] = deque(maxlen=episode_stat_window)
    recent_episode_collisions: deque[float] = deque(maxlen=episode_stat_window)
    debug_first_minibatch_ratio = bool(getattr(cfg, "debug_first_minibatch_ratio", False))
    debug_first_minibatch_ratio_printed = False
    for local_update in range(total_updates):
        update = resume_update + local_update
        imitation_coef_curr = _imitation_coef_at(update)
        imitation_enabled_curr = imitation_enabled and imitation_coef_curr > 0.0
        update_start = time.perf_counter()
        buffers = [RolloutBuffer(capacity=cfg.buffer_size) for _ in range(num_envs)]
        ep_reward = 0.0
        steps_count = 0
        completed_episode_count = 0
        gu_queue_sum = 0.0
        uav_queue_sum = 0.0
        sat_queue_sum = 0.0
        gu_queue_max = 0.0
        uav_queue_max = 0.0
        sat_queue_max = 0.0
        gu_drop_sum = 0.0
        uav_drop_sum = 0.0
        sat_drop_sum = 0.0
        sat_processed_sum = 0.0
        sat_incoming_sum = 0.0
        connected_sat_count_sum = 0.0
        connected_sat_dist_mean_sum = 0.0
        connected_sat_dist_p95_sum = 0.0
        connected_sat_elevation_deg_mean_sum = 0.0
        connected_sat_elevation_deg_min_sum = 0.0
        energy_mean_sum = 0.0
        dynamics_time_sum = 0.0
        orbit_visible_time_sum = 0.0
        assoc_access_time_sum = 0.0
        backhaul_queue_time_sum = 0.0
        reward_time_sum = 0.0
        obs_time_sum = 0.0
        state_time_sum = 0.0
        step_total_time_sum = 0.0
        r_service_ratio_sum = 0.0
        r_drop_ratio_sum = 0.0
        r_queue_pen_sum = 0.0
        r_queue_topk_sum = 0.0
        r_centroid_sum = 0.0
        centroid_dist_mean_sum = 0.0
        r_bw_align_sum = 0.0
        r_sat_score_sum = 0.0
        r_assoc_ratio_sum = 0.0
        r_queue_delta_sum = 0.0
        r_dist_sum = 0.0
        r_dist_delta_sum = 0.0
        r_energy_sum = 0.0
        r_close_risk_sum = 0.0
        r_collision_penalty_sum = 0.0
        r_battery_penalty_sum = 0.0
        r_fail_penalty_sum = 0.0
        r_term_service_sum = 0.0
        r_term_throughput_access_sum = 0.0
        r_term_throughput_backhaul_sum = 0.0
        r_term_queue_gu_arrival_sum = 0.0
        r_term_drop_sum = 0.0
        r_term_drop_gu_sum = 0.0
        r_term_drop_uav_sum = 0.0
        r_term_drop_sat_sum = 0.0
        r_term_drop_step_sum = 0.0
        r_term_queue_sum = 0.0
        r_term_topk_sum = 0.0
        r_term_assoc_sum = 0.0
        r_term_q_delta_sum = 0.0
        r_term_dist_sum = 0.0
        r_term_dist_delta_sum = 0.0
        r_term_centroid_sum = 0.0
        r_term_bw_align_sum = 0.0
        r_term_sat_score_sum = 0.0
        r_term_energy_sum = 0.0
        r_term_accel_sum = 0.0
        r_term_close_risk_sum = 0.0
        imitation_loss_sum = 0.0
        danger_imitation_loss_sum = 0.0
        residual_reg_loss_sum = 0.0
        reward_raw_sum = 0.0
        reward_aux_sum = 0.0
        x_acc_sum = 0.0
        x_rel_sum = 0.0
        g_pre_sum = 0.0
        d_pre_sum = 0.0
        processed_ratio_eval_sum = 0.0
        drop_ratio_eval_sum = 0.0
        pre_backlog_steps_eval_sum = 0.0
        assoc_centroid_dist_norm_sum = 0.0
        stage1_assoc_centroid_weight_sum = 0.0
        stage1_assoc_centroid_term_sum = 0.0
        sat_overlap_eval_sum = 0.0
        stage3_sat_overlap_weight_sum = 0.0
        stage3_sat_overlap_term_sum = 0.0
        sat_cf_credit_sampled_contexts_sum = 0.0
        sat_cf_credit_selected_q_sum = 0.0
        sat_cf_credit_baseline_q_sum = 0.0
        sat_cf_credit_sum = 0.0
        sat_cf_credit_abs_sum = 0.0
        stage1_aux_agent_sum = np.zeros((num_agents,), dtype=np.float64)
        stage3_overlap_agent_sum = np.zeros((num_agents,), dtype=np.float64)
        d_sys_report_sum = 0.0
        arrival_sum_sum = 0.0
        outflow_sum_sum = 0.0
        service_norm_sum = 0.0
        throughput_access_norm_sum = 0.0
        throughput_backhaul_norm_sum = 0.0
        sat_processed_norm_sum = 0.0
        drop_norm_sum = 0.0
        gu_drop_norm_sum = 0.0
        uav_drop_norm_sum = 0.0
        sat_drop_norm_sum = 0.0
        outflow_arrival_ratio_step_sum = 0.0
        sat_incoming_arrival_ratio_step_sum = 0.0
        sat_processed_arrival_ratio_step_sum = 0.0
        sat_processed_incoming_ratio_step_sum = 0.0
        gu_drop_ratio_step_sum = 0.0
        uav_drop_ratio_step_sum = 0.0
        sat_drop_ratio_step_sum = 0.0
        drop_sum_total = 0.0
        drop_sum_active_total = 0.0
        sat_drop_sum_step_total = 0.0
        drop_event_sum = 0.0
        queue_pen_gu_sum = 0.0
        queue_pen_uav_sum = 0.0
        queue_pen_sat_sum = 0.0
        gu_queue_fill_fraction_sum = 0.0
        uav_queue_fill_fraction_sum = 0.0
        sat_queue_fill_fraction_sum = 0.0
        gu_queue_arrival_steps_sum = 0.0
        uav_queue_arrival_steps_sum = 0.0
        sat_queue_arrival_steps_sum = 0.0
        queue_total_sum = 0.0
        queue_total_active_sum = 0.0
        q_norm_active_sum = 0.0
        prev_q_norm_active_sum = 0.0
        q_norm_delta_sum = 0.0
        queue_delta_gu_sum = 0.0
        queue_delta_uav_sum = 0.0
        queue_delta_sat_sum = 0.0
        q_norm_tail_q0_sum = 0.0
        q_norm_tail_excess_sum = 0.0
        queue_weight_sum = 0.0
        q_delta_weight_sum = 0.0
        crash_weight_sum = 0.0
        centroid_transfer_ratio_sum = 0.0
        danger_imitation_active_rate_sum = 0.0
        intervention_norm_sum = 0.0
        intervention_rate_sum = 0.0
        intervention_norm_top1_sum = 0.0
        assoc_unfair_step_sum = 0.0
        assoc_unfair_max_gu_count_sum = 0.0
        collision_event_sum = 0.0
        avoidance_eta_eff_sum = 0.0
        avoidance_eta_exec_sum = 0.0
        avoidance_collision_rate_ema_sum = 0.0
        avoidance_prev_episode_collision_rate_sum = 0.0
        filter_active_ratio_sum = 0.0
        projected_delta_norm_mean_sum = 0.0
        fallback_count_sum = 0.0
        boundary_filter_count_sum = 0.0
        pairwise_filter_count_sum = 0.0
        pairwise_filter_active_ratio_sum = 0.0
        pairwise_projected_delta_norm_sum = 0.0
        pairwise_fallback_count_sum = 0.0
        pairwise_candidate_infeasible_count_sum = 0.0
        arrival_rate_eff_sum = 0.0
        assoc_unfair_episode_fracs: List[float] = []
        q_norm_active_values: List[float] = []
        queue_total_active_values: List[float] = []
        q_norm_active_nonzero_count = 0
        q_norm_tail_hit_count = 0
        q_norm_active_max = 0.0
        queue_total_active_max = 0.0
        rollout_logprob_part_accel: list[np.ndarray] = []
        rollout_logprob_part_bw: list[np.ndarray] = []
        rollout_logprob_part_sat: list[np.ndarray] = []
        rollout_gu_queue_arrival_steps_by_env: list[list[float]] = [[] for _ in range(num_envs)]
        rollout_gu_drop_ratio_step_by_env: list[list[float]] = [[] for _ in range(num_envs)]
        sat_cf_credit_sample_prob = 0.0
        if sat_counterfactual_credit_enabled:
            denom = max(int(cfg.buffer_size) * max(int(num_envs), 1) * max(int(num_agents), 1), 1)
            sat_cf_credit_sample_prob = min(
                1.0,
                float(sat_counterfactual_credit_samples_per_update) / float(denom),
            )

        rollout_start = time.perf_counter()
        state_fetch_time = 0.0
        policy_forward_time = 0.0
        action_pack_time = 0.0
        env_step_time = 0.0
        for step in range(cfg.buffer_size):
            state_fetch_start = time.perf_counter()
            curr_state_batch_np = np.asarray(state_batch_np, dtype=np.float32)
            if curr_state_batch_np.shape[0] != num_envs:
                raise ValueError(
                    f"Expected {num_envs} states from env, got {curr_state_batch_np.shape[0]}"
                )
            state_fetch_time += time.perf_counter() - state_fetch_start

            action_pack_start = time.perf_counter()
            per_env_obs_lists = obs_lists_env
            curr_obs_step_batch_np = np.asarray(obs_step_batch_np, dtype=np.float32)
            if curr_obs_step_batch_np.shape != (num_envs, num_agents, obs_dim):
                raise ValueError(
                    f"Expected obs_step batch shape {(num_envs, num_agents, obs_dim)}, got {tuple(curr_obs_step_batch_np.shape)}"
                )
            obs_batch = curr_obs_step_batch_np.reshape(num_envs * num_agents, obs_dim)
            if not np.isfinite(obs_batch).all():
                print(f"NaN/Inf detected in obs_batch at update={update}, step={step}")
                raise ValueError("obs_batch contains NaN/Inf")
            obs_tensor = torch.from_numpy(obs_batch).to(device)
            obs_step_tensor = torch.from_numpy(curr_obs_step_batch_np).to(device)
            action_pack_time += time.perf_counter() - action_pack_start

            policy_forward_start = time.perf_counter()
            with torch.inference_mode():
                curr_state_t = torch.from_numpy(curr_state_batch_np).to(device)
                policy_out = actor.act(
                    obs_tensor,
                    required_heads=rollout_policy_heads,
                    compute_logprob=False,
                )
                if critic_multihead_value_enabled:
                    value = _stack_value_head_dict(critic.forward_heads(curr_state_t, obs_step_tensor))
                else:
                    value = critic(curr_state_t, obs_step_tensor)
            for key, value_tensor in (policy_out.dist_out or {}).items():
                if torch.is_tensor(value_tensor) and not torch.isfinite(value_tensor).all():
                    print(f"NaN/Inf detected in actor dist_out['{key}'] at update={update}, step={step}")
                    raise ValueError(f"actor dist_out['{key}'] contains NaN/Inf")
            teacher_accel_all = None
            teacher_bw_all = None
            teacher_sat_all = None
            if teacher_actor is not None and teacher_exec_heads:
                with torch.inference_mode():
                    teacher_out = teacher_actor.act(
                        obs_tensor,
                        deterministic=teacher_deterministic,
                        required_heads=teacher_exec_heads,
                        compute_logprob=False,
                    )
                teacher_accel_all = (
                    teacher_out.accel.cpu().numpy()
                    if teacher_out.accel is not None
                    else None
                )
                teacher_bw_all = teacher_out.bw_action.cpu().numpy() if teacher_out.bw_action is not None else None
                teacher_sat_all = (
                    teacher_out.sat_select_mask.cpu().numpy()
                    if teacher_out.sat_select_mask is not None
                    else None
                )
            policy_forward_time += time.perf_counter() - policy_forward_start

            action_pack_start = time.perf_counter()
            accel_actions = policy_out.accel.cpu().numpy() if policy_out.accel is not None else None
            bw_logits_all = policy_out.bw_action.cpu().numpy() if policy_out.bw_action is not None else None
            sat_logits_all = (
                policy_out.sat_select_mask.cpu().numpy()
                if policy_out.sat_select_mask is not None
                else None
            )
            if exec_accel_source == "policy" and accel_actions is None:
                raise ValueError("exec_accel_source='policy' requires accel head output from actor.act().")
            if cfg.enable_bw_action and exec_bw_source == "policy" and bw_logits_all is None:
                raise ValueError("exec_bw_source='policy' requires bw head output from actor.act().")
            if not cfg.fixed_satellite_strategy and exec_sat_source == "policy" and sat_logits_all is None:
                raise ValueError("exec_sat_source='policy' requires sat head output from actor.act().")
            if exec_accel_source == "teacher" and teacher_accel_all is None:
                raise ValueError("exec_accel_source='teacher' requires accel head output from teacher actor.")
            if cfg.enable_bw_action and exec_bw_source == "teacher" and teacher_bw_all is None:
                raise ValueError("exec_bw_source='teacher' requires bw head output from teacher actor.")
            if not cfg.fixed_satellite_strategy and exec_sat_source == "teacher" and teacher_sat_all is None:
                raise ValueError("exec_sat_source='teacher' requires sat head output from teacher actor.")

            action_dicts = []
            accel_cmd_list: List[np.ndarray] = []
            bw_exec_list: List[np.ndarray | None] = []
            sat_exec_list: List[np.ndarray | None] = []
            for env_idx in range(num_envs):
                sl = slice(env_idx * num_agents, (env_idx + 1) * num_agents)
                obs_list = per_env_obs_lists[env_idx]
                accel_policy_env = accel_actions[sl] if accel_actions is not None else None
                bw_policy_env = bw_logits_all[sl] if bw_logits_all is not None else None
                sat_policy_env = sat_logits_all[sl] if sat_logits_all is not None else None
                accel_teacher_env = teacher_accel_all[sl] if teacher_accel_all is not None else None
                bw_teacher_env = teacher_bw_all[sl] if teacher_bw_all is not None else None
                sat_teacher_env = teacher_sat_all[sl] if teacher_sat_all is not None else None
                heur_accel = None
                heur_bw = None
                heur_sat = None
                if need_heuristic_exec:
                    heur_accel, heur_bw, heur_sat = _resolve_exec_heuristic_triplet(
                        obs_list,
                        cfg,
                        env,
                        exec_accel_source,
                        exec_bw_source,
                        exec_sat_source,
                    )

                accel_cmd = _select_exec_values(
                    exec_accel_source,
                    accel_policy_env,
                    accel_teacher_env,
                    heur_accel,
                    (len(obs_list), 2),
                )
                accel_cmd = _project_normalized_accel_np(np.clip(accel_cmd, -1.0, 1.0))

                bw_valid_mask = np.stack(
                    [o.get("bw_valid_mask", o["users_mask"]) for o in obs_list],
                    axis=0,
                ).astype(np.float32, copy=False)
                sat_valid_mask = np.stack(
                    [o.get("sat_valid_mask", o["sats_mask"]) for o in obs_list],
                    axis=0,
                ).astype(np.float32, copy=False)

                bw_exec = None
                if cfg.enable_bw_action:
                    bw_raw = _compose_bw_exec_values(
                        exec_bw_source,
                        bw_policy_env,
                        bw_teacher_env,
                        heur_bw,
                        (len(obs_list), cfg.num_gu),
                        cfg,
                    )
                    bw_exec = _bw_action_to_full_gu_from_obs(obs_list, np.clip(bw_raw, 0.0, None), cfg)

                sat_exec = None
                if not cfg.fixed_satellite_strategy:
                    sat_raw = _select_exec_values(
                        exec_sat_source,
                        sat_policy_env,
                        sat_teacher_env,
                        heur_sat,
                        (len(obs_list), cfg.sats_obs_max),
                    )
                    sat_exec = np.clip(sat_raw, 0.0, 1.0) * sat_valid_mask

                action_dict = assemble_actions(
                    cfg,
                    env.agents,
                    accel_cmd,
                    bw_alloc=bw_exec,
                    sat_select_mask=sat_exec,
                )
                action_dicts.append(action_dict)
                accel_cmd_list.append(accel_cmd)
                bw_exec_list.append(bw_exec)
                sat_exec_list.append(sat_exec)
            action_pack_time += time.perf_counter() - action_pack_start

            env_step_start = time.perf_counter()
            if num_envs > 1:
                sat_cf_request = None
                if sat_cf_credit_sample_prob > 0.0:
                    sat_cf_request = {"sample_prob": sat_cf_credit_sample_prob}
                next_obs_env, rewards_env, terms_env, truncs_env, _ = env.step(
                    action_dicts,
                    auto_reset=True,
                    sat_cf_request=sat_cf_request,
                )
                step_stats = getattr(env, "last_step_stats", None)
                if not isinstance(step_stats, list) or len(step_stats) != num_envs:
                    raise ValueError("Vectorized env must expose last_step_stats after step().")
            else:
                next_obs, rewards, terms, truncs, _ = env.step(action_dicts[0])
                post_step_obs = next_obs
                step_stats = [_single_env_step_stats(env)]
                step_stats[0]["post_step_obs"] = post_step_obs
                step_stats[0]["post_step_global_state"] = np.asarray(env.get_global_state(), dtype=np.float32)
                done_scalar = list(terms.values())[0] or list(truncs.values())[0]
                if done_scalar:
                    next_obs, _ = env.reset()
                next_obs_env = [next_obs]
                rewards_env = [rewards]
                terms_env = [terms]
                truncs_env = [truncs]
            env_step_time += time.perf_counter() - env_step_start

            state_fetch_start = time.perf_counter()
            next_state_batch_np = _get_state_batch(env)
            if next_state_batch_np.shape[0] != num_envs:
                raise ValueError(
                    f"Expected {num_envs} next states from env, got {next_state_batch_np.shape[0]}"
                )
            state_fetch_time += time.perf_counter() - state_fetch_start

            action_pack_start = time.perf_counter()
            next_obs_env = list(next_obs_env)
            next_obs_lists_env, next_obs_step_batch_np = _flatten_obs_env_batch(next_obs_env, cfg)
            buffer_next_obs_env = []
            for env_idx, next_obs_after_reset in enumerate(next_obs_env):
                stats = step_stats[env_idx] if step_stats[env_idx] is not None else {}
                post_step_obs = stats.get("post_step_obs") if isinstance(stats, dict) else None
                buffer_next_obs_env.append(post_step_obs if isinstance(post_step_obs, dict) else next_obs_after_reset)
            _, buffer_next_obs_step_batch_np = _flatten_obs_env_batch(buffer_next_obs_env, cfg)
            action_pack_time += time.perf_counter() - action_pack_start

            value_np = value.detach().cpu().numpy().astype(np.float32, copy=False)
            if critic_multihead_value_enabled:
                value_np = value_np.reshape(num_envs, len(VALUE_HEAD_NAMES))
            else:
                value_np = value_np.reshape(num_envs)

            action_vec_exec_env = []
            sat_indices_exec_env = []
            for env_idx in range(num_envs):
                stats = step_stats[env_idx] if step_stats[env_idx] is not None else {}
                fallback_accel = accel_cmd_list[env_idx] * cfg.a_max
                accel_exec = np.asarray(stats.get("last_exec_accel", fallback_accel), dtype=np.float32)
                accel_exec_norm = _project_normalized_accel_np(accel_exec / max(cfg.a_max, 1e-6))
                exec_parts = [accel_exec_norm]
                if cfg.enable_bw_action and bw_exec_list[env_idx] is not None:
                    exec_bw = np.asarray(
                        stats.get("last_exec_bw_alloc", bw_exec_list[env_idx]),
                        dtype=np.float32,
                    )
                    exec_parts.append(exec_bw)
                if not cfg.fixed_satellite_strategy and sat_exec_list[env_idx] is not None:
                    exec_sat = np.asarray(
                        stats.get("last_exec_sat_select_mask", sat_exec_list[env_idx]),
                        dtype=np.float32,
                    )
                    exec_parts.append(exec_sat)
                action_vec_exec_env.append(np.concatenate(exec_parts, axis=1).astype(np.float32, copy=False))
                sat_indices_exec_env.append(
                    np.asarray(
                        stats.get(
                            "last_exec_sat_indices",
                            np.full(
                                (num_agents, max(int(getattr(cfg, "sat_action_select_k", getattr(cfg, "sat_num_select", cfg.N_RF)) or cfg.N_RF), 0)),
                                -1,
                                dtype=np.int64,
                            ),
                        ),
                        dtype=np.int64,
                    )
                )

            action_vec_exec = np.concatenate(action_vec_exec_env, axis=0).astype(np.float32, copy=False)
            with torch.inference_mode():
                action_vec_exec_t = torch.from_numpy(action_vec_exec).to(device)
                sat_indices_exec = np.concatenate(sat_indices_exec_env, axis=0)
                sat_indices_exec_t = torch.from_numpy(sat_indices_exec).to(device)
                logprob_parts, _ = actor.evaluate_actions_parts(
                    obs_tensor,
                    action_vec_exec_t,
                    sat_indices=sat_indices_exec_t,
                    out=policy_out.dist_out,
                    heads=train_head_names,
                    need_entropy=False,
                )
                if "accel" in logprob_parts:
                    rollout_logprob_part_accel.append(logprob_parts["accel"].detach().cpu().numpy().reshape(-1))
                if "bw" in logprob_parts:
                    rollout_logprob_part_bw.append(logprob_parts["bw"].detach().cpu().numpy().reshape(-1))
                if "sat" in logprob_parts:
                    rollout_logprob_part_sat.append(logprob_parts["sat"].detach().cpu().numpy().reshape(-1))
                logprobs_all = _sum_selected_parts(logprob_parts, train_heads).detach().cpu().numpy()

            for env_idx in range(num_envs):
                sl = slice(env_idx * num_agents, (env_idx + 1) * num_agents)
                rewards = rewards_env[env_idx]
                terms = terms_env[env_idx]
                truncs = truncs_env[env_idx]
                stats = step_stats[env_idx] if step_stats[env_idx] is not None else {}
                reward_aux, reward_aux_info = _compute_train_reward_adjustment(
                    stats,
                    cfg,
                    update,
                    aux_schedule_total_updates,
                )

                reward_scalar = float(list(rewards.values())[0]) + reward_aux
                terminated_scalar = bool(list(terms.values())[0])
                truncated_scalar = bool(list(truncs.values())[0])
                done_scalar = terminated_scalar or truncated_scalar
                post_step_state = np.asarray(
                    stats.get("post_step_global_state", next_state_batch_np[env_idx]),
                    dtype=np.float32,
                )
                sat_cf_credit_env = np.zeros((num_agents,), dtype=np.float32)
                sat_cf_credit_mask_env = np.zeros((num_agents,), dtype=np.float32)
                if sat_counterfactual_credit_enabled:
                    sat_cf_rows = list(stats.get("sat_cf_credit_rows", []) or [])
                    if sat_cf_rows:
                        sat_cf_credit_env, sat_cf_credit_mask_env, sat_cf_metrics = _compute_sat_counterfactual_credit(
                            sat_cf_rows,
                            reward_scalar,
                            post_step_state,
                            buffer_next_obs_step_batch_np[env_idx],
                            terminated_scalar,
                            critic,
                            device,
                            cfg,
                            update,
                            aux_schedule_total_updates,
                        )
                        sat_cf_credit_sampled_contexts_sum += float(sat_cf_metrics["sampled_contexts"])
                        sat_cf_credit_selected_q_sum += float(sat_cf_metrics["selected_q_mean"]) * float(
                            sat_cf_metrics["sampled_contexts"]
                        )
                        sat_cf_credit_baseline_q_sum += float(sat_cf_metrics["baseline_q_mean"]) * float(
                            sat_cf_metrics["sampled_contexts"]
                        )
                        sat_cf_credit_sum += float(sat_cf_metrics["credit_mean"]) * float(
                            sat_cf_metrics["sampled_contexts"]
                        )
                        sat_cf_credit_abs_sum += float(sat_cf_metrics["credit_abs_mean"]) * float(
                            sat_cf_metrics["sampled_contexts"]
                        )
                ep_reward += reward_scalar
                episode_return_env[env_idx] += float(reward_scalar)
                reward_aux_sum += reward_aux
                assoc_centroid_dist_norm_sum += reward_aux_info["stage1_assoc_centroid_dist_norm_mean"]
                stage1_assoc_centroid_weight_sum += reward_aux_info["stage1_assoc_centroid_weight"]
                stage1_assoc_centroid_term_sum += reward_aux_info["stage1_assoc_centroid_term"]
                sat_overlap_eval_sum += reward_aux_info["stage3_sat_overlap_eval"]
                stage3_sat_overlap_weight_sum += reward_aux_info["stage3_sat_overlap_weight"]
                stage3_sat_overlap_term_sum += reward_aux_info["stage3_sat_overlap_term"]
                stage1_aux_agent_vals = np.asarray(
                    reward_aux_info["stage1_assoc_centroid_dist_norm_uav"],
                    dtype=np.float32,
                ).reshape(-1)
                stage3_overlap_agent_vals = np.asarray(
                    reward_aux_info["stage3_sat_overlap_uav"],
                    dtype=np.float32,
                ).reshape(-1)
                if stage1_aux_agent_vals.size < num_agents:
                    stage1_aux_agent_vals = np.pad(
                        stage1_aux_agent_vals,
                        (0, num_agents - stage1_aux_agent_vals.size),
                    )
                if stage3_overlap_agent_vals.size < num_agents:
                    stage3_overlap_agent_vals = np.pad(
                        stage3_overlap_agent_vals,
                        (0, num_agents - stage3_overlap_agent_vals.size),
                    )
                stage1_aux_agent_sum += stage1_aux_agent_vals[:num_agents]
                stage3_overlap_agent_sum += stage3_overlap_agent_vals[:num_agents]
                steps_count += 1
                total_env_steps += 1
                episode_step_counts_env[env_idx] += 1

                gu_queue_sum += float(stats.get("gu_queue_mean", 0.0))
                uav_queue_sum += float(stats.get("uav_queue_mean", 0.0))
                sat_queue_sum += float(stats.get("sat_queue_mean", 0.0))
                gu_queue_max = max(gu_queue_max, float(stats.get("gu_queue_max", 0.0)))
                uav_queue_max = max(uav_queue_max, float(stats.get("uav_queue_max", 0.0)))
                sat_queue_max = max(sat_queue_max, float(stats.get("sat_queue_max", 0.0)))
                gu_drop_sum += float(stats.get("gu_drop_sum", 0.0))
                uav_drop_sum += float(stats.get("uav_drop_sum", 0.0))
                sat_drop_sum += float(stats.get("sat_drop_sum", 0.0))
                sat_processed_sum += float(stats.get("sat_processed_sum", 0.0))
                sat_incoming_sum += float(stats.get("sat_incoming_sum", 0.0))
                connected_sat_count_sum += float(stats.get("connected_sat_count", 0.0))
                connected_sat_dist_mean_sum += float(stats.get("connected_sat_dist_mean", 0.0))
                connected_sat_dist_p95_sum += float(stats.get("connected_sat_dist_p95", 0.0))
                connected_sat_elevation_deg_mean_sum += float(
                    stats.get("connected_sat_elevation_deg_mean", 0.0)
                )
                connected_sat_elevation_deg_min_sum += float(
                    stats.get("connected_sat_elevation_deg_min", 0.0)
                )
                if cfg.energy_enabled:
                    energy_mean_sum += float(stats.get("energy_mean", 0.0))
                dynamics_time_sum += float(stats.get("dynamics_time_sec", 0.0))
                orbit_visible_time_sum += float(stats.get("orbit_visible_time_sec", 0.0))
                assoc_access_time_sum += float(stats.get("assoc_access_time_sec", 0.0))
                backhaul_queue_time_sum += float(stats.get("backhaul_queue_time_sec", 0.0))
                reward_time_sum += float(stats.get("reward_time_sec", 0.0))
                obs_time_sum += float(stats.get("obs_time_sec", 0.0))
                state_time_sum += float(stats.get("state_time_sec", 0.0))
                step_total_time_sum += float(stats.get("step_total_time_sec", 0.0))

                parts = stats.get("reward_parts", None)
                gu_queue_arrival_steps_step = 0.0
                gu_drop_ratio_step = 0.0
                if parts:
                    x_acc_sum += float(parts.get("x_acc", 0.0))
                    x_rel_sum += float(parts.get("x_rel", 0.0))
                    g_pre_sum += float(parts.get("g_pre", 0.0))
                    d_pre_sum += float(parts.get("d_pre", 0.0))
                    processed_ratio_eval_sum += float(parts.get("processed_ratio_eval", 0.0))
                    drop_ratio_eval_sum += float(parts.get("drop_ratio_eval", 0.0))
                    pre_backlog_steps_eval_sum += float(parts.get("pre_backlog_steps_eval", 0.0))
                    d_sys_report_sum += float(parts.get("D_sys_report", 0.0))
                    r_service_ratio_sum += float(parts.get("service_ratio", 0.0))
                    r_drop_ratio_sum += float(parts.get("drop_ratio", 0.0))
                    arrival_sum_sum += float(parts.get("arrival_sum", 0.0))
                    outflow_sum_sum += float(parts.get("outflow_sum", 0.0))
                    service_norm_sum += float(parts.get("service_norm", 0.0))
                    throughput_access_norm_sum += float(parts.get("throughput_access_norm", 0.0))
                    throughput_backhaul_norm_sum += float(parts.get("throughput_backhaul_norm", 0.0))
                    sat_processed_norm_sum += float(parts.get("sat_processed_norm", 0.0))
                    drop_norm_sum += float(parts.get("drop_norm", 0.0))
                    gu_drop_norm_sum += float(parts.get("gu_drop_norm", 0.0))
                    uav_drop_norm_sum += float(parts.get("uav_drop_norm", 0.0))
                    sat_drop_norm_sum += float(parts.get("sat_drop_norm", 0.0))
                    outflow_arrival_ratio_step_sum += float(parts.get("outflow_arrival_ratio_step", 0.0))
                    sat_incoming_arrival_ratio_step_sum += float(parts.get("sat_incoming_arrival_ratio_step", 0.0))
                    sat_processed_arrival_ratio_step_sum += float(parts.get("sat_processed_arrival_ratio_step", 0.0))
                    sat_processed_incoming_ratio_step_sum += float(
                        parts.get("sat_processed_incoming_ratio_step", 0.0)
                    )
                    gu_drop_ratio_step = float(parts.get("gu_drop_ratio_step", 0.0))
                    gu_drop_ratio_step_sum += gu_drop_ratio_step
                    uav_drop_ratio_step_sum += float(parts.get("uav_drop_ratio_step", 0.0))
                    sat_drop_ratio_step_sum += float(parts.get("sat_drop_ratio_step", 0.0))
                    drop_sum_total += float(parts.get("drop_sum", 0.0))
                    drop_sum_active_total += float(parts.get("drop_sum_active", 0.0))
                    sat_drop_sum_step_total += float(parts.get("sat_drop_sum", 0.0))
                    drop_event_sum += float(parts.get("drop_event", 0.0))
                    queue_pen_gu_sum += float(parts.get("queue_pen_gu", 0.0))
                    queue_pen_uav_sum += float(parts.get("queue_pen_uav", 0.0))
                    queue_pen_sat_sum += float(parts.get("queue_pen_sat", 0.0))
                    gu_queue_fill_fraction_sum += float(parts.get("gu_queue_fill_fraction", 0.0))
                    uav_queue_fill_fraction_sum += float(parts.get("uav_queue_fill_fraction", 0.0))
                    sat_queue_fill_fraction_sum += float(parts.get("sat_queue_fill_fraction", 0.0))
                    gu_queue_arrival_steps_step = float(parts.get("gu_queue_arrival_steps", 0.0))
                    gu_queue_arrival_steps_sum += gu_queue_arrival_steps_step
                    uav_queue_arrival_steps_sum += float(parts.get("uav_queue_arrival_steps", 0.0))
                    sat_queue_arrival_steps_sum += float(parts.get("sat_queue_arrival_steps", 0.0))
                    queue_total_sum += float(parts.get("queue_total", 0.0))
                    queue_total_active_step = float(parts.get("queue_total_active", 0.0))
                    q_norm_active_step = float(parts.get("q_norm_active", 0.0))
                    q_norm_tail_q0_step = max(float(parts.get("q_norm_tail_q0", 0.0)), 0.0)
                    queue_total_active_sum += queue_total_active_step
                    q_norm_active_sum += q_norm_active_step
                    prev_q_norm_active_sum += float(parts.get("prev_q_norm_active", 0.0))
                    q_norm_delta_sum += float(parts.get("q_norm_delta", 0.0))
                    queue_delta_gu_sum += float(parts.get("queue_delta_gu", 0.0))
                    queue_delta_uav_sum += float(parts.get("queue_delta_uav", 0.0))
                    queue_delta_sat_sum += float(parts.get("queue_delta_sat", 0.0))
                    q_norm_tail_q0_sum += q_norm_tail_q0_step
                    q_norm_tail_excess_sum += float(parts.get("q_norm_tail_excess", 0.0))
                    queue_weight_sum += float(parts.get("queue_weight", 0.0))
                    q_delta_weight_sum += float(parts.get("q_delta_weight", 0.0))
                    crash_weight_sum += float(parts.get("crash_weight", 0.0))
                    centroid_transfer_ratio_sum += float(parts.get("centroid_transfer_ratio", 0.0))
                    collision_event_sum += float(parts.get("collision_event", 0.0))
                    avoidance_eta_eff_sum += float(parts.get("avoidance_eta_eff", 0.0))
                    avoidance_eta_exec_sum += float(parts.get("avoidance_eta_exec", 0.0))
                    avoidance_collision_rate_ema_sum += float(parts.get("avoidance_collision_rate_ema", 0.0))
                    avoidance_prev_episode_collision_rate_sum += float(
                        parts.get("avoidance_prev_episode_collision_rate", 0.0)
                    )
                    filter_active_ratio_sum += float(parts.get("filter_active_ratio", 0.0))
                    projected_delta_norm_mean_sum += float(parts.get("projected_delta_norm_mean", 0.0))
                    fallback_count_sum += float(parts.get("fallback_count", 0.0))
                    boundary_filter_count_sum += float(parts.get("boundary_filter_count", 0.0))
                    pairwise_filter_count_sum += float(parts.get("pairwise_filter_count", 0.0))
                    pairwise_filter_active_ratio_sum += float(parts.get("pairwise_filter_active_ratio", 0.0))
                    pairwise_projected_delta_norm_sum += float(parts.get("pairwise_projected_delta_norm", 0.0))
                    pairwise_fallback_count_sum += float(parts.get("pairwise_fallback_count", 0.0))
                    pairwise_candidate_infeasible_count_sum += float(
                        parts.get("pairwise_candidate_infeasible_count", 0.0)
                    )
                    arrival_rate_eff_sum += float(parts.get("arrival_rate_eff", 0.0))
                    q_norm_active_values.append(q_norm_active_step)
                    queue_total_active_values.append(queue_total_active_step)
                    if q_norm_active_step > 0.0:
                        q_norm_active_nonzero_count += 1
                    if q_norm_active_step > q_norm_tail_q0_step:
                        q_norm_tail_hit_count += 1
                    q_norm_active_max = max(q_norm_active_max, q_norm_active_step)
                    queue_total_active_max = max(queue_total_active_max, queue_total_active_step)
                    r_queue_pen_sum += float(parts.get("queue_pen", 0.0))
                    r_queue_topk_sum += float(parts.get("queue_topk", 0.0))
                    r_centroid_sum += float(parts.get("centroid_reward", 0.0))
                    centroid_dist_mean_sum += float(parts.get("centroid_dist_mean", 0.0))
                    r_bw_align_sum += float(parts.get("bw_align", 0.0))
                    r_sat_score_sum += float(parts.get("sat_score", 0.0))
                    r_assoc_ratio_sum += float(parts.get("assoc_ratio", 0.0))
                    r_queue_delta_sum += float(parts.get("queue_delta", 0.0))
                    r_dist_sum += float(parts.get("dist_reward", 0.0))
                    r_dist_delta_sum += float(parts.get("dist_delta", 0.0))
                    r_energy_sum += float(parts.get("energy_reward", 0.0))
                    r_close_risk_sum += float(parts.get("close_risk", 0.0))
                    r_collision_penalty_sum += float(parts.get("collision_penalty", 0.0))
                    r_battery_penalty_sum += float(parts.get("battery_penalty", 0.0))
                    r_fail_penalty_sum += float(parts.get("fail_penalty", 0.0))
                    r_term_service_sum += float(parts.get("term_service", 0.0))
                    term_throughput_access = float(parts.get("term_throughput_access", 0.0))
                    term_throughput_backhaul = float(parts.get("term_throughput_backhaul", 0.0))
                    term_queue_gu_arrival = float(parts.get("term_queue_gu_arrival", 0.0))
                    r_term_throughput_access_sum += term_throughput_access
                    r_term_throughput_backhaul_sum += term_throughput_backhaul
                    r_term_queue_gu_arrival_sum += term_queue_gu_arrival
                    episode_term_throughput_access_env[env_idx] += term_throughput_access
                    episode_term_throughput_backhaul_env[env_idx] += term_throughput_backhaul
                    episode_term_queue_gu_arrival_env[env_idx] += term_queue_gu_arrival
                    r_term_drop_sum += float(parts.get("term_drop", 0.0))
                    r_term_drop_gu_sum += float(parts.get("term_drop_gu", 0.0))
                    r_term_drop_uav_sum += float(parts.get("term_drop_uav", 0.0))
                    r_term_drop_sat_sum += float(parts.get("term_drop_sat", 0.0))
                    r_term_drop_step_sum += float(parts.get("term_drop_step", 0.0))
                    r_term_queue_sum += float(parts.get("term_queue", 0.0))
                    r_term_topk_sum += float(parts.get("term_topk", 0.0))
                    r_term_assoc_sum += float(parts.get("term_assoc", 0.0))
                    r_term_q_delta_sum += float(parts.get("term_q_delta", 0.0))
                    r_term_dist_sum += float(parts.get("term_dist", 0.0))
                    r_term_dist_delta_sum += float(parts.get("term_dist_delta", 0.0))
                    r_term_centroid_sum += float(parts.get("term_centroid", 0.0))
                    r_term_bw_align_sum += float(parts.get("term_bw_align", 0.0))
                    r_term_sat_score_sum += float(parts.get("term_sat_score", 0.0))
                    r_term_energy_sum += float(parts.get("term_energy", 0.0))
                    r_term_accel_sum += float(parts.get("term_accel", 0.0))
                    r_term_close_risk_sum += float(parts.get("term_close_risk", 0.0))
                    reward_raw_sum += float(parts.get("reward_raw", 0.0))
                    danger_imitation_active_rate_sum += float(parts.get("danger_imitation_active_rate", 0.0))
                    intervention_norm_sum += float(parts.get("intervention_norm", 0.0))
                    intervention_rate_sum += float(parts.get("intervention_rate", 0.0))
                    intervention_norm_top1_sum += float(parts.get("intervention_norm_top1", 0.0))
                    episode_collision_env[env_idx] = max(
                        episode_collision_env[env_idx],
                        1.0 if float(parts.get("collision_event", 0.0)) > 0.5 else 0.0,
                    )
                    assoc_unfair_step_value = float(parts.get("assoc_unfair_step", 0.0))
                    assoc_unfair_step_sum += assoc_unfair_step_value
                    assoc_unfair_max_gu_count_sum += float(parts.get("assoc_unfair_max_gu_count", 0.0))
                    if assoc_unfair_step_value > 0.5:
                        episode_assoc_unfair_step_counts_env[env_idx] += 1

                rollout_gu_queue_arrival_steps_by_env[env_idx].append(gu_queue_arrival_steps_step)
                rollout_gu_drop_ratio_step_by_env[env_idx].append(gu_drop_ratio_step)

                if done_scalar:
                    episode_steps_curr = int(episode_step_counts_env[env_idx])
                    unfair_steps_curr = int(episode_assoc_unfair_step_counts_env[env_idx])
                    unfair_frac = (
                        float(unfair_steps_curr) / float(max(episode_steps_curr, 1))
                        if episode_steps_curr > 0
                        else 0.0
                    )
                    recent_episode_returns.append(float(episode_return_env[env_idx]))
                    recent_episode_lengths.append(float(episode_steps_curr))
                    recent_episode_term_throughput_access.append(
                        float(episode_term_throughput_access_env[env_idx])
                    )
                    recent_episode_term_throughput_backhaul.append(
                        float(episode_term_throughput_backhaul_env[env_idx])
                    )
                    recent_episode_term_queue_gu_arrival.append(
                        float(episode_term_queue_gu_arrival_env[env_idx])
                    )
                    recent_episode_collisions.append(float(episode_collision_env[env_idx]))
                    completed_episode_count += 1
                    assoc_unfair_episode_fracs.append(unfair_frac)
                    episode_return_env[env_idx] = 0.0
                    episode_term_throughput_access_env[env_idx] = 0.0
                    episode_term_throughput_backhaul_env[env_idx] = 0.0
                    episode_term_queue_gu_arrival_env[env_idx] = 0.0
                    episode_collision_env[env_idx] = 0.0
                    episode_step_counts_env[env_idx] = 0
                    episode_assoc_unfair_step_counts_env[env_idx] = 0

                danger_target_env, danger_mask_env = _build_danger_imitation_step_data(stats, cfg, num_agents)
                buffers[env_idx].add(
                    curr_obs_step_batch_np[env_idx],
                    action_vec_exec_env[env_idx],
                    logprobs_all[sl],
                    reward_scalar,
                    value_np[env_idx],
                    terminated_scalar,
                    truncated_scalar,
                    curr_state_batch_np[env_idx],
                    post_step_state,
                    buffer_next_obs_step_batch_np[env_idx],
                    _build_imitation_target(per_env_obs_lists[env_idx]),
                    danger_target_env,
                    danger_mask_env,
                    sat_indices_exec_env[env_idx],
                    sat_cf_credit_env,
                    sat_cf_credit_mask_env,
                )

            obs_lists_env = next_obs_lists_env
            obs_step_batch_np = next_obs_step_batch_np
            state_batch_np = next_state_batch_np
        rollout_time = time.perf_counter() - rollout_start

        # Prepare batches
        buffer_data = [buf.as_arrays() for buf in buffers]
        all_rewards = np.concatenate([data[4] for data in buffer_data], axis=0)
        if getattr(cfg, "reward_norm_enabled", False) and reward_rms is not None:
            reward_rms.update(all_rewards)

        obs_arr_list = []
        act_arr_list = []
        logp_arr_list = []
        state_arr_list = []
        imitation_arr_list = []
        danger_imitation_target_list = []
        danger_imitation_mask_list = []
        sat_indices_list = []
        sat_cf_credit_list = []
        sat_cf_credit_mask_list = []
        adv_list = []
        adv_head_lists = {head_name: [] for head_name in VALUE_HEAD_NAMES}
        rets_list = []
        ret_head_lists = {head_name: [] for head_name in VALUE_HEAD_NAMES}
        bw_bad_queue_lists = []
        bw_bad_drop_lists = []
        clip_val = float(getattr(cfg, "reward_norm_clip", 0.0) or 0.0)
        reward_clip_hits = 0
        reward_clip_total = 0
        for env_idx, (
            obs_arr_e,
            next_obs_arr_e,
            act_arr_e,
            logp_arr_e,
            rewards_e,
            values_e,
            terminated_e,
            truncated_e,
            state_arr_e,
            next_state_arr_e,
            imitation_arr_e,
            danger_imitation_target_e,
            danger_imitation_mask_e,
            sat_indices_e,
            sat_cf_credit_e,
            sat_cf_credit_mask_e,
        ) in enumerate(buffer_data):
            rewards_proc = rewards_e
            if getattr(cfg, "reward_norm_enabled", False) and reward_rms is not None:
                rewards_proc = (rewards_proc - reward_rms.mean) / (np.sqrt(reward_rms.var) + 1e-8)
                if clip_val > 0:
                    reward_clip_hits += int(np.count_nonzero(np.abs(rewards_proc) > clip_val))
                    reward_clip_total += int(rewards_proc.size)
                    rewards_proc = np.clip(rewards_proc, -clip_val, clip_val)
            values_e = np.asarray(values_e, dtype=np.float32)
            with torch.inference_mode():
                next_state_t_e = torch.from_numpy(next_state_arr_e).to(device)
                next_obs_t_e = torch.from_numpy(next_obs_arr_e).to(device)
                if critic_multihead_value_enabled:
                    bootstrap_values_e = _stack_value_head_dict(
                        critic.forward_heads(next_state_t_e, next_obs_t_e)
                    ).detach().cpu().numpy()
                else:
                    bootstrap_values_e = critic(next_state_t_e, next_obs_t_e).detach().cpu().numpy().reshape(-1)
            bootstrap_values_e = np.asarray(bootstrap_values_e, dtype=np.float32)
            if critic_multihead_value_enabled:
                bootstrap_values_e = np.where(terminated_e[:, None] > 0.5, 0.0, bootstrap_values_e)
            else:
                bootstrap_values_e = np.where(terminated_e > 0.5, 0.0, bootstrap_values_e)
            episode_boundaries_e = np.logical_or(terminated_e > 0.5, truncated_e > 0.5)
            if critic_multihead_value_enabled:
                adv_heads_e, ret_heads_e = _compute_per_head_gae_targets(
                    rewards_proc,
                    values_e,
                    bootstrap_values_e,
                    episode_boundaries_e,
                    cfg.gamma,
                    cfg.gae_lambda,
                )
                for head_name in VALUE_HEAD_NAMES:
                    ret_head_lists[head_name].append(ret_heads_e[head_name])
                    if ppo_per_head_advantage_enabled:
                        adv_head_lists[head_name].append(adv_heads_e[head_name])
                if ppo_per_head_advantage_enabled:
                    adv_e = np.mean(np.stack([adv_heads_e[head_name] for head_name in VALUE_HEAD_NAMES], axis=1), axis=1)
                else:
                    adv_e = np.mean(np.stack([adv_heads_e[head_name] for head_name in VALUE_HEAD_NAMES], axis=1), axis=1)
                rets_e = np.mean(np.stack([ret_heads_e[head_name] for head_name in VALUE_HEAD_NAMES], axis=1), axis=1)
            else:
                adv_e, rets_e = compute_gae(
                    rewards_proc,
                    values_e,
                    bootstrap_values_e,
                    episode_boundaries_e,
                    cfg.gamma,
                    cfg.gae_lambda,
                )
            episode_boundaries_e = np.asarray(episode_boundaries_e, dtype=bool)
            bw_bad_queue_e, bw_bad_drop_e = _compute_future_badness_signals(
                np.asarray(rollout_gu_queue_arrival_steps_by_env[env_idx], dtype=np.float32),
                np.asarray(rollout_gu_drop_ratio_step_by_env[env_idx], dtype=np.float32),
                episode_boundaries_e,
                bw_badness_horizon,
            )
            obs_arr_list.append(obs_arr_e)
            act_arr_list.append(act_arr_e)
            logp_arr_list.append(logp_arr_e)
            state_arr_list.append(state_arr_e)
            imitation_arr_list.append(imitation_arr_e)
            danger_imitation_target_list.append(danger_imitation_target_e)
            danger_imitation_mask_list.append(danger_imitation_mask_e)
            sat_indices_list.append(sat_indices_e)
            sat_cf_credit_list.append(sat_cf_credit_e)
            sat_cf_credit_mask_list.append(sat_cf_credit_mask_e)
            adv_list.append(adv_e)
            rets_list.append(rets_e)
            bw_bad_queue_lists.append(bw_bad_queue_e)
            bw_bad_drop_lists.append(bw_bad_drop_e)

        obs_arr = np.concatenate(obs_arr_list, axis=0)
        act_arr = np.concatenate(act_arr_list, axis=0)
        logp_arr = np.concatenate(logp_arr_list, axis=0)
        state_arr = np.concatenate(state_arr_list, axis=0)
        imitation_arr = np.concatenate(imitation_arr_list, axis=0)
        danger_imitation_target_arr = np.concatenate(danger_imitation_target_list, axis=0)
        danger_imitation_mask_arr = np.concatenate(danger_imitation_mask_list, axis=0)
        sat_indices_arr = np.concatenate(sat_indices_list, axis=0)
        sat_cf_credit_arr = np.concatenate(sat_cf_credit_list, axis=0)
        sat_cf_credit_mask_arr = np.concatenate(sat_cf_credit_mask_list, axis=0)
        rets = np.concatenate(rets_list, axis=0)
        adv_clip = float(getattr(cfg, "adv_clip", 5.0) or 0.0)
        adv_raw_ref = np.concatenate(adv_list, axis=0)
        adv = adv_raw_ref
        adv_by_head: Dict[str, np.ndarray] = {}
        ret_by_head: Dict[str, np.ndarray] = {}
        bw_false_positive_mask_rate = 0.0
        bw_false_positive_bad_score_mean = 0.0
        bw_false_positive_bad_tau = float(getattr(cfg, "bw_false_positive_bad_tau", 1.0) or 1.0)
        if ppo_per_head_advantage_enabled:
            adv_head_stack = []
            for head_name in VALUE_HEAD_NAMES:
                adv_head_final, _ = _normalize_advantages(np.concatenate(adv_head_lists[head_name], axis=0), adv_clip)
                adv_by_head[head_name] = adv_head_final
                ret_by_head[head_name] = np.concatenate(ret_head_lists[head_name], axis=0).astype(np.float32, copy=False)
                adv_head_stack.append(adv_head_final)
            adv = np.mean(np.stack(adv_head_stack, axis=1), axis=1).astype(np.float32, copy=False)
            _, adv_stats = _normalize_advantages(adv_raw_ref, adv_clip)
        else:
            adv, adv_stats = _normalize_advantages(adv, adv_clip)
        bw_bad_score = None
        if (
            (bw_continuous_credit_enabled or bool(getattr(cfg, "bw_false_positive_mask_enabled", False)))
            and bw_bad_queue_lists
            and bw_bad_drop_lists
        ):
            bad_queue = np.concatenate(bw_bad_queue_lists, axis=0).astype(np.float32, copy=False)
            bad_drop = np.concatenate(bw_bad_drop_lists, axis=0).astype(np.float32, copy=False)
            bw_bad_score = (_zscore_array(bad_queue) + _zscore_array(bad_drop)).astype(np.float32, copy=False)
        if bw_continuous_credit_enabled and bw_bad_score is not None:
            bw_train_adv = adv.astype(np.float32, copy=True) - bw_continuous_credit_alpha * bw_bad_score
            if bw_continuous_credit_renorm_enabled:
                bw_train_adv, _ = _normalize_advantages(bw_train_adv, adv_clip)
            adv_by_head["bw"] = bw_train_adv.astype(np.float32, copy=False)
        if bool(getattr(cfg, "bw_false_positive_mask_enabled", False)) and bw_bad_score is not None:
            bw_train_adv = adv_by_head.get("bw", adv).astype(np.float32, copy=True)
            bw_mask = np.logical_and(bw_train_adv > 0.0, bw_bad_score > bw_false_positive_bad_tau)
            bw_train_adv[bw_mask] = 0.0
            adv_by_head["bw"] = bw_train_adv
            bw_false_positive_mask_rate = float(np.mean(bw_mask.astype(np.float32))) if bw_mask.size else 0.0
            bw_false_positive_bad_score_mean = (
                float(np.mean(bw_bad_score[bw_mask])) if np.any(bw_mask) else 0.0
            )
        adv_raw_mean = adv_stats["raw_mean"]
        adv_raw_std = adv_stats["raw_std"]
        adv_preclip_mean = adv_stats["preclip_mean"]
        adv_preclip_std = adv_stats["preclip_std"]
        adv_clip_frac = adv_stats["clip_frac"]
        adv_postclip_mean = adv_stats["postclip_mean"]
        adv_postclip_std = adv_stats["postclip_std"]
        adv_norm_mean = adv_postclip_mean
        adv_norm_std = adv_postclip_std

        T, N, _ = obs_arr.shape
        obs_flat = obs_arr.reshape(T * N, -1)
        act_flat = act_arr.reshape(T * N, -1)
        logp_flat = logp_arr.reshape(T * N)
        imitation_flat = imitation_arr.reshape(T * N, -1)
        danger_imitation_target_flat = danger_imitation_target_arr.reshape(T * N, -1)
        danger_imitation_mask_flat = danger_imitation_mask_arr.reshape(T * N, -1)
        sat_indices_flat = sat_indices_arr.reshape(T * N, -1)
        sat_cf_credit_flat = sat_cf_credit_arr.reshape(T * N).astype(np.float32, copy=False)
        sat_cf_credit_mask_flat = sat_cf_credit_mask_arr.reshape(T * N) > 0.5
        adv_flat = np.repeat(adv, N)
        adv_flat_by_head = {head_name: np.repeat(adv_head, N) for head_name, adv_head in adv_by_head.items()}
        if sat_counterfactual_credit_enabled and np.any(sat_cf_credit_mask_flat):
            sat_cf_credit_values = sat_cf_credit_flat[sat_cf_credit_mask_flat]
            sat_cf_credit_norm, _ = _normalize_advantages(sat_cf_credit_values, adv_clip)
            sat_train_adv_flat = adv_flat.astype(np.float32, copy=True)
            sat_train_adv_flat[sat_cf_credit_mask_flat] = (
                (1.0 - sat_counterfactual_credit_weight) * sat_train_adv_flat[sat_cf_credit_mask_flat]
                + sat_counterfactual_credit_weight * sat_cf_credit_norm
            ).astype(np.float32, copy=False)
            adv_flat_by_head["sat"] = sat_train_adv_flat

        # Convert to torch
        obs_step_t = torch.from_numpy(obs_arr).to(device)
        obs_flat_t = torch.from_numpy(obs_flat).to(device)
        act_flat_t = torch.from_numpy(act_flat).to(device)
        logp_flat_t = torch.from_numpy(logp_flat).to(device)
        adv_flat_t = torch.from_numpy(adv_flat).to(device)
        adv_flat_by_head_t = {
            head_name: torch.from_numpy(adv_flat_by_head[head_name]).to(device)
            for head_name in VALUE_HEAD_NAMES
            if head_name in adv_flat_by_head
        }
        state_t = torch.from_numpy(state_arr).to(device)
        ret_t = torch.from_numpy(rets).to(device)
        ret_t_by_head = {
            head_name: torch.from_numpy(ret_head_arr).to(device)
            for head_name, ret_head_arr in ret_by_head.items()
        }
        imitation_flat_t = torch.from_numpy(imitation_flat).to(device)
        danger_imitation_target_flat_t = torch.from_numpy(danger_imitation_target_flat).to(device)
        danger_imitation_mask_flat_t = torch.from_numpy(danger_imitation_mask_flat).to(device)
        sat_indices_flat_t = torch.from_numpy(sat_indices_flat).to(device)
        cached_ctx_flat_t = None
        if cache_actor_ctx:
            with torch.no_grad():
                cached_ctx_flat_t = actor.encode_ctx(obs_flat_t).detach()
            if not torch.isfinite(cached_ctx_flat_t).all():
                print(f"NaN/Inf detected in cached_ctx_flat_t at update={update}")
                raise ValueError("cached_ctx_flat_t contains NaN/Inf")
        logp_accel_flat_t = None
        logp_bw_flat_t = None
        logp_sat_flat_t = None
        if rollout_logprob_part_accel:
            logp_accel_flat_t = torch.from_numpy(
                np.concatenate(rollout_logprob_part_accel, axis=0).astype(np.float32, copy=False)
            ).to(device)
        if rollout_logprob_part_bw:
            logp_bw_flat_t = torch.from_numpy(
                np.concatenate(rollout_logprob_part_bw, axis=0).astype(np.float32, copy=False)
            ).to(device)
        if rollout_logprob_part_sat:
            logp_sat_flat_t = torch.from_numpy(
                np.concatenate(rollout_logprob_part_sat, axis=0).astype(np.float32, copy=False)
            ).to(device)
        logprob_part_tensors = {
            "accel": logp_accel_flat_t,
            "bw": logp_bw_flat_t,
            "sat": logp_sat_flat_t,
        }
        if not torch.isfinite(obs_flat_t).all():
            print(f"NaN/Inf detected in obs_flat_t at update={update}")
            raise ValueError("obs_flat_t contains NaN/Inf")
        if not torch.isfinite(act_flat_t).all():
            print(f"NaN/Inf detected in act_flat_t at update={update}")
            raise ValueError("act_flat_t contains NaN/Inf")
        if not torch.isfinite(logp_flat_t).all():
            print(f"NaN/Inf detected in logp_flat_t at update={update}")
            raise ValueError("logp_flat_t contains NaN/Inf")
        if not torch.isfinite(adv_flat_t).all():
            print(f"NaN/Inf detected in adv_flat_t at update={update}")
            raise ValueError("adv_flat_t contains NaN/Inf")
        if not torch.isfinite(ret_t).all():
            print(f"NaN/Inf detected in ret_t at update={update}")
            raise ValueError("ret_t contains NaN/Inf")
        for name, param in actor.named_parameters():
            if not torch.isfinite(param).all():
                print(f"NaN/Inf detected in actor param {name} before PPO update at update={update}")
                raise ValueError("actor parameters contain NaN/Inf before PPO update")

        batch_size = len(obs_flat)
        minibatch_size = max(1, batch_size // cfg.num_mini_batch)
        indices = np.arange(batch_size)
        imitation_mask = None
        if imitation_enabled_curr:
            mask_parts = []
            mask_parts.append(np.ones((2,), dtype=np.float32) if imitation_accel else np.zeros((2,), dtype=np.float32))
            if cfg.enable_bw_action:
                if imitation_bw:
                    mask_parts.append(np.ones((cfg.users_obs_max,), dtype=np.float32))
                else:
                    mask_parts.append(np.zeros((cfg.users_obs_max,), dtype=np.float32))
            if not cfg.fixed_satellite_strategy:
                if imitation_sat:
                    mask_parts.append(np.ones((cfg.sats_obs_max,), dtype=np.float32))
                else:
                    mask_parts.append(np.zeros((cfg.sats_obs_max,), dtype=np.float32))
            imitation_mask = torch.from_numpy(np.concatenate(mask_parts)).to(device)
            imitation_mask_sum = float(imitation_mask.sum().item())
        else:
            imitation_mask_sum = 0.0

        policy_losses = []
        value_losses = []
        entropies = []
        entropy_accel_values = []
        entropy_bw_values = []
        entropy_sat_values = []
        accel_std_values = []
        bw_std_values = []
        bw_concentration_values = []
        approx_kls = []
        approx_kl_accel_values = []
        approx_kl_bw_values = []
        approx_kl_sat_values = []
        clip_fracs = []
        clip_frac_accel_values = []
        clip_frac_bw_values = []
        clip_frac_sat_values = []
        ratio_samples = []
        log_ratio_abs_samples = []
        log_ratio_mean_accel_values = []
        log_ratio_mean_bw_values = []
        log_ratio_mean_sat_values = []
        log_ratio_var_accel_values = []
        log_ratio_var_bw_values = []
        log_ratio_var_sat_values = []
        bw_danger_sample_rates = []
        bw_danger_tightclip_hit_rates = []
        bw_danger_adv_threshold_values = []
        bw_danger_valid_count_threshold_values = []
        bw_danger_queue_max_threshold_values = []
        sat_sup_loss_top1_sum = 0.0
        sat_sup_loss_topk_sum = 0.0
        sat_sup_top1_acc_sum = 0.0
        sat_sup_exact_topk_sum = 0.0
        critic_epochs = int(getattr(cfg, "critic_epochs", 0) or cfg.ppo_epochs)
        critic_kl_stop_coupled = bool(getattr(cfg, "critic_kl_stop_coupled", True))
        ppo_headwise_surrogate_enabled = bool(getattr(cfg, "ppo_headwise_surrogate_enabled", False))
        ppo_headwise_kl_stop_enabled = bool(getattr(cfg, "ppo_headwise_kl_stop_enabled", True))
        use_headwise_surrogate = ppo_headwise_surrogate_enabled or (
            sat_counterfactual_credit_enabled and "sat" in adv_flat_by_head_t
        )
        bw_false_positive_mask_enabled = bool(getattr(cfg, "bw_false_positive_mask_enabled", False))
        actor_epochs_executed = 0
        actor_minibatches_executed = 0
        actor_minibatches_executed_by_head = {"accel": 0, "bw": 0, "sat": 0}
        critic_updates_executed = 0
        kl_stop_triggered = 0.0
        kl_stop_triggered_by_head = {"accel": 0.0, "bw": 0.0, "sat": 0.0}

        def _run_critic_update() -> float:
            if critic_multihead_value_enabled:
                value_pred_heads = critic.forward_heads(state_t, obs_step_t)
                if ppo_per_head_advantage_enabled:
                    value_loss = torch.stack(
                        [F.mse_loss(value_pred_heads[head_name], ret_t_by_head[head_name]) for head_name in VALUE_HEAD_NAMES],
                        dim=0,
                    ).mean()
                else:
                    value_loss = torch.stack(
                        [F.mse_loss(value_pred_heads[head_name], ret_t) for head_name in VALUE_HEAD_NAMES],
                        dim=0,
                    ).mean()
            else:
                value_pred = critic(state_t, obs_step_t)
                value_loss = F.mse_loss(value_pred, ret_t)
            critic_optim.zero_grad()
            (cfg.value_coef * value_loss).backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
            critic_optim.step()
            return float(value_loss.item())

        optim_start = time.perf_counter()
        stop_early = False
        head_active = {name: bool(train_heads.get(name, False)) for name in ("accel", "bw", "sat")}
        for epoch_idx in range(cfg.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]
                if not torch.isfinite(obs_flat_t[mb_idx]).all():
                    print(f"NaN/Inf detected in obs minibatch at update={update}")
                    raise ValueError("obs minibatch contains NaN/Inf")
                if not torch.isfinite(act_flat_t[mb_idx]).all():
                    print(f"NaN/Inf detected in act minibatch at update={update}")
                    raise ValueError("act minibatch contains NaN/Inf")
                ctx_mb = cached_ctx_flat_t[mb_idx] if cached_ctx_flat_t is not None else None
                current_head_names = (
                    tuple(head for head in train_head_names if head_active.get(head, False))
                    if use_headwise_surrogate
                    else train_head_names
                )
                if not current_head_names:
                    stop_early = True
                    break
                out = actor.forward(obs_flat_t[mb_idx], required_heads=current_head_names, ctx=ctx_mb)
                logprob_parts, entropy_parts = actor.evaluate_actions_parts(
                    obs_flat_t[mb_idx],
                    act_flat_t[mb_idx],
                    sat_indices=sat_indices_flat_t[mb_idx],
                    out=out,
                    heads=current_head_names,
                )
                part_metric_specs = (
                    (
                        "accel",
                        logprob_part_tensors["accel"],
                        approx_kl_accel_values,
                        clip_frac_accel_values,
                        log_ratio_mean_accel_values,
                        log_ratio_var_accel_values,
                    ),
                    (
                        "bw",
                        logprob_part_tensors["bw"],
                        approx_kl_bw_values,
                        clip_frac_bw_values,
                        log_ratio_mean_bw_values,
                        log_ratio_var_bw_values,
                    ),
                    (
                        "sat",
                        logprob_part_tensors["sat"],
                        approx_kl_sat_values,
                        clip_frac_sat_values,
                        log_ratio_mean_sat_values,
                        log_ratio_var_sat_values,
                    ),
                )
                if use_headwise_surrogate:
                    policy_terms = []
                    entropy = torch.zeros_like(adv_flat_t[mb_idx])
                    log_ratio = torch.zeros_like(adv_flat_t[mb_idx])
                    part_approx_kls: Dict[str, torch.Tensor] = {}
                    for part_name, old_part_t, approx_store, clip_store, mean_store, var_store in part_metric_specs:
                        if part_name not in logprob_parts or old_part_t is None:
                            continue
                        part_adv_t = adv_flat_by_head_t.get(part_name, adv_flat_t)[mb_idx]
                        (
                            policy_term,
                            part_log_ratio,
                            part_ratio,
                            approx_part,
                            bw_danger_metrics,
                        ) = _compute_part_policy_term(
                            part_name=part_name,
                            new_logprob=logprob_parts[part_name],
                            old_logprob=old_part_t[mb_idx],
                            part_adv_t=part_adv_t,
                            clip_ratio=float(cfg.clip_ratio),
                            cfg=cfg,
                            out=out,
                        )
                        policy_terms.append(policy_term)
                        entropy = entropy + entropy_parts.get(
                            part_name, torch.zeros_like(adv_flat_t[mb_idx], device=device)
                        )
                        log_ratio = log_ratio + part_log_ratio
                        part_approx_kls[part_name] = approx_part
                        approx_store.append(float(approx_part.item()))
                        clip_store.append(float(((part_ratio - 1.0).abs() > cfg.clip_ratio).float().mean().item()))
                        mean_store.append(float(part_log_ratio.mean().item()))
                        var_store.append(float(part_log_ratio.var(unbiased=False).item()))
                        if part_name == "bw":
                            bw_danger_sample_rates.append(bw_danger_metrics["sample_rate"])
                            bw_danger_tightclip_hit_rates.append(bw_danger_metrics["tightclip_hit_rate"])
                            bw_danger_adv_threshold_values.append(bw_danger_metrics["adv_threshold"])
                            bw_danger_valid_count_threshold_values.append(
                                bw_danger_metrics["valid_count_threshold"]
                            )
                            bw_danger_queue_max_threshold_values.append(
                                bw_danger_metrics["queue_max_threshold"]
                            )
                    if not policy_terms:
                        stop_early = True
                        break
                    log_ratio = torch.clamp(log_ratio, -8.0, 8.0)
                    ratio = torch.exp(log_ratio)
                    policy_loss = sum(policy_terms)
                    approx_kl = (-log_ratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_ratio).float().mean()
                else:
                    new_logp = _sum_selected_parts(logprob_parts, train_heads)
                    entropy = _sum_selected_parts(entropy_parts, train_heads)
                    if not torch.isfinite(new_logp).all():
                        print(f"NaN/Inf detected in new_logp at update={update}")
                        raise ValueError("new_logp contains NaN/Inf")

                    log_ratio = new_logp - logp_flat_t[mb_idx]
                    log_ratio = torch.clamp(log_ratio, -8.0, 8.0)
                    ratio = torch.exp(log_ratio)
                    part_approx_kls = {}
                    if ppo_per_head_advantage_enabled or (bw_false_positive_mask_enabled and "bw" in adv_flat_by_head_t):
                        policy_terms = []
                        for part_name, old_part_t, approx_store, clip_store, mean_store, var_store in part_metric_specs:
                            if part_name not in logprob_parts or old_part_t is None:
                                continue
                            part_adv_t = adv_flat_by_head_t.get(part_name, adv_flat_t)[mb_idx]
                            (
                                policy_term,
                                part_log_ratio,
                                part_ratio,
                                approx_part,
                                bw_danger_metrics,
                            ) = _compute_part_policy_term(
                                part_name=part_name,
                                new_logprob=logprob_parts[part_name],
                                old_logprob=old_part_t[mb_idx],
                                part_adv_t=part_adv_t,
                                clip_ratio=float(cfg.clip_ratio),
                                cfg=cfg,
                                out=out,
                            )
                            policy_terms.append(policy_term)
                            part_approx_kls[part_name] = approx_part
                            approx_store.append(float(approx_part.item()))
                            clip_store.append(float(((part_ratio - 1.0).abs() > cfg.clip_ratio).float().mean().item()))
                            mean_store.append(float(part_log_ratio.mean().item()))
                            var_store.append(float(part_log_ratio.var(unbiased=False).item()))
                            if part_name == "bw":
                                bw_danger_sample_rates.append(bw_danger_metrics["sample_rate"])
                                bw_danger_tightclip_hit_rates.append(bw_danger_metrics["tightclip_hit_rate"])
                                bw_danger_adv_threshold_values.append(bw_danger_metrics["adv_threshold"])
                                bw_danger_valid_count_threshold_values.append(
                                    bw_danger_metrics["valid_count_threshold"]
                                )
                                bw_danger_queue_max_threshold_values.append(
                                    bw_danger_metrics["queue_max_threshold"]
                                )
                        if not policy_terms:
                            stop_early = True
                            break
                        policy_loss = sum(policy_terms)
                    else:
                        surr1 = ratio * adv_flat_t[mb_idx]
                        surr2 = (
                            torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv_flat_t[mb_idx]
                        )
                        policy_loss = -torch.min(surr1, surr2).mean()
                        for part_name, old_part_t, approx_store, clip_store, mean_store, var_store in part_metric_specs:
                            if part_name not in logprob_parts or old_part_t is None:
                                continue
                            part_log_ratio = torch.clamp(logprob_parts[part_name] - old_part_t[mb_idx], -8.0, 8.0)
                            part_ratio = torch.exp(part_log_ratio)
                            approx_part = (old_part_t[mb_idx] - logprob_parts[part_name]).mean()
                            part_approx_kls[part_name] = approx_part
                            approx_store.append(float(approx_part.item()))
                            clip_store.append(float(((part_ratio - 1.0).abs() > cfg.clip_ratio).float().mean().item()))
                            mean_store.append(float(part_log_ratio.mean().item()))
                            var_store.append(float(part_log_ratio.var(unbiased=False).item()))
                    approx_kl = (logp_flat_t[mb_idx] - new_logp).mean()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_ratio).float().mean()
                if (
                    debug_first_minibatch_ratio
                    and not debug_first_minibatch_ratio_printed
                    and epoch_idx == 0
                    and start == 0
                ):
                    ratio_np = ratio.detach().cpu().numpy().reshape(-1)
                    lines = [
                        (
                            f"[debug_first_minibatch_ratio] update={update} epoch=0 minibatch=0 "
                            f"ratio_mean={float(np.mean(ratio_np)):.6f} "
                            f"ratio_min={float(np.min(ratio_np)):.6f} "
                            f"ratio_max={float(np.max(ratio_np)):.6f}"
                        )
                    ]
                    for part_name, old_part_t in (
                        ("accel", logp_accel_flat_t),
                        ("bw", logp_bw_flat_t),
                        ("sat", logp_sat_flat_t),
                    ):
                        if part_name not in logprob_parts or old_part_t is None:
                            continue
                        old_np = old_part_t[mb_idx].detach().cpu().numpy().reshape(-1)
                        new_np = logprob_parts[part_name].detach().cpu().numpy().reshape(-1)
                        diff_np = new_np - old_np
                        lines.append(
                            f"  {part_name}_logprob_old mean/min/max="
                            f"{float(np.mean(old_np)):.6f}/{float(np.min(old_np)):.6f}/{float(np.max(old_np)):.6f}"
                        )
                        lines.append(
                            f"  {part_name}_logprob_new mean/min/max="
                            f"{float(np.mean(new_np)):.6f}/{float(np.min(new_np)):.6f}/{float(np.max(new_np)):.6f}"
                        )
                        lines.append(
                            f"  {part_name}_logprob_delta mean/min/max="
                            f"{float(np.mean(diff_np)):.6f}/{float(np.min(diff_np)):.6f}/{float(np.max(diff_np)):.6f}"
                        )
                    print("\n".join(lines))
                    debug_first_minibatch_ratio_printed = True
                kl_coef = float(getattr(cfg, "kl_coef", 0.0) or 0.0)
                if kl_coef > 0:
                    policy_loss = policy_loss + kl_coef * approx_kl
                if getattr(cfg, "kl_stop", False) and (
                    not use_headwise_surrogate or not ppo_headwise_kl_stop_enabled
                ):
                    target_kl = float(getattr(cfg, "target_kl", 0.0) or 0.0)
                    if target_kl > 0 and float(approx_kl.item()) > target_kl:
                        stop_early = True
                        kl_stop_triggered = 1.0
                head_should_stop_after_mb: list[str] = []
                if getattr(cfg, "kl_stop", False) and use_headwise_surrogate and ppo_headwise_kl_stop_enabled:
                    target_kl = float(getattr(cfg, "target_kl", 0.0) or 0.0)
                    if target_kl > 0:
                        for part_name in current_head_names:
                            approx_part = part_approx_kls.get(part_name)
                            if approx_part is not None and float(approx_part.item()) > target_kl:
                                head_should_stop_after_mb.append(part_name)
                                kl_stop_triggered = 1.0
                                kl_stop_triggered_by_head[part_name] = 1.0

                sat_sup_loss = torch.zeros((), device=device)
                sat_sup_metrics = {
                    "sat_sup_loss_top1": 0.0,
                    "sat_sup_loss_topk": 0.0,
                    "sat_sup_top1_acc": 0.0,
                    "sat_sup_exact_topk": 0.0,
                }
                sat_supervision_coef = float(getattr(cfg, "sat_supervision_coef", 0.0) or 0.0)
                if (
                    bool(getattr(cfg, "sat_supervision_enabled", False))
                    and sat_supervision_coef > 0.0
                    and "sat" in current_head_names
                    and "sat_logits" in out
                    and "sat_projected_rate" in out
                    and "sat_valid_mask" in out
                ):
                    sat_sup_loss, sat_sup_metrics = compute_sat_supervision_loss(
                        sat_logits=out["sat_logits"],
                        sat_projected_rate=out["sat_projected_rate"],
                        sat_valid_mask=out["sat_valid_mask"],
                        k=int(getattr(cfg, "sat_action_select_k", getattr(cfg, "sat_num_select", 2)) or 2),
                        top1_coef=float(getattr(cfg, "sat_supervision_top1_coef", 1.0) or 1.0),
                        topk_coef=float(getattr(cfg, "sat_supervision_topk_coef", 0.5) or 0.5),
                    )

                imitation_loss = torch.tensor(0.0, device=device)
                if imitation_enabled_curr and imitation_mask is not None and imitation_mask_sum > 0:
                    pred_action = actor.act(
                        obs_flat_t[mb_idx],
                        deterministic=True,
                        compute_logprob=False,
                        ctx=ctx_mb,
                    ).action
                    target_action = imitation_flat_t[mb_idx]
                    diff = (pred_action - target_action) * imitation_mask
                    imitation_loss = (diff.pow(2).sum(-1) / (imitation_mask_sum + 1e-9)).mean()

                residual_reg_loss = torch.tensor(0.0, device=device)
                if "bw" in current_head_names and bw_residual_l2_coef > 0.0 and "bw_alpha" in out:
                    residual_reg_loss = out["bw_alpha"].pow(2).mean()

                danger_imitation_loss = torch.tensor(0.0, device=device)
                if danger_imitation_enabled and "mu" in out:
                    pred_accel = squash_action(out["mu"])
                    target_accel = danger_imitation_target_flat_t[mb_idx]
                    danger_mask = danger_imitation_mask_flat_t[mb_idx]
                    active = torch.sum(danger_mask, dim=-1) > 0.0
                    if torch.any(active):
                        diff = (pred_accel - target_accel) * danger_mask
                        denom = torch.sum(danger_mask, dim=-1) + 1e-9
                        per_row = diff.pow(2).sum(-1) / denom
                        danger_imitation_loss = per_row[active].mean()

                actor_optim.zero_grad()
                actor_loss = (
                    policy_loss
                    + imitation_coef_curr * imitation_loss
                    + bw_residual_l2_coef * residual_reg_loss
                    + danger_imitation_coef * danger_imitation_loss
                    - cfg.entropy_coef * entropy.mean()
                    + sat_supervision_coef * sat_sup_loss
                )
                actor_loss.backward()
                for name, param in actor.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        print(f"NaN/Inf detected in actor grad {name} at update={update}")
                        raise ValueError("actor gradient contains NaN/Inf")
                torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                if "bw" in current_head_names and not np.isclose(bw_head_grad_scale, 1.0):
                    _apply_bw_head_grad_scale(actor, bw_head_grad_scale)
                actor_optim.step()
                for name, param in actor.named_parameters():
                    if not torch.isfinite(param).all():
                        print(f"NaN/Inf detected in actor param {name} after step at update={update}")
                        raise ValueError("actor parameters contain NaN/Inf after step")
                actor_minibatches_executed += 1
                for head_name in current_head_names:
                    actor_minibatches_executed_by_head[head_name] += 1
                actor_epochs_executed = max(actor_epochs_executed, epoch_idx + 1)
                if use_headwise_surrogate:
                    for head_name in head_should_stop_after_mb:
                        head_active[head_name] = False

                policy_losses.append(policy_loss.item())
                entropies.append(entropy.mean().item())
                entropy_accel_values.append(float(entropy_parts.get("accel", torch.zeros(1, device=device)).mean().item()))
                entropy_bw_values.append(float(entropy_parts.get("bw", torch.zeros(1, device=device)).mean().item()))
                entropy_sat_values.append(float(entropy_parts.get("sat", torch.zeros(1, device=device)).mean().item()))
                if "accel" in current_head_names:
                    accel_std_values.append(
                        float(torch.exp(torch.clamp(actor.log_std.detach(), -5.0, 2.0)).mean().item())
                    )
                if "bw" in current_head_names and "bw_alpha" in out and "bw_valid_mask" in out:
                    bw_alpha = out["bw_alpha"]
                    bw_valid = out["bw_valid_mask"] > 0.5
                    if torch.any(bw_valid):
                        bw_alpha_masked = torch.where(bw_valid, bw_alpha, torch.zeros_like(bw_alpha))
                        bw_alpha0 = bw_alpha_masked.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                        bw_var = bw_alpha_masked * (bw_alpha0 - bw_alpha_masked) / (
                            bw_alpha0.pow(2) * (bw_alpha0 + 1.0)
                        )
                        bw_std = torch.sqrt(torch.clamp(bw_var, min=0.0))
                        bw_std_values.append(float(bw_std[bw_valid].mean().item()))
                        bw_concentration_values.append(float(bw_alpha0.mean().item()))
                approx_kls.append(float(approx_kl.item()))
                clip_fracs.append(float(clip_frac.item()))
                ratio_samples.append(ratio.detach().cpu().numpy().reshape(-1))
                log_ratio_abs_samples.append(log_ratio.abs().detach().cpu().numpy().reshape(-1))
                imitation_loss_sum += float(imitation_loss.item())
                danger_imitation_loss_sum += float(danger_imitation_loss.item())
                residual_reg_loss_sum += float(residual_reg_loss.item())
                sat_sup_loss_top1_sum += float(sat_sup_metrics["sat_sup_loss_top1"])
                sat_sup_loss_topk_sum += float(sat_sup_metrics["sat_sup_loss_topk"])
                sat_sup_top1_acc_sum += float(sat_sup_metrics["sat_sup_top1_acc"])
                sat_sup_exact_topk_sum += float(sat_sup_metrics["sat_sup_exact_topk"])
                if use_headwise_surrogate and not any(head_active.get(head, False) for head in train_head_names):
                    stop_early = True
                if stop_early:
                    break
            if stop_early:
                break

            if critic_kl_stop_coupled:
                value_losses.append(_run_critic_update())
                critic_updates_executed += 1
        if not critic_kl_stop_coupled:
            for _ in range(max(critic_epochs, 0)):
                value_losses.append(_run_critic_update())
                critic_updates_executed += 1
        optim_time = time.perf_counter() - optim_start

        update_time = time.perf_counter() - update_start
        steps_count = max(1, steps_count)
        rollout_reward_per_step = ep_reward / steps_count
        episode_reward_stats = _rolling_window_summary(recent_episode_returns)
        episode_reward = episode_reward_stats["mean"]
        episode_length_mean = float(np.mean(recent_episode_lengths)) if recent_episode_lengths else 0.0
        episode_term_throughput_access = (
            float(np.mean(recent_episode_term_throughput_access))
            if recent_episode_term_throughput_access
            else 0.0
        )
        episode_term_throughput_backhaul = (
            float(np.mean(recent_episode_term_throughput_backhaul))
            if recent_episode_term_throughput_backhaul
            else 0.0
        )
        episode_term_queue_gu_arrival = (
            float(np.mean(recent_episode_term_queue_gu_arrival))
            if recent_episode_term_queue_gu_arrival
            else 0.0
        )
        collision_rate = float(np.mean(recent_episode_collisions)) if recent_episode_collisions else 0.0
        log_std_terms: List[np.ndarray] = []
        if train_heads["accel"]:
            log_std_terms.append(torch.clamp(actor.log_std.detach(), -5.0, 2.0).cpu().numpy().reshape(-1))
        if log_std_terms:
            log_std_vec = np.concatenate(log_std_terms, axis=0)
        else:
            log_std_vec = np.zeros((1,), dtype=np.float32)
        action_std_vec = np.exp(log_std_vec)
        if ratio_samples:
            ratio_vec = np.concatenate(ratio_samples, axis=0)
            ratio_p50 = float(np.percentile(ratio_vec, 50.0))
            ratio_p90 = float(np.percentile(ratio_vec, 90.0))
            ratio_p99 = float(np.percentile(ratio_vec, 99.0))
        else:
            ratio_p50 = 1.0
            ratio_p90 = 1.0
            ratio_p99 = 1.0
        if log_ratio_abs_samples:
            log_ratio_abs_vec = np.concatenate(log_ratio_abs_samples, axis=0)
            log_ratio_abs_mean = float(np.mean(log_ratio_abs_vec))
        else:
            log_ratio_abs_mean = 0.0
        explained_variance = 0.0
        returns_np = np.asarray(rets, dtype=np.float32).reshape(-1)
        ret_mean = float(np.mean(returns_np)) if returns_np.size else 0.0
        ret_std = float(np.std(returns_np)) if returns_np.size else 0.0
        ret_p95 = float(np.percentile(returns_np, 95.0)) if returns_np.size else 0.0
        if returns_np.size > 1:
            with torch.inference_mode():
                value_pred_eval = critic(state_t, obs_step_t).detach().cpu().numpy().reshape(-1)
            returns_var = float(np.var(returns_np))
            if returns_var > 1e-8:
                explained_variance = 1.0 - float(np.var(returns_np - value_pred_eval)) / returns_var
        reward_rms_mean = float(reward_rms.mean) if reward_rms is not None else float("nan")
        reward_rms_var = float(reward_rms.var) if reward_rms is not None else float("nan")
        assoc_unfair_episode_frac_mean = (
            float(np.mean(assoc_unfair_episode_fracs)) if assoc_unfair_episode_fracs else 0.0
        )
        assoc_unfair_episode_frac_p95 = (
            float(np.percentile(assoc_unfair_episode_fracs, 95.0)) if assoc_unfair_episode_fracs else 0.0
        )
        assoc_unfair_episode_frac_max = (
            float(np.max(assoc_unfair_episode_fracs)) if assoc_unfair_episode_fracs else 0.0
        )
        metrics = {
            "episode_reward": episode_reward,
            "episode_reward_std": episode_reward_stats["std"],
            "episode_reward_p25": episode_reward_stats["p25"],
            "episode_reward_p75": episode_reward_stats["p75"],
            "episode_length_mean": episode_length_mean,
            "completed_episode_count": float(completed_episode_count),
            "rollout_reward_per_step": rollout_reward_per_step,
            "reward_raw": reward_raw_sum / steps_count,
            "reward_aux": reward_aux_sum / steps_count,
            "x_acc": x_acc_sum / steps_count,
            "x_rel": x_rel_sum / steps_count,
            "g_pre": g_pre_sum / steps_count,
            "d_pre": d_pre_sum / steps_count,
            "processed_ratio_eval": processed_ratio_eval_sum / steps_count,
            "drop_ratio_eval": drop_ratio_eval_sum / steps_count,
            "pre_backlog_steps_eval": pre_backlog_steps_eval_sum / steps_count,
            "assoc_centroid_dist_norm_mean": assoc_centroid_dist_norm_sum / steps_count,
            "stage1_assoc_centroid_weight": stage1_assoc_centroid_weight_sum / steps_count,
            "stage1_assoc_centroid_term": stage1_assoc_centroid_term_sum / steps_count,
            "sat_overlap_eval": sat_overlap_eval_sum / steps_count,
            "stage3_sat_overlap_weight": stage3_sat_overlap_weight_sum / steps_count,
            "stage3_sat_overlap_term": stage3_sat_overlap_term_sum / steps_count,
            "D_sys_report": d_sys_report_sum / steps_count,
            "episode_term_throughput_access": episode_term_throughput_access,
            "episode_term_throughput_backhaul": episode_term_throughput_backhaul,
            "episode_term_queue_gu_arrival": episode_term_queue_gu_arrival,
            "throughput_access_norm": throughput_access_norm_sum / steps_count,
            "throughput_backhaul_norm": throughput_backhaul_norm_sum / steps_count,
            "gu_queue_arrival_steps": gu_queue_arrival_steps_sum / steps_count,
            "gu_queue_mean": gu_queue_sum / steps_count,
            "uav_queue_mean": uav_queue_sum / steps_count,
            "queue_total_active": queue_total_active_sum / steps_count,
            "collision_rate": collision_rate,
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "ret_mean": ret_mean,
            "ret_std": ret_std,
            "ret_p95": ret_p95,
            "explained_variance": explained_variance,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "entropy_accel": float(np.mean(entropy_accel_values)) if entropy_accel_values else 0.0,
            "entropy_bw": float(np.mean(entropy_bw_values)) if entropy_bw_values else 0.0,
            "entropy_sat": float(np.mean(entropy_sat_values)) if entropy_sat_values else 0.0,
            "accel_std_mean": float(np.mean(accel_std_values)) if accel_std_values else 0.0,
            "bw_std_mean": float(np.mean(bw_std_values)) if bw_std_values else 0.0,
            "bw_concentration_mean": float(np.mean(bw_concentration_values)) if bw_concentration_values else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "approx_kl_accel": float(np.mean(approx_kl_accel_values)) if approx_kl_accel_values else 0.0,
            "approx_kl_bw": float(np.mean(approx_kl_bw_values)) if approx_kl_bw_values else 0.0,
            "approx_kl_sat": float(np.mean(approx_kl_sat_values)) if approx_kl_sat_values else 0.0,
            "clip_frac": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            "clip_frac_accel": float(np.mean(clip_frac_accel_values)) if clip_frac_accel_values else 0.0,
            "clip_frac_bw": float(np.mean(clip_frac_bw_values)) if clip_frac_bw_values else 0.0,
            "clip_frac_sat": float(np.mean(clip_frac_sat_values)) if clip_frac_sat_values else 0.0,
            "ratio_p50": ratio_p50,
            "ratio_p90": ratio_p90,
            "ratio_p99": ratio_p99,
            "log_ratio_abs_mean": log_ratio_abs_mean,
            "log_ratio_mean_accel": float(np.mean(log_ratio_mean_accel_values)) if log_ratio_mean_accel_values else 0.0,
            "log_ratio_mean_bw": float(np.mean(log_ratio_mean_bw_values)) if log_ratio_mean_bw_values else 0.0,
            "log_ratio_mean_sat": float(np.mean(log_ratio_mean_sat_values)) if log_ratio_mean_sat_values else 0.0,
            "log_ratio_var_accel": float(np.mean(log_ratio_var_accel_values)) if log_ratio_var_accel_values else 0.0,
            "log_ratio_var_bw": float(np.mean(log_ratio_var_bw_values)) if log_ratio_var_bw_values else 0.0,
            "log_ratio_var_sat": float(np.mean(log_ratio_var_sat_values)) if log_ratio_var_sat_values else 0.0,
            "bw_false_positive_mask_rate": bw_false_positive_mask_rate,
            "bw_false_positive_bad_score_mean": bw_false_positive_bad_score_mean,
            "bw_false_positive_bad_tau": bw_false_positive_bad_tau,
            "bw_danger_sample_rate": float(np.mean(bw_danger_sample_rates)) if bw_danger_sample_rates else 0.0,
            "bw_danger_tightclip_hit_rate": (
                float(np.mean(bw_danger_tightclip_hit_rates)) if bw_danger_tightclip_hit_rates else 0.0
            ),
            "bw_danger_adv_threshold": (
                float(np.mean(bw_danger_adv_threshold_values)) if bw_danger_adv_threshold_values else 0.0
            ),
            "bw_danger_valid_count_threshold": (
                float(np.mean(bw_danger_valid_count_threshold_values))
                if bw_danger_valid_count_threshold_values
                else 0.0
            ),
            "bw_danger_queue_max_threshold": (
                float(np.mean(bw_danger_queue_max_threshold_values))
                if bw_danger_queue_max_threshold_values
                else 0.0
            ),
            "reward_rms_mean": reward_rms_mean,
            "reward_rms_var": reward_rms_var,
            "danger_imitation_loss": danger_imitation_loss_sum / max(1, len(policy_losses)),
            "residual_reg_loss": residual_reg_loss_sum / max(1, len(policy_losses)),
            "sat_sup_loss_top1": sat_sup_loss_top1_sum / max(1, len(policy_losses)),
            "sat_sup_loss_topk": sat_sup_loss_topk_sum / max(1, len(policy_losses)),
            "sat_sup_top1_acc": sat_sup_top1_acc_sum / max(1, len(policy_losses)),
            "sat_sup_exact_topk": sat_sup_exact_topk_sum / max(1, len(policy_losses)),
            "sat_cf_credit_sampled_contexts": sat_cf_credit_sampled_contexts_sum,
            "sat_cf_credit_selected_q_mean": (
                sat_cf_credit_selected_q_sum / sat_cf_credit_sampled_contexts_sum
                if sat_cf_credit_sampled_contexts_sum > 0.0
                else 0.0
            ),
            "sat_cf_credit_baseline_q_mean": (
                sat_cf_credit_baseline_q_sum / sat_cf_credit_sampled_contexts_sum
                if sat_cf_credit_sampled_contexts_sum > 0.0
                else 0.0
            ),
            "sat_cf_credit_mean": (
                sat_cf_credit_sum / sat_cf_credit_sampled_contexts_sum
                if sat_cf_credit_sampled_contexts_sum > 0.0
                else 0.0
            ),
            "sat_cf_credit_abs_mean": (
                sat_cf_credit_abs_sum / sat_cf_credit_sampled_contexts_sum
                if sat_cf_credit_sampled_contexts_sum > 0.0
                else 0.0
            ),
            "actor_epochs_executed": float(actor_epochs_executed),
            "actor_minibatches_executed": float(actor_minibatches_executed),
            "actor_minibatches_executed_accel": float(actor_minibatches_executed_by_head["accel"]),
            "actor_minibatches_executed_bw": float(actor_minibatches_executed_by_head["bw"]),
            "actor_minibatches_executed_sat": float(actor_minibatches_executed_by_head["sat"]),
            "critic_updates_executed": float(critic_updates_executed),
            "kl_stop_triggered": kl_stop_triggered,
            "kl_stop_triggered_accel": kl_stop_triggered_by_head["accel"],
            "kl_stop_triggered_bw": kl_stop_triggered_by_head["bw"],
            "kl_stop_triggered_sat": kl_stop_triggered_by_head["sat"],
            "danger_imitation_coef": danger_imitation_coef,
            "danger_imitation_active_rate": danger_imitation_active_rate_sum / steps_count,
            "log_std_mean": float(np.mean(log_std_vec)),
            "actor_lr": float(actor_optim.param_groups[0]["lr"]),
            "critic_lr": float(critic_optim.param_groups[0]["lr"]),
            "update_time_sec": update_time,
            "rollout_time_sec": rollout_time,
            "optim_time_sec": optim_time,
            "dynamics_time_sec": dynamics_time_sum / steps_count,
            "orbit_visible_time_sec": orbit_visible_time_sum / steps_count,
            "assoc_access_time_sec": assoc_access_time_sum / steps_count,
            "backhaul_queue_time_sec": backhaul_queue_time_sum / steps_count,
            "reward_time_sec": reward_time_sum / steps_count,
            "obs_time_sec": obs_time_sum / steps_count,
            "state_time_sec": state_time_sum / steps_count,
            "global_state_time_sec": state_time_sum / steps_count,
            "step_total_time_sec": step_total_time_sum / steps_count,
            "env_steps_per_sec": steps_count / max(1e-9, rollout_time),
            "update_steps_per_sec": steps_count / max(1e-9, update_time),
            "total_env_steps": float(total_env_steps),
            "total_time_sec": time.perf_counter() - training_start,
        }
        for agent_idx in range(num_agents):
            metrics[f"stage1_aux_agent{agent_idx}"] = float(stage1_aux_agent_sum[agent_idx] / steps_count)
            metrics[f"stage3_overlap_agent{agent_idx}"] = float(stage3_overlap_agent_sum[agent_idx] / steps_count)

        logger.log(
            update,
            metrics,
        )
        if actor_sched is not None:
            actor_sched.step()
        if critic_sched is not None:
            critic_sched.step()
        completed_updates = update + 1
        progress.update(local_update + 1)
        if save_interval_updates > 0 and (update + 1) % save_interval_updates == 0:
            checkpoint_suffix = f"u{update + 1:04d}"
            _save_checkpoints(log_dir, actor, critic, suffix=checkpoint_suffix)
            _save_train_state(
                log_dir,
                actor,
                critic,
                actor_optim,
                critic_optim,
                actor_sched,
                critic_sched,
                reward_rms,
                update=update + 1,
                planned_total_updates=planned_total_updates,
                total_env_steps=total_env_steps,
                reward_history=reward_history,
                best_ma=best_ma,
                no_improve=no_improve,
                checkpoint_eval_state=checkpoint_eval_state,
                checkpoint_eval_fixed_summary=checkpoint_eval_fixed_summary,
                total_time_sec=time.perf_counter() - training_start,
                suffix=checkpoint_suffix,
            )

        checkpoint_eval_due = (
            checkpoint_eval_enabled
            and (update + 1) >= checkpoint_eval_start_update
            and ((update + 1 - checkpoint_eval_start_update) % checkpoint_eval_interval == 0)
        )
        if checkpoint_eval_due and checkpoint_eval_fixed_summary is not None:
            checkpoint_suffix = f"u{update + 1:04d}"
            _save_checkpoints(log_dir, actor, critic, suffix=checkpoint_suffix)
            _save_train_state(
                log_dir,
                actor,
                critic,
                actor_optim,
                critic_optim,
                actor_sched,
                critic_sched,
                reward_rms,
                update=update + 1,
                planned_total_updates=planned_total_updates,
                total_env_steps=total_env_steps,
                reward_history=reward_history,
                best_ma=best_ma,
                no_improve=no_improve,
                checkpoint_eval_state=checkpoint_eval_state,
                checkpoint_eval_fixed_summary=checkpoint_eval_fixed_summary,
                total_time_sec=time.perf_counter() - training_start,
                suffix=checkpoint_suffix,
            )
            checkpoint_eval_summary = _checkpoint_eval_summary(
                cfg,
                actor=actor,
                device=device,
                episodes=checkpoint_eval_episodes,
                episode_seed_base=checkpoint_eval_episode_seed_base,
                exec_accel_source=exec_accel_source,
                exec_bw_source=exec_bw_source,
                exec_sat_source=exec_sat_source,
                teacher_actor=teacher_actor,
                teacher_deterministic=teacher_deterministic,
                need_heuristic_exec=need_heuristic_exec,
                fixed_baseline=False,
            )
            checkpoint_eval_flags = _update_checkpoint_eval_state(
                checkpoint_eval_state,
                checkpoint_eval_summary,
                cfg,
            )
            checkpoint_eval_row = {
                "update": update + 1,
                "checkpoint_suffix": checkpoint_suffix,
                "episodes": checkpoint_eval_summary["episodes"],
                "reward_sum": checkpoint_eval_summary["reward_sum"],
                "processed_ratio_eval": checkpoint_eval_summary["processed_ratio_eval"],
                "drop_ratio_eval": checkpoint_eval_summary["drop_ratio_eval"],
                "pre_backlog_steps_eval": checkpoint_eval_summary["pre_backlog_steps_eval"],
                "D_sys_report": checkpoint_eval_summary["D_sys_report"],
                "x_acc_mean": checkpoint_eval_summary["x_acc_mean"],
                "x_rel_mean": checkpoint_eval_summary["x_rel_mean"],
                "g_pre_mean": checkpoint_eval_summary["g_pre_mean"],
                "d_pre_mean": checkpoint_eval_summary["d_pre_mean"],
                "sat_overlap_eval": checkpoint_eval_summary["sat_overlap_eval"],
                "collision_episode_fraction": checkpoint_eval_summary["collision_episode_fraction"],
                "fixed_reward_sum": checkpoint_eval_fixed_summary["reward_sum"],
                "fixed_processed_ratio_eval": checkpoint_eval_fixed_summary["processed_ratio_eval"],
                "fixed_drop_ratio_eval": checkpoint_eval_fixed_summary["drop_ratio_eval"],
                "fixed_pre_backlog_steps_eval": checkpoint_eval_fixed_summary["pre_backlog_steps_eval"],
                "fixed_D_sys_report": checkpoint_eval_fixed_summary["D_sys_report"],
                "fixed_x_acc_mean": checkpoint_eval_fixed_summary["x_acc_mean"],
                "fixed_x_rel_mean": checkpoint_eval_fixed_summary["x_rel_mean"],
                "fixed_g_pre_mean": checkpoint_eval_fixed_summary["g_pre_mean"],
                "fixed_d_pre_mean": checkpoint_eval_fixed_summary["d_pre_mean"],
                "fixed_sat_overlap_eval": checkpoint_eval_fixed_summary["sat_overlap_eval"],
                "fixed_collision_episode_fraction": checkpoint_eval_fixed_summary["collision_episode_fraction"],
            }
            checkpoint_eval_row.update(checkpoint_eval_flags)
            _append_checkpoint_eval_row(checkpoint_eval_csv_path, checkpoint_eval_row)
            if checkpoint_eval_flags["model_improved"] > 0.5:
                _save_checkpoints(log_dir, actor, critic, suffix="best")
                _save_train_state(
                    log_dir,
                    actor,
                    critic,
                    actor_optim,
                    critic_optim,
                    actor_sched,
                    critic_sched,
                    reward_rms,
                    update=update + 1,
                    planned_total_updates=planned_total_updates,
                    total_env_steps=total_env_steps,
                    reward_history=reward_history,
                    best_ma=best_ma,
                    no_improve=no_improve,
                    checkpoint_eval_state=checkpoint_eval_state,
                    checkpoint_eval_fixed_summary=checkpoint_eval_fixed_summary,
                    total_time_sec=time.perf_counter() - training_start,
                    suffix="best",
                )
                best_checkpoint_saved = True
            print(
                "Checkpoint eval "
                f"{checkpoint_suffix}: "
                f"reward={checkpoint_eval_summary['reward_sum']:.4f} "
                f"({checkpoint_eval_fixed_policy} ref {checkpoint_eval_fixed_summary['reward_sum']:.4f}), "
                f"processed={checkpoint_eval_summary['processed_ratio_eval']:.4f} "
                f"({checkpoint_eval_fixed_policy} ref {checkpoint_eval_fixed_summary['processed_ratio_eval']:.4f}), "
                f"drop={checkpoint_eval_summary['drop_ratio_eval']:.4f} "
                f"({checkpoint_eval_fixed_policy} ref {checkpoint_eval_fixed_summary['drop_ratio_eval']:.4f}), "
                f"pre_backlog={checkpoint_eval_summary['pre_backlog_steps_eval']:.4f} "
                f"({checkpoint_eval_fixed_policy} ref {checkpoint_eval_fixed_summary['pre_backlog_steps_eval']:.4f}), "
                f"sat_overlap={checkpoint_eval_summary['sat_overlap_eval']:.4f} "
                f"({checkpoint_eval_fixed_policy} ref {checkpoint_eval_fixed_summary['sat_overlap_eval']:.4f}), "
                f"collision={checkpoint_eval_summary['collision_episode_fraction']:.4f} "
                f"({checkpoint_eval_fixed_policy} ref {checkpoint_eval_fixed_summary['collision_episode_fraction']:.4f}), "
                f"model_improved={int(checkpoint_eval_flags['model_improved'])}, "
                f"worse_streak={int(checkpoint_eval_flags['quality_worse_streak'])}, "
                f"reward_plateau_streak={int(checkpoint_eval_flags['reward_plateau_streak'])}, "
                f"collision_gate={int(checkpoint_eval_flags['collision_gate_passed'])}"
            )
            if checkpoint_eval_early_stop_enabled and checkpoint_eval_flags["early_stop_triggered"] > 0.5:
                if checkpoint_eval_flags["reward_early_stop_triggered"] > 0.5:
                    print(
                        f"Checkpoint-eval early stopping at update {update + 1}: "
                        "reward_sum plateaued and collision gate passed."
                    )
                else:
                    print(
                        f"Checkpoint-eval early stopping at update {update + 1}: "
                        "processed/drop/pre_backlog quality kept worsening."
                    )
                break

        if cfg.early_stop_enabled:
            reward_history.append(episode_reward)
            if update + 1 >= cfg.early_stop_min_updates and len(reward_history) >= cfg.early_stop_window:
                ma = float(np.mean(reward_history[-cfg.early_stop_window :]))
                if ma > best_ma + cfg.early_stop_min_delta:
                    best_ma = ma
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= cfg.early_stop_patience:
                    print(
                        f"Early stopping at update {update + 1}: "
                        f"moving average={ma:.6f}, best={best_ma:.6f}"
                    )
                    break

    if checkpoint_eval_enabled and not best_checkpoint_saved:
        _save_checkpoints(log_dir, actor, critic, suffix="best")
        _save_train_state(
            log_dir,
            actor,
            critic,
            actor_optim,
            critic_optim,
            actor_sched,
            critic_sched,
            reward_rms,
            update=completed_updates,
            planned_total_updates=planned_total_updates,
            total_env_steps=total_env_steps,
            reward_history=reward_history,
            best_ma=best_ma,
            no_improve=no_improve,
            checkpoint_eval_state=checkpoint_eval_state,
            checkpoint_eval_fixed_summary=checkpoint_eval_fixed_summary,
            total_time_sec=time.perf_counter() - training_start,
            suffix="best",
        )

    _save_checkpoints(log_dir, actor, critic)
    _save_train_state(
        log_dir,
        actor,
        critic,
        actor_optim,
        critic_optim,
        actor_sched,
        critic_sched,
        reward_rms,
        update=completed_updates,
        planned_total_updates=planned_total_updates,
        total_env_steps=total_env_steps,
        reward_history=reward_history,
        best_ma=best_ma,
        no_improve=no_improve,
        checkpoint_eval_state=checkpoint_eval_state,
        checkpoint_eval_fixed_summary=checkpoint_eval_fixed_summary,
        total_time_sec=time.perf_counter() - training_start,
    )

    progress.close()
    logger.close()
    return actor, critic
