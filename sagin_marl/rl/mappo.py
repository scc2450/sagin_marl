from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .policy import ActorNet, batch_flatten_obs
from .critic import CriticNet
from .buffer import RolloutBuffer
from .action_assembler import assemble_actions
from .baselines import queue_aware_policy
from ..utils.normalization import RunningMeanStd
from ..env.config import ablation_flag


def compute_gae(rewards, values, dones, gamma, lam):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_adv = 0.0
    last_value = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * last_value * mask - values[t]
        last_adv = delta + gamma * lam * mask * last_adv
        advantages[t] = last_adv
        last_value = values[t]
    returns = advantages + values
    return advantages, returns


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
    parts = getattr(env, "last_reward_parts", None) or {}
    sat_processed = getattr(env, "last_sat_processed", None)
    sat_incoming = getattr(env, "last_sat_incoming", None)

    return {
        "last_exec_accel": np.asarray(getattr(env, "last_exec_accel", default_accel), dtype=np.float32),
        "gu_queue_mean": _safe_mean(getattr(env, "gu_queue", 0.0)),
        "uav_queue_mean": _safe_mean(getattr(env, "uav_queue", 0.0)),
        "sat_queue_mean": _safe_mean(getattr(env, "sat_queue", 0.0)),
        "gu_queue_max": _safe_max(getattr(env, "gu_queue", 0.0)),
        "uav_queue_max": _safe_max(getattr(env, "uav_queue", 0.0)),
        "sat_queue_max": _safe_max(getattr(env, "sat_queue", 0.0)),
        "gu_drop_sum": _safe_sum(getattr(env, "gu_drop", 0.0)),
        "uav_drop_sum": _safe_sum(getattr(env, "uav_drop", 0.0)),
        "sat_processed_sum": _safe_sum(sat_processed) if sat_processed is not None else 0.0,
        "sat_incoming_sum": _safe_sum(sat_incoming) if sat_incoming is not None else 0.0,
        "energy_mean": _safe_mean(getattr(env, "uav_energy", 0.0)),
        "reward_parts": dict(parts),
    }


def _get_state_batch(env) -> np.ndarray:
    if hasattr(env, "get_global_state_batch"):
        states = env.get_global_state_batch()
        return np.asarray(states, dtype=np.float32)
    return np.expand_dims(np.asarray(env.get_global_state(), dtype=np.float32), axis=0)


def _set_module_requires_grad(module, enabled: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = bool(enabled)


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


def _normalize_exec_source(raw: str | None) -> str:
    src = str("policy" if raw is None else raw).strip().lower()
    allowed = {"policy", "teacher", "heuristic", "zero"}
    if src not in allowed:
        raise ValueError(f"Invalid exec source '{raw}'. Allowed: {sorted(allowed)}")
    return src


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
    if source == "heuristic" and heuristic_values is not None:
        return np.asarray(heuristic_values, dtype=np.float32)
    return np.zeros(shape, dtype=np.float32)


def train(
    env,
    cfg,
    log_dir: str,
    total_updates: int = 50,
    init_actor_path: str | None = None,
    init_critic_path: str | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_envs = int(getattr(env, "num_envs", 1))

    # Determine observation and state dimensions
    obs_sample_raw, _ = env.reset()
    if num_envs > 1:
        if not isinstance(obs_sample_raw, list) or len(obs_sample_raw) != num_envs:
            raise ValueError("Vectorized env reset() must return a list of per-env observations.")
        obs_sample_env = obs_sample_raw
    else:
        obs_sample_env = [obs_sample_raw]

    num_agents = len(env.agents)
    obs_dim = batch_flatten_obs(list(obs_sample_env[0].values()), cfg).shape[1]
    global_state = _get_state_batch(env)[0]
    state_dim = global_state.shape[0]

    actor = ActorNet(obs_dim, cfg).to(device)
    critic = CriticNet(state_dim, cfg).to(device)
    if init_actor_path:
        try:
            state = torch.load(init_actor_path, map_location=device)
            actor.load_state_dict(state, strict=False)
            print(f"Loaded actor init from {init_actor_path} (strict=False)")
        except Exception as exc:
            print(f"Warning: failed to load actor init from {init_actor_path}: {exc}")
    if init_critic_path:
        try:
            state = torch.load(init_critic_path, map_location=device)
            critic.load_state_dict(state, strict=False)
            print(f"Loaded critic init from {init_critic_path} (strict=False)")
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

    train_shared_backbone = bool(getattr(cfg, "train_shared_backbone", True))
    _set_module_requires_grad(actor.mu_head, train_heads["accel"])
    actor.log_std.requires_grad = train_heads["accel"]
    _set_module_requires_grad(actor.bw_head, train_heads["bw"])
    if actor.bw_log_std is not None:
        actor.bw_log_std.requires_grad = train_heads["bw"]
    _set_module_requires_grad(actor.sat_head, train_heads["sat"])
    if actor.sat_log_std is not None:
        actor.sat_log_std.requires_grad = train_heads["sat"]
    if not train_shared_backbone:
        _set_module_requires_grad(actor.obs_norm, False)
        _set_module_requires_grad(actor.fc1, False)
        _set_module_requires_grad(actor.fc2, False)

    actor_params = [p for p in actor.parameters() if p.requires_grad]
    if not actor_params:
        raise ValueError("Actor has no trainable parameters after stage freeze settings.")

    actor_optim = torch.optim.Adam(actor_params, lr=cfg.actor_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    lr_decay_enabled = bool(getattr(cfg, "lr_decay_enabled", False))
    lr_final_factor = float(getattr(cfg, "lr_final_factor", 0.1) or 0.1)
    lr_final_factor = float(np.clip(lr_final_factor, 0.0, 1.0))
    actor_sched = None
    critic_sched = None
    if lr_decay_enabled and total_updates > 1 and lr_final_factor < 1.0:
        def _linear_lr(step_idx: int) -> float:
            progress = min(max(step_idx, 0), total_updates - 1) / max(total_updates - 1, 1)
            return 1.0 - (1.0 - lr_final_factor) * progress

        actor_sched = torch.optim.lr_scheduler.LambdaLR(actor_optim, lr_lambda=_linear_lr)
        critic_sched = torch.optim.lr_scheduler.LambdaLR(critic_optim, lr_lambda=_linear_lr)

    from ..utils.logging import MetricLogger
    from ..utils.progress import Progress

    metric_fields = [
        "episode_reward",
        "policy_loss",
        "value_loss",
        "entropy",
        "reward_rms_sigma",
        "reward_clip_frac",
        "approx_kl",
        "clip_frac",
        "adv_raw_mean",
        "adv_raw_std",
        "adv_preclip_mean",
        "adv_preclip_std",
        "adv_postclip_mean",
        "adv_postclip_std",
        "adv_clip_frac",
        "adv_norm_mean",
        "adv_norm_std",
        "imitation_loss",
        "imitation_coef",
        "actor_lr",
        "critic_lr",
        "log_std_mean",
        "action_std_mean",
        "r_service_ratio",
        "r_drop_ratio",
        "r_queue_pen",
        "r_queue_topk",
        "r_centroid",
        "centroid_dist_mean",
        "r_bw_align",
        "r_sat_score",
        "r_assoc_ratio",
        "r_queue_delta",
        "r_dist",
        "r_dist_delta",
        "r_energy",
        "r_collision_penalty",
        "r_battery_penalty",
        "r_fail_penalty",
        "r_term_service",
        "r_term_drop",
        "r_term_queue",
        "r_term_topk",
        "r_term_assoc",
        "r_term_q_delta",
        "r_term_dist",
        "r_term_dist_delta",
        "r_term_centroid",
        "r_term_bw_align",
        "r_term_sat_score",
        "r_term_energy",
        "r_term_accel",
        "reward_raw",
        "arrival_sum",
        "outflow_sum",
        "service_norm",
        "drop_norm",
        "drop_sum",
        "queue_total",
        "queue_total_active",
        "queue_total_active_p95",
        "queue_total_active_p99",
        "queue_total_active_max",
        "q_norm_active",
        "q_norm_active_p95",
        "q_norm_active_p99",
        "q_norm_active_max",
        "q_norm_active_nonzero_rate",
        "q_norm_tail_hit_rate",
        "prev_q_norm_active",
        "q_norm_delta",
        "q_norm_tail_q0",
        "q_norm_tail_excess",
        "queue_weight",
        "q_delta_weight",
        "crash_weight",
        "centroid_transfer_ratio",
        "collision_rate",
        "avoidance_eta_eff",
        "avoidance_eta_exec",
        "avoidance_collision_rate_ema",
        "avoidance_prev_episode_collision_rate",
        "arrival_rate_eff",
        "gu_queue_mean",
        "uav_queue_mean",
        "sat_queue_mean",
        "gu_queue_max",
        "uav_queue_max",
        "sat_queue_max",
        "gu_drop_sum",
        "uav_drop_sum",
        "sat_processed_sum",
        "sat_incoming_sum",
        "energy_mean",
        "update_time_sec",
        "rollout_time_sec",
        "optim_time_sec",
        "env_steps",
        "env_steps_per_sec",
        "update_steps_per_sec",
        "total_env_steps",
        "total_time_sec",
    ]
    logger = MetricLogger(log_dir, fieldnames=metric_fields)
    progress = Progress(total_updates, desc="Train")
    training_start = time.perf_counter()
    total_env_steps = 0
    best_ma = -float("inf")
    no_improve = 0
    reward_history = []
    reward_rms = RunningMeanStd() if getattr(cfg, "reward_norm_enabled", False) else None
    imitation_coef_start = float(getattr(cfg, "imitation_coef", 0.0) or 0.0)
    imitation_coef_final_cfg = getattr(cfg, "imitation_coef_final", None)
    imitation_coef_final = (
        imitation_coef_start
        if imitation_coef_final_cfg is None
        else float(imitation_coef_final_cfg)
    )
    imitation_coef_decay_updates = int(getattr(cfg, "imitation_coef_decay_updates", 0) or 0)

    def _imitation_coef_at(update_idx: int) -> float:
        if imitation_coef_decay_updates <= 0:
            return imitation_coef_start
        progress = min(
            1.0,
            float(max(update_idx, 0)) / float(max(imitation_coef_decay_updates - 1, 1)),
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
    imitation_coef_curr = _imitation_coef_at(0)
    imitation_enabled_curr = imitation_enabled and imitation_coef_curr > 0.0
    bw_scale = float(getattr(cfg, "bw_logit_scale", 1.0) or 1.0)
    sat_scale = float(getattr(cfg, "sat_logit_scale", 1.0) or 1.0)
    exec_accel_source = _normalize_exec_source(getattr(cfg, "exec_accel_source", "policy"))
    exec_bw_source = _normalize_exec_source(getattr(cfg, "exec_bw_source", "policy"))
    exec_sat_source = _normalize_exec_source(getattr(cfg, "exec_sat_source", "policy"))
    teacher_deterministic = bool(getattr(cfg, "exec_teacher_deterministic", True))
    need_teacher_exec = "teacher" in {exec_accel_source, exec_bw_source, exec_sat_source}
    need_heuristic_exec = "heuristic" in {exec_accel_source, exec_bw_source, exec_sat_source}

    teacher_actor = None
    if need_teacher_exec:
        teacher_path = getattr(cfg, "exec_teacher_actor_path", None) or init_actor_path
        if not teacher_path:
            raise ValueError(
                "Execution override uses 'teacher' but no teacher checkpoint is provided. "
                "Set exec_teacher_actor_path or pass --init_actor."
            )
        teacher_actor = ActorNet(obs_dim, cfg).to(device)
        state = torch.load(teacher_path, map_location=device)
        teacher_actor.load_state_dict(state, strict=False)
        teacher_actor.eval()
        for p in teacher_actor.parameters():
            p.requires_grad = False
    print(
        "Train heads: "
        f"accel={train_heads['accel']}, bw={train_heads['bw']}, sat={train_heads['sat']} | "
        "Exec sources: "
        f"accel={exec_accel_source}, bw={exec_bw_source}, sat={exec_sat_source}"
    )

    obs_raw, _ = env.reset()
    if num_envs > 1:
        if not isinstance(obs_raw, list) or len(obs_raw) != num_envs:
            raise ValueError("Vectorized env reset() must return a list of per-env observations.")
        obs_env = obs_raw
    else:
        obs_env = [obs_raw]

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

    for update in range(total_updates):
        imitation_coef_curr = _imitation_coef_at(update)
        imitation_enabled_curr = imitation_enabled and imitation_coef_curr > 0.0
        update_start = time.perf_counter()
        buffers = [RolloutBuffer(capacity=cfg.buffer_size) for _ in range(num_envs)]
        ep_reward = 0.0
        steps_count = 0
        gu_queue_sum = 0.0
        uav_queue_sum = 0.0
        sat_queue_sum = 0.0
        gu_queue_max = 0.0
        uav_queue_max = 0.0
        sat_queue_max = 0.0
        gu_drop_sum = 0.0
        uav_drop_sum = 0.0
        sat_processed_sum = 0.0
        sat_incoming_sum = 0.0
        energy_mean_sum = 0.0
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
        r_collision_penalty_sum = 0.0
        r_battery_penalty_sum = 0.0
        r_fail_penalty_sum = 0.0
        r_term_service_sum = 0.0
        r_term_drop_sum = 0.0
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
        imitation_loss_sum = 0.0
        reward_raw_sum = 0.0
        arrival_sum_sum = 0.0
        outflow_sum_sum = 0.0
        service_norm_sum = 0.0
        drop_norm_sum = 0.0
        drop_sum_total = 0.0
        queue_total_sum = 0.0
        queue_total_active_sum = 0.0
        q_norm_active_sum = 0.0
        prev_q_norm_active_sum = 0.0
        q_norm_delta_sum = 0.0
        q_norm_tail_q0_sum = 0.0
        q_norm_tail_excess_sum = 0.0
        queue_weight_sum = 0.0
        q_delta_weight_sum = 0.0
        crash_weight_sum = 0.0
        centroid_transfer_ratio_sum = 0.0
        collision_event_sum = 0.0
        avoidance_eta_eff_sum = 0.0
        avoidance_eta_exec_sum = 0.0
        avoidance_collision_rate_ema_sum = 0.0
        avoidance_prev_episode_collision_rate_sum = 0.0
        arrival_rate_eff_sum = 0.0
        q_norm_active_values: List[float] = []
        queue_total_active_values: List[float] = []
        q_norm_active_nonzero_count = 0
        q_norm_tail_hit_count = 0
        q_norm_active_max = 0.0
        queue_total_active_max = 0.0

        rollout_start = time.perf_counter()
        for step in range(cfg.buffer_size):
            per_env_obs_lists = [list(obs_e.values()) for obs_e in obs_env]
            per_env_obs_batch = [batch_flatten_obs(obs_list, cfg) for obs_list in per_env_obs_lists]
            obs_batch = np.concatenate(per_env_obs_batch, axis=0)
            if not np.isfinite(obs_batch).all():
                print(f"NaN/Inf detected in obs_batch at update={update}, step={step}")
                raise ValueError("obs_batch contains NaN/Inf")
            obs_tensor = torch.from_numpy(obs_batch).to(device)

            state_batch_np = _get_state_batch(env)
            if state_batch_np.shape[0] != num_envs:
                raise ValueError(
                    f"Expected {num_envs} states from env, got {state_batch_np.shape[0]}"
                )
            with torch.inference_mode():
                policy_out = actor.act(obs_tensor)
                value = critic(torch.from_numpy(state_batch_np).to(device))
            if not torch.isfinite(policy_out.dist_out["mu"]).all():
                print(f"NaN/Inf detected in actor mu at update={update}, step={step}")
                raise ValueError("actor mu contains NaN/Inf")

            accel_actions = policy_out.accel.cpu().numpy()
            bw_logits_all = policy_out.bw_logits.cpu().numpy() if policy_out.bw_logits is not None else None
            sat_logits_all = policy_out.sat_logits.cpu().numpy() if policy_out.sat_logits is not None else None
            teacher_accel_all = None
            teacher_bw_all = None
            teacher_sat_all = None
            if teacher_actor is not None:
                with torch.inference_mode():
                    teacher_out = teacher_actor.act(obs_tensor, deterministic=teacher_deterministic)
                teacher_accel_all = teacher_out.accel.cpu().numpy()
                teacher_bw_all = teacher_out.bw_logits.cpu().numpy() if teacher_out.bw_logits is not None else None
                teacher_sat_all = teacher_out.sat_logits.cpu().numpy() if teacher_out.sat_logits is not None else None

            action_dicts = []
            accel_cmd_list: List[np.ndarray] = []
            bw_exec_list: List[np.ndarray | None] = []
            sat_exec_list: List[np.ndarray | None] = []
            for env_idx in range(num_envs):
                sl = slice(env_idx * num_agents, (env_idx + 1) * num_agents)
                obs_list = per_env_obs_lists[env_idx]
                accel_policy_env = accel_actions[sl]
                bw_policy_env = bw_logits_all[sl] if bw_logits_all is not None else None
                sat_policy_env = sat_logits_all[sl] if sat_logits_all is not None else None
                accel_teacher_env = teacher_accel_all[sl] if teacher_accel_all is not None else None
                bw_teacher_env = teacher_bw_all[sl] if teacher_bw_all is not None else None
                sat_teacher_env = teacher_sat_all[sl] if teacher_sat_all is not None else None
                heur_accel = None
                heur_bw = None
                heur_sat = None
                if need_heuristic_exec:
                    heur_accel, heur_bw, heur_sat = queue_aware_policy(obs_list, cfg)

                accel_cmd = _select_exec_values(
                    exec_accel_source,
                    accel_policy_env,
                    accel_teacher_env,
                    heur_accel,
                    (len(obs_list), 2),
                )
                accel_cmd = np.clip(accel_cmd, -1.0, 1.0).astype(np.float32, copy=False)

                users_mask = np.stack([o["users_mask"] for o in obs_list], axis=0)
                sats_mask = np.stack([o["sats_mask"] for o in obs_list], axis=0)
                if cfg.doppler_enabled and use_heuristic_mask:
                    nu_norm = np.stack([o["sats"][:, 6] for o in obs_list], axis=0)
                    doppler_ok = (np.abs(nu_norm) <= 1.0).astype(np.float32)
                    sats_mask = sats_mask * doppler_ok

                bw_exec = None
                if cfg.enable_bw_action:
                    bw_raw = _select_exec_values(
                        exec_bw_source,
                        bw_policy_env,
                        bw_teacher_env,
                        heur_bw,
                        (len(obs_list), cfg.users_obs_max),
                    )
                    bw_exec = bw_raw * users_mask

                sat_exec = None
                if not cfg.fixed_satellite_strategy:
                    sat_raw = _select_exec_values(
                        exec_sat_source,
                        sat_policy_env,
                        sat_teacher_env,
                        heur_sat,
                        (len(obs_list), cfg.sats_obs_max),
                    )
                    sat_exec = sat_raw * sats_mask

                action_dict = assemble_actions(
                    cfg,
                    env.agents,
                    accel_cmd,
                    bw_logits=bw_exec,
                    sat_logits=sat_exec,
                )
                action_dicts.append(action_dict)
                accel_cmd_list.append(accel_cmd)
                bw_exec_list.append(bw_exec)
                sat_exec_list.append(sat_exec)

            if num_envs > 1:
                next_obs_env, rewards_env, terms_env, truncs_env, _ = env.step(action_dicts, auto_reset=True)
                step_stats = getattr(env, "last_step_stats", None)
                if not isinstance(step_stats, list) or len(step_stats) != num_envs:
                    raise ValueError("Vectorized env must expose last_step_stats after step().")
            else:
                next_obs, rewards, terms, truncs, _ = env.step(action_dicts[0])
                step_stats = [_single_env_step_stats(env)]
                done_scalar = list(terms.values())[0] or list(truncs.values())[0]
                if done_scalar:
                    next_obs, _ = env.reset()
                next_obs_env = [next_obs]
                rewards_env = [rewards]
                terms_env = [terms]
                truncs_env = [truncs]

            value_np = value.detach().cpu().numpy().reshape(num_envs)

            action_vec_exec_env = []
            for env_idx in range(num_envs):
                stats = step_stats[env_idx] if step_stats[env_idx] is not None else {}
                fallback_accel = accel_cmd_list[env_idx] * cfg.a_max
                accel_exec = np.asarray(stats.get("last_exec_accel", fallback_accel), dtype=np.float32)
                accel_exec_norm = accel_exec / max(cfg.a_max, 1e-6)
                accel_exec_norm = np.clip(accel_exec_norm, -1.0, 1.0).astype(np.float32, copy=False)
                exec_parts = [accel_exec_norm]
                if cfg.enable_bw_action and bw_exec_list[env_idx] is not None:
                    exec_parts.append(bw_exec_list[env_idx])
                if not cfg.fixed_satellite_strategy and sat_exec_list[env_idx] is not None:
                    exec_parts.append(sat_exec_list[env_idx])
                action_vec_exec_env.append(np.concatenate(exec_parts, axis=1).astype(np.float32, copy=False))

            action_vec_exec = np.concatenate(action_vec_exec_env, axis=0).astype(np.float32, copy=False)
            with torch.inference_mode():
                action_vec_exec_t = torch.from_numpy(action_vec_exec).to(device)
                logprob_parts, _ = actor.evaluate_actions_parts(obs_tensor, action_vec_exec_t, out=policy_out.dist_out)
                logprobs_all = _sum_selected_parts(logprob_parts, train_heads).detach().cpu().numpy()

            for env_idx in range(num_envs):
                sl = slice(env_idx * num_agents, (env_idx + 1) * num_agents)
                rewards = rewards_env[env_idx]
                terms = terms_env[env_idx]
                truncs = truncs_env[env_idx]

                reward_scalar = list(rewards.values())[0]
                done_scalar = list(terms.values())[0] or list(truncs.values())[0]
                ep_reward += reward_scalar
                steps_count += 1
                total_env_steps += 1

                stats = step_stats[env_idx] if step_stats[env_idx] is not None else {}
                gu_queue_sum += float(stats.get("gu_queue_mean", 0.0))
                uav_queue_sum += float(stats.get("uav_queue_mean", 0.0))
                sat_queue_sum += float(stats.get("sat_queue_mean", 0.0))
                gu_queue_max = max(gu_queue_max, float(stats.get("gu_queue_max", 0.0)))
                uav_queue_max = max(uav_queue_max, float(stats.get("uav_queue_max", 0.0)))
                sat_queue_max = max(sat_queue_max, float(stats.get("sat_queue_max", 0.0)))
                gu_drop_sum += float(stats.get("gu_drop_sum", 0.0))
                uav_drop_sum += float(stats.get("uav_drop_sum", 0.0))
                sat_processed_sum += float(stats.get("sat_processed_sum", 0.0))
                sat_incoming_sum += float(stats.get("sat_incoming_sum", 0.0))
                if cfg.energy_enabled:
                    energy_mean_sum += float(stats.get("energy_mean", 0.0))

                parts = stats.get("reward_parts", None)
                if parts:
                    r_service_ratio_sum += float(parts.get("service_ratio", 0.0))
                    r_drop_ratio_sum += float(parts.get("drop_ratio", 0.0))
                    arrival_sum_sum += float(parts.get("arrival_sum", 0.0))
                    outflow_sum_sum += float(parts.get("outflow_sum", 0.0))
                    service_norm_sum += float(parts.get("service_norm", 0.0))
                    drop_norm_sum += float(parts.get("drop_norm", 0.0))
                    drop_sum_total += float(parts.get("drop_sum", 0.0))
                    queue_total_sum += float(parts.get("queue_total", 0.0))
                    queue_total_active_step = float(parts.get("queue_total_active", 0.0))
                    q_norm_active_step = float(parts.get("q_norm_active", 0.0))
                    q_norm_tail_q0_step = max(float(parts.get("q_norm_tail_q0", 0.0)), 0.0)
                    queue_total_active_sum += queue_total_active_step
                    q_norm_active_sum += q_norm_active_step
                    prev_q_norm_active_sum += float(parts.get("prev_q_norm_active", 0.0))
                    q_norm_delta_sum += float(parts.get("q_norm_delta", 0.0))
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
                    r_collision_penalty_sum += float(parts.get("collision_penalty", 0.0))
                    r_battery_penalty_sum += float(parts.get("battery_penalty", 0.0))
                    r_fail_penalty_sum += float(parts.get("fail_penalty", 0.0))
                    r_term_service_sum += float(parts.get("term_service", 0.0))
                    r_term_drop_sum += float(parts.get("term_drop", 0.0))
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
                    reward_raw_sum += float(parts.get("reward_raw", 0.0))

                buffers[env_idx].add(
                    per_env_obs_batch[env_idx],
                    action_vec_exec_env[env_idx],
                    logprobs_all[sl],
                    reward_scalar,
                    float(value_np[env_idx]),
                    done_scalar,
                    state_batch_np[env_idx],
                    _build_imitation_target(per_env_obs_lists[env_idx]),
                )

            obs_env = list(next_obs_env)
        rollout_time = time.perf_counter() - rollout_start

        # Prepare batches
        buffer_data = [buf.as_arrays() for buf in buffers]
        all_rewards = np.concatenate([data[3] for data in buffer_data], axis=0)
        if getattr(cfg, "reward_norm_enabled", False) and reward_rms is not None:
            reward_rms.update(all_rewards)

        obs_arr_list = []
        act_arr_list = []
        logp_arr_list = []
        state_arr_list = []
        imitation_arr_list = []
        adv_list = []
        rets_list = []
        clip_val = float(getattr(cfg, "reward_norm_clip", 0.0) or 0.0)
        reward_clip_hits = 0
        reward_clip_total = 0
        for obs_arr_e, act_arr_e, logp_arr_e, rewards_e, values_e, dones_e, state_arr_e, imitation_arr_e in buffer_data:
            rewards_proc = rewards_e
            if getattr(cfg, "reward_norm_enabled", False) and reward_rms is not None:
                rewards_proc = (rewards_proc - reward_rms.mean) / (np.sqrt(reward_rms.var) + 1e-8)
                if clip_val > 0:
                    reward_clip_hits += int(np.count_nonzero(np.abs(rewards_proc) > clip_val))
                    reward_clip_total += int(rewards_proc.size)
                    rewards_proc = np.clip(rewards_proc, -clip_val, clip_val)
            adv_e, rets_e = compute_gae(rewards_proc, values_e, dones_e, cfg.gamma, cfg.gae_lambda)
            obs_arr_list.append(obs_arr_e)
            act_arr_list.append(act_arr_e)
            logp_arr_list.append(logp_arr_e)
            state_arr_list.append(state_arr_e)
            imitation_arr_list.append(imitation_arr_e)
            adv_list.append(adv_e)
            rets_list.append(rets_e)

        obs_arr = np.concatenate(obs_arr_list, axis=0)
        act_arr = np.concatenate(act_arr_list, axis=0)
        logp_arr = np.concatenate(logp_arr_list, axis=0)
        state_arr = np.concatenate(state_arr_list, axis=0)
        imitation_arr = np.concatenate(imitation_arr_list, axis=0)
        adv = np.concatenate(adv_list, axis=0)
        rets = np.concatenate(rets_list, axis=0)
        adv_raw_mean = float(np.mean(adv))
        adv_raw_std = float(np.std(adv))
        adv_preclip = (adv - adv_raw_mean) / (adv_raw_std + 1e-8)
        adv_preclip_mean = float(np.mean(adv_preclip))
        adv_preclip_std = float(np.std(adv_preclip))
        adv_clip = float(getattr(cfg, "adv_clip", 5.0) or 0.0)
        if adv_clip > 0.0:
            adv = np.clip(adv_preclip, -adv_clip, adv_clip)
            adv_clip_frac = float(np.count_nonzero(np.abs(adv_preclip) > adv_clip)) / float(adv_preclip.size)
        else:
            adv = adv_preclip
            adv_clip_frac = 0.0
        adv_postclip_mean = float(np.mean(adv))
        adv_postclip_std = float(np.std(adv))
        adv_norm_mean = adv_postclip_mean
        adv_norm_std = adv_postclip_std

        T, N, _ = obs_arr.shape
        obs_flat = obs_arr.reshape(T * N, -1)
        act_flat = act_arr.reshape(T * N, -1)
        logp_flat = logp_arr.reshape(T * N)
        imitation_flat = imitation_arr.reshape(T * N, -1)
        adv_flat = np.repeat(adv, N)

        # Convert to torch
        obs_flat_t = torch.from_numpy(obs_flat).to(device)
        act_flat_t = torch.from_numpy(act_flat).to(device)
        logp_flat_t = torch.from_numpy(logp_flat).to(device)
        adv_flat_t = torch.from_numpy(adv_flat).to(device)
        state_t = torch.from_numpy(state_arr).to(device)
        ret_t = torch.from_numpy(rets).to(device)
        imitation_flat_t = torch.from_numpy(imitation_flat).to(device)
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
        approx_kls = []
        clip_fracs = []

        optim_start = time.perf_counter()
        stop_early = False
        for _ in range(cfg.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]
                if not torch.isfinite(obs_flat_t[mb_idx]).all():
                    print(f"NaN/Inf detected in obs minibatch at update={update}")
                    raise ValueError("obs minibatch contains NaN/Inf")
                if not torch.isfinite(act_flat_t[mb_idx]).all():
                    print(f"NaN/Inf detected in act minibatch at update={update}")
                    raise ValueError("act minibatch contains NaN/Inf")
                out = actor.forward(obs_flat_t[mb_idx])
                logprob_parts, entropy_parts = actor.evaluate_actions_parts(obs_flat_t[mb_idx], act_flat_t[mb_idx], out=out)
                new_logp = _sum_selected_parts(logprob_parts, train_heads)
                entropy = _sum_selected_parts(entropy_parts, train_heads)
                if not torch.isfinite(new_logp).all():
                    print(f"NaN/Inf detected in new_logp at update={update}")
                    raise ValueError("new_logp contains NaN/Inf")

                log_ratio = new_logp - logp_flat_t[mb_idx]
                log_ratio = torch.clamp(log_ratio, -8.0, 8.0)
                ratio = torch.exp(log_ratio)
                surr1 = ratio * adv_flat_t[mb_idx]
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv_flat_t[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                approx_kl = (logp_flat_t[mb_idx] - new_logp).mean()
                clip_frac = ((ratio - 1.0).abs() > cfg.clip_ratio).float().mean()
                kl_coef = float(getattr(cfg, "kl_coef", 0.0) or 0.0)
                if kl_coef > 0:
                    policy_loss = policy_loss + kl_coef * approx_kl
                if getattr(cfg, "kl_stop", False):
                    target_kl = float(getattr(cfg, "target_kl", 0.0) or 0.0)
                    if target_kl > 0 and float(approx_kl.item()) > target_kl:
                        stop_early = True

                imitation_loss = torch.tensor(0.0, device=device)
                if imitation_enabled_curr and imitation_mask is not None and imitation_mask_sum > 0:
                    pred_parts = [torch.tanh(out["mu"])]
                    if cfg.enable_bw_action:
                        pred_parts.append(torch.tanh(out["bw_mu"]) * bw_scale)
                    if not cfg.fixed_satellite_strategy:
                        pred_parts.append(torch.tanh(out["sat_mu"]) * sat_scale)
                    pred_action = torch.cat(pred_parts, dim=-1)
                    target_action = imitation_flat_t[mb_idx]
                    diff = (pred_action - target_action) * imitation_mask
                    imitation_loss = (diff.pow(2).sum(-1) / (imitation_mask_sum + 1e-9)).mean()

                actor_optim.zero_grad()
                (policy_loss + imitation_coef_curr * imitation_loss - cfg.entropy_coef * entropy.mean()).backward()
                for name, param in actor.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        print(f"NaN/Inf detected in actor grad {name} at update={update}")
                        raise ValueError("actor gradient contains NaN/Inf")
                torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                actor_optim.step()
                for name, param in actor.named_parameters():
                    if not torch.isfinite(param).all():
                        print(f"NaN/Inf detected in actor param {name} after step at update={update}")
                        raise ValueError("actor parameters contain NaN/Inf after step")

                policy_losses.append(policy_loss.item())
                entropies.append(entropy.mean().item())
                approx_kls.append(float(approx_kl.item()))
                clip_fracs.append(float(clip_frac.item()))
                imitation_loss_sum += float(imitation_loss.item())
                if stop_early:
                    break
            if stop_early:
                break

            # critic update (full batch for simplicity)
            value_pred = critic(state_t)
            value_loss = F.mse_loss(value_pred, ret_t)
            critic_optim.zero_grad()
            (cfg.value_coef * value_loss).backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
            critic_optim.step()
            value_losses.append(value_loss.item())
        optim_time = time.perf_counter() - optim_start

        update_time = time.perf_counter() - update_start
        steps_count = max(1, steps_count)
        episode_reward = ep_reward / steps_count
        reward_rms_sigma = (
            float(np.sqrt(reward_rms.var))
            if getattr(cfg, "reward_norm_enabled", False) and reward_rms is not None
            else 0.0
        )
        reward_clip_frac = (
            float(reward_clip_hits) / float(reward_clip_total)
            if reward_clip_total > 0
            else 0.0
        )
        log_std_terms: List[np.ndarray] = []
        if train_heads["accel"]:
            log_std_terms.append(torch.clamp(actor.log_std.detach(), -5.0, 2.0).cpu().numpy().reshape(-1))
        if train_heads["bw"] and actor.bw_log_std is not None:
            log_std_terms.append(torch.clamp(actor.bw_log_std.detach(), -5.0, 2.0).cpu().numpy().reshape(-1))
        if train_heads["sat"] and actor.sat_log_std is not None:
            log_std_terms.append(torch.clamp(actor.sat_log_std.detach(), -5.0, 2.0).cpu().numpy().reshape(-1))
        if log_std_terms:
            log_std_vec = np.concatenate(log_std_terms, axis=0)
        else:
            log_std_vec = np.zeros((1,), dtype=np.float32)
        action_std_vec = np.exp(log_std_vec)
        metrics = {
            "episode_reward": episode_reward,
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "reward_rms_sigma": reward_rms_sigma,
            "reward_clip_frac": reward_clip_frac,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clip_frac": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            "adv_raw_mean": adv_raw_mean,
            "adv_raw_std": adv_raw_std,
            "adv_preclip_mean": adv_preclip_mean,
            "adv_preclip_std": adv_preclip_std,
            "adv_postclip_mean": adv_postclip_mean,
            "adv_postclip_std": adv_postclip_std,
            "adv_clip_frac": adv_clip_frac,
            "adv_norm_mean": adv_norm_mean,
            "adv_norm_std": adv_norm_std,
            "imitation_loss": imitation_loss_sum / max(1, len(policy_losses)),
            "imitation_coef": imitation_coef_curr,
            "actor_lr": float(actor_optim.param_groups[0]["lr"]),
            "critic_lr": float(critic_optim.param_groups[0]["lr"]),
            "log_std_mean": float(np.mean(log_std_vec)),
            "action_std_mean": float(np.mean(action_std_vec)),
            "r_service_ratio": r_service_ratio_sum / steps_count,
            "r_drop_ratio": r_drop_ratio_sum / steps_count,
            "r_queue_pen": r_queue_pen_sum / steps_count,
            "r_queue_topk": r_queue_topk_sum / steps_count,
            "r_centroid": r_centroid_sum / steps_count,
            "centroid_dist_mean": centroid_dist_mean_sum / steps_count,
            "r_bw_align": r_bw_align_sum / steps_count,
            "r_sat_score": r_sat_score_sum / steps_count,
            "r_assoc_ratio": r_assoc_ratio_sum / steps_count,
            "r_queue_delta": r_queue_delta_sum / steps_count,
            "r_dist": r_dist_sum / steps_count,
            "r_dist_delta": r_dist_delta_sum / steps_count,
            "r_energy": r_energy_sum / steps_count,
            "r_collision_penalty": r_collision_penalty_sum / steps_count,
            "r_battery_penalty": r_battery_penalty_sum / steps_count,
            "r_fail_penalty": r_fail_penalty_sum / steps_count,
            "r_term_service": r_term_service_sum / steps_count,
            "r_term_drop": r_term_drop_sum / steps_count,
            "r_term_queue": r_term_queue_sum / steps_count,
            "r_term_topk": r_term_topk_sum / steps_count,
            "r_term_assoc": r_term_assoc_sum / steps_count,
            "r_term_q_delta": r_term_q_delta_sum / steps_count,
            "r_term_dist": r_term_dist_sum / steps_count,
            "r_term_dist_delta": r_term_dist_delta_sum / steps_count,
            "r_term_centroid": r_term_centroid_sum / steps_count,
            "r_term_bw_align": r_term_bw_align_sum / steps_count,
            "r_term_sat_score": r_term_sat_score_sum / steps_count,
            "r_term_energy": r_term_energy_sum / steps_count,
            "r_term_accel": r_term_accel_sum / steps_count,
            "reward_raw": reward_raw_sum / steps_count,
            "arrival_sum": arrival_sum_sum / steps_count,
            "outflow_sum": outflow_sum_sum / steps_count,
            "service_norm": service_norm_sum / steps_count,
            "drop_norm": drop_norm_sum / steps_count,
            "drop_sum": drop_sum_total / steps_count,
            "queue_total": queue_total_sum / steps_count,
            "queue_total_active": queue_total_active_sum / steps_count,
            "queue_total_active_p95": (
                float(np.percentile(queue_total_active_values, 95)) if queue_total_active_values else 0.0
            ),
            "queue_total_active_p99": (
                float(np.percentile(queue_total_active_values, 99)) if queue_total_active_values else 0.0
            ),
            "queue_total_active_max": queue_total_active_max,
            "q_norm_active": q_norm_active_sum / steps_count,
            "q_norm_active_p95": float(np.percentile(q_norm_active_values, 95)) if q_norm_active_values else 0.0,
            "q_norm_active_p99": float(np.percentile(q_norm_active_values, 99)) if q_norm_active_values else 0.0,
            "q_norm_active_max": q_norm_active_max,
            "q_norm_active_nonzero_rate": float(q_norm_active_nonzero_count) / float(steps_count),
            "q_norm_tail_hit_rate": float(q_norm_tail_hit_count) / float(steps_count),
            "prev_q_norm_active": prev_q_norm_active_sum / steps_count,
            "q_norm_delta": q_norm_delta_sum / steps_count,
            "q_norm_tail_q0": q_norm_tail_q0_sum / steps_count,
            "q_norm_tail_excess": q_norm_tail_excess_sum / steps_count,
            "queue_weight": queue_weight_sum / steps_count,
            "q_delta_weight": q_delta_weight_sum / steps_count,
            "crash_weight": crash_weight_sum / steps_count,
            "centroid_transfer_ratio": centroid_transfer_ratio_sum / steps_count,
            "collision_rate": collision_event_sum / steps_count,
            "avoidance_eta_eff": avoidance_eta_eff_sum / steps_count,
            "avoidance_eta_exec": avoidance_eta_exec_sum / steps_count,
            "avoidance_collision_rate_ema": avoidance_collision_rate_ema_sum / steps_count,
            "avoidance_prev_episode_collision_rate": (
                avoidance_prev_episode_collision_rate_sum / steps_count
            ),
            "arrival_rate_eff": arrival_rate_eff_sum / steps_count,
            "gu_queue_mean": gu_queue_sum / steps_count,
            "uav_queue_mean": uav_queue_sum / steps_count,
            "sat_queue_mean": sat_queue_sum / steps_count,
            "gu_queue_max": gu_queue_max,
            "uav_queue_max": uav_queue_max,
            "sat_queue_max": sat_queue_max,
            "gu_drop_sum": gu_drop_sum,
            "uav_drop_sum": uav_drop_sum,
            "sat_processed_sum": sat_processed_sum,
            "sat_incoming_sum": sat_incoming_sum,
            "energy_mean": (energy_mean_sum / steps_count) if cfg.energy_enabled else 0.0,
            "update_time_sec": update_time,
            "rollout_time_sec": rollout_time,
            "optim_time_sec": optim_time,
            "env_steps": float(steps_count),
            "env_steps_per_sec": steps_count / max(1e-9, rollout_time),
            "update_steps_per_sec": steps_count / max(1e-9, update_time),
            "total_env_steps": float(total_env_steps),
            "total_time_sec": time.perf_counter() - training_start,
        }

        logger.log(
            update,
            metrics,
        )
        if actor_sched is not None:
            actor_sched.step()
        if critic_sched is not None:
            critic_sched.step()
        progress.update(update + 1)

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

    # Save checkpoints
    os.makedirs(log_dir, exist_ok=True)
    torch.save(actor.state_dict(), os.path.join(log_dir, "actor.pt"))
    torch.save(critic.state_dict(), os.path.join(log_dir, "critic.pt"))

    progress.close()
    logger.close()
    return actor, critic
