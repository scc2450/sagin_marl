from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from scripts.diagnose_frontend_critic import (
    _compute_mc_returns,
    _explained_variance,
    _get_state_batch,
    _pearson,
    _spearman,
    _summary,
    _write_csv,
    _write_json,
)
from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.rl.action_assembler import assemble_actions
from sagin_marl.rl.critic import CriticNet, VALUE_HEAD_NAMES
from sagin_marl.rl.mappo import (
    _configure_actor_trainability,
    _compute_per_head_gae_targets,
    _compute_train_reward_adjustment,
    _normalize_advantages,
    _single_env_step_stats,
    _stack_value_head_dict,
    _sum_selected_parts,
    compute_gae,
)
from sagin_marl.rl.baselines import queue_aware_bw_policy
from sagin_marl.rl.policy import ActorNet, batch_flatten_obs
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env

DEFAULT_ROLLOUT_BUFFERS = 4
DEFAULT_SEED_BASE = 49000
DEFAULT_REPLAY_SEED_BASE = 50000
DIRICHLET_EPS = 1e-8


@dataclass
class RunSpec:
    name: str
    run_dir: Path
    actor_ckpt: str = "actor.pt"
    critic_ckpt: str = "critic.pt"
    train_state_ckpt: str | None = "train_state.pt"
    update_idx: int | None = None


@dataclass
class RunBundle:
    name: str
    run_dir: Path
    cfg: Any
    actor: ActorNet
    critic: CriticNet
    train_state_path: Path | None
    update_idx: int
    planned_total_updates: int
    multihead_value: bool


def _load_actor(cfg, obs_dim: int, checkpoint_path: Path) -> ActorNet:
    actor = ActorNet(obs_dim, cfg).to(torch.device("cpu"))
    load_checkpoint_forgiving(actor, str(checkpoint_path), map_location="cpu", strict=True)
    actor.eval()
    return actor


def _load_critic(cfg, obs_dim: int, state_dim: int, checkpoint_path: Path) -> CriticNet:
    critic = CriticNet(state_dim, obs_dim, int(cfg.num_uav), cfg).to(torch.device("cpu"))
    load_checkpoint_forgiving(critic, str(checkpoint_path), map_location="cpu", strict=True)
    critic.eval()
    return critic


def _parse_run_spec(raw: str) -> RunSpec:
    if "=" not in raw:
        raise ValueError(
            "Run spec must use NAME=RUN_DIR or NAME=RUN_DIR|actor_ckpt|critic_ckpt|train_state_ckpt|update_idx"
        )
    name, payload = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("Run spec name cannot be empty.")
    parts = [part.strip() for part in payload.split("|")]
    if len(parts) > 5:
        raise ValueError(
            "Run spec supports at most five fields: RUN_DIR|actor_ckpt|critic_ckpt|train_state_ckpt|update_idx"
        )
    run_dir_raw = parts[0] if parts else ""
    if not run_dir_raw:
        raise ValueError("Run spec must include a run directory.")
    actor_ckpt = parts[1] if len(parts) >= 2 and parts[1] else "actor.pt"
    critic_ckpt = parts[2] if len(parts) >= 3 and parts[2] else "critic.pt"
    train_state_raw = parts[3] if len(parts) >= 4 else "train_state.pt"
    if not train_state_raw or str(train_state_raw).strip().lower() in {"none", "-", "null"}:
        train_state_ckpt: str | None = None
    else:
        train_state_ckpt = train_state_raw
    update_idx = int(parts[4]) if len(parts) >= 5 and parts[4] else None
    return RunSpec(
        name=name,
        run_dir=Path(run_dir_raw),
        actor_ckpt=actor_ckpt,
        critic_ckpt=critic_ckpt,
        train_state_ckpt=train_state_ckpt,
        update_idx=update_idx,
    )


def _resolve_config_path(run_dir: Path) -> Path:
    source_path = run_dir / "config_source.yaml"
    if source_path.is_file():
        return source_path
    config_path = run_dir / "config.yaml"
    if config_path.is_file():
        return config_path
    raise FileNotFoundError(f"Could not find config_source.yaml or config.yaml under {run_dir}")


def _load_train_state_payload(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Train state at {path} was not a dict payload.")
    return payload


def _infer_update_idx(run_dir: Path, train_state_payload: dict[str, Any] | None) -> int:
    if train_state_payload is not None:
        update = int(train_state_payload.get("update", 0) or 0)
        if update > 0:
            return update
    metrics_path = run_dir / "metrics.csv"
    if metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            return int(rows[-1].get("step", 0) or 0) + 1
    return 0


def _make_dims(cfg) -> tuple[int, int]:
    env = make_structured_env(cfg, mode="script")
    try:
        obs, _ = env.reset(seed=0)
        obs_dim = int(batch_flatten_obs(list(obs.values()), cfg).shape[1])
        state_dim = int(_get_state_batch(env)[0].shape[0])
        return obs_dim, state_dim
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def _load_bundle(spec: RunSpec) -> RunBundle:
    run_dir = spec.run_dir
    cfg = load_config(str(_resolve_config_path(run_dir)))
    obs_dim, state_dim = _make_dims(cfg)
    actor = _load_actor(cfg, obs_dim, run_dir / spec.actor_ckpt)
    critic = _load_critic(cfg, obs_dim, state_dim, run_dir / spec.critic_ckpt)
    train_state_path = None if spec.train_state_ckpt is None else run_dir / spec.train_state_ckpt
    train_state_payload = _load_train_state_payload(train_state_path)
    update_idx = spec.update_idx if spec.update_idx is not None else _infer_update_idx(run_dir, train_state_payload)
    planned_total_updates = int(
        (train_state_payload or {}).get("planned_total_updates", 0) or getattr(cfg, "total_updates", 0) or update_idx
    )
    return RunBundle(
        name=spec.name,
        run_dir=run_dir,
        cfg=cfg,
        actor=actor,
        critic=critic,
        train_state_path=train_state_path,
        update_idx=update_idx,
        planned_total_updates=max(int(planned_total_updates), int(update_idx)),
        multihead_value=bool(getattr(cfg, "critic_multihead_value_enabled", False)),
    )


def _bw_agent_features(obs_list: list[dict[str, Any]], bw_alpha: np.ndarray, bw_mask: np.ndarray, bw_action: np.ndarray) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for agent_idx, obs in enumerate(obs_list):
        valid = np.asarray(bw_mask[agent_idx] > 0.5, dtype=bool)
        alpha = np.asarray(bw_alpha[agent_idx], dtype=np.float64)
        action = np.asarray(bw_action[agent_idx], dtype=np.float64)
        users = np.asarray(obs["users"], dtype=np.float64)
        if np.any(valid):
            alpha_valid = alpha[valid]
            action_valid = action[valid]
            eta_valid = users[valid, 3]
            queue_valid = users[valid, 2]
            action_safe = np.clip(action_valid, 1e-9, 1.0)
            entropy = -float(np.sum(action_safe * np.log(action_safe)))
            eta_sorted = np.sort(eta_valid)[::-1]
            eta_gap = float(eta_sorted[0] - eta_sorted[1]) if eta_sorted.size >= 2 else float(eta_sorted[0]) if eta_sorted.size == 1 else 0.0
            rows.append(
                {
                    "bw_valid_count": float(valid.sum()),
                    "bw_alpha0_old": float(alpha_valid.sum()),
                    "bw_alpha_valid_mean_old": float(np.mean(alpha_valid)),
                    "bw_action_entropy_old": entropy,
                    "bw_action_top1_share_old": float(np.max(action_valid)),
                    "bw_eta_top1_old": float(eta_sorted[0]) if eta_sorted.size else 0.0,
                    "bw_eta_gap_old": eta_gap,
                    "bw_queue_max_old": float(np.max(queue_valid)) if queue_valid.size else 0.0,
                }
            )
        else:
            rows.append(
                {
                    "bw_valid_count": 0.0,
                    "bw_alpha0_old": 0.0,
                    "bw_alpha_valid_mean_old": 0.0,
                    "bw_action_entropy_old": 0.0,
                    "bw_action_top1_share_old": 0.0,
                    "bw_eta_top1_old": 0.0,
                    "bw_eta_gap_old": 0.0,
                    "bw_queue_max_old": 0.0,
                }
            )
    return rows


def _mean_alloc_from_alpha(alpha: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_f = np.asarray(mask, dtype=np.float32)
    alpha_f = np.asarray(alpha, dtype=np.float32) * mask_f
    denom = float(np.sum(alpha_f))
    if denom <= 1e-8:
        return np.zeros_like(alpha_f, dtype=np.float32)
    return (alpha_f / denom).astype(np.float32, copy=False)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _alloc_behavior_metrics(obs: dict[str, Any], alloc: np.ndarray, queue_target: np.ndarray) -> dict[str, float]:
    valid = np.asarray(obs.get("bw_valid_mask", obs["users_mask"]) > 0.0, dtype=bool)
    alloc = np.asarray(alloc, dtype=np.float32)
    users = np.asarray(obs["users"], dtype=np.float32)
    eta = users[:, 3]
    queue = users[:, 2]
    alloc_valid = alloc[valid]
    if alloc_valid.size == 0:
        return {
            "bw_mean_entropy": 0.0,
            "bw_mean_top1_share": 0.0,
            "bw_eta_top1_mass": 0.0,
            "bw_queue_top1_mass": 0.0,
            "bw_l1_to_queueaware": 0.0,
            "bw_cos_to_queueaware": 0.0,
        }
    safe = np.clip(alloc_valid, 1e-9, 1.0)
    eta_valid_idx = np.flatnonzero(valid)[int(np.argmax(eta[valid]))]
    queue_valid_idx = np.flatnonzero(valid)[int(np.argmax(queue[valid]))]
    return {
        "bw_mean_entropy": float(-np.sum(safe * np.log(safe))),
        "bw_mean_top1_share": float(np.max(alloc_valid)),
        "bw_eta_top1_mass": float(alloc[eta_valid_idx]),
        "bw_queue_top1_mass": float(alloc[queue_valid_idx]),
        "bw_l1_to_queueaware": float(np.sum(np.abs(alloc - queue_target))),
        "bw_cos_to_queueaware": _cosine_similarity(alloc, queue_target),
    }


def _dirichlet_logprob_np(alpha_valid: np.ndarray, action_valid: np.ndarray, eps: float = DIRICHLET_EPS) -> float:
    alpha = np.asarray(alpha_valid, dtype=np.float64).reshape(-1)
    action = np.asarray(action_valid, dtype=np.float64).reshape(-1)
    if alpha.size < 2 or action.size < 2:
        return 0.0
    alpha = np.clip(alpha, eps, None)
    action = np.clip(action, eps, 1.0)
    alpha0 = float(np.sum(alpha))
    return float(
        math.lgamma(alpha0)
        - float(sum(math.lgamma(float(a)) for a in alpha))
        + float(np.dot(alpha - 1.0, np.log(action)))
    )


def _dirichlet_mean_concentration_np(alpha_valid: np.ndarray, eps: float = DIRICHLET_EPS) -> tuple[np.ndarray, float]:
    alpha = np.asarray(alpha_valid, dtype=np.float64).reshape(-1)
    alpha = np.clip(alpha, eps, None)
    alpha0 = max(float(np.sum(alpha)), eps)
    return alpha / alpha0, alpha0


def _dirichlet_logprob_decomposition(
    alpha_old_valid: np.ndarray,
    alpha_new_valid: np.ndarray,
    action_valid: np.ndarray,
    eps: float = DIRICHLET_EPS,
) -> dict[str, float]:
    alpha_old = np.asarray(alpha_old_valid, dtype=np.float64).reshape(-1)
    alpha_new = np.asarray(alpha_new_valid, dtype=np.float64).reshape(-1)
    action = np.asarray(action_valid, dtype=np.float64).reshape(-1)
    if alpha_old.size < 2 or alpha_new.size < 2 or action.size < 2:
        return {
            "logprob_bw_old_np": 0.0,
            "logprob_bw_new_np": 0.0,
            "logprob_bw_mean_component": 0.0,
            "logprob_bw_concentration_component": 0.0,
            "logprob_bw_mean_component_meanfirst": 0.0,
            "logprob_bw_mean_component_concfirst": 0.0,
            "logprob_bw_concentration_component_meanfirst": 0.0,
            "logprob_bw_concentration_component_concfirst": 0.0,
            "logprob_bw_decomp_path_gap": 0.0,
            "logprob_bw_decomp_residual": 0.0,
            "delta_mean_alloc_l1": 0.0,
            "delta_mean_alloc_top1": 0.0,
            "alpha0_old": 0.0,
            "alpha0_new": 0.0,
            "delta_alpha0": 0.0,
            "delta_log_alpha0": 0.0,
            "mean_old_entropy": 0.0,
            "mean_new_entropy": 0.0,
            "delta_mean_entropy": 0.0,
            "decomp_abs_mean_share": 0.0,
            "decomp_abs_concentration_share": 0.0,
        }

    mean_old, alpha0_old = _dirichlet_mean_concentration_np(alpha_old, eps=eps)
    mean_new, alpha0_new = _dirichlet_mean_concentration_np(alpha_new, eps=eps)

    logprob_old = _dirichlet_logprob_np(alpha_old, action, eps=eps)
    logprob_new = _dirichlet_logprob_np(alpha_new, action, eps=eps)

    alpha_oldconc_newmean = np.clip(mean_new * alpha0_old, eps, None)
    alpha_newconc_oldmean = np.clip(mean_old * alpha0_new, eps, None)
    logprob_oldconc_newmean = _dirichlet_logprob_np(alpha_oldconc_newmean, action, eps=eps)
    logprob_newconc_oldmean = _dirichlet_logprob_np(alpha_newconc_oldmean, action, eps=eps)

    mean_component_meanfirst = logprob_oldconc_newmean - logprob_old
    concentration_component_meanfirst = logprob_new - logprob_oldconc_newmean
    concentration_component_concfirst = logprob_newconc_oldmean - logprob_old
    mean_component_concfirst = logprob_new - logprob_newconc_oldmean

    mean_component = 0.5 * (mean_component_meanfirst + mean_component_concfirst)
    concentration_component = 0.5 * (
        concentration_component_meanfirst + concentration_component_concfirst
    )
    abs_total = abs(mean_component) + abs(concentration_component)
    mean_old_safe = np.clip(mean_old, eps, 1.0)
    mean_new_safe = np.clip(mean_new, eps, 1.0)
    mean_old_entropy = float(-np.sum(mean_old_safe * np.log(mean_old_safe)))
    mean_new_entropy = float(-np.sum(mean_new_safe * np.log(mean_new_safe)))

    return {
        "logprob_bw_old_np": float(logprob_old),
        "logprob_bw_new_np": float(logprob_new),
        "logprob_bw_mean_component": float(mean_component),
        "logprob_bw_concentration_component": float(concentration_component),
        "logprob_bw_mean_component_meanfirst": float(mean_component_meanfirst),
        "logprob_bw_mean_component_concfirst": float(mean_component_concfirst),
        "logprob_bw_concentration_component_meanfirst": float(concentration_component_meanfirst),
        "logprob_bw_concentration_component_concfirst": float(concentration_component_concfirst),
        "logprob_bw_decomp_path_gap": float(mean_component_meanfirst - mean_component_concfirst),
        "logprob_bw_decomp_residual": float(
            (logprob_new - logprob_old) - (mean_component + concentration_component)
        ),
        "delta_mean_alloc_l1": float(np.sum(np.abs(mean_new - mean_old))),
        "delta_mean_alloc_top1": float(np.max(np.abs(mean_new - mean_old))),
        "alpha0_old": float(alpha0_old),
        "alpha0_new": float(alpha0_new),
        "delta_alpha0": float(alpha0_new - alpha0_old),
        "delta_log_alpha0": float(math.log(alpha0_new) - math.log(alpha0_old)),
        "mean_old_entropy": mean_old_entropy,
        "mean_new_entropy": mean_new_entropy,
        "delta_mean_entropy": float(mean_new_entropy - mean_old_entropy),
        "decomp_abs_mean_share": float(abs(mean_component) / max(abs_total, eps)),
        "decomp_abs_concentration_share": float(abs(concentration_component) / max(abs_total, eps)),
    }


def _collect_signal_data(bundle: RunBundle, rollout_buffers: int, seed_base: int) -> dict[str, Any]:
    cfg = bundle.cfg
    env = make_structured_env(cfg, mode="script")
    try:
        adv_clip = float(getattr(cfg, "adv_clip", 5.0) or 0.0)
        signal_rows: list[dict[str, float]] = []
        adv_raw_chunks: list[np.ndarray] = []
        ret_chunks: list[np.ndarray] = []
        value_chunks: list[np.ndarray] = []
        mc_chunks: list[np.ndarray] = []
        delta_chunks: list[np.ndarray] = []
        err_chunks: list[np.ndarray] = []
        frontend_chunks: dict[str, list[np.ndarray]] = {
            "gu_drop_ratio_step": [],
            "gu_queue_arrival_steps": [],
            "throughput_access_norm": [],
        }
        for rollout_idx in range(int(rollout_buffers)):
            obs, _ = env.reset(seed=int(seed_base) + rollout_idx)
            rewards_list: list[float] = []
            values_list: list[Any] = []
            bootstrap_list: list[Any] = []
            boundaries_list: list[bool] = []
            frontend_lists = {k: [] for k in frontend_chunks}
            for _ in range(int(cfg.buffer_size)):
                obs_list = list(obs.values())
                obs_batch = batch_flatten_obs(obs_list, cfg).astype(np.float32, copy=False)
                obs_step = np.expand_dims(obs_batch, axis=0)
                state_batch = _get_state_batch(env).astype(np.float32, copy=False)
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs_batch)
                    obs_step_t = torch.from_numpy(obs_step)
                    state_t = torch.from_numpy(state_batch)
                    policy_out = bundle.actor.act(obs_t, deterministic=False, compute_logprob=False)
                    if bundle.multihead_value:
                        value_heads = _stack_value_head_dict(bundle.critic.forward_heads(state_t, obs_step_t)).cpu().numpy()[0]
                        value_now: Any = value_heads.astype(np.float32, copy=False)
                    else:
                        value_now = float(bundle.critic(state_t, obs_step_t).cpu().numpy().reshape(-1)[0])
                accel_cmd = (
                    policy_out.accel.cpu().numpy()
                    if policy_out.accel is not None
                    else np.zeros((len(obs_list), 2), dtype=np.float32)
                )
                bw_exec = policy_out.bw_action.cpu().numpy() if policy_out.bw_action is not None else None
                sat_exec = policy_out.sat_select_mask.cpu().numpy() if policy_out.sat_select_mask is not None else None
                action_dict = assemble_actions(cfg, env.agents, accel_cmd, bw_alloc=bw_exec, sat_select_mask=sat_exec)
                next_obs, rewards, terms, truncs, _ = env.step(action_dict)
                stats = _single_env_step_stats(env)
                stats["post_step_global_state"] = np.asarray(env.get_global_state(), dtype=np.float32)
                terminated = bool(list(terms.values())[0])
                truncated = bool(list(truncs.values())[0])
                done = terminated or truncated
                if done:
                    obs_after_reset, _ = env.reset()
                    next_obs_for_buffer = obs_after_reset
                else:
                    next_obs_for_buffer = next_obs
                next_obs_batch = batch_flatten_obs(list(next_obs_for_buffer.values()), cfg).astype(np.float32, copy=False)
                next_obs_step = np.expand_dims(next_obs_batch, axis=0)
                next_state = np.expand_dims(np.asarray(stats["post_step_global_state"], dtype=np.float32), axis=0)
                with torch.no_grad():
                    next_state_t = torch.from_numpy(next_state)
                    next_obs_t = torch.from_numpy(next_obs_step)
                    if bundle.multihead_value:
                        bootstrap_now = _stack_value_head_dict(bundle.critic.forward_heads(next_state_t, next_obs_t)).cpu().numpy()[0]
                        bootstrap_now = bootstrap_now.astype(np.float32, copy=False)
                        if terminated:
                            bootstrap_now[:] = 0.0
                    else:
                        bootstrap_now = float(bundle.critic(next_state_t, next_obs_t).cpu().numpy().reshape(-1)[0])
                        if terminated:
                            bootstrap_now = 0.0
                reward_aux, _ = _compute_train_reward_adjustment(
                    stats,
                    cfg,
                    bundle.update_idx,
                    bundle.planned_total_updates,
                )
                reward_scalar = float(list(rewards.values())[0]) + float(reward_aux)
                parts = dict(stats.get("reward_parts", {}) or {})
                rewards_list.append(reward_scalar)
                values_list.append(value_now)
                bootstrap_list.append(bootstrap_now)
                boundaries_list.append(done)
                frontend_lists["gu_drop_ratio_step"].append(float(parts.get("gu_drop_ratio_step", 0.0)))
                frontend_lists["gu_queue_arrival_steps"].append(float(parts.get("gu_queue_arrival_steps", 0.0)))
                frontend_lists["throughput_access_norm"].append(float(parts.get("throughput_access_norm", 0.0)))
                obs = next_obs_for_buffer
            rewards_arr = np.asarray(rewards_list, dtype=np.float32)
            boundaries_arr = np.asarray(boundaries_list, dtype=bool)
            if bundle.multihead_value:
                values_arr = np.asarray(values_list, dtype=np.float32).reshape(-1, len(VALUE_HEAD_NAMES))
                bootstrap_arr = np.asarray(bootstrap_list, dtype=np.float32).reshape(-1, len(VALUE_HEAD_NAMES))
                adv_by_head, ret_by_head = _compute_per_head_gae_targets(
                    rewards_arr,
                    values_arr,
                    bootstrap_arr,
                    boundaries_arr,
                    float(cfg.gamma),
                    float(cfg.gae_lambda),
                )
                signal_idx = VALUE_HEAD_NAMES.index("bw")
                adv_raw = adv_by_head["bw"]
                ret = ret_by_head["bw"]
                values_signal = values_arr[:, signal_idx]
                bootstrap_signal = bootstrap_arr[:, signal_idx]
            else:
                values_signal = np.asarray(values_list, dtype=np.float32)
                bootstrap_signal = np.asarray(bootstrap_list, dtype=np.float32)
                adv_raw, ret = compute_gae(
                    rewards_arr,
                    values_signal,
                    bootstrap_signal,
                    boundaries_arr,
                    float(cfg.gamma),
                    float(cfg.gae_lambda),
                )
            mc = _compute_mc_returns(rewards_arr, boundaries_arr, float(cfg.gamma))
            next_mc = np.zeros_like(mc)
            if mc.size > 1:
                next_mc[:-1] = np.where(boundaries_arr[:-1], 0.0, mc[1:])
            delta = rewards_arr + float(cfg.gamma) * bootstrap_signal - values_signal
            err_term = float(cfg.gamma) * (bootstrap_signal - next_mc) - (values_signal - mc)
            adv_raw_chunks.append(adv_raw.astype(np.float32, copy=False))
            ret_chunks.append(ret.astype(np.float32, copy=False))
            value_chunks.append(values_signal.astype(np.float32, copy=False))
            mc_chunks.append(mc.astype(np.float32, copy=False))
            delta_chunks.append(delta.astype(np.float32, copy=False))
            err_chunks.append(err_term.astype(np.float32, copy=False))
            for key in frontend_chunks:
                frontend_chunks[key].append(np.asarray(frontend_lists[key], dtype=np.float32))
        adv_raw_all = np.concatenate(adv_raw_chunks, axis=0)
        adv_final_all, adv_stats = _normalize_advantages(adv_raw_all, adv_clip)
        ret_all = np.concatenate(ret_chunks, axis=0)
        value_all = np.concatenate(value_chunks, axis=0)
        mc_all = np.concatenate(mc_chunks, axis=0)
        delta_all = np.concatenate(delta_chunks, axis=0)
        err_all = np.concatenate(err_chunks, axis=0)
        frontend_all = {key: np.concatenate(chunks, axis=0) for key, chunks in frontend_chunks.items()}
        for idx in range(int(adv_raw_all.shape[0])):
            signal_rows.append(
                {
                    "strategy": bundle.name,
                    "adv_raw": float(adv_raw_all[idx]),
                    "adv_final": float(adv_final_all[idx]),
                    "delta_signal": float(delta_all[idx]),
                    "value_signal": float(value_all[idx]),
                    "mc_return": float(mc_all[idx]),
                    "err_term": float(err_all[idx]),
                    "gu_drop_ratio_step": float(frontend_all["gu_drop_ratio_step"][idx]),
                    "gu_queue_arrival_steps": float(frontend_all["gu_queue_arrival_steps"][idx]),
                    "throughput_access_norm": float(frontend_all["throughput_access_norm"][idx]),
                }
            )
        return {
            "rows": signal_rows,
            "adv_stats": adv_stats,
            "ret_mean": float(np.mean(ret_all)),
            "value_mean": float(np.mean(value_all)),
            "mc_mean": float(np.mean(mc_all)),
        }
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def _collect_replay_rows(
    bundle: RunBundle,
    rollout_buffers: int,
    seed_base: int,
    replay_seed_base: int,
) -> dict[str, Any]:
    cfg = bundle.cfg
    if bundle.train_state_path is None or not bundle.train_state_path.is_file():
        raise FileNotFoundError(
            f"Replay diagnostics for {bundle.name} require a train state checkpoint, but none was found."
        )
    state = torch.load(bundle.train_state_path, map_location="cpu")
    actor = ActorNet(_make_dims(cfg)[0], cfg).to(torch.device("cpu"))
    actor.load_state_dict(state["actor_state_dict"])
    actor.train()
    raw_train_accel = getattr(cfg, "train_accel", None)
    raw_train_bw = getattr(cfg, "train_bw", None)
    raw_train_sat = getattr(cfg, "train_sat", None)
    train_accel = True if raw_train_accel is None else bool(raw_train_accel)
    train_bw = bool(cfg.enable_bw_action) if raw_train_bw is None else bool(raw_train_bw)
    train_sat = (not bool(cfg.fixed_satellite_strategy)) if raw_train_sat is None else bool(raw_train_sat)
    if not cfg.enable_bw_action:
        train_bw = False
    if cfg.fixed_satellite_strategy:
        train_sat = False
    train_heads = {
        "accel": train_accel,
        "bw": train_bw,
        "sat": train_sat,
    }
    _configure_actor_trainability(actor, cfg, train_heads)
    actor_params = [param for param in actor.parameters() if param.requires_grad]
    actor_optim = torch.optim.Adam(actor_params, lr=float(cfg.actor_lr))
    try:
        actor_optim.load_state_dict(state["actor_optimizer_state_dict"])
    except ValueError as exc:
        print(
            f"[warn] replay optimizer state mismatch for {bundle.name}; "
            f"using fresh Adam state instead. Details: {exc}"
        )
    rollout = _collect_signal_data(bundle, rollout_buffers=rollout_buffers, seed_base=seed_base)
    env = make_structured_env(cfg, mode="script")
    try:
        obs, _ = env.reset(seed=int(replay_seed_base))
        obs_rows: list[np.ndarray] = []
        act_rows: list[np.ndarray] = []
        sat_rows: list[np.ndarray] = []
        old_bw_logp_rows: list[np.ndarray] = []
        old_bw_alpha_rows: list[np.ndarray] = []
        old_bw_mask_rows: list[np.ndarray] = []
        adv_bw_step: list[float] = []
        feature_rows: list[dict[str, float]] = []
        feature_contexts: list[tuple[dict[str, Any], np.ndarray]] = []
        step_rows: list[dict[str, float]] = []
        episode_id = 0
        for step_idx in range(int(cfg.buffer_size)):
            row_base = rollout["rows"][step_idx]
            obs_list = list(obs.values())
            obs_batch = batch_flatten_obs(obs_list, cfg).astype(np.float32, copy=False)
            with torch.no_grad():
                obs_t = torch.from_numpy(obs_batch)
                out = actor.act(obs_t, deterministic=False, compute_logprob=False)
            accel_cmd = out.accel.cpu().numpy()
            bw_exec = out.bw_action.cpu().numpy() if out.bw_action is not None else None
            sat_exec = out.sat_select_mask.cpu().numpy() if out.sat_select_mask is not None else None
            action_dict = assemble_actions(cfg, env.agents, accel_cmd, bw_alloc=bw_exec, sat_select_mask=sat_exec)
            next_obs, rewards, terms, truncs, _ = env.step(action_dict)
            stats = _single_env_step_stats(env)
            parts = dict(getattr(env, "last_reward_parts", {}) or {})
            step_rows.append(
                {
                    "step_idx": float(step_idx),
                    "episode_id": float(episode_id),
                    "throughput_access_norm": float(parts.get("throughput_access_norm", 0.0)),
                    "throughput_backhaul_norm": float(parts.get("throughput_backhaul_norm", 0.0)),
                    "gu_queue_arrival_steps": float(parts.get("gu_queue_arrival_steps", 0.0)),
                    "gu_drop_ratio_step": float(parts.get("gu_drop_ratio_step", 0.0)),
                }
            )
            fallback_accel = accel_cmd * float(cfg.a_max)
            accel_exec = np.asarray(stats.get("last_exec_accel", fallback_accel), dtype=np.float32)
            accel_norm = np.clip(accel_exec / max(float(cfg.a_max), 1e-6), -1.0, 1.0)
            exec_parts = [accel_norm]
            exec_bw = np.asarray(stats.get("last_exec_bw_alloc", bw_exec), dtype=np.float32)
            exec_parts.append(exec_bw)
            exec_sat = np.asarray(stats.get("last_exec_sat_select_mask", sat_exec), dtype=np.float32)
            exec_parts.append(exec_sat)
            action_vec = np.concatenate(exec_parts, axis=1).astype(np.float32, copy=False)
            sat_indices = np.asarray(stats.get("last_exec_sat_indices"), dtype=np.int64)
            with torch.no_grad():
                act_t = torch.from_numpy(action_vec)
                sat_t = torch.from_numpy(sat_indices)
                logp_parts, _ = actor.evaluate_actions_parts(
                    obs_t,
                    act_t,
                    sat_indices=sat_t,
                    out=out.dist_out,
                    heads=VALUE_HEAD_NAMES,
                    need_entropy=False,
                )
            bw_alpha = out.dist_out["bw_alpha"].detach().cpu().numpy()
            bw_mask = out.dist_out["bw_valid_mask"].detach().cpu().numpy()
            bw_action = out.bw_action.detach().cpu().numpy()
            per_agent_feats = _bw_agent_features(obs_list, bw_alpha, bw_mask, bw_action)
            queue_target = queue_aware_bw_policy(obs_list, cfg)
            for agent_idx in range(int(cfg.num_uav)):
                feat = dict(per_agent_feats[agent_idx])
                mean_alloc_old = _mean_alloc_from_alpha(bw_alpha[agent_idx], bw_mask[agent_idx] > 0.5)
                old_behavior = _alloc_behavior_metrics(obs_list[agent_idx], mean_alloc_old, queue_target[agent_idx])
                feat.update({f"{k}_old": v for k, v in old_behavior.items()})
                feat["step_idx"] = float(step_idx)
                feat["agent_idx"] = float(agent_idx)
                feat["adv_bw_final"] = float(row_base["adv_final"])
                feat["gu_drop_ratio_step"] = float(row_base["gu_drop_ratio_step"])
                feat["gu_queue_arrival_steps"] = float(row_base["gu_queue_arrival_steps"])
                feat["throughput_access_norm"] = float(row_base["throughput_access_norm"])
                feature_rows.append(feat)
                feature_contexts.append((obs_list[agent_idx], np.asarray(queue_target[agent_idx], dtype=np.float32)))
            obs_rows.append(obs_batch)
            act_rows.append(action_vec)
            sat_rows.append(sat_indices)
            old_bw_logp_rows.append(logp_parts["bw"].detach().cpu().numpy().reshape(-1))
            old_bw_alpha_rows.append(bw_alpha.astype(np.float32, copy=False))
            old_bw_mask_rows.append(bw_mask.astype(np.float32, copy=False))
            adv_bw_step.extend([float(row_base["adv_final"])] * int(cfg.num_uav))
            done = bool(list(terms.values())[0] or list(truncs.values())[0])
            if done:
                obs = env.reset()[0]
                episode_id += 1
            else:
                obs = next_obs
        obs_flat = np.concatenate(obs_rows, axis=0)
        act_flat = np.concatenate(act_rows, axis=0)
        sat_flat = np.concatenate(sat_rows, axis=0)
        old_bw_logp_flat = np.concatenate(old_bw_logp_rows, axis=0)
        old_bw_alpha_flat = np.concatenate(old_bw_alpha_rows, axis=0)
        old_bw_mask_flat = np.concatenate(old_bw_mask_rows, axis=0)
        adv_bw_flat = np.asarray(adv_bw_step, dtype=np.float32)
        batch_size = obs_flat.shape[0]
        minibatch_size = max(int(batch_size // max(int(getattr(cfg, "num_mini_batch", 1) or 1), 1)), 1)
        replay_rows: list[dict[str, float]] = []
        for start in range(0, batch_size, minibatch_size):
            mb_idx = np.arange(start, min(start + minibatch_size, batch_size))
            obs_t = torch.from_numpy(obs_flat[mb_idx])
            act_t = torch.from_numpy(act_flat[mb_idx])
            sat_t = torch.from_numpy(sat_flat[mb_idx])
            old_bw_t = torch.from_numpy(old_bw_logp_flat[mb_idx])
            adv_t = torch.from_numpy(adv_bw_flat[mb_idx])
            out = actor.forward(obs_t, required_heads=VALUE_HEAD_NAMES)
            logprob_parts, _ = actor.evaluate_actions_parts(
                obs_t, act_t, sat_indices=sat_t, out=out, heads=VALUE_HEAD_NAMES, need_entropy=False
            )
            part_log_ratio = torch.clamp(logprob_parts["bw"] - old_bw_t, -8.0, 8.0)
            part_ratio = torch.exp(part_log_ratio)
            surr1 = part_ratio * adv_t
            surr2 = torch.clamp(part_ratio, 1.0 - float(cfg.clip_ratio), 1.0 + float(cfg.clip_ratio)) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()
            actor_optim.zero_grad()
            policy_loss.backward()
            actor_optim.step()
            with torch.no_grad():
                out_new = actor.forward(obs_t, required_heads=VALUE_HEAD_NAMES)
                logprob_new, _ = actor.evaluate_actions_parts(
                    obs_t, act_t, sat_indices=sat_t, out=out_new, heads=VALUE_HEAD_NAMES, need_entropy=False
                )
            bw_alpha_new = out_new["bw_alpha"].detach().cpu().numpy()
            for local_idx, sample_idx in enumerate(mb_idx):
                feat = dict(feature_rows[int(sample_idx)])
                obs_ref, queue_target_single = feature_contexts[int(sample_idx)]
                valid_mask = np.asarray(old_bw_mask_flat[int(sample_idx)] > 0.5, dtype=bool)
                valid_old = old_bw_alpha_flat[int(sample_idx)][valid_mask]
                valid_new = bw_alpha_new[local_idx][valid_mask]
                bw_action_exec = np.asarray(
                    act_flat[int(sample_idx), 2 : 2 + int(cfg.users_obs_max)],
                    dtype=np.float64,
                )
                valid_action = bw_action_exec[valid_mask]
                decomp = _dirichlet_logprob_decomposition(valid_old, valid_new, valid_action)
                mean_alloc_new = _mean_alloc_from_alpha(bw_alpha_new[local_idx], valid_mask)
                log_ratio_val = float((logprob_new["bw"][local_idx] - old_bw_t[local_idx]).item())
                ratio_val = float(math.exp(max(min(log_ratio_val, 8.0), -8.0)))
                new_behavior = _alloc_behavior_metrics(
                    obs_ref,
                    mean_alloc_new,
                    queue_target_single,
                )
                feat.update(
                    {
                        "log_ratio_bw": log_ratio_val,
                        "ratio_bw": ratio_val,
                        "abs_log_ratio_bw": float(abs(log_ratio_val)),
                        "clip_hit_bw": float(abs(ratio_val - 1.0) > float(cfg.clip_ratio)),
                        "delta_alpha_valid_abs_mean": float(np.mean(np.abs(valid_new - valid_old))) if valid_old.size else 0.0,
                        "alpha0_new": float(valid_new.sum()) if valid_new.size else 0.0,
                        "logprob_bw_old_torch": float(old_bw_t[local_idx].item()),
                        "logprob_bw_new_torch": float(logprob_new["bw"][local_idx].item()),
                    }
                )
                feat.update(decomp)
                feat["logprob_bw_old_gap_np_vs_torch"] = feat["logprob_bw_old_np"] - feat["logprob_bw_old_torch"]
                feat["logprob_bw_new_gap_np_vs_torch"] = feat["logprob_bw_new_np"] - feat["logprob_bw_new_torch"]
                feat.update({f"{k}_new": v for k, v in new_behavior.items()})
                feat["delta_bw_mean_entropy"] = feat["bw_mean_entropy_new"] - feat["bw_mean_entropy_old"]
                feat["delta_bw_mean_top1_share"] = feat["bw_mean_top1_share_new"] - feat["bw_mean_top1_share_old"]
                feat["delta_bw_eta_top1_mass"] = feat["bw_eta_top1_mass_new"] - feat["bw_eta_top1_mass_old"]
                feat["delta_bw_queue_top1_mass"] = feat["bw_queue_top1_mass_new"] - feat["bw_queue_top1_mass_old"]
                feat["delta_bw_l1_to_queueaware"] = feat["bw_l1_to_queueaware_new"] - feat["bw_l1_to_queueaware_old"]
                feat["delta_bw_cos_to_queueaware"] = feat["bw_cos_to_queueaware_new"] - feat["bw_cos_to_queueaware_old"]
                replay_rows.append(feat)
        return {"sample_rows": replay_rows, "step_rows": step_rows}
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def _summarize_signal(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload["rows"]
    metrics = ["gu_drop_ratio_step", "gu_queue_arrival_steps", "throughput_access_norm"]
    adv = np.asarray([row["adv_final"] for row in rows], dtype=np.float64)
    delta = np.asarray([row["delta_signal"] for row in rows], dtype=np.float64)
    err = np.asarray([row["err_term"] for row in rows], dtype=np.float64)
    value = np.asarray([row["value_signal"] for row in rows], dtype=np.float64)
    mc = np.asarray([row["mc_return"] for row in rows], dtype=np.float64)
    out: dict[str, Any] = {
        "strategy": name,
        "num_env_steps": int(len(rows)),
        "adv_raw_std": float(payload["adv_stats"]["raw_std"]),
        "adv_clip_hit_rate": float(payload["adv_stats"]["clip_frac"]),
        "adv_raw_abs_p95": float(np.percentile(np.abs([row["adv_raw"] for row in rows]), 95.0)),
        "adv_raw_abs_p99": float(np.percentile(np.abs([row["adv_raw"] for row in rows]), 99.0)),
        "corr_value_mc_pearson": _pearson(value, mc),
        "corr_value_mc_spearman": _spearman(value, mc),
        "value_mae_mc": float(np.mean(np.abs(value - mc))),
        "explained_variance_mc": _explained_variance(value, mc),
    }
    for metric in metrics:
        arr = np.asarray([row[metric] for row in rows], dtype=np.float64)
        out[f"corr_{metric}_adv_final_pearson"] = _pearson(arr, adv)
        out[f"corr_{metric}_adv_final_spearman"] = _spearman(arr, adv)
        out[f"corr_{metric}_delta_pearson"] = _pearson(arr, delta)
        out[f"corr_{metric}_delta_spearman"] = _spearman(arr, delta)
        out[f"corr_{metric}_err_term_pearson"] = _pearson(arr, err)
        out[f"corr_{metric}_err_term_spearman"] = _spearman(arr, err)
    return out


def _summarize_replay_rows(rows: list[dict[str, float]]) -> dict[str, Any]:
    log_ratio = np.asarray([row["log_ratio_bw"] for row in rows], dtype=np.float64)
    ratio = np.asarray([row["ratio_bw"] for row in rows], dtype=np.float64)
    clip_hits = np.asarray([row["clip_hit_bw"] for row in rows], dtype=np.float64)
    summary = {
        "num_agent_samples": int(len(rows)),
        "clip_hit_rate_bw": float(np.mean(clip_hits)) if clip_hits.size else 0.0,
        "log_ratio_bw": _summary(log_ratio.tolist()),
        "ratio_bw": _summary(ratio.tolist()),
    }

    def _row_stats(source_rows: list[dict[str, float]]) -> dict[str, Any]:
        if not source_rows:
            return {}
        fields = [
            "abs_log_ratio_bw",
            "logprob_bw_mean_component",
            "logprob_bw_concentration_component",
            "delta_mean_alloc_l1",
            "delta_mean_alloc_top1",
            "delta_log_alpha0",
            "decomp_abs_mean_share",
            "decomp_abs_concentration_share",
            "logprob_bw_decomp_path_gap",
            "logprob_bw_decomp_residual",
            "logprob_bw_old_gap_np_vs_torch",
            "logprob_bw_new_gap_np_vs_torch",
        ]
        return {name: _summary([float(row[name]) for row in source_rows]) for name in fields}

    clipped = [row for row in rows if row["clip_hit_bw"] > 0.5]
    focus_rows = clipped if clipped else rows
    summary["focus_source"] = "clipped" if clipped else "all"
    feature_names = [
        "bw_valid_count",
        "bw_alpha0_old",
        "bw_alpha_valid_mean_old",
        "bw_action_entropy_old",
        "bw_action_top1_share_old",
        "bw_eta_top1_old",
        "bw_eta_gap_old",
        "bw_queue_max_old",
        "adv_bw_final",
        "gu_drop_ratio_step",
        "gu_queue_arrival_steps",
        "throughput_access_norm",
        "delta_alpha_valid_abs_mean",
    ]
    summary["feature_means_all"] = {
        name: float(np.mean([row[name] for row in rows])) if rows else 0.0 for name in feature_names
    }
    summary["feature_means_clipped"] = {
        name: float(np.mean([row[name] for row in clipped])) if clipped else 0.0 for name in feature_names
    }
    summary["feature_means_focus"] = {
        name: float(np.mean([row[name] for row in focus_rows])) if focus_rows else 0.0 for name in feature_names
    }
    abs_log_ratio_rows = np.abs(log_ratio)
    if abs_log_ratio_rows.size > 0:
        abs_log_ratio_p90 = float(np.percentile(abs_log_ratio_rows, 90.0))
        top_abs_rows = [row for row in rows if abs(float(row["log_ratio_bw"])) >= abs_log_ratio_p90]
    else:
        abs_log_ratio_p90 = 0.0
        top_abs_rows = []
    summary["abs_log_ratio_bw_p90"] = abs_log_ratio_p90
    summary["top_abs_log_ratio_count"] = int(len(top_abs_rows))
    summary["decomposition_stats_all"] = _row_stats(rows)
    summary["decomposition_stats_focus"] = _row_stats(focus_rows)
    summary["decomposition_stats_top_abs_log_ratio"] = _row_stats(top_abs_rows)
    if rows:
        abs_mean_component = np.asarray(
            [abs(float(row["logprob_bw_mean_component"])) for row in rows],
            dtype=np.float64,
        )
        abs_concentration_component = np.asarray(
            [abs(float(row["logprob_bw_concentration_component"])) for row in rows],
            dtype=np.float64,
        )
        valid_count = np.asarray([float(row["bw_valid_count"]) for row in rows], dtype=np.float64)
        delta_mean_alloc_l1 = np.asarray([float(row["delta_mean_alloc_l1"]) for row in rows], dtype=np.float64)
        abs_delta_log_alpha0 = np.asarray([abs(float(row["delta_log_alpha0"])) for row in rows], dtype=np.float64)
        summary["decomposition_correlations_all"] = {
            "corr_abs_log_ratio_abs_mean_component_pearson": _pearson(abs_log_ratio_rows, abs_mean_component),
            "corr_abs_log_ratio_abs_mean_component_spearman": _spearman(abs_log_ratio_rows, abs_mean_component),
            "corr_abs_log_ratio_abs_concentration_component_pearson": _pearson(
                abs_log_ratio_rows,
                abs_concentration_component,
            ),
            "corr_abs_log_ratio_abs_concentration_component_spearman": _spearman(
                abs_log_ratio_rows,
                abs_concentration_component,
            ),
            "corr_valid_count_abs_mean_component_pearson": _pearson(valid_count, abs_mean_component),
            "corr_valid_count_abs_mean_component_spearman": _spearman(valid_count, abs_mean_component),
            "corr_valid_count_abs_concentration_component_pearson": _pearson(
                valid_count,
                abs_concentration_component,
            ),
            "corr_valid_count_abs_concentration_component_spearman": _spearman(
                valid_count,
                abs_concentration_component,
            ),
            "corr_delta_mean_alloc_l1_abs_mean_component_pearson": _pearson(
                delta_mean_alloc_l1,
                abs_mean_component,
            ),
            "corr_delta_mean_alloc_l1_abs_mean_component_spearman": _spearman(
                delta_mean_alloc_l1,
                abs_mean_component,
            ),
            "corr_abs_delta_log_alpha0_abs_concentration_component_pearson": _pearson(
                abs_delta_log_alpha0,
                abs_concentration_component,
            ),
            "corr_abs_delta_log_alpha0_abs_concentration_component_spearman": _spearman(
                abs_delta_log_alpha0,
                abs_concentration_component,
            ),
        }
    else:
        summary["decomposition_correlations_all"] = {}
    direction_fields = [
        "delta_bw_mean_entropy",
        "delta_bw_mean_top1_share",
        "delta_bw_eta_top1_mass",
        "delta_bw_queue_top1_mass",
        "delta_bw_l1_to_queueaware",
        "delta_bw_cos_to_queueaware",
    ]
    summary["direction_means_clipped"] = {
        name: float(np.mean([row[name] for row in clipped])) if clipped else 0.0 for name in direction_fields
    }
    summary["direction_means_focus"] = {
        name: float(np.mean([row[name] for row in focus_rows])) if focus_rows else 0.0 for name in direction_fields
    }
    summary["direction_sign_rates_clipped"] = {
        f"{name}_positive_rate": (
            float(np.mean([row[name] > 0.0 for row in clipped])) if clipped else 0.0
        )
        for name in direction_fields
    }
    summary["direction_sign_rates_focus"] = {
        f"{name}_positive_rate": (
            float(np.mean([row[name] > 0.0 for row in focus_rows])) if focus_rows else 0.0
        )
        for name in direction_fields
    }
    if rows:
        q_abs_adv = float(np.median(np.abs([row["adv_bw_final"] for row in rows])))
        adv_focus_rows = [row for row in rows if float(row["adv_bw_final"]) > q_abs_adv]
    else:
        q_abs_adv = 0.0
        adv_focus_rows = []
    summary["q_abs_adv_median_all"] = q_abs_adv
    summary["adv_gt_pos_q_count"] = int(len(adv_focus_rows))
    summary["direction_means_adv_gt_pos_q"] = {
        name: float(np.mean([row[name] for row in adv_focus_rows])) if adv_focus_rows else 0.0
        for name in direction_fields
    }
    summary["direction_sign_rates_adv_gt_pos_q"] = {
        f"{name}_positive_rate": (
            float(np.mean([row[name] > 0.0 for row in adv_focus_rows])) if adv_focus_rows else 0.0
        )
        for name in direction_fields
    }
    return summary


def _bucket_label(adv: float, q: float) -> str:
    if adv > q:
        return "adv_gt_pos_q"
    if 0.0 < adv <= q:
        return "adv_pos_small"
    if -q <= adv < 0.0:
        return "adv_neg_small"
    return "adv_lt_neg_q"


def _summarize_window_buckets(rows: list[dict[str, float]], step_rows: list[dict[str, float]]) -> dict[str, Any]:
    clipped = [row for row in rows if row["clip_hit_bw"] > 0.5]
    focus_rows = clipped if clipped else rows
    if not focus_rows:
        return {"q_abs_adv_median": 0.0, "windows": {}}
    q = float(np.median(np.abs([row["adv_bw_final"] for row in focus_rows])))
    step_map = {int(row["step_idx"]): row for row in step_rows}
    horizons = [3, 5, 10]
    bucket_names = ["adv_gt_pos_q", "adv_pos_small", "adv_neg_small", "adv_lt_neg_q"]
    summary: dict[str, Any] = {"q_abs_adv_median": q, "focus_source": ("clipped" if clipped else "all"), "windows": {}}
    for bucket_name in bucket_names:
        bucket_rows = [row for row in focus_rows if _bucket_label(float(row["adv_bw_final"]), q) == bucket_name]
        entry: dict[str, Any] = {"count": len(bucket_rows)}
        if not bucket_rows:
            summary["windows"][bucket_name] = entry
            continue
        entry["curr_throughput_access_norm_mean"] = float(np.mean([row["throughput_access_norm"] for row in bucket_rows]))
        entry["curr_gu_queue_arrival_steps_mean"] = float(np.mean([row["gu_queue_arrival_steps"] for row in bucket_rows]))
        entry["curr_gu_drop_ratio_step_mean"] = float(np.mean([row["gu_drop_ratio_step"] for row in bucket_rows]))
        for horizon in horizons:
            delta_queue_vals: list[float] = []
            cum_drop_vals: list[float] = []
            future_access_vals: list[float] = []
            future_backhaul_vals: list[float] = []
            valid_window_count = 0
            for row in bucket_rows:
                step_idx = int(row["step_idx"])
                base = step_map.get(step_idx)
                future = step_map.get(step_idx + horizon)
                if base is None or future is None:
                    continue
                if int(base["episode_id"]) != int(future["episode_id"]):
                    continue
                valid_window_count += 1
                delta_queue_vals.append(float(future["gu_queue_arrival_steps"] - base["gu_queue_arrival_steps"]))
                cum_drop = 0.0
                future_access_sum = 0.0
                future_backhaul_sum = 0.0
                for idx in range(step_idx + 1, step_idx + horizon + 1):
                    nxt = step_map.get(idx)
                    if nxt is None or int(nxt["episode_id"]) != int(base["episode_id"]):
                        break
                    cum_drop += float(nxt["gu_drop_ratio_step"])
                    future_access_sum += float(nxt["throughput_access_norm"])
                    future_backhaul_sum += float(nxt["throughput_backhaul_norm"])
                cum_drop_vals.append(cum_drop)
                future_access_vals.append(future_access_sum)
                future_backhaul_vals.append(future_backhaul_sum)
            entry[f"window_{horizon}_count"] = valid_window_count
            entry[f"window_{horizon}_delta_gu_queue_mean"] = float(np.mean(delta_queue_vals)) if delta_queue_vals else None
            entry[f"window_{horizon}_cum_gu_drop_mean"] = float(np.mean(cum_drop_vals)) if cum_drop_vals else None
            entry[f"window_{horizon}_cum_access_mean"] = float(np.mean(future_access_vals)) if future_access_vals else None
            entry[f"window_{horizon}_cum_backhaul_mean"] = float(np.mean(future_backhaul_vals)) if future_backhaul_vals else None
        summary["windows"][bucket_name] = entry
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare bw signal / structure diagnostics across one or more run directories. "
            "Use NAME=RUN_DIR for final checkpoints, or NAME=RUN_DIR|actor_ckpt|critic_ckpt|train_state_ckpt|update_idx "
            "to override defaults."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help=(
            "Run spec in the form NAME=RUN_DIR or "
            "NAME=RUN_DIR|actor_ckpt|critic_ckpt|train_state_ckpt|update_idx. Repeat for multiple runs."
        ),
    )
    parser.add_argument(
        "--replay-name",
        action="append",
        default=[],
        help="Run name to generate replay-based bw structural diagnostics for. Repeat to diagnose multiple runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <first-run-dir>/bw_diag_compare.",
    )
    parser.add_argument("--rollout-buffers", type=int, default=DEFAULT_ROLLOUT_BUFFERS)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--replay-seed-base", type=int, default=DEFAULT_REPLAY_SEED_BASE)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_specs = [_parse_run_spec(raw) for raw in args.run]
    if not run_specs:
        raise ValueError("At least one --run spec is required.")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else run_specs[0].run_dir / "bw_diag_compare"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    bundles: list[RunBundle] = []
    bundles_by_name: dict[str, RunBundle] = {}
    signal_rows: list[dict[str, Any]] = []
    signal_payloads: dict[str, dict[str, Any]] = {}
    for spec in run_specs:
        bundle = _load_bundle(spec)
        bundles.append(bundle)
        bundles_by_name[bundle.name] = bundle
        payload = _collect_signal_data(bundle, rollout_buffers=int(args.rollout_buffers), seed_base=int(args.seed_base))
        signal_payloads[bundle.name] = payload
        signal_rows.append(_summarize_signal(bundle.name, payload))

    signal_compare_csv = output_dir / "signal_compare.csv"
    _write_csv(signal_compare_csv, signal_rows, list(signal_rows[0].keys()))

    replay_names = args.replay_name or []
    replay_outputs: dict[str, Any] = {}
    for replay_name in replay_names:
        if replay_name not in bundles_by_name:
            raise ValueError(f"--replay-name '{replay_name}' did not match any provided --run spec.")
        replay_bundle = _collect_replay_rows(
            bundles_by_name[replay_name],
            rollout_buffers=int(args.rollout_buffers),
            seed_base=int(args.seed_base),
            replay_seed_base=int(args.replay_seed_base),
        )
        replay_rows = replay_bundle["sample_rows"]
        replay_summary = _summarize_replay_rows(replay_rows)
        window_summary = _summarize_window_buckets(replay_rows, replay_bundle["step_rows"])
        top_rows = sorted(replay_rows, key=lambda row: abs(row["log_ratio_bw"]), reverse=True)[:200]
        top_csv = output_dir / f"{replay_name}_top_bw_logratio_samples.csv"
        if top_rows:
            _write_csv(top_csv, top_rows, list(top_rows[0].keys()))
        replay_outputs[replay_name] = {
            "top_bw_logratio_samples_csv": str(top_csv),
            "bw_replay": replay_summary,
            "bw_window_buckets": window_summary,
        }

    summary = {
        "signal_compare_csv": str(signal_compare_csv),
        "signal_compare": signal_rows,
        "replay": replay_outputs,
    }
    _write_json(output_dir / "summary.json", summary)
    print(f"Wrote diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
