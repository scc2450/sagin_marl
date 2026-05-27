from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
BW_DIAG_PATH = SCRIPT_DIR / "diagnose_bw_perhead_vs_oldjoint.py"
BW_SPEC = importlib.util.spec_from_file_location("diagnose_bw_perhead_vs_oldjoint", BW_DIAG_PATH)
if BW_SPEC is None or BW_SPEC.loader is None:
    raise RuntimeError(f"Failed to load module spec from {BW_DIAG_PATH}")
BW_DIAG = importlib.util.module_from_spec(BW_SPEC)
sys.modules[BW_SPEC.name] = BW_DIAG
BW_SPEC.loader.exec_module(BW_DIAG)

RunSpec = BW_DIAG.RunSpec
_load_bundle = BW_DIAG._load_bundle
_get_state_batch = BW_DIAG._get_state_batch

from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.rl.action_assembler import assemble_actions
from sagin_marl.rl.mappo import (
    _build_danger_imitation_step_data,
    _configure_actor_trainability,
    _compute_train_reward_adjustment,
    _normalize_advantages,
    _selected_train_head_names,
    _single_env_step_stats,
    _sum_selected_parts,
    compute_gae,
)
from sagin_marl.rl.policy import batch_flatten_obs
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def _parse_updates(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if token:
            out.append(int(token))
    if not out:
        raise ValueError("No checkpoint updates were provided.")
    return out


def _load_metrics_index(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        raw = row.get("step")
        if raw is not None and raw != "":
            out[int(raw)] = row
    return out


def _load_checkpoint_eval_index(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        raw = row.get("update")
        if raw is not None and raw != "":
            out[int(raw)] = row
    return out


def _metric_row_for_update(metrics_index: dict[int, dict[str, str]], update: int) -> tuple[int | None, dict[str, str] | None]:
    if update in metrics_index:
        return update, metrics_index[update]
    if (update - 1) in metrics_index:
        return update - 1, metrics_index[update - 1]
    return None, None


def _safe_float(row: dict[str, Any] | None, key: str) -> float | None:
    if row is None:
        return None
    value = row.get(key)
    if value is None or value == "":
        return None
    return float(value)


def _summary(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "p99": float(np.percentile(arr, 99.0)),
        "max": float(np.max(arr)),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _train_heads_from_cfg(cfg) -> dict[str, bool]:
    raw_train_accel = getattr(cfg, "train_accel", None)
    raw_train_bw = getattr(cfg, "train_bw", None)
    raw_train_sat = getattr(cfg, "train_sat", None)
    train_accel = True if raw_train_accel is None else bool(raw_train_accel)
    train_bw = bool(cfg.enable_bw_action) if raw_train_bw is None else bool(raw_train_bw)
    train_sat = (not bool(cfg.fixed_satellite_strategy)) if raw_train_sat is None else bool(raw_train_sat)
    if not bool(cfg.enable_bw_action):
        train_bw = False
    if bool(cfg.fixed_satellite_strategy):
        train_sat = False
    return {
        "accel": train_accel,
        "bw": train_bw,
        "sat": train_sat,
    }


def _seed_everything(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _guard_supported_cfg(bundle) -> None:
    cfg = bundle.cfg
    if bundle.multihead_value:
        raise NotImplementedError(
            "This diagnostic currently targets scalar-value joint runs; critic_multihead_value_enabled=true is not supported."
        )
    if bool(getattr(cfg, "ppo_headwise_surrogate_enabled", False)):
        raise NotImplementedError(
            "This diagnostic currently targets pure-MAPPO style runs with ppo_headwise_surrogate_enabled=false."
        )
    if bool(getattr(cfg, "ppo_per_head_advantage_enabled", False)):
        raise NotImplementedError(
            "This diagnostic currently targets runs with ppo_per_head_advantage_enabled=false."
        )
    if bool(getattr(cfg, "bw_false_positive_mask_enabled", False)):
        raise NotImplementedError(
            "This diagnostic currently does not support bw_false_positive_mask_enabled=true."
        )
    if bool(getattr(cfg, "bw_continuous_credit_enabled", False)):
        raise NotImplementedError(
            "This diagnostic currently does not support bw_continuous_credit_enabled=true."
        )
    imitation_coef = float(getattr(cfg, "imitation_coef", 0.0) or 0.0)
    if bool(getattr(cfg, "imitation_enabled", False)) and imitation_coef > 0.0:
        raise NotImplementedError("This diagnostic currently does not support imitation_enabled=true.")
    sat_supervision_coef = float(getattr(cfg, "sat_supervision_coef", 0.0) or 0.0)
    if bool(getattr(cfg, "sat_supervision_enabled", False)) and sat_supervision_coef > 0.0:
        raise NotImplementedError("This diagnostic currently does not support sat_supervision_enabled=true.")

def _collect_rollout_payload(bundle, rollout_buffers: int, seed_base: int) -> dict[str, Any]:
    _guard_supported_cfg(bundle)
    cfg = bundle.cfg
    train_heads = _train_heads_from_cfg(cfg)
    train_head_names = _selected_train_head_names(train_heads)
    if not train_head_names:
        raise ValueError("No trainable action heads were enabled for this run.")

    env = make_structured_env(cfg, mode="script")
    try:
        obs_arr_list: list[np.ndarray] = []
        next_obs_arr_list: list[np.ndarray] = []
        act_arr_list: list[np.ndarray] = []
        logp_arr_list: list[np.ndarray] = []
        state_arr_list: list[np.ndarray] = []
        next_state_arr_list: list[np.ndarray] = []
        rewards_list: list[np.ndarray] = []
        values_list: list[np.ndarray] = []
        terminated_list: list[np.ndarray] = []
        truncated_list: list[np.ndarray] = []
        sat_indices_list: list[np.ndarray] = []
        danger_target_list: list[np.ndarray] = []
        danger_mask_list: list[np.ndarray] = []
        logp_part_lists = {name: [] for name in ("accel", "bw", "sat")}
        context = {
            "gu_drop_ratio_step": [],
            "gu_queue_arrival_steps": [],
            "throughput_access_norm": [],
            "bw_valid_count": [],
            "bw_alpha0": [],
        }

        num_agents = int(cfg.num_uav)
        sat_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 0)

        for rollout_idx in range(int(rollout_buffers)):
            rollout_seed = int(seed_base) + int(bundle.update_idx) * 100 + rollout_idx
            _seed_everything(rollout_seed)
            obs, _ = env.reset(seed=rollout_seed)

            obs_e: list[np.ndarray] = []
            next_obs_e: list[np.ndarray] = []
            act_e: list[np.ndarray] = []
            logp_e: list[np.ndarray] = []
            state_e: list[np.ndarray] = []
            next_state_e: list[np.ndarray] = []
            rewards_e: list[float] = []
            values_e: list[float] = []
            terminated_e: list[bool] = []
            truncated_e: list[bool] = []
            sat_indices_e: list[np.ndarray] = []
            danger_target_e: list[np.ndarray] = []
            danger_mask_e: list[np.ndarray] = []
            logp_part_e = {name: [] for name in ("accel", "bw", "sat")}

            for _ in range(int(cfg.buffer_size)):
                obs_list = list(obs.values())
                obs_batch = batch_flatten_obs(obs_list, cfg).astype(np.float32, copy=False)
                obs_step = np.expand_dims(obs_batch, axis=0)
                state_batch = _get_state_batch(env).astype(np.float32, copy=False)

                with torch.no_grad():
                    obs_t = torch.from_numpy(obs_batch)
                    obs_step_t = torch.from_numpy(obs_step)
                    state_t = torch.from_numpy(state_batch)
                    act_out = bundle.actor.act(obs_t, deterministic=False, compute_logprob=False)
                    value_now = float(bundle.critic(state_t, obs_step_t).cpu().numpy().reshape(-1)[0])

                accel_cmd = (
                    act_out.accel.cpu().numpy()
                    if act_out.accel is not None
                    else np.zeros((num_agents, 2), dtype=np.float32)
                )
                bw_exec = act_out.bw_action.cpu().numpy() if act_out.bw_action is not None else None
                sat_exec = act_out.sat_select_mask.cpu().numpy() if act_out.sat_select_mask is not None else None

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
                next_state_batch = np.expand_dims(np.asarray(stats["post_step_global_state"], dtype=np.float32), axis=0)

                reward_aux, _ = _compute_train_reward_adjustment(
                    stats,
                    cfg,
                    bundle.update_idx,
                    bundle.planned_total_updates,
                )
                reward_scalar = float(list(rewards.values())[0]) + float(reward_aux)

                fallback_accel = accel_cmd * float(cfg.a_max)
                accel_exec = np.asarray(stats.get("last_exec_accel", fallback_accel), dtype=np.float32)
                accel_exec_norm = np.clip(accel_exec / max(float(cfg.a_max), 1e-6), -1.0, 1.0).astype(
                    np.float32,
                    copy=False,
                )
                exec_parts = [accel_exec_norm]
                if bool(cfg.enable_bw_action):
                    exec_bw = np.asarray(stats.get("last_exec_bw_alloc", bw_exec), dtype=np.float32)
                    exec_parts.append(exec_bw)
                if not bool(cfg.fixed_satellite_strategy):
                    if sat_exec is None:
                        sat_exec = np.zeros((num_agents, getattr(cfg, "sats_obs_max", 0)), dtype=np.float32)
                    exec_sat = np.asarray(stats.get("last_exec_sat_select_mask", sat_exec), dtype=np.float32)
                    exec_parts.append(exec_sat)
                action_vec = np.concatenate(exec_parts, axis=1).astype(np.float32, copy=False)

                sat_indices = np.asarray(
                    stats.get("last_exec_sat_indices", np.full((num_agents, sat_k), -1, dtype=np.int64)),
                    dtype=np.int64,
                )
                with torch.no_grad():
                    act_t = torch.from_numpy(action_vec)
                    sat_t = torch.from_numpy(sat_indices)
                    logprob_parts, _ = bundle.actor.evaluate_actions_parts(
                        obs_t,
                        act_t,
                        sat_indices=sat_t,
                        out=act_out.dist_out,
                        heads=train_head_names,
                        need_entropy=False,
                    )
                    logprob_total = _sum_selected_parts(logprob_parts, train_heads)

                bw_valid_tensor = act_out.dist_out.get("bw_valid_count")
                if bw_valid_tensor is not None:
                    context["bw_valid_count"].extend(
                        bw_valid_tensor.detach().cpu().numpy().astype(np.float64).reshape(-1).tolist()
                    )
                bw_alpha_tensor = act_out.dist_out.get("bw_alpha")
                bw_mask_tensor = act_out.dist_out.get("bw_valid_mask")
                if bw_alpha_tensor is not None and bw_mask_tensor is not None:
                    bw_alpha = bw_alpha_tensor.detach().cpu().numpy().astype(np.float64)
                    bw_mask = bw_mask_tensor.detach().cpu().numpy().astype(np.float64)
                    alpha0 = np.sum(bw_alpha * (bw_mask > 0.5), axis=1)
                    context["bw_alpha0"].extend(alpha0.astype(np.float64).reshape(-1).tolist())

                reward_parts = dict(stats.get("reward_parts", {}) or {})
                context["gu_drop_ratio_step"].append(float(reward_parts.get("gu_drop_ratio_step", 0.0)))
                context["gu_queue_arrival_steps"].append(float(reward_parts.get("gu_queue_arrival_steps", 0.0)))
                context["throughput_access_norm"].append(float(reward_parts.get("throughput_access_norm", 0.0)))

                danger_target, danger_mask = _build_danger_imitation_step_data(stats, cfg, num_agents)

                obs_e.append(obs_batch.astype(np.float32, copy=False))
                next_obs_e.append(next_obs_batch.astype(np.float32, copy=False))
                act_e.append(action_vec.astype(np.float32, copy=False))
                logp_e.append(logprob_total.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1))
                state_e.append(np.asarray(state_batch[0], dtype=np.float32))
                next_state_e.append(np.asarray(next_state_batch[0], dtype=np.float32))
                rewards_e.append(float(reward_scalar))
                values_e.append(float(value_now))
                terminated_e.append(bool(terminated))
                truncated_e.append(bool(truncated))
                sat_indices_e.append(sat_indices.astype(np.int64, copy=False))
                danger_target_e.append(danger_target.astype(np.float32, copy=False))
                danger_mask_e.append(danger_mask.astype(np.float32, copy=False))
                for head_name in ("accel", "bw", "sat"):
                    if head_name in logprob_parts:
                        logp_part_e[head_name].append(
                            logprob_parts[head_name].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
                        )

                obs = next_obs_for_buffer

            obs_arr_list.append(np.stack(obs_e, axis=0))
            next_obs_arr_list.append(np.stack(next_obs_e, axis=0))
            act_arr_list.append(np.stack(act_e, axis=0))
            logp_arr_list.append(np.stack(logp_e, axis=0))
            state_arr_list.append(np.stack(state_e, axis=0))
            next_state_arr_list.append(np.stack(next_state_e, axis=0))
            rewards_list.append(np.asarray(rewards_e, dtype=np.float32))
            values_list.append(np.asarray(values_e, dtype=np.float32))
            terminated_list.append(np.asarray(terminated_e, dtype=np.float32))
            truncated_list.append(np.asarray(truncated_e, dtype=np.float32))
            sat_indices_list.append(np.stack(sat_indices_e, axis=0))
            danger_target_list.append(np.stack(danger_target_e, axis=0))
            danger_mask_list.append(np.stack(danger_mask_e, axis=0))
            for head_name in ("accel", "bw", "sat"):
                if logp_part_e[head_name]:
                    logp_part_lists[head_name].append(np.stack(logp_part_e[head_name], axis=0))

        obs_arr = np.concatenate(obs_arr_list, axis=0)
        next_obs_arr = np.concatenate(next_obs_arr_list, axis=0)
        act_arr = np.concatenate(act_arr_list, axis=0)
        logp_arr = np.concatenate(logp_arr_list, axis=0)
        state_arr = np.concatenate(state_arr_list, axis=0)
        next_state_arr = np.concatenate(next_state_arr_list, axis=0)
        rewards_arr = np.concatenate(rewards_list, axis=0)
        values_arr = np.concatenate(values_list, axis=0)
        terminated_arr = np.concatenate(terminated_list, axis=0)
        truncated_arr = np.concatenate(truncated_list, axis=0)
        sat_indices_arr = np.concatenate(sat_indices_list, axis=0)
        danger_target_arr = np.concatenate(danger_target_list, axis=0)
        danger_mask_arr = np.concatenate(danger_mask_list, axis=0)

        with torch.no_grad():
            bootstrap_values = bundle.critic(
                torch.from_numpy(next_state_arr),
                torch.from_numpy(next_obs_arr),
            ).detach().cpu().numpy().reshape(-1)
        bootstrap_values = np.where(terminated_arr > 0.5, 0.0, bootstrap_values.astype(np.float32))
        episode_boundaries = np.logical_or(terminated_arr > 0.5, truncated_arr > 0.5)
        adv_raw, _rets = compute_gae(
            rewards_arr.astype(np.float32),
            values_arr.astype(np.float32),
            bootstrap_values.astype(np.float32),
            episode_boundaries,
            float(cfg.gamma),
            float(cfg.gae_lambda),
        )
        adv_clip = float(getattr(cfg, "adv_clip", 5.0) or 0.0)
        adv, adv_stats = _normalize_advantages(np.asarray(adv_raw, dtype=np.float32), adv_clip)

        T, N, _ = obs_arr.shape
        obs_flat = obs_arr.reshape(T * N, -1).astype(np.float32, copy=False)
        act_flat = act_arr.reshape(T * N, -1).astype(np.float32, copy=False)
        logp_flat = logp_arr.reshape(T * N).astype(np.float32, copy=False)
        sat_indices_flat = sat_indices_arr.reshape(T * N, -1).astype(np.int64, copy=False)
        danger_target_flat = danger_target_arr.reshape(T * N, -1).astype(np.float32, copy=False)
        danger_mask_flat = danger_mask_arr.reshape(T * N, -1).astype(np.float32, copy=False)
        adv_flat = np.repeat(adv.astype(np.float32, copy=False), N).astype(np.float32, copy=False)
        logp_part_flat = {
            head_name: (
                np.concatenate(logp_part_lists[head_name], axis=0).reshape(T * N).astype(np.float32, copy=False)
                if logp_part_lists[head_name]
                else None
            )
            for head_name in ("accel", "bw", "sat")
        }
        return {
            "obs_flat": obs_flat,
            "act_flat": act_flat,
            "logp_flat": logp_flat,
            "sat_indices_flat": sat_indices_flat,
            "danger_target_flat": danger_target_flat,
            "danger_mask_flat": danger_mask_flat,
            "adv_flat": adv_flat,
            "logp_part_flat": logp_part_flat,
            "adv_stats": adv_stats,
            "context": {
                "gu_drop_ratio_step_summary": _summary(context["gu_drop_ratio_step"]),
                "gu_queue_arrival_steps_summary": _summary(context["gu_queue_arrival_steps"]),
                "throughput_access_norm_summary": _summary(context["throughput_access_norm"]),
                "bw_valid_count_summary": _summary(context["bw_valid_count"]),
                "bw_alpha0_summary": _summary(context["bw_alpha0"]),
            },
        }
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

def _replay_actor_update(bundle, rollout: dict[str, Any], shuffle_seed: int) -> dict[str, Any]:
    _guard_supported_cfg(bundle)
    cfg = bundle.cfg
    state_payload = torch.load(bundle.train_state_path, map_location="cpu")

    actor = bundle.actor
    actor.load_state_dict(state_payload["actor_state_dict"])
    actor.train()

    train_heads = _train_heads_from_cfg(cfg)
    train_head_names = _selected_train_head_names(train_heads)
    _configure_actor_trainability(actor, cfg, train_heads)

    actor_params = [param for param in actor.parameters() if param.requires_grad]
    actor_optim = torch.optim.Adam(actor_params, lr=float(cfg.actor_lr))
    actor_optim.load_state_dict(state_payload["actor_optimizer_state_dict"])

    obs_flat_t = torch.from_numpy(rollout["obs_flat"])
    act_flat_t = torch.from_numpy(rollout["act_flat"])
    logp_flat_t = torch.from_numpy(rollout["logp_flat"])
    sat_indices_flat_t = torch.from_numpy(rollout["sat_indices_flat"])
    danger_target_flat_t = torch.from_numpy(rollout["danger_target_flat"])
    danger_mask_flat_t = torch.from_numpy(rollout["danger_mask_flat"])
    adv_flat_t = torch.from_numpy(rollout["adv_flat"])
    logprob_part_tensors = {
        head_name: (
            torch.from_numpy(values).to(torch.float32)
            if values is not None
            else None
        )
        for head_name, values in rollout["logp_part_flat"].items()
    }

    batch_size = int(obs_flat_t.shape[0])
    minibatch_size = max(1, batch_size // max(int(getattr(cfg, "num_mini_batch", 1) or 1), 1))
    indices = np.arange(batch_size)
    np.random.seed(int(shuffle_seed))

    danger_imitation_coef = max(float(getattr(cfg, "danger_imitation_coef", 0.0) or 0.0), 0.0)
    danger_imitation_enabled = bool(getattr(cfg, "danger_imitation_enabled", False)) and danger_imitation_coef > 0.0
    entropy_coef = float(getattr(cfg, "entropy_coef", 0.0) or 0.0)

    approx_kls: list[float] = []
    clip_fracs: list[float] = []
    ratio_samples: list[np.ndarray] = []
    log_ratio_abs_samples: list[np.ndarray] = []
    approx_kl_by_head = {name: [] for name in ("accel", "bw", "sat")}
    clip_frac_by_head = {name: [] for name in ("accel", "bw", "sat")}
    log_ratio_mean_by_head = {name: [] for name in ("accel", "bw", "sat")}
    log_ratio_var_by_head = {name: [] for name in ("accel", "bw", "sat")}
    log_ratio_abs_mean_by_head = {name: [] for name in ("accel", "bw", "sat")}

    actor_minibatches_executed = 0
    kl_stop_triggered = 0.0
    stop_early = False

    for _epoch in range(int(getattr(cfg, "ppo_epochs", 1) or 1)):
        np.random.shuffle(indices)
        for start in range(0, batch_size, minibatch_size):
            mb_idx_np = indices[start : start + minibatch_size]
            mb_idx = torch.from_numpy(mb_idx_np).long()
            out = actor.forward(obs_flat_t[mb_idx], required_heads=train_head_names)
            logprob_parts, entropy_parts = actor.evaluate_actions_parts(
                obs_flat_t[mb_idx],
                act_flat_t[mb_idx],
                sat_indices=sat_indices_flat_t[mb_idx],
                out=out,
                heads=train_head_names,
            )

            new_logp = _sum_selected_parts(logprob_parts, train_heads)
            log_ratio = torch.clamp(new_logp - logp_flat_t[mb_idx], -8.0, 8.0)
            ratio = torch.exp(log_ratio)
            surr1 = ratio * adv_flat_t[mb_idx]
            surr2 = torch.clamp(ratio, 1.0 - float(cfg.clip_ratio), 1.0 + float(cfg.clip_ratio)) * adv_flat_t[mb_idx]
            policy_loss = -torch.min(surr1, surr2).mean()

            entropy = _sum_selected_parts(entropy_parts, train_heads)
            for part_name in ("accel", "bw", "sat"):
                old_part_t = logprob_part_tensors.get(part_name)
                if part_name not in logprob_parts or old_part_t is None:
                    continue
                part_log_ratio = torch.clamp(logprob_parts[part_name] - old_part_t[mb_idx], -8.0, 8.0)
                part_ratio = torch.exp(part_log_ratio)
                approx_part = (old_part_t[mb_idx] - logprob_parts[part_name]).mean()
                approx_kl_by_head[part_name].append(float(approx_part.item()))
                clip_frac_by_head[part_name].append(
                    float(((part_ratio - 1.0).abs() > float(cfg.clip_ratio)).float().mean().item())
                )
                log_ratio_mean_by_head[part_name].append(float(part_log_ratio.mean().item()))
                log_ratio_var_by_head[part_name].append(float(part_log_ratio.var(unbiased=False).item()))
                log_ratio_abs_mean_by_head[part_name].append(float(part_log_ratio.abs().mean().item()))

            approx_kl = (logp_flat_t[mb_idx] - new_logp).mean()
            clip_frac = ((ratio - 1.0).abs() > float(cfg.clip_ratio)).float().mean()

            actor_loss = policy_loss - entropy_coef * entropy.mean()
            if danger_imitation_enabled and "mu" in out:
                pred_accel = torch.tanh(out["mu"])
                target_accel = danger_target_flat_t[mb_idx]
                danger_mask = danger_mask_flat_t[mb_idx]
                active = torch.sum(danger_mask, dim=-1) > 0.0
                if torch.any(active):
                    diff = (pred_accel - target_accel) * danger_mask
                    denom = torch.sum(danger_mask, dim=-1) + 1e-9
                    per_row = diff.pow(2).sum(-1) / denom
                    actor_loss = actor_loss + danger_imitation_coef * per_row[active].mean()

            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), float(cfg.max_grad_norm))
            actor_optim.step()

            approx_kls.append(float(approx_kl.item()))
            clip_fracs.append(float(clip_frac.item()))
            ratio_samples.append(ratio.detach().cpu().numpy().reshape(-1))
            log_ratio_abs_samples.append(log_ratio.abs().detach().cpu().numpy().reshape(-1))
            actor_minibatches_executed += 1

            if bool(getattr(cfg, "kl_stop", False)):
                target_kl = float(getattr(cfg, "target_kl", 0.0) or 0.0)
                if target_kl > 0.0 and float(approx_kl.item()) > target_kl:
                    stop_early = True
                    kl_stop_triggered = 1.0
                    break
        if stop_early:
            break

    ratio_vec = np.concatenate(ratio_samples, axis=0) if ratio_samples else np.ones((1,), dtype=np.float32)
    log_ratio_abs_vec = (
        np.concatenate(log_ratio_abs_samples, axis=0) if log_ratio_abs_samples else np.zeros((1,), dtype=np.float32)
    )
    return {
        "actor_minibatches_executed": int(actor_minibatches_executed),
        "kl_stop_triggered": float(kl_stop_triggered),
        "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
        "clip_frac": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
        "ratio_p50": float(np.percentile(ratio_vec, 50.0)),
        "ratio_p90": float(np.percentile(ratio_vec, 90.0)),
        "ratio_p99": float(np.percentile(ratio_vec, 99.0)),
        "log_ratio_abs_mean": float(np.mean(log_ratio_abs_vec)),
        "approx_kl_by_head": {
            head_name: float(np.mean(values)) if values else 0.0 for head_name, values in approx_kl_by_head.items()
        },
        "clip_frac_by_head": {
            head_name: float(np.mean(values)) if values else 0.0 for head_name, values in clip_frac_by_head.items()
        },
        "log_ratio_mean_by_head": {
            head_name: float(np.mean(values)) if values else 0.0 for head_name, values in log_ratio_mean_by_head.items()
        },
        "log_ratio_var_by_head": {
            head_name: float(np.mean(values)) if values else 0.0 for head_name, values in log_ratio_var_by_head.items()
        },
        "log_ratio_abs_mean_by_head": {
            head_name: float(np.mean(values)) if values else 0.0
            for head_name, values in log_ratio_abs_mean_by_head.items()
        },
    }


def _dominant_head(metric_by_head: dict[str, float]) -> str:
    active = {k: float(v) for k, v in metric_by_head.items()}
    if not active:
        return "none"
    return max(active.items(), key=lambda item: item[1])[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=str,
        default=r"runs\phase1_actions\joint_3heads_fading_interference_ka_vsat_joint_puremappo_criticdecoupled_u0600_subproc12_t2_20260331",
    )
    parser.add_argument("--updates", type=str, default="200,250,300,350")
    parser.add_argument("--rollout-buffers", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=62000)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=r"runs\joint_head_replay_timeline_20260401",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_index = _load_metrics_index(run_dir / "metrics.csv")
    checkpoint_eval_index = _load_checkpoint_eval_index(run_dir / "checkpoint_eval.csv")
    updates = _parse_updates(args.updates)

    rows: list[dict[str, Any]] = []
    per_update: list[dict[str, Any]] = []

    for update in updates:
        spec = RunSpec(
            name=f"u{update:04d}",
            run_dir=run_dir,
            actor_ckpt=f"actor_u{update:04d}.pt",
            critic_ckpt=f"critic_u{update:04d}.pt",
            train_state_ckpt=f"train_state_u{update:04d}.pt",
            update_idx=update,
        )
        bundle = _load_bundle(spec)
        rollout = _collect_rollout_payload(bundle, rollout_buffers=int(args.rollout_buffers), seed_base=int(args.seed_base))
        replay = _replay_actor_update(bundle, rollout, shuffle_seed=int(args.seed_base) + update * 1000 + 17)

        metrics_step, metrics_row = _metric_row_for_update(metrics_index, update)
        checkpoint_eval_row = checkpoint_eval_index.get(update)
        row = {
            "update": update,
            "metrics_step": metrics_step,
            "train_metrics_approx_kl": _safe_float(metrics_row, "approx_kl"),
            "train_metrics_clip_frac": _safe_float(metrics_row, "clip_frac"),
            "train_metrics_log_ratio_abs_mean": _safe_float(metrics_row, "log_ratio_abs_mean"),
            "train_metrics_entropy_bw": _safe_float(metrics_row, "entropy_bw"),
            "checkpoint_eval_reward_sum": _safe_float(checkpoint_eval_row, "reward_sum"),
            "checkpoint_eval_processed_ratio": _safe_float(checkpoint_eval_row, "processed_ratio_eval"),
            "checkpoint_eval_drop_ratio": _safe_float(checkpoint_eval_row, "drop_ratio_eval"),
            "checkpoint_eval_pre_backlog": _safe_float(checkpoint_eval_row, "pre_backlog_steps_eval"),
            "replay_actor_minibatches_executed": replay["actor_minibatches_executed"],
            "replay_kl_stop_triggered": replay["kl_stop_triggered"],
            "replay_approx_kl": replay["approx_kl"],
            "replay_clip_frac": replay["clip_frac"],
            "replay_log_ratio_abs_mean": replay["log_ratio_abs_mean"],
            "replay_ratio_p90": replay["ratio_p90"],
            "replay_ratio_p99": replay["ratio_p99"],
            "replay_approx_kl_accel": replay["approx_kl_by_head"]["accel"],
            "replay_approx_kl_bw": replay["approx_kl_by_head"]["bw"],
            "replay_approx_kl_sat": replay["approx_kl_by_head"]["sat"],
            "replay_clip_frac_accel": replay["clip_frac_by_head"]["accel"],
            "replay_clip_frac_bw": replay["clip_frac_by_head"]["bw"],
            "replay_clip_frac_sat": replay["clip_frac_by_head"]["sat"],
            "replay_log_ratio_abs_mean_accel": replay["log_ratio_abs_mean_by_head"]["accel"],
            "replay_log_ratio_abs_mean_bw": replay["log_ratio_abs_mean_by_head"]["bw"],
            "replay_log_ratio_abs_mean_sat": replay["log_ratio_abs_mean_by_head"]["sat"],
            "rollout_bw_valid_count_mean": rollout["context"]["bw_valid_count_summary"]["mean"],
            "rollout_bw_valid_count_p90": rollout["context"]["bw_valid_count_summary"]["p90"],
            "rollout_bw_alpha0_mean": rollout["context"]["bw_alpha0_summary"]["mean"],
            "rollout_bw_alpha0_p90": rollout["context"]["bw_alpha0_summary"]["p90"],
            "rollout_throughput_access_norm_mean": rollout["context"]["throughput_access_norm_summary"]["mean"],
            "rollout_gu_queue_arrival_steps_mean": rollout["context"]["gu_queue_arrival_steps_summary"]["mean"],
            "rollout_gu_drop_ratio_step_mean": rollout["context"]["gu_drop_ratio_step_summary"]["mean"],
            "dominant_head_by_clip": _dominant_head(replay["clip_frac_by_head"]),
            "dominant_head_by_approx_kl": _dominant_head(replay["approx_kl_by_head"]),
            "dominant_head_by_abs_log_ratio": _dominant_head(replay["log_ratio_abs_mean_by_head"]),
        }
        rows.append(row)
        per_update.append(
            {
                "update": update,
                "row": row,
                "replay": replay,
                "rollout_context": rollout["context"],
            }
        )

    early_rows = rows[: min(3, len(rows))]
    summary = {
        "run_dir": str(run_dir),
        "updates": updates,
        "rollout_buffers": int(args.rollout_buffers),
        "seed_base": int(args.seed_base),
        "rows": per_update,
        "dominant_head_counts": {
            "by_clip": dict(Counter(row["dominant_head_by_clip"] for row in rows)),
            "by_approx_kl": dict(Counter(row["dominant_head_by_approx_kl"] for row in rows)),
            "by_abs_log_ratio": dict(Counter(row["dominant_head_by_abs_log_ratio"] for row in rows)),
        },
        "early_window_mean": {
            "updates": [int(row["update"]) for row in early_rows],
            "replay_clip_frac_accel": float(np.mean([row["replay_clip_frac_accel"] for row in early_rows])) if early_rows else 0.0,
            "replay_clip_frac_bw": float(np.mean([row["replay_clip_frac_bw"] for row in early_rows])) if early_rows else 0.0,
            "replay_clip_frac_sat": float(np.mean([row["replay_clip_frac_sat"] for row in early_rows])) if early_rows else 0.0,
            "replay_approx_kl_accel": float(np.mean([row["replay_approx_kl_accel"] for row in early_rows])) if early_rows else 0.0,
            "replay_approx_kl_bw": float(np.mean([row["replay_approx_kl_bw"] for row in early_rows])) if early_rows else 0.0,
            "replay_approx_kl_sat": float(np.mean([row["replay_approx_kl_sat"] for row in early_rows])) if early_rows else 0.0,
            "replay_log_ratio_abs_mean_accel": float(
                np.mean([row["replay_log_ratio_abs_mean_accel"] for row in early_rows])
            )
            if early_rows
            else 0.0,
            "replay_log_ratio_abs_mean_bw": float(
                np.mean([row["replay_log_ratio_abs_mean_bw"] for row in early_rows])
            )
            if early_rows
            else 0.0,
            "replay_log_ratio_abs_mean_sat": float(
                np.mean([row["replay_log_ratio_abs_mean_sat"] for row in early_rows])
            )
            if early_rows
            else 0.0,
        },
    }

    _write_csv(out_dir / "timeline_replay_rows.csv", rows)
    _write_json(out_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
