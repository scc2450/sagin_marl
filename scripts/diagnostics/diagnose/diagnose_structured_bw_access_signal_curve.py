from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import (
    StructuredMAPPO,
    _current_obs_list,
    _heuristic_bw,
    _refresh_stage_obs_cache,
    _split_dataclass_by_counts,
    _to_device_dataclass,
)
from sagin_marl.rl.structured_parallel_eval import (
    batched_policy_accel_actions,
    batched_policy_bw_outputs,
    batched_policy_sat_pair_indices,
)
from sagin_marl.rl.structured_stage_builders import world_state_to_torch
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _safe_corr(x: list[float], y: list[float]) -> float:
    if len(x) <= 1 or len(y) <= 1 or len(x) != len(y):
        return 0.0
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if float(np.std(xa)) <= 1.0e-12 or float(np.std(ya)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def _make_local_drivers(cfg, num_envs: int) -> list[StructuredControlDriver]:
    drivers: list[StructuredControlDriver] = []
    for _ in range(max(int(num_envs), 1)):
        env = make_structured_env(cfg, mode="script")
        drivers.append(as_structured_driver(env))
    return drivers


def _close_local_drivers(drivers: list[StructuredControlDriver]) -> None:
    for driver in drivers:
        close_fn = getattr(driver.env, "close", None)
        if callable(close_fn):
            close_fn()


def _discounted_future_return(
    learner: StructuredMAPPO,
    driver: StructuredControlDriver,
    *,
    deterministic: bool,
    initial_discount: float,
) -> float:
    gamma = float(learner.gamma)
    discount = float(initial_discount)
    total = 0.0
    dummy_buffer = StructuredRolloutBuffer()
    done = False
    while not done:
        step = learner.collect_env_step(driver, dummy_buffer, deterministic=deterministic)
        reward = float(next(iter(step.rewards.values())))
        total += discount * reward
        terminated = bool(next(iter(step.terminations.values())))
        truncated = bool(next(iter(step.truncations.values())))
        done = terminated or truncated
        discount *= gamma
    return float(total)


def _load_run_bundle(cfg, run_dir: Path, update: int, device: torch.device):
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    load_checkpoint_forgiving(bundle.actor, str(run_dir / f"actor_u{int(update):04d}.pt"), map_location=device, strict=True)
    load_checkpoint_forgiving(bundle.critic, str(run_dir / f"critic_u{int(update):04d}.pt"), map_location=device, strict=True)
    bundle.actor.to(device).eval()
    bundle.critic.to(device).eval()
    learner = StructuredMAPPO(
        actor=bundle.actor,
        critic=bundle.critic,
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_ratio=float(getattr(cfg, "clip_ratio", 0.2) or 0.2),
        value_coef=float(getattr(cfg, "value_coef", 0.5) or 0.5),
        entropy_coef=float(getattr(cfg, "entropy_coef", 0.0) or 0.0),
        max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5) or 0.5),
        ppo_epochs=int(getattr(cfg, "ppo_epochs", 1) or 1),
        num_mini_batch=int(getattr(cfg, "num_mini_batch", 1) or 1),
        device=device,
        target_mode="step_level",
        cfg=cfg,
        train_accel=bool(True if getattr(cfg, "train_accel", None) is None else getattr(cfg, "train_accel")),
        train_sat=bool(True if getattr(cfg, "train_sat", None) is None else getattr(cfg, "train_sat")),
        train_bw=bool(True if getattr(cfg, "train_bw", None) is None else getattr(cfg, "train_bw")),
        exec_accel_source=str(getattr(cfg, "exec_accel_source", "policy") or "policy"),
        exec_sat_source=str(getattr(cfg, "exec_sat_source", "policy") or "policy"),
        exec_bw_source=str(getattr(cfg, "exec_bw_source", "policy") or "policy"),
    )
    return bundle.actor, bundle.critic, learner


def _normalize_bw_action(action: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(np.asarray(action, dtype=np.float32))
    valid_arr = np.asarray(valid_mask, dtype=bool)
    for agent_idx in range(int(out.shape[0])):
        valid = valid_arr[agent_idx]
        valid_count = int(np.sum(valid))
        if valid_count <= 0:
            continue
        row = np.asarray(action[agent_idx], dtype=np.float32)[valid]
        row = np.clip(row, 0.0, None)
        row_sum = float(np.sum(row))
        if row_sum <= 1.0e-12:
            out[agent_idx, valid] = 1.0 / float(valid_count)
        else:
            out[agent_idx, valid] = row / row_sum
    return out


def _interp_bw_action(
    policy_action: np.ndarray,
    ref_action: np.ndarray,
    valid_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    mixed = (1.0 - float(alpha)) * np.asarray(policy_action, dtype=np.float32) + float(alpha) * np.asarray(ref_action, dtype=np.float32)
    mixed = np.where(np.asarray(valid_mask, dtype=bool), mixed, 0.0)
    return _normalize_bw_action(mixed, valid_mask)


def _bw_l1_distance(left: np.ndarray, right: np.ndarray, valid_mask: np.ndarray) -> float:
    valid_arr = np.asarray(valid_mask, dtype=bool)
    values: list[float] = []
    for agent_idx in range(int(valid_arr.shape[0])):
        valid = valid_arr[agent_idx]
        valid_count = int(np.sum(valid))
        if valid_count <= 0:
            continue
        left_row = np.asarray(left[agent_idx], dtype=np.float64)[valid]
        right_row = np.asarray(right[agent_idx], dtype=np.float64)[valid]
        values.append(float(0.5 * np.sum(np.abs(left_row - right_row))))
    return _safe_mean(values)


def _reward_parts_summary(driver: StructuredControlDriver) -> dict[str, float]:
    parts = dict(getattr(driver.env, "last_reward_parts", {}) or {})
    return {
        "x_acc": float(parts.get("x_acc", 0.0) or 0.0),
        "x_rel": float(parts.get("x_rel", 0.0) or 0.0),
        "term_pre_drop": float(parts.get("term_pre_drop", 0.0) or 0.0),
        "term_pre_backlog": float(parts.get("term_pre_backlog", 0.0) or 0.0),
        "reward_raw": float(parts.get("reward_raw", 0.0) or 0.0),
    }


def _prepare_policy_bw_step(
    driver: StructuredControlDriver,
    actor,
    learner: StructuredMAPPO,
    cfg,
    device: torch.device,
    *,
    base_policy_deterministic: bool,
    bw_anchor_deterministic: bool,
    reference_source: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    accel_world_state = driver.begin_step()
    accel_action = batched_policy_accel_actions(actor, [accel_world_state], device, base_policy_deterministic)[0]
    sat_world_state = driver.run_accel_stage(accel_action)
    _refresh_stage_obs_cache(driver)
    obs_after_accel = _current_obs_list(driver)
    sat_snapshot = driver.build_sat_stage_snapshot(sat_world_state)
    sat_pair_index = batched_policy_sat_pair_indices(actor, [sat_snapshot], device, base_policy_deterministic)[0]
    sat_action = driver.decode_sat_pair_actions([], sat_pair_index)
    bw_world_state = driver.run_sat_stage(sat_action)
    bw_snapshot = driver.build_bw_stage_snapshot(bw_world_state)
    bw_eval = batched_policy_bw_outputs(actor, [bw_snapshot], device, bw_anchor_deterministic)
    local_state = _split_dataclass_by_counts(bw_eval.local_state, bw_eval.agent_counts)[0]
    valid_mask = (
        ((local_state.user_mask > 0.5) & (local_state.bw_valid_mask > 0.5))
        .detach()
        .cpu()
        .numpy()
    )
    policy_action = _normalize_bw_action(np.asarray(bw_eval.actions[0], dtype=np.float32), valid_mask)
    ref_action = _normalize_bw_action(_heuristic_bw(obs_after_accel, cfg, reference_source), valid_mask)
    with torch.no_grad():
        world_batch = _to_device_dataclass(world_state_to_torch(bw_snapshot.world_state), device)
        state_value = float(learner.critic.value_bw(world_batch).reshape(-1)[0].item())
    return policy_action, ref_action, valid_mask, state_value


def diagnose_update(
    run_dir: Path,
    update: int,
    *,
    state_seed_base: int,
    device: torch.device,
    num_envs: int,
    max_state_samples: int,
    alphas: list[float],
    base_policy_deterministic: bool,
    bw_anchor_deterministic: bool,
    continuation_deterministic: bool,
    reference_source: str,
    progress_every: int,
) -> dict[str, Any]:
    cfg = load_config(str(run_dir / "config_source.yaml"))
    actor, _critic, learner = _load_run_bundle(cfg, run_dir, int(update), device)
    drivers = _make_local_drivers(cfg, int(num_envs))
    slot_active = [True for _ in drivers]
    next_episode = len(drivers)
    state_counter = 0
    action_eval_counter = 0

    sorted_alphas = sorted(float(alpha) for alpha in alphas)
    alpha_keys = [f"{alpha:.2f}" for alpha in sorted_alphas]
    per_alpha_selected: dict[str, list[float]] = {key: [] for key in alpha_keys}
    per_alpha_prefix_reward: dict[str, list[float]] = {key: [] for key in alpha_keys}
    per_alpha_prefix_reward_discounted: dict[str, list[float]] = {key: [] for key in alpha_keys}
    per_alpha_prefix_x_acc_discounted: dict[str, list[float]] = {key: [] for key in alpha_keys}
    per_alpha_prefix_x_rel_discounted: dict[str, list[float]] = {key: [] for key in alpha_keys}
    per_alpha_prefix_term_pre_drop_discounted: dict[str, list[float]] = {key: [] for key in alpha_keys}
    per_alpha_prefix_term_pre_backlog_discounted: dict[str, list[float]] = {key: [] for key in alpha_keys}
    per_alpha_step1_l1_policy: dict[str, list[float]] = {key: [] for key in alpha_keys}
    per_alpha_step2_l1_policy: dict[str, list[float]] = {key: [] for key in alpha_keys}
    per_alpha_prefix_l1_policy: dict[str, list[float]] = {key: [] for key in alpha_keys}
    per_alpha_step1_l1_ref: dict[str, list[float]] = {key: [] for key in alpha_keys}
    per_alpha_step2_l1_ref: dict[str, list[float]] = {key: [] for key in alpha_keys}
    per_alpha_prefix_l1_ref: dict[str, list[float]] = {key: [] for key in alpha_keys}
    state_curve_rows: list[dict[str, Any]] = []
    state_alpha_return_corrs: list[float] = []
    state_alpha_access_corrs: list[float] = []
    best_alpha_values: list[float] = []
    best_alpha_access_values: list[float] = []
    best_return_gain_vs_alpha0: list[float] = []
    best_access_gain_vs_alpha0: list[float] = []

    for slot, driver in enumerate(drivers):
        driver.env.reset(seed=int(state_seed_base) + slot)

    try:
        while any(slot_active) and state_counter < int(max_state_samples):
            active_indices = [slot for slot, is_active in enumerate(slot_active) if is_active]
            if not active_indices:
                break

            accel_world_states = [drivers[slot].begin_step() for slot in active_indices]
            accel_actions = batched_policy_accel_actions(actor, accel_world_states, device, base_policy_deterministic)
            sat_world_states = [
                drivers[slot].run_accel_stage(action)
                for slot, action in zip(active_indices, accel_actions)
            ]
            obs_after_accel_many: list[list[dict[str, np.ndarray]]] = []
            for slot in active_indices:
                _refresh_stage_obs_cache(drivers[slot])
                obs_after_accel_many.append(_current_obs_list(drivers[slot]))
            sat_snapshots = [
                drivers[slot].build_sat_stage_snapshot(world_state)
                for slot, world_state in zip(active_indices, sat_world_states)
            ]
            sat_pair_indices = batched_policy_sat_pair_indices(actor, sat_snapshots, device, base_policy_deterministic)
            sat_actions = [
                drivers[slot].decode_sat_pair_actions([], pair_idx)
                for slot, pair_idx in zip(active_indices, sat_pair_indices)
            ]
            bw_world_states = [
                drivers[slot].run_sat_stage(action)
                for slot, action in zip(active_indices, sat_actions)
            ]
            bw_snapshots = [
                drivers[slot].build_bw_stage_snapshot(world_state)
                for slot, world_state in zip(active_indices, bw_world_states)
            ]
            bw_eval = batched_policy_bw_outputs(actor, bw_snapshots, device, bw_anchor_deterministic)
            bw_state_groups = _split_dataclass_by_counts(bw_eval.local_state, bw_eval.agent_counts)

            for local_slot, slot in enumerate(active_indices):
                if state_counter >= int(max_state_samples):
                    break
                driver = drivers[slot]
                pre_driver = copy.deepcopy(driver)
                local_state = bw_state_groups[local_slot]
                step1_policy_action = np.asarray(bw_eval.actions[local_slot], dtype=np.float32)
                step1_valid_mask = (
                    ((local_state.user_mask > 0.5) & (local_state.bw_valid_mask > 0.5))
                    .detach()
                    .cpu()
                    .numpy()
                )
                step1_policy_action = _normalize_bw_action(step1_policy_action, step1_valid_mask)
                step1_ref_action = _normalize_bw_action(
                    _heuristic_bw(obs_after_accel_many[local_slot], cfg, reference_source),
                    step1_valid_mask,
                )
                with torch.no_grad():
                    world_batch = _to_device_dataclass(world_state_to_torch(bw_snapshots[local_slot].world_state), device)
                    state_value = float(learner.critic.value_bw(world_batch).reshape(-1)[0].item())

                alpha_rows: list[dict[str, Any]] = []
                selected_returns: list[float] = []
                alpha_values_used: list[float] = []
                for alpha in sorted_alphas:
                    alpha_key = f"{float(alpha):.2f}"
                    cf_driver = copy.deepcopy(pre_driver)

                    step1_action = _interp_bw_action(step1_policy_action, step1_ref_action, step1_valid_mask, alpha)
                    step1_result = cf_driver.execute_stage_bw_and_step(step1_action)
                    step1_reward = float(next(iter(step1_result.rewards.values())))
                    step1_parts = _reward_parts_summary(cf_driver)
                    terminated_1 = bool(next(iter(step1_result.terminations.values())))
                    truncated_1 = bool(next(iter(step1_result.truncations.values())))
                    done_1 = terminated_1 or truncated_1

                    prefix_reward = float(step1_reward)
                    prefix_reward_discounted = float(step1_reward)
                    prefix_x_acc_discounted = float(step1_parts["x_acc"])
                    prefix_x_rel_discounted = float(step1_parts["x_rel"])
                    prefix_term_pre_drop_discounted = float(step1_parts["term_pre_drop"])
                    prefix_term_pre_backlog_discounted = float(step1_parts["term_pre_backlog"])
                    prefix_l1_policy = _bw_l1_distance(step1_action, step1_policy_action, step1_valid_mask)
                    prefix_l1_ref = _bw_l1_distance(step1_action, step1_ref_action, step1_valid_mask)
                    step2_policy_l1 = 0.0
                    step2_ref_l1 = 0.0
                    step2_reward = 0.0
                    step2_state_value = 0.0
                    step2_x_acc = 0.0
                    step2_x_rel = 0.0
                    step2_term_pre_drop = 0.0
                    step2_term_pre_backlog = 0.0

                    total_return = float(step1_reward)
                    if not done_1:
                        step2_policy_action, step2_ref_action, step2_valid_mask, step2_state_value = _prepare_policy_bw_step(
                            cf_driver,
                            actor,
                            learner,
                            cfg,
                            device,
                            base_policy_deterministic=bool(base_policy_deterministic),
                            bw_anchor_deterministic=bool(bw_anchor_deterministic),
                            reference_source=str(reference_source),
                        )
                        step2_action = _interp_bw_action(step2_policy_action, step2_ref_action, step2_valid_mask, alpha)
                        step2_result = cf_driver.execute_stage_bw_and_step(step2_action)
                        step2_reward = float(next(iter(step2_result.rewards.values())))
                        step2_parts = _reward_parts_summary(cf_driver)
                        step2_x_acc = float(step2_parts["x_acc"])
                        step2_x_rel = float(step2_parts["x_rel"])
                        step2_term_pre_drop = float(step2_parts["term_pre_drop"])
                        step2_term_pre_backlog = float(step2_parts["term_pre_backlog"])
                        prefix_reward += float(step2_reward)
                        prefix_reward_discounted += float(cfg.gamma) * float(step2_reward)
                        prefix_x_acc_discounted += float(cfg.gamma) * step2_x_acc
                        prefix_x_rel_discounted += float(cfg.gamma) * step2_x_rel
                        prefix_term_pre_drop_discounted += float(cfg.gamma) * step2_term_pre_drop
                        prefix_term_pre_backlog_discounted += float(cfg.gamma) * step2_term_pre_backlog
                        step2_policy_l1 = _bw_l1_distance(step2_action, step2_policy_action, step2_valid_mask)
                        step2_ref_l1 = _bw_l1_distance(step2_action, step2_ref_action, step2_valid_mask)
                        prefix_l1_policy += step2_policy_l1
                        prefix_l1_ref += step2_ref_l1
                        total_return += float(cfg.gamma) * float(step2_reward)
                        terminated_2 = bool(next(iter(step2_result.terminations.values())))
                        truncated_2 = bool(next(iter(step2_result.truncations.values())))
                        done_2 = terminated_2 or truncated_2
                        if not done_2:
                            total_return += _discounted_future_return(
                                learner,
                                cf_driver,
                                deterministic=bool(continuation_deterministic),
                                initial_discount=float(cfg.gamma) * float(cfg.gamma),
                            )

                    per_alpha_selected[alpha_key].append(float(total_return))
                    per_alpha_prefix_reward[alpha_key].append(float(prefix_reward))
                    per_alpha_prefix_reward_discounted[alpha_key].append(float(prefix_reward_discounted))
                    per_alpha_prefix_x_acc_discounted[alpha_key].append(float(prefix_x_acc_discounted))
                    per_alpha_prefix_x_rel_discounted[alpha_key].append(float(prefix_x_rel_discounted))
                    per_alpha_prefix_term_pre_drop_discounted[alpha_key].append(float(prefix_term_pre_drop_discounted))
                    per_alpha_prefix_term_pre_backlog_discounted[alpha_key].append(float(prefix_term_pre_backlog_discounted))
                    per_alpha_step1_l1_policy[alpha_key].append(float(_bw_l1_distance(step1_action, step1_policy_action, step1_valid_mask)))
                    per_alpha_step2_l1_policy[alpha_key].append(float(step2_policy_l1))
                    per_alpha_prefix_l1_policy[alpha_key].append(float(prefix_l1_policy))
                    per_alpha_step1_l1_ref[alpha_key].append(float(_bw_l1_distance(step1_action, step1_ref_action, step1_valid_mask)))
                    per_alpha_step2_l1_ref[alpha_key].append(float(step2_ref_l1))
                    per_alpha_prefix_l1_ref[alpha_key].append(float(prefix_l1_ref))
                    alpha_rows.append(
                        {
                            "alpha": float(alpha),
                            "selected_return": float(total_return),
                            "prefix_reward_undiscounted": float(prefix_reward),
                            "prefix_reward_discounted": float(prefix_reward_discounted),
                            "prefix_x_acc_discounted": float(prefix_x_acc_discounted),
                            "prefix_x_rel_discounted": float(prefix_x_rel_discounted),
                            "prefix_term_pre_drop_discounted": float(prefix_term_pre_drop_discounted),
                            "prefix_term_pre_backlog_discounted": float(prefix_term_pre_backlog_discounted),
                            "step1_x_acc": float(step1_parts["x_acc"]),
                            "step2_x_acc": float(step2_x_acc),
                            "step1_x_rel": float(step1_parts["x_rel"]),
                            "step2_x_rel": float(step2_x_rel),
                            "step1_l1_to_policy": float(per_alpha_step1_l1_policy[alpha_key][-1]),
                            "step2_l1_to_policy": float(step2_policy_l1),
                            "prefix_l1_to_policy": float(prefix_l1_policy),
                            "step1_l1_to_reference": float(per_alpha_step1_l1_ref[alpha_key][-1]),
                            "step2_l1_to_reference": float(step2_ref_l1),
                            "prefix_l1_to_reference": float(prefix_l1_ref),
                            "step2_state_value": float(step2_state_value),
                        }
                    )
                    selected_returns.append(float(total_return))
                    alpha_values_used.append(float(alpha))
                    action_eval_counter += 1

                if alpha_rows:
                    state_counter += 1
                    alpha0_return = float(alpha_rows[0]["selected_return"])
                    alpha0_access = float(alpha_rows[0]["prefix_x_acc_discounted"])
                    best_row = max(alpha_rows, key=lambda row: float(row["selected_return"]))
                    best_access_row = max(alpha_rows, key=lambda row: float(row["prefix_x_acc_discounted"]))
                    state_alpha_return_corrs.append(_safe_corr(alpha_values_used, selected_returns))
                    state_alpha_access_corrs.append(
                        _safe_corr(
                            alpha_values_used,
                            [float(row["prefix_x_acc_discounted"]) for row in alpha_rows],
                        )
                    )
                    best_alpha_values.append(float(best_row["alpha"]))
                    best_alpha_access_values.append(float(best_access_row["alpha"]))
                    best_return_gain_vs_alpha0.append(float(best_row["selected_return"]) - alpha0_return)
                    best_access_gain_vs_alpha0.append(float(best_access_row["prefix_x_acc_discounted"]) - alpha0_access)
                    state_curve_rows.append(
                        {
                            "state_index": int(state_counter),
                            "state_value": float(state_value),
                            "step1_policy_l1_to_reference": float(_bw_l1_distance(step1_policy_action, step1_ref_action, step1_valid_mask)),
                            "best_alpha": float(best_row["alpha"]),
                            "best_alpha_access": float(best_access_row["alpha"]),
                            "best_return_gain_vs_alpha0": float(best_row["selected_return"]) - alpha0_return,
                            "best_access_gain_vs_alpha0": float(best_access_row["prefix_x_acc_discounted"]) - alpha0_access,
                            "alpha_rows": alpha_rows,
                        }
                    )
                    if int(progress_every) > 0 and (state_counter % int(progress_every) == 0):
                        print(
                            json.dumps(
                                {
                                    "update": int(update),
                                    "state_index": int(state_counter),
                                    "action_eval_count": int(action_eval_counter),
                                    "best_alpha": float(best_row["alpha"]),
                                    "best_return_gain_vs_alpha0": float(best_row["selected_return"]) - alpha0_return,
                                    "best_alpha_access": float(best_access_row["alpha"]),
                                    "best_access_gain_vs_alpha0": float(best_access_row["prefix_x_acc_discounted"]) - alpha0_access,
                                },
                                ensure_ascii=False,
                            )
                        )

                live_step = driver.execute_stage_bw_and_step(step1_policy_action)
                terminated = bool(next(iter(live_step.terminations.values())))
                truncated = bool(next(iter(live_step.truncations.values())))
                if terminated or truncated:
                    if next_episode < int(max_state_samples) + len(drivers):
                        driver.env.reset(seed=int(state_seed_base) + int(next_episode))
                        next_episode += 1
                    else:
                        slot_active[slot] = False
    finally:
        _close_local_drivers(drivers)

    alpha_curve = {
        alpha_key: {
            "selected_return": _summarize(per_alpha_selected[alpha_key]),
            "prefix_reward_undiscounted": _summarize(per_alpha_prefix_reward[alpha_key]),
            "prefix_reward_discounted": _summarize(per_alpha_prefix_reward_discounted[alpha_key]),
            "prefix_x_acc_discounted": _summarize(per_alpha_prefix_x_acc_discounted[alpha_key]),
            "prefix_x_rel_discounted": _summarize(per_alpha_prefix_x_rel_discounted[alpha_key]),
            "prefix_term_pre_drop_discounted": _summarize(per_alpha_prefix_term_pre_drop_discounted[alpha_key]),
            "prefix_term_pre_backlog_discounted": _summarize(per_alpha_prefix_term_pre_backlog_discounted[alpha_key]),
            "step1_l1_to_policy": _summarize(per_alpha_step1_l1_policy[alpha_key]),
            "step2_l1_to_policy": _summarize(per_alpha_step2_l1_policy[alpha_key]),
            "prefix_l1_to_policy": _summarize(per_alpha_prefix_l1_policy[alpha_key]),
            "step1_l1_to_reference": _summarize(per_alpha_step1_l1_ref[alpha_key]),
            "step2_l1_to_reference": _summarize(per_alpha_step2_l1_ref[alpha_key]),
            "prefix_l1_to_reference": _summarize(per_alpha_prefix_l1_ref[alpha_key]),
        }
        for alpha_key in alpha_keys
    }
    return {
        "update": int(update),
        "state_seed_base": int(state_seed_base),
        "reference_source": str(reference_source),
        "base_policy_deterministic": bool(base_policy_deterministic),
        "bw_anchor_deterministic": bool(bw_anchor_deterministic),
        "continuation_deterministic": bool(continuation_deterministic),
        "num_envs": int(num_envs),
        "max_state_samples": int(max_state_samples),
        "alphas": [float(alpha) for alpha in sorted_alphas],
        "state_sample_count": int(len(state_curve_rows)),
        "action_eval_count": int(action_eval_counter),
        "alpha_curve": alpha_curve,
        "best_alpha": _summarize(best_alpha_values),
        "best_alpha_access": _summarize(best_alpha_access_values),
        "best_return_gain_vs_alpha0": _summarize(best_return_gain_vs_alpha0),
        "best_access_gain_vs_alpha0": _summarize(best_access_gain_vs_alpha0),
        "state_alpha_return_corr": _summarize(state_alpha_return_corrs),
        "state_alpha_access_corr": _summarize(state_alpha_access_corrs),
        "state_rows": state_curve_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--updates", type=int, nargs="+", default=[100])
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--max-state-samples", type=int, default=4)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--state-seed-base", type=int, default=123450)
    parser.add_argument("--torch-seed-base", type=int, default=939393)
    parser.add_argument("--base-policy-mode", choices=["deterministic", "stochastic"], default="stochastic")
    parser.add_argument("--bw-anchor-mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--continuation-mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--reference-source", choices=["queue_aware", "cluster_center_queue_aware"], default="queue_aware")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--out-name", type=str, default="structured_bw_access_signal_curve_diag.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "num_envs": int(args.num_envs),
        "max_state_samples": int(args.max_state_samples),
        "alphas": [float(alpha) for alpha in args.alphas],
        "state_seed_base": int(args.state_seed_base),
        "torch_seed_base": int(args.torch_seed_base),
        "base_policy_mode": str(args.base_policy_mode),
        "bw_anchor_mode": str(args.bw_anchor_mode),
        "continuation_mode": str(args.continuation_mode),
        "reference_source": str(args.reference_source),
        "device": str(device),
        "updates": {},
    }
    for update in args.updates:
        state_seed = int(args.state_seed_base) + int(update) * 1000
        torch_seed = int(args.torch_seed_base) + int(update)
        _set_all_seeds(torch_seed)
        update_summary = diagnose_update(
            run_dir,
            int(update),
            state_seed_base=state_seed,
            device=device,
            num_envs=int(args.num_envs),
            max_state_samples=int(args.max_state_samples),
            alphas=[float(alpha) for alpha in args.alphas],
            base_policy_deterministic=(args.base_policy_mode == "deterministic"),
            bw_anchor_deterministic=(args.bw_anchor_mode == "deterministic"),
            continuation_deterministic=(args.continuation_mode == "deterministic"),
            reference_source=str(args.reference_source),
            progress_every=int(args.progress_every),
        )
        key = f"u{int(update):04d}"
        summary["updates"][key] = update_summary
        alpha_curve = update_summary["alpha_curve"]
        alpha0_key = f"{float(sorted(update_summary['alphas'])[0]):.2f}"
        alpha1_key = f"{float(sorted(update_summary['alphas'])[-1]):.2f}"
        print(
            json.dumps(
                {
                    "update": int(update),
                    "state_sample_count": int(update_summary["state_sample_count"]),
                    "action_eval_count": int(update_summary["action_eval_count"]),
                    "alpha0_selected_return_mean": float(alpha_curve[alpha0_key]["selected_return"]["mean"]),
                    "alpha1_selected_return_mean": float(alpha_curve[alpha1_key]["selected_return"]["mean"]),
                    "alpha0_prefix_x_acc_discounted_mean": float(alpha_curve[alpha0_key]["prefix_x_acc_discounted"]["mean"]),
                    "alpha1_prefix_x_acc_discounted_mean": float(alpha_curve[alpha1_key]["prefix_x_acc_discounted"]["mean"]),
                    "best_return_gain_vs_alpha0_mean": float(update_summary["best_return_gain_vs_alpha0"]["mean"]),
                    "best_access_gain_vs_alpha0_mean": float(update_summary["best_access_gain_vs_alpha0"]["mean"]),
                },
                ensure_ascii=False,
            )
        )

    out_path = run_dir / str(args.out_name)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
