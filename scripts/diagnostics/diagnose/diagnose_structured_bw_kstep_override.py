from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(ROOT, "sagin_marl")):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        raise RuntimeError("Could not locate repository root.")
    ROOT = parent
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


def _discounted_future_trace(
    learner: StructuredMAPPO,
    driver: StructuredControlDriver,
    *,
    deterministic: bool,
    initial_discount: float,
    max_trace_steps: int,
) -> tuple[list[float], float]:
    gamma = float(learner.gamma)
    discount = float(initial_discount)
    total = 0.0
    traced: list[float] = []
    dummy_buffer = StructuredRolloutBuffer()
    done = False
    step_index = 0
    while not done:
        step = learner.collect_env_step(driver, dummy_buffer, deterministic=deterministic)
        reward = float(next(iter(step.rewards.values())))
        discounted_reward = discount * reward
        total += discounted_reward
        if step_index < int(max_trace_steps):
            traced.append(float(discounted_reward))
        terminated = bool(next(iter(step.terminations.values())))
        truncated = bool(next(iter(step.truncations.values())))
        done = terminated or truncated
        discount *= gamma
        step_index += 1
    traced_total = float(np.sum(np.asarray(traced, dtype=np.float64))) if traced else 0.0
    tail = float(total - traced_total)
    return traced, tail


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


def _prepare_policy_bw_step(
    driver: StructuredControlDriver,
    actor,
    learner: StructuredMAPPO,
    cfg,
    device: torch.device,
    *,
    deterministic: bool,
    reference_source: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    accel_world_state = driver.begin_step()
    accel_action = batched_policy_accel_actions(actor, [accel_world_state], device, deterministic)[0]
    sat_world_state = driver.run_accel_stage(accel_action)
    _refresh_stage_obs_cache(driver)
    obs_after_accel = _current_obs_list(driver)
    sat_snapshot = driver.build_sat_stage_snapshot(sat_world_state)
    sat_pair_index = batched_policy_sat_pair_indices(actor, [sat_snapshot], device, deterministic)[0]
    sat_action = driver.decode_sat_pair_actions([], sat_pair_index)
    bw_world_state = driver.run_sat_stage(sat_action)
    bw_snapshot = driver.build_bw_stage_snapshot(bw_world_state)
    bw_eval = batched_policy_bw_outputs(actor, [bw_snapshot], device, deterministic)
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


def _execute_prefix_override(
    pre_driver: StructuredControlDriver,
    learner: StructuredMAPPO,
    actor,
    cfg,
    device: torch.device,
    *,
    k_override: int,
    current_policy_action: np.ndarray,
    current_ref_action: np.ndarray,
    base_policy_deterministic: bool,
    continuation_deterministic: bool,
    reference_source: str,
    max_trace_steps: int,
) -> tuple[float, float, int, list[float], float]:
    cf_driver = copy.deepcopy(pre_driver)
    gamma = float(cfg.gamma)
    discount = 1.0
    total_return = 0.0
    prefix_reward = 0.0
    bw_steps_executed = 0
    trace_rewards: list[float] = []
    tail_discounted_return = 0.0

    first_action = current_ref_action if int(k_override) >= 1 else current_policy_action
    step_result = cf_driver.execute_stage_bw_and_step(first_action)
    reward = float(next(iter(step_result.rewards.values())))
    discounted_reward = discount * reward
    total_return += discounted_reward
    prefix_reward += reward
    if len(trace_rewards) < int(max_trace_steps):
        trace_rewards.append(float(discounted_reward))
    else:
        tail_discounted_return += float(discounted_reward)
    discount *= gamma
    bw_steps_executed += 1
    terminated = bool(next(iter(step_result.terminations.values())))
    truncated = bool(next(iter(step_result.truncations.values())))
    done = terminated or truncated

    while (not done) and bw_steps_executed < int(k_override):
        _policy_action, ref_action, _valid_mask, _state_value = _prepare_policy_bw_step(
            cf_driver,
            actor,
            learner,
            cfg,
            device,
            deterministic=bool(base_policy_deterministic),
            reference_source=reference_source,
        )
        step_result = cf_driver.execute_stage_bw_and_step(ref_action)
        reward = float(next(iter(step_result.rewards.values())))
        discounted_reward = discount * reward
        total_return += discounted_reward
        prefix_reward += reward
        if len(trace_rewards) < int(max_trace_steps):
            trace_rewards.append(float(discounted_reward))
        else:
            tail_discounted_return += float(discounted_reward)
        discount *= gamma
        bw_steps_executed += 1
        terminated = bool(next(iter(step_result.terminations.values())))
        truncated = bool(next(iter(step_result.truncations.values())))
        done = terminated or truncated

    if not done:
        future_trace, future_tail = _discounted_future_trace(
            learner,
            cf_driver,
            deterministic=bool(continuation_deterministic),
            initial_discount=float(discount),
            max_trace_steps=max(int(max_trace_steps) - len(trace_rewards), 0),
        )
        trace_rewards.extend(future_trace)
        tail_discounted_return += float(future_tail)
        total_return += float(np.sum(np.asarray(future_trace, dtype=np.float64))) + float(future_tail)
    return float(total_return), float(prefix_reward), int(bw_steps_executed), trace_rewards, float(tail_discounted_return)


def diagnose_update(
    run_dir: Path,
    update: int,
    *,
    state_seed_base: int,
    device: torch.device,
    num_envs: int,
    max_state_samples: int,
    override_ks: list[int],
    base_policy_deterministic: bool,
    bw_anchor_deterministic: bool,
    continuation_deterministic: bool,
    reference_source: str,
    progress_every: int,
    max_trace_steps: int,
) -> dict[str, Any]:
    cfg = load_config(str(run_dir / "config_source.yaml"))
    actor, _critic, learner = _load_run_bundle(cfg, run_dir, int(update), device)
    drivers = _make_local_drivers(cfg, int(num_envs))
    slot_active = [True for _ in drivers]
    next_episode = len(drivers)
    state_counter = 0
    action_eval_counter = 0

    sorted_ks = sorted({max(int(k), 0) for k in override_ks})
    per_k_selected: dict[str, list[float]] = {str(k): [] for k in sorted_ks}
    per_k_prefix_reward: dict[str, list[float]] = {str(k): [] for k in sorted_ks}
    per_k_steps_executed: dict[str, list[float]] = {str(k): [] for k in sorted_ks}
    per_k_trace_discounted: dict[str, list[list[float]]] = {str(k): [[] for _ in range(int(max_trace_steps))] for k in sorted_ks}
    per_k_trace_tail: dict[str, list[float]] = {str(k): [] for k in sorted_ks}
    state_curve_rows: list[dict[str, Any]] = []
    state_k_return_corrs: list[float] = []
    best_k_values: list[float] = []
    best_return_gain_vs_k0: list[float] = []

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
                policy_anchor = np.asarray(bw_eval.actions[local_slot], dtype=np.float32)
                valid_mask = (
                    ((local_state.user_mask > 0.5) & (local_state.bw_valid_mask > 0.5))
                    .detach()
                    .cpu()
                    .numpy()
                )
                policy_anchor = _normalize_bw_action(policy_anchor, valid_mask)
                ref_action = _normalize_bw_action(
                    _heuristic_bw(obs_after_accel_many[local_slot], cfg, reference_source),
                    valid_mask,
                )
                with torch.no_grad():
                    world_batch = _to_device_dataclass(world_state_to_torch(bw_snapshots[local_slot].world_state), device)
                    state_value = float(learner.critic.value_bw(world_batch).reshape(-1)[0].item())

                k_rows: list[dict[str, float]] = []
                selected_returns: list[float] = []
                k_values_used: list[float] = []
                for k_override in sorted_ks:
                    selected_return, prefix_reward, steps_executed, trace_rewards, tail_discounted_return = _execute_prefix_override(
                        pre_driver,
                        learner,
                        actor,
                        cfg,
                        device,
                        k_override=int(k_override),
                        current_policy_action=policy_anchor,
                        current_ref_action=ref_action,
                        base_policy_deterministic=bool(base_policy_deterministic),
                        continuation_deterministic=bool(continuation_deterministic),
                        reference_source=str(reference_source),
                        max_trace_steps=int(max_trace_steps),
                    )
                    k_key = str(int(k_override))
                    per_k_selected[k_key].append(float(selected_return))
                    per_k_prefix_reward[k_key].append(float(prefix_reward))
                    per_k_steps_executed[k_key].append(float(steps_executed))
                    for trace_idx in range(int(max_trace_steps)):
                        value = float(trace_rewards[trace_idx]) if trace_idx < len(trace_rewards) else 0.0
                        per_k_trace_discounted[k_key][trace_idx].append(value)
                    per_k_trace_tail[k_key].append(float(tail_discounted_return))
                    k_rows.append(
                        {
                            "k_override": int(k_override),
                            "selected_return": float(selected_return),
                            "prefix_reward_undiscounted": float(prefix_reward),
                            "override_steps_executed": int(steps_executed),
                            "trace_discounted_rewards": [float(v) for v in trace_rewards],
                            "tail_discounted_return": float(tail_discounted_return),
                        }
                    )
                    selected_returns.append(float(selected_return))
                    k_values_used.append(float(k_override))
                    action_eval_counter += 1

                if k_rows:
                    state_counter += 1
                    k0_return = float(k_rows[0]["selected_return"])
                    best_row = max(k_rows, key=lambda row: float(row["selected_return"]))
                    state_k_return_corrs.append(_safe_corr(k_values_used, selected_returns))
                    best_k_values.append(float(best_row["k_override"]))
                    best_return_gain_vs_k0.append(float(best_row["selected_return"]) - k0_return)
                    state_curve_rows.append(
                        {
                            "state_index": int(state_counter),
                            "state_value": float(state_value),
                            "policy_anchor_l1_to_reference": float(_bw_l1_distance(policy_anchor, ref_action, valid_mask)),
                            "best_k": int(best_row["k_override"]),
                            "best_return_gain_vs_k0": float(best_row["selected_return"]) - k0_return,
                            "k_rows": k_rows,
                        }
                    )
                    if int(progress_every) > 0 and (state_counter % int(progress_every) == 0):
                        print(
                            json.dumps(
                                {
                                    "update": int(update),
                                    "state_index": int(state_counter),
                                    "action_eval_count": int(action_eval_counter),
                                    "best_k": int(best_row["k_override"]),
                                    "best_return_gain_vs_k0": float(best_row["selected_return"]) - k0_return,
                                },
                                ensure_ascii=False,
                            )
                        )

                live_step = driver.execute_stage_bw_and_step(policy_anchor)
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

    k_curve = {
        k_key: {
            "selected_return": _summarize(per_k_selected[k_key]),
            "prefix_reward_undiscounted": _summarize(per_k_prefix_reward[k_key]),
            "override_steps_executed": _summarize(per_k_steps_executed[k_key]),
            "trace_discounted_rewards": {
                f"step_{trace_idx + 1}": _summarize(per_k_trace_discounted[k_key][trace_idx])
                for trace_idx in range(int(max_trace_steps))
            },
            "tail_discounted_return": _summarize(per_k_trace_tail[k_key]),
        }
        for k_key in per_k_selected
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
        "override_ks": [int(k) for k in sorted_ks],
        "max_trace_steps": int(max_trace_steps),
        "state_sample_count": int(len(state_curve_rows)),
        "action_eval_count": int(action_eval_counter),
        "k_curve": k_curve,
        "best_k": _summarize(best_k_values),
        "best_return_gain_vs_k0": _summarize(best_return_gain_vs_k0),
        "state_k_return_corr": _summarize(state_k_return_corrs),
        "state_rows": state_curve_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--updates", type=int, nargs="+", default=[100])
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--max-state-samples", type=int, default=4)
    parser.add_argument("--override-ks", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument("--state-seed-base", type=int, default=123450)
    parser.add_argument("--torch-seed-base", type=int, default=929292)
    parser.add_argument("--base-policy-mode", choices=["deterministic", "stochastic"], default="stochastic")
    parser.add_argument("--bw-anchor-mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--continuation-mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--reference-source", choices=["queue_aware", "cluster_center_queue_aware"], default="queue_aware")
    parser.add_argument("--max-trace-steps", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--out-name", type=str, default="structured_bw_kstep_override_diag.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "num_envs": int(args.num_envs),
        "max_state_samples": int(args.max_state_samples),
        "override_ks": [int(k) for k in args.override_ks],
        "state_seed_base": int(args.state_seed_base),
        "torch_seed_base": int(args.torch_seed_base),
        "base_policy_mode": str(args.base_policy_mode),
        "bw_anchor_mode": str(args.bw_anchor_mode),
        "continuation_mode": str(args.continuation_mode),
        "reference_source": str(args.reference_source),
        "max_trace_steps": int(args.max_trace_steps),
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
            override_ks=[int(k) for k in args.override_ks],
            base_policy_deterministic=(args.base_policy_mode == "deterministic"),
            bw_anchor_deterministic=(args.bw_anchor_mode == "deterministic"),
            continuation_deterministic=(args.continuation_mode == "deterministic"),
            reference_source=str(args.reference_source),
            progress_every=int(args.progress_every),
            max_trace_steps=int(args.max_trace_steps),
        )
        key = f"u{int(update):04d}"
        summary["updates"][key] = update_summary
        k_curve = update_summary["k_curve"]
        k0_key = str(int(sorted(update_summary["override_ks"])[0]))
        kmax_key = str(int(sorted(update_summary["override_ks"])[-1]))
        print(
            json.dumps(
                {
                    "update": int(update),
                    "state_sample_count": int(update_summary["state_sample_count"]),
                    "action_eval_count": int(update_summary["action_eval_count"]),
                    "k0_selected_return_mean": float(k_curve[k0_key]["selected_return"]["mean"]),
                    "kmax_selected_return_mean": float(k_curve[kmax_key]["selected_return"]["mean"]),
                    "best_return_gain_vs_k0_mean": float(update_summary["best_return_gain_vs_k0"]["mean"]),
                },
                ensure_ascii=False,
            )
        )

    out_path = run_dir / str(args.out_name)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
