from __future__ import annotations

import argparse
import copy
import json
import math
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
from sagin_marl.rl.structured_mappo import StructuredMAPPO, _split_dataclass_by_counts, _to_device_dataclass
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


def _rankdata_avg(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    order = np.argsort(arr, kind="mergesort")
    ranks = np.zeros_like(arr, dtype=np.float64)
    i = 0
    while i < arr.size:
        j = i + 1
        while j < arr.size and arr[order[j]] == arr[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _safe_spearman(x: list[float], y: list[float]) -> float:
    if len(x) <= 1 or len(y) <= 1 or len(x) != len(y):
        return 0.0
    return _safe_corr(_rankdata_avg(x).tolist(), _rankdata_avg(y).tolist())


def _quantile_delta(order_values: list[float], target_values: list[float], *, lower_is_flatter: bool, frac: float = 0.25) -> float:
    if not order_values or len(order_values) != len(target_values):
        return 0.0
    n = len(order_values)
    k = max(1, int(round(float(frac) * n)))
    order = np.argsort(np.asarray(order_values, dtype=np.float64))
    target = np.asarray(target_values, dtype=np.float64)
    if lower_is_flatter:
        flatter = target[order[:k]]
        sharper = target[order[-k:]]
    else:
        flatter = target[order[-k:]]
        sharper = target[order[:k]]
    return float(np.mean(flatter) - np.mean(sharper))


def _top2_margin(row: np.ndarray) -> float:
    if row.size <= 1:
        return 0.0
    top2 = np.partition(row, -2)[-2:]
    return float(np.max(top2) - np.min(top2))


def _row_shape_metrics(row: np.ndarray) -> dict[str, float]:
    k = int(row.size)
    if k <= 0:
        return {
            "valid_count": 0.0,
            "top1": 0.0,
            "top1_excess": 0.0,
            "top1_margin": 0.0,
            "l1_to_uniform": 0.0,
            "entropy_norm": 0.0,
            "entropy_gap_norm": 0.0,
        }
    uniform = np.full((k,), 1.0 / max(k, 1), dtype=np.float64)
    clipped = np.clip(np.asarray(row, dtype=np.float64), 1.0e-12, 1.0)
    top1 = float(np.max(clipped))
    entropy = float(-np.sum(clipped * np.log(clipped)))
    entropy_norm = 1.0 if k <= 1 else float(entropy / max(math.log(k), 1.0e-12))
    return {
        "valid_count": float(k),
        "top1": top1,
        "top1_excess": float(top1 - 1.0 / max(k, 1)),
        "top1_margin": _top2_margin(clipped),
        "l1_to_uniform": float(0.5 * np.sum(np.abs(clipped - uniform))),
        "entropy_norm": entropy_norm,
        "entropy_gap_norm": float(1.0 - entropy_norm),
    }


def _sample_shape_metrics(action: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    metric_lists: dict[str, list[float]] = {
        "valid_count": [],
        "top1": [],
        "top1_excess": [],
        "top1_margin": [],
        "l1_to_uniform": [],
        "entropy_norm": [],
        "entropy_gap_norm": [],
    }
    active_agents = 0
    for agent_idx in range(int(action.shape[0])):
        valid = np.asarray(valid_mask[agent_idx], dtype=bool)
        if int(np.sum(valid)) <= 1:
            continue
        row = np.asarray(action[agent_idx], dtype=np.float64)[valid]
        row_sum = float(np.sum(row))
        if row_sum <= 1.0e-12:
            continue
        row = row / row_sum
        row_metrics = _row_shape_metrics(row)
        for key, value in row_metrics.items():
            metric_lists[key].append(float(value))
        active_agents += 1
    out = {f"{key}_mean": _safe_mean(values) for key, values in metric_lists.items()}
    out["active_agent_count"] = float(active_agents)
    return out


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
) -> float:
    gamma = float(learner.gamma)
    discount = gamma
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


def _history_tail_from_csv(run_dir: Path, update: int) -> dict[str, float]:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return {}
    import csv

    target_row = None
    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(float(row.get("update", 0) or 0.0)) == int(update):
                    target_row = row
            except ValueError:
                continue
    if target_row is None:
        return {}
    keep_keys = (
        "bw_grad_ratio_aux_to_policy",
        "bw_loc_head_grad_ratio_aux_to_policy",
        "clip_frac_bw",
        "approx_kl_bw",
        "entropy_bw",
    )
    out: dict[str, float] = {}
    for key in keep_keys:
        if key in target_row:
            try:
                out[key] = float(target_row[key])
            except ValueError:
                pass
    return out


def diagnose_update(
    run_dir: Path,
    update: int,
    *,
    state_seed_base: int,
    device: torch.device,
    num_envs: int,
    max_state_samples: int,
    actions_per_state: int,
    base_policy_deterministic: bool,
    continuation_deterministic: bool,
    return_mode: str,
    progress_every: int,
) -> dict[str, Any]:
    cfg = load_config(str(run_dir / "config_source.yaml"))
    actor, _critic, learner = _load_run_bundle(cfg, run_dir, int(update), device)
    drivers = _make_local_drivers(cfg, int(num_envs))
    slot_active = [True for _ in drivers]
    next_episode = len(drivers)
    state_counter = 0
    action_eval_counter = 0

    metric_specs = [
        ("l1_to_uniform_mean", True),
        ("entropy_norm_mean", False),
        ("top1_excess_mean", True),
        ("top1_margin_mean", True),
    ]
    metric_rollup: dict[str, dict[str, list[float]]] = {
        name: {
            "corr_selected_return": [],
            "corr_immediate_reward": [],
            "corr_one_step_boot": [],
            "delta_selected_return_flatter_minus_sharper": [],
            "delta_immediate_reward_flatter_minus_sharper": [],
            "delta_one_step_boot_flatter_minus_sharper": [],
        }
        for name, _ in metric_specs
    }
    state_value_list: list[float] = []
    state_avg_selected_return_list: list[float] = []
    state_avg_immediate_reward_list: list[float] = []
    state_avg_one_step_boot_list: list[float] = []
    state_avg_l1_list: list[float] = []
    state_avg_entropy_norm_list: list[float] = []
    state_selected_minus_value_list: list[float] = []
    state_best_selected_minus_mean_list: list[float] = []
    valid_count_mean_list: list[float] = []
    sampled_valid_agent_count_list: list[float] = []
    per_state_action_count_list: list[float] = []

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
            bw_eval = batched_policy_bw_outputs(actor, bw_snapshots, device, base_policy_deterministic)
            bw_state_groups = _split_dataclass_by_counts(bw_eval.local_state, bw_eval.agent_counts)

            for local_slot, slot in enumerate(active_indices):
                if state_counter >= int(max_state_samples):
                    break
                driver = drivers[slot]
                pre_driver = copy.deepcopy(driver)
                local_state = bw_state_groups[local_slot]
                base_action_env = np.asarray(bw_eval.actions[local_slot], dtype=np.float32)
                valid_mask = (
                    ((local_state.user_mask > 0.5) & (local_state.bw_valid_mask > 0.5))
                    .detach()
                    .cpu()
                    .numpy()
                )
                with torch.no_grad():
                    world_batch = _to_device_dataclass(world_state_to_torch(bw_snapshots[local_slot].world_state), device)
                    state_value = float(learner.critic.value_bw(world_batch).reshape(-1)[0].item())

                per_state_metrics: dict[str, list[float]] = {name: [] for name, _ in metric_specs}
                per_state_selected_return: list[float] = []
                per_state_immediate_reward: list[float] = []
                per_state_one_step_boot: list[float] = []
                per_state_valid_count: list[float] = []
                per_state_active_agent_count: list[float] = []

                for _ in range(int(actions_per_state)):
                    with torch.inference_mode():
                        sample_out = actor.act_bw(local_state, deterministic=False)
                    action = sample_out.action.detach().cpu().numpy()
                    shape_metrics = _sample_shape_metrics(action, valid_mask)
                    if float(shape_metrics["active_agent_count"]) <= 0.0:
                        continue
                    cf_driver = copy.deepcopy(pre_driver)
                    step_result, next_world_state = cf_driver.execute_stage_bw_and_prepare_next_accel(action)
                    immediate_reward = float(next(iter(step_result.rewards.values())))
                    next_value = float(learner.bootstrap_value(next_world_state))
                    one_step_boot = immediate_reward + float(cfg.gamma) * next_value
                    if return_mode == "full_episode":
                        selected_return = immediate_reward + _discounted_future_return(
                            learner,
                            cf_driver,
                            deterministic=bool(continuation_deterministic),
                        )
                    else:
                        selected_return = one_step_boot
                    per_state_selected_return.append(float(selected_return))
                    per_state_immediate_reward.append(float(immediate_reward))
                    per_state_one_step_boot.append(float(one_step_boot))
                    per_state_valid_count.append(float(shape_metrics["valid_count_mean"]))
                    per_state_active_agent_count.append(float(shape_metrics["active_agent_count"]))
                    for name, _ in metric_specs:
                        per_state_metrics[name].append(float(shape_metrics[name]))
                    action_eval_counter += 1

                if len(per_state_selected_return) >= 2:
                    state_counter += 1
                    state_value_list.append(float(state_value))
                    state_avg_selected_return_list.append(_safe_mean(per_state_selected_return))
                    state_avg_immediate_reward_list.append(_safe_mean(per_state_immediate_reward))
                    state_avg_one_step_boot_list.append(_safe_mean(per_state_one_step_boot))
                    state_avg_l1_list.append(_safe_mean(per_state_metrics["l1_to_uniform_mean"]))
                    state_avg_entropy_norm_list.append(_safe_mean(per_state_metrics["entropy_norm_mean"]))
                    state_selected_minus_value_list.append(_safe_mean(per_state_selected_return) - float(state_value))
                    state_best_selected_minus_mean_list.append(
                        float(np.max(np.asarray(per_state_selected_return, dtype=np.float64))) - _safe_mean(per_state_selected_return)
                    )
                    valid_count_mean_list.append(_safe_mean(per_state_valid_count))
                    sampled_valid_agent_count_list.append(_safe_mean(per_state_active_agent_count))
                    per_state_action_count_list.append(float(len(per_state_selected_return)))

                    for name, lower_is_flatter in metric_specs:
                        metric_values = per_state_metrics[name]
                        metric_rollup[name]["corr_selected_return"].append(
                            _safe_spearman(metric_values, per_state_selected_return)
                        )
                        metric_rollup[name]["corr_immediate_reward"].append(
                            _safe_spearman(metric_values, per_state_immediate_reward)
                        )
                        metric_rollup[name]["corr_one_step_boot"].append(
                            _safe_spearman(metric_values, per_state_one_step_boot)
                        )
                        metric_rollup[name]["delta_selected_return_flatter_minus_sharper"].append(
                            _quantile_delta(metric_values, per_state_selected_return, lower_is_flatter=lower_is_flatter, frac=0.25)
                        )
                        metric_rollup[name]["delta_immediate_reward_flatter_minus_sharper"].append(
                            _quantile_delta(metric_values, per_state_immediate_reward, lower_is_flatter=lower_is_flatter, frac=0.25)
                        )
                        metric_rollup[name]["delta_one_step_boot_flatter_minus_sharper"].append(
                            _quantile_delta(metric_values, per_state_one_step_boot, lower_is_flatter=lower_is_flatter, frac=0.25)
                        )
                    if int(progress_every) > 0 and (state_counter % int(progress_every) == 0):
                        print(
                            json.dumps(
                                {
                                    "update": int(update),
                                    "state_index": int(state_counter),
                                    "action_eval_count": int(action_eval_counter),
                                    "selected_return_mode": str(return_mode),
                                    "avg_l1_to_uniform": float(_safe_mean(per_state_metrics["l1_to_uniform_mean"])),
                                    "avg_selected_return": float(_safe_mean(per_state_selected_return)),
                                },
                                ensure_ascii=False,
                            )
                        )

                live_step = driver.execute_stage_bw_and_step(base_action_env)
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

    metric_summary = {
        name: {subkey: _summarize(values) for subkey, values in parts.items()}
        for name, parts in metric_rollup.items()
    }
    state_level = {
        "state_value": _summarize(state_value_list),
        "avg_selected_return": _summarize(state_avg_selected_return_list),
        "avg_immediate_reward": _summarize(state_avg_immediate_reward_list),
        "avg_one_step_boot": _summarize(state_avg_one_step_boot_list),
        "avg_l1_to_uniform": _summarize(state_avg_l1_list),
        "avg_entropy_norm": _summarize(state_avg_entropy_norm_list),
        "avg_selected_minus_value": _summarize(state_selected_minus_value_list),
        "best_selected_minus_mean": _summarize(state_best_selected_minus_mean_list),
        "valid_count_mean": _summarize(valid_count_mean_list),
        "active_agent_count_mean": _summarize(sampled_valid_agent_count_list),
        "actions_per_state_realized": _summarize(per_state_action_count_list),
        "corr_avg_l1_to_uniform_vs_state_value": _safe_corr(state_avg_l1_list, state_value_list),
        "corr_avg_entropy_norm_vs_state_value": _safe_corr(state_avg_entropy_norm_list, state_value_list),
        "corr_avg_l1_to_uniform_vs_avg_selected_return": _safe_corr(state_avg_l1_list, state_avg_selected_return_list),
        "corr_avg_entropy_norm_vs_avg_selected_return": _safe_corr(state_avg_entropy_norm_list, state_avg_selected_return_list),
    }
    return {
        "update": int(update),
        "state_seed_base": int(state_seed_base),
        "base_policy_deterministic": bool(base_policy_deterministic),
        "continuation_deterministic": bool(continuation_deterministic),
        "return_mode": str(return_mode),
        "num_envs": int(num_envs),
        "max_state_samples": int(max_state_samples),
        "actions_per_state_requested": int(actions_per_state),
        "state_sample_count": int(len(state_value_list)),
        "action_eval_count": int(action_eval_counter),
        "logged_metrics": _history_tail_from_csv(run_dir, int(update)),
        "within_state_metrics": metric_summary,
        "state_level": state_level,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--updates", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--max-state-samples", type=int, default=8)
    parser.add_argument("--actions-per-state", type=int, default=4)
    parser.add_argument("--state-seed-base", type=int, default=123450)
    parser.add_argument("--torch-seed-base", type=int, default=717171)
    parser.add_argument("--base-policy-mode", choices=["deterministic", "stochastic"], default="stochastic")
    parser.add_argument("--continuation-mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--return-mode", choices=["one_step_boot", "full_episode"], default="one_step_boot")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--out-name", type=str, default="structured_bw_within_state_returns_diag.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "num_envs": int(args.num_envs),
        "max_state_samples": int(args.max_state_samples),
        "actions_per_state": int(args.actions_per_state),
        "state_seed_base": int(args.state_seed_base),
        "torch_seed_base": int(args.torch_seed_base),
        "base_policy_mode": str(args.base_policy_mode),
        "continuation_mode": str(args.continuation_mode),
        "return_mode": str(args.return_mode),
        "device": str(device),
        "updates": {},
    }
    for update in args.updates:
        _set_all_seeds(int(args.torch_seed_base) + int(update))
        update_summary = diagnose_update(
            run_dir,
            int(update),
            state_seed_base=int(args.state_seed_base) + int(update) * 1000,
            device=device,
            num_envs=int(args.num_envs),
            max_state_samples=int(args.max_state_samples),
            actions_per_state=int(args.actions_per_state),
            base_policy_deterministic=(args.base_policy_mode == "deterministic"),
            continuation_deterministic=(args.continuation_mode == "deterministic"),
            return_mode=str(args.return_mode),
            progress_every=int(args.progress_every),
        )
        summary["updates"][f"u{int(update):04d}"] = update_summary
        l1 = update_summary["within_state_metrics"]["l1_to_uniform_mean"]
        print(
            json.dumps(
                {
                    "update": int(update),
                    "state_sample_count": int(update_summary["state_sample_count"]),
                    "action_eval_count": int(update_summary["action_eval_count"]),
                    "selected_return_mode": str(args.return_mode),
                    "within_state_l1_corr_selected_return_mean": float(l1["corr_selected_return"]["mean"]),
                    "within_state_l1_delta_selected_return_flatter_minus_sharper_mean": float(
                        l1["delta_selected_return_flatter_minus_sharper"]["mean"]
                    ),
                    "within_state_l1_delta_reward_flatter_minus_sharper_mean": float(
                        l1["delta_immediate_reward_flatter_minus_sharper"]["mean"]
                    ),
                    "state_level_corr_avg_l1_vs_state_value": float(
                        update_summary["state_level"]["corr_avg_l1_to_uniform_vs_state_value"]
                    ),
                },
                ensure_ascii=False,
            )
        )
    out_path = run_dir / str(args.out_name)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
