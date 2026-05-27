from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import (
    _collate_dataclass,
    _heuristic_bw,
    _split_dataclass_by_counts,
    _split_tensor_by_counts,
    _to_device_dataclass,
)
from sagin_marl.rl.structured_parallel_eval import (
    batched_policy_accel_actions,
    batched_policy_bw_outputs,
    batched_policy_sat_pair_indices,
    current_obs_many,
    last_reward_parts_many,
    looks_like_driver_group,
    refresh_stage_obs_cache_many,
    reset_at,
    reset_many,
)
from sagin_marl.rl.structured_stage_builders import build_batched_local_bw_states_from_snapshot
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_env_group
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


def _safe_corr(x: list[float], y: list[float]) -> float | None:
    if len(x) <= 1 or len(y) <= 1 or len(x) != len(y):
        return None
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if float(np.std(xa)) <= 1.0e-12 or float(np.std(ya)) <= 1.0e-12:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def _rankdata_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _safe_spearman_desc(pred: np.ndarray, truth: np.ndarray) -> float:
    if pred.size <= 1 or truth.size <= 1 or pred.size != truth.size:
        return 0.0
    corr = _safe_corr(_rankdata_desc(pred).tolist(), _rankdata_desc(truth).tolist())
    return float(corr or 0.0)


def _pairwise_concordance_desc(pred: np.ndarray, truth: np.ndarray, eps: float = 1.0e-9) -> float:
    if pred.size <= 1 or truth.size <= 1 or pred.size != truth.size:
        return 0.0
    total = 0.0
    hits = 0.0
    for i in range(pred.size):
        for j in range(i + 1, pred.size):
            pred_diff = float(pred[i] - pred[j])
            truth_diff = float(truth[i] - truth[j])
            if abs(pred_diff) <= eps and abs(truth_diff) <= eps:
                hits += 1.0
                total += 1.0
                continue
            if abs(pred_diff) <= eps or abs(truth_diff) <= eps:
                total += 1.0
                hits += 0.5
                continue
            total += 1.0
            if pred_diff * truth_diff > 0.0:
                hits += 1.0
    if total <= 0.0:
        return 0.0
    return float(hits / total)


def _normalize_bw_action(action: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    valid_mask_arr = np.asarray(valid_mask, dtype=bool)
    out = np.zeros(valid_mask_arr.shape, dtype=np.float32)
    action_arr = np.asarray(action, dtype=np.float32)
    for u in range(int(valid_mask_arr.shape[0])):
        valid = valid_mask_arr[u]
        valid_count = int(np.sum(valid))
        if valid_count <= 0:
            continue
        row = np.clip(action_arr[u, valid], 0.0, None)
        row_sum = float(np.sum(row))
        if row_sum <= 1.0e-12:
            out[u, valid] = 1.0 / float(valid_count)
        else:
            out[u, valid] = row / row_sum
    return out


def _uniform_bw_action(valid_mask: np.ndarray) -> np.ndarray:
    valid_mask_arr = np.asarray(valid_mask, dtype=bool)
    out = np.zeros(valid_mask_arr.shape, dtype=np.float32)
    for u in range(int(valid_mask_arr.shape[0])):
        valid = valid_mask_arr[u]
        valid_count = int(np.sum(valid))
        if valid_count > 0:
            out[u, valid] = 1.0 / float(valid_count)
    return out


def _random_bw_action(valid_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    valid_mask_arr = np.asarray(valid_mask, dtype=bool)
    out = np.zeros(valid_mask_arr.shape, dtype=np.float32)
    for u in range(int(valid_mask_arr.shape[0])):
        valid = np.flatnonzero(valid_mask_arr[u])
        if valid.size <= 0:
            continue
        if valid.size == 1:
            out[u, valid[0]] = 1.0
            continue
        weights = rng.dirichlet(np.ones(valid.size, dtype=np.float64)).astype(np.float32)
        out[u, valid] = weights
    return out


def _topk_bw_action_from_scores(scores: np.ndarray, valid_mask: np.ndarray, k: int) -> np.ndarray:
    valid_mask_arr = np.asarray(valid_mask, dtype=bool)
    score_arr = np.asarray(scores, dtype=np.float32)
    out = np.zeros(valid_mask_arr.shape, dtype=np.float32)
    for u in range(int(valid_mask_arr.shape[0])):
        valid_idx = np.flatnonzero(valid_mask_arr[u])
        if valid_idx.size <= 0:
            continue
        row_scores = score_arr[u, valid_idx]
        order = valid_idx[np.argsort(row_scores)[::-1]]
        chosen = order[: max(1, min(int(k), order.size))]
        out[u, chosen] = 1.0 / float(chosen.size)
    return out


def _slot_score_matrix(
    obs_after_accel: list[dict[str, np.ndarray]],
    cfg,
    *,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    num_agents = len(obs_after_accel)
    scores = np.zeros((num_agents, cfg.users_obs_max), dtype=np.float32)
    valid_mask = np.zeros((num_agents, cfg.users_obs_max), dtype=bool)
    for u, obs in enumerate(obs_after_accel):
        users = np.asarray(obs["users"], dtype=np.float32)
        users_mask = np.asarray(obs["users_mask"] > 0.0, dtype=bool)
        bw_valid = np.asarray(obs.get("bw_valid_mask", obs["users_mask"]) > 0.0, dtype=bool)
        valid = users_mask & bw_valid
        valid_mask[u] = valid
        if not np.any(valid):
            continue
        q = np.asarray(users[:, 2], dtype=np.float32)
        eta = np.asarray(users[:, 3], dtype=np.float32)
        prev = np.asarray(users[:, 4], dtype=np.float32)
        if mode == "queue":
            slot_score = np.clip(q, 0.0, None)
        elif mode == "eta":
            slot_score = np.clip(eta, 0.0, None)
        elif mode == "qeta":
            slot_score = np.clip(q, 0.0, None) * (0.5 + np.clip(eta, 0.0, None))
        elif mode == "qeta_prev":
            slot_score = np.clip(q, 0.0, None) * (0.5 + np.clip(eta, 0.0, None)) * (
                1.0 + 0.3 * np.clip(prev, 0.0, None)
            )
        else:
            raise ValueError(f"Unsupported slot score mode: {mode}")
        score_row = np.zeros((cfg.users_obs_max,), dtype=np.float32)
        score_row[valid] = slot_score[valid]
        scores[u] = score_row
    return scores, valid_mask


def _make_policy_panel_actions(
    cfg,
    snapshot,
    obs_after_accel: list[dict[str, np.ndarray]],
    *,
    heuristic_bw_source: str,
    random_count: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    snapshot_valid_mask = np.asarray(snapshot.bw_valid_mask, dtype=bool)
    actions: dict[str, np.ndarray] = {
        "uniform": _uniform_bw_action(snapshot_valid_mask),
        "heuristic": _normalize_bw_action(_heuristic_bw(obs_after_accel, cfg, heuristic_bw_source), snapshot_valid_mask),
    }
    for mode in ("queue", "eta", "qeta", "qeta_prev"):
        scores, valid_mask = _slot_score_matrix(obs_after_accel, cfg, mode=mode)
        valid_mask = valid_mask & snapshot_valid_mask
        for top_k in (1, 2):
            actions[f"{mode}_top{top_k}"] = _topk_bw_action_from_scores(scores, valid_mask, top_k)
    for sample_idx in range(max(int(random_count), 0)):
        actions[f"random_{sample_idx}"] = _random_bw_action(snapshot_valid_mask, rng)
    return actions


def _cpu_dataclass(batch: Any) -> Any:
    if not is_dataclass(batch):
        raise TypeError("_cpu_dataclass expects a dataclass instance")
    kwargs = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if not torch.is_tensor(value):
            raise TypeError(f"Unsupported field type for {field.name}: {type(value)!r}")
        kwargs[field.name] = value.detach().cpu()
    return type(batch)(**kwargs)


def _clone_bw_only_actor(base_actor, device: torch.device):
    actor = copy.deepcopy(base_actor)
    actor.to(device)
    for param in actor.parameters():
        param.requires_grad_(False)
    for param in actor.bw_policy.parameters():
        param.requires_grad_(True)
    actor.train()
    return actor


def _as_driver_list(env_group) -> list[StructuredControlDriver]:
    if isinstance(env_group, list):
        return as_structured_drivers(env_group)
    return [as_structured_driver(env_group)]


def _driver_capacity(drivers) -> int:
    if looks_like_driver_group(drivers):
        return int(len(drivers))
    return int(len(drivers))


def _load_bw_stage_state_many(drivers, snapshots: list[dict[str, Any]], *, indices: list[int] | None = None) -> None:
    if not snapshots:
        return
    if looks_like_driver_group(drivers):
        drivers.load_bw_stage_state_many(snapshots, indices=indices)
        return
    selected = drivers if indices is None else [drivers[int(index)] for index in indices]
    for driver, snapshot in zip(selected, snapshots):
        driver.load_bw_stage_state(snapshot)


def _execute_stage_bw_and_step_many(drivers, bw_actions: list[np.ndarray], *, indices: list[int] | None = None):
    if not bw_actions:
        return []
    if looks_like_driver_group(drivers):
        return drivers.execute_stage_bw_and_step_many(bw_actions, indices=indices)
    selected = drivers if indices is None else [drivers[int(index)] for index in indices]
    return [driver.execute_stage_bw_and_step(action) for driver, action in zip(selected, bw_actions)]


def _execute_bw_and_prepare_next_accel_many(drivers, bw_actions: list[np.ndarray], *, indices: list[int] | None = None):
    if not bw_actions:
        return [], []
    if looks_like_driver_group(drivers):
        return drivers.execute_bw_and_prepare_next_accel_many(bw_actions, indices=indices)
    selected = drivers if indices is None else [drivers[int(index)] for index in indices]
    step_results = []
    next_world_states = []
    for driver, action in zip(selected, bw_actions):
        step_result, next_world = driver.execute_stage_bw_and_prepare_next_accel(action)
        step_results.append(step_result)
        next_world_states.append(next_world)
    return step_results, next_world_states


def _policy_followup_bw_actions_many(
    drivers,
    *,
    indices: list[int],
    next_world_states: list[Any],
    follow_actor,
    follow_device: torch.device,
    deterministic: bool,
    bw_deterministic_readout: str,
    bw_deterministic_opt_steps: int,
    bw_deterministic_step_size: float,
) -> list[np.ndarray]:
    if not next_world_states:
        return []
    accel_actions = batched_policy_accel_actions(follow_actor, next_world_states, follow_device, deterministic)
    if looks_like_driver_group(drivers):
        sat_snapshots = drivers.run_accel_and_prepare_sat_many(accel_actions, indices=indices)
    else:
        selected = [drivers[int(index)] for index in indices]
        sat_world_states = [driver.run_accel_stage(action) for driver, action in zip(selected, accel_actions)]
        sat_snapshots = [
            driver.build_sat_stage_snapshot(world_state)
            for driver, world_state in zip(selected, sat_world_states)
        ]
    sat_pair_indices = batched_policy_sat_pair_indices(follow_actor, sat_snapshots, follow_device, deterministic)
    if looks_like_driver_group(drivers):
        bw_snapshots = drivers.run_sat_and_prepare_bw_many(sat_pair_indices, indices=indices)
    else:
        selected = [drivers[int(index)] for index in indices]
        sat_actions = [
            driver.decode_sat_pair_actions([], pair_indices)
            for driver, pair_indices in zip(selected, sat_pair_indices)
        ]
        bw_world_states = [driver.run_sat_stage(action) for driver, action in zip(selected, sat_actions)]
        bw_snapshots = [
            driver.build_bw_stage_snapshot(world_state)
            for driver, world_state in zip(selected, bw_world_states)
        ]
    bw_eval = _batched_bw_actions_from_snapshots(
        follow_actor,
        bw_snapshots,
        device=follow_device,
        deterministic=deterministic,
        bw_deterministic_readout=bw_deterministic_readout,
        bw_deterministic_opt_steps=int(bw_deterministic_opt_steps),
        bw_deterministic_step_size=float(bw_deterministic_step_size),
    )
    return [np.asarray(action, dtype=np.float32) for action in bw_eval["actions"]]


def _rollout_k_many_with_fixed_follow(
    drivers,
    snapshot_states: list[dict[str, Any]],
    *,
    first_bw_actions: list[np.ndarray],
    follow_actor,
    follow_device: torch.device,
    follow_deterministic: bool,
    bw_deterministic_readout: str,
    bw_deterministic_opt_steps: int,
    bw_deterministic_step_size: float,
    gamma: float,
    k_steps: int,
) -> list[dict[str, float]]:
    if not snapshot_states:
        return []
    if len(snapshot_states) != len(first_bw_actions):
        raise ValueError("snapshot_states and first_bw_actions must have the same length")
    if len(snapshot_states) > _driver_capacity(drivers):
        raise ValueError("Requested rollout batch exceeds available driver capacity")
    totals = [
        {
            "reward": 0.0,
            "x_acc": 0.0,
            "x_rel": 0.0,
            "d_pre": 0.0,
            "pre_backlog_steps_eval": 0.0,
            "processed_ratio_eval": 0.0,
            "drop_ratio_eval": 0.0,
            "steps_executed": 0.0,
        }
        for _ in snapshot_states
    ]
    selected_indices = list(range(len(snapshot_states)))
    _load_bw_stage_state_many(drivers, snapshot_states, indices=selected_indices)
    current_actions = [np.asarray(action, dtype=np.float32) for action in first_bw_actions]
    active_local_indices = list(range(len(snapshot_states)))
    discount = 1.0
    for step_idx in range(max(int(k_steps), 0)):
        if not active_local_indices:
            break
        active_driver_indices = [selected_indices[int(local_idx)] for local_idx in active_local_indices]
        active_actions = [current_actions[int(local_idx)] for local_idx in active_local_indices]
        step_results, next_world_states = _execute_bw_and_prepare_next_accel_many(
            drivers,
            active_actions,
            indices=active_driver_indices,
        )
        reward_parts = last_reward_parts_many(drivers, indices=active_driver_indices)
        next_active_local_indices: list[int] = []
        next_active_world_states: list[Any] = []
        for offset, local_idx in enumerate(active_local_indices):
            parts = dict(reward_parts[offset] or {})
            step_result = step_results[offset]
            totals[int(local_idx)]["reward"] += discount * float(next(iter(step_result.rewards.values())))
            for key in (
                "x_acc",
                "x_rel",
                "d_pre",
                "pre_backlog_steps_eval",
                "processed_ratio_eval",
                "drop_ratio_eval",
            ):
                totals[int(local_idx)][key] += discount * float(parts.get(key, 0.0))
            totals[int(local_idx)]["steps_executed"] += 1.0
            done = bool(next(iter(step_result.terminations.values())) or next(iter(step_result.truncations.values())))
            if done or step_idx + 1 >= int(k_steps):
                continue
            next_active_local_indices.append(int(local_idx))
            next_active_world_states.append(next_world_states[offset])
        if not next_active_local_indices or step_idx + 1 >= int(k_steps):
            break
        discount *= float(gamma)
        next_driver_indices = [selected_indices[int(local_idx)] for local_idx in next_active_local_indices]
        next_actions = _policy_followup_bw_actions_many(
            drivers,
            indices=next_driver_indices,
            next_world_states=next_active_world_states,
            follow_actor=follow_actor,
            follow_device=follow_device,
            deterministic=follow_deterministic,
            bw_deterministic_readout=bw_deterministic_readout,
            bw_deterministic_opt_steps=int(bw_deterministic_opt_steps),
            bw_deterministic_step_size=float(bw_deterministic_step_size),
        )
        for local_idx, action in zip(next_active_local_indices, next_actions):
            current_actions[int(local_idx)] = np.asarray(action, dtype=np.float32)
        active_local_indices = next_active_local_indices
    return totals


def _rollout_panel_action_scores(
    eval_drivers,
    snapshot_state: dict[str, Any],
    *,
    panel_actions: list[np.ndarray],
    follow_actor,
    follow_device: torch.device,
    follow_deterministic: bool,
    bw_deterministic_readout: str,
    bw_deterministic_opt_steps: int,
    bw_deterministic_step_size: float,
    gamma: float,
    k_steps: int,
) -> list[float]:
    if not panel_actions:
        return []
    chunk_size = max(1, _driver_capacity(eval_drivers))
    panel_scores: list[float] = []
    for start in range(0, len(panel_actions), chunk_size):
        chunk_actions = panel_actions[start : start + chunk_size]
        chunk_totals = _rollout_k_many_with_fixed_follow(
            eval_drivers,
            [snapshot_state for _ in chunk_actions],
            first_bw_actions=chunk_actions,
            follow_actor=follow_actor,
            follow_device=follow_device,
            follow_deterministic=follow_deterministic,
            bw_deterministic_readout=bw_deterministic_readout,
            bw_deterministic_opt_steps=int(bw_deterministic_opt_steps),
            bw_deterministic_step_size=float(bw_deterministic_step_size),
            gamma=float(gamma),
            k_steps=int(k_steps),
        )
        panel_scores.extend([float(totals["reward"]) for totals in chunk_totals])
    return panel_scores


def _step_reward_parts(step_result, env) -> dict[str, float]:
    reward = float(next(iter(step_result.rewards.values())))
    parts = dict(getattr(env, "last_reward_parts", {}) or {})
    return {
        "reward": reward,
        "x_acc": float(parts.get("x_acc", 0.0) or 0.0),
        "x_rel": float(parts.get("x_rel", 0.0) or 0.0),
        "d_pre": float(parts.get("d_pre", 0.0) or 0.0),
        "pre_backlog_steps_eval": float(parts.get("pre_backlog_steps_eval", 0.0) or 0.0),
        "processed_ratio_eval": float(parts.get("processed_ratio_eval", 0.0) or 0.0),
        "drop_ratio_eval": float(parts.get("drop_ratio_eval", 0.0) or 0.0),
        "reward_raw": float(parts.get("reward_raw", reward) or reward),
    }


def _policy_followup_bw_action(
    driver: StructuredControlDriver,
    actor,
    device: torch.device,
    *,
    deterministic: bool,
    bw_deterministic_readout: str,
    bw_deterministic_opt_steps: int,
    bw_deterministic_step_size: float,
) -> np.ndarray:
    accel_world = driver.begin_step()
    accel_action = batched_policy_accel_actions(actor, [accel_world], device, deterministic)[0]
    sat_world = driver.run_accel_stage(accel_action)
    sat_snapshot = driver.build_sat_stage_snapshot(sat_world)
    sat_pair_idx = batched_policy_sat_pair_indices(actor, [sat_snapshot], device, deterministic)[0]
    sat_action = driver.decode_sat_pair_actions([], sat_pair_idx)
    bw_world = driver.run_sat_stage(sat_action)
    bw_snapshot = driver.build_bw_stage_snapshot(bw_world)
    bw_action = _batched_bw_actions_from_snapshots(
        actor,
        [bw_snapshot],
        device=device,
        deterministic=deterministic,
        bw_deterministic_readout=bw_deterministic_readout,
        bw_deterministic_opt_steps=int(bw_deterministic_opt_steps),
        bw_deterministic_step_size=float(bw_deterministic_step_size),
    )["actions"][0]
    return np.asarray(bw_action, dtype=np.float32)


def _batched_bw_actions_from_snapshots(
    actor,
    bw_snapshots: list[Any],
    *,
    device: torch.device,
    deterministic: bool,
    bw_deterministic_readout: str,
    bw_deterministic_opt_steps: int,
    bw_deterministic_step_size: float,
) -> dict[str, Any]:
    if not bw_snapshots:
        return {"actions": [], "local_state": None, "agent_counts": []}
    readout = str(bw_deterministic_readout).strip().lower()
    use_custom_det = bool(deterministic) and readout not in {"latent_mean_pushforward", "mode", "default"}
    if not use_custom_det:
        out = batched_policy_bw_outputs(actor, bw_snapshots, device, deterministic)
        return {
            "actions": out.actions,
            "local_state": out.local_state,
            "agent_counts": out.agent_counts,
        }
    bw_world_states = [snapshot.world_state for snapshot in bw_snapshots]
    agent_counts = [int(world_state.uav_nodes.shape[1]) for world_state in bw_world_states]
    bw_world_batch_cpu = _collate_dataclass(bw_world_states, torch.device("cpu"))
    bw_candidate_indices_cpu = torch.stack(
        [torch.as_tensor(snapshot.candidate_indices, dtype=torch.long) for snapshot in bw_snapshots],
        dim=0,
    )
    bw_valid_mask_cpu = torch.stack(
        [torch.as_tensor(snapshot.bw_valid_mask, dtype=torch.bool) for snapshot in bw_snapshots],
        dim=0,
    )
    bw_batch_cpu = build_batched_local_bw_states_from_snapshot(
        bw_world_batch_cpu,
        bw_candidate_indices_cpu,
        bw_valid_mask_cpu,
    )
    bw_batch = _to_device_dataclass(bw_batch_cpu, device)
    action = actor.bw_policy.deterministic_action(
        bw_batch,
        readout=readout,
        opt_steps=int(bw_deterministic_opt_steps),
        opt_step_size=float(bw_deterministic_step_size),
    )
    actions = [piece.numpy() for piece in _split_tensor_by_counts(action.detach().cpu(), agent_counts)]
    return {
        "actions": actions,
        "local_state": bw_batch,
        "agent_counts": agent_counts,
    }


def _rollout_k_with_fixed_follow(
    driver: StructuredControlDriver,
    snapshot_state: dict[str, Any],
    *,
    first_bw_action: np.ndarray,
    follow_actor,
    follow_device: torch.device,
    follow_deterministic: bool,
    bw_deterministic_readout: str,
    bw_deterministic_opt_steps: int,
    bw_deterministic_step_size: float,
    gamma: float,
    k_steps: int,
) -> dict[str, float]:
    totals = _rollout_k_many_with_fixed_follow(
        [driver],
        [snapshot_state],
        first_bw_actions=[np.asarray(first_bw_action, dtype=np.float32)],
        follow_actor=follow_actor,
        follow_device=follow_device,
        follow_deterministic=follow_deterministic,
        bw_deterministic_readout=bw_deterministic_readout,
        bw_deterministic_opt_steps=int(bw_deterministic_opt_steps),
        bw_deterministic_step_size=float(bw_deterministic_step_size),
        gamma=float(gamma),
        k_steps=int(k_steps),
    )
    return totals[0]


def _sum_by_counts(value: torch.Tensor, counts: list[int]) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    cursor = 0
    for count in counts:
        pieces.append(value[cursor : cursor + int(count)].sum())
        cursor += int(count)
    return torch.stack(pieces, dim=0) if pieces else value.new_zeros((0,))


def _valid_bw_mask(local_state) -> torch.Tensor:
    return (local_state.user_mask > 0.5) & (local_state.bw_valid_mask > 0.5)


def _masked_simplex_kl(target: torch.Tensor, mode: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    target_valid = torch.where(valid_mask, target.clamp_min(eps), torch.zeros_like(target))
    mode_valid = torch.where(valid_mask, mode.clamp_min(eps), torch.zeros_like(mode))
    target_valid = target_valid / target_valid.sum(dim=-1, keepdim=True).clamp_min(eps)
    mode_valid = mode_valid / mode_valid.sum(dim=-1, keepdim=True).clamp_min(eps)
    kl = target_valid * (torch.log(target_valid.clamp_min(eps)) - torch.log(mode_valid.clamp_min(eps)))
    return (kl * valid_mask.to(dtype=kl.dtype)).sum(dim=-1)


def _select_negative(entry: dict[str, Any], *, prefer_policy_gap_min: float) -> tuple[np.ndarray, str, float]:
    best_score = float(entry["best_score"])
    policy_det_score = float(entry["policy_det_score"])
    if (best_score - policy_det_score) >= float(prefer_policy_gap_min):
        return np.asarray(entry["policy_det_action"], dtype=np.float32), "policy_det", float(policy_det_score)
    return np.asarray(entry["worst_action"], dtype=np.float32), "worst_panel", float(entry["worst_score"])


def _load_actor(
    run_dir: Path,
    update: int,
    device: torch.device,
    *,
    actor_checkpoint: str | None = None,
    bw_parameterization: str | None = None,
    bw_alpha_init_bias: float | None = None,
    bw_alpha_max: float | None = None,
    bw_tau_enabled: bool | None = None,
    bw_tau_init_bias: float | None = None,
    bw_tau_min: float | None = None,
    bw_tau_max: float | None = None,
):
    cfg = load_config(str(run_dir / "config_source.yaml"))
    original_parameterization = str(getattr(cfg, "structured_bw_parameterization", "legacy") or "legacy")
    original_tau_enabled = bool(getattr(cfg, "structured_bw_tau_enabled", False))
    if bw_parameterization is not None:
        setattr(cfg, "structured_bw_parameterization", str(bw_parameterization))
    if bw_alpha_init_bias is not None:
        setattr(cfg, "structured_bw_alpha_init_bias", float(bw_alpha_init_bias))
    if bw_alpha_max is not None:
        setattr(cfg, "structured_bw_alpha_max", float(bw_alpha_max))
    if bw_tau_enabled is not None:
        setattr(cfg, "structured_bw_tau_enabled", bool(bw_tau_enabled))
    if bw_tau_init_bias is not None:
        setattr(cfg, "structured_bw_tau_init_bias", float(bw_tau_init_bias))
    if bw_tau_min is not None:
        setattr(cfg, "structured_bw_tau_min", float(bw_tau_min))
    if bw_tau_max is not None:
        setattr(cfg, "structured_bw_tau_max", float(bw_tau_max))
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    actor_ckpt = Path(actor_checkpoint) if actor_checkpoint is not None else run_dir / f"actor_u{int(update):04d}.pt"
    same_parameterization = bw_parameterization is None or str(bw_parameterization).strip().lower() == original_parameterization.strip().lower()
    same_tau_enabled = bw_tau_enabled is None or bool(bw_tau_enabled) == original_tau_enabled
    strict = same_parameterization and same_tau_enabled
    load_checkpoint_forgiving(bundle.actor, str(actor_ckpt), map_location=device, strict=strict)
    bundle.actor.to(device).eval()
    return cfg, bundle.actor


def _collect_panel_bank(
    *,
    run_dir: Path,
    update: int,
    device: torch.device,
    actor_checkpoint: str | None,
    output_dir: Path,
    episodes: int,
    panel_states: int,
    policy_mode: str,
    panel_random_count: int,
    heuristic_bw_source: str,
    k_steps: int,
    seed: int,
    prefer_policy_gap_min: float,
    min_best_gap: float,
    num_envs: int,
    bw_deterministic_readout: str,
    bw_deterministic_opt_steps: int,
    bw_deterministic_step_size: float,
    bw_parameterization: str | None,
    bw_alpha_init_bias: float | None,
    bw_alpha_max: float | None,
    bw_tau_enabled: bool | None,
    bw_tau_init_bias: float | None,
    bw_tau_min: float | None,
    bw_tau_max: float | None,
    vec_backend: str,
) -> dict[str, Any]:
    cfg, base_actor = _load_actor(
        run_dir,
        int(update),
        device,
        actor_checkpoint=actor_checkpoint,
        bw_parameterization=bw_parameterization,
        bw_alpha_init_bias=bw_alpha_init_bias,
        bw_alpha_max=bw_alpha_max,
        bw_tau_enabled=bw_tau_enabled,
        bw_tau_init_bias=bw_tau_init_bias,
        bw_tau_min=bw_tau_min,
        bw_tau_max=bw_tau_max,
    )
    deterministic = str(policy_mode) == "deterministic"
    active_slots = max(1, min(int(num_envs), int(episodes)))
    env_group = make_structured_env_group(cfg, num_envs=active_slots, backend=vec_backend)
    drivers = env_group if looks_like_driver_group(env_group) else _as_driver_list(env_group)
    panel_eval_slots = max(1, min(max(int(panel_random_count) + 10, 1), max(int(num_envs), 1)))
    eval_group = make_structured_env_group(cfg, num_envs=panel_eval_slots, backend=vec_backend)
    eval_drivers = eval_group if looks_like_driver_group(eval_group) else _as_driver_list(eval_group)
    slot_active = [True for _ in range(active_slots)]
    next_episode = active_slots
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))
    gamma = float(cfg.gamma)
    initial_seeds = [int(seed) + 1000 + slot for slot in range(active_slots)]
    reset_many(drivers, initial_seeds)
    try:
        while any(slot_active) and len(rows) < int(panel_states):
            active_indices = [slot for slot, is_active in enumerate(slot_active) if is_active]
            if not active_indices:
                break
            if looks_like_driver_group(drivers):
                accel_world_states = drivers.prepare_accel_stage_many(indices=active_indices)
            else:
                accel_world_states = [drivers[slot].begin_step() for slot in active_indices]
            accel_actions = batched_policy_accel_actions(base_actor, accel_world_states, device, deterministic)
            if looks_like_driver_group(drivers):
                sat_snapshots = drivers.run_accel_and_prepare_sat_many(accel_actions, indices=active_indices)
            else:
                sat_world_states = [
                    drivers[slot].run_accel_stage(action)
                    for slot, action in zip(active_indices, accel_actions)
                ]
                sat_snapshots = [
                    drivers[slot].build_sat_stage_snapshot(world_state)
                    for slot, world_state in zip(active_indices, sat_world_states)
                ]
            refresh_stage_obs_cache_many(drivers, indices=active_indices)
            obs_after_accel_groups = current_obs_many(drivers, indices=active_indices)
            sat_pair_indices = batched_policy_sat_pair_indices(base_actor, sat_snapshots, device, deterministic)
            if looks_like_driver_group(drivers):
                bw_snapshots = drivers.run_sat_and_prepare_bw_many(sat_pair_indices, indices=active_indices)
                snapshot_states = drivers.export_bw_stage_state_many(indices=active_indices)
            else:
                sat_actions = [
                    drivers[slot].decode_sat_pair_actions([], sat_pair_idx)
                    for slot, sat_pair_idx in zip(active_indices, sat_pair_indices)
                ]
                bw_world_states = [
                    drivers[slot].run_sat_stage(action)
                    for slot, action in zip(active_indices, sat_actions)
                ]
                bw_snapshots = [
                    drivers[slot].build_bw_stage_snapshot(world_state)
                    for slot, world_state in zip(active_indices, bw_world_states)
                ]
                snapshot_states = [drivers[slot].export_bw_stage_state() for slot in active_indices]
            bw_eval = _batched_bw_actions_from_snapshots(
                base_actor,
                bw_snapshots,
                device=device,
                deterministic=deterministic,
                bw_deterministic_readout=bw_deterministic_readout,
                bw_deterministic_opt_steps=int(bw_deterministic_opt_steps),
                bw_deterministic_step_size=float(bw_deterministic_step_size),
            )
            local_state_groups = _split_dataclass_by_counts(bw_eval["local_state"], bw_eval["agent_counts"])
            policy_det_actions = [np.asarray(action, dtype=np.float32) for action in bw_eval["actions"]]
            for local_idx, slot in enumerate(active_indices):
                if len(rows) >= int(panel_states):
                    break
                snapshot = bw_snapshots[local_idx]
                valid_mask = np.asarray(snapshot.bw_valid_mask, dtype=bool)
                if np.all(valid_mask.sum(axis=1) <= 1):
                    continue
                snapshot_state = snapshot_states[local_idx]
                local_state_cpu = _cpu_dataclass(local_state_groups[local_idx])
                obs_after_accel = obs_after_accel_groups[local_idx]
                panel_actions = _make_policy_panel_actions(
                    cfg,
                    snapshot,
                    obs_after_accel,
                    heuristic_bw_source=heuristic_bw_source,
                    random_count=int(panel_random_count),
                    rng=rng,
                )
                panel_names = list(panel_actions.keys())
                panel_action_values = [np.asarray(panel_actions[name], dtype=np.float32) for name in panel_names]
                panel_scores = _rollout_panel_action_scores(
                    eval_drivers,
                    snapshot_state,
                    panel_actions=panel_action_values,
                    follow_actor=base_actor,
                    follow_device=device,
                    follow_deterministic=deterministic,
                    bw_deterministic_readout=bw_deterministic_readout,
                    bw_deterministic_opt_steps=int(bw_deterministic_opt_steps),
                    bw_deterministic_step_size=float(bw_deterministic_step_size),
                    gamma=gamma,
                    k_steps=int(k_steps),
                )
                if not panel_scores:
                    continue
                best_idx = int(np.argmax(np.asarray(panel_scores, dtype=np.float64)))
                worst_idx = int(np.argmin(np.asarray(panel_scores, dtype=np.float64)))
                best_score = float(panel_scores[best_idx])
                worst_score = float(panel_scores[worst_idx])
                mean_score = float(np.mean(np.asarray(panel_scores, dtype=np.float64)))
                policy_det_action = policy_det_actions[local_idx]
                policy_det_score = float(
                    _rollout_k_many_with_fixed_follow(
                        eval_drivers,
                        [snapshot_state],
                        first_bw_actions=[policy_det_action],
                        follow_actor=base_actor,
                        follow_device=device,
                        follow_deterministic=deterministic,
                        bw_deterministic_readout=bw_deterministic_readout,
                        bw_deterministic_opt_steps=int(bw_deterministic_opt_steps),
                        bw_deterministic_step_size=float(bw_deterministic_step_size),
                        gamma=gamma,
                        k_steps=int(k_steps),
                    )[0]["reward"]
                )
                if (best_score - mean_score) < float(min_best_gap):
                    continue
                neg_action, neg_kind, neg_score = _select_negative(
                    {
                        "best_score": best_score,
                        "policy_det_score": policy_det_score,
                        "policy_det_action": policy_det_action,
                        "worst_action": np.asarray(panel_actions[panel_names[worst_idx]], dtype=np.float32),
                        "worst_score": worst_score,
                    },
                    prefer_policy_gap_min=float(prefer_policy_gap_min),
                )
                rows.append(
                    {
                        "episode_slot": int(slot),
                        "t": int(snapshot_state.get("env_state", {}).get("t", 0)),
                        "snapshot_state": snapshot_state,
                        "snapshot": snapshot,
                        "local_state": local_state_cpu,
                        "panel_names": panel_names,
                        "panel_actions": panel_action_values,
                        "panel_scores": [float(score) for score in panel_scores],
                        "best_idx": int(best_idx),
                        "worst_idx": int(worst_idx),
                        "best_action": np.asarray(panel_actions[panel_names[best_idx]], dtype=np.float32),
                        "best_score": best_score,
                        "worst_action": np.asarray(panel_actions[panel_names[worst_idx]], dtype=np.float32),
                        "worst_score": worst_score,
                        "mean_score": mean_score,
                        "gap_best_mean": float(best_score - mean_score),
                        "gap_best_worst": float(best_score - worst_score),
                        "policy_det_action": policy_det_action,
                        "policy_det_score": policy_det_score,
                        "gap_best_policy": float(best_score - policy_det_score),
                        "negative_action": np.asarray(neg_action, dtype=np.float32),
                        "negative_kind": str(neg_kind),
                        "negative_score": float(neg_score),
                    }
                )
            step_results = _execute_stage_bw_and_step_many(drivers, policy_det_actions, indices=active_indices)
            for local_idx, slot in enumerate(active_indices):
                step_result = step_results[local_idx]
                done = bool(next(iter(step_result.terminations.values())) or next(iter(step_result.truncations.values())))
                if done:
                    if next_episode < int(episodes):
                        reset_at(drivers, slot, int(seed) + 1000 + int(next_episode))
                        next_episode += 1
                    else:
                        slot_active[slot] = False
            if len(rows) >= int(panel_states):
                break
    finally:
        close_structured_env_group(env_group)
        close_structured_env_group(eval_group)
    score_rows = [
        {
            "t": int(entry["t"]),
            "best_score": float(entry["best_score"]),
            "mean_score": float(entry["mean_score"]),
            "policy_det_score": float(entry["policy_det_score"]),
            "gap_best_mean": float(entry["gap_best_mean"]),
            "gap_best_policy": float(entry["gap_best_policy"]),
            "negative_kind": str(entry["negative_kind"]),
        }
        for entry in rows
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "panel_bank_rows.csv").open("w", newline="", encoding="utf-8") as f:
        if score_rows:
            writer = csv.DictWriter(f, fieldnames=list(score_rows[0].keys()))
            writer.writeheader()
            writer.writerows(score_rows)
    return {
        "cfg": cfg,
        "base_actor": base_actor,
        "entries": rows,
        "collection_summary": {
            "episodes": int(episodes),
            "panel_states": int(len(rows)),
            "policy_mode": str(policy_mode),
            "panel_random_count": int(panel_random_count),
            "heuristic_bw_source": str(heuristic_bw_source),
            "k_steps": int(k_steps),
            "vec_backend": str(vec_backend),
            "bw_deterministic_readout": str(bw_deterministic_readout),
            "bw_deterministic_opt_steps": int(bw_deterministic_opt_steps),
            "bw_deterministic_step_size": float(bw_deterministic_step_size),
            "gap_best_mean": _summarize([float(entry["gap_best_mean"]) for entry in rows]),
            "gap_best_policy": _summarize([float(entry["gap_best_policy"]) for entry in rows]),
            "policy_det_score": _summarize([float(entry["policy_det_score"]) for entry in rows]),
        },
    }


def _split_entries(entries: list[dict[str, Any]], holdout_size: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not entries:
        return [], []
    informative = sorted(entries, key=lambda row: float(row["gap_best_policy"]), reverse=True)
    rng = np.random.default_rng(int(seed))
    candidate_count = max(int(holdout_size) * 3, int(holdout_size))
    candidate_count = min(candidate_count, len(informative))
    candidate_pool = informative[:candidate_count]
    holdout_idx = set(rng.choice(len(candidate_pool), size=min(int(holdout_size), len(candidate_pool)), replace=False).tolist())
    holdout: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    for idx, entry in enumerate(candidate_pool):
        (holdout if idx in holdout_idx else train).append(entry)
    train.extend(informative[candidate_count:])
    return train, holdout


def _make_training_batch(entries: list[dict[str, Any]], indices: list[int], device: torch.device) -> dict[str, Any]:
    batch_entries = [entries[int(idx)] for idx in indices]
    local_states_cpu = [entry["local_state"] for entry in batch_entries]
    local_state = _to_device_dataclass(_collate_dataclass(local_states_cpu, torch.device("cpu")), device)
    counts = [int(entry["local_state"].user_mask.shape[0]) for entry in batch_entries]
    best_action = torch.cat(
        [torch.as_tensor(entry["best_action"], dtype=torch.float32) for entry in batch_entries],
        dim=0,
    ).to(device)
    negative_action = torch.cat(
        [torch.as_tensor(entry["negative_action"], dtype=torch.float32) for entry in batch_entries],
        dim=0,
    ).to(device)
    best_score = torch.as_tensor([float(entry["best_score"]) for entry in batch_entries], dtype=torch.float32, device=device)
    negative_score = torch.as_tensor([float(entry["negative_score"]) for entry in batch_entries], dtype=torch.float32, device=device)
    return {
        "entries": batch_entries,
        "local_state": local_state,
        "counts": counts,
        "best_action": best_action,
        "negative_action": negative_action,
        "best_score": best_score,
        "negative_score": negative_score,
    }


def _build_soft_target_action(
    entries: list[dict[str, Any]],
    *,
    top_k: int,
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    target_actions: list[torch.Tensor] = []
    temp = max(float(temperature), 1.0e-6)
    use_top_k = max(int(top_k), 1)
    for entry in entries:
        scores = np.asarray(entry["panel_scores"], dtype=np.float64)
        actions = [np.asarray(action, dtype=np.float32) for action in entry["panel_actions"]]
        order = np.argsort(scores)[::-1][: min(use_top_k, scores.size)]
        chosen_scores = scores[order]
        chosen_actions = np.stack([actions[int(idx)] for idx in order], axis=0).astype(np.float32, copy=False)
        logits = (chosen_scores - float(np.max(chosen_scores))) / temp
        weights = np.exp(logits)
        weights = weights / np.sum(weights)
        target = np.tensordot(weights.astype(np.float32, copy=False), chosen_actions, axes=(0, 0))
        target_actions.append(torch.as_tensor(target, dtype=torch.float32))
    return torch.cat(target_actions, dim=0).to(device)


def _build_pref_pair_batch(
    entries: list[dict[str, Any]],
    *,
    device: torch.device,
    pairing_mode: str,
    pair_top_k: int,
) -> dict[str, Any]:
    pair_local_states_cpu: list[Any] = []
    pair_counts: list[int] = []
    pair_best_actions: list[torch.Tensor] = []
    pair_negative_actions: list[torch.Tensor] = []
    pair_gaps: list[float] = []
    top_k = max(int(pair_top_k), 2)
    for entry in entries:
        local_state_cpu = entry["local_state"]
        count = int(local_state_cpu.user_mask.shape[0])
        pair_specs: list[tuple[np.ndarray, np.ndarray, float]] = []
        if pairing_mode == "single":
            pair_specs.append(
                (
                    np.asarray(entry["best_action"], dtype=np.float32),
                    np.asarray(entry["negative_action"], dtype=np.float32),
                    float(entry["best_score"]) - float(entry["negative_score"]),
                )
            )
        elif pairing_mode == "topk_all_pairs":
            scores = np.asarray(entry["panel_scores"], dtype=np.float64)
            actions = [np.asarray(action, dtype=np.float32) for action in entry["panel_actions"]]
            order = np.argsort(scores)[::-1][: min(top_k, scores.size)]
            for left in range(len(order)):
                hi = int(order[left])
                for right in range(left + 1, len(order)):
                    lo = int(order[right])
                    gap = float(scores[hi] - scores[lo])
                    if gap <= 1.0e-8:
                        continue
                    pair_specs.append((actions[hi], actions[lo], gap))
            if not pair_specs:
                pair_specs.append(
                    (
                        np.asarray(entry["best_action"], dtype=np.float32),
                        np.asarray(entry["negative_action"], dtype=np.float32),
                        float(entry["best_score"]) - float(entry["negative_score"]),
                    )
                )
        else:
            raise ValueError(f"Unsupported pairing_mode: {pairing_mode}")
        for best_action_np, negative_action_np, gap in pair_specs:
            pair_local_states_cpu.append(local_state_cpu)
            pair_counts.append(count)
            pair_best_actions.append(torch.as_tensor(best_action_np, dtype=torch.float32))
            pair_negative_actions.append(torch.as_tensor(negative_action_np, dtype=torch.float32))
            pair_gaps.append(float(gap))
    if not pair_local_states_cpu:
        raise RuntimeError("No preference pairs were built for the current batch.")
    pair_local_state = _to_device_dataclass(_collate_dataclass(pair_local_states_cpu, torch.device("cpu")), device)
    pair_best_action = torch.cat(pair_best_actions, dim=0).to(device)
    pair_negative_action = torch.cat(pair_negative_actions, dim=0).to(device)
    pair_gap = torch.as_tensor(pair_gaps, dtype=torch.float32, device=device)
    return {
        "local_state": pair_local_state,
        "counts": pair_counts,
        "best_action": pair_best_action,
        "negative_action": pair_negative_action,
        "gap": pair_gap,
        "pair_count": int(len(pair_counts)),
    }


def _offline_pref_update(
    actor,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    *,
    pairing_mode: str,
    pair_top_k: int,
    pref_clip_scale: float,
    pref_weight_max: float,
    pull_mode: str,
    pull_coef: float,
    bw_deterministic_readout: str,
    bw_deterministic_opt_steps: int,
    bw_deterministic_step_size: float,
    soft_target_top_k: int,
    soft_target_temp: float,
    grad_clip: float,
) -> dict[str, float]:
    pair_batch = _build_pref_pair_batch(
        batch["entries"],
        device=batch["best_action"].device,
        pairing_mode=str(pairing_mode),
        pair_top_k=int(pair_top_k),
    )
    pair_local_state = pair_batch["local_state"]
    pair_counts = pair_batch["counts"]
    pair_best_action = pair_batch["best_action"]
    pair_negative_action = pair_batch["negative_action"]
    gap = pair_batch["gap"].clamp_min(0.0)
    weight = torch.clamp(gap / max(float(pref_clip_scale), 1.0e-8), 0.0, float(pref_weight_max))
    optimizer.zero_grad(set_to_none=True)
    best_out = actor.evaluate_bw(pair_local_state, pair_best_action)
    neg_out = actor.evaluate_bw(pair_local_state, pair_negative_action)
    best_logprob = _sum_by_counts(best_out.logprob, pair_counts)
    neg_logprob = _sum_by_counts(neg_out.logprob, pair_counts)
    pref_margin = best_logprob - neg_logprob
    pref_loss = -(weight * F.logsigmoid(pref_margin)).mean()
    total_loss = pref_loss
    pull_loss = torch.zeros((), dtype=torch.float32, device=batch["best_action"].device)
    if float(pull_coef) > 0.0 and str(pull_mode) != "none":
        local_state = batch["local_state"]
        counts = batch["counts"]
        mode_action = actor.bw_policy.deterministic_action(
            local_state,
            readout=bw_deterministic_readout,
            opt_steps=int(bw_deterministic_opt_steps),
            opt_step_size=float(bw_deterministic_step_size),
        )
        valid_mask = _valid_bw_mask(local_state)
        if str(pull_mode) == "hard":
            target_action = batch["best_action"]
        elif str(pull_mode) == "soft":
            target_action = _build_soft_target_action(
                batch["entries"],
                top_k=int(soft_target_top_k),
                temperature=float(soft_target_temp),
                device=batch["best_action"].device,
            )
        else:
            raise ValueError(f"Unsupported pull_mode: {pull_mode}")
        kl_per_agent = _masked_simplex_kl(target_action, mode_action, valid_mask)
        kl_per_state = _sum_by_counts(kl_per_agent, counts)
        pull_loss = kl_per_state.mean()
        total_loss = total_loss + float(pull_coef) * pull_loss
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.bw_policy.parameters(), max_norm=float(grad_clip))
    optimizer.step()
    with torch.no_grad():
        return {
            "loss_total": float(total_loss.detach().cpu().item()),
            "loss_pref": float(pref_loss.detach().cpu().item()),
            "loss_pull": float(pull_loss.detach().cpu().item()),
            "margin_mean": float(pref_margin.detach().mean().cpu().item()),
            "gap_mean": float(gap.detach().mean().cpu().item()),
            "weight_mean": float(weight.detach().mean().cpu().item()),
            "pair_count_mean": float(pair_batch["pair_count"]),
        }


def _evaluate_ranking_metrics(actor, entries: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    if not entries:
        return {
            "state_count": 0,
            "top1_hit": 0.0,
            "pairwise_acc": 0.0,
            "spearman": 0.0,
            "pref_margin_best_negative": 0.0,
            "best_logprob_minus_policydet": 0.0,
        }
    top1_hits: list[float] = []
    pairwise: list[float] = []
    spearmans: list[float] = []
    pref_margins: list[float] = []
    best_minus_policydet: list[float] = []
    policydet_rows: list[float] = []
    for entry in entries:
        local_state = _to_device_dataclass(_collate_dataclass([entry["local_state"]], torch.device("cpu")), device)
        counts = [int(entry["local_state"].user_mask.shape[0])]
        pred_scores: list[float] = []
        for action in entry["panel_actions"]:
            action_tensor = torch.as_tensor(action, dtype=torch.float32, device=device)
            out = actor.evaluate_bw(local_state, action_tensor)
            pred_scores.append(float(_sum_by_counts(out.logprob, counts)[0].detach().cpu().item()))
        truth = np.asarray(entry["panel_scores"], dtype=np.float64)
        pred = np.asarray(pred_scores, dtype=np.float64)
        best_idx = int(entry["best_idx"])
        top1_hits.append(1.0 if int(np.argmax(pred)) == best_idx else 0.0)
        pairwise.append(_pairwise_concordance_desc(pred, truth))
        spearmans.append(_safe_spearman_desc(pred, truth))
        best_action_tensor = torch.as_tensor(entry["best_action"], dtype=torch.float32, device=device)
        negative_action_tensor = torch.as_tensor(entry["negative_action"], dtype=torch.float32, device=device)
        policy_det_tensor = torch.as_tensor(entry["policy_det_action"], dtype=torch.float32, device=device)
        best_lp = float(_sum_by_counts(actor.evaluate_bw(local_state, best_action_tensor).logprob, counts)[0].detach().cpu().item())
        neg_lp = float(_sum_by_counts(actor.evaluate_bw(local_state, negative_action_tensor).logprob, counts)[0].detach().cpu().item())
        policy_lp = float(_sum_by_counts(actor.evaluate_bw(local_state, policy_det_tensor).logprob, counts)[0].detach().cpu().item())
        pref_margins.append(best_lp - neg_lp)
        best_minus_policydet.append(best_lp - policy_lp)
        policydet_rows.append(float(entry["policy_det_score"]))
    return {
        "state_count": int(len(entries)),
        "top1_hit": float(np.mean(np.asarray(top1_hits, dtype=np.float64))),
        "pairwise_acc": float(np.mean(np.asarray(pairwise, dtype=np.float64))),
        "spearman": float(np.mean(np.asarray(spearmans, dtype=np.float64))),
        "pref_margin_best_negative": float(np.mean(np.asarray(pref_margins, dtype=np.float64))),
        "best_logprob_minus_policydet": float(np.mean(np.asarray(best_minus_policydet, dtype=np.float64))),
        "base_policy_det_score_summary": _summarize(policydet_rows),
    }


def _evaluate_deterministic_quality(
    actor,
    entries: list[dict[str, Any]],
    *,
    cfg,
    follow_actor,
    follow_device: torch.device,
    follow_deterministic: bool,
    bw_deterministic_readout: str,
    bw_deterministic_opt_steps: int,
    bw_deterministic_step_size: float,
    k_steps: int,
    device: torch.device,
    num_envs: int,
    vec_backend: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    gamma = float(cfg.gamma)
    eval_slots = max(1, min(int(num_envs), max(len(entries), 1)))
    eval_group = make_structured_env_group(cfg, num_envs=eval_slots, backend=vec_backend)
    eval_drivers = eval_group if looks_like_driver_group(eval_group) else _as_driver_list(eval_group)
    try:
        chunk_size = max(1, _driver_capacity(eval_drivers))
        for start in range(0, len(entries), chunk_size):
            chunk_entries = entries[start : start + chunk_size]
            local_states_cpu = [entry["local_state"] for entry in chunk_entries]
            local_state = _to_device_dataclass(_collate_dataclass(local_states_cpu, torch.device("cpu")), device)
            counts = [int(entry["local_state"].user_mask.shape[0]) for entry in chunk_entries]
            with torch.inference_mode():
                det_action_batch = actor.bw_policy.deterministic_action(
                    local_state,
                    readout=bw_deterministic_readout,
                    opt_steps=int(bw_deterministic_opt_steps),
                    opt_step_size=float(bw_deterministic_step_size),
                )
            det_actions = [
                piece.numpy()
                for piece in _split_tensor_by_counts(det_action_batch.detach().cpu(), counts)
            ]
            totals_many = _rollout_k_many_with_fixed_follow(
                eval_drivers,
                [entry["snapshot_state"] for entry in chunk_entries],
                first_bw_actions=[action.astype(np.float32, copy=False) for action in det_actions],
                follow_actor=follow_actor,
                follow_device=follow_device,
                follow_deterministic=follow_deterministic,
                bw_deterministic_readout=bw_deterministic_readout,
                bw_deterministic_opt_steps=int(bw_deterministic_opt_steps),
                bw_deterministic_step_size=float(bw_deterministic_step_size),
                gamma=gamma,
                k_steps=int(k_steps),
            )
            for offset, (entry, totals) in enumerate(zip(chunk_entries, totals_many)):
                rows.append(
                    {
                        "index": int(start + offset),
                        "best_score": float(entry["best_score"]),
                        "policy_det_score_base": float(entry["policy_det_score"]),
                        "det_score_current": float(totals["reward"]),
                        "det_gap_to_best": float(entry["best_score"] - totals["reward"]),
                        "det_improve_vs_base": float(totals["reward"] - float(entry["policy_det_score"])),
                        "det_x_acc": float(totals["x_acc"]),
                        "det_x_rel": float(totals["x_rel"]),
                        "det_d_pre": float(totals["d_pre"]),
                        "det_pre_backlog": float(totals["pre_backlog_steps_eval"]),
                    }
                )
    finally:
        close_structured_env_group(eval_group)
    return {
        "det_score_current": _summarize([float(row["det_score_current"]) for row in rows]),
        "det_gap_to_best": _summarize([float(row["det_gap_to_best"]) for row in rows]),
        "det_improve_vs_base": _summarize([float(row["det_improve_vs_base"]) for row in rows]),
        "det_x_acc": _summarize([float(row["det_x_acc"]) for row in rows]),
        "det_x_rel": _summarize([float(row["det_x_rel"]) for row in rows]),
        "det_d_pre": _summarize([float(row["det_d_pre"]) for row in rows]),
        "det_pre_backlog": _summarize([float(row["det_pre_backlog"]) for row in rows]),
    }, rows


def _run_variant(
    *,
    name: str,
    base_actor,
    train_entries: list[dict[str, Any]],
    holdout_entries: list[dict[str, Any]],
    device: torch.device,
    cfg,
    follow_actor,
    follow_deterministic: bool,
    k_steps: int,
    steps: int,
    batch_size: int,
    lr: float,
    pairing_mode: str,
    pair_top_k: int,
    pref_clip_scale: float,
    pref_weight_max: float,
    pull_mode: str,
    pull_coef: float,
    pull_start_step: int,
    bw_deterministic_readout: str,
    bw_deterministic_opt_steps: int,
    bw_deterministic_step_size: float,
    soft_target_top_k: int,
    soft_target_temp: float,
    grad_clip: float,
    seed: int,
    num_envs: int,
    vec_backend: str,
) -> dict[str, Any]:
    if name == "baseline":
        actor = copy.deepcopy(base_actor).to(device).eval()
        history: list[dict[str, float]] = []
    else:
        actor = _clone_bw_only_actor(base_actor, device)
        optimizer = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=float(lr))
        rng = np.random.default_rng(int(seed) + hash(name) % 100000)
        history = []
        if train_entries:
            for step_idx in range(max(int(steps), 0)):
                batch_indices = rng.integers(0, len(train_entries), size=min(int(batch_size), len(train_entries))).tolist()
                batch = _make_training_batch(train_entries, batch_indices, device)
                stats = _offline_pref_update(
                    actor,
                    optimizer,
                    batch,
                    pairing_mode=str(pairing_mode),
                    pair_top_k=int(pair_top_k),
                    pref_clip_scale=float(pref_clip_scale),
                    pref_weight_max=float(pref_weight_max),
                    pull_mode=str(pull_mode),
                    pull_coef=(
                        0.0
                        if int(step_idx) < int(pull_start_step)
                        else float(pull_coef)
                    ),
                    bw_deterministic_readout=str(bw_deterministic_readout),
                    bw_deterministic_opt_steps=int(bw_deterministic_opt_steps),
                    bw_deterministic_step_size=float(bw_deterministic_step_size),
                    soft_target_top_k=int(soft_target_top_k),
                    soft_target_temp=float(soft_target_temp),
                    grad_clip=float(grad_clip),
                )
                stats["step"] = float(step_idx + 1)
                history.append(stats)
        actor.eval()
    ranking = _evaluate_ranking_metrics(actor, holdout_entries, device)
    det_summary, det_rows = _evaluate_deterministic_quality(
        actor,
        holdout_entries,
        cfg=cfg,
        follow_actor=follow_actor,
        follow_device=device,
        follow_deterministic=follow_deterministic,
        bw_deterministic_readout=str(bw_deterministic_readout),
        bw_deterministic_opt_steps=int(bw_deterministic_opt_steps),
        bw_deterministic_step_size=float(bw_deterministic_step_size),
        k_steps=int(k_steps),
        device=device,
        num_envs=int(num_envs),
        vec_backend=str(vec_backend),
    )
    return {
        "name": str(name),
        "pairing_mode": str(pairing_mode),
        "pull_mode": str(pull_mode),
        "pull_coef": float(pull_coef),
        "pull_start_step": int(pull_start_step),
        "actor_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in actor.state_dict().items()
        },
        "ranking": ranking,
        "deterministic": det_summary,
        "history_tail": history[-20:],
        "history_summary": {
            "loss_total": _summarize([float(row["loss_total"]) for row in history]),
            "loss_pref": _summarize([float(row["loss_pref"]) for row in history]),
            "loss_pull": _summarize([float(row["loss_pull"]) for row in history]),
            "margin_mean": _summarize([float(row["margin_mean"]) for row in history]),
            "gap_mean": _summarize([float(row["gap_mean"]) for row in history]),
            "weight_mean": _summarize([float(row["weight_mean"]) for row in history]),
            "pair_count_mean": _summarize([float(row["pair_count_mean"]) for row in history]),
        },
        "heldout_rows": det_rows,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled offline broad-to-local BW audit.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--actor_checkpoint", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--panel_states", type=int, default=32)
    parser.add_argument("--holdout_size", type=int, default=8)
    parser.add_argument("--policy_mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument(
        "--bw_deterministic_readout",
        choices=["latent_mean_pushforward", "simplex_argmax_logprob"],
        default="latent_mean_pushforward",
    )
    parser.add_argument("--bw_deterministic_opt_steps", type=int, default=8)
    parser.add_argument("--bw_deterministic_step_size", type=float, default=0.5)
    parser.add_argument(
        "--bw_parameterization",
        type=str,
        default=None,
        choices=["legacy", "score_alpha_scalar_scale", "score_alpha_kappa_dirichlet"],
    )
    parser.add_argument("--bw_alpha_init_bias", type=float, default=None)
    parser.add_argument("--bw_alpha_max", type=float, default=None)
    parser.add_argument("--bw_tau_enabled", action="store_true")
    parser.add_argument("--bw_tau_init_bias", type=float, default=None)
    parser.add_argument("--bw_tau_min", type=float, default=None)
    parser.add_argument("--bw_tau_max", type=float, default=None)
    parser.add_argument("--panel_random_count", type=int, default=2)
    parser.add_argument("--heuristic_bw_source", type=str, default="queue_aware")
    parser.add_argument("--k_steps", type=int, default=2)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--pair_top_k", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--pref_clip_scale", type=float, default=0.1)
    parser.add_argument("--pref_weight_max", type=float, default=5.0)
    parser.add_argument("--best_action_pull_coef", type=float, default=0.1)
    parser.add_argument("--soft_target_pull_coef", type=float, default=0.1)
    parser.add_argument("--soft_target_pull_coef_low", type=float, default=0.03)
    parser.add_argument("--soft_target_top_k", type=int, default=4)
    parser.add_argument("--soft_target_temp", type=float, default=0.1)
    parser.add_argument("--soft_target_pull_delay_frac", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--prefer_policy_gap_min", type=float, default=1.0e-3)
    parser.add_argument("--min_best_gap", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vec_backend", type=str, default="sync", choices=["sync", "subproc"])
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    _set_all_seeds(int(args.seed))
    run_dir = Path(args.run_dir)
    device = torch.device(args.device)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("runs") / f"bw_broad2local_offline_audit_u{int(args.update):04d}_20260406"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = _collect_panel_bank(
        run_dir=run_dir,
        update=int(args.update),
        device=device,
        actor_checkpoint=None if args.actor_checkpoint is None else str(args.actor_checkpoint),
        output_dir=out_dir,
        episodes=int(args.episodes),
        panel_states=int(args.panel_states),
        policy_mode=str(args.policy_mode),
        panel_random_count=int(args.panel_random_count),
        heuristic_bw_source=str(args.heuristic_bw_source),
        k_steps=int(args.k_steps),
        seed=int(args.seed),
        prefer_policy_gap_min=float(args.prefer_policy_gap_min),
        min_best_gap=float(args.min_best_gap),
        num_envs=int(args.num_envs),
        bw_deterministic_readout=str(args.bw_deterministic_readout),
        bw_deterministic_opt_steps=int(args.bw_deterministic_opt_steps),
        bw_deterministic_step_size=float(args.bw_deterministic_step_size),
        bw_parameterization=None if args.bw_parameterization is None else str(args.bw_parameterization),
        bw_alpha_init_bias=None if args.bw_alpha_init_bias is None else float(args.bw_alpha_init_bias),
        bw_alpha_max=None if args.bw_alpha_max is None else float(args.bw_alpha_max),
        bw_tau_enabled=True if bool(args.bw_tau_enabled) else None,
        bw_tau_init_bias=None if args.bw_tau_init_bias is None else float(args.bw_tau_init_bias),
        bw_tau_min=None if args.bw_tau_min is None else float(args.bw_tau_min),
        bw_tau_max=None if args.bw_tau_max is None else float(args.bw_tau_max),
        vec_backend=str(args.vec_backend),
    )
    cfg = bank["cfg"]
    base_actor = bank["base_actor"]
    entries = bank["entries"]
    train_entries, holdout_entries = _split_entries(entries, int(args.holdout_size), int(args.seed) + 17)

    pull_delay_step = int(round(max(float(args.soft_target_pull_delay_frac), 0.0) * max(int(args.steps), 0)))
    variants = [
        ("baseline", "single", "none", 0.0, 0),
        ("panel_pref_only", "single", "none", 0.0, 0),
        ("panel_pref_plus_pull", "single", "hard", float(args.best_action_pull_coef), 0),
        ("panel_pref_topk_pairs", "topk_all_pairs", "none", 0.0, 0),
        ("panel_pref_topk_pairs_plus_pull", "topk_all_pairs", "hard", float(args.best_action_pull_coef), 0),
        ("panel_pref_topk_pairs_plus_softpull", "topk_all_pairs", "soft", float(args.soft_target_pull_coef), 0),
        (
            "panel_pref_topk_pairs_plus_softpull_low",
            "topk_all_pairs",
            "soft",
            float(args.soft_target_pull_coef_low),
            0,
        ),
        (
            "panel_pref_topk_pairs_plus_softpull_delayed",
            "topk_all_pairs",
            "soft",
            float(args.soft_target_pull_coef),
            pull_delay_step,
        ),
    ]
    variant_results: list[dict[str, Any]] = []
    heldout_rows_all: list[dict[str, Any]] = []
    for variant_name, pairing_mode, pull_mode, pull_coef, pull_start_step in variants:
        result = _run_variant(
            name=variant_name,
            base_actor=base_actor,
            train_entries=train_entries,
            holdout_entries=holdout_entries,
            device=device,
            cfg=cfg,
            follow_actor=base_actor,
            follow_deterministic=(str(args.policy_mode) == "deterministic"),
            k_steps=int(args.k_steps),
            steps=0 if variant_name == "baseline" else int(args.steps),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            pairing_mode=str(pairing_mode),
            pair_top_k=int(args.pair_top_k),
            pref_clip_scale=float(args.pref_clip_scale),
            pref_weight_max=float(args.pref_weight_max),
            pull_mode=str(pull_mode),
            pull_coef=float(pull_coef),
            pull_start_step=int(pull_start_step),
            bw_deterministic_readout=str(args.bw_deterministic_readout),
            bw_deterministic_opt_steps=int(args.bw_deterministic_opt_steps),
            bw_deterministic_step_size=float(args.bw_deterministic_step_size),
            soft_target_top_k=int(args.soft_target_top_k),
            soft_target_temp=float(args.soft_target_temp),
            grad_clip=float(args.grad_clip),
            seed=int(args.seed),
            num_envs=int(args.num_envs),
            vec_backend=str(args.vec_backend),
        )
        actor_path = out_dir / f"variant_actor_{variant_name}.pt"
        torch.save(result["actor_state_dict"], actor_path)
        result["actor_checkpoint"] = str(actor_path.resolve())
        result.pop("actor_state_dict", None)
        variant_results.append(result)
        for row in result["heldout_rows"]:
            row_copy = dict(row)
            row_copy["variant"] = str(variant_name)
            heldout_rows_all.append(row_copy)

    summary = {
        "run_dir": str(run_dir),
        "update": int(args.update),
        "actor_checkpoint": None if args.actor_checkpoint is None else str(Path(args.actor_checkpoint).resolve()),
        "policy_mode": str(args.policy_mode),
        "bw_deterministic_readout": str(args.bw_deterministic_readout),
        "bw_deterministic_opt_steps": int(args.bw_deterministic_opt_steps),
        "bw_deterministic_step_size": float(args.bw_deterministic_step_size),
        "bw_parameterization": None if args.bw_parameterization is None else str(args.bw_parameterization),
        "bw_alpha_init_bias": None if args.bw_alpha_init_bias is None else float(args.bw_alpha_init_bias),
        "bw_alpha_max": None if args.bw_alpha_max is None else float(args.bw_alpha_max),
        "bw_tau_enabled": bool(args.bw_tau_enabled),
        "bw_tau_init_bias": None if args.bw_tau_init_bias is None else float(args.bw_tau_init_bias),
        "bw_tau_min": None if args.bw_tau_min is None else float(args.bw_tau_min),
        "bw_tau_max": None if args.bw_tau_max is None else float(args.bw_tau_max),
        "device": str(device),
        "vec_backend": str(args.vec_backend),
        "collection": bank["collection_summary"],
        "split": {
            "train_count": int(len(train_entries)),
            "holdout_count": int(len(holdout_entries)),
            "train_gap_best_policy": _summarize([float(entry["gap_best_policy"]) for entry in train_entries]),
            "holdout_gap_best_policy": _summarize([float(entry["gap_best_policy"]) for entry in holdout_entries]),
        },
        "variants": [
            {
                "name": result["name"],
                "pairing_mode": result["pairing_mode"],
                "pull_mode": result["pull_mode"],
                "pull_coef": result["pull_coef"],
                "pull_start_step": result["pull_start_step"],
                "actor_checkpoint": result.get("actor_checkpoint"),
                "ranking": result["ranking"],
                "deterministic": result["deterministic"],
                "history_summary": result["history_summary"],
            }
            for result in variant_results
        ],
    }
    _write_json(out_dir / "summary.json", summary)
    _write_csv(out_dir / "heldout_rows.csv", heldout_rows_all)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
