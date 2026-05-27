from __future__ import annotations

import argparse
import csv
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

from scripts.diagnostics.audit.audit_bw_broad2local_offline import (
    _as_driver_list,
    _batched_bw_actions_from_snapshots,
)
from sagin_marl.env import channel
from sagin_marl.env.config import load_config
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.baselines import queue_aware_bw_policy
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_parallel_eval import (
    batched_policy_accel_actions,
    batched_policy_sat_pair_indices,
    looks_like_driver_group,
)
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_env_group
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving


ACCESS_RULES: tuple[dict[str, float], ...] = (
    {
        "name": "overlap_bid_q_eta",
        "eta_bias": 0.5,
        "eta_weight": 1.0,
        "prev_bonus": 0.0,
        "dist_penalty": 0.0,
        "own_load_penalty": 0.0,
    },
    {
        "name": "overlap_bid_q_eta_prev",
        "eta_bias": 0.5,
        "eta_weight": 1.0,
        "prev_bonus": 0.3,
        "dist_penalty": 0.0,
        "own_load_penalty": 0.0,
    },
    {
        "name": "overlap_bid_q_only",
        "eta_bias": 1.0,
        "eta_weight": 0.0,
        "prev_bonus": 0.0,
        "dist_penalty": 0.0,
        "own_load_penalty": 0.0,
    },
    {
        "name": "overlap_bid_q_eta_dist",
        "eta_bias": 0.5,
        "eta_weight": 1.0,
        "prev_bonus": 0.15,
        "dist_penalty": 0.75,
        "own_load_penalty": 0.0,
    },
    {
        "name": "overlap_bid_q_eta_load",
        "eta_bias": 0.5,
        "eta_weight": 1.0,
        "prev_bonus": 0.15,
        "dist_penalty": 0.0,
        "own_load_penalty": 0.75,
    },
)


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_checkpoint(run_dir: Path, checkpoint: str | None, update: int | None) -> Path:
    if checkpoint:
        return Path(checkpoint)
    if update is not None:
        return run_dir / f"actor_u{int(update):04d}.pt"
    final_ckpt = run_dir / "actor_final.pt"
    if final_ckpt.exists():
        return final_ckpt
    return run_dir / "actor.pt"


def _load_actor(cfg, checkpoint: Path, *, hidden_dim: int, embed_dim: int, device: torch.device):
    bundle = build_structured_modules_from_config(cfg, hidden_dim=hidden_dim, embed_dim=embed_dim)
    load_checkpoint_forgiving(bundle.actor, str(checkpoint), map_location=device, strict=True)
    bundle.actor.to(device).eval()
    return bundle.actor


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _new_episode_accumulator() -> dict[str, float]:
    return {
        "reward_sum": 0.0,
        "processed_ratio_sum": 0.0,
        "drop_ratio_sum": 0.0,
        "pre_backlog_sum": 0.0,
        "d_sys_sum": 0.0,
        "x_acc_sum": 0.0,
        "x_rel_sum": 0.0,
        "g_pre_sum": 0.0,
        "d_pre_sum": 0.0,
        "collision_any": 0.0,
        "steps": 0.0,
    }


def _episode_row_from_accumulator(episode_index: int, acc: dict[str, float]) -> dict[str, float]:
    denom = float(max(int(acc["steps"]), 1))
    return {
        "episode": float(episode_index),
        "reward_sum": float(acc["reward_sum"]),
        "processed_ratio_eval": float(acc["processed_ratio_sum"] / denom),
        "drop_ratio_eval": float(acc["drop_ratio_sum"] / denom),
        "pre_backlog_steps_eval": float(acc["pre_backlog_sum"] / denom),
        "D_sys_report": float(acc["d_sys_sum"] / denom),
        "x_acc_mean": float(acc["x_acc_sum"] / denom),
        "x_rel_mean": float(acc["x_rel_sum"] / denom),
        "g_pre_mean": float(acc["g_pre_sum"] / denom),
        "d_pre_mean": float(acc["d_pre_sum"] / denom),
        "collision_episode_fraction": float(acc["collision_any"]),
    }


def _episode_metrics_template() -> dict[str, float]:
    return {
        "reward_sum": 0.0,
        "processed_ratio_eval": 0.0,
        "drop_ratio_eval": 0.0,
        "pre_backlog_steps_eval": 0.0,
        "D_sys_report": 0.0,
        "x_acc_mean": 0.0,
        "x_rel_mean": 0.0,
        "g_pre_mean": 0.0,
        "d_pre_mean": 0.0,
        "collision_episode_fraction": 0.0,
    }


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


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _current_obs_list(driver: StructuredControlDriver) -> list[dict[str, np.ndarray]]:
    env = driver.env
    return [env._get_obs(i) for i in range(len(env.agents))]


def _uniform_bw_from_obs(obs_list: list[dict[str, np.ndarray]], cfg) -> np.ndarray:
    out = np.zeros((cfg.num_uav, cfg.users_obs_max), dtype=np.float32)
    for u, obs in enumerate(obs_list):
        valid = np.asarray(obs.get("bw_valid_mask", obs["users_mask"]) > 0.0, dtype=bool)
        if np.any(valid):
            out[u, valid] = 1.0 / float(np.sum(valid))
    return out


def _refresh_stage_obs_cache(driver: StructuredControlDriver) -> None:
    if driver._stage_assoc is None or driver._stage_candidates is None:
        raise RuntimeError("stage caches are unavailable")
    env = driver.env
    env._cached_assoc = np.asarray(driver._stage_assoc, dtype=np.int32).copy()
    env._cached_candidates = [list(c) for c in driver._stage_candidates]
    if driver._stage_bw_valid_mask is not None:
        env._cached_bw_valid_mask = np.asarray(driver._stage_bw_valid_mask, dtype=np.float32).copy()
    _, env._cached_eta = env._compute_access_rates(
        env._cached_assoc,
        env._cached_candidates,
        env._dummy_actions(),
        record_exec=False,
    )
    if driver._stage_sat_pos is not None and driver._stage_sat_vel is not None and driver._stage_visible is not None:
        env._cache_sat_obs(driver._stage_sat_pos, driver._stage_sat_vel, driver._stage_visible)


def _apply_access_stage_override(
    driver,
    *,
    assoc: np.ndarray,
    candidates: list[list[int]],
) -> None:
    if hasattr(driver, "override_access_stage_state") and not hasattr(driver, "env"):
        driver.override_access_stage_state(
            np.asarray(assoc, dtype=np.int32),
            [list(int(idx) for idx in cand) for cand in candidates],
        )
        return
    env = driver.env
    driver._stage_assoc = np.asarray(assoc, dtype=np.int32).copy()
    driver._stage_candidates = [list(int(idx) for idx in cand[: env.cfg.users_obs_max]) for cand in candidates]
    driver._stage_bw_valid_mask = env._build_bw_valid_mask(driver._stage_assoc, driver._stage_candidates)
    driver._refresh_world_build_cache()
    _refresh_stage_obs_cache(driver)


def _assoc_candidate_users(
    env,
    assoc: np.ndarray,
    *,
    max_keep: int,
) -> list[list[int]]:
    candidates: list[list[int]] = [[] for _ in range(int(env.cfg.num_uav))]
    for gu_idx, u in enumerate(np.asarray(assoc, dtype=np.int32).tolist()):
        if u >= 0:
            candidates[int(u)].append(int(gu_idx))
    for u in range(int(env.cfg.num_uav)):
        if len(candidates[u]) > int(max_keep):
            candidates[u].sort(key=lambda idx: float(env.gu_queue[int(idx)]), reverse=True)
            candidates[u] = candidates[u][: int(max_keep)]
    return candidates


def _feasible_overlap_candidate_lists(
    env,
    *,
    per_uav_keep: int,
    per_user_max_uav: int,
) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    cfg = env.cfg
    num_gu = int(cfg.num_gu)
    num_uav = int(cfg.num_uav)
    if num_gu <= 0 or num_uav <= 0:
        return [[] for _ in range(num_uav)], np.zeros((num_gu, num_uav), dtype=np.float32), np.zeros((num_gu, num_uav), dtype=np.float32)
    diff = env.gu_pos[:, None, :] - env.uav_pos[None, :, :]
    d2d = np.linalg.norm(diff, axis=2)
    d3d = np.sqrt(d2d * d2d + float(cfg.uav_height) ** 2)
    phi = np.arcsin(float(cfg.uav_height) / (d3d + 1.0e-9))
    pathloss = channel.pathloss_db(d3d, phi, cfg).astype(np.float32, copy=False)
    feasible = pathloss <= float(cfg.pl_threshold_db)
    candidates: list[list[int]] = [[] for _ in range(num_uav)]
    for gu_idx in range(num_gu):
        feasible_u = np.flatnonzero(feasible[gu_idx])
        if feasible_u.size <= 0:
            continue
        ordered = feasible_u[np.argsort(pathloss[gu_idx, feasible_u], kind="stable")]
        keep_u = ordered[: max(int(per_user_max_uav), 1)]
        for u in keep_u.tolist():
            candidates[int(u)].append(int(gu_idx))
    for u in range(num_uav):
        if len(candidates[u]) > int(per_uav_keep):
            candidates[u].sort(key=lambda idx: float(env.gu_queue[int(idx)]), reverse=True)
            candidates[u] = candidates[u][: int(per_uav_keep)]
    return candidates, pathloss.astype(np.float32, copy=False), d2d.astype(np.float32, copy=False)


def _reference_eta_matrix(env) -> np.ndarray:
    cfg = env.cfg
    gain_matrix = env._compute_access_link_gain_matrix().astype(np.float32, copy=False)
    ref_snr = channel.snr_linear(
        float(cfg.gu_tx_power),
        gain_matrix,
        float(cfg.noise_density),
        float(cfg.b_acc),
    )
    return np.asarray(channel.spectral_efficiency(ref_snr), dtype=np.float32)


def _local_bid_score(
    *,
    env,
    u: int,
    gu_idx: int,
    eta_ref: np.ndarray,
    d2d: np.ndarray,
    own_assoc_count: np.ndarray,
    rule: dict[str, float],
) -> float:
    cfg = env.cfg
    q_norm = float(env.gu_queue[int(gu_idx)] / max(float(cfg.queue_max_gu), 1.0e-9))
    eta = float(eta_ref[int(gu_idx), int(u)])
    prev = 1.0 if int(env.prev_association[int(gu_idx)]) == int(u) else 0.0
    dist_norm = float(d2d[int(gu_idx), int(u)] / max(float(cfg.map_size), 1.0e-9))
    own_load = float(own_assoc_count[int(u)] / max(float(cfg.num_gu), 1.0))
    score = q_norm * (float(rule["eta_bias"]) + float(rule["eta_weight"]) * eta)
    score = score * (1.0 + float(rule["prev_bonus"]) * prev)
    score = score / (1.0 + float(rule["dist_penalty"]) * dist_norm)
    score = score / (1.0 + float(rule["own_load_penalty"]) * own_load)
    return float(score)


def _match_assoc_from_local_bids(
    *,
    base_assoc: np.ndarray,
    overlap_candidates: list[list[int]],
    env,
    eta_ref: np.ndarray,
    pathloss: np.ndarray,
    d2d: np.ndarray,
    rule: dict[str, float],
) -> np.ndarray:
    assoc = np.asarray(base_assoc, dtype=np.int32).copy()
    own_assoc_count = np.bincount(
        np.asarray(base_assoc, dtype=np.int32)[np.asarray(base_assoc, dtype=np.int32) >= 0],
        minlength=int(env.cfg.num_uav),
    ).astype(np.float32, copy=False)
    contenders: dict[int, list[tuple[float, float, int]]] = {}
    for u, cand_list in enumerate(overlap_candidates):
        for gu_idx in cand_list:
            score = _local_bid_score(
                env=env,
                u=int(u),
                gu_idx=int(gu_idx),
                eta_ref=eta_ref,
                d2d=d2d,
                own_assoc_count=own_assoc_count,
                rule=rule,
            )
            contenders.setdefault(int(gu_idx), []).append((score, -float(pathloss[int(gu_idx), int(u)]), int(u)))
    for gu_idx, items in contenders.items():
        best_score, _neg_pl, best_u = max(items, key=lambda item: (float(item[0]), float(item[1]), -int(item[2])))
        if best_score > 0.0:
            assoc[int(gu_idx)] = int(best_u)
    return assoc.astype(np.int32, copy=False)


def _build_access_candidate_panel(
    *,
    driver: StructuredControlDriver,
    snapshot_state: dict[str, Any],
    hard_policy_det_action: np.ndarray,
    overlap_per_uav_keep: int,
    overlap_per_user_max_uav: int,
    include_uniform: bool,
) -> list[dict[str, Any]]:
    driver.load_bw_stage_state(snapshot_state)
    env = driver.env
    cfg = env.cfg
    base_assoc = np.asarray(snapshot_state["stage_assoc"], dtype=np.int32).copy()
    base_candidates = [list(int(idx) for idx in cand) for cand in snapshot_state["stage_candidates"]]
    base_obs = _current_obs_list(driver)
    panel: list[dict[str, Any]] = []
    panel.append(
        {
            "name": "hard_assoc_policydet",
            "assoc": base_assoc.copy(),
            "candidates": [list(c) for c in base_candidates],
            "bw_action": np.asarray(hard_policy_det_action, dtype=np.float32).copy(),
            "snapshot_state": snapshot_state,
        }
    )
    panel.append(
        {
            "name": "hard_assoc_heurbw",
            "assoc": base_assoc.copy(),
            "candidates": [list(c) for c in base_candidates],
            "bw_action": np.asarray(queue_aware_bw_policy(base_obs, cfg), dtype=np.float32),
            "snapshot_state": snapshot_state,
        }
    )
    if include_uniform:
        panel.append(
            {
                "name": "hard_assoc_uniform",
                "assoc": base_assoc.copy(),
                "candidates": [list(c) for c in base_candidates],
                "bw_action": _uniform_bw_from_obs(base_obs, cfg),
                "snapshot_state": snapshot_state,
            }
        )

    overlap_candidates, pathloss, d2d = _feasible_overlap_candidate_lists(
        env,
        per_uav_keep=int(overlap_per_uav_keep),
        per_user_max_uav=int(overlap_per_user_max_uav),
    )
    eta_ref = _reference_eta_matrix(env)
    for rule in ACCESS_RULES:
        assoc_new = _match_assoc_from_local_bids(
            base_assoc=base_assoc,
            overlap_candidates=overlap_candidates,
            env=env,
            eta_ref=eta_ref,
            pathloss=pathloss,
            d2d=d2d,
            rule=rule,
        )
        candidates_new = _assoc_candidate_users(
            env,
            assoc_new,
            max_keep=int(cfg.users_obs_max),
        )
        driver.load_bw_stage_state(snapshot_state)
        _apply_access_stage_override(driver, assoc=assoc_new, candidates=candidates_new)
        obs_new = _current_obs_list(driver)
        panel.append(
            {
                "name": f"{rule['name']}_heurbw",
                "assoc": assoc_new.copy(),
                "candidates": [list(c) for c in candidates_new],
                "bw_action": np.asarray(queue_aware_bw_policy(obs_new, cfg), dtype=np.float32),
                "snapshot_state": snapshot_state,
            }
        )
        if include_uniform:
            panel.append(
                {
                    "name": f"{rule['name']}_uniform",
                    "assoc": assoc_new.copy(),
                    "candidates": [list(c) for c in candidates_new],
                    "bw_action": _uniform_bw_from_obs(obs_new, cfg),
                    "snapshot_state": snapshot_state,
                }
            )
    return panel


def _apply_candidate_and_step(
    driver: StructuredControlDriver,
    candidate: dict[str, Any],
) -> Any:
    driver.load_bw_stage_state(candidate["snapshot_state"])
    _apply_access_stage_override(
        driver,
        assoc=np.asarray(candidate["assoc"], dtype=np.int32),
        candidates=[list(c) for c in candidate["candidates"]],
    )
    return driver.execute_stage_bw_and_step(np.asarray(candidate["bw_action"], dtype=np.float32))


def _execute_access_and_prepare_next_accel(
    driver: StructuredControlDriver,
    candidate: dict[str, Any],
) -> tuple[Any, Any]:
    driver.load_bw_stage_state(candidate["snapshot_state"])
    _apply_access_stage_override(
        driver,
        assoc=np.asarray(candidate["assoc"], dtype=np.int32),
        candidates=[list(c) for c in candidate["candidates"]],
    )
    return driver.execute_stage_bw_and_prepare_next_accel(np.asarray(candidate["bw_action"], dtype=np.float32))


def _policy_followup_bw_actions(
    drivers,
    *,
    local_indices: list[int],
    next_world_states: list[Any],
    follow_actor,
    follow_device: torch.device,
    deterministic: bool,
    bw_deterministic_readout: str,
    bw_deterministic_opt_steps: int,
    bw_deterministic_step_size: float,
) -> list[np.ndarray]:
    if not local_indices:
        return []
    accel_actions = batched_policy_accel_actions(follow_actor, next_world_states, follow_device, deterministic)
    if looks_like_driver_group(drivers):
        sat_snapshots = drivers.run_accel_and_prepare_sat_many(accel_actions, indices=local_indices)
    else:
        sat_world_states = [
            drivers[int(local_idx)].run_accel_stage(action)
            for local_idx, action in zip(local_indices, accel_actions)
        ]
        sat_snapshots = [
            drivers[int(local_idx)].build_sat_stage_snapshot(world_state)
            for local_idx, world_state in zip(local_indices, sat_world_states)
        ]
    sat_pair_indices = batched_policy_sat_pair_indices(follow_actor, sat_snapshots, follow_device, deterministic)
    if looks_like_driver_group(drivers):
        bw_snapshots = drivers.run_sat_and_prepare_bw_many(sat_pair_indices, indices=local_indices)
    else:
        sat_actions = [
            drivers[int(local_idx)].decode_sat_pair_actions([], pair_indices)
            for local_idx, pair_indices in zip(local_indices, sat_pair_indices)
        ]
        bw_world_states = [
            drivers[int(local_idx)].run_sat_stage(action)
            for local_idx, action in zip(local_indices, sat_actions)
        ]
        bw_snapshots = [
            drivers[int(local_idx)].build_bw_stage_snapshot(world_state)
            for local_idx, world_state in zip(local_indices, bw_world_states)
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


def _rollout_access_panel_scores(
    eval_drivers,
    panel_candidates: list[dict[str, Any]],
    *,
    follow_actor,
    follow_device: torch.device,
    follow_deterministic: bool,
    bw_deterministic_readout: str,
    bw_deterministic_opt_steps: int,
    bw_deterministic_step_size: float,
    gamma: float,
    k_steps: int,
) -> list[float]:
    if not panel_candidates:
        return []
    if len(panel_candidates) > len(eval_drivers):
        raise ValueError("panel size exceeds available eval driver capacity")
    totals = [0.0 for _ in panel_candidates]
    current_actions = [np.asarray(candidate["bw_action"], dtype=np.float32) for candidate in panel_candidates]
    active_local_indices = list(range(len(panel_candidates)))
    discount = 1.0
    for step_idx in range(max(int(k_steps), 0)):
        if not active_local_indices:
            break
        next_active_local_indices: list[int] = []
        next_world_states: list[Any] = []
        if looks_like_driver_group(eval_drivers):
            if step_idx == 0:
                eval_drivers.load_bw_stage_state_many(
                    [panel_candidates[int(local_idx)]["snapshot_state"] for local_idx in active_local_indices],
                    indices=active_local_indices,
                )
                eval_drivers.override_access_stage_state_many(
                    [np.asarray(panel_candidates[int(local_idx)]["assoc"], dtype=np.int32) for local_idx in active_local_indices],
                    [[list(c) for c in panel_candidates[int(local_idx)]["candidates"]] for local_idx in active_local_indices],
                    indices=active_local_indices,
                )
            step_results, next_world_states_batch = eval_drivers.execute_bw_and_prepare_next_accel_many(
                [np.asarray(current_actions[int(local_idx)], dtype=np.float32) for local_idx in active_local_indices],
                indices=active_local_indices,
            )
            for local_idx, step_result, next_world_state in zip(active_local_indices, step_results, next_world_states_batch):
                reward = float(next(iter(step_result.rewards.values())))
                totals[int(local_idx)] += discount * reward
                done = bool(next(iter(step_result.terminations.values())) or next(iter(step_result.truncations.values())))
                if done or step_idx + 1 >= int(k_steps):
                    continue
                next_active_local_indices.append(int(local_idx))
                next_world_states.append(next_world_state)
        else:
            for local_idx in active_local_indices:
                if step_idx == 0:
                    step_result, next_world_state = _execute_access_and_prepare_next_accel(
                        eval_drivers[int(local_idx)],
                        panel_candidates[int(local_idx)],
                    )
                else:
                    step_result, next_world_state = eval_drivers[int(local_idx)].execute_stage_bw_and_prepare_next_accel(
                        np.asarray(current_actions[int(local_idx)], dtype=np.float32)
                    )
                reward = float(next(iter(step_result.rewards.values())))
                totals[int(local_idx)] += discount * reward
                done = bool(next(iter(step_result.terminations.values())) or next(iter(step_result.truncations.values())))
                if done or step_idx + 1 >= int(k_steps):
                    continue
                next_active_local_indices.append(int(local_idx))
                next_world_states.append(next_world_state)
        if not next_active_local_indices or step_idx + 1 >= int(k_steps):
            break
        discount *= float(gamma)
        next_actions = _policy_followup_bw_actions(
            eval_drivers,
            local_indices=next_active_local_indices,
            next_world_states=next_world_states,
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
    return [float(total) for total in totals]


def _evaluate_access_oracle(
    *,
    cfg,
    actor,
    device: torch.device,
    episodes: int,
    episode_seed_base: int | None,
    num_envs: int,
    k_steps: int,
    overlap_per_uav_keep: int,
    overlap_per_user_max_uav: int,
    include_uniform: bool,
    seed: int,
    bw_deterministic_opt_steps: int,
    bw_deterministic_step_size: float,
    dump_bank: bool,
    eval_vec_backend: str,
) -> tuple[dict[str, Any], list[dict[str, float]], list[dict[str, Any]], list[dict[str, Any]]]:
    active_slots = max(min(int(num_envs), int(episodes)), 1)
    env_group = make_structured_env_group(cfg, num_envs=active_slots, backend="sync")
    drivers = _as_driver_list(env_group)
    eval_panel_capacity = max(1, 2 + len(ACCESS_RULES) * (2 if include_uniform else 1) + int(include_uniform))
    eval_group = make_structured_env_group(cfg, num_envs=eval_panel_capacity, backend=str(eval_vec_backend))
    eval_drivers = eval_group if looks_like_driver_group(eval_group) else _as_driver_list(eval_group)
    rows: list[dict[str, float]] = []
    step_rows: list[dict[str, Any]] = []
    bank_rows: list[dict[str, Any]] = []
    selected_gain_vs_hard_heur: list[float] = []
    selected_gain_vs_hard_policy: list[float] = []
    selected_beats_hard_heur: list[float] = []
    selected_beats_hard_policy: list[float] = []
    selection_source_counts: dict[str, int] = {}
    slot_episode = list(range(active_slots))
    slot_active = [True for _ in range(active_slots)]
    slot_acc = [_new_episode_accumulator() for _ in range(active_slots)]
    next_episode = active_slots
    initial_seeds = [
        None if episode_seed_base is None else int(episode_seed_base) + slot
        for slot in range(active_slots)
    ]
    _ = np.random.default_rng(int(seed))
    for slot, driver in enumerate(drivers):
        driver.env.reset(seed=initial_seeds[slot])
    try:
        while len(rows) < int(episodes):
            active_indices = [slot for slot, is_active in enumerate(slot_active) if is_active]
            if not active_indices:
                break

            accel_world_states = [drivers[slot].begin_step() for slot in active_indices]
            accel_actions = batched_policy_accel_actions(actor, accel_world_states, device, True)
            sat_world_states = [
                drivers[slot].run_accel_stage(action)
                for slot, action in zip(active_indices, accel_actions)
            ]
            sat_snapshots = [
                drivers[slot].build_sat_stage_snapshot(world_state)
                for slot, world_state in zip(active_indices, sat_world_states)
            ]
            sat_pair_indices = batched_policy_sat_pair_indices(actor, sat_snapshots, device, True)
            sat_actions = [
                drivers[slot].decode_sat_pair_actions([], pair_indices)
                for slot, pair_indices in zip(active_indices, sat_pair_indices)
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
            hard_policy_det_actions = _batched_bw_actions_from_snapshots(
                actor,
                bw_snapshots,
                device=device,
                deterministic=True,
                bw_deterministic_readout="latent_mean_pushforward",
                bw_deterministic_opt_steps=int(bw_deterministic_opt_steps),
                bw_deterministic_step_size=float(bw_deterministic_step_size),
            )["actions"]

            selected_candidates: list[dict[str, Any]] = []
            gamma = float(cfg.gamma)
            for local_slot, slot in enumerate(active_indices):
                panel = _build_access_candidate_panel(
                    driver=drivers[slot],
                    snapshot_state=snapshot_states[local_slot],
                    hard_policy_det_action=np.asarray(hard_policy_det_actions[local_slot], dtype=np.float32),
                    overlap_per_uav_keep=int(overlap_per_uav_keep),
                    overlap_per_user_max_uav=int(overlap_per_user_max_uav),
                    include_uniform=bool(include_uniform),
                )
                candidate_scores = _rollout_access_panel_scores(
                    eval_drivers,
                    panel,
                    follow_actor=actor,
                    follow_device=device,
                    follow_deterministic=True,
                    bw_deterministic_readout="latent_mean_pushforward",
                    bw_deterministic_opt_steps=int(bw_deterministic_opt_steps),
                    bw_deterministic_step_size=float(bw_deterministic_step_size),
                    gamma=gamma,
                    k_steps=int(k_steps),
                )
                best_idx = int(np.argmax(np.asarray(candidate_scores, dtype=np.float64)))
                best_candidate = dict(panel[best_idx])
                selected_candidates.append(best_candidate)
                selection_source_counts[str(best_candidate["name"])] = int(
                    selection_source_counts.get(str(best_candidate["name"]), 0) + 1
                )
                score_map = {
                    str(candidate["name"]): float(score)
                    for candidate, score in zip(panel, candidate_scores)
                }
                hard_heur_score = float(score_map.get("hard_assoc_heurbw", 0.0))
                hard_policy_score = float(score_map.get("hard_assoc_policydet", 0.0))
                best_score = float(candidate_scores[best_idx])
                selected_gain_vs_hard_heur.append(float(best_score - hard_heur_score))
                selected_gain_vs_hard_policy.append(float(best_score - hard_policy_score))
                selected_beats_hard_heur.append(float(best_score > hard_heur_score + 1.0e-9))
                selected_beats_hard_policy.append(float(best_score > hard_policy_score + 1.0e-9))
                assoc_reassigned = int(
                    np.sum(
                        np.asarray(best_candidate["assoc"], dtype=np.int32)
                        != np.asarray(snapshot_states[local_slot]["stage_assoc"], dtype=np.int32)
                    )
                )
                step_rows.append(
                    {
                        "episode": int(slot_episode[slot]),
                        "slot": int(slot),
                        "t": int(snapshot_states[local_slot].get("env_state", {}).get("t", 0)),
                        "candidate_count": int(len(panel)),
                        "selected_name": str(best_candidate["name"]),
                        "selected_score": best_score,
                        "hard_assoc_heurbw_score": hard_heur_score,
                        "hard_assoc_policydet_score": hard_policy_score,
                        "selected_minus_hard_heurbw": float(best_score - hard_heur_score),
                        "selected_minus_hard_policydet": float(best_score - hard_policy_score),
                        "assoc_reassigned_users": int(assoc_reassigned),
                    }
                )
                if dump_bank:
                    bank_rows.append(
                        {
                            "episode": int(slot_episode[slot]),
                            "t": int(snapshot_states[local_slot].get("env_state", {}).get("t", 0)),
                            "snapshot_state": snapshot_states[local_slot],
                            "selected_name": str(best_candidate["name"]),
                            "selected_assoc": np.asarray(best_candidate["assoc"], dtype=np.int32).copy(),
                            "selected_candidates": [list(int(idx) for idx in cand) for cand in best_candidate["candidates"]],
                            "selected_bw_action": np.asarray(best_candidate["bw_action"], dtype=np.float32).copy(),
                            "hard_policy_det_action": np.asarray(hard_policy_det_actions[local_slot], dtype=np.float32).copy(),
                            "score_hard_assoc_heurbw": float(hard_heur_score),
                            "score_hard_assoc_policydet": float(hard_policy_score),
                            "score_selected": float(best_score),
                        }
                    )

            for local_slot, slot in enumerate(active_indices):
                step_result = _apply_candidate_and_step(drivers[slot], selected_candidates[local_slot])
                reward_parts = dict(getattr(drivers[slot].env, "last_reward_parts", {}) or {})
                acc = slot_acc[slot]
                acc["reward_sum"] += float(next(iter(step_result.rewards.values())))
                acc["steps"] += 1.0
                acc["processed_ratio_sum"] += float(reward_parts.get("processed_ratio_eval", 0.0))
                acc["drop_ratio_sum"] += float(reward_parts.get("drop_ratio_eval", 0.0))
                acc["pre_backlog_sum"] += float(reward_parts.get("pre_backlog_steps_eval", 0.0))
                acc["d_sys_sum"] += float(reward_parts.get("D_sys_report", 0.0))
                acc["x_acc_sum"] += float(reward_parts.get("x_acc", 0.0))
                acc["x_rel_sum"] += float(reward_parts.get("x_rel", 0.0))
                acc["g_pre_sum"] += float(reward_parts.get("g_pre", 0.0))
                acc["d_pre_sum"] += float(reward_parts.get("d_pre", 0.0))
                acc["collision_any"] = max(acc["collision_any"], float(reward_parts.get("collision_event", 0.0)))
                done = bool(next(iter(step_result.terminations.values())) or next(iter(step_result.truncations.values())))
                if not done:
                    continue
                rows.append(_episode_row_from_accumulator(slot_episode[slot], acc))
                slot_acc[slot] = _new_episode_accumulator()
                if next_episode < int(episodes):
                    seed_next = None if episode_seed_base is None else int(episode_seed_base) + next_episode
                    drivers[slot].env.reset(seed=seed_next)
                    slot_episode[slot] = int(next_episode)
                    next_episode += 1
                else:
                    slot_active[slot] = False
                    slot_episode[slot] = -1

        rows.sort(key=lambda row: int(row["episode"]))
        totals = _episode_metrics_template()
        for row in rows:
            for key in totals:
                totals[key] += float(row[key])
        scale = 1.0 / float(max(len(rows), 1))
        summary = {key: float(val * scale) for key, val in totals.items()}
        summary.update(
            {
                "selection_step_count": int(len(step_rows)),
                "selected_gain_vs_hard_heurbw": _summarize(selected_gain_vs_hard_heur),
                "selected_gain_vs_hard_policydet": _summarize(selected_gain_vs_hard_policy),
                "selected_beats_hard_heurbw_frac": _mean(selected_beats_hard_heur),
                "selected_beats_hard_policydet_frac": _mean(selected_beats_hard_policy),
                "selection_source_hist": {str(k): int(v) for k, v in sorted(selection_source_counts.items())},
            }
        )
        return summary, rows, step_rows, bank_rows
    finally:
        close_structured_env_group(env_group)
        close_structured_env_group(eval_group)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--update", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--episode_seed_base", type=int, default=63000)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--k_steps", type=int, default=2)
    parser.add_argument("--overlap_per_uav_keep", type=int, default=6)
    parser.add_argument("--overlap_per_user_max_uav", type=int, default=2)
    parser.add_argument("--include_uniform", action="store_true")
    parser.add_argument("--bw_deterministic_opt_steps", type=int, default=8)
    parser.add_argument("--bw_deterministic_step_size", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dump_bank", action="store_true")
    parser.add_argument("--eval_vec_backend", choices=["sync", "subproc"], default="sync")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_all_seeds(int(args.seed))
    run_dir = Path(args.run_dir)
    cfg_path = args.config or str(run_dir / "config_source.yaml")
    cfg = load_config(cfg_path)
    checkpoint = _resolve_checkpoint(run_dir, args.checkpoint, args.update)
    device = torch.device(args.device)
    actor = _load_actor(
        cfg,
        checkpoint,
        hidden_dim=None if args.hidden_dim is None else int(args.hidden_dim),
        embed_dim=None if args.embed_dim is None else int(args.embed_dim),
        device=device,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, episode_rows, step_rows, bank_rows = _evaluate_access_oracle(
        cfg=cfg,
        actor=actor,
        device=device,
        episodes=int(args.episodes),
        episode_seed_base=args.episode_seed_base,
        num_envs=int(args.num_envs),
        k_steps=int(args.k_steps),
        overlap_per_uav_keep=int(args.overlap_per_uav_keep),
        overlap_per_user_max_uav=int(args.overlap_per_user_max_uav),
        include_uniform=bool(args.include_uniform),
        seed=int(args.seed),
        bw_deterministic_opt_steps=int(args.bw_deterministic_opt_steps),
        bw_deterministic_step_size=float(args.bw_deterministic_step_size),
        dump_bank=bool(args.dump_bank),
        eval_vec_backend=str(args.eval_vec_backend),
    )

    _write_csv(out_dir / "per_episode.csv", episode_rows)
    _write_csv(out_dir / "select_steps.csv", step_rows)
    if bool(args.dump_bank):
        torch.save(bank_rows, out_dir / "oracle_bank.pt")
    payload = {
        "config": str(Path(cfg_path)),
        "checkpoint": str(checkpoint),
        "episodes": int(args.episodes),
        "episode_seed_base": int(args.episode_seed_base) if args.episode_seed_base is not None else None,
        "num_envs": int(args.num_envs),
        "k_steps": int(args.k_steps),
        "overlap_per_uav_keep": int(args.overlap_per_uav_keep),
        "overlap_per_user_max_uav": int(args.overlap_per_user_max_uav),
        "include_uniform": bool(args.include_uniform),
        "dump_bank": bool(args.dump_bank),
        "eval_vec_backend": str(args.eval_vec_backend),
        "bank_size": int(len(bank_rows)),
        "summary": summary,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        f"Summary: reward={summary['reward_sum']:.4f} "
        f"processed={summary['processed_ratio_eval']:.4f} "
        f"drop={summary['drop_ratio_eval']:.4f} "
        f"pre_backlog={summary['pre_backlog_steps_eval']:.4f} "
        f"beats_hard_heurbw={summary['selected_beats_hard_heurbw_frac']:.4f} "
        f"beats_hard_policydet={summary['selected_beats_hard_policydet_frac']:.4f}"
    )


if __name__ == "__main__":
    main()
