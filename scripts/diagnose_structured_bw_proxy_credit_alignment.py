from __future__ import annotations

import argparse
import copy
import json
import math
import os
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
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import (
    _split_cpu_tensor_by_counts,
    _split_dataclass_by_counts,
)
from sagin_marl.rl.structured_parallel_eval import (
    batched_policy_accel_actions,
    batched_policy_bw_outputs,
    batched_policy_sat_pair_indices,
)
from sagin_marl.rl.structured_types import LocalBwState
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


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
                hits += 0.5
                total += 1.0
                continue
            total += 1.0
            if pred_diff * truth_diff > 0.0:
                hits += 1.0
    if total <= 0.0:
        return 0.0
    return float(hits / total)


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


def _bucket_mean(counts: list[int], values: list[float]) -> list[dict[str, float]]:
    buckets: dict[int, list[float]] = {}
    for count, value in zip(counts, values):
        buckets.setdefault(int(count), []).append(float(value))
    rows: list[dict[str, float]] = []
    for count in sorted(buckets):
        arr = np.asarray(buckets[count], dtype=np.float64)
        rows.append({"count": float(count), "n": float(arr.size), "mean": float(np.mean(arr))})
    return rows


def _load_actor(run_dir: Path, update: int, device: torch.device):
    cfg = load_config(str(run_dir / "config_source.yaml"))
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    actor_ckpt = run_dir / f"actor_u{int(update):04d}.pt"
    load_checkpoint_forgiving(bundle.actor, str(actor_ckpt), map_location=device, strict=True)
    bundle.actor.to(device).eval()
    return cfg, bundle.actor


def _latent_mask_from_bw_state(local_state: LocalBwState) -> tuple[torch.Tensor, torch.Tensor]:
    valid = (local_state.user_mask > 0.5) & (local_state.bw_valid_mask > 0.5)
    latent = valid.clone()
    if valid.shape[-1] > 0:
        valid_count = valid.sum(dim=-1)
        ref_idx = torch.where(
            valid,
            torch.arange(valid.shape[-1], device=valid.device, dtype=torch.long).view(1, -1).expand_as(valid),
            torch.full_like(valid, -1, dtype=torch.long),
        ).amax(dim=-1)
        active_rows = torch.nonzero(valid_count > 1, as_tuple=False).flatten()
        if active_rows.numel() > 0:
            latent[active_rows, ref_idx[active_rows]] = False
    return valid, latent


def _effective_logits(local_state: LocalBwState, loc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid_mask, latent_mask = _latent_mask_from_bw_state(local_state)
    valid_np = valid_mask.detach().cpu().numpy()
    latent_np = latent_mask.detach().cpu().numpy()
    effective = np.zeros_like(loc, dtype=np.float32)
    effective[latent_np] = np.asarray(loc[latent_np], dtype=np.float32)
    return valid_np, effective


def _reallocate_toward_slot(
    base_action: np.ndarray,
    valid_mask: np.ndarray,
    target_slot: int,
    delta: float,
    *,
    eps: float = 1.0e-8,
) -> tuple[np.ndarray | None, float]:
    valid = np.asarray(valid_mask, dtype=bool)
    if not valid[int(target_slot)]:
        return None, 0.0
    donor_mask = valid.copy()
    donor_mask[int(target_slot)] = False
    donor_mass = float(np.sum(np.asarray(base_action, dtype=np.float64)[donor_mask]))
    if donor_mass <= eps:
        return None, 0.0
    used_delta = min(float(delta), 0.5 * donor_mass)
    if used_delta <= eps:
        return None, 0.0
    out = np.asarray(base_action, dtype=np.float32).copy()
    scale = float((donor_mass - used_delta) / max(donor_mass, eps))
    out[donor_mask] = out[donor_mask] * scale
    out[int(target_slot)] = float(out[int(target_slot)] + used_delta)
    out[~valid] = 0.0
    norm = float(np.sum(out[valid]))
    if norm <= eps:
        return None, 0.0
    out[valid] = out[valid] / norm
    return out.astype(np.float32, copy=False), float(used_delta)


def _step_reward(driver: StructuredControlDriver, bw_action: np.ndarray) -> float:
    step = driver.execute_stage_bw_and_step(bw_action)
    return float(next(iter(step.rewards.values())))


def _step_reward_and_result(driver: StructuredControlDriver, bw_action: np.ndarray) -> tuple[float, Any]:
    step = driver.execute_stage_bw_and_step(bw_action)
    reward = float(next(iter(step.rewards.values())))
    return reward, step


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


def _proxy_reward_components(
    driver: StructuredControlDriver,
    bw_action: np.ndarray,
    realized_arrival: np.ndarray,
    rate_matrix: np.ndarray,
) -> dict[str, float]:
    env = driver.env
    cfg = env.cfg
    if driver._stage_assoc is None or driver._stage_candidates is None:
        raise RuntimeError("bw proxy requires accel stage cache")
    if str(getattr(cfg, "reward_mode", "dense") or "dense").strip().lower() != "controllable_flow":
        raise NotImplementedError("Current bw proxy probe supports reward_mode=controllable_flow only.")

    action_dict = driver._dummy_action_dict()
    bw_arr = np.asarray(bw_action, dtype=np.float32)
    for u, agent in enumerate(env.agents):
        action_dict[agent]["bw_alloc"] = bw_arr[u]

    access_rates, _ = env._compute_access_rates(driver._stage_assoc, driver._stage_candidates, action_dict, record_exec=False)
    arrival = np.asarray(realized_arrival, dtype=np.float32)
    q_gu_before = np.asarray(env.gu_queue, dtype=np.float32) + arrival
    gu_outflow = np.minimum(q_gu_before, np.asarray(access_rates, dtype=np.float32) * float(cfg.tau0)).astype(np.float32)
    q_gu_after_raw = q_gu_before - gu_outflow
    gu_drop = np.maximum(q_gu_after_raw - float(cfg.queue_max_gu), 0.0).astype(np.float32)
    q_gu_after = np.minimum(q_gu_after_raw, float(cfg.queue_max_gu)).astype(np.float32)

    assoc = np.asarray(driver._stage_assoc, dtype=np.int32)
    valid_assoc = assoc >= 0
    if np.any(valid_assoc):
        inflow_uav = np.bincount(
            assoc[valid_assoc],
            weights=gu_outflow[valid_assoc],
            minlength=int(cfg.num_uav),
        ).astype(np.float32)
    else:
        inflow_uav = np.zeros((cfg.num_uav,), dtype=np.float32)

    q_uav_before = np.asarray(env.uav_queue, dtype=np.float32) + inflow_uav
    total_rate = np.sum(np.asarray(rate_matrix, dtype=np.float32), axis=1).astype(np.float32)
    uav_outflow = np.minimum(q_uav_before, total_rate * float(cfg.tau0)).astype(np.float32)
    q_uav_after_raw = q_uav_before - uav_outflow
    uav_drop = np.maximum(q_uav_after_raw - float(cfg.queue_max_uav), 0.0).astype(np.float32)
    q_uav_after = np.minimum(q_uav_after_raw, float(cfg.queue_max_uav)).astype(np.float32)

    outflow_matrix = np.zeros_like(rate_matrix, dtype=np.float32)
    mask = total_rate > 0.0
    if np.any(mask):
        outflow_matrix[mask] = (rate_matrix[mask] / total_rate[mask, None]) * uav_outflow[mask, None]
    sat_incoming = np.sum(outflow_matrix, axis=0).astype(np.float32)
    compute_rate = float(cfg.sat_cpu_freq) / max(float(cfg.task_cycles_per_bit), 1e-9)
    q_sat_before = np.asarray(env.sat_queue, dtype=np.float32) + sat_incoming
    sat_processed = np.minimum(q_sat_before, compute_rate * float(cfg.tau0)).astype(np.float32)
    q_sat_after_raw = q_sat_before - sat_processed
    sat_drop = np.maximum(q_sat_after_raw - float(cfg.queue_max_sat), 0.0).astype(np.float32)

    arrival_ref = float(env._arrival_ref())
    x_acc = float(np.sum(gu_outflow) / arrival_ref)
    x_rel = float(np.sum(sat_incoming) / arrival_ref)
    d_pre = float((np.sum(gu_drop) + np.sum(uav_drop)) / arrival_ref)
    b_pre_steps = float((np.sum(q_gu_after) + np.sum(q_uav_after)) / arrival_ref)
    processed_ratio_eval = float(np.sum(sat_processed) / arrival_ref)
    drop_ratio_eval = float((np.sum(gu_drop) + np.sum(uav_drop) + np.sum(sat_drop)) / arrival_ref)
    pre_backlog_steps_eval = b_pre_steps
    reward_proxy = (
        float(getattr(cfg, "reward_w_access", 0.5) or 0.0) * x_acc
        + float(getattr(cfg, "reward_w_relay", 0.5) or 0.0) * x_rel
        - float(getattr(cfg, "reward_w_pre_drop", 1.0) or 0.0) * d_pre
        - float(getattr(cfg, "reward_w_pre_backlog", 0.08) or 0.0) * math.log1p(b_pre_steps)
    )
    return {
        "reward_proxy": reward_proxy,
        "access_proxy": x_acc,
        "relay_proxy": x_rel,
        "drop_proxy": -d_pre,
        "backlog_proxy": -math.log1p(b_pre_steps),
        "processed_ratio_eval": processed_ratio_eval,
        "drop_ratio_eval": drop_ratio_eval,
        "pre_backlog_steps_eval": pre_backlog_steps_eval,
    }


def diagnose_update(
    run_dir: Path,
    update: int,
    *,
    episodes: int,
    episode_seed_base: int,
    deterministic: bool,
    device: torch.device,
    num_envs: int,
    cf_delta: float,
    max_agent_samples: int,
) -> dict[str, Any]:
    cfg, actor = _load_actor(run_dir, int(update), device)
    active_slots = max(min(int(num_envs), int(episodes)), 1)
    drivers = _make_local_drivers(cfg, active_slots)
    slot_active = [True for _ in range(active_slots)]
    next_episode = active_slots
    sampled = 0
    counterfactual_evals = 0

    valid_user_count: list[int] = []
    exact_best_gain: list[float] = []
    flow_proxy_vs_exact_spearman: list[float] = []
    flow_proxy_vs_exact_top1_hit: list[float] = []
    flow_proxy_vs_exact_pairwise_acc: list[float] = []
    access_proxy_vs_exact_spearman: list[float] = []
    access_proxy_vs_exact_top1_hit: list[float] = []
    access_proxy_vs_exact_pairwise_acc: list[float] = []
    loc_vs_flow_proxy_spearman: list[float] = []
    loc_vs_access_proxy_spearman: list[float] = []
    flow_proxy_vs_exact_pearson: list[float] = []
    access_proxy_vs_exact_pearson: list[float] = []

    initial_seeds = [int(episode_seed_base) + slot for slot in range(active_slots)]
    for slot, seed in enumerate(initial_seeds):
        drivers[slot].env.reset(seed=int(seed))

    try:
        while any(slot_active) and sampled < int(max_agent_samples):
            active_indices = [slot for slot, is_active in enumerate(slot_active) if is_active]
            if not active_indices:
                break

            accel_world_states = [drivers[slot].begin_step() for slot in active_indices]
            accel_actions = batched_policy_accel_actions(actor, accel_world_states, device, deterministic)
            sat_world_states = [
                drivers[slot].run_accel_stage(action)
                for slot, action in zip(active_indices, accel_actions)
            ]
            sat_snapshots = [
                drivers[slot].build_sat_stage_snapshot(world_state)
                for slot, world_state in zip(active_indices, sat_world_states)
            ]
            sat_pair_indices = batched_policy_sat_pair_indices(actor, sat_snapshots, device, deterministic)
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
            bw_eval = batched_policy_bw_outputs(actor, bw_snapshots, device, deterministic)
            bw_state_groups = _split_dataclass_by_counts(bw_eval.local_state, bw_eval.agent_counts)
            loc_groups = [
                piece.numpy()
                for piece in _split_cpu_tensor_by_counts(bw_eval.out.loc, bw_eval.agent_counts)
            ]

            for local_slot, slot in enumerate(active_indices):
                if sampled >= int(max_agent_samples):
                    break
                driver = drivers[slot]
                pre_driver = copy.deepcopy(driver)
                state_group = bw_state_groups[local_slot]
                loc_group = np.asarray(loc_groups[local_slot], dtype=np.float32)
                base_action_env = np.asarray(bw_eval.actions[local_slot], dtype=np.float32)
                valid_mask_group, effective_loc_group = _effective_logits(state_group, loc_group)

                base_reward, live_step = _step_reward_and_result(driver, base_action_env)
                realized_arrival = np.asarray(driver.env.last_gu_arrival, dtype=np.float32)
                sat_pos = np.asarray(pre_driver._stage_sat_pos, dtype=np.float32)
                sat_vel = np.asarray(pre_driver._stage_sat_vel, dtype=np.float32)
                sat_selection = pre_driver._stage_sat_selection or [[] for _ in range(cfg.num_uav)]
                rate_matrix, _ = pre_driver.env._compute_backhaul_rates(sat_pos, sat_vel, sat_selection)
                base_proxy = _proxy_reward_components(pre_driver, base_action_env, realized_arrival, rate_matrix)

                for u in range(int(cfg.num_uav)):
                    if sampled >= int(max_agent_samples):
                        break
                    valid = np.asarray(valid_mask_group[u], dtype=bool)
                    valid_count = int(np.sum(valid))
                    if valid_count <= 1:
                        continue
                    base_action_u = np.asarray(base_action_env[u], dtype=np.float32)
                    exact_credit: list[float] = []
                    flow_credit: list[float] = []
                    access_credit: list[float] = []
                    target_slots = np.flatnonzero(valid)
                    evaluated_target_slots: list[int] = []

                    for target_slot in target_slots.tolist():
                        cf_action_u, _used_delta = _reallocate_toward_slot(
                            base_action_u,
                            valid,
                            int(target_slot),
                            float(cf_delta),
                        )
                        if cf_action_u is None:
                            continue
                        cf_action_env = np.asarray(base_action_env, dtype=np.float32).copy()
                        cf_action_env[u] = cf_action_u
                        cf_reward = _step_reward(copy.deepcopy(pre_driver), cf_action_env)
                        cf_proxy = _proxy_reward_components(pre_driver, cf_action_env, realized_arrival, rate_matrix)
                        exact_credit.append(float(cf_reward - base_reward))
                        flow_credit.append(float(cf_proxy["reward_proxy"] - base_proxy["reward_proxy"]))
                        access_credit.append(float(cf_proxy["access_proxy"] - base_proxy["access_proxy"]))
                        evaluated_target_slots.append(int(target_slot))
                        counterfactual_evals += 1

                    if len(exact_credit) <= 1:
                        continue

                    exact_arr = np.asarray(exact_credit, dtype=np.float64)
                    flow_arr = np.asarray(flow_credit, dtype=np.float64)
                    access_arr = np.asarray(access_credit, dtype=np.float64)
                    loc_eval = np.asarray(effective_loc_group[u, evaluated_target_slots], dtype=np.float64)
                    valid_user_count.append(valid_count)
                    exact_best_gain.append(float(np.max(exact_arr)))

                    flow_proxy_vs_exact_spearman.append(_safe_spearman_desc(flow_arr, exact_arr))
                    flow_proxy_vs_exact_top1_hit.append(
                        float(int(np.argmax(flow_arr)) == int(np.argmax(exact_arr)))
                    )
                    flow_proxy_vs_exact_pairwise_acc.append(
                        _pairwise_concordance_desc(flow_arr, exact_arr)
                    )
                    access_proxy_vs_exact_spearman.append(_safe_spearman_desc(access_arr, exact_arr))
                    access_proxy_vs_exact_top1_hit.append(
                        float(int(np.argmax(access_arr)) == int(np.argmax(exact_arr)))
                    )
                    access_proxy_vs_exact_pairwise_acc.append(
                        _pairwise_concordance_desc(access_arr, exact_arr)
                    )
                    loc_vs_flow_proxy_spearman.append(_safe_spearman_desc(loc_eval, flow_arr))
                    loc_vs_access_proxy_spearman.append(_safe_spearman_desc(loc_eval, access_arr))
                    flow_proxy_vs_exact_pearson.append(float(_safe_corr(flow_arr.tolist(), exact_arr.tolist()) or 0.0))
                    access_proxy_vs_exact_pearson.append(float(_safe_corr(access_arr.tolist(), exact_arr.tolist()) or 0.0))
                    sampled += 1

                done = bool(any(live_step.terminations.values()) or any(live_step.truncations.values()))
                if done:
                    if next_episode < int(episodes):
                        driver.env.reset(seed=int(episode_seed_base) + int(next_episode))
                        next_episode += 1
                    else:
                        slot_active[slot] = False
    finally:
        _close_local_drivers(drivers)

    return {
        "update": int(update),
        "episodes": int(episodes),
        "episode_seed_base": int(episode_seed_base),
        "deterministic": bool(deterministic),
        "num_envs": int(active_slots),
        "sample_count": int(sampled),
        "counterfactual_eval_count": int(counterfactual_evals),
        "cf_delta": float(cf_delta),
        "valid_user_count": _summarize([float(x) for x in valid_user_count]),
        "exact_best_gain": _summarize(exact_best_gain),
        "flow_proxy_vs_exact_spearman": _summarize(flow_proxy_vs_exact_spearman),
        "flow_proxy_vs_exact_top1_hit": _summarize(flow_proxy_vs_exact_top1_hit),
        "flow_proxy_vs_exact_pairwise_acc": _summarize(flow_proxy_vs_exact_pairwise_acc),
        "flow_proxy_vs_exact_pearson": _summarize(flow_proxy_vs_exact_pearson),
        "access_proxy_vs_exact_spearman": _summarize(access_proxy_vs_exact_spearman),
        "access_proxy_vs_exact_top1_hit": _summarize(access_proxy_vs_exact_top1_hit),
        "access_proxy_vs_exact_pairwise_acc": _summarize(access_proxy_vs_exact_pairwise_acc),
        "access_proxy_vs_exact_pearson": _summarize(access_proxy_vs_exact_pearson),
        "loc_vs_flow_proxy_spearman": _summarize(loc_vs_flow_proxy_spearman),
        "loc_vs_access_proxy_spearman": _summarize(loc_vs_access_proxy_spearman),
        "flow_spearman_by_valid_user_count": _bucket_mean(valid_user_count, flow_proxy_vs_exact_spearman),
        "access_spearman_by_valid_user_count": _bucket_mean(valid_user_count, access_proxy_vs_exact_spearman),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--updates", type=int, nargs="+", default=[100, 200])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-seed-base", type=int, default=86000)
    parser.add_argument("--policy-mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--cf-delta", type=float, default=0.05)
    parser.add_argument("--max-agent-samples", type=int, default=96)
    parser.add_argument("--out-name", type=str, default="structured_bw_proxy_credit_alignment.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "episodes": int(args.episodes),
        "episode_seed_base": int(args.episode_seed_base),
        "policy_mode": str(args.policy_mode),
        "num_envs": int(args.num_envs),
        "cf_delta": float(args.cf_delta),
        "max_agent_samples": int(args.max_agent_samples),
        "updates": {},
    }

    for update in args.updates:
        update_summary = diagnose_update(
            run_dir,
            int(update),
            episodes=int(args.episodes),
            episode_seed_base=int(args.episode_seed_base) + int(update) * 1000,
            deterministic=(args.policy_mode == "deterministic"),
            device=device,
            num_envs=int(args.num_envs),
            cf_delta=float(args.cf_delta),
            max_agent_samples=int(args.max_agent_samples),
        )
        summary["updates"][f"u{int(update):04d}"] = update_summary
        print(
            json.dumps(
                {
                    "update": int(update),
                    "sample_count": update_summary["sample_count"],
                    "flow_spearman_mean": update_summary["flow_proxy_vs_exact_spearman"]["mean"],
                    "flow_top1_hit_mean": update_summary["flow_proxy_vs_exact_top1_hit"]["mean"],
                    "access_spearman_mean": update_summary["access_proxy_vs_exact_spearman"]["mean"],
                    "access_top1_hit_mean": update_summary["access_proxy_vs_exact_top1_hit"]["mean"],
                }
            )
        )

    out_path = run_dir / str(args.out_name)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
