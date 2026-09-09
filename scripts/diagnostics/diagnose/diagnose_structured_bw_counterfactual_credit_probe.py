from __future__ import annotations

import argparse
import copy
import json
import os
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
                total += 1.0
                hits += 0.5
                continue
            total += 1.0
            if pred_diff * truth_diff > 0.0:
                hits += 1.0
    if total <= 0.0:
        return 0.0
    return float(hits / total)


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


def _step_reward_and_parts(driver: StructuredControlDriver, bw_action: np.ndarray) -> tuple[float, dict[str, float]]:
    step = driver.execute_stage_bw_and_step(bw_action)
    reward = float(next(iter(step.rewards.values())))
    parts_raw = dict(getattr(driver.env, "last_reward_parts", {}) or {})
    parts: dict[str, float] = {}
    for key, value in parts_raw.items():
        if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
            parts[str(key)] = float(value)
    parts.setdefault("reward_raw", reward)
    parts.setdefault("processed_ratio_eval", 0.0)
    parts.setdefault("drop_ratio_eval", 0.0)
    parts.setdefault("pre_backlog_steps_eval", 0.0)
    return reward, parts


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
    cf_delta_used: list[float] = []
    best_reward_gain: list[float] = []
    worst_reward_gain: list[float] = []
    positive_best_reward_gain: list[float] = []
    positive_target_fraction: list[float] = []

    loc_vs_reward_credit_spearman: list[float] = []
    action_vs_reward_credit_spearman: list[float] = []
    loc_vs_reward_credit_top1_hit: list[float] = []
    action_vs_reward_credit_top1_hit: list[float] = []
    loc_vs_reward_credit_pairwise_acc: list[float] = []
    action_vs_reward_credit_pairwise_acc: list[float] = []
    loc_vs_reward_raw_credit_spearman: list[float] = []
    loc_vs_processed_gain_spearman: list[float] = []
    loc_vs_drop_gain_spearman: list[float] = []
    loc_vs_backlog_gain_spearman: list[float] = []

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
                state_group = bw_state_groups[local_slot]
                loc_group = np.asarray(loc_groups[local_slot], dtype=np.float32)
                base_action_env = np.asarray(bw_eval.actions[local_slot], dtype=np.float32)
                valid_mask_group, effective_loc_group = _effective_logits(state_group, loc_group)

                base_reward, base_parts = _step_reward_and_parts(copy.deepcopy(driver), base_action_env)
                base_reward_raw = float(base_parts.get("reward_raw", base_reward))
                base_processed = float(base_parts.get("processed_ratio_eval", 0.0))
                base_drop = float(base_parts.get("drop_ratio_eval", 0.0))
                base_backlog = float(base_parts.get("pre_backlog_steps_eval", 0.0))

                for u in range(int(cfg.num_uav)):
                    if sampled >= int(max_agent_samples):
                        break
                    valid = np.asarray(valid_mask_group[u], dtype=bool)
                    valid_count = int(np.sum(valid))
                    if valid_count <= 1:
                        continue
                    base_action_u = np.asarray(base_action_env[u], dtype=np.float32)
                    loc_valid = np.asarray(effective_loc_group[u, valid], dtype=np.float64)
                    action_valid = np.asarray(base_action_u[valid], dtype=np.float64)
                    reward_credit: list[float] = []
                    reward_raw_credit: list[float] = []
                    processed_gain: list[float] = []
                    drop_gain: list[float] = []
                    backlog_gain: list[float] = []
                    deltas: list[float] = []
                    target_slots = np.flatnonzero(valid)

                    for target_slot in target_slots.tolist():
                        cf_action_u, used_delta = _reallocate_toward_slot(
                            base_action_u,
                            valid,
                            int(target_slot),
                            float(cf_delta),
                        )
                        if cf_action_u is None or used_delta <= 0.0:
                            continue
                        cf_action_env = np.asarray(base_action_env, dtype=np.float32).copy()
                        cf_action_env[u] = cf_action_u
                        cf_reward, cf_parts = _step_reward_and_parts(copy.deepcopy(driver), cf_action_env)
                        reward_credit.append(float(cf_reward - base_reward))
                        reward_raw_credit.append(float(cf_parts.get("reward_raw", cf_reward) - base_reward_raw))
                        processed_gain.append(float(cf_parts.get("processed_ratio_eval", 0.0) - base_processed))
                        drop_gain.append(float(base_drop - cf_parts.get("drop_ratio_eval", 0.0)))
                        backlog_gain.append(float(base_backlog - cf_parts.get("pre_backlog_steps_eval", 0.0)))
                        deltas.append(float(used_delta))
                        counterfactual_evals += 1

                    if len(reward_credit) <= 1:
                        continue

                    reward_credit_arr = np.asarray(reward_credit, dtype=np.float64)
                    reward_raw_credit_arr = np.asarray(reward_raw_credit, dtype=np.float64)
                    processed_gain_arr = np.asarray(processed_gain, dtype=np.float64)
                    drop_gain_arr = np.asarray(drop_gain, dtype=np.float64)
                    backlog_gain_arr = np.asarray(backlog_gain, dtype=np.float64)

                    valid_user_count.append(valid_count)
                    cf_delta_used.append(_safe_mean(deltas))
                    best_reward_gain.append(float(np.max(reward_credit_arr)))
                    worst_reward_gain.append(float(np.min(reward_credit_arr)))
                    positive_best_reward_gain.append(float(np.max(reward_credit_arr) > 0.0))
                    positive_target_fraction.append(float(np.mean(reward_credit_arr > 0.0)))

                    loc_vs_reward_credit_spearman.append(_safe_spearman_desc(loc_valid, reward_credit_arr))
                    action_vs_reward_credit_spearman.append(_safe_spearman_desc(action_valid, reward_credit_arr))
                    loc_vs_reward_credit_top1_hit.append(
                        float(int(np.argmax(loc_valid)) == int(np.argmax(reward_credit_arr)))
                    )
                    action_vs_reward_credit_top1_hit.append(
                        float(int(np.argmax(action_valid)) == int(np.argmax(reward_credit_arr)))
                    )
                    loc_vs_reward_credit_pairwise_acc.append(
                        _pairwise_concordance_desc(loc_valid, reward_credit_arr)
                    )
                    action_vs_reward_credit_pairwise_acc.append(
                        _pairwise_concordance_desc(action_valid, reward_credit_arr)
                    )
                    loc_vs_reward_raw_credit_spearman.append(
                        _safe_spearman_desc(loc_valid, reward_raw_credit_arr)
                    )
                    loc_vs_processed_gain_spearman.append(
                        _safe_spearman_desc(loc_valid, processed_gain_arr)
                    )
                    loc_vs_drop_gain_spearman.append(
                        _safe_spearman_desc(loc_valid, drop_gain_arr)
                    )
                    loc_vs_backlog_gain_spearman.append(
                        _safe_spearman_desc(loc_valid, backlog_gain_arr)
                    )
                    sampled += 1

                live_step = driver.execute_stage_bw_and_step(base_action_env)
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
        "counterfactual_mode": "local_finite_difference",
        "sample_count": int(sampled),
        "counterfactual_eval_count": int(counterfactual_evals),
        "cf_delta_requested": float(cf_delta),
        "valid_user_count": _summarize([float(x) for x in valid_user_count]),
        "cf_delta_used": _summarize(cf_delta_used),
        "best_reward_gain": _summarize(best_reward_gain),
        "worst_reward_gain": _summarize(worst_reward_gain),
        "positive_best_reward_gain_rate": float(np.mean(np.asarray(positive_best_reward_gain, dtype=np.float64)))
        if positive_best_reward_gain
        else 0.0,
        "positive_target_fraction": _summarize(positive_target_fraction),
        "loc_vs_reward_credit_spearman": _summarize(loc_vs_reward_credit_spearman),
        "action_vs_reward_credit_spearman": _summarize(action_vs_reward_credit_spearman),
        "loc_vs_reward_credit_top1_hit": _summarize(loc_vs_reward_credit_top1_hit),
        "action_vs_reward_credit_top1_hit": _summarize(action_vs_reward_credit_top1_hit),
        "loc_vs_reward_credit_pairwise_acc": _summarize(loc_vs_reward_credit_pairwise_acc),
        "action_vs_reward_credit_pairwise_acc": _summarize(action_vs_reward_credit_pairwise_acc),
        "loc_vs_reward_raw_credit_spearman": _summarize(loc_vs_reward_raw_credit_spearman),
        "loc_vs_processed_gain_spearman": _summarize(loc_vs_processed_gain_spearman),
        "loc_vs_drop_gain_spearman": _summarize(loc_vs_drop_gain_spearman),
        "loc_vs_backlog_gain_spearman": _summarize(loc_vs_backlog_gain_spearman),
        "spearman_by_valid_user_count": _bucket_mean(valid_user_count, loc_vs_reward_credit_spearman),
        "top1_hit_by_valid_user_count": _bucket_mean(valid_user_count, loc_vs_reward_credit_top1_hit),
        "pairwise_acc_by_valid_user_count": _bucket_mean(valid_user_count, loc_vs_reward_credit_pairwise_acc),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--updates", type=int, nargs="+", default=[100, 200])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-seed-base", type=int, default=82000)
    parser.add_argument("--policy-mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--cf-delta", type=float, default=0.05)
    parser.add_argument("--max-agent-samples", type=int, default=128)
    parser.add_argument("--out-name", type=str, default="structured_bw_counterfactual_credit_probe.json")
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
                    "cf_eval_count": update_summary["counterfactual_eval_count"],
                    "loc_reward_spearman_mean": update_summary["loc_vs_reward_credit_spearman"]["mean"],
                    "action_reward_spearman_mean": update_summary["action_vs_reward_credit_spearman"]["mean"],
                    "loc_top1_hit_mean": update_summary["loc_vs_reward_credit_top1_hit"]["mean"],
                    "positive_best_gain_rate": update_summary["positive_best_reward_gain_rate"],
                }
            )
        )

    out_path = run_dir / str(args.out_name)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
