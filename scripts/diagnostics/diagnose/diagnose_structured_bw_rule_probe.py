from __future__ import annotations

import argparse
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
from sagin_marl.rl.structured_parallel_eval import (
    batched_policy_accel_actions,
    batched_policy_bw_outputs,
    batched_policy_sat_pair_indices,
    looks_like_driver_group,
    reset_at,
    reset_many,
)
from sagin_marl.rl.structured_mappo import _collate_dataclass
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_env_group
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


RULE_ORDER = [
    "uniform",
    "queue",
    "eta_ref",
    "queue_eta",
    "prev_only",
    "queue_eta_prev",
]


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _rankdata_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _safe_corr(x: list[float], y: list[float]) -> float | None:
    if len(x) <= 1 or len(y) <= 1 or len(x) != len(y):
        return None
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if float(np.std(xa)) <= 1.0e-12 or float(np.std(ya)) <= 1.0e-12:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def _safe_spearman_desc(pred: np.ndarray, truth: np.ndarray) -> float:
    if pred.size <= 1 or truth.size <= 1 or pred.size != truth.size:
        return 0.0
    return float(_safe_corr(_rankdata_desc(pred).tolist(), _rankdata_desc(truth).tolist()) or 0.0)


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


def _normalize_weights(weights: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(weights, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if not np.any(valid):
        return out
    clipped = np.clip(np.asarray(weights, dtype=np.float32), 0.0, None) * valid.astype(np.float32, copy=False)
    denom = float(np.sum(clipped))
    if denom > 1.0e-9:
        out = clipped / denom
    else:
        out[valid] = 1.0 / float(np.sum(valid))
    return out.astype(np.float32, copy=False)


def _l1_align(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - 0.5 * np.sum(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))))


def _load_actor(run_dir: Path, update: int, device: torch.device):
    cfg = load_config(str(run_dir / "config_source.yaml"))
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    actor_ckpt = run_dir / f"actor_u{int(update):04d}.pt"
    load_checkpoint_forgiving(bundle.actor, str(actor_ckpt), map_location=device, strict=True)
    bundle.actor.to(device).eval()
    return cfg, bundle.actor


def _empty_method_stats() -> dict[str, dict[str, list[float]]]:
    return {
        rule: {"align": [], "spearman": [], "top1_hit": []}
        for rule in RULE_ORDER
    }


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _rule_payloads_from_bw_state(local_state: Any, assoc_bonus: float) -> list[dict[str, Any]]:
    user_nodes = _to_numpy(local_state.user_nodes)
    user_edges = _to_numpy(local_state.user_edges)
    user_mask = _to_numpy(local_state.user_mask) > 0.5
    bw_valid_mask = (_to_numpy(local_state.bw_valid_mask) > 0.5) & user_mask

    rows: list[dict[str, Any]] = []
    for row in range(int(user_nodes.shape[0])):
        valid = np.asarray(bw_valid_mask[row], dtype=bool)
        q = np.zeros((user_nodes.shape[1],), dtype=np.float32)
        eta_ref = np.zeros_like(q)
        prev = np.zeros_like(q)
        if np.any(valid):
            q[valid] = np.clip(np.asarray(user_nodes[row, valid, 2], dtype=np.float32), 0.0, None)
            eta_ref[valid] = np.clip(np.asarray(user_edges[row, valid, 7], dtype=np.float32), 0.0, None)
            prev[valid] = np.clip(np.asarray(user_edges[row, valid, 6], dtype=np.float32), 0.0, 1.0)
        q_eta = q * (0.5 + eta_ref)
        q_eta_prev = q_eta * (1.0 + float(assoc_bonus) * prev)
        targets = {
            "uniform": _normalize_weights(np.ones_like(q, dtype=np.float32), valid),
            "queue": _normalize_weights(q, valid),
            "eta_ref": _normalize_weights(eta_ref, valid),
            "queue_eta": _normalize_weights(q_eta, valid),
            "prev_only": _normalize_weights(prev, valid),
            "queue_eta_prev": _normalize_weights(q_eta_prev, valid),
        }
        rows.append({"valid_mask": valid, "targets": targets})
    return rows


def _record_rule_metrics(
    method_stats: dict[str, dict[str, list[float]]],
    action: np.ndarray,
    targets: dict[str, np.ndarray],
    valid_mask: np.ndarray,
) -> None:
    valid = np.asarray(valid_mask, dtype=bool)
    alloc = _normalize_weights(np.asarray(action, dtype=np.float32), valid)
    alloc_valid = alloc[valid]
    for rule in RULE_ORDER:
        target = np.asarray(targets[rule], dtype=np.float32)
        target_valid = target[valid]
        method_stats[rule]["align"].append(_l1_align(alloc, target))
        method_stats[rule]["spearman"].append(_safe_spearman_desc(alloc_valid, target_valid))
        top1_hit = 0.0
        if alloc_valid.size > 0 and target_valid.size > 0:
            top1_hit = 1.0 if int(np.argmax(alloc_valid)) == int(np.argmax(target_valid)) else 0.0
        method_stats[rule]["top1_hit"].append(float(top1_hit))


def _summarize_method_stats(
    counts: list[int],
    method_stats: dict[str, dict[str, list[float]]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rule in RULE_ORDER:
        align_vals = method_stats[rule]["align"]
        spearman_vals = method_stats[rule]["spearman"]
        top1_vals = method_stats[rule]["top1_hit"]
        out[rule] = {
            "align": _summarize(align_vals),
            "spearman": _summarize(spearman_vals),
            "top1_hit": _summarize(top1_vals),
            "align_by_valid_user_count": _bucket_mean(counts, align_vals),
            "spearman_by_valid_user_count": _bucket_mean(counts, spearman_vals),
            "top1_hit_by_valid_user_count": _bucket_mean(counts, top1_vals),
        }
    return out


def _best_rule_by_align(method_summary: dict[str, Any]) -> dict[str, float]:
    rows = {rule: float(method_summary[rule]["align"]["mean"]) for rule in RULE_ORDER}
    best_rule = max(rows, key=rows.get)
    return {"rule": best_rule, "align_mean": rows[best_rule]}


def diagnose_update(
    run_dir: Path,
    update: int,
    *,
    episodes: int,
    episode_seed_base: int,
    deterministic: bool,
    device: torch.device,
    num_envs: int,
    vec_backend: str,
) -> dict[str, Any]:
    cfg, actor = _load_actor(run_dir, int(update), device)
    assoc_bonus = float(getattr(cfg, "baseline_assoc_bonus", 0.3))

    valid_user_count: list[int] = []
    policy_stats = _empty_method_stats()
    heuristic_stats = _empty_method_stats()
    active_slots = max(min(int(num_envs), int(episodes)), 1)
    if active_slots <= 1:
        env = make_structured_env(cfg, mode="script")
        try:
            for ep in range(int(episodes)):
                env.reset(seed=int(episode_seed_base) + ep)
                driver = as_structured_driver(env)
                done = False
                while not done:
                    z0 = driver.begin_step()
                    accel_states = driver.build_local_accel_states(z0)
                    accel_batch = _collate_dataclass(accel_states, device)
                    with torch.inference_mode():
                        accel_out = actor.act_accel(accel_batch, deterministic=deterministic)
                    z1 = driver.run_accel_stage(accel_out.action.detach().cpu().numpy())

                    sat_states = driver.build_sat_pair_candidates(z1)
                    sat_batch = _collate_dataclass(sat_states, device)
                    with torch.inference_mode():
                        sat_out = actor.act_sat_pair(sat_batch, deterministic=deterministic)
                    sat_action = driver.decode_sat_pair_actions(sat_states, sat_out.subset_index.detach().cpu().tolist())
                    z2 = driver.run_sat_stage(sat_action)

                    bw_states = driver.build_bw_valid_context(z2)
                    bw_batch = _collate_dataclass(bw_states, device)
                    with torch.inference_mode():
                        bw_out = actor.act_bw(bw_batch, deterministic=deterministic)
                    policy_bw = bw_out.action.detach().cpu().numpy()
                    rule_payloads = _rule_payloads_from_bw_state(bw_batch, assoc_bonus)

                    for u in range(int(cfg.num_uav)):
                        payload = rule_payloads[u]
                        valid_mask = np.asarray(payload["valid_mask"], dtype=bool)
                        valid_count = int(np.sum(valid_mask))
                        if valid_count <= 0:
                            continue
                        valid_user_count.append(valid_count)
                        targets = payload["targets"]
                        _record_rule_metrics(policy_stats, policy_bw[u], targets, valid_mask)
                        _record_rule_metrics(heuristic_stats, targets["queue_eta_prev"], targets, valid_mask)

                    step = driver.execute_stage_bw_and_step(policy_bw)
                    done = bool(any(step.terminations.values()) or any(step.truncations.values()))
        finally:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()
        policy_summary = _summarize_method_stats(valid_user_count, policy_stats)
        heuristic_summary = _summarize_method_stats(valid_user_count, heuristic_stats)
        return {
            "update": int(update),
            "episodes": int(episodes),
            "episode_seed_base": int(episode_seed_base),
            "deterministic": bool(deterministic),
            "num_envs": int(active_slots),
            "vec_backend": "sync",
            "sample_count": int(len(valid_user_count)),
            "valid_user_count": _summarize([float(x) for x in valid_user_count]),
            "policy": {
                "best_rule_by_align": _best_rule_by_align(policy_summary),
                "rules": policy_summary,
            },
            "heuristic": {
                "best_rule_by_align": _best_rule_by_align(heuristic_summary),
                "rules": heuristic_summary,
            },
        }

    env_group = make_structured_env_group(cfg, num_envs=active_slots, backend=vec_backend)
    drivers = env_group if looks_like_driver_group(env_group) else as_structured_drivers(env_group)
    slot_active = [True for _ in range(active_slots)]
    next_episode = active_slots
    initial_seeds = [int(episode_seed_base) + slot for slot in range(active_slots)]
    reset_many(drivers, initial_seeds)
    try:
        while any(slot_active):
            active_indices = [slot for slot, is_active in enumerate(slot_active) if is_active]
            if not active_indices:
                break
            if looks_like_driver_group(drivers):
                accel_world_states = drivers.prepare_accel_stage_many(indices=active_indices)
            else:
                accel_world_states = [drivers[slot].begin_step() for slot in active_indices]
            accel_actions = batched_policy_accel_actions(actor, accel_world_states, device, deterministic)

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

            sat_pair_indices = batched_policy_sat_pair_indices(actor, sat_snapshots, device, deterministic)
            if looks_like_driver_group(drivers):
                bw_snapshots = drivers.run_sat_and_prepare_bw_many(sat_pair_indices, indices=active_indices)
            else:
                sat_actions = [
                    drivers[slot].decode_sat_pair_actions((), pair_indices)
                    for slot, pair_indices in zip(active_indices, sat_pair_indices)
                ]
                bw_world_states = [drivers[slot].run_sat_stage(action) for slot, action in zip(active_indices, sat_actions)]
                bw_snapshots = [
                    drivers[slot].build_bw_stage_snapshot(world_state)
                    for slot, world_state in zip(active_indices, bw_world_states)
                ]

            bw_eval = batched_policy_bw_outputs(actor, bw_snapshots, device, deterministic)
            policy_bw = bw_eval.actions
            rule_payloads = _rule_payloads_from_bw_state(bw_eval.local_state, assoc_bonus)

            for local_slot, slot in enumerate(active_indices):
                for u in range(int(cfg.num_uav)):
                    row_index = local_slot * int(cfg.num_uav) + u
                    payload = rule_payloads[row_index]
                    valid_mask = np.asarray(payload["valid_mask"], dtype=bool)
                    valid_count = int(np.sum(valid_mask))
                    if valid_count <= 0:
                        continue
                    valid_user_count.append(valid_count)
                    targets = payload["targets"]
                    _record_rule_metrics(policy_stats, policy_bw[local_slot][u], targets, valid_mask)
                    _record_rule_metrics(heuristic_stats, targets["queue_eta_prev"], targets, valid_mask)

            if looks_like_driver_group(drivers):
                step_results = drivers.execute_stage_bw_and_step_many(policy_bw, indices=active_indices)
            else:
                step_results = [
                    drivers[slot].execute_stage_bw_and_step(action)
                    for slot, action in zip(active_indices, policy_bw)
                ]

            for local_slot, slot in enumerate(active_indices):
                done = bool(any(step_results[local_slot].terminations.values()) or any(step_results[local_slot].truncations.values()))
                if not done:
                    continue
                if next_episode < int(episodes):
                    reset_at(drivers, slot, int(episode_seed_base) + next_episode)
                    next_episode += 1
                else:
                    slot_active[slot] = False
    finally:
        close_structured_env_group(env_group)

    policy_summary = _summarize_method_stats(valid_user_count, policy_stats)
    heuristic_summary = _summarize_method_stats(valid_user_count, heuristic_stats)
    return {
        "update": int(update),
        "episodes": int(episodes),
        "episode_seed_base": int(episode_seed_base),
        "deterministic": bool(deterministic),
        "num_envs": int(active_slots),
        "vec_backend": str(vec_backend),
        "sample_count": int(len(valid_user_count)),
        "valid_user_count": _summarize([float(x) for x in valid_user_count]),
        "policy": {
            "best_rule_by_align": _best_rule_by_align(policy_summary),
            "rules": policy_summary,
        },
        "heuristic": {
            "best_rule_by_align": _best_rule_by_align(heuristic_summary),
            "rules": heuristic_summary,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--updates", type=int, nargs="+", default=[50, 100, 150, 200])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-seed-base", type=int, default=76000)
    parser.add_argument("--policy-mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--num_envs", type=int, default=20)
    parser.add_argument("--vec_backend", choices=["sync", "subproc"], default="sync")
    parser.add_argument("--out-name", type=str, default="structured_bw_rule_probe.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "episodes": int(args.episodes),
        "episode_seed_base": int(args.episode_seed_base),
        "policy_mode": str(args.policy_mode),
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
            vec_backend=str(args.vec_backend),
        )
        summary["updates"][f"u{int(update):04d}"] = update_summary
        print(
            json.dumps(
                {
                    "update": int(update),
                    "policy_best_rule": update_summary["policy"]["best_rule_by_align"]["rule"],
                    "policy_best_align": update_summary["policy"]["best_rule_by_align"]["align_mean"],
                    "heur_best_rule": update_summary["heuristic"]["best_rule_by_align"]["rule"],
                    "heur_best_align": update_summary["heuristic"]["best_rule_by_align"]["align_mean"],
                }
            )
        )

    out_path = run_dir / str(args.out_name)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
