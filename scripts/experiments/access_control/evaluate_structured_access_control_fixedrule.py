from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.experiments.access_control.access_control_imitation_common import (
    get_access_rule,
    predict_access_assoc_fixedrule_from_snapshot,
    queue_aware_bw_policy_from_snapshot,
)
from scripts.experiments.access_control.evaluate_structured_access_control_oracle import (
    _apply_access_stage_override,
    _episode_metrics_template,
    _episode_row_from_accumulator,
    _load_actor,
    _new_episode_accumulator,
    _resolve_checkpoint,
    _write_csv,
)
from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_parallel_eval import (
    batched_policy_accel_actions,
    batched_policy_sat_pair_indices,
    looks_like_driver_group,
    reset_at,
    reset_many,
)
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver_group


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _evaluate(
    *,
    cfg,
    actor,
    device: torch.device,
    episodes: int,
    episode_seed_base: int | None,
    num_envs: int,
    vec_backend: str,
    access_rule: str,
    overlap_per_uav_keep: int,
    overlap_per_user_max_uav: int,
):
    active_slots = max(min(int(num_envs), int(episodes)), 1)
    drivers = make_structured_driver_group(cfg, num_envs=active_slots, backend=str(vec_backend))
    rows: list[dict[str, float]] = []
    slot_episode = list(range(active_slots))
    slot_active = [True for _ in range(active_slots)]
    slot_acc = [_new_episode_accumulator() for _ in range(active_slots)]
    next_episode = active_slots
    initial_seeds = [None if episode_seed_base is None else int(episode_seed_base) + slot for slot in range(active_slots)]
    reset_many(drivers, initial_seeds)
    try:
        while len(rows) < int(episodes):
            active_indices = [slot for slot, is_active in enumerate(slot_active) if is_active]
            if not active_indices:
                break
            if looks_like_driver_group(drivers):
                accel_world_states = drivers.begin_step_many(indices=active_indices)
                accel_actions = batched_policy_accel_actions(actor, accel_world_states, device, True)
                sat_snapshots = drivers.run_accel_and_prepare_sat_many(accel_actions, indices=active_indices)
                sat_pair_indices = batched_policy_sat_pair_indices(actor, sat_snapshots, device, True)
                _ = drivers.run_sat_and_prepare_bw_many(sat_pair_indices, indices=active_indices)
                snapshot_states = drivers.export_bw_stage_state_many(indices=active_indices)
                assocs_pred: list[np.ndarray] = []
                candidates_pred_many: list[list[list[int]]] = []
                bw_actions: list[np.ndarray] = []
                for snapshot_state in snapshot_states:
                    assoc_pred, candidates_pred = predict_access_assoc_fixedrule_from_snapshot(
                        snapshot_state,
                        cfg,
                        rule_name=str(access_rule),
                        overlap_per_uav_keep=int(overlap_per_uav_keep),
                        overlap_per_user_max_uav=int(overlap_per_user_max_uav),
                    )
                    bw_action = queue_aware_bw_policy_from_snapshot(
                        snapshot_state,
                        cfg,
                        assoc=assoc_pred,
                        candidates=candidates_pred,
                    )
                    assocs_pred.append(assoc_pred)
                    candidates_pred_many.append(candidates_pred)
                    bw_actions.append(np.asarray(bw_action, dtype=np.float32))
                drivers.load_bw_stage_state_many(snapshot_states, indices=active_indices)
                drivers.override_access_stage_state_many(assocs_pred, candidates_pred_many, indices=active_indices)
                step_results = drivers.execute_stage_bw_and_step_many(bw_actions, indices=active_indices)
                reward_parts_many = drivers.last_reward_parts_many(indices=active_indices)
                for slot, step_result, reward_parts in zip(active_indices, step_results, reward_parts_many):
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
                        reset_at(drivers, slot, seed_next)
                        slot_episode[slot] = int(next_episode)
                        next_episode += 1
                    else:
                        slot_active[slot] = False
                        slot_episode[slot] = -1
            else:
                accel_world_states = [drivers[slot].begin_step() for slot in active_indices]
                accel_actions = batched_policy_accel_actions(actor, accel_world_states, device, True)
                sat_world_states = [drivers[slot].run_accel_stage(action) for slot, action in zip(active_indices, accel_actions)]
                sat_snapshots = [drivers[slot].build_sat_stage_snapshot(world_state) for slot, world_state in zip(active_indices, sat_world_states)]
                sat_pair_indices = batched_policy_sat_pair_indices(actor, sat_snapshots, device, True)
                sat_actions = [drivers[slot].decode_sat_pair_actions([], pair_indices) for slot, pair_indices in zip(active_indices, sat_pair_indices)]
                _ = [drivers[slot].run_sat_stage(action) for slot, action in zip(active_indices, sat_actions)]
                snapshot_states = [drivers[slot].export_bw_stage_state() for slot in active_indices]
                for local_slot, slot in enumerate(active_indices):
                    assoc_pred, candidates_pred = predict_access_assoc_fixedrule_from_snapshot(
                        snapshot_states[local_slot],
                        cfg,
                        rule_name=str(access_rule),
                        overlap_per_uav_keep=int(overlap_per_uav_keep),
                        overlap_per_user_max_uav=int(overlap_per_user_max_uav),
                    )
                    bw_action = queue_aware_bw_policy_from_snapshot(
                        snapshot_states[local_slot],
                        cfg,
                        assoc=assoc_pred,
                        candidates=candidates_pred,
                    )
                    drivers[slot].load_bw_stage_state(snapshot_states[local_slot])
                    _apply_access_stage_override(drivers[slot], assoc=assoc_pred, candidates=candidates_pred)
                    step_result = drivers[slot].execute_stage_bw_and_step(np.asarray(bw_action, dtype=np.float32))
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
        return summary, rows
    finally:
        close_structured_env_group(drivers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--update", type=int, default=None)
    parser.add_argument("--access_rule", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--episode_seed_base", type=int, default=65000)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--vec_backend", choices=["sync", "subproc"], default="sync")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overlap_per_uav_keep", type=int, default=6)
    parser.add_argument("--overlap_per_user_max_uav", type=int, default=2)
    parser.add_argument("--out_dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_all_seeds(int(args.seed))
    _ = get_access_rule(str(args.access_rule))
    run_dir = Path(args.run_dir)
    cfg_path = args.config or str(run_dir / "config_source.yaml")
    cfg = load_config(cfg_path)
    checkpoint = _resolve_checkpoint(run_dir, args.checkpoint, args.update)
    device = torch.device(args.device)
    actor = _load_actor(
        cfg,
        checkpoint,
        hidden_dim=int(args.hidden_dim),
        embed_dim=int(args.embed_dim),
        device=device,
    )
    summary, rows = _evaluate(
        cfg=cfg,
        actor=actor,
        device=device,
        episodes=int(args.episodes),
        episode_seed_base=args.episode_seed_base,
        num_envs=int(args.num_envs),
        vec_backend=str(args.vec_backend),
        access_rule=str(args.access_rule),
        overlap_per_uav_keep=int(args.overlap_per_uav_keep),
        overlap_per_user_max_uav=int(args.overlap_per_user_max_uav),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "per_episode.csv", rows)
    payload = {
        "run_dir": str(run_dir),
        "config": str(cfg_path),
        "checkpoint": str(checkpoint),
        "access_rule": str(args.access_rule),
        "episodes": int(args.episodes),
        "episode_seed_base": int(args.episode_seed_base) if args.episode_seed_base is not None else None,
        "num_envs": int(args.num_envs),
        "vec_backend": str(args.vec_backend),
        "summary": summary,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(
        f"Summary: rule={args.access_rule} reward={summary['reward_sum']:.4f} "
        f"processed={summary['processed_ratio_eval']:.4f} "
        f"drop={summary['drop_ratio_eval']:.4f} "
        f"pre_backlog={summary['pre_backlog_steps_eval']:.4f}"
    )


if __name__ == "__main__":
    main()
