from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(ROOT, "sagin_marl")):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        raise RuntimeError("Could not locate repository root.")
    ROOT = parent
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import audit_bw_broad2local_offline as audit
from sagin_marl.rl.structured_parallel_eval import (
    batched_policy_accel_actions,
    batched_policy_sat_pair_indices,
    current_obs_many,
    looks_like_driver_group,
    refresh_stage_obs_cache_many,
    reset_many,
)
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_env_group


def _collect_one_snapshot(
    *,
    cfg,
    actor,
    device: torch.device,
    backend: str,
    num_envs: int,
    seed: int,
    heuristic_bw_source: str,
    panel_random_count: int,
) -> dict | None:
    env_group = make_structured_env_group(cfg, num_envs=int(num_envs), backend=str(backend))
    drivers = env_group if looks_like_driver_group(env_group) else audit._as_driver_list(env_group)
    rng = np.random.default_rng(int(seed))
    try:
        reset_many(drivers, [int(seed) + 1000 + idx for idx in range(int(num_envs))])
        attempts = max(int(num_envs) * 2, 4)
        for _ in range(attempts):
            indices = list(range(int(num_envs)))
            if looks_like_driver_group(drivers):
                accel_world_states = drivers.prepare_accel_stage_many(indices=indices)
            else:
                accel_world_states = [drivers[idx].begin_step() for idx in indices]
            accel_actions = batched_policy_accel_actions(actor, accel_world_states, device, True)
            if looks_like_driver_group(drivers):
                sat_snapshots = drivers.run_accel_and_prepare_sat_many(accel_actions, indices=indices)
            else:
                sat_world_states = [drivers[idx].run_accel_stage(action) for idx, action in zip(indices, accel_actions)]
                sat_snapshots = [drivers[idx].build_sat_stage_snapshot(world_state) for idx, world_state in zip(indices, sat_world_states)]
            refresh_stage_obs_cache_many(drivers, indices=indices)
            obs_after_accel_groups = current_obs_many(drivers, indices=indices)
            sat_pair_indices = batched_policy_sat_pair_indices(actor, sat_snapshots, device, True)
            if looks_like_driver_group(drivers):
                bw_snapshots = drivers.run_sat_and_prepare_bw_many(sat_pair_indices, indices=indices)
                snapshot_states = drivers.export_bw_stage_state_many(indices=indices)
            else:
                sat_actions = [drivers[idx].decode_sat_pair_actions([], pair_idx) for idx, pair_idx in zip(indices, sat_pair_indices)]
                bw_world_states = [drivers[idx].run_sat_stage(action) for idx, action in zip(indices, sat_actions)]
                bw_snapshots = [drivers[idx].build_bw_stage_snapshot(world_state) for idx, world_state in zip(indices, bw_world_states)]
                snapshot_states = [drivers[idx].export_bw_stage_state() for idx in indices]
            for local_idx in range(len(indices)):
                snapshot = bw_snapshots[local_idx]
                valid_mask = np.asarray(snapshot.bw_valid_mask, dtype=bool)
                valid_counts = valid_mask.sum(axis=1).astype(int).tolist()
                if np.all(valid_mask.sum(axis=1) <= 1):
                    continue
                panel_actions = audit._make_policy_panel_actions(
                    cfg,
                    snapshot,
                    obs_after_accel_groups[local_idx],
                    heuristic_bw_source=str(heuristic_bw_source),
                    random_count=int(panel_random_count),
                    rng=rng,
                )
                return {
                    "snapshot_state": snapshot_states[local_idx],
                    "valid_counts": valid_counts,
                    "panel_names": list(panel_actions.keys()),
                    "panel_actions": [np.asarray(panel_actions[name], dtype=np.float32) for name in panel_actions.keys()],
                    "source_backend": str(backend),
                }
        return None
    finally:
        close_structured_env_group(env_group)


def _score_snapshot(
    *,
    cfg,
    actor,
    device: torch.device,
    snapshot_state: dict,
    panel_actions: list[np.ndarray],
    backend: str,
    num_envs: int,
    bw_deterministic_readout: str,
    bw_deterministic_opt_steps: int,
    bw_deterministic_step_size: float,
) -> list[float]:
    eval_group = make_structured_env_group(cfg, num_envs=int(num_envs), backend=str(backend))
    eval_drivers = eval_group if looks_like_driver_group(eval_group) else audit._as_driver_list(eval_group)
    try:
        return audit._rollout_panel_action_scores(
            eval_drivers,
            snapshot_state,
            panel_actions=panel_actions,
            follow_actor=actor,
            follow_device=device,
            follow_deterministic=True,
            bw_deterministic_readout=str(bw_deterministic_readout),
            bw_deterministic_opt_steps=int(bw_deterministic_opt_steps),
            bw_deterministic_step_size=float(bw_deterministic_step_size),
            gamma=float(cfg.gamma),
            k_steps=2,
        )
    finally:
        close_structured_env_group(eval_group)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug one BW panel snapshot under sync vs subproc replay.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--actor_checkpoint", type=str, default=None)
    parser.add_argument("--collect_backend", choices=["sync", "subproc"], default="sync")
    parser.add_argument("--score_backends", nargs="+", default=["sync", "subproc"])
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--heuristic_bw_source", type=str, default="queue_aware")
    parser.add_argument("--panel_random_count", type=int, default=2)
    parser.add_argument(
        "--bw_deterministic_readout",
        choices=["latent_mean_pushforward", "simplex_argmax_logprob"],
        default="latent_mean_pushforward",
    )
    parser.add_argument("--bw_deterministic_opt_steps", type=int, default=8)
    parser.add_argument("--bw_deterministic_step_size", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device(str(args.device))
    run_dir = Path(args.run_dir)
    cfg, actor = audit._load_actor(
        run_dir,
        int(args.update),
        device,
        actor_checkpoint=None if args.actor_checkpoint is None else str(args.actor_checkpoint),
    )
    snapshot_info = _collect_one_snapshot(
        cfg=cfg,
        actor=actor,
        device=device,
        backend=str(args.collect_backend),
        num_envs=int(args.num_envs),
        seed=int(args.seed),
        heuristic_bw_source=str(args.heuristic_bw_source),
        panel_random_count=int(args.panel_random_count),
    )
    if snapshot_info is None:
        print(
            {
                "status": "no_snapshot",
                "collect_backend": str(args.collect_backend),
                "num_envs": int(args.num_envs),
            }
        )
        return
    payload = {
        "status": "ok",
        "collect_backend": str(args.collect_backend),
        "valid_counts": snapshot_info["valid_counts"],
        "panel_names": snapshot_info["panel_names"],
    }
    for backend in args.score_backends:
        scores = _score_snapshot(
            cfg=cfg,
            actor=actor,
            device=device,
            snapshot_state=snapshot_info["snapshot_state"],
            panel_actions=snapshot_info["panel_actions"],
            backend=str(backend),
            num_envs=max(int(args.num_envs), len(snapshot_info["panel_actions"])),
            bw_deterministic_readout=str(args.bw_deterministic_readout),
            bw_deterministic_opt_steps=int(args.bw_deterministic_opt_steps),
            bw_deterministic_step_size=float(args.bw_deterministic_step_size),
        )
        arr = np.asarray(scores, dtype=np.float64)
        payload[f"{backend}_scores"] = [float(round(x, 6)) for x in arr.tolist()]
        payload[f"{backend}_gap_best_mean"] = float(round(float(np.max(arr) - np.mean(arr)), 6))
        payload[f"{backend}_gap_best_worst"] = float(round(float(np.max(arr) - np.min(arr)), 6))
    print(payload)


if __name__ == "__main__":
    main()
