from __future__ import annotations

import argparse
import json
import os
import sys

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
from sagin_marl.rl.structured_mappo import _to_device_dataclass
from sagin_marl.rl.structured_stage_builders import build_local_bw_states_from_snapshot
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def _resolve_torch_device(device_arg: str) -> torch.device:
    requested = str(device_arg).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return torch.device(device_arg)


def _snapshot_to_local_bw_state(snapshot, device: torch.device):
    local_state = build_local_bw_states_from_snapshot(snapshot)[0]
    return _to_device_dataclass(local_state, device)


def _collect_bw_snapshots(cfg, actor, *, device: torch.device, num_states: int, seed: int) -> list[dict]:
    env = make_structured_env(cfg, mode="script")
    driver = as_structured_driver(env)
    snapshots: list[dict] = []
    try:
        env.reset(seed=int(seed))
        episode_idx = 0
        while len(snapshots) < int(num_states):
            driver.begin_step()
            accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
            _ = driver.run_accel_stage(accel_zero)
            select_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 1)
            bw_world = driver.run_sat_stage(np.full((cfg.num_uav, select_k), -1, dtype=np.int64))
            bw_snapshot = driver.build_bw_stage_snapshot(bw_world)
            snapshot_state = driver.export_bw_stage_state()
            local_state = _snapshot_to_local_bw_state(bw_snapshot, device)
            snapshots.append({"snapshot_state": snapshot_state, "local_state": local_state})
            with torch.no_grad():
                out = actor.act_bw(local_state, deterministic=True)
            step_result, _ = driver.execute_stage_bw_and_prepare_next_accel(
                np.asarray(out.action.detach().cpu().numpy(), dtype=np.float32)
            )
            done = bool(list(step_result.terminations.values())[0] or list(step_result.truncations.values())[0])
            if done and len(snapshots) < int(num_states):
                episode_idx += 1
                env.reset(seed=int(seed) + episode_idx)
                driver = as_structured_driver(env)
        return snapshots[: int(num_states)]
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def _rollout_with_kstep_good_splice(
    cfg,
    *,
    snapshot_state: dict,
    bad_actor,
    good_actor,
    device: torch.device,
    good_steps: int,
) -> float:
    env = make_structured_env(cfg, mode="script")
    driver = as_structured_driver(env)
    total_reward = 0.0
    bw_step_idx = 0
    try:
        driver.load_bw_stage_state(snapshot_state)
        while True:
            bw_snapshot = driver.build_bw_stage_snapshot()
            local_state = _to_device_dataclass(build_local_bw_states_from_snapshot(bw_snapshot)[0], device)
            actor = good_actor if bw_step_idx < int(good_steps) else bad_actor
            with torch.no_grad():
                out = actor.act_bw(local_state, deterministic=True)
            action = np.asarray(out.action.detach().cpu().numpy(), dtype=np.float32)
            step_result, _ = driver.execute_stage_bw_and_prepare_next_accel(action)
            total_reward += float(next(iter(step_result.rewards.values())))
            bw_step_idx += 1
            done = bool(list(step_result.terminations.values())[0] or list(step_result.truncations.values())[0])
            if done:
                break
            driver.begin_step()
            accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
            _ = driver.run_accel_stage(accel_zero)
            select_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 1)
            _ = driver.run_sat_stage(np.full((cfg.num_uav, select_k), -1, dtype=np.int64))
        return float(total_reward)
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--bad_actor", type=str, required=True)
    parser.add_argument("--good_actor", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_states", type=int, default=4)
    parser.add_argument("--k_values", type=str, default="0,1,2,5,10,20,100")
    args = parser.parse_args()

    device = _resolve_torch_device(args.device)
    cfg = load_config(str(args.config))
    k_values = [int(v) for v in str(args.k_values).split(",") if str(v).strip()]
    os.makedirs(args.run_dir, exist_ok=True)

    bundle_bad = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    bad_actor = bundle_bad.actor.to(device)
    load_checkpoint_forgiving(bad_actor, str(args.bad_actor), map_location=device, strict=True)
    bad_actor.eval()

    bundle_good = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    good_actor = bundle_good.actor.to(device)
    load_checkpoint_forgiving(good_actor, str(args.good_actor), map_location=device, strict=True)
    good_actor.eval()

    snapshots = _collect_bw_snapshots(cfg, bad_actor, device=device, num_states=int(args.num_states), seed=int(args.seed))

    rows = []
    for idx, snap in enumerate(snapshots):
        k_scores = {}
        for k in k_values:
            score = _rollout_with_kstep_good_splice(
                cfg,
                snapshot_state=snap["snapshot_state"],
                bad_actor=bad_actor,
                good_actor=good_actor,
                device=device,
                good_steps=int(k),
            )
            k_scores[int(k)] = float(score)
        rows.append({"state_index": idx, "scores": k_scores})
        pretty = " | ".join(f"K={k}:{k_scores[k]:.3f}" for k in k_values)
        print(f"State {idx:02d} | {pretty}")

    summary = {
        "num_states": len(rows),
        "k_values": k_values,
        "score_means": {
            str(k): float(np.mean([row["scores"][k] for row in rows])) for k in k_values
        },
        "gain_over_k0_means": {
            str(k): float(np.mean([row["scores"][k] - row["scores"][0] for row in rows])) for k in k_values
        },
        "rows": rows,
    }
    with open(os.path.join(args.run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
