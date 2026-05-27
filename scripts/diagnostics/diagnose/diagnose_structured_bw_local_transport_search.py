from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
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


def _run_bw_candidate_to_episode_end(
    cfg,
    actor,
    *,
    snapshot_state: dict,
    first_action: np.ndarray,
    device: torch.device,
) -> float:
    env = make_structured_env(cfg, mode="script")
    driver = as_structured_driver(env)
    total_reward = 0.0
    try:
        driver.load_bw_stage_state(snapshot_state)
        action = np.asarray(first_action, dtype=np.float32)
        while True:
            step_result, _ = driver.execute_stage_bw_and_prepare_next_accel(action)
            total_reward += float(next(iter(step_result.rewards.values())))
            done = bool(list(step_result.terminations.values())[0] or list(step_result.truncations.values())[0])
            if done:
                break
            driver.begin_step()
            accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
            _ = driver.run_accel_stage(accel_zero)
            select_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 1)
            bw_world = driver.run_sat_stage(np.full((cfg.num_uav, select_k), -1, dtype=np.int64))
            snapshot = driver.build_bw_stage_snapshot(bw_world)
            local_state = _to_device_dataclass(build_local_bw_states_from_snapshot(snapshot)[0], device)
            with torch.no_grad():
                out = actor.act_bw(local_state, deterministic=True)
            action = np.asarray(out.action.detach().cpu().numpy(), dtype=np.float32)
        return float(total_reward)
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def _masked_l1(a: np.ndarray, b: np.ndarray, valid_mask: np.ndarray) -> float:
    mask = np.asarray(valid_mask, dtype=bool)
    return float(np.abs(a - b)[mask].sum())


def _generate_transport_candidates(action: np.ndarray, valid_mask: np.ndarray, deltas: list[float]) -> list[np.ndarray]:
    base = np.asarray(action, dtype=np.float32).copy()
    mask = np.asarray(valid_mask, dtype=bool)
    candidates: list[np.ndarray] = []
    for u in range(base.shape[0]):
        valid = np.flatnonzero(mask[u])
        for i in valid:
            for j in valid:
                if int(i) == int(j):
                    continue
                for delta in deltas:
                    amount = min(float(delta), float(base[u, i]))
                    if amount <= 1.0e-6:
                        continue
                    cand = base.copy()
                    cand[u, i] -= amount
                    cand[u, j] += amount
                    candidates.append(cand)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--bad_actor", type=str, required=True)
    parser.add_argument("--good_actor", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_states", type=int, default=4)
    parser.add_argument("--deltas", type=str, default="0.05,0.1,0.2,0.3")
    args = parser.parse_args()

    device = _resolve_torch_device(args.device)
    cfg = load_config(str(args.config))
    deltas = [float(v) for v in str(args.deltas).split(",") if str(v).strip()]
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
        local_state = snap["local_state"]
        valid_mask = np.asarray(
            ((local_state.user_mask > 0.5) & (local_state.bw_valid_mask > 0.5)).detach().cpu().numpy(),
            dtype=bool,
        )
        with torch.no_grad():
            bad_det = np.asarray(bad_actor.act_bw(local_state, deterministic=True).action.detach().cpu().numpy(), dtype=np.float32)
            good_det = np.asarray(good_actor.act_bw(local_state, deterministic=True).action.detach().cpu().numpy(), dtype=np.float32)
        candidates = _generate_transport_candidates(bad_det, valid_mask, deltas)
        scores = [
            _run_bw_candidate_to_episode_end(cfg, bad_actor, snapshot_state=snap["snapshot_state"], first_action=action, device=device)
            for action in candidates
        ]
        best_idx = int(np.argmax(np.asarray(scores, dtype=np.float64)))
        best_action = np.asarray(candidates[best_idx], dtype=np.float32)
        best_score = float(scores[best_idx])
        bad_score = _run_bw_candidate_to_episode_end(cfg, bad_actor, snapshot_state=snap["snapshot_state"], first_action=bad_det, device=device)
        good_score = _run_bw_candidate_to_episode_end(cfg, good_actor, snapshot_state=snap["snapshot_state"], first_action=good_det, device=device)
        rows.append(
            {
                "state_index": idx,
                "num_candidates": len(candidates),
                "bad_score": bad_score,
                "transport_best_score": best_score,
                "good_score": good_score,
                "transport_gap": best_score - bad_score,
                "good_gap": good_score - bad_score,
                "transport_minus_good": best_score - good_score,
                "bad_to_good_l1": _masked_l1(bad_det, good_det, valid_mask),
                "bad_to_transport_best_l1": _masked_l1(bad_det, best_action, valid_mask),
                "transport_best_to_good_l1": _masked_l1(best_action, good_det, valid_mask),
            }
        )
        print(
            f"State {idx:02d} | bad={bad_score:.3f} | transport_best={best_score:.3f} | "
            f"good={good_score:.3f} | transport-good={best_score - good_score:.3f}"
        )

    summary = {
        "num_states": len(rows),
        "deltas": deltas,
        "transport_gap_mean": float(np.mean([row["transport_gap"] for row in rows])) if rows else 0.0,
        "good_gap_mean": float(np.mean([row["good_gap"] for row in rows])) if rows else 0.0,
        "transport_minus_good_mean": float(np.mean([row["transport_minus_good"] for row in rows])) if rows else 0.0,
        "bad_to_good_l1_mean": float(np.mean([row["bad_to_good_l1"] for row in rows])) if rows else 0.0,
        "bad_to_transport_best_l1_mean": float(np.mean([row["bad_to_transport_best_l1"] for row in rows])) if rows else 0.0,
        "transport_best_to_good_l1_mean": float(np.mean([row["transport_best_to_good_l1"] for row in rows])) if rows else 0.0,
        "rows": rows,
    }
    with open(os.path.join(args.run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
