from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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


def _feature_vector(local_state) -> np.ndarray:
    user_nodes = np.asarray(local_state.user_nodes.detach().cpu().numpy(), dtype=np.float32)[0]
    user_edges = np.asarray(local_state.user_edges.detach().cpu().numpy(), dtype=np.float32)[0]
    valid = np.asarray(
        ((local_state.user_mask > 0.5) & (local_state.bw_valid_mask > 0.5)).detach().cpu().numpy(),
        dtype=bool,
    )[0]
    queue = user_nodes[:, 2]
    rel_x = user_edges[:, 0]
    rel_y = user_edges[:, 1]
    eta = user_edges[:, -1]
    feat = np.stack([queue, rel_x, rel_y, eta], axis=-1)
    feat = feat * valid[:, None].astype(np.float32)
    return feat.reshape(-1).astype(np.float32)


def _masked_l1(a: np.ndarray, b: np.ndarray, valid_mask: np.ndarray) -> float:
    mask = np.asarray(valid_mask, dtype=bool)
    return float(np.abs(a - b)[mask].sum())


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--bad_actor", type=str, required=True)
    parser.add_argument("--good_actor", type=str, required=True)
    parser.add_argument("--oracle_run", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    device = _resolve_torch_device(args.device)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(args.config))

    bundle_bad = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    bad_actor = bundle_bad.actor.to(device)
    load_checkpoint_forgiving(bad_actor, str(args.bad_actor), map_location=device, strict=True)
    bad_actor.eval()

    bundle_good = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    good_actor = bundle_good.actor.to(device)
    load_checkpoint_forgiving(good_actor, str(args.good_actor), map_location=device, strict=True)
    good_actor.eval()

    per_state = json.loads((Path(args.oracle_run) / "per_state.json").read_text(encoding="utf-8"))
    num_states = int(len(per_state))
    snapshots = _collect_bw_snapshots(cfg, bad_actor, device=device, num_states=num_states, seed=int(args.seed))

    rows = []
    for idx, (saved, snap) in enumerate(zip(per_state, snapshots, strict=True)):
        feat_now = _feature_vector(snap["local_state"])
        feat_saved = np.asarray(saved["feature"], dtype=np.float32)
        feat_diff = float(np.max(np.abs(feat_now - feat_saved))) if feat_now.size else 0.0
        with torch.no_grad():
            bad_det = np.asarray(bad_actor.act_bw(snap["local_state"], deterministic=True).action.detach().cpu().numpy(), dtype=np.float32)
            good_det = np.asarray(good_actor.act_bw(snap["local_state"], deterministic=True).action.detach().cpu().numpy(), dtype=np.float32)
        local_best = np.asarray(saved["budget_best"]["256"]["action"], dtype=np.float32)
        valid_mask = np.asarray(saved["valid_mask"], dtype=bool)
        with torch.no_grad():
            bad_det_eval = bad_actor.evaluate_bw(
                snap["local_state"],
                torch.as_tensor(bad_det, dtype=torch.float32, device=device),
            )
            local_best_eval = bad_actor.evaluate_bw(
                snap["local_state"],
                torch.as_tensor(local_best, dtype=torch.float32, device=device),
            )
            good_det_eval = bad_actor.evaluate_bw(
                snap["local_state"],
                torch.as_tensor(good_det, dtype=torch.float32, device=device),
            )
        bad_score = _run_bw_candidate_to_episode_end(cfg, bad_actor, snapshot_state=snap["snapshot_state"], first_action=bad_det, device=device)
        local_best_score = _run_bw_candidate_to_episode_end(cfg, bad_actor, snapshot_state=snap["snapshot_state"], first_action=local_best, device=device)
        good_first_bad_follow_score = _run_bw_candidate_to_episode_end(
            cfg,
            bad_actor,
            snapshot_state=snap["snapshot_state"],
            first_action=good_det,
            device=device,
        )
        good_score = _run_bw_candidate_to_episode_end(cfg, good_actor, snapshot_state=snap["snapshot_state"], first_action=good_det, device=device)
        rows.append(
            {
                "state_index": idx,
                "feature_max_abs_diff": feat_diff,
                "bad_score": bad_score,
                "local_best_score": local_best_score,
                "good_first_bad_follow_score": good_first_bad_follow_score,
                "good_score": good_score,
                "local_gap": local_best_score - bad_score,
                "good_first_bad_follow_gap": good_first_bad_follow_score - bad_score,
                "good_gap": good_score - bad_score,
                "good_minus_local_best": good_score - local_best_score,
                "bad_to_local_best_l1": _masked_l1(bad_det, local_best, valid_mask),
                "bad_to_good_l1": _masked_l1(bad_det, good_det, valid_mask),
                "local_best_to_good_l1": _masked_l1(local_best, good_det, valid_mask),
                "bad_logprob_of_bad_det": float(bad_det_eval.logprob.detach().cpu().item()),
                "bad_logprob_of_local_best": float(local_best_eval.logprob.detach().cpu().item()),
                "bad_logprob_of_good_det": float(good_det_eval.logprob.detach().cpu().item()),
            }
        )
        print(
            f"State {idx:02d} | bad={bad_score:.3f} | local_best={local_best_score:.3f} | "
            f"good={good_score:.3f} | good-local={good_score - local_best_score:.3f}"
        )

    summary = {
        "num_states": num_states,
        "feature_max_abs_diff_max": float(max(row["feature_max_abs_diff"] for row in rows)) if rows else 0.0,
        "bad_score_mean": float(np.mean([row["bad_score"] for row in rows])) if rows else 0.0,
        "local_best_score_mean": float(np.mean([row["local_best_score"] for row in rows])) if rows else 0.0,
        "good_first_bad_follow_score_mean": float(np.mean([row["good_first_bad_follow_score"] for row in rows])) if rows else 0.0,
        "good_score_mean": float(np.mean([row["good_score"] for row in rows])) if rows else 0.0,
        "local_gap_mean": float(np.mean([row["local_gap"] for row in rows])) if rows else 0.0,
        "good_first_bad_follow_gap_mean": float(np.mean([row["good_first_bad_follow_gap"] for row in rows])) if rows else 0.0,
        "good_gap_mean": float(np.mean([row["good_gap"] for row in rows])) if rows else 0.0,
        "good_minus_local_best_mean": float(np.mean([row["good_minus_local_best"] for row in rows])) if rows else 0.0,
        "bad_to_local_best_l1_mean": float(np.mean([row["bad_to_local_best_l1"] for row in rows])) if rows else 0.0,
        "bad_to_good_l1_mean": float(np.mean([row["bad_to_good_l1"] for row in rows])) if rows else 0.0,
        "local_best_to_good_l1_mean": float(np.mean([row["local_best_to_good_l1"] for row in rows])) if rows else 0.0,
        "bad_logprob_of_bad_det_mean": float(np.mean([row["bad_logprob_of_bad_det"] for row in rows])) if rows else 0.0,
        "bad_logprob_of_local_best_mean": float(np.mean([row["bad_logprob_of_local_best"] for row in rows])) if rows else 0.0,
        "bad_logprob_of_good_det_mean": float(np.mean([row["bad_logprob_of_good_det"] for row in rows])) if rows else 0.0,
        "good_better_than_local_best_frac": float(
            np.mean([1.0 if row["good_score"] > row["local_best_score"] else 0.0 for row in rows])
        )
        if rows
        else 0.0,
        "rows": rows,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
