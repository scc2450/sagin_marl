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


def _sample_wide_dirichlet_actions(
    det_mean: np.ndarray,
    valid_mask: np.ndarray,
    *,
    num_samples: int,
    concentration: float,
) -> list[np.ndarray]:
    mean = np.asarray(det_mean, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    conc = max(float(concentration), 1.0e-3)
    actions: list[np.ndarray] = []
    for _ in range(max(int(num_samples), 0)):
        out = np.zeros_like(mean, dtype=np.float32)
        for u in range(mask.shape[0]):
            valid = np.flatnonzero(mask[u])
            if valid.size == 0:
                continue
            row_mean = np.asarray(mean[u, valid], dtype=np.float64)
            row_mean = np.clip(row_mean, 1.0e-6, None)
            row_mean = row_mean / row_mean.sum()
            alpha = np.clip(row_mean * conc, 1.0e-3, None)
            out[u, valid] = np.random.dirichlet(alpha).astype(np.float32)
        actions.append(out)
    return actions


def _masked_l1(a: np.ndarray, b: np.ndarray, valid_mask: np.ndarray) -> float:
    mask = np.asarray(valid_mask, dtype=bool)
    return float(np.abs(a - b)[mask].sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--bad_actor", type=str, required=True)
    parser.add_argument("--good_actor", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_states", type=int, default=4)
    parser.add_argument("--num_candidates", type=int, default=256)
    parser.add_argument("--wide_dirichlet_concentration", type=float, default=0.25)
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
        candidates = _sample_wide_dirichlet_actions(
            bad_det,
            valid_mask,
            num_samples=int(args.num_candidates),
            concentration=float(args.wide_dirichlet_concentration),
        )
        dists = np.asarray([_masked_l1(action, good_det, valid_mask) for action in candidates], dtype=np.float64)
        row = {
            "state_index": idx,
            "bad_to_good_l1": _masked_l1(bad_det, good_det, valid_mask),
            "min_candidate_to_good_l1": float(dists.min()),
            "p10_candidate_to_good_l1": float(np.quantile(dists, 0.1)),
            "median_candidate_to_good_l1": float(np.quantile(dists, 0.5)),
            "frac_l1_le_0.1": float(np.mean(dists <= 0.1)),
            "frac_l1_le_0.2": float(np.mean(dists <= 0.2)),
            "frac_l1_le_0.3": float(np.mean(dists <= 0.3)),
            "frac_l1_le_0.5": float(np.mean(dists <= 0.5)),
        }
        rows.append(row)
        print(
            f"State {idx:02d} | bad->good={row['bad_to_good_l1']:.3f} | "
            f"minCand={row['min_candidate_to_good_l1']:.3f} | "
            f"frac<=0.3={row['frac_l1_le_0.3']:.3f}"
        )

    summary = {
        "num_states": len(rows),
        "num_candidates": int(args.num_candidates),
        "wide_dirichlet_concentration": float(args.wide_dirichlet_concentration),
        "bad_to_good_l1_mean": float(np.mean([row["bad_to_good_l1"] for row in rows])) if rows else 0.0,
        "min_candidate_to_good_l1_mean": float(np.mean([row["min_candidate_to_good_l1"] for row in rows])) if rows else 0.0,
        "median_candidate_to_good_l1_mean": float(np.mean([row["median_candidate_to_good_l1"] for row in rows])) if rows else 0.0,
        "frac_l1_le_0.1_mean": float(np.mean([row["frac_l1_le_0.1"] for row in rows])) if rows else 0.0,
        "frac_l1_le_0.2_mean": float(np.mean([row["frac_l1_le_0.2"] for row in rows])) if rows else 0.0,
        "frac_l1_le_0.3_mean": float(np.mean([row["frac_l1_le_0.3"] for row in rows])) if rows else 0.0,
        "frac_l1_le_0.5_mean": float(np.mean([row["frac_l1_le_0.5"] for row in rows])) if rows else 0.0,
        "rows": rows,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
