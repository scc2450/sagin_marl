from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

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
from sagin_marl.rl.baselines import queue_aware_bw_policy, uniform_bw_policy
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import _to_device_dataclass
from sagin_marl.rl.structured_stage_builders import build_local_bw_states_from_snapshot
from sagin_marl.rl.structured_types import BwStageSnapshot
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "max": float(np.max(arr)),
    }


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size <= 1 or y.size <= 1 or x.size != y.size:
        return 0.0
    if float(np.std(x)) <= 1.0e-12 or float(np.std(y)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _load_actor(run_dir: Path, device: torch.device):
    cfg = load_config(str(run_dir / "config_source.yaml"))
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    ckpt_path = run_dir / "actor_final.pt"
    load_checkpoint_forgiving(bundle.actor, str(ckpt_path), map_location=device, strict=True)
    actor = bundle.actor.to(device).eval()
    return cfg, actor, ckpt_path


def _collect_snapshots(
    cfg,
    *,
    episodes: int,
    max_states: int,
    seed: int,
    step_policy: str,
) -> tuple[list[BwStageSnapshot], Counter]:
    env = make_structured_env(cfg, mode="script")
    driver = as_structured_driver(env)
    snapshots: list[BwStageSnapshot] = []
    order_counter: Counter = Counter()
    select_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 1)
    sat_dummy = np.full((cfg.num_uav, select_k), -1, dtype=np.int64)
    try:
        for ep in range(int(episodes)):
            env.reset(seed=int(seed) + ep)
            done = False
            while not done and len(snapshots) < int(max_states):
                driver.begin_step()
                accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
                z1 = driver.run_accel_stage(accel_zero)
                z2 = driver.run_sat_stage(sat_dummy)
                snapshot = driver.build_bw_stage_snapshot(z2)
                order_key = tuple(int(x) for x in np.asarray(snapshot.candidate_indices[0], dtype=np.int64).tolist())
                order_counter[order_key] += 1
                valid_count = int(np.asarray(snapshot.bw_valid_mask[0], dtype=bool).sum())
                if valid_count >= 2:
                    snapshots.append(snapshot)
                if step_policy == "queue_aware_bw":
                    obs = {agent: env._get_obs(idx) for idx, agent in enumerate(env.agents)}
                    bw_action = queue_aware_bw_policy(list(obs.values()), cfg)
                else:
                    bw_action = uniform_bw_policy(cfg.num_uav, cfg.users_obs_max)
                step_result = driver.execute_stage_bw_and_step(bw_action)
                done = bool(list(step_result.terminations.values())[0] or list(step_result.truncations.values())[0])
                if len(snapshots) >= int(max_states):
                    break
            if len(snapshots) >= int(max_states):
                break
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
    return snapshots, order_counter


def _make_permutation(valid_mask: np.ndarray, rng: np.random.Generator, mode: str) -> np.ndarray:
    valid_idx = np.flatnonzero(np.asarray(valid_mask, dtype=bool))
    invalid_idx = np.flatnonzero(~np.asarray(valid_mask, dtype=bool))
    if mode == "reverse":
        perm_valid = valid_idx[::-1]
    elif mode == "random":
        perm_valid = valid_idx.copy()
        rng.shuffle(perm_valid)
    else:
        raise ValueError(f"Unsupported permutation mode: {mode}")
    return np.concatenate([perm_valid, invalid_idx]).astype(np.int64, copy=False)


def _candidate_features(snapshot: BwStageSnapshot) -> tuple[np.ndarray, np.ndarray]:
    ws = snapshot.world_state
    cand = np.asarray(snapshot.candidate_indices[0], dtype=np.int64)
    valid = cand >= 0
    queue = np.zeros_like(cand, dtype=np.float32)
    eta = np.zeros_like(cand, dtype=np.float32)
    if np.any(valid):
        idx = cand[valid]
        queue[valid] = np.asarray(ws.gu_nodes[0, idx, 2], dtype=np.float32)
        eta[valid] = np.asarray(ws.uav_gu_edges[0, 0, idx, -1], dtype=np.float32)
    return queue, eta


def _eval_action(actor, snapshot: BwStageSnapshot, device: torch.device) -> np.ndarray:
    state = build_local_bw_states_from_snapshot(snapshot)[0]
    state = _to_device_dataclass(state, device)
    with torch.inference_mode():
        action = actor.act_bw(state, deterministic=True).action.detach().cpu().numpy()
    return np.asarray(action[0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--state_config", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--max_states", type=int, default=128)
    parser.add_argument("--step_policy", choices=["uniform_bw", "queue_aware_bw"], default="uniform_bw")
    parser.add_argument("--random_perms", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_path", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_all_seeds(int(args.seed))
    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    cfg, actor, ckpt_path = _load_actor(run_dir, device)
    state_cfg = load_config(str(args.state_config)) if args.state_config else cfg
    snapshots, order_counter = _collect_snapshots(
        state_cfg,
        episodes=int(args.episodes),
        max_states=int(args.max_states),
        seed=int(args.seed),
        step_policy=str(args.step_policy),
    )
    rng = np.random.default_rng(int(args.seed) + 17)

    reverse_l1: list[float] = []
    reverse_corr: list[float] = []
    reverse_top1_same: list[float] = []
    random_l1: list[float] = []
    random_corr: list[float] = []
    random_top1_same: list[float] = []
    examples: list[dict[str, object]] = []

    for state_idx, snapshot in enumerate(snapshots):
        valid_mask = np.asarray(snapshot.bw_valid_mask[0], dtype=bool)
        orig_action = _eval_action(actor, snapshot, device)
        queue, eta = _candidate_features(snapshot)

        reverse_perm = _make_permutation(valid_mask, rng, mode="reverse")
        reverse_snapshot = BwStageSnapshot(
            world_state=snapshot.world_state,
            candidate_indices=np.asarray(snapshot.candidate_indices, dtype=np.int64)[:, reverse_perm],
            bw_valid_mask=np.asarray(snapshot.bw_valid_mask, dtype=bool)[:, reverse_perm],
        )
        reverse_action = _eval_action(actor, reverse_snapshot, device)
        reverse_mapped = np.zeros_like(orig_action)
        reverse_mapped[reverse_perm] = reverse_action
        reverse_l1.append(float(np.abs(orig_action[valid_mask] - reverse_mapped[valid_mask]).sum()))
        reverse_corr.append(_safe_corr(orig_action[valid_mask], reverse_mapped[valid_mask]))
        reverse_top1_same.append(
            1.0 if int(np.argmax(orig_action[valid_mask])) == int(np.argmax(reverse_mapped[valid_mask])) else 0.0
        )

        for _ in range(int(args.random_perms)):
            perm = _make_permutation(valid_mask, rng, mode="random")
            perm_snapshot = BwStageSnapshot(
                world_state=snapshot.world_state,
                candidate_indices=np.asarray(snapshot.candidate_indices, dtype=np.int64)[:, perm],
                bw_valid_mask=np.asarray(snapshot.bw_valid_mask, dtype=bool)[:, perm],
            )
            perm_action = _eval_action(actor, perm_snapshot, device)
            mapped = np.zeros_like(orig_action)
            mapped[perm] = perm_action
            random_l1.append(float(np.abs(orig_action[valid_mask] - mapped[valid_mask]).sum()))
            random_corr.append(_safe_corr(orig_action[valid_mask], mapped[valid_mask]))
            random_top1_same.append(
                1.0 if int(np.argmax(orig_action[valid_mask])) == int(np.argmax(mapped[valid_mask])) else 0.0
            )

        if len(examples) < 3:
            examples.append(
                {
                    "state_index": int(state_idx),
                    "candidate_indices": np.asarray(snapshot.candidate_indices[0], dtype=np.int64).tolist(),
                    "bw_valid_mask": valid_mask.astype(int).tolist(),
                    "queue": queue.tolist(),
                    "eta": eta.tolist(),
                    "orig_action": orig_action.tolist(),
                    "reverse_perm": reverse_perm.tolist(),
                    "reverse_mapped_action": reverse_mapped.tolist(),
                    "reverse_l1": reverse_l1[-1],
                }
            )

    order_rows = [
        {"order": list(order), "count": int(count)}
        for order, count in order_counter.most_common(10)
    ]
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "parameterization": str(getattr(cfg, "structured_bw_parameterization", "unknown")),
        "step_policy": str(args.step_policy),
        "states": int(len(snapshots)),
        "order_patterns_top10": order_rows,
        "reverse_perm": {
            "l1": _summarize(reverse_l1),
            "corr_mean": float(np.mean(np.asarray(reverse_corr, dtype=np.float64))) if reverse_corr else 0.0,
            "top1_same_mean": float(np.mean(np.asarray(reverse_top1_same, dtype=np.float64))) if reverse_top1_same else 0.0,
        },
        "random_perm": {
            "l1": _summarize(random_l1),
            "corr_mean": float(np.mean(np.asarray(random_corr, dtype=np.float64))) if random_corr else 0.0,
            "top1_same_mean": float(np.mean(np.asarray(random_top1_same, dtype=np.float64))) if random_top1_same else 0.0,
            "num_trials": int(len(random_l1)),
        },
        "examples": examples,
    }
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
