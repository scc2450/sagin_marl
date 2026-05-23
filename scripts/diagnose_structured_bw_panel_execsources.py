from __future__ import annotations

import argparse
import json
import os
import random
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
from sagin_marl.rl.baselines import queue_aware_bw_policy
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import _to_device_dataclass
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


def _load_actor(run_dir: Path, device: torch.device):
    cfg = load_config(str(run_dir / "config_source.yaml"))
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    ckpt_path = run_dir / "actor_final.pt"
    load_checkpoint_forgiving(bundle.actor, str(ckpt_path), map_location=device, strict=True)
    actor = bundle.actor.to(device).eval()
    return cfg, actor, ckpt_path


def _zero_sat_action(cfg) -> np.ndarray:
    select_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 1)
    return np.full((cfg.num_uav, select_k), -1, dtype=np.int64)


def _make_snapshot(driver: StructuredControlDriver, cfg) -> tuple[dict[str, Any], Any]:
    driver.begin_step()
    accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
    z1 = driver.run_accel_stage(accel_zero)
    z2 = driver.run_sat_stage(_zero_sat_action(cfg))
    snapshot = driver.build_bw_stage_snapshot(z2)
    snapshot_state = driver.export_bw_stage_state()
    return snapshot_state, snapshot


def _bw_action_from_actor(actor, snapshot, device: torch.device, deterministic: bool) -> np.ndarray:
    bw_states = driver_local_states = None
    from sagin_marl.rl.structured_stage_builders import build_local_bw_states_from_snapshot

    bw_states = build_local_bw_states_from_snapshot(snapshot)
    local_state = _to_device_dataclass(bw_states[0], device)
    with torch.inference_mode():
        out = actor.act_bw(local_state, deterministic=bool(deterministic))
    return np.asarray(out.action.detach().cpu().numpy(), dtype=np.float32)


def _heuristic_action(obs_list, cfg) -> np.ndarray:
    return np.asarray(queue_aware_bw_policy(obs_list, cfg), dtype=np.float32)


def _random_action(valid_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    valid_mask_arr = np.asarray(valid_mask, dtype=bool)
    out = np.zeros(valid_mask_arr.shape, dtype=np.float32)
    for u in range(int(valid_mask_arr.shape[0])):
        valid = np.flatnonzero(valid_mask_arr[u])
        if valid.size <= 0:
            continue
        if valid.size == 1:
            out[u, valid[0]] = 1.0
            continue
        weights = rng.dirichlet(np.ones(valid.size, dtype=np.float64)).astype(np.float32)
        out[u, valid] = weights
    return out


def _rollout_from_snapshot(
    *,
    snapshot_state: dict[str, Any],
    first_action: np.ndarray,
    cfg,
    actor,
    device: torch.device,
    k_steps: int,
    gamma: float,
    follow_deterministic: bool,
) -> dict[str, float]:
    probe_env = make_structured_env(cfg, mode="script")
    probe_driver = as_structured_driver(probe_env)
    try:
        probe_driver.load_bw_stage_state(snapshot_state)
        discounted_reward = 0.0
        discounted_weighted = 0.0
        discount = 1.0
        action = np.asarray(first_action, dtype=np.float32)
        for step in range(int(k_steps)):
            step_result, _next_world = probe_driver.execute_stage_bw_and_prepare_next_accel(action)
            reward = float(next(iter(step_result.rewards.values())))
            weighted = float(getattr(step_result, "bw_weighted_workload_delta_reward", 0.0) or 0.0)
            discounted_reward += discount * reward
            discounted_weighted += discount * weighted
            done = bool(list(step_result.terminations.values())[0] or list(step_result.truncations.values())[0])
            if done or step == int(k_steps) - 1:
                break
            probe_driver.begin_step()
            accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
            z1 = probe_driver.run_accel_stage(accel_zero)
            z2 = probe_driver.run_sat_stage(_zero_sat_action(cfg))
            action = _bw_action_from_actor(actor, probe_driver.build_bw_stage_snapshot(z2), device, deterministic=follow_deterministic)
            discount *= float(gamma)
        return {
            "reward": float(discounted_reward),
            "weighted": float(discounted_weighted),
        }
    finally:
        close_fn = getattr(probe_env, "close", None)
        if callable(close_fn):
            close_fn()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--states", type=int, default=24)
    parser.add_argument("--sample_count", type=int, default=16)
    parser.add_argument("--random_count", type=int, default=4)
    parser.add_argument("--k_steps", type=int, default=2)
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
    rng = np.random.default_rng(int(args.seed) + 17)
    env = make_structured_env(cfg, mode="script")
    driver = as_structured_driver(env)
    collected = 0
    state_rows: list[dict[str, Any]] = []
    beat_sample_frac: list[float] = []
    beat_heur_frac: list[float] = []
    best_minus_det_reward: list[float] = []
    best_minus_det_weighted: list[float] = []
    source_hist: dict[str, int] = {}

    try:
        for ep in range(int(args.episodes)):
            env.reset(seed=int(args.seed) + ep)
            done = False
            while not done and collected < int(args.states):
                snapshot_state, snapshot = _make_snapshot(driver, cfg)
                valid_mask = np.asarray(snapshot.bw_valid_mask, dtype=bool)
                obs = {agent: env._get_obs(idx) for idx, agent in enumerate(env.agents)}
                obs_list = list(obs.values())

                candidates: list[tuple[str, np.ndarray]] = []
                latent_det = _bw_action_from_actor(actor, snapshot, device, deterministic=True)
                candidates.append(("latent_det", latent_det))
                heuristic = _heuristic_action(obs_list, cfg)
                candidates.append(("heuristic", heuristic))
                for i in range(int(args.sample_count)):
                    candidates.append((f"sample_{i}", _bw_action_from_actor(actor, snapshot, device, deterministic=False)))
                for i in range(int(args.random_count)):
                    candidates.append((f"random_{i}", _random_action(valid_mask, rng)))

                scored: list[dict[str, Any]] = []
                for name, action in candidates:
                    totals = _rollout_from_snapshot(
                        snapshot_state=snapshot_state,
                        first_action=action,
                        cfg=cfg,
                        actor=actor,
                        device=device,
                        k_steps=int(args.k_steps),
                        gamma=float(cfg.gamma),
                        follow_deterministic=True,
                    )
                    scored.append(
                        {
                            "name": str(name),
                            "reward": float(totals["reward"]),
                            "weighted": float(totals["weighted"]),
                        }
                    )
                best = max(scored, key=lambda row: row["reward"])
                score_map = {row["name"]: row for row in scored}
                latent_row = score_map["latent_det"]
                heuristic_row = score_map["heuristic"]
                sample_rows = [row for row in scored if row["name"].startswith("sample_")]
                best_sample = max(sample_rows, key=lambda row: row["reward"]) if sample_rows else None

                source_hist[str(best["name"])] = int(source_hist.get(str(best["name"]), 0) + 1)
                best_minus_det_reward.append(float(best["reward"] - latent_row["reward"]))
                best_minus_det_weighted.append(float(best["weighted"] - latent_row["weighted"]))
                if best_sample is not None:
                    beat_sample_frac.append(float(best_sample["reward"] > latent_row["reward"] + 1.0e-9))
                beat_heur_frac.append(float(best["reward"] > heuristic_row["reward"] + 1.0e-9))

                if len(state_rows) < 6:
                    state_rows.append(
                        {
                            "episode": int(ep),
                            "t": int(snapshot_state.get("env_state", {}).get("t", 0)),
                            "candidate_names": [row["name"] for row in scored],
                            "reward_scores": [float(row["reward"]) for row in scored],
                            "weighted_scores": [float(row["weighted"]) for row in scored],
                            "best_name": str(best["name"]),
                            "best_reward": float(best["reward"]),
                            "latent_det_reward": float(latent_row["reward"]),
                            "heuristic_reward": float(heuristic_row["reward"]),
                        }
                    )

                selected_action = heuristic
                step_result = driver.execute_stage_bw_and_step(selected_action)
                done = bool(list(step_result.terminations.values())[0] or list(step_result.truncations.values())[0])
                collected += 1
                if collected >= int(args.states):
                    break
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

    payload = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "states": int(collected),
        "k_steps": int(args.k_steps),
        "sample_count": int(args.sample_count),
        "random_count": int(args.random_count),
        "best_minus_latent_reward": _summarize(best_minus_det_reward),
        "best_minus_latent_weighted": _summarize(best_minus_det_weighted),
        "best_sample_beats_latent_frac": float(np.mean(np.asarray(beat_sample_frac, dtype=np.float64))) if beat_sample_frac else 0.0,
        "best_reward_beats_heuristic_frac": float(np.mean(np.asarray(beat_heur_frac, dtype=np.float64))) if beat_heur_frac else 0.0,
        "selection_source_hist": {str(k): int(v) for k, v in sorted(source_hist.items())},
        "examples": state_rows,
    }
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
