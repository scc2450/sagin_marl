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

from scripts.diagnostics.audit.audit_bw_broad2local_offline import _collect_panel_bank
from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.baselines import queue_aware_bw_policy
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import _to_device_dataclass
from sagin_marl.rl.structured_stage_builders import build_local_bw_states_from_snapshot
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
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _drop_batch_axis(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    if out.ndim >= 1 and out.shape[0] == 1:
        return out[0]
    return out


def _load_actor(run_dir: Path, device: torch.device, actor_checkpoint: str | None):
    cfg = load_config(str(run_dir / "config_source.yaml"))
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    ckpt = Path(actor_checkpoint) if actor_checkpoint is not None else run_dir / "actor_final.pt"
    load_checkpoint_forgiving(bundle.actor, str(ckpt), map_location=device, strict=True)
    actor = bundle.actor.to(device).eval()
    return cfg, actor, ckpt


def _bw_action_from_actor(actor, snapshot, device: torch.device, deterministic: bool) -> np.ndarray:
    bw_state = _to_device_dataclass(build_local_bw_states_from_snapshot(snapshot)[0], device)
    with torch.no_grad():
        out = actor.act_bw(bw_state, deterministic=bool(deterministic))
    return np.asarray(out.action.detach().cpu().numpy(), dtype=np.float32)


def _rollout_weighted_from_snapshot(
    *,
    snapshot_state: dict[str, Any],
    first_action: np.ndarray,
    cfg,
    actor,
    device: torch.device,
    k_steps: int,
    gamma: float,
    follow_deterministic: bool,
    bw_follow_mode: str,
) -> float:
    probe_env = make_structured_env(cfg, mode="script")
    probe_driver = as_structured_driver(probe_env)
    bw_follow_mode_l = str(bw_follow_mode).strip().lower()
    if bw_follow_mode_l not in {"policy", "queue_aware"}:
        raise ValueError(f"Unsupported bw_follow_mode: {bw_follow_mode}")
    try:
        probe_driver.load_bw_stage_state(snapshot_state)
        total = 0.0
        discount = 1.0
        action = np.asarray(first_action, dtype=np.float32)
        for step in range(int(k_steps)):
            step_result, _next_world = probe_driver.execute_stage_bw_and_prepare_next_accel(action)
            step_score = float(getattr(step_result, "bw_weighted_workload_delta_reward", 0.0) or 0.0)
            total += discount * step_score
            done = bool(list(step_result.terminations.values())[0] or list(step_result.truncations.values())[0])
            if done or step == int(k_steps) - 1:
                break
            probe_driver.begin_step()
            accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
            _ = probe_driver.run_accel_stage(accel_zero)
            select_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 1)
            z2 = probe_driver.run_sat_stage(np.full((cfg.num_uav, select_k), -1, dtype=np.int64))
            if bw_follow_mode_l == "queue_aware":
                obs = {agent: probe_env._get_obs(idx) for idx, agent in enumerate(probe_env.agents)}
                action = np.asarray(queue_aware_bw_policy(list(obs.values()), cfg), dtype=np.float32)
            else:
                bw_snapshot = probe_driver.build_bw_stage_snapshot(z2)
                action = _bw_action_from_actor(actor, bw_snapshot, device, deterministic=follow_deterministic)
            discount *= float(gamma)
        return float(total)
    finally:
        close_fn = getattr(probe_env, "close", None)
        if callable(close_fn):
            close_fn()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--actor_checkpoint", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--panel_states", type=int, default=8)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--vec_backend", choices=["sync", "subproc"], default="sync")
    parser.add_argument("--seed", type=int, default=22420)
    parser.add_argument("--heuristic_bw_source", type=str, default="queue_aware")
    parser.add_argument("--k_long", type=int, default=20)
    parser.add_argument("--bw_follow_mode", choices=["policy", "queue_aware"], default="queue_aware")
    parser.add_argument("--policy_mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--panel_random_count", type=int, default=2)
    parser.add_argument("--prefer_policy_gap_min", type=float, default=1.0e-3)
    parser.add_argument("--min_best_gap", type=float, default=0.02)
    parser.add_argument("--bw_deterministic_opt_steps", type=int, default=8)
    parser.add_argument("--bw_deterministic_step_size", type=float, default=0.5)
    parser.add_argument("--max_examples", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_path", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_all_seeds(int(args.seed))
    run_dir = Path(args.run_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cfg, actor, ckpt_path = _load_actor(
        run_dir,
        device,
        None if args.actor_checkpoint is None else str(args.actor_checkpoint),
    )

    bank = _collect_panel_bank(
        run_dir=run_dir,
        update=0,
        device=device,
        actor_checkpoint=None if args.actor_checkpoint is None else str(args.actor_checkpoint),
        output_dir=out_path.parent,
        episodes=int(args.episodes),
        panel_states=int(args.panel_states),
        policy_mode=str(args.policy_mode),
        panel_random_count=int(args.panel_random_count),
        heuristic_bw_source=str(args.heuristic_bw_source),
        k_steps=int(args.k_long),
        seed=int(args.seed),
        prefer_policy_gap_min=float(args.prefer_policy_gap_min),
        min_best_gap=float(args.min_best_gap),
        num_envs=int(args.num_envs),
        bw_deterministic_readout="latent_mean_pushforward",
        bw_deterministic_opt_steps=int(args.bw_deterministic_opt_steps),
        bw_deterministic_step_size=float(args.bw_deterministic_step_size),
        bw_parameterization=None,
        bw_alpha_init_bias=None,
        bw_alpha_max=None,
        bw_tau_enabled=None,
        bw_tau_init_bias=None,
        bw_tau_min=None,
        bw_tau_max=None,
        vec_backend=str(args.vec_backend),
    )
    entries = list(bank["entries"])
    differing_examples: list[dict[str, Any]] = []
    best_name_same: list[float] = []
    gap_values: list[float] = []

    for state_idx, entry in enumerate(entries):
        candidate_names = list(entry["panel_names"]) + ["policy_det"]
        candidate_actions = [np.asarray(action, dtype=np.float32) for action in entry["panel_actions"]] + [
            np.asarray(entry["policy_det_action"], dtype=np.float32)
        ]
        score_k1: list[float] = []
        score_k20: list[float] = []
        for action in candidate_actions:
            score_k1.append(
                _rollout_weighted_from_snapshot(
                    snapshot_state=entry["snapshot_state"],
                    first_action=action,
                    cfg=cfg,
                    actor=actor,
                    device=device,
                    k_steps=1,
                    gamma=float(cfg.gamma),
                    follow_deterministic=(str(args.policy_mode) == "deterministic"),
                    bw_follow_mode=str(args.bw_follow_mode),
                )
            )
            score_k20.append(
                _rollout_weighted_from_snapshot(
                    snapshot_state=entry["snapshot_state"],
                    first_action=action,
                    cfg=cfg,
                    actor=actor,
                    device=device,
                    k_steps=int(args.k_long),
                    gamma=float(cfg.gamma),
                    follow_deterministic=(str(args.policy_mode) == "deterministic"),
                    bw_follow_mode=str(args.bw_follow_mode),
                )
            )
        score_k1_arr = np.asarray(score_k1, dtype=np.float64)
        score_k20_arr = np.asarray(score_k20, dtype=np.float64)
        best_k1 = int(np.argmax(score_k1_arr))
        best_k20 = int(np.argmax(score_k20_arr))
        best_name_same.append(float(best_k1 == best_k20))
        gap_values.append(float(score_k20_arr[best_k20] - score_k20_arr[best_k1]))

        local_state = entry["local_state"]
        user_nodes = _drop_batch_axis(np.asarray(local_state.user_nodes, dtype=np.float32))
        user_edges = _drop_batch_axis(np.asarray(local_state.user_edges, dtype=np.float32))
        bw_valid_mask = _drop_batch_axis(np.asarray(local_state.bw_valid_mask, dtype=bool))
        queue = np.asarray(user_nodes[:, 2], dtype=np.float32).tolist() if user_nodes.ndim == 2 and user_nodes.shape[-1] >= 3 else []
        eta = np.asarray(user_edges[:, -1], dtype=np.float32).tolist() if user_edges.ndim == 2 and user_edges.shape[-1] >= 1 else []

        if best_k1 != best_k20 and len(differing_examples) < int(args.max_examples):
            differing_examples.append(
                {
                    "state_index": int(state_idx),
                    "t": int(entry["t"]),
                    "queue": queue,
                    "eta": eta,
                    "valid_mask": np.asarray(bw_valid_mask, dtype=np.int32).tolist(),
                    "candidate_names": candidate_names,
                    "score_k1": [float(x) for x in score_k1_arr.tolist()],
                    "score_k20": [float(x) for x in score_k20_arr.tolist()],
                    "best_k1_name": str(candidate_names[best_k1]),
                    "best_k20_name": str(candidate_names[best_k20]),
                    "best_k1_action": np.asarray(candidate_actions[best_k1], dtype=np.float32).tolist(),
                    "best_k20_action": np.asarray(candidate_actions[best_k20], dtype=np.float32).tolist(),
                    "k20_gain_of_switch": float(score_k20_arr[best_k20] - score_k20_arr[best_k1]),
                }
            )

    payload = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "bw_follow_mode": str(args.bw_follow_mode),
        "policy_mode": str(args.policy_mode),
        "states": int(len(entries)),
        "k_long": int(args.k_long),
        "best_name_same_frac": float(np.mean(np.asarray(best_name_same, dtype=np.float64))) if best_name_same else 0.0,
        "k20_gain_of_switch": _summarize(gap_values),
        "examples": differing_examples,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
