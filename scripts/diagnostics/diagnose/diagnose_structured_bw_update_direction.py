from __future__ import annotations

import argparse
import json
import os
import random
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
from sagin_marl.rl.baselines import queue_aware_bw_policy
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO, _to_device_dataclass
from sagin_marl.rl.structured_stage_builders import build_local_bw_states_from_snapshot
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_env_group, run_structured_training
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


def _zero_sat_action(cfg) -> np.ndarray:
    select_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 1)
    return np.full((cfg.num_uav, select_k), -1, dtype=np.int64)


def _make_snapshot(driver: StructuredControlDriver, cfg) -> tuple[dict[str, Any], Any, list[dict[str, Any]]]:
    driver.begin_step()
    accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
    z1 = driver.run_accel_stage(accel_zero)
    z2 = driver.run_sat_stage(_zero_sat_action(cfg))
    snapshot = driver.build_bw_stage_snapshot(z2)
    snapshot_state = driver.export_bw_stage_state()
    obs = {agent: driver.env._get_obs(idx) for idx, agent in enumerate(driver.env.agents)}
    return snapshot_state, snapshot, list(obs.values())


def _heuristic_action(obs_list, cfg) -> np.ndarray:
    return np.asarray(queue_aware_bw_policy(obs_list, cfg), dtype=np.float32)


def _bw_action_from_actor(actor, snapshot, device: torch.device, deterministic: bool) -> np.ndarray:
    bw_states = build_local_bw_states_from_snapshot(snapshot)
    local_state = _to_device_dataclass(bw_states[0], device)
    with torch.inference_mode():
        out = actor.act_bw(local_state, deterministic=bool(deterministic))
    return np.asarray(out.action.detach().cpu().numpy(), dtype=np.float32)


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
            action = _bw_action_from_actor(
                actor,
                probe_driver.build_bw_stage_snapshot(z2),
                device,
                deterministic=follow_deterministic,
            )
            discount *= float(gamma)
        return {
            "reward": float(discounted_reward),
            "weighted": float(discounted_weighted),
        }
    finally:
        close_fn = getattr(probe_env, "close", None)
        if callable(close_fn):
            close_fn()


def _collect_panel(cfg, *, episodes: int, states: int, seed: int) -> list[dict[str, Any]]:
    env = make_structured_env(cfg, mode="script")
    driver = as_structured_driver(env)
    entries: list[dict[str, Any]] = []
    try:
        for ep in range(int(episodes)):
            env.reset(seed=int(seed) + ep)
            done = False
            while not done and len(entries) < int(states):
                snapshot_state, snapshot, obs_list = _make_snapshot(driver, cfg)
                heuristic_action = _heuristic_action(obs_list, cfg)
                entries.append(
                    {
                        "episode": int(ep),
                        "t": int(snapshot_state.get("env_state", {}).get("t", 0)),
                        "snapshot_state": snapshot_state,
                        "snapshot": snapshot,
                        "heuristic_action": heuristic_action,
                    }
                )
                step_result = driver.execute_stage_bw_and_step(heuristic_action)
                done = bool(list(step_result.terminations.values())[0] or list(step_result.truncations.values())[0])
                if len(entries) >= int(states):
                    break
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
    return entries


def _evaluate_panel(actor, panel: list[dict[str, Any]], cfg, device: torch.device, *, k_steps: int) -> dict[str, Any]:
    reward_values: list[float] = []
    weighted_values: list[float] = []
    heuristic_reward_values: list[float] = []
    heuristic_weighted_values: list[float] = []
    l1_to_heur_values: list[float] = []
    heuristic_beats_actor: list[float] = []
    rows: list[dict[str, Any]] = []
    for entry in panel:
        snapshot = entry["snapshot"]
        snapshot_state = entry["snapshot_state"]
        heuristic_action = np.asarray(entry["heuristic_action"], dtype=np.float32)
        actor_action = _bw_action_from_actor(actor, snapshot, device, deterministic=True)
        actor_roll = _rollout_from_snapshot(
            snapshot_state=snapshot_state,
            first_action=actor_action,
            cfg=cfg,
            actor=actor,
            device=device,
            k_steps=int(k_steps),
            gamma=float(cfg.gamma),
            follow_deterministic=True,
        )
        heuristic_roll = _rollout_from_snapshot(
            snapshot_state=snapshot_state,
            first_action=heuristic_action,
            cfg=cfg,
            actor=actor,
            device=device,
            k_steps=int(k_steps),
            gamma=float(cfg.gamma),
            follow_deterministic=True,
        )
        valid_mask = np.asarray(snapshot.bw_valid_mask, dtype=bool)
        l1_value = float(np.abs(actor_action - heuristic_action)[valid_mask].sum()) if np.any(valid_mask) else 0.0
        reward_values.append(float(actor_roll["reward"]))
        weighted_values.append(float(actor_roll["weighted"]))
        heuristic_reward_values.append(float(heuristic_roll["reward"]))
        heuristic_weighted_values.append(float(heuristic_roll["weighted"]))
        l1_to_heur_values.append(l1_value)
        heuristic_beats_actor.append(float(heuristic_roll["reward"] > actor_roll["reward"] + 1.0e-9))
        if len(rows) < 6:
            rows.append(
                {
                    "episode": int(entry["episode"]),
                    "t": int(entry["t"]),
                    "actor_reward": float(actor_roll["reward"]),
                    "heuristic_reward": float(heuristic_roll["reward"]),
                    "actor_weighted": float(actor_roll["weighted"]),
                    "heuristic_weighted": float(heuristic_roll["weighted"]),
                    "l1_to_heur": float(l1_value),
                }
            )
    return {
        "actor_reward": _summarize(reward_values),
        "actor_weighted": _summarize(weighted_values),
        "heuristic_reward": _summarize(heuristic_reward_values),
        "heuristic_weighted": _summarize(heuristic_weighted_values),
        "l1_to_heur": _summarize(l1_to_heur_values),
        "heuristic_beats_actor_frac": float(np.mean(heuristic_beats_actor)) if heuristic_beats_actor else 0.0,
        "examples": rows,
    }


def _build_learner(run_dir: Path, device: torch.device):
    cfg = load_config(str(run_dir / "config_source.yaml"))
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    actor_ckpt = run_dir / "actor_final.pt"
    critic_ckpt = run_dir / "critic_final.pt"
    load_checkpoint_forgiving(bundle.actor, str(actor_ckpt), map_location=device, strict=True)
    load_checkpoint_forgiving(bundle.critic, str(critic_ckpt), map_location=device, strict=True)
    actor = bundle.actor.to(device)
    critic = bundle.critic.to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(cfg.actor_lr))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(cfg.critic_lr))
    learner = StructuredMAPPO(
        actor=actor,
        critic=critic,
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_ratio=float(cfg.clip_ratio),
        value_coef=float(cfg.value_coef),
        entropy_coef=float(cfg.entropy_coef),
        max_grad_norm=float(cfg.max_grad_norm),
        ppo_epochs=int(cfg.ppo_epochs),
        num_mini_batch=int(cfg.num_mini_batch),
        target_mode="step_level",
        danger_imitation_enabled=bool(getattr(cfg, "danger_imitation_enabled", False)),
        danger_imitation_coef=float(getattr(cfg, "danger_imitation_coef", 0.0) or 0.0),
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        device=device,
        cfg=cfg,
        train_accel=bool(True if getattr(cfg, "train_accel", None) is None else getattr(cfg, "train_accel")),
        train_sat=bool(True if getattr(cfg, "train_sat", None) is None else getattr(cfg, "train_sat")),
        train_bw=bool(True if getattr(cfg, "train_bw", None) is None else getattr(cfg, "train_bw")),
        exec_accel_source=str(getattr(cfg, "exec_accel_source", "policy") or "policy"),
        exec_sat_source=str(getattr(cfg, "exec_sat_source", "policy") or "policy"),
        exec_bw_source=str(getattr(cfg, "exec_bw_source", "policy") or "policy"),
    )
    actor.eval()
    critic.eval()
    return cfg, actor, critic, learner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--panel_episodes", type=int, default=6)
    parser.add_argument("--panel_states", type=int, default=16)
    parser.add_argument("--panel_seed", type=int, default=35000)
    parser.add_argument("--k_steps", type=int, default=20)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--rollout_env_steps", type=int, default=20)
    parser.add_argument("--rollout_seed_base", type=int, default=45000)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--vec_backend", choices=["sync", "subproc"], default="sync")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_path", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    cfg = load_config(str(run_dir / "config_source.yaml"))
    _set_all_seeds(int(args.panel_seed))
    panel = _collect_panel(
        cfg,
        episodes=int(args.panel_episodes),
        states=int(args.panel_states),
        seed=int(args.panel_seed),
    )
    trials: list[dict[str, Any]] = []
    reward_deltas: list[float] = []
    weighted_deltas: list[float] = []
    l1_deltas: list[float] = []
    heur_gap_deltas: list[float] = []

    for trial_idx in range(int(args.trials)):
        seed = int(args.rollout_seed_base) + trial_idx
        _set_all_seeds(seed)
        cfg_trial, actor, critic, learner = _build_learner(run_dir, device)
        pre_eval = _evaluate_panel(actor, panel, cfg_trial, device, k_steps=int(args.k_steps))
        env_group = make_structured_env_group(cfg_trial, num_envs=max(int(args.num_envs), 1), backend=str(args.vec_backend))
        try:
            history = run_structured_training(
                env_group,
                learner,
                num_updates=1,
                rollout_env_steps=int(args.rollout_env_steps),
                reset_seed=seed,
                reset_on_start=True,
            )
        finally:
            close_structured_env_group(env_group)
        post_eval = _evaluate_panel(actor, panel, cfg_trial, device, k_steps=int(args.k_steps))
        metrics = history[0] if history else None
        reward_delta = float(post_eval["actor_reward"]["mean"] - pre_eval["actor_reward"]["mean"])
        weighted_delta = float(post_eval["actor_weighted"]["mean"] - pre_eval["actor_weighted"]["mean"])
        l1_delta = float(post_eval["l1_to_heur"]["mean"] - pre_eval["l1_to_heur"]["mean"])
        heur_gap_pre = float(pre_eval["heuristic_reward"]["mean"] - pre_eval["actor_reward"]["mean"])
        heur_gap_post = float(post_eval["heuristic_reward"]["mean"] - post_eval["actor_reward"]["mean"])
        heur_gap_delta = float(heur_gap_post - heur_gap_pre)
        reward_deltas.append(reward_delta)
        weighted_deltas.append(weighted_delta)
        l1_deltas.append(l1_delta)
        heur_gap_deltas.append(heur_gap_delta)
        trials.append(
            {
                "trial": int(trial_idx),
                "seed": int(seed),
                "pre": pre_eval,
                "post": post_eval,
                "delta": {
                    "actor_reward_mean": reward_delta,
                    "actor_weighted_mean": weighted_delta,
                    "l1_to_heur_mean": l1_delta,
                    "heuristic_gap_mean": heur_gap_delta,
                },
                "update_metrics": (
                    None
                    if metrics is None
                    else {
                        "env_reward_mean": float(metrics.env_reward_mean),
                        "bw_train_reward_mean": float(metrics.bw_train_reward_mean),
                        "policy_loss": float(metrics.policy_loss),
                        "value_loss_bw": float(metrics.value_loss_bw),
                        "approx_kl_bw": float(metrics.approx_kl_bw),
                        "clip_frac_bw": float(metrics.clip_frac_bw),
                        "entropy_bw": float(metrics.entropy_bw),
                        "clean_group_debug": getattr(learner, "_bw_clean_last_group_debug", None),
                    }
                ),
            }
        )

    payload = {
        "run_dir": str(run_dir),
        "panel_states": int(len(panel)),
        "k_steps": int(args.k_steps),
        "rollout_env_steps": int(args.rollout_env_steps),
        "trials": trials,
        "delta_summary": {
            "actor_reward_mean": _summarize(reward_deltas),
            "actor_weighted_mean": _summarize(weighted_deltas),
            "l1_to_heur_mean": _summarize(l1_deltas),
            "heuristic_gap_mean": _summarize(heur_gap_deltas),
        },
    }
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
