from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.rl.action_assembler import assemble_actions
from sagin_marl.rl.critic import CriticNet
from sagin_marl.rl.mappo import _stack_value_head_dict, batch_flatten_obs, compute_gae
from sagin_marl.rl.policy import ActorNet
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size <= 1 or y.size <= 1:
        return 0.0
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError("x and y must have the same size")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 1.0e-12 or y_std <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return np.zeros((0,), dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_vals = a[order]
    start = 0
    n = int(a.size)
    while start < n:
        end = start + 1
        while end < n and sorted_vals[end] == sorted_vals[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size <= 1 or y.size <= 1:
        return 0.0
    return _safe_corr(_rankdata(np.asarray(x)), _rankdata(np.asarray(y)))


def _explained_variance(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size <= 1 or target.size <= 1:
        return 0.0
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    target_var = float(np.var(target))
    if target_var <= 1.0e-12:
        return 0.0
    return float(1.0 - np.var(target - pred) / target_var)


def _pair_summary(lhs: np.ndarray, rhs: np.ndarray) -> Dict[str, float]:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    diff = lhs - rhs
    return {
        "n": int(lhs.size),
        "lhs_mean": float(np.mean(lhs)) if lhs.size else 0.0,
        "rhs_mean": float(np.mean(rhs)) if rhs.size else 0.0,
        "gap_mean": float(np.mean(diff)) if diff.size else 0.0,
        "mae": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rmse": float(np.sqrt(np.mean(diff**2))) if diff.size else 0.0,
        "pearson": _safe_corr(lhs, rhs),
        "spearman": _safe_spearman(lhs, rhs),
        "explained_variance_lhs_as_pred": _explained_variance(lhs, rhs),
    }


def _critic_value_scalar(critic: CriticNet, state_t: torch.Tensor, obs_step_t: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        out = critic(state_t, obs_step_t)
    if isinstance(out, dict):
        arr = _stack_value_head_dict(out).detach().cpu().numpy()
        return np.mean(arr, axis=-1).astype(np.float32, copy=False).reshape(-1)
    return out.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)


def _load_models(run_dir: Path, update: int, device: torch.device):
    cfg_path = run_dir / "config_source.yaml"
    cfg = load_config(str(cfg_path))
    env = make_structured_env(cfg, mode="script")
    try:
        obs, _ = env.reset(seed=0)
        obs_list = list(obs.values())
        obs_dim = batch_flatten_obs(obs_list, cfg).shape[1]
        state_dim = int(np.asarray(env.get_global_state(), dtype=np.float32).shape[0])
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
    actor = ActorNet(obs_dim, cfg).to(device)
    critic = CriticNet(state_dim, obs_dim, cfg.num_uav, cfg).to(device)
    load_checkpoint_forgiving(actor, str(run_dir / f"actor_u{update:04d}.pt"), map_location=device, strict=True)
    load_checkpoint_forgiving(critic, str(run_dir / f"critic_u{update:04d}.pt"), map_location=device, strict=True)
    actor.eval()
    critic.eval()
    return cfg, actor, critic


def _collect_full_episode(
    cfg,
    actor: ActorNet,
    critic: CriticNet,
    *,
    device: torch.device,
    seed: int,
    deterministic: bool,
) -> Dict[str, np.ndarray]:
    env = make_structured_env(cfg, mode="script")
    rewards: List[float] = []
    values: List[float] = []
    next_values: List[float] = []
    terminated: List[float] = []
    truncated: List[float] = []
    try:
        obs, _ = env.reset(seed=int(seed))
        done = False
        while not done:
            obs_list = list(obs.values())
            obs_batch = batch_flatten_obs(obs_list, cfg).astype(np.float32, copy=False)
            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
            obs_step_tensor = torch.tensor(
                obs_batch.reshape(1, cfg.num_uav, obs_batch.shape[1]),
                dtype=torch.float32,
                device=device,
            )
            state = np.asarray(env.get_global_state(), dtype=np.float32)
            state_tensor = torch.tensor(state.reshape(1, -1), dtype=torch.float32, device=device)
            value = float(_critic_value_scalar(critic, state_tensor, obs_step_tensor)[0])
            with torch.inference_mode():
                policy_out = actor.act(obs_tensor, deterministic=deterministic, compute_logprob=False)
            accel = (
                policy_out.accel.detach().cpu().numpy()
                if policy_out.accel is not None
                else np.zeros((cfg.num_uav, 2), dtype=np.float32)
            )
            bw = (
                policy_out.bw_action.detach().cpu().numpy()
                if policy_out.bw_action is not None
                else np.zeros((cfg.num_uav, cfg.users_obs_max), dtype=np.float32)
            )
            sat_mask = (
                policy_out.sat_select_mask.detach().cpu().numpy()
                if policy_out.sat_select_mask is not None
                else np.zeros((cfg.num_uav, cfg.sats_obs_max), dtype=np.float32)
            )
            actions = assemble_actions(cfg, env.agents, accel, bw_alloc=bw, sat_select_mask=sat_mask)
            next_obs, reward_dict, term_dict, trunc_dict, _ = env.step(actions)
            reward = float(next(iter(reward_dict.values())))
            term = bool(next(iter(term_dict.values())))
            trunc = bool(next(iter(trunc_dict.values())))
            if term:
                next_value = 0.0
            else:
                next_obs_list = list(next_obs.values())
                next_obs_batch = batch_flatten_obs(next_obs_list, cfg).astype(np.float32, copy=False)
                next_state = np.asarray(env.get_global_state(), dtype=np.float32)
                next_obs_step_tensor = torch.tensor(
                    next_obs_batch.reshape(1, cfg.num_uav, next_obs_batch.shape[1]),
                    dtype=torch.float32,
                    device=device,
                )
                next_state_tensor = torch.tensor(next_state.reshape(1, -1), dtype=torch.float32, device=device)
                next_value = float(_critic_value_scalar(critic, next_state_tensor, next_obs_step_tensor)[0])
            rewards.append(reward)
            values.append(value)
            next_values.append(float(next_value))
            terminated.append(float(term))
            truncated.append(float(trunc))
            obs = next_obs
            done = term or trunc
        return {
            "rewards": np.asarray(rewards, dtype=np.float32),
            "values": np.asarray(values, dtype=np.float32),
            "next_values": np.asarray(next_values, dtype=np.float32),
            "terminated": np.asarray(terminated, dtype=np.float32),
            "truncated": np.asarray(truncated, dtype=np.float32),
        }
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def _compute_targets_for_prefix(
    episode: Dict[str, np.ndarray],
    *,
    prefix_steps: int,
    gamma: float,
    gae_lambda: float,
) -> Dict[str, np.ndarray]:
    rewards = episode["rewards"][:prefix_steps]
    values = episode["values"][:prefix_steps]
    next_values = episode["next_values"][:prefix_steps]
    terminated = episode["terminated"][:prefix_steps]
    truncated = episode["truncated"][:prefix_steps]
    boundaries = np.logical_or(terminated > 0.5, truncated > 0.5)
    _train_adv, train_ret = compute_gae(
        rewards,
        values,
        next_values,
        boundaries,
        float(gamma),
        float(gae_lambda),
    )
    _mc_adv, mc_ret = compute_gae(
        rewards,
        values,
        next_values,
        boundaries,
        float(gamma),
        1.0,
    )
    return {
        "values": np.asarray(values, dtype=np.float64),
        "train_returns": np.asarray(train_ret, dtype=np.float64),
        "mc_returns": np.asarray(mc_ret, dtype=np.float64),
    }


def diagnose_run(
    run_dir: Path,
    *,
    updates: Iterable[int],
    rollout_steps_list: Iterable[int],
    episodes: int,
    episode_seed_base: int,
    deterministic: bool,
    device: torch.device,
) -> Dict[str, object]:
    results: Dict[str, object] = {
        "run_dir": str(run_dir),
        "episodes": int(episodes),
        "episode_seed_base": int(episode_seed_base),
        "deterministic": bool(deterministic),
        "updates": {},
    }
    rollout_steps_list = [int(x) for x in rollout_steps_list]
    for update in updates:
        cfg, actor, critic = _load_models(run_dir, int(update), device)
        per_rollout_values: Dict[int, List[float]] = {steps: [] for steps in rollout_steps_list}
        per_rollout_train: Dict[int, List[float]] = {steps: [] for steps in rollout_steps_list}
        per_rollout_mc: Dict[int, List[float]] = {steps: [] for steps in rollout_steps_list}
        episode_rows: List[Dict[str, float]] = []
        for ep_idx in range(int(episodes)):
            seed = int(episode_seed_base) + ep_idx
            episode = _collect_full_episode(cfg, actor, critic, device=device, seed=seed, deterministic=deterministic)
            full_steps = int(episode["rewards"].shape[0])
            episode_rows.append({"seed": float(seed), "full_episode_steps": float(full_steps)})
            for rollout_steps in rollout_steps_list:
                prefix_steps = min(int(rollout_steps), full_steps)
                targets = _compute_targets_for_prefix(
                    episode,
                    prefix_steps=prefix_steps,
                    gamma=float(cfg.gamma),
                    gae_lambda=float(cfg.gae_lambda),
                )
                per_rollout_values[rollout_steps].extend(targets["values"].tolist())
                per_rollout_train[rollout_steps].extend(targets["train_returns"].tolist())
                per_rollout_mc[rollout_steps].extend(targets["mc_returns"].tolist())
        update_summary: Dict[str, object] = {"episode_rows": episode_rows}
        for rollout_steps in rollout_steps_list:
            values = np.asarray(per_rollout_values[rollout_steps], dtype=np.float64)
            train_returns = np.asarray(per_rollout_train[rollout_steps], dtype=np.float64)
            mc_returns = np.asarray(per_rollout_mc[rollout_steps], dtype=np.float64)
            update_summary[f"steps_{int(rollout_steps)}"] = {
                "train_target_vs_mc": _pair_summary(train_returns, mc_returns),
                "value_vs_train_target": _pair_summary(values, train_returns),
                "value_vs_mc": _pair_summary(values, mc_returns),
            }
        results["updates"][f"u{int(update):04d}"] = update_summary
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--updates", type=int, nargs="+", default=[100, 150, 200])
    parser.add_argument("--rollout_steps", type=int, nargs="+", default=[50, 400])
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--episode_seed_base", type=int, default=42000)
    parser.add_argument("--policy_mode", choices=["deterministic", "stochastic"], default="stochastic")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_path = Path(args.out) if args.out else run_dir / "old_bootstrap_target_diag.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = diagnose_run(
        run_dir,
        updates=args.updates,
        rollout_steps_list=args.rollout_steps,
        episodes=int(args.episodes),
        episode_seed_base=int(args.episode_seed_base),
        deterministic=(args.policy_mode == "deterministic"),
        device=device,
    )
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote old bootstrap target summary to {out_path}")
    for update_key, update_summary in summary["updates"].items():
        print(f"[{update_key}]")
        for rollout_steps in args.rollout_steps:
            block = update_summary[f"steps_{int(rollout_steps)}"]
            target_cmp = block["train_target_vs_mc"]
            value_cmp = block["value_vs_mc"]
            print(
                f"  steps={int(rollout_steps)}: "
                f"train_vs_mc pearson={target_cmp['pearson']:.3f} "
                f"ev={target_cmp['explained_variance_lhs_as_pred']:.3f} "
                f"gap={target_cmp['gap_mean']:.3f} | "
                f"value_vs_mc pearson={value_cmp['pearson']:.3f} "
                f"ev={value_cmp['explained_variance_lhs_as_pred']:.3f}"
            )


if __name__ == "__main__":
    main()
