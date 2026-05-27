from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

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
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO
from sagin_marl.rl.structured_rollout_debug import copy_prefix_rollout_buffer
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size <= 1 or y.size <= 1:
        return 0.0
    if x.size != y.size:
        raise ValueError("x and y must have the same size")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 1e-12 or y_std <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


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
    }


def _component_summary(component: np.ndarray, total: np.ndarray) -> Dict[str, float]:
    component = np.asarray(component, dtype=np.float64)
    total = np.asarray(total, dtype=np.float64)
    abs_total_mean = float(np.mean(np.abs(total))) if total.size else 0.0
    return {
        "mean": float(np.mean(component)) if component.size else 0.0,
        "abs_mean": float(np.mean(np.abs(component))) if component.size else 0.0,
        "share_of_abs_total": (
            float(np.mean(np.abs(component))) / abs_total_mean if abs_total_mean > 1.0e-12 else 0.0
        ),
    }


def _load_bundle(cfg, run_dir: Path, update: int, device: torch.device):
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    load_checkpoint_forgiving(bundle.actor, str(run_dir / f"actor_u{update:04d}.pt"), map_location=device, strict=True)
    load_checkpoint_forgiving(bundle.critic, str(run_dir / f"critic_u{update:04d}.pt"), map_location=device, strict=True)
    bundle.actor.to(device).eval()
    bundle.critic.to(device).eval()
    return bundle


def _collect_full_episode(
    cfg,
    actor,
    critic,
    *,
    device: torch.device,
    deterministic: bool,
    seed: int,
) -> StructuredRolloutBuffer:
    env = make_structured_env(cfg, mode="script")
    try:
        env.reset(seed=int(seed))
        driver = as_structured_driver(env)
        buffer = StructuredRolloutBuffer()
        algo = StructuredMAPPO(
            actor,
            critic,
            gamma=float(cfg.gamma),
            gae_lambda=float(cfg.gae_lambda),
            clip_ratio=float(getattr(cfg, "clip_ratio", 0.2) or 0.2),
            value_coef=float(getattr(cfg, "value_coef", 0.5) or 0.5),
            entropy_coef=float(getattr(cfg, "entropy_coef", 0.0) or 0.0),
            max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5) or 0.5),
            ppo_epochs=int(getattr(cfg, "ppo_epochs", 1) or 1),
            num_mini_batch=int(getattr(cfg, "num_mini_batch", 1) or 1),
            device=device,
        )
        done = False
        while not done:
            step_result = algo.collect_env_step(driver, buffer, deterministic=deterministic)
            terminated = bool(next(iter(step_result.terminations.values())))
            truncated = bool(next(iter(step_result.truncations.values())))
            done = terminated or truncated
        return buffer
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def _prefix_buffer(full_buffer: StructuredRolloutBuffer, rollout_env_steps: int) -> StructuredRolloutBuffer:
    return copy_prefix_rollout_buffer(full_buffer, int(rollout_env_steps))


def _episode_bias_diagnostics(
    cfg,
    actor,
    critic,
    *,
    device: torch.device,
    deterministic: bool,
    seed: int,
    rollout_env_steps: int,
) -> Dict[str, np.ndarray]:
    full_buffer = _collect_full_episode(
        cfg,
        actor,
        critic,
        device=device,
        deterministic=deterministic,
        seed=seed,
    )
    prefix = _prefix_buffer(full_buffer, int(rollout_env_steps))
    prefix_return_view = prefix.build_return_view()
    if int(prefix_return_view.transition_count) % 3 != 0:
        raise ValueError("Expected prefix transitions grouped by 3 stages")

    full_mc = full_buffer.compute_gae(gamma_env=float(cfg.gamma), gae_lambda=1.0, bootstrap_value=0.0)
    mc_full = np.asarray(full_mc["returns"], dtype=np.float64)
    values_prefix = np.asarray(prefix_return_view.values, dtype=np.float64)
    rewards_prefix = np.asarray(prefix_return_view.rewards, dtype=np.float64)
    terminals_prefix = np.asarray(np.logical_or(prefix_return_view.terminated, prefix_return_view.truncated), dtype=bool)
    num_steps = int(prefix_return_view.transition_count) // 3

    algo = StructuredMAPPO(
        actor,
        critic,
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_ratio=float(getattr(cfg, "clip_ratio", 0.2) or 0.2),
        value_coef=float(getattr(cfg, "value_coef", 0.5) or 0.5),
        entropy_coef=float(getattr(cfg, "entropy_coef", 0.0) or 0.0),
        max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5) or 0.5),
        ppo_epochs=int(getattr(cfg, "ppo_epochs", 1) or 1),
        num_mini_batch=int(getattr(cfg, "num_mini_batch", 1) or 1),
        device=device,
    )
    final_bootstrap_value = 0.0
    bootstrap_worlds = prefix.build_bootstrap_world_state_dict()
    if bootstrap_worlds:
        final_bootstrap_value = float(algo.bootstrap_value(next(iter(bootstrap_worlds.values()))))

    mc_step = np.asarray([mc_full[step_idx * 3] for step_idx in range(num_steps)], dtype=np.float64)
    step_reward = np.asarray([rewards_prefix[step_idx * 3 + 2] for step_idx in range(num_steps)], dtype=np.float64)
    terminal_step = np.asarray([terminals_prefix[step_idx * 3 + 2] for step_idx in range(num_steps)], dtype=bool)

    next_value = np.zeros((num_steps,), dtype=np.float64)
    next_mc = np.zeros((num_steps,), dtype=np.float64)
    for step_idx in range(num_steps):
        if terminal_step[step_idx]:
            next_value[step_idx] = 0.0
            next_mc[step_idx] = 0.0
        elif step_idx == num_steps - 1:
            next_value[step_idx] = float(final_bootstrap_value)
            next_mc[step_idx] = float(mc_full[num_steps * 3]) if (num_steps * 3) < mc_full.size else 0.0
        else:
            next_value[step_idx] = float(values_prefix[(step_idx + 1) * 3])
            next_mc[step_idx] = float(mc_full[(step_idx + 1) * 3])

    gamma = float(cfg.gamma)
    lam = float(cfg.gae_lambda)
    step_target = np.zeros((num_steps,), dtype=np.float64)
    direct_bias = np.zeros((num_steps,), dtype=np.float64)
    tail_bias = np.zeros((num_steps,), dtype=np.float64)

    for step_idx in range(num_steps - 1, -1, -1):
        if terminal_step[step_idx]:
            step_target[step_idx] = step_reward[step_idx]
            direct_bias[step_idx] = 0.0
            tail_bias[step_idx] = 0.0
            continue
        if step_idx == num_steps - 1:
            step_target[step_idx] = step_reward[step_idx] + gamma * next_value[step_idx]
            direct_bias[step_idx] = 0.0
            tail_bias[step_idx] = gamma * (next_value[step_idx] - next_mc[step_idx])
            continue
        step_target[step_idx] = step_reward[step_idx] + gamma * ((1.0 - lam) * next_value[step_idx] + lam * step_target[step_idx + 1])
        direct_bias[step_idx] = gamma * (1.0 - lam) * (next_value[step_idx] - next_mc[step_idx]) + gamma * lam * direct_bias[step_idx + 1]
        tail_bias[step_idx] = gamma * lam * tail_bias[step_idx + 1]

    step_bias = step_target - mc_step
    closure = step_bias - direct_bias - tail_bias
    return {
        "step_target": step_target,
        "mc_step": mc_step,
        "step_bias": step_bias,
        "next_value": next_value,
        "next_mc": next_mc,
        "next_value_error": next_value - next_mc,
        "direct_bias_component": direct_bias,
        "tail_bias_component": tail_bias,
        "closure": closure,
        "step_reward": step_reward,
        "terminal_step": terminal_step.astype(np.float64),
    }


def diagnose_run(
    run_dir: Path,
    *,
    updates: Iterable[int],
    episodes: int,
    episode_seed_base: int,
    deterministic: bool,
    rollout_env_steps: int,
    device: torch.device,
) -> Dict[str, object]:
    cfg_path = run_dir / "config_source.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config_source.yaml in {run_dir}")
    cfg = load_config(str(cfg_path))
    results: Dict[str, object] = {
        "run_dir": str(run_dir),
        "episodes": int(episodes),
        "episode_seed_base": int(episode_seed_base),
        "deterministic": bool(deterministic),
        "rollout_env_steps": int(rollout_env_steps),
        "updates": {},
    }
    for update in updates:
        bundle = _load_bundle(cfg, run_dir, int(update), device)
        next_value_all: List[float] = []
        next_mc_all: List[float] = []
        next_err_all: List[float] = []
        step_bias_all: List[float] = []
        direct_bias_all: List[float] = []
        tail_bias_all: List[float] = []
        closure_all: List[float] = []
        reward_all: List[float] = []
        terminal_all: List[float] = []
        episode_rows: List[Dict[str, float]] = []
        for ep_idx in range(int(episodes)):
            seed = int(episode_seed_base) + ep_idx
            diag = _episode_bias_diagnostics(
                cfg,
                bundle.actor,
                bundle.critic,
                device=device,
                deterministic=deterministic,
                seed=seed,
                rollout_env_steps=int(rollout_env_steps),
            )
            next_value_all.extend(diag["next_value"].tolist())
            next_mc_all.extend(diag["next_mc"].tolist())
            next_err_all.extend(diag["next_value_error"].tolist())
            step_bias_all.extend(diag["step_bias"].tolist())
            direct_bias_all.extend(diag["direct_bias_component"].tolist())
            tail_bias_all.extend(diag["tail_bias_component"].tolist())
            closure_all.extend(diag["closure"].tolist())
            reward_all.extend(diag["step_reward"].tolist())
            terminal_all.extend(diag["terminal_step"].tolist())
            episode_rows.append(
                {
                    "seed": float(seed),
                    "step_bias_mean": float(np.mean(diag["step_bias"])) if diag["step_bias"].size else 0.0,
                    "direct_bias_mean": float(np.mean(diag["direct_bias_component"])) if diag["direct_bias_component"].size else 0.0,
                    "tail_bias_mean": float(np.mean(diag["tail_bias_component"])) if diag["tail_bias_component"].size else 0.0,
                }
            )
        next_value_arr = np.asarray(next_value_all, dtype=np.float64)
        next_mc_arr = np.asarray(next_mc_all, dtype=np.float64)
        next_err_arr = np.asarray(next_err_all, dtype=np.float64)
        step_bias_arr = np.asarray(step_bias_all, dtype=np.float64)
        direct_bias_arr = np.asarray(direct_bias_all, dtype=np.float64)
        tail_bias_arr = np.asarray(tail_bias_all, dtype=np.float64)
        closure_arr = np.asarray(closure_all, dtype=np.float64)
        reward_arr = np.asarray(reward_all, dtype=np.float64)
        terminal_arr = np.asarray(terminal_all, dtype=np.float64)
        results["updates"][f"u{int(update):04d}"] = {
            "episode_rows": episode_rows,
            "next_value_vs_next_mc": _pair_summary(next_value_arr, next_mc_arr),
            "next_value_error_vs_step_bias": _pair_summary(next_err_arr, step_bias_arr),
            "step_bias_decomposition": {
                "step_bias": _component_summary(step_bias_arr, step_bias_arr),
                "direct_next_value_bias_component": _component_summary(direct_bias_arr, step_bias_arr),
                "tail_bootstrap_bias_component": _component_summary(tail_bias_arr, step_bias_arr),
                "closure_residual_max_abs": float(np.max(np.abs(closure_arr))) if closure_arr.size else 0.0,
            },
            "step_reward_mean": float(np.mean(reward_arr)) if reward_arr.size else 0.0,
            "terminal_step_fraction": float(np.mean(terminal_arr)) if terminal_arr.size else 0.0,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--updates", type=int, nargs="+", default=[100, 150, 200])
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--episode_seed_base", type=int, default=42000)
    parser.add_argument("--policy_mode", choices=["deterministic", "stochastic"], default="stochastic")
    parser.add_argument("--rollout_env_steps", type=int, default=50)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_path = Path(args.out) if args.out else run_dir / "structured_step_bootstrap_bias_diag.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = diagnose_run(
        run_dir,
        updates=args.updates,
        episodes=int(args.episodes),
        episode_seed_base=int(args.episode_seed_base),
        deterministic=(args.policy_mode == "deterministic"),
        rollout_env_steps=int(args.rollout_env_steps),
        device=device,
    )
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote structured step-bootstrap bias summary to {out_path}")
    for update_key, update_summary in summary["updates"].items():
        nv = update_summary["next_value_vs_next_mc"]
        bias = update_summary["step_bias_decomposition"]
        print(f"[{update_key}]")
        print(
            "  "
            f"next_value_vs_next_mc_rho={nv['pearson']:.3f} "
            f"next_gap={nv['gap_mean']:.3f} "
            f"direct_bias_share={bias['direct_next_value_bias_component']['share_of_abs_total']:.3f} "
            f"tail_bias_share={bias['tail_bootstrap_bias_component']['share_of_abs_total']:.3f}"
        )


if __name__ == "__main__":
    main()
