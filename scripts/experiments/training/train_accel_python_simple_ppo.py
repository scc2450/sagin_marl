from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import sys

import numpy as np
import torch

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.baselines import queue_aware_bw_policy, queue_aware_sat_policy
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import _current_obs_list, _refresh_stage_obs_cache, _sat_mask_to_ids
from sagin_marl.rl.structured_stage_builders import world_state_to_torch


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _cat_dataclass(items: list[Any]) -> Any:
    if not items:
        raise ValueError("cannot concatenate an empty dataclass list")
    kwargs: dict[str, torch.Tensor] = {}
    for field in fields(items[0]):
        values = []
        for item in items:
            value = getattr(item, field.name)
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            values.append(value)
        kwargs[field.name] = torch.cat(values, dim=0)
    return type(items[0])(**kwargs)


def _index_dataclass(batch: Any, indices: torch.Tensor, *, device: torch.device | None = None) -> Any:
    kwargs: dict[str, torch.Tensor] = {}
    for field in fields(batch):
        value = getattr(batch, field.name).index_select(0, indices.to(getattr(batch, field.name).device))
        if device is not None:
            value = value.to(device)
        kwargs[field.name] = value
    return type(batch)(**kwargs)


def _to_device_dataclass(batch: Any, device: torch.device) -> Any:
    kwargs = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        kwargs[field.name] = value.to(device)
    return type(batch)(**kwargs)


def _detach_cpu_dataclass(batch: Any) -> Any:
    kwargs = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        kwargs[field.name] = value.detach().cpu().clone()
    return type(batch)(**kwargs)


def _explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(true)
    if int(mask.sum()) <= 1:
        return 0.0
    pred = pred[mask]
    true = true[mask]
    var_y = float(np.var(true))
    if var_y <= 1e-12:
        return 0.0
    return float(1.0 - np.var(true - pred) / var_y)


def _safe_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else 0.0


def _reset_env(env: SaginParallelEnv, seed: int) -> StructuredControlDriver:
    env.reset(seed=int(seed))
    return StructuredControlDriver(env)


def _run_fixed_sat_bw_step(
    driver: StructuredControlDriver,
    accel_action: np.ndarray,
    cfg: Any,
) -> tuple[float, bool, dict[str, float], torch.Tensor, torch.Tensor]:
    driver.run_accel_stage(accel_action)
    _refresh_stage_obs_cache(driver)
    obs_after_accel = _current_obs_list(driver)
    sat_mask = np.asarray(queue_aware_sat_policy(obs_after_accel, cfg), dtype=np.float32)
    sat_action = _sat_mask_to_ids(driver, sat_mask)
    driver.run_sat_stage(sat_action)
    bw_action = np.asarray(queue_aware_bw_policy(obs_after_accel, cfg), dtype=np.float32)
    result, _next_world = driver.execute_stage_bw_and_prepare_next_accel(
        bw_action,
        capture_auxiliary_outputs=True,
        materialize_agent_dicts=False,
    )
    reward = float(result.team_reward)
    done = bool(result.terminated or result.truncated)
    parts = {str(k): float(v) for k, v in dict(result.reward_parts or {}).items() if isinstance(v, (int, float, np.floating))}
    target = result.danger_imitation_target
    mask = result.danger_imitation_mask
    if target is None:
        target_t = torch.zeros((int(cfg.num_uav), 2), dtype=torch.float32)
    else:
        target_t = torch.as_tensor(target, dtype=torch.float32).reshape(int(cfg.num_uav), 2)
    if mask is None:
        mask_t = torch.zeros((int(cfg.num_uav), 2), dtype=torch.float32)
    else:
        mask_t = torch.as_tensor(mask, dtype=torch.float32).reshape(int(cfg.num_uav), 2)
    return reward, done, parts, target_t, mask_t


def _compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    steps, envs = rewards.shape
    advantages = np.zeros((steps, envs), dtype=np.float32)
    last_adv = np.zeros((envs,), dtype=np.float32)
    for t in range(steps - 1, -1, -1):
        if t == steps - 1:
            next_values = np.zeros((envs,), dtype=np.float32)
        else:
            next_values = values[t + 1]
        next_nonterminal = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + float(gamma) * next_values * next_nonterminal - values[t]
        last_adv = delta + float(gamma) * float(gae_lambda) * next_nonterminal * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages.reshape(-1), returns.reshape(-1)


def _collect_rollout(
    *,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    envs: list[SaginParallelEnv],
    drivers: list[StructuredControlDriver],
    cfg: Any,
    device: torch.device,
    rollout_env_steps: int,
    seed_base: int,
    episode_counters: list[int],
) -> tuple[dict[str, Any], list[StructuredControlDriver]]:
    num_envs = len(envs)
    num_uav = int(cfg.num_uav)
    local_batches = []
    world_batches = []
    action_batches = []
    logprob_batches = []
    value_batches = []
    reward_batches = []
    done_batches = []
    danger_target_batches = []
    danger_mask_batches = []
    ep_returns: list[float] = []
    ep_lengths: list[int] = []
    live_ep_returns = getattr(_collect_rollout, "_live_ep_returns", [0.0 for _ in range(num_envs)])
    live_ep_lengths = getattr(_collect_rollout, "_live_ep_lengths", [0 for _ in range(num_envs)])
    reward_part_sums: dict[str, list[float]] = {}
    entropy_values: list[float] = []
    std_values: list[float] = []

    actor.eval()
    critic.eval()
    with torch.no_grad():
        for _step in range(int(rollout_env_steps)):
            worlds = [world_state_to_torch(driver.begin_step()) for driver in drivers]
            locals_flat = []
            for driver in drivers:
                locals_flat.extend(driver.build_local_accel_states())
            world_batch_cpu = _cat_dataclass(worlds)
            local_batch_cpu = _cat_dataclass(locals_flat)
            local_batch_dev = _to_device_dataclass(local_batch_cpu, device)
            world_batch_dev = _to_device_dataclass(world_batch_cpu, device)
            out = actor.act_accel(local_batch_dev, deterministic=False)
            values = critic.value_accel(world_batch_dev).reshape(num_envs)
            action_env = out.action.reshape(num_envs, num_uav, 2).detach().cpu()
            logprob_env = out.logprob.reshape(num_envs, num_uav).sum(dim=1).detach().cpu()
            entropy_values.extend(out.entropy.detach().cpu().reshape(-1).tolist())
            std_values.extend(out.std.detach().cpu().reshape(-1).tolist())

            rewards = np.zeros((num_envs,), dtype=np.float32)
            dones = np.zeros((num_envs,), dtype=bool)
            danger_targets = torch.zeros((num_envs, num_uav, 2), dtype=torch.float32)
            danger_masks = torch.zeros((num_envs, num_uav, 2), dtype=torch.float32)
            for env_idx, driver in enumerate(list(drivers)):
                reward, done, parts, danger_target, danger_mask = _run_fixed_sat_bw_step(
                    driver,
                    action_env[env_idx].numpy().astype(np.float32),
                    cfg,
                )
                rewards[env_idx] = float(reward)
                dones[env_idx] = bool(done)
                danger_targets[env_idx] = danger_target
                danger_masks[env_idx] = danger_mask
                for key, value in parts.items():
                    reward_part_sums.setdefault(key, []).append(float(value))
                live_ep_returns[env_idx] += float(reward)
                live_ep_lengths[env_idx] += 1
                if done:
                    ep_returns.append(float(live_ep_returns[env_idx]))
                    ep_lengths.append(int(live_ep_lengths[env_idx]))
                    live_ep_returns[env_idx] = 0.0
                    live_ep_lengths[env_idx] = 0
                    episode_counters[env_idx] += 1
                    reset_seed = int(seed_base) + 100_000 + episode_counters[env_idx] * max(num_envs, 1) + env_idx
                    drivers[env_idx] = _reset_env(envs[env_idx], reset_seed)

            local_batches.append(_detach_cpu_dataclass(local_batch_cpu))
            world_batches.append(_detach_cpu_dataclass(world_batch_cpu))
            action_batches.append(action_env.clone())
            logprob_batches.append(logprob_env.clone())
            value_batches.append(values.detach().cpu().clone())
            reward_batches.append(torch.as_tensor(rewards, dtype=torch.float32))
            done_batches.append(torch.as_tensor(dones, dtype=torch.bool))
            danger_target_batches.append(danger_targets)
            danger_mask_batches.append(danger_masks)

    _collect_rollout._live_ep_returns = live_ep_returns
    _collect_rollout._live_ep_lengths = live_ep_lengths
    rewards_np = torch.stack(reward_batches, dim=0).numpy()
    dones_np = torch.stack(done_batches, dim=0).numpy().astype(bool)
    values_np = torch.stack(value_batches, dim=0).numpy()
    advantages, returns = _compute_gae(
        rewards_np,
        dones_np,
        values_np,
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
    )
    payload = {
        "local": _cat_dataclass(local_batches),
        "world": _cat_dataclass(world_batches),
        "actions": torch.stack(action_batches, dim=0).reshape(-1, num_uav, 2),
        "old_logprobs": torch.stack(logprob_batches, dim=0).reshape(-1),
        "values": torch.stack(value_batches, dim=0).reshape(-1),
        "rewards": torch.stack(reward_batches, dim=0).reshape(-1),
        "dones": torch.stack(done_batches, dim=0).reshape(-1),
        "returns": torch.as_tensor(returns, dtype=torch.float32),
        "advantages": torch.as_tensor(advantages, dtype=torch.float32),
        "danger_targets": torch.stack(danger_target_batches, dim=0).reshape(-1, num_uav, 2),
        "danger_masks": torch.stack(danger_mask_batches, dim=0).reshape(-1, num_uav, 2),
        "episode_returns": ep_returns,
        "episode_lengths": ep_lengths,
        "reward_part_means": {key: _safe_mean(vals) for key, vals in reward_part_sums.items()},
        "entropy_mean": _safe_mean(entropy_values),
        "std_mean": _safe_mean(std_values),
    }
    return payload, drivers


def _ppo_update(
    *,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    rollout: dict[str, Any],
    cfg: Any,
    device: torch.device,
) -> dict[str, float]:
    actor.train()
    critic.train()
    num_uav = int(cfg.num_uav)
    transition_count = int(rollout["actions"].shape[0])
    if transition_count <= 0:
        return {}
    advantages = rollout["advantages"].clone()
    if bool(getattr(cfg, "actor_advantage_normalize_enabled", True)) and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    returns = rollout["returns"]
    old_logprobs = rollout["old_logprobs"]
    old_values = rollout["values"]
    clip_ratio = float(getattr(cfg, "clip_ratio", 0.2) or 0.2)
    value_coef = float(getattr(cfg, "value_coef", 0.5) or 0.5)
    entropy_coef = float(getattr(cfg, "entropy_coef", 0.0) or 0.0)
    danger_enabled = bool(getattr(cfg, "danger_imitation_enabled", False))
    danger_coef = float(getattr(cfg, "danger_imitation_coef", 0.0) or 0.0)
    max_grad_norm = float(getattr(cfg, "max_grad_norm", 0.5) or 0.5)
    ppo_epochs = max(int(getattr(cfg, "ppo_epochs", 1) or 1), 1)
    num_mini_batch = max(int(getattr(cfg, "num_mini_batch", 1) or 1), 1)
    batch_size = transition_count
    mini_batch_size = max(batch_size // num_mini_batch, 1)
    flat_actions = rollout["actions"].reshape(transition_count * num_uav, 2)
    flat_danger_targets = rollout["danger_targets"].reshape(transition_count * num_uav, 2)
    flat_danger_masks = rollout["danger_masks"].reshape(transition_count * num_uav, 2)

    losses: dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_frac": [],
        "danger_imitation_loss": [],
    }
    for _epoch in range(ppo_epochs):
        perm = torch.randperm(batch_size)
        for start in range(0, batch_size, mini_batch_size):
            mb_idx_cpu = perm[start : start + mini_batch_size]
            if mb_idx_cpu.numel() <= 0:
                continue
            mb_idx = mb_idx_cpu.to(device=device, dtype=torch.long)
            row_idx = (
                mb_idx_cpu.unsqueeze(1) * num_uav
                + torch.arange(num_uav, dtype=torch.long).unsqueeze(0)
            ).reshape(-1)
            local_mb = _index_dataclass(rollout["local"], row_idx, device=device)
            world_mb = _index_dataclass(rollout["world"], mb_idx_cpu, device=device)
            action_mb = flat_actions.index_select(0, row_idx).to(device)
            old_logp_mb = old_logprobs.index_select(0, mb_idx_cpu).to(device)
            adv_mb = advantages.index_select(0, mb_idx_cpu).to(device)
            ret_mb = returns.index_select(0, mb_idx_cpu).to(device)

            value_pred = critic.value_accel(world_mb).reshape(-1)
            value_loss = 0.5 * (value_pred - ret_mb).pow(2).mean()
            critic_optimizer.zero_grad(set_to_none=True)
            (value_coef * value_loss).backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optimizer.step()

            actor_out = actor.evaluate_accel(local_mb, action_mb)
            new_logp = actor_out.logprob.reshape(-1, num_uav).sum(dim=1)
            entropy = actor_out.entropy.reshape(-1, num_uav).sum(dim=1).mean()
            ratio = torch.exp(new_logp - old_logp_mb)
            unclipped = ratio * adv_mb
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_mb
            policy_loss = -torch.min(unclipped, clipped).mean()
            danger_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            if danger_enabled and danger_coef > 0.0:
                target = flat_danger_targets.index_select(0, row_idx).to(device)
                mask = flat_danger_masks.index_select(0, row_idx).to(device)
                active = torch.sum(mask, dim=-1) > 0.0
                if torch.any(active):
                    pred = torch.tanh(actor_out.mean).reshape_as(target)
                    diff = (pred - target) * mask
                    denom = torch.sum(mask, dim=-1).clamp_min(1e-8)
                    danger_loss = (diff.pow(2).sum(dim=-1) / denom)[active].mean()
            actor_loss = policy_loss - entropy_coef * entropy + danger_coef * danger_loss
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.accel_policy.parameters(), max_grad_norm)
            actor_optimizer.step()

            with torch.no_grad():
                log_ratio = new_logp - old_logp_mb
                approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
                clip_frac = ((ratio - 1.0).abs() > clip_ratio).float().mean()
            losses["policy_loss"].append(float(policy_loss.detach().cpu()))
            losses["value_loss"].append(float(value_loss.detach().cpu()))
            losses["entropy"].append(float(entropy.detach().cpu()))
            losses["approx_kl"].append(float(approx_kl.detach().cpu()))
            losses["clip_frac"].append(float(clip_frac.detach().cpu()))
            losses["danger_imitation_loss"].append(float(danger_loss.detach().cpu()))

    with torch.no_grad():
        value_pred_all = []
        for start in range(0, transition_count, 1024):
            idx = torch.arange(start, min(start + 1024, transition_count), dtype=torch.long)
            world_mb = _index_dataclass(rollout["world"], idx, device=device)
            value_pred_all.append(critic.value_accel(world_mb).detach().cpu())
        value_pred_np = torch.cat(value_pred_all, dim=0).numpy()
    returns_np = returns.numpy()
    old_values_np = old_values.numpy()
    return {
        "policy_loss": _safe_mean(losses["policy_loss"]),
        "value_loss": _safe_mean(losses["value_loss"]),
        "entropy": _safe_mean(losses["entropy"]),
        "approx_kl": _safe_mean(losses["approx_kl"]),
        "clip_frac": _safe_mean(losses["clip_frac"]),
        "danger_imitation_loss": _safe_mean(losses["danger_imitation_loss"]),
        "explained_variance_before": _explained_variance(old_values_np, returns_np),
        "explained_variance_after": _explained_variance(value_pred_np, returns_np),
    }


@torch.no_grad()
def _eval_policy(
    *,
    actor: torch.nn.Module,
    cfg: Any,
    device: torch.device,
    seed: int,
    episodes: int,
) -> dict[str, float]:
    if episodes <= 0:
        return {}
    returns: list[float] = []
    lengths: list[int] = []
    for ep in range(int(episodes)):
        env = SaginParallelEnv(copy.deepcopy(cfg))
        driver = _reset_env(env, int(seed) + ep)
        total = 0.0
        length = 0
        for _ in range(int(cfg.T_steps)):
            world = driver.begin_step()
            del world
            local = _cat_dataclass(driver.build_local_accel_states())
            out = actor.act_accel(_to_device_dataclass(local, device), deterministic=True)
            action = out.action.reshape(int(cfg.num_uav), 2).detach().cpu().numpy().astype(np.float32)
            reward, done, _parts, _danger_target, _danger_mask = _run_fixed_sat_bw_step(driver, action, cfg)
            total += float(reward)
            length += 1
            if done:
                break
        returns.append(total)
        lengths.append(length)
    return {
        "eval_return_mean": _safe_mean(returns),
        "eval_length_mean": _safe_mean(lengths),
        "eval_reward_mean": _safe_mean(np.asarray(returns) / np.maximum(np.asarray(lengths), 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Python StructuredControlDriver simple PPO for accel-only diagnostics.")
    parser.add_argument("--config", default="configs/tmp/accel_actor_capacity_c4critic/structured_accel_3uav20gu_A0_baseline.yaml")
    parser.add_argument("--run_dir", default="runs/diagnostics/python_simple_accel_ppo")
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--rollout_env_steps", type=int, default=250)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actor_lr", type=float, default=None)
    parser.add_argument("--critic_lr", type=float, default=None)
    parser.add_argument("--eval_episodes", type=int, default=0)
    args = parser.parse_args()

    _set_seed(int(args.seed))
    cfg = load_config(str(args.config))
    if args.actor_lr is not None:
        cfg.actor_lr = float(args.actor_lr)
    if args.critic_lr is not None:
        cfg.critic_lr = float(args.critic_lr)
    cfg.train_accel = True
    cfg.train_sat = False
    cfg.train_bw = False
    cfg.exec_sat_source = "queue_aware"
    cfg.exec_bw_source = "queue_aware"
    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_structured_modules_from_config(cfg)
    actor = bundle.actor.to(device)
    critic = bundle.critic.to(device)
    if critic is None:
        raise RuntimeError("critic is required for simple PPO")
    actor_optimizer = torch.optim.Adam(actor.accel_policy.parameters(), lr=float(cfg.actor_lr))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(cfg.critic_lr))
    envs = [SaginParallelEnv(copy.deepcopy(cfg)) for _ in range(int(args.num_envs))]
    episode_counters = [0 for _ in envs]
    drivers = [_reset_env(env, int(args.seed) + i) for i, env in enumerate(envs)]
    print(
        "PythonSimpleAccelPPO "
        f"config={args.config} reward_mode={getattr(cfg, 'reward_mode', '')} "
        f"envs={len(envs)} rollout={int(args.rollout_env_steps)} device={device} "
        f"actor_lr={float(cfg.actor_lr):.3g} critic_lr={float(cfg.critic_lr):.3g}"
    )
    for update in range(1, int(args.updates) + 1):
        t0 = time.perf_counter()
        rollout, drivers = _collect_rollout(
            actor=actor,
            critic=critic,
            envs=envs,
            drivers=drivers,
            cfg=cfg,
            device=device,
            rollout_env_steps=int(args.rollout_env_steps),
            seed_base=int(args.seed),
            episode_counters=episode_counters,
        )
        collect_sec = time.perf_counter() - t0
        metrics = _ppo_update(
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            rollout=rollout,
            cfg=cfg,
            device=device,
        )
        eval_metrics = _eval_policy(
            actor=actor,
            cfg=cfg,
            device=device,
            seed=int(args.seed) + 10_000 + update * 100,
            episodes=int(args.eval_episodes),
        )
        reward_mean = float(rollout["rewards"].float().mean().item())
        reward_std = float(rollout["rewards"].float().std(unbiased=False).item())
        done_rate = float(rollout["dones"].float().mean().item())
        ep_len_mean = _safe_mean(rollout["episode_lengths"])
        ep_ret_mean = _safe_mean(rollout["episode_returns"])
        parts = rollout["reward_part_means"]
        x_acc = float(parts.get("x_acc", 0.0))
        x_rel = float(parts.get("x_rel", 0.0))
        processed_ratio = float(parts.get("processed_ratio", 0.0))
        print(
            f"u={update:03d} "
            f"r={reward_mean:.6f}±{reward_std:.6f} "
            f"ep_ret={ep_ret_mean:.3f} ep_len={ep_len_mean:.1f} done={done_rate:.3f} "
            f"EV={metrics.get('explained_variance_after', 0.0):.3f} "
            f"KL={metrics.get('approx_kl', 0.0):.5f} clip={metrics.get('clip_frac', 0.0):.3f} "
            f"ent={metrics.get('entropy', 0.0):.3f} std={rollout['std_mean']:.3f} "
            f"x_acc={x_acc:.3f} x_rel={x_rel:.3f} proc={processed_ratio:.3f} "
            f"collect={collect_sec:.1f}s"
        )
        if eval_metrics:
            print(
                f"  eval r={eval_metrics['eval_reward_mean']:.6f} "
                f"ret={eval_metrics['eval_return_mean']:.3f} len={eval_metrics['eval_length_mean']:.1f}"
            )
        with (run_dir / "last_metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "update": int(update),
                        "reward_mean": float(reward_mean),
                        "episode_return_mean": float(ep_ret_mean),
                        "episode_length_mean": float(ep_len_mean),
                        "done_rate": float(done_rate),
                        "explained_variance_after": float(metrics.get("explained_variance_after", 0.0)),
                        "approx_kl": float(metrics.get("approx_kl", 0.0)),
                        "clip_frac": float(metrics.get("clip_frac", 0.0)),
                        "entropy": float(metrics.get("entropy", 0.0)),
                        "x_acc": float(x_acc),
                        "x_rel": float(x_rel),
                        "processed_ratio": float(processed_ratio),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
