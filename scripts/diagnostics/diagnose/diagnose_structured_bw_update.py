from __future__ import annotations

import argparse
import copy
import json
import os
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
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO, _index_dataclass
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _safe_corr(x: list[float], y: list[float]) -> float:
    if len(x) <= 1 or len(y) <= 1 or len(x) != len(y):
        return 0.0
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if float(np.std(xa)) <= 1.0e-12 or float(np.std(ya)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _bucket_mean(counts: list[int], values: list[float]) -> list[dict[str, float]]:
    if len(counts) != len(values):
        raise ValueError("counts and values must have the same length")
    buckets: dict[int, list[float]] = {}
    for count, value in zip(counts, values):
        buckets.setdefault(int(count), []).append(float(value))
    rows: list[dict[str, float]] = []
    for count in sorted(buckets):
        values_np = np.asarray(buckets[count], dtype=np.float64)
        rows.append(
            {
                "count": float(count),
                "n": float(values_np.size),
                "mean": float(np.mean(values_np)),
            }
        )
    return rows


def _latent_mask(user_mask: torch.Tensor, bw_valid_mask: torch.Tensor) -> torch.Tensor:
    valid = (user_mask > 0.5) & (bw_valid_mask > 0.5)
    if valid.shape[-1] == 0:
        return valid
    latent = valid.clone()
    valid_counts = valid.sum(dim=-1)
    ref_idx = torch.where(
        valid,
        torch.arange(valid.shape[-1], device=valid.device, dtype=torch.long).view(1, -1).expand_as(valid),
        torch.full_like(valid, -1, dtype=torch.long),
    ).amax(dim=-1)
    active = torch.nonzero(valid_counts > 1, as_tuple=False).flatten()
    if active.numel() > 0:
        latent[active, ref_idx[active]] = False
    return latent


def _load_models(cfg, run_dir: Path, update: int, device: torch.device):
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    actor_ckpt = run_dir / f"actor_u{int(update):04d}.pt"
    critic_ckpt = run_dir / f"critic_u{int(update):04d}.pt"
    load_checkpoint_forgiving(bundle.actor, str(actor_ckpt), map_location=device, strict=True)
    load_checkpoint_forgiving(bundle.critic, str(critic_ckpt), map_location=device, strict=True)
    actor_current = bundle.actor.to(device)
    actor_old = copy.deepcopy(bundle.actor).to(device).eval()
    critic = bundle.critic.to(device).eval()
    return actor_current, actor_old, critic


def _collect_rollout(
    cfg,
    actor,
    critic,
    *,
    env_steps: int,
    seed_base: int,
    deterministic: bool,
    device: torch.device,
    target_mode: str,
) -> StructuredRolloutBuffer:
    env = make_structured_env(cfg, mode="script")
    try:
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
            target_mode=target_mode,
        )
        buffer = StructuredRolloutBuffer()
        env.reset(seed=int(seed_base))
        driver = as_structured_driver(env)
        total_env_steps = 0
        episode_idx = 0
        while total_env_steps < int(env_steps):
            step_result = algo.collect_env_step(driver, buffer, deterministic=deterministic)
            total_env_steps += 1
            terminated = bool(next(iter(step_result.terminations.values())))
            truncated = bool(next(iter(step_result.truncations.values())))
            if terminated or truncated:
                episode_idx += 1
                env.reset(seed=int(seed_base) + episode_idx)
                driver = as_structured_driver(env)
        return buffer
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def _build_bw_cache(algo: StructuredMAPPO, buffer: StructuredRolloutBuffer) -> dict[str, Any] | None:
    rollout_views = buffer.build_rollout_views(algo.device)
    training_batch = rollout_views.training_view.stage_batches.get(2)
    if training_batch is None or int(training_batch.num_samples) <= 0:
        return None
    advantages = torch.from_numpy(
        algo.compute_returns_and_advantages(
            buffer,
            rollout_views.bootstrap_view,
            return_view=rollout_views.return_view,
        )["advantages"]
    )
    advantages = advantages.to(algo.device)
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1.0e-8)
    idx = torch.as_tensor(
        np.asarray(training_batch.transition_indices, dtype=np.int64),
        device=algo.device,
        dtype=torch.long,
    )
    num_samples = int(training_batch.num_samples)
    num_agents = int(training_batch.num_agents)
    local_batch = training_batch.local_batch
    joint_actions = training_batch.actions.to(algo.device)
    flat_indices = torch.arange(num_samples * num_agents, device=algo.device, dtype=torch.long).reshape(num_samples, num_agents)
    old_logprobs = training_batch.old_logprobs.to(algo.device)
    bw_adv = advantages.index_select(0, idx)
    return {
        "local_batch": local_batch,
        "joint_actions": joint_actions,
        "flat_indices": flat_indices,
        "old_logprobs": old_logprobs,
        "advantages": bw_adv,
        "num_samples": num_samples,
        "num_agents": num_agents,
    }


def diagnose_update(
    run_dir: Path,
    update: int,
    *,
    effective_num_envs: int,
    rollout_env_steps: int | None,
    seed_base: int,
    deterministic: bool,
    target_mode: str,
    device: torch.device,
) -> dict[str, Any]:
    cfg = load_config(str(run_dir / "config_source.yaml"))
    actor, actor_old, critic = _load_models(cfg, run_dir, int(update), device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(getattr(cfg, "actor_lr", 1.0e-4) or 1.0e-4))
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
        actor_optimizer=actor_optimizer,
        critic_optimizer=None,
        device=device,
        target_mode=target_mode,
    )
    rollout_steps = int(rollout_env_steps if rollout_env_steps is not None else getattr(cfg, "buffer_size", 50))
    env_steps_total = rollout_steps * int(effective_num_envs)
    buffer = _collect_rollout(
        cfg,
        actor_old,
        critic,
        env_steps=env_steps_total,
        seed_base=seed_base,
        deterministic=deterministic,
        device=device,
        target_mode=target_mode,
    )
    cache = _build_bw_cache(algo, buffer)
    if cache is None:
        return {
            "update": int(update),
            "env_steps_total": int(env_steps_total),
            "bw_stage_samples": 0,
        }

    bw_valid_counts: list[int] = []
    bw_latent_counts: list[int] = []
    bw_entropy_raw: list[float] = []
    bw_entropy_objective: list[float] = []
    bw_entropy_per_dim_raw: list[float] = []
    bw_log_scale_latent_mean: list[float] = []
    bw_abs_log_ratio_agent: list[float] = []
    bw_abs_log_ratio_agent_raw: list[float] = []
    bw_top1_alloc: list[float] = []

    joint_abs_log_ratio: list[float] = []
    joint_abs_log_ratio_raw: list[float] = []
    joint_clip_indicator: list[float] = []
    joint_approx_kl_proxy: list[float] = []
    joint_mean_valid_count: list[float] = []
    joint_max_valid_count: list[float] = []

    epoch_rows: list[dict[str, float]] = []
    minibatch_counter = 0

    for epoch_idx in range(algo.ppo_epochs):
        rel_idx = np.arange(int(cache["num_samples"]), dtype=np.int64)
        np.random.shuffle(rel_idx)
        minibatch_size = max(1, int(np.ceil(rel_idx.size / max(algo.num_mini_batch, 1))))
        minibatches = [
            torch.as_tensor(rel_idx[start : start + minibatch_size], device=algo.device, dtype=torch.long)
            for start in range(0, rel_idx.size, minibatch_size)
        ]
        epoch_joint_abs_log_ratio: list[float] = []
        epoch_joint_clip: list[float] = []
        epoch_joint_approx_kl: list[float] = []
        for mb_rel in minibatches:
            minibatch_counter += 1
            flat_mb = cache["flat_indices"].index_select(0, mb_rel).reshape(-1)
            local_batch_mb = _index_dataclass(cache["local_batch"], flat_mb)
            joint_actions_mb = cache["joint_actions"].index_select(0, mb_rel)
            num_samples = int(joint_actions_mb.shape[0])
            num_agents = int(cache["num_agents"])

            new_joint_logprob, joint_entropy, actor_out = algo._stage_actor_eval_from_batch(
                2,
                local_batch_mb,
                joint_actions_mb,
                num_agents,
            )
            flat_action = joint_actions_mb.to(algo.device).reshape(num_samples * num_agents, -1)
            with torch.no_grad():
                old_actor_out = actor_old.evaluate_bw(local_batch_mb, flat_action)

            new_logprob_agent = actor_out.logprob.reshape(num_samples, num_agents)
            old_logprob_agent = old_actor_out.logprob.reshape(num_samples, num_agents)
            log_ratio_agent = new_logprob_agent - old_logprob_agent
            abs_log_ratio_agent = log_ratio_agent.abs()

            new_logprob_agent_raw = (
                actor_out.logprob_raw.reshape(num_samples, num_agents)
                if actor_out.logprob_raw is not None
                else new_logprob_agent
            )
            old_logprob_agent_raw = (
                old_actor_out.logprob_raw.reshape(num_samples, num_agents)
                if old_actor_out.logprob_raw is not None
                else old_logprob_agent
            )
            log_ratio_agent_raw = new_logprob_agent_raw - old_logprob_agent_raw
            abs_log_ratio_agent_raw = log_ratio_agent_raw.abs()

            joint_log_ratio = new_joint_logprob - cache["old_logprobs"].index_select(0, mb_rel)
            ratio = torch.exp(joint_log_ratio)
            approx_kl = -joint_log_ratio
            clip_lower, clip_upper = algo._stage_ratio_clip_bounds(
                2,
                actor_out,
                num_samples=num_samples,
                num_agents=num_agents,
                dtype=ratio.dtype,
                device=ratio.device,
            )
            clip_indicator = algo._clip_indicator_with_bounds(ratio, clip_lower, clip_upper).to(torch.float32)
            joint_log_ratio_raw = log_ratio_agent_raw.sum(dim=-1)

            valid_mask = (local_batch_mb.user_mask > 0.5) & (local_batch_mb.bw_valid_mask > 0.5)
            latent_mask = _latent_mask(local_batch_mb.user_mask, local_batch_mb.bw_valid_mask)
            valid_count = valid_mask.sum(dim=-1).reshape(num_samples, num_agents)
            latent_count = latent_mask.sum(dim=-1).reshape(num_samples, num_agents)
            entropy_agent = actor_out.entropy.reshape(num_samples, num_agents)
            entropy_agent_raw = (
                actor_out.entropy_raw.reshape(num_samples, num_agents)
                if actor_out.entropy_raw is not None
                else entropy_agent
            )
            log_scale = actor_out.log_scale.reshape(num_samples, num_agents, -1)
            action_agent = actor_out.action.reshape(num_samples, num_agents, -1)
            valid_mask_view = valid_mask.reshape(num_samples, num_agents, -1)
            latent_mask_view = latent_mask.reshape(num_samples, num_agents, -1)

            denom = latent_count.to(entropy_agent.dtype).clamp_min(1.0)
            entropy_per_dim_raw = torch.where(
                latent_count > 0,
                entropy_agent_raw / denom,
                torch.zeros_like(entropy_agent_raw),
            )
            log_scale_latent_mean = torch.zeros_like(entropy_agent)
            if torch.any(latent_mask_view):
                masked_sum = (log_scale * latent_mask_view.to(log_scale.dtype)).sum(dim=-1)
                log_scale_latent_mean = torch.where(latent_count > 0, masked_sum / denom, torch.zeros_like(masked_sum))
            top1_alloc = torch.where(
                valid_count > 0,
                (action_agent * valid_mask_view.to(action_agent.dtype)).amax(dim=-1),
                torch.zeros_like(entropy_agent),
            )

            bw_valid_counts.extend([int(x) for x in valid_count.reshape(-1).detach().cpu().tolist()])
            bw_latent_counts.extend([int(x) for x in latent_count.reshape(-1).detach().cpu().tolist()])
            bw_entropy_raw.extend([float(x) for x in entropy_agent_raw.reshape(-1).detach().cpu().tolist()])
            bw_entropy_objective.extend([float(x) for x in entropy_agent.reshape(-1).detach().cpu().tolist()])
            bw_entropy_per_dim_raw.extend([float(x) for x in entropy_per_dim_raw.reshape(-1).detach().cpu().tolist()])
            bw_log_scale_latent_mean.extend([float(x) for x in log_scale_latent_mean.reshape(-1).detach().cpu().tolist()])
            bw_abs_log_ratio_agent.extend([float(x) for x in abs_log_ratio_agent.reshape(-1).detach().cpu().tolist()])
            bw_abs_log_ratio_agent_raw.extend([float(x) for x in abs_log_ratio_agent_raw.reshape(-1).detach().cpu().tolist()])
            bw_top1_alloc.extend([float(x) for x in top1_alloc.reshape(-1).detach().cpu().tolist()])

            mean_valid_count = valid_count.to(torch.float32).mean(dim=-1)
            max_valid_count = valid_count.to(torch.float32).amax(dim=-1)
            joint_abs = joint_log_ratio.abs()
            joint_abs_log_ratio.extend([float(x) for x in joint_abs.detach().cpu().tolist()])
            joint_abs_log_ratio_raw.extend([float(x) for x in joint_log_ratio_raw.abs().detach().cpu().tolist()])
            joint_clip_indicator.extend([float(x) for x in clip_indicator.detach().cpu().tolist()])
            joint_approx_kl_proxy.extend([float(x) for x in approx_kl.detach().cpu().tolist()])
            joint_mean_valid_count.extend([float(x) for x in mean_valid_count.detach().cpu().tolist()])
            joint_max_valid_count.extend([float(x) for x in max_valid_count.detach().cpu().tolist()])

            epoch_joint_abs_log_ratio.extend([float(x) for x in joint_abs.detach().cpu().tolist()])
            epoch_joint_clip.extend([float(x) for x in clip_indicator.detach().cpu().tolist()])
            epoch_joint_approx_kl.extend([float(x) for x in approx_kl.detach().cpu().tolist()])

            adv = cache["advantages"].index_select(0, mb_rel)
            surr1 = ratio * adv
            surr2 = algo._clip_ratio_with_bounds(ratio, clip_lower, clip_upper) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_mean = joint_entropy.mean()
            loss = policy_loss - algo.entropy_coef * entropy_mean
            if loss.requires_grad:
                actor_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), algo.max_grad_norm)
                actor_optimizer.step()

        epoch_rows.append(
            {
                "epoch": float(epoch_idx),
                "joint_abs_log_ratio_mean": _safe_mean(epoch_joint_abs_log_ratio),
                "joint_clip_frac_mean": _safe_mean(epoch_joint_clip),
                "joint_approx_kl_mean": _safe_mean(epoch_joint_approx_kl),
            }
        )

    return {
        "update": int(update),
        "env_steps_total": int(env_steps_total),
        "rollout_env_steps": int(rollout_steps),
        "effective_num_envs": int(effective_num_envs),
        "bw_stage_samples": int(cache["num_samples"]),
        "num_agents": int(cache["num_agents"]),
        "actor_update_steps": int(minibatch_counter),
        "joint": {
            "abs_log_ratio": _summarize(joint_abs_log_ratio),
            "abs_log_ratio_raw": _summarize(joint_abs_log_ratio_raw),
            "approx_kl_proxy": _summarize(joint_approx_kl_proxy),
            "clip_indicator_mean": _safe_mean(joint_clip_indicator),
            "abs_log_ratio_vs_mean_valid_user_count_corr": _safe_corr(joint_mean_valid_count, joint_abs_log_ratio),
            "abs_log_ratio_vs_max_valid_user_count_corr": _safe_corr(joint_max_valid_count, joint_abs_log_ratio),
            "abs_log_ratio_raw_vs_mean_valid_user_count_corr": _safe_corr(joint_mean_valid_count, joint_abs_log_ratio_raw),
            "abs_log_ratio_raw_vs_max_valid_user_count_corr": _safe_corr(joint_max_valid_count, joint_abs_log_ratio_raw),
            "clip_indicator_vs_mean_valid_user_count_corr": _safe_corr(joint_mean_valid_count, joint_clip_indicator),
            "clip_indicator_vs_max_valid_user_count_corr": _safe_corr(joint_max_valid_count, joint_clip_indicator),
            "abs_log_ratio_by_max_valid_user_count": _bucket_mean([int(x) for x in joint_max_valid_count], joint_abs_log_ratio),
            "abs_log_ratio_raw_by_max_valid_user_count": _bucket_mean([int(x) for x in joint_max_valid_count], joint_abs_log_ratio_raw),
            "approx_kl_by_max_valid_user_count": _bucket_mean([int(x) for x in joint_max_valid_count], joint_approx_kl_proxy),
            "clip_indicator_by_max_valid_user_count": _bucket_mean([int(x) for x in joint_max_valid_count], joint_clip_indicator),
        },
        "per_agent": {
            "sample_count": int(len(bw_valid_counts)),
            "valid_user_count": _summarize([float(x) for x in bw_valid_counts]),
            "latent_dim_count": _summarize([float(x) for x in bw_latent_counts]),
            "entropy_raw": _summarize(bw_entropy_raw),
            "entropy_objective": _summarize(bw_entropy_objective),
            "entropy_per_dim_raw": _summarize(bw_entropy_per_dim_raw),
            "log_scale_latent_mean": _summarize(bw_log_scale_latent_mean),
            "abs_log_ratio": _summarize(bw_abs_log_ratio_agent),
            "abs_log_ratio_raw": _summarize(bw_abs_log_ratio_agent_raw),
            "top1_allocation": _summarize(bw_top1_alloc),
            "abs_log_ratio_vs_valid_user_count_corr": _safe_corr([float(x) for x in bw_valid_counts], bw_abs_log_ratio_agent),
            "abs_log_ratio_raw_vs_valid_user_count_corr": _safe_corr([float(x) for x in bw_valid_counts], bw_abs_log_ratio_agent_raw),
            "entropy_objective_vs_valid_user_count_corr": _safe_corr([float(x) for x in bw_valid_counts], bw_entropy_objective),
            "entropy_per_dim_raw_vs_valid_user_count_corr": _safe_corr([float(x) for x in bw_valid_counts], bw_entropy_per_dim_raw),
            "log_scale_vs_valid_user_count_corr": _safe_corr([float(x) for x in bw_valid_counts], bw_log_scale_latent_mean),
            "abs_log_ratio_vs_entropy_objective_corr": _safe_corr(bw_abs_log_ratio_agent, bw_entropy_objective),
            "abs_log_ratio_raw_vs_entropy_per_dim_raw_corr": _safe_corr(bw_abs_log_ratio_agent_raw, bw_entropy_per_dim_raw),
            "abs_log_ratio_vs_log_scale_corr": _safe_corr(bw_abs_log_ratio_agent, bw_log_scale_latent_mean),
            "top1_allocation_vs_valid_user_count_corr": _safe_corr([float(x) for x in bw_valid_counts], bw_top1_alloc),
            "abs_log_ratio_by_valid_user_count": _bucket_mean(bw_valid_counts, bw_abs_log_ratio_agent),
            "abs_log_ratio_raw_by_valid_user_count": _bucket_mean(bw_valid_counts, bw_abs_log_ratio_agent_raw),
            "entropy_objective_by_valid_user_count": _bucket_mean(bw_valid_counts, bw_entropy_objective),
            "entropy_per_dim_raw_by_valid_user_count": _bucket_mean(bw_valid_counts, bw_entropy_per_dim_raw),
            "log_scale_by_valid_user_count": _bucket_mean(bw_valid_counts, bw_log_scale_latent_mean),
            "top1_allocation_by_valid_user_count": _bucket_mean(bw_valid_counts, bw_top1_alloc),
        },
        "epochs": epoch_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--updates", type=int, nargs="+", default=[100, 150, 200])
    parser.add_argument("--effective-num-envs", type=int, default=12)
    parser.add_argument("--rollout-env-steps", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=62000)
    parser.add_argument("--policy-mode", choices=["deterministic", "stochastic"], default="stochastic")
    parser.add_argument("--target-mode", choices=["stage_chained", "step_level"], default="step_level")
    parser.add_argument("--out-name", type=str, default="structured_bw_update_diag.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "effective_num_envs": int(args.effective_num_envs),
        "rollout_env_steps": None if args.rollout_env_steps is None else int(args.rollout_env_steps),
        "seed_base": int(args.seed_base),
        "policy_mode": str(args.policy_mode),
        "target_mode": str(args.target_mode),
        "updates": {},
    }
    for update in args.updates:
        update_summary = diagnose_update(
            run_dir,
            int(update),
            effective_num_envs=int(args.effective_num_envs),
            rollout_env_steps=args.rollout_env_steps,
            seed_base=int(args.seed_base) + int(update) * 1000,
            deterministic=(args.policy_mode == "deterministic"),
            target_mode=str(args.target_mode),
            device=device,
        )
        summary["updates"][f"u{int(update):04d}"] = update_summary
        print(
            json.dumps(
                {
                    "update": int(update),
                    "bw_abs_log_ratio_vs_valid_count_corr": update_summary["per_agent"][
                        "abs_log_ratio_vs_valid_user_count_corr"
                    ],
                    "bw_abs_log_ratio_raw_vs_valid_count_corr": update_summary["per_agent"][
                        "abs_log_ratio_raw_vs_valid_user_count_corr"
                    ],
                    "bw_entropy_objective_vs_valid_count_corr": update_summary["per_agent"][
                        "entropy_objective_vs_valid_user_count_corr"
                    ],
                    "bw_entropy_per_dim_raw_vs_valid_count_corr": update_summary["per_agent"][
                        "entropy_per_dim_raw_vs_valid_user_count_corr"
                    ],
                    "bw_joint_clip_mean": update_summary["joint"]["clip_indicator_mean"],
                    "bw_abs_log_ratio_mean": update_summary["per_agent"]["abs_log_ratio"]["mean"],
                }
            )
        )
    out_path = run_dir / str(args.out_name)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
