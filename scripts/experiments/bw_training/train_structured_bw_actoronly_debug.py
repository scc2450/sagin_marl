import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_critic import ZeroStructuredCritic
from sagin_marl.rl.structured_eval import evaluate_structured_actor_exec_sources
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO
from sagin_marl.rl.structured_train import (
    _is_done,
    _looks_like_structured_driver_group,
    _normalize_env_group,
    _reset_env_at,
    close_structured_env_group,
    make_structured_env_group,
)
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.utils.progress import Progress
from sagin_marl.utils.seeding import set_seed
from sagin_marl.rl.structured_train import as_structured_driver


def _resolve_torch_device(device_arg: str) -> torch.device:
    requested = str(device_arg).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested via --device, but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def _collect_rollout(
    cfg,
    learner: StructuredMAPPO,
    *,
    num_envs: int,
    vec_backend: str,
    rollout_env_steps: int,
    reset_seed: int,
) -> tuple[StructuredRolloutBuffer, dict[str, float]]:
    env_group = make_structured_env_group(cfg, int(num_envs), backend=str(vec_backend))
    try:
        structured_group = env_group if _looks_like_structured_driver_group(env_group) else None
        if structured_group is not None:
            actual_num_envs = len(structured_group)
            structured_group.reset_many([int(reset_seed) + env_index for env_index in range(actual_num_envs)])
            drivers = structured_group
            envs = None
        else:
            envs = _normalize_env_group(env_group)
            actual_num_envs = len(envs)
            for env_index, env in enumerate(envs):
                _reset_env_at(env, int(reset_seed) + env_index)
            drivers = [
                as_structured_driver(env)
                for env in envs
            ]

        reset_counters = [0 for _ in range(actual_num_envs)]
        buffer = StructuredRolloutBuffer()
        rollout_reward_sum = 0.0
        rollout_bw_access_sum = 0.0
        rollout_steps_total = 0
        for _ in range(int(rollout_env_steps)):
            results = learner.collect_env_steps(drivers, buffer, deterministic=False)
            rollout_steps_total += len(results)
            for env_index, result in enumerate(results):
                rollout_reward_sum += float(next(iter(result.rewards.values())))
                rollout_bw_access_sum += float(getattr(result, "bw_access_reward", 0.0) or 0.0)
                if not _is_done(result):
                    continue
                reset_counters[env_index] += 1
                seed = int(reset_seed) + reset_counters[env_index] * actual_num_envs + env_index
                if structured_group is not None:
                    structured_group.reset_at(env_index, seed)
                else:
                    if envs is None:
                        raise RuntimeError("envs should be materialized for non-group drivers")
                    _reset_env_at(envs[env_index], seed)
        return buffer, {
            "rollout_reward_mean": float(rollout_reward_sum / max(rollout_steps_total, 1)),
            "rollout_bw_access_reward_mean": float(rollout_bw_access_sum / max(rollout_steps_total, 1)),
            "rollout_env_steps_total": float(rollout_steps_total),
        }
    finally:
        close_structured_env_group(env_group)


def _build_bw_stage_batch(buffer: StructuredRolloutBuffer, device: torch.device) -> dict[str, Any] | None:
    rollout_views = buffer.build_rollout_views(device)
    stage_batch = rollout_views.training_view.stage_batches.get(2)
    if stage_batch is None or int(stage_batch.num_samples) <= 0:
        return None
    support_mask_override = (
        stage_batch.bw_support_masks.to(device=device, dtype=torch.float32)
        if stage_batch.bw_support_masks is not None
        else None
    )
    return {
        "rollout_views": rollout_views,
        "stage_batch": stage_batch,
        "local_batch": stage_batch.local_batch,
        "joint_actions": stage_batch.actions.to(device=device, dtype=torch.float32),
        "num_samples": int(stage_batch.num_samples),
        "num_agents": int(stage_batch.num_agents),
        "support_mask_override": support_mask_override,
    }


def _actor_only_bw_update(
    actor,
    optimizer: torch.optim.Optimizer,
    learner: StructuredMAPPO,
    buffer: StructuredRolloutBuffer,
    *,
    entropy_coef: float,
    max_grad_norm: float,
    stagewise_adv_norm: bool,
    device: torch.device,
) -> dict[str, float]:
    stage_batch = _build_bw_stage_batch(buffer, device)
    if stage_batch is None:
        return {
            "policy_loss": 0.0,
            "entropy_mean": 0.0,
            "adv_mean": 0.0,
            "adv_std": 0.0,
            "grad_norm": 0.0,
            "bw_stage_samples": 0.0,
        }
    rollout_views = stage_batch["rollout_views"]
    training_batch = stage_batch["stage_batch"]
    gae = learner.compute_returns_and_advantages(
        buffer,
        rollout_views.bootstrap_view,
        return_view=rollout_views.return_view,
    )
    advantages_all = torch.from_numpy(gae["advantages"]).to(device=device, dtype=torch.float32)
    stage_idx = torch.as_tensor(
        np.asarray(training_batch.transition_indices, dtype=np.int64),
        device=device,
        dtype=torch.long,
    )
    advantages = advantages_all.index_select(0, stage_idx)
    if bool(stagewise_adv_norm) and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1.0e-8)

    actor.train()
    bw_out = actor.evaluate_bw(
        stage_batch["local_batch"],
        stage_batch["joint_actions"].reshape(stage_batch["num_samples"] * stage_batch["num_agents"], -1),
        support_mask_override=stage_batch["support_mask_override"],
    )
    joint_logprob = bw_out.logprob.reshape(stage_batch["num_samples"], stage_batch["num_agents"]).sum(dim=1)
    entropy_mean = bw_out.entropy.reshape(stage_batch["num_samples"], stage_batch["num_agents"]).mean()
    policy_loss = -(advantages.detach() * joint_logprob).mean()
    total_loss = policy_loss - float(entropy_coef) * entropy_mean

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    grad_sq = torch.zeros((), dtype=torch.float32, device=device)
    params = [param for param in actor.bw_policy.parameters() if param.grad is not None]
    for param in params:
        grad_sq = grad_sq + param.grad.detach().to(dtype=torch.float32).pow(2).sum()
    if max_grad_norm > 0.0 and params:
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    optimizer.step()
    actor.eval()

    adv_np = advantages.detach().cpu().numpy().astype(np.float64, copy=False)
    return {
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "entropy_mean": float(entropy_mean.detach().cpu().item()),
        "adv_mean": float(np.mean(adv_np)) if adv_np.size > 0 else 0.0,
        "adv_std": float(np.std(adv_np)) if adv_np.size > 0 else 0.0,
        "grad_norm": float(torch.sqrt(grad_sq).detach().cpu().item()),
        "bw_stage_samples": float(stage_batch["num_samples"]),
    }


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--rollout_env_steps", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--vec_backend", type=str, default="sync", choices=["sync", "subproc"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--init_actor", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint_eval_interval", type=int, default=5)
    parser.add_argument("--checkpoint_eval_episodes", type=int, default=12)
    parser.add_argument("--checkpoint_eval_episode_seed_base", type=int, default=53000)
    parser.add_argument("--checkpoint_eval_num_envs", type=int, default=8)
    args = parser.parse_args()

    device = _resolve_torch_device(args.device)
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    set_seed(int(cfg.seed))
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, run_dir / "config_source.yaml")

    bundle = build_structured_modules_from_config(cfg, hidden_dim=int(args.hidden_dim), embed_dim=int(args.embed_dim))
    actor = bundle.actor.to(device)
    if args.init_actor:
        load_checkpoint_forgiving(actor, str(args.init_actor), map_location=device, strict=True)
    actor.train()
    optimizer = torch.optim.Adam(actor.bw_policy.parameters(), lr=float(getattr(cfg, "actor_lr", 3.0e-4) or 3.0e-4))
    learner = StructuredMAPPO(
        actor=actor,
        critic=ZeroStructuredCritic(device),
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_ratio=float(getattr(cfg, "clip_ratio", 0.2) or 0.2),
        value_coef=0.0,
        entropy_coef=float(getattr(cfg, "entropy_coef", 0.0) or 0.0),
        max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5) or 0.5),
        ppo_epochs=1,
        num_mini_batch=1,
        actor_optimizer=None,
        critic_optimizer=None,
        device=device,
        target_mode="step_level",
        cfg=cfg,
        train_accel=False,
        train_sat=False,
        train_bw=True,
        exec_accel_source=str(getattr(cfg, "exec_accel_source", "policy") or "policy"),
        exec_sat_source=str(getattr(cfg, "exec_sat_source", "policy") or "policy"),
        exec_bw_source=str(getattr(cfg, "exec_bw_source", "policy") or "policy"),
    )

    fixed_summary, _ = evaluate_structured_actor_exec_sources(
        cfg,
        actor,
        device=device,
        episodes=int(args.checkpoint_eval_episodes),
        episode_seed_base=int(args.checkpoint_eval_episode_seed_base),
        deterministic=True,
        num_envs=int(args.checkpoint_eval_num_envs),
        vec_backend=str(args.vec_backend),
        exec_accel_source="zero",
        exec_sat_source=str(getattr(cfg, "exec_sat_source", "zero") or "zero"),
        exec_bw_source="queue_aware",
    )
    initial_summary, _ = evaluate_structured_actor_exec_sources(
        cfg,
        actor,
        device=device,
        episodes=int(args.checkpoint_eval_episodes),
        episode_seed_base=int(args.checkpoint_eval_episode_seed_base),
        deterministic=True,
        num_envs=int(args.checkpoint_eval_num_envs),
        vec_backend=str(args.vec_backend),
    )

    rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = [
        {
            "update": 0,
            "reward_sum": float(initial_summary["reward_sum"]),
            "processed_ratio_eval": float(initial_summary["processed_ratio_eval"]),
            "drop_ratio_eval": float(initial_summary["drop_ratio_eval"]),
            "pre_backlog_steps_eval": float(initial_summary["pre_backlog_steps_eval"]),
            "fixed_reward_sum": float(fixed_summary["reward_sum"]),
            "fixed_processed_ratio_eval": float(fixed_summary["processed_ratio_eval"]),
            "fixed_drop_ratio_eval": float(fixed_summary["drop_ratio_eval"]),
            "fixed_pre_backlog_steps_eval": float(fixed_summary["pre_backlog_steps_eval"]),
        }
    ]

    rollout_env_steps = int(args.rollout_env_steps if args.rollout_env_steps is not None else getattr(cfg, "buffer_size", 100))
    progress = Progress(max(int(args.updates), 0), desc="ActorOnly")
    wall_start = time.perf_counter()
    for update_idx in range(1, int(args.updates) + 1):
        rollout_seed = int(cfg.seed) + (update_idx - 1) * max(int(args.num_envs), 1) * 1000
        buffer, rollout_stats = _collect_rollout(
            cfg,
            learner,
            num_envs=int(args.num_envs),
            vec_backend=str(args.vec_backend),
            rollout_env_steps=rollout_env_steps,
            reset_seed=rollout_seed,
        )
        update_stats = _actor_only_bw_update(
            actor,
            optimizer,
            learner,
            buffer,
            entropy_coef=float(getattr(cfg, "entropy_coef", 0.0) or 0.0),
            max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5) or 0.5),
            stagewise_adv_norm=True,
            device=device,
        )
        row = {
            "update": int(update_idx),
            "rollout_reward_mean": float(rollout_stats["rollout_reward_mean"]),
            "rollout_bw_access_reward_mean": float(rollout_stats["rollout_bw_access_reward_mean"]),
            "policy_loss": float(update_stats["policy_loss"]),
            "entropy_mean": float(update_stats["entropy_mean"]),
            "adv_mean": float(update_stats["adv_mean"]),
            "adv_std": float(update_stats["adv_std"]),
            "grad_norm": float(update_stats["grad_norm"]),
            "bw_stage_samples": float(update_stats["bw_stage_samples"]),
            "wall_sec": float(time.perf_counter() - wall_start),
        }
        rows.append(row)
        progress.desc = (
            f"ActorOnly r={row['rollout_reward_mean']:.3f} "
            f"pl={row['policy_loss']:.3f}"
        )
        progress.update(update_idx)

        if int(args.checkpoint_eval_interval) > 0 and update_idx % int(args.checkpoint_eval_interval) == 0:
            summary, _ = evaluate_structured_actor_exec_sources(
                cfg,
                actor,
                device=device,
                episodes=int(args.checkpoint_eval_episodes),
                episode_seed_base=int(args.checkpoint_eval_episode_seed_base),
                deterministic=True,
                num_envs=int(args.checkpoint_eval_num_envs),
                vec_backend=str(args.vec_backend),
            )
            checkpoint_rows.append(
                {
                    "update": int(update_idx),
                    "reward_sum": float(summary["reward_sum"]),
                    "processed_ratio_eval": float(summary["processed_ratio_eval"]),
                    "drop_ratio_eval": float(summary["drop_ratio_eval"]),
                    "pre_backlog_steps_eval": float(summary["pre_backlog_steps_eval"]),
                    "fixed_reward_sum": float(fixed_summary["reward_sum"]),
                    "fixed_processed_ratio_eval": float(fixed_summary["processed_ratio_eval"]),
                    "fixed_drop_ratio_eval": float(fixed_summary["drop_ratio_eval"]),
                    "fixed_pre_backlog_steps_eval": float(fixed_summary["pre_backlog_steps_eval"]),
                }
            )
            torch.save(actor.state_dict(), run_dir / f"actor_u{int(update_idx):04d}.pt")

    actor.eval()
    torch.save(actor.state_dict(), run_dir / "actor_final.pt")
    final_det_summary, final_det_rows = evaluate_structured_actor_exec_sources(
        cfg,
        actor,
        device=device,
        episodes=int(args.checkpoint_eval_episodes),
        episode_seed_base=int(args.checkpoint_eval_episode_seed_base),
        deterministic=True,
        num_envs=int(args.checkpoint_eval_num_envs),
        vec_backend=str(args.vec_backend),
    )
    final_stoch_summary, final_stoch_rows = evaluate_structured_actor_exec_sources(
        cfg,
        actor,
        device=device,
        episodes=int(args.checkpoint_eval_episodes),
        episode_seed_base=int(args.checkpoint_eval_episode_seed_base),
        deterministic=False,
        num_envs=int(args.checkpoint_eval_num_envs),
        vec_backend=str(args.vec_backend),
    )

    _save_csv(run_dir / "train_rows.csv", rows)
    _save_csv(run_dir / "checkpoint_eval.csv", checkpoint_rows)
    _save_csv(run_dir / "eval_det_ep12.csv", final_det_rows)
    _save_csv(run_dir / "eval_stoch_ep12.csv", final_stoch_rows)
    summary_payload = {
        "run_dir": str(run_dir),
        "config": str(args.config),
        "updates": int(args.updates),
        "num_envs": int(args.num_envs),
        "vec_backend": str(args.vec_backend),
        "device": str(device),
        "fixed_summary": fixed_summary,
        "initial_summary": initial_summary,
        "final_det_summary": final_det_summary,
        "final_stoch_summary": final_stoch_summary,
        "best_checkpoint_reward": (
            float(max((row["reward_sum"] for row in checkpoint_rows), default=float("-inf")))
        ),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)
    print(
        f"Actor-only final det: reward={final_det_summary['reward_sum']:.4f} "
        f"processed={final_det_summary['processed_ratio_eval']:.4f} "
        f"drop={final_det_summary['drop_ratio_eval']:.4f} "
        f"pre_backlog={final_det_summary['pre_backlog_steps_eval']:.4f}"
    )
    print(
        f"Fixed queue-aware BW: reward={fixed_summary['reward_sum']:.4f} "
        f"processed={fixed_summary['processed_ratio_eval']:.4f} "
        f"drop={fixed_summary['drop_ratio_eval']:.4f} "
        f"pre_backlog={fixed_summary['pre_backlog_steps_eval']:.4f}"
    )


if __name__ == "__main__":
    main()
