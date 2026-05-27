from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from sagin_marl.env.config import load_config
from sagin_marl.rl.stage_mcgae import stage_optimizer_params as _stage_optimizer_params
from sagin_marl.rl.structured_eval import evaluate_structured_actor_exec_sources
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver_group
from scripts.train_joint_mcgae import (
    _actor_only_stage_batch,
    _collect_joint_rollout,
    _force_joint_config,
    _make_joint_learner,
    _normalize_stage_advantage,
    _set_seed,
)
from scripts.train_stage_mcgae import _stage_actor_update_full_stage, _stage_gae_from_mc_targets, _train_stage_critic_on_stage


STAGE_ACCEL = 0


def _load_joint_checkpoint(learner: Any, checkpoint: Path, device: torch.device) -> dict[str, Any]:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if "actor" not in state or "critic" not in state:
        raise RuntimeError(f"{checkpoint} is not a joint checkpoint with actor/critic state.")
    learner.actor.load_state_dict(state["actor"], strict=True)
    learner.critic.load_state_dict(state["critic"], strict=True)
    sync_native = getattr(learner, "_sync_native_actor_cuda_bindings_after_update", None)
    if callable(sync_native):
        sync_native()
    return state


def _eval_actor(
    cfg: Any,
    actor: torch.nn.Module,
    *,
    device: torch.device,
    episodes: int,
    seed: int,
    num_envs: int,
) -> dict[str, float]:
    actor.eval()
    summary, _rows = evaluate_structured_actor_exec_sources(
        cfg,
        actor,
        device=device,
        episodes=int(episodes),
        episode_seed_base=int(seed),
        deterministic=True,
        num_envs=int(num_envs),
        vec_backend="sync",
        exec_accel_source="policy",
        exec_sat_source="policy",
        exec_bw_source="policy",
    )
    return {str(k): float(v) for k, v in summary.items()}


def _format_lr(lr: float) -> str:
    return f"{float(lr):.0e}".replace("+", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="One-update accel LR causal test from joint MC-GAE checkpoints.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--rollout_env_steps", type=int, default=250)
    parser.add_argument("--rollout_seed", type=int, default=910000)
    parser.add_argument("--eval_episodes", type=int, default=64)
    parser.add_argument("--eval_seed", type=int, default=990000)
    parser.add_argument("--reward_mode", default="positive_weighted_workload_level")
    parser.add_argument("--critic_lr", type=float, default=3.0e-4)
    parser.add_argument("--critic_epochs", type=int, default=5)
    parser.add_argument("--critic_minibatches", type=int, default=8)
    parser.add_argument("--critic_update_microbatch_size", type=int, default=1000)
    parser.add_argument("--actor_epochs", type=int, default=5)
    parser.add_argument("--actor_minibatches", type=int, default=1)
    parser.add_argument("--actor_lr", action="append", type=float, default=None)
    parser.add_argument("--torch_threads", type=int, default=1)
    args = parser.parse_args()

    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available.")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []

    cfg = load_config(args.config)
    _force_joint_config(cfg, reward_mode=str(args.reward_mode))
    cfg.checkpoint_eval_enabled = False
    cfg.train_trace_enabled = False

    for ckpt_text in args.checkpoint:
        checkpoint = Path(ckpt_text)
        tag = checkpoint.stem
        print(f"[checkpoint] {checkpoint}", flush=True)
        _set_seed(45210)
        learner = _make_joint_learner(cfg, device=device)
        ckpt_state = _load_joint_checkpoint(learner, checkpoint, device)
        base_actor_state = {k: v.detach().clone() for k, v in learner.actor.state_dict().items()}

        t0 = time.perf_counter()
        eval_before = _eval_actor(
            cfg,
            learner.actor,
            device=device,
            episodes=int(args.eval_episodes),
            seed=int(args.eval_seed),
            num_envs=int(args.num_envs),
        )
        print(
            f"  baseline eval reward={eval_before.get('reward_sum', float('nan')):.4f} "
            f"processed={eval_before.get('processed_ratio_eval', float('nan')):.4f} "
            f"collision={eval_before.get('collision_episode_fraction', float('nan')):.4f}",
            flush=True,
        )

        group = make_structured_driver_group(cfg, num_envs=int(args.num_envs), backend="sync", mode="train")
        try:
            views, _all_returns, stage_targets, reward_stats = _collect_joint_rollout(
                learner,
                group,
                rollout_env_steps=int(args.rollout_env_steps),
                device=device,
                seed=int(args.rollout_seed),
            )
        finally:
            close_structured_env_group(group)
        collect_sec = time.perf_counter() - t0

        stage_batch = views.training_view.stage_batches[STAGE_ACCEL]
        critic_optimizer = torch.optim.Adam(learner.critic.parameters(), lr=float(args.critic_lr))
        critic_stats, after_values, _trace = _train_stage_critic_on_stage(
            learner,
            stage_id=STAGE_ACCEL,
            stage_batch=stage_batch,
            target=stage_targets[STAGE_ACCEL],
            optimizer=critic_optimizer,
            lr=float(args.critic_lr),
            epochs=int(args.critic_epochs),
            minibatches=int(args.critic_minibatches),
            update_microbatch_size=int(args.critic_update_microbatch_size),
            diagnose_timing=False,
        )
        _returns_gae, stage_adv, _stage_values = _stage_gae_from_mc_targets(
            learner,
            stage_id=STAGE_ACCEL,
            stage_batch=stage_batch,
            mc_target=stage_targets[STAGE_ACCEL],
            device=device,
            stage_values=after_values,
        )
        stage_adv_norm = _normalize_stage_advantage(stage_adv, enabled=True)
        actor_stage_batch = _actor_only_stage_batch(stage_batch)

        # Drop large rollout views before repeated actor/eval passes.
        del views
        del stage_targets
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)

        row_base: dict[str, float | str] = {
            "checkpoint": str(checkpoint),
            "checkpoint_tag": tag,
            "checkpoint_update": float(ckpt_state.get("update", float("nan"))),
            "variant": "no_update",
            "actor_lr": 0.0,
            "collect_sec": float(collect_sec),
            "rollout_accel_mc_return_mean": float(reward_stats.get("accel_mc_return_mean", float("nan"))),
            "critic_ev_after": float(critic_stats.get("critic_ev_after", float("nan"))),
            "raw_adv_std": float(stage_adv.detach().float().std(unbiased=False).cpu().item()),
            "norm_adv_std": float(stage_adv_norm.detach().float().std(unbiased=False).cpu().item()),
            **{f"eval_{k}": float(v) for k, v in eval_before.items()},
        }
        rows.append(row_base)

        actor_lrs = [float(v) for v in (args.actor_lr if args.actor_lr is not None else [3.0e-4, 1.0e-4, 3.0e-5])]
        for lr in actor_lrs:
            learner.actor.load_state_dict(base_actor_state, strict=True)
            sync_native = getattr(learner, "_sync_native_actor_cuda_bindings_after_update", None)
            if callable(sync_native):
                sync_native()
            opt = torch.optim.Adam(_stage_optimizer_params(learner.actor, STAGE_ACCEL), lr=float(lr))
            t_update = time.perf_counter()
            actor_stats = _stage_actor_update_full_stage(
                learner,
                stage_id=STAGE_ACCEL,
                stage_batch=actor_stage_batch,
                stage_advantages=stage_adv_norm,
                optimizer=opt,
                epochs=int(args.actor_epochs),
                minibatches=int(args.actor_minibatches),
                parity_dump_dir=None,
                parity_dump_tag=None,
                parity_topk=0,
            )
            sync_native = getattr(learner, "_sync_native_actor_cuda_bindings_after_update", None)
            if callable(sync_native):
                sync_native()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            update_sec = time.perf_counter() - t_update
            eval_after = _eval_actor(
                cfg,
                learner.actor,
                device=device,
                episodes=int(args.eval_episodes),
                seed=int(args.eval_seed),
                num_envs=int(args.num_envs),
            )
            delta_reward = float(eval_after.get("reward_sum", 0.0)) - float(eval_before.get("reward_sum", 0.0))
            delta_processed = float(eval_after.get("processed_ratio_eval", 0.0)) - float(
                eval_before.get("processed_ratio_eval", 0.0)
            )
            print(
                f"  lr={lr:.1e} eval reward={eval_after.get('reward_sum', float('nan')):.4f} "
                f"delta={delta_reward:+.4f} kl={actor_stats.get('approx_kl_accel', float('nan')):.4g} "
                f"clip={actor_stats.get('clip_frac_accel', float('nan')):.3f}",
                flush=True,
            )
            rows.append(
                {
                    "checkpoint": str(checkpoint),
                    "checkpoint_tag": tag,
                    "checkpoint_update": float(ckpt_state.get("update", float("nan"))),
                    "variant": f"accel_lr_{_format_lr(lr)}",
                    "actor_lr": float(lr),
                    "collect_sec": float(collect_sec),
                    "actor_update_sec": float(update_sec),
                    "rollout_accel_mc_return_mean": float(reward_stats.get("accel_mc_return_mean", float("nan"))),
                    "critic_ev_after": float(critic_stats.get("critic_ev_after", float("nan"))),
                    "raw_adv_std": float(stage_adv.detach().float().std(unbiased=False).cpu().item()),
                    "norm_adv_std": float(stage_adv_norm.detach().float().std(unbiased=False).cpu().item()),
                    **{f"actor_{k}": float(v) for k, v in actor_stats.items()},
                    **{f"eval_{k}": float(v) for k, v in eval_after.items()},
                    "delta_eval_reward_sum": float(delta_reward),
                    "delta_eval_processed_ratio": float(delta_processed),
                }
            )

    out_csv = run_dir / "accel_one_update_lr_effect.csv"
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    with (run_dir / "accel_one_update_lr_effect.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
