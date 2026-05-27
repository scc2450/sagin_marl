from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import is_dataclass
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
from sagin_marl.rl.structured_mappo import (
    StructuredMAPPO,
    _collate_dataclass,
    _split_cpu_tensor_and_sums_by_counts,
)
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


def _load_actor_critic(
    run_dir: Path,
    update: int,
    device: torch.device,
    *,
    actor_checkpoint: str | None,
    critic_checkpoint: str | None,
):
    cfg = load_config(str(run_dir / "config_source.yaml"))
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    actor_ckpt = Path(actor_checkpoint) if actor_checkpoint is not None else run_dir / f"actor_u{int(update):04d}.pt"
    critic_ckpt = Path(critic_checkpoint) if critic_checkpoint is not None else run_dir / f"critic_u{int(update):04d}.pt"
    load_checkpoint_forgiving(bundle.actor, str(actor_ckpt), map_location=device, strict=True)
    load_checkpoint_forgiving(bundle.critic, str(critic_ckpt), map_location=device, strict=True)
    bundle.actor.to(device).eval()
    bundle.critic.to(device).eval()
    return cfg, bundle.actor, bundle.critic


def _collect_rollout(
    cfg: Any,
    actor: Any,
    critic: Any,
    *,
    env_steps: int,
    seed: int,
    deterministic: bool,
    device: torch.device,
) -> StructuredRolloutBuffer:
    env = make_structured_env(cfg, mode="script")
    try:
        algo = StructuredMAPPO(
            actor=actor,
            critic=critic,
            gamma=float(cfg.gamma),
            gae_lambda=float(cfg.gae_lambda),
            clip_ratio=float(getattr(cfg, "clip_ratio", 0.2) or 0.2),
            value_coef=float(getattr(cfg, "value_coef", 0.5) or 0.5),
            entropy_coef=float(getattr(cfg, "entropy_coef", 0.0) or 0.0),
            max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5) or 0.5),
            ppo_epochs=int(getattr(cfg, "ppo_epochs", 1) or 1),
            num_mini_batch=int(getattr(cfg, "num_mini_batch", 1) or 1),
            device=device,
            cfg=cfg,
        )
        buffer = StructuredRolloutBuffer()
        env.reset(seed=int(seed))
        driver = as_structured_driver(env)
        episode_idx = 0
        for _ in range(int(env_steps)):
            step_result = algo.collect_env_step(driver, buffer, deterministic=deterministic)
            terminated = bool(next(iter(step_result.terminations.values())))
            truncated = bool(next(iter(step_result.truncations.values())))
            if terminated or truncated:
                episode_idx += 1
                env.reset(seed=int(seed) + episode_idx)
                driver = as_structured_driver(env)
        return buffer
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def _stack_optional(values: list[torch.Tensor | None], *, device: torch.device) -> torch.Tensor | None:
    if not values or any(value is None for value in values):
        return None
    return torch.stack([value.to(device=device) for value in values], dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--actor_checkpoint", type=str, default=None)
    parser.add_argument("--critic_checkpoint", type=str, default=None)
    parser.add_argument("--env_steps", type=int, default=24)
    parser.add_argument("--policy_mode", choices=["deterministic", "stochastic"], default="stochastic")
    parser.add_argument("--seed", type=int, default=34567)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_all_seeds(int(args.seed))
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    cfg, actor, critic = _load_actor_critic(
        run_dir,
        int(args.update),
        device,
        actor_checkpoint=None if args.actor_checkpoint is None else str(args.actor_checkpoint),
        critic_checkpoint=None if args.critic_checkpoint is None else str(args.critic_checkpoint),
    )
    buffer = _collect_rollout(
        cfg,
        actor,
        critic,
        env_steps=int(args.env_steps),
        seed=int(args.seed),
        deterministic=(str(args.policy_mode) == "deterministic"),
        device=device,
    )
    rollout_views = buffer.build_rollout_views(device)
    bw_stage_batch = rollout_views.training_view.stage_batches.get(2)
    if bw_stage_batch is None or int(bw_stage_batch.num_samples) <= 0:
        raise RuntimeError("No BW transitions found in collected rollout.")
    local_batch = bw_stage_batch.local_batch
    counts = [int(bw_stage_batch.num_agents)] * int(bw_stage_batch.num_samples)
    flat_action = bw_stage_batch.actions.to(device=device, dtype=torch.float32).reshape(-1, bw_stage_batch.actions.shape[-1])
    old_joint = bw_stage_batch.old_logprobs.to(device=device, dtype=torch.float32)
    old_per_agent = (
        bw_stage_batch.old_logprobs_per_agent.to(device=device, dtype=torch.float32)
        if bw_stage_batch.old_logprobs_per_agent is not None
        else None
    )
    support_mask = (
        bw_stage_batch.bw_support_masks.to(device=device, dtype=torch.float32)
        if bw_stage_batch.bw_support_masks is not None
        else None
    )
    flat_support_mask = None
    if support_mask is not None:
        flat_support_mask = torch.cat([mask.to(dtype=torch.float32) for mask in support_mask.unbind(dim=0)], dim=0)

    with torch.inference_mode():
        eval_out = actor.evaluate_bw(local_batch, flat_action, support_mask_override=flat_support_mask)
        _, recomputed_joint_list = _split_cpu_tensor_and_sums_by_counts(eval_out.logprob, counts)
        recomputed_joint = torch.stack(recomputed_joint_list, dim=0).to(device=device, dtype=torch.float32)
        recomputed_per_agent = eval_out.logprob.reshape(-1).to(device=device, dtype=torch.float32)

        default_joint = None
        if flat_support_mask is not None:
            eval_out_no_override = actor.evaluate_bw(local_batch, flat_action, support_mask_override=None)
            _, default_joint_list = _split_cpu_tensor_and_sums_by_counts(eval_out_no_override.logprob, counts)
            default_joint = torch.stack(default_joint_list, dim=0).to(device=device, dtype=torch.float32)

    joint_abs_diff = (recomputed_joint - old_joint).abs().detach().cpu().numpy().astype(np.float64)
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "update": int(args.update),
        "actor_checkpoint": None if args.actor_checkpoint is None else str(Path(args.actor_checkpoint).resolve()),
        "critic_checkpoint": None if args.critic_checkpoint is None else str(Path(args.critic_checkpoint).resolve()),
        "policy_mode": str(args.policy_mode),
        "env_steps": int(args.env_steps),
        "bw_transition_count": int(bw_stage_batch.num_samples),
        "device": str(device),
        "joint_old_logprob_abs_diff": _summarize(joint_abs_diff.tolist()),
        "joint_old_logprob_exact_replay_frac_1e-5": float(np.mean(joint_abs_diff <= 1.0e-5)),
        "action_simplex_sum_abs_err": _summarize(
            [
                float(abs(torch.as_tensor(action, dtype=torch.float32).sum(dim=-1) - 1.0).max().item())
                for action in bw_stage_batch.actions.detach().cpu()
            ]
        ),
        "action_min_value": _summarize(
            [float(torch.as_tensor(batches["actions"][i], dtype=torch.float32).min().item()) for i in bw_indices]
        ),
        "support_mask_present_count": int(sum(1 for value in support_mask_values if value is not None)),
    }
    if old_per_agent is not None:
        old_per_agent_flat = torch.cat([value.to(device=device).reshape(-1) for value in per_agent_values if value is not None], dim=0)
        per_agent_abs_diff = (recomputed_per_agent - old_per_agent_flat).abs().detach().cpu().numpy().astype(np.float64)
        summary["per_agent_old_logprob_abs_diff"] = _summarize(per_agent_abs_diff.tolist())
        summary["per_agent_old_logprob_exact_replay_frac_1e-5"] = float(np.mean(per_agent_abs_diff <= 1.0e-5))
    if default_joint is not None:
        default_joint_abs_diff = (default_joint - old_joint).abs().detach().cpu().numpy().astype(np.float64)
        summary["no_support_override_joint_abs_diff"] = _summarize(default_joint_abs_diff.tolist())
        summary["support_override_required_frac_1e-4"] = float(np.mean(default_joint_abs_diff > 1.0e-4))

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
