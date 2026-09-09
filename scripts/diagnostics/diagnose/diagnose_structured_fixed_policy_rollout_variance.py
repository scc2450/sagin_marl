from __future__ import annotations

import argparse
import csv
import os
import sys
import types
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_critic import ZeroStructuredCritic
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO
from sagin_marl.rl.structured_train import (
    close_structured_env_group,
    make_structured_env_group,
    run_structured_training,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num_updates", type=int, default=10)
    parser.add_argument("--rollout_env_steps", type=int, default=80)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--backend", type=str, default="sync", choices=["sync", "subproc"])
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--csv_out", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    bundle = build_structured_modules_from_config(
        cfg,
        hidden_dim=int(args.hidden_dim),
        embed_dim=int(args.embed_dim),
        build_critic=False,
    )
    actor = bundle.actor
    learner = StructuredMAPPO(
        actor=actor,
        critic=ZeroStructuredCritic(torch.device("cpu")),
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_ratio=cfg.clip_ratio,
        value_coef=cfg.value_coef,
        entropy_coef=cfg.entropy_coef,
        max_grad_norm=cfg.max_grad_norm,
        ppo_epochs=1,
        num_mini_batch=1,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=1.0e-3),
        critic_optimizer=None,
        target_mode="step_level",
        cfg=cfg,
        train_accel=bool(getattr(cfg, "train_accel", True)),
        train_sat=bool(getattr(cfg, "train_sat", True)),
        train_bw=bool(getattr(cfg, "train_bw", True)),
        exec_accel_source=str(getattr(cfg, "exec_accel_source", "policy") or "policy"),
        exec_sat_source=str(getattr(cfg, "exec_sat_source", "policy") or "policy"),
        exec_bw_source=str(getattr(cfg, "exec_bw_source", "policy") or "policy"),
    )

    def _no_update(self, buffer, bootstrap_world_state=None):
        del buffer, bootstrap_world_state
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
        }

    learner.update = types.MethodType(_no_update, learner)

    env_group = make_structured_env_group(cfg, num_envs=int(args.num_envs), backend=str(args.backend))
    try:
        rows = run_structured_training(
            env_group,
            learner,
            num_updates=int(args.num_updates),
            rollout_env_steps=int(args.rollout_env_steps),
            reset_seed=int(cfg.seed),
        )
    finally:
        close_structured_env_group(env_group)

    fieldnames = [
        "update",
        "env_reward_mean",
        "rollout_reward_per_step",
        "episode_reward",
        "episode_reward_std",
        "completed_episode_count",
        "episodes_finished",
    ]
    payload = []
    for idx, row in enumerate(rows, start=1):
        payload.append(
            {
                "update": idx,
                "env_reward_mean": float(row.env_reward_mean),
                "rollout_reward_per_step": float(row.rollout_reward_per_step),
                "episode_reward": float(row.episode_reward),
                "episode_reward_std": float(row.episode_reward_std),
                "completed_episode_count": int(row.completed_episode_count),
                "episodes_finished": int(row.episodes_finished),
            }
        )

    if args.csv_out:
        out_path = os.path.abspath(args.csv_out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(payload)

    writer = csv.DictWriter(
        __import__("sys").stdout,
        fieldnames=fieldnames,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(payload)


if __name__ == "__main__":
    main()
