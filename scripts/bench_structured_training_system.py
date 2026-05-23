from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO
from sagin_marl.rl.structured_system_benchmark import (
    benchmark_structured_training,
    summarize_structured_benchmark,
)
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_env_group


def _resolve_torch_device(device_arg: str) -> torch.device:
    requested = str(device_arg).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def _build_learner(cfg, *, device: torch.device, hidden_dim: int, embed_dim: int) -> StructuredMAPPO:
    bundle = build_structured_modules_from_config(cfg, hidden_dim=hidden_dim, embed_dim=embed_dim)
    actor = bundle.actor
    critic = bundle.critic
    actor_optim = torch.optim.Adam(actor.parameters(), lr=float(getattr(cfg, "actor_lr", 3.0e-4) or 3.0e-4))
    critic_optim = torch.optim.Adam(critic.parameters(), lr=float(getattr(cfg, "critic_lr", 1.0e-3) or 1.0e-3))
    return StructuredMAPPO(
        actor=actor,
        critic=critic,
        gamma=float(getattr(cfg, "gamma", 0.99) or 0.99),
        gae_lambda=float(getattr(cfg, "gae_lambda", 0.95) or 0.95),
        clip_ratio=float(getattr(cfg, "clip_ratio", 0.2) or 0.2),
        value_coef=float(getattr(cfg, "value_coef", 0.5) or 0.5),
        entropy_coef=float(getattr(cfg, "entropy_coef", 0.01) or 0.01),
        max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5) or 0.5),
        ppo_epochs=max(int(getattr(cfg, "ppo_epochs", 1) or 1), 1),
        num_mini_batch=max(int(getattr(cfg, "num_mini_batch", 1) or 1), 1),
        actor_optimizer=actor_optim,
        critic_optimizer=critic_optim,
        device=device,
        cfg=cfg,
        train_accel=bool(getattr(cfg, "train_accel", True)),
        train_sat=bool(getattr(cfg, "train_sat", True)),
        train_bw=bool(getattr(cfg, "train_bw", True)),
        exec_accel_source=str(getattr(cfg, "exec_accel_source", "policy") or "policy"),
        exec_sat_source=str(getattr(cfg, "exec_sat_source", "policy") or "policy"),
        exec_bw_source=str(getattr(cfg, "exec_bw_source", "policy") or "policy"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the formal structured training benchmark with unified metrics.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-updates", type=int, default=3)
    parser.add_argument("--warmup-updates", type=int, default=1)
    parser.add_argument("--rollout-env-steps", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--env-tensor-backend",
        choices=["device", "config", "auto", "cpu", "cuda"],
        default="device",
        help="Structured env tensor backend. Defaults to following --device so CPU/GPU benchmarks are not hybrid by accident.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--sample-interval-sec", type=float, default=0.05)
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--json-path", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = _resolve_torch_device(str(args.device))
    env_tensor_backend = str(args.env_tensor_backend).strip().lower()
    if env_tensor_backend == "device":
        cfg.structured_env_tensor_backend = "cuda" if device.type == "cuda" else "cpu"
    elif env_tensor_backend != "config":
        cfg.structured_env_tensor_backend = env_tensor_backend
    rollout_env_steps = (
        int(args.rollout_env_steps)
        if args.rollout_env_steps is not None
        else int(getattr(cfg, "buffer_size", 32) or 32)
    )
    env_group = make_structured_env_group(cfg, num_envs=max(int(args.num_envs), 1), backend="sync", mode="train")
    try:
        learner = _build_learner(
            cfg,
            device=device,
            hidden_dim=max(int(args.hidden_dim), 1),
            embed_dim=max(int(args.embed_dim), 1),
        )
        rows = benchmark_structured_training(
            env_group,
            learner,
            num_updates=max(int(args.num_updates), 1),
            warmup_updates=max(int(args.warmup_updates), 0),
            rollout_env_steps=max(int(rollout_env_steps), 1),
            reset_seed=int(cfg.seed),
            sample_interval_sec=float(args.sample_interval_sec),
        )
    finally:
        close_structured_env_group(env_group)

    row_payloads = [asdict(row) for row in rows]
    summary = summarize_structured_benchmark(rows)

    if args.csv_path:
        with open(args.csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row_payloads[0].keys()))
            writer.writeheader()
            writer.writerows(row_payloads)
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump({"rows": row_payloads, "summary": summary}, f, indent=2, ensure_ascii=False)

    print(
        "structured_benchmark "
        f"updates={int(summary['updates'])} "
        f"device={device.type} "
        f"env_tensor_backend={str(getattr(cfg, 'structured_env_tensor_backend', 'unknown'))} "
        f"env_steps_per_sec={float(summary['env_steps_per_sec']):.2f} "
        f"samples_per_sec={float(summary['samples_per_sec']):.2f}"
    )
    print(
        "structured_benchmark_timing "
        f"rollout={float(summary['rollout_total_time_sec']):.4f}s "
        f"update={float(summary['update_total_time_sec']):.4f}s "
        f"iter={float(summary['iteration_time_sec']):.4f}s"
    )
    print(
        "structured_benchmark_util "
        f"cpu={float(summary['cpu_util_percent']):.2f}% "
        f"gpu={float(summary['gpu_util_percent']):.2f}% "
        f"gpu_mem={float(summary['gpu_memory_util_percent']):.2f}%"
    )


if __name__ == "__main__":
    main()
