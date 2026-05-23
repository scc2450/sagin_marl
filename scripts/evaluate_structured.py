from __future__ import annotations

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_eval import (
    evaluate_structured_actor,
    evaluate_structured_actor_exec_sources,
)
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving


def _resolve_torch_device(device_arg: str) -> torch.device:
    requested = str(device_arg).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is not available.")
    return torch.device(requested)


def _resolve_eval_fieldnames(rows: list[dict[str, float]]) -> list[str]:
    preferred = [
        "episode",
        "reward_sum",
        "bw_weighted_workload_delta_sum",
        "bw_weighted_workload_level_sum",
        "processed_ratio_eval",
        "drop_ratio_eval",
        "pre_backlog_steps_eval",
        "D_sys_report",
        "x_acc_mean",
        "x_rel_mean",
        "g_pre_mean",
        "d_pre_mean",
        "sat_overlap_eval",
        "collision_episode_fraction",
    ]
    extras: list[str] = []
    seen = set(preferred)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                extras.append(key)
    return preferred + extras


def _resolve_eval_paths(run_dir: str | None, checkpoint: str | None, out: str | None) -> tuple[str, str]:
    if run_dir:
        checkpoint = checkpoint or os.path.join(run_dir, "actor_final.pt")
        out = out or os.path.join(run_dir, "eval_trained.csv")
    else:
        checkpoint = checkpoint or "runs/structured/actor_final.pt"
        out = out or "runs/structured/eval_trained.csv"
    return checkpoint, out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--policy_mode", type=str, default="deterministic", choices=["deterministic", "stochastic"])
    parser.add_argument("--episode_seed_base", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--vec_backend", type=str, default="sync", choices=["sync", "subproc"])
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--structured_env_tensor_backend",
        type=str,
        default=None,
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument("--respect_exec_sources", action="store_true")
    parser.add_argument("--exec_accel_source", type=str, default=None)
    parser.add_argument("--exec_sat_source", type=str, default=None)
    parser.add_argument("--exec_bw_source", type=str, default=None)
    parser.add_argument("--T_steps", type=int, default=None)
    args = parser.parse_args()

    checkpoint, out_path = _resolve_eval_paths(args.run_dir, args.checkpoint, args.out)
    config_path = args.config or (os.path.join(args.run_dir, "config_source.yaml") if args.run_dir else "configs/phase1.yaml")
    cfg = load_config(config_path)
    if args.T_steps is not None:
        cfg.T_steps = int(args.T_steps)
    device = _resolve_torch_device(args.device)
    if args.structured_env_tensor_backend is not None:
        cfg.structured_env_tensor_backend = str(args.structured_env_tensor_backend)
    bundle = build_structured_modules_from_config(cfg, hidden_dim=args.hidden_dim, embed_dim=args.embed_dim)
    actor = bundle.actor
    info = load_checkpoint_forgiving(actor, checkpoint, map_location=device, strict=False)
    if info.get("adapted_keys"):
        print(f"Loaded actor with adapted tensors from {checkpoint}: {len(info['adapted_keys'])}")
    actor.to(device)
    actor.eval()
    if args.respect_exec_sources:
        summary, rows = evaluate_structured_actor_exec_sources(
            cfg,
            actor,
            device=device,
            episodes=int(args.episodes),
            episode_seed_base=args.episode_seed_base,
            deterministic=args.policy_mode != "stochastic",
            num_envs=int(args.num_envs),
            vec_backend=str(args.vec_backend),
            exec_accel_source=args.exec_accel_source,
            exec_sat_source=args.exec_sat_source,
            exec_bw_source=args.exec_bw_source,
        )
    else:
        summary, rows = evaluate_structured_actor(
            cfg,
            actor,
            device=device,
            episodes=int(args.episodes),
            episode_seed_base=args.episode_seed_base,
            deterministic=args.policy_mode != "stochastic",
            num_envs=int(args.num_envs),
            vec_backend=str(args.vec_backend),
        )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fieldnames = _resolve_eval_fieldnames(rows)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Summary: reward={summary['reward_sum']:.4f} "
        f"processed={summary['processed_ratio_eval']:.4f} "
        f"drop={summary['drop_ratio_eval']:.4f} "
        f"pre_backlog={summary['pre_backlog_steps_eval']:.4f} "
        f"sat_overlap={summary['sat_overlap_eval']:.4f} "
        f"collision={summary['collision_episode_fraction']:.4f}"
    )


if __name__ == "__main__":
    main()
