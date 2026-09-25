from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sagin_marl.env.config import load_config
from sagin_marl.rl.distributed_queue import DQS_REVISION, settings_from_config
import torch

from sagin_marl.rl.structured_eval import (
    _evaluate_structured_baseline_policy_with_traces,
    _fixed_policy_exec_sources,
    evaluate_structured_actor_exec_sources,
)


def _resolve_fieldnames(rows: list[dict[str, float]]) -> list[str]:
    preferred = [
        "episode",
        "step_count",
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
    seen = set(preferred)
    extras: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                extras.append(key)
    return preferred + extras


def _default_out_path(config_path: str, baseline_policy: str) -> Path:
    config = Path(config_path)
    parent = config.parent if config.parent != Path("") else Path(".")
    return parent / f"eval_{baseline_policy}.csv"


def apply_dq_overrides(cfg, baseline: str, movement_weight=None, switch_weight=None):
    values = {"movement_weight": movement_weight, "switch_weight": switch_weight}
    if any(value is not None for value in values.values()) and baseline not in {
        "distributed_queue_a", "distributed_queue_b", "distributed_queue_c"
    }:
        raise ValueError("DQ weight overrides require a distributed_queue baseline")
    for name, value in values.items():
        if value is not None:
            setattr(cfg, f"baseline_dq_{name}", value)
    return settings_from_config(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--baseline", "--baseline_policy", dest="baseline_policy", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode_seed_base", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--summary_out", type=str, default=None)
    parser.add_argument("--T_steps", type=int, default=None)
    parser.add_argument(
        "--structured_env_tensor_backend",
        type=str,
        default=None,
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--access_bw_decision_interval", type=int, default=None)
    parser.add_argument("--sat_decision_interval", type=int, default=None)
    parser.add_argument("--dq_movement_weight", type=float, default=None)
    parser.add_argument("--dq_switch_weight", type=float, default=None)
    args = parser.parse_args()
    if args.num_envs < 1 or args.episodes < 1:
        parser.error("num_envs and episodes must be positive")

    cfg = load_config(args.config)
    if args.T_steps is not None:
        cfg.T_steps = int(args.T_steps)
    if args.structured_env_tensor_backend is not None:
        cfg.structured_env_tensor_backend = str(args.structured_env_tensor_backend)

    for field in ("access_bw_decision_interval", "sat_decision_interval"):
        value = getattr(args, field)
        if value is not None:
            if value < 1:
                parser.error(f"{field} must be positive")
            setattr(cfg, field, value)

    try:
        dq_settings = apply_dq_overrides(
            cfg, args.baseline_policy, args.dq_movement_weight, args.dq_switch_weight)
    except ValueError as exc:
        parser.error(str(exc))

    sources = _fixed_policy_exec_sources(args.baseline_policy)
    if sources is not None:
        if not torch.cuda.is_available():
            parser.error("Native fixed-policy evaluation requires CUDA")
        if args.structured_env_tensor_backend == "cpu":
            parser.error("Native fixed-policy evaluation cannot use the CPU backend")
        summary, rows = evaluate_structured_actor_exec_sources(
            cfg, torch.nn.Linear(1, 1), device=torch.device("cuda"),
            episodes=args.episodes, num_envs=args.num_envs,
            episode_seed_base=args.episode_seed_base, deterministic=True,
            exec_accel_source=sources[0], exec_sat_source=sources[1],
            exec_bw_source=sources[2],
        )
    else:
        if args.num_envs != 1:
            parser.error("Legacy fixed policies require num_envs=1")
        summary, rows, _traces, _actions, _reset_rollouts = _evaluate_structured_baseline_policy_with_traces(
            cfg, baseline_policy=args.baseline_policy, episodes=args.episodes,
            episode_seed_base=args.episode_seed_base,
        )

    out_path = Path(args.out) if args.out else _default_out_path(args.config, str(args.baseline_policy))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_resolve_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "baseline": args.baseline_policy,
        "dqs_revision": DQS_REVISION if args.baseline_policy == "distributed_queue_c" else None,
        "config_path": str(Path(args.config).resolve()),
        "input_config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "effective_config": asdict(cfg),
        "dq_settings": asdict(dq_settings) if args.baseline_policy.startswith("distributed_queue_") else None,
        "exec_sources": sources, "episodes": args.episodes, "num_envs": args.num_envs,
        "episode_seed_base": args.episode_seed_base, "deterministic": True,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if sources is not None else "legacy",
    }
    out_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"Summary[{args.baseline_policy}]: reward={summary['reward_sum']:.4f} "
        f"processed={summary['processed_ratio_eval']:.4f} "
        f"drop={summary['drop_ratio_eval']:.4f} "
        f"pre_backlog={summary['pre_backlog_steps_eval']:.4f} "
        f"collision={summary['collision_episode_fraction']:.4f} "
        f"out={out_path}"
    )


if __name__ == "__main__":
    main()
