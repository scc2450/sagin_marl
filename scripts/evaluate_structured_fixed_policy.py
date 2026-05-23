from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_eval import _evaluate_structured_baseline_policy_with_traces


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
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.T_steps is not None:
        cfg.T_steps = int(args.T_steps)
    if args.structured_env_tensor_backend is not None:
        cfg.structured_env_tensor_backend = str(args.structured_env_tensor_backend)

    summary, rows, _traces, _actions, _reset_rollouts = _evaluate_structured_baseline_policy_with_traces(
        cfg,
        baseline_policy=str(args.baseline_policy),
        episodes=int(args.episodes),
        episode_seed_base=args.episode_seed_base,
    )

    out_path = Path(args.out) if args.out else _default_out_path(args.config, str(args.baseline_policy))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_resolve_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)

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
