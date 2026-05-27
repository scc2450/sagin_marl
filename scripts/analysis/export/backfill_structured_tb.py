from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict

from torch.utils.tensorboard import SummaryWriter

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _init_structured_tb_layout(writer: SummaryWriter) -> None:
    layout = {
        "Structured/Train": {
            "Reward": ["Multiline", ["env_reward_mean"]],
            "PPO": ["Multiline", ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"]],
            "DangerImitation": [
                "Multiline",
                ["danger_imitation_loss", "danger_imitation_active_rate"],
            ],
            "Episodes": ["Multiline", ["episodes_finished"]],
        },
        "Structured/Eval": {
            "CheckpointEval": [
                "Multiline",
                [
                    "checkpoint_eval/reward_sum",
                    "checkpoint_eval/processed_ratio_eval",
                    "checkpoint_eval/drop_ratio_eval",
                    "checkpoint_eval/pre_backlog_steps_eval",
                    "checkpoint_eval/sat_overlap_eval",
                    "checkpoint_eval/collision_episode_fraction",
                ],
            ],
            "FixedReference": [
                "Multiline",
                [
                    "checkpoint_eval/fixed_reward_sum",
                    "checkpoint_eval/fixed_processed_ratio_eval",
                    "checkpoint_eval/fixed_drop_ratio_eval",
                    "checkpoint_eval/fixed_pre_backlog_steps_eval",
                    "checkpoint_eval/fixed_sat_overlap_eval",
                    "checkpoint_eval/fixed_collision_episode_fraction",
                ],
            ],
        },
    }
    writer.add_custom_scalars(layout)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _maybe_float(row: Dict[str, str], key: str) -> float | None:
    raw = row.get(key)
    if raw is None or raw == "":
        return None
    return float(raw)


def _infer_num_envs(run_dir: Path) -> int | None:
    log_path = run_dir / "bootstrap.log"
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"structured_trainer envs=(\d+)", text)
    return int(m.group(1)) if m else None


def _infer_rollout_env_steps(run_dir: Path) -> int | None:
    log_path = run_dir / "bootstrap.log"
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"update:start idx=\d+ rollout_env_steps=(\d+)", text)
    return int(m.group(1)) if m else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill TensorBoard event files for a structured run directory.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--rollout_env_steps", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.csv not found under {run_dir}")

    rollout_env_steps = args.rollout_env_steps if args.rollout_env_steps is not None else _infer_rollout_env_steps(run_dir)
    num_envs = args.num_envs if args.num_envs is not None else _infer_num_envs(run_dir)
    if rollout_env_steps is None or num_envs is None:
        raise ValueError(
            "Could not infer rollout_env_steps/num_envs from bootstrap.log; "
            "please pass --rollout_env_steps and --num_envs explicitly."
        )

    train_rows = _read_csv_rows(metrics_path)
    eval_rows = _read_csv_rows(run_dir / "checkpoint_eval.csv")

    writer = SummaryWriter(str(run_dir))
    _init_structured_tb_layout(writer)
    try:
        if eval_rows:
            first = eval_rows[0]
            fixed_summary = {
                "reward_sum": _maybe_float(first, "fixed_reward_sum"),
                "processed_ratio_eval": _maybe_float(first, "fixed_processed_ratio_eval"),
                "drop_ratio_eval": _maybe_float(first, "fixed_drop_ratio_eval"),
                "pre_backlog_steps_eval": _maybe_float(first, "fixed_pre_backlog_steps_eval"),
                "sat_overlap_eval": _maybe_float(first, "fixed_sat_overlap_eval"),
                "collision_episode_fraction": _maybe_float(first, "fixed_collision_episode_fraction"),
            }
            for key, value in fixed_summary.items():
                if value is not None:
                    writer.add_scalar(f"checkpoint_eval/fixed_{key}", float(value), 0)

        for row in train_rows:
            update_idx = int(float(row["update"]))
            total_env_steps = int(update_idx * int(rollout_env_steps) * int(num_envs))
            for key in (
                "env_reward_mean",
                "episodes_finished",
                "policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "clip_frac",
                "danger_imitation_loss",
                "danger_imitation_active_rate",
            ):
                value = _maybe_float(row, key)
                if value is not None:
                    writer.add_scalar(key, float(value), update_idx)
            writer.add_scalar("total_env_steps", float(total_env_steps), update_idx)

        for row in eval_rows:
            update_idx = int(float(row["update"]))
            for key in (
                "reward_sum",
                "processed_ratio_eval",
                "drop_ratio_eval",
                "pre_backlog_steps_eval",
                "sat_overlap_eval",
                "collision_episode_fraction",
            ):
                value = _maybe_float(row, key)
                if value is not None:
                    writer.add_scalar(f"checkpoint_eval/{key}", float(value), update_idx)
                fixed_value = _maybe_float(row, f"fixed_{key}")
                if fixed_value is not None:
                    writer.add_scalar(f"checkpoint_eval/fixed_{key}", float(fixed_value), update_idx)
    finally:
        writer.close()

    print(
        f"Backfilled TensorBoard events for {run_dir} "
        f"(updates={len(train_rows)}, eval_rows={len(eval_rows)}, "
        f"rollout_env_steps={rollout_env_steps}, num_envs={num_envs})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
