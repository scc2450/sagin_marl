from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_eval import evaluate_structured_fixed_policy


def _make_candidate(base_cfg, name: str, **updates):
    return name, replace(base_cfg, **updates)


def _candidate_specs(base_cfg):
    return [
        _make_candidate(base_cfg, "base_v2"),
        _make_candidate(
            base_cfg,
            "hetero_preload",
            arrival_base_hetero=1.0,
            queue_init_gu_steps=4.0,
            gu_init_cluster_std=300.0,
            task_arrival_rate=7.5e5,
            b_acc=5.0e6,
        ),
        _make_candidate(
            base_cfg,
            "hetero_tighter_bw",
            arrival_base_hetero=1.0,
            queue_init_gu_steps=4.0,
            gu_init_cluster_std=320.0,
            task_arrival_rate=7.5e5,
            b_acc=4.0e6,
        ),
        _make_candidate(
            base_cfg,
            "sticky_hotspot_rho6",
            traffic_model="sticky_subset_hotspot",
            arrival_base_hetero=0.7,
            hotspot_num_subsets=3,
            hotspot_subset_size=2,
            hotspot_rho=6.0,
            hotspot_on_mean_steps=18.0,
            hotspot_off_mean_steps=6.0,
            queue_init_gu_steps=4.0,
            gu_init_cluster_std=300.0,
            task_arrival_rate=7.5e5,
            b_acc=5.0e6,
        ),
        _make_candidate(
            base_cfg,
            "sticky_hotspot_rho8_tightbw",
            traffic_model="sticky_subset_hotspot",
            arrival_base_hetero=0.9,
            hotspot_num_subsets=4,
            hotspot_subset_size=2,
            hotspot_rho=8.0,
            hotspot_on_mean_steps=20.0,
            hotspot_off_mean_steps=5.0,
            queue_init_gu_steps=5.0,
            gu_init_cluster_std=320.0,
            task_arrival_rate=8.0e5,
            b_acc=4.0e6,
        ),
        _make_candidate(
            base_cfg,
            "sticky_hotspot_rho10",
            traffic_model="sticky_subset_hotspot",
            arrival_base_hetero=1.0,
            hotspot_num_subsets=4,
            hotspot_subset_size=2,
            hotspot_rho=10.0,
            hotspot_on_mean_steps=20.0,
            hotspot_off_mean_steps=5.0,
            queue_init_gu_steps=6.0,
            gu_init_cluster_std=350.0,
            task_arrival_rate=8.0e5,
            b_acc=4.0e6,
        ),
        _make_candidate(
            base_cfg,
            "sticky_hotspot_reward_amp",
            traffic_model="sticky_subset_hotspot",
            arrival_base_hetero=0.8,
            hotspot_num_subsets=4,
            hotspot_subset_size=2,
            hotspot_rho=8.0,
            hotspot_on_mean_steps=20.0,
            hotspot_off_mean_steps=5.0,
            queue_init_gu_steps=5.0,
            gu_init_cluster_std=320.0,
            task_arrival_rate=8.0e5,
            b_acc=4.0e6,
            reward_w_access=1.0,
            reward_w_pre_backlog=0.2,
        ),
        _make_candidate(
            base_cfg,
            "homogeneous_reward_amp",
            arrival_base_hetero=0.8,
            queue_init_gu_steps=5.0,
            gu_init_cluster_std=320.0,
            task_arrival_rate=8.0e5,
            b_acc=4.0e6,
            reward_w_access=1.0,
            reward_w_pre_backlog=0.2,
        ),
    ]


def _eval_pair(cfg, *, episodes: int, episode_seed_base: int | None) -> dict[str, Any]:
    zero = evaluate_structured_fixed_policy(
        cfg,
        baseline_policy="zero",
        episodes=int(episodes),
        episode_seed_base=episode_seed_base,
    )
    queue = evaluate_structured_fixed_policy(
        cfg,
        baseline_policy="queue_aware",
        episodes=int(episodes),
        episode_seed_base=episode_seed_base,
    )
    return {
        "zero_reward": float(zero["reward_sum"]),
        "zero_processed": float(zero["processed_ratio_eval"]),
        "zero_drop": float(zero["drop_ratio_eval"]),
        "zero_backlog": float(zero["pre_backlog_steps_eval"]),
        "queue_reward": float(queue["reward_sum"]),
        "queue_processed": float(queue["processed_ratio_eval"]),
        "queue_drop": float(queue["drop_ratio_eval"]),
        "queue_backlog": float(queue["pre_backlog_steps_eval"]),
        "reward_gap": float(queue["reward_sum"] - zero["reward_sum"]),
        "processed_gap": float(queue["processed_ratio_eval"] - zero["processed_ratio_eval"]),
        "drop_gap": float(zero["drop_ratio_eval"] - queue["drop_ratio_eval"]),
        "backlog_gap": float(zero["pre_backlog_steps_eval"] - queue["pre_backlog_steps_eval"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--episode_seed_base", type=int, default=52000)
    parser.add_argument("--out_csv", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows: list[dict[str, Any]] = []
    for name, cand_cfg in _candidate_specs(cfg):
        summary = _eval_pair(
            cand_cfg,
            episodes=int(args.episodes),
            episode_seed_base=args.episode_seed_base,
        )
        row = {
            "name": name,
            "traffic_model": str(cand_cfg.traffic_model),
            "task_arrival_rate": float(cand_cfg.task_arrival_rate),
            "arrival_base_hetero": float(cand_cfg.arrival_base_hetero),
            "gu_init_cluster_std": float(cand_cfg.gu_init_cluster_std),
            "queue_init_gu_steps": float(cand_cfg.queue_init_gu_steps or 0.0),
            "b_acc": float(cand_cfg.b_acc),
            "reward_w_access": float(cand_cfg.reward_w_access),
            "reward_w_pre_backlog": float(cand_cfg.reward_w_pre_backlog),
            "hotspot_num_subsets": int(cand_cfg.hotspot_num_subsets),
            "hotspot_subset_size": int(cand_cfg.hotspot_subset_size),
            "hotspot_rho": float(cand_cfg.hotspot_rho),
            "hotspot_on_mean_steps": float(cand_cfg.hotspot_on_mean_steps),
            "hotspot_off_mean_steps": float(cand_cfg.hotspot_off_mean_steps),
        }
        row.update(summary)
        rows.append(row)
        print(
            f"{name}: reward_gap={row['reward_gap']:.3f} "
            f"processed_gap={row['processed_gap']:.4f} "
            f"drop_gap={row['drop_gap']:.4f} "
            f"backlog_gap={row['backlog_gap']:.4f}"
        )

    rows.sort(key=lambda item: float(item["reward_gap"]), reverse=True)
    out_csv = args.out_csv or os.path.join("runs", "_tmp_bw_sanity_gap_search.csv")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["name"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
