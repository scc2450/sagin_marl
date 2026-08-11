from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRIC_COLUMNS = (
    "reward_sum",
    "processed_ratio_eval",
    "drop_ratio_eval",
    "pre_backlog_steps_eval",
    "D_sys_report",
    "queue_total_mean",
    "collision_episode_fraction",
    "near_collision_ratio",
    "near_collision_steps",
    "terminated_early",
    "sat_overlap_eval",
    "episode_length",
    "arrival_step_mean",
    "outflow_arrival_ratio",
)

IDENTITY_COLUMNS = (
    "sweep",
    "point_index",
    "num_uav",
    "num_gu",
    "map_size_m",
    "uav_density_per_km2",
    "method_id",
    "paper_label",
    "method_family",
)

RAW_COLUMNS = (
    "event",
    "rc",
    "sweep",
    "point_index",
    "num_uav",
    "num_gu",
    "map_size_m",
    "uav_density_per_km2",
    "method_id",
    "paper_label",
    "method_family",
    "training_seed",
    "checkpoint_role",
    "eval_seed_base",
    "episodes",
    "num_envs",
    "label",
    "config_path",
    "checkpoint_path",
    "baseline_policy",
    "out_dir",
) + METRIC_COLUMNS

METHOD_ORDER = {
    "relcritic": 0,
    "globalcritic": 1,
    "cluster_center_queue_aware": 2,
    "maxweight_lyapunov": 3,
}


def _float(row: dict[str, str], key: str) -> float:
    value = str(row.get(key, "")).strip()
    if not value:
        return math.nan
    return float(value)


def _sort_key(row: dict[str, str]) -> tuple[object, ...]:
    return (
        int(float(row.get("num_uav", "0") or 0)),
        METHOD_ORDER.get(row.get("method_id", ""), 999),
        str(row.get("training_seed", "")),
        int(float(row.get("eval_seed_base", "0") or 0)),
    )


def _read_completed(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_labels: set[str] = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("event") not in {"end", "skip_existing"}:
                    continue
                if str(row.get("reward_sum", "")).strip() == "":
                    continue
                label = str(row.get("label", ""))
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                rows.append(row)
    rows.sort(key=_sort_key)
    return rows


def _write_raw(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in RAW_COLUMNS})


def _group_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row.get(key, "")) for key in IDENTITY_COLUMNS)


def _write_aggregate(rows: list[dict[str, str]], path: Path) -> None:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row)].append(row)

    fieldnames = list(IDENTITY_COLUMNS) + [
        "n_rows",
        "n_training_seeds",
        "n_eval_seed_bases",
        "episodes_per_eval_seed_base",
        "total_episodes",
    ]
    for metric in METRIC_COLUMNS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])

    out_rows: list[dict[str, object]] = []
    for key, group_rows in groups.items():
        out: dict[str, object] = {col: value for col, value in zip(IDENTITY_COLUMNS, key)}
        training_seeds = {r.get("training_seed", "") for r in group_rows if r.get("training_seed", "")}
        eval_seed_bases = {r.get("eval_seed_base", "") for r in group_rows if r.get("eval_seed_base", "")}
        episodes_values = {_float(r, "episodes") for r in group_rows}
        episodes_per_seed = next(iter(episodes_values)) if len(episodes_values) == 1 else math.nan
        out["n_rows"] = len(group_rows)
        out["n_training_seeds"] = len(training_seeds)
        out["n_eval_seed_bases"] = len(eval_seed_bases)
        out["episodes_per_eval_seed_base"] = episodes_per_seed
        out["total_episodes"] = (
            int(episodes_per_seed * len(group_rows)) if not math.isnan(episodes_per_seed) else ""
        )
        for metric in METRIC_COLUMNS:
            values = [_float(r, metric) for r in group_rows]
            values = [v for v in values if not math.isnan(v)]
            out[f"{metric}_mean"] = mean(values) if values else ""
            out[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        out_rows.append(out)

    out_rows.sort(
        key=lambda row: (
            int(float(str(row.get("num_uav", "0") or 0))),
            METHOD_ORDER.get(str(row.get("method_id", "")), 999),
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-csv", action="append", required=True)
    parser.add_argument(
        "--raw-out",
        default="docs/paper/table_sources/phase4_uav_density_collision_smoke_raw_20260715.csv",
    )
    parser.add_argument(
        "--aggregate-out",
        default="docs/paper/table_sources/phase4_uav_density_collision_smoke_aggregate_20260715.csv",
    )
    args = parser.parse_args()

    rows = _read_completed([Path(item) for item in args.status_csv])
    _write_raw(rows, Path(args.raw_out))
    _write_aggregate(rows, Path(args.aggregate_out))
    print(f"completed_rows={len(rows)} raw={args.raw_out} aggregate={args.aggregate_out}")


if __name__ == "__main__":
    main()
