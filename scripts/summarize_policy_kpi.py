from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _load_series(csv_path: Path, key: str) -> np.ndarray:
    values: list[float] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get(key, "")
            if raw is None or raw == "":
                continue
            values.append(float(raw))
    if not values:
        raise ValueError(f"No numeric values for '{key}' in {csv_path}")
    return np.asarray(values, dtype=np.float64)


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="Pairs in the form label=path/to/eval.csv",
    )
    args = parser.parse_args()

    pairs: list[tuple[str, Path]] = []
    for item in args.input:
        if "=" not in item:
            raise ValueError(f"Invalid --input entry: {item}. Expected label=path")
        label, path = item.split("=", 1)
        pairs.append((label, Path(path)))

    print(
        "policy,episodes,"
        "reward_sum_mean,"
        "collision_rate_mean,"
        "near_collision_ratio_mean,"
        "min_inter_uav_dist_mean,min_inter_uav_dist_min,"
        "queue_total_active_mean,queue_total_active_p95,queue_total_active_p99,"
        "outflow_arrival_ratio_mean,outflow_arrival_ratio_p05,"
        "drop_ratio_mean"
    )
    for label, csv_path in pairs:
        reward = _load_series(csv_path, "reward_sum")
        collision = _load_series(csv_path, "collision")
        near_collision = _load_series(csv_path, "near_collision_ratio")
        min_dist = _load_series(csv_path, "min_inter_uav_dist")
        q = _load_series(csv_path, "queue_total_active")
        ratio = _load_series(csv_path, "outflow_arrival_ratio")
        drop = _load_series(csv_path, "drop_ratio")
        print(
            ",".join(
                [
                    label,
                    str(int(q.size)),
                    _fmt(float(np.mean(reward))),
                    _fmt(float(np.mean(collision))),
                    _fmt(float(np.mean(near_collision))),
                    _fmt(float(np.mean(min_dist))),
                    _fmt(float(np.min(min_dist))),
                    _fmt(float(np.mean(q))),
                    _fmt(float(np.percentile(q, 95))),
                    _fmt(float(np.percentile(q, 99))),
                    _fmt(float(np.mean(ratio))),
                    _fmt(float(np.percentile(ratio, 5))),
                    _fmt(float(np.mean(drop))),
                ]
            )
        )


if __name__ == "__main__":
    main()
