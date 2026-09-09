#!/usr/bin/env python3
"""Aggregate nominal phase4 selected-eval rich metrics.

Input is the nominal_load_x1p00 summary tree copied from the formal parameter
sweep.  The nominal point uses the same selected checkpoints, held-out seed
bases, and episode count as the Section 5 source-scenario comparison, while
retaining queue-layer and flow-decomposition metrics that were not registered
in the first compact main table.

Run with:

    /opt/homebrew/Caskroom/miniconda/base/envs/rl/bin/python \
        docs/paper/reproduction/aggregate_phase4_nominal_rich_metrics_20260715.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
TABLE_DIR = ROOT / "docs/paper/evidence_tables"
SOURCE_DIR = TABLE_DIR / "phase4_nominal_selected_rich_summaries_20260715/nominal_load_x1p00"
RAW_OUT = TABLE_DIR / "phase4_nominal_selected_rich_raw_20260715.csv"
AGG_OUT = TABLE_DIR / "phase4_nominal_selected_rich_aggregate_20260715.csv"

NUM_GU = 20
NUM_UAV = 3
NUM_SAT = 144

METHOD_LABELS = {
    "relcritic": "STARS",
    "globalcritic": "STARS-GC",
    "mappo_like": "HA-PPO",
    "cluster_center_queue_aware": "QCCS",
    "maxweight_lyapunov": "Lyapunov",
    "queue_aware_bw": "QBS",
    "static_uniform": "Uniform",
}

METHOD_ORDER = {
    "relcritic": 0,
    "globalcritic": 1,
    "mappo_like": 2,
    "cluster_center_queue_aware": 3,
    "maxweight_lyapunov": 4,
    "queue_aware_bw": 5,
    "static_uniform": 6,
}

SUMMARY_METRICS = [
    "reward_sum",
    "processed_ratio_eval",
    "drop_ratio_eval",
    "pre_backlog_steps_eval",
    "D_sys_report",
    "gu_queue_mean",
    "uav_queue_mean",
    "sat_queue_mean",
    "queue_total_mean",
    "arrival_sum",
    "outflow_sum",
    "backhaul_sum",
    "sat_processed_sum",
    "outflow_arrival_ratio",
    "sat_incoming_arrival_ratio",
    "sat_processed_arrival_ratio",
    "sat_processed_incoming_ratio",
    "drop_ratio",
    "active_drop_ratio",
    "gu_drop_ratio",
    "uav_drop_ratio",
    "sat_drop_ratio",
    "collision_episode_fraction",
    "episode_length",
    "terminated_early",
]


def _parse_path(path: Path) -> dict[str, object]:
    rel = path.relative_to(SOURCE_DIR)
    parts = rel.parts
    if parts[0] == "learned":
        method_id = parts[1]
        seed = int(parts[2].removeprefix("seed"))
        checkpoint = parts[3]
        seed_base = int(parts[4].removeprefix("seedbase"))
        return {
            "method_family": "learned",
            "method_id": method_id,
            "paper_label": METHOD_LABELS.get(method_id, method_id),
            "training_seed": seed,
            "checkpoint": checkpoint,
            "eval_seed_base": seed_base,
        }
    if parts[0] == "fixed":
        method_id = parts[1]
        seed_base = int(parts[2].removeprefix("seedbase"))
        return {
            "method_family": "fixed",
            "method_id": method_id,
            "paper_label": METHOD_LABELS.get(method_id, method_id),
            "training_seed": "",
            "checkpoint": "fixed",
            "eval_seed_base": seed_base,
        }
    raise ValueError(f"Cannot parse summary path: {path}")


def _read_raw() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(SOURCE_DIR.rglob("*_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        row = {
            **_parse_path(path),
            "label": payload.get("label", ""),
            "episodes": int(payload.get("episodes", summary.get("episodes", 0))),
            "num_envs": int(payload.get("num_envs", 0)),
            "summary_json": str(path.relative_to(ROOT)),
        }
        for metric in SUMMARY_METRICS:
            row[metric] = summary.get(metric, np.nan)

        # Per-node queue means are emitted by the evaluator. Convert them to
        # layer-total Mbit so the figure explains where the system backlog sits.
        row["gu_queue_layer_mbit"] = float(row["gu_queue_mean"]) * NUM_GU / 1.0e6
        row["uav_queue_layer_mbit"] = float(row["uav_queue_mean"]) * NUM_UAV / 1.0e6
        row["sat_queue_layer_mbit"] = float(row["sat_queue_mean"]) * NUM_SAT / 1.0e6
        row["queue_total_mbit"] = float(row["queue_total_mean"]) / 1.0e6
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No summary JSON files found under {SOURCE_DIR}")

    df = pd.DataFrame(rows)
    df["_method_order"] = df["method_id"].map(METHOD_ORDER).fillna(999).astype(int)
    return df.sort_values(
        ["_method_order", "training_seed", "eval_seed_base"],
        na_position="last",
    ).drop(columns=["_method_order"])


def _aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = SUMMARY_METRICS + [
        "gu_queue_layer_mbit",
        "uav_queue_layer_mbit",
        "sat_queue_layer_mbit",
        "queue_total_mbit",
    ]
    rows: list[dict[str, object]] = []
    for (method_id, paper_label, method_family), group in raw.groupby(
        ["method_id", "paper_label", "method_family"],
        sort=False,
    ):
        row: dict[str, object] = {
            "method_id": method_id,
            "paper_label": paper_label,
            "method_family": method_family,
            "n_rows": len(group),
            "n_training_seeds": group["training_seed"].replace("", np.nan).dropna().nunique(),
            "n_eval_seed_bases": group["eval_seed_base"].nunique(),
            "episodes_per_row": int(group["episodes"].iloc[0]),
            "total_episodes": int(group["episodes"].sum()),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = values.mean() if len(values) else np.nan
            row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
        rows.append(row)

    out = pd.DataFrame(rows)
    out["_method_order"] = out["method_id"].map(METHOD_ORDER).fillna(999).astype(int)
    return out.sort_values("_method_order").drop(columns=["_method_order"])


def main() -> None:
    raw = _read_raw()
    agg = _aggregate(raw)
    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_OUT, index=False)
    agg.to_csv(AGG_OUT, index=False)
    print(f"raw rows={len(raw)} -> {RAW_OUT}")
    print(f"aggregate rows={len(agg)} -> {AGG_OUT}")
    print(
        agg[
            [
                "paper_label",
                "n_rows",
                "gu_queue_layer_mbit_mean",
                "uav_queue_layer_mbit_mean",
                "sat_queue_layer_mbit_mean",
                "outflow_arrival_ratio_mean",
                "sat_incoming_arrival_ratio_mean",
                "sat_processed_arrival_ratio_mean",
                "collision_episode_fraction_mean",
                "episode_length_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
