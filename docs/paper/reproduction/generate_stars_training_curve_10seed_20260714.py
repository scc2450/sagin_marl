#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = Path("/tmp/stars_training_curves_20260714")
TABLE_DIR = REPO_ROOT / "docs" / "paper" / "evidence_tables"
FIGURE_SOURCE_DIR = (
    REPO_ROOT / "docs" / "paper" / "reproduction" / "generated_figures" / "performance_evaluation_20260714"
)
MANUSCRIPT_FIGURE_DIR = REPO_ROOT / "docs" / "paper" / "manuscript_overleaf" / "figures"

CURVE_CSV = TABLE_DIR / "phase4_stars_training_checkpoint_curves_10seed_20260714.csv"
SUMMARY_CSV = TABLE_DIR / "phase4_stars_training_checkpoint_summary_10seed_20260714.csv"
QCCS_REF_JSON = TABLE_DIR / "phase4_qccs_checkpoint_validation_ref_20260714_summary.json"
FIGURE_STEM = "phase4_stars_training_curve_10seed_20260714"

STARS_BLUE = "#0072B2"
QCCS_GREEN = "#009E73"


def _seed_from_name(path: Path) -> int:
    match = re.search(r"seed(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot parse seed from {path.name}")
    return int(match.group(1))


def _run_label(path: Path) -> str:
    return path.name.removesuffix("__checkpoint_eval.csv")


def load_curves() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"{SOURCE_DIR} does not exist; extract /tmp/stars_training_curves_20260714.tgz first"
        )

    frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for csv_path in sorted(SOURCE_DIR.glob("*__checkpoint_eval.csv")):
        seed = _seed_from_name(csv_path)
        run = _run_label(csv_path)
        frame = pd.read_csv(csv_path)
        frame["method"] = "STARS"
        frame["training_seed"] = seed
        frame["run_label"] = run
        frames.append(frame)

        stop_path = SOURCE_DIR / f"{run}__training_stop.json"
        stop = json.loads(stop_path.read_text()) if stop_path.exists() else {}
        summary_rows.append(
            {
                "method": "STARS",
                "training_seed": seed,
                "run_label": run,
                "has_training_stop": stop_path.exists(),
                "completed_updates": stop.get("completed_updates", np.nan),
                "stop_reason": stop.get("stop_reason", ""),
            }
        )

    if not frames:
        raise RuntimeError(f"No checkpoint_eval CSV files found in {SOURCE_DIR}")

    curve_df = pd.concat(frames, ignore_index=True)
    curve_df["update"] = pd.to_numeric(curve_df["update"], errors="coerce").astype(int)
    for col in [
        "reward_sum",
        "processed_ratio_eval",
        "drop_ratio_eval",
        "pre_backlog_steps_eval",
        "D_sys_report",
    ]:
        curve_df[col] = pd.to_numeric(curve_df[col], errors="coerce")

    # A resumed run contributes a later segment for the same seed. If any update
    # overlaps, keep the last segment in lexical run order so the resumed segment
    # wins.
    curve_df = (
        curve_df.sort_values(["training_seed", "update", "run_label"])
        .drop_duplicates(["training_seed", "update"], keep="last")
        .sort_values(["training_seed", "update"])
        .reset_index(drop=True)
    )

    summary_df = pd.DataFrame(summary_rows)
    best_rows = (
        curve_df.loc[curve_df.groupby("training_seed")["reward_sum"].idxmax()]
        .sort_values("training_seed")
        .reset_index(drop=True)
    )
    final_rows = (
        curve_df.sort_values(["training_seed", "update"])
        .groupby("training_seed", as_index=False)
        .tail(1)
        .sort_values("training_seed")
        .reset_index(drop=True)
    )
    summary_df = summary_df.sort_values(["training_seed", "run_label"]).reset_index(drop=True)
    seed_summary = (
        summary_df.groupby("training_seed", as_index=False)
        .agg(
            run_labels=("run_label", ";".join),
            has_training_stop=("has_training_stop", "max"),
            completed_updates=("completed_updates", "max"),
            stop_reason=("stop_reason", lambda values: ";".join(v for v in values.astype(str) if v)),
        )
        .merge(
            best_rows[
                [
                    "training_seed",
                    "update",
                    "reward_sum",
                    "processed_ratio_eval",
                    "drop_ratio_eval",
                    "D_sys_report",
                ]
            ].rename(
                columns={
                    "update": "selected_update",
                    "reward_sum": "selected_reward_sum",
                    "processed_ratio_eval": "selected_processed_ratio",
                    "drop_ratio_eval": "selected_drop_ratio",
                    "D_sys_report": "selected_D_sys",
                }
            ),
            on="training_seed",
            how="left",
        )
        .merge(
            final_rows[
                [
                    "training_seed",
                    "update",
                    "reward_sum",
                    "processed_ratio_eval",
                    "drop_ratio_eval",
                    "D_sys_report",
                ]
            ].rename(
                columns={
                    "update": "final_update",
                    "reward_sum": "final_reward_sum",
                    "processed_ratio_eval": "final_processed_ratio",
                    "drop_ratio_eval": "final_drop_ratio",
                    "D_sys_report": "final_D_sys",
                }
            ),
            on="training_seed",
            how="left",
        )
        .sort_values("training_seed")
        .reset_index(drop=True)
    )
    return curve_df, seed_summary


def plot(curve_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    FIGURE_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.9, 2.9))

    plot_df = curve_df[curve_df["update"] <= 525].copy()
    aggregate = (
        plot_df.groupby("update", as_index=False)
        .agg(
            median_reward=("reward_sum", "median"),
            q25_reward=("reward_sum", lambda values: values.quantile(0.25)),
            q75_reward=("reward_sum", lambda values: values.quantile(0.75)),
            n=("reward_sum", "count"),
        )
        .sort_values("update")
    )
    aggregate_plot = aggregate[aggregate["n"] >= 4].copy()
    ax.plot(
        aggregate_plot["update"],
        aggregate_plot["median_reward"],
        color=STARS_BLUE,
        linewidth=2.0,
        marker="o",
        markersize=3.0,
    )
    ax.fill_between(
        aggregate_plot["update"].to_numpy(dtype=float),
        aggregate_plot["q25_reward"].to_numpy(dtype=float),
        aggregate_plot["q75_reward"].to_numpy(dtype=float),
        color=STARS_BLUE,
        alpha=0.18,
        linewidth=0,
    )

    qccs_reward = None
    if QCCS_REF_JSON.exists():
        payload = json.loads(QCCS_REF_JSON.read_text(encoding="utf-8"))
        qccs_reward = float(payload["summary"]["reward_sum"])
        ax.axhline(
            qccs_reward,
            color=QCCS_GREEN,
            linestyle=(0, (4, 2)),
            linewidth=1.3,
        )
        ax.text(500, qccs_reward + 1.0, "QCCS", color=QCCS_GREEN, fontsize=7.2, ha="right", va="bottom")
    ax.text(500, 61.5, "STARS", color=STARS_BLUE, fontsize=7.6, ha="right", va="center")

    ax.set_xlabel("Training update")
    ax.set_ylabel("Validation reward")
    ax.set_xlim(0, 525)
    ax.set_xticks([25, 125, 225, 325, 425, 525])
    ax.set_ylim(25, 76)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    for out_dir in [FIGURE_SOURCE_DIR, MANUSCRIPT_FIGURE_DIR]:
        fig.savefig(out_dir / f"{FIGURE_STEM}.pdf")
        fig.savefig(out_dir / f"{FIGURE_STEM}.png", dpi=300)
        fig.savefig(out_dir / f"{FIGURE_STEM}.svg")
    plt.close(fig)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    curve_df, summary_df = load_curves()
    curve_df.to_csv(CURVE_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    plot(curve_df, summary_df)

    print(f"wrote {CURVE_CSV}")
    print(f"wrote {SUMMARY_CSV}")
    print(f"wrote {MANUSCRIPT_FIGURE_DIR / (FIGURE_STEM + '.pdf')}")
    print(f"seeds={summary_df['training_seed'].nunique()}")


if __name__ == "__main__":
    main()
