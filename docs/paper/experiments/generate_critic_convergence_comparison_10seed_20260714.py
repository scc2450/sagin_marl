#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
TABLE_DIR = REPO_ROOT / "docs" / "paper" / "table_sources"
FIGURE_SOURCE_DIR = (
    REPO_ROOT / "docs" / "paper" / "figure_sources" / "performance_evaluation_20260714"
)
MANUSCRIPT_FIGURE_DIR = REPO_ROOT / "docs" / "paper" / "manuscript" / "figures"

STARS_CURVE_CSV = TABLE_DIR / "phase4_stars_training_checkpoint_curves_10seed_20260714.csv"
STARS_SUMMARY_CSV = TABLE_DIR / "phase4_stars_training_checkpoint_summary_10seed_20260714.csv"
STARS_GC_SOURCE_DIR = Path("/tmp/stars_gc_training_curves_20260714")
STARS_GC_CURVE_CSV = TABLE_DIR / "phase4_stars_gc_training_checkpoint_curves_10seed_20260714.csv"
STARS_GC_SUMMARY_CSV = TABLE_DIR / "phase4_stars_gc_training_checkpoint_summary_10seed_20260714.csv"
BEST_SO_FAR_CSV = TABLE_DIR / "phase4_critic_convergence_best_so_far_10seed_20260714.csv"
QCCS_REF_JSON = TABLE_DIR / "phase4_qccs_checkpoint_validation_ref_20260714_summary.json"

COMBINED_STEM = "phase4_critic_convergence_best_so_far_combined_10seed_20260714"
PANELS_STEM = "phase4_critic_convergence_best_so_far_panels_10seed_20260714"

COLORS = {
    "STARS": "#0072B2",
    "STARS-GC": "#D55E00",
    "QCCS": "#009E73",
}

plt.rcParams.update(
    {
        "font.size": 7.5,
        "axes.labelsize": 8.2,
        "axes.titlesize": 8.2,
        "legend.fontsize": 7.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _seed_from_name(path: Path) -> int:
    match = re.search(r"seed(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot parse seed from {path.name}")
    return int(match.group(1))


def _run_label(path: Path) -> str:
    return path.name.removesuffix("__checkpoint_eval.csv")


def _load_gc_curves() -> tuple[pd.DataFrame, pd.DataFrame]:
    if STARS_GC_CURVE_CSV.exists() and STARS_GC_SUMMARY_CSV.exists():
        return pd.read_csv(STARS_GC_CURVE_CSV), pd.read_csv(STARS_GC_SUMMARY_CSV)
    if not STARS_GC_SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"{STARS_GC_SOURCE_DIR} does not exist; extract /tmp/stars_gc_training_curves_20260714.tgz first"
        )

    frames: list[pd.DataFrame] = []
    stop_rows: list[dict[str, object]] = []
    for csv_path in sorted(STARS_GC_SOURCE_DIR.glob("*__checkpoint_eval.csv")):
        seed = _seed_from_name(csv_path)
        run = _run_label(csv_path)
        frame = pd.read_csv(csv_path)
        frame["method"] = "STARS-GC"
        frame["training_seed"] = seed
        frame["run_label"] = run
        frames.append(frame)

        stop_path = STARS_GC_SOURCE_DIR / f"{run}__training_stop.json"
        stop = json.loads(stop_path.read_text()) if stop_path.exists() else {}
        stop_rows.append(
            {
                "method": "STARS-GC",
                "training_seed": seed,
                "run_label": run,
                "has_training_stop": stop_path.exists(),
                "completed_updates": stop.get("completed_updates", np.nan),
                "stop_reason": stop.get("stop_reason", ""),
            }
        )

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
    curve_df = (
        curve_df.sort_values(["training_seed", "update", "run_label"])
        .drop_duplicates(["training_seed", "update"], keep="last")
        .sort_values(["training_seed", "update"])
        .reset_index(drop=True)
    )

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
    stop_df = pd.DataFrame(stop_rows)
    summary_df = (
        stop_df.groupby("training_seed", as_index=False)
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
    curve_df.to_csv(STARS_GC_CURVE_CSV, index=False)
    summary_df.to_csv(STARS_GC_SUMMARY_CSV, index=False)
    return curve_df, summary_df


def _best_so_far(curve_df: pd.DataFrame, method: str) -> pd.DataFrame:
    updates = np.arange(25, 526, 25)
    rows: list[dict[str, object]] = []
    for seed in sorted(curve_df["training_seed"].unique()):
        group = (
            curve_df[curve_df["training_seed"] == seed]
            .sort_values("update")[["update", "reward_sum"]]
            .drop_duplicates("update", keep="last")
        )
        series = group.set_index("update")["reward_sum"].reindex(updates)
        best_so_far = series.ffill().cummax().ffill()
        for update, reward in best_so_far.items():
            if pd.isna(reward):
                continue
            rows.append(
                {
                    "method": method,
                    "training_seed": int(seed),
                    "update": int(update),
                    "best_reward": float(reward),
                }
            )
    return pd.DataFrame(rows)


def _aggregate(best_df: pd.DataFrame) -> pd.DataFrame:
    return (
        best_df.groupby(["method", "update"], as_index=False)["best_reward"]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
            n="count",
        )
        .sort_values(["method", "update"])
        .reset_index(drop=True)
    )


def _qccs_reward() -> float | None:
    if not QCCS_REF_JSON.exists():
        return None
    payload = json.loads(QCCS_REF_JSON.read_text(encoding="utf-8"))
    return float(payload["summary"]["reward_sum"])


def _style_axes(ax: plt.Axes) -> None:
    ax.set_xlim(20, 525)
    ax.set_xticks([25, 125, 225, 325, 425, 525])
    ax.set_ylim(25, 75)
    ax.grid(axis="y", color="#dddddd", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_method(ax: plt.Axes, agg: pd.DataFrame, method: str, *, label_prefix: str = "") -> None:
    data = agg[agg["method"] == method]
    color = COLORS[method]
    ax.fill_between(
        data["update"].to_numpy(dtype=float),
        data["q25"].to_numpy(dtype=float),
        data["q75"].to_numpy(dtype=float),
        color=color,
        alpha=0.13,
        linewidth=0,
    )
    ax.plot(
        data["update"],
        data["median"],
        color=color,
        linewidth=1.8,
        marker="o",
        markevery=2,
        markersize=2.8,
    )


def _save(fig: plt.Figure, stem: str) -> None:
    FIGURE_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for out_dir in [FIGURE_SOURCE_DIR, MANUSCRIPT_FIGURE_DIR]:
        fig.savefig(out_dir / f"{stem}.pdf")
        fig.savefig(out_dir / f"{stem}.png", dpi=300)
        fig.savefig(out_dir / f"{stem}.svg")
    plt.close(fig)


def plot_combined(agg: pd.DataFrame, qccs_reward: float | None) -> None:
    fig, ax = plt.subplots(figsize=(4.65, 2.55))
    _plot_method(ax, agg, "STARS")
    _plot_method(ax, agg, "STARS-GC")
    if qccs_reward is not None:
        ax.axhline(
            qccs_reward,
            color=COLORS["QCCS"],
            linestyle=(0, (4, 2)),
            linewidth=1.15,
        )
        ax.text(505, qccs_reward + 0.9, "QCCS", color=COLORS["QCCS"], fontsize=7.1, ha="right")
    _style_axes(ax)
    ax.set_xlabel("Training update")
    ax.set_ylabel("Best-so-far validation reward")

    stars_tail = agg[(agg["method"] == "STARS") & (agg["update"] == 525)]["median"].iloc[0]
    gc_tail = agg[(agg["method"] == "STARS-GC") & (agg["update"] == 525)]["median"].iloc[0]
    ax.text(505, stars_tail + 1.2, "STARS", color=COLORS["STARS"], fontsize=7.4, ha="right")
    ax.text(505, gc_tail + 1.1, "STARS-GC", color=COLORS["STARS-GC"], fontsize=7.4, ha="right")
    fig.tight_layout()
    _save(fig, COMBINED_STEM)


def plot_panels(agg: pd.DataFrame, qccs_reward: float | None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.45), sharex=True, sharey=True)
    for ax, method, panel in zip(axes, ["STARS", "STARS-GC"], ["(a)", "(b)"], strict=True):
        _plot_method(ax, agg, method)
        if qccs_reward is not None:
            ax.axhline(
                qccs_reward,
                color=COLORS["QCCS"],
                linestyle=(0, (4, 2)),
                linewidth=1.1,
            )
            ax.text(500, qccs_reward + 0.9, "QCCS", color=COLORS["QCCS"], fontsize=7.0, ha="right")
        _style_axes(ax)
        ax.text(
            0.03,
            0.93,
            f"{panel} {method}",
            transform=ax.transAxes,
            color=COLORS[method],
            fontsize=8.0,
            va="top",
            ha="left",
        )
        ax.set_xlabel("Training update")
    axes[0].set_ylabel("Best-so-far validation reward")

    fig.tight_layout()
    _save(fig, PANELS_STEM)


def main() -> None:
    stars_curve = pd.read_csv(STARS_CURVE_CSV)
    _ = pd.read_csv(STARS_SUMMARY_CSV)
    gc_curve, _gc_summary = _load_gc_curves()
    best_df = pd.concat(
        [
            _best_so_far(stars_curve, "STARS"),
            _best_so_far(gc_curve, "STARS-GC"),
        ],
        ignore_index=True,
    )
    best_df.to_csv(BEST_SO_FAR_CSV, index=False)
    agg = _aggregate(best_df)
    qccs_reward = _qccs_reward()
    plot_combined(agg, qccs_reward)
    plot_panels(agg, qccs_reward)
    print(f"wrote {MANUSCRIPT_FIGURE_DIR / (COMBINED_STEM + '.pdf')}")
    print(f"wrote {MANUSCRIPT_FIGURE_DIR / (PANELS_STEM + '.pdf')}")
    print(agg.groupby('method')['n'].max().to_dict())


if __name__ == "__main__":
    main()
