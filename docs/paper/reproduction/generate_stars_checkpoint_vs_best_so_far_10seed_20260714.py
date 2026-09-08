#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
TABLE_DIR = REPO_ROOT / "docs" / "paper" / "evidence_tables"
FIGURE_SOURCE_DIR = (
    REPO_ROOT / "docs" / "paper" / "reproduction" / "generated_figures" / "performance_evaluation_20260714"
)
MANUSCRIPT_FIGURE_DIR = REPO_ROOT / "docs" / "paper" / "manuscript_overleaf" / "figures"

CURVE_CSV = TABLE_DIR / "phase4_stars_training_checkpoint_curves_10seed_20260714.csv"
QCCS_REF_JSON = TABLE_DIR / "phase4_qccs_checkpoint_validation_ref_20260714_summary.json"
OUT_STEM = "phase4_stars_checkpoint_vs_best_so_far_10seed_20260714"

STARS_BLUE = "#0072B2"
QCCS_GREEN = "#009E73"
SEED_TRACE = "#7A9BB7"

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


def _qccs_reward() -> float | None:
    if not QCCS_REF_JSON.exists():
        return None
    payload = json.loads(QCCS_REF_JSON.read_text(encoding="utf-8"))
    return float(payload["summary"]["reward_sum"])


def _best_so_far_df(curve_df: pd.DataFrame) -> pd.DataFrame:
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
                    "training_seed": int(seed),
                    "update": int(update),
                    "reward_sum": float(reward),
                }
            )
    return pd.DataFrame(rows)


def _agg(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("update", as_index=False)["reward_sum"]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
            n="count",
        )
        .sort_values("update")
    )


def _plot_panel(
    ax: plt.Axes,
    panel_df: pd.DataFrame,
    title: str,
    panel_label: str,
    qccs_reward: float | None,
    *,
    error_updates: set[int],
) -> None:
    for seed_index, (_seed, group) in enumerate(panel_df.groupby("training_seed", sort=True)):
        ax.plot(
            group["update"],
            group["reward_sum"],
            color=SEED_TRACE,
            linewidth=0.6,
            alpha=0.22,
            label="Individual seeds" if seed_index == 0 else "_nolegend_",
            zorder=1,
        )

    agg = _agg(panel_df)
    ax.plot(
        agg["update"],
        agg["median"],
        color=STARS_BLUE,
        linewidth=1.9,
        label="Median",
        zorder=3,
    )
    err_df = agg[agg["update"].isin(error_updates)].copy()
    ax.errorbar(
        err_df["update"].to_numpy(dtype=float),
        err_df["median"].to_numpy(dtype=float),
        yerr=np.vstack(
            [
                (err_df["median"] - err_df["q25"]).to_numpy(dtype=float),
                (err_df["q75"] - err_df["median"]).to_numpy(dtype=float),
            ]
        ),
        fmt="o",
        color=STARS_BLUE,
        ecolor=STARS_BLUE,
        elinewidth=0.85,
        capsize=2.0,
        capthick=0.85,
        markersize=3.0,
        markerfacecolor="white",
        markeredgewidth=0.85,
        label="IQR",
        zorder=4,
    )
    if qccs_reward is not None:
        ax.axhline(
            qccs_reward,
            color=QCCS_GREEN,
            linestyle=(0, (4, 2)),
            linewidth=1.05,
        label="QCCS",
        )
        ax.text(535, qccs_reward + 0.8, "QCCS", color=QCCS_GREEN, fontsize=6.8, ha="right")

    ax.text(
        0.03,
        0.93,
        f"{panel_label} {title}",
        transform=ax.transAxes,
        color="#222222",
        fontsize=8.0,
        ha="left",
        va="top",
    )
    ax.set_xlim(20, 550)
    ax.set_xticks([25, 125, 225, 325, 425, 525])
    ax.set_ylim(25, 75)
    ax.grid(axis="y", color="#dddddd", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Training update")


def main() -> None:
    curve_df = pd.read_csv(CURVE_CSV)
    curve_df = curve_df[curve_df["update"] <= 525].copy()
    best_df = _best_so_far_df(curve_df)
    qccs_reward = _qccs_reward()

    fig, axes = plt.subplots(1, 2, figsize=(6.95, 2.65), sharey=True)
    _plot_panel(
        axes[0],
        curve_df,
        "Raw checkpoint",
        "(a)",
        qccs_reward,
        error_updates={125, 225, 325, 425, 525},
    )
    _plot_panel(
        axes[1],
        best_df,
        "Best-so-far",
        "(b)",
        qccs_reward,
        error_updates={125, 225, 325, 425, 525},
    )
    axes[0].set_ylabel("Validation reward")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    ordered_labels = ["Individual seeds", "Median", "IQR", "QCCS"]
    fig.legend(
        [by_label[label] for label in ordered_labels],
        ordered_labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.52, 1.00),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    FIGURE_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for out_dir in [FIGURE_SOURCE_DIR, MANUSCRIPT_FIGURE_DIR]:
        fig.savefig(out_dir / f"{OUT_STEM}.pdf")
        fig.savefig(out_dir / f"{OUT_STEM}.png", dpi=300)
        fig.savefig(out_dir / f"{OUT_STEM}.svg")
    plt.close(fig)
    print(f"wrote {MANUSCRIPT_FIGURE_DIR / (OUT_STEM + '.pdf')}")


if __name__ == "__main__":
    main()
