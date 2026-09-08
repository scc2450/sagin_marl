#!/usr/bin/env python3
"""Generate the Section 5 critic-ablation convergence panel.

The figure uses the registered 10-seed best-so-far checkpoint-validation table
and compares the relational critic against the global-critic ablation. It does
not include the QCCS reference line, so the panel reads as a direct learned
critic ablation.

Run with:

    /opt/homebrew/Caskroom/miniconda/base/envs/rl/bin/python \
        docs/paper/reproduction/generate_section5_critic_ablation_convergence_20260715.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sagin_marl_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sagin_marl_matplotlib")
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
TABLE_DIR = ROOT / "docs/paper/evidence_tables"
FIGURE_SOURCE_DIR = ROOT / "docs/paper/reproduction/generated_figures/performance_evaluation_20260714"
MANUSCRIPT_FIGURE_DIR = ROOT / "docs/paper/manuscript_overleaf/figures"

SOURCE = TABLE_DIR / "phase4_critic_convergence_best_so_far_10seed_20260714.csv"
OUT_STEM = "sec5_critic_ablation_best_so_far_reward_10seed_20260715"

METHOD_ORDER = ["STARS", "STARS-GC"]

COLORS = {
    "STARS": "#0072B2",
    "STARS-GC": "#D55E00",
}

LINE_STYLES = {
    "STARS": "-",
    "STARS-GC": "--",
}

MARKERS = {
    "STARS": "o",
    "STARS-GC": "s",
}

IQR_UPDATES = {125, 225, 325, 425, 525}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 7.4,
            "axes.labelsize": 7.8,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "grid.linewidth": 0.45,
            "lines.linewidth": 1.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["method", "update"], as_index=False)["best_reward"]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
            n="count",
        )
        .sort_values(["method", "update"])
        .reset_index(drop=True)
    )


def save_pdf(fig: plt.Figure) -> None:
    FIGURE_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    source = FIGURE_SOURCE_DIR / f"{OUT_STEM}.pdf"
    fig.savefig(source)
    shutil.copyfile(source, MANUSCRIPT_FIGURE_DIR / source.name)
    print(source)
    print(MANUSCRIPT_FIGURE_DIR / source.name)


def plot() -> None:
    df = pd.read_csv(SOURCE)
    df = df[df["method"].isin(METHOD_ORDER)].copy()
    agg = aggregate(df)

    fig, ax = plt.subplots(1, 1, figsize=(3.45, 2.72), constrained_layout=False)
    for method in METHOD_ORDER:
        seed_rows = df[df["method"] == method].copy()
        rows = agg[agg["method"] == method].sort_values("update")
        color = COLORS[method]
        x = rows["update"].astype(float).to_numpy()
        median = rows["median"].astype(float).to_numpy()
        q25 = rows["q25"].astype(float).to_numpy()
        q75 = rows["q75"].astype(float).to_numpy()

        for index, (_seed, group) in enumerate(seed_rows.groupby("training_seed", sort=True)):
            group = group.sort_values("update")
            ax.plot(
                group["update"],
                group["best_reward"],
                color=color,
                linewidth=0.42,
                alpha=0.16,
                label=f"{method} seeds" if index == 0 else "_nolegend_",
                zorder=1,
            )

        ax.plot(
            x,
            median,
            label=method,
            color=color,
            linestyle=LINE_STYLES[method],
            marker=MARKERS[method],
            markevery=4,
            markersize=3.1,
            linewidth=1.65 if method == "STARS" else 1.25,
            markeredgewidth=0.65,
            alpha=0.96 if method == "STARS" else 0.86,
            zorder=5 if method == "STARS" else 4,
        )
        err = rows[rows["update"].isin(IQR_UPDATES)].copy()
        ax.errorbar(
            err["update"].astype(float).to_numpy(),
            err["median"].astype(float).to_numpy(),
            yerr=[
                (err["median"] - err["q25"]).astype(float).to_numpy(),
                (err["q75"] - err["median"]).astype(float).to_numpy(),
            ],
            fmt=MARKERS[method],
            color=color,
            ecolor=color,
            elinewidth=0.70,
            capsize=2.0,
            capthick=0.70,
            markersize=3.0,
            markerfacecolor="white",
            markeredgewidth=0.80,
            label=f"{method} IQR",
            zorder=6 if method == "STARS" else 5,
        )

    ax.set_xlim(20, 530)
    ax.set_xticks([25, 125, 225, 325, 425, 525])
    ax.set_ylim(25, 75)
    ax.set_xlabel("Training update")
    ax.set_ylabel("Best-so-far validation reward")
    ax.grid(axis="y", color="#D9D9D9", alpha=0.72)
    ax.tick_params(axis="both", length=3.0, width=0.65)
    handles, labels = ax.get_legend_handles_labels()
    keep = [index for index, label in enumerate(labels) if label in METHOD_ORDER]
    ax.legend(
        [handles[index] for index in keep],
        [labels[index] for index in keep],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        ncol=2,
        frameon=False,
        handlelength=1.55,
        handletextpad=0.35,
        columnspacing=0.85,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.19, top=0.80)
    save_pdf(fig)
    plt.close(fig)

    summary = agg[agg["update"].isin([25, 125, 225, 325, 425, 525])]
    print(summary[["method", "update", "median", "q25", "q75", "n"]].to_string(index=False))


def main() -> None:
    configure_style()
    plot()


if __name__ == "__main__":
    main()
