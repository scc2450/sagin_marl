#!/usr/bin/env python3
"""Generate PDF-only UAV-density safety smoke figures for Section 5 triage.

Run with:

    /opt/homebrew/Caskroom/miniconda/base/envs/rl/bin/python \
        docs/paper/reproduction/generate_uav_density_collision_smoke_figure_20260715.py
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

SOURCE = TABLE_DIR / "phase4_uav_density_collision_smoke_aggregate_20260715.csv"

METHOD_ORDER = ["STARS", "STARS-GC", "QCCS", "Lyapunov"]

COLORS = {
    "STARS": "#0072B2",
    "STARS-GC": "#D55E00",
    "QCCS": "#009E73",
    "Lyapunov": "#E69F00",
}

LINE_STYLES = {
    "STARS": "-",
    "STARS-GC": "--",
    "QCCS": "-",
    "Lyapunov": "--",
}

MARKERS = {
    "STARS": "o",
    "STARS-GC": "s",
    "QCCS": "D",
    "Lyapunov": "P",
}


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
            "legend.fontsize": 6.2,
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


def save_pdf(fig: plt.Figure, basename: str) -> None:
    FIGURE_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    source = FIGURE_SOURCE_DIR / f"{basename}.pdf"
    fig.savefig(source)
    shutil.copyfile(source, MANUSCRIPT_FIGURE_DIR / source.name)
    print(source)


def _add_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.30),
        ncol=4,
        frameon=False,
        handlelength=1.45,
        handletextpad=0.30,
        columnspacing=0.68,
        borderaxespad=0.0,
    )


def plot_panel(
    df: pd.DataFrame,
    *,
    mean_col: str,
    std_col: str,
    y_label: str,
    basename: str,
    y_lim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(3.45, 2.72), constrained_layout=False)
    for method in METHOD_ORDER:
        rows = df[df["paper_label"] == method].sort_values("num_uav")
        x = rows["num_uav"].astype(float).to_numpy()
        y = rows[mean_col].astype(float).to_numpy()
        yerr = rows[std_col].astype(float).fillna(0.0).to_numpy()
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            label=method,
            color=COLORS[method],
            linestyle=LINE_STYLES[method],
            marker=MARKERS[method],
            markersize=3.4,
            linewidth=1.65 if method == "STARS" else 1.15,
            elinewidth=0.50,
            capsize=1.5,
            capthick=0.50,
            markeredgewidth=0.65,
            alpha=0.95 if method != "STARS-GC" else 0.78,
            zorder=5 if method == "STARS" else 3,
        )

    ax.set_xlabel(r"Number of UAVs $N_{\mathrm{U}}$")
    ax.set_ylabel(y_label)
    ax.set_xticks([2, 3, 4, 5, 6])
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.grid(axis="y", color="#D9D9D9", alpha=0.72)
    ax.tick_params(axis="both", length=3.0, width=0.65)
    _add_legend(ax)
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.19, top=0.76)
    save_pdf(fig, basename)
    plt.close(fig)


def main() -> None:
    configure_style()
    df = pd.read_csv(SOURCE)
    df = df[df["paper_label"].isin(METHOD_ORDER)].copy()
    plot_panel(
        df,
        mean_col="collision_episode_fraction_mean",
        std_col="collision_episode_fraction_std",
        y_label="Collision episode fraction",
        basename="sec5_uav_density_collision_fraction_20260715",
        y_lim=(0.0, 1.06),
    )
    plot_panel(
        df,
        mean_col="episode_length_mean",
        std_col="episode_length_std",
        y_label="Average safe runtime (slots)",
        basename="sec5_uav_density_safe_runtime_20260715",
        y_lim=(0.0, 260.0),
    )


if __name__ == "__main__":
    main()
