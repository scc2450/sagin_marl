#!/usr/bin/env python3
"""Generate Section 5 nominal mechanism figures.

These figures use the nominal x1p00 selected-checkpoint held-out rich metrics
registered on 2026-07-15. The queue and flow figures use only methods with
complete rich nominal coverage. The collision figure uses the formal selected
main table and keeps the same six-method set as the mechanism figures.

Run with:

    /opt/homebrew/Caskroom/miniconda/base/envs/rl/bin/python \
        docs/paper/experiments/generate_section5_nominal_mechanism_figures_20260715.py
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
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
TABLE_DIR = ROOT / "docs/paper/table_sources"
FIGURE_SOURCE_DIR = ROOT / "docs/paper/figure_sources/performance_evaluation_20260714"
MANUSCRIPT_FIGURE_DIR = ROOT / "docs/paper/manuscript/figures"

RICH_SOURCE = TABLE_DIR / "phase4_nominal_selected_rich_aggregate_20260715.csv"
MAIN_SOURCE = TABLE_DIR / "phase4_formal_heldout_source_selected_main_20260713.csv"

METHOD_ORDER = ["STARS", "STARS-GC", "QCCS", "Lyapunov", "QBS", "Uniform"]
COLLISION_ORDER = ["STARS", "STARS-GC", "QCCS", "Lyapunov", "QBS", "Uniform"]

METHOD_COLORS = {
    "STARS": "#0072B2",
    "STARS-GC": "#D55E00",
    "HA-PPO": "#CC79A7",
    "QCCS": "#009E73",
    "Lyapunov": "#E69F00",
    "QBS": "#56B4E9",
    "Uniform": "#666666",
}

MAIN_LABELS = {
    "RelCritic": "STARS",
    "GlobalCritic": "STARS-GC",
    "MAPPO-like": "HA-PPO",
    "cluster_center_queue_aware": "QCCS",
    "maxweight_lyapunov": "Lyapunov",
    "queue_aware_bw": "QBS",
    "static_uniform": "Uniform",
}

METRIC_COLORS = {
    "Total queue": "#4D4D4D",
    "GU queue": "#5DA5DA",
    "UAV queue": "#F17CB0",
    "Access": "#0072B2",
    "Backhaul": "#009E73",
    "Processed": "#E69F00",
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
            "legend.fontsize": 6.4,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "grid.linewidth": 0.45,
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


def _rich_complete() -> pd.DataFrame:
    df = pd.read_csv(RICH_SOURCE)
    df = df[(df["paper_label"].isin(METHOD_ORDER)) & (df["n_rows"] >= 3)].copy()
    df["paper_label"] = pd.Categorical(df["paper_label"], categories=METHOD_ORDER, ordered=True)
    return df.sort_values("paper_label").reset_index(drop=True)


def _add_bar_labels(
    ax: plt.Axes,
    y_values: np.ndarray,
    x_values: np.ndarray,
    *,
    fmt: str,
    pad: float,
    fontsize: float = 6.3,
) -> None:
    for yi, value in zip(y_values, x_values):
        ax.text(
            float(value) + pad,
            float(yi),
            fmt.format(float(value)),
            va="center",
            ha="left",
            fontsize=fontsize,
            color="#333333",
        )


def plot_queue_decomposition() -> None:
    df = _rich_complete()
    labels = df["paper_label"].astype(str).tolist()
    y = np.arange(len(df), dtype=float)
    offsets = [-0.22, 0.0, 0.22]
    metrics = [
        ("Total queue", "queue_total_mbit_mean"),
        ("GU queue", "gu_queue_layer_mbit_mean"),
        ("UAV queue", "uav_queue_layer_mbit_mean"),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(3.45, 2.88), constrained_layout=False)
    for offset, (label, mean_col) in zip(offsets, metrics):
        values = df[mean_col].astype(float).to_numpy()
        ax.barh(
            y + offset,
            values,
            height=0.18,
            color=METRIC_COLORS[label],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.35,
            label=label,
        )
        _add_bar_labels(ax, y + offset, values, fmt="{:.0f}", pad=7.0)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Queue workload (Mbit)")
    ax.set_xlim(0.0, max(df["queue_total_mbit_mean"]) * 1.26)
    ax.grid(axis="x", color="#D9D9D9", alpha=0.72)
    ax.tick_params(axis="both", length=3.0, width=0.65)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        ncol=3,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.16, top=0.80)
    save_pdf(fig, "sec5_nominal_queue_decomposition_20260715")
    plt.close(fig)


def plot_flow_decomposition() -> None:
    df = _rich_complete()
    labels = df["paper_label"].astype(str).tolist()
    y = np.arange(len(df), dtype=float)
    offsets = [-0.22, 0.0, 0.22]
    metrics = [
        ("Access", "outflow_arrival_ratio_mean"),
        ("Backhaul", "sat_incoming_arrival_ratio_mean"),
        ("Processed", "sat_processed_arrival_ratio_mean"),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(3.45, 2.88), constrained_layout=False)
    for offset, (label, mean_col) in zip(offsets, metrics):
        values = df[mean_col].astype(float).to_numpy()
        ax.barh(
            y + offset,
            values,
            height=0.18,
            color=METRIC_COLORS[label],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.35,
            label=label,
        )
        _add_bar_labels(ax, y + offset, values, fmt="{:.3f}", pad=0.010)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Arrival-normalized ratio")
    ax.set_xlim(0.0, 1.08)
    ax.grid(axis="x", color="#D9D9D9", alpha=0.72)
    ax.tick_params(axis="both", length=3.0, width=0.65)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        ncol=3,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.16, top=0.80)
    save_pdf(fig, "sec5_nominal_flow_decomposition_20260715")
    plt.close(fig)


def plot_collision_fraction() -> None:
    df = pd.read_csv(MAIN_SOURCE)
    df["paper_label"] = df["method"].map(MAIN_LABELS)
    df = df[df["paper_label"].isin(COLLISION_ORDER)].copy()
    df["paper_label"] = pd.Categorical(df["paper_label"], categories=COLLISION_ORDER, ordered=True)
    df = df.sort_values("paper_label").reset_index(drop=True)

    labels = df["paper_label"].astype(str).tolist()
    y = np.arange(len(df), dtype=float)
    values = df["Collision_mean"].astype(float).to_numpy()
    colors = [METHOD_COLORS[label] for label in labels]

    fig, ax = plt.subplots(1, 1, figsize=(3.45, 2.88), constrained_layout=False)
    ax.barh(
        y,
        values,
        height=0.48,
        color=colors,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.35,
    )
    for yi, value in zip(y, values):
        ax.text(
            value + 0.0016,
            yi,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=6.8,
            color="#333333",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Collision episode fraction")
    ax.set_xlim(0.0, max(0.050, values.max() * 1.28))
    ax.grid(axis="x", color="#D9D9D9", alpha=0.72)
    ax.tick_params(axis="both", length=3.0, width=0.65)
    fig.subplots_adjust(left=0.20, right=0.955, bottom=0.16, top=0.96)
    save_pdf(fig, "sec5_collision_episode_fraction_20260715")
    plt.close(fig)


def main() -> None:
    configure_style()
    plot_queue_decomposition()
    plot_flow_decomposition()
    plot_collision_fraction()


if __name__ == "__main__":
    main()
