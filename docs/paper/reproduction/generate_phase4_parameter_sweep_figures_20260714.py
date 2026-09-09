"""Generate formal no-HA-PPO parameter-sweep figures for Section 5.

The script consumes the registered aggregate CSV under docs/paper/evidence_tables.
It keeps manuscript-facing outputs under docs/paper/manuscript_overleaf/figures
and may write inspection copies under docs/paper/reproduction/generated_figures.
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
TABLE_DIR = ROOT / "docs/paper/evidence_tables"
FIG_SRC_DIR = ROOT / "docs/paper/reproduction/generated_figures/performance_evaluation_20260714"
FIG_MANUSCRIPT_DIR = ROOT / "docs/paper/manuscript_overleaf/figures"

SOURCE = TABLE_DIR / "phase4_formal_parameter_sweeps_nohappo_aggregate_20260714.csv"

METHOD_ORDER = [
    "relcritic",
    "globalcritic",
    "cluster_center_queue_aware",
    "maxweight_lyapunov",
    "queue_aware_bw",
    "static_uniform",
]

LABELS = {
    "relcritic": "STARS",
    "globalcritic": "STARS-GC",
    "cluster_center_queue_aware": "QCCS",
    "maxweight_lyapunov": "Lyapunov",
    "queue_aware_bw": "QBS",
    "static_uniform": "Uniform",
}

COLORS = {
    "STARS": "#0072B2",
    "STARS-GC": "#D55E00",
    "QCCS": "#009E73",
    "Lyapunov": "#E69F00",
    "QBS": "#56B4E9",
    "Uniform": "#666666",
}

LINE_STYLES = {
    "STARS": "-",
    "STARS-GC": "--",
    "QCCS": "-",
    "Lyapunov": "--",
    "QBS": ":",
    "Uniform": ":",
}

MARKERS = {
    "STARS": "o",
    "STARS-GC": "s",
    "QCCS": "D",
    "Lyapunov": "p",
    "QBS": "v",
    "Uniform": "x",
}

METRICS = [
    ("reward_sum_mean", "reward_sum_std", "Reward", None),
    ("processed_ratio_eval_mean", "processed_ratio_eval_std", "Processed ratio", (0.0, 1.04)),
    ("drop_ratio_eval_mean", "drop_ratio_eval_std", "Drop ratio", (0.0, 0.62)),
    ("D_sys_report_mean", "D_sys_report_std", "System delay", None),
]

METRIC_SLUGS = {
    "Reward": "reward",
    "Processed ratio": "processed",
    "Drop ratio": "drop",
    "System delay": "dsys",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_all(fig: plt.Figure, basename: str) -> None:
    FIG_SRC_DIR.mkdir(parents=True, exist_ok=True)
    FIG_MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        src = FIG_SRC_DIR / f"{basename}.{ext}"
        dst = FIG_MANUSCRIPT_DIR / f"{basename}.{ext}"
        fig.savefig(src, bbox_inches="tight", dpi=300)
        shutil.copyfile(src, dst)


def _method_rows(df: pd.DataFrame, method_id: str) -> pd.DataFrame:
    sub = df[df["method_id"] == method_id].copy()
    return sub.sort_values("point_index")


def _plot_metric_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: np.ndarray,
    mean_col: str,
    std_col: str,
    title: str,
    ylim: tuple[float, float] | None,
) -> None:
    for method_id in METHOD_ORDER:
        sub = _method_rows(df, method_id)
        label = LABELS[method_id]
        means = sub[mean_col].astype(float).to_numpy()
        stds = sub[std_col].astype(float).fillna(0.0).to_numpy()
        ax.errorbar(
            x,
            means,
            yerr=stds,
            label=label,
            color=COLORS[label],
            linestyle=LINE_STYLES[label],
            marker=MARKERS[label],
            markersize=4.2,
            linewidth=1.45 if label != "STARS" else 1.9,
            elinewidth=0.55,
            capsize=1.6,
            alpha=0.92 if label != "STARS-GC" else 0.78,
            zorder=4 if label == "STARS" else 3,
        )
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#D8D8D8", alpha=0.7)
    ax.tick_params(axis="both", length=3, width=0.65)


def _shared_legend(fig: plt.Figure, axes: np.ndarray) -> None:
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=6,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.0,
        handletextpad=0.45,
    )


def _set_sweep_ticks(ax: plt.Axes, x: np.ndarray) -> None:
    ax.set_xticks(x)
    ax.set_xticklabels([f"{value:g}" for value in x])


def _metric_by_title(title: str) -> tuple[str, str, str, tuple[float, float] | None]:
    for metric in METRICS:
        if metric[2] == title:
            return metric
    raise ValueError(f"unknown metric title: {title}")


def plot_single_metric_figures(
    sub: pd.DataFrame,
    x: np.ndarray,
    xlabel: str,
    basename_prefix: str,
) -> None:
    """Export one metric per figure for flexible LaTeX subfloat composition."""
    for mean_col, std_col, title, ylim in METRICS:
        slug = METRIC_SLUGS[title]
        fig, ax = plt.subplots(1, 1, figsize=(3.55, 2.55))
        _plot_metric_panel(ax, sub, x, mean_col, std_col, title, ylim)
        ax.set_title("")
        _set_sweep_ticks(ax, x)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(title)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.28),
            ncol=3,
            frameon=False,
            columnspacing=0.9,
            handlelength=1.7,
            handletextpad=0.35,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        save_all(fig, f"{basename_prefix}_{slug}_single_20260714")
        plt.close(fig)


def plot_metric_pair_figure(
    sub: pd.DataFrame,
    x: np.ndarray,
    xlabel: str,
    metric_titles: tuple[str, str],
    basename: str,
) -> None:
    """Export the recommended paper-facing two-panel sensitivity figure."""
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55), sharex=True)
    for ax, metric_title in zip(axes, metric_titles):
        mean_col, std_col, title, ylim = _metric_by_title(metric_title)
        _plot_metric_panel(ax, sub, x, mean_col, std_col, title, ylim)
        _set_sweep_ticks(ax, x)
    axes[0].set_ylabel("Value")
    fig.supxlabel(xlabel, y=0.02)
    _shared_legend(fig, axes)
    fig.tight_layout(rect=(0, 0.08, 1, 0.88))
    save_all(fig, basename)
    plt.close(fig)


def plot_load_sensitivity(df: pd.DataFrame) -> None:
    sub = df[df["sweep"] == "load"].copy()
    x = (
        sub[sub["method_id"] == METHOD_ORDER[0]]
        .sort_values("point_index")["task_arrival_rate_mbit_per_gu_slot"]
        .astype(float)
        .to_numpy()
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.65), sharex=True)
    for ax, (mean_col, std_col, title, ylim) in zip(axes.ravel(), METRICS):
        _plot_metric_panel(ax, sub, x, mean_col, std_col, title, ylim)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{value:g}" for value in x])
    fig.supxlabel("Task arrival rate (Mbit/GU/slot)", y=0.02)
    axes[0, 0].set_ylabel("Value")
    axes[1, 0].set_ylabel("Value")
    _shared_legend(fig, axes)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save_all(fig, "perf_eval_offered_load_sensitivity_nohappo_20260714")
    plt.close(fig)

    plot_single_metric_figures(
        sub,
        x,
        "Task arrival rate (Mbit/GU/slot)",
        "perf_eval_offered_load",
    )
    plot_metric_pair_figure(
        sub,
        x,
        "Task arrival rate (Mbit/GU/slot)",
        ("Processed ratio", "Drop ratio"),
        "perf_eval_offered_load_processed_drop_pair_20260714",
    )


def plot_resource_sensitivity(df: pd.DataFrame) -> None:
    sub = df[df["sweep"] == "resource"].copy()
    ref = sub[sub["method_id"] == METHOD_ORDER[0]].sort_values("point_index")
    x = ref["b_acc_mhz"].astype(float).to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.65), sharex=True)
    for ax, (mean_col, std_col, title, ylim) in zip(axes.ravel(), METRICS):
        _plot_metric_panel(ax, sub, x, mean_col, std_col, title, ylim)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{value:g}" for value in x])
    fig.supxlabel(r"Access bandwidth $B_U$ (MHz)", y=0.02)
    axes[0, 0].set_ylabel("Value")
    axes[1, 0].set_ylabel("Value")
    _shared_legend(fig, axes)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save_all(fig, "perf_eval_resource_capacity_sensitivity_nohappo_20260714")
    plt.close(fig)

    plot_single_metric_figures(
        sub,
        x,
        r"Access bandwidth $B_U$ (MHz)",
        "perf_eval_resource_capacity",
    )
    plot_metric_pair_figure(
        sub,
        x,
        r"Access bandwidth $B_U$ (MHz)",
        ("Processed ratio", "System delay"),
        "perf_eval_resource_capacity_processed_dsys_pair_20260714",
    )


def main() -> None:
    configure_style()
    df = pd.read_csv(SOURCE)
    df["point_index"] = df["point_index"].astype(int)
    for method_id in METHOD_ORDER:
        if method_id not in set(df["method_id"]):
            raise ValueError(f"missing method in {SOURCE}: {method_id}")
    plot_load_sensitivity(df)
    plot_resource_sensitivity(df)


if __name__ == "__main__":
    main()
