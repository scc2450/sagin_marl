#!/usr/bin/env python3
"""Export publication-oriented single-panel PDF figures for Section 5.

This script reads only registered evidence tables under docs/paper/table_sources.
It writes all generated panels to figure_sources, and copies the selected
manuscript-facing panels to docs/paper/manuscript/figures.

Use the local paper plotting environment:

    /opt/homebrew/Caskroom/miniconda/base/envs/rl/bin/python \
        docs/paper/experiments/generate_section5_single_panel_figures_20260714.py
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
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

SWEEP_SOURCE = TABLE_DIR / "phase4_formal_parameter_sweeps_nohappo_aggregate_20260714.csv"
STARS_CURVE_SOURCE = TABLE_DIR / "phase4_stars_training_checkpoint_curves_10seed_20260714.csv"
QCCS_REF_SOURCE = TABLE_DIR / "phase4_qccs_checkpoint_validation_ref_20260714_summary.json"

METHOD_ORDER = [
    "relcritic",
    "globalcritic",
    "cluster_center_queue_aware",
    "maxweight_lyapunov",
    "queue_aware_bw",
    "static_uniform",
]
CORE_QUEUE_METHOD_ORDER = [
    "relcritic",
    "cluster_center_queue_aware",
    "maxweight_lyapunov",
]

LABELS = {
    "relcritic": "STARS",
    "globalcritic": "STARS-GC",
    "cluster_center_queue_aware": "QCCS",
    "maxweight_lyapunov": "Lyapunov",
    "queue_aware_bw": "QBS",
    "static_uniform": "Uniform",
}

# Okabe-Ito inspired, fixed across Section 5.
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
    "Lyapunov": "P",
    "QBS": "v",
    "Uniform": "x",
}


@dataclass(frozen=True)
class SweepPanel:
    sweep: str
    x_col: str
    x_label: str
    mean_col: str
    std_col: str
    y_label: str
    basename: str
    y_lim: tuple[float, float] | None = None
    y_scale: float = 1.0
    manuscript: bool = True
    method_order: tuple[str, ...] = tuple(METHOD_ORDER)


SELECTED_SWEEP_PANELS = [
    SweepPanel(
        sweep="load",
        x_col="task_arrival_rate_mbit_per_gu_slot",
        x_label="Task arrival rate (Mbit/GU/slot)",
        mean_col="reward_sum_mean",
        std_col="reward_sum_std",
        y_label="Reward",
        basename="sec5_load_reward_20260714",
    ),
    SweepPanel(
        sweep="load",
        x_col="task_arrival_rate_mbit_per_gu_slot",
        x_label="Task arrival rate (Mbit/GU/slot)",
        mean_col="processed_ratio_eval_mean",
        std_col="processed_ratio_eval_std",
        y_label="Processed ratio",
        basename="sec5_load_processed_ratio_20260714",
        y_lim=(0.0, 1.04),
    ),
    SweepPanel(
        sweep="load",
        x_col="task_arrival_rate_mbit_per_gu_slot",
        x_label="Task arrival rate (Mbit/GU/slot)",
        mean_col="drop_ratio_eval_mean",
        std_col="drop_ratio_eval_std",
        y_label="Drop ratio",
        basename="sec5_load_drop_ratio_20260714",
        y_lim=(0.0, 0.66),
    ),
    SweepPanel(
        sweep="load",
        x_col="task_arrival_rate_mbit_per_gu_slot",
        x_label="Task arrival rate (Mbit/GU/slot)",
        mean_col="queue_total_mean_mean",
        std_col="queue_total_mean_std",
        y_label="Queue workload (Mbit)",
        basename="sec5_load_queue_workload_20260715",
        y_scale=1e-6,
    ),
    SweepPanel(
        sweep="resource",
        x_col="b_acc_mhz",
        x_label=r"Access bandwidth $B_U$ (MHz)",
        mean_col="reward_sum_mean",
        std_col="reward_sum_std",
        y_label="Reward",
        basename="sec5_resource_reward_20260714",
    ),
    SweepPanel(
        sweep="resource",
        x_col="b_acc_mhz",
        x_label=r"Access bandwidth $B_U$ (MHz)",
        mean_col="processed_ratio_eval_mean",
        std_col="processed_ratio_eval_std",
        y_label="Processed ratio",
        basename="sec5_resource_processed_ratio_20260714",
        y_lim=(0.0, 1.04),
    ),
    SweepPanel(
        sweep="resource",
        x_col="b_acc_mhz",
        x_label=r"Access bandwidth $B_U$ (MHz)",
        mean_col="D_sys_report_mean",
        std_col="D_sys_report_std",
        y_label=r"System delay $D_{\mathrm{sys}}$",
        basename="sec5_resource_system_delay_20260714",
    ),
    SweepPanel(
        sweep="resource",
        x_col="b_acc_mhz",
        x_label=r"Access bandwidth $B_U$ (MHz)",
        mean_col="queue_total_mean_mean",
        std_col="queue_total_mean_std",
        y_label="Queue workload (Mbit)",
        basename="sec5_resource_queue_workload_20260715",
        y_scale=1e-6,
    ),
]

APPENDIX_SWEEP_PANELS = [
    SweepPanel(
        sweep="load",
        x_col="task_arrival_rate_mbit_per_gu_slot",
        x_label="Task arrival rate (Mbit/GU/slot)",
        mean_col="D_sys_report_mean",
        std_col="D_sys_report_std",
        y_label=r"System delay $D_{\mathrm{sys}}$",
        basename="sec5_appendix_load_system_delay_20260714",
        manuscript=False,
    ),
    SweepPanel(
        sweep="resource",
        x_col="b_acc_mhz",
        x_label=r"Access bandwidth $B_U$ (MHz)",
        mean_col="drop_ratio_eval_mean",
        std_col="drop_ratio_eval_std",
        y_label="Drop ratio",
        basename="sec5_appendix_resource_drop_ratio_20260714",
        y_lim=(0.0, 0.66),
        manuscript=False,
    ),
]

CORE_QUEUE_SWEEP_PANELS = [
    SweepPanel(
        sweep="load",
        x_col="task_arrival_rate_mbit_per_gu_slot",
        x_label="Task arrival rate (Mbit/GU/slot)",
        mean_col="queue_total_mean_mean",
        std_col="queue_total_mean_std",
        y_label="Queue workload (Mbit)",
        basename="sec5_load_queue_workload_core_methods_20260715",
        y_scale=1e-6,
        method_order=tuple(CORE_QUEUE_METHOD_ORDER),
    ),
    SweepPanel(
        sweep="resource",
        x_col="b_acc_mhz",
        x_label=r"Access bandwidth $B_U$ (MHz)",
        mean_col="queue_total_mean_mean",
        std_col="queue_total_mean_std",
        y_label="Queue workload (Mbit)",
        basename="sec5_resource_queue_workload_core_methods_20260715",
        y_scale=1e-6,
        method_order=tuple(CORE_QUEUE_METHOD_ORDER),
    ),
]


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


def save_all(fig: plt.Figure, basename: str, *, manuscript: bool) -> None:
    FIGURE_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    source = FIGURE_SOURCE_DIR / f"{basename}.pdf"
    fig.savefig(source)
    if manuscript:
        shutil.copyfile(source, MANUSCRIPT_FIGURE_DIR / source.name)


def _method_rows(df: pd.DataFrame, method_id: str) -> pd.DataFrame:
    return df[df["method_id"] == method_id].sort_values("point_index")


def _add_method_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.32),
        ncol=3,
        frameon=False,
        handlelength=1.45,
        handletextpad=0.30,
        columnspacing=0.72,
        borderaxespad=0.0,
    )


def _set_ticks(ax: plt.Axes, values: np.ndarray) -> None:
    ax.set_xticks(values)
    ax.set_xticklabels([f"{value:g}" for value in values])


def plot_sweep_panel(df: pd.DataFrame, panel: SweepPanel) -> None:
    sub = df[df["sweep"] == panel.sweep].copy()
    ref = _method_rows(sub, panel.method_order[0])
    x_values = ref[panel.x_col].astype(float).to_numpy()

    fig, ax = plt.subplots(1, 1, figsize=(3.45, 2.72), constrained_layout=False)
    for method_id in panel.method_order:
        rows = _method_rows(sub, method_id)
        label = LABELS[method_id]
        means = rows[panel.mean_col].astype(float).to_numpy() * panel.y_scale
        stds = rows[panel.std_col].astype(float).fillna(0.0).to_numpy() * panel.y_scale
        ax.errorbar(
            x_values,
            means,
            yerr=stds,
            label=label,
            color=COLORS[label],
            linestyle=LINE_STYLES[label],
            marker=MARKERS[label],
            markersize=3.4,
            linewidth=1.65 if label == "STARS" else 1.15,
            elinewidth=0.50,
            capsize=1.5,
            capthick=0.50,
            markeredgewidth=0.65,
            alpha=0.95 if label != "STARS-GC" else 0.78,
            zorder=5 if label == "STARS" else 3,
        )

    if panel.y_lim is not None:
        ax.set_ylim(*panel.y_lim)
    _set_ticks(ax, x_values)
    ax.set_xlabel(panel.x_label)
    ax.set_ylabel(panel.y_label)
    ax.grid(axis="y", color="#D9D9D9", alpha=0.72)
    ax.tick_params(axis="both", length=3.0, width=0.65)
    _add_method_legend(ax)
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.19, top=0.74)
    save_all(fig, panel.basename, manuscript=panel.manuscript)
    plt.close(fig)


def _qccs_reward() -> float | None:
    if not QCCS_REF_SOURCE.exists():
        return None
    payload = json.loads(QCCS_REF_SOURCE.read_text(encoding="utf-8"))
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
        rewards = group.set_index("update")["reward_sum"].reindex(updates)
        best_so_far = rewards.ffill().cummax()
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


def _aggregate_reward(panel_df: pd.DataFrame) -> pd.DataFrame:
    return (
        panel_df.groupby("update", as_index=False)["reward_sum"]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .sort_values("update")
    )


def plot_checkpoint_panel(
    panel_df: pd.DataFrame,
    basename: str,
    *,
    qccs_reward: float | None,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(3.45, 2.68), constrained_layout=False)

    for i, (_seed, group) in enumerate(panel_df.groupby("training_seed", sort=True)):
        group = group.sort_values("update")
        ax.plot(
            group["update"],
            group["reward_sum"],
            color="#7A9BB7",
            linewidth=0.50,
            alpha=0.24,
            label="Seeds" if i == 0 else "_nolegend_",
            zorder=1,
        )

    agg = _aggregate_reward(panel_df)
    ax.plot(
        agg["update"],
        agg["median"],
        color=COLORS["STARS"],
        linewidth=1.65,
        label="Median",
        zorder=4,
    )
    err = agg[agg["update"].isin({125, 225, 325, 425, 525})].copy()
    ax.errorbar(
        err["update"].to_numpy(dtype=float),
        err["median"].to_numpy(dtype=float),
        yerr=np.vstack(
            [
                (err["median"] - err["q25"]).to_numpy(dtype=float),
                (err["q75"] - err["median"]).to_numpy(dtype=float),
            ]
        ),
        fmt="o",
        color=COLORS["STARS"],
        ecolor=COLORS["STARS"],
        elinewidth=0.70,
        capsize=2.0,
        capthick=0.70,
        markersize=2.9,
        markerfacecolor="white",
        markeredgewidth=0.8,
        label="IQR",
        zorder=5,
    )
    if qccs_reward is not None:
        ax.axhline(
            qccs_reward,
            color=COLORS["QCCS"],
            linestyle=(0, (4, 2)),
            linewidth=1.0,
            label="QCCS",
            zorder=2,
        )

    ax.set_xlim(20, 535)
    ax.set_ylim(25, 75)
    ax.set_xticks([25, 125, 225, 325, 425, 525])
    ax.set_xlabel("Training update")
    ax.set_ylabel("Validation reward")
    ax.grid(axis="y", color="#D9D9D9", alpha=0.72)
    ax.tick_params(axis="both", length=3.0, width=0.65)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=4,
        frameon=False,
        handlelength=1.35,
        columnspacing=0.62,
        handletextpad=0.30,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.18, top=0.78)
    save_all(fig, basename, manuscript=True)
    plt.close(fig)


def plot_checkpoint_figures() -> None:
    curve_df = pd.read_csv(STARS_CURVE_SOURCE)
    curve_df = curve_df[curve_df["update"] <= 525].copy()
    qccs_reward = _qccs_reward()
    plot_checkpoint_panel(
        curve_df,
        "sec5_stars_raw_checkpoint_reward_20260714",
        qccs_reward=qccs_reward,
    )
    plot_checkpoint_panel(
        _best_so_far_df(curve_df),
        "sec5_stars_best_so_far_reward_20260714",
        qccs_reward=qccs_reward,
    )


def main() -> None:
    configure_style()
    sweep_df = pd.read_csv(SWEEP_SOURCE)
    sweep_df["point_index"] = sweep_df["point_index"].astype(int)
    missing_methods = set(METHOD_ORDER) - set(sweep_df["method_id"])
    if missing_methods:
        raise ValueError(f"missing methods in {SWEEP_SOURCE}: {sorted(missing_methods)}")

    for panel in SELECTED_SWEEP_PANELS + APPENDIX_SWEEP_PANELS + CORE_QUEUE_SWEEP_PANELS:
        plot_sweep_panel(sweep_df, panel)
    plot_checkpoint_figures()


if __name__ == "__main__":
    main()
