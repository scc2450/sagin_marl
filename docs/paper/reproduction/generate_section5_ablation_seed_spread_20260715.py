#!/usr/bin/env python3
"""Generate Section 5 critic-ablation seed-spread PDF panels.

Use the local paper plotting environment:

    /opt/homebrew/Caskroom/miniconda/base/envs/rl/bin/python \
        docs/paper/reproduction/generate_section5_ablation_seed_spread_20260715.py
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
TABLE_DIR = ROOT / "docs/paper/evidence_tables"
FIGURE_SOURCE_DIR = ROOT / "docs/paper/reproduction/generated_figures/performance_evaluation_20260714"
MANUSCRIPT_FIGURE_DIR = ROOT / "docs/paper/manuscript_overleaf/figures"

STARS_SUMMARY = TABLE_DIR / "phase4_stars_training_checkpoint_summary_10seed_20260714.csv"
STARS_GC_SUMMARY = TABLE_DIR / "phase4_stars_gc_training_checkpoint_summary_10seed_20260714.csv"
QCCS_REF = TABLE_DIR / "phase4_qccs_checkpoint_validation_ref_20260714_summary.json"

COLORS = {
    "STARS": "#0072B2",
    "STARS-GC": "#D55E00",
    "QCCS": "#009E73",
}
MARKERS = {
    "STARS": "o",
    "STARS-GC": "s",
}


@dataclass(frozen=True)
class PanelSpec:
    value_col: str
    y_label: str
    basename: str
    qccs_key: str
    y_lim: tuple[float, float] | None = None
    lower_is_better: bool = False


PANELS = [
    PanelSpec(
        value_col="selected_reward_sum",
        y_label="Selected reward",
        basename="sec5_ablation_selected_reward_seed_spread_20260715",
        qccs_key="reward_sum",
        y_lim=(20.0, 75.0),
    ),
    PanelSpec(
        value_col="selected_D_sys",
        y_label=r"System delay $D_{\mathrm{sys}}$",
        basename="sec5_ablation_selected_dsys_seed_spread_20260715",
        qccs_key="D_sys_report",
        y_lim=(0.0, 115.0),
        lower_is_better=True,
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
            "legend.fontsize": 6.4,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "grid.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_data() -> pd.DataFrame:
    stars = pd.read_csv(STARS_SUMMARY)
    stars["method"] = "STARS"
    gc = pd.read_csv(STARS_GC_SUMMARY)
    gc["method"] = "STARS-GC"
    return pd.concat([stars, gc], ignore_index=True)


def qccs_summary() -> dict[str, float]:
    payload = json.loads(QCCS_REF.read_text(encoding="utf-8"))
    return {key: float(value) for key, value in payload["summary"].items() if isinstance(value, (int, float))}


def save_pdf(fig: plt.Figure, basename: str) -> None:
    FIGURE_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    source = FIGURE_SOURCE_DIR / f"{basename}.pdf"
    fig.savefig(source)
    shutil.copyfile(source, MANUSCRIPT_FIGURE_DIR / source.name)
    plt.close(fig)


def plot_panel(df: pd.DataFrame, spec: PanelSpec, qccs: dict[str, float]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(3.45, 2.72), constrained_layout=False)
    x_positions = {"STARS": 0.0, "STARS-GC": 1.0}
    legend_handles = []

    for method in ["STARS", "STARS-GC"]:
        sub = df[df["method"] == method].copy().sort_values("training_seed")
        values = sub[spec.value_col].astype(float).to_numpy()
        x0 = x_positions[method]
        jitter = np.linspace(-0.085, 0.085, len(values))
        ax.scatter(
            x0 + jitter,
            values,
            s=19,
            marker=MARKERS[method],
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.86,
            zorder=4,
            label=method,
        )
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        ax.vlines(
            x0,
            q25,
            q75,
            color=COLORS[method],
            linewidth=1.3,
            zorder=3,
        )
        ax.hlines(
            median,
            x0 - 0.22,
            x0 + 0.22,
            color=COLORS[method],
            linewidth=2.0,
            zorder=5,
        )
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=COLORS[method],
                marker=MARKERS[method],
                linestyle="",
                markersize=4.0,
                label=method,
            )
        )

    qccs_value = qccs.get(spec.qccs_key)
    if qccs_value is not None:
        ax.axhline(
            qccs_value,
            color=COLORS["QCCS"],
            linestyle=(0, (4, 2)),
            linewidth=1.05,
            zorder=2,
            label="QCCS",
        )
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=COLORS["QCCS"],
                linestyle=(0, (4, 2)),
                linewidth=1.05,
                label="QCCS",
            )
        )

    ax.set_xlim(-0.5, 1.5)
    if spec.y_lim is not None:
        ax.set_ylim(*spec.y_lim)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["STARS", "STARS-GC"])
    ax.set_ylabel(spec.y_label)
    ax.grid(axis="y", color="#D9D9D9", alpha=0.72)
    ax.tick_params(axis="both", length=3.0, width=0.65)

    note = "Lower is better" if spec.lower_is_better else "Higher is better"
    ax.text(
        0.02,
        0.97,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        color="#444444",
    )
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=3,
        frameon=False,
        handlelength=1.25,
        handletextpad=0.32,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.16, top=0.80)
    save_pdf(fig, spec.basename)


def main() -> None:
    configure_style()
    df = load_data()
    qccs = qccs_summary()
    for spec in PANELS:
        plot_panel(df, spec, qccs)


if __name__ == "__main__":
    main()
