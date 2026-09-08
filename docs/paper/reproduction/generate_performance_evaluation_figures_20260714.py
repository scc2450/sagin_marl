"""Generate draft Performance Evaluation figures for the paper.

The script only uses registered table-source CSV files under docs/paper.
It writes reproducible figure sources plus LaTeX-ready copies.
"""

from __future__ import annotations

import shutil
import os
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
EXP_DIR = ROOT / "docs/paper/reproduction"

MAIN_SOURCE = TABLE_DIR / "phase4_formal_heldout_source_selected_main_20260713.csv"
SEED_SOURCE = TABLE_DIR / "phase4_formal_heldout_source_learned_seed_rows_20260713.csv"
CURVE_SOURCE = TABLE_DIR / "phase4_source_validation_checkpoint_curves_finalset_20260713.csv"
SAME_SCALE_SOURCE = TABLE_DIR / "phase3_same_scale_zero_shot_raw_rows.csv"
SCALE_TRANSFER_SOURCE = TABLE_DIR / "phase3_scale_transfer_zero_shot_raw_rows.csv"
LARGE_SCALE_SOURCE = TABLE_DIR / "phase3_6uav_load_transfer_new_multiseed_aggregate_rows.csv"

MAIN_METHODS = [
    "RelCritic",
    "GlobalCritic",
    "MAPPO-like",
    "cluster_center_queue_aware",
    "maxweight_lyapunov",
    "queue_aware_bw",
    "static_uniform",
]

PAPER_LABELS = {
    "RelCritic": "STARS",
    "GlobalCritic": "STARS-GC",
    "MAPPO-like": "HA-PPO",
    "cluster_center_queue_aware": "QCCS",
    "maxweight_lyapunov": "Lyapunov",
    "queue_aware_bw": "QBS",
    "static_uniform": "Uniform",
    "phase2_bootstrap_seed45211_u575": "Structured",
    "phase2_mc_seed45211_u400": "Structured-MC",
    "queue_aware": "Queue-aware",
    "bootstrap_seed45210_best_u0375": "Structured",
    "bootstrap_seed45211_best_u0375": "Structured",
    "bootstrap_seed61723_best_u0325": "Structured",
}

COLORS = {
    "STARS": "#2E6F8E",
    "STARS-GC": "#C76D43",
    "HA-PPO": "#6B7280",
    "QCCS": "#4E8B57",
    "Lyapunov": "#8A6F3D",
    "QBS": "#B95E5E",
    "Uniform": "#7A7A7A",
    "Structured": "#2E6F8E",
    "Structured-MC": "#6E5EA8",
    "Queue-aware": "#B95E5E",
}

SAME_SCALE_ORDER = [
    "load_low_3uav20gu_t250",
    "hotspot_sparse_3uav20gu_t250",
    "hotspot_wide_3uav20gu_t250",
    "load_high_3uav20gu_t250",
    "sat_visibility_harder_3uav20gu_t250",
    "k10_3uav20gu_t250",
]

SAME_SCALE_LABELS = {
    "load_low_3uav20gu_t250": "low load",
    "hotspot_sparse_3uav20gu_t250": "sparse hotspots",
    "hotspot_wide_3uav20gu_t250": "wide hotspots",
    "load_high_3uav20gu_t250": "high load",
    "sat_visibility_harder_3uav20gu_t250": "hard visibility",
    "k10_3uav20gu_t250": "K=10",
}

SCALE_ORDER = [
    "scale_num_gu10_3uav10gu_t250",
    "scale_num_gu30_3uav30gu_t250",
    "scale_num_uav2_2uav20gu_t250",
    "scale_num_uav4_4uav20gu_t250",
    "scale_visible_sats4_3uav20gu_t250",
    "scale_visible_sats8_3uav20gu_t250",
]

SCALE_LABELS = {
    "scale_num_gu10_3uav10gu_t250": "10 GUs",
    "scale_num_gu30_3uav30gu_t250": "30 GUs",
    "scale_num_uav2_2uav20gu_t250": "2 UAVs",
    "scale_num_uav4_4uav20gu_t250": "4 UAVs",
    "scale_visible_sats4_3uav20gu_t250": "4 visible sats",
    "scale_visible_sats8_3uav20gu_t250": "8 visible sats",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def label_method(method: str) -> str:
    return PAPER_LABELS.get(method, method)


def save_all(fig: plt.Figure, basename: str) -> None:
    FIG_SRC_DIR.mkdir(parents=True, exist_ok=True)
    FIG_MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG_SRC_DIR / f"{basename}.{ext}", bbox_inches="tight", dpi=300)
        shutil.copyfile(
            FIG_SRC_DIR / f"{basename}.{ext}",
            FIG_MANUSCRIPT_DIR / f"{basename}.{ext}",
        )


def load_main_table() -> pd.DataFrame:
    df = pd.read_csv(MAIN_SOURCE)
    df = df[df["method"].isin(MAIN_METHODS)].copy()
    df["method"] = pd.Categorical(df["method"], categories=MAIN_METHODS, ordered=True)
    df = df.sort_values("method")
    df["label"] = df["method"].astype(str).map(label_method)
    return df


def plot_training_dynamics() -> None:
    curve = pd.read_csv(CURVE_SOURCE)
    method_order = ["RelCritic", "GlobalCritic", "MAPPO-like"]
    color_map = {m: COLORS[label_method(m)] for m in method_order}

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), sharey=True)
    for ax, method in zip(axes, method_order):
        sub = curve[curve["method"] == method].copy()
        for seed, group in sub.groupby("training_seed"):
            group = group.sort_values("update")
            ax.plot(
                group["update"],
                group["reward_sum"],
                color=color_map[method],
                alpha=0.32,
                linewidth=0.9,
            )
            selected_row = group.loc[group["reward_sum"].idxmax()]
            final_row = group.sort_values("update").iloc[-1]
            ax.scatter(
                selected_row["update"],
                selected_row["reward_sum"],
                marker="*",
                s=55,
                color=color_map[method],
                edgecolor="black",
                linewidth=0.35,
                zorder=4,
            )
            ax.scatter(
                final_row["update"],
                final_row["reward_sum"],
                marker="s",
                s=26,
                facecolor="white",
                edgecolor=color_map[method],
                linewidth=1.0,
                zorder=4,
            )

        agg = (
            sub.groupby("update", as_index=False)["reward_sum"]
            .agg(["mean", "std"])
            .reset_index()
        )
        ax.plot(agg["update"], agg["mean"], color=color_map[method], linewidth=1.8)
        ax.fill_between(
            agg["update"].to_numpy(),
            (agg["mean"] - agg["std"]).to_numpy(),
            (agg["mean"] + agg["std"]).to_numpy(),
            color=color_map[method],
            alpha=0.16,
            linewidth=0,
        )
        ax.set_title(label_method(method))
        ax.set_xlabel("Training update")
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.55)

    axes[0].set_ylabel("Validation reward")
    handles = [
        plt.Line2D([0], [0], color="#333333", linewidth=1.8, label="mean"),
        plt.Line2D(
            [0],
            [0],
            marker="*",
            linestyle="",
            markerfacecolor="#333333",
            markeredgecolor="black",
            markersize=8,
            label="selected",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="#333333",
            markersize=5,
            label="final",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.035), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save_all(fig, "perf_eval_training_dynamics_20260714")
    plt.close(fig)


def plot_main_comparison() -> None:
    df = load_main_table()
    metrics = [
        ("Reward_mean", "Reward_std", "Reward", None),
        ("Processed_mean", "Processed_std", "Processed ratio", (0, 1.02)),
        ("Drop_mean", "Drop_std", "Drop ratio", (0, 0.52)),
        ("D_sys_mean", "D_sys_std", "System delay", None),
    ]
    y = np.arange(len(df))

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8))
    for ax, (mean_col, std_col, title, xlim) in zip(axes.ravel(), metrics):
        labels = df["label"].tolist()
        values = df[mean_col].astype(float).to_numpy()
        errors = df[std_col].astype(float).fillna(0.0).to_numpy()
        colors = [COLORS.get(label, "#7A7A7A") for label in labels]
        ax.barh(y, values, xerr=errors, color=colors, edgecolor="#303030", linewidth=0.35, capsize=2.0)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="x", color="#D8D8D8", linewidth=0.55)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if title in {"Drop ratio", "System delay"}:
            ax.set_xlabel("lower is better")
        else:
            ax.set_xlabel("higher is better")

    fig.tight_layout()
    save_all(fig, "perf_eval_main_comparison_20260714")
    plt.close(fig)


def plot_seed_spread() -> None:
    seed_df = pd.read_csv(SEED_SOURCE)
    seed_df = seed_df[
        (seed_df["checkpoint"] == "selected")
        & (seed_df["method"].isin(["RelCritic", "GlobalCritic"]))
    ].copy()
    methods = ["RelCritic", "GlobalCritic"]
    metrics = [
        ("Reward_mean", "Reward"),
        ("Processed_mean", "Processed ratio"),
        ("Drop_mean", "Drop ratio"),
        ("D_sys_mean", "System delay"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.35))
    rng_offsets = np.array([-0.045, 0.0, 0.045])
    for ax, (col, title) in zip(axes, metrics):
        for pos, method in enumerate(methods):
            values = seed_df[seed_df["method"] == method][col].astype(float).to_numpy()
            label = label_method(method)
            offsets = rng_offsets[: len(values)] if len(values) <= len(rng_offsets) else np.linspace(-0.06, 0.06, len(values))
            ax.scatter(
                np.full(len(values), pos) + offsets,
                values,
                s=34,
                color=COLORS[label],
                edgecolor="black",
                linewidth=0.35,
                zorder=3,
            )
            ax.errorbar(
                pos,
                values.mean(),
                yerr=values.std(ddof=1) if len(values) > 1 else 0.0,
                color="black",
                marker="_",
                markersize=13,
                capsize=3.5,
                linewidth=1.0,
                zorder=4,
            )
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([label_method(m) for m in methods], rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.55)

    fig.tight_layout()
    save_all(fig, "perf_eval_ablation_seed_spread_20260714")
    plt.close(fig)


def aggregate_raw_sweep(
    source: Path,
    scenario_order: list[str],
    selected_methods: list[str],
) -> pd.DataFrame:
    df = pd.read_csv(source)
    df = df[
        (df["status"] == "ok")
        & (df["scenario"].isin(scenario_order))
        & (df["method"].isin(selected_methods))
    ].copy()
    df["scenario"] = pd.Categorical(df["scenario"], categories=scenario_order, ordered=True)
    df["label"] = df["method"].map(label_method)
    rows = []
    for (scenario, method, label), group in df.groupby(["scenario", "method", "label"], observed=True):
        rows.append(
            {
                "scenario": str(scenario),
                "method": str(method),
                "label": str(label),
                "n": len(group),
                "Processed_mean": group["processed_ratio_eval"].astype(float).mean(),
                "Processed_std": group["processed_ratio_eval"].astype(float).std(ddof=1),
                "Drop_mean": group["drop_ratio_eval"].astype(float).mean(),
                "Drop_std": group["drop_ratio_eval"].astype(float).std(ddof=1),
                "D_sys_mean": group["D_sys_report"].astype(float).mean(),
                "D_sys_std": group["D_sys_report"].astype(float).std(ddof=1),
            }
        )
    out = pd.DataFrame(rows)
    out["scenario"] = pd.Categorical(out["scenario"], categories=scenario_order, ordered=True)
    out["method"] = pd.Categorical(out["method"], categories=selected_methods, ordered=True)
    return out.sort_values(["scenario", "method"]).reset_index(drop=True)


def plot_sweep_panels(
    agg: pd.DataFrame,
    basename: str,
    scenario_labels: dict[str, str],
    selected_methods: list[str],
) -> None:
    metrics = [
        ("Processed_mean", "Processed_std", "Processed ratio"),
        ("Drop_mean", "Drop_std", "Drop ratio"),
        ("D_sys_mean", "D_sys_std", "System delay"),
    ]
    labels = [scenario_labels[s] for s in agg["scenario"].cat.categories]
    x = np.arange(len(labels))
    width = 0.18 if len(selected_methods) >= 4 else 0.22

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.65))
    for ax, (col, std_col, title) in zip(axes, metrics):
        for idx, method in enumerate(selected_methods):
            method_rows = agg[agg["method"] == method].set_index("scenario").reindex(agg["scenario"].cat.categories)
            label = label_method(method)
            offset = (idx - (len(selected_methods) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                method_rows[col].astype(float).to_numpy(),
                yerr=method_rows[std_col].astype(float).fillna(0.0).to_numpy(),
                width=width,
                color=COLORS.get(label, "#7A7A7A"),
                edgecolor="#303030",
                linewidth=0.3,
                capsize=1.4,
                label=label,
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=32, ha="right")
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.55)
        if title in {"Processed ratio", "Drop ratio"}:
            ax.set_ylim(0, 1.03)
        if title == "System delay":
            ax.set_ylabel("slots")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=len(selected_methods), frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save_all(fig, basename)
    plt.close(fig)


def plot_same_scale_sensitivity() -> None:
    selected_methods = [
        "phase2_bootstrap_seed45211_u575",
        "cluster_center_queue_aware",
        "maxweight_lyapunov",
        "queue_aware",
    ]
    agg = aggregate_raw_sweep(SAME_SCALE_SOURCE, SAME_SCALE_ORDER, selected_methods)
    plot_sweep_panels(
        agg,
        "perf_eval_same_scale_sensitivity_draft_20260714",
        SAME_SCALE_LABELS,
        selected_methods,
    )


def plot_scale_transfer() -> None:
    selected_methods = [
        "phase2_bootstrap_seed45211_u575",
        "cluster_center_queue_aware",
        "maxweight_lyapunov",
        "queue_aware",
    ]
    agg = aggregate_raw_sweep(SCALE_TRANSFER_SOURCE, SCALE_ORDER, selected_methods)
    plot_sweep_panels(
        agg,
        "perf_eval_scale_transfer_draft_20260714",
        SCALE_LABELS,
        selected_methods,
    )


def plot_large_scale_load_case() -> None:
    df = pd.read_csv(LARGE_SCALE_SOURCE)
    scenario_order = ["6uav40gu", "6uav60gu", "6uav80gu"]
    baseline_methods = ["cluster_center_queue_aware", "maxweight_lyapunov", "static_uniform"]
    learned = df[df["kind"] == "learned"].copy()
    learned_rows = []
    for scenario, group in learned.groupby("scenario"):
        learned_rows.append(
            {
                "scenario": scenario,
                "label": "Structured",
                "Processed_mean": group["processed_ratio_eval_mean"].astype(float).mean(),
                "Processed_std": group["processed_ratio_eval_mean"].astype(float).std(ddof=1),
                "Drop_mean": group["drop_ratio_eval_mean"].astype(float).mean(),
                "Drop_std": group["drop_ratio_eval_mean"].astype(float).std(ddof=1),
                "D_sys_mean": group["D_sys_report_mean"].astype(float).mean(),
                "D_sys_std": group["D_sys_report_mean"].astype(float).std(ddof=1),
            }
        )
    baseline = df[df["method"].isin(baseline_methods)].copy()
    baseline_rows = []
    for _, row in baseline.iterrows():
        baseline_rows.append(
            {
                "scenario": row["scenario"],
                "label": label_method(row["method"]),
                "Processed_mean": row["processed_ratio_eval_mean"],
                "Processed_std": row["processed_ratio_eval_std"],
                "Drop_mean": row["drop_ratio_eval_mean"],
                "Drop_std": row["drop_ratio_eval_std"],
                "D_sys_mean": row["D_sys_report_mean"],
                "D_sys_std": row["D_sys_report_std"],
            }
        )
    agg = pd.DataFrame(learned_rows + baseline_rows)
    agg["scenario"] = pd.Categorical(agg["scenario"], categories=scenario_order, ordered=True)
    labels = ["40 GUs", "60 GUs", "80 GUs"]
    methods = ["Structured", "QCCS", "Lyapunov", "Uniform"]
    metrics = [
        ("Processed_mean", "Processed_std", "Processed ratio"),
        ("Drop_mean", "Drop_std", "Drop ratio"),
        ("D_sys_mean", "D_sys_std", "System delay"),
    ]
    x = np.arange(len(labels))
    width = 0.18

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.55))
    for ax, (col, std_col, title) in zip(axes, metrics):
        for idx, method in enumerate(methods):
            rows = agg[agg["label"] == method].set_index("scenario").reindex(scenario_order)
            offset = (idx - (len(methods) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                rows[col].astype(float).to_numpy(),
                yerr=rows[std_col].astype(float).fillna(0.0).to_numpy(),
                width=width,
                color=COLORS.get(method, "#7A7A7A"),
                edgecolor="#303030",
                linewidth=0.3,
                capsize=1.4,
                label=method,
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.55)
        if title in {"Processed ratio", "Drop ratio"}:
            ax.set_ylim(0, 1.03)
        if title == "System delay":
            ax.set_ylabel("slots")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    save_all(fig, "perf_eval_large_scale_load_case_draft_20260714")
    plt.close(fig)


def write_manifest() -> None:
    main = load_main_table()
    rel = main[main["method"].astype(str) == "RelCritic"].iloc[0]
    qccs = main[main["method"].astype(str) == "cluster_center_queue_aware"].iloc[0]
    mappo = main[main["method"].astype(str) == "MAPPO-like"].iloc[0]

    note = f"""# Performance Evaluation Figure Drafts (2026-07-14)

Generated by `docs/paper/reproduction/generate_performance_evaluation_figures_20260714.py`.

## Ready for Section 5

- `perf_eval_training_dynamics_20260714`: checkpoint validation dynamics. It marks selected best checkpoints with stars and final checkpoints with squares.
- `perf_eval_main_comparison_20260714`: main formal held-out comparison with seven schedulers: STARS, STARS-GC, HA-PPO, QCCS, Lyapunov, QBS, and Uniform.
- `perf_eval_ablation_seed_spread_20260714`: seed-level STARS versus STARS-GC spread, intended to make the GlobalCritic variance visible.

Main result signal from `phase4_formal_heldout_source_selected_main_20260713.csv`:

- STARS reward/process/drop/D_sys: {rel['Reward_mean']:.3f}, {rel['Processed_mean']:.3f}, {rel['Drop_mean']:.3f}, {rel['D_sys_mean']:.3f}.
- QCCS reward/process/drop/D_sys: {qccs['Reward_mean']:.3f}, {qccs['Processed_mean']:.3f}, {qccs['Drop_mean']:.3f}, {qccs['D_sys_mean']:.3f}.
- HA-PPO reward/process/drop/D_sys: {mappo['Reward_mean']:.3f}, {mappo['Processed_mean']:.3f}, {mappo['Drop_mean']:.3f}, {mappo['D_sys_mean']:.3f}.

## Draft Sensitivity Figures

- `perf_eval_same_scale_sensitivity_draft_20260714`: uses phase3 same-scale zero-shot rows. It is useful for deciding the robustness story, but it is not yet a unified phase4 formal STARS evaluation.
- `perf_eval_scale_transfer_draft_20260714`: uses phase3 nearby scale-transfer zero-shot rows. Same caveat as above.
- `perf_eval_large_scale_load_case_draft_20260714`: uses phase3 6-UAV load-transfer aggregate rows. Treat as a case-study/generalization figure unless the current selected phase4 checkpoints are re-evaluated in this protocol.

## Figure Copies

Source figures are under:

- `docs/paper/reproduction/generated_figures/performance_evaluation_20260714/`

LaTeX-ready copies are under:

- `docs/paper/manuscript_overleaf/figures/`

## Writing Boundary

For the paper draft, the first three figures can support the controlled source-scenario Section 5 claims. The draft sensitivity figures should either be placed in the merged later part of Performance Evaluation with explicit protocol wording, or regenerated after a unified formal sweep using the current selected checkpoints.
"""
    # Narrative figure notes are maintained in phase4_learning_ablation_runbook.md.
    _ = note


def main() -> None:
    configure_style()
    plot_training_dynamics()
    plot_main_comparison()
    plot_seed_spread()
    plot_same_scale_sensitivity()
    plot_scale_transfer()
    plot_large_scale_load_case()
    write_manifest()


if __name__ == "__main__":
    main()
