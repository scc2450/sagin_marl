"""Plot the completed primary 100-GU scan matrix without GC or historical QCCS."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sagin_marl_scan_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2] / "paper/reproduction"))
from generate_section5_single_panel_figures_20260714 import configure_style

ORDER = ["stars", "distributed_queue_c", "maxweight_lyapunov", "queue_aware_bw", "static_uniform"]
LABELS = dict(zip(ORDER, ["STARS", "DQS", "Lyapunov", "QBS", "Uniform"]))
COLORS = dict(zip(ORDER, ["#0072B2", "#009E73", "#E69F00", "#56B4E9", "#666666"]))
MARKERS = dict(zip(ORDER, ["o", "D", "P", "v", "x"]))
PANELS = [
    ("reward_sum", "Episode reward", 1.0),
    ("processed_ratio_eval", "Processed (%)", 100.0),
    ("drop_ratio_eval", "Dropped (%)", 100.0),
    ("D_sys_report", r"System delay $D_{\mathrm{sys}}$", 1.0),
    ("pre_backlog_steps_eval", "Pre-backlog (slots)", 1.0),
    ("queue_total_mean", "Queue workload (Mbit)", 1e-6),
]


def validate_primary(data, manifest):
    if set(data.method) - set(ORDER) or set(data.paper_label) - set(LABELS.values()):
        raise ValueError("Unexpected method, GC or historical QCCS in the scan")
    primary = data[data.checkpoint_kind.isin(["fixed", "selected"])].copy()
    actual = set(primary.point)
    expected = {point["point"] for point in manifest["points"]}
    if actual != expected:
        raise ValueError("Incomplete primary scan points")
    for point in expected:
        block = primary[primary.point == point]
        if set(block.method) != set(ORDER):
            raise ValueError("Incomplete primary method coverage")
        for method in ORDER:
            rows = block[block.method == method]
            seeds = manifest["training_seeds"] if method == "stars" else [-1]
            for seed in seeds:
                subset = rows[rows.training_seed.fillna(-1) == seed]
                if len(subset) != 64:
                    raise ValueError("Expected64episodes per policy per point")
                if set(subset.seed_base) != {1980000, 1981000}:
                    raise ValueError("Wrong evaluation seed bases")
                if subset[["seed_base", "episode"]].duplicated().any():
                    raise ValueError("Duplicate episode rows")
                if not np.isfinite(subset[[metric for metric, _, _ in PANELS]].to_numpy()).all():
                    raise ValueError("Non-finite plotting metric")
    return primary


def summaries(data):
    frame = data.copy()
    frame["training_seed"] = frame.training_seed.fillna(-1).astype(int)
    metrics = [metric for metric, _, _ in PANELS] + ["collision_episode_fraction"]
    per_seed = frame.groupby(["axis", "point", "multiplier", "total_arrival_mbps",
        "method", "checkpoint_kind", "training_seed"], as_index=False)[metrics].mean()
    mean = per_seed.groupby(["axis", "point", "multiplier", "total_arrival_mbps",
        "method", "checkpoint_kind"], as_index=False)[metrics].mean()
    spread = per_seed.groupby(["axis", "point", "method", "checkpoint_kind"], as_index=False)[metrics].std()
    return per_seed, mean, spread


def draw(ax, axis, metric, label, scale, mean, spread):
    x_col = "total_arrival_mbps" if axis == "load" else "multiplier"
    for method in ORDER:
        rows = mean[(mean.axis == axis) & (mean.method == method)
                    & mean.checkpoint_kind.isin(["selected", "fixed"])].sort_values(x_col)
        ax.plot(rows[x_col], rows[metric] * scale, label=LABELS[method],
                color=COLORS[method], marker=MARKERS[method], ms=3, lw=1.3)
        if method == "stars":
            deviations = rows[["point"]].merge(spread[(spread.axis == axis)
                & (spread.method == method) & (spread.checkpoint_kind == "selected")],
                on="point", validate="one_to_one")[metric].fillna(0).to_numpy() * scale
            ax.fill_between(rows[x_col].to_numpy(), rows[metric].to_numpy() * scale - deviations,
                rows[metric].to_numpy() * scale + deviations, color=COLORS[method], alpha=0.14, linewidth=0)
    ax.set_xlabel("Total task arrival rate (Mbit/s)" if axis == "load" else "Joint resource multiplier")
    ax.set_ylabel(label)
    ax.set_ylim(bottom=0)
    if metric in {"processed_ratio_eval", "drop_ratio_eval"}:
        ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#D9D9D9", alpha=0.72)
    ax.set_axisbelow(True)


def save(fig, out, name):
    for ext in ("png", "pdf"):
        folder = out / ext
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"{name}.{ext}", dpi=240, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    data = pd.read_csv(root / "episodes.csv")
    legacy_label = (data.method == "maxweight_lyapunov") & (data.paper_label == "MaxWeight/Lyapunov")
    data.loc[legacy_label, "paper_label"] = "Lyapunov"
    primary = validate_primary(data, manifest)
    per_seed, mean, spread = summaries(data)
    out = root / "figures"
    out.mkdir(exist_ok=True)
    per_seed.to_csv(out / "per_training_seed.csv", index=False)
    mean.to_csv(out / "means.csv", index=False)
    spread.to_csv(out / "training_seed_sd.csv", index=False)
    configure_style()
    for axis in ("load", "resource"):
        for metric, label, scale in PANELS:
            fig, ax = plt.subplots(figsize=(3.45, 2.95))
            fig.subplots_adjust(left=0.20, right=0.97, bottom=0.18, top=0.78)
            draw(ax, axis, metric, label, scale, mean, spread)
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
            save(fig, out, f"{axis}_{metric}")
        fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.4))
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.10, top=0.88, hspace=0.36, wspace=0.30)
        for ax, (metric, label, scale) in zip(axes.flat, PANELS):
            draw(ax, axis, metric, label, scale, mean, spread)
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper center", ncol=5, frameon=False)
        save(fig, out, f"{axis}_overview")
    final_rows = data[data.checkpoint_kind == "final"]
    final_complete = len(final_rows) == len(manifest["points"]) * len(manifest["training_seeds"]) * 64
    if final_complete:
        for axis in ("load", "resource"):
            fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.25))
            fig.subplots_adjust(left=0.06, right=0.98, bottom=0.20, top=0.80, wspace=0.35)
            x_col = "total_arrival_mbps" if axis == "load" else "multiplier"
            for ax, (metric, label, scale) in zip(axes, PANELS[:2] + [PANELS[3]]):
                for kind, style in (("selected", "-"), ("final", "--")):
                    rows = mean[(mean.axis == axis) & (mean.method == "stars")
                        & (mean.checkpoint_kind == kind)].sort_values(x_col)
                    ax.plot(rows[x_col], rows[metric] * scale, style, marker="o", ms=3, label=kind)
                ax.set_xlabel("Total arrival (Mbit/s)" if axis == "load" else "Joint resource multiplier")
                ax.set_ylabel(label)
                ax.set_ylim(bottom=0)
                ax.grid(axis="y", alpha=0.3)
            fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
            save(fig, out, f"{axis}_selected_vs_final")
    provenance = dict(source_commit=manifest["source_commit"],
        input_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                      for name in ("manifest.json", "episodes.csv")},
        checkpoint_identities=manifest["checkpoints"],
        primary_rows=len(primary), final_rows=len(final_rows), final_complete=final_complete,
        methods=[LABELS[method] for method in ORDER],
        uncertainty="STARS mean and descriptive +/-1 sample SD of3independent training-seed means; fixed curves are64episode means without training-seed bands",
        resources=manifest["resource_axis"], evidence_role=manifest["evidence_role"])
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    report = [
        "# 100-GU Parameter Scans", "",
        "Primary methods: STARS, DQS, Lyapunov, QBS, Uniform. GC and QCCS are excluded.",
        "STARS checkpoints were selected before scanning; no per-point model selection or retraining.",
        "STARS bands show descriptive training-seed SD (n=3), not an episode-level confidence interval.",
        "Fixed baselines use64paired episodes each. Seeds1980000/1981000 are reused screening seeds.",
        "Resource scaling changes access bandwidth, backhaul bandwidth and satellite CPU jointly.",
        "The45211original selected/final are used; the separate u500-to-u700 continuation is excluded.", "",
        "![Load scan](png/load_overview.png)", "", "![Resource scan](png/resource_overview.png)", "",
        "CSV files retain per-training-seed means, queue metrics and collision fractions.",
        f"Final-checkpoint diagnostics complete: {final_complete}.",
    ]
    if final_complete:
        report += ["", "![Load selected/final](png/load_selected_vs_final.png)",
                   "", "![Resource selected/final](png/resource_selected_vs_final.png)"]
    (out / "README.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
