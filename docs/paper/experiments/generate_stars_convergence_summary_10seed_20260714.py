#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json


REPO_ROOT = Path(__file__).resolve().parents[3]
TABLE_DIR = REPO_ROOT / "docs" / "paper" / "table_sources"
FIGURE_SOURCE_DIR = (
    REPO_ROOT / "docs" / "paper" / "figure_sources" / "performance_evaluation_20260714"
)
MANUSCRIPT_FIGURE_DIR = REPO_ROOT / "docs" / "paper" / "manuscript" / "figures"

CURVE_CSV = TABLE_DIR / "phase4_stars_training_checkpoint_curves_10seed_20260714.csv"
SUMMARY_CSV = TABLE_DIR / "phase4_stars_training_checkpoint_summary_10seed_20260714.csv"
QCCS_REF_JSON = TABLE_DIR / "phase4_qccs_checkpoint_validation_ref_20260714_summary.json"
OUT_STEM = "phase4_stars_convergence_summary_10seed_20260714"

STARS_BLUE = "#0072B2"
QCCS_GREEN = "#009E73"
SEED_TRACE = "#7A9BB7"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def main() -> None:
    curve_df = pd.read_csv(CURVE_CSV)
    summary_df = pd.read_csv(SUMMARY_CSV)
    seeds = sorted(curve_df["training_seed"].unique())
    updates = np.arange(25, 526, 25)

    rows = []
    for seed in seeds:
        group = (
            curve_df[curve_df["training_seed"] == seed]
            .sort_values("update")[["update", "reward_sum"]]
            .drop_duplicates("update", keep="last")
        )
        series = group.set_index("update")["reward_sum"].reindex(updates)
        # Carry the last available best value forward after early stopping. This
        # represents the selected-checkpoint envelope, not the raw final policy.
        best_so_far = series.ffill().cummax().ffill()
        for update, reward in best_so_far.items():
            rows.append({"training_seed": seed, "update": int(update), "best_reward": float(reward)})

    best_df = pd.DataFrame(rows)
    agg = (
        best_df.groupby("update", as_index=False)["best_reward"]
        .agg(median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))
        .sort_values("update")
    )

    qccs_reward = None
    if QCCS_REF_JSON.exists():
        payload = json.loads(QCCS_REF_JSON.read_text(encoding="utf-8"))
        qccs_reward = float(payload["summary"]["reward_sum"])

    fig, ax = plt.subplots(figsize=(4.65, 2.55))
    for seed_index, (_seed, group) in enumerate(best_df.groupby("training_seed", sort=True)):
        ax.plot(
            group["update"],
            group["best_reward"],
            color=SEED_TRACE,
            linewidth=0.65,
            alpha=0.22,
            label="Individual seeds" if seed_index == 0 else "_nolegend_",
            zorder=1,
        )

    ax.plot(
        agg["update"],
        agg["median"],
        color=STARS_BLUE,
        linewidth=2.0,
        label="STARS median",
        zorder=3,
    )
    err_df = agg[agg["update"].isin([125, 225, 325, 425, 525])].copy()
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
        elinewidth=0.9,
        capsize=2.2,
        capthick=0.9,
        markersize=3.2,
        markerfacecolor="white",
        markeredgewidth=0.9,
        label="IQR across seeds",
        zorder=4,
    )
    if qccs_reward is not None:
        ax.axhline(
            qccs_reward,
            color=QCCS_GREEN,
            linestyle=(0, (4, 2)),
            linewidth=1.2,
            label="QCCS reference",
        )

    ax.set_xlabel("Training update")
    ax.set_ylabel("Best-so-far validation reward")
    ax.set_xlim(20, 550)
    ax.set_xticks([25, 125, 225, 325, 425, 525])
    ax.set_ylim(28, 74)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False, handlelength=2.4)
    fig.tight_layout()

    FIGURE_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for out_dir in [FIGURE_SOURCE_DIR, MANUSCRIPT_FIGURE_DIR]:
        fig.savefig(out_dir / f"{OUT_STEM}.pdf")
        fig.savefig(out_dir / f"{OUT_STEM}.png", dpi=300)
        fig.savefig(out_dir / f"{OUT_STEM}.svg")
    plt.close(fig)

    print(f"wrote {MANUSCRIPT_FIGURE_DIR / (OUT_STEM + '.pdf')}")
    print(f"seeds={len(seeds)} max_update=525")


if __name__ == "__main__":
    main()
