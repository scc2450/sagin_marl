"""Generate Section 5 formal held-out tables and figures.

Inputs are the small formal-evaluation summary JSON files and training assets
pulled from friday into /tmp during the 2026-07-13 paper-prep session.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("SAGIN_MARL_ROOT", Path(__file__).resolve().parents[3])).resolve()
SUMMARY_DIR = Path(os.environ.get("PHASE4_SUMMARY_DIR", "/tmp/phase4_formal_summaries_json"))
STATUS_DIR = Path(os.environ.get("PHASE4_STATUS_DIR", "/tmp/phase4_formal_summaries"))
TRAIN_ROOT = Path(os.environ.get("PHASE4_TRAIN_ROOT", "/tmp/phase4_train_assets_extract"))

TABLE_DIR = ROOT / "docs/paper/table_sources"
FIG_SRC_DIR = ROOT / "docs/paper/figure_sources/phase4_section5_performance_20260713"
FIG_MANUSCRIPT_DIR = ROOT / "docs/paper/manuscript/figures"
EXP_DIR = ROOT / "docs/paper/experiments"

METHOD_LABEL = {
    "relcritic": "RelCritic",
    "globalcritic": "GlobalCritic",
    "mappo_like": "MAPPO-like",
    "cluster_center_queue_aware": "cluster_center_queue_aware",
    "maxweight_lyapunov": "maxweight_lyapunov",
    "observable_cluster_queue_aware": "observable_cluster_queue_aware",
    "queue_aware_bw": "queue_aware_bw",
    "static_uniform": "static_uniform",
}
LEARNED_ORDER = ["RelCritic", "GlobalCritic", "MAPPO-like"]
FIXED_ORDER = [
    "cluster_center_queue_aware",
    "maxweight_lyapunov",
    "observable_cluster_queue_aware",
    "queue_aware_bw",
    "static_uniform",
]
MAIN_ORDER = LEARNED_ORDER + FIXED_ORDER
SUMMARY_METRICS = {
    "reward_sum": "Reward",
    "processed_ratio_eval": "Processed",
    "drop_ratio_eval": "Drop",
    "pre_backlog_steps_eval": "Backlog",
    "D_sys_report": "D_sys",
    "collision_episode_fraction": "Collision",
}


def parse_summary_name(name: str) -> dict[str, object]:
    stem = name.removesuffix("_summary.json")
    if "_fixed_seedbase" in stem:
        method, seedbase = stem.split("_fixed_seedbase")
        return {
            "row_type": "fixed",
            "method_key": method,
            "method": METHOD_LABEL.get(method, method),
            "training_seed": np.nan,
            "checkpoint": "fixed",
            "episode_seed_base": int(seedbase),
        }

    match = re.match(
        r"(?P<method>.+)_seed(?P<seed>\d+)_(?P<checkpoint>selected|final)_seedbase(?P<seedbase>\d+)$",
        stem,
    )
    if not match:
        raise ValueError(f"Cannot parse summary filename: {name}")
    method_key = match.group("method")
    return {
        "row_type": "learned",
        "method_key": method_key,
        "method": METHOD_LABEL.get(method_key, method_key),
        "training_seed": int(match.group("seed")),
        "checkpoint": match.group("checkpoint"),
        "episode_seed_base": int(match.group("seedbase")),
    }


def format_pm(row: pd.Series, metric: str) -> str:
    mean = row[f"{metric}_mean"]
    std = row.get(f"{metric}_std", np.nan)
    if pd.isna(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} +/- {std:.3f}"


def markdown_table(df: pd.DataFrame, metrics: list[str]) -> str:
    header = ["Method", "Checkpoint", "n train seeds"] + metrics
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for _, row in df.iterrows():
        n_seeds = "" if pd.isna(row.get("n_train_seeds", np.nan)) else str(int(row["n_train_seeds"]))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["checkpoint"]),
                    n_seeds,
                    *[format_pm(row, metric) for metric in metrics],
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def save_figure_all_formats(fig: plt.Figure, basename: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_SRC_DIR / f"{basename}.{ext}", bbox_inches="tight", dpi=240)
    for ext in ("png", "pdf", "svg"):
        shutil.copyfile(
            FIG_SRC_DIR / f"{basename}.{ext}",
            FIG_MANUSCRIPT_DIR / f"{basename}.{ext}",
        )


def load_raw_formal_rows() -> pd.DataFrame:
    rows = []
    for path in sorted(SUMMARY_DIR.glob("*_summary.json")):
        data = json.loads(path.read_text())
        summary = data["summary"]
        row = {
            **parse_summary_name(path.name),
            "label": data.get("label", ""),
            "config": data.get("config", ""),
            "checkpoint_path": data.get("base_checkpoint", ""),
            "episodes": int(data.get("episodes", summary.get("episodes", 0))),
            "num_envs": int(data.get("num_envs", 0)),
            "policy_mode": data.get("policy_mode", ""),
            "exec_sources": json.dumps(data.get("exec_sources", {}), sort_keys=True),
            "local_summary_json": str(path),
        }
        for source_key, output_key in SUMMARY_METRICS.items():
            row[output_key] = summary.get(source_key, np.nan)
        rows.append(row)

    raw_df = pd.DataFrame(rows)
    if len(raw_df) != 69:
        raise RuntimeError(f"Expected 69 formal summary rows, got {len(raw_df)}")
    return raw_df.sort_values(
        ["row_type", "method", "training_seed", "checkpoint", "episode_seed_base"],
        na_position="last",
    ).reset_index(drop=True)


def aggregate_tables(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    learned_raw = raw_df[raw_df["row_type"] == "learned"].copy()
    seed_rows = []
    for (method, training_seed, checkpoint), group in learned_raw.groupby(
        ["method", "training_seed", "checkpoint"],
        dropna=False,
    ):
        row = {
            "method": method,
            "row_type": "learned",
            "training_seed": int(training_seed),
            "checkpoint": checkpoint,
            "n_eval_seed_bases": group["episode_seed_base"].nunique(),
            "episodes_per_eval_seed_base": int(group["episodes"].iloc[0]),
            "total_episodes": int(group["episodes"].sum()),
            "checkpoint_path": group["checkpoint_path"].dropna().iloc[0],
        }
        for metric in SUMMARY_METRICS.values():
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std_eval_seed_base"] = group[metric].std(ddof=1)
        seed_rows.append(row)

    seed_df = pd.DataFrame(seed_rows).sort_values(["method", "training_seed", "checkpoint"])

    companion_rows = []
    for (method, checkpoint), group in seed_df.groupby(["method", "checkpoint"], dropna=False):
        row = {
            "method": method,
            "row_type": "learned",
            "checkpoint": checkpoint,
            "n_train_seeds": group["training_seed"].nunique(),
            "n_eval_seed_bases_per_seed": int(group["n_eval_seed_bases"].iloc[0]),
            "episodes_per_eval_seed_base": int(group["episodes_per_eval_seed_base"].iloc[0]),
        }
        for metric in SUMMARY_METRICS.values():
            values = group[f"{metric}_mean"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1)
        companion_rows.append(row)
    learned_companion = pd.DataFrame(companion_rows)
    learned_companion["method"] = pd.Categorical(
        learned_companion["method"],
        categories=LEARNED_ORDER,
        ordered=True,
    )
    learned_companion["checkpoint"] = pd.Categorical(
        learned_companion["checkpoint"],
        categories=["selected", "final"],
        ordered=True,
    )
    learned_companion = learned_companion.sort_values(["method", "checkpoint"]).reset_index(drop=True)

    fixed_rows = []
    for method, group in raw_df[raw_df["row_type"] == "fixed"].groupby("method"):
        row = {
            "method": method,
            "row_type": "fixed",
            "checkpoint": "fixed",
            "n_train_seeds": np.nan,
            "n_eval_seed_bases_per_seed": group["episode_seed_base"].nunique(),
            "episodes_per_eval_seed_base": int(group["episodes"].iloc[0]),
        }
        for metric in SUMMARY_METRICS.values():
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        fixed_rows.append(row)
    fixed_companion = pd.DataFrame(fixed_rows)

    selected_learned = learned_companion[learned_companion["checkpoint"].astype(str) == "selected"]
    main_df = pd.concat([selected_learned, fixed_companion], ignore_index=True)
    main_df["method"] = pd.Categorical(main_df["method"].astype(str), categories=MAIN_ORDER, ordered=True)
    main_df = main_df.sort_values("method").reset_index(drop=True)
    return seed_df, learned_companion, main_df


def load_training_assets(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    learned_raw = raw_df[raw_df["row_type"] == "learned"].copy()
    records = []
    for method, seed, checkpoint_path in learned_raw[
        ["method", "training_seed", "checkpoint_path"]
    ].drop_duplicates().itertuples(index=False):
        run_dir = TRAIN_ROOT / Path(checkpoint_path).parent
        if not run_dir.exists():
            raise RuntimeError(f"Missing run dir for {method} seed {seed}: {run_dir}")
        records.append(
            {
                "method": method,
                "training_seed": int(seed),
                "run_dir": run_dir,
                "remote_run_dir": str(Path(checkpoint_path).parent),
            }
        )
    run_df = pd.DataFrame(records).drop_duplicates().sort_values(["method", "training_seed"])

    curve_rows = []
    runtime_rows = []
    selected_updates = {}
    final_updates = {}

    for rec in run_df.itertuples(index=False):
        checkpoint_eval = pd.read_csv(rec.run_dir / "checkpoint_eval.csv")
        checkpoint_eval["method"] = rec.method
        checkpoint_eval["training_seed"] = rec.training_seed
        checkpoint_eval["remote_run_dir"] = rec.remote_run_dir
        curve_rows.append(checkpoint_eval)

        metrics = pd.read_csv(rec.run_dir / "metrics.csv")
        stop = json.loads((rec.run_dir / "training_stop.json").read_text())
        best = checkpoint_eval.loc[checkpoint_eval["reward_sum"].idxmax()]
        selected_update = int(best["update"])
        final_update = int(stop.get("completed_updates", int(metrics["update"].max())))
        wall_sec = float(metrics["iteration_sec"].sum()) if "iteration_sec" in metrics else np.nan

        selected_updates[(rec.method, rec.training_seed)] = selected_update
        final_updates[(rec.method, rec.training_seed)] = final_update
        runtime_rows.append(
            {
                "method": rec.method,
                "training_seed": rec.training_seed,
                "remote_run_dir": rec.remote_run_dir,
                "training_wall_clock_sec_from_iteration_sum": wall_sec,
                "training_wall_clock_hours_from_iteration_sum": wall_sec / 3600
                if not math.isnan(wall_sec)
                else np.nan,
                "gpu": "friday GPU0/GPU1 (formal queue placement retained in status logs)",
                "num_envs": 64,
                "rollout_env_steps": 250,
                "rollout_transitions_per_update": 64 * 250,
                "selected_update": selected_update,
                "final_update": final_update,
                "completed_updates": int(stop.get("completed_updates", np.nan)),
                "max_updates": int(stop.get("max_updates", np.nan)),
                "stop_reason": stop.get("stop_reason", ""),
                "resume_run": "resume_" in str(rec.remote_run_dir),
                "selected_reward_sum_checkpoint_eval": float(best["reward_sum"]),
                "final_reward_sum_checkpoint_eval": float(
                    checkpoint_eval.sort_values("update").iloc[-1]["reward_sum"]
                ),
            }
        )

    curve_df = pd.concat(curve_rows, ignore_index=True)
    curve_df["method"] = pd.Categorical(curve_df["method"], categories=LEARNED_ORDER, ordered=True)
    curve_df = curve_df.sort_values(["method", "training_seed", "update"]).reset_index(drop=True)
    runtime_df = pd.DataFrame(runtime_rows).sort_values(["method", "training_seed"]).reset_index(drop=True)
    return curve_df, runtime_df, selected_updates, final_updates


def formal_eval_wallclock_summary() -> pd.DataFrame:
    frames = []
    for path in sorted(STATUS_DIR.glob("status_gpu*.csv")):
        frame = pd.read_csv(path)
        frame["status_file"] = path.name
        frames.append(frame)
    status_df = pd.concat(frames, ignore_index=True)
    status_df["started_at_dt"] = pd.to_datetime(status_df["started_at"], errors="coerce")
    status_df["finished_at_dt"] = pd.to_datetime(status_df["finished_at"], errors="coerce")
    status_df["wall_clock_sec"] = (
        status_df["finished_at_dt"] - status_df["started_at_dt"]
    ).dt.total_seconds()
    ok = status_df[(status_df["kind"].isin(["learned", "fixed"])) & (status_df["rc"] == 0)].copy()

    rows = []
    for (kind, method, checkpoint), group in ok.groupby(["kind", "method", "checkpoint"], dropna=False):
        rows.append(
            {
                "kind": kind,
                "method": METHOD_LABEL.get(method, method),
                "checkpoint": checkpoint,
                "rows": len(group),
                "episodes_total": int(pd.to_numeric(group["episodes"], errors="coerce").sum()),
                "wall_clock_sec_sum": float(group["wall_clock_sec"].sum()),
                "wall_clock_min_sum": float(group["wall_clock_sec"].sum() / 60),
                "wall_clock_sec_mean_per_row": float(group["wall_clock_sec"].mean()),
                "status_files": ";".join(sorted(group["status_file"].unique())),
            }
        )
    return pd.DataFrame(rows).sort_values(["kind", "method", "checkpoint"]).reset_index(drop=True)


def plot_training_curves(curve_df: pd.DataFrame, selected_updates: dict, final_updates: dict) -> None:
    colors = {
        "RelCritic": "#2f6f8f",
        "GlobalCritic": "#c06c45",
        "MAPPO-like": "#6a6f7a",
    }
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2), sharey=True)
    for ax, method in zip(axes, LEARNED_ORDER):
        sub = curve_df[curve_df["method"].astype(str) == method]
        for seed, group in sub.groupby("training_seed"):
            ax.plot(group["update"], group["reward_sum"], color=colors[method], alpha=0.35, linewidth=1.1)
            selected_update = selected_updates[(method, int(seed))]
            final_update = final_updates[(method, int(seed))]
            selected_row = group.iloc[(group["update"] - selected_update).abs().argmin()]
            final_row = group.iloc[(group["update"] - final_update).abs().argmin()]
            ax.scatter(
                [selected_row["update"]],
                [selected_row["reward_sum"]],
                color=colors[method],
                marker="*",
                s=70,
                edgecolor="black",
                linewidth=0.35,
                zorder=4,
            )
            ax.scatter(
                [final_row["update"]],
                [final_row["reward_sum"]],
                color="white",
                marker="s",
                s=36,
                edgecolor=colors[method],
                linewidth=1.1,
                zorder=4,
            )
        aggregate = sub.groupby("update", as_index=False)["reward_sum"].agg(["mean", "std"]).reset_index()
        ax.plot(aggregate["update"], aggregate["mean"], color=colors[method], linewidth=2.0)
        ax.fill_between(
            aggregate["update"].to_numpy(),
            (aggregate["mean"] - aggregate["std"]).to_numpy(),
            (aggregate["mean"] + aggregate["std"]).to_numpy(),
            color=colors[method],
            alpha=0.16,
            linewidth=0,
        )
        ax.set_title(method)
        ax.set_xlabel("Training update")
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    axes[0].set_ylabel("Checkpoint validation reward")
    legend_handles = [
        plt.Line2D([0], [0], color="#333333", linewidth=2, label="mean over seeds"),
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="#333333",
            markeredgecolor="black",
            markersize=9,
            label="selected best",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="white",
            markeredgecolor="#333333",
            markersize=6,
            label="final",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Checkpoint validation curves with selected and final checkpoints", y=1.02)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    save_figure_all_formats(fig, "phase4_training_validation_curves_20260713")
    plt.close(fig)


def plot_main_bars(main_df: pd.DataFrame) -> None:
    colors = {
        "RelCritic": "#2f6f8f",
        "GlobalCritic": "#c06c45",
        "MAPPO-like": "#6a6f7a",
        "cluster_center_queue_aware": "#4b8f58",
        "maxweight_lyapunov": "#8a6f3d",
        "observable_cluster_queue_aware": "#8b5f8d",
        "queue_aware_bw": "#b35b5b",
        "static_uniform": "#777777",
    }
    metrics = [
        ("Reward_mean", "Reward"),
        ("Processed_mean", "Processed"),
        ("Drop_mean", "Drop"),
        ("D_sys_mean", "D_sys"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.4))
    values = main_df.copy()
    values["method_str"] = values["method"].astype(str)
    x = np.arange(len(values))
    for ax, (column, label) in zip(axes.ravel(), metrics):
        error_column = column.replace("_mean", "_std")
        ax.bar(
            x,
            values[column].astype(float),
            yerr=values[error_column].astype(float),
            capsize=2.5,
            color=[colors[m] for m in values["method_str"]],
            edgecolor="#333333",
            linewidth=0.4,
        )
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(values["method_str"], rotation=35, ha="right")
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        if label in {"Processed", "Drop"}:
            ax.set_ylim(0, 1.02)
    fig.suptitle("Formal held-out source-scenario performance, selected learned checkpoints", y=1.02)
    fig.tight_layout()
    save_figure_all_formats(fig, "phase4_main_performance_grouped_bars_20260713")
    plt.close(fig)


def plot_globalcritic_seed_level(seed_df: pd.DataFrame) -> None:
    colors = {"RelCritic": "#2f6f8f", "GlobalCritic": "#c06c45"}
    seed_plot = seed_df[
        (seed_df["checkpoint"] == "selected")
        & (seed_df["method"].isin(["RelCritic", "GlobalCritic"]))
    ].copy()
    metrics = [
        ("Reward_mean", "Reward"),
        ("Processed_mean", "Processed"),
        ("Drop_mean", "Drop"),
        ("D_sys_mean", "D_sys"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10.8, 3.1))
    for ax, (column, label) in zip(axes, metrics):
        for i, method in enumerate(["RelCritic", "GlobalCritic"]):
            values = seed_plot[seed_plot["method"] == method][column].astype(float).to_numpy()
            jitter = np.linspace(-0.07, 0.07, len(values)) if len(values) else []
            ax.scatter(
                np.full(len(values), i) + jitter,
                values,
                color=colors[method],
                s=42,
                edgecolor="black",
                linewidth=0.35,
                zorder=3,
            )
            mean = values.mean()
            std = values.std(ddof=1) if len(values) > 1 else 0.0
            ax.errorbar(
                [i],
                [mean],
                yerr=[std],
                color="black",
                marker="_",
                markersize=14,
                capsize=4,
                linewidth=1.1,
                zorder=4,
            )
        ax.set_title(label)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["RelCritic", "GlobalCritic"], rotation=20, ha="right")
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.suptitle("Seed-level spread for GlobalCritic versus RelCritic, selected checkpoints", y=1.02)
    fig.tight_layout()
    save_figure_all_formats(fig, "phase4_globalcritic_seed_level_20260713")
    plt.close(fig)


def write_freeze_note(
    main_df: pd.DataFrame,
    learned_companion: pd.DataFrame,
    runtime_df: pd.DataFrame,
) -> None:
    runtime_agg = (
        runtime_df.groupby("method")
        .agg(
            training_wall_clock_hours_mean=("training_wall_clock_hours_from_iteration_sum", "mean"),
            training_wall_clock_hours_std=("training_wall_clock_hours_from_iteration_sum", "std"),
            selected_update_mean=("selected_update", "mean"),
            final_update_mean=("final_update", "mean"),
            completed_updates_mean=("completed_updates", "mean"),
        )
        .reindex(LEARNED_ORDER)
        .reset_index()
    )

    runtime_lines = [
        "| Method | train wall-clock h | selected update | final update | completed updates |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in runtime_agg.iterrows():
        runtime_lines.append(
            f"| {row['method']} | "
            f"{row['training_wall_clock_hours_mean']:.2f} +/- {row['training_wall_clock_hours_std']:.2f} | "
            f"{row['selected_update_mean']:.0f} | "
            f"{row['final_update_mean']:.0f} | "
            f"{row['completed_updates_mean']:.0f} |"
        )

    note = f"""# Section 5 Performance Assets Freeze (2026-07-13)

Purpose: freeze the Section 5 main performance evidence after the formal held-out source-scenario evaluation completed on 2026-07-12/13.

## Protocol

- Scenario: `3uav20gu_t250`.
- Held-out policy mode: deterministic.
- Evaluation seed bases: `980000`, `981000`, `982000`.
- Episodes per seed base: 64, i.e. 192 held-out episodes per learned training seed or fixed baseline.
- Learned methods: RelCritic, GlobalCritic, and MAPPO-like, each with 3 training seeds.
- Fixed baselines in the main table: `cluster_center_queue_aware`, `maxweight_lyapunov`, `observable_cluster_queue_aware`, `queue_aware_bw`, and `static_uniform`.
- Excluded row: one `rc=143` MAPPO-like slow-path/manual probe in `status_gpu1.csv`; it is not counted in any table.

## Main Table Decision

Use selected checkpoints for the Section 5 main table. Put fixed baselines in the same table. Do not center the final checkpoint rows in the main table; use them as stability/selection companion evidence or appendix material.

Main-table source CSV:

- `docs/paper/table_sources/phase4_formal_heldout_source_selected_main_20260713.csv`

{markdown_table(main_df, ["Reward", "Processed", "Drop", "Backlog", "D_sys", "Collision"])}

## Selected Versus Final Companion Table

Source CSV:

- `docs/paper/table_sources/phase4_formal_heldout_source_learned_selected_final_20260713.csv`

{markdown_table(learned_companion, ["Reward", "Processed", "Drop", "Backlog", "D_sys", "Collision"])}

Writing use: this companion table supports the checkpoint-selection narrative. RelCritic selected is both strongest and more stable than final; GlobalCritic remains high variance; MAPPO-like remains weak even after completing the third seed.

## Figures To Use

Source figure directory:

- `docs/paper/figure_sources/phase4_section5_performance_20260713/`

LaTeX-ready copies:

- `docs/paper/manuscript/figures/phase4_training_validation_curves_20260713.pdf`
- `docs/paper/manuscript/figures/phase4_main_performance_grouped_bars_20260713.pdf`
- `docs/paper/manuscript/figures/phase4_globalcritic_seed_level_20260713.pdf`

Figure roles:

- `phase4_training_validation_curves_20260713`: training/checkpoint-validation curves. Star marks selected best; square marks final.
- `phase4_main_performance_grouped_bars_20260713`: main performance grouped bars for reward, processed ratio, drop ratio, and `D_sys`; this avoids a reward-only presentation.
- `phase4_globalcritic_seed_level_20260713`: seed-level RelCritic versus GlobalCritic display. Use this when explaining the large GlobalCritic variance.

## GlobalCritic Seed-Level Interpretation

GlobalCritic should not be summarized only by the mean. The selected-checkpoint aggregate has large training-seed variance, so Section 5 should explicitly show the three seed-level points or error bars. The conservative claim is that global-only critic can occasionally find a usable seed, but it is not robust under this protocol.

## MAPPO-like Fairness Boundary

MAPPO-like is a same-scenario, same held-out seed, same hybrid/masked SAGIN interface learned baseline. It uses a flat learned actor/critic inside the same PPO/GAE-style training and evaluation pipeline, keeping reward, safety handling, action interface, and held-out protocol aligned.

Do not describe it as a faithful reproduction of a specific external MAPPO, MADDPG, or SAGIN paper. The paper-facing wording should be: MAPPO-like adapter baseline under our hybrid action interface, used to test whether a flat learned multi-agent baseline suffices when the topology-aware actor and relational critic are removed.

## Runtime And Resource Cost

Runtime/resource source CSV:

- `docs/paper/table_sources/phase4_runtime_resource_summary_20260713.csv`
- `docs/paper/table_sources/phase4_formal_eval_wallclock_summary_20260713.csv`

Training resource protocol: `num_envs=64`, `rollout_env_steps=250`, i.e. 16,000 environment transitions per update. Training was run on friday GPUs; exact GPU queue placement is retained in status logs, with the formal held-out queue split across GPU0/GPU1.

{chr(10).join(runtime_lines)}

Use these numbers to add a compact Section 5 paragraph or table reporting training cost, selected update, final update, and environment-step budget. For update-to-episode discussion, translate updates as `updates * 64` rollout episodes or `updates * 64 * 250` environment steps.

## Paper Claim Boundary

Supported now:

- RelCritic selected is the main source-scenario result and outperforms both fixed baselines and the learned ablations on reward, processed ratio, drop ratio, backlog, and `D_sys`.
- The relational critic ablation is strong evidence because GlobalCritic has high variance and worse mean performance under the same training/evaluation interface.
- MAPPO-like is complete at 3 seeds and provides a fair in-pipeline learned baseline, but only within the adapter-baseline boundary above.
- Final checkpoints are useful for stability/selection analysis, not for replacing the selected-checkpoint main result.

Not supported without additional experiments:

- Claims that this MAPPO-like baseline reproduces or defeats any particular external MAPPO/MADDPG implementation.
- Claims that final checkpoint behavior is the primary performance target.
- Broad larger-scale generalization claims from this Section 5 source-scenario evidence alone.
"""
    (EXP_DIR / "section5_performance_assets_20260713.md").write_text(note)


def main() -> None:
    for path in (TABLE_DIR, FIG_SRC_DIR, FIG_MANUSCRIPT_DIR, EXP_DIR):
        path.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    raw_df = load_raw_formal_rows()
    seed_df, learned_companion, main_df = aggregate_tables(raw_df)
    curve_df, runtime_df, selected_updates, final_updates = load_training_assets(raw_df)
    wall_df = formal_eval_wallclock_summary()

    raw_df.to_csv(TABLE_DIR / "phase4_formal_heldout_source_raw_rows_20260713.csv", index=False)
    seed_df.to_csv(TABLE_DIR / "phase4_formal_heldout_source_learned_seed_rows_20260713.csv", index=False)
    learned_companion.to_csv(
        TABLE_DIR / "phase4_formal_heldout_source_learned_selected_final_20260713.csv",
        index=False,
    )
    main_df.to_csv(TABLE_DIR / "phase4_formal_heldout_source_selected_main_20260713.csv", index=False)
    curve_df.to_csv(TABLE_DIR / "phase4_source_validation_checkpoint_curves_finalset_20260713.csv", index=False)
    runtime_df.to_csv(TABLE_DIR / "phase4_runtime_resource_summary_20260713.csv", index=False)
    wall_df.to_csv(TABLE_DIR / "phase4_formal_eval_wallclock_summary_20260713.csv", index=False)

    plot_training_curves(curve_df, selected_updates, final_updates)
    plot_main_bars(main_df)
    plot_globalcritic_seed_level(seed_df)
    write_freeze_note(main_df, learned_companion, runtime_df)

    print("generated section5 performance assets")
    print(f"raw_rows={len(raw_df)}")
    print(f"seed_rows={len(seed_df)}")
    print(f"main_rows={len(main_df)}")
    print(f"curve_rows={len(curve_df)}")
    print(f"runtime_rows={len(runtime_df)}")
    print(f"wall_rows={len(wall_df)}")


if __name__ == "__main__":
    main()
