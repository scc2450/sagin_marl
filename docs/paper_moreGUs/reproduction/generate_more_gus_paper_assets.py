#!/usr/bin/env python3
"""Register compact 100-GU evidence, then render single-panel manuscript PDFs.

--register reads the explicitly named completed runs. Ordinary rendering reads
only registered evidence; raw logs and episode-level evaluation dumps stay out
of docs/paper. Historical figures and selected/final diagnostics are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sagin_marl_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_more_gus_training_figures import Inputs, load_run, load_episodes, attach_continuation
from generate_section5_single_panel_figures_20260714 import configure_style

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "docs/paper_moreGUs/evidence_tables/more_gus_20260925"
FIGURES = ROOT / "docs/paper_moreGUs/manuscript_overleaf/figures"
PREVIEWS = ROOT / "docs/paper_moreGUs/reproduction/generated_figures/more_gus_20260925"
TRAIN = ROOT / "runs/experiments/more_gus_training"
RUN_NAMES = {
    45211: "bootstrap_phase4_3uav100gu_k1_seed45211_20260920_120135",
    45210: "bootstrap_phase4_3uav100gu_k1_seed45210_20260920_183803",
    61723: "bootstrap_phase4_3uav100gu_k1_seed61723_20260920_183803",
}
CONTINUATION = "bootstrap_phase4_3uav100gu_k1_seed45211_u500_to_u700_20260922_132212"
SCAN = ROOT / "runs/experiments/more_gus_scans/stars_dqs_revised_k1_20260925"
GC = TRAIN / "global_critic_3seed_k1_20260924"
METHODS = ["stars", "distributed_queue_c", "maxweight_lyapunov", "queue_aware_bw", "static_uniform"]
LABELS = dict(zip(METHODS, ["STARS", "DQS", "Lyapunov", "QBS", "Uniform"]))
COLORS = dict(zip(METHODS, ["#0072B2", "#009E73", "#E69F00", "#56B4E9", "#666666"]))
STYLES = dict(zip(METHODS, ["-", "-", "--", ":", ":"]))
MARKERS = dict(zip(METHODS, ["o", "D", "P", "v", "x"]))
SEED_COLORS = dict(zip(RUN_NAMES, ["#0072B2", "#009E73", "#E69F00"]))
PANELS = [
    ("reward_sum", "Reward", 1., "reward"),
    ("processed_ratio_eval", "Processed ratio (%)", 100., "processed_ratio"),
    ("drop_ratio_eval", "Drop ratio (%)", 100., "drop_ratio"),
    ("D_sys_report", "Queue-based delay proxy", 1., "delay_proxy"),
    ("pre_backlog_steps_eval", "Pre-backlog (slots)", 1., "pre_backlog"),
    ("queue_total_mean", "Queue workload (Mbit)", 1e-6, "queue_workload"),
]
METRICS = [p[0] for p in PANELS] + ["collision_episode_fraction"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def validate_critic_configs(reference, candidate):
    differences = {k for k in reference.keys() | candidate.keys()
                   if reference.get(k) != candidate.get(k)}
    # GC serializes later-added heuristic defaults; learned actions do not use them.
    defaults = {"baseline_dq_movement_weight": .02, "baseline_dq_switch_weight": .01}
    serialized_defaults = {k for k, value in defaults.items()
                           if k not in reference and candidate.get(k) == value}
    if (differences - serialized_defaults != {"critic_value_mode"}
            or reference.get("critic_value_mode") != "relational"
            or candidate.get("critic_value_mode") != "global_only"):
        raise ValueError(f"GC changes beyond critic/default serialization: {differences}")
    return {k: [reference.get(k), candidate.get(k)] for k in sorted(differences)}


def register_evidence():
    inputs = Inputs()
    review = inputs.json(TRAIN / "analysis/data/provenance.json")
    scan_manifest = inputs.json(SCAN / "manifest.json")
    if inputs.json(SCAN / "verification.json")["status"] != "passed":
        raise ValueError("Unverified revised scan")
    scan = inputs.csv(SCAN / "figures/per_training_seed.csv")
    scan = scan[scan.checkpoint_kind.isin(["selected", "fixed"])].copy()
    scan["paper_label"] = scan.method.map(LABELS)
    scan["episodes"] = 64
    scan["eval_seed_bases"] = "1980000;1981000"
    scan["source_metrics"] = relative(SCAN / "figures/per_training_seed.csv")
    scan["metrics_row_selector"] = scan.apply(lambda r: f"point={r.point};method={r.method};training_seed={r.training_seed};checkpoint_kind={r.checkpoint_kind}", axis=1)
    if len(scan) != 98 or scan.paper_label.isna().any():
        raise ValueError("Incomplete primary scan summaries")
    curves, selected, selections, gc_curves, stops = [], [], [], [], []
    config_differences = {}
    original = {}
    resumed = load_run(TRAIN / CONTINUATION, inputs)
    for seed, name in RUN_NAMES.items():
        run = load_run(TRAIN / name, inputs)
        original[seed] = run
        curve = attach_continuation(run, resumed) if seed == 45211 else run["curve"].assign(phase="original")
        curves.append(curve[["update", "phase", *METRICS[:-2], "collision_episode_fraction"]].assign(training_seed=seed))
        update = int(run["curve"].loc[run["curve"].model_improved > 0, "update"].iloc[-1])
        selections.append(dict(training_seed=seed, selected_update=update,
            original_final_update=int(run["result"]["stop"]["completed_updates"])))
        rows = load_episodes(run, "selected", inputs)
        values = rows[[*METRICS, "gu_queue_mbit", "uav_queue_mbit", "sat_queue_mbit", "queue_total_mbit"]].mean().to_dict()
        selected.append(dict(method="STARS", training_seed=seed, checkpoint_update=update,
            episodes=len(rows), eval_seed_bases="1980000;1981000", source_run=relative(run["path"]), **values))
    reference = {k: v for k, v in original[45211]["config"].items() if k != "seed"}
    for seed, run in original.items():
        if {k: v for k, v in run["config"].items() if k != "seed"} != reference:
            raise ValueError("Independent STARS run configs differ beyond seed")
        gc = load_run(GC / f"seed{seed}", inputs)
        config_differences[str(seed)] = validate_critic_configs(run["config"], gc["config"])
        gc_curves.append(gc["curve"][["update", *[m for m in METRICS if m in gc["curve"]]]].assign(training_seed=seed, method="STARS-GC"))
        update = int(gc["curve"].loc[gc["curve"].model_improved > 0, "update"].iloc[-1])
        rows = load_episodes(gc, "selected", inputs)
        values = rows[[*METRICS, "gu_queue_mbit", "uav_queue_mbit", "sat_queue_mbit", "queue_total_mbit"]].mean().to_dict()
        selected.append(dict(method="STARS-GC", training_seed=seed, checkpoint_update=update,
            episodes=len(rows), eval_seed_bases="1980000;1981000", source_run=relative(gc["path"]), **values))
        stops.append(dict(training_seed=seed, completed_updates=gc["result"]["stop"]["completed_updates"],
            stop_reason=gc["result"]["stop"]["stop_reason"], selected_update=update, status="pending_review"))
    curves = pd.concat(curves, ignore_index=True)
    selected = pd.DataFrame(selected)
    base_curves = curves[curves.phase == "original"].assign(method="STARS").drop(columns="phase")
    critic_curves = pd.concat([base_curves, *gc_curves], ignore_index=True)
    episode_path = TRAIN / RUN_NAMES[61723] / "renders/u700_seed1980000_native/episode.json"
    episode = inputs.json(episode_path)
    hotspot = inputs.json(TRAIN / RUN_NAMES[61723] / "renders/hotspot_tape_seed1980000.json")
    check = hotspot["checks"][str(episode["checkpoint_update"])]
    if (not check["initial_state_match"] or check["saved_episode_sha256"] != sha(episode_path)
            or hotspot["episode_seed"] != episode["seed"]):
        raise ValueError("Unmatched hotspot overlay")
    states, traces = episode["states"], episode["step_traces"][0]
    if [s["t"] for s in states] != list(range(250)) or len(traces) != 250:
        raise ValueError("Incomplete illustrative episode")
    paths = pd.DataFrame([dict(step=s["t"], uav=u, x=xy[0], y=xy[1])
                          for s in states for u, xy in enumerate(s["uav_pos"])])
    gu = np.array(states[0]["gu_pos"])
    mask = np.array(hotspot["hotspot_mask"], dtype=float)
    if mask.shape != (250, 100):
        raise ValueError("Wrong hotspot dimensions")
    users = pd.DataFrame(dict(gu_id=np.arange(len(gu)), x=gu[:, 0], y=gu[:, 1], hotspot_fraction=mask.mean(0)))
    trace = pd.DataFrame(traces)[["t", "reward", "gu_queue_sum", "uav_queue_sum", "sat_queue_sum"]]
    tables = dict(scan_seed_means=scan, training_curves=curves, training_selection=pd.DataFrame(selections),
        selected_policy_means=selected, critic_curves=critic_curves, critic_status=pd.DataFrame(stops),
        episode_paths=paths, episode_users=users, episode_traces=trace)
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(TABLES / f"{name}.csv", index=False)
    manifest = dict(scenario="3UAV/100GU/22clusters", T_steps=250, bandwidth_interval=1, satellite_interval=1,
        source_sha256=inputs.hashes, scan_source_commits=scan_manifest["execution_sources"],
        independent_training_seeds=list(RUN_NAMES), continuation=CONTINUATION,
        continuation_role="Only seed45211 resumes after u500; reward early stopping disabled; not a fourth seed",
        review_source=relative(TRAIN / "analysis"), review_episode=review["episode"],
        eval_protocol="Selected-checkpoint screening: 64 episodes, seed bases 1980000/1981000, deterministic, num_envs=32",
        validation_protocol="32 fixed validation episodes per checkpoint, seed base 910000; raw means, no smoothing",
        selected_final_comparison_exported=False, gc_status="pending_review",
        gc_config_differences=config_differences,
        gc_config_note="DQS penalty fields absent in original STARS config are serialized as .02/.01 defaults in GC; learned-policy execution does not read these heuristic-only fields",
        gc_caveat="Training degradation and collision cases remain unexplained; separate critic ablation, not an independent learned algorithm",
        episode_note="Illustrative one-env episode; paths t=0..249, post-step queue/reward t=1..250; not aggregate evidence",
        map_size=original[61723]["config"]["map_size"],
        table_sha256={name + ".csv": sha(TABLES / f"{name}.csv") for name in tables})
    write_json(TABLES / "manifest.json", manifest)
    index = TABLES.parent / "registry_index.csv"
    columns = ["file", "row_count", "table_families", "row_scopes", "scenario_count", "sample_scenarios"]
    registry = pd.read_csv(index, keep_default_na=False) if index.exists() else pd.DataFrame(columns=columns)
    prefix = relative(TABLES).split("evidence_tables/")[1] + "/"
    registry = registry[~registry.file.str.startswith(prefix)].copy()
    records = [dict(file=prefix + name + ".csv", row_count=len(frame), table_families="more_gus_paper_assets",
        row_scopes="illustrative episode" if name.startswith("episode_") else "validation or selected-policy summary",
        scenario_count=1, sample_scenarios="3uav100gu_22clusters_k1") for name, frame in tables.items()]
    pd.concat([registry, pd.DataFrame(records)], ignore_index=True).to_csv(index, index=False)
    return manifest


def nice_ceiling(value):
    if value <= 0:
        return 1.
    unit = 10 ** math.floor(math.log10(value))
    return math.ceil(value / unit * 4) / 4 * unit


def focused_limits(metric, core_low, core_high):
    """Fixed rule using all strong-method points, never per-point axis choices."""
    low, high = float(np.nanmin(core_low)), float(np.nanmax(core_high))
    if metric == "processed_ratio_eval":
        return max(0., math.floor((low - 1.) / 5.) * 5.), 101.
    return 0., nice_ceiling(max(high, 1e-9) * 1.10)


class Renderer:
    def __init__(self, tables=TABLES, figures=FIGURES, previews=PREVIEWS):
        self.tables, self.figures, self.previews = tables, figures, previews
        self.records = []
        self.figures.mkdir(parents=True, exist_ok=True)
        self.previews.mkdir(parents=True, exist_ok=True)

    def frame(self, name):
        return pd.read_csv(self.tables / (name + ".csv"))

    def canvas(self, *, review=False, square=False):
        fig, ax = plt.subplots(figsize=(3.45, 3.2 if square else 2.72))
        fig.subplots_adjust(left=.19, right=.98, bottom=.22, top=.73 if review else .75)
        ax.grid(axis="y", color="#D9D9D9", linewidth=.45, alpha=.72)
        ax.set_axisbelow(True)
        ax.tick_params(length=3., width=.65)
        if review:
            fig.text(.98, .975, "PENDING REVIEW", ha="right", va="top", fontsize=6.2, color="#A33A20")
        return fig, ax

    def legend(self, ax, ncol=3, handles=None):
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.32), ncol=ncol,
                  frameon=False, handlelength=1.45, handletextpad=.3, columnspacing=.72, borderaxespad=0.)

    def save(self, fig, name, caption, tables, *, status="candidate", **extra):
        stem = f"sec5_100gu_{name}_20260925"
        pdf, png = self.figures / (stem + ".pdf"), self.previews / (stem + ".png")
        fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(png, dpi=240, facecolor="white")
        self.records.append(dict(name=stem, pdf=relative(pdf), pdf_sha256=sha(pdf),
            status=status, caption=caption, source_tables=tables, **extra))
        plt.close(fig)

    def scans(self):
        data = self.frame("scan_seed_means")
        for axis in ("load", "resource"):
            for metric, label, factor, name in PANELS:
                fig, ax = self.canvas()
                series, lows, highs = [], [], []
                for method in METHODS:
                    part = data[(data.axis == axis) & (data.method == method)]
                    grouped = part.groupby("multiplier")[metric].agg(["mean", "std", "count"]).reset_index()
                    x = grouped.multiplier.to_numpy() * (40 if axis == "load" else 1)
                    mean = grouped["mean"].to_numpy() * factor
                    sd = grouped["std"].fillna(0).to_numpy() * factor if method == "stars" else np.zeros(len(mean))
                    if not (grouped["count"] == (3 if method == "stars" else 1)).all():
                        raise ValueError("Invalid number of training seeds in sweep")
                    series.append((method, x, mean, sd))
                    if method in METHODS[:3]:
                        lows.extend(mean - sd)
                        highs.extend(mean + sd)
                ylim = focused_limits(metric, lows, highs)
                clipped = []
                for method, x, mean, sd in series:
                    ax.errorbar(x, mean, yerr=sd if method == "stars" else None,
                        label=LABELS[method], color=COLORS[method], linestyle=STYLES[method], marker=MARKERS[method],
                        ms=3.4, lw=1.65 if method == "stars" else 1.15, elinewidth=.5, capsize=1.5, capthick=.5,
                        markeredgewidth=.65, zorder=5 if method == "stars" else 3)
                    for xx, yy in zip(x, mean):
                        if yy < ylim[0] or yy > ylim[1]:
                            edge = ylim[0] if yy < ylim[0] else ylim[1]
                            ax.scatter([xx], [edge], marker="v" if yy < ylim[0] else "^", s=16,
                                facecolor="white", edgecolor=COLORS[method], linewidth=.7, clip_on=False, zorder=6)
                            clipped.append(dict(method=LABELS[method], x=float(xx), value=float(yy)))
                ax.set_ylim(*ylim)
                ax.set_xticks(series[0][1])
                ax.set_xticklabels([f"{v:g}" for v in series[0][1]])
                ax.set_xlabel("Total task arrival rate (Mbit/s)" if axis == "load" else "Joint resource multiplier")
                ax.set_ylabel(label)
                self.legend(ax)
                if clipped:
                    fig.text(.19, .025, "Open boundary markers denote off-scale values.", fontsize=5.8, color="#555555")
                caption = ("STARS, DQS and Lyapunov define the displayed y-range; QBS/Uniform may be off-scale. "
                    "STARS error bars: descriptive SD of three independent training-seed means. Fixed policies: means of 64 paired screening episodes; "
                    "no training-seed error bars. Resource scaling jointly changes access bandwidth, backhaul and CPU. D_sys is a queue-based proxy.")
                self.save(fig, f"{axis}_{name}", caption, ["scan_seed_means.csv"], axis_limits=list(ylim), off_scale=clipped)

    def training(self):
        data = self.frame("training_curves")
        panels = [PANELS[0], ("best_reward", "Best validation reward", 1., "best_so_far_reward"), *PANELS[1:5]]
        for metric, label, factor, name in panels:
            fig, ax = self.canvas()
            if metric == "best_reward":
                data = data.sort_values(["training_seed", "update"]).copy()
                data[metric] = data.groupby("training_seed").reward_sum.cummax()
            values = data[metric].to_numpy() * factor
            for seed, color in SEED_COLORS.items():
                rows = data[data.training_seed == seed].sort_values("update")
                base, tail = rows[rows.phase == "original"], rows[rows.phase == "continuation"]
                ax.plot(base["update"], base[metric] * factor, color=color, lw=1.15, label=f"Seed {seed}")
                if len(tail):
                    joined = pd.concat([base.tail(1), tail])
                    ax.plot(joined["update"], joined[metric] * factor, color=color, lw=1.15, ls="--")
                    ax.axvline(base["update"].max(), color="#999999", lw=.6, ls=":")
            ax.set_xlim(0, 715)
            ax.set_xticks([0, 100, 200, 300, 400, 500, 600, 700])
            ax.set_xlabel("Training update")
            if metric in {"D_sys_report", "pre_backlog_steps_eval"}:
                if (values <= 0).any():
                    raise ValueError("Non-positive training metric on log scale")
                ax.set_yscale("log")
                ax.set_ylim(values.min() / 1.3, values.max() * 1.3)
                label += " (log scale)"
            elif metric == "drop_ratio_eval":
                ax.set_yscale("symlog", linthresh=.05, linscale=.7)
                ax.set_ylim(0, values.max() * 1.2)
                label += " (symlog)"
            else:
                ax.set_ylim(0, 101 if metric == "processed_ratio_eval" else nice_ceiling(values.max() * 1.05))
            ax.set_ylabel(label)
            self.legend(ax)
            fig.text(.19, .025, "Dashed: seed 45211 continuation after u500.", fontsize=5.8, color="#555555")
            self.save(fig, "training_" + name, "32 fixed validation episodes per point; raw means without smoothing. "
                "Only seed 45211 resumes after u500 with reward early stopping disabled; not a fourth independent seed. "
                "Best-so-far is not a convergence claim; no selected/final comparison panel is exported.", ["training_curves.csv"])

    def selected_queues(self):
        rows = self.frame("selected_policy_means")
        rows = rows[rows.method == "STARS"].set_index("training_seed").loc[list(RUN_NAMES)].reset_index()
        fig, ax = self.canvas()
        left = np.zeros(len(rows))
        for layer, label, color in (("gu", "GU", "#5DA5DA"), ("uav", "UAV", "#F17CB0"), ("sat", "SAT", "#777777")):
            values = rows[f"{layer}_queue_mbit"].to_numpy()
            if np.any(values > 1e-8):
                ax.barh(np.arange(len(rows)), values, left=left, height=.55, color=color, edgecolor="white", lw=.4, label=label)
            left += values
        for i, value in enumerate(left):
            ax.text(value + max(left) * .02, i, f"{value:.2f}", va="center", fontsize=6.6)
        ax.set_yticks(np.arange(len(rows)), [f"Seed {s}" for s in rows.training_seed])
        ax.invert_yaxis()
        ax.set_xlim(0, nice_ceiling(max(left) * 1.15))
        ax.set_xlabel("Mean queue workload (Mbit)")
        self.legend(ax)
        self.save(fig, "selected_queue_decomposition", "Selected STARS checkpoints only; 64 screening episodes per seed. "
            "Layer totals use 100 GU, 3 UAV and actual satellite counts. Zero SAT queues are omitted. No endpoint comparison.", ["selected_policy_means.csv"])

    def episode(self, manifest):
        paths, users, traces = (self.frame(name) for name in ("episode_paths", "episode_users", "episode_traces"))
        fig, ax = self.canvas(square=True)
        ax.scatter(users.x, users.y, s=3 + 28 * users.hotspot_fraction, color="#BBBBBB", edgecolors="#777777", lw=.25)
        for (u, rows), color in zip(paths.groupby("uav"), SEED_COLORS.values()):
            ax.plot(rows.x, rows.y, color=color, lw=.65, label=f"UAV {u+1}")
            ax.scatter(rows.x.iloc[0], rows.y.iloc[0], color=color, s=16, marker="o", zorder=4)
            ax.scatter(rows.x.iloc[-1], rows.y.iloc[-1], color=color, s=26, marker="*", zorder=4)
        ax.set(xlim=(-25, manifest["map_size"]+25), ylim=(-25, manifest["map_size"]+25), xlabel="x (m)", ylabel="y (m)")
        ax.set_aspect("equal")
        self.legend(ax)
        self.save(fig, "episode_trajectory", manifest["episode_note"] + "; seed 61723/u700, episode 1980000. GU marker area increases with hotspot occupancy; circles=start, stars=last recorded position.", ["episode_paths.csv", "episode_users.csv"])
        for kind in ("queues", "reward"):
            fig, ax = self.canvas()
            if kind == "queues":
                for layer, color in (("gu", "#5DA5DA"), ("uav", "#F17CB0")):
                    ax.plot(traces.t, traces[layer + "_queue_sum"] / 1e6, color=color, lw=1.0, label=layer.upper())
                self.legend(ax, ncol=2)
                ax.set_ylabel("Queue workload (Mbit)")
            else:
                ax.plot(traces.t, traces.reward, color=COLORS["stars"], lw=1.)
                ax.set_ylabel("Step reward")
            ax.set_xlim(0, 250)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("Completed environment steps")
            self.save(fig, "episode_" + kind, manifest["episode_note"], ["episode_traces.csv"])

    def critic(self):
        data = self.frame("critic_curves")
        fig, ax = self.canvas(review=True)
        colors = {"STARS": COLORS["stars"], "STARS-GC": "#D55E00"}
        for method, color in colors.items():
            block = data[data.method == method]
            for _, rows in block.groupby("training_seed"):
                ax.plot(rows["update"], rows.reward_sum, color=color, lw=.5, alpha=.28)
            grouped = block.groupby("update").reward_sum.agg(["mean", "std", "count"])
            grouped = grouped[grouped["count"] == 3]
            ax.plot(grouped.index, grouped["mean"], color=color, lw=1.5, ls="-" if method == "STARS" else "--", label=method)
            ax.fill_between(grouped.index, grouped["mean"]-grouped["std"], grouped["mean"]+grouped["std"], color=color, alpha=.12, linewidth=0)
        ax.set(xlim=(0,715), ylim=(0,nice_ceiling(data.reward_sum.max()*1.1)), xlabel="Training update", ylabel="Validation reward")
        self.legend(ax, ncol=2)
        self.save(fig, "critic_ablation_training_review", "PENDING REVIEW: critic-only ablation. "
            "Thin lines are actual independent runs; thick means and SD bands require all 3 seeds at that update. "
            "No continuation, imputed tails or selected/final comparison. GC stops at u375; same stopping rule and cap 700, not equal executed updates. "
            "GC degradation and collisions are unresolved.", ["critic_curves.csv", "critic_status.csv"], status="pending_review")
        selected = self.frame("selected_policy_means")
        panels = PANELS[:4] + [("collision_episode_fraction", "Collision episodes (%)", 100., "collision")]
        for metric, label, factor, name in panels:
            fig, ax = self.canvas(review=True)
            for i, (method, color) in enumerate(colors.items()):
                values = selected[selected.method == method][metric].to_numpy() * factor
                ax.bar(i, values.mean(), yerr=values.std(ddof=1), width=.52, color=color, alpha=.8,
                    error_kw=dict(elinewidth=.6, capsize=2, capthick=.6))
                ax.scatter(i + np.linspace(-.11,.11,len(values)), values, s=12, facecolor="white", edgecolor="#333333", lw=.6, zorder=4)
            ax.set_xticks([0,1], ["STARS", "STARS-GC"])
            ax.set_ylabel(label)
            ax.set_ylim(bottom=0)
            self.save(fig, "critic_ablation_" + name + "_review", "PENDING REVIEW: existing selected checkpoints; "
                "64 paired screening episodes per training seed, 3 independent seeds per method. Bars=mean, error bars=training-seed SD, "
                "dots=seed means. The axis has a physical lower bound of zero; any negative SD extent is clipped. "
                "GC is a critic ablation, not an independent algorithm. Degradation/collision causes remain unresolved.",
                ["selected_policy_means.csv", "critic_status.csv"], status="pending_review")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true", help="Refresh compact evidence from the registered run families")
    parser.add_argument("--check_only", action="store_true")
    args = parser.parse_args()
    manifest = register_evidence() if args.register else json.loads((TABLES / "manifest.json").read_text())
    for name, digest in manifest["table_sha256"].items():
        if sha(TABLES / name) != digest:
            raise ValueError(f"Registered evidence hash mismatch: {name}")
    if args.check_only:
        print(f"Verified {len(manifest['table_sha256'])} registered tables")
        return
    configure_style()
    renderer = Renderer()
    renderer.scans()
    renderer.training()
    renderer.selected_queues()
    renderer.episode(manifest)
    renderer.critic()
    write_json(TABLES / "figure_index.json", dict(
        generator=relative(Path(__file__)), generator_sha256=sha(Path(__file__)),
        style_helper="docs/paper/reproduction/generate_section5_single_panel_figures_20260714.py",
        style_helper_sha256=sha(ROOT / "docs/paper/reproduction/generate_section5_single_panel_figures_20260714.py"),
        manifest_sha256=sha(TABLES / "manifest.json"), figures=renderer.records,
        selected_final_comparison_exported=False, formats=["pdf"], preview_format="png"))
    print(f"Rendered {len(renderer.records)} single-panel PDFs; PNG previews in {relative(PREVIEWS)}")


if __name__ == "__main__":
    main()
