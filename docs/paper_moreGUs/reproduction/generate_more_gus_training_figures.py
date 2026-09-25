#!/usr/bin/env python3
"""Plot completed training campaigns without training, GPU access, or extrapolation.

Read explicit run directories, reuse the Section 5 style/best-so-far helper,
and save a single report with PNG/PDF figures and traceable intermediate data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sagin_marl_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sagin_marl_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parents[2] / "paper/reproduction"))
from generate_section5_single_panel_figures_20260714 import (
    _best_so_far_df,
    configure_style,
)

COLORS = ("#0072B2", "#009E73", "#E69F00")
SUMMARY_METRICS = (
    "reward_sum", "processed_ratio_eval", "drop_ratio_eval",
    "pre_backlog_steps_eval", "D_sys_report", "queue_total_mean",
    "gu_queue_mean", "uav_queue_mean", "sat_queue_mean",
    "outflow_arrival_ratio", "sat_incoming_arrival_ratio",
    "sat_processed_arrival_ratio",
)


class Inputs:
    def __init__(self) -> None:
        self.hashes: dict[str, str] = {}

    @staticmethod
    def key(path: Path) -> str:
        try:
            return path.resolve().relative_to(Path(__file__).resolve().parents[3]).as_posix()
        except ValueError:
            return str(path.resolve())

    def read(self, path: Path) -> bytes:
        data = path.read_bytes()
        self.hashes[self.key(path)] = hashlib.sha256(data).hexdigest()
        return data

    def json(self, path: Path) -> dict:
        return json.loads(self.read(path))

    def csv(self, path: Path) -> pd.DataFrame:
        import io
        return pd.read_csv(io.BytesIO(self.read(path)))


def load_run(path: Path, inputs: Inputs) -> dict:
    manifest = inputs.json(path / "manifest.json")
    result = inputs.json(path / "result.json")
    status = inputs.json(path / "status.json")
    if status.get("status") != "complete" or (path / "failure.json").exists():
        raise ValueError(f"Incomplete/failed campaign: {path}")
    config = yaml.safe_load(inputs.read(path / "config.yaml"))
    curve = inputs.csv(path / "train/checkpoint_eval.csv")
    curve["update"] = curve["update"].astype(int)
    if curve["update"].duplicated().any() or not curve["update"].is_monotonic_increasing:
        raise ValueError(f"Unordered/duplicate updates: {path}")
    if not np.isfinite(curve[["reward_sum", "pre_backlog_steps_eval", "D_sys_report"]]).all().all():
        raise ValueError(f"Non-finite validation data: {path}")
    if not (curve["episodes"] == 32).all() or not (np.diff(curve["update"]) == 25).all():
        raise ValueError(f"Unexpected validation protocol: {path}")
    stop = result["stop"]
    if int(curve["update"].iloc[-1]) != int(stop["completed_updates"]):
        raise ValueError(f"Final validation is missing: {path}")
    return dict(path=path, manifest=manifest, result=result, config=config, curve=curve)


def attach_continuation(parent: dict, continuation: dict) -> pd.DataFrame:
    manifest = continuation["manifest"]
    end = int(parent["curve"]["update"].iloc[-1])
    if (manifest.get("seed") != parent["manifest"]["seed"]
            or manifest.get("start_update") != end
            or Path(manifest["reference_run"]).name != parent["path"].name):
        raise ValueError("Continuation must resume this parent with the same seed")
    differences = {k for k in parent["config"].keys() | continuation["config"].keys()
                   if parent["config"].get(k) != continuation["config"].get(k)}
    if differences != {"checkpoint_eval_reward_early_stop_enabled"}:
        raise ValueError(f"Unexpected continuation config changes: {differences}")
    if continuation["config"]["checkpoint_eval_reward_early_stop_enabled"]:
        raise ValueError("Expected explicitly disabled reward early stopping")
    if int(continuation["curve"]["update"].iloc[0]) != end + 25:
        raise ValueError("Continuation has a gap or an overlapping update")
    return pd.concat([
        parent["curve"].assign(phase="original"),
        continuation["curve"].assign(phase="continuation"),
    ], ignore_index=True)


def load_episodes(run: dict, variant: str, inputs: Inputs) -> pd.DataFrame:
    frames = []
    for seed_base in run["manifest"]["postrun_seed_bases"]:
        label = f"{variant}_seed{seed_base}"
        folder = run["path"] / "evaluations" / label
        meta = inputs.json(folder / f"{label}_summary.json")
        if (meta["policy_mode"] != "deterministic" or meta["episodes"] != 32
                or meta["num_envs"] != 32
                or meta["episode_seed_base"] != seed_base
                or meta["access_bw_decision_interval"] != 1
                or meta["sat_decision_interval"] != 1):
            raise ValueError(f"Unmatched evaluation protocol: {folder}")
        data = inputs.csv(folder / f"{label}_episodes.csv")
        if len(data) != 32 or data["episode"].duplicated().any():
            raise ValueError(f"Invalid episode coverage: {folder}")
        if not np.isfinite(data[list(SUMMARY_METRICS)]).all().all():
            raise ValueError(f"Missing/non-finite metrics: {folder}")
        for metric in SUMMARY_METRICS:
            if not np.isclose(data[metric].mean(), meta["summary"][metric], rtol=1e-5, atol=1e-7):
                raise ValueError(f"CSV/summary mismatch: {folder}: {metric}")
        cfg = run["config"]
        layer_total = sum(data[f"{layer}_queue_mean"] * cfg[f"num_{layer}"]
                          for layer in ("gu", "uav", "sat"))
        if not np.allclose(layer_total, data["queue_total_mean"], rtol=1e-5, atol=1.0):
            raise ValueError(f"Per-node/layer-total queue mismatch: {folder}")
        for layer in ("gu", "uav", "sat"):
            data[f"{layer}_queue_mbit"] = data[f"{layer}_queue_mean"] * cfg[f"num_{layer}"] / 1e6
        data["queue_total_mbit"] = data["queue_total_mean"] / 1e6
        frames.append(data.assign(eval_seed_base=seed_base, checkpoint=variant))
    return pd.concat(frames, ignore_index=True)


def padded_limits(values: np.ndarray, fraction: float = 0.07) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    pad = max(float(np.ptp(values)) * fraction, max(abs(values).max(), 1.0) * 0.02)
    return float(values.min() - pad), float(values.max() + pad)


def polish(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.65, pad=4)


def save_figure(fig: plt.Figure, out: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        folder = out / ext
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"{name}.{ext}", dpi=220, facecolor="white")
    plt.close(fig)


def seed_legend(fig: plt.Figure, colors: dict[int, str], resumed_seed: int) -> None:
    handles = [Line2D([], [], color=color, lw=1.8, label=f"Seed {seed}")
               for seed, color in colors.items()]
    handles.append(Line2D([], [], color="#555555", ls="--", lw=1.5,
                          label=f"Seed {resumed_seed}: continuation"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.91), columnspacing=1.6)


def draw_curves(ax: plt.Axes, curves: pd.DataFrame, metric: str,
                colors: dict[int, str], scale: float = 1.0) -> None:
    for seed, group in curves.groupby("training_seed", sort=False):
        base = group[group.phase == "original"]
        ax.plot(base["update"], base[metric] * scale, color=colors[seed],
                lw=1.6, marker="o", ms=2.5)
        tail = group[group.phase == "continuation"]
        if not tail.empty:
            joined = pd.concat([base.tail(1), tail])
            ax.plot(joined["update"], joined[metric] * scale, color=colors[seed],
                    lw=1.6, ls="--", marker="o", ms=2.5)
            ax.axvline(float(base["update"].iloc[-1]), color="#999999", ls=":", lw=0.8)
    ax.set_xlim(0, float(curves["update"].max()) + 18)
    tick_step = max(25, int(np.ceil(float(curves["update"].max()) / 200)) * 25)
    ax.xaxis.set_major_locator(MultipleLocator(tick_step))
    ax.set_xlabel("Training update")
    polish(ax)


def plot_training(curves: pd.DataFrame, colors: dict, seed: int, out: Path) -> None:
    best = _best_so_far_df(curves)
    best = best.merge(curves[["training_seed", "update", "phase"]],
                      on=["training_seed", "update"], validate="one_to_one")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.19, top=0.76, wspace=0.17)
    for ax, data, title in zip(axes, (curves, best), ("(a) Raw validation reward", "(b) Best reward observed so far")):
        draw_curves(ax, data, "reward_sum", colors)
        ax.set_title(title, pad=10)
        ax.set_ylim(*padded_limits(curves["reward_sum"].to_numpy()))
        ax.set_ylabel("Episode reward")
    fig.suptitle("100-GU bootstrap training", y=0.98, fontsize=15)
    seed_legend(fig, colors, seed)
    fig.text(0.075, 0.045, "32 fixed validation episodes per point; raw means, no smoothing.\n"
             "Dashed: resumed after u500 with reward early stopping disabled. Best-so-far is not convergence.", fontsize=8.5)
    save_figure(fig, out, "training_reward")


def plot_services(curves: pd.DataFrame, colors: dict, seed: int, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.3), sharex=True)
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.12, top=0.85, hspace=0.43, wspace=0.27)
    metrics = [("processed_ratio_eval", "Processed ratio (%)", 100, "linear"),
               ("drop_ratio_eval", "Drop ratio (%)", 100, "symlog"),
               ("pre_backlog_steps_eval", "Pre-backlog (steps)", 1, "log"),
               ("D_sys_report", r"System delay $D_{\mathrm{sys}}$", 1, "log")]
    for i, (ax, (metric, label, factor, scale)) in enumerate(zip(axes.flat, metrics)):
        values = curves[metric].to_numpy() * factor
        draw_curves(ax, curves, metric, colors, factor)
        ax.set_title(f"({chr(97+i)}) {label}", pad=10)
        ax.set_ylabel(label + (f" [{scale}]" if scale != "linear" else ""))
        if scale == "symlog":
            ax.set_yscale("symlog", linthresh=0.05, linscale=0.7)
            ax.set_ylim(-0.004, max(values) * 1.4)
        elif scale == "log":
            if min(values) <= 0:
                raise ValueError(f"Non-positive value on log axis: {metric}")
            ax.set_yscale("log")
            ax.set_ylim(min(values) / 1.4, max(values) * 1.4)
        else:
            low, high = padded_limits(values)
            ax.set_ylim(low, max(high, 101.0))
        if scale != "linear":
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    fig.suptitle("Service quality during training", y=0.985, fontsize=15)
    seed_legend(fig, colors, seed)
    fig.text(0.085, 0.025, "Same checkpoint-validation episodes as the reward curves. All seeds use their actual update numbers.\n"
             "Log/symlog scales are explicit; no old-scenario reference lines are included.", fontsize=8.5)
    save_figure(fig, out, "training_service_quality")


def plot_checkpoints(summary: pd.DataFrame, colors: dict, seed: int, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.6))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.77, bottom=0.23, wspace=0.35)
    for ax, metric, label in zip(axes, ("reward_sum", "pre_backlog_steps_eval", "D_sys_report"),
                               ("Held-out reward", "Pre-backlog (steps)", r"System delay $D_{\mathrm{sys}}$")):
        for train_seed, color in colors.items():
            group = summary[summary.training_seed == train_seed].set_index("checkpoint")
            y = group.loc[["selected", "final"], metric].to_numpy()
            ax.plot([0, 1], y, color=color, lw=1.4, alpha=0.85)
            ax.scatter([0], y[:1], color=color, s=32, zorder=4)
            ax.scatter([1], y[1:], facecolors="white", edgecolors=color, marker="s", s=36, zorder=4)
        ax.set_xticks([0, 1], ["Selected", "Endpoint u700"])
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylim(*padded_limits(summary[metric].to_numpy(), 0.12))
        ax.set_title(label, pad=10)
        ax.set_ylabel(label)
        polish(ax)
    fig.suptitle("Selected policy versus training endpoint", y=0.985, fontsize=15)
    handles = [Line2D([], [], color=c, lw=1.7, label=f"Seed {s}") for s, c in colors.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.92))
    fig.text(0.065, 0.055, f"Each point: mean of 64 paired screening episodes. Selection used validation, not these episodes.\n"
             f"Seed {seed}: u700 is the continuation endpoint; original early-stop u500 remains in the report.\n"
             "Evaluation episodes are not independent training seeds; no cross-seed confidence interval is claimed.", fontsize=8.2)
    save_figure(fig, out, "selected_vs_endpoint")


def mechanism_decisions(summary: pd.DataFrame) -> dict:
    queue = summary["queue_total_mbit"]
    flow = summary[["outflow_arrival_ratio", "sat_incoming_arrival_ratio", "sat_processed_arrival_ratio"]].to_numpy()
    return dict(queue_min_mbit=float(queue.min()), queue_max_mbit=float(queue.max()),
                queue_panel=bool(queue.max() - queue.min() > 1 and queue.max() / max(queue.min(), 1e-9) > 1.5),
                flow_min_pct=float(flow.min() * 100), flow_max_pct=float(flow.max() * 100),
                flow_panel=bool(flow.min() < 0.99 or np.ptp(flow) >= 0.01),
                rule="Descriptive display filter, not a significance test: queue range >1 Mbit and max/min >1.5; flow below 99% or range >=1 percentage point.")


def plot_decomposition(summary: pd.DataFrame, decisions: dict, out: Path) -> None:
    labels = [f"{int(r.training_seed)}  {'selected' if r.checkpoint == 'selected' else 'u700'}"
              for r in summary.itertuples()]
    if decisions["queue_panel"]:
        fig, ax = plt.subplots(figsize=(8.6, 4.7))
        fig.subplots_adjust(left=0.21, right=0.96, bottom=0.17, top=0.78)
        left = np.zeros(len(summary))
        for layer, color in (("gu", "#5DA5DA"), ("uav", "#F17CB0"), ("sat", "#777777")):
            values = summary[f"{layer}_queue_mbit"].to_numpy()
            if np.any(values > 1e-8):
                ax.barh(np.arange(len(summary)), values, left=left, color=color,
                        height=0.65, label=f"{layer.upper()} queue", edgecolor="white", linewidth=0.4)
            left += values
        for i, total in enumerate(left):
            ax.text(total + max(left) * 0.015, i, f"{total:.2f}", va="center", fontsize=9)
        ax.set_yticks(np.arange(len(summary)), labels)
        ax.invert_yaxis()
        ax.set_xlim(0, max(left) * 1.17)
        ax.set_xlabel("Mean queue workload (Mbit; layer totals)")
        ax.grid(axis="x", color="#D9D9D9", lw=0.5)
        ax.set_axisbelow(True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)
        fig.suptitle("Where the queue workload accumulates", y=0.98, fontsize=15)
        fig.text(0.21, 0.035, "Means over 64 screening episodes. SAT queues are zero and omitted.\n"
                 "Seed 45211 u700 is a continuation, not the original stopping point.", fontsize=8.3)
        save_figure(fig, out, "queue_decomposition")
    if decisions["flow_panel"]:
        fig, ax = plt.subplots(figsize=(8.6, 4.7))
        for offset, key, label, color in [(-0.22, "outflow_arrival_ratio", "Access", "#0072B2"),
                                        (0, "sat_incoming_arrival_ratio", "Backhaul", "#009E73"),
                                        (0.22, "sat_processed_arrival_ratio", "Processed", "#E69F00")]:
            ax.barh(np.arange(len(summary)) + offset, summary[key] * 100,
                    height=0.19, label=label, color=color)
        ax.set_yticks(np.arange(len(summary)), labels)
        ax.invert_yaxis()
        ax.set_xlabel("Flow / arrivals (%)")
        ax.legend(frameon=False)
        fig.tight_layout()
        save_figure(fig, out, "flow_decomposition")


def plot_episode(path: Path, hotspot: Path | None, inputs: Inputs, out: Path) -> dict:
    data = inputs.json(path)
    states = data["states"]
    traces = data["step_traces"][0]
    if [s["t"] for s in states] != list(range(len(states))) or len(traces) != len(states):
        raise ValueError("Incomplete episode timeline")
    positions = np.asarray([s["uav_pos"] for s in states])
    gu = np.asarray(states[0]["gu_pos"])
    sizes = np.full(len(gu), 13.0)
    if hotspot is not None:
        tape = inputs.json(hotspot)
        check = tape["checks"][str(data["checkpoint_update"])]
        if (tape["episode_seed"] != data["seed"] or not check["initial_state_match"]
                or check["saved_episode_sha256"] != inputs.hashes[inputs.key(path)]):
            raise ValueError("Hotspot tape does not match this captured trajectory")
        mask = np.asarray(tape["hotspot_mask"], dtype=float)
        if mask.shape != (len(states), len(gu)):
            raise ValueError("Hotspot/trajectory shape mismatch")
        sizes += 160 * mask.mean(axis=0)
    fig = plt.figure(figsize=(11.4, 5.5))
    grid = fig.add_gridspec(2, 2, left=0.065, right=0.98, bottom=0.19, top=0.83,
                           width_ratios=[1.05, 1.15], wspace=0.3, hspace=0.45)
    ax = fig.add_subplot(grid[:, 0])
    ax.scatter(gu[:, 0], gu[:, 1], s=sizes, color="#BBBBBB", edgecolors="#777777", lw=0.3, zorder=2)
    for i, color in enumerate(COLORS):
        ax.plot(positions[:, i, 0], positions[:, i, 1], color=color, lw=1.0, alpha=0.85, label=f"UAV {i+1}")
        ax.scatter(*positions[0, i], color=color, s=35, marker="o", zorder=4)
        ax.scatter(*positions[-1, i], color=color, s=60, marker="*", zorder=4)
    cfg = yaml.safe_load(inputs.read(path.parents[2] / "config.yaml"))
    ax.set_xlim(-25, cfg["map_size"] + 25)
    ax.set_ylim(-25, cfg["map_size"] + 25)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("(a) Recorded paths, t = 0...249", pad=10)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 0.94))
    queue_ax = fig.add_subplot(grid[0, 1])
    reward_ax = fig.add_subplot(grid[1, 1], sharex=queue_ax)
    time = np.array([s["t"] for s in traces])
    queue_ax.plot(time, [s["gu_queue_sum"] / 1e6 for s in traces], color="#5DA5DA", lw=1.2, label="GU")
    queue_ax.plot(time, [s["uav_queue_sum"] / 1e6 for s in traces], color="#F17CB0", lw=1.2, label="UAV")
    queue_ax.set_ylabel("Queue (Mbit)")
    queue_ax.set_title("(b) Post-step queues", pad=8)
    queue_ax.legend(ncol=2, frameon=False, loc="upper right")
    reward_ax.plot(time, [s["reward"] for s in traces], color="#0072B2", lw=1.0)
    reward_ax.set_ylabel("Step reward")
    reward_ax.set_title("(c) Environment reward", pad=8)
    reward_ax.set_xlabel("Completed environment steps")
    for target, values in [(queue_ax, [s["gu_queue_sum"] / 1e6 for s in traces] + [s["uav_queue_sum"] / 1e6 for s in traces]),
                           (reward_ax, [s["reward"] for s in traces])]:
        target.set_xlim(0, len(states) + 5)
        target.set_ylim(*padded_limits(np.array(values)))
        polish(target)
    fig.suptitle(f"Policy case study: seed {cfg['seed']}, u{data['checkpoint_update']}, episode {data['seed']}", y=0.985, fontsize=14)
    gu_note = "Grey GU marker size increases with time spent in a hotspot." if hotspot else "Grey markers: ground users."
    fig.text(0.065, 0.06, gu_note + " Circles: start; stars: last recorded position.\n"
             "Pre-action positions end at t249; queue/reward traces include all 250 steps. One illustrative episode, not an aggregate test.", fontsize=8.2)
    save_figure(fig, out, "policy_episode")
    return dict(seed=data["seed"], training_seed=cfg["seed"], update=data["checkpoint_update"],
                reward=data["summary"]["reward_sum"], path_lengths=data["path_lengths_m"],
                hotspot_overlay=hotspot is not None)


def write_report(out: Path, summary: pd.DataFrame, parent: dict, decisions: dict,
                 colors: dict, episode: dict | None) -> None:
    lines = ["# 100 GU 训练结果图表汇总", "", "## 数据口径", "",
             "场景为 3 UAV / 100 GU / 22 clusters，T=250，BW/SAT 决策间隔均为 1。这里只整理 bootstrap 训练，不加入旧场景基线或尚待重测的扫描数据。",
             "", "三个独立训练 seed 为 45211、45210、61723。45211 仅接入保持原 seed 的 u500→u700 续训，未纳入 rollseed55211 分支。续训关闭 reward early stopping，因此 u500 后使用虚线；checkpoint 接续不等价于未中断训练，不能把它当成额外的独立 seed。",
             "", "validation 每 25 updates 使用固定 32 episodes（seed base 910000）。Selected 由原训练的选择规则决定，不按本次评估结果重新挑选。评估使用 seed bases 1980000/1981000，各 32 episodes；这是已有的 held-out screening，不是后续调参后的全新独立测试集。",
             "", "## 训练过程", "", "![训练 reward](png/training_reward.png)", "",
             "左图保留原始回撤，右图仅表示截至该 update 已观测到的最好 reward，不代表当前策略稳定性。两图使用相同横纵尺度，按真实 update 绘制，没有平滑、补点或外推。",
             "", "![服务指标](png/training_service_quality.png)", "",
             "Processed/drop 表示完成与丢弃比例，backlog 衡量队列负担，D_sys 使用评估器现有的系统时延指标口径，并非额外测量的逐任务时延分位数。drop 使用保留零值的 symlog，backlog/D_sys 使用 log；尺度已标在图中。",
             "", "## Selected 与训练终点", "", "![Selected 与终点](png/selected_vs_endpoint.png)", "",
             "每个点是一个训练 seed 的 checkpoint 在同一组 64 episodes 上的均值。评估 episodes 不作为独立训练重复，本图不计算跨训练 seed 的置信区间。",
             "", "| 训练 seed | Checkpoint | Update | Reward | Processed (%) | Drop (%) | Backlog | D_sys |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples():
        lines.append(f"| {row.training_seed} | {row.checkpoint} | {row.checkpoint_update} | {row.reward_sum:.3f} | {100*row.processed_ratio_eval:.4f} | {100*row.drop_ratio_eval:.4f} | {row.pre_backlog_steps_eval:.4f} | {row.D_sys_report:.4f} |")
    old = [v for k, v in parent["result"]["evaluations"].items() if k.startswith("final_")]
    original_reward = sum(v["reward_sum"] * v["episodes"] for v in old) / sum(v["episodes"] for v in old)
    lines.extend(["", f"45211 原始训练在 u500 以 checkpoint_reward_plateau 停止，当时 final screening reward 为 {original_reward:.3f}。上表 final 使用其续训 u700，不能据此改写原始早停结果。61723 selected 与 final 均为 u700，独立重复评估产生的微小数值差别不代表不同策略。",
                  "", "## 队列与流量图筛选", "",
                  f"Selected/endpoint 的平均总队列范围为 {decisions['queue_min_mbit']:.2f}–{decisions['queue_max_mbit']:.2f} Mbit。队列图{'保留' if decisions['queue_panel'] else '省略'}。按 config 中真实节点数把每节点均值换算为层总量，并检查 GU+UAV+SAT 与总队列一致，没有沿用旧脚本的 20 GU 常量。"])
    if decisions["queue_panel"]:
        lines.extend(["", "![队列分解](png/queue_decomposition.png)", "",
                      "GU/UAV 的积压分布及总量差异为 backlog 变化提供补充描述；它们本身不证明训练回撤的因果机制。"])
    lines.extend(["", f"接入/回传/处理流量与到达量之比均落在 {decisions['flow_min_pct']:.3f}%–{decisions['flow_max_pct']:.3f}% 范围。{'保留流量图。' if decisions['flow_panel'] else '三环节均接近满额且绝对差异不足 1 个百分点，因此省略流量图，不放大坐标制造差异。'}",
                  "", "显示筛选仅为可读性规则，不是显著性检验：队列范围超过 1 Mbit 且最大/最小超过 1.5；流量任一项低于 99% 或范围达到 1 个百分点才保留。判定值记录于 data/provenance.json。"])
    if decisions["flow_panel"]:
        lines.extend(["", "![流量分解](png/flow_decomposition.png)"])
    if episode:
        lines.extend(["", "## 策略行为案例", "", "![策略行为](png/policy_episode.png)", "",
                      f"使用已有 native capture：训练 seed{episode['training_seed']}、u{episode['update']}、固定 episode seed {episode['seed']}，reward={episode['reward']:.3f}。这是已展示 episode 的静态整理，没有重新挑选最高 reward 轨迹。灰色 GU 点大小反映该 episode 的热点暴露时长；轨迹包含动作前 t0–249，队列与 reward 包含完整 250 步动作后结果。",
                      "", "此图只解释行为，不代表多 episode 的平均效果，也不能单凭轨迹断言策略优于尚待重测的基线。"])
    lines.extend(["", "## 文件与复现", "", "png/ 为查看用图，pdf/ 为矢量图；本次不输出 SVG，也不覆盖历史图。data/ 保留绘图曲线、逐 episode 指标、汇总表和输入文件 SHA256。命令见 data/provenance.json。旧论文样式及 best-so-far 逻辑来自现有 Section 5 生成器，其固定 u525/reward 范围限制已移除。",
                  "", "负载/带宽扫描、算法与基线对比等待新数据；本次未重新训练或使用 GPU。"])
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dirs", type=Path, nargs=3, required=True)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--episode_json", type=Path)
    parser.add_argument("--hotspot_json", type=Path)
    parser.add_argument("--check_only", action="store_true")
    args = parser.parse_args()
    inputs = Inputs()
    runs = [load_run(p, inputs) for p in args.run_dirs]
    continuation = load_run(args.continuation, inputs)
    seeds = [r["manifest"]["seed"] for r in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Run dirs must be distinct independent training seeds")
    reference = {k: v for k, v in runs[0]["config"].items() if k != "seed"}
    for run in runs:
        if {k: v for k, v in run["config"].items() if k != "seed"} != reference:
            raise ValueError("From-scratch configs differ by more than seed")
        for key in ("validation_seed_base", "postrun_seed_bases", "return_target", "num_envs", "rollout_env_steps"):
            if run["manifest"][key] != runs[0]["manifest"][key] or continuation["manifest"][key] != runs[0]["manifest"][key]:
                raise ValueError(f"Unmatched protocol: {key}")
    resumed_seed = continuation["manifest"]["seed"]
    parent = next(r for r in runs if r["manifest"]["seed"] == resumed_seed)
    curve_frames, episode_frames = [], []
    for run in runs:
        seed = run["manifest"]["seed"]
        curve = attach_continuation(run, continuation) if seed == resumed_seed else run["curve"].assign(phase="original")
        curve_frames.append(curve.assign(training_seed=seed))
        improved = run["curve"].loc[run["curve"]["model_improved"] > 0, "update"]
        selected_update = int(improved.iloc[-1])
        final_run = continuation if seed == resumed_seed else run
        for variant, source, update in [("selected", run, selected_update), ("final", final_run, int(final_run["result"]["stop"]["completed_updates"]))]:
            episode_frames.append(load_episodes(source, variant, inputs).assign(
                training_seed=seed, checkpoint_update=update, source_run=source["path"].name))
    curves = pd.concat(curve_frames, ignore_index=True)
    episodes = pd.concat(episode_frames, ignore_index=True)
    for seed, group in episodes.groupby("training_seed"):
        identities = [set(map(tuple, g[["eval_seed_base", "episode"]].to_numpy())) for _, g in group.groupby("checkpoint")]
        if identities[0] != identities[1] or len(identities[0]) != 64:
            raise ValueError(f"Selected/final episodes not paired: {seed}")
    metrics = list(SUMMARY_METRICS) + [f"{k}_queue_mbit" for k in ("gu", "uav", "sat")] + ["queue_total_mbit"]
    summary = episodes.groupby(["training_seed", "checkpoint", "checkpoint_update"], sort=False)[metrics].mean().reset_index()
    decisions = mechanism_decisions(summary)
    print(json.dumps(dict(validation_rows=len(curves), evaluation_rows=len(episodes), decisions=decisions), indent=2))
    if args.check_only:
        return
    out = args.out_dir
    (out / "data").mkdir(parents=True, exist_ok=True)
    curves.to_csv(out / "data/validation_curves.csv", index=False)
    episodes.to_csv(out / "data/heldout_episodes.csv", index=False)
    summary.to_csv(out / "data/heldout_summary.csv", index=False)
    configure_style()
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
                         "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
    colors = dict(zip(seeds, COLORS))
    plot_training(curves, colors, resumed_seed, out)
    plot_services(curves, colors, resumed_seed, out)
    plot_checkpoints(summary, colors, resumed_seed, out)
    plot_decomposition(summary, decisions, out)
    episode = plot_episode(args.episode_json, args.hotspot_json, inputs, out) if args.episode_json else None
    write_report(out, summary, parent, decisions, colors, episode)
    import shlex
    from datetime import datetime, timezone
    helper = Path(__file__).with_name("generate_section5_single_panel_figures_20260714.py")
    provenance = dict(generated_at=datetime.now(timezone.utc).isoformat(),
                      command=shlex.join([sys.executable, *sys.argv]), input_sha256=inputs.hashes,
                      continuation=args.continuation.name, continuation_parent=parent["path"].name,
                      independent_training_seeds=seeds, metric_decisions=decisions,
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
                      episode=episode, formats=["png", "pdf"])
    (out / "data/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {out / 'README.md'}")


if __name__ == "__main__":
    main()
