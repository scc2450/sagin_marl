from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _pick_chinese_font() -> str | None:
    preferred = [
        "SimSun",
        "NSimSun",
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return name
    return None


def _load_tb_scalars(run_dir: Path, tags: list[str]) -> dict[str, dict[int, tuple[float, float]]]:
    data: dict[str, dict[int, tuple[float, float]]] = {tag: {} for tag in tags}
    for event_path in sorted(run_dir.glob("events.out.tfevents.*")):
        acc = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
        acc.Reload()
        available = set(acc.Tags().get("scalars", []))
        for tag in tags:
            if tag not in available:
                continue
            for item in acc.Scalars(tag):
                prev = data[tag].get(int(item.step))
                if prev is None or float(item.wall_time) >= prev[0]:
                    data[tag][int(item.step)] = (float(item.wall_time), float(item.value))
    return data


def _safe_float(value: object, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_metrics_by_env_step(run_dir: Path) -> dict[int, dict[str, float]]:
    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists():
        return {}
    rows: dict[int, dict[str, float]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            env_step = _safe_float(row.get("total_env_steps"))
            if not math.isfinite(env_step):
                continue
            key = int(round(env_step))
            rows[key] = {k: _safe_float(v) for k, v in row.items()}
    return rows


def _rolling_quantile(values: np.ndarray, window: int, q: float) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    half = max(int(window) // 2, 1)
    for i in range(values.size):
        lo = max(0, i - half)
        hi = min(values.size, i + half + 1)
        out[i] = float(np.quantile(values[lo:hi], q))
    return out


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    half = max(int(window) // 2, 1)
    for i in range(values.size):
        lo = max(0, i - half)
        hi = min(values.size, i + half + 1)
        out[i] = float(np.mean(values[lo:hi]))
    return out


def _write_curve_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument(
        "--workload-tag",
        default="D_sys_report",
        choices=["D_sys_report", "pre_backlog_steps_eval"],
        help="Logged workload-like metric used to reconstruct the final positive reward mapping.",
    )
    parser.add_argument(
        "--mode",
        default="episode_positive_250",
        choices=["episode_positive_250", "workload_positive", "logged_scaled"],
        help=(
            "episode_positive_250 rescales logged episode_reward to a positive per-step reward over the target horizon; "
            "workload_positive maps the selected workload tag to reward; "
            "logged_scaled rescales logged episode_reward without clipping."
        ),
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "thesis_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    tags = [
        args.workload_tag,
        "episode_length_mean",
        "rollout_reward_per_step",
        "episode_reward",
        "pre_backlog_steps_eval",
        "D_sys_report",
    ]
    scalars = _load_tb_scalars(run_dir, tags)
    metrics = _load_metrics_by_env_step(run_dir)

    workload_from_tb = scalars.get(args.workload_tag, {})
    if workload_from_tb:
        workload_by_step = {step: value for step, (_wall, value) in workload_from_tb.items()}
    else:
        workload_by_step = {
            step: row[args.workload_tag]
            for step, row in metrics.items()
            if args.workload_tag in row and math.isfinite(row[args.workload_tag])
        }

    episode_length_by_step = {
        step: value
        for step, (_wall, value) in scalars.get("episode_length_mean", {}).items()
        if math.isfinite(value)
    }
    for step, row in metrics.items():
        value = row.get("episode_length_mean", math.nan)
        if math.isfinite(value):
            episode_length_by_step[step] = value

    logged_episode_by_step = {
        step: value
        for step, (_wall, value) in scalars.get("episode_reward", {}).items()
        if math.isfinite(value)
    }
    for step, row in metrics.items():
        value = row.get("episode_reward", math.nan)
        if math.isfinite(value):
            logged_episode_by_step[step] = value

    rollout_reward_per_step_by_step = {
        step: value
        for step, (_wall, value) in scalars.get("rollout_reward_per_step", {}).items()
        if math.isfinite(value)
    }
    for step, row in metrics.items():
        value = row.get("rollout_reward_per_step", math.nan)
        if math.isfinite(value):
            rollout_reward_per_step_by_step[step] = value

    if args.mode in {"episode_positive_250", "logged_scaled"}:
        common_steps = sorted(set(logged_episode_by_step) & set(episode_length_by_step))
    else:
        common_steps = sorted(set(workload_by_step) & set(episode_length_by_step))
    if not common_steps:
        raise RuntimeError(f"No common TensorBoard scalar steps found in {run_dir}")

    x = np.asarray(common_steps, dtype=np.float64)
    workload = np.asarray([max(workload_by_step.get(s, math.nan), 0.0) for s in common_steps], dtype=np.float64)
    episode_len = np.asarray([max(episode_length_by_step[s], 0.0) for s in common_steps], dtype=np.float64)

    if args.mode in {"episode_positive_250", "logged_scaled"}:
        logged_episode = np.asarray([logged_episode_by_step.get(s, math.nan) for s in common_steps], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            reward_per_step = np.divide(
                logged_episode,
                np.maximum(episode_len, 1.0),
                out=np.zeros_like(logged_episode, dtype=np.float64),
                where=np.isfinite(logged_episode),
            )
        if args.mode == "episode_positive_250":
            reward_per_step = np.clip(reward_per_step, 0.0, 1.0)
    else:
        reward_per_step = 1.0 / (1.0 + np.log1p(workload))
    reward = reward_per_step * float(args.horizon)

    window = max(int(args.window), 3)
    mean = _rolling_mean(reward, window)
    if args.mode == "episode_positive_250":
        band_source = np.asarray(
            [
                np.clip(rollout_reward_per_step_by_step.get(s, reward_per_step[i]), 0.0, 1.0)
                * float(args.horizon)
                for i, s in enumerate(common_steps)
            ],
            dtype=np.float64,
        )
    else:
        band_source = reward
    q25 = _rolling_quantile(band_source, window, 0.25)
    q75 = _rolling_quantile(band_source, window, 0.75)

    if args.mode == "episode_positive_250":
        stem = "reward_curve"
    elif args.mode == "workload_positive":
        stem = "reward_curve_workload_positive"
    else:
        stem = "reward_curve_logged_scaled"

    rows: list[dict[str, float]] = []
    old_step = scalars.get("rollout_reward_per_step", {})
    old_episode = scalars.get("episode_reward", {})
    for i, step in enumerate(common_steps):
        rows.append(
            {
                "total_env_steps": float(step),
                "workload": float(workload[i]),
                "episode_length_mean": float(episode_len[i]),
                "horizon": float(args.horizon),
                "reward_per_step": float(reward_per_step[i]),
                "reward": float(reward[i]),
                "rolling_mean": float(mean[i]),
                "rolling_q25": float(q25[i]),
                "rolling_q75": float(q75[i]),
                "band_source_reward": float(band_source[i]),
                "logged_rollout_reward_per_step": float(
                    old_step.get(step, (math.nan, metrics.get(step, {}).get("rollout_reward_per_step", math.nan)))[1]
                ),
                "logged_episode_reward": float(
                    old_episode.get(step, (math.nan, metrics.get(step, {}).get("episode_reward", math.nan)))[1]
                ),
            }
        )
    csv_path = out_dir / f"{stem}.csv"
    _write_curve_csv(csv_path, rows)

    font_name = _pick_chinese_font()
    if font_name:
        mpl.rcParams["font.family"] = font_name
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.size"] = 9
    mpl.rcParams["axes.labelsize"] = 9
    mpl.rcParams["xtick.labelsize"] = 8
    mpl.rcParams["ytick.labelsize"] = 8
    mpl.rcParams["legend.fontsize"] = 8

    x_million = x / 1.0e6
    fig, ax = plt.subplots(figsize=(4.9, 3.25), dpi=200)
    ax.fill_between(
        x_million,
        q25,
        q75,
        color="#4C78A8",
        alpha=0.18,
        linewidth=0.0,
        label="滑动四分位区间",
    )
    ax.plot(x_million, mean, color="#1F4E79", linewidth=1.9, label="滑动平均")
    ax.set_xlabel("环境交互步数/百万步")
    ax.set_ylabel("奖励")
    if args.mode in {"episode_positive_250", "workload_positive"}:
        ax.set_ylim(0.0, float(args.horizon))
    ax.grid(True, color="#D9D9D9", linewidth=0.45, alpha=0.85)
    ax.legend(frameon=False, loc="lower right")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    fig.tight_layout(pad=0.3)

    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    print(f"Saved {csv_path}")
    print(
        f"points={len(common_steps)} mode={args.mode} "
        f"workload_tag={args.workload_tag if args.mode == 'workload_positive' else 'unused'} "
        f"source={'tensorboard' if (args.mode == 'workload_positive' and workload_from_tb) else 'metrics.csv'} "
        f"horizon={int(args.horizon)} window={window} font={font_name or 'matplotlib default'}"
    )


if __name__ == "__main__":
    main()
