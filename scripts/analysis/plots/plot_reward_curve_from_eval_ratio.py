from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_eval import evaluate_structured_actor_exec_sources_with_traces
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving


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


def _safe_float(value: object, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_latest_training_curve(metrics_path: Path) -> tuple[np.ndarray, np.ndarray]:
    latest: dict[int, float] = {}
    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step_value = _safe_float(row.get("total_env_steps"), _safe_float(row.get("step")))
            reward = _safe_float(row.get("episode_reward"))
            if not math.isfinite(step_value) or not math.isfinite(reward):
                continue
            latest[int(round(step_value))] = reward
    if not latest:
        raise RuntimeError(f"No episode_reward curve found in {metrics_path}")
    steps = np.asarray(sorted(latest), dtype=np.float64)
    rewards = np.asarray([latest[int(step)] for step in steps], dtype=np.float64)
    return steps, rewards


def _mean_csv_column(path: Path, column: str) -> float:
    values: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = _safe_float(row.get(column))
            if math.isfinite(value):
                values.append(value)
    if not values:
        raise RuntimeError(f"No finite {column!r} values found in {path}")
    return float(np.mean(values))


def _old_logged_reward_reference(run_dir: Path, old_curve: np.ndarray, override: float | None) -> tuple[float, str]:
    if override is not None and math.isfinite(float(override)) and float(override) > 0.0:
        return float(override), "override"
    candidates = sorted(run_dir.glob("eval_trained_best*_n*.csv"))
    if candidates:
        path = candidates[-1]
        return _mean_csv_column(path, "reward_sum"), str(path)
    if old_curve.size == 0:
        raise RuntimeError("Cannot infer old logged reward reference from an empty training curve.")
    return float(old_curve[-1]), "training_curve_last"


def _resolve_device(device_arg: str) -> torch.device:
    requested = str(device_arg).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(requested)


def _evaluate_same_rollout_ratio(
    cfg,
    actor,
    *,
    old_reward_mode: str,
    horizon: int,
    device: torch.device,
    episodes: int,
    episode_seed_base: int | None,
    vec_backend: str,
) -> tuple[dict[str, float], list[dict[str, float]], float, list[dict[str, float | str]]]:
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg.T_steps = int(horizon)
    eval_cfg.reward_mode = str(old_reward_mode)
    summary, rows, traces = evaluate_structured_actor_exec_sources_with_traces(
        eval_cfg,
        actor,
        device=device,
        episodes=int(episodes),
        episode_seed_base=episode_seed_base,
        deterministic=True,
        num_envs=1,
        vec_backend=str(vec_backend),
    )
    pair_rows: list[dict[str, float | str]] = []
    old_sums: list[float] = []
    new_sums: list[float] = []
    for ep_idx, trace in enumerate(traces):
        old_sum = 0.0
        new_sum = 0.0
        for step in trace:
            old_sum += float(step.get("reward", 0.0) or 0.0)
            level = float(step.get("bw_weighted_workload_level_reward", 0.0) or 0.0)
            workload = max(-level, 0.0)
            new_sum += 1.0 / (1.0 + math.log1p(workload))
        old_sums.append(old_sum)
        new_sums.append(new_sum)
        collision = max((float(step.get("collision_event", 0.0) or 0.0) for step in trace), default=0.0)
        pair_rows.append(
            {
                "episode": float(ep_idx),
                "old_reward_sum": float(old_sum),
                "new_reward_sum": float(new_sum),
                "ratio": float(new_sum / old_sum) if abs(old_sum) > 1.0e-12 else math.nan,
                "step_count": float(len(trace)),
                "collision_episode_fraction": float(collision),
                "new_reward_source": "same_rollout_from_bw_weighted_workload_level",
            }
        )
    old_mean = float(np.mean(old_sums)) if old_sums else float(summary["reward_sum"])
    new_mean = float(np.mean(new_sums)) if new_sums else math.nan
    if abs(old_mean) < 1.0e-12:
        raise RuntimeError("Old reward evaluation is too close to zero; cannot compute scale ratio.")
    return summary, rows, float(new_mean / old_mean), pair_rows


def _write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reference-run", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--old-reward-mode", default="controllable_flow")
    parser.add_argument("--new-reward-mode", default="positive_weighted_workload_level")
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-seed-base", type=int, default=42000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--vec-backend", default="sync", choices=["sync", "subproc"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--old-logged-reference", type=float, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "thesis_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config_source.yaml"
    if not config_path.exists():
        config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config_source.yaml or config.yaml found in {run_dir}")
    metrics_path = run_dir / "metrics.csv"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / "actor_best.pt"

    cfg = load_config(str(config_path))
    cfg.T_steps = int(args.horizon)
    device = _resolve_device(args.device)

    bundle = build_structured_modules_from_config(cfg, hidden_dim=args.hidden_dim, embed_dim=args.embed_dim)
    actor = bundle.actor.to(device)
    load_checkpoint_forgiving(actor, str(checkpoint_path), map_location=device, strict=False)
    actor.eval()

    old_summary, old_rows, same_rollout_ratio, paired_rows = _evaluate_same_rollout_ratio(
        cfg,
        actor,
        old_reward_mode=str(args.old_reward_mode),
        horizon=int(args.horizon),
        device=device,
        episodes=int(args.episodes),
        episode_seed_base=int(args.episode_seed_base),
        vec_backend=str(args.vec_backend),
    )
    old_reward = float(old_summary["reward_sum"])
    new_reward = old_reward * float(same_rollout_ratio)

    steps, old_curve = _load_latest_training_curve(metrics_path)
    old_logged_ref, old_logged_ref_source = _old_logged_reward_reference(
        run_dir,
        old_curve,
        args.old_logged_reference,
    )
    curve_scale_ratio = new_reward / old_logged_ref
    scaled_mean = old_curve * curve_scale_ratio

    curve_rows: list[dict[str, float | str]] = []
    for i in range(steps.size):
        curve_rows.append(
            {
                "total_env_steps": float(steps[i]),
                "old_episode_reward": float(old_curve[i]),
                "scale_ratio": float(curve_scale_ratio),
                "reward": float(scaled_mean[i]),
                "old_logged_reference": float(old_logged_ref),
                "old_logged_reference_source": str(old_logged_ref_source),
            }
        )
    _write_csv(out_dir / "reward_curve.csv", curve_rows)

    eval_rows: list[dict[str, float | str]] = [
        {
            "mode": str(args.old_reward_mode),
            "reward_sum_mean": float(old_summary["reward_sum"]),
            "processed_ratio_eval": float(old_summary["processed_ratio_eval"]),
            "drop_ratio_eval": float(old_summary["drop_ratio_eval"]),
            "pre_backlog_steps_eval": float(old_summary["pre_backlog_steps_eval"]),
            "collision_episode_fraction": float(old_summary["collision_episode_fraction"]),
        },
        {
            "mode": str(args.new_reward_mode),
            "reward_sum_mean": float(new_reward),
            "processed_ratio_eval": float(old_summary["processed_ratio_eval"]),
            "drop_ratio_eval": float(old_summary["drop_ratio_eval"]),
            "pre_backlog_steps_eval": float(old_summary["pre_backlog_steps_eval"]),
            "collision_episode_fraction": float(old_summary["collision_episode_fraction"]),
        },
        {
            "mode": "same_rollout_reward_ratio",
            "reward_sum_mean": float(same_rollout_ratio),
            "processed_ratio_eval": math.nan,
            "drop_ratio_eval": math.nan,
            "pre_backlog_steps_eval": math.nan,
            "collision_episode_fraction": math.nan,
        },
        {
            "mode": "old_logged_reference",
            "reward_sum_mean": float(old_logged_ref),
            "processed_ratio_eval": math.nan,
            "drop_ratio_eval": math.nan,
            "pre_backlog_steps_eval": math.nan,
            "collision_episode_fraction": math.nan,
        },
        {
            "mode": "curve_scale_ratio",
            "reward_sum_mean": float(curve_scale_ratio),
            "processed_ratio_eval": math.nan,
            "drop_ratio_eval": math.nan,
            "pre_backlog_steps_eval": math.nan,
            "collision_episode_fraction": math.nan,
        },
    ]
    _write_csv(out_dir / "reward_curve_eval_ratio.csv", eval_rows)

    _write_csv(out_dir / "reward_curve_eval_episode_pairs.csv", paired_rows)

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

    x_million = steps / 1.0e6
    fig, ax = plt.subplots(figsize=(4.9, 3.25), dpi=200)
    ax.plot(x_million, scaled_mean, color="#1F4E79", linewidth=1.9)
    ax.set_xlabel("环境交互步数/百万步")
    ax.set_ylabel("回合奖励")
    ymax = max(float(np.nanmax(scaled_mean)) * 1.08, 1.0)
    ax.set_ylim(0.0, ymax)
    ax.grid(True, color="#D9D9D9", linewidth=0.45, alpha=0.85)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    fig.tight_layout(pad=0.3)

    pdf_path = out_dir / "reward_curve.pdf"
    png_path = out_dir / "reward_curve.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    print(f"Saved {out_dir / 'reward_curve.csv'}")
    print(f"Saved {out_dir / 'reward_curve_eval_ratio.csv'}")
    print(
        f"old250_reward={old_reward:.6f} new250_reward={new_reward:.6f} "
        f"same_rollout_ratio={same_rollout_ratio:.8f} old_logged_ref={old_logged_ref:.6f} "
        f"curve_scale_ratio={curve_scale_ratio:.8f} font={font_name or 'matplotlib default'}"
    )


if __name__ == "__main__":
    main()
