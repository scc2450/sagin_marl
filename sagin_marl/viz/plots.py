from __future__ import annotations

import csv
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_reward_band(
    mean: float,
    std: float | None,
    p25: float | None,
    p75: float | None,
) -> tuple[float | None, float | None, str | None]:
    if p25 is not None and p75 is not None:
        return float(p25), float(p75), "Reward IQR (p25-p75)"
    if std is not None:
        std_val = abs(float(std))
        return float(mean) - std_val, float(mean) + std_val, "Reward Band (mean +/- std)"
    return None, None, None


def plot_learning_curve(csv_path: str, out_path: str) -> None:
    steps: List[int] = []
    rewards: List[float] = []
    lower_band: List[float] = []
    upper_band: List[float] = []
    band_label: str | None = None

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = _safe_float(row.get("step"))
            reward = _safe_float(row.get("episode_reward"))
            if step is None or reward is None:
                continue
            band_lo, band_hi, band_name = _resolve_reward_band(
                mean=reward,
                std=_safe_float(row.get("episode_reward_std")),
                p25=_safe_float(row.get("episode_reward_p25")),
                p75=_safe_float(row.get("episode_reward_p75")),
            )
            steps.append(int(step))
            rewards.append(float(reward))
            lower_band.append(float("nan") if band_lo is None else float(band_lo))
            upper_band.append(float("nan") if band_hi is None else float(band_hi))
            if band_label is None and band_name is not None:
                band_label = band_name

    if not steps:
        raise ValueError(f"No valid learning-curve rows found in {csv_path}")

    steps_np = np.asarray(steps, dtype=np.int64)
    rewards_np = np.asarray(rewards, dtype=np.float64)
    lower_np = np.asarray(lower_band, dtype=np.float64)
    upper_np = np.asarray(upper_band, dtype=np.float64)

    plt.figure(figsize=(7, 4.5))
    plt.plot(steps_np, rewards_np, label="Episode Reward", linewidth=1.8, color="tab:blue")
    valid_band = np.isfinite(lower_np) & np.isfinite(upper_np)
    if band_label is not None and np.any(valid_band):
        plt.fill_between(
            steps_np,
            lower_np,
            upper_np,
            where=valid_band,
            interpolate=True,
            alpha=0.18,
            color="tab:blue",
            label=band_label,
        )
        plt.plot(
            steps_np,
            lower_np,
            linewidth=0.9,
            linestyle="--",
            color="tab:blue",
            alpha=0.5,
            label="_nolegend_",
        )
        plt.plot(
            steps_np,
            upper_np,
            linewidth=0.9,
            linestyle="--",
            color="tab:blue",
            alpha=0.5,
            label="_nolegend_",
        )
    plt.xlabel("Update")
    plt.ylabel("Reward")
    plt.title("Learning Curve")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_trajectories(gu_pos: np.ndarray, uav_traj: List[np.ndarray], out_path: str) -> None:
    plt.figure(figsize=(5, 5))
    plt.scatter(gu_pos[:, 0], gu_pos[:, 1], s=10, c="tab:blue", label="GU")
    for i, traj in enumerate(uav_traj):
        plt.plot(traj[:, 0], traj[:, 1], label=f"UAV {i}")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("UAV Trajectories")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
