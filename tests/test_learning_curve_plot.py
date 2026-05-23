from __future__ import annotations

import csv

import numpy as np

from sagin_marl.rl.mappo import _rolling_window_summary
from sagin_marl.viz.plots import _resolve_reward_band, plot_learning_curve


def test_rolling_window_summary_reports_mean_std_and_iqr():
    values = [1.0, 2.0, 3.0, 4.0]
    stats = _rolling_window_summary(values)

    assert abs(stats["mean"] - float(np.mean(values))) < 1e-9
    assert abs(stats["std"] - float(np.std(values))) < 1e-9
    assert abs(stats["p25"] - float(np.percentile(values, 25.0))) < 1e-9
    assert abs(stats["p75"] - float(np.percentile(values, 75.0))) < 1e-9


def test_resolve_reward_band_prefers_iqr_over_std():
    lower, upper, label = _resolve_reward_band(mean=10.0, std=3.0, p25=8.0, p75=11.0)

    assert lower == 8.0
    assert upper == 11.0
    assert label == "Reward IQR (p25-p75)"


def test_plot_learning_curve_writes_band_plot(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    out_path = tmp_path / "learning_curve.png"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "episode_reward", "episode_reward_std", "episode_reward_p25", "episode_reward_p75"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "step": 1,
                "episode_reward": 10.0,
                "episode_reward_std": 2.0,
                "episode_reward_p25": 9.0,
                "episode_reward_p75": 11.0,
            }
        )
        writer.writerow(
            {
                "step": 2,
                "episode_reward": 12.0,
                "episode_reward_std": 1.5,
                "episode_reward_p25": 11.0,
                "episode_reward_p75": 13.0,
            }
        )

    plot_learning_curve(str(csv_path), str(out_path))

    assert out_path.exists()
    assert out_path.stat().st_size > 0
