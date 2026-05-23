from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_export_tb_scalars_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "export_tb_scalars.py"
    spec = importlib.util.spec_from_file_location("export_tb_scalars", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_uncertainty_companion_tags_include_reward_band_fields():
    module = _load_export_tb_scalars_module()

    assert module._uncertainty_companion_tags("episode_reward") == [
        "episode_reward_p25",
        "episode_reward_p75",
        "episode_reward_std",
    ]
    assert module._uncertainty_companion_tags("episode_reward_p25") == []


def test_resolve_uncertainty_band_prefers_iqr_and_aligns_steps():
    module = _load_export_tb_scalars_module()

    series_map = {
        "episode_reward": (
            np.asarray([1, 2, 3], dtype=np.int64),
            np.asarray([10.0, 11.0, 12.0], dtype=np.float64),
        ),
        "episode_reward_p25": (
            np.asarray([2, 3], dtype=np.int64),
            np.asarray([9.0, 10.0], dtype=np.float64),
        ),
        "episode_reward_p75": (
            np.asarray([2, 3], dtype=np.int64),
            np.asarray([12.0, 13.0], dtype=np.float64),
        ),
        "episode_reward_std": (
            np.asarray([1, 2, 3], dtype=np.int64),
            np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        ),
    }

    steps, lower, upper, label = module._resolve_uncertainty_band("episode_reward", series_map)

    assert np.array_equal(steps, np.asarray([2, 3], dtype=np.int64))
    assert np.allclose(lower, np.asarray([9.0, 10.0], dtype=np.float64))
    assert np.allclose(upper, np.asarray([12.0, 13.0], dtype=np.float64))
    assert label == "IQR (p25-p75)"


def test_resolve_uncertainty_band_falls_back_to_mean_plus_minus_std():
    module = _load_export_tb_scalars_module()

    series_map = {
        "episode_reward": (
            np.asarray([5, 6], dtype=np.int64),
            np.asarray([20.0, 22.0], dtype=np.float64),
        ),
        "episode_reward_std": (
            np.asarray([5, 6], dtype=np.int64),
            np.asarray([2.0, 3.0], dtype=np.float64),
        ),
    }

    steps, lower, upper, label = module._resolve_uncertainty_band("episode_reward", series_map)

    assert np.array_equal(steps, np.asarray([5, 6], dtype=np.int64))
    assert np.allclose(lower, np.asarray([18.0, 19.0], dtype=np.float64))
    assert np.allclose(upper, np.asarray([22.0, 25.0], dtype=np.float64))
    assert label == "Mean +/- std"
