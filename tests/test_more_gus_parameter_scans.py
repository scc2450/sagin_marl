from pathlib import Path
import importlib.util
import sys

import pandas as pd
import pytest

from scripts.experiments.more_gus.run_parameter_scans import check_seed_only


def plot_module():
    path = Path(__file__).resolve().parents[1] / "docs/paper/reproduction"
    sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("scan_plot_test", path / "generate_more_gus_scan_figures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    plot = plot_module()
    manifest = dict(points=[dict(point="load_1")], training_seeds=[45211, 45210, 61723])
    rows = []
    for method in plot.ORDER:
        for seed in manifest["training_seeds"] if method == "stars" else [-1]:
            for seed_base in (1980000, 1981000):
                for episode in range(32):
                    row = dict(point="load_1", axis="load", multiplier=1, total_arrival_mbps=40,
                        method=method, paper_label=plot.LABELS[method],
                        checkpoint_kind="selected" if method == "stars" else "fixed",
                        training_seed=seed, seed_base=seed_base, episode=episode, collision_episode_fraction=0)
                    row.update({key: float(seed % 5 + 1) for key, _, _ in plot.PANELS})
                    rows.append(row)
    return plot, pd.DataFrame(rows), manifest


def test_only_seed_may_differ_between_stars_runs():
    check_seed_only(dict(seed=1, critic_value_mode="relational"), dict(seed=2, critic_value_mode="relational"))
    with pytest.raises(ValueError, match="beyond seed"):
        check_seed_only(dict(seed=1, b_acc=4e6), dict(seed=2, b_acc=2e6))


def test_virtualenv_python_path_is_not_dereferenced(tmp_path):
    from scripts.experiments.more_gus.run_parameter_scans import interpreter_path
    target = tmp_path / "system-python"
    target.touch()
    virtual = tmp_path / "venv-python"
    virtual.symlink_to(target)
    assert interpreter_path(virtual) == str(virtual)
    assert interpreter_path(virtual) != str(virtual.resolve())


def test_plot_requires_complete_matrix_and_excludes_old_methods():
    plot, data, manifest = fixture()
    assert len(plot.validate_primary(data, manifest)) == 448
    with pytest.raises(ValueError, match="64episodes"):
        plot.validate_primary(data.iloc[1:], manifest)
    changed = data.copy()
    changed.loc[0, "paper_label"] = "QCCS"
    with pytest.raises(ValueError, match="Unexpected"):
        plot.validate_primary(changed, manifest)
    changed.loc[0, "paper_label"] = "STARS-GC"
    with pytest.raises(ValueError, match="Unexpected"):
        plot.validate_primary(changed, manifest)


def test_spread_is_over_training_seed_means_not_pooled_episodes():
    plot, data, _ = fixture()
    per_seed, mean, spread = plot.summaries(data)
    values = per_seed[per_seed.method == "stars"]["reward_sum"]
    assert len(values) == 3
    assert mean[mean.method == "stars"]["reward_sum"].item() == pytest.approx(values.mean())
    assert spread[spread.method == "stars"]["reward_sum"].item() == pytest.approx(values.std())
    assert pd.isna(spread[spread.method == "static_uniform"]["reward_sum"].item())


def test_single_panel_renders(tmp_path):
    plot, data, _ = fixture()
    _, mean, spread = plot.summaries(data)
    fig, ax = plot.plt.subplots(figsize=(3.45, 2.95))
    plot.draw(ax, "load", "reward_sum", "Episode reward", 1, mean, spread)
    plot.save(fig, tmp_path, "smoke")
    assert (tmp_path / "png/smoke.png").stat().st_size > 1000
    assert (tmp_path / "pdf/smoke.pdf").stat().st_size > 1000
