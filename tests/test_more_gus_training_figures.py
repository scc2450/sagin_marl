"""Offline checks for plotting protocol and the historical range-limit fix."""
import importlib
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")


@pytest.fixture
def plots(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "docs/paper_moreGUs/reproduction"))
    return importlib.import_module("generate_more_gus_training_figures")


def test_best_so_far_keeps_actual_endpoints(plots):
    data = pd.DataFrame({"training_seed": [1, 1, 2, 2],
                         "update": [25, 50, 25, 700],
                         "reward_sum": [100, 90, 80, 210]})
    result = plots._best_so_far_df(data)
    assert len(result) == len(data)
    assert result[result.training_seed == 1]["update"].tolist() == [25, 50]
    assert result[result.training_seed == 1]["reward_sum"].tolist() == [100, 100]
    assert result[result.training_seed == 2]["update"].tolist() == [25, 700]


def test_legacy_plot_no_longer_clips_new_data(plots, monkeypatch):
    legacy = importlib.import_module("generate_section5_single_panel_figures_20260714")
    captured = {}

    def capture(fig, *_args, **_kwargs):
        captured["x"] = fig.axes[0].get_xlim()
        captured["y"] = fig.axes[0].get_ylim()

    monkeypatch.setattr(legacy, "save_all", capture)
    legacy.plot_checkpoint_panel(pd.DataFrame({
        "training_seed": [1, 1], "update": [25, 700], "reward_sum": [20, 210],
    }), "unused", qccs_reward=250)
    assert captured["x"][0] < 25 and captured["x"][1] > 700
    assert captured["y"][0] < 20 and captured["y"][1] > 250


def test_informative_queue_but_saturated_flow(plots):
    data = pd.DataFrame({"queue_total_mbit": [5.5, 73.7],
                         "outflow_arrival_ratio": [.993, .999],
                         "sat_incoming_arrival_ratio": [.991, .999],
                         "sat_processed_arrival_ratio": [.991, .999]})
    decisions = plots.mechanism_decisions(data)
    assert decisions["queue_panel"]
    assert not decisions["flow_panel"]
    data["queue_total_mbit"] = [5.5, 5.6]
    assert not plots.mechanism_decisions(data)["queue_panel"]


def continuation_fixture():
    parent = dict(path=Path("parent"), manifest={"seed": 1},
                  config={"seed": 1, "checkpoint_eval_reward_early_stop_enabled": True},
                  curve=pd.DataFrame({"update": [475, 500]}))
    child = dict(manifest={"seed": 1, "start_update": 500, "reference_run": "/remote/parent"},
                 config={"seed": 1, "checkpoint_eval_reward_early_stop_enabled": False},
                 curve=pd.DataFrame({"update": [525, 550]}))
    return parent, child


def test_single_continuation_preserves_boundary(plots):
    parent, child = continuation_fixture()
    result = plots.attach_continuation(parent, child)
    assert result["update"].tolist() == [475, 500, 525, 550]
    assert result["phase"].tolist() == ["original", "original", "continuation", "continuation"]


@pytest.mark.parametrize("failure", ["changed_seed", "overlap", "config"])
def test_invalid_continuation_rejected(plots, failure):
    parent, child = continuation_fixture()
    if failure == "changed_seed":
        child["manifest"]["seed"] = 2
    elif failure == "overlap":
        child["curve"]["update"] = [500, 525]
    else:
        child["config"]["actor_lr"] = 0.1
    with pytest.raises(ValueError):
        plots.attach_continuation(parent, child)
