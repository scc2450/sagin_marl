import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "docs/paper_moreGUs/reproduction/generate_dqs_revision_figures.py"
    spec = importlib.util.spec_from_file_location("dqs_revision_figures", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def fixture():
    script = module()
    row = {metric: "1" for metric in script.METRICS}
    row.update(axis="load", multiplier="1", seed_base="1980000", episode="0", episode_length="250",
               method="distributed_queue_c", paper_label="DQS", checkpoint_kind="fixed")
    return script, row


def test_revision_preserves_controls_and_changes_only_dqs_metrics():
    script, row = fixture()
    control = dict(row, method="stars", paper_label="STARS", checkpoint_kind="final")
    update = dict(row, reward_sum="50", processed_ratio_eval=".99")
    result = script.replace_dqs([row, control], [update])
    assert result[1] == control
    assert result[0]["reward_sum"] == "50"
    assert result[0]["paper_label"] == "DQS"
    assert row["reward_sum"] == "1"


def test_revision_rejects_unpaired_or_incomplete_replacements():
    script, row = fixture()
    for updates in ([], [row, row], [dict(row, arrival_step_mean="2")], [dict(row, episode_length="249")]):
        with pytest.raises(ValueError):
            script.replace_dqs([row], updates)
