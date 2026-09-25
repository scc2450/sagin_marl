"""Offline guardrails for the registered single-panel 100-GU paper assets."""
import importlib
import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")


@pytest.fixture
def plots(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "docs/paper_moreGUs/reproduction"))
    return importlib.import_module("generate_more_gus_paper_assets")


def test_only_known_unused_default_serialization_is_allowed(plots):
    original = dict(critic_value_mode="relational", seed=45211)
    candidate = dict(original, critic_value_mode="global_only",
                     baseline_dq_movement_weight=.02, baseline_dq_switch_weight=.01)
    diff = plots.validate_critic_configs(original, candidate)
    assert len(diff) == 3
    candidate["baseline_dq_switch_weight"] = 0
    with pytest.raises(ValueError, match="beyond critic"):
        plots.validate_critic_configs(original, candidate)


def test_learning_config_difference_is_rejected(plots):
    with pytest.raises(ValueError, match="beyond critic"):
        plots.validate_critic_configs(
            dict(critic_value_mode="relational", actor_lr=.0003),
            dict(critic_value_mode="global_only", actor_lr=.001))


def test_registered_evidence_and_protocol(plots):
    manifest = json.loads((plots.TABLES / "manifest.json").read_text())
    assert len(manifest["table_sha256"]) == 9
    for name, digest in manifest["table_sha256"].items():
        assert plots.sha(plots.TABLES / name) == digest
    assert manifest["gc_status"] == "pending_review"
    assert not manifest["selected_final_comparison_exported"]
    scan = pd.read_csv(plots.TABLES / "scan_seed_means.csv")
    assert len(scan) == 98
    assert set(scan.paper_label) == {"STARS", "DQS", "Lyapunov", "QBS", "Uniform"}
    assert set(scan.checkpoint_kind) == {"selected", "fixed"}
    assert set(scan.episodes) == {64}
    selected = pd.read_csv(plots.TABLES / "selected_policy_means.csv")
    assert set(selected.method) == {"STARS", "STARS-GC"}
    assert len(selected) == 6 and selected.episodes.eq(64).all()


def test_single_continuation_and_no_imputed_gc_tail(plots):
    curves = pd.read_csv(plots.TABLES / "training_curves.csv")
    tail = curves[curves.phase == "continuation"]
    assert set(tail.training_seed) == {45211}
    assert tail["update"].tolist() == list(range(525, 701, 25))
    assert len(curves) == 84
    critics = pd.read_csv(plots.TABLES / "critic_curves.csv")
    assert critics[critics.method == "STARS-GC"].groupby("training_seed")["update"].max().eq(375).all()
    assert critics[(critics.method == "STARS") & (critics.training_seed == 45211)]["update"].max() == 500


def test_every_panel_is_single_axis_and_gc_flagged(plots, monkeypatch, tmp_path):
    plots.configure_style()
    renderer = plots.Renderer(figures=tmp_path / "pdf", previews=tmp_path / "png")
    records = []

    def capture(fig, name, caption, tables, **metadata):
        assert len(fig.axes) == 1
        if "critic_ablation" in name:
            assert metadata["status"] == "pending_review"
            assert "PENDING REVIEW" in [t.get_text() for t in fig.texts]
        records.append((name, metadata))
        plots.plt.close(fig)

    monkeypatch.setattr(renderer, "save", capture)
    renderer.scans()
    renderer.training()
    renderer.selected_queues()
    renderer.episode(json.loads((plots.TABLES / "manifest.json").read_text()))
    renderer.critic()
    assert len(records) == 28
    assert not any("final" in name or "endpoint" in name for name, _ in records)
    scans = [metadata for name, metadata in records if name.startswith(("load_", "resource_"))]
    assert len(scans) == 12
    assert any(m["off_scale"] for m in scans)
    assert all(point["method"] in {"QBS", "Uniform"} for m in scans for point in m["off_scale"])


def test_registry_has_exact_table_coverage(plots):
    registry = pd.read_csv(plots.TABLES.parent / "registry_index.csv")
    registered = registry[registry.file.str.startswith("more_gus_20260925/")]
    assert len(registered) == 9 and not registered.file.duplicated().any()
    for row in registered.itertuples():
        assert len(pd.read_csv(plots.TABLES.parent / row.file)) == row.row_count
