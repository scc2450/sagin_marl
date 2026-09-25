from pathlib import Path
import importlib.util
from dataclasses import asdict
import json
import sys
import tarfile
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from scripts.experiments.more_gus.run_parameter_scans import check_seed_only
from scripts.experiments.more_gus import run_parameter_scans as scans
from sagin_marl.env.config import SaginConfig, update_config


def plot_module():
    path = Path(__file__).resolve().parents[1] / "docs/paper_moreGUs/reproduction"
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


def recovery_fixture(tmp_path):
    old, new = tmp_path / "failed", tmp_path / "recovered"
    config = asdict(update_config(SaginConfig(), dict(structured_env_tensor_backend="cuda")))
    manifests = []
    for root in (old, new):
        (root / "source/scripts/experiments/more_gus").mkdir(parents=True)
        (root / "source/evaluator.py").write_text("unchanged")
        (root / "source/scripts/experiments/more_gus/run_parameter_scans.py").write_text(root.name)
        with tarfile.open(root / "source.tar", "w") as handle:
            for path in sorted((root / "source").rglob("*.py")):
                handle.add(path, arcname=str(path.relative_to(root / "source")))
        (root / "config.yaml").write_text(yaml.safe_dump(config))
        jobs = [dict(output=f"evaluations/{index}", config="config.yaml", method="static_uniform",
            command=["python", str(root / "source/evaluator.py"), str(root / f"evaluations/{index}")],
            rows_file="episodes.csv", summary_file="summary.json", axis="load", multiplier=1,
            point="load_1", paper_label="Uniform", training_seed=None, checkpoint_kind="fixed",
            seed_base=index * 1000, total_arrival_mbps=40, access_mhz=4, backhaul_mhz=10,
            satellite_cpu_ghz=50) for index in range(2)]
        manifest = dict(source_commit=root.name, source_archive_sha256=scans.sha(root / "source.tar"),
            points=[], checkpoints=[], input_sha256={}, training_seeds=[], episode_seed_bases=[0, 1000],
            episodes=32, num_envs=32, policy_mode="deterministic", primary_methods=["Uniform"],
            primary_jobs=2, jobs=jobs, config_sha256={"config.yaml": scans.sha(root / "config.yaml")},
            gpu=0, plot_python="plot-python")
        scans.write_json(root / "manifest.json", manifest)
        manifests.append(manifest)
    scans.write_json(old / "status.json", dict(status="failed", completed=1))
    for job in manifests[0]["jobs"]:
        dest = old / job["output"]
        dest.mkdir(parents=True)
        write_fake_evaluation(dest, config)
    scans.write_json(old / "evaluations/0/completion.json", dict(status="complete", episodes=32))
    return old, new, manifests[1], config


def write_fake_evaluation(dest, config):
    scans.write_csv(dest / "episodes.csv", [dict(episode=i, arrival_sum=100,
        **{metric: 1 for metric in scans.METRICS}) for i in range(32)])
    scans.write_json(dest / "summary.json", {})
    scans.write_json(dest / "episodes.metadata.json", dict(effective_config=config))
    (dest / "evaluation.log").write_text("finished")


def test_recovery_revalidates_completed_and_reruns_incomplete_without_overwriting(tmp_path, monkeypatch):
    old, new, manifest, config = recovery_fixture(tmp_path)
    recovery = scans.reuse_completed(new, manifest, old)
    assert list(recovery["imported_jobs"]) == ["evaluations/0"]
    assert recovery["excluded_incomplete_outputs"] == ["evaluations/1"]
    assert not (new / "evaluations/1").exists()
    old_marker = (old / "evaluations/0/completion.json").read_bytes()
    manifest["recovery"] = recovery
    scans.write_json(new / "manifest.json", manifest)
    scans.write_json(new / "status.json", dict(status="prepared"))
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[0] == "python":
            write_fake_evaluation(Path(command[-1]), config)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(scans.subprocess, "run", fake_run)
    scans.run(SimpleNamespace(run_dir=new))
    assert [cmd for cmd in commands if cmd[0] == "python"] == [manifest["jobs"][1]["command"]]
    assert json.loads((new / "status.json").read_text())["episodes"] == 64
    assert (new / "evaluations/0/completion.json").read_bytes() == old_marker
    assert json.loads((old / "status.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("changed", ["config", "source", "command", "active", "count"])
def test_recovery_rejects_incompatible_or_active_evidence(tmp_path, changed):
    old, new, manifest, _ = recovery_fixture(tmp_path)
    if changed == "config":
        manifest["config_sha256"]["config.yaml"] = "different"
    elif changed == "source":
        (old / "source/evaluator.py").write_text("changed")
    elif changed == "command":
        manifest["jobs"][0]["command"].append("--different")
    else:
        scans.write_json(old / "status.json", dict(status="running" if changed == "active" else "failed",
            completed=2 if changed == "count" else 1))
    with pytest.raises((ValueError, RuntimeError)):
        scans.reuse_completed(new, manifest, old)


def test_recovery_rejects_tampered_import_before_launch(tmp_path, monkeypatch):
    old, new, manifest, _ = recovery_fixture(tmp_path)
    manifest["recovery"] = scans.reuse_completed(new, manifest, old)
    scans.write_json(new / "manifest.json", manifest)
    scans.write_json(new / "status.json", dict(status="prepared"))
    (new / "evaluations/0/episodes.csv").write_text("tampered")
    monkeypatch.setattr(scans.subprocess, "run", lambda *a, **kw: pytest.fail("Must not launch"))
    with pytest.raises(RuntimeError, match="Recovered evidence changed"):
        scans.run(SimpleNamespace(run_dir=new))
