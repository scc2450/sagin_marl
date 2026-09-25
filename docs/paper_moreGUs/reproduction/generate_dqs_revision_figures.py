"""Replace only DQS rows in a new 100-GU scan, preserving original evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_more_gus_scan_figures import PANELS

METRICS = tuple(metric for metric, _, _ in PANELS) + (
    "collision_episode_fraction", "arrival_step_mean", "episode_length",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def identity(row):
    return (row["axis"], float(row["multiplier"]), int(row["seed_base"]), int(float(row["episode"])))


def replace_dqs(reference, updates):
    import math
    old = {identity(row): row for row in reference if row["method"] == "distributed_queue_c"}
    new = {identity(row): row for row in updates}
    if len(new) != len(updates) or set(old) != set(new):
        raise ValueError("Replacement DQS episodes are incomplete or duplicated")
    for key, row in new.items():
        if row["method"] != "distributed_queue_c" or float(row["episode_length"]) != 250:
            raise ValueError("Replacement method or episode horizon changed")
        if any(not math.isfinite(float(row[field])) for field in METRICS):
            raise ValueError("Non-finite replacement metric")
        if float(row["arrival_step_mean"]) != float(old[key]["arrival_step_mean"]):
            raise ValueError("Paired exogenous arrivals changed")
    protected = {"axis", "multiplier", "method", "seed_base", "episode"}
    output = []
    for row in reference:
        updated = dict(row)
        if row["method"] == "distributed_queue_c":
            updated.update({k: v for k, v in new[identity(row)].items() if k in row and k not in protected})
        output.append(updated)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--load", required=True)
    parser.add_argument("--resource", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    reference, load, resource = map(lambda s: Path(s).resolve(), (args.reference, args.load, args.resource))
    manifests = {str(p): read_json(p / "manifest.json") for p in (reference, load, resource)}
    for root in (reference, load, resource):
        if read_json(root / "status.json")["status"] != "completed":
            raise ValueError(f"Incomplete input campaign: {root}")
    original = manifests[str(reference)]
    revisions = set()
    verified_configs = 0
    for root in (load, resource):
        manifest = manifests[str(root)]
        if manifest["num_envs"] != 32 or manifest["episodes_per_seed_base"] != 32:
            raise ValueError("Changed native evaluation batching")
        for job in manifest["jobs"]:
            point = job["point"]
            old_config = yaml.safe_load((reference / "configs" / point / "seed45211.yaml").read_text())
            new_config = yaml.safe_load((root / (point + ".yaml")).read_text())
            if old_config != new_config:
                raise ValueError(f"Reference config changed: {point}")
            metadata = read_json(root / job["output"] / "episodes.metadata.json")
            expected = dict(old_config, baseline_dq_movement_weight=0., baseline_dq_switch_weight=0.,
                            structured_env_tensor_backend="cuda")
            if metadata["effective_config"] != expected or not metadata["deterministic"]:
                raise ValueError("Effective config or deterministic evaluation drift")
            if metadata["exec_sources"] != ["distributed_queue_c", "lyapunov", "lyapunov"]:
                raise ValueError("Unexpected DQS execution components")
            if metadata["episode_seed_base"] != job["seed_base"] or metadata["num_envs"] != 32:
                raise ValueError("Seed or batching metadata drift")
            revisions.add(metadata["dqs_revision"])
            verified_configs += 1
    if revisions != {"service-forecast-pressure-v1"}:
        raise ValueError(f"Mixed or unexpected DQS revisions: {revisions}")
    old_rows = read_csv(reference / "episodes.csv")
    updates = read_csv(load / "episodes.csv") + read_csv(resource / "episodes.csv")
    rows = replace_dqs(old_rows, updates)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    with (out / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {key: original[key] for key in (
        "points", "checkpoints", "training_seeds", "episode_seed_bases", "resource_axis",
        "primary_methods", "excluded_methods", "checkpoint_rule")}
    manifest.update(source_commit=manifests[str(load)]["source_commit"], dqs_revision=next(iter(revisions)),
        evidence_role="Reused screening seeds; only DQS was rerun. Other policies retain all original selected/final evidence.",
        reference_source_commit=original["source_commit"],
        execution_sources={str(root): dict(source_commit=manifests[str(root)]["source_commit"],
            manifest_sha256=sha(root / "manifest.json"), episodes_sha256=sha(root / "episodes.csv"))
            for root in (reference, load, resource)},
        newly_evaluated_rows=len(updates), unchanged_rows=len(rows)-len(updates),
        generator_sha256=sha(Path(__file__)),
        plotting_sources={str(path.relative_to(Path(__file__).resolve().parents[3])): sha(path) for path in (
            Path(__file__).with_name("generate_more_gus_scan_figures.py"),
            Path(__file__).resolve().parents[2] / "paper/reproduction/generate_section5_single_panel_figures_20260714.py")})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verification = dict(status="passed", verified_effective_configs=verified_configs,
        matched_arrivals=len(updates), replacements=len(updates), rows=len(rows),
        unchanged_controls=all(before == after for before, after in zip(old_rows, rows)
                               if before["method"] != "distributed_queue_c"),
        observed_dqs_collision_episodes=sum(float(r["collision_episode_fraction"]) > 0 for r in updates))
    (out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    import generate_more_gus_scan_figures as plot
    plot.PANELS = [(m, "Queue-based delay proxy" if m == "D_sys_report" else label, scale)
                   for m, label, scale in plot.PANELS]
    sys.argv = [str(Path(plot.__file__)), "--run_dir", str(out)]
    plot.main()
    print(json.dumps(verification))


if __name__ == "__main__":
    main()
