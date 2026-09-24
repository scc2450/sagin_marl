"""Evaluate frozen 100-GU baselines through the dedicated CLI, or prepare a scan."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import time

import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sagin_marl.env.config import load_config

METHODS = (
    "distributed_queue_c",
    "maxweight_lyapunov", "queue_aware_bw", "static_uniform",
)
PAPER_LABELS = {
    "distributed_queue_c": "DQS",
    "maxweight_lyapunov": "MaxWeight/Lyapunov",
    "queue_aware_bw": "QBS",
    "static_uniform": "Uniform",
}
LOAD_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
RESOURCE_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
METRICS = (
    "reward_sum", "processed_ratio_eval", "drop_ratio_eval",
    "pre_backlog_steps_eval", "D_sys_report", "collision_episode_fraction",
    "queue_total_mean", "gu_queue_mean", "uav_queue_mean", "sat_queue_mean",
    "arrival_step_mean", "outflow_arrival_ratio", "sat_overlap_eval", "episode_length",
)
RESOURCE_FIELDS = ("b_acc", "b_backhaul_per_sat", "sat_cpu_freq")


def write_json(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def point_config(base, axis, multiplier):
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Scan multipliers must be finite and positive")
    if axis not in {"nominal", "load", "resource"}:
        raise ValueError(f"Unknown scan axis: {axis}")
    if axis == "nominal" and multiplier != 1:
        raise ValueError("Nominal evaluation requires multiplier=1")
    if base.get("resource_scale_enabled", False):
        raise ValueError("Disable implicit resource scaling in the training protocol first")
    result = dict(base)
    fields = ("task_arrival_rate",) if axis == "load" else RESOURCE_FIELDS if axis == "resource" else ()
    for field in fields:
        result[field] = base[field] * multiplier
    return result


def check_protocol(config, episodes, num_envs, seeds):
    for field, expected in {
        "num_uav": 3, "num_gu": 100, "gu_init_num_clusters": 22,
        "T_steps": 250, "access_bw_decision_interval": 1, "sat_decision_interval": 1,
        "safety_shield_enabled": True, "safety_shield_solver": "NATIVE_CUDA",
        "avoidance_enabled": False, "fixed_satellite_strategy": False,
    }.items():
        if config.get(field) != expected:
            raise ValueError(f"Training protocol mismatch: {field}={config.get(field)!r}; expected {expected!r}")
    if num_envs != 32 or episodes < 32 or episodes % 32:
        raise ValueError("Use num_envs=32 and a positive multiple of 32 episodes")
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("Provide distinct nonnegative episode seed bases")
    ordered = sorted(seeds)
    if any(b < a + episodes for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Episode seed ranges overlap")


def command_for(source, config, dest, method, seed, episodes, num_envs):
    command = [
        sys.executable, str(source / "scripts/evaluation/evaluate_structured_fixed_policy.py"),
        "--config", str(config), "--baseline", method,
        "--episodes", str(episodes), "--num_envs", str(num_envs),
        "--episode_seed_base", str(seed), "--structured_env_tensor_backend", "cuda",
        "--out", str(dest / "episodes.csv"), "--summary_out", str(dest / "summary.json"),
    ]
    if method == "distributed_queue_c":
        command += ["--dq_movement_weight", "0", "--dq_switch_weight", "0"]
    return command


def aggregate(rows):
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in ("axis", "multiplier", "method"))
        groups.setdefault(key, []).append(row)
    output = []
    for (axis, multiplier, method), group in groups.items():
        result = dict(axis=axis, multiplier=multiplier, method=method,
                      paper_label=PAPER_LABELS.get(method, method), episodes=len(group))
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            result[metric + "_mean"] = statistics.mean(values)
            result[metric + "_episode_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(result)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--axis", choices=("nominal", "load", "resource"), default="nominal")
    parser.add_argument("--multipliers", nargs="+", type=float)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seed_bases", nargs="+", type=int, default=[1980000, 1981000])
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout_seconds", type=int, default=600)
    parser.add_argument("--plan_only", action="store_true")
    args = parser.parse_args()
    grid = args.multipliers or (
        [1.0] if args.axis == "nominal" else LOAD_GRID if args.axis == "load" else RESOURCE_GRID)
    training_config = Path(args.training_config).resolve()
    training_manifest_path = training_config.parent / "manifest.json"
    training_manifest = json.loads(training_manifest_path.read_text())
    training_sha = hashlib.sha256(training_config.read_bytes()).hexdigest()
    if training_manifest.get("config_sha256", {}).get(training_config.name) != training_sha:
        raise RuntimeError("Training configuration does not match its frozen manifest")
    cfg = asdict(load_config(str(training_config)))
    check_protocol(cfg, args.episodes, args.num_envs, args.seed_bases)
    if len(set(grid)) != len(grid) or len(set(args.methods)) != len(args.methods):
        parser.error("Duplicate multipliers or methods are not allowed")
    if args.gpu < 0 or args.timeout_seconds < 1:
        parser.error("GPU index and timeout are invalid")
    points = [point_config(cfg, args.axis, multiplier) for multiplier in grid]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "sagin_marl", "scripts", "configs"],
        cwd=REPO, text=True)
    if dirty.strip():
        raise RuntimeError("Commit executable source/config changes before freezing; documentation edits may remain")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    archive = root / "source.tar"
    with archive.open("wb") as handle:
        subprocess.run(["git", "archive", commit, "sagin_marl", "scripts", "configs"],
                       cwd=REPO, stdout=handle, check=True)
    source = root / "source"
    source.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(source, filter="data")
    (root / "training_config.yaml").write_bytes(training_config.read_bytes())
    jobs, point_records = [], []
    for multiplier, config in zip(grid, points):
        point = f"{args.axis}_{multiplier:g}"
        config_path = root / f"{point}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        point_records.append(dict(name=point, multiplier=multiplier, config=config_path.name,
            config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            changed_fields={key: value for key, value in config.items() if value != cfg[key]}))
        for method in args.methods:
            for seed in args.seed_bases:
                dest = root / point / method / f"seed{seed}"
                jobs.append(dict(point=point, axis=args.axis, multiplier=multiplier, method=method,
                    seed_base=seed, output=str(dest.relative_to(root)),
                    command=command_for(source, config_path, dest, method, seed, args.episodes, args.num_envs)))
    manifest = dict(
        source_commit=commit, source_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        training_config_origin=str(training_config),
        training_config_sha256=training_sha,
        training_source_commit=training_manifest["source_commit"],
        training_manifest_sha256=hashlib.sha256(training_manifest_path.read_bytes()).hexdigest(),
        created_at=time.time(), python=sys.version, gpu=args.gpu,
        methods=args.methods, paper_labels=PAPER_LABELS, points=point_records, jobs=jobs,
        episodes_per_seed_base=args.episodes, num_envs=args.num_envs, episode_seed_bases=args.seed_bases,
        c_weights=dict(movement_weight=0, switch_weight=0),
        evidence_role="Paired post-training screening; reused seeds, not an untouched final test",
        resource_axis="Joint scaling of access bandwidth, per-satellite backhaul bandwidth, and satellite CPU",
        learned_baselines="Not included: new-scene checkpoint identities must be supplied separately",
    )
    write_json(root / "manifest.json", manifest)
    write_json(root / "status.json", dict(status="planned", jobs=len(jobs), completed=0))
    if args.plan_only:
        print(f"Planned {len(jobs)} evaluations in {root}", flush=True)
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    combined = []
    completed = 0
    try:
        for job in jobs:
            dest = root / job["output"]
            dest.mkdir(parents=True)
            started = time.time()
            write_json(root / "status.json", dict(status="running", jobs=len(jobs),
                completed=completed, current=job["output"], pid=os.getpid()))
            print(f"START {job['output']}", flush=True)
            with (dest / "evaluation.log").open("w") as handle:
                result = subprocess.run(job["command"], cwd=source, env=env, stdout=handle,
                    stderr=subprocess.STDOUT, timeout=args.timeout_seconds)
            if result.returncode:
                raise RuntimeError(f"{job['output']} exited {result.returncode}; inspect evaluation.log")
            with (dest / "episodes.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != args.episodes:
                raise RuntimeError("Incomplete episode output")
            for row in rows:
                if any(not math.isfinite(float(row[metric])) for metric in METRICS):
                    raise RuntimeError("Non-finite evaluation metric")
                combined.append(dict(axis=args.axis, multiplier=job["multiplier"], method=job["method"],
                    seed_base=job["seed_base"], **row))
            metadata = json.loads((dest / "episodes.metadata.json").read_text())
            expected = points[grid.index(job["multiplier"])]
            if job["method"] == "distributed_queue_c":
                expected = dict(expected, baseline_dq_movement_weight=0.0, baseline_dq_switch_weight=0.0)
            # The CLI explicitly selects CUDA; all other effective settings must match.
            expected = dict(expected, structured_env_tensor_backend="cuda")
            if metadata["effective_config"] != json.loads(json.dumps(expected)):
                raise RuntimeError("Evaluator effective configuration drift")
            write_json(dest / "completion.json", dict(elapsed_seconds=time.time() - started,
                finished_at=time.time(), episodes=len(rows), effective_config_verified=True))
            completed += 1
            print(f"END {job['output']} seconds={time.time() - started:.1f}", flush=True)
        write_csv(root / "episodes.csv", combined)
        write_csv(root / "aggregate.csv", aggregate(combined))
        write_json(root / "status.json", dict(status="completed", jobs=len(jobs),
            completed=completed, episodes=len(combined), finished_at=time.time()))
    except BaseException as exc:
        write_json(root / "status.json", dict(status="failed", jobs=len(jobs),
            completed=completed, error=str(exc), failed_at=time.time()))
        raise


if __name__ == "__main__":
    main()
