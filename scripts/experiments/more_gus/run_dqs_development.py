"""Frozen, paired DQS development and locked-candidate validation campaigns."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.experiments.more_gus.run_fixed_baselines import (
    METRICS, aggregate, check_config_roundtrip, check_protocol, point_config, write_csv, write_json,
)
from sagin_marl.env.config import load_config

# These are component experiments, not additional manuscript baseline names.
VARIANTS = {
    "legacy": ("distributed_queue_c", "queue_aware", "queue_aware"),
    "maxweight": ("lyapunov", "lyapunov", "lyapunov"),
    "pressure_motion": ("lyapunov", "queue_aware", "queue_aware"),
    "pressure_resource": ("distributed_queue_c", "lyapunov", "lyapunov"),
    "forecast": ("distributed_queue_c", "queue_aware", "queue_aware"),
    "forecast_pressure": ("distributed_queue_c", "lyapunov", "lyapunov"),
    "production": ("distributed_queue_c", "lyapunov", "lyapunov"),
}
POINTS = {"nominal": ("nominal", 1.), "load2": ("load", 2.), "resource05": ("resource", .5)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(args):
    import sagin_marl.rl.distributed_queue as dq
    import sagin_marl.rl.structured_eval as evaluation
    from scripts.evaluation import evaluate_structured_fixed_policy as dedicated
    sources = VARIANTS[args.variant]
    evaluation._FIXED_POLICY_EXEC_SOURCE_MAP["distributed_queue_c"] = sources
    settings = dq.DQSettings(movement_weight=0, switch_weight=0,
                            horizon_steps=args.horizon if args.variant.startswith("forecast") else 5)
    if args.variant.startswith("forecast"):
        def action(obs, cfg, variant="c", settings=None):
            effective = replace(dq.settings_from_config(cfg), horizon_steps=args.horizon)
            return dq.service_consistent_queue_action(obs, cfg, variant, effective)
        dq.distributed_queue_action = action
    elif args.variant != "production":
        dq.distributed_queue_action = dq.legacy_distributed_queue_action
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest / "candidate.json", dict(
        variant=args.variant, exec_sources=sources, settings=asdict(settings),
        forecast="local effective-SNR approximation" if args.variant.startswith("forecast") else None,
        source_file_sha256=sha(REPO / "sagin_marl/rl/distributed_queue.py"),
        observation_only=True, seed_base=args.seed, protocol="T250, K=1, num_envs=32",
    ))
    sys.argv = [str(Path(dedicated.__file__)), "--config", args.config,
        "--baseline", "distributed_queue_c", "--episodes", "32", "--num_envs", "32",
        "--episode_seed_base", str(args.seed), "--structured_env_tensor_backend", "cuda",
        "--dq_movement_weight", "0", "--dq_switch_weight", "0",
        "--out", str(dest / "episodes.csv"), "--summary_out", str(dest / "summary.json")]
    dedicated.main()
    metadata_path = dest / "episodes.metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["dqs_revision"] = (dq.DQS_REVISION if args.variant in {"production", "forecast_pressure"}
                                else "development:" + args.variant)
    metadata["dq_settings"] = asdict(settings)
    write_json(metadata_path, metadata)


def campaign(args):
    cfg_path = Path(args.config).resolve()
    config = asdict(load_config(str(cfg_path)))
    seeds = args.seeds or ([2190000] if args.role == "development" else [2290000, 2291000])
    permitted = {2190000, 2191000} if args.role == "development" else {2290000, 2291000}
    if not set(seeds) <= permitted:
        raise ValueError("Use the reserved, disjoint development/held-out seed ranges")
    check_protocol(config, 32, 32, seeds)
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "sagin_marl", "scripts", "configs"],
                                    cwd=REPO, text=True)
    if dirty.strip():
        raise RuntimeError("Commit executable changes before freezing a campaign")
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
    configs, jobs = {}, []
    for point in args.points:
        axis, multiplier = POINTS[point]
        path = root / f"{point}.yaml"
        path.write_text(yaml.safe_dump(point_config(config, axis, multiplier), sort_keys=False))
        check_config_roundtrip(path)
        configs[point] = dict(path=path.name, sha256=sha(path), axis=axis, multiplier=multiplier)
        for variant in args.variants:
            for seed in seeds:
                dest = root / point / variant / f"seed{seed}"
                cmd = [sys.executable, str(source / Path(__file__).relative_to(REPO)),
                    "--worker", "--config", str(path), "--variant", variant,
                    "--horizon", str(args.horizon), "--seed", str(seed), "--output", str(dest)]
                jobs.append(dict(point=point, variant=variant, seed=seed, axis=axis, multiplier=multiplier,
                    output=str(dest.relative_to(root)), command=cmd))
    manifest = dict(source_commit=commit, source_archive_sha256=sha(archive),
        reference_config=str(cfg_path), reference_config_sha256=sha(cfg_path),
        role=args.role, seed_bases=seeds, num_envs=32, episodes_per_batch=32,
        configs=configs, variants={v: VARIANTS[v] for v in args.variants}, horizon=args.horizon,
        zero_movement_switch_penalties=True, gpu=args.gpu, created_at=time.time(), jobs=jobs,
        selection_rule="Development only; prioritize processed/drop/backlog with zero observed collisions; report reward too",
        heldout_rule="No selection or tuning on held-out outcomes; failed candidates remain reported",
        attempts_policy="At most one identical-command retry for SIGSEGV; all attempt logs retained")
    write_json(root / "manifest.json", manifest)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    rows, pairing = [], {}
    try:
        for index, job in enumerate(jobs):
            dest = root / job["output"]
            dest.mkdir(parents=True)
            write_json(root / "status.json", dict(status="running", completed=index,
                       total=len(jobs), current=job["output"], pid=os.getpid()))
            started = time.time()
            print(f"START {job['output']}", flush=True)
            attempts = []
            for attempt in range(2):
                with (dest / f"attempt{attempt}.log").open("w") as log:
                    result = subprocess.run(job["command"], cwd=source, env=env,
                        stdout=log, stderr=subprocess.STDOUT, timeout=900)
                attempts.append(dict(attempt=attempt, returncode=result.returncode))
                write_json(dest / "attempts.json", attempts)
                if result.returncode != -11:
                    break
            if result.returncode:
                raise RuntimeError(f"{job['output']} exited {result.returncode}")
            with (dest / "episodes.csv").open() as handle:
                batch = list(csv.DictReader(handle))
            if len(batch) != 32:
                raise RuntimeError("Incomplete episode output")
            for ep, row in enumerate(batch):
                if any(not math.isfinite(float(row[m])) for m in METRICS) or float(row["episode_length"]) != 250:
                    raise RuntimeError("Invalid episode metric or horizon")
                key = (job["point"], job["seed"], ep)
                arrival = float(row["arrival_step_mean"])
                if key in pairing and pairing[key] != arrival:
                    raise RuntimeError("Paired arrival mismatch")
                pairing[key] = arrival
                rows.append(dict(axis=job["axis"], multiplier=job["multiplier"], method=job["variant"],
                                 seed_base=job["seed"], **row))
            write_csv(root / "episodes.csv", rows)
            write_json(root / "aggregate.json", aggregate(rows))
            write_json(dest / "completion.json", dict(seconds=time.time()-started, rows=len(batch)))
            print(f"DONE {job['output']} {time.time()-started:.1f}s", flush=True)
        write_json(root / "status.json", dict(status="completed", completed=len(jobs), total=len(jobs), rows=len(rows)))
    except Exception as exc:
        write_json(root / "status.json", dict(status="failed", completed=index, total=len(jobs), error=str(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--run_dir")
    parser.add_argument("--role", choices=("development", "heldout"), default="development")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--points", nargs="+", choices=POINTS, default=list(POINTS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.horizon < 1 or args.gpu < 0:
        parser.error("Invalid horizon or GPU")
    if args.worker:
        if not args.variant or not args.output or args.seed is None:
            parser.error("Worker requires variant, output and seed")
        worker(args)
    else:
        if not args.run_dir:
            parser.error("Campaign requires run_dir")
        campaign(args)


if __name__ == "__main__":
    main()
