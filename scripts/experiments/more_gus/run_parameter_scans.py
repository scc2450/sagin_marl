"""Run frozen STARS and fixed-policy scans; omit GC until its new training is complete."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.experiments.more_gus.run_fixed_baselines import (
    METHODS, PAPER_LABELS, LOAD_GRID, RESOURCE_GRID, METRICS,
    check_protocol, command_for, point_config, write_json, write_csv,
)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def reference_config(path):
    from sagin_marl.env.config import load_config
    manifest = json.loads((path / "manifest.json").read_text())
    if sha(path / "config.yaml") != manifest["config_sha256"]["config.yaml"]:
        raise ValueError(f"Changed frozen config: {path}")
    if json.loads((path / "status.json").read_text())["status"] != "complete" or (path / "failure.json").exists():
        raise ValueError(f"Incomplete training: {path}")
    if "start_update" in manifest or "continuation" in str(path.name):
        raise ValueError("Use independent original runs, not a continuation")
    cfg = asdict(load_config(str(path / "config.yaml")))
    check_protocol(cfg, 32, 32, [1980000, 1981000])
    if cfg["critic_value_mode"] != "relational":
        raise ValueError("Only STARS belongs in this learned-method roster")
    return cfg, manifest


def check_seed_only(base, other):
    changed = {key for key in base.keys() | other.keys() if base.get(key) != other.get(key)}
    if not changed <= {"seed"}:
        raise ValueError(f"Training protocols differ beyond seed: {changed}")


def interpreter_path(value):
    # Resolving a venv's python symlink would select the system interpreter.
    return str(Path(value).expanduser().absolute())


def prepare(args):
    import torch
    plot_python = interpreter_path(args.plot_python)
    plot_runtime = json.loads(subprocess.check_output([plot_python, "-c",
        "import json,importlib.metadata as m; print(json.dumps({p:m.version(p) for p in ('numpy','pandas','matplotlib','pyyaml')}))"], text=True))
    runs = [Path(p).resolve() for p in args.training_runs]
    loaded = [reference_config(p) for p in runs]
    seeds = [cfg["seed"] for cfg, _ in loaded]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Supply exactly three independent STARS training runs")
    base = loaded[0][0]
    for cfg, _ in loaded[1:]:
        check_seed_only(base, cfg)
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--",
        "sagin_marl", "scripts", "configs", "docs/paper/reproduction/generate_more_gus_scan_figures.py"],
        cwd=REPO, text=True)
    if dirty.strip():
        raise RuntimeError("Commit scan source first; unrelated paper edits may remain")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    archive = root / "source.tar"
    with archive.open("wb") as handle:
        subprocess.run(["git", "archive", commit, "sagin_marl", "scripts", "configs",
            "docs/paper/reproduction/generate_more_gus_scan_figures.py",
            "docs/paper/reproduction/generate_section5_single_panel_figures_20260714.py"],
            cwd=REPO, stdout=handle, check=True)
    source = root / "source"
    source.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(source, filter="data")
    checkpoints, provenance = [], {}
    for path, (cfg, manifest) in zip(runs, loaded):
        seed = cfg["seed"]
        for name in ("manifest.json", "config.yaml", "train/training_stop.json"):
            provenance[str((path / name).relative_to(REPO))] = sha(path / name)
        for kind, filename in (("selected", "best_checkpoint.pt"), ("final", "final.pt")):
            original = path / "train" / filename
            dest = root / "checkpoints" / f"seed{seed}" / f"{kind}.pt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, dest)
            digest = sha(dest)
            if digest != sha(original):
                raise RuntimeError("Checkpoint copy mismatch")
            state = torch.load(dest, map_location="cpu", weights_only=False)
            update = int(state["update"])
            del state
            checkpoints.append(dict(training_seed=seed, kind=kind, update=update,
                path=str(dest.relative_to(root)), sha256=digest,
                training_run=str(path.relative_to(REPO)), original=str(original.relative_to(REPO))))
            provenance[str(original.relative_to(REPO))] = digest
    jobs, points, configs = [], [], {}
    for axis, grid in (("load", LOAD_GRID), ("resource", RESOURCE_GRID)):
        for multiplier in grid:
            point = f"{axis}_{multiplier:g}"
            effective = point_config(base, axis, multiplier)
            point_info = dict(axis=axis, multiplier=multiplier, point=point,
                total_arrival_mbps=effective["task_arrival_rate"] * effective["num_gu"] / 1e6,
                access_mhz=effective["b_acc"] / 1e6,
                backhaul_mhz=effective["b_backhaul_per_sat"] / 1e6,
                satellite_cpu_ghz=effective["sat_cpu_freq"] / 1e9)
            points.append(point_info)
            for seed in seeds:
                path = root / "configs" / point / f"seed{seed}.yaml"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(yaml.safe_dump(dict(effective, seed=seed), sort_keys=False))
                configs[str(path.relative_to(root))] = sha(path)
            for method in METHODS:
                for eval_seed in (1980000, 1981000):
                    dest = root / "evaluations" / point / method / f"seedbase{eval_seed}"
                    cfg_path = root / "configs" / point / f"seed{seeds[0]}.yaml"
                    jobs.append(dict(**point_info, method=method, paper_label=PAPER_LABELS[method],
                        training_seed=None, checkpoint_kind="fixed", seed_base=eval_seed,
                        config=str(cfg_path.relative_to(root)), output=str(dest.relative_to(root)),
                        command=command_for(source, cfg_path, dest, method, eval_seed, 32, 32),
                        rows_file="episodes.csv", summary_file="summary.json"))
            for checkpoint in checkpoints:
                seed, kind = checkpoint["training_seed"], checkpoint["kind"]
                for eval_seed in (1980000, 1981000):
                    dest = root / "evaluations" / point / "stars" / f"seed{seed}_{kind}" / f"seedbase{eval_seed}"
                    cfg_path = root / "configs" / point / f"seed{seed}.yaml"
                    command = [sys.executable, str(source / "scripts/evaluate_structured_mixed_heads_native.py"),
                        "--config", str(cfg_path), "--base_checkpoint", str(root / checkpoint["path"]),
                        "--device", "cuda", "--episodes", "32", "--num_envs", "32",
                        "--episode_seed_base", str(eval_seed), "--policy_mode", "deterministic",
                        "--out_dir", str(dest), "--label", "eval"]
                    jobs.append(dict(**point_info, method="stars", paper_label="STARS",
                        training_seed=seed, checkpoint_kind=kind, seed_base=eval_seed,
                        checkpoint_update=checkpoint["update"], checkpoint=checkpoint["path"],
                        config=str(cfg_path.relative_to(root)), output=str(dest.relative_to(root)),
                        command=command, rows_file="eval_episodes.csv", summary_file="eval_summary.json"))
    jobs.sort(key=lambda job: job["checkpoint_kind"] == "final")
    manifest = dict(source_commit=commit, source_archive_sha256=sha(archive), gpu=args.gpu,
        plot_python=plot_python, plot_runtime=plot_runtime,
        created_at=time.time(), points=points, checkpoints=checkpoints, jobs=jobs, config_sha256=configs,
        input_sha256=provenance, training_seeds=seeds, episode_seed_bases=[1980000, 1981000],
        episodes=32, num_envs=32, policy_mode="deterministic",
        primary_methods=["STARS", "DQS", "MaxWeight/Lyapunov", "QBS", "Uniform"],
        excluded_methods=["STARS-GC", "QCCS"],
        checkpoint_rule="Original training-selected checkpoints fixed across every scan point; original finals diagnostic only",
        resource_axis="Joint access/backhaul/CPU multiplier, not an access-only experiment",
        evidence_role="Reused screening seeds; final locked-test evidence must use fresh seeds",
        primary_jobs=sum(job["checkpoint_kind"] != "final" for job in jobs))
    write_json(root / "manifest.json", manifest)
    write_json(root / "status.json", dict(status="prepared", completed=0, jobs=len(jobs),
        primary_jobs=manifest["primary_jobs"], primary_figures_ready=False))
    print(f"Prepared {len(jobs)} jobs, {manifest['primary_jobs']} primary, in {root}", flush=True)


def run(args):
    root = Path(args.run_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if json.loads((root / "status.json").read_text())["status"] != "prepared":
        raise RuntimeError("Only a freshly prepared run may start; no implicit overwrite/resume")
    if sha(root / "source.tar") != manifest["source_archive_sha256"]:
        raise RuntimeError("Source archive changed")
    for path, digest in manifest["config_sha256"].items():
        if sha(root / path) != digest:
            raise RuntimeError(f"Frozen config changed: {path}")
    for checkpoint in manifest["checkpoints"]:
        if sha(root / checkpoint["path"]) != checkpoint["sha256"]:
            raise RuntimeError("Frozen checkpoint changed")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(manifest["gpu"]),
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1")
    env.pop("PYTHONPATH", None)
    combined, arrivals = [], {}
    completed, primary_ready = 0, False
    source = root / "source"
    try:
        for job in manifest["jobs"]:
            dest = root / job["output"]
            dest.mkdir(parents=True, exist_ok=False)
            started = time.time()
            write_json(root / "status.json", dict(status="running", completed=completed,
                jobs=len(manifest["jobs"]), primary_jobs=manifest["primary_jobs"],
                current=job["output"], primary_figures_ready=primary_ready, pid=os.getpid()))
            print(f"START {job['output']}", flush=True)
            with (dest / "evaluation.log").open("w") as handle:
                result = subprocess.run(job["command"], cwd=source, env=env,
                    stdout=handle, stderr=subprocess.STDOUT, timeout=900)
            if result.returncode:
                raise RuntimeError(f"{job['output']} exited {result.returncode}")
            with (dest / job["rows_file"]).open() as handle:
                rows = list(csv.DictReader(handle))
            summary = json.loads((dest / job["summary_file"]).read_text())
            if job["method"] == "stars":
                if any(summary["load_info"].get(key) for key in
                       ("missing_keys", "unexpected_keys", "adapted_keys", "skipped_keys")):
                    raise RuntimeError("Checkpoint load was not exact")
                if summary["num_envs"] != 32 or summary["policy_mode"] != "deterministic":
                    raise RuntimeError("Learned evaluation protocol drift")
            else:
                meta = json.loads((dest / "episodes.metadata.json").read_text())
                expected = yaml.safe_load((root / job["config"]).read_text())
                expected["structured_env_tensor_backend"] = "cuda"
                if job["method"] == "distributed_queue_c":
                    expected.update(baseline_dq_movement_weight=0.0, baseline_dq_switch_weight=0.0)
                if meta["effective_config"] != expected:
                    raise RuntimeError("Fixed-policy config drift")
            if len(rows) != 32 or len({row["episode"] for row in rows}) != 32:
                raise RuntimeError("Incomplete/duplicate episode coverage")
            for row in rows:
                if any(not math.isfinite(float(row[key])) for key in METRICS):
                    raise RuntimeError("Non-finite metric")
                key = (job["point"], job["seed_base"], row["episode"])
                arrival = float(row["arrival_sum"])
                if key in arrivals and not math.isclose(arrivals[key], arrival, rel_tol=1e-6, abs_tol=1):
                    raise RuntimeError("Paired exogenous arrival mismatch")
                arrivals[key] = arrival
                fields = {key: job[key] for key in
                    ("axis", "multiplier", "point", "method", "paper_label",
                     "training_seed", "checkpoint_kind", "seed_base",
                     "total_arrival_mbps", "access_mhz", "backhaul_mhz", "satellite_cpu_ghz")}
                combined.append(dict(**fields, **row))
            write_json(dest / "completion.json", dict(status="complete", episodes=32,
                elapsed_seconds=time.time() - started, finished_at=time.time()))
            completed += 1
            # Persist partial evidence, but publish primary figures only after the full primary matrix.
            write_csv(root / "episodes.csv", combined)
            if completed in (manifest["primary_jobs"], len(manifest["jobs"])):
                command = [manifest["plot_python"], str(source / "docs/paper/reproduction/generate_more_gus_scan_figures.py"),
                           "--run_dir", str(root)]
                subprocess.run(command, env=env, check=True)
                primary_ready = True
            print(f"END {job['output']} seconds={time.time() - started:.1f}", flush=True)
        write_json(root / "status.json", dict(status="completed", completed=completed,
            jobs=len(manifest["jobs"]), primary_figures_ready=True, episodes=len(combined), finished_at=time.time()))
    except BaseException as exc:
        write_json(root / "status.json", dict(status="failed", completed=completed,
            jobs=len(manifest["jobs"]), primary_figures_ready=primary_ready, error=repr(exc), failed_at=time.time()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare", "run"))
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--training_runs", nargs="+")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--plot_python", default=sys.executable)
    args = parser.parse_args()
    if args.phase == "prepare":
        if not args.training_runs or args.gpu < 0:
            parser.error("Preparation needs --training_runs and a nonnegative GPU")
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
