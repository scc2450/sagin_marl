"""Queue critic-only 100-GU ablations from a hash-verified frozen STARS protocol."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def ablation_config(reference, seed):
    result = dict(reference, seed=seed, critic_value_mode="global_only")
    if reference["critic_value_mode"] != "relational":
        raise ValueError("Reference must be the relational STARS critic")
    differences = {key for key in result if result[key] != reference[key]}
    if not differences <= {"seed", "critic_value_mode"}:
        raise ValueError(f"Non-critic changes: {differences}")
    return result


def prepare(args):
    from sagin_marl.env.config import load_config
    reference = Path(args.reference_run).resolve()
    reference_manifest = json.loads((reference / "manifest.json").read_text())
    if sha(reference / "config.yaml") != reference_manifest["config_sha256"]["config.yaml"]:
        raise ValueError("Reference config hash mismatch")
    if json.loads((reference / "preflight.json").read_text())["status"] != "passed":
        raise ValueError("Reference protocol did not pass acceptance")
    cfg = asdict(load_config(str(reference / "config.yaml")))
    from scripts.experiments.more_gus.run_fixed_baselines import check_protocol
    check_protocol(cfg, 32, 32, [1980000, 1981000])
    if reference_manifest["max_updates"] != 700:
        raise ValueError("Expected the existing 700-update hard cap")
    seeds = args.seeds
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Training seeds must be distinct nonnegative integers")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "sagin_marl", "scripts", "configs"], cwd=REPO, text=True)
    if dirty.strip():
        raise RuntimeError("Commit executable source first; unrelated paper edits may remain")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = root / "source"
    source.mkdir()
    with (root / "source.tar").open("wb") as handle:
        subprocess.run(["git", "archive", commit, "sagin_marl", "scripts", "configs"],
                       cwd=REPO, stdout=handle, check=True)
    with tarfile.open(root / "source.tar") as handle:
        handle.extractall(source, filter="data")
    checked = {}
    for name in (
        "scripts/train_joint_mcgae.py", "sagin_marl/rl/structured_mappo.py",
        "sagin_marl/rl/structured_critic.py", "sagin_marl/rl/structured_eval.py",
        "sagin_marl/env/native_cuda/kernels.cu",
    ):
        if sha(source / name) != sha(reference / "source" / name):
            raise RuntimeError(f"Training/evaluation source drift: {name}")
        checked[name] = sha(source / name)
    (root / "reference_config.yaml").write_bytes((reference / "config.yaml").read_bytes())
    configs = {}
    for seed in seeds:
        child = root / f"seed{seed}"
        child.mkdir()
        effective = ablation_config(cfg, seed)
        small = dict(effective, checkpoint_eval_interval_updates=1,
            checkpoint_eval_start_update=1, checkpoint_eval_episodes=8,
            checkpoint_eval_episode_seed_base=1910000, checkpoint_eval_early_stop_enabled=False)
        scale = dict(effective, checkpoint_eval_interval_updates=3,
            checkpoint_eval_start_update=3, checkpoint_eval_episode_seed_base=1920000,
            checkpoint_eval_early_stop_enabled=False)
        hashes = {}
        for name, payload in (("config.yaml", effective), ("small_config.yaml", small), ("scale_config.yaml", scale)):
            (child / name).write_text(yaml.safe_dump(payload, sort_keys=False))
            hashes[name] = sha(child / name)
        configs[str(seed)] = hashes
        write_json(child / "manifest.json", dict(seed=seed, gpu=args.gpu,
            source_commit=commit, reference_run=str(reference), max_updates=700,
            config_sha256=hashes, source_parent="../source",
            changed_fields={key: [cfg[key], effective[key]] for key in cfg if cfg[key] != effective[key]},
            return_target="bootstrap_gae", postrun_seed_bases=[1980000, 1981000]))
        write_json(child / "status.json", dict(status="queued", seed=seed))
    write_json(root / "manifest.json", dict(source_commit=commit, gpu=args.gpu, seeds=seeds,
        reference_run=str(reference), reference_config_sha256=sha(reference / "config.yaml"),
        reference_preflight_sha256=sha(reference / "preflight.json"),
        source_archive_sha256=sha(root / "source.tar"), configs=configs,
        training_source_checks=checked, max_updates=700, num_envs=64, rollout_env_steps=250,
        critic_change="relational -> global_only; actor/reward/safety/optimizer/stopping unchanged",
        created_at=time.time()))
    write_json(root / "status.json", dict(status="prepared", queued_seeds=seeds, completed_seeds=[]))


def run(args):
    root = Path(args.run_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if json.loads((root / "status.json").read_text())["status"] != "prepared":
        raise RuntimeError("Queue is not freshly prepared; do not implicitly resume")
    source = root / "source"
    if sha(root / "source.tar") != manifest["source_archive_sha256"]:
        raise RuntimeError("Source archive changed")
    for key, value in dict(CUDA_VISIBLE_DEVICES=str(manifest["gpu"]), OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1",
        PYTHONFAULTHANDLER="1", PYTHONPATH="").items():
        os.environ[key] = value
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("frozen_training_controller",
        source / "scripts/experiments/more_gus/run_bootstrap_training.py")
    jobs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jobs)
    completed = []
    from types import SimpleNamespace
    try:
        for seed in manifest["seeds"]:
            child = root / f"seed{seed}"
            for name, digest in manifest["configs"][str(seed)].items():
                if sha(child / name) != digest:
                    raise RuntimeError(f"Frozen config changed: {seed}/{name}")
            write_json(root / "status.json", dict(status="running", phase="preflight",
                current_seed=seed, completed_seeds=completed, pid=os.getpid(), time=time.time()))
            settings = SimpleNamespace(seed=seed, gpu=manifest["gpu"], max_updates=700)
            small = child / "preflight" / "resume_smoke"
            jobs.run_command(child, source, "smoke_2u", jobs.train_command(
                child, "small_config.yaml", small, settings, envs=8, updates=2, save_every=1))
            jobs.run_command(child, source, "smoke_resume_u3", jobs.train_command(
                child, "small_config.yaml", small, settings, envs=8, updates=3, save_every=1,
                resume=small / "checkpoint_update0002.pt"))
            resume_report = jobs.check_training(small, 3)
            scale = child / "preflight" / "scale64_1u"
            jobs.run_command(child, source, "scale64_1u", jobs.train_command(
                child, "scale_config.yaml", scale, settings, envs=64, updates=1, save_every=1))
            scale_report = jobs.check_training(scale, 1)
            evaluation = jobs.evaluate(child, source, "smoke_eval", "config.yaml",
                scale / "final.pt", seed=1920000, episodes=32, envs=32)
            write_json(child / "preflight.json", dict(status="passed",
                scope="New GC resume and full-scale CUDA smoke; unchanged environment protocol reused from reference",
                resume=resume_report, scale=scale_report, evaluation=evaluation,
                caveat="A cold-start EV gate may skip actor updates; this smoke is not evidence of learned performance",
                reference_run=manifest["reference_run"], finished_at=time.time()))
            write_json(root / "status.json", dict(status="running", phase="train",
                current_seed=seed, completed_seeds=completed, pid=os.getpid(), time=time.time()))
            jobs.train(child, source, settings)
            write_json(child / "status.json", dict(status="complete", phase="train", time=time.time()))
            completed.append(seed)
        write_json(root / "status.json", dict(status="complete", completed_seeds=completed, time=time.time()))
    except BaseException as exc:
        failure = dict(status="failed", current_seed=seed, completed_seeds=completed,
                       error=repr(exc), time=time.time())
        write_json(root / "status.json", failure)
        write_json(child / "failure.json", failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "run"), required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--reference_run")
    parser.add_argument("--seeds", type=int, nargs="+", default=[45211, 45210, 61723])
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    if args.phase == "prepare":
        if not args.reference_run or args.gpu < 0:
            parser.error("Preparation needs --reference_run and a nonnegative GPU")
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
