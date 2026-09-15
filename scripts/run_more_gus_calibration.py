"""Small, resumable native-GPU capacity calibration; no learned checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed_base", type=int, default=970000)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = repo / "configs/experiments/more_gus/structured_joint_mcgae_3uav100gu_22clusters_t250.yaml"
    digest = hashlib.sha256(base.read_bytes()).hexdigest()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    identity = dict(commit=commit, config_sha256=digest, episodes=args.episodes,
                    seed_base=args.seed_base, cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"))
    manifest = root / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["identity"] != identity:
        raise RuntimeError("Run identity changed; use a new run directory.")
    if not manifest.exists():
        import torch
        manifest.write_text(json.dumps(dict(identity=identity, python=sys.version,
            torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0), started_at=time.time(),
            purpose="Parameter screening, not formal held-out evidence",
            protocol="native CUDA; K_bw=5; K_sat=1; common episode seeds",
        ), indent=2))
    results = []
    for load in (30, 40, 50):
        for bandwidth in (2, 4, 6):
            cfg = yaml.safe_load(base.read_text())
            cfg["task_arrival_rate"] = load * 1e6 / cfg["num_gu"]
            cfg["b_acc"] = bandwidth * 1e6
            config_path = root / f"load{load}_bw{bandwidth}.yaml"
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            for policy in ("queue_aware_bw", "cluster_center_queue_aware", "observable_cluster_queue_aware"):
                label = f"load{load}_bw{bandwidth}_{policy}"
                out = root / label
                out.mkdir(exist_ok=True)
                status_path = out / "status.json"
                summary_path = out / f"{label}_summary.json"
                command = [sys.executable, str(repo / "scripts/evaluate_structured_mixed_heads_native.py"),
                    "--config", str(config_path), "--baseline_policy", policy,
                    "--episodes", str(args.episodes), "--num_envs", str(args.episodes),
                    "--episode_seed_base", str(args.seed_base), "--device", "cuda",
                    "--access_bw_decision_interval", "5", "--sat_decision_interval", "1",
                    "--out_dir", str(out), "--label", label]
                record = dict(label=label, load_mbps=load, bandwidth_mhz=bandwidth,
                              policy=policy, command=command)
                cached = json.loads(status_path.read_text()) if status_path.exists() else {}
                if cached.get("status") == "complete" and summary_path.exists():
                    record = cached
                else:
                    record.update(status="running", started_at=time.time())
                    status_path.write_text(json.dumps(record, indent=2))
                    print(f"START {label}", flush=True)
                    with (out / "eval.log").open("a") as log:
                        try:
                            completed = subprocess.run(command, cwd=repo, stdout=log,
                                stderr=subprocess.STDOUT, timeout=args.timeout)
                            record["returncode"] = completed.returncode
                            record["status"] = "complete" if completed.returncode == 0 and summary_path.exists() else "failed"
                        except subprocess.TimeoutExpired:
                            record["status"] = "timeout"
                    record["elapsed_seconds"] = time.time() - record["started_at"]
                    status_path.write_text(json.dumps(record, indent=2))
                if record["status"] == "complete":
                    record["summary"] = json.loads(summary_path.read_text())["summary"]
                results.append(record)
                (root / "results.json").write_text(json.dumps(results, indent=2))
                print(f"{record['status'].upper()} {label}", flush=True)
                # Avoid launching the entire grid against a broken runtime.
                if len(results) == 1 and record["status"] != "complete":
                    raise RuntimeError(f"Initial native evaluation failed; inspect {out / 'eval.log'}")
    failed = sum(r["status"] != "complete" for r in results)
    (root / "completion.json").write_text(json.dumps(dict(
        completed_at=time.time(), total=len(results), failed=failed), indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
