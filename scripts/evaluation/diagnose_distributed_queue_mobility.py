"""Paired interval screening via the dedicated baseline CLI; policy unchanged."""
from __future__ import annotations
import argparse
import csv
from dataclasses import replace
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import sagin_marl.rl.distributed_queue as dq

def longest_true(values):
    return max((sum(1 for _ in group) for value, group in itertools.groupby(values) if value), default=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--seed_bases", nargs="+", type=int, default=[974000, 975000])
    parser.add_argument("--intervals", nargs="+", type=int, default=[1, 5])
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = ROOT / "configs/experiments/more_gus/structured_joint_mcgae_3uav100gu_22clusters_t250.yaml"
    (root / "config.yaml").write_bytes(config.read_bytes())
    manifest = dict(commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        purpose="K alignment and passive mobility diagnosis; no score/weight changes",
        protocol="40/4; T250; 8 environments/8 episodes per batch; SAT K1",
        hover_threshold_mps=0.1, commands=[])
    spec = importlib.util.spec_from_file_location("baseline_cli", ROOT / "scripts/evaluation/evaluate_structured_fixed_policy.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    reports = []
    original = dq.distributed_queue_action
    for interval in args.intervals:
        for seed in args.seed_bases:
            dest = root / f"k{interval}_seed{seed}"
            dest.mkdir()
            samples = []
            def capture(obs, cfg, variant="c", settings=dq.DQSettings()):
                action, diag = original(obs, cfg, variant, settings)
                # Diagnostic only: no alternate action reaches the environment.
                alternative, alternative_diag = original(obs, cfg, variant, replace(
                    settings, movement_weight=0.0, switch_weight=0.0))
                selected_score = alternative_diag["candidate_scores"].gather(1, diag["selected"][:, None]).squeeze(1)
                margin = alternative_diag["score"] - selected_score
                samples.append(dict(
                    pos=(obs.ego_features[:, :2] * cfg.map_size).cpu().numpy().copy(),
                    speed=(obs.ego_features[:, 2:4] * cfg.v_max).norm(dim=-1).cpu().numpy().copy(),
                    alternative_margin=margin.cpu().numpy().copy(),
                    idle_action=(action.norm(dim=-1)<1e-6).cpu().numpy().copy(),
                    alternative_moves=(alternative.norm(dim=-1)>1e-6).cpu().numpy().copy()))
                return action, diag
            command = ["evaluate_structured_fixed_policy.py","--config",str(root/"config.yaml"),
                "--baseline","distributed_queue_c","--episodes","8","--num_envs","8",
                "--episode_seed_base",str(seed),"--structured_env_tensor_backend","cuda",
                "--access_bw_decision_interval",str(interval),"--sat_decision_interval","1",
                "--out",str(dest/"episodes.csv"),"--summary_out",str(dest/"summary.json")]
            manifest["commands"].append(command)
            (root/"manifest.json").write_text(json.dumps(manifest,indent=2))
            previous_argv = sys.argv
            dq.distributed_queue_action = capture
            try:
                sys.argv = command
                cli.main()
            finally:
                dq.distributed_queue_action = original
                sys.argv = previous_argv
            rows = list(csv.DictReader((dest/"episodes.csv").open()))
            summary = json.loads((dest/"summary.json").read_text())
            speeds = np.stack([s["speed"] for s in samples]).reshape(-1,8,3)
            positions = np.stack([s["pos"] for s in samples]).reshape(-1,8,3,2)
            idle = np.stack([s["idle_action"] for s in samples]).reshape(-1,8,3)
            alternate = np.stack([s["alternative_moves"] for s in samples]).reshape(-1,8,3)
            margins = np.stack([s["alternative_margin"] for s in samples]).reshape(-1,8,3)
            agents=[]
            for episode,row in enumerate(rows):
                n=int(float(row["episode_length"]))
                for agent in range(3):
                    v=speeds[:n,episode,agent]
                    hovering=v<0.1
                    stationary_choice=hovering & idle[:n,episode,agent]
                    agents.append(dict(episode_seed=seed+episode,uav=agent,steps=n,
                        mean_speed_mps=float(v.mean()),hover_fraction=float(hovering.mean()),
                        entire_episode_stationary=bool((v<1e-3).all()),
                        longest_hover_steps=longest_true(hovering),
                        observed_path_m=float(np.linalg.norm(np.diff(positions[:n,episode,agent],axis=0),axis=-1).sum()),
                        stationary_decisions=int(stationary_choice.sum()),
                        positive_margin_stationary=int((stationary_choice & (margins[:n,episode,agent]>1e-6)).sum()),
                        stationary_margin_sum=float(margins[:n,episode,agent][stationary_choice].sum()),
                        stationary_margin_max=float(margins[:n,episode,agent][stationary_choice].max()) if stationary_choice.any() else 0.0,
                        zero_regularizer_would_accelerate=int((stationary_choice & alternate[:n,episode,agent]).sum())))
            report=dict(interval=interval,seed_base=seed,summary=summary,agents=agents,
                measurement="Pre-action observations; path spans t0 to t(length-1), omitting final transition",
                counterfactual="Remove movement and switch penalties together on same observation; no alternate rollout")
            (dest/"mobility.json").write_text(json.dumps(report,indent=2))
            reports.append(report)
            (root/"reports.json").write_text(json.dumps(reports,indent=2))
            print("DONE",interval,seed,flush=True)

if __name__ == "__main__":
    main()
