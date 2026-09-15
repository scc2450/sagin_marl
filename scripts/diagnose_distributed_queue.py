"""Diagnostic-only global state capture; never passed to policy scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from sagin_marl.env.config import load_config
import sagin_marl.rl.distributed_queue as dq
import sagin_marl.rl.structured_eval as ev

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    original_action = dq.distributed_queue_action
    for bw in (2, 4):
        for variant in ("b", "c"):
            records = []
            cfg = load_config("configs/experiments/more_gus/structured_joint_mcgae_3uav100gu_22clusters_t250.yaml")
            cfg.b_acc = bw * 1e6
            cfg.access_bw_decision_interval = 5
            cfg.sat_decision_interval = 1
            def capture_action(obs, cfg, variant="c", settings=dq.DQSettings()):
                action, diag = original_action(obs, cfg, variant, settings)
                if len(records) < 12:
                    ba, bd = original_action(obs, cfg, "b", settings)
                    record = {"step": len(records) + 1,
                        "ego": obs.ego_features[6:9].cpu().tolist(),
                        "peer_tokens": obs.peer_tokens[6:9].cpu().tolist(),
                        "peer_mask": obs.peer_mask[6:9].cpu().tolist(),
                        "action": action[6:9].cpu().tolist(), "b_action_same_obs": ba[6:9].cpu().tolist(),
                        "b_selected": bd["selected"][6:9].cpu().tolist(),
                        "diag": {k: v[6:9].cpu().tolist() for k,v in diag.items()}}
                    records.append(record)
                return action, diag
            dq.distributed_queue_action = capture_action
            try:
                summary, rows = ev.evaluate_structured_actor_exec_sources(
                    cfg, torch.nn.Linear(1,1).cuda(), device=torch.device("cuda"),
                    episodes=8, num_envs=8, episode_seed_base=971000, deterministic=True,
                    exec_accel_source="distributed_queue_" + variant,
                    exec_sat_source="queue_aware", exec_bw_source="queue_aware")
                result = dict(bandwidth_mhz=bw, variant=variant, seed=971002,
                    commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                    source_sha256=hashlib.sha256(Path(dq.__file__).read_bytes()).hexdigest(),
                    summary=summary, target_episode=rows[2],
                    records=records[:min(12, int(rows[2]["episode_length"]))])
                (root / f"bw{bw}_{variant}.json").write_text(json.dumps(result, indent=2, default=str))
                print(bw, variant, summary["episode_length"], summary["processed_ratio_eval"], flush=True)
            finally:
                dq.distributed_queue_action = original_action

if __name__ == "__main__":
    main()
