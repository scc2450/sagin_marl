from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv


def _fmt(x: float) -> str:
    return f"{x:,.3g}"


def _estimate_once(env: SaginParallelEnv, cfg, seed: int) -> tuple[float, float, float, int, int, int]:
    env.reset(seed=seed)
    effective_arrival_rate = float(getattr(env, "effective_task_arrival_rate", cfg.task_arrival_rate))
    assoc = env._associate_users()
    candidates = env._build_candidate_users(assoc)

    dummy_actions = {
        agent: {
            "accel": np.zeros(2, dtype=np.float32),
            "bw_logits": np.zeros(cfg.users_obs_max, dtype=np.float32),
            "sat_logits": np.zeros(cfg.sats_obs_max, dtype=np.float32),
        }
        for agent in env.agents
    }

    access_rates, _ = env._compute_access_rates(assoc, candidates, dummy_actions)
    sat_pos, sat_vel = env.orbit.get_states(env.t * cfg.tau0)
    visible = env._visible_sats_sorted(sat_pos)
    sat_selection = env._select_satellites(sat_pos, sat_vel, dummy_actions, visible)
    rate_matrix, _ = env._compute_backhaul_rates(sat_pos, sat_vel, sat_selection)

    access_cap = float(np.sum(access_rates)) * cfg.tau0
    backhaul_cap = float(np.sum(rate_matrix)) * cfg.tau0
    assoc_count = int(np.sum(assoc >= 0))
    active_sats = {l for sels in sat_selection for l in sels}
    active_sat_count = len(active_sats)
    link_count = int(sum(len(sels) for sels in sat_selection))
    arrival_effective = cfg.num_gu * effective_arrival_rate * cfg.tau0
    return arrival_effective, access_cap, backhaul_cap, assoc_count, active_sat_count, link_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase1.yaml")
    parser.add_argument("--samples", type=int, default=3, help="Number of random resets to average.")
    parser.add_argument(
        "--sat_mode",
        type=str,
        default="config",
        choices=["config", "fixed", "topk", "sample"],
        help="Override satellite selection mode for the estimate.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.sat_mode != "config":
        if args.sat_mode == "fixed":
            cfg.fixed_satellite_strategy = True
        else:
            cfg.fixed_satellite_strategy = False
            cfg.sat_select_mode = args.sat_mode
    env = SaginParallelEnv(cfg)

    arrival_effective_vals = []
    access_caps = []
    backhaul_caps = []
    assoc_counts = []
    active_sats = []
    link_counts = []
    for i in range(max(1, args.samples)):
        arrival_effective, access_cap, backhaul_cap, assoc_count, active_sat_count, link_count = _estimate_once(
            env, cfg, cfg.seed + i
        )
        arrival_effective_vals.append(arrival_effective)
        access_caps.append(access_cap)
        backhaul_caps.append(backhaul_cap)
        assoc_counts.append(assoc_count)
        active_sats.append(active_sat_count)
        link_counts.append(link_count)

    arrival_raw = cfg.num_gu * cfg.task_arrival_rate * cfg.tau0
    arrival_effective = float(np.mean(arrival_effective_vals))
    access_cap = float(np.mean(access_caps))
    backhaul_cap = float(np.mean(backhaul_caps))
    assoc_count = float(np.mean(assoc_counts))
    active_sat = float(np.mean(active_sats))
    link_count = float(np.mean(link_counts))

    compute_cap = cfg.num_sat * (cfg.sat_cpu_freq / cfg.task_cycles_per_bit) * cfg.tau0
    compute_cap_eff = active_sat * (cfg.sat_cpu_freq / cfg.task_cycles_per_bit) * cfg.tau0
    bottleneck = min(access_cap, backhaul_cap, compute_cap_eff)
    util_raw = float("inf") if bottleneck <= 0 else arrival_raw / bottleneck
    util_effective = float("inf") if bottleneck <= 0 else arrival_effective / bottleneck

    print("Throughput sanity check (bits/slot)")
    print(f"- Arrival raw:  {_fmt(arrival_raw)} (num_gu * task_arrival_rate * tau0)")
    print(f"- Arrival eff:  {_fmt(arrival_effective)} (effective_task_arrival_rate from env reset)")
    print(f"- Access cap:   {_fmt(access_cap)} (avg over {len(access_caps)} samples)")
    print(f"- Backhaul cap: {_fmt(backhaul_cap)} (avg over {len(backhaul_caps)} samples)")
    print(f"- Compute cap:  {_fmt(compute_cap)} (theoretical, all sats)")
    print(f"- Compute cap*: {_fmt(compute_cap_eff)} (effective, active sats)")
    print(f"- Active sats:  {_fmt(active_sat)} / {cfg.num_sat}")
    print(f"- Links:        {_fmt(link_count)} (avg UAV-sat links)")
    mode = "fixed" if cfg.fixed_satellite_strategy else cfg.sat_select_mode
    print(f"- Sat mode:     {mode} (N_RF={cfg.N_RF})")
    print(f"- Assoc GU:     {_fmt(assoc_count)} / {cfg.num_gu}")
    print(f"- Bottleneck:   {_fmt(bottleneck)}")
    if bottleneck <= 0:
        print("! Bottleneck is zero. Check pathloss threshold or visibility settings.")
        return

    print(f"- Util raw:     {_fmt(util_raw)} (arrival_raw / bottleneck)")
    print(f"- Util eff:     {_fmt(util_effective)} (arrival_effective / bottleneck)")
    if util_effective < 0.4:
        print("Assessment: underloaded (queues likely near zero).")
    elif util_effective > 1.5:
        print("Assessment: overloaded (queues likely grow).")
    else:
        print("Assessment: balanced (queues should show dynamics).")


if __name__ == "__main__":
    main()
