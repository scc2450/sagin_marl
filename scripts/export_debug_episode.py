from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.rl.action_assembler import assemble_actions
from sagin_marl.rl.policy import ActorNet, batch_flatten_obs


def _resolve_paths(run_dir: str | None, checkpoint: str | None, out_csv: str | None, out_md: str | None, episode_index: int) -> tuple[str, str, str]:
    if run_dir:
        checkpoint = checkpoint or os.path.join(run_dir, "actor.pt")
        base_dir = os.path.join(run_dir, "debug_exports")
    else:
        checkpoint = checkpoint or "runs/phase1/actor.pt"
        base_dir = "runs/phase1/debug_exports"
    os.makedirs(base_dir, exist_ok=True)
    stem = f"debug_ep{episode_index:02d}"
    out_csv = out_csv or os.path.join(base_dir, f"{stem}.csv")
    out_md = out_md or os.path.join(base_dir, f"{stem}.md")
    return checkpoint, out_csv, out_md


def _episode_seed(seed_base: int | None, episode_index: int) -> int | None:
    if seed_base is None:
        return None
    return int(seed_base) + int(episode_index)


def _pairwise_stats(pos: np.ndarray, d_alert: float) -> Dict[str, float]:
    num_uav = pos.shape[0]
    min_dist = float("inf")
    min_i = -1
    min_j = -1
    pair_count_alert = 0
    pair_values: Dict[str, float] = {}
    for i in range(num_uav):
        for j in range(i + 1, num_uav):
            dist = float(np.linalg.norm(pos[i] - pos[j]))
            pair_values[f"d_{i}_{j}"] = dist
            if dist < min_dist:
                min_dist = dist
                min_i = i
                min_j = j
            if d_alert > 0.0 and dist < d_alert:
                pair_count_alert += 1
    if not np.isfinite(min_dist):
        min_dist = 0.0
    out: Dict[str, float] = {
        "min_inter_uav_dist": min_dist,
        "min_pair_i": float(min_i),
        "min_pair_j": float(min_j),
        "pairs_below_alert": float(pair_count_alert),
    }
    out.update(pair_values)
    return out


def _termination_reason(cfg, collision_any: bool, steps: int, energy_depleted: bool) -> str:
    if collision_any:
        return "collision"
    if energy_depleted and steps < int(cfg.T_steps):
        return "energy"
    if steps < int(cfg.T_steps):
        return "early_non_collision"
    return "time_limit"


def _policy_actions(actor: ActorNet, obs, device: torch.device):
    obs_batch = batch_flatten_obs(list(obs.values()), actor.cfg)
    obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
    with torch.no_grad():
        policy_out = actor.act(obs_tensor, deterministic=True)
    accel_norm = policy_out.accel.cpu().numpy()
    bw_logits = policy_out.bw_logits.cpu().numpy() if policy_out.bw_logits is not None else None
    sat_logits = policy_out.sat_logits.cpu().numpy() if policy_out.sat_logits is not None else None
    actions = assemble_actions(actor.cfg, list(obs.keys()), accel_norm, bw_logits=bw_logits, sat_logits=sat_logits)
    return actions, accel_norm


def _run_episode(env: SaginParallelEnv, actor: ActorNet, device: torch.device, record: bool) -> tuple[list[dict], dict]:
    cfg = env.cfg
    rows: list[dict] = []
    done = False
    step_idx = 0
    collision_any = False
    energy_depleted = False
    last_queue = 0.0
    min_dist_overall = float("inf")
    min_dist_step = -1
    min_dist_pair = (-1, -1)
    max_correction_norm = 0.0

    obs = {agent: env._get_obs(i) for i, agent in enumerate(env.agents)}

    while not done:
        pos_before = env.uav_pos.copy()
        vel_before = env.uav_vel.copy()
        pair_before = _pairwise_stats(pos_before, float(cfg.avoidance_alert_factor) * float(cfg.d_safe))

        actions, accel_norm = _policy_actions(actor, obs, device)
        accel_cmd = np.clip(accel_norm, -1.0, 1.0) * float(cfg.a_max)

        obs, rewards, terms, truncs, _ = env.step(actions)

        pos_after = env.uav_pos.copy()
        vel_after = env.uav_vel.copy()
        exec_accel = np.asarray(env.last_exec_accel, dtype=np.float32)
        correction = exec_accel - accel_cmd
        correction_norms = np.linalg.norm(correction, axis=1)
        pair_after = _pairwise_stats(pos_after, float(cfg.avoidance_alert_factor) * float(cfg.d_safe))
        min_dist_after = float(pair_after["min_inter_uav_dist"])
        if min_dist_after < min_dist_overall:
            min_dist_overall = min_dist_after
            min_dist_step = step_idx
            min_dist_pair = (int(pair_after["min_pair_i"]), int(pair_after["min_pair_j"]))
        max_correction_norm = max(max_correction_norm, float(np.max(correction_norms)))

        parts = dict(getattr(env, "last_reward_parts", {}) or {})
        collision_now = bool(parts.get("collision_event", 0.0) > 0.5)
        collision_any = collision_any or collision_now
        if cfg.energy_enabled:
            energy_depleted = bool(np.any(env.uav_energy <= 0.0))
        reward_scalar = float(list(rewards.values())[0])
        terminated = bool(list(terms.values())[0])
        truncated = bool(list(truncs.values())[0])
        done = terminated or truncated
        last_queue = float(parts.get("queue_total_active", 0.0))

        if record:
            row: dict[str, object] = {
                "step": step_idx,
                "reward": reward_scalar,
                "done": int(done),
                "terminated": int(terminated),
                "truncated": int(truncated),
                "collision_event": int(collision_now),
                "queue_total_active": last_queue,
                "avoidance_eta_exec": float(parts.get("avoidance_eta_exec", 0.0)),
                "avoidance_eta_eff": float(parts.get("avoidance_eta_eff", 0.0)),
                "min_inter_uav_dist_before": float(pair_before["min_inter_uav_dist"]),
                "min_inter_uav_dist_after": min_dist_after,
                "min_pair_i_after": int(pair_after["min_pair_i"]),
                "min_pair_j_after": int(pair_after["min_pair_j"]),
                "pairs_below_alert_after": int(pair_after["pairs_below_alert"]),
            }
            for key, value in pair_before.items():
                if key.startswith("d_"):
                    row[f"{key}_before"] = float(value)
            for key, value in pair_after.items():
                if key.startswith("d_"):
                    row[f"{key}_after"] = float(value)
            for i in range(cfg.num_uav):
                row[f"uav{i}_pos_x_before"] = float(pos_before[i, 0])
                row[f"uav{i}_pos_y_before"] = float(pos_before[i, 1])
                row[f"uav{i}_vel_x_before"] = float(vel_before[i, 0])
                row[f"uav{i}_vel_y_before"] = float(vel_before[i, 1])
                row[f"uav{i}_policy_ax_norm"] = float(accel_norm[i, 0])
                row[f"uav{i}_policy_ay_norm"] = float(accel_norm[i, 1])
                row[f"uav{i}_policy_ax_cmd"] = float(accel_cmd[i, 0])
                row[f"uav{i}_policy_ay_cmd"] = float(accel_cmd[i, 1])
                row[f"uav{i}_exec_ax"] = float(exec_accel[i, 0])
                row[f"uav{i}_exec_ay"] = float(exec_accel[i, 1])
                row[f"uav{i}_corr_ax"] = float(correction[i, 0])
                row[f"uav{i}_corr_ay"] = float(correction[i, 1])
                row[f"uav{i}_corr_norm"] = float(correction_norms[i])
                row[f"uav{i}_corr_active"] = int(correction_norms[i] > 1e-6)
                row[f"uav{i}_pos_x_after"] = float(pos_after[i, 0])
                row[f"uav{i}_pos_y_after"] = float(pos_after[i, 1])
                row[f"uav{i}_vel_x_after"] = float(vel_after[i, 0])
                row[f"uav{i}_vel_y_after"] = float(vel_after[i, 1])
            for key, value in parts.items():
                if isinstance(value, (int, float, np.integer, np.floating, bool)):
                    row[f"reward_{key}"] = float(value)
            rows.append(row)

        step_idx += 1

    summary = {
        "steps": step_idx,
        "collision": collision_any,
        "energy_depleted": energy_depleted,
        "termination_reason": _termination_reason(cfg, collision_any, step_idx, energy_depleted),
        "final_queue_total_active": last_queue,
        "min_inter_uav_dist": min_dist_overall,
        "min_dist_step": min_dist_step,
        "min_dist_pair": min_dist_pair,
        "max_correction_norm": max_correction_norm,
    }
    return rows, summary


def _write_csv(rows: List[dict], out_csv: str) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with Path(out_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _find_row(rows: List[dict], step_idx: int) -> dict | None:
    for row in rows:
        if int(row["step"]) == int(step_idx):
            return row
    return None


def _correction_trend(rows: List[dict], end_step: int) -> str:
    if end_step <= 0:
        return "insufficient history"
    start = max(0, end_step - 50)
    window = [row for row in rows if start <= int(row["step"]) < end_step]
    if len(window) < 8:
        return "insufficient history"
    max_corr = np.asarray(
        [max(float(v) for k, v in row.items() if k.endswith("_corr_norm")) for row in window],
        dtype=np.float64,
    )
    split = max(1, len(max_corr) // 2)
    first = float(np.mean(max_corr[:split]))
    second = float(np.mean(max_corr[split:]))
    if second > first + 0.05:
        return f"increasing ({first:.3f} -> {second:.3f})"
    if second + 0.05 < first:
        return f"decreasing ({first:.3f} -> {second:.3f})"
    return f"flat ({first:.3f} -> {second:.3f})"


def _crowding_type(min_row: dict, d_alert: float) -> str:
    if min_row is None:
        return "unknown"
    count = 0
    for key, value in min_row.items():
        if key.startswith("d_") and key.endswith("_after") and float(value) < d_alert:
            count += 1
    if count <= 1:
        return "two-uav approach"
    return "multi-uav crowding"


def _write_summary_md(
    out_md: str,
    run_dir: str | None,
    checkpoint: str,
    config_path: str,
    episode_index: int,
    seed_base: int | None,
    rows: List[dict],
    summary: dict,
    cfg,
) -> None:
    min_row = _find_row(rows, int(summary["min_dist_step"]))
    collision_step = next((int(r["step"]) for r in rows if int(r["collision_event"]) == 1), None)
    last_step = int(rows[-1]["step"]) if rows else -1
    q_at_min = float(min_row["queue_total_active"]) if min_row is not None else float("nan")
    q_20 = _find_row(rows, max(0, last_step - 20))
    q_50 = _find_row(rows, max(0, last_step - 50))
    d_alert = float(cfg.avoidance_alert_factor) * float(cfg.d_safe)
    late_collision = bool(summary["collision"] and collision_step is not None and collision_step >= int(0.75 * cfg.T_steps))
    correction_trend = _correction_trend(rows, collision_step if collision_step is not None else last_step)
    seed = _episode_seed(seed_base, episode_index)

    lines = [
        f"# Debug Episode Summary",
        "",
        f"- config: `{config_path}`",
        f"- run_dir: `{run_dir}`" if run_dir else f"- checkpoint: `{checkpoint}`",
        f"- checkpoint: `{checkpoint}`",
        f"- reproduction_mode: sequential_resets_to_episode_index",
        f"- episode_index: `{episode_index}`",
        f"- episode_seed_base: `{seed_base}`",
        f"- episode_seed: `{seed}`",
        f"- steps: `{summary['steps']}`",
        f"- termination_reason: `{summary['termination_reason']}`",
        f"- collision: `{int(bool(summary['collision']))}`",
        f"- late_collision: `{int(late_collision)}`",
        f"- min_inter_uav_dist: `{float(summary['min_inter_uav_dist']):.6f}`",
        f"- min_dist_step: `{summary['min_dist_step']}`",
        f"- min_dist_pair: `{summary['min_dist_pair'][0]}-{summary['min_dist_pair'][1]}`",
        f"- queue_total_active_at_min_dist: `{q_at_min:.6f}`",
        f"- final_queue_total_active: `{float(summary['final_queue_total_active']):.6f}`",
        f"- max_correction_norm: `{float(summary['max_correction_norm']):.6f}`",
        f"- correction_trend_last_50_before_end_or_collision: `{correction_trend}`",
        f"- crowding_pattern_at_min_dist: `{_crowding_type(min_row, d_alert)}`",
    ]

    if q_20 is not None:
        lines.append(f"- queue_total_active_20_steps_before_end: `{float(q_20['queue_total_active']):.6f}`")
    if q_50 is not None:
        lines.append(f"- queue_total_active_50_steps_before_end: `{float(q_50['queue_total_active']):.6f}`")

    if collision_step is not None:
        lines.append(f"- collision_step: `{collision_step}`")
    alert_step = next((int(r["step"]) for r in rows if float(r["min_inter_uav_dist_after"]) < d_alert), None)
    if alert_step is not None:
        lines.append(f"- first_step_below_alert_dist: `{alert_step}`")

    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--episode_index", type=int, required=True)
    parser.add_argument("--episode_seed_base", type=int, default=None)
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--out_md", type=str, default=None)
    args = parser.parse_args()

    checkpoint, out_csv, out_md = _resolve_paths(
        args.run_dir, args.checkpoint, args.out_csv, args.out_md, args.episode_index
    )

    cfg = load_config(args.config)
    env = SaginParallelEnv(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Match evaluate.py exactly: one unseeded reset happens before the seeded
    # episode loop, and that reset also advances episode_idx/curriculum state.
    obs_probe, _ = env.reset()
    obs_dim = batch_flatten_obs(list(obs_probe.values()), cfg).shape[1]
    actor = ActorNet(obs_dim, cfg).to(device)
    actor.load_state_dict(torch.load(checkpoint, map_location=device))
    actor.eval()

    # Replay prior episodes in the same order as evaluate.py so episode_idx,
    # curriculum state, and adaptive avoidance evolve identically.
    for ep in range(args.episode_index):
        env.reset(seed=_episode_seed(args.episode_seed_base, ep))
        _run_episode(env, actor, device, record=False)

    env.reset(seed=_episode_seed(args.episode_seed_base, args.episode_index))
    rows, summary = _run_episode(env, actor, device, record=True)
    _write_csv(rows, out_csv)
    _write_summary_md(out_md, args.run_dir, checkpoint, args.config, args.episode_index, args.episode_seed_base, rows, summary, cfg)

    print(f"Saved debug CSV to {out_csv}")
    print(f"Saved summary MD to {out_md}")
    print(
        "Episode metrics: "
        f"steps={summary['steps']} "
        f"termination={summary['termination_reason']} "
        f"collision={int(bool(summary['collision']))} "
        f"min_inter_uav_dist={float(summary['min_inter_uav_dist']):.6f} "
        f"final_queue_total_active={float(summary['final_queue_total_active']):.6f}"
    )


if __name__ == "__main__":
    main()
