from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.rl.action_assembler import assemble_actions
from sagin_marl.rl.baselines import cluster_center_queue_aware_policy, queue_aware_bw_policy
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


CFG_PATH = Path("configs/phase1_actions_curriculum_joint_3heads_fading_interference_vsat_precomp_joint_puremappo_criticdecoupled.yaml")
OUTPUT_DIR = Path("runs/phase1_actions/bw_variant_fixed_frontsat_20260331")
EPISODES = 30
SEED_BASE = 53000


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _uniform_bw(obs_list, cfg) -> np.ndarray:
    out = np.zeros((len(obs_list), cfg.users_obs_max), dtype=np.float32)
    for i, obs in enumerate(obs_list):
        valid = np.asarray(obs.get("bw_valid_mask", obs["users_mask"]) > 0.0, dtype=bool)
        if np.any(valid):
            out[i, valid] = 1.0 / float(np.sum(valid))
    return out


def _aggressive_bw(obs_list, cfg) -> np.ndarray:
    del cfg
    out = np.zeros((len(obs_list), len(obs_list[0]["users_mask"])), dtype=np.float32)
    for i, obs in enumerate(obs_list):
        valid = np.asarray(obs.get("bw_valid_mask", obs["users_mask"]) > 0.0, dtype=bool)
        if not np.any(valid):
            continue
        eta = np.asarray(obs["users"][:, 3], dtype=np.float32)
        scaled = np.full_like(eta, -1e9, dtype=np.float32)
        scaled[valid] = 8.0 * eta[valid]
        scaled_valid = scaled[valid] - float(np.max(scaled[valid]))
        weights = np.exp(scaled_valid).astype(np.float32, copy=False)
        denom = float(np.sum(weights))
        if denom > 1e-8:
            out[i, valid] = weights / denom
        else:
            out[i, valid] = 1.0 / float(np.sum(valid))
    return out


def _step_metrics(parts: dict[str, Any]) -> dict[str, float]:
    return {
        "throughput_access_norm": float(parts.get("throughput_access_norm", 0.0)),
        "throughput_backhaul_norm": float(parts.get("throughput_backhaul_norm", 0.0)),
        "processed_ratio_eval": float(parts.get("processed_ratio_eval", 0.0)),
        "drop_ratio_eval": float(parts.get("drop_ratio_eval", 0.0)),
        "pre_backlog_steps_eval": float(parts.get("pre_backlog_steps_eval", 0.0)),
        "gu_queue_arrival_steps": float(parts.get("gu_queue_arrival_steps", 0.0)),
        "uav_queue_arrival_steps": float(parts.get("uav_queue_arrival_steps", 0.0)),
        "sat_queue_arrival_steps": float(parts.get("sat_queue_arrival_steps", 0.0)),
        "gu_drop_ratio_step": float(parts.get("gu_drop_ratio_step", 0.0)),
        "uav_drop_ratio_step": float(parts.get("uav_drop_ratio_step", 0.0)),
        "sat_drop_ratio_step": float(parts.get("sat_drop_ratio_step", 0.0)),
    }


def _evaluate_variant(name: str, bw_fn) -> tuple[list[dict[str, Any]], dict[str, float]]:
    cfg = load_config(str(CFG_PATH))
    env = make_structured_env(cfg, mode="script")
    per_episode: list[dict[str, Any]] = []
    for ep in range(EPISODES):
        obs, _ = env.reset(seed=SEED_BASE + ep)
        done = False
        steps = 0
        reward_sum = 0.0
        accum = {key: 0.0 for key in _step_metrics({}).keys()}
        while not done:
            obs_list = list(obs.values())
            centers = getattr(env, "gu_cluster_centers", None)
            counts = getattr(env, "gu_cluster_counts", None)
            accel, _, sat = cluster_center_queue_aware_policy(obs_list, cfg, centers, counts)
            bw = bw_fn(obs_list, cfg)
            actions = assemble_actions(cfg, env.agents, accel, bw_alloc=bw, sat_select_mask=sat)
            obs, rewards, terms, truncs, _ = env.step(actions)
            reward_sum += float(list(rewards.values())[0])
            parts = dict(getattr(env, "last_reward_parts", {}) or {})
            step_vals = _step_metrics(parts)
            for key in accum:
                accum[key] += step_vals[key]
            steps += 1
            done = bool(list(terms.values())[0] or list(truncs.values())[0])
        row = {"variant": name, "episode": ep, "seed": SEED_BASE + ep, "reward_sum": reward_sum}
        for key, total in accum.items():
            row[key] = total / max(steps, 1)
        per_episode.append(row)
    summary = {"variant": name}
    metric_keys = [key for key in per_episode[0].keys() if key not in {"variant", "episode", "seed"}]
    for key in metric_keys:
        vals = np.asarray([row[key] for row in per_episode], dtype=np.float64)
        summary[key] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))
    return per_episode, summary


def main() -> None:
    variants = [
        ("aggressive_bw", _aggressive_bw),
        ("uniform_bw", _uniform_bw),
        ("queue_aware_bw", queue_aware_bw_policy),
    ]
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, float]] = []
    for name, bw_fn in variants:
        rows, summary = _evaluate_variant(name, bw_fn)
        all_rows.extend(rows)
        summaries.append(summary)
    _write_csv(OUTPUT_DIR / "per_episode.csv", all_rows, list(all_rows[0].keys()))
    _write_csv(OUTPUT_DIR / "summary.csv", summaries, list(summaries[0].keys()))
    _write_json(
        OUTPUT_DIR / "summary.json",
        {
            "config": str(CFG_PATH),
            "episodes": EPISODES,
            "seed_base": SEED_BASE,
            "summary_csv": str(OUTPUT_DIR / "summary.csv"),
            "per_episode_csv": str(OUTPUT_DIR / "per_episode.csv"),
            "summaries": summaries,
        },
    )
    print(f"Wrote diagnostics to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
