from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.rl.action_assembler import assemble_actions
from sagin_marl.rl.baselines import _sat_heuristic_score, cluster_center_queue_aware_policy
from sagin_marl.rl.policy import ActorNet, batch_flatten_obs
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-seed-base", type=int, default=42000)
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args()


def _current_visible(env: SaginParallelEnv, sat_pos: np.ndarray) -> list[list[int]]:
    visible = [list(v) for v in getattr(env, "last_visible_candidates", [])]
    if len(visible) != env.cfg.num_uav:
        visible = env._visible_sats_sorted(sat_pos)
    return [list(v[: env.cfg.sats_obs_max]) for v in visible]


def _selected_slots_for_mask(
    env: SaginParallelEnv,
    u: int,
    cand: list[int],
    sat_raw: np.ndarray,
    sat_pos: np.ndarray,
    sat_vel: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    cfg = env.cfg
    sat_raw = np.asarray(sat_raw, dtype=np.float32)[: len(cand)]
    valid_flags = np.ones((len(cand),), dtype=bool)
    if cfg.doppler_enabled and len(cand) > 0:
        cand_idx = np.asarray(cand, dtype=np.int32)
        raw_nu = env._doppler_many(u, cand_idx, sat_pos, sat_vel)
        nu_eff, _ = env._effective_doppler_array(u, cand_idx, raw_nu)
        valid_flags = np.abs(nu_eff) <= cfg.nu_max
    valid_slots = np.flatnonzero(valid_flags)
    if valid_slots.size == 0:
        return [], valid_flags
    chosen_slots = np.flatnonzero((sat_raw > 0.5) & valid_flags)
    if chosen_slots.size > cfg.N_RF:
        order = np.argsort(-sat_raw[chosen_slots], kind="stable")
        chosen_slots = chosen_slots[order[: cfg.N_RF]]
    if chosen_slots.size == 0:
        best_slot = int(valid_slots[int(np.argmax(sat_raw[valid_slots]))])
        chosen_slots = np.array([best_slot], dtype=np.int64)
    chosen_slots = chosen_slots[: cfg.N_RF]
    return [int(x) for x in chosen_slots.tolist()], valid_flags


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _load_actor(cfg_path: Path, checkpoint_path: Path, device: torch.device) -> tuple[Any, ActorNet]:
    cfg = load_config(str(cfg_path))
    env = make_structured_env(cfg, mode="script")
    obs, _ = env.reset(seed=0)
    obs_dim = batch_flatten_obs(list(obs.values()), cfg).shape[1]
    actor = ActorNet(obs_dim, cfg).to(device)
    load_checkpoint_forgiving(actor, str(checkpoint_path), map_location=device, strict=True)
    actor.eval()
    return cfg, actor


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config_source.yaml"
    checkpoint_path = run_dir / f"actor_u{args.update:04d}.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, actor = _load_actor(cfg_path, checkpoint_path, device)
    env = make_structured_env(cfg, mode="script")
    selection_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    policy_only_counter: Counter[int] = Counter()
    heuristic_only_counter: Counter[int] = Counter()

    for ep in range(int(args.episodes)):
        seed = int(args.episode_seed_base) + ep
        obs, _ = env.reset(seed=seed)
        done = False
        step = 0
        reward_sum = 0.0
        while not done:
            obs_list = list(obs.values())
            sat_pos, sat_vel = env._get_orbit_states()
            visible = _current_visible(env, sat_pos)
            elev_matrix = env._get_elevation_matrix(sat_pos)
            heur_accel, heur_bw, heur_sat = cluster_center_queue_aware_policy(
                obs_list,
                cfg,
                getattr(env, "gu_cluster_centers", None),
                getattr(env, "gu_cluster_counts", None),
            )
            obs_batch = batch_flatten_obs(obs_list, cfg)
            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
            with torch.no_grad():
                policy_out = actor.act(obs_tensor, deterministic=True)
            policy_sat = (
                policy_out.sat_select_mask.detach().cpu().numpy()
                if policy_out.sat_select_mask is not None
                else np.zeros((cfg.num_uav, cfg.sats_obs_max), dtype=np.float32)
            )

            for u, agent in enumerate(env.agents):
                cand = visible[u][: cfg.sats_obs_max]
                if not cand:
                    continue
                sat_obs = np.asarray(obs[agent]["sats"], dtype=np.float32)
                sat_valid_mask = np.asarray(obs[agent].get("sat_valid_mask", obs[agent]["sats_mask"]) > 0.0)
                heur_scores = _sat_heuristic_score(sat_obs, sat_valid_mask, cfg)
                rank_data = env._sat_candidate_rank_data(
                    u,
                    np.asarray(cand, dtype=np.int32),
                    sat_pos,
                    elev_values=elev_matrix[u, np.asarray(cand, dtype=np.int32)],
                )
                policy_slots, valid_flags = _selected_slots_for_mask(env, u, cand, policy_sat[u], sat_pos, sat_vel)
                heur_slots, _ = _selected_slots_for_mask(env, u, cand, heur_sat[u], sat_pos, sat_vel)
                valid_slots = np.flatnonzero(valid_flags & sat_valid_mask[: len(cand)])
                if valid_slots.size == 0:
                    continue
                heur_order = valid_slots[np.argsort(heur_scores[valid_slots])[::-1]]
                heur_rank_lookup = {int(slot): int(rank + 1) for rank, slot in enumerate(heur_order.tolist())}
                policy_ids = [int(cand[idx]) for idx in policy_slots]
                heur_ids = [int(cand[idx]) for idx in heur_slots]
                policy_only = sorted(set(policy_ids) - set(heur_ids))
                heur_only = sorted(set(heur_ids) - set(policy_ids))
                for sat_id in policy_only:
                    policy_only_counter[sat_id] += 1
                for sat_id in heur_only:
                    heuristic_only_counter[sat_id] += 1

                def collect(slots: list[int], key: str) -> float | None:
                    if not slots:
                        return None
                    vals = np.asarray(rank_data[key], dtype=np.float64)[np.asarray(slots, dtype=np.int64)]
                    return float(np.mean(vals))

                def collect_heur_rank(slots: list[int]) -> float | None:
                    if not slots:
                        return None
                    ranks = [heur_rank_lookup.get(int(slot)) for slot in slots if int(slot) in heur_rank_lookup]
                    return _safe_mean([float(x) for x in ranks if x is not None])

                row = {
                    "episode": ep,
                    "seed": seed,
                    "step": step,
                    "uav": u,
                    "policy_sat_ids": "|".join(str(x) for x in policy_ids),
                    "heur_sat_ids": "|".join(str(x) for x in heur_ids),
                    "policy_only_sat_ids": "|".join(str(x) for x in policy_only),
                    "heur_only_sat_ids": "|".join(str(x) for x in heur_only),
                    "exact_match": int(set(policy_ids) == set(heur_ids)),
                    "jaccard": float(len(set(policy_ids) & set(heur_ids)) / max(len(set(policy_ids) | set(heur_ids)), 1)),
                    "policy_env_rank_mean": _safe_mean([float(x + 1) for x in policy_slots]),
                    "heur_env_rank_mean": _safe_mean([float(x + 1) for x in heur_slots]),
                    "policy_heur_rank_mean": collect_heur_rank(policy_slots),
                    "heur_heur_rank_mean": collect_heur_rank(heur_slots),
                    "policy_distance_mean": collect(policy_slots, "distance"),
                    "heur_distance_mean": collect(heur_slots, "distance"),
                    "policy_se_mean": collect(policy_slots, "spectral_efficiency"),
                    "heur_se_mean": collect(heur_slots, "spectral_efficiency"),
                    "policy_queue_norm_mean": collect(policy_slots, "queue_norm"),
                    "heur_queue_norm_mean": collect(heur_slots, "queue_norm"),
                    "policy_candidate_score_mean": collect(policy_slots, "score"),
                    "heur_candidate_score_mean": collect(heur_slots, "score"),
                    "policy_heur_score_mean": _safe_mean([float(heur_scores[idx]) for idx in policy_slots]),
                    "heur_heur_score_mean": _safe_mean([float(heur_scores[idx]) for idx in heur_slots]),
                    "policy_elevation_mean": collect(policy_slots, "elevation"),
                    "heur_elevation_mean": collect(heur_slots, "elevation"),
                    "policy_valid_count": len(policy_slots),
                    "heur_valid_count": len(heur_slots),
                    "visible_count": len(cand),
                }
                selection_rows.append(row)

            actions = assemble_actions(cfg, env.agents, heur_accel, bw_alloc=heur_bw, sat_select_mask=policy_sat)
            obs, rewards, terms, truncs, _ = env.step(actions)
            reward_sum += float(list(rewards.values())[0])
            done = bool(list(terms.values())[0] or list(truncs.values())[0])
            step += 1

        episode_rows.append({
            "episode": ep,
            "seed": seed,
            "reward_sum": reward_sum,
            "steps": step,
            "uav_queue_mean_end": float(np.mean(env.uav_queue)),
            "sat_queue_mean_end": float(np.mean(env.sat_queue)),
            "connected_sat_dist_mean_last": float(getattr(env, "last_connected_sat_dist_mean", 0.0)),
        })

    def col_mean(name: str) -> float | None:
        vals = []
        for row in selection_rows:
            val = row.get(name)
            if val is not None:
                vals.append(float(val))
        return _safe_mean(vals)

    worst_rows = sorted(
        selection_rows,
        key=lambda row: (float(row["policy_heur_score_mean"] or -1e9) - float(row["heur_heur_score_mean"] or -1e9)),
    )[:20]

    summary = {
        "run_dir": str(run_dir),
        "update": int(args.update),
        "episodes": int(args.episodes),
        "episode_seed_base": int(args.episode_seed_base),
        "selection_row_count": len(selection_rows),
        "exact_match_fraction": col_mean("exact_match"),
        "mean_jaccard": col_mean("jaccard"),
        "policy_vs_heur_selected": {
            "env_rank_mean": {"policy": col_mean("policy_env_rank_mean"), "heuristic": col_mean("heur_env_rank_mean")},
            "heur_rank_mean": {"policy": col_mean("policy_heur_rank_mean"), "heuristic": col_mean("heur_heur_rank_mean")},
            "distance_mean": {"policy": col_mean("policy_distance_mean"), "heuristic": col_mean("heur_distance_mean")},
            "spectral_efficiency_mean": {"policy": col_mean("policy_se_mean"), "heuristic": col_mean("heur_se_mean")},
            "queue_norm_mean": {"policy": col_mean("policy_queue_norm_mean"), "heuristic": col_mean("heur_queue_norm_mean")},
            "candidate_score_mean": {"policy": col_mean("policy_candidate_score_mean"), "heuristic": col_mean("heur_candidate_score_mean")},
            "heuristic_score_mean": {"policy": col_mean("policy_heur_score_mean"), "heuristic": col_mean("heur_heur_score_mean")},
            "elevation_mean": {"policy": col_mean("policy_elevation_mean"), "heuristic": col_mean("heur_elevation_mean")},
        },
        "top_policy_only_satellites": [{"sat_id": int(k), "count": int(v)} for k, v in policy_only_counter.most_common(10)],
        "top_heuristic_only_satellites": [{"sat_id": int(k), "count": int(v)} for k, v in heuristic_only_counter.most_common(10)],
        "worst_heuristic_score_gap_samples": worst_rows,
        "episode_rows": episode_rows,
        "selection_csv": str(out_dir / "selection_rows.csv"),
        "summary_json": str(out_dir / "summary.json"),
    }

    with (out_dir / "selection_rows.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(selection_rows[0].keys()) if selection_rows else ["episode", "seed", "step", "uav"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selection_rows:
            writer.writerow(row)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
