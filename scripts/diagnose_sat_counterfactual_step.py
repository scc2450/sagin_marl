from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

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
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode-seed-base", type=int, default=42000)
    parser.add_argument("--step-stride", type=int, default=4)
    parser.add_argument("--max-contexts", type=int, default=300)
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args()


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "p50": None, "p90": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return None
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if np.std(xa) <= 1e-12 or np.std(ya) <= 1e-12:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def _rank_desc(values: list[float], target_idx: int) -> int:
    order = np.argsort(-np.asarray(values, dtype=np.float64), kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1)
    return int(ranks[target_idx])


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


def _combo_mask(num_slots: int, combo: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros((num_slots,), dtype=np.float32)
    for idx in combo:
        mask[int(idx)] = 1.0
    return mask


def _load_actor(cfg_path: Path, checkpoint_path: Path, device: torch.device) -> tuple[Any, ActorNet]:
    cfg = load_config(str(cfg_path))
    env = make_structured_env(cfg, mode="script")
    obs, _ = env.reset(seed=0)
    obs_dim = batch_flatten_obs(list(obs.values()), cfg).shape[1]
    actor = ActorNet(obs_dim, cfg).to(device)
    load_checkpoint_forgiving(actor, str(checkpoint_path), map_location=device, strict=True)
    actor.eval()
    env.close()
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
    combo_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    global_policy_combo_logit_sum: list[float] = []
    global_heur_combo_score_sum: list[float] = []
    global_reward_step: list[float] = []
    global_backhaul_step: list[float] = []
    global_outflow_step: list[float] = []
    contexts_done = 0

    try:
        for ep in range(int(args.episodes)):
            seed = int(args.episode_seed_base) + ep
            obs, _ = env.reset(seed=seed)
            done = False
            step = 0
            while not done and contexts_done < int(args.max_contexts):
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
                    sat_out = actor.forward(obs_tensor, required_heads=["sat"])
                sat_logits = sat_out["sat_logits"].detach().cpu().numpy()
                policy_sat = actor.act(obs_tensor, deterministic=True, required_heads=["sat"], compute_logprob=False).sat_select_mask
                policy_sat = policy_sat.detach().cpu().numpy() if policy_sat is not None else np.zeros((cfg.num_uav, cfg.sats_obs_max), dtype=np.float32)

                if step % max(int(args.step_stride), 1) == 0:
                    for u, agent in enumerate(env.agents):
                        if contexts_done >= int(args.max_contexts):
                            break
                        cand = visible[u][: cfg.sats_obs_max]
                        if not cand:
                            continue
                        sat_obs = np.asarray(obs[agent]["sats"], dtype=np.float32)
                        sat_valid_mask = np.asarray(obs[agent].get("sat_valid_mask", obs[agent]["sats_mask"]) > 0.0)
                        heur_scores = _sat_heuristic_score(sat_obs, sat_valid_mask, cfg)
                        policy_slots, valid_flags = _selected_slots_for_mask(env, u, cand, policy_sat[u], sat_pos, sat_vel)
                        heur_slots, _ = _selected_slots_for_mask(env, u, cand, heur_sat[u], sat_pos, sat_vel)
                        valid_slots = np.flatnonzero(valid_flags & sat_valid_mask[: len(cand)])
                        if valid_slots.size == 0:
                            continue
                        k = min(int(cfg.N_RF), int(valid_slots.size))
                        if k <= 0:
                            continue
                        combos = list(itertools.combinations(valid_slots.tolist(), k))
                        if not combos:
                            continue
                        policy_combo = tuple(policy_slots[:k])
                        heur_combo = tuple(heur_slots[:k])
                        if len(policy_combo) != k:
                            continue
                        if len(heur_combo) != k:
                            heur_combo = combos[0]

                        rank_data = env._sat_candidate_rank_data(
                            u,
                            np.asarray(cand, dtype=np.int32),
                            sat_pos,
                            elev_values=elev_matrix[u, np.asarray(cand, dtype=np.int32)],
                        )
                        uav_queue_pre = float(env.uav_queue[u])
                        reward_list: list[float] = []
                        backhaul_list: list[float] = []
                        outflow_list: list[float] = []
                        logit_list: list[float] = []
                        heur_score_list: list[float] = []
                        combo_keys: list[tuple[int, ...]] = []

                        for combo in combos:
                            combo_keys.append(tuple(int(x) for x in combo))
                            combo_mask = _combo_mask(cfg.sats_obs_max, combo)
                            sat_mask_all = np.asarray(policy_sat, dtype=np.float32).copy()
                            sat_mask_all[u] = combo_mask
                            actions = assemble_actions(cfg, env.agents, heur_accel, bw_alloc=heur_bw, sat_select_mask=sat_mask_all)
                            env_cf = copy.deepcopy(env)
                            try:
                                _, rewards_cf, _, _, _ = env_cf.step(actions)
                                reward_cf = float(list(rewards_cf.values())[0])
                                parts_cf = getattr(env_cf, "last_reward_parts", {})
                                assoc_after = np.asarray(env_cf.last_association, dtype=np.int32)
                                inflow_target = 0.0
                                if assoc_after.size > 0 and hasattr(env_cf, "last_gu_outflow"):
                                    inflow_target = float(np.sum(env_cf.last_gu_outflow[assoc_after == u]))
                                outflow_target = max(uav_queue_pre + inflow_target - float(env_cf.uav_queue[u]), 0.0)
                                reward_list.append(reward_cf)
                                backhaul_list.append(float(parts_cf.get("throughput_backhaul_norm", 0.0)))
                                outflow_list.append(float(outflow_target / max(float(cfg.tau0), 1e-9)))
                                combo_logit = float(np.sum(sat_logits[u, np.asarray(combo, dtype=np.int64)]))
                                combo_heur_score = float(np.sum(heur_scores[np.asarray(combo, dtype=np.int64)]))
                                logit_list.append(combo_logit)
                                heur_score_list.append(combo_heur_score)
                                global_policy_combo_logit_sum.append(combo_logit)
                                global_heur_combo_score_sum.append(combo_heur_score)
                                global_reward_step.append(reward_cf)
                                global_backhaul_step.append(float(parts_cf.get("throughput_backhaul_norm", 0.0)))
                                global_outflow_step.append(float(outflow_target / max(float(cfg.tau0), 1e-9)))

                                combo_rows.append(
                                    {
                                        "episode": ep,
                                        "seed": seed,
                                        "step": step,
                                        "uav": u,
                                        "combo_slots": "|".join(str(x) for x in combo),
                                        "combo_sat_ids": "|".join(str(int(cand[idx])) for idx in combo),
                                        "is_policy_combo": int(tuple(combo) == policy_combo),
                                        "is_heur_combo": int(tuple(combo) == heur_combo),
                                        "policy_combo_logit_sum": combo_logit,
                                        "heuristic_combo_score_sum": combo_heur_score,
                                        "reward_step": reward_cf,
                                        "throughput_backhaul_norm_step": float(parts_cf.get("throughput_backhaul_norm", 0.0)),
                                        "processed_ratio_eval_step": float(parts_cf.get("processed_ratio_eval", 0.0)),
                                        "pre_backlog_steps_eval_step": float(parts_cf.get("pre_backlog_steps_eval", 0.0)),
                                        "target_uav_outflow_rate_step": float(outflow_target / max(float(cfg.tau0), 1e-9)),
                                        "target_connected_sat_dist_mean": float(np.mean(np.asarray(rank_data["distance"], dtype=np.float64)[np.asarray(combo, dtype=np.int64)])),
                                        "target_selected_se_mean": float(np.mean(np.asarray(rank_data["spectral_efficiency"], dtype=np.float64)[np.asarray(combo, dtype=np.int64)])),
                                        "target_selected_elevation_mean": float(np.mean(np.asarray(rank_data["elevation"], dtype=np.float64)[np.asarray(combo, dtype=np.int64)])),
                                    }
                                )
                            finally:
                                env_cf.close()

                        if not reward_list:
                            continue
                        policy_idx = combo_keys.index(policy_combo) if policy_combo in combo_keys else None
                        heur_idx = combo_keys.index(heur_combo) if heur_combo in combo_keys else None
                        if policy_idx is None or heur_idx is None:
                            continue

                        context_rows.append(
                            {
                                "episode": ep,
                                "seed": seed,
                                "step": step,
                                "uav": u,
                                "num_combos": len(combo_keys),
                                "policy_slots": "|".join(str(x) for x in policy_combo),
                                "heur_slots": "|".join(str(x) for x in heur_combo),
                                "policy_reward_rank": _rank_desc(reward_list, policy_idx),
                                "heur_reward_rank": _rank_desc(reward_list, heur_idx),
                                "policy_backhaul_rank": _rank_desc(backhaul_list, policy_idx),
                                "heur_backhaul_rank": _rank_desc(backhaul_list, heur_idx),
                                "policy_outflow_rank": _rank_desc(outflow_list, policy_idx),
                                "heur_outflow_rank": _rank_desc(outflow_list, heur_idx),
                                "policy_reward_step": reward_list[policy_idx],
                                "heur_reward_step": reward_list[heur_idx],
                                "best_reward_step": float(max(reward_list)),
                                "policy_backhaul_step": backhaul_list[policy_idx],
                                "heur_backhaul_step": backhaul_list[heur_idx],
                                "best_backhaul_step": float(max(backhaul_list)),
                                "policy_outflow_step": outflow_list[policy_idx],
                                "heur_outflow_step": outflow_list[heur_idx],
                                "best_outflow_step": float(max(outflow_list)),
                                "corr_logit_reward": _pearson(logit_list, reward_list),
                                "corr_logit_backhaul": _pearson(logit_list, backhaul_list),
                                "corr_logit_outflow": _pearson(logit_list, outflow_list),
                                "corr_heur_reward": _pearson(heur_score_list, reward_list),
                                "corr_heur_backhaul": _pearson(heur_score_list, backhaul_list),
                                "corr_heur_outflow": _pearson(heur_score_list, outflow_list),
                            }
                        )
                        contexts_done += 1

                actions_rollout = assemble_actions(cfg, env.agents, heur_accel, bw_alloc=heur_bw, sat_select_mask=policy_sat)
                obs, rewards, terms, truncs, _ = env.step(actions_rollout)
                done = bool(list(terms.values())[0] or list(truncs.values())[0])
                step += 1

        summary = {
            "run_dir": str(run_dir.resolve()),
            "update": int(args.update),
            "episodes": int(args.episodes),
            "episode_seed_base": int(args.episode_seed_base),
            "step_stride": int(args.step_stride),
            "max_contexts": int(args.max_contexts),
            "contexts_evaluated": len(context_rows),
            "combo_rows": len(combo_rows),
            "policy_vs_immediate": {
                "policy_reward_rank": _summarize([float(r["policy_reward_rank"]) for r in context_rows]),
                "heur_reward_rank": _summarize([float(r["heur_reward_rank"]) for r in context_rows]),
                "policy_backhaul_rank": _summarize([float(r["policy_backhaul_rank"]) for r in context_rows]),
                "heur_backhaul_rank": _summarize([float(r["heur_backhaul_rank"]) for r in context_rows]),
                "policy_outflow_rank": _summarize([float(r["policy_outflow_rank"]) for r in context_rows]),
                "heur_outflow_rank": _summarize([float(r["heur_outflow_rank"]) for r in context_rows]),
                "policy_best_reward_fraction": _safe_mean([1.0 if int(r["policy_reward_rank"]) == 1 else 0.0 for r in context_rows]),
                "heur_best_reward_fraction": _safe_mean([1.0 if int(r["heur_reward_rank"]) == 1 else 0.0 for r in context_rows]),
                "policy_best_backhaul_fraction": _safe_mean([1.0 if int(r["policy_backhaul_rank"]) == 1 else 0.0 for r in context_rows]),
                "heur_best_backhaul_fraction": _safe_mean([1.0 if int(r["heur_backhaul_rank"]) == 1 else 0.0 for r in context_rows]),
                "policy_best_outflow_fraction": _safe_mean([1.0 if int(r["policy_outflow_rank"]) == 1 else 0.0 for r in context_rows]),
                "heur_best_outflow_fraction": _safe_mean([1.0 if int(r["heur_outflow_rank"]) == 1 else 0.0 for r in context_rows]),
            },
            "global_combo_correlations": {
                "policy_logit_vs_reward": _pearson(global_policy_combo_logit_sum, global_reward_step),
                "policy_logit_vs_backhaul": _pearson(global_policy_combo_logit_sum, global_backhaul_step),
                "policy_logit_vs_outflow": _pearson(global_policy_combo_logit_sum, global_outflow_step),
                "heur_score_vs_reward": _pearson(global_heur_combo_score_sum, global_reward_step),
                "heur_score_vs_backhaul": _pearson(global_heur_combo_score_sum, global_backhaul_step),
                "heur_score_vs_outflow": _pearson(global_heur_combo_score_sum, global_outflow_step),
            },
            "worst_policy_reward_rank_contexts": sorted(context_rows, key=lambda r: (-int(r["policy_reward_rank"]), float(r["policy_reward_step"] - r["best_reward_step"])))[:20],
            "context_csv": str((out_dir / "context_rows.csv").resolve()),
            "combo_csv": str((out_dir / "combo_rows.csv").resolve()),
            "summary_json": str((out_dir / "summary.json").resolve()),
        }

        with (out_dir / "context_rows.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(context_rows[0].keys()) if context_rows else ["episode", "seed", "step", "uav"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(context_rows)

        with (out_dir / "combo_rows.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(combo_rows[0].keys()) if combo_rows else ["episode", "seed", "step", "uav", "combo_slots"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combo_rows)

        with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
