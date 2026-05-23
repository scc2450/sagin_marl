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
from sagin_marl.rl.critic import CriticNet
from sagin_marl.rl.mappo import _normalize_advantages, compute_gae
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


def _load_models(run_dir: Path, update: int, device: torch.device) -> tuple[Any, ActorNet, CriticNet, int, int]:
    cfg_path = run_dir / "config_source.yaml"
    cfg = load_config(str(cfg_path))
    env = make_structured_env(cfg, mode="script")
    obs, _ = env.reset(seed=0)
    obs_dim = batch_flatten_obs(list(obs.values()), cfg).shape[1]
    state_dim = int(env.get_global_state().shape[0])
    actor = ActorNet(obs_dim, cfg).to(device)
    critic = CriticNet(state_dim, obs_dim, cfg.num_uav, cfg).to(device)
    load_checkpoint_forgiving(actor, str(run_dir / f"actor_u{update:04d}.pt"), map_location=device, strict=True)
    load_checkpoint_forgiving(critic, str(run_dir / f"critic_u{update:04d}.pt"), map_location=device, strict=True)
    actor.eval()
    critic.eval()
    env.close()
    return cfg, actor, critic, obs_dim, state_dim


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, actor, critic, obs_dim, state_dim = _load_models(run_dir, int(args.update), device)

    env = make_structured_env(cfg, mode="script")
    context_rows: list[dict[str, Any]] = []
    combo_rows: list[dict[str, Any]] = []
    rollout_steps: list[dict[str, Any]] = []
    contexts_done = 0

    try:
        for ep in range(int(args.episodes)):
            seed = int(args.episode_seed_base) + ep
            np.random.seed(seed)
            torch.manual_seed(seed)
            obs, _ = env.reset(seed=seed)
            done = False
            step = 0
            while not done and contexts_done < int(args.max_contexts):
                obs_list = list(obs.values())
                state = np.asarray(env.get_global_state(), dtype=np.float32)
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
                obs_step_tensor = torch.tensor(obs_batch.reshape(1, cfg.num_uav, obs_dim), dtype=torch.float32, device=device)
                state_tensor = torch.tensor(state.reshape(1, state_dim), dtype=torch.float32, device=device)
                with torch.no_grad():
                    policy_out = actor.act(obs_tensor, deterministic=False, compute_logprob=False)
                    value = critic(state_tensor, obs_step_tensor).detach().cpu().numpy().reshape(-1)[0]
                    sat_forward = actor.forward(obs_tensor, required_heads=["sat"])
                accel_actions = policy_out.accel.detach().cpu().numpy() if policy_out.accel is not None else np.zeros((cfg.num_uav, 2), dtype=np.float32)
                sat_mask_policy = policy_out.sat_select_mask.detach().cpu().numpy() if policy_out.sat_select_mask is not None else np.zeros((cfg.num_uav, cfg.sats_obs_max), dtype=np.float32)
                sat_logits = sat_forward["sat_logits"].detach().cpu().numpy()

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
                        policy_slots, valid_flags = _selected_slots_for_mask(env, u, cand, sat_mask_policy[u], sat_pos, sat_vel)
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
                        heur_combo = tuple(heur_slots[:k]) if len(heur_slots) >= k else combos[0]
                        if len(policy_combo) != k:
                            continue

                        reward_list: list[float] = []
                        backhaul_list: list[float] = []
                        outflow_list: list[float] = []
                        logit_list: list[float] = []
                        heur_score_list: list[float] = []
                        combo_keys: list[tuple[int, ...]] = []
                        uav_queue_pre = float(env.uav_queue[u])

                        for combo in combos:
                            combo_keys.append(tuple(int(x) for x in combo))
                            sat_mask_all = np.asarray(sat_mask_policy, dtype=np.float32).copy()
                            sat_mask_all[u] = _combo_mask(cfg.sats_obs_max, combo)
                            actions_cf = assemble_actions(cfg, env.agents, accel_actions, bw_alloc=heur_bw, sat_select_mask=sat_mask_all)
                            env_cf = copy.deepcopy(env)
                            try:
                                _, rewards_cf, _, _, _ = env_cf.step(actions_cf)
                                reward_cf = float(list(rewards_cf.values())[0])
                                parts_cf = getattr(env_cf, "last_reward_parts", {})
                                assoc_after = np.asarray(env_cf.last_association, dtype=np.int32)
                                inflow_target = 0.0
                                if assoc_after.size > 0 and hasattr(env_cf, "last_gu_outflow"):
                                    inflow_target = float(np.sum(env_cf.last_gu_outflow[assoc_after == u]))
                                outflow_target = max(uav_queue_pre + inflow_target - float(env_cf.uav_queue[u]), 0.0)
                                combo_logit = float(np.sum(sat_logits[u, np.asarray(combo, dtype=np.int64)]))
                                combo_heur_score = float(np.sum(heur_scores[np.asarray(combo, dtype=np.int64)]))
                                reward_list.append(reward_cf)
                                backhaul_list.append(float(parts_cf.get("throughput_backhaul_norm", 0.0)))
                                outflow_list.append(float(outflow_target / max(float(cfg.tau0), 1e-9)))
                                logit_list.append(combo_logit)
                                heur_score_list.append(combo_heur_score)
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
                                        "policy_logit_sum": combo_logit,
                                        "heur_score_sum": combo_heur_score,
                                        "reward_step": reward_cf,
                                        "backhaul_step": float(parts_cf.get("throughput_backhaul_norm", 0.0)),
                                        "outflow_step": float(outflow_target / max(float(cfg.tau0), 1e-9)),
                                    }
                                )
                            finally:
                                env_cf.close()

                        if policy_combo not in combo_keys or heur_combo not in combo_keys:
                            continue
                        policy_idx = combo_keys.index(policy_combo)
                        heur_idx = combo_keys.index(heur_combo)
                        context_rows.append(
                            {
                                "episode": ep,
                                "seed": seed,
                                "step": step,
                                "uav": u,
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
                                "policy_local_reward_gap": float(max(reward_list) - reward_list[policy_idx]),
                                "policy_local_backhaul_gap": float(max(backhaul_list) - backhaul_list[policy_idx]),
                                "policy_local_outflow_gap": float(max(outflow_list) - outflow_list[policy_idx]),
                                "corr_logit_reward": _pearson(logit_list, reward_list),
                                "corr_logit_backhaul": _pearson(logit_list, backhaul_list),
                                "corr_logit_outflow": _pearson(logit_list, outflow_list),
                                "corr_heur_reward": _pearson(heur_score_list, reward_list),
                                "corr_heur_backhaul": _pearson(heur_score_list, backhaul_list),
                                "corr_heur_outflow": _pearson(heur_score_list, outflow_list),
                            }
                        )
                        contexts_done += 1

                actions = assemble_actions(cfg, env.agents, accel_actions, bw_alloc=heur_bw, sat_select_mask=sat_mask_policy)
                next_obs, rewards, terms, truncs, _ = env.step(actions)
                reward_scalar = float(list(rewards.values())[0])
                done = bool(list(terms.values())[0] or list(truncs.values())[0])
                rollout_steps.append(
                    {
                        "episode": ep,
                        "seed": seed,
                        "step": step,
                        "value": float(value),
                        "reward": reward_scalar,
                        "boundary": int(done),
                        "next_state": np.asarray(env.get_global_state(), dtype=np.float32).copy(),
                        "next_obs_batch": batch_flatten_obs(list(next_obs.values()), cfg).reshape(cfg.num_uav, obs_dim).copy(),
                    }
                )
                obs = next_obs
                step += 1
                if done:
                    break

        if not rollout_steps:
            raise RuntimeError("No rollout steps collected.")

        rewards_arr = np.asarray([row["reward"] for row in rollout_steps], dtype=np.float32)
        values_arr = np.asarray([row["value"] for row in rollout_steps], dtype=np.float32)
        boundaries_arr = np.asarray([bool(row["boundary"]) for row in rollout_steps], dtype=bool)
        bootstrap_values = np.zeros_like(rewards_arr, dtype=np.float32)
        for idx, row in enumerate(rollout_steps):
            if boundaries_arr[idx]:
                bootstrap_values[idx] = 0.0
            else:
                next_state_t = torch.tensor(row["next_state"].reshape(1, state_dim), dtype=torch.float32, device=device)
                next_obs_t = torch.tensor(row["next_obs_batch"].reshape(1, cfg.num_uav, obs_dim), dtype=torch.float32, device=device)
                with torch.no_grad():
                    bootstrap_values[idx] = float(critic(next_state_t, next_obs_t).detach().cpu().numpy().reshape(-1)[0])
        adv_raw, _ = compute_gae(rewards_arr, values_arr, bootstrap_values, boundaries_arr, float(cfg.gamma), float(cfg.gae_lambda))
        adv_final, adv_stats = _normalize_advantages(adv_raw, float(getattr(cfg, "adv_clip", 5.0) or 0.0))

        step_to_adv = {int(row["step"]): (float(adv_raw[idx]), float(adv_final[idx]), float(rewards_arr[idx])) for idx, row in enumerate(rollout_steps)}
        for row in context_rows:
            adv_raw_v, adv_final_v, reward_v = step_to_adv[int(row["step"])]
            row["joint_adv_raw"] = adv_raw_v
            row["joint_adv_final"] = adv_final_v
            row["joint_reward_step_actual"] = reward_v
            row["positive_joint_adv"] = int(adv_final_v > 0.0)
            row["bad_local_reward_but_positive_adv"] = int((row["policy_local_reward_gap"] > 1e-6) and (adv_final_v > 0.0))
            row["bad_local_backhaul_but_positive_adv"] = int((row["policy_local_backhaul_gap"] > 1e-9) and (adv_final_v > 0.0))
            row["bad_local_outflow_but_positive_adv"] = int((row["policy_local_outflow_gap"] > 1e-6) and (adv_final_v > 0.0))

        summary = {
            "run_dir": str(run_dir.resolve()),
            "update": int(args.update),
            "episodes": int(args.episodes),
            "episode_seed_base": int(args.episode_seed_base),
            "step_stride": int(args.step_stride),
            "max_contexts": int(args.max_contexts),
            "contexts_evaluated": len(context_rows),
            "combo_rows": len(combo_rows),
            "adv_stats": adv_stats,
            "credit_mismatch": {
                "positive_joint_adv_fraction": _safe_mean([float(r["positive_joint_adv"]) for r in context_rows]),
                "bad_local_reward_but_positive_adv_fraction": _safe_mean([float(r["bad_local_reward_but_positive_adv"]) for r in context_rows]),
                "bad_local_backhaul_but_positive_adv_fraction": _safe_mean([float(r["bad_local_backhaul_but_positive_adv"]) for r in context_rows]),
                "bad_local_outflow_but_positive_adv_fraction": _safe_mean([float(r["bad_local_outflow_but_positive_adv"]) for r in context_rows]),
                "corr_joint_adv_vs_policy_local_reward_gap": _pearson(
                    [float(r["joint_adv_final"]) for r in context_rows],
                    [float(r["policy_local_reward_gap"]) for r in context_rows],
                ),
                "corr_joint_adv_vs_policy_local_backhaul_gap": _pearson(
                    [float(r["joint_adv_final"]) for r in context_rows],
                    [float(r["policy_local_backhaul_gap"]) for r in context_rows],
                ),
                "corr_joint_adv_vs_policy_local_outflow_gap": _pearson(
                    [float(r["joint_adv_final"]) for r in context_rows],
                    [float(r["policy_local_outflow_gap"]) for r in context_rows],
                ),
            },
            "rank_summary": {
                "policy_reward_rank": _summarize([float(r["policy_reward_rank"]) for r in context_rows]),
                "heur_reward_rank": _summarize([float(r["heur_reward_rank"]) for r in context_rows]),
                "policy_backhaul_rank": _summarize([float(r["policy_backhaul_rank"]) for r in context_rows]),
                "heur_backhaul_rank": _summarize([float(r["heur_backhaul_rank"]) for r in context_rows]),
            },
            "worst_credit_mismatch_contexts": sorted(
                [r for r in context_rows if int(r["positive_joint_adv"]) == 1],
                key=lambda r: (-float(r["policy_local_reward_gap"]), -float(r["policy_local_backhaul_gap"])),
            )[:20],
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
