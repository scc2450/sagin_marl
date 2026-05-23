from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
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
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-seed-base", type=int, default=42000)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument(
        "--modes",
        type=str,
        default="policy_identity,policy_reverse,policy_random,heuristic_sat",
    )
    return parser.parse_args()


def _summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "p50": None,
            "p90": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _summarize_row_metrics(rows: list[dict[str, Any]], exclude_keys: set[str] | None = None) -> dict[str, float | None]:
    if not rows:
        return {}
    exclude = set() if exclude_keys is None else set(exclude_keys)
    out: dict[str, float | None] = {}
    for key in rows[0].keys():
        if key in exclude:
            continue
        values: list[float] = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                values.append(float(value))
        if not values:
            continue
        for stat_name, stat_value in _summarize(values).items():
            out[f"{key}_{stat_name}"] = stat_value
    return out


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


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


def _make_perm_to_orig(valid_mask: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    sat_slots = int(valid_mask.shape[0])
    perm_to_orig = np.arange(sat_slots, dtype=np.int64)
    valid_idx = np.flatnonzero(valid_mask > 0.5)
    if valid_idx.size <= 1:
        return perm_to_orig
    if mode == "policy_identity":
        ordered = valid_idx
    elif mode == "policy_reverse":
        ordered = valid_idx[::-1]
    elif mode == "policy_random":
        ordered = rng.permutation(valid_idx)
    else:
        raise ValueError(f"Unsupported permutation mode: {mode}")
    perm_to_orig[valid_idx] = ordered
    return perm_to_orig


def _permute_obs(obs: dict[str, np.ndarray], perm_to_orig: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in obs.items():
        arr = np.asarray(value)
        if key == "sats":
            out[key] = np.array(arr[perm_to_orig], copy=True)
        elif key in {"sats_mask", "sat_valid_mask"}:
            out[key] = np.array(arr[perm_to_orig], copy=True)
        else:
            out[key] = np.array(arr, copy=True)
    return out


def _map_mask_back(mask_perm: np.ndarray, perm_to_orig: np.ndarray) -> np.ndarray:
    mask_orig = np.zeros_like(mask_perm, dtype=np.float32)
    mask_orig[perm_to_orig] = np.asarray(mask_perm, dtype=np.float32)
    return mask_orig


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


def _step_metric_row(mode: str, episode: int, seed: int, reward_sum: float, steps: int, metric_sums: dict[str, float], extra: dict[str, Any]) -> dict[str, Any]:
    denom = max(int(steps), 1)
    row = {
        "mode": mode,
        "episode": int(episode),
        "episode_seed": int(seed),
        "reward_sum": float(reward_sum),
        "steps": int(steps),
        "processed_ratio_eval": float(metric_sums["processed_ratio_eval"] / denom),
        "drop_ratio_eval": float(metric_sums["drop_ratio_eval"] / denom),
        "pre_backlog_steps_eval": float(metric_sums["pre_backlog_steps_eval"] / denom),
        "throughput_backhaul_norm": float(metric_sums["throughput_backhaul_norm"] / denom),
        "throughput_access_norm": float(metric_sums["throughput_access_norm"] / denom),
        "assoc_dist_mean": float(metric_sums["assoc_dist_mean"] / denom),
        "connected_sat_dist_mean": float(metric_sums["connected_sat_dist_mean"] / denom),
        "elapsed_sec": float(extra.pop("elapsed_sec")),
    }
    row.update(extra)
    return row


def main() -> None:
    args = parse_args()
    modes = [token.strip() for token in str(args.modes).split(",") if token.strip()]
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config_source.yaml"
    checkpoint_path = run_dir / f"actor_u{args.update:04d}.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, actor = _load_actor(cfg_path, checkpoint_path, device)

    summary_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    mode_summaries: dict[str, Any] = {}

    for mode_idx, mode in enumerate(modes):
        env = make_structured_env(cfg, mode="script")
        episode_rows: list[dict[str, Any]] = []
        print(f"[sat-slot-perm] evaluating mode={mode}")
        try:
            for ep in range(int(args.episodes)):
                seed = int(args.episode_seed_base) + ep
                perm_rng = np.random.default_rng(seed + 100003 * (mode_idx + 1))
                obs, _ = env.reset(seed=seed)
                reward_sum = 0.0
                steps = 0
                metric_sums = {
                    "processed_ratio_eval": 0.0,
                    "drop_ratio_eval": 0.0,
                    "pre_backlog_steps_eval": 0.0,
                    "throughput_backhaul_norm": 0.0,
                    "throughput_access_norm": 0.0,
                    "assoc_dist_mean": 0.0,
                    "connected_sat_dist_mean": 0.0,
                }
                ep_policy_env_ranks: list[float] = []
                ep_policy_heur_ranks: list[float] = []
                ep_policy_distance: list[float] = []
                ep_policy_se: list[float] = []
                ep_policy_elevation: list[float] = []
                ep_exact_match: list[float] = []
                ep_jaccard: list[float] = []
                t0 = time.perf_counter()

                done = False
                step = 0
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
                        ctx = actor.encode_ctx(obs_tensor)
                        accel_out = actor.act(
                            obs_tensor,
                            deterministic=True,
                            required_heads=["accel"],
                            compute_logprob=False,
                            ctx=ctx,
                        )
                    accel_actions = accel_out.accel.detach().cpu().numpy() if accel_out.accel is not None else np.zeros((cfg.num_uav, 2), dtype=np.float32)

                    if mode == "heuristic_sat":
                        sat_masks_orig = np.asarray(heur_sat, dtype=np.float32)
                        perm_to_orig_list = [np.arange(cfg.sats_obs_max, dtype=np.int64) for _ in range(cfg.num_uav)]
                    else:
                        perm_to_orig_list: list[np.ndarray] = []
                        permuted_obs_list: list[dict[str, np.ndarray]] = []
                        for obs_agent in obs_list:
                            sat_valid_mask = np.asarray(obs_agent.get("sat_valid_mask", obs_agent["sats_mask"]), dtype=np.float32)
                            perm_to_orig = _make_perm_to_orig(sat_valid_mask[: cfg.sats_obs_max], mode, perm_rng)
                            perm_to_orig_list.append(perm_to_orig)
                            permuted_obs_list.append(_permute_obs(obs_agent, perm_to_orig))
                        perm_batch = batch_flatten_obs(permuted_obs_list, cfg)
                        perm_tensor = torch.tensor(perm_batch, dtype=torch.float32, device=device)
                        with torch.no_grad():
                            sat_out = actor.act(
                                perm_tensor,
                                deterministic=True,
                                required_heads=["sat"],
                                compute_logprob=False,
                                ctx=ctx,
                            )
                        sat_perm = sat_out.sat_select_mask.detach().cpu().numpy() if sat_out.sat_select_mask is not None else np.zeros((cfg.num_uav, cfg.sats_obs_max), dtype=np.float32)
                        sat_masks_orig = np.stack(
                            [_map_mask_back(sat_perm[u], perm_to_orig_list[u]) for u in range(cfg.num_uav)],
                            axis=0,
                        ).astype(np.float32)

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
                        chosen_slots, valid_flags = _selected_slots_for_mask(env, u, cand, sat_masks_orig[u], sat_pos, sat_vel)
                        heur_slots, _ = _selected_slots_for_mask(env, u, cand, heur_sat[u], sat_pos, sat_vel)
                        valid_slots = np.flatnonzero(valid_flags & sat_valid_mask[: len(cand)])
                        if valid_slots.size == 0:
                            continue
                        heur_order = valid_slots[np.argsort(heur_scores[valid_slots])[::-1]]
                        heur_rank_lookup = {int(slot): int(rank + 1) for rank, slot in enumerate(heur_order.tolist())}
                        chosen_sat_ids = [int(cand[idx]) for idx in chosen_slots]
                        heur_sat_ids = [int(cand[idx]) for idx in heur_slots]
                        jaccard = float(len(set(chosen_sat_ids) & set(heur_sat_ids)) / max(len(set(chosen_sat_ids) | set(heur_sat_ids)), 1))
                        exact_match = int(set(chosen_sat_ids) == set(heur_sat_ids))
                        ep_exact_match.append(float(exact_match))
                        ep_jaccard.append(jaccard)

                        chosen_env_ranks = [float(slot + 1) for slot in chosen_slots]
                        chosen_heur_ranks = [float(heur_rank_lookup.get(int(slot), 0.0)) for slot in chosen_slots if int(slot) in heur_rank_lookup]
                        chosen_distance = [float(np.asarray(rank_data["distance"], dtype=np.float64)[slot]) for slot in chosen_slots]
                        chosen_se = [float(np.asarray(rank_data["spectral_efficiency"], dtype=np.float64)[slot]) for slot in chosen_slots]
                        chosen_elev = [float(np.asarray(rank_data["elevation"], dtype=np.float64)[slot]) for slot in chosen_slots]
                        ep_policy_env_ranks.extend(chosen_env_ranks)
                        ep_policy_heur_ranks.extend(chosen_heur_ranks)
                        ep_policy_distance.extend(chosen_distance)
                        ep_policy_se.extend(chosen_se)
                        ep_policy_elevation.extend(chosen_elev)

                        selection_rows.append(
                            {
                                "mode": mode,
                                "episode": ep,
                                "seed": seed,
                                "step": step,
                                "uav": u,
                                "selected_slots": "|".join(str(x) for x in chosen_slots),
                                "heur_slots": "|".join(str(x) for x in heur_slots),
                                "selected_sat_ids": "|".join(str(x) for x in chosen_sat_ids),
                                "heur_sat_ids": "|".join(str(x) for x in heur_sat_ids),
                                "exact_match": exact_match,
                                "jaccard": jaccard,
                                "selected_env_rank_mean": _safe_mean(chosen_env_ranks),
                                "selected_heur_rank_mean": _safe_mean(chosen_heur_ranks),
                                "selected_distance_mean": _safe_mean(chosen_distance),
                                "selected_se_mean": _safe_mean(chosen_se),
                                "selected_elevation_mean": _safe_mean(chosen_elev),
                            }
                        )

                    actions = assemble_actions(cfg, env.agents, accel_actions, bw_alloc=heur_bw, sat_select_mask=sat_masks_orig)
                    obs, rewards, terms, truncs, _ = env.step(actions)
                    reward_sum += float(list(rewards.values())[0])
                    done = bool(list(terms.values())[0] or list(truncs.values())[0])
                    steps += 1
                    step += 1

                    assoc_dist = 0.0
                    if cfg.num_gu > 0 and hasattr(env, "last_association"):
                        assoc = np.asarray(env.last_association, dtype=np.int32)
                        mask = assoc >= 0
                        if np.any(mask):
                            gu_pos = env.gu_pos[mask]
                            u_idx = assoc[mask].astype(np.int32)
                            uav_pos = env.uav_pos[u_idx]
                            assoc_dist = float(np.mean(np.linalg.norm(gu_pos - uav_pos, axis=1)))
                    metric_sums["assoc_dist_mean"] += assoc_dist
                    metric_sums["connected_sat_dist_mean"] += float(getattr(env, "last_connected_sat_dist_mean", 0.0))

                    parts = getattr(env, "last_reward_parts", None)
                    if parts:
                        metric_sums["processed_ratio_eval"] += float(parts.get("processed_ratio_eval", 0.0))
                        metric_sums["drop_ratio_eval"] += float(parts.get("drop_ratio_eval", 0.0))
                        metric_sums["pre_backlog_steps_eval"] += float(parts.get("pre_backlog_steps_eval", 0.0))
                        metric_sums["throughput_backhaul_norm"] += float(parts.get("throughput_backhaul_norm", 0.0))
                        metric_sums["throughput_access_norm"] += float(parts.get("throughput_access_norm", 0.0))

                row = _step_metric_row(
                    mode=mode,
                    episode=ep,
                    seed=seed,
                    reward_sum=reward_sum,
                    steps=steps,
                    metric_sums=metric_sums,
                    extra={
                        "selected_env_rank_mean": _safe_mean(ep_policy_env_ranks),
                        "selected_heur_rank_mean": _safe_mean(ep_policy_heur_ranks),
                        "selected_distance_mean": _safe_mean(ep_policy_distance),
                        "selected_se_mean": _safe_mean(ep_policy_se),
                        "selected_elevation_mean": _safe_mean(ep_policy_elevation),
                        "exact_match_fraction": _safe_mean(ep_exact_match),
                        "mean_jaccard": _safe_mean(ep_jaccard),
                        "elapsed_sec": time.perf_counter() - t0,
                    },
                )
                summary_rows.append(row)
                episode_rows.append(row)

                step_rows.append(
                    {
                        "mode": mode,
                        "episode": ep,
                        "seed": seed,
                        "reward_sum": reward_sum,
                        "steps": steps,
                    }
                )

            mode_summary = {
                "mode": mode,
                "run_dir": str(run_dir),
                "update": int(args.update),
                "episodes": int(args.episodes),
                "episode_seed_base": int(args.episode_seed_base),
            }
            mode_summary.update(
                _summarize_row_metrics(
                    episode_rows,
                    exclude_keys={"mode", "episode", "episode_seed"},
                )
            )
            mode_summaries[mode] = mode_summary
        finally:
            env.close()

    summary_path = out_dir / "summary.json"
    episode_csv_path = out_dir / "episode_rows.csv"
    selection_csv_path = out_dir / "selection_rows.csv"
    steps_csv_path = out_dir / "step_rows.csv"

    with episode_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else ["mode", "episode", "episode_seed"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    with selection_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(selection_rows[0].keys()) if selection_rows else ["mode", "episode", "step", "uav"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selection_rows)

    with steps_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(step_rows[0].keys()) if step_rows else ["mode", "episode", "seed"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(step_rows)

    payload = {
        "meta": {
            "run_dir": str(run_dir.resolve()),
            "update": int(args.update),
            "episodes": int(args.episodes),
            "episode_seed_base": int(args.episode_seed_base),
            "modes": modes,
        },
        "mode_summaries": mode_summaries,
        "episode_csv": str(episode_csv_path.resolve()),
        "selection_csv": str(selection_csv_path.resolve()),
        "steps_csv": str(steps_csv_path.resolve()),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
