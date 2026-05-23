from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.baselines import queue_aware_bw_policy
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import _collate_dataclass
from sagin_marl.rl.structured_types import LocalBwState
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _safe_corr(x: list[float], y: list[float]) -> float | None:
    if len(x) <= 1 or len(y) <= 1 or len(x) != len(y):
        return None
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if float(np.std(xa)) <= 1.0e-12 or float(np.std(ya)) <= 1.0e-12:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _bucket_mean(counts: list[int], values: list[float]) -> list[dict[str, float]]:
    if len(counts) != len(values):
        raise ValueError("counts and values must have the same length")
    buckets: dict[int, list[float]] = {}
    for count, value in zip(counts, values):
        buckets.setdefault(int(count), []).append(float(value))
    rows: list[dict[str, float]] = []
    for count in sorted(buckets):
        arr = np.asarray(buckets[count], dtype=np.float64)
        rows.append(
            {
                "count": float(count),
                "n": float(arr.size),
                "mean": float(np.mean(arr)),
            }
        )
    return rows


def _bucket_corr(
    counts: list[int],
    x: list[float],
    y: list[float],
    *,
    min_n: int,
) -> list[dict[str, float | None]]:
    if len(counts) != len(x) or len(counts) != len(y):
        raise ValueError("counts/x/y length mismatch")
    buckets: dict[int, tuple[list[float], list[float]]] = {}
    for count, xv, yv in zip(counts, x, y):
        xs, ys = buckets.setdefault(int(count), ([], []))
        xs.append(float(xv))
        ys.append(float(yv))
    rows: list[dict[str, float | None]] = []
    for count in sorted(buckets):
        xs, ys = buckets[count]
        corr = _safe_corr(xs, ys) if len(xs) >= int(min_n) else None
        rows.append(
            {
                "count": float(count),
                "n": float(len(xs)),
                "corr": corr,
                "x_mean": _safe_mean(xs),
                "y_mean": _safe_mean(ys),
            }
        )
    return rows


def _weighted_bucket_corr_mean(rows: list[dict[str, float | None]]) -> float:
    weights: list[float] = []
    corrs: list[float] = []
    for row in rows:
        corr = row.get("corr")
        if corr is None:
            continue
        weights.append(float(row.get("n", 0.0)))
        corrs.append(float(corr))
    if not corrs:
        return 0.0
    w = np.asarray(weights, dtype=np.float64)
    c = np.asarray(corrs, dtype=np.float64)
    return float(np.sum(w * c) / np.sum(w))


def _normalize_weights(weights: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(weights, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if not np.any(valid):
        return out
    clipped = np.clip(np.asarray(weights, dtype=np.float32), 0.0, None) * valid.astype(np.float32, copy=False)
    denom = float(np.sum(clipped))
    if denom > 1.0e-9:
        out = clipped / denom
    else:
        out[valid] = 1.0 / float(np.sum(valid))
    return out.astype(np.float32, copy=False)


def _l1_align(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - 0.5 * np.sum(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))))


def _latent_mask_from_bw_state(local_state: LocalBwState) -> torch.Tensor:
    valid = (local_state.user_mask > 0.5) & (local_state.bw_valid_mask > 0.5)
    if valid.shape[-1] == 0:
        return valid
    latent = valid.clone()
    valid_count = valid.sum(dim=-1)
    ref_idx = torch.where(
        valid,
        torch.arange(valid.shape[-1], device=valid.device, dtype=torch.long).view(1, -1).expand_as(valid),
        torch.full_like(valid, -1, dtype=torch.long),
    ).amax(dim=-1)
    active_rows = torch.nonzero(valid_count > 1, as_tuple=False).flatten()
    if active_rows.numel() > 0:
        latent[active_rows, ref_idx[active_rows]] = False
    return latent


def _refresh_stage_obs_cache(driver: StructuredControlDriver) -> None:
    if driver._stage_assoc is None or driver._stage_candidates is None:
        raise RuntimeError("run_accel_stage must be called before refreshing stage obs cache")
    env = driver.env
    env._cached_assoc = driver._stage_assoc.copy()
    env._cached_candidates = [list(c) for c in driver._stage_candidates]
    if driver._stage_bw_valid_mask is not None:
        env._cached_bw_valid_mask = driver._stage_bw_valid_mask.copy()
    dummy_actions = env._dummy_actions()
    _, env._cached_eta = env._compute_access_rates(
        driver._stage_assoc,
        driver._stage_candidates,
        dummy_actions,
        record_exec=False,
    )
    if driver._stage_sat_pos is not None and driver._stage_sat_vel is not None and driver._stage_visible is not None:
        env._cache_sat_obs(driver._stage_sat_pos, driver._stage_sat_vel, driver._stage_visible)


def _current_obs_list(env: SaginParallelEnv) -> list[dict[str, np.ndarray]]:
    return [env._get_obs(i) for i in range(len(env.agents))]


def _load_actor(run_dir: Path, update: int, device: torch.device):
    cfg = load_config(str(run_dir / "config_source.yaml"))
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    actor_ckpt = run_dir / f"actor_u{int(update):04d}.pt"
    load_checkpoint_forgiving(bundle.actor, str(actor_ckpt), map_location=device, strict=True)
    bundle.actor.to(device).eval()
    return cfg, bundle.actor


def _build_bw_action_dict(
    driver: StructuredControlDriver,
    sat_action: np.ndarray,
    bw_action: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    env = driver.env
    cfg = env.cfg
    action_dict = driver._dummy_action_dict()
    for u, agent in enumerate(env.agents):
        action_dict[agent]["bw_alloc"] = np.asarray(bw_action[u], dtype=np.float32)
        sat_mask = np.zeros((cfg.sats_obs_max,), dtype=np.float32)
        visible = driver._stage_visible[u][: cfg.sats_obs_max] if driver._stage_visible is not None else []
        slot_lookup = {int(sat_idx): slot for slot, sat_idx in enumerate(visible)}
        for sat_idx in np.asarray(sat_action[u], dtype=np.int64).tolist():
            sat_idx = int(sat_idx)
            if sat_idx < 0:
                continue
            slot = slot_lookup.get(sat_idx)
            if slot is not None and slot < cfg.sats_obs_max:
                sat_mask[slot] = 1.0
        action_dict[agent]["sat_select_mask"] = sat_mask
    return action_dict


def _local_env_target(obs: dict[str, np.ndarray], assoc_bonus: float = 0.2) -> np.ndarray:
    valid_mask = np.asarray(obs["bw_valid_mask"] > 0.0, dtype=bool)
    users = np.asarray(obs["users"], dtype=np.float32)
    if not np.any(valid_mask):
        return np.zeros((users.shape[0],), dtype=np.float32)
    q = np.clip(users[:, 2], 0.0, None)
    eta = np.clip(users[:, 3], 0.0, None)
    prev = np.clip(users[:, 4], 0.0, 1.0)
    weights = q * (0.5 + eta) * (1.0 + float(assoc_bonus) * prev)
    return _normalize_weights(weights.astype(np.float32, copy=False), valid_mask)


def diagnose_update(
    run_dir: Path,
    update: int,
    *,
    episodes: int,
    episode_seed_base: int,
    deterministic: bool,
    min_bucket_n: int,
    device: torch.device,
) -> dict[str, Any]:
    cfg, actor = _load_actor(run_dir, int(update), device)
    env = make_structured_env(cfg, mode="script")
    valid_user_count: list[int] = []
    latent_dim_count: list[int] = []

    entropy_raw: list[float] = []
    entropy_per_dim_raw: list[float] = []
    log_scale_latent_mean: list[float] = []
    top1_allocation: list[float] = []

    l1_gap_vs_heuristic: list[float] = []
    align_gap_vs_env_target: list[float] = []
    qweighted_rate_gap_vs_heuristic: list[float] = []

    try:
        for ep in range(int(episodes)):
            env.reset(seed=int(episode_seed_base) + ep)
            driver = as_structured_driver(env)
            done = False
            while not done:
                z0 = driver.begin_step()
                accel_states = driver.build_local_accel_states(z0)
                accel_batch = _collate_dataclass(accel_states, device)
                with torch.inference_mode():
                    accel_out = actor.act_accel(accel_batch, deterministic=deterministic)
                z1 = driver.run_accel_stage(accel_out.action.detach().cpu().numpy())
                _refresh_stage_obs_cache(driver)
                obs_after_accel = _current_obs_list(env)

                sat_states = driver.build_sat_pair_candidates(z1)
                sat_batch = _collate_dataclass(sat_states, device)
                with torch.inference_mode():
                    sat_out = actor.act_sat_pair(sat_batch, deterministic=deterministic)
                sat_action = driver.decode_sat_pair_actions(sat_states, sat_out.subset_index.detach().cpu().tolist())
                z2 = driver.run_sat_stage(sat_action)

                bw_states = driver.build_bw_valid_context(z2)
                bw_batch = _collate_dataclass(bw_states, device)
                with torch.inference_mode():
                    bw_out = actor.act_bw(bw_batch, deterministic=deterministic)
                policy_bw = bw_out.action.detach().cpu().numpy()
                heur_bw = queue_aware_bw_policy(obs_after_accel, cfg)

                policy_actions = _build_bw_action_dict(driver, sat_action, policy_bw)
                heur_actions = _build_bw_action_dict(driver, sat_action, heur_bw)
                rates_policy, _ = env._compute_access_rates(
                    driver._stage_assoc,
                    driver._stage_candidates,
                    policy_actions,
                    record_exec=False,
                )
                rates_heur, _ = env._compute_access_rates(
                    driver._stage_assoc,
                    driver._stage_candidates,
                    heur_actions,
                    record_exec=False,
                )

                valid_mask_np = ((bw_batch.user_mask > 0.5) & (bw_batch.bw_valid_mask > 0.5)).detach().cpu().numpy()
                latent_mask_np = _latent_mask_from_bw_state(bw_batch).detach().cpu().numpy()
                entropy_raw_np = (
                    bw_out.entropy_raw.detach().cpu().numpy()
                    if bw_out.entropy_raw is not None
                    else bw_out.entropy.detach().cpu().numpy()
                )
                log_scale_np = bw_out.log_scale.detach().cpu().numpy()

                snapshot = driver.build_bw_stage_snapshot(z2)
                candidate_indices = np.asarray(snapshot.candidate_indices, dtype=np.int64)

                for u in range(int(cfg.num_uav)):
                    valid_mask = np.asarray(valid_mask_np[u], dtype=bool)
                    valid_count = int(np.sum(valid_mask))
                    if valid_count <= 0:
                        continue

                    latent_mask = np.asarray(latent_mask_np[u], dtype=bool)
                    latent_count = int(np.sum(latent_mask))
                    policy_local = _normalize_weights(np.asarray(policy_bw[u], dtype=np.float32), valid_mask)
                    heur_local = _normalize_weights(np.asarray(heur_bw[u], dtype=np.float32), valid_mask)
                    env_target = _local_env_target(obs_after_accel[u], assoc_bonus=0.2)

                    valid_user_count.append(valid_count)
                    latent_dim_count.append(latent_count)
                    ent_raw = float(entropy_raw_np[u])
                    entropy_raw.append(ent_raw)
                    entropy_per_dim_raw.append(ent_raw / float(max(latent_count, 1)) if latent_count > 0 else 0.0)
                    if latent_count > 0:
                        log_scale_latent_mean.append(float(np.mean(log_scale_np[u, latent_mask])))
                    else:
                        log_scale_latent_mean.append(0.0)
                    top1_allocation.append(float(np.max(policy_local[valid_mask])))

                    l1_gap_vs_heuristic.append(float(0.5 * np.sum(np.abs(policy_local - heur_local))))
                    align_gap_vs_env_target.append(_l1_align(policy_local, env_target) - _l1_align(heur_local, env_target))

                    assoc_idx = candidate_indices[u, valid_mask]
                    assoc_idx = assoc_idx[assoc_idx >= 0]
                    if assoc_idx.size > 0:
                        q_norm = np.asarray(env.gu_queue[assoc_idx], dtype=np.float32) / max(float(cfg.queue_max_gu), 1.0e-9)
                        q_weight = _normalize_weights(q_norm, np.ones_like(q_norm, dtype=bool))
                        qweighted_rate_gap_vs_heuristic.append(
                            float(np.sum(q_weight * rates_policy[assoc_idx]) - np.sum(q_weight * rates_heur[assoc_idx]))
                        )
                    else:
                        qweighted_rate_gap_vs_heuristic.append(0.0)

                step = driver.execute_stage_bw_and_step(policy_bw)
                done = bool(any(step.terminations.values()) or any(step.truncations.values()))
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

    summary = {
        "update": int(update),
        "episodes": int(episodes),
        "episode_seed_base": int(episode_seed_base),
        "deterministic": bool(deterministic),
        "bucket_min_n": int(min_bucket_n),
        "sample_count": int(len(valid_user_count)),
        "valid_user_count": _summarize([float(x) for x in valid_user_count]),
        "latent_dim_count": _summarize([float(x) for x in latent_dim_count]),
        "geometry": {
            "entropy_raw": _summarize(entropy_raw),
            "entropy_per_dim_raw": _summarize(entropy_per_dim_raw),
            "log_scale_latent_mean": _summarize(log_scale_latent_mean),
            "top1_allocation": _summarize(top1_allocation),
        },
        "gaps": {
            "l1_gap_vs_heuristic": _summarize(l1_gap_vs_heuristic),
            "align_gap_vs_env_target": _summarize(align_gap_vs_env_target),
            "qweighted_rate_gap_vs_heuristic": _summarize(qweighted_rate_gap_vs_heuristic),
        },
        "gap_by_valid_user_count": {
            "l1_gap_vs_heuristic": _bucket_mean(valid_user_count, l1_gap_vs_heuristic),
            "align_gap_vs_env_target": _bucket_mean(valid_user_count, align_gap_vs_env_target),
            "qweighted_rate_gap_vs_heuristic": _bucket_mean(valid_user_count, qweighted_rate_gap_vs_heuristic),
        },
        "overall_correlations": {
            "entropy_raw_vs_l1_gap": _safe_corr(entropy_raw, l1_gap_vs_heuristic),
            "entropy_raw_vs_align_gap": _safe_corr(entropy_raw, align_gap_vs_env_target),
            "entropy_per_dim_raw_vs_l1_gap": _safe_corr(entropy_per_dim_raw, l1_gap_vs_heuristic),
            "entropy_per_dim_raw_vs_align_gap": _safe_corr(entropy_per_dim_raw, align_gap_vs_env_target),
            "log_scale_latent_mean_vs_l1_gap": _safe_corr(log_scale_latent_mean, l1_gap_vs_heuristic),
            "log_scale_latent_mean_vs_align_gap": _safe_corr(log_scale_latent_mean, align_gap_vs_env_target),
            "top1_allocation_vs_l1_gap": _safe_corr(top1_allocation, l1_gap_vs_heuristic),
            "top1_allocation_vs_align_gap": _safe_corr(top1_allocation, align_gap_vs_env_target),
            "entropy_per_dim_raw_vs_rate_gap": _safe_corr(entropy_per_dim_raw, qweighted_rate_gap_vs_heuristic),
            "log_scale_latent_mean_vs_rate_gap": _safe_corr(log_scale_latent_mean, qweighted_rate_gap_vs_heuristic),
            "top1_allocation_vs_rate_gap": _safe_corr(top1_allocation, qweighted_rate_gap_vs_heuristic),
        },
        "within_valid_count_correlations": {},
    }

    within_pairs = {
        "entropy_raw_vs_l1_gap": (entropy_raw, l1_gap_vs_heuristic),
        "entropy_raw_vs_align_gap": (entropy_raw, align_gap_vs_env_target),
        "entropy_per_dim_raw_vs_l1_gap": (entropy_per_dim_raw, l1_gap_vs_heuristic),
        "entropy_per_dim_raw_vs_align_gap": (entropy_per_dim_raw, align_gap_vs_env_target),
        "log_scale_latent_mean_vs_l1_gap": (log_scale_latent_mean, l1_gap_vs_heuristic),
        "log_scale_latent_mean_vs_align_gap": (log_scale_latent_mean, align_gap_vs_env_target),
        "top1_allocation_vs_l1_gap": (top1_allocation, l1_gap_vs_heuristic),
        "top1_allocation_vs_align_gap": (top1_allocation, align_gap_vs_env_target),
        "entropy_per_dim_raw_vs_rate_gap": (entropy_per_dim_raw, qweighted_rate_gap_vs_heuristic),
        "log_scale_latent_mean_vs_rate_gap": (log_scale_latent_mean, qweighted_rate_gap_vs_heuristic),
        "top1_allocation_vs_rate_gap": (top1_allocation, qweighted_rate_gap_vs_heuristic),
    }
    for name, (xs, ys) in within_pairs.items():
        rows = _bucket_corr(valid_user_count, xs, ys, min_n=int(min_bucket_n))
        summary["within_valid_count_correlations"][name] = {
            "weighted_mean_corr": _weighted_bucket_corr_mean(rows),
            "rows": rows,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--updates", type=int, nargs="+", default=[50, 100, 150, 200])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-seed-base", type=int, default=73000)
    parser.add_argument("--policy-mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--bucket-min-n", type=int, default=20)
    parser.add_argument("--out-name", type=str, default="structured_bw_gap_geometry_diag.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "episodes": int(args.episodes),
        "episode_seed_base": int(args.episode_seed_base),
        "policy_mode": str(args.policy_mode),
        "bucket_min_n": int(args.bucket_min_n),
        "updates": {},
    }
    for update in args.updates:
        update_summary = diagnose_update(
            run_dir,
            int(update),
            episodes=int(args.episodes),
            episode_seed_base=int(args.episode_seed_base) + int(update) * 1000,
            deterministic=(args.policy_mode == "deterministic"),
            min_bucket_n=int(args.bucket_min_n),
            device=device,
        )
        summary["updates"][f"u{int(update):04d}"] = update_summary
        print(
            json.dumps(
                {
                    "update": int(update),
                    "l1_gap_mean": update_summary["gaps"]["l1_gap_vs_heuristic"]["mean"],
                    "align_gap_mean": update_summary["gaps"]["align_gap_vs_env_target"]["mean"],
                    "rate_gap_mean": update_summary["gaps"]["qweighted_rate_gap_vs_heuristic"]["mean"],
                    "bucket_corr_entropy_per_dim_vs_align_gap": update_summary["within_valid_count_correlations"][
                        "entropy_per_dim_raw_vs_align_gap"
                    ]["weighted_mean_corr"],
                    "bucket_corr_log_scale_vs_align_gap": update_summary["within_valid_count_correlations"][
                        "log_scale_latent_mean_vs_align_gap"
                    ]["weighted_mean_corr"],
                    "bucket_corr_top1_vs_align_gap": update_summary["within_valid_count_correlations"][
                        "top1_allocation_vs_align_gap"
                    ]["weighted_mean_corr"],
                }
            )
        )

    out_path = run_dir / str(args.out_name)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
