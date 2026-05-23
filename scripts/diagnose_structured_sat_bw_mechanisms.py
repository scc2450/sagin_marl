from __future__ import annotations

import argparse
import json
import math
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
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import _collate_dataclass
from sagin_marl.rl.structured_types import LocalBwState, LocalSatState
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _safe_corr(x: list[float], y: list[float]) -> float:
    if len(x) <= 1 or len(y) <= 1 or len(x) != len(y):
        return 0.0
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if float(np.std(xa)) <= 1.0e-12 or float(np.std(ya)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def _rankdata_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _safe_spearman_desc(pred: np.ndarray, truth: np.ndarray) -> float:
    if pred.size <= 1 or truth.size <= 1 or pred.size != truth.size:
        return 0.0
    return _safe_corr(_rankdata_desc(pred), _rankdata_desc(truth))


def _argmax_rank01(pred: np.ndarray, truth: np.ndarray) -> float:
    n = int(pred.size)
    if n <= 1:
        return 0.0
    chosen = int(np.argmax(pred))
    order = np.argsort(-truth, kind="mergesort")
    rank = int(np.flatnonzero(order == chosen)[0])
    return float(rank) / float(max(n - 1, 1))


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
    for c, v in zip(counts, values):
        buckets.setdefault(int(c), []).append(float(v))
    rows: list[dict[str, float]] = []
    for key in sorted(buckets):
        arr = buckets[key]
        rows.append({"count": float(key), "n": float(len(arr)), "mean": _safe_mean(arr)})
    return rows


def _load_actor(run_dir: Path, update: int, device: torch.device):
    cfg = load_config(str(run_dir / "config_source.yaml"))
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    actor_ckpt = run_dir / f"actor_u{int(update):04d}.pt"
    load_checkpoint_forgiving(bundle.actor, str(actor_ckpt), map_location=device, strict=True)
    bundle.actor.to(device).eval()
    return cfg, bundle.actor


def _local_subset_quality(local_state: LocalSatState) -> torch.Tensor:
    subset_members = local_state.subset_members
    safe_members = subset_members.clamp_min(0)
    batch_size, subset_count, member_slots = safe_members.shape
    edge_dim = int(local_state.sat_edges.shape[-1])
    expanded_edges = local_state.sat_edges.unsqueeze(1).expand(-1, subset_count, -1, -1)
    gather_index = safe_members.unsqueeze(-1).expand(-1, -1, -1, edge_dim)
    gathered_edges = torch.gather(expanded_edges, 2, gather_index)
    member_mask = subset_members >= 0
    se = gathered_edges[..., 7]
    projected_bw = gathered_edges[..., 10]
    return (se * projected_bw * member_mask.to(se.dtype)).sum(dim=-1)


def _latent_mask_from_bw_state(local_state: LocalBwState) -> torch.Tensor:
    valid = (local_state.user_mask > 0.5) & (local_state.bw_valid_mask > 0.5)
    if valid.shape[-1] == 0:
        return valid
    latent = valid.clone()
    valid_counts = valid.sum(dim=-1)
    ref_idx = torch.where(
        valid,
        torch.arange(valid.shape[-1], device=valid.device, dtype=torch.long).view(1, -1).expand_as(valid),
        torch.full_like(valid, -1, dtype=torch.long),
    ).amax(dim=-1)
    active = torch.nonzero(valid_counts > 1, as_tuple=False).flatten()
    if active.numel() > 0:
        latent[active, ref_idx[active]] = False
    return latent


def diagnose_update(
    run_dir: Path,
    update: int,
    *,
    episodes: int,
    episode_seed_base: int,
    deterministic: bool,
    device: torch.device,
) -> dict[str, Any]:
    cfg, actor = _load_actor(run_dir, int(update), device)
    env = make_structured_env(cfg, mode="script")
    sat_entropy: list[float] = []
    sat_entropy_norm: list[float] = []
    sat_valid_subset_count: list[int] = []
    sat_quality_spearman: list[float] = []
    sat_argmax_rank01: list[float] = []
    sat_top1_margin: list[float] = []

    bw_entropy_raw: list[float] = []
    bw_entropy_objective: list[float] = []
    bw_entropy_per_dim_raw: list[float] = []
    bw_valid_user_count: list[int] = []
    bw_latent_dim_count: list[int] = []
    bw_log_scale_latent_mean: list[float] = []
    bw_top1_alloc: list[float] = []

    try:
        for ep in range(int(episodes)):
            seed = int(episode_seed_base) + ep
            env.reset(seed=seed)
            driver = as_structured_driver(env)
            done = False
            while not done:
                z0 = driver.begin_step()
                accel_states = driver.build_local_accel_states(z0)
                accel_batch = _collate_dataclass(accel_states, device)
                with torch.no_grad():
                    accel_out = actor.act_accel(accel_batch, deterministic=deterministic)
                z1 = driver.run_accel_stage(accel_out.action.cpu().numpy())

                sat_states = driver.build_sat_pair_candidates(z1)
                sat_batch = _collate_dataclass(sat_states, device)
                with torch.no_grad():
                    sat_out = actor.act_sat_pair(sat_batch, deterministic=deterministic)
                    subset_quality = _local_subset_quality(sat_batch).cpu().numpy()
                subset_mask = (sat_batch.subset_mask > 0.5).cpu().numpy()
                sat_logits = sat_out.logits.detach().cpu().numpy()
                sat_entropy_np = sat_out.entropy.detach().cpu().numpy()
                for row_idx in range(int(sat_logits.shape[0])):
                    valid = subset_mask[row_idx]
                    count = int(np.sum(valid))
                    sat_valid_subset_count.append(count)
                    ent = float(sat_entropy_np[row_idx])
                    sat_entropy.append(ent)
                    if count > 1:
                        sat_entropy_norm.append(ent / math.log(float(count)))
                    else:
                        sat_entropy_norm.append(0.0)
                    if count <= 0:
                        sat_quality_spearman.append(0.0)
                        sat_argmax_rank01.append(0.0)
                        sat_top1_margin.append(0.0)
                        continue
                    row_logits = sat_logits[row_idx, valid]
                    row_quality = subset_quality[row_idx, valid]
                    sat_quality_spearman.append(_safe_spearman_desc(row_logits, row_quality))
                    sat_argmax_rank01.append(_argmax_rank01(row_logits, row_quality))
                    if row_logits.size >= 2:
                        top2 = np.sort(row_logits)[-2:]
                        sat_top1_margin.append(float(top2[-1] - top2[-2]))
                    else:
                        sat_top1_margin.append(0.0)

                sat_action = driver.decode_sat_pair_actions(sat_states, sat_out.pair_index.cpu().tolist())
                z2 = driver.run_sat_stage(sat_action)

                bw_states = driver.build_bw_valid_context(z2)
                bw_batch = _collate_dataclass(bw_states, device)
                with torch.no_grad():
                    bw_out = actor.act_bw(bw_batch, deterministic=deterministic)
                valid_user_mask = ((bw_batch.user_mask > 0.5) & (bw_batch.bw_valid_mask > 0.5)).cpu().numpy()
                latent_mask = _latent_mask_from_bw_state(bw_batch).cpu().numpy()
                bw_entropy_raw_np = (
                    bw_out.entropy_raw.detach().cpu().numpy()
                    if bw_out.entropy_raw is not None
                    else bw_out.entropy.detach().cpu().numpy()
                )
                bw_entropy_objective_np = bw_out.entropy.detach().cpu().numpy()
                bw_log_scale_np = bw_out.log_scale.detach().cpu().numpy()
                bw_action_np = bw_out.action.detach().cpu().numpy()
                for row_idx in range(int(bw_entropy_objective_np.shape[0])):
                    valid_count = int(np.sum(valid_user_mask[row_idx]))
                    latent_count = int(np.sum(latent_mask[row_idx]))
                    bw_valid_user_count.append(valid_count)
                    bw_latent_dim_count.append(latent_count)
                    ent_raw = float(bw_entropy_raw_np[row_idx])
                    ent_objective = float(bw_entropy_objective_np[row_idx])
                    bw_entropy_raw.append(ent_raw)
                    bw_entropy_objective.append(ent_objective)
                    bw_entropy_per_dim_raw.append(
                        ent_raw / float(max(latent_count, 1)) if latent_count > 0 else 0.0
                    )
                    if latent_count > 0:
                        bw_log_scale_latent_mean.append(float(np.mean(bw_log_scale_np[row_idx, latent_mask[row_idx]])))
                    else:
                        bw_log_scale_latent_mean.append(0.0)
                    if valid_count > 0:
                        bw_top1_alloc.append(float(np.max(bw_action_np[row_idx, valid_user_mask[row_idx]])))
                    else:
                        bw_top1_alloc.append(0.0)

                step = driver.execute_stage_bw_and_step(bw_out.action.cpu().numpy())
                done = bool(any(step.terminations.values()) or any(step.truncations.values()))
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

    sat_summary = {
        "sample_count": int(len(sat_entropy)),
        "valid_subset_count": _summarize([float(x) for x in sat_valid_subset_count]),
        "entropy_raw": _summarize(sat_entropy),
        "entropy_norm": _summarize(sat_entropy_norm),
        "entropy_vs_valid_subset_count_corr": _safe_corr([float(x) for x in sat_valid_subset_count], sat_entropy),
        "entropy_norm_vs_valid_subset_count_corr": _safe_corr([float(x) for x in sat_valid_subset_count], sat_entropy_norm),
        "quality_alignment": {
            "spearman_mean": _safe_mean(sat_quality_spearman),
            "argmax_rank01_mean": _safe_mean(sat_argmax_rank01),
            "top1_margin_mean": _safe_mean(sat_top1_margin),
        },
        "entropy_by_valid_subset_count": _bucket_mean(sat_valid_subset_count, sat_entropy),
        "entropy_norm_by_valid_subset_count": _bucket_mean(sat_valid_subset_count, sat_entropy_norm),
    }
    bw_summary = {
        "sample_count": int(len(bw_entropy_raw)),
        "valid_user_count": _summarize([float(x) for x in bw_valid_user_count]),
        "latent_dim_count": _summarize([float(x) for x in bw_latent_dim_count]),
        "entropy_raw": _summarize(bw_entropy_raw),
        "entropy_objective": _summarize(bw_entropy_objective),
        "entropy_per_dim_raw": _summarize(bw_entropy_per_dim_raw),
        "log_scale_latent_mean": _summarize(bw_log_scale_latent_mean),
        "top1_allocation": _summarize(bw_top1_alloc),
        "entropy_raw_vs_valid_user_count_corr": _safe_corr([float(x) for x in bw_valid_user_count], bw_entropy_raw),
        "entropy_objective_vs_valid_user_count_corr": _safe_corr(
            [float(x) for x in bw_valid_user_count], bw_entropy_objective
        ),
        "entropy_per_dim_raw_vs_valid_user_count_corr": _safe_corr(
            [float(x) for x in bw_valid_user_count], bw_entropy_per_dim_raw
        ),
        "entropy_per_dim_raw_vs_log_scale_corr": _safe_corr(bw_log_scale_latent_mean, bw_entropy_per_dim_raw),
        "valid_user_count_vs_log_scale_corr": _safe_corr([float(x) for x in bw_valid_user_count], bw_log_scale_latent_mean),
        "top1_alloc_vs_entropy_per_dim_raw_corr": _safe_corr(bw_top1_alloc, bw_entropy_per_dim_raw),
        "entropy_raw_by_valid_user_count": _bucket_mean(bw_valid_user_count, bw_entropy_raw),
        "entropy_objective_by_valid_user_count": _bucket_mean(bw_valid_user_count, bw_entropy_objective),
        "entropy_per_dim_raw_by_valid_user_count": _bucket_mean(bw_valid_user_count, bw_entropy_per_dim_raw),
        "log_scale_by_valid_user_count": _bucket_mean(bw_valid_user_count, bw_log_scale_latent_mean),
        "top1_alloc_by_valid_user_count": _bucket_mean(bw_valid_user_count, bw_top1_alloc),
    }
    return {
        "update": int(update),
        "episodes": int(episodes),
        "episode_seed_base": int(episode_seed_base),
        "deterministic": bool(deterministic),
        "sat": sat_summary,
        "bw": bw_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--updates", type=int, nargs="+", default=[100, 150, 200])
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--episode-seed-base", type=int, default=52000)
    parser.add_argument("--policy-mode", choices=["deterministic", "stochastic"], default="stochastic")
    parser.add_argument("--out-name", type=str, default="structured_sat_bw_mechanism_diag.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "episodes": int(args.episodes),
        "episode_seed_base": int(args.episode_seed_base),
        "policy_mode": str(args.policy_mode),
        "updates": {},
    }
    for update in args.updates:
        result = diagnose_update(
            run_dir,
            int(update),
            episodes=int(args.episodes),
            episode_seed_base=int(args.episode_seed_base),
            deterministic=(args.policy_mode == "deterministic"),
            device=device,
        )
        summary["updates"][f"u{int(update):04d}"] = result
        print(
            json.dumps(
                {
                    "update": int(update),
                    "sat_entropy_norm_mean": result["sat"]["entropy_norm"]["mean"],
                    "sat_quality_spearman_mean": result["sat"]["quality_alignment"]["spearman_mean"],
                    "bw_entropy_objective_mean": result["bw"]["entropy_objective"]["mean"],
                    "bw_entropy_per_dim_raw_mean": result["bw"]["entropy_per_dim_raw"]["mean"],
                    "bw_log_scale_latent_mean": result["bw"]["log_scale_latent_mean"]["mean"],
                },
                ensure_ascii=False,
            )
        )
    out_path = run_dir / str(args.out_name)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
