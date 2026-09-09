from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(ROOT, "sagin_marl")):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        raise RuntimeError("Could not locate repository root.")
    ROOT = parent
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.structured_actor import _attend
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
    buckets: dict[int, list[float]] = {}
    for count, value in zip(counts, values):
        buckets.setdefault(int(count), []).append(float(value))
    rows: list[dict[str, float]] = []
    for count in sorted(buckets):
        arr = np.asarray(buckets[count], dtype=np.float64)
        rows.append({"count": float(count), "n": float(arr.size), "mean": float(np.mean(arr))})
    return rows


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


def _load_actor(run_dir: Path, update: int, device: torch.device):
    cfg = load_config(str(run_dir / "config_source.yaml"))
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    actor_ckpt = run_dir / f"actor_u{int(update):04d}.pt"
    load_checkpoint_forgiving(bundle.actor, str(actor_ckpt), map_location=device, strict=True)
    bundle.actor.to(device).eval()
    return cfg, bundle.actor


def _latent_mask_from_bw_state(local_state: LocalBwState) -> tuple[torch.Tensor, torch.Tensor]:
    valid = (local_state.user_mask > 0.5) & (local_state.bw_valid_mask > 0.5)
    latent = valid.clone()
    if valid.shape[-1] > 0:
        valid_count = valid.sum(dim=-1)
        ref_idx = torch.where(
            valid,
            torch.arange(valid.shape[-1], device=valid.device, dtype=torch.long).view(1, -1).expand_as(valid),
            torch.full_like(valid, -1, dtype=torch.long),
        ).amax(dim=-1)
        active_rows = torch.nonzero(valid_count > 1, as_tuple=False).flatten()
        if active_rows.numel() > 0:
            latent[active_rows, ref_idx[active_rows]] = False
    return valid, latent


def _dispersion_ratio(x: np.ndarray) -> float:
    if x.ndim != 2 or x.shape[0] <= 1:
        return 0.0
    center = np.mean(x, axis=0, keepdims=True)
    spread = np.sqrt(np.mean(np.sum((x - center) ** 2, axis=-1)))
    base = np.mean(np.linalg.norm(x, axis=-1))
    return float(spread / max(base, 1.0e-8))


def _effective_logits(loc_row: np.ndarray, valid_row: np.ndarray, latent_row: np.ndarray) -> np.ndarray:
    logits = np.zeros_like(loc_row, dtype=np.float32)
    logits[np.asarray(latent_row, dtype=bool)] = np.asarray(loc_row[np.asarray(latent_row, dtype=bool)], dtype=np.float32)
    return logits[np.asarray(valid_row, dtype=bool)]


def _bw_intermediates(bw_policy, local_state: LocalBwState) -> dict[str, torch.Tensor]:
    valid_mask = (local_state.user_mask > 0.5) & (local_state.bw_valid_mask > 0.5)
    ego_0 = bw_policy.ego_encoder(local_state.ego_uav_after_sat)
    sat_0 = bw_policy.sat_encoder(torch.cat([local_state.sat_nodes, local_state.sat_edges], dim=-1))
    user_raw = torch.cat([local_state.user_nodes, local_state.user_edges], dim=-1)
    user_0 = bw_policy.user_encoder(user_raw)

    sat_ctx_1 = _attend(ego_0, sat_0, local_state.sat_mask)
    query_1 = bw_policy.query_proj_1(torch.cat([ego_0, sat_ctx_1], dim=-1))
    user_ctx_1 = _attend(query_1, user_0, valid_mask)

    sat_1 = sat_0 + bw_policy.sat_refine(torch.cat([sat_0, ego_0.unsqueeze(1).expand_as(sat_0)], dim=-1))
    user_1 = user_0 + bw_policy.user_refine(
        torch.cat(
            [
                user_0,
                ego_0.unsqueeze(1).expand_as(user_0),
                sat_ctx_1.unsqueeze(1).expand_as(user_0),
                user_ctx_1.unsqueeze(1).expand_as(user_0),
            ],
            dim=-1,
        )
    )

    query_2 = bw_policy.query_proj_2(torch.cat([ego_0, sat_ctx_1, user_ctx_1], dim=-1))
    sat_ctx_2 = _attend(query_2, sat_1, local_state.sat_mask)
    user_ctx_2 = _attend(query_2, user_1, valid_mask)
    fused = bw_policy.user_fusion(
        torch.cat(
            [
                ego_0.unsqueeze(1).expand_as(user_1),
                sat_ctx_2.unsqueeze(1).expand_as(user_1),
                user_ctx_2.unsqueeze(1).expand_as(user_1),
                user_1,
            ],
            dim=-1,
        )
    )
    loc_fused = bw_policy.loc_head(fused).squeeze(-1)
    loc_user0 = bw_policy.loc_head(user_0).squeeze(-1)
    loc_user1 = bw_policy.loc_head(user_1).squeeze(-1)
    loc_readout = str(getattr(bw_policy, "loc_readout", "fused")).strip().lower()
    if loc_readout == "user_only":
        loc_source = user_1
        loc_actual = loc_user1
    elif loc_readout == "user0":
        loc_source = user_0
        loc_actual = loc_user0
    else:
        loc_source = fused
        loc_actual = loc_fused
    return {
        "user_raw": user_raw,
        "user_0": user_0,
        "user_1": user_1,
        "fused": fused,
        "loc_source": loc_source,
        "loc_actual": loc_actual,
        "loc_fused": loc_fused,
        "loc_user0": loc_user0,
        "loc_user1": loc_user1,
    }


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
    valid_user_count: list[int] = []
    raw_dispersion: list[float] = []
    user0_dispersion: list[float] = []
    user1_dispersion: list[float] = []
    fused_dispersion: list[float] = []
    loc_source_dispersion: list[float] = []
    actual_logit_range: list[float] = []
    actual_logit_std: list[float] = []
    fused_readout_logit_range: list[float] = []
    fused_readout_logit_std: list[float] = []
    user0_readout_logit_range: list[float] = []
    user0_readout_logit_std: list[float] = []
    user1_readout_logit_range: list[float] = []
    user1_readout_logit_std: list[float] = []

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

                sat_states = driver.build_sat_pair_candidates(z1)
                sat_batch = _collate_dataclass(sat_states, device)
                with torch.inference_mode():
                    sat_out = actor.act_sat_pair(sat_batch, deterministic=deterministic)
                sat_action = driver.decode_sat_pair_actions(sat_states, sat_out.subset_index.detach().cpu().tolist())
                z2 = driver.run_sat_stage(sat_action)

                bw_states = driver.build_bw_valid_context(z2)
                bw_batch = _collate_dataclass(bw_states, device)
                with torch.inference_mode():
                    bw_repr = _bw_intermediates(actor.bw_policy, bw_batch)
                    bw_out = actor.act_bw(bw_batch, deterministic=deterministic)

                valid_mask, latent_mask = _latent_mask_from_bw_state(bw_batch)
                valid_mask_np = valid_mask.detach().cpu().numpy()
                latent_mask_np = latent_mask.detach().cpu().numpy()
                user_raw_np = bw_repr["user_raw"].detach().cpu().numpy()
                user0_np = bw_repr["user_0"].detach().cpu().numpy()
                user1_np = bw_repr["user_1"].detach().cpu().numpy()
                fused_np = bw_repr["fused"].detach().cpu().numpy()
                loc_source_np = bw_repr["loc_source"].detach().cpu().numpy()
                loc_actual_np = bw_repr["loc_actual"].detach().cpu().numpy()
                loc_fused_np = bw_repr["loc_fused"].detach().cpu().numpy()
                loc_user0_np = bw_repr["loc_user0"].detach().cpu().numpy()
                loc_user1_np = bw_repr["loc_user1"].detach().cpu().numpy()

                for u in range(int(cfg.num_uav)):
                    valid = np.asarray(valid_mask_np[u], dtype=bool)
                    valid_count = int(np.sum(valid))
                    if valid_count <= 0:
                        continue
                    latent = np.asarray(latent_mask_np[u], dtype=bool)
                    valid_user_count.append(valid_count)
                    raw_dispersion.append(_dispersion_ratio(user_raw_np[u, valid, :]))
                    user0_dispersion.append(_dispersion_ratio(user0_np[u, valid, :]))
                    user1_dispersion.append(_dispersion_ratio(user1_np[u, valid, :]))
                    fused_dispersion.append(_dispersion_ratio(fused_np[u, valid, :]))
                    loc_source_dispersion.append(_dispersion_ratio(loc_source_np[u, valid, :]))
                    actual_logits_valid = _effective_logits(loc_actual_np[u], valid, latent)
                    fused_logits_valid = _effective_logits(loc_fused_np[u], valid, latent)
                    user0_logits_valid = _effective_logits(loc_user0_np[u], valid, latent)
                    user1_logits_valid = _effective_logits(loc_user1_np[u], valid, latent)
                    actual_logit_range.append(float(np.max(actual_logits_valid) - np.min(actual_logits_valid)))
                    actual_logit_std.append(float(np.std(actual_logits_valid)))
                    fused_readout_logit_range.append(float(np.max(fused_logits_valid) - np.min(fused_logits_valid)))
                    fused_readout_logit_std.append(float(np.std(fused_logits_valid)))
                    user0_readout_logit_range.append(float(np.max(user0_logits_valid) - np.min(user0_logits_valid)))
                    user0_readout_logit_std.append(float(np.std(user0_logits_valid)))
                    user1_readout_logit_range.append(float(np.max(user1_logits_valid) - np.min(user1_logits_valid)))
                    user1_readout_logit_std.append(float(np.std(user1_logits_valid)))

                step = driver.execute_stage_bw_and_step(bw_out.action.detach().cpu().numpy())
                done = bool(any(step.terminations.values()) or any(step.truncations.values()))
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

    return {
        "update": int(update),
        "episodes": int(episodes),
        "episode_seed_base": int(episode_seed_base),
        "deterministic": bool(deterministic),
        "loc_readout": str(getattr(actor.bw_policy, "loc_readout", "fused")),
        "sample_count": int(len(valid_user_count)),
        "valid_user_count": _summarize([float(x) for x in valid_user_count]),
        "raw_dispersion": _summarize(raw_dispersion),
        "user0_dispersion": _summarize(user0_dispersion),
        "user1_dispersion": _summarize(user1_dispersion),
        "fused_dispersion": _summarize(fused_dispersion),
        "loc_source_dispersion": _summarize(loc_source_dispersion),
        "effective_logit_range": _summarize(actual_logit_range),
        "effective_logit_std": _summarize(actual_logit_std),
        "effective_logit_range_fused_readout": _summarize(fused_readout_logit_range),
        "effective_logit_std_fused_readout": _summarize(fused_readout_logit_std),
        "effective_logit_range_user0_readout": _summarize(user0_readout_logit_range),
        "effective_logit_std_user0_readout": _summarize(user0_readout_logit_std),
        "effective_logit_range_user1_readout": _summarize(user1_readout_logit_range),
        "effective_logit_std_user1_readout": _summarize(user1_readout_logit_std),
        "dispersion_chain_means": {
            "raw": _safe_mean(raw_dispersion),
            "user0": _safe_mean(user0_dispersion),
            "user1": _safe_mean(user1_dispersion),
            "fused": _safe_mean(fused_dispersion),
            "loc_source": _safe_mean(loc_source_dispersion),
            "logit_range": _safe_mean(actual_logit_range),
            "logit_std": _safe_mean(actual_logit_std),
            "logit_range_fused_readout": _safe_mean(fused_readout_logit_range),
            "logit_range_user0_readout": _safe_mean(user0_readout_logit_range),
            "logit_range_user1_readout": _safe_mean(user1_readout_logit_range),
        },
        "corr": {
            "raw_to_logit_range": _safe_corr(raw_dispersion, actual_logit_range),
            "user0_to_logit_range": _safe_corr(user0_dispersion, actual_logit_range),
            "user1_to_logit_range": _safe_corr(user1_dispersion, actual_logit_range),
            "fused_to_logit_range": _safe_corr(fused_dispersion, actual_logit_range),
            "loc_source_to_logit_range": _safe_corr(loc_source_dispersion, actual_logit_range),
        },
        "raw_by_valid_user_count": _bucket_mean(valid_user_count, raw_dispersion),
        "user0_by_valid_user_count": _bucket_mean(valid_user_count, user0_dispersion),
        "user1_by_valid_user_count": _bucket_mean(valid_user_count, user1_dispersion),
        "fused_by_valid_user_count": _bucket_mean(valid_user_count, fused_dispersion),
        "loc_source_by_valid_user_count": _bucket_mean(valid_user_count, loc_source_dispersion),
        "logit_range_by_valid_user_count": _bucket_mean(valid_user_count, actual_logit_range),
        "logit_range_fused_readout_by_valid_user_count": _bucket_mean(valid_user_count, fused_readout_logit_range),
        "logit_range_user0_readout_by_valid_user_count": _bucket_mean(valid_user_count, user0_readout_logit_range),
        "logit_range_user1_readout_by_valid_user_count": _bucket_mean(valid_user_count, user1_readout_logit_range),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--updates", type=int, nargs="+", default=[50, 100, 150, 200])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-seed-base", type=int, default=81000)
    parser.add_argument("--policy-mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--out-name", type=str, default="structured_bw_representation_probe.json")
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
        update_summary = diagnose_update(
            run_dir,
            int(update),
            episodes=int(args.episodes),
            episode_seed_base=int(args.episode_seed_base) + int(update) * 1000,
            deterministic=(args.policy_mode == "deterministic"),
            device=device,
        )
        summary["updates"][f"u{int(update):04d}"] = update_summary
        print(
            json.dumps(
                {
                    "update": int(update),
                    "loc_readout": update_summary["loc_readout"],
                    "raw_disp": update_summary["dispersion_chain_means"]["raw"],
                    "user0_disp": update_summary["dispersion_chain_means"]["user0"],
                    "user1_disp": update_summary["dispersion_chain_means"]["user1"],
                    "fused_disp": update_summary["dispersion_chain_means"]["fused"],
                    "loc_source_disp": update_summary["dispersion_chain_means"]["loc_source"],
                    "logit_range": update_summary["dispersion_chain_means"]["logit_range"],
                    "fused_readout_logit_range": update_summary["dispersion_chain_means"]["logit_range_fused_readout"],
                    "user0_readout_logit_range": update_summary["dispersion_chain_means"]["logit_range_user0_readout"],
                    "user1_readout_logit_range": update_summary["dispersion_chain_means"]["logit_range_user1_readout"],
                }
            )
        )

    out_path = run_dir / str(args.out_name)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
