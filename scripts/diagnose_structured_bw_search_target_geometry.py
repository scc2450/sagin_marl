from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import yaml

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_stage_builders import build_local_bw_states_from_snapshot
from sagin_marl.rl.structured_mappo import _to_device_dataclass
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_torch_device(device_arg: str) -> torch.device:
    requested = str(device_arg).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return torch.device(device_arg)


def _save_config(run_dir: Path, cfg, config_path: str, extra: dict[str, Any]) -> None:
    data = asdict(cfg)
    data["_config_source"] = config_path
    data["_target_geometry_args"] = extra
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    src = Path(config_path)
    if src.is_file():
        (run_dir / "config_source.yaml").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def _sample_wide_dirichlet_actions(
    det_mean: np.ndarray,
    valid_mask: np.ndarray,
    *,
    num_samples: int,
    concentration: float,
) -> list[np.ndarray]:
    mean = np.asarray(det_mean, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    conc = max(float(concentration), 1.0e-3)
    actions: list[np.ndarray] = []
    for _ in range(max(int(num_samples), 0)):
        out = np.zeros_like(mean, dtype=np.float32)
        for u in range(mask.shape[0]):
            valid = np.flatnonzero(mask[u])
            if valid.size == 0:
                continue
            row_mean = np.asarray(mean[u, valid], dtype=np.float64)
            row_mean = np.clip(row_mean, 1.0e-6, None)
            row_mean = row_mean / row_mean.sum()
            alpha = np.clip(row_mean * conc, 1.0e-3, None)
            out[u, valid] = np.random.dirichlet(alpha).astype(np.float32)
        actions.append(out)
    return actions


def _snapshot_to_local_bw_state(snapshot: Any, device: torch.device):
    local_state = build_local_bw_states_from_snapshot(snapshot)[0]
    return _to_device_dataclass(local_state, device)


def _collect_bw_snapshots(
    cfg,
    actor,
    *,
    device: torch.device,
    num_states: int,
    seed: int,
    collect_deterministic: bool,
) -> list[dict[str, Any]]:
    env = make_structured_env(cfg, mode="script")
    driver = as_structured_driver(env)
    snapshots: list[dict[str, Any]] = []
    try:
        env.reset(seed=int(seed))
        episode_idx = 0
        while len(snapshots) < int(num_states):
            driver.begin_step()
            accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
            _ = driver.run_accel_stage(accel_zero)
            select_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 1)
            bw_world = driver.run_sat_stage(np.full((cfg.num_uav, select_k), -1, dtype=np.int64))
            bw_snapshot = driver.build_bw_stage_snapshot(bw_world)
            snapshot_state = driver.export_bw_stage_state()
            local_state = _snapshot_to_local_bw_state(bw_snapshot, device)
            snapshots.append({"snapshot_state": snapshot_state, "local_state": local_state})
            with torch.no_grad():
                out = actor.act_bw(local_state, deterministic=bool(collect_deterministic))
            step_result, _ = driver.execute_stage_bw_and_prepare_next_accel(
                np.asarray(out.action.detach().cpu().numpy(), dtype=np.float32)
            )
            done = bool(list(step_result.terminations.values())[0] or list(step_result.truncations.values())[0])
            if done and len(snapshots) < int(num_states):
                episode_idx += 1
                env.reset(seed=int(seed) + episode_idx)
                driver = as_structured_driver(env)
        return snapshots[: int(num_states)]
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def _run_bw_candidate_to_episode_end(
    cfg,
    actor,
    *,
    snapshot_state: dict[str, Any],
    first_action: np.ndarray,
    device: torch.device,
    follow_deterministic: bool,
) -> float:
    env = make_structured_env(cfg, mode="script")
    driver = as_structured_driver(env)
    total_reward = 0.0
    try:
        driver.load_bw_stage_state(snapshot_state)
        action = np.asarray(first_action, dtype=np.float32)
        while True:
            step_result, _ = driver.execute_stage_bw_and_prepare_next_accel(action)
            total_reward += float(next(iter(step_result.rewards.values())))
            done = bool(list(step_result.terminations.values())[0] or list(step_result.truncations.values())[0])
            if done:
                break
            driver.begin_step()
            accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
            _ = driver.run_accel_stage(accel_zero)
            select_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 1)
            bw_world = driver.run_sat_stage(np.full((cfg.num_uav, select_k), -1, dtype=np.int64))
            snapshot = driver.build_bw_stage_snapshot(bw_world)
            local_state = _to_device_dataclass(build_local_bw_states_from_snapshot(snapshot)[0], device)
            with torch.no_grad():
                out = actor.act_bw(local_state, deterministic=bool(follow_deterministic))
            action = np.asarray(out.action.detach().cpu().numpy(), dtype=np.float32)
        return float(total_reward)
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def _feature_vector(local_state) -> np.ndarray:
    user_nodes = np.asarray(local_state.user_nodes.detach().cpu().numpy(), dtype=np.float32)[0]
    user_edges = np.asarray(local_state.user_edges.detach().cpu().numpy(), dtype=np.float32)[0]
    valid = np.asarray(
        ((local_state.user_mask > 0.5) & (local_state.bw_valid_mask > 0.5)).detach().cpu().numpy(),
        dtype=bool,
    )[0]
    queue = user_nodes[:, 2]
    rel_x = user_edges[:, 0]
    rel_y = user_edges[:, 1]
    eta = user_edges[:, -1]
    feat = np.stack([queue, rel_x, rel_y, eta], axis=-1)
    feat = feat * valid[:, None].astype(np.float32)
    return feat.reshape(-1).astype(np.float32)


def _masked_l1(a: np.ndarray, b: np.ndarray, valid_mask: np.ndarray) -> float:
    mask = np.asarray(valid_mask, dtype=bool)
    return float(np.abs(a - b)[mask].sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--init_actor", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_states", type=int, default=6)
    parser.add_argument("--pool_size", type=int, default=256)
    parser.add_argument("--budgets", type=str, default="32,128,256")
    parser.add_argument("--wide_dirichlet_concentration", type=float, default=0.25)
    parser.add_argument("--collect_deterministic", action="store_true")
    parser.add_argument("--follow_deterministic", action="store_true")
    args = parser.parse_args()

    _set_all_seeds(int(args.seed))
    device = _resolve_torch_device(args.device)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(args.config))
    budgets = sorted({int(v) for v in str(args.budgets).split(",") if str(v).strip()})
    if not budgets:
        raise ValueError("At least one budget is required.")
    max_budget = max(budgets)
    if int(args.pool_size) < max_budget:
        raise ValueError("pool_size must be >= max budget.")
    _save_config(
        run_dir,
        cfg,
        os.path.abspath(args.config),
        {
            "init_actor": str(args.init_actor),
            "num_states": int(args.num_states),
            "pool_size": int(args.pool_size),
            "budgets": budgets,
            "wide_dirichlet_concentration": float(args.wide_dirichlet_concentration),
            "collect_deterministic": bool(args.collect_deterministic),
            "follow_deterministic": bool(args.follow_deterministic),
        },
    )

    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    actor = bundle.actor.to(device)
    load_checkpoint_forgiving(actor, str(args.init_actor), map_location=device, strict=True)
    actor.eval()

    snapshots = _collect_bw_snapshots(
        cfg,
        actor,
        device=device,
        num_states=int(args.num_states),
        seed=int(args.seed),
        collect_deterministic=bool(args.collect_deterministic),
    )

    state_records: list[dict[str, Any]] = []
    for idx, record in enumerate(snapshots):
        local_state = record["local_state"]
        with torch.no_grad():
            det_out = actor.act_bw(local_state, deterministic=True)
        det_mean = np.asarray(det_out.action.detach().cpu().numpy(), dtype=np.float32)
        valid_mask = np.asarray(
            ((local_state.user_mask > 0.5) & (local_state.bw_valid_mask > 0.5)).detach().cpu().numpy(),
            dtype=bool,
        )
        candidates = _sample_wide_dirichlet_actions(
            det_mean,
            valid_mask,
            num_samples=int(args.pool_size),
            concentration=float(args.wide_dirichlet_concentration),
        )
        scores = [
            _run_bw_candidate_to_episode_end(
                cfg,
                actor,
                snapshot_state=record["snapshot_state"],
                first_action=action,
                device=device,
                follow_deterministic=bool(args.follow_deterministic),
            )
            for action in candidates
        ]
        budget_best: dict[int, dict[str, Any]] = {}
        for budget in budgets:
            budget_scores = np.asarray(scores[:budget], dtype=np.float64)
            best_idx = int(np.argmax(budget_scores))
            budget_best[int(budget)] = {
                "index": best_idx,
                "reward": float(budget_scores[best_idx]),
                "action": np.asarray(candidates[best_idx], dtype=np.float32).tolist(),
            }
        feature = _feature_vector(local_state)
        state_records.append(
            {
                "state_index": int(idx),
                "valid_mask": valid_mask.astype(bool).tolist(),
                "feature": feature.tolist(),
                "budget_best": budget_best,
            }
        )
        print(
            f"State {idx:02d} | "
            + " | ".join(
                f"N={budget}: reward={budget_best[budget]['reward']:.3f}, idx={budget_best[budget]['index']}"
                for budget in budgets
            )
        )

    stability_summary: dict[str, Any] = {}
    max_key = int(max_budget)
    for budget in budgets:
        if budget == max_key:
            continue
        reward_regrets = []
        action_l1s = []
        same_idx = 0
        for record in state_records:
            valid_mask = np.asarray(record["valid_mask"], dtype=bool)
            action_small = np.asarray(record["budget_best"][budget]["action"], dtype=np.float32)
            action_big = np.asarray(record["budget_best"][max_key]["action"], dtype=np.float32)
            reward_small = float(record["budget_best"][budget]["reward"])
            reward_big = float(record["budget_best"][max_key]["reward"])
            reward_regrets.append(reward_big - reward_small)
            action_l1s.append(_masked_l1(action_small, action_big, valid_mask))
            if int(record["budget_best"][budget]["index"]) == int(record["budget_best"][max_key]["index"]):
                same_idx += 1
        stability_summary[f"{budget}_vs_{max_key}"] = {
            "same_argmax_frac": float(same_idx / max(len(state_records), 1)),
            "reward_regret_mean": float(np.mean(np.asarray(reward_regrets, dtype=np.float64))),
            "reward_regret_max": float(np.max(np.asarray(reward_regrets, dtype=np.float64))),
            "action_l1_mean": float(np.mean(np.asarray(action_l1s, dtype=np.float64))),
            "action_l1_max": float(np.max(np.asarray(action_l1s, dtype=np.float64))),
        }

    features = np.asarray([record["feature"] for record in state_records], dtype=np.float64)
    feat_mean = features.mean(axis=0, keepdims=True)
    feat_std = features.std(axis=0, keepdims=True)
    feat_std = np.where(feat_std > 1.0e-6, feat_std, 1.0)
    norm_features = (features - feat_mean) / feat_std
    n = norm_features.shape[0]
    state_dists = np.zeros((n, n), dtype=np.float64)
    action_dists = np.zeros((n, n), dtype=np.float64)
    reward_transfer_regret = np.zeros((n, n), dtype=np.float64)
    best_actions = [np.asarray(record["budget_best"][max_key]["action"], dtype=np.float32) for record in state_records]
    valid_masks = [np.asarray(record["valid_mask"], dtype=bool) for record in state_records]
    best_rewards = [float(record["budget_best"][max_key]["reward"]) for record in state_records]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            state_dists[i, j] = float(np.linalg.norm(norm_features[i] - norm_features[j]))
            action_dists[i, j] = _masked_l1(best_actions[i], best_actions[j], valid_masks[i])
            transferred = _run_bw_candidate_to_episode_end(
                cfg,
                actor,
                snapshot_state=snapshots[i]["snapshot_state"],
                first_action=best_actions[j],
                device=device,
                follow_deterministic=bool(args.follow_deterministic),
            )
            reward_transfer_regret[i, j] = float(best_rewards[i] - transferred)

    neighbor_rows: list[dict[str, Any]] = []
    nn_action_l1 = []
    far_action_l1 = []
    nn_transfer = []
    far_transfer = []
    for i in range(n):
        row = state_dists[i].copy()
        row[i] = np.inf
        nn_idx = int(np.argmin(row))
        far_idx = int(np.argmax(np.where(np.isfinite(row), row, -np.inf)))
        neighbor_rows.append(
            {
                "state_index": int(i),
                "nearest_index": nn_idx,
                "nearest_state_dist": float(state_dists[i, nn_idx]),
                "nearest_action_l1": float(action_dists[i, nn_idx]),
                "nearest_transfer_regret": float(reward_transfer_regret[i, nn_idx]),
                "farthest_index": far_idx,
                "farthest_state_dist": float(state_dists[i, far_idx]),
                "farthest_action_l1": float(action_dists[i, far_idx]),
                "farthest_transfer_regret": float(reward_transfer_regret[i, far_idx]),
            }
        )
        nn_action_l1.append(action_dists[i, nn_idx])
        far_action_l1.append(action_dists[i, far_idx])
        nn_transfer.append(reward_transfer_regret[i, nn_idx])
        far_transfer.append(reward_transfer_regret[i, far_idx])

    iu = np.triu_indices(n, k=1)
    pair_state = state_dists[iu]
    pair_action = action_dists[iu]
    if pair_state.size > 1 and np.std(pair_state) > 1.0e-12 and np.std(pair_action) > 1.0e-12:
        state_action_corr = float(np.corrcoef(pair_state, pair_action)[0, 1])
    else:
        state_action_corr = float("nan")

    summary = {
        "num_states": int(n),
        "pool_size": int(args.pool_size),
        "budgets": budgets,
        "wide_dirichlet_concentration": float(args.wide_dirichlet_concentration),
        "stability": stability_summary,
        "neighborhood": {
            "state_action_distance_corr": state_action_corr,
            "nearest_action_l1_mean": float(np.mean(np.asarray(nn_action_l1, dtype=np.float64))),
            "farthest_action_l1_mean": float(np.mean(np.asarray(far_action_l1, dtype=np.float64))),
            "nearest_transfer_regret_mean": float(np.mean(np.asarray(nn_transfer, dtype=np.float64))),
            "farthest_transfer_regret_mean": float(np.mean(np.asarray(far_transfer, dtype=np.float64))),
            "pairs": neighbor_rows,
        },
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (run_dir / "per_state.json").open("w", encoding="utf-8") as f:
        json.dump(state_records, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
