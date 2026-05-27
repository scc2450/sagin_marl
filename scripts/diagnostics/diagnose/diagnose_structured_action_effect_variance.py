from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.rl.baselines import cluster_center_queue_aware_policy, queue_aware_policy
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import _collate_dataclass
from sagin_marl.rl.structured_parallel_eval import _refresh_stage_obs_cache, _sat_mask_to_ids
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver_group
from sagin_marl.rl.structured_types import LocalBwState
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving


STAGE_ID = {"accel": 0, "sat": 1, "bw": 2}


def _set_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _summ(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _env_reward(step_result: Any) -> float:
    return float(next(iter(step_result.rewards.values())))


def _done(step_result: Any) -> bool:
    return bool(next(iter(step_result.terminations.values())) or next(iter(step_result.truncations.values())))


def _obs_list(driver: Any) -> list[dict[str, np.ndarray]]:
    return [driver.env._get_obs(i) for i in range(len(driver.env.agents))]


def _baseline_bundle(driver: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = driver.env.cfg
    obs = _obs_list(driver)
    return cluster_center_queue_aware_policy(
        obs,
        cfg,
        getattr(driver.env, "gu_cluster_centers", None),
        getattr(driver.env, "gu_cluster_counts", None),
    )


def _queue_bundle(driver: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return queue_aware_policy(_obs_list(driver), driver.env.cfg)


def _load_actor(run_dir: Path, device: torch.device):
    cfg = load_config(str(run_dir / "config_source.yaml"))
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    ckpt = run_dir / "actor_final.pt"
    load_checkpoint_forgiving(bundle.actor, str(ckpt), map_location=device, strict=True)
    bundle.actor.to(device).eval()
    return cfg, bundle.actor


def _actor_accel(driver: Any, actor: Any, device: torch.device, deterministic: bool) -> np.ndarray:
    local_states = driver.build_local_accel_states()
    batch = _collate_dataclass(local_states, device)
    with torch.inference_mode():
        out = actor.act_accel(batch, deterministic=bool(deterministic))
    return np.asarray(out.action.detach().cpu().numpy(), dtype=np.float32).reshape(driver.env.cfg.num_uav, 2)


def _actor_sat(driver: Any, actor: Any, device: torch.device, deterministic: bool) -> np.ndarray:
    snapshot = driver.build_sat_stage_snapshot()
    local_state = snapshot.local_state
    if local_state is None:
        raise RuntimeError("SAT snapshot has no local_state")
    local_state = _collate_dataclass([local_state], device)
    with torch.inference_mode():
        out = actor.act_sat(local_state, deterministic=bool(deterministic))
    subset_idx = out.subset_index.detach().cpu().to(dtype=torch.long).view(-1).tolist()
    return driver.decode_sat_subset_actions([], subset_idx)


def _bw_local_state_from_snapshot(snapshot: Any, device: torch.device) -> LocalBwState:
    return LocalBwState(
        ego_features=torch.as_tensor(snapshot.ego_features, dtype=torch.float32, device=device),
        selected_sat_tokens=torch.as_tensor(snapshot.selected_sat_tokens, dtype=torch.float32, device=device),
        selected_sat_mask=torch.as_tensor(snapshot.selected_sat_mask, dtype=torch.bool, device=device),
        gu_tokens=torch.as_tensor(snapshot.gu_tokens, dtype=torch.float32, device=device),
        gu_mask=torch.as_tensor(snapshot.gu_mask, dtype=torch.bool, device=device),
        bw_valid_mask=torch.as_tensor(snapshot.bw_valid_mask, dtype=torch.bool, device=device),
    )


def _actor_bw(driver: Any, actor: Any, device: torch.device, deterministic: bool) -> np.ndarray:
    snapshot = driver.build_bw_stage_snapshot()
    local_state = _bw_local_state_from_snapshot(snapshot, device)
    with torch.inference_mode():
        out = actor.act_bw(local_state, deterministic=bool(deterministic))
    return np.asarray(out.action.detach().cpu().numpy(), dtype=np.float32)


def _stage_ref_action(stage: str, driver: Any, actor: Any, device: torch.device) -> np.ndarray:
    if stage == "accel":
        return _actor_accel(driver, actor, device, deterministic=True)
    if stage == "sat":
        return _actor_sat(driver, actor, device, deterministic=True)
    if stage == "bw":
        return _actor_bw(driver, actor, device, deterministic=True)
    raise ValueError(stage)


def _stage_sample_action(stage: str, driver: Any, actor: Any, device: torch.device) -> np.ndarray:
    if stage == "accel":
        return _actor_accel(driver, actor, device, deterministic=False)
    if stage == "sat":
        return _actor_sat(driver, actor, device, deterministic=False)
    if stage == "bw":
        return _actor_bw(driver, actor, device, deterministic=False)
    raise ValueError(stage)


def _action_distance(stage: str, ref: np.ndarray, action: np.ndarray) -> float:
    ref_arr = np.asarray(ref)
    act_arr = np.asarray(action)
    if stage == "accel":
        return float(np.linalg.norm((act_arr - ref_arr).reshape(-1)))
    if stage == "sat":
        return float(np.mean(act_arr != ref_arr))
    if stage == "bw":
        return float(np.sum(np.abs(act_arr - ref_arr)))
    return 0.0


def _reset_to_accel_snapshot(driver: Any, snapshot: dict[str, Any]) -> None:
    driver.env.load_runtime_state(
        snapshot["env_state"],
        refresh_observation_cache=False,
        refresh_global_state_cache=False,
    )
    driver._clear_step()
    driver.begin_step()


def _run_accel_action(driver: Any, action: np.ndarray) -> None:
    driver.run_accel_stage(np.asarray(action, dtype=np.float32))


def _run_sat_action(driver: Any, action: np.ndarray) -> None:
    driver.run_sat_stage(np.asarray(action, dtype=np.int64))


def _run_bw_action(driver: Any, action: np.ndarray) -> tuple[float, bool]:
    result = driver.execute_stage_bw_and_step(np.asarray(action, dtype=np.float32))
    return _env_reward(result), _done(result)


def _choose_accel_action(source: str, driver: Any, actor: Any, device: torch.device, deterministic: bool) -> np.ndarray:
    source_l = str(source).strip().lower()
    if source_l == "policy":
        return _actor_accel(driver, actor, device, deterministic=deterministic)
    if source_l == "cluster_center_queue_aware":
        accel, _bw, _sat = _baseline_bundle(driver)
        return np.asarray(accel, dtype=np.float32)
    if source_l == "queue_aware":
        accel, _bw, _sat = _queue_bundle(driver)
        return np.asarray(accel, dtype=np.float32)
    if source_l == "zero":
        return np.zeros((driver.env.cfg.num_uav, 2), dtype=np.float32)
    raise ValueError(f"Unsupported accel source: {source}")


def _choose_sat_action(source: str, driver: Any, actor: Any, device: torch.device, deterministic: bool) -> np.ndarray:
    source_l = str(source).strip().lower()
    if source_l == "policy":
        return _actor_sat(driver, actor, device, deterministic=deterministic)
    if source_l in {"queue_aware", "cluster_center_queue_aware"}:
        _refresh_stage_obs_cache(driver)
        if source_l == "cluster_center_queue_aware":
            _accel, _bw, sat_mask = _baseline_bundle(driver)
        else:
            _accel, _bw, sat_mask = _queue_bundle(driver)
        return _sat_mask_to_ids(driver, np.asarray(sat_mask, dtype=np.float32))
    if source_l == "zero":
        k = max(int(getattr(driver.env.cfg, "sat_action_select_k", driver.env.cfg.N_RF) or driver.env.cfg.N_RF), 1)
        return np.full((driver.env.cfg.num_uav, k), -1, dtype=np.int64)
    raise ValueError(f"Unsupported sat source: {source}")


def _valid_queue_bw_action(driver: Any) -> np.ndarray:
    cfg = driver.env.cfg
    assoc = np.asarray(driver._stage_assoc, dtype=np.int64).reshape(int(cfg.num_gu))
    queue = np.asarray(driver.env.gu_queue, dtype=np.float32).reshape(int(cfg.num_gu))
    eta = np.asarray(
        driver._stage_eta_ref_feature
        if driver._stage_eta_ref_feature is not None
        else np.zeros((int(cfg.num_uav), int(cfg.num_gu)), dtype=np.float32),
        dtype=np.float32,
    )
    out = np.zeros((int(cfg.num_uav), int(cfg.num_gu)), dtype=np.float32)
    for u in range(int(cfg.num_uav)):
        valid = assoc == int(u)
        if not bool(np.any(valid)):
            continue
        weights = np.maximum(queue, 0.0) * (0.5 + np.maximum(eta[u], 0.0))
        weights = np.where(valid, weights, 0.0).astype(np.float32, copy=False)
        denom = float(np.sum(weights))
        if denom <= 1.0e-8:
            out[u, valid] = 1.0 / float(np.count_nonzero(valid))
        else:
            out[u] = weights / denom
    return out


def _choose_bw_action(source: str, driver: Any, actor: Any, device: torch.device, deterministic: bool) -> np.ndarray:
    source_l = str(source).strip().lower()
    if source_l == "policy":
        return _actor_bw(driver, actor, device, deterministic=deterministic)
    if source_l in {"queue_aware", "cluster_center_queue_aware"}:
        return _valid_queue_bw_action(driver)
    if source_l == "zero":
        return np.zeros((driver.env.cfg.num_uav, driver.env.cfg.num_gu), dtype=np.float32)
    raise ValueError(f"Unsupported bw source: {source}")


def _finish_one_step_after_stage(
    *,
    first_stage: str,
    driver: Any,
    actor: Any,
    device: torch.device,
    first_action: np.ndarray,
    follow_deterministic: bool,
) -> tuple[float, bool]:
    cfg = driver.env.cfg
    if first_stage == "accel":
        _run_accel_action(driver, first_action)
        sat_action = _choose_sat_action(cfg.exec_sat_source, driver, actor, device, deterministic=follow_deterministic)
        _run_sat_action(driver, sat_action)
        bw_action = _choose_bw_action(cfg.exec_bw_source, driver, actor, device, deterministic=follow_deterministic)
        return _run_bw_action(driver, bw_action)
    if first_stage == "sat":
        _run_sat_action(driver, first_action)
        bw_action = _choose_bw_action(cfg.exec_bw_source, driver, actor, device, deterministic=follow_deterministic)
        return _run_bw_action(driver, bw_action)
    if first_stage == "bw":
        return _run_bw_action(driver, first_action)
    raise ValueError(first_stage)


def _run_reference_full_step(driver: Any, actor: Any, device: torch.device, deterministic: bool) -> tuple[float, bool]:
    cfg = driver.env.cfg
    driver.begin_step()
    accel_action = _choose_accel_action(cfg.exec_accel_source, driver, actor, device, deterministic=deterministic)
    _run_accel_action(driver, accel_action)
    sat_action = _choose_sat_action(cfg.exec_sat_source, driver, actor, device, deterministic=deterministic)
    _run_sat_action(driver, sat_action)
    bw_action = _choose_bw_action(cfg.exec_bw_source, driver, actor, device, deterministic=deterministic)
    return _run_bw_action(driver, bw_action)


def _restore_stage_snapshot(stage: str, driver: Any, snapshot: dict[str, Any]) -> None:
    if stage == "accel":
        _reset_to_accel_snapshot(driver, snapshot)
    elif stage == "sat":
        driver.load_sat_stage_state(snapshot)
    elif stage == "bw":
        driver.load_bw_stage_state(snapshot)
    else:
        raise ValueError(stage)


def _return_from_snapshot(
    *,
    stage: str,
    driver: Any,
    snapshot: dict[str, Any],
    first_action: np.ndarray,
    actor: Any,
    device: torch.device,
    gamma: float,
    follow_deterministic: bool,
    horizon_steps: int,
) -> float:
    _restore_stage_snapshot(stage, driver, snapshot)
    total = 0.0
    discount = 1.0
    steps = 0
    reward, done = _finish_one_step_after_stage(
        first_stage=stage,
        driver=driver,
        actor=actor,
        device=device,
        first_action=first_action,
        follow_deterministic=follow_deterministic,
    )
    total += discount * float(reward)
    discount *= float(gamma)
    steps += 1
    max_steps = int(horizon_steps)
    while (
        not done
        and int(driver.env.t) < int(driver.env.cfg.T_steps)
        and (max_steps <= 0 or steps < max_steps)
    ):
        reward, done = _run_reference_full_step(
            driver,
            actor,
            device,
            deterministic=follow_deterministic,
        )
        total += discount * float(reward)
        discount *= float(gamma)
        steps += 1
    return float(total)


def _collect_rows(
    *,
    stage: str,
    cfg: Any,
    actor: Any,
    device: torch.device,
    episodes: int,
    episode_seed_base: int,
    sample_steps: list[int],
    samples_per_row: int,
    collect_deterministic: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_set = {int(x) for x in sample_steps}
    for ep in range(int(episodes)):
        group = make_structured_driver_group(cfg, num_envs=1, backend="sync", mode="script")
        driver = group[0]
        env = driver.env
        try:
            group.reset_many([int(episode_seed_base) + int(ep)])
            done = False
            while not done and int(env.t) < int(cfg.T_steps):
                t = int(env.t)
                pre_step_state = env.export_runtime_state()
                driver.begin_step()
                if stage == "accel":
                    if t in sample_set:
                        snapshot = {"env_state": pre_step_state}
                        ref_action = _stage_ref_action(stage, driver, actor, device)
                        sample_actions = [
                            _stage_sample_action(stage, driver, actor, device)
                            for _ in range(max(int(samples_per_row), 0))
                        ]
                        rows.append(
                            {
                                "episode": ep,
                                "seed": int(episode_seed_base) + int(ep),
                                "t": t,
                                "snapshot": snapshot,
                                "ref_action": ref_action,
                                "sample_actions": sample_actions,
                            }
                        )
                    accel_action = _choose_accel_action(cfg.exec_accel_source, driver, actor, device, deterministic=collect_deterministic)
                else:
                    accel_action = _choose_accel_action(cfg.exec_accel_source, driver, actor, device, deterministic=collect_deterministic)
                _run_accel_action(driver, accel_action)

                if stage == "sat" and t in sample_set:
                    snapshot = driver.export_sat_stage_state()
                    ref_action = _stage_ref_action(stage, driver, actor, device)
                    sample_actions = [
                        _stage_sample_action(stage, driver, actor, device)
                        for _ in range(max(int(samples_per_row), 0))
                    ]
                    rows.append(
                        {
                            "episode": ep,
                            "seed": int(episode_seed_base) + int(ep),
                            "t": t,
                            "snapshot": snapshot,
                            "ref_action": ref_action,
                            "sample_actions": sample_actions,
                        }
                    )
                sat_action = _choose_sat_action(cfg.exec_sat_source, driver, actor, device, deterministic=collect_deterministic)
                _run_sat_action(driver, sat_action)

                if stage == "bw" and t in sample_set:
                    snapshot = driver.export_bw_stage_state()
                    ref_action = _stage_ref_action(stage, driver, actor, device)
                    sample_actions = [
                        _stage_sample_action(stage, driver, actor, device)
                        for _ in range(max(int(samples_per_row), 0))
                    ]
                    rows.append(
                        {
                            "episode": ep,
                            "seed": int(episode_seed_base) + int(ep),
                            "t": t,
                            "snapshot": snapshot,
                            "ref_action": ref_action,
                            "sample_actions": sample_actions,
                        }
                    )
                bw_action = _choose_bw_action(cfg.exec_bw_source, driver, actor, device, deterministic=collect_deterministic)
                reward, done = _run_bw_action(driver, bw_action)
                del reward
        finally:
            close_structured_env_group(group)
    return rows


def _diagnose_rows(
    *,
    stage: str,
    cfg: Any,
    actor: Any,
    device: torch.device,
    rows: list[dict[str, Any]],
    follow_deterministic: bool,
    horizon_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eval_group = make_structured_driver_group(cfg, num_envs=1, backend="sync", mode="script")
    eval_driver = eval_group[0]
    out_rows: list[dict[str, Any]] = []
    try:
        for idx, row in enumerate(rows):
            ref_action = np.asarray(row["ref_action"])
            ref_return = _return_from_snapshot(
                stage=stage,
                driver=eval_driver,
                snapshot=row["snapshot"],
                first_action=ref_action,
                actor=actor,
                device=device,
                gamma=float(cfg.gamma),
                follow_deterministic=follow_deterministic,
                horizon_steps=int(horizon_steps),
            )
            candidate_returns: list[float] = []
            candidate_distances: list[float] = []
            for action in row["sample_actions"]:
                action_arr = np.asarray(action)
                candidate_returns.append(
                    _return_from_snapshot(
                        stage=stage,
                        driver=eval_driver,
                        snapshot=row["snapshot"],
                        first_action=action_arr,
                        actor=actor,
                        device=device,
                        gamma=float(cfg.gamma),
                        follow_deterministic=follow_deterministic,
                        horizon_steps=int(horizon_steps),
                    )
                )
                candidate_distances.append(_action_distance(stage, ref_action, action_arr))
            deltas = np.asarray(candidate_returns, dtype=np.float64) - float(ref_return)
            all_returns = np.asarray([ref_return] + candidate_returns, dtype=np.float64)
            out_rows.append(
                {
                    "row": int(idx),
                    "stage": stage,
                    "episode": int(row["episode"]),
                    "seed": int(row["seed"]),
                    "t": int(row["t"]),
                    "ref_return": float(ref_return),
                    "candidate_return_mean": float(np.mean(candidate_returns)) if candidate_returns else float(ref_return),
                    "candidate_return_std": float(np.std(candidate_returns)) if candidate_returns else 0.0,
                    "within_action_return_var": float(np.var(all_returns)),
                    "delta_mean": float(np.mean(deltas)) if deltas.size else 0.0,
                    "delta_abs_mean": float(np.mean(np.abs(deltas))) if deltas.size else 0.0,
                    "delta_abs_max": float(np.max(np.abs(deltas))) if deltas.size else 0.0,
                    "best_gain": float(np.max(deltas)) if deltas.size else 0.0,
                    "positive_frac": float(np.mean(deltas > 0.0)) if deltas.size else 0.0,
                    "action_distance_mean": float(np.mean(candidate_distances)) if candidate_distances else 0.0,
                    "action_distance_max": float(np.max(candidate_distances)) if candidate_distances else 0.0,
                }
            )
    finally:
        close_structured_env_group(eval_group)

    ref_returns = np.asarray([r["ref_return"] for r in out_rows], dtype=np.float64)
    action_vars = np.asarray([r["within_action_return_var"] for r in out_rows], dtype=np.float64)
    delta_abs = np.asarray([r["delta_abs_mean"] for r in out_rows], dtype=np.float64)
    by_t: dict[str, dict[str, float]] = {}
    ratios_by_row: list[float] = []
    for t in sorted({int(r["t"]) for r in out_rows}):
        idxs = [i for i, r in enumerate(out_rows) if int(r["t"]) == int(t)]
        ref_t = ref_returns[idxs]
        action_var_t = action_vars[idxs]
        traj_var_t = float(np.var(ref_t)) if ref_t.size > 1 else 0.0
        mean_action_var_t = float(np.mean(action_var_t)) if action_var_t.size else 0.0
        ratio_t = mean_action_var_t / traj_var_t if traj_var_t > 1.0e-12 else 0.0
        by_t[str(t)] = {
            "n": float(len(idxs)),
            "ref_return_mean": float(np.mean(ref_t)) if ref_t.size else 0.0,
            "trajectory_std_same_t": float(np.std(ref_t)) if ref_t.size else 0.0,
            "trajectory_var_same_t": traj_var_t,
            "action_std_mean_same_t": float(np.mean(np.sqrt(np.maximum(action_var_t, 0.0)))) if action_var_t.size else 0.0,
            "action_var_mean_same_t": mean_action_var_t,
            "action_var_to_trajectory_var": ratio_t,
        }
        if traj_var_t > 1.0e-12:
            ratios_by_row.extend((action_var_t / traj_var_t).tolist())

    global_traj_var = float(np.var(ref_returns)) if ref_returns.size > 1 else 0.0
    global_traj_std = float(np.std(ref_returns)) if ref_returns.size > 1 else 0.0
    action_var_mean = float(np.mean(action_vars)) if action_vars.size else 0.0
    action_std_mean = float(np.mean(np.sqrt(np.maximum(action_vars, 0.0)))) if action_vars.size else 0.0
    summary = {
        "stage": stage,
        "n_rows": int(len(out_rows)),
        "samples_per_row": int(len(rows[0]["sample_actions"])) if rows else 0,
        "horizon_steps": int(horizon_steps),
        "ref_return": _summ(ref_returns),
        "delta_abs_mean_per_row": _summ(delta_abs),
        "best_gain": _summ([r["best_gain"] for r in out_rows]),
        "action_distance_mean": _summ([r["action_distance_mean"] for r in out_rows]),
        "global_trajectory_std": global_traj_std,
        "global_trajectory_var": global_traj_var,
        "mean_within_action_std": action_std_mean,
        "mean_within_action_var": action_var_mean,
        "global_action_var_to_trajectory_var": action_var_mean / global_traj_var if global_traj_var > 1.0e-12 else 0.0,
        "mean_delta_abs_to_global_trajectory_std": float(np.mean(delta_abs) / global_traj_std)
        if global_traj_std > 1.0e-12 and delta_abs.size
        else 0.0,
        "same_t_action_var_to_trajectory_var": _summ(ratios_by_row),
        "by_t": by_t,
    }
    return out_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--stage", choices=["accel", "sat", "bw"], required=True)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--episode_seed_base", type=int, default=73000)
    parser.add_argument("--sample_steps", type=str, default="0,50,100,150,200")
    parser.add_argument("--samples_per_row", type=int, default=4)
    parser.add_argument("--horizon_steps", type=int, default=80, help="Fixed diagnostic horizon; <=0 means episode remaining.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seeds(int(args.seed))
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = str(args.stage)
    device = torch.device(args.device)
    cfg, actor = _load_actor(run_dir, device)
    sample_steps = [int(x.strip()) for x in str(args.sample_steps).split(",") if x.strip()]
    rows = _collect_rows(
        stage=stage,
        cfg=cfg,
        actor=actor,
        device=device,
        episodes=int(args.episodes),
        episode_seed_base=int(args.episode_seed_base),
        sample_steps=sample_steps,
        samples_per_row=int(args.samples_per_row),
        collect_deterministic=True,
    )
    detail_rows, summary = _diagnose_rows(
        stage=stage,
        cfg=cfg,
        actor=actor,
        device=device,
        rows=rows,
        follow_deterministic=True,
        horizon_steps=int(args.horizon_steps),
    )
    payload = {
        "run_dir": str(run_dir),
        "config": str(run_dir / "config_source.yaml"),
        "stage": stage,
        "episodes": int(args.episodes),
        "episode_seed_base": int(args.episode_seed_base),
        "sample_steps": sample_steps,
        "samples_per_row": int(args.samples_per_row),
        "horizon_steps": int(args.horizon_steps),
        "method": (
            "For each sampled stage snapshot, reload the identical environment/RNG state, "
            "evaluate deterministic ref action and stochastic first-action samples, then "
            "follow deterministic current policy/baselines for a fixed diagnostic horizon "
            "(or to episode end when horizon_steps<=0). Trajectory fluctuation is "
            "ref-return variance across episode seeds at the same t."
        ),
        "summary": summary,
    }
    json_path = out_dir / f"{stage}_action_effect_variance.json"
    csv_path = out_dir / f"{stage}_action_effect_variance_rows.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(detail_rows[0].keys()) if detail_rows else [
            "row",
            "stage",
            "episode",
            "seed",
            "t",
            "ref_return",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
