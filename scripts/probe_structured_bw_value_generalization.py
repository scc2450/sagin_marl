from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sagin_marl.env.config import SaginConfig, load_config, update_config
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import _collate_dataclass, _index_dataclass, _to_device_dataclass
from sagin_marl.rl.structured_stage_builders import build_local_bw_states_from_snapshot
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving


def _load_run_config(run_dir: Path):
    config_path = run_dir / "config.yaml"
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        data = {key: value for key, value in data.items() if not str(key).startswith("_")}
        return update_config(SaginConfig(), data)
    return load_config(str(run_dir / "config_source.yaml"))


def _zero_sat_action(cfg) -> np.ndarray:
    select_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 1)
    return np.full((cfg.num_uav, select_k), -1, dtype=np.int64)


@contextmanager
def _temporary_torch_seed(seed: int | None, device: torch.device):
    if seed is None:
        yield
        return
    cpu_state = torch.random.get_rng_state()
    cuda_states = None
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
    torch.manual_seed(int(seed))
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _bw_action_from_actor(
    actor,
    snapshot,
    device: torch.device,
    *,
    deterministic: bool,
    seed: int | None,
) -> np.ndarray:
    local_state = build_local_bw_states_from_snapshot(snapshot)[0]
    local_state = _to_device_dataclass(local_state, device)
    with _temporary_torch_seed(seed, device), torch.inference_mode():
        out = actor.act_bw(local_state, deterministic=bool(deterministic))
    return np.asarray(out.action.detach().cpu().numpy(), dtype=np.float32)


def _collect_policy_bw_snapshots(
    cfg,
    actor,
    device: torch.device,
    *,
    count: int,
    seed: int,
    deterministic: bool,
) -> list[dict[str, Any]]:
    driver = make_structured_driver(cfg, backend="sync")
    entries: list[dict[str, Any]] = []
    try:
        episode_idx = 0
        driver.env.reset(seed=int(seed) + episode_idx)
        done = False
        while len(entries) < int(count):
            if done:
                episode_idx += 1
                driver.env.reset(seed=int(seed) + episode_idx)
                done = False
            driver.begin_step()
            z1 = driver.run_accel_stage(np.zeros((cfg.num_uav, 2), dtype=np.float32))
            z2 = driver.run_sat_stage(_zero_sat_action(cfg))
            snapshot = driver.build_bw_stage_snapshot(z2)
            snapshot_state = driver.export_bw_stage_state()
            entries.append(
                {
                    "episode": int(episode_idx),
                    "t": int(snapshot_state.get("env_state", {}).get("t", 0) or 0),
                    "snapshot_state": snapshot_state,
                    "world_state": snapshot.world_state,
                }
            )
            action = _bw_action_from_actor(
                actor,
                snapshot,
                device,
                deterministic=bool(deterministic),
                seed=None if deterministic else int(seed) + int(len(entries)) * 10_000,
            )
            step_result, _ = driver.execute_stage_bw_and_prepare_next_accel(action)
            done = bool(next(iter(step_result.terminations.values())) or next(iter(step_result.truncations.values())))
        return entries
    finally:
        close_structured_env_group(driver)


def _bucket_index(value: float, edges: list[float]) -> int:
    out = 0
    for edge in edges:
        if float(value) >= float(edge):
            out += 1
    return int(out)


def _entry_state_features(entry: dict[str, Any], cfg) -> dict[str, Any]:
    snapshot_state = entry.get("snapshot_state", {})
    env_state = snapshot_state.get("env_state", {}) if isinstance(snapshot_state, dict) else {}
    arrival_ref = max(float(env_state.get("arrival_ref_bits_per_step", getattr(cfg, "task_arrival_rate", 1.0)) or 1.0), 1.0e-9)
    gu_queue = np.asarray(env_state.get("gu_queue", []), dtype=np.float64).reshape(-1)
    uav_queue = np.asarray(env_state.get("uav_queue", []), dtype=np.float64).reshape(-1)
    sat_queue = np.asarray(env_state.get("sat_queue", []), dtype=np.float64).reshape(-1)
    queue_total_steps = float((gu_queue.sum() + uav_queue.sum() + sat_queue.sum()) / arrival_ref)
    gu_mean = float(np.mean(gu_queue)) if gu_queue.size else 0.0
    gu_std = float(np.std(gu_queue)) if gu_queue.size else 0.0
    queue_imbalance_cv = float(gu_std / max(abs(gu_mean), 1.0e-9))
    prev_queue_sum = float(env_state.get("prev_queue_sum", gu_queue.sum() + uav_queue.sum() + sat_queue.sum()) or 0.0)
    backlog_slope_steps = float((gu_queue.sum() + uav_queue.sum() + sat_queue.sum() - prev_queue_sum) / arrival_ref)
    hotspot_idx = int(env_state.get("_hotspot_active_idx", env_state.get("last_hotspot_index", -1)) or -1)
    hotspot_active = int(hotspot_idx >= 0)
    t = int(env_state.get("t", entry.get("t", 0)) or 0)
    t_steps = max(int(getattr(cfg, "T_steps", 1) or 1), 1)
    time_frac = float(t) / float(t_steps)
    features = {
        "t": int(t),
        "time_frac": float(time_frac),
        "time_bin": int(_bucket_index(time_frac, [1.0 / 3.0, 2.0 / 3.0])),
        "queue_total_steps": float(queue_total_steps),
        "queue_total_bin": int(_bucket_index(queue_total_steps, [3.0, 6.0, 10.0])),
        "queue_imbalance_cv": float(queue_imbalance_cv),
        "queue_imbalance_bin": int(_bucket_index(queue_imbalance_cv, [0.25, 0.75, 1.5])),
        "hotspot_idx": int(hotspot_idx),
        "hotspot_active": int(hotspot_active),
        "backlog_slope_steps": float(backlog_slope_steps),
        "backlog_slope_bin": int(_bucket_index(backlog_slope_steps, [-0.25, 0.25, 1.0])),
    }
    features["bucket_key"] = (
        f"t{features['time_bin']}"
        f"|q{features['queue_total_bin']}"
        f"|imb{features['queue_imbalance_bin']}"
        f"|hs{features['hotspot_active']}"
        f"|s{features['backlog_slope_bin']}"
    )
    return features


def _attach_state_features(entries: list[dict[str, Any]], cfg) -> list[dict[str, Any]]:
    for entry in entries:
        entry["features"] = _entry_state_features(entry, cfg)
    return entries


def _select_stratified_entries(entries: list[dict[str, Any]], *, count: int, seed: int) -> list[dict[str, Any]]:
    if len(entries) <= int(count):
        return list(entries)
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        features = entry.get("features") or {}
        key = str(features.get("bucket_key", "unknown"))
        groups.setdefault(key, []).append(entry)
    rng = np.random.default_rng(int(seed))
    for values in groups.values():
        rng.shuffle(values)
    keys = sorted(groups, key=lambda key: (-len(groups[key]), key))
    selected: list[dict[str, Any]] = []
    used_ids: set[int] = set()
    while len(selected) < int(count):
        added = False
        for key in keys:
            group = groups[key]
            while group and id(group[0]) in used_ids:
                group.pop(0)
            if not group:
                continue
            item = group.pop(0)
            selected.append(item)
            used_ids.add(id(item))
            added = True
            if len(selected) >= int(count):
                break
        if not added:
            break
    if len(selected) < int(count):
        remaining = [entry for entry in entries if id(entry) not in used_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: int(count) - len(selected)])
    return selected[: int(count)]


def _collect_value_entries(
    cfg,
    actor,
    device: torch.device,
    *,
    count: int,
    seed: int,
    deterministic: bool,
    stratified: bool,
    pool_multiplier: int,
) -> list[dict[str, Any]]:
    if not bool(stratified):
        return _attach_state_features(
            _collect_policy_bw_snapshots(
                cfg,
                actor,
                device,
                count=int(count),
                seed=int(seed),
                deterministic=bool(deterministic),
            ),
            cfg,
        )
    pool_count = max(int(count), int(count) * max(int(pool_multiplier), 1))
    pool_entries = _attach_state_features(
        _collect_policy_bw_snapshots(
            cfg,
            actor,
            device,
            count=int(pool_count),
            seed=int(seed),
            deterministic=bool(deterministic),
        ),
        cfg,
    )
    return _select_stratified_entries(pool_entries, count=int(count), seed=int(seed) + 17)


def _rollout_policy_value_from_snapshot(
    driver,
    snapshot_state: dict[str, Any],
    cfg,
    actor,
    device: torch.device,
    *,
    gamma: float,
    horizon: int,
    deterministic: bool,
    seed: int,
    resample_env_rng: bool,
    stop_on_truncation: bool,
) -> float:
    driver.load_bw_stage_state(snapshot_state)
    if bool(resample_env_rng):
        driver.env.rng = np.random.default_rng(int(seed))
    total = 0.0
    discount = 1.0
    for step_idx in range(max(int(horizon), 1)):
        snapshot = driver.build_bw_stage_snapshot()
        action = _bw_action_from_actor(
            actor,
            snapshot,
            device,
            deterministic=bool(deterministic),
            seed=None if deterministic else int(seed) + int(step_idx) * 997,
        )
        step_result, _ = driver.execute_stage_bw_and_prepare_next_accel(action)
        total += discount * float(next(iter(step_result.rewards.values())))
        terminated = bool(next(iter(step_result.terminations.values())))
        truncated = bool(next(iter(step_result.truncations.values())))
        done = terminated or (bool(stop_on_truncation) and truncated)
        if done or step_idx + 1 >= int(horizon):
            break
        _ = driver.run_accel_stage(np.zeros((cfg.num_uav, 2), dtype=np.float32))
        _ = driver.run_sat_stage(_zero_sat_action(cfg))
        discount *= float(gamma)
    return float(total)


def _estimate_mc_targets(
    cfg,
    actor,
    device: torch.device,
    entries: list[dict[str, Any]],
    *,
    mc_rollouts: int,
    horizon: int,
    gamma: float,
    deterministic: bool,
    seed: int,
    resample_env_rng: bool,
    stop_on_truncation: bool,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    driver = make_structured_driver(cfg, backend="sync")
    means: list[float] = []
    stds: list[float] = []
    ses: list[float] = []
    try:
        for idx, entry in enumerate(entries):
            returns: list[float] = []
            for mc_idx in range(max(int(mc_rollouts), 1)):
                rollout_seed = int(seed) + int(idx) * 100_000 + int(mc_idx) * 1_009
                returns.append(
                    _rollout_policy_value_from_snapshot(
                        driver,
                        entry["snapshot_state"],
                        cfg,
                        actor,
                        device,
                        gamma=float(gamma),
                        horizon=int(horizon),
                        deterministic=bool(deterministic),
                        seed=int(rollout_seed),
                        resample_env_rng=bool(resample_env_rng),
                        stop_on_truncation=bool(stop_on_truncation),
                    )
                )
            arr = np.asarray(returns, dtype=np.float64)
            means.append(float(np.mean(arr)))
            stds.append(float(np.std(arr)))
            ses.append(float(np.std(arr) / math.sqrt(max(arr.size, 1))))
            if (idx + 1) % max(1, min(16, len(entries))) == 0 or idx + 1 == len(entries):
                print(f"{label}: MC targets {idx + 1}/{len(entries)}", flush=True)
        return (
            np.asarray(means, dtype=np.float32),
            np.asarray(stds, dtype=np.float32),
            np.asarray(ses, dtype=np.float32),
        )
    finally:
        close_structured_env_group(driver)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(finite)) <= 1:
        return 0.0
    x = x[finite]
    y = y[finite]
    if float(np.std(x)) <= 1.0e-12 or float(np.std(y)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _metrics(pred: np.ndarray, target: np.ndarray, target_se: np.ndarray | None = None) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    err = pred - target
    target_var = float(np.var(target)) if target.size > 1 else 0.0
    residual_var = float(np.var(err)) if err.size > 1 else 0.0
    out = {
        "count": float(target.size),
        "target_mean": float(np.mean(target)) if target.size else 0.0,
        "target_std": float(np.std(target)) if target.size else 0.0,
        "pred_mean": float(np.mean(pred)) if pred.size else 0.0,
        "pred_std": float(np.std(pred)) if pred.size else 0.0,
        "bias": float(np.mean(err)) if err.size else 0.0,
        "mae": float(np.mean(np.abs(err))) if err.size else 0.0,
        "mse": float(np.mean(err * err)) if err.size else 0.0,
        "rmse": float(np.sqrt(np.mean(err * err))) if err.size else 0.0,
        "corr": _safe_corr(pred, target),
        "explained_variance": 0.0 if target_var <= 1.0e-12 else float(1.0 - residual_var / target_var),
    }
    if target_se is not None:
        target_se_arr = np.asarray(target_se, dtype=np.float64).reshape(-1)
        out["target_mc_se_mean"] = float(np.mean(target_se_arr)) if target_se_arr.size else 0.0
        out["rmse_over_mc_se_mean"] = out["rmse"] / max(out["target_mc_se_mean"], 1.0e-12)
    return out


def _group_metrics_by_feature(
    entries: list[dict[str, Any]],
    pred: np.ndarray,
    target: np.ndarray,
    target_se: np.ndarray,
    *,
    feature: str,
) -> list[dict[str, float | str]]:
    groups: dict[str, list[int]] = {}
    for idx, entry in enumerate(entries):
        features = entry.get("features") or {}
        key = str(features.get(feature, "unknown"))
        groups.setdefault(key, []).append(int(idx))
    rows: list[dict[str, float | str]] = []
    pred_arr = np.asarray(pred, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    se_arr = np.asarray(target_se, dtype=np.float64)
    for key in sorted(groups, key=lambda value: (str(value))):
        idx = np.asarray(groups[key], dtype=np.int64)
        metric = _metrics(pred_arr[idx], target_arr[idx], se_arr[idx])
        row: dict[str, float | str] = {"feature": str(feature), "bucket": str(key)}
        row.update(metric)
        rows.append(row)
    return rows


def _feature_distribution(entries: list[dict[str, Any]], *, feature: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        key = str((entry.get("features") or {}).get(feature, "unknown"))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _predict_critic(critic, world_states: list[Any], device: torch.device, *, batch_size: int) -> np.ndarray:
    if not world_states:
        return np.zeros((0,), dtype=np.float32)
    batch = _collate_dataclass(world_states, device)
    preds: list[np.ndarray] = []
    critic.eval()
    with torch.inference_mode():
        n = int(len(world_states))
        for start in range(0, n, max(int(batch_size), 1)):
            idx = torch.arange(start, min(start + int(batch_size), n), device=device, dtype=torch.long)
            pred = critic.value_bw(_index_dataclass(batch, idx))
            preds.append(pred.detach().cpu().numpy().reshape(-1).astype(np.float32))
    return np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.float32)


def _train_critic_supervised(
    critic,
    *,
    train_world_states: list[Any],
    train_targets: np.ndarray,
    holdout_world_states: list[Any],
    holdout_targets: np.ndarray,
    holdout_se: np.ndarray,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    eval_epochs: set[int],
) -> tuple[list[dict[str, float]], dict[str, float]]:
    train_batch = _collate_dataclass(train_world_states, device)
    targets = torch.as_tensor(train_targets, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(critic.parameters(), lr=float(lr))
    rows: list[dict[str, float]] = []

    def record(epoch: int, train_loss: float) -> None:
        train_pred = _predict_critic(critic, train_world_states, device, batch_size=batch_size)
        holdout_pred = _predict_critic(critic, holdout_world_states, device, batch_size=batch_size)
        row = {"epoch": float(epoch), "train_loss_last": float(train_loss)}
        row.update({f"train_{k}": v for k, v in _metrics(train_pred, train_targets).items()})
        row.update({f"holdout_{k}": v for k, v in _metrics(holdout_pred, holdout_targets, holdout_se).items()})
        rows.append(row)

    record(0, float("nan"))
    n = int(targets.numel())
    for epoch in range(1, max(int(epochs), 0) + 1):
        critic.train()
        perm = torch.randperm(n, device=device)
        loss_values: list[float] = []
        for start in range(0, n, max(int(batch_size), 1)):
            idx = perm[start : start + max(int(batch_size), 1)]
            pred = critic.value_bw(_index_dataclass(train_batch, idx))
            loss = F.mse_loss(pred.reshape(-1), targets.index_select(0, idx).reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            opt.step()
            loss_values.append(float(loss.item()))
        if epoch in eval_epochs or epoch == int(epochs):
            record(epoch, float(np.mean(loss_values)) if loss_values else 0.0)
    return rows, rows[-1] if rows else {}


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--actor_checkpoint", type=str, default=None)
    parser.add_argument("--critic_checkpoint", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--train_states", type=int, default=96)
    parser.add_argument("--holdout_states", type=int, default=48)
    parser.add_argument("--stratified_collection", action="store_true")
    parser.add_argument("--stratified_pool_multiplier", type=int, default=4)
    parser.add_argument("--mc_rollouts", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--policy_mode", choices=["deterministic", "stochastic"], default="stochastic")
    parser.add_argument("--resample_env_rng", action="store_true")
    parser.add_argument(
        "--continue_through_truncation",
        action="store_true",
        help="Ignore time-limit truncation and keep rolling to --horizon; useful as a continuing-value approximation.",
    )
    parser.add_argument("--train_seed", type=int, default=61000)
    parser.add_argument("--holdout_seed", type=int, default=71000)
    parser.add_argument("--mc_seed", type=int, default=81000)
    parser.add_argument("--epochs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cfg = _load_run_config(run_dir)
    bundle = build_structured_modules_from_config(cfg, hidden_dim=int(args.hidden_dim), embed_dim=int(args.embed_dim))
    actor = bundle.actor.to(device)
    online_critic = bundle.critic.to(device)
    actor_ckpt = Path(args.actor_checkpoint) if args.actor_checkpoint else run_dir / "actor_final.pt"
    critic_ckpt = Path(args.critic_checkpoint) if args.critic_checkpoint else run_dir / "critic_final.pt"
    load_checkpoint_forgiving(actor, str(actor_ckpt), map_location=device, strict=True)
    load_checkpoint_forgiving(online_critic, str(critic_ckpt), map_location=device, strict=True)
    actor.eval()
    online_critic.eval()
    deterministic = str(args.policy_mode) == "deterministic"
    print(
        "Fixed-policy value probe | "
        f"run={run_dir} | policy={args.policy_mode} | train={args.train_states} | "
        f"holdout={args.holdout_states} | mc={args.mc_rollouts} | horizon={args.horizon} | "
        f"target={'continue' if args.continue_through_truncation else 'finite'}",
        flush=True,
    )

    train_entries = _collect_value_entries(
        cfg,
        actor,
        device,
        count=int(args.train_states),
        seed=int(args.train_seed),
        deterministic=bool(deterministic),
        stratified=bool(args.stratified_collection),
        pool_multiplier=int(args.stratified_pool_multiplier),
    )
    holdout_entries = _collect_value_entries(
        cfg,
        actor,
        device,
        count=int(args.holdout_states),
        seed=int(args.holdout_seed),
        deterministic=bool(deterministic),
        stratified=bool(args.stratified_collection),
        pool_multiplier=int(args.stratified_pool_multiplier),
    )
    train_targets, train_mc_std, train_mc_se = _estimate_mc_targets(
        cfg,
        actor,
        device,
        train_entries,
        mc_rollouts=int(args.mc_rollouts),
        horizon=int(args.horizon),
        gamma=float(cfg.gamma),
        deterministic=bool(deterministic),
        seed=int(args.mc_seed),
        resample_env_rng=bool(args.resample_env_rng),
        stop_on_truncation=not bool(args.continue_through_truncation),
        label="train",
    )
    holdout_targets, holdout_mc_std, holdout_mc_se = _estimate_mc_targets(
        cfg,
        actor,
        device,
        holdout_entries,
        mc_rollouts=int(args.mc_rollouts),
        horizon=int(args.horizon),
        gamma=float(cfg.gamma),
        deterministic=bool(deterministic),
        seed=int(args.mc_seed) + 50_000_000,
        resample_env_rng=bool(args.resample_env_rng),
        stop_on_truncation=not bool(args.continue_through_truncation),
        label="holdout",
    )

    train_world_states = [entry["world_state"] for entry in train_entries]
    holdout_world_states = [entry["world_state"] for entry in holdout_entries]
    online_train_pred = _predict_critic(online_critic, train_world_states, device, batch_size=int(args.batch_size))
    online_holdout_pred = _predict_critic(online_critic, holdout_world_states, device, batch_size=int(args.batch_size))
    online_summary = {
        "train": _metrics(online_train_pred, train_targets, train_mc_se),
        "holdout": _metrics(online_holdout_pred, holdout_targets, holdout_mc_se),
    }

    eval_epochs = {0, 4, 16, 32, 64, 128, int(args.epochs)}
    eval_epochs = {epoch for epoch in eval_epochs if epoch >= 0 and epoch <= int(args.epochs)}
    loaded_critic = copy.deepcopy(online_critic).to(device)
    loaded_rows, loaded_final = _train_critic_supervised(
        loaded_critic,
        train_world_states=train_world_states,
        train_targets=train_targets,
        holdout_world_states=holdout_world_states,
        holdout_targets=holdout_targets,
        holdout_se=holdout_mc_se,
        device=device,
        epochs=int(args.epochs),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        eval_epochs=eval_epochs,
    )
    fresh_bundle = build_structured_modules_from_config(cfg, hidden_dim=int(args.hidden_dim), embed_dim=int(args.embed_dim))
    fresh_critic = fresh_bundle.critic.to(device)
    fresh_rows, fresh_final = _train_critic_supervised(
        fresh_critic,
        train_world_states=train_world_states,
        train_targets=train_targets,
        holdout_world_states=holdout_world_states,
        holdout_targets=holdout_targets,
        holdout_se=holdout_mc_se,
        device=device,
        epochs=int(args.epochs),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        eval_epochs=eval_epochs,
    )
    loaded_holdout_pred = _predict_critic(loaded_critic, holdout_world_states, device, batch_size=int(args.batch_size))
    fresh_holdout_pred = _predict_critic(fresh_critic, holdout_world_states, device, batch_size=int(args.batch_size))

    _write_csv(out_dir / "loaded_critic_fit.csv", loaded_rows)
    _write_csv(out_dir / "fresh_critic_fit.csv", fresh_rows)
    group_feature_names = [
        "bucket_key",
        "time_bin",
        "queue_total_bin",
        "queue_imbalance_bin",
        "hotspot_active",
        "backlog_slope_bin",
    ]
    holdout_group_rows: list[dict[str, float | str]] = []
    holdout_group_summary: dict[str, Any] = {}
    for feature in group_feature_names:
        by_model = {
            "online": _group_metrics_by_feature(
                holdout_entries,
                online_holdout_pred,
                holdout_targets,
                holdout_mc_se,
                feature=feature,
            ),
            "loaded_supervised": _group_metrics_by_feature(
                holdout_entries,
                loaded_holdout_pred,
                holdout_targets,
                holdout_mc_se,
                feature=feature,
            ),
            "fresh_supervised": _group_metrics_by_feature(
                holdout_entries,
                fresh_holdout_pred,
                holdout_targets,
                holdout_mc_se,
                feature=feature,
            ),
        }
        holdout_group_summary[feature] = by_model
        for model_name, rows in by_model.items():
            for row in rows:
                out_row = {"model": model_name}
                out_row.update(row)
                holdout_group_rows.append(out_row)
    _write_csv(out_dir / "holdout_by_feature.csv", holdout_group_rows)
    target_rows = []
    for split, entries, targets, stds, ses in (
        ("train", train_entries, train_targets, train_mc_std, train_mc_se),
        ("holdout", holdout_entries, holdout_targets, holdout_mc_std, holdout_mc_se),
    ):
        for idx, (entry, target, std, se) in enumerate(zip(entries, targets, stds, ses)):
            features = entry.get("features") or {}
            target_rows.append(
                {
                    "split": split,
                    "index": idx,
                    "episode": int(entry["episode"]),
                    "t": int(entry["t"]),
                    "target_mc_mean": float(target),
                    "target_mc_std": float(std),
                    "target_mc_se": float(se),
                    "bucket_key": str(features.get("bucket_key", "")),
                    "time_bin": int(features.get("time_bin", 0)),
                    "queue_total_steps": float(features.get("queue_total_steps", 0.0)),
                    "queue_total_bin": int(features.get("queue_total_bin", 0)),
                    "queue_imbalance_cv": float(features.get("queue_imbalance_cv", 0.0)),
                    "queue_imbalance_bin": int(features.get("queue_imbalance_bin", 0)),
                    "hotspot_idx": int(features.get("hotspot_idx", -1)),
                    "hotspot_active": int(features.get("hotspot_active", 0)),
                    "backlog_slope_steps": float(features.get("backlog_slope_steps", 0.0)),
                    "backlog_slope_bin": int(features.get("backlog_slope_bin", 0)),
                }
            )
    _write_csv(out_dir / "targets.csv", target_rows)

    summary = {
        "run_dir": str(run_dir),
        "actor_checkpoint": str(actor_ckpt),
        "critic_checkpoint": str(critic_ckpt),
        "config": {
            "T_steps": int(cfg.T_steps),
            "gamma": float(cfg.gamma),
            "gae_lambda": float(cfg.gae_lambda),
            "policy_mode": str(args.policy_mode),
            "train_states": int(args.train_states),
            "holdout_states": int(args.holdout_states),
            "mc_rollouts": int(args.mc_rollouts),
            "horizon": int(args.horizon),
            "resample_env_rng": bool(args.resample_env_rng),
            "target_mode": "continue_through_truncation" if args.continue_through_truncation else "finite_to_truncation",
            "stratified_collection": bool(args.stratified_collection),
            "stratified_pool_multiplier": int(args.stratified_pool_multiplier),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
        },
        "state_bank_distribution": {
            "train": {
                feature: _feature_distribution(train_entries, feature=feature)
                for feature in group_feature_names
            },
            "holdout": {
                feature: _feature_distribution(holdout_entries, feature=feature)
                for feature in group_feature_names
            },
        },
        "target_summary": {
            "train_mc_std_mean": float(np.mean(train_mc_std)) if train_mc_std.size else 0.0,
            "train_mc_se_mean": float(np.mean(train_mc_se)) if train_mc_se.size else 0.0,
            "holdout_mc_std_mean": float(np.mean(holdout_mc_std)) if holdout_mc_std.size else 0.0,
            "holdout_mc_se_mean": float(np.mean(holdout_mc_se)) if holdout_mc_se.size else 0.0,
        },
        "online_critic": online_summary,
        "holdout_by_feature": holdout_group_summary,
        "loaded_critic_supervised_final": loaded_final,
        "fresh_critic_supervised_final": fresh_final,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
