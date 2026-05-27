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
import yaml

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.structured_eval import (
    evaluate_structured_actor_exec_sources,
    evaluate_structured_fixed_policy,
)
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import _collate_dataclass, _to_device_dataclass
from sagin_marl.rl.structured_stage_builders import build_local_bw_states_from_snapshot
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
        raise RuntimeError("CUDA was requested via --device, but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def _save_config(run_dir: Path, cfg, config_path: str, extra: dict[str, Any]) -> None:
    data = asdict(cfg)
    data["_config_source"] = config_path
    data["_sequence_distill_args"] = extra
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    src = Path(config_path)
    if src.is_file():
        (run_dir / "config_source.yaml").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def _write_metrics_csv(run_dir: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _masked_simplex_kl(target: torch.Tensor, pred: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    eps = 1.0e-8
    target_valid = torch.where(valid_mask, target.clamp_min(eps), torch.zeros_like(target))
    pred_valid = torch.where(valid_mask, pred.clamp_min(eps), torch.zeros_like(pred))
    target_valid = target_valid / target_valid.sum(dim=-1, keepdim=True).clamp_min(eps)
    pred_valid = pred_valid / pred_valid.sum(dim=-1, keepdim=True).clamp_min(eps)
    kl = target_valid * (torch.log(target_valid.clamp_min(eps)) - torch.log(pred_valid.clamp_min(eps)))
    return (kl * valid_mask.to(dtype=kl.dtype)).sum(dim=-1)


def _sample_wide_dirichlet_action(
    det_mean: np.ndarray,
    valid_mask: np.ndarray,
    *,
    concentration: float,
) -> np.ndarray:
    mean = np.asarray(det_mean, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    conc = max(float(concentration), 1.0e-3)
    out = np.zeros_like(mean, dtype=np.float32)
    for u in range(mask.shape[0]):
        valid = np.flatnonzero(mask[u])
        if valid.size == 0:
            continue
        row_mean = np.asarray(mean[u, valid], dtype=np.float64)
        row_mean = np.clip(row_mean, 1.0e-6, None)
        row_mean = row_mean / row_mean.sum()
        alpha = np.clip(row_mean * conc, 1.0e-3, None)
        sample = np.random.dirichlet(alpha).astype(np.float32)
        out[u, valid] = sample
    return out


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
            local_state = _snapshot_to_local_bw_state(bw_snapshot, device)
            snapshots.append(
                {
                    "snapshot_state": driver.export_bw_stage_state(),
                    "local_state": local_state,
                }
            )
            with torch.no_grad():
                out = actor.act_bw(local_state, deterministic=True)
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


def _sample_snapshot_bank(
    base_bank: list[dict[str, Any]],
    aux_bank: list[dict[str, Any]],
    dynamic_bank: list[dict[str, Any]],
    *,
    total_count: int,
    base_count: int,
    aux_count: int,
) -> tuple[list[dict[str, Any]], int, int]:
    total = max(int(total_count), 0)
    want_base = max(int(base_count), 0)
    want_aux = max(int(aux_count), 0)
    if total <= 0:
        return [], 0, 0, 0
    chosen: list[dict[str, Any]] = []
    base_pick = min(want_base, len(base_bank), total)
    aux_pick = min(want_aux, len(aux_bank), max(total - base_pick, 0))
    dyn_pick = min(max(total - base_pick - aux_pick, 0), len(dynamic_bank))
    if base_pick > 0:
        base_idx = np.random.choice(len(base_bank), size=base_pick, replace=False)
        chosen.extend([base_bank[int(i)] for i in base_idx])
    if aux_pick > 0:
        aux_idx = np.random.choice(len(aux_bank), size=aux_pick, replace=False)
        chosen.extend([aux_bank[int(i)] for i in aux_idx])
    if dyn_pick > 0:
        dyn_idx = np.random.choice(len(dynamic_bank), size=dyn_pick, replace=False)
        chosen.extend([dynamic_bank[int(i)] for i in dyn_idx])
    remaining = total - len(chosen)
    if remaining > 0:
        candidate_pools = [pool for pool in (base_bank, aux_bank, dynamic_bank) if len(pool) > 0]
        remaining_pool = max(candidate_pools, key=len) if candidate_pools else []
        if len(remaining_pool) > 0:
            replace = len(remaining_pool) < remaining
            extra_idx = np.random.choice(len(remaining_pool), size=remaining, replace=replace)
            chosen.extend([remaining_pool[int(i)] for i in extra_idx])
    np.random.shuffle(chosen)
    return chosen[:total], int(base_pick), int(aux_pick), int(dyn_pick)


def _rollout_sequence_candidate_to_episode_end(
    cfg,
    actor,
    *,
    snapshot_state: dict[str, Any],
    device: torch.device,
    sequence_horizon: int,
    follow_deterministic: bool,
    follow_actor=None,
    wide_dirichlet_concentration: float,
    collect_prefix: bool,
) -> dict[str, Any]:
    env = make_structured_env(cfg, mode="script")
    driver = as_structured_driver(env)
    total_reward = 0.0
    prefix_states: list[Any] = []
    prefix_actions: list[np.ndarray] = []
    bw_step_idx = 0
    try:
        driver.load_bw_stage_state(snapshot_state)
        while True:
            bw_snapshot = driver.build_bw_stage_snapshot()
            local_state = _snapshot_to_local_bw_state(bw_snapshot, device)
            with torch.no_grad():
                det_out = actor.act_bw(local_state, deterministic=True)
            det_mean = np.asarray(det_out.action.detach().cpu().numpy(), dtype=np.float32)
            if bw_step_idx < int(sequence_horizon):
                valid_mask = np.asarray(local_state.bw_valid_mask.detach().cpu().numpy(), dtype=bool)
                action = _sample_wide_dirichlet_action(
                    det_mean,
                    valid_mask,
                    concentration=float(wide_dirichlet_concentration),
                )
                if collect_prefix:
                    prefix_states.append(local_state)
                    prefix_actions.append(np.asarray(action, dtype=np.float32))
            else:
                rollout_actor = follow_actor if follow_actor is not None else actor
                if bool(follow_deterministic):
                    with torch.no_grad():
                        follow_out = rollout_actor.act_bw(local_state, deterministic=True)
                    action = np.asarray(follow_out.action.detach().cpu().numpy(), dtype=np.float32)
                else:
                    with torch.no_grad():
                        sample_out = rollout_actor.act_bw(local_state, deterministic=False)
                    action = np.asarray(sample_out.action.detach().cpu().numpy(), dtype=np.float32)
            step_result, _ = driver.execute_stage_bw_and_prepare_next_accel(action)
            total_reward += float(next(iter(step_result.rewards.values())))
            bw_step_idx += 1
            done = bool(list(step_result.terminations.values())[0] or list(step_result.truncations.values())[0])
            if done:
                break
            driver.begin_step()
            accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
            _ = driver.run_accel_stage(accel_zero)
            select_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 1)
            _ = driver.run_sat_stage(np.full((cfg.num_uav, select_k), -1, dtype=np.int64))
        return {
            "reward": float(total_reward),
            "prefix_states": prefix_states,
            "prefix_actions": prefix_actions,
            "prefix_len": int(len(prefix_actions)),
            "bw_steps": int(bw_step_idx),
        }
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def _rollout_current_deterministic_sequence(
    cfg,
    actor,
    *,
    snapshot_state: dict[str, Any],
    device: torch.device,
    follow_actor=None,
) -> float:
    env = make_structured_env(cfg, mode="script")
    driver = as_structured_driver(env)
    total_reward = 0.0
    bw_step_idx = 0
    try:
        driver.load_bw_stage_state(snapshot_state)
        while True:
            bw_snapshot = driver.build_bw_stage_snapshot()
            local_state = _snapshot_to_local_bw_state(bw_snapshot, device)
            rollout_actor = follow_actor if (follow_actor is not None and bw_step_idx > 0) else actor
            with torch.no_grad():
                out = rollout_actor.act_bw(local_state, deterministic=True)
            action = np.asarray(out.action.detach().cpu().numpy(), dtype=np.float32)
            step_result, _ = driver.execute_stage_bw_and_prepare_next_accel(action)
            total_reward += float(next(iter(step_result.rewards.values())))
            bw_step_idx += 1
            done = bool(list(step_result.terminations.values())[0] or list(step_result.truncations.values())[0])
            if done:
                break
            driver.begin_step()
            accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
            _ = driver.run_accel_stage(accel_zero)
            select_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 1)
            _ = driver.run_sat_stage(np.full((cfg.num_uav, select_k), -1, dtype=np.int64))
        return float(total_reward)
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def _distill_sequence_step(
    actor,
    optimizer: torch.optim.Optimizer,
    *,
    sequence_pairs: list[tuple[Any, np.ndarray]],
    device: torch.device,
    batch_size: int,
    epochs: int,
) -> dict[str, float]:
    if not sequence_pairs:
        return {"distill_loss": 0.0, "target_l1": 0.0, "target_steps": 0.0}
    local_states = [pair[0] for pair in sequence_pairs]
    targets = [
        torch.as_tensor(pair[1], dtype=torch.float32, device=device).reshape(1, -1)
        for pair in sequence_pairs
    ]
    valid_masks = [((state.user_mask > 0.5) & (state.bw_valid_mask > 0.5)).to(device=device) for state in local_states]
    order = np.arange(len(local_states), dtype=np.int64)
    loss_values: list[float] = []
    target_l1_values: list[float] = []
    for _ in range(max(int(epochs), 1)):
        np.random.shuffle(order)
        for start in range(0, len(order), max(int(batch_size), 1)):
            mb_idx = order[start : start + max(int(batch_size), 1)]
            mb_states = [local_states[int(i)] for i in mb_idx]
            mb_targets = torch.cat([targets[int(i)] for i in mb_idx], dim=0)
            mb_mask = torch.cat([valid_masks[int(i)] for i in mb_idx], dim=0)
            if len(mb_states) == 1:
                batch_state = mb_states[0]
            else:
                batch_state = _collate_dataclass(mb_states, device)
            out = actor.act_bw(batch_state, deterministic=True)
            pred = out.det_mean if out.det_mean is not None else out.action
            loss = _masked_simplex_kl(mb_targets, pred, mb_mask).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            optimizer.step()
            loss_values.append(float(loss.detach().item()))
            target_l1 = ((pred.detach() - mb_targets).abs() * mb_mask.to(dtype=pred.dtype)).sum(dim=-1).mean()
            target_l1_values.append(float(target_l1.item()))
    return {
        "distill_loss": float(np.mean(np.asarray(loss_values, dtype=np.float64))) if loss_values else 0.0,
        "target_l1": float(np.mean(np.asarray(target_l1_values, dtype=np.float64))) if target_l1_values else 0.0,
        "target_steps": float(len(sequence_pairs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--init_actor", type=str, required=True)
    parser.add_argument("--follow_actor_checkpoint", type=str, default="")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--states_per_update", type=int, default=4)
    parser.add_argument("--reuse_initial_snapshots", action="store_true")
    parser.add_argument("--anchored_replay_bank", action="store_true")
    parser.add_argument("--base_bank_size", type=int, default=0)
    parser.add_argument("--base_bank_sample_size", type=int, default=0)
    parser.add_argument("--aux_bank_actor_checkpoint", type=str, default="")
    parser.add_argument("--aux_bank_size", type=int, default=0)
    parser.add_argument("--aux_bank_sample_size", type=int, default=0)
    parser.add_argument("--append_states_per_update", type=int, default=0)
    parser.add_argument("--dynamic_bank_capacity", type=int, default=0)
    parser.add_argument("--sequence_horizon", type=int, default=20)
    parser.add_argument("--sequence_candidates", type=int, default=8)
    parser.add_argument("--wide_dirichlet_concentration", type=float, default=0.25)
    parser.add_argument("--follow_deterministic", action="store_true")
    parser.add_argument("--improvement_margin", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--distill_epochs", type=int, default=4)
    parser.add_argument("--actor_lr", type=float, default=3.0e-4)
    parser.add_argument("--eval_episodes", type=int, default=8)
    parser.add_argument("--eval_seed_base", type=int, default=42000)
    args = parser.parse_args()

    _set_all_seeds(int(args.seed))
    device = _resolve_torch_device(args.device)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(args.config))
    _save_config(
        run_dir,
        cfg,
        os.path.abspath(args.config),
        {
            "init_actor": str(args.init_actor),
            "follow_actor_checkpoint": str(args.follow_actor_checkpoint),
            "updates": int(args.updates),
            "states_per_update": int(args.states_per_update),
            "reuse_initial_snapshots": bool(args.reuse_initial_snapshots),
            "anchored_replay_bank": bool(args.anchored_replay_bank),
            "base_bank_size": int(args.base_bank_size),
            "base_bank_sample_size": int(args.base_bank_sample_size),
            "aux_bank_actor_checkpoint": str(args.aux_bank_actor_checkpoint),
            "aux_bank_size": int(args.aux_bank_size),
            "aux_bank_sample_size": int(args.aux_bank_sample_size),
            "append_states_per_update": int(args.append_states_per_update),
            "dynamic_bank_capacity": int(args.dynamic_bank_capacity),
            "sequence_horizon": int(args.sequence_horizon),
            "sequence_candidates": int(args.sequence_candidates),
            "wide_dirichlet_concentration": float(args.wide_dirichlet_concentration),
            "follow_deterministic": bool(args.follow_deterministic),
            "improvement_margin": float(args.improvement_margin),
            "batch_size": int(args.batch_size),
            "distill_epochs": int(args.distill_epochs),
            "actor_lr": float(args.actor_lr),
        },
    )

    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    actor = bundle.actor.to(device)
    load_checkpoint_forgiving(actor, str(args.init_actor), map_location=device, strict=True)
    optimizer = torch.optim.Adam(actor.parameters(), lr=float(args.actor_lr))
    follow_actor = None
    if str(args.follow_actor_checkpoint).strip():
        follow_bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
        follow_actor = follow_bundle.actor.to(device)
        load_checkpoint_forgiving(
            follow_actor,
            str(args.follow_actor_checkpoint),
            map_location=device,
            strict=True,
        )
        follow_actor.eval()
    aux_bank_actor = None
    if str(args.aux_bank_actor_checkpoint).strip():
        aux_bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
        aux_bank_actor = aux_bundle.actor.to(device)
        load_checkpoint_forgiving(
            aux_bank_actor,
            str(args.aux_bank_actor_checkpoint),
            map_location=device,
            strict=True,
        )
        aux_bank_actor.eval()

    fixed_summary = evaluate_structured_fixed_policy(
        cfg,
        baseline_policy=str(getattr(cfg, "checkpoint_eval_fixed_policy", "queue_aware_bw")),
        episodes=int(args.eval_episodes),
        episode_seed_base=int(args.eval_seed_base),
    )
    print(
        "Fixed reference "
        f"{getattr(cfg, 'checkpoint_eval_fixed_policy', 'queue_aware_bw')}: "
        f"reward={fixed_summary['reward_sum']:.3f} | processed={fixed_summary['processed_ratio_eval']:.3f} | "
        f"drop={fixed_summary['drop_ratio_eval']:.4f} | backlog={fixed_summary['pre_backlog_steps_eval']:.2f}"
    )

    fixed_snapshot_records: list[dict[str, Any]] | None = None
    if bool(args.reuse_initial_snapshots):
        fixed_snapshot_records = _collect_bw_snapshots(
            cfg,
            actor,
            device=device,
            num_states=int(args.states_per_update),
            seed=int(args.seed) + 1000,
        )
    base_snapshot_bank: list[dict[str, Any]] = []
    aux_snapshot_bank: list[dict[str, Any]] = []
    dynamic_snapshot_bank: list[dict[str, Any]] = []
    if bool(args.anchored_replay_bank):
        base_bank_size = int(args.base_bank_size) if int(args.base_bank_size) > 0 else int(args.states_per_update)
        base_snapshot_bank = _collect_bw_snapshots(
            cfg,
            actor,
            device=device,
            num_states=base_bank_size,
            seed=int(args.seed) + 1000,
        )
        if aux_bank_actor is not None:
            aux_bank_size = int(args.aux_bank_size) if int(args.aux_bank_size) > 0 else max(int(args.states_per_update) // 2, 1)
            aux_snapshot_bank = _collect_bw_snapshots(
                cfg,
                aux_bank_actor,
                device=device,
                num_states=aux_bank_size,
                seed=int(args.seed) + 3000,
            )

    metrics_rows: list[dict[str, float]] = []
    for update_idx in range(1, int(args.updates) + 1):
        actor.eval()
        base_sample_count = 0
        aux_sample_count = 0
        dyn_sample_count = 0
        if bool(args.anchored_replay_bank):
            base_sample_target = int(args.base_bank_sample_size) if int(args.base_bank_sample_size) > 0 else int(args.states_per_update)
            aux_sample_target = int(args.aux_bank_sample_size) if int(args.aux_bank_sample_size) > 0 else 0
            snapshot_records, base_sample_count, aux_sample_count, dyn_sample_count = _sample_snapshot_bank(
                base_snapshot_bank,
                aux_snapshot_bank,
                dynamic_snapshot_bank,
                total_count=int(args.states_per_update),
                base_count=base_sample_target,
                aux_count=aux_sample_target,
            )
        elif fixed_snapshot_records is not None:
            snapshot_records = fixed_snapshot_records
        else:
            snapshot_records = _collect_bw_snapshots(
                cfg,
                actor,
                device=device,
                num_states=int(args.states_per_update),
                seed=int(args.seed) + 1000 * update_idx,
            )
        best_scores: list[float] = []
        current_scores: list[float] = []
        prefix_lengths: list[int] = []
        sequence_pairs: list[tuple[Any, np.ndarray]] = []
        positive_gaps: list[float] = []
        positive_gap_count = 0
        total_search_bw_steps = 0
        for snapshot_record in snapshot_records:
            current_score = _rollout_current_deterministic_sequence(
                cfg,
                actor,
                snapshot_state=snapshot_record["snapshot_state"],
                device=device,
                follow_actor=follow_actor,
            )
            current_scores.append(float(current_score))
            best_result: dict[str, Any] | None = None
            for candidate_idx in range(max(int(args.sequence_candidates), 1)):
                result = _rollout_sequence_candidate_to_episode_end(
                    cfg,
                    actor,
                    snapshot_state=snapshot_record["snapshot_state"],
                    device=device,
                    sequence_horizon=int(args.sequence_horizon),
                    follow_deterministic=bool(args.follow_deterministic),
                    follow_actor=follow_actor,
                    wide_dirichlet_concentration=float(args.wide_dirichlet_concentration),
                    collect_prefix=True,
                )
                total_search_bw_steps += int(result.get("bw_steps", 0))
                if best_result is None or float(result["reward"]) > float(best_result["reward"]):
                    best_result = result
            assert best_result is not None
            best_scores.append(float(best_result["reward"]))
            gap = float(best_result["reward"]) - float(current_score)
            if gap > float(args.improvement_margin):
                positive_gap_count += 1
                positive_gaps.append(gap)
                prefix_lengths.append(int(best_result["prefix_len"]))
                sequence_pairs.extend(list(zip(best_result["prefix_states"], best_result["prefix_actions"])))

        actor.train()
        distill_metrics = _distill_sequence_step(
            actor,
            optimizer,
            sequence_pairs=sequence_pairs,
            device=device,
            batch_size=int(args.batch_size),
            epochs=int(args.distill_epochs),
        )
        actor.eval()
        eval_summary, _ = evaluate_structured_actor_exec_sources(
            cfg,
            actor,
            device=device,
            episodes=int(args.eval_episodes),
            episode_seed_base=int(args.eval_seed_base),
            deterministic=True,
            num_envs=1,
            vec_backend="sync",
            exec_accel_source=str(getattr(cfg, "exec_accel_source", "policy")),
            exec_sat_source=str(getattr(cfg, "exec_sat_source", "policy")),
            exec_bw_source=str(getattr(cfg, "exec_bw_source", "policy")),
        )
        if bool(args.anchored_replay_bank) and int(args.append_states_per_update) > 0:
            appended = _collect_bw_snapshots(
                cfg,
                actor,
                device=device,
                num_states=int(args.append_states_per_update),
                seed=int(args.seed) + 200000 + 1000 * update_idx,
            )
            dynamic_snapshot_bank.extend(appended)
            cap = max(int(args.dynamic_bank_capacity), 0)
            if cap > 0 and len(dynamic_snapshot_bank) > cap:
                dynamic_snapshot_bank = dynamic_snapshot_bank[-cap:]
        row = {
            "update": float(update_idx),
            "sequence_best_reward_mean": float(np.mean(np.asarray(best_scores, dtype=np.float64))),
            "sequence_current_reward_mean": float(np.mean(np.asarray(current_scores, dtype=np.float64))),
            "sequence_gap_mean": float(
                np.mean(np.asarray(best_scores, dtype=np.float64) - np.asarray(current_scores, dtype=np.float64))
            ),
            "positive_gap_frac": float(positive_gap_count / max(len(snapshot_records), 1)),
            "mean_gap_given_positive": float(np.mean(np.asarray(positive_gaps, dtype=np.float64))) if positive_gaps else 0.0,
            "sequence_prefix_len_mean": float(np.mean(np.asarray(prefix_lengths, dtype=np.float64))) if prefix_lengths else 0.0,
            "teacher_search_bw_steps": float(total_search_bw_steps),
            "teacher_cost_per_gain": float(total_search_bw_steps / max(float(np.sum(np.asarray(positive_gaps, dtype=np.float64))), 1.0e-8)),
            "base_bank_size": float(len(base_snapshot_bank)),
            "aux_bank_size": float(len(aux_snapshot_bank)),
            "dynamic_bank_size": float(len(dynamic_snapshot_bank)),
            "base_bank_sample_count": float(base_sample_count),
            "aux_bank_sample_count": float(aux_sample_count),
            "dynamic_bank_sample_count": float(dyn_sample_count),
            "distill_target_steps": float(distill_metrics["target_steps"]),
            "distill_loss": float(distill_metrics["distill_loss"]),
            "distill_target_l1": float(distill_metrics["target_l1"]),
            "eval_reward_sum": float(eval_summary["reward_sum"]),
            "eval_processed_ratio": float(eval_summary["processed_ratio_eval"]),
            "eval_drop_ratio": float(eval_summary["drop_ratio_eval"]),
            "eval_pre_backlog": float(eval_summary["pre_backlog_steps_eval"]),
            "fixed_reward_sum": float(fixed_summary["reward_sum"]),
            "fixed_processed_ratio": float(fixed_summary["processed_ratio_eval"]),
            "fixed_drop_ratio": float(fixed_summary["drop_ratio_eval"]),
            "fixed_pre_backlog": float(fixed_summary["pre_backlog_steps_eval"]),
        }
        metrics_rows.append(row)
        _write_metrics_csv(run_dir, metrics_rows)
        torch.save(actor.state_dict(), run_dir / f"actor_u{update_idx:04d}.pt")
        print(
            f"Update {update_idx:02d} | seq_gap={row['sequence_gap_mean']:.3f} | "
            f"prefix_len={row['sequence_prefix_len_mean']:.1f} | eval_reward={row['eval_reward_sum']:.3f} | "
            f"processed={row['eval_processed_ratio']:.3f} | drop={row['eval_drop_ratio']:.4f} | "
            f"backlog={row['eval_pre_backlog']:.2f}"
        )

    torch.save(actor.state_dict(), run_dir / "actor_final.pt")
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "updates": int(args.updates),
                "states_per_update": int(args.states_per_update),
                "reuse_initial_snapshots": bool(args.reuse_initial_snapshots),
                "anchored_replay_bank": bool(args.anchored_replay_bank),
                "base_bank_size": int(args.base_bank_size),
                "base_bank_sample_size": int(args.base_bank_sample_size),
                "aux_bank_actor_checkpoint": str(args.aux_bank_actor_checkpoint),
                "aux_bank_size": int(args.aux_bank_size),
                "aux_bank_sample_size": int(args.aux_bank_sample_size),
                "append_states_per_update": int(args.append_states_per_update),
                "dynamic_bank_capacity": int(args.dynamic_bank_capacity),
                "sequence_horizon": int(args.sequence_horizon),
                "sequence_candidates": int(args.sequence_candidates),
                "wide_dirichlet_concentration": float(args.wide_dirichlet_concentration),
                "follow_deterministic": bool(args.follow_deterministic),
                "final_metrics": metrics_rows[-1] if metrics_rows else {},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
