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

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

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
    data["_search_distill_args"] = extra
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


def _uniform_action(valid_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(valid_mask, dtype=bool)
    out = np.zeros(mask.shape, dtype=np.float32)
    for u in range(mask.shape[0]):
        valid = np.flatnonzero(mask[u])
        if valid.size > 0:
            out[u, valid] = 1.0 / float(valid.size)
    return out


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
            sample = np.random.dirichlet(alpha).astype(np.float32)
            out[u, valid] = sample
        actions.append(out)
    return actions


def _snapshot_to_local_bw_state(snapshot: Any, device: torch.device):
    local_state = build_local_bw_states_from_snapshot(snapshot)[0]
    return _to_device_dataclass(local_state, device)


def _sample_policy_actions(
    actor,
    snapshot_record: dict[str, Any],
    *,
    device: torch.device,
    stochastic_samples: int,
    wide_dirichlet_samples: int,
    wide_dirichlet_concentration: float,
    include_det_mean: bool,
    include_uniform: bool,
) -> list[np.ndarray]:
    local_state = snapshot_record["local_state"]
    actions: list[np.ndarray] = []
    with torch.no_grad():
        if include_det_mean:
            det_out = actor.act_bw(local_state, deterministic=True)
            actions.append(np.asarray(det_out.action.detach().cpu().numpy(), dtype=np.float32))
        for _ in range(max(int(stochastic_samples), 0)):
            out = actor.act_bw(local_state, deterministic=False)
            actions.append(np.asarray(out.action.detach().cpu().numpy(), dtype=np.float32))
        if int(wide_dirichlet_samples) > 0:
            det_out = actor.act_bw(local_state, deterministic=True)
            det_mean = np.asarray(det_out.action.detach().cpu().numpy(), dtype=np.float32)
            valid_mask = np.asarray(local_state.bw_valid_mask.detach().cpu().numpy(), dtype=bool)
            actions.extend(
                _sample_wide_dirichlet_actions(
                    det_mean,
                    valid_mask,
                    num_samples=int(wide_dirichlet_samples),
                    concentration=float(wide_dirichlet_concentration),
                )
            )
    if include_uniform:
        valid_mask = np.asarray(local_state.bw_valid_mask.detach().cpu().numpy(), dtype=bool)
        actions.append(_uniform_action(valid_mask))
    return actions


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
            snapshots.append(
                {
                    "snapshot_state": snapshot_state,
                    "local_state": local_state,
                }
            )
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


def _distill_step(
    actor,
    optimizer: torch.optim.Optimizer,
    *,
    snapshot_records: list[dict[str, Any]],
    target_actions: list[np.ndarray],
    device: torch.device,
    batch_size: int,
    epochs: int,
) -> dict[str, float]:
    local_states = [record["local_state"] for record in snapshot_records]
    targets = [
        torch.as_tensor(action, dtype=torch.float32, device=device).reshape(1, -1)
        for action in target_actions
    ]
    valid_masks = [((state.user_mask > 0.5) & (state.bw_valid_mask > 0.5)).to(device=device) for state in local_states]
    order = np.arange(len(local_states), dtype=np.int64)
    loss_values: list[float] = []
    target_l1_values: list[float] = []
    if len(order) == 0:
        return {"distill_loss": 0.0, "target_l1": 0.0}
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
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--init_actor", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--states_per_update", type=int, default=8)
    parser.add_argument("--stochastic_candidates", type=int, default=16)
    parser.add_argument("--wide_dirichlet_candidates", type=int, default=0)
    parser.add_argument("--wide_dirichlet_concentration", type=float, default=0.25)
    parser.add_argument("--include_det_mean", action="store_true")
    parser.add_argument("--include_uniform", action="store_true")
    parser.add_argument("--collect_deterministic", action="store_true")
    parser.add_argument("--follow_deterministic", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
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
            "updates": int(args.updates),
            "states_per_update": int(args.states_per_update),
            "stochastic_candidates": int(args.stochastic_candidates),
            "wide_dirichlet_candidates": int(args.wide_dirichlet_candidates),
            "wide_dirichlet_concentration": float(args.wide_dirichlet_concentration),
            "include_det_mean": bool(args.include_det_mean),
            "include_uniform": bool(args.include_uniform),
            "collect_deterministic": bool(args.collect_deterministic),
            "follow_deterministic": bool(args.follow_deterministic),
            "batch_size": int(args.batch_size),
            "distill_epochs": int(args.distill_epochs),
            "actor_lr": float(args.actor_lr),
        },
    )

    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    actor = bundle.actor.to(device)
    load_checkpoint_forgiving(actor, str(args.init_actor), map_location=device, strict=True)
    optimizer = torch.optim.Adam(actor.parameters(), lr=float(args.actor_lr))

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

    metrics_rows: list[dict[str, float]] = []
    for update_idx in range(1, int(args.updates) + 1):
        actor.eval()
        snapshot_states = _collect_bw_snapshots(
            cfg,
            actor,
            device=device,
            num_states=int(args.states_per_update),
            seed=int(args.seed) + 1000 * update_idx,
            collect_deterministic=bool(args.collect_deterministic),
        )
        target_actions: list[np.ndarray] = []
        best_scores: list[float] = []
        current_scores: list[float] = []
        candidate_counts: list[int] = []
        target_from_det = 0
        target_from_uniform = 0
        target_from_wide = 0
        for snapshot_record in snapshot_states:
            candidates = _sample_policy_actions(
                actor,
                snapshot_record,
                device=device,
                stochastic_samples=int(args.stochastic_candidates),
                wide_dirichlet_samples=int(args.wide_dirichlet_candidates),
                wide_dirichlet_concentration=float(args.wide_dirichlet_concentration),
                include_det_mean=bool(args.include_det_mean),
                include_uniform=bool(args.include_uniform),
            )
            if not candidates:
                raise RuntimeError("No candidate actions generated for search-distill.")
            labels = [f"sample_{i}" for i in range(max(int(args.stochastic_candidates), 0))]
            if args.include_det_mean:
                labels = ["det_mean"] + labels
            labels.extend([f"wide_{i}" for i in range(max(int(args.wide_dirichlet_candidates), 0))])
            if args.include_uniform:
                labels.append("uniform")
            scores = [
                _run_bw_candidate_to_episode_end(
                    cfg,
                    actor,
                    snapshot_state=snapshot_record["snapshot_state"],
                    first_action=action,
                    device=device,
                    follow_deterministic=bool(args.follow_deterministic),
                )
                for action in candidates
            ]
            best_idx = int(np.argmax(np.asarray(scores, dtype=np.float64)))
            target_actions.append(np.asarray(candidates[best_idx], dtype=np.float32))
            best_scores.append(float(scores[best_idx]))
            current_scores.append(float(scores[0]))
            candidate_counts.append(int(len(candidates)))
            if labels[best_idx] == "det_mean":
                target_from_det += 1
            if labels[best_idx] == "uniform":
                target_from_uniform += 1
            if labels[best_idx].startswith("wide_"):
                target_from_wide += 1

        actor.train()
        distill_metrics = _distill_step(
            actor,
            optimizer,
            snapshot_records=snapshot_states,
            target_actions=target_actions,
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
        row = {
            "update": float(update_idx),
            "search_best_reward_mean": float(np.mean(np.asarray(best_scores, dtype=np.float64))),
            "search_current_reward_mean": float(np.mean(np.asarray(current_scores, dtype=np.float64))),
            "search_gap_mean": float(np.mean(np.asarray(best_scores, dtype=np.float64) - np.asarray(current_scores, dtype=np.float64))),
            "candidate_count_mean": float(np.mean(np.asarray(candidate_counts, dtype=np.float64))),
            "target_from_det_frac": float(target_from_det / max(len(snapshot_states), 1)),
            "target_from_wide_frac": float(target_from_wide / max(len(snapshot_states), 1)),
            "target_from_uniform_frac": float(target_from_uniform / max(len(snapshot_states), 1)),
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
            f"Update {update_idx:02d} | gap={row['search_gap_mean']:.3f} | "
            f"eval_reward={row['eval_reward_sum']:.3f} | processed={row['eval_processed_ratio']:.3f} | "
            f"drop={row['eval_drop_ratio']:.4f} | backlog={row['eval_pre_backlog']:.2f}"
        )

    torch.save(actor.state_dict(), run_dir / "actor_final.pt")
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "updates": int(args.updates),
                "states_per_update": int(args.states_per_update),
                "stochastic_candidates": int(args.stochastic_candidates),
                "wide_dirichlet_candidates": int(args.wide_dirichlet_candidates),
                "wide_dirichlet_concentration": float(args.wide_dirichlet_concentration),
                "include_det_mean": bool(args.include_det_mean),
                "include_uniform": bool(args.include_uniform),
                "collect_deterministic": bool(args.collect_deterministic),
                "follow_deterministic": bool(args.follow_deterministic),
                "final_metrics": metrics_rows[-1] if metrics_rows else {},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
