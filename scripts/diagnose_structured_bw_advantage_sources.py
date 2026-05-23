from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_checkpoint import load_structured_train_state
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO, _collate_dataclass
from sagin_marl.rl.structured_train import (
    _is_done,
    _looks_like_structured_driver,
    _looks_like_structured_driver_group,
    _normalize_env_group,
    _reset_env_at,
    close_structured_env_group,
    make_structured_env_group,
)
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


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


def _safe_corr(x: list[float], y: list[float]) -> float:
    if len(x) <= 1 or len(y) <= 1 or len(x) != len(y):
        return 0.0
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if float(np.std(xa)) <= 1.0e-12 or float(np.std(ya)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def _rankdata_avg(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    order = np.argsort(arr, kind="mergesort")
    ranks = np.zeros_like(arr, dtype=np.float64)
    i = 0
    while i < arr.size:
        j = i + 1
        while j < arr.size and arr[order[j]] == arr[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _safe_spearman(x: list[float], y: list[float]) -> float:
    if len(x) <= 1 or len(y) <= 1 or len(x) != len(y):
        return 0.0
    return _safe_corr(_rankdata_avg(x).tolist(), _rankdata_avg(y).tolist())


def _quantile_report(order_values: list[float], target_values: list[float], frac: float = 0.2) -> dict[str, float]:
    if not order_values or len(order_values) != len(target_values):
        return {"top_mean": 0.0, "bottom_mean": 0.0, "delta_top_minus_bottom": 0.0}
    n = len(order_values)
    k = max(1, int(round(float(frac) * n)))
    order = np.argsort(np.asarray(order_values, dtype=np.float64))
    target = np.asarray(target_values, dtype=np.float64)
    bottom = target[order[:k]]
    top = target[order[-k:]]
    top_mean = float(np.mean(top))
    bottom_mean = float(np.mean(bottom))
    return {
        "top_mean": top_mean,
        "bottom_mean": bottom_mean,
        "delta_top_minus_bottom": top_mean - bottom_mean,
    }


def _subset_split_report(
    subset_mask: np.ndarray,
    order_values: list[float],
    target_values: list[float],
    *,
    lower_is_flatter: bool,
    frac: float = 0.2,
) -> dict[str, float]:
    mask = np.asarray(subset_mask, dtype=bool)
    if int(np.sum(mask)) <= 1:
        return {"flatter_mean": 0.0, "sharper_mean": 0.0, "delta_flatter_minus_sharper": 0.0}
    order_arr = np.asarray(order_values, dtype=np.float64)[mask]
    target_arr = np.asarray(target_values, dtype=np.float64)[mask]
    n = int(order_arr.size)
    k = max(1, int(round(float(frac) * n)))
    order = np.argsort(order_arr)
    if lower_is_flatter:
        flatter = target_arr[order[:k]]
        sharper = target_arr[order[-k:]]
    else:
        flatter = target_arr[order[-k:]]
        sharper = target_arr[order[:k]]
    flatter_mean = float(np.mean(flatter))
    sharper_mean = float(np.mean(sharper))
    return {
        "flatter_mean": flatter_mean,
        "sharper_mean": sharper_mean,
        "delta_flatter_minus_sharper": flatter_mean - sharper_mean,
    }


def _build_learner_from_train_state(
    cfg,
    run_dir: Path,
    update: int,
    *,
    device: torch.device,
) -> tuple[StructuredMAPPO, dict[str, Any]]:
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    actor = bundle.actor.to(device)
    critic = bundle.critic.to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(getattr(cfg, "actor_lr", 1.0e-4) or 1.0e-4))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(getattr(cfg, "critic_lr", 1.0e-4) or 1.0e-4))
    state_path = run_dir / f"train_state_u{int(update):04d}.pt"
    meta = load_structured_train_state(
        str(state_path),
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        device=device,
    )
    actor.eval()
    critic.eval()
    learner = StructuredMAPPO(
        actor=actor,
        critic=critic,
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_ratio=float(getattr(cfg, "clip_ratio", 0.2) or 0.2),
        value_coef=float(getattr(cfg, "value_coef", 0.5) or 0.5),
        entropy_coef=float(getattr(cfg, "entropy_coef", 0.0) or 0.0),
        max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5) or 0.5),
        ppo_epochs=int(getattr(cfg, "ppo_epochs", 1) or 1),
        num_mini_batch=int(getattr(cfg, "num_mini_batch", 1) or 1),
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        device=device,
        target_mode="step_level",
        danger_imitation_enabled=bool(getattr(cfg, "danger_imitation_enabled", False)),
        danger_imitation_coef=float(getattr(cfg, "danger_imitation_coef", 0.0) or 0.0),
        cfg=cfg,
        train_accel=bool(True if getattr(cfg, "train_accel", None) is None else getattr(cfg, "train_accel")),
        train_sat=bool(True if getattr(cfg, "train_sat", None) is None else getattr(cfg, "train_sat")),
        train_bw=bool(True if getattr(cfg, "train_bw", None) is None else getattr(cfg, "train_bw")),
        exec_accel_source=str(getattr(cfg, "exec_accel_source", "policy") or "policy"),
        exec_sat_source=str(getattr(cfg, "exec_sat_source", "policy") or "policy"),
        exec_bw_source=str(getattr(cfg, "exec_bw_source", "policy") or "policy"),
    )
    return learner, meta


def _collect_training_rollout(
    cfg,
    learner: StructuredMAPPO,
    *,
    num_envs: int,
    vec_backend: str,
    rollout_env_steps: int,
    seed_base: int,
) -> StructuredRolloutBuffer:
    env_group = make_structured_env_group(cfg, num_envs=int(num_envs), backend=str(vec_backend))
    try:
        structured_group = env_group if _looks_like_structured_driver_group(env_group) else None
        if structured_group is not None:
            actual_num_envs = len(structured_group)
            structured_group.reset_many([int(seed_base) + env_index for env_index in range(actual_num_envs)])
            drivers = structured_group
            envs = None
        else:
            envs = _normalize_env_group(env_group)
            actual_num_envs = len(envs)
            for env_index, env in enumerate(envs):
                _reset_env_at(env, int(seed_base) + env_index)
            drivers = [
                as_structured_driver(env)
                for env in envs
            ]

        reset_counters = [0 for _ in range(actual_num_envs)]
        buffer = StructuredRolloutBuffer()
        for _ in range(int(rollout_env_steps)):
            results = learner.collect_env_steps(drivers, buffer, deterministic=False)
            for env_index, result in enumerate(results):
                if not _is_done(result):
                    continue
                reset_counters[env_index] += 1
                seed = int(seed_base) + reset_counters[env_index] * actual_num_envs + env_index
                if structured_group is not None:
                    structured_group.reset_at(env_index, seed)
                else:
                    if envs is None:
                        raise RuntimeError("envs should be materialized for non-group drivers")
                    _reset_env_at(envs[env_index], seed)
        return buffer
    finally:
        close_structured_env_group(env_group)


def _top2_margin(row: np.ndarray) -> float:
    if row.size <= 1:
        return 0.0
    top2 = np.partition(row, -2)[-2:]
    return float(np.max(top2) - np.min(top2))


def _row_shape_metrics(row: np.ndarray) -> dict[str, float]:
    k = int(row.size)
    if k <= 0:
        return {
            "valid_count": 0.0,
            "top1": 0.0,
            "top1_excess": 0.0,
            "top1_margin": 0.0,
            "l1_to_uniform": 0.0,
            "entropy_norm": 0.0,
            "entropy_gap_norm": 0.0,
        }
    uniform = np.full((k,), 1.0 / max(k, 1), dtype=np.float64)
    clipped = np.clip(np.asarray(row, dtype=np.float64), 1.0e-12, 1.0)
    top1 = float(np.max(clipped))
    entropy = float(-np.sum(clipped * np.log(clipped)))
    entropy_norm = 1.0 if k <= 1 else float(entropy / max(math.log(k), 1.0e-12))
    return {
        "valid_count": float(k),
        "top1": top1,
        "top1_excess": float(top1 - 1.0 / max(k, 1)),
        "top1_margin": _top2_margin(clipped),
        "l1_to_uniform": float(0.5 * np.sum(np.abs(clipped - uniform))),
        "entropy_norm": entropy_norm,
        "entropy_gap_norm": float(1.0 - entropy_norm),
    }


def _sample_shape_metrics(action: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    metric_lists: dict[str, list[float]] = {
        "valid_count": [],
        "top1": [],
        "top1_excess": [],
        "top1_margin": [],
        "l1_to_uniform": [],
        "entropy_norm": [],
        "entropy_gap_norm": [],
    }
    active_agents = 0
    for agent_idx in range(int(action.shape[0])):
        valid = np.asarray(valid_mask[agent_idx], dtype=bool)
        if int(np.sum(valid)) <= 1:
            continue
        row = np.asarray(action[agent_idx], dtype=np.float64)[valid]
        row_sum = float(np.sum(row))
        if row_sum <= 1.0e-12:
            continue
        row = row / row_sum
        row_metrics = _row_shape_metrics(row)
        for key, value in row_metrics.items():
            metric_lists[key].append(float(value))
        active_agents += 1
    out = {f"{key}_mean": _safe_mean(values) for key, values in metric_lists.items()}
    out["active_agent_count"] = float(active_agents)
    return out


def _history_tail(meta: dict[str, Any]) -> dict[str, float]:
    rows = [dict(row) for row in (meta.get("history_rows", []) or [])]
    if not rows:
        return {}
    tail = rows[-1]
    keep_keys = (
        "policy_loss",
        "value_loss",
        "entropy_bw",
        "approx_kl_bw",
        "clip_frac_bw",
        "bw_flow_proxy_aux_loss",
        "bw_flow_proxy_regression_loss",
        "bw_flow_proxy_pairwise_acc",
        "bw_grad_ratio_aux_to_policy",
        "bw_loc_head_grad_ratio_aux_to_policy",
    )
    return {
        key: float(tail.get(key, 0.0) or 0.0)
        for key in keep_keys
        if key in tail
    }


def _next_accel_values_for_stage2(
    return_view,
    stage_idx_np: np.ndarray,
    *,
    learner: StructuredMAPPO,
    bootstrap_world_states: dict[int, Any] | None,
) -> np.ndarray:
    stage_ids = np.asarray(return_view.stage_ids, dtype=np.int64)
    env_indices = np.asarray(return_view.env_indices, dtype=np.int64)
    values = np.asarray(return_view.values, dtype=np.float64)
    terminated = np.asarray(return_view.terminated, dtype=np.float64) > 0.5
    truncated = np.asarray(return_view.truncated, dtype=np.float64) > 0.5

    next_accel = np.zeros((stage_idx_np.size,), dtype=np.float64)
    env_to_next_boot = {
        int(env_index): float(learner.bootstrap_value(world_state))
        for env_index, world_state in (bootstrap_world_states or {}).items()
    }
    for row_idx, transition_idx in enumerate(stage_idx_np.tolist()):
        if terminated[transition_idx] or truncated[transition_idx]:
            next_accel[row_idx] = 0.0
            continue
        env_index = int(env_indices[transition_idx])
        found = False
        for follow_idx in range(int(transition_idx) + 1, len(stage_ids)):
            if int(env_indices[follow_idx]) == env_index and int(stage_ids[follow_idx]) == 0:
                next_accel[row_idx] = float(values[follow_idx])
                found = True
                break
        if not found:
            next_accel[row_idx] = float(env_to_next_boot.get(env_index, 0.0))
    return next_accel


def diagnose_update(
    run_dir: Path,
    update: int,
    *,
    num_envs: int,
    vec_backend: str,
    rollout_env_steps: int | None,
    seed_base: int,
    torch_seed_base: int,
    device: torch.device,
) -> dict[str, Any]:
    cfg = load_config(str(run_dir / "config_source.yaml"))
    rollout_steps = int(rollout_env_steps if rollout_env_steps is not None else getattr(cfg, "buffer_size", 50))
    _set_all_seeds(int(torch_seed_base) + int(update))
    learner, meta = _build_learner_from_train_state(cfg, run_dir, int(update), device=device)
    buffer = _collect_training_rollout(
        cfg,
        learner,
        num_envs=int(num_envs),
        vec_backend=str(vec_backend),
        rollout_env_steps=int(rollout_steps),
        seed_base=int(seed_base) + int(update) * 1000,
    )
    rollout_views = buffer.build_rollout_views(learner.device)
    bw_stage_batch = rollout_views.training_view.stage_batches.get(2)
    bootstrap_world_states = buffer.build_bootstrap_world_state_dict(rollout_views.bootstrap_view)

    if bw_stage_batch is None or int(bw_stage_batch.num_samples) <= 0:
        return {
            "update": int(update),
            "rollout_env_steps": int(rollout_steps),
            "num_envs": int(num_envs),
            "vec_backend": str(vec_backend),
            "bw_stage_samples": 0,
            "logged_metrics": _history_tail(meta),
        }

    gae_raw = learner.compute_returns_and_advantages(
        buffer,
        rollout_views.bootstrap_view,
        return_view=rollout_views.return_view,
    )
    returns_raw = np.asarray(gae_raw["returns"], dtype=np.float64)
    advantages_raw = np.asarray(gae_raw["advantages"], dtype=np.float64)
    advantages_norm = advantages_raw.copy()
    if advantages_norm.size > 1:
        advantages_norm = (advantages_norm - advantages_norm.mean()) / max(float(np.std(advantages_norm)), 1.0e-8)

    stage_idx_np = np.asarray(bw_stage_batch.transition_indices, dtype=np.int64)
    stage_actions = bw_stage_batch.actions.detach().cpu().numpy().astype(np.float32, copy=False)
    num_samples = int(bw_stage_batch.num_samples)
    num_agents = int(bw_stage_batch.num_agents)
    flat_local_batch = bw_stage_batch.local_batch
    valid_masks = (
        ((flat_local_batch.user_mask > 0.5) & (flat_local_batch.bw_valid_mask > 0.5))
        .reshape(num_samples, num_agents, -1)
        .detach()
        .cpu()
        .numpy()
    )

    sample_metrics_by_name: dict[str, list[float]] = {}
    reward_list: list[float] = []
    target_list: list[float] = []
    advantage_raw_list: list[float] = []
    advantage_norm_list: list[float] = []
    value_list: list[float] = []
    bootstrap_total_list: list[float] = []
    next_value_term_list: list[float] = []
    tail_term_list: list[float] = []

    rewards_all = np.asarray(rollout_views.return_view.rewards, dtype=np.float64)
    values_all = np.asarray(rollout_views.return_view.values, dtype=np.float64)
    next_accel_values = _next_accel_values_for_stage2(
        rollout_views.return_view,
        stage_idx_np,
        learner=learner,
        bootstrap_world_states=bootstrap_world_states,
    )
    next_value_terms = float(cfg.gamma) * (1.0 - float(cfg.gae_lambda)) * next_accel_values

    for valid_mask, action, stage_transition_idx, next_value_term in zip(
        valid_masks,
        stage_actions,
        stage_idx_np.tolist(),
        next_value_terms.tolist(),
    ):
        sample_metrics = _sample_shape_metrics(action, valid_mask)
        if float(sample_metrics["active_agent_count"]) <= 0.0:
            continue
        target = float(returns_raw[stage_transition_idx])
        reward = float(rewards_all[stage_transition_idx])
        value = float(values_all[stage_transition_idx])
        bootstrap_total = float(target - reward)
        tail_term = float(bootstrap_total - next_value_term)
        advantage_raw = float(advantages_raw[stage_transition_idx])
        advantage_norm = float(advantages_norm[stage_transition_idx])

        reward_list.append(reward)
        target_list.append(target)
        advantage_raw_list.append(advantage_raw)
        advantage_norm_list.append(advantage_norm)
        value_list.append(value)
        bootstrap_total_list.append(bootstrap_total)
        next_value_term_list.append(float(next_value_term))
        tail_term_list.append(tail_term)
        for key, value_metric in sample_metrics.items():
            sample_metrics_by_name.setdefault(key, []).append(float(value_metric))

    component_series = {
        "reward": reward_list,
        "bootstrap_total": bootstrap_total_list,
        "next_value_term": next_value_term_list,
        "tail_term": tail_term_list,
        "target": target_list,
        "value": value_list,
        "advantage_raw": advantage_raw_list,
        "advantage_norm": advantage_norm_list,
    }

    flatness_metric_names = [
        ("l1_to_uniform_mean", True),
        ("entropy_norm_mean", False),
        ("top1_excess_mean", True),
        ("top1_margin_mean", True),
    ]
    component_relations: dict[str, Any] = {}
    high_adv_mask = np.asarray(advantage_norm_list, dtype=np.float64) >= float(
        np.percentile(np.asarray(advantage_norm_list, dtype=np.float64), 80.0)
    )

    for metric_name, lower_is_flatter in flatness_metric_names:
        metric_values = sample_metrics_by_name.get(metric_name, [])
        row: dict[str, Any] = {
            "metric_summary": _summarize(metric_values),
            "corr_with_advantage_norm_pearson": _safe_corr(metric_values, advantage_norm_list),
            "corr_with_advantage_norm_spearman": _safe_spearman(metric_values, advantage_norm_list),
            "top20_vs_bottom20_by_advantage_norm": _quantile_report(advantage_norm_list, metric_values, frac=0.2),
            "within_high_advantage_flatter_vs_sharper": {},
            "component_correlations": {},
        }
        for component_name, component_values in component_series.items():
            row["component_correlations"][component_name] = {
                "pearson": _safe_corr(metric_values, component_values),
                "spearman": _safe_spearman(metric_values, component_values),
            }
            row["within_high_advantage_flatter_vs_sharper"][component_name] = _subset_split_report(
                high_adv_mask,
                metric_values,
                component_values,
                lower_is_flatter=lower_is_flatter,
                frac=0.2,
            )
        component_relations[metric_name] = row

    top_adv_components = {
        component_name: _quantile_report(advantage_norm_list, component_values, frac=0.2)
        for component_name, component_values in component_series.items()
    }

    return {
        "update": int(update),
        "rollout_env_steps": int(rollout_steps),
        "num_envs": int(num_envs),
        "vec_backend": str(vec_backend),
        "bw_stage_samples": int(len(advantage_norm_list)),
        "logged_metrics": _history_tail(meta),
        "advantage_norm": _summarize(advantage_norm_list),
        "components": {name: _summarize(values) for name, values in component_series.items()},
        "top20_vs_bottom20_by_advantage_norm": top_adv_components,
        "flatness_component_relations": component_relations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--updates", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--vec-backend", type=str, default="sync")
    parser.add_argument("--rollout-env-steps", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=123450)
    parser.add_argument("--torch-seed-base", type=int, default=616161)
    parser.add_argument("--out-name", type=str, default="structured_bw_advantage_sources_diag.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "num_envs": int(args.num_envs),
        "vec_backend": str(args.vec_backend),
        "rollout_env_steps": None if args.rollout_env_steps is None else int(args.rollout_env_steps),
        "seed_base": int(args.seed_base),
        "torch_seed_base": int(args.torch_seed_base),
        "device": str(device),
        "updates": {},
    }
    for update in args.updates:
        update_summary = diagnose_update(
            run_dir,
            int(update),
            num_envs=int(args.num_envs),
            vec_backend=str(args.vec_backend),
            rollout_env_steps=args.rollout_env_steps,
            seed_base=int(args.seed_base),
            torch_seed_base=int(args.torch_seed_base),
            device=device,
        )
        summary["updates"][f"u{int(update):04d}"] = update_summary
        rel = update_summary.get("flatness_component_relations", {}).get("l1_to_uniform_mean", {})
        print(
            json.dumps(
                {
                    "update": int(update),
                    "bw_stage_samples": int(update_summary.get("bw_stage_samples", 0)),
                    "l1_corr_advantage_norm": float(rel.get("corr_with_advantage_norm_pearson", 0.0)),
                    "l1_corr_reward": float(rel.get("component_correlations", {}).get("reward", {}).get("pearson", 0.0)),
                    "l1_corr_bootstrap_total": float(
                        rel.get("component_correlations", {}).get("bootstrap_total", {}).get("pearson", 0.0)
                    ),
                    "l1_corr_value": float(rel.get("component_correlations", {}).get("value", {}).get("pearson", 0.0)),
                    "top20_adv_minus_bottom20_reward": float(
                        update_summary.get("top20_vs_bottom20_by_advantage_norm", {})
                        .get("reward", {})
                        .get("delta_top_minus_bottom", 0.0)
                    ),
                    "top20_adv_minus_bottom20_bootstrap_total": float(
                        update_summary.get("top20_vs_bottom20_by_advantage_norm", {})
                        .get("bootstrap_total", {})
                        .get("delta_top_minus_bottom", 0.0)
                    ),
                    "top20_adv_minus_bottom20_value": float(
                        update_summary.get("top20_vs_bottom20_by_advantage_norm", {})
                        .get("value", {})
                        .get("delta_top_minus_bottom", 0.0)
                    ),
                },
                ensure_ascii=False,
            )
        )
    out_path = run_dir / str(args.out_name)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
