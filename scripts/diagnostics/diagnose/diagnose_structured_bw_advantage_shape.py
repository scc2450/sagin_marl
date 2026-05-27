from __future__ import annotations

import argparse
import json
import math
import os
import random
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
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_checkpoint import load_structured_train_state
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO, _index_dataclass
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


def _quantile_report(advantages: list[float], values: list[float], frac: float = 0.2) -> dict[str, float]:
    if not advantages or len(advantages) != len(values):
        return {
            "top_mean": 0.0,
            "bottom_mean": 0.0,
            "delta_top_minus_bottom": 0.0,
        }
    n = len(advantages)
    k = max(1, int(round(float(frac) * n)))
    order = np.argsort(np.asarray(advantages, dtype=np.float64))
    arr = np.asarray(values, dtype=np.float64)
    bottom = arr[order[:k]]
    top = arr[order[-k:]]
    top_mean = float(np.mean(top))
    bottom_mean = float(np.mean(bottom))
    return {
        "top_mean": top_mean,
        "bottom_mean": bottom_mean,
        "delta_top_minus_bottom": top_mean - bottom_mean,
    }


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
    advantages_raw = torch.from_numpy(gae_raw["advantages"]).to(device)
    advantages_norm = advantages_raw.clone()
    if advantages_norm.numel() > 1:
        advantages_norm = (
            advantages_norm - advantages_norm.mean()
        ) / advantages_norm.std(unbiased=False).clamp_min(1.0e-8)

    stage_actions = bw_stage_batch.actions.detach().cpu().numpy().astype(np.float32, copy=False)
    num_samples = int(bw_stage_batch.num_samples)
    num_agents = int(bw_stage_batch.num_agents)
    flat_local_batch = bw_stage_batch.local_batch
    flat_indices = torch.arange(num_samples * num_agents, device=learner.device, dtype=torch.long).reshape(
        num_samples, num_agents
    )

    with torch.no_grad():
        mode_out = learner.actor.act_bw(flat_local_batch, deterministic=True)
    mode_actions = mode_out.action.reshape(num_samples, num_agents, -1).detach().cpu().numpy()
    valid_masks = (
        ((flat_local_batch.user_mask > 0.5) & (flat_local_batch.bw_valid_mask > 0.5))
        .reshape(num_samples, num_agents, -1)
        .detach()
        .cpu()
        .numpy()
    )

    stage_idx = torch.as_tensor(
        np.asarray(bw_stage_batch.transition_indices, dtype=np.int64),
        device=learner.device,
        dtype=torch.long,
    )
    adv_norm_bw = advantages_norm.index_select(
        0, stage_idx,
    ).detach().cpu().numpy()
    adv_raw_bw = advantages_raw.index_select(
        0, stage_idx,
    ).detach().cpu().numpy()

    sample_metrics_by_name: dict[str, list[float]] = {}
    mode_metrics_by_name: dict[str, list[float]] = {}
    valid_count_means: list[float] = []
    active_agent_counts: list[float] = []
    adv_norm_list: list[float] = []
    adv_raw_list: list[float] = []

    for sample_idx in range(num_samples):
        sampled_metrics = _sample_shape_metrics(stage_actions[sample_idx], valid_masks[sample_idx])
        mode_metrics = _sample_shape_metrics(mode_actions[sample_idx], valid_masks[sample_idx])
        if float(sampled_metrics["active_agent_count"]) <= 0.0:
            continue
        adv_norm_list.append(float(adv_norm_bw[sample_idx]))
        adv_raw_list.append(float(adv_raw_bw[sample_idx]))
        valid_count_means.append(float(sampled_metrics["valid_count_mean"]))
        active_agent_counts.append(float(sampled_metrics["active_agent_count"]))
        for key, value in sampled_metrics.items():
            sample_metrics_by_name.setdefault(key, []).append(float(value))
        for key, value in mode_metrics.items():
            mode_metrics_by_name.setdefault(key, []).append(float(value))

    def _metric_relation(values: list[float]) -> dict[str, Any]:
        return {
            "value_summary": _summarize(values),
            "corr_with_advantage_norm_pearson": _safe_corr(adv_norm_list, values),
            "corr_with_advantage_norm_spearman": _safe_spearman(adv_norm_list, values),
            "corr_with_advantage_raw_pearson": _safe_corr(adv_raw_list, values),
            "corr_with_advantage_raw_spearman": _safe_spearman(adv_raw_list, values),
            "top20_vs_bottom20_by_advantage_norm": _quantile_report(adv_norm_list, values, frac=0.2),
        }

    sampled_relations = {
        key: _metric_relation(values)
        for key, values in sample_metrics_by_name.items()
        if key not in {"active_agent_count"}
    }
    mode_relations = {
        key: _metric_relation(values)
        for key, values in mode_metrics_by_name.items()
        if key not in {"active_agent_count"}
    }

    return {
        "update": int(update),
        "rollout_env_steps": int(rollout_steps),
        "num_envs": int(num_envs),
        "vec_backend": str(vec_backend),
        "bw_stage_samples": int(len(adv_norm_list)),
        "num_agents": int(num_agents),
        "logged_metrics": _history_tail(meta),
        "advantage_norm": _summarize(adv_norm_list),
        "advantage_raw": _summarize(adv_raw_list),
        "valid_user_count_mean": _summarize(valid_count_means),
        "active_agent_count": _summarize(active_agent_counts),
        "sampled_action": sampled_relations,
        "mode_action": mode_relations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--updates", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--vec-backend", type=str, default="sync")
    parser.add_argument("--rollout-env-steps", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=123450)
    parser.add_argument("--torch-seed-base", type=int, default=515151)
    parser.add_argument("--out-name", type=str, default="structured_bw_advantage_shape_diag.json")
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
        sampled = update_summary.get("sampled_action", {})
        print(
            json.dumps(
                {
                    "update": int(update),
                    "bw_stage_samples": int(update_summary.get("bw_stage_samples", 0)),
                    "sampled_top1_excess_corr_adv_norm": float(
                        sampled.get("top1_excess_mean", {}).get("corr_with_advantage_norm_pearson", 0.0)
                    ),
                    "sampled_top1_margin_corr_adv_norm": float(
                        sampled.get("top1_margin_mean", {}).get("corr_with_advantage_norm_pearson", 0.0)
                    ),
                    "sampled_l1_to_uniform_corr_adv_norm": float(
                        sampled.get("l1_to_uniform_mean", {}).get("corr_with_advantage_norm_pearson", 0.0)
                    ),
                    "sampled_entropy_gap_corr_adv_norm": float(
                        sampled.get("entropy_gap_norm_mean", {}).get("corr_with_advantage_norm_pearson", 0.0)
                    ),
                    "mode_top1_margin_corr_adv_norm": float(
                        update_summary.get("mode_action", {})
                        .get("top1_margin_mean", {})
                        .get("corr_with_advantage_norm_pearson", 0.0)
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
