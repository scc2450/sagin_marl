from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_critic import ZeroStructuredCritic
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import (
    StructuredMAPPO,
    _collate_dataclass,
    _index_dataclass,
    _to_cpu_tensor,
    _to_device_dataclass,
)
from sagin_marl.rl.structured_stage_builders import build_local_bw_states_from_snapshot
from sagin_marl.rl.structured_train import (
    _is_done,
    _looks_like_structured_driver,
    _looks_like_structured_driver_group,
    _normalize_env_group,
    _reset_env_at,
    close_structured_env_group,
    make_structured_env_group,
)
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _safe_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() <= 0 or b.numel() <= 0:
        return 0.0
    denom = float(a.norm().item()) * float(b.norm().item())
    if denom <= 1.0e-12:
        return 0.0
    return float(torch.dot(a, b).item() / denom)


def _flatten_grads(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> torch.Tensor:
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    flat_parts: list[torch.Tensor] = []
    for param, grad in zip(params, grads):
        if grad is None:
            flat_parts.append(torch.zeros_like(param, dtype=torch.float32).reshape(-1))
        else:
            flat_parts.append(grad.detach().to(dtype=torch.float32).reshape(-1))
    if not flat_parts:
        return torch.zeros((0,), dtype=torch.float32, device=loss.device)
    return torch.cat(flat_parts, dim=0)


def _pairwise_gradient_stats(grad_matrix: torch.Tensor) -> dict[str, float]:
    num_rows = int(grad_matrix.shape[0])
    if num_rows <= 0:
        return {
            "mean_pairwise_cos": 0.0,
            "neg_cos_frac": 0.0,
            "cancel_ratio": 0.0,
            "mean_grad_norm": 0.0,
            "batch_grad_norm": 0.0,
            "num_pairs": 0.0,
        }
    row_norms = grad_matrix.norm(dim=1)
    mean_grad = grad_matrix.mean(dim=0)
    cancel_ratio = float(
        mean_grad.norm().item() / max(float(row_norms.mean().item()), 1.0e-12)
    )
    if num_rows <= 1:
        return {
            "mean_pairwise_cos": 0.0,
            "neg_cos_frac": 0.0,
            "cancel_ratio": float(cancel_ratio),
            "mean_grad_norm": float(row_norms.mean().item()),
            "batch_grad_norm": float(mean_grad.norm().item()),
            "num_pairs": 0.0,
        }
    cosine_values: list[float] = []
    neg_count = 0
    pair_count = 0
    for left in range(num_rows):
        for right in range(left + 1, num_rows):
            pair_count += 1
            cos_value = _cosine(grad_matrix[left], grad_matrix[right])
            cosine_values.append(float(cos_value))
            if cos_value < 0.0:
                neg_count += 1
    return {
        "mean_pairwise_cos": float(np.mean(cosine_values, dtype=np.float64)) if cosine_values else 0.0,
        "neg_cos_frac": float(neg_count / max(pair_count, 1)),
        "cancel_ratio": float(cancel_ratio),
        "mean_grad_norm": float(row_norms.mean().item()),
        "batch_grad_norm": float(mean_grad.norm().item()),
        "num_pairs": float(pair_count),
    }


def _done_from_step_result(step_result: Any) -> bool:
    return bool(
        list(step_result.terminations.values())[0]
        or list(step_result.truncations.values())[0]
    )


def _zero_sat_action(cfg) -> np.ndarray:
    select_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 1)
    return np.full((cfg.num_uav, select_k), -1, dtype=np.int64)


def _det_bw_action_from_snapshot(actor, snapshot: Any, device: torch.device) -> np.ndarray:
    bw_states = build_local_bw_states_from_snapshot(snapshot)
    local_state = _to_device_dataclass(bw_states[0], device)
    with torch.inference_mode():
        out = actor.act_bw(local_state, deterministic=True)
    return np.asarray(out.action.detach().cpu().numpy(), dtype=np.float32).reshape(-1)


def _collect_panel_rows(
    *,
    cfg,
    actor,
    device: torch.device,
    panel_states: int,
    seed_base: int,
) -> tuple[list[dict[str, Any]], int]:
    env = make_structured_env(cfg, mode="script")
    driver = as_structured_driver(env)
    rows: list[dict[str, Any]] = []
    episode_count = 0
    try:
        while len(rows) < int(panel_states):
            env.reset(seed=int(seed_base) + int(episode_count))
            done = False
            while not done and len(rows) < int(panel_states):
                driver.begin_step()
                accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
                driver.run_accel_stage(accel_zero)
                z2 = driver.run_sat_stage(_zero_sat_action(cfg))
                snapshot = driver.build_bw_stage_snapshot(z2)
                panel_action = _det_bw_action_from_snapshot(actor, snapshot, device)
                rows.append(
                    {
                        "episode": int(episode_count),
                        "t": int(getattr(env, "t", 0)),
                        "snapshot_state": dict(driver.export_bw_stage_state() or {}),
                        "local_state": build_local_bw_states_from_snapshot(snapshot)[0],
                    }
                )
                step_result = driver.execute_stage_bw_and_step(
                    np.asarray(panel_action, dtype=np.float32).reshape(
                        int(cfg.num_uav),
                        int(cfg.users_obs_max),
                    )
                )
                done = _done_from_step_result(step_result)
            episode_count += 1
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
    return rows, int(episode_count)


def _make_learner(cfg, actor_ckpt: str, device: torch.device) -> tuple[Any, StructuredMAPPO]:
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64, build_critic=False)
    actor = bundle.actor.to(device)
    load_checkpoint_forgiving(actor, actor_ckpt, map_location=device, strict=True)
    learner = StructuredMAPPO(
        actor=actor,
        critic=ZeroStructuredCritic(device),
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_ratio=float(getattr(cfg, "clip_ratio", 0.2) or 0.2),
        value_coef=float(getattr(cfg, "value_coef", 0.5) or 0.5),
        entropy_coef=float(getattr(cfg, "entropy_coef", 0.0) or 0.0),
        max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5) or 0.5),
        ppo_epochs=int(getattr(cfg, "ppo_epochs", 1) or 1),
        num_mini_batch=int(getattr(cfg, "num_mini_batch", 1) or 1),
        actor_optimizer=torch.optim.Adam(
            actor.parameters(),
            lr=float(getattr(cfg, "actor_lr", 1.0e-3) or 1.0e-3),
        ),
        critic_optimizer=None,
        device=device,
        target_mode="step_level",
        cfg=cfg,
        train_accel=bool(getattr(cfg, "train_accel", True)),
        train_sat=bool(getattr(cfg, "train_sat", True)),
        train_bw=bool(getattr(cfg, "train_bw", True)),
        exec_accel_source=str(getattr(cfg, "exec_accel_source", "policy") or "policy"),
        exec_sat_source=str(getattr(cfg, "exec_sat_source", "policy") or "policy"),
        exec_bw_source=str(getattr(cfg, "exec_bw_source", "policy") or "policy"),
    )
    actor.eval()
    return actor, learner


def _collect_training_rollout(
    cfg,
    learner: StructuredMAPPO,
    *,
    num_envs: int,
    vec_backend: str,
    rollout_env_steps: int,
    reset_seed: int,
) -> StructuredRolloutBuffer:
    env_group = make_structured_env_group(cfg, int(num_envs), backend=str(vec_backend))
    try:
        structured_group = env_group if _looks_like_structured_driver_group(env_group) else None
        if structured_group is not None:
            actual_num_envs = len(structured_group)
            structured_group.reset_many([int(reset_seed) + env_index for env_index in range(actual_num_envs)])
            drivers = structured_group
            envs = None
        else:
            envs = _normalize_env_group(env_group)
            actual_num_envs = len(envs)
            for env_index, env in enumerate(envs):
                _reset_env_at(env, int(reset_seed) + env_index)
            drivers = [
                as_structured_driver(env)
                for env in envs
            ]

        rollout_deterministic = bool(getattr(learner, "bw_clean_per_user_enabled", False))
        reset_counters = [0 for _ in range(actual_num_envs)]
        buffer = StructuredRolloutBuffer()
        for _ in range(int(rollout_env_steps)):
            results = learner.collect_env_steps(drivers, buffer, deterministic=rollout_deterministic)
            for env_index, result in enumerate(results):
                if not _is_done(result):
                    continue
                reset_counters[env_index] += 1
                seed = int(reset_seed) + reset_counters[env_index] * actual_num_envs + env_index
                if structured_group is not None:
                    structured_group.reset_at(env_index, seed)
                else:
                    if envs is None:
                        raise RuntimeError("envs should be materialized for non-group drivers")
                    _reset_env_at(envs[env_index], seed)
        return buffer
    finally:
        close_structured_env_group(env_group)


def _sample_bw_rows_from_buffer(
    buffer: StructuredRolloutBuffer,
    *,
    max_states: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rollout_views = buffer.build_rollout_views(torch.device("cpu"))
    bw_stage_batch = rollout_views.training_view.stage_batches.get(2)
    if bw_stage_batch is None or int(bw_stage_batch.num_samples) <= 0:
        raise RuntimeError("No BW transitions collected from rollout buffer.")
    bw_idx_np = np.asarray(bw_stage_batch.transition_indices, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    if bw_idx_np.size > int(max_states):
        chosen_idx = np.sort(rng.choice(bw_idx_np, size=int(max_states), replace=False))
    else:
        chosen_idx = np.asarray(bw_idx_np, dtype=np.int64)
    transition_to_pos = {int(transition_idx): pos for pos, transition_idx in enumerate(bw_idx_np.tolist())}
    rows: list[dict[str, Any]] = []
    for idx in chosen_idx.tolist():
        pos = int(transition_to_pos[int(idx)])
        rows.append(
            {
                "buffer_index": int(idx),
                "snapshot_state": dict((bw_stage_batch.bw_stage_states or [None])[pos] or {}),
                "local_state": _index_dataclass(
                    bw_stage_batch.local_batch,
                    torch.as_tensor([pos], dtype=torch.long),
                ),
            }
        )
    return rows, {
        "bw_transitions_total": int(bw_idx_np.size),
        "bw_rows_selected": int(len(rows)),
    }


def _select_grad_params(actor, scope: str) -> tuple[list[torch.nn.Parameter], str]:
    bw_policy = getattr(actor, "bw_policy", None)
    if bw_policy is None:
        raise RuntimeError("Structured actor is missing bw_policy.")
    scope_l = str(scope).strip().lower()
    module_map = {
        "loc_head": "loc_head",
        "ego_encoder": "ego_encoder",
        "sat_encoder": "sat_encoder",
        "sat_refine": "sat_refine",
        "query_proj_1": "query_proj_1",
        "query_proj_2": "query_proj_2",
        "user_encoder": "user_encoder",
        "user_refine": "user_refine",
        "user_fusion": "user_fusion",
    }
    module_name = module_map.get(scope_l)
    if module_name is not None:
        module = getattr(bw_policy, module_name, None)
        if module is None:
            params = [param for param in bw_policy.parameters() if param.requires_grad]
            return params, "bw_policy_fallback"
        params = [param for param in module.parameters() if param.requires_grad]
        if not params:
            raise RuntimeError(f"bw_policy.{module_name} has no trainable parameters.")
        return params, f"bw_policy.{module_name}"
    if scope_l == "bw_policy":
        params = [param for param in bw_policy.parameters() if param.requires_grad]
        if not params:
            raise RuntimeError("bw_policy has no trainable parameters.")
        return params, "bw_policy"
    raise ValueError(f"Unsupported --param_scope: {scope}")


def _compute_target_beats_flags(
    *,
    learner: StructuredMAPPO,
    cfg,
    snapshot_states: list[dict[str, Any]],
    ref_actions: np.ndarray,
    target_actions: np.ndarray,
    valid_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    num_samples = int(target_actions.shape[0])
    valid_mask_f = np.asarray(valid_masks, dtype=np.float32)
    target_gap = np.sum(
        np.abs(np.asarray(target_actions, dtype=np.float32) - np.asarray(ref_actions, dtype=np.float32)) * valid_mask_f,
        axis=-1,
        dtype=np.float64,
    )
    target_beats_flags = np.zeros((num_samples,), dtype=np.float32)
    active_target_idx = np.flatnonzero(target_gap > 1.0e-8)
    if active_target_idx.size <= 0:
        return target_beats_flags, target_gap.astype(np.float32, copy=False)
    ref_returns_np = getattr(learner, "_bw_clean_last_ref_returns", None)
    if ref_returns_np is None or int(len(ref_returns_np)) != num_samples:
        ref_returns_np = np.full((num_samples,), np.nan, dtype=np.float32)
    else:
        ref_returns_np = np.asarray(ref_returns_np, dtype=np.float32).copy()
    missing_ref_mask = ~np.isfinite(ref_returns_np[active_target_idx])
    if bool(np.any(missing_ref_mask)):
        missing_idx = active_target_idx[missing_ref_mask]
        ref_returns_missing = learner._bw_clean_rollout_returns_parallel(
            snapshot_states=[dict(snapshot_states[int(idx)] or {}) for idx in missing_idx.tolist()],
            first_actions=[
                np.asarray(ref_actions[int(idx)], dtype=np.float32).reshape(
                    int(cfg.num_uav),
                    int(cfg.users_obs_max),
                )
                for idx in missing_idx.tolist()
            ],
        )
        ref_returns_np[missing_idx] = np.asarray(ref_returns_missing, dtype=np.float32)
    target_returns = learner._bw_clean_rollout_returns_parallel(
        snapshot_states=[dict(snapshot_states[int(idx)] or {}) for idx in active_target_idx.tolist()],
        first_actions=[
            np.asarray(target_actions[int(idx)], dtype=np.float32).reshape(
                int(cfg.num_uav),
                int(cfg.users_obs_max),
            )
            for idx in active_target_idx.tolist()
        ],
    )
    target_beats_flags[active_target_idx] = (
        np.asarray(target_returns, dtype=np.float32) > (ref_returns_np[active_target_idx] + 1.0e-6)
    ).astype(np.float32, copy=False)
    return target_beats_flags, target_gap.astype(np.float32, copy=False)


def _prepare_clean_dataset(
    *,
    learner: StructuredMAPPO,
    cfg,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    local_states = [row["local_state"] for row in rows]
    snapshot_states = [dict(row["snapshot_state"] or {}) for row in rows]
    local_batch = _collate_dataclass(local_states, learner.device)
    with torch.inference_mode():
        ref_out = learner.actor.act_bw(local_batch, deterministic=True)
    ref_actions = _to_cpu_tensor(ref_out.action).numpy().astype(np.float32, copy=False)
    valid_masks = (local_batch.user_mask > 0.5) & (local_batch.bw_valid_mask > 0.5)
    valid_masks_np = _to_cpu_tensor(valid_masks).numpy().astype(bool, copy=False)
    target_actions_np, rho_values_np, utility_l1_values_np = learner._bw_clean_target_actions_parallel(
        snapshot_states=[dict(state or {}) for state in snapshot_states],
        ref_actions=ref_actions,
        valid_masks=valid_masks_np,
    )
    target_beats_flags, target_gap_np = _compute_target_beats_flags(
        learner=learner,
        cfg=cfg,
        snapshot_states=snapshot_states,
        ref_actions=ref_actions,
        target_actions=np.asarray(target_actions_np, dtype=np.float32),
        valid_masks=valid_masks_np,
    )
    train_idx_np = np.flatnonzero(target_beats_flags > 0.5)
    target_actions_t = torch.as_tensor(
        np.asarray(target_actions_np, dtype=np.float32),
        dtype=torch.float32,
        device=learner.device,
    )
    return {
        "local_batch": local_batch,
        "valid_masks": valid_masks.to(device=learner.device),
        "target_actions": target_actions_t,
        "target_beats_flags": np.asarray(target_beats_flags, dtype=np.float32),
        "train_idx_np": np.asarray(train_idx_np, dtype=np.int64),
        "target_gap_np": np.asarray(target_gap_np, dtype=np.float32),
        "rho_values_np": np.asarray(rho_values_np, dtype=np.float32),
        "utility_l1_values_np": np.asarray(utility_l1_values_np, dtype=np.float32),
    }


def _dataset_gradient_stats(
    *,
    learner: StructuredMAPPO,
    cfg,
    rows: list[dict[str, Any]],
    params: list[torch.nn.Parameter],
    label: str,
) -> dict[str, Any]:
    prepared = _prepare_clean_dataset(learner=learner, cfg=cfg, rows=rows)
    train_idx_np = np.asarray(prepared["train_idx_np"], dtype=np.int64)
    if train_idx_np.size <= 0:
        return {
            "label": str(label),
            "rows_total": int(len(rows)),
            "train_samples": 0,
            "clean_target_beats_ref_frac": float(np.mean(prepared["target_beats_flags"], dtype=np.float64))
            if len(rows) > 0
            else 0.0,
            "target_gap": _safe_summary(np.asarray(prepared["target_gap_np"], dtype=np.float32)),
            "rho": _safe_summary(np.asarray(prepared["rho_values_np"], dtype=np.float32)),
            "utility_l1": _safe_summary(np.asarray(prepared["utility_l1_values_np"], dtype=np.float32)),
            "batch_loss": 0.0,
            "mean_pairwise_cos": 0.0,
            "neg_cos_frac": 0.0,
            "cancel_ratio": 0.0,
            "mean_grad_norm": 0.0,
            "batch_grad_norm": 0.0,
            "num_pairs": 0.0,
            "batch_grad": torch.zeros((0,), dtype=torch.float32, device=learner.device),
        }
    train_idx_t = torch.as_tensor(train_idx_np, dtype=torch.long, device=learner.device)
    local_batch_train = _index_dataclass(prepared["local_batch"], train_idx_t)
    target_actions_train = prepared["target_actions"].index_select(0, train_idx_t)
    valid_masks_train = prepared["valid_masks"].index_select(0, train_idx_t)

    learner.actor.eval()
    learner.actor.zero_grad(set_to_none=True)
    actor_out = learner.actor.act_bw(local_batch_train, deterministic=True)
    batch_loss = learner._bw_clean_huber_loss(actor_out.action, target_actions_train, valid_masks_train)
    batch_grad = _flatten_grads(batch_loss, params)

    per_sample_grads: list[torch.Tensor] = []
    for sample_pos in range(int(train_idx_np.size)):
        sample_idx_t = torch.as_tensor([sample_pos], dtype=torch.long, device=learner.device)
        local_state_i = _index_dataclass(local_batch_train, sample_idx_t)
        target_i = target_actions_train.index_select(0, sample_idx_t)
        valid_i = valid_masks_train.index_select(0, sample_idx_t)
        learner.actor.zero_grad(set_to_none=True)
        actor_out_i = learner.actor.act_bw(local_state_i, deterministic=True)
        loss_i = learner._bw_clean_huber_loss(actor_out_i.action, target_i, valid_i)
        per_sample_grads.append(_flatten_grads(loss_i, params))
    grad_matrix = torch.stack(per_sample_grads, dim=0)
    pairwise_stats = _pairwise_gradient_stats(grad_matrix)
    return {
        "label": str(label),
        "rows_total": int(len(rows)),
        "train_samples": int(train_idx_np.size),
        "clean_target_beats_ref_frac": float(np.mean(prepared["target_beats_flags"], dtype=np.float64)),
        "target_gap": _safe_summary(np.asarray(prepared["target_gap_np"], dtype=np.float32)),
        "rho": _safe_summary(np.asarray(prepared["rho_values_np"], dtype=np.float32)),
        "utility_l1": _safe_summary(np.asarray(prepared["utility_l1_values_np"], dtype=np.float32)),
        "batch_loss": float(batch_loss.item()),
        "mean_pairwise_cos": float(pairwise_stats["mean_pairwise_cos"]),
        "neg_cos_frac": float(pairwise_stats["neg_cos_frac"]),
        "cancel_ratio": float(pairwise_stats["cancel_ratio"]),
        "mean_grad_norm": float(pairwise_stats["mean_grad_norm"]),
        "batch_grad_norm": float(pairwise_stats["batch_grad_norm"]),
        "num_pairs": float(pairwise_stats["num_pairs"]),
        "batch_grad": batch_grad,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--actor_checkpoint", required=True)
    parser.add_argument("--batch_states", type=int, default=24)
    parser.add_argument("--panel_states", type=int, default=24)
    parser.add_argument("--rollout_env_steps", type=int, default=80)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--vec_backend", choices=["sync", "subproc"], default="sync")
    parser.add_argument("--batch_seed", type=int, default=51000)
    parser.add_argument("--panel_seed", type=int, default=52000)
    parser.add_argument("--sample_seed", type=int, default=53000)
    parser.add_argument(
        "--param_scope",
        default="loc_head",
        help=(
            "Which BW actor parameter block to measure. Supported: "
            "loc_head, ego_encoder, sat_encoder, sat_refine, query_proj_1, query_proj_2, "
            "user_encoder, user_refine, user_fusion, bw_policy"
        ),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--json_out", type=str, default=None)
    args = parser.parse_args()

    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    cfg = load_config(str(args.config))
    device = torch.device(args.device)
    _set_all_seeds(int(getattr(cfg, "seed", 0) or 0))

    actor, learner = _make_learner(cfg, str(args.actor_checkpoint), device)
    params, scope_used = _select_grad_params(actor, str(args.param_scope))

    rollout_buffer = _collect_training_rollout(
        cfg,
        learner,
        num_envs=int(args.num_envs),
        vec_backend=str(args.vec_backend),
        rollout_env_steps=int(args.rollout_env_steps),
        reset_seed=int(args.batch_seed),
    )
    batch_rows, batch_collection = _sample_bw_rows_from_buffer(
        rollout_buffer,
        max_states=int(args.batch_states),
        seed=int(args.sample_seed),
    )
    panel_rows, panel_episode_count = _collect_panel_rows(
        cfg=cfg,
        actor=actor,
        device=device,
        panel_states=int(args.panel_states),
        seed_base=int(args.panel_seed),
    )

    batch_stats = _dataset_gradient_stats(
        learner=learner,
        cfg=cfg,
        rows=batch_rows,
        params=params,
        label="online_batch",
    )
    panel_stats = _dataset_gradient_stats(
        learner=learner,
        cfg=cfg,
        rows=panel_rows,
        params=params,
        label="fixed_panel",
    )

    batch_grad = batch_stats.pop("batch_grad")
    panel_grad = panel_stats.pop("batch_grad")
    payload = {
        "config": str(args.config),
        "actor_checkpoint": str(args.actor_checkpoint),
        "device": str(device),
        "param_scope_requested": str(args.param_scope),
        "param_scope_used": str(scope_used),
        "param_count": int(sum(int(param.numel()) for param in params)),
        "collection": {
            "batch_seed": int(args.batch_seed),
            "panel_seed": int(args.panel_seed),
            "sample_seed": int(args.sample_seed),
            "rollout_env_steps": int(args.rollout_env_steps),
            "num_envs": int(args.num_envs),
            "vec_backend": str(args.vec_backend),
            "batch": batch_collection,
            "panel_episodes": int(panel_episode_count),
        },
        "cross": {
            "batch_vs_panel_cos": float(_cosine(batch_grad, panel_grad)),
            "batch_grad_norm": float(batch_grad.norm().item()) if batch_grad.numel() > 0 else 0.0,
            "panel_grad_norm": float(panel_grad.norm().item()) if panel_grad.numel() > 0 else 0.0,
        },
        "batch": batch_stats,
        "panel": panel_stats,
    }

    json_text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(json_text)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_text, encoding="utf-8")


if __name__ == "__main__":
    main()
