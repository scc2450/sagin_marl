from __future__ import annotations

import argparse
import copy
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
from sagin_marl.rl.structured_critic import ZeroStructuredCritic
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO, _collate_dataclass, _index_dataclass, _to_device_dataclass
from sagin_marl.rl.structured_stage_builders import build_local_bw_states_from_snapshot
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


def _masked_l1(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    mask_arr = np.asarray(mask, dtype=np.float32)
    if a_arr.ndim == 1:
        a_arr = a_arr.reshape(1, -1)
    else:
        a_arr = a_arr.reshape(a_arr.shape[0], -1)
    if b_arr.ndim == 1:
        b_arr = b_arr.reshape(1, -1)
    else:
        b_arr = b_arr.reshape(b_arr.shape[0], -1)
    if mask_arr.ndim == 1:
        mask_arr = mask_arr.reshape(1, -1)
    else:
        mask_arr = mask_arr.reshape(mask_arr.shape[0], -1)
    return np.sum(np.abs(a_arr - b_arr) * mask_arr, axis=-1, dtype=np.float64)


def _collect_panel_bank(
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
                z1 = driver.run_accel_stage(accel_zero)
                z2 = driver.run_sat_stage(_zero_sat_action(cfg))
                snapshot = driver.build_bw_stage_snapshot(z2)
                snapshot_state = driver.export_bw_stage_state()
                local_state = build_local_bw_states_from_snapshot(snapshot)[0]
                ref_action = _det_bw_action_from_snapshot(actor, snapshot, device)
                rows.append(
                    {
                        "episode": int(episode_count),
                        "t": int(getattr(env, "t", 0)),
                        "snapshot_state": dict(snapshot_state or {}),
                        "local_state": local_state,
                        "valid_mask": np.asarray(snapshot.bw_valid_mask, dtype=bool).reshape(-1),
                        "ref_action": np.asarray(ref_action, dtype=np.float32).reshape(-1),
                    }
                )
                step_result = driver.execute_stage_bw_and_step(
                    np.asarray(ref_action, dtype=np.float32).reshape(int(cfg.num_uav), int(cfg.users_obs_max))
                )
                done = _done_from_step_result(step_result)
            episode_count += 1
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
    return rows, int(episode_count)


def _make_learner(cfg, actor, device: torch.device, actor_lr: float) -> StructuredMAPPO:
    trainable_params = [param for param in actor.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable actor parameters remain after train-scope selection.")
    return StructuredMAPPO(
        actor=actor,
        critic=ZeroStructuredCritic(device),
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_ratio=float(getattr(cfg, "clip_ratio", 0.2) or 0.2),
        value_coef=float(getattr(cfg, "value_coef", 0.5) or 0.5),
        entropy_coef=float(getattr(cfg, "entropy_coef", 0.0) or 0.0),
        max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5) or 0.5),
        ppo_epochs=1,
        num_mini_batch=1,
        actor_optimizer=torch.optim.Adam(trainable_params, lr=float(actor_lr)),
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


def _configure_train_scope(actor, scope: str) -> str:
    scope_l = str(scope).strip().lower()
    for param in actor.parameters():
        param.requires_grad_(False)
    if scope_l in {"all", "actor", "full"}:
        for param in actor.parameters():
            param.requires_grad_(True)
        return "actor"
    bw_policy = getattr(actor, "bw_policy", None)
    if bw_policy is None:
        raise RuntimeError("Structured actor is missing bw_policy.")
    module_map = {
        "loc_head": "loc_head",
        "user_fusion": "user_fusion",
        "user_refine": "user_refine",
        "user_encoder": "user_encoder",
    }
    module_name = module_map.get(scope_l)
    if module_name is None:
        raise ValueError(f"Unsupported --train_scope: {scope}")
    module = getattr(bw_policy, module_name, None)
    if module is None:
        raise RuntimeError(f"bw_policy has no module {module_name!r}.")
    trainable_count = 0
    for param in module.parameters():
        param.requires_grad_(True)
        trainable_count += int(param.numel())
    if trainable_count <= 0:
        raise RuntimeError(f"No trainable parameters found for bw_policy.{module_name}.")
    return f"bw_policy.{module_name}"


def _build_actor_from_checkpoint(cfg, actor_checkpoint: str, device: torch.device):
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64, build_critic=False)
    actor = bundle.actor.to(device).eval()
    strict_load = str(getattr(cfg, "structured_bw_loc_readout", "fused") or "fused").strip().lower() not in {
        "user0_residual_fused",
        "fused_moe2",
        "fused_ctx_dot",
        "pairwise_comparator",
    }
    load_checkpoint_forgiving(actor, actor_checkpoint, map_location=device, strict=bool(strict_load))
    return actor


def _module_param_vector(module: torch.nn.Module) -> torch.Tensor:
    flat_parts = [param.detach().reshape(-1).cpu().to(dtype=torch.float32) for param in module.parameters()]
    if not flat_parts:
        return torch.zeros((0,), dtype=torch.float32)
    return torch.cat(flat_parts, dim=0)


def _refresh_bw_loc_head(
    *,
    actor,
    cfg,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    try:
        from sagin_marl.rl.structured_actor import BwLocReadoutHead
    except ImportError as exc:
        raise RuntimeError("Current structured actor does not expose the legacy BwLocReadoutHead.") from exc

    bw_policy = getattr(actor, "bw_policy", None)
    if bw_policy is None:
        raise RuntimeError("Structured actor is missing bw_policy.")
    old_head = getattr(bw_policy, "loc_head", None)
    if old_head is None:
        raise RuntimeError("BW policy is missing loc_head.")
    if not isinstance(old_head, BwLocReadoutHead):
        raise RuntimeError(f"Expected BwLocReadoutHead, got {type(old_head)!r}.")
    old_vec = _module_param_vector(old_head)
    embed_dim = int(old_head.local_head[0].in_features)
    hidden_dim = int(old_head.local_head[0].out_features)
    old_training = bool(old_head.training)
    _set_all_seeds(int(seed))
    new_head = BwLocReadoutHead(
        embed_dim=int(embed_dim),
        hidden_dim=int(hidden_dim),
        mode=str(getattr(old_head, "mode", getattr(cfg, "structured_bw_loc_readout", "fused"))),
        rule_dim=int(getattr(old_head, "rule_dim", getattr(cfg, "structured_bw_loc_rule_dim", 32)) or 32),
        residual_max_scale=float(
            getattr(old_head, "residual_max_scale", getattr(cfg, "structured_bw_loc_residual_max_scale", 0.2)) or 0.0
        ),
        residual_gate_init=float(getattr(cfg, "structured_bw_loc_residual_gate_init", -5.0) or -5.0),
    ).to(device)
    new_head.train(old_training)
    bw_policy.loc_head = new_head
    new_vec = _module_param_vector(new_head)
    if int(old_vec.numel()) != int(new_vec.numel()):
        raise RuntimeError("Fresh loc_head changed parameter count unexpectedly.")
    delta_vec = old_vec - new_vec
    return {
        "seed": float(int(seed)),
        "param_count": float(int(old_vec.numel())),
        "param_l1_delta": float(delta_vec.abs().sum().item()),
        "param_l2_delta": float(delta_vec.pow(2).sum().sqrt().item()),
        "old_param_l2": float(old_vec.pow(2).sum().sqrt().item()),
        "new_param_l2": float(new_vec.pow(2).sum().sqrt().item()),
    }


def _full_episode_return_from_snapshot(
    *,
    cfg,
    actor,
    device: torch.device,
    snapshot_state: dict[str, Any],
    first_action: np.ndarray,
) -> float:
    probe_env = make_structured_env(cfg, mode="script")
    probe_driver = as_structured_driver(probe_env)
    try:
        probe_driver.load_bw_stage_state(dict(snapshot_state or {}))
        total_reward = 0.0
        action = np.asarray(first_action, dtype=np.float32).reshape(int(cfg.num_uav), int(cfg.users_obs_max))
        while True:
            step_result, _next_world = probe_driver.execute_stage_bw_and_prepare_next_accel(action)
            total_reward += float(next(iter(step_result.rewards.values())))
            if _done_from_step_result(step_result):
                break
            probe_driver.begin_step()
            accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
            z1 = probe_driver.run_accel_stage(accel_zero)
            z2 = probe_driver.run_sat_stage(_zero_sat_action(cfg))
            next_snapshot = probe_driver.build_bw_stage_snapshot(z2)
            action = _det_bw_action_from_snapshot(actor, next_snapshot, device).reshape(int(cfg.num_uav), int(cfg.users_obs_max))
        return float(total_reward)
    finally:
        close_fn = getattr(probe_env, "close", None)
        if callable(close_fn):
            close_fn()


def _evaluate_panel(
    *,
    cfg,
    actor,
    device: torch.device,
    snapshot_states: list[dict[str, Any]],
    fixed_target_actions: np.ndarray,
) -> dict[str, Any]:
    current_returns: list[float] = []
    target_returns: list[float] = []
    for idx, snapshot_state in enumerate(snapshot_states):
        probe_env = make_structured_env(cfg, mode="script")
        probe_driver = as_structured_driver(probe_env)
        try:
            snapshot = probe_driver.load_bw_stage_state(dict(snapshot_state or {}))
            current_action = _det_bw_action_from_snapshot(actor, snapshot, device)
        finally:
            close_fn = getattr(probe_env, "close", None)
            if callable(close_fn):
                close_fn()
        current_returns.append(
            _full_episode_return_from_snapshot(
                cfg=cfg,
                actor=actor,
                device=device,
                snapshot_state=dict(snapshot_state or {}),
                first_action=current_action,
            )
        )
        target_returns.append(
            _full_episode_return_from_snapshot(
                cfg=cfg,
                actor=actor,
                device=device,
                snapshot_state=dict(snapshot_state or {}),
                    first_action=np.asarray(fixed_target_actions[idx], dtype=np.float32).reshape(-1),
                )
            )
    current_returns_np = np.asarray(current_returns, dtype=np.float32)
    target_returns_np = np.asarray(target_returns, dtype=np.float32)
    return {
        "current_return": _safe_summary(current_returns_np),
        "fixed_target_return": _safe_summary(target_returns_np),
        "target_minus_current": _safe_summary(np.asarray(target_returns_np - current_returns_np, dtype=np.float32)),
        "fixed_target_beats_current_frac": float(np.mean((target_returns_np > current_returns_np + 1.0e-6).astype(np.float32))),
    }


def _masked_centered_score_targets(
    target_action: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    mask = valid_mask.to(dtype=torch.bool)
    mask_f = mask.to(dtype=target_action.dtype)
    target_probs = torch.where(mask, target_action.clamp_min(0.0), torch.zeros_like(target_action))
    target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    target_log = torch.where(
        mask,
        torch.log(target_probs.clamp_min(float(eps))),
        torch.zeros_like(target_probs),
    )
    target_mean = (target_log * mask_f).sum(dim=-1, keepdim=True) / mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
    centered = target_log - target_mean
    return torch.where(mask, centered, torch.zeros_like(centered))


def _masked_center_scores(
    pred_score: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    mask = valid_mask.to(dtype=torch.bool)
    mask_f = mask.to(dtype=pred_score.dtype)
    safe_score = torch.where(mask, pred_score, torch.zeros_like(pred_score))
    score_mean = (safe_score * mask_f).sum(dim=-1, keepdim=True) / mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
    centered = safe_score - score_mean
    return torch.where(mask, centered, torch.zeros_like(centered))


def _score_reg_loss(
    pred_score: torch.Tensor,
    target_action: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    per_sample = _score_reg_loss_per_sample(
        pred_score=pred_score,
        target_action=target_action,
        valid_mask=valid_mask,
        eps=eps,
    )
    active = valid_mask.to(dtype=torch.bool).any(dim=-1)
    if not bool(torch.any(active)):
        return torch.zeros((), dtype=pred_score.dtype, device=pred_score.device)
    return per_sample[active].mean()


def _score_reg_loss_per_sample(
    *,
    pred_score: torch.Tensor,
    target_action: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    mask = valid_mask.to(dtype=torch.bool)
    mask_f = mask.to(dtype=pred_score.dtype)
    pred_centered = _masked_center_scores(pred_score, valid_mask)
    target_centered = _masked_centered_score_targets(target_action, valid_mask, eps=eps)
    loss = torch.nn.functional.huber_loss(pred_centered, target_centered, reduction="none")
    per_sample = (loss * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp_min(1.0)
    active = mask.any(dim=-1)
    return torch.where(active, per_sample, torch.zeros_like(per_sample))


def _offline_fit_one_step(
    *,
    learner: StructuredMAPPO,
    local_batch_train: Any,
    target_actions: torch.Tensor,
    valid_masks_train: torch.Tensor,
    fit_loss: str,
    sample_order_np: np.ndarray | None = None,
) -> dict[str, float]:
    with torch.inference_mode():
        old_out = learner.actor.act_bw(local_batch_train, deterministic=True)
    old_actions = old_out.action.detach()
    actor_out = learner.actor.act_bw(local_batch_train, deterministic=True)
    pred_action = actor_out.action
    pred_score = actor_out.score
    fit_loss_l = str(fit_loss).strip().lower()
    if fit_loss_l == "masked_kl":
        if pred_score is None:
            raise RuntimeError("BW fixed-teacher fit with masked_kl requires actor score output.")
        base_loss = learner._bw_clean_masked_kl_loss(pred_score, target_actions, valid_masks_train)
    elif fit_loss_l == "score_reg_only":
        if pred_score is None:
            raise RuntimeError("BW fixed-teacher fit with score_reg_only requires actor score output.")
        base_loss = _score_reg_loss(pred_score, target_actions, valid_masks_train)
    else:
        base_loss = learner._bw_clean_huber_loss(pred_action, target_actions, valid_masks_train)
    trust_kl = learner._bw_clean_simplex_kl(old_actions, pred_action, valid_masks_train)
    kl_coef = float(
        np.clip(
            float(getattr(learner, "_bw_clean_trust_region_kl_coef", learner.bw_clean_trust_region_kl_coef_init)),
            learner.bw_clean_trust_region_kl_coef_min,
            learner.bw_clean_trust_region_kl_coef_max,
        )
    )
    loss = base_loss
    if learner.bw_clean_trust_region_enabled:
        loss = loss + float(kl_coef) * trust_kl
    learner.actor_optimizer.zero_grad()
    if learner.bw_clean_grad_aggregation == "pcgrad":
        if fit_loss_l == "masked_kl":
            assert pred_score is not None
            sample_losses = learner._bw_clean_masked_kl_loss_per_sample(pred_score, target_actions, valid_masks_train)
        elif fit_loss_l == "score_reg_only":
            assert pred_score is not None
            sample_losses = _score_reg_loss_per_sample(
                pred_score=pred_score,
                target_action=target_actions,
                valid_mask=valid_masks_train,
            )
        else:
            sample_losses = learner._bw_clean_huber_loss_per_sample(pred_action, target_actions, valid_masks_train)
        if learner.bw_clean_trust_region_enabled:
            sample_losses = sample_losses + float(kl_coef) * learner._bw_clean_simplex_kl_per_sample(
                old_actions,
                pred_action,
                valid_masks_train,
            )
        if sample_order_np is not None and int(sample_losses.shape[0]) > 0:
            order_t = torch.as_tensor(
                np.asarray(sample_order_np, dtype=np.int64),
                dtype=torch.long,
                device=sample_losses.device,
            )
            sample_losses = sample_losses.index_select(0, order_t)
        learner._bw_clean_pcgrad_backward(
            sample_losses=sample_losses,
            params=[param for param in learner.actor.parameters() if param.requires_grad],
            task_group_size=int(learner.bw_clean_pcgrad_task_group_size),
        )
    else:
        loss.backward()
    torch.nn.utils.clip_grad_norm_(learner.actor.parameters(), learner.max_grad_norm)

    actor_state_before = None
    optimizer_state_before = None
    grad_snapshots: list[torch.Tensor | None] = []
    base_lrs: list[float] = []
    if learner.bw_clean_trust_region_enabled:
        actor_state_before = {name: tensor.detach().clone() for name, tensor in learner.actor.state_dict().items()}
        optimizer_state_before = copy.deepcopy(learner.actor_optimizer.state_dict())
        grad_snapshots = [
            None if param.grad is None else param.grad.detach().clone()
            for param in learner.actor.parameters()
        ]
        base_lrs = [float(group.get("lr", 0.0)) for group in learner.actor_optimizer.param_groups]

    learner.actor_optimizer.step()
    with torch.inference_mode():
        updated_out = learner.actor.act_bw(local_batch_train, deterministic=True)
    measured_kl = float(learner._bw_clean_simplex_kl(old_actions, updated_out.action, valid_masks_train).item())
    control_kl = measured_kl
    if learner.bw_clean_trust_region_enabled and measured_kl > float(learner.bw_clean_trust_region_target_kl):
        accepted = False
        for trial in range(1, int(learner.bw_clean_trust_region_max_backtracks) + 1):
            assert actor_state_before is not None
            assert optimizer_state_before is not None
            learner.actor.load_state_dict(actor_state_before)
            learner.actor_optimizer.load_state_dict(optimizer_state_before)
            learner.actor_optimizer.zero_grad(set_to_none=True)
            for param, grad in zip(learner.actor.parameters(), grad_snapshots):
                param.grad = None if grad is None else grad.detach().clone()
            lr_scale = float(learner.bw_clean_trust_region_backtrack_factor) ** int(trial)
            for group, base_lr in zip(learner.actor_optimizer.param_groups, base_lrs):
                group["lr"] = float(base_lr) * lr_scale
            learner.actor_optimizer.step()
            with torch.inference_mode():
                updated_out = learner.actor.act_bw(local_batch_train, deterministic=True)
            control_kl = float(learner._bw_clean_simplex_kl(old_actions, updated_out.action, valid_masks_train).item())
            if control_kl <= float(learner.bw_clean_trust_region_target_kl):
                measured_kl = control_kl
                accepted = True
                break
        for group, base_lr in zip(learner.actor_optimizer.param_groups, base_lrs):
            group["lr"] = float(base_lr)
        if not accepted:
            assert actor_state_before is not None
            assert optimizer_state_before is not None
            learner.actor.load_state_dict(actor_state_before)
            learner.actor_optimizer.load_state_dict(optimizer_state_before)
            learner.actor_optimizer.zero_grad(set_to_none=True)
            measured_kl = 0.0

    if learner.bw_clean_trust_region_enabled:
        if control_kl > 1.5 * float(learner.bw_clean_trust_region_target_kl):
            kl_coef = min(float(kl_coef) * 2.0, float(learner.bw_clean_trust_region_kl_coef_max))
        elif measured_kl < 0.5 * float(learner.bw_clean_trust_region_target_kl):
            kl_coef = max(float(kl_coef) * 0.5, float(learner.bw_clean_trust_region_kl_coef_min))
    learner._bw_clean_trust_region_kl_coef = float(kl_coef)

    with torch.inference_mode():
        final_out = learner.actor.act_bw(local_batch_train, deterministic=True)
    target_gap = _masked_l1(
        final_out.action.detach().cpu().numpy(),
        target_actions.detach().cpu().numpy(),
        valid_masks_train.detach().cpu().numpy(),
    )
    return {
        "policy_loss": float(base_loss.item()),
        "measured_kl": float(measured_kl),
        "kl_coef": float(kl_coef),
        "mean_target_gap": float(np.mean(target_gap, dtype=np.float64)) if target_gap.size > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--actor_checkpoint", required=True)
    parser.add_argument("--panel_states", type=int, default=32)
    parser.add_argument("--panel_seed", type=int, default=35000)
    parser.add_argument("--teacher_horizon", type=int, default=5)
    parser.add_argument("--offline_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, nargs="*", default=[0, 1, 2, 5, 10, 20])
    parser.add_argument("--actor_lr", type=float, default=1.0e-3)
    parser.add_argument(
        "--fresh_loc_head",
        action="store_true",
        help="Reinitialize the fit actor's bw_policy.loc_head after building the fixed teacher bank/targets.",
    )
    parser.add_argument(
        "--fresh_loc_head_seed",
        type=int,
        default=None,
        help="Optional seed used when reinitializing a fresh loc_head. Defaults to cfg.seed.",
    )
    parser.add_argument(
        "--train_scope",
        default="all",
        help="Which actor parameters to update during offline fit: all, loc_head, user_fusion, user_refine, user_encoder",
    )
    parser.add_argument(
        "--grad_aggregation",
        choices=["mean", "pcgrad"],
        default="mean",
        help="Clean BW gradient aggregation used during offline fixed-bank fit.",
    )
    parser.add_argument(
        "--pcgrad_group_mode",
        choices=["random", "target_stats"],
        default=None,
        help="Optional override for bw_clean_pcgrad_group_mode.",
    )
    parser.add_argument(
        "--pcgrad_task_group_size",
        type=int,
        default=None,
        help="Optional override for bw_clean_pcgrad_task_group_size.",
    )
    parser.add_argument(
        "--loc_readout",
        choices=["fused", "fused_moe2", "fused_ctx_dot", "pairwise_comparator", "user_only", "user0", "user0_residual_fused"],
        default=None,
        help="Optional override for structured_bw_loc_readout.",
    )
    parser.add_argument(
        "--clean_loss",
        choices=["huber", "masked_kl"],
        default=None,
        help="Optional override for bw_clean_per_user_loss.",
    )
    parser.add_argument(
        "--fit_loss",
        choices=["huber", "masked_kl", "score_reg_only"],
        default=None,
        help="Offline fit loss used by this diagnostic script. Defaults to cfg clean loss when omitted.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json_out", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setattr(cfg, "bw_clean_grad_aggregation", str(args.grad_aggregation))
    if args.pcgrad_group_mode is not None:
        setattr(cfg, "bw_clean_pcgrad_group_mode", str(args.pcgrad_group_mode))
    if args.pcgrad_task_group_size is not None:
        setattr(cfg, "bw_clean_pcgrad_task_group_size", max(int(args.pcgrad_task_group_size), 1))
    if args.loc_readout is not None:
        setattr(cfg, "structured_bw_loc_readout", str(args.loc_readout))
    if args.clean_loss is not None:
        setattr(cfg, "bw_clean_per_user_loss", str(args.clean_loss))
    device = torch.device(args.device)
    _set_all_seeds(int(getattr(cfg, "seed", 0) or 0))
    fit_loss = (
        str(args.fit_loss).strip().lower()
        if args.fit_loss is not None
        else str(getattr(cfg, "bw_clean_per_user_loss", "huber") or "huber").strip().lower()
    )

    teacher_actor = _build_actor_from_checkpoint(cfg, str(args.actor_checkpoint), device)

    panel_rows, episode_count = _collect_panel_bank(
        cfg=cfg,
        actor=teacher_actor,
        device=device,
        panel_states=int(args.panel_states),
        seed_base=int(args.panel_seed),
    )
    snapshot_states = [dict(row["snapshot_state"] or {}) for row in panel_rows]
    fixed_ref_actions = np.stack([np.asarray(row["ref_action"], dtype=np.float32) for row in panel_rows], axis=0)
    valid_masks = np.stack([np.asarray(row["valid_mask"], dtype=bool) for row in panel_rows], axis=0)
    local_states = [row["local_state"] for row in panel_rows]

    teacher_learner = _make_learner(cfg, teacher_actor, device, float(args.actor_lr))
    teacher_learner.bw_clean_per_user_horizon = int(args.teacher_horizon)

    target_actions_np, rho_np, utility_np = teacher_learner._bw_clean_target_actions_parallel(
        snapshot_states=[dict(state or {}) for state in snapshot_states],
        ref_actions=fixed_ref_actions,
        valid_masks=valid_masks,
    )
    ref_returns_np = np.asarray(
        getattr(teacher_learner, "_bw_clean_last_ref_returns", np.zeros((len(snapshot_states),), dtype=np.float32)),
        dtype=np.float32,
    )
    target_returns_np = np.asarray(
        teacher_learner._bw_clean_rollout_returns_parallel(
            snapshot_states=[dict(state or {}) for state in snapshot_states],
            first_actions=[
                np.asarray(target_actions_np[idx], dtype=np.float32).reshape(int(cfg.num_uav), int(cfg.users_obs_max))
                for idx in range(target_actions_np.shape[0])
            ],
        ),
        dtype=np.float32,
    )
    improving_mask = target_returns_np > (ref_returns_np + 1.0e-6)
    train_idx_np = np.flatnonzero(improving_mask.astype(np.bool_))

    fit_actor = _build_actor_from_checkpoint(cfg, str(args.actor_checkpoint), device)
    fresh_loc_head_stats: dict[str, float] | None = None
    if bool(args.fresh_loc_head):
        fresh_loc_head_seed = (
            int(args.fresh_loc_head_seed)
            if args.fresh_loc_head_seed is not None
            else int(getattr(cfg, "seed", 0) or 0)
        )
        fresh_loc_head_stats = _refresh_bw_loc_head(
            actor=fit_actor,
            cfg=cfg,
            device=device,
            seed=int(fresh_loc_head_seed),
        )
    train_scope_used = _configure_train_scope(fit_actor, str(args.train_scope))
    learner = _make_learner(cfg, fit_actor, device, float(args.actor_lr))
    learner.bw_clean_per_user_horizon = int(args.teacher_horizon)

    local_batch = _collate_dataclass(local_states, device)
    train_idx_t = torch.as_tensor(train_idx_np, dtype=torch.long, device=device)
    local_batch_train = _index_dataclass(local_batch, train_idx_t)
    target_actions_t = torch.as_tensor(np.asarray(target_actions_np[train_idx_np], dtype=np.float32), dtype=torch.float32, device=device)
    valid_masks_train = torch.as_tensor(np.asarray(valid_masks[train_idx_np], dtype=np.float32), dtype=torch.float32, device=device) > 0.5
    target_stats_train_np = learner._bw_clean_target_stats_features(
        target_actions_np=np.asarray(target_actions_np[train_idx_np], dtype=np.float32),
        ref_actions_np=np.asarray(fixed_ref_actions[train_idx_np], dtype=np.float32),
        valid_masks_np=np.asarray(valid_masks[train_idx_np], dtype=bool),
    )
    target_stats_order_np: np.ndarray | None = None
    if (
        train_idx_np.size > 0
        and learner.bw_clean_grad_aggregation == "pcgrad"
        and learner.bw_clean_pcgrad_group_mode == "target_stats"
        and int(learner.bw_clean_pcgrad_task_group_size) > 1
    ):
        target_stats_order_np = learner._bw_clean_target_stats_order(
            target_actions_np=np.asarray(target_actions_np[train_idx_np], dtype=np.float32),
            ref_actions_np=np.asarray(fixed_ref_actions[train_idx_np], dtype=np.float32),
            valid_masks_np=np.asarray(valid_masks[train_idx_np], dtype=bool),
        )

    eval_schedule = sorted({step for step in list(args.eval_steps) if 0 <= int(step) <= int(args.offline_steps)})
    history: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []

    try:
        for step in range(0, int(args.offline_steps) + 1):
            if step in eval_schedule:
                panel_eval = _evaluate_panel(
                    cfg=cfg,
                    actor=learner.actor,
                    device=device,
                    snapshot_states=snapshot_states,
                    fixed_target_actions=target_actions_np,
                )
                with torch.inference_mode():
                    current_out = learner.actor.act_bw(local_batch_train, deterministic=True)
                current_gap = _masked_l1(
                    current_out.action.detach().cpu().numpy(),
                    target_actions_t.detach().cpu().numpy(),
                    valid_masks_train.detach().cpu().numpy(),
                )
                history.append(
                    {
                        "offline_step": int(step),
                        "panel_eval": panel_eval,
                        "current_target_gap": _safe_summary(np.asarray(current_gap, dtype=np.float64)),
                        "trust_region_kl_coef": float(getattr(learner, "_bw_clean_trust_region_kl_coef", learner.bw_clean_trust_region_kl_coef_init)),
                    }
                )
            if step == int(args.offline_steps):
                break
            if train_idx_np.size <= 0:
                continue
            learner._bw_clean_last_group_debug = None
            sample_order_np: np.ndarray | None = None
            if learner.bw_clean_grad_aggregation == "pcgrad":
                if target_stats_order_np is None:
                    sample_order_np = np.arange(int(train_idx_np.size), dtype=np.int64)
                    np.random.shuffle(sample_order_np)
                else:
                    sample_order_np = np.asarray(target_stats_order_np, dtype=np.int64).copy()
                    group_size = max(int(learner.bw_clean_pcgrad_task_group_size), 1)
                    if group_size > 1 and int(sample_order_np.size) > group_size:
                        blocks = [
                            sample_order_np[start : start + group_size]
                            for start in range(0, int(sample_order_np.size), group_size)
                        ]
                        np.random.shuffle(blocks)
                        sample_order_np = np.concatenate(blocks, axis=0)
                learner._bw_clean_last_group_debug = {
                    "grad_aggregation": str(learner.bw_clean_grad_aggregation),
                    "group_mode": str(learner.bw_clean_pcgrad_group_mode),
                    "task_group_size": int(learner.bw_clean_pcgrad_task_group_size),
                    "num_train_samples": int(train_idx_np.size),
                    "minibatch_size": int(train_idx_np.size),
                    "epochs": [
                        learner._bw_clean_group_debug_payload(
                            epoch_index=0,
                            order_np=np.asarray(sample_order_np, dtype=np.int64),
                            train_idx_np=np.asarray(train_idx_np, dtype=np.int64),
                            target_stats_np=target_stats_train_np,
                            minibatch_size=int(train_idx_np.size),
                            task_group_size=int(learner.bw_clean_pcgrad_task_group_size),
                        )
                    ],
                    "target_stats_overall": {
                        "valid_count_mean": float(np.mean(target_stats_train_np["valid_count"], dtype=np.float64))
                        if train_idx_np.size > 0
                        else 0.0,
                        "target_entropy_mean": float(np.mean(target_stats_train_np["target_entropy"], dtype=np.float64))
                        if train_idx_np.size > 0
                        else 0.0,
                        "target_top1_mass_mean": float(np.mean(target_stats_train_np["target_top1_mass"], dtype=np.float64))
                        if train_idx_np.size > 0
                        else 0.0,
                        "target_gap_l1_mean": float(np.mean(target_stats_train_np["target_gap_l1"], dtype=np.float64))
                        if train_idx_np.size > 0
                        else 0.0,
                    },
                }
            update_stats = _offline_fit_one_step(
                learner=learner,
                local_batch_train=local_batch_train,
                target_actions=target_actions_t,
                valid_masks_train=valid_masks_train,
                fit_loss=fit_loss,
                sample_order_np=sample_order_np,
            )
            update_payload = {"after_offline_step": int(step + 1), **update_stats}
            if getattr(learner, "_bw_clean_last_group_debug", None) is not None:
                update_payload["clean_group_debug"] = learner._bw_clean_last_group_debug
            updates.append(update_payload)
    finally:
        learner._close_bw_clean_probe_group()
        teacher_learner._close_bw_clean_probe_group()

    payload = {
        "config": os.path.abspath(args.config),
        "actor_checkpoint": os.path.abspath(args.actor_checkpoint),
        "fresh_loc_head": bool(args.fresh_loc_head),
        "fresh_loc_head_seed": (
            int(args.fresh_loc_head_seed)
            if args.fresh_loc_head_seed is not None
            else int(getattr(cfg, "seed", 0) or 0)
        ),
        "fresh_loc_head_stats": fresh_loc_head_stats,
        "fit_loss": str(fit_loss),
        "train_scope_requested": str(args.train_scope),
        "train_scope_used": str(train_scope_used),
        "grad_aggregation": str(args.grad_aggregation),
        "pcgrad_group_mode": str(getattr(cfg, "bw_clean_pcgrad_group_mode", "random")),
        "pcgrad_task_group_size": int(getattr(cfg, "bw_clean_pcgrad_task_group_size", 32) or 1),
        "panel": {
            "states": int(len(panel_rows)),
            "episodes": int(episode_count),
            "seed_base": int(args.panel_seed),
            "teacher_horizon": int(args.teacher_horizon),
        },
        "fixed_teacher_bank": {
            "improving_frac": float(np.mean(improving_mask.astype(np.float32))) if improving_mask.size > 0 else 0.0,
            "improving_count": int(train_idx_np.size),
            "target_gap": _safe_summary(_masked_l1(target_actions_np, fixed_ref_actions, valid_masks)),
            "local_gain": _safe_summary(np.asarray(target_returns_np - ref_returns_np, dtype=np.float32)),
            "rho": _safe_summary(np.asarray(rho_np, dtype=np.float64)),
            "utility_l1": _safe_summary(np.asarray(utility_np, dtype=np.float64)),
        },
        "history": history,
        "updates": updates,
    }

    if args.json_out:
        out_path = Path(args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
