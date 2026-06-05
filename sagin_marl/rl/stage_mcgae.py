from __future__ import annotations

import random
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
import torch

from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO, _index_dataclass


STAGE_ID = {"accel": 0, "sat": 1, "bw": 2}


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def stage_optimizer_params(actor: torch.nn.Module, stage_id: int) -> list[torch.nn.Parameter]:
    if int(stage_id) == 0:
        module = getattr(actor, "accel_policy", None)
    elif int(stage_id) == 1:
        module = getattr(actor, "sat_subset_policy", None)
    elif int(stage_id) == 2:
        module = getattr(actor, "bw_policy", None)
    else:
        module = None
    if module is None:
        return []
    return [param for param in module.parameters() if param.requires_grad]


def make_stage_optimizers(actor: torch.nn.Module, actor_lr: float) -> dict[int, torch.optim.Optimizer]:
    out: dict[int, torch.optim.Optimizer] = {}
    for stage_id in (0, 1, 2):
        params = stage_optimizer_params(actor, stage_id)
        if params:
            out[stage_id] = torch.optim.Adam(params, lr=float(actor_lr))
    return out


def force_single_stage_config(cfg: Any, *, stage_id: int, reward_mode: str | None) -> None:
    if reward_mode:
        cfg.reward_mode = str(reward_mode)
    cfg.train_accel = bool(stage_id == 0)
    cfg.train_sat = bool(stage_id == 1)
    cfg.train_bw = bool(stage_id == 2)
    cfg.exec_accel_source = "policy" if stage_id == 0 else "cluster_center_queue_aware"
    cfg.exec_sat_source = "policy" if stage_id == 1 else "queue_aware"
    cfg.exec_bw_source = "policy" if stage_id == 2 else "queue_aware"
    cfg.structured_actor_update_mode = "ppo"
    cfg.accel_update_mode = "ppo"
    cfg.sat_update_mode = "ppo"
    cfg.bw_update_mode = "ppo"
    cfg.checkpoint_eval_enabled = False
    cfg.train_trace_enabled = False
    cfg.structured_env_backend = "native"
    cfg.structured_env_tensor_backend = "cuda"


def make_learner(cfg: Any, *, device: torch.device, stage_id: int) -> tuple[StructuredMAPPO, Any, Any]:
    bundle = build_structured_modules_from_config(cfg)
    actor = bundle.actor.to(device)
    critic = bundle.critic.to(device) if bundle.critic is not None else None
    if critic is None:
        raise RuntimeError("stage MC-GAE requires a critic.")
    learner = StructuredMAPPO(
        actor=actor,
        critic=critic,
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_ratio=float(cfg.clip_ratio),
        value_coef=float(cfg.value_coef),
        entropy_coef=float(cfg.entropy_coef),
        max_grad_norm=float(cfg.max_grad_norm),
        ppo_epochs=int(cfg.ppo_epochs),
        num_mini_batch=int(cfg.num_mini_batch),
        actor_optimizer=torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=float(cfg.actor_lr)),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=float(cfg.critic_lr)),
        device=device,
        cfg=cfg,
        train_accel=bool(stage_id == 0),
        train_sat=bool(stage_id == 1),
        train_bw=bool(stage_id == 2),
        exec_accel_source=str(cfg.exec_accel_source),
        exec_sat_source=str(cfg.exec_sat_source),
        exec_bw_source=str(cfg.exec_bw_source),
    )
    return learner, actor, critic


def collect_one_rollout(
    learner: StructuredMAPPO,
    group: Any,
    *,
    rollout_env_steps: int,
    device: torch.device,
    target: str,
) -> tuple[StructuredRolloutBuffer, Any, torch.Tensor]:
    begin_native_rollout = getattr(learner, "begin_native_rollout", None)
    if callable(begin_native_rollout):
        begin_native_rollout(
            group,
            rollout_env_steps=int(rollout_env_steps),
            num_envs=int(len(group)),
        )
    buffer = StructuredRolloutBuffer()
    learner.collect_env_horizon_native_tensor_policy(
        group,
        buffer=buffer,
        horizon=int(rollout_env_steps),
        deterministic=False,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    rollout_views = buffer.build_rollout_views(device)
    returns, _advantages, _value_override = compute_returns_for_views(
        learner,
        buffer,
        rollout_views,
        target=target,
        device=device,
    )
    return buffer, rollout_views, returns


def _mc_returns_for_views(
    learner: StructuredMAPPO,
    buffer: StructuredRolloutBuffer,
    *,
    batch_view: Any,
    return_view: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value_override = torch.zeros((int(batch_view.transition_count),), dtype=torch.float32, device=device)
    old_lam = float(learner.gae_lambda)
    learner.gae_lambda = 1.0
    try:
        gae = learner.compute_returns_and_advantages(
            buffer,
            None,
            return_view=return_view,
            value_override=value_override,
        )
    finally:
        learner.gae_lambda = old_lam
    returns = torch.as_tensor(gae["returns"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(gae["advantages"], dtype=torch.float32, device=device)
    return returns, advantages, value_override.to(device=device, dtype=torch.float32)


def _train_gae_returns_for_views(
    learner: StructuredMAPPO,
    buffer: StructuredRolloutBuffer,
    views: Any,
    *,
    batch_view: Any,
    return_view: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value_override = learner._rollout_value_override_from_training_view(batch_view)
    learner._apply_rollout_value_override_to_views(
        batch_view=batch_view,
        return_view=return_view,
        value_override=value_override,
    )
    gae = learner.compute_returns_and_advantages(
        buffer,
        views.bootstrap_view,
        return_view=return_view,
        value_override=value_override,
    )
    returns = torch.as_tensor(gae["returns"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(gae["advantages"], dtype=torch.float32, device=device)
    return returns, advantages, value_override.to(device=device, dtype=torch.float32)


def _nstep_returns_from_view(
    learner: StructuredMAPPO,
    views: Any,
    *,
    value_override: torch.Tensor,
    horizon: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return_view = views.return_view
    num_steps = int(return_view.transition_count)
    values_np = value_override.detach().cpu().numpy().astype(np.float32, copy=False).reshape(num_steps)
    rewards_np = np.asarray(return_view.rewards, dtype=np.float32).reshape(num_steps)
    stage_ids_np = np.asarray(return_view.stage_ids, dtype=np.int64).reshape(num_steps)
    env_indices_np = np.asarray(return_view.env_indices, dtype=np.int64).reshape(num_steps)
    terminated_np = np.asarray(return_view.terminated, dtype=bool).reshape(num_steps)
    truncated_np = np.asarray(return_view.truncated, dtype=bool).reshape(num_steps)

    gamma_env = float(learner.gamma)
    horizon_i = max(int(horizon), 1)
    returns_np = np.zeros((num_steps,), dtype=np.float32)
    advantages_np = np.zeros((num_steps,), dtype=np.float32)
    by_env: dict[int, list[int]] = {}
    for transition_idx, env_idx in enumerate(env_indices_np.tolist()):
        by_env.setdefault(int(env_idx), []).append(int(transition_idx))

    for indices in by_env.values():
        ordered = sorted(indices)
        local_pos = {int(idx): pos for pos, idx in enumerate(ordered)}
        for start_pos, start_idx in enumerate(ordered):
            acc = 0.0
            discount = 1.0
            env_boundaries = 0
            bootstrap_idx: int | None = None
            for pos in range(start_pos, len(ordered)):
                idx = int(ordered[pos])
                stage_id = int(stage_ids_np[idx])
                acc += float(discount) * float(rewards_np[idx])
                gamma_step = gamma_env if stage_id == 2 else 1.0
                ended = bool(stage_id == 2 and (terminated_np[idx] or truncated_np[idx]))
                if ended:
                    bootstrap_idx = None
                    break
                if stage_id == 2:
                    env_boundaries += 1
                    if env_boundaries >= horizon_i:
                        next_pos = pos + 1
                        bootstrap_idx = int(ordered[next_pos]) if next_pos < len(ordered) else None
                        discount *= float(gamma_step)
                        break
                discount *= float(gamma_step)
            if bootstrap_idx is not None and int(bootstrap_idx) in local_pos:
                acc += float(discount) * float(values_np[int(bootstrap_idx)])
            returns_np[int(start_idx)] = float(acc)
            advantages_np[int(start_idx)] = float(acc) - float(values_np[int(start_idx)])

    returns = torch.as_tensor(returns_np, dtype=torch.float32, device=device)
    advantages = torch.as_tensor(advantages_np, dtype=torch.float32, device=device)
    return returns, advantages


def compute_returns_for_views(
    learner: StructuredMAPPO,
    buffer: StructuredRolloutBuffer,
    views: Any,
    *,
    target: str,
    device: torch.device,
    target_mix_alpha: float | None = None,
    return_nstep_horizon: int | None = None,
    return_info: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, Any],
]:
    batch_view = views.training_view
    return_view = views.return_view
    target_l = str(target).strip().lower()
    info: dict[str, Any] = {"target": target_l}
    if target_l == "mc":
        returns, advantages, value_override = _mc_returns_for_views(
            learner,
            buffer,
            batch_view=batch_view,
            return_view=return_view,
            device=device,
        )
    elif target_l == "train_gae":
        returns, advantages, value_override = _train_gae_returns_for_views(
            learner,
            buffer,
            views,
            batch_view=batch_view,
            return_view=return_view,
            device=device,
        )
    elif target_l == "mixed":
        alpha = 0.5 if target_mix_alpha is None else float(target_mix_alpha)
        alpha = min(max(alpha, 0.0), 1.0)
        mc_returns, _mc_advantages, _mc_value_override = _mc_returns_for_views(
            learner,
            buffer,
            batch_view=batch_view,
            return_view=return_view,
            device=device,
        )
        gae_returns, gae_advantages, value_override = _train_gae_returns_for_views(
            learner,
            buffer,
            views,
            batch_view=batch_view,
            return_view=return_view,
            device=device,
        )
        returns = alpha * mc_returns + (1.0 - alpha) * gae_returns
        advantages = returns - value_override.to(device=device, dtype=torch.float32).reshape(-1)
        info.update(
            {
                "target_mix_alpha": float(alpha),
                "mc_returns": mc_returns.detach(),
                "bootstrap_gae_returns": gae_returns.detach(),
                "bootstrap_gae_advantages": gae_advantages.detach(),
            }
        )
    elif target_l == "bootstrap_mc_aux":
        mc_returns, _mc_advantages, _mc_value_override = _mc_returns_for_views(
            learner,
            buffer,
            batch_view=batch_view,
            return_view=return_view,
            device=device,
        )
        returns, advantages, value_override = _train_gae_returns_for_views(
            learner,
            buffer,
            views,
            batch_view=batch_view,
            return_view=return_view,
            device=device,
        )
        info.update(
            {
                "mc_aux_returns": mc_returns.detach(),
                "bootstrap_gae_returns": returns.detach(),
                "bootstrap_gae_advantages": advantages.detach(),
            }
        )
    elif target_l == "nstep":
        horizon = int(1 if return_nstep_horizon is None else return_nstep_horizon)
        value_override = learner._rollout_value_override_from_training_view(batch_view)
        learner._apply_rollout_value_override_to_views(
            batch_view=batch_view,
            return_view=return_view,
            value_override=value_override,
        )
        returns, advantages = _nstep_returns_from_view(
            learner,
            views,
            value_override=value_override,
            horizon=int(horizon),
            device=device,
        )
        info.update({"return_nstep_horizon": float(max(int(horizon), 1))})
    else:
        raise ValueError("target must be one of {'mc', 'train_gae', 'mixed', 'bootstrap_mc_aux', 'nstep'}.")
    if return_info:
        return returns, advantages, value_override.to(device=device, dtype=torch.float32), info
    return returns, advantages, value_override.to(device=device, dtype=torch.float32)


def clone_dataclass_tensors(batch: Any, *, device: torch.device | None = None) -> Any:
    field_names = getattr(batch, "_tensor_fields", None)
    if field_names is None:
        if not is_dataclass(batch):
            raise TypeError("clone_dataclass_tensors expects a tensor-field dataclass")
        field_names = tuple(field.name for field in fields(batch))
    kwargs = {}
    for field_name in field_names:
        value = getattr(batch, field_name)
        if not torch.is_tensor(value):
            raise TypeError(f"Unsupported field type for {field_name}: {type(value)!r}")
        tensor = value.detach().clone()
        if device is not None and tensor.device != device:
            tensor = tensor.to(device)
        kwargs[str(field_name)] = tensor
    return type(batch)(**kwargs)


def explained_variance_np(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if target.size <= 1:
        return 0.0
    var = float(np.var(target))
    if var <= 1.0e-12:
        return 0.0
    return 1.0 - float(np.var(target - pred)) / var


def corr_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        return 0.0
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) <= 2:
        return 0.0
    a = a[mask]
    b = b[mask]
    if float(np.std(a)) <= 1.0e-12 or float(np.std(b)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def summarize_np(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return {"mean": 0.0, "std": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10.0)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def eval_critic(
    learner: StructuredMAPPO,
    *,
    stage_id: int,
    world_bank: Any,
    target: torch.Tensor,
    batch_size: int,
    pad_to_batch_size: bool = False,
) -> tuple[np.ndarray, dict[str, float]]:
    preds: list[torch.Tensor] = []
    n = int(target.numel())
    with torch.no_grad():
        for start in range(0, n, max(int(batch_size), 1)):
            stop = min(start + int(batch_size), n)
            idx = torch.arange(start, stop, device=target.device, dtype=torch.long)
            real_count = int(idx.numel())
            if bool(pad_to_batch_size) and real_count > 0 and real_count < int(batch_size):
                pad = idx.new_full((int(batch_size) - real_count,), int(idx[-1].item()))
                idx = torch.cat([idx, pad], dim=0)
            pred = learner._stage_value_eval_from_batch(int(stage_id), _index_dataclass(world_bank, idx))
            # torch.compile may run this value path under CUDA graphs; clone so
            # the next compiled invocation cannot overwrite the stored output.
            preds.append(pred[:real_count].detach().clone())
    pred_t = torch.cat(preds, dim=0).to(device=target.device, dtype=torch.float32)
    pred_np = pred_t.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    stats = {
        "mse": float(np.mean((pred_np - target_np) ** 2)),
        "mae": float(np.mean(np.abs(pred_np - target_np))),
        "ev": explained_variance_np(pred_np, target_np),
        "corr": corr_np(pred_np, target_np),
        "pred": summarize_np(pred_np),
        "target": summarize_np(target_np),
        "residual": summarize_np(target_np - pred_np),
    }
    return pred_np, stats


# Backward-compatible names for scripts that still import underscored helpers.
_clone_dataclass_tensors = clone_dataclass_tensors
_collect_one_rollout = collect_one_rollout
_compute_returns_for_views = compute_returns_for_views
_corr = corr_np
_eval_critic = eval_critic
_explained_variance_np = explained_variance_np
_force_single_stage_config = force_single_stage_config
_make_learner = make_learner
_make_stage_optimizers = make_stage_optimizers
_set_seed = set_seed
_stage_optimizer_params = stage_optimizer_params
_summ = summarize_np
