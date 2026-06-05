from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO
from sagin_marl.rl.stage_mcgae import (
    compute_returns_for_views,
    stage_optimizer_params as _stage_optimizer_params,
)
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver_group
from sagin_marl.utils.torch_compile_cache import report_torch_compile_cache

from scripts.train_stage_mcgae import (
    STAGE_NAME,
    _configure_accel_safety_for_training,
    _cuda_mem,
    _enable_strict_compile_global,
    _normalize_stage_advantage,
    _stage_actor_update_full_stage,
    _stage_cfg_value,
    _stage_gae_from_mc_targets,
    _stage_indices,
    _summ_tensor,
    _sync_danger_imitation_to_learner,
    _train_stage_critic_on_stage,
    _write_metrics,
)


STAGES = (0, 1, 2)
STAGE_ACTOR_PREFIX = {
    0: "accel_policy.",
    1: "sat_subset_policy.",
    2: "bw_policy.",
}


def _append_phase_trace(run_dir: Path, *, update: int, phase: str, event: str, **payload: Any) -> None:
    """Write a low-overhead heartbeat so interrupted runs reveal the hot phase."""

    row: dict[str, Any] = {
        "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "perf_counter": time.perf_counter(),
        "update": int(update),
        "phase": str(phase),
        "event": str(event),
    }
    row.update(payload)
    trace_path = run_dir / "phase_trace.jsonl"
    with trace_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _stage_metric_suffix(stage_id: int) -> str:
    return STAGE_NAME[int(stage_id)]


def _stage_hparam_key(prefix: str, stage_id: int) -> str:
    return f"{prefix}_{_stage_metric_suffix(stage_id)}"


class _ActorOnlyStageBatch:
    def __init__(
        self,
        *,
        local_batch: Any,
        actions: torch.Tensor,
        old_logprobs: torch.Tensor,
        latent_actions: torch.Tensor | None = None,
        danger_imitation_targets: torch.Tensor | None = None,
        danger_imitation_masks: torch.Tensor | None = None,
        transition_indices: np.ndarray | None = None,
        env_indices: np.ndarray | None = None,
        bw_ref_actions: torch.Tensor | None = None,
        old_logprobs_per_agent: torch.Tensor | None = None,
        bw_tau: torch.Tensor | None = None,
        bw_kappa: torch.Tensor | None = None,
        bw_valid_count: torch.Tensor | None = None,
        bw_latent_count: torch.Tensor | None = None,
        bw_logprob_raw_per_agent: torch.Tensor | None = None,
    ) -> None:
        self.local_batch = local_batch
        self.actions = actions
        self.old_logprobs = old_logprobs
        self.latent_actions = latent_actions
        self.danger_imitation_targets = danger_imitation_targets
        self.danger_imitation_masks = danger_imitation_masks
        self.transition_indices = (
            np.asarray(transition_indices, dtype=np.int64).copy()
            if transition_indices is not None
            else np.arange(int(actions.shape[0]), dtype=np.int64)
        )
        self.env_indices = (
            np.asarray(env_indices, dtype=np.int64).copy()
            if env_indices is not None
            else np.full((int(actions.shape[0]),), -1, dtype=np.int64)
        )
        self.bw_ref_actions = bw_ref_actions
        self.old_logprobs_per_agent = old_logprobs_per_agent
        self.bw_tau = bw_tau
        self.bw_kappa = bw_kappa
        self.bw_valid_count = bw_valid_count
        self.bw_latent_count = bw_latent_count
        self.bw_logprob_raw_per_agent = bw_logprob_raw_per_agent

    @property
    def num_samples(self) -> int:
        return int(self.actions.shape[0])

    @property
    def num_agents(self) -> int:
        return int(self.actions.shape[1]) if self.actions.ndim >= 2 else 1


def _actor_only_stage_batch(stage_batch: Any) -> _ActorOnlyStageBatch:
    def _clone_tensor_dataclass(batch: Any) -> Any:
        field_names = getattr(batch, "_tensor_fields", None)
        if field_names is None:
            from dataclasses import fields, is_dataclass

            if not is_dataclass(batch):
                raise TypeError("_clone_tensor_dataclass expects a tensor-field dataclass")
            field_names = tuple(field.name for field in fields(batch))
        kwargs = {}
        for field_name in field_names:
            value = getattr(batch, str(field_name))
            if not torch.is_tensor(value):
                raise TypeError(f"Unsupported field type for {field_name}: {type(value)!r}")
            kwargs[str(field_name)] = value.detach().clone()
        return type(batch)(**kwargs)

    def _clone_optional(value: torch.Tensor | None) -> torch.Tensor | None:
        return None if value is None else value.detach().clone()

    return _ActorOnlyStageBatch(
        local_batch=_clone_tensor_dataclass(stage_batch.local_batch),
        actions=stage_batch.actions.detach().clone(),
        old_logprobs=stage_batch.old_logprobs.detach().clone(),
        latent_actions=_clone_optional(getattr(stage_batch, "latent_actions", None)),
        danger_imitation_targets=_clone_optional(getattr(stage_batch, "danger_imitation_targets", None)),
        danger_imitation_masks=_clone_optional(getattr(stage_batch, "danger_imitation_masks", None)),
        transition_indices=getattr(stage_batch, "transition_indices", None),
        env_indices=getattr(stage_batch, "env_indices", None),
        bw_ref_actions=_clone_optional(getattr(stage_batch, "bw_ref_actions", None)),
        old_logprobs_per_agent=_clone_optional(getattr(stage_batch, "old_logprobs_per_agent", None)),
        bw_tau=_clone_optional(getattr(stage_batch, "bw_tau", None)),
        bw_kappa=_clone_optional(getattr(stage_batch, "bw_kappa", None)),
        bw_valid_count=_clone_optional(getattr(stage_batch, "bw_valid_count", None)),
        bw_latent_count=_clone_optional(getattr(stage_batch, "bw_latent_count", None)),
        bw_logprob_raw_per_agent=_clone_optional(getattr(stage_batch, "bw_logprob_raw_per_agent", None)),
    )


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _tensor_ev(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = pred.detach().to(dtype=torch.float32).reshape(-1)
    t = target.detach().to(device=p.device, dtype=torch.float32).reshape(-1)
    if int(t.numel()) <= 1:
        return 0.0
    var = torch.var(t, unbiased=False)
    if float(var.detach().cpu().item()) <= 1.0e-12:
        return 0.0
    ev = 1.0 - torch.var(t - p, unbiased=False) / var
    return float(ev.detach().cpu().item())


def _prefix_keys(values: dict[str, float], prefix: str, *, keep_prefixed_stage_keys: bool = False) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in values.items():
        if keep_prefixed_stage_keys and any(key.endswith(f"_{name}") for name in STAGE_NAME.values()):
            out[key] = float(value)
        else:
            out[f"{prefix}_{key}"] = float(value)
    return out


def _force_joint_config(cfg: Any, *, reward_mode: str) -> None:
    cfg.reward_mode = str(reward_mode)
    cfg.train_accel = True
    cfg.train_sat = True
    cfg.train_bw = True
    cfg.exec_accel_source = "policy"
    cfg.exec_sat_source = "policy"
    cfg.exec_bw_source = "policy"
    cfg.structured_actor_update_mode = "ppo"
    cfg.accel_update_mode = "ppo"
    cfg.sat_update_mode = "ppo"
    cfg.bw_update_mode = "ppo"
    cfg.actor_advantage_normalize_enabled = True
    cfg.stagewise_advantage_norm_enabled = True
    cfg.checkpoint_eval_enabled = False
    cfg.train_trace_enabled = False
    _configure_accel_safety_for_training(cfg, stage_id=0)


def _make_joint_learner(cfg: Any, *, device: torch.device) -> StructuredMAPPO:
    bundle = build_structured_modules_from_config(cfg)
    actor = bundle.actor.to(device)
    critic = bundle.critic.to(device) if bundle.critic is not None else None
    if critic is None:
        raise RuntimeError("joint MC-GAE training requires a critic.")
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
        train_accel=True,
        train_sat=True,
        train_bw=True,
        exec_accel_source=str(cfg.exec_accel_source),
        exec_sat_source=str(cfg.exec_sat_source),
        exec_bw_source=str(cfg.exec_bw_source),
    )
    _sync_danger_imitation_to_learner(learner, cfg)
    return learner


def _optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    lr_f = float(lr)
    for group in optimizer.param_groups:
        group["lr"] = lr_f


def _make_stage_actor_optimizer(
    params: list[torch.nn.Parameter],
    *,
    lr: float,
    optimizer_name: str,
) -> torch.optim.Optimizer:
    name = str(optimizer_name or "adam").strip().lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=float(lr))
    if name == "sgd":
        return torch.optim.SGD(params, lr=float(lr))
    raise ValueError(f"Unsupported actor optimizer {optimizer_name!r}; expected 'adam' or 'sgd'.")


def _stage_actor_optimizer_name(args: argparse.Namespace, stage_id: int) -> str:
    stage_specific = {
        0: getattr(args, "accel_actor_optimizer", None),
        1: getattr(args, "sat_actor_optimizer", None),
        2: getattr(args, "bw_actor_optimizer", None),
    }.get(int(stage_id))
    if stage_specific is not None and str(stage_specific).strip():
        return str(stage_specific).strip().lower()
    return str(getattr(args, "actor_optimizer", "adam") or "adam").strip().lower()


def _maybe_adjust_stage_actor_lr(
    optimizer: torch.optim.Optimizer,
    *,
    stage_id: int,
    stage_metrics: dict[str, float],
    hparams: dict[str, float | int],
) -> dict[str, float]:
    stage_name = _stage_metric_suffix(stage_id)
    current_lr = _optimizer_lr(optimizer)
    out = {
        f"actor_lr_used_{stage_name}": float(current_lr),
        f"actor_lr_next_{stage_name}": float(current_lr),
        f"actor_lr_decayed_{stage_name}": 0.0,
        f"actor_lr_grew_{stage_name}": 0.0,
        f"actor_lr_high_count_{stage_name}": float(hparams.get(f"actor_lr_high_count_{stage_name}", 0)),
        f"actor_lr_low_count_{stage_name}": float(hparams.get(f"actor_lr_low_count_{stage_name}", 0)),
    }
    if not bool(int(hparams.get("stage_actor_dynamic_lr_enabled", 0))):
        return out
    kl = float(stage_metrics.get(f"approx_kl_{stage_name}", 0.0))
    clip_frac = float(stage_metrics.get(f"clip_frac_{stage_name}", 0.0))
    target_kl = float(hparams.get(f"actor_target_kl_{stage_name}", 0.0))
    high_kl_threshold = target_kl * float(hparams.get("stage_actor_lr_high_kl_frac", 1.5))
    low_kl_threshold = target_kl * float(hparams.get("stage_actor_lr_low_kl_frac", 0.3))
    hard_kl_threshold = target_kl * float(hparams.get("stage_actor_lr_hard_kl_frac", 3.0))
    high_clip_threshold = float(hparams.get("stage_actor_lr_high_clip", 0.3))
    low_clip_threshold = float(hparams.get("stage_actor_lr_low_clip", 0.05))
    hard_clip_threshold = float(hparams.get("stage_actor_lr_hard_clip", 0.6))
    high = (high_kl_threshold > 0.0 and kl > high_kl_threshold) or (
        high_clip_threshold > 0.0 and clip_frac > high_clip_threshold
    )
    hard_high = (hard_kl_threshold > 0.0 and kl > hard_kl_threshold) or (
        hard_clip_threshold > 0.0 and clip_frac > hard_clip_threshold
    )
    low = (low_kl_threshold > 0.0 and kl < low_kl_threshold) and (
        low_clip_threshold > 0.0 and clip_frac < low_clip_threshold
    )
    high_count_key = f"actor_lr_high_count_{stage_name}"
    low_count_key = f"actor_lr_low_count_{stage_name}"
    high_count = int(hparams.get(high_count_key, 0) or 0)
    low_count = int(hparams.get(low_count_key, 0) or 0)
    high_count = high_count + 1 if high else 0
    low_count = low_count + 1 if (low and not high) else 0
    hparams[high_count_key] = int(high_count)
    hparams[low_count_key] = int(low_count)
    out[f"actor_lr_high_count_{stage_name}"] = float(high_count)
    out[f"actor_lr_low_count_{stage_name}"] = float(low_count)

    decay_patience = max(int(hparams.get("stage_actor_lr_decay_patience", 3) or 0), 1)
    grow_patience = max(int(hparams.get("stage_actor_lr_grow_patience", 10) or 0), 1)
    decay = min(max(float(hparams.get("stage_actor_lr_decay_factor", 0.5)), 0.0), 1.0)
    grow = max(float(hparams.get("stage_actor_lr_grow_factor", 1.25)), 1.0)
    lr_min = max(float(hparams.get(f"actor_lr_min_{stage_name}", 0.0)), 0.0)
    lr_max = max(float(hparams.get(f"actor_lr_max_{stage_name}", float("inf"))), lr_min)
    if hard_high or high_count >= decay_patience:
        next_lr = max(float(current_lr) * float(decay), float(lr_min))
        if next_lr < current_lr:
            _set_optimizer_lr(optimizer, next_lr)
            hparams[high_count_key] = 0
            hparams[low_count_key] = 0
            out[f"actor_lr_next_{stage_name}"] = float(next_lr)
            out[f"actor_lr_decayed_{stage_name}"] = 1.0
            out[f"actor_lr_high_count_{stage_name}"] = 0.0
            out[f"actor_lr_low_count_{stage_name}"] = 0.0
        return out
    if low_count >= grow_patience:
        next_lr = min(float(current_lr) * float(grow), float(lr_max))
        if next_lr > current_lr:
            _set_optimizer_lr(optimizer, next_lr)
            hparams[high_count_key] = 0
            hparams[low_count_key] = 0
            out[f"actor_lr_next_{stage_name}"] = float(next_lr)
            out[f"actor_lr_grew_{stage_name}"] = 1.0
            out[f"actor_lr_high_count_{stage_name}"] = 0.0
            out[f"actor_lr_low_count_{stage_name}"] = 0.0
        return out
    out[f"actor_lr_next_{stage_name}"] = float(current_lr)
    return out


def _ev_is_finite(value: float) -> bool:
    return math.isfinite(float(value))


def _skipped_stage_actor_stats(
    *,
    stage_id: int,
    stage_batch: Any,
    stage_advantages: torch.Tensor,
    hparams: dict[str, float | int],
    actor_optimizers: dict[int, torch.optim.Optimizer],
) -> dict[str, float]:
    stage_name = STAGE_NAME[int(stage_id)]
    suffix = _stage_metric_suffix(int(stage_id))
    adv = stage_advantages.detach().to(dtype=torch.float32).reshape(-1)
    current_lr = float(
        actor_optimizers[int(stage_id)].param_groups[0].get("lr", hparams.get(f"actor_lr_{suffix}", 0.0))
    )
    return {
        "actor_samples": float(int(getattr(stage_batch, "num_samples", int(adv.numel())))),
        "actor_epochs": float(hparams.get("actor_epochs", 0)),
        "actor_epochs_completed": 0.0,
        "actor_early_stop": 0.0,
        "actor_early_stop_epoch": -1.0,
        "actor_early_stop_reason_code": 0.0,
        "actor_kl_stop_threshold": float(hparams.get(f"actor_kl_stop_threshold_{suffix}", 0.0)),
        "actor_clip_stop_threshold": 0.0,
        "actor_minibatches": float(hparams.get("actor_minibatches", 0)),
        "actor_outer_minibatches": 0.0,
        f"policy_loss_{stage_name}": 0.0,
        f"entropy_{stage_name}": 0.0,
        f"actor_loss_{stage_name}": 0.0,
        f"grad_norm_{stage_name}": 0.0,
        f"approx_kl_{stage_name}": 0.0,
        f"clip_frac_{stage_name}": 0.0,
        "old_logprob_mean": float("nan"),
        "old_logprob_parity_abs_mean": float("nan"),
        "old_logprob_parity_abs_max": float("nan"),
        "old_logprob_parity_bad_frac": float("nan"),
        "adv_mean": float(adv.mean().detach().cpu().item()) if int(adv.numel()) > 0 else 0.0,
        "adv_std": float(adv.std(unbiased=False).detach().cpu().item()) if int(adv.numel()) > 1 else 0.0,
        "actor_old_logprob_sec": 0.0,
        "actor_update_loop_sec": 0.0,
        "actor_compile_enabled": 0.0,
        "actor_chunk_size": 0.0,
        "danger_imitation_loss": 0.0,
        "danger_imitation_active_rate": 0.0,
        f"actor_lr_used_{suffix}": current_lr,
        f"actor_lr_next_{suffix}": current_lr,
        f"actor_lr_decayed_{suffix}": 0.0,
        f"actor_lr_grew_{suffix}": 0.0,
        f"actor_lr_high_count_{suffix}": float(hparams.get(f"actor_lr_high_count_{suffix}", 0)),
        f"actor_lr_low_count_{suffix}": float(hparams.get(f"actor_lr_low_count_{suffix}", 0)),
        f"actor_skipped_by_critic_ev_{suffix}": 1.0,
    }


def _rng_state_payload(device: torch.device) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        payload["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return payload


def _restore_rng_state_payload(payload: Any, device: torch.device) -> None:
    if not isinstance(payload, dict):
        return
    python_state = payload.get("python_random")
    if python_state is not None:
        random.setstate(python_state)
    numpy_state = payload.get("numpy_random")
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    torch_cpu = payload.get("torch_cpu")
    if torch_cpu is not None:
        torch.set_rng_state(torch_cpu.detach().cpu() if torch.is_tensor(torch_cpu) else torch_cpu)
    torch_cuda_all = payload.get("torch_cuda_all")
    if device.type == "cuda" and torch.cuda.is_available() and torch_cuda_all is not None:
        torch.cuda.set_rng_state_all(
            [state.detach().cpu() if torch.is_tensor(state) else state for state in torch_cuda_all]
        )


def _save_joint_checkpoint(
    path: Path,
    *,
    learner: StructuredMAPPO,
    critic_optimizer: torch.optim.Optimizer,
    actor_optimizers: dict[int, torch.optim.Optimizer],
    cfg: Any,
    args: argparse.Namespace,
    hparams: dict[str, float],
    update: int,
    completed: bool,
    device: torch.device,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": learner.actor.state_dict(),
            "critic": learner.critic.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "actor_optimizers": {int(stage_id): opt.state_dict() for stage_id, opt in actor_optimizers.items()},
            "cfg": vars(cfg),
            "args": vars(args),
            "hparams": dict(hparams),
            "update": int(update),
            "completed": bool(completed),
            "rng_state": _rng_state_payload(device),
        },
        path,
    )


def _stage_actor_state_dict(actor: torch.nn.Module, stage_id: int) -> dict[str, torch.Tensor]:
    prefix = STAGE_ACTOR_PREFIX[int(stage_id)]
    state = actor.state_dict()
    return {
        key: value.detach().cpu().clone()
        for key, value in state.items()
        if key.startswith(prefix)
    }


def _save_stage_best_checkpoint(
    path: Path,
    *,
    learner: StructuredMAPPO,
    cfg: Any,
    args: argparse.Namespace,
    hparams: dict[str, float],
    update: int,
    stage_id: int,
    metric_name: str,
    metric_value: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage_name = STAGE_NAME[int(stage_id)]
    torch.save(
        {
            "actor": _stage_actor_state_dict(learner.actor, int(stage_id)),
            "stage_id": int(stage_id),
            "stage_name": str(stage_name),
            "stage_actor_prefix": STAGE_ACTOR_PREFIX[int(stage_id)],
            "metric_name": str(metric_name),
            "metric_value": float(metric_value),
            "update": int(update),
            "cfg": vars(cfg),
            "args": vars(args),
            "hparams": dict(hparams),
        },
        path,
    )


def _load_existing_metrics(path: Path, *, max_update: int | None = None) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row: dict[str, float] = {}
            for key, value in raw.items():
                if key is None or value is None or value == "":
                    continue
                try:
                    row[str(key)] = float(value)
                except ValueError:
                    continue
            if max_update is not None and row.get("update", float("inf")) > float(max_update):
                continue
            rows.append(row)
    return rows


def _canonical_return_target(target: str) -> str:
    target_l = str(target).strip().lower()
    if target_l == "bootstrap_gae":
        target_l = "train_gae"
    if target_l in {"n_step", "n-step"}:
        target_l = "nstep"
    if target_l not in {"mc", "train_gae", "mixed", "bootstrap_mc_aux", "nstep"}:
        raise ValueError(
            "return_target must be one of {'mc', 'bootstrap_gae', 'mixed', 'bootstrap_mc_aux', 'nstep'}."
        )
    return target_l


def _return_target_label(target: str) -> str:
    target_l = _canonical_return_target(target)
    if target_l == "train_gae":
        return "bootstrap_gae"
    return target_l


def _return_target_code(target: str) -> float:
    return {
        "mc": 0.0,
        "train_gae": 1.0,
        "mixed": 2.0,
        "bootstrap_mc_aux": 3.0,
        "nstep": 4.0,
    }[_canonical_return_target(target)]


def _target_mix_alpha_for_update(
    *,
    update: int,
    alpha_start: float,
    alpha_end: float,
    warmup_updates: int,
    anneal_updates: int,
) -> float:
    update_i = max(int(update), 0)
    warmup_i = max(int(warmup_updates), 0)
    anneal_i = max(int(anneal_updates), 0)
    start = float(alpha_start)
    end = float(alpha_end)
    if update_i < warmup_i:
        return float(start)
    if anneal_i <= 0:
        return float(end)
    progress = min(max((update_i - warmup_i) / float(anneal_i), 0.0), 1.0)
    return float(start + (end - start) * progress)


def _target_advantage_after_critic(target: str) -> bool:
    return _canonical_return_target(target) in {"mixed", "nstep"}


def _return_target_for_update(
    *,
    base_return_target: str,
    schedule: str,
    update: int,
    switch_update: int,
) -> str:
    schedule_l = str(schedule).strip().lower()
    if schedule_l == "fixed":
        return _canonical_return_target(base_return_target)
    if schedule_l == "mc_then_bootstrap":
        if int(switch_update) < 0:
            raise ValueError("--target_switch_update must be >= 0.")
        return "mc" if int(update) < int(switch_update) else "train_gae"
    raise ValueError("return_target_schedule must be one of {'fixed', 'mc_then_bootstrap'}.")


def _collect_joint_rollout(
    learner: StructuredMAPPO,
    group: Any,
    *,
    rollout_env_steps: int,
    device: torch.device,
    seed: int,
    return_target: str,
    target_mix_alpha: float = 0.0,
    return_nstep_horizon: int = 1,
) -> tuple[
    Any,
    torch.Tensor,
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[str, float],
]:
    target_l = _canonical_return_target(return_target)
    reset_many = getattr(group, "reset_many", None)
    if callable(reset_many):
        reset_many([int(seed) + env for env in range(len(group))])

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
    views = buffer.build_rollout_views(device)
    returns, advantages, _value_override, target_info = compute_returns_for_views(
        learner,
        buffer,
        views,
        target=target_l,
        device=device,
        target_mix_alpha=float(target_mix_alpha),
        return_nstep_horizon=int(return_nstep_horizon),
        return_info=True,
    )
    del buffer
    stage_targets: dict[int, torch.Tensor] = {}
    stage_advantages: dict[int, torch.Tensor] = {}
    stage_aux_targets: dict[int, torch.Tensor] = {}
    extra_stage_targets: dict[str, dict[int, torch.Tensor]] = {}
    extra_return_vectors = {
        "mc": target_info.get("mc_returns"),
        "bootstrap_gae": target_info.get("bootstrap_gae_returns"),
        "mc_aux": target_info.get("mc_aux_returns"),
    }
    for stage_id in STAGES:
        stage_batch = views.training_view.stage_batches.get(int(stage_id))
        if stage_batch is None or int(stage_batch.num_samples) <= 0:
            raise RuntimeError(f"joint rollout contains no {STAGE_NAME[int(stage_id)]} stage samples.")
        idx = _stage_indices(stage_batch, device=device)
        stage_targets[int(stage_id)] = returns.index_select(0, idx).detach().to(device=device, dtype=torch.float32)
        stage_advantages[int(stage_id)] = (
            advantages.index_select(0, idx).detach().to(device=device, dtype=torch.float32)
        )
        for label, vector in extra_return_vectors.items():
            if not torch.is_tensor(vector):
                continue
            extra_stage_targets.setdefault(str(label), {})[int(stage_id)] = vector.index_select(0, idx).detach().to(
                device=device,
                dtype=torch.float32,
            )
    if target_l == "bootstrap_mc_aux" and "mc_aux" in extra_stage_targets:
        stage_aux_targets = dict(extra_stage_targets["mc_aux"])

    reward_stats: dict[str, float] = {}
    target_label = _return_target_label(target_l)
    reward_stats["return_target_code"] = _return_target_code(target_l)
    reward_stats["active_return_target_code"] = _return_target_code(target_l)
    reward_stats["target_mix_alpha"] = float(target_info.get("target_mix_alpha", target_mix_alpha))
    reward_stats["return_nstep_horizon"] = float(target_info.get("return_nstep_horizon", return_nstep_horizon))
    rewards = getattr(views.training_view, "rewards", None)
    if rewards is not None:
        rt = torch.as_tensor(rewards, dtype=torch.float32, device=device).reshape(-1)
        reward_stats["transition_reward_mean"] = float(rt.mean().detach().cpu().item())
        reward_stats["transition_reward_std"] = float(rt.std(unbiased=False).detach().cpu().item())
        reward_stats["transition_reward_min"] = float(rt.min().detach().cpu().item())
        reward_stats["transition_reward_max"] = float(rt.max().detach().cpu().item())
    for stage_id, target in stage_targets.items():
        name = STAGE_NAME[int(stage_id)]
        stats = _summ_tensor(target)
        for stat_name, stat_value in stats.items():
            reward_stats[f"{name}_target_return_{stat_name}"] = float(stat_value)
            reward_stats[f"{name}_{target_label}_return_{stat_name}"] = float(stat_value)
        for extra_label, targets_by_stage in extra_stage_targets.items():
            extra_target = targets_by_stage.get(int(stage_id))
            if extra_target is None:
                continue
            extra_stats = _summ_tensor(extra_target)
            for stat_name, stat_value in extra_stats.items():
                reward_stats[f"{name}_{extra_label}_return_{stat_name}"] = float(stat_value)
        if target_l == "mc":
            reward_stats[f"{name}_mc_return_mean"] = float(stats["mean"])
            reward_stats[f"{name}_mc_return_std"] = float(stats["std"])
            reward_stats[f"{name}_mc_return_min"] = float(stats["min"])
            reward_stats[f"{name}_mc_return_max"] = float(stats["max"])
        mc_extra = extra_stage_targets.get("mc", {}).get(int(stage_id))
        gae_extra = extra_stage_targets.get("bootstrap_gae", {}).get(int(stage_id))
        if mc_extra is not None and gae_extra is not None:
            diff_stats = _summ_tensor(mc_extra - gae_extra)
            for stat_name, stat_value in diff_stats.items():
                reward_stats[f"{name}_mc_gae_target_diff_{stat_name}"] = float(stat_value)
    return views, returns, stage_targets, stage_advantages, stage_aux_targets, reward_stats


def _rollout_diagnostics_from_views(views: Any) -> dict[str, float]:
    """Aggregate rollout-level safety diagnostics without per-step Python work."""

    out: dict[str, float] = {}

    def _add_np_summary(prefix: str, values: list[float] | np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size <= 0:
            return
        out[f"{prefix}_mean"] = float(np.mean(arr))
        out[f"{prefix}_std"] = float(np.std(arr))
        out[f"{prefix}_min"] = float(np.min(arr))
        out[f"{prefix}_p10"] = float(np.quantile(arr, 0.10))
        out[f"{prefix}_p25"] = float(np.quantile(arr, 0.25))
        out[f"{prefix}_p50"] = float(np.quantile(arr, 0.50))
        out[f"{prefix}_p75"] = float(np.quantile(arr, 0.75))
        out[f"{prefix}_p90"] = float(np.quantile(arr, 0.90))
        out[f"{prefix}_max"] = float(np.max(arr))

    try:
        accel_stage = views.training_view.stage_batches.get(0)
        sat_macro_stage = views.training_view.stage_batches.get(1)
        bw_macro_stage = views.training_view.stage_batches.get(2)
        bw_return_stage = views.return_view.stage_batches.get(2)
    except AttributeError:
        return out

    if accel_stage is not None and torch.is_tensor(getattr(accel_stage, "actions", None)):
        actions = accel_stage.actions.detach()
        if actions.numel() > 0 and actions.ndim >= 3:
            norms = torch.linalg.vector_norm(actions.to(dtype=torch.float32), dim=-1)
            out["accel_policy_action_norm_mean"] = float(norms.mean().detach().cpu().item())
            out["accel_policy_action_norm_top1"] = float(norms.max().detach().cpu().item())
        latent = getattr(accel_stage, "latent_actions", None)
        if torch.is_tensor(latent) and latent.numel() > 0 and latent.ndim >= 3:
            latent_norms = torch.linalg.vector_norm(latent.detach().to(dtype=torch.float32), dim=-1)
            out["accel_latent_action_norm_mean"] = float(latent_norms.mean().detach().cpu().item())
            out["accel_latent_action_norm_top1"] = float(latent_norms.max().detach().cpu().item())

    if bw_return_stage is None:
        return out

    for macro_name, macro_stage in (("sat", sat_macro_stage), ("bw", bw_macro_stage)):
        duration_value = getattr(macro_stage, "duration", None) if macro_stage is not None else None
        if torch.is_tensor(duration_value):
            durations = duration_value.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
        elif duration_value is None:
            durations = np.ones((int(getattr(macro_stage, "num_samples", 0) or 0),), dtype=np.int64)
        else:
            durations = np.asarray(duration_value, dtype=np.int64).reshape(-1)
        if durations.size > 0:
            durations = np.maximum(durations, 1)
            out[f"{macro_name}_macro_duration_mean"] = float(np.mean(durations))
            out[f"{macro_name}_macro_duration_min"] = float(np.min(durations))
            out[f"{macro_name}_macro_duration_max"] = float(np.max(durations))
            out[f"{macro_name}_macro_transition_count"] = float(durations.size)

    reward_parts = getattr(bw_return_stage, "reward_part_arrays", {}) or {}

    def _part(name: str) -> np.ndarray | None:
        value = reward_parts.get(name)
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        return arr if arr.size > 0 else None

    def _add_part_summary(source_key: str, metric_prefix: str | None = None) -> None:
        arr = _part(source_key)
        if arr is None:
            return
        _add_np_summary(metric_prefix or f"rollout_{source_key}", arr)

    def _add_ratio_summary(prefix: str, numerator_key: str, denominator_key: str) -> None:
        numerator = _part(numerator_key)
        denominator = _part(denominator_key)
        if numerator is None or denominator is None or numerator.size != denominator.size:
            return
        numerator64 = numerator.astype(np.float64, copy=False)
        denominator64 = denominator.astype(np.float64, copy=False)
        valid = (
            np.isfinite(numerator64)
            & np.isfinite(denominator64)
            & (denominator64 > np.finfo(np.float64).tiny)
        )
        if not np.any(valid):
            return
        _add_np_summary(prefix, numerator64[valid] / denominator64[valid])

    # Per-rollout training curves used in the thesis figures.  These arrays are
    # already exported by the native rollout runtime; this block only aggregates
    # them once per update, outside the environment hot path.
    for source_key in (
        "processed_ratio_eval",
        "drop_ratio_eval",
        "pre_backlog_steps_eval",
        "D_sys_report",
        "gu_queue_sum",
        "uav_queue_sum",
        "sat_queue_sum",
        "queue_total_sum",
        "drop_sum",
        "drop_sum_active",
        "expire_sum",
        "gu_drop_sum",
        "uav_drop_sum",
        "sat_drop_sum",
        "arrival_sum",
        "arrival_ref",
        "outflow_sum",
        "backhaul_sum",
        "sat_processed_sum",
        "overflow_risk_mean",
        "downstream_pressure_mean",
        "service_gap_mean",
        "service_gap_risk_mean",
        "reward_raw",
    ):
        _add_part_summary(source_key)

    # Normalized flow ratios for access/backhaul/processing curves.  We keep
    # both realized-arrival and configured-reference denominators because the
    # former shows how much of this rollout's demand moved, while the latter is
    # stable across stochastic traffic realizations.
    for denominator_key, suffix in (("arrival_sum", "arrival"), ("arrival_ref", "arrival_ref")):
        _add_ratio_summary(f"rollout_access_outflow_per_{suffix}", "outflow_sum", denominator_key)
        _add_ratio_summary(f"rollout_backhaul_per_{suffix}", "backhaul_sum", denominator_key)
        _add_ratio_summary(f"rollout_sat_processed_per_{suffix}", "sat_processed_sum", denominator_key)
        _add_ratio_summary(f"rollout_drop_per_{suffix}", "drop_sum", denominator_key)

    for source_key, metric_key, max_key in (
        ("collision_event", "rollout_collision_rate", None),
        ("intervention_rate", "rollout_intervention_rate", "rollout_intervention_rate_max"),
        ("intervention_norm", "rollout_policy_exec_diff_norm_mean", "rollout_policy_exec_diff_norm_max"),
        (
            "intervention_norm_top1",
            "rollout_policy_exec_diff_norm_top1_mean",
            "rollout_policy_exec_diff_norm_top1_max",
        ),
        ("danger_imitation_active_rate", "rollout_danger_imitation_active_rate", None),
        ("close_risk", "rollout_close_risk_mean", None),
        ("term_close_risk", "rollout_term_close_risk_rate", None),
    ):
        arr = _part(source_key)
        if arr is None:
            continue
        out[metric_key] = float(np.mean(arr))
        if source_key in {"collision_event", "term_close_risk"}:
            out[f"{metric_key}_count"] = float(np.sum(arr > 0.5))
        if max_key is not None:
            out[max_key] = float(np.max(arr))

    if torch.is_tensor(getattr(bw_return_stage, "terminated", None)):
        terminated = bw_return_stage.terminated.detach().cpu().numpy().astype(bool, copy=False).reshape(-1)
    else:
        terminated = np.asarray(getattr(bw_return_stage, "terminated", []), dtype=bool).reshape(-1)
    if torch.is_tensor(getattr(bw_return_stage, "truncated", None)):
        truncated = bw_return_stage.truncated.detach().cpu().numpy().astype(bool, copy=False).reshape(-1)
    else:
        truncated = np.asarray(getattr(bw_return_stage, "truncated", []), dtype=bool).reshape(-1)
    if torch.is_tensor(getattr(bw_return_stage, "rewards", None)):
        rewards = bw_return_stage.rewards.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    else:
        rewards = np.asarray(getattr(bw_return_stage, "rewards", []), dtype=np.float64).reshape(-1)
    env_indices = np.asarray(getattr(bw_return_stage, "env_indices", []), dtype=np.int64).reshape(-1)
    primitive_durations = np.ones((terminated.size,), dtype=np.int64)
    if (
        terminated.size > 0
        and truncated.size == terminated.size
        and env_indices.size == terminated.size
    ):
        done = np.logical_or(terminated, truncated)
        out["rollout_terminated_rate"] = float(np.mean(terminated))
        out["rollout_truncated_rate"] = float(np.mean(truncated))
        out["rollout_done_count"] = float(np.sum(done))

        lengths: list[int] = []
        completed_lengths: list[int] = []
        episode_rewards: list[float] = []
        completed_episode_rewards: list[float] = []
        tail_count = 0
        has_rewards = rewards.size == terminated.size
        for env_index in np.unique(env_indices):
            positions = np.nonzero(env_indices == env_index)[0]
            current_len = 0
            current_reward = 0.0
            for pos in positions:
                current_len += int(primitive_durations[pos])
                if has_rewards:
                    current_reward += float(rewards[pos])
                if bool(done[pos]):
                    lengths.append(current_len)
                    completed_lengths.append(current_len)
                    if has_rewards:
                        episode_rewards.append(current_reward)
                        completed_episode_rewards.append(current_reward)
                    current_len = 0
                    current_reward = 0.0
            if current_len > 0:
                lengths.append(current_len)
                if has_rewards:
                    episode_rewards.append(current_reward)
                tail_count += 1
        if lengths:
            length_arr = np.asarray(lengths, dtype=np.float32)
            out["rollout_episode_length_mean"] = float(np.mean(length_arr))
            out["rollout_episode_length_min"] = float(np.min(length_arr))
            out["rollout_episode_length_max"] = float(np.max(length_arr))
            out["rollout_episode_segment_count"] = float(len(lengths))
            out["rollout_episode_tail_count"] = float(tail_count)
        if completed_lengths:
            completed_arr = np.asarray(completed_lengths, dtype=np.float32)
            out["rollout_completed_episode_length_mean"] = float(np.mean(completed_arr))
            out["rollout_completed_episode_length_min"] = float(np.min(completed_arr))
            out["rollout_completed_episode_count"] = float(len(completed_lengths))
        if has_rewards:
            _add_np_summary("rollout_episode_reward", episode_rewards)
            _add_np_summary("rollout_completed_episode_reward", completed_episode_rewards)
    return out


def _resolve_training_hparams(cfg: Any, args: argparse.Namespace) -> dict[str, float | int]:
    generic_actor_lr = float(
        args.actor_lr
        if args.actor_lr is not None
        else _stage_cfg_value(cfg, "stage_mcgae_actor_lr", "sat_mcgae_actor_lr", 3.0e-4)
    )
    stage_actor_lrs = {
        0: float(
            args.accel_actor_lr
            if args.accel_actor_lr is not None
            else _stage_cfg_value(cfg, "stage_mcgae_accel_actor_lr", "stage_mcgae_actor_lr", generic_actor_lr)
        ),
        1: float(
            args.sat_actor_lr
            if args.sat_actor_lr is not None
            else _stage_cfg_value(cfg, "stage_mcgae_sat_actor_lr", "stage_mcgae_actor_lr", generic_actor_lr)
        ),
        2: float(
            args.bw_actor_lr
            if args.bw_actor_lr is not None
            else _stage_cfg_value(cfg, "stage_mcgae_bw_actor_lr", "stage_mcgae_actor_lr", generic_actor_lr)
        ),
    }
    target_kls = {
        0: float(_stage_cfg_value(cfg, "stage_actor_target_kl_accel", "stage_actor_target_kl", 0.02)),
        1: float(_stage_cfg_value(cfg, "stage_actor_target_kl_sat", "stage_actor_target_kl", 0.02)),
        2: float(_stage_cfg_value(cfg, "stage_actor_target_kl_bw", "stage_actor_target_kl", 0.02)),
    }
    lr_mins = {
        0: float(_stage_cfg_value(cfg, "stage_actor_lr_min_accel", "stage_actor_lr_min", 3.0e-6)),
        1: float(_stage_cfg_value(cfg, "stage_actor_lr_min_sat", "stage_actor_lr_min", 1.0e-5)),
        2: float(_stage_cfg_value(cfg, "stage_actor_lr_min_bw", "stage_actor_lr_min", 1.0e-5)),
    }
    lr_maxs = {
        0: float(_stage_cfg_value(cfg, "stage_actor_lr_max_accel", "stage_actor_lr_max", 6.0e-4)),
        1: float(_stage_cfg_value(cfg, "stage_actor_lr_max_sat", "stage_actor_lr_max", 2.0e-3)),
        2: float(_stage_cfg_value(cfg, "stage_actor_lr_max_bw", "stage_actor_lr_max", 2.0e-3)),
    }
    hparams: dict[str, float | int] = {
        "cold_critic_lr": float(
            args.cold_critic_lr
            if args.cold_critic_lr is not None
            else _stage_cfg_value(cfg, "stage_mcgae_cold_critic_lr", "sat_mcgae_cold_critic_lr", 1.0e-3)
        ),
        "cold_critic_epochs": int(
            args.cold_critic_epochs
            if args.cold_critic_epochs is not None
            else _stage_cfg_value(cfg, "stage_mcgae_cold_critic_epochs", "sat_mcgae_cold_critic_epochs", 20)
        ),
        "tracking_critic_lr": float(
            args.tracking_critic_lr
            if args.tracking_critic_lr is not None
            else _stage_cfg_value(cfg, "stage_mcgae_tracking_critic_lr", "sat_mcgae_tracking_critic_lr", 3.0e-4)
        ),
        "tracking_critic_epochs": int(
            args.tracking_critic_epochs
            if args.tracking_critic_epochs is not None
            else _stage_cfg_value(cfg, "stage_mcgae_tracking_critic_epochs", "sat_mcgae_tracking_critic_epochs", 5)
        ),
        "critic_minibatches": int(
            args.critic_minibatches
            if args.critic_minibatches is not None
            else _stage_cfg_value(cfg, "stage_mcgae_critic_minibatches", "sat_mcgae_critic_minibatches", 8)
        ),
        "critic_update_microbatch_size": int(
            args.critic_update_microbatch_size
            if args.critic_update_microbatch_size is not None
            else _stage_cfg_value(
                cfg,
                "stage_mcgae_critic_update_microbatch_size",
                "sat_mcgae_critic_update_microbatch_size",
                0,
            )
        ),
        "critic_ev_gate_enabled": int(
            bool(_stage_cfg_value(cfg, "stage_mcgae_critic_ev_gate_enabled", "sat_mcgae_critic_ev_gate_enabled", False))
        ),
        "critic_ev_soft_target": float(
            _stage_cfg_value(cfg, "stage_mcgae_critic_ev_soft_target", "sat_mcgae_critic_ev_soft_target", 0.96)
        ),
        "critic_ev_hard_floor": float(
            _stage_cfg_value(cfg, "stage_mcgae_critic_ev_hard_floor", "sat_mcgae_critic_ev_hard_floor", 0.90)
        ),
        "critic_ev_extra_epochs": int(
            _stage_cfg_value(cfg, "stage_mcgae_critic_ev_extra_epochs", "sat_mcgae_critic_ev_extra_epochs", 3)
        ),
        "critic_ev_max_retries": int(
            _stage_cfg_value(cfg, "stage_mcgae_critic_ev_max_retries", "sat_mcgae_critic_ev_max_retries", 2)
        ),
        "actor_lr": float(generic_actor_lr),
        "actor_epochs": int(
            args.actor_epochs
            if args.actor_epochs is not None
            else _stage_cfg_value(cfg, "stage_mcgae_actor_epochs", "sat_mcgae_actor_epochs", 5)
        ),
        "actor_minibatches": int(
            args.actor_minibatches
            if args.actor_minibatches is not None
            else _stage_cfg_value(cfg, "stage_mcgae_actor_minibatches", "sat_mcgae_actor_minibatches", 1)
        ),
        "stage_actor_kl_early_stop_enabled": int(
            bool(getattr(cfg, "stage_actor_kl_early_stop_enabled", False))
        ),
        "stage_actor_dynamic_lr_enabled": int(
            bool(getattr(cfg, "stage_actor_dynamic_lr_enabled", False))
        ),
        "stage_actor_kl_stop_multiplier": float(
            getattr(cfg, "stage_actor_kl_stop_multiplier", 1.5) or 1.5
        ),
        "stage_actor_lr_decay_factor": float(
            getattr(cfg, "stage_actor_lr_decay_factor", 0.5) or 0.5
        ),
        "stage_actor_lr_grow_factor": float(
            getattr(cfg, "stage_actor_lr_grow_factor", 1.25) or 1.25
        ),
        "stage_actor_lr_decay_patience": int(
            getattr(cfg, "stage_actor_lr_decay_patience", 3) or 3
        ),
        "stage_actor_lr_grow_patience": int(
            getattr(cfg, "stage_actor_lr_grow_patience", 10) or 10
        ),
        "stage_actor_lr_high_kl_frac": float(
            getattr(cfg, "stage_actor_lr_high_kl_frac", 1.5) or 1.5
        ),
        "stage_actor_lr_low_kl_frac": float(
            getattr(cfg, "stage_actor_lr_low_kl_frac", 0.3) or 0.3
        ),
        "stage_actor_lr_high_clip": float(
            getattr(cfg, "stage_actor_lr_high_clip", 0.3) or 0.3
        ),
        "stage_actor_lr_low_clip": float(
            getattr(cfg, "stage_actor_lr_low_clip", 0.05) or 0.05
        ),
        "stage_actor_lr_hard_kl_frac": float(
            getattr(cfg, "stage_actor_lr_hard_kl_frac", 3.0) or 3.0
        ),
        "stage_actor_lr_hard_clip": float(
            getattr(cfg, "stage_actor_lr_hard_clip", 0.6) or 0.6
        ),
    }
    for stage_id in STAGES:
        suffix = _stage_metric_suffix(int(stage_id))
        hparams[f"actor_lr_{suffix}"] = float(stage_actor_lrs[int(stage_id)])
        hparams[f"actor_lr_min_{suffix}"] = float(lr_mins[int(stage_id)])
        hparams[f"actor_lr_max_{suffix}"] = float(lr_maxs[int(stage_id)])
        hparams[f"actor_target_kl_{suffix}"] = float(target_kls[int(stage_id)])
        hparams[f"actor_kl_stop_threshold_{suffix}"] = float(
            target_kls[int(stage_id)] * float(hparams["stage_actor_kl_stop_multiplier"])
        )
        hparams[f"actor_lr_high_count_{suffix}"] = 0
        hparams[f"actor_lr_low_count_{suffix}"] = 0
    return hparams


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint accel/sat/bw MC-critic + A_gae(V) training loop.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--rollout_env_steps", type=int, default=250)
    parser.add_argument(
        "--return_target",
        choices=("mc", "bootstrap_gae", "mixed", "bootstrap_mc_aux", "nstep"),
        default="mc",
        help=(
            "Return/advantage target for the joint loop. 'mc' preserves the formal mainline; "
            "'bootstrap_gae' uses the existing ordinary critic-bootstrapped GAE target path; "
            "'mixed' linearly mixes MC and bootstrap-GAE targets; 'bootstrap_mc_aux' trains "
            "the critic with a MC auxiliary loss; 'nstep' uses truncated MC with value bootstrap."
        ),
    )
    parser.add_argument(
        "--return_target_schedule",
        choices=("fixed", "mc_then_bootstrap"),
        default="fixed",
        help=(
            "Schedule for selecting the active return target each update. 'fixed' preserves the "
            "legacy behavior; 'mc_then_bootstrap' uses MC before --target_switch_update and "
            "bootstrap-GAE afterwards."
        ),
    )
    parser.add_argument(
        "--target_switch_update",
        type=int,
        default=100,
        help=(
            "Zero-based update index where mc_then_bootstrap switches to bootstrap-GAE. "
            "The default 100 means updates 1-100 use MC and updates 101+ use bootstrap-GAE."
        ),
    )
    parser.add_argument("--target_mix_alpha_start", type=float, default=1.0)
    parser.add_argument("--target_mix_alpha_end", type=float, default=0.3)
    parser.add_argument("--target_mix_warmup_updates", type=int, default=50)
    parser.add_argument("--target_mix_anneal_updates", type=int, default=200)
    parser.add_argument("--mc_aux_critic_coef", type=float, default=0.3)
    parser.add_argument("--return_nstep_horizon", type=int, default=64)
    parser.add_argument(
        "--structured_env_backend",
        default=None,
        help="Override cfg.structured_env_backend, e.g. 'python' for Mac/CPU smoke runs.",
    )
    parser.add_argument(
        "--structured_env_tensor_backend",
        default=None,
        help="Override cfg.structured_env_tensor_backend, e.g. 'cpu' when CUDA is unavailable.",
    )
    parser.add_argument("--reward_mode", default="positive_weighted_workload_level")
    parser.add_argument("--cold_critic_lr", type=float, default=None)
    parser.add_argument("--cold_critic_epochs", type=int, default=None)
    parser.add_argument("--tracking_critic_lr", type=float, default=None)
    parser.add_argument("--tracking_critic_epochs", type=int, default=None)
    parser.add_argument("--critic_minibatches", type=int, default=None)
    parser.add_argument("--critic_update_microbatch_size", type=int, default=None)
    parser.add_argument("--actor_lr", type=float, default=None)
    parser.add_argument("--accel_actor_lr", type=float, default=None)
    parser.add_argument("--sat_actor_lr", type=float, default=None)
    parser.add_argument("--bw_actor_lr", type=float, default=None)
    parser.add_argument("--actor_optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--accel_actor_optimizer", choices=("adam", "sgd"), default=None)
    parser.add_argument("--sat_actor_optimizer", choices=("adam", "sgd"), default=None)
    parser.add_argument("--bw_actor_optimizer", choices=("adam", "sgd"), default=None)
    parser.add_argument("--bw_fixed_tau", type=float, default=None)
    parser.add_argument("--bw_fixed_kappa", type=float, default=None)
    parser.add_argument("--actor_epochs", type=int, default=None)
    parser.add_argument("--actor_minibatches", type=int, default=None)
    parser.add_argument("--access_bw_decision_interval", type=int, default=None)
    parser.add_argument("--sat_decision_interval", type=int, default=None)
    parser.add_argument(
        "--disable_danger_imitation",
        action="store_true",
        help="Disable the accel danger-imitation auxiliary loss after applying the joint training safety defaults.",
    )
    parser.add_argument(
        "--disable_torch_compile",
        action="store_true",
        help="Disable torch.compile-backed actor/critic paths; useful for Windows CUDA smoke runs.",
    )
    parser.add_argument("--seed", type=int, default=45210)
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument(
        "--disable_stage_best_save",
        action="store_true",
        help="Disable lightweight per-stage best-head checkpoints based on each stage target-return mean.",
    )
    parser.add_argument(
        "--stage_best_dir",
        default=None,
        help="Directory for per-stage best-head checkpoints. Defaults to <run_dir>/best_stage_heads.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from a joint MC-GAE checkpoint. --updates is interpreted as the target total update count.",
    )
    parser.add_argument("--bw_parity_topk", type=int, default=16)
    parser.add_argument(
        "--diagnose_critic_timing",
        action="store_true",
        help="Record per-stage critic timing rows without changing the update math.",
    )
    args = parser.parse_args()

    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    _set_seed(int(args.seed))

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    _force_joint_config(cfg, reward_mode=str(args.reward_mode))
    if bool(args.disable_torch_compile):
        cfg.critic_compile_enabled = False
        cfg.stage_actor_compile_enabled = False
        cfg.accel_actor_compile_enabled = False
        cfg.sat_actor_compile_enabled = False
        cfg.bw_actor_compile_enabled = False
    if args.structured_env_backend is not None:
        cfg.structured_env_backend = str(args.structured_env_backend)
    if args.structured_env_tensor_backend is not None:
        cfg.structured_env_tensor_backend = str(args.structured_env_tensor_backend)
    env_backend_l = str(getattr(cfg, "structured_env_backend", "") or "").strip().lower()
    env_tensor_backend_l = str(getattr(cfg, "structured_env_tensor_backend", "") or "").strip().lower()
    if (
        (env_backend_l in {"python", "cpu", "mac", "legacy"} or env_tensor_backend_l == "cpu")
        and str(getattr(cfg, "safety_shield_solver", "") or "").strip().upper() == "NATIVE_CUDA"
    ):
        # The native fused shield is a CUDA kernel.  Keep Mac smoke runs
        # dependency-light by disabling it instead of requiring cvxpy.
        cfg.safety_shield_enabled = False
        cfg.safety_shield_solver = "CLARABEL"
    if bool(args.disable_danger_imitation):
        cfg.danger_imitation_enabled = False
        cfg.danger_imitation_coef = 0.0
    if args.bw_fixed_tau is not None:
        if not (float(args.bw_fixed_tau) > 0.0):
            raise ValueError("--bw_fixed_tau must be positive when set.")
        cfg.bw_fixed_tau = float(args.bw_fixed_tau)
    if args.bw_fixed_kappa is not None:
        if not (float(args.bw_fixed_kappa) > 0.0):
            raise ValueError("--bw_fixed_kappa must be positive when set.")
        cfg.bw_fixed_kappa = float(args.bw_fixed_kappa)
    if args.access_bw_decision_interval is not None:
        if int(args.access_bw_decision_interval) < 1:
            raise ValueError("--access_bw_decision_interval must be >= 1.")
        cfg.access_bw_decision_interval = int(args.access_bw_decision_interval)
    if args.sat_decision_interval is not None:
        if int(args.sat_decision_interval) < 1:
            raise ValueError("--sat_decision_interval must be >= 1.")
        cfg.sat_decision_interval = int(args.sat_decision_interval)
    report_torch_compile_cache(context="train_joint_mcgae", device=device, cfg=cfg)
    if device.type == "cuda":
        _enable_strict_compile_global()
    hparams = _resolve_training_hparams(cfg, args)
    return_target = _canonical_return_target(str(args.return_target))
    return_target_label = _return_target_label(return_target)
    return_target_schedule = str(args.return_target_schedule).strip().lower()
    if int(args.target_switch_update) < 0:
        raise ValueError("--target_switch_update must be >= 0.")
    if int(args.target_mix_warmup_updates) < 0:
        raise ValueError("--target_mix_warmup_updates must be >= 0.")
    if int(args.target_mix_anneal_updates) < 0:
        raise ValueError("--target_mix_anneal_updates must be >= 0.")
    if int(args.return_nstep_horizon) <= 0:
        raise ValueError("--return_nstep_horizon must be > 0.")
    if float(args.mc_aux_critic_coef) < 0.0:
        raise ValueError("--mc_aux_critic_coef must be >= 0.")
    target_metric_by_stage = {
        int(stage_id): f"{STAGE_NAME[int(stage_id)]}_target_return_mean"
        for stage_id in STAGES
    }

    if str(getattr(cfg, "critic_value_mode", "")).lower() == "global_linear":
        raise RuntimeError("joint MC-GAE needs a trainable relational critic; critic_value_mode=global_linear is unsupported.")
    if bool(getattr(cfg, "bw_flow_proxy_aux_enabled", False)):
        raise RuntimeError(
            "train_joint_mcgae.py currently implements PPO+entropy for BW, but not bw_flow_proxy_aux_loss. "
            "Disable bw_flow_proxy_aux_enabled or add that auxiliary term explicitly."
        )

    learner = _make_joint_learner(cfg, device=device)
    critic_params = [param for param in learner.critic.parameters() if param.requires_grad]
    if not critic_params:
        raise RuntimeError("joint MC-GAE training requires trainable critic parameters.")
    critic_optimizer = torch.optim.Adam(critic_params, lr=float(hparams["cold_critic_lr"]))

    actor_optimizers: dict[int, torch.optim.Optimizer] = {}
    actor_optimizer_names: dict[int, str] = {}
    for stage_id in STAGES:
        params = _stage_optimizer_params(learner.actor, int(stage_id))
        if not params:
            raise RuntimeError(f"{STAGE_NAME[int(stage_id)]} actor has no trainable parameters.")
        stage_lr = float(hparams[f"actor_lr_{_stage_metric_suffix(int(stage_id))}"])
        optimizer_name = _stage_actor_optimizer_name(args, int(stage_id))
        actor_optimizer_names[int(stage_id)] = optimizer_name
        actor_optimizers[int(stage_id)] = _make_stage_actor_optimizer(
            params,
            lr=stage_lr,
            optimizer_name=optimizer_name,
        )

    resume_state: dict[str, Any] | None = None
    start_update = 0
    if args.resume is not None and str(args.resume).strip():
        resume_path = Path(str(args.resume))
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        resume_state = torch.load(resume_path, map_location=device, weights_only=False)
        learner.actor.load_state_dict(resume_state["actor"])
        learner.critic.load_state_dict(resume_state["critic"])
        if "critic_optimizer" in resume_state:
            critic_optimizer.load_state_dict(resume_state["critic_optimizer"])
        if "actor_optimizers" in resume_state:
            opt_state = resume_state["actor_optimizers"]
            for stage_id, optimizer in actor_optimizers.items():
                key_options = (stage_id, str(stage_id))
                for key in key_options:
                    if key in opt_state:
                        optimizer.load_state_dict(opt_state[key])
                        break
        start_update = max(int(resume_state.get("update", 0) or 0), 0)
        saved_hparams = resume_state.get("hparams")
        if isinstance(saved_hparams, dict):
            for stage_id in STAGES:
                suffix = _stage_metric_suffix(int(stage_id))
                for key in (f"actor_lr_high_count_{suffix}", f"actor_lr_low_count_{suffix}"):
                    if key in saved_hparams:
                        hparams[key] = int(saved_hparams.get(key, 0) or 0)
        for group_opt in critic_optimizer.param_groups:
            group_opt["lr"] = float(hparams["cold_critic_lr"] if start_update <= 0 else hparams["tracking_critic_lr"])
        # Keep checkpoint actor optimizer moments and active lrs so dynamic
        # per-stage decay resumes exactly where the run stopped.  Fresh runs
        # still use the config/CLI stage-specific lrs set above.

    group = make_structured_driver_group(cfg, num_envs=int(args.num_envs), backend="sync", mode="train")
    metrics: list[dict[str, float]] = _load_existing_metrics(run_dir / "metrics.csv", max_update=start_update)
    stage_best_dir = Path(args.stage_best_dir) if args.stage_best_dir else run_dir / "best_stage_heads"
    stage_best_enabled = not bool(args.disable_stage_best_save)
    if stage_best_enabled:
        stage_best_dir.mkdir(parents=True, exist_ok=True)
    stage_best_values: dict[int, float] = {int(stage_id): float("-inf") for stage_id in STAGES}
    stage_best_updates: dict[int, int] = {int(stage_id): 0 for stage_id in STAGES}
    for prev_row in metrics:
        update_prev = int(float(prev_row.get("update", 0.0) or 0.0))
        for stage_id in STAGES:
            stage_name = STAGE_NAME[int(stage_id)]
            metric_name = target_metric_by_stage[int(stage_id)]
            value = float(prev_row.get(metric_name, float("nan")))
            if math.isfinite(value) and value > stage_best_values[int(stage_id)]:
                stage_best_values[int(stage_id)] = float(value)
                stage_best_updates[int(stage_id)] = int(update_prev)
    completed_updates = int(start_update)
    try:
        group_is_native = all(
            hasattr(group, attr)
            for attr in ("native_rollout_runtime", "begin_native_main_kernel_rollout", "native_rollout_program")
        )
        if group_is_native:
            learner.bind_native_runtime_contract(group)
            sync_native = getattr(learner, "_sync_native_actor_cuda_bindings_after_update", None)
            if callable(sync_native):
                sync_native()
        if resume_state is not None:
            _restore_rng_state_payload(resume_state.get("rng_state"), device)
            print(
                f"Resumed joint MC-GAE from {args.resume} at update {start_update}; "
                f"target updates={int(args.updates)}.",
                flush=True,
            )
        print(
            "JOINT MC-GAE train | "
            f"envs={int(args.num_envs)} rollout={int(args.rollout_env_steps)} updates={int(args.updates)} "
            f"reward={cfg.reward_mode} critic={int(hparams['cold_critic_epochs'])}/{int(hparams['tracking_critic_epochs'])} "
            f"actor_epochs={int(hparams['actor_epochs'])} actor_minibatches={int(hparams['actor_minibatches'])} "
            f"actor_lr=({float(hparams['actor_lr_accel']):.2g},{float(hparams['actor_lr_sat']):.2g},{float(hparams['actor_lr_bw']):.2g}) "
            f"actor_opt=({actor_optimizer_names[0]},{actor_optimizer_names[1]},{actor_optimizer_names[2]}) "
            f"return_target={return_target_label} "
            f"return_target_schedule={return_target_schedule} "
            f"target_switch_update={int(args.target_switch_update)} "
            f"target_mix=({float(args.target_mix_alpha_start):.3g}->{float(args.target_mix_alpha_end):.3g},"
            f"warmup={int(args.target_mix_warmup_updates)},anneal={int(args.target_mix_anneal_updates)}) "
            f"mc_aux_coef={float(args.mc_aux_critic_coef):.3g} "
            f"nstep={int(args.return_nstep_horizon)} "
            f"kl_stop={bool(int(hparams['stage_actor_kl_early_stop_enabled']))} "
            f"dyn_lr={bool(int(hparams['stage_actor_dynamic_lr_enabled']))} "
            f"critic_ev_gate={bool(int(hparams['critic_ev_gate_enabled']))} "
            f"exec=({cfg.exec_accel_source},{cfg.exec_sat_source},{cfg.exec_bw_source}) "
            f"safety=(avoidance={bool(getattr(cfg, 'avoidance_enabled', False))}, "
            f"native_shield={bool(getattr(cfg, 'safety_shield_enabled', False)) and str(getattr(cfg, 'safety_shield_solver', '') or '').strip().upper() == 'NATIVE_CUDA'}, "
            f"danger={bool(getattr(learner, 'danger_imitation_enabled', False))}, "
            f"danger_mode={getattr(cfg, 'danger_imitation_trigger_mode', '')})",
            flush=True,
        )

        for update in range(start_update, max(int(args.updates), 0)):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            update_t0 = time.perf_counter()
            rollout_seed = int(args.seed) + update * 100_000
            _append_phase_trace(run_dir, update=update + 1, phase="update", event="start", seed=rollout_seed)
            active_return_target = _return_target_for_update(
                base_return_target=return_target,
                schedule=return_target_schedule,
                update=int(update),
                switch_update=int(args.target_switch_update),
            )
            active_return_target_label = _return_target_label(active_return_target)
            target_mix_alpha = _target_mix_alpha_for_update(
                update=int(update),
                alpha_start=float(args.target_mix_alpha_start),
                alpha_end=float(args.target_mix_alpha_end),
                warmup_updates=int(args.target_mix_warmup_updates),
                anneal_updates=int(args.target_mix_anneal_updates),
            )

            t_collect = time.perf_counter()
            _append_phase_trace(run_dir, update=update + 1, phase="collect", event="start")
            views, _all_returns, stage_targets, stage_advantages, stage_aux_targets, reward_stats = _collect_joint_rollout(
                learner,
                group,
                rollout_env_steps=int(args.rollout_env_steps),
                device=device,
                seed=rollout_seed,
                return_target=active_return_target,
                target_mix_alpha=float(target_mix_alpha),
                return_nstep_horizon=int(args.return_nstep_horizon),
            )
            rollout_diag_stats = _rollout_diagnostics_from_views(views)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            collect_sec = time.perf_counter() - t_collect
            _append_phase_trace(run_dir, update=update + 1, phase="collect", event="end", sec=float(collect_sec))

            critic_lr = float(hparams["cold_critic_lr"] if update == 0 else hparams["tracking_critic_lr"])
            critic_epochs = int(hparams["cold_critic_epochs"] if update == 0 else hparams["tracking_critic_epochs"])

            t_critic = time.perf_counter()
            _append_phase_trace(
                run_dir,
                update=update + 1,
                phase="critic",
                event="start",
                lr=float(critic_lr),
                epochs=int(critic_epochs),
            )
            critic_metrics: dict[str, float] = {}
            stage_values_after_critic: dict[int, torch.Tensor] = {}
            actor_skip_by_stage: dict[int, bool] = {}
            for stage_id in STAGES:
                stage_batch = views.training_view.stage_batches[int(stage_id)]
                stage_name = STAGE_NAME[int(stage_id)]
                _append_phase_trace(
                    run_dir,
                    update=update + 1,
                    phase=f"critic_{stage_name}",
                    event="start",
                    samples=int(stage_batch.num_samples),
                )
                stats, after_values, trace_rows = _train_stage_critic_on_stage(
                    learner,
                    stage_id=int(stage_id),
                    stage_batch=stage_batch,
                    target=stage_targets[int(stage_id)],
                    aux_target=stage_aux_targets.get(int(stage_id)),
                    aux_coef=float(args.mc_aux_critic_coef) if active_return_target == "bootstrap_mc_aux" else 0.0,
                    optimizer=critic_optimizer,
                    lr=critic_lr,
                    epochs=critic_epochs,
                    minibatches=int(hparams["critic_minibatches"]),
                    update_microbatch_size=int(hparams["critic_update_microbatch_size"]),
                    diagnose_timing=bool(args.diagnose_critic_timing),
                )
                initial_ev_after = float(stats.get("critic_ev_after", float("nan")))
                initial_stats = dict(stats)
                cumulative_stat_keys = (
                    "critic_clone_sec",
                    "critic_eval_before_sec",
                    "critic_train_loop_sec",
                    "critic_cache_clear_sec",
                    "critic_eval_after_sec",
                )
                cumulative_stats = {
                    key: float(initial_stats.get(key, 0.0) or 0.0)
                    for key in cumulative_stat_keys
                }
                retry_count = 0
                extra_epochs_total = 0
                gate_enabled = bool(int(hparams.get("critic_ev_gate_enabled", 0)))
                soft_target = float(hparams.get("critic_ev_soft_target", 0.0))
                hard_floor = float(hparams.get("critic_ev_hard_floor", float("-inf")))
                extra_epochs = max(int(hparams.get("critic_ev_extra_epochs", 0) or 0), 0)
                max_retries = max(int(hparams.get("critic_ev_max_retries", 0) or 0), 0)
                if gate_enabled and extra_epochs > 0 and max_retries > 0:
                    while retry_count < max_retries:
                        ev_after = float(stats.get("critic_ev_after", float("nan")))
                        if _ev_is_finite(ev_after) and ev_after >= soft_target:
                            break
                        retry_count += 1
                        extra_epochs_total += int(extra_epochs)
                        _append_phase_trace(
                            run_dir,
                            update=update + 1,
                            phase=f"critic_{stage_name}",
                            event="retry_start",
                            retry=int(retry_count),
                            ev_after=float(ev_after),
                            extra_epochs=int(extra_epochs),
                        )
                        retry_stats, retry_values, retry_trace_rows = _train_stage_critic_on_stage(
                            learner,
                            stage_id=int(stage_id),
                            stage_batch=stage_batch,
                            target=stage_targets[int(stage_id)],
                            aux_target=stage_aux_targets.get(int(stage_id)),
                            aux_coef=(
                                float(args.mc_aux_critic_coef)
                                if active_return_target == "bootstrap_mc_aux"
                                else 0.0
                            ),
                            optimizer=critic_optimizer,
                            lr=critic_lr,
                            epochs=int(extra_epochs),
                            minibatches=int(hparams["critic_minibatches"]),
                            update_microbatch_size=int(hparams["critic_update_microbatch_size"]),
                            eval_before_enabled=False,
                            diagnose_timing=bool(args.diagnose_critic_timing),
                        )
                        stats = retry_stats
                        after_values = retry_values
                        for key in cumulative_stat_keys:
                            cumulative_stats[key] = float(cumulative_stats.get(key, 0.0)) + float(
                                retry_stats.get(key, 0.0) or 0.0
                            )
                        if trace_rows is not None:
                            trace_rows.extend(retry_trace_rows)
                        _append_phase_trace(
                            run_dir,
                            update=update + 1,
                            phase=f"critic_{stage_name}",
                            event="retry_end",
                            retry=int(retry_count),
                            ev_after=float(stats.get("critic_ev_after", float("nan"))),
                        )
                stage_values_after_critic[int(stage_id)] = after_values.detach()
                final_ev_after = float(stats.get("critic_ev_after", float("nan")))
                below_hard = bool(
                    gate_enabled
                    and (
                        (not _ev_is_finite(final_ev_after))
                        or final_ev_after < float(hard_floor)
                    )
                )
                actor_skip_by_stage[int(stage_id)] = bool(below_hard)
                stats["critic_ev_gate_enabled"] = float(1.0 if gate_enabled else 0.0)
                stats["critic_ev_gate_soft_target"] = float(soft_target)
                stats["critic_ev_gate_hard_floor"] = float(hard_floor)
                stats["critic_ev_initial_after"] = float(initial_ev_after)
                stats["critic_ev_final_after"] = float(final_ev_after)
                stats["critic_ev_retry_count"] = float(retry_count)
                stats["critic_ev_extra_epochs_total"] = float(extra_epochs_total)
                stats["critic_epochs"] = float(int(critic_epochs) + int(extra_epochs_total))
                for key, value in cumulative_stats.items():
                    stats[key] = float(value)
                stats["critic_low_confidence_actor_update"] = float(
                    1.0 if gate_enabled and (not below_hard) and final_ev_after < soft_target else 0.0
                )
                stats["critic_skip_actor_update"] = float(1.0 if below_hard else 0.0)
                critic_metrics.update(_prefix_keys(stats, f"{stage_name}_critic"))
                _append_phase_trace(
                    run_dir,
                    update=update + 1,
                    phase=f"critic_{stage_name}",
                    event="end",
                    ev_after=float(stats.get("critic_ev_after", float("nan"))),
                    retries=int(retry_count),
                    skip_actor=bool(below_hard),
                )
                if trace_rows:
                    trace_path = run_dir / f"critic_trace_update{update + 1:04d}_{stage_name}.jsonl"
                    with trace_path.open("w", encoding="utf-8") as f:
                        for row in trace_rows:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            critic_sec = time.perf_counter() - t_critic
            _append_phase_trace(run_dir, update=update + 1, phase="critic", event="end", sec=float(critic_sec))

            t_actor = time.perf_counter()
            _append_phase_trace(run_dir, update=update + 1, phase="actor", event="start")
            actor_metrics: dict[str, float] = {}
            adv_metrics: dict[str, float] = {}
            stage_actor_inputs: dict[int, torch.Tensor] = {}
            actor_stage_batches: dict[int, _ActorOnlyStageBatch] = {}
            for stage_id in STAGES:
                stage_batch = views.training_view.stage_batches[int(stage_id)]
                stage_name = STAGE_NAME[int(stage_id)]
                if active_return_target == "mc":
                    _returns_gae, stage_adv, stage_values = _stage_gae_from_mc_targets(
                        learner,
                        stage_id=int(stage_id),
                        stage_batch=stage_batch,
                        mc_target=stage_targets[int(stage_id)],
                        device=device,
                        stage_values=stage_values_after_critic.get(int(stage_id)),
                    )
                elif _target_advantage_after_critic(active_return_target):
                    _returns_gae = stage_targets[int(stage_id)].detach().to(device=device, dtype=torch.float32)
                    stage_values = stage_values_after_critic[int(stage_id)].detach().to(
                        device=device,
                        dtype=torch.float32,
                    )
                    stage_adv = _returns_gae - stage_values
                else:
                    _returns_gae = stage_targets[int(stage_id)].detach().to(device=device, dtype=torch.float32)
                    stage_adv = stage_advantages[int(stage_id)].detach().to(device=device, dtype=torch.float32)
                    stage_values = stage_values_after_critic[int(stage_id)].detach().to(
                        device=device,
                        dtype=torch.float32,
                    )
                stage_adv_norm = _normalize_stage_advantage(
                    stage_adv,
                    enabled=bool(getattr(cfg, "actor_advantage_normalize_enabled", True)),
                )
                adv_raw = _summ_tensor(stage_adv)
                adv_norm = _summ_tensor(stage_adv_norm)
                value_stats = _summ_tensor(stage_values)
                adv_metrics.update({f"{stage_name}_raw_adv_{k}": v for k, v in adv_raw.items()})
                adv_metrics.update({f"{stage_name}_norm_adv_{k}": v for k, v in adv_norm.items()})
                adv_metrics.update({f"{stage_name}_value_{k}": v for k, v in value_stats.items()})
                adv_metrics[f"{stage_name}_critic_final_ev"] = _tensor_ev(stage_values, stage_targets[int(stage_id)])
                stage_actor_inputs[int(stage_id)] = stage_adv_norm
                actor_stage_batches[int(stage_id)] = _actor_only_stage_batch(stage_batch)

            t_prune = time.perf_counter()
            del views
            del stage_targets
            del stage_advantages
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
            adv_metrics["pre_actor_prune_rollout_view_sec"] = float(time.perf_counter() - t_prune)

            for stage_id in STAGES:
                stage_batch = actor_stage_batches[int(stage_id)]
                stage_name = STAGE_NAME[int(stage_id)]
                stage_adv_norm = stage_actor_inputs[int(stage_id)]
                _append_phase_trace(
                    run_dir,
                    update=update + 1,
                    phase=f"actor_{stage_name}",
                    event="start",
                    samples=int(stage_batch.num_samples),
                )
                if int(stage_id) == 2 and device.type == "cuda":
                    # BW actor backward allocates substantially larger
                    # temporaries than accel/SAT.  After critic/value evals
                    # the CUDA caching allocator can be left in a fragmented
                    # state; returning unused blocks before BW avoids the
                    # observed 2x-3x slowdown without changing tensors that
                    # are still live.
                    t_bw_cache = time.perf_counter()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(device)
                    adv_metrics["bw_pre_actor_empty_cache_sec"] = float(time.perf_counter() - t_bw_cache)
                if bool(actor_skip_by_stage.get(int(stage_id), False)):
                    stats = _skipped_stage_actor_stats(
                        stage_id=int(stage_id),
                        stage_batch=stage_batch,
                        stage_advantages=stage_adv_norm,
                        hparams=hparams,
                        actor_optimizers=actor_optimizers,
                    )
                else:
                    stats = _stage_actor_update_full_stage(
                        learner,
                        stage_id=int(stage_id),
                        stage_batch=stage_batch,
                        stage_advantages=stage_adv_norm,
                        optimizer=actor_optimizers[int(stage_id)],
                        epochs=int(hparams["actor_epochs"]),
                        minibatches=int(hparams["actor_minibatches"]),
                        parity_dump_dir=run_dir / "diagnostics" / "bw_parity",
                        parity_dump_tag=f"u{update + 1:04d}_{stage_name}",
                        parity_topk=int(args.bw_parity_topk),
                        kl_stop_threshold=(
                            float(hparams[f"actor_kl_stop_threshold_{stage_name}"])
                            if bool(int(hparams.get("stage_actor_kl_early_stop_enabled", 0)))
                            else None
                        ),
                    )
                    stats.update(
                        _maybe_adjust_stage_actor_lr(
                            actor_optimizers[int(stage_id)],
                            stage_id=int(stage_id),
                            stage_metrics=stats,
                            hparams=hparams,
                        )
                    )
                actor_metrics.update(_prefix_keys(stats, f"{stage_name}_actor", keep_prefixed_stage_keys=True))
                _append_phase_trace(
                    run_dir,
                    update=update + 1,
                    phase=f"actor_{stage_name}",
                    event="end",
                    kl=float(stats.get(f"approx_kl_{stage_name}", stats.get("approx_kl", float("nan")))),
                    samples=float(stats.get("actor_samples", float("nan"))),
                )
            sync_native = getattr(learner, "_sync_native_actor_cuda_bindings_after_update", None)
            _append_phase_trace(run_dir, update=update + 1, phase="native_sync", event="start")
            if callable(sync_native):
                sync_native()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            _append_phase_trace(run_dir, update=update + 1, phase="native_sync", event="end")
            actor_sec = time.perf_counter() - t_actor
            _append_phase_trace(run_dir, update=update + 1, phase="actor", event="end", sec=float(actor_sec))

            empty_cache_sec = 0.0
            if device.type == "cuda":
                t_cache = time.perf_counter()
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
                empty_cache_sec = time.perf_counter() - t_cache

            row: dict[str, float] = {
                "update": float(update + 1),
                "collect_sec": float(collect_sec),
                "critic_sec": float(critic_sec),
                "actor_sec": float(actor_sec),
                "empty_cache_sec": float(empty_cache_sec),
                "iteration_sec": float(time.perf_counter() - update_t0),
                "critic_lr": float(critic_lr),
                "critic_epochs": float(critic_epochs),
                "critic_minibatches": float(hparams["critic_minibatches"]),
                "critic_ev_gate_enabled": float(hparams["critic_ev_gate_enabled"]),
                "critic_ev_soft_target": float(hparams["critic_ev_soft_target"]),
                "critic_ev_hard_floor": float(hparams["critic_ev_hard_floor"]),
                "critic_ev_extra_epochs": float(hparams["critic_ev_extra_epochs"]),
                "critic_ev_max_retries": float(hparams["critic_ev_max_retries"]),
                "base_return_target_code": _return_target_code(return_target),
                "active_return_target_code": _return_target_code(active_return_target),
                "return_target_schedule_code": 0.0 if return_target_schedule == "fixed" else 1.0,
                "target_switch_update": float(args.target_switch_update),
                "target_mix_alpha": float(target_mix_alpha),
                "target_mix_alpha_start": float(args.target_mix_alpha_start),
                "target_mix_alpha_end": float(args.target_mix_alpha_end),
                "target_mix_warmup_updates": float(args.target_mix_warmup_updates),
                "target_mix_anneal_updates": float(args.target_mix_anneal_updates),
                "mc_aux_critic_coef": (
                    float(args.mc_aux_critic_coef) if active_return_target == "bootstrap_mc_aux" else 0.0
                ),
                "return_nstep_horizon": float(args.return_nstep_horizon),
                "actor_lr": float(hparams["actor_lr"]),
                "actor_lr_config_accel": float(hparams["actor_lr_accel"]),
                "actor_lr_config_sat": float(hparams["actor_lr_sat"]),
                "actor_lr_config_bw": float(hparams["actor_lr_bw"]),
                "actor_optimizer_accel_code": float(0.0 if actor_optimizer_names[0] == "adam" else 1.0),
                "actor_optimizer_sat_code": float(0.0 if actor_optimizer_names[1] == "adam" else 1.0),
                "actor_optimizer_bw_code": float(0.0 if actor_optimizer_names[2] == "adam" else 1.0),
                "actor_epochs": float(hparams["actor_epochs"]),
                "actor_minibatches": float(hparams["actor_minibatches"]),
                "stage_actor_kl_early_stop_enabled": float(hparams["stage_actor_kl_early_stop_enabled"]),
                "stage_actor_dynamic_lr_enabled": float(hparams["stage_actor_dynamic_lr_enabled"]),
                **reward_stats,
                **rollout_diag_stats,
                **critic_metrics,
                **adv_metrics,
                **actor_metrics,
                **_cuda_mem("update_end", device),
            }
            stage_best_metrics: dict[str, float] = {}
            if stage_best_enabled:
                summary_payload: dict[str, Any] = {}
                for stage_id in STAGES:
                    stage_name = STAGE_NAME[int(stage_id)]
                    metric_name = target_metric_by_stage[int(stage_id)]
                    value = float(row.get(metric_name, float("nan")))
                    improved = bool(math.isfinite(value) and value > stage_best_values[int(stage_id)])
                    if improved:
                        stage_best_values[int(stage_id)] = float(value)
                        stage_best_updates[int(stage_id)] = int(update + 1)
                        _save_stage_best_checkpoint(
                            stage_best_dir / f"best_{stage_name}.pt",
                            learner=learner,
                            cfg=cfg,
                            args=args,
                            hparams=hparams,
                            update=update + 1,
                            stage_id=int(stage_id),
                            metric_name=metric_name,
                            metric_value=value,
                        )
                    stage_best_metrics[f"stage_best_metric_{stage_name}"] = float(stage_best_values[int(stage_id)])
                    stage_best_metrics[f"stage_best_update_{stage_name}"] = float(stage_best_updates[int(stage_id)])
                    stage_best_metrics[f"stage_best_improved_{stage_name}"] = 1.0 if improved else 0.0
                    summary_payload[stage_name] = {
                        "metric_name": metric_name,
                        "metric_value": float(stage_best_values[int(stage_id)]),
                        "update": int(stage_best_updates[int(stage_id)]),
                        "path": str(stage_best_dir / f"best_{stage_name}.pt"),
                    }
                row.update(stage_best_metrics)
                with (stage_best_dir / "summary.json").open("w", encoding="utf-8") as f:
                    json.dump(summary_payload, f, ensure_ascii=False, indent=2)
            metrics.append(row)
            _write_metrics(run_dir / "metrics.csv", metrics)
            _append_phase_trace(
                run_dir,
                update=update + 1,
                phase="update",
                event="end",
                sec=float(row["iteration_sec"]),
            )
            if int(args.save_every) > 0 and (update + 1) % int(args.save_every) == 0:
                _save_joint_checkpoint(
                    run_dir / f"checkpoint_update{update + 1:04d}.pt",
                    learner=learner,
                    critic_optimizer=critic_optimizer,
                    actor_optimizers=actor_optimizers,
                    cfg=cfg,
                    args=args,
                    hparams=hparams,
                    update=update + 1,
                    completed=True,
                    device=device,
                )
            completed_updates = update + 1

            print(
                f"Update {update + 1}/{int(args.updates)} "
                f"{active_return_target_label}[a={row[target_metric_by_stage[0]]:.3f},"
                f"s={row[target_metric_by_stage[1]]:.3f},b={row[target_metric_by_stage[2]]:.3f}] "
                f"ev[a={row['accel_critic_final_ev']:.3f},s={row['sat_critic_final_ev']:.3f},b={row['bw_critic_final_ev']:.3f}] "
                f"kl[a={row.get('approx_kl_accel', 0.0):.4f},s={row.get('approx_kl_sat', 0.0):.4f},b={row.get('approx_kl_bw', 0.0):.4f}] "
                f"time[c={collect_sec:.1f},v={critic_sec:.1f},p={actor_sec:.1f},tot={row['iteration_sec']:.1f}]",
                flush=True,
            )
        _save_joint_checkpoint(
            run_dir / "final.pt",
            learner=learner,
            critic_optimizer=critic_optimizer,
            actor_optimizers=actor_optimizers,
            cfg=cfg,
            args=args,
            hparams=hparams,
            update=completed_updates,
            completed=True,
            device=device,
        )
    except BaseException:
        _save_joint_checkpoint(
            run_dir / "checkpoint_crash.pt",
            learner=learner,
            critic_optimizer=critic_optimizer,
            actor_optimizers=actor_optimizers,
            cfg=cfg,
            args=args,
            hparams=hparams,
            update=completed_updates,
            completed=False,
            device=device,
        )
        raise
    finally:
        close_structured_env_group(group)


if __name__ == "__main__":
    main()
