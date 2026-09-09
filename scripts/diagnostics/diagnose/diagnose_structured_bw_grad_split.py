from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Sequence

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
from sagin_marl.rl.structured_mappo import (
    StructuredMAPPO,
    _bw_valid_and_effective_loc,
    _collate_dataclass,
    _grad_l2_norm,
    _index_dataclass,
    _masked_simplex_probs,
)
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


def _tensor_l2_norm(grad: torch.Tensor | None, *, device: torch.device) -> torch.Tensor:
    if grad is None:
        return torch.zeros((), dtype=torch.float32, device=device)
    return grad.detach().to(dtype=torch.float32).pow(2).sum().sqrt()


def _tensor_cosine(a: torch.Tensor | None, b: torch.Tensor | None, *, device: torch.device) -> torch.Tensor:
    if a is None or b is None:
        return torch.zeros((), dtype=torch.float32, device=device)
    a_flat = a.detach().to(dtype=torch.float32).reshape(-1)
    b_flat = b.detach().to(dtype=torch.float32).reshape(-1)
    a_norm = a_flat.norm()
    b_norm = b_flat.norm()
    if float(a_norm.item()) <= 1.0e-12 or float(b_norm.item()) <= 1.0e-12:
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.dot(a_flat, b_flat) / (a_norm * b_norm)


def _top1_margin(probs: torch.Tensor) -> torch.Tensor:
    if probs.shape[-1] <= 1:
        return probs.amax(dim=-1)
    top2 = probs.topk(k=2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


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
    actor.train()
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


def _build_bw_cache(algo: StructuredMAPPO, buffer: StructuredRolloutBuffer) -> dict[str, Any] | None:
    rollout_views = buffer.build_rollout_views(algo.device)
    bw_stage_batch = rollout_views.training_view.stage_batches.get(2)
    if bw_stage_batch is None or int(bw_stage_batch.num_samples) <= 0:
        return None
    gae = algo.compute_returns_and_advantages(
        buffer,
        rollout_views.bootstrap_view,
        return_view=rollout_views.return_view,
    )
    returns = torch.from_numpy(gae["returns"]).to(algo.device)
    advantages = torch.from_numpy(gae["advantages"]).to(algo.device)
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1.0e-8)
    stage_idx = torch.as_tensor(
        np.asarray(bw_stage_batch.transition_indices, dtype=np.int64),
        device=algo.device,
        dtype=torch.long,
    )
    joint_actions = bw_stage_batch.actions.to(algo.device)
    num_samples = int(bw_stage_batch.num_samples)
    num_agents = int(bw_stage_batch.num_agents)
    flat_local_batch = bw_stage_batch.local_batch
    flat_indices = torch.arange(num_samples * num_agents, device=algo.device, dtype=torch.long).reshape(
        num_samples, num_agents
    )
    return {
        "num_samples": int(num_samples),
        "num_agents": int(num_agents),
        "local_batch": flat_local_batch,
        "joint_actions": joint_actions,
        "flat_indices": flat_indices,
        "old_logprobs": bw_stage_batch.old_logprobs.to(algo.device),
        "advantages": advantages.index_select(0, stage_idx),
        "bw_flow_proxy_scores": torch.stack(
            [
                score
                if score is not None
                else torch.zeros_like(action, dtype=torch.float32)
                for score, action in zip(
                    ([] if bw_stage_batch.bw_flow_proxy_scores is None else list(bw_stage_batch.bw_flow_proxy_scores)),
                    list(bw_stage_batch.actions),
                )
            ],
            dim=0,
        ).to(algo.device),
        "bw_flow_proxy_masks": torch.stack(
            [
                mask
                if mask is not None
                else torch.zeros_like(action, dtype=torch.float32)
                for mask, action in zip(
                    ([] if bw_stage_batch.bw_flow_proxy_masks is None else list(bw_stage_batch.bw_flow_proxy_masks)),
                    list(bw_stage_batch.actions),
                )
            ],
            dim=0,
        ).to(algo.device),
        "returns": returns.index_select(0, stage_idx),
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
        "bw_grad_norm_policy",
        "bw_grad_norm_aux_scaled",
        "bw_grad_ratio_aux_to_policy",
        "bw_loc_head_grad_norm_policy",
        "bw_loc_head_grad_norm_aux_scaled",
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
    cache = _build_bw_cache(learner, buffer)
    if cache is None:
        return {
            "update": int(update),
            "rollout_env_steps": int(rollout_steps),
            "num_envs": int(num_envs),
            "vec_backend": str(vec_backend),
            "bw_stage_samples": 0,
            "logged_metrics": _history_tail(meta),
        }

    bw_policy = getattr(learner.actor, "bw_policy", None)
    if bw_policy is None:
        raise RuntimeError("Structured actor is missing bw_policy")
    bw_params = [param for param in bw_policy.parameters() if param.requires_grad]
    loc_head = getattr(bw_policy, "loc_head", None)
    log_scale_head = getattr(bw_policy, "log_scale_head", None)
    loc_head_params = [] if loc_head is None else [param for param in loc_head.parameters() if param.requires_grad]
    log_scale_head_params = (
        [] if log_scale_head is None else [param for param in log_scale_head.parameters() if param.requires_grad]
    )

    per_mb: dict[str, list[float]] = {
        "policy_loss": [],
        "entropy_mean": [],
        "clip_frac": [],
        "pairwise_acc": [],
        "pair_count": [],
        "aux_loss_unscaled": [],
        "aux_loss_scaled": [],
        "regression_loss": [],
        "policy_loc_grad_norm": [],
        "policy_log_scale_grad_norm": [],
        "policy_total_loc_grad_norm": [],
        "policy_total_log_scale_grad_norm": [],
        "aux_loc_grad_norm_scaled": [],
        "aux_log_scale_grad_norm_scaled": [],
        "loc_grad_ratio_aux_to_policy": [],
        "loc_grad_ratio_aux_to_policy_total": [],
        "log_scale_to_loc_policy_ratio": [],
        "log_scale_to_loc_policy_total_ratio": [],
        "loc_grad_cos_aux_vs_policy": [],
        "loc_grad_cos_aux_vs_policy_total": [],
        "bw_param_grad_norm_policy": [],
        "bw_param_grad_norm_aux_scaled": [],
        "bw_param_grad_ratio_aux_to_policy": [],
        "loc_head_grad_norm_policy": [],
        "loc_head_grad_norm_aux_scaled": [],
        "loc_head_grad_ratio_aux_to_policy": [],
        "log_scale_head_grad_norm_policy": [],
        "log_scale_head_grad_norm_aux_scaled": [],
        "policy_mode_top1": [],
        "policy_mode_top1_margin": [],
    }
    epoch_rows: list[dict[str, float]] = []
    actor_update_steps = 0

    for epoch_idx in range(learner.ppo_epochs):
        rel_idx = np.arange(int(cache["num_samples"]), dtype=np.int64)
        np.random.shuffle(rel_idx)
        minibatch_size = max(1, int(math.ceil(rel_idx.size / max(learner.num_mini_batch, 1))))
        minibatches = [
            torch.as_tensor(rel_idx[start : start + minibatch_size], device=learner.device, dtype=torch.long)
            for start in range(0, rel_idx.size, minibatch_size)
        ]
        epoch_metrics: dict[str, list[float]] = {
            "policy_loc_grad_norm": [],
            "policy_log_scale_grad_norm": [],
            "aux_loc_grad_norm_scaled": [],
            "loc_grad_ratio_aux_to_policy": [],
            "loc_grad_cos_aux_vs_policy": [],
            "policy_mode_top1_margin": [],
            "pairwise_acc": [],
        }

        for mb_rel in minibatches:
            actor_update_steps += 1
            flat_mb = cache["flat_indices"].index_select(0, mb_rel).reshape(-1)
            local_batch_mb = _index_dataclass(cache["local_batch"], flat_mb)
            joint_actions_mb = cache["joint_actions"].index_select(0, mb_rel)
            num_samples = int(joint_actions_mb.shape[0])
            num_agents = int(cache["num_agents"])

            new_logprob, joint_entropy, actor_out = learner._stage_actor_eval_from_batch(
                2,
                local_batch_mb,
                joint_actions_mb,
                num_agents,
            )
            log_ratio = new_logprob - cache["old_logprobs"].index_select(0, mb_rel)
            ratio = torch.exp(log_ratio)
            clip_lower, clip_upper = learner._stage_ratio_clip_bounds(
                2,
                actor_out,
                num_samples=num_samples,
                num_agents=num_agents,
                dtype=ratio.dtype,
                device=ratio.device,
            )
            adv = cache["advantages"].index_select(0, mb_rel)
            surr1 = ratio * adv
            surr2 = learner._clip_ratio_with_bounds(ratio, clip_lower, clip_upper) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_mean = joint_entropy.mean()
            policy_total_loss = policy_loss - learner.entropy_coef * entropy_mean

            proxy_scores_mb = cache["bw_flow_proxy_scores"].index_select(0, mb_rel).reshape(
                -1,
                joint_actions_mb.shape[-1],
            )
            proxy_masks_mb = cache["bw_flow_proxy_masks"].index_select(0, mb_rel).reshape(
                -1,
                joint_actions_mb.shape[-1],
            )
            aux_loss, pair_acc, pair_count, regression_loss = learner._bw_flow_proxy_aux_loss(
                local_batch_mb,
                actor_out,
                proxy_scores_mb,
                proxy_masks_mb,
            )
            scaled_aux_loss = learner.bw_flow_proxy_aux_coef * aux_loss
            total_loss = policy_total_loss + scaled_aux_loss

            grad_policy_loc = torch.autograd.grad(
                policy_loss,
                actor_out.loc,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
            grad_policy_log_scale = torch.autograd.grad(
                policy_loss,
                actor_out.log_scale,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
            grad_policy_total_loc = torch.autograd.grad(
                policy_total_loss,
                actor_out.loc,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
            grad_policy_total_log_scale = torch.autograd.grad(
                policy_total_loss,
                actor_out.log_scale,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
            grad_aux_loc = torch.autograd.grad(
                scaled_aux_loss,
                actor_out.loc,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
            grad_aux_log_scale = torch.autograd.grad(
                scaled_aux_loss,
                actor_out.log_scale,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]

            policy_loc_grad_norm = _tensor_l2_norm(grad_policy_loc, device=learner.device)
            policy_log_scale_grad_norm = _tensor_l2_norm(grad_policy_log_scale, device=learner.device)
            policy_total_loc_grad_norm = _tensor_l2_norm(grad_policy_total_loc, device=learner.device)
            policy_total_log_scale_grad_norm = _tensor_l2_norm(grad_policy_total_log_scale, device=learner.device)
            aux_loc_grad_norm = _tensor_l2_norm(grad_aux_loc, device=learner.device)
            aux_log_scale_grad_norm = _tensor_l2_norm(grad_aux_log_scale, device=learner.device)

            bw_param_grad_norm_policy = _grad_l2_norm(policy_loss, bw_params)
            bw_param_grad_norm_aux_scaled = _grad_l2_norm(scaled_aux_loss, bw_params)
            loc_head_grad_norm_policy = _grad_l2_norm(policy_loss, loc_head_params)
            loc_head_grad_norm_aux_scaled = _grad_l2_norm(scaled_aux_loss, loc_head_params)
            log_scale_head_grad_norm_policy = _grad_l2_norm(policy_loss, log_scale_head_params)
            log_scale_head_grad_norm_aux_scaled = _grad_l2_norm(scaled_aux_loss, log_scale_head_params)

            valid_mask, effective_loc = _bw_valid_and_effective_loc(local_batch_mb, actor_out.loc)
            mode_probs = _masked_simplex_probs(effective_loc, valid_mask)
            top1 = mode_probs.amax(dim=-1)
            top1_margin = _top1_margin(mode_probs)
            clip_frac = learner._clip_indicator_with_bounds(ratio, clip_lower, clip_upper).mean()

            loc_grad_ratio_aux_to_policy = aux_loc_grad_norm / policy_loc_grad_norm.clamp_min(1.0e-8)
            loc_grad_ratio_aux_to_policy_total = aux_loc_grad_norm / policy_total_loc_grad_norm.clamp_min(1.0e-8)
            log_scale_to_loc_policy_ratio = policy_log_scale_grad_norm / policy_loc_grad_norm.clamp_min(1.0e-8)
            log_scale_to_loc_policy_total_ratio = (
                policy_total_log_scale_grad_norm / policy_total_loc_grad_norm.clamp_min(1.0e-8)
            )
            bw_param_grad_ratio_aux_to_policy = (
                bw_param_grad_norm_aux_scaled / bw_param_grad_norm_policy.clamp_min(1.0e-8)
            )
            loc_head_grad_ratio_aux_to_policy = (
                loc_head_grad_norm_aux_scaled / loc_head_grad_norm_policy.clamp_min(1.0e-8)
            )
            loc_grad_cos_aux_vs_policy = _tensor_cosine(grad_aux_loc, grad_policy_loc, device=learner.device)
            loc_grad_cos_aux_vs_policy_total = _tensor_cosine(
                grad_aux_loc,
                grad_policy_total_loc,
                device=learner.device,
            )

            row = {
                "policy_loss": float(policy_loss.item()),
                "entropy_mean": float(entropy_mean.item()),
                "clip_frac": float(clip_frac.item()),
                "pairwise_acc": float(pair_acc.item()),
                "pair_count": float(pair_count.item()),
                "aux_loss_unscaled": float(aux_loss.item()),
                "aux_loss_scaled": float(scaled_aux_loss.item()),
                "regression_loss": float(regression_loss.item()),
                "policy_loc_grad_norm": float(policy_loc_grad_norm.item()),
                "policy_log_scale_grad_norm": float(policy_log_scale_grad_norm.item()),
                "policy_total_loc_grad_norm": float(policy_total_loc_grad_norm.item()),
                "policy_total_log_scale_grad_norm": float(policy_total_log_scale_grad_norm.item()),
                "aux_loc_grad_norm_scaled": float(aux_loc_grad_norm.item()),
                "aux_log_scale_grad_norm_scaled": float(aux_log_scale_grad_norm.item()),
                "loc_grad_ratio_aux_to_policy": float(loc_grad_ratio_aux_to_policy.item()),
                "loc_grad_ratio_aux_to_policy_total": float(loc_grad_ratio_aux_to_policy_total.item()),
                "log_scale_to_loc_policy_ratio": float(log_scale_to_loc_policy_ratio.item()),
                "log_scale_to_loc_policy_total_ratio": float(log_scale_to_loc_policy_total_ratio.item()),
                "loc_grad_cos_aux_vs_policy": float(loc_grad_cos_aux_vs_policy.item()),
                "loc_grad_cos_aux_vs_policy_total": float(loc_grad_cos_aux_vs_policy_total.item()),
                "bw_param_grad_norm_policy": float(bw_param_grad_norm_policy.item()),
                "bw_param_grad_norm_aux_scaled": float(bw_param_grad_norm_aux_scaled.item()),
                "bw_param_grad_ratio_aux_to_policy": float(bw_param_grad_ratio_aux_to_policy.item()),
                "loc_head_grad_norm_policy": float(loc_head_grad_norm_policy.item()),
                "loc_head_grad_norm_aux_scaled": float(loc_head_grad_norm_aux_scaled.item()),
                "loc_head_grad_ratio_aux_to_policy": float(loc_head_grad_ratio_aux_to_policy.item()),
                "log_scale_head_grad_norm_policy": float(log_scale_head_grad_norm_policy.item()),
                "log_scale_head_grad_norm_aux_scaled": float(log_scale_head_grad_norm_aux_scaled.item()),
                "policy_mode_top1": float(top1.mean().item()),
                "policy_mode_top1_margin": float(top1_margin.mean().item()),
            }
            for key, value in row.items():
                per_mb[key].append(float(value))
                if key in epoch_metrics:
                    epoch_metrics[key].append(float(value))

            learner.actor_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(learner.actor.parameters(), learner.max_grad_norm)
            learner.actor_optimizer.step()

        epoch_rows.append(
            {
                "epoch": float(epoch_idx),
                "policy_loc_grad_norm_mean": _safe_mean(epoch_metrics["policy_loc_grad_norm"]),
                "policy_log_scale_grad_norm_mean": _safe_mean(epoch_metrics["policy_log_scale_grad_norm"]),
                "aux_loc_grad_norm_scaled_mean": _safe_mean(epoch_metrics["aux_loc_grad_norm_scaled"]),
                "loc_grad_ratio_aux_to_policy_mean": _safe_mean(epoch_metrics["loc_grad_ratio_aux_to_policy"]),
                "loc_grad_cos_aux_vs_policy_mean": _safe_mean(epoch_metrics["loc_grad_cos_aux_vs_policy"]),
                "policy_mode_top1_margin_mean": _safe_mean(epoch_metrics["policy_mode_top1_margin"]),
                "pairwise_acc_mean": _safe_mean(epoch_metrics["pairwise_acc"]),
            }
        )

    return {
        "update": int(update),
        "rollout_env_steps": int(rollout_steps),
        "num_envs": int(num_envs),
        "vec_backend": str(vec_backend),
        "bw_stage_samples": int(cache["num_samples"]),
        "num_agents": int(cache["num_agents"]),
        "actor_update_steps": int(actor_update_steps),
        "logged_metrics": _history_tail(meta),
        "minibatch_summary": {key: _summarize(values) for key, values in per_mb.items()},
        "epochs": epoch_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--updates", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--vec-backend", type=str, default="sync")
    parser.add_argument("--rollout-env-steps", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=123450)
    parser.add_argument("--torch-seed-base", type=int, default=424242)
    parser.add_argument("--out-name", type=str, default="structured_bw_grad_split_diag.json")
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
        mb = update_summary.get("minibatch_summary", {})
        print(
            json.dumps(
                {
                    "update": int(update),
                    "bw_stage_samples": int(update_summary.get("bw_stage_samples", 0)),
                    "policy_loc_grad_norm_mean": float(mb.get("policy_loc_grad_norm", {}).get("mean", 0.0)),
                    "policy_log_scale_grad_norm_mean": float(
                        mb.get("policy_log_scale_grad_norm", {}).get("mean", 0.0)
                    ),
                    "aux_loc_grad_norm_scaled_mean": float(
                        mb.get("aux_loc_grad_norm_scaled", {}).get("mean", 0.0)
                    ),
                    "aux_log_scale_grad_norm_scaled_mean": float(
                        mb.get("aux_log_scale_grad_norm_scaled", {}).get("mean", 0.0)
                    ),
                    "loc_grad_ratio_aux_to_policy_mean": float(
                        mb.get("loc_grad_ratio_aux_to_policy", {}).get("mean", 0.0)
                    ),
                    "loc_grad_cos_aux_vs_policy_mean": float(
                        mb.get("loc_grad_cos_aux_vs_policy", {}).get("mean", 0.0)
                    ),
                    "bw_param_grad_ratio_aux_to_policy_mean": float(
                        mb.get("bw_param_grad_ratio_aux_to_policy", {}).get("mean", 0.0)
                    ),
                    "loc_head_grad_ratio_aux_to_policy_mean": float(
                        mb.get("loc_head_grad_ratio_aux_to_policy", {}).get("mean", 0.0)
                    ),
                    "policy_mode_top1_margin_mean": float(
                        mb.get("policy_mode_top1_margin", {}).get("mean", 0.0)
                    ),
                    "pairwise_acc_mean": float(mb.get("pairwise_acc", {}).get("mean", 0.0)),
                },
                ensure_ascii=False,
            )
        )

    out_path = run_dir / str(args.out_name)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
