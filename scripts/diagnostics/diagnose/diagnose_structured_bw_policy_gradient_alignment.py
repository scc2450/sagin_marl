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
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO, _collate_dataclass
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _teacher_bw_from_local_state(local_state: Any, assoc_bonus: float) -> torch.Tensor:
    valid_mask = (local_state.user_mask > 0.5) & (local_state.bw_valid_mask > 0.5)
    q = torch.clamp(local_state.user_nodes[..., 2], min=0.0)
    eta = torch.clamp(local_state.user_edges[..., 6], min=0.0)
    prev = torch.clamp(local_state.user_edges[..., 5], min=0.0)
    weights = q * (0.5 + eta)
    if assoc_bonus > 0.0:
        weights = weights * (1.0 + assoc_bonus * prev)
    weights = torch.where(valid_mask, torch.clamp(weights, min=0.0), torch.zeros_like(weights))
    denom = weights.sum(dim=-1, keepdim=True)
    teacher = torch.where(
        denom > 1.0e-8,
        weights / denom.clamp_min(1.0e-8),
        torch.zeros_like(weights),
    )
    valid_count = valid_mask.to(dtype=teacher.dtype).sum(dim=-1, keepdim=True)
    uniform = torch.where(
        valid_mask,
        torch.ones_like(teacher),
        torch.zeros_like(teacher),
    )
    uniform = torch.where(
        valid_count > 0.5,
        uniform / valid_count.clamp_min(1.0),
        torch.zeros_like(uniform),
    )
    teacher = torch.where(denom > 1.0e-8, teacher, uniform)
    return teacher


def _masked_simplex_kl(
    target: torch.Tensor,
    mode: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    target_valid = torch.where(valid_mask, target.clamp_min(eps), torch.zeros_like(target))
    mode_valid = torch.where(valid_mask, mode.clamp_min(eps), torch.zeros_like(mode))
    target_valid = target_valid / target_valid.sum(dim=-1, keepdim=True).clamp_min(eps)
    mode_valid = mode_valid / mode_valid.sum(dim=-1, keepdim=True).clamp_min(eps)
    kl = target_valid * (torch.log(target_valid.clamp_min(eps)) - torch.log(mode_valid.clamp_min(eps)))
    return (kl * valid_mask.to(dtype=kl.dtype)).sum(dim=-1)


def _flatten_grads(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> torch.Tensor:
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
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


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    denom = float(a.norm().item()) * float(b.norm().item())
    if denom <= 1.0e-12:
        return 0.0
    return float(torch.dot(a, b).item() / denom)


def _mean_abs(value: torch.Tensor) -> float:
    return float(value.detach().abs().mean().item()) if value.numel() > 0 else 0.0


def _safe_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().reshape(-1).to(dtype=torch.float32)
    y = y.detach().reshape(-1).to(dtype=torch.float32, device=x.device)
    finite = torch.isfinite(x) & torch.isfinite(y)
    if int(finite.sum().item()) <= 1:
        return 0.0
    x = x[finite]
    y = y[finite]
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    x_std = x_centered.pow(2).mean().sqrt()
    y_std = y_centered.pow(2).mean().sqrt()
    if float(x_std.item()) <= 1.0e-8 or float(y_std.item()) <= 1.0e-8:
        return 0.0
    return float(((x_centered * y_centered).mean() / (x_std * y_std).clamp_min(1.0e-8)).item())


def _masked_l1_per_row(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((a - b).abs() * mask.to(dtype=a.dtype)).sum(dim=-1)


def _mean_masked_cosine(delta: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    delta = delta * mask.to(dtype=delta.dtype)
    target = target * mask.to(dtype=target.dtype)
    delta_norm = delta.pow(2).sum(dim=-1).sqrt()
    target_norm = target.pow(2).sum(dim=-1).sqrt()
    valid = (delta_norm > 1.0e-12) & (target_norm > 1.0e-12)
    if not torch.any(valid):
        return 0.0
    cosine = (delta * target).sum(dim=-1) / (delta_norm * target_norm).clamp_min(1.0e-12)
    return float(cosine[valid].mean().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--actor_path", type=str, required=True)
    parser.add_argument("--critic_path", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--rollout_env_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch_threads", type=int, default=1)
    args = parser.parse_args()

    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    _set_all_seeds(int(args.seed))
    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(args.config))
    bundle = build_structured_modules_from_config(
        cfg,
        hidden_dim=int(args.hidden_dim),
        embed_dim=int(args.embed_dim),
    )
    actor = bundle.actor.to(device)
    critic = bundle.critic.to(device)
    load_checkpoint_forgiving(actor, str(args.actor_path), map_location=device, strict=True)
    load_checkpoint_forgiving(critic, str(args.critic_path), map_location=device, strict=True)
    actor.train()
    critic.eval()

    trainer = StructuredMAPPO(
        actor,
        critic,
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_ratio=float(cfg.clip_ratio),
        value_coef=float(cfg.value_coef),
        entropy_coef=float(cfg.entropy_coef),
        max_grad_norm=float(cfg.max_grad_norm),
        ppo_epochs=int(cfg.ppo_epochs),
        num_mini_batch=int(cfg.num_mini_batch),
        danger_imitation_enabled=bool(getattr(cfg, "danger_imitation_enabled", False)),
        danger_imitation_coef=float(getattr(cfg, "danger_imitation_coef", 0.0) or 0.0),
        actor_optimizer=None,
        critic_optimizer=None,
        device=device,
        target_mode="step_level",
        joint_stage_updates=False,
        cfg=cfg,
        train_accel=bool(getattr(cfg, "train_accel", True)),
        train_sat=bool(getattr(cfg, "train_sat", True)),
        train_bw=bool(getattr(cfg, "train_bw", True)),
        exec_accel_source=str(getattr(cfg, "exec_accel_source", "policy")),
        exec_sat_source=str(getattr(cfg, "exec_sat_source", "policy")),
        exec_bw_source=str(getattr(cfg, "exec_bw_source", "policy")),
    )

    env = make_structured_env(cfg, mode="script")
    driver = as_structured_driver(env)
    buffer = StructuredRolloutBuffer()
    try:
        env.reset(seed=int(args.seed))
        for _ in range(int(args.rollout_env_steps)):
            trainer.collect_env_step(driver, buffer, deterministic=False)
        bootstrap_world_state = buffer.build_bootstrap_view()
        gae = trainer.compute_returns_and_advantages(buffer, bootstrap_world_state)
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

    rollout_views = buffer.build_rollout_views(device)
    bw_stage_batch = rollout_views.training_view.stage_batches.get(2)
    if bw_stage_batch is None or int(bw_stage_batch.num_samples) <= 0:
        raise RuntimeError("No BW transitions collected.")
    stage_ids = batches["stage_ids"]
    bw_adv = torch.from_numpy(gae["advantages"][bw_idx_np]).to(device)
    if bw_adv.numel() > 1 and bool(getattr(cfg, "stagewise_advantage_norm_enabled", False)):
        bw_adv = (bw_adv - bw_adv.mean()) / bw_adv.std(unbiased=False).clamp_min(1.0e-8)
    elif bw_adv.numel() > 1:
        bw_adv = (bw_adv - bw_adv.mean()) / bw_adv.std(unbiased=False).clamp_min(1.0e-8)

    bw_local_batch = bw_stage_batch.local_batch
    bw_world_batch = bw_stage_batch.world_batch
    joint_actions = bw_stage_batch.actions.to(device)
    old_logprobs = bw_stage_batch.old_logprobs.to(device)

    num_samples = int(joint_actions.shape[0])
    num_agents = int(joint_actions.shape[1])
    flat_local_batch = bw_local_batch
    new_logprob, entropy, actor_out = trainer._stage_actor_eval_from_batch(
        2,
        flat_local_batch,
        joint_actions,
        num_agents,
    )
    cache = {
        "advantages": bw_adv,
        "old_logprobs": old_logprobs,
        "num_agents": num_agents,
        "local_batch": bw_local_batch,
        "flat_indices": torch.arange(
            num_samples * num_agents,
            device=device,
            dtype=torch.long,
        ).reshape(num_samples, num_agents),
    }
    (
        ppo_policy_loss,
        entropy_mean,
        approx_kl,
        clip_frac,
        corr_valid,
        corr_latent,
        kappa_mean,
        kappa_p10,
        kappa_p90,
        kappa_hi_frac,
    ) = trainer._stage_policy_terms(
        2,
        actor_out=actor_out,
        cache=cache,
        mb_rel=torch.arange(num_samples, device=device, dtype=torch.long),
        joint_actions_mb=joint_actions,
        new_logprob=new_logprob,
        entropy=entropy,
    )

    valid_mask = (bw_local_batch.user_mask > 0.5) & (bw_local_batch.bw_valid_mask > 0.5)
    teacher_action = _teacher_bw_from_local_state(
        bw_local_batch,
        assoc_bonus=float(getattr(cfg, "baseline_assoc_bonus", 0.3) or 0.0),
    )
    det_action = actor.bw_policy.deterministic_action(
        bw_local_batch,
        readout="latent_mean_pushforward",
    )
    det_actor_out = actor.evaluate_bw(bw_local_batch, det_action)
    det_weighted_logprob_loss = -(bw_adv * det_actor_out.logprob.reshape(num_samples, num_agents).sum(dim=1)).mean()
    imitation_loss = _masked_simplex_kl(teacher_action, det_action, valid_mask).mean()
    imitation_l1 = (
        (det_action - teacher_action).abs() * valid_mask.to(dtype=det_action.dtype)
    ).sum(dim=-1).mean()
    sample_teacher_l1_per_sample = (
        (joint_actions.reshape(num_samples * num_agents, -1) - teacher_action).abs()
        * valid_mask.to(dtype=det_action.dtype)
    ).sum(dim=-1)
    sample_teacher_l1 = sample_teacher_l1_per_sample.mean()
    positive_adv = bw_adv > 0.0
    negative_adv = bw_adv < 0.0

    bw_params = [param for param in actor.bw_policy.parameters() if param.requires_grad]
    loc_head_params = [param for param in actor.bw_policy.loc_head.parameters() if param.requires_grad]
    grad_ppo_all = _flatten_grads(ppo_policy_loss, bw_params)
    grad_detlogprob_all = _flatten_grads(det_weighted_logprob_loss, bw_params)
    grad_imitation_all = _flatten_grads(imitation_loss, bw_params)
    grad_ppo_loc = _flatten_grads(ppo_policy_loss, loc_head_params)
    grad_detlogprob_loc = _flatten_grads(det_weighted_logprob_loss, loc_head_params)
    grad_imitation_loc = _flatten_grads(imitation_loss, loc_head_params)

    actor_step = copy.deepcopy(actor).to(device)
    actor_step.train()
    actor_step_optimizer = torch.optim.Adam(actor_step.parameters(), lr=float(getattr(cfg, "actor_lr", 3.0e-4)))

    def _ppo_loss_for_actor(actor_model: Any) -> tuple[torch.Tensor, Any]:
        actor_out_model = actor_model.evaluate_bw(
            bw_local_batch,
            joint_actions.reshape(num_samples * num_agents, -1),
        )
        new_logprob_model = actor_out_model.logprob.reshape(num_samples, num_agents).sum(dim=1)
        entropy_model = actor_out_model.entropy.reshape(num_samples, num_agents).sum(dim=1)
        policy_loss_model, *_rest = trainer._stage_policy_terms(
            2,
            actor_out=actor_out_model,
            cache=cache,
            mb_rel=torch.arange(num_samples, device=device, dtype=torch.long),
            joint_actions_mb=joint_actions,
            new_logprob=new_logprob_model,
            entropy=entropy_model,
        )
        return policy_loss_model, actor_out_model

    det_before = actor_step.bw_policy.deterministic_action(
        bw_local_batch,
        readout="latent_mean_pushforward",
    )
    det_teacher_l1_before = _masked_l1_per_row(det_before, teacher_action, valid_mask)
    det_teacher_kl_before = _masked_simplex_kl(teacher_action, det_before, valid_mask)
    ppo_step_loss_before, _ = _ppo_loss_for_actor(actor_step)

    actor_step_optimizer.zero_grad(set_to_none=True)
    ppo_step_loss_before.backward()
    torch.nn.utils.clip_grad_norm_(actor_step.parameters(), float(getattr(cfg, "max_grad_norm", 0.5) or 0.5))
    actor_step_optimizer.step()

    det_after = actor_step.bw_policy.deterministic_action(
        bw_local_batch,
        readout="latent_mean_pushforward",
    )
    det_teacher_l1_after = _masked_l1_per_row(det_after, teacher_action, valid_mask)
    det_teacher_kl_after = _masked_simplex_kl(teacher_action, det_after, valid_mask)
    ppo_step_loss_after, _ = _ppo_loss_for_actor(actor_step)
    det_delta = det_after - det_before
    teacher_delta = teacher_action - det_before
    improved_mask = det_teacher_l1_after < det_teacher_l1_before
    worsened_mask = det_teacher_l1_after > det_teacher_l1_before

    summary = {
        "config": str(args.config),
        "actor_path": str(args.actor_path),
        "critic_path": str(args.critic_path),
        "num_bw_samples": int(num_samples),
        "ppo_policy_loss": float(ppo_policy_loss.detach().item()),
        "det_weighted_logprob_loss": float(det_weighted_logprob_loss.detach().item()),
        "imitation_loss": float(imitation_loss.detach().item()),
        "imitation_l1": float(imitation_l1.detach().item()),
        "sample_teacher_l1": float(sample_teacher_l1.detach().item()),
        "adv_teacher_l1_corr": _safe_corr(bw_adv, sample_teacher_l1_per_sample),
        "sample_teacher_l1_pos_adv": (
            float(sample_teacher_l1_per_sample[positive_adv].mean().item()) if torch.any(positive_adv) else 0.0
        ),
        "sample_teacher_l1_neg_adv": (
            float(sample_teacher_l1_per_sample[negative_adv].mean().item()) if torch.any(negative_adv) else 0.0
        ),
        "approx_kl_bw": float(approx_kl.detach().item()),
        "clip_frac_bw": float(clip_frac.detach().item()),
        "entropy_bw": float(entropy_mean.detach().item()),
        "bw_kappa_mean": float(kappa_mean.detach().item()),
        "bw_kappa_p10": float(kappa_p10.detach().item()),
        "bw_kappa_p90": float(kappa_p90.detach().item()),
        "bw_kappa_hi_frac": float(kappa_hi_frac.detach().item()),
        "bw_abs_log_ratio_corr_valid_count": float(corr_valid.detach().item()),
        "bw_abs_log_ratio_corr_latent_count": float(corr_latent.detach().item()),
        "grad_cosine_all": _cosine(grad_ppo_all, grad_imitation_all),
        "grad_cosine_loc_head": _cosine(grad_ppo_loc, grad_imitation_loc),
        "grad_cosine_detlogprob_all": _cosine(grad_detlogprob_all, grad_imitation_all),
        "grad_cosine_detlogprob_loc_head": _cosine(grad_detlogprob_loc, grad_imitation_loc),
        "grad_norm_ppo_all": float(grad_ppo_all.norm().item()),
        "grad_norm_detlogprob_all": float(grad_detlogprob_all.norm().item()),
        "grad_norm_imitation_all": float(grad_imitation_all.norm().item()),
        "grad_norm_ppo_loc_head": float(grad_ppo_loc.norm().item()),
        "grad_norm_detlogprob_loc_head": float(grad_detlogprob_loc.norm().item()),
        "grad_norm_imitation_loc_head": float(grad_imitation_loc.norm().item()),
        "grad_abs_mean_ppo_all": _mean_abs(grad_ppo_all),
        "grad_abs_mean_imitation_all": _mean_abs(grad_imitation_all),
        "one_step_det_teacher_l1_before": float(det_teacher_l1_before.mean().item()),
        "one_step_det_teacher_l1_after": float(det_teacher_l1_after.mean().item()),
        "one_step_det_teacher_kl_before": float(det_teacher_kl_before.mean().item()),
        "one_step_det_teacher_kl_after": float(det_teacher_kl_after.mean().item()),
        "one_step_det_teacher_l1_delta": float((det_teacher_l1_after - det_teacher_l1_before).mean().item()),
        "one_step_det_teacher_kl_delta": float((det_teacher_kl_after - det_teacher_kl_before).mean().item()),
        "one_step_det_improved_frac": float(improved_mask.to(dtype=torch.float32).mean().item()),
        "one_step_det_worsened_frac": float(worsened_mask.to(dtype=torch.float32).mean().item()),
        "one_step_det_toward_teacher_cosine": _mean_masked_cosine(det_delta, teacher_delta, valid_mask),
        "one_step_ppo_loss_before": float(ppo_step_loss_before.detach().item()),
        "one_step_ppo_loss_after": float(ppo_step_loss_after.detach().item()),
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
