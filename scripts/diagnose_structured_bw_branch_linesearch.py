from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_bw_update_direction import (
    _bw_action_from_actor,
    _rollout_from_snapshot_with_driver,
    build_bw_advantage_probe_context,
    collect_bw_snapshot_panel,
    compute_bw_branch_advantage_override,
    evaluate_bw_advantage_alignment,
    evaluate_bw_snapshot_panel,
)
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _parse_float_list(value: str) -> list[float]:
    items: list[float] = []
    for part in str(value).replace(";", ",").split(","):
        text = part.strip()
        if not text:
            continue
        items.append(float(text))
    return items


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    xa = np.asarray(x, dtype=np.float64).reshape(-1)
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    if xa.size <= 1 or ya.size <= 1 or xa.size != ya.size:
        return 0.0
    finite = np.isfinite(xa) & np.isfinite(ya)
    if int(np.sum(finite)) <= 1:
        return 0.0
    xa = xa[finite]
    ya = ya[finite]
    if float(np.std(xa)) <= 1.0e-12 or float(np.std(ya)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def _flatten_module_params(module: torch.nn.Module) -> torch.Tensor:
    parts = [
        param.detach().reshape(-1).to(dtype=torch.float32, device="cpu")
        for param in module.parameters()
        if param.requires_grad
    ]
    if not parts:
        return torch.zeros((0,), dtype=torch.float32)
    return torch.cat(parts, dim=0)


def _enable_module(module: torch.nn.Module | None) -> list[str]:
    enabled: list[str] = []
    if module is None:
        return enabled
    for name, param in module.named_parameters():
        param.requires_grad_(True)
        enabled.append(str(name))
    return enabled


def _configure_bw_freeze(actor: torch.nn.Module, freeze_mode: str) -> dict[str, object]:
    mode = str(freeze_mode or "none").strip().lower()
    if mode not in {"none", "mean_only", "concentration_only"}:
        raise ValueError("freeze_mode must be one of {'none', 'mean_only', 'concentration_only'}.")
    for param in actor.parameters():
        param.requires_grad_(False)
    if mode == "none":
        for param in actor.parameters():
            param.requires_grad_(True)
        trainable_names = [name for name, param in actor.named_parameters() if param.requires_grad]
        return {"freeze_mode": mode, "trainable_param_names": trainable_names}

    bw_policy = getattr(actor, "bw_policy", None)
    if bw_policy is None:
        raise RuntimeError("Structured actor is missing bw_policy.")

    trainable_param_names: list[str] = []
    if mode == "mean_only":
        for attr in ("loc_head", "alpha_head", "tau_head", "log_scale_scalar_head"):
            module = getattr(bw_policy, attr, None)
            for local_name in _enable_module(module):
                trainable_param_names.append(f"bw_policy.{attr}.{local_name}")
        for attr in (
            "lowdim_q_coef_raw",
            "lowdim_eta_coef_raw",
            "lowdim_qeta_coef_raw",
            "lowdim_alpha_raw",
            "lowdim_tau_raw",
            "lowdim_log_scale_scalar",
        ):
            param = getattr(bw_policy, attr, None)
            if isinstance(param, torch.nn.Parameter):
                param.requires_grad_(True)
                trainable_param_names.append(f"bw_policy.{attr}")
    else:
        for attr in ("kappa_head",):
            module = getattr(bw_policy, attr, None)
            for local_name in _enable_module(module):
                trainable_param_names.append(f"bw_policy.{attr}.{local_name}")
        for attr in ("lowdim_kappa_raw",):
            param = getattr(bw_policy, attr, None)
            if isinstance(param, torch.nn.Parameter):
                param.requires_grad_(True)
                trainable_param_names.append(f"bw_policy.{attr}")

    if not trainable_param_names:
        raise RuntimeError(f"freeze_mode={mode} left no trainable BW parameters.")
    return {"freeze_mode": mode, "trainable_param_names": trainable_param_names}


def _resolve_run_artifact(run_dir: Path, name: str) -> Path:
    path = run_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Expected '{name}' under {run_dir}, but it was not found.")
    return path


def _resolve_config_path(run_dir: Path) -> Path:
    for name in ("config_source.yaml", "config.yaml"):
        path = run_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No config_source.yaml or config.yaml found under {run_dir}.")


def _build_actor_critic_and_learner(
    *,
    cfg_template,
    actor_path: Path,
    critic_path: Path,
    device: torch.device,
    hidden_dim: int,
    embed_dim: int,
    actor_lr: float,
    critic_lr: float,
    ppo_epochs: int,
    num_mini_batch: int,
    value_coef: float,
    entropy_coef: float,
    freeze_mode: str,
):
    cfg_runtime = copy.deepcopy(cfg_template)
    cfg_runtime.bw_actor_advantage_override_mode = "branch_delta"
    cfg_runtime.update_direction_probe_enabled = True
    cfg_runtime.critic_warmup_before_actor_epochs = 0
    cfg_runtime.critic_warmup_recompute_advantages = False
    bundle = build_structured_modules_from_config(
        cfg_runtime,
        hidden_dim=int(hidden_dim),
        embed_dim=int(embed_dim),
    )
    actor = bundle.actor.to(device)
    critic = bundle.critic.to(device)
    load_checkpoint_forgiving(actor, str(actor_path), map_location=device, strict=True)
    load_checkpoint_forgiving(critic, str(critic_path), map_location=device, strict=True)
    freeze_info = _configure_bw_freeze(actor, freeze_mode)
    trainable_actor_params = [param for param in actor.parameters() if param.requires_grad]
    actor_optimizer = torch.optim.Adam(trainable_actor_params, lr=float(actor_lr))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(critic_lr))
    learner = StructuredMAPPO(
        actor=actor,
        critic=critic,
        gamma=float(cfg_runtime.gamma),
        gae_lambda=float(cfg_runtime.gae_lambda),
        clip_ratio=float(cfg_runtime.clip_ratio),
        value_coef=float(value_coef),
        entropy_coef=float(entropy_coef),
        max_grad_norm=float(cfg_runtime.max_grad_norm),
        ppo_epochs=int(ppo_epochs),
        num_mini_batch=int(num_mini_batch),
        target_mode="step_level",
        danger_imitation_enabled=bool(getattr(cfg_runtime, "danger_imitation_enabled", False)),
        danger_imitation_coef=float(getattr(cfg_runtime, "danger_imitation_coef", 0.0) or 0.0),
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        device=device,
        cfg=cfg_runtime,
        train_accel=bool(True if getattr(cfg_runtime, "train_accel", None) is None else getattr(cfg_runtime, "train_accel")),
        train_sat=bool(True if getattr(cfg_runtime, "train_sat", None) is None else getattr(cfg_runtime, "train_sat")),
        train_bw=bool(True if getattr(cfg_runtime, "train_bw", None) is None else getattr(cfg_runtime, "train_bw")),
        exec_accel_source=str(getattr(cfg_runtime, "exec_accel_source", "policy") or "policy"),
        exec_sat_source=str(getattr(cfg_runtime, "exec_sat_source", "policy") or "policy"),
        exec_bw_source=str(getattr(cfg_runtime, "exec_bw_source", "policy") or "policy"),
    )
    actor.train()
    critic.train()
    return cfg_runtime, actor, critic, learner, freeze_info


def _collect_fixed_rollout(
    *,
    learner: StructuredMAPPO,
    cfg,
    rollout_env_steps: int,
    rollout_seed: int,
) -> tuple[StructuredRolloutBuffer, dict[int, object]]:
    env = make_structured_env(cfg, mode="script")
    driver = as_structured_driver(env)
    buffer = StructuredRolloutBuffer()
    try:
        env.reset(seed=int(rollout_seed))
        for _ in range(int(rollout_env_steps)):
            learner.collect_env_step(driver, buffer, deterministic=False)
        bootstrap_world_state = buffer.build_bootstrap_view()
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
    return buffer, bootstrap_world_state


def _full_batch_bw_logprob_metrics(
    *,
    pre_actor,
    post_actor,
    probe_context: dict[str, object],
    device: torch.device,
    raw_branch_advantages: np.ndarray,
) -> dict[str, float]:
    local_batch = probe_context["local_batch"]
    joint_actions = probe_context["joint_actions"]
    old_logprobs = probe_context["old_logprobs"]
    support_masks = probe_context.get("support_masks")
    num_samples = int(joint_actions.shape[0])
    num_agents = int(probe_context["num_agents"])
    flat_action = joint_actions.to(device).reshape(num_samples * num_agents, -1)
    flat_support_mask = None
    if support_masks is not None:
        flat_support_mask = support_masks.reshape(num_samples * num_agents, -1).to(device)
    with torch.inference_mode():
        pre_eval = pre_actor.evaluate_bw(local_batch, flat_action, support_mask_override=flat_support_mask)
        post_eval = post_actor.evaluate_bw(local_batch, flat_action, support_mask_override=flat_support_mask)
    pre_joint_logprob = pre_eval.logprob.reshape(num_samples, num_agents).sum(dim=1)
    post_joint_logprob = post_eval.logprob.reshape(num_samples, num_agents).sum(dim=1)
    old_joint_logprob = old_logprobs.to(device)
    replay_abs = (pre_joint_logprob - old_joint_logprob).abs().detach().cpu().numpy().astype(np.float64)
    delta_logprob = (post_joint_logprob - old_joint_logprob).detach().cpu().numpy().astype(np.float64)
    exact_delta = (post_joint_logprob - pre_joint_logprob).detach().cpu().numpy().astype(np.float64)
    branch = np.asarray(raw_branch_advantages, dtype=np.float64).reshape(-1)
    if branch.shape[0] != delta_logprob.shape[0]:
        raise ValueError(
            f"branch labels have {branch.shape[0]} entries, but full batch has {delta_logprob.shape[0]} BW samples."
        )
    return {
        "sample_count": int(delta_logprob.shape[0]),
        "old_logprob_replay_abs_diff_mean": float(np.mean(replay_abs)) if replay_abs.size > 0 else 0.0,
        "mean_abs_delta_logprob": float(np.mean(np.abs(delta_logprob))) if delta_logprob.size > 0 else 0.0,
        "mean_branch_delta_times_delta_logprob": float(np.mean(branch * delta_logprob)) if branch.size > 0 else 0.0,
        "corr_branch_delta_vs_delta_logprob": _safe_corr(branch, delta_logprob),
        "approx_kl_bw_old_to_new": float(np.mean(-delta_logprob)) if delta_logprob.size > 0 else 0.0,
        "approx_kl_bw_pre_to_new": float(np.mean(-exact_delta)) if exact_delta.size > 0 else 0.0,
    }


def _full_batch_bw_geometry_metrics(
    *,
    pre_actor,
    post_actor,
    probe_context: dict[str, object],
) -> dict[str, float]:
    local_batch = probe_context["local_batch"]
    with torch.inference_mode():
        pre_out = pre_actor.act_bw(local_batch, deterministic=False)
        post_out = post_actor.act_bw(local_batch, deterministic=False)
    valid_mask = getattr(local_batch, "bw_valid_mask", None)
    if valid_mask is None:
        raise RuntimeError("BW local batch is missing bw_valid_mask.")
    valid = valid_mask > 0.5
    pre_det = getattr(pre_out, "det_mean", None)
    post_det = getattr(post_out, "det_mean", None)
    if pre_det is None or post_det is None:
        raise RuntimeError("BW policy output is missing det_mean.")
    det_abs = (post_det - pre_det).abs()
    det_l1 = det_abs.sum(dim=-1)
    det_l1_valid = (det_abs * valid.to(dtype=det_abs.dtype)).sum(dim=-1)
    metrics = {
        "det_mean_l1_mean": float(det_l1.mean().item()) if det_l1.numel() > 0 else 0.0,
        "det_mean_l1_valid_mean": float(det_l1_valid.mean().item()) if det_l1_valid.numel() > 0 else 0.0,
        "entropy_pre_mean": float(pre_out.entropy.mean().item()) if pre_out.entropy.numel() > 0 else 0.0,
        "entropy_post_mean": float(post_out.entropy.mean().item()) if post_out.entropy.numel() > 0 else 0.0,
        "entropy_delta_mean": float((post_out.entropy - pre_out.entropy).mean().item()) if pre_out.entropy.numel() > 0 else 0.0,
    }
    pre_kappa = getattr(pre_out, "kappa", None)
    post_kappa = getattr(post_out, "kappa", None)
    if pre_kappa is not None and post_kappa is not None:
        metrics.update(
            {
                "kappa_pre_mean": float(pre_kappa.mean().item()),
                "kappa_post_mean": float(post_kappa.mean().item()),
                "kappa_delta_mean": float((post_kappa - pre_kappa).mean().item()),
                "kappa_abs_delta_mean": float((post_kappa - pre_kappa).abs().mean().item()),
            }
        )
    else:
        metrics.update(
            {
                "kappa_pre_mean": 0.0,
                "kappa_post_mean": 0.0,
                "kappa_delta_mean": 0.0,
                "kappa_abs_delta_mean": 0.0,
            }
        )
    return metrics


def _estimate_local_distribution_gain(
    *,
    pre_actor,
    post_actor,
    snapshot_states: list[dict[str, object]],
    cfg,
    device: torch.device,
    horizon: int,
    action_samples: int,
    seed_base: int,
    follow_policy_mode: str,
) -> dict[str, object]:
    follow_mode = str(follow_policy_mode or "stochastic").strip().lower()
    if follow_mode not in {"deterministic", "stochastic"}:
        raise ValueError("follow_policy_mode must be one of {'deterministic', 'stochastic'}.")
    action_samples_eff = max(int(action_samples), 1)
    if follow_mode == "deterministic":
        action_samples_eff = 1

    env = make_structured_env(cfg, mode="script")
    driver = as_structured_driver(env)
    rows: list[dict[str, float]] = []
    try:
        for state_idx, snapshot_state in enumerate(snapshot_states):
            snapshot = driver.load_bw_stage_state(snapshot_state)
            old_det_action = _bw_action_from_actor(pre_actor, snapshot, device, deterministic=True)
            new_det_action = _bw_action_from_actor(post_actor, snapshot, device, deterministic=True)
            old_det_roll = _rollout_from_snapshot_with_driver(
                probe_driver=driver,
                snapshot_state=snapshot_state,
                first_action=np.asarray(old_det_action, dtype=np.float32),
                cfg=cfg,
                actor=pre_actor,
                device=device,
                k_steps=int(horizon),
                gamma=float(cfg.gamma),
                follow_deterministic=(follow_mode == "deterministic"),
                sample_seed_base=None if follow_mode == "deterministic" else int(seed_base) + int(state_idx) * 100_000 + 500,
            )
            new_det_roll = _rollout_from_snapshot_with_driver(
                probe_driver=driver,
                snapshot_state=snapshot_state,
                first_action=np.asarray(new_det_action, dtype=np.float32),
                cfg=cfg,
                actor=pre_actor,
                device=device,
                k_steps=int(horizon),
                gamma=float(cfg.gamma),
                follow_deterministic=(follow_mode == "deterministic"),
                sample_seed_base=None if follow_mode == "deterministic" else int(seed_base) + int(state_idx) * 100_000 + 500,
            )
            old_qs: list[float] = []
            new_qs: list[float] = []
            for sample_idx in range(int(action_samples_eff)):
                sample_seed = int(seed_base) + int(state_idx) * 100_000 + int(sample_idx) * 1_000
                old_action = _bw_action_from_actor(
                    pre_actor,
                    snapshot,
                    device,
                    deterministic=False,
                    sample_seed=int(sample_seed) + 1,
                )
                new_action = _bw_action_from_actor(
                    post_actor,
                    snapshot,
                    device,
                    deterministic=False,
                    sample_seed=int(sample_seed) + 2,
                )
                old_roll = _rollout_from_snapshot_with_driver(
                    probe_driver=driver,
                    snapshot_state=snapshot_state,
                    first_action=np.asarray(old_action, dtype=np.float32),
                    cfg=cfg,
                    actor=pre_actor,
                    device=device,
                    k_steps=int(horizon),
                    gamma=float(cfg.gamma),
                    follow_deterministic=(follow_mode == "deterministic"),
                    sample_seed_base=None if follow_mode == "deterministic" else int(sample_seed) + 100,
                )
                new_roll = _rollout_from_snapshot_with_driver(
                    probe_driver=driver,
                    snapshot_state=snapshot_state,
                    first_action=np.asarray(new_action, dtype=np.float32),
                    cfg=cfg,
                    actor=pre_actor,
                    device=device,
                    k_steps=int(horizon),
                    gamma=float(cfg.gamma),
                    follow_deterministic=(follow_mode == "deterministic"),
                    sample_seed_base=None if follow_mode == "deterministic" else int(sample_seed) + 100,
                )
                old_qs.append(float(old_roll["reward"]))
                new_qs.append(float(new_roll["reward"]))
            old_q_mean = float(np.mean(np.asarray(old_qs, dtype=np.float64))) if old_qs else 0.0
            new_q_mean = float(np.mean(np.asarray(new_qs, dtype=np.float64))) if new_qs else 0.0
            rows.append(
                {
                    "state_idx": float(state_idx),
                    "old_q_mean": old_q_mean,
                    "new_q_mean": new_q_mean,
                    "delta_local": float(new_q_mean - old_q_mean),
                    "old_det_q": float(old_det_roll["reward"]),
                    "new_det_q": float(new_det_roll["reward"]),
                    "delta_local_det": float(new_det_roll["reward"] - old_det_roll["reward"]),
                }
            )
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

    delta_local = np.asarray([row["delta_local"] for row in rows], dtype=np.float64)
    delta_local_det = np.asarray([row["delta_local_det"] for row in rows], dtype=np.float64)
    return {
        "sample_count": int(len(rows)),
        "action_samples_per_policy": int(action_samples_eff),
        "follow_policy_mode": str(follow_mode),
        "delta_local_mean": float(np.mean(delta_local)) if delta_local.size > 0 else 0.0,
        "delta_local_pos_frac": float(np.mean(delta_local > 0.0)) if delta_local.size > 0 else 0.0,
        "delta_local_abs_mean": float(np.mean(np.abs(delta_local))) if delta_local.size > 0 else 0.0,
        "delta_local_det_mean": float(np.mean(delta_local_det)) if delta_local_det.size > 0 else 0.0,
        "delta_local_det_pos_frac": float(np.mean(delta_local_det > 0.0)) if delta_local_det.size > 0 else 0.0,
        "delta_local_det_abs_mean": float(np.mean(np.abs(delta_local_det))) if delta_local_det.size > 0 else 0.0,
        "examples": rows[:8],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--actor_path", type=str, default=None)
    parser.add_argument("--critic_path", type=str, default=None)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--freeze_mode", type=str, default="none")
    parser.add_argument("--rollout_env_steps", type=int, default=100)
    parser.add_argument("--rollout_seed", type=int, default=61000)
    parser.add_argument("--branch_horizon", type=int, default=10)
    parser.add_argument("--branch_samples", type=int, default=2)
    parser.add_argument("--branch_follow_policy_mode", type=str, default="stochastic")
    parser.add_argument("--branch_normalize", action="store_true")
    parser.add_argument("--local_eval_action_samples", type=int, default=6)
    parser.add_argument("--panel_states", type=int, default=12)
    parser.add_argument("--panel_episodes", type=int, default=6)
    parser.add_argument("--panel_seed", type=int, default=35000)
    parser.add_argument("--panel_eval_seed_base", type=int, default=72000)
    parser.add_argument("--panel_k_steps", type=int, default=20)
    parser.add_argument("--panel_actor_policy_mode", type=str, default="stochastic")
    parser.add_argument("--panel_actor_policy_samples", type=int, default=4)
    parser.add_argument("--probe_sample_limit", type=int, default=12)
    parser.add_argument("--probe_true_mc_samples", type=int, default=4)
    parser.add_argument("--probe_seed_base", type=int, default=91000)
    parser.add_argument("--actor_lr_scales", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--ppo_epochs", type=int, default=1)
    parser.add_argument("--num_mini_batch", type=int, default=1)
    parser.add_argument("--value_coef", type=float, default=0.0)
    parser.add_argument("--entropy_coef", type=float, default=0.0)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch_threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    config_path = _resolve_config_path(run_dir)
    actor_path = Path(args.actor_path) if args.actor_path is not None else _resolve_run_artifact(run_dir, "actor_final.pt")
    critic_path = Path(args.critic_path) if args.critic_path is not None else _resolve_run_artifact(run_dir, "critic_final.pt")
    cfg_template = load_config(str(config_path))
    actor_lr_base = float(getattr(cfg_template, "actor_lr", 3.0e-4))
    critic_lr_base = float(getattr(cfg_template, "critic_lr", 1.0e-3))
    actor_lr_scales = _parse_float_list(args.actor_lr_scales)
    if not actor_lr_scales:
        raise ValueError("actor_lr_scales must provide at least one positive scale.")

    _set_all_seeds(int(args.rollout_seed))
    cfg_base, base_actor, _base_critic, base_learner, base_freeze_info = _build_actor_critic_and_learner(
        cfg_template=cfg_template,
        actor_path=actor_path,
        critic_path=critic_path,
        device=device,
        hidden_dim=int(args.hidden_dim),
        embed_dim=int(args.embed_dim),
        actor_lr=float(actor_lr_base),
        critic_lr=float(critic_lr_base),
        ppo_epochs=int(args.ppo_epochs),
        num_mini_batch=int(args.num_mini_batch),
        value_coef=float(args.value_coef),
        entropy_coef=float(args.entropy_coef),
        freeze_mode=str(args.freeze_mode),
    )
    base_actor.eval()

    buffer, bootstrap_world_state = _collect_fixed_rollout(
        learner=base_learner,
        cfg=cfg_base,
        rollout_env_steps=int(args.rollout_env_steps),
        rollout_seed=int(args.rollout_seed),
    )
    branch_summary = compute_bw_branch_advantage_override(
        learner=base_learner,
        buffer=buffer,
        bootstrap_world_state=bootstrap_world_state,
        cfg=cfg_base,
        device=device,
        horizon=int(args.branch_horizon),
        branch_samples=int(args.branch_samples),
        branch_seed=int(args.probe_seed_base),
        ref_mode="deterministic",
        follow_policy_mode=str(args.branch_follow_policy_mode),
        normalize=bool(args.branch_normalize),
    )
    if branch_summary is None:
        raise RuntimeError("Failed to compute branch advantage override on the fixed rollout.")
    base_probe_context = build_bw_advantage_probe_context(
        base_learner,
        buffer,
        bootstrap_world_state,
        sample_limit=max(int(args.probe_sample_limit), 1),
        sample_seed=int(args.probe_seed_base),
    )
    if base_probe_context is None:
        raise RuntimeError("Failed to build fixed BW probe context from the rollout.")
    local_snapshot_states = list(base_probe_context.get("snapshot_states", []) or [])
    if not local_snapshot_states:
        raise RuntimeError("Fixed BW probe context did not include snapshot_states for local evaluation.")

    panel = collect_bw_snapshot_panel(
        cfg_base,
        episodes=max(int(args.panel_episodes), 1),
        states=max(int(args.panel_states), 1),
        seed=int(args.panel_seed),
    )
    panel_pre_stochastic = evaluate_bw_snapshot_panel(
        base_actor,
        panel,
        cfg_base,
        device,
        k_steps=int(args.panel_k_steps),
        deterministic=False,
        actor_policy_samples=max(int(args.panel_actor_policy_samples), 1),
        seed_base=int(args.panel_eval_seed_base),
    )
    panel_pre_deterministic = evaluate_bw_snapshot_panel(
        base_actor,
        panel,
        cfg_base,
        device,
        k_steps=int(args.panel_k_steps),
        deterministic=True,
        actor_policy_samples=1,
        seed_base=None,
    )
    base_actor_params = _flatten_module_params(base_actor)

    scale_results: list[dict[str, object]] = []
    best_scale_by_branch_product = None
    best_branch_product = None
    for scale_idx, scale in enumerate(actor_lr_scales):
        actor_lr = float(actor_lr_base) * float(scale)
        scale_seed = int(args.rollout_seed) + 10_000 + int(scale_idx) * 1_000
        _set_all_seeds(scale_seed)
        cfg_scale, actor_scale, _critic_scale, learner_scale, freeze_info = _build_actor_critic_and_learner(
            cfg_template=cfg_template,
            actor_path=actor_path,
            critic_path=critic_path,
            device=device,
            hidden_dim=int(args.hidden_dim),
            embed_dim=int(args.embed_dim),
            actor_lr=float(actor_lr),
            critic_lr=float(critic_lr_base),
            ppo_epochs=int(args.ppo_epochs),
            num_mini_batch=int(args.num_mini_batch),
            value_coef=float(args.value_coef),
            entropy_coef=float(args.entropy_coef),
            freeze_mode=str(args.freeze_mode),
        )
        learner_scale.set_actor_advantage_override(2, branch_summary["advantages"])
        update_metrics = learner_scale.update(buffer, bootstrap_world_state)
        learner_scale.actor.eval()

        probe_context = build_bw_advantage_probe_context(
            learner_scale,
            buffer,
            bootstrap_world_state,
            sample_limit=max(int(args.probe_sample_limit), 1),
            sample_seed=int(args.probe_seed_base) + int(scale_idx) * 1_000,
        )
        if probe_context is None:
            raise RuntimeError("Failed to build BW probe context after actor update.")
        full_batch_metrics = _full_batch_bw_logprob_metrics(
            pre_actor=base_actor,
            post_actor=learner_scale.actor,
            probe_context=probe_context,
            device=device,
            raw_branch_advantages=np.asarray(branch_summary["raw_branch_advantages"], dtype=np.float32),
        )
        geometry_metrics = _full_batch_bw_geometry_metrics(
            pre_actor=base_actor,
            post_actor=learner_scale.actor,
            probe_context=probe_context,
        )
        local_eval = _estimate_local_distribution_gain(
            pre_actor=base_actor,
            post_actor=learner_scale.actor,
            snapshot_states=local_snapshot_states,
            cfg=cfg_scale,
            device=device,
            horizon=int(args.branch_horizon),
            action_samples=max(int(args.local_eval_action_samples), 1),
            seed_base=int(args.probe_seed_base) + 1_500_000 + int(scale_idx) * 5_000,
            follow_policy_mode=str(args.branch_follow_policy_mode),
        )
        probe_eval = evaluate_bw_advantage_alignment(
            pre_actor=base_actor,
            post_actor=learner_scale.actor,
            probe_context=probe_context,
            cfg=cfg_scale,
            device=device,
            k_steps=int(args.panel_k_steps),
            true_mc_enabled=max(int(args.probe_true_mc_samples), 0) > 0,
            true_mc_samples=max(int(args.probe_true_mc_samples), 0),
            true_mc_seed=int(args.probe_seed_base) + 500_000 + int(scale_idx) * 1_000,
            branch_enabled=True,
            branch_horizons=[int(args.branch_horizon)],
            branch_samples=max(int(args.branch_samples), 1),
            branch_seed=int(args.probe_seed_base) + 800_000 + int(scale_idx) * 1_000,
            branch_ref_mode="deterministic",
            branch_follow_policy_mode=str(args.branch_follow_policy_mode),
        )
        panel_post_stochastic = evaluate_bw_snapshot_panel(
            learner_scale.actor,
            panel,
            cfg_scale,
            device,
            k_steps=int(args.panel_k_steps),
            deterministic=False,
            actor_policy_samples=max(int(args.panel_actor_policy_samples), 1),
            seed_base=int(args.panel_eval_seed_base),
        )
        panel_post_deterministic = evaluate_bw_snapshot_panel(
            learner_scale.actor,
            panel,
            cfg_scale,
            device,
            k_steps=int(args.panel_k_steps),
            deterministic=True,
            actor_policy_samples=1,
            seed_base=None,
        )
        actor_scale_params = _flatten_module_params(actor_scale)
        param_delta = actor_scale_params - base_actor_params
        branch_primary = dict(probe_eval.get("branch_alignment", {}).get("primary", {}) or {})
        true_primary = dict(probe_eval.get("true_action_alignment", {}) or {})
        scale_result = {
            "scale": float(scale),
            "actor_lr": float(actor_lr),
            "freeze_info": {
                "freeze_mode": str(freeze_info.get("freeze_mode", "")),
                "trainable_param_count": int(len(freeze_info.get("trainable_param_names", []))),
                "trainable_param_names": list(freeze_info.get("trainable_param_names", [])),
            },
            "update_metrics": {
                "policy_loss": float(update_metrics.get("policy_loss", 0.0)),
                "value_loss_bw": float(update_metrics.get("value_loss_bw", 0.0)),
                "approx_kl_bw": float(update_metrics.get("approx_kl_bw", 0.0)),
                "clip_frac_bw": float(update_metrics.get("clip_frac_bw", 0.0)),
                "entropy_bw": float(update_metrics.get("entropy_bw", 0.0)),
            },
            "full_batch_metrics": {
                **full_batch_metrics,
                **geometry_metrics,
                "param_delta_l2": float(param_delta.norm().item()),
                "param_delta_linf": float(param_delta.abs().max().item()) if param_delta.numel() > 0 else 0.0,
            },
            "local_eval": local_eval,
            "panel_delta_stochastic": {
                "delta_actor_reward_mean": float(panel_post_stochastic["actor_reward"]["mean"] - panel_pre_stochastic["actor_reward"]["mean"]),
                "delta_actor_weighted_mean": float(panel_post_stochastic["actor_weighted"]["mean"] - panel_pre_stochastic["actor_weighted"]["mean"]),
                "delta_heuristic_gap_mean": float(
                    (panel_post_stochastic["heuristic_reward"]["mean"] - panel_post_stochastic["actor_reward"]["mean"])
                    - (panel_pre_stochastic["heuristic_reward"]["mean"] - panel_pre_stochastic["actor_reward"]["mean"])
                ),
                "delta_l1_to_heur_mean": float(panel_post_stochastic["l1_to_heur"]["mean"] - panel_pre_stochastic["l1_to_heur"]["mean"]),
                "actor_reward_improved_state_frac": float(
                    np.mean(
                        (
                            np.asarray(panel_post_stochastic["state_actor_reward_means"], dtype=np.float64)
                            - np.asarray(panel_pre_stochastic["state_actor_reward_means"], dtype=np.float64)
                        )
                        > 0.0
                    )
                )
                if panel_post_stochastic["state_actor_reward_means"] and panel_pre_stochastic["state_actor_reward_means"]
                else 0.0,
            },
            "panel_delta_deterministic": {
                "delta_actor_reward_mean": float(panel_post_deterministic["actor_reward"]["mean"] - panel_pre_deterministic["actor_reward"]["mean"]),
                "delta_actor_weighted_mean": float(panel_post_deterministic["actor_weighted"]["mean"] - panel_pre_deterministic["actor_weighted"]["mean"]),
                "delta_heuristic_gap_mean": float(
                    (panel_post_deterministic["heuristic_reward"]["mean"] - panel_post_deterministic["actor_reward"]["mean"])
                    - (panel_pre_deterministic["heuristic_reward"]["mean"] - panel_pre_deterministic["actor_reward"]["mean"])
                ),
                "delta_l1_to_heur_mean": float(panel_post_deterministic["l1_to_heur"]["mean"] - panel_pre_deterministic["l1_to_heur"]["mean"]),
                "actor_reward_improved_state_frac": float(
                    np.mean(
                        (
                            np.asarray(panel_post_deterministic["state_actor_reward_means"], dtype=np.float64)
                            - np.asarray(panel_pre_deterministic["state_actor_reward_means"], dtype=np.float64)
                        )
                        > 0.0
                    )
                )
                if panel_post_deterministic["state_actor_reward_means"] and panel_pre_deterministic["state_actor_reward_means"]
                else 0.0,
            },
            "probe_alignment": {
                "corr_advantage_vs_delta_logprob": float(probe_eval["logprob_alignment"]["corr_advantage_vs_delta_logprob"]),
                "mean_abs_delta_logprob": float(full_batch_metrics["mean_abs_delta_logprob"]),
                "corr_branch_delta_vs_delta_logprob": float(branch_primary.get("corr_branch_delta_vs_delta_logprob", 0.0)),
                "mean_branch_delta_times_delta_logprob": float(branch_primary.get("mean_branch_delta_times_delta_logprob", 0.0)),
                "corr_true_adv_mc_vs_delta_logprob": float(true_primary.get("corr_true_adv_mc_vs_delta_logprob", 0.0)),
                "mean_true_adv_mc_times_delta_logprob": float(true_primary.get("mean_true_adv_mc_times_delta_logprob", 0.0)),
                "branch_delta_abs_mean": float(branch_primary.get("branch_delta_abs_mean", 0.0)),
                "true_adv_mc_abs_mean": float(true_primary.get("true_adv_mc_abs_mean", 0.0)),
                "judgement": str(probe_eval["judgement"]),
            },
        }
        scale_results.append(scale_result)
        branch_product = float(scale_result["full_batch_metrics"]["mean_branch_delta_times_delta_logprob"])
        if best_branch_product is None or branch_product > best_branch_product:
            best_branch_product = branch_product
            best_scale_by_branch_product = float(scale)
        del learner_scale
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "actor_path": str(actor_path),
        "critic_path": str(critic_path),
        "device": str(device),
        "rollout_env_steps": int(args.rollout_env_steps),
        "rollout_seed": int(args.rollout_seed),
        "branch_horizon": int(args.branch_horizon),
        "branch_samples": int(args.branch_samples),
        "branch_follow_policy_mode": str(args.branch_follow_policy_mode),
        "branch_normalize": bool(args.branch_normalize),
        "panel_states": int(len(panel)),
        "panel_actor_policy_mode": str(args.panel_actor_policy_mode),
        "panel_actor_policy_samples": int(args.panel_actor_policy_samples),
        "freeze_mode": str(args.freeze_mode),
        "freeze_info": {
            "freeze_mode": str(base_freeze_info.get("freeze_mode", "")),
            "trainable_param_count": int(len(base_freeze_info.get("trainable_param_names", []))),
            "trainable_param_names": list(base_freeze_info.get("trainable_param_names", [])),
        },
        "clean_update": {
            "ppo_epochs": int(args.ppo_epochs),
            "num_mini_batch": int(args.num_mini_batch),
            "value_coef": float(args.value_coef),
            "entropy_coef": float(args.entropy_coef),
        },
        "teacher_summary": {
            "sample_count": int(branch_summary["sample_count"]),
            "branch_abs_mean": float(branch_summary["branch_abs_mean"]),
            "branch_std": float(branch_summary["branch_std"]),
            "corr_default_vs_branch": float(branch_summary["corr_default_vs_branch"]),
            "corr_raw_default_vs_branch": float(branch_summary["corr_raw_default_vs_branch"]),
            "sign_agree_default_vs_branch": float(branch_summary["sign_agree_default_vs_branch"]),
            "sign_agree_raw_default_vs_branch": float(branch_summary["sign_agree_raw_default_vs_branch"]),
        },
        "panel_pre_stochastic": {
            "actor_reward_mean": float(panel_pre_stochastic["actor_reward"]["mean"]),
            "actor_weighted_mean": float(panel_pre_stochastic["actor_weighted"]["mean"]),
            "heuristic_reward_mean": float(panel_pre_stochastic["heuristic_reward"]["mean"]),
            "heuristic_gap_mean": float(panel_pre_stochastic["heuristic_reward"]["mean"] - panel_pre_stochastic["actor_reward"]["mean"]),
        },
        "panel_pre_deterministic": {
            "actor_reward_mean": float(panel_pre_deterministic["actor_reward"]["mean"]),
            "actor_weighted_mean": float(panel_pre_deterministic["actor_weighted"]["mean"]),
            "heuristic_reward_mean": float(panel_pre_deterministic["heuristic_reward"]["mean"]),
            "heuristic_gap_mean": float(panel_pre_deterministic["heuristic_reward"]["mean"] - panel_pre_deterministic["actor_reward"]["mean"]),
        },
        "actor_lr_base": float(actor_lr_base),
        "actor_lr_scales": [float(x) for x in actor_lr_scales],
        "best_scale_by_branch_product": best_scale_by_branch_product,
        "scale_results": scale_results,
    }
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
