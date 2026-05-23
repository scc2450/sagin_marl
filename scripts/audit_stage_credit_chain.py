from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
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
from sagin_marl.rl.structured_mappo import StructuredMAPPO, _index_dataclass
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver_group

from scripts.audit_stage_ppo_credit_alignment import (
    STAGE_ID,
    _force_single_stage_config,
    _make_stage_optimizers,
    _normalize_advantages_like_update,
    _stage_action_samples,
    _stage_optimizer_params,
)
from scripts.audit_stage_qpi_action_credit import (
    _branch_returns_qpi,
    _qpi_policy_gradient_diagnostics,
)
from scripts.diagnose_reward_action_sensitivity import _corr, _summ


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _standardize_np(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    std = float(np.std(arr))
    if std <= 1.0e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - float(np.mean(arr))) / std


def _row_agent_indices(*, rows: int, actions: int, num_agents: int, device: torch.device) -> torch.Tensor:
    row_base = torch.arange(int(rows), dtype=torch.long, device=device).view(int(rows), 1, 1) * int(num_agents)
    agent = torch.arange(int(num_agents), dtype=torch.long, device=device).view(1, 1, int(num_agents))
    return (row_base + agent).expand(int(rows), int(actions), int(num_agents)).reshape(-1)


def _eval_candidate_logprob(
    learner: StructuredMAPPO,
    *,
    stage_id: int,
    selected_local: Any,
    candidate_actions: torch.Tensor,
    num_agents: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = int(candidate_actions.shape[0])
    actions = int(candidate_actions.shape[1])
    local_flat = _index_dataclass(
        selected_local,
        _row_agent_indices(rows=rows, actions=actions, num_agents=int(num_agents), device=learner.device),
    )
    flat_actions = candidate_actions.reshape(rows * actions, int(num_agents), -1)
    logprob, entropy, _out = learner._stage_actor_eval_from_batch(
        int(stage_id),
        local_flat,
        flat_actions,
        int(num_agents),
    )
    return logprob.reshape(rows, actions), entropy.reshape(rows, actions)


def _candidate_shift_summary(*, q_adv: np.ndarray, delta_logprob: np.ndarray) -> dict[str, Any]:
    q = np.asarray(q_adv, dtype=np.float64).reshape(-1)
    d = np.asarray(delta_logprob, dtype=np.float64).reshape(-1)
    pos = q > 0.0
    neg = q < 0.0
    return {
        "corr_qadv_delta_logprob": _corr(q, d),
        "mean_qadv_times_delta_logprob": float(np.mean(q * d)) if q.size else 0.0,
        "mean_delta_logprob_q_positive": float(np.mean(d[pos])) if np.any(pos) else 0.0,
        "mean_delta_logprob_q_negative": float(np.mean(d[neg])) if np.any(neg) else 0.0,
        "q_positive_frac": float(np.mean(pos)) if q.size else 0.0,
        "delta_logprob": _summ(d),
    }


def _per_state_shift_summary(*, q_mean: np.ndarray, q_adv: np.ndarray, delta_logprob: np.ndarray) -> dict[str, Any]:
    q = np.asarray(q_adv, dtype=np.float64)
    q_abs = np.asarray(q_mean, dtype=np.float64)
    d = np.asarray(delta_logprob, dtype=np.float64)
    if q.ndim != 2 or d.shape != q.shape or q_abs.shape != q.shape or q.shape[0] <= 0:
        return {
            "per_state_q_weighted_shift_positive_frac": 0.0,
            "best_candidate_logprob_up_frac": 0.0,
            "top_minus_bottom_margin_up_frac": 0.0,
        }
    weighted = np.sum(q * d, axis=1)
    best_idx = np.argmax(q_abs, axis=1)
    worst_idx = np.argmin(q_abs, axis=1)
    row_idx = np.arange(q.shape[0])
    best_delta = d[row_idx, best_idx]
    worst_delta = d[row_idx, worst_idx]
    margin_delta = best_delta - worst_delta
    return {
        "per_state_q_weighted_shift": _summ(weighted),
        "per_state_q_weighted_shift_positive_frac": float(np.mean(weighted > 0.0)),
        "best_candidate_delta_logprob": _summ(best_delta),
        "best_candidate_logprob_up_frac": float(np.mean(best_delta > 0.0)),
        "worst_candidate_delta_logprob": _summ(worst_delta),
        "top_minus_bottom_margin_delta": _summ(margin_delta),
        "top_minus_bottom_margin_up_frac": float(np.mean(margin_delta > 0.0)),
    }


def _probe_actor_update(
    learner: StructuredMAPPO,
    *,
    stage_id: int,
    selected_local: Any,
    candidate_actions: torch.Tensor,
    selected_advantage: np.ndarray | None,
    candidate_advantage: np.ndarray | None,
    num_agents: int,
    lr: float,
    optimizer_name: str,
    entropy_coef: float,
    clip_ratio: float,
) -> dict[str, Any]:
    params = _stage_optimizer_params(learner.actor, int(stage_id))
    if not params:
        return {"available": False}

    actor_state = copy.deepcopy(learner.actor.state_dict())
    try:
        with torch.no_grad():
            pre_all, _pre_ent = _eval_candidate_logprob(
                learner,
                stage_id=int(stage_id),
                selected_local=selected_local,
                candidate_actions=candidate_actions,
                num_agents=int(num_agents),
            )
            pre_selected = pre_all[:, 0].detach()

        if str(optimizer_name).lower() == "sgd":
            opt: torch.optim.Optimizer = torch.optim.SGD(params, lr=float(lr))
        else:
            opt = torch.optim.Adam(params, lr=float(lr))

        if candidate_advantage is not None:
            rows = int(candidate_actions.shape[0])
            acts = int(candidate_actions.shape[1])
            local_flat = _index_dataclass(
                selected_local,
                _row_agent_indices(rows=rows, actions=acts, num_agents=int(num_agents), device=learner.device),
            )
            flat_actions = candidate_actions.reshape(rows * acts, int(num_agents), -1)
            logprob, entropy, _out = learner._stage_actor_eval_from_batch(
                int(stage_id),
                local_flat,
                flat_actions,
                int(num_agents),
            )
            adv = torch.as_tensor(
                np.asarray(candidate_advantage, dtype=np.float32).reshape(rows * acts),
                dtype=logprob.dtype,
                device=learner.device,
            )
            policy_loss = -(adv.detach() * logprob).mean()
            entropy_mean = entropy.mean() if int(entropy.numel()) > 0 else torch.zeros((), device=learner.device)
            loss = policy_loss - float(entropy_coef) * entropy_mean
        elif selected_advantage is not None:
            selected_actions = candidate_actions[:, 0:1]
            logprob_mat, entropy_mat = _eval_candidate_logprob(
                learner,
                stage_id=int(stage_id),
                selected_local=selected_local,
                candidate_actions=selected_actions,
                num_agents=int(num_agents),
            )
            logprob = logprob_mat[:, 0]
            entropy = entropy_mat[:, 0]
            adv = torch.as_tensor(
                np.asarray(selected_advantage, dtype=np.float32).reshape(-1),
                dtype=logprob.dtype,
                device=learner.device,
            )
            old_lp = pre_selected.to(device=learner.device, dtype=logprob.dtype)
            ratio = torch.exp(torch.clamp(logprob - old_lp.detach(), min=-20.0, max=20.0))
            clipped = torch.clamp(ratio, 1.0 - float(clip_ratio), 1.0 + float(clip_ratio))
            policy_loss = -torch.minimum(ratio * adv.detach(), clipped * adv.detach()).mean()
            entropy_mean = entropy.mean() if int(entropy.numel()) > 0 else torch.zeros((), device=learner.device)
            loss = policy_loss - float(entropy_coef) * entropy_mean
        else:
            raise ValueError("selected_advantage or candidate_advantage is required.")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, float(learner.max_grad_norm))
        opt.step()

        with torch.no_grad():
            post_all, post_ent = _eval_candidate_logprob(
                learner,
                stage_id=int(stage_id),
                selected_local=selected_local,
                candidate_actions=candidate_actions,
                num_agents=int(num_agents),
            )
        delta = (post_all - pre_all).detach().cpu().numpy().astype(np.float64)
        return {
            "available": True,
            "optimizer": str(optimizer_name),
            "lr": float(lr),
            "loss": float(loss.detach().cpu().item()),
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "entropy": float(entropy_mean.detach().cpu().item()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu().item()),
            "candidate_delta_logprob": delta,
            "post_entropy": _summ(post_ent.detach().cpu().numpy()),
        }
    finally:
        learner.actor.load_state_dict(actor_state, strict=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(STAGE_ID), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reward_mode", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--rollout_env_steps", type=int, default=250)
    parser.add_argument("--sample_rows", type=int, default=8)
    parser.add_argument("--policy_action_samples", type=int, default=4)
    parser.add_argument("--continuations", type=int, default=4)
    parser.add_argument("--follow", choices=["deterministic", "stochastic"], default="stochastic")
    parser.add_argument("--min_horizon", type=int, default=20)
    parser.add_argument("--branch_horizon_cap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=45210)
    parser.add_argument("--reset_seed_base", type=int, default=None)
    parser.add_argument(
        "--pre_collect_rollouts",
        type=int,
        default=0,
        help="Optional rollout count to collect before the audited rollout, mirroring critic-only train-bank collection.",
    )
    parser.add_argument(
        "--selected_stage_samples",
        default="",
        help="Optional comma-separated stage_batch sample indices. Overrides random row sampling.",
    )
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--probe_lr", type=float, default=None)
    parser.add_argument("--probe_optimizer", choices=["adam", "sgd"], default="adam")
    args = parser.parse_args()

    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    _set_seed(int(args.seed))

    cfg = load_config(args.config)
    stage_id = STAGE_ID[str(args.stage)]
    _force_single_stage_config(cfg, stage_id=stage_id, reward_mode=args.reward_mode)
    bundle = build_structured_modules_from_config(cfg)
    actor = bundle.actor.to(device)
    critic = bundle.critic.to(device) if bundle.critic is not None else None
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
        actor_stage_optimizers=_make_stage_optimizers(actor, float(cfg.actor_lr)),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=float(cfg.critic_lr)) if critic is not None else None,
        device=device,
        cfg=cfg,
        train_accel=bool(stage_id == 0),
        train_sat=bool(stage_id == 1),
        train_bw=bool(stage_id == 2),
        exec_accel_source=str(cfg.exec_accel_source),
        exec_sat_source=str(cfg.exec_sat_source),
        exec_bw_source=str(cfg.exec_bw_source),
    )
    group = make_structured_driver_group(cfg, num_envs=int(args.num_envs), backend="sync", mode="train")
    try:
        learner.bind_native_runtime_contract(group)
        learner.begin_native_rollout(group, rollout_env_steps=int(args.rollout_env_steps), num_envs=int(args.num_envs))
        pre_collect = max(int(args.pre_collect_rollouts), 0)
        for ridx in range(pre_collect):
            reset_many = getattr(group, "reset_many", None)
            if callable(reset_many):
                reset_many([int(args.seed) + ridx * 10_000 + env for env in range(int(args.num_envs))])
            pre_buffer = StructuredRolloutBuffer()
            learner.collect_env_horizon_native_tensor_policy(
                group,
                buffer=pre_buffer,
                horizon=int(args.rollout_env_steps),
                deterministic=False,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            # Build views as critic-only does, so the runtime/history binding path is exercised.
            _ = pre_buffer.build_rollout_views(device)
        if hasattr(group, "reset_many"):
            reset_seed_base = int(args.seed) if args.reset_seed_base is None else int(args.reset_seed_base)
            group.reset_many([reset_seed_base + i for i in range(int(args.num_envs))])
        buffer = StructuredRolloutBuffer()
        learner.collect_env_horizon_native_tensor_policy(
            group,
            buffer=buffer,
            horizon=int(args.rollout_env_steps),
            deterministic=False,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        views = buffer.build_rollout_views(device)
        batch_view = views.training_view
        return_view = views.return_view
        learner._refresh_actor_old_logprobs_from_training_view(batch_view)
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
        adv_raw = torch.as_tensor(gae["advantages"], dtype=torch.float32, device=device)
        adv_norm = _normalize_advantages_like_update(
            cfg,
            advantages_raw=adv_raw,
            stage_ids=np.asarray(batch_view.stage_ids, dtype=np.int64),
        )
        returns = torch.as_tensor(gae["returns"], dtype=torch.float32, device=device)
        values = value_override.to(device=device, dtype=torch.float32)

        stage_batch = batch_view.stage_batches[int(stage_id)]
        num_agents = int(stage_batch.num_agents)
        transition_indices = np.asarray(stage_batch.transition_indices, dtype=np.int64)
        history_rows_all = ((transition_indices - int(stage_id)) // 3).astype(np.int64)
        step_indices_all = history_rows_all // max(int(args.num_envs), 1)
        horizons_all = np.maximum(int(cfg.T_steps) - step_indices_all, 1).astype(np.int64)
        eligible = np.flatnonzero(horizons_all >= max(int(args.min_horizon), 1))
        if eligible.size <= 0:
            raise RuntimeError("no eligible rows after min_horizon filtering.")
        selected_override = str(args.selected_stage_samples or "").strip()
        if selected_override:
            requested = np.asarray(
                [int(x.strip()) for x in selected_override.split(",") if x.strip()],
                dtype=np.int64,
            )
            eligible_set = set(int(x) for x in eligible.tolist())
            selected_np = np.asarray([int(x) for x in requested.tolist() if int(x) in eligible_set], dtype=np.int64)
            if selected_np.size <= 0:
                raise RuntimeError(
                    "--selected_stage_samples did not contain any eligible stage samples "
                    f"(min_horizon={int(args.min_horizon)})."
                )
            selected_np = np.unique(selected_np)
            sample_count = int(selected_np.size)
        else:
            rng = np.random.default_rng(int(args.seed) + 37)
            sample_count = min(max(int(args.sample_rows), 1), int(eligible.size))
            selected_np = np.sort(rng.choice(eligible, size=sample_count, replace=False))
        selected_t = torch.as_tensor(selected_np, dtype=torch.long, device=device)
        selected_transitions = transition_indices[selected_np]
        selected_global_t = torch.as_tensor(selected_transitions, dtype=torch.long, device=device)
        flat_rows = (
            selected_t.view(-1, 1)
            * int(num_agents)
            + torch.arange(num_agents, dtype=torch.long, device=device).view(1, int(num_agents))
        ).reshape(-1)
        selected_local = _index_dataclass(stage_batch.local_batch, flat_rows)
        sampled_actions = stage_batch.actions.index_select(0, selected_t).to(device=device)
        labels, action_tensor, eval_actions = _stage_action_samples(
            actor,
            stage_id=int(stage_id),
            selected_local=selected_local,
            rollout_actions=sampled_actions,
            sample_count=sample_count,
            num_agents=int(num_agents),
            policy_samples=int(args.policy_action_samples),
            device=device,
        )
        action_count = int(action_tensor.shape[1])
        policy_actions = action_tensor[:, 1:].contiguous()
        policy_labels = labels[1:]
        policy_action_count = int(policy_actions.shape[1])

        history_rows = history_rows_all[selected_np]
        horizons = horizons_all[selected_np]
        if int(args.branch_horizon_cap) > 0:
            horizons = np.minimum(horizons, int(args.branch_horizon_cap)).astype(np.int64, copy=False)
        continuations = max(int(args.continuations), 1)
        branch_rows = np.repeat(history_rows, action_count * continuations)
        branch_horizons = np.repeat(horizons, action_count * continuations)
        branch_actions = (
            action_tensor[:, :, None]
            .expand(sample_count, action_count, continuations, *action_tensor.shape[2:])
            .reshape(sample_count * action_count * continuations, int(num_agents), -1)
            .contiguous()
        )
        reward_mode = str(args.reward_mode or getattr(cfg, "reward_mode", "env_reward") or "env_reward")
        returns_by_mode = _branch_returns_qpi(
            learner,
            stage_id=int(stage_id),
            history_rows=branch_rows.tolist(),
            first_actions=branch_actions,
            horizons=branch_horizons.tolist(),
            reward_modes=[reward_mode],
            follow_deterministic=(str(args.follow) == "deterministic"),
        )
        ret_cube = (
            returns_by_mode[reward_mode]
            .detach()
            .cpu()
            .numpy()
            .reshape(sample_count, action_count, continuations)
            .astype(np.float64)
        )
        q_mean = ret_cube.mean(axis=2)
        q_std = ret_cube.std(axis=2)
        q_policy_mean = q_mean[:, 1:]
        q_adv = q_policy_mean - q_policy_mean.mean(axis=1, keepdims=True)
        q_adv_norm_all = _standardize_np(q_adv.reshape(-1)).reshape(q_adv.shape)
        q_rollout_adv = q_adv[:, 0]
        q_rollout_adv_norm = _standardize_np(q_rollout_adv)

        selected_adv_norm = adv_norm.index_select(0, selected_global_t).detach().cpu().numpy().astype(np.float64)
        selected_adv_raw = adv_raw.index_select(0, selected_global_t).detach().cpu().numpy().astype(np.float64)
        selected_returns = returns.index_select(0, selected_global_t).detach().cpu().numpy().astype(np.float64)
        selected_values = values.index_select(0, selected_global_t).detach().cpu().numpy().astype(np.float64)

        with torch.no_grad():
            pre_policy_logprob, _pre_policy_entropy = _eval_candidate_logprob(
                learner,
                stage_id=int(stage_id),
                selected_local=selected_local,
                candidate_actions=policy_actions,
                num_agents=int(num_agents),
            )
        pre_policy_logprob_np = pre_policy_logprob.detach().cpu().numpy().astype(np.float64)

        probe_lr = float(args.probe_lr) if args.probe_lr is not None else float(cfg.actor_lr)
        ppo_probe = _probe_actor_update(
            learner,
            stage_id=int(stage_id),
            selected_local=selected_local,
            candidate_actions=policy_actions,
            selected_advantage=selected_adv_norm,
            candidate_advantage=None,
            num_agents=int(num_agents),
            lr=probe_lr,
            optimizer_name=str(args.probe_optimizer),
            entropy_coef=float(learner.entropy_coef_by_stage[int(stage_id)]),
            clip_ratio=float(learner.clip_ratio),
        )
        oracle_selected_probe = _probe_actor_update(
            learner,
            stage_id=int(stage_id),
            selected_local=selected_local,
            candidate_actions=policy_actions,
            selected_advantage=q_rollout_adv_norm,
            candidate_advantage=None,
            num_agents=int(num_agents),
            lr=probe_lr,
            optimizer_name=str(args.probe_optimizer),
            entropy_coef=0.0,
            clip_ratio=float(learner.clip_ratio),
        )
        oracle_all_probe = _probe_actor_update(
            learner,
            stage_id=int(stage_id),
            selected_local=selected_local,
            candidate_actions=policy_actions,
            selected_advantage=None,
            candidate_advantage=q_adv_norm_all,
            num_agents=int(num_agents),
            lr=probe_lr,
            optimizer_name=str(args.probe_optimizer),
            entropy_coef=0.0,
            clip_ratio=float(learner.clip_ratio),
        )

        def _strip_probe(probe: dict[str, Any]) -> dict[str, Any]:
            out = {k: v for k, v in probe.items() if k != "candidate_delta_logprob"}
            delta = np.asarray(probe.get("candidate_delta_logprob", np.zeros_like(q_adv)), dtype=np.float64)
            out["candidate_shift_vs_qpi"] = _candidate_shift_summary(q_adv=q_adv, delta_logprob=delta)
            out["per_state_shift_vs_qpi"] = _per_state_shift_summary(
                q_mean=q_policy_mean,
                q_adv=q_adv,
                delta_logprob=delta,
            )
            out["rollout_action_shift_vs_adv"] = {
                "corr_rollout_qadv_delta": _corr(q_rollout_adv, delta[:, 0] if delta.ndim == 2 and delta.shape[1] else delta.reshape(-1)),
                "corr_ppo_adv_delta": _corr(selected_adv_norm, delta[:, 0] if delta.ndim == 2 and delta.shape[1] else delta.reshape(-1)),
                "mean_delta_rollout_q_positive": float(np.mean(delta[:, 0][q_rollout_adv > 0.0]))
                if delta.ndim == 2 and np.any(q_rollout_adv > 0.0)
                else 0.0,
                "mean_delta_rollout_q_negative": float(np.mean(delta[:, 0][q_rollout_adv < 0.0]))
                if delta.ndim == 2 and np.any(q_rollout_adv < 0.0)
                else 0.0,
            }
            return out

        action_effect_var = np.var(q_policy_mean, axis=1)
        continuation_var = np.mean(np.var(ret_cube[:, 1:, :], axis=2), axis=1)
        q_est_se = np.mean(q_std[:, 1:] / math.sqrt(max(continuations, 1)), axis=1)
        q_gap = np.max(q_policy_mean, axis=1) - np.min(q_policy_mean, axis=1)
        q_gap_reliable = q_gap > (2.0 * q_est_se)
        qpi_grad = _qpi_policy_gradient_diagnostics(
            learner,
            stage_id=int(stage_id),
            selected_local=selected_local,
            action_tensor=action_tensor,
            q_adv=q_adv,
            num_agents=int(num_agents),
        )

        payload = {
            "config": str(args.config),
            "stage": str(args.stage),
            "stage_id": int(stage_id),
            "reward_mode": reward_mode,
            "follow": str(args.follow),
            "num_envs": int(args.num_envs),
            "rollout_env_steps": int(args.rollout_env_steps),
            "sample_rows": int(sample_count),
            "policy_action_samples": int(args.policy_action_samples),
            "continuations": int(continuations),
            "action_labels": labels,
            "policy_action_labels": policy_labels,
            "probe_optimizer": str(args.probe_optimizer),
            "probe_lr": probe_lr,
            "chain": {
                "step1_qpi_credit": {
                    "q_mean": _summ(q_mean),
                    "q_std_per_action": _summ(q_std),
                    "q_policy_adv": _summ(q_adv),
                    "q_rollout_adv": _summ(q_rollout_adv),
                    "q_action_effect_std": float(math.sqrt(max(float(np.mean(action_effect_var)), 0.0))),
                    "single_rollout_continuation_std": float(
                        math.sqrt(max(float(np.mean(continuation_var)), 0.0))
                    ),
                    "q_gap": _summ(q_gap),
                    "q_estimation_se": _summ(q_est_se),
                    "q_gap_gt_2se_frac": float(np.mean(q_gap_reliable)),
                    "best_minus_ref": _summ(np.max(q_mean, axis=1) - q_mean[:, 0]),
                    "rollout_minus_ref": _summ(q_mean[:, 1] - q_mean[:, 0]),
                },
                "step2_advantage_readout": {
                    "corr_ppo_norm_adv_vs_qpi_rollout_adv": _corr(selected_adv_norm, q_rollout_adv),
                    "corr_ppo_raw_adv_vs_qpi_rollout_adv": _corr(selected_adv_raw, q_rollout_adv),
                    "sign_agree_ppo_norm_vs_qpi_rollout_adv": float(
                        np.mean(np.sign(selected_adv_norm) == np.sign(q_rollout_adv))
                    ),
                    "sign_agree_ppo_raw_vs_qpi_rollout_adv": float(
                        np.mean(np.sign(selected_adv_raw) == np.sign(q_rollout_adv))
                    ),
                    "ppo_norm_adv": _summ(selected_adv_norm),
                    "ppo_raw_adv": _summ(selected_adv_raw),
                    "return": _summ(selected_returns),
                    "value": _summ(selected_values),
                    "residual": _summ(selected_returns - selected_values),
                },
                "step3_actor_loss_update": {
                    "ppo_selected_adv_update": _strip_probe(ppo_probe),
                    "qpi_selected_adv_update": _strip_probe(oracle_selected_probe),
                    "qpi_all_candidate_update": _strip_probe(oracle_all_probe),
                    "pre_policy_logprob": _summ(pre_policy_logprob_np),
                },
                "qpi_policy_gradient": qpi_grad,
            },
            "rows": [
                {
                    "stage_sample": int(selected_np[i]),
                    "transition_index": int(selected_transitions[i]),
                    "history_row": int(history_rows[i]),
                    "step": int(step_indices_all[selected_np][i]),
                    "horizon": int(horizons[i]),
                    "ppo_adv_norm": float(selected_adv_norm[i]),
                    "ppo_adv_raw": float(selected_adv_raw[i]),
                    "return": float(selected_returns[i]),
                    "value": float(selected_values[i]),
                    "q_rollout_adv": float(q_rollout_adv[i]),
                    "q_mean": [float(x) for x in q_mean[i].tolist()],
                    "q_std": [float(x) for x in q_std[i].tolist()],
                }
                for i in range(sample_count)
            ],
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "stage": str(args.stage),
                    "reward_mode": reward_mode,
                    "follow": str(args.follow),
                    "sample_rows": int(sample_count),
                    "policy_action_samples": int(args.policy_action_samples),
                    "continuations": int(continuations),
                    "q_gap_gt_2se_frac": payload["chain"]["step1_qpi_credit"]["q_gap_gt_2se_frac"],
                    "corr_ppo_norm_adv_vs_qpi_rollout_adv": payload["chain"]["step2_advantage_readout"][
                        "corr_ppo_norm_adv_vs_qpi_rollout_adv"
                    ],
                    "ppo_update_q_shift": payload["chain"]["step3_actor_loss_update"]["ppo_selected_adv_update"][
                        "candidate_shift_vs_qpi"
                    ]["mean_qadv_times_delta_logprob"],
                    "qpi_selected_update_q_shift": payload["chain"]["step3_actor_loss_update"][
                        "qpi_selected_adv_update"
                    ]["candidate_shift_vs_qpi"]["mean_qadv_times_delta_logprob"],
                    "qpi_all_update_q_shift": payload["chain"]["step3_actor_loss_update"][
                        "qpi_all_candidate_update"
                    ]["candidate_shift_vs_qpi"]["mean_qadv_times_delta_logprob"],
                    "out": str(out_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        close_structured_env_group(group)


if __name__ == "__main__":
    main()
