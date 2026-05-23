from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_mappo import _index_dataclass
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver_group

from scripts.audit_stage_critic_only_fit import (
    _clone_dataclass_tensors,
    _compute_returns_for_views,
    _collect_one_rollout,
    _collect_stage_bank,
    _eval_critic,
    _explained_variance_np,
    _heldout_vpi_ceiling_probe,
    _make_learner,
    _train_critic_only,
)
from scripts.audit_stage_credit_chain import (
    _candidate_shift_summary,
    _eval_candidate_logprob,
    _per_state_shift_summary,
    _row_agent_indices,
    _standardize_np,
)
from scripts.audit_stage_ppo_credit_alignment import (
    STAGE_ID,
    _force_single_stage_config,
    _normalize_advantages_like_update,
    _stage_action_samples,
    _stage_optimizer_params,
)
from scripts.audit_stage_qpi_action_credit import _branch_returns_qpi
from scripts.diagnose_reward_action_sensitivity import _corr, _summ


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _parse_epoch_checkpoints(text: str) -> list[int]:
    out: set[int] = {0}
    for piece in str(text).replace(";", ",").split(","):
        piece = piece.strip()
        if piece:
            out.add(max(int(piece), 0))
    return sorted(out)


def _optional_bool_arg(value: str | None) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled", "enable"}:
        return True
    if text in {"0", "false", "no", "off", "disabled", "disable"}:
        return False
    raise ValueError(f"cannot parse boolean argument value {value!r}")


def _stage_sample_metadata(*, cfg: Any, stage_id: int, num_envs: int, stage_batch: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transition_indices = np.asarray(stage_batch.transition_indices, dtype=np.int64)
    history_rows = ((transition_indices - int(stage_id)) // 3).astype(np.int64)
    step_indices = history_rows // max(int(num_envs), 1)
    horizons = np.maximum(int(cfg.T_steps) - step_indices, 1).astype(np.int64)
    return history_rows, step_indices, horizons


def _apply_qpi_all_candidate_actor_update(
    learner: Any,
    *,
    cfg: Any,
    stage_id: int,
    num_envs: int,
    views: Any,
    reward_mode: str,
    sample_rows: int,
    policy_action_samples: int,
    continuations: int,
    follow_deterministic: bool,
    min_horizon: int,
    branch_horizon_cap: int,
    seed: int,
    lr: float,
    optimizer_name: str,
    future_random_mode: str = "resample",
) -> dict[str, Any]:
    stage_batch = views.training_view.stage_batches[int(stage_id)]
    num_agents = int(stage_batch.num_agents)
    history_rows_all, step_indices_all, horizons_all = _stage_sample_metadata(
        cfg=cfg,
        stage_id=int(stage_id),
        num_envs=int(num_envs),
        stage_batch=stage_batch,
    )
    eligible = np.flatnonzero(horizons_all >= max(int(min_horizon), 1))
    if eligible.size <= 0:
        return {"enabled": False, "reason": "no eligible rows"}
    rng = np.random.default_rng(int(seed) + 41)
    sample_count = min(max(int(sample_rows), 1), int(eligible.size))
    selected_np = np.sort(rng.choice(eligible, size=sample_count, replace=False))
    selected_t = torch.as_tensor(selected_np, dtype=torch.long, device=learner.device)
    flat_rows = (
        selected_t.view(-1, 1)
        * int(num_agents)
        + torch.arange(num_agents, dtype=torch.long, device=learner.device).view(1, int(num_agents))
    ).reshape(-1)
    selected_local = _index_dataclass(stage_batch.local_batch, flat_rows)
    rollout_actions = stage_batch.actions.index_select(0, selected_t).to(device=learner.device)
    labels, action_tensor, _eval_actions = _stage_action_samples(
        learner.actor,
        stage_id=int(stage_id),
        selected_local=selected_local,
        rollout_actions=rollout_actions,
        sample_count=sample_count,
        num_agents=num_agents,
        policy_samples=int(policy_action_samples),
        device=learner.device,
    )
    policy_action_tensor = action_tensor[:, 1:].contiguous()
    policy_labels = labels[1:]
    policy_action_count = int(policy_action_tensor.shape[1])
    continuations_i = max(int(continuations), 1)
    history_rows = history_rows_all[selected_np]
    horizons = horizons_all[selected_np]
    if int(branch_horizon_cap) > 0:
        horizons = np.minimum(horizons, int(branch_horizon_cap)).astype(np.int64, copy=False)
    branch_rows = np.repeat(history_rows, policy_action_count * continuations_i)
    branch_horizons = np.repeat(horizons, policy_action_count * continuations_i)
    branch_actions = (
        policy_action_tensor[:, :, None]
        .expand(sample_count, policy_action_count, continuations_i, *policy_action_tensor.shape[2:])
        .reshape(sample_count * policy_action_count * continuations_i, int(num_agents), -1)
        .contiguous()
    )
    returns_by_mode = _branch_returns_qpi(
        learner,
        stage_id=int(stage_id),
        history_rows=branch_rows.tolist(),
        first_actions=branch_actions,
        horizons=branch_horizons.tolist(),
        reward_modes=[str(reward_mode)],
        follow_deterministic=bool(follow_deterministic),
        future_random_mode=str(future_random_mode),
        future_random_seed=int(seed) + 991_001,
    )
    ret_cube = (
        returns_by_mode[str(reward_mode)]
        .detach()
        .cpu()
        .numpy()
        .reshape(sample_count, policy_action_count, continuations_i)
        .astype(np.float64)
    )
    q_mean = ret_cube.mean(axis=2)
    q_adv = q_mean - q_mean.mean(axis=1, keepdims=True)
    q_adv_norm = _standardize_np(q_adv.reshape(-1)).reshape(q_adv.shape)

    params = _stage_optimizer_params(learner.actor, int(stage_id))
    if not params:
        return {"enabled": False, "reason": "no trainable stage actor params"}
    with torch.no_grad():
        pre_logprob, pre_entropy = _eval_candidate_logprob(
            learner,
            stage_id=int(stage_id),
            selected_local=selected_local,
            candidate_actions=policy_action_tensor,
            num_agents=int(num_agents),
        )
    if str(optimizer_name).lower() == "sgd":
        opt: torch.optim.Optimizer = torch.optim.SGD(params, lr=float(lr))
    else:
        opt = torch.optim.Adam(params, lr=float(lr))
    rows = int(policy_action_tensor.shape[0])
    acts = int(policy_action_tensor.shape[1])
    local_flat = _index_dataclass(
        selected_local,
        _row_agent_indices(rows=rows, actions=acts, num_agents=int(num_agents), device=learner.device),
    )
    flat_actions = policy_action_tensor.reshape(rows * acts, int(num_agents), -1)
    logprob, entropy, _out = learner._stage_actor_eval_from_batch(
        int(stage_id),
        local_flat,
        flat_actions,
        int(num_agents),
    )
    adv_t = torch.as_tensor(
        q_adv_norm.reshape(rows * acts),
        dtype=logprob.dtype,
        device=learner.device,
    )
    policy_loss = -(adv_t.detach() * logprob).mean()
    entropy_mean = entropy.mean() if int(entropy.numel()) > 0 else torch.zeros((), device=learner.device)
    loss = policy_loss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(params, float(learner.max_grad_norm))
    opt.step()
    with torch.no_grad():
        post_logprob, post_entropy = _eval_candidate_logprob(
            learner,
            stage_id=int(stage_id),
            selected_local=selected_local,
            candidate_actions=policy_action_tensor,
            num_agents=int(num_agents),
        )
    delta = (post_logprob - pre_logprob).detach().cpu().numpy().astype(np.float64)
    return {
        "enabled": True,
        "sample_rows": int(sample_count),
        "policy_action_samples": int(policy_action_count),
        "continuations": int(continuations_i),
        "follow": "deterministic" if bool(follow_deterministic) else "stochastic",
        "future_random_mode": str(future_random_mode),
        "action_labels": policy_labels,
        "lr": float(lr),
        "optimizer": str(optimizer_name),
        "loss": float(loss.detach().cpu().item()),
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "entropy_pre": _summ(pre_entropy.detach().cpu().numpy()),
        "entropy_post": _summ(post_entropy.detach().cpu().numpy()),
        "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu().item()),
        "q_mean": _summ(q_mean),
        "q_adv": _summ(q_adv),
        "return_cube": _summ(ret_cube),
        "candidate_shift_vs_qpi": _candidate_shift_summary(q_adv=q_adv, delta_logprob=delta),
        "per_state_shift_vs_qpi": _per_state_shift_summary(q_mean=q_mean, q_adv=q_adv, delta_logprob=delta),
        "rows": [
            {
                "stage_sample": int(selected_np[i]),
                "history_row": int(history_rows[i]),
                "step": int(step_indices_all[selected_np][i]),
                "horizon": int(horizons[i]),
                "q_mean": [float(x) for x in q_mean[i].tolist()],
            }
            for i in range(sample_count)
        ],
    }


def _apply_selected_advantage_actor_update(
    learner: Any,
    *,
    cfg: Any,
    stage_id: int,
    num_envs: int,
    buffer: Any,
    views: Any,
    reward_mode: str,
    advantage_kind: str,
    sample_rows: int,
    policy_action_samples: int,
    continuations: int,
    follow_deterministic: bool,
    min_horizon: int,
    branch_horizon_cap: int,
    seed: int,
    lr: float,
    optimizer_name: str,
    full_ppo: bool,
    ppo_epochs: int,
    num_minibatches: int,
    future_random_mode: str = "resample",
    skip_qpi_probe: bool = False,
) -> dict[str, Any]:
    stage_batch = views.training_view.stage_batches[int(stage_id)]
    num_agents = int(stage_batch.num_agents)
    history_rows_all, step_indices_all, horizons_all = _stage_sample_metadata(
        cfg=cfg,
        stage_id=int(stage_id),
        num_envs=int(num_envs),
        stage_batch=stage_batch,
    )
    eligible = np.flatnonzero(horizons_all >= max(int(min_horizon), 1))
    if eligible.size <= 0:
        return {"enabled": False, "reason": "no eligible rows"}
    if str(advantage_kind).lower() == "gae":
        _returns, advantages_raw, _values = _compute_returns_for_views(
            learner,
            buffer,
            views,
            target="train_gae",
            device=learner.device,
        )
    elif str(advantage_kind).lower() == "mc":
        returns_mc, _advantages_unused, values = _compute_returns_for_views(
            learner,
            buffer,
            views,
            target="mc",
            device=learner.device,
        )
        advantages_raw = returns_mc - values
    else:
        raise ValueError("advantage_kind must be one of {'gae', 'mc'}.")
    advantages_norm = _normalize_advantages_like_update(
        cfg,
        advantages_raw=advantages_raw,
        stage_ids=np.asarray(views.training_view.stage_ids, dtype=np.int64),
    )

    rng = np.random.default_rng(int(seed) + 43)
    sample_count = min(max(int(sample_rows), 1), int(eligible.size))
    selected_np = np.sort(rng.choice(eligible, size=sample_count, replace=False))
    selected_t = torch.as_tensor(selected_np, dtype=torch.long, device=learner.device)
    transition_indices = np.asarray(stage_batch.transition_indices, dtype=np.int64)
    selected_global_t = torch.as_tensor(transition_indices[selected_np], dtype=torch.long, device=learner.device)
    flat_rows = (
        selected_t.view(-1, 1)
        * int(num_agents)
        + torch.arange(num_agents, dtype=torch.long, device=learner.device).view(1, int(num_agents))
    ).reshape(-1)
    selected_local = _index_dataclass(stage_batch.local_batch, flat_rows)
    rollout_actions = stage_batch.actions.index_select(0, selected_t).to(device=learner.device)
    labels, action_tensor, _eval_actions = _stage_action_samples(
        learner.actor,
        stage_id=int(stage_id),
        selected_local=selected_local,
        rollout_actions=rollout_actions,
        sample_count=sample_count,
        num_agents=num_agents,
        policy_samples=int(policy_action_samples),
        device=learner.device,
    )
    policy_action_tensor = action_tensor[:, 1:].contiguous()
    policy_labels = labels[1:]
    policy_action_count = int(policy_action_tensor.shape[1])

    continuations_i = max(int(continuations), 1)
    history_rows = history_rows_all[selected_np]
    horizons = horizons_all[selected_np]
    if bool(skip_qpi_probe):
        ret_cube = np.zeros((sample_count, policy_action_count, continuations_i), dtype=np.float64)
        q_mean = np.zeros((sample_count, policy_action_count), dtype=np.float64)
        q_adv = np.zeros_like(q_mean)
    else:
        if int(branch_horizon_cap) > 0:
            horizons = np.minimum(horizons, int(branch_horizon_cap)).astype(np.int64, copy=False)
        branch_rows = np.repeat(history_rows, policy_action_count * continuations_i)
        branch_horizons = np.repeat(horizons, policy_action_count * continuations_i)
        branch_actions = (
            policy_action_tensor[:, :, None]
            .expand(sample_count, policy_action_count, continuations_i, *policy_action_tensor.shape[2:])
            .reshape(sample_count * policy_action_count * continuations_i, int(num_agents), -1)
            .contiguous()
        )
        returns_by_mode = _branch_returns_qpi(
            learner,
            stage_id=int(stage_id),
            history_rows=branch_rows.tolist(),
            first_actions=branch_actions,
            horizons=branch_horizons.tolist(),
            reward_modes=[str(reward_mode)],
            follow_deterministic=bool(follow_deterministic),
            future_random_mode=str(future_random_mode),
            future_random_seed=int(seed) + 997_003,
        )
        ret_cube = (
            returns_by_mode[str(reward_mode)]
            .detach()
            .cpu()
            .numpy()
            .reshape(sample_count, policy_action_count, continuations_i)
            .astype(np.float64)
        )
        q_mean = ret_cube.mean(axis=2)
        q_adv = q_mean - q_mean.mean(axis=1, keepdims=True)
    selected_adv = advantages_norm.index_select(0, selected_global_t).detach().to(device=learner.device)

    params = _stage_optimizer_params(learner.actor, int(stage_id))
    if not params:
        return {"enabled": False, "reason": "no trainable stage actor params"}
    if str(optimizer_name).lower() == "sgd":
        opt: torch.optim.Optimizer = torch.optim.SGD(params, lr=float(lr))
    else:
        opt = torch.optim.Adam(params, lr=float(lr))
    with torch.no_grad():
        pre_all, pre_entropy_all = _eval_candidate_logprob(
            learner,
            stage_id=int(stage_id),
            selected_local=selected_local,
            candidate_actions=policy_action_tensor,
            num_agents=int(num_agents),
        )
        pre_selected = pre_all[:, 0].detach()

    entropy_coef = float(learner.entropy_coef_by_stage[int(stage_id)])
    grad_norm_value = 0.0
    policy_loss_value = 0.0
    entropy_value = 0.0
    loss_value = 0.0
    update_samples = int(sample_count)
    update_epochs = 1
    update_minibatches = 1
    if bool(full_ppo):
        stage_sample_count = int(stage_batch.num_samples)
        all_stage_idx = torch.arange(stage_sample_count, dtype=torch.long, device=learner.device)
        all_global_idx = torch.as_tensor(transition_indices, dtype=torch.long, device=learner.device)
        all_adv = advantages_norm.index_select(0, all_global_idx).detach().to(device=learner.device)
        all_actions = stage_batch.actions.to(device=learner.device)
        with torch.no_grad():
            old_chunks: list[torch.Tensor] = []
            batch_eval = 1024
            for start in range(0, stage_sample_count, batch_eval):
                idx = all_stage_idx[start : start + batch_eval]
                flat_idx = (
                    idx.view(-1, 1) * int(num_agents)
                    + torch.arange(num_agents, dtype=torch.long, device=learner.device).view(1, int(num_agents))
                ).reshape(-1)
                local_i = _index_dataclass(stage_batch.local_batch, flat_idx)
                action_i = all_actions.index_select(0, idx)
                lp_i, _ent_i, _out_i = learner._stage_actor_eval_from_batch(
                    int(stage_id),
                    local_i,
                    action_i,
                    int(num_agents),
                )
                old_chunks.append(lp_i.detach())
            old_lp_all = torch.cat(old_chunks, dim=0)
        update_samples = stage_sample_count
        update_epochs = max(int(ppo_epochs), 1)
        update_minibatches = max(int(num_minibatches), 1)
        mb_size = max(1, int(np.ceil(stage_sample_count / update_minibatches)))
        for _epoch in range(update_epochs):
            order = torch.randperm(stage_sample_count, device=learner.device)
            for start in range(0, stage_sample_count, mb_size):
                idx = order[start : start + mb_size]
                flat_idx = (
                    idx.view(-1, 1) * int(num_agents)
                    + torch.arange(num_agents, dtype=torch.long, device=learner.device).view(1, int(num_agents))
                ).reshape(-1)
                local_i = _index_dataclass(stage_batch.local_batch, flat_idx)
                action_i = all_actions.index_select(0, idx)
                logprob, entropy, _out = learner._stage_actor_eval_from_batch(
                    int(stage_id),
                    local_i,
                    action_i,
                    int(num_agents),
                )
                adv_i = all_adv.index_select(0, idx).to(dtype=logprob.dtype)
                old_i = old_lp_all.index_select(0, idx).to(dtype=logprob.dtype)
                ratio = torch.exp(torch.clamp(logprob - old_i.detach(), min=-20.0, max=20.0))
                clipped = torch.clamp(ratio, 1.0 - float(learner.clip_ratio), 1.0 + float(learner.clip_ratio))
                policy_loss = -torch.minimum(ratio * adv_i.detach(), clipped * adv_i.detach()).mean()
                entropy_mean = entropy.mean() if int(entropy.numel()) > 0 else torch.zeros((), device=learner.device)
                loss = policy_loss - entropy_coef * entropy_mean
                opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(params, float(learner.max_grad_norm))
                opt.step()
                grad_norm_value = float(torch.as_tensor(grad_norm).detach().cpu().item())
                policy_loss_value = float(policy_loss.detach().cpu().item())
                entropy_value = float(entropy_mean.detach().cpu().item())
                loss_value = float(loss.detach().cpu().item())
    else:
        selected_actions = policy_action_tensor[:, 0:1].contiguous()
        logprob_mat, entropy_mat = _eval_candidate_logprob(
            learner,
            stage_id=int(stage_id),
            selected_local=selected_local,
            candidate_actions=selected_actions,
            num_agents=int(num_agents),
        )
        logprob = logprob_mat[:, 0]
        entropy = entropy_mat[:, 0]
        old_lp = pre_selected.to(device=learner.device, dtype=logprob.dtype)
        adv = selected_adv.to(dtype=logprob.dtype)
        ratio = torch.exp(torch.clamp(logprob - old_lp.detach(), min=-20.0, max=20.0))
        clipped = torch.clamp(ratio, 1.0 - float(learner.clip_ratio), 1.0 + float(learner.clip_ratio))
        policy_loss = -torch.minimum(ratio * adv.detach(), clipped * adv.detach()).mean()
        entropy_mean = entropy.mean() if int(entropy.numel()) > 0 else torch.zeros((), device=learner.device)
        loss = policy_loss - entropy_coef * entropy_mean
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, float(learner.max_grad_norm))
        opt.step()
        grad_norm_value = float(torch.as_tensor(grad_norm).detach().cpu().item())
        policy_loss_value = float(policy_loss.detach().cpu().item())
        entropy_value = float(entropy_mean.detach().cpu().item())
        loss_value = float(loss.detach().cpu().item())
    with torch.no_grad():
        post_all, post_entropy_all = _eval_candidate_logprob(
            learner,
            stage_id=int(stage_id),
            selected_local=selected_local,
            candidate_actions=policy_action_tensor,
            num_agents=int(num_agents),
        )
    delta = (post_all - pre_all).detach().cpu().numpy().astype(np.float64)
    selected_adv_np = selected_adv.detach().cpu().numpy().astype(np.float64)
    return {
        "enabled": True,
        "kind": str(advantage_kind).lower(),
        "full_ppo": bool(full_ppo),
        "update_samples": int(update_samples),
        "update_epochs": int(update_epochs),
        "update_minibatches": int(update_minibatches),
        "sample_rows": int(sample_count),
        "policy_action_samples": int(policy_action_count),
        "continuations": int(continuations_i),
        "follow": "deterministic" if bool(follow_deterministic) else "stochastic",
        "future_random_mode": str(future_random_mode),
        "qpi_probe_enabled": not bool(skip_qpi_probe),
        "action_labels": policy_labels,
        "lr": float(lr),
        "optimizer": str(optimizer_name),
        "loss": float(loss_value),
        "policy_loss": float(policy_loss_value),
        "entropy_coef": entropy_coef,
        "entropy_pre": _summ(pre_entropy_all.detach().cpu().numpy()),
        "entropy_post": _summ(post_entropy_all.detach().cpu().numpy()),
        "entropy_update_last": float(entropy_value),
        "grad_norm": float(grad_norm_value),
        "selected_advantage": _summ(selected_adv_np),
        "corr_selected_adv_vs_qpi_rollout_adv": None
        if bool(skip_qpi_probe)
        else _corr(selected_adv_np, q_adv[:, 0]),
        "sign_agree_selected_adv_qpi_rollout_adv": None
        if bool(skip_qpi_probe)
        else float(np.mean(np.sign(selected_adv_np) == np.sign(q_adv[:, 0]))),
        "q_mean": _summ(q_mean),
        "q_adv": _summ(q_adv),
        "return_cube": _summ(ret_cube),
        "candidate_shift_vs_qpi": {}
        if bool(skip_qpi_probe)
        else _candidate_shift_summary(q_adv=q_adv, delta_logprob=delta),
        "per_state_shift_vs_qpi": {}
        if bool(skip_qpi_probe)
        else _per_state_shift_summary(q_mean=q_mean, q_adv=q_adv, delta_logprob=delta),
        "rows": [
            {
                "stage_sample": int(selected_np[i]),
                "transition_index": int(transition_indices[selected_np][i]),
                "history_row": int(history_rows[i]),
                "step": int(step_indices_all[selected_np][i]),
                "horizon": int(horizons[i]),
                "selected_advantage": float(selected_adv_np[i]),
                "q_rollout_adv": float(q_adv[i, 0]),
                "q_mean": [float(x) for x in q_mean[i].tolist()],
            }
            for i in range(sample_count)
        ],
    }


def _eval_value_on_vhat(
    learner: Any,
    *,
    stage_id: int,
    vpi_probe: dict[str, Any],
    world_batch: Any,
    sample_indices: Sequence[int],
) -> dict[str, Any]:
    if not bool(vpi_probe.get("enabled", False)):
        return {"enabled": False, "reason": vpi_probe.get("reason", "vpi probe disabled")}
    selected_t = torch.as_tensor(list(sample_indices), dtype=torch.long, device=learner.device)
    selected_world = _index_dataclass(world_batch, selected_t)
    target = np.asarray([row["vhat"] for row in vpi_probe.get("rows", [])], dtype=np.float64)
    with torch.no_grad():
        pred = learner._stage_value_eval_from_batch(int(stage_id), selected_world).detach().cpu().numpy().astype(np.float64)
    return {
        "enabled": True,
        "ev": _explained_variance_np(pred, target),
        "corr": _corr(pred, target),
        "mse": float(np.mean((pred - target) ** 2)),
        "mae": float(np.mean(np.abs(pred - target))),
        "pred": _summ(pred),
        "target": _summ(target),
        "residual": _summ(target - pred),
    }


def _incremental_relearn(
    learner: Any,
    *,
    stage_id: int,
    train_world: Any,
    train_target: torch.Tensor,
    heldout_world: Any,
    heldout_target: torch.Tensor,
    vpi_world: Any,
    vpi_probe: dict[str, Any],
    vpi_sample_indices: Sequence[int],
    checkpoints: Sequence[int],
    minibatches: int,
    lr: float,
    max_grad_norm: float,
) -> list[dict[str, Any]]:
    params = [p for p in learner.critic.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=float(lr))
    n = int(train_target.numel())
    mb_size = max(1, int(np.ceil(n / max(int(minibatches), 1))))
    ckpts = sorted({max(int(x), 0) for x in checkpoints})
    max_epoch = max(ckpts) if ckpts else 0
    history: list[dict[str, Any]] = []

    def _record(epoch: int) -> None:
        _pred_train, train_stats = _eval_critic(
            learner,
            stage_id=int(stage_id),
            world_bank=train_world,
            target=train_target,
            batch_size=max(1024, mb_size),
        )
        _pred_heldout, heldout_stats = _eval_critic(
            learner,
            stage_id=int(stage_id),
            world_bank=heldout_world,
            target=heldout_target,
            batch_size=max(1024, int(np.ceil(int(heldout_target.numel()) / max(int(minibatches), 1)))),
        )
        history.append(
            {
                "epoch": int(epoch),
                "train_mc": train_stats,
                "heldout_mc": heldout_stats,
                "heldout_vhat": _eval_value_on_vhat(
                    learner,
                    stage_id=int(stage_id),
                    vpi_probe=vpi_probe,
                    world_batch=vpi_world,
                    sample_indices=vpi_sample_indices,
                ),
            }
        )

    for epoch in range(0, max_epoch + 1):
        if epoch in ckpts:
            _record(epoch)
        if epoch >= max_epoch:
            break
        order = torch.randperm(n, device=train_target.device)
        for start in range(0, n, mb_size):
            idx = order[start : start + mb_size]
            pred = learner._stage_value_eval_from_batch(int(stage_id), _index_dataclass(train_world, idx))
            loss = F.mse_loss(pred, train_target.index_select(0, idx))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(max_grad_norm))
            opt.step()
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(STAGE_ID), default="sat")
    parser.add_argument("--config", required=True)
    parser.add_argument("--reward_mode", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--rollout_env_steps", type=int, default=250)
    parser.add_argument("--target", choices=["mc"], default="mc")
    parser.add_argument("--pre_train_rollouts", type=int, default=8)
    parser.add_argument("--pre_heldout_rollouts", type=int, default=2)
    parser.add_argument("--pre_critic_epochs", type=int, default=30)
    parser.add_argument("--post_train_rollouts", type=int, default=8)
    parser.add_argument("--post_heldout_rollouts", type=int, default=2)
    parser.add_argument("--relearn_epoch_checkpoints", default="0,1,3,5,10,20,30")
    parser.add_argument(
        "--relearn_critic_lrs",
        default="",
        help="Comma-separated critic learning rates for post-update relearn. Defaults to --critic_lr.",
    )
    parser.add_argument("--critic_minibatches", type=int, default=8)
    parser.add_argument("--critic_lr", type=float, default=None)
    parser.add_argument("--critic_value_mode", choices=["relational", "global_only", "global_linear"], default=None)
    parser.add_argument("--critic_message_layers", type=int, default=None)
    parser.add_argument("--critic_value_head_hidden", type=int, default=None)
    parser.add_argument("--actor_advantage_normalize_enabled", default=None)
    parser.add_argument("--stagewise_advantage_norm_enabled", default=None)
    parser.add_argument("--actor_update_rows", type=int, default=16)
    parser.add_argument("--actor_update_policy_action_samples", type=int, default=4)
    parser.add_argument("--actor_update_continuations", type=int, default=2)
    parser.add_argument("--actor_update_lr", type=float, default=3e-4)
    parser.add_argument(
        "--refit_critic_on_actor_batch_epochs",
        type=int,
        default=0,
        help="If >0, train the critic on the actor-update rollout MC targets before computing A_gae/A_mc.",
    )
    parser.add_argument("--refit_critic_on_actor_batch_minibatches", type=int, default=None)
    parser.add_argument("--actor_update_kind", choices=["qpi_all", "gae", "mc", "none"], default="qpi_all")
    parser.add_argument("--actor_update_optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--actor_update_follow", choices=["deterministic", "stochastic"], default="stochastic")
    parser.add_argument("--actor_update_future_random", choices=["copy", "resample"], default="resample")
    parser.add_argument(
        "--actor_update_full_ppo",
        action="store_true",
        help="For gae/mc actor updates, update on the full stage rollout instead of only sampled probe rows.",
    )
    parser.add_argument("--actor_update_epochs", type=int, default=None)
    parser.add_argument("--actor_update_minibatches", type=int, default=None)
    parser.add_argument(
        "--skip_actor_update_qpi_probe",
        action="store_true",
        help="Skip expensive branch-return Qpi diagnostics around the actor update; the actor update still runs.",
    )
    parser.add_argument("--vpi_rows", type=int, default=16)
    parser.add_argument("--vpi_policy_action_samples", type=int, default=4)
    parser.add_argument("--vpi_continuations", type=int, default=2)
    parser.add_argument("--vpi_follow", choices=["deterministic", "stochastic"], default="stochastic")
    parser.add_argument("--min_horizon", type=int, default=20)
    parser.add_argument("--branch_horizon_cap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=45210)
    parser.add_argument("--torch_threads", type=int, default=1)
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
    if args.critic_lr is not None:
        cfg.critic_lr = float(args.critic_lr)
    if args.critic_value_mode is not None:
        cfg.critic_value_mode = str(args.critic_value_mode)
    if args.critic_message_layers is not None:
        cfg.critic_message_layers = int(args.critic_message_layers)
    if args.critic_value_head_hidden is not None:
        cfg.critic_value_head_hidden = int(args.critic_value_head_hidden)
    actor_adv_norm = _optional_bool_arg(args.actor_advantage_normalize_enabled)
    if actor_adv_norm is not None:
        cfg.actor_advantage_normalize_enabled = bool(actor_adv_norm)
    stagewise_adv_norm = _optional_bool_arg(args.stagewise_advantage_norm_enabled)
    if stagewise_adv_norm is not None:
        cfg.stagewise_advantage_norm_enabled = bool(stagewise_adv_norm)

    learner, _actor, _critic = _make_learner(cfg, device=device, stage_id=stage_id)
    group = make_structured_driver_group(cfg, num_envs=int(args.num_envs), backend="sync", mode="train")
    try:
        learner.bind_native_runtime_contract(group)

        pre_train_world, pre_train_target, pre_train_summaries = _collect_stage_bank(
            learner,
            group,
            stage_id=int(stage_id),
            rollout_env_steps=int(args.rollout_env_steps),
            rollouts=int(args.pre_train_rollouts),
            seed_base=int(args.seed) + 10_000,
            device=device,
            target=str(args.target),
        )
        pre_heldout_world, pre_heldout_target, pre_heldout_summaries = _collect_stage_bank(
            learner,
            group,
            stage_id=int(stage_id),
            rollout_env_steps=int(args.rollout_env_steps),
            rollouts=int(args.pre_heldout_rollouts),
            seed_base=int(args.seed) + 200_000,
            device=device,
            target=str(args.target),
        )
        pre_history = _train_critic_only(
            learner,
            stage_id=int(stage_id),
            train_world=pre_train_world,
            train_target=pre_train_target,
            heldout_world=pre_heldout_world,
            heldout_target=pre_heldout_target,
            epochs=int(args.pre_critic_epochs),
            minibatches=int(args.critic_minibatches),
            lr=float(cfg.critic_lr),
            max_grad_norm=float(cfg.max_grad_norm),
            eval_every=max(1, int(args.pre_critic_epochs) // 3),
        )
        critic_pi0_state = copy.deepcopy(learner.critic.state_dict())

        reset_many = getattr(group, "reset_many", None)
        if callable(reset_many):
            reset_many([int(args.seed) + 900_000 + env for env in range(int(args.num_envs))])
        update_buffer, update_views, _update_returns = _collect_one_rollout(
            learner,
            group,
            rollout_env_steps=int(args.rollout_env_steps),
            device=device,
            target=str(args.target),
        )
        actor_batch_refit_history: list[dict[str, float]] = []
        if int(args.refit_critic_on_actor_batch_epochs) > 0:
            update_stage_batch = update_views.training_view.stage_batches[int(stage_id)]
            update_transition_idx = torch.as_tensor(
                np.asarray(update_stage_batch.transition_indices, dtype=np.int64),
                dtype=torch.long,
                device=device,
            )
            update_target = _update_returns.index_select(0, update_transition_idx).detach()
            actor_batch_refit_history = _train_critic_only(
                learner,
                stage_id=int(stage_id),
                train_world=update_stage_batch.world_batch,
                train_target=update_target,
                heldout_world=update_stage_batch.world_batch,
                heldout_target=update_target,
                epochs=int(args.refit_critic_on_actor_batch_epochs),
                minibatches=(
                    int(args.refit_critic_on_actor_batch_minibatches)
                    if args.refit_critic_on_actor_batch_minibatches is not None
                    else int(args.critic_minibatches)
                ),
                lr=float(cfg.critic_lr),
                max_grad_norm=float(cfg.max_grad_norm),
                eval_every=max(1, int(args.refit_critic_on_actor_batch_epochs) // 4),
            )
        del _update_returns
        if str(args.actor_update_kind) == "none":
            actor_update = {"enabled": False, "kind": "none", "reason": "actor update skipped by request"}
        elif str(args.actor_update_kind) == "qpi_all":
            actor_update = _apply_qpi_all_candidate_actor_update(
                learner,
                cfg=cfg,
                stage_id=int(stage_id),
                num_envs=int(args.num_envs),
                views=update_views,
                reward_mode=str(args.reward_mode or cfg.reward_mode),
                sample_rows=int(args.actor_update_rows),
                policy_action_samples=int(args.actor_update_policy_action_samples),
                continuations=int(args.actor_update_continuations),
                follow_deterministic=str(args.actor_update_follow) == "deterministic",
                min_horizon=int(args.min_horizon),
                branch_horizon_cap=int(args.branch_horizon_cap),
                seed=int(args.seed),
                lr=float(args.actor_update_lr),
                optimizer_name=str(args.actor_update_optimizer),
                future_random_mode=str(args.actor_update_future_random),
            )
        else:
            actor_update = _apply_selected_advantage_actor_update(
                learner,
                cfg=cfg,
                stage_id=int(stage_id),
                num_envs=int(args.num_envs),
                buffer=update_buffer,
                views=update_views,
                reward_mode=str(args.reward_mode or cfg.reward_mode),
                advantage_kind=str(args.actor_update_kind),
                sample_rows=int(args.actor_update_rows),
                policy_action_samples=int(args.actor_update_policy_action_samples),
                continuations=int(args.actor_update_continuations),
                follow_deterministic=str(args.actor_update_follow) == "deterministic",
                min_horizon=int(args.min_horizon),
                branch_horizon_cap=int(args.branch_horizon_cap),
                seed=int(args.seed),
                lr=float(args.actor_update_lr),
                optimizer_name=str(args.actor_update_optimizer),
                full_ppo=bool(args.actor_update_full_ppo),
                ppo_epochs=int(args.actor_update_epochs) if args.actor_update_epochs is not None else int(cfg.ppo_epochs),
                num_minibatches=(
                    int(args.actor_update_minibatches)
                    if args.actor_update_minibatches is not None
                    else int(cfg.num_mini_batch)
                ),
                future_random_mode=str(args.actor_update_future_random),
                skip_qpi_probe=bool(args.skip_actor_update_qpi_probe),
            )
        critic_after_actor_update_before_relearn = copy.deepcopy(learner.critic.state_dict())

        # Branch replay used by the actor-update probe can leave a large
        # selected-env native workspace alive.  Recreate the rollout group
        # before collecting pi1 banks so the post-update critic relearn audit
        # measures the policy/critic question rather than workspace pressure.
        close_structured_env_group(group)
        group = make_structured_driver_group(cfg, num_envs=int(args.num_envs), backend="sync", mode="train")
        learner.bind_native_runtime_contract(group)
        if device.type == "cuda":
            torch.cuda.empty_cache()

        post_train_world, post_train_target, post_train_summaries = _collect_stage_bank(
            learner,
            group,
            stage_id=int(stage_id),
            rollout_env_steps=int(args.rollout_env_steps),
            rollouts=int(args.post_train_rollouts),
            seed_base=int(args.seed) + 1_000_000,
            device=device,
            target=str(args.target),
        )
        post_heldout_world, post_heldout_target, post_heldout_summaries = _collect_stage_bank(
            learner,
            group,
            stage_id=int(stage_id),
            rollout_env_steps=int(args.rollout_env_steps),
            rollouts=int(args.post_heldout_rollouts),
            seed_base=int(args.seed) + 2_000_000,
            device=device,
            target=str(args.target),
        )
        # Build one native rollout for the Vhat probe so branch replay history is available.
        reset_many = getattr(group, "reset_many", None)
        if callable(reset_many):
            reset_many([int(args.seed) + 3_000_000 + env for env in range(int(args.num_envs))])
        vpi_buffer, vpi_views, _vpi_returns = _collect_one_rollout(
            learner,
            group,
            rollout_env_steps=int(args.rollout_env_steps),
            device=device,
            target=str(args.target),
        )
        del vpi_buffer, _vpi_returns
        vpi_stage_batch = vpi_views.training_view.stage_batches[int(stage_id)]
        vpi_probe = _heldout_vpi_ceiling_probe(
            learner,
            cfg=cfg,
            stage_id=int(stage_id),
            num_envs=int(args.num_envs),
            heldout_views=vpi_views,
            reward_mode=str(args.reward_mode or cfg.reward_mode),
            sample_rows=int(args.vpi_rows),
            policy_action_samples=int(args.vpi_policy_action_samples),
            continuations=int(args.vpi_continuations),
            min_horizon=int(args.min_horizon),
            branch_horizon_cap=int(args.branch_horizon_cap),
            seed=int(args.seed) + 7,
            device=device,
            initial_critic_state=critic_pi0_state,
            follow_deterministic=str(args.vpi_follow) == "deterministic",
        )
        vpi_sample_indices = [int(row["stage_sample"]) for row in vpi_probe.get("rows", [])]
        vpi_world_bank = _clone_dataclass_tensors(vpi_stage_batch.world_batch, device=device)

        relearn_lrs = [
            float(item.strip())
            for item in str(args.relearn_critic_lrs).replace(";", ",").split(",")
            if item.strip()
        ]
        if not relearn_lrs:
            relearn_lrs = [float(cfg.critic_lr)]
        relearn_by_lr: dict[str, list[dict[str, Any]]] = {}
        for lr_value in relearn_lrs:
            # Reset to the same post-actor-update critic before measuring each
            # relearn LR.  Actor remains pi1; train/heldout/Vhat rows are shared.
            learner.critic.load_state_dict(critic_after_actor_update_before_relearn, strict=True)
            relearn_by_lr[f"{float(lr_value):.8g}"] = _incremental_relearn(
                learner,
                stage_id=int(stage_id),
                train_world=post_train_world,
                train_target=post_train_target,
                heldout_world=post_heldout_world,
                heldout_target=post_heldout_target,
                vpi_world=vpi_world_bank,
                vpi_probe=vpi_probe,
                vpi_sample_indices=vpi_sample_indices,
                checkpoints=_parse_epoch_checkpoints(str(args.relearn_epoch_checkpoints)),
                minibatches=int(args.critic_minibatches),
                lr=float(lr_value),
                max_grad_norm=float(cfg.max_grad_norm),
            )
        relearn_history = relearn_by_lr[f"{float(relearn_lrs[0]):.8g}"]

        result = {
            "config": {
                "stage": str(args.stage),
                "stage_id": int(stage_id),
                "config": str(args.config),
                "reward_mode": str(args.reward_mode or cfg.reward_mode),
                "num_envs": int(args.num_envs),
                "rollout_env_steps": int(args.rollout_env_steps),
                "target": str(args.target),
                "critic_lr": float(cfg.critic_lr),
                "critic_value_mode": str(getattr(cfg, "critic_value_mode", "")),
                "critic_message_layers": int(getattr(cfg, "critic_message_layers", -1)),
                "critic_value_head_hidden": int(getattr(cfg, "critic_value_head_hidden", -1)),
                "actor_advantage_normalize_enabled": bool(getattr(cfg, "actor_advantage_normalize_enabled", False)),
                "stagewise_advantage_norm_enabled": bool(getattr(cfg, "stagewise_advantage_norm_enabled", False)),
                "pre_critic_epochs": int(args.pre_critic_epochs),
                "actor_update_kind": str(args.actor_update_kind),
                "actor_update_full_ppo": bool(args.actor_update_full_ppo),
                "actor_update_future_random": str(args.actor_update_future_random),
                "skip_actor_update_qpi_probe": bool(args.skip_actor_update_qpi_probe),
                "refit_critic_on_actor_batch_epochs": int(args.refit_critic_on_actor_batch_epochs),
                "refit_critic_on_actor_batch_minibatches": (
                    int(args.refit_critic_on_actor_batch_minibatches)
                    if args.refit_critic_on_actor_batch_minibatches is not None
                    else int(args.critic_minibatches)
                ),
                "relearn_critic_lrs": [float(x) for x in relearn_lrs],
                "relearn_epoch_checkpoints": _parse_epoch_checkpoints(str(args.relearn_epoch_checkpoints)),
                "seed": int(args.seed),
            },
            "pre_pi0_fit": {
                "train_summaries": pre_train_summaries,
                "heldout_summaries": pre_heldout_summaries,
                "history": pre_history,
            },
            "actor_batch_refit_before_update": actor_batch_refit_history,
            "actor_update_pi0_to_pi1": actor_update,
            "post_pi1_banks": {
                "train_summaries": post_train_summaries,
                "heldout_summaries": post_heldout_summaries,
            },
            "post_pi1_vhat_probe": vpi_probe,
            "relearn_from_pi0_critic_on_pi1": relearn_history,
            "relearn_from_pi0_critic_on_pi1_by_lr": relearn_by_lr,
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result["config"], indent=2, ensure_ascii=False))
        print("pre final heldout EV:", pre_history[-1]["heldout_ev"] if pre_history else None)
        print("actor update q-shift:", actor_update.get("candidate_shift_vs_qpi", {}).get("mean_qadv_times_delta_logprob"))
        print("relearn checkpoints:")
        for lr_key, history in relearn_by_lr.items():
            print(f"  lr={lr_key}")
            for row in history:
                print(
                    f"    epoch={row['epoch']} heldout_mc_ev={row['heldout_mc']['ev']:.4f} "
                    f"vhat_ev={row['heldout_vhat'].get('ev', 0.0):.4f} "
                    f"vhat_mae={row['heldout_vhat'].get('mae', 0.0):.4f}"
                )
        print(f"wrote {out_path}")
    finally:
        close_structured_env_group(group)


if __name__ == "__main__":
    main()
