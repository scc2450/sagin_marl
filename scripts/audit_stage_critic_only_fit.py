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
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO, _collate_dataclass, _index_dataclass
from sagin_marl.rl.stage_mcgae import (
    STAGE_ID,
    clone_dataclass_tensors as _clone_dataclass_tensors,
    collect_one_rollout as _collect_one_rollout,
    compute_returns_for_views as _compute_returns_for_views,
    eval_critic as _eval_critic,
    explained_variance_np as _explained_variance_np,
    force_single_stage_config as _force_single_stage_config,
    make_learner as _make_learner,
    set_seed as _set_seed,
)
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver_group

from scripts.audit_stage_ppo_credit_alignment import (
    _normalize_advantages_like_update,
    _stage_action_samples,
)
from scripts.audit_stage_credit_chain import (
    _candidate_shift_summary,
    _per_state_shift_summary,
    _probe_actor_update,
    _standardize_np,
)
from scripts.audit_stage_qpi_action_credit import _branch_returns_qpi
from scripts.diagnose_reward_action_sensitivity import (
    _branch_returns_from_history_for_modes,
    _corr,
    _summ,
)


def _collect_stage_bank(
    learner: StructuredMAPPO,
    group: Any,
    *,
    stage_id: int,
    rollout_env_steps: int,
    rollouts: int,
    seed_base: int,
    device: torch.device,
    target: str,
) -> tuple[Any, torch.Tensor, list[dict[str, float]]]:
    worlds: list[Any] = []
    targets: list[torch.Tensor] = []
    summaries: list[dict[str, float]] = []
    for ridx in range(max(int(rollouts), 1)):
        reset_many = getattr(group, "reset_many", None)
        if callable(reset_many):
            reset_many([int(seed_base) + ridx * 10_000 + env for env in range(len(group))])
        buffer, views, returns = _collect_one_rollout(
            learner,
            group,
            rollout_env_steps=int(rollout_env_steps),
            device=device,
            target=target,
        )
        stage_batch = views.training_view.stage_batches.get(int(stage_id))
        if stage_batch is None or int(stage_batch.num_samples) <= 0:
            continue
        idx = torch.as_tensor(
            np.asarray(stage_batch.transition_indices, dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        stage_returns = returns.index_select(0, idx).detach()
        # Native rollout views point into reusable runtime buffers.  Clone the
        # bank rows immediately; otherwise collecting a later rollout can
        # silently mutate previously stored critic inputs while targets stay
        # unchanged.
        worlds.append(_clone_dataclass_tensors(stage_batch.world_batch, device=device))
        targets.append(stage_returns)
        summaries.append(
            {
                "rollout": float(ridx),
                "samples": float(stage_returns.numel()),
                "return_mean": float(stage_returns.mean().detach().cpu().item()),
                "return_std": float(stage_returns.std(unbiased=False).detach().cpu().item()),
            }
        )
    if not worlds:
        raise RuntimeError("no stage samples collected for critic-only bank.")
    world_bank = _collate_dataclass(worlds, device)
    target_bank = torch.cat(targets, dim=0).to(device=device, dtype=torch.float32)
    return world_bank, target_bank, summaries


def _train_critic_only(
    learner: StructuredMAPPO,
    *,
    stage_id: int,
    train_world: Any,
    train_target: torch.Tensor,
    heldout_world: Any,
    heldout_target: torch.Tensor,
    epochs: int,
    minibatches: int,
    lr: float,
    max_grad_norm: float,
    eval_every: int,
    accumulate_full_batch: bool = False,
) -> list[dict[str, float]]:
    params = [p for p in learner.critic.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=float(lr))
    n = int(train_target.numel())
    history: list[dict[str, float]] = []
    for epoch in range(max(int(epochs), 0) + 1):
        if epoch == 0 or epoch == int(epochs) or (int(eval_every) > 0 and epoch % int(eval_every) == 0):
            _pred_train, train_stats = _eval_critic(
                learner,
                stage_id=stage_id,
                world_bank=train_world,
                target=train_target,
                batch_size=max(1024, int(np.ceil(n / max(int(minibatches), 1)))),
            )
            _pred_heldout, heldout_stats = _eval_critic(
                learner,
                stage_id=stage_id,
                world_bank=heldout_world,
                target=heldout_target,
                batch_size=max(1024, int(np.ceil(int(heldout_target.numel()) / max(int(minibatches), 1)))),
            )
            history.append(
                {
                    "epoch": float(epoch),
                    "train_ev": float(train_stats["ev"]),
                    "heldout_ev": float(heldout_stats["ev"]),
                    "train_mse": float(train_stats["mse"]),
                    "heldout_mse": float(heldout_stats["mse"]),
                    "train_corr": float(train_stats["corr"]),
                    "heldout_corr": float(heldout_stats["corr"]),
                }
            )
        if epoch >= int(epochs):
            break
        order = torch.randperm(n, device=train_target.device)
        mb_size = max(1, int(np.ceil(n / max(int(minibatches), 1))))
        if bool(accumulate_full_batch):
            opt.zero_grad(set_to_none=True)
            for start in range(0, n, mb_size):
                idx = order[start : start + mb_size]
                pred = learner._stage_value_eval_from_batch(int(stage_id), _index_dataclass(train_world, idx))
                # Weight by microbatch size so accumulated gradients match full-batch MSE.
                loss = F.mse_loss(pred, train_target.index_select(0, idx)) * (float(idx.numel()) / float(max(n, 1)))
                loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(max_grad_norm))
            opt.step()
        else:
            for start in range(0, n, mb_size):
                idx = order[start : start + mb_size]
                pred = learner._stage_value_eval_from_batch(int(stage_id), _index_dataclass(train_world, idx))
                loss = F.mse_loss(pred, train_target.index_select(0, idx))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, float(max_grad_norm))
                opt.step()
    return history


def _train_critic_bank_once(
    learner: StructuredMAPPO,
    *,
    stage_id: int,
    train_world: Any,
    train_target: torch.Tensor,
    epochs: int,
    minibatches: int,
    lr: float,
    max_grad_norm: float,
) -> dict[str, float]:
    params = [p for p in learner.critic.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=float(lr))
    n = int(train_target.numel())
    if n <= 0:
        return {"samples": 0.0, "target_mean": 0.0, "target_std": 0.0, "final_train_ev": 0.0, "final_train_mse": 0.0}
    mb_size = max(1, int(np.ceil(n / max(int(minibatches), 1))))
    for _epoch in range(max(int(epochs), 0)):
        order = torch.randperm(n, device=train_target.device)
        for start in range(0, n, mb_size):
            idx = order[start : start + mb_size]
            pred = learner._stage_value_eval_from_batch(int(stage_id), _index_dataclass(train_world, idx))
            loss = F.mse_loss(pred, train_target.index_select(0, idx))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(max_grad_norm))
            opt.step()
    _pred, stats = _eval_critic(
        learner,
        stage_id=stage_id,
        world_bank=train_world,
        target=train_target,
        batch_size=max(1024, mb_size),
    )
    return {
        "samples": float(n),
        "target_mean": float(train_target.mean().detach().cpu().item()),
        "target_std": float(train_target.std(unbiased=False).detach().cpu().item()),
        "final_train_ev": float(stats["ev"]),
        "final_train_mse": float(stats["mse"]),
        "final_train_corr": float(stats["corr"]),
    }


def _run_fixed_policy_fitted_eval(
    learner: StructuredMAPPO,
    group: Any,
    *,
    stage_id: int,
    rollout_env_steps: int,
    rounds: int,
    train_rollouts_per_round: int,
    seed_base: int,
    device: torch.device,
    target: str,
    epochs_per_round: int,
    minibatches: int,
    lr: float,
    max_grad_norm: float,
    start_round: int = 1,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    for round_idx in range(max(int(rounds), 0)):
        train_world, train_target, _summaries = _collect_stage_bank(
            learner,
            group,
            stage_id=int(stage_id),
            rollout_env_steps=int(rollout_env_steps),
            rollouts=int(train_rollouts_per_round),
            seed_base=int(seed_base) + 1_000_000 * (round_idx + 1),
            device=device,
            target=str(target),
        )
        round_stats = _train_critic_bank_once(
            learner,
            stage_id=int(stage_id),
            train_world=train_world,
            train_target=train_target,
            epochs=int(epochs_per_round),
            minibatches=int(minibatches),
            lr=float(lr),
            max_grad_norm=float(max_grad_norm),
        )
        round_stats["round"] = float(int(start_round) + round_idx)
        history.append(round_stats)
    return history


def _heldout_oracle_probe(
    learner: StructuredMAPPO,
    *,
    cfg: Any,
    stage_id: int,
    num_envs: int,
    heldout_views: Any,
    heldout_returns: torch.Tensor,
    reward_mode: str,
    sample_rows: int,
    policy_action_samples: int,
    min_horizon: int,
    branch_horizon_cap: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    stage_batch = heldout_views.training_view.stage_batches[int(stage_id)]
    num_samples = int(stage_batch.num_samples)
    num_agents = int(stage_batch.num_agents)
    transition_indices = np.asarray(stage_batch.transition_indices, dtype=np.int64)
    history_rows_all = ((transition_indices - int(stage_id)) // 3).astype(np.int64)
    step_indices_all = history_rows_all // max(int(num_envs), 1)
    horizons_all = np.maximum(int(cfg.T_steps) - step_indices_all, 1).astype(np.int64)
    eligible = np.flatnonzero(horizons_all >= max(int(min_horizon), 1))
    if eligible.size <= 0:
        return {"enabled": False, "reason": "no eligible rows"}
    rng = np.random.default_rng(int(seed) + 91)
    sample_count = min(max(int(sample_rows), 1), int(eligible.size))
    selected_np = np.sort(rng.choice(eligible, size=sample_count, replace=False))
    selected_t = torch.as_tensor(selected_np, dtype=torch.long, device=device)
    flat_rows = (
        selected_t.view(-1, 1)
        * int(num_agents)
        + torch.arange(num_agents, dtype=torch.long, device=device).view(1, int(num_agents))
    ).reshape(-1)
    selected_local = _index_dataclass(stage_batch.local_batch, flat_rows)
    sampled_actions = stage_batch.actions.index_select(0, selected_t).to(device=device)
    labels, action_tensor, _eval_actions = _stage_action_samples(
        learner.actor,
        stage_id=int(stage_id),
        selected_local=selected_local,
        rollout_actions=sampled_actions,
        sample_count=sample_count,
        num_agents=num_agents,
        policy_samples=int(policy_action_samples),
        device=device,
    )
    action_count = int(action_tensor.shape[1])
    history_rows = history_rows_all[selected_np]
    horizons = horizons_all[selected_np]
    if int(branch_horizon_cap) > 0:
        horizons = np.minimum(horizons, int(branch_horizon_cap)).astype(np.int64, copy=False)
    returns_by_mode = _branch_returns_from_history_for_modes(
        learner,
        stage_id=int(stage_id),
        history_rows=np.repeat(history_rows, action_count).tolist(),
        first_actions=action_tensor.reshape(sample_count * action_count, int(num_agents), -1),
        horizons=np.repeat(horizons, action_count).tolist(),
        reward_modes=[str(reward_mode)],
    )
    branch_matrix = returns_by_mode[str(reward_mode)].detach().cpu().numpy().reshape(sample_count, action_count)
    sample_return = branch_matrix[:, 1]
    stochastic_returns = branch_matrix[:, 1:]
    oracle_policy_adv = sample_return - np.mean(stochastic_returns, axis=1)
    selected_global_t = torch.as_tensor(transition_indices[selected_np], dtype=torch.long, device=device)
    selected_returns = heldout_returns.index_select(0, selected_global_t).detach()
    with torch.no_grad():
        selected_world = _index_dataclass(stage_batch.world_batch, selected_t)
        selected_values = learner._stage_value_eval_from_batch(int(stage_id), selected_world).detach()
    residual = (selected_returns - selected_values).detach().cpu().numpy()
    return {
        "enabled": True,
        "sample_rows": int(sample_count),
        "action_labels": labels,
        "corr_return_vs_oracle_policy_adv": _corr(selected_returns.detach().cpu().numpy(), oracle_policy_adv),
        "corr_value_vs_oracle_policy_adv": _corr(selected_values.detach().cpu().numpy(), oracle_policy_adv),
        "corr_residual_vs_oracle_policy_adv": _corr(residual, oracle_policy_adv),
        "sign_agree_residual_oracle_policy_adv": float(np.mean(np.sign(residual) == np.sign(oracle_policy_adv))),
        "return": _summ(selected_returns.detach().cpu().numpy()),
        "value": _summ(selected_values.detach().cpu().numpy()),
        "residual": _summ(residual),
        "oracle_policy_adv": _summ(oracle_policy_adv),
        "branch_returns": _summ(branch_matrix),
        "rows": [
            {
                "stage_sample": int(selected_np[i]),
                "transition_index": int(transition_indices[selected_np][i]),
                "history_row": int(history_rows[i]),
                "step": int(step_indices_all[selected_np][i]),
                "horizon": int(horizons[i]),
                "return": float(selected_returns[i].detach().cpu().item()),
                "value": float(selected_values[i].detach().cpu().item()),
                "residual": float(residual[i]),
                "oracle_policy_adv": float(oracle_policy_adv[i]),
            }
            for i in range(sample_count)
        ],
    }


def _heldout_vpi_ceiling_probe(
    learner: StructuredMAPPO,
    *,
    cfg: Any,
    stage_id: int,
    num_envs: int,
    heldout_views: Any,
    reward_mode: str,
    sample_rows: int,
    policy_action_samples: int,
    continuations: int,
    min_horizon: int,
    branch_horizon_cap: int,
    seed: int,
    device: torch.device,
    initial_critic_state: dict[str, torch.Tensor] | None,
    follow_deterministic: bool,
    future_random_mode: str = "resample",
) -> dict[str, Any]:
    stage_batch = heldout_views.training_view.stage_batches[int(stage_id)]
    num_agents = int(stage_batch.num_agents)
    transition_indices = np.asarray(stage_batch.transition_indices, dtype=np.int64)
    history_rows_all = ((transition_indices - int(stage_id)) // 3).astype(np.int64)
    step_indices_all = history_rows_all // max(int(num_envs), 1)
    horizons_all = np.maximum(int(cfg.T_steps) - step_indices_all, 1).astype(np.int64)
    eligible = np.flatnonzero(horizons_all >= max(int(min_horizon), 1))
    if eligible.size <= 0:
        return {"enabled": False, "reason": "no eligible rows"}
    rng = np.random.default_rng(int(seed) + 177)
    sample_count = min(max(int(sample_rows), 1), int(eligible.size))
    selected_np = np.sort(rng.choice(eligible, size=sample_count, replace=False))
    selected_t = torch.as_tensor(selected_np, dtype=torch.long, device=device)
    flat_rows = (
        selected_t.view(-1, 1)
        * int(num_agents)
        + torch.arange(num_agents, dtype=torch.long, device=device).view(1, int(num_agents))
    ).reshape(-1)
    selected_local = _index_dataclass(stage_batch.local_batch, flat_rows)
    sampled_actions = stage_batch.actions.index_select(0, selected_t).to(device=device)
    labels, action_tensor, _eval_actions = _stage_action_samples(
        learner.actor,
        stage_id=int(stage_id),
        selected_local=selected_local,
        rollout_actions=sampled_actions,
        sample_count=sample_count,
        num_agents=num_agents,
        policy_samples=int(policy_action_samples),
        device=device,
    )
    # The deterministic ref is useful for debugging but not part of Vπ.  The
    # rollout action and extra stochastic samples are policy samples.
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
        future_random_seed=int(seed) + 771_001,
    )
    ret_cube = (
        returns_by_mode[str(reward_mode)]
        .detach()
        .cpu()
        .numpy()
        .reshape(sample_count, policy_action_count, continuations_i)
        .astype(np.float64)
    )
    vhat = ret_cube.mean(axis=(1, 2))
    action_mean = ret_cube.mean(axis=2)
    vhat_se = ret_cube.reshape(sample_count, -1).std(axis=1) / np.sqrt(max(policy_action_count * continuations_i, 1))
    selected_world = _index_dataclass(stage_batch.world_batch, selected_t)
    with torch.no_grad():
        final_values = learner._stage_value_eval_from_batch(int(stage_id), selected_world).detach().cpu().numpy()
    initial_values: np.ndarray | None = None
    if initial_critic_state is not None:
        final_state = copy.deepcopy(learner.critic.state_dict())
        try:
            learner.critic.load_state_dict(initial_critic_state, strict=True)
            with torch.no_grad():
                initial_values = learner._stage_value_eval_from_batch(int(stage_id), selected_world).detach().cpu().numpy()
        finally:
            learner.critic.load_state_dict(final_state, strict=True)
    initial_stats = None
    if initial_values is not None:
        initial_stats = {
            "mse": float(np.mean((initial_values - vhat) ** 2)),
            "mae": float(np.mean(np.abs(initial_values - vhat))),
            "ev": _explained_variance_np(initial_values, vhat),
            "corr": _corr(initial_values, vhat),
            "pred": _summ(initial_values),
            "residual": _summ(vhat - initial_values),
        }
    final_stats = {
        "mse": float(np.mean((final_values - vhat) ** 2)),
        "mae": float(np.mean(np.abs(final_values - vhat))),
        "ev": _explained_variance_np(final_values, vhat),
        "corr": _corr(final_values, vhat),
        "pred": _summ(final_values),
        "target": _summ(vhat),
        "target_se": _summ(vhat_se),
        "residual": _summ(vhat - final_values),
    }
    return {
        "enabled": True,
        "sample_rows": int(sample_count),
        "policy_action_samples": int(policy_action_count),
        "continuations": int(continuations_i),
        "follow": "deterministic" if bool(follow_deterministic) else "stochastic",
        "future_random_mode": str(future_random_mode),
        "action_labels": policy_labels,
        "initial": initial_stats,
        "final": final_stats,
        "vhat": _summ(vhat),
        "action_mean": _summ(action_mean),
        "return_cube": _summ(ret_cube),
        "rows": [
            {
                "stage_sample": int(selected_np[i]),
                "transition_index": int(transition_indices[selected_np][i]),
                "history_row": int(history_rows[i]),
                "step": int(step_indices_all[selected_np][i]),
                "horizon": int(horizons[i]),
                "vhat": float(vhat[i]),
                "vhat_se": float(vhat_se[i]),
                "value_initial": float(initial_values[i]) if initial_values is not None else None,
                "value_final": float(final_values[i]),
                "action_returns": [float(x) for x in action_mean[i].tolist()],
            }
            for i in range(sample_count)
        ],
    }


def _strip_actor_probe(probe: dict[str, Any], *, q_mean: np.ndarray, q_adv: np.ndarray) -> dict[str, Any]:
    out = {k: v for k, v in probe.items() if k != "candidate_delta_logprob"}
    delta = np.asarray(probe.get("candidate_delta_logprob", np.zeros_like(q_adv)), dtype=np.float64)
    out["candidate_shift_vs_qpi"] = _candidate_shift_summary(q_adv=q_adv, delta_logprob=delta)
    out["per_state_shift_vs_qpi"] = _per_state_shift_summary(
        q_mean=q_mean,
        q_adv=q_adv,
        delta_logprob=delta,
    )
    return out


def _heldout_advantage_actor_probe(
    learner: StructuredMAPPO,
    *,
    cfg: Any,
    stage_id: int,
    num_envs: int,
    heldout_buffer: StructuredRolloutBuffer,
    heldout_views: Any,
    reward_mode: str,
    sample_rows: int,
    policy_action_samples: int,
    continuations: int,
    min_horizon: int,
    branch_horizon_cap: int,
    seed: int,
    device: torch.device,
    follow_deterministic: bool,
    probe_lr: float,
    probe_lrs: Sequence[float] | None,
    probe_optimizer: str,
    future_random_mode: str = "resample",
) -> dict[str, Any]:
    stage_batch = heldout_views.training_view.stage_batches[int(stage_id)]
    num_agents = int(stage_batch.num_agents)
    transition_indices = np.asarray(stage_batch.transition_indices, dtype=np.int64)
    history_rows_all = ((transition_indices - int(stage_id)) // 3).astype(np.int64)
    step_indices_all = history_rows_all // max(int(num_envs), 1)
    horizons_all = np.maximum(int(cfg.T_steps) - step_indices_all, 1).astype(np.int64)
    eligible = np.flatnonzero(horizons_all >= max(int(min_horizon), 1))
    if eligible.size <= 0:
        return {"enabled": False, "reason": "no eligible rows"}

    mc_returns, _mc_adv_dummy, _zero_values = _compute_returns_for_views(
        learner,
        heldout_buffer,
        heldout_views,
        target="mc",
        device=device,
    )
    gae_returns, gae_advantages, final_values = _compute_returns_for_views(
        learner,
        heldout_buffer,
        heldout_views,
        target="train_gae",
        device=device,
    )
    del gae_returns, _mc_adv_dummy, _zero_values
    mc_advantages = mc_returns - final_values
    mc_adv_norm = _normalize_advantages_like_update(
        cfg,
        advantages_raw=mc_advantages,
        stage_ids=np.asarray(heldout_views.training_view.stage_ids, dtype=np.int64),
    )
    gae_adv_norm = _normalize_advantages_like_update(
        cfg,
        advantages_raw=gae_advantages,
        stage_ids=np.asarray(heldout_views.training_view.stage_ids, dtype=np.int64),
    )

    rng = np.random.default_rng(int(seed) + 233)
    sample_count = min(max(int(sample_rows), 1), int(eligible.size))
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
    labels, action_tensor, _eval_actions = _stage_action_samples(
        learner.actor,
        stage_id=int(stage_id),
        selected_local=selected_local,
        rollout_actions=sampled_actions,
        sample_count=sample_count,
        num_agents=num_agents,
        policy_samples=int(policy_action_samples),
        device=device,
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
        future_random_seed=int(seed) + 881_003,
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
    q_adv_norm_all = _standardize_np(q_adv.reshape(-1)).reshape(q_adv.shape)
    q_rollout_adv = q_adv[:, 0]
    q_rollout_adv_norm = _standardize_np(q_rollout_adv)

    selected_mc_raw = mc_advantages.index_select(0, selected_global_t).detach().cpu().numpy().astype(np.float64)
    selected_mc_norm = mc_adv_norm.index_select(0, selected_global_t).detach().cpu().numpy().astype(np.float64)
    selected_gae_raw = gae_advantages.index_select(0, selected_global_t).detach().cpu().numpy().astype(np.float64)
    selected_gae_norm = gae_adv_norm.index_select(0, selected_global_t).detach().cpu().numpy().astype(np.float64)
    selected_value = final_values.index_select(0, selected_global_t).detach().cpu().numpy().astype(np.float64)
    selected_mc_return = mc_returns.index_select(0, selected_global_t).detach().cpu().numpy().astype(np.float64)
    q_value_adv = q_mean - selected_value.reshape(-1, 1)
    q_rollout_value_adv = q_value_adv[:, 0]
    q_row_mean_minus_value = q_mean.mean(axis=1) - selected_value
    q_mean_adv_scale = float(np.std(q_adv.reshape(-1)))
    q_value_baseline_error_scale = float(np.std(q_row_mean_minus_value))

    entropy_coef = float(learner.entropy_coef_by_stage[int(stage_id)])

    lr_values = [float(x) for x in (probe_lrs if probe_lrs else [float(probe_lr)])]
    if not lr_values:
        lr_values = [float(probe_lr)]

    def _run_actor_probes_for_lr(lr_value: float) -> dict[str, Any]:
        mc_probe = _probe_actor_update(
            learner,
            stage_id=int(stage_id),
            selected_local=selected_local,
            candidate_actions=policy_action_tensor,
            selected_advantage=selected_mc_norm,
            candidate_advantage=None,
            num_agents=int(num_agents),
            lr=float(lr_value),
            optimizer_name=str(probe_optimizer),
            entropy_coef=entropy_coef,
            clip_ratio=float(learner.clip_ratio),
        )
        gae_probe = _probe_actor_update(
            learner,
            stage_id=int(stage_id),
            selected_local=selected_local,
            candidate_actions=policy_action_tensor,
            selected_advantage=selected_gae_norm,
            candidate_advantage=None,
            num_agents=int(num_agents),
            lr=float(lr_value),
            optimizer_name=str(probe_optimizer),
            entropy_coef=entropy_coef,
            clip_ratio=float(learner.clip_ratio),
        )
        qpi_selected_probe = _probe_actor_update(
            learner,
            stage_id=int(stage_id),
            selected_local=selected_local,
            candidate_actions=policy_action_tensor,
            selected_advantage=q_rollout_adv_norm,
            candidate_advantage=None,
            num_agents=int(num_agents),
            lr=float(lr_value),
            optimizer_name=str(probe_optimizer),
            entropy_coef=0.0,
            clip_ratio=float(learner.clip_ratio),
        )
        qpi_all_probe = _probe_actor_update(
            learner,
            stage_id=int(stage_id),
            selected_local=selected_local,
            candidate_actions=policy_action_tensor,
            selected_advantage=None,
            candidate_advantage=q_adv_norm_all,
            num_agents=int(num_agents),
            lr=float(lr_value),
            optimizer_name=str(probe_optimizer),
            entropy_coef=0.0,
            clip_ratio=float(learner.clip_ratio),
        )
        return {
            "mc_adv_update": _strip_actor_probe(mc_probe, q_mean=q_mean, q_adv=q_adv),
            "gae_adv_update": _strip_actor_probe(gae_probe, q_mean=q_mean, q_adv=q_adv),
            "qpi_selected_update": _strip_actor_probe(qpi_selected_probe, q_mean=q_mean, q_adv=q_adv),
            "qpi_all_candidate_update": _strip_actor_probe(qpi_all_probe, q_mean=q_mean, q_adv=q_adv),
        }

    actor_probe_by_lr = {f"{float(lr_value):.8g}": _run_actor_probes_for_lr(float(lr_value)) for lr_value in lr_values}
    primary_lr_key = f"{float(lr_values[0]):.8g}"

    return {
        "enabled": True,
        "sample_rows": int(sample_count),
        "policy_action_samples": int(policy_action_count),
        "continuations": int(continuations_i),
        "follow": "deterministic" if bool(follow_deterministic) else "stochastic",
        "future_random_mode": str(future_random_mode),
        "action_labels": policy_labels,
        "alignment": {
            "corr_mc_raw_vs_qpi_rollout_adv": _corr(selected_mc_raw, q_rollout_adv),
            "corr_mc_norm_vs_qpi_rollout_adv": _corr(selected_mc_norm, q_rollout_adv),
            "sign_agree_mc_norm_qpi_rollout_adv": float(
                np.mean(np.sign(selected_mc_norm) == np.sign(q_rollout_adv))
            ),
            "corr_gae_raw_vs_qpi_rollout_adv": _corr(selected_gae_raw, q_rollout_adv),
            "corr_gae_norm_vs_qpi_rollout_adv": _corr(selected_gae_norm, q_rollout_adv),
            "sign_agree_gae_norm_qpi_rollout_adv": float(
                np.mean(np.sign(selected_gae_norm) == np.sign(q_rollout_adv))
            ),
            "corr_mc_norm_vs_gae_norm": _corr(selected_mc_norm, selected_gae_norm),
            "corr_value_vs_qpi_rollout_adv": _corr(selected_value, q_rollout_adv),
            "corr_mc_return_vs_qpi_rollout_adv": _corr(selected_mc_return, q_rollout_adv),
            "corr_qpi_mean_adv_vs_qpi_value_adv_all": _corr(q_adv.reshape(-1), q_value_adv.reshape(-1)),
            "sign_agree_qpi_mean_adv_vs_qpi_value_adv_all": float(
                np.mean(np.sign(q_adv.reshape(-1)) == np.sign(q_value_adv.reshape(-1)))
            ),
            "corr_qpi_mean_adv_vs_qpi_value_adv_selected": _corr(q_rollout_adv, q_rollout_value_adv),
            "sign_agree_qpi_mean_adv_vs_qpi_value_adv_selected": float(
                np.mean(np.sign(q_rollout_adv) == np.sign(q_rollout_value_adv))
            ),
            "q_value_baseline_error_to_action_gap_scale": float(
                q_value_baseline_error_scale / max(q_mean_adv_scale, 1.0e-12)
            ),
        },
        "summary": {
            "qpi_rollout_adv": _summ(q_rollout_adv),
            "qpi_rollout_value_adv": _summ(q_rollout_value_adv),
            "qpi_mean_adv_all": _summ(q_adv),
            "qpi_value_adv_all": _summ(q_value_adv),
            "q_row_mean_minus_value": _summ(q_row_mean_minus_value),
            "q_mean_adv_scale": q_mean_adv_scale,
            "q_value_baseline_error_scale": q_value_baseline_error_scale,
            "mc_adv_raw": _summ(selected_mc_raw),
            "mc_adv_norm": _summ(selected_mc_norm),
            "gae_adv_raw": _summ(selected_gae_raw),
            "gae_adv_norm": _summ(selected_gae_norm),
            "value": _summ(selected_value),
            "mc_return": _summ(selected_mc_return),
            "q_mean": _summ(q_mean),
            "return_cube": _summ(ret_cube),
        },
        "actor_probe": actor_probe_by_lr[primary_lr_key],
        "actor_probe_by_lr": actor_probe_by_lr,
        "rows": [
            {
                "stage_sample": int(selected_np[i]),
                "transition_index": int(selected_transitions[i]),
                "history_row": int(history_rows[i]),
                "step": int(step_indices_all[selected_np][i]),
                "horizon": int(horizons[i]),
                "mc_adv_norm": float(selected_mc_norm[i]),
                "gae_adv_norm": float(selected_gae_norm[i]),
                "qpi_rollout_adv": float(q_rollout_adv[i]),
                "qpi_rollout_value_adv": float(q_rollout_value_adv[i]),
                "value": float(selected_value[i]),
                "mc_return": float(selected_mc_return[i]),
                "q_mean": [float(x) for x in q_mean[i].tolist()],
            }
            for i in range(sample_count)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(STAGE_ID), default="sat")
    parser.add_argument("--config", required=True)
    parser.add_argument("--reward_mode", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--rollout_env_steps", type=int, default=250)
    parser.add_argument("--train_rollouts", type=int, default=4)
    parser.add_argument("--critic_epochs", type=int, default=40)
    parser.add_argument("--critic_minibatches", type=int, default=8)
    parser.add_argument(
        "--critic_accumulate_full_batch",
        action="store_true",
        help="Use critic_minibatches as microbatches, accumulate gradients, and do one optimizer step per epoch.",
    )
    parser.add_argument("--critic_lr", type=float, default=None)
    parser.add_argument("--critic_value_mode", choices=["relational", "global_only", "global_linear"], default=None)
    parser.add_argument("--critic_message_layers", type=int, default=None)
    parser.add_argument("--critic_value_head_hidden", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--fit_rounds", type=int, default=1)
    parser.add_argument("--fit_round_epochs", type=int, default=None)
    parser.add_argument("--fit_round_train_rollouts", type=int, default=None)
    parser.add_argument("--target", choices=["mc", "train_gae"], default="train_gae")
    parser.add_argument("--oracle_rows", type=int, default=16)
    parser.add_argument("--policy_action_samples", type=int, default=4)
    parser.add_argument("--vpi_rows", type=int, default=8)
    parser.add_argument("--vpi_policy_action_samples", type=int, default=4)
    parser.add_argument("--vpi_continuations", type=int, default=4)
    parser.add_argument("--vpi_follow", choices=["deterministic", "stochastic"], default="stochastic")
    parser.add_argument("--vpi_future_random", choices=["copy", "resample"], default="resample")
    parser.add_argument("--advantage_probe_rows", type=int, default=8)
    parser.add_argument("--advantage_policy_action_samples", type=int, default=4)
    parser.add_argument("--advantage_continuations", type=int, default=4)
    parser.add_argument("--advantage_follow", choices=["deterministic", "stochastic"], default="stochastic")
    parser.add_argument("--advantage_future_random", choices=["copy", "resample"], default="resample")
    parser.add_argument("--advantage_probe_lr", type=float, default=None)
    parser.add_argument("--advantage_probe_lrs", default="")
    parser.add_argument("--advantage_probe_optimizer", choices=["adam", "sgd"], default="adam")
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

    learner, _actor, _critic = _make_learner(cfg, device=device, stage_id=stage_id)
    group = make_structured_driver_group(cfg, num_envs=int(args.num_envs), backend="sync", mode="train")
    try:
        learner.bind_native_runtime_contract(group)
        learner.begin_native_rollout(group, rollout_env_steps=int(args.rollout_env_steps), num_envs=int(args.num_envs))
        fit_rounds = max(int(args.fit_rounds), 1)
        fit_round_epochs = int(args.fit_round_epochs) if args.fit_round_epochs is not None else int(args.critic_epochs)
        fit_round_train_rollouts = (
            int(args.fit_round_train_rollouts) if args.fit_round_train_rollouts is not None else int(args.train_rollouts)
        )
        train_world, train_target, train_summaries = _collect_stage_bank(
            learner,
            group,
            stage_id=stage_id,
            rollout_env_steps=int(args.rollout_env_steps),
            rollouts=int(args.train_rollouts),
            seed_base=int(args.seed),
            device=device,
            target=str(args.target),
        )
        reset_many = getattr(group, "reset_many", None)
        if callable(reset_many):
            reset_many([int(args.seed) + 9_000_000 + env for env in range(int(args.num_envs))])
        heldout_buffer, heldout_views, heldout_returns_all = _collect_one_rollout(
            learner,
            group,
            rollout_env_steps=int(args.rollout_env_steps),
            device=device,
            target=str(args.target),
        )
        heldout_stage = heldout_views.training_view.stage_batches[int(stage_id)]
        heldout_idx = torch.as_tensor(
            np.asarray(heldout_stage.transition_indices, dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        heldout_world = heldout_stage.world_batch
        heldout_target = heldout_returns_all.index_select(0, heldout_idx).detach()

        initial_pred_train, initial_train_stats = _eval_critic(
            learner,
            stage_id=stage_id,
            world_bank=train_world,
            target=train_target,
            batch_size=2048,
        )
        del initial_pred_train
        initial_pred_heldout, initial_heldout_stats = _eval_critic(
            learner,
            stage_id=stage_id,
            world_bank=heldout_world,
            target=heldout_target,
            batch_size=2048,
        )
        del initial_pred_heldout
        initial_critic_state = copy.deepcopy(learner.critic.state_dict())
        if fit_rounds <= 1:
            history = _train_critic_only(
                learner,
                stage_id=stage_id,
                train_world=train_world,
                train_target=train_target,
                heldout_world=heldout_world,
                heldout_target=heldout_target,
                epochs=int(args.critic_epochs),
                minibatches=int(args.critic_minibatches),
                lr=float(cfg.critic_lr),
                max_grad_norm=float(cfg.max_grad_norm),
                eval_every=int(args.eval_every),
                accumulate_full_batch=bool(args.critic_accumulate_full_batch),
            )
            fitted_eval_history: list[dict[str, float]] = []
        else:
            first_stats = _train_critic_bank_once(
                learner,
                stage_id=stage_id,
                train_world=train_world,
                train_target=train_target,
                epochs=int(fit_round_epochs),
                minibatches=int(args.critic_minibatches),
                lr=float(cfg.critic_lr),
                max_grad_norm=float(cfg.max_grad_norm),
            )
            first_stats["round"] = 1.0
            fitted_eval_history = [first_stats]
            fitted_eval_history.extend(
                _run_fixed_policy_fitted_eval(
                    learner,
                    group,
                    stage_id=stage_id,
                    rollout_env_steps=int(args.rollout_env_steps),
                    rounds=int(fit_rounds) - 1,
                    train_rollouts_per_round=int(fit_round_train_rollouts),
                    seed_base=int(args.seed),
                    device=device,
                    target=str(args.target),
                    epochs_per_round=int(fit_round_epochs),
                    minibatches=int(args.critic_minibatches),
                    lr=float(cfg.critic_lr),
                    max_grad_norm=float(cfg.max_grad_norm),
                    start_round=2,
                )
            )
            history = []
        final_pred_train, final_train_stats = _eval_critic(
            learner,
            stage_id=stage_id,
            world_bank=train_world,
            target=train_target,
            batch_size=2048,
        )
        del final_pred_train
        final_pred_heldout, final_heldout_stats = _eval_critic(
            learner,
            stage_id=stage_id,
            world_bank=heldout_world,
            target=heldout_target,
            batch_size=2048,
        )
        del final_pred_heldout
        oracle_probe = _heldout_oracle_probe(
            learner,
            cfg=cfg,
            stage_id=stage_id,
            num_envs=int(args.num_envs),
            heldout_views=heldout_views,
            heldout_returns=heldout_returns_all,
            reward_mode=str(args.reward_mode or getattr(cfg, "reward_mode", "env_reward") or "env_reward"),
            sample_rows=int(args.oracle_rows),
            policy_action_samples=int(args.policy_action_samples),
            min_horizon=int(args.min_horizon),
            branch_horizon_cap=int(args.branch_horizon_cap),
            seed=int(args.seed),
            device=device,
        )
        vpi_ceiling_probe = _heldout_vpi_ceiling_probe(
            learner,
            cfg=cfg,
            stage_id=stage_id,
            num_envs=int(args.num_envs),
            heldout_views=heldout_views,
            reward_mode=str(args.reward_mode or getattr(cfg, "reward_mode", "env_reward") or "env_reward"),
            sample_rows=int(args.vpi_rows),
            policy_action_samples=int(args.vpi_policy_action_samples),
            continuations=int(args.vpi_continuations),
            min_horizon=int(args.min_horizon),
            branch_horizon_cap=int(args.branch_horizon_cap),
            seed=int(args.seed),
            device=device,
            initial_critic_state=initial_critic_state,
            follow_deterministic=(str(args.vpi_follow) == "deterministic"),
            future_random_mode=str(args.vpi_future_random),
        )
        probe_lrs = [
            float(item.strip())
            for item in str(args.advantage_probe_lrs).replace(";", ",").split(",")
            if item.strip()
        ]
        if not probe_lrs:
            probe_lrs = [
                float(args.advantage_probe_lr) if args.advantage_probe_lr is not None else float(cfg.actor_lr)
            ]
        advantage_actor_probe = _heldout_advantage_actor_probe(
            learner,
            cfg=cfg,
            stage_id=stage_id,
            num_envs=int(args.num_envs),
            heldout_buffer=heldout_buffer,
            heldout_views=heldout_views,
            reward_mode=str(args.reward_mode or getattr(cfg, "reward_mode", "env_reward") or "env_reward"),
            sample_rows=int(args.advantage_probe_rows),
            policy_action_samples=int(args.advantage_policy_action_samples),
            continuations=int(args.advantage_continuations),
            min_horizon=int(args.min_horizon),
            branch_horizon_cap=int(args.branch_horizon_cap),
            seed=int(args.seed),
            device=device,
            follow_deterministic=(str(args.advantage_follow) == "deterministic"),
            probe_lr=float(probe_lrs[0]),
            probe_lrs=probe_lrs,
            probe_optimizer=str(args.advantage_probe_optimizer),
            future_random_mode=str(args.advantage_future_random),
        )
        payload = {
            "config": str(args.config),
            "stage": str(args.stage),
            "stage_id": int(stage_id),
            "reward_mode": str(args.reward_mode or getattr(cfg, "reward_mode", "env_reward") or "env_reward"),
            "target": str(args.target),
            "num_envs": int(args.num_envs),
            "rollout_env_steps": int(args.rollout_env_steps),
            "train_rollouts": int(args.train_rollouts),
            "train_samples": int(train_target.numel()),
            "heldout_samples": int(heldout_target.numel()),
            "critic_epochs": int(args.critic_epochs),
            "critic_minibatches": int(args.critic_minibatches),
            "critic_lr": float(cfg.critic_lr),
            "critic_message_layers": int(getattr(cfg, "critic_message_layers", 0)),
            "fit_rounds": int(fit_rounds),
            "fit_round_epochs": int(fit_round_epochs),
            "fit_round_train_rollouts": int(fit_round_train_rollouts),
            "train_rollout_summaries": train_summaries,
            "initial": {
                "train": initial_train_stats,
                "heldout": initial_heldout_stats,
            },
            "history": history,
            "final": {
                "train": final_train_stats,
                "heldout": final_heldout_stats,
            },
            "heldout_oracle_probe_after_critic": oracle_probe,
            "heldout_vpi_ceiling_probe": vpi_ceiling_probe,
            "fitted_eval_history": fitted_eval_history,
            "heldout_advantage_actor_probe": advantage_actor_probe,
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "stage": str(args.stage),
                    "target": str(args.target),
                    "fit_rounds": int(fit_rounds),
                    "train_samples": int(train_target.numel()),
                    "heldout_samples": int(heldout_target.numel()),
                    "initial_heldout_ev": initial_heldout_stats["ev"],
                    "final_heldout_ev": final_heldout_stats["ev"],
                    "initial_heldout_mse": initial_heldout_stats["mse"],
                    "final_heldout_mse": final_heldout_stats["mse"],
                    "oracle_probe": {
                        k: v
                        for k, v in oracle_probe.items()
                        if k
                        in {
                            "enabled",
                            "corr_return_vs_oracle_policy_adv",
                            "corr_value_vs_oracle_policy_adv",
                            "corr_residual_vs_oracle_policy_adv",
                            "sign_agree_residual_oracle_policy_adv",
                        }
                    },
                    "vpi_ceiling": {
                        "enabled": bool(vpi_ceiling_probe.get("enabled", False)),
                        "initial_ev": None
                        if vpi_ceiling_probe.get("initial") is None
                        else vpi_ceiling_probe["initial"]["ev"],
                        "final_ev": None
                        if vpi_ceiling_probe.get("final") is None
                        else vpi_ceiling_probe["final"]["ev"],
                        "initial_corr": None
                        if vpi_ceiling_probe.get("initial") is None
                        else vpi_ceiling_probe["initial"]["corr"],
                        "final_corr": None
                        if vpi_ceiling_probe.get("final") is None
                        else vpi_ceiling_probe["final"]["corr"],
                    },
                    "advantage_probe": {
                        "enabled": bool(advantage_actor_probe.get("enabled", False)),
                        "corr_mc_norm_vs_qpi": advantage_actor_probe.get("alignment", {}).get(
                            "corr_mc_norm_vs_qpi_rollout_adv"
                        ),
                        "corr_gae_norm_vs_qpi": advantage_actor_probe.get("alignment", {}).get(
                            "corr_gae_norm_vs_qpi_rollout_adv"
                        ),
                        "mc_q_shift": advantage_actor_probe.get("actor_probe", {})
                        .get("mc_adv_update", {})
                        .get("candidate_shift_vs_qpi", {})
                        .get("mean_qadv_times_delta_logprob"),
                        "gae_q_shift": advantage_actor_probe.get("actor_probe", {})
                        .get("gae_adv_update", {})
                        .get("candidate_shift_vs_qpi", {})
                        .get("mean_qadv_times_delta_logprob"),
                        "qpi_all_q_shift": advantage_actor_probe.get("actor_probe", {})
                        .get("qpi_all_candidate_update", {})
                        .get("candidate_shift_vs_qpi", {})
                        .get("mean_qadv_times_delta_logprob"),
                    },
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
