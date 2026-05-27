from __future__ import annotations

import argparse
import copy
import json
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
from sagin_marl.rl import structured_sat_actor_schema as sat_schema
from sagin_marl.rl.stage_mcgae import (
    STAGE_ID,
    force_single_stage_config as _force_single_stage_config,
    make_stage_optimizers as _make_stage_optimizers,
    set_seed as _set_seed,
    stage_optimizer_params as _stage_optimizer_params,
)
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver_group

from scripts.diagnose_reward_action_sensitivity import (
    _branch_returns_from_history_for_modes,
    _corr,
    _summ,
)


def _stage_action_samples(
    actor: torch.nn.Module,
    *,
    stage_id: int,
    selected_local: Any,
    rollout_actions: torch.Tensor,
    sample_count: int,
    num_agents: int,
    policy_samples: int,
    device: torch.device,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    labels = ["ref", "rollout_sample"]
    actions: list[torch.Tensor] = []
    with torch.no_grad():
        if int(stage_id) == 0:
            ref_out = actor.act_accel(selected_local, deterministic=True)
            ref_actions = ref_out.action.reshape(sample_count, num_agents, -1).to(device=device, dtype=torch.float32)
            actions.append(ref_actions)
            actions.append(rollout_actions.to(device=device, dtype=torch.float32).reshape(sample_count, num_agents, -1))
            for i in range(max(int(policy_samples), 0)):
                alt = actor.act_accel(selected_local, deterministic=False)
                actions.append(alt.action.reshape(sample_count, num_agents, -1).to(device=device, dtype=torch.float32))
                labels.append(f"policy_sample_{i}")
        elif int(stage_id) == 1:
            ref_out = actor.act_sat(selected_local, deterministic=True)
            ref_actions = ref_out.subset_index.reshape(sample_count, num_agents).to(device=device, dtype=torch.long)
            actions.append(ref_actions)
            actions.append(rollout_actions.to(device=device, dtype=torch.long).reshape(sample_count, num_agents))
            for i in range(max(int(policy_samples), 0)):
                alt = actor.act_sat(selected_local, deterministic=False)
                actions.append(alt.subset_index.reshape(sample_count, num_agents).to(device=device, dtype=torch.long))
                labels.append(f"policy_sample_{i}")
        elif int(stage_id) == 2:
            ref_out = actor.act_bw(selected_local, deterministic=True)
            ref_actions = ref_out.action.reshape(sample_count, num_agents, -1).to(device=device, dtype=torch.float32)
            actions.append(ref_actions)
            actions.append(rollout_actions.to(device=device, dtype=torch.float32).reshape(sample_count, num_agents, -1))
            for i in range(max(int(policy_samples), 0)):
                alt = actor.act_bw(selected_local, deterministic=False)
                actions.append(alt.action.reshape(sample_count, num_agents, -1).to(device=device, dtype=torch.float32))
                labels.append(f"policy_sample_{i}")
        else:
            raise ValueError(f"unsupported stage_id={stage_id!r}")
    action_tensor = torch.stack(actions, dim=1)
    eval_actions = actions[1]
    return labels, action_tensor, eval_actions


def _normalize_advantages_like_update(
    cfg: Any,
    *,
    advantages_raw: torch.Tensor,
    stage_ids: np.ndarray,
) -> torch.Tensor:
    advantages = advantages_raw.clone()
    if not bool(getattr(cfg, "actor_advantage_normalize_enabled", True)) or int(advantages.numel()) <= 1:
        return advantages
    if bool(getattr(cfg, "stagewise_advantage_norm_enabled", False)):
        for stage_id in (0, 1, 2):
            idx_np = np.flatnonzero(np.asarray(stage_ids, dtype=np.int64) == int(stage_id))
            if idx_np.size <= 1:
                continue
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=advantages.device)
            vals = advantages.index_select(0, idx)
            vals = (vals - vals.mean()) / vals.std(unbiased=False).clamp_min(1.0e-8)
            advantages.index_copy_(0, idx, vals)
        return advantages
    return (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1.0e-8)


def _selected_only_ppo_delta(
    learner: StructuredMAPPO,
    *,
    stage_id: int,
    selected_local: Any,
    joint_actions: torch.Tensor,
    num_agents: int,
    advantages: torch.Tensor,
    old_logprob: torch.Tensor,
) -> tuple[np.ndarray, dict[str, float]]:
    actor_state = copy.deepcopy(learner.actor.state_dict())
    params = _stage_optimizer_params(learner.actor, int(stage_id))
    if not params:
        return np.zeros((int(joint_actions.shape[0]),), dtype=np.float64), {
            "selected_only_loss": 0.0,
            "selected_only_policy_loss": 0.0,
            "selected_only_entropy": 0.0,
            "selected_only_grad_norm": 0.0,
        }
    opt = torch.optim.Adam(params, lr=float(learner.cfg.actor_lr))
    with torch.no_grad():
        pre_logprob, _pre_entropy, _pre_out = learner._stage_actor_eval_from_batch(
            int(stage_id),
            selected_local,
            joint_actions,
            int(num_agents),
        )
    new_logprob, entropy, _actor_out = learner._stage_actor_eval_from_batch(
        int(stage_id),
        selected_local,
        joint_actions,
        int(num_agents),
    )
    adv = advantages.to(device=learner.device, dtype=new_logprob.dtype).reshape_as(new_logprob).detach()
    old_lp = old_logprob.to(device=learner.device, dtype=new_logprob.dtype).reshape_as(new_logprob).detach()
    ratio = torch.exp(torch.clamp(new_logprob - old_lp, min=-20.0, max=20.0))
    clipped = torch.clamp(ratio, 1.0 - float(learner.clip_ratio), 1.0 + float(learner.clip_ratio))
    policy_loss = -torch.minimum(ratio * adv, clipped * adv).mean()
    entropy_mean = entropy.mean() if int(entropy.numel()) > 0 else torch.zeros((), device=learner.device)
    loss = policy_loss - float(learner.entropy_coef_by_stage[int(stage_id)]) * entropy_mean
    opt.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(params, float(learner.max_grad_norm))
    opt.step()
    with torch.no_grad():
        post_logprob, _post_entropy, _post_out = learner._stage_actor_eval_from_batch(
            int(stage_id),
            selected_local,
            joint_actions,
            int(num_agents),
        )
    delta = (post_logprob - pre_logprob).detach().cpu().numpy()
    metrics = {
        "selected_only_loss": float(loss.detach().cpu().item()),
        "selected_only_policy_loss": float(policy_loss.detach().cpu().item()),
        "selected_only_entropy": float(entropy_mean.detach().cpu().item()),
        "selected_only_grad_norm": float(torch.as_tensor(grad_norm).detach().cpu().item()),
    }
    learner.actor.load_state_dict(actor_state, strict=True)
    return np.asarray(delta, dtype=np.float64), metrics


def _per_row_ppo_delta(
    learner: StructuredMAPPO,
    *,
    stage_id: int,
    selected_local: Any,
    joint_actions: torch.Tensor,
    num_agents: int,
    advantages: torch.Tensor,
    old_logprob: torch.Tensor,
) -> np.ndarray:
    row_count = int(joint_actions.shape[0])
    if row_count <= 0:
        return np.zeros((0,), dtype=np.float64)
    deltas: list[float] = []
    agent_offsets = torch.arange(int(num_agents), dtype=torch.long, device=learner.device)
    for row_idx in range(row_count):
        local_idx = int(row_idx) * int(num_agents) + agent_offsets
        local_i = _index_dataclass(selected_local, local_idx)
        action_i = joint_actions[row_idx : row_idx + 1]
        adv_i = advantages[row_idx : row_idx + 1]
        old_i = old_logprob[row_idx : row_idx + 1]
        delta_i, _metrics = _selected_only_ppo_delta(
            learner,
            stage_id=int(stage_id),
            selected_local=local_i,
            joint_actions=action_i,
            num_agents=int(num_agents),
            advantages=adv_i,
            old_logprob=old_i,
        )
        deltas.append(float(delta_i.reshape(-1)[0]) if delta_i.size else 0.0)
    return np.asarray(deltas, dtype=np.float64)


def _flat_grad_vector(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> torch.Tensor:
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    pieces: list[torch.Tensor] = []
    for param, grad in zip(params, grads):
        if grad is None:
            pieces.append(torch.zeros(param.numel(), dtype=torch.float32, device=loss.device))
        else:
            pieces.append(grad.detach().reshape(-1).to(dtype=torch.float32))
    if not pieces:
        return torch.zeros((0,), dtype=torch.float32, device=loss.device)
    return torch.cat(pieces, dim=0)


def _per_row_policy_grad_matrix(
    learner: StructuredMAPPO,
    *,
    stage_id: int,
    selected_local: Any,
    joint_actions: torch.Tensor,
    num_agents: int,
    weights: np.ndarray,
) -> torch.Tensor:
    params = _stage_optimizer_params(learner.actor, int(stage_id))
    row_count = int(joint_actions.shape[0])
    if row_count <= 0 or not params:
        return torch.zeros((0, 0), dtype=torch.float32)
    weights_t = torch.as_tensor(np.asarray(weights, dtype=np.float32), dtype=torch.float32, device=learner.device)
    agent_offsets = torch.arange(int(num_agents), dtype=torch.long, device=learner.device)
    rows: list[torch.Tensor] = []
    for row_idx in range(row_count):
        local_idx = int(row_idx) * int(num_agents) + agent_offsets
        local_i = _index_dataclass(selected_local, local_idx)
        action_i = joint_actions[row_idx : row_idx + 1]
        logprob_i, _entropy_i, _out_i = learner._stage_actor_eval_from_batch(
            int(stage_id),
            local_i,
            action_i,
            int(num_agents),
        )
        loss_i = -(weights_t[row_idx].detach() * logprob_i.reshape(()))
        rows.append(_flat_grad_vector(loss_i, params).cpu())
    return torch.stack(rows, dim=0)


def _cosine_matrix(grad_matrix: torch.Tensor) -> np.ndarray:
    if int(grad_matrix.numel()) <= 0:
        return np.zeros((0, 0), dtype=np.float64)
    g = grad_matrix.to(dtype=torch.float32)
    norm = torch.linalg.vector_norm(g, dim=1, keepdim=True).clamp_min(1.0e-12)
    unit = g / norm
    return (unit @ unit.T).detach().cpu().numpy().astype(np.float64)


def _cos_values_for_pairs(cos: np.ndarray, labels: list[str], *, within: bool) -> np.ndarray:
    n = int(cos.shape[0])
    values: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            same = labels[i] == labels[j]
            if same == bool(within):
                values.append(float(cos[i, j]))
    return np.asarray(values, dtype=np.float64)


def _cos_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return {"count": 0.0, "mean": 0.0, "p10": 0.0, "p50": 0.0, "conflict_frac": 0.0}
    return {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "p10": float(np.percentile(arr, 10.0)),
        "p50": float(np.percentile(arr, 50.0)),
        "conflict_frac": float(np.mean(arr < 0.0)),
    }


def _group_conflict_report(
    *,
    cos: np.ndarray,
    labels_by_kind: dict[str, list[str]],
    adv: np.ndarray,
    oracle: np.ndarray,
) -> dict[str, Any]:
    n = int(cos.shape[0])
    all_pairs: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.append(float(cos[i, j]))
    report: dict[str, Any] = {"overall": _cos_summary(np.asarray(all_pairs, dtype=np.float64)), "groups": {}}
    for kind, labels in labels_by_kind.items():
        if len(labels) != n:
            continue
        within = _cos_values_for_pairs(cos, labels, within=True)
        cross = _cos_values_for_pairs(cos, labels, within=False)
        counts: dict[str, int] = {}
        adv_means: dict[str, list[float]] = {}
        oracle_means: dict[str, list[float]] = {}
        for label, adv_i, oracle_i in zip(labels, adv, oracle):
            label_s = str(label)
            counts[label_s] = int(counts.get(label_s, 0) + 1)
            adv_means.setdefault(label_s, []).append(float(adv_i))
            oracle_means.setdefault(label_s, []).append(float(oracle_i))
        label_payload = {
            label: {
                "count": int(count),
                "adv_mean": float(np.mean(adv_means[label])),
                "oracle_policy_adv_mean": float(np.mean(oracle_means[label])),
            }
            for label, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        }
        within_summary = _cos_summary(within)
        cross_summary = _cos_summary(cross)
        report["groups"][kind] = {
            "within": within_summary,
            "cross": cross_summary,
            "within_minus_cross_mean": float(within_summary["mean"] - cross_summary["mean"]),
            "labels": label_payload,
        }
    return report


def _safe_quantile_bin(values: np.ndarray, *, low_name: str, high_name: str) -> list[str]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return ["unknown" for _ in arr]
    med = float(np.median(finite))
    return [high_name if np.isfinite(v) and float(v) >= med else low_name for v in arr]


def _sat_action_global_sets(
    local_state: Any,
    *,
    action: torch.Tensor,
    sample_count: int,
    num_agents: int,
) -> list[tuple[int, ...]]:
    candidate_ids = local_state.candidate_sat_ids.detach().cpu().to(dtype=torch.long)
    members = local_state.subset_members.detach().cpu().to(dtype=torch.long)
    if members.ndim == 3:
        members_base = members[0]
    else:
        members_base = members
    action_cpu = action.detach().cpu().to(dtype=torch.long).reshape(sample_count * num_agents)
    chosen = action_cpu.clamp(min=0, max=max(int(members_base.shape[0]) - 1, 0))
    selected_members = members_base.index_select(0, chosen)
    safe_members = selected_members.clamp_min(0)
    rows = torch.arange(sample_count * num_agents, dtype=torch.long).view(-1, 1).expand_as(safe_members)
    safe_slots = safe_members.clamp(max=max(int(candidate_ids.shape[1]) - 1, 0))
    selected_global = candidate_ids[rows, safe_slots]
    selected_global = torch.where(selected_members >= 0, selected_global, torch.full_like(selected_global, -1))
    selected_global = selected_global.reshape(sample_count, num_agents, -1)
    out: list[tuple[int, ...]] = []
    for row_idx in range(sample_count):
        ids = sorted({int(x) for x in selected_global[row_idx].reshape(-1).tolist() if int(x) >= 0})
        out.append(tuple(ids))
    return out


def _sat_row_group_labels(
    *,
    local_state: Any,
    action_tensor: torch.Tensor,
    step_indices: np.ndarray,
    cfg: Any,
    sample_count: int,
    num_agents: int,
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    candidate_ids = local_state.candidate_sat_ids.detach().cpu().to(dtype=torch.long).reshape(sample_count, num_agents, -1)
    valid_mask = (
        local_state.sat_mask.detach().cpu().to(dtype=torch.bool)
        & local_state.sat_valid_mask.detach().cpu().to(dtype=torch.bool)
    ).reshape(sample_count, num_agents, -1)
    sat_tokens = local_state.sat_tokens.detach().cpu().to(dtype=torch.float32).reshape(sample_count, num_agents, -1, local_state.sat_tokens.shape[-1])
    demand = local_state.demand_features.detach().cpu().to(dtype=torch.float32).reshape(sample_count, num_agents, -1)
    visible_sets: list[tuple[int, ...]] = []
    visible_counts: list[float] = []
    arrival_pressure: list[float] = []
    load_pressure: list[float] = []
    queue_pressure: list[float] = []
    for row_idx in range(sample_count):
        ids = sorted(
            {
                int(candidate_ids[row_idx, u, k].item())
                for u in range(num_agents)
                for k in range(int(candidate_ids.shape[-1]))
                if bool(valid_mask[row_idx, u, k].item()) and int(candidate_ids[row_idx, u, k].item()) >= 0
            }
        )
        visible_sets.append(tuple(ids))
        visible_counts.append(float(len(ids)))
        arrival_pressure.append(float(demand[row_idx, :, sat_schema.DEMAND_CELL_EXPECTED_ARRIVAL_STEPS_SUM].mean().item()))
        mask_f = valid_mask[row_idx].to(dtype=torch.float32)
        denom = float(mask_f.sum().clamp_min(1.0).item())
        load_pressure.append(
            float((sat_tokens[row_idx, :, :, sat_schema.SAT_LAST_SELECTED_LOAD_FRAC] * mask_f).sum().item() / denom)
        )
        queue_pressure.append(float((sat_tokens[row_idx, :, :, sat_schema.SAT_QUEUE_FILL] * mask_f).sum().item() / denom))
    ref_sets = _sat_action_global_sets(
        local_state,
        action=action_tensor[:, 0],
        sample_count=sample_count,
        num_agents=num_agents,
    )
    sample_sets = _sat_action_global_sets(
        local_state,
        action=action_tensor[:, 1],
        sample_count=sample_count,
        num_agents=num_agents,
    )
    jaccards: list[float] = []
    for ref, sample in zip(ref_sets, sample_sets):
        ref_s = set(ref)
        sample_s = set(sample)
        union = ref_s | sample_s
        jaccards.append(float(len(ref_s & sample_s) / max(len(union), 1)))
    t_steps = max(float(getattr(cfg, "T_steps", 1) or 1), 1.0)
    step_frac = np.asarray(step_indices, dtype=np.float64) / t_steps
    step_bin = ["early" if x < 0.33 else "mid" if x < 0.66 else "late" for x in step_frac]
    labels = {
        "step_bin": step_bin,
        "arrival_bin": _safe_quantile_bin(np.asarray(arrival_pressure), low_name="arrival_low", high_name="arrival_high"),
        "visible_count_bin": _safe_quantile_bin(np.asarray(visible_counts), low_name="visible_few", high_name="visible_many"),
        "sat_load_bin": _safe_quantile_bin(np.asarray(load_pressure), low_name="load_low", high_name="load_high"),
        "sat_queue_bin": _safe_quantile_bin(np.asarray(queue_pressure), low_name="sat_queue_low", high_name="sat_queue_high"),
        "sample_ref_jaccard_bin": _safe_quantile_bin(np.asarray(jaccards), low_name="sample_ref_low_overlap", high_name="sample_ref_high_overlap"),
        "visible_sat_set": ["|".join(map(str, ids)) if ids else "none" for ids in visible_sets],
        "ref_selected_set": ["|".join(map(str, ids)) if ids else "none" for ids in ref_sets],
        "sample_selected_set": ["|".join(map(str, ids)) if ids else "none" for ids in sample_sets],
    }
    row_meta = [
        {
            "visible_sat_set": labels["visible_sat_set"][i],
            "ref_selected_set": labels["ref_selected_set"][i],
            "sample_selected_set": labels["sample_selected_set"][i],
            "visible_count": float(visible_counts[i]),
            "arrival_pressure": float(arrival_pressure[i]),
            "sat_load_pressure": float(load_pressure[i]),
            "sat_queue_pressure": float(queue_pressure[i]),
            "sample_ref_jaccard": float(jaccards[i]),
            "step_bin": labels["step_bin"][i],
        }
        for i in range(sample_count)
    ]
    return labels, row_meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage", choices=sorted(STAGE_ID), required=True)
    parser.add_argument("--reward_mode", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--rollout_env_steps", type=int, default=250)
    parser.add_argument("--sample_rows", type=int, default=16)
    parser.add_argument("--policy_action_samples", type=int, default=4)
    parser.add_argument("--min_horizon", type=int, default=20)
    parser.add_argument("--branch_horizon_cap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=45200)
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--accel_log_std_init", type=float, default=None)
    parser.add_argument("--conflict_analysis", action="store_true")
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
    if args.accel_log_std_init is not None:
        cfg.accel_log_std_init = float(args.accel_log_std_init)

    bundle = build_structured_modules_from_config(cfg)
    actor = bundle.actor.to(device)
    critic = bundle.critic.to(device) if bundle.critic is not None else None
    actor_optimizer = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=float(cfg.actor_lr))
    actor_stage_optimizers = _make_stage_optimizers(actor, float(cfg.actor_lr))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(cfg.critic_lr)) if critic is not None else None
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
        actor_optimizer=actor_optimizer,
        actor_stage_optimizers=actor_stage_optimizers,
        critic_optimizer=critic_optimizer,
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
        if hasattr(group, "reset_many"):
            group.reset_many([int(args.seed) + i for i in range(int(args.num_envs))])
        learner.bind_native_runtime_contract(group)
        learner.begin_native_rollout(group, rollout_env_steps=int(args.rollout_env_steps), num_envs=int(args.num_envs))
        buffer = StructuredRolloutBuffer()
        learner.collect_env_horizon_native_tensor_policy(
            group,
            buffer=buffer,
            horizon=int(args.rollout_env_steps),
            deterministic=False,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        rollout_views = buffer.build_rollout_views(device)
        batch_view = rollout_views.training_view
        return_view = rollout_views.return_view
        learner._refresh_actor_old_logprobs_from_training_view(batch_view)
        value_override = learner._rollout_value_override_from_training_view(batch_view)
        learner._apply_rollout_value_override_to_views(
            batch_view=batch_view,
            return_view=return_view,
            value_override=value_override,
        )
        gae = learner.compute_returns_and_advantages(
            buffer,
            rollout_views.bootstrap_view,
            return_view=return_view,
            value_override=value_override,
        )
        advantages_raw = torch.as_tensor(gae["advantages"], dtype=torch.float32, device=device)
        returns = torch.as_tensor(gae["returns"], dtype=torch.float32, device=device)
        values = value_override.to(device=device, dtype=torch.float32)
        advantages = _normalize_advantages_like_update(
            cfg,
            advantages_raw=advantages_raw,
            stage_ids=np.asarray(batch_view.stage_ids, dtype=np.int64),
        )

        stage_batch = batch_view.stage_batches[stage_id]
        num_samples = int(stage_batch.num_samples)
        num_agents = int(stage_batch.num_agents)
        transition_indices = np.asarray(stage_batch.transition_indices, dtype=np.int64)
        history_rows_all = ((transition_indices - int(stage_id)) // 3).astype(np.int64)
        step_indices_all = history_rows_all // int(args.num_envs)
        horizons_all = np.maximum(int(cfg.T_steps) - step_indices_all, 1).astype(np.int64)
        eligible = np.flatnonzero(horizons_all >= max(int(args.min_horizon), 1))
        if eligible.size <= 0:
            raise RuntimeError(f"no eligible {args.stage} rows after min_horizon filtering.")
        rng = np.random.default_rng(int(args.seed) + 19)
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
            stage_id=stage_id,
            selected_local=selected_local,
            rollout_actions=sampled_actions,
            sample_count=sample_count,
            num_agents=num_agents,
            policy_samples=int(args.policy_action_samples),
            device=device,
        )
        action_count = int(action_tensor.shape[1])

        with torch.no_grad():
            old_logprob, old_entropy, _old_out = learner._stage_actor_eval_from_batch(
                stage_id,
                selected_local,
                eval_actions,
                int(num_agents),
            )

        history_rows = history_rows_all[selected_np]
        horizons = horizons_all[selected_np]
        if int(args.branch_horizon_cap) > 0:
            horizons = np.minimum(horizons, int(args.branch_horizon_cap)).astype(np.int64, copy=False)
        branch_rows = np.repeat(history_rows, action_count)
        branch_horizons = np.repeat(horizons, action_count)
        branch_actions = action_tensor.reshape(sample_count * action_count, int(num_agents), -1)
        reward_mode = str(args.reward_mode or getattr(cfg, "reward_mode", "env_reward") or "env_reward")
        returns_by_mode = _branch_returns_from_history_for_modes(
            learner,
            stage_id=stage_id,
            history_rows=branch_rows.tolist(),
            first_actions=branch_actions,
            horizons=branch_horizons.tolist(),
            reward_modes=[reward_mode],
        )
        branch_matrix = returns_by_mode[reward_mode].detach().cpu().numpy().reshape(sample_count, action_count)
        ref_return = branch_matrix[:, 0]
        sample_return = branch_matrix[:, 1]
        stochastic_returns = branch_matrix[:, 1:]
        oracle_policy_adv = sample_return - np.mean(stochastic_returns, axis=1)
        oracle_ref_delta = sample_return - ref_return

        selected_adv = advantages.index_select(0, selected_global_t).detach()
        selected_raw_adv = advantages_raw.index_select(0, selected_global_t).detach()
        selected_returns = returns.index_select(0, selected_global_t).detach()
        selected_values = values.index_select(0, selected_global_t).detach()

        selected_only_delta_logprob, selected_only_metrics = _selected_only_ppo_delta(
            learner,
            stage_id=stage_id,
            selected_local=selected_local,
            joint_actions=eval_actions,
            num_agents=int(num_agents),
            advantages=selected_adv,
            old_logprob=old_logprob.detach(),
        )
        per_row_delta_logprob = _per_row_ppo_delta(
            learner,
            stage_id=stage_id,
            selected_local=selected_local,
            joint_actions=eval_actions,
            num_agents=int(num_agents),
            advantages=selected_adv,
            old_logprob=old_logprob.detach(),
        )

        update_metrics = learner.update(buffer, rollout_views=rollout_views)
        with torch.no_grad():
            new_logprob, new_entropy, _new_out = learner._stage_actor_eval_from_batch(
                stage_id,
                selected_local,
                eval_actions,
                int(num_agents),
            )
        delta_logprob = (new_logprob - old_logprob).detach().cpu().numpy()
        adv_np = selected_adv.detach().cpu().numpy()
        raw_adv_np = selected_raw_adv.detach().cpu().numpy()
        old_logprob_np = old_logprob.detach().cpu().numpy()
        old_entropy_np = old_entropy.detach().cpu().numpy()
        new_entropy_np = new_entropy.detach().cpu().numpy()

        conflict_payload: dict[str, Any] = {}
        row_meta: list[dict[str, Any]] = [{} for _ in range(sample_count)]
        if bool(args.conflict_analysis):
            ppo_grad_matrix = _per_row_policy_grad_matrix(
                learner,
                stage_id=stage_id,
                selected_local=selected_local,
                joint_actions=eval_actions,
                num_agents=int(num_agents),
                weights=adv_np,
            )
            oracle_grad_matrix = _per_row_policy_grad_matrix(
                learner,
                stage_id=stage_id,
                selected_local=selected_local,
                joint_actions=eval_actions,
                num_agents=int(num_agents),
                weights=oracle_policy_adv,
            )
            ppo_cos = _cosine_matrix(ppo_grad_matrix)
            oracle_cos = _cosine_matrix(oracle_grad_matrix)
            base_labels: dict[str, list[str]] = {
                "adv_sign": ["adv_pos" if x > 0.0 else "adv_neg" if x < 0.0 else "adv_zero" for x in adv_np],
                "oracle_sign": [
                    "oracle_pos" if x > 0.0 else "oracle_neg" if x < 0.0 else "oracle_zero"
                    for x in oracle_policy_adv
                ],
            }
            if int(stage_id) == 1:
                sat_labels, row_meta = _sat_row_group_labels(
                    local_state=selected_local,
                    action_tensor=action_tensor,
                    step_indices=step_indices_all[selected_np],
                    cfg=cfg,
                    sample_count=sample_count,
                    num_agents=int(num_agents),
                )
                base_labels.update(sat_labels)
            conflict_payload = {
                "description": (
                    "Pairwise cosine conflict among per-row policy gradients. "
                    "ppo_adv uses the actual normalized PPO advantage; oracle_adv uses "
                    "G(sample)-mean_policy_samples G for the same rows."
                ),
                "ppo_adv_gradient": _group_conflict_report(
                    cos=ppo_cos,
                    labels_by_kind=base_labels,
                    adv=adv_np,
                    oracle=oracle_policy_adv,
                ),
                "oracle_adv_gradient": _group_conflict_report(
                    cos=oracle_cos,
                    labels_by_kind=base_labels,
                    adv=adv_np,
                    oracle=oracle_policy_adv,
                ),
            }

        alignment = {
            "corr_norm_adv_vs_oracle_policy_adv": _corr(adv_np, oracle_policy_adv),
            "corr_raw_adv_vs_oracle_policy_adv": _corr(raw_adv_np, oracle_policy_adv),
            "sign_agree_norm_adv_oracle_policy_adv": float(np.mean(np.sign(adv_np) == np.sign(oracle_policy_adv))),
            "sign_agree_raw_adv_oracle_policy_adv": float(np.mean(np.sign(raw_adv_np) == np.sign(oracle_policy_adv))),
            "corr_norm_adv_vs_oracle_ref_delta": _corr(adv_np, oracle_ref_delta),
            "corr_raw_adv_vs_oracle_ref_delta": _corr(raw_adv_np, oracle_ref_delta),
            "corr_norm_adv_vs_delta_logprob": _corr(adv_np, delta_logprob),
            "corr_raw_adv_vs_delta_logprob": _corr(raw_adv_np, delta_logprob),
            "corr_oracle_policy_adv_vs_delta_logprob": _corr(oracle_policy_adv, delta_logprob),
            "corr_oracle_ref_delta_vs_delta_logprob": _corr(oracle_ref_delta, delta_logprob),
            "oracle_policy_positive_frac": float(np.mean(oracle_policy_adv > 0.0)),
            "norm_adv_positive_frac": float(np.mean(adv_np > 0.0)),
            "mean_delta_logprob_when_oracle_policy_positive": float(np.mean(delta_logprob[oracle_policy_adv > 0.0]))
            if np.any(oracle_policy_adv > 0.0)
            else 0.0,
            "mean_delta_logprob_when_oracle_policy_negative": float(np.mean(delta_logprob[oracle_policy_adv < 0.0]))
            if np.any(oracle_policy_adv < 0.0)
            else 0.0,
            "mean_delta_logprob_when_adv_positive": float(np.mean(delta_logprob[adv_np > 0.0]))
            if np.any(adv_np > 0.0)
            else 0.0,
            "mean_delta_logprob_when_adv_negative": float(np.mean(delta_logprob[adv_np < 0.0]))
            if np.any(adv_np < 0.0)
            else 0.0,
            "selected_only_corr_norm_adv_vs_delta_logprob": _corr(adv_np, selected_only_delta_logprob),
            "selected_only_corr_oracle_policy_adv_vs_delta_logprob": _corr(oracle_policy_adv, selected_only_delta_logprob),
            "selected_only_mean_delta_logprob_when_oracle_policy_positive": float(
                np.mean(selected_only_delta_logprob[oracle_policy_adv > 0.0])
            )
            if np.any(oracle_policy_adv > 0.0)
            else 0.0,
            "selected_only_mean_delta_logprob_when_oracle_policy_negative": float(
                np.mean(selected_only_delta_logprob[oracle_policy_adv < 0.0])
            )
            if np.any(oracle_policy_adv < 0.0)
            else 0.0,
            "per_row_corr_norm_adv_vs_delta_logprob": _corr(adv_np, per_row_delta_logprob),
            "per_row_corr_oracle_policy_adv_vs_delta_logprob": _corr(oracle_policy_adv, per_row_delta_logprob),
            "per_row_adv_direction_agree_frac": float(np.mean((adv_np * per_row_delta_logprob) > 0.0)),
        }
        payload = {
            "config": str(args.config),
            "stage": str(args.stage),
            "stage_id": int(stage_id),
            "reward_mode": reward_mode,
            "num_envs": int(args.num_envs),
            "rollout_env_steps": int(args.rollout_env_steps),
            "sample_rows": int(sample_count),
            "policy_action_samples": int(args.policy_action_samples),
            "action_labels": labels,
            "exec_sources": {
                "accel": str(cfg.exec_accel_source),
                "sat": str(cfg.exec_sat_source),
                "bw": str(cfg.exec_bw_source),
            },
            "update_metrics": {k: float(v) for k, v in update_metrics.items() if isinstance(v, (int, float))},
            "summary": {
                "advantage_norm": _summ(adv_np),
                "advantage_raw": _summ(raw_adv_np),
                "return": _summ(selected_returns),
                "value": _summ(selected_values),
                "old_logprob": _summ(old_logprob_np),
                "old_entropy": _summ(old_entropy_np),
                "new_entropy": _summ(new_entropy_np),
                "oracle_policy_adv": _summ(oracle_policy_adv),
                "oracle_ref_delta": _summ(oracle_ref_delta),
                "delta_logprob_after_update": _summ(delta_logprob),
                "selected_only_delta_logprob": _summ(selected_only_delta_logprob),
                "per_row_delta_logprob": _summ(per_row_delta_logprob),
                "branch_returns": _summ(branch_matrix),
            },
            "selected_only_update": selected_only_metrics,
            "alignment": alignment,
            "conflict_analysis": conflict_payload,
            "rows": [
                {
                    "stage_sample": int(selected_np[i]),
                    "transition_index": int(selected_transitions[i]),
                    "history_row": int(history_rows[i]),
                    "step": int(step_indices_all[selected_np][i]),
                    "horizon": int(horizons[i]),
                    "adv_norm": float(adv_np[i]),
                    "adv_raw": float(raw_adv_np[i]),
                    "return": float(selected_returns[i].detach().cpu().item()),
                    "value": float(selected_values[i].detach().cpu().item()),
                    "ref_return": float(ref_return[i]),
                    "sample_return": float(sample_return[i]),
                    "oracle_policy_adv": float(oracle_policy_adv[i]),
                    "oracle_ref_delta": float(oracle_ref_delta[i]),
                    "delta_logprob": float(delta_logprob[i]),
                    **row_meta[i],
                }
                for i in range(sample_count)
            ],
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"alignment": alignment, "summary": payload["summary"], "out": str(out_path)}, ensure_ascii=False, indent=2))
    finally:
        close_structured_env_group(group)


if __name__ == "__main__":
    main()
