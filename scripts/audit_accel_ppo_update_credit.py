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

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import (
    StructuredMAPPO,
    _StructuredMAPPOGpuActorBridge,
    _index_dataclass,
)
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver_group


def _load_state_dict_file(module: torch.nn.Module, path: str | None, device: torch.device, key: str | None = None) -> None:
    if not path:
        return
    payload = torch.load(path, map_location=device)
    if key is not None and isinstance(payload, dict) and key in payload:
        payload = payload[key]
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    module.load_state_dict(payload, strict=False)


def _make_actor_stage_optimizers(actor: torch.nn.Module, actor_lr: float) -> dict[int, torch.optim.Optimizer]:
    stage_modules = {
        0: getattr(actor, "accel_policy", None),
        1: getattr(actor, "sat_subset_policy", None),
        2: getattr(actor, "bw_policy", None),
    }
    optimizers: dict[int, torch.optim.Optimizer] = {}
    for stage_id, module in stage_modules.items():
        if module is None:
            continue
        params = [param for param in module.parameters() if param.requires_grad]
        if params:
            optimizers[int(stage_id)] = torch.optim.Adam(params, lr=float(actor_lr))
    return optimizers


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) <= 1:
        return 0.0
    x = x[mask]
    y = y[mask]
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 1.0e-12 or sy <= 1.0e-12:
        return 0.0
    return float(np.mean((x - np.mean(x)) * (y - np.mean(y))) / (sx * sy))


def _summ(x: np.ndarray | torch.Tensor) -> dict[str, float]:
    if torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "p50": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "max": float(np.max(arr)),
    }


def _reload_actor_state(actor: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    actor.load_state_dict({k: v.detach().clone() for k, v in state.items()}, strict=True)


def _accel_selected_logprob(
    learner: StructuredMAPPO,
    *,
    selected_local: Any,
    sampled_actions: torch.Tensor,
    num_agents: int,
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    logprob, entropy, actor_out = learner._stage_actor_eval_from_batch(
        0,
        selected_local,
        sampled_actions,
        int(num_agents),
    )
    return logprob, entropy, actor_out


def _selected_danger_loss(
    learner: StructuredMAPPO,
    *,
    actor_out: Any,
    stage_batch: Any,
    selected_t: torch.Tensor,
    sample_count: int,
    num_agents: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    zero = torch.zeros((), dtype=torch.float32, device=learner.device)
    if actor_out is None or stage_batch.danger_imitation_targets is None or stage_batch.danger_imitation_masks is None:
        return zero, zero
    target = stage_batch.danger_imitation_targets.index_select(0, selected_t).to(learner.device, dtype=torch.float32)
    mask = stage_batch.danger_imitation_masks.index_select(0, selected_t).to(learner.device, dtype=torch.float32)
    target = target.reshape(int(sample_count), int(num_agents), 2)
    mask = mask.reshape_as(target)
    pred = torch.tanh(actor_out.mean).reshape_as(target)
    active = torch.sum(mask, dim=-1) > 0.0
    active_rate = active.to(torch.float32).mean() if int(active.numel()) > 0 else zero
    if not torch.any(active):
        return zero, active_rate
    diff = (pred - target) * mask
    denom = torch.sum(mask, dim=-1).clamp_min(1.0)
    per_agent = diff.pow(2).sum(dim=-1) / denom
    return per_agent[active].mean(), active_rate


def _one_selected_accel_step_delta_logprob(
    learner: StructuredMAPPO,
    *,
    actor_state: dict[str, torch.Tensor],
    selected_local: Any,
    sampled_actions: torch.Tensor,
    num_agents: int,
    advantages: torch.Tensor,
    old_logprob: torch.Tensor,
    stage_batch: Any,
    selected_t: torch.Tensor,
    mode: str,
    lr: float,
    entropy_coef: float,
    danger_coef: float,
) -> dict[str, Any]:
    _reload_actor_state(learner.actor, actor_state)
    opt = torch.optim.Adam([p for p in learner.actor.accel_policy.parameters() if p.requires_grad], lr=float(lr))
    pre_logprob, _pre_entropy, _pre_out = _accel_selected_logprob(
        learner,
        selected_local=selected_local,
        sampled_actions=sampled_actions,
        num_agents=int(num_agents),
    )
    new_logprob, entropy, actor_out = _accel_selected_logprob(
        learner,
        selected_local=selected_local,
        sampled_actions=sampled_actions,
        num_agents=int(num_agents),
    )
    ratio = torch.exp(torch.clamp(new_logprob - old_logprob.detach(), min=-20.0, max=20.0))
    clipped = torch.clamp(ratio, 1.0 - float(learner.clip_ratio), 1.0 + float(learner.clip_ratio))
    ppo_loss = -torch.minimum(ratio * advantages.detach(), clipped * advantages.detach()).mean()
    entropy_mean = entropy.mean()
    danger_loss, danger_rate = _selected_danger_loss(
        learner,
        actor_out=actor_out,
        stage_batch=stage_batch,
        selected_t=selected_t,
        sample_count=int(sampled_actions.shape[0]),
        num_agents=int(num_agents),
    )
    if mode == "ppo":
        loss = ppo_loss
    elif mode == "ppo_entropy":
        loss = ppo_loss - float(entropy_coef) * entropy_mean
    elif mode == "danger":
        loss = float(danger_coef) * danger_loss
    elif mode == "ppo_entropy_danger":
        loss = ppo_loss - float(entropy_coef) * entropy_mean + float(danger_coef) * danger_loss
    else:
        raise ValueError(f"Unsupported selected-step mode: {mode}")
    if not loss.requires_grad:
        delta = np.zeros((int(sampled_actions.shape[0]),), dtype=np.float64)
        return {
            "loss": float(loss.detach().cpu().item()),
            "ppo_loss": float(ppo_loss.detach().cpu().item()),
            "entropy": float(entropy_mean.detach().cpu().item()),
            "danger_loss": float(danger_loss.detach().cpu().item()),
            "danger_active_rate": float(danger_rate.detach().cpu().item()),
            "grad_norm": 0.0,
            "delta_logprob": _summ(delta),
            "delta_logprob_array": delta,
        }
    opt.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(learner.actor.accel_policy.parameters(), float(learner.max_grad_norm))
    opt.step()
    with torch.no_grad():
        post_logprob, _post_entropy, _post_out = _accel_selected_logprob(
            learner,
            selected_local=selected_local,
            sampled_actions=sampled_actions,
            num_agents=int(num_agents),
        )
    delta = (post_logprob - pre_logprob).detach().cpu().numpy()
    return {
        "loss": float(loss.detach().cpu().item()),
        "ppo_loss": float(ppo_loss.detach().cpu().item()),
        "entropy": float(entropy_mean.detach().cpu().item()),
        "danger_loss": float(danger_loss.detach().cpu().item()),
        "danger_active_rate": float(danger_rate.detach().cpu().item()),
        "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu().item()),
        "delta_logprob": _summ(delta),
        "delta_logprob_array": delta,
    }


def _finite_debug(value: torch.Tensor | None) -> dict[str, Any]:
    if not torch.is_tensor(value):
        return {"present": False}
    flat = value.detach().reshape(-1)
    finite = torch.isfinite(flat)
    out: dict[str, Any] = {
        "present": True,
        "shape": list(value.shape),
        "finite": bool(finite.all().detach().cpu().item()),
        "nonfinite_count": int((~finite).sum().detach().cpu().item()),
    }
    if bool(finite.any().detach().cpu().item()):
        good = flat[finite].to(dtype=torch.float32)
        out.update(
            {
                "mean": float(good.mean().detach().cpu().item()),
                "min": float(good.min().detach().cpu().item()),
                "max": float(good.max().detach().cpu().item()),
            }
        )
    return out


def _step_result_debug(result: Any) -> dict[str, Any]:
    names = (
        "team_rewards",
        "bw_access_rewards",
        "bw_weighted_workload_delta_rewards",
        "bw_weighted_workload_level_rewards",
        "bw_gu_queue_level_rewards",
        "bw_system_queue_level_rewards",
        "bw_gu_service_queue_rewards",
        "terminated",
        "truncated",
    )
    payload = {name: _finite_debug(getattr(result, name, None)) for name in names}
    reward_parts = getattr(result, "reward_part_tensors", None)
    if isinstance(reward_parts, dict):
        payload["reward_parts"] = {str(k): _finite_debug(v) for k, v in reward_parts.items()}
    return payload


class _FirstAccelActionBridge:
    """Override only the first accel action; all other stages follow learner exec_* sources."""

    def __init__(
        self,
        *,
        base_bridge: _StructuredMAPPOGpuActorBridge,
        first_actions: torch.Tensor,
    ) -> None:
        self.base_bridge = base_bridge
        self.first_actions = first_actions.detach()
        self.step_index = 0

    def bind_source_modes(self, runtime: Any) -> None:
        self.base_bridge.bind_source_modes(runtime)

    def begin_horizon(self, *, horizon: int, runtime: Any, deterministic: bool) -> None:
        self.bind_source_modes(runtime)
        self.base_bridge.begin_horizon(horizon=horizon, runtime=runtime, deterministic=deterministic)

    def end_horizon(self, *, results: Sequence[Any], runtime: Any) -> None:
        del results, runtime

    def __call__(self, *, step_index: int, runtime: Any):
        del runtime
        self.step_index = int(step_index)
        return self

    def begin_step(self, *, deterministic: bool) -> None:
        self.base_bridge.begin_step(deterministic=deterministic)

    def write_accel_action(
        self,
        accel_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        deterministic: bool,
    ) -> None:
        if int(self.step_index) != 0:
            self.base_bridge.write_accel_action(
                accel_obs,
                runtime=runtime,
                num_envs=num_envs,
                deterministic=True,
            )
            return
        del accel_obs, deterministic
        action_dst = getattr(runtime.main, "live_accel_action", None)
        if not torch.is_tensor(action_dst):
            raise RuntimeError("branch replay requires live_accel_action buffer.")
        action_t = self.first_actions.to(device=action_dst.device, dtype=action_dst.dtype).reshape_as(action_dst)
        action_dst.copy_(action_t)
        logprob_dst = getattr(runtime.main, "live_accel_old_logprob", None)
        if torch.is_tensor(logprob_dst):
            logprob_dst.zero_()

    def write_sat_action(self, *args, **kwargs) -> None:
        self.base_bridge.write_sat_action(*args, **kwargs)

    def write_bw_action(self, *args, **kwargs) -> None:
        self.base_bridge.write_bw_action(*args, **kwargs)


def _branch_returns_from_history(
    learner: StructuredMAPPO,
    *,
    history_rows: Sequence[int],
    first_actions: torch.Tensor,
    horizons: Sequence[int],
) -> torch.Tensor:
    rollout_program = getattr(learner, "_native_rollout_program", None)
    drivers = None if rollout_program is None else getattr(rollout_program, "drivers", None)
    runtime = None if rollout_program is None else getattr(rollout_program, "runtime", None)
    history = None if runtime is None else getattr(runtime, "history", None)
    if drivers is None or history is None:
        raise RuntimeError("native rollout program/history is required for branch replay.")
    source_num_envs = int(getattr(history, "num_envs", 0) or 0)
    if source_num_envs <= 0:
        raise RuntimeError("cannot infer source rollout env count.")
    sub_batch_program_factory = getattr(drivers, "native_sub_batch_rollout_program", None)
    prepare_branch = getattr(drivers, "prepare_native_branch_replay_from_history", None)
    if not callable(sub_batch_program_factory) or not callable(prepare_branch):
        raise RuntimeError("native branch replay support is unavailable.")

    rows_tuple = tuple(int(x) for x in history_rows)
    horizons_t = torch.as_tensor(list(horizons), dtype=torch.long, device=learner.device).reshape(-1)
    if len(rows_tuple) != int(first_actions.shape[0]) or len(rows_tuple) != int(horizons_t.numel()):
        raise ValueError("history_rows, first_actions, and horizons must have the same length.")
    returns_out = torch.empty((len(rows_tuple),), dtype=torch.float32, device=learner.device)
    first_actions = first_actions.detach().to(device=learner.device, dtype=torch.float32)
    learner._ensure_native_actor_cuda_bindings(sync=False)
    for horizon_i in torch.unique(horizons_t).detach().cpu().tolist():
        horizon_i = max(int(horizon_i), 1)
        group_idx_t = torch.nonzero(horizons_t == horizon_i, as_tuple=False).reshape(-1)
        group_idx = group_idx_t.detach().cpu().numpy().astype(np.int64).tolist()
        chunk_rows = tuple(rows_tuple[i] for i in group_idx)
        chunk_actions = first_actions.index_select(0, group_idx_t.to(device=learner.device)).contiguous()
        selected_envs = tuple(int(row % source_num_envs) for row in chunk_rows)
        program = sub_batch_program_factory(
            capacity=horizon_i,
            selected_indices=selected_envs,
            allow_duplicate_indices=True,
        )
        sub_runtime = program.runtime
        begin_horizon = getattr(program._step_program.executor, "_runtime_begin_horizon", None)
        if not callable(begin_horizon):
            raise RuntimeError("sub-batch program cannot begin a horizon.")
        begin_horizon(num_steps=horizon_i)
        program._step_program._horizon_started = True
        prepare_branch(history_rows=chunk_rows, horizon=horizon_i, stage_id=0)
        bridge = _FirstAccelActionBridge(
            base_bridge=_StructuredMAPPOGpuActorBridge(learner),
            first_actions=chunk_actions,
        )
        bridge.begin_horizon(horizon=horizon_i, runtime=sub_runtime, deterministic=True)
        results: list[Any] = []
        for step_index in range(horizon_i):
            step_result = program.replay_step(
                actor_bridge=bridge(step_index=step_index, runtime=sub_runtime),
                deterministic=True,
                rollout_tail=bool(step_index + 1 >= horizon_i),
            )
            step_view = (
                sub_runtime.result.step_result_views[step_index]
                if step_index < len(sub_runtime.result.step_result_views)
                else step_result
            )
            reward_t = learner._native_bw_target_reward_tensor(
                step_view,
                bw_target_mode="env_reward",
                bw_reward_w_access=float(learner.bw_reward_w_access),
                device=learner.device,
            )
            if not bool(torch.isfinite(reward_t).all().detach().cpu().item()):
                debug = {
                    "horizon": int(horizon_i),
                    "step_index": int(step_index),
                    "chunk_rows": [int(x) for x in chunk_rows],
                    "selected_envs": [int(x) for x in selected_envs],
                    "step_result": _step_result_debug(step_view),
                }
                raise RuntimeError(
                    "native accel branch replay produced non-finite reward:\n"
                    + json.dumps(debug, ensure_ascii=False, indent=2)
                )
            results.append(step_view)
        bridge.end_horizon(results=results, runtime=sub_runtime)
        group_returns = learner._discounted_native_returns(
            results,
            device=learner.device,
            gamma=float(learner.gamma),
            bw_target_mode="env_reward",
            bw_reward_w_access=float(learner.bw_reward_w_access),
        )
        if not bool(torch.isfinite(group_returns).all().detach().cpu().item()):
            debug = {
                "horizon": int(horizon_i),
                "chunk_rows": [int(x) for x in chunk_rows],
                "selected_envs": [int(x) for x in selected_envs],
                "group_returns": _finite_debug(group_returns),
            }
            raise RuntimeError(
                "native accel branch replay produced non-finite discounted returns:\n"
                + json.dumps(debug, ensure_ascii=False, indent=2)
            )
        returns_out.index_copy_(0, group_idx_t.to(device=learner.device), group_returns)
    return returns_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--rollout_env_steps", type=int, default=250)
    parser.add_argument("--sample_rows", type=int, default=24)
    parser.add_argument("--branch_horizon_cap", type=int, default=0)
    parser.add_argument("--policy_baseline_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--actor_checkpoint", default=None)
    parser.add_argument("--critic_checkpoint", default=None)
    parser.add_argument("--train_state", default=None)
    args = parser.parse_args()

    _set_seed(int(args.seed))
    cfg = load_config(args.config)
    device = torch.device(args.device)
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    actor = bundle.actor.to(device)
    critic = bundle.critic.to(device)
    if args.train_state:
        _load_state_dict_file(actor, args.train_state, device, key="actor_state_dict")
        _load_state_dict_file(critic, args.train_state, device, key="critic_state_dict")
    _load_state_dict_file(actor, args.actor_checkpoint, device)
    _load_state_dict_file(critic, args.critic_checkpoint, device)
    actor_optimizer = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=float(cfg.actor_lr))
    actor_stage_optimizers = _make_actor_stage_optimizers(actor, float(cfg.actor_lr))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(cfg.critic_lr))
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
        target_mode="step_level",
        danger_imitation_enabled=bool(getattr(cfg, "danger_imitation_enabled", False)),
        danger_imitation_coef=float(getattr(cfg, "danger_imitation_coef", 0.0) or 0.0),
        actor_optimizer=actor_optimizer,
        actor_stage_optimizers=actor_stage_optimizers,
        critic_optimizer=critic_optimizer,
        device=device,
        cfg=cfg,
        train_accel=bool(getattr(cfg, "train_accel", True)),
        train_sat=bool(getattr(cfg, "train_sat", True)),
        train_bw=bool(getattr(cfg, "train_bw", True)),
        exec_accel_source=str(getattr(cfg, "exec_accel_source", "policy") or "policy"),
        exec_sat_source=str(getattr(cfg, "exec_sat_source", "policy") or "policy"),
        exec_bw_source=str(getattr(cfg, "exec_bw_source", "policy") or "policy"),
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
            buffer,
            horizon=int(args.rollout_env_steps),
            deterministic=False,
        )
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
        advantages_raw = torch.from_numpy(gae["advantages"]).to(device=device, dtype=torch.float32)
        returns = torch.from_numpy(gae["returns"]).to(device=device, dtype=torch.float32)
        values_for_advantage = value_override.to(device=device, dtype=torch.float32)
        advantages = advantages_raw.clone()
        if bool(getattr(cfg, "actor_advantage_normalize_enabled", True)) and advantages.numel() > 1:
            if bool(getattr(cfg, "stagewise_advantage_norm_enabled", False)):
                stage_ids = np.asarray(batch_view.stage_ids, dtype=np.int64)
                for stage_id in (0, 1, 2):
                    idx_np = np.flatnonzero(stage_ids == stage_id)
                    if idx_np.size <= 1:
                        continue
                    idx_t = torch.as_tensor(idx_np, dtype=torch.long, device=device)
                    vals = advantages.index_select(0, idx_t)
                    vals = (vals - vals.mean()) / vals.std(unbiased=False).clamp_min(1.0e-8)
                    advantages.index_copy_(0, idx_t, vals)
            else:
                advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1.0e-8)

        stage_batch = batch_view.stage_batches[0]
        num_samples = int(stage_batch.num_samples)
        num_agents = int(stage_batch.num_agents)
        rng = np.random.default_rng(int(args.seed))
        sample_count = min(max(int(args.sample_rows), 1), num_samples)
        selected_np = np.sort(rng.choice(num_samples, size=sample_count, replace=False))
        selected_t = torch.as_tensor(selected_np, dtype=torch.long, device=device)
        transition_indices = np.asarray(stage_batch.transition_indices, dtype=np.int64)
        selected_transitions = transition_indices[selected_np]
        selected_global_t = torch.as_tensor(selected_transitions, dtype=torch.long, device=device)
        selected_adv = advantages.index_select(0, selected_global_t)
        selected_raw_adv = advantages_raw.index_select(0, selected_global_t)
        selected_returns = returns.index_select(0, selected_global_t)
        selected_values = values_for_advantage.index_select(0, selected_global_t)

        agent_offsets = torch.arange(num_agents, dtype=torch.long, device=device).view(1, num_agents)
        flat_rows = (selected_t.view(-1, 1) * num_agents + agent_offsets).reshape(-1)
        selected_local = _index_dataclass(stage_batch.local_batch, flat_rows)
        sampled_actions = stage_batch.actions.index_select(0, selected_t).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            ref_out = actor.act_accel(selected_local, deterministic=True)
            ref_actions = ref_out.action.reshape(sample_count, num_agents, -1).to(device=device, dtype=torch.float32)
            policy_alt_actions: list[torch.Tensor] = []
            for _ in range(max(int(args.policy_baseline_samples), 0)):
                alt_out = actor.act_accel(selected_local, deterministic=False)
                policy_alt_actions.append(
                    alt_out.action.reshape(sample_count, num_agents, -1).to(device=device, dtype=torch.float32)
                )
            old_logprob, _old_entropy, _old_out = learner._stage_actor_eval_from_batch(
                0,
                selected_local,
                sampled_actions,
                num_agents,
            )
        history_rows = ((selected_transitions - 0) // 3).astype(np.int64)
        step_indices = history_rows // int(args.num_envs)
        horizons = np.maximum(int(cfg.T_steps) - step_indices, 1).astype(np.int64)
        if int(args.branch_horizon_cap) > 0:
            horizons = np.minimum(horizons, int(args.branch_horizon_cap)).astype(np.int64, copy=False)
        paired_rows = np.repeat(history_rows, 2)
        paired_horizons = np.repeat(horizons, 2)
        paired_actions = torch.empty((sample_count * 2, num_agents, 2), dtype=torch.float32, device=device)
        paired_actions[0::2] = ref_actions
        paired_actions[1::2] = sampled_actions
        branch_returns = _branch_returns_from_history(
            learner,
            history_rows=paired_rows.tolist(),
            first_actions=paired_actions,
            horizons=paired_horizons.tolist(),
        )
        ref_returns = branch_returns[0::2]
        sample_returns = branch_returns[1::2]
        branch_delta = sample_returns - ref_returns
        policy_mean_returns = None
        policy_delta = None
        if policy_alt_actions:
            alt_rows = np.repeat(history_rows, len(policy_alt_actions))
            alt_horizons = np.repeat(horizons, len(policy_alt_actions))
            alt_actions = torch.cat(policy_alt_actions, dim=0)
            alt_returns = _branch_returns_from_history(
                learner,
                history_rows=alt_rows.tolist(),
                first_actions=alt_actions,
                horizons=alt_horizons.tolist(),
            ).reshape(len(policy_alt_actions), sample_count)
            policy_mean_returns = alt_returns.mean(dim=0)
            policy_delta = sample_returns - policy_mean_returns

        selected_actor_state = copy.deepcopy(actor.state_dict())
        selected_step_payload: dict[str, Any] = {}
        selected_old_logprob = old_logprob.detach()
        selected_adv = selected_adv.detach()
        for mode in ("ppo", "ppo_entropy", "danger", "ppo_entropy_danger"):
            step_payload = _one_selected_accel_step_delta_logprob(
                learner,
                actor_state=selected_actor_state,
                selected_local=selected_local,
                sampled_actions=sampled_actions,
                num_agents=int(num_agents),
                advantages=selected_adv,
                old_logprob=selected_old_logprob,
                stage_batch=stage_batch,
                selected_t=selected_t,
                mode=mode,
                lr=float(cfg.actor_lr),
                entropy_coef=float(getattr(cfg, "entropy_coef_accel", getattr(cfg, "entropy_coef", 0.0)) or 0.0),
                danger_coef=float(getattr(cfg, "danger_imitation_coef", 0.0) or 0.0),
            )
            delta_arr = np.asarray(step_payload.pop("delta_logprob_array"), dtype=np.float64)
            step_payload["corr_norm_adv_vs_delta_logprob"] = _corr(adv_np if "adv_np" in locals() else selected_adv.detach().cpu().numpy(), delta_arr)
            if policy_delta is not None:
                policy_delta_tmp = policy_delta.detach().cpu().numpy()
                step_payload["corr_policy_delta_vs_delta_logprob"] = _corr(policy_delta_tmp, delta_arr)
                step_payload["mean_delta_logprob_when_policy_positive"] = (
                    float(np.mean(delta_arr[policy_delta_tmp > 0.0])) if np.any(policy_delta_tmp > 0.0) else 0.0
                )
                step_payload["mean_delta_logprob_when_policy_negative"] = (
                    float(np.mean(delta_arr[policy_delta_tmp < 0.0])) if np.any(policy_delta_tmp < 0.0) else 0.0
                )
            selected_step_payload[mode] = step_payload
        _reload_actor_state(actor, selected_actor_state)

        update_metrics = learner.update(buffer, rollout_views=rollout_views)
        with torch.no_grad():
            new_logprob, _new_entropy, _new_out = learner._stage_actor_eval_from_batch(
                0,
                selected_local,
                sampled_actions,
                num_agents,
            )
        delta_logprob = new_logprob - old_logprob

        adv_np = selected_adv.detach().cpu().numpy()
        raw_adv_np = selected_raw_adv.detach().cpu().numpy()
        branch_np = branch_delta.detach().cpu().numpy()
        policy_delta_np = None if policy_delta is None else policy_delta.detach().cpu().numpy()
        dlogp_np = delta_logprob.detach().cpu().numpy()
        payload = {
            "config": str(args.config),
            "num_envs": int(args.num_envs),
            "rollout_env_steps": int(args.rollout_env_steps),
            "sample_rows": int(sample_count),
            "exec_sources": {
                "accel": str(getattr(cfg, "exec_accel_source", "policy") or "policy"),
                "sat": str(getattr(cfg, "exec_sat_source", "policy") or "policy"),
                "bw": str(getattr(cfg, "exec_bw_source", "policy") or "policy"),
            },
            "update_metrics": {k: float(v) for k, v in update_metrics.items() if isinstance(v, (int, float))},
            "advantage": {
                "normalized": _summ(adv_np),
                "raw": _summ(raw_adv_np),
                "return": _summ(selected_returns),
                "value": _summ(selected_values),
            },
            "branch_delta": _summ(branch_np),
            "policy_delta": None if policy_delta_np is None else _summ(policy_delta_np),
            "delta_logprob": _summ(dlogp_np),
            "alignment": {
                "corr_norm_adv_vs_branch_delta": _corr(adv_np, branch_np),
                "corr_raw_adv_vs_branch_delta": _corr(raw_adv_np, branch_np),
                "sign_agree_norm_adv_branch_delta": float(np.mean(np.sign(adv_np) == np.sign(branch_np))),
                "sign_agree_raw_adv_branch_delta": float(np.mean(np.sign(raw_adv_np) == np.sign(branch_np))),
                "corr_norm_adv_vs_delta_logprob": _corr(adv_np, dlogp_np),
                "corr_raw_adv_vs_delta_logprob": _corr(raw_adv_np, dlogp_np),
                "corr_branch_delta_vs_delta_logprob": _corr(branch_np, dlogp_np),
                "mean_delta_logprob_when_branch_positive": float(np.mean(dlogp_np[branch_np > 0.0]))
                if np.any(branch_np > 0.0)
                else 0.0,
                "mean_delta_logprob_when_branch_negative": float(np.mean(dlogp_np[branch_np < 0.0]))
                if np.any(branch_np < 0.0)
                else 0.0,
                "branch_positive_frac": float(np.mean(branch_np > 0.0)),
                "norm_adv_positive_frac": float(np.mean(adv_np > 0.0)),
            },
            "selected_rows_one_step_update": selected_step_payload,
            "rows": [
                {
                    "sample": int(selected_np[i]),
                    "transition_index": int(selected_transitions[i]),
                    "history_row": int(history_rows[i]),
                    "step": int(step_indices[i]),
                    "horizon": int(horizons[i]),
                    "adv_norm": float(adv_np[i]),
                    "adv_raw": float(raw_adv_np[i]),
                    "return": float(selected_returns[i].detach().cpu().item()),
                    "value": float(selected_values[i].detach().cpu().item()),
                    "ref_return": float(ref_returns[i].detach().cpu().item()),
                    "sample_return": float(sample_returns[i].detach().cpu().item()),
                    "policy_mean_return": None
                    if policy_mean_returns is None
                    else float(policy_mean_returns[i].detach().cpu().item()),
                    "branch_delta": float(branch_np[i]),
                    "policy_delta": None if policy_delta_np is None else float(policy_delta_np[i]),
                    "delta_logprob": float(dlogp_np[i]),
                }
                for i in range(sample_count)
            ],
        }
        if policy_delta_np is not None:
            payload["alignment"].update(
                {
                    "corr_norm_adv_vs_policy_delta": _corr(adv_np, policy_delta_np),
                    "corr_raw_adv_vs_policy_delta": _corr(raw_adv_np, policy_delta_np),
                    "sign_agree_norm_adv_policy_delta": float(np.mean(np.sign(adv_np) == np.sign(policy_delta_np))),
                    "sign_agree_raw_adv_policy_delta": float(np.mean(np.sign(raw_adv_np) == np.sign(policy_delta_np))),
                    "corr_policy_delta_vs_delta_logprob": _corr(policy_delta_np, dlogp_np),
                    "mean_delta_logprob_when_policy_positive": float(np.mean(dlogp_np[policy_delta_np > 0.0]))
                    if np.any(policy_delta_np > 0.0)
                    else 0.0,
                    "mean_delta_logprob_when_policy_negative": float(np.mean(dlogp_np[policy_delta_np < 0.0]))
                    if np.any(policy_delta_np < 0.0)
                    else 0.0,
                    "policy_positive_frac": float(np.mean(policy_delta_np > 0.0)),
                }
            )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload["alignment"], ensure_ascii=False, indent=2))
        print(f"wrote {out_path}")
    finally:
        close_structured_env_group(group)


if __name__ == "__main__":
    main()
