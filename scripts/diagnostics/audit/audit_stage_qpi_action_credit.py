from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
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
from sagin_marl.rl.stage_mcgae import (
    STAGE_ID,
    force_single_stage_config as _force_single_stage_config,
    make_stage_optimizers as _make_stage_optimizers,
)
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver_group

from scripts.diagnostics.audit.audit_stage_ppo_credit_alignment import (
    _normalize_advantages_like_update,
    _stage_action_samples,
)
from scripts.diagnostics.diagnose.diagnose_reward_action_sensitivity import (
    _corr,
    _discounted_returns_for_modes,
    _summ,
)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1.0e-12:
        return 0.0
    return float(np.dot(av, bv) / denom)


class _FirstActionQpiBridge:
    """Override only the first action; later decisions can be stochastic."""

    def __init__(
        self,
        *,
        base_bridge: _StructuredMAPPOGpuActorBridge,
        stage_id: int,
        first_actions: torch.Tensor,
        follow_deterministic: bool,
    ) -> None:
        self.base_bridge = base_bridge
        self.stage_id = int(stage_id)
        self.first_actions = first_actions.detach()
        self.follow_deterministic = bool(follow_deterministic)
        self.step_index = 0

    def bind_source_modes(self, runtime: Any) -> None:
        self.base_bridge.bind_source_modes(runtime)

    def begin_horizon(self, *, horizon: int, runtime: Any, deterministic: bool) -> None:
        del horizon, deterministic
        self.bind_source_modes(runtime)
        self.base_bridge.begin_horizon(horizon=0, runtime=runtime, deterministic=self.follow_deterministic)

    def end_horizon(self, *, results: Sequence[Any], runtime: Any) -> None:
        del results, runtime

    def __call__(self, *, step_index: int, runtime: Any):
        del runtime
        self.step_index = int(step_index)
        return self

    def begin_step(self, *, deterministic: bool) -> None:
        del deterministic
        self.base_bridge.begin_step(deterministic=self.follow_deterministic)

    def write_accel_action(
        self,
        accel_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        deterministic: bool,
    ) -> None:
        if int(self.step_index) != 0 or int(self.stage_id) != 0:
            self.base_bridge.write_accel_action(
                accel_obs,
                runtime=runtime,
                num_envs=num_envs,
                deterministic=self.follow_deterministic,
            )
            return
        del accel_obs, deterministic
        action_dst = getattr(runtime.main, "live_accel_action", None)
        if not torch.is_tensor(action_dst):
            raise RuntimeError("Qpi branch replay requires live_accel_action buffer.")
        action_dst.copy_(self.first_actions.to(device=action_dst.device, dtype=action_dst.dtype).reshape_as(action_dst))
        logprob_dst = getattr(runtime.main, "live_accel_old_logprob", None)
        if torch.is_tensor(logprob_dst):
            logprob_dst.zero_()

    def write_sat_action(
        self,
        sat_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        sat_max_select: int,
        deterministic: bool,
    ) -> None:
        if int(self.step_index) != 0 or int(self.stage_id) != 1:
            self.base_bridge.write_sat_action(
                sat_obs,
                runtime=runtime,
                num_envs=num_envs,
                sat_max_select=sat_max_select,
                deterministic=self.follow_deterministic,
            )
            return
        del sat_obs, sat_max_select, deterministic
        action_dst = getattr(runtime.main, "live_sat_subset_index", None)
        if not torch.is_tensor(action_dst):
            raise RuntimeError("Qpi SAT branch replay requires live_sat_subset_index buffer.")
        action_dst.copy_(self.first_actions.to(device=action_dst.device, dtype=action_dst.dtype).reshape_as(action_dst))
        for name in ("live_sat_old_logprobs_per_agent", "live_sat_entropy_per_agent"):
            value = getattr(runtime.main, name, None)
            if torch.is_tensor(value):
                value.zero_()

    def write_bw_action(
        self,
        bw_obs: Any,
        *,
        runtime: Any,
        num_envs: int,
        deterministic: bool,
    ) -> None:
        if int(self.step_index) != 0 or int(self.stage_id) != 2:
            self.base_bridge.write_bw_action(
                bw_obs,
                runtime=runtime,
                num_envs=num_envs,
                deterministic=self.follow_deterministic,
            )
            return
        del bw_obs, deterministic
        action_dst = getattr(runtime.main, "live_bw_action", None)
        if not torch.is_tensor(action_dst):
            raise RuntimeError("Qpi BW branch replay requires live_bw_action buffer.")
        action_dst.copy_(self.first_actions.to(device=action_dst.device, dtype=action_dst.dtype).reshape_as(action_dst))
        ref_dst = getattr(runtime.main, "live_bw_ref_action", None)
        if torch.is_tensor(ref_dst):
            ref_dst.copy_(self.first_actions.to(device=ref_dst.device, dtype=ref_dst.dtype).reshape_as(ref_dst))
        for name in (
            "live_bw_old_logprob",
            "live_bw_old_logprobs_per_agent",
            "live_bw_entropy_per_agent",
            "live_bw_logprob_raw_per_agent",
            "live_bw_entropy_raw_per_agent",
        ):
            value = getattr(runtime.main, name, None)
            if torch.is_tensor(value):
                value.zero_()


def _branch_returns_qpi(
    learner: StructuredMAPPO,
    *,
    stage_id: int,
    history_rows: Sequence[int],
    first_actions: torch.Tensor,
    horizons: Sequence[int],
    reward_modes: Sequence[str],
    follow_deterministic: bool,
    future_random_mode: str = "copy",
    future_random_seed: int | None = None,
) -> dict[str, torch.Tensor]:
    rollout_program = getattr(learner, "_native_rollout_program", None)
    drivers = None if rollout_program is None else getattr(rollout_program, "drivers", None)
    runtime = None if rollout_program is None else getattr(rollout_program, "runtime", None)
    history = None if runtime is None else getattr(runtime, "history", None)
    if drivers is None or history is None:
        raise RuntimeError("native rollout program/history is required for Qpi branch replay.")
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
    first_actions = first_actions.detach().to(device=learner.device).contiguous()
    out = {str(mode): torch.empty((len(rows_tuple),), dtype=torch.float32, device=learner.device) for mode in reward_modes}
    stage_id_i = int(stage_id)
    future_random_mode_i = str(future_random_mode or "copy").strip().lower()
    if future_random_mode_i not in {"copy", "resample"}:
        raise ValueError(f"future_random_mode must be 'copy' or 'resample', got {future_random_mode!r}.")
    learner._ensure_native_actor_cuda_bindings(sync=False)
    for group_ordinal, horizon_i in enumerate(torch.unique(horizons_t).detach().cpu().tolist()):
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
        branch_seed = None
        if future_random_seed is not None:
            branch_seed = int(future_random_seed) + 1_000_003 * int(horizon_i) + 97_003 * int(group_ordinal)
        prepare_branch(
            history_rows=chunk_rows,
            horizon=horizon_i,
            stage_id=stage_id_i,
            future_random_mode=future_random_mode_i,
            future_random_seed=branch_seed,
        )
        bridge = _FirstActionQpiBridge(
            base_bridge=_StructuredMAPPOGpuActorBridge(learner),
            stage_id=stage_id_i,
            first_actions=chunk_actions,
            follow_deterministic=bool(follow_deterministic),
        )
        bridge.begin_horizon(horizon=horizon_i, runtime=sub_runtime, deterministic=bool(follow_deterministic))
        results: list[Any] = []
        if stage_id_i == 0:
            step_result = program.replay_step(
                actor_bridge=bridge(step_index=0, runtime=sub_runtime),
                deterministic=bool(follow_deterministic),
                rollout_tail=bool(horizon_i <= 1),
            )
            results.append(sub_runtime.result.step_result_views[0] if sub_runtime.result.step_result_views else step_result)
        elif stage_id_i == 1:
            step_bridge = bridge(step_index=0, runtime=sub_runtime)
            step_bridge.begin_step(deterministic=bool(follow_deterministic))
            sat_obs = sub_runtime.main.live_sat_obs
            if sat_obs is None:
                raise RuntimeError("SAT Qpi branch replay missing SAT live obs at snapshot.")
            step_bridge.write_sat_action(
                sat_obs,
                runtime=sub_runtime,
                num_envs=int(len(chunk_rows)),
                sat_max_select=int(getattr(sub_runtime.main, "sat_max_select", 1) or 1),
                deterministic=bool(follow_deterministic),
            )
            bw_obs = program._step_program.executor._runtime_step_publish_bw_obs(
                max_visible=program.fixed_visible_sat_width
            )
            step_bridge.write_bw_action(
                bw_obs,
                runtime=sub_runtime,
                num_envs=int(len(chunk_rows)),
                deterministic=bool(follow_deterministic),
            )
            first_result = program._step_program.executor._runtime_step_finish_bw(
                max_visible=program.fixed_visible_sat_width,
                rollout_tail=bool(horizon_i <= 1),
            )
            results.append(sub_runtime.result.step_result_views[0] if sub_runtime.result.step_result_views else first_result)
        else:
            step_bridge = bridge(step_index=0, runtime=sub_runtime)
            step_bridge.begin_step(deterministic=bool(follow_deterministic))
            bw_obs = sub_runtime.main.live_bw_obs
            if bw_obs is None:
                raise RuntimeError("BW Qpi branch replay missing BW live obs at snapshot.")
            step_bridge.write_bw_action(
                bw_obs,
                runtime=sub_runtime,
                num_envs=int(len(chunk_rows)),
                deterministic=bool(follow_deterministic),
            )
            first_result = program._step_program.executor._runtime_step_finish_bw(
                max_visible=program.fixed_visible_sat_width,
                rollout_tail=bool(horizon_i <= 1),
            )
            results.append(sub_runtime.result.step_result_views[0] if sub_runtime.result.step_result_views else first_result)
        for step_index in range(1, horizon_i):
            step_result = program.replay_step(
                actor_bridge=bridge(step_index=step_index, runtime=sub_runtime),
                deterministic=bool(follow_deterministic),
                rollout_tail=bool(step_index + 1 >= horizon_i),
            )
            step_view = (
                sub_runtime.result.step_result_views[step_index]
                if step_index < len(sub_runtime.result.step_result_views)
                else step_result
            )
            results.append(step_view)
        bridge.end_horizon(results=results, runtime=sub_runtime)
        returns_by_mode = _discounted_returns_for_modes(
            results,
            reward_modes=reward_modes,
            cfg=learner.cfg,
            gamma=float(learner.gamma),
            device=learner.device,
        )
        for mode, values in returns_by_mode.items():
            if not bool(torch.isfinite(values).all().detach().cpu().item()):
                raise RuntimeError(f"non-finite Qpi branch returns for mode {mode!r}.")
            out[str(mode)].index_copy_(0, group_idx_t.to(device=learner.device), values)
    return out


def _flat_grad_vector(loss: torch.Tensor, params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    grads = torch.autograd.grad(loss, list(params), retain_graph=False, create_graph=False, allow_unused=True)
    pieces: list[torch.Tensor] = []
    for param, grad in zip(params, grads):
        pieces.append(torch.zeros_like(param).reshape(-1) if grad is None else grad.detach().reshape(-1))
    return torch.cat(pieces) if pieces else loss.new_zeros((0,))


def _stage_params(actor: torch.nn.Module, stage_id: int) -> list[torch.nn.Parameter]:
    module = (
        actor.accel_policy
        if int(stage_id) == 0
        else actor.sat_subset_policy
        if int(stage_id) == 1
        else actor.bw_policy
    )
    return [p for p in module.parameters() if p.requires_grad]


def _qpi_policy_gradient_diagnostics(
    learner: StructuredMAPPO,
    *,
    stage_id: int,
    selected_local: Any,
    action_tensor: torch.Tensor,
    q_adv: np.ndarray,
    num_agents: int,
) -> dict[str, float]:
    # q_adv is [rows, policy_actions], where policy_actions exclude deterministic ref.
    rows = int(q_adv.shape[0])
    acts = int(q_adv.shape[1])
    if rows <= 0 or acts <= 0:
        return {}
    params = _stage_params(learner.actor, int(stage_id))
    if not params:
        return {}
    device = learner.device
    action_policy = action_tensor[:, 1 : 1 + acts].to(device=device)
    # Flatten row/action pairs while repeating local state per action.
    pair_count = rows * acts
    action_flat = action_policy.reshape(pair_count, int(num_agents), -1)
    local_indices = (
        torch.arange(rows, dtype=torch.long, device=device).view(rows, 1, 1) * int(num_agents)
        + torch.arange(int(num_agents), dtype=torch.long, device=device).view(1, 1, int(num_agents))
    ).expand(rows, acts, int(num_agents)).reshape(-1)
    local_flat = _index_dataclass(selected_local, local_indices)
    logprob, _entropy, _out = learner._stage_actor_eval_from_batch(
        int(stage_id),
        local_flat,
        action_flat,
        int(num_agents),
    )
    adv_t = torch.as_tensor(q_adv.reshape(pair_count), dtype=logprob.dtype, device=device)
    loss = -(adv_t.detach() * logprob).mean()
    mean_grad = _flat_grad_vector(loss, params).detach().cpu().numpy()

    row_grads: list[np.ndarray] = []
    for row in range(rows):
        lp_row, _ent_row, _out_row = learner._stage_actor_eval_from_batch(
            int(stage_id),
            _index_dataclass(
                selected_local,
                (
                    torch.full((acts, int(num_agents)), int(row), dtype=torch.long, device=device) * int(num_agents)
                    + torch.arange(int(num_agents), dtype=torch.long, device=device).view(1, int(num_agents))
                ).reshape(-1),
            ),
            action_policy[row].reshape(acts, int(num_agents), -1),
            int(num_agents),
        )
        adv_row = torch.as_tensor(q_adv[row], dtype=lp_row.dtype, device=device)
        loss_row = -(adv_row.detach() * lp_row).mean()
        row_grads.append(_flat_grad_vector(loss_row, params).detach().cpu().numpy())
    grad_mat = np.stack(row_grads, axis=0) if row_grads else np.zeros((0, int(mean_grad.size)), dtype=np.float64)
    row_norms = np.linalg.norm(grad_mat, axis=1) if grad_mat.size else np.zeros((0,), dtype=np.float64)
    mean_norm = float(np.linalg.norm(mean_grad))
    mean_row_norm = float(np.mean(row_norms)) if row_norms.size else 0.0
    cancellation = mean_norm / max(mean_row_norm, 1.0e-12)
    split_cos = 0.0
    if rows >= 4 and grad_mat.size:
        first = np.mean(grad_mat[0::2], axis=0)
        second = np.mean(grad_mat[1::2], axis=0)
        split_cos = _cosine(first, second)
    return {
        "qpi_grad_mean_norm": mean_norm,
        "qpi_grad_mean_row_norm": mean_row_norm,
        "qpi_grad_cancellation_ratio": cancellation,
        "qpi_grad_split_half_cosine": split_cos,
        "qpi_grad_row_norm": _summ(row_norms),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(STAGE_ID), default="sat")
    parser.add_argument("--config", required=True)
    parser.add_argument("--reward_mode", default="sat_relay_processed")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--rollout_env_steps", type=int, default=250)
    parser.add_argument("--sample_rows", type=int, default=8)
    parser.add_argument("--policy_action_samples", type=int, default=4)
    parser.add_argument("--continuations", type=int, default=4)
    parser.add_argument("--follow", choices=["deterministic", "stochastic"], default="stochastic")
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
        num_samples = int(stage_batch.num_samples)
        num_agents = int(stage_batch.num_agents)
        transition_indices = np.asarray(stage_batch.transition_indices, dtype=np.int64)
        history_rows_all = ((transition_indices - int(stage_id)) // 3).astype(np.int64)
        step_indices_all = history_rows_all // max(int(args.num_envs), 1)
        horizons_all = np.maximum(int(cfg.T_steps) - step_indices_all, 1).astype(np.int64)
        eligible = np.flatnonzero(horizons_all >= max(int(args.min_horizon), 1))
        if eligible.size <= 0:
            raise RuntimeError("no eligible rows after min_horizon filtering.")
        rng = np.random.default_rng(int(args.seed) + 29)
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
        labels, action_tensor, _eval_actions = _stage_action_samples(
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
        continuations = max(int(args.continuations), 1)
        history_rows = history_rows_all[selected_np]
        horizons = horizons_all[selected_np]
        if int(args.branch_horizon_cap) > 0:
            horizons = np.minimum(horizons, int(args.branch_horizon_cap)).astype(np.int64, copy=False)
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
        stochastic_q = q_mean[:, 1:]
        q_adv = stochastic_q - stochastic_q.mean(axis=1, keepdims=True)
        rollout_q_adv = q_mean[:, 1] - stochastic_q.mean(axis=1)
        selected_adv_norm = adv_norm.index_select(0, selected_global_t).detach().cpu().numpy()
        selected_adv_raw = adv_raw.index_select(0, selected_global_t).detach().cpu().numpy()
        selected_returns = returns.index_select(0, selected_global_t).detach().cpu().numpy()
        selected_values = values.index_select(0, selected_global_t).detach().cpu().numpy()

        action_effect_var = np.var(q_mean[:, 1:], axis=1)
        continuation_var = np.mean(np.var(ret_cube[:, 1:, :], axis=2), axis=1)
        action_effect_std = float(math.sqrt(max(float(np.mean(action_effect_var)), 0.0)))
        continuation_std = float(math.sqrt(max(float(np.mean(continuation_var)), 0.0)))
        credit_snr = action_effect_std / max(continuation_std, 1.0e-12)
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
            "branch_horizon_cap": int(args.branch_horizon_cap),
            "action_labels": labels,
            "qpi": {
                "return_cube": _summ(ret_cube),
                "q_mean": _summ(q_mean),
                "q_std_per_action": _summ(q_std),
                "action_effect_std": action_effect_std,
                "continuation_std": continuation_std,
                "action_effect_to_continuation_snr": credit_snr,
                "best_minus_ref": _summ(np.max(q_mean, axis=1) - q_mean[:, 0]),
                "rollout_minus_ref": _summ(q_mean[:, 1] - q_mean[:, 0]),
                "rollout_q_adv": _summ(rollout_q_adv),
                "corr_adv_norm_vs_rollout_q_adv": _corr(selected_adv_norm, rollout_q_adv),
                "corr_adv_raw_vs_rollout_q_adv": _corr(selected_adv_raw, rollout_q_adv),
                "sign_agree_adv_norm_rollout_q_adv": float(np.mean(np.sign(selected_adv_norm) == np.sign(rollout_q_adv))),
                "corr_return_vs_rollout_q_adv": _corr(selected_returns, rollout_q_adv),
                "corr_value_vs_rollout_q_adv": _corr(selected_values, rollout_q_adv),
                "corr_residual_vs_rollout_q_adv": _corr(selected_returns - selected_values, rollout_q_adv),
            },
            "qpi_policy_gradient": qpi_grad,
            "selected": {
                "adv_norm": _summ(selected_adv_norm),
                "adv_raw": _summ(selected_adv_raw),
                "return": _summ(selected_returns),
                "value": _summ(selected_values),
                "residual": _summ(selected_returns - selected_values),
            },
            "rows": [
                {
                    "stage_sample": int(selected_np[i]),
                    "transition_index": int(selected_transitions[i]),
                    "history_row": int(history_rows[i]),
                    "step": int(step_indices_all[selected_np][i]),
                    "horizon": int(horizons[i]),
                    "adv_norm": float(selected_adv_norm[i]),
                    "adv_raw": float(selected_adv_raw[i]),
                    "return": float(selected_returns[i]),
                    "value": float(selected_values[i]),
                    "rollout_q_adv": float(rollout_q_adv[i]),
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
                    "actions": int(action_count),
                    "continuations": int(continuations),
                    "action_effect_std": action_effect_std,
                    "continuation_std": continuation_std,
                    "action_effect_to_continuation_snr": credit_snr,
                    "corr_adv_norm_vs_rollout_q_adv": payload["qpi"]["corr_adv_norm_vs_rollout_q_adv"],
                    "qpi_grad_split_half_cosine": qpi_grad.get("qpi_grad_split_half_cosine", 0.0),
                    "qpi_grad_cancellation_ratio": qpi_grad.get("qpi_grad_cancellation_ratio", 0.0),
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
