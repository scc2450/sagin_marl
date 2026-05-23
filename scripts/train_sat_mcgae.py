from __future__ import annotations

import argparse
import csv
import json
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
import torch.nn.functional as F

from scripts.audit_stage_critic_only_fit import (
    _clone_dataclass_tensors,
    _collect_one_rollout,
    _eval_critic,
    _make_learner,
)
from scripts.audit_stage_ppo_credit_alignment import _force_single_stage_config, _stage_optimizer_params

from sagin_marl.env.config import load_config
from sagin_marl.env.native_cuda import bindings as native_cuda
from sagin_marl.rl.structured_mappo import _index_dataclass
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver_group
from sagin_marl.utils.torch_compile_cache import report_torch_compile_cache


STAGE_SAT = 1


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _summ_tensor(x: torch.Tensor) -> dict[str, float]:
    if int(x.numel()) <= 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    y = x.detach().to(dtype=torch.float32).reshape(-1)
    return {
        "mean": float(y.mean().detach().cpu().item()),
        "std": float(y.std(unbiased=False).detach().cpu().item()),
        "min": float(y.min().detach().cpu().item()),
        "max": float(y.max().detach().cpu().item()),
    }


def _cuda_mem(prefix: str, device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        f"{prefix}_cuda_alloc_mb": float(torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)),
        f"{prefix}_cuda_reserved_mb": float(torch.cuda.memory_reserved(device) / (1024.0 * 1024.0)),
        f"{prefix}_cuda_max_alloc_mb": float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
        f"{prefix}_cuda_max_reserved_mb": float(torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)),
    }


def _stage_indices(stage_batch: Any, *, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(
        np.asarray(stage_batch.transition_indices, dtype=np.int64),
        dtype=torch.long,
        device=device,
    )


def _collect_sat_rollout(
    learner: Any,
    group: Any,
    *,
    rollout_env_steps: int,
    device: torch.device,
    seed: int | None,
) -> tuple[Any, Any, torch.Tensor, torch.Tensor, dict[str, float]]:
    if seed is not None:
        reset_many = getattr(group, "reset_many", None)
        if callable(reset_many):
            reset_many([int(seed) + env for env in range(len(group))])
    buffer, views, returns = _collect_one_rollout(
        learner,
        group,
        rollout_env_steps=int(rollout_env_steps),
        device=device,
        target="mc",
    )
    stage_batch = views.training_view.stage_batches.get(STAGE_SAT)
    if stage_batch is None or int(stage_batch.num_samples) <= 0:
        raise RuntimeError("SAT stage rollout contains no samples.")
    idx = _stage_indices(stage_batch, device=device)
    sat_mc_target = returns.index_select(0, idx).detach().to(device=device, dtype=torch.float32)
    reward_stats: dict[str, float] = {}
    rewards = getattr(views.training_view, "rewards", None)
    if rewards is not None:
        rt = torch.as_tensor(rewards, dtype=torch.float32, device=device).reshape(-1)
        reward_stats["transition_reward_mean"] = float(rt.mean().detach().cpu().item())
        reward_stats["transition_reward_std"] = float(rt.std(unbiased=False).detach().cpu().item())
    reward_stats["sat_mc_return_mean"] = float(sat_mc_target.mean().detach().cpu().item())
    reward_stats["sat_mc_return_std"] = float(sat_mc_target.std(unbiased=False).detach().cpu().item())
    return buffer, views, idx, sat_mc_target, reward_stats


def _train_sat_critic_on_stage(
    learner: Any,
    *,
    stage_batch: Any,
    target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    lr: float,
    epochs: int,
    minibatches: int,
    update_microbatch_size: int = 0,
    diagnose_timing: bool = False,
) -> tuple[dict[str, float], torch.Tensor, list[dict[str, float]]]:
    trace_rows: list[dict[str, float]] = []

    def _sync() -> None:
        if learner.device.type == "cuda":
            torch.cuda.synchronize(learner.device)

    # Clone the runtime-backed world rows before optimizer steps mutate/reuse native buffers.
    t_clone = time.perf_counter()
    world = _clone_dataclass_tensors(stage_batch.world_batch, device=learner.device)
    _sync()
    clone_sec = time.perf_counter() - t_clone

    t_before = time.perf_counter()
    before_pred, before_stats = _eval_critic(
        learner,
        stage_id=STAGE_SAT,
        world_bank=world,
        target=target,
        batch_size=max(1024, int(np.ceil(int(target.numel()) / max(int(minibatches), 1)))),
    )
    _sync()
    eval_before_sec = time.perf_counter() - t_before
    del before_pred
    for group in optimizer.param_groups:
        group["lr"] = float(lr)
    params = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
    n = int(target.numel())
    if n <= 0:
        raise RuntimeError("SAT critic target is empty.")
    mb_size = max(1, int(np.ceil(n / max(int(minibatches), 1))))
    update_microbatch_size_i = max(int(update_microbatch_size or 0), 0)
    target = target.detach().to(device=learner.device, dtype=torch.float32).reshape(-1)
    t_train = time.perf_counter()
    for epoch_idx in range(max(int(epochs), 0)):
        epoch_t0 = time.perf_counter()
        order = torch.randperm(n, device=target.device)
        if diagnose_timing:
            _sync()
        epoch_index_sec = 0.0
        epoch_forward_sec = 0.0
        epoch_loss_sec = 0.0
        epoch_zero_sec = 0.0
        epoch_backward_sec = 0.0
        epoch_clip_sec = 0.0
        epoch_step_sec = 0.0
        epoch_mb_min_sec: float | None = None
        epoch_mb_max_sec = 0.0
        epoch_mb_count = 0
        for start in range(0, n, mb_size):
            mb_t0 = time.perf_counter()
            idx_full = order[start : start + mb_size]
            full_count = int(idx_full.numel())
            use_update_microbatches = update_microbatch_size_i > 0 and full_count > update_microbatch_size_i
            if diagnose_timing:
                t = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                _sync()
                epoch_zero_sec += time.perf_counter() - t
                chunk_size = update_microbatch_size_i if use_update_microbatches else full_count
                for chunk_start in range(0, full_count, max(chunk_size, 1)):
                    idx = idx_full[chunk_start : chunk_start + chunk_size]
                    loss_scale = float(int(idx.numel())) / float(max(full_count, 1))

                    t = time.perf_counter()
                    batch = _index_dataclass(world, idx)
                    _sync()
                    epoch_index_sec += time.perf_counter() - t

                    t = time.perf_counter()
                    pred = learner._stage_value_eval_from_batch(STAGE_SAT, batch)
                    _sync()
                    epoch_forward_sec += time.perf_counter() - t

                    t = time.perf_counter()
                    loss = F.mse_loss(pred, target.index_select(0, idx)) * loss_scale
                    _sync()
                    epoch_loss_sec += time.perf_counter() - t

                    t = time.perf_counter()
                    loss.backward()
                    _sync()
                    epoch_backward_sec += time.perf_counter() - t

                t = time.perf_counter()
                torch.nn.utils.clip_grad_norm_(params, float(learner.max_grad_norm))
                _sync()
                epoch_clip_sec += time.perf_counter() - t

                t = time.perf_counter()
                optimizer.step()
                _sync()
                epoch_step_sec += time.perf_counter() - t
            else:
                optimizer.zero_grad(set_to_none=True)
                chunk_size = update_microbatch_size_i if use_update_microbatches else full_count
                for chunk_start in range(0, full_count, max(chunk_size, 1)):
                    idx = idx_full[chunk_start : chunk_start + chunk_size]
                    loss_scale = float(int(idx.numel())) / float(max(full_count, 1))
                    pred = learner._stage_value_eval_from_batch(STAGE_SAT, _index_dataclass(world, idx))
                    loss = F.mse_loss(pred, target.index_select(0, idx)) * loss_scale
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(params, float(learner.max_grad_norm))
                optimizer.step()
            if diagnose_timing:
                mb_sec = time.perf_counter() - mb_t0
                epoch_mb_count += 1
                epoch_mb_min_sec = mb_sec if epoch_mb_min_sec is None else min(epoch_mb_min_sec, mb_sec)
                epoch_mb_max_sec = max(epoch_mb_max_sec, mb_sec)
        if diagnose_timing:
            _sync()
            epoch_sec = time.perf_counter() - epoch_t0
            trace_rows.append(
                {
                    "epoch": float(epoch_idx + 1),
                    "epoch_sec": float(epoch_sec),
                    "epoch_index_sec": float(epoch_index_sec),
                    "epoch_forward_sec": float(epoch_forward_sec),
                    "epoch_loss_sec": float(epoch_loss_sec),
                    "epoch_zero_sec": float(epoch_zero_sec),
                    "epoch_backward_sec": float(epoch_backward_sec),
                    "epoch_clip_sec": float(epoch_clip_sec),
                    "epoch_step_sec": float(epoch_step_sec),
                    "epoch_minibatches": float(epoch_mb_count),
                    "epoch_mb_min_sec": float(epoch_mb_min_sec or 0.0),
                    "epoch_mb_max_sec": float(epoch_mb_max_sec),
                    **_cuda_mem("epoch_end", learner.device),
                }
            )
    if learner.device.type == "cuda":
        torch.cuda.synchronize(learner.device)
    train_loop_sec = time.perf_counter() - t_train

    # PyTorch optimizer/backward leaves a large cached block set behind.  The
    # following value eval is inference-only and was repeatedly 4-10x slower
    # without releasing the cache first; this changes allocator state, not math.
    cache_clear_sec = 0.0
    if learner.device.type == "cuda":
        t_cache = time.perf_counter()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(learner.device)
        cache_clear_sec = time.perf_counter() - t_cache

    t_after = time.perf_counter()
    after_pred_np, after_stats = _eval_critic(
        learner,
        stage_id=STAGE_SAT,
        world_bank=world,
        target=target,
        batch_size=max(1024, int(np.ceil(int(target.numel()) / max(int(minibatches), 1)))),
    )
    if learner.device.type == "cuda":
        torch.cuda.synchronize(learner.device)
    eval_after_sec = time.perf_counter() - t_after
    after_values = torch.as_tensor(after_pred_np, dtype=torch.float32, device=learner.device).reshape(-1)
    return {
        "critic_lr": float(lr),
        "critic_epochs": float(epochs),
        "critic_minibatches": float(minibatches),
        "critic_update_microbatch_size": float(update_microbatch_size_i),
        "critic_clone_sec": float(clone_sec),
        "critic_eval_before_sec": float(eval_before_sec),
        "critic_train_loop_sec": float(train_loop_sec),
        "critic_cache_clear_sec": float(cache_clear_sec),
        "critic_eval_after_sec": float(eval_after_sec),
        "critic_ev_before": float(before_stats["ev"]),
        "critic_ev_after": float(after_stats["ev"]),
        "critic_mse_before": float(before_stats["mse"]),
        "critic_mse_after": float(after_stats["mse"]),
        "critic_corr_after": float(after_stats["corr"]),
        "critic_train_final_ev": float(after_stats["ev"]),
        "critic_train_final_mse": float(after_stats["mse"]),
    }, after_values, trace_rows


def _eval_sat_values_for_stage(
    learner: Any,
    *,
    stage_batch: Any,
    device: torch.device,
) -> torch.Tensor:
    sample_count = int(stage_batch.num_samples)
    if sample_count <= 0:
        return torch.zeros((0,), dtype=torch.float32, device=device)
    chunks: list[torch.Tensor] = []
    batch_size = 2048
    with torch.no_grad():
        for start in range(0, sample_count, batch_size):
            idx = torch.arange(start, min(start + batch_size, sample_count), dtype=torch.long, device=device)
            pred = learner._stage_value_eval_from_batch(
                STAGE_SAT,
                _index_dataclass(stage_batch.world_batch, idx),
            )
            # Compiled CUDA value calls may reuse graph output storage across
            # invocations; clone before keeping the chunk.
            chunks.append(pred.detach().to(device=device, dtype=torch.float32).clone())
    return torch.cat(chunks, dim=0)


def _dense_sat_stage_layout(stage_batch: Any, *, sample_count: int) -> tuple[int, int] | None:
    transition_idx = np.asarray(stage_batch.transition_indices, dtype=np.int64).reshape(-1)
    env_idx = np.asarray(stage_batch.env_indices, dtype=np.int64).reshape(-1)
    if int(transition_idx.size) != int(sample_count) or int(env_idx.size) != int(sample_count):
        return None
    if sample_count <= 0 or env_idx.size <= 0:
        return None
    if int(env_idx.min()) != 0:
        return None
    num_envs = int(env_idx.max()) + 1
    if num_envs <= 0 or sample_count % num_envs != 0:
        return None
    num_steps = sample_count // num_envs
    expected_env = np.tile(np.arange(num_envs, dtype=np.int64), num_steps)
    if not np.array_equal(env_idx, expected_env):
        return None
    expected_transition = (
        np.arange(num_steps, dtype=np.int64).reshape(num_steps, 1) * (3 * num_envs)
        + 3 * np.arange(num_envs, dtype=np.int64).reshape(1, num_envs)
        + STAGE_SAT
    ).reshape(-1)
    if not np.array_equal(transition_idx, expected_transition):
        return None
    return int(num_steps), int(num_envs)


def _sat_only_gae_native_dense(
    learner: Any,
    *,
    stage_batch: Any,
    sat_values: torch.Tensor,
    sat_mc_target: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if device.type != "cuda":
        return None
    sample_count = int(stage_batch.num_samples)
    layout = _dense_sat_stage_layout(stage_batch, sample_count=sample_count)
    if layout is None:
        return None
    num_steps, num_envs = layout
    terminated = getattr(stage_batch, "terminated", None)
    truncated = getattr(stage_batch, "truncated", None)
    if not torch.is_tensor(terminated) or not torch.is_tensor(truncated):
        return None
    values = sat_values.detach().to(device=device, dtype=torch.float32).reshape(-1).contiguous()
    mc = sat_mc_target.detach().to(device=device, dtype=torch.float32).reshape(-1).contiguous()
    term = terminated.detach().to(device=device, dtype=torch.bool).reshape(-1).contiguous()
    trunc = truncated.detach().to(device=device, dtype=torch.bool).reshape(-1).contiguous()
    if int(values.numel()) != sample_count or int(mc.numel()) != sample_count:
        return None
    if int(term.numel()) != sample_count or int(trunc.numel()) != sample_count:
        return None
    ret = torch.empty_like(values)
    adv = torch.empty_like(values)
    native_cuda.stage_mc_gae(
        values,
        mc,
        term,
        trunc,
        ret,
        adv,
        num_steps=num_steps,
        num_envs=num_envs,
        gamma=float(learner.gamma),
        gae_lambda=float(learner.gae_lambda),
    )
    return ret, adv


def _sat_only_gae_from_mc_targets(
    learner: Any,
    *,
    stage_batch: Any,
    sat_mc_target: torch.Tensor,
    views: Any,
    device: torch.device,
    sat_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build SAT-decision GAE using only SAT critic values.

    The rollout MC target already contains the finite-horizon return from each
    SAT decision point.  For adjacent SAT decisions in the same env episode:

        MC_t = r_between_t + gamma * MC_{t+1}

    so we can recover the collapsed reward between SAT decisions without
    evaluating accel/bw critics.  This keeps the SAT-only loop honest and avoids
    the expensive all-stage value override path.
    """

    sample_count = int(stage_batch.num_samples)
    if sat_values is None:
        sat_values = _eval_sat_values_for_stage(learner, stage_batch=stage_batch, device=device)
    else:
        sat_values = sat_values.detach().to(device=device, dtype=torch.float32).reshape(-1)
    mc = sat_mc_target.detach().to(device=device, dtype=torch.float32).reshape(-1)
    if int(mc.numel()) != sample_count:
        raise RuntimeError(f"SAT MC target length {int(mc.numel())} != samples {sample_count}.")
    if int(sat_values.numel()) != sample_count:
        raise RuntimeError(f"SAT value length {int(sat_values.numel())} != samples {sample_count}.")

    native_result = _sat_only_gae_native_dense(
        learner,
        stage_batch=stage_batch,
        sat_values=sat_values,
        sat_mc_target=mc,
        device=device,
    )
    if native_result is not None:
        ret, adv = native_result
        return ret, adv, sat_values

    transition_idx = np.asarray(stage_batch.transition_indices, dtype=np.int64).reshape(-1)
    env_idx = np.asarray(stage_batch.env_indices, dtype=np.int64).reshape(-1)
    if int(transition_idx.size) != sample_count or int(env_idx.size) != sample_count:
        raise RuntimeError("SAT stage transition/env index length mismatch.")
    if np.any((transition_idx - STAGE_SAT) % 3 != 0):
        raise RuntimeError("SAT stage transition indices are not stage-1 aligned.")

    return_view = views.return_view
    terminated = np.asarray(return_view.terminated, dtype=bool).reshape(-1)
    truncated = np.asarray(return_view.truncated, dtype=bool).reshape(-1)
    transition_count = int(return_view.transition_count)

    adv = torch.zeros((sample_count,), dtype=torch.float32, device=device)
    ret = torch.zeros((sample_count,), dtype=torch.float32, device=device)
    gamma = float(learner.gamma)
    lam = float(learner.gae_lambda)

    by_env: dict[int, list[int]] = {}
    order = np.argsort(transition_idx, kind="stable")
    for pos in order.tolist():
        by_env.setdefault(int(env_idx[pos]), []).append(int(pos))

    for positions in by_env.values():
        next_adv = torch.zeros((), dtype=torch.float32, device=device)
        next_value = torch.zeros((), dtype=torch.float32, device=device)
        next_mc = torch.zeros((), dtype=torch.float32, device=device)
        have_next = False
        for pos in reversed(positions):
            sat_tidx = int(transition_idx[pos])
            bw_tidx = sat_tidx + 1
            ended_here = False
            if 0 <= bw_tidx < transition_count:
                ended_here = bool(terminated[bw_tidx] or truncated[bw_tidx])
            if have_next and not ended_here:
                collapsed_reward = mc[pos] - float(gamma) * next_mc
                bootstrap_value = next_value
                bootstrap_adv = next_adv
            else:
                collapsed_reward = mc[pos]
                bootstrap_value = torch.zeros((), dtype=torch.float32, device=device)
                bootstrap_adv = torch.zeros((), dtype=torch.float32, device=device)
            delta = collapsed_reward + float(gamma) * bootstrap_value - sat_values[pos]
            adv_pos = delta + float(gamma) * float(lam) * bootstrap_adv
            adv[pos] = adv_pos
            ret[pos] = adv_pos + sat_values[pos]
            next_adv = adv_pos
            next_value = sat_values[pos]
            next_mc = mc[pos]
            have_next = not ended_here
    return ret, adv, sat_values


def _normalize_stage_advantage(adv: torch.Tensor, *, enabled: bool) -> torch.Tensor:
    adv = adv.detach().to(dtype=torch.float32)
    if not bool(enabled) or int(adv.numel()) <= 1:
        return adv
    return (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1.0e-8)


def _sat_actor_update_full_stage(
    learner: Any,
    *,
    stage_batch: Any,
    stage_advantages: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    minibatches: int,
) -> dict[str, float]:
    params = _stage_optimizer_params(learner.actor, STAGE_SAT)
    if not params:
        raise RuntimeError("SAT actor has no trainable parameters.")
    device = learner.device
    num_agents = int(stage_batch.num_agents)
    sample_count = int(stage_batch.num_samples)
    all_idx = torch.arange(sample_count, dtype=torch.long, device=device)
    all_actions = stage_batch.actions.to(device=device)
    batch_eval = 1024
    old_chunks: list[torch.Tensor] = []
    old_t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, sample_count, batch_eval):
            idx = all_idx[start : start + batch_eval]
            flat_idx = (
                idx.view(-1, 1) * int(num_agents)
                + torch.arange(num_agents, dtype=torch.long, device=device).view(1, int(num_agents))
            ).reshape(-1)
            local_i = _index_dataclass(stage_batch.local_batch, flat_idx)
            action_i = all_actions.index_select(0, idx)
            lp_i, _ent_i, _out_i = learner._stage_actor_eval_from_batch(STAGE_SAT, local_i, action_i, num_agents)
            old_chunks.append(lp_i.detach())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    old_logprob_sec = time.perf_counter() - old_t0
    old_logprob = torch.cat(old_chunks, dim=0).to(device=device, dtype=torch.float32)
    adv = stage_advantages.detach().to(device=device, dtype=torch.float32).reshape(-1)
    if int(adv.numel()) != sample_count:
        raise RuntimeError(f"advantage length {int(adv.numel())} != SAT stage samples {sample_count}.")
    entropy_coef = float(learner.entropy_coef_by_stage[STAGE_SAT])
    mb_size = max(1, int(np.ceil(sample_count / max(int(minibatches), 1))))
    last_policy_loss = 0.0
    last_entropy = 0.0
    last_loss = 0.0
    last_grad_norm = 0.0
    last_kl = 0.0
    last_clip_frac = 0.0
    update_loop_sec = 0.0
    for _epoch in range(max(int(epochs), 1)):
        epoch_t0 = time.perf_counter()
        order = torch.randperm(sample_count, device=device)
        for start in range(0, sample_count, mb_size):
            idx = order[start : start + mb_size]
            optimizer.zero_grad(set_to_none=True)
            mb_total = int(idx.numel())
            chunk_size = 2048
            policy_sum = 0.0
            entropy_sum = 0.0
            kl_sum = 0.0
            clip_sum = 0.0
            for chunk_start in range(0, mb_total, chunk_size):
                chunk_idx = idx[chunk_start : chunk_start + chunk_size]
                weight = float(chunk_idx.numel()) / float(max(mb_total, 1))
                flat_idx = (
                    chunk_idx.view(-1, 1) * int(num_agents)
                    + torch.arange(num_agents, dtype=torch.long, device=device).view(1, int(num_agents))
                ).reshape(-1)
                local_i = _index_dataclass(stage_batch.local_batch, flat_idx)
                action_i = all_actions.index_select(0, chunk_idx)
                logprob, entropy, _out = learner._stage_actor_eval_from_batch(STAGE_SAT, local_i, action_i, num_agents)
                old_i = old_logprob.index_select(0, chunk_idx).to(dtype=logprob.dtype)
                adv_i = adv.index_select(0, chunk_idx).to(dtype=logprob.dtype)
                log_ratio = torch.clamp(logprob - old_i.detach(), min=-20.0, max=20.0)
                ratio = torch.exp(log_ratio)
                clipped = torch.clamp(ratio, 1.0 - float(learner.clip_ratio), 1.0 + float(learner.clip_ratio))
                policy_loss = -torch.minimum(ratio * adv_i.detach(), clipped * adv_i.detach()).mean()
                entropy_mean = entropy.mean() if int(entropy.numel()) > 0 else torch.zeros((), device=device)
                loss = (policy_loss - entropy_coef * entropy_mean) * float(weight)
                loss.backward()
                with torch.no_grad():
                    policy_sum += float(policy_loss.detach().cpu().item()) * float(weight)
                    entropy_sum += float(entropy_mean.detach().cpu().item()) * float(weight)
                    kl_sum += float((old_i.detach() - logprob.detach()).mean().cpu().item()) * float(weight)
                    clip_sum += (
                        float((torch.abs(ratio.detach() - 1.0) > float(learner.clip_ratio)).float().mean().cpu().item())
                        * float(weight)
                    )
            grad_norm = torch.nn.utils.clip_grad_norm_(params, float(learner.max_grad_norm))
            optimizer.step()
            with torch.no_grad():
                last_policy_loss = float(policy_sum)
                last_entropy = float(entropy_sum)
                last_loss = float(policy_sum - entropy_coef * entropy_sum)
                last_grad_norm = float(torch.as_tensor(grad_norm).detach().cpu().item())
                last_kl = float(kl_sum)
                last_clip_frac = float(clip_sum)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        update_loop_sec += time.perf_counter() - epoch_t0
    return {
        "actor_samples": float(sample_count),
        "actor_epochs": float(epochs),
        "actor_minibatches": float(minibatches),
        "policy_loss_sat": float(last_policy_loss),
        "entropy_sat": float(last_entropy),
        "actor_loss_sat": float(last_loss),
        "grad_norm_sat": float(last_grad_norm),
        "approx_kl_sat": float(last_kl),
        "clip_frac_sat": float(last_clip_frac),
        "old_logprob_mean": float(old_logprob.mean().detach().cpu().item()),
        "adv_mean": float(adv.mean().detach().cpu().item()),
        "adv_std": float(adv.std(unbiased=False).detach().cpu().item()),
        "actor_old_logprob_sec": float(old_logprob_sec),
        "actor_update_loop_sec": float(update_loop_sec),
    }


def _write_metrics(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="SAT-only MC-critic + A_gae(V) training loop.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--rollout_env_steps", type=int, default=250)
    parser.add_argument("--reward_mode", default="positive_weighted_workload_level")
    parser.add_argument("--cold_critic_lr", type=float, default=None)
    parser.add_argument("--cold_critic_epochs", type=int, default=None)
    parser.add_argument("--tracking_critic_lr", type=float, default=None)
    parser.add_argument("--tracking_critic_epochs", type=int, default=None)
    parser.add_argument("--critic_minibatches", type=int, default=None)
    parser.add_argument("--critic_update_microbatch_size", type=int, default=None)
    parser.add_argument("--actor_lr", type=float, default=None)
    parser.add_argument("--actor_epochs", type=int, default=None)
    parser.add_argument("--actor_minibatches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=45210)
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument(
        "--diagnose_critic_timing",
        action="store_true",
        help="Record per-epoch/per-minibatch critic timing without changing the training update math.",
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
    _force_single_stage_config(cfg, stage_id=STAGE_SAT, reward_mode=str(args.reward_mode))
    report_torch_compile_cache(context="train_sat_mcgae", device=device, cfg=cfg)
    cfg.actor_advantage_normalize_enabled = True
    cfg.stagewise_advantage_norm_enabled = True
    cfg.checkpoint_eval_enabled = False
    cfg.train_trace_enabled = False

    cold_critic_lr = float(
        args.cold_critic_lr
        if args.cold_critic_lr is not None
        else getattr(cfg, "sat_mcgae_cold_critic_lr", 1.0e-3)
    )
    cold_critic_epochs = int(
        args.cold_critic_epochs
        if args.cold_critic_epochs is not None
        else getattr(cfg, "sat_mcgae_cold_critic_epochs", 20)
    )
    tracking_critic_lr = float(
        args.tracking_critic_lr
        if args.tracking_critic_lr is not None
        else getattr(cfg, "sat_mcgae_tracking_critic_lr", 3.0e-4)
    )
    tracking_critic_epochs = int(
        args.tracking_critic_epochs
        if args.tracking_critic_epochs is not None
        else getattr(cfg, "sat_mcgae_tracking_critic_epochs", 5)
    )
    critic_minibatches = int(
        args.critic_minibatches
        if args.critic_minibatches is not None
        else getattr(cfg, "sat_mcgae_critic_minibatches", 8)
    )
    critic_update_microbatch_size = int(
        args.critic_update_microbatch_size
        if args.critic_update_microbatch_size is not None
        else getattr(cfg, "sat_mcgae_critic_update_microbatch_size", 0)
    )
    actor_lr = float(args.actor_lr if args.actor_lr is not None else getattr(cfg, "sat_mcgae_actor_lr", 3.0e-4))
    actor_epochs = int(args.actor_epochs if args.actor_epochs is not None else getattr(cfg, "sat_mcgae_actor_epochs", 5))
    actor_minibatches = int(
        args.actor_minibatches
        if args.actor_minibatches is not None
        else getattr(cfg, "sat_mcgae_actor_minibatches", 1)
    )

    if str(getattr(cfg, "critic_value_mode", "")).lower() == "global_linear":
        raise RuntimeError(
            "scripts/train_sat_mcgae.py needs a trainable critic; "
            "critic_value_mode=global_linear is not compatible with MC-GAE critic fitting. "
            "Use critic_value_mode=relational for this script."
        )

    learner, _actor, _critic = _make_learner(cfg, device=device, stage_id=STAGE_SAT)
    critic_params = [param for param in learner.critic.parameters() if param.requires_grad]
    if not critic_params:
        raise RuntimeError("SAT MC-GAE training requires trainable critic parameters.")
    critic_optimizer = torch.optim.Adam(critic_params, lr=float(cold_critic_lr))
    sat_params = _stage_optimizer_params(learner.actor, STAGE_SAT)
    if not sat_params:
        raise RuntimeError("SAT actor has no trainable parameters.")
    actor_optimizer = torch.optim.Adam(sat_params, lr=float(actor_lr))
    group = make_structured_driver_group(cfg, num_envs=int(args.num_envs), backend="sync", mode="train")
    metrics: list[dict[str, float]] = []
    try:
        learner.bind_native_runtime_contract(group)
        print(
            "SAT MC-GAE train | "
            f"envs={int(args.num_envs)} rollout={int(args.rollout_env_steps)} updates={int(args.updates)} "
            f"reward={cfg.reward_mode} critic={cold_critic_epochs}/{tracking_critic_epochs} "
            f"actor_epochs={actor_epochs}",
            flush=True,
        )
        for update in range(max(int(args.updates), 0)):
            if bool(args.diagnose_critic_timing) and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            update_t0 = time.perf_counter()
            rollout_seed = int(args.seed) + update * 100_000
            t0 = time.perf_counter()
            buffer, views, _stage_idx, sat_mc_target, reward_stats = _collect_sat_rollout(
                learner,
                group,
                rollout_env_steps=int(args.rollout_env_steps),
                device=device,
                seed=rollout_seed,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            collect_sec = time.perf_counter() - t0
            stage_batch = views.training_view.stage_batches[STAGE_SAT]

            critic_lr = float(cold_critic_lr if update == 0 else tracking_critic_lr)
            critic_epochs = int(cold_critic_epochs if update == 0 else tracking_critic_epochs)
            t1 = time.perf_counter()
            critic_stats, stage_values_after_critic, critic_trace_rows = _train_sat_critic_on_stage(
                learner,
                stage_batch=stage_batch,
                target=sat_mc_target,
                optimizer=critic_optimizer,
                lr=critic_lr,
                epochs=critic_epochs,
                minibatches=int(critic_minibatches),
                update_microbatch_size=int(critic_update_microbatch_size),
                diagnose_timing=bool(args.diagnose_critic_timing),
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            critic_sec = time.perf_counter() - t1

            t2 = time.perf_counter()
            _returns_gae, stage_adv, stage_values = _sat_only_gae_from_mc_targets(
                learner,
                stage_batch=stage_batch,
                sat_mc_target=sat_mc_target,
                views=views,
                device=device,
                sat_values=stage_values_after_critic,
            )
            stage_adv_norm = _normalize_stage_advantage(stage_adv, enabled=True)
            actor_stats = _sat_actor_update_full_stage(
                learner,
                stage_batch=stage_batch,
                stage_advantages=stage_adv_norm,
                optimizer=actor_optimizer,
                epochs=int(actor_epochs),
                minibatches=int(actor_minibatches),
            )
            sync_native = getattr(learner, "_sync_native_actor_cuda_bindings_after_update", None)
            if callable(sync_native):
                sync_native()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            actor_sec = time.perf_counter() - t2
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
                "cuda_empty_cache_sec": float(empty_cache_sec),
                "iteration_sec": float(time.perf_counter() - update_t0),
                "critic_lr": float(critic_lr),
                "critic_epochs": float(critic_epochs),
                "samples": float(stage_batch.num_samples),
                "sat_value_mean": float(stage_values.mean().detach().cpu().item()),
                "sat_value_std": float(stage_values.std(unbiased=False).detach().cpu().item()),
                **reward_stats,
                **critic_stats,
                **actor_stats,
                **{f"raw_adv_{k}": v for k, v in _summ_tensor(stage_adv).items()},
                **{f"norm_adv_{k}": v for k, v in _summ_tensor(stage_adv_norm).items()},
                **_cuda_mem("update_end", device),
            }
            metrics.append(row)
            _write_metrics(run_dir / "metrics.csv", metrics)
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            if bool(args.diagnose_critic_timing):
                with (run_dir / "critic_timing.jsonl").open("a", encoding="utf-8") as f:
                    for trace_row in critic_trace_rows:
                        trace_out = {
                            "update": float(update + 1),
                            "critic_lr": float(critic_lr),
                            "critic_epochs": float(critic_epochs),
                            "critic_minibatches": float(critic_minibatches),
                            "critic_update_microbatch_size": float(critic_update_microbatch_size),
                            "samples": float(stage_batch.num_samples),
                            "collect_sec": float(collect_sec),
                            "critic_sec": float(critic_sec),
                            "actor_sec": float(actor_sec),
                            **trace_row,
                        }
                        f.write(json.dumps(trace_out, ensure_ascii=False, sort_keys=True) + "\n")
            print(
                f"Update {update + 1}/{int(args.updates)} "
                f"mc={row['sat_mc_return_mean']:.3f} critic_ev={row['critic_ev_after']:.3f} "
                f"adv_std={row['raw_adv_std']:.3f} kl={row['approx_kl_sat']:.5f} "
                f"ent={row['entropy_sat']:.3f} time={row['iteration_sec']:.1f}s",
                flush=True,
            )
            if int(args.save_every) > 0 and (update + 1) % int(args.save_every) == 0:
                torch.save(
                    {
                        "update": update + 1,
                        "actor": learner.actor.state_dict(),
                        "critic": learner.critic.state_dict(),
                        "config": vars(cfg),
                    },
                    run_dir / f"checkpoint_u{update + 1:04d}.pt",
                )
        torch.save(
            {
                "update": int(args.updates),
                "actor": learner.actor.state_dict(),
                "critic": learner.critic.state_dict(),
                "config": vars(cfg),
            },
            run_dir / "checkpoint_final.pt",
        )
    finally:
        close_structured_env_group(group)


if __name__ == "__main__":
    main()
