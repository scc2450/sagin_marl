from __future__ import annotations

import argparse
import csv
import json
import math
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
import torch._dynamo as torch_dynamo
import torch.nn.functional as F

from sagin_marl.env.config import load_config
from sagin_marl.env.native_cuda import bindings as native_cuda
from sagin_marl.rl.distributions import squash_action
from sagin_marl.rl.structured_actor import _safe_categorical_logits
from sagin_marl.rl.structured_mappo import _index_dataclass, _slice_dataclass
from sagin_marl.rl.stage_mcgae import (
    clone_dataclass_tensors as _clone_dataclass_tensors,
    collect_one_rollout as _collect_one_rollout,
    eval_critic as _eval_critic,
    force_single_stage_config as _force_single_stage_config,
    make_learner as _make_learner,
    stage_optimizer_params as _stage_optimizer_params,
)
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver_group
from sagin_marl.utils.torch_compile_cache import report_torch_compile_cache


STAGE_ID = {"accel": 0, "sat": 1, "bw": 2}
STAGE_NAME = {0: "accel", 1: "sat", 2: "bw"}


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


def _summ_tensor_extended(x: torch.Tensor) -> dict[str, float]:
    stats = _summ_tensor(x)
    if int(x.numel()) <= 0:
        stats.update({"p10": 0.0, "p50": 0.0, "p90": 0.0})
        return stats
    y = x.detach().to(dtype=torch.float32).reshape(-1)
    stats.update(
        {
            "p10": float(torch.quantile(y, 0.10).detach().cpu().item()),
            "p50": float(torch.quantile(y, 0.50).detach().cpu().item()),
            "p90": float(torch.quantile(y, 0.90).detach().cpu().item()),
        }
    )
    return stats


def _prefixed_stats(prefix: str, x: torch.Tensor, *, extended: bool = False) -> dict[str, float]:
    stats = _summ_tensor_extended(x) if bool(extended) else _summ_tensor(x)
    return {f"{prefix}_{key}": float(value) for key, value in stats.items()}


def _critic_np_stats(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    if int(p.size) != int(t.size) or int(t.size) <= 0:
        return {"ev": float("nan"), "mse": float("nan"), "corr": float("nan")}
    mse = float(np.mean((p - t) ** 2))
    if int(t.size) <= 1:
        ev = 0.0
        corr = 0.0
    else:
        var = float(np.var(t))
        ev = 0.0 if var <= 1.0e-12 else 1.0 - float(np.var(t - p)) / var
        if float(np.std(p)) <= 1.0e-12 or float(np.std(t)) <= 1.0e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(p, t)[0, 1])
    return {"ev": float(ev), "mse": float(mse), "corr": float(corr)}


def _cuda_mem(prefix: str, device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        f"{prefix}_cuda_alloc_mb": float(torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)),
        f"{prefix}_cuda_reserved_mb": float(torch.cuda.memory_reserved(device) / (1024.0 * 1024.0)),
        f"{prefix}_cuda_max_alloc_mb": float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
        f"{prefix}_cuda_max_reserved_mb": float(torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)),
    }


def _stage_cfg_value(cfg: Any, generic_name: str, sat_name: str, default: Any) -> Any:
    generic = getattr(cfg, generic_name, None)
    if generic is not None:
        return generic
    return getattr(cfg, sat_name, default)


def _stage_cfg_bool(cfg: Any, generic_name: str, sat_name: str, default: bool) -> bool:
    value = _stage_cfg_value(cfg, generic_name, sat_name, None)
    if value is None:
        return bool(default)
    return bool(value)


def _configure_accel_safety_for_training(cfg: Any, *, stage_id: int) -> None:
    if int(stage_id) != 0:
        return

    # Native safety shield and the regular avoidance layer are alternatives in
    # the CUDA accel step kernel.  Use the safer native shield by default for
    # accel training now that profiling shows no meaningful speed penalty.
    cfg.safety_shield_enabled = True
    cfg.safety_shield_solver = "NATIVE_CUDA"
    cfg.avoidance_enabled = False
    cfg.danger_imitation_enabled = True
    if float(getattr(cfg, "danger_imitation_coef", 0.0) or 0.0) <= 0.0:
        cfg.danger_imitation_coef = 0.1
    cfg.danger_imitation_trigger_mode = "intervention_any"


def _sync_danger_imitation_to_learner(learner: Any, cfg: Any) -> None:
    coef = max(float(getattr(cfg, "danger_imitation_coef", 0.0) or 0.0), 0.0)
    learner.danger_imitation_coef = coef
    learner.danger_imitation_enabled = bool(getattr(cfg, "danger_imitation_enabled", False)) and coef > 0.0


def _stage_indices(stage_batch: Any, *, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(
        np.asarray(stage_batch.transition_indices, dtype=np.int64),
        dtype=torch.long,
        device=device,
    )


def _as_bool_np(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().to("cpu", dtype=torch.bool).numpy().reshape(-1)
    return np.asarray(value, dtype=bool).reshape(-1)


def _collect_stage_rollout(
    learner: Any,
    group: Any,
    *,
    stage_id: int,
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
    stage_batch = views.training_view.stage_batches.get(int(stage_id))
    if stage_batch is None or int(stage_batch.num_samples) <= 0:
        raise RuntimeError(f"{STAGE_NAME[int(stage_id)]} stage rollout contains no samples.")
    idx = _stage_indices(stage_batch, device=device)
    mc_target = returns.index_select(0, idx).detach().to(device=device, dtype=torch.float32)
    reward_stats: dict[str, float] = {}
    rewards = getattr(views.training_view, "rewards", None)
    if rewards is not None:
        rt = torch.as_tensor(rewards, dtype=torch.float32, device=device).reshape(-1)
        reward_stats["transition_reward_mean"] = float(rt.mean().detach().cpu().item())
        reward_stats["transition_reward_std"] = float(rt.std(unbiased=False).detach().cpu().item())
    name = STAGE_NAME[int(stage_id)]
    reward_stats["stage_mc_return_mean"] = float(mc_target.mean().detach().cpu().item())
    reward_stats["stage_mc_return_std"] = float(mc_target.std(unbiased=False).detach().cpu().item())
    reward_stats[f"{name}_mc_return_mean"] = reward_stats["stage_mc_return_mean"]
    reward_stats[f"{name}_mc_return_std"] = reward_stats["stage_mc_return_std"]
    return buffer, views, idx, mc_target, reward_stats


def _train_stage_critic_on_stage(
    learner: Any,
    *,
    stage_id: int,
    stage_batch: Any,
    target: torch.Tensor,
    aux_target: torch.Tensor | None = None,
    aux_coef: float = 0.0,
    optimizer: torch.optim.Optimizer,
    lr: float,
    epochs: int,
    minibatches: int,
    update_microbatch_size: int = 0,
    eval_before_enabled: bool | None = None,
    diagnose_timing: bool = False,
) -> tuple[dict[str, float], torch.Tensor, list[dict[str, float]]]:
    trace_rows: list[dict[str, float]] = []
    stage_id_i = int(stage_id)

    def _sync() -> None:
        if learner.device.type == "cuda":
            torch.cuda.synchronize(learner.device)

    # Runtime-backed world rows can be reused by native history buffers; clone
    # before optimizer steps so the critic target sees a stable stage bank.
    t_clone = time.perf_counter()
    world = _clone_dataclass_tensors(stage_batch.world_batch, device=learner.device)
    _sync()
    clone_sec = time.perf_counter() - t_clone

    n = int(target.numel())
    update_microbatch_size_i = max(int(update_microbatch_size or 0), 0)
    # When torch.compile strict recompile checks are enabled, the critic value
    # function must see a stable batch shape across updates.  Do not shrink the
    # requested microbatch to a divisor of the current sample count; instead pad
    # the final chunk to the requested fixed size.
    critic_value_chunk_size = update_microbatch_size_i if update_microbatch_size_i > 0 else min(max(n, 1), 2048)
    eval_batch_size = max(1, int(critic_value_chunk_size))
    eval_before_i = _stage_cfg_bool(
        getattr(learner, "cfg", None),
        "stage_mcgae_critic_eval_before_enabled",
        "sat_mcgae_critic_eval_before_enabled",
        False,
    )
    if eval_before_enabled is not None:
        eval_before_i = bool(eval_before_enabled)
    if bool(eval_before_i):
        t_before = time.perf_counter()
        before_pred, before_stats = _eval_critic(
            learner,
            stage_id=stage_id_i,
            world_bank=world,
            target=target,
            batch_size=eval_batch_size,
            pad_to_batch_size=True,
        )
        _sync()
        eval_before_sec = time.perf_counter() - t_before
        del before_pred
    else:
        eval_before_sec = 0.0
        before_stats = {"ev": float("nan"), "mse": float("nan")}

    for group in optimizer.param_groups:
        group["lr"] = float(lr)
    params = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
    if n <= 0:
        raise RuntimeError(f"{STAGE_NAME[stage_id_i]} critic target is empty.")
    target = target.detach().to(device=learner.device, dtype=torch.float32).reshape(-1)
    aux_coef_f = max(float(aux_coef or 0.0), 0.0)
    if aux_target is not None and aux_coef_f > 0.0:
        aux_target_t: torch.Tensor | None = aux_target.detach().to(
            device=learner.device,
            dtype=torch.float32,
        ).reshape(-1)
        if int(aux_target_t.numel()) != n:
            raise RuntimeError(
                f"{STAGE_NAME[stage_id_i]} aux critic target length {int(aux_target_t.numel())} != samples {n}."
            )
    else:
        aux_target_t = None
        aux_coef_f = 0.0
    train_main_loss_sum = 0.0
    train_aux_loss_sum = 0.0
    train_loss_chunks = 0

    t_train = time.perf_counter()
    outer_minibatches = _env_group_minibatch_indices(
        stage_batch,
        sample_count=n,
        minibatches=max(int(minibatches), 1),
        device=target.device,
    )
    if not outer_minibatches:
        raise RuntimeError(f"{STAGE_NAME[stage_id_i]} critic has no optimizer minibatches.")
    for epoch_idx in range(max(int(epochs), 0)):
        epoch_t0 = time.perf_counter()
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
        for group_idx in _shuffled_group_order(len(outer_minibatches)):
            mb_t0 = time.perf_counter()
            idx_full = outer_minibatches[int(group_idx)]
            full_count = int(idx_full.numel())
            if full_count <= 0:
                continue
            use_update_microbatches = update_microbatch_size_i > 0
            if diagnose_timing:
                t = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                _sync()
                epoch_zero_sec += time.perf_counter() - t
                chunk_size = update_microbatch_size_i if use_update_microbatches else full_count
                for chunk_start in range(0, full_count, max(chunk_size, 1)):
                    idx = idx_full[chunk_start : chunk_start + chunk_size]
                    real_count = int(idx.numel())
                    if real_count <= 0:
                        continue
                    forward_idx = idx
                    if real_count < int(chunk_size):
                        pad = idx.new_full((int(chunk_size) - real_count,), int(idx[-1].item()))
                        forward_idx = torch.cat([idx, pad], dim=0)
                    loss_scale = float(real_count) / float(max(full_count, 1))

                    t = time.perf_counter()
                    batch = _index_dataclass(world, forward_idx)
                    _sync()
                    epoch_index_sec += time.perf_counter() - t

                    t = time.perf_counter()
                    pred = learner._stage_value_eval_from_batch(stage_id_i, batch)[:real_count]
                    _sync()
                    epoch_forward_sec += time.perf_counter() - t

                    t = time.perf_counter()
                    main_loss = F.mse_loss(pred, target.index_select(0, idx))
                    if aux_target_t is not None:
                        aux_loss = F.mse_loss(pred, aux_target_t.index_select(0, idx))
                        loss = (main_loss + aux_coef_f * aux_loss) * loss_scale
                        train_aux_loss_sum += float(aux_loss.detach().cpu().item())
                    else:
                        aux_loss = None
                        loss = main_loss * loss_scale
                    train_main_loss_sum += float(main_loss.detach().cpu().item())
                    train_loss_chunks += 1
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
                    real_count = int(idx.numel())
                    if real_count <= 0:
                        continue
                    forward_idx = idx
                    if real_count < int(chunk_size):
                        pad = idx.new_full((int(chunk_size) - real_count,), int(idx[-1].item()))
                        forward_idx = torch.cat([idx, pad], dim=0)
                    loss_scale = float(real_count) / float(max(full_count, 1))
                    pred = learner._stage_value_eval_from_batch(stage_id_i, _index_dataclass(world, forward_idx))[:real_count]
                    main_loss = F.mse_loss(pred, target.index_select(0, idx))
                    if aux_target_t is not None:
                        aux_loss = F.mse_loss(pred, aux_target_t.index_select(0, idx))
                        loss = (main_loss + aux_coef_f * aux_loss) * loss_scale
                        train_aux_loss_sum += float(aux_loss.detach().cpu().item())
                    else:
                        loss = main_loss * loss_scale
                    train_main_loss_sum += float(main_loss.detach().cpu().item())
                    train_loss_chunks += 1
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
            trace_rows.append(
                {
                    "epoch": float(epoch_idx + 1),
                    "epoch_sec": float(time.perf_counter() - epoch_t0),
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

    cache_clear_sec = 0.0
    if learner.device.type == "cuda":
        t_cache = time.perf_counter()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(learner.device)
        cache_clear_sec = time.perf_counter() - t_cache

    t_after = time.perf_counter()
    after_pred_np, after_stats = _eval_critic(
        learner,
        stage_id=stage_id_i,
        world_bank=world,
        target=target,
        batch_size=eval_batch_size,
        pad_to_batch_size=True,
    )
    if learner.device.type == "cuda":
        torch.cuda.synchronize(learner.device)
    eval_after_sec = time.perf_counter() - t_after
    after_values = torch.as_tensor(after_pred_np, dtype=torch.float32, device=learner.device).reshape(-1)
    if aux_target_t is not None:
        aux_after_stats = _critic_np_stats(after_pred_np, aux_target_t.detach().cpu().numpy())
    else:
        aux_after_stats = {"ev": float("nan"), "mse": float("nan"), "corr": float("nan")}
    denom = float(max(int(train_loss_chunks), 1))
    return {
        "critic_lr": float(lr),
        "critic_epochs": float(epochs),
        "critic_minibatches": float(minibatches),
        "critic_outer_minibatches": float(len(outer_minibatches)),
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
        "critic_aux_coef": float(aux_coef_f),
        "critic_loss_main": float(train_main_loss_sum / denom),
        "critic_loss_aux": float(train_aux_loss_sum / denom) if aux_target_t is not None else 0.0,
        "critic_aux_ev_after": float(aux_after_stats["ev"]),
        "critic_aux_mse_after": float(aux_after_stats["mse"]),
        "critic_aux_corr_after": float(aux_after_stats["corr"]),
    }, after_values, trace_rows


def _eval_stage_values_for_stage(
    learner: Any,
    *,
    stage_id: int,
    stage_batch: Any,
    device: torch.device,
) -> torch.Tensor:
    sample_count = int(stage_batch.num_samples)
    if sample_count <= 0:
        return torch.zeros((0,), dtype=torch.float32, device=device)
    chunks: list[torch.Tensor] = []
    cfg = getattr(learner, "cfg", None)
    preferred = int(
        getattr(cfg, "stage_mcgae_critic_update_microbatch_size", 0)
        or getattr(cfg, "sat_mcgae_critic_update_microbatch_size", 0)
        or 2048
    )
    # Fixed-shape compiled eval is handled by padding the final chunk.  Do not
    # search for an exact divisor of sample_count here: BW macro rows can be
    # 3201/3202/etc., and divisor search can collapse to tiny chunks.
    batch_size = max(1, int(preferred))
    with torch.no_grad():
        for start in range(0, sample_count, batch_size):
            stop = min(start + batch_size, sample_count)
            idx = torch.arange(start, stop, dtype=torch.long, device=device)
            real_count = int(idx.numel())
            if real_count <= 0:
                continue
            forward_idx = idx
            if real_count < int(batch_size):
                pad = idx.new_full((int(batch_size) - real_count,), int(idx[-1].item()))
                forward_idx = torch.cat([idx, pad], dim=0)
            pred = learner._stage_value_eval_from_batch(
                int(stage_id),
                _index_dataclass(stage_batch.world_batch, forward_idx),
            )
            chunks.append(pred[:real_count].detach().to(device=device, dtype=torch.float32).clone())
    return torch.cat(chunks, dim=0)


def _dense_stage_layout(stage_batch: Any, *, stage_id: int, sample_count: int) -> tuple[int, int] | None:
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
        + int(stage_id)
    ).reshape(-1)
    if not np.array_equal(transition_idx, expected_transition):
        return None
    return int(num_steps), int(num_envs)


def _stage_gae_native_dense(
    learner: Any,
    *,
    stage_id: int,
    stage_batch: Any,
    stage_values: torch.Tensor,
    mc_target: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if device.type != "cuda":
        return None
    cfg = getattr(learner, "cfg", None)
    if str(getattr(cfg, "structured_env_backend", "native") or "native").strip().lower() != "native":
        return None
    durations = getattr(stage_batch, "duration", None)
    if torch.is_tensor(durations) and not bool(torch.all(durations.reshape(-1).to(device=device) == 1).item()):
        return None
    sample_count = int(stage_batch.num_samples)
    layout = _dense_stage_layout(stage_batch, stage_id=int(stage_id), sample_count=sample_count)
    if layout is None:
        return None
    num_steps, num_envs = layout
    terminated = getattr(stage_batch, "terminated", None)
    truncated = getattr(stage_batch, "truncated", None)
    if not torch.is_tensor(terminated) or not torch.is_tensor(truncated):
        return None
    values = stage_values.detach().to(device=device, dtype=torch.float32).reshape(-1).contiguous()
    mc = mc_target.detach().to(device=device, dtype=torch.float32).reshape(-1).contiguous()
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


def _stage_gae_from_mc_targets(
    learner: Any,
    *,
    stage_id: int,
    stage_batch: Any,
    mc_target: torch.Tensor,
    device: torch.device,
    stage_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build same-stage GAE from finite-horizon MC returns.

    For adjacent decisions of the same stage in the same env episode:

        MC_t = collapsed_reward_t + gamma * MC_{t+1}

    so the per-stage reward between two same-stage decisions can be recovered
    without evaluating the other two stage critics.
    """

    sample_count = int(stage_batch.num_samples)
    if stage_values is None:
        stage_values = _eval_stage_values_for_stage(
            learner,
            stage_id=int(stage_id),
            stage_batch=stage_batch,
            device=device,
        )
    else:
        stage_values = stage_values.detach().to(device=device, dtype=torch.float32).reshape(-1)
    mc = mc_target.detach().to(device=device, dtype=torch.float32).reshape(-1)
    if int(mc.numel()) != sample_count:
        raise RuntimeError(f"MC target length {int(mc.numel())} != samples {sample_count}.")
    if int(stage_values.numel()) != sample_count:
        raise RuntimeError(f"value length {int(stage_values.numel())} != samples {sample_count}.")

    duration_attr_probe = getattr(stage_batch, "duration", None)
    if duration_attr_probe is not None:
        if torch.is_tensor(duration_attr_probe):
            duration_probe = duration_attr_probe.detach().to(device=device).reshape(-1)
            has_macro_rows = bool(torch.any(duration_probe != 1).item()) if int(duration_probe.numel()) > 0 else False
        else:
            duration_probe_np = np.asarray(duration_attr_probe, dtype=np.int64).reshape(-1)
            has_macro_rows = bool(np.any(duration_probe_np != 1)) if int(duration_probe_np.size) > 0 else False
        if has_macro_rows:
            # Macro modes hold one stage action over multiple primitive steps,
            # but the return target is still gathered from the primitive-step
            # MC chain at the macro-start transition.  Do not rebuild a
            # duration-compressed SMDP reward here: the actor row is sparse,
            # not a new SMDP transition with a collapsed reward.
            adv = mc - stage_values
            return mc, adv, stage_values

    native_result = _stage_gae_native_dense(
        learner,
        stage_id=int(stage_id),
        stage_batch=stage_batch,
        stage_values=stage_values,
        mc_target=mc,
        device=device,
    )
    if native_result is not None:
        ret, adv = native_result
        return ret, adv, stage_values

    transition_idx = np.asarray(stage_batch.transition_indices, dtype=np.int64).reshape(-1)
    env_idx = np.asarray(stage_batch.env_indices, dtype=np.int64).reshape(-1)
    if int(transition_idx.size) != sample_count or int(env_idx.size) != sample_count:
        raise RuntimeError("stage transition/env index length mismatch.")
    if np.any((transition_idx - int(stage_id)) % 3 != 0):
        raise RuntimeError(f"transition indices are not stage-{int(stage_id)} aligned.")

    terminated = _as_bool_np(getattr(stage_batch, "terminated"))
    truncated = _as_bool_np(getattr(stage_batch, "truncated"))
    if int(terminated.size) != sample_count or int(truncated.size) != sample_count:
        raise RuntimeError("stage terminal flag length mismatch.")
    duration_attr = getattr(stage_batch, "duration", None)
    if torch.is_tensor(duration_attr):
        durations_np = duration_attr.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
    elif duration_attr is None:
        durations_np = np.ones((sample_count,), dtype=np.int64)
    else:
        durations_np = np.asarray(duration_attr, dtype=np.int64).reshape(-1)
    if int(durations_np.size) != sample_count:
        raise RuntimeError("stage duration length mismatch.")
    durations_np = np.maximum(durations_np.astype(np.int64, copy=False), 1)

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
            ended_here = bool(terminated[pos] or truncated[pos])
            duration_i = max(int(durations_np[pos]), 1)
            gamma_d = float(gamma) ** duration_i
            gamma_lam_d = (float(gamma) * float(lam)) ** duration_i
            if have_next and not ended_here:
                collapsed_reward = mc[pos] - float(gamma_d) * next_mc
                bootstrap_value = next_value
                bootstrap_adv = next_adv
            else:
                collapsed_reward = mc[pos]
                bootstrap_value = torch.zeros((), dtype=torch.float32, device=device)
                bootstrap_adv = torch.zeros((), dtype=torch.float32, device=device)
            delta = collapsed_reward + float(gamma_d) * bootstrap_value - stage_values[pos]
            adv_pos = delta + float(gamma_lam_d) * bootstrap_adv
            adv[pos] = adv_pos
            ret[pos] = adv_pos + stage_values[pos]
            next_adv = adv_pos
            next_value = stage_values[pos]
            next_mc = mc[pos]
            have_next = not ended_here
    return ret, adv, stage_values


def _normalize_stage_advantage(adv: torch.Tensor, *, enabled: bool) -> torch.Tensor:
    adv = adv.detach().to(dtype=torch.float32)
    if not bool(enabled) or int(adv.numel()) <= 1:
        return adv
    return (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1.0e-8)


def _enable_strict_compile_global() -> None:
    torch_dynamo.config.suppress_errors = False
    torch_dynamo.config.error_on_recompile = True
    if hasattr(torch_dynamo.config, "fail_on_recompile_limit_hit"):
        torch_dynamo.config.fail_on_recompile_limit_hit = True


def _strict_chunk_size(total: int, preferred: int = 2048) -> int:
    total_i = max(int(total), 1)
    preferred_i = min(max(int(preferred), 1), total_i)
    for size in range(preferred_i, 0, -1):
        if total_i % size == 0:
            return int(size)
    return total_i


def _flat_minibatch_indices(sample_count: int, minibatches: int, device: torch.device) -> list[torch.Tensor]:
    sample_count_i = int(sample_count)
    if sample_count_i <= 0:
        return []
    mb_count = max(int(minibatches), 1)
    mb_size = max(1, int(math.ceil(sample_count_i / mb_count)))
    rows = torch.arange(sample_count_i, dtype=torch.long, device=device)
    return [rows[start : min(start + mb_size, sample_count_i)] for start in range(0, sample_count_i, mb_size)]


def _env_group_minibatch_indices(
    stage_batch: Any,
    *,
    sample_count: int,
    minibatches: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Build outer optimizer minibatches from whole env trajectories.

    The forward/backward microbatch may still be padded to a fixed shape, but
    the optimizer step should see approximately `minibatches` env groups rather
    than an exact divisor of the flattened row count.  This keeps K>1 BW macro
    rows from turning 3202 samples into thousands of tiny optimizer steps.
    """

    sample_count_i = int(sample_count)
    if sample_count_i <= 0:
        return []
    env_indices = getattr(stage_batch, "env_indices", None)
    if env_indices is None:
        return _flat_minibatch_indices(sample_count_i, minibatches, device)
    env_np = np.asarray(env_indices, dtype=np.int64).reshape(-1)
    if int(env_np.size) != sample_count_i:
        return _flat_minibatch_indices(sample_count_i, minibatches, device)
    valid_env_np = env_np[env_np >= 0]
    if int(valid_env_np.size) <= 0:
        return _flat_minibatch_indices(sample_count_i, minibatches, device)
    unique_envs = np.unique(valid_env_np)
    group_count = max(1, min(int(minibatches), int(unique_envs.size)))
    groups: list[torch.Tensor] = []
    for env_group in np.array_split(unique_envs, group_count):
        if int(env_group.size) <= 0:
            continue
        row_np = np.nonzero(np.isin(env_np, env_group))[0].astype(np.int64, copy=False)
        if int(row_np.size) <= 0:
            continue
        groups.append(torch.as_tensor(row_np, dtype=torch.long, device=device))
    return groups or _flat_minibatch_indices(sample_count_i, minibatches, device)


def _shuffled_group_order(group_count: int) -> list[int]:
    if int(group_count) <= 1:
        return [0]
    return np.random.permutation(int(group_count)).astype(np.int64).tolist()


def _actor_compile_enabled(learner: Any, stage_id: int) -> bool:
    if learner.device.type != "cuda":
        return False
    if not hasattr(torch, "compile"):
        return False
    default_enabled = bool(getattr(learner.cfg, "stage_actor_compile_enabled", True))
    stage_attr = {
        0: "accel_actor_compile_enabled",
        1: "sat_actor_compile_enabled",
        2: "bw_actor_compile_enabled",
    }.get(int(stage_id))
    if stage_attr is None:
        return default_enabled
    return bool(getattr(learner.cfg, stage_attr, default_enabled))


def _accel_logprob_entropy_mean(
    policy: Any,
    local_state: Any,
    action: torch.Tensor,
    *,
    need_entropy: bool,
    latent_action: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ctx = policy._context(local_state)
    mean = policy.mu_head(ctx)
    std = torch.clamp(policy.log_std, -5.0, 2.0).exp().reshape((1,) * max(mean.ndim - 1, 0) + (2,)).expand_as(mean)

    if latent_action is not None:
        z = latent_action.to(dtype=mean.dtype, device=mean.device)
    else:
        eps = 1.0e-4
        scale_t = torch.as_tensor(float(policy.action_scale), dtype=action.dtype, device=action.device)
        t = action.to(dtype=mean.dtype, device=mean.device) / scale_t
        action_radius = torch.linalg.vector_norm(t, dim=-1, keepdim=True)
        squashed_radius = action_radius.clamp(max=1.0 - eps)
        raw_radius = 0.5 * (torch.log1p(squashed_radius) - torch.log1p(-squashed_radius))
        direction = torch.where(action_radius > eps, t / action_radius.clamp_min(eps), t)
        z = direction * raw_radius
    logprob_z = (
        -0.5 * ((z - mean) / std).pow(2)
        - torch.log(std)
        - 0.5 * math.log(2.0 * math.pi)
    ).sum(dim=-1)
    logprob = logprob_z
    if latent_action is None:
        radius_ratio = torch.where(
            squashed_radius > eps,
            squashed_radius / raw_radius.clamp_min(eps),
            torch.ones_like(squashed_radius),
        )
        log_det = (
            int(t.shape[-1]) * torch.log(scale_t)
            + (int(t.shape[-1]) - 1) * torch.log(radius_ratio.clamp_min(eps))
            + torch.log((1.0 - squashed_radius.pow(2)).clamp_min(eps))
        ).squeeze(-1)
        logprob = logprob_z - log_det
    if bool(need_entropy):
        entropy = (0.5 + 0.5 * math.log(2.0 * math.pi) + torch.log(std)).sum(dim=-1)
    else:
        entropy = torch.zeros_like(logprob)
    return logprob, entropy, mean


def _sat_logprob_entropy(policy: Any, local_state: Any, subset_index: torch.Tensor, *, need_entropy: bool) -> tuple[torch.Tensor, torch.Tensor]:
    logits = policy._compute_logits(local_state)
    subset_mask = policy._legal_subset_mask(local_state, logits)
    safe_logits = _safe_categorical_logits(logits, subset_mask)
    legal_count = subset_mask.sum(dim=-1)
    log_probs = torch.log_softmax(safe_logits, dim=-1)
    chosen = subset_index.to(device=logits.device, dtype=torch.long).reshape(-1)
    chosen_safe = chosen.clamp(min=0, max=max(int(logits.shape[1]) - 1, 0))
    logprob_raw = log_probs.gather(1, chosen_safe.unsqueeze(1)).squeeze(1)
    multi_legal = legal_count > 1
    logprob = torch.where(multi_legal, logprob_raw, torch.zeros_like(logprob_raw))
    if bool(need_entropy):
        probs = torch.softmax(safe_logits, dim=-1)
        entropy_raw = -(probs * log_probs).sum(dim=-1)
        entropy = torch.where(multi_legal, entropy_raw, torch.zeros_like(entropy_raw))
    else:
        entropy = torch.zeros_like(logprob)
    return logprob, entropy


def _bw_logprob_entropy(policy: Any, local_state: Any, action: torch.Tensor, *, need_entropy: bool) -> tuple[torch.Tensor, torch.Tensor]:
    score, det_mean, _alpha, kappa, _tau, valid_count, _latent_count, valid = policy._params(local_state)
    del score
    eps = 1.0e-8
    mask_f = valid.to(dtype=det_mean.dtype)
    mean_valid = torch.where(valid, det_mean.clamp_min(eps), torch.zeros_like(det_mean))
    mean_sum = mean_valid.sum(dim=-1, keepdim=True).clamp_min(eps)
    normalized_mean = torch.where(valid, mean_valid / mean_sum, torch.zeros_like(mean_valid))
    normalized_mean = torch.where(valid_count.unsqueeze(-1) == 1, mask_f, normalized_mean)
    normalized_mean = torch.where(valid_count.unsqueeze(-1) <= 0, torch.zeros_like(normalized_mean), normalized_mean)
    kappa_s = kappa.clamp_min(eps)
    concentration = normalized_mean * kappa_s.unsqueeze(-1)
    masked_concentration = torch.where(valid, concentration.clamp_min(eps), torch.ones_like(concentration))

    action_eval = action.to(dtype=det_mean.dtype, device=det_mean.device).masked_fill(~valid, 0.0)
    action_eval = torch.where(valid_count.unsqueeze(-1) == 1, mask_f, action_eval)
    action_eval = torch.where(valid_count.unsqueeze(-1) <= 0, torch.zeros_like(action_eval), action_eval)
    action_valid = torch.where(valid, action_eval.clamp_min(eps), torch.zeros_like(action_eval))
    action_sum = action_valid.sum(dim=-1, keepdim=True).clamp_min(eps)
    probs = torch.where(valid, action_valid / action_sum, torch.zeros_like(action_valid))
    action_safe = torch.where(valid, probs.clamp_min(eps), torch.ones_like(probs))
    alpha0 = (concentration * mask_f).sum(dim=-1).clamp_min(eps)
    logprob_raw = (
        torch.lgamma(alpha0)
        - torch.lgamma(masked_concentration).sum(dim=-1)
        + ((masked_concentration - 1.0) * torch.log(action_safe)).sum(dim=-1)
    )
    logprob_raw = torch.where(valid_count >= 2, logprob_raw, torch.zeros_like(logprob_raw))
    denom = policy._objective_denominator(valid_count, logprob_raw.dtype)
    logprob = logprob_raw / denom
    if bool(need_entropy):
        log_beta = torch.lgamma(masked_concentration).sum(dim=-1) - torch.lgamma(alpha0)
        entropy_raw = (
            log_beta
            + (alpha0 - valid_count.to(dtype=alpha0.dtype)) * torch.digamma(alpha0)
            - ((masked_concentration - 1.0) * torch.digamma(masked_concentration)).sum(dim=-1)
        )
        entropy_raw = torch.where(valid_count >= 2, entropy_raw, torch.zeros_like(entropy_raw))
        entropy = entropy_raw / denom
    else:
        entropy = torch.zeros_like(logprob)
    return logprob, entropy


def _make_stage_actor_loss_chunk_fn(
    learner: Any,
    *,
    stage_id: int,
    num_agents: int,
    need_entropy: bool,
    danger_enabled: bool,
) -> Any:
    stage_id_i = int(stage_id)
    num_agents_i = int(num_agents)
    entropy_coef = float(learner.entropy_coef_by_stage[stage_id_i])
    danger_coef = float(getattr(learner, "danger_imitation_coef", 0.0))
    clip_ratio = float(learner.clip_ratio)
    actor = learner.actor

    def _loss_chunk(
        local_i: Any,
        action_i: torch.Tensor,
        latent_i: torch.Tensor,
        old_i: torch.Tensor,
        adv_i: torch.Tensor,
        danger_targets_i: torch.Tensor,
        danger_masks_i: torch.Tensor,
        valid_i: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = int(action_i.shape[0])
        if stage_id_i == 0:
            flat_action = action_i.reshape(rows * num_agents_i, -1)
            logprob_agent, entropy_agent, mean = _accel_logprob_entropy_mean(
                actor.accel_policy,
                local_i,
                flat_action,
                need_entropy=bool(need_entropy),
                latent_action=latent_i.reshape(rows * num_agents_i, -1),
            )
            logprob = logprob_agent.reshape(rows, num_agents_i).sum(dim=1)
            entropy = entropy_agent.reshape(rows, num_agents_i).sum(dim=1)
            danger_loss = torch.zeros((), dtype=logprob.dtype, device=logprob.device)
            danger_active = torch.zeros((), dtype=logprob.dtype, device=logprob.device)
            if bool(danger_enabled):
                target = danger_targets_i.to(dtype=mean.dtype, device=mean.device).reshape(rows, num_agents_i, 2)
                mask = danger_masks_i.to(dtype=mean.dtype, device=mean.device).reshape(rows, num_agents_i, 2)
                pred = squash_action(mean, actor.accel_policy.action_scale).reshape(rows, num_agents_i, 2)
                active = (mask.sum(dim=-1) > 0).to(dtype=mean.dtype)
                denom = mask.sum(dim=-1).clamp_min(1.0)
                per_agent = (((pred - target) * mask).pow(2).sum(dim=-1)) / denom
                active_count = active.sum()
                danger_loss = (per_agent * active).sum() / active_count.clamp_min(1.0)
                danger_active = active.mean()
        elif stage_id_i == 1:
            logprob_agent, entropy_agent = _sat_logprob_entropy(
                actor.sat_subset_policy,
                local_i,
                action_i.reshape(-1),
                need_entropy=bool(need_entropy),
            )
            logprob = logprob_agent.reshape(rows, num_agents_i).sum(dim=1)
            entropy = entropy_agent.reshape(rows, num_agents_i).sum(dim=1)
            danger_loss = torch.zeros((), dtype=logprob.dtype, device=logprob.device)
            danger_active = torch.zeros((), dtype=logprob.dtype, device=logprob.device)
        else:
            flat_action = action_i.reshape(rows * num_agents_i, -1)
            logprob_agent, entropy_agent = _bw_logprob_entropy(
                actor.bw_policy,
                local_i,
                flat_action,
                need_entropy=bool(need_entropy),
            )
            logprob = logprob_agent.reshape(rows, num_agents_i).sum(dim=1)
            entropy = entropy_agent.reshape(rows, num_agents_i).sum(dim=1)
            danger_loss = torch.zeros((), dtype=logprob.dtype, device=logprob.device)
            danger_active = torch.zeros((), dtype=logprob.dtype, device=logprob.device)

        old_t = old_i.to(dtype=logprob.dtype)
        adv_t = adv_i.to(dtype=logprob.dtype)
        log_ratio = torch.clamp(logprob - old_t.detach(), min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)
        clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        policy_loss = -_masked_row_mean(torch.minimum(ratio * adv_t.detach(), clipped * adv_t.detach()), valid_i)
        entropy_mean = _masked_row_mean(entropy, valid_i) if bool(need_entropy) else torch.zeros((), dtype=logprob.dtype, device=logprob.device)
        loss = policy_loss - entropy_coef * entropy_mean + danger_coef * danger_loss
        approx_kl = _masked_row_mean(old_t.detach() - logprob.detach(), valid_i)
        clip_frac = _masked_row_mean((torch.abs(ratio.detach() - 1.0) > clip_ratio).to(dtype=logprob.dtype), valid_i)
        return loss, policy_loss.detach(), entropy_mean.detach(), approx_kl.detach(), clip_frac.detach(), danger_loss.detach(), danger_active.detach()

    return _loss_chunk


class _StageActorLossChunkModule(torch.nn.Module):
    def __init__(
        self,
        actor: torch.nn.Module,
        *,
        stage_id: int,
        num_agents: int,
        need_entropy: bool,
        danger_enabled: bool,
        entropy_coef: float,
        danger_coef: float,
        clip_ratio: float,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.stage_id = int(stage_id)
        self.num_agents = int(num_agents)
        self.need_entropy = bool(need_entropy)
        self.danger_enabled = bool(danger_enabled)
        self.entropy_coef = float(entropy_coef)
        self.danger_coef = float(danger_coef)
        self.clip_ratio = float(clip_ratio)

    def forward(
        self,
        local_i: Any,
        action_i: torch.Tensor,
        latent_i: torch.Tensor,
        old_i: torch.Tensor,
        adv_i: torch.Tensor,
        danger_targets_i: torch.Tensor,
        danger_masks_i: torch.Tensor,
        valid_i: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = int(action_i.shape[0])
        if self.stage_id == 0:
            flat_action = action_i.reshape(rows * self.num_agents, -1)
            logprob_agent, entropy_agent, mean = _accel_logprob_entropy_mean(
                self.actor.accel_policy,
                local_i,
                flat_action,
                need_entropy=self.need_entropy,
                latent_action=latent_i.reshape(rows * self.num_agents, -1),
            )
            logprob = logprob_agent.reshape(rows, self.num_agents).sum(dim=1)
            entropy = entropy_agent.reshape(rows, self.num_agents).sum(dim=1)
            danger_loss = torch.zeros((), dtype=logprob.dtype, device=logprob.device)
            danger_active = torch.zeros((), dtype=logprob.dtype, device=logprob.device)
            if self.danger_enabled:
                target = danger_targets_i.to(dtype=mean.dtype, device=mean.device).reshape(rows, self.num_agents, 2)
                mask = danger_masks_i.to(dtype=mean.dtype, device=mean.device).reshape(rows, self.num_agents, 2)
                pred = squash_action(mean, self.actor.accel_policy.action_scale).reshape(rows, self.num_agents, 2)
                active = (mask.sum(dim=-1) > 0).to(dtype=mean.dtype)
                denom = mask.sum(dim=-1).clamp_min(1.0)
                per_agent = (((pred - target) * mask).pow(2).sum(dim=-1)) / denom
                active_count = active.sum()
                danger_loss = (per_agent * active).sum() / active_count.clamp_min(1.0)
                danger_active = active.mean()
        elif self.stage_id == 1:
            logprob_agent, entropy_agent = _sat_logprob_entropy(
                self.actor.sat_subset_policy,
                local_i,
                action_i.reshape(-1),
                need_entropy=self.need_entropy,
            )
            logprob = logprob_agent.reshape(rows, self.num_agents).sum(dim=1)
            entropy = entropy_agent.reshape(rows, self.num_agents).sum(dim=1)
            danger_loss = torch.zeros((), dtype=logprob.dtype, device=logprob.device)
            danger_active = torch.zeros((), dtype=logprob.dtype, device=logprob.device)
        else:
            flat_action = action_i.reshape(rows * self.num_agents, -1)
            logprob_agent, entropy_agent = _bw_logprob_entropy(
                self.actor.bw_policy,
                local_i,
                flat_action,
                need_entropy=self.need_entropy,
            )
            logprob = logprob_agent.reshape(rows, self.num_agents).sum(dim=1)
            entropy = entropy_agent.reshape(rows, self.num_agents).sum(dim=1)
            danger_loss = torch.zeros((), dtype=logprob.dtype, device=logprob.device)
            danger_active = torch.zeros((), dtype=logprob.dtype, device=logprob.device)

        old_t = old_i.to(dtype=logprob.dtype)
        adv_t = adv_i.to(dtype=logprob.dtype)
        log_ratio = torch.clamp(logprob - old_t.detach(), min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)
        clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        policy_loss = -_masked_row_mean(torch.minimum(ratio * adv_t.detach(), clipped * adv_t.detach()), valid_i)
        entropy_mean = _masked_row_mean(entropy, valid_i) if self.need_entropy else torch.zeros((), dtype=logprob.dtype, device=logprob.device)
        loss = policy_loss - self.entropy_coef * entropy_mean + self.danger_coef * danger_loss
        approx_kl = _masked_row_mean(old_t.detach() - logprob.detach(), valid_i)
        clip_frac = _masked_row_mean((torch.abs(ratio.detach() - 1.0) > self.clip_ratio).to(dtype=logprob.dtype), valid_i)
        return (
            loss,
            policy_loss,
            entropy_mean,
            approx_kl,
            clip_frac,
            danger_loss,
            danger_active,
        )


def _masked_row_mean(values: torch.Tensor, valid_i: torch.Tensor) -> torch.Tensor:
    weights = valid_i.to(dtype=values.dtype, device=values.device).reshape(-1)
    denom = weights.sum().clamp_min(1.0)
    return (values.reshape(-1) * weights).sum() / denom


class _AccelActorLossChunkModule(torch.nn.Module):
    def __init__(
        self,
        policy: torch.nn.Module,
        *,
        num_agents: int,
        need_entropy: bool,
        danger_enabled: bool,
        entropy_coef: float,
        danger_coef: float,
        clip_ratio: float,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.num_agents = int(num_agents)
        self.need_entropy = bool(need_entropy)
        self.danger_enabled = bool(danger_enabled)
        self.entropy_coef = float(entropy_coef)
        self.danger_coef = float(danger_coef)
        self.clip_ratio = float(clip_ratio)

    def forward(
        self,
        local_i: Any,
        action_i: torch.Tensor,
        latent_i: torch.Tensor,
        old_i: torch.Tensor,
        adv_i: torch.Tensor,
        danger_targets_i: torch.Tensor,
        danger_masks_i: torch.Tensor,
        valid_i: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = int(action_i.shape[0])
        flat_action = action_i.reshape(rows * self.num_agents, -1)
        logprob_agent, entropy_agent, mean = _accel_logprob_entropy_mean(
            self.policy,
            local_i,
            flat_action,
            need_entropy=self.need_entropy,
            latent_action=latent_i.reshape(rows * self.num_agents, -1),
        )
        logprob = logprob_agent.reshape(rows, self.num_agents).sum(dim=1)
        entropy = entropy_agent.reshape(rows, self.num_agents).sum(dim=1)
        danger_loss = torch.zeros((), dtype=logprob.dtype, device=logprob.device)
        danger_active = torch.zeros((), dtype=logprob.dtype, device=logprob.device)
        row_valid = valid_i.to(dtype=mean.dtype, device=mean.device).reshape(rows, 1)
        if self.danger_enabled:
            target = danger_targets_i.to(dtype=mean.dtype, device=mean.device).reshape(rows, self.num_agents, 2)
            mask = danger_masks_i.to(dtype=mean.dtype, device=mean.device).reshape(rows, self.num_agents, 2)
            pred = squash_action(mean, self.policy.action_scale).reshape(rows, self.num_agents, 2)
            active = (mask.sum(dim=-1) > 0).to(dtype=mean.dtype)
            denom = mask.sum(dim=-1).clamp_min(1.0)
            per_agent = (((pred - target) * mask).pow(2).sum(dim=-1)) / denom
            active_weight = active * row_valid
            active_count = active_weight.sum()
            danger_loss = (per_agent * active_weight).sum() / active_count.clamp_min(1.0)
            danger_active = active_weight.sum() / (row_valid.sum().clamp_min(1.0) * float(self.num_agents))
        old_t = old_i.to(dtype=logprob.dtype)
        adv_t = adv_i.to(dtype=logprob.dtype)
        log_ratio = torch.clamp(logprob - old_t.detach(), min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)
        clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        policy_loss = -_masked_row_mean(torch.minimum(ratio * adv_t.detach(), clipped * adv_t.detach()), valid_i)
        entropy_mean = _masked_row_mean(entropy, valid_i) if self.need_entropy else torch.zeros((), dtype=logprob.dtype, device=logprob.device)
        loss = policy_loss - self.entropy_coef * entropy_mean + self.danger_coef * danger_loss
        approx_kl = _masked_row_mean(old_t.detach() - logprob.detach(), valid_i)
        clip_frac = _masked_row_mean((torch.abs(ratio.detach() - 1.0) > self.clip_ratio).to(dtype=logprob.dtype), valid_i)
        return (
            loss,
            policy_loss.detach(),
            entropy_mean.detach(),
            approx_kl.detach(),
            clip_frac.detach(),
            danger_loss.detach(),
            danger_active.detach(),
        )


class _AccelActorLossNoDangerModule(torch.nn.Module):
    def __init__(
        self,
        policy: torch.nn.Module,
        *,
        num_agents: int,
        need_entropy: bool,
        entropy_coef: float,
        clip_ratio: float,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.num_agents = int(num_agents)
        self.need_entropy = bool(need_entropy)
        self.entropy_coef = float(entropy_coef)
        self.clip_ratio = float(clip_ratio)

    def forward(
        self,
        local_i: Any,
        action_i: torch.Tensor,
        latent_i: torch.Tensor,
        old_i: torch.Tensor,
        adv_i: torch.Tensor,
        danger_targets_i: torch.Tensor,
        danger_masks_i: torch.Tensor,
        valid_i: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del danger_targets_i, danger_masks_i
        rows = int(action_i.shape[0])
        flat_action = action_i.reshape(rows * self.num_agents, -1)
        logprob_agent, entropy_agent, _mean = _accel_logprob_entropy_mean(
            self.policy,
            local_i,
            flat_action,
            need_entropy=self.need_entropy,
            latent_action=latent_i.reshape(rows * self.num_agents, -1),
        )
        logprob = logprob_agent.reshape(rows, self.num_agents).sum(dim=1)
        entropy = entropy_agent.reshape(rows, self.num_agents).sum(dim=1)
        old_t = old_i.to(dtype=logprob.dtype)
        adv_t = adv_i.to(dtype=logprob.dtype)
        log_ratio = torch.clamp(logprob - old_t.detach(), min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)
        clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        policy_loss = -_masked_row_mean(torch.minimum(ratio * adv_t.detach(), clipped * adv_t.detach()), valid_i)
        entropy_mean = _masked_row_mean(entropy, valid_i) if self.need_entropy else torch.zeros((), dtype=logprob.dtype, device=logprob.device)
        loss = policy_loss - self.entropy_coef * entropy_mean
        approx_kl = _masked_row_mean(old_t.detach() - logprob.detach(), valid_i)
        clip_frac = _masked_row_mean((torch.abs(ratio.detach() - 1.0) > self.clip_ratio).to(dtype=logprob.dtype), valid_i)
        zero = torch.zeros((), dtype=logprob.dtype, device=logprob.device)
        return loss, policy_loss, entropy_mean, approx_kl, clip_frac, zero, zero


class _AccelActorLossDangerModule(torch.nn.Module):
    def __init__(
        self,
        policy: torch.nn.Module,
        *,
        num_agents: int,
        need_entropy: bool,
        entropy_coef: float,
        danger_coef: float,
        clip_ratio: float,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.num_agents = int(num_agents)
        self.need_entropy = bool(need_entropy)
        self.entropy_coef = float(entropy_coef)
        self.danger_coef = float(danger_coef)
        self.clip_ratio = float(clip_ratio)

    def forward(
        self,
        local_i: Any,
        action_i: torch.Tensor,
        latent_i: torch.Tensor,
        old_i: torch.Tensor,
        adv_i: torch.Tensor,
        danger_targets_i: torch.Tensor,
        danger_masks_i: torch.Tensor,
        valid_i: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = int(action_i.shape[0])
        flat_action = action_i.reshape(rows * self.num_agents, -1)
        logprob_agent, entropy_agent, mean = _accel_logprob_entropy_mean(
            self.policy,
            local_i,
            flat_action,
            need_entropy=self.need_entropy,
            latent_action=latent_i.reshape(rows * self.num_agents, -1),
        )
        logprob = logprob_agent.reshape(rows, self.num_agents).sum(dim=1)
        entropy = entropy_agent.reshape(rows, self.num_agents).sum(dim=1)
        row_valid = valid_i.to(dtype=mean.dtype, device=mean.device).reshape(rows, 1)
        target = danger_targets_i.to(dtype=mean.dtype, device=mean.device).reshape(rows, self.num_agents, 2)
        mask = danger_masks_i.to(dtype=mean.dtype, device=mean.device).reshape(rows, self.num_agents, 2)
        pred = squash_action(mean, self.policy.action_scale).reshape(rows, self.num_agents, 2)
        active = (mask.sum(dim=-1) > 0).to(dtype=mean.dtype)
        denom = mask.sum(dim=-1).clamp_min(1.0)
        per_agent = (((pred - target) * mask).pow(2).sum(dim=-1)) / denom
        active_weight = active * row_valid
        active_count = active_weight.sum()
        danger_loss = (per_agent * active_weight).sum() / active_count.clamp_min(1.0)
        danger_active = active_weight.sum() / (row_valid.sum().clamp_min(1.0) * float(self.num_agents))
        old_t = old_i.to(dtype=logprob.dtype)
        adv_t = adv_i.to(dtype=logprob.dtype)
        log_ratio = torch.clamp(logprob - old_t.detach(), min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)
        clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        policy_loss = -_masked_row_mean(torch.minimum(ratio * adv_t.detach(), clipped * adv_t.detach()), valid_i)
        entropy_mean = _masked_row_mean(entropy, valid_i) if self.need_entropy else torch.zeros((), dtype=logprob.dtype, device=logprob.device)
        loss = policy_loss - self.entropy_coef * entropy_mean + self.danger_coef * danger_loss
        approx_kl = _masked_row_mean(old_t.detach() - logprob.detach(), valid_i)
        clip_frac = _masked_row_mean((torch.abs(ratio.detach() - 1.0) > self.clip_ratio).to(dtype=logprob.dtype), valid_i)
        return loss, policy_loss, entropy_mean, approx_kl, clip_frac, danger_loss, danger_active


class _SatActorLossChunkModule(torch.nn.Module):
    def __init__(
        self,
        policy: torch.nn.Module,
        *,
        num_agents: int,
        need_entropy: bool,
        entropy_coef: float,
        clip_ratio: float,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.num_agents = int(num_agents)
        self.need_entropy = bool(need_entropy)
        self.entropy_coef = float(entropy_coef)
        self.clip_ratio = float(clip_ratio)

    def forward(
        self,
        local_i: Any,
        action_i: torch.Tensor,
        latent_i: torch.Tensor,
        old_i: torch.Tensor,
        adv_i: torch.Tensor,
        danger_targets_i: torch.Tensor,
        danger_masks_i: torch.Tensor,
        valid_i: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del latent_i, danger_targets_i, danger_masks_i
        rows = int(action_i.shape[0])
        logprob_agent, entropy_agent = _sat_logprob_entropy(
            self.policy,
            local_i,
            action_i.reshape(-1),
            need_entropy=self.need_entropy,
        )
        logprob = logprob_agent.reshape(rows, self.num_agents).sum(dim=1)
        entropy = entropy_agent.reshape(rows, self.num_agents).sum(dim=1)
        zero = torch.zeros((), dtype=logprob.dtype, device=logprob.device)
        old_t = old_i.to(dtype=logprob.dtype)
        adv_t = adv_i.to(dtype=logprob.dtype)
        log_ratio = torch.clamp(logprob - old_t.detach(), min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)
        clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        policy_loss = -_masked_row_mean(torch.minimum(ratio * adv_t.detach(), clipped * adv_t.detach()), valid_i)
        entropy_mean = _masked_row_mean(entropy, valid_i) if self.need_entropy else torch.zeros((), dtype=logprob.dtype, device=logprob.device)
        loss = policy_loss - self.entropy_coef * entropy_mean
        approx_kl = _masked_row_mean(old_t.detach() - logprob.detach(), valid_i)
        clip_frac = _masked_row_mean((torch.abs(ratio.detach() - 1.0) > self.clip_ratio).to(dtype=logprob.dtype), valid_i)
        return (
            loss,
            policy_loss,
            entropy_mean,
            approx_kl,
            clip_frac,
            zero,
            zero,
        )


class _BwActorLossChunkModule(torch.nn.Module):
    def __init__(
        self,
        policy: torch.nn.Module,
        *,
        num_agents: int,
        need_entropy: bool,
        entropy_coef: float,
        clip_ratio: float,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.num_agents = int(num_agents)
        self.need_entropy = bool(need_entropy)
        self.entropy_coef = float(entropy_coef)
        self.clip_ratio = float(clip_ratio)

    def forward(
        self,
        local_i: Any,
        action_i: torch.Tensor,
        latent_i: torch.Tensor,
        old_i: torch.Tensor,
        adv_i: torch.Tensor,
        danger_targets_i: torch.Tensor,
        danger_masks_i: torch.Tensor,
        valid_i: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del latent_i, danger_targets_i, danger_masks_i
        rows = int(action_i.shape[0])
        flat_action = action_i.reshape(rows * self.num_agents, -1)
        logprob_agent, entropy_agent = _bw_logprob_entropy(
            self.policy,
            local_i,
            flat_action,
            need_entropy=self.need_entropy,
        )
        logprob = logprob_agent.reshape(rows, self.num_agents).sum(dim=1)
        entropy = entropy_agent.reshape(rows, self.num_agents).sum(dim=1)
        zero = torch.zeros((), dtype=logprob.dtype, device=logprob.device)
        old_t = old_i.to(dtype=logprob.dtype)
        adv_t = adv_i.to(dtype=logprob.dtype)
        log_ratio = torch.clamp(logprob - old_t.detach(), min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)
        clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        policy_loss = -_masked_row_mean(torch.minimum(ratio * adv_t.detach(), clipped * adv_t.detach()), valid_i)
        entropy_mean = _masked_row_mean(entropy, valid_i) if self.need_entropy else torch.zeros((), dtype=logprob.dtype, device=logprob.device)
        loss = policy_loss - self.entropy_coef * entropy_mean
        approx_kl = _masked_row_mean(old_t.detach() - logprob.detach(), valid_i)
        clip_frac = _masked_row_mean((torch.abs(ratio.detach() - 1.0) > self.clip_ratio).to(dtype=logprob.dtype), valid_i)
        return (
            loss,
            policy_loss,
            entropy_mean,
            approx_kl,
            clip_frac,
            zero,
            zero,
        )


def _ppo_actor_loss_outputs(
    *,
    logprob: torch.Tensor,
    entropy: torch.Tensor,
    old_i: torch.Tensor,
    adv_i: torch.Tensor,
    valid_i: torch.Tensor,
    danger_loss: torch.Tensor,
    danger_active: torch.Tensor,
    need_entropy: bool,
    entropy_coef: float,
    danger_coef: float,
    clip_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    old_t = old_i.to(dtype=logprob.dtype)
    adv_t = adv_i.to(dtype=logprob.dtype)
    log_ratio = torch.clamp(logprob - old_t.detach(), min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1.0 - float(clip_ratio), 1.0 + float(clip_ratio))
    policy_loss = -_masked_row_mean(torch.minimum(ratio * adv_t.detach(), clipped * adv_t.detach()), valid_i)
    entropy_mean = _masked_row_mean(entropy, valid_i) if bool(need_entropy) else torch.zeros((), dtype=logprob.dtype, device=logprob.device)
    loss = policy_loss - float(entropy_coef) * entropy_mean + float(danger_coef) * danger_loss
    approx_kl = _masked_row_mean(old_t.detach() - logprob.detach(), valid_i)
    clip_frac = _masked_row_mean((torch.abs(ratio.detach() - 1.0) > float(clip_ratio)).to(dtype=logprob.dtype), valid_i)
    return (
        loss,
        policy_loss.detach(),
        entropy_mean.detach(),
        approx_kl.detach(),
        clip_frac.detach(),
        danger_loss.detach(),
        danger_active.detach(),
    )


def _get_compiled_stage_actor_loss_chunk_fn(
    learner: Any,
    *,
    stage_id: int,
    num_agents: int,
    need_entropy: bool,
    danger_enabled: bool,
) -> Any:
    cache = getattr(learner, "_compiled_stage_actor_loss_chunk_fns", None)
    if cache is None:
        cache = {}
        setattr(learner, "_compiled_stage_actor_loss_chunk_fns", cache)
    key = (int(stage_id), int(num_agents), bool(need_entropy), bool(danger_enabled))
    if key not in cache:
        if int(stage_id) == 0:
            if bool(danger_enabled):
                module = _AccelActorLossDangerModule(
                    learner.actor.accel_policy,
                    num_agents=int(num_agents),
                    need_entropy=bool(need_entropy),
                    entropy_coef=float(learner.entropy_coef_by_stage[int(stage_id)]),
                    danger_coef=float(getattr(learner, "danger_imitation_coef", 0.0)),
                    clip_ratio=float(learner.clip_ratio),
                )
            else:
                module = _AccelActorLossNoDangerModule(
                    learner.actor.accel_policy,
                    num_agents=int(num_agents),
                    need_entropy=bool(need_entropy),
                    entropy_coef=float(learner.entropy_coef_by_stage[int(stage_id)]),
                    clip_ratio=float(learner.clip_ratio),
                )
        elif int(stage_id) == 1:
            module = _SatActorLossChunkModule(
                learner.actor.sat_subset_policy,
                num_agents=int(num_agents),
                need_entropy=bool(need_entropy),
                entropy_coef=float(learner.entropy_coef_by_stage[int(stage_id)]),
                clip_ratio=float(learner.clip_ratio),
            )
        else:
            module = _BwActorLossChunkModule(
                learner.actor.bw_policy,
                num_agents=int(num_agents),
                need_entropy=bool(need_entropy),
                entropy_coef=float(learner.entropy_coef_by_stage[int(stage_id)]),
                clip_ratio=float(learner.clip_ratio),
            )
        cache[key] = torch.compile(
            module,
            fullgraph=True,
            options={
                "triton.cudagraphs": False,
                "triton.cudagraph_trees": False,
            },
        )
    return cache[key]


def _stage_actor_update_full_stage(
    learner: Any,
    *,
    stage_id: int,
    stage_batch: Any,
    stage_advantages: torch.Tensor,
    stage_raw_advantages: torch.Tensor | None = None,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    minibatches: int,
    parity_dump_dir: Path | None = None,
    parity_dump_tag: str | None = None,
    parity_topk: int = 0,
    kl_stop_threshold: float | None = None,
    clip_stop_threshold: float | None = None,
) -> dict[str, float]:
    stage_id_i = int(stage_id)
    stage_name = STAGE_NAME[stage_id_i]
    params = _stage_optimizer_params(learner.actor, stage_id_i)
    if not params:
        raise RuntimeError(f"{stage_name} actor has no trainable parameters.")
    device = learner.device
    num_agents = int(stage_batch.num_agents)
    sample_count = int(stage_batch.num_samples)
    all_actions = stage_batch.actions.to(device=device)
    all_latent_actions = getattr(stage_batch, "latent_actions", None)
    if stage_id_i == 0:
        if all_latent_actions is None:
            raise RuntimeError(
                "accel MC-GAE/PPO actor update requires latent_actions so the PPO ratio is evaluated in "
                "pre-squash Normal z space. Recollect rollout with accel latent history enabled."
            )
        all_latent_actions = all_latent_actions.to(device=device)
    else:
        all_latent_actions = all_actions

    def _slice_local_samples(start: int, end: int) -> Any:
        return _slice_dataclass(stage_batch.local_batch, int(start) * int(num_agents), int(end) * int(num_agents))

    def _slice_sample_actions(start: int, end: int) -> torch.Tensor:
        return all_actions[int(start) : int(end)]

    def _slice_sample_latent_actions(start: int, end: int) -> torch.Tensor:
        return all_latent_actions[int(start) : int(end)]

    def _index_local_samples(sample_idx: torch.Tensor) -> Any:
        sample_idx = sample_idx.to(device=device, dtype=torch.long).reshape(-1)
        agent_offsets = torch.arange(num_agents, dtype=torch.long, device=device)
        flat_idx = (sample_idx.reshape(-1, 1) * int(num_agents) + agent_offsets.reshape(1, -1)).reshape(-1)
        return _index_dataclass(stage_batch.local_batch, flat_idx)

    def _index_sample_actions(sample_idx: torch.Tensor) -> torch.Tensor:
        return all_actions.index_select(0, sample_idx.to(device=device, dtype=torch.long).reshape(-1))

    def _index_sample_latent_actions(sample_idx: torch.Tensor) -> torch.Tensor:
        return all_latent_actions.index_select(0, sample_idx.to(device=device, dtype=torch.long).reshape(-1))

    def _finalize_bw_shape_stats(prefix: str, chunks: dict[str, list[torch.Tensor]]) -> dict[str, float]:
        if not chunks.get("tau"):
            return {}
        out: dict[str, float] = {}

        def _cat(name: str) -> torch.Tensor:
            values = chunks.get(name, [])
            if not values:
                return torch.empty((0,), dtype=torch.float32, device=device)
            return torch.cat([value.detach().to(device=device, dtype=torch.float32).reshape(-1) for value in values], dim=0)

        for name in (
            "tau",
            "kappa",
            "valid_count",
            "latent_count",
            "det_mean_top1",
            "det_mean_top2_gap",
            "det_mean_entropy_norm",
            "det_mean_uniform_gap",
            "action_det_l1",
            "score_top2_gap",
            "alpha_valid_min",
        ):
            out.update(_prefixed_stats(f"{prefix}_{name}", _cat(name), extended=True))

        alpha_values = _cat("alpha_valid")
        if int(alpha_values.numel()) > 0:
            out[f"{prefix}_alpha_lt1_frac"] = float(
                (alpha_values < 1.0).to(dtype=torch.float32).mean().detach().cpu().item()
            )
        else:
            out[f"{prefix}_alpha_lt1_frac"] = 0.0

        bw_policy = getattr(getattr(learner, "actor", None), "bw_policy", None)
        tau_values = _cat("tau")
        tau_min = float(getattr(bw_policy, "tau_min", float("nan")))
        tau_max = float(getattr(bw_policy, "tau_max", float("nan")))
        if int(tau_values.numel()) > 0 and math.isfinite(tau_min) and math.isfinite(tau_max):
            tau_span = max(float(tau_max) - float(tau_min), 1.0e-12)
            out[f"{prefix}_tau_low_frac"] = float(
                (tau_values <= float(tau_min) + 0.1 * tau_span)
                .to(dtype=torch.float32)
                .mean()
                .detach()
                .cpu()
                .item()
            )
            out[f"{prefix}_tau_high_frac"] = float(
                (tau_values >= float(tau_max) - 0.1 * tau_span)
                .to(dtype=torch.float32)
                .mean()
                .detach()
                .cpu()
                .item()
            )
        kappa_values = _cat("kappa")
        kappa_min = float(getattr(bw_policy, "kappa_min", float("nan")))
        kappa_max = float(getattr(bw_policy, "kappa_max", float("nan")))
        if int(kappa_values.numel()) > 0 and math.isfinite(kappa_min) and math.isfinite(kappa_max):
            kappa_span = max(float(kappa_max) - float(kappa_min), 1.0e-12)
            out[f"{prefix}_kappa_low_frac"] = float(
                (kappa_values <= float(kappa_min) + 0.1 * kappa_span)
                .to(dtype=torch.float32)
                .mean()
                .detach()
                .cpu()
                .item()
            )
            out[f"{prefix}_kappa_high_frac"] = float(
                (kappa_values >= float(kappa_max) - 0.1 * kappa_span)
                .to(dtype=torch.float32)
                .mean()
                .detach()
                .cpu()
                .item()
            )
        return out

    def _append_bw_shape_chunk(
        chunks: dict[str, list[torch.Tensor]],
        *,
        out_i: Any,
        local_i: Any,
        action_i: torch.Tensor,
    ) -> None:
        tau = getattr(out_i, "tau", None)
        kappa = getattr(out_i, "kappa", None)
        det_mean = getattr(out_i, "det_mean", None)
        if tau is None or kappa is None or det_mean is None:
            return
        valid = (
            local_i.gu_mask.to(device=device, dtype=torch.bool)
            & local_i.bw_valid_mask.to(device=device, dtype=torch.bool)
        )
        det = det_mean.detach().to(device=device, dtype=torch.float32).reshape(valid.shape)
        action_flat = action_i.detach().to(device=device, dtype=torch.float32).reshape(valid.shape)
        valid_count = valid.sum(dim=-1).to(dtype=torch.float32)
        latent_count = torch.clamp(valid_count - 1.0, min=0.0)
        active_rows = valid_count >= 2.0
        det_valid = torch.where(valid, det.clamp_min(1.0e-12), torch.zeros_like(det))
        entropy = -(det_valid * torch.log(det_valid.clamp_min(1.0e-12))).sum(dim=-1)
        log_count = torch.log(valid_count.clamp_min(1.0))
        entropy_norm = torch.where(active_rows, entropy / log_count.clamp_min(1.0e-12), torch.zeros_like(entropy))
        uniform_gap = torch.where(active_rows, log_count - entropy, torch.zeros_like(entropy))
        det_masked = det.masked_fill(~valid, -float("inf"))
        top_k = min(2, int(det_masked.shape[-1]))
        top_values = torch.topk(det_masked, k=max(top_k, 1), dim=-1).values
        top1 = torch.where(active_rows, top_values[:, 0], torch.zeros_like(valid_count))
        if top_k >= 2:
            top2_gap = torch.where(active_rows, top_values[:, 0] - top_values[:, 1], torch.zeros_like(valid_count))
        else:
            top2_gap = torch.zeros_like(valid_count)
        action_det_l1 = torch.where(
            active_rows,
            (torch.where(valid, action_flat, torch.zeros_like(action_flat)) - det_valid).abs().sum(dim=-1),
            torch.zeros_like(valid_count),
        )
        score = getattr(out_i, "score", None)
        if score is not None:
            score_t = score.detach().to(device=device, dtype=torch.float32).reshape(valid.shape).masked_fill(~valid, -float("inf"))
            score_top = torch.topk(score_t, k=max(top_k, 1), dim=-1).values
            if top_k >= 2:
                score_top = torch.where(torch.isfinite(score_top), score_top, torch.zeros_like(score_top))
                score_gap = torch.where(active_rows, score_top[:, 0] - score_top[:, 1], torch.zeros_like(valid_count))
            else:
                score_gap = torch.zeros_like(valid_count)
        else:
            score_gap = torch.zeros_like(valid_count)
        alpha = getattr(out_i, "alpha", None)
        if alpha is not None and int(alpha.numel()) > 0:
            alpha_t = alpha.detach().to(device=device, dtype=torch.float32).reshape(valid.shape)
            alpha_valid = alpha_t[valid]
            alpha_min = torch.where(
                active_rows,
                alpha_t.masked_fill(~valid, float("inf")).amin(dim=-1),
                torch.zeros_like(valid_count),
            )
        else:
            alpha_valid = torch.empty((0,), dtype=torch.float32, device=device)
            alpha_min = torch.zeros_like(valid_count)

        chunks.setdefault("tau", []).append(tau.detach().to(dtype=torch.float32).reshape(-1))
        chunks.setdefault("kappa", []).append(kappa.detach().to(dtype=torch.float32).reshape(-1))
        chunks.setdefault("valid_count", []).append(valid_count.detach().reshape(-1))
        chunks.setdefault("latent_count", []).append(latent_count.detach().reshape(-1))
        chunks.setdefault("det_mean_top1", []).append(top1.detach().reshape(-1))
        chunks.setdefault("det_mean_top2_gap", []).append(top2_gap.detach().reshape(-1))
        chunks.setdefault("det_mean_entropy_norm", []).append(entropy_norm.detach().reshape(-1))
        chunks.setdefault("det_mean_uniform_gap", []).append(uniform_gap.detach().reshape(-1))
        chunks.setdefault("action_det_l1", []).append(action_det_l1.detach().reshape(-1))
        chunks.setdefault("score_top2_gap", []).append(score_gap.detach().reshape(-1))
        chunks.setdefault("alpha_valid", []).append(alpha_valid.detach().reshape(-1))
        chunks.setdefault("alpha_valid_min", []).append(alpha_min.detach().reshape(-1))

    def _eval_stage_logprob_all(*, bw_shape_prefix: str | None = None) -> tuple[torch.Tensor, dict[str, float]]:
        batch_eval = 1024
        chunks: list[torch.Tensor] = []
        bw_shape_chunks: dict[str, list[torch.Tensor]] = {}
        with torch.no_grad():
            for start in range(0, sample_count, batch_eval):
                end = min(int(start) + int(batch_eval), int(sample_count))
                local_i = _slice_local_samples(start, end)
                action_i = _slice_sample_actions(start, end)
                latent_i = _slice_sample_latent_actions(start, end)
                lp_i, _ent_i, _out_i = learner._stage_actor_eval_from_batch(
                    stage_id_i,
                    local_i,
                    action_i,
                    num_agents,
                    compute_entropy=False,
                    latent_actions=latent_i if stage_id_i == 0 else None,
                )
                chunks.append(lp_i.detach())
                if bw_shape_prefix is not None and stage_id_i == 2:
                    _append_bw_shape_chunk(bw_shape_chunks, out_i=_out_i, local_i=local_i, action_i=action_i)
        logprob_all = torch.cat(chunks, dim=0).to(device=device, dtype=torch.float32)
        return logprob_all, _finalize_bw_shape_stats(str(bw_shape_prefix), bw_shape_chunks)

    def _bw_shape_delta_stats(pre: dict[str, float], post: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for metric in (
            "tau_mean",
            "tau_p50",
            "tau_low_frac",
            "tau_high_frac",
            "kappa_mean",
            "kappa_p50",
            "kappa_low_frac",
            "kappa_high_frac",
            "det_mean_top1_mean",
            "det_mean_top2_gap_mean",
            "det_mean_entropy_norm_mean",
            "det_mean_uniform_gap_mean",
            "action_det_l1_mean",
            "score_top2_gap_mean",
            "alpha_valid_min_mean",
            "alpha_lt1_frac",
        ):
            pre_key = f"bw_shape_pre_{metric}"
            post_key = f"bw_shape_post_{metric}"
            if pre_key in pre and post_key in post:
                out[f"bw_shape_delta_{metric}"] = float(post[post_key]) - float(pre[pre_key])
        return out

    old_t0 = time.perf_counter()
    old_logprob, bw_shape_pre_stats = _eval_stage_logprob_all(
        bw_shape_prefix="bw_shape_pre" if stage_id_i == 2 else None
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    old_logprob_sec = time.perf_counter() - old_t0

    cfg = getattr(learner, "cfg", None)
    parity_enabled = bool(getattr(cfg, "stage_actor_logprob_parity_check_enabled", True))
    parity_abs_mean = float("nan")
    parity_abs_max = float("nan")
    parity_bad_frac = float("nan")
    if parity_enabled:
        stored_old = stage_batch.old_logprobs.to(device=device, dtype=torch.float32).reshape(-1)
        if int(stored_old.numel()) != int(old_logprob.numel()):
            raise RuntimeError(
                f"{stage_name} old-logprob parity shape mismatch: "
                f"stored={tuple(stored_old.shape)} replay={tuple(old_logprob.shape)}."
            )
        abs_tol = max(float(getattr(cfg, "stage_actor_logprob_parity_abs_tol", 1.0e-3) or 0.0), 0.0)
        rel_tol = max(float(getattr(cfg, "stage_actor_logprob_parity_rel_tol", 1.0e-4) or 0.0), 0.0)
        diff = old_logprob.detach() - stored_old.detach()
        abs_diff = diff.abs()
        scale = torch.maximum(old_logprob.detach().abs(), stored_old.detach().abs())
        allowed = float(abs_tol) + float(rel_tol) * scale
        bad = abs_diff > allowed
        parity_abs_mean = float(abs_diff.mean().detach().cpu().item()) if int(abs_diff.numel()) else 0.0
        parity_abs_max = float(abs_diff.max().detach().cpu().item()) if int(abs_diff.numel()) else 0.0
        parity_bad_frac = float(bad.to(dtype=torch.float32).mean().detach().cpu().item()) if int(bad.numel()) else 0.0
        if stage_id_i == 2 and parity_dump_dir is not None and int(parity_topk) > 0 and int(abs_diff.numel()) > 0:
            dump_dir = Path(parity_dump_dir)
            dump_dir.mkdir(parents=True, exist_ok=True)
            topk = min(int(parity_topk), int(abs_diff.numel()))
            top_abs, sample_top_idx = torch.topk(abs_diff.detach(), k=topk)
            agent_offsets = torch.arange(num_agents, device=device, dtype=torch.long)
            flat_agent_idx = (sample_top_idx.reshape(-1, 1) * int(num_agents) + agent_offsets.reshape(1, -1)).reshape(-1)
            full_valid = (
                stage_batch.local_batch.gu_mask.to(device=device, dtype=torch.bool)
                & stage_batch.local_batch.bw_valid_mask.to(device=device, dtype=torch.bool)
            )
            local_top = _index_dataclass(stage_batch.local_batch, flat_agent_idx)
            action_top = all_actions.index_select(0, sample_top_idx)
            valid_top = full_valid.index_select(0, flat_agent_idx).reshape(topk, num_agents, -1)
            with torch.no_grad():
                replay_top, _ent_top, out_top = learner._stage_actor_eval_from_batch(
                    stage_id_i,
                    local_top,
                    action_top,
                    num_agents,
                    compute_entropy=False,
                    latent_actions=None,
                )
                det_mean_top = out_top.det_mean.reshape(topk, num_agents, -1)

                def _masked_min(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                    values_f = values.detach().to(dtype=torch.float32)
                    mask_b = mask.detach().to(dtype=torch.bool)
                    inf = torch.full_like(values_f, float("inf"))
                    masked = torch.where(mask_b, values_f, inf)
                    mins = masked.amin(dim=-1)
                    return torch.where(torch.isfinite(mins), mins, torch.full_like(mins, float("nan")))

                action_min_valid = _masked_min(action_top, valid_top)
                det_mean_min_valid = _masked_min(det_mean_top, valid_top)
                transition_indices_np = np.asarray(
                    getattr(stage_batch, "transition_indices", np.arange(sample_count, dtype=np.int64)),
                    dtype=np.int64,
                ).reshape(-1)
                env_indices_np = np.asarray(
                    getattr(stage_batch, "env_indices", np.full(sample_count, -1, dtype=np.int64)),
                    dtype=np.int64,
                ).reshape(-1)
                sample_top_cpu = sample_top_idx.detach().cpu().numpy().astype(np.int64, copy=False)
                transition_top_np = (
                    transition_indices_np[sample_top_cpu]
                    if transition_indices_np.size >= sample_count
                    else np.full((topk,), -1, dtype=np.int64)
                )
                env_top_np = (
                    env_indices_np[sample_top_cpu]
                    if env_indices_np.size >= sample_count
                    else np.full((topk,), -1, dtype=np.int64)
                )
                # Structured rollout indices are stage-interleaved:
                # transition = t * (3 * num_envs) + env * 3 + stage_id.
                # Prefer env_indices for env_id; derive step only when num_envs is known.
                num_envs_guess = int(env_indices_np.max() + 1) if env_indices_np.size > 0 and int(env_indices_np.max()) >= 0 else 0
                if num_envs_guess > 0:
                    step_top_np = transition_top_np // max(3 * num_envs_guess, 1)
                    stage_top_np = transition_top_np % 3
                else:
                    step_top_np = np.full((topk,), -1, dtype=np.int64)
                    stage_top_np = np.full((topk,), -1, dtype=np.int64)
                uav_idx_top = torch.arange(num_agents, device=device, dtype=torch.long).reshape(1, num_agents).expand(topk, -1)
            payload: dict[str, Any] = {
                "stage": stage_name,
                "tag": str(parity_dump_tag or ""),
                "sample_count": int(sample_count),
                "num_agents": int(num_agents),
                "top_indices": sample_top_idx.detach().cpu(),
                "flat_agent_indices": flat_agent_idx.detach().cpu(),
                "transition_indices": torch.as_tensor(transition_top_np, dtype=torch.long),
                "env_indices": torch.as_tensor(env_top_np, dtype=torch.long),
                "step_indices": torch.as_tensor(step_top_np, dtype=torch.long),
                "stage_indices": torch.as_tensor(stage_top_np, dtype=torch.long),
                "uav_indices": uav_idx_top.detach().cpu(),
                "top_abs_diff": top_abs.detach().cpu(),
                "stored_old_logprob": stored_old.index_select(0, sample_top_idx).detach().cpu(),
                "replay_old_logprob": old_logprob.index_select(0, sample_top_idx).detach().cpu(),
                "replay_old_logprob_top_recomputed": replay_top.detach().cpu(),
                "diff": diff.index_select(0, sample_top_idx).detach().cpu(),
                "allowed": allowed.index_select(0, sample_top_idx).detach().cpu(),
                "failed": bool(bad.any().detach().cpu().item()),
                "worst_index": int(abs_diff.argmax().detach().cpu().item()),
                "parity_abs_mean": parity_abs_mean,
                "parity_abs_max": parity_abs_max,
                "parity_bad_frac": parity_bad_frac,
                "bad_indices": bad.nonzero(as_tuple=False).reshape(-1).detach().cpu(),
                "actions": action_top.detach().cpu(),
                "valid_mask": valid_top.detach().cpu(),
                "replay_det_mean": det_mean_top.detach().cpu(),
                "action_min_all": action_top.amin(dim=-1).detach().cpu(),
                "action_min_valid": action_min_valid.detach().cpu(),
                "det_mean_min_all": det_mean_top.amin(dim=-1).detach().cpu(),
                "det_mean_min_valid": det_mean_min_valid.detach().cpu(),
            }
            if getattr(out_top, "kappa", None) is not None:
                payload["replay_kappa"] = out_top.kappa.reshape(topk, num_agents).detach().cpu()
            if getattr(out_top, "tau", None) is not None:
                payload["replay_tau"] = out_top.tau.reshape(topk, num_agents).detach().cpu()
            if getattr(out_top, "valid_count", None) is not None:
                payload["replay_valid_count"] = out_top.valid_count.reshape(topk, num_agents).detach().cpu()
            bw_ref_actions = getattr(stage_batch, "bw_ref_actions", None)
            if bw_ref_actions is not None:
                payload["stored_ref_action"] = (
                    bw_ref_actions.to(device=device)
                    .index_select(0, sample_top_idx)
                    .detach()
                    .cpu()
                )
            per_agent_old = getattr(stage_batch, "old_logprobs_per_agent", None)
            if per_agent_old is not None:
                payload["stored_old_logprob_per_agent"] = (
                    per_agent_old.to(device=device, dtype=torch.float32)
                    .index_select(0, sample_top_idx)
                    .detach()
                    .cpu()
                )
            stored_dump_fields = (
                ("bw_tau", "stored_tau", torch.float32),
                ("bw_kappa", "stored_kappa", torch.float32),
                ("bw_valid_count", "stored_valid_count", torch.long),
                ("bw_latent_count", "stored_latent_count", torch.long),
                ("bw_logprob_raw_per_agent", "stored_logprob_raw_per_agent", torch.float32),
            )
            for attr_name, payload_name, dtype in stored_dump_fields:
                value = getattr(stage_batch, attr_name, None)
                if value is None:
                    continue
                payload[payload_name] = (
                    value.to(device=device, dtype=dtype)
                    .index_select(0, sample_top_idx)
                    .detach()
                    .cpu()
                )
            torch.save(payload, dump_dir / f"bw_parity_topk_{str(parity_dump_tag or 'untagged')}.pt")
        if bool(bad.any().detach().cpu().item()):
            worst = int(abs_diff.argmax().detach().cpu().item())
            stored_v = float(stored_old[worst].detach().cpu().item())
            replay_v = float(old_logprob[worst].detach().cpu().item())
            diff_v = float(diff[worst].detach().cpu().item())
            allowed_v = float(allowed[worst].detach().cpu().item())
            raise RuntimeError(
                f"{stage_name} old-logprob parity check failed: "
                f"sample={worst}, stored/native={stored_v:.9g}, replay/torch={replay_v:.9g}, "
                f"diff={diff_v:.9g}, allowed={allowed_v:.9g}, "
                f"abs_mean={parity_abs_mean:.9g}, abs_max={parity_abs_max:.9g}, "
                f"bad_frac={parity_bad_frac:.9g}, abs_tol={abs_tol:.3g}, rel_tol={rel_tol:.3g}. "
                "Do not bypass this by choosing one logprob source; fix the stage action/mask/logprob parity."
            )

    adv = stage_advantages.detach().to(device=device, dtype=torch.float32).reshape(-1)
    if int(adv.numel()) != sample_count:
        raise RuntimeError(f"advantage length {int(adv.numel())} != {stage_name} stage samples {sample_count}.")
    raw_adv: torch.Tensor | None = None
    if stage_raw_advantages is not None:
        raw_adv = stage_raw_advantages.detach().to(device=device, dtype=torch.float32).reshape(-1)
        if int(raw_adv.numel()) != sample_count:
            raise RuntimeError(
                f"raw advantage length {int(raw_adv.numel())} != {stage_name} stage samples {sample_count}."
            )
    entropy_coef = float(learner.entropy_coef_by_stage[stage_id_i])
    outer_minibatches = _env_group_minibatch_indices(
        stage_batch,
        sample_count=sample_count,
        minibatches=max(int(minibatches), 1),
        device=device,
    )
    if not outer_minibatches:
        raise RuntimeError(f"{stage_name} actor has no optimizer minibatches.")

    last_policy_loss = 0.0
    last_entropy = 0.0
    last_loss = 0.0
    last_grad_norm = 0.0
    last_kl = 0.0
    last_clip_frac = 0.0
    last_danger_loss = 0.0
    last_danger_active = 0.0
    update_loop_sec = 0.0
    epochs_completed = 0
    early_stop_triggered = False
    early_stop_epoch = -1
    early_stop_reason = ""
    kl_stop_threshold_eff = (
        float(kl_stop_threshold)
        if kl_stop_threshold is not None and float(kl_stop_threshold) > 0.0
        else 0.0
    )
    clip_stop_threshold_eff = (
        float(clip_stop_threshold)
        if clip_stop_threshold is not None and float(clip_stop_threshold) > 0.0
        else 0.0
    )
    need_entropy = abs(float(entropy_coef)) > 0.0
    actor_chunk_preferred = int(
        getattr(
            learner,
            "actor_update_microbatch_size",
            getattr(getattr(learner, "cfg", None), "actor_update_microbatch_size", 1024),
        )
        or 1024
    )
    actor_chunk_preferred = max(int(actor_chunk_preferred), 1)
    danger_enabled = (
        stage_id_i == 0
        and bool(getattr(learner, "danger_imitation_enabled", False))
        and float(getattr(learner, "danger_imitation_coef", 0.0) or 0.0) != 0.0
    )
    danger_targets_all = getattr(stage_batch, "danger_imitation_targets", None)
    danger_masks_all = getattr(stage_batch, "danger_imitation_masks", None)
    if bool(danger_enabled) and danger_targets_all is not None and danger_masks_all is not None:
        danger_targets_all = danger_targets_all.to(device=device)
        danger_masks_all = danger_masks_all.to(device=device)
    else:
        danger_enabled = False
        danger_targets_all = torch.zeros((sample_count, num_agents, 2), dtype=torch.float32, device=device)
        danger_masks_all = torch.zeros((sample_count, num_agents, 2), dtype=torch.float32, device=device)
    if _actor_compile_enabled(learner, stage_id_i):
        loss_chunk_fn = _get_compiled_stage_actor_loss_chunk_fn(
            learner,
            stage_id=stage_id_i,
            num_agents=num_agents,
            need_entropy=need_entropy,
            danger_enabled=danger_enabled,
        )
        actor_loss_compile_enabled = True
    else:
        loss_chunk_fn = _make_stage_actor_loss_chunk_fn(
            learner,
            stage_id=stage_id_i,
            num_agents=num_agents,
            need_entropy=need_entropy,
            danger_enabled=danger_enabled,
        )
        actor_loss_compile_enabled = False

    def _grad_norm_for_named_prefix(prefix: str) -> float:
        total_sq = 0.0
        matched = False
        for name, param in learner.actor.named_parameters():
            if not str(name).startswith(str(prefix)):
                continue
            grad = param.grad
            if grad is None:
                continue
            matched = True
            norm_v = float(grad.detach().to(dtype=torch.float32).norm(2).detach().cpu().item())
            total_sq += norm_v * norm_v
        return math.sqrt(total_sq) if matched else 0.0

    def _series_stats(prefix: str, values: list[float]) -> dict[str, float]:
        if not values:
            return {f"{prefix}_mean": 0.0, f"{prefix}_max": 0.0}
        arr = np.asarray(values, dtype=np.float64)
        return {f"{prefix}_mean": float(np.mean(arr)), f"{prefix}_max": float(np.max(arr))}

    grad_norm_values: list[float] = []
    grad_clip_count = 0
    grad_step_count = 0
    bw_grad_group_values: dict[str, list[float]] = {
        "bw_grad_norm_policy": [],
        "bw_grad_norm_score_head": [],
        "bw_grad_norm_tau_head": [],
        "bw_grad_norm_kappa_head": [],
    }
    max_grad_norm_eff = float(learner.max_grad_norm)

    for _epoch in range(max(int(epochs), 1)):
        epoch_t0 = time.perf_counter()
        epoch_sample_total = 0
        epoch_kl_weighted = 0.0
        epoch_clip_weighted = 0.0
        for group_idx in _shuffled_group_order(len(outer_minibatches)):
            idx_full = outer_minibatches[int(group_idx)]
            mb_total = int(idx_full.numel())
            if mb_total <= 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            chunk_size = 2048
            policy_sum_t = torch.zeros((), dtype=torch.float32, device=device)
            entropy_sum_t = torch.zeros((), dtype=torch.float32, device=device)
            kl_sum_t = torch.zeros((), dtype=torch.float32, device=device)
            clip_sum_t = torch.zeros((), dtype=torch.float32, device=device)
            danger_loss_sum_t = torch.zeros((), dtype=torch.float32, device=device)
            danger_active_sum_t = torch.zeros((), dtype=torch.float32, device=device)
            chunk_size = actor_chunk_preferred
            for chunk_start in range(0, mb_total, chunk_size):
                idx = idx_full[int(chunk_start) : int(chunk_start) + int(chunk_size)]
                real_count = int(idx.numel())
                if real_count <= 0:
                    continue
                weight = float(real_count) / float(max(mb_total, 1))
                pad_count = int(chunk_size) - int(real_count)
                forward_idx = idx
                if bool(actor_loss_compile_enabled) and pad_count > 0 and real_count > 0:
                    pad = idx.new_full((pad_count,), int(idx[-1].detach().item()))
                    forward_idx = torch.cat([idx, pad], dim=0)
                local_i = _index_local_samples(forward_idx)
                action_i = _index_sample_actions(forward_idx)
                latent_i = _index_sample_latent_actions(forward_idx)
                old_i = old_logprob.index_select(0, forward_idx)
                adv_i = adv.index_select(0, forward_idx)
                danger_targets_i = danger_targets_all.index_select(0, forward_idx)
                danger_masks_i = danger_masks_all.index_select(0, forward_idx)
                valid_i = torch.ones((real_count,), dtype=torch.float32, device=device)
                if bool(actor_loss_compile_enabled) and pad_count > 0 and real_count > 0:
                    adv_i = adv_i.clone()
                    adv_i[real_count:] = 0.0
                    danger_masks_i = danger_masks_i.clone()
                    danger_masks_i[real_count:] = 0.0
                    valid_i = torch.cat([valid_i, valid_i.new_zeros((pad_count,))], dim=0)
                loss, policy_loss, entropy_mean, approx_kl, clip_frac, danger_loss, danger_active = loss_chunk_fn(
                    local_i,
                    action_i,
                    latent_i,
                    old_i,
                    adv_i,
                    danger_targets_i,
                    danger_masks_i,
                    valid_i,
                )
                (loss * float(weight)).backward()
                with torch.no_grad():
                    w = float(weight)
                    policy_sum_t = policy_sum_t + policy_loss.detach().to(dtype=torch.float32) * w
                    entropy_sum_t = entropy_sum_t + entropy_mean.detach().to(dtype=torch.float32) * w
                    kl_sum_t = kl_sum_t + approx_kl.detach().to(dtype=torch.float32) * w
                    clip_sum_t = clip_sum_t + clip_frac.detach().to(dtype=torch.float32) * w
                    danger_loss_sum_t = danger_loss_sum_t + danger_loss.detach().to(dtype=torch.float32) * w
                    danger_active_sum_t = danger_active_sum_t + danger_active.detach().to(dtype=torch.float32) * w
            if stage_id_i == 2:
                bw_grad_group_values["bw_grad_norm_policy"].append(_grad_norm_for_named_prefix("bw_policy"))
                bw_grad_group_values["bw_grad_norm_score_head"].append(_grad_norm_for_named_prefix("bw_policy.score_head"))
                bw_grad_group_values["bw_grad_norm_tau_head"].append(_grad_norm_for_named_prefix("bw_policy.tau_head"))
                bw_grad_group_values["bw_grad_norm_kappa_head"].append(_grad_norm_for_named_prefix("bw_policy.kappa_head"))
            grad_norm = torch.nn.utils.clip_grad_norm_(params, float(learner.max_grad_norm))
            optimizer.step()
            with torch.no_grad():
                policy_sum = float(policy_sum_t.detach().cpu().item())
                entropy_sum = float(entropy_sum_t.detach().cpu().item())
                danger_loss_sum = float(danger_loss_sum_t.detach().cpu().item())
                danger_active_sum = float(danger_active_sum_t.detach().cpu().item())
                last_policy_loss = float(policy_sum)
                last_entropy = float(entropy_sum)
                last_danger_loss = float(danger_loss_sum)
                last_danger_active = float(danger_active_sum)
                last_loss = float(
                    policy_sum
                    - entropy_coef * entropy_sum
                    + float(getattr(learner, "danger_imitation_coef", 0.0)) * danger_loss_sum
                )
                last_grad_norm = float(torch.as_tensor(grad_norm).detach().cpu().item())
                grad_step_count += 1
                grad_norm_values.append(float(last_grad_norm))
                if max_grad_norm_eff > 0.0 and float(last_grad_norm) > max_grad_norm_eff:
                    grad_clip_count += 1
                last_kl = float(kl_sum_t.detach().cpu().item())
                last_clip_frac = float(clip_sum_t.detach().cpu().item())
                epoch_sample_total += int(mb_total)
                epoch_kl_weighted += float(last_kl) * float(mb_total)
                epoch_clip_weighted += float(last_clip_frac) * float(mb_total)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        update_loop_sec += time.perf_counter() - epoch_t0
        epochs_completed += 1
        if epoch_sample_total > 0:
            last_kl = float(epoch_kl_weighted / float(epoch_sample_total))
            last_clip_frac = float(epoch_clip_weighted / float(epoch_sample_total))
        if kl_stop_threshold_eff > 0.0 and last_kl > kl_stop_threshold_eff:
            early_stop_triggered = True
            early_stop_epoch = int(_epoch + 1)
            early_stop_reason = "kl"
            break
        if clip_stop_threshold_eff > 0.0 and last_clip_frac > clip_stop_threshold_eff:
            early_stop_triggered = True
            early_stop_epoch = int(_epoch + 1)
            early_stop_reason = "clip"
            break

    credit_diag_t0 = time.perf_counter()
    post_logprob, bw_shape_post_stats = _eval_stage_logprob_all(
        bw_shape_prefix="bw_shape_post" if stage_id_i == 2 else None
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    credit_diag_sec = time.perf_counter() - credit_diag_t0
    bw_shape_delta_stats = _bw_shape_delta_stats(bw_shape_pre_stats, bw_shape_post_stats)

    def _masked_scalar_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        mask_b = mask.detach().to(device=values.device, dtype=torch.bool).reshape(-1)
        values_f = values.detach().to(dtype=torch.float32).reshape(-1)
        if int(mask_b.sum().detach().cpu().item()) <= 0:
            return 0.0
        return float(values_f[mask_b].mean().detach().cpu().item())

    def _masked_scalar_sum_abs(values: torch.Tensor, mask: torch.Tensor) -> float:
        mask_b = mask.detach().to(device=values.device, dtype=torch.bool).reshape(-1)
        values_f = values.detach().to(dtype=torch.float32).reshape(-1)
        if int(mask_b.sum().detach().cpu().item()) <= 0:
            return 0.0
        return float(values_f[mask_b].abs().sum().detach().cpu().item())

    def _masked_scalar_frac(mask: torch.Tensor, denom_mask: torch.Tensor) -> float:
        denom_b = denom_mask.detach().to(dtype=torch.bool).reshape(-1)
        denom_count = int(denom_b.sum().detach().cpu().item())
        if denom_count <= 0:
            return 0.0
        mask_b = mask.detach().to(dtype=torch.bool).reshape(-1)
        return float((mask_b & denom_b).to(dtype=torch.float32).sum().detach().cpu().item() / float(denom_count))

    def _masked_scalar_stats(prefix: str, values: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
        mask_b = mask.detach().to(device=values.device, dtype=torch.bool).reshape(-1)
        values_f = values.detach().to(dtype=torch.float32).reshape(-1)
        if int(mask_b.sum().detach().cpu().item()) <= 0:
            return {
                f"{prefix}_mean": 0.0,
                f"{prefix}_p05": 0.0,
                f"{prefix}_p10": 0.0,
                f"{prefix}_p25": 0.0,
                f"{prefix}_p50": 0.0,
                f"{prefix}_p75": 0.0,
                f"{prefix}_p90": 0.0,
                f"{prefix}_p95": 0.0,
            }
        selected = values_f[mask_b]
        quantiles = torch.tensor([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], device=selected.device)
        qs = torch.quantile(selected, quantiles).detach().cpu()
        return {
            f"{prefix}_mean": float(selected.mean().detach().cpu().item()),
            f"{prefix}_p05": float(qs[0].item()),
            f"{prefix}_p10": float(qs[1].item()),
            f"{prefix}_p25": float(qs[2].item()),
            f"{prefix}_p50": float(qs[3].item()),
            f"{prefix}_p75": float(qs[4].item()),
            f"{prefix}_p90": float(qs[5].item()),
            f"{prefix}_p95": float(qs[6].item()),
        }

    with torch.no_grad():
        delta_logprob = (post_logprob - old_logprob).detach().to(dtype=torch.float32).reshape(-1)
        ratio_full = torch.exp(torch.clamp(delta_logprob, min=-20.0, max=20.0))
        credit = (adv.detach().to(dtype=torch.float32).reshape(-1) * delta_logprob).detach()
        adv_pos = adv > 0.0
        adv_neg = adv < 0.0
        adv_nonzero = adv.abs() > 1.0e-8
        direction_agree = ((adv_pos & (delta_logprob > 0.0)) | (adv_neg & (delta_logprob < 0.0))) & adv_nonzero
        direction_wrong = ((adv_pos & (delta_logprob < 0.0)) | (adv_neg & (delta_logprob > 0.0))) & adv_nonzero
        nonzero_count = int(adv_nonzero.sum().detach().cpu().item())
        nonzero_denom = float(max(nonzero_count, 1))
        credit_stats = _summ_tensor(credit)
        delta_stats = _summ_tensor(delta_logprob)
        ratio_stats = _summ_tensor(ratio_full)
        full_clip_frac = float(
            (torch.abs(ratio_full - 1.0) > float(learner.clip_ratio))
            .to(dtype=torch.float32)
            .mean()
            .detach()
            .cpu()
            .item()
        )
        credit_positive_frac = float((credit > 0.0).to(dtype=torch.float32).mean().detach().cpu().item())
        direction_agree_frac = float(direction_agree.to(dtype=torch.float32).sum().detach().cpu().item() / nonzero_denom)
        direction_wrong_frac = float(direction_wrong.to(dtype=torch.float32).sum().detach().cpu().item() / nonzero_denom)
        adv_pos_frac = float(adv_pos.to(dtype=torch.float32).mean().detach().cpu().item())
        adv_neg_frac = float(adv_neg.to(dtype=torch.float32).mean().detach().cpu().item())
        pos_count = int(adv_pos.sum().detach().cpu().item())
        neg_count = int(adv_neg.sum().detach().cpu().item())
        pos_delta_positive_frac = (
            float(((delta_logprob > 0.0) & adv_pos).to(dtype=torch.float32).sum().detach().cpu().item() / float(pos_count))
            if pos_count > 0
            else 0.0
        )
        neg_delta_negative_frac = (
            float(((delta_logprob < 0.0) & adv_neg).to(dtype=torch.float32).sum().detach().cpu().item() / float(neg_count))
            if neg_count > 0
            else 0.0
        )
        pos_up = adv_pos & (delta_logprob > 0.0)
        pos_down = adv_pos & (delta_logprob < 0.0)
        neg_down = adv_neg & (delta_logprob < 0.0)
        neg_up = adv_neg & (delta_logprob > 0.0)
        correct_delta_abs_sum = _masked_scalar_sum_abs(delta_logprob, direction_agree)
        wrong_delta_abs_sum = _masked_scalar_sum_abs(delta_logprob, direction_wrong)
        pos_up_abs_sum = _masked_scalar_sum_abs(delta_logprob, pos_up)
        pos_down_abs_sum = _masked_scalar_sum_abs(delta_logprob, pos_down)
        neg_down_abs_sum = _masked_scalar_sum_abs(delta_logprob, neg_down)
        neg_up_abs_sum = _masked_scalar_sum_abs(delta_logprob, neg_up)
        credit_positive_abs_sum = _masked_scalar_sum_abs(credit, credit > 0.0)
        credit_negative_abs_sum = _masked_scalar_sum_abs(credit, credit < 0.0)
        adv_pos_abs_sum = _masked_scalar_sum_abs(adv, adv_pos)
        adv_neg_abs_sum = _masked_scalar_sum_abs(adv, adv_neg)
        ratio_upper_clip = ratio_full > (1.0 + float(learner.clip_ratio))
        ratio_lower_clip = ratio_full < (1.0 - float(learner.clip_ratio))
        pos_active = adv_pos & (~ratio_upper_clip)
        neg_active = adv_neg & (~ratio_lower_clip)
        pos_active_abs_sum = _masked_scalar_sum_abs(adv, pos_active)
        neg_active_abs_sum = _masked_scalar_sum_abs(adv, neg_active)
        all_mask = torch.ones_like(adv_pos, dtype=torch.bool)
        delta_magnitude_diag: dict[str, float] = {
            **_masked_scalar_stats("delta_logprob", delta_logprob, all_mask),
            **_masked_scalar_stats("delta_logprob_abs", delta_logprob.abs(), all_mask),
            **_masked_scalar_stats("delta_logprob_when_adv_positive", delta_logprob, adv_pos),
            **_masked_scalar_stats("delta_logprob_when_adv_negative", delta_logprob, adv_neg),
            **_masked_scalar_stats("credit", credit, all_mask),
            **_masked_scalar_stats("credit_when_adv_positive", credit, adv_pos),
            **_masked_scalar_stats("credit_when_adv_negative", credit, adv_neg),
            "delta_logprob_abs_gt_0p02_frac": _masked_scalar_frac(delta_logprob.abs() > 0.02, all_mask),
            "delta_logprob_abs_gt_0p05_frac": _masked_scalar_frac(delta_logprob.abs() > 0.05, all_mask),
            "delta_logprob_abs_gt_0p10_frac": _masked_scalar_frac(delta_logprob.abs() > 0.10, all_mask),
            "delta_logprob_abs_gt_0p20_frac": _masked_scalar_frac(delta_logprob.abs() > 0.20, all_mask),
            "delta_logprob_abs_gt_0p05_when_adv_positive_frac": _masked_scalar_frac(
                delta_logprob.abs() > 0.05, adv_pos
            ),
            "delta_logprob_abs_gt_0p05_when_adv_negative_frac": _masked_scalar_frac(
                delta_logprob.abs() > 0.05, adv_neg
            ),
            "delta_logprob_direction_correct_abs_sum": float(correct_delta_abs_sum),
            "delta_logprob_direction_wrong_abs_sum": float(wrong_delta_abs_sum),
            "delta_logprob_direction_correct_abs_share": float(
                correct_delta_abs_sum / max(correct_delta_abs_sum + wrong_delta_abs_sum, 1.0e-12)
            ),
            "delta_logprob_adv_positive_up_abs_sum": float(pos_up_abs_sum),
            "delta_logprob_adv_positive_down_abs_sum": float(pos_down_abs_sum),
            "delta_logprob_adv_positive_up_abs_share": float(
                pos_up_abs_sum / max(pos_up_abs_sum + pos_down_abs_sum, 1.0e-12)
            ),
            "delta_logprob_adv_negative_down_abs_sum": float(neg_down_abs_sum),
            "delta_logprob_adv_negative_up_abs_sum": float(neg_up_abs_sum),
            "delta_logprob_adv_negative_down_abs_share": float(
                neg_down_abs_sum / max(neg_down_abs_sum + neg_up_abs_sum, 1.0e-12)
            ),
            "credit_positive_abs_sum": float(credit_positive_abs_sum),
            "credit_negative_abs_sum": float(credit_negative_abs_sum),
            "credit_positive_abs_share": float(
                credit_positive_abs_sum / max(credit_positive_abs_sum + credit_negative_abs_sum, 1.0e-12)
            ),
            "adv_positive_abs_mass": float(adv_pos_abs_sum),
            "adv_negative_abs_mass": float(adv_neg_abs_sum),
            "adv_positive_abs_mass_share": float(adv_pos_abs_sum / max(adv_pos_abs_sum + adv_neg_abs_sum, 1.0e-12)),
            "adv_negative_abs_mass_share": float(adv_neg_abs_sum / max(adv_pos_abs_sum + adv_neg_abs_sum, 1.0e-12)),
            "adv_positive_upper_clip_frac": _masked_scalar_frac(ratio_upper_clip, adv_pos),
            "adv_negative_lower_clip_frac": _masked_scalar_frac(ratio_lower_clip, adv_neg),
            "adv_positive_active_abs_mass": float(pos_active_abs_sum),
            "adv_negative_active_abs_mass": float(neg_active_abs_sum),
            "adv_positive_active_abs_mass_share": float(
                pos_active_abs_sum / max(pos_active_abs_sum + neg_active_abs_sum, 1.0e-12)
            ),
            "adv_negative_active_abs_mass_share": float(
                neg_active_abs_sum / max(pos_active_abs_sum + neg_active_abs_sum, 1.0e-12)
            ),
        }
        raw_adv_diag: dict[str, float] = {"raw_adv_available": 0.0}
        if raw_adv is not None:
            raw_adv_t = raw_adv.detach().to(dtype=torch.float32).reshape(-1)
            raw_pos = raw_adv_t > 0.0
            raw_neg = raw_adv_t < 0.0
            raw_nonzero = raw_adv_t.abs() > 1.0e-8
            raw_credit = (raw_adv_t * delta_logprob).detach()
            raw_direction_agree = (
                ((raw_pos & (delta_logprob > 0.0)) | (raw_neg & (delta_logprob < 0.0))) & raw_nonzero
            )
            raw_direction_wrong = (
                ((raw_pos & (delta_logprob < 0.0)) | (raw_neg & (delta_logprob > 0.0))) & raw_nonzero
            )
            norm_raw_nonzero = adv_nonzero & raw_nonzero
            norm_raw_agree = ((adv_pos & raw_pos) | (adv_neg & raw_neg)) & norm_raw_nonzero
            norm_raw_disagree = ((adv_pos & raw_neg) | (adv_neg & raw_pos)) & norm_raw_nonzero
            raw_credit_stats = _summ_tensor(raw_credit)
            raw_pos_up = raw_pos & (delta_logprob > 0.0)
            raw_pos_down = raw_pos & (delta_logprob < 0.0)
            raw_neg_down = raw_neg & (delta_logprob < 0.0)
            raw_neg_up = raw_neg & (delta_logprob > 0.0)
            raw_pos_up_abs_sum = _masked_scalar_sum_abs(delta_logprob, raw_pos_up)
            raw_pos_down_abs_sum = _masked_scalar_sum_abs(delta_logprob, raw_pos_down)
            raw_neg_down_abs_sum = _masked_scalar_sum_abs(delta_logprob, raw_neg_down)
            raw_neg_up_abs_sum = _masked_scalar_sum_abs(delta_logprob, raw_neg_up)
            raw_credit_positive_abs_sum = _masked_scalar_sum_abs(raw_credit, raw_credit > 0.0)
            raw_credit_negative_abs_sum = _masked_scalar_sum_abs(raw_credit, raw_credit < 0.0)

            def _frac_mask(mask: torch.Tensor, denom_mask: torch.Tensor) -> float:
                denom_count = int(denom_mask.to(dtype=torch.bool).sum().detach().cpu().item())
                if denom_count <= 0:
                    return 0.0
                return float((mask & denom_mask).to(dtype=torch.float32).sum().detach().cpu().item() / float(denom_count))

            raw_pos_count = int(raw_pos.sum().detach().cpu().item())
            raw_neg_count = int(raw_neg.sum().detach().cpu().item())
            raw_nonzero_count = int(raw_nonzero.sum().detach().cpu().item())
            norm_raw_count = int(norm_raw_nonzero.sum().detach().cpu().item())
            raw_adv_diag = {
                "raw_adv_available": 1.0,
                "raw_adv_positive_frac": float(raw_pos.to(dtype=torch.float32).mean().detach().cpu().item()),
                "raw_adv_negative_frac": float(raw_neg.to(dtype=torch.float32).mean().detach().cpu().item()),
                "norm_raw_adv_sign_agree_frac": float(
                    norm_raw_agree.to(dtype=torch.float32).sum().detach().cpu().item() / float(max(norm_raw_count, 1))
                ),
                "norm_raw_adv_sign_disagree_frac": float(
                    norm_raw_disagree.to(dtype=torch.float32).sum().detach().cpu().item() / float(max(norm_raw_count, 1))
                ),
                "norm_adv_positive_raw_adv_negative_within_norm_positive_frac": _frac_mask(raw_neg, adv_pos),
                "norm_adv_negative_raw_adv_positive_within_norm_negative_frac": _frac_mask(raw_pos, adv_neg),
                "raw_adv_positive_delta_positive_frac": (
                    float(((delta_logprob > 0.0) & raw_pos).to(dtype=torch.float32).sum().detach().cpu().item() / float(raw_pos_count))
                    if raw_pos_count > 0
                    else 0.0
                ),
                "raw_adv_positive_delta_negative_frac": (
                    float(((delta_logprob < 0.0) & raw_pos).to(dtype=torch.float32).sum().detach().cpu().item() / float(raw_pos_count))
                    if raw_pos_count > 0
                    else 0.0
                ),
                "raw_adv_negative_delta_negative_frac": (
                    float(((delta_logprob < 0.0) & raw_neg).to(dtype=torch.float32).sum().detach().cpu().item() / float(raw_neg_count))
                    if raw_neg_count > 0
                    else 0.0
                ),
                "raw_adv_negative_delta_positive_frac": (
                    float(((delta_logprob > 0.0) & raw_neg).to(dtype=torch.float32).sum().detach().cpu().item() / float(raw_neg_count))
                    if raw_neg_count > 0
                    else 0.0
                ),
                "delta_logprob_when_raw_adv_positive_mean": _masked_scalar_mean(delta_logprob, raw_pos),
                "delta_logprob_when_raw_adv_negative_mean": _masked_scalar_mean(delta_logprob, raw_neg),
                **_masked_scalar_stats("delta_logprob_when_raw_adv_positive", delta_logprob, raw_pos),
                **_masked_scalar_stats("delta_logprob_when_raw_adv_negative", delta_logprob, raw_neg),
                "delta_logprob_raw_adv_positive_up_abs_sum": float(raw_pos_up_abs_sum),
                "delta_logprob_raw_adv_positive_down_abs_sum": float(raw_pos_down_abs_sum),
                "delta_logprob_raw_adv_positive_up_abs_share": float(
                    raw_pos_up_abs_sum / max(raw_pos_up_abs_sum + raw_pos_down_abs_sum, 1.0e-12)
                ),
                "delta_logprob_raw_adv_negative_down_abs_sum": float(raw_neg_down_abs_sum),
                "delta_logprob_raw_adv_negative_up_abs_sum": float(raw_neg_up_abs_sum),
                "delta_logprob_raw_adv_negative_down_abs_share": float(
                    raw_neg_down_abs_sum / max(raw_neg_down_abs_sum + raw_neg_up_abs_sum, 1.0e-12)
                ),
                "raw_credit_mean": float(raw_credit_stats["mean"]),
                "raw_credit_std": float(raw_credit_stats["std"]),
                "raw_credit_min": float(raw_credit_stats["min"]),
                "raw_credit_max": float(raw_credit_stats["max"]),
                **_masked_scalar_stats("raw_credit_nonzero", raw_credit, raw_nonzero),
                "raw_credit_positive_frac": float(
                    (raw_credit > 0.0).to(dtype=torch.float32).mean().detach().cpu().item()
                ),
                "raw_credit_positive_abs_sum": float(raw_credit_positive_abs_sum),
                "raw_credit_negative_abs_sum": float(raw_credit_negative_abs_sum),
                "raw_credit_positive_abs_share": float(
                    raw_credit_positive_abs_sum
                    / max(raw_credit_positive_abs_sum + raw_credit_negative_abs_sum, 1.0e-12)
                ),
                "raw_credit_direction_agree_frac": float(
                    raw_direction_agree.to(dtype=torch.float32).sum().detach().cpu().item() / float(max(raw_nonzero_count, 1))
                ),
                "raw_credit_direction_wrong_frac": float(
                    raw_direction_wrong.to(dtype=torch.float32).sum().detach().cpu().item() / float(max(raw_nonzero_count, 1))
                ),
                "raw_credit_when_raw_adv_positive_mean": _masked_scalar_mean(raw_credit, raw_pos),
                "raw_credit_when_raw_adv_negative_mean": _masked_scalar_mean(raw_credit, raw_neg),
            }

    grad_diag: dict[str, float] = {
        "grad_norm_mean": float(np.mean(np.asarray(grad_norm_values, dtype=np.float64))) if grad_norm_values else 0.0,
        "grad_norm_max": float(np.max(np.asarray(grad_norm_values, dtype=np.float64))) if grad_norm_values else 0.0,
        "grad_norm_clip_threshold": float(max_grad_norm_eff),
        "grad_norm_clip_frac": float(grad_clip_count / max(grad_step_count, 1)),
    }
    if stage_id_i == 2:
        for key, values in bw_grad_group_values.items():
            grad_diag.update(_series_stats(key, values))

    return {
        "actor_samples": float(sample_count),
        "actor_epochs": float(epochs),
        "actor_epochs_completed": float(epochs_completed),
        "actor_early_stop": float(1.0 if early_stop_triggered else 0.0),
        "actor_early_stop_epoch": float(early_stop_epoch),
        "actor_early_stop_reason_code": float({"": 0, "kl": 1, "clip": 2}.get(early_stop_reason, 0)),
        "actor_kl_stop_threshold": float(kl_stop_threshold_eff),
        "actor_clip_stop_threshold": float(clip_stop_threshold_eff),
        "actor_minibatches": float(minibatches),
        "actor_outer_minibatches": float(len(outer_minibatches)),
        f"policy_loss_{stage_name}": float(last_policy_loss),
        f"entropy_{stage_name}": float(last_entropy),
        f"actor_loss_{stage_name}": float(last_loss),
        f"grad_norm_{stage_name}": float(last_grad_norm),
        f"approx_kl_{stage_name}": float(last_kl),
        f"clip_frac_{stage_name}": float(last_clip_frac),
        "full_update_kl": float((-delta_logprob).mean().detach().cpu().item()),
        "full_update_clip_frac": float(full_clip_frac),
        "full_update_ratio_mean": float(ratio_stats["mean"]),
        "full_update_ratio_std": float(ratio_stats["std"]),
        "full_update_ratio_min": float(ratio_stats["min"]),
        "full_update_ratio_max": float(ratio_stats["max"]),
        "delta_logprob_mean": float(delta_stats["mean"]),
        "delta_logprob_std": float(delta_stats["std"]),
        "delta_logprob_min": float(delta_stats["min"]),
        "delta_logprob_max": float(delta_stats["max"]),
        "delta_logprob_abs_mean": float(delta_logprob.abs().mean().detach().cpu().item()),
        "delta_logprob_when_adv_positive_mean": _masked_scalar_mean(delta_logprob, adv_pos),
        "delta_logprob_when_adv_negative_mean": _masked_scalar_mean(delta_logprob, adv_neg),
        **delta_magnitude_diag,
        "credit_mean": float(credit_stats["mean"]),
        "credit_std": float(credit_stats["std"]),
        "credit_min": float(credit_stats["min"]),
        "credit_max": float(credit_stats["max"]),
        "credit_positive_frac": float(credit_positive_frac),
        "credit_direction_agree_frac": float(direction_agree_frac),
        "credit_direction_wrong_frac": float(direction_wrong_frac),
        "credit_adv_positive_delta_positive_frac": float(pos_delta_positive_frac),
        "credit_adv_negative_delta_negative_frac": float(neg_delta_negative_frac),
        "credit_when_adv_positive_mean": _masked_scalar_mean(credit, adv_pos),
        "credit_when_adv_negative_mean": _masked_scalar_mean(credit, adv_neg),
        "adv_positive_frac": float(adv_pos_frac),
        "adv_negative_frac": float(adv_neg_frac),
        **raw_adv_diag,
        "old_logprob_mean": float(old_logprob.mean().detach().cpu().item()),
        "post_logprob_mean": float(post_logprob.mean().detach().cpu().item()),
        "old_logprob_parity_abs_mean": float(parity_abs_mean),
        "old_logprob_parity_abs_max": float(parity_abs_max),
        "old_logprob_parity_bad_frac": float(parity_bad_frac),
        "adv_mean": float(adv.mean().detach().cpu().item()),
        "adv_std": float(adv.std(unbiased=False).detach().cpu().item()),
        "actor_old_logprob_sec": float(old_logprob_sec),
        "actor_update_loop_sec": float(update_loop_sec),
        "actor_credit_diag_sec": float(credit_diag_sec),
        "actor_compile_enabled": float(1.0 if actor_loss_compile_enabled else 0.0),
        "actor_chunk_size": float(actor_chunk_preferred),
        "danger_imitation_loss": float(last_danger_loss),
        "danger_imitation_active_rate": float(last_danger_active),
        **grad_diag,
        **bw_shape_pre_stats,
        **bw_shape_post_stats,
        **bw_shape_delta_stats,
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


def _apply_exec_overrides(cfg: Any, stage_id: int, args: argparse.Namespace) -> None:
    if args.exec_accel_source is not None:
        cfg.exec_accel_source = str(args.exec_accel_source)
    if args.exec_sat_source is not None:
        cfg.exec_sat_source = str(args.exec_sat_source)
    if args.exec_bw_source is not None:
        cfg.exec_bw_source = str(args.exec_bw_source)
    stage_source = {
        0: str(cfg.exec_accel_source),
        1: str(cfg.exec_sat_source),
        2: str(cfg.exec_bw_source),
    }[int(stage_id)]
    if stage_source != "policy":
        raise ValueError(
            f"training stage {STAGE_NAME[int(stage_id)]} requires its exec source to be policy; got {stage_source!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic single-stage MC-critic + A_gae(V) training loop.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--stage", choices=sorted(STAGE_ID), default="sat")
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--rollout_env_steps", type=int, default=250)
    parser.add_argument("--reward_mode", default="positive_weighted_workload_level")
    parser.add_argument("--exec_accel_source", default=None)
    parser.add_argument("--exec_sat_source", default=None)
    parser.add_argument("--exec_bw_source", default=None)
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

    stage_id = int(STAGE_ID[str(args.stage)])
    stage_name = STAGE_NAME[stage_id]
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    _force_single_stage_config(cfg, stage_id=stage_id, reward_mode=str(args.reward_mode))
    report_torch_compile_cache(context=f"train_stage_mcgae:{stage_name}", device=device, cfg=cfg)
    if device.type == "cuda":
        _enable_strict_compile_global()
    _apply_exec_overrides(cfg, stage_id, args)
    _configure_accel_safety_for_training(cfg, stage_id=stage_id)
    cfg.actor_advantage_normalize_enabled = True
    cfg.stagewise_advantage_norm_enabled = True
    cfg.checkpoint_eval_enabled = False
    cfg.train_trace_enabled = False

    cold_critic_lr = float(
        args.cold_critic_lr
        if args.cold_critic_lr is not None
        else _stage_cfg_value(cfg, "stage_mcgae_cold_critic_lr", "sat_mcgae_cold_critic_lr", 1.0e-3)
    )
    cold_critic_epochs = int(
        args.cold_critic_epochs
        if args.cold_critic_epochs is not None
        else _stage_cfg_value(cfg, "stage_mcgae_cold_critic_epochs", "sat_mcgae_cold_critic_epochs", 20)
    )
    tracking_critic_lr = float(
        args.tracking_critic_lr
        if args.tracking_critic_lr is not None
        else _stage_cfg_value(cfg, "stage_mcgae_tracking_critic_lr", "sat_mcgae_tracking_critic_lr", 3.0e-4)
    )
    tracking_critic_epochs = int(
        args.tracking_critic_epochs
        if args.tracking_critic_epochs is not None
        else _stage_cfg_value(cfg, "stage_mcgae_tracking_critic_epochs", "sat_mcgae_tracking_critic_epochs", 5)
    )
    critic_minibatches = int(
        args.critic_minibatches
        if args.critic_minibatches is not None
        else _stage_cfg_value(cfg, "stage_mcgae_critic_minibatches", "sat_mcgae_critic_minibatches", 8)
    )
    critic_update_microbatch_size = int(
        args.critic_update_microbatch_size
        if args.critic_update_microbatch_size is not None
        else _stage_cfg_value(
            cfg,
            "stage_mcgae_critic_update_microbatch_size",
            "sat_mcgae_critic_update_microbatch_size",
            0,
        )
    )
    actor_lr = float(
        args.actor_lr
        if args.actor_lr is not None
        else _stage_cfg_value(cfg, "stage_mcgae_actor_lr", "sat_mcgae_actor_lr", 3.0e-4)
    )
    actor_epochs = int(
        args.actor_epochs
        if args.actor_epochs is not None
        else _stage_cfg_value(cfg, "stage_mcgae_actor_epochs", "sat_mcgae_actor_epochs", 5)
    )
    actor_minibatches = int(
        args.actor_minibatches
        if args.actor_minibatches is not None
        else _stage_cfg_value(cfg, "stage_mcgae_actor_minibatches", "sat_mcgae_actor_minibatches", 1)
    )

    if str(getattr(cfg, "critic_value_mode", "")).lower() == "global_linear":
        raise RuntimeError(
            "scripts/train_stage_mcgae.py needs a trainable critic; "
            "critic_value_mode=global_linear is not compatible with MC-GAE critic fitting. "
            "Use critic_value_mode=relational for this script."
        )
    if stage_id == 2 and bool(getattr(cfg, "bw_flow_proxy_aux_enabled", False)):
        raise RuntimeError(
            "train_stage_mcgae.py currently implements PPO+entropy for BW, but not the optional "
            "bw_flow_proxy_aux_loss. Disable bw_flow_proxy_aux_enabled or add that auxiliary term explicitly."
        )

    learner, _actor, _critic = _make_learner(cfg, device=device, stage_id=stage_id)
    _sync_danger_imitation_to_learner(learner, cfg)
    critic_params = [param for param in learner.critic.parameters() if param.requires_grad]
    if not critic_params:
        raise RuntimeError("MC-GAE training requires trainable critic parameters.")
    critic_optimizer = torch.optim.Adam(critic_params, lr=float(cold_critic_lr))
    stage_params = _stage_optimizer_params(learner.actor, stage_id)
    if not stage_params:
        raise RuntimeError(f"{stage_name} actor has no trainable parameters.")
    actor_optimizer = torch.optim.Adam(stage_params, lr=float(actor_lr))
    group = make_structured_driver_group(cfg, num_envs=int(args.num_envs), backend="sync", mode="train")

    metrics: list[dict[str, float]] = []
    try:
        learner.bind_native_runtime_contract(group)
        print(
            f"{stage_name.upper()} MC-GAE train | "
            f"envs={int(args.num_envs)} rollout={int(args.rollout_env_steps)} updates={int(args.updates)} "
            f"reward={cfg.reward_mode} critic={cold_critic_epochs}/{tracking_critic_epochs} "
            f"actor_epochs={actor_epochs} exec=({cfg.exec_accel_source},{cfg.exec_sat_source},{cfg.exec_bw_source}) "
            f"safety=(avoidance={bool(getattr(cfg, 'avoidance_enabled', False))}, "
            f"native_shield={bool(getattr(cfg, 'safety_shield_enabled', False)) and str(getattr(cfg, 'safety_shield_solver', '') or '').strip().upper() == 'NATIVE_CUDA'}, "
            f"danger={bool(getattr(learner, 'danger_imitation_enabled', False))}, "
            f"danger_mode={getattr(cfg, 'danger_imitation_trigger_mode', '')})",
            flush=True,
        )
        for update in range(max(int(args.updates), 0)):
            if bool(args.diagnose_critic_timing) and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            update_t0 = time.perf_counter()
            rollout_seed = int(args.seed) + update * 100_000

            t0 = time.perf_counter()
            _buffer, views, _stage_idx, mc_target, reward_stats = _collect_stage_rollout(
                learner,
                group,
                stage_id=stage_id,
                rollout_env_steps=int(args.rollout_env_steps),
                device=device,
                seed=rollout_seed,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            collect_sec = time.perf_counter() - t0
            stage_batch = views.training_view.stage_batches[stage_id]

            critic_lr = float(cold_critic_lr if update == 0 else tracking_critic_lr)
            critic_epochs = int(cold_critic_epochs if update == 0 else tracking_critic_epochs)
            t1 = time.perf_counter()
            critic_stats, stage_values_after_critic, critic_trace_rows = _train_stage_critic_on_stage(
                learner,
                stage_id=stage_id,
                stage_batch=stage_batch,
                target=mc_target,
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
            _returns_gae, stage_adv, stage_values = _stage_gae_from_mc_targets(
                learner,
                stage_id=stage_id,
                stage_batch=stage_batch,
                mc_target=mc_target,
                device=device,
                stage_values=stage_values_after_critic,
            )
            stage_adv_norm = _normalize_stage_advantage(
                stage_adv,
                enabled=bool(getattr(cfg, "actor_advantage_normalize_enabled", True)),
            )
            actor_stats = _stage_actor_update_full_stage(
                learner,
                stage_id=stage_id,
                stage_batch=stage_batch,
                stage_advantages=stage_adv_norm,
                stage_raw_advantages=stage_adv,
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
                "stage_id": float(stage_id),
                "collect_sec": float(collect_sec),
                "critic_sec": float(critic_sec),
                "actor_sec": float(actor_sec),
                "cuda_empty_cache_sec": float(empty_cache_sec),
                "iteration_sec": float(time.perf_counter() - update_t0),
                "critic_lr": float(critic_lr),
                "critic_epochs": float(critic_epochs),
                "samples": float(stage_batch.num_samples),
                f"{stage_name}_value_mean": float(stage_values.mean().detach().cpu().item()),
                f"{stage_name}_value_std": float(stage_values.std(unbiased=False).detach().cpu().item()),
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
                            "stage_id": float(stage_id),
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
                f"mc={row['stage_mc_return_mean']:.3f} critic_ev={row['critic_ev_after']:.3f} "
                f"adv_std={row['raw_adv_std']:.3f} kl={row[f'approx_kl_{stage_name}']:.5f} "
                f"ent={row[f'entropy_{stage_name}']:.3f} time={row['iteration_sec']:.1f}s",
                flush=True,
            )
            if int(args.save_every) > 0 and (update + 1) % int(args.save_every) == 0:
                torch.save(
                    {
                        "update": update + 1,
                        "stage": stage_name,
                        "actor": learner.actor.state_dict(),
                        "critic": learner.critic.state_dict(),
                        "config": vars(cfg),
                    },
                    run_dir / f"checkpoint_u{update + 1:04d}.pt",
                )

        torch.save(
            {
                "update": int(args.updates),
                "stage": stage_name,
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
