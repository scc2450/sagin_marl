from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.rl.stage_mcgae import (
    STAGE_ID,
    clone_dataclass_tensors as _clone_dataclass_tensors,
    collect_one_rollout as _collect_one_rollout,
    force_single_stage_config as _force_single_stage_config,
    make_learner as _make_learner,
    set_seed as _set_seed,
)
from sagin_marl.rl.structured_mappo import _collate_dataclass, _index_dataclass
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver_group
from scripts.diagnostics.audit.audit_stage_critic_only_fit import (
    _collect_stage_bank,
    _heldout_vpi_ceiling_probe,
    _train_critic_only,
)


def _ev(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    finite = np.isfinite(pred) & np.isfinite(target)
    pred = pred[finite]
    target = target[finite]
    if int(pred.size) <= 1:
        return 0.0
    var = float(np.var(target))
    if var <= 1.0e-12:
        return 0.0
    return float(1.0 - np.var(target - pred) / var)


def _corr(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    finite = np.isfinite(pred) & np.isfinite(target)
    pred = pred[finite]
    target = target[finite]
    if int(pred.size) <= 1 or float(np.std(pred)) <= 1.0e-12 or float(np.std(target)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(pred, target)[0, 1])


def _summ(x: np.ndarray) -> dict[str, float]:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if int(arr.size) <= 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _flat_tensor(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.bool or not torch.is_floating_point(t):
        t = t.to(torch.float32)
    return t.reshape(int(t.shape[0]), -1)


def _world_global_features(world_state: Any) -> torch.Tensor:
    return _flat_tensor(world_state.global_scalars)


def _world_raw_features(world_state: Any) -> tuple[torch.Tensor, list[tuple[str, int]]]:
    parts: list[torch.Tensor] = []
    part_dims: list[tuple[str, int]] = []
    for item in fields(world_state):
        value = getattr(world_state, item.name)
        if torch.is_tensor(value):
            flat = _flat_tensor(value)
            parts.append(flat)
            part_dims.append((item.name, int(flat.shape[1])))
    return torch.cat(parts, dim=1), part_dims


def _ridge_predict(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    *,
    ridge: float,
) -> np.ndarray:
    x_train = x_train.detach().to(device="cpu", dtype=torch.float64)
    x_eval = x_eval.detach().to(device="cpu", dtype=torch.float64)
    y_train = y_train.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    finite = torch.isfinite(x_train).all(dim=1) & torch.isfinite(y_train)
    idx = torch.nonzero(finite, as_tuple=False).flatten()
    x_train = x_train.index_select(0, idx)
    y_train = y_train.index_select(0, idx)
    mu = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-6)
    xs = (x_train - mu) / std
    xe = (x_eval - mu) / std
    design = torch.cat([torch.ones((int(xs.shape[0]), 1), dtype=torch.float64), xs], dim=1)
    xtx = design.transpose(0, 1).matmul(design)
    reg = torch.eye(int(xtx.shape[0]), dtype=torch.float64) * max(float(ridge), 0.0)
    reg[0, 0] = 0.0
    xty = design.transpose(0, 1).matmul(y_train)
    try:
        coef = torch.linalg.solve(xtx + reg, xty)
    except RuntimeError:
        coef = torch.linalg.lstsq(xtx + reg, xty).solution
    eval_design = torch.cat([torch.ones((int(xe.shape[0]), 1), dtype=torch.float64), xe], dim=1)
    return eval_design.matmul(coef).numpy()


def _critic_values(learner: Any, stage_id: int, world_state: Any, *, batch_size: int = 1024) -> np.ndarray:
    values: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(world_state.global_scalars.shape[0]), int(batch_size)):
            end = min(start + int(batch_size), int(world_state.global_scalars.shape[0]))
            idx = torch.arange(start, end, dtype=torch.long, device=world_state.global_scalars.device)
            values.append(
                learner._stage_value_eval_from_batch(stage_id, _index_dataclass(world_state, idx))
                .detach()
                .cpu()
            )
    return torch.cat(values, dim=0).numpy()


def _system_context(learner: Any, world_state: Any, *, batch_size: int = 1024) -> torch.Tensor:
    contexts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(world_state.global_scalars.shape[0]), int(batch_size)):
            end = min(start + int(batch_size), int(world_state.global_scalars.shape[0]))
            idx = torch.arange(start, end, dtype=torch.long, device=world_state.global_scalars.device)
            contexts.append(learner.critic._system_context(_index_dataclass(world_state, idx)).detach().cpu())
    return torch.cat(contexts, dim=0)


def _system_context_layers(learner: Any, world_state: Any, *, batch_size: int = 1024) -> dict[str, torch.Tensor]:
    critic = learner.critic
    outputs: dict[str, list[torch.Tensor]] = {
        "global_embed": [],
        "system_initial": [],
    }
    for idx in range(len(getattr(critic, "relational_blocks", []))):
        outputs[f"system_after_block{idx + 1}"] = []
    with torch.no_grad():
        for start in range(0, int(world_state.global_scalars.shape[0]), int(batch_size)):
            end = min(start + int(batch_size), int(world_state.global_scalars.shape[0]))
            row_idx = torch.arange(start, end, dtype=torch.long, device=world_state.global_scalars.device)
            batch = _index_dataclass(world_state, row_idx)
            global_embed = critic.global_scalar_encoder(critic.global_input_norm(batch.global_scalars))
            outputs["global_embed"].append(global_embed.detach().cpu())
            if critic.value_mode == "global_only":
                continue
            gu_mask = batch.gu_mask.to(dtype=torch.bool)
            sat_mask = batch.sat_mask.to(dtype=torch.bool)
            uav_gu_mask = batch.uav_gu_mask.to(dtype=torch.bool)
            uav_sat_mask = batch.uav_sat_mask.to(dtype=torch.bool)
            uav_uav_mask = batch.uav_uav_mask.to(dtype=torch.bool)
            gu_tokens = critic.gu_encoder(critic.gu_input_norm(batch.gu_nodes))
            uav_tokens = critic.uav_encoder(critic.uav_input_norm(batch.uav_nodes))
            sat_tokens = critic.sat_encoder(critic.sat_input_norm(batch.sat_nodes))
            ug_edges = critic.ug_edge_encoder(critic.ug_edge_input_norm(batch.uav_gu_edges))
            us_edges = critic.us_edge_encoder(critic.us_edge_input_norm(batch.uav_sat_edges))
            uu_edges = critic.uu_edge_encoder(critic.uu_edge_input_norm(batch.uav_uav_edges))
            uav_local_summary, sat_local_summary = critic._local_summaries(batch)
            uav_local = critic.uav_local_summary_encoder(uav_local_summary)
            sat_local = critic.sat_local_summary_encoder(sat_local_summary)
            system_token = critic.system_token.view(1, -1).expand(int(batch.uav_nodes.shape[0]), -1) + global_embed
            outputs["system_initial"].append(system_token.detach().cpu())
            for block_idx, block in enumerate(critic.relational_blocks):
                gu_tokens, uav_tokens, sat_tokens, system_token = block(
                    gu_tokens,
                    uav_tokens,
                    sat_tokens,
                    ug_edges,
                    us_edges,
                    uu_edges,
                    gu_mask,
                    sat_mask,
                    uav_gu_mask,
                    uav_sat_mask,
                    uav_uav_mask,
                    uav_local,
                    sat_local,
                    system_token,
                    global_embed,
                )
                outputs[f"system_after_block{block_idx + 1}"].append(system_token.detach().cpu())
    return {name: torch.cat(chunks, dim=0) for name, chunks in outputs.items() if chunks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(STAGE_ID), default="sat")
    parser.add_argument("--config", required=True)
    parser.add_argument("--reward_mode", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--rollout_env_steps", type=int, default=250)
    parser.add_argument("--train_rollouts", type=int, default=8)
    parser.add_argument("--critic_epochs", type=int, default=30)
    parser.add_argument("--critic_minibatches", type=int, default=8)
    parser.add_argument("--critic_lr", type=float, default=None)
    parser.add_argument("--critic_message_layers", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--vpi_rows", type=int, default=64)
    parser.add_argument("--train_vpi_rows", type=int, default=0)
    parser.add_argument("--vpi_policy_action_samples", type=int, default=4)
    parser.add_argument("--vpi_continuations", type=int, default=2)
    parser.add_argument("--vpi_follow", choices=["deterministic", "stochastic"], default="stochastic")
    parser.add_argument("--min_horizon", type=int, default=20)
    parser.add_argument("--branch_horizon_cap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=45210)
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--global_ridge", type=float, default=1.0e-3)
    parser.add_argument("--system_ridge", type=float, default=1.0e-3)
    parser.add_argument("--raw_ridge", type=float, default=1.0e-2)
    parser.add_argument("--include_raw_probe", action="store_true")
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
    if args.critic_message_layers is not None:
        cfg.critic_message_layers = int(args.critic_message_layers)

    learner, _actor, _critic = _make_learner(cfg, device=device, stage_id=stage_id)
    group = make_structured_driver_group(cfg, num_envs=int(args.num_envs), backend="sync", mode="train")
    started = time.time()
    try:
        learner.bind_native_runtime_contract(group)
        reset_many = getattr(group, "reset_many", None)
        if callable(reset_many):
            reset_many([int(args.seed) + env for env in range(int(args.num_envs))])
        train_probe_buffer, train_probe_views, train_probe_returns_all = _collect_one_rollout(
            learner,
            group,
            rollout_env_steps=int(args.rollout_env_steps),
            device=device,
            target="mc",
        )
        train_probe_stage = train_probe_views.training_view.stage_batches.get(int(stage_id))
        if train_probe_stage is None or int(train_probe_stage.num_samples) <= 0:
            raise RuntimeError("no stage samples collected for train V_hat probe rollout.")
        train_probe_idx = torch.as_tensor(
            np.asarray(train_probe_stage.transition_indices, dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        train_probe_target = train_probe_returns_all.index_select(0, train_probe_idx).detach()
        train_summaries = [
            {
                "rollout": 0.0,
                "samples": float(train_probe_target.numel()),
                "return_mean": float(train_probe_target.mean().detach().cpu().item()),
                "return_std": float(train_probe_target.std(unbiased=False).detach().cpu().item()),
                "probe_rollout": 1.0,
            }
        ]
        # Native rollout views can alias reusable runtime storage.  Own the
        # first rollout before collecting any extra train rollouts; otherwise
        # later rollout collection can mutate these inputs while the target
        # tensor below stays fixed.
        train_world_parts = [_clone_dataclass_tensors(train_probe_stage.world_batch, device=device)]
        train_target_parts = [train_probe_target]
        remaining_train_rollouts = max(int(args.train_rollouts) - 1, 0)
        if remaining_train_rollouts > 0:
            extra_train_world, extra_train_target, extra_train_summaries = _collect_stage_bank(
                learner,
                group,
                stage_id=stage_id,
                rollout_env_steps=int(args.rollout_env_steps),
                rollouts=int(remaining_train_rollouts),
                seed_base=int(args.seed) + 10_000,
                device=device,
                target="mc",
            )
            train_world_parts.append(extra_train_world)
            train_target_parts.append(extra_train_target)
            for item in extra_train_summaries:
                item = dict(item)
                item["rollout"] = float(item.get("rollout", 0.0)) + 1.0
                item["probe_rollout"] = 0.0
                train_summaries.append(item)
        train_world = _collate_dataclass(train_world_parts, device)
        train_target = torch.cat(train_target_parts, dim=0).to(device=device, dtype=torch.float32)

        # Kept for historical comparability: train_world/train_target above are the same bank that
        # trains the critic, and train_probe_views lets us estimate V_pi on a subset of that bank.
        _unused_legacy_train_bank = _collect_stage_bank
        if False:
            train_world, train_target, train_summaries = _collect_stage_bank(
            learner,
            group,
            stage_id=stage_id,
            rollout_env_steps=int(args.rollout_env_steps),
            rollouts=int(args.train_rollouts),
            seed_base=int(args.seed),
            device=device,
            target="mc",
            )
        reset_many = getattr(group, "reset_many", None)
        if callable(reset_many):
            reset_many([int(args.seed) + 9_000_000 + env for env in range(int(args.num_envs))])
        heldout_buffer, heldout_views, heldout_returns_all = _collect_one_rollout(
            learner,
            group,
            rollout_env_steps=int(args.rollout_env_steps),
            device=device,
            target="mc",
        )
        heldout_stage = heldout_views.training_view.stage_batches[int(stage_id)]
        heldout_idx = torch.as_tensor(
            np.asarray(heldout_stage.transition_indices, dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        heldout_world = heldout_stage.world_batch
        heldout_target = heldout_returns_all.index_select(0, heldout_idx).detach()

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
        )

        vpi = _heldout_vpi_ceiling_probe(
            learner,
            cfg=cfg,
            stage_id=stage_id,
            num_envs=int(args.num_envs),
            heldout_views=heldout_views,
            reward_mode=str(cfg.reward_mode),
            sample_rows=int(args.vpi_rows),
            policy_action_samples=int(args.vpi_policy_action_samples),
            continuations=int(args.vpi_continuations),
            min_horizon=int(args.min_horizon),
            branch_horizon_cap=int(args.branch_horizon_cap),
            seed=int(args.seed),
            device=device,
            initial_critic_state=None,
            follow_deterministic=str(args.vpi_follow) == "deterministic",
        )
        rows = list(vpi.get("rows", []))
        selected_np = np.asarray([int(row["stage_sample"]) for row in rows], dtype=np.int64)
        selected_t = torch.as_tensor(selected_np, dtype=torch.long, device=device)
        benchmark_world = _index_dataclass(heldout_world, selected_t)
        vhat = np.asarray([float(row["vhat"]) for row in rows], dtype=np.float64)
        vhat_se = np.asarray([float(row["vhat_se"]) for row in rows], dtype=np.float64)
        actual_pred = np.asarray([float(row["value_final"]) for row in rows], dtype=np.float64)

        pred_global = _ridge_predict(
            _world_global_features(train_world),
            train_target,
            _world_global_features(benchmark_world),
            ridge=float(args.global_ridge),
        )
        train_ctx = _system_context(learner, train_world)
        bench_ctx = _system_context(learner, benchmark_world)
        pred_system = _ridge_predict(train_ctx, train_target, bench_ctx, ridge=float(args.system_ridge))
        train_layers = _system_context_layers(learner, train_world)
        bench_layers = _system_context_layers(learner, benchmark_world)
        layer_predictions = {
            name: _ridge_predict(train_layers[name], train_target, bench_layers[name], ridge=float(args.system_ridge))
            for name in sorted(train_layers)
            if name in bench_layers
        }

        raw_part_dims: list[tuple[str, int]] = []
        pred_raw = None
        if bool(args.include_raw_probe):
            train_raw, raw_part_dims = _world_raw_features(train_world)
            bench_raw, _ = _world_raw_features(benchmark_world)
            pred_raw = _ridge_predict(train_raw, train_target, bench_raw, ridge=float(args.raw_ridge))

        heldout_actual = _critic_values(learner, stage_id, heldout_world)
        heldout_global = _ridge_predict(
            _world_global_features(train_world),
            train_target,
            _world_global_features(heldout_world),
            ridge=float(args.global_ridge),
        )
        heldout_system = _ridge_predict(train_ctx, train_target, _system_context(learner, heldout_world), ridge=float(args.system_ridge))

        predictions = {
            "actual_critic": actual_pred,
            "global_linear": pred_global,
            "system_context_linear": pred_system,
        }
        for name, pred in layer_predictions.items():
            predictions[f"{name}_linear"] = pred
        if pred_raw is not None:
            predictions["raw_all_linear"] = pred_raw

        benchmark_metrics = {
            name: {
                "ev": _ev(pred, vhat),
                "corr": _corr(pred, vhat),
                "mae": float(np.mean(np.abs(np.asarray(pred) - vhat))),
                "residual": _summ(vhat - np.asarray(pred)),
            }
            for name, pred in predictions.items()
        }

        train_benchmark = None
        train_vpi_rows = int(args.train_vpi_rows if int(args.train_vpi_rows) > 0 else args.vpi_rows)
        if train_vpi_rows > 0:
            train_vpi = _heldout_vpi_ceiling_probe(
                learner,
                cfg=cfg,
                stage_id=stage_id,
                num_envs=int(args.num_envs),
                heldout_views=train_probe_views,
                reward_mode=str(cfg.reward_mode),
                sample_rows=int(train_vpi_rows),
                policy_action_samples=int(args.vpi_policy_action_samples),
                continuations=int(args.vpi_continuations),
                min_horizon=int(args.min_horizon),
                branch_horizon_cap=int(args.branch_horizon_cap),
                seed=int(args.seed) + 7_000_000,
                device=device,
                initial_critic_state=None,
                follow_deterministic=str(args.vpi_follow) == "deterministic",
            )
            train_rows = list(train_vpi.get("rows", []))
            train_selected_np = np.asarray([int(row["stage_sample"]) for row in train_rows], dtype=np.int64)
            train_selected_t = torch.as_tensor(train_selected_np, dtype=torch.long, device=device)
            train_bench_world = _index_dataclass(train_probe_stage.world_batch, train_selected_t)
            train_vhat = np.asarray([float(row["vhat"]) for row in train_rows], dtype=np.float64)
            train_actual = np.asarray([float(row["value_final"]) for row in train_rows], dtype=np.float64)
            train_pred_global = _ridge_predict(
                _world_global_features(train_world),
                train_target,
                _world_global_features(train_bench_world),
                ridge=float(args.global_ridge),
            )
            train_bench_ctx = _system_context(learner, train_bench_world)
            train_pred_system = _ridge_predict(train_ctx, train_target, train_bench_ctx, ridge=float(args.system_ridge))
            train_benchmark = {
                "vpi_rows": int(len(train_rows)),
                "vhat": _summ(train_vhat),
                "metrics": {
                    "actual_critic": {
                        "ev": _ev(train_actual, train_vhat),
                        "corr": _corr(train_actual, train_vhat),
                        "mae": float(np.mean(np.abs(train_actual - train_vhat))),
                        "residual": _summ(train_vhat - train_actual),
                    },
                    "global_linear": {
                        "ev": _ev(train_pred_global, train_vhat),
                        "corr": _corr(train_pred_global, train_vhat),
                        "mae": float(np.mean(np.abs(train_pred_global - train_vhat))),
                        "residual": _summ(train_vhat - train_pred_global),
                    },
                    "system_context_linear": {
                        "ev": _ev(train_pred_system, train_vhat),
                        "corr": _corr(train_pred_system, train_vhat),
                        "mae": float(np.mean(np.abs(train_pred_system - train_vhat))),
                        "residual": _summ(train_vhat - train_pred_system),
                    },
                },
                "rows": [
                    {
                        **row,
                        "pred_actual_critic": float(train_actual[idx]),
                        "pred_global_linear": float(train_pred_global[idx]),
                        "pred_system_context_linear": float(train_pred_system[idx]),
                    }
                    for idx, row in enumerate(train_rows)
                ],
            }
        heldout_single_metrics = {
            "actual_critic": {
                "ev": _ev(heldout_actual, heldout_target.detach().cpu().numpy()),
                "corr": _corr(heldout_actual, heldout_target.detach().cpu().numpy()),
            },
            "global_linear": {
                "ev": _ev(heldout_global, heldout_target.detach().cpu().numpy()),
                "corr": _corr(heldout_global, heldout_target.detach().cpu().numpy()),
            },
            "system_context_linear": {
                "ev": _ev(heldout_system, heldout_target.detach().cpu().numpy()),
                "corr": _corr(heldout_system, heldout_target.detach().cpu().numpy()),
            },
        }

        out = {
            "stage": str(args.stage),
            "stage_id": int(stage_id),
            "reward_mode": str(cfg.reward_mode),
            "config": str(args.config),
            "seed": int(args.seed),
            "num_envs": int(args.num_envs),
            "rollout_env_steps": int(args.rollout_env_steps),
            "train_rollouts": int(args.train_rollouts),
            "critic_epochs": int(args.critic_epochs),
            "critic_minibatches": int(args.critic_minibatches),
            "critic_lr": float(cfg.critic_lr),
            "critic_message_layers": int(getattr(cfg, "critic_message_layers", 0)),
            "train_samples": int(train_target.numel()),
            "heldout_samples": int(heldout_target.numel()),
            "train_rollout_summaries": train_summaries,
            "critic_history": history,
            "benchmark": {
                "vpi_rows": int(len(rows)),
                "vpi_policy_action_samples": int(vpi.get("policy_action_samples", 0)),
                "vpi_continuations": int(vpi.get("continuations", 0)),
                "vpi_follow": str(vpi.get("follow", "")),
                "vhat": _summ(vhat),
                "vhat_se": _summ(vhat_se),
                "metrics": benchmark_metrics,
                "rows": [
                    {
                        **row,
                        **{f"pred_{name}": float(predictions[name][idx]) for name in predictions},
                    }
                    for idx, row in enumerate(rows)
                ],
            },
            "train_benchmark": train_benchmark,
            "heldout_single_return_metrics": heldout_single_metrics,
            "feature_dims": {
                "global_scalars": int(_world_global_features(train_world).shape[1]),
                "system_context": int(train_ctx.shape[1]),
                **{name: int(value.shape[1]) for name, value in train_layers.items()},
                "raw_all": int(sum(dim for _, dim in raw_part_dims)) if raw_part_dims else None,
            },
            "raw_feature_parts": raw_part_dims,
            "elapsed_sec": max(time.time() - started, 0.0),
        }
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"benchmark": out["benchmark"]["metrics"], "heldout_single": heldout_single_metrics}, ensure_ascii=False, indent=2))
        print(f"wrote {output_path}")
    finally:
        close_structured_env_group(group)


if __name__ == "__main__":
    main()
