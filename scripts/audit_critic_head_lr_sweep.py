from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sagin_marl.env.config import load_config
from sagin_marl.rl.stage_mcgae import (
    STAGE_ID,
    collect_one_rollout as _collect_one_rollout,
    force_single_stage_config as _force_single_stage_config,
    make_learner as _make_learner,
    set_seed as _set_seed,
)
from sagin_marl.rl.structured_mappo import _collate_dataclass, _index_dataclass
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver_group
from scripts.audit_fixed_critic_benchmark import (
    _corr,
    _critic_values,
    _ev,
    _ridge_predict,
    _summ,
    _system_context,
    _world_global_features,
)
from scripts.diagnostics.audit.audit_stage_critic_only_fit import (
    _collect_stage_bank,
    _heldout_vpi_ceiling_probe,
    _train_critic_only,
)


def _parse_float_list(text: str) -> list[float]:
    out: list[float] = []
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    return out


def _parse_variant_specs(text: str) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for raw in str(text).replace("\n", ",").split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts = [part.strip() for part in raw.split(":")]
        if len(parts) < 4:
            raise ValueError(
                "variant spec must be optimizer:target:lr:weight_decay[:input_norm], "
                f"got {raw!r}"
            )
        variants.append(
            {
                "optimizer": parts[0].lower(),
                "target": parts[1].lower(),
                "lr": float(parts[2]),
                "weight_decay": float(parts[3]),
                "input_norm": bool(int(parts[4])) if len(parts) >= 5 and parts[4] != "" else False,
                "name": raw,
            }
        )
    return variants


def _make_head(kind: str, dim: int, hidden: int) -> nn.Module:
    kind = str(kind).strip().lower()
    if kind == "linear":
        return nn.Linear(dim, 1)
    if kind == "relu":
        return nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    if kind == "silu":
        return nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
    if kind == "tanh":
        return nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
    if kind == "linear_relu_residual":
        class ResidualHead(nn.Module):
            def __init__(self, d: int, h: int) -> None:
                super().__init__()
                self.linear = nn.Linear(d, 1)
                self.residual = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 1))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x) + self.residual(x)

        return ResidualHead(dim, hidden)
    raise ValueError(f"unknown head kind: {kind}")


def _eval_pred(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    return {
        "ev": _ev(pred, target),
        "corr": _corr(pred, target),
        "mse": float(np.mean((pred - target) ** 2)),
        "mae": float(np.mean(np.abs(pred - target))),
        "pred": _summ(pred),
        "target": _summ(target),
        "residual": _summ(target - pred),
    }


def _head_predict(head: nn.Module, x: torch.Tensor, *, batch_size: int = 4096) -> np.ndarray:
    preds: list[torch.Tensor] = []
    head.eval()
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), int(batch_size)):
            preds.append(head(x[start : start + int(batch_size)]).reshape(-1).detach().cpu())
    return torch.cat(preds, dim=0).numpy()


def _train_head_only(
    *,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    heldout_x: torch.Tensor,
    heldout_y: torch.Tensor,
    bench_x: torch.Tensor,
    bench_y: np.ndarray,
    kind: str,
    hidden: int,
    lr: float,
    epochs: int,
    minibatches: int,
    max_grad_norm: float,
    seed: int,
) -> dict[str, Any]:
    _set_seed(int(seed))
    head = _make_head(kind, int(train_x.shape[1]), int(hidden)).to(device=train_x.device)
    opt = torch.optim.Adam(head.parameters(), lr=float(lr))
    n = int(train_y.numel())
    mb_size = max(1, int(np.ceil(n / max(int(minibatches), 1))))
    history: list[dict[str, float]] = []
    for epoch in range(max(int(epochs), 0) + 1):
        if epoch == 0 or epoch == int(epochs):
            train_pred = _head_predict(head, train_x)
            heldout_pred = _head_predict(head, heldout_x)
            history.append(
                {
                    "epoch": float(epoch),
                    "train_ev": float(_ev(train_pred, train_y.detach().cpu().numpy())),
                    "heldout_ev": float(_ev(heldout_pred, heldout_y.detach().cpu().numpy())),
                    "train_mse": float(np.mean((train_pred - train_y.detach().cpu().numpy()) ** 2)),
                    "heldout_mse": float(np.mean((heldout_pred - heldout_y.detach().cpu().numpy()) ** 2)),
                }
            )
        if epoch >= int(epochs):
            break
        order = torch.randperm(n, device=train_x.device)
        head.train()
        for start in range(0, n, mb_size):
            idx = order[start : start + mb_size]
            pred = head(train_x.index_select(0, idx)).reshape(-1)
            loss = F.mse_loss(pred, train_y.index_select(0, idx).reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), float(max_grad_norm))
            opt.step()
    train_pred = _head_predict(head, train_x)
    heldout_pred = _head_predict(head, heldout_x)
    bench_pred = _head_predict(head, bench_x)
    return {
        "kind": str(kind),
        "hidden": int(hidden),
        "lr": float(lr),
        "epochs": int(epochs),
        "history": history,
        "train_mc": _eval_pred(train_pred, train_y.detach().cpu().numpy()),
        "heldout_mc": _eval_pred(heldout_pred, heldout_y.detach().cpu().numpy()),
        "benchmark_vhat": _eval_pred(bench_pred, bench_y),
    }


def _train_full_variant(
    *,
    cfg: Any,
    stage_id: int,
    train_world: Any,
    train_target: torch.Tensor,
    heldout_world: Any,
    heldout_target: torch.Tensor,
    benchmark_world: Any,
    benchmark_vhat: np.ndarray,
    train_benchmark_world: Any | None,
    train_benchmark_vhat: np.ndarray | None,
    device: torch.device,
    variant: dict[str, Any],
    epochs: int,
    minibatches: int,
    eval_every: int,
    max_grad_norm: float,
    seed: int,
) -> dict[str, Any]:
    _set_seed(int(seed))
    cfg_variant = copy.deepcopy(cfg)
    if bool(variant.get("input_norm", False)):
        setattr(cfg_variant, "structured_critic_input_norm_enabled", True)
    learner, _actor, _critic = _make_learner(cfg_variant, device=device, stage_id=stage_id)
    params = [p for p in learner.critic.parameters() if p.requires_grad]
    opt_name = str(variant.get("optimizer", "adam")).lower()
    lr = float(variant.get("lr", cfg_variant.critic_lr))
    weight_decay = float(variant.get("weight_decay", 0.0))
    if opt_name == "adam":
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "adamw":
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"unsupported optimizer: {opt_name}")

    target_mode = str(variant.get("target", "raw")).lower()
    target_mean = torch.zeros((), dtype=torch.float32, device=device)
    target_std = torch.ones((), dtype=torch.float32, device=device)
    train_loss_target = train_target
    if target_mode in {"norm", "normalized", "standardized", "z"}:
        target_mean = train_target.mean().detach()
        target_std = train_target.std(unbiased=False).detach().clamp_min(1.0e-6)
        train_loss_target = (train_target - target_mean) / target_std
        target_mode = "norm"
    elif target_mode == "raw":
        pass
    else:
        raise ValueError(f"unsupported target mode: {target_mode}")

    def predict_value(world: Any, *, batch_size: int = 1024) -> np.ndarray:
        raw_chunks: list[torch.Tensor] = []
        n_rows = int(world.global_scalars.shape[0])
        with torch.no_grad():
            for start in range(0, n_rows, int(batch_size)):
                end = min(start + int(batch_size), n_rows)
                idx = torch.arange(start, end, dtype=torch.long, device=device)
                raw = learner._stage_value_eval_from_batch(stage_id, _index_dataclass(world, idx))
                if target_mode == "norm":
                    raw = raw * target_std + target_mean
                raw_chunks.append(raw.detach().cpu())
        return torch.cat(raw_chunks, dim=0).numpy()

    n = int(train_target.numel())
    mb_size = max(1, int(np.ceil(n / max(int(minibatches), 1))))
    history: list[dict[str, float]] = []
    for epoch in range(max(int(epochs), 0) + 1):
        if epoch == 0 or epoch == int(epochs) or (int(eval_every) > 0 and epoch % int(eval_every) == 0):
            train_pred = predict_value(train_world)
            heldout_pred = predict_value(heldout_world)
            history.append(
                {
                    "epoch": float(epoch),
                    "train_ev": float(_ev(train_pred, train_target.detach().cpu().numpy())),
                    "heldout_ev": float(_ev(heldout_pred, heldout_target.detach().cpu().numpy())),
                    "train_mse": float(np.mean((train_pred - train_target.detach().cpu().numpy()) ** 2)),
                    "heldout_mse": float(np.mean((heldout_pred - heldout_target.detach().cpu().numpy()) ** 2)),
                }
            )
        if epoch >= int(epochs):
            break
        order = torch.randperm(n, device=device)
        for start in range(0, n, mb_size):
            idx = order[start : start + mb_size]
            pred = learner._stage_value_eval_from_batch(stage_id, _index_dataclass(train_world, idx))
            loss = F.mse_loss(pred, train_loss_target.index_select(0, idx))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(max_grad_norm))
            opt.step()

    pred_vhat = predict_value(benchmark_world)
    pred_train_vhat = None
    if train_benchmark_world is not None and train_benchmark_vhat is not None:
        pred_train_vhat = predict_value(train_benchmark_world)
    pred_heldout = predict_value(heldout_world)
    pred_train = predict_value(train_world)
    result = {
        "name": str(variant.get("name", "")),
        "optimizer": opt_name,
        "target": target_mode,
        "lr": lr,
        "weight_decay": weight_decay,
        "input_norm": bool(variant.get("input_norm", False)),
        "target_mean": float(target_mean.detach().cpu().item()),
        "target_std": float(target_std.detach().cpu().item()),
        "history": history,
        "benchmark_vhat": _eval_pred(pred_vhat, benchmark_vhat),
        "heldout_mc": _eval_pred(pred_heldout, heldout_target.detach().cpu().numpy()),
        "train_mc": _eval_pred(pred_train, train_target.detach().cpu().numpy()),
    }
    if pred_train_vhat is not None:
        result["train_benchmark_vhat"] = _eval_pred(pred_train_vhat, train_benchmark_vhat)
    return result


def _vhat_rows_to_world(stage_world: Any, rows: list[dict[str, Any]], *, device: torch.device) -> tuple[Any, np.ndarray]:
    selected_np = np.asarray([int(row["stage_sample"]) for row in rows], dtype=np.int64)
    selected_t = torch.as_tensor(selected_np, dtype=torch.long, device=device)
    return _index_dataclass(stage_world, selected_t), np.asarray([float(row["vhat"]) for row in rows], dtype=np.float64)


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
    parser.add_argument("--base_critic_epochs", type=int, default=30)
    parser.add_argument("--critic_minibatches", type=int, default=8)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--head_epochs", type=int, default=120)
    parser.add_argument("--head_lrs", default="3e-4,1e-3")
    parser.add_argument("--head_hidden", type=int, default=256)
    parser.add_argument("--head_kinds", default="linear,relu,silu,linear_relu_residual")
    parser.add_argument("--full_lrs", default="1e-4,3e-4,1e-3,3e-3")
    parser.add_argument("--full_critic_epochs", type=int, default=30)
    parser.add_argument("--optimizer_variants", default="")
    parser.add_argument("--skip_head_only", action="store_true")
    parser.add_argument("--skip_full_lr_sweep", action="store_true")
    parser.add_argument("--critic_message_layers", type=int, default=None)
    parser.add_argument("--vpi_rows", type=int, default=32)
    parser.add_argument("--vpi_policy_action_samples", type=int, default=4)
    parser.add_argument("--vpi_continuations", type=int, default=2)
    parser.add_argument("--vpi_follow", choices=["deterministic", "stochastic"], default="stochastic")
    parser.add_argument("--min_horizon", type=int, default=20)
    parser.add_argument("--branch_horizon_cap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=45210)
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--global_ridge", type=float, default=1.0e-3)
    parser.add_argument("--system_ridge", type=float, default=1.0e-3)
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
    if args.critic_message_layers is not None:
        cfg.critic_message_layers = int(args.critic_message_layers)

    started = time.time()
    data_learner, _actor, _critic = _make_learner(cfg, device=device, stage_id=stage_id)
    group = make_structured_driver_group(cfg, num_envs=int(args.num_envs), backend="sync", mode="train")
    try:
        data_learner.bind_native_runtime_contract(group)
        reset_many = getattr(group, "reset_many", None)
        if callable(reset_many):
            reset_many([int(args.seed) + env for env in range(int(args.num_envs))])
        train_probe_buffer, train_probe_views, train_probe_returns_all = _collect_one_rollout(
            data_learner,
            group,
            rollout_env_steps=int(args.rollout_env_steps),
            device=device,
            target="mc",
        )
        train_probe_stage = train_probe_views.training_view.stage_batches[int(stage_id)]
        train_probe_idx = torch.as_tensor(
            np.asarray(train_probe_stage.transition_indices, dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        train_probe_target = train_probe_returns_all.index_select(0, train_probe_idx).detach()
        train_world_parts = [train_probe_stage.world_batch]
        train_target_parts = [train_probe_target]
        train_summaries = [
            {
                "rollout": 0.0,
                "samples": float(train_probe_target.numel()),
                "return_mean": float(train_probe_target.mean().detach().cpu().item()),
                "return_std": float(train_probe_target.std(unbiased=False).detach().cpu().item()),
                "probe_rollout": 1.0,
            }
        ]
        remaining_train_rollouts = max(int(args.train_rollouts) - 1, 0)
        if remaining_train_rollouts > 0:
            extra_train_world, extra_train_target, extra_train_summaries = _collect_stage_bank(
                data_learner,
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

        if callable(reset_many):
            reset_many([int(args.seed) + 9_000_000 + env for env in range(int(args.num_envs))])
        heldout_buffer, heldout_views, heldout_returns_all = _collect_one_rollout(
            data_learner,
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
        vpi = _heldout_vpi_ceiling_probe(
            data_learner,
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
        train_vpi = _heldout_vpi_ceiling_probe(
            data_learner,
            cfg=cfg,
            stage_id=stage_id,
            num_envs=int(args.num_envs),
            heldout_views=train_probe_views,
            reward_mode=str(cfg.reward_mode),
            sample_rows=int(args.vpi_rows),
            policy_action_samples=int(args.vpi_policy_action_samples),
            continuations=int(args.vpi_continuations),
            min_horizon=int(args.min_horizon),
            branch_horizon_cap=int(args.branch_horizon_cap),
            seed=int(args.seed) + 7_000_000,
            device=device,
            initial_critic_state=None,
            follow_deterministic=str(args.vpi_follow) == "deterministic",
        )
    finally:
        close_structured_env_group(group)

    rows = list(vpi.get("rows", []))
    benchmark_world, vhat = _vhat_rows_to_world(heldout_world, rows, device=device)
    train_rows = list(train_vpi.get("rows", []))
    train_benchmark_world, train_vhat = _vhat_rows_to_world(train_probe_stage.world_batch, train_rows, device=device)

    base_learner, _base_actor, _base_critic = _make_learner(cfg, device=device, stage_id=stage_id)
    base_history = _train_critic_only(
        base_learner,
        stage_id=stage_id,
        train_world=train_world,
        train_target=train_target,
        heldout_world=heldout_world,
        heldout_target=heldout_target,
        epochs=int(args.base_critic_epochs),
        minibatches=int(args.critic_minibatches),
        lr=float(cfg.critic_lr),
        max_grad_norm=float(cfg.max_grad_norm),
        eval_every=int(args.eval_every),
    )
    train_ctx = _system_context(base_learner, train_world).to(device=device)
    heldout_ctx = _system_context(base_learner, heldout_world).to(device=device)
    bench_ctx = _system_context(base_learner, benchmark_world).to(device=device)

    base_actual_vhat = _critic_values(base_learner, stage_id, benchmark_world)
    base_actual_train_vhat = _critic_values(base_learner, stage_id, train_benchmark_world)
    base_actual_heldout = _critic_values(base_learner, stage_id, heldout_world)
    ridge_system_vhat = _ridge_predict(train_ctx, train_target, bench_ctx, ridge=float(args.system_ridge))
    train_bench_ctx = _system_context(base_learner, train_benchmark_world).to(device=device)
    ridge_system_train_vhat = _ridge_predict(train_ctx, train_target, train_bench_ctx, ridge=float(args.system_ridge))
    ridge_global_vhat = _ridge_predict(
        _world_global_features(train_world),
        train_target,
        _world_global_features(benchmark_world),
        ridge=float(args.global_ridge),
    )
    ridge_global_train_vhat = _ridge_predict(
        _world_global_features(train_world),
        train_target,
        _world_global_features(train_benchmark_world),
        ridge=float(args.global_ridge),
    )

    head_results: list[dict[str, Any]] = []
    if not bool(args.skip_head_only):
        head_kinds = [item.strip() for item in str(args.head_kinds).split(",") if item.strip()]
        head_lrs = _parse_float_list(args.head_lrs)
        for kind in head_kinds:
            for lr in head_lrs:
                head_results.append(
                    _train_head_only(
                        train_x=train_ctx,
                        train_y=train_target,
                        heldout_x=heldout_ctx,
                        heldout_y=heldout_target,
                        bench_x=bench_ctx,
                        bench_y=vhat,
                        kind=kind,
                        hidden=int(args.head_hidden),
                        lr=float(lr),
                        epochs=int(args.head_epochs),
                        minibatches=int(args.critic_minibatches),
                        max_grad_norm=float(cfg.max_grad_norm),
                        seed=int(args.seed) + 20_000 + len(head_results),
                    )
                )

    lr_results: list[dict[str, Any]] = []
    if not bool(args.skip_full_lr_sweep):
        full_lrs = _parse_float_list(args.full_lrs)
        for lr_idx, lr in enumerate(full_lrs):
            _set_seed(int(args.seed) + 30_000)
            learner, _actor, _critic = _make_learner(cfg, device=device, stage_id=stage_id)
            history = _train_critic_only(
                learner,
                stage_id=stage_id,
                train_world=train_world,
                train_target=train_target,
                heldout_world=heldout_world,
                heldout_target=heldout_target,
                epochs=int(args.full_critic_epochs),
                minibatches=int(args.critic_minibatches),
                lr=float(lr),
                max_grad_norm=float(cfg.max_grad_norm),
                eval_every=int(args.eval_every),
            )
            pred_vhat = _critic_values(learner, stage_id, benchmark_world)
            pred_heldout = _critic_values(learner, stage_id, heldout_world)
            lr_results.append(
                {
                    "lr": float(lr),
                    "history": history,
                    "benchmark_vhat": _eval_pred(pred_vhat, vhat),
                    "heldout_mc": _eval_pred(pred_heldout, heldout_target.detach().cpu().numpy()),
                }
            )
            del learner, _actor, _critic
            if device.type == "cuda":
                torch.cuda.empty_cache()

    optimizer_variant_results: list[dict[str, Any]] = []
    for variant in _parse_variant_specs(args.optimizer_variants):
        optimizer_variant_results.append(
            _train_full_variant(
                cfg=cfg,
                stage_id=stage_id,
                train_world=train_world,
                train_target=train_target,
                heldout_world=heldout_world,
                heldout_target=heldout_target,
                benchmark_world=benchmark_world,
                benchmark_vhat=vhat,
                train_benchmark_world=train_benchmark_world,
                train_benchmark_vhat=train_vhat,
                device=device,
                variant=variant,
                epochs=int(args.full_critic_epochs),
                minibatches=int(args.critic_minibatches),
                eval_every=int(args.eval_every),
                max_grad_norm=float(cfg.max_grad_norm),
                seed=int(args.seed) + 40_000 + len(optimizer_variant_results),
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = {
        "stage": str(args.stage),
        "stage_id": int(stage_id),
        "reward_mode": str(cfg.reward_mode),
        "config": str(args.config),
        "seed": int(args.seed),
        "train_samples": int(train_target.numel()),
        "heldout_samples": int(heldout_target.numel()),
        "train_rollout_summaries": train_summaries,
        "vpi": {
            "rows": int(len(rows)),
            "train_rows": int(len(train_rows)),
            "policy_action_samples": int(vpi.get("policy_action_samples", 0)),
            "continuations": int(vpi.get("continuations", 0)),
            "follow": str(vpi.get("follow", "")),
            "vhat": _summ(vhat),
            "train_vhat": _summ(train_vhat),
        },
        "base_critic": {
            "critic_lr": float(cfg.critic_lr),
            "history": base_history,
            "benchmark_vhat": _eval_pred(base_actual_vhat, vhat),
            "train_benchmark_vhat": _eval_pred(base_actual_train_vhat, train_vhat),
            "heldout_mc": _eval_pred(base_actual_heldout, heldout_target.detach().cpu().numpy()),
        },
        "ridge_probes": {
            "system_context_vhat": _eval_pred(ridge_system_vhat, vhat),
            "system_context_train_vhat": _eval_pred(ridge_system_train_vhat, train_vhat),
            "global_scalars_vhat": _eval_pred(ridge_global_vhat, vhat),
            "global_scalars_train_vhat": _eval_pred(ridge_global_train_vhat, train_vhat),
        },
        "head_only": head_results,
        "full_lr_sweep": lr_results,
        "optimizer_variants": optimizer_variant_results,
        "elapsed_sec": max(time.time() - started, 0.0),
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "base_vhat": out["base_critic"]["benchmark_vhat"],
        "base_train_vhat": out["base_critic"]["train_benchmark_vhat"],
        "ridge_vhat": out["ridge_probes"],
        "head_only": [
            {
                "kind": item["kind"],
                "lr": item["lr"],
                "vhat_ev": item["benchmark_vhat"]["ev"],
                "train_vhat_ev": None
                if item.get("train_benchmark_vhat") is None
                else item["train_benchmark_vhat"]["ev"],
                "heldout_ev": item["heldout_mc"]["ev"],
                "train_ev": item["train_mc"]["ev"],
            }
            for item in head_results
        ],
        "full_lr_sweep": [
            {
                "lr": item["lr"],
                "vhat_ev": item["benchmark_vhat"]["ev"],
                "heldout_ev": item["heldout_mc"]["ev"],
                "train_ev": item["history"][-1]["train_ev"] if item["history"] else None,
            }
            for item in lr_results
        ],
        "optimizer_variants": [
            {
                "name": item["name"],
                "optimizer": item["optimizer"],
                "target": item["target"],
                "lr": item["lr"],
                "weight_decay": item["weight_decay"],
                "input_norm": item["input_norm"],
                "vhat_ev": item["benchmark_vhat"]["ev"],
                "heldout_ev": item["heldout_mc"]["ev"],
                "train_ev": item["train_mc"]["ev"],
            }
            for item in optimizer_variant_results
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
