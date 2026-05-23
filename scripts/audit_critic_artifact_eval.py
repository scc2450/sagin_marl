from __future__ import annotations

import argparse
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

from scripts.audit_fixed_critic_benchmark import (
    _corr,
    _critic_values,
    _ev,
    _ridge_predict,
    _summ,
    _system_context,
    _system_context_layers,
    _world_global_features,
)
from scripts.audit_stage_critic_only_fit import (
    STAGE_ID,
    _clone_dataclass_tensors,
    _collect_one_rollout,
    _collect_stage_bank,
    _heldout_vpi_ceiling_probe,
    _index_dataclass,
    _make_learner,
    _train_critic_only,
)
from scripts.audit_stage_ppo_credit_alignment import _force_single_stage_config, _set_seed
from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver_group


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


def _selected_world(world: Any, rows: list[dict[str, Any]], *, device: torch.device) -> Any:
    idx = torch.as_tensor([int(row["stage_sample"]) for row in rows], dtype=torch.long, device=device)
    return _index_dataclass(world, idx)


def _rows_vhat(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(row["vhat"]) for row in rows], dtype=np.float64)


def _dataclass_to_device(batch: Any, device: torch.device) -> Any:
    from dataclasses import fields, is_dataclass

    if not is_dataclass(batch):
        raise TypeError("expected dataclass batch")
    values = {}
    for item in fields(batch):
        value = getattr(batch, item.name)
        values[item.name] = value.detach().to(device=device) if torch.is_tensor(value) else value
    return type(batch)(**values)


def make_artifact(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    _set_seed(int(args.seed))
    cfg = load_config(args.config)
    stage_id = STAGE_ID[str(args.stage)]
    _force_single_stage_config(cfg, stage_id=stage_id, reward_mode=args.reward_mode)

    learner, _actor, _critic = _make_learner(cfg, device=device, stage_id=stage_id)
    group = make_structured_driver_group(cfg, num_envs=int(args.num_envs), backend="sync", mode="train")
    started = time.time()
    try:
        learner.bind_native_runtime_contract(group)
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
        _heldout_buffer, heldout_views, heldout_returns_all = _collect_one_rollout(
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
        heldout_world = _clone_dataclass_tensors(heldout_stage.world_batch, device=device)
        heldout_target = heldout_returns_all.index_select(0, heldout_idx).detach()
        heldout_vpi = _heldout_vpi_ceiling_probe(
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
    finally:
        close_structured_env_group(group)

    payload = {
        "meta": {
            "stage": str(args.stage),
            "stage_id": int(stage_id),
            "config": str(args.config),
            "reward_mode": str(cfg.reward_mode),
            "seed": int(args.seed),
            "num_envs": int(args.num_envs),
            "rollout_env_steps": int(args.rollout_env_steps),
            "train_rollouts": int(args.train_rollouts),
            "vpi_rows": int(args.vpi_rows),
            "vpi_policy_action_samples": int(args.vpi_policy_action_samples),
            "vpi_continuations": int(args.vpi_continuations),
            "vpi_follow": str(args.vpi_follow),
            "elapsed_sec": max(time.time() - started, 0.0),
            "train_rollout_summaries": train_summaries,
        },
        "train_world": _dataclass_to_device(train_world, torch.device("cpu")),
        "train_target": train_target.detach().cpu(),
        "heldout_world": _dataclass_to_device(heldout_world, torch.device("cpu")),
        "heldout_target": heldout_target.detach().cpu(),
        "heldout_vpi_rows": list(heldout_vpi.get("rows", [])),
    }
    out_path = Path(args.artifact)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(json.dumps({"artifact": str(out_path), "meta": payload["meta"]}, ensure_ascii=False, indent=2))


def eval_artifact(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    meta = dict(artifact["meta"])
    cfg = load_config(str(args.config or meta["config"]))
    stage_id = int(meta["stage_id"])
    _force_single_stage_config(cfg, stage_id=stage_id, reward_mode=str(args.reward_mode or meta["reward_mode"]))
    if args.critic_lr is not None:
        cfg.critic_lr = float(args.critic_lr)
    if args.critic_message_layers is not None:
        cfg.critic_message_layers = int(args.critic_message_layers)
    if args.critic_value_head_hidden is not None:
        cfg.critic_value_head_hidden = int(args.critic_value_head_hidden)
    if bool(args.critic_stage_specific_paths_enabled):
        cfg.critic_stage_specific_paths_enabled = True
    for attr in (
        "critic_sat_hidden",
        "critic_sat_embed_dim",
        "critic_sat_encoder_mlp_layers",
        "critic_sat_message_mlp_layers",
        "critic_sat_message_layers",
        "critic_sat_value_head_hidden",
        "critic_sat_value_head_layers",
    ):
        value = getattr(args, attr)
        if value is not None:
            setattr(cfg, attr, int(value))

    train_world = _dataclass_to_device(artifact["train_world"], device)
    heldout_world = _dataclass_to_device(artifact["heldout_world"], device)
    train_target = artifact["train_target"].to(device=device, dtype=torch.float32)
    heldout_target = artifact["heldout_target"].to(device=device, dtype=torch.float32)
    heldout_rows = list(artifact["heldout_vpi_rows"])
    benchmark_world = _selected_world(heldout_world, heldout_rows, device=device)
    vhat = _rows_vhat(heldout_rows)

    _set_seed(int(args.init_seed))
    learner, _actor, _critic = _make_learner(cfg, device=device, stage_id=stage_id)
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
    actual_vhat = _critic_values(learner, stage_id, benchmark_world)
    actual_heldout = _critic_values(learner, stage_id, heldout_world)
    train_ctx = _system_context(learner, train_world)
    bench_ctx = _system_context(learner, benchmark_world)
    global_pred = _ridge_predict(
        _world_global_features(train_world),
        train_target,
        _world_global_features(benchmark_world),
        ridge=float(args.global_ridge),
    )
    system_pred = _ridge_predict(train_ctx, train_target, bench_ctx, ridge=float(args.system_ridge))
    train_layers = _system_context_layers(learner, train_world)
    bench_layers = _system_context_layers(learner, benchmark_world)
    layer_metrics = {}
    for name in sorted(train_layers):
        if name not in bench_layers:
            continue
        pred = _ridge_predict(train_layers[name], train_target, bench_layers[name], ridge=float(args.system_ridge))
        layer_metrics[f"{name}_linear"] = _eval_pred(pred, vhat)

    out = {
        "artifact": str(args.artifact),
        "meta": meta,
        "stage_id": int(stage_id),
        "critic_lr": float(cfg.critic_lr),
        "critic_message_layers": int(getattr(cfg, "critic_message_layers", 0)),
        "critic_value_head_hidden": int(getattr(cfg, "critic_value_head_hidden", 0)),
        "critic_stage_specific_paths_enabled": bool(getattr(cfg, "critic_stage_specific_paths_enabled", False)),
        "critic_sat_message_layers": int(getattr(cfg, "critic_sat_message_layers", 0)),
        "critic_sat_value_head_hidden": int(getattr(cfg, "critic_sat_value_head_hidden", 0)),
        "init_seed": int(args.init_seed),
        "critic_epochs": int(args.critic_epochs),
        "critic_minibatches": int(args.critic_minibatches),
        "history": history,
        "benchmark": {
            "actual_critic": _eval_pred(actual_vhat, vhat),
            "global_linear": _eval_pred(global_pred, vhat),
            "system_context_linear": _eval_pred(system_pred, vhat),
            **layer_metrics,
        },
        "heldout_mc": {
            "actual_critic": _eval_pred(actual_heldout, heldout_target.detach().cpu().numpy()),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out_path),
                "critic_message_layers": out["critic_message_layers"],
                "critic_value_head_hidden": out["critic_value_head_hidden"],
                "init_seed": out["init_seed"],
                "actual_vhat_ev": out["benchmark"]["actual_critic"]["ev"],
                "system_context_vhat_ev": out["benchmark"]["system_context_linear"]["ev"],
                "heldout_mc_ev": out["heldout_mc"]["actual_critic"]["ev"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["make", "eval"], required=True)
    parser.add_argument("--stage", choices=sorted(STAGE_ID), default="sat")
    parser.add_argument("--config", default="")
    parser.add_argument("--reward_mode", default=None)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--rollout_env_steps", type=int, default=250)
    parser.add_argument("--train_rollouts", type=int, default=8)
    parser.add_argument("--vpi_rows", type=int, default=32)
    parser.add_argument("--vpi_policy_action_samples", type=int, default=4)
    parser.add_argument("--vpi_continuations", type=int, default=2)
    parser.add_argument("--vpi_follow", choices=["deterministic", "stochastic"], default="stochastic")
    parser.add_argument("--min_horizon", type=int, default=20)
    parser.add_argument("--branch_horizon_cap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=45210)
    parser.add_argument("--init_seed", type=int, default=45210)
    parser.add_argument("--critic_lr", type=float, default=None)
    parser.add_argument("--critic_message_layers", type=int, default=None)
    parser.add_argument("--critic_value_head_hidden", type=int, default=None)
    parser.add_argument("--critic_stage_specific_paths_enabled", action="store_true")
    parser.add_argument("--critic_sat_hidden", type=int, default=None)
    parser.add_argument("--critic_sat_embed_dim", type=int, default=None)
    parser.add_argument("--critic_sat_encoder_mlp_layers", type=int, default=None)
    parser.add_argument("--critic_sat_message_mlp_layers", type=int, default=None)
    parser.add_argument("--critic_sat_message_layers", type=int, default=None)
    parser.add_argument("--critic_sat_value_head_hidden", type=int, default=None)
    parser.add_argument("--critic_sat_value_head_layers", type=int, default=None)
    parser.add_argument("--critic_epochs", type=int, default=30)
    parser.add_argument("--critic_minibatches", type=int, default=8)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--global_ridge", type=float, default=1.0e-3)
    parser.add_argument("--system_ridge", type=float, default=1.0e-3)
    parser.add_argument("--torch_threads", type=int, default=1)
    args = parser.parse_args()

    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    if args.mode == "make":
        if not args.config:
            raise ValueError("--config is required in make mode")
        make_artifact(args)
    else:
        if not args.out:
            raise ValueError("--out is required in eval mode")
        eval_artifact(args)


if __name__ == "__main__":
    main()
