from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver_group
from scripts.audit_fixed_critic_benchmark import _corr, _ev, _ridge_predict, _world_global_features
from scripts.audit_stage_critic_only_fit import (
    _clone_dataclass_tensors,
    _collect_one_rollout,
    _collect_stage_bank,
    _collate_dataclass,
    _eval_critic,
    _heldout_vpi_ceiling_probe,
    _index_dataclass,
    _make_learner,
    _train_critic_only,
)
from scripts.audit_stage_ppo_credit_alignment import STAGE_ID, _force_single_stage_config, _set_seed
from scripts.diagnose_reward_action_sensitivity import _summ


def _parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for piece in str(text).replace(";", ",").split(","):
        piece = piece.strip()
        if piece:
            out.append(int(piece))
    return out


def _to_device(batch: Any, device: torch.device) -> Any:
    if not is_dataclass(batch):
        raise TypeError("expected dataclass batch")
    values: dict[str, Any] = {}
    for item in fields(batch):
        value = getattr(batch, item.name)
        values[item.name] = value.detach().to(device=device) if torch.is_tensor(value) else value
    return type(batch)(**values)


def _eval_pred(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    return {
        "ev": _ev(pred, target),
        "corr": _corr(pred, target),
        "mse": float(np.mean((pred - target) ** 2)) if target.size else 0.0,
        "mae": float(np.mean(np.abs(pred - target))) if target.size else 0.0,
        "pred": _summ(pred),
        "target": _summ(target),
        "residual": _summ(target - pred),
    }


def _group_eval_by_seed(
    *,
    rows: list[dict[str, Any]],
    pred: np.ndarray,
    vhat: np.ndarray,
    single_mc: np.ndarray,
    global_pred: np.ndarray,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seeds = sorted({int(row["seed_base"]) for row in rows})
    for seed in seeds:
        idx = np.asarray([i for i, row in enumerate(rows) if int(row["seed_base"]) == int(seed)], dtype=np.int64)
        if idx.size <= 0:
            continue
        out.append(
            {
                "seed_base": int(seed),
                "rows": int(idx.size),
                "critic_vs_vhat": _eval_pred(pred[idx], vhat[idx]),
                "critic_vs_single_mc": _eval_pred(pred[idx], single_mc[idx]),
                "global_linear_vs_vhat": _eval_pred(global_pred[idx], vhat[idx]),
                "single_minus_vhat": _summ(single_mc[idx] - vhat[idx]),
            }
        )
    return out


def make_artifact(args: argparse.Namespace) -> None:
    started = time.time()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    _set_seed(int(args.seed))
    cfg = load_config(args.config)
    stage_id = STAGE_ID[str(args.stage)]
    _force_single_stage_config(cfg, stage_id=stage_id, reward_mode=args.reward_mode)

    learner, _actor, _critic = _make_learner(cfg, device=device, stage_id=stage_id)
    group = make_structured_driver_group(cfg, num_envs=int(args.num_envs), backend="sync", mode="train")
    world_parts: list[Any] = []
    vhat_parts: list[torch.Tensor] = []
    vhat_se_parts: list[torch.Tensor] = []
    single_mc_parts: list[torch.Tensor] = []
    row_records: list[dict[str, Any]] = []
    probe_summaries: list[dict[str, Any]] = []
    try:
        learner.bind_native_runtime_contract(group)
        reset_many = getattr(group, "reset_many", None)
        for seed_base in _parse_int_list(str(args.heldout_seed_bases)):
            if callable(reset_many):
                reset_many([int(seed_base) + env for env in range(int(args.num_envs))])
            _buffer, views, returns_all = _collect_one_rollout(
                learner,
                group,
                rollout_env_steps=int(args.rollout_env_steps),
                device=device,
                target="mc",
            )
            stage_batch = views.training_view.stage_batches[int(stage_id)]
            probe = _heldout_vpi_ceiling_probe(
                learner,
                cfg=cfg,
                stage_id=int(stage_id),
                num_envs=int(args.num_envs),
                heldout_views=views,
                reward_mode=str(args.reward_mode or cfg.reward_mode),
                sample_rows=int(args.rows_per_seed),
                policy_action_samples=int(args.policy_action_samples),
                continuations=int(args.continuations),
                min_horizon=int(args.min_horizon),
                branch_horizon_cap=int(args.branch_horizon_cap),
                seed=int(seed_base) + int(args.seed),
                device=device,
                initial_critic_state=None,
                follow_deterministic=str(args.follow) == "deterministic",
                future_random_mode=str(args.future_random),
            )
            rows = list(probe.get("rows", []))
            if not rows:
                continue
            stage_indices = torch.as_tensor(
                [int(row["stage_sample"]) for row in rows],
                dtype=torch.long,
                device=device,
            )
            selected_world = _index_dataclass(stage_batch.world_batch, stage_indices)
            transition_indices = np.asarray(stage_batch.transition_indices, dtype=np.int64)
            global_indices = torch.as_tensor(
                [int(transition_indices[int(row["stage_sample"])]) for row in rows],
                dtype=torch.long,
                device=device,
            )
            single_mc = returns_all.index_select(0, global_indices).detach()
            vhat = torch.as_tensor([float(row["vhat"]) for row in rows], dtype=torch.float32, device=device)
            vhat_se = torch.as_tensor([float(row["vhat_se"]) for row in rows], dtype=torch.float32, device=device)
            world_parts.append(_clone_dataclass_tensors(selected_world, device=torch.device("cpu")))
            vhat_parts.append(vhat.detach().cpu())
            vhat_se_parts.append(vhat_se.detach().cpu())
            single_mc_parts.append(single_mc.detach().cpu())
            for local_idx, row in enumerate(rows):
                rec = dict(row)
                rec["seed_base"] = int(seed_base)
                rec["artifact_row"] = int(len(row_records))
                rec["single_mc"] = float(single_mc[local_idx].detach().cpu().item())
                row_records.append(rec)
            probe_summaries.append(
                {
                    "seed_base": int(seed_base),
                    "rows": int(len(rows)),
                    "vhat": probe.get("vhat", {}),
                    "target_se": probe.get("final", {}).get("target_se", probe.get("target_se", {})),
                    "return_cube": probe.get("return_cube", {}),
                }
            )
    finally:
        close_structured_env_group(group)

    if not world_parts:
        raise RuntimeError("no Vhat rows collected.")
    vhat_world = _collate_dataclass(world_parts, torch.device("cpu"))
    vhat = torch.cat(vhat_parts, dim=0).to(dtype=torch.float32)
    vhat_se = torch.cat(vhat_se_parts, dim=0).to(dtype=torch.float32)
    single_mc = torch.cat(single_mc_parts, dim=0).to(dtype=torch.float32)
    artifact = {
        "meta": {
            "stage": str(args.stage),
            "stage_id": int(stage_id),
            "config": str(args.config),
            "reward_mode": str(args.reward_mode or cfg.reward_mode),
            "num_envs": int(args.num_envs),
            "rollout_env_steps": int(args.rollout_env_steps),
            "heldout_seed_bases": _parse_int_list(str(args.heldout_seed_bases)),
            "rows_per_seed": int(args.rows_per_seed),
            "policy_action_samples_extra": int(args.policy_action_samples),
            "policy_action_count": int(args.policy_action_samples) + 1,
            "continuations": int(args.continuations),
            "effective_samples_per_state": (int(args.policy_action_samples) + 1) * int(args.continuations),
            "follow": str(args.follow),
            "future_random": str(args.future_random),
            "min_horizon": int(args.min_horizon),
            "branch_horizon_cap": int(args.branch_horizon_cap),
            "seed": int(args.seed),
            "elapsed_sec": max(time.time() - started, 0.0),
        },
        "vhat_world": vhat_world,
        "vhat": vhat,
        "vhat_se": vhat_se,
        "single_mc": single_mc,
        "rows": row_records,
        "probe_summaries": probe_summaries,
    }
    out_path = Path(args.artifact)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, out_path)
    summary = {
        "artifact": str(out_path),
        "meta": artifact["meta"],
        "rows": int(vhat.numel()),
        "vhat": _summ(vhat.numpy()),
        "vhat_se": _summ(vhat_se.numpy()),
        "single_mc": _summ(single_mc.numpy()),
        "single_minus_vhat": _summ((single_mc - vhat).numpy()),
    }
    summary_path = out_path.with_suffix(out_path.suffix + ".json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def eval_artifact(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    meta = dict(artifact["meta"])
    cfg = load_config(str(args.config or meta["config"]))
    stage_id = int(meta["stage_id"])
    _force_single_stage_config(cfg, stage_id=stage_id, reward_mode=str(args.reward_mode or meta["reward_mode"]))
    if args.critic_lr is not None:
        cfg.critic_lr = float(args.critic_lr)
    if args.critic_value_mode is not None:
        cfg.critic_value_mode = str(args.critic_value_mode)
    if args.critic_message_layers is not None:
        cfg.critic_message_layers = int(args.critic_message_layers)
    if args.critic_value_head_hidden is not None:
        cfg.critic_value_head_hidden = int(args.critic_value_head_hidden)
    _set_seed(int(args.init_seed))
    learner, _actor, _critic = _make_learner(cfg, device=device, stage_id=stage_id)
    group = make_structured_driver_group(cfg, num_envs=int(args.num_envs), backend="sync", mode="train")
    try:
        learner.bind_native_runtime_contract(group)
        train_world, train_target, train_summaries = _collect_stage_bank(
            learner,
            group,
            stage_id=stage_id,
            rollout_env_steps=int(args.rollout_env_steps),
            rollouts=int(args.train_rollouts),
            seed_base=int(args.train_seed_base),
            device=device,
            target=str(args.train_target),
        )
        history = _train_critic_only(
            learner,
            stage_id=stage_id,
            train_world=train_world,
            train_target=train_target,
            heldout_world=train_world,
            heldout_target=train_target,
            epochs=int(args.critic_epochs),
            minibatches=int(args.critic_minibatches),
            lr=float(cfg.critic_lr),
            max_grad_norm=float(cfg.max_grad_norm),
            eval_every=max(1, int(args.critic_epochs) // 3),
        )
        vhat_world = _to_device(artifact["vhat_world"], device)
        vhat = artifact["vhat"].to(device=device, dtype=torch.float32)
        single_mc = artifact["single_mc"].to(device=device, dtype=torch.float32)
        pred, vhat_stats = _eval_critic(
            learner,
            stage_id=stage_id,
            world_bank=vhat_world,
            target=vhat,
            batch_size=max(1024, int(vhat.numel())),
        )
        single_stats = _eval_pred(pred, single_mc.detach().cpu().numpy())
        global_pred = _ridge_predict(
            _world_global_features(train_world),
            train_target,
            _world_global_features(vhat_world),
            ridge=float(args.global_ridge),
        )
        pred_np = np.asarray(pred, dtype=np.float64).reshape(-1)
        vhat_np = vhat.detach().cpu().numpy().astype(np.float64)
        single_np = single_mc.detach().cpu().numpy().astype(np.float64)
        global_np = np.asarray(global_pred, dtype=np.float64).reshape(-1)
        global_vhat_stats = _eval_pred(global_pred, vhat.detach().cpu().numpy())
        result = {
            "artifact": str(args.artifact),
            "artifact_meta": meta,
            "eval_config": {
                "train_seed_base": int(args.train_seed_base),
                "train_rollouts": int(args.train_rollouts),
                "train_target": str(args.train_target),
                "critic_epochs": int(args.critic_epochs),
                "critic_lr": float(cfg.critic_lr),
                "critic_message_layers": int(getattr(cfg, "critic_message_layers", -1)),
                "critic_value_head_hidden": int(getattr(cfg, "critic_value_head_hidden", -1)),
                "init_seed": int(args.init_seed),
            },
            "train_summaries": train_summaries,
            "train_history": history,
            "critic_vs_vhat": vhat_stats,
            "critic_vs_single_mc": single_stats,
            "global_linear_vs_vhat": global_vhat_stats,
            "per_seed": _group_eval_by_seed(
                rows=list(artifact.get("rows", [])),
                pred=pred_np,
                vhat=vhat_np,
                single_mc=single_np,
                global_pred=global_np,
            ),
            "vhat_se": _summ(artifact["vhat_se"].numpy()),
            "single_minus_vhat": _summ((artifact["single_mc"] - artifact["vhat"]).numpy()),
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "out": str(out_path),
                    "rows": int(vhat.numel()),
                    "critic_vhat_ev": result["critic_vs_vhat"]["ev"],
                    "critic_single_mc_ev": result["critic_vs_single_mc"]["ev"],
                    "global_vhat_ev": result["global_linear_vs_vhat"]["ev"],
                    "vhat_se": result["vhat_se"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        close_structured_env_group(group)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["make", "eval"], required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--stage", choices=sorted(STAGE_ID), default="sat")
    parser.add_argument("--config", default=None)
    parser.add_argument("--reward_mode", default=None)
    parser.add_argument("--out", default="runs/diagnostics/critic_relearn_20260505/vhat_eval.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--rollout_env_steps", type=int, default=250)
    parser.add_argument("--heldout_seed_bases", default="245210,255210,305210,9045210,9055210")
    parser.add_argument("--rows_per_seed", type=int, default=8)
    parser.add_argument("--policy_action_samples", type=int, default=7)
    parser.add_argument("--continuations", type=int, default=8)
    parser.add_argument("--follow", choices=["deterministic", "stochastic"], default="stochastic")
    parser.add_argument("--future_random", choices=["copy", "resample"], default="resample")
    parser.add_argument("--min_horizon", type=int, default=20)
    parser.add_argument("--branch_horizon_cap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=45210)
    parser.add_argument("--train_seed_base", type=int, default=45210)
    parser.add_argument("--train_rollouts", type=int, default=8)
    parser.add_argument("--train_target", choices=["mc", "train_gae"], default="mc")
    parser.add_argument("--critic_epochs", type=int, default=30)
    parser.add_argument("--critic_minibatches", type=int, default=8)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--critic_value_mode", choices=["relational", "global_only", "global_linear"], default=None)
    parser.add_argument("--critic_message_layers", type=int, default=1)
    parser.add_argument("--critic_value_head_hidden", type=int, default=None)
    parser.add_argument("--init_seed", type=int, default=45210)
    parser.add_argument("--global_ridge", type=float, default=1e-3)
    parser.add_argument("--torch_threads", type=int, default=1)
    args = parser.parse_args()
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    if args.mode == "make" and not args.config:
        raise ValueError("--config is required for --mode make")
    if args.mode == "make":
        make_artifact(args)
    else:
        eval_artifact(args)


if __name__ == "__main__":
    main()
