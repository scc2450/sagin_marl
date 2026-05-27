from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from scripts.audit_fixed_critic_benchmark import (
    _ev,
    _corr,
    _ridge_predict,
    _summ,
    _world_global_features,
    _world_raw_features,
)


def _to_device_dataclass(batch: Any, device: torch.device) -> Any:
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


def _world_feature_groups(world_state: Any) -> dict[str, torch.Tensor]:
    groups: dict[str, torch.Tensor] = {}
    for item in fields(world_state):
        value = getattr(world_state, item.name)
        if torch.is_tensor(value):
            groups[item.name] = _tensor_flat(value.detach())
    return groups


def _concat_groups(groups: dict[str, torch.Tensor], names: list[str]) -> torch.Tensor:
    parts = [groups[name] for name in names if name in groups]
    if not parts:
        raise RuntimeError(f"no groups found for {names}")
    return torch.cat(parts, dim=1)


def _feature_shift_by_seed(
    x_train: torch.Tensor,
    x_eval: torch.Tensor,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    x_train_cpu = x_train.detach().to(device="cpu", dtype=torch.float64)
    x_eval_cpu = x_eval.detach().to(device="cpu", dtype=torch.float64)
    mu = x_train_cpu.mean(dim=0, keepdim=True)
    std = x_train_cpu.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-6)
    z = ((x_eval_cpu - mu) / std).numpy()
    out: list[dict[str, Any]] = []
    for seed in sorted({int(row["seed_base"]) for row in rows}):
        idx = np.asarray([i for i, row in enumerate(rows) if int(row["seed_base"]) == seed], dtype=np.int64)
        block = z[idx]
        abs_block = np.abs(block[np.isfinite(block)])
        if abs_block.size == 0:
            out.append({"seed_base": int(seed), "rows": int(idx.size)})
            continue
        row_l2 = np.sqrt(np.mean(np.square(block), axis=1))
        out.append(
            {
                "seed_base": int(seed),
                "rows": int(idx.size),
                "mean_abs_z": float(np.mean(abs_block)),
                "p90_abs_z": float(np.percentile(abs_block, 90)),
                "max_abs_z": float(np.max(abs_block)),
                "frac_abs_z_gt3": float(np.mean(abs_block > 3.0)),
                "frac_abs_z_gt5": float(np.mean(abs_block > 5.0)),
                "row_l2_z_mean": float(np.mean(row_l2)),
                "row_l2_z_max": float(np.max(row_l2)),
            }
        )
    return out


def _tensor_flat(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.bool or not torch.is_floating_point(t):
        t = t.to(dtype=torch.float32)
    else:
        t = t.to(dtype=torch.float32)
    return t.reshape(int(t.shape[0]), -1)


def _joint_flatten_local(
    local_batch: Any,
    *,
    num_samples: int,
    num_agents: int,
    include_action_space: bool,
) -> tuple[torch.Tensor, list[tuple[str, int]]]:
    """Flatten per-agent SAT local obs into one system row per transition."""

    if not is_dataclass(local_batch):
        raise TypeError("expected LocalSatState dataclass")
    parts: list[torch.Tensor] = []
    dims: list[tuple[str, int]] = []
    flat_rows = int(num_samples) * int(num_agents)
    skip_names = set()
    if not include_action_space:
        skip_names.update({"candidate_sat_ids", "subset_members", "subset_mask"})
    for item in fields(local_batch):
        name = item.name
        if name in skip_names:
            continue
        value = getattr(local_batch, name)
        if not torch.is_tensor(value):
            continue
        tensor = value.detach()
        if tensor.ndim <= 0:
            continue
        if int(tensor.shape[0]) == flat_rows:
            reshaped = tensor.reshape(int(num_samples), int(num_agents), *tensor.shape[1:])
            flat = _tensor_flat(reshaped)
        elif int(tensor.shape[0]) == int(num_samples):
            flat = _tensor_flat(tensor)
        else:
            # Constant subset tables without a row dimension do not carry state
            # information.  subset_mask/candidate_sat_ids are the state-specific
            # action-space part.
            continue
        parts.append(flat)
        dims.append((name, int(flat.shape[1])))
    if not parts:
        raise RuntimeError("no local SAT tensors were flattened")
    return torch.cat(parts, dim=1), dims


def _stage_local_indices(stage_indices: torch.Tensor, *, num_agents: int) -> torch.Tensor:
    offsets = torch.arange(int(num_agents), dtype=torch.long, device=stage_indices.device).view(1, -1)
    return (stage_indices.reshape(-1, 1) * int(num_agents) + offsets).reshape(-1)


def _collect_train_bank(
    learner: Any,
    group: Any,
    *,
    stage_id: int,
    train_seed_base: int,
    train_rollouts: int,
    rollout_env_steps: int,
    device: torch.device,
) -> tuple[Any, Any, torch.Tensor, list[dict[str, float]]]:
    world_parts: list[Any] = []
    local_parts: list[Any] = []
    target_parts: list[torch.Tensor] = []
    summaries: list[dict[str, float]] = []
    reset_many = getattr(group, "reset_many", None)
    for ridx in range(max(int(train_rollouts), 1)):
        if callable(reset_many):
            reset_many([int(train_seed_base) + ridx * 10_000 + env for env in range(len(group))])
        _buffer, views, returns = _collect_one_rollout(
            learner,
            group,
            rollout_env_steps=int(rollout_env_steps),
            device=device,
            target="mc",
        )
        stage_batch = views.training_view.stage_batches.get(int(stage_id))
        if stage_batch is None or int(stage_batch.num_samples) <= 0:
            continue
        idx = torch.as_tensor(
            np.asarray(stage_batch.transition_indices, dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        target = returns.index_select(0, idx).detach()
        world_parts.append(_clone_dataclass_tensors(stage_batch.world_batch, device=device))
        local_parts.append(_clone_dataclass_tensors(stage_batch.local_batch, device=device))
        target_parts.append(target)
        summaries.append(
            {
                "rollout": float(ridx),
                "samples": float(target.numel()),
                "return_mean": float(target.mean().detach().cpu().item()),
                "return_std": float(target.std(unbiased=False).detach().cpu().item()),
            }
        )
    if not world_parts:
        raise RuntimeError("no train stage samples collected")
    return (
        _collate_dataclass(world_parts, device),
        _collate_dataclass(local_parts, device),
        torch.cat(target_parts, dim=0).to(device=device, dtype=torch.float32),
        summaries,
    )


def _dataclass_max_abs(a: Any, b: Any) -> dict[str, float]:
    if not (is_dataclass(a) and is_dataclass(b)):
        raise TypeError("expected dataclasses")
    out: dict[str, float] = {}
    for item in fields(a):
        name = item.name
        if not hasattr(b, name):
            continue
        av = getattr(a, name)
        bv = getattr(b, name)
        if not (torch.is_tensor(av) and torch.is_tensor(bv)):
            continue
        if tuple(av.shape) != tuple(bv.shape):
            out[name] = float("inf")
            continue
        if av.dtype == torch.bool or bv.dtype == torch.bool:
            diff = (av.to(torch.bool) != bv.to(torch.bool)).to(torch.float32)
        else:
            diff = (av.to(dtype=torch.float32) - bv.to(device=av.device, dtype=torch.float32)).abs()
        out[name] = float(diff.max().detach().cpu().item()) if diff.numel() else 0.0
    out["__max__"] = max(out.values()) if out else 0.0
    return out


def _collect_artifact_locals(
    learner: Any,
    group: Any,
    *,
    artifact_rows: list[dict[str, Any]],
    artifact_world: Any,
    stage_id: int,
    rollout_env_steps: int,
    num_envs: int,
    device: torch.device,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    reset_many = getattr(group, "reset_many", None)
    world_parts: list[Any] = []
    local_parts: list[Any] = []
    diagnostics: list[dict[str, Any]] = []
    rows_by_seed: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for artifact_idx, row in enumerate(artifact_rows):
        rows_by_seed.setdefault(int(row["seed_base"]), []).append((artifact_idx, row))
    for seed_base in sorted(rows_by_seed):
        if callable(reset_many):
            reset_many([int(seed_base) + env for env in range(int(num_envs))])
        _buffer, views, _returns = _collect_one_rollout(
            learner,
            group,
            rollout_env_steps=int(rollout_env_steps),
            device=device,
            target="mc",
        )
        stage_batch = views.training_view.stage_batches.get(int(stage_id))
        if stage_batch is None:
            raise RuntimeError(f"missing stage batch for seed {seed_base}")
        stage_indices = torch.as_tensor(
            [int(row["stage_sample"]) for _artifact_idx, row in rows_by_seed[seed_base]],
            dtype=torch.long,
            device=device,
        )
        selected_world = _clone_dataclass_tensors(_index_dataclass(stage_batch.world_batch, stage_indices), device=device)
        flat_local_idx = _stage_local_indices(stage_indices, num_agents=int(stage_batch.num_agents))
        selected_local = _clone_dataclass_tensors(_index_dataclass(stage_batch.local_batch, flat_local_idx), device=device)
        artifact_indices = torch.as_tensor(
            [int(artifact_idx) for artifact_idx, _row in rows_by_seed[seed_base]],
            dtype=torch.long,
            device=device,
        )
        artifact_world_sel = _to_device_dataclass(_index_dataclass(artifact_world, artifact_indices.cpu()), device)
        diff = _dataclass_max_abs(selected_world, artifact_world_sel)
        diagnostics.append(
            {
                "seed_base": int(seed_base),
                "rows": int(stage_indices.numel()),
                "world_replay_max_abs": diff,
            }
        )
        world_parts.append(selected_world)
        local_parts.append(selected_local)
    return _collate_dataclass(world_parts, device), _collate_dataclass(local_parts, device), diagnostics


def _per_seed_eval(rows: list[dict[str, Any]], pred: np.ndarray, target: np.ndarray) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seed in sorted({int(row["seed_base"]) for row in rows}):
        idx = np.asarray([i for i, row in enumerate(rows) if int(row["seed_base"]) == seed], dtype=np.int64)
        out.append({"seed_base": int(seed), "rows": int(idx.size), **_eval_pred(pred[idx], target[idx])})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--reward_mode", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--rollout_env_steps", type=int, default=250)
    parser.add_argument("--train_seed_base", type=int, default=45210)
    parser.add_argument("--train_rollouts", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=45210)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    meta = dict(artifact["meta"])
    cfg = load_config(str(args.config or meta["config"]))
    stage_id = int(meta.get("stage_id", STAGE_ID.get(str(meta.get("stage", "sat")), 1)))
    _force_single_stage_config(cfg, stage_id=stage_id, reward_mode=str(args.reward_mode or meta["reward_mode"]))
    _set_seed(int(args.seed))
    learner, _actor, _critic = _make_learner(cfg, device=device, stage_id=stage_id)
    group = make_structured_driver_group(cfg, num_envs=int(args.num_envs), backend="sync", mode="train")
    try:
        learner.bind_native_runtime_contract(group)
        train_world, train_local, train_target, train_summaries = _collect_train_bank(
            learner,
            group,
            stage_id=stage_id,
            train_seed_base=int(args.train_seed_base),
            train_rollouts=int(args.train_rollouts),
            rollout_env_steps=int(args.rollout_env_steps),
            device=device,
        )
        artifact_world = _to_device_dataclass(artifact["vhat_world"], device)
        artifact_rows = list(artifact["rows"])
        replay_world, artifact_local, replay_diagnostics = _collect_artifact_locals(
            learner,
            group,
            artifact_rows=artifact_rows,
            artifact_world=artifact_world,
            stage_id=stage_id,
            rollout_env_steps=int(args.rollout_env_steps),
            num_envs=int(args.num_envs),
            device=device,
        )
    finally:
        close_structured_env_group(group)

    vhat = artifact["vhat"].detach().cpu().numpy().astype(np.float64)
    train_samples = int(train_target.numel())
    train_agents = int(train_local.ego_features.shape[0] // max(train_samples, 1))
    eval_samples = int(len(artifact_rows))
    eval_agents = int(artifact_local.ego_features.shape[0] // max(eval_samples, 1))
    local_core_train, local_core_dims = _joint_flatten_local(
        train_local,
        num_samples=train_samples,
        num_agents=train_agents,
        include_action_space=False,
    )
    local_core_eval, _ = _joint_flatten_local(
        artifact_local,
        num_samples=eval_samples,
        num_agents=eval_agents,
        include_action_space=False,
    )
    local_full_train, local_full_dims = _joint_flatten_local(
        train_local,
        num_samples=train_samples,
        num_agents=train_agents,
        include_action_space=True,
    )
    local_full_eval, _ = _joint_flatten_local(
        artifact_local,
        num_samples=eval_samples,
        num_agents=eval_agents,
        include_action_space=True,
    )
    world_global_train = _world_global_features(train_world)
    world_global_eval = _world_global_features(artifact_world)
    world_raw_train, world_raw_dims = _world_raw_features(train_world)
    world_raw_eval, _ = _world_raw_features(artifact_world)
    replay_world_raw_eval, _ = _world_raw_features(replay_world)
    world_groups_train = _world_feature_groups(train_world)
    world_groups_eval = _world_feature_groups(artifact_world)

    probes: dict[str, dict[str, Any]] = {}
    feature_sets = {
        "critic_global_scalars": (world_global_train, world_global_eval),
        "critic_world_raw": (world_raw_train, world_raw_eval),
        "critic_world_raw_replayed": (world_raw_train, replay_world_raw_eval),
        "sat_local_core_no_action_space": (local_core_train, local_core_eval),
        "sat_local_full_with_action_space": (local_full_train, local_full_eval),
    }
    for name, (x_train, x_eval) in feature_sets.items():
        pred = _ridge_predict(x_train, train_target, x_eval, ridge=float(args.ridge))
        probes[name] = {
            "overall": _eval_pred(pred, vhat),
            "per_seed": _per_seed_eval(artifact_rows, pred, vhat),
            "feature_dim": int(x_train.shape[1]),
            "shift_by_seed": _feature_shift_by_seed(x_train, x_eval, artifact_rows),
        }
    world_group_probes: dict[str, dict[str, Any]] = {}
    for group_name in sorted(world_groups_train.keys()):
        if group_name not in world_groups_eval:
            continue
        x_train = world_groups_train[group_name]
        x_eval = world_groups_eval[group_name]
        pred = _ridge_predict(x_train, train_target, x_eval, ridge=float(args.ridge))
        world_group_probes[group_name] = {
            "overall": _eval_pred(pred, vhat),
            "per_seed": _per_seed_eval(artifact_rows, pred, vhat),
            "feature_dim": int(x_train.shape[1]),
            "shift_by_seed": _feature_shift_by_seed(x_train, x_eval, artifact_rows),
        }
    world_combo_probes: dict[str, dict[str, Any]] = {}
    group_order = [name for name, _dim in world_raw_dims]
    for group_name in group_order:
        if group_name == "global_scalars":
            continue
        combo_names = ["global_scalars", group_name]
        if all(name in world_groups_train and name in world_groups_eval for name in combo_names):
            x_train = _concat_groups(world_groups_train, combo_names)
            x_eval = _concat_groups(world_groups_eval, combo_names)
            pred = _ridge_predict(x_train, train_target, x_eval, ridge=float(args.ridge))
            world_combo_probes[f"global_plus_{group_name}"] = {
                "overall": _eval_pred(pred, vhat),
                "per_seed": _per_seed_eval(artifact_rows, pred, vhat),
                "feature_dim": int(x_train.shape[1]),
                "shift_by_seed": _feature_shift_by_seed(x_train, x_eval, artifact_rows),
            }
    for group_name in group_order:
        keep_names = [name for name in group_order if name != group_name]
        if all(name in world_groups_train and name in world_groups_eval for name in keep_names):
            x_train = _concat_groups(world_groups_train, keep_names)
            x_eval = _concat_groups(world_groups_eval, keep_names)
            pred = _ridge_predict(x_train, train_target, x_eval, ridge=float(args.ridge))
            world_combo_probes[f"world_minus_{group_name}"] = {
                "overall": _eval_pred(pred, vhat),
                "per_seed": _per_seed_eval(artifact_rows, pred, vhat),
                "feature_dim": int(x_train.shape[1]),
            }

    out = {
        "artifact": str(args.artifact),
        "meta": meta,
        "eval": {
            "stage_id": int(stage_id),
            "config": str(args.config or meta["config"]),
            "reward_mode": str(args.reward_mode or meta["reward_mode"]),
            "train_seed_base": int(args.train_seed_base),
            "train_rollouts": int(args.train_rollouts),
            "ridge": float(args.ridge),
            "train_samples": int(train_samples),
            "train_agents": int(train_agents),
            "eval_samples": int(eval_samples),
            "eval_agents": int(eval_agents),
            "train_summaries": train_summaries,
        },
        "feature_dims": {
            "local_core": local_core_dims,
            "local_full": local_full_dims,
            "world_raw": world_raw_dims,
        },
        "replay_diagnostics": replay_diagnostics,
        "vhat": _summ(vhat),
        "probes": probes,
        "world_group_probes": world_group_probes,
        "world_combo_probes": world_combo_probes,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out_path),
                "replay_max_abs_by_seed": {
                    str(item["seed_base"]): item["world_replay_max_abs"].get("__max__", None)
                    for item in replay_diagnostics
                },
                "overall_ev": {name: data["overall"]["ev"] for name, data in probes.items()},
                "overall_corr": {name: data["overall"]["corr"] for name, data in probes.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
