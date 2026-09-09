from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(ROOT, "sagin_marl")):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        raise RuntimeError("Could not locate repository root.")
    ROOT = parent
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import yaml

from sagin_marl.env.config import SaginConfig, load_config, update_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.rl.critic import CriticNet
from sagin_marl.rl.mappo import _configure_actor_trainability
from sagin_marl.rl.policy import ActorNet, batch_flatten_obs
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


DEFAULT_RUNS = [
    "runs/phase1_actions/curriculum_formal_u1500_subproc12_t2/stage1_accel",
    "runs/phase1_actions/curriculum_formal_u1500_subproc12_t2/stage2_bw",
    "runs/phase1_actions/curriculum_formal_u1500_subproc12_t2/stage3_sat",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _ema(values: list[float], beta: float = 0.9) -> list[float]:
    out: list[float] = []
    state: float | None = None
    for value in values:
        if state is None:
            state = value
        else:
            state = beta * state + (1.0 - beta) * value
        out.append(float(state))
    return out


def _module_entries(actor) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    if getattr(actor, "encoder_type", "flat_mlp") == "flat_mlp":
        items.extend(
            [
                ("obs_norm", getattr(actor, "obs_norm", None)),
                ("fc1", getattr(actor, "fc1", None)),
                ("fc2", getattr(actor, "fc2", None)),
            ]
        )
    else:
        items.extend(
            [
                ("own_encoder", getattr(actor, "own_encoder", None)),
                ("danger_nbr_encoder", getattr(actor, "danger_nbr_encoder", None)),
                ("users_encoder", getattr(actor, "users_encoder", None)),
                ("sats_encoder", getattr(actor, "sats_encoder", None)),
                ("nbrs_encoder", getattr(actor, "nbrs_encoder", None)),
                ("fusion_fc1", getattr(actor, "fusion_fc1", None)),
                ("fusion_fc2", getattr(actor, "fusion_fc2", None)),
            ]
        )
    items.extend(
        [
            ("mu_head", getattr(actor, "mu_head", None)),
            ("bw_user_encoder", getattr(actor, "bw_user_encoder", None)),
            ("bw_scorer", getattr(actor, "bw_scorer", None)),
            ("sat_action_encoder", getattr(actor, "sat_action_encoder", None)),
            ("sat_scorer", getattr(actor, "sat_scorer", None)),
        ]
    )
    return items


def _module_report(name: str, module: Any) -> dict[str, Any] | None:
    if module is None:
        return None
    total = 0
    trainable = 0
    for param in module.parameters():
        total += int(param.numel())
        if param.requires_grad:
            trainable += int(param.numel())
    if total <= 0:
        return None
    if trainable <= 0:
        state = "frozen"
    elif trainable >= total:
        state = "trainable"
    else:
        state = "partial"
    return {
        "name": name,
        "state": state,
        "total_params": total,
        "trainable_params": trainable,
    }


def _load_run_cfg(run_dir: Path) -> tuple[Any, str]:
    config_source_path = run_dir / "config_source.yaml"
    if config_source_path.exists():
        return load_config(str(config_source_path)), str(config_source_path.resolve())

    config_path = run_dir / "config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if isinstance(data, dict):
        data.pop("_config_source", None)
    cfg = update_config(SaginConfig(), data)
    return cfg, str(config_path.resolve())


def _inspect_run(run_dir: Path) -> dict[str, Any]:
    cfg, cfg_path = _load_run_cfg(run_dir)
    env = make_structured_env(cfg, mode="script")
    try:
        obs, _ = env.reset(seed=0)
        obs_list = list(obs.values())
        obs_batch = batch_flatten_obs(obs_list, cfg)
        obs_dim = int(obs_batch.shape[1])
        sample = obs_list[0]
        state_dim = int(np.asarray(env.get_global_state(), dtype=np.float32).shape[0])
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

    actor = ActorNet(obs_dim, cfg)
    train_heads = {
        "accel": bool(getattr(cfg, "train_accel", True)),
        "bw": bool(getattr(cfg, "train_bw", bool(getattr(cfg, "enable_bw_action", False)))),
        "sat": bool(getattr(cfg, "train_sat", not bool(getattr(cfg, "fixed_satellite_strategy", False)))),
    }
    _configure_actor_trainability(actor, cfg, train_heads)
    critic = CriticNet(state_dim, obs_dim, len(obs_list), cfg)

    actor_modules: list[dict[str, Any]] = []
    for name, module in _module_entries(actor):
        report = _module_report(name, module)
        if report is not None:
            actor_modules.append(report)

    log_std_report = {
        "name": "log_std",
        "state": "trainable" if bool(actor.log_std.requires_grad) else "frozen",
        "total_params": int(actor.log_std.numel()),
        "trainable_params": int(actor.log_std.numel()) if bool(actor.log_std.requires_grad) else 0,
    }
    mu_idx = next((idx for idx, item in enumerate(actor_modules) if item["name"] == "mu_head"), len(actor_modules))
    actor_modules.insert(mu_idx, log_std_report)

    actor_trainable_names = [name for name, param in actor.named_parameters() if param.requires_grad]
    critic_trainable_names = [name for name, param in critic.named_parameters() if param.requires_grad]
    actor_trainable_params = int(sum(param.numel() for param in actor.parameters() if param.requires_grad))
    critic_trainable_params = int(sum(param.numel() for param in critic.parameters() if param.requires_grad))

    return {
        "run_dir": str(run_dir.resolve()),
        "config_source": cfg_path,
        "obs": {
            "obs_dim": obs_dim,
            "own_dim": int(sample["own"].shape[0]),
            "users_shape": [int(x) for x in sample["users"].shape],
            "users_mask_dim": int(sample["users_mask"].shape[0]),
            "bw_valid_mask_dim": int(sample["bw_valid_mask"].shape[0]),
            "sats_shape": [int(x) for x in sample["sats"].shape],
            "sats_mask_dim": int(sample["sats_mask"].shape[0]),
            "sat_valid_mask_dim": int(sample["sat_valid_mask"].shape[0]),
            "nbrs_shape": [int(x) for x in sample["nbrs"].shape],
            "nbrs_mask_dim": int(sample["nbrs_mask"].shape[0]),
            "danger_nbr_shape": [int(x) for x in sample["danger_nbr"].shape] if "danger_nbr" in sample else None,
            "own_has_stage1_assoc_features": int(sample["own"].shape[0]) >= 10,
            "own_feature_tail_sample": [float(x) for x in sample["own"][-3:]],
        },
        "train_flags": {
            "train_accel": train_heads["accel"],
            "train_bw": train_heads["bw"],
            "train_sat": train_heads["sat"],
            "train_shared_backbone": bool(getattr(cfg, "train_shared_backbone", True)),
            "train_fusion": bool(getattr(cfg, "train_fusion", False)),
            "train_fusion_last_layer": bool(getattr(cfg, "train_fusion_last_layer", False)),
        },
        "optimizer_groups": [
            {
                "name": "actor_trainable",
                "param_count": actor_trainable_params,
                "param_tensors": len(actor_trainable_names),
            },
            {
                "name": "critic_trainable",
                "param_count": critic_trainable_params,
                "param_tensors": len(critic_trainable_names),
            },
        ],
        "actor_module_freeze_map": actor_modules,
        "actor_requires_grad_true": actor_trainable_names,
        "critic_requires_grad_true": critic_trainable_names,
    }


def _export_reward_components(run_dir: Path) -> Path:
    cfg, _ = _load_run_cfg(run_dir)
    in_path = run_dir / "metrics.csv"
    out_dir = run_dir / "diagnostics_20260325"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "reward_components_by_update.csv"

    rows_in: list[dict[str, str]] = []
    with in_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_in.append(dict(row))

    r_access_values: list[float] = []
    r_relay_values: list[float] = []
    r_pre_backlog_values: list[float] = []
    r_pre_drop_values: list[float] = []
    r_assoc_values: list[float] = []
    r_sat_values: list[float] = []
    reward_total_values: list[float] = []

    rows_out: list[dict[str, Any]] = []
    for row in rows_in:
        x_acc = _safe_float(row.get("x_acc"))
        x_rel = _safe_float(row.get("x_rel"))
        d_pre = _safe_float(row.get("d_pre"))
        backlog_steps = _safe_float(row.get("pre_backlog_steps_eval"))
        reward_raw = _safe_float(row.get("reward_raw"))
        reward_aux = _safe_float(row.get("reward_aux"))
        reward_total = _safe_float(row.get("rollout_reward_per_step"), reward_raw + reward_aux)
        r_access = float(getattr(cfg, "reward_w_access", 0.5) or 0.0) * x_acc
        r_relay = float(getattr(cfg, "reward_w_relay", 0.5) or 0.0) * x_rel
        r_pre_backlog = -float(getattr(cfg, "reward_w_pre_backlog", 0.08) or 0.0) * math.log1p(backlog_steps)
        r_pre_drop = -float(getattr(cfg, "reward_w_pre_drop", 1.0) or 0.0) * d_pre
        r_assoc = _safe_float(row.get("stage1_assoc_centroid_term"))
        r_sat = _safe_float(row.get("stage3_sat_overlap_term"))

        r_access_values.append(r_access)
        r_relay_values.append(r_relay)
        r_pre_backlog_values.append(r_pre_backlog)
        r_pre_drop_values.append(r_pre_drop)
        r_assoc_values.append(r_assoc)
        r_sat_values.append(r_sat)
        reward_total_values.append(reward_total)

        rows_out.append(
            {
                "step": int(_safe_float(row.get("step"))),
                "r_access": r_access,
                "r_relay": r_relay,
                "r_pre_backlog": r_pre_backlog,
                "r_pre_drop": r_pre_drop,
                "r_assoc_centroid": r_assoc,
                "r_sat_overlap": r_sat,
                "reward_raw": reward_raw,
                "reward_aux": reward_aux,
                "reward_total": reward_total,
            }
        )

    ema_access = _ema(r_access_values)
    ema_relay = _ema(r_relay_values)
    ema_backlog = _ema(r_pre_backlog_values)
    ema_drop = _ema(r_pre_drop_values)
    ema_assoc = _ema(r_assoc_values)
    ema_sat = _ema(r_sat_values)
    ema_total = _ema(reward_total_values)

    for idx, row in enumerate(rows_out):
        row["r_access_ema_beta0.9"] = ema_access[idx]
        row["r_relay_ema_beta0.9"] = ema_relay[idx]
        row["r_pre_backlog_ema_beta0.9"] = ema_backlog[idx]
        row["r_pre_drop_ema_beta0.9"] = ema_drop[idx]
        row["r_assoc_centroid_ema_beta0.9"] = ema_assoc[idx]
        row["r_sat_overlap_ema_beta0.9"] = ema_sat[idx]
        row["reward_total_ema_beta0.9"] = ema_total[idx]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="runs/phase1_actions/diagnostics_20260325",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    for run in args.runs:
        run_dir = Path(run)
        report = _inspect_run(run_dir)
        reward_csv = _export_reward_components(run_dir)
        report["reward_components_csv"] = str(reward_csv.resolve())
        summary.append(report)

    summary_path = out_dir / "stage_diagnostics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(summary_path.resolve())


if __name__ == "__main__":
    main()
