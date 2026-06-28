from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_eval import (
    _evaluate_structured_baseline_policy_with_traces,
    _fixed_policy_exec_sources,
    evaluate_structured_actor_exec_sources,
)
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.utils.checkpoint import load_state_dict_forgiving


def _actor_state_from_checkpoint(path: str | Path) -> dict[str, Any]:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint {path} is not a dictionary")
    actor_state = ckpt.get("actor")
    if isinstance(actor_state, dict):
        return dict(actor_state)
    if all(isinstance(k, str) for k in ckpt.keys()):
        return dict(ckpt)
    raise KeyError(f"checkpoint {path} does not contain an actor state_dict")


def _replace_prefix(base: dict[str, Any], source: dict[str, Any], prefix: str) -> int:
    prefix = str(prefix)
    count = 0
    for key, value in source.items():
        if key.startswith(prefix):
            if key not in base:
                raise KeyError(f"source key {key!r} is not present in base actor state")
            if tuple(value.shape) != tuple(base[key].shape):
                raise ValueError(
                    f"shape mismatch for {key}: source={tuple(value.shape)} base={tuple(base[key].shape)}"
                )
            base[key] = value.detach().clone()
            count += 1
    if count <= 0:
        raise ValueError(f"no keys matched prefix {prefix!r}")
    return count


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "episode",
        "reward_sum",
        "bw_weighted_workload_delta_sum",
        "bw_weighted_workload_level_sum",
        "processed_ratio_eval",
        "drop_ratio_eval",
        "pre_backlog_steps_eval",
        "D_sys_report",
        "x_acc_mean",
        "x_rel_mean",
        "g_pre_mean",
        "d_pre_mean",
        "sat_overlap_eval",
        "collision_episode_fraction",
        "terminated_early",
        "episode_length",
        "gu_queue_mean",
        "uav_queue_mean",
        "sat_queue_mean",
        "queue_total_mean",
        "arrival_sum",
        "outflow_sum",
        "backhaul_sum",
        "sat_processed_sum",
        "drop_sum",
        "outflow_arrival_ratio",
        "sat_incoming_arrival_ratio",
        "sat_processed_arrival_ratio",
        "sat_processed_incoming_ratio",
        "drop_ratio",
    ]
    seen = set(preferred)
    extra: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                extra.append(key)
    return preferred + extra


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base_checkpoint", default=None)
    parser.add_argument("--sat_checkpoint", default=None)
    parser.add_argument("--accel_checkpoint", default=None)
    parser.add_argument("--bw_checkpoint", default=None)
    parser.add_argument("--baseline_policy", default=None)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--episode_seed_base", type=int, default=900000)
    parser.add_argument("--policy_mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--exec_accel_source", default="policy")
    parser.add_argument("--exec_sat_source", default="policy")
    parser.add_argument("--exec_bw_source", default="policy")
    parser.add_argument("--access_bw_decision_interval", type=int, default=None)
    parser.add_argument("--sat_decision_interval", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--label", default="mixed")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.access_bw_decision_interval is not None:
        cfg.access_bw_decision_interval = max(int(args.access_bw_decision_interval), 1)
    if args.sat_decision_interval is not None:
        cfg.sat_decision_interval = max(int(args.sat_decision_interval), 1)
    cfg.exec_accel_source = str(args.exec_accel_source)
    cfg.exec_sat_source = str(args.exec_sat_source)
    cfg.exec_bw_source = str(args.exec_bw_source)

    requested = str(args.device).lower()
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    else:
        device = torch.device(requested)

    replaced: dict[str, int] = {}
    info: dict[str, Any] = {}
    effective_exec_sources = (
        str(args.exec_accel_source),
        str(args.exec_sat_source),
        str(args.exec_bw_source),
    )
    if args.baseline_policy:
        exec_sources = _fixed_policy_exec_sources(str(args.baseline_policy))
        if exec_sources is None:
            effective_exec_sources = (str(args.baseline_policy),) * 3
            info["fixed_policy_backend"] = "structured_python"
            summary, rows, _traces, _actions, _reset_rollouts = _evaluate_structured_baseline_policy_with_traces(
                cfg,
                baseline_policy=str(args.baseline_policy),
                episodes=int(args.episodes),
                episode_seed_base=int(args.episode_seed_base),
            )
        else:
            effective_exec_sources = tuple(str(item) for item in exec_sources)
            dummy_actor = torch.nn.Linear(1, 1).to(device)
            summary, rows = evaluate_structured_actor_exec_sources(
                cfg,
                dummy_actor,
                device=device,
                episodes=int(args.episodes),
                episode_seed_base=int(args.episode_seed_base),
                deterministic=True,
                num_envs=int(args.num_envs),
                vec_backend="sync",
                exec_accel_source=exec_sources[0],
                exec_sat_source=exec_sources[1],
                exec_bw_source=exec_sources[2],
            )
    else:
        if not args.base_checkpoint:
            raise ValueError("--base_checkpoint is required unless --baseline_policy is set")
        mixed_state = _actor_state_from_checkpoint(args.base_checkpoint)
        if args.accel_checkpoint:
            replaced["accel_policy"] = _replace_prefix(
                mixed_state, _actor_state_from_checkpoint(args.accel_checkpoint), "accel_policy."
            )
        if args.sat_checkpoint:
            replaced["sat_subset_policy"] = _replace_prefix(
                mixed_state, _actor_state_from_checkpoint(args.sat_checkpoint), "sat_subset_policy."
            )
        if args.bw_checkpoint:
            replaced["bw_policy"] = _replace_prefix(
                mixed_state, _actor_state_from_checkpoint(args.bw_checkpoint), "bw_policy."
            )

        bundle = build_structured_modules_from_config(
            cfg,
            hidden_dim=None if args.hidden_dim is None else int(args.hidden_dim),
            embed_dim=None if args.embed_dim is None else int(args.embed_dim),
        )
        info = load_state_dict_forgiving(bundle.actor, mixed_state, strict=True)
        bundle.actor.to(device).eval()

        summary, rows = evaluate_structured_actor_exec_sources(
            cfg,
            bundle.actor,
            device=device,
            episodes=int(args.episodes),
            episode_seed_base=int(args.episode_seed_base),
            deterministic=str(args.policy_mode) != "stochastic",
            num_envs=int(args.num_envs),
            vec_backend="sync",
            exec_accel_source=str(args.exec_accel_source),
            exec_sat_source=str(args.exec_sat_source),
            exec_bw_source=str(args.exec_bw_source),
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    row_payload = {
        "label": str(args.label),
        "baseline_policy": str(args.baseline_policy or ""),
        "config": str(args.config),
        "base_checkpoint": str(args.base_checkpoint or ""),
        "sat_checkpoint": str(args.sat_checkpoint or ""),
        "accel_checkpoint": str(args.accel_checkpoint or ""),
        "bw_checkpoint": str(args.bw_checkpoint or ""),
        "replaced": replaced,
        "load_info": info,
        "episodes": int(args.episodes),
        "num_envs": int(args.num_envs),
        "episode_seed_base": int(args.episode_seed_base),
        "policy_mode": str(args.policy_mode),
        "exec_sources": {
            "accel": effective_exec_sources[0],
            "sat": effective_exec_sources[1],
            "bw": effective_exec_sources[2],
        },
        "access_bw_decision_interval": int(getattr(cfg, "access_bw_decision_interval", 1)),
        "sat_decision_interval": int(getattr(cfg, "sat_decision_interval", 1)),
        "summary": summary,
    }
    with (out_dir / f"{args.label}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(row_payload, f, ensure_ascii=False, indent=2, default=str)
    with (out_dir / f"{args.label}_episodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"{args.label}: reward={summary['reward_sum']:.4f} "
        f"processed={summary['processed_ratio_eval']:.4f} "
        f"drop={summary['drop_ratio_eval']:.4f} "
        f"pre_backlog={summary['pre_backlog_steps_eval']:.4f} "
        f"sat_overlap={summary['sat_overlap_eval']:.4f} "
        f"collision={summary['collision_episode_fraction']:.4f}"
    )
    print(f"replaced={replaced} out_dir={out_dir}")


if __name__ == "__main__":
    main()
