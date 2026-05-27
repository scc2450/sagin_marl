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
from sagin_marl.rl.structured_eval import evaluate_structured_actor_exec_sources
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.utils.checkpoint import load_state_dict_forgiving


STAGES = {
    "accel": ("accel_policy.", "accel_checkpoint"),
    "sat": ("sat_subset_policy.", "sat_checkpoint"),
    "bw": ("bw_policy.", "bw_checkpoint"),
}

CASES = [
    ("rule_ref", "规则补齐参考", (False, False, False)),
    ("learn_accel_only", "仅学习运动控制", (True, False, False)),
    ("learn_bw_only", "仅学习接入带宽分配", (False, False, True)),
    ("learn_sat_only", "仅学习卫星回传选择", (False, True, False)),
    ("learn_all_heads", "完整三动作头", (True, True, True)),
]


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


def _load_selected_heads(actor: torch.nn.Module, selected: tuple[bool, bool, bool], paths: dict[str, Path]) -> dict[str, Any]:
    selected_state: dict[str, Any] = {}
    stage_names = ("accel", "sat", "bw")
    for use_stage, stage_name in zip(selected, stage_names):
        if not use_stage:
            continue
        prefix, _ = STAGES[stage_name]
        state = _actor_state_from_checkpoint(paths[stage_name])
        bad = [key for key in state if not str(key).startswith(prefix)]
        if bad:
            raise ValueError(f"{paths[stage_name]} has keys outside {prefix!r}: {bad[:5]}")
        overlap = set(selected_state).intersection(state)
        if overlap:
            raise ValueError(f"overlapping actor keys: {sorted(overlap)[:5]}")
        selected_state.update({key: value.detach().clone() for key, value in state.items()})
    info = load_state_dict_forgiving(actor, selected_state, strict=False)
    loaded = sorted(selected_state)
    return {
        "loaded_key_count": len(loaded),
        "loaded_prefixes": [STAGES[name][0] for flag, name in zip(selected, stage_names) if flag],
        "missing_key_count": len(info.get("missing_keys", [])),
        "unexpected_key_count": len(info.get("unexpected_keys", [])),
        "skipped_keys": info.get("skipped_keys", []),
    }


def _exec_sources(selected: tuple[bool, bool, bool]) -> tuple[str, str, str]:
    use_accel, use_sat, use_bw = selected
    return (
        "policy" if use_accel else "cluster_center_queue_aware",
        "policy" if use_sat else "queue_aware",
        "policy" if use_bw else "queue_aware",
    )


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "method_id",
        "method_name",
        "episodes",
        "reward_sum",
        "processed_ratio_eval",
        "drop_ratio_eval",
        "pre_backlog_steps_eval",
        "D_sys_report",
        "collision_episode_fraction",
        "episode_length",
        "x_acc_mean",
        "x_rel_mean",
        "outflow_arrival_ratio",
        "sat_incoming_arrival_ratio",
        "sat_processed_arrival_ratio",
        "gu_queue_mean",
        "uav_queue_mean",
        "sat_queue_mean",
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
    parser.add_argument("--accel_checkpoint", required=True)
    parser.add_argument("--sat_checkpoint", required=True)
    parser.add_argument("--bw_checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--episode_seed_base", type=int, default=903000)
    parser.add_argument("--access_bw_decision_interval", type=int, default=5)
    parser.add_argument("--sat_decision_interval", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.access_bw_decision_interval = max(int(args.access_bw_decision_interval), 1)
    cfg.sat_decision_interval = max(int(args.sat_decision_interval), 1)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    else:
        device = torch.device(args.device)

    paths = {
        "accel": Path(args.accel_checkpoint),
        "sat": Path(args.sat_checkpoint),
        "bw": Path(args.bw_checkpoint),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []

    for method_id, method_name, selected in CASES:
        bundle = build_structured_modules_from_config(cfg)
        load_info = _load_selected_heads(bundle.actor, selected, paths)
        bundle.actor.to(device).eval()
        exec_sources = _exec_sources(selected)
        summary, rows = evaluate_structured_actor_exec_sources(
            cfg,
            bundle.actor,
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
        row = {"method_id": method_id, "method_name": method_name, **summary}
        summaries.append(row)
        all_rows.extend({"method_id": method_id, "method_name": method_name, **ep} for ep in rows)
        payload = {
            "method_id": method_id,
            "method_name": method_name,
            "selected_heads": {
                "accel": bool(selected[0]),
                "sat": bool(selected[1]),
                "bw": bool(selected[2]),
            },
            "exec_sources": {
                "accel": exec_sources[0],
                "sat": exec_sources[1],
                "bw": exec_sources[2],
            },
            "load_info": load_info,
            "summary": summary,
        }
        payloads.append(payload)
        (out_dir / f"{method_id}_summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        with (out_dir / f"{method_id}_episodes.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_fieldnames(rows))
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"{method_id}: reward={summary['reward_sum']:.4f} "
            f"processed={summary['processed_ratio_eval']:.4f} "
            f"drop={summary['drop_ratio_eval']:.4f} "
            f"pre_backlog={summary['pre_backlog_steps_eval']:.4f} "
            f"collision={summary['collision_episode_fraction']:.4f} "
            f"exec={exec_sources}"
        )

    with (out_dir / "action_head_ablation_summaries.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_fieldnames(summaries))
        writer.writeheader()
        writer.writerows(summaries)
    with (out_dir / "action_head_ablation_episodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_fieldnames(all_rows))
        writer.writeheader()
        writer.writerows(all_rows)
    (out_dir / "action_head_ablation_payload.json").write_text(
        json.dumps(
            {
                "config": str(args.config),
                "checkpoints": {name: str(path) for name, path in paths.items()},
                "episodes": int(args.episodes),
                "num_envs": int(args.num_envs),
                "episode_seed_base": int(args.episode_seed_base),
                "access_bw_decision_interval": int(cfg.access_bw_decision_interval),
                "sat_decision_interval": int(cfg.sat_decision_interval),
                "methods": payloads,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
