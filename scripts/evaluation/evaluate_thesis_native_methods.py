from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_eval import (
    _evaluate_structured_baseline_policy_with_traces,
    _fixed_policy_exec_sources,
    evaluate_structured_actor_exec_sources,
)
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.utils.checkpoint import load_state_dict_forgiving




THESIS_BASELINES = [
    ("static_uniform", "静止基准"),
    ("random_feasible", "随机基准"),
    ("link_priority", "链路优先规则"),
    ("demand_priority", "需求优先规则"),
    ("cluster_center_queue_aware", "簇中心运动规则"),
    ("queue_aware", "队列感知规则"),
]

MAXWEIGHT_BASELINE_ID = "maxweight_lyapunov"


DPP_ABLATION_BASELINES = [
    ("dpp_no_mobility", "MaxWeight/Lyapunov 无移动消融"),
    ("dpp_equal_bw", "MaxWeight/Lyapunov 等带宽消融"),
    ("dpp_greedy_sat", "MaxWeight/Lyapunov heuristic SAT 消融"),
    ("topology_dpp", "拓扑感知 one-step DPP"),
]


LYAPUNOV_CANDIDATES: list[dict[str, float]] = [
    {},
    {"baseline_lyapunov_urgency_alpha": 5.0},
    {"baseline_lyapunov_urgency_alpha": 6.0},
    {"baseline_lyapunov_urgency_alpha": 8.0},
    {
        "baseline_accel_gain": 0.5,
        "baseline_repulse_gain": 48.0,
        "baseline_repulse_radius_factor": 10.0,
        "baseline_lyapunov_urgency_alpha": 5.0,
        "baseline_cluster_cruise_speed": 15.0,
    },
    {
        "baseline_accel_gain": 0.5,
        "baseline_repulse_gain": 48.0,
        "baseline_repulse_radius_factor": 10.0,
        "baseline_lyapunov_urgency_alpha": 5.0,
        "baseline_cluster_cruise_speed": 18.0,
    },
    {
        "baseline_accel_gain": 0.5,
        "baseline_repulse_gain": 48.0,
        "baseline_repulse_radius_factor": 10.0,
        "baseline_lyapunov_urgency_alpha": 5.0,
        "baseline_cluster_cruise_speed": 20.0,
    },
    {
        "baseline_accel_gain": 0.5,
        "baseline_repulse_gain": 48.0,
        "baseline_repulse_radius_factor": 10.0,
        "baseline_lyapunov_urgency_alpha": 5.0,
        "baseline_cluster_cruise_speed": 22.0,
    },
    {
        "baseline_accel_gain": 0.5,
        "baseline_repulse_gain": 48.0,
        "baseline_repulse_radius_factor": 10.0,
        "baseline_lyapunov_urgency_alpha": 5.0,
        "baseline_cluster_cruise_speed": 25.0,
    },
    {
        "baseline_accel_gain": 0.5,
        "baseline_repulse_gain": 48.0,
        "baseline_repulse_radius_factor": 10.0,
        "baseline_lyapunov_urgency_alpha": 6.0,
        "baseline_cluster_cruise_speed": 18.0,
    },
    {
        "baseline_accel_gain": 0.5,
        "baseline_repulse_gain": 48.0,
        "baseline_repulse_radius_factor": 10.0,
        "baseline_lyapunov_urgency_alpha": 6.0,
        "baseline_cluster_cruise_speed": 20.0,
    },
    {
        "baseline_accel_gain": 0.5,
        "baseline_repulse_gain": 48.0,
        "baseline_repulse_radius_factor": 10.0,
        "baseline_lyapunov_urgency_alpha": 8.0,
        "baseline_cluster_cruise_speed": 20.0,
    },
    {
        "baseline_accel_gain": 0.4,
        "baseline_repulse_gain": 32.0,
        "baseline_repulse_radius_factor": 10.0,
        "baseline_lyapunov_urgency_alpha": 5.0,
        "baseline_cluster_cruise_speed": 20.0,
    },
    {
        "baseline_accel_gain": 0.4,
        "baseline_repulse_gain": 48.0,
        "baseline_repulse_radius_factor": 10.0,
        "baseline_lyapunov_urgency_alpha": 5.0,
        "baseline_cluster_cruise_speed": 20.0,
    },
    {
        "baseline_accel_gain": 0.6,
        "baseline_repulse_gain": 48.0,
        "baseline_repulse_radius_factor": 10.0,
        "baseline_lyapunov_urgency_alpha": 5.0,
        "baseline_cluster_cruise_speed": 20.0,
    },
    {
        "baseline_accel_gain": 0.5,
        "baseline_repulse_gain": 32.0,
        "baseline_repulse_radius_factor": 8.0,
        "baseline_lyapunov_urgency_alpha": 5.0,
        "baseline_cluster_cruise_speed": 20.0,
    },
    {
        "baseline_accel_gain": 0.5,
        "baseline_repulse_gain": 64.0,
        "baseline_repulse_radius_factor": 12.0,
        "baseline_lyapunov_urgency_alpha": 5.0,
        "baseline_cluster_cruise_speed": 20.0,
    },
    {
        "baseline_accel_gain": 0.5,
        "baseline_repulse_gain": 48.0,
        "baseline_repulse_radius_factor": 10.0,
        "baseline_lyapunov_urgency_alpha": 5.0,
        "baseline_cluster_cruise_speed": 20.0,
        "baseline_cluster_slow_radius": 80.0,
    },
    {
        "baseline_accel_gain": 0.5,
        "baseline_repulse_gain": 48.0,
        "baseline_repulse_radius_factor": 10.0,
        "baseline_lyapunov_urgency_alpha": 5.0,
        "baseline_cluster_cruise_speed": 20.0,
        "baseline_cluster_stop_radius": 10.0,
    },
]


PREFERRED_FIELDS = [
    "method_id",
    "method_name",
    "episodes",
    "reward_sum",
    "processed_ratio_eval",
    "drop_ratio_eval",
    "pre_backlog_steps_eval",
    "D_sys_report",
    "gu_queue_mean",
    "uav_queue_mean",
    "sat_queue_mean",
    "queue_total_mean",
    "x_acc_mean",
    "x_rel_mean",
    "outflow_arrival_ratio",
    "sat_incoming_arrival_ratio",
    "sat_processed_arrival_ratio",
    "drop_ratio",
    "collision_episode_fraction",
    "terminated_early",
    "episode_length",
    "sat_overlap_eval",
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


def _actor_state_from_stage_checkpoints(
    *,
    accel_checkpoint: Path,
    sat_checkpoint: Path,
    bw_checkpoint: Path,
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    expected = {
        "accel": ("accel_policy.", accel_checkpoint),
        "sat": ("sat_subset_policy.", sat_checkpoint),
        "bw": ("bw_policy.", bw_checkpoint),
    }
    for stage_name, (prefix, path) in expected.items():
        stage_state = _actor_state_from_checkpoint(path)
        if not stage_state:
            raise ValueError(f"{stage_name} checkpoint {path} has an empty actor state")
        bad_keys = [key for key in stage_state if not str(key).startswith(prefix)]
        if bad_keys:
            raise ValueError(
                f"{stage_name} checkpoint {path} contains keys outside prefix {prefix!r}: "
                f"{bad_keys[:5]}"
            )
        overlap = set(state).intersection(stage_state)
        if overlap:
            raise ValueError(f"stage checkpoints contain overlapping actor keys: {sorted(overlap)[:5]}")
        state.update({key: value.detach().clone() for key, value in stage_state.items()})
    return state


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    seen = set(PREFERRED_FIELDS)
    extra: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                extra.append(str(key))
    return [key for key in PREFERRED_FIELDS if any(key in row for row in rows)] + extra


def _apply_overrides(cfg, overrides: dict[str, float]) -> None:
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"unknown config field for Lyapunov tuning: {key}")
        setattr(cfg, key, float(value))


def _load_lyapunov_params(*, params_json: str | None, params_file: str | None) -> dict[str, float]:
    payload: dict[str, Any] = {}
    if params_file:
        with Path(params_file).open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict) and isinstance(loaded.get("best_params"), dict):
            loaded = loaded["best_params"]
        if not isinstance(loaded, dict):
            raise TypeError(f"Lyapunov params file {params_file} must contain a JSON object.")
        payload.update(loaded)
    if params_json:
        loaded = json.loads(str(params_json))
        if not isinstance(loaded, dict):
            raise TypeError("--lyapunov_params_json must be a JSON object.")
        payload.update(loaded)
    return {str(key): float(value) for key, value in payload.items()}


def _lyapunov_score(summary: dict[str, float]) -> float:
    reward = float(summary.get("reward_sum", 0.0))
    collision = float(summary.get("collision_episode_fraction", 0.0))
    early = float(summary.get("terminated_early", 0.0))
    return reward - 500.0 * collision - 50.0 * early


def _tune_lyapunov(
    cfg,
    *,
    episodes: int,
    num_envs: int,
    device: torch.device,
    episode_seed_base: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    tune_rows: list[dict[str, Any]] = []
    best_params: dict[str, float] = {}
    best_score = -float("inf")
    for idx, params in enumerate(LYAPUNOV_CANDIDATES):
        candidate_cfg = copy.deepcopy(cfg)
        _apply_overrides(candidate_cfg, params)
        summary, _ = _evaluate_baseline(
            candidate_cfg,
            baseline_policy=MAXWEIGHT_BASELINE_ID,
            episodes=max(int(episodes), 1),
            num_envs=int(num_envs),
            device=device,
            episode_seed_base=int(episode_seed_base),
        )
        score = _lyapunov_score(summary)
        row = {"candidate": idx, "score": score, **params, **summary}
        tune_rows.append(row)
        print(
            f"  cand={idx:02d} score={score:.4f} reward={summary.get('reward_sum', 0.0):.4f} "
            f"processed={summary.get('processed_ratio_eval', 0.0):.4f} "
            f"drop={summary.get('drop_ratio_eval', 0.0):.4f} params={params}"
        )
        if score > best_score:
            best_score = score
            best_params = dict(params)
    return best_params, tune_rows


def _evaluate_policy(
    cfg,
    *,
    checkpoint: Path | None,
    accel_checkpoint: Path | None = None,
    sat_checkpoint: Path | None = None,
    bw_checkpoint: Path | None = None,
    device: torch.device,
    episodes: int,
    num_envs: int,
    episode_seed_base: int,
    deterministic: bool,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    bundle = build_structured_modules_from_config(cfg)
    stage_paths = [accel_checkpoint, sat_checkpoint, bw_checkpoint]
    if any(path is not None for path in stage_paths):
        if not all(path is not None for path in stage_paths):
            raise ValueError("stage-best evaluation requires accel/sat/bw checkpoints together")
        actor_state = _actor_state_from_stage_checkpoints(
            accel_checkpoint=accel_checkpoint,
            sat_checkpoint=sat_checkpoint,
            bw_checkpoint=bw_checkpoint,
        )
    else:
        if checkpoint is None:
            raise ValueError("--checkpoint is required when stage-best checkpoints are not provided")
        actor_state = _actor_state_from_checkpoint(checkpoint)
    info = load_state_dict_forgiving(bundle.actor, actor_state, strict=True)
    if info.get("missing_keys") or info.get("unexpected_keys"):
        raise RuntimeError(f"checkpoint load mismatch: {info}")
    bundle.actor.to(device).eval()
    return evaluate_structured_actor_exec_sources(
        cfg,
        bundle.actor,
        device=device,
        episodes=int(episodes),
        episode_seed_base=int(episode_seed_base),
        deterministic=bool(deterministic),
        num_envs=int(num_envs),
        vec_backend="sync",
        exec_accel_source="policy",
        exec_sat_source="policy",
        exec_bw_source="policy",
    )


def _evaluate_baseline(
    cfg,
    *,
    baseline_policy: str,
    episodes: int,
    num_envs: int,
    device: torch.device,
    episode_seed_base: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    exec_sources = _fixed_policy_exec_sources(str(baseline_policy))
    if exec_sources is None:
        summary, rows, _traces, _actions, _reset_rollouts = _evaluate_structured_baseline_policy_with_traces(
            cfg,
            baseline_policy=str(baseline_policy),
            episodes=int(episodes),
            episode_seed_base=int(episode_seed_base),
        )
        return summary, rows
    dummy_actor = torch.nn.Linear(1, 1).to(device)
    summary, rows = evaluate_structured_actor_exec_sources(
        cfg,
        dummy_actor,
        device=device,
        episodes=int(episodes),
        num_envs=int(num_envs),
        episode_seed_base=int(episode_seed_base),
        deterministic=True,
        vec_backend="sync",
        exec_accel_source=exec_sources[0],
        exec_sat_source=exec_sources[1],
        exec_bw_source=exec_sources[2],
    )
    return summary, rows


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--accel_checkpoint", default=None)
    parser.add_argument("--sat_checkpoint", default=None)
    parser.add_argument("--bw_checkpoint", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--episode_seed_base", type=int, default=900000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--policy_mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--access_bw_decision_interval", type=int, default=None)
    parser.add_argument("--sat_decision_interval", type=int, default=None)
    parser.add_argument("--lyapunov_tune_episodes", type=int, default=16)
    parser.add_argument("--skip_lyapunov_tune", action="store_true")
    parser.add_argument("--only_lyapunov_tune", action="store_true")
    parser.add_argument("--lyapunov_params_json", default=None)
    parser.add_argument("--lyapunov_params_file", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.access_bw_decision_interval is not None:
        cfg.access_bw_decision_interval = max(int(args.access_bw_decision_interval), 1)
    if args.sat_decision_interval is not None:
        cfg.sat_decision_interval = max(int(args.sat_decision_interval), 1)

    requested = str(args.device).lower()
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    else:
        device = torch.device(requested)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.only_lyapunov_tune:
        print("[tune] lyapunov")
        best_params, tune_rows = _tune_lyapunov(
            cfg,
            episodes=max(int(args.lyapunov_tune_episodes), 1),
            num_envs=int(args.num_envs),
            device=device,
            episode_seed_base=int(args.episode_seed_base),
        )
        _write_rows(out_dir / "lyapunov_tune.csv", tune_rows)
        with (out_dir / "lyapunov_tune.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": str(args.config),
                    "episodes": int(args.lyapunov_tune_episodes),
                    "num_envs": int(args.num_envs),
                    "episode_seed_base": int(args.episode_seed_base),
                    "best_params": best_params,
                    "rows": tune_rows,
                },
                f,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        print(f"best_params={best_params}")
        print(f"wrote {out_dir}")
        return

    method_summaries: list[dict[str, Any]] = []
    all_episode_rows: list[dict[str, Any]] = []
    total_methods = 1 + len(THESIS_BASELINES) + 1 + len(DPP_ABLATION_BASELINES)
    preset_lyapunov_params = _load_lyapunov_params(
        params_json=args.lyapunov_params_json,
        params_file=args.lyapunov_params_file,
    )

    print(f"[1/{total_methods}] proposed")
    has_stage_best = bool(args.accel_checkpoint and args.sat_checkpoint and args.bw_checkpoint)
    has_partial_stage_best = bool(args.accel_checkpoint or args.sat_checkpoint or args.bw_checkpoint) and not has_stage_best
    if has_partial_stage_best:
        raise ValueError("--accel_checkpoint, --sat_checkpoint, and --bw_checkpoint must be provided together")
    if not args.checkpoint and not has_stage_best:
        raise ValueError("--checkpoint or all three stage-best checkpoints are required unless --only_lyapunov_tune is set")
    proposed_summary, proposed_rows = _evaluate_policy(
        cfg,
        checkpoint=None if not args.checkpoint else Path(args.checkpoint),
        accel_checkpoint=None if not args.accel_checkpoint else Path(args.accel_checkpoint),
        sat_checkpoint=None if not args.sat_checkpoint else Path(args.sat_checkpoint),
        bw_checkpoint=None if not args.bw_checkpoint else Path(args.bw_checkpoint),
        device=device,
        episodes=int(args.episodes),
        num_envs=int(args.num_envs),
        episode_seed_base=int(args.episode_seed_base),
        deterministic=str(args.policy_mode) != "stochastic",
    )
    proposed_summary_row = {"method_id": "proposed", "method_name": "本文方法", **proposed_summary}
    method_summaries.append(proposed_summary_row)
    all_episode_rows.extend({"method_id": "proposed", "method_name": "本文方法", **row} for row in proposed_rows)

    for index, (method_id, method_name) in enumerate(THESIS_BASELINES, start=2):
        print(f"[{index}/{total_methods}] {method_id}")
        summary, rows = _evaluate_baseline(
            cfg,
            baseline_policy=method_id,
            episodes=int(args.episodes),
            num_envs=int(args.num_envs),
            device=device,
            episode_seed_base=int(args.episode_seed_base),
        )
        method_summaries.append({"method_id": method_id, "method_name": method_name, **summary})
        all_episode_rows.extend({"method_id": method_id, "method_name": method_name, **row} for row in rows)

    lyapunov_cfg = copy.deepcopy(cfg)
    tune_rows: list[dict[str, Any]] = []
    best_params: dict[str, float] = dict(preset_lyapunov_params)
    if not args.skip_lyapunov_tune:
        print("[tune] lyapunov")
        best_params, tune_rows = _tune_lyapunov(
            cfg,
            episodes=max(int(args.lyapunov_tune_episodes), 1),
            num_envs=int(args.num_envs),
            device=device,
            episode_seed_base=int(args.episode_seed_base),
        )
        _apply_overrides(lyapunov_cfg, best_params)
        _write_rows(out_dir / "lyapunov_tune.csv", tune_rows)
    elif best_params:
        _apply_overrides(lyapunov_cfg, best_params)
    maxweight_index = 2 + len(THESIS_BASELINES)
    print(f"[{maxweight_index}/{total_methods}] {MAXWEIGHT_BASELINE_ID}")
    lyapunov_summary, lyapunov_rows = _evaluate_baseline(
        lyapunov_cfg,
        baseline_policy=MAXWEIGHT_BASELINE_ID,
        episodes=int(args.episodes),
        num_envs=int(args.num_envs),
        device=device,
        episode_seed_base=int(args.episode_seed_base),
    )
    method_summaries.append(
        {
            "method_id": MAXWEIGHT_BASELINE_ID,
            "method_name": "MaxWeight/Lyapunov 队列方法",
            "lyapunov_params": json.dumps(best_params, ensure_ascii=False, sort_keys=True),
            **lyapunov_summary,
        }
    )
    all_episode_rows.extend(
        {"method_id": MAXWEIGHT_BASELINE_ID, "method_name": "MaxWeight/Lyapunov 队列方法", **row}
        for row in lyapunov_rows
    )

    for offset, (method_id, method_name) in enumerate(DPP_ABLATION_BASELINES, start=1):
        index = maxweight_index + offset
        print(f"[{index}/{total_methods}] {method_id}")
        summary, rows = _evaluate_baseline(
            lyapunov_cfg,
            baseline_policy=method_id,
            episodes=int(args.episodes),
            num_envs=int(args.num_envs),
            device=device,
            episode_seed_base=int(args.episode_seed_base),
        )
        method_summaries.append(
            {
                "method_id": method_id,
                "method_name": method_name,
                "lyapunov_params": json.dumps(best_params, ensure_ascii=False, sort_keys=True),
                **summary,
            }
        )
        all_episode_rows.extend({"method_id": method_id, "method_name": method_name, **row} for row in rows)

    _write_rows(out_dir / "method_summaries.csv", method_summaries)
    _write_rows(out_dir / "all_episodes.csv", all_episode_rows)
    with (out_dir / "method_summaries.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": str(args.config),
                "checkpoint": str(args.checkpoint),
                "accel_checkpoint": str(args.accel_checkpoint or ""),
                "sat_checkpoint": str(args.sat_checkpoint or ""),
                "bw_checkpoint": str(args.bw_checkpoint or ""),
                "episodes": int(args.episodes),
                "num_envs": int(args.num_envs),
                "episode_seed_base": int(args.episode_seed_base),
                "access_bw_decision_interval": int(getattr(cfg, "access_bw_decision_interval", 1)),
                "sat_decision_interval": int(getattr(cfg, "sat_decision_interval", 1)),
                "lyapunov_best_params": best_params,
                "summaries": method_summaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
