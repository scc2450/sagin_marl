from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.experiments.bw_training.evaluate_structured_bw_select import (
    _build_candidate_panels,
    _episode_metrics_template,
    _episode_row_from_accumulator,
    _new_episode_accumulator,
    _prepare_bw_stage_many,
)
from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import _normalize_exec_source
from sagin_marl.rl.structured_parallel_eval import (
    batched_policy_accel_actions,
    last_reward_parts_many,
    looks_like_driver_group,
    reset_at,
    reset_many,
)
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_env_group
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _resolve_checkpoint(run_dir: Path, checkpoint: str | None, update: int | None) -> Path:
    if checkpoint:
        return Path(checkpoint)
    if update is not None:
        return run_dir / f"actor_u{int(update):04d}.pt"
    final_ckpt = run_dir / "actor_final.pt"
    if final_ckpt.exists():
        return final_ckpt
    return run_dir / "actor.pt"


def _load_actor(cfg, checkpoint: Path, *, hidden_dim: int, embed_dim: int, device: torch.device):
    bundle = build_structured_modules_from_config(cfg, hidden_dim=hidden_dim, embed_dim=embed_dim)
    load_checkpoint_forgiving(bundle.actor, str(checkpoint), map_location=device, strict=True)
    bundle.actor.to(device).eval()
    return bundle.actor


def _source_bucket(name: str) -> np.ndarray:
    lowered = str(name).strip().lower()
    out = np.zeros((6,), dtype=np.float32)
    if lowered == "latent_det":
        out[0] = 1.0
    elif lowered == "heuristic":
        out[1] = 1.0
    elif lowered.startswith("sample_"):
        out[2] = 1.0
    elif lowered.startswith("random_"):
        out[3] = 1.0
    elif lowered == "simplex_det":
        out[4] = 1.0
    else:
        out[5] = 1.0
    return out


def _infer_local_state_field_shapes(entries: list[dict[str, Any]]) -> dict[str, tuple[int, ...]]:
    field_shapes: dict[str, list[int]] = {}
    for entry in entries:
        local_state = entry["local_state"]
        if not is_dataclass(local_state):
            raise TypeError("local_state must be a dataclass")
        for field in fields(local_state):
            value = getattr(local_state, field.name)
            if torch.is_tensor(value):
                arr = value.detach().cpu().numpy()
            else:
                arr = np.asarray(value)
            shape = tuple(int(dim) for dim in arr.shape)
            prev = field_shapes.get(field.name)
            if prev is None:
                field_shapes[field.name] = list(shape)
                continue
            if len(prev) != len(shape):
                raise ValueError(f"inconsistent rank for local_state field {field.name}: {tuple(prev)} vs {shape}")
            for dim_idx, dim in enumerate(shape):
                prev[dim_idx] = max(int(prev[dim_idx]), int(dim))
    return {name: tuple(int(dim) for dim in dims) for name, dims in field_shapes.items()}


def _pad_to_shape(arr: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    if arr.shape == target_shape:
        return arr
    if arr.ndim != len(target_shape):
        raise ValueError(f"rank mismatch: arr.shape={arr.shape}, target_shape={target_shape}")
    padded = np.zeros(target_shape, dtype=np.float32)
    slices = tuple(slice(0, min(int(src), int(dst))) for src, dst in zip(arr.shape, target_shape))
    padded[slices] = arr[slices].astype(np.float32, copy=False)
    return padded


def _flatten_local_state(local_state: Any, field_shapes: dict[str, tuple[int, ...]] | None = None) -> np.ndarray:
    if not is_dataclass(local_state):
        raise TypeError("local_state must be a dataclass")
    pieces: list[np.ndarray] = []
    for field in fields(local_state):
        value = getattr(local_state, field.name)
        if torch.is_tensor(value):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)
        if field_shapes is not None and field.name in field_shapes:
            arr = _pad_to_shape(arr.astype(np.float32, copy=False), field_shapes[field.name])
        pieces.append(arr.astype(np.float32, copy=False).reshape(-1))
    return np.concatenate(pieces, axis=0).astype(np.float32, copy=False)


def _build_context_and_candidates(
    entry: dict[str, Any],
    field_shapes: dict[str, tuple[int, ...]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    local_state = _flatten_local_state(entry["local_state"], field_shapes)
    candidate_actions = [np.asarray(action, dtype=np.float32).reshape(-1) for action in entry["candidate_actions"]]
    candidate_scores = np.asarray(entry["candidate_scores"], dtype=np.float32)
    selected_idx = int(entry["selected_idx"])
    policy_det_idx = int(entry["policy_det_idx"])
    heuristic_idx = None if entry.get("heuristic_idx") is None else int(entry["heuristic_idx"])
    policy_det_action = candidate_actions[policy_det_idx]
    heuristic_action = np.zeros_like(policy_det_action) if heuristic_idx is None else candidate_actions[heuristic_idx]
    context = np.concatenate([local_state, policy_det_action, heuristic_action], axis=0).astype(np.float32, copy=False)
    cand_features: list[np.ndarray] = []
    for name, action in zip(entry["candidate_names"], candidate_actions):
        cand_features.append(
            np.concatenate(
                [action, action - policy_det_action, action - heuristic_action, _source_bucket(str(name))],
                axis=0,
            ).astype(np.float32, copy=False)
        )
    return context, np.stack(cand_features, axis=0), candidate_scores, np.asarray(selected_idx, dtype=np.int64)


def _choose_holdout_episodes(entries: list[dict[str, Any]], holdout_episodes: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unique_episodes = sorted({int(entry["episode"]) for entry in entries})
    if not unique_episodes:
        return [], []
    holdout_count = min(max(int(holdout_episodes), 1), max(len(unique_episodes) - 1, 1))
    rng = np.random.default_rng(int(seed))
    holdout_episode_set = set(rng.choice(unique_episodes, size=holdout_count, replace=False).tolist())
    train_entries = [entry for entry in entries if int(entry["episode"]) not in holdout_episode_set]
    holdout_entries = [entry for entry in entries if int(entry["episode"]) in holdout_episode_set]
    if not train_entries or not holdout_entries:
        split = max(1, len(entries) // 5)
        holdout_entries = list(entries[:split])
        train_entries = list(entries[split:])
    return train_entries, holdout_entries


class BwSelectScorer(nn.Module):
    def __init__(self, context_dim: int, candidate_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.context_net = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.candidate_net = nn.Sequential(
            nn.LayerNorm(candidate_dim),
            nn.Linear(candidate_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, context: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
        context_embed = self.context_net(context).unsqueeze(1).expand(-1, candidate_features.shape[1], -1)
        candidate_embed = self.candidate_net(candidate_features)
        return self.head(torch.cat([context_embed, candidate_embed], dim=-1)).squeeze(-1)


def _prepare_split(entries: list[dict[str, Any]], field_shapes: dict[str, tuple[int, ...]]) -> dict[str, Any]:
    contexts: list[np.ndarray] = []
    candidate_features: list[np.ndarray] = []
    candidate_scores: list[np.ndarray] = []
    selected_idx: list[int] = []
    rows: list[dict[str, Any]] = []
    for entry in entries:
        context, cand_feat, cand_scores, target = _build_context_and_candidates(entry, field_shapes)
        contexts.append(context)
        candidate_features.append(cand_feat)
        candidate_scores.append(cand_scores)
        selected_idx.append(int(target))
        rows.append(
            {
                "episode": int(entry["episode"]),
                "t": int(entry["t"]),
                "selected_name": str(entry["selected_name"]),
                "policy_det_idx": int(entry["policy_det_idx"]),
                "heuristic_idx": None if entry.get("heuristic_idx") is None else int(entry["heuristic_idx"]),
            }
        )
    return {
        "contexts": np.stack(contexts, axis=0).astype(np.float32),
        "candidate_features": np.stack(candidate_features, axis=0).astype(np.float32),
        "candidate_scores": np.stack(candidate_scores, axis=0).astype(np.float32),
        "selected_idx": np.asarray(selected_idx, dtype=np.int64),
        "rows": rows,
    }


def _tensor_split(split: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "contexts": torch.as_tensor(split["contexts"], dtype=torch.float32, device=device),
        "candidate_features": torch.as_tensor(split["candidate_features"], dtype=torch.float32, device=device),
        "candidate_scores": torch.as_tensor(split["candidate_scores"], dtype=torch.float32, device=device),
        "selected_idx": torch.as_tensor(split["selected_idx"], dtype=torch.long, device=device),
    }


def _soft_target(candidate_scores: torch.Tensor, temperature: float) -> torch.Tensor:
    centered = candidate_scores - candidate_scores.max(dim=1, keepdim=True).values
    return torch.softmax(centered / float(temperature), dim=1)


def _evaluate_offline(
    *,
    model: BwSelectScorer,
    split_name: str,
    split: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tensors = _tensor_split(split, device)
    with torch.inference_mode():
        logits = model(tensors["contexts"], tensors["candidate_features"])
        pred_idx = torch.argmax(logits, dim=1)
    pred_idx_np = pred_idx.detach().cpu().numpy().astype(np.int64)
    scores_np = split["candidate_scores"]
    selected_idx_np = split["selected_idx"]
    rows: list[dict[str, Any]] = []
    pred_scores: list[float] = []
    winner_scores: list[float] = []
    base_scores: list[float] = []
    heuristic_scores: list[float] = []
    pred_gain_vs_base: list[float] = []
    pred_gain_vs_heuristic: list[float] = []
    for idx, row in enumerate(split["rows"]):
        candidate_scores = scores_np[idx]
        chosen = int(pred_idx_np[idx])
        winner = int(selected_idx_np[idx])
        base_idx = int(row["policy_det_idx"])
        heuristic_idx = row["heuristic_idx"]
        pred_score = float(candidate_scores[chosen])
        winner_score = float(candidate_scores[winner])
        base_score = float(candidate_scores[base_idx])
        heuristic_score = base_score if heuristic_idx is None else float(candidate_scores[int(heuristic_idx)])
        pred_scores.append(pred_score)
        winner_scores.append(winner_score)
        base_scores.append(base_score)
        heuristic_scores.append(heuristic_score)
        pred_gain_vs_base.append(pred_score - base_score)
        pred_gain_vs_heuristic.append(pred_score - heuristic_score)
        rows.append(
            {
                "split": split_name,
                "episode": int(row["episode"]),
                "t": int(row["t"]),
                "pred_idx": int(chosen),
                "target_idx": int(winner),
                "pred_score": pred_score,
                "winner_score": winner_score,
                "base_score": base_score,
                "heuristic_score": heuristic_score,
                "pred_gain_vs_base": float(pred_score - base_score),
                "pred_gain_vs_heuristic": float(pred_score - heuristic_score),
                "top1_hit": float(chosen == winner),
            }
        )
    pred_mean = float(np.mean(np.asarray(pred_scores, dtype=np.float64)))
    base_mean = float(np.mean(np.asarray(base_scores, dtype=np.float64)))
    winner_mean = float(np.mean(np.asarray(winner_scores, dtype=np.float64)))
    capture_ratio = (pred_mean - base_mean) / max(winner_mean - base_mean, 1.0e-8)
    summary = {
        "state_count": int(len(rows)),
        "pred_score": _summarize(pred_scores),
        "winner_score": _summarize(winner_scores),
        "base_score": _summarize(base_scores),
        "heuristic_score": _summarize(heuristic_scores),
        "pred_gain_vs_base": _summarize(pred_gain_vs_base),
        "pred_gain_vs_heuristic": _summarize(pred_gain_vs_heuristic),
        "top1_acc": float(np.mean(np.asarray([row["top1_hit"] for row in rows], dtype=np.float64))),
        "pred_beats_base_frac": float(np.mean(np.asarray([v > 1.0e-6 for v in pred_gain_vs_base], dtype=np.float64))),
        "pred_beats_heuristic_frac": float(np.mean(np.asarray([v > 1.0e-6 for v in pred_gain_vs_heuristic], dtype=np.float64))),
        "capture_ratio": float(capture_ratio),
    }
    return summary, rows


def _live_context_features(
    local_state: Any,
    candidate_actions: list[np.ndarray],
    policy_det_idx: int,
    heuristic_idx: int | None,
    field_shapes: dict[str, tuple[int, ...]],
) -> tuple[np.ndarray, np.ndarray]:
    context_state = _flatten_local_state(local_state, field_shapes)
    policy_det_action = np.asarray(candidate_actions[int(policy_det_idx)], dtype=np.float32).reshape(-1)
    heuristic_action = (
        np.zeros_like(policy_det_action)
        if heuristic_idx is None
        else np.asarray(candidate_actions[int(heuristic_idx)], dtype=np.float32).reshape(-1)
    )
    context = np.concatenate([context_state, policy_det_action, heuristic_action], axis=0).astype(np.float32, copy=False)
    cand_feats: list[np.ndarray] = []
    for action in candidate_actions:
        flat = np.asarray(action, dtype=np.float32).reshape(-1)
        cand_feats.append(
            np.concatenate([flat, flat - policy_det_action, flat - heuristic_action], axis=0).astype(np.float32, copy=False)
        )
    return context, np.stack(cand_feats, axis=0)


def _augment_candidate_features(cand_feats: np.ndarray, candidate_names: list[str]) -> np.ndarray:
    onehots = np.stack([_source_bucket(name) for name in candidate_names], axis=0)
    return np.concatenate([cand_feats, onehots.astype(np.float32)], axis=1)


def _evaluate_live(
    *,
    cfg,
    actor,
    model: BwSelectScorer,
    device: torch.device,
    episodes: int,
    episode_seed_base: int | None,
    num_envs: int,
    vec_backend: str,
    sample_count: int,
    random_count: int,
    include_heuristic: bool,
    include_simplex: bool,
    heuristic_policy: str,
    bw_deterministic_opt_steps: int,
    bw_deterministic_step_size: float,
    seed: int,
    field_shapes: dict[str, tuple[int, ...]],
    exec_accel_source: str,
    exec_sat_source: str,
) -> tuple[dict[str, Any], list[dict[str, float]], list[dict[str, Any]]]:
    active_slots = max(min(int(num_envs), int(episodes)), 1)
    env_group = make_structured_env_group(cfg, num_envs=active_slots, backend=vec_backend)
    drivers = env_group
    if not looks_like_driver_group(drivers):
        from scripts.experiments.bw_training.evaluate_structured_bw_select import _as_driver_list

        drivers = _as_driver_list(env_group)
    rows: list[dict[str, float]] = []
    select_rows: list[dict[str, Any]] = []
    slot_episode = list(range(active_slots))
    slot_active = [True for _ in range(active_slots)]
    slot_acc = [_new_episode_accumulator() for _ in range(active_slots)]
    next_episode = active_slots
    initial_seeds = [
        None if episode_seed_base is None else int(episode_seed_base) + slot
        for slot in range(active_slots)
    ]
    rng = np.random.default_rng(int(seed))
    reset_many(drivers, initial_seeds)
    try:
        while len(rows) < int(episodes):
            active_indices = [slot for slot, is_active in enumerate(slot_active) if is_active]
            if not active_indices:
                break
            if looks_like_driver_group(drivers):
                accel_world_states = drivers.prepare_accel_stage_many(indices=active_indices)
            else:
                accel_world_states = [drivers[slot].begin_step() for slot in active_indices]
            bw_snapshots, obs_after_accel_many, _snapshot_states = _prepare_bw_stage_many(
                drivers=drivers,
                indices=active_indices,
                accel_world_states=accel_world_states,
                actor=actor,
                cfg=cfg,
                device=device,
                heuristic_policy=heuristic_policy,
                exec_accel_source=exec_accel_source,
                exec_sat_source=exec_sat_source,
            )
            candidate_names_many, candidate_actions_many, local_states_many = _build_candidate_panels(
                actor=actor,
                cfg=cfg,
                device=device,
                bw_snapshots=bw_snapshots,
                obs_after_accel_many=obs_after_accel_many,
                heuristic_policy=heuristic_policy,
                include_heuristic=include_heuristic,
                include_simplex=include_simplex,
                sample_count=int(sample_count),
                random_count=int(random_count),
                bw_deterministic_opt_steps=int(bw_deterministic_opt_steps),
                bw_deterministic_step_size=float(bw_deterministic_step_size),
                rng=rng,
            )
            selected_actions: list[np.ndarray] = []
            for local_slot, slot in enumerate(active_indices):
                candidate_names = list(candidate_names_many[local_slot])
                candidate_actions = [np.asarray(action, dtype=np.float32) for action in candidate_actions_many[local_slot]]
                policy_det_idx = int(candidate_names.index("latent_det"))
                heuristic_idx = None if "heuristic" not in candidate_names else int(candidate_names.index("heuristic"))
                context, cand_feats = _live_context_features(
                    local_states_many[local_slot],
                    candidate_actions,
                    policy_det_idx,
                    heuristic_idx,
                    field_shapes,
                )
                cand_feats = _augment_candidate_features(cand_feats, candidate_names)
                with torch.inference_mode():
                    logits = model(
                        torch.as_tensor(context, dtype=torch.float32, device=device).unsqueeze(0),
                        torch.as_tensor(cand_feats, dtype=torch.float32, device=device).unsqueeze(0),
                    )[0]
                best_idx = int(torch.argmax(logits).item())
                selected_actions.append(candidate_actions[best_idx])
                select_rows.append(
                    {
                        "episode": int(slot_episode[slot]),
                        "slot": int(slot),
                        "selected_name": str(candidate_names[best_idx]),
                        "selected_idx": int(best_idx),
                    }
                )
            if looks_like_driver_group(drivers):
                step_results = drivers.execute_stage_bw_and_step_many(selected_actions, indices=active_indices)
            else:
                step_results = [
                    drivers[slot].execute_stage_bw_and_step(action)
                    for slot, action in zip(active_indices, selected_actions)
                ]
            reward_parts_many = last_reward_parts_many(drivers, indices=active_indices)
            for local_slot, slot in enumerate(active_indices):
                step_result = step_results[local_slot]
                reward_parts = dict(reward_parts_many[local_slot] or {})
                acc = slot_acc[slot]
                acc["reward_sum"] += float(list(step_result.rewards.values())[0])
                acc["steps"] += 1.0
                acc["processed_ratio_sum"] += float(reward_parts.get("processed_ratio_eval", 0.0))
                acc["drop_ratio_sum"] += float(reward_parts.get("drop_ratio_eval", 0.0))
                acc["pre_backlog_sum"] += float(reward_parts.get("pre_backlog_steps_eval", 0.0))
                acc["d_sys_sum"] += float(reward_parts.get("D_sys_report", 0.0))
                acc["x_acc_sum"] += float(reward_parts.get("x_acc", 0.0))
                acc["x_rel_sum"] += float(reward_parts.get("x_rel", 0.0))
                acc["g_pre_sum"] += float(reward_parts.get("g_pre", 0.0))
                acc["d_pre_sum"] += float(reward_parts.get("d_pre", 0.0))
                acc["throughput_access_sum"] += float(reward_parts.get("throughput_access_norm", 0.0))
                acc["throughput_backhaul_sum"] += float(reward_parts.get("throughput_backhaul_norm", 0.0))
                acc["sat_processed_norm_sum"] += float(reward_parts.get("sat_processed_norm", 0.0))
                acc["sat_processed_incoming_ratio_sum"] += float(reward_parts.get("sat_processed_incoming_ratio_step", 0.0))
                acc["assoc_centroid_sum"] += float(reward_parts.get("assoc_centroid_dist_norm_mean", 0.0))
                acc["assoc_centroid_valid_frac_sum"] += float(reward_parts.get("assoc_centroid_valid_uav_frac", 0.0))
                acc["collision_any"] = max(acc["collision_any"], float(reward_parts.get("collision_event", 0.0)))
                done = bool(list(step_result.terminations.values())[0] or list(step_result.truncations.values())[0])
                if not done:
                    continue
                rows.append(_episode_row_from_accumulator(slot_episode[slot], acc))
                slot_acc[slot] = _new_episode_accumulator()
                if next_episode < int(episodes):
                    seed_next = None if episode_seed_base is None else int(episode_seed_base) + next_episode
                    reset_at(drivers, slot, seed_next)
                    slot_episode[slot] = int(next_episode)
                    next_episode += 1
                else:
                    slot_active[slot] = False
                    slot_episode[slot] = -1
        rows.sort(key=lambda row: int(row["episode"]))
        totals = _episode_metrics_template()
        for row in rows:
            for key in totals:
                totals[key] += float(row[key])
        scale = 1.0 / float(max(len(rows), 1))
        summary = {key: float(val * scale) for key, val in totals.items()}
        summary["selection_source_hist"] = dict(sorted(Counter(str(row["selected_name"]) for row in select_rows).items()))
        return summary, rows, select_rows
    finally:
        close_structured_env_group(env_group)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--select_bank", type=str, required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--score_temp", type=float, default=0.05)
    parser.add_argument("--hard_ce_coef", type=float, default=0.25)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--holdout_episodes", type=int, default=2)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=45678)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--live_episodes", type=int, default=0)
    parser.add_argument("--live_episode_seed_base", type=int, default=62000)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--vec_backend", choices=["sync", "subproc"], default="sync")
    parser.add_argument("--out_dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_all_seeds(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    bank = torch.load(str(Path(args.select_bank)), map_location="cpu", weights_only=False)
    entries = list(bank["entries"])
    field_shapes = _infer_local_state_field_shapes(entries)
    train_entries, holdout_entries = _choose_holdout_episodes(entries, int(args.holdout_episodes), int(args.seed))
    train_split = _prepare_split(train_entries, field_shapes)
    holdout_split = _prepare_split(holdout_entries, field_shapes)
    context_dim = int(train_split["contexts"].shape[1])
    candidate_dim = int(train_split["candidate_features"].shape[2])
    candidate_count = int(train_split["candidate_features"].shape[1])
    model = BwSelectScorer(context_dim=context_dim, candidate_dim=candidate_dim, hidden_dim=int(args.hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    train_tensors = _tensor_split(train_split, device)
    rng = np.random.default_rng(int(args.seed))

    history: list[dict[str, float]] = []
    batch_size = min(max(int(args.batch_size), 1), len(train_entries))
    for step_idx in range(max(int(args.steps), 0)):
        batch_indices = np.arange(len(train_entries)) if batch_size >= len(train_entries) else rng.choice(len(train_entries), size=batch_size, replace=False)
        batch_indices_t = torch.as_tensor(batch_indices, dtype=torch.long, device=device)
        contexts = train_tensors["contexts"].index_select(0, batch_indices_t)
        candidate_features = train_tensors["candidate_features"].index_select(0, batch_indices_t)
        candidate_scores = train_tensors["candidate_scores"].index_select(0, batch_indices_t)
        selected_idx = train_tensors["selected_idx"].index_select(0, batch_indices_t)
        logits = model(contexts, candidate_features)
        target_probs = _soft_target(candidate_scores, float(args.score_temp))
        log_probs = F.log_softmax(logits, dim=1)
        distill_kl = F.kl_div(log_probs, target_probs, reduction="batchmean")
        hard_ce = F.cross_entropy(logits, selected_idx)
        loss = distill_kl + float(args.hard_ce_coef) * hard_ce
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
        optimizer.step()
        if step_idx == 0 or (step_idx + 1) % max(int(args.log_interval), 1) == 0 or step_idx == int(args.steps) - 1:
            with torch.inference_mode():
                batch_top1 = float((torch.argmax(logits, dim=1) == selected_idx).float().mean().item())
            history.append(
                {
                    "step": float(step_idx + 1),
                    "loss_total": float(loss.detach().cpu().item()),
                    "loss_distill_kl": float(distill_kl.detach().cpu().item()),
                    "loss_hard_ce": float(hard_ce.detach().cpu().item()),
                    "batch_top1_acc": batch_top1,
                }
            )

    model.eval()
    train_summary, train_rows = _evaluate_offline(model=model, split_name="train", split=train_split, device=device)
    holdout_summary, holdout_rows = _evaluate_offline(model=model, split_name="holdout", split=holdout_split, device=device)

    live_summary = None
    live_rows: list[dict[str, Any]] = []
    live_select_rows: list[dict[str, Any]] = []
    meta = dict(bank.get("meta", {}))
    if int(args.live_episodes) > 0:
        cfg_path = str(meta.get("config"))
        run_dir = Path(str(meta.get("run_dir")))
        checkpoint = _resolve_checkpoint(run_dir, str(meta.get("checkpoint")), None)
        cfg = load_config(cfg_path)
        exec_accel_source = _normalize_exec_source(meta.get("exec_accel_source", getattr(cfg, "exec_accel_source", "policy")))
        exec_sat_source = _normalize_exec_source(meta.get("exec_sat_source", getattr(cfg, "exec_sat_source", "policy")))
        actor = _load_actor(cfg, checkpoint, hidden_dim=256, embed_dim=64, device=device)
        live_summary, live_rows, live_select_rows = _evaluate_live(
            cfg=cfg,
            actor=actor,
            model=model,
            device=device,
            episodes=int(args.live_episodes),
            episode_seed_base=int(args.live_episode_seed_base),
            num_envs=int(args.num_envs),
            vec_backend=str(args.vec_backend),
            sample_count=int(meta.get("sample_count", 0)),
            random_count=int(meta.get("random_count", 0)),
            include_heuristic=bool(meta.get("include_heuristic", False)),
            include_simplex=bool(meta.get("include_simplex", False)),
            heuristic_policy=str(meta.get("heuristic_policy", "cluster_center_queue_aware")),
            bw_deterministic_opt_steps=int(meta.get("bw_deterministic_opt_steps", 8)),
            bw_deterministic_step_size=float(meta.get("bw_deterministic_step_size", 0.5)),
            seed=int(args.seed) + 1000,
            field_shapes=field_shapes,
            exec_accel_source=str(exec_accel_source),
            exec_sat_source=str(exec_sat_source),
        )

    summary = {
        "select_bank": str(Path(args.select_bank).resolve()),
        "bank_entry_count": int(len(entries)),
        "train_state_count": int(len(train_entries)),
        "holdout_state_count": int(len(holdout_entries)),
        "candidate_count": int(candidate_count),
        "context_dim": int(context_dim),
        "candidate_dim": int(candidate_dim),
        "local_state_field_shapes": {name: list(shape) for name, shape in field_shapes.items()},
        "steps": int(args.steps),
        "batch_size": int(batch_size),
        "hidden_dim": int(args.hidden_dim),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "score_temp": float(args.score_temp),
        "hard_ce_coef": float(args.hard_ce_coef),
        "grad_clip": float(args.grad_clip),
        "train_split": train_summary,
        "holdout_split": holdout_summary,
        "train_history_tail": history[-10:],
        "live_eval": live_summary,
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "context_dim": context_dim,
            "candidate_dim": candidate_dim,
            "hidden_dim": int(args.hidden_dim),
            "field_shapes": field_shapes,
            "summary": summary,
        },
        out_dir / "selector_model.pt",
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(out_dir / "train_history.csv", history)
    _write_csv(out_dir / "offline_train_rows.csv", train_rows)
    _write_csv(out_dir / "offline_holdout_rows.csv", holdout_rows)
    if live_rows:
        _write_csv(out_dir / "live_eval_rows.csv", live_rows)
    if live_select_rows:
        _write_csv(out_dir / "live_select_rows.csv", live_select_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
