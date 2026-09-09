from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from scripts.diagnostics.audit.audit_bw_broad2local_offline import _rollout_panel_action_scores
from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import _collate_dataclass, _to_device_dataclass
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
    bundle.actor.to(device)
    return bundle.actor


def _expand_local_states(entries: list[dict[str, Any]], candidate_count: int) -> list[Any]:
    expanded: list[Any] = []
    for entry in entries:
        expanded.extend([entry["local_state"]] * int(candidate_count))
    return expanded


def _candidate_adv_weights(scores: np.ndarray, clip_value: float) -> np.ndarray:
    centered = scores.astype(np.float32, copy=False)
    centered = centered - float(np.mean(centered))
    denom = float(np.std(centered))
    if denom < 1.0e-6:
        return np.zeros_like(centered, dtype=np.float32)
    centered = centered / denom
    return np.clip(centered, -float(clip_value), float(clip_value)).astype(np.float32, copy=False)


def _candidate_target_probs(scores: np.ndarray, temperature: float) -> np.ndarray:
    adv = _candidate_adv_weights(scores, clip_value=10.0)
    logits = adv / max(float(temperature), 1.0e-6)
    logits = logits - float(np.max(logits))
    exp_logits = np.exp(logits).astype(np.float32, copy=False)
    probs = exp_logits / np.clip(float(np.sum(exp_logits)), 1.0e-8, None)
    return probs.astype(np.float32, copy=False)


def _choose_entries(
    entries: list[dict[str, Any]],
    *,
    max_states: int,
    seed: int,
) -> list[dict[str, Any]]:
    if max_states <= 0 or len(entries) <= int(max_states):
        return list(entries)
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(len(entries), size=int(max_states), replace=False).tolist()
    chosen.sort()
    return [entries[int(idx)] for idx in chosen]


def _split_by_holdout_episodes(
    entries: list[dict[str, Any]],
    *,
    holdout_episodes: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not entries:
        return [], []
    unique_episodes = sorted({int(entry["episode"]) for entry in entries})
    if len(unique_episodes) <= 1:
        split = max(1, len(entries) // 5)
        return list(entries[split:]), list(entries[:split])
    holdout_count = min(max(int(holdout_episodes), 1), len(unique_episodes) - 1)
    rng = np.random.default_rng(int(seed))
    holdout_episode_set = set(rng.choice(unique_episodes, size=holdout_count, replace=False).tolist())
    train_entries = [entry for entry in entries if int(entry["episode"]) not in holdout_episode_set]
    holdout_entries = [entry for entry in entries if int(entry["episode"]) in holdout_episode_set]
    if not train_entries or not holdout_entries:
        split = max(1, len(entries) // 5)
        return list(entries[split:]), list(entries[:split])
    return train_entries, holdout_entries


def _panel_entry_rows(
    entry: dict[str, Any],
    *,
    eval_drivers: Any,
    follow_actor: Any,
    device: torch.device,
    gamma: float,
    k_steps: int,
    bw_deterministic_opt_steps: int,
    bw_deterministic_step_size: float,
    score_temp: float,
    adv_clip: float,
    min_valid_count: int,
    min_panel_std: float,
) -> list[dict[str, Any]]:
    candidate_names = list(entry["candidate_names"])
    candidate_actions = [np.asarray(action, dtype=np.float32) for action in entry["candidate_actions"]]
    policy_det_idx = int(entry["policy_det_idx"])
    heuristic_idx = None if entry.get("heuristic_idx") is None else int(entry["heuristic_idx"])
    if policy_det_idx < 0 or policy_det_idx >= len(candidate_actions):
        raise IndexError("policy_det_idx is out of bounds")
    base_joint = np.asarray(candidate_actions[policy_det_idx], dtype=np.float32)
    if base_joint.ndim != 2:
        raise ValueError(f"candidate action must have shape [num_uav, num_user], got {base_joint.shape}")
    local_state = entry["local_state"]
    bw_valid_mask = np.asarray(local_state.bw_valid_mask, dtype=bool)
    panel_entries: list[dict[str, Any]] = []
    for uav_idx in range(base_joint.shape[0]):
        valid_count = int(np.asarray(bw_valid_mask[uav_idx], dtype=np.int64).sum())
        if valid_count < int(min_valid_count):
            continue
        panel_actions: list[np.ndarray] = []
        for cand_action in candidate_actions:
            replaced = np.asarray(base_joint, dtype=np.float32).copy()
            replaced[uav_idx] = np.asarray(cand_action, dtype=np.float32)[uav_idx]
            panel_actions.append(replaced)
        panel_scores = _rollout_panel_action_scores(
            eval_drivers,
            entry["snapshot_state"],
            panel_actions=panel_actions,
            follow_actor=follow_actor,
            follow_device=device,
            follow_deterministic=True,
            bw_deterministic_readout="latent_mean_pushforward",
            bw_deterministic_opt_steps=int(bw_deterministic_opt_steps),
            bw_deterministic_step_size=float(bw_deterministic_step_size),
            gamma=float(gamma),
            k_steps=int(k_steps),
        )
        if len(panel_scores) != len(panel_actions):
            raise RuntimeError("panel_scores and panel_actions length mismatch")
        score_arr = np.asarray(panel_scores, dtype=np.float32)
        score_std = float(np.std(score_arr))
        if score_std < float(min_panel_std):
            continue
        adv_weights = _candidate_adv_weights(score_arr, clip_value=float(adv_clip))
        target_probs = _candidate_target_probs(score_arr, temperature=float(score_temp))
        best_idx = int(np.argmax(score_arr))
        base_score = float(score_arr[policy_det_idx])
        heur_score = base_score if heuristic_idx is None else float(score_arr[int(heuristic_idx)])
        panel_entries.append(
            {
                "episode": int(entry["episode"]),
                "t": int(entry["t"]),
                "uav_idx": int(uav_idx),
                "local_state": local_state,
                "candidate_names": list(candidate_names),
                "candidate_actions": panel_actions,
                "candidate_scores": score_arr.astype(np.float32, copy=False),
                "adv_weights": adv_weights.astype(np.float32, copy=False),
                "target_probs": target_probs.astype(np.float32, copy=False),
                "policy_det_idx": int(policy_det_idx),
                "heuristic_idx": heuristic_idx,
                "selected_idx": int(best_idx),
                "selected_name": str(candidate_names[best_idx]),
                "selected_score": float(score_arr[best_idx]),
                "base_score": base_score,
                "heuristic_score": heur_score,
                "oracle_gain_vs_base": float(score_arr[best_idx] - base_score),
                "oracle_gain_vs_heuristic": float(score_arr[best_idx] - heur_score),
                "score_std": score_std,
                "valid_count": int(valid_count),
            }
        )
    return panel_entries


def _prepare_panel_bank(
    *,
    select_bank_entries: list[dict[str, Any]],
    eval_drivers: Any,
    follow_actor: Any,
    device: torch.device,
    gamma: float,
    k_steps: int,
    bw_deterministic_opt_steps: int,
    bw_deterministic_step_size: float,
    score_temp: float,
    adv_clip: float,
    min_valid_count: int,
    min_panel_std: float,
) -> list[dict[str, Any]]:
    panel_entries: list[dict[str, Any]] = []
    for entry in select_bank_entries:
        panel_entries.extend(
            _panel_entry_rows(
                entry,
                eval_drivers=eval_drivers,
                follow_actor=follow_actor,
                device=device,
                gamma=float(gamma),
                k_steps=int(k_steps),
                bw_deterministic_opt_steps=int(bw_deterministic_opt_steps),
                bw_deterministic_step_size=float(bw_deterministic_step_size),
                score_temp=float(score_temp),
                adv_clip=float(adv_clip),
                min_valid_count=int(min_valid_count),
                min_panel_std=float(min_panel_std),
            )
        )
    return panel_entries


def _evaluate_panel_split(
    actor,
    panel_entries: list[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    if not panel_entries:
        return {
            "panel_count": 0,
            "pred_score": _summarize([]),
            "winner_score": _summarize([]),
            "base_score": _summarize([]),
            "heuristic_score": _summarize([]),
            "pred_gain_vs_base": _summarize([]),
            "pred_gain_vs_heuristic": _summarize([]),
            "top1_acc": 0.0,
            "pred_beats_base_frac": 0.0,
            "pred_beats_heuristic_frac": 0.0,
            "capture_ratio": 0.0,
        }, rows

    actor.eval()
    pred_scores: list[float] = []
    winner_scores: list[float] = []
    base_scores: list[float] = []
    heur_scores: list[float] = []
    pred_gain_vs_base: list[float] = []
    pred_gain_vs_heuristic: list[float] = []
    top1_hits: list[float] = []
    beats_base: list[float] = []
    beats_heuristic: list[float] = []
    capture_terms: list[float] = []

    with torch.inference_mode():
        for start in range(0, len(panel_entries), int(batch_size)):
            batch_entries = panel_entries[start : start + int(batch_size)]
            candidate_count = len(batch_entries[0]["candidate_actions"])
            local_states = _expand_local_states(batch_entries, candidate_count)
            state_batch_cpu = _collate_dataclass(local_states, torch.device("cpu"))
            state_batch = _to_device_dataclass(state_batch_cpu, device)
            candidate_actions = np.stack(
                [np.stack(entry["candidate_actions"], axis=0) for entry in batch_entries],
                axis=0,
            ).astype(np.float32, copy=False)
            batch_size_local, candidate_count, num_uav, user_count = candidate_actions.shape
            action_tensor = torch.as_tensor(candidate_actions, dtype=torch.float32, device=device).reshape(
                batch_size_local * candidate_count * num_uav,
                user_count,
            )
            actor_out = actor.evaluate_bw(state_batch, action_tensor)
            per_agent_logprob = actor_out.logprob.reshape(batch_size_local, candidate_count, num_uav)
            uav_indices = torch.as_tensor(
                [int(entry["uav_idx"]) for entry in batch_entries],
                dtype=torch.long,
                device=device,
            ).view(batch_size_local, 1, 1)
            panel_logprob = per_agent_logprob.gather(
                dim=2,
                index=uav_indices.expand(-1, candidate_count, 1),
            ).squeeze(-1)
            chosen_idx = torch.argmax(panel_logprob, dim=1).detach().cpu().numpy().astype(np.int64)

            for local_idx, entry in enumerate(batch_entries):
                candidate_scores = np.asarray(entry["candidate_scores"], dtype=np.float32)
                pred_idx = int(chosen_idx[local_idx])
                winner_idx = int(entry["selected_idx"])
                base_idx = int(entry["policy_det_idx"])
                heuristic_idx = entry["heuristic_idx"]
                pred_score = float(candidate_scores[pred_idx])
                winner_score = float(candidate_scores[winner_idx])
                base_score = float(candidate_scores[base_idx])
                heuristic_score = base_score if heuristic_idx is None else float(candidate_scores[int(heuristic_idx)])
                pred_scores.append(pred_score)
                winner_scores.append(winner_score)
                base_scores.append(base_score)
                heur_scores.append(heuristic_score)
                pred_gain_vs_base.append(pred_score - base_score)
                pred_gain_vs_heuristic.append(pred_score - heuristic_score)
                top1_hits.append(float(pred_idx == winner_idx))
                beats_base.append(float(pred_score > base_score + 1.0e-9))
                beats_heuristic.append(float(pred_score > heuristic_score + 1.0e-9))
                denom = winner_score - base_score
                if denom > 1.0e-8:
                    capture_terms.append((pred_score - base_score) / denom)
                rows.append(
                    {
                        "episode": int(entry["episode"]),
                        "t": int(entry["t"]),
                        "uav_idx": int(entry["uav_idx"]),
                        "pred_idx": int(pred_idx),
                        "winner_idx": int(winner_idx),
                        "base_idx": int(base_idx),
                        "heuristic_idx": -1 if heuristic_idx is None else int(heuristic_idx),
                        "pred_score": pred_score,
                        "winner_score": winner_score,
                        "base_score": base_score,
                        "heuristic_score": heuristic_score,
                        "pred_gain_vs_base": pred_score - base_score,
                        "pred_gain_vs_heuristic": pred_score - heuristic_score,
                        "top1_hit": float(pred_idx == winner_idx),
                    }
                )

    summary = {
        "panel_count": int(len(panel_entries)),
        "pred_score": _summarize(pred_scores),
        "winner_score": _summarize(winner_scores),
        "base_score": _summarize(base_scores),
        "heuristic_score": _summarize(heur_scores),
        "pred_gain_vs_base": _summarize(pred_gain_vs_base),
        "pred_gain_vs_heuristic": _summarize(pred_gain_vs_heuristic),
        "top1_acc": _mean(top1_hits),
        "pred_beats_base_frac": _mean(beats_base),
        "pred_beats_heuristic_frac": _mean(beats_heuristic),
        "capture_ratio": _mean(capture_terms),
    }
    return summary, rows


def _train_epoch_step(
    actor,
    optimizer,
    batch_entries: list[dict[str, Any]],
    *,
    device: torch.device,
) -> dict[str, float]:
    actor.train()
    candidate_count = len(batch_entries[0]["candidate_actions"])
    local_states = _expand_local_states(batch_entries, candidate_count)
    state_batch_cpu = _collate_dataclass(local_states, torch.device("cpu"))
    state_batch = _to_device_dataclass(state_batch_cpu, device)
    candidate_actions = np.stack(
        [np.stack(entry["candidate_actions"], axis=0) for entry in batch_entries],
        axis=0,
    ).astype(np.float32, copy=False)
    adv_weights = np.stack(
        [np.asarray(entry["adv_weights"], dtype=np.float32) for entry in batch_entries],
        axis=0,
    ).astype(np.float32, copy=False)
    batch_size_local, candidate_count, num_uav, user_count = candidate_actions.shape
    action_tensor = torch.as_tensor(candidate_actions, dtype=torch.float32, device=device).reshape(
        batch_size_local * candidate_count * num_uav,
        user_count,
    )
    adv_tensor = torch.as_tensor(adv_weights, dtype=torch.float32, device=device)
    actor_out = actor.evaluate_bw(state_batch, action_tensor)
    per_agent_logprob = actor_out.logprob.reshape(batch_size_local, candidate_count, num_uav)
    uav_indices = torch.as_tensor(
        [int(entry["uav_idx"]) for entry in batch_entries],
        dtype=torch.long,
        device=device,
    ).view(batch_size_local, 1, 1)
    panel_logprob = per_agent_logprob.gather(
        dim=2,
        index=uav_indices.expand(-1, candidate_count, 1),
    ).squeeze(-1)
    loss_ce = -(adv_tensor * panel_logprob).sum(dim=1).mean()
    batch_target_top1 = torch.argmax(adv_tensor, dim=1)
    batch_pred_top1 = torch.argmax(panel_logprob, dim=1)
    top1_acc = float((batch_target_top1 == batch_pred_top1).to(dtype=torch.float32).mean().item())
    optimizer.zero_grad(set_to_none=True)
    loss_ce.backward()
    grad_sq = 0.0
    for param in actor.parameters():
        if not param.requires_grad or param.grad is None:
            continue
        grad_sq += float(param.grad.detach().pow(2).sum().item())
    grad_norm = float(grad_sq ** 0.5)
    torch.nn.utils.clip_grad_norm_(
        [param for param in actor.parameters() if param.requires_grad],
        max_norm=1.0,
    )
    optimizer.step()
    return {
        "loss_total": float(loss_ce.item()),
        "batch_top1_acc": top1_acc,
        "grad_norm": grad_norm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BW panel-advantage estimator v0 from a select bank.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--update", type=int, default=None)
    parser.add_argument("--select_bank", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_envs", type=int, default=12)
    parser.add_argument("--vec_backend", choices=["sync", "subproc"], default="sync")
    parser.add_argument("--max_states", type=int, default=960)
    parser.add_argument("--holdout_episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260408)
    parser.add_argument("--k_steps", type=int, default=2)
    parser.add_argument("--score_temp", type=float, default=0.5)
    parser.add_argument("--adv_clip", type=float, default=2.5)
    parser.add_argument("--min_valid_count", type=int, default=2)
    parser.add_argument("--min_panel_std", type=float, default=1.0e-4)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--bw_deterministic_opt_steps", type=int, default=8)
    parser.add_argument("--bw_deterministic_step_size", type=float, default=0.5)
    args = parser.parse_args()

    _set_all_seeds(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir)
    config_path = Path(args.config) if args.config else (run_dir / "config_source.yaml")
    cfg = load_config(str(config_path))
    checkpoint = _resolve_checkpoint(run_dir, args.checkpoint, args.update)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    actor = _load_actor(cfg, checkpoint, hidden_dim=int(args.hidden_dim), embed_dim=int(args.embed_dim), device=device)
    actor.eval()
    for name, param in actor.named_parameters():
        param.requires_grad = name.startswith("bw_policy.")

    bank = torch.load(str(Path(args.select_bank)), map_location="cpu", weights_only=False)
    raw_entries = list(bank["entries"] if isinstance(bank, dict) and "entries" in bank else bank)
    selected_entries = _choose_entries(raw_entries, max_states=int(args.max_states), seed=int(args.seed))

    eval_slots = max(1, min(int(args.num_envs), 16))
    eval_drivers = make_structured_env_group(cfg, num_envs=eval_slots, backend=args.vec_backend)
    try:
        panel_entries = _prepare_panel_bank(
            select_bank_entries=selected_entries,
            eval_drivers=eval_drivers,
            follow_actor=actor,
            device=device,
            gamma=float(cfg.gamma),
            k_steps=int(args.k_steps),
            bw_deterministic_opt_steps=int(args.bw_deterministic_opt_steps),
            bw_deterministic_step_size=float(args.bw_deterministic_step_size),
            score_temp=float(args.score_temp),
            adv_clip=float(args.adv_clip),
            min_valid_count=int(args.min_valid_count),
            min_panel_std=float(args.min_panel_std),
        )
    finally:
        close_structured_env_group(eval_drivers)

    train_entries, holdout_entries = _split_by_holdout_episodes(
        panel_entries,
        holdout_episodes=int(args.holdout_episodes),
        seed=int(args.seed),
    )
    if not train_entries:
        raise RuntimeError("Train split is empty.")
    if not holdout_entries:
        raise RuntimeError("Holdout split is empty.")

    panel_bank_path = out_dir / "panel_bank.pt"
    torch.save(
        {
            "meta": {
                "config": str(config_path.resolve()),
                "checkpoint": str(checkpoint.resolve()),
                "seed": int(args.seed),
                "k_steps": int(args.k_steps),
                "score_temp": float(args.score_temp),
                "adv_clip": float(args.adv_clip),
                "max_states": int(args.max_states),
                "min_valid_count": int(args.min_valid_count),
                "min_panel_std": float(args.min_panel_std),
            },
            "entries": panel_entries,
        },
        panel_bank_path,
    )

    optimizer = torch.optim.AdamW(
        [param for param in actor.parameters() if param.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    initial_holdout_summary, initial_holdout_rows = _evaluate_panel_split(
        actor,
        holdout_entries,
        device=device,
        batch_size=int(args.batch_size),
    )
    best_snapshot = {
        "step": 0,
        "state_dict": {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()},
        "holdout_capture": float(initial_holdout_summary["capture_ratio"]),
        "holdout_top1": float(initial_holdout_summary["top1_acc"]),
    }

    history: list[dict[str, float]] = []
    rng = np.random.default_rng(int(args.seed))
    for step in range(1, int(args.steps) + 1):
        batch_indices = rng.choice(len(train_entries), size=min(int(args.batch_size), len(train_entries)), replace=False)
        batch_entries = [train_entries[int(idx)] for idx in batch_indices.tolist()]
        train_metrics = _train_epoch_step(actor, optimizer, batch_entries, device=device)
        row = {"step": float(step), **train_metrics}
        if step == 1 or step % 50 == 0 or step == int(args.steps):
            holdout_summary_step, _ = _evaluate_panel_split(
                actor,
                holdout_entries,
                device=device,
                batch_size=int(args.batch_size),
            )
            row["holdout_capture_ratio"] = float(holdout_summary_step["capture_ratio"])
            row["holdout_top1_acc"] = float(holdout_summary_step["top1_acc"])
            row["holdout_pred_beats_base_frac"] = float(holdout_summary_step["pred_beats_base_frac"])
            if (
                float(holdout_summary_step["capture_ratio"]) > float(best_snapshot["holdout_capture"])
                or (
                    abs(float(holdout_summary_step["capture_ratio"]) - float(best_snapshot["holdout_capture"])) < 1.0e-9
                    and float(holdout_summary_step["top1_acc"]) > float(best_snapshot["holdout_top1"])
                )
            ):
                best_snapshot = {
                    "step": int(step),
                    "state_dict": {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()},
                    "holdout_capture": float(holdout_summary_step["capture_ratio"]),
                    "holdout_top1": float(holdout_summary_step["top1_acc"]),
                }
        history.append(row)

    actor.load_state_dict(best_snapshot["state_dict"], strict=True)
    actor.eval()

    train_summary, train_rows = _evaluate_panel_split(
        actor,
        train_entries,
        device=device,
        batch_size=int(args.batch_size),
    )
    holdout_summary, holdout_rows = _evaluate_panel_split(
        actor,
        holdout_entries,
        device=device,
        batch_size=int(args.batch_size),
    )

    torch.save(actor.state_dict(), out_dir / "actor_bw_panel_adv_v0.pt")
    _write_csv(out_dir / "train_history.csv", history)
    _write_csv(out_dir / "train_panel_rows.csv", train_rows)
    _write_csv(out_dir / "holdout_panel_rows.csv", holdout_rows)

    signal_oracle_gain = [float(entry["oracle_gain_vs_base"]) for entry in panel_entries]
    signal_oracle_gain_heur = [float(entry["oracle_gain_vs_heuristic"]) for entry in panel_entries]
    signal_score_std = [float(entry["score_std"]) for entry in panel_entries]
    signal_valid_count = [float(entry["valid_count"]) for entry in panel_entries]
    summary = {
        "run_dir": str(run_dir.resolve()),
        "config": str(config_path.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "select_bank": str(Path(args.select_bank).resolve()),
        "panel_bank": str(panel_bank_path.resolve()),
        "device": str(device),
        "vec_backend": str(args.vec_backend),
        "num_envs": int(args.num_envs),
        "selected_state_count": int(len(selected_entries)),
        "panel_count": int(len(panel_entries)),
        "train_panel_count": int(len(train_entries)),
        "holdout_panel_count": int(len(holdout_entries)),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "k_steps": int(args.k_steps),
        "score_temp": float(args.score_temp),
        "adv_clip": float(args.adv_clip),
        "best_step": int(best_snapshot["step"]),
        "signal": {
            "oracle_gain_vs_base": _summarize(signal_oracle_gain),
            "oracle_gain_vs_heuristic": _summarize(signal_oracle_gain_heur),
            "panel_score_std": _summarize(signal_score_std),
            "valid_count": _summarize(signal_valid_count),
        },
        "train_split": train_summary,
        "holdout_split": holdout_summary,
        "gate_reference": {
            "continue_requires": {
                "holdout_capture_ratio_min": 0.45,
                "holdout_pred_beats_base_frac_min": 0.60,
                "holdout_pred_gain_vs_base_mean_min": 0.0,
            },
            "abandon_if": {
                "holdout_capture_ratio_below": 0.35,
                "holdout_pred_beats_base_frac_below": 0.55,
            },
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
