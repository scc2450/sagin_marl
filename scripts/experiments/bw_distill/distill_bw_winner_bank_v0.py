from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.diagnostics.audit.audit_bw_broad2local_offline import (
    _as_driver_list,
    _clone_bw_only_actor,
    _driver_capacity,
    _load_actor,
    _masked_simplex_kl,
    _rollout_k_many_with_fixed_follow,
    _sum_by_counts,
    _valid_bw_mask,
)
from sagin_marl.rl.structured_mappo import _collate_dataclass, _split_tensor_by_counts, _to_device_dataclass


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_winner_bank(path: Path) -> dict[str, Any]:
    return torch.load(str(path), map_location="cpu", weights_only=False)


def _entries_by_indices(entries: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
    index_set = set(int(i) for i in indices)
    return [entry for entry in entries if int(entry["index"]) in index_set]


def _make_supervised_batch(entries: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    local_states_cpu = [entry["local_state"] for entry in entries]
    local_state = _to_device_dataclass(_collate_dataclass(local_states_cpu, torch.device("cpu")), device)
    counts = [int(entry["local_state"].user_mask.shape[0]) for entry in entries]
    winner_action = torch.cat(
        [torch.as_tensor(entry["winner_action"], dtype=torch.float32) for entry in entries],
        dim=0,
    ).to(device)
    base_action = torch.cat(
        [torch.as_tensor(entry["policy_det_action"], dtype=torch.float32) for entry in entries],
        dim=0,
    ).to(device)
    winner_gain = torch.as_tensor(
        [float(entry["winner_gain_vs_policy_det"]) for entry in entries],
        dtype=torch.float32,
        device=device,
    )
    return {
        "entries": entries,
        "local_state": local_state,
        "counts": counts,
        "winner_action": winner_action,
        "base_action": base_action,
        "winner_gain": winner_gain,
    }


def _compute_distill_loss(
    actor,
    batch: dict[str, Any],
    *,
    anchor_coef: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    local_state = batch["local_state"]
    counts = batch["counts"]
    winner_action = batch["winner_action"]
    base_action = batch["base_action"]
    valid_mask = _valid_bw_mask(local_state)
    student_action = actor.bw_policy.deterministic_action(local_state, readout="latent_mean_pushforward")

    winner_kl_per_agent = _masked_simplex_kl(winner_action, student_action, valid_mask)
    winner_kl_per_state = _sum_by_counts(winner_kl_per_agent, counts)
    winner_kl_loss = winner_kl_per_state.mean()

    total_loss = winner_kl_loss
    base_anchor_loss = torch.zeros((), dtype=torch.float32, device=winner_kl_loss.device)
    if float(anchor_coef) > 0.0:
        base_kl_per_agent = _masked_simplex_kl(base_action, student_action, valid_mask)
        base_kl_per_state = _sum_by_counts(base_kl_per_agent, counts)
        base_anchor_loss = base_kl_per_state.mean()
        total_loss = total_loss + float(anchor_coef) * base_anchor_loss

    with torch.no_grad():
        l1_per_agent = (
            (student_action - winner_action).abs() * valid_mask.to(dtype=student_action.dtype)
        ).sum(dim=-1)
        l1_per_state = _sum_by_counts(l1_per_agent, counts)
        target_mass = winner_action.sum(dim=-1).clamp_min(1.0e-8)
        student_mass = student_action.sum(dim=-1)
    return total_loss, {
        "loss_total": float(total_loss.detach().cpu().item()),
        "loss_winner_kl": float(winner_kl_loss.detach().cpu().item()),
        "loss_base_anchor": float(base_anchor_loss.detach().cpu().item()),
        "winner_l1_state_mean": float(l1_per_state.mean().detach().cpu().item()),
        "target_mass_mean": float(target_mass.mean().detach().cpu().item()),
        "student_mass_mean": float(student_mass.mean().detach().cpu().item()),
    }


def _evaluate_split(
    *,
    entries: list[dict[str, Any]],
    student_actor,
    follow_actor,
    cfg,
    device: torch.device,
    num_envs: int,
    vec_backend: str,
    k_steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not entries:
        return {
            "state_count": 0,
            "base_det_score": _summarize([]),
            "heuristic_score": _summarize([]),
            "winner_upper_bound_score": _summarize([]),
            "student_det_score": _summarize([]),
            "student_gain_vs_base": _summarize([]),
            "student_gain_vs_heuristic": _summarize([]),
            "upper_bound_gain_vs_base": _summarize([]),
            "capture_ratio": 0.0,
            "student_beats_base_frac": 0.0,
            "student_beats_heuristic_frac": 0.0,
            "student_hits_or_exceeds_winner_frac": 0.0,
            "winner_kl_state": _summarize([]),
        }, []

    from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_env_group

    gamma = float(cfg.gamma)
    eval_slots = max(1, min(int(num_envs), max(len(entries), 1)))
    eval_group = make_structured_env_group(cfg, num_envs=eval_slots, backend=str(vec_backend))
    eval_drivers = eval_group if not isinstance(eval_group, list) else _as_driver_list(eval_group)
    if isinstance(eval_group, list):
        eval_drivers = _as_driver_list(eval_group)
    rows: list[dict[str, Any]] = []
    base_scores: list[float] = []
    heuristic_scores: list[float] = []
    winner_scores: list[float] = []
    student_scores: list[float] = []
    student_gain_vs_base: list[float] = []
    student_gain_vs_heuristic: list[float] = []
    upper_bound_gain_vs_base: list[float] = []
    winner_kl_state_values: list[float] = []
    try:
        chunk_size = max(1, _driver_capacity(eval_drivers))
        for start in range(0, len(entries), chunk_size):
            chunk_entries = entries[start : start + chunk_size]
            local_states_cpu = [entry["local_state"] for entry in chunk_entries]
            local_state = _to_device_dataclass(_collate_dataclass(local_states_cpu, torch.device("cpu")), device)
            counts = [int(entry["local_state"].user_mask.shape[0]) for entry in chunk_entries]
            with torch.inference_mode():
                student_action_batch = student_actor.bw_policy.deterministic_action(
                    local_state,
                    readout="latent_mean_pushforward",
                )
                valid_mask = _valid_bw_mask(local_state)
                winner_action_batch = torch.cat(
                    [torch.as_tensor(entry["winner_action"], dtype=torch.float32) for entry in chunk_entries],
                    dim=0,
                ).to(device)
                winner_kl_per_agent = _masked_simplex_kl(winner_action_batch, student_action_batch, valid_mask)
                winner_kl_per_state = _sum_by_counts(winner_kl_per_agent, counts)
            student_actions = [
                piece.numpy().astype(np.float32, copy=False)
                for piece in _split_tensor_by_counts(student_action_batch.detach().cpu(), counts)
            ]
            totals_many = _rollout_k_many_with_fixed_follow(
                eval_drivers,
                [entry["snapshot_state"] for entry in chunk_entries],
                first_bw_actions=student_actions,
                follow_actor=follow_actor,
                follow_device=device,
                follow_deterministic=True,
                bw_deterministic_readout="latent_mean_pushforward",
                bw_deterministic_opt_steps=8,
                bw_deterministic_step_size=0.5,
                gamma=gamma,
                k_steps=int(k_steps),
            )
            for entry, totals, winner_kl_state in zip(chunk_entries, totals_many, winner_kl_per_state.detach().cpu().tolist()):
                base_det_score = float(entry["policy_det_score"])
                heuristic_score = float(entry["heuristic_score"])
                winner_score = float(entry["winner_score"])
                student_score = float(totals["reward"])
                row = {
                    "index": int(entry["index"]),
                    "split": str(entry["split"]),
                    "winner_source": str(entry["winner_source"]),
                    "base_det_score": base_det_score,
                    "heuristic_score": heuristic_score,
                    "winner_upper_bound_score": winner_score,
                    "student_det_score": student_score,
                    "student_gain_vs_base": float(student_score - base_det_score),
                    "student_gain_vs_heuristic": float(student_score - heuristic_score),
                    "upper_bound_gain_vs_base": float(winner_score - base_det_score),
                    "winner_kl_state": float(winner_kl_state),
                }
                rows.append(row)
                base_scores.append(base_det_score)
                heuristic_scores.append(heuristic_score)
                winner_scores.append(winner_score)
                student_scores.append(student_score)
                student_gain_vs_base.append(float(row["student_gain_vs_base"]))
                student_gain_vs_heuristic.append(float(row["student_gain_vs_heuristic"]))
                upper_bound_gain_vs_base.append(float(row["upper_bound_gain_vs_base"]))
                winner_kl_state_values.append(float(row["winner_kl_state"]))
    finally:
        close_structured_env_group(eval_group)

    base_mean = float(np.mean(np.asarray(base_scores, dtype=np.float64)))
    student_mean = float(np.mean(np.asarray(student_scores, dtype=np.float64)))
    winner_mean = float(np.mean(np.asarray(winner_scores, dtype=np.float64)))
    upper_bound_gain = winner_mean - base_mean
    student_gain = student_mean - base_mean
    capture_ratio = student_gain / upper_bound_gain if abs(upper_bound_gain) > 1.0e-8 else 0.0
    return {
        "state_count": int(len(entries)),
        "base_det_score": _summarize(base_scores),
        "heuristic_score": _summarize(heuristic_scores),
        "winner_upper_bound_score": _summarize(winner_scores),
        "student_det_score": _summarize(student_scores),
        "student_gain_vs_base": _summarize(student_gain_vs_base),
        "student_gain_vs_heuristic": _summarize(student_gain_vs_heuristic),
        "upper_bound_gain_vs_base": _summarize(upper_bound_gain_vs_base),
        "capture_ratio": float(capture_ratio),
        "student_beats_base_frac": float(np.mean(np.asarray([v > 1.0e-6 for v in student_gain_vs_base], dtype=np.float64))),
        "student_beats_heuristic_frac": float(
            np.mean(np.asarray([v > 1.0e-6 for v in student_gain_vs_heuristic], dtype=np.float64))
        ),
        "student_hits_or_exceeds_winner_frac": float(
            np.mean(np.asarray([s >= w - 1.0e-6 for s, w in zip(student_scores, winner_scores)], dtype=np.float64))
        ),
        "winner_kl_state": _summarize(winner_kl_state_values),
    }, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--winner_bank", type=str, required=True)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--update", type=int, default=None)
    parser.add_argument("--actor_checkpoint", type=str, default=None)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--anchor_coef", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--k_steps", type=int, default=2)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--vec_backend", choices=["sync", "subproc"], default="sync")
    parser.add_argument("--seed", type=int, default=45678)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_all_seeds(int(args.seed))
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = _load_winner_bank(Path(args.winner_bank))
    entries = list(bank["entries"])
    train_entries = _entries_by_indices(entries, list(bank["train_indices"]))
    holdout_entries = _entries_by_indices(entries, list(bank["holdout_indices"]))
    if not train_entries:
        raise RuntimeError("Winner bank train split is empty.")
    run_dir = Path(args.run_dir) if args.run_dir is not None else Path(str(bank["meta"]["run_dir"]))
    update = int(args.update) if args.update is not None else int(bank["meta"]["update"])
    actor_checkpoint = (
        str(args.actor_checkpoint)
        if args.actor_checkpoint is not None
        else (
            None
            if bank["meta"].get("actor_checkpoint") is None
            else str(bank["meta"]["actor_checkpoint"])
        )
    )

    cfg, base_actor = _load_actor(
        run_dir,
        update,
        device,
        actor_checkpoint=actor_checkpoint,
    )
    student_actor = _clone_bw_only_actor(base_actor, device)
    optimizer = torch.optim.Adam([p for p in student_actor.parameters() if p.requires_grad], lr=float(args.lr))
    rng = np.random.default_rng(int(args.seed))

    history: list[dict[str, float]] = []
    batch_size = min(max(int(args.batch_size), 1), len(train_entries))
    for step_idx in range(max(int(args.steps), 0)):
        batch_indices = (
            list(range(len(train_entries)))
            if batch_size >= len(train_entries)
            else rng.choice(len(train_entries), size=batch_size, replace=False).tolist()
        )
        batch = _make_supervised_batch([train_entries[int(i)] for i in batch_indices], device)
        optimizer.zero_grad(set_to_none=True)
        loss, stats = _compute_distill_loss(student_actor, batch, anchor_coef=float(args.anchor_coef))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_actor.bw_policy.parameters(), max_norm=float(args.grad_clip))
        optimizer.step()
        if step_idx == 0 or (step_idx + 1) % max(int(args.log_interval), 1) == 0 or step_idx == int(args.steps) - 1:
            row = {"step": float(step_idx + 1)}
            row.update(stats)
            history.append(row)

    student_actor.eval()
    base_actor.eval()
    train_summary, train_rows = _evaluate_split(
        entries=train_entries,
        student_actor=student_actor,
        follow_actor=base_actor,
        cfg=cfg,
        device=device,
        num_envs=int(args.num_envs),
        vec_backend=str(args.vec_backend),
        k_steps=int(args.k_steps),
    )
    holdout_summary, holdout_rows = _evaluate_split(
        entries=holdout_entries,
        student_actor=student_actor,
        follow_actor=base_actor,
        cfg=cfg,
        device=device,
        num_envs=int(args.num_envs),
        vec_backend=str(args.vec_backend),
        k_steps=int(args.k_steps),
    )

    summary = {
        "winner_bank": str(Path(args.winner_bank).resolve()),
        "run_dir": str(run_dir),
        "update": int(update),
        "actor_checkpoint": None if actor_checkpoint is None else str(Path(actor_checkpoint).resolve()),
        "steps": int(args.steps),
        "batch_size": int(batch_size),
        "lr": float(args.lr),
        "anchor_coef": float(args.anchor_coef),
        "grad_clip": float(args.grad_clip),
        "k_steps": int(args.k_steps),
        "num_envs": int(args.num_envs),
        "vec_backend": str(args.vec_backend),
        "device": str(device),
        "train_split": train_summary,
        "holdout_split": holdout_summary,
        "train_history_tail": history[-10:],
    }

    torch.save(student_actor.state_dict(), out_dir / "student_actor.pt")
    torch.save(
        {
            "actor_state_dict": student_actor.state_dict(),
            "summary": summary,
        },
        out_dir / "student_bundle.pt",
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(out_dir / "train_history.csv", history)
    _write_csv(out_dir / "train_eval_rows.csv", train_rows)
    _write_csv(out_dir / "holdout_eval_rows.csv", holdout_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
