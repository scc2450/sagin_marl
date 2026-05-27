from __future__ import annotations

import argparse
import copy
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
    _batched_bw_actions_from_snapshots,
    _collect_panel_bank,
    _rollout_panel_action_scores,
)
from sagin_marl.rl.structured_mappo import _to_device_dataclass
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_env_group


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


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _split_winner_entries(
    entries: list[dict[str, Any]],
    *,
    holdout_size: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not entries or int(holdout_size) <= 0:
        return [int(entry["index"]) for entry in entries], []
    informative = sorted(entries, key=lambda row: float(row["winner_gain_vs_policy_det"]), reverse=True)
    rng = np.random.default_rng(int(seed))
    candidate_count = max(int(holdout_size) * 3, int(holdout_size))
    candidate_count = min(candidate_count, len(informative))
    candidate_pool = informative[:candidate_count]
    holdout_positions = set(
        rng.choice(
            len(candidate_pool),
            size=min(int(holdout_size), len(candidate_pool)),
            replace=False,
        ).tolist()
    )
    holdout_indices: list[int] = []
    train_indices: list[int] = []
    for pos, entry in enumerate(candidate_pool):
        (holdout_indices if pos in holdout_positions else train_indices).append(int(entry["index"]))
    train_indices.extend(int(entry["index"]) for entry in informative[candidate_count:])
    return train_indices, holdout_indices


def _winner_bank_manifest_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        rows.append(
            {
                "index": int(entry["index"]),
                "split": str(entry["split"]),
                "winner_source": str(entry["winner_source"]),
                "winner_score": float(entry["winner_score"]),
                "policy_det_score": float(entry["policy_det_score"]),
                "heuristic_score": float(entry["heuristic_score"]),
                "simplex_det_score": float(entry["simplex_det_score"]),
                "winner_gain_vs_policy_det": float(entry["winner_gain_vs_policy_det"]),
                "winner_gain_vs_heuristic": float(entry["winner_gain_vs_heuristic"]),
                "winner_gain_vs_simplex_det": float(entry["winner_gain_vs_simplex_det"]),
                "panel_best_score": float(entry["panel_best_score"]),
                "sample_best_of_1": float(
                    entry["sample_scores"][0] if entry["sample_scores"] else 0.0
                ),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--actor_checkpoint", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--panel_states", type=int, default=32)
    parser.add_argument("--sample_count", type=int, default=32)
    parser.add_argument("--best_of", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--policy_mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--vec_backend", choices=["sync", "subproc"], default="sync")
    parser.add_argument("--heuristic_bw_source", type=str, default="queue_aware")
    parser.add_argument("--k_steps", type=int, default=2)
    parser.add_argument("--panel_random_count", type=int, default=2)
    parser.add_argument("--prefer_policy_gap_min", type=float, default=1.0e-3)
    parser.add_argument("--min_best_gap", type=float, default=0.02)
    parser.add_argument("--bw_deterministic_opt_steps", type=int, default=8)
    parser.add_argument("--bw_deterministic_step_size", type=float, default=0.5)
    parser.add_argument("--holdout_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_all_seeds(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir)
    device = torch.device(args.device)

    bank = _collect_panel_bank(
        run_dir=run_dir,
        update=int(args.update),
        device=device,
        actor_checkpoint=None if args.actor_checkpoint is None else str(args.actor_checkpoint),
        output_dir=out_dir,
        episodes=int(args.episodes),
        panel_states=int(args.panel_states),
        policy_mode=str(args.policy_mode),
        panel_random_count=int(args.panel_random_count),
        heuristic_bw_source=str(args.heuristic_bw_source),
        k_steps=int(args.k_steps),
        seed=int(args.seed),
        prefer_policy_gap_min=float(args.prefer_policy_gap_min),
        min_best_gap=float(args.min_best_gap),
        num_envs=int(args.num_envs),
        bw_deterministic_readout="latent_mean_pushforward",
        bw_deterministic_opt_steps=int(args.bw_deterministic_opt_steps),
        bw_deterministic_step_size=float(args.bw_deterministic_step_size),
        bw_parameterization=None,
        bw_alpha_init_bias=None,
        bw_alpha_max=None,
        bw_tau_enabled=None,
        bw_tau_init_bias=None,
        bw_tau_min=None,
        bw_tau_max=None,
        vec_backend=str(args.vec_backend),
    )
    cfg = bank["cfg"]
    actor = bank["base_actor"]
    entries = list(bank["entries"])
    eval_group = make_structured_env_group(cfg, num_envs=max(1, int(args.num_envs)), backend=str(args.vec_backend))

    best_of_values = sorted(set(max(1, int(v)) for v in args.best_of if int(v) > 0))
    sample_count = max(int(args.sample_count), max(best_of_values, default=1))

    rows: list[dict[str, Any]] = []
    latent_scores: list[float] = []
    simplex_scores: list[float] = []
    heuristic_scores: list[float] = []
    panel_best_scores: list[float] = []
    sample_mean_scores: list[float] = []
    best_of_score_lists: dict[int, list[float]] = {n: [] for n in best_of_values}
    simplex_minus_latent: list[float] = []
    heuristic_minus_latent: list[float] = []
    heuristic_minus_simplex: list[float] = []
    winner_scores: list[float] = []
    winner_gain_vs_policy_det: list[float] = []
    winner_gain_vs_heuristic: list[float] = []
    winner_gain_vs_simplex: list[float] = []
    winner_entries: list[dict[str, Any]] = []
    win_counts: dict[str, list[float]] = {
        "simplex_beats_latent": [],
        "heuristic_beats_latent": [],
        "heuristic_beats_simplex": [],
    }
    for n in best_of_values:
        win_counts[f"bestof{n}_beats_latent"] = []
        win_counts[f"bestof{n}_beats_simplex"] = []
        win_counts[f"bestof{n}_beats_heuristic"] = []
        win_counts[f"bestof{n}_beats_panelbest"] = []

    try:
        for idx, entry in enumerate(entries):
            panel_name_to_action = {
                str(name): np.asarray(action, dtype=np.float32)
                for name, action in zip(entry["panel_names"], entry["panel_actions"])
            }
            heuristic_action = panel_name_to_action.get("heuristic")
            if heuristic_action is None:
                continue
            snapshot = entry["snapshot"]
            local_state = _to_device_dataclass(entry["local_state"], device)
            latent_action = np.asarray(entry["policy_det_action"], dtype=np.float32)
            simplex_action = np.asarray(
                _batched_bw_actions_from_snapshots(
                    actor,
                    [snapshot],
                    device=device,
                    deterministic=True,
                    bw_deterministic_readout="simplex_argmax_logprob",
                    bw_deterministic_opt_steps=int(args.bw_deterministic_opt_steps),
                    bw_deterministic_step_size=float(args.bw_deterministic_step_size),
                )["actions"][0],
                dtype=np.float32,
            )
            stochastic_actions: list[np.ndarray] = []
            with torch.inference_mode():
                for _ in range(int(sample_count)):
                    stochastic_actions.append(
                        np.asarray(actor.act_bw(local_state, deterministic=False).action.detach().cpu().numpy(), dtype=np.float32)
                    )

            base_scores = _rollout_panel_action_scores(
                eval_group,
                entry["snapshot_state"],
                panel_actions=[latent_action, simplex_action, heuristic_action],
                follow_actor=actor,
                follow_device=device,
                follow_deterministic=(str(args.policy_mode) == "deterministic"),
                bw_deterministic_readout="latent_mean_pushforward",
                bw_deterministic_opt_steps=int(args.bw_deterministic_opt_steps),
                bw_deterministic_step_size=float(args.bw_deterministic_step_size),
                gamma=float(cfg.gamma),
                k_steps=int(args.k_steps),
            )
            sample_scores = _rollout_panel_action_scores(
                eval_group,
                entry["snapshot_state"],
                panel_actions=stochastic_actions,
                follow_actor=actor,
                follow_device=device,
                follow_deterministic=(str(args.policy_mode) == "deterministic"),
                bw_deterministic_readout="latent_mean_pushforward",
                bw_deterministic_opt_steps=int(args.bw_deterministic_opt_steps),
                bw_deterministic_step_size=float(args.bw_deterministic_step_size),
                gamma=float(cfg.gamma),
                k_steps=int(args.k_steps),
            )
            latent_score, simplex_score, heuristic_score = [float(v) for v in base_scores]
            sample_mean = _mean(sample_scores)
            candidate_sources = ["latent_det", "simplex_det", "heuristic"] + [
                f"sample_{sample_idx}" for sample_idx in range(len(sample_scores))
            ]
            candidate_actions = [latent_action, simplex_action, heuristic_action] + list(stochastic_actions)
            candidate_scores = [latent_score, simplex_score, heuristic_score] + [float(v) for v in sample_scores]
            winner_idx = int(np.argmax(np.asarray(candidate_scores, dtype=np.float64)))
            winner_source = str(candidate_sources[winner_idx])
            winner_action = np.asarray(candidate_actions[winner_idx], dtype=np.float32)
            winner_score = float(candidate_scores[winner_idx])
            row: dict[str, Any] = {
                "index": int(idx),
                "best_panel_score": float(entry["best_score"]),
                "latent_det_score": latent_score,
                "simplex_det_score": simplex_score,
                "heuristic_score": heuristic_score,
                "sample_mean_score": sample_mean,
                "sample_best_of_1": float(sample_scores[0]) if sample_scores else 0.0,
                "simplex_minus_latent": float(simplex_score - latent_score),
                "heuristic_minus_latent": float(heuristic_score - latent_score),
                "heuristic_minus_simplex": float(heuristic_score - simplex_score),
            }
            for n in best_of_values:
                best_score = float(np.max(np.asarray(sample_scores[:n], dtype=np.float64))) if sample_scores[:n] else 0.0
                row[f"sample_best_of_{n}"] = best_score
                row[f"sample_best_of_{n}_minus_latent"] = float(best_score - latent_score)
                row[f"sample_best_of_{n}_minus_simplex"] = float(best_score - simplex_score)
                row[f"sample_best_of_{n}_minus_heuristic"] = float(best_score - heuristic_score)
                row[f"sample_best_of_{n}_minus_panelbest"] = float(best_score - float(entry["best_score"]))
                best_of_score_lists[n].append(best_score)
                win_counts[f"bestof{n}_beats_latent"].append(1.0 if best_score > latent_score + 1.0e-6 else 0.0)
                win_counts[f"bestof{n}_beats_simplex"].append(1.0 if best_score > simplex_score + 1.0e-6 else 0.0)
                win_counts[f"bestof{n}_beats_heuristic"].append(1.0 if best_score > heuristic_score + 1.0e-6 else 0.0)
                win_counts[f"bestof{n}_beats_panelbest"].append(1.0 if best_score > float(entry["best_score"]) + 1.0e-6 else 0.0)
            rows.append(row)
            latent_scores.append(latent_score)
            simplex_scores.append(simplex_score)
            heuristic_scores.append(heuristic_score)
            panel_best_scores.append(float(entry["best_score"]))
            sample_mean_scores.append(sample_mean)
            winner_scores.append(winner_score)
            winner_gain_vs_policy_det.append(float(winner_score - latent_score))
            winner_gain_vs_heuristic.append(float(winner_score - heuristic_score))
            winner_gain_vs_simplex.append(float(winner_score - simplex_score))
            simplex_minus_latent.append(float(simplex_score - latent_score))
            heuristic_minus_latent.append(float(heuristic_score - latent_score))
            heuristic_minus_simplex.append(float(heuristic_score - simplex_score))
            win_counts["simplex_beats_latent"].append(1.0 if simplex_score > latent_score + 1.0e-6 else 0.0)
            win_counts["heuristic_beats_latent"].append(1.0 if heuristic_score > latent_score + 1.0e-6 else 0.0)
            win_counts["heuristic_beats_simplex"].append(1.0 if heuristic_score > simplex_score + 1.0e-6 else 0.0)
            winner_entries.append(
                {
                    "index": int(idx),
                    "snapshot_state": copy.deepcopy(entry["snapshot_state"]),
                    "snapshot": copy.deepcopy(entry["snapshot"]),
                    "local_state": copy.deepcopy(entry["local_state"]),
                    "panel_names": [str(name) for name in entry["panel_names"]],
                    "panel_actions": [np.asarray(action, dtype=np.float32) for action in entry["panel_actions"]],
                    "panel_scores": [float(score) for score in entry["panel_scores"]],
                    "panel_best_score": float(entry["best_score"]),
                    "policy_det_action": np.asarray(latent_action, dtype=np.float32),
                    "policy_det_score": float(latent_score),
                    "simplex_det_action": np.asarray(simplex_action, dtype=np.float32),
                    "simplex_det_score": float(simplex_score),
                    "heuristic_action": np.asarray(heuristic_action, dtype=np.float32),
                    "heuristic_score": float(heuristic_score),
                    "sample_actions": [np.asarray(action, dtype=np.float32) for action in stochastic_actions],
                    "sample_scores": [float(score) for score in sample_scores],
                    "sample_best_of": {
                        int(n): float(np.max(np.asarray(sample_scores[:n], dtype=np.float64))) if sample_scores[:n] else 0.0
                        for n in best_of_values
                    },
                    "candidate_sources": list(candidate_sources),
                    "candidate_actions": [np.asarray(action, dtype=np.float32) for action in candidate_actions],
                    "candidate_scores": [float(score) for score in candidate_scores],
                    "winner_source": winner_source,
                    "winner_action": np.asarray(winner_action, dtype=np.float32),
                    "winner_score": float(winner_score),
                    "winner_gain_vs_policy_det": float(winner_score - latent_score),
                    "winner_gain_vs_heuristic": float(winner_score - heuristic_score),
                    "winner_gain_vs_simplex_det": float(winner_score - simplex_score),
                }
            )
    finally:
        close_structured_env_group(eval_group)

    train_indices, holdout_indices = _split_winner_entries(
        winner_entries,
        holdout_size=int(args.holdout_size),
        seed=int(args.seed),
    )
    train_index_set = set(train_indices)
    holdout_index_set = set(holdout_indices)
    winner_source_counts: dict[str, int] = {}
    for entry in winner_entries:
        split = "holdout" if int(entry["index"]) in holdout_index_set else "train"
        entry["split"] = split
        source = str(entry["winner_source"])
        winner_source_counts[source] = int(winner_source_counts.get(source, 0) + 1)

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "update": int(args.update),
        "actor_checkpoint": None if args.actor_checkpoint is None else str(Path(args.actor_checkpoint).resolve()),
        "policy_mode": str(args.policy_mode),
        "panel_state_count": int(len(rows)),
        "sample_count": int(sample_count),
        "best_of": [int(v) for v in best_of_values],
        "k_steps": int(args.k_steps),
        "bw_deterministic_opt_steps": int(args.bw_deterministic_opt_steps),
        "bw_deterministic_step_size": float(args.bw_deterministic_step_size),
        "vec_backend": str(args.vec_backend),
        "device": str(device),
        "scores": {
            "panel_best": _summarize(panel_best_scores),
            "latent_det": _summarize(latent_scores),
            "simplex_det": _summarize(simplex_scores),
            "heuristic": _summarize(heuristic_scores),
            "sample_mean": _summarize(sample_mean_scores),
        },
        "deltas": {
            "simplex_minus_latent": _summarize(simplex_minus_latent),
            "heuristic_minus_latent": _summarize(heuristic_minus_latent),
            "heuristic_minus_simplex": _summarize(heuristic_minus_simplex),
            "winner_minus_latent": _summarize(winner_gain_vs_policy_det),
            "winner_minus_heuristic": _summarize(winner_gain_vs_heuristic),
            "winner_minus_simplex": _summarize(winner_gain_vs_simplex),
        },
        "win_rates": {
            "simplex_beats_latent": _mean(win_counts["simplex_beats_latent"]),
            "heuristic_beats_latent": _mean(win_counts["heuristic_beats_latent"]),
            "heuristic_beats_simplex": _mean(win_counts["heuristic_beats_simplex"]),
        },
        "winner_bank": {
            "path": str((out_dir / "winner_bank.pt").resolve()),
            "manifest_csv": str((out_dir / "winner_bank_manifest.csv").resolve()),
            "entry_count": int(len(winner_entries)),
            "train_count": int(len(train_indices)),
            "holdout_count": int(len(holdout_indices)),
            "winner_score": _summarize(winner_scores),
            "winner_gain_vs_policy_det": _summarize(winner_gain_vs_policy_det),
            "winner_gain_vs_heuristic": _summarize(winner_gain_vs_heuristic),
            "winner_gain_vs_simplex": _summarize(winner_gain_vs_simplex),
            "winner_source_counts": winner_source_counts,
        },
    }
    for n in best_of_values:
        summary["scores"][f"sample_best_of_{n}"] = _summarize(best_of_score_lists[n])
        summary["deltas"][f"sample_best_of_{n}_minus_latent"] = _summarize(
            [float(row[f"sample_best_of_{n}_minus_latent"]) for row in rows]
        )
        summary["deltas"][f"sample_best_of_{n}_minus_simplex"] = _summarize(
            [float(row[f"sample_best_of_{n}_minus_simplex"]) for row in rows]
        )
        summary["deltas"][f"sample_best_of_{n}_minus_heuristic"] = _summarize(
            [float(row[f"sample_best_of_{n}_minus_heuristic"]) for row in rows]
        )
        summary["deltas"][f"sample_best_of_{n}_minus_panelbest"] = _summarize(
            [float(row[f"sample_best_of_{n}_minus_panelbest"]) for row in rows]
        )
        summary["win_rates"][f"sample_best_of_{n}_beats_latent"] = _mean(win_counts[f"bestof{n}_beats_latent"])
        summary["win_rates"][f"sample_best_of_{n}_beats_simplex"] = _mean(win_counts[f"bestof{n}_beats_simplex"])
        summary["win_rates"][f"sample_best_of_{n}_beats_heuristic"] = _mean(win_counts[f"bestof{n}_beats_heuristic"])
        summary["win_rates"][f"sample_best_of_{n}_beats_panelbest"] = _mean(win_counts[f"bestof{n}_beats_panelbest"])

    torch.save(
        {
            "meta": {
                "run_dir": str(run_dir),
                "update": int(args.update),
                "actor_checkpoint": None if args.actor_checkpoint is None else str(Path(args.actor_checkpoint).resolve()),
                "policy_mode": str(args.policy_mode),
                "panel_state_count": int(len(winner_entries)),
                "sample_count": int(sample_count),
                "best_of": [int(v) for v in best_of_values],
                "k_steps": int(args.k_steps),
                "bw_deterministic_opt_steps": int(args.bw_deterministic_opt_steps),
                "bw_deterministic_step_size": float(args.bw_deterministic_step_size),
                "holdout_size": int(args.holdout_size),
                "seed": int(args.seed),
            },
            "train_indices": train_indices,
            "holdout_indices": holdout_indices,
            "entries": winner_entries,
        },
        out_dir / "winner_bank.pt",
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(out_dir / "per_state_rows.csv", rows)
    _write_csv(out_dir / "winner_bank_manifest.csv", _winner_bank_manifest_rows(winner_entries))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
