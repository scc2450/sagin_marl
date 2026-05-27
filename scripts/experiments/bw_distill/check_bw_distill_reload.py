from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnostics.audit.audit_bw_broad2local_offline import _load_actor
from scripts.experiments.bw_distill.distill_bw_winner_bank_v0 import _entries_by_indices, _evaluate_split, _load_winner_bank

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--winner_bank", type=str, required=True)
    parser.add_argument("--follow_run_dir", type=str, required=True)
    parser.add_argument("--follow_update", type=int, required=True)
    parser.add_argument("--follow_checkpoint", type=str, required=True)
    parser.add_argument("--student_checkpoint", type=str, required=True)
    parser.add_argument("--split", choices=["train", "holdout", "all"], default="holdout")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--vec_backend", choices=["sync", "subproc"], default="sync")
    parser.add_argument("--k_steps", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bank = _load_winner_bank(Path(args.winner_bank))
    entries = list(bank["entries"])
    if str(args.split) == "train":
        entries = _entries_by_indices(entries, list(bank["train_indices"]))
    elif str(args.split) == "holdout":
        entries = _entries_by_indices(entries, list(bank["holdout_indices"]))
    device = torch.device(args.device)
    cfg, follow_actor = _load_actor(
        Path(args.follow_run_dir),
        int(args.follow_update),
        device,
        actor_checkpoint=str(args.follow_checkpoint),
    )
    _, student_actor = _load_actor(
        Path(args.follow_run_dir),
        int(args.follow_update),
        device,
        actor_checkpoint=str(args.student_checkpoint),
    )
    follow_actor.eval()
    student_actor.eval()
    summary, _rows = _evaluate_split(
        entries=entries,
        student_actor=student_actor,
        follow_actor=follow_actor,
        cfg=cfg,
        device=device,
        num_envs=int(args.num_envs),
        vec_backend=str(args.vec_backend),
        k_steps=int(args.k_steps),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
