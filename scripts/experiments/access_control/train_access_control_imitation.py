from __future__ import annotations

import argparse
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
import torch.nn.functional as F

from scripts.experiments.access_control.access_control_imitation_common import AccessBidScorer, AccessUserGroup, build_overlap_user_choices, pair_feature_names
from scripts.experiments.access_control.evaluate_structured_access_control_oracle import _write_csv
from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_train import (
    close_structured_env_group,
    make_structured_driver,
)


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_bank(path: Path) -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, list):
        raise TypeError(f"Expected list payload in {path}")
    return payload


def _build_groups(
    cfg,
    bank_rows: list[dict[str, Any]],
    *,
    overlap_per_uav_keep: int,
    overlap_per_user_max_uav: int,
) -> tuple[list[AccessUserGroup], dict[str, Any]]:
    driver = make_structured_driver(cfg, backend="sync")
    groups: list[AccessUserGroup] = []
    total_snapshots = 0
    skipped_rows = 0
    try:
        for row in bank_rows:
            snapshot_state = dict(row["snapshot_state"])
            selected_assoc = np.asarray(row["selected_assoc"], dtype=np.int32)
            if selected_assoc.ndim != 1:
                skipped_rows += 1
                continue
            driver.load_bw_stage_state(snapshot_state)
            row_groups, _meta = build_overlap_user_choices(
                driver.env,
                base_assoc=np.asarray(snapshot_state["stage_assoc"], dtype=np.int32),
                selected_assoc=selected_assoc,
                overlap_per_uav_keep=int(overlap_per_uav_keep),
                overlap_per_user_max_uav=int(overlap_per_user_max_uav),
            )
            groups.extend(row_groups)
            total_snapshots += 1
    finally:
        close_structured_env_group(driver)
    return groups, {
        "snapshot_count": int(total_snapshots),
        "group_count": int(len(groups)),
        "skipped_rows": int(skipped_rows),
        "feature_names": pair_feature_names(),
    }


def _split_groups(groups: list[AccessUserGroup], holdout_frac: float, seed: int) -> tuple[list[AccessUserGroup], list[AccessUserGroup]]:
    indices = list(range(len(groups)))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    holdout_size = max(1, int(round(float(len(indices)) * float(holdout_frac)))) if indices else 0
    holdout_idx = set(indices[:holdout_size])
    train = [groups[i] for i in range(len(groups)) if i not in holdout_idx]
    holdout = [groups[i] for i in range(len(groups)) if i in holdout_idx]
    return train, holdout


def _iter_batches(groups: list[AccessUserGroup], batch_size: int):
    for start in range(0, len(groups), max(int(batch_size), 1)):
        yield groups[start : start + max(int(batch_size), 1)]


def _evaluate_groups(model: AccessBidScorer, groups: list[AccessUserGroup], device: torch.device) -> dict[str, float]:
    if not groups:
        return {"loss": 0.0, "top1_acc": 0.0, "group_count": 0.0}
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0
    with torch.inference_mode():
        for group in groups:
            feats = torch.as_tensor(group.features, device=device)
            logits = model(feats).squeeze(-1).unsqueeze(0)
            target = torch.tensor([int(group.target_index)], dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, target, reduction="mean")
            pred = int(torch.argmax(logits, dim=1).item())
            total_loss += float(loss.item())
            total_correct += float(pred == int(group.target_index))
            total_count += 1
    return {
        "loss": float(total_loss / max(total_count, 1)),
        "top1_acc": float(total_correct / max(total_count, 1)),
        "group_count": float(total_count),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--bank", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-5)
    parser.add_argument("--holdout_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overlap_per_uav_keep", type=int, default=6)
    parser.add_argument("--overlap_per_user_max_uav", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_all_seeds(int(args.seed))
    cfg = load_config(args.config)
    bank_rows = _load_bank(Path(args.bank))
    groups, meta = _build_groups(
        cfg,
        bank_rows,
        overlap_per_uav_keep=int(args.overlap_per_uav_keep),
        overlap_per_user_max_uav=int(args.overlap_per_user_max_uav),
    )
    train_groups, holdout_groups = _split_groups(groups, float(args.holdout_frac), int(args.seed))
    device = torch.device(args.device)
    model = AccessBidScorer(input_dim=len(pair_feature_names()), hidden_dim=int(args.hidden_dim)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    history: list[dict[str, float]] = []
    best_holdout_acc = -1.0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "access_bid_model.pt"
    for epoch in range(1, max(int(args.epochs), 1) + 1):
        random.shuffle(train_groups)
        model.train()
        train_loss_sum = 0.0
        train_correct = 0.0
        train_count = 0
        for batch in _iter_batches(train_groups, int(args.batch_size)):
            feats = torch.as_tensor(
                np.concatenate([group.features for group in batch], axis=0),
                dtype=torch.float32,
                device=device,
            )
            logits_flat = model(feats).squeeze(-1)
            losses = []
            offset = 0
            for group in batch:
                count = int(group.features.shape[0])
                group_logits = logits_flat[offset : offset + count].unsqueeze(0)
                target = torch.tensor([int(group.target_index)], dtype=torch.long, device=device)
                loss = F.cross_entropy(group_logits, target, reduction="mean") * float(group.weight)
                losses.append(loss)
                pred = int(torch.argmax(group_logits, dim=1).item())
                train_correct += float(pred == int(group.target_index))
                train_count += 1
                offset += count
            loss = torch.stack(losses).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss_sum += float(loss.item()) * float(len(batch))
        train_metrics = {
            "train_loss": float(train_loss_sum / max(train_count, 1)),
            "train_top1_acc": float(train_correct / max(train_count, 1)),
        }
        holdout_metrics = _evaluate_groups(model, holdout_groups, device)
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_metrics["train_loss"]),
            "train_top1_acc": float(train_metrics["train_top1_acc"]),
            "holdout_loss": float(holdout_metrics["loss"]),
            "holdout_top1_acc": float(holdout_metrics["top1_acc"]),
        }
        history.append(row)
        if float(holdout_metrics["top1_acc"]) > best_holdout_acc:
            best_holdout_acc = float(holdout_metrics["top1_acc"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_names": pair_feature_names(),
                    "hidden_dim": int(args.hidden_dim),
                    "config": str(args.config),
                    "bank": str(args.bank),
                    "overlap_per_uav_keep": int(args.overlap_per_uav_keep),
                    "overlap_per_user_max_uav": int(args.overlap_per_user_max_uav),
                },
                best_path,
            )
    _write_csv(out_dir / "train_history.csv", history)
    summary = {
        "config": str(args.config),
        "bank": str(args.bank),
        "device": str(device),
        "seed": int(args.seed),
        "hidden_dim": int(args.hidden_dim),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "holdout_frac": float(args.holdout_frac),
        "group_build_meta": meta,
        "train_group_count": int(len(train_groups)),
        "holdout_group_count": int(len(holdout_groups)),
        "best_holdout_top1_acc": float(best_holdout_acc),
        "final_train_loss": float(history[-1]["train_loss"]) if history else 0.0,
        "final_train_top1_acc": float(history[-1]["train_top1_acc"]) if history else 0.0,
        "final_holdout_loss": float(history[-1]["holdout_loss"]) if history else 0.0,
        "final_holdout_top1_acc": float(history[-1]["holdout_top1_acc"]) if history else 0.0,
        "model_path": str(best_path),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(
        f"Summary: train_groups={len(train_groups)} holdout_groups={len(holdout_groups)} "
        f"best_holdout_top1_acc={best_holdout_acc:.4f}"
    )


if __name__ == "__main__":
    main()
