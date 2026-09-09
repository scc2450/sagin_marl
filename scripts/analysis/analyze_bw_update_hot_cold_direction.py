from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_bw_update_direction import (
    _bw_action_from_actor,
    collect_bw_snapshot_panel,
)
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--panel_episodes", type=int, default=8)
    parser.add_argument("--panel_states", type=int, default=16)
    parser.add_argument("--panel_seed", type=int, default=35000)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--out_json", default=None)
    return parser


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 0:
        return 0.0
    return float(np.mean(arr))


def _safe_frac(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    arr = np.asarray(flags, dtype=np.float64)
    return float(np.mean(arr))


def _decode_action_by_gu(
    action: np.ndarray,
    stage_candidates: list[list[int]],
    num_gu: int,
) -> np.ndarray:
    action_arr = np.asarray(action, dtype=np.float64)
    if action_arr.ndim == 1:
        action_arr = action_arr[None, :]
    gu_alloc = np.zeros((int(num_gu),), dtype=np.float64)
    for u in range(action_arr.shape[0]):
        candidates_u = list(stage_candidates[u]) if u < len(stage_candidates) else []
        for slot, value in enumerate(action_arr[u].reshape(-1).tolist()):
            gu_idx = int(candidates_u[slot]) if slot < len(candidates_u) else -1
            if 0 <= gu_idx < int(num_gu):
                gu_alloc[gu_idx] += float(value)
    return gu_alloc


def _hot_state_summary(entry: dict[str, Any], actor_action: np.ndarray) -> dict[str, Any]:
    snapshot_state = dict(entry["snapshot_state"])
    env_state = dict(snapshot_state.get("env_state") or {})
    hotspot_mask = np.asarray(env_state.get("last_hotspot_mask", []), dtype=np.float64).reshape(-1)
    num_gu = int(hotspot_mask.size)
    stage_candidates = [list(c) for c in (snapshot_state.get("stage_candidates") or [])]
    heuristic_action = np.asarray(entry["heuristic_action"], dtype=np.float32)
    actor_alloc = _decode_action_by_gu(actor_action, stage_candidates, num_gu)
    heur_alloc = _decode_action_by_gu(heuristic_action, stage_candidates, num_gu)
    hot_idx = np.flatnonzero(hotspot_mask > 0.5).astype(np.int64)
    cold_idx = np.flatnonzero(hotspot_mask <= 0.5).astype(np.int64)
    hot_count = int(hot_idx.size)
    cold_count = int(cold_idx.size)
    hot_alloc = float(actor_alloc[hot_idx].sum()) if hot_count > 0 else 0.0
    cold_alloc = float(actor_alloc[cold_idx].sum()) if cold_count > 0 else float(actor_alloc.sum())
    heur_hot_alloc = float(heur_alloc[hot_idx].sum()) if hot_count > 0 else 0.0
    heur_cold_alloc = float(heur_alloc[cold_idx].sum()) if cold_count > 0 else float(heur_alloc.sum())
    hot_share = hot_alloc / max(float(hot_alloc + cold_alloc), 1.0e-8)
    heur_hot_share = heur_hot_alloc / max(float(heur_hot_alloc + heur_cold_alloc), 1.0e-8)
    return {
        "episode": int(entry["episode"]),
        "t": int(entry["t"]),
        "hot_idx": [int(x) for x in hot_idx.tolist()],
        "cold_idx": [int(x) for x in cold_idx.tolist()],
        "hot_count": int(hot_count),
        "cold_count": int(cold_count),
        "hot_alloc": float(hot_alloc),
        "cold_alloc": float(cold_alloc),
        "hot_minus_cold": float(hot_alloc - cold_alloc),
        "hot_share": float(hot_share),
        "heur_hot_alloc": float(heur_hot_alloc),
        "heur_cold_alloc": float(heur_cold_alloc),
        "heur_hot_share": float(heur_hot_share),
        "abs_hot_share_gap_to_heur": float(abs(hot_share - heur_hot_share)),
        "action_l1_to_heur": float(np.abs(actor_alloc - heur_alloc).sum()),
    }


def _load_actor(run_dir: Path, cfg: Any, ckpt_path: Path, device: torch.device):
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    load_checkpoint_forgiving(bundle.actor, str(ckpt_path), map_location=device, strict=True)
    actor = bundle.actor.to(device)
    actor.eval()
    return actor


def _merge_probe_metrics(run_dir: Path) -> dict[int, dict[str, float]]:
    csv_path = run_dir / "update_direction_probe.csv"
    if not csv_path.exists():
        return {}
    rows: dict[int, dict[str, float]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            update = int(row["update"])
            rows[update] = {
                key: float(value)
                for key, value in row.items()
                if key != "update" and key != "alignment_judgement" and value not in {"", None}
                and key
                not in {
                    "branch_ref_mode",
                    "branch_follow_policy_mode",
                    "actor_policy_mode",
                }
            }
            rows[update]["alignment_judgement"] = str(row.get("alignment_judgement", "") or "")
    return rows


def main() -> None:
    args = _build_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    config_path = Path(args.config).resolve() if args.config else (run_dir / "config_source.yaml")
    cfg = load_config(str(config_path))
    device = torch.device(str(args.device))
    panel = collect_bw_snapshot_panel(
        cfg,
        episodes=int(args.panel_episodes),
        states=int(args.panel_states),
        seed=int(args.panel_seed),
    )
    probe_metrics = _merge_probe_metrics(run_dir)

    state_groups = {"off": [], "hot0": [], "hot1": [], "other_hot": []}
    for idx, entry in enumerate(panel):
        env_state = dict(entry["snapshot_state"].get("env_state") or {})
        hot_mask = np.asarray(env_state.get("last_hotspot_mask", []), dtype=np.float64).reshape(-1)
        hot_idx = np.flatnonzero(hot_mask > 0.5).astype(np.int64)
        if hot_idx.size <= 0:
            state_groups["off"].append(int(idx))
        elif hot_idx.size == 1 and int(hot_idx[0]) == 0:
            state_groups["hot0"].append(int(idx))
        elif hot_idx.size == 1 and int(hot_idx[0]) == 1:
            state_groups["hot1"].append(int(idx))
        else:
            state_groups["other_hot"].append(int(idx))

    rows: list[dict[str, Any]] = []
    per_update_states: dict[int, list[dict[str, Any]]] = {}
    prev_state_summaries: list[dict[str, Any]] | None = None

    ckpt_paths = sorted(run_dir.glob("actor_u*.pt"))
    for ckpt_path in ckpt_paths:
        stem = ckpt_path.stem
        update = int(stem.split("_u")[-1])
        actor = _load_actor(run_dir, cfg, ckpt_path, device)
        state_summaries: list[dict[str, Any]] = []
        for entry in panel:
            actor_action = _bw_action_from_actor(actor, entry["snapshot"], device, deterministic=True)
            state_summaries.append(_hot_state_summary(entry, actor_action))
        per_update_states[update] = state_summaries

        hot_states = [s for s in state_summaries if s["hot_count"] > 0]
        hot0_states = [s for s in hot_states if s["hot_idx"] == [0]]
        hot1_states = [s for s in hot_states if s["hot_idx"] == [1]]

        row: dict[str, Any] = {
            "update": int(update),
            "panel_hot_states": int(len(hot_states)),
            "panel_off_states": int(len(state_groups["off"])),
            "mean_hot_alloc": _safe_mean([s["hot_alloc"] for s in hot_states]),
            "mean_cold_alloc": _safe_mean([s["cold_alloc"] for s in hot_states]),
            "mean_hot_minus_cold": _safe_mean([s["hot_minus_cold"] for s in hot_states]),
            "mean_hot_share": _safe_mean([s["hot_share"] for s in hot_states]),
            "mean_heur_hot_share": _safe_mean([s["heur_hot_share"] for s in hot_states]),
            "mean_abs_hot_share_gap_to_heur": _safe_mean([s["abs_hot_share_gap_to_heur"] for s in hot_states]),
            "mean_action_l1_to_heur_by_gu": _safe_mean([s["action_l1_to_heur"] for s in hot_states]),
            "hot_gt_cold_frac": _safe_frac([s["hot_alloc"] > s["cold_alloc"] + 1.0e-9 for s in hot_states]),
            "moved_hot_enough_frac_vs_heur": _safe_frac(
                [s["hot_share"] >= s["heur_hot_share"] - 1.0e-9 for s in hot_states]
            ),
            "mean_hot_share_hot0": _safe_mean([s["hot_share"] for s in hot0_states]),
            "mean_hot_share_hot1": _safe_mean([s["hot_share"] for s in hot1_states]),
        }

        if prev_state_summaries is None:
            row.update(
                {
                    "delta_mean_hot_alloc_from_prev": 0.0,
                    "delta_mean_cold_alloc_from_prev": 0.0,
                    "delta_mean_hot_minus_cold_from_prev": 0.0,
                    "delta_mean_hot_share_from_prev": 0.0,
                    "hot_share_increased_frac_from_prev": 0.0,
                    "cold_alloc_decreased_frac_from_prev": 0.0,
                    "moved_closer_to_heur_hot_share_frac_from_prev": 0.0,
                    "moved_further_from_heur_hot_share_frac_from_prev": 0.0,
                }
            )
        else:
            delta_hot_alloc = []
            delta_cold_alloc = []
            delta_hot_minus_cold = []
            delta_hot_share = []
            hot_share_increased = []
            cold_alloc_decreased = []
            moved_closer = []
            moved_further = []
            for prev_s, curr_s in zip(prev_state_summaries, state_summaries):
                if curr_s["hot_count"] <= 0:
                    continue
                dh = float(curr_s["hot_alloc"] - prev_s["hot_alloc"])
                dc = float(curr_s["cold_alloc"] - prev_s["cold_alloc"])
                dgap = float(curr_s["hot_minus_cold"] - prev_s["hot_minus_cold"])
                dshare = float(curr_s["hot_share"] - prev_s["hot_share"])
                prev_gap = float(abs(prev_s["hot_share"] - prev_s["heur_hot_share"]))
                curr_gap = float(abs(curr_s["hot_share"] - curr_s["heur_hot_share"]))
                delta_hot_alloc.append(dh)
                delta_cold_alloc.append(dc)
                delta_hot_minus_cold.append(dgap)
                delta_hot_share.append(dshare)
                hot_share_increased.append(dshare > 1.0e-9)
                cold_alloc_decreased.append(dc < -1.0e-9)
                moved_closer.append(curr_gap + 1.0e-9 < prev_gap)
                moved_further.append(curr_gap > prev_gap + 1.0e-9)
            row.update(
                {
                    "delta_mean_hot_alloc_from_prev": _safe_mean(delta_hot_alloc),
                    "delta_mean_cold_alloc_from_prev": _safe_mean(delta_cold_alloc),
                    "delta_mean_hot_minus_cold_from_prev": _safe_mean(delta_hot_minus_cold),
                    "delta_mean_hot_share_from_prev": _safe_mean(delta_hot_share),
                    "hot_share_increased_frac_from_prev": _safe_frac(hot_share_increased),
                    "cold_alloc_decreased_frac_from_prev": _safe_frac(cold_alloc_decreased),
                    "moved_closer_to_heur_hot_share_frac_from_prev": _safe_frac(moved_closer),
                    "moved_further_from_heur_hot_share_frac_from_prev": _safe_frac(moved_further),
                }
            )
        prev_state_summaries = state_summaries

        if update in probe_metrics:
            row.update(probe_metrics[update])
        rows.append(row)

    out_csv = Path(args.out_csv) if args.out_csv else (run_dir / "update_hot_cold_direction.csv")
    out_json = Path(args.out_json) if args.out_json else (run_dir / "update_hot_cold_direction.json")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "panel_states": int(len(panel)),
        "panel_state_groups": {key: [int(v) for v in values] for key, values in state_groups.items()},
        "rows": rows,
        "per_update_states": {
            str(update): state_summaries for update, state_summaries in per_update_states.items()
        },
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
