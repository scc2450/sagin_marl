from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
MATRIX_PATH = SCRIPT_DIR / "evaluate_partner_swap_matrix.py"
SPEC = importlib.util.spec_from_file_location("evaluate_partner_swap_matrix", MATRIX_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Failed to load module spec from {MATRIX_PATH}")
MATRIX_MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MATRIX_MOD
SPEC.loader.exec_module(MATRIX_MOD)

CellSpec = MATRIX_MOD.CellSpec
evaluate_cell = MATRIX_MOD.evaluate_cell


def _parse_updates(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("No checkpoint updates were provided.")
    return out


def _load_metrics_index(metrics_path: Path) -> dict[int, dict[str, str]]:
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    index: dict[int, dict[str, str]] = {}
    for row in rows:
        step_raw = row.get("step")
        if step_raw is None:
            continue
        index[int(step_raw)] = row
    return index


def _load_checkpoint_eval_index(checkpoint_eval_path: Path) -> dict[int, dict[str, str]]:
    with checkpoint_eval_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    index: dict[int, dict[str, str]] = {}
    for row in rows:
        update_raw = row.get("update")
        if update_raw is None:
            continue
        index[int(update_raw)] = row
    return index


def _metric_row_for_update(metrics_index: dict[int, dict[str, str]], update: int) -> tuple[int | None, dict[str, str] | None]:
    if update in metrics_index:
        return update, metrics_index[update]
    if (update - 1) in metrics_index:
        return update - 1, metrics_index[update - 1]
    return None, None


def _build_cells(config_path: str, checkpoint_path: str, update: int) -> list[tuple[str, CellSpec]]:
    prefix = f"u{update:04d}"
    return [
        (
            "heuristic_all",
            CellSpec(
                name=f"{prefix}__heuristic_all",
                bw_label="heuristic",
                partner_label="heuristic_all",
                config_path=config_path,
                primary_checkpoint=checkpoint_path,
                exec_accel_source="cluster_center_queue_aware",
                exec_bw_source="cluster_center_queue_aware",
                exec_sat_source="cluster_center_queue_aware",
            ),
        ),
        (
            "native_joint",
            CellSpec(
                name=f"{prefix}__native_joint",
                bw_label="joint",
                partner_label="native_joint",
                config_path=config_path,
                primary_checkpoint=checkpoint_path,
                exec_accel_source="policy",
                exec_bw_source="policy",
                exec_sat_source="policy",
            ),
        ),
        (
            "accel_only",
            CellSpec(
                name=f"{prefix}__accel_only",
                bw_label="heur_bw",
                partner_label="learned_accel",
                config_path=config_path,
                primary_checkpoint=checkpoint_path,
                exec_accel_source="policy",
                exec_bw_source="cluster_center_queue_aware",
                exec_sat_source="cluster_center_queue_aware",
            ),
        ),
        (
            "bw_only",
            CellSpec(
                name=f"{prefix}__bw_only",
                bw_label="learned_bw",
                partner_label="heur_accel_sat",
                config_path=config_path,
                primary_checkpoint=checkpoint_path,
                exec_accel_source="cluster_center_queue_aware",
                exec_bw_source="policy",
                exec_sat_source="cluster_center_queue_aware",
            ),
        ),
        (
            "sat_only",
            CellSpec(
                name=f"{prefix}__sat_only",
                bw_label="heur_bw",
                partner_label="learned_sat",
                config_path=config_path,
                primary_checkpoint=checkpoint_path,
                exec_accel_source="cluster_center_queue_aware",
                exec_bw_source="cluster_center_queue_aware",
                exec_sat_source="policy",
            ),
        ),
    ]


def _safe_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    return float(value)


def _delta_against(anchor: dict[str, Any], row: dict[str, Any], metric: str) -> float | None:
    av = _safe_float(anchor, metric)
    rv = _safe_float(row, metric)
    if av is None or rv is None:
        return None
    return rv - av


def _closeness_to_native(anchor: dict[str, Any], native: dict[str, Any], row: dict[str, Any]) -> float | None:
    metrics = [
        "reward_sum",
        "processed_ratio_eval",
        "drop_ratio_eval",
        "pre_backlog_steps_eval",
    ]
    score = 0.0
    count = 0
    for metric in metrics:
        native_delta = _delta_against(anchor, native, metric)
        row_delta = _delta_against(anchor, row, metric)
        if native_delta is None or row_delta is None:
            continue
        denom = max(abs(native_delta), 1.0e-9)
        score += abs(row_delta - native_delta) / denom
        count += 1
    if count == 0:
        return None
    return score / float(count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=str,
        default=r"runs\phase1_actions\joint_3heads_fading_interference_ka_vsat_joint_puremappo_criticdecoupled_u0600_subproc12_t2_20260331",
    )
    parser.add_argument(
        "--updates",
        type=str,
        default="200,250,300,350,400,450,500,550",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode-seed-base", type=int, default=42000)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=r"runs\joint_head_timeline_u0600_20260401",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config_path = str(run_dir / "config_source.yaml")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_index = _load_metrics_index(run_dir / "metrics.csv")
    checkpoint_eval_index = _load_checkpoint_eval_index(run_dir / "checkpoint_eval.csv")
    updates = _parse_updates(args.updates)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timeline_rows: list[dict[str, Any]] = []
    per_update_summary: list[dict[str, Any]] = []

    for update in updates:
        checkpoint_path = str(run_dir / f"actor_u{update:04d}.pt")
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        print(f"[head-timeline] evaluating update {update}")

        update_rows: dict[str, dict[str, Any]] = {}
        for label, cell in _build_cells(config_path, checkpoint_path, update):
            rows, _ = evaluate_cell(
                cell=cell,
                episodes=int(args.episodes),
                episode_seed_base=args.episode_seed_base,
                device=device,
            )
            if not rows:
                raise RuntimeError(f"No evaluation rows returned for {cell.name}")
            row = dict(rows[0])
            row["update"] = int(update)
            row["cell_label"] = label
            update_rows[label] = row
            timeline_rows.append(row)

        metric_step, metric_row = _metric_row_for_update(metrics_index, update)
        checkpoint_eval_row = checkpoint_eval_index.get(update)
        training_fields = [
            "approx_kl",
            "clip_frac",
            "entropy_accel",
            "entropy_bw",
            "entropy_sat",
            "log_ratio_abs_mean",
            "processed_ratio_eval",
            "drop_ratio_eval",
            "pre_backlog_steps_eval",
            "reward_raw",
        ]
        training_snapshot = {
            "metrics_step": metric_step,
        }
        for key in training_fields:
            training_snapshot[key] = None if metric_row is None else _safe_float(metric_row, key)

        checkpoint_eval_snapshot = {}
        for key in ["reward_sum", "processed_ratio_eval", "drop_ratio_eval", "pre_backlog_steps_eval"]:
            checkpoint_eval_snapshot[key] = None if checkpoint_eval_row is None else _safe_float(checkpoint_eval_row, key)

        heuristic_row = update_rows["heuristic_all"]
        native_row = update_rows["native_joint"]
        delta_metrics = [
            "reward_sum",
            "processed_ratio_eval",
            "drop_ratio_eval",
            "pre_backlog_steps_eval",
            "assoc_dist_mean",
            "assoc_count_step_max_mean",
            "bw_valid_count_p90",
        ]
        deltas_vs_heuristic: dict[str, dict[str, float | None]] = {}
        for label, row in update_rows.items():
            metric_deltas: dict[str, float | None] = {}
            for metric in delta_metrics:
                metric_deltas[metric] = _delta_against(heuristic_row, row, metric)
            metric_deltas["closeness_to_native"] = _closeness_to_native(heuristic_row, native_row, row)
            deltas_vs_heuristic[label] = metric_deltas

        candidate_scores = {
            label: deltas_vs_heuristic[label]["closeness_to_native"]
            for label in ("accel_only", "bw_only", "sat_only")
        }
        ranked_candidates = sorted(
            (
                (label, score)
                for label, score in candidate_scores.items()
                if score is not None
            ),
            key=lambda item: item[1],
        )
        per_update_summary.append(
            {
                "update": int(update),
                "training_snapshot": training_snapshot,
                "checkpoint_eval_snapshot": checkpoint_eval_snapshot,
                "rows": update_rows,
                "deltas_vs_heuristic": deltas_vs_heuristic,
                "closest_single_head_to_native": None if not ranked_candidates else ranked_candidates[0][0],
                "closest_single_head_score": None if not ranked_candidates else ranked_candidates[0][1],
                "single_head_ranking": ranked_candidates,
            }
        )

    csv_path = out_dir / "timeline_rows.csv"
    if timeline_rows:
        fieldnames = list(timeline_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(timeline_rows)

    summary_path = out_dir / "summary.json"
    payload = {
        "meta": {
            "run_dir": str(run_dir.resolve()),
            "updates": updates,
            "episodes": int(args.episodes),
            "episode_seed_base": int(args.episode_seed_base) if args.episode_seed_base is not None else None,
            "device": str(device),
        },
        "per_update": per_update_summary,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote timeline rows to {csv_path}")
    print(f"Wrote summary json to {summary_path}")


if __name__ == "__main__":
    main()
