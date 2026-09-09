from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


METRIC_FIELDS = {
    "reward": ("rollout_episode_reward_mean", "episode_reward_mean"),
    "processed": ("rollout_processed_ratio_eval_mean", "processed_ratio_eval"),
    "drop": ("rollout_drop_ratio_eval_mean", "drop_ratio_eval"),
    "pre_backlog": ("rollout_pre_backlog_steps_eval_mean", "pre_backlog_steps_eval"),
    "sat_overlap": ("rollout_sat_overlap_eval_mean", "sat_overlap_eval"),
    "bw_kl": ("bw_actor_full_update_kl", "bw_actor_full_kl"),
    "bw_clip": ("bw_actor_full_update_clip_frac", "bw_actor_full_clip_frac"),
    "bw_abs_dlogp": ("bw_actor_delta_logprob_abs_mean",),
    "credit_mean": ("bw_actor_credit_mean",),
    "credit_positive_frac": ("bw_actor_credit_positive_frac",),
    "pos_up": (
        "bw_actor_credit_adv_positive_delta_positive_frac",
        "bw_actor_raw_adv_positive_delta_positive_frac",
        "bw_actor_delta_logprob_when_raw_adv_positive_frac",
    ),
    "neg_down": ("bw_actor_credit_adv_negative_delta_negative_frac",),
    "raw_pos_up": ("bw_actor_raw_adv_positive_delta_positive_frac",),
    "guard_accepted": ("bw_actor_guard_accepted",),
    "guard_skipped": ("bw_actor_guard_skipped",),
    "guard_step_scale": ("bw_actor_guard_step_scale",),
    "branch_product": (
        "bw_actor_local_adv_branch_product",
        "bw_actor_guard_branch_mean_delta_times_delta_logprob_mean",
        "bw_actor_raw_branch_delta_times_delta_logprob_mean",
    ),
    "local_adv_enabled": ("bw_actor_local_adv_enabled",),
    "local_adv_product": ("bw_actor_local_adv_branch_product",),
    "local_adv_pos_up": ("bw_actor_local_adv_branch_positive_logprob_up_frac",),
    "local_adv_nonzero": ("bw_actor_local_adv_target_nonzero_frac",),
    "local_adv_sample_count": ("bw_actor_local_adv_sample_count",),
    "probe_h2_product": ("bw_probe_native_h2_mean_branch_delta_times_delta_logprob",),
    "probe_h5_product": ("bw_probe_native_h5_mean_branch_delta_times_delta_logprob",),
    "probe_h10_product": ("bw_probe_native_h10_mean_branch_delta_times_delta_logprob",),
}

EVAL_FIELDS = {
    "eval_reward": "reward_sum",
    "eval_processed": "processed_ratio_eval",
    "eval_drop": "drop_ratio_eval",
    "eval_pre_backlog": "pre_backlog_steps_eval",
    "eval_sat_overlap": "sat_overlap_eval",
    "eval_collision": "collision_episode_fraction",
}


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _fmt(value: object, digits: int = 4) -> str:
    number = _safe_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def _first_value(row: dict[str, str], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = _safe_float(row.get(name))
        if value is not None:
            return value
    return None


def _mean_field(rows: list[dict[str, str]], names: tuple[str, ...]) -> float | None:
    values: list[float] = []
    for row in rows:
        value = _first_value(row, names)
        if value is not None:
            values.append(value)
    return _mean(values)


def _read_metrics(run_dir: Path) -> list[dict[str, str]]:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return []
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _read_eval_means(run_dir: Path) -> dict[str, float]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("native_eval_seed*/*_summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        summary = payload.get("summary") if isinstance(payload, dict) else None
        if isinstance(summary, dict):
            summaries.append(summary)
    out: dict[str, float] = {}
    for out_key, json_key in EVAL_FIELDS.items():
        values = []
        for summary in summaries:
            value = _safe_float(summary.get(json_key))
            if value is not None:
                values.append(value)
        mean_value = _mean(values)
        if mean_value is not None:
            out[out_key] = mean_value
    return out


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _infer_rows_from_dirs(run_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pattern = re.compile(r"(?P<scenario>.+)_(?P<method>[^_]+(?:_[^_]+)*)_u(?P<base>\d+)to(?P<target>\d+)$")
    for child in sorted(run_root.iterdir()):
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        rows.append(
            {
                "job_id": "",
                "phase": "",
                "scenario": match.group("scenario") if match else "",
                "base_key": "",
                "base_update": match.group("base") if match else "",
                "target_update": match.group("target") if match else "",
                "fixed_tau": "",
                "method": match.group("method") if match else child.name,
                "run_dir": str(child),
                "checkpoint": "",
                "command": "",
            }
        )
    return rows


def _selected_tail(rows: list[dict[str, str]], tail_rows: int) -> list[dict[str, str]]:
    if tail_rows <= 0:
        return rows
    return rows[-tail_rows:]


def summarize_run(manifest_row: dict[str, str], *, tail_rows: int) -> dict[str, object]:
    run_dir = Path(manifest_row.get("run_dir", ""))
    metrics = _read_metrics(run_dir)
    selected = _selected_tail(metrics, tail_rows)
    last = metrics[-1] if metrics else {}

    out: dict[str, object] = {
        "job_id": manifest_row.get("job_id", ""),
        "phase": manifest_row.get("phase", ""),
        "scenario": manifest_row.get("scenario", ""),
        "base_key": manifest_row.get("base_key", ""),
        "base_update": manifest_row.get("base_update", ""),
        "target_update": manifest_row.get("target_update", ""),
        "fixed_tau": manifest_row.get("fixed_tau", ""),
        "method": manifest_row.get("method", ""),
        "run_dir": str(run_dir),
        "status": "ok" if metrics else "missing_metrics",
        "updates_recorded": len(metrics),
    }
    update_value = _safe_float(last.get("update"))
    if update_value is not None:
        out["last_update"] = int(update_value)

    for out_key, names in METRIC_FIELDS.items():
        if out_key in {"reward", "processed", "drop", "pre_backlog", "sat_overlap"}:
            out[out_key] = _first_value(last, names)
        else:
            out[out_key] = _mean_field(selected, names)

    out.update(_read_eval_means(run_dir))
    return out


def add_freeze_relative(rows: list[dict[str, object]]) -> None:
    by_scenario: dict[str, dict[str, object]] = {}
    for row in rows:
        method = str(row.get("method", ""))
        if method == "freeze_bw":
            by_scenario[str(row.get("scenario", ""))] = row
    for row in rows:
        freeze = by_scenario.get(str(row.get("scenario", "")))
        if not freeze:
            continue
        for key in ("reward", "eval_reward"):
            value = _safe_float(row.get(key))
            freeze_value = _safe_float(freeze.get(key))
            if value is not None and freeze_value is not None:
                row[f"{key}_vs_freeze"] = value - freeze_value


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "job_id",
        "phase",
        "scenario",
        "method",
        "base_key",
        "base_update",
        "target_update",
        "fixed_tau",
        "status",
        "updates_recorded",
        "last_update",
        "reward",
        "reward_vs_freeze",
        "processed",
        "drop",
        "pre_backlog",
        "sat_overlap",
        "bw_kl",
        "bw_clip",
        "bw_abs_dlogp",
        "credit_mean",
        "credit_positive_frac",
        "pos_up",
        "neg_down",
        "raw_pos_up",
        "guard_accepted",
        "guard_skipped",
        "guard_step_scale",
        "branch_product",
        "local_adv_enabled",
        "local_adv_product",
        "local_adv_pos_up",
        "local_adv_nonzero",
        "local_adv_sample_count",
        "probe_h2_product",
        "probe_h5_product",
        "probe_h10_product",
        "eval_reward",
        "eval_reward_vs_freeze",
        "eval_processed",
        "eval_drop",
        "eval_pre_backlog",
        "eval_sat_overlap",
        "eval_collision",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def print_markdown(rows: list[dict[str, object]]) -> None:
    rows_sorted = sorted(
        rows,
        key=lambda row: (
            str(row.get("scenario", "")),
            _safe_float(row.get("bw_kl")) if _safe_float(row.get("bw_kl")) is not None else 999.0,
            str(row.get("method", "")),
        ),
    )
    print("| scenario | method | reward | vs freeze | bw_kl | clip | abs_dlogp | pos_up | neg_down | branch_prod | local_pos_up | eval_reward |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows_sorted:
        reward_value = row.get("eval_reward") if row.get("eval_reward") not in (None, "") else row.get("reward")
        vs_freeze = (
            row.get("eval_reward_vs_freeze")
            if row.get("eval_reward_vs_freeze") not in (None, "")
            else row.get("reward_vs_freeze")
        )
        print(
            "| {scenario} | {method} | {reward} | {vs_freeze} | {bw_kl} | {bw_clip} | {bw_abs} | {pos_up} | {neg_down} | {branch} | {local_pos_up} | {eval_reward} |".format(
                scenario=row.get("scenario", ""),
                method=row.get("method", ""),
                reward=_fmt(reward_value),
                vs_freeze=_fmt(vs_freeze),
                bw_kl=_fmt(row.get("bw_kl"), digits=5),
                bw_clip=_fmt(row.get("bw_clip"), digits=4),
                bw_abs=_fmt(row.get("bw_abs_dlogp"), digits=4),
                pos_up=_fmt(row.get("pos_up"), digits=4),
                neg_down=_fmt(row.get("neg_down"), digits=4),
                branch=_fmt(row.get("branch_product"), digits=5),
                local_pos_up=_fmt(row.get("local_adv_pos_up"), digits=4),
                eval_reward=_fmt(row.get("eval_reward"), digits=4),
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--tail_rows", type=int, default=1)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    manifest = Path(args.manifest) if args.manifest else run_root / "matched_step_manifest.csv"
    manifest_rows = _manifest_rows(manifest) if manifest.exists() else _infer_rows_from_dirs(run_root)
    rows = [summarize_run(row, tail_rows=int(args.tail_rows)) for row in manifest_rows]
    add_freeze_relative(rows)

    if args.out_csv:
        write_csv(rows, Path(args.out_csv))
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    if args.markdown:
        print()
        print_markdown(rows)


if __name__ == "__main__":
    main()
