from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


BASE_FIELDS = [
    "job_id",
    "case_name",
    "base_key",
    "base_update",
    "target_update",
    "fixed_tau",
    "mode",
    "status",
    "updates_recorded",
    "last_update",
]

SCALAR_METRICS = {
    "reward": ("rollout_episode_reward_mean", "episode_reward_mean"),
    "processed": ("rollout_processed_ratio_eval_mean", "processed_ratio_eval"),
    "drop": ("rollout_drop_ratio_eval_mean", "drop_ratio_eval"),
    "bw_kl": ("bw_actor_full_update_kl", "bw_actor_full_kl"),
    "bw_clip": ("bw_actor_full_update_clip_frac", "bw_actor_full_clip_frac"),
    "bw_abs_dlogp": ("bw_actor_delta_logprob_abs_mean",),
    "credit_mean": ("bw_actor_credit_mean",),
    "raw_credit_mean": ("bw_actor_raw_credit_mean",),
    "pos_up": (
        "bw_actor_credit_adv_positive_delta_positive_frac",
        "bw_actor_raw_adv_positive_delta_positive_frac",
    ),
    "neg_down": ("bw_actor_credit_adv_negative_delta_negative_frac",),
    "probe_action_gap_samples": ("bw_probe_action_gap_sample_count",),
    "probe_corr_adv_action_gap": ("bw_probe_corr_advantage_vs_action_gap_k",),
    "probe_pos_adv_gap_positive_frac": ("bw_probe_pos_adv_action_gap_positive_frac",),
    "probe_neg_adv_gap_negative_frac": ("bw_probe_neg_adv_action_gap_negative_frac",),
    "probe_mean_gap_pos_adv": ("bw_probe_mean_action_gap_k_pos_adv",),
    "probe_mean_gap_neg_adv": ("bw_probe_mean_action_gap_k_neg_adv",),
    "probe_corr_adv_branch_delta": ("bw_probe_corr_advantage_vs_branch_delta",),
    "probe_corr_raw_adv_branch_delta": ("bw_probe_corr_raw_advantage_vs_branch_delta",),
    "probe_sign_agree_raw_adv_branch_delta": ("bw_probe_sign_agree_raw_advantage_branch_delta",),
    "probe_raw_pos_branch_delta_positive_frac": ("bw_probe_raw_pos_branch_delta_positive_frac",),
    "probe_branch_delta_pos_logprob_up_frac": ("bw_probe_branch_delta_pos_logprob_up_frac",),
    "probe_corr_branch_delta_dlogp": ("bw_probe_corr_branch_delta_vs_delta_logprob",),
    "probe_true_mc_samples": ("bw_probe_true_mc_sample_count",),
    "probe_corr_adv_true_mc": ("bw_probe_corr_advantage_vs_true_adv_mc",),
    "probe_corr_raw_adv_true_mc": ("bw_probe_corr_raw_advantage_vs_true_adv_mc",),
    "probe_sign_agree_adv_true_mc": ("bw_probe_sign_agree_advantage_true_adv_mc",),
    "probe_corr_true_mc_dlogp": ("bw_probe_corr_true_adv_mc_vs_delta_logprob",),
    "probe_return_target_q_error": ("bw_probe_return_target_sampled_q_error_abs_mean",),
    "probe_value_policy_q_error": ("bw_probe_value_policy_q_error_abs_mean",),
    "native_branch_samples": ("bw_probe_native_branch_sample_count",),
    "native_corr_adv_branch_delta": ("bw_probe_native_corr_advantage_vs_branch_delta",),
    "native_corr_raw_adv_branch_delta": ("bw_probe_native_corr_raw_advantage_vs_branch_delta",),
    "native_sign_agree_raw_adv_branch_delta": ("bw_probe_native_sign_agree_raw_advantage_branch_delta",),
    "native_raw_pos_branch_delta_positive_frac": ("bw_probe_native_raw_pos_branch_delta_positive_frac",),
    "native_corr_branch_delta_dlogp": ("bw_probe_native_corr_branch_delta_vs_delta_logprob",),
    "native_branch_delta_pos_logprob_up_frac": ("bw_probe_native_branch_delta_pos_logprob_up_frac",),
    "native_branch_product": ("bw_probe_native_mean_branch_delta_times_delta_logprob",),
}

NATIVE_HORIZON_METRICS = [
    "corr_advantage_vs_branch_delta",
    "corr_raw_advantage_vs_branch_delta",
    "sign_agree_raw_advantage_branch_delta",
    "raw_pos_branch_delta_positive_frac",
    "corr_branch_delta_vs_delta_logprob",
    "branch_delta_pos_logprob_up_frac",
    "mean_branch_delta_times_delta_logprob",
]


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


def _read_metrics(run_dir: Path) -> list[dict[str, str]]:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return []
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _infer_rows_from_dirs(run_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pattern = re.compile(
        r"(?P<case_name>.+_u(?P<base>\d+)to_u?(?P<target>\d+))_(?P<mode>probe_freeze|tiny_ppo|ppo_bw1e6)$"
    )
    for child in sorted(run_root.iterdir()):
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        rows.append(
            {
                "job_id": "",
                "case_name": match.group("case_name") if match else child.name,
                "base_key": "",
                "base_update": match.group("base") if match else "",
                "target_update": match.group("target") if match else "",
                "fixed_tau": "",
                "mode": match.group("mode") if match else "",
                "run_dir": str(child),
                "checkpoint": "",
                "command": "",
            }
        )
    return rows


def _latest_probe_json(run_dir: Path) -> dict[str, Any]:
    probe_dir = run_dir / "diagnostics" / "bw_advantage_alignment"
    if not probe_dir.exists():
        return {}
    candidates = sorted(probe_dir.glob("update*.json"))
    if not candidates:
        return {}
    try:
        payload = json.loads(candidates[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _horizon_metrics_from_json(payload: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    by_horizon = payload.get("native_branch_alignment_by_horizon")
    if not isinstance(by_horizon, list):
        return out
    for item in by_horizon:
        if not isinstance(item, dict):
            continue
        horizon = _safe_float(item.get("horizon"))
        summary = item.get("summary")
        if horizon is None or not isinstance(summary, dict):
            continue
        prefix = f"native_h{int(horizon)}"
        for metric in NATIVE_HORIZON_METRICS:
            value = _safe_float(summary.get(metric))
            if value is not None:
                out[f"{prefix}_{metric}"] = value
    return out


def summarize_run(manifest_row: dict[str, str]) -> dict[str, object]:
    run_dir = Path(manifest_row.get("run_dir", ""))
    metrics = _read_metrics(run_dir)
    last = metrics[-1] if metrics else {}
    payload = _latest_probe_json(run_dir)

    out: dict[str, object] = {
        "job_id": manifest_row.get("job_id", ""),
        "case_name": manifest_row.get("case_name", ""),
        "base_key": manifest_row.get("base_key", ""),
        "base_update": manifest_row.get("base_update", ""),
        "target_update": manifest_row.get("target_update", ""),
        "fixed_tau": manifest_row.get("fixed_tau", ""),
        "mode": manifest_row.get("mode", ""),
        "status": "ok" if metrics else "missing_metrics",
        "updates_recorded": len(metrics),
        "run_dir": str(run_dir),
    }
    update_value = _safe_float(last.get("update"))
    if update_value is not None:
        out["last_update"] = int(update_value)

    for out_key, names in SCALAR_METRICS.items():
        out[out_key] = _first_value(last, names)
    out.update(_horizon_metrics_from_json(payload))
    return out


def _collect_fieldnames(rows: list[dict[str, object]]) -> list[str]:
    dynamic = sorted(
        key
        for row in rows
        for key in row
        if key.startswith("native_h") and key not in BASE_FIELDS
    )
    fixed = BASE_FIELDS + list(SCALAR_METRICS.keys()) + dynamic + ["run_dir"]
    return list(dict.fromkeys(fixed))


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = _collect_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def print_markdown(rows: list[dict[str, object]]) -> None:
    rows_sorted = sorted(rows, key=lambda row: (str(row.get("case_name", "")), str(row.get("mode", ""))))
    print(
        "| case | mode | KL | pos_up | native_corr_raw_adv_h20 | native_sign_h20 | native_raw_pos_h20 | native_branch*dlogp_h20 | true_mc_corr | true_mc_sign |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows_sorted:
        print(
            "| {case} | {mode} | {kl} | {pos_up} | {corr_h20} | {sign_h20} | {raw_pos_h20} | {prod_h20} | {true_corr} | {true_sign} |".format(
                case=row.get("case_name", ""),
                mode=row.get("mode", ""),
                kl=_fmt(row.get("bw_kl"), digits=5),
                pos_up=_fmt(row.get("pos_up"), digits=4),
                corr_h20=_fmt(row.get("native_h20_corr_raw_advantage_vs_branch_delta"), digits=4),
                sign_h20=_fmt(row.get("native_h20_sign_agree_raw_advantage_branch_delta"), digits=4),
                raw_pos_h20=_fmt(row.get("native_h20_raw_pos_branch_delta_positive_frac"), digits=4),
                prod_h20=_fmt(row.get("native_h20_mean_branch_delta_times_delta_logprob"), digits=6),
                true_corr=_fmt(row.get("probe_corr_raw_adv_true_mc"), digits=4),
                true_sign=_fmt(row.get("probe_sign_agree_adv_true_mc"), digits=4),
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    manifest = Path(args.manifest) if args.manifest else run_root / "h2_probe_manifest.csv"
    manifest_rows = _manifest_rows(manifest) if manifest.exists() else _infer_rows_from_dirs(run_root)
    rows = [summarize_run(row) for row in manifest_rows]

    if args.out_csv:
        write_csv(rows, Path(args.out_csv))
    elif rows:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=_collect_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)

    if args.markdown:
        if rows:
            print()
        print_markdown(rows)


if __name__ == "__main__":
    main()
