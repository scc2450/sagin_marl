from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rolling_mean(values: list[float], window: int) -> list[float]:
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo : i + 1]
        out.append(sum(chunk) / max(1, len(chunk)))
    return out


def _slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    xmean = sum(xs) / n
    ymean = sum(values) / n
    num = sum((x - xmean) * (y - ymean) for x, y in zip(xs, values))
    den = sum((x - xmean) ** 2 for x in xs)
    return num / den if den else 0.0


def _row_update(row: dict[str, str], fallback: int) -> int:
    value = _safe_float(row.get("update"))
    return int(value) if value is not None and math.isfinite(value) else int(fallback)


def _filter_rows_by_update(
    rows: list[dict[str, str]],
    *,
    start_update: int | None,
    end_update: int | None,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        update = _row_update(row, idx)
        if start_update is not None and update < int(start_update):
            continue
        if end_update is not None and update > int(end_update):
            continue
        out.append(row)
    return out


def _series(rows: list[dict[str, str]], name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _safe_float(row.get(name))
        if value is not None and math.isfinite(value):
            values.append(float(value))
    return values


def _load_metrics_rows(run_dir: str | Path) -> list[dict[str, str]]:
    metrics_path = Path(run_dir) / "metrics.csv"
    with metrics_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _fmt(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "missing"
    return f"{float(value):.6g}"


def _fmt_table(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.4g}"


def _selected_rows_for_compare(
    rows_all: list[dict[str, str]],
    *,
    start_update: int | None,
    end_update: int | None,
    tail_rows: int,
) -> list[dict[str, str]]:
    rows = _filter_rows_by_update(rows_all, start_update=start_update, end_update=end_update)
    if int(tail_rows) > 0:
        rows = rows[-int(tail_rows) :]
    return rows


def _mean_column(rows: list[dict[str, str]], name: str) -> float | None:
    return _mean(_series(rows, name))


def _load_eval_summary_means(run_dir: str | Path) -> dict[str, float]:
    run_path = Path(run_dir)
    summaries: list[dict[str, object]] = []
    for summary_path in sorted(run_path.glob("native_eval_seed*/*_summary.json")):
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        summary = payload.get("summary") if isinstance(payload, dict) else None
        if isinstance(summary, dict):
            summaries.append(summary)
    if not summaries:
        return {}
    keys = (
        "reward_sum",
        "processed_ratio_eval",
        "drop_ratio_eval",
        "pre_backlog_steps_eval",
        "sat_overlap_eval",
        "collision_episode_fraction",
    )
    out: dict[str, float] = {}
    for key in keys:
        values: list[float] = []
        for summary in summaries:
            value = _safe_float(summary.get(key))
            if value is not None and math.isfinite(value):
                values.append(value)
        mean_value = _mean(values)
        if mean_value is not None:
            out[key] = mean_value
    return out


def _run_kind(name: str) -> str:
    lower = str(name).lower()
    if "freeze" in lower:
        return "freeze"
    if "branchscore" in lower or "best_score" in lower or "best-score" in lower:
        return "best_score"
    if "branchbest" in lower or "best_safe" in lower or "best-safe" in lower:
        return "best_safe"
    if "branchhard" in lower or "hard" in lower:
        return "hard_branch"
    if "klguard" in lower or "kl_guard" in lower:
        return "kl_guard"
    if "bw1e6" in lower or "1e-6" in lower or "lower" in lower:
        return "lower_lr"
    if "ppo" in lower or "baseline" in lower:
        return "ppo"
    return "unknown"


def _mean_optional(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return _mean(finite)


def _print_compare_interpretation(entries: list[dict[str, object]]) -> None:
    if not entries:
        return
    print("\nMechanism interpretation hints")

    with_eval = [entry for entry in entries if entry.get("eval_reward") is not None]
    if not with_eval:
        print("- no deterministic eval summaries were found; treat this as update-direction evidence only.")

    low_kl_bad_branch: list[str] = []
    for entry in entries:
        kl = entry.get("bw_kl")
        h_product = entry.get("h_product_mean")
        if kl is None or h_product is None:
            continue
        if float(kl) <= 0.03 and float(h_product) < 0.0:
            low_kl_bad_branch.append(str(entry["name"]))
    if low_kl_bad_branch:
        print(
            "- low-KL runs still show negative branch product: "
            + ", ".join(low_kl_bad_branch)
            + ". This supports branch/local-credit mismatch over a pure KL overshoot explanation."
        )

    best_safe_entries = [entry for entry in entries if entry.get("kind") == "best_safe"]
    best_score_entries = [entry for entry in entries if entry.get("kind") == "best_score"]
    for entry in best_score_entries:
        h_product = entry.get("h_product_mean")
        branch_product = entry.get("branch_product")
        branch_signal = _mean_optional([h_product, branch_product])
        if branch_signal is None:
            continue
        if branch_signal < 0.0:
            print(
                f"- {entry['name']} still has negative branch product ({_fmt(branch_signal)}); "
                "best-score selection did not find a branch-improving safe candidate."
            )
        else:
            print(
                f"- {entry['name']} has non-negative branch product ({_fmt(branch_signal)}); "
                "branch-aware candidate selection may be enough before adding loss-level penalties."
            )
    for entry in best_safe_entries:
        fallback_product = entry.get("fallback_product")
        h_product = entry.get("h_product_mean")
        branch_signal = _mean_optional([fallback_product, h_product])
        if branch_signal is None:
            continue
        if branch_signal < 0.0:
            print(
                f"- {entry['name']} keeps branch product negative ({_fmt(branch_signal)}); "
                "interpret best-safe as damage minimization, not a direction fix."
            )
        else:
            print(
                f"- {entry['name']} has non-negative branch product ({_fmt(branch_signal)}); "
                "this is evidence that branch-aware candidate selection can find a safer direction."
            )

    kl_entries = [entry for entry in entries if entry.get("kind") == "kl_guard"]
    lower_entries = [entry for entry in entries if entry.get("kind") == "lower_lr"]
    ppo_entries = [entry for entry in entries if entry.get("kind") == "ppo"]
    reference_entries = kl_entries or lower_entries or ppo_entries
    for best_entry in best_safe_entries:
        best_product = _mean_optional([best_entry.get("fallback_product"), best_entry.get("h_product_mean")])
        if best_product is None:
            continue
        ref_products = [
            _mean_optional([entry.get("fallback_product"), entry.get("h_product_mean")])
            for entry in reference_entries
        ]
        ref_products = [float(value) for value in ref_products if value is not None and math.isfinite(float(value))]
        if not ref_products:
            continue
        ref_mean = _mean(ref_products)
        if ref_mean is not None and best_product > ref_mean:
            print(
                f"- {best_entry['name']} has a better branch product than KL/lr/PPO references "
                f"({_fmt(best_product)} vs {_fmt(ref_mean)}); branch-aware selection is carrying information."
            )
        elif ref_mean is not None:
            print(
                f"- {best_entry['name']} does not improve branch product over KL/lr/PPO references "
                f"({_fmt(best_product)} vs {_fmt(ref_mean)}); prioritize counterfactual advantage or BW shape diagnostics."
            )

    if with_eval:
        best_eval = max(with_eval, key=lambda entry: float(entry.get("eval_reward") or -float("inf")))
        print(
            f"- best eval reward among compared runs: {best_eval['name']} "
            f"reward={_fmt(best_eval.get('eval_reward'))}, "
            f"processed={_fmt(best_eval.get('processed'))}, "
            f"drop={_fmt(best_eval.get('drop'))}, backlog={_fmt(best_eval.get('backlog'))}."
        )
        for entry in with_eval:
            processed = entry.get("processed")
            drop = entry.get("drop")
            backlog = entry.get("backlog")
            h_product = entry.get("h_product_mean")
            if h_product is not None and float(h_product) >= 0.0 and drop is not None and float(drop) > 0.05:
                print(
                    f"- {entry['name']} has non-negative branch product but high drop ({_fmt(drop)}); "
                    "check whether short-horizon branch replay is optimizing the wrong reward component."
                )
            if processed is not None and backlog is not None and float(processed) > 0.9 and float(backlog) > 20.0:
                print(
                    f"- {entry['name']} has high processed ratio but large backlog ({_fmt(backlog)}); "
                    "do not call it fixed without reward-decomposition consistency."
                )


def _print_compare_run_dirs(
    run_dirs: list[str],
    *,
    stage: str,
    start_update: int | None,
    end_update: int | None,
    tail_rows: int,
) -> None:
    prefix = f"{stage}_actor_"
    columns = [
        "run",
        "updates",
        "rows",
        "BW KL",
        "clip",
        "credit",
        "raw credit",
        "raw+ dlogp",
        "raw+ up",
        "guard",
        "step",
        "fallback",
        "branch prod",
        "h2 prod",
        "h5 prod",
        "h10 prod",
        "eval reward",
        "processed",
        "drop",
        "backlog",
    ]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    entries: list[dict[str, object]] = []
    for run_dir in run_dirs:
        run_path = Path(run_dir)
        try:
            rows_all = _load_metrics_rows(run_path)
        except OSError as exc:
            print(
                "| "
                + " | ".join(
                    [
                        run_path.name,
                        "missing metrics.csv",
                        "",
                        str(exc).replace("|", "/"),
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
                + " |"
            )
            continue
        rows = _selected_rows_for_compare(
            rows_all,
            start_update=start_update,
            end_update=end_update,
            tail_rows=tail_rows,
        )
        if not rows:
            print(
                "| "
                + " | ".join(
                    [
                        run_path.name,
                        "no selected rows",
                        "0",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
                + " |"
            )
            continue
        updates = [_row_update(row, idx) for idx, row in enumerate(rows, start=1)]
        eval_means = _load_eval_summary_means(run_path)
        h2_product = _mean_column(rows, "bw_probe_native_h2_mean_branch_delta_times_delta_logprob")
        h5_product = _mean_column(rows, "bw_probe_native_h5_mean_branch_delta_times_delta_logprob")
        h10_product = _mean_column(rows, "bw_probe_native_h10_mean_branch_delta_times_delta_logprob")
        guard_accepted = _mean_column(rows, f"{prefix}guard_accepted")
        guard_skipped = _mean_column(rows, f"{prefix}guard_skipped")
        guard_accept_mode = _mean_column(rows, f"{prefix}guard_branch_accept_mode_code")
        guard_text = ""
        if guard_accepted is not None or guard_skipped is not None:
            guard_text = f"a={_fmt_table(guard_accepted)} s={_fmt_table(guard_skipped)}"
            if guard_accept_mode is not None and guard_accept_mode > 0.5:
                guard_text += " best"
        row_values = [
            run_path.name,
            f"{min(updates)}..{max(updates)}",
            str(len(rows)),
            _fmt_table(_mean_column(rows, f"{prefix}full_update_kl")),
            _fmt_table(_mean_column(rows, f"{prefix}full_update_clip_frac")),
            _fmt_table(_mean_column(rows, f"{prefix}credit_mean")),
            _fmt_table(_mean_column(rows, f"{prefix}raw_credit_mean")),
            _fmt_table(_mean_column(rows, f"{prefix}delta_logprob_when_raw_adv_positive_mean")),
            _fmt_table(_mean_column(rows, f"{prefix}raw_adv_positive_delta_positive_frac")),
            guard_text,
            _fmt_table(_mean_column(rows, f"{prefix}guard_step_scale")),
            _fmt_table(_mean_column(rows, f"{prefix}guard_branch_fallback_product")),
            _fmt_table(_mean_column(rows, f"{prefix}guard_branch_mean_delta_times_delta_logprob_mean")),
            _fmt_table(h2_product),
            _fmt_table(h5_product),
            _fmt_table(h10_product),
            _fmt_table(eval_means.get("reward_sum")),
            _fmt_table(eval_means.get("processed_ratio_eval")),
            _fmt_table(eval_means.get("drop_ratio_eval")),
            _fmt_table(eval_means.get("pre_backlog_steps_eval")),
        ]
        print("| " + " | ".join(str(value).replace("|", "/") for value in row_values) + " |")
        entries.append(
            {
                "name": run_path.name,
                "kind": _run_kind(run_path.name),
                "bw_kl": _mean_column(rows, f"{prefix}full_update_kl"),
                "branch_product": _mean_column(rows, f"{prefix}guard_branch_mean_delta_times_delta_logprob_mean"),
                "fallback_product": _mean_column(rows, f"{prefix}guard_branch_fallback_product"),
                "h_product_mean": _mean_optional([h2_product, h5_product, h10_product]),
                "eval_reward": eval_means.get("reward_sum"),
                "processed": eval_means.get("processed_ratio_eval"),
                "drop": eval_means.get("drop_ratio_eval"),
                "backlog": eval_means.get("pre_backlog_steps_eval"),
            }
        )
    _print_compare_interpretation(entries)


def _print_metric_row(label: str, values: list[float]) -> None:
    if not values:
        print(f"{label:62s} missing")
        return
    print(
        f"{label:62s} "
        f"mean={_fmt(_mean(values)):>10s} "
        f"first={_fmt(values[0]):>10s} "
        f"last={_fmt(values[-1]):>10s} "
        f"min={_fmt(min(values)):>10s} "
        f"max={_fmt(max(values)):>10s}"
    )


def _print_phase2_bw_summary(
    rows_all: list[dict[str, str]],
    *,
    stage: str,
    start_update: int | None,
    end_update: int | None,
) -> None:
    rows = _filter_rows_by_update(rows_all, start_update=start_update, end_update=end_update)
    if not rows:
        print("No rows selected for Phase 2 BW summary.")
        return
    updates = [_row_update(row, idx) for idx, row in enumerate(rows, start=1)]
    prefix = f"{stage}_actor_"
    print("\nPhase 2 BW/raw-vs-norm diagnostic")
    print(f"- stage: {stage}")
    print(f"- rows: {len(rows)}")
    print(f"- update window: {min(updates)}..{max(updates)}")

    groups: list[tuple[str, list[str]]] = [
        (
            "PPO update scale / old normalized-adv credit",
            [
                "full_update_kl",
                "full_update_clip_frac",
                "delta_logprob_abs_mean",
                "credit_mean",
                "credit_direction_agree_frac",
                "credit_adv_positive_delta_positive_frac",
                "credit_adv_negative_delta_negative_frac",
                "delta_logprob_when_adv_positive_mean",
                "delta_logprob_when_adv_negative_mean",
                "old_logprob_mean",
            ],
        ),
        (
            "Raw-vs-norm advantage alignment",
            [
                "raw_adv_available",
                "raw_adv_positive_frac",
                "raw_adv_negative_frac",
                "norm_raw_adv_sign_agree_frac",
                "norm_raw_adv_sign_disagree_frac",
                "norm_adv_positive_raw_adv_negative_within_norm_positive_frac",
                "norm_adv_negative_raw_adv_positive_within_norm_negative_frac",
            ],
        ),
        (
            "Raw-advantage update direction",
            [
                "raw_adv_positive_delta_positive_frac",
                "raw_adv_positive_delta_negative_frac",
                "raw_adv_negative_delta_negative_frac",
                "raw_adv_negative_delta_positive_frac",
                "delta_logprob_when_raw_adv_positive_mean",
                "delta_logprob_when_raw_adv_negative_mean",
                "raw_credit_mean",
                "raw_credit_direction_agree_frac",
                "raw_credit_direction_wrong_frac",
                "raw_credit_when_raw_adv_positive_mean",
                "raw_credit_when_raw_adv_negative_mean",
            ],
        ),
        (
            "Guarded BW candidate / branch-alignment acceptance",
            [
                "guard_enabled",
                "guard_accepted",
                "guard_skipped",
                "guard_attempts",
                "guard_step_scale",
                "guard_reject_code",
                "guard_candidate_reject_code",
                "guard_branch_enabled",
                "guard_branch_available",
                "guard_branch_horizon_count",
                "guard_branch_accept_mode_code",
                "guard_branch_selected_attempt",
                "guard_branch_selected_step_scale",
                "guard_branch_mean_delta_times_delta_logprob_mean",
                "guard_branch_mean_delta_times_delta_logprob_min",
                "guard_branch_positive_logprob_up_frac_mean",
                "guard_branch_positive_logprob_up_frac_min",
                "guard_branch_corr_delta_logprob_mean",
                "guard_branch_corr_delta_logprob_min",
                "guard_branch_fallback_mode_code",
                "guard_branch_fallback_accepted",
                "guard_branch_fallback_reject_code",
                "guard_branch_fallback_attempt",
                "guard_branch_fallback_step_scale",
                "guard_branch_fallback_product",
            ],
        ),
        (
            "Guarded BW rejected-candidate last attempt",
            [
                "guard_last_full_update_kl",
                "guard_last_full_update_clip_frac",
                "guard_last_credit_mean",
                "guard_last_credit_when_adv_positive_mean",
                "guard_last_delta_logprob_when_adv_positive_mean",
                "guard_last_raw_credit_mean",
                "guard_last_raw_credit_when_raw_adv_positive_mean",
                "guard_last_delta_logprob_when_raw_adv_positive_mean",
                "guard_last_raw_adv_positive_delta_positive_frac",
                "guard_last_guard_branch_available",
                "guard_last_guard_branch_mean_delta_times_delta_logprob_mean",
                "guard_last_guard_branch_mean_delta_times_delta_logprob_min",
                "guard_last_guard_branch_positive_logprob_up_frac_mean",
                "guard_last_guard_branch_positive_logprob_up_frac_min",
                "guard_last_guard_branch_corr_delta_logprob_mean",
                "guard_last_guard_branch_corr_delta_logprob_min",
                "guard_best_attempt",
                "guard_best_step_scale",
                "guard_best_reject_code",
                "guard_best_guard_branch_mean_delta_times_delta_logprob_mean",
                "guard_best_guard_branch_positive_logprob_up_frac_min",
                "guard_best_guard_branch_corr_delta_logprob_min",
            ],
        ),
    ]

    metric_cache: dict[str, list[float]] = {}
    for title, metric_names in groups:
        print(f"\n{title}")
        for metric in metric_names:
            column = f"{prefix}{metric}"
            values = _series(rows, column)
            metric_cache[metric] = values
            _print_metric_row(column, values)

    probe_groups: list[tuple[str, list[str]]] = [
        (
            "BW advantage-alignment probe / logprob movement",
            [
                "bw_probe_sample_count",
                "bw_probe_judgement_code",
                "bw_probe_old_logprob_replay_abs_diff_mean",
                "bw_probe_corr_advantage_vs_delta_logprob",
                "bw_probe_pos_adv_logprob_up_frac",
                "bw_probe_mean_delta_logprob_pos_adv",
                "bw_probe_mean_delta_logprob_neg_adv",
            ],
        ),
        (
            "Native BW branch sample-vs-ref alignment",
            [
                "bw_probe_native_branch_sample_count",
                "bw_probe_native_branch_horizon",
                "bw_probe_native_corr_advantage_vs_branch_delta",
                "bw_probe_native_corr_raw_advantage_vs_branch_delta",
                "bw_probe_native_sign_agree_raw_advantage_branch_delta",
                "bw_probe_native_raw_pos_branch_delta_positive_frac",
                "bw_probe_native_corr_branch_delta_vs_delta_logprob",
                "bw_probe_native_branch_delta_pos_logprob_up_frac",
                "bw_probe_native_branch_delta_abs_mean",
                "bw_probe_native_mean_abs_delta_logprob",
            ],
        ),
    ]
    probe_cache: dict[str, list[float]] = {}
    for title, metric_names in probe_groups:
        any_values = any(_series(rows, metric) for metric in metric_names)
        if not any_values:
            continue
        print(f"\n{title}")
        for metric in metric_names:
            values = _series(rows, metric)
            probe_cache[metric] = values
            _print_metric_row(metric, values)
    horizon_ids: list[int] = []
    seen_horizons: set[int] = set()
    for row in rows:
        for key in row:
            match = re.match(r"bw_probe_native_h(\d+)_branch_sample_count$", str(key))
            if not match:
                continue
            horizon = int(match.group(1))
            values = _series(rows, str(key))
            if horizon not in seen_horizons and values:
                seen_horizons.add(horizon)
                horizon_ids.append(horizon)
    for horizon in sorted(horizon_ids):
        print(f"\nNative BW branch horizon h{horizon}")
        for suffix in (
            "branch_sample_count",
            "corr_raw_advantage_vs_branch_delta",
            "sign_agree_raw_advantage_branch_delta",
            "raw_pos_branch_delta_positive_frac",
            "corr_branch_delta_vs_delta_logprob",
            "branch_delta_pos_logprob_up_frac",
            "branch_delta_abs_mean",
            "mean_branch_delta_times_delta_logprob",
            "mean_abs_delta_logprob",
        ):
            metric = f"bw_probe_native_h{int(horizon)}_{suffix}"
            values = _series(rows, metric)
            probe_cache[metric] = values
            _print_metric_row(metric, values)

    raw_available = _mean(metric_cache.get("raw_adv_available", []))
    norm_raw_agree = _mean(metric_cache.get("norm_raw_adv_sign_agree_frac", []))
    norm_pos_raw_neg = _mean(
        metric_cache.get("norm_adv_positive_raw_adv_negative_within_norm_positive_frac", [])
    )
    norm_neg_raw_pos = _mean(
        metric_cache.get("norm_adv_negative_raw_adv_positive_within_norm_negative_frac", [])
    )
    raw_pos_frac = _mean(metric_cache.get("raw_adv_positive_frac", []))
    raw_neg_frac = _mean(metric_cache.get("raw_adv_negative_frac", []))
    raw_pos_up = _mean(metric_cache.get("raw_adv_positive_delta_positive_frac", []))
    raw_pos_dlogp = _mean(metric_cache.get("delta_logprob_when_raw_adv_positive_mean", []))
    norm_pos_up = _mean(metric_cache.get("credit_adv_positive_delta_positive_frac", []))
    norm_pos_dlogp = _mean(metric_cache.get("delta_logprob_when_adv_positive_mean", []))
    guard_enabled = _mean(metric_cache.get("guard_enabled", []))
    guard_branch_enabled = _mean(metric_cache.get("guard_branch_enabled", []))
    guard_branch_available = _mean(metric_cache.get("guard_branch_available", []))
    guard_branch_product = _mean(metric_cache.get("guard_branch_mean_delta_times_delta_logprob_mean", []))
    guard_branch_pos_frac = _mean(metric_cache.get("guard_branch_positive_logprob_up_frac_min", []))
    guard_branch_corr = _mean(metric_cache.get("guard_branch_corr_delta_logprob_min", []))
    guard_last_branch_available = _mean(metric_cache.get("guard_last_guard_branch_available", []))
    guard_last_branch_product = _mean(
        metric_cache.get("guard_last_guard_branch_mean_delta_times_delta_logprob_mean", [])
    )
    guard_last_branch_pos_frac = _mean(metric_cache.get("guard_last_guard_branch_positive_logprob_up_frac_min", []))
    guard_last_branch_corr = _mean(metric_cache.get("guard_last_guard_branch_corr_delta_logprob_min", []))
    guard_fallback_accepted = _mean(metric_cache.get("guard_branch_fallback_accepted", []))
    guard_fallback_reject_code = _mean(metric_cache.get("guard_branch_fallback_reject_code", []))
    guard_fallback_product = _mean(metric_cache.get("guard_branch_fallback_product", []))
    guard_fallback_step = _mean(metric_cache.get("guard_branch_fallback_step_scale", []))

    print("\nInterpretation flags")
    if raw_available is None or raw_available < 0.5:
        print("- raw advantage diagnostics are missing in this metrics.csv; rerun with the 2026-06-06 instrumentation.")
    raw_sign_saturated = (
        (raw_pos_frac is not None and raw_pos_frac > 0.9)
        or (raw_neg_frac is not None and raw_neg_frac > 0.9)
    )
    if raw_sign_saturated:
        print(
            "- raw advantage sign is saturated; raw sign is a weak good/bad classifier here. "
            "Check critic/value calibration and true/branch advantage next."
        )
    if norm_raw_agree is not None and norm_raw_agree < 0.8:
        print("- norm/raw advantage signs disagree often; H1 advantage normalization/baseline mismatch should be checked first.")
    if norm_neg_raw_pos is not None and norm_neg_raw_pos > 0.5:
        print(
            "- many normalized-negative samples are still raw-positive; centering is turning below-average positive returns "
            "into PPO-negative samples."
        )
    if norm_pos_raw_neg is not None and norm_pos_raw_neg > 0.2:
        print("- many normalized-positive samples are raw-negative; do not interpret A_norm>0 as an absolute good action.")
    if raw_pos_up is not None and raw_pos_up < 0.5:
        if raw_sign_saturated:
            print(
                "- fewer than half of raw-positive samples get higher logprob, but raw-positive covers most samples; "
                "treat this as evidence of broad density movement, not as a clean good-action failure by itself."
            )
        else:
            print("- fewer than half of raw-positive samples get higher logprob; this is not just a normalized-advantage artifact.")
    if raw_pos_dlogp is not None and raw_pos_dlogp <= 0.0:
        if raw_sign_saturated:
            print("- mean delta_logprob for raw-positive samples is non-positive under saturated raw signs; prioritize H2 calibration probes.")
        else:
            print("- mean delta_logprob for raw-positive samples is non-positive; continue to H2/H3/H4 probes.")
    if (
        norm_raw_agree is not None
        and norm_raw_agree >= 0.8
        and raw_pos_up is not None
        and raw_pos_up < 0.5
    ):
        print("- norm/raw signs mostly agree, but raw-positive samples still are not reinforced; prioritize BW shape/step-scale diagnostics.")
    if (
        norm_pos_up is not None
        and raw_pos_up is not None
        and norm_pos_dlogp is not None
        and raw_pos_dlogp is not None
    ):
        print(
            "- normalized-positive vs raw-positive comparison: "
            f"pos_up {norm_pos_up:.3f} vs {raw_pos_up:.3f}, "
            f"mean_dlogp {norm_pos_dlogp:.3g} vs {raw_pos_dlogp:.3g}."
        )
    native_branch_samples = _mean(probe_cache.get("bw_probe_native_branch_sample_count", []))
    native_raw_branch_agree = _mean(probe_cache.get("bw_probe_native_sign_agree_raw_advantage_branch_delta", []))
    native_raw_pos_branch_pos = _mean(probe_cache.get("bw_probe_native_raw_pos_branch_delta_positive_frac", []))
    native_branch_dlogp_corr = _mean(probe_cache.get("bw_probe_native_corr_branch_delta_vs_delta_logprob", []))
    if native_branch_samples is not None and native_branch_samples > 0:
        if native_raw_branch_agree is not None and native_raw_branch_agree < 0.5:
            print(
                "- raw advantage sign poorly matches native branch_delta; H2 critic/target calibration is plausible."
            )
        if native_raw_pos_branch_pos is not None and native_raw_pos_branch_pos < 0.5:
            print(
                "- fewer than half of raw-positive BW samples beat the deterministic reference in native branch replay."
            )
        if native_branch_dlogp_corr is not None and native_branch_dlogp_corr <= 0.0:
            print(
                "- branch_delta is not positively correlated with post-update delta_logprob; PPO movement may be "
                "decoupled from BW local counterfactual improvement."
            )
    if (
        (guard_branch_enabled is not None and guard_branch_enabled > 0.5)
        or (guard_enabled is not None and guard_enabled > 0.5 and guard_last_branch_available is not None)
    ):
        if guard_branch_available is None or guard_branch_available < 0.5:
            if guard_last_branch_available is not None and guard_last_branch_available > 0.5:
                print(
                    "- guarded branch-alignment rejected all candidates; last rejected attempt: "
                    f"mean(branch_delta*dlogp)={_fmt(guard_last_branch_product)}, "
                    f"min branch-positive-up={_fmt(guard_last_branch_pos_frac)}, "
                    f"min corr(branch_delta,dlogp)={_fmt(guard_last_branch_corr)}."
                )
                if guard_last_branch_product is not None and guard_last_branch_product < 0.0:
                    print("- last rejected candidate moved against branch_delta on average, matching the H2/H3 failure mode.")
            else:
                print("- guarded branch-alignment was enabled, but no native branch replay samples were available.")
        else:
            print(
                "- guarded branch-alignment summary: "
                f"mean(branch_delta*dlogp)={_fmt(guard_branch_product)}, "
                f"min branch-positive-up={_fmt(guard_branch_pos_frac)}, "
                f"min corr(branch_delta,dlogp)={_fmt(guard_branch_corr)}."
            )
            if guard_branch_product is not None and guard_branch_product < 0.0:
                print("- guarded candidates still move against branch_delta on average; reject/backtrack thresholds may be active.")
        if guard_fallback_accepted is not None and guard_fallback_accepted > 0.5:
            print(
                "- branch fallback accepted a best-safe candidate: "
                f"step={_fmt(guard_fallback_step)}, product={_fmt(guard_fallback_product)}, "
                f"candidate_reject_code={_fmt(guard_fallback_reject_code)}. "
                "Treat this as adaptive damage minimization, not proof of branch improvement."
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Run directory containing metrics.csv",
    )
    parser.add_argument(
        "--compare_run_dirs",
        nargs="+",
        default=None,
        help=(
            "Print a compact markdown comparison table for multiple run directories. "
            "Each directory should contain metrics.csv; native_eval_seed* summaries are included when present."
        ),
    )
    parser.add_argument(
        "--compare_tail_rows",
        type=int,
        default=0,
        help="When comparing runs, summarize only the last N selected metrics rows; 0 uses all selected rows.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=(
            "episode_reward,reward_raw,policy_loss,value_loss,entropy,"
            "reward_rms_sigma,reward_clip_frac,"
            "approx_kl,clip_frac,adv_raw_std,adv_norm_std,"
            "r_term_centroid,r_term_accel,"
            "gu_queue_mean,queue_total_active,drop_sum,gu_drop_sum"
        ),
        help="Comma-separated metric names to analyze",
    )
    parser.add_argument("--window", type=int, default=20, help="Rolling mean window size")
    parser.add_argument(
        "--phase2_bw",
        action="store_true",
        help="Print the Phase 2 BW raw-vs-norm advantage diagnostic summary.",
    )
    parser.add_argument("--stage", choices=["accel", "sat", "bw"], default="bw")
    parser.add_argument("--start_update", type=int, default=None)
    parser.add_argument("--end_update", type=int, default=None)
    args = parser.parse_args()

    if args.compare_run_dirs:
        _print_compare_run_dirs(
            [str(path) for path in args.compare_run_dirs],
            stage=str(args.stage),
            start_update=args.start_update,
            end_update=args.end_update,
            tail_rows=max(int(args.compare_tail_rows), 0),
        )
        return

    if args.run_dir is None:
        parser.error("--run_dir is required unless --compare_run_dirs is provided.")

    rows = _load_metrics_rows(args.run_dir)

    if not rows:
        print("No rows found.")
        return

    if bool(args.phase2_bw):
        _print_phase2_bw_summary(
            rows,
            stage=str(args.stage),
            start_update=args.start_update,
            end_update=args.end_update,
        )
        return

    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
    print(f"rows {len(rows)}")
    available = set(rows[0].keys())
    for name in metric_names:
        if name not in available:
            print(f"\n{name}")
            print("- missing in metrics.csv (skipped)")
            continue
        series: list[float] = []
        invalid = 0
        for r in rows:
            val = _safe_float(r.get(name))
            if val is None:
                invalid += 1
                continue
            series.append(val)
        if not series:
            print(f"\n{name}")
            print("- no valid numeric values (skipped)")
            continue
        roll = _rolling_mean(series, max(1, args.window))
        start = roll[0]
        end = roll[-1]
        slope = _slope(roll)
        nonzero = sum(1 for x in series if abs(x) > 1e-12)
        is_const = (max(series) - min(series)) <= 1e-12
        print(f"\n{name}")
        print(f"- rolling mean start: {start:.6g}  end: {end:.6g}  slope: {slope:.6g}")
        print(f"- min: {min(series):.6g}  max: {max(series):.6g}")
        print(f"- nonzero: {nonzero}/{len(series)}  constant: {is_const}")
        if invalid > 0:
            print(f"- ignored non-numeric rows: {invalid}")


if __name__ == "__main__":
    main()
