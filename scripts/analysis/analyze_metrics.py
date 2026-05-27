from __future__ import annotations

import argparse
import csv
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Run directory containing metrics.csv",
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
    args = parser.parse_args()

    metrics_path = Path(args.run_dir) / "metrics.csv"
    with metrics_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No rows found.")
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
