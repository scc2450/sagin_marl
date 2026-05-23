from __future__ import annotations

import argparse
import fnmatch
import hashlib
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


Series = Tuple[np.ndarray, np.ndarray]
BandSeries = Tuple[np.ndarray, np.ndarray, np.ndarray, str]


def _split_patterns(values: Sequence[str] | None) -> List[str]:
    if not values:
        return []
    out: List[str] = []
    for item in values:
        for part in str(item).split(","):
            p = part.strip()
            if p:
                out.append(p)
    return out


def _match_any(text: str, patterns: Sequence[str]) -> bool:
    if not patterns:
        return True
    return any(fnmatch.fnmatchcase(text, pat) for pat in patterns)


def _sanitize_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = name.strip("._-")
    return name or "metric"


def _uncertainty_companion_tags(tag: str) -> List[str]:
    if tag.endswith(("_p25", "_p75", "_std")):
        return []
    return [f"{tag}_p25", f"{tag}_p75", f"{tag}_std"]


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    if values.size == 0 or alpha <= 0.0:
        return values
    out = np.empty_like(values, dtype=np.float64)
    out[0] = float(values[0])
    one_minus = 1.0 - alpha
    for idx in range(1, values.size):
        out[idx] = alpha * out[idx - 1] + one_minus * float(values[idx])
    return out


def _find_run_dirs(logdir: Path) -> List[Path]:
    event_files = sorted(logdir.rglob("events.out.tfevents.*"))
    run_dirs = sorted({p.parent for p in event_files})
    return run_dirs


def _load_scalar_series(
    run_dir: Path,
    tag_patterns: Sequence[str],
    step_min: int | None,
    step_max: int | None,
) -> Dict[str, Series]:
    ea = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={"scalars": 0},
    )
    ea.Reload()
    tags = sorted(ea.Tags().get("scalars", []))
    primary_tags = [tag for tag in tags if _match_any(tag, tag_patterns)]
    selected_tags = set(primary_tags)
    for tag in primary_tags:
        selected_tags.update(name for name in _uncertainty_companion_tags(tag) if name in tags)
    data: Dict[str, Series] = {}
    for tag in tags:
        if tag not in selected_tags:
            continue
        events = ea.Scalars(tag)
        if not events:
            continue
        steps = np.asarray([ev.step for ev in events], dtype=np.int64)
        values = np.asarray([ev.value for ev in events], dtype=np.float64)
        mask = np.ones_like(steps, dtype=bool)
        if step_min is not None:
            mask &= steps >= int(step_min)
        if step_max is not None:
            mask &= steps <= int(step_max)
        if not np.any(mask):
            continue
        data[tag] = (steps[mask], values[mask])
    return data


def _aligned_values(base_steps: np.ndarray, other: Series) -> Tuple[np.ndarray, np.ndarray] | None:
    other_steps, other_values = other
    other_map = {int(step): float(value) for step, value in zip(other_steps, other_values)}
    keep_mask = np.asarray([int(step) in other_map for step in base_steps], dtype=bool)
    if not np.any(keep_mask):
        return None
    aligned_steps = base_steps[keep_mask]
    aligned_values = np.asarray([other_map[int(step)] for step in aligned_steps], dtype=np.float64)
    return aligned_steps, aligned_values


def _resolve_uncertainty_band(tag: str, series_map: Dict[str, Series]) -> BandSeries | None:
    if tag.endswith(("_p25", "_p75", "_std")):
        return None
    center = series_map.get(tag)
    if center is None:
        return None
    center_steps, center_values = center

    p25 = series_map.get(f"{tag}_p25")
    p75 = series_map.get(f"{tag}_p75")
    if p25 is not None and p75 is not None:
        aligned_lower = _aligned_values(center_steps, p25)
        aligned_upper = _aligned_values(center_steps, p75)
        if aligned_lower is not None and aligned_upper is not None:
            lower_steps, lower_values = aligned_lower
            upper_steps, upper_values = aligned_upper
            upper_map = {int(step): float(value) for step, value in zip(upper_steps, upper_values)}
            keep_mask = np.asarray([int(step) in upper_map for step in lower_steps], dtype=bool)
            if np.any(keep_mask):
                band_steps = lower_steps[keep_mask]
                lower_band = lower_values[keep_mask]
                upper_band = np.asarray([upper_map[int(step)] for step in band_steps], dtype=np.float64)
                return band_steps, lower_band, upper_band, "IQR (p25-p75)"

    std = series_map.get(f"{tag}_std")
    if std is not None:
        aligned_std = _aligned_values(center_steps, std)
        if aligned_std is not None:
            band_steps, std_values = aligned_std
            center_map = {int(step): float(value) for step, value in zip(center_steps, center_values)}
            center_band = np.asarray([center_map[int(step)] for step in band_steps], dtype=np.float64)
            std_abs = np.abs(std_values)
            return band_steps, center_band - std_abs, center_band + std_abs, "Mean +/- std"
    return None


def _save_single_plot(
    out_path: Path,
    title: str,
    tag: str,
    series: Series,
    band: BandSeries | None,
    ema_alpha: float,
    dpi: int,
) -> None:
    steps, values = series
    values_plot = _ema(values, ema_alpha)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(steps, values_plot, linewidth=1.6, label=tag if band is not None else None)
    if band is not None:
        band_steps, lower, upper, band_label = band
        lower_plot = _ema(lower, ema_alpha)
        upper_plot = _ema(upper, ema_alpha)
        ax.fill_between(
            band_steps,
            lower_plot,
            upper_plot,
            alpha=0.18,
            linewidth=0.0,
            label=band_label,
        )
        ax.plot(
            band_steps,
            lower_plot,
            linewidth=0.9,
            linestyle="--",
            alpha=0.5,
            label="_nolegend_",
        )
        ax.plot(
            band_steps,
            upper_plot,
            linewidth=0.9,
            linestyle="--",
            alpha=0.5,
            label="_nolegend_",
        )
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(tag)
    ax.grid(True, alpha=0.25)
    if band is not None:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _save_overlay_plot(
    out_path: Path,
    tag: str,
    run_series: Sequence[Tuple[str, Series]],
    ema_alpha: float,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for run_name, series in run_series:
        steps, values = series
        values_plot = _ema(values, ema_alpha)
        ax.plot(steps, values_plot, linewidth=1.4, label=run_name)
    ax.set_title(f"{tag} (overlay)")
    ax.set_xlabel("step")
    ax.set_ylabel(tag)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _series_filename(tag: str, ext: str) -> str:
    base = _sanitize_name(tag)
    digest = hashlib.md5(tag.encode("utf-8")).hexdigest()[:8]
    return f"{base}__{digest}.{ext}"


def _resolve_outdir(logdir: Path, outdir: str | None) -> Path:
    if outdir:
        return Path(outdir)
    return logdir / "tb_plots"


def _iter_selected_runs(logdir: Path, run_patterns: Sequence[str]) -> Iterable[Tuple[str, Path]]:
    for run_dir in _find_run_dirs(logdir):
        rel = run_dir.relative_to(logdir)
        run_name = "." if str(rel) == "." else str(rel).replace("\\", "/")
        if _match_any(run_name, run_patterns):
            yield run_name, run_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch export TensorBoard scalar curves to image files."
    )
    parser.add_argument("--logdir", type=str, required=True, help="Root directory containing TensorBoard event files.")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory. Default: <logdir>/tb_plots")
    parser.add_argument("--run", dest="run_patterns", action="append", default=[], help="Run glob pattern, repeatable.")
    parser.add_argument("--tag", dest="tag_patterns", action="append", default=[], help="Scalar tag glob pattern, repeatable.")
    parser.add_argument("--step_min", type=int, default=None, help="Minimum global step (inclusive).")
    parser.add_argument("--step_max", type=int, default=None, help="Maximum global step (inclusive).")
    parser.add_argument("--ema_alpha", type=float, default=0.0, help="EMA smoothing alpha in [0,1). 0 means disabled.")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"], help="Image format.")
    parser.add_argument("--dpi", type=int, default=150, help="Image DPI.")
    parser.add_argument("--overlay", action="store_true", help="Also export overlay plots by tag across selected runs.")
    args = parser.parse_args()

    logdir = Path(args.logdir).resolve()
    if not logdir.exists():
        raise FileNotFoundError(f"logdir does not exist: {logdir}")
    outdir = _resolve_outdir(logdir, args.outdir).resolve()

    run_patterns = _split_patterns(args.run_patterns)
    tag_patterns = _split_patterns(args.tag_patterns)
    if not tag_patterns:
        tag_patterns = ["*"]

    ema_alpha = float(args.ema_alpha)
    if not (0.0 <= ema_alpha < 1.0):
        raise ValueError("--ema_alpha must be in [0, 1).")

    selected_runs = list(_iter_selected_runs(logdir, run_patterns))
    if not selected_runs:
        print("No run directories with event files matched.")
        return 0

    overlay_map: Dict[str, List[Tuple[str, Series]]] = {}
    run_plot_count = 0
    matched_runs = 0
    for run_name, run_dir in selected_runs:
        series_map = _load_scalar_series(
            run_dir=run_dir,
            tag_patterns=tag_patterns,
            step_min=args.step_min,
            step_max=args.step_max,
        )
        if not series_map:
            continue
        matched_runs += 1
        run_out = outdir / "by_run" / (_sanitize_name(run_name) if run_name != "." else "root")
        export_tags = sorted(tag for tag in series_map.keys() if _match_any(tag, tag_patterns))
        for tag in export_tags:
            series = series_map[tag]
            band = _resolve_uncertainty_band(tag, series_map)
            filename = _series_filename(tag, args.format)
            _save_single_plot(
                out_path=run_out / filename,
                title=f"{run_name} | {tag}",
                tag=tag,
                series=series,
                band=band,
                ema_alpha=ema_alpha,
                dpi=args.dpi,
            )
            run_plot_count += 1
            if args.overlay:
                overlay_map.setdefault(tag, []).append((run_name, series))

    overlay_plot_count = 0
    if args.overlay:
        overlay_out = outdir / "overlay"
        for tag, run_series in sorted(overlay_map.items()):
            filename = _series_filename(tag, args.format)
            _save_overlay_plot(
                out_path=overlay_out / filename,
                tag=tag,
                run_series=run_series,
                ema_alpha=ema_alpha,
                dpi=args.dpi,
            )
            overlay_plot_count += 1

    print(f"Runs matched: {matched_runs}/{len(selected_runs)}")
    print(f"Run-level plots: {run_plot_count}")
    if args.overlay:
        print(f"Overlay plots: {overlay_plot_count}")
    print(f"Output: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
