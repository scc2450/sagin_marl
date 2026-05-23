from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


def _font_properties() -> fm.FontProperties:
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
    ]
    for path in candidates:
        if path.exists():
            fm.fontManager.addfont(str(path))
            return fm.FontProperties(fname=str(path))
    return fm.FontProperties()


def _read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {str(row["method_id"]): row for row in csv.DictReader(f)}


def _autolabel(ax, bars, fmt: str, dy: float) -> None:
    for bar in bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + dy,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8.0,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    font_prop = _font_properties()
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    rows = _read_rows(Path(args.summary))
    order = [
        ("rule_ref", "规则补齐\n参考"),
        ("learn_accel_only", "仅学习\n运动"),
        ("learn_bw_only", "仅学习\n带宽"),
        ("learn_sat_only", "仅学习\n卫星"),
        ("learn_all_heads", "完整三\n动作头"),
    ]
    labels = [name for _, name in order]
    workload = np.array([float(rows[mid]["pre_backlog_steps_eval"]) for mid, _ in order])
    processed = np.array([float(rows[mid]["processed_ratio_eval"]) for mid, _ in order])
    access = np.array([float(rows[mid]["outflow_arrival_ratio"]) for mid, _ in order])
    backhaul = np.array([float(rows[mid]["sat_incoming_arrival_ratio"]) for mid, _ in order])
    fig, axes = plt.subplots(1, 4, figsize=(10.8, 3.25))
    colors = ["#4E79A7", "#59A14F", "#F28E2B", "#76B7B2"]
    x = np.arange(len(order))

    panels = [
        (axes[0], workload, "队列时间负载", "时隙", colors[0], "{:.2f}", 4.2, 7.05),
        (axes[1], processed, "任务处理比例", "", colors[1], "{:.3f}", 0.875, 0.928),
        (axes[2], access, "归一化接入吞吐", "", colors[2], "{:.3f}", 0.88, 0.928),
        (axes[3], backhaul, "归一化回传吞吐", "", colors[3], "{:.3f}", 0.88, 0.928),
    ]
    for ax, values, title, ylabel, color, fmt, ymin, ymax_fixed in panels:
        bars = ax.bar(x, values, width=0.62, color=color)
        ax.set_title(title, fontproperties=font_prop, fontsize=10.5, pad=6)
        if ylabel:
            ax.set_ylabel(ylabel, fontproperties=font_prop, fontsize=9.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontproperties=font_prop, fontsize=8.5)
        ymax = max(float(values.max()), 1.0e-9)
        ax.set_ylim(float(ymin), float(ymax_fixed))
        ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.35)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#666666")
        ax.spines["bottom"].set_color("#666666")
        ax.tick_params(axis="y", labelsize=8.5)
        _autolabel(ax, bars, fmt, (float(ymax_fixed) - float(ymin)) * 0.025)

    fig.tight_layout(w_pad=1.0, pad=0.6)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "fig4-10-action-head-ablation-preview.pdf"
    png = out_dir / "fig4-10-action-head-ablation-preview.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
