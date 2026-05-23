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
        Path(r"C:\Windows\Fonts\Noto Sans SC (TrueType).otf"),
        Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for path in candidates:
        if path.exists():
            fm.fontManager.addfont(str(path))
            return fm.FontProperties(fname=str(path))
    return fm.FontProperties()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    font_prop = _font_properties()
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    rows: dict[str, dict[str, str]] = {}
    with Path(args.summary).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows[str(row["method_id"])] = row

    order = [
        ("proposed", "本文方法"),
        ("cluster_center_queue_aware", "簇中心运动规则"),
        ("lyapunov", "李雅普诺夫队列方法"),
    ]
    methods = [name for method_id, name in order]
    max_queue = [float(rows[method_id]["uav_mean_queue_max_steps"]) for method_id, _ in order]
    imbalance = [float(rows[method_id]["uav_queue_imbalance_steps"]) for method_id, _ in order]
    max_queue_p90 = [float(rows[method_id]["uav_mean_queue_max_steps_p90"]) for method_id, _ in order]

    fig, ax = plt.subplots(figsize=(6.8, 3.75))
    y = np.arange(len(methods))
    bar_h = 0.22
    colors = ["#3E6FB6", "#D98634", "#6AA84F"]
    ax.barh(y - bar_h, max_queue, height=bar_h, color=colors[0], label="最大平均队列")
    ax.barh(y, imbalance, height=bar_h, color=colors[1], label="队列不均衡程度")
    ax.barh(y + bar_h, max_queue_p90, height=bar_h, color=colors[2], label="最大队列90分位")

    x_max = max(max(max_queue), max(imbalance), max(max_queue_p90))
    label_dx = x_max * 0.018
    for yi, value in zip(y - bar_h, max_queue):
        ax.text(value + label_dx, yi, f"{value:.2f}", va="center", ha="left", fontsize=8.5)
    for yi, value in zip(y, imbalance):
        ax.text(value + label_dx, yi, f"{value:.2f}", va="center", ha="left", fontsize=8.5)
    for yi, value in zip(y + bar_h, max_queue_p90):
        ax.text(value + label_dx, yi, f"{value:.2f}", va="center", ha="left", fontsize=8.5)

    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontproperties=font_prop, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("队列时间负载（时隙）", fontproperties=font_prop, fontsize=10)
    ax.set_xlim(0, x_max * 1.18)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    for text in legend.get_texts():
        text.set_fontproperties(font_prop)
        text.set_fontsize(9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(pad=0.8)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "fig4-x-uav-queue-imbalance-preview.pdf"
    png_path = out_dir / "fig4-x-uav-queue-imbalance-preview.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
