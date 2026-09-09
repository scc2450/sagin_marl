from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def _mean(path: Path, column: str) -> float:
    df = pd.read_csv(path)
    if column not in df.columns:
        raise KeyError(f"{column} not found in {path}")
    return float(df[column].mean())


def _bar_labels(ax: plt.Axes, bars, *, fmt: str, offset: float) -> None:
    for bar in bars:
        value = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8.0,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--on_metrics", required=True)
    parser.add_argument("--off_metrics", required=True)
    parser.add_argument("--on_eval", required=True)
    parser.add_argument("--off_eval", required=True)
    parser.add_argument("--on_fairness", required=True)
    parser.add_argument("--off_fairness", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    font_prop = _font_properties()
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.size"] = 9
    mpl.rcParams["axes.labelsize"] = 9
    mpl.rcParams["xtick.labelsize"] = 8
    mpl.rcParams["ytick.labelsize"] = 8
    mpl.rcParams["legend.fontsize"] = 8

    on_train = pd.read_csv(args.on_metrics)
    off_train = pd.read_csv(args.off_metrics)
    on_eval = Path(args.on_eval)
    off_eval = Path(args.off_eval)

    colors = ["#3E6FB6", "#D98634"]

    reward = np.array([_mean(on_eval, "reward_sum"), _mean(off_eval, "reward_sum")])
    processed = np.array([
        _mean(on_eval, "processed_ratio_eval"),
        _mean(off_eval, "processed_ratio_eval"),
    ])
    dropped = np.array([
        _mean(on_eval, "drop_ratio_eval"),
        _mean(off_eval, "drop_ratio_eval"),
    ])
    backlog = np.array([
        _mean(on_eval, "pre_backlog_steps_eval"),
        _mean(off_eval, "pre_backlog_steps_eval"),
    ])
    uav_backlog = np.array([
        _mean(Path(args.on_fairness), "uav_mean_queue_avg_steps"),
        _mean(Path(args.off_fairness), "uav_mean_queue_avg_steps"),
    ])

    fig_train, ax0 = plt.subplots(figsize=(5.9, 3.35))
    ax0.plot(
        on_train["update"],
        on_train["rollout_intervention_rate"],
        color=colors[0],
        linewidth=1.25,
        label="开启安全辅助监督",
    )
    ax0.plot(
        off_train["update"],
        off_train["rollout_intervention_rate"],
        color=colors[1],
        linewidth=1.25,
        label="关闭安全辅助监督",
    )
    ax0.set_title("训练过程安全修正介入率", fontproperties=font_prop, fontsize=10.5)
    ax0.set_xlabel("训练更新次数", fontproperties=font_prop)
    ax0.set_ylabel("介入率", fontproperties=font_prop)
    ax0.set_ylim(0.0, max(0.5, float(max(on_train["rollout_intervention_rate"].max(), off_train["rollout_intervention_rate"].max())) * 1.08))
    leg = ax0.legend(loc="upper right", frameon=False)
    for text in leg.get_texts():
        text.set_fontproperties(font_prop)

    for ax in [ax0]:
        ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.35)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#666666")
        ax.spines["bottom"].set_color("#666666")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_train.tight_layout(pad=0.8)
    train_pdf = out_dir / "preview_fig4-11a-danger-imitation-training.pdf"
    train_png = out_dir / "preview_fig4-11a-danger-imitation-training.png"
    fig_train.savefig(train_pdf, bbox_inches="tight")
    fig_train.savefig(train_png, dpi=300, bbox_inches="tight")

    fig_eval, axes = plt.subplots(1, 3, figsize=(9.2, 3.2))
    ax1, ax2, ax3 = axes.ravel()
    labels = ["开启", "关闭"]
    x = np.arange(len(labels))

    bars = ax1.bar(x, reward, width=0.58, color=colors)
    ax1.set_title("平均回合奖励", fontproperties=font_prop, fontsize=10.5)
    ax1.set_ylabel("回合奖励", fontproperties=font_prop)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontproperties=font_prop)
    ax1.set_ylim(0.0, float(reward.max()) * 1.18)
    _bar_labels(ax1, bars, fmt="{:.2f}", offset=float(reward.max()) * 0.025)

    width = 0.34
    bars_p = ax2.bar(x - width / 2.0, processed, width=width, color="#59A14F", label="任务处理比例")
    bars_d = ax2.bar(x + width / 2.0, dropped, width=width, color="#E15759", label="任务丢弃率")
    ax2.set_title("任务处理与丢弃", fontproperties=font_prop, fontsize=10.5)
    ax2.set_ylabel("比例", fontproperties=font_prop)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontproperties=font_prop)
    ax2.set_ylim(0.0, 1.08)
    _bar_labels(ax2, bars_p, fmt="{:.3f}", offset=0.025)
    _bar_labels(ax2, bars_d, fmt="{:.3f}", offset=0.025)
    leg = ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, handlelength=1.4)
    for text in leg.get_texts():
        text.set_fontproperties(font_prop)

    bars_b = ax3.bar(x - width / 2.0, backlog, width=width, color="#4E79A7", label="系统队列")
    bars_u = ax3.bar(x + width / 2.0, uav_backlog, width=width, color="#76B7B2", label="无人机侧队列")
    ax3.set_title("平均队列时间负载", fontproperties=font_prop, fontsize=10.5)
    ax3.set_ylabel("时隙", fontproperties=font_prop)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontproperties=font_prop)
    queue_ymax = float(max(backlog.max(), uav_backlog.max()))
    ax3.set_ylim(0.0, queue_ymax * 1.18)
    _bar_labels(ax3, bars_b, fmt="{:.2f}", offset=queue_ymax * 0.025)
    _bar_labels(ax3, bars_u, fmt="{:.2f}", offset=queue_ymax * 0.025)
    leg = ax3.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, handlelength=1.4)
    for text in leg.get_texts():
        text.set_fontproperties(font_prop)

    for ax in axes.ravel():
        ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.35)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#666666")
        ax.spines["bottom"].set_color("#666666")

    fig_eval.tight_layout(pad=0.8, w_pad=1.2)
    eval_pdf = out_dir / "preview_fig4-11b-danger-imitation-eval.pdf"
    eval_png = out_dir / "preview_fig4-11b-danger-imitation-eval.png"
    fig_eval.savefig(eval_pdf, bbox_inches="tight")
    fig_eval.savefig(eval_png, dpi=300, bbox_inches="tight")
    print(train_pdf)
    print(train_png)
    print(eval_pdf)
    print(eval_png)


if __name__ == "__main__":
    main()
