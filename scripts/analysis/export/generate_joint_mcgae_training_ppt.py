# -*- coding: utf-8 -*-
"""Generate the short Chinese PPT for the joint MC-GAE training summary."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


OUT = Path("docs/assets/slides/joint_mcgae_training_flow_summary_20260514.pptx")
ASSET_DIR = Path("docs/assets/ppt_formula_assets")
ASSET_DIR.mkdir(parents=True, exist_ok=True)

FONT_CN = "Microsoft YaHei"
BLACK = RGBColor(18, 18, 18)
GRAY = RGBColor(85, 85, 85)
BLUE = RGBColor(22, 88, 160)
LIGHT = RGBColor(244, 247, 250)
BORDER = RGBColor(210, 220, 230)


def set_run(run, size=20, bold=False, color=BLACK):
    run.font.name = FONT_CN
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color


def add_textbox(slide, x, y, w, h, text, size=20, bold=False, color=BLACK, align=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    p = tf.paragraphs[0]
    if align is not None:
        p.alignment = align
    r = p.add_run()
    r.text = text
    set_run(r, size=size, bold=bold, color=color)
    return box


def add_title(slide, title):
    add_textbox(slide, 0.65, 0.35, 12.0, 0.55, title, size=28, bold=True)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.65), Inches(1.22), Inches(12.0), Inches(0.03))
    line.fill.solid()
    line.fill.fore_color.rgb = BLUE
    line.line.fill.background()


def add_bullets(slide, x, y, w, h, items, size=19):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ""
        p.space_after = Pt(7)
        r = p.add_run()
        r.text = f"{i + 1}. {item}"
        set_run(r, size=size)
    return box


def formula_png(name: str, expr: str, fontsize=24) -> Path:
    path = ASSET_DIR / f"{name}.png"
    fig = plt.figure(figsize=(9.5, 0.8), dpi=220)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.5, 0.5, expr, ha="center", va="center", fontsize=fontsize, color="black")
    fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def add_formula(slide, x, y, w, h, img_path: Path):
    bg = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    bg.fill.solid()
    bg.fill.fore_color.rgb = LIGHT
    bg.line.color.rgb = BORDER
    slide.shapes.add_picture(str(img_path), Inches(x + 0.22), Inches(y + 0.12), width=Inches(w - 0.44), height=Inches(h - 0.24))


def add_table(slide, x, y, headers, rows, col_widths, font_size=16):
    table = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(x), Inches(y), Inches(sum(col_widths)), Inches(0.42 * (len(rows) + 1))).table
    for j, width in enumerate(col_widths):
        table.columns[j].width = Inches(width)
    for j, header in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = header
        cell.fill.solid()
        cell.fill.fore_color.rgb = LIGHT
        for paragraph in cell.text_frame.paragraphs:
            paragraph.alignment = PP_ALIGN.CENTER
            for run in paragraph.runs:
                set_run(run, size=font_size, bold=True)
    for i, row in enumerate(rows, start=1):
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            cell.text = str(value)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(255, 255, 255)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.alignment = PP_ALIGN.CENTER
                for run in paragraph.runs:
                    set_run(run, size=font_size)
    return table


def main() -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    f_gae = formula_png(
        "gae_advantage",
        r"$A_t^{\mathrm{GAE}}=\sum_l(\gamma\lambda)^l\,[r_t+\gamma V(s_{t+1})-V(s_t)]$",
        24,
    )
    f_mc = formula_png(
        "mc_return",
        r"$R_t^{\mathrm{MC}}=\sum_k \gamma^k r_{t+k}\qquad \mathbb{E}[R_t^{\mathrm{MC}}\mid s_t]=V_\pi(s_t)$",
        24,
    )
    f_macro = formula_png(
        "bw_macro",
        r"$\mathrm{BW\ action}\ \rightarrow\ \mathrm{reuse\ for\ 5\ primitive\ steps}$",
        24,
    )
    f_gain = formula_png(
        "stage_gain",
        r"$A{+}B-B=+0.97\qquad S{+}B-B=+1.11\qquad A{+}S{+}B-B=+1.61$",
        23,
    )

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "1. 原 PPO 训练信号不稳定")
    add_formula(slide, 1.0, 1.50, 11.3, 0.85, f_gae)
    add_bullets(
        slide,
        0.95,
        2.55,
        5.9,
        1.65,
        [
            "GAE 目标依赖价值函数自举。",
            "早期价值函数很差时，critic 学完整时域价值很慢。",
            "actor 的优势估计不能稳定反映当前阶段动作好坏。",
        ],
        size=19,
    )
    add_table(
        slide,
        7.05,
        2.25,
        ["审计项", "数值"],
        [
            ["GAE 目标均值", "约 -0.146"],
            ["完整时域价值均值", "约 -2.012"],
            ["GAE 反复拟合后 EV", "约 0.003"],
            ["MC 目标拟合 EV", "约 0.997"],
        ],
        [3.0, 2.2],
        font_size=15,
    )
    add_textbox(slide, 0.95, 5.55, 11.4, 0.65, "结论：critic 不是完全学不了价值函数；主要问题是 GAE 自举目标早期尺度不对、传播太慢。", size=20, bold=True, color=BLUE)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "2. 改成 MC-GAE critic 训练")
    add_textbox(slide, 0.9, 1.5, 5.2, 0.35, "旧流程", size=21, bold=True, color=BLUE)
    add_bullets(slide, 0.95, 1.95, 5.2, 1.55, ["采样一批轨迹", "用当前 critic 计算 GAE", "用 GAE 回报训练 critic", "用 GAE 优势更新 actor"], size=18)
    add_textbox(slide, 6.8, 1.5, 5.2, 0.35, "新流程", size=21, bold=True, color=BLUE)
    add_bullets(slide, 6.85, 1.95, 5.8, 1.75, ["采样 64 个环境 × 250 步", "用有限时域 MC 回报训练 critic", "critic 训好后重新计算 GAE(V)", "actor 仍用 PPO 裁剪目标更新"], size=18)
    add_formula(slide, 1.2, 4.2, 10.9, 0.9, f_mc)
    add_textbox(slide, 1.0, 5.65, 11.4, 0.8, "MC 回报提供正确的价值尺度；重新计算的 GAE(V) 为 actor 提供更低方差的优势估计。", size=21, bold=True, color=BLUE, align=PP_ALIGN.CENTER)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "3. 先保证 critic 可用，再更新 actor")
    add_table(
        slide,
        0.9,
        1.55,
        ["阶段", "设置"],
        [
            ["第一轮冷启动拟合", "学习率 1e-3，基础 20 轮，EV 不够则补训"],
            ["后续跟踪拟合", "学习率 3e-4，基础 5 轮，EV 不够则补训"],
        ],
        [2.7, 8.8],
        font_size=15,
    )
    add_table(
        slide,
        0.9,
        2.95,
        ["阶段", "第 1 轮 EV", "第 300 轮 EV"],
        [["加速度", "0.934", "0.963"], ["选星", "0.922", "0.971"], ["带宽", "0.939", "0.977"]],
        [2.5, 2.8, 2.8],
        font_size=16,
    )
    add_bullets(
        slide,
        0.95,
        5.15,
        11.2,
        1.05,
        [
            "KL 早停：单轮 actor 更新内 KL 过大就提前停止。",
            "学习率衰减：KL 或裁剪比例持续过大时降低 actor 学习率。",
            "学习率增长：当前关闭，避免后期学习率涨得过激。",
        ],
        size=18,
    )
    add_textbox(slide, 0.95, 6.45, 11.3, 0.38, "第 1 轮需要多训 critic；后续策略变化较小后，较少训练轮数基本能跟上。", size=19, bold=True, color=BLUE)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "4. BW macro K=5：让带宽动作影响持续更久")
    add_bullets(
        slide,
        0.95,
        1.55,
        11.3,
        1.05,
        [
            "只改 MC-GAE critic 后，收益仍更像主要来自加速度阶段。",
            "进一步加入带宽宏决策：带宽决策间隔 = 5。",
        ],
        size=20,
    )
    add_formula(slide, 1.2, 3.0, 10.8, 0.75, f_macro)
    add_bullets(
        slide,
        0.95,
        4.12,
        11.3,
        1.2,
        ["环境仍按原始步长一步一步执行。", "回报、价值和 return 仍逐步计算。", "带宽 actor 只在宏决策起点更新。"],
        size=20,
    )
    add_textbox(slide, 0.95, 6.02, 11.3, 0.5, "目的：避免一步带宽决策很快被下一步覆盖，增强带宽动作 credit。", size=21, bold=True, color=BLUE, align=PP_ALIGN.CENTER)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "5. 整体结果：联合训练超过规则策略")
    add_bullets(
        slide,
        0.95,
        1.45,
        5.8,
        1.5,
        ["3 UAV / 20 GU，时域长度 250。", "64 个环境，训练 300 次更新。", "奖励：positive weighted workload level。", "选星 K=1，带宽 K=5。"],
        size=18,
    )
    add_table(slide, 7.0, 1.55, ["策略", "回报"], [["规则基线", "45.28"], ["最终策略", "58.34"], ["最佳阶段头组合", "59.33"]], [3.1, 2.0], font_size=17)
    add_table(slide, 3.15, 4.15, ["对比", "提升"], [["最终策略 - 规则基线", "+13.06"], ["最佳阶段头 - 规则基线", "+14.05"]], [3.9, 2.5], font_size=18)
    add_textbox(slide, 0.95, 6.0, 11.4, 0.55, "结论：MC-GAE critic + 带宽 K=5 后，联合策略显著超过规则策略。", size=22, bold=True, color=BLUE, align=PP_ALIGN.CENTER)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "6. 各阶段贡献：主要收益来自带宽，选星/加速度有组合叠加")
    add_textbox(slide, 0.9, 1.45, 5.5, 0.35, "最终检查点相对规则", size=20, bold=True, color=BLUE)
    add_table(slide, 0.9, 1.9, ["组合", "提升"], [["仅加速度", "+0.35"], ["仅选星", "+0.11"], ["仅带宽", "+10.60"], ["三阶段联合", "+13.06"]], [2.5, 2.0], font_size=16)
    add_textbox(slide, 6.9, 1.45, 5.5, 0.35, "最佳阶段头相对规则", size=20, bold=True, color=BLUE)
    add_table(slide, 6.9, 1.9, ["组合", "提升"], [["仅加速度", "+0.29"], ["仅选星", "+0.13"], ["仅带宽", "+12.45"], ["加速度+带宽", "+13.41"], ["选星+带宽", "+13.55"], ["三阶段联合", "+14.05"]], [2.7, 2.0], font_size=15)
    add_formula(slide, 1.1, 5.15, 11.1, 0.8, f_gain)
    add_textbox(slide, 0.95, 6.35, 11.5, 0.55, "结论：带宽是主要收益来源；选星和加速度单独收益小，但二者在带宽基础上都有叠加。", size=20, bold=True, color=BLUE, align=PP_ALIGN.CENTER)

    for idx, slide in enumerate(prs.slides, start=1):
        add_textbox(slide, 12.15, 7.05, 0.6, 0.22, str(idx), size=10, color=GRAY, align=PP_ALIGN.RIGHT)

    prs.save(OUT)
    print(OUT.resolve())


if __name__ == "__main__":
    main()
