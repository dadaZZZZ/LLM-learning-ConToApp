# -*- coding: utf-8 -*-
"""
第 1.3.2 节插图：广播机制直观示意图
以正文例子 (2,3) + (3,) 为例，画清三步：
  ① a (2,3) 与 b (3,) 的形状
  ② 对齐：给 b 前面补一维 -> (1,3)，再沿第 0 维拉伸成 (2,3)（虚线框表示"逻辑复制"，不占内存）
  ③ 逐元素相加 -> 结果 (2,3)
右侧两张规则卡：右对齐 + 补一 / 尺寸相等或一方为 1。
布局：网格区 x 40..480，规则卡 x 560..668，横向不重叠（间隔 80）。
产物：fig1_2_broadcast.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

_FONT = "/Library/Fonts/Arial Unicode.ttf"
fm.fontManager.addfont(_FONT)
plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=_FONT).get_name()]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))

C_TXT = "#2C2C2A"
C_SEC = "#888780"
C_GRID = "#888780"
C_A_FILL = "#E6F1FB"
C_A_EDGE = "#185FA5"
C_A_TXT = "#0C447C"
C_B_FILL = "#FAEEDA"
C_B_EDGE = "#BA7517"
C_B_TXT = "#854F0B"
C_STRETCH_FILL = "#FBEAF0"
C_STRETCH_EDGE = "#D4537E"
C_R_FILL = "#EAF3DE"
C_R_EDGE = "#639922"
C_R_TXT = "#3B6D11"
C_PLUS = "#2C2C2A"
C_CARD1_FILL = "#E6F1FB"
C_CARD1_EDGE = "#185FA5"
C_CARD2_FILL = "#E1F5EE"
C_CARD2_EDGE = "#0F6E56"
C_DIM = "#888780"


def rbox(ax, x, y, w, h, radius, **kw):
    return FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={radius}", **kw)


def draw_grid(ax, x, y, rows, cols, cw, ch, values, fill, edge, txt,
              stretch_rows=(), dashed_edge=()):
    for r in range(rows):
        for c in range(cols):
            is_stretch = r in stretch_rows
            f = C_STRETCH_FILL if is_stretch else fill
            e = C_STRETCH_EDGE if is_stretch else edge
            ls = (0, (3, 2)) if r in dashed_edge else "solid"
            ax.add_patch(Rectangle((x + c * cw, y + (rows - 1 - r) * ch),
                                   cw, ch, facecolor=f, edgecolor=e,
                                   linewidth=0.8, linestyle=ls))
            ax.text(x + c * cw + cw / 2, y + (rows - 1 - r) * ch + ch / 2,
                    values[r][c], fontsize=12, ha="center", va="center",
                    color=C_STRETCH_EDGE if is_stretch else txt,
                    fontweight="bold" if not is_stretch else "normal")


def main():
    fig = plt.figure(figsize=(13.6, 7.8), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 680)
    ax.set_ylim(0, 390)
    ax.axis("off")

    ax.text(40, 372, "广播：(2, 3) + (3,) 是怎么对齐相加的",
            fontsize=13.5, fontweight="bold", color=C_TXT, va="center")
    ax.text(40, 354, "b 是 (3,)，对齐时先在前面补一维成 (1, 3)，再沿第 0 维拉伸成 (2, 3)",
            fontsize=10.5, color=C_SEC, va="center")

    cw, ch = 40, 34          # 单元宽高（缩小，给规则卡留空间）
    base_y = 150

    # ---- a 数组 (2,3)：x 60..180 ----
    axa = 60
    ax.text(axa + 1.5 * cw, base_y + 2 * ch + 32, "a  形状 (2, 3)", fontsize=12,
            fontweight="bold", color=C_A_TXT, ha="center", va="center")
    draw_grid(ax, axa, base_y, 2, 3, cw, ch, [["1", "2", "3"], ["4", "5", "6"]],
              C_A_FILL, C_A_EDGE, C_A_TXT)
    ax.text(axa - 8, base_y + ch, "2", fontsize=11, color=C_DIM, ha="right", va="center")
    ax.text(axa + 1.5 * cw, base_y - 14, "3", fontsize=11, color=C_DIM, ha="center", va="center")

    # plus（a 右 180 与 b 左 214 之间中点 197）
    ax.text(197, base_y + ch, "+", fontsize=22, fontweight="bold",
            color=C_PLUS, ha="center", va="center")

    # ---- b 数组：x 214..334 ----
    axb = 214
    # 原始 (3,) 单行
    ax.text(axb + 1.5 * cw, base_y + 2 * ch + 32, "b  形状 (3,)", fontsize=12,
            fontweight="bold", color=C_B_TXT, ha="center", va="center")
    draw_grid(ax, axb, base_y + ch, 1, 3, cw, ch, [["10", "20", "30"]],
              C_B_FILL, C_B_EDGE, C_B_TXT)
    ax.text(axb - 8, base_y + ch + ch / 2, "1", fontsize=11, color=C_DIM, ha="right", va="center")

    # 对齐后 (2,3)：第二行是拉伸出来的逻辑复制
    ax.text(axb + 1.5 * cw, base_y + 10, "对齐后 (1,3) → 拉伸 (2,3)",
            fontsize=10, color=C_STRETCH_EDGE, ha="center", va="center")
    draw_grid(ax, axb, base_y - ch, 2, 3, cw, ch, [["10", "20", "30"], ["10", "20", "30"]],
              C_B_FILL, C_B_EDGE, C_B_TXT, stretch_rows={1}, dashed_edge={1})
    # 向下箭头：原始 b -> 对齐后
    ax.add_patch(FancyArrowPatch((axb + 1.5 * cw, base_y + ch - 5),
                                 (axb + 1.5 * cw, base_y - ch + ch + 8),
                                 arrowstyle="-|>", mutation_scale=13,
                                 color=C_B_EDGE, linewidth=1.2))
    ax.text(axb + 1.5 * cw + 10, base_y - ch / 2, "复制一行", fontsize=9.5,
            color=C_STRETCH_EDGE, ha="left", va="center")

    # equals（b 右 334 与结果左 360 之间中点 347）
    ax.text(347, base_y + ch, "=", fontsize=22, fontweight="bold",
            color=C_PLUS, ha="center", va="center")

    # ---- 结果 (2,3)：x 360..480 ----
    axr = 360
    ax.text(axr + 1.5 * cw, base_y + 2 * ch + 32, "结果  形状 (2, 3)", fontsize=12,
            fontweight="bold", color=C_R_TXT, ha="center", va="center")
    draw_grid(ax, axr, base_y, 2, 3, cw, ch, [["11", "22", "33"], ["14", "25", "36"]],
              C_R_FILL, C_R_EDGE, C_R_TXT)

    # 底部要点条（x 40..528，不与规则卡重叠）
    ax.add_patch(rbox(ax, 40, 30, 488, 40, 8, facecolor="#F1EFE8",
                      edgecolor=C_GRID, linewidth=0.5))
    ax.text(54, 58, "拉伸是「逻辑复制」：numpy 不真的把数据拷贝两遍（那会浪费内存），",
            fontsize=10.5, color=C_SEC, ha="left", va="center")
    ax.text(54, 42, "只在计算时按「这一行重复两遍」来算——零拷贝，所以广播又快又省。",
            fontsize=10.5, color=C_SEC, ha="left", va="center")

    # ---- 右侧规则卡：x 560..668（与网格区右边界 480 间隔 80）----
    ax.add_patch(rbox(ax, 560, 250, 108, 92, 10, facecolor=C_CARD1_FILL,
                      edgecolor=C_CARD1_EDGE, linewidth=0.6))
    ax.text(614, 326, "规则一", fontsize=11.5, fontweight="bold", color=C_CARD1_EDGE,
            ha="center", va="center")
    ax.text(614, 308, "从最右端", fontsize=10, color=C_CARD1_EDGE, ha="center", va="center")
    ax.text(614, 292, "开始对齐", fontsize=10, color=C_CARD1_EDGE, ha="center", va="center")
    ax.text(614, 274, "维度逐位比较", fontsize=10, color=C_CARD1_EDGE, ha="center", va="center")

    ax.add_patch(rbox(ax, 560, 138, 108, 100, 10, facecolor=C_CARD2_FILL,
                      edgecolor=C_CARD2_EDGE, linewidth=0.6))
    ax.text(614, 222, "规则二", fontsize=11.5, fontweight="bold", color=C_CARD2_EDGE,
            ha="center", va="center")
    ax.text(614, 202, "每维要么相等", fontsize=10, color=C_CARD2_EDGE, ha="center", va="center")
    ax.text(614, 186, "要么一方为 1", fontsize=10, color=C_CARD2_EDGE, ha="center", va="center")
    ax.text(614, 166, "否则报错", fontsize=10, color=C_CARD2_EDGE, ha="center", va="center")

    out = os.path.join(HERE, "fig1_2_broadcast.png")
    fig.savefig(out)
    plt.close(fig)
    print("已保存:", out)


if __name__ == "__main__":
    main()
