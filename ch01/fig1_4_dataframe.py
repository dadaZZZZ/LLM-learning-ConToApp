# -*- coding: utf-8 -*-
"""
第 1.4.1 节插图：DataFrame 结构直观图示
把 1.4.1 代码块里的 df 画成"行索引 × 列"的二维表格，
右侧点出"每列是一个 Series"与"按标签对齐（对不上补 NaN）"，
底部收在 .loc / .iloc 取法区别。
产物：fig1_4_dataframe.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

_FONT = "/Library/Fonts/Arial Unicode.ttf"
fm.fontManager.addfont(_FONT)
plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=_FONT).get_name()]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- 配色（与书稿其它图一致的扁平色板）----
C_GRID = "#888780"      # 网格线
C_FRAME = "#5F5E5A"     # 外框
C_HDR_FILL = "#B5D4F4"  # 列头底（蓝100）
C_HDR_EDGE = "#185FA5"  # 列头边（蓝600）
C_HDR_TXT = "#0C447C"   # 列头文字（蓝800）
C_HDR_SUB = "#185FA5"   # 列头 dtype 小字
C_CAT = "#3C3489"       # category 列文字（紫800）
C_PRICE = "#0F6E56"     # price 列文字（青600）
C_QTY = "#5F5E5A"       # qty 列文字（灰600）
C_TXT = "#2C2C2A"       # 主文字
C_SEC = "#888780"       # 次级文字
C_CARD_S_FILL = "#E1F5EE"   # Series 卡底（青50）
C_CARD_S_EDGE = "#0F6E56"
C_CARD_A_FILL = "#E6F1FB"   # 对齐卡底（蓝50）
C_CARD_A_EDGE = "#185FA5"
C_TAKE_FILL = "#F1EFE8"     # 底部要点底（灰50）
C_RED = "#A32D2D"           # 警示红（红600）


def main():
    fig = plt.figure(figsize=(13.6, 8.64), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 680)
    ax.set_ylim(0, 432)
    ax.axis("off")

    # 标题
    ax.text(40, 420, "DataFrame 的结构：行索引 × 列，每列是一个 Series",
            fontsize=13.5, fontweight="bold", color=C_TXT, va="center")
    ax.text(40, 402, "df = pd.DataFrame({ 'category': [...], 'price': [...], 'qty': [...] })",
            fontsize=10.5, color=C_SEC, va="center")

    # 行索引标签（竖排）
    ax.text(48, 222, "行索引 index（label）", fontsize=10.5, color=C_SEC,
            ha="center", va="center", rotation=90)

    # 列头（三个 dtype 胶囊）
    headers = [("category", "object"), ("price", "int64"), ("qty", "int64")]
    col_x = [160, 304, 448]   # 列中心 x
    def rbox(x, y, w, h, radius, **kw):
        return FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                              **kw)

    for (name, dt), cx in zip(headers, col_x):
        rect = rbox(cx - 70, 352, 140, 30, 6,
                    facecolor=C_HDR_FILL, edgecolor=C_HDR_EDGE, linewidth=0.6)
        ax.add_patch(rect)
        ax.text(cx, 372, name, fontsize=11.5, fontweight="bold",
                color=C_HDR_TXT, ha="center", va="center")
        ax.text(cx, 358, "dtype: " + dt, fontsize=9.5,
                color=C_HDR_SUB, ha="center", va="center")

    # 表格网格：x 90..518，y 98..323（5 行）
    x0, x1, y0, y1 = 90, 518, 98, 323
    row_h = (y1 - y0) / 5
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                               facecolor="none", edgecolor=C_FRAME, linewidth=0.8))
    for vx in (230, 374):
        ax.plot([vx, vx], [y0, y1], color=C_GRID, linewidth=0.6)
    for k in range(1, 5):
        yy = y0 + k * row_h
        ax.plot([x0, x1], [yy, yy], color=C_GRID, linewidth=0.6)

    # 行索引数字
    for i in range(5):
        cy = y1 - (i + 0.5) * row_h
        ax.text(80, cy, str(i), fontsize=11.5, color=C_SEC, ha="right", va="center")

    # 数据单元格
    cat_vals = ["'A'", "'B'", "'A'", "'B'", "'A'"]
    price_vals = ["10", "20", "15", "25", "12"]
    qty_vals = ["1", "2", "3", "4", "5"]
    for i in range(5):
        cy = y1 - (i + 0.5) * row_h
        ax.text(col_x[0], cy, cat_vals[i], fontsize=11.5, color=C_CAT, ha="center", va="center")
        ax.text(col_x[1], cy, price_vals[i], fontsize=11.5, color=C_PRICE, ha="center", va="center")
        ax.text(col_x[2], cy, qty_vals[i], fontsize=11.5, color=C_QTY, ha="center", va="center")

    # 表格下方说明
    ax.text(304, 84, "5 行 × 3 列 = 一张表；每列同 dtype，可整体做数值运算",
            fontsize=10.5, color=C_SEC, ha="center", va="center")

    # 引导线：price 列 -> Series 卡
    arrow = FancyArrowPatch((374, 262), (540, 322),
                            arrowstyle="-|>", mutation_scale=12,
                            linestyle=(0, (4, 3)), color=C_HDR_EDGE, linewidth=1.0)
    ax.add_patch(arrow)

    # Series 卡
    ax.add_patch(rbox(540, 312, 140, 86, 10,
                      facecolor=C_CARD_S_FILL, edgecolor=C_CARD_S_EDGE, linewidth=0.6))
    ax.text(610, 380, "df['price']", fontsize=11.5, fontweight="bold",
            color="#085041", ha="center", va="center")
    ax.text(610, 364, "是一个 Series（一列）", fontsize=9.5,
            color=C_CARD_S_EDGE, ha="center", va="center")
    ax.plot([552, 668], [352, 352], color="#5DCAA5", linewidth=0.6)
    ax.text(560, 338, "0   10", fontsize=10.5, color=C_CARD_S_EDGE, ha="left", va="center")
    ax.text(560, 324, "2   15", fontsize=10.5, color=C_CARD_S_EDGE, ha="left", va="center")

    # 对齐卡
    ax.add_patch(rbox(540, 194, 140, 106, 10,
                      facecolor=C_CARD_A_FILL, edgecolor=C_CARD_A_EDGE, linewidth=0.6))
    ax.text(610, 284, "按标签对齐", fontsize=11.5, fontweight="bold",
            color=C_HDR_TXT, ha="center", va="center")
    ax.plot([552, 668], [272, 272], color="#85B7EB", linewidth=0.6)
    ax.text(560, 254, "df1 + df2", fontsize=10.5, color=C_HDR_EDGE, ha="left", va="center",
            family="monospace")
    ax.text(560, 236, "不是按位置，而是按", fontsize=10, color=C_HDR_EDGE, ha="left", va="center")
    ax.text(560, 222, "行 index 匹配", fontsize=10, color=C_HDR_EDGE, ha="left", va="center")
    ax.text(560, 206, "对不上 → 自动补 NaN", fontsize=10, color=C_RED, ha="left", va="center")

    # 底部要点条
    ax.add_patch(rbox(90, 40, 590, 34, 8,
                      facecolor=C_TAKE_FILL, edgecolor=C_GRID, linewidth=0.5))
    ax.text(104, 57, ".loc[标签] 按名字取　·　.iloc[位置] 按序号取　——　索引一旦重置或重复，运算结果就会和“看着以为”的不同",
            fontsize=10, color=C_SEC, ha="left", va="center")

    out = os.path.join(HERE, "fig1_4_dataframe.png")
    fig.savefig(out)
    plt.close(fig)
    print("已保存:", out)


if __name__ == "__main__":
    main()
