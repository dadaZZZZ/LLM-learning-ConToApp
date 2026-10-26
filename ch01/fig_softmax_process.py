# -*- coding: utf-8 -*-
"""图：softmax 计算过程示意（以一行 x=[2.0, 1.0, 0.1] 为例）。

实跑数字（numpy 2.x）：
  m = 2.0
  exp(x-m) = [1.0000, 0.3679, 0.1496]
  s = 1.5174
  p = [0.6590, 0.2424, 0.0986]   （和 = 1.0）

布局：左→右 5 个阶段
  1. logits  x=[2.0, 1.0, 0.1]
  2. 减最大值 m=2.0  →  [0.0, -1.0, -1.9]
  3. 取指数 exp      →  [1.0000, 0.3679, 0.1496]
  4. 求和 s=1.5174
  5. 归一化 p        →  [0.6590, 0.2424, 0.0986]  和=1

产物：ch01/fig_softmax_process.png  (13.6 × 6.4 in)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------- 中文字体 ----------
fm.fontManager.addfont("/Library/Fonts/Arial Unicode.ttf")
plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

# ---------- 画布 ----------
fig, ax = plt.subplots(figsize=(13.0, 6.4), dpi=150)
ax.set_xlim(0, 1060)
ax.set_ylim(0, 480)
ax.axis("off")

# 标题
ax.text(530, 452, "softmax 计算过程（一行 x = [2.0, 1.0, 0.1]）",
        ha="center", va="center", fontsize=18, fontweight="bold")

# ---------- 配色 ----------
C_X      = "#3b82f6"   # logits 蓝
C_SHIFT  = "#8b5cf6"   # 减最大值 紫
C_EXP    = "#f59e0b"   # 取指数 橙
C_SUM    = "#64748b"   # 求和 灰
C_PROB   = "#10b981"   # 归一化 绿
C_BG     = "#f1f5f9"
C_TXT    = "#1e293b"
C_SUB    = "#475569"

# ---------- 5 个阶段卡（每张宽 200px） ----------
# 每个卡: (x0, x1, 标题, 副标题, 元素列表, 颜色)
cards = [
    (   10,  210, "① logits",     "原始打分 x",
        [("2.0",  "x₁"), ("1.0",  "x₂"), ("0.1",  "x₃")], C_X),
    ( 220,  420, "② 减最大值",     "m = max(x) = 2.0",
        [("0.0",  "2.0 − 2.0"), ("-1.0", "1.0 − 2.0"), ("-1.9", "0.1 − 2.0")], C_SHIFT),
    ( 430,  630, "③ 取指数",       "e^(x - m)",
        [("1.0000", "e^0"), ("0.3679", "e^-1"), ("0.1496", "e^-1.9")], C_EXP),
    ( 640,  840, "④ 求和",         "s = sum(e^(x - m))",
        [("1.5174", "= 1.0000 + 0.3679 + 0.1496")], C_SUM),
    ( 850, 1050, "⑤ 归一化",       "p = e^(x-m) / s",
        [("0.6590", "1.0000/1.5174"), ("0.2424", "0.3679/1.5174"), ("0.0986", "0.1496/1.5174")], C_PROB),
]

# 卡高恢复原值 310；小框行高从 36 加到 54（1.5 倍）
card_y0, card_y1 = 70, 380
row_h = 54   # 小框高度 = 原 36 × 1.5
for (x0, x1, title, sub, elems, color) in cards:
    # 卡背景
    ax.add_patch(FancyBboxPatch(
        (x0, card_y0), x1 - x0, card_y1 - card_y0,
        boxstyle="round,pad=0,rounding_size=10",
        linewidth=1.2, edgecolor=color, facecolor=C_BG, zorder=1))
    # 标题条
    ax.add_patch(FancyBboxPatch(
        (x0, card_y1 - 56), x1 - x0, 56,
        boxstyle="round,pad=0,rounding_size=10",
        linewidth=0, facecolor=color, zorder=2))
    ax.text((x0 + x1) / 2, card_y1 - 28, title,
            ha="center", va="center", fontsize=15, fontweight="bold", color="white", zorder=3)
    ax.text((x0 + x1) / 2, card_y1 - 72, sub,
            ha="center", va="center", fontsize=11, color=C_SUB, zorder=3)

    # 元素行
    n = len(elems)
    area_top = card_y1 - 90
    area_bot = card_y0 + 18
    area_h = area_top - area_bot
    slot = area_h / n
    # 卡内可用宽度（去掉左右各 12px 内边距）
    inner_w = (x1 - x0) - 24
    for k, (val, note) in enumerate(elems):
        cy = area_top - slot * (k + 0.5)
        # 跳过"空值占位行"：不画框、只画右侧斜体说明（用于 ④ 求和卡的第二行注释）
        if val == "":
            ax.text((x0 + x1) / 2, cy, note, ha="center", va="center",
                    fontsize=10, color=C_SUB, style="italic", zorder=4)
            continue
        ax.add_patch(FancyBboxPatch(
            (x0 + 12, cy - row_h / 2), inner_w, row_h,
            boxstyle="round,pad=0,rounding_size=6",
            linewidth=0.8, edgecolor=color, facecolor="white", zorder=3))
        # 单元素且 val+note 总长 > 14 字符 → 拆成两行（val 上行、note 下行）
        if n == 1 and len(val) + len(note) > 14:
            cy2 = cy + 14   # 上行
            cy3 = cy - 14   # 下行
            ax.text((x0 + x1) / 2, cy2, val, ha="center", va="center",
                    fontsize=15, fontweight="bold", color=color, zorder=4)
            ax.text((x0 + x1) / 2, cy3, note, ha="center", va="center",
                    fontsize=11, color=C_SUB, zorder=4)
        else:
            ax.text(x0 + 24, cy, val, ha="left", va="center",
                    fontsize=14, fontweight="bold", color=color, zorder=4)
            ax.text(x1 - 18, cy, note, ha="right", va="center",
                    fontsize=9.5, color=C_SUB, zorder=4)

# ---------- 阶段间箭头 ----------
arrow_y = (card_y0 + card_y1) / 2
gaps = [(210, 220), (420, 430), (630, 640), (840, 850)]
for xa, xb in gaps:
    ax.add_patch(FancyArrowPatch(
        (xa + 2, arrow_y), (xb - 2, arrow_y),
        arrowstyle="-|>", mutation_scale=22,
        linewidth=2.2, color="#0f172a", zorder=5))

# ---------- 底部要点条 ----------
note_y = 38
ax.add_patch(FancyBboxPatch(
    (10, note_y - 22), 1040, 44,
    boxstyle="round,pad=0,rounding_size=8",
    linewidth=1, edgecolor="#cbd5e1", facecolor="#f8fafc", zorder=1))
ax.text(530, note_y,
        "关键点：先减 m 防止 exp 溢出（数值稳定）；最后除 s 保证 p 之和 = 1（归一化为概率）",
        ha="center", va="center", fontsize=12, color=C_TXT, zorder=3)

import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_softmax_process.png")
plt.tight_layout()
plt.savefig(out, bbox_inches="tight", facecolor="white")
print("OK", out)
