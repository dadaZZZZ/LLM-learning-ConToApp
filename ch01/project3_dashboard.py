# -*- coding: utf-8 -*-
"""
项目三参考答案：多子图分析看板
要求：基于项目二结果，用 subplots 画 2×2 看板
  - 各类别销售额（柱状）
  - 各类别记录数（柱状）
  - 价格分布（直方图）
  - 销售额随时间的趋势（折线）
配置中文显示，保存为 ch01_report.png。
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 注册系统中文字体，避免图表中文乱码
_FONT = "/Library/Fonts/Arial Unicode.ttf"
fm.fontManager.addfont(_FONT)
plt.rcParams['font.sans-serif'] = [fm.FontProperties(fname=_FONT).get_name()]
plt.rcParams['axes.unicode_minus'] = False               # 负号正常

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    from project2_clean import clean
    df, agg = clean()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # 1) 各类别销售额
    axes[0, 0].bar(agg['类别'], agg['总销售额'], color='#4C72B0')
    axes[0, 0].set_title('各类别销售额')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2) 各类别记录数
    axes[0, 1].bar(agg['类别'], agg['记录数'], color='#55A868')
    axes[0, 1].set_title('各类别记录数')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3) 价格分布直方图
    axes[1, 0].hist(df['单价(元)'], bins=10, color='#C44E52')
    axes[1, 0].set_title('单价分布')

    # 4) 销售额时间趋势
    df['订单日期'] = pd.to_datetime(df['订单日期'], errors='coerce')
    ts = df.dropna(subset=['订单日期']).groupby('订单日期')['销售额'].sum()
    axes[1, 1].plot(ts.index, ts.values, marker='o', color='#8172B3')
    axes[1, 1].set_title('销售额时间趋势')
    axes[1, 1].tick_params(axis='x', rotation=45)

    fig.tight_layout()
    out = os.path.join(HERE, "ch01_report.png")
    fig.savefig(out, dpi=120)
    print("已保存:", out)


if __name__ == "__main__":
    main()
