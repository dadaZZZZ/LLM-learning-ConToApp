# -*- coding: utf-8 -*-
"""
项目二参考答案：清洗一份真实销售数据（含 dtype 诊断 + 脏点定位）
要求：
  1. 读取 store_sales_dirty.csv（真实中文脏数据）；
  2. 清洗前做 dtype 诊断 + 逐类脏点定位（字符串混入 / 空格 / 负值 / 重复）；
  3. 处理缺失值、删除重复行、裁剪异常（数量<0）；
  4. 演示"类别未 strip 会 8 组 vs strip 后 11 组"的脏数据代价；
  5. 按「类别」分组聚合，输出每类总销售额与记录数；
  6. 打印清洗前后关键指标对比。
注：真实文件列名为中文：订单日期 / 商品名称 / 类别 / 数量 / 单价(元) / 门店。
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "store_sales_dirty.csv")


def diagnose(df: pd.DataFrame) -> None:
    """清洗前的体检：dtype + 逐类脏点定位（行号从 1 计，对齐 CSV 数据行）。"""
    print("=" * 60)
    print("【诊断一】各列 dtype 与缺失")
    print(df.dtypes.to_string())
    print("缺失计数:")
    print(df.isna().sum().to_string())

    print("\n【诊断二】脏点定位（行号为 CSV 数据行号，从 1 起）")
    price = df["单价(元)"]
    bad_price = df.index[(~price.isna()) & ~price.astype(str).str.strip().str.match(r"^\d+(\.\d+)?$")]
    print(f"  单价混入非数值字符串的行: {[i + 1 for i in bad_price]}")
    space_cat = df.index[df["类别"].astype(str) != df["类别"].astype(str).str.strip()]
    print(f"  类别含首尾空格的行: {[i + 1 for i in space_cat]}")
    neg = df.index[df["数量"] < 0]
    print(f"  数量为负的行: {[i + 1 for i in neg]}")
    dup = df.index[df.duplicated(keep=False)].tolist()
    print(f"  整行重复的行: {sorted(i + 1 for i in dup)}")

    print("\n【诊断三】脏数据的代价：类别去重前 vs strip 后")
    n_raw = df["类别"].astype(str).nunique()
    n_strip = df["类别"].astype(str).str.strip().nunique()
    print(f"  未 strip 的唯一类别数 = {n_raw}（' 小吃 ' 与 '小吃' 被当成两个）")
    print(f"  strip 后的唯一类别数 = {n_strip}")
    print(f"  → 若漏掉 strip，按类别聚合会多出 {n_raw - n_strip} 个'幽灵类别'")
    print("=" * 60)


def clean(csv_path: str = CSV):
    df = pd.read_csv(csv_path)
    print("原始形状:", df.shape)
    print(df.head(3).to_string())
    diagnose(df)

    # 1) 单价：去掉 "元" 后缀 -> 数值列（原含 "6.0元" 字符串与空值）
    price = (df["单价(元)"].astype(str)
             .str.replace("元", "", regex=False).str.strip())
    df["单价(元)"] = pd.to_numeric(price, errors="coerce")

    # 2) 类别 / 门店：去除首尾空格
    df["类别"] = df["类别"].astype(str).str.strip()
    df["门店"] = df["门店"].astype(str).str.strip()

    # 3) 数量：转数值
    df["数量"] = pd.to_numeric(df["数量"], errors="coerce")

    # 4) 缺失值：单价填均值（注意：此处用全表统计量仅因这是唯一一份数据；
    #    真实工程里必须只用训练集算，见 1.4.4 旁白）
    df["单价(元)"] = df["单价(元)"].fillna(df["单价(元)"].mean())

    # 5) 异常裁剪：数量不应为负
    before = len(df)
    df = df[df["数量"] >= 0]

    # 6) 去重
    df = df.drop_duplicates()

    # 派生：销售额
    df["销售额"] = df["数量"] * df["单价(元)"]

    agg = (df.groupby("类别")
             .agg(总销售额=("销售额", "sum"),
                  记录数=("销售额", "size"),
                  均价=("单价(元)", "mean"))
             .reset_index())

    print(f"\n清洗后形状: {df.shape}  (剔除 {before - len(df)} 行)")
    print(f"分组聚合得到 {len(agg)} 个类别（strip 后才是真实类别数）")
    print(agg.to_string())

    # 清洗前后对比
    print("\n[对比] 行数:", before, "->", len(df))
    print("[对比] 单价缺失:", int(df["单价(元)"].isna().sum()), "（应为0）")
    print("[对比] 重复行:", int(df.duplicated().sum()), "（应为0）")
    print("[对比] 负数量:", int((df["数量"] < 0).sum()), "（应为0）")
    return df, agg


if __name__ == "__main__":
    clean()
