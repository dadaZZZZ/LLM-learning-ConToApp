# -*- coding: utf-8 -*-
"""
第 1.3.5 节演示：dtype 陷阱 + 内存布局 + 向量化为何快
产出三张结论：
  ① 同一列数据：数值 dtype 可聚合，object(dtype=O) 直接抛 TypeError；
  ② 混入一个 "元" 字，整列被迫退化为 object，mean 立刻失败；
  ③ 不同 dtype 的 itemsize / 总内存差异（float64 vs int32 vs object=O）；
  ④ 计时：对 10 万元素求和，数值列 vs object 列（逐元素走解释器）的开销。
产物：dtype_demo.png
"""
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_FONT = "/Library/Fonts/Arial Unicode.ttf"
fm.fontManager.addfont(_FONT)
plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=_FONT).get_name()]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))


def time_of(fn, rep=5, warm=1):
    for _ in range(warm):
        fn()
    best = float("inf")
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _sum_obj(arr):
    """object(dtype=O) 列求和：逐元素走 Python 解释器。"""
    s = 0.0
    for x in arr:
        s += x
    return s


def main():
    rng = np.random.default_rng(0)
    vals = rng.normal(loc=10.0, scale=3.0, size=1_000_000)

    float_col = vals.astype(np.float64)
    int32_col = vals.astype(np.int32)
    int64_col = vals.astype(np.int64)
    # 关键：dtype=object（O）数组，每个格子是一个独立的 Python float 对象
    obj_col = np.empty(len(vals), dtype=object)
    obj_col[:] = vals

    print("【一】dtype / 内存 / 能否聚合")
    for name, col in [("float64", float_col), ("int32", int32_col),
                      ("int64", int64_col), ("object(O)", obj_col)]:
        if col.dtype == object:
            can = "TypeError"
            t = None
        else:
            can = "可以"
            t = time_of(lambda c=col: c.mean())
        print(f"  {name:10s} itemsize={col.dtype.itemsize}B  "
              f"总内存={col.nbytes/1e6:5.2f}MB  mean={can}"
              + (f"  耗时={t*1e3:.3f}ms" if t else ""))

    # ---- ② 一个 "元" 字毁整列 ----
    print("\n【二】一个 '元' 字如何毁掉整列")
    prices = np.array([2.0, 3.5, 5.0, 4.5, 8.0, 12.5, 6.0, 3.5, 5.0, 2.0])
    print("  干净列 dtype =", prices.dtype, " mean =", prices.mean())
    dirty = np.empty(len(prices), dtype=object)
    dirty[:] = prices
    dirty[7] = "6.0元"                       # 混入一个带单位的字符串
    print(f"  混入 '6.0元' 后 dtype = {dirty.dtype}")
    try:
        dirty.mean()
    except TypeError as e:
        print("  dirty.mean() 抛错:", type(e).__name__)
    # 修复：统一转回数值
    fixed = pd_to_numeric(dirty)
    print("  修复后 dtype =", fixed.dtype, " mean =", round(float(fixed.mean()), 4))

    # ---- ③ 内存布局对比（object 真实开销 = 指针数组 + 每个独立对象）----
    obj_size = sys.getsizeof(1.0)                       # 一个 Python float ≈ 24B
    mem64, mem32 = float_col.nbytes, int32_col.nbytes
    memobj_real = obj_col.nbytes + len(obj_col) * obj_size   # 指针 + 对象
    print(f"\n【三】同样 1e6 个数：float64={mem64/1e6:.0f}MB  "
          f"int32={mem32/1e6:.0f}MB  object(O)≈{memobj_real/1e6:.0f}MB"
          f"（object 是 float64 的 {memobj_real/mem64:.0f} 倍："
          f"8MB 指针 + {obj_size}B/个 独立对象）")

    # ---- ④ 计时：数值列 vs object 列求和（10 万元素）----
    small = 100_000
    t_num = time_of(lambda: vals[:small].sum())
    t_obj = time_of(lambda: _sum_obj(obj_col[:small]))
    print(f"\n【四】{small//10000}0 万元素求和：数值列 .sum()={t_num*1e3:.3f}ms  "
          f"object 列逐元素累加={t_obj*1e3:.2f}ms  差 {t_obj/t_num:.0f} 倍")

    # ---- 图：左=dtype 内存柱状，右=数值列 vs object 列求和 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    mems = [mem64 / 1e6, mem32 / 1e6, memobj_real / 1e6]
    axes[0].bar(["float64", "int32", "object(O)"], mems,
                color=["#4C72B0", "#55A868", "#C44E52"])
    for i, v in enumerate(mems):
        axes[0].text(i, v + 0.1, f"{v:.0f}MB", ha="center", fontsize=10)
    axes[0].set_ylabel("总内存 (MB)")
    axes[0].set_title("同一 1e6 个数的三种 dtype 内存占用")

    axes[1].bar(["数值列 .sum()", "object 列逐元素累加"],
                [t_num * 1e3, t_obj * 1e3], color=["#55A868", "#C44E52"])
    for i, v in enumerate([t_num * 1e3, t_obj * 1e3]):
        axes[1].text(i, v * 1.05, f"{v:.2f}ms", ha="center", fontsize=10)
    axes[1].set_ylabel("耗时 (ms)")
    axes[1].set_title(f"{small//10000}0 万元素求和：数值列 vs object 列")
    axes[1].set_yscale("log")
    fig.tight_layout()
    out = os.path.join(HERE, "dtype_demo.png")
    fig.savefig(out, dpi=120)
    print("\n已保存:", out)


def pd_to_numeric(arr):
    out = np.empty(len(arr), dtype=np.float64)
    for i, x in enumerate(arr):
        try:
            out[i] = float(str(x).replace("元", "").strip())
        except ValueError:
            out[i] = np.nan
    return out


if __name__ == "__main__":
    main()
