# -*- coding: utf-8 -*-
"""
项目一参考答案：向量化 vs 循环 —— 三算子基准 + 规模扫描
要求：
  1. 对 softmax / 批量矩阵乘 / 行 L2 归一化 三个算子，分别用循环与向量化实现；
  2. 用 time.perf_counter 计时（预热 + 取最优），输出加速倍数；
  3. 扫描 4 个矩阵规模，画"加速比随规模变化"曲线；
  4. 固定随机种子，结果可复现。
产物：project1_timing.png + project1_results.json
"""
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 注册系统中文字体，避免图表中文乱码
_FONT = "/Library/Fonts/Arial Unicode.ttf"
fm.fontManager.addfont(_FONT)
plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=_FONT).get_name()]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 0
REP = 11          # 每格重复次数（取最优，抑制调度噪声）
WARM = 3          # 预热次数（预热页表 / 指令缓存 / numpy 首次内核）


# ---------- 算子：循环版 ----------
def softmax_loop(X):
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        m = X[i].max()
        e = np.exp(X[i] - m)          # numpy 单行向量（已够向量化）
        out[i] = e / e.sum()
    return out


def matmul_loop(X, W):
    n, k = X.shape
    out = np.zeros((n, W.shape[1]))
    for i in range(n):
        for j in range(W.shape[1]):
            s = 0.0
            for t in range(k):        # 纯 Python 三重循环：解释器逐元素求和
                s += X[i, t] * W[t, j]
            out[i, j] = s
    return out


def normalize_loop(M):
    out = np.zeros_like(M)
    for i in range(M.shape[0]):
        row = M[i]
        out[i] = row / np.sqrt(np.sum(row ** 2))
    return out


# ---------- 算子：向量化版 ----------
def softmax_vec(X):
    m = np.max(X, axis=1, keepdims=True)      # 减最大值保证数值稳定
    e = np.exp(X - m)
    return e / np.sum(e, axis=1, keepdims=True)


def matmul_vec(X, W):
    return X @ W


def normalize_vec(M):
    return M / np.linalg.norm(M, axis=1, keepdims=True)


OPS = [
    ("softmax", softmax_loop, softmax_vec, 50, 8),     # (5000,50) 上循环版偏慢，缩放
    ("matmul", matmul_loop, matmul_vec, 20, 4),        # (2000,20)@(...,4) 纯 Python 三重循环
    ("行归一化", normalize_loop, normalize_vec, 40, 12),
]


def time_best(fn, args, rep=REP, warm=WARM):
    for _ in range(warm):
        fn(*args)
    best = float("inf")
    for _ in range(rep):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    np.random.seed(SEED)
    results = {"seed": SEED, "rep": REP, "warm": WARM, "ops": {}}

    # 规模扫描（行维），列维随行维等比缩放
    sizes = [500, 2000, 5000, 20000]
    scan = {}
    for name, lf, vf, sc, wc in OPS:
        ratios = []
        for n in sizes:
            k, m = int(n * sc / 5000), int(n * wc / 5000)
            k = max(2, k)
            m = max(2, m)
            X = np.random.randn(n, k)
            W = np.random.randn(k, m) if name == "matmul" else None
            args = (X, W) if name == "matmul" else (X,)
            t_loop = time_best(lf, args)
            t_vec = time_best(vf, args)
            ratios.append(t_loop / t_vec)
        scan[name] = ratios

    # 单点基准（正文引用的代表规模），输出加速比 + 一致性
    for name, lf, vf, sc, wc in OPS:
        n = 2000
        k, m = int(n * sc / 5000), int(n * wc / 5000)
        k, m = max(2, k), max(2, m)
        X = np.random.randn(n, k)
        if name == "matmul":
            W = np.random.randn(k, m)
            r_loop, r_vec = matmul_loop(X, W), matmul_vec(X, W)
            tl, tv = time_best(lf, (X, W)), time_best(vf, (X, W))
        else:
            r_loop, r_vec = lf(X), vf(X)
            tl, tv = time_best(lf, (X,)), time_best(vf, (X,))
        diff = float(np.abs(r_loop - r_vec).max())
        results["ops"][name] = {
            "shape": [n, k] + ([m] if name == "matmul" else []),
            "loop_ms": tl * 1e3, "vec_ms": tv * 1e3,
            "speedup": tl / tv, "max_diff": diff,
        }
        print(f"[{name}] 形状{n}x{k}{('x'+str(m)) if name=='matmul' else ''}  "
              f"循环 {tl*1e3:.3f} ms -> 向量化 {tv*1e3:.4f} ms  "
              f"加速 {tl/tv:.1f}x  最大差异 {diff:.2e}")

    print("\n规模扫描（加速比）:")
    for name, ratios in scan.items():
        print(f"  {name}: " + "  ".join(f"{r:.0f}x" for r in ratios))

    # ---- 图：左=单点三算子加速比，右=加速比随规模变化 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    names = [op[0] for op in OPS]
    speeds = [results["ops"][op[0]]["speedup"] for op in OPS]
    bars = axes[0].bar(names, speeds, color=["#4C72B0", "#55A868", "#C44E52"])
    for b, s in zip(bars, speeds):
        axes[0].text(b.get_x() + b.get_width() / 2, s * 1.02, f"{s:.0f}x",
                     ha="center", va="bottom", fontsize=10)
    axes[0].set_ylabel("加速倍数（循环 / 向量化）")
    axes[0].set_title("三个算子：向量化相对纯循环的加速")
    axes[0].set_ylim(0, max(speeds) * 1.18)

    for name, ratios in scan.items():
        axes[1].plot([f"{s/1000:g}k" for s in sizes], ratios, marker="o", linewidth=2)
        axes[1].annotate(f"{ratios[-1]:.0f}x", (len(sizes) - 1, ratios[-1]),
                         textcoords="offset points", xytext=(6, 0))
    axes[1].set_ylabel("加速倍数")
    axes[1].set_xlabel("矩阵行数（列维等比缩放）")
    axes[1].set_title("加速比随规模变化（越大越拉开）")
    axes[1].grid(alpha=0.3)
    axes[1].set_yscale("log")
    fig.tight_layout()
    out = os.path.join(HERE, "project1_timing.png")
    fig.savefig(out, dpi=120)
    print("已保存:", out)

    with open(os.path.join(HERE, "project1_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("已保存 project1_results.json")


if __name__ == "__main__":
    main()
