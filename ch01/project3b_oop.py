# -*- coding: utf-8 -*-
"""
第 1.6 节演示：面向对象组织机器学习流水线
  ① LinearModel 补全 fit()：循环调用 train_step，记录每轮 loss，画收敛曲线；
  ② 继承示例：BaseDataset -> CsvDataset（加从 CSV 加载 + 按批迭代）；
  ③ 防数据泄漏的标准化工具类 FittedStandardizer（只在训练集 fit，再 transform）。
产物：oop_curve.png（loss 收敛曲线 + 标准化前后对照）
"""
import os

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
SEED = 0


class FittedStandardizer:
    """标准化器：参数只能在训练集上 fit，之后 transform 到任意集合。

    关键纪律：fit 与 transform 分开，杜绝"在整张表上算 mean/std 再回灌训练集"
    这种数据泄漏（见 1.4.4 旁白）。
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0      # 防零除
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise RuntimeError("must fit() before transform()")
        return (X - self.mean_) / self.std_


class LinearModel:
    def __init__(self, n_features, lr=0.1):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = lr
        self.loss_history = []

    def predict(self, X):
        return X @ self.w + self.b

    def train_step(self, X, y):
        pred = self.predict(X)
        err = pred - y
        self.w -= self.lr * (X.T @ err) / len(X)
        self.b -= self.lr * err.mean()
        return float(np.mean(err ** 2))

    def fit(self, X, y, epochs=2000, verbose_every=200):
        for ep in range(epochs):
            loss = self.train_step(X, y)
            self.loss_history.append(loss)
            if ep % verbose_every == 0:
                print(f"  epoch {ep:4d}  MSE={loss:.6f}")
        return self


class BaseDataset:
    """数据集基类：只规定"可取长度"这一契约。"""

    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)

    def __len__(self):
        return len(self.data)


class CsvDataset(BaseDataset):
    """继承 BaseDataset，扩展：记录列名 + 按批迭代（供训练循环用）。"""

    def __init__(self, data, columns, batch_size=32, seed=SEED):
        super().__init__(data)
        self.columns = list(columns)
        self.batch_size = batch_size
        self._rng = np.random.default_rng(seed)

    def iter_batches(self, shuffle=True):
        n = len(self)
        idx = self._rng.permutation(n) if shuffle else np.arange(n)
        for start in range(0, n, self.batch_size):
            b = idx[start:start + self.batch_size]
            yield self.data[b]


def main():
    rng = np.random.default_rng(SEED)
    # 造一份有噪声的线性数据：y = 2*x0 + 3*x1 + 噪声
    X_raw = rng.normal(size=(500, 2)) * 5.0
    y = 2 * X_raw[:, 0] + 3 * X_raw[:, 1] + rng.normal(scale=0.5, size=500)

    # 防泄漏：标准化器只 fit 训练集
    n_tr = 400
    X_tr_raw, X_va_raw = X_raw[:n_tr], X_raw[n_tr:]
    y_tr, y_va = y[:n_tr], y[n_tr:]

    scaler = FittedStandardizer().fit(X_tr_raw)
    X_tr = scaler.transform(X_tr_raw)
    X_va = scaler.transform(X_va_raw)      # 用训练集的 mean/std 套到验证集
    print("[FittedStandardizer] mean_ =", np.round(scaler.mean_, 3),
          "std_ =", np.round(scaler.std_, 3))

    # 训练
    model = LinearModel(n_features=2, lr=0.05)
    print("[LinearModel] 开始训练 (2000 epochs, 全批量):")
    model.fit(X_tr, y_tr, epochs=2000, verbose_every=500)

    tr_mse = float(np.mean((model.predict(X_tr) - y_tr) ** 2))
    va_mse = float(np.mean((model.predict(X_va) - y_va) ** 2))
    print(f"训练 MSE = {tr_mse:.4f}   验证 MSE = {va_mse:.4f}")
    print("学到的 w =", np.round(model.w, 3), "（真值在标准化空间）")

    # 继承演示
    ds = CsvDataset(X_va, columns=["x0", "x1"], batch_size=64)
    n_batches = 0
    for batch in ds.iter_batches(shuffle=False):
        n_batches += 1
        assert batch.ndim == 2
    print(f"[CsvDataset] 长度 {len(ds)}，按 64 一批共 {n_batches} 个 batch（继承 BaseDataset）")

    # ---- 图：左=loss 收敛曲线，右=训练 vs 验证 MSE 对比 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(model.loss_history, color="#4C72B0", linewidth=1.6)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("LinearModel 收敛曲线（全批量梯度下降）")
    axes[0].grid(alpha=0.3)

    axes[1].bar(["训练集 MSE", "验证集 MSE"], [tr_mse, va_mse],
                color=["#55A868", "#C44E52"])
    for i, v in enumerate([tr_mse, va_mse]):
        axes[1].text(i, v * 1.02, f"{v:.4f}", ha="center", fontsize=10)
    axes[1].set_title("泛化检验：训练与验证集误差接近 = 无过拟合")
    fig.tight_layout()
    out = os.path.join(HERE, "oop_curve.png")
    fig.savefig(out, dpi=120)
    print("已保存:", out)


if __name__ == "__main__":
    main()
