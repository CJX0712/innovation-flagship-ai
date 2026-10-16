"""INV: ``ModelConfig.activation`` 真正生效 — 补齐 gelu 激活 (训练端 + 推理端)。

历史缺陷: ``activation`` 仅支持 "swiglu" | "gelu"，但推理端 (dense + MoE FFN)
与训练端 FFN 都硬编码了 SwiGLU/silu，完全忽略 "gelu" 选项 —— 字段名不副实。
本文件三项硬校验防止回归:

1. ``gelu`` 数值对照 Hendrycks & Gimpel 2016 的 tanh-approx 公式 (pointwise < 1e-9);
2. 训练端 ``TinyTrainer.forward`` 与推理端 ``Transformer`` 在 ``{"swiglu","gelu"}``
   两种 activation 下 max|diff| ≤ 1e-9 (swiglu 必须仍 ≤ 1e-9, 证明默认路径未变);
3. ``Tensor.gelu`` 的解析梯度通过中心差分有限差分校验 (rel < 1e-5, 小梯度点 abs < 1e-8)。
"""
from __future__ import annotations

import numpy as np
import pytest

from ifai import rng
from ifai.core.ops import gelu as ops_gelu
from ifai.models.config import ModelConfig
from ifai.models.transformer import Transformer
from ifai.train.autodiff import Tensor
from ifai.train.trainer import TinyTrainer


def _ref_gelu(x: np.ndarray) -> np.ndarray:
    """Hendrycks & Gimpel 2016 tanh-approx GELU 参考实现 (黄金标准)。"""
    x = np.asarray(x, dtype=np.float64)
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


# ---------------- 1. gelu 公式数值对照 ----------------

def test_gelu_formula():
    """ops.gelu 与 Hendrycks & Gimpel 2016 公式 pointwise 误差 < 1e-9。"""
    rng.seed(20240101)
    samples = rng.get_rng().standard_normal((1000,)) * 4.0   # 宽范围覆盖正/负/近零
    got = ops_gelu(samples)
    ref = _ref_gelu(samples)
    err = float(np.max(np.abs(got - ref)))
    assert err < 1e-9, f"gelu 与参考公式偏差过大: max|err|={err:.3e}"


# ---------------- 2. 训练端 / 推理端 activation 一致 ----------------

@pytest.mark.parametrize("activation", ["swiglu", "gelu"])
def test_train_infer_activation_match(tmp_path, activation):
    """TinyTrainer.forward 与 Transformer (加载同参数) 在两种 activation 下
    输出 max|diff| ≤ 1e-9。``swiglu`` 配置必须仍 ≤ 1e-9 (回归, 默认路径未变)。"""
    rng.seed(123 + (0 if activation == "swiglu" else 7))
    cfg = ModelConfig(
        vocab_size=48, d_model=16, n_layers=2, n_heads=4,
        n_kv_heads=2, ffn_mult=2, max_seq_len=32, activation=activation,
    )
    tr = TinyTrainer(cfg)
    ids = rng.get_rng().integers(0, cfg.vocab_size, size=(2, 8))
    out_trainer = np.array(tr.forward(ids).data)

    model = Transformer.load(tr.save(str(tmp_path / "m")), verify=True)
    out_infer = model.forward(ids)
    diff = float(np.max(np.abs(out_trainer - out_infer)))
    assert diff <= 1e-9, (
        f"activation={activation} 训练/推理激活函数分叉: max|diff|={diff:.3e}")


# ---------------- 3. Tensor.gelu 梯度校验 ----------------

def test_gelu_gradcheck():
    """Tensor.gelu 的反向解析梯度对照中心差分 (eps=1e-5)。

    判据: 相对误差 < 1e-5；对梯度接近 0 的点放宽到绝对误差 < 1e-8。
    """
    rng.seed(99)
    x_np = rng.get_rng().standard_normal((4, 5)) * 1.5
    x = Tensor(x_np, requires_grad=True)
    y = x.gelu().sum()
    y.backward()
    analytic = np.array(x.grad, dtype=np.float64)

    eps = 1e-5
    numeric = np.zeros_like(x_np)
    for i in range(x_np.shape[0]):
        for j in range(x_np.shape[1]):
            xp = x_np.copy(); xp[i, j] += eps
            xm = x_np.copy(); xm[i, j] -= eps
            numeric[i, j] = (_ref_gelu(xp).sum() - _ref_gelu(xm).sum()) / (2.0 * eps)

    diff = np.abs(analytic - numeric)
    # 小梯度点 (|numeric| < 1e-4) 用绝对误差 1e-8 放行, 否则用相对误差 1e-5
    tol = np.where(np.abs(numeric) < 1e-4, 1e-8, 1e-5 * np.abs(numeric))
    violate = diff > tol
    assert not np.any(violate), (
        f"gelu 梯度校验失败: max|diff|={diff.max():.3e}, "
        f"最大点 analytic={analytic[diff==diff.max()][0]:.3e} "
        f"numeric={numeric[diff==diff.max()][0]:.3e}")
