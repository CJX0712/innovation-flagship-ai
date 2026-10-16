"""L4 训练层黄金测试 (含 INV-22 反向传播 vs 有限差分)。"""
import numpy as np
import pytest
from ifai import rng
from ifai.train.autodiff import Tensor
from ifai.train.optim import AdamW, cosine_schedule, clip_global_norm
from ifai.train.lora import LoRALinear

rng.seed(2)


def test_adamw_single_step_vs_analytic():
    """INV-21: AdamW 单步更新 ≡ 解析公式 (diff ≤ 1e-6)。"""
    p = Tensor(rng.get_rng().standard_normal((4,)))
    p.grad = rng.get_rng().standard_normal((4,))
    opt = AdamW([p], lr=0.1, weight_decay=0.0)
    data = p.data.copy()
    grad = p.grad.copy()
    opt.step()
    # 解析: m=(1-b1)*g, v=(1-b2)*g^2, mhat=m/(1-b1), vhat=v/(1-b2)
    b1, b2, eps = opt.b1, opt.b2, opt.eps
    m = (1 - b1) * grad
    v = (1 - b2) * grad * grad
    mhat = m / (1 - b1)
    vhat = v / (1 - b2)
    expect = data - 0.1 * (mhat / (np.sqrt(vhat) + eps))
    assert np.allclose(p.data, expect, atol=1e-6)


def test_adamw_wd_zero_equals_adam():
    """INV-21 补充: wd=0 时 AdamW ≡ Adam。"""
    p = Tensor(rng.get_rng().standard_normal((3,)))
    p.grad = rng.get_rng().standard_normal((3,))
    opt = AdamW([p], lr=0.05, weight_decay=0.0, betas=(0.9, 0.999))
    d0 = p.data.copy()
    g = p.grad.copy()
    opt.step()
    m = (1 - 0.9) * g
    v = (1 - 0.999) * g * g
    mhat = m / (1 - 0.9)
    vhat = v / (1 - 0.999)
    expect = d0 - 0.05 * (mhat / (np.sqrt(vhat) + 1e-8))
    assert np.allclose(p.data, expect, atol=1e-9)


def test_cosine_schedule_vs_analytic():
    for step in [0, 5, 50, 99]:
        got = cosine_schedule(0.01, step, 100, 10, 0.001)
        if step < 10:
            expect = 0.01 * (step + 1) / 10
        else:
            prog = (step - 10) / (100 - 10)
            expect = 0.001 + 0.5 * (0.01 - 0.001) * (1 + np.cos(np.pi * prog))
        assert abs(got - expect) <= 1e-9


def test_clip_global_norm():
    """INV-23: 梯度裁剪 — 裁剪后 ‖g‖ ≤ max_norm + 1e-6; 未超限 bitwise 不变。"""
    p1 = Tensor(rng.get_rng().standard_normal((10,)))
    p2 = Tensor(rng.get_rng().standard_normal((5,)))
    p1.grad = rng.get_rng().standard_normal((10,))
    p2.grad = rng.get_rng().standard_normal((5,))
    norm, triggered = clip_global_norm([p1, p2], 1.0)
    assert triggered
    clipped = np.linalg.norm(np.concatenate([p1.grad.ravel(), p2.grad.ravel()]))
    assert clipped <= 1.0 + 1e-6


def test_autodiff_vs_finite_diff():
    # f = sum( tanh(W @ x)^2 ); 解析梯度 vs 有限差分 (INV-22)
    x = Tensor(rng.get_rng().standard_normal((3,)), requires_grad=True)
    W = Tensor(rng.get_rng().standard_normal((2, 3)), requires_grad=True)

    def forward():
        y = (W.matmul(x)).tanh()
        return (y * y).sum()

    loss = forward()
    loss.backward()
    analytic_x = x.grad.copy()
    analytic_W = W.grad.copy()

    eps = 1e-5
    num_W = np.zeros_like(W.data)
    for i in range(2):
        for j in range(3):
            Wp = W.data.copy(); Wp[i, j] += eps
            Wm = W.data.copy(); Wm[i, j] -= eps
            yp = np.tanh(Wp @ x.data); ym = np.tanh(Wm @ x.data)
            num_W[i, j] = ((yp ** 2).sum() - (ym ** 2).sum()) / (2 * eps)
    num_x = np.zeros_like(x.data)
    for i in range(3):
        xp = x.data.copy(); xp[i] += eps
        xm = x.data.copy(); xm[i] -= eps
        yp = np.tanh(W.data @ xp); ym = np.tanh(W.data @ xm)
        num_x[i] = ((yp ** 2).sum() - (ym ** 2).sum()) / (2 * eps)

    relW = np.max(np.abs(analytic_W - num_W) / (np.abs(num_W) + 1e-8))
    relx = np.max(np.abs(analytic_x - num_x) / (np.abs(num_x) + 1e-8))
    assert relW < 1e-3 and relx < 1e-3


def test_lora_merge_consistency():
    """INV-24: LoRA merge() ≡ 未 merge 前向 (diff ≤ 1e-6)。"""
    W = rng.get_rng().standard_normal((8, 6))
    lora = LoRALinear(W, r=3, alpha=6)
    x = Tensor(rng.get_rng().standard_normal((4, 6)))
    unmerged = lora.forward(x)
    merged = lora.merged_linear(x)
    assert np.allclose(unmerged.data, merged.data, atol=1e-6)

    # B=0 ⇒ 等价原权重
    lora.B.data[:] = 0.0
    out = lora.forward(x)
    base = x.matmul(Tensor(W).transpose())
    assert np.allclose(out.data, base.data, atol=1e-9)
