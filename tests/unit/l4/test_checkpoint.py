"""INV-25 (L4) 梯度检查点: loss/grad ≡ 无检查点 (``diff ≤ 1e-5``)。

开启检查点后, 每层前向的层内激活被丢弃, 反向传播时按层重算。由于被检查点的
段是确定性函数, 重算的前向与原始前向逐位相同, 故 loss 与全部梯度必须一致。
"""
from __future__ import annotations

import numpy as np

from ifai import rng
from ifai.models.config import ModelConfig
from ifai.train.autodiff import Tensor
from ifai.train.checkpoint import (
    checkpoint,
    checkpoint_segments,
    checkpointed,
    grad_diff,
    verify_checkpoint,
)
from ifai.train.trainer import TinyTrainer

rng.seed(25)

TOL = 1e-5


def _trainer(n_layers: int = 3) -> TinyTrainer:
    cfg = ModelConfig(vocab_size=40, d_model=32, n_layers=n_layers, n_heads=4,
                      n_kv_heads=2, ffn_mult=2, max_seq_len=32)
    return TinyTrainer(cfg)


def _batch(vocab: int = 40, b: int = 2, t: int = 8):
    xb = rng.get_rng().integers(0, vocab, size=(b, t))
    yb = rng.get_rng().integers(0, vocab, size=(b, t))
    return xb, yb


def _zero_grad(tr: TinyTrainer) -> None:
    for p in tr.params.values():
        if p.grad is not None:
            p.grad = np.zeros_like(p.data)


def test_gradient_checkpoint_loss_and_grad_identical():
    """INV-25 主判据: 开/关检查点的 loss 与梯度差 ≤ 1e-5。"""
    tr = _trainer(n_layers=3)
    xb, yb = _batch()

    def loss_fn(enabled: bool) -> Tensor:
        tr.enable_gradient_checkpointing(enabled)
        return tr.loss(xb, yb)

    def params_fn():
        _zero_grad(tr)
        return tr.params

    ok, rep = verify_checkpoint(loss_fn, params_fn, tol=TOL)
    assert ok, f"检查点与非检查点不一致: {rep}"
    assert rep["loss_diff"] <= TOL
    assert rep["grad_diff"] <= TOL


def test_gradient_checkpoint_multilayer_deeper_model():
    """更深模型 (6 层) 仍逐位等价。"""
    tr = _trainer(n_layers=6)
    xb, yb = _batch()

    def loss_fn(enabled: bool) -> Tensor:
        tr.enable_gradient_checkpointing(enabled)
        return tr.loss(xb, yb)

    def params_fn():
        _zero_grad(tr)
        return tr.params

    ok, rep = verify_checkpoint(loss_fn, params_fn, tol=TOL)
    assert ok, rep
    assert rep["grad_diff"] <= TOL


def test_gradient_checkpoint_grads_nonzero():
    """对照有效性: 梯度必须非全零 (否则"等价"是平凡成立的)。"""
    tr = _trainer(n_layers=2)
    xb, yb = _batch()
    _zero_grad(tr)
    tr.enable_gradient_checkpointing(True)
    loss = tr.loss(xb, yb)
    loss.backward()
    nz = sum(int(np.count_nonzero(p.grad)) for p in tr.params.values() if p.grad is not None)
    assert nz > 0, "检查点路径下梯度全为零, 无法证明等价性"
    assert float(loss.data) > 0.0


def test_checkpoint_returns_detached_leaf():
    """检查点输出是新叶子节点: 不持有原图的中间激活。"""
    x = Tensor(rng.get_rng().standard_normal((2, 3)), requires_grad=True)
    w = Tensor(rng.get_rng().standard_normal((3, 4)), requires_grad=True)

    def seg(t: Tensor) -> Tensor:
        h = t.matmul(w)
        return h * h

    y = checkpoint(seg, x)
    assert y._prev == [], "检查点输出不应保留计算图前驱 (激活已被丢弃)"
    assert np.allclose(y.data, seg(Tensor(x.data)).data, atol=1e-12)
    assert y.requires_grad is True


def test_checkpoint_backward_matches_plain_backward():
    """纯 autograd 对照: 检查点版梯度 == 直接版梯度。"""
    data = rng.get_rng().standard_normal((2, 3))
    wdata = rng.get_rng().standard_normal((3, 4))

    def build(enabled: bool):
        x = Tensor(data.copy(), requires_grad=True)
        w = Tensor(wdata.copy(), requires_grad=True)

        def seg(t: Tensor) -> Tensor:
            h = t.matmul(w)
            return (h * h).sum()

        out = checkpoint(seg, x, enabled=enabled)
        out.backward()
        return float(out.data), x, w

    loss_a, xa, wa = build(False)
    loss_b, xb, wb = build(True)
    assert abs(loss_a - loss_b) <= TOL
    assert float(np.max(np.abs(xa.grad - xb.grad))) <= TOL
    assert float(np.max(np.abs(wa.grad - wb.grad))) <= TOL


def test_checkpoint_disabled_is_passthrough():
    """enabled=False 时退化为直接调用 (用于对照实验)。"""
    x = Tensor(rng.get_rng().standard_normal((2, 2)), requires_grad=True)
    y = checkpoint(lambda t: t * 2.0, x, enabled=False)
    assert y is not None
    assert np.allclose(y.data, x.data * 2.0, atol=1e-12)


def test_checkpoint_decorator_matches_functional():
    """装饰器版本与函数式版本数值一致。"""
    w = Tensor(rng.get_rng().standard_normal((3, 3)), requires_grad=True)

    @checkpointed
    def block(t: Tensor) -> Tensor:
        return (t.matmul(w)).tanh()

    x = Tensor(rng.get_rng().standard_normal((2, 3)), requires_grad=True)
    y = block(x)
    assert np.allclose(y.data, np.tanh(x.data @ w.data), atol=1e-12)


def test_grad_diff_detects_mismatch():
    """grad_diff 应能识别不一致的梯度 (判据有判别力)。"""
    a = Tensor(np.ones((2, 2)), requires_grad=True)
    b = Tensor(np.ones((2, 2)), requires_grad=True)
    a.grad = np.ones((2, 2))
    b.grad = np.ones((2, 2)) * 2
    assert grad_diff({"p": a}, {"p": b}) >= 1.0
    assert grad_diff({"p": a}, {"p": a}) == 0.0


def test_checkpoint_segments_cover_all_items():
    """分段策略必须覆盖全部 n 项 (不丢层)。"""
    segs = checkpoint_segments(10, 3)
    assert segs[0][0] == 0
    assert segs[-1][1] == 10
    covered = sum(e - s for s, e in segs)
    assert covered == 10
    assert checkpoint_segments(0, 4) == []

    # 逐段检查点 (每段一层) 时段落数 == 层数
    assert len(checkpoint_segments(6, 6)) == 6
