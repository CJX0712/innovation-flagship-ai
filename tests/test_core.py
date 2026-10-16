"""L0/L1 内核层黄金测试 (每条测试标注其服务的可验证不变量 INV)。

INV 映射: 01 分块GEMM / 02 在线softmax注意力 / 03+04 RoPE / 05 SwiGLU /
06 RMSNorm / 07 量化误差界 / 08 logsumexp / 09 softmax 平移不变。
"""
import numpy as np
import pytest
from ifai.core import ops, numeric, quant
from ifai import rng

rng.seed(0)


def test_logsumexp_vs_naive():
    """INV-08: logsumexp ≡ 朴素参考; 且 x 含 ±1e4 时不溢出 (平移参考对照)。"""
    x = rng.get_rng().standard_normal((50, 7))
    mine = numeric.logsumexp(x, axis=-1)
    ref = np.log(np.sum(np.exp(x), axis=-1))
    assert np.allclose(mine, ref, atol=1e-6)
    # 溢出稳定性: ±1e4 量级输入, 朴素公式会溢出, 稳定版必须有限
    # 且与"平移后的朴素参考"一致 (数学恒等: log Σexp(x) ≡ m + log Σexp(x-m))
    big = np.array([[1e4, 1e4 - 1.0, -1e4, 0.0]])
    mine_big = numeric.logsumexp(big, axis=-1)
    m = big.max(axis=-1, keepdims=True)
    ref_big = m.squeeze(-1) + np.log(np.sum(np.exp(big - m), axis=-1))
    assert np.all(np.isfinite(mine_big)), "logsumexp 在 ±1e4 输入下溢出"
    assert np.allclose(mine_big, ref_big, atol=1e-6)


def test_softmax_shift_invariant():
    """INV-09: softmax 平移不变 softmax(x) == softmax(x + c)。"""
    x = rng.get_rng().standard_normal((10, 5))
    assert np.allclose(numeric.stable_softmax(x), numeric.stable_softmax(x + 3.0), atol=1e-9)


def test_tiled_matmul_vs_naive():
    """INV-01: 分块 GEMM ≡ 朴素矩阵乘 (diff ≤ 1e-5)。"""
    a = rng.get_rng().standard_normal((16, 24))
    b = rng.get_rng().standard_normal((24, 12))
    assert np.allclose(ops.tiled_matmul(a, b), a @ b, atol=1e-5)


def test_attention_flash_vs_naive():
    """INV-02: 在线 softmax 分块注意力 ≡ 朴素 softmax 注意力 (diff ≤ 1e-5)。"""
    q = rng.get_rng().standard_normal((6, 8))
    k = rng.get_rng().standard_normal((6, 8))
    v = rng.get_rng().standard_normal((6, 8))
    scale = 1.0 / 8 ** 0.5
    out1 = ops.attention_flash(q, k, v, scale, causal=True)
    out2 = ops.attention_naive(q, k, v, scale, causal=True)
    assert np.allclose(out1, out2, atol=1e-5)


def test_rmsnorm_norm():
    """INV-06: RMSNorm 归一后 ‖·‖₂ ≈ √d。"""
    x = rng.get_rng().standard_normal((4, 16))
    w = np.ones(16)
    y = ops.rms_norm(x, w)
    # 归一后每个向量的 RMS≈1
    rms = np.sqrt(np.mean(y ** 2, axis=-1))
    assert np.allclose(rms, 1.0, atol=1e-6)


def test_rope_vs_analytic():
    """INV-03: RoPE 旋转 ≡ 解析 2D 旋转矩阵 (θ_i = 10000^(-2i/d))。"""
    d = 8
    x = rng.get_rng().standard_normal(d)
    pos = np.array([3.0])
    rot = ops.apply_rope(x[None], pos, d)[0]
    theta = pos[0] / (10000 ** (np.arange(0, d // 2) / (d // 2)))
    ang = np.arctan2(x[1::2], x[0::2])
    new_ang = ang + theta
    r = np.hypot(x[0::2], x[1::2])
    expect = np.empty(d)
    expect[0::2] = r * np.cos(new_ang)
    expect[1::2] = r * np.sin(new_ang)
    assert np.allclose(rot, expect, atol=1e-6)


def test_inv_04_rope_relative_shift_invariance():
    """INV-04: RoPE 相对性 — ⟨R_m q, R_n k⟩ 仅依赖 (m−n)。

    平移 (m,n) → (m+c, n+c) 后内积 diff ≤ 1e-6 (PRD 判据)。
    """
    d = 8
    q = rng.get_rng().standard_normal(d)
    k = rng.get_rng().standard_normal(d)
    for m, n in [(3.0, 7.0), (11.0, 2.0)]:
        base = float(ops.apply_rope(q[None], np.array([m]), d)[0]
                     @ ops.apply_rope(k[None], np.array([n]), d)[0])
        for c in (5.0, 100.0):
            shifted = float(ops.apply_rope(q[None], np.array([m + c]), d)[0]
                            @ ops.apply_rope(k[None], np.array([n + c]), d)[0])
            assert abs(shifted - base) <= 1e-6, (
                f"平移 c={c}: 内积 {base} → {shifted} (diff {abs(shifted-base):.2e} > 1e-6)")


def test_swiglu_vs_split():
    """INV-05: SwiGLU 融合 ≡ 分离计算 (diff ≤ 1e-6)。"""
    x = rng.get_rng().standard_normal((3, 16))
    w1 = rng.get_rng().standard_normal((16, 16))
    w2 = rng.get_rng().standard_normal((16, 16))
    fused = ops.swiglu(x, w1, w2)
    g = x @ w1.T
    u = x @ w2.T
    split = (g / (1 + np.exp(-g))) * u
    assert np.allclose(fused, split, atol=1e-6)


def test_int8_quant_error_bound():
    """INV-07: 量化-反量化误差有界 |x − dequant(quant(x))| ≤ scale/2 + 1e-6。"""
    x = rng.get_rng().standard_normal((20, 30)) * 4
    q, scale = quant.quantize_int8_per_channel(x)
    xr = quant.dequantize_int8(q, scale)
    assert np.all(np.abs(x - xr) <= scale / 2 + 1e-6)


def test_int4_pack_roundtrip():
    """INT4 打包/解包往返无损 (INV-07 的整数量化路径补充)。"""
    x = rng.get_rng().integers(-8, 8, size=100).astype(np.int8)
    packed = quant.pack_int4(x)
    back = quant.unpack_int4(packed, len(x))
    assert np.array_equal(back, x)
