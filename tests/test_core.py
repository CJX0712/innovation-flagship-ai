"""L0/L1 内核层黄金测试。"""
import numpy as np
import pytest
from ifai.core import ops, numeric, quant
from ifai import rng

rng.seed(0)


def test_logsumexp_vs_naive():
    x = rng.get_rng().standard_normal((50, 7))
    mine = numeric.logsumexp(x, axis=-1)
    ref = np.log(np.sum(np.exp(x), axis=-1))
    assert np.allclose(mine, ref, atol=1e-6)


def test_softmax_shift_invariant():
    x = rng.get_rng().standard_normal((10, 5))
    assert np.allclose(numeric.stable_softmax(x), numeric.stable_softmax(x + 3.0), atol=1e-9)


def test_tiled_matmul_vs_naive():
    a = rng.get_rng().standard_normal((16, 24))
    b = rng.get_rng().standard_normal((24, 12))
    assert np.allclose(ops.tiled_matmul(a, b), a @ b, atol=1e-5)


def test_attention_flash_vs_naive():
    q = rng.get_rng().standard_normal((6, 8))
    k = rng.get_rng().standard_normal((6, 8))
    v = rng.get_rng().standard_normal((6, 8))
    scale = 1.0 / 8 ** 0.5
    out1 = ops.attention_flash(q, k, v, scale, causal=True)
    out2 = ops.attention_naive(q, k, v, scale, causal=True)
    assert np.allclose(out1, out2, atol=1e-5)


def test_rmsnorm_norm():
    x = rng.get_rng().standard_normal((4, 16))
    w = np.ones(16)
    y = ops.rms_norm(x, w)
    # 归一后每个向量的 RMS≈1
    rms = np.sqrt(np.mean(y ** 2, axis=-1))
    assert np.allclose(rms, 1.0, atol=1e-6)


def test_rope_vs_analytic():
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


def test_swiglu_vs_split():
    x = rng.get_rng().standard_normal((3, 16))
    w1 = rng.get_rng().standard_normal((16, 16))
    w2 = rng.get_rng().standard_normal((16, 16))
    fused = ops.swiglu(x, w1, w2)
    g = x @ w1.T
    u = x @ w2.T
    split = (g / (1 + np.exp(-g))) * u
    assert np.allclose(fused, split, atol=1e-6)


def test_int8_quant_error_bound():
    x = rng.get_rng().standard_normal((20, 30)) * 4
    q, scale = quant.quantize_int8_per_channel(x)
    xr = quant.dequantize_int8(q, scale)
    assert np.all(np.abs(x - xr) <= scale / 2 + 1e-6)


def test_int4_pack_roundtrip():
    x = rng.get_rng().integers(-8, 8, size=100).astype(np.int8)
    packed = quant.pack_int4(x)
    back = quant.unpack_int4(packed, len(x))
    assert np.array_equal(back, x)
