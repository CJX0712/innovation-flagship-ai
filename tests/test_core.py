"""L0/L1 内核层黄金测试 (每条测试标注其服务的可验证不变量 INV)。

INV 映射: 01 分块GEMM / 02 在线softmax注意力 / 03+04 RoPE / 05 SwiGLU /
06 RMSNorm / 07 量化误差界 / 08 logsumexp / 09 softmax 平移不变。
"""
import numpy as np
import pytest
from ifai.core import ops, numeric, quant
from ifai import rng
from ifai.models.config import ModelConfig
from ifai.models.transformer import Transformer

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
    # 回归: batch=1 沿 axis 约简后必须保持 (1,) 形状 (不得压成 0 维标量)
    one = np.array([[1.0, 2.0, 3.0]])
    out1 = numeric.logsumexp(one, axis=1)
    assert out1.shape == (1,), f"batch=1 形状回归: {out1.shape}"
    assert out1[0] == pytest.approx(np.log(np.exp(one).sum()), abs=1e-12)
    assert numeric.logsumexp(one, axis=0).shape == (3,)
    assert numeric.logsumexp(one).shape == ()  # 全约简才允许标量


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


def test_int8_model_quantization_inference():
    """INT8 量化接入推理前向 (v0.7.6): 整模型 per-channel RTN 后
    1) INV-07 逐权重重建误差 ≤ scale/2 (真实模型权重, 非合成矩阵);
    2) int8 RTN 不翻转 top-1 决策 (前向一致性);
    3) 前向误差有限且受控 (经验上 ≤ 1.0, 主判据为 INV-07 + top-1)。
    零重写: LLM.int8 (Dettmers et al. 2022) per-channel 对称量化思想。
    """
    rng.seed(123)
    cfg = ModelConfig(
        vocab_size=64, d_model=16, n_layers=2, n_heads=4, n_kv_heads=2,
        ffn_mult=2, tie_word_embeddings=True,
    )
    model = Transformer(cfg)
    ids = rng.get_rng().integers(0, cfg.vocab_size, size=(2, 8))
    fp64_out = model.forward(ids)
    info = model.quantize_int8()
    q_out = model.forward(ids)
    # 1) INV-07 逐权重 (真实模型权重, 含 embed/q/k/v/o/ffn/路由/专家)
    for k, (scale, orig) in info.items():
        dq = model.params[k]
        assert np.all(np.abs(orig - dq) <= scale / 2 + 1e-9), (
            f"{k} 违反 INV-07 重建误差界")
    # 2) top-1 决策不变 (int8 RTN 不得翻转模型输出)
    assert np.array_equal(fp64_out.argmax(-1), q_out.argmax(-1)), (
        "INT8 RTN 翻转了模型 top-1 决策")
    # 3) 前向误差有限且受控 (经验界 0.05 ≫ 实测 0.0018, 主判据为 INV-07 + top-1)
    err = float(np.max(np.abs(fp64_out - q_out)))
    assert np.isfinite(err) and err <= 0.05, f"INT8 前向误差失控: {err:.3e}"


def test_int4_pack_roundtrip():
    """INT4 打包/解包往返无损 (INV-07 的整数量化路径补充)。"""
    x = rng.get_rng().integers(-8, 8, size=100).astype(np.int8)
    packed = quant.pack_int4(x)
    back = quant.unpack_int4(packed, len(x))
    assert np.array_equal(back, x)


def test_int4_quant_error_bound():
    """INV-07 (INT4 口径): 对称 [-7,7] 15 电平 RTN, 误差 ≤ scale/2 + 1e-6。"""
    x = rng.get_rng().standard_normal((20, 30)) * 4
    q, scale = quant.quantize_int4_per_channel(x)
    assert (np.abs(q) <= 7).all(), "INT4 量化值越出 [-7,7]"
    xr = quant.dequantize_int4(q, scale)
    assert np.all(np.abs(x - xr) <= scale / 2 + 1e-6)


def test_int4_model_quantization_inference():
    """INT4 量化接入推理前向 (v0.7.11): 整模型 per-channel INT4 RTN 后
    1) INV-07 逐权重重建误差 ≤ scale/2 (真实模型权重);
    2) top-1 翻转率受控 (INT4 粗于 INT8, 允许偶发翻转 ≤ 2/16; 实测 20 seed 最大 1);
    3) 前向误差有限且受控 (20 seed 实测 max 0.047 ⇒ 界 0.06)。
    补齐 README L0 承诺的 "int8/int4/fp16 量化" 中 int4 的模型级路径。
    """
    rng.seed(123)
    cfg = ModelConfig(
        vocab_size=64, d_model=16, n_layers=2, n_heads=4, n_kv_heads=2,
        ffn_mult=2, tie_word_embeddings=True,
    )
    model = Transformer(cfg)
    ids = rng.get_rng().integers(0, cfg.vocab_size, size=(2, 8))
    fp64_out = model.forward(ids)
    info = model.quantize_int4()
    q_out = model.forward(ids)
    # 1) INV-07 逐权重 (INT4 口径)
    for k, (scale, orig) in info.items():
        dq = model.params[k]
        assert np.all(np.abs(orig - dq) <= scale / 2 + 1e-9), (
            f"{k} 违反 INV-07 (INT4) 重建误差界")
    # 2) top-1 翻转率受控 (INT4 粗量化, 允许偶发翻转)
    flips = int((fp64_out.argmax(-1) != q_out.argmax(-1)).sum())
    assert flips <= 2, f"INT4 RTN top-1 翻转过多: {flips}/{fp64_out.argmax(-1).size}"
    # 3) 前向误差受控 (INT4 口径, 实测 max 0.047)
    err = float(np.max(np.abs(fp64_out - q_out)))
    assert np.isfinite(err) and err <= 0.06, f"INT4 前向误差失控: {err:.3e}"


def test_fp16_model_quantization_inference():
    """FP16 模型量化接入推理 (v0.7.13): 整模型半精度往返后
    1) 逐权重误差 |orig−dq| ≤ 1e-3·|orig| + 1e-7 (fp16 round-to-nearest:
       ulp/2 相对 ≈ 4.88e-4; 子正规 flush 绝对粒度 ~6e-8);
    2) top-1 决策不翻转;
    3) 前向误差 ≪ INT8 (实测 7.6e-5, 界 1e-3)。
    补齐 README L0 承诺 "int8/int4/fp16 量化" 的最后一档模型级路径
    (接线 L0 死代码 quantize_fp16)。
    """
    rng.seed(123)
    cfg = ModelConfig(
        vocab_size=64, d_model=16, n_layers=2, n_heads=4, n_kv_heads=2,
        ffn_mult=2, tie_word_embeddings=True,
    )
    model = Transformer(cfg)
    ids = rng.get_rng().integers(0, cfg.vocab_size, size=(2, 8))
    fp64_out = model.forward(ids)
    info = model.quantize_fp16()
    q_out = model.forward(ids)
    # 1) 逐权重半精度误差界
    for k, orig in info.items():
        dq = model.params[k]
        assert np.all(np.abs(orig - dq) <= 1e-3 * np.abs(orig) + 1e-7), (
            f"{k} 违反 FP16 误差界")
    # 2) top-1 决策不变
    assert np.array_equal(fp64_out.argmax(-1), q_out.argmax(-1)), (
        "FP16 往返翻转了模型 top-1 决策")
    # 3) 前向误差受控 (远优于 INT8/INT4)
    err = float(np.max(np.abs(fp64_out - q_out)))
    assert np.isfinite(err) and err <= 1e-3, f"FP16 前向误差失控: {err:.3e}"


def test_safe_cross_entropy_vs_manual():
    """safe_cross_entropy ≡ 手工 log-softmax NLL; 梯度 == p − onehot。"""
    logits = rng.get_rng().standard_normal((6, 5)) * 3
    targets = rng.get_rng().integers(0, 5, size=6)
    loss, grad = numeric.safe_cross_entropy(logits, targets)
    # 手工参考
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    ref = -logp[np.arange(6), targets].mean()
    assert abs(loss - ref) <= 1e-10
    p = np.exp(logp)
    assert np.allclose(grad, (p - np.eye(5)[targets]) / 6, atol=1e-12)


def test_safe_cross_entropy_extreme_logits_stable():
    """±1e4 量级 logits 下损失/梯度有限且正确 (数值稳定, INV-08 同源)。"""
    logits = np.array([[1e4, 0.0, -1e4], [0.0, 1e4, -1e4]])
    targets = np.array([0, 1])
    loss, grad = numeric.safe_cross_entropy(logits, targets)
    assert np.isfinite(loss) and np.all(np.isfinite(grad))
    assert abs(loss) <= 1e-6  # 目标类恰为最大 → 损失≈0
    assert abs(grad[0, 0]) <= 1e-9  # p≈1 → grad≈0


def test_safe_cross_entropy_ignore_and_smoothing():
    """ignore_index 行不计入损失且梯度为 0; label_smoothing 增大损失。"""
    logits = rng.get_rng().standard_normal((4, 6))
    targets = np.array([1, 2, -100, 4])
    loss, grad = numeric.safe_cross_entropy(logits, targets)
    assert np.all(grad[2] == 0.0), "被忽略行的梯度必须为 0"
    # 被忽略行不计入均值: 与 3 行有效子集的损失逐位一致
    sub = np.array([1, 2, 4])
    loss_sub, _ = numeric.safe_cross_entropy(logits[[0, 1, 3]], sub)
    assert abs(loss - loss_sub) <= 1e-12
    # label smoothing: 损失 == 手工平滑 q 的交叉熵 (不假设方向 —
    # 自信错误时平滑可能降低损失, 故直接对照公式)
    eps_ls = 0.1
    loss_ls, grad_ls = numeric.safe_cross_entropy(logits, targets, label_smoothing=eps_ls)
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    q = np.zeros((4, 6))
    q[np.arange(4), np.where(targets == -100, 0, targets)] = 1.0
    q = (1 - eps_ls) * q + eps_ls / 6
    mask = (targets != -100).astype(float)
    ref = float((-(q * logp).sum(axis=1) * mask).sum() / mask.sum())
    assert abs(loss_ls - ref) <= 1e-12
    assert np.allclose(grad_ls, (np.exp(logp) - q) * mask[:, None] / 3, atol=1e-12)
    assert np.all(np.isfinite(grad_ls))
    # 1D 输入
    l1, g1 = numeric.safe_cross_entropy(logits[0], np.array(targets[0]))
    assert np.isfinite(l1) and g1.shape == (1, 6)


def test_clip_by_norm_inv23():
    """INV-23 基元: clip_by_norm 超限缩放 / 未限 bitwise 不变 / L1 范数。"""
    g = np.array([3.0, 4.0])  # L2 范数 5
    clipped, flag = numeric.clip_by_norm(g, max_norm=2.5)
    assert flag is True
    assert abs(np.linalg.norm(clipped) - 2.5) <= 1e-12
    assert np.allclose(clipped, g * (2.5 / 5.0))
    same, flag2 = numeric.clip_by_norm(g, max_norm=10.0)
    assert flag2 is False and np.array_equal(same, g)  # 未超限 bitwise 不变
    g1 = np.array([1.0, -2.0, 3.0])  # L1 范数 6
    c1, flag3 = numeric.clip_by_norm(g1, max_norm=3.0, norm_type=1.0)
    assert flag3 is True and abs(np.abs(c1).sum() - 3.0) <= 1e-12
    # max_norm<=0 不触发裁剪 (守卫分支)
    c0, flag0 = numeric.clip_by_norm(g, max_norm=0.0)
    assert flag0 is False and np.array_equal(c0, g)
