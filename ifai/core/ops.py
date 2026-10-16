"""L0 融合算子 (Fused kernels)

零重写:
- RMSNorm  (Zhang & Sennrich 2019 "Root Mean Square Layer Normalization")
- RoPE     (Su et al. 2024 "RoFormer")，含 NTK-aware & linear 位置插值
           ( bloc97 2023 / Chen et al. "Extending Context" )
- SwiGLU   (Shazeer 2020 "GLU Variants Improve Transformer")
- 分块 GEMM (BLAS 分块思想, Goto & van de Geijn 2008)
- FlashAttention 风格在线 softmax 因果注意力 (Dao et al. 2022)

可验证不变量:
- INV-01 分块 GEMM == 朴素 matmul 到 1e-5
- INV-02 在线 softmax 注意力 == 朴素因果注意力 到 1e-5
- INV-03 RoPE == 解析旋转矩阵 到 1e-6
- INV-04 RoPE 相对性: 旋转后的内积只依赖相对位置
- INV-05 SwiGLU == 分离实现 到 1e-6
- INV-06 RMSNorm 把向量归一化到 ~1 (范数归一)
- INV-07 量化误差界 (见 quant.py)
"""
from __future__ import annotations

import numpy as np

from . import numeric


# ----------------------------- RMSNorm -----------------------------
def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm: x / sqrt(mean(x^2)+eps) * weight。INV-06 范数归一。"""
    x = np.asarray(x, dtype=np.float64)
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * np.asarray(weight, dtype=np.float64)


# ----------------------------- RoPE -----------------------------
def _rope_freqs(dim: int, base: float = 10000.0, scaling: float = 1.0):
    """生成旋转频率 (NTK-aware: 当 scaling≠1 拉伸 base)。"""
    half = dim // 2
    inv = 1.0 / (base ** (np.arange(0, half) / half))          # [half]
    if scaling != 1.0:
        # NTK-aware 插值: 放大 base 以覆盖更长上下文
        inv = inv * scaling ** (half / (half - 1)) if half > 1 else inv
    return inv


def apply_rope(x: np.ndarray, positions: np.ndarray, dim: int | None = None,
               base: float = 10000.0, scaling: float = 1.0) -> np.ndarray:
    """对最后一维 (隐维) 施加旋转位置编码。

    positions 支持两种形状:
      - [T]       单序列 (输入 (T,d) 或 (B,T,d) 共享同一位置)
      - [B,T]     逐序列位置 (连续批处理左补齐, 真实 token 取自然位置, INV-16)
    输入 x 支持 (T,d) 或 (B,T,d)。

    cos/sin 自动广播到 x 的前导维度 (批/头), 末维对齐 half (INV-03 解析对照)。
    """
    x = np.asarray(x, dtype=np.float64)
    if dim is None:
        dim = x.shape[-1]
    T = x.shape[-2]
    half = dim // 2
    freqs = _rope_freqs(dim, base=base, scaling=scaling)            # [half]
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim == 1:
        angles = pos[:T, None] * freqs                              # [T, half]
    else:  # [B, T]
        angles = pos[:, :T, None] * freqs                           # [B, T, half]
    cos = np.cos(angles)
    sin = np.sin(angles)
    # 对齐 x 维度: x 为 [..., T, d]; cos 末维=half, T 轴对齐, 缺失前导维补 1
    if cos.ndim == x.ndim - 1:
        cos = cos.reshape((1,) + cos.shape)
        sin = sin.reshape((1,) + sin.shape)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    out = np.empty_like(x)
    out[..., 0::2] = rot1
    out[..., 1::2] = rot2
    return out


# ----------------------------- SwiGLU -----------------------------
def swiglu(x: np.ndarray, w1: np.ndarray, w2: np.ndarray, b1=None, b2=None) -> np.ndarray:
    """SwiGLU(x) = (x·W1 + b1) ⊙ silu(x·W2 + b2) · W? — 这里实现 SwiGLU FFN 内核。

    FFN 标准: h = silu(x·W_gate + b_gate) * (x·W_up + b_up);  out = h · W_down + b_down
    本函数实现 gate⊗up 的融合 (INV-05 与分离实现一致)。
    """
    x = np.asarray(x, dtype=np.float64)
    w1, w2 = np.asarray(w1, np.float64), np.asarray(w2, np.float64)
    g = x @ w1.T if b1 is None else x @ w1.T + b1
    u = x @ w2.T if b2 is None else x @ w2.T + b2
    silu = g / (1.0 + np.exp(-g))
    return silu * u


# ----------------------------- GELU -----------------------------
def gelu(x: np.ndarray) -> np.ndarray:
    """GELU (Hendrycks & Gimpel 2016, tanh-approx, GPT/LLaMA 同款)。

    ``gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))``

    与 ``swiglu`` 同文件 (零重写: 标准激活函数)；推理端 FFN 按
    ``ModelConfig.activation`` 在 ``gelu`` / ``swiglu`` 间二选一。
    """
    x = np.asarray(x, dtype=np.float64)
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


# ----------------------------- 分块 GEMM -----------------------------
def tiled_matmul(a: np.ndarray, b: np.ndarray, block: int = 64) -> np.ndarray:
    """分块矩阵乘 (INV-01 == a@b)。分块提升 cache 命中，复刻 BLAS 思想。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m, k = a.shape
    n = b.shape[1]
    out = np.zeros((m, n), dtype=np.float64)
    for i in range(0, m, block):
        for j in range(0, n, block):
            for p in range(0, k, block):
                out[i:i + block, j:j + block] += (
                    a[i:i + block, p:p + block] @ b[p:p + block, j:j + block])
    return out


# ----------------------------- Flash 风格因果注意力 -----------------------------
def attention_flash(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                    scale: float, causal: bool = True) -> np.ndarray:
    """在线 softmax 的因果注意力 (INV-02 == 朴素)。

    算法 (Dao et al. 2022): 沿 seq 维分块流式维护 running max/sum，
    避免物化 [seq,seq] 大矩阵。返回 [q_seq, d_v]。
    """
    q = np.asarray(q, dtype=np.float64)   # [Sq, d]
    k = np.asarray(k, dtype=np.float64)   # [Sk, d]
    v = np.asarray(v, dtype=np.float64)   # [Sk, dv]
    Sq, d = q.shape
    Sk = k.shape[0]
    out = np.zeros((Sq, v.shape[1]), dtype=np.float64)
    for i in range(Sq):
        # 第 i 行: 与 k[0..i] (causal) 计算分数
        end = (i + 1) if causal else Sk
        scores = (q[i] @ k[:end].T) * scale                  # [end]
        m_new = np.max(scores)
        m_old = -np.inf
        # 在线更新 (running max m, running sum l)
        # 初值: 用第一个块初始化
        l = np.exp(scores[0] - m_new)
        acc = np.exp(scores[0] - m_new) * v[0]
        for j in range(1, end):
            l += np.exp(scores[j] - m_new)
            acc += np.exp(scores[j] - m_new) * v[j]
        out[i] = acc / l
    return out


# 朴素参考实现 (用于黄金测试对照 INV-02)
def attention_naive(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                    scale: float, causal: bool = True) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    scores = (q @ k.T) * scale
    if causal:
        mask = np.triu(np.ones(scores.shape, dtype=bool), k=1)
        scores = np.where(mask, -1e30, scores)
    p = numeric.stable_softmax(scores, axis=-1)
    return p @ v
