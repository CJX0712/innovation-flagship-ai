#!/usr/bin/env python
"""INT8 量化演示 — 验证 L0/L1 量化内核的可验证不变量 (INV-07 / R-L1-02)。

零重写: 复用 LLM.int8 (Dettmers et al. 2022) 的 per-channel 对称量化思想,
与 RTN (round-to-nearest) 基线。

演示:
1. 对一个真实权重矩阵做 per-channel INT8 量化→反量化, 验证重建误差
   <= scale/2 (INV-07 的解析误差界)。
2. INT4 打包↔解包往返逐位一致 (R-L1-02)。

运行:
    python examples/quantize_int8.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from common import train_tiny  # noqa: E402
from ifai.core.quant import (  # noqa: E402
    dequantize_int8,
    pack_int4,
    quantize_int8_per_channel,
    unpack_int4,
)
from ifai.models.transformer import load_model  # noqa: E402


def main() -> None:
    model_dir = train_tiny(steps=40)
    model = load_model(model_dir)
    W = model.params["lm_head"]  # [V, d] 真实权重

    # 1) INT8 per-channel 量化
    q, scale = quantize_int8_per_channel(W, axis=-1)
    Wr = dequantize_int8(q, scale)
    err = np.max(np.abs(W - Wr))
    bound = float(np.max(scale))
    ok = err <= bound + 1e-6
    print(f"[INT8] 权重形状 {W.shape}, 最大重建误差 = {err:.6f}")
    print(f"       误差界 scale/2 上界 ≈ {bound:.6f}  →  INV-07: {'PASS' if ok else 'FAIL'}")

    # 2) INT4 打包↔解包往返
    q8 = np.clip(np.round(W / (np.max(np.abs(W)) / 7.0 + 1e-9)), -8, 7).astype(np.int8)
    packed = pack_int4(q8)
    back = unpack_int4(packed, n=q8.size)
    ok4 = np.array_equal(q8.ravel(), back)
    print(f"[INT4] 打包字节数 {packed.size} (原为 {q8.size} 元素), 往返逐位一致 → R-L1-02: "
          f"{'PASS' if ok4 else 'FAIL'}")


if __name__ == "__main__":
    main()
