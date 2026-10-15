#!/usr/bin/env python
"""LoRA 低秩适配演示 — 验证 INV-24 (merge 等价) 并展示一次极简微调。

零重写: 复用 Hu et al. 2022 "LoRA: Low-Rank Adaptation of Large Language Models"。
W_eff = W + (alpha/r)·B·A。

演示:
1. merge (W + scale·B@A) 与未 merge 前向逐元素一致到 1e-6 (INV-24)。
2. B=0 时 == 冻结原权重 (INV-24)。
3. 可训练参数占比 ≪ 100% (通常 ≤5%)。
4. 一次基于手写 autodiff 的极简回归微调, 损失下降。

运行:
    python examples/lora_finetune.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from common import train_tiny  # noqa: E402
from ifai.models.transformer import load_model  # noqa: E402
from ifai.train.autodiff import Tensor  # noqa: E402
from ifai.train.lora import LoRALinear  # noqa: E402


def main() -> None:
    model_dir = train_tiny(steps=40)
    model = load_model(model_dir)
    # 选取一个真实权重矩阵 (逐层参数以 "layer{i}." 为前缀, 顶层含 lm_head)
    key = next((k for k in model.params if k.endswith("ffn_down")), "lm_head")
    W = model.params[key]  # [d_out, d_in] 真实权重
    d_out, d_in = W.shape
    r, alpha = 4, 8.0
    print(f"[LoRA] 适配权重: {key}  形状 {W.shape}")

    lora = LoRALinear(W, r=r, alpha=alpha)

    x = Tensor(np.random.default_rng(0).standard_normal((4, d_in)))

    # 1) merge 等价
    fwd = lora.forward(x).data
    merged = lora.merged_linear(x).data
    diff_merge = float(np.max(np.abs(fwd - merged)))
    print(f"[INV-24] merge 前向 vs 未 merge 最大差 = {diff_merge:.2e}  → "
          f"{'PASS' if diff_merge < 1e-6 else 'FAIL'}")

    # 2) B=0 等价 (冻结)
    lora.B.data = np.zeros_like(lora.B.data)
    base = x.matmul(Tensor(W).transpose()).data
    fwd_b0 = lora.forward(x).data
    diff_b0 = float(np.max(np.abs(fwd_b0 - base)))
    print(f"[INV-24] B=0 时 == 原权重 最大差 = {diff_b0:.2e}  → "
          f"{'PASS' if diff_b0 < 1e-6 else 'FAIL'}")

    # 3) 可训练参数占比
    n_train = r * d_in + d_out * r
    n_total = d_out * d_in
    print(f"[LoRA] 可训练参数 {n_train} / 原 {n_total} = {n_train / n_total * 100:.2f}%")

    # 4) 极简微调: 让 LoRA 拟合一个固定的低秩扰动目标, 验证损失下降
    rng2 = np.random.default_rng(1)
    U = rng2.standard_normal((d_out, r))
    V = rng2.standard_normal((r, d_in))
    Delta = U @ V  # 固定低秩目标扰动
    target = x.matmul(Tensor(W).transpose()) + x.matmul(Tensor(Delta).transpose())
    lora2 = LoRALinear(W, r=r, alpha=alpha)
    lr = 0.05
    losses = []
    for step in range(60):
        lora2.A.grad = np.zeros_like(lora2.A.data)
        lora2.B.grad = np.zeros_like(lora2.B.data)
        out = lora2.forward(x)
        diff = out - target
        loss = (diff * diff).mean()
        loss.backward()
        for p in lora2.trainable_params:
            p.data = p.data - lr * p.grad
        losses.append(float(loss.data))
    print(f"[finetune] 损失 {losses[0]:.4f} → {losses[-1]:.4f} (下降={'是' if losses[-1] <= losses[0] else '否'})")


if __name__ == "__main__":
    main()
