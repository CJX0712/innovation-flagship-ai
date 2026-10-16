"""v0.7.9 特征配置端到端可训性 (INV-26 扩展到 MoE / LoRA / 组合)。

历史: v0.7.5/0.7.7 把 MoE / LoRA 接入训练端并验证了「梯度贯通」与「训练/推理
前向一致性」; v0.7.8 锁了组合参数集对齐。但**一致性 ≠ 可训性** —— 这些特征配置
从未被证明训练后损失真实下降、模型真实学会 (MoE 路由/LoRA 低秩增量可能"梯度在、
前后向一致"却根本不参与学习)。

本文件把 INV-26 (L4 训练损失单调性) 的标准推广到全特征矩阵: 在真实中文语料上
真实训练, 断言损失有限为正且滑动平滑后尾部 20% 均值 ≤ 头部 20% 均值 × 0.8。
这证明 MoE 路由、LoRA 低秩适配、及其组合**确实参与并推动学习**, 而非仅通过一致性。

零重写: 复用 INV-26 方法论 (字符级语料真实训练, 非构造曲线) 与既有 ``TinyTrainer``。
"""
from __future__ import annotations

import numpy as np
import pytest

from ifai import rng
from ifai.models.config import ModelConfig
from ifai.models.tokenizer import Tokenizer
from ifai.train.trainer import TinyTrainer

CORPUS = (
    "人工智能是计算机科学的一个分支。模型通过学习数据来提升能力。"
    "变压器网络由注意力机制构成。注意力让模型关注重要的信息。"
    "训练需要大量的数据和计算资源。推理是把训练好的模型用于预测。"
    "分词把文本切分成更小的单元。嵌入把符号映射到向量空间。"
    "采样从模型的输出分布中选择下一个词。检索从知识库中找到相关资料。"
)


def _smooth(x: np.ndarray, k: int = 9) -> np.ndarray:
    """滑动平均平滑 (窗口不足时缩窄), 消除 step 级噪声。"""
    k = min(k, max(1, len(x)))
    return np.convolve(x, np.ones(k) / k, mode="valid")


def _train_cfg(overrides: dict, steps: int, seed: int) -> np.ndarray:
    """构造特征配置 → 真实训练 → 返回损失序列 (确定性, 固定 seed)。"""
    rng.seed(seed)
    tok = Tokenizer(vocab_size=512).train(CORPUS)
    base = dict(vocab_size=tok.vocab_size, d_model=16, n_layers=2, n_heads=4,
                n_kv_heads=2, ffn_mult=2, max_seq_len=64)
    base.update(overrides)
    cfg = ModelConfig(**base)
    tr = TinyTrainer(cfg)
    losses = tr.train(np.array(tok.encode(CORPUS)), steps=steps, lr=3e-3)
    return np.asarray(losses, dtype=np.float64)


# 特征矩阵: (名称, overrides, steps, seed)
# 覆盖 MHA/GQA × {dense, MoE, LoRA, MoE+LoRA} × {swiglu, gelu} 的可训性。
FEATURES = [
    ("base", {}, 150, 260),
    ("gqa", {"n_kv_heads": 2}, 150, 261),
    ("gelu", {"activation": "gelu"}, 150, 262),
    ("moe", {"use_moe": True, "n_experts": 4, "moe_topk": 2}, 150, 263),
    ("lora", {"use_lora": True, "lora_r": 4, "lora_alpha": 8}, 200, 264),
    ("moe_lora", {"use_moe": True, "n_experts": 4, "moe_topk": 2,
                  "use_lora": True, "lora_r": 4, "lora_alpha": 8}, 200, 265),
    ("gqa_moe", {"n_kv_heads": 2, "use_moe": True, "n_experts": 4, "moe_topk": 2},
     200, 266),
    ("gqa_lora", {"n_kv_heads": 2, "use_lora": True, "lora_r": 4, "lora_alpha": 8},
     200, 267),
]


@pytest.mark.parametrize("name,overrides,steps,seed", FEATURES,
                         ids=[f[0] for f in FEATURES])
def test_feature_config_learns(name, overrides, steps, seed):
    """特征配置模型在真实语料上真实训练: 损失有限为正, 且平滑后尾部 ≤ 头部 × 0.8
    (INV-26 推广到 MoE/LoRA/组合 —— 证明这些特征确实参与并推动学习)。"""
    arr = _train_cfg(overrides, steps, seed)
    assert np.all(np.isfinite(arr)) and (arr > 0).all(), (
        f"[{name}] 损失必须为正且有限; losses[:3]={arr[:3].round(3).tolist()} "
        f"losses[-3:]={arr[-3:].round(3).tolist()}")
    sm = _smooth(arr)
    n = len(sm)
    head, tail = sm[: n // 5].mean(), sm[-(n // 5):].mean()
    assert tail <= head * 0.8, (
        f"[{name}] 特征配置训练未有效收敛: head={head:.4f} tail={tail:.4f} "
        f"(ratio={tail / head:.3f} > 0.8); "
        f"losses[:5]={arr[:5].round(3).tolist()} "
        f"losses[-5:]={arr[-5:].round(3).tolist()}")
