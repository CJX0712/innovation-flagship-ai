"""INV-26 (L4) 训练损失单调性: 滑动平滑后尾部 20% 均值 ≤ 头部 20% 均值 × 0.8。

以微型模型在字符级语料上真实训练 (非构造曲线), 验证训练循环确实在收敛。
"""
from __future__ import annotations

import numpy as np

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
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="valid")


def test_inv_26_loss_monotonic_decrease():
    """INV-26 主判据: 平滑后 tail20% 均值 ≤ head20% 均值 × 0.8。"""
    rng.seed(26)
    tok = Tokenizer(vocab_size=512).train(CORPUS)
    ids = tok.encode(CORPUS)
    # 模型词表必须 ≥ 分词器实际词表 (该实现 vocab_size 为软目标: 仅限 merge 数,
    # 基础字符数不设上限, 见 ifai/models/tokenizer.py::train)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=2, n_heads=4,
                      n_kv_heads=2, ffn_mult=2, max_seq_len=64)
    trainer = TinyTrainer(cfg)
    losses = trainer.train(np.array(ids), steps=150, lr=3e-3)

    arr = np.asarray(losses, dtype=np.float64)
    assert np.all(np.isfinite(arr)) and (arr > 0).all(), "损失必须为正且有限"
    sm = _smooth(arr)
    n = len(sm)
    head, tail = sm[: n // 5].mean(), sm[-(n // 5):].mean()
    assert tail <= head * 0.8, (
        f"训练未有效收敛: head={head:.4f} tail={tail:.4f} "
        f"(ratio={tail / head:.3f} > 0.8); losses[:5]={arr[:5].round(3).tolist()} "
        f"losses[-5:]={arr[-5:].round(3).tolist()}")


def test_inv_26_loss_finite_and_positive_short_run():
    """短程训练的损失同样有限为正 (守门: 防止 NaN 混入平滑窗口)。"""
    rng.seed(62)
    tok = Tokenizer(vocab_size=512).train(CORPUS)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=16, n_layers=1, n_heads=2,
                      n_kv_heads=2, ffn_mult=2, max_seq_len=64)
    trainer = TinyTrainer(cfg)
    losses = trainer.train(np.array(tok.encode(CORPUS)), steps=30, lr=3e-3)
    arr = np.asarray(losses, dtype=np.float64)
    assert np.all(np.isfinite(arr)) and (arr > 0).all()
