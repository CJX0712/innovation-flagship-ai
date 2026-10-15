"""示例公共工具: 训练一个微型模型到临时目录, 供各示例复用。

零重写: 复用 ``ifai`` 训练管线 (TinyTrainer, 参考 Karpathy nanogpt 极简循环)。
"""
from __future__ import annotations

import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ifai import rng  # noqa: E402
from ifai.models.config import ModelConfig  # noqa: E402
from ifai.models.tokenizer import Tokenizer  # noqa: E402
from ifai.train.trainer import TinyTrainer  # noqa: E402

CORPUS = (
    "人工智能是计算机科学的一个分支。模型通过学习数据来提升能力。"
    "变压器网络由注意力机制构成。注意力让模型关注重要的信息。"
    "训练需要大量的数据和计算资源。推理是把训练好的模型用于预测。"
    "分词把文本切分成更小的单元。嵌入把符号映射到向量空间。"
    "采样从模型的输出分布中选择下一个词。检索从知识库中找到相关资料。"
    "智能体可以调用工具来完成任务。记忆帮助智能体积累经验。"
    "量化可以降低模型的内存占用。低秩适配能以很少的参数微调大模型。"
)


def train_tiny(
    out_dir: str | None = None,
    vocab: int = 256,
    d_model: int = 48,
    n_layers: int = 2,
    n_heads: int = 4,
    n_kv_heads: int = 2,
    ffn_mult: int = 4,
    seq_len: int = 96,
    steps: int = 50,
    lr: float = 3e-3,
    seed: int = 0,
) -> str:
    """训练一个微型模型, 返回模型目录 (含 params.npz / manifest.json / tokenizer.json)。"""
    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="ifai_tiny_")
    os.makedirs(out_dir, exist_ok=True)
    rng.seed(seed)
    cfg = ModelConfig(
        vocab_size=vocab,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        ffn_mult=ffn_mult,
        max_seq_len=seq_len,
    )
    tok = Tokenizer(vocab_size=vocab).train(CORPUS)
    tok.to_json(os.path.join(out_dir, "tokenizer.json"))
    trainer = TinyTrainer(cfg)
    trainer.train(tok.encode(CORPUS), steps=steps, lr=lr)
    trainer.save(out_dir)
    return out_dir
