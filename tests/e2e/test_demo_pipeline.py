"""端到端演示管线 (R-L9-02 / G1): 训练微型模型 → SDK chat/search/run。

覆盖 L9 SDK 三件套在真实 (微型) 模型上的端到端可用性。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
from ifai import rng  # noqa: E402
from ifai.models.config import ModelConfig  # noqa: E402
from ifai.models.tokenizer import Tokenizer  # noqa: E402
from ifai.models.transformer import load_model  # noqa: E402
from ifai.sdk import IFAI  # noqa: E402
from ifai.train.trainer import TinyTrainer  # noqa: E402

CORPUS = (
    "人工智能是计算机科学的一个分支。模型通过学习数据来提升能力。"
    "变压器网络由注意力机制构成。注意力让模型关注重要的信息。"
    "训练需要大量的数据和计算资源。推理是把训练好的模型用于预测。"
)

VOCAB = 128


def _tiny_model_dir(out: str) -> str:
    os.makedirs(out, exist_ok=True)
    rng.seed(0)
    cfg = ModelConfig(
        vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=4,
        n_kv_heads=2, ffn_mult=4, max_seq_len=64,
    )
    tok = Tokenizer(vocab_size=128).train(CORPUS)
    tok.to_json(f"{out}/tokenizer.json")
    trainer = TinyTrainer(cfg)
    trainer.train(tok.encode(CORPUS), steps=20, lr=3e-3)
    trainer.save(out)
    return out


def test_sdk_end_to_end(tmp_path):
    model_dir = _tiny_model_dir(str(tmp_path / "m"))
    ai = IFAI(model_dir=model_dir, seed=0)

    out = ai.chat("人工智能", max_new=8, temperature=0)
    assert isinstance(out, str), "chat 应返回字符串"

    res = ai.search(
        "注意力",
        corpus=["模型关注信息。", "训练需要数据。", "注意力聚焦关键内容。"],
        top_k=2,
    )
    assert isinstance(res, list) and len(res) <= 2, "search 应返回 top_k 列表"

    ans = ai.run("计算 (3+5)*2")
    assert isinstance(ans, str), "run 应返回字符串"


def test_model_load_and_forward(tmp_path):
    model_dir = _tiny_model_dir(str(tmp_path / "m2"))
    model = load_model(model_dir)
    tok = Tokenizer.from_json(open(f"{model_dir}/tokenizer.json", encoding="utf-8").read())
    ids = tok.encode("人工智能") or [0]
    logits = model.forward(np.array([ids]))
    assert logits.shape == (1, len(ids), VOCAB), "输出维度应等于模型词表大小"


def test_sdk_chat_completion_openai_schema(tmp_path):
    # INV-36: SDK chat_completion 返回 OpenAI 严格 schema; 与服务端单一实现一致 (INV-37)
    from ifai.serve.api import OpenAIServer

    model_dir = _tiny_model_dir(str(tmp_path / "m3"))
    ai = IFAI(model_dir=model_dir, seed=0)
    msgs = [{"role": "user", "content": "人工智能"}]
    comp = ai.chat_completion(msgs, max_new=8, temperature=0)

    assert comp["object"] == "chat.completion"
    assert comp["model"] == "ifai"
    msg = comp["choices"][0]["message"]
    assert msg["role"] == "assistant" and isinstance(msg["content"], str)
    assert comp["choices"][0]["finish_reason"] == "stop"
    assert comp["usage"]["total_tokens"] == len(msg["content"])
    # 单一实现: SDK 与 L7 服务端对同一 messages (同 prompt / 同贪心) 输出逐字一致
    srv = OpenAIServer(ai.model, ai.tok, seed=0)
    srv_msg = srv.chat(msgs, temperature=0, max_tokens=8)["choices"][0]["message"]
    assert msg["content"] == srv_msg["content"] and srv_msg["role"] == "assistant"
