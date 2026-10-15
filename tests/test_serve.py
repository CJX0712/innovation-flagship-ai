"""L7 服务层黄金测试 (INV-36 schema 字段, INV-37 流式==非流式)。"""
import numpy as np
import pytest
from ifai import rng
from ifai.models.config import ModelConfig
from ifai.models.transformer import Transformer
from ifai.models.tokenizer import Tokenizer
from ifai.serve.api import OpenAIServer

rng.seed(6)

_CORPUS = "人工智能 模型 注意力 训练 推理 分词 采样 检索 智能体 记忆"


@pytest.fixture
def server():
    cfg = ModelConfig(vocab_size=512, d_model=64, n_layers=2, n_heads=4,
                      n_kv_heads=2, ffn_mult=2, max_seq_len=64)
    model = Transformer(cfg)
    tok = Tokenizer(vocab_size=512).train(_CORPUS)
    return OpenAIServer(model, tok, seed=0)


def test_openai_schema_fields(server):
    m = server.models()
    assert m["object"] == "list" and "data" in m and "id" in m["data"][0]
    st, comp = server.route("POST", "/v1/completions",
                            {"prompt": "人工智能", "max_tokens": 4, "temperature": 0})
    assert "choices" in comp and "usage" in comp and "id" in comp


def test_streaming_equals_nonstreaming(server):
    streamed = "".join(server._gen_tokens("人工智能", max_new=8, temperature=0,
                                          top_p=1.0, stream=True))
    nonstream = server.completions("人工智能", temperature=0, max_tokens=8)["choices"][0]["text"]
    assert streamed == nonstream
