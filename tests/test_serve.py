"""L7 服务层黄金测试 (INV-36 schema 字段, INV-37 流式==非流式)。"""
import numpy as np
import pytest
from ifai import rng
from ifai.models.config import ModelConfig
from ifai.models.transformer import Transformer
from ifai.models.tokenizer import Tokenizer
from ifai.serve.api import OpenAIServer
from http.server import ThreadingHTTPServer

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


def test_chat_message_content_schema(server):
    # INV-36: chat/completions 严格返回 OpenAI 的 choices[].message.content
    st, chat = server.route(
        "POST", "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "人工智能"}],
         "max_tokens": 4, "temperature": 0},
    )
    assert st == 200
    assert chat["object"] == "chat.completion"
    msg = chat["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert isinstance(msg["content"], str) and "usage" in chat
    # 与非流式 completions 文本一致 (INV-37)
    comp = server.completions("人工智能", temperature=0, max_tokens=4)["choices"][0]["text"]
    assert msg["content"] == comp


def test_real_http_server_end_to_end(server):
    """真实 HTTP 层 (ThreadingHTTPServer + Handler): GET/POST/404/坏 JSON。

    此前测试只直调 route(); 本测试用临时端口起真服务, 验证 OpenAI 生态
    (urllib/openai SDK) 实际走到的字节级契约。
    """
    import json
    import threading
    import urllib.error
    import urllib.request
    from ifai.serve.api import create_handler

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(server))
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"
    try:
        # GET /v1/models
        with urllib.request.urlopen(f"{base}/v1/models", timeout=10) as r:
            models = json.loads(r.read())
        assert r.status == 200 and models["object"] == "list"
        # POST /v1/chat/completions → OpenAI 严格 schema (INV-36)
        req = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=json.dumps({"messages": [{"role": "user", "content": "模型"}],
                             "max_tokens": 4, "temperature": 0}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            chat = json.loads(r.read())
        assert chat["object"] == "chat.completion"
        assert chat["choices"][0]["message"]["role"] == "assistant"
        # POST /v1/completions
        req2 = urllib.request.Request(
            f"{base}/v1/completions",
            data=json.dumps({"prompt": "模型", "max_tokens": 3, "temperature": 0}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req2, timeout=10) as r:
            comp = json.loads(r.read())
        assert comp["object"] == "text_completion" and "choices" in comp
        # 404 未知路径
        try:
            urllib.request.urlopen(f"{base}/v1/nope", timeout=10)
            raise AssertionError("未知路径应 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404
        # 坏 JSON → 按空 body 处理, 仍 200 (路由默认参数)
        req3 = urllib.request.Request(
            f"{base}/v1/completions", data=b"not-json{{",
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req3, timeout=10) as r:
            assert r.status == 200
    finally:
        httpd.shutdown()
        httpd.server_close()
