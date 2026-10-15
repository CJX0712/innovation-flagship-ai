"""L7 OpenAI 兼容 HTTP 服务 (REST)

零重写: 兼容 OpenAI REST 协议 (v1/models, v1/completions, v1/chat/completions)
以便现有生态 (LangChain / OpenAI SDK) 零改动接入。chat/completions 严格返回
OpenAI 的 choices[].message.content 结构 (INV-36); 生成长度恒定, 保证流式
消费与非流式 content 一致 (INV-37)。

可验证:
- INV-36 响应字段集与 OpenAI schema diff 为空 (见 tests)
- INV-37 流式 content 拼接 == 非流式 content
"""
from __future__ import annotations

import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, List

import numpy as np

from .. import rng
from ..engine.sampler import Sampler
from ..models.transformer import Transformer
from ..models.tokenizer import Tokenizer


def format_messages(messages: List[dict]) -> str:
    """将 OpenAI chat messages 拼接为单条 prompt (逐行 "role: content")。

    零重写: 与 OpenAI chat 语义一致, 供服务端与 L9 SDK 共用单一实现。

    Args:
        messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]

    Returns:
        拼接后的单条 prompt 字符串。
    """
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def chat_completion(text: str, model_name: str) -> dict:
    """构造 OpenAI 严格的 ``chat.completion`` 响应体 (INV-36)。

    零重写: 字段集与 OpenAI ``/v1/chat/completions`` schema 完全一致 (diff 为空),
    且 ``choices[].message.content`` 与 ``/v1/completions`` 的 ``choices[].text``
    逐字一致 (INV-37)。服务端 ``OpenAIServer.chat`` 与 L9 SDK ``IFAI.chat_completion``
    共用本函数, 保证单一事实来源。

    Args:
        text: 已生成的续写文本。
        model_name: 模型标识, 回填至响应 ``model`` 字段。

    Returns:
        符合 OpenAI schema 的 dict。
    """
    n = len(text)
    return {
        "id": "chatcmpl-ifai",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": n,
                  "total_tokens": n},
    }


class OpenAIServer:
    def __init__(self, model: Transformer, tokenizer: Tokenizer,
                 max_new_tokens: int = 64, seed: int = 0):
        self.model = model
        self.tok = tokenizer
        self.max_new_tokens = max_new_tokens
        self.sampler = Sampler(seed=seed)
        self.model_name = "ifai-flagship"

    # ----- 逻辑层 (可在不联网情况下被测) -----
    def _gen_tokens(self, prompt: str, max_new: int, temperature: float,
                    top_p: float, stream: bool = False):
        """生成 token 迭代器。stream 仅决定调用方是否增量消费;
        生成长度恒为 max_new (或遇 <eos> 提前结束), 保证流式==非流式 (INV-37)。"""
        ids = self.tok.encode(prompt)
        if not ids:
            ids = [self.tok.token2id.get("<bos>", 0)]
        for _ in range(max_new):
            logits = self.model.forward(np.array([ids]))[0, -1, :]
            if temperature == 0:
                nxt = self.sampler.greedy(logits)
            else:
                nxt = self.sampler.sample(logits, temperature=temperature, top_p=top_p)
            ids.append(nxt)
            if self.tok.id2token.get(nxt) == "<eos>":
                break
            tok = self.tok.decode([nxt])
            yield tok

    def completions(self, prompt: str, temperature=0.8, top_p=1.0,
                    max_tokens=None, stream=False) -> dict | str:
        max_new = max_tokens or self.max_new_tokens
        text = "".join(self._gen_tokens(prompt, max_new, temperature, top_p, stream))
        obj = {"id": "cmpl-ifai", "object": "text_completion",
               "model": self.model_name,
               "choices": [{"text": text, "index": 0,
                            "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 0, "completion_tokens": len(text),
                         "total_tokens": len(text)}}
        return obj

    def chat(self, messages: List[dict], **kw) -> dict | str:
        prompt = format_messages(messages)
        comp = self.completions(prompt, **kw)
        # OpenAI 严格兼容: chat/completions 返回 choices[].message.content
        # (INV-36 字段集与 OpenAI schema diff 为空; 与 completions 的 text 一致, INV-37)
        return chat_completion(comp["choices"][0]["text"], self.model_name)

    def models(self) -> dict:
        return {"object": "list",
                "data": [{"id": self.model_name, "object": "model",
                          "owned_by": "morningstar"}]}

    # ----- HTTP 路由 -----
    def route(self, method: str, path: str, body: dict | None = None):
        if path == "/v1/models" and method == "GET":
            return 200, self.models()
        if path == "/v1/completions" and method == "POST":
            b = body or {}
            return 200, self.completions(b.get("prompt", ""), b.get("temperature", 0.8),
                                         b.get("top_p", 1.0), b.get("max_tokens"),
                                         b.get("stream", False))
        if path == "/v1/chat/completions" and method == "POST":
            b = body or {}
            return 200, self.chat(b.get("messages", []), temperature=b.get("temperature", 0.8),
                                  top_p=b.get("top_p", 1.0), max_tokens=b.get("max_tokens"),
                                  stream=b.get("stream", False))
        return 404, {"error": "not found"}


def create_handler(server: OpenAIServer):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, payload):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

        def do_GET(self):
            st, pl = server.route("GET", self.path)
            self._send(st, pl)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                body = {}
            st, pl = server.route("POST", self.path, body)
            self._send(st, pl)

        def log_message(self, *a):
            pass

    return Handler


def serve(server: OpenAIServer, host: str = "127.0.0.1", port: int = 8000):
    httpd = ThreadingHTTPServer((host, port), create_handler(server))
    httpd.serve_forever()
