#!/usr/bin/env python
"""OpenAI 兼容服务客户端演示 — 启动本地服务并用标准库 urllib 发起请求 (L7)。

零重写: 复用 OpenAI REST / SSE 协议约定 (chat/completions, /v1/models)。

运行:
    python examples/serve_client.py
(会在本机 127.0.0.1:8099 临时启动一个 HTTP 服务并自动关闭)
"""
from __future__ import annotations

import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import train_tiny  # noqa: E402
from ifai.models.tokenizer import Tokenizer  # noqa: E402
from ifai.models.transformer import load_model  # noqa: E402
from ifai.serve.api import OpenAIServer, create_handler  # noqa: E402

HOST, PORT = "127.0.0.1", 8099


def _request(method: str, path: str, payload=None):
    url = f"http://{HOST}:{PORT}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def main() -> None:
    model_dir = train_tiny(steps=40)
    tok = Tokenizer.from_json(open(f"{model_dir}/tokenizer.json", encoding="utf-8").read())
    model = load_model(model_dir)
    srv = OpenAIServer(model, tok, seed=0)

    httpd = ThreadingHTTPServer((HOST, PORT), create_handler(srv))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"本地服务已启动: http://{HOST}:{PORT}")

    try:
        models = _request("GET", "/v1/models")
        print("\n[/v1/models]:", models["data"][0]["id"])

        chat = _request(
            "POST", "/v1/chat/completions",
            {"model": "ifai-flagship",
             "messages": [{"role": "user", "content": "人工智能"}],
             "max_tokens": 12, "temperature": 0},
        )
        choice = chat["choices"][0]
        reply = choice.get("message", {}).get("content") or choice.get("text", "")
        print("[/v1/chat/completions] 回复:", reply)
        print("\n演示完成: OpenAI 兼容接口可用 (INV-36/INV-37)。")
    except urllib.error.URLError as e:
        print("请求失败:", e)
    finally:
        httpd.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    main()
