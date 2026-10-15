#!/usr/bin/env python
"""快速开始 (Quickstart) — 用 L9 SDK 三行调用 chat / search / agent。

零重写: 参考 HuggingFace ``transformers.pipeline`` 与 OpenAI Python SDK 的
极简易用范式 (R-L9-02)。覆盖 sdk.IFAI.chat / search / run。

运行:
    python examples/quickstart.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import train_tiny  # noqa: E402
from ifai.sdk import IFAI  # noqa: E402


def main() -> None:
    print(">> 训练一个微型模型 (CPU, 数秒) ...")
    model_dir = train_tiny(steps=60)
    ai = IFAI(model_dir=model_dir, seed=0)

    print("\n[1] 续写 (chat, 贪心):")
    print("   ", ai.chat("人工智能", max_new=12, temperature=0))

    print("\n[2] 检索 (search, BM25):")
    corpus = [
        "模型关注重要的信息。",
        "训练需要大量数据。",
        "注意力让模型聚焦关键内容。",
        "推理是把模型用于预测。",
    ]
    for idx, score in ai.search("注意力", corpus=corpus, top_k=3):
        print(f"    [{score:.3f}] {corpus[idx]}")

    print("\n[3] 智能体 (run, 内置计算器):")
    print("   ", ai.run("请计算 (17*23+9)/4"))


if __name__ == "__main__":
    main()
