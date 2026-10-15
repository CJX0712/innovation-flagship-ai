"""L9 Python SDK — 面向开发者的高层接口

零重写: 参考 HuggingFace ``transformers.pipeline`` 与 OpenAI Python SDK 的
极简易用范式。覆盖 R-L9-02 (公开 API 100% docstring)。

示例:
    from ifai.sdk import IFAI
    ai = IFAI(model_dir="ifai_model")
    print(ai.chat("你好"))
    print(ai.search("注意力", corpus=["..."]))
    print(ai.run("计算 (3+5)*2"))
"""
from __future__ import annotations

import os
import numpy as np

from . import rng
from .models.tokenizer import Tokenizer
from .models.transformer import load_model
from .engine.sampler import Sampler
from .agent.tools import default_registry
from .agent.react import ReActAgent
from .retrieval.bm25 import BM25


class IFAI:
    """旗舰 AI 高层客户端。"""

    def __init__(self, model_dir: str = "./ifai_model", seed: int = 0):
        """载入模型与分词器。

        Args:
            model_dir: TinyTrainer/``ifai train`` 输出的目录, 内含 params.npz,
                       manifest.json, tokenizer.json。
            seed: 全局随机种子 (可复现生成)。
        """
        rng.seed(seed)
        tok_path = os.path.join(model_dir, "tokenizer.json")
        self.tok = Tokenizer.from_json(open(tok_path, encoding="utf-8").read())
        self.model = load_model(model_dir)
        self.sampler = Sampler(seed=seed)

    def chat(self, prompt: str, max_new: int = 32, temperature: float = 0.8) -> str:
        """生成续写文本 (贪心/采样)。"""
        ids = self.tok.encode(prompt) or [0]
        for _ in range(max_new):
            logits = self.model.forward(np.array([ids]))[0, -1]
            nxt = self.sampler.greedy(logits) if temperature == 0 else \
                self.sampler.sample(logits, temperature=temperature)
            ids.append(nxt)
            if self.tok.id2token.get(nxt) == "<eos>":
                break
        return "".join(self.tok.decode(ids[len(self.tok.encode(prompt)):] if self.tok.encode(prompt) else ids[1:])).strip() or "(空)"

    def search(self, query: str, corpus: list[str], top_k: int = 5) -> list[tuple[int, float]]:
        """BM25 检索, 返回 [(doc_idx, score)...]。"""
        bm25 = BM25().fit([d.split() for d in corpus])
        return bm25.search(query.split(), top_k=top_k)

    def run(self, question: str, max_steps: int = 4) -> str:
        """运行 ReAct 智能体 (内置计算器工具)。"""
        def llm(messages):
            u = messages[-1]["content"]
            if "计算" in u or any(c in u for c in "+-*/"):
                expr = u.split("计算")[-1].strip()
                return f'Thought: 调用计算器。\nAction: calculator(expression="{expr}")'
            return "Finish: 已回答 (演示)"
        agent = ReActAgent(llm, default_registry(), max_steps=max_steps)
        return agent.run(question)
