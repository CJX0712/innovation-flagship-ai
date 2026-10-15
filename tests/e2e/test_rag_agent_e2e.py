"""端到端 RAG + 智能体 (INV-27/30/33/35): 混合检索管线与 ReAct 计算器。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ifai.agent.react import ReActAgent  # noqa: E402
from ifai.agent.tools import default_registry  # noqa: E402
from ifai.retrieval import (  # noqa: E402
    BM25,
    CrossEncoderReranker,
    DenseIndex,
    reciprocal_rank_fusion,
)


def _tok(s: str):
    return list(s.replace(" ", ""))


DOCS = [
    "注意力机制让模型聚焦输入的关键部分。",
    "变压器网络由多头注意力堆叠而成。",
    "量化技术能显著降低模型显存占用。",
    "低秩适配 LoRA 用极少参数微调大模型。",
    "检索增强生成从外部知识库取回相关资料。",
]


def test_hybrid_retrieval_pipeline():
    corpus = [_tok(d) for d in DOCS]
    query = _tok("注意力 量化")

    bm25 = BM25().fit(corpus)
    r1 = bm25.search(query, top_k=2)
    dense = DenseIndex(dim=32).fit(corpus)
    r2 = dense.search(query, top_k=2)

    assert len(r1) == 2 and len(r2) == 2, "BM25 与稠密检索各返回 top_k"

    fused = reciprocal_rank_fusion([[i for i, _ in r1], [i for i, _ in r2]])
    assert len(fused) >= 1, "RRF 融合应有结果"

    reranked = CrossEncoderReranker(dim=32).rerank(query, corpus, top_k=2)
    assert len(reranked) == 2, "Cross-Encoder 重排返回 top_k"

    # 融合结果应包含 BM25 命中的文档之一
    fused_idx = {i for i, _ in fused}
    bm25_idx = {i for i, _ in r1}
    assert fused_idx & bm25_idx, "融合结果应与单路检索有交集"


def test_react_agent_calculator():
    def llm(messages):
        user = messages[-1]["content"]
        if user.startswith("Observation:"):
            return f"Finish: {user.split('Observation:', 1)[1].strip()}"
        return 'Thought: 计算。\nAction: calculator(expression="(3+5)*2")'

    agent = ReActAgent(llm, default_registry(), max_steps=4)
    out = agent.run("请帮我计算一个算式")
    assert "16" in out, f"智能体应通过计算器得到 16, 实际: {out}"
