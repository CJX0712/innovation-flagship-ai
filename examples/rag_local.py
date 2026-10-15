#!/usr/bin/env python
"""本地 RAG 演示 — BM25 + 稠密双塔 + RRF 融合 + Cross-Encoder 重排 (L5)。

零重写: BM25 (Robertson & Zaragoza 2009), HNSW 思想 (Malkov & Yashunin 2018),
RRF (Cormack et al. 2009), Cross-Encoder 重排 (Nogueira & Cho 2019)。

运行:
    python examples/rag_local.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ifai.retrieval import (  # noqa: E402
    BM25,
    CrossEncoderReranker,
    DenseIndex,
    reciprocal_rank_fusion,
)


def main() -> None:
    docs = [
        "注意力机制让模型聚焦输入的关键部分。",
        "变压器网络由多头注意力堆叠而成。",
        "量化技术能显著降低模型显存占用。",
        "低秩适配 LoRA 用极少参数微调大模型。",
        "检索增强生成从外部知识库取回相关资料。",
    ]
    tok = lambda s: list(s.replace(" ", ""))  # 字符级切分 (演示用, 真实场景用 BPE)
    corpus = [tok(d) for d in docs]
    query = tok("注意力 量化")

    bm25 = BM25().fit(corpus)
    bm25_res = bm25.search(query, top_k=3)
    dense = DenseIndex(dim=64).fit(corpus)
    dense_res = dense.search(query, top_k=3)

    print("[BM25] Top-3:")
    for i, s in bm25_res:
        print(f"    [{s:.3f}] {docs[i]}")

    print("\n[Dense] Top-3:")
    for i, s in dense_res:
        print(f"    [{s:.3f}] {docs[i]}")

    fused = reciprocal_rank_fusion(
        [[i for i, _ in bm25_res], [i for i, _ in dense_res]], k=60
    )
    print("\n[RRF 融合] Top-3:")
    for i, s in fused[:3]:
        print(f"    [{s:.3f}] {docs[i]}")

    reranker = CrossEncoderReranker(dim=64)
    reranked = reranker.rerank(query, corpus, top_k=3)
    print("\n[Cross-Encoder 重排] Top-3:")
    for i, s in reranked:
        print(f"    [{s:.3f}] {docs[i]}")


if __name__ == "__main__":
    main()
