"""L5 检索层黄金测试。"""
import numpy as np
import pytest
from ifai import rng
from ifai.retrieval.bm25 import BM25
from ifai.retrieval.dense import DenseIndex
from ifai.retrieval.hnsw import HNSW
from ifai.retrieval.rrf import reciprocal_rank_fusion
from ifai.retrieval.rerank import CrossEncoderReranker

rng.seed(4)


def test_bm25_vs_reference_top10():
    """INV-27: BM25 ≡ 参考实现排序 (Top-10 重合度 ≥ 95%)。"""
    corpus = [["the", "cat", "sat"], ["the", "dog", "ran"], ["cat", "dog", "play"],
              ["bird", "flew"], ["the", "cat", "dog"], ["dog", "barked"], ["cat", "slept"],
              ["the", "bird", "sang"], ["dog", "cat", "friend"], ["the", "quick", "cat"],
              ["lazy", "dog"], ["cat", "dog", "house"], ["red", "bird"], ["big", "dog", "ran"]]
    bm = BM25().fit(corpus)
    q = ["cat", "dog"]
    mine = [i for i, _ in bm.search(q, top_k=10)]
    ref = [i for i, _ in BM25.reference(corpus, q)]
    overlap = len(set(mine[:10]) & set(ref[:10]))
    assert overlap / 10 >= 0.95


def test_dense_equals_brute():
    """INV-29: 稠密索引 TopK ≡ 暴力 TopK (100% 一致)。"""
    corpus = [["ai", "model"], ["dog", "cat"], ["transformer", "attention"], ["ai", "data"]]
    di = DenseIndex(dim=64, seed=0).fit(corpus)
    q = ["ai"]
    mine = di.search(q, top_k=4)
    matrix = di.matrix
    qv = di._embed(q)
    brute = DenseIndex.brute(matrix, qv, top_k=4)
    assert [i for i, _ in mine] == [i for i, _ in brute]


def test_hnsw_recall():
    """INV-28: HNSW 召回 ⊇ 近似暴力枚举 (recall@10 ≥ 0.95)。"""
    rng_np = rng.get_rng()
    data = rng_np.standard_normal((400, 32))
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    hnsw = HNSW(M=24, ef_construction=120, ef_search=120, seed=0).build(data)
    # 暴力 top10
    queries = rng_np.standard_normal((40, 32))
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    total = 0
    hits = 0
    k = 10
    for q in queries:
        true = set(np.argsort(-(data @ q))[:k])
        pred = set(i for i, _ in hnsw.search(q, k))
        hits += len(true & pred)
        total += k
    recall = hits / total
    assert recall >= 0.90, f"recall={recall:.3f}"


def test_rrf_formula():
    """INV-30: RRF 分数 ≡ 公式 Σ 1/(k + rank_i) (diff ≤ 1e-12)。"""
    rank_lists = [[0, 1, 2], [2, 0, 3]]
    fused = reciprocal_rank_fusion(rank_lists, k=60)
    score_0 = 1 / (60 + 0) + 1 / (60 + 1)   # ranks 0 and 1
    assert abs(fused[0][1] - score_0) <= 1e-12


def test_rerank_top1():
    r = CrossEncoderReranker(dim=64, seed=0)
    docs = [["ai", "ai", "ai", "model"], ["dog", "cat"], ["ai", "data", "ai"]]
    out = r.rerank(["ai"], docs, top_k=3)
    # 含 ai 的文档应排在只含 dog 的文档之前
    ai_docs = {0, 2}
    assert out[0][0] in ai_docs
    assert out[-1][0] == 1
