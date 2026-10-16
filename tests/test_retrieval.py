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
    """INV-28: HNSW 召回 ⊇ 近似暴力枚举 (recall@10 ≥ 0.95, PRD R-L5-03 规定参数
    M=16 / efC=200 / efS=64)。

    v0.7.13 真 bug 修复回归: 旧实现当新节点提升 max_level 时 enter 跳到零边
    孤立点 (插入搜索从"它自己"出发 → 自环无真实边) → 图不连通, 400 节点实测
    40 个失联, recall 恒 0.900 (恰 = 可达比例) 且 efS 拉满无效。修复 (插入
    统一以旧 enter 为入口 + 自环守卫) 后全连通且 recall = 1.000 (5 seed)。
    """
    from collections import deque
    rng_np = rng.get_rng()
    data = rng_np.standard_normal((400, 32))
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    hnsw = HNSW(M=16, ef_construction=200, ef_search=64, seed=0).build(data)
    # 回归守卫: 图必须全连通 (历史 bug 的直接可观测信号)
    seen = {hnsw.enter}
    dq = deque([hnsw.enter])
    while dq:
        cur = dq.popleft()
        for nb in hnsw.layers.get(cur, []):
            if nb not in seen:
                seen.add(nb)
                dq.append(nb)
    assert len(seen) == len(data), f"图不连通: 仅 {len(seen)}/{len(data)} 可达"
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
    assert recall >= 0.95, f"recall={recall:.3f}"


def test_dense_rerank_cross_process_reproducible(tmp_path):
    """v0.7.13 真 bug 回归: 检索编码必须跨进程可复现。

    旧实现 DenseIndex/CrossEncoderReranker 用 Python 内建 ``hash()`` — 受
    PYTHONHASHSEED 随机化影响, 同语料跨进程 fit 出不同向量, 违反自身
    「确定性可复现」声明。sha256 稳定哈希修复后, 不同 PYTHONHASHSEED 的两个
    子进程必须产出逐位一致结果。
    """
    import json
    import os
    import subprocess
    import sys

    script = (
        "import json;"
        "from ifai.retrieval.dense import DenseIndex;"
        "from ifai.retrieval.rerank import CrossEncoderReranker;"
        "corpus=[['人工智能','模型'],['检索','向量'],['注意力','机制']];"
        "di=DenseIndex(dim=64,seed=0).fit(corpus);"
        "r1=di.search(['人工智能'],3);"
        "rr=CrossEncoderReranker(dim=64,seed=0);"
        "r2=rr.rerank(['人工智能'],corpus,2);"
        "print(json.dumps([r1,r2]))"
    )
    outs = []
    for seed_env in ("0", "42"):
        env = {**os.environ, "PYTHONHASHSEED": seed_env}
        p = subprocess.run([sys.executable, "-c", script], env=env,
                           capture_output=True, text=True)
        assert p.returncode == 0, f"子进程失败 (PYTHONHASHSEED={seed_env}): {p.stderr}"
        outs.append(p.stdout.strip())
    assert outs[0] == outs[1], (
        f"跨进程不可复现 (PYTHONHASHSEED 0 vs 42):\n{outs[0]}\nvs\n{outs[1]}")


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
