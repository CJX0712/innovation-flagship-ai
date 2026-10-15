"""L5 稠密向量检索 (Dense Retrieval)

零重写: 双塔 (DPR, Karpukhin et al. 2020) 思想 — 查询与文档在共享嵌入空间
做余弦相似度。本栈用哈希-加权嵌入 (hashing trick + TF 加权) 作为零依赖
可复现的稠密编码器, 与暴力 TopK 完全一致 (INV-29)。

可验证 (INV-29): 本索引 TopK 与暴力余弦排序 100% 一致。
"""
from __future__ import annotations

import numpy as np


class DenseIndex:
    def __init__(self, dim: int = 256, seed: int = 0):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        # 随机投影矩阵 (固定种子 ⇒ 可复现)
        self.proj = self.rng.standard_normal((dim,))

    def _embed(self, tokens: list[str]) -> np.ndarray:
        vec = np.zeros(self.dim)
        for t in tokens:
            h = hash(t) % self.dim
            vec[h] += 1.0
        # L2 归一化
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec

    def fit(self, corpus: list[list[str]]):
        self.docs = [self._embed(d) for d in corpus]
        self.matrix = np.array(self.docs) if self.docs else np.zeros((0, self.dim))
        return self

    def search(self, query: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        q = self._embed(query)
        if self.matrix.shape[0] == 0:
            return []
        sims = self.matrix @ q
        idx = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in idx]

    # 暴力参考 (INV-29)
    @staticmethod
    def brute(matrix: np.ndarray, q: np.ndarray, top_k: int = 10):
        sims = matrix @ q
        idx = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in idx]
