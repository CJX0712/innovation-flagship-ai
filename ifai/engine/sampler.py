"""L3 采样器 (temperature / top-k / top-p / min-p / 重复惩罚)

零重写: 复现现代解码采样标准栈
- top-k (Fan et al. 2018), top-p/nucleus (Holtzman et al. 2020)
- min-p (Nguyen et al. 2024), 重复惩罚 (repetition penalty, Keskar et al. 2019)

可验证:
- INV-18 过滤+重归一后 Σp = 1 ± 1e-6
- INV-19 temperature→0 或 top_k=1 == 贪心 (argmax)
"""
from __future__ import annotations

import numpy as np

from .. import rng
from ..core import numeric


class Sampler:
    def __init__(self, seed: int | None = None):
        self._rng = rng.get_rng() if seed is None else np.random.default_rng(seed)

    # 返回归一化概率 (INV-18)
    def probs(self, logits: np.ndarray, temperature: float = 1.0,
              top_k: int = 0, top_p: float = 1.0, min_p: float = 0.0,
              repetition_penalty: float = 1.0, forbidden: set[int] | None = None) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64).copy()
        if forbidden:
            logits[list(forbidden)] = -np.inf
        if repetition_penalty != 1.0:
            # 对"已出现"的 token 做除法惩罚 (penalty>1 抑制)
            logits = np.where(logits > 0, logits / repetition_penalty,
                              logits * repetition_penalty)
        if temperature == 0:                       # INV-19 退化为 argmax
            p = np.zeros_like(logits)
            p[int(np.argmax(logits))] = 1.0
            return p
        logits = logits / max(temperature, 1e-8)
        # top-k
        if top_k and top_k > 0:
            kth = np.sort(logits)[-top_k]
            logits = np.where(logits < kth, -np.inf, logits)
        # min-p
        if min_p > 0:
            pmax = np.max(numeric.stable_softmax(logits))
            logits = np.where(numeric.stable_softmax(logits) < min_p * pmax, -np.inf, logits)
        probs = numeric.stable_softmax(logits)
        # top-p (nucleus)
        if top_p < 1.0:
            order = np.argsort(-probs)
            cum = np.cumsum(probs[order])
            cutoff = order[cum > top_p]
            probs[cutoff] = 0.0
            if probs.sum() == 0:
                probs = numeric.stable_softmax(logits)
        s = probs.sum()
        return probs / s

    def greedy(self, logits: np.ndarray, forbidden: set[int] | None = None) -> int:
        logits = np.asarray(logits, dtype=np.float64).copy()
        if forbidden:
            logits[list(forbidden)] = -np.inf
        return int(np.argmax(logits))

    def sample(self, logits: np.ndarray, temperature: float = 1.0, top_k: int = 0,
               top_p: float = 1.0, min_p: float = 0.0, repetition_penalty: float = 1.0,
               forbidden: set[int] | None = None) -> int:
        p = self.probs(logits, temperature, top_k, top_p, min_p,
                       repetition_penalty, forbidden)
        if abs(p.sum() - 1.0) > 1e-6:
            p = p / p.sum()
        return int(self._rng.choice(len(p), p=p))
