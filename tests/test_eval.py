"""L8 评测层黄金测试 (INV-39 困惑度对照)。"""
import numpy as np
import pytest
from ifai import rng
from ifai.models.config import ModelConfig
from ifai.models.transformer import Transformer
from ifai.eval.perplexity import perplexity, perplexity_manual
from ifai.eval.benchmark import benchmark_generation

rng.seed(5)


def test_perplexity_vs_manual():
    cfg = ModelConfig(vocab_size=60, d_model=24, n_layers=2, n_heads=3,
                      n_kv_heads=3, ffn_mult=2, max_seq_len=32)
    m = Transformer(cfg)
    seq = rng.get_rng().integers(0, 60, size=20).tolist()
    p1 = perplexity(m, seq)
    p2 = perplexity_manual(m, seq)
    assert abs(p1 - p2) < 1e-6


def test_benchmark_runs():
    def gen(prompt, n):
        return n
    stats = benchmark_generation(gen, "x", max_new_tokens=10, runs=3)
    assert stats["runs"] == 3 and stats["tok_per_s_mean"] > 0
