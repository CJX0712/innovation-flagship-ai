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


def test_gsm8k_mini_oracle_and_zero():
    """GSM8K-Mini: oracle 答案函数全对; 恒空答案全错 (extract None)。"""
    from ifai.eval.harness import GSM8K_Mini
    g = GSM8K_Mini()
    rep = g.evaluate(lambda q: dict(g.DATA)[q])
    assert rep["total"] >= 8 and rep["correct"] == rep["total"]
    rep0 = g.evaluate(lambda q: "我不知道")
    assert rep0["correct"] == 0 and all(s["pred"] is None for s in rep0["samples"])


def test_gsm8k_extract_number_takes_last():
    """extract_number 取最后一个数字 (含负号)。"""
    from ifai.eval.harness import GSM8K_Mini
    ex = GSM8K_Mini.extract_number
    assert ex("答案是 42。") == 42
    assert ex("3 + 5 = 8, 所以等于 8 元") == 8
    assert ex("温度是 -5 度") == -5
    assert ex("没有数字") is None


def test_regression_baseline_gate(tmp_path):
    """RegressionBaseline: GREEN / RED / NO_BASELINE 三态判定。"""
    import os
    from ifai.eval.harness import RegressionBaseline
    p = tmp_path / "BASELINE.md"
    p.write_text("tok/s mean: 100.0", encoding="utf-8")
    cur = {"tok_per_s_mean": 95.0}
    assert RegressionBaseline.check(cur, str(p))["status"] == "GREEN"      # 掉 5% < 15%
    red = RegressionBaseline.check({"tok_per_s_mean": 80.0}, str(p))       # 掉 20%
    assert red["status"] == "RED" and abs(red["drop"] - 0.20) <= 1e-9
    assert RegressionBaseline.check(cur, str(tmp_path / "无此文件.md"))["status"] == "NO_BASELINE"
