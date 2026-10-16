"""INV-20 (L3) 分块 prefill ≡ 整段 prefill (``diff ≤ 1e-5``)。

分块路径: 每块只用历史 KV + 当前块的 query (峰值激活 ∝ 块长);
整段路径: 一次性前向 (基准)。两者 logits 与 KV cache 必须逐位对齐。
"""
from __future__ import annotations

import numpy as np
import pytest

from ifai import rng
from ifai.engine.prefill import (
    ChunkedPrefill,
    chunked_prefill,
    full_prefill,
    max_abs_diff,
    verify_prefill,
)
from ifai.models.config import ModelConfig
from ifai.models.transformer import Transformer

rng.seed(23)

TOL = 1e-5


def _model() -> Transformer:
    cfg = ModelConfig(vocab_size=48, d_model=32, n_layers=3, n_heads=4,
                      n_kv_heads=2, ffn_mult=2, max_seq_len=128)
    return Transformer(cfg)


def test_chunked_prefill_equals_whole():
    """多种块长下, 分块累积前向 == 整段前向 (INV-20 主判据)。"""
    m = _model()
    ids = rng.get_rng().integers(0, 48, size=(2, 24))
    prefill = ChunkedPrefill(m, chunk_size=8)
    for cs in (1, 2, 3, 5, 7, 8, 13, 24, 64):
        rep = prefill.verify(ids, chunk_size=cs)
        assert rep["max_abs_diff"] <= TOL, f"chunk={cs} 差异 {rep['max_abs_diff']} 超过 {TOL}"
        assert rep["equal"] is True
        assert rep["n_tokens"] == 24


def test_chunked_prefill_kv_cache_equals_whole():
    """更强判据: 两条路径的 KV cache 也逐位相同 (保证后续 decode 对齐)。"""
    m = _model()
    ids = rng.get_rng().integers(0, 48, size=(1, 20))
    prefill = ChunkedPrefill(m, chunk_size=6)
    rep = prefill.verify_kv(ids, chunk_size=6)
    assert rep["equal"] is True, f"KV 差异 {rep['max_abs_diff']} 超过 {TOL}"
    assert rep["n_layers"] == 2 * m.cfg.n_layers      # 每层 K 与 V


def test_chunked_prefill_last_logits_match():
    """decode 起点: 分块 prefill 的末位 logits == 整段 prefill。"""
    m = _model()
    ids = rng.get_rng().integers(0, 48, size=(1, 17))
    p = ChunkedPrefill(m, chunk_size=4)
    a = p.last_logits(ids)
    b = p.run_whole(ids).last_logits()
    assert max_abs_diff(a, b) <= TOL


def test_chunked_prefill_long_prompt():
    """长 prompt (64 token, 块长 8) 仍严格等价。"""
    m = _model()
    ids = rng.get_rng().integers(0, 48, size=(1, 64))
    rep = verify_prefill(m, ids, chunk_size=8, tol=TOL)
    assert rep["equal"] is True
    assert rep["n_chunks"] == 8


def test_functional_api_matches_class_api():
    """函数式入口与类入口结果一致。"""
    m = _model()
    ids = rng.get_rng().integers(0, 48, size=(1, 15))
    a = chunked_prefill(m, ids, chunk_size=5).logits
    b = ChunkedPrefill(m, chunk_size=5).run(ids).logits
    c = full_prefill(m, ids).logits
    assert max_abs_diff(a, b) <= TOL
    assert max_abs_diff(a, c) <= TOL


def test_chunked_prefill_rejects_bad_chunk_size():
    """非法块长应被拒绝 (参数校验)。"""
    m = _model()
    with pytest.raises(ValueError):
        ChunkedPrefill(m, chunk_size=0)


def test_max_abs_diff_shape_mismatch_is_inf():
    """形状不一致时返回 inf (防止静默通过)。"""
    assert max_abs_diff(np.zeros((2, 3)), np.zeros((3, 2))) == float("inf")
