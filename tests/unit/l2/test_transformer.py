"""INV-11 (L2) Causal masking: 位置 i 的输出不依赖 j>i 的 token。

判据 (PRD): 扰动未来 token, 之前位置的输出 **bitwise 不变**。
"""
from __future__ import annotations

import numpy as np
import pytest

from ifai import rng
from ifai.models.config import ModelConfig
from ifai.models.transformer import Transformer, verify_causal_mask

rng.seed(21)


def _model(vocab: int = 48, d_model: int = 32, n_layers: int = 2,
           n_heads: int = 4, n_kv_heads: int = 2) -> Transformer:
    cfg = ModelConfig(vocab_size=vocab, d_model=d_model, n_layers=n_layers,
                      n_heads=n_heads, n_kv_heads=n_kv_heads, ffn_mult=2,
                      max_seq_len=64)
    return Transformer(cfg)


def test_causal_mask_prefix_bitwise_invariant():
    """扰动 position 之后的 token, [0, position] 的输出必须逐位相同。"""
    m = _model()
    ids = rng.get_rng().integers(0, 48, size=(2, 12))
    for pos in range(11):
        prefix_equal, n_changed = verify_causal_mask(m, ids, position=pos)
        assert prefix_equal, f"position={pos}: 未来 token 影响了过去位置的输出"
        assert n_changed > 0, f"position={pos}: 扰动未产生差异, 检验失去判别力"


def test_causal_mask_single_sequence_and_last_position():
    """一维输入 + 默认 position (最后一个 token) 同样满足 INV-11。"""
    m = _model()
    ids = rng.get_rng().integers(0, 48, size=(9,))
    prefix_equal, n_changed = verify_causal_mask(m, ids)
    assert prefix_equal
    assert n_changed > 0


def test_causal_mask_prefix_equal_to_short_prefix_forward():
    """更强判据: 前缀前向的结果 == 长序列前向的前缀部分 (bitwise)。"""
    m = _model()
    ids = rng.get_rng().integers(0, 48, size=(1, 10))
    full = m.forward(ids)
    short = m.forward(ids[:, :6])
    assert np.array_equal(full[:, :6, :], short), \
        "前缀前向与长序列前缀输出应逐位相同 (causal masking 的直接推论)"


def test_causal_mask_holds_with_past_kv():
    """带历史 KV 的分块路径 (INV-20 依赖) 同样保持因果性。"""
    m = _model()
    ids = rng.get_rng().integers(0, 48, size=(1, 10))
    pos_all = np.arange(10)
    whole = m.forward(ids, position_ids=pos_all)
    # 前 6 个 token 作为历史, 后 4 个 token 作为当前块
    _, past = m.forward(ids[:, :6], position_ids=pos_all[:6], return_kv=True)
    tail = m.forward(ids[:, 6:], position_ids=pos_all[6:], past_kv=past)
    assert np.array_equal(whole[:, 6:, :], tail), \
        "带历史 KV 的增量前向必须与整段前向逐位一致 (INV-11/INV-20 交叉验证)"


def test_perturbing_future_does_not_change_prefix_manual():
    """手写版对照: 直接改未来 token 的 id, 逐位比较 logits。"""
    m = _model(vocab=64)
    ids = rng.get_rng().integers(0, 32, size=(1, 8))
    base = m.forward(ids)
    alt = ids.copy()
    alt[:, 5:] = (alt[:, 5:] + 32)          # 全部换成另一半词表的 token
    after = m.forward(alt)
    assert np.array_equal(base[:, :5, :], after[:, :5, :])
    assert not np.array_equal(base[:, 5:, :], after[:, 5:, :])
