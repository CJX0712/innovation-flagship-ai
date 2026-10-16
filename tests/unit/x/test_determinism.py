"""INV-40 (X) 确定性: 同 seed 两次运行 **bitwise 一致**。

覆盖四条确定性链路 (对应 PRD R-L2-05 / R-L8-06):
1. 参数初始化: 同 seed 两次构建, 逐参数 ``array_equal`` (bitwise);
2. 前向: 同模型同输入两次前向 logits bitwise 一致;
3. 训练: 两次同 seed 训练各走一步后参数 bitwise 一致;
4. 采样: 同 seed Sampler 对同一 logits 两次采样 token 一致。
"""
from __future__ import annotations

import numpy as np

from ifai import rng
from ifai.engine.sampler import Sampler
from ifai.models.config import ModelConfig
from ifai.models.transformer import Transformer
from ifai.train.trainer import TinyTrainer


def _cfg() -> ModelConfig:
    return ModelConfig(vocab_size=50, d_model=24, n_layers=2, n_heads=4,
                       n_kv_heads=2, ffn_mult=2, max_seq_len=32)


def _build(seed: int) -> Transformer:
    rng.seed(seed)
    return Transformer(_cfg())


def test_inv_40_init_bitwise_identical():
    """同 seed 两次构建: 每个参数数组 bitwise 相等。"""
    a, b = _build(40), _build(40)
    assert set(a.params) == set(b.params)
    for k in a.params:
        assert np.array_equal(a.params[k], b.params[k]), f"参数 {k} 非 bitwise 一致"
    # 不同 seed 应当不同 (判别力对照)
    c = _build(41)
    assert not np.array_equal(a.params["embed"], c.params["embed"])


def test_inv_40_forward_bitwise_identical():
    """同模型同输入两次前向 + 两个同 seed 模型前向, logits 均 bitwise 一致。"""
    m = _build(7)
    ids = rng.get_rng().integers(0, 50, size=(2, 10))
    l1 = m.forward(ids)
    l2 = m.forward(ids)
    assert np.array_equal(l1, l2), "同模型两次前向必须 bitwise 一致"
    assert np.array_equal(_build(7).forward(ids), l1), "同 seed 重建模型前向必须 bitwise 一致"


def test_inv_40_train_step_bitwise_identical():
    """两次同 seed 训练: 初始参数与一步 AdamW 后参数均 bitwise 一致。"""
    rng.seed(99)
    ids = rng.get_rng().integers(0, 50, size=(160,))  # 1D token 语流 (train 内部切窗)
    inits, afters = [], []
    for _ in range(2):
        rng.seed(99)  # 每次构建前重置: 确定性要求"同 seed 两次运行", 而非同一状态连续两次
        tr = TinyTrainer(_cfg())
        inits.append({k: v.data.copy() for k, v in tr.params.items()})
        tr.train(ids, steps=1, lr=1e-2)
        afters.append({k: v.data.copy() for k, v in tr.params.items()})
    for k in inits[0]:
        assert np.array_equal(inits[0][k], inits[1][k]), f"初始参数 {k} 非 bitwise 一致"
        assert np.array_equal(afters[0][k], afters[1][k]), f"一步训练后 {k} 非 bitwise 一致"
    # 训练一步后参数必须与初始不同 (对照: 证明"一致"不是没训练)
    assert not np.array_equal(afters[0]["embed"], inits[0]["embed"])


def test_inv_40_sampler_seed_deterministic():
    """同 seed Sampler 对同一 logits 两次采样 token 一致 (贪心与采样两路)。"""
    logits = rng.get_rng().standard_normal(64) * 3
    s1, s2 = Sampler(seed=123), Sampler(seed=123)
    assert s1.greedy(logits) == s2.greedy(logits)
    assert s1.sample(logits, temperature=0.8) == s2.sample(logits, temperature=0.8)
    assert s1.sample(logits, temperature=0.8, top_k=5, top_p=0.9) == \
        s2.sample(logits, temperature=0.8, top_k=5, top_p=0.9)
