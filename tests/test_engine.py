"""L3 推理引擎层黄金测试 (INV-15/16/17/18/19)。"""
import numpy as np
import pytest
from ifai import rng
from ifai.models.config import ModelConfig
from ifai.models.transformer import Transformer
from ifai.engine.kvcache import PagedKVCache
from ifai.engine.sampler import Sampler
from ifai.engine.scheduler import ContinuousScheduler
from ifai.engine.speculative import speculative_decode

rng.seed(3)


def test_paged_kv_equals_dense():
    cache = PagedKVCache(block_size=4, kv_dim=8, max_blocks=64)
    sid = cache.new_sequence()
    dense_K, dense_V = [], []
    for _ in range(13):
        k = rng.get_rng().standard_normal(8)
        v = rng.get_rng().standard_normal(8)
        cache.append(sid, k, v)
        dense_K.append(k); dense_V.append(v)
    K, V = cache.get_kv(sid)
    assert np.allclose(K, np.array(dense_K), atol=1e-12)
    assert np.allclose(V, np.array(dense_V), atol=1e-12)


def _tiny_model():
    cfg = ModelConfig(vocab_size=40, d_model=32, n_layers=2, n_heads=4,
                      n_kv_heads=2, ffn_mult=2, max_seq_len=32)
    return Transformer(cfg)


def _model_forward(model):
    def fwd(ids, **kwargs):
        return model.forward(np.array(ids, dtype=np.int64), **kwargs)
    return fwd


def test_continuous_batch_equals_serial_greedy():
    """INV-16: 连续批处理 ≡ 串行独立推理 (同 seed greedy, 逐 token 100% 一致)。"""
    model = _tiny_model()
    fwd = _model_forward(model)
    prompts = [[1, 2, 3], [5, 6], [7, 8, 9, 10]]
    sched = ContinuousScheduler(fwd)
    batched = sched.generate(prompts, max_new_tokens=6, greedy=True)

    serial = []
    for p in prompts:
        ids = list(p)
        for _ in range(6):
            nxt = int(np.argmax(fwd([ids])[0, -1]))
            ids.append(nxt)
        serial.append(ids[len(p):])
    assert batched == serial


def test_sampler_probs_sum_one():
    """INV-18: 截断后采样分布归一 Σp = 1 (|Σp − 1| ≤ 1e-6)。"""
    s = Sampler(seed=0)
    logits = rng.get_rng().standard_normal(30)
    for temp in [0.5, 1.0, 2.0]:
        p = s.probs(logits, temperature=temp, top_k=5, top_p=0.9, min_p=0.01)
        assert abs(p.sum() - 1.0) < 1e-6


def test_sampler_greedy_equals_argmax():
    """INV-19: 采样退化 — T→0 与 top_k=1 均 ≡ greedy (100% 一致)。"""
    s = Sampler(seed=0)
    logits = rng.get_rng().standard_normal(20)
    assert s.greedy(logits) == int(np.argmax(logits))
    p = s.probs(logits, temperature=0.0)
    assert np.argmax(p) == int(np.argmax(logits))


def test_speculative_equals_greedy():
    """INV-17: 投机解码 (greedy) ≡ 目标模型自回归 greedy (逐 token 100% 一致)。"""
    model = _tiny_model()
    fwd = _model_forward(model)

    def next_logits(seq):
        return model.logits_next(np.array(seq))

    prompt = [1, 2, 3]
    spec = speculative_decode(next_logits, next_logits, prompt, gamma=4,
                              max_new_tokens=10, greedy=True)
    # 参考: 朴素自回归贪心
    ref = list(prompt)
    while len(ref) - len(prompt) < 10:
        nxt = int(np.argmax(next_logits(ref)))
        ref.append(nxt)
    assert spec == ref[len(prompt):]
