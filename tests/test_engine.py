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


def _spec_accept_run() -> list:
    """整块确定性运行: seed → logits → 投机解码 (草稿≡目标, 全接受)。"""
    rng.seed(5)
    V = 12
    base = rng.get_rng().standard_normal(V) * 2
    logits_fn = lambda seq: base + 0.1 * len(seq)
    return speculative_decode(logits_fn, logits_fn, [0, 1], gamma=3,
                              max_new_tokens=9, greedy=False, temperature=0.9)


def test_speculative_sampling_mode_accept_path():
    """采样模式 + 草稿≡目标: 接受概率恒为 1, 全部候选被接受且同 seed 可复现。"""
    spec = _spec_accept_run()
    V = 12
    assert len(spec) == 9 and all(0 <= t < V for t in spec)
    assert spec == _spec_accept_run(), "同 seed 两次运行应逐 token 一致"


def test_speculative_sampling_mode_reject_path():
    """采样模式 + 对抗草稿 (恒提 p_t≈0 的 token): 必走拒绝分支,
    输出退化为目标模型贪心序列 (拒绝后回填 argmax)。"""
    V = 10
    # 目标分布: token 0 概率≈0 (logit 很低), 其余可区分
    def target_next(seq):
        lg = np.full(V, -1.0)
        lg[0] = -50.0
        lg[1 + (len(seq) % (V - 1))] = 5.0  # 随前缀长度轮换的强 token
        return lg

    def draft_next(seq):
        lg = np.full(V, -30.0)
        lg[0] = 10.0  # 草稿执着于目标几乎不会选的 token → 接受率≈0
        return lg

    rng.seed(9)
    spec = speculative_decode(target_next, draft_next, [3], gamma=4,
                              max_new_tokens=6, greedy=False, temperature=1.0)
    # 每个位置都应是目标模型的 argmax (拒绝 → 回填 argmax)
    seq = [3]
    for _ in range(6):
        seq.append(int(np.argmax(target_next(seq))))
    assert spec == seq[1:], f"拒绝路径应退化为贪心: {spec} vs {seq[1:]}"


def test_sampler_edge_branches():
    """采样器边界分支: forbidden / repetition_penalty / top_k=1 / min_p / top_p 截断。"""
    from ifai.engine.sampler import Sampler
    s = Sampler(seed=3)
    logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    # forbidden: 被禁 token 永不被选; greedy 同样生效
    for _ in range(20):
        assert s.sample(logits, forbidden={0}) != 0
        assert s.greedy(logits, forbidden={0}) == 1
    # repetition_penalty>1 抑制已出现的正 logit
    p_pen = s.probs(logits, temperature=1.0, repetition_penalty=2.0)
    p_raw = s.probs(logits, temperature=1.0)
    assert p_pen[0] < p_raw[0] and abs(p_pen.sum() - 1.0) <= 1e-9
    # top_k=1 ≡ greedy (INV-19 补充: 概率路径)
    assert s.sample(logits, temperature=0.5, top_k=1) == int(np.argmax(logits))
    # min_p: 过滤 p < min_p*pmax 的尾部 (此处保留 top-2), 过滤位归零且归一
    p_min = s.probs(logits, temperature=1.0, min_p=0.3)
    assert p_min[2:] .sum() == 0.0 and p_min[0] > 0.7 and p_min[1] > 0.2
    assert abs(p_min.sum() - 1.0) <= 1e-9
    # top_p 极小: 核集只含 argmax (修复后不再全零回退)
    p_tiny = s.probs(logits, temperature=1.0, top_p=1e-9)
    assert abs(p_tiny.sum() - 1.0) <= 1e-9 and p_tiny[np.argmax(logits)] > 0.999
    # top_p=0.9: 核集 = 累计质量 ≥ 0.9 的最小前缀 (此处 top-3, 质量 ≈0.956)
    p_nuc = s.probs(logits, temperature=1.0, top_p=0.9)
    kept = np.nonzero(p_nuc)[0]
    assert kept.tolist() == [0, 1, 2], f"核集应为 top-3: {kept}"
    assert (p_nuc * logits).sum() > 0 and abs(p_nuc.sum() - 1.0) <= 1e-9
    assert p_nuc[:3].sum() >= 0.9, "核集累计质量必须 ≥ top_p"
