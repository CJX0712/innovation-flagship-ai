"""INV-13 / INV-14 (L2) 参数量守恒 与 MoE 路由守恒。

- **INV-13** 可训练参数计数 == 按配置**解析计算**值 (精确整数相等)。
  解析公式独立于 ``count_params`` 推导 (对照实现交叉验证), 同时覆盖
  权重绑定 (tied, INV-12) 与非绑定、稠密 FFN 与 MoE 四种配置。
- **INV-14** MoE 路由: 选中专家的 softmax 概率和 == 1; 每个 token 恰好
  分配 ``moe_topk`` 个互不相同专家 (分配守恒); 独立重算整条 MoE 前向
  与实现输出逐位一致 (对照)。
"""
from __future__ import annotations

import numpy as np
import pytest

from ifai import rng
from ifai.models.config import ModelConfig
from ifai.models.transformer import Transformer

rng.seed(13)


def _analytic_param_count(c: ModelConfig) -> int:
    """按配置解析计算参数量 (不读 params 字典, 纯算术)。"""
    d, V = c.d_model, c.vocab_size
    d_kv = c.n_kv_heads * c.d_head
    if c.use_moe:
        # attn_norm + q + k + v + o + ffn_norm + E*(gate+up+down) + router
        per_layer = (2 * d + d * d + 2 * d * d_kv + d * d
                     + 3 * d * c.d_ff * c.n_experts + d * c.n_experts)
    else:
        per_layer = 2 * d + d * d + 2 * d * d_kv + d * d + 3 * d * c.d_ff
    head = V * d * (1 if c.tie_word_embeddings else 2)  # tied 只计一次 (INV-12)
    return head + d + c.n_layers * per_layer  # + final_norm


@pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
def test_inv_13_param_count_matches_analytic(tied, moe):
    """INV-13 主判据: count_params() == 解析公式 (精确整数, 4 种配置)。"""
    cfg = ModelConfig(vocab_size=71, d_model=24, n_layers=3, n_heads=4,
                      n_kv_heads=2, ffn_mult=2, max_seq_len=32,
                      tie_word_embeddings=tied, use_moe=moe,
                      n_experts=4, moe_topk=2)
    m = Transformer(cfg)
    assert m.count_params() == _analytic_param_count(cfg), (
        f"tied={tied} moe={moe}: count={m.count_params()} "
        f"analytic={_analytic_param_count(cfg)}")
    # tied 时 lm_head 与 embed 必须是同一对象 (INV-12 交叉确认)
    if tied:
        assert m.params["lm_head"] is m.params["embed"]


def test_inv_13_tied_not_double_counted():
    """回归: tied 模型的参数量不得把共享 embedding 计两次。"""
    cfg = ModelConfig(vocab_size=100, d_model=16, n_layers=1, n_heads=2,
                      n_kv_heads=2, ffn_mult=2, max_seq_len=16,
                      tie_word_embeddings=True)
    m = Transformer(cfg)
    unique = len({id(v) for v in m.params.values()})
    assert m.count_params() == _analytic_param_count(cfg)
    assert unique < len(m.params), "tied 配置下应存在共享数组 (lm_head is embed)"


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """测试内独立 softmax (不 import 实现, 保持对照独立性)。"""
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def test_inv_14_moe_routing_conservation():
    """INV-14: 选中概率和 == 1 + 分配守恒 + 独立重算 MoE 前向逐位一致。"""
    cfg = ModelConfig(vocab_size=60, d_model=16, n_layers=1, n_heads=2,
                      n_kv_heads=2, use_moe=True, n_experts=4, moe_topk=2)
    m = Transformer(cfg)
    B, T, d = 3, 5, 16
    h = rng.get_rng().standard_normal((B, T, d))
    pre = "layer0."

    # 独立重算路由
    router_logits = h @ m.params[pre + "router"]                    # [B,T,E]
    topk = np.argsort(router_logits, axis=-1)[..., -cfg.moe_topk:]  # [B,T,k]
    # 守恒 1: 每 token 恰好 moe_topk 个互不相同专家
    for b in range(B):
        for t in range(T):
            assert len(set(topk[b, t].tolist())) == cfg.moe_topk
    # 守恒 2: 总分配数 == n_tokens * topk
    total_assign = sum(int(np.sum(topk == e)) for e in range(cfg.n_experts))
    assert total_assign == B * T * cfg.moe_topk
    # 归一: 选中 logits 的 softmax 概率和 == 1
    sel = np.take_along_axis(router_logits, topk, axis=-1)
    w = _softmax(sel, axis=-1)
    assert np.allclose(w.sum(axis=-1), 1.0, atol=1e-12)

    # 独立重算整条 MoE 前向 (专家计算 + 加权聚合) 与实现对照
    out_ref = np.zeros((B, T, d))
    for e in range(cfg.n_experts):
        gate = h @ m.params[pre + f"e{e}_gate"]
        up = h @ m.params[pre + f"e{e}_up"]
        ex = (gate / (1.0 + np.exp(-gate)) * up) @ m.params[pre + f"e{e}_down"]
        out_ref += np.sum(w * (topk == e), axis=-1, keepdims=True) * ex
    out = m._moe(h, pre, cfg)
    assert np.max(np.abs(out - out_ref)) <= 1e-12


def test_moe_forward_and_save_load_roundtrip(tmp_path):
    """MoE 模型走完整 forward (覆盖 Transformer.forward 的 use_moe 分支) +
    save/load 往返参数量守恒 (INV-13 × MoE 配置)。"""
    cfg = ModelConfig(vocab_size=48, d_model=16, n_layers=2, n_heads=2,
                      n_kv_heads=2, use_moe=True, n_experts=3, moe_topk=2,
                      max_seq_len=32)
    m = Transformer(cfg)
    ids = rng.get_rng().integers(0, 48, size=(2, 6))
    out = m.forward(ids)
    assert out.shape == (2, 6, 48) and np.all(np.isfinite(out))
    d = m.save(str(tmp_path / "moe"))
    m2 = Transformer.load(d)
    assert m2.count_params() == m.count_params() == cfg_analytic(cfg)
    assert np.array_equal(m.forward(ids), m2.forward(ids))


def cfg_analytic(c: ModelConfig) -> int:
    return _analytic_param_count(c)


def test_forward_with_past_kv_and_padding_mask_extension():
    """past_kv + key_padding_mask 仅覆盖历史段 → 触发掩码右扩展分支 (INV-20/11)。"""
    from ifai.engine.kvcache import PagedKVCache
    cfg = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=2,
                      n_kv_heads=2, ffn_mult=2, max_seq_len=32)
    m = Transformer(cfg)
    past_ids = rng.get_rng().integers(0, 32, size=(1, 4))
    cur_ids = rng.get_rng().integers(0, 32, size=(1, 2))
    _, kv = m.forward(past_ids, return_kv=True)
    # pad 掩码只给历史 4 位 (当前块无 pad) → 内部需扩展到 P+T
    pad = np.zeros((1, 4), dtype=bool)
    out = m.forward(cur_ids, past_kv=kv, key_padding_mask=pad)
    assert out.shape == (1, 2, 32) and np.all(np.isfinite(out))
    # 无掩码前向与全 False 掩码前向逐位一致
    out_nomask = m.forward(cur_ids, past_kv=kv)
    assert np.array_equal(out, out_nomask)
