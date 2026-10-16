"""v0.7.0 训练器批量可微注意力: 梯度贯通 + 前向语义一致性 (回归防护)。

历史缺陷: 旧 ``_layer_forward`` 经 ``.data`` 取 raw numpy 后以
``Tensor(ctx)`` 重建叶子, 切断 attn_norm/q_proj/k_proj/v_proj 梯度链 —
注意力参数从未收到梯度 (INV-26 单调性仅靠 FFN/embed 路径仍可达成,
缺陷被掩盖)。本文件三项硬校验防止回归:

1. trainer 前向与推理端 ``Transformer`` 数学一致 (save→load 往返对照);
2. 注意力参数一步 backward 后梯度非 None 且范数 > 0;
3. 注意力参数梯度通过中心差分精确校验 (相对误差 < 1e-5)。
"""
from __future__ import annotations

import numpy as np
import pytest

from ifai import rng
from ifai.models.config import ModelConfig
from ifai.models.transformer import Transformer
from ifai.train.trainer import TinyTrainer


def _make_cfg(n_kv: int | None, n_heads: int = 4) -> ModelConfig:
    return ModelConfig(
        vocab_size=48, d_model=16, n_layers=2, n_heads=n_heads,
        n_kv_heads=n_kv, ffn_mult=2, max_seq_len=32,
        tie_word_embeddings=True,
    )


def _zero_grads(trainer: TinyTrainer) -> None:
    for p in trainer.params.values():
        p.grad = np.zeros_like(p.data)


# ---------------- 1. 前向数学一致性 (trainer vs 推理端) ----------------

@pytest.mark.parametrize("n_kv", [None, 2, 1])  # MHA / GQA / MQA
def test_forward_matches_inference_engine(tmp_path, n_kv):
    """TinyTrainer.forward 与 Transformer (加载同参数) 输出 max|diff| ≤ 1e-9。"""
    rng.seed(70 + (0 if n_kv is None else n_kv))
    cfg = _make_cfg(n_kv)
    tr = TinyTrainer(cfg)
    ids = rng.get_rng().integers(0, cfg.vocab_size, size=(2, 8))
    out_trainer = np.array(tr.forward(ids).data)

    model = Transformer.load(tr.save(str(tmp_path / "m")), verify=True)
    out_infer = model.forward(ids)
    diff = float(np.max(np.abs(out_trainer - out_infer)))
    assert diff <= 1e-9, f"trainer 与推理端前向不一致: max|diff|={diff:.3e}"


@pytest.mark.parametrize("n_kv", [None, 2, 1])          # MHA / GQA / MQA
@pytest.mark.parametrize("scaling", [1.0, 2.0, 4.0])     # NTK-aware 拉伸因子
def test_rope_scaling_matches_inference(tmp_path, n_kv, scaling):
    """训练端 RoPE (含 rope_scaling) 与推理端 apply_rope 须逐位一致 —
    否则 save→load 后可复现性破坏 (v0.7.1 修复: trainer 旧实现忽略 rope_scaling)。"""
    rng.seed(80 + (0 if n_kv is None else n_kv))
    cfg = _make_cfg(n_kv)
    cfg.rope_scaling = scaling
    tr = TinyTrainer(cfg)
    ids = rng.get_rng().integers(0, cfg.vocab_size, size=(2, 8))
    out_trainer = np.array(tr.forward(ids).data)

    model = Transformer.load(tr.save(str(tmp_path / "m")), verify=True)
    out_infer = model.forward(ids)
    diff = float(np.max(np.abs(out_trainer - out_infer)))
    assert diff <= 1e-9, (
        f"rope_scaling={scaling} 训练/推理 RoPE 分叉: max|diff|={diff:.3e}")


# ---------------- 2. 注意力参数梯度非零 ----------------

@pytest.mark.parametrize("n_kv", [None, 2, 1])
def test_attention_params_receive_gradients(n_kv):
    """一步 backward 后 attn_norm/q/k/v/o_proj 梯度全部非 None 且范数 > 0。"""
    rng.seed(71)
    cfg = _make_cfg(n_kv)
    tr = TinyTrainer(cfg)
    ids = rng.get_rng().integers(0, cfg.vocab_size, size=(2, 8))
    tgt = rng.get_rng().integers(0, cfg.vocab_size, size=(2, 8))

    _zero_grads(tr)
    tr.loss(ids, tgt).backward()

    for key in ["layer0.attn_norm", "layer0.q_proj", "layer0.k_proj",
                "layer0.v_proj", "layer0.o_proj", "layer1.q_proj"]:
        g = tr.params[key].grad
        assert g is not None, f"{key} 梯度为 None (梯度链断裂回归!)"
        norm = float(np.linalg.norm(g))
        assert norm > 0.0, f"{key} 梯度范数为 0 (梯度链断裂回归!)"


# ---------------- 3. 有限差分梯度校验 ----------------

def _central_diff(trainer: TinyTrainer, key: str, idx, ids, tgt,
                  eps: float = 1e-5) -> float:
    """对 param[key][idx] 做中心差分 (逐点恢复, 前向确定性)。

    eps=1e-5: 小梯度点 (~1e-6) 的差分噪声 ~1e-15/2eps ≈ 5e-11, 远小于信号。
    """
    param = trainer.params[key]
    old = param.data[idx]
    param.data[idx] = old + eps
    lp = float(trainer.loss(ids, tgt).data)
    param.data[idx] = old - eps
    lm = float(trainer.loss(ids, tgt).data)
    param.data[idx] = old
    return (lp - lm) / (2.0 * eps)


@pytest.mark.parametrize("n_kv", [2, 1])
def test_gradcheck_attention_params_finite_difference(n_kv):
    """注意力参数解析梯度 vs 中心差分: 相对误差 < 1e-5 (INV-22 同口径)。"""
    rng.seed(72)
    cfg = _make_cfg(n_kv)
    tr = TinyTrainer(cfg)
    ids = rng.get_rng().integers(0, cfg.vocab_size, size=(2, 6))
    tgt = rng.get_rng().integers(0, cfg.vocab_size, size=(2, 6))

    _zero_grads(tr)
    tr.loss(ids, tgt).backward()

    samples = [
        ("layer0.attn_norm", 0),
        ("layer0.q_proj", (0, 0)),
        ("layer0.k_proj", (0, 0)),
        ("layer0.v_proj", (0, 0)),
        ("layer0.o_proj", (0, 0)),
        ("layer1.q_proj", (-1, -1)),
        ("layer1.v_proj", (-1, -1)),
    ]
    for key, idx in samples:
        analytic = float(tr.params[key].grad[idx])
        numeric = _central_diff(tr, key, idx, ids, tgt)
        rel = abs(analytic - numeric) / max(abs(analytic), abs(numeric), 1e-8)
        # 小梯度点差分噪声主导, 放行 abs 判据 (abs diff < 1e-8 ⇒ 一致)
        assert rel < 1e-5 or abs(analytic - numeric) < 1e-8, (
            f"{key}{idx} 梯度校验失败: analytic={analytic:.6e} "
            f"numeric={numeric:.6e} rel_err={rel:.3e}")


def test_causal_mask_blocks_future_positions():
    """因果偏置: softmax 后未来位置注意力权重严格为 0 (INV-11 训练端)。"""
    bias = TinyTrainer._causal_bias(5)
    assert bias[0, 0] == 0.0 and bias[4, 4] == 0.0
    assert (bias[np.triu_indices(5, 1)] == -1e30).all()
    assert (bias[np.tril_indices(5)] == 0.0).all()
