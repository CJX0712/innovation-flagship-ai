"""v0.7.10 训练端 INT8 量化感知训练 (QAT, Jacob et al. 2018) — 防回归。

历史缺口: INT8 量化此前**仅推理端可用** (`Transformer.quantize_int8`,
LLM.int8 per-channel RTN)。量化故事残缺 — QAT (STE 伪量化) 让训练期前向
直接"看见"量化扰动, 训出的权重天然适配 INT8 部署。

本文件锁定:
1. STE 节点前向逐位 == dequant(quant(W)) (与推理端同 per-channel RTN 方案);
2. INV-07 重建误差界 ≤ scale/2 在 STE 节点上成立;
3. STE 反向 = 恒等直通 (含伪量化图 vs 无伪量化图梯度逐位一致);
4. use_qat=True 确实注入伪量化 (同权重前向出现确定性扰动) + 梯度贯通;
5. QAT 训练模型的 INT8 部署误差 ≤ 非 QAT 基线 (核心验收: QAT 不劣化部署)。
"""
from __future__ import annotations

import numpy as np
import pytest

from ifai import rng
from ifai.core import quant
from ifai.models.config import ModelConfig
from ifai.models.transformer import Transformer
from ifai.train.autodiff import Tensor
from ifai.train.trainer import TinyTrainer


# ---------------- 1. STE 前向 = dequant(quant(W)) ----------------

def test_fake_quant_matches_dequant():
    """STE 节点前向必须逐位等于推理端同方案的 quantize→dequantize 往返。"""
    rng.seed(1)
    W = rng.get_rng().normal(0, 0.05, size=(16, 12))
    got = Tensor(W).fake_quant_int8_per_channel().data
    q, scale = quant.quantize_int8_per_channel(W)
    want = quant.dequantize_int8(q, scale)
    assert np.array_equal(got, want), "STE 前向与 dequant(quant(W)) 不逐位一致"


# ---------------- 2. INV-07 重建误差界 (STE 节点上) ----------------

def test_fake_quant_inv07_bound():
    """逐通道重建误差 |W − dequant| ≤ scale/2 + 1e-9 (INV-07 在 STE 节点复刻)。"""
    rng.seed(2)
    W = rng.get_rng().normal(0, 0.05, size=(16, 12))
    node = Tensor(W).fake_quant_int8_per_channel()
    q, scale = quant.quantize_int8_per_channel(W)
    err = np.abs(W - node.data)
    bound = scale / 2 + 1e-9
    assert (err <= bound).all(), (
        f"INV-07 违反: max|W-dequant|={err.max():.3e} > scale/2 (max)={bound.max():.3e}")


# ---------------- 3. STE 反向 = 恒等直通 ----------------

def test_fake_quant_ste_gradient_identity():
    """含伪量化图与无伪量化图对 W 的梯度必须一致 (STE = 恒等):
    y = sum(x @ Wq), dW 应等于无伪量化时 x@W 的解析梯度 x.sum(axis=0) 广播。"""
    rng.seed(3)
    x_np = rng.get_rng().normal(0, 1.0, size=(4, 6))     # [B, in]
    w_np = rng.get_rng().normal(0, 0.05, size=(6, 5))    # [in, out]

    # 含伪量化图
    x1 = Tensor(x_np, requires_grad=False)
    w1 = Tensor(w_np.copy())
    (x1.matmul(w1.fake_quant_int8_per_channel())).sum().backward()

    # 无伪量化参照图 (同数据)
    x2 = Tensor(x_np, requires_grad=False)
    w2 = Tensor(w_np.copy())
    (x2.matmul(w2)).sum().backward()

    assert w1.grad is not None and w2.grad is not None
    assert np.allclose(w1.grad, w2.grad, atol=1e-12), (
        "STE 反向与无伪量化梯度不一致 (恒等直通被破坏)")
    # 解析二次验证: dW[i,o] = Σ_b x[b,i]
    expected = np.tile(x_np.sum(axis=0)[:, None], (1, 5))
    assert np.allclose(w1.grad, expected, atol=1e-12), "STE 梯度偏离解析解"


# ---------------- 4. use_qat=True 注入伪量化 + 梯度贯通 ----------------

def _tiny_cfg(**kw) -> ModelConfig:
    base = dict(vocab_size=48, d_model=16, n_layers=2, n_heads=4, n_kv_heads=2,
                ffn_mult=2, max_seq_len=32, tie_word_embeddings=True)
    base.update(kw)
    return ModelConfig(**base)


def test_qat_injects_fake_quant():
    """同权重下 use_qat=True 前向与 use_qat=False 存在确定性量化扰动
    (证明伪量化确实接入), 且 QAT 路径梯度贯通 (有限且非零)。"""
    rng.seed(40)
    tr_q = TinyTrainer(_tiny_cfg(use_qat=True))
    rng.seed(40)
    tr_f = TinyTrainer(_tiny_cfg(use_qat=False))
    # 同 seed ⇒ 同初始化权重
    for k, v in tr_f.params.items():
        assert np.array_equal(np.asarray(tr_q.params[k].data), np.asarray(v.data)), \
            f"种子对齐失败: {k}"
    ids = rng.get_rng().integers(0, 48, size=(2, 8))
    out_q = np.array(tr_q.forward(ids).data)
    out_f = np.array(tr_f.forward(ids).data)
    diff = float(np.max(np.abs(out_q - out_f)))
    assert diff > 0.0, "use_qat=True 前向与关闭时完全一致 — 伪量化未被注入!"

    # QAT 路径梯度贯通
    tgt = rng.get_rng().integers(0, 48, size=(2, 8))
    tr_q.loss(ids, tgt).backward()
    for key in ["layer0.q_proj", "layer0.ffn_gate", "embed"]:
        g = tr_q.params[key].grad
        assert g is not None, f"{key} QAT 梯度为 None (STE 断裂!)"
        assert np.all(np.isfinite(g)), f"{key} QAT 梯度含非有限值"
        assert float(np.linalg.norm(g)) > 0.0, f"{key} QAT 梯度范数为 0 (STE 断裂!)"


# ---------------- 5. QAT 模型 INT8 部署精度 (核心验收) ----------------

def _nll(logits: np.ndarray, tgt: np.ndarray) -> float:
    """数值稳定 NLL (logsumexp)。"""
    m = logits.max(-1, keepdims=True)
    lse = m + np.log(np.exp(logits - m).sum(-1, keepdims=True))
    logp = logits - lse
    B, T, _ = logits.shape
    return float(-logp[np.arange(B)[:, None], np.arange(T)[None, :], tgt].mean())


def test_qat_int8_deploy_accuracy(tmp_path):
    """QAT 训练模型 vs 非 QAT 基线: 各自 save→load→quantize_int8 部署,
    比较**部署后损失劣化** (int8_loss − fp64_loss, 模型内自比较, 同一评估批):

    - 基线: fp64 前向是被优化的函数, int8 部署是扰动 ⇒ 劣化 > 0;
    - QAT:  训练优化的正是量化前向 dequant(quant(W)), 与部署 quantize_int8
      同方案 ⇒ int8 前向即训练时行为, 劣化 ≈ 0 或为负。

    验收: QAT 的部署劣化 ≤ 基线 (QAT 不得劣化 INT8 部署), 且两者均有界。
    """
    cfg_kw = dict(vocab_size=48, d_model=16, n_layers=2, n_heads=4, n_kv_heads=2,
                  ffn_mult=2, max_seq_len=32, tie_word_embeddings=True)
    rng.seed(700)
    eval_ids = rng.get_rng().integers(0, 48, size=(2, 8))   # 共享评估批 (公平)
    eval_tgt = rng.get_rng().integers(0, 48, size=(2, 8))

    drops = {}
    for name, seed in [("qat", 71), ("base", 72)]:
        rng.seed(seed)
        cfg = ModelConfig(**cfg_kw, use_qat=(name == "qat"))
        tr = TinyTrainer(cfg)
        corp = rng.get_rng().integers(0, cfg.vocab_size, size=300)
        tr.train(corp, steps=250, lr=3e-3, batch_size=4, seq_len=16)
        path = tr.save(str(tmp_path / name))
        m = Transformer.load(path)
        loss_fp = _nll(np.asarray(m.forward(eval_ids)), eval_tgt)
        m.quantize_int8()
        loss_int8 = _nll(np.asarray(m.forward(eval_ids)), eval_tgt)
        assert np.isfinite(loss_fp) and np.isfinite(loss_int8)
        drops[name] = loss_int8 - loss_fp

    assert drops["qat"] <= drops["base"] + 1e-9, (
        f"QAT 劣化 INT8 部署: qat_drop={drops['qat']:.4f} > "
        f"base_drop={drops['base']:.4f}")
    assert abs(drops["qat"]) <= 0.5 and abs(drops["base"]) <= 0.5, (
        f"部署劣化超界: qat={drops['qat']:.4f} base={drops['base']:.4f}")
