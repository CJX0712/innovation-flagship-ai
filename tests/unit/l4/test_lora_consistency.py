"""v0.7.7 LoRA 接入 + 训练/推理 save→load 一致性 (防回归)。

历史缺口: ``ifai/train/lora.py::LoRALinear`` 仅是独立参考实现, 未接入 TinyTrainer
与推理端 ``Transformer`` —— LoRA (Hu et al. 2022) 这一旗舰级微调方法此前**不可用**。
本文件填补: 训练端/推理端按同一公式 ``W + (α/r)·B@A`` 应用 LoRA, 并经
trainer.save → Transformer.load 往返对照 (INV-24)。
"""
from __future__ import annotations

import numpy as np
import pytest

from ifai import rng
from ifai.models.config import ModelConfig
from ifai.models.transformer import Transformer
from ifai.train.trainer import TinyTrainer


def _make_lora_cfg(n_kv: int = 2, activation: str = "swiglu") -> ModelConfig:
    return ModelConfig(
        vocab_size=48, d_model=16, n_layers=2, n_heads=4, n_kv_heads=n_kv,
        ffn_mult=2, max_seq_len=32, tie_word_embeddings=True,
        activation=activation, use_lora=True, lora_r=4, lora_alpha=8,
    )


def _zero_grads(tr: TinyTrainer) -> None:
    for p in tr.params.values():
        if p.grad is not None:
            p.grad = np.zeros_like(p.data)


# ---------------- 1. 前向数学一致性 (LoRA trainer vs 推理端) ----------------

@pytest.mark.parametrize("activation", ["swiglu", "gelu"])
def test_lora_forward_matches_inference(tmp_path, activation):
    """LoRA 模型训练后 trainer 前向与推理端 Transformer (加载同参数) 输出
    ``max|diff| ≤ 1e-9`` (swiglu / gelu)；先训练数步使 A/B 脱离 B=0 初始。"""
    rng.seed(130 if activation == "swiglu" else 131)
    cfg = _make_lora_cfg(activation=activation)
    tr = TinyTrainer(cfg)
    ids = rng.get_rng().integers(0, cfg.vocab_size, size=(2, 8))
    corpus = rng.get_rng().integers(0, cfg.vocab_size, size=300)
    tr.train(corpus, steps=20, lr=1e-2, batch_size=4, seq_len=16)
    out_trainer = np.array(tr.forward(ids).data)

    model = Transformer.load(tr.save(str(tmp_path / "m")), verify=True)
    out_infer = model.forward(ids)
    diff = float(np.max(np.abs(out_trainer - out_infer)))
    assert diff <= 1e-9, f"LoRA 训练/推理前向不一致 ({activation}): max|diff|={diff:.3e}"


# ---------------- 2. LoRA 梯度贯通 + 基座冻结 ----------------

def test_lora_gradient_flow_and_frozen_base():
    """训练数步 (使 B 脱离 0) 后一步 backward: A/B 梯度非 None 且范数 > 0
    (可训练); LoRA 目标基座 (q_proj 等) 被冻结 ⇒ 梯度为 None (LoRA 语义)。

    注: B=0 初始时 ∂L/∂A ≡ 0 (A 被 B 乘掉), 故需先训练令 B 非零。
    """
    rng.seed(132)
    cfg = _make_lora_cfg()
    tr = TinyTrainer(cfg)
    corpus = rng.get_rng().integers(0, cfg.vocab_size, size=300)
    tr.train(corpus, steps=10, lr=1e-2, batch_size=4, seq_len=16)
    ids = rng.get_rng().integers(0, cfg.vocab_size, size=(2, 6))
    tgt = rng.get_rng().integers(0, cfg.vocab_size, size=(2, 6))

    _zero_grads(tr)  # 仅清零可训练参数 (A/B); 冻结基座不在内
    tr.loss(ids, tgt).backward()

    for key in ["layer0.q_proj_lora_a", "layer0.q_proj_lora_b",
                "layer1.o_proj_lora_a", "layer1.o_proj_lora_b"]:
        g = tr.params[key].grad
        assert g is not None, f"{key} LoRA 梯度为 None"
        assert float(np.linalg.norm(g)) > 0.0, f"{key} LoRA 梯度范数为 0"
    # 基座被冻结 (LoRA 语义): backward 不累积梯度 (requires_grad=False),
    # 故其 grad 恒为全 0 (或 None), 不存在任何梯度流。
    g_q = tr.params["layer0.q_proj"].grad
    g_o = tr.params["layer1.o_proj"].grad
    assert (g_q is None or float(np.linalg.norm(g_q)) == 0.0), "基座 q_proj 应冻结 (无梯度)"
    assert (g_o is None or float(np.linalg.norm(g_o)) == 0.0), "基座 o_proj 应冻结 (无梯度)"


# ---------------- 3. INV-24: B=0 初始等价基座前向 ----------------

def test_lora_zero_init_equals_base():
    """B=0 初始时 LoRA 增量 ≡ 0 ⇒ 前向与纯基座 (use_lora=False) 等价 (INV-24)。"""
    rng.seed(133)
    base_cfg = _make_lora_cfg()
    base_cfg.use_lora = False
    lora_cfg = _make_lora_cfg()
    tr_base = TinyTrainer(base_cfg)
    tr_lora = TinyTrainer(lora_cfg)
    # 复制同一基座权重到 LoRA 模型 (确保除 LoRA 增量外完全一致)
    for k in ["embed", "final_norm", "layer0.attn_norm", "layer0.q_proj",
              "layer0.k_proj", "layer0.v_proj", "layer0.o_proj", "layer0.ffn_norm",
              "layer0.ffn_gate", "layer0.ffn_up", "layer0.ffn_down",
              "layer1.attn_norm", "layer1.q_proj", "layer1.k_proj", "layer1.v_proj",
              "layer1.o_proj", "layer1.ffn_norm", "layer1.ffn_gate",
              "layer1.ffn_up", "layer1.ffn_down"]:
        tr_lora.params[k].data = np.array(tr_base.params[k].data)
    ids = rng.get_rng().integers(0, base_cfg.vocab_size, size=(2, 8))
    out_base = np.array(tr_base.forward(ids).data)
    out_lora = np.array(tr_lora.forward(ids).data)
    diff = float(np.max(np.abs(out_base - out_lora)))
    assert diff <= 1e-12, f"LoRA B=0 初始未等价基座: max|diff|={diff:.3e}"
