"""v0.7.5 TinyTrainer MoE 支持 + 训练/推理 save→load 一致性 (防回归)。

历史缺口: TinyTrainer._build/_layer_forward 此前仅建 dense FFN 参数,
``use_moe=True`` 模型**无法训练** (推理端 ``Transformer._moe`` 存在但训练端
无对应路径), train→inference 对 MoE 配置轴从未被测试触达。本文件填补:

1. MoE 模型 trainer 前向与推理端 ``Transformer`` (加载同参数) 输出
   ``max|diff| ≤ 1e-9`` (swiglu + gelu 两种激活);
2. MoE 路由/专家参数一步 backward 后梯度非 None 且范数 > 0 (梯度链贯通,
   router 经 getitem 散射、专家经 matmul 回传)。
"""
from __future__ import annotations

import numpy as np
import pytest

from ifai import rng
from ifai.models.config import ModelConfig
from ifai.models.transformer import Transformer
from ifai.train.trainer import TinyTrainer


def _make_moe_cfg(n_kv: int = 2, activation: str = "swiglu") -> ModelConfig:
    return ModelConfig(
        vocab_size=48, d_model=16, n_layers=2, n_heads=4,
        n_kv_heads=n_kv, ffn_mult=2, max_seq_len=32,
        tie_word_embeddings=True,
        use_moe=True, n_experts=4, moe_topk=2,
        activation=activation,
    )


def _zero_grads(tr: TinyTrainer) -> None:
    for p in tr.params.values():
        p.grad = np.zeros_like(p.data)


# ---------------- 1. 前向数学一致性 (MoE trainer vs 推理端) ----------------

@pytest.mark.parametrize("activation", ["swiglu", "gelu"])
def test_moe_forward_matches_inference(tmp_path, activation):
    """TinyTrainer(MoE).forward 与 Transformer (加载同参数) 输出 max|diff| ≤ 1e-9。"""
    rng.seed(90 if activation == "swiglu" else 91)
    cfg = _make_moe_cfg(activation=activation)
    tr = TinyTrainer(cfg)
    ids = rng.get_rng().integers(0, cfg.vocab_size, size=(2, 8))
    out_trainer = np.array(tr.forward(ids).data)

    model = Transformer.load(tr.save(str(tmp_path / "m")), verify=True)
    out_infer = model.forward(ids)
    diff = float(np.max(np.abs(out_trainer - out_infer)))
    assert diff <= 1e-9, (
        f"MoE 训练/推理前向不一致 ({activation}): max|diff|={diff:.3e}")


# ---------------- 2. MoE 参数梯度贯通 ----------------

def test_moe_params_receive_gradients():
    """一步 backward 后 router / 专家 gate/up/down 梯度全部非 None 且范数 > 0。"""
    rng.seed(92)
    cfg = _make_moe_cfg()
    tr = TinyTrainer(cfg)
    ids = rng.get_rng().integers(0, cfg.vocab_size, size=(2, 8))
    tgt = rng.get_rng().integers(0, cfg.vocab_size, size=(2, 8))

    _zero_grads(tr)
    tr.loss(ids, tgt).backward()

    for key in ["layer0.router", "layer0.e0_gate", "layer0.e0_up",
                "layer0.e0_down", "layer1.e3_down", "layer1.e2_gate"]:
        g = tr.params[key].grad
        assert g is not None, f"{key} 梯度为 None (MoE 梯度链断裂回归!)"
        assert float(np.linalg.norm(g)) > 0.0, f"{key} 梯度范数为 0 (MoE 梯度链断裂回归!)"
