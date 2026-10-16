"""v0.7.8 特征组合一致性矩阵 (防回归)。

历史: v0.7.5 / v0.7.6 / v0.7.7 分别将 MoE / INT8 / LoRA 接入训练端+推理端, 并
各自单轴验证 train→inference save→load ``max|diff| ≤ 1e-9``。但**特征组合**
(MoE+LoRA / GQA+MoE / GQA+LoRA / 激活×MoE / 未绑定×MoE+LoRA 等) 从未被测试
触达 —— 这是五轴一致性矩阵最后的盲区。

经 Recon 确认: ``TinyTrainer._build`` 与 ``Transformer._init_params`` 对组合
配置的**参数名由 flag 组合唯一决定** (顺序不同无影响), 且两端 ``_layer_forward``
/ ``forward`` 均对 q/k/v/o 四投影应用 LoRA、均支持 MoE 分支 ⇒ MoE+LoRA 等组合
在两端**真实可用**。本文件将其锁进测试, 防止未来单侧改动某组合的参数命名导致
save→load 静默分叉。

锁定三件事:
1. 参数集严格对齐: 任意组合 ``set(TinyTrainer.params) == set(Transformer.params)``;
2. 前向数学一致性: 训练端前向 vs 推理端 (加载同参数) 输出 ``max|diff| ≤ 1e-9``;
3. INV-24 组合保持: 含 LoRA 的组合在 B=0 初始时仍严格等价对应基座 (≤1e-12)。
"""
from __future__ import annotations

import numpy as np
import pytest

from ifai import rng
from ifai.models.config import ModelConfig
from ifai.models.transformer import Transformer
from ifai.train.trainer import TinyTrainer


def _cfg(moe: bool, lora: bool, n_kv: int, activation: str, tie: bool) -> ModelConfig:
    """构造组合配置 (确定性, 不依赖 PYTHONHASHSEED)。"""
    return ModelConfig(
        vocab_size=48, d_model=16, n_layers=2, n_heads=4, n_kv_heads=n_kv,
        ffn_mult=2, max_seq_len=32,
        tie_word_embeddings=tie,
        use_moe=moe, n_experts=4 if moe else 0, moe_topk=2 if moe else 1,
        activation=activation,
        use_lora=lora, lora_r=4, lora_alpha=8,
    )


def _seed_for(moe: bool, lora: bool, n_kv: int, activation: str, tie: bool) -> int:
    """稳定整数种子 (避开 hash() 受 PYTHONHASHSEED 影响)。"""
    return (1000 + (1 if moe else 0) * 100000 + (1 if lora else 0) * 10000
            + n_kv * 1000 + (1 if activation == "gelu" else 0) * 100
            + (1 if tie else 0) * 10)


# 组合矩阵: (moe, lora, n_kv, activation, tie)
# 覆盖三特征轴 × 激活 × GQA × 绑定 的交叉面, 含核心新组合 MoE+LoRA。
COMBOS = [
    (False, False, 4, "swiglu", True),   # 基线 MHA
    (False, False, 2, "swiglu", True),   # 基线 GQA
    (False, False, 4, "gelu",   True),   # 基线 gelu
    (True,  False, 4, "swiglu", True),   # MoE MHA
    (True,  False, 2, "swiglu", True),   # MoE GQA
    (True,  False, 4, "gelu",   True),   # MoE gelu
    (False, True,  4, "swiglu", True),   # LoRA MHA
    (False, True,  2, "swiglu", True),   # LoRA GQA
    (False, True,  4, "gelu",   True),   # LoRA gelu
    (True,  True,  4, "swiglu", True),   # MoE+LoRA MHA  (核心新组合)
    (True,  True,  2, "swiglu", True),   # MoE+LoRA GQA  (核心新组合)
    (True,  True,  4, "gelu",   True),   # MoE+LoRA gelu (核心新组合)
    (True,  True,  2, "gelu",   False),  # MoE+LoRA GQA gelu 未绑定 (全组合)
]


# ---------------- 1. 参数集严格对齐 ----------------

@pytest.mark.parametrize("moe,lora,n_kv,activation,tie", COMBOS)
def test_composition_param_parity(moe, lora, n_kv, activation, tie):
    """任意组合下训练端与推理端参数键集合完全相同 (防单侧命名漂移)。"""
    cfg = _cfg(moe, lora, n_kv, activation, tie)
    tr_keys = set(TinyTrainer(cfg).params.keys())
    tf_keys = set(Transformer(cfg).params.keys())
    assert tr_keys == tf_keys, (
        f"参数集不一致 (moe={moe},lora={lora},n_kv={n_kv},{activation},tie={tie}):\n"
        f"  trainer-only={sorted(tr_keys - tf_keys)}\n"
        f"  infer-only={sorted(tf_keys - tr_keys)}")
    # 必备结构断言: 组合须真实携带各自特征参数 (防止开关被静默忽略)
    if moe:
        assert any(k.endswith(".router") for k in tr_keys), "MoE 组合缺 router"
        assert any(".e0_gate" in k for k in tr_keys), "MoE 组合缺专家权重"
    if lora:
        assert any(k.endswith("_lora_a") for k in tr_keys), "LoRA 组合缺 lora_a"
        assert any(k.endswith("_lora_b") for k in tr_keys), "LoRA 组合缺 lora_b"
    if not moe:
        assert any(k.endswith(".ffn_gate") for k in tr_keys), "dense 组合缺 ffn_gate"


# ---------------- 2. 前向数学一致性 (save→load) ----------------

@pytest.mark.parametrize("moe,lora,n_kv,activation,tie", COMBOS)
def test_composition_forward_save_load(tmp_path, moe, lora, n_kv, activation, tie):
    """训练端前向 vs 推理端 (加载同参数) 输出 ``max|diff| ≤ 1e-9``。"""
    rng.seed(_seed_for(moe, lora, n_kv, activation, tie))
    cfg = _cfg(moe, lora, n_kv, activation, tie)
    tr = TinyTrainer(cfg)
    ids = rng.get_rng().integers(0, cfg.vocab_size, size=(2, 8))
    out_trainer = np.array(tr.forward(ids).data)
    model = Transformer.load(tr.save(str(tmp_path / "m")), verify=True)
    out_infer = model.forward(ids)
    diff = float(np.max(np.abs(out_trainer - out_infer)))
    assert diff <= 1e-9, (
        f"组合前向不一致 (moe={moe},lora={lora},n_kv={n_kv},{activation},tie={tie}): "
        f"max|diff|={diff:.3e}")


# ---------------- 3. INV-24: LoRA B=0 组合等价基座 ----------------

@pytest.mark.parametrize("moe,n_kv,activation,tie", [
    (False, 4, "swiglu", True),   # LoRA 单独 (MHA)
    (True,  2, "swiglu", True),   # MoE+LoRA (GQA)
    (True,  2, "gelu",   False),  # MoE+LoRA (GQA, gelu, 未绑定) 全组合
])
def test_composition_lora_zero_init_equals_base(moe, n_kv, activation, tie):
    """含 LoRA 的组合在 B=0 初始时, 前向与关 LoRA 的同配置基座等价 (INV-24)。"""
    rng.seed(777)
    lora_cfg = _cfg(moe, True, n_kv, activation, tie)
    base_cfg = _cfg(moe, False, n_kv, activation, tie)
    tr_lora = TinyTrainer(lora_cfg)
    tr_base = TinyTrainer(base_cfg)
    # 复制同一基座权重 (确保除 LoRA 增量外完全一致)
    shared = [k for k in tr_base.params if not k.endswith(("_lora_a", "_lora_b"))]
    for k in shared:
        tr_lora.params[k].data = np.array(tr_base.params[k].data)
    ids = rng.get_rng().integers(0, base_cfg.vocab_size, size=(2, 8))
    out_base = np.array(tr_base.forward(ids).data)
    out_lora = np.array(tr_lora.forward(ids).data)
    diff = float(np.max(np.abs(out_base - out_lora)))
    assert diff <= 1e-12, f"LoRA B=0 组合未等价基座: max|diff|={diff:.3e}"
