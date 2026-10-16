# L4 · 训练 / 微调 (Training & Fine-tuning) — 设计说明

## 设计目标

L4 提供 **ifai** 的训练能力：手写反向模式自动微分、LoRA 低秩适配、MoE 稀疏专家
可微训练、QAT 量化感知训练（STE 伪量化）、AdamW 优化器与调度、以及一个可在数十秒内
于 CPU 上训完微型模型并产出与 L2 兼容权重文件的 `TinyTrainer`。目标是在
「零深度学习框架」前提下，证明训练管线的正确性。

本层覆盖 `ifai/train/`（autodiff.py、lora.py、optim.py、trainer.py）。

## 复用世界顶级成果（零重写 · 集大成）

- **反向模式自动微分**（Griewank & Walther 2008，"Evaluating Derivatives"）：复用
  计算图 + 拓扑排序反向遍历的标准反向模式；每个算子实现精确解析梯度（matmul/add/
  mul/sigmoid/tanh/relu/exp/log/mean/sum/transpose/pow/getitem/concat 等），
  `_unbroadcast` 处理 NumPy 广播反向。INV-22：手写反向 vs 有限差分校验。
- **LoRA 低秩适配**（Hu et al. 2022，"LoRA: Low-Rank Adaptation of Large Language
  Models"）：复用 `W_eff = W + (α/r)·B·A`（B∈R^{d×r}, A∈R^{r×k}）；工程上冻结原权重
  只训 A/B，可训练参数量 `(d+r)·k ≪ d·k`（通常 ≤5%）。INV-24：merge 后与未 merge
  前向一致（≤1e-6），`B=0` 时 == 原权重。训练端 `_apply_lora` 与推理端同公式接入
  q/k/v/o 四投影，save→load 往返 ≤1e-9。
- **MoE / Switch 稀疏专家可微训练**（Fedus et al. 2022，"Switch Transformers"）：
  训练端 `_moe_forward` 严格镜像推理端 `Transformer._moe`（top-k 路由 → 仅对选中
  专家 router 分数 softmax ⇒ 概率和=1，INV-14）；反向经 `getitem` 散射回 router、
  经专家 matmul 回各专家权重。`use_moe=True` 模型训练/推理两端同名同结构。
- **QAT 量化感知训练**（Jacob et al. 2018，"Quantization and Training of Neural
  Networks for Efficient Integer-Arithmetic-Only Inference"）：`use_qat=True` 时
  训练期对全部 2D 线性权重做 STE 伪量化（`Tensor.fake_quant_int8_per_channel`：
  前向 = dequant(quant(W)) 复用 L0 同一 per-channel RTN 方案，反向恒等直通忽略
  d(scale)/dW）——训练出的权重天然适配推理端 `quantize_int8` 部署，部署后损失
  劣化 ≤ 非 QAT 基线（黄金测试锁定）。
- **AdamW 解耦权重衰减**（Loshchilov & Hutter 2019，ICLR）：复用解耦权重衰减更新
  `p -= lr·(mhat/(√vhat+eps) + wd·p)`；工程上 `weight_decay=0` 时 == 标准 Adam
  （INV-21）。偏差校正 `bc1/bc2` 严格按解析公式。
- **余弦退火 + 线性预热**（Loshchilov & Hutter 2016，SGDR / HuggingFace
  `get_cosine_schedule_with_warmup`）：复用「线性预热 + 余弦退火到 min_lr」解析调度
  （R-L4-02：与解析公式一致到 1e-9）。
- **全局范数梯度裁剪**（Pascanu et al. 2013，"On the difficulty of training recurrent
  neural networks"）：复用按全局 L2 范数统一缩放；INV-23：裁剪后范数 ≤ max_norm+1e-6，
  未超限时梯度 bitwise 不变。
- **nanogpt 极简训练循环范式**（Karpathy，"nanogpt"）：复用「随机 batch → forward →
  backward → clip → step」的极简循环作为 `TinyTrainer` 骨架，产出与 L2 同结构权重。

## 关键不变量 / 验收

- **INV-21** AdamW 单步 == 解析公式（≤ 1e-6）；`weight_decay=0` 时 == Adam。
- **INV-22** 手写反向模式梯度 == 有限差分梯度（自动微分正确性对照）。
- **INV-23** 全局范数裁剪后 `‖g‖ ≤ max_norm + 1e-6`；未超限时梯度不变。
- **INV-24** LoRA `merge(W + scale·B@A)` 与未 merge 前向一致（≤ 1e-6）。
- **INV-25** 训练产出参数文件与 `ifai.models.Transformer` 同名同结构（可互加载）。
- **INV-26** 训练过程中损失单调下降（微型模型数十秒内可见）。

### 训练/推理一致性三层保证（黄金测试锁定）

1. **五轴单轴一致性**：注意力(n_kv×rope_scaling) / 激活(swiglu|gelu) / MoE / INT8 /
   LoRA —— 各轴 `TinyTrainer` 前向 vs 推理端 `Transformer.load(tr.save(...))` 前向
   `max|diff| ≤ 1e-9`（`tests/unit/l4/test_attention_grad.py` / `test_moe_consistency.py`
   / `test_lora_consistency.py` / `tests/test_core.py`）。
2. **特征组合矩阵**（`test_composition_consistency.py`）：MHA/GQA × {MoE, LoRA,
   MoE+LoRA} × {swiglu, gelu} × {绑定, 未绑定} 13 组合 —— 参数集严格对齐 +
   前向 save→load ≤1e-9 + LoRA B=0 组合等价基座（≤1e-12, INV-24）。
3. **端到端可训性**（`test_feature_learnability.py`）：8 个特征配置在真实中文语料
   真实训练，INV-26 损失单调性（平滑尾部 20% 均值 ≤ 头部 20% 均值 × 0.8）全部达标
   —— 证明 MoE 路由 / LoRA 低秩增量 / QAT 伪量化确实参与并推动学习（一致性 ≠ 可训性）。

## 工程取舍

1. **自研 autograd 而非 PyTorch**：为在「零框架」约束下仍可训练，并让 INV-22 成为
   梯度正确性的黄金基准；代价是速度，故仅用于微型模型。
2. **TinyTrainer 与 Transformer 参数同构**：`to_numpy_params()` 产出与 L2 完全一致的
   key 集合，`ifai demo` 可直接加载训练结果推理，形成「训练→推理」闭环。
3. **RoPE 在 autograd 中可微表达**：`_rope` 用可微算子实现偶数维旋转并在反向时按位
   分流，确保位置编码的梯度能正确回传（而非 detach 近似）。
4. **CPU 优先的微型化**：默认 `seq_len=32, batch=8, steps=200`，显存与耗时均适配
   笔记本级 CPU，契合本栈教学/参考定位。
