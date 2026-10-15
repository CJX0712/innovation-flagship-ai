# L4 · 训练 / 微调 (Training & Fine-tuning) — 设计说明

## 设计目标

L4 提供 **ifai** 的训练能力：手写反向模式自动微分、LoRA 低秩适配、AdamW 优化器与
调度、以及一个可在数十秒内于 CPU 上训完微型模型并产出与 L2 兼容权重文件的
`TinyTrainer`。目标是在「零深度学习框架」前提下，证明训练管线的正确性。

本层覆盖 `ifai/train/`（autodiff.py、lora.py、optim.py、trainer.py）。

## 复用世界顶级成果（零重写 · 集大成）

- **反向模式自动微分**（Griewank & Walther 2008，"Evaluating Derivatives"）：复用
  计算图 + 拓扑排序反向遍历的标准反向模式；每个算子实现精确解析梯度（matmul/add/
  mul/sigmoid/tanh/relu/exp/log/mean/sum/transpose/pow/getitem/concat 等），
  `_unbroadcast` 处理 NumPy 广播反向。INV-22：手写反向 vs 有限差分校验。
- **LoRA 低秩适配**（Hu et al. 2022，"LoRA: Low-Rank Adaptation of Large Language
  Models"）：复用 `W_eff = W + (α/r)·B·A`（B∈R^{d×r}, A∈R^{r×k}）；工程上冻结原权重
  只训 A/B，可训练参数量 `(d+r)·k ≪ d·k`（通常 ≤5%）。INV-24：merge 后与未 merge
  前向一致（≤1e-6），`B=0` 时 == 原权重。
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

## 工程取舍

1. **自研 autograd 而非 PyTorch**：为在「零框架」约束下仍可训练，并让 INV-22 成为
   梯度正确性的黄金基准；代价是速度，故仅用于微型模型。
2. **TinyTrainer 与 Transformer 参数同构**：`to_numpy_params()` 产出与 L2 完全一致的
   key 集合，`ifai demo` 可直接加载训练结果推理，形成「训练→推理」闭环。
3. **RoPE 在 autograd 中可微表达**：`_rope` 用可微算子实现偶数维旋转并在反向时按位
   分流，确保位置编码的梯度能正确回传（而非 detach 近似）。
4. **CPU 优先的微型化**：默认 `seq_len=32, batch=8, steps=200`，显存与耗时均适配
   笔记本级 CPU，契合本栈教学/参考定位。
