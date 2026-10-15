# L0 · 算子内核 (Core Kernels) — 设计说明

## 设计目标

L0 是 **ifai** 的计算底座，提供 Transformer 推理/训练所需的融合算子、数值稳定
原语与量化器。所有实现均为纯 `numpy`（`float64`）的零依赖参考实现，目标是「在
CPU 上精确复现世界顶级算子的解析行为」，并通过「黄金测试」对照保证数值正确。

本层覆盖 `ifai/core/`（numeric.py、ops.py、quant.py）。数值稳定工具（logsumexp /
稳定 softmax / 安全交叉熵）按本栈分层归入 L0。

## 复用世界顶级成果（零重写 · 集大成）

- **RMSNorm**（Zhang & Sennrich 2019，"Root Mean Square Layer Normalization"）：
  复用 `x / sqrt(mean(x²)+eps) * w` 解析公式，工程上全程 `float64`、无近似、无融合
  误差，直接作为 Pre-LN 归一化（INV-06 范数归一）。
- **RoPE 旋转位置编码**（Su et al. 2024，"RoFormer"）：复用偶奇维旋转矩阵解析公式
  （INV-03 与解析旋转矩阵一致到 1e-6）；工程上**支持 `[T]` 与 `[B,T]` 两种位置形状**，
  后者用于连续批处理左补齐时「真实 token 取自然位置」（INV-16 的前置条件）。
- **NTK-aware 线性位置插值**（bloc97 2023 / Chen et al. "Extending Context"）：
  复用「缩放 base 以覆盖更长上下文」的思想——当 `scaling≠1` 时拉伸 `base`，
  实现上下文外推而无需重训。
- **SwiGLU**（Shazeer 2020，"GLU Variants Improve Transformer"）：复用
  `silu(x·W_gate) ⊙ (x·W_up) · W_down` 的 GLU 变体 FFN；工程上实现为**融合算子**
  （gate/up 一次算完再逐元素 silu 融合，INV-05 与分离实现一致到 1e-6）。
- **分块 GEMM / BLAS 分块思想**（Goto & van de Geijn 2008）：复用分块提升 cache
  命中率的矩阵乘思想，工程上以 `block` 分块累加，INV-01 与朴素 `a@b` 一致到 1e-5。
- **FlashAttention 在线 softmax 因果注意力**（Dao et al. 2022）：复用「沿 seq 维
  分块流式维护 running max/sum」的在线 softmax，避免物化 `[S,S]` 大矩阵；
  INV-02 与朴素因果注意力一致到 1e-5（朴素实现保留于 `attention_naive` 作对照）。
- **logsumexp 数值稳定**（Murphy 2012，"Machine Learning: a Probabilistic
  Perspective" §2.5.4）：复用 `max + log(sum(exp(x-max)))` 两路分解防止 exp 上溢/
  下溢；INV-08 与朴素 `log(sum(exp(x)))` 一致到 1e-6。
- **稳定 softmax / 平移不变**：复用「减行最大值」的平移不变技巧（INV-09：
  `softmax(x+c)==softmax(x)`），作为所有概率归一的统一入口。
- **安全交叉熵 + 标签平滑**（Szegedy et al. 2016，label smoothing）：复用目标分布
  `(1-ls)·onehot + ls/C`；用 logsumexp 稳定计算 log-prob，防止 `log(0)`，并同时返回
  对 logits 的梯度 `p - q`。
- **LLM.int8 每通道对称量化**（Dettmers et al. 2022）：复用 per-channel symmetric
  量化思想——`scale = max_abs/127`，INT8 对称映射；工程上 `float64` 模拟以保证可复现。
- **RTN 舍入到最近量化**（Frantar et al. 2022 GPTQ / Lin et al. 2023 AWQ 作为可选
  进阶）：复用 round-to-nearest 作为 INT8/INT4 基线量化；INV-07 重建误差 ≤ scale/2。

## 关键不变量 / 验收

- **INV-01** 分块 GEMM == 朴素 matmul（≤ 1e-5）。
- **INV-02** 在线 softmax 因果注意力 == 朴素因果注意力（≤ 1e-5）。
- **INV-03** RoPE == 解析旋转矩阵（≤ 1e-6）；且旋转后内积只依赖相对位置（INV-04）。
- **INV-05** SwiGLU 融合 == 分离实现（≤ 1e-6）。
- **INV-06** RMSNorm 将向量归一化到范数 ~1。
- **INV-07** 量化重建误差界：`|x - dequant(quant(x))| ≤ scale/2 + 1e-6`。
- **INV-08** logsumexp 稳定式 == 朴素 `log(sum(exp(x)))`（≤ 1e-6，解析解对照）。
- **INV-09** `softmax(x+c) == softmax(x)`（平移不变）。
- **R-L1-02** INT4 `pack ↔ unpack` 往返逐位一致（量化器内验证）。
- **R-L1-03** logits ∈ [-1e4, 1e4] 无 `inf/nan`（由 `stable_softmax` 保证）。

## 工程取舍

1. **纯 `numpy` + `float64`，零 GPU/零 C 扩展**：所有算子是教学级参考实现，目的是
   让「解析公式 ↔ 数值结果」可被黄金测试严格对照，而非追求吞吐。正确性优先于性能。
2. **保留朴素参考实现**：`attention_naive`、`BM25.reference`、`DenseIndex.brute` 等
   「对照实现」与融合实现并存，使每个 INV 都有解析/暴力基准可证。
3. **确定性优先**：不引入任何随机源到算子层；随机性仅来自 L1 全局 RNG，保证 `--seed`
   下 bitwise 可复现（INV-40）。
4. **量化器模拟而非真降精度**：INT8/INT4/fp16 均以 `float64`→`int8`/`uint8`→`float64`
   往返模拟误差，便于在 CPU 上验证误差界，而非依赖硬件低精度单元。
