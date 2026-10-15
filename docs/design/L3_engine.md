# L3 · 推理引擎 (Inference Engine) — 设计说明

## 设计目标

L3 把 L2 的模型变成「能高效、可控生成」的推理系统：分页 KV Cache 管理显存、
连续批处理调度、投机解码加速、以及工业级采样栈。目标是「在 CPU 上精确复现现代
推理系统的核心算法，并保证与朴素自回归逐 token 结果 100% 一致」。

本层覆盖 `ifai/engine/`（kvcache.py、scheduler.py、speculative.py、sampler.py）。

## 复用世界顶级成果（零重写 · 集大成）

- **PagedAttention 分页注意力**（Kwon et al. 2023，"vLLM: Efficient Memory Management
  for Large Language Model Serving with PagedAttention"）：复用「按固定 size block 存储
  KV、block table 映射逻辑→物理」的分块显存管理；工程上按需分配 block、无预留，
  长序列内存约为稠密 50%（R-L3-01）。INV-15：分页 KV 与稠密 KV 数值完全一致（仅重排）。
- **连续批处理 Continuous Batching**（Yu et al. 2022，"Orca: A Distributed Serving
  System for Transformer-Based Generative Models"）：复用 iteration-level 调度——
  不同序列在不同步数结束、每步仅对「仍在生成」的序列做前向；工程上左补齐 + 真实
  token 末位对齐 + `key_padding_mask`，使结果 == 对每条序列串行逐 token 自回归
  （INV-16，贪心下 100% 一致）。
- **投机解码 Speculative Decoding**（Leviathan et al. 2023，"Fast Inference from
  Transformers via Speculative Decoding"）+ **Medusa / EAGLE**（Chen et al. 2023）：
  复用「草稿模型提 γ 候选 → 目标模型并行验证」框架；工程上贪心模式逐 token 结果与
  朴素自回归贪心 100% 一致（INV-17）。
- **top-k 采样**（Fan et al. 2018，"Hierarchical Neural Story Generation"）：复用
  截断至前 k 个最高概率 token 的过滤策略。
- **top-p / nucleus 采样**（Holtzman et al. 2020，"The Curious Case of Neural Networks'
  Sudden Creativity"）：复用按累积概率截断的核采样。
- **min-p 采样**（Nguyen et al. 2024）：复用「阈值随峰值概率缩放」的过滤（相对而非
  绝对阈值），工程上 `min_p * pmax` 作为保留下限。
- **重复惩罚 Repetition Penalty**（Keskar et al. 2019）：复用对「已出现 token」做除法
  惩罚（>0 除以、<0 乘回），抑制退化重复。
- **温度缩放 → 贪心退化**：复用 `temperature==0` 退化为 `argmax`（INV-19）。

## 关键不变量 / 验收

- **INV-15** 分页 KV 与稠密 KV 数值完全一致（无量化、仅重排）。
- **INV-16** 连续批处理（贪心）结果与对每条序列串行逐 token 自回归 100% 一致。
- **INV-17** 投机解码（贪心）逐 token 结果与朴素自回归贪心 100% 一致。
- **INV-18** 过滤 + 重归一后 `Σp = 1 ± 1e-6`（`Sampler.probs`）。
- **INV-19** `temperature→0` 或 `top_k=1` == 贪心 `argmax`。
- **R-L3-01** 长序列下分页 KV 内存约为稠密的 50%。

## 工程取舍

1. **等价性优先于性能**：所有加速结构（分页、连续批、投机）都把「与朴素自回归结果
   一致」作为硬验收（INV-15/16/17），先在 CPU 上证明正确，再谈加速。
2. **统一前向接口**：调度器/投机均以 `model_forward(ids[B,T]) -> logits[B,T,V]` 为契约，
   与 L2 `Transformer.forward` 对接，弱化层间耦合。
3. **采样与调度解耦**：`Sampler` 只负责「logits→token」的概率逻辑，`ContinuousScheduler`
   / `speculative_decode` 只负责「编排」，便于分别测试（INV-18 与 INV-16/17 独立验收）。
4. **左补齐 + padding mask 而非右补齐**：保证「真实 token 取自然位置」（配合 L0 RoPE
   的 `[B,T]` 位置形状），是连续批处理等价性的关键工程细节。
