# L1 · 张量 / 数值基础 (Numerical Foundation) — 设计说明

## 设计目标

L1 提供全栈**统一的全局随机源**与「张量/数值基础」约定，是确定性（determinism）
与可复现性（reproducibility）的唯一真相来源。所有上层（采样、训练、分词、检索）的
随机性都必须经由本层，绝不私自 `np.random` 或 `random`。

本层对应 `ifai/rng.py`（全局统一随机源）。

## 复用世界顶级成果（零重写 · 集大成）

- **NumPy `Generator` / PCG64 伪随机数生成器**（O'Neill 2014，"PCG: A Family of
  Simple Fast Space-Efficient Statistically Good Algorithms for Random Number
  Generation"）：复用 NumPy 官方 `numpy.random.Generator`（默认 PCG64），**不重写
  PRNG**。工程上所有模块通过 `get_rng()` 取用同一全局实例，统一 `spawn()` 派生隔离
  子源，避免各自为战的随机状态。

## 关键不变量 / 验收

- **INV-40** 全局种子可复现：`seed(N)` 后两次运行同一随机调用 bitwise 一致
  （`seed(0); a=get_rng().random(); seed(0); b=get_rng().random()` ⇒ `a==b`）。
- 派生源隔离：`spawn(child_seed)` 返回独立 `Generator`，不影响全局源，保证并行/子任务
  间随机流互不污染。
- 默认安全：`get_rng()` 未初始化时默认 `seed(0)`，任何入口都不会因「未设种子」而引入
  不可复现行为。

## 工程取舍

1. **单一随机入口，拒绝散落随机**：这是「可复现」的硬约束。L3 采样器、L4 TinyTrainer、
   L5 稠密/重排编码器、L6 工具均调用 `ifai.rng.get_rng()`，使 `--seed N` 成为「两次
   运行 bitwise 一致」的充分条件。
2. **复用成熟 PRNG 而非自研**：PCG64 具有良好的统计性与速度，且 NumPy 已维护；自研
   PRNG 既易出错又无收益，违背「零重写」原则。
3. **CPU-only 友好**：`Generator` 为纯 Python/NumPy，无 GPU 状态，天然适配本栈的
   CPU 参考实现定位。
4. **`spawn` 用于确定性并行**：以「全局种子 ⊕ 子种子」派生子源，使多进程/多序列场景
   仍可精确复现，而非依赖 OS 线程调度。
