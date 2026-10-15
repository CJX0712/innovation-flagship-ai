# L8 · 评测 / 对齐 (Evaluation & Alignment) — 设计说明

## 设计目标

L8 是 **ifai** 的质量门禁：以困惑度衡量语言模型拟合度、以吞吐/延迟基准做回归保护、
以 GSM8K-mini 验证推理管线正确性。目标是「用业界标准度量 + 可复现基准」把每层
INV 的验收收敛成可发布的证据。

本层覆盖 `ifai/eval/`（perplexity.py、benchmark.py、harness.py）。

## 复用世界顶级成果（零重写 · 集大成）

- **困惑度 Perplexity**（Jelinek et al. 1977，语言模型评价标准）：复用
  `PPL = exp(mean NLL)` 作为对测试序列「惊讶度」的事实标准度量；INV-39：
  `PPL == exp(mean NLL)`，与手工公式一致到 1e-6（`perplexity_manual` 作对照）。
- **MLPerf 吞吐/延迟基准方法论**（MLPerf，MLCommons）：复用「固定 prompt 与步数、
  多次运行统计均值/变异系数 CV」的系统评估方法论，作为回归基线与性能门禁
  （R-L8-02：3 次运行 CV ≤ 15%，写入 `BASELINE.md`）。
- **GSM8K 数学推理评测**（Cobbe et al. 2021，"Training Verifiers to Solve Math Word
  Problems"）：复用其「解题→抽取最终数字→比对」范式；工程上内置 20 道极简离线题
  （`GSM8K_Mini`），仅验证管线正确性（能否产出可解析数字答案），不硬设准确率门槛。
- **HELM 多维评测 + CI 基线保护**（Liang et al. 2022，"Holistic Evaluation of Language
  Models"）：复用「多维 + 基线保护」思想；`RegressionBaseline` 在吞吐相对 `BASELINE.md`
  掉落 >15% 时亮红灯（RED），守住院内性能不回归。

## 关键不变量 / 验收

- **INV-39** `PPL == exp(mean NLL)`，与手工 exp(mean NLL) 一致（≤ 1e-6）。
- **INV-40** 全局可复现（归属 L1 `ifai/rng.py`）：评测随机性（如采样评测）受 `--seed`
  控制，两次运行 bitwise 一致。本评测层承诺「评测调用全栈随机源经由 L1」，使基准
  可重复——该不变量在 `rng.py` 实现，于本层被消费与声明。
- **R-L8-02** 3 次基准运行变异系数 `CV ≤ 15%`，结果写入 `BASELINE.md`。

## 工程取舍

1. **困惑度用 L0 稳定原语**：`perplexity` 复用 `core.numeric.logsumexp` 计算 NLL，
   避免数值溢出，使 INV-39 在极大/极小 logits 下仍成立。
2. **微型模型只验证管线、不强求准确率**：随机初始化微型模型准确率天然极低，故
   GSM8K-mini 仅验证「能否产出可解析答案」，符合 PRD Q4，不做误判式硬门槛。
3. **基准用 CV 而非单次**：以 3 次运行均值 + 变异系数作为门禁，降低 CPU 抖动带来的
   误报（R-L8-02）。
4. **回归门禁读 `BASELINE.md`**：`RegressionBaseline.check` 无基线时返回 `NO_BASELINE`
   而非直接红灯，先建基线再守门，符合渐进式工程实践。
