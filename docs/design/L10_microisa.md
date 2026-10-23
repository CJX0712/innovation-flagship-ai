# L10 · 微指令 ISA (Micro-Instruction ISA) — 设计说明

## 设计目标

L10 把前九层的能力**编排成可枚举、可逐条断言的指令集**。一个系统宣称自己
有多少算子是廉价的; 真正有价值的是**每一条都能被单独执行、单独断言**。

本层交付 **1024 条可执行微指令** (32 家族 × 32 成员), 全部是**真实可调用的
函数**, 不是文档字符串。门禁对每一条跑真实输入并断言数学性质 —— 失败就如实
报失败, 不注水。

## 零重写 · 集大成 (移植声明)

32 家族 × 32 成员 ISA **移植自同门旗舰 [`tianji-ai`](https://github.com/CJX0712/tianji-ai)**
的 `tianji/l10_microops.py` (MIT, 作者同为晨星) —— 该实现已在 tianji-ai 上
通过 1024/1024 门禁。

移植**只改依赖指向, 不改算法体**:

| 依赖 | 原 (tianji-ai) | 现 (ifai) | 理由 |
|---|---|---|---|
| `l2norm` | `tianji.l4_deltanet` | `ifai.core.ops.l2norm` | 本次一并补入本仓 L0 |
| `spectral_norm` | `tianji.l0_numerics` | `ifai.core.ops.spectral_norm` | 同上 |
| `newton_schulz` | `tianji.l0_numerics` | `ifai.core.ops.newton_schulz` | 同上 |
| `logsumexp` | `tianji.l0_numerics` | `ifai.core.numeric.logsumexp` | 本仓已有 INV-08 黄金测试 |
| `softmax` | `tianji.l0_numerics` | `ifai.core.numeric.stable_softmax` | 本仓已有 INV-09 黄金测试 |

**为什么不直接复制一份**: 那会造出两套语义相同的实现, 悄悄分叉 —— 正是
ifai v0.7.13 审计批过的「模块存在但从未接线」同型缺口的镜像版。
**为什么补 L0 而不是直接依赖外部**: 移植后的 `newton_schulz` / `spectral_norm`
若无本仓消费方, 就是「移植进来的死代码」。补进 L0 后, 它们同时被 L10 的
`spectral_normalize` / `orthogonalize` 消费, 且有本仓自己的黄金测试
(`tests/unit/l10/test_l10_microisa.py`) 锁住语义。

## 超越点: 数学性质断言层 (`_properties.py`)

移植源的门禁只断言「不抛异常 + 输出有限」—— 这只能证明「算子能跑」,
不能证明「算子对」。本层追加**家族级数学不变量断言**:

| 家族 | 断言内容 | 守护对象 |
|---|---|---|
| `smax` | softmax 非负 + 行和 = 1; 平移不变 softmax(x+c) == softmax(x) | INV-09 |
| `norm` | RMSNorm 行均方 = 1; LayerNorm/zscore 零均值 (+单位方差); l2norm 严格单位范数 | INV-06 |
| `prob` | KL 散度非负 (Shannon 1948) | 教科书性质 |
| `loss` | 损失非负 | 教科书性质 |
| `metr` | 指标有限; 比例型 ∈ [0,1] | 教科书性质 |
| `attn` | Flash 风格在线 softmax ≡ 朴素注意力 | **INV-02** |
| `optm` | Newton-Schulz 奇异值 ∈ (0.5, 1.5); clip_value ∈ [-1,1]; clip_norm1 范数 ≤ 1 | **Muon 公开验收标准** (Keller Jordan 2024) |
| `spec` | 幂迭代谱范数 ≡ 精确 SVD (≤1e-6); 幂迭代从下方逼近; gram/cov 半正定; symmetric/skew 对称性 | Golub & Van Loan §4.2 |
| `topk` | Top-K 掩码每行恰好 min(k, d) 个 1; Top-K 均值 ∈ [min, max] | 稀疏选择定义 |
| `dist` | 距离/范数非负 | 教科书性质 |
| `quant` | 输出全有限 | **INV-07** 前提 |
| `rope` | 真旋转成员保范数 (正交变换) | **INV-03** |
| `ract` | 统计量有限; 非负量非负 | 教科书性质 |

**豁免纪律**: 26 条因语义特殊不适用家族共性 (如 `rope_cos_only` 是逐元素乘
三角函数、本就不是旋转; `smax.log_softmax` 是对数而非分布)。**每条豁免都
显式登记并注明理由**, 豁免清单被门禁上报 —— 杜绝「悄悄关掉断言」。
`tests/unit/l10/test_l10_microisa.py` 另外断言: 豁免比例 < 10%, 且豁免清单里
的成员名必须真实存在 (防止写错名字导致豁免失效)。

## 32 家族语义类型

`FAMILY_KIND` 区分两类家族, 刻意不把「凑数」伪装成「独立算子」:

- **`core`** (15 族): 家族内成员语义彼此不同 —— `uact` / `bact` / `ract` /
  `norm` / `gate` / `moer` / `misc` / `grad` / `init` / `scan` / `seqm` /
  `cache` / `specf` / `topk` / `indx`。
- **`sweep`** (17 族): 同一算子的参数扫描, 作用是覆盖数值边界 ——
  `smax` (温度) / `rope` (base) / `mask` (滑窗宽) / `attn` (scale) /
  `loss` (Lp) / `metr` (top-k) / `quant` / `smpl` (温度) / `optm` /
  `dist` (Lp 阶) / `prob` / `spec` (幂迭代) / `conv` / `pool` / `shap` /
  `cast` / `rand`。

读者可自行判断哪些家族的「独立算子密度」更高 —— 这是刻意的诚实。

## 关键不变量 / 验收

- **ISA 数量不变量**: 恰好 1024 条 / 32 家族 / 每族 32 条; 全局键唯一且带
  家族前缀 (不同家族会出现同名成员, 如 `entropy` 同属 `ract`/`smax`/`prob`)。
- **可执行性**: 全部 1024 条能被确定性输入执行, 不抛异常, 输出全有限
  (掩码族的 -inf 属刻意行为, 显式豁免而非放宽阈值)。
- **可证性**: 390 条带数学性质的微指令逐条通过断言; 26 条显式豁免且注明理由。
- **确定性**: 固定种子两次运行结果逐条一致 (跨进程可复现)。
- **L0 同源性**: `norm.l2norm` 必须与本仓 L0 同源 → 输出严格单位范数。

## 工程取舍

1. **移植而非重写**: 1024 条微指令手写既慢又难以保证质量。移植已被门禁证明
   过的资产, 再在其上叠加**本仓独有的性质断言层**, 是「零重写 · 集大成」
   的完整示范 —— 复用的同时超越。
2. **断言写在家族级, 但判据逐成员**: 家族给出共性, 但**不能套用统一模板**。
   本轮开发中曾三次犯下「把 A 家族的性质套到 B 家族」的错 (softmax 归一性
   套到 `norm` / NS 奇异值带套到整个 `optm` / Flash 等价性套到错误 scale),
   最终改为**逐成员读懂真实语义后写判据** —— 断言的正确性优先于断言的数量。
3. **豁免必须显式**: 允许「这条不适用」, 但不允许「悄悄不查」。豁免清单
   上报 + 比例上限 + 成员名校验三重约束。
4. **CLI 可达**: `ifai microisa` 提供浏览 (家族/成员) 与执行 (单条) 入口 ——
   微指令层不能只是测试里的内部实现。
