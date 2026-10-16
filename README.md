# 创新旗舰 AI · Innovation Flagship AI

> 十层架构 · 零重写 · 集大成 · CPU 可运行 · 可复现 · 可验证
> 作者 **晨星 (CJX0712)** · 许可 **MIT**

---

## 一、这是什么

**创新旗舰 AI** 是一个**旗舰级 AI 算法内核**，目标是把世界顶级的机器学习 / 深度学习 / 强化学习数学与算法，以**纯 JavaScript 单文件、零依赖、零构建**的形式重新实现，并配套一套**可被独立交叉验证的硬不变量（invariant）自检体系**。

它不追求堆参数、拼算力，而是追求：

- **旗舰零重写**：先检索世界上已有的最优方案（PyTorch / JAX / llama.cpp / FAISS / scikit-learn / Spinning Up …），直接复用其公式与结构，不闭门造车。
- **旗舰集大成**：每一行代码都站在世界顶级成果之上——Adam/AdamW 复刻 PyTorch，RMSNorm/RoPE 复刻 llama.cpp，注意力复刻 Vaswani / GQA，检索复刻 FAISS 暴力精确召回。
- **旗舰唯一目标**：锻造一个**架构最先进、性能最强、代码最优、文档最全**的旗舰级内核。
- **旗舰闭环交付**：完整开源至 GitHub，**可运行 / 可复现 / 可验证**，署名晨星。
- **旗舰自治执行**：自主安装 / 配置 / 诊断 / 修复一切依赖与故障，不停摆、不请示。

---

## 二、十层架构（2046 项微指令）

旗舰 AI 采用自顶向下的**十层分解架构**，逐层将「世界级 AI 系统」拆解为可独立实现、可独立验证的微指令：

| 层 | 名称 | 项数 | 状态 |
|----|------|------|------|
| L1 | 基础设施 / 数值内核 | 2 | 部分落地 |
| L2 | 线性代数 / 稳定数值 / 概率图基础 | 4 | 部分落地 |
| L3 | 优化器 / 自动微分 / 损失激活归一化 / 基础网络 | 8 | 部分落地 |
| L4 | 注意力 / 位置编码 / 经典注意力 | 16 | 部分落地 |
| L5 | 检索 / 概率分布 / 图算法 / 强化学习 | 32 | 部分落地 |
| L6 | 序列建模 / Transformer 主干 | 64 | 规划中 |
| L7 | 多模态 / 生成模型 | 128 | 规划中 |
| L8 | 大模型预训练 / 对齐 | 256 | 规划中 |
| L9 | 推理优化 / 部署 | 512 | 规划中 |
| L10 | 微指令级操作（最终层） | **1024** | **done 70 / planned 954** |
| **合计** | | **2046** | |

- L1–L10 项数按 2·2ⁿ 增长（L1=2, L2=4, …, L10=1024），合计 ∑ = **2046**。
- L10 的 1024 项微指令完整分类法见 [`docs/L10-index.json`](docs/L10-index.json) 与 [`docs/L10-taxonomy.md`](docs/L10-taxonomy.md)（8 族 / 64 组 / 每组 16 op）。
- 当前已落码的 **26 个内核模块**，覆盖了 L1–L6 的基础与核心算法，并直接支撑 L10 中 **74 项 done** 微指令（其余 950 项标记为 planned，等待后续波次）。

---

## 三、模块清单（26 个内核 · 202 条不变量 · 0 失败）

所有模块均满足 `@forge-module` 统一契约：`selfCheck()` 返回 `forge-selfcheck/1.0` schema、代码 hash 幂等、不良输入抛错、CLI 退出码规范（0/1/2/3/4/66）。每条不变量都有 **≥1 条可被独立交叉验证的硬标准**（残差界 / 正交归一 / 梯度检验 / 闭式对照 / 参考重实现），阈值常量化于 `TOL`。

| 模块 | 层 | 不变量 ID 前缀 | 条数 | 复用来源 |
|------|----|---------------|------|----------|
| `src/core-numeric.js` | L1 | INV-NUM-001..008 | 8 | 分块 GEMM / SHA-256(FIPS 180-4) / Kahan / Acklam 逆 CDF |
| `src/core-verify.js` | L1 | INV-VER-001..004 | 4 | 中心差分梯度检验 / 报告聚合一致性 |
| `src/stable-forge.js` | L2 | INV-STB-001..004 | 4 | scipy.logsumexp / 稳定 softmax / log-sigmoid / softplus |
| `src/linalg-forge.js` | L2 | INV-LIN-001..008 | 8 | LAPACK(dgetrf/Cholesky/QR) / one-sided Jacobi SVD/EVD |
| `src/pca-forge.js` | L2 | INV-PCA-001..008 | 8 | 协方差 / Jacobi 特征分解 / 方差解释率 |
| `src/gmm-forge.js` | L2 | INV-GMM-001..008 | 8 | EM 算法 / logsumexp 责任度 / 半正定协方差 |
| `src/nn-forge.js` | L3 | INV-NN-001..008 | 8 | 数值梯度检验 / XOR / softmax |
| `src/attn-forge.js` | L4 | INV-ATTN-001..008 | 8 | 经典注意力 / 因果掩码 / 数值梯度 |
| `src/optim-forge.js` | L3 | INV-OPT-001..008 | 8 | PyTorch(Adam/AdamW/SGD/Nesterov/RMSProp/AdaGrad) |
| `src/autodiff-forge.js` | L3 | INV-AUT-001..008 | 8 | JAX/PyTorch autograd（前向 Dual + 反向 tape） |
| `src/loss-act-norm-forge.js` | L3 | INV-LAN-001..008 | 8 | RMSNorm / LayerNorm / GELU / SwiGLU / CE-MSE |
| `src/attention-forge.js` | L4 | INV-ATT-001..008 | 8 | Vaswani SDPA / MHA / GQA / 因果注意力 |
| `src/rope-forge.js` | L4 | INV-ROPE-001..008 | 8 | Su et al. Rotary Position Embedding |
| `src/transformer-forge.js` | L4 | INV-TRA-001..008 | 8 | Vaswani 2017 / GPT-2 pre-LN / llama.cpp RMSNorm+SwiGLU |
| `src/kv-cache-forge.js` | L4 | INV-KVC-001..008 | 8 | vLLM PagedAttention / llama.cpp / HF DynamicCache |
| `src/position-encoding-forge.js` | L4 | INV-POS-001..008 | 8 | ALiBi(Press 2021) / 位置插值(Chen 2023) / T5 相对桶 |
| `src/moe-forge.js` | L4 | INV-MOE-001..008 | 8 | Shazeer 2017 / Switch Transformer / Mixtral / DeepSeekMoE |
| `src/quant-forge.js` | L4 | INV-QNT-001..008 | 9 | llama.cpp k-quants / LLM.int8() / GPTQ / SmoothQuant |
| `src/speculative-forge.js` | L4 | INV-SPC-001..008 | 8 | Leviathan 2023 推测解码 / Medusa / EAGLE |
| `src/generation-forge.js` | L4 | INV-GEN-001..008 | 8 | HF generate / vLLM / llama.cpp / OpenAI streaming（集成封顶） |
| `src/sampling-forge.js` | L5 | INV-SMP-001..008 | 8 | Holtzman 2019 top-p / HF LogitsProcessor / Fan 2018 top-k |
| `src/retrieval-forge.js` | L5 | INV-RET-001..008 | 8 | FAISS / hnswlib 暴力精确召回（余弦 / L2 / top-k） |
| `src/prob-forge.js` | L5 | INV-PRB-001..008 | 8 | 高斯 / Bernoulli / KL 散度 / 逆 CDF 采样 |
| `src/graph-forge.js` | L5 | INV-GRAPH-001..008 | 8 | BFS / Dijkstra（独立重实现校验） |
| `src/rl-forge.js` | L5 | INV-RL-001..008 | 8 | 值迭代 / 贝尔曼最优性 / REINFORCE 策略梯度 |
| `src/dqn-forge.js` | L6 | INV-DQN-001..008 | 8 | Mnih 2015 DQN / 目标网络 / Double DQN(van Hasselt 2015) / Polyak / ε-贪婪 |

> 注：`attn-forge.js`（L4，INV-ATTN-*）为经典注意力模块的零重写迁移包装；`attention-forge.js`（L4，INV-ATT-*）为新写的多头 / GQA / 因果注意力。两者并存、互不冲突。

完整不变量清单（含每条的验证方法与容差）见 **[`INVARIANTS.md`](INVARIANTS.md)**。

---

## 四、如何运行

### 环境要求
- **Node.js ≥ 18**（仅用内置 `fs` / `path` / `os` / `child_process`，**无任何 npm 依赖、无 bundler、无 TypeScript**）。
- 浏览器（离线 `file://` 直接打开 `index.html` 即可自检，无需服务器）。

### 命令行自检（Node 端）
```bash
# 1) 聚合自检：26 模块 / 202 不变量，全部跑一遍
npm test                 # ≡ node run-all.js
npm run selfcheck        # ≡ node run-all.js

# 2) 契约自检：校验 @forge-module 契约（schema / hash 幂等 / 不良输入 / 退出码）
npm run harness          # ≡ node tests/harness.js

# 3) 一键验证（聚合 + 契约）
npm run verify           # ≡ node run-all.js && node tests/harness.js

# 4) L10 分类法生成 + 校验
npm run l10              # ≡ node tools/gen-l10.js && node tools/verify-l10.js

# 5) 复用合规校验：每个模块必须声明 @reuse，且 done 微指令必须有实现模块
npm run check:reuse      # ≡ node tools/check-reuse.js
```

### 浏览器端自检（离线面板）
直接双击打开 **`index.html`**，面板会逐模块加载 `src/*.js`（经典 `<script src>`，禁用 ESM 以避免 `file://` CORS 阻断），并展示每个模块的不变量通过情况与代码 hash。

### 单模块自检
每个 `*-forge.js` / `core-*` 文件均支持独立运行：
```bash
node src/optim-forge.js --self-check     # 该模块自检，退出码 0=全过 / 1=有失败
node src/optim-forge.js --json           # stdout 输出纯 JSON 自检报告（日志走 stderr）
```

---

## 五、统一契约与不变量体系

### `@forge-module` 契约
- 文件头声明 `@forge-module`、版本、层、作者、许可。
- `selfCheck(opts)` 返回 `forge-selfcheck/1.0` schema：`{ schema, module, version, layer, passed, failed, skipped, invariants[], cases[], hash }`。
- `hash` 仅覆盖 `[id, status, actual]`（排除 env / duration），保证**幂等可复现**。
- 不良输入必须**抛错**（契约自检器 `tests/harness.js` 会刻意传入坏输入验证）。
- CLI 退出码：`0` 全过 / `1` 有失败 / `2` 参数错误 / `3` 契约缺失 / `4` 环境异常 / `66` 保留。

### `@reuse` 五档复用声明
每个模块头以 `@reuse NAME|URL|LICENSE|GRADE|note` 声明其世界顶级复用来源：
- **A** 直接复用世界顶级实现（公式级）
- **B** 复用其数学/算法结构
- **C** 复用接口/契约约定
- **D** 复用测试向量/黄金参照
- **X** 零重写迁移包装

`tools/check-reuse.js` 强制每个模块 ≥ 1 条 `@reuse`，并交叉校验 L10 done 项均有实现模块（孤儿检查）。

### `TOL` 九档容差
`EXACT=0` / `ULP=1e-16` / `TIGHT=1e-15` / `ORTHO=1e-14` / `RESID=1e-12` / `SOLVE=1e-10` / `GRAD=1e-6` / `MONO=-1e-9` / `CONV=1e-6`。阈值常量化，杜绝「视情况放宽」。

---

## 六、项目结构

```
innovation-flagship-ai/
├── index.html              # 浏览器离线自检面板（零构建）
├── run-all.js              # 聚合自检运行器（自动发现 src/*.js）
├── package.json            # 脚本：test/selfcheck/harness/verify/l10/check:reuse
├── LICENSE                 # MIT（署名 晨星 / CJX0712）
├── src/                    # 26 个零依赖单文件内核
│   ├── core-numeric.js  core-verify.js  stable-forge.js
│   ├── linalg-forge.js  pca-forge.js  gmm-forge.js
│   ├── nn-forge.js  attn-forge.js
│   ├── optim-forge.js  autodiff-forge.js  loss-act-norm-forge.js
│   ├── attention-forge.js  rope-forge.js  transformer-forge.js
│   ├── kv-cache-forge.js  position-encoding-forge.js  moe-forge.js
│   ├── quant-forge.js  speculative-forge.js  generation-forge.js
│   └── retrieval-forge.js  prob-forge.js  graph-forge.js
│   └── rl-forge.js  sampling-forge.js  dqn-forge.js
├── tests/
│   ├── harness.js          # @forge-module 契约自检器
│   └── independent/        # 独立黄金参照复核（不采信模块自检）
│       └── INV-DQN-007-check.js
├── tools/
│   ├── gen-l10.js          # L10 微指令分类法生成
│   ├── verify-l10.js       # L10 校验（唯一性 / 字段完整 / 状态分布）
│   ├── check-reuse.js      # 复用合规校验
│   └── env-check.js        # 运行环境检查
└── docs/                   # 架构 / PRD / 复用矩阵 / L10 分类法
    ├── ARCHITECTURE.md  PRD.md  REUSE-MATRIX.md
    ├── L10-index.json   L10-taxonomy.md
    └── class-diagram.mermaid  sequence-diagram.mermaid
```

---

## 七、当前状态与路线图

- ✅ **W1–W2**：基础设施与数学内核（8 模块 / 57 不变量）全绿
- ✅ **W3**：优化器 / 自动微分 / 损失激活归一化（3 模块 / 24 不变量）全绿
- ✅ **W4**：注意力（SDPA/MHA/GQA/因果）+ 旋转位置编码 RoPE（2 模块 / 16 不变量）全绿
- ✅ **W5**：向量检索 / 概率分布 / 图算法 / 强化学习（4 模块 / 32 不变量）全绿
- ✅ **W6**：L10 分类法 done 计数修正 + 复用合规校验 `check-reuse.js`
- ✅ **W7**：README / INVARIANTS.md / CI / GitHub 闭环交付
- ✅ **W8**：Transformer 模型组件（嵌入 / 正弦位置编码 / 预归一化残差 / SwiGLU 前馈 / 编码器前向）+ 生成采样内核（温度 / 贪心 / top-k / top-p 核采样 / 重复惩罚 / 批量采样）——2 模块 / 16 不变量全绿
- ✅ **W9**：KV 缓存与增量解码（预分配 / 分页块表 / 增量等价全量前向 / 逐出）+ 位置编码族（ALiBi 线性偏置 / T5 相对桶 / 位置插值）+ 混合专家 MoE（top-k 门控 / 路由 / 负载均衡 / 容量裁剪 / 共享专家）——3 模块 / 24 不变量全绿
- ✅ **W10**：模型量化（int8/int4 真位压缩 / 逐通道 / k-quants 超块 / 量化矩阵乘）+ 推测解码（草稿提议 / 目标并行验证 / 拒绝采样 / **分布等价**）+ 端到端生成闭环（KV 增量 / 流式 / 前缀缓存 / 批量）——3 模块 / 25 用例全绿
- ✅ **W11（进行中）**：深度强化学习内核 dqn-forge（经验回放 / 目标网络 / ε-贪婪 / DQN 更新 / Double DQN / Polyak / **收敛到值迭代**，皇冠不变量 INV-DQN-007 经手搓黄金参照 + 3 种子独立复核确认）——1 模块 / 8 不变量全绿；学习率调度器 scheduler-forge 与扩散模型内核 diffusion-forge（DDPM/DDIM）并行建造中
- 🔜 **W11+ / W12+**：F2-G2 学习率调度（阶跃 / 余弦退火 / warmup / 余弦重启 / 1cycle）、F5-G7 扩散模型（β 调度 / 前向加噪 / DDPM 祖先采样 / DDIM 确定性采样）、F4-G5 余项（并行解码、树状注意力、缓存压缩、注意力下沉）、F4-G6 余项（混合精度、SmoothQuant、QAT、稀疏量化）、F5 概率与生成余项（MCMC / VAE / 流模型）、F6 策略梯度、F7 检索/图/系统——按十层架构逐波推进，最终填满 L10 的 1024 项微指令。

**当前验证基线（W11 dqn 集成后本地复核）：**
- `npm run verify` → **26 模块 · 通过 202 · 失败 0**
- `npm run harness` → 26/26 模块满足 `@forge-module` 契约（含不良输入抛错登记）
- `npm run check:reuse` → 0 ERROR / 5 ADVISORY（X 级自研声明提示，不影响退出码）
- `npm run l10` → 1024 项 / done 74 / planned 950 / ID 唯一 / 字段完整

---

## 八、许可与署名

- **许可**：MIT License（见 [`LICENSE`](LICENSE)）。
- **作者**：晨星 (CJX0712)。
- **复用来源**：所有第三方算法均在其相应 LICENSE 下合规复用（PyTorch、JAX、llama.cpp、FAISS、scikit-learn、Spinning Up 等），详见各模块 `@reuse` 头与 [`docs/REUSE-MATRIX.md`](docs/REUSE-MATRIX.md)。

> 创新旗舰 AI —— 站在世界顶级成果之上，锻造世界顶级 AI。
