# 创新旗舰 AI（Innovation Flagship AI）· 产品需求文档

| 项 | 内容 |
|---|---|
| 文档版本 | v1.0（PRD 阶段产出） |
| 撰写角色 | 产品经理 · 许清楚 |
| 项目名（snake_case） | `innovation_flagship_ai` |
| 主语言 | 简体中文（关键术语/数据保留英文） |
| 核心实现语言 | 纯 JavaScript（ES2022 单文件，零依赖、零构建） |
| 运行目标 | 浏览器（单文件 HTML，内联 CSS/JS）+ Node v22.22.2 无头 |
| 运行环境 | Windows 11 / Git Bash / Node v22.22.2 / Python 3.13.12 |
| 开源署名 | 晨星（GitHub: `CJX0712`） |
| 上游约束 | 旗舰零重写 / 旗舰集大成 / 旗舰唯一目标 / 旗舰闭环交付 / 旗舰自治执行 |

### 原始需求复述（用户原话要点）

1. **旗舰零重写**：禁止从零造轮子。任何需求，先检索世界已有的最优解（顶级数学、顶级算法、顶级代码、顶级开源项目），直接复用、组合、超越。
2. **旗舰集大成**：每一行代码、每一个模块，都必须站在世界顶级成果之上；凡有更强的现成方案，立即替换自写。
3. **旗舰唯一目标**：锻造世界顶级 AI——架构最先进、性能最强、代码最优、文档最全。
4. **旗舰闭环交付**：成果必须完整开源至 GitHub，可运行、可复现、可验证，作者署名：晨星。
5. **旗舰自治执行**：自主安装、配置、诊断、修复一切依赖与故障，不停摆、不请示琐事，直奔最终交付。

---

## 1. 产品目标

### 1.1 一句话产品定义

> 一个**纯 JS 单文件、零依赖零构建**的「可验证 AI 算法内核库」：每个模块都显式站在世界顶级开源成果之上（复用/移植/交叉验证），并自带**至少一条可被独立复现的硬不变量**，浏览器一键自检 + Node 无头全绿，完整开源至 GitHub。

### 1.2 三条可量化成功标准（必须可验证）

| ID | 成功标准 | 量化指标 | 验证方式（可自动化） |
|---|---|---|---|
| **S1 · 自检全绿** | 所有交付模块自带 `self-check`，浏览器与 Node 双端全绿 | 模块总数 **≥ 8**（其中 **4 个为存量迁移**：`pca-forge` / `gmm-forge` / `nn-forge` / `attn-forge`，来源见 §12；**新增 ≥ 4**）；`node run-all.js` 汇总退出码 **0**；单模块 `node <mod>.js --self-check` 退出码 **0**；浏览器面板断言通过率 **100%** | 脚本解析 JSON 报告，断言 `failed === 0` |
| **S2 · 硬不变量** | 每个模块 ≥ 1 条可独立交叉验证的硬不变量，阈值写死在代码中 | 硬不变量总数 **≥ 8**（每模块 ≥ 1）；典型阈值：Jacobi 残差 `‖Av−λv‖ ≤ 1e-12`、特征向量正交归一 `≤ 1e-14`、协方差对称+半正定 `≥ -1e-12`、梯度检验相对误差 `≤ 1e-6`、EM 对数似然单调不减 `Δ ≥ -1e-9`、责任度每行 `Σ=1 ±1e-12`、投机解码输出分布与原模型分布一致（拒绝采样可证）、Dijkstra 与 Bellman-Ford 距离一致 `≤ 1e-12` | 每个不变量一条独立断言 + 一条「独立实现」对照路径 |
| **S3 · 可复现交付** | 仓库一次 push 完成，陌生人可一键复现 | `git clone` → `node run-all.js` → 退出码 `0`，耗时 **≤ 120 s**；`README.md` 含复现命令三连（clone / node run-all / 打开 HTML）；`LICENSE` 存在；作者署名 `晨星 / CJX0712`；零外部网络请求、零 `npm install` | CI 工作流 + 本地无头复跑 |
| **S4 · 零重写合规** | 每个模块必须标注真实顶级来源 | 每模块 `REUSE.md` 至少 1 个**真实存在的**顶级项目/论文引用（含链接）；合规覆盖率 **100%**；自写率（无任何来源标注的代码段）**≤ 20%** | 静态扫描 `REUSE.md` 存在性与链接可达性 |

> 说明：S1–S4 中，S1/S2/S3 为**硬性门槛**（任一不满足即判定未交付）；S4 为**过程合规门槛**。

---

## 2. 目标用户与使用场景

| 用户角色 | 典型场景 | 核心诉求 |
|---|---|---|
| 算法研究者 / 研究生 | 打开单文件 HTML，逐条查看某算法的不变量数值与残差 | 结论**可验证**，不是"跑通了"而是"证明了" |
| 教师 / 课程设计者 | 课堂投影自检面板，边改参数边看数学性质变化 | 数学性质**可视化、可交互、零环境成本** |
| 前端 / 全栈工程师 | `import` 单个 `.js` 到 Node 或浏览器，零构建接入产品 | **零依赖、零构建、即插即用**的即拿即走 |
| AI 工程师 / 研究员 | 用本库作为**独立第二实现**，交叉验证自研的 PyTorch/C++ 版本 | 独立代码路径 + 硬不变量对照 |
| 开源贡献者 | fork 后补一条不变量或补一个族的微指令 | **低门槛、可定位**的贡献单元（1 个微指令 = 1 个 PR） |
| 作者本人（晨星） | 十层分解架构逐层补齐、长期演进，1024 项微指令逐步填满 | 统一命名规范、统一验证总纲、可持续扩展 |

---

## 3. 现有顶级方案复用矩阵（核心章节）

> 检索日期：2026-10-15。所有项目/论文均为检索中实际命中的真实成果。
> 许可列说明：标注 `*` 者为检索中**未直接读到 LICENSE 文本**，落地前需复核；其余为检索结果中明示。

### 3.1 数值线性代数 / BLAS / LAPACK

| 领域/模块 | 世界顶级现成方案（项目 + 链接） | 许可 | 关键算法/数学 | 复用方式 | 超越点 |
|---|---|---|---|---|---|
| 稠密线性代数 | **LAPACK** — <https://github.com/Reference-LAPACK/lapack> | BSD-3 | LU/PA=LU、Cholesky、Householder QR、Golub-Kahan SVD、对称 EVD、Schur | **移植算法**（不移植 Fortran 代码） | 单文件 JS、零构建；每步附可验证残差界 |
| 基础线性代数子程序 | **OpenBLAS** — <https://github.com/OpenMathLib/OpenBLAS> | BSD-3 | GEMM 分块/打包、GEMV、AXPY、DOT | **移植分块与打包策略**到 JS `Float64Array` | 浏览器内可跑 + 自检面板可视化 |
| JS 矩阵库（参照） | **ml-matrix** — <https://github.com/mljs/matrix> | MIT | LU/QR/Cholesky/EVD/SVD、线性相关 | **参照实现做交叉验证** | 更小、可逐行审计、自带不变量断言 |
| JS 数值分析（参照） | **numeric.js** — <https://github.com/sloisel/numeric> | MIT | 稀疏矩阵、无约束优化、特征值/奇异值 | **参照实现做交叉验证** | 增加硬不变量与确定性报告 |
| NIST 矩阵库 JS 移植 | **JAMA** — <https://math.nist.gov/javanumerics/jama/>；**JamaJS** — <https://github.com/dragonfly-ai/JamaJS>（Apache-2.0）；**JAMA4JS** — <https://github.com/conceptualGabrielPutnam/JAMA4JS> | NIST 公有域 / Apache-2.0 | Cholesky、LU、QR、SVD、对称与非对称 EVD | **参照实现做交叉验证**（`‖A−QR‖`、`‖Av−λv‖` 双路比对） | 我们的实现无 Maven/Java 工具链依赖，Node 直接跑 |
| 科学计算 JS 生态 | **@stdlib/stdlib** — <https://github.com/stdlib-js/stdlib> | Apache-2.0 | BLAS 风格 typed-array API、特殊函数、统计分布 | **参照 API 约定**（命名/参数序/stride） | API 对齐生态 + 不变量层是增量 |

### 3.2 优化器 / 自动微分

| 领域/模块 | 世界顶级现成方案 | 许可 | 关键算法/数学 | 复用方式 | 超越点 |
|---|---|---|---|---|---|
| 一阶自适应优化器 | **Adam** (arXiv 1412.6980)；**AdamW / Decoupled Weight Decay** (arXiv 1711.05101) | 论文 | 一阶/二阶矩 EMA、偏置校正、解耦权重衰减 | **直接移植公式** | 偏置校正项可验证；权重衰减等价性可测 |
| 拟牛顿 / 二阶 | **L-BFGS 两圈递归** — Nocedal & Wright《Numerical Optimization》2nd, Algorithm 7.5；`scipy.optimize` L-BFGS-B；PyTorch `optim.LBFGS` | 书 / BSD-3 / BSD-3 | two-loop recursion、Strong Wolfe 线搜索、Sherman-Morrison-Woodbury | **移植算法** | 线搜索条件写成断言；`sᵀy > 0` 曲率条件自检 |
| JS/TS 优化库（参照） | **numopt-js** — <https://github.com/takuto-NA/numopt-js> | 开源* | GD / BFGS / L-BFGS / Gauss-Newton / LM / CMA-ES、Adjoint BFGS | **参照实现做交叉验证** | 收敛轨迹 + 梯度范数曲线纳入自检 |
| 反向模式自动微分 | **micrograd** — <https://github.com/karpathy/micrograd> | MIT | 计算图 tape + 拓扑序反向传播 | **移植架构**并扩展算子 | 增加 gradcheck 硬不变量（相对误差 ≤ 1e-6） |
| 工业级 autodiff（参照） | **JAX** — <https://github.com/jax-ml/jax>（jvp / vjp / linearize / grad-of-grad） | Apache-2.0 | 前向 JVP、反向 VJP、线性化 | **参照语义** | 不做 JIT/编译器，只取数学语义 + 不变量 |
| 极简全栈 DL（参照） | **tinygrad** — <https://github.com/tinygrad/tinygrad> | MIT | 惰性求值、算子融合、autograd、~25 个底层算子后端 | **参照极简设计哲学** | 我们的"极简"约束更硬：单文件、零依赖 |
| JS 标量 autograd（参照） | **tinygrad (JS)** — <https://github.com/arafathusayn/tinygrad> | MIT | 标量 Value + 运算符重载 + MLP | **参照实现做交叉验证** | 增加数值-解析梯度比对断言 |

### 3.3 Transformer / 注意力 / 大模型组件

| 领域/模块 | 世界顶级现成方案 | 许可 | 关键算法/数学 | 复用方式 | 超越点 |
|---|---|---|---|---|---|
| IO 感知注意力 | **FlashAttention** — <https://github.com/Dao-AILab/flash-attention>（arXiv 2205.14135）；FlashAttention-2（arXiv 2307.08691） | BSD-3 | 分块 tiling + **online softmax**（ running max / running sum 递推） | **移植 online-softmax 数学**（不移植 CUDA kernel） | 硬不变量：分块结果 vs 朴素 softmax 注意力，`max abs diff ≤ 1e-12` |
| 旋转位置编码 | **RoPE / RoFormer** — arXiv 2104.09864；实现参照 `flash_attn.layers.rotary`、`llama.cpp` | 论文 / BSD-3 / MIT | 复数旋转、interleaved 与 non-interleaved 两种布局 | **直接移植公式** | 不变量：内积 `⟨RoPE(q,m), RoPE(k,n)⟩` 仅依赖 `m−n` |
| 归一化 | **RMSNorm** — arXiv 1910.07467 | 论文 | 去均值中心化的 RMS 缩放 | **直接移植** | 不变量：缩放后 RMS = 1 ± 1e-12 |
| 门控 FFN | **SwiGLU / GLU Variants** — arXiv 2002.05202 | 论文 | `silu(W1x) ⊙ (W2x)`，隐维取 8/3·d 并取整到 256 倍数 | **直接移植** | 与 GELU-FFN 对照基线 |
| 分组查询注意力 | **GQA** — arXiv 2305.13245（Ainslie et al. 2023） | 论文 | KV 头分组共享（LLaMA-3 8B：32 Q 头 / 8 KV 头） | **直接移植** | 不变量：KV cache 体积比 = `n_kv / n_q`，可量化 |
| 参考大模型实现 | **nanoGPT** / **lit-llama** / **NanoLLM** — <https://github.com/omkarbhad/nanollm>（MIT，LLaMA-3 架构全家桶） | MIT | RMSNorm + RoPE + SwiGLU + GQA + FlashAttn-2 + KV Cache + DDP | **参照架构组织方式** | 我们同架构但纯 JS、可自检、无 GPU 依赖 |
| 分词器 | **SentencePiece** — <https://github.com/google/sentencepiece>；**tiktoken** — <https://github.com/openai/tiktoken> | Apache-2.0 / MIT | BPE 合并学习、正则化、字节级回退 | **移植 BPE 合并算法** | 不变量：`decode(encode(s)) === s` 往返一致 |

### 3.4 推理优化（KV Cache / 量化 / 投机解码）

| 领域/模块 | 世界顶级现成方案 | 许可 | 关键算法/数学 | 复用方式 | 超越点 |
|---|---|---|---|---|---|
| 本地推理引擎 | **llama.cpp** — <https://github.com/ggml-org/llama.cpp>（+ `ggml`） | MIT | **k-quants 分块量化**（Q4_K_M / Q5_K_M / Q6_K）、GGUF 容器、**KV cache 量化**、mmap | **移植量化数学 + KV cache 量化思想** | 不变量：量化-反量化相对误差界可测；长序列 cache 占用可测 |
| 分页 KV cache / 服务 | **vLLM** — <https://github.com/vllm-project/vllm>；**PagedAttention** — arXiv 2309.06180 | Apache-2.0 / 论文 | 非连续分页 KV cache、连续批处理、prefix caching、chunked prefill | **移植分页分配思想**为纯 JS block table | 不变量：分页 cache 与非分页 cache 输出逐 token 一致 |
| 投机解码 | **Speculative Decoding** — arXiv 2211.17192（Leviathan et al.）；**EAGLE** — <https://github.com/SafeAILab/EAGLE>；**Medusa** — <https://github.com/FasterDecoding/Medusa> | 论文 / 开源* | draft 小模型生成 K token → 目标模型单前向并行验证 → **拒绝采样修正** | **移植 draft-verify + 拒绝采样** | 硬不变量：投机解码输出分布与目标模型分布**严格一致**（拒绝采样可证，非近似） |
| 权重量化 | **GPTQ** — arXiv 2210.17323（<https://github.com/IST-DASLab/gptq>）；**AWQ** — arXiv 2306.00978（<https://github.com/mit-han-lab/llm-awq>） | 开源* | 二阶（Hessian）误差补偿逐列量化 / 激活感知通道保护 | **移植核心公式**（小矩阵规模，不追求 GPU kernel） | 不变量：层输出 MSE 与基线对比表 |
| 浏览器端推理（参照） | **transformers.js** — <https://github.com/huggingface/transformers.js>（ONNX Runtime Web，WASM / WebGPU） | Apache-2.0 | pipeline API、dtype (`fp32/fp16/q8/q4`)、模型缓存 | **参照 API 与 dtype 约定** | 我们零模型下载、纯算法内核、可离线自检 |

### 3.5 检索（向量 / 稀疏 / 混合）

| 领域/模块 | 世界顶级现成方案 | 许可 | 关键算法/数学 | 复用方式 | 超越点 |
|---|---|---|---|---|---|
| 图索引 ANN | **hnswlib** — <https://github.com/nmslib/hnswlib>（Malkov & Yashunin, arXiv 1603.09320） | Apache-2.0 | 分层可导航小世界图、层间跳转、`M` / `efConstruction` / `ef` 调参 | **移植 HNSW 算法**为纯 JS | 硬不变量：`recall@10 ≥ 0.95` 对比暴力精确检索（同数据集同种子） |
| 大规模向量检索（参照） | **FAISS** — <https://github.com/facebookresearch/faiss> | MIT | IVF / PQ / OPQ / HNSW | **参照实现做交叉验证** | 单文件、浏览器可跑 |
| 稀疏检索 | **Okapi BM25**；`rank_bm25`；**pg_textsearch**（Timescale，Postgres BM25 扩展） | 开源* | IDF、TF 饱和、长度归一化 | **直接移植公式** | 不变量：与暴力打分逐条一致 `≤ 1e-12` |
| 混合检索融合（参照） | **RRF（Reciprocal Rank Fusion）**、ColBERT late interaction、MMR | 论文 | 倒数排名融合、最大边界相关性去重 | **直接移植 + 参照** | 融合顺序稳定性可测 |

### 3.6 强化学习

| 领域/模块 | 世界顶级现成方案 | 许可 | 关键算法/数学 | 复用方式 | 超越点 |
|---|---|---|---|---|---|
| 策略优化 | **PPO** — arXiv 1707.06347；**CleanRL** — <https://github.com/vwxyzjn/cleanrl>（JMLR 2022，`ppo.py` 321 LOC 单文件） | 论文 / MIT | GAE `A_t = Σ(γλ)^l δ_{t+l}`、clip surrogate、entropy bonus | **直接借鉴单文件范式 + 移植公式** | 硬不变量：λ=1 时 GAE 与蒙特卡洛回报一致 `≤ 1e-10` |
| 基于模型 RL / 规划 | **MuZero** — arXiv 1911.08265；**EfficientZero** — <https://github.com/YeWR/EfficientZero>（NeurIPS 2021） | 开源* | 潜动态模型 + MCTS（UCT / PUCT）、奖励与价值预测 | **移植 MCTS/PUCT 数学** | 硬不变量：**Bellman 残差 → 0**；UCT 选择序列可复现 |
| 值方法（参照） | CleanRL `dqn.py` / `c51.py` / `sac_continuous_action.py` | MIT | 分布al RL、最大熵 RL | **参照实现做交叉验证** | 同一份不变量清单 |

### 3.7 概率推断 / 概率图模型

| 领域/模块 | 世界顶级现成方案 | 许可 | 关键算法/数学 | 复用方式 | 超越点 |
|---|---|---|---|---|---|
| 通用贝叶斯推断 | **PyMC** — <https://github.com/pymc-devs/pymc>；**Stan** — <https://github.com/stan-dev/stan> | Apache-2.0 / BSD-3 | **NUTS**（多层自适应 HMC，dual-averaging 步长自适应）、ADVI、SMC | **参照实现做交叉验证** | 不变量：NUTS 后验均值 vs 共轭解析后验 `|Δ| ≤ 3·MCSE` |
| JAX 采样器（参照） | **NumPyro** / **BlackJAX** | Apache-2.0 | HMC / NUTS / SMC 可组合采样器 | **参照语义** | 纯 JS 版本 |
| JS 概率编程（参照） | **WebPPL** — <https://github.com/probmods/webppl>；**@tangent.to/mc** — <https://github.com/tangent-to/mc>（JS NUTS + Gelman-Rubin R-hat + ESS） | MIT / GPL-3.0（注意：参照不引入，避免许可污染） | MCMC、NUTS、R-hat / ESS 诊断 | **仅参照数学与诊断口径**（不引入代码，规避 GPL） | 我们的诊断口径与 Stan 一致，但自写实现 |
| 概率图模型 | Koller & Friedman《Probabilistic Graphical Models》；和积 / max-product / junction tree | 书 | 因子、变量消去、信念传播、团树 | **移植算法** | 不变量：小图上 BP 结果与暴力枚举边缘一致 |

### 3.8 图算法

| 领域/模块 | 世界顶级现成方案 | 许可 | 关键算法/数学 | 复用方式 | 超越点 |
|---|---|---|---|---|---|
| 图分析 | **NetworkX** — <https://github.com/networkx/networkx>；**EasyGraph**（Patterns 2023）；graph-tool | BSD-3 / 开源* | Dijkstra、Bellman-Ford、BFS/DFS、PageRank、Louvain、SCC | **移植算法 + 参照做交叉验证** | 不变量：Dijkstra 与 Bellman-Ford 距离一致；PageRank `Σp = 1 ± 1e-12` |
| 图表示学习（参照） | node2vec / DeepWalk / GCN / GraphSAGE / GAT | 论文 | 随机游走、消息传递、注意力聚合 | **移植聚合算子** | 置换等变性（permutation equivariance）不变量 |

### 3.9 数值稳定 / 验证工程

| 领域/模块 | 世界顶级现成方案 | 许可 | 关键算法/数学 | 复用方式 | 超越点 |
|---|---|---|---|---|---|
| 稳定化基本式 | **scipy.special.logsumexp** — <https://github.com/scipy/scipy>（max-shift 技巧）；`log1p` / `expm1` / `hypot` | BSD-3 | `logsumexp(x) = m + log Σ exp(x_i − m)` | **直接移植** | 硬不变量：与高精度（mpmath/大数）对照 `≤ 1e-13` |
| 性质测试（property-based） | **fast-check** — <https://github.com/dubzzz/fast-check>；Hypothesis | MIT / MPL-2.0 | 性质测试、反例收缩（shrinking） | **参照范式**（自写轻量版，零依赖） | 每模块 ≥ 200 随机用例的不变量扫描 |
| 浮点误差分析 | Higham《Accuracy and Stability of Numerical Algorithms》 | 书 | 前向/后向误差、条件数、补偿求和（Kahan / Neumaier） | **移植补偿求和** | 不变量：Kahan 求和 vs 高精度求和误差界 |

### 3.10 复用矩阵统计

| 复用方式 | 条目数（约） | 说明 |
|---|---|---|
| 直接移植公式 / 数学 | 11 | AdamW、RoPE、RMSNorm、SwiGLU、GQA、BM25、RRF、online-softmax、logsumexp、PPO-clip、BPE |
| 移植算法（重新表达为纯 JS） | 8 | LAPACK 分解族、L-BFGS two-loop、HNSW、MCTS/PUCT、Dijkstra/Louvain、变量消去、NUTS 数学、GPTQ/AWQ 核心式 |
| 参照实现做交叉验证 | 9 | ml-matrix、numeric.js、JAMA/JamaJS、numopt-js、tinygrad、FAISS、CleanRL、PyMC/Stan、NetworkX |
| 参照架构 / API / 范式 | 5 | nanoGPT/lit-llama/NanoLLM、transformers.js、@stdlib API、JAX autodiff 语义、CleanRL 单文件范式 |
| **合计** | **≈ 33 条** | 覆盖需求方要求的全部 11 个领域 |

**未检索到权威方案的项**（如实标注）：
- 「1024 项微指令级操作」的统一分类法：学界/工业界**无现成标准 taxonomy**，本项目自行定义（见第 6 章）。
- 纯 JS 单文件 + 每模块硬不变量的**组合形态**：未检索到同类开源项目（现有 JS 库均不带硬不变量自检面板）。此为本项目原创增量。

---

## 4. 用户故事

| # | 用户故事（`作为 <角色>，我 <动作>，以便 <价值>`） |
|---|---|
| US-1 | 作为**算法研究者**，我打开单个 HTML 文件就能看到某算法的不变量数值与残差，以便在不配置任何环境的前提下验证实现是否真的正确。 |
| US-2 | 作为**前端工程师**，我 `import` 一个 `.js` 文件到现有项目，以便零构建、零依赖地获得一个可信的 AI 算法内核。 |
| US-3 | 作为**AI 工程师**，我用本库作为独立第二实现跑同一组输入，以便交叉验证我自研的 PyTorch/C++ 版本是否引入实现偏差。 |
| US-4 | 作为**教师**，我在课堂上调大矩阵规模并观察 Jacobi 残差变化，以便让学生直观理解数值误差与收敛性的关系。 |
| US-5 | 作为**作者本人（晨星）**，我看到每个模块的 `REUSE.md` 都指向一个真实顶级来源，以便确认自己没有重复造轮子。 |
| US-6 | 作为**开源贡献者**，我只需补一个族下的 1 条微指令及其不变量就能提交 PR，以便低门槛参与一个大型项目。 |
| US-7 | 作为**使用者**，我在 CI 中跑 `node run-all.js` 并拿到退出码 0，以便把本库当作可信依赖纳入自己的流水线。 |
| US-8 | 作为**研究者**，我读 README 中「算法 → 顶级来源 → 链接」三列表，以便快速沿引用链追溯到原始论文与工程实现。 |
| US-9 | 作为**性能工程师**，我看自检面板上的耗时与内存峰值，以便判断该算法在浏览器端是否可接受。 |
| US-10 | 作为**长期维护者**，我按 8 族 × 8 组 × 16 微指令的编号体系定位任意一项，以便在 1024 项规模下仍能精确定位与增量补齐。 |

---

## 5. 需求池（P0 / P1 / P2）

> P0 = 必须交付；P1 = 应当交付；P2 = 可选加分。

| ID | 需求 | 优先级 | 验收标准（可验证） | 复用来源 |
|---|---|---|---|---|
| R-01 | 建立统一仓库骨架与命名规范（`forge` 系列、单文件、零依赖） | P0 | `ls src/*.js` ≥ 8 个模块；每文件头部含 `@reuse` 来源注释块 | CleanRL 单文件范式、tinygrad 极简哲学 |
| R-02 | 每个模块内置 `selfCheck()`，Node 端 `node <mod>.js --self-check` 退出码 0 | P0 | 8/8 模块退出码 0；JSON 报告含 `passed/failed/duration` | 自研（参照 property-based 范式） |
| R-03 | 每模块 ≥ 1 条硬不变量，阈值写死在代码中 | P0 | 硬不变量总数 ≥ 8；阈值常量集中导出 | 用户「验证总纲」 |
| R-04 | `run-all.js` 聚合运行器，无头一键跑全部模块 | P0 | `node run-all.js` 退出码 0，≤ 120 s，输出汇总表 | 自研 |
| R-05 | 单文件 HTML 自检面板（内联 CSS/JS，零外部依赖） | P0 | 双击打开即运行；断言通过率 100%；无 `http(s)://` 外链请求 | 自研 |
| R-06 | 每模块 `REUSE.md`：顶级来源 + 链接 + 数学式 + 复用方式 + 超越点 | P0 | 覆盖率 100%；外链随机抽检 5 条可达 | 第 3 章矩阵 |
| R-07 | 数值线性代数内核：LU / QR / Cholesky / SVD / 对称 EVD | P0 | 不变量：`‖A−LU‖`、`‖A−QR‖`、`‖Av−λv‖ ≤ 1e-12`、正交归一 `≤ 1e-14` | LAPACK、JAMA、ml-matrix |
| R-08 | 数值稳定工具包：`logsumexp` / `softmax-stable` / `log-sigmoid` / Kahan 求和 | P0 | 与高精度对照误差 `≤ 1e-13`；`x=[1000,1000]` 不溢出 | scipy.special.logsumexp、Higham |
| R-09 | 优化器族：SGD / Momentum / Adam / AdamW / L-BFGS(two-loop) | P0 | 凸二次函数上收敛到 `‖∇f‖ ≤ 1e-8`；AdamW 与 Adam+正则 在 λ=0 时等价 | Adam 1412.6980、AdamW 1711.05101、Nocedal & Wright |
| R-10 | 自动微分引擎（tape-based 反向模式）+ gradcheck | P0 | 与中心差分梯度相对误差 `≤ 1e-6`（对齐 `nn-forge` 9.23e-7、`attn-forge` 5.86e-7） | micrograd、JAX 语义 |
| R-11 | 注意力内核：online-softmax 分块 attention | P0 | 分块 vs 朴素 `max abs diff ≤ 1e-12` | FlashAttention 2205.14135 / 2307.08691 |
| R-12 | RoPE / RMSNorm / SwiGLU / GQA 四件套 | P1 | RoPE 相对位置只依赖 `m−n`；RMS 归一化后 RMS=1±1e-12；GQA KV 比例可测 | 2104.09864、1910.07467、2002.05202、2305.13245 |
| R-13 | KV Cache（含 GQA 共享 + 分页 block table） | P1 | 分页与非分页逐 token 输出一致 `≤ 1e-12`；长序列内存下降可测 | vLLM / PagedAttention 2309.06180、llama.cpp |
| R-14 | 量化：对称/非对称 int8、per-channel、k-quants 分块 | P1 | 量化-反量化相对误差界表；与 fp32 层输出 MSE 对比表 | llama.cpp k-quants、GPTQ、AWQ |
| R-15 | 投机解码（draft-verify + 拒绝采样） | P1 | 输出分布与原模型**严格一致**（拒绝采样证明 + 1000 次采样分布 KS 检验） | 2211.17192、EAGLE、Medusa |
| R-16 | HNSW 向量检索（纯 JS） | P1 | `recall@10 ≥ 0.95` vs 暴力精确（固定种子数据集） | hnswlib / arXiv 1603.09320、FAISS |
| R-17 | BM25 稀疏检索 + 倒排索引 | P1 | 与暴力打分逐条一致 `≤ 1e-12` | Okapi BM25、rank_bm25 |
| R-18 | 混合检索融合（RRF / 加权 / MMR） | P2 | 融合结果稳定、可复现；顺序无关性可测 | RRF、ColBERT |
| R-19 | PPO（含 GAE、clip surrogate） | P1 | λ=1 时 GAE 与蒙特卡洛回报一致 `≤ 1e-10`；clip 边界行为单测 | PPO 1707.06347、CleanRL |
| R-20 | MCTS / PUCT（MuZero 风格规划） | P1 | Bellman 残差单调下降 → `≤ 1e-6`；UCT 选择序列可复现 | MuZero 1911.08265、EfficientZero |
| R-21 | 概率推断：NUTS + 共轭解析对照 | P1 | 正态-正态共轭后验均值 `|Δ| ≤ 3·MCSE`；R-hat ≤ 1.01 | PyMC、Stan、NumPyro、BlackJAX |
| R-22 | 图算法：Dijkstra / Bellman-Ford / PageRank / Louvain | P1 | Dijkstra 与 Bellman-Ford 距离一致；PageRank `Σp=1±1e-12`；modularity 可复现 | NetworkX、EasyGraph |
| R-23 | 微指令层（L10）脚手架：8 族 × 8 组 × 16 项编号体系 + 占位清单 | P1 | 生成 `1024` 行清单文件，编号唯一、总数核算 = 1024 | 自研 taxonomy（第 6 章） |
| R-24 | `README.md`：三连复现命令 + 复用矩阵 + 不变量总表 | P0 | 陌生人按 README 操作一次成功 | 自研 |
| R-25 | LICENSE + 署名 `晨星 / CJX0712` + GitHub 一次 push 闭环 | P0 | 远端仓库可见；`git log` 作者署名正确 | 自研 |
| R-26 | CI 工作流（GitHub Actions）：Node 22 无头跑 `run-all.js` | P1 | badge 绿；退出码 0 | 自研 |
| R-27 | 浏览器/Node 双端一致性：同一份代码两端结果逐位一致 | P1 | 双端 JSON 报告 `hash` 相同 | 自研 |
| R-28 | 依赖与故障自治：环境自检脚本 + 修复建议输出 | P2 | 缺 Node/权限异常时给出可执行修复命令，不静默失败 | 自研 |

**优先级汇总**：P0 = 11 条（R-01…R-11、R-24、R-25）；P1 = 14 条；P2 = 3 条。

---

## 6. 十层分解架构的 PRD 视角映射

### 6.1 分层原则（PRD 建议）

采用**二进制分解**：第 n 层承载 `2ⁿ` 个功能单元，层间为「组合 → 细化」关系。
总数核算：`Σ(n=1..10) 2ⁿ = 2 + 4 + 8 + 16 + 32 + 64 + 128 + 256 + 512 + 1024 = 2046`。

### 6.2 十层映射表

| 层号 | 层名 | 该层职责 | 模块数 | 已有成果 | 本次待补 |
|---|---|---|---|---|---|
| **L1** | 根层 · 双核 | 一切的地基：**数值内核**（Float64Array 上的确定运算）+ **验证内核**（断言/不变量/报告） | **2** | 各 forge 项目各自内联了这两块，尚未抽成公共层 | 抽出 `core-numeric` + `core-verify` 两个公共内核 |
| **L2** | 基础层 · 四支柱 | 线性代数、概率统计、数值方法、优化理论 | **4** | `pca-forge` 隐式含线代 + 统计（**存量待迁移**，见 §12） | 概率分布族、数值稳定包、优化理论文档化 |
| **L3** | 方法层 · 八算法族 | 分解、回归、分类、聚类、降维、采样、图、序列 | **8** | — | 8 个算法族入口（每族可独立自检） |
| **L4** | 模型层 · 16 经典模型 | 线性/岭/Lasso、Logistic、SVM、朴素贝叶斯、kNN、决策树、随机森林、PCA、GMM、k-means、HMM、LDA、GBDT、Isolation Forest、谱聚类、CRF | **16** | `pca-forge`、`gmm-forge`（2/16，**存量待迁移**，见 §12） | 其余 14 个 |
| **L5** | 深度层 · 32 组件 | MLP、激活、损失、归一化、卷积、池化、RNN/LSTM/GRU、Dropout、初始化、正则…（32 项） | **32** | `nn-forge`（约 1/32 覆盖面） | 补齐 32 项组件与各自不变量 |
| **L6** | 大模型层 · 64 组件 | Attention 全家桶、RoPE/RMSNorm/SwiGLU/GQA、MoE、Tokenizer、位置编码变体、长上下文策略…（64 项） | **64** | `attn-forge`（注意力核心 1 项） | 63 项 |
| **L7** | 系统层 · 128 组件 | KV Cache、量化、投机解码、连续批处理、显存调度、算子融合、混合精度、分布式策略…（128 项） | **128** | — | 128 项（本轮优先 KV Cache / 量化 / 投机解码 3 组） |
| **L8** | 应用与智能体层 · 256 能力 | RAG、工具调用、规划、记忆、多智能体路由、评测、Guardrail、可观测…（256 项） | **256** | — | 256 项（本轮不展开，仅预留编号） |
| **L9** | 组合层 · 512 复合算子 | 由 L1–L8 元素组合而成的可复用 pipeline / 组合子（512 项） | **512** | — | 512 项（本轮不展开，仅预留编号） |
| **L10** | **微指令层 · 1024 微指令** | 不可再分的最小可验证操作单元（1024 项） | **1024** | — | 分类法定义 + 脚手架清单（见 6.3） |
| **合计** | — | — | **2046** | 4 个 forge 系列 | 见各层「待补」 |

### 6.3 L10 微指令层：1024 项分类法（Taxonomy）

**核算公式**：`1024 = 8 族(Family) × 8 组(Group) × 16 微指令(Op)`
- 8 族 × 8 组 = **64 组**
- 64 组 × 16 微指令 = **1024 项** ✔
- 编号规范：`L10-F{k}-G{j}-O{m}`，`k,j ∈ [0,7]`，`m ∈ [0,15]`；全局唯一 ID = `F{k}G{j}O{m}`（如 `F0G1O3`）。

| 族 ID | 族名 | 8 个组（Group） |
|---|---|---|
| **F0** | TENSOR 张量/线性代数 | F0G0 BLAS-1 向量 · F0G1 BLAS-2 矩阵-向量 · F0G2 BLAS-3 矩阵-矩阵 · F0G3 分解（LU/QR/Chol/SVD/EVD） · F0G4 求解与最小二乘 · F0G5 稀疏与结构化 · F0G6 张量收缩与重排（einsum/transpose/广播） · F0G7 范数与度量 |
| **F1** | NUMERIC 数值稳定 | F1G0 浮点与误差（ulp/Kahan/EF T） · F1G1 稳定化基本式（logsumexp/log1p/softmax-stable） · F1G2 特殊函数（lgamma/erf/digamma） · F1G3 一维求根与优化（Brent/Newton/黄金分割） · F1G4 插值与积分（样条/Gauss-Legendre） · F1G5 迭代法（幂法/Lanczos/Arnoldi/Givens） · F1G6 随机与采样（sfc32/Ziggurat/alias/Gumbel-max） · F1G7 病态诊断（rcond/平衡化/迭代精化/NaN 守卫） |
| **F2** | OPTIM 优化与自动微分 | F2G0 一阶方法（SGD/Momentum/Adam/AdamW…） · F2G1 二阶与拟牛顿（L-BFGS/BFGS/CG/LM） · F2G2 学习率调度 · F2G3 正则与约束（decay/clip/投影/EMA） · F2G4 自动微分（tape/JVP/VJP/HVP/checkpoint） · F2G5 梯度检验 · F2G6 损失与目标（CE/focal/对比/DPO/PPO-clip） · F2G7 收敛诊断 |
| **F3** | PROB 概率与推断 | F3G0 连续分布密度 · F3G1 离散分布 · F3G2 多元分布 · F3G3 采样器（MH/Gibbs/HMC/NUTS/SMC） · F3G4 变分推断（ELBO/ADVI/重参数化） · F3G5 EM 与混合模型 · F3G6 统计量与信息量（Welford/熵/MI/KL） · F3G7 概率图模型（变量消去/BP/团树） |
| **F4** | GRAPH 图与离散结构 | F4G0 表示与构造 · F4G1 遍历（BFS/DFS/拓扑/SCC） · F4G2 最短路（Dijkstra/BF/Floyd/A*） · F4G3 MST 与匹配 · F4G4 中心性与排序（PageRank/HITS/Katz） · F4G5 社区发现（Louvain/Leiden/谱聚类） · F4G6 流与割（Dinic/推重标号/最小割） · F4G7 图上学习（随机游走/Laplacian/消息传递） |
| **F5** | ATTN 注意力与序列 | F5G0 注意力核心（scaled-dot/causal/online-softmax/GQA/MQA） · F5G1 位置编码（sinusoidal/RoPE/ALiBi/YaRN） · F5G2 归一化（LayerNorm/RMSNorm/QK-Norm） · F5G3 FFN 与激活（GELU/SwiGLU/MoE 路由） · F5G4 分词与嵌入（BPE/WordPiece/Unigram） · F5G5 解码与采样（top-k/top-p/beam） · F5G6 KV Cache 与长上下文 · F5G7 训练稳定（混合精度/梯度检查点/ZeRO 分片） |
| **F6** | SEARCH 检索/量化/调度 | F6G0 向量检索（暴力/HNSW/IVF/PQ） · F6G1 稀疏检索（BM25/TF-IDF/倒排/WAND） · F6G2 混合检索与融合（RRF/重排/MMR/ColBERT） · F6G3 相似度与距离 · F6G4 量化与压缩（int8/per-channel/GPTQ/AWQ/k-quants/fp8） · F6G5 投机解码与加速（draft/verify/Medusa/EAGLE/早退） · F6G6 调度与批处理（连续批处理/prefix cache/paged） · F6G7 哈希与数据结构（Bloom/Count-Min/MinHash/并查集/堆） |
| **F7** | VERIFY 验证/自检/交付 | F7G0 不变量断言 · F7G1 交叉验证（解析vs数值/暴力vs贪心/参照vs主实现） · F7G2 自检运行时（self-check/退出码/JSON 报告/面板） · F7G3 文档与报告生成 · F7G4 确定性与可复现（种子/黄金文件/快照） · F7G5 性能基准（FLOPs/吞吐/分位/内存/回归阈值） · F7G6 错误与容错（校验/降级/日志/依赖自检） · F7G7 交付与开源（仓库结构/LICENSE/CI/署名） |

**每组的 16 条微指令生成规则**（保证可穷举、不重不漏）：
1. `O0` 该组**核心算子**（canonical op）
2. `O1–O3` 核心算子的**三个变体**（如 inplace / batched / masked）
3. `O4–O6` **边界与退化情形**处理（空输入 / 单点 / 零权重 / NaN 守卫）
4. `O7–O9` **数值稳定版本**（max-shift、补偿求和、对数域）
5. `O10–O11` **梯度 / 导数**版本（VJP / JVP）
6. `O12–O13` **不变量断言**（该组主不变量 + 次不变量）
7. `O14` **参照实现交叉验证**钩子
8. `O15` **性能与规模**探针（耗时/内存/规模缩放记录）

核算复核：`16 条/组 × 8 组/族 × 8 族 = 16 × 64 = 1024` ✔

**本轮（PRD 阶段）只交付**：F0–F7 的分类法定义 + `docs/L10-taxonomy.md` 中 1024 行编号清单（编号唯一、总数 = 1024 可脚本校验）。实现按 P0/P1 分批填。

---

## 7. UI / 交互设计约束

| 约束项 | 要求 |
|---|---|
| 产物形态 | **①** 纯 JS 单文件算法内核（`src/<name>-forge.js`），浏览器 `<script>` 与 Node `import/require` 双可用；**②** 单文件 HTML 自检面板（`index.html`，CSS/JS 全部内联） |
| 外部依赖 | **零**。单文件 HTML 中不得出现任何 `http://` / `https://` 资源引用（字体、CDN、图片均禁止） |
| 语言 | 界面文案**中文为主**；算法名、数学符号、关键数值保留英文/原文（如 `logsumexp`、`‖Av−λv‖`、`recall@10`） |
| 时间显示 | 用人话相对时间：`刚刚` / `3 分钟前` / `2 小时前` / `昨天` / `3 天前`；超过 7 天显示具体日期 `2026-10-08` |
| 面板布局 | 顶部：总览条（`8/8 全绿`、总耗时、最差残差）；中部：模块卡片网格（每卡显示模块名 / 复用来源 / 不变量条目 / 状态）；底部：点击卡片展开断言明细表 |
| 状态色 | 通过 `#16a34a`（绿）· 失败 `#dc2626`（红）· 跳过 `#94a3b8`（灰）· 警告 `#d97706`（橙） |
| 数值展示 | 科学计数法统一 `1.78e-15`；阈值右侧标注判定 `≤ 1e-12 ✔` |
| 交互 | 面板提供「重跑」「复制 JSON 报告」「展开全部/折叠全部」三个按钮；键盘 `R` 重跑、`E` 展开全部 |
| 可访问性 | 每个状态色旁必须有文字标签（不能仅靠颜色传达状态） |
| 终端输出 | Node 端输出与面板同源 JSON；`--json` 参数输出机器可读报告；默认输出人类可读表格 |

---

## 8. 待确认问题

> 用户已全权授权。**以下默认按推荐方案继续，不再回头确认。**

| # | 待确认点 | 假设 / 推荐方案（默认继续） |
|---|---|---|
| Q1 | 第十层 1024 项是一次性全实现，还是先建分类法？ | **假设**：本轮只交付分类法 + 1024 行编号清单 + 脚手架；实现按 P0/P1 分批填。默认按此继续。 |
| Q2 | 十层是否严格采用 `2ⁿ` 二进制分解？ | **假设**：是。总计 2046 项，L10 = 1024。默认按此继续。 |
| Q3 | 许可选择？ | **假设**：仓库主体 **MIT**（与 ml-matrix / micrograd / tinygrad / llama.cpp 生态一致）；**禁止引入 GPL 系依赖**（如 `@tangent.to/mc` 仅参照数学，不引入代码）。默认按此继续。 |
| Q4 | 是否引入任何 npm 依赖？ | **假设**：**不引入**。运行时零依赖；仅 Node 内置模块（可考虑 `node:test` / `node:assert`）。默认按此继续。 |
| Q5 | `pca-forge` / `gmm-forge` / `nn-forge` / `attn-forge` **不在本仓库内**（位于 `C:/Users/Administrator/WorkBuddy/2026-10-12-03-38-35/jsforge/`，实测 32/32 全绿），是否要迁入？ | **假设（2026-10-16 修正）**：**迁移而非重建**。原文件算法主体**逐字不动**地复制到 `src/`，仅在外层补 `@reuse` 头 + 统一 `selfCheck()` 出口 + `REUSE.md`；迁移后必须仍为 32/32 全绿。默认按此继续。 |
| Q6 | 新增模块优先顺序？ | **假设**：先补齐 P0 的通用层（内核、稳定包、优化器、autodiff、Flash-attn 数学），再进 P1 的 LLM/系统层。默认按此继续。 |
| Q7 | 硬不变量阈值是否统一？ | **假设**：**不统一**。各模块按数学界给出独立阈值，但必须在代码中以 `INVARIANT_TOL` 常量显式声明。默认按此继续。 |
| Q8 | 是否需要 Python 侧对照？ | **假设**：**不需要**交付物中含 Python；仅在开发期用 Python 3.13 做高精度对照（mpmath），结果固化为黄金文件。默认按此继续。 |

---

## 9. In scope / Out of scope

### In scope（本轮必须做）

- 纯 JS 单文件算法内核 ≥ 8 个模块，浏览器 + Node v22 双端可跑
- 每模块 `selfCheck()` + ≥1 条硬不变量 + `REUSE.md` 顶级来源标注
- `run-all.js` 无头聚合运行器（退出码 0）
- 单文件 HTML 自检面板（零外部依赖、中文界面）
- 第 3 章复用矩阵中所有「直接移植 / 移植算法」条目对应的真实链接与数学式
- L10 的 1024 项分类法定义 + 编号清单（总数可脚本核算 = 1024）
- README（三连复现命令）+ LICENSE(MIT) + 署名 `晨星 / CJX0712` + GitHub 一次 push 闭环

### Out of scope（本轮明确不做）

- **训练大模型**：不预训练、不微调、不下载任何模型权重
- **GPU / CUDA / Triton kernel**：只取数学，不实现底层 kernel；不追求与 CUDA 实现的性能对标
- **分布式 / 多机**：不做 DDP、ZeRO 实际通信，仅做单机算法层分片语义
- **Python 交付物**：Python 仅用于开发期高精度对照，不作为交付内容
- **Web 服务 / 后端 API**：不做 HTTP 服务、不做数据库、不做用户系统
- **L8（256 项应用/智能体）与 L9（512 项组合子）的实现**：本轮仅预留编号，不实现
- **可视化图表库 / 动效**：面板只用内联 CSS，不引入图表库
- **移动端适配**：仅保证桌面浏览器可用（最小宽度 1024px 为设计基线）
- **GPL / AGPL 系依赖引入**：明确禁止

---

## 10. 文件清单

| 路径 | 说明 |
|---|---|
| `docs/PRD.md` | 本文档（产品需求文档） |
| `docs/REUSE-MATRIX.md` | 复用矩阵完整版（由第 3 章独立成文，供架构师直接引用）— *待架构师/程序生成* |
| `docs/L10-taxonomy.md` | 1024 项微指令编号清单（8 族 × 8 组 × 16 项）— *待程序生成并脚本校验总数* |
| `docs/INVARIANTS.md` | 硬不变量总表（模块 / 不变量 / 阈值 / 验证方式）— *待程序生成* |
| `src/<name>-forge.js` | 各算法模块单文件（≥ 8 个） |
| `run-all.js` | 无头聚合运行器 |
| `index.html` | 单文件自检面板 |
| `README.md` / `LICENSE` | 复现说明 / MIT 许可 + 署名 |

---

## 11. 检索过的来源清单（本文所有外链均来自以下检索）

| # | 来源 | 用途 | 链接 |
|---|---|---|---|
| 1 | ml-matrix (mljs/matrix) | JS 线性代数参照实现 | <https://github.com/mljs/matrix> |
| 2 | numeric.js (sloisel/numeric) | JS 数值分析参照 | <https://github.com/sloisel/numeric> |
| 3 | JAMA (NIST/MathWorks) | 经典矩阵库 | <https://math.nist.gov/javanumerics/jama/> |
| 4 | JamaJS | JAMA 的 JS 移植 | <https://github.com/dragonfly-ai/JamaJS> |
| 5 | JAMA4JS | JAMA 的 JS 翻译 + 验证测试 | <https://github.com/conceptualGabrielPutnam/JAMA4JS> |
| 6 | @stdlib/stdlib | JS 科学计算 API 约定 | <https://github.com/stdlib-js/stdlib> |
| 7 | numopt-js | JS 优化器（L-BFGS 等）参照 | <https://github.com/takuto-NA/numopt-js> |
| 8 | micrograd | autograd 架构来源 | <https://github.com/karpathy/micrograd> |
| 9 | JAX | autodiff 语义参照 | <https://github.com/jax-ml/jax> |
| 10 | tinygrad | 极简全栈 DL 设计哲学 | <https://github.com/tinygrad/tinygrad> |
| 11 | tinygrad (JS) | JS 标量 autograd 参照 | <https://github.com/arafathusayn/tinygrad> |
| 12 | FlashAttention (Dao-AILab) | online-softmax / tiling | <https://github.com/Dao-AILab/flash-attention> |
| 13 | ROCm/flash-attention · Building Blocks | RoPE / RMSNorm / SwiGLU 实现细节 | <https://deepwiki.com/ROCm/flash-attention/5-building-blocks> |
| 14 | NanoLLM (omkarbhad) | LLaMA-3 架构全家桶参照 | <https://github.com/omkarbhad/nanollm> |
| 15 | gpt-sota-opt | GQA+FlashAttn2+RMSNorm+SwiGLU+RoPE 组合参照 | <https://github.cc/lioarce01/gpt-sota-opt> |
| 16 | llm-from-scratch (wwj-2017-1117) | RMSNorm/SwiGLU/RoPE/FlashAttn2 教学实现 | <https://www.github.com/wwj-2017-1117/llm-from-scratch> |
| 17 | llama.cpp | k-quants / GGUF / KV cache 量化 | <https://github.com/ggml-org/llama.cpp> |
| 18 | vLLM | PagedAttention / 连续批处理 | <https://github.com/vllm-project/vllm> |
| 19 | EAGLE | 投机解码工程实现 | <https://github.com/SafeAILab/EAGLE> |
| 20 | Medusa | 并行解码头 | <https://github.com/FasterDecoding/Medusa> |
| 21 | GPTQ | 二阶权重量化 | <https://github.com/IST-DASLab/gptq> |
| 22 | AWQ | 激活感知量化 | <https://github.com/mit-han-lab/llm-awq> |
| 23 | transformers.js | 浏览器推理 API / dtype 约定 | <https://github.com/huggingface/transformers.js> |
| 24 | hnswlib | HNSW 权威实现 | <https://github.com/nmslib/hnswlib> |
| 25 | FAISS | 大规模向量检索参照 | <https://github.com/facebookresearch/faiss> |
| 26 | CleanRL | PPO 单文件范式 | <https://github.com/vwxyzjn/cleanrl> |
| 27 | CleanRL 论文 (arXiv 2111.08819) | 单文件实现论证 | <https://arxiv.org/pdf/2111.08819> |
| 28 | EfficientZero | MuZero 系样本高效 RL | <https://github.com/YeWR/EfficientZero> |
| 29 | EfficientZero 论文 (arXiv 2111.00210) | 算法细节 | <https://arxiv.org/pdf/2111.00210v2> |
| 30 | PyMC | NUTS / ADVI 参照 | <https://github.com/pymc-devs/pymc> |
| 31 | Stan | NUTS 金标准实现 | <https://github.com/stan-dev/stan> |
| 32 | Awesome-PPL 清单 | PPL 全景（含 WebPPL/NumPyro/BlackJAX） | <https://github.es/shohei81/awsome-PPL> |
| 33 | @tangent.to/mc | JS NUTS + R-hat/ESS（**仅参照数学，不引入代码**，GPL-3.0） | <https://github.com/tangent-to/mc> |
| 34 | WebPPL | 浏览器端 PPL | <https://github.com/probmods/webppl> |
| 35 | NetworkX | 图算法参照 | <https://github.com/networkx/networkx> |
| 36 | EasyGraph (Patterns 2023) | 高性能图分析参照 | <https://www.cell.com/patterns/fulltext/S2666-3899(23)00218-0> |
| 37 | SciPy | `logsumexp` 等稳定化实现 | <https://github.com/scipy/scipy> |
| 38 | fast-check | 性质测试范式 | <https://github.com/dubzzz/fast-check> |
| 39 | SentencePiece | BPE/Unigram 分词 | <https://github.com/google/sentencepiece> |
| 40 | tiktoken | BPE 编码参照 | <https://github.com/openai/tiktoken> |

**论文引用**（均在上述检索结果中被明确提及）：
- FlashAttention-2 — arXiv 2307.08691 · FlashAttention — arXiv 2205.14135
- RoPE / RoFormer — arXiv 2104.09864 · RMSNorm — arXiv 1910.07467 · SwiGLU / GLU Variants — arXiv 2002.05202
- GQA — Ainslie et al. 2023（arXiv 2305.13245） · PagedAttention — arXiv 2309.06180
- Speculative Decoding — Leviathan et al.（arXiv 2211.17192）
- GPTQ — arXiv 2210.17323 · AWQ — arXiv 2306.00978
- HNSW — Malkov & Yashunin（arXiv 1603.09320）
- PPO — Schulman et al.（arXiv 1707.06347） · MuZero — arXiv 1911.08265
- Adam — arXiv 1412.6980 · AdamW — arXiv 1711.05101
- 书目：《Numerical Optimization》(Nocedal & Wright, 2nd, Algorithm 7.5)、《Accuracy and Stability of Numerical Algorithms》(Higham)、《Probabilistic Graphical Models》(Koller & Friedman)、《Bayesian Data Analysis》(Gelman et al.)

**合规声明**：本文所有项目名与链接均来自 2026-10-15 的实际网络检索结果，未编造。许可列标注 `*` 者需在落地前以各仓库 `LICENSE` 文件复核。

---

---

## 12. 存量资产清单（2026-10-16 磁盘核实 + 实测）

> 背景：项目启动时主理人提供的「历史上下文」提到 4 个 forge 成果。经 2026-10-16 实际勘察：
> **它们确实存在，但不在本仓库内**，位于上一个会话目录。已实际运行验证。

### 12.1 存量位置与实测结果

| 项 | 内容 |
|---|---|
| 存量路径 | `C:/Users/Administrator/WorkBuddy/2026-10-12-03-38-35/jsforge/` |
| 本仓库内现状 | `src/` 与 `tests/` **为空**（0 个文件）；git 已 init，但 `main` 分支**尚无 commit** |
| 实测命令 | `node run-node.js` |
| **实测结果** | **32/32 全部通过**（4 模块 × 8 断言） |

### 12.2 四个存量模块明细（实测数值）

| 模块 | 行数 | 自检 | 关键实测数值 |
|---|---|---|---|
| `pca-forge.js` | 263 | **8/8** ✅ | 协方差对称 `\|A−Aᵀ\|=0.00e+0`；半正定 `λmin=1.38e+0`；**特征向量正交 `2.22e-16`**；归一 `2.22e-16`；重构 `\|A−VΛVᵀ\|=2.22e-15`；**Jacobi 残差 `‖Av−λv‖=1.09e-15`**；**全分量重构无损 `‖X−X̂‖=1.78e-15`**；方差解释率 `evr=[0.755,0.245] Σ=1.0000` |
| `gmm-forge.js` | 271 | **8/8** ✅ | 混合权重 `Σπ=1.000000000000`；协方差对称 `0.00e+0`；半正定 `λmin=1.10e-1`；**对数似然单调不降** `LL[79]=−831.55`；`‖μ−μ*‖=0.013`、`‖Σ−Σ*‖=0.056`；**logsumexp 稳定** `lse=−998.559810`（朴素 `=−Inf`）；双簇聚类一致率 `acc=1.0000` |
| `nn-forge.js` | 257 | **8/8** ✅ | 训练后损失下降 `L0=0.814 → L1=0.0000`；**数值梯度 `maxRelErr=2.57e-11`**（阈值 1e-5，历史标称 **9.23e-7** 量级）；单步 SGD 损失下降 `0.3647→0.1849`；XOR 可解 `acc=1.00`；softmax `Σ=1.000000000000` |
| `attn-forge.js` | 173 | **8/8** ✅ | 权重每行 `max\|Σ−1\|=0.00e+0`；非负 `min=0.00e+0`；**数值梯度 `maxRelErr=3.38e-9`**（历史标称 **5.86e-7** 量级）；因果掩码上三角 `0.00e+0`；加权求和平验 `O=ΣP·V`，`maxerr=0.00e+0` |
| 附属文件 | | | `index.html`（5.6 KB 单文件面板）、`run-node.js`、`serve.js`、`package.json`、`README.md`、`gmm-forge` 同级 |

### 12.3 对 PRD 的影响（已回改）

| 原表述 | 修正后 |
|---|---|
| S1：「已有 4 个模块」 | **「4 个为存量迁移 + 新增 ≥4」**（存量不在仓库内，需先迁入） |
| §8 Q5：「保留原有实现不动」 | **「迁移而非重建」**：算法主体逐字不动复制到 `src/`，外层补 `@reuse` 头 + 统一 `selfCheck()` 出口 + `REUSE.md`；迁移后仍须 32/32 全绿 |
| §6.2 L2/L4「已有成果」 | 标注为「**存量待迁移**」，非「已在仓库」 |
| S2 引用数值 | **Jacobi 残差 = 1.09e-15**（非 1.78e-15）；1.78e-15 是「全分量重构无损」指标，二者不可混用 |

### 12.4 给架构师/程序员的落地指令

1. **不要从零重写这 4 个模块** —— 直接 `cp` 自 `2026-10-12-03-38-35/jsforge/*.js` 到 `src/`。
2. 迁移后**先跑一次基线**（预期 32/32），再补 `@reuse` 头与统一契约包装；**每包装一次复跑一次**，保证不退化。
3. 这 4 个模块贡献 **32 条已通过的不变量**，可直接计入 S2（下限 ≥8）——即 **S2 在迁移完成时就已达标**，新增模块的 52 条为增量。
4. `index.html`（存量 5.6 KB 单文件面板）可作为 R-05 自检面板的**起点**直接改造，不必从零写。

---

*文档结束 · 创新旗舰 AI · 产品经理 许清楚 · 署名 晨星（CJX0712）*
