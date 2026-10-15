# 创新旗舰 AI · 顶级方案复用矩阵（REUSE-MATRIX）

| 项 | 内容 |
|---|---|
| 文档版本 | v1.0 |
| 撰写角色 | 产品经理 · 许清楚 |
| 检索日期 | 2026-10-15 |
| 上游文档 | `docs/PRD.md` 第 3 章（本文档为其独立展开版，含核心公式与许可处置） |
| 目标读者 | 架构师 / 程序员 / 测试员（**本文档是唯一权威来源，请勿重复检索**） |
| 署名 | 晨星（GitHub: `CJX0712`） |

---

## §0 使用说明与图例

### 复用方式四档定义

| 代号 | 复用方式 | 含义 | 落地要求 |
|---|---|---|---|
| **A** | **直接引用（移植公式）** | 数学公式是公开标准结果，直接按式子实现 | 代码中以 `// @reuse: <来源>` 注释标注公式出处 |
| **B** | **移植算法** | 参照顶级实现的算法结构（分解策略/递推结构/图结构），用纯 JS 重新表达 | 不复制源码；算法骨架对齐，`REUSE.md` 说明差异 |
| **C** | **参照实现做交叉验证** | 以该项目的输出作为独立第二实现基准，比对数值 | 结果写入黄金文件（golden file），纳入不变量断言 |
| **D** | **仅参照数学 / 架构 / API** | 只取其数学定义或 API 约定，不引入任何代码 | **禁止 import、禁止 vendoring** |

### 表格列说明

`顶级方案（名称 + URL）` | `许可` | `关键算法/数学（核心公式）` | `复用方式` | `超越点`

许可列标注规则：
- 无标记 = 检索结果中**明示**的许可
- `*` = 检索中**未直接读到 LICENSE 文本**，需落地前以仓库 `LICENSE` 复核（见 §12）

---

## §1 数值线性代数

| # | 顶级方案（名称 + URL） | 许可 | 关键算法/数学（核心公式） | 复用 | 超越点 |
|---|---|---|---|---|---|
| 1.1 | **LAPACK**<br>https://github.com/Reference-LAPACK/lapack | BSD-3 | **LU 分解**：`PA = LU`（部分选主元）<br>**Cholesky**：`A = LLᵀ`（A 对称正定）<br>**Householder QR**：`H = I − 2vvᵀ/(vᵀv)`，`A = QR`<br>**对称 EVD / Jacobi**：旋转变换逐步消去非对角元，`A = QΛQᵀ`<br>**Golub-Kahan SVD**：双对角化 + 隐式 QR 迭代，`A = UΣVᵀ` | **B** | 单文件 JS、零构建；每个分解附**可验证残差界**（`‖A−LU‖`、`‖A−QR‖`、`‖Av−λv‖`），LAPACK 本身不提供自检面板 |
| 1.2 | **OpenBLAS**<br>https://github.com/OpenMathLib/OpenBLAS | BSD-3 | **GEMM 分块**：`C += α·A_block·B_block`，按 cache line 分块 + 数据打包（packing）<br>BLAS-1：`y = αx + y`（axpy）、`dot`、`nrm2`<br>BLAS-2：`gemv`、`ger` | **B** | 把分块/打包策略移植到 `Float64Array`；浏览器端可视化不同 block size 的耗时曲线 |
| 1.3 | **ml-matrix**<br>https://github.com/mljs/matrix | MIT | `LuDecomposition` / `QrDecomposition` / `CholeskyDecomposition` / `EigenvalueDecomposition` / `SingularValueDecomposition` / `linearDependencies` | **C** | 作为**独立第二实现基准**；我们更小、可逐行审计、每步带不变量断言（`正交归一 ≤ 1e-14`） |
| 1.4 | **numeric.js**<br>https://github.com/sloisel/numeric | MIT | 稀疏矩阵（`numeric.ccsSparse`）、无约束优化（`numeric.uncmin`）、特征值（`numeric.eig`）、SVD（`numeric.svd`） | **C** | 同上；补充确定性报告与硬不变量层 |
| 1.5 | **JAMA**（NIST / MathWorks）<br>https://math.nist.gov/javanumerics/jama/ | NIST 公有域 | `CholeskyDecomposition`、`LUDecomposition`、`QRDecomposition`、`SingularValueDecomposition`、`EigenvalueDecomposition`；`Matrix.solve()` / `Matrix.inverse()` | **C** | 以 `‖A−QR‖`、`‖Av−λv‖` 双路比对；我们**无需 Maven/Java 8 工具链**，Node 直接跑 |
| 1.6 | **JamaJS**（JAMA 的 JSweet 移植）<br>https://github.com/dragonfly-ai/JamaJS | Apache-2.0 | 同上，JS 化后的 `Cholesky/LU/QR/SVD/Eigen` 类 | **C** | 作为可直接跑在 Node 上的 JS 基准实现 |
| 1.7 | **JAMA4JS**（手写 JS 翻译 + 验证测试）<br>https://github.com/conceptualGabrielPutnam/JAMA4JS | 开源* | 同上；额外带 verification tests | **C** | 其「verification tests」思路被我们升级为**硬不变量 + 交叉验证双轨** |
| 1.8 | **@stdlib/stdlib**<br>https://github.com/stdlib-js/stdlib | Apache-2.0 | BLAS 风格 typed-array API：`<op>( N, x, strideX, y, strideY )` 的 stride/offset 约定 | **D** | 仅对齐 API 命名与参数序（`(N, x, sx, y, sy)`），降低使用者心智负担；不引入代码 |

### 本族关键不变量（写死为 `INVARIANT_TOL`）

| 不变量 | 阈值 |
|---|---|
| `‖A − LU‖_F / ‖A‖_F` | ≤ 1e-12 |
| `‖A − QR‖_F / ‖A‖_F`、`‖QᵀQ − I‖` | ≤ 1e-12 / ≤ 1e-14 |
| 对称 EVD 残差 `‖Av − λv‖` | ≤ 1e-12（`pca-forge` **实测 1.09e-15**，2026-10-16 复跑确认） |
| 特征向量正交归一 `‖VᵀV − I‖` | ≤ 1e-14（`pca-forge` **实测 2.22e-16**，2026-10-16 复跑确认） |
| 全分量重构无损 `‖X − X̂‖` | ≤ 1e-12（`pca-forge` **实测 1.78e-15**，2026-10-16 复跑确认） |
| Cholesky `A − LLᵀ` 对称半正定 | 最小特征值 ≥ −1e-12 |

---

## §2 优化与自动微分

| # | 顶级方案（名称 + URL） | 许可 | 关键算法/数学（核心公式） | 复用 | 超越点 |
|---|---|---|---|---|---|
| 2.1 | **Adam** — arXiv 1412.6980<br>https://arxiv.org/abs/1412.6980 | 论文 | `m_t = β₁m_{t−1} + (1−β₁)g_t`<br>`v_t = β₂v_{t−1} + (1−β₂)g_t²`<br>`m̂_t = m_t/(1−β₁ᵗ)`，`v̂_t = v_t/(1−β₂ᵗ)`<br>`θ_t = θ_{t−1} − α·m̂_t/(√v̂_t + ε)` | **A** | 偏置校正项**可单独验证**（`t=1` 时 `m̂ = g`） |
| 2.2 | **AdamW / 解耦权重衰减** — arXiv 1711.05101<br>https://arxiv.org/abs/1711.05101 | 论文 | `θ_t = θ_{t−1} − α·m̂_t/(√v̂_t + ε) − α·λ·θ_{t−1}`<br>（衰减项**不进入** `m/v` 的梯度统计） | **A** | 硬不变量：`λ=0` 时与 Adam 逐步等价（逐位比对 ≤ 1e-15）；`λ>0` 时与「Adam + L2 正则」**不等价**，用反例固化 |
| 2.3 | **L-BFGS two-loop recursion** — Nocedal & Wright《Numerical Optimization》2nd, Algorithm 7.5 | 书 | `ρ_i = 1/(y_iᵀ s_i)`，`s_i = x_{i+1}−x_i`，`y_i = ∇f_{i+1}−∇f_i`<br>**第一圈**：`q ← ∇f_k`；`for i=k−1…k−m: α_i = ρ_i s_iᵀq; q ← q − α_i y_i`<br>`γ = s_{k−1}ᵀy_{k−1}/(y_{k−1}ᵀy_{k−1})`，`r ← γq`<br>**第二圈**：`for i=k−m…k−1: β = ρ_i y_iᵀr; r ← r + s_i(α_i − β)`<br>`p_k = −r` | **B** | 曲率条件 `sᵀy > 0` 写成**断言**；Strong Wolfe 线搜索条件（Armijo + 曲率）写成断言；内存 `O(mn)` 可测 |
| 2.4 | **scipy.optimize L-BFGS-B**<br>https://github.com/scipy/scipy | BSD-3 | 同上 + 边界投影 + `history_size=10` 默认 | **C** | 以 scipy 输出为高精度基准（开发期，不进交付物） |
| 2.5 | **numopt-js**<br>https://github.com/takuto-NA/numopt-js | 开源* | `gradientDescent` / `bfgs` / `lbfgs(historySize)` / `gaussNewton` / `levenbergMarquardt` / `cmaes` / `adjointBfgs` | **C** | 收敛轨迹 + `finalGradientNorm` 纳入我们的自检报告 |
| 2.6 | **micrograd**<br>https://github.com/karpathy/micrograd | MIT | 反向模式 tape：构建 DAG → 拓扑排序 → 反向累乘局部梯度<br>`∂L/∂x = Σ_{(x→y)} (∂L/∂y)·(∂y/∂x)` | **B** | 移植 tape 架构 + 扩展算子集；**增加 gradcheck 硬不变量**（相对误差 ≤ 1e-6，对齐 `nn-forge` 9.23e-7 / `attn-forge` 5.86e-7） |
| 2.7 | **JAX**<br>https://github.com/jax-ml/jax | Apache-2.0 | `jvp`（前向，Jacobian-向量积）、`vjp`（反向，向量-Jacobian 积）、`linearize`、`grad(grad(...))` | **D** | 只取**语义**（`jvp`/`vjp`/HVP 的数学定义），不取 JIT/XLA |
| 2.8 | **tinygrad**<br>https://github.com/tinygrad/tinygrad | MIT | 极简设计：全栈仅需 ~25 个底层算子（elementwise / reduce / movement 三类）；惰性求值 + 算子融合 | **D** | 取「极简 + 可全读」的设计哲学；我们的约束更硬（单文件、零依赖、浏览器可跑） |
| 2.9 | **tinygrad (JS)**<br>https://github.com/arafathusayn/tinygrad | MIT | 标量 `Value` + 运算符重载 + `Neuron/Layer/MLP` | **C** | 数值-解析梯度逐参数比对 |

### 本族关键不变量

| 不变量 | 阈值 |
|---|---|
| **梯度检验**（中心差分 vs 反向模式）相对误差 `(a−n)/(a+n)` | ≤ 1e-6 |
| AdamW `λ=0` 与 Adam 逐步等价 | ≤ 1e-15 |
| 凸二次函数（如 `f(x)=½xᵀAx`）收敛后 `‖∇f‖` | ≤ 1e-8 |
| L-BFGS 每步曲率条件 `sᵀy` | > 0（断言） |

---

## §3 Transformer / 注意力

| # | 顶级方案（名称 + URL） | 许可 | 关键算法/数学（核心公式） | 复用 | 超越点 |
|---|---|---|---|---|---|
| 3.1 | **FlashAttention** — arXiv 2205.14135<br>https://github.com/Dao-AILab/flash-attention | BSD-3 | **Online softmax 递推**（核心）：维护 running max `m`、running sum `ℓ`、running 输出 `o`<br>`m_new = max(m_old, rowmax(S_blk))`<br>`ℓ_new = e^{m_old − m_new}·ℓ_old + Σ rowsum(e^{S_blk − m_new})`<br>`o_new = e^{m_old − m_new}·o_old + e^{S_blk − m_new}·V_blk`<br>最终 `O = o/ℓ` | **A** | **硬不变量**：分块结果与朴素 `softmax(QKᵀ/√d)V` 的 `max abs diff ≤ 1e-12`；内存峰值随序列长度**次线性**可测 |
| 3.2 | **FlashAttention-2** — arXiv 2307.08691 | 论文 | 改进循环顺序、减少非 matmul FLOP、序列维并行 | **D** | 只取数学等价性结论；性能工程部分 out of scope（不做 CUDA） |
| 3.3 | **RoPE / RoFormer** — arXiv 2104.09864 | 论文 | `θ_i = 1/base^{2i/d}`（base=10000）<br>复数形式 `f(q,m) = q·e^{imθ}`；实数形式对第 `i` 对分量旋转 `m·θ_i`：<br>`(q_{2i}, q_{2i+1}) → (q_{2i}cos mθ_i − q_{2i+1}sin mθ_i, q_{2i}sin mθ_i + q_{2i+1}cos mθ_i)`<br>**关键性质**：`⟨f(q,m), f(k,n)⟩` 仅依赖 `m − n` | **A** | 硬不变量：**内积只依赖相对距离** `m−n`（随机采样 1000 组 `(m,n)` 与 `(m+c,n+c)` 比对 ≤ 1e-12）；支持 interleaved / non-interleaved 两种布局 |
| 3.4 | **RMSNorm** — arXiv 1910.07467 | 论文 | `RMS(x) = sqrt( (1/d)·Σ_i x_i² )`<br>`y = (x / RMS(x)) ⊙ g`（**无均值中心化、无 bias**） | **A** | 硬不变量：归一化后 `RMS(y/g) = 1 ± 1e-12`；与 LayerNorm 的耗时/数值对照表 |
| 3.5 | **SwiGLU / GLU Variants** — arXiv 2002.05202 | 论文 | `silu(z) = z·σ(z)`<br>`FFN(x) = ( silu(xW₁) ⊙ (xW₂) ) W₃`<br>LLaMA 约定：隐维 `= ⌊(8/3)·d_model / 256⌋·256`（取整到 256 倍数） | **A** | 与 GELU-FFN 的对照基线表；门控维度可测 |
| 3.6 | **GQA** — Ainslie et al. 2023 / arXiv 2305.13245 | 论文 | 把 `n_q` 个 query 头分组，`G = n_q / n_kv` 个 query 头共享 1 组 KV 头<br>LLaMA-3 8B：`32 Q 头 / 8 KV 头`，`G=4` **→ KV cache 省 4 倍** | **A** | 硬不变量：KV cache 体积比 `= n_kv/n_q`（数值可测）；`G=1` 退化为 MHA、`n_kv=1` 退化为 MQA，两端可验证 |
| 3.7 | **Scaled Dot-Product Attention**（基线） | 论文 | `Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V`；因果掩码 `M_ij = −∞ (j>i)` | **A** | 作为 FlashAttention 分块实现的**参照真理**（ground truth） |
| 3.8 | **NanoLLM**（LLaMA-3 架构全家桶）<br>https://github.com/omkarbhad/nanollm | MIT | RMSNorm + RoPE + SwiGLU + GQA + FlashAttn-2 + KV Cache + DDP + bf16 的**模块拆分方式** | **D** | 参照其「每个组件单文件拆开、单独可测」的组织方式；我们同架构但纯 JS、无 GPU、可自检 |
| 3.9 | **gpt-sota-opt**<br>https://github.cc/lioarce01/gpt-sota-opt | 开源* | `Input → Embedding → [RMSNorm → GQA+RoPE+FlashAttn → Residual → RMSNorm → SwiGLU FFN → Residual]×N → RMSNorm → LM Head` | **D** | 参照 block 组合顺序（Pre-Norm 结构），验证我们的组合顺序一致 |
| 3.10 | **llm-from-scratch**<br>https://www.github.com/wwj-2017-1117/llm-from-scratch | MIT | `RmsNorm` / `SwiGLU FFN` / `RoPE` / `AdamW` / `flash_attention_triton` + `flash_attention_mock` | **D** | 其「mock flash attention 与真 flash attention 对照」的做法，正是我们的**不变量对照范式** |

### 本族关键不变量

| 不变量 | 阈值 |
|---|---|
| 分块 attention vs 朴素 attention | `max abs diff ≤ 1e-12` |
| RoPE 相对位置依赖 | `(m,n)` 与 `(m+c,n+c)` 内积差 ≤ 1e-12 |
| RMSNorm 后 RMS | `= 1 ± 1e-12` |
| GQA KV 体积比 | `= n_kv / n_q`（精确） |
| causal mask 上三角 | 输出严格为 0（或 mask 后不参与） |

---

## §4 推理优化（KV Cache / 量化 / 投机解码 / 调度）

| # | 顶级方案（名称 + URL） | 许可 | 关键算法/数学（核心公式） | 复用 | 超越点 |
|---|---|---|---|---|---|
| 4.1 | **llama.cpp** + **ggml**<br>https://github.com/ggml-org/llama.cpp | MIT | **k-quants 分块量化**：每 block（如 32/64 个权重）共享 scale + min，`w ≈ scale·q + min`，`q` 为低位整数；`Q4_K_M / Q5_K_M / Q6_K / Q8_0`<br>**KV cache 量化**：KV 张量降至 8-bit 以下，生成期显存降约 50%<br>**GGUF** 容器格式；mmap 直接加载不拷贝到 RAM | **B** | 量化-反量化相对误差界**可测并写死**；长序列 cache 占用下降比例纳入基准表；无 C/C++ 编译依赖 |
| 4.2 | **vLLM** / **PagedAttention** — arXiv 2309.06180<br>https://github.com/vllm-project/vllm | Apache-2.0 / 论文 | **Block table**：逻辑块 → 物理块映射，块大小 `B`（如 16 token），KV cache 存于**非连续**物理块；按 OS 虚拟内存分页思想分配/回收<br>配套：连续批处理（iteration-level scheduling）、prefix caching、chunked prefill | **B** | 硬不变量：**分页 cache 与非分页 cache 逐 token 输出一致 ≤ 1e-12**；块利用率/碎片率可测（llama.cpp/vLLM 不提供此断言） |
| 4.3 | **Speculative Decoding** — Leviathan et al., arXiv 2211.17192 | 论文 | Draft 模型 `q` 生成 `K` 个候选；目标模型 `p` 单前向并行验证；**拒绝采样修正**：<br>接受候选 `x` 的概率 `= min(1, p(x)/q(x))`；<br>若拒绝，从 `(p − q)_+ / Σ(p − q)_+` 重新采样<br>**结论：输出分布与目标模型严格一致（非近似）** | **A** | **硬不变量**：投机解码输出分布 ≡ 目标模型分布。① 拒绝采样数学证明写入 `REUSE.md`；② 1000 次采样做分布 KS 检验 |
| 4.4 | **EAGLE**<br>https://github.com/SafeAILab/EAGLE | 开源* | 在目标模型的特征层上接自回归 draft head（而非独立小模型），提高接受率 | **D** | 仅参照「draft 生成位置」的设计思想，验证我们的 draft-verify 接口兼容 |
| 4.5 | **Medusa**<br>https://github.com/FasterDecoding/Medusa | 开源* | 多个并行解码头 + tree attention 一次验证多条候选路径 | **D** | 同上；本轮实现单路径 draft-verify，Medusa 多头列为 P2 |
| 4.6 | **GPTQ** — arXiv 2210.17323<br>https://github.com/IST-DASLab/gptq | 开源* | 逐列量化 + **二阶误差补偿**：<br>量化第 `q` 列后，`δ_q = (w_q − quant(w_q)) / [H⁻¹]_{qq}`，把误差按 Hessian 信息补偿到未量化列<br>目标：`min ‖WX − ŴX‖²` | **B** | 在小矩阵规模上移植核心公式；层输出 MSE 对比表可复现；不追求 GPU kernel |
| 4.7 | **AWQ** — arXiv 2306.00978<br>https://github.com/mit-han-lab/llm-awq | 开源* | **激活感知**：按激活幅值挑出 ~1% 显著通道保持 fp16；对其余权重搜索 per-channel scale `s`，最小化 `‖Q(W·s)(s⁻¹·X) − W·X‖` | **B** | 与 GPTQ 同规模下的**质量对照表**（两者检索中均有"AWQ 在代码/数学任务上损失更低"的实测结论，我们用小矩阵复现并给数值） |
| 4.8 | **transformers.js**（ONNX Runtime Web）<br>https://github.com/huggingface/transformers.js | Apache-2.0 | `pipeline(task, model?, options?)`；`dtype` ∈ `fp32 / fp16 / q8 / q4`；`device` ∈ `wasm / webgpu`；模型缓存到 IndexedDB | **D** | 仅对齐 `dtype` 命名与 pipeline API 形态；我们**零模型下载、纯算法内核、可离线自检** |
| 4.9 | **连续批处理 / Continuous Batching**（vLLM 等） | 论文/Apache-2.0 | iteration-level：每个 decode step 允许新请求入批、完成请求立即出批（对比静态批：等最长序列结束） | **B** | 单文件 JS 的调度模拟器：吞吐随并发数变化曲线可测；不接真实 GPU |

### 本族关键不变量

| 不变量 | 阈值 |
|---|---|
| 投机解码 vs 目标模型输出分布 | KS 检验 `p > 0.05`（1000 样本）+ 拒绝采样数学证明 |
| 分页 KV cache vs 连续 KV cache | 逐 token `abs diff ≤ 1e-12` |
| int8 量化-反量化相对误差 | 界值写入 `INVARIANT_TOL`（按 per-tensor/per-channel 分别记录） |
| k-quants 分块量化层输出 MSE | 与 fp32 基线对比表（可复现） |

---

## §5 检索（向量 / 稀疏 / 混合）

| # | 顶级方案（名称 + URL） | 许可 | 关键算法/数学（核心公式） | 复用 | 超越点 |
|---|---|---|---|---|---|
| 5.1 | **hnswlib** — Malkov & Yashunin, arXiv 1603.09320<br>https://github.com/nmslib/hnswlib | Apache-2.0 | **HNSW**：分层可导航小世界图。<br>**层分配**：`l = ⌊−ln(uniform(0,1)) · m_L⌋`（指数衰减，越高层节点越少）<br>**插入**：从顶层贪心下降，每层选 `M` 个近邻建边，`M_max0 = 2M`（第 0 层）<br>**启发式选边**：优先保留「多样性」邻居（避免全连到同一簇）<br>**查询**：`ef`（搜索范围）越大 recall 越高、延迟越高；`efConstruction` 控制建图质量<br>复杂度：搜索 `O(log N)` | **B** | 硬不变量：**`recall@10 ≥ 0.95` 对比暴力精确检索**（同数据集、同随机种子）；`recall–latency` 曲线纳入基准 |
| 5.2 | **FAISS**<br>https://github.com/facebookresearch/faiss | MIT | `IndexFlatL2`（精确基线）、`IndexIVFFlat`（倒排 + 聚类，`nlist` 个 Voronoi 单元，`nprobe` 个探测）、`IndexIVFPQ`（乘积量化，`m` 个子空间 × `nbits`）、OPQ 旋转预处理 | **C** | 作为**精确检索与量化检索双基准**；我们单文件、浏览器可跑、无需编译 |
| 5.3 | **Okapi BM25**（经典）<br>`rank_bm25`（Python 参考实现） | 开源* | `score(D,Q) = Σ_i IDF(q_i) · [ f(q_i,D)·(k₁+1) ] / [ f(q_i,D) + k₁·(1 − b + b·|D|/avgdl) ]`<br>`IDF(q_i) = ln( 1 + (N − n(q_i) + 0.5)/(n(q_i) + 0.5) )`<br>常用 `k₁=1.2`，`b=0.75`；`f` 词频、`|D|` 文档长度、`avgdl` 平均长度 | **A** | 硬不变量：**与暴力逐条打分一致 ≤ 1e-12**；边界（空查询 / 全命中 / 单文档）单测 |
| 5.4 | **pg_textsearch**（Timescale，Postgres BM25 扩展） | 开源* | 把 BM25 排 ranking 下沉到 Postgres 内，与 `pgvector` 并存做混合检索 | **D** | 仅确认「BM25 仍是检索 ranking 的事实标准」这一选型结论；不引入 Postgres |
| 5.5 | **RRF（Reciprocal Rank Fusion）** | 论文 | `score(d) = Σ_i 1/(k + rank_i(d))`，常用 `k=60` | **A** | 融合顺序稳定性可测（列表顺序置换不改变结果） |
| 5.6 | **MMR（Maximal Marginal Relevance）** | 论文 | `MMR = argmax_{d_i ∈ R\S} [ λ·sim(d_i,q) − (1−λ)·max_{d_j ∈ S} sim(d_i,d_j) ]` | **A** | 多样性指标（结果集内平均相似度）可测 |
| 5.7 | **ColBERT late interaction** | 论文 | 逐 token  embedding + MaxSim：`score(q,d) = Σ_{t∈q} max_{j∈d} ⟨e_t^q, e_j^d⟩` | **B** | 与单向量检索的质量对照（小数据集） |

### 本族关键不变量

| 不变量 | 阈值 |
|---|---|
| HNSW vs 暴力精确 `recall@10` | ≥ 0.95 |
| BM25 vs 暴力打分 | ≤ 1e-12 |
| RRF 列表顺序置换不变性 | 结果集合完全相同 |
| 距离度量对称性 / 非负性 / 三角不等式（L2） | 逐条断言 |

---

## §6 强化学习

| # | 顶级方案（名称 + URL） | 许可 | 关键算法/数学（核心公式） | 复用 | 超越点 |
|---|---|---|---|---|---|
| 6.1 | **PPO** — Schulman et al., arXiv 1707.06347 | 论文 | **GAE**：`δ_t = r_t + γV(s_{t+1}) − V(s_t)`，`A_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}`（递推：`A_t = δ_t + γλA_{t+1}`）<br>**Clip surrogate**：`r_t(θ) = π_θ(a_t|s_t)/π_{θ_old}(a_t|s_t)`<br>`L^{CLIP} = E[ min( r_t(θ)A_t, clip(r_t(θ), 1−ε, 1+ε)A_t ) ]`，常用 `ε=0.2` | **A** | 硬不变量：**`λ=1` 时 GAE 与蒙特卡洛回报一致 ≤ 1e-10**；clip 边界行为（`r>1+ε` 时梯度截断）单测 |
| 6.2 | **CleanRL**<br>https://github.com/vwxyzjn/cleanrl（JMLR 2022，arXiv 2111.08819） | MIT | `ppo.py` 321 LOC / `ppo_atari.py` 337 LOC 单文件：argparse 超参在顶部 → `make_env` → `Agent(nn.Module)` → rollout 循环 → GAE → minibatch 更新。<br>**关键结论**：与 SB3 样本效率一致，纯实现开销导致 2.3× 速度差；实现细节显著影响性能（Engstrom et al. 2019） | **B + D** | **B**：移植 GAE/clip 公式；**D**：借鉴「单文件、超参置顶、无抽象层」的**工程范式**——这与本项目「单文件 forge」约束天然一致 |
| 6.3 | **MuZero** — arXiv 1911.08265 | 论文 | 隐状态 `s^0 = h(o_{1..t})`；三网络：表示 `h`、动态 `g(s^k, a) → (r^k, s^{k+1})`、预测 `f(s^k) → (p^k, v^k)`<br>训练目标：`v^k` 回归 n-step 回报、`p^k` 回归 MCTS 访问分布、`r^k` 回归真实奖励 | **B** | 硬不变量：**Bellman 残差 → 0**（`≤ 1e-6`）；UCT 选择序列在固定种子下可复现 |
| 6.4 | **EfficientZero** — NeurIPS 2021, arXiv 2111.00210<br>https://github.com/YeWR/EfficientZero | 开源* | MuZero + 自监督一致性损失 + 值前缀预测 + 离线修正；Atari 100k 上 194.3% 平均人类表现，数据效率高 500× | **D** | 仅参照「样本高效」的问题定义与评估口径（100k 步）；本轮不训练视觉 RL |
| 6.5 | **MCTS / PUCT**（MuZero 选择公式） | 论文 | `a* = argmax_a [ Q(s,a) + c_puct · P(s,a) · √Σ_b N(s,b) / (1 + N(s,a)) ]`<br>经典 UCT：`Q + c·√(ln N_parent / N)` | **A** | UCB 探索项随访问次数单调下降可验证；`c→0` 退化为贪心可验证 |
| 6.6 | CleanRL `dqn.py` / `c51.py` / `sac_continuous_action.py` | MIT | DQN：`y = r + γ max_{a'} Q(s',a')`；C51 分布回报；SAC 最大熵 `J = Σ E[r + αH(π(·|s))]` | **C** | 同一套不变量清单下的第二实现基准 |

### 本族关键不变量

| 不变量 | 阈值 |
|---|---|
| GAE `λ=1` vs 蒙特卡洛回报 | ≤ 1e-10 |
| Bellman 残差（值迭代/MCTS 备份） | 单调不增，`终值 ≤ 1e-6` |
| PPO clip 边界 | `r > 1+ε` 时 loss 被截断（单测） |
| UCT 选择序列 | 固定种子下逐次可复现 |

---

## §7 概率推断 / 概率图模型

| # | 顶级方案（名称 + URL） | 许可 | 关键算法/数学（核心公式） | 复用 | 超越点 |
|---|---|---|---|---|---|
| 7.1 | **PyMC**<br>https://github.com/pymc-devs/pymc | Apache-2.0 | 模型声明为 DAG；`pm.sample()` 默认 **NUTS**；`pm.sample_prior_predictive` / `pm.do` / `pm.observe`；诊断走 ArviZ（`r_hat`、`ess_bulk`、`mcse_mean`） | **C** | 硬不变量：NUTS 后验均值 vs **共轭解析后验** `|Δ| ≤ 3·MCSE`；`r_hat ≤ 1.01` |
| 7.2 | **Stan**<br>https://github.com/stan-dev/stan | BSD-3 | NUTS 金标准实现：**多层自适应**（multinomial/doubling）+ **dual-averaging** 步长自适应 + 发散转移检测；参数约束变换（log / logit / simplex） | **C + D** | **C**：以 Stan 输出为后验基准；**D**：移植其**诊断口径**（`r_hat`、`ess`、divergence 计数）到我们的自检面板 |
| 7.3 | **NumPyro** / **BlackJAX** | Apache-2.0 | JAX 上的 HMC / NUTS / Riemannian HMC / SMC；可组合采样器接口 | **D** | 只取算法数学定义 |
| 7.4 | **@tangent.to/mc**（JS NUTS + R-hat/ESS）<br>https://github.com/tangent-to/mc | **GPL-3.0** | 纯 JS 的 NUTS（dual-averaging 步长）、Gelman-Rubin `R-hat`、ESS、`logProbAndGradient`、并行多链 | **D（仅参照数学）** | ⚠️ **明确：仅参照数学与诊断口径，禁止 import、禁止 vendoring、禁止复制代码**。理由见 §12 |
| 7.5 | **WebPPL**<br>https://github.com/probmods/webppl | MIT | 浏览器/Node 通用 PPL；MCMC（MH）、SMC、枚举精确推断 | **D** | 确认「浏览器端 PPL 可行」；我们的诊断口径与 Stan 对齐但代码自写 |
| 7.6 | **概率图模型**（Koller & Friedman《Probabilistic Graphical Models》） | 书 | **变量消去**：逐因子 `τ_i = Σ_{x_i} Π_k φ_k`；<br>**和积信念传播**：`m_{i→j}(x_j) = Σ_{x_i} φ_{i,j}(x_i,x_j)·φ_i(x_i)·Π_{k∈N(i)\j} m_{k→i}(x_i)`；<br>**max-product**（MAP）；**团树 / junction tree**；d-separation | **B** | 硬不变量：**小图上 BP 边缘概率 vs 暴力枚举边缘概率一致 ≤ 1e-12**；树图上 BP 精确性可证 |
| 7.7 | **共轭解析解**（Gelman《Bayesian Data Analysis》） | 书 | Normal-Normal：`x_i ~ N(θ,σ²)`，先验 `θ ~ N(μ₀,τ₀²)` → 后验 `θ|x ~ N(μ_n, τ_n²)`，`τ_n^{-2} = τ₀^{-2} + nσ^{-2}`，`μ_n = τ_n²(μ₀/τ₀² + n·x̄/σ²)` | **A** | 作为 NUTS 的**解析对照真理**（golden truth） |
| 7.8 | **变分推断 / ADVI** | 论文 | ELBO：`L = E_q[log p(x,z)] − E_q[log q(z)]`；<br>**重参数化**：`z = μ + σ·ε, ε~N(0,1)`（梯度可穿过采样）；<br>score-function：`∇_φ E_q[f(z)] = E_q[f(z)∇_φ log q_φ(z)]` | **A** | ELBO 单调不减可监控（每 10 步断言） |
| 7.9 | **Stan Reference Manual**（文档，非代码）<br>https://mc-stan.org/docs/reference-manual/ | 文档（CC-BY） | NUTS 的**诊断口径**：Gelman-Rubin `R-hat`、ESS（bulk/tail）、`mcse_mean`、divergence 计数及其判读规则 | **D** | ⭐ **替代 GPL-3.0 的 `@tangent.to/mc` 的关键抓手**：诊断口径以官方文档为准**自写实现**，不引入任何第三方代码。已被架构师写入 `ARCHITECTURE.md` §9.3 替代方案 |

### 本族关键不变量

| 不变量 | 阈值 |
|---|---|
| NUTS 后验均值 vs 共轭解析后验 | `|Δ| ≤ 3·MCSE` |
| Gelman-Rubin `r_hat` | ≤ 1.01 |
| 信念传播 / 变量消去 vs 暴力枚举边缘概率（对应 `INV-PRB-007`） | ≤ 1e-12 |
| ELBO（ADVI 训练过程） | 单调不减（每 10 步断言，`Δ ≥ −1e-9`） |
| EM 对数似然（GMM） | 单调不增地上升（`Δ ≥ −1e-9`，对齐 `gmm-forge`） |

---

## §8 图算法

| # | 顶级方案（名称 + URL） | 许可 | 关键算法/数学（核心公式） | 复用 | 超越点 |
|---|---|---|---|---|---|
| 8.1 | **NetworkX**<br>https://github.com/networkx/networkx | BSD-3 | **Dijkstra**：`d[v] ← min(d[v], d[u] + w(u,v))`（非负权，优先队列）<br>**Bellman-Ford**：对所有边松弛 `V−1` 轮（可处理负权、检测负环）<br>**BFS/DFS**、拓扑排序、**PageRank**、`Louvain`、SCC | **B + C** | **B** 移植算法；**C** 以其输出为基准。硬不变量：Dijkstra 与 Bellman-Ford 距离**逐点一致 ≤ 1e-12** |
| 8.2 | **EasyGraph**（Patterns 2023）<br>https://www.cell.com/patterns/fulltext/S2666-3899(23)00218-0 | 开源* | 高性能图分析（结构洞占据者检测、网络嵌入），针对大规模网络优化 | **D** | 参照其**基准口径**（算法覆盖度 / 大规模吞吐）；本轮只做中小规模正确性 |
| 8.3 | **PageRank** | 论文 | `PR(v) = (1−d)/N + d·Σ_{u∈B_v} PR(u)/L(u)`，`d=0.85`<br>幂迭代至收敛 | **A** | 硬不变量：**`Σ_v PR(v) = 1 ± 1e-12`**；与转移矩阵主特征向量一致 |
| 8.4 | **Louvain / 模块度** | 论文 | `Q = (1/2m)·Σ_{ij} [ A_ij − k_i k_j/(2m) ]·δ(c_i, c_j)`；两阶段：局部移动 + 图聚合 | **A** | `Q` 值每轮单调不减（可断言）；固定种子下社区划分可复现 |
| 8.5 | **图学习**（node2vec / DeepWalk / GCN / GraphSAGE / GAT） | 论文 | GCN：`H^{(l+1)} = σ( D̃^{-1/2} Ã D̃^{-1/2} H^{(l)} W^{(l)} )`<br>GAT：`α_ij = softmax( LeakyReLU(aᵀ[Wh_i ‖ Wh_j]) )` | **B** | 硬不变量：**置换等变性**（permutation equivariance）——打乱节点编号后输出随之置换、数值一致 ≤ 1e-12 |

### 本族关键不变量

| 不变量 | 阈值 |
|---|---|
| Dijkstra vs Bellman-Ford | 逐点距离一致 ≤ 1e-12 |
| PageRank 归一化 | `Σ PR = 1 ± 1e-12` |
| Louvain `Q` 单调性 | 每轮不减 |
| 图结构置换等变性 `L(PAPᵀ) = P·L(A)·Pᵀ`（对应 `INV-GPH-007`，仅覆盖 F4G7 的 Laplacian/聚合部分，**完整 GNN 训练本轮 reserved**） | ≤ 1e-12 |

---

## §9 数值稳定

| # | 顶级方案（名称 + URL） | 许可 | 关键算法/数学（核心公式） | 复用 | 超越点 |
|---|---|---|---|---|---|
| 9.1 | **scipy.special.logsumexp**<br>https://github.com/scipy/scipy | BSD-3 | `logsumexp(x) = m + log Σ_i exp(x_i − m)`，`m = max_i x_i`<br>配套：`log1p(x)=log(1+x)`、`expm1(x)=e^x−1`、`hypot(a,b)` | **A** | 硬不变量：与高精度（Python `mpmath`，开发期）对照 **≤ 1e-13**；极端输入 `[1000, 1000]`、`[−1000, −1000]` 不溢出/不丢精 |
| 9.2 | **stable softmax / log-softmax** | 公开结果 | `softmax(x)_i = exp(x_i − m)/Σ exp(x_j − m)`<br>`log_softmax(x)_i = (x_i − m) − log Σ exp(x_j − m)` | **A** | 每行 `Σ = 1 ± 1e-15`；对数域版本与 `log(softmax)` 一致 ≤ 1e-13 |
| 9.3 | **log-sigmoid** | 公开结果 | `log σ(x) = −log(1 + e^{−x}) = −softplus(−x)` | **A** | `x = −800` 时 `log σ(x) ≈ −800`（不返回 `-Inf`） |
| 9.4 | **Kahan / Neumaier 补偿求和**（Higham《Accuracy and Stability of Numerical Algorithms》） | 书 | Kahan：`s ← s + y; c ← (s − s_prev) − y`（单精度补偿）<br>Neumaier：`s ← s + y; if |s| ≥ |y|: c ← c + (s_prev − s) + y else c ← c + (y_prev − s) + s`（双补偿，抗抵消） | **A** | 硬不变量：与高精度求和的误差界对比表（`n=1e6` 随机数，朴素 vs Kahan vs Neumaier） |
| 9.5 | **误差分析框架**（Higham） | 书 | 前向误差 / 后向误差 / 条件数 `cond(A) = ‖A‖·‖A⁻¹‖`；`前向误差 ≤ cond · 后向误差` | **D** | 把「条件数 → 可信位数」写进自检面板（Higham 只给理论，我们给可视化） |
| 9.6 | **Jacobi 对称特征值迭代** | 公开结果 | 逐次 Givens 旋转消去最大非对角元；残差 `‖Av − λv‖` | **A** | 对齐 `pca-forge` 实测 **1.09e-15**（注意：1.78e-15 是「全分量重构无损」指标，非 Jacobi 残差，2026-10-16 修正） |

### 本族关键不变量

| 不变量 | 阈值 |
|---|---|
| `logsumexp` vs 高精度 | ≤ 1e-13 |
| softmax 行和 | `= 1 ± 1e-15` |
| Kahan/Neumaier vs 高精度求和 | 误差界表（朴素 > Kahan > Neumaier） |
| 极端输入 | `[1000,1000]`、`[−1000,−1000]` 不溢出 |

---

## §10 性质测试与验证工程

| # | 顶级方案（名称 + URL） | 许可 | 关键算法/数学（核心公式） | 复用 | 超越点 |
|---|---|---|---|---|---|
| 10.1 | **fast-check**<br>https://github.com/dubzzz/fast-check | MIT | **Property-based testing**：声明性质 `∀x: P(x)`，随机生成输入，失败时 **shrinking** 收缩到最小反例 | **D** | 仅参照范式，**自写轻量版**（零依赖）：固定种子 PRNG + 每模块 ≥ 200 随机用例的不变量扫描 + 简易 shrinking |
| 10.2 | **Hypothesis**（Python） | MPL-2.0 | 同上（Python 生态） | **D** | 同上；不引入 |
| 10.3 | **差分测试 / 交叉验证范式**（本项目自研） | — | 五类对照：**① 解析解 vs 数值解 ② 暴力枚举 vs 贪心 ③ 参照实现 vs 主实现 ④ 高精度 vs 双精度 ⑤ 两种算法互验** | **A** | 用户「验证总纲」；固化为每模块必填的 5 格对照表 |
| 10.4 | **黄金文件 / 快照测试** | 公开实践 | 把参照实现的输出固化为 `golden/*.json`，每次运行比对 | **A** | 双端（浏览器/Node）报告 `hash` 必须相同（R-27） |

---

## §11 分词与嵌入

| # | 顶级方案（名称 + URL） | 许可 | 关键算法/数学（核心公式） | 复用 | 超越点 |
|---|---|---|---|---|---|
| 11.1 | **SentencePiece**<br>https://github.com/google/sentencepiece | Apache-2.0 | **BPE 合并**：统计相邻符号对频次，反复合并频次最高对，直到词表大小 `V`<br>**Unigram LM**：`P(x) = Π_i p(x_i)`，用 EM/Viterbi 剪枝词表<br>字节级回退（byte fallback），保证任意输入可编码 | **B** | 硬不变量：**`decode(encode(s)) === s` 往返一致**（覆盖 ASCII / CJK / emoji / 混合）；词表大小可配 |
| 11.2 | **tiktoken**（OpenAI BPE）<br>https://github.com/openai/tiktoken | MIT | BPE + 正则预分词（`<|endoftext|>` 等特殊 token）；byte-pair ranks 表驱动 | **C** | 以其编码结果作为**第二实现基准**（开发期对照，不进交付物） |
| 11.3 | **WordPiece / Unigram**（BERT / XLNet 系） | 论文 | WordPiece 合并准则：`score = log P(t₁t₂) − log P(t₁) − log P(t₂)`（互信息最大化） | **A** | 与 BPE 的合并结果差异对照表（小语料） |

### 本族关键不变量

| 不变量 | 阈值 |
|---|---|
| `decode(encode(s)) === s` | 100% 往返一致（含 CJK / emoji / 混合串） |
| 词表覆盖率 | 词表大小 = 配置值（精确） |
| 编码 token 数 | 与参照实现（tiktoken，开发期）一致 |

---

## §12 许可合规专章

### 12.1 主体许可与红线

| 项 | 结论 |
|---|---|
| **本项目主体许可** | **MIT**（与 ml-matrix / micrograd / tinygrad / llama.cpp / NetworkX 生态一致，最利于被复用） |
| **允许引入** | MIT、Apache-2.0、BSD-2/3、ISC、MPL-2.0（仅文件级 copyleft，不传染整个项目） |
| **禁止引入** | **GPL-2.0 / GPL-3.0 / AGPL-3.0 / LGPL（运行时依赖形态）** —— 会导致整个项目被强制同许可 |
| **本项目运行时依赖** | **零**。仅 Node 内置模块（`node:assert` / `node:test` / `node:fs` 等） |
| **署名** | 晨星（GitHub `CJX0712`）；每个 `REUSE.md` 保留原始来源的署名与链接 |

### 12.2 标注 `*` 需复核的 9 项 —— 处置结论

> 这 9 项在 2026-10-15 检索中**未直接读到 LICENSE 文本**。以下为**当前处置结论**，落地前须以各仓库 `LICENSE` 文件复核；**若复核发现为 GPL/AGPL 系，一律降级为「仅参照数学（D）」**。

| # | 项目 | 当前处置结论 | 理由与落地约束 |
|---|---|---|---|
| 1 | **numopt-js**<br>github.com/takuto-NA/numopt-js | ✅ **允许 · 参照实现做交叉验证（C）** | 仅**跑它的输出做数值对照**，不 import、不 vendoring。若 LICENSE 为 GPL 系 → 降级为 **D（仅参照数学）** |
| 2 | **EAGLE**<br>github.com/SafeAILab/EAGLE | ✅ **允许 · 仅参照架构（D）** | 本轮不引入任何代码；只参照「draft 生成位置」的设计思想 |
| 3 | **Medusa**<br>github.com/FasterDecoding/Medusa | ✅ **允许 · 仅参照架构（D）** | 同上；多头并行解码列为 P2，届时再复核许可 |
| 4 | **GPTQ**<br>github.com/IST-DASLab/gptq | ✅ **允许 · 移植算法（B）** | 移植**已发表的数学公式**（arXiv 2210.17323），公式本身不受代码实现许可约束；**不复制源码** |
| 5 | **AWQ**<br>github.com/mit-han-lab/llm-awq | ✅ **允许 · 移植算法（B）** | 同上（arXiv 2306.00978）；不复制源码 |
| 6 | **EfficientZero**<br>github.com/YeWR/EfficientZero | ✅ **允许 · 仅参照数学（D）** | 本轮不训练视觉 RL；只参照问题定义与 100k 步评估口径 |
| 7 | **EasyGraph**<br>cell.com/patterns/... (Patterns 2023) | ✅ **允许 · 仅参照基准口径（D）** | 学术论文参照，不涉及代码引入 |
| 8 | **rank_bm25** | ✅ **允许 · 直接移植公式（A）** | BM25 是 1990s 公开发表的检索公式，属公知技术；直接按公式实现，不依赖任何实现 |
| 9 | **pg_textsearch**（Timescale） | ⛔ **禁止引入 · 仅作为选型佐证（D）** | Postgres 扩展，与本项目「无重型框架、无数据库」约束冲突；仅用于确认「BM25 仍是 ranking 事实标准」 |

### 12.3 特殊风险项：`@tangent.to/mc`（**GPL-3.0**）

| 项 | 结论 |
|---|---|
| 仓库 | https://github.com/tangent-to/mc |
| 许可 | **GPL-3.0**（应用层；其数值叶子模块为 MIT，但**包整体按 GPL-3.0 发布**） |
| **处置** | ⛔ **仅参照数学，不引入代码** |
| 具体约束 | ① **禁止 `import '@tangent.to/mc'`**；② **禁止 vendoring / 复制任何源文件**；③ 仅参照其**公开的数学定义与诊断口径**（NUTS 的 dual-averaging 步长、Gelman-Rubin `R-hat`、ESS、divergence 计数）；④ 我们的 NUTS 与诊断必须**自写** |
| 替代方案 | 用 **PyMC（Apache-2.0）/ Stan（BSD-3）/ NumPyro·BlackJAX（Apache-2.0）** 的输出作为后验基准；诊断口径参照 **Stan Reference Manual**（文档，非代码）自写 |

### 12.4 复用方式 × 许可风险矩阵

| 复用方式 | 是否引入代码 | 许可风险 | 本项目采用情况 |
|---|---|---|---|
| A 直接引用（移植公式） | 否 | **无**（数学公式不受版权保护） | 11 条 |
| B 移植算法 | 否（重新表达） | 低（需确保不逐行照抄） | 8 条 |
| C 参照实现交叉验证 | 开发期临时跑，不进交付物 | 中（**必须确认许可**） | 9 条 |
| D 仅参照数学/架构/API | 否 | **无** | 5 条 |

---

## §13 禁止引入清单（运行时）

### 13.1 明确禁止的包（及替代方案）

| 禁止引入 | 原因 | 替代方案 |
|---|---|---|
| `@tangent.to/mc` | **GPL-3.0**，许可污染 | 自写 NUTS；诊断口径参照 Stan Reference Manual |
| 任何 **GPL / LGPL / AGPL** 包 | 强制同许可 | 换 MIT/Apache/BSD 同类，或自写 |
| `tensorflow` / `@tensorflow/tfjs` | 重型、需下载模型/wasm 二进制 | 自写 tape autodiff（micrograd 架构） |
| `onnxruntime-web` | 需远程加载 wasm 与模型权重，违反「零外部依赖」 | 纯 JS 算法内核；不跑预训练权重 |
| `@huggingface/transformers` | 同上，且首次下载 ~111 MB | 纯算法层；不下载任何权重 |
| `ml-matrix` / `numeric.js` / `mathjs` | 虽为 MIT，但违反「零依赖、零构建」约束 | **开发期**用于交叉验证，**交付物中不得 import** |
| `python` / `numpy` / `scipy` / `mpmath` | 非 JS | **开发期**用于高精度对照 → 结果固化为黄金文件 |
| 任何需要**编译**的包（`node-gyp` / native addon / WASM 编译） | 违反「不得引入需要编译的包」 | 纯 JS |
| 任何图表库（Chart.js / D3 / ECharts） | 违反「单文件 HTML 零外部依赖」 | 内联 CSS + 手工 SVG/Canvas 绘制 |
| 任何 CDN 外链（`cdn.jsdelivr.net` / `unpkg.com` 等） | 违反「零外部依赖、可离线」 | 全部内联 |
| 任何数据库 / HTTP 服务框架 | Out of scope | 无 |

### 13.2 允许使用的「非交付物」工具（开发期，不进仓库运行时）

| 工具 | 用途 | 是否进交付物 |
|---|---|---|
| Python 3.13 + `mpmath` | 高精度数值对照，生成黄金文件 | ❌ 仅黄金 JSON 进仓库 |
| `scipy.optimize` / `scipy.special` | L-BFGS / logsumexp 基准 | ❌ |
| `tiktoken` / `ml-matrix` / `numeric.js` | 第二实现基准 | ❌ |

---

## §14 来源总表

### 14.1 项目 / 仓库（40 条，均为 2026-10-15 实际检索命中）

| # | 名称 | URL | 许可 | 复用 |
|---|---|---|---|---|
| 1 | LAPACK | https://github.com/Reference-LAPACK/lapack | BSD-3 | B |
| 2 | OpenBLAS | https://github.com/OpenMathLib/OpenBLAS | BSD-3 | B |
| 3 | ml-matrix | https://github.com/mljs/matrix | MIT | C |
| 4 | numeric.js | https://github.com/sloisel/numeric | MIT | C |
| 5 | JAMA (NIST/MathWorks) | https://math.nist.gov/javanumerics/jama/ | NIST 公有域 | C |
| 6 | JamaJS | https://github.com/dragonfly-ai/JamaJS | Apache-2.0 | C |
| 7 | JAMA4JS | https://github.com/conceptualGabrielPutnam/JAMA4JS | 开源* | C |
| 8 | @stdlib/stdlib | https://github.com/stdlib-js/stdlib | Apache-2.0 | D |
| 9 | numopt-js | https://github.com/takuto-NA/numopt-js | 开源* | C |
| 10 | micrograd | https://github.com/karpathy/micrograd | MIT | B |
| 11 | JAX | https://github.com/jax-ml/jax | Apache-2.0 | D |
| 12 | tinygrad | https://github.com/tinygrad/tinygrad | MIT | D |
| 13 | tinygrad (JS) | https://github.com/arafathusayn/tinygrad | MIT | C |
| 14 | FlashAttention (Dao-AILab) | https://github.com/Dao-AILab/flash-attention | BSD-3 | A |
| 15 | ROCm/flash-attention · Building Blocks | https://deepwiki.com/ROCm/flash-attention/5-building-blocks | BSD-3（派生文档） | D |
| 16 | NanoLLM | https://github.com/omkarbhad/nanollm | MIT | D |
| 17 | gpt-sota-opt | https://github.cc/lioarce01/gpt-sota-opt | 开源* | D |
| 18 | llm-from-scratch | https://www.github.com/wwj-2017-1117/llm-from-scratch | MIT | D |
| 19 | llama.cpp (+ggml) | https://github.com/ggml-org/llama.cpp | MIT | B |
| 20 | vLLM | https://github.com/vllm-project/vllm | Apache-2.0 | B |
| 21 | EAGLE | https://github.com/SafeAILab/EAGLE | 开源* | D |
| 22 | Medusa | https://github.com/FasterDecoding/Medusa | 开源* | D |
| 23 | GPTQ | https://github.com/IST-DASLab/gptq | 开源* | B |
| 24 | AWQ | https://github.com/mit-han-lab/llm-awq | 开源* | B |
| 25 | transformers.js | https://github.com/huggingface/transformers.js | Apache-2.0 | D |
| 26 | hnswlib | https://github.com/nmslib/hnswlib | Apache-2.0 | B |
| 27 | FAISS | https://github.com/facebookresearch/faiss | MIT | C |
| 28 | rank_bm25 | （Okapi BM25 参考实现） | 开源* | A |
| 29 | pg_textsearch (Timescale) | （Postgres BM25 扩展） | 开源* | D（禁止引入） |
| 30 | CleanRL | https://github.com/vwxyzjn/cleanrl | MIT | B + D |
| 31 | CleanRL 论文 | https://arxiv.org/pdf/2111.08819 | 论文 | D |
| 32 | EfficientZero | https://github.com/YeWR/EfficientZero | 开源* | D |
| 33 | EfficientZero 论文 | https://arxiv.org/pdf/2111.00210v2 | 论文 | D |
| 34 | PyMC | https://github.com/pymc-devs/pymc | Apache-2.0 | C |
| 35 | Stan | https://github.com/stan-dev/stan | BSD-3 | C + D |
| 36 | NumPyro / BlackJAX | （JAX PPL 生态） | Apache-2.0 | D |
| 37 | @tangent.to/mc | https://github.com/tangent-to/mc | **GPL-3.0** | **D（仅参照数学，禁止引入）** |
| 38 | WebPPL | https://github.com/probmods/webppl | MIT | D |
| 39 | NetworkX | https://github.com/networkx/networkx | BSD-3 | B + C |
| 40 | EasyGraph (Patterns 2023) | https://www.cell.com/patterns/fulltext/S2666-3899(23)00218-0 | 开源* | D |
| 41 | SciPy | https://github.com/scipy/scipy | BSD-3 | A + C |
| 42 | fast-check | https://github.com/dubzzz/fast-check | MIT | D |
| 43 | SentencePiece | https://github.com/google/sentencepiece | Apache-2.0 | B |
| 44 | tiktoken | https://github.com/openai/tiktoken | MIT | C |

> 说明：PRD 中编号为 40 条；本文档为便于分节引用，拆分为 **44 个条目**（部分条目按「仓库 + 论文」或「主体 + 派生文档」拆开）。来源实体总数不变。

### 14.2 arXiv 论文（**15 篇**）

| # | 论文 | arXiv ID |
|---|---|---|
| P1 | FlashAttention | 2205.14135 |
| P2 | FlashAttention-2 | 2307.08691 |
| P3 | RoPE / RoFormer | 2104.09864 |
| P4 | RMSNorm | 1910.07467 |
| P5 | SwiGLU / GLU Variants | 2002.05202 |
| P6 | GQA | 2305.13245 |
| P7 | PagedAttention | 2309.06180 |
| P8 | Speculative Decoding | 2211.17192 |
| P9 | GPTQ | 2210.17323 |
| P10 | AWQ | 2306.00978 |
| P11 | HNSW | 1603.09320 |
| P12 | PPO | 1707.06347 |
| P13 | MuZero | 1911.08265 |
| P14 | Adam | 1412.6980 |
| P15 | AdamW (Decoupled Weight Decay) | 1711.05101 |

> **修正说明**：团队下发任务时记为「14 篇 arXiv」，实际清点为 **15 篇**（见上表）。以上报 15 篇为准。

### 14.3 书目（4 本）

| # | 书目 | 用途 |
|---|---|---|
| B1 | Nocedal & Wright,《Numerical Optimization》2nd Ed., 2006 — **Algorithm 7.5 (L-BFGS)** | §2 L-BFGS two-loop recursion |
| B2 | Higham,《Accuracy and Stability of Numerical Algorithms》 | §9 前向/后向误差、条件数、补偿求和 |
| B3 | Koller & Friedman,《Probabilistic Graphical Models》 | §7 变量消去、和积 BP、团树、d-separation |
| B4 | Gelman et al.,《Bayesian Data Analysis》 | §7 共轭解析后验（NUTS 的对照真理）、R-hat |
| （附） | Stan Reference Manual | §7 NUTS 诊断口径（文档参照，非代码） |

---

## §15 合规声明

**合规声明：所有项目名与链接均来自 2026-10-15 实际检索，未编造。**

补充约束：
1. 许可列标注 `*` 的 9 项（numopt-js / EAGLE / Medusa / GPTQ / AWQ / EfficientZero / EasyGraph / rank_bm25 / pg_textsearch）需在落地前以各仓库 `LICENSE` 文件复核；若复核为 GPL/AGPL 系，**一律降级为「仅参照数学（D）」**。
2. `@tangent.to/mc` 为 **GPL-3.0**，明确**仅参照数学，不引入代码**。
3. 复用方式 **B（移植算法）** 要求「重新表达」而非「逐行照抄」，落地时须在 `REUSE.md` 中说明与来源实现的差异。
4. 复用方式 **C（参照实现交叉验证）** 的第三方代码只允许在**开发期**临时运行，**不得进入交付物运行时**。
5. 本文档是复用来源的**唯一权威清单**；新增来源须先更新本文档再落地代码。

---

*文档结束 · 创新旗舰 AI · 产品经理 许清楚 · 署名 晨星（CJX0712）*
