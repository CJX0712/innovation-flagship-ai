# 零重写审计基线（PRIOR ART）— innovation-flagship-ai

| 项 | 内容 |
| --- | --- |
| 文档版本 | v1.0 |
| 作者 | 高见远（Gao）— Architect |
| 发起人 / 署名 | 晨星（GitHub: `CJX0712`） |
| 作用 | 「零重写原则」的可审计基线：`CITATION.md` 的**内容蓝本**与 `scripts/audit_no_reinvention.py` 的**校验依据** |
| 相关 | `docs/ARCHITECTURE.md`（各层职责表）、`docs/TASKS.md`（各任务需在 `DESIGN.md` 写明借用来源） |

---

## 0. 本文件的地位与用法

### 0.1 三条铁律的地基

PRD §1.1 的三条铁律中，**第一条「零重写原则」**要求：任何子模块必须先回答「全球最优公开解是什么」，再实现。本文件即为**这一问一答的标准答案表**。

- 每行 = 一条「我们复用了什么 / 我们如何组合或超越」。
- `CITATION.md` 是给人类评审看的**排版版**，`docs/PRIOR_ART.md` 是给机器看的**结构版（含代码文件路径字段）**。二者必须同步。
- `scripts/audit_no_reinvention.py` 会：① 解析 `CITATION.md` 条目数；② 校验 L0–L9 十层每层条目 ≥ 1；③ 校验每条的「涉及代码」字段指向的文件**真实存在**；④ AST 扫描关键函数的 docstring 是否含 `Priors:` 段。**四项全绿才算 INV-42 通过。**

### 0.2 「代码文件」列的使用约定

- 列中的路径为**计划路径**（`ARCHITECTURE.md` §5 的文件清单），由工程师在实现后**回填确认**（若路径变更，必须同步更新本表）。
- 一个条目可对应多个文件；一个文件可支撑多个条目。
- 标注 **P0/P1/P2** 表示相关实现需求的优先级；P2 被裁撤时，对应条目标注 `[DEFERRED]` 但**不得从表中删除**（保持"每个机制都有出处"的完整性）。

### 0.3 「借鉴 vs 自创」判定标准

| 判定 | 说明 | 处置 |
| --- | --- | --- |
| ✅ 合格复用 | 有明确的公开论文/官方实现可对照，我们做了移植/简化/组合 | 记入本表 + CITATION |
| ✅ 合格的工程组合 | 单个技术点都有出处，**组合方式**是本项目的原创贡献 | 记入本表并在「超越/组合之处」列写明组合逻辑 |
| ⚠️ 必要的新实现 | 因环境限制（CPU/仅 numpy）必须替换原技术的实现手段，但**数学等价或有解析对照** | 记入本表，并在「超越/组合之处」写明「如何进行等价性验证」（对应某条 INV） |
| ❌ 自创核心算法（无出处、无对照） | 违反零重写原则 | **CI 拒绝合入**，需在 PR 说明并补出处 |

---

## 1. L0 — 算子 / 内核层

| # | 复用的顶级成果 | 出处（论文/实现 + 年份） | 方法要点 | 我们如何复用 | 我们超越 / 组合之处 | 涉及代码文件（回填） | 优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-1 | **FlashAttention** | Dao, Fu, Ermon, Rudra, Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, 2022（arXiv:2205.14135） | 在线 softmax（running max + running sum 重标定）、分块 I/O 感知、永不物化 N×N 分数矩阵 | 把 tile-level online softmax 移植到 numpy；在 tile 上调用 BLAS | 增加 **causal block-skip**（完全被遮蔽的上三角 tile 直接跳过，非仅对角 mask）；将同じ online softmax 同时复用于 prefill 与增量 decode（统一内核），并提供 `_ref` 朴素实现作为 INV-02 对照 | `ifai/l0_kernel/attention.py` | P0 |
| 0-2 | **Anatomy of High-Performance Matrix Multiplication（cache-blocking / tiling GEMM）** | Goto & van de Geijn, *Anatomy of High-Performance Matrix Multiplication*, ACM TOMS 2008；BLAS/GotoBLAS 传统 | 矩阵分块到各级 cache、packing、宏观/微观 kernel 分层 | 用 `TileSpec` 描述 tile 尺寸与循环序，上层做任务编排，tile 内走 BLAS | 提出 **双后端**：`blas`（快）与 `tiled`（归约顺序完全自控 → **bitwise 确定性**）。这是对原技术面向"可复现性"目标的重新取舍：牺牲少量速度换取 INV-40 的严格确定性 | `ifai/l0_kernel/tiling.py`, `gemm.py` | P0 |
| 0-3 | **RoFormer / RoPE** | Su, Lu, Malleda 等. *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021（arXiv:2104.09864）；NTK-aware scaling（bloc97, 2023；Reddit / 社区） | 用复旋转注入绝对位置，使内积仅依赖相对位置；θ_i = 10000^(-2i/d) | 复数乘法拆成实数 half-rotate，fused 到 QK 投影后 | 增加 **NTK-aware / 线性插值 scaling** 的统一开关；提供 `rope_ref`（解析 2D 旋转矩阵）构成 INV-03、INV-04（相对性平移不变）的对照 | `ifai/l0_kernel/rope.py` | P0 |
| 0-4 | **SwiGLU / GLU Variants** | Shazeer. *GLU Variants Improve Transformer*, 2020（arXiv:2002.05202）；PaLM 采用 SwiGLU FFN | FFN = W_down(SiLU(W_gate x) ⊙ (W_up x)) | 实现 single-pass 融合版本 | 明确以「**峰值临时内存 ≤ 分离实现 60%**」作为可量化验收（R-L0-06），而非仅声称"融合"；这是把口头优化变成不变量 | `ifai/l0_kernel/activation.py` | P0 |
| 0-5 | **RMSNorm** | Zhang & Sennrich. *Root Mean Square Layer Normalization*, 2019（arXiv:1910.07467）；LLaMA 采用 pre-RMSNorm | 只做方差归一化重缩放，去掉均值中心化，计算更省 | 实现含可调 eps 与 **fused residual add** | 增加"1e3 量级输入仍无溢出"（数值鲁棒）验收；并对——ge: ‖RMSNorm(x)‖₂=√d 的几何性质做断言（INV-06） | `ifai/l0_kernel/norm.py` | P0 |
| 0-6 | **LLM.int8() per-channel quantization** | Dettmers, Lewis, Belkada, Zettlemoyer. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*, 2022（arXiv:2208.07339） | 激活 per-token、权重 per-channel 的向量级量化，混合精度分解处理离群特征 | 采用其 per-channel / per-token 的粒度定义与界 Equations | 因无 GPU，我们把「离群特征分解」简化为：**per-channel 权重 + per-token 激活** 两档粒度并用 CALLER 的组合方式，并以「PPL 退化 ≤5%」作为行为等价门槛；同时在文档中**诚实披露 CPU 上 int8 未必更快**，把收益点从"加速"改为"显存/磁盘 4×" | `ifai/l0_kernel/quant_kernel.py`, `gemm_quant.py` | P0 |
| 0-7 | **低位打包（bit-packing）** | 通用实践；参考 GPTQ / llama.cpp ggml 量化实现（Frantar 2022；ggerganov 2023） | 把多个低比特整数打包进 int32/uint8 字 | 实现 `pack_int4/unpack_int4` | 要求 **逐位可逆**（unpack(pack(x)) == x）作为不变量，并纳入 INV-07 的边界论证 | `ifai/l0_kernel/packing.py` | P1 |

---

## 2. L1 — 张量 / 数值层

| # | 复用的顶级成果 | 出处 | 方法要点 | 我们如何复用 | 我们超越 / 组合之处 | 涉及代码文件 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1-1 | **NumPy 数组编程（strides / view / broadcasting）** | Harris et al. *Array programming with NumPy*, Nature 585, 2020（doi:10.1038/s41586-020-2649-2） | 多维数组的 stride 视图、广播语义、延迟求值 | `NDArray` 直接持有 `np.ndarray`，`__array__` 透传 | 把"零拷贝"做成**可断言契约**（`np.asarray(x).base is nda._data.base`），防止后期有人无意引入拷贝——这是把工程约定变成测试的典型 | `ifai/l1_tensor/ndarray.py` | P0 |
| 1-2 | **稳定 log-sum-exp / safe softmax** | 数值分析标准实践；Blanchard et al. *Accurately Computing the log-sum-exp and softmax functions*, 2019-2021（arXiv:1903.10631 等） | 减去最大值后再 exp，避免上溢；log Σexp(x) = m + log Σexp(x-m) | 实现套件包含 `-->` 恒等式对照 | 把「x ∈ [-1e4, 1e4] 无 inf/nan」与「与解析值 diff ≤1e-6」写成 INV-08/09，而非只是实现 | `ifai/l1_tensor/numerics.py` | P0 |
| 1-3 | **RTN / min-max & percentile calibration** | Krishnamoorthi. *Quantizing deep convolutional networks for efficient inference*, 2018；Vanhoucke et al. *Improving the speed of neural networks on CPUs*, 2011 | Round-to-nearest 均匀量化，min-max 或百分位标定 | 实现对称 / 非对称两种 zp 语义 | 以「**重建误差 ≤ scale/2**」这一理论界作为验收（INV-07），即声称的效果必须落在数学界内 | `ifai/l1_tensor/quantizer.py`, `ifai/l0_kernel/quant_kernel.py` | P0 |
| 1-4 | **GPTQ / OBQ（Optimal Brain Quantization）** | Frantar, Ashkboos, Hoefler, Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, 2022（arXiv:2210.17323）；OBQ 源自 Frantar & Alistarh 2022 / Hassibi 1993 OBS | 利用 Hessian 信息逐列做最优量化，误差补偿到未量化列 | 实现校准集驱动的 Hessian（Gram）估计 + 逐列更新 | 对 PCORDER 与迭代节crypto Om: 实现**简化版 OBQ**（不含 Cholesky 稳定化的完整工程），并在文档中标注"这是原始方法约束下的简化，不是超越"——**诚实区分组合与简化** | `ifai/l1_tensor/gptq.py` | P1 |
| 1-5 | **信息论基础：熵 / KL / 温度 / top-k 截断重归一化** | Cover & Thomas. *Elements of Information Theory*, 2006；Holtzman et al. *The Curious Case of Neural Text Degeneration*（nucleus sampling）, 2020 | Shannon 熵、KL 散度定义与性质；截断后重新归一 | 实现 `entropy / kl / temperature_scale / top_k_truncate` | 以 **KL(p‖p)=0 ±1e-9** 与 **KL 非负恒成立**作为性质型不变量（R-L1-06），把信息论性质变成单测 | `ifai/l1_tensor/dist.py` | P1 |
| 1-6 | **einops 风格变换** | Rogozhnikov. *Einops: Clear and Reliable Tensor Manipulations with Einstein-like Notation*, 2022（arXiv:2202.07665） | 用模式串表达 reshape/transpose/reduce，避免索引魔法数 | 实现 `rearrange` 的**子集**（支持合并/拆分轴） | 仅为子集（不实现 reduce/repeat 全语法），并在 docstring 明确边界；目的是减少魔改 reshape 散布各处带来的形状 bug | `ifai/l1_tensor/ops.py` | P0 |

---

## 3. L2 — 模型架构层

| # | 复用的顶级成果 | 出处 | 方法要点 | 我们如何复用 | 我们超越 / 组合之处 | 涉及代码文件 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2-1 | **Transformer** | Vaswani et al. *Attention Is All You Need*, 2017（arXiv:1706.03762） | 多头注意力 + FFN + 残差 + 归一化的 Decoder 骨架 | 完整采用 Decoder-only 结构 | **Pre-LN 变体**（非原论文的 Post-LN）： | `ifai/l2_model/transformer.py`, `block.py` | P0 |
| 2-2 | **Pre-LN vs Post-LN 的收敛性研究** | Xiong et al. *On Layer Normalization in the Transformer Architecture*, 2020（arXiv:2002.04745）；GPT-2 采用 Pre-LN（Radford et al. 2019） | Pre-LN 使残差分支初始方差不受 LN 影响，warmup 可省、训练更稳 | 采用 pre-RMSNorm（LLaMA 风格，结合 0-5） | 与 RMSNorm 组合，形成「pre-RMSNorm + RoPE + GQA + SwiGLU」的**标准旗舰配方**，并把每一项的出处都单独列条 | `ifai/l2_model/block.py`, `norm.py` | P0 |
| 2-3 | **GQA（Grouped-Query Attention）** | Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lehmkühl, Sanghai. *GQA: Training Generalized Multi-Query Transformer Models from Multi-head Checkpoints*, 2023（arXiv:2305.13245） | K/V 头按组共享，MHA 与 MQA 的插值 | 实现 `n_kv_heads` 整除关系与 `repeat_kv` | 以「`n_kv == n_q` 时 ≡ MHA（≤1e-6）」（INV-10）作为**退化一致性**验收，确保通用化分支不破坏特例 | `ifai/l2_model/attention.py` | P0 |
| 2-4 | **Switch Transformer / MoE + load-balancing loss** | Fedus, Zoph, Shazeer. *Switch Transformers*, 2021/2022（arXiv:2101.03961）；Shazeer et al. *Outrageously Large Neural Networks*, 2017 | top-k 稀疏路由 + 辅助损失使专家负载均衡 | 实现 top-2 专家与标准 aux loss | 以「Σ router softmax = 1」「token 分配守恒」「随机数据上负载 CV ≤0.3」三条**结构性守恒律**（INV-14）作为验收，比"loss 下降"更早发现问题 | `ifai/l2_model/moe.py` | P1 |
| 2-5 | **权重绑定（Tied Embeddings）** | Press & Wolf. *Using the Output Embedding to Improve Language Models*, 2017（arXiv:1608.05859）；Inan et al. 2017 | 输入词嵌入与输出 softmax 权重共享 | `lm_head.weight is embed_tokens.weight` | 以「`is` 判定」和「参数量精确减少 \|V\|×d」两条同时验收（INV-12/13），防止"看起来共享实则复制"的实现退化 | `ifai/l2_model/transformer.py`, `weights.py` | P0 |
| 2-6 | **BPE / Byte-level BPE** | Sennrich, Haddow, Birch. *Neural Machine Translation of Rare Words with Subword Units*, 2016（arXiv:1508.07909）；GPT-2 bytes→unicode 映射（Radford et al. 2019） | 频率最高的相邻 pair 合并；字节级 fallback 保证 OOV 可表示 | 实现 pair 统计训练 + 字节级可逆映射 + 编码解码 | 增加 **round-trip 无损测试**与 `vocab_size` 与 `ModelConfig` 的一致性断言，避免 tokenizer 与模型不同步这一常见集成 bug | `ifai/l2_model/tokenizer/*` | P0 |
| 2-7 | **上下文长度外推：NTK-aware / YaRN** | bloc97. NTK-Aware Scaled RoPE（2023, 社区）；Peng et al. *YaRN: Efficient Context Window Extension*, 2023（arXiv:2309.00071） | 调整 RoPE 基底或分段插值以延长有效上下文 | 实现 NTK-aware 与线性插值两种缩放 | 以「外推 2× 长度后 PPL 上升 ≤10%」量化（R-L2-07, P2）；P2 若延期，接口保留、实现留 TODO 并标注 `[DEFERRED]` | `ifai/l2_model/rope_scale.py`, `ifai/l0_kernel/rope.py` | P2 |
| 2-8 | **safetensors 风格权重序列化 + manifest** | HuggingFace. *safetensors*, 2022（Apache-2.0）；一般的对象存储 manifest 实践 | 张量扁平存储 + 头部元数据描述 shape/dtype/offset | 采用 npz + json manifest（含 sha256 全局校验） | 把"save→load round-trip 误差 = 0"做成硬验收（R-L2-06），因为 99% 的仓库都不保证这点 | `ifai/l2_model/weights.py` | P0 |

---

## 4. L3 — 推理引擎层

| # | 复用的顶级成果 | 出处 | 方法要点 | 我们如何复用 | 我们超越 / 组合之处 | 涉及代码文件 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3-1 | **PagedAttention / vLLM** | Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica. *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023（arXiv:2309.06180） | 把 KV Cache 切成 block，用块表做逻辑到物理映射，按需分配 | 实现 `(n_blocks, block_size, n_kv_heads, head_dim)` 预分配池 + 块表 + refcount | 以「**分页 ≡ 稠密**」（INV-15）作为首要验收，确保内存优化不以正确性为代价；并把在线 softmax 注意力（0-1）与分页读取**组合**在同一条 decode 路径上，这是"复试_llFlashAttention × PagedAttention"的跨层组合 | `ifai/l3_engine/kv_cache.py`, `block_manager.py` | P0 |
| 3-2 | **Continuous Batching / Orca** | Yu, Jeong, Lee, Choi, Kim. *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI 2022 | iteration-level scheduling，每完成一个序列立刻空出槽位 | 实现 admit/step/preempt 的迭代级调度 | 把「**greedy 下 batch ≡ 串行逐 token 100% 一致**」（INV-16）作为核心不变量。多数开源实现对这条只做近似说明，我们把它做成硬门禁，这是本仓库最值得称道的工程质量决策 | `ifai/l3_engine/scheduler.py`, `sequence.py` | P0 |
| 3-3 | **Speculative Decoding** | Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative Decoding*, 2023（arXiv:2211.17192）；Chen et al. 2023 | 小模型草稿 + 大模型并行验证，拒绝采样保证分布不变 | 实现自草稿（用小 shard 的同模型）+ **树状候选** | ① 采用 Medusa/EAGLE 风格的**树状候选**（Cai et al. *Medusa*, 2024；Li et al. *EAGLE*, 2024）而非单链；② 以「**greedy 下逐 token 100% 一致**」（INV-17）作为验收而非仅仅"看起来快"。这两条组合是本项目在 L3 的主要原创贡献点（工程组合原创） | `ifai/l3_engine/spec_decode.py` | P1 |
| 3-4 | **min-p sampling** | Nguyen, Dudy, Huber, Gehrmann. *Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs*, 2024（arXiv:2407.01082） | 动态阈值 p ≥ min_p × p_max，替代固定 top-p | 实现并与 top-p/top-k 共存 | 把采样器做成**统一截断管线**（penalty → top-k → top-p → min-p → renorm），并以「Σp=1 ±1e-6」与「T→0 / top_k=1 ≡ greedy」双不变量（INV-18/19）验收 | `ifai/l3_engine/sampler.py` | P0 |
| 3-5 | **Chunked Prefill** | Agrawal et al. *Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills*, 2023（arXiv:2308.16369） | 长 prompt 分块做 prefill，降低峰值占用并与 decode 混合 | chunk=512 统一入口 | 以「分块 ≡ 整段」（INV-20）验收；并与 3-1 的分页写入组合 | `ifai/l3_engine/prefill.py` | P1 |
| 3-6 | **Prefix Caching（radix / hash 简化）** | Zheng et al. *SGLang: Efficient Execution of Structured Language Model Programs*, 2024（arXiv:2312.07104）的 RadixAttention；vLLM automatic prefix caching | 共享相同前缀的 KV block，跨请求复用 | 用块内容哈希 + LRU 的简化实现 | 以「命中/未命中路径输出 ≤1e-5」验收（防止缓存污染静默改变结果） | `ifai/l3_engine/prefix_cache.py` | P2 |

---

## 5. L4 — 训练与微调层

| # | 复用的顶级成果 | 出处 | 方法要点 | 我们如何复用 | 我们超越 / 组合之处 | 涉及代码文件 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4-1 | **Adam / AdamW** | Kingma & Ba. *Adam*, 2015（arXiv:1412.6980）；Loshchilov & Hutter. *Decoupled Weight Decay Regularization*, 2019（arXiv:1711.05101） | 一阶/二阶动量自适应；权重衰减与自适应项解耦 | 实现标准 AdamW | 以「单步更新 ≡ 解析公式 ≤1e-6」且「wd=0 ≡ Adam」（INV-21）双口径验收；后者能抓住"误把 wd 加进 grad"的常见实现 bug | `ifai/l4_train/optim.py` | P0 |
| 4-2 | **SGDR 余弦退火 + warmup** | Loshchilov & Hutter. *SGDR*, 2017（arXiv:1608.03983）；Goyal et al. *Accurate, Large Minibatch SGD*, 2017 | 余弦形学习率衰减；训练初期线性 warmup | 实现 `cosine_with_warmup` | 以「lr 曲线与解析值 ≤1e-9」且「warmup 结束恰为 peak」验收（R-L4-02） | `ifai/l4_train/schedule.py` | P0 |
| 4-3 | **Gradient Clipping（global L2 norm）** | Pascanu, Mikolov, Bengio. *On the difficulty of training recurrent neural networks*, 2013（arXiv:1211.5063） | 全局梯度范数超阈值时按比例缩放 | 实现全局 L2 clip | 除「裁剪后 ‖g‖ ≤ max_norm」外，额外要求「**未超限时梯度不被修改（bitwise 相同）**」（INV-23）——这条能抓住"所有情况都做一次归一化"的错误实现 | `ifai/l4_train/grad_utils.py` | P0 |
| 4-4 | **Reverse-mode Automatic Differentiation** | Rumelhart, Hinton, Williams. *Learning representations by back-propagating errors*, 1986；Baydin et al. *Automatic Differentiation in Machine Learning*, 2018（arXiv:1502.05767） | 反向模式链式法则、计算图/磁带 | 实现极简 tape（`TensorGrad` + `grad_fn` 闭包 + 拓扑逆序） | 把「**每个算子的梯度都必须与有限差分对照**」（INV-22，≥20 组随机样本，相对误差 ≤1e-3）作为**唯一的可信来源**。这是本项目为"教学 + 可信"付出的关键冗余 | `ifai/l4_train/autodiff/*` | P0 |
| 4-5 | **Gradient Checkpointing（重计算）** | Chen, Xu, Guestrin, Li, Lin. *Training Deep Nets with Sublinear Memory Cost*, 2016（arXiv:1604.06174） | 丢弃中间激活，反向时重算，以时间换空间 | 实现按层重计算 | 以「loss 与 grad ≡ 无检查点 ≤1e-5」验收（INV-25），忘记校验 grad 只看 loss 是常见疏漏 | `ifai/l4_train/checkpoint.py` | P1 |
| 4-6 | **LoRA** | Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Chen. *LoRA: Low-Rank Adaptation*, 2022（arXiv:2106.09685） | 冻结主干，插入低秩 BA 增量，`y += (α/r)·B·A·x` | 实现 `LoRALinear` 与 merge/unmerge | 以「merge ≡ 未 merge ≤1e-6」且「B=0 ≡ 冻结基线」（INV-24）双口径验收；前者抓数值，后者抓分支污染 | `ifai/l4_train/lora.py` | P0 |
| 4-7 | **nanoGPT 风格极简训练循环** | Karpathy. *nanoGPT*, 2022（MIT License） | 极简、可读的训练循环与数据管线 | 借鉴其结构组织思想（分离 dataset/trainer/config）而非代码 | 加入 **可复现 shuffle**（走全局 RNG）+ **loss 单调性断言**（INV-26：尾 20% ≤ 前 20%×0.8），把"训练看起来在降"变成可 CI 断言的量 | `ifai/l4_train/trainer.py`, `dataset.py`, `loss.py` | P0 |

---

## 6. L5 — 检索与记忆层

| # | 复用的顶级成果 | 出处 | 方法要点 | 我们如何复用 | 我们超越 / 组合之处 | 涉及代码文件 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5-1 | **Okapi BM25** | Robertson, Walker, Jones, Hancock-Beaulieu, Gatford. *Okapi at TREC-3*, 1994；Manning/Raghavan/Schütze *IIR* 教科书 | 词频饱和 + 文档长度归一化，`k1=1.5, b=0.75` | 实现倒排 + df/idf + BM25 打分 | 提供 `bm25_ref` 独立参考实现，用「≥1000 文档 × ≥50 查询 Top-10 重合 ≥95%」（INV-27）验收——多数仓库不写参考实现所以无法证真 | `ifai/l5_retrieval/bm25.py` | P0 |
| 5-2 | **HNSW** | Malkov & Yashunin. *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*, 2018（arXiv:1603.09320） | 分层可导航小世界图、进入点+层渐渐ာ基建、启发式邻居选择 | 实现 M=16/efC=200/efS=64 标准参数 | ① **随机层分配走全局 RNG**（可复现索引构建）；② 以「recall@10 ≥0.95 vs 暴力」与「QPS ≥5×」双指标验收（INV-28） | `ifai/l5_retrieval/hnsw.py` | P0 |
| 5-3 | **Flat / Brute-force TopK（作为 ANNS 真值）** | 向量检索标准基线（FAISS 的 IndexFlat，Johnson, Douze, Jégou 2019） | 全量扫描内积/余弦 | 实现 `FlatIndex` | 明确把暴力实现列为**一等公民**而非只用 ANNS：所有近似结构必须有暴力对照，这是"可验证原则"在检索层的落地 | `ifai/l5_retrieval/vector_index.py` | P0 |
| 5-4 | **RRF（Reciprocal Rank Fusion）** | Cormack, Clarke, Buettcher. *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*, SIGIR 2009 | `score = Σ 1/(k + rank_i)`，无需分数归一化 | 实现 k=60 的 RRF | ① 以「与公式逐项一致 ≤1e-12」（INV-30）验收；② **与 BM25 + HNSW 组合成混合检索管线**，这是本层的核心工程贡献 | `ifai/l5_retrieval/fusion.py` | P0 |
| 5-5 | **Cross-Encoder / MonoBERT 重排** | Nogueira & Cho. *Passage Re-ranking with BERT*, 2019（arXiv:1901.04085） | query 与 doc 拼接后交叉打分，精度高于双塔 | 实现微型交叉打分器 | 组合成「**多路召回 → RRF 融合 → Cross-Encoder 重排**」的完整级联；以「NDCG@10 不低于融合前」作为相对判定（避免用微型模型做绝对质量声明） | `ifai/l5_retrieval/rerank.py`, `pipeline.py` | P1 |
| 5-6 | **DPR 双塔稠密检索** | Karpukhin, Oğuz, Min, Lewis, Wu, Edunov, Chen, Yih. *Dense Passage Retrieval for Open-Domain QA*, 2020（arXiv:2004.04906） | query tower / doc tower 分离编码，内积相似度 | 实现 d≤128 微型双塔，复用 L2 Transformer | 把 vector dim 限制在 ≤128 以适配 CPU；采用**相邻 chunk 为正例**的自监督训练信号，避免外部标注依赖 | `ifai/l5_retrieval/encoder.py` | P0 |
| 5-7 | **MemGPT 分层记忆** | Packer, Fang, Wooders, Gonzalez, Stoica. *MemGPT: Towards LLMs as Operating Systems*, 2023（arXiv:2310.08560） | 主上下文 / 外部存储的显式分层与换入换出 | 实现情景记忆 / 语义记忆二分 + 容量上限与换出策略 | 增加 **语义记忆去重归并**（相似事实聚类合并），并以「注入 50 条事实后召回 ≥90%」（R-L5-06）量化 | `ifai/l5_retrieval/memory/*` | P1 |
| 5-8 | **Recursive / overlapping chunking** | 通用 RAG 工程实践（LangChain / LlamaIndex 的 text splitters） | 按分隔符递归切分 + 重叠窗口避免跨界语义割裂 | 实现三种策略 | 以「**chunk 拼接 == 原文（归一化后 100% 覆盖）**」（INV-31）验收——这是最容易被忽略却最基础的检索数据正确性保证 | `ifai/l5_retrieval/chunk.py` | P1 |
| 5-9 | **索引持久化与增量索引** | 倒排/ANN 索引的工程标准（Lucene 分段思想；FAISS IVF 增量） | 分段落盘 + 增量追加 | 实现 npz + json meta + sha256 校验 | 以「reload 前后检索结果 **100% 一致**」（INV-32）验收，防止"索引增删 Lance a gastrointestinal hartelijkheid chuẩn lung" 这类隐性不一致 | `ifai/l5_retrieval/persist.py` | P1 |

---

## 7. L6 — 智能体层

| # | 复用的顶级成果 | 出处 | 方法要点 | 我们如何复用 | 我们超越 / 组合之处 | 涉及代码文件 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6-1 | **ReAct** | Yao, Zhao, Yu, Du, Narasimhan, Cao. *ReAct: Synergizing Reasoning and Acting in Language Models*, 2023（arXiv:2210.03629） | Thought / Action / Observation 交替，把推理与外部工具调用交织 | 实现主循环 + 提示模板 + 解析器 | 增加 **格式纠错重试**（解析失败时注入纠错提示，≤2 次恢复）与 **超步优雅终止**（强制给出结论），把论文里的"max step"变成可用的产品行为 | `ifai/l6_agent/react.py`, `prompt.py`, `parser.py` | P0 |
| 6-2 | **Reflexion** | Shinn, Cassano, Gopinath, Narasimhan, Yao. *Reflexion: Language Agents with Verbal Reinforcement Learning*, 2023（arXiv:2303.11366） | 失败后生成语言反思并写入记忆，再试一次 | 实现 ≤3 轮反思包装 | 把反思文本**写入 L5 情景记忆**（而非丢弃），形成「反思 → 记忆 → 下次调用」的闭环，这是跨层组合 | `ifai/l6_agent/reflexion.py` | P1 |
| 6-3 | **Toolformer / Function Calling Schema** | Schick, Dwivedi-Yu, Dessì 等. *Toolformer*, 2023（arXiv:2302.04761）；OpenAI Function Calling API Schema | 模型自主决定何时调用工具；结构化工具描述 | 实现 `ToolSpec` + JSON Schema 声明 + `to_openai_tools()` | 因不可引入 jsonschema 依赖，**自写 Draft-07 子集校验器**（`tools/schema.py`），并以「所有内置工具通过校验 100%」（INV-33）验收。这份校验器同时被 L7 复用验证 OpenAI 请求体——**一处实现、两层受益** | `ifai/l6_agent/tools/base.py`, `tools/schema.py` | P0 |
| 6-4 | **Plan-and-Solve / Plan-and-Execute** | Wang, Xu, Lan, Hu, Lan, Lee, Lim. *Plan-and-Solve Prompting*, 2023（arXiv:2305.04091）；LLM+P（Liu et al. 2023） | 先生成计划再逐步执行 | 实现 Planner/Executor | 要求生成的计划是**无环 DAG**（显式环检测 + 拓扑排序），并以「可完整执行」验收（R-L6-04），避免"伪计划" | `ifai/l6_agent/planner.py`, `executor.py` | P1 |
| 6-5 | **AutoGen / CAMEL 多智能体编排** | Wu, Bansal, Zhang, Wang 等. *AutoGen*, 2023（arXiv:2308.08155）；Li, Hammoud 等. *CAMEL*, 2023（arXiv:2303.17760） | 角色扮演 + 对话式协作 + 消息路由 | 实现角色 + 黑板 + 消息总线 | 以「消息总线无丢包」与「2 智能体协作完成内置任务」验收（R-L6-06, P2） | `ifai/l6_agent/orchestrator.py` | P2 |
| 6-6 | **AST 白名单沙箱 / 受限执行** | Python `ast` 官方能力与社区 Sandbox 实践；参考 RestrictedPython、pysandbox 等（Zope Community, MIT/BSD） | 静态语法树节点白名单，禁止危险节点 | 实现 AST 白名单 + subprocess + 超时 + 输出截断 + cwd 隔离 | 增加 **30 条越权/逃逸内置用例集**（`assets/data/sandbox_escape_30.jsonl`）并要求 100% 拦截（INV-34）；同时**在 README 明示其为中等强度，非安全边界**——这是诚实工程表达 | `ifai/l6_agent/tools/sandbox.py` | P0 |
| 6-7 | **轨迹记录与可回放** | W&B / MLflow 的实验追踪思想；确定性重放 Debugging 实践 | step 级结构化日志 + 随机种子 + 可重放 | 实现 `TraceStep` JSON + 同 seed 回放比对 | 以「**同 seed 回放 100% 一致**」（INV-35）验收；这把 debug 便利性变成了可断言的产品能力 | `ifai/l6_agent/trace.py` | P1 |

---

## 8. L7 — 服务与编排层

| # | 复用的顶级成果 | 出处 | 方法要点 | 我们如何复用 | 我们超越 / 组合之处 | 涉及代码文件 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7-1 | **OpenAI REST API 规范** | OpenAI. *API Reference / Chat Completions*, 2020 起持续更新；`openai-python` 的 pydantic schema（Apache-2.0） | `/v1/chat/completions` 等端点与响应字段语义 | 实现四个端点 + 错误体格式 | 建立**字段快照常量表**（`OPENAI_FIELDS`）并以「字段集 diff 为空」（INV-36）验收；不联网也能回归校验。这是对"不断变化的外部 API"最稳妥的工程处理 | `ifai/l7_server/schema.py`, `errors.py` | P0 |
| 7-2 | **SSE（Server-Sent Events）** | W3C. *HTML Living Standard — Server-sent events*；WHATWG | `text/event-stream`、`data:` 帧、`\n\n` 分隔 | 手写 SSE 编码 + keep-alive 心跳 + `[DONE]` | 以「**Σ chunks 的 delta.content == 非流式 content（字符串完全相等）**」（INV-37）验收。绝大多数实现不做这个断言，而这正是流式最常炸的地方 | `ifai/l7_server/sse.py` | P0 |
| 7-3 | **OWASP Top 10 for LLM Applications** | OWASP Foundation. *Top 10 Risks for LLM Applications*, 2023/2025 | Prompt Injection、不安全输出处理、DoS、敏感信息泄露等分类 | 把分类映射为本地规则集（注入特征、长度/成本限额、输出过滤） | 以「**红队拦截率 ≥90% 且正常误杀率 ≤5%**」双指标验收（INV-38）——单看拦截率会导致"全都拒答"的退化通过；这与 ML 界的 precision/recall 权衡一脉相承 | `ifai/l7_server/guardrail.py` | P0 |
| 7-4 | **NVIDIA NeMo Guardrails 思想** | Rebedea, Dinu, Sreedhar, Parisien, Cohen. *NeMo Guardrails*, 2023（arXiv:2310.10501） | 可编程护栏、对话 rail 与输入输出轨 | 借鉴其「输入轨 / 输出轨」分层结构与 verdict 语义 | verdict 结构化（`rule_id` / `reason` / `score`），使每一次拦截**可被审计与复现** | `ifai/l7_server/guardrail.py` | P0 |
| 7-5 | **有界并发与背压（Bounded queue / admission control）** | 排队论与服务端工程标准（Little's Law；Little 1961） | 队列长度上限、超时拒绝、信号量控制并发 | 实现 `BoundedQueue`（max_concurrency + 超时 429） | 以「1000 并发请求丢失 0」验收（R-L7-03），并让 `trace_id` 贯穿队列到 metrics | `ifai/l7_server/queue.py` | P1 |
| 7-6 | **Prometheus Exposition Format** | Prometheus Authors. *Exposition Formats*, 2012 起（Apache-2.0） | 文本格式的 counter/gauge/histogram 表达法 | 实现最小文本渲染器 | 指标与本地计时**偏差 ≤10%**（R-L7-05）被显式验收，避免"自难自栗子"的指标 | `ifai/l7_server/metrics.py` | P1 |
| 7-7 | **优雅启停 / readiness probe** | Kubernetes. *Pod Lifecycle & Probes*（CNCF, Apache-2.0）；SIGTERM 处理的生产实践 | 存量请求处理完毕再退出；`/healthz` 就绪探针 | 实现 `/healthz` + SIGTERM 优雅退出 | P2（Windows 信号语义差异，文档中明示限制） | `ifai/l7_server/server.py`, `app.py` | P2 |

---

## 9. L8 — 评测与对齐层

| # | 复用的顶级成果 | 出处 | 方法要点 | 我们如何复用 | 我们超越 / 组合之处 | 涉及代码文件 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8-1 | **Perplexity 标准定义** | Jelinek et al. *Perplexity — A Measure of the Difficulty of Speech Recognition Tasks*, 1977；神经语言模型标准实践 | PPL = exp(平均负对数似然) | 实现滑动窗口似然 | 以「**与手工计算 exp(mean NLL) diff ≤1e-6**」（INV-39）验收 | `ifai/l8_eval/perplexity.py` | P0 |
| 8-2 | **GSM8K 数据集** | Cobbe, Kosaraju, Bavarian, Chen 等. *Training Verifiers to Solve Math Word Problems*, 2021（arXiv:2110.14168） | 8.5k 小学应用题 + 计算器标注 | 提供下载脚本；内置 **20 题自造 mini 子集**供离线 CI | 遵循"**管线正确性而非绝对准确率**"为门槛（微型模型不承诺解题率），避免用不诚实的小数字装点门面 | `ifai/l8_eval/gsm8k.py`, `scripts/download_gsm8k.py` | P1 |
| 8-3 | **Chain-of-Thought Prompting** | Wei, Wang, Schuurmans 等. *Chain-of-Thought Prompting*, 2022（arXiv:2201.11903） | few-shot 中间推理步骤提示 | 评测使用 CoT few-shot 提示模板 | 与自写**答案抽取器**（100% 准确率单测）配合，使"能解题"与"能解析出答案"解耦验证 | `ifai/l8_eval/gsm8k.py`, `answer_extract.py` | P1 |
| 8-4 | **HELM 多维评测** | Liang, Bommasani, Lee 等. *Holistic Evaluation of Language Models*, 2022（arXiv:2211.09110） | 多场景 × 多指标的整体评测矩阵，而非单一分数 | 借鉴其"多维"思想，组织 ppl / bench / math / fact / safety 五类评测 | 输出**结构化 JSON 报告 + Markdown 渲染**（`report.py`），并把指标与 `BASELINE.md` 绑定做 CI 门禁 | `ifai/l8_eval/report.py`, `taskset.py` | P1 |
| 8-5 | **统计回归门禁（变异系数 CV）** | 通用统计质量控制；CV = std/mean 的工业基准实践 | 多次重复取均值与变异系数，控制噪声 | bench 重复 3 次，CV ≤15% | 把「**绝对指标不可跨机器比较**」的现实显式写入架构：CI 用 `BASELINE.md` 实测基线做 **±15% 相对回归保护**，发布才用绝对指标。这一工程决策避免了"穷人剽窃 shoving the CI red"，属于本项目的成熟之处 | `ifai/l8_eval/baseline.py`, `bench.py` | P0 |
| 8-6 | **确定性 / 可复现性验证** | 科学计算的可复现性实践；`numpy.random.Generator` 显式流设计 | 同 seed 同比特结果 | 全局统一 RNG + `--seed` | 以「**同 seed 两次运行 logits bitwise 一致**」（INV-40）验收；并提供 `deterministic` 双后端（见 0-2）作为达成手段 | `ifai/common/rng.py`, `ifai/l8_eval/determinism.py` | P0 |

---

## 10. L9 — 应用与接口层 & X 横切

| # | 复用的顶级成果 | 出处 | 方法要点 | 我们如何复用 | 我们超越 / 组合之处 | 涉及代码文件 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 9-1 | **llama.cpp 风格开箱即用体验** | Gerganov, Gerganov. *llama.cpp*, 2023（MIT） | 一条命令跑通、零外部依赖、CPU 优先 | `pip install -e . && ifai demo` ≤120s | 增加 **`ifai demo` 打印本次调用经过的 L0→L9 链路图**（`link_view.py`），把"黑盒生成"变成"可见的教学链路"——这是面向 U1（学习者）的差异化设计 | `ifai/l9_app/commands/demo.py`, `link_view.py` | P0 |
| 9-2 | **argparse CLI 约定与 `selftest` 概念** | Python 标准库 argparse；自证系统的概念源起（SQLite 的 `PRAGMA integrity_check`；Postfix `postfix check`） | 子命令 + `--help` + 自诊断 | 十个子命令 + `--seed/--verbose`；`ifai selftest [--layer] [--inv]` | 把 42 条不变量做成 `--inv INV-17` **可单独执行单元**，使"可验证原则"从口号变成一条可执行命令 | `ifai/l9_app/cli.py`, `commands/selftest.py` | P0 |
| 9-3 | **高阶 SDK API 形态（OpenAI SDK / LangChain 风格）** | OpenAI. *openai-python*（Apache-2.0）；Chase. *LangChain*, 2022（MIT） | 面向任务的精简 facade：`client.chat` / `retriever.get_relevant_documents` | 提供 `LLM.chat` / `Retriever.search` / `Agent.run` 三件套 | 三个 API 背后分别映射到「L3+L2」「L5」「L6+L5」，并在 `examples/quickstart.py` 一键连贯演示，形成最小学习路径 | `ifai/l9_app/sdk.py`, `examples/quickstart.py` | P0 |
| X-1 | **单向分层依赖（Clean Architecture / Hexagonal）** | Martin. *Clean Architecture*, 2017；Cockburn. *Hexagonal Architecture*, 2005 | 依赖向内收敛、端口与适配器 | 十层单向 + `common/protocols.py` 定义 `LLMPort` 等 Protocol | 把架构规则做成 **AST 静态检查**（`scripts/check_layering.py`，INV-41 "0 反向 import"）——架构不腐化的唯一可靠方式 | `scripts/check_layering.py`, `ifai/common/protocols.py` | P0 |
| X-2 | **PEP 621 打包 / PEP 561 类型标记 / pytest + coverage** | Python Packaging Authority. *PEP 621*, 2020；*PEP 561*, 2018；pytest / coverage.py 项目（MIT） | 声明式元数据、可安装包、类型提示分发 | `pyproject.toml` + `py.typed` + pytest/cov | 行覆盖 ≥80% 作为硬门槛（R-X-02）；配合分层的黄金测试，形成双保险 | `pyproject.toml`, `ifai/py.typed`, `tests/*` | P0 |
| X-3 | **持续集成矩阵** | GitHub Actions 文档；软件工程的 matrix testing 实践 | 多 OS × 多 Python 版本跑同一套测试 | ubuntu + windows × py3.11/3.12/3.13 | 明确「**windows + 3.13 必过**，其余 best-effort 但 badge 可见」，把注意力放在真实目标环境而非虚荣矩阵 | `.github/workflows/ci.yml` | P1 |

---

## 11. 审计执行表（给 `scripts/audit_no_reinvention.py` 看的最小集合）

| 层 | 要求最少条目数 | 本表条目编号 | 必须存在的 P0 条目 |
| --- | --- | --- | --- |
| L0 | ≥ 3 | 0-1 … 0-7（7 条） | 0-1, 0-2, 0-3, 0-5 |
| L1 | ≥ 2 | 1-1 … 1-6（6 条） | 1-1, 1-2, 1-3 |
| L2 | ≥ 3 | 2-1 … 2-8（8 条） | 2-1, 2-3, 2-5, 2-6, 2-8 |
| L3 | ≥ 3 | 3-1 … 3-6（6 条） | 3-1, 3-2, 3-4 |
| L4 | ≥ 3 | 4-1 … 4-7（7 条） | 4-1, 4-3, 4-4, 4-6 |
| L5 | ≥ 3 | 5-1 … 5-9（9 条） | 5-1, 5-2, 5-3, 5-4, 5-6 |
| L6 | ≥ 3 | 6-1 … 6-7（7 条） | 6-1, 6-3, 6-6 |
| L7 | ≥ 2 | 7-1 … 7-7（7 条） | 7-1, 7-2, 7-3 |
| L8 | ≥ 2 | 8-1 … 8-6（6 条） | 8-1, 8-5, 8-6 |
| L9/X | ≥ 2 | 9-1 … 9-3, X-1 … X-3（6 条） | 9-1, 9-2, X-1, X-2 |
| **合计** | **≥ 25 条（PRD G3）** | **本表共 68 条** | — |

**关联不变量：INV-42**（L0–L9 每层均有 citation 条目，覆盖率 10/10）。

## 12. 延链路与新 treadmill：如何添加新条目

1. 在实现某个新算子/算法**之前**，先在本表追加一行（哪怕是 `[PLANNED]` 状态），并在对应层 `DESIGN.md` 写 `Priors:` 段。
2. 实现完成后，回填「涉及代码文件」列的实际路径。
3. 在 `CITATION.md` 追加人类可读条目（含论文名、年份、arXiv 号或 URL、许可）。
4. 若creditized/free 的第三方代码被直接复制（哪怕片段），必须在 `NOTICE` 记录其原始许可。
5. 运行 `python scripts/audit_no_reinvention.py` 确认 0 缺失。

> **诚实原则**：复述 Hui: 我们不能声称"超越"了我们只是"移植"的东西。「超越/组合之处」列只允许填写 (a) 可验证的性能/内存改进，(b) 跨技术的组合方式，(c) 更强/更严格的验证口径（如新增的不变量），(d) 面向环境约束的等价替换（须指明等价性如何验证）。**禁止填写 Marketing 性空话。**

---

*本文件随实现进展回填；任何新增的核心实现都必须先在此登记出处。*
