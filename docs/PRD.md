# 创新旗舰 AI（Innovation Flagship AI）— 产品需求文档（PRD）

| 项 | 内容 |
| --- | --- |
| 文档版本 | v1.0 |
| 作者 | 许清楚（Xu）— Product Manager |
| 项目发起人 / 署名 | 晨星（GitHub: `CJX0712`） |
| 项目代号 | `innovation-flagship-ai` |
| 包 / 命令名 | `ifai` |
| 编程语言 | Python 3.13（兼容 3.11+），核心依赖 numpy；torch 仅作可选对照测试 |
| License | Apache-2.0 |
| 运行环境约束 | Windows x86-64，**无 GPU**，CPU-only 完整推理栈 |
| 文档语言 | 中文（关键术语保留英文） |

---

## 1. 项目定位与一句话定义

一个**零重复造轮子**的旗舰级 AI 系统开源仓库：以「复用 + 组合 + 超越」全球已有最优解为唯一构建原则，把从**算子内核**到**应用接口**的十层技术栈（L0→L9）全部实现、全部可独立验证，并在**纯 CPU 环境**下做到 `pip install -e . && ifai demo` 一条命令出结果。

### 1.1 三条铁律（设计约束，优先级高于任何功能需求）

1. **零重写原则**：任何子模块必须先回答「全球最优公开解是什么」并在 `CITATION.md` 中给出出处（论文 / 官方开源实现），再实现。禁止无出处的"自创"核心算法。
2. **可验证原则**：每一层必须有**黄金对照测试**（解析解 / 暴力枚举 / 参考实现 / 数值梯度 / 守恒律 / 单调性），`ifai selftest` 一键全绿。
3. **闭环原则**：可安装、可运行、可复现、可验证、可贡献（文档 + 示例 + CI 齐备）。

---

## 2. 产品目标（Goals，3 条，均可验收）

| ID | 目标 | 验收口径 |
| --- | --- | --- |
| **G1** | **一条命令跑通的 CPU 全栈**：无需 GPU、无需外网、无需手动下载模型 | 干净 venv 中执行 `pip install -e . && ifai demo`，**端到端 ≤ 120s** 产出可读文本结果；全程 0 网络请求（离线可跑） |
| **G2** | **十层每层可独立验证**：每层至少 1 个黄金对照测试 | `ifai selftest` 输出 L0–L9 全部 PASS；总计 **≥ 30 条不变量**在 CI 中被断言（见 §5） |
| **G3** | **集大成且零重写**：核心算法 100% 有出处，且组合出超越单点方案的系统能力 | `CITATION.md` 覆盖 L0–L9 全部核心算法（**≥ 25 条**出处）；README 明确列出每层「借鉴来源 + 本仓库差异」；审计脚本 `scripts/audit_no_reinvention.py` 检查每层均有 citation 条目，0 缺失 |

---

## 3. 目标用户与核心场景

### 3.1 目标用户

| 用户 | 画像 | 核心诉求 |
| --- | --- | --- |
| U1 学习 / 研究者（学生、高校、自学者） | 想"看懂并跑通"现代 LLM 全栈，但只有一台无 GPU 的笔记本 | 代码可读、每层可单独运行、有解析对照、有出处论文 |
| U2 工程集成者（中小团队后端 / 应用工程师） | 需要一个可嵌入的 CPU 推理 + 检索 + 智能体库，或直接替换 OpenAI 端点 | OpenAI 兼容 API、Python SDK、稳定可复现、体积小 |
| U3 开源评审者 / 贡献者 | 评估仓库是否"名副其实"，或想改进某一层 | 清晰的层次边界、可信的基准、可复现的测试、明确贡献路径 |

### 3.2 核心场景

**场景 A（对应 G1 / U1）：离线教学演示**
用户克隆仓库 → 建 venv → `pip install -e . && ifai demo` → 120 秒内看到微型模型流式生成的一段文本 + 一份"本 demo 调用到的 L0→L9 链路图"。用户随后可逐层运行 `ifai selftest --layer L3` 单独验证推理引擎。

**场景 B（对应 U2）：本地 OpenAI 兼容服务**
用户 `ifai serve --port 8000` → 用现有 OpenAI SDK（仅改 `base_url`）调用 `/v1/chat/completions` 得到 SSE 流式响应 → `ifai index ./docs && ifai search "..."` 构建本地 RAG → `ifai bench` 得到本机吞吐/延迟报告。

**场景 C（对应 U3）：贡献与回归保护**
用户修改 L0 的分块 GEMM → 运行 `pytest` + `ifai bench` → CI 自动比对 `BASELINE.md` 基线（性能回归 > 15% 或任意黄金测试失败即红灯）→ 提交 PR。

---

## 4. 用户故事（US）

> 格式：Given（前置） / When（动作） / Then（可验收结果）

| ID | 用户故事 |
| --- | --- |
| **US-01** | **Given** 一台无 GPU 的 Windows 机器且已安装 Python 3.13，**When** 我在干净虚拟环境执行 `pip install -e . && ifai demo`，**Then** 我在 ≤120s 内看到流式生成的文本输出，且终端打印出本次调用经过的 L0→L9 链路摘要 |
| **US-02** | **Given** 我想确认注意力实现是否正确，**When** 我运行 `ifai selftest --layer L0`，**Then** 系统输出 FlashAttention 风格分块注意力与朴素实现的 `max_abs_diff ≤ 1e-5`，以及峰值内存对比，且结论为 PASS |
| **US-03** | **Given** 我关心量化是否可用，**When** 我执行 `ifai quantize --bits 8 --scheme per-channel` 并随后 `ifai eval --task ppl`，**Then** 我得到量化前后困惑度对比，且 INT8 困惑度退化 ≤ 5%（INT4 ≤ 15%） |
| **US-04** | **Given** 我已在本地微调了一个模型，**When** 我执行 `ifai train --lora-r 8 --corpus data/tiny.txt`，**Then** 训练在 ≤60s 内完成（CPU），loss 曲线单调下降，且 `merge()` 后的权重推理输出与未 merge 输出误差 ≤ 1e-6 |
| **US-05** | **Given** 我有一批本地文档，**When** 我执行 `ifai index ./docs` 然后 `ifai search "如何配置 KV Cache"`，**Then** 我得到 BM25 + 稠密向量 + RRF 融合后的重排结果，且 HNSW 检索相对暴力枚举 `recall@10 ≥ 0.95` |
| **US-06** | **Given** 我需要模型调用工具，**When** 我用 Agent 提问"计算 (17*23+9)/4 并解释"，**Then** Agent 通过 ReAct 调用计算器工具给出正确结果，完整轨迹以 JSON 落盘并可回放（同 seed 结果一致） |
| **US-07** | **Given** 我已有使用 OpenAI SDK 的代码，**When** 我启动 `ifai serve` 并把 `base_url` 指向 `http://localhost:8000/v1`，**Then** 非流式与 SSE 流式两种调用均返回符合 OpenAI schema 的响应，且**流式 chunk 拼接结果与流式关闭时的 content 完全一致** |
| **US-08** | **Given** 我是评审者，**When** 我运行 `pytest && ifai bench && ifai eval --task gsm8k`，**Then** 我得到覆盖率报告、吞吐/延迟报告（写入 `BASELINE.md`）与数学推理评测报告，且所有黄金不变量断言通过 |

---

## 5. 需求池（Requirements Pool）

**优先级定义**：P0 = 必须（MVP 交付硬门槛）；P1 = 应该有（旗舰完整性）；P2 = 可以有（延期不阻塞发布）。
**参考机（REF）**：Windows x86-64 / 4 物理核 / Python 3.13 / numpy 2.x / 无 GPU。所有绝对数值指标均以 REF 为准，CI 以 `BASELINE.md` 实测基线做 ±15% 回归保护。

### L0 — 算子 / 内核层（Operator & Kernel）

| ID | 需求描述 | 优先级 | 验收标准（可量化） |
| --- | --- | --- | --- |
| R-L0-01 | Cache-blocking 分块 GEMM（支持 tiling 参数与循环序可调） | P0 | 与 `numpy.matmul` `max_abs_diff ≤ 1e-5`；相对朴素三重循环在 256³ 上加速 **≥ 2×** |
| R-L0-02 | FlashAttention 风格在线 softmax 分块注意力（含 causal mask） | P0 | 与朴素 attention `max_abs_diff ≤ 1e-5`；峰值内存 **O(N)**，@N=4096 时 ≤ 朴素实现 **20%** |
| R-L0-03 | INT8 量化-反量化内核（per-token × per-channel，对称 / 非对称） | P0 | 反量化重建误差 ≤ `scale/2 + 1e-6`；INT8 下模型 PPL 退化 **≤ 5%** |
| R-L0-04 | INT4 分组量化内核（group ∈ {32, 64, 128}） | P1 | PPL 退化 **≤ 15%**；与逐元素参考实现误差 ≤ 1 个量化步长 |
| R-L0-05 | RoPE 融合算子（含 NTK-aware / 线性插值 scaling） | P0 | 与解析旋转矩阵 `max_abs_diff ≤ 1e-6`；外推 2× 长度后不出现 NaN/Inf |
| R-L0-06 | SwiGLU 融合算子（单 pass，避免中间大张量） | P0 | 与分离实现 `max_abs_diff ≤ 1e-6`；峰值临时内存 ≤ 分离实现 **60%** |
| R-L0-07 | RMSNorm（含 fused residual add，支持 eps 可调） | P0 | 与参考实现 `max_abs_diff ≤ 1e-6`；输入元素量级 1e3 时结果有界无溢出 |

### L1 — 张量 / 数值层（Tensor & Numerics）

| ID | 需求描述 | 优先级 | 验收标准 |
| --- | --- | --- | --- |
| R-L1-01 | NDArray 抽象（shape / dtype / strides / device=cpu），与 numpy 零拷贝互操作 | P0 | `np.asarray(x)` 零拷贝（`.base` 相同）；与 numpy 往返误差 **= 0** |
| R-L1-02 | dtype 体系：fp32 / fp16 / int8 / int4(packed) / fp32→bf16(可选) | P0 | fp32↔fp16 往返相对误差 **≤ 1e-3**；int4 pack/unpack 往返 **逐位一致** |
| R-L1-03 | 数值稳定工具：logsumexp / stable softmax / safe cross-entropy | P0 | logits ∈ [-1e4, 1e4] 时**无 inf/nan**；与解析值 `diff ≤ 1e-6` |
| R-L1-04 | 量化器 RTN（minmax / percentile 标定，对称 + 非对称） | P0 | 重建误差 ≤ 理论界 `scale/2`；标定与反标定往返 `diff ≤ 1e-6` |
| R-L1-05 | GPTQ 风格逐列量化（校准集驱动，OBQ 简化版） | P1 | 同 bit 宽下 PPL **优于 RTN ≥ 10%**；量化耗时 ≤ 60s @ REF |
| R-L1-06 | 分布工具：熵 / KL / 温度缩放 / top-k 截断概率 | P1 | KL(p‖p) = 0 ± 1e-9；KL 非负性恒成立 |

### L2 — 模型架构层（Model Architecture）

| ID | 需求描述 | 优先级 | 验收标准 |
| --- | --- | --- | --- |
| R-L2-01 | Transformer Decoder（pre-RMSNorm + RoPE + GQA + SwiGLU FFN） | P0 | 微型配置参数量 **≤ 2M**；逐层 shape 断言 100% 通过；前向无 NaN |
| R-L2-02 | GQA 分组查询注意力（n_kv_heads 整除 n_heads） | P0 | `n_kv == n_q` 时与 MHA 输出 `diff ≤ 1e-6`；KV Cache 形状缩小为 **1/g** |
| R-L2-03 | MoE 稀疏专家（top-k=2 路由 + 负载均衡辅助损失） | P1 | 路由概率和 = 1（±1e-6）；token 分配守恒；随机数据上专家负载 CV **≤ 0.3** |
| R-L2-04 | 权重绑定（tie word embeddings） | P0 | `lm_head.weight` 与 `embed_tokens.weight` 为同一 tensor（`is` 判定通过）；参数量减少 `|V|×d` |
| R-L2-05 | 配置驱动（dataclass + JSON round-trip） | P0 | 同一 config 两次加载，同输入 logits **bitwise 一致** |
| R-L2-06 | 权重序列化（npz/safetensors 风格 + manifest） | P0 | save→load round-trip 误差 **= 0**；manifest 含 sha256 校验 |
| R-L2-07 | 上下文扩展（NTK-aware / YaRN 简化） | P2 | 外推至 2× 训练长度，PPL 上升 **≤ 10%** |

### L3 — 推理引擎层（Inference Engine）

| ID | 需求描述 | 优先级 | 验收标准 |
| --- | --- | --- | --- |
| R-L3-01 | PagedAttention 风格分页 KV Cache（block_size=16，块表管理） | P0 | 与稠密 KV 输出 `max_abs_diff ≤ 1e-5`；@n=8192 峰值内存 ≤ 预分配稠密方案 **50%** |
| R-L3-02 | 连续批处理调度器（iteration-level admit / preempt） | P0 | greedy 下 batch 结果与串行独立结果**逐 token 100% 一致**；concurrency=8 时聚合吞吐 ≥ 串行 **2×** |
| R-L3-03 | 投机解码（自草稿 / 树状候选，k ≥ 4） | P1 | greedy 下与目标模型自回归**逐 token 100% 一致**；接受率 ≥ 0.4；端到端加速 ≥ 1.3× |
| R-L3-04 | 采样器（temperature / top-p / top-k / min-p / 重复 & 频率惩罚） | P0 | 截断后概率和 = 1 ± 1e-6；`T→0`、`top_k=1` 退化为 greedy（100% 一致） |
| R-L3-05 | Prefill chunking（chunk=512）+ 增量 decode | P1 | 分块 prefill 与整段 prefill 输出 `diff ≤ 1e-5` |
| R-L3-06 | 前缀缓存（prefix caching，哈希/基数树简化版） | P2 | 基准场景命中率 ≥ 50%；命中与未命中路径输出 `diff ≤ 1e-5` |

### L4 — 训练与微调层（Training & Fine-tuning）

| ID | 需求描述 | 优先级 | 验收标准 |
| --- | --- | --- | --- |
| R-L4-01 | AdamW（decoupled weight decay，β1/β2/ε 可调） | P0 | 单步更新与解析公式 `diff ≤ 1e-6`；wd=0 时等价于 Adam |
| R-L4-02 | 余弦退火 + 线性预热调度 | P0 | lr 曲线与解析值 `diff ≤ 1e-9`；warmup 结束时恰为 peak lr |
| R-L4-03 | 梯度裁剪（global L2 norm） | P0 | 裁剪后 `‖g‖ ≤ max_norm + 1e-6`；未超限时梯度**不被修改**（bitwise） |
| R-L4-04 | 反向传播（手写 autodiff，覆盖 matmul/softmax/norm/rope/loss） | P0 | 与有限差分数值梯度相对误差 **≤ 1e-3**（≥ 20 组随机样本） |
| R-L4-05 | 梯度检查点（按层重计算） | P1 | loss 与 grad 与无检查点 `diff ≤ 1e-5`；峰值内存降低 ≥ 30% |
| R-L4-06 | LoRA（r ∈ {1,2,4,8,16}，α 可配，dropout=0） | P0 | `merge()` 后与未 merge 前向 `diff ≤ 1e-6`；`B=0` 时等价冻结基线；可训练参数 ≤ 全参 **5%** |
| R-L4-07 | 微型语料秒级训练（默认 demo 语料） | P0 | CPU 训练 **≤ 60s**；loss 尾部 20% 均值 ≤ 前 20% 均值 × **0.8** |
| R-L4-08 | 混合精度（fp32 主权重 + fp16 计算） | P2 | 与 fp32 训练 loss 曲线偏差 ≤ 5% |

### L5 — 检索与记忆层（Retrieval & Memory）

| ID | 需求描述 | 优先级 | 验收标准 |
| --- | --- | --- | --- |
| R-L5-01 | BM25（k1=1.5，b=0.75） | P0 | 与参考实现 Top-10 **重合度 ≥ 95%**（语料 ≥ 1000 文档，≥ 50 查询） |
| R-L5-02 | 稠密向量检索（微型双塔 encoder，d ≤ 128） | P0 | 索引结果与暴力 TopK **100% 一致** |
| R-L5-03 | HNSW 近似最近邻（M=16, efC=200, efS=64） | P0 | `recall@10 ≥ 0.95`（vs 暴力）；QPS ≥ 暴力 **5×** @ N=10k, d=64 |
| R-L5-04 | RRF 混合融合（k=60） | P0 | 与 RRF 公式逐项一致（≤ 1e-12）；构造数据上融合 Top-10 含各路 Top-1 比例 ≥ 90% |
| R-L5-05 | Cross-Encoder 风格重排 | P1 | 内置 100 查询评测，重排后 **NDCG@10 不低于**融合前 |
| R-L5-06 | 情景记忆（append-only + 时间衰减）与语义记忆（去重归并） | P1 | 注入 50 条事实后，会话结束检索召回 **≥ 90%** |
| R-L5-07 | 分块策略（固定 / 递归字符 / 重叠窗口） | P1 | chunk 拼接对原文覆盖 **100%**，无遗漏无重叠错位 |
| R-L5-08 | 索引持久化与增量更新 | P1 | reload 后检索结果与持久化前 **100% 一致** |

### L6 — 智能体层（Agent）

| ID | 需求描述 | 优先级 | 验收标准 |
| --- | --- | --- | --- |
| R-L6-01 | ReAct 循环（Thought / Action / Observation，最大步数可配） | P0 | 内置 20 个可解任务成功率 **≥ 70%**；超步数时优雅终止并给出结论 |
| R-L6-02 | 工具调用：计算器 / 检索 / 代码执行沙箱（JSON Schema 声明） | P0 | 工具 schema 校验通过率 100%；参数解析错误可在 **≤ 2 次**重试内恢复 |
| R-L6-03 | 代码沙箱安全（AST 白名单 + subprocess 超时 + 输出截断） | P0 | 内置 **30 条越权/逃逸用例拦截率 100%**；超时用例 100% 被终止（默认 3s） |
| R-L6-04 | Planner / Executor 任务分解 | P1 | ≥3 步任务生成的计划为**无环 DAG**且可完整执行 |
| R-L6-05 | Reflexion 自我反思（≤ 3 轮） | P1 | 构造的"首次失败"任务集上二次成功率提升 **≥ 20pp** |
| R-L6-06 | 多智能体编排（角色 + 黑板 / 消息总线） | P2 | 2 智能体协作完成内置任务，消息总线无丢包 |
| R-L6-07 | 轨迹记录与回放（trace JSON） | P1 | 同 seed 回放结果与原始运行 **100% 一致** |

### L7 — 服务与编排层（Serving & Orchestration）

| ID | 需求描述 | 优先级 | 验收标准 |
| --- | --- | --- | --- |
| R-L7-01 | OpenAI 兼容 HTTP API：`/v1/chat/completions`、`/v1/completions`、`/v1/models`、`/v1/embeddings` | P0 | 响应字段集与 OpenAI 官方 schema **一致度 100%**（pydantic 校验 + 字段 diff）；错误码语义一致 |
| R-L7-02 | SSE 流式输出 | P0 | **流式 chunk 内容拼接 == 非流式 content**（同 seed、greedy） |
| R-L7-03 | 并发路由与排队（max_concurrency、排队超时） | P1 | 1000 并发压测**请求丢失 0**；超限返回 429 或排队，不崩溃 |
| R-L7-04 | 护栏：输入/输出过滤、prompt-injection 检测、长度与成本限额 | P0 | 内置 100 条红队用例拦截率 **≥ 90%**；100 条正常用例误杀率 **≤ 5%** |
| R-L7-05 | 可观测性：trace_id、TTFT / TPOT / tokens/s、Prometheus 文本格式 `/metrics` | P1 | 指标与本地计时偏差 **≤ 10%**；每个响应可关联唯一 trace_id |
| R-L7-06 | 优雅启停与 `/healthz` | P2 | SIGTERM 后存量请求处理完毕再退出 |

### L8 — 评测与对齐层（Evaluation & Alignment）

| ID | 需求描述 | 优先级 | 验收标准 |
| --- | --- | --- | --- |
| R-L8-01 | 困惑度评测 | P0 | 与手工计算（平均 NLL 取 exp）`diff ≤ 1e-6` |
| R-L8-02 | 吞吐 / 延迟基准（prefill / decode / TTFT / TPOT），输出 JSON + Markdown | P0 | 重复 3 次变异系数 **CV ≤ 15%**；结果写入 `BASELINE.md` |
| R-L8-03 | GSM8K 风格数学推理评测（few-shot CoT + 答案抽取器） | P1 | 答案抽取器单测准确率 **100%**；评测管线可完整跑通（准确率不设硬门槛，微型模型仅验证管线） |
| R-L8-04 | 事实性与安全性评测 | P1 | 输出结构化报告（JSON），含逐条判定 |
| R-L8-05 | 回归基线 + CI 门禁 | P0 | 性能回归 **> 15%** 或任意黄金测试失败 → CI 红灯 |
| R-L8-06 | 确定性自检 | P0 | 同 seed 两次运行 logits **bitwise 一致** |

### L9 — 应用与接口层（Application & Interface）

| ID | 需求描述 | 优先级 | 验收标准 |
| --- | --- | --- | --- |
| R-L9-01 | CLI `ifai`：`demo / serve / bench / eval / train / quantize / index / search / selftest / version` | P0 | `--help` 覆盖 **100%** 子命令；每个子命令有 e2e 冒烟测试 |
| R-L9-02 | Python SDK 高阶 API：`LLM.chat / Retriever.search / Agent.run` | P0 | `examples/quickstart.py` 一键运行成功；公开 API 100% 有 docstring |
| R-L9-03 | 示例集 ≥ 6 个（demo / quantize / rag / agent / lora / serve） | P1 | 每个示例在 ≤ 120s 内一键跑通 |
| R-L9-04 | 可选 WebUI（零构建依赖静态页 + 后端 API） | P2 | 打开即用，支持对话与检索 |
| R-L9-05 | 打包与发布：`pyproject.toml`、`pip install -e .`、语义化版本 | P0 | 干净 venv 安装 **≤ 60s** 且 `import ifai` 成功 |

### X — 横切工程需求（Cross-cutting）

| ID | 需求描述 | 优先级 | 验收标准 |
| --- | --- | --- | --- |
| R-X-01 | 分层目录与单向依赖（L0 ← L1 ← … ← L9，禁止反向 import） | P0 | `scripts/check_layering.py` 通过，0 违规 |
| R-X-02 | 测试体系：pytest，每层 ≥ 1 黄金测试，总体行覆盖 | P0 | 行覆盖 **≥ 80%**；黄金测试全绿 |
| R-X-03 | 文档集：README / ARCHITECTURE / 各层 DESIGN.md / API / CONTRIBUTING / CHANGELOG / CITATION / LICENSE | P0 | 文档检查脚本验证链接与必备章节 0 缺失 |
| R-X-04 | License Apache-2.0 + 第三方许可聚合 `NOTICE` | P0 | `LICENSE` 完整；所有借鉴实现的许可在 `NOTICE` 列出 |
| R-X-05 | 署名：晨星（GitHub `CJX0712`） | P0 | README / LICENSE copyright / CITATION / `pyproject.toml` authors 四处一致署名 |
| R-X-06 | CI：GitHub Actions（ubuntu-latest + windows-latest，py 3.11 / 3.12 / 3.13） | P1 | 全矩阵绿灯；**3.13 + windows 为必过** |
| R-X-07 | 零重写审计 | P0 | `scripts/audit_no_reinvention.py` 检查每层均有 citation 条目，0 缺失 |
| R-X-08 | 全局可复现 RNG（python `random` / numpy / 自研采样器统一 seed） | P0 | `ifai --seed N` 两次运行输出 bitwise 一致 |

---

## 6. 关键不变量清单（Invariants）

> 每条不变量必须写成断言，且在 `ifai selftest` 中可执行。验证方式分为五类：
> **A**=解析解对照 / **B**=数值解（有限差分）对照 / **C**=暴力枚举或朴素实现对照 / **D**=参考实现对照 / **E**=结构性（守恒、单调、退化一致性）

| ID | 层级 | 不变量 | 验证方式 | 判据 |
| --- | --- | --- | --- | --- |
| INV-01 | L0 | 分块 GEMM ≡ 朴素矩阵乘 | C | `max_abs_diff ≤ 1e-5` |
| INV-02 | L0 | 在线 softmax 分块注意力 ≡ 朴素 softmax attention | C | `max_abs_diff ≤ 1e-5` |
| INV-03 | L0 | RoPE 旋转 ≡ 解析 2D 旋转矩阵（θ_i = 10000^(-2i/d)） | A | `max_abs_diff ≤ 1e-6` |
| INV-04 | L0 | RoPE 相对性：⟨R_m q, R_n k⟩ 仅依赖 (m−n) | E | 平移 (m,n) 与 (m+c,n+c) 结果 `diff ≤ 1e-6` |
| INV-05 | L0 | SwiGLU 融合 ≡ 分离计算 | C | `max_abs_diff ≤ 1e-6` |
| INV-06 | L0 | RMSNorm 缩放不变性方向：RMSNorm(a·x) 符号一致且范数归一 | E | `‖RMSNorm(x)‖₂ ≈ √d`（`diff ≤ 1e-5`），对任意 a>0 成立 |
| INV-07 | L0/L1 | 量化-反量化误差有界 | A | `|x − dequant(quant(x))| ≤ scale/2 + 1e-6` |
| INV-08 | L1 | logsumexp 稳定性：log Σexp(x_i) ≡ log Σexp(x_i − max)·exp(max) | A | `diff ≤ 1e-6`，且 x 含 ±1e4 时不溢出 |
| INV-09 | L1 | softmax 平移不变性 | E | `softmax(x) == softmax(x + c)`，`diff ≤ 1e-9` |
| INV-10 | L2 | GQA(n_kv = n_q) ≡ MHA | E | `max_abs_diff ≤ 1e-6` |
| INV-11 | L2 | Causal masking：位置 i 的输出不依赖 j>i 的 token | E | 扰动未来 token，输出 bitwise 不变 |
| INV-12 | L2 | 权重绑定：lm_head 与 embedding 为同一对象 | E | `a is b` 且数值逐位相同 |
| INV-13 | L2 | 参数量守恒：可训练参数计数 == 按配置解析计算值 | A | 精确相等（整数） |
| INV-14 | L2 | MoE 路由：Σ softmax(router logits) = 1，且 Σ 专家分配 token 数 = 输入 token 数 | E | `diff ≤ 1e-6` / 整数守恒 |
| INV-15 | L3 | 分页 KV Cache ≡ 稠密 KV Cache | C | `max_abs_diff ≤ 1e-5` |
| INV-16 | L3 | 连续批处理 ≡ 串行独立推理（同 seed、greedy） | C | 逐 token **100% 一致** |
| INV-17 | L3 | 投机解码（greedy）≡ 目标模型自回归 greedy | C | 逐 token **100% 一致**（这是投机解码正确性的核心不变量） |
| INV-18 | L3 | 采样分布归一：截断后 Σp = 1 | E | `|Σp − 1| ≤ 1e-6` |
| INV-19 | L3 | 采样退化：`T→0` 与 `top_k=1` 均 ≡ greedy | E | 100% 一致 |
| INV-20 | L3 | 分块 prefill ≡ 整段 prefill | C | `diff ≤ 1e-5` |
| INV-21 | L4 | AdamW 单步更新 ≡ 解析公式 | A | `diff ≤ 1e-6`；`wd=0` 时 ≡ Adam |
| INV-22 | L4 | 反向传播梯度 ≡ 有限差分数值梯度 | B | 相对误差 **≤ 1e-3**（≥ 20 组样本） |
| INV-23 | L4 | 梯度裁剪：裁剪后 `‖g‖ ≤ max_norm`；未超限不改梯度 | E | `≤ max_norm + 1e-6` / bitwise 相同 |
| INV-24 | L4 | LoRA `merge()` ≡ 未 merge 前向 | E | `diff ≤ 1e-6`；`B=0` 时 ≡ 冻结基线 |
| INV-25 | L4 | 梯度检查点：loss/grad ≡ 无检查点 | C | `diff ≤ 1e-5` |
| INV-26 | L4 | 训练损失单调性（滑动平滑后） | E | 尾部 20% 均值 ≤ 前 20% 均值 × 0.8 |
| INV-27 | L5 | BM25 ≡ 参考实现排序 | D | Top-10 重合度 **≥ 95%** |
| INV-28 | L5 | HNSW 召回 ⊇ 近似暴力枚举 | C | `recall@10 ≥ 0.95` |
| INV-29 | L5 | 稠密索引 TopK ≡ 暴力 TopK | C | 100% 一致 |
| INV-30 | L5 | RRF 分数 ≡ 公式 `Σ 1/(k + rank_i)` | A | `diff ≤ 1e-12` |
| INV-31 | L5 | 分块覆盖完整性 | E | chunk 拼接 == 原文（归一化后 100%） |
| INV-32 | L5 | 索引持久化一致性 | E | reload 前后检索结果 100% 一致 |
| INV-33 | L6 | 工具 schema 可校验性 | E | 所有内置工具通过 JSON Schema Draft-07 校验 |
| INV-34 | L6 | 沙箱隔离：危险操作被拦截 | E | 30 条越权用例 **100% 拦截**；超时 100% 终止 |
| INV-35 | L6 | 轨迹可回放：同 seed 重放 ≡ 原始轨迹 | E | 100% 一致 |
| INV-36 | L7 | OpenAI schema 兼容 | D | 字段集 diff 为空 |
| INV-37 | L7 | 流式 ≡ 非流式：Σ chunks == 完整 content | C | 字符串完全相等 |
| INV-38 | L7 | 护栏拦截 / 误杀 | E | 红队拦截率 ≥ 90%，正常误杀 ≤ 5% |
| INV-39 | L8 | 困惑度 ≡ exp(平均 NLL) | A | `diff ≤ 1e-6` |
| INV-40 | L8 | 确定性：同 seed 两次运行 bitwise 一致 | E | `np.array_equal(..., equal_nan=True)` |
| INV-41 | X | 分层依赖单向 | E | 静态分析 0 反向 import |
| INV-42 | X | 零重写审计：每层有出处 | E | citation 覆盖 10/10 层 |

---

## 7. 非功能需求（NFR）

| 类别 | 要求 |
| --- | --- |
| **性能** | 微型模型（≤ 2M 参数）在 REF 上：单流 decode **≥ 50 tok/s**；连续批处理 concurrency=16 聚合吞吐 **≥ 200 tok/s**；`ifai demo` 冷启动到首 token ≤ 60s。绝对指标用于发布门槛，**CI 使用 `BASELINE.md` 实测基线做 ±15% 回归保护**（跨机器不因绝对指标飘红） |
| **资源** | 常驻内存峰值 ≤ 512MB（不含索引）；仓库体积 ≤ 30MB（含微型权重）；无 GPU 依赖；可选 torch 仅用于对照测试 |
| **可复现** | 全局统一 seed（CLI `--seed` 与 SDK `set_seed()`），覆盖 `random` / `numpy` / 自研采样器；同 seed 两次运行 bitwise 一致（INV-40）；基准与评测结果固化到 `BASELINE.md` |
| **可测试性** | pytest；每层 ≥ 1 黄金测试；总体行覆盖 ≥ 80%；`ifai selftest --layer Lx` 可单独验证任意层；不变量以纯函数断言形式存在，可脱离 CLI 调用 |
| **文档** | 必备清单：`README.md`（30 秒上手）、`ARCHITECTURE.md`（十层图与依赖方向）、每层 `DESIGN.md`（借鉴来源 + 本仓库差异 + 复杂度）、`docs/API.md`、`CONTRIBUTING.md`、`CHANGELOG.md`、`CITATION.md`、`NOTICE`、`LICENSE`；所有公开 API 有 docstring 与可运行示例 |
| **开源合规** | Apache-2.0；`NOTICE` 聚合全部第三方许可；`CITATION.md` 为每个核心算法给出论文/实现的出处；数据集仅使用许可明确的来源，不内嵌受限数据 |
| **署名** | 晨星（GitHub `CJX0712`）署名出现在 README、LICENSE copyright 行、CITATION、`pyproject.toml` authors 四处，保持一致 |
| **兼容性** | Python 3.11 / 3.12 / 3.13（3.13 为首要目标）；Windows / Linux / macOS；核心路径（L0–L4）仅依赖 numpy |
| **安全** | 代码沙箱默认关闭网络与文件系统写权限（AST 白名单 + 禁用 import）；护栏默认开启；不在日志中落盘用户输入明文（可开关） |
| **可维护性** | 单向分层依赖（INV-41）；每层可独立替换实现而不影响上层接口；公共接口变更需更新 `CHANGELOG.md` 与版本语义 |

---

## 8. 交付范围与里程碑建议（供工程排期参考）

| 阶段 | 内容 | 出口标准 |
| --- | --- | --- |
| M1 内核与数值 | L0 + L1 | `ifai selftest --layer L0,L1` 全绿；INV-01..09 通过 |
| M2 模型与引擎 | L2 + L3 | 微型模型可生成；INV-10..20 通过；`ifai demo` 可用 |
| M3 训练与检索 | L4 + L5 | LoRA 训练 ≤ 60s；BM25/HNSW/RRF 达标；INV-21..32 通过 |
| M4 智能体与服务 | L6 + L7 | OpenAI 兼容 API + SSE 可用；沙箱拦截 100%；INV-33..38 通过 |
| M5 评测与交付 | L8 + L9 + 横切 | CI 全矩阵绿灯；文档齐备；覆盖率 ≥ 80%；发布 v0.1.0 |

---

## 9. 待确认问题（Open Questions ≤ 5，附默认假设）

| # | 问题 | 默认假设（若 24h 内未回复即按此执行） |
| --- | --- | --- |
| Q1 | 是否把 PyTorch 作为依赖？ | **否（硬依赖仅 numpy）**。torch 仅作为可选对照测试依赖（`pip install .[dev]`），用于交叉验证 L0/L2/L4 结果；CPU 推理路径 100% numpy，保证无 GPU 环境可跑 |
| Q2 | 代码执行沙箱需要多强的隔离？ | **跨平台中等强度**：AST 白名单（禁用 `import` / `open` / `eval` / `exec`）+ subprocess 子进程 + 超时（默认 3s）+ 输出截断 + 工作目录隔离。不依赖 Unix `resource` rlimit（Windows 不可用）；若后续需要强隔离再引入容器方案（P2） |
| Q3 | 微型模型权重是否随仓库发布？ | **是**：随仓发布 ≤ 10MB 的 `npz/safetensors` 风格权重（含 sha256 manifest），保证 `ifai demo` 完全离线；**同时**保留 `ifai train` 从零训练路径（≤ 60s）以便完全复现。若仓库体积成为问题，退化为 GitHub Release 附件 + 首次运行下载（带校验） |
| Q4 | GSM8K 数据集如何处理？ | **不随仓**：提供 `scripts/download_gsm8k.py` 下载脚本（遵循原数据集许可），并内置 **20 题自造 mini 子集**用于离线 CI；评测以「管线正确性」而非绝对准确率为门槛 |
| Q5 | 是否交付 WebUI？ | **P2，非阻塞**：仅在 M1–M4 全部完成后提供零构建依赖的静态页 + 后端 API；若时间不足则从 v0.1.0 范围中移除，不影响发布 |

---

## 10. 明确排除项（Out of Scope）

- 自研预训练大模型（≥ 100M 参数）与大规模语料爬取
- GPU / CUDA / Triton 内核（本机无 GPU；未来可作为可选后端）
- 分布式训练与多机推理
- 任何需要联网或密钥的第三方 LLM API 代理（护栏仅做本地规则）
- 重复实现已有成熟轮子（违反零重写原则的模块一律拒绝合入）
