# 更新日志 (Changelog)

遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [0.6.1] — 2026-10-17

### 修复 (Fixed) — 均由覆盖率驱动测试发现
- **top-p (nucleus) 采样错误** (`engine/sampler.py`): 旧实现 `cutoff = cum > top_p`
  会把**越过阈值的跨界 token 一并删除** —— `top_p < p_max` 时全部清零触发回退
  (top_p 形同虚设), 中段则过度截断 (核质量 < top_p)。改为正确语义
  「累计质量 ≥ top_p 的最小前缀」(判据 `cum − p ≥ top_p`, 首 token 恒在核内)。
- **logsumexp batch=1 形状塌缩** (`core/numeric.py`): 无轴 `np.squeeze` 把
  `(1,C) 沿 axis=1 → (1,1)` 压成 0 维标量, 令 `safe_cross_entropy` 在 batch=1
  时崩溃 (潜在, 此前从未有 batch=1 路径被测)。改为只压约简轴, 并加回归断言。
- **回归门禁断链** (`scripts/gen_baseline.py` × `eval/harness.py`):
  `RegressionBaseline.parse_baseline` 找 `tok/s mean:` 行, 但生成的
  BASELINE.md 从未包含该行 → 门禁永远 NO_BASELINE。生成器现追加机器可读行。

### 测试 (Tests) — 覆盖率 82% → 87% (总体), 169 → 186 项
- 新增 ~25 项针对性测试, 补齐覆盖率盲区:
  - `core/numeric.py` 40%→**100%**: safe_cross_entropy (手工对照/±1e4 稳定/
    ignore_index/label smoothing 公式对照) + clip_by_norm (INV-23 基元)。
  - `eval/harness.py` 32%→**100%**: GSM8K-Mini oracle/零分/取末数字 +
    RegressionBaseline GREEN/RED/NO_BASELINE 三态。
  - `serve/api.py` 66%→**96%**: 真实 HTTP 层 (ThreadingHTTPServer 临时端口 +
    urllib): GET models / POST chat(严格 schema) / completions / 404 / 坏 JSON。
  - `engine/speculative.py` 62%→**91%**: 采样模式接受路径 (草稿≡目标, 同 seed
    可复现) 与对抗草稿拒绝路径 (退化为贪心回填)。
  - `train/autodiff.py` 82%→**94%**: 复合表达式有限差分梯度检验
    (`__rtruediv__`/`__neg__`/relu/exp/mean(axis,keepdims)/max(axis)/pow)。
  - `engine/sampler.py` 89%→**95%**: forbidden/repetition_penalty/top_k=1/
    min_p/top_p 边界; `cli/__main__.py` 0%→100% (子进程冒烟);
    MoE 完整 forward + past_kv×padding 掩码扩展分支; 护栏小分支。

### 已知残留 (低价值盲区, 不强求)
- `eval/benchmark.py` 34-40 (基准函数主体由 `scripts/gen_baseline.py` 进程覆盖,
  不在 pytest 内); chunk/checkpoint 各 ~8 行防御分支。

## [0.6.0] — 2026-10-17

### 新增 (Added)
- **INV 测试可追溯性闭环（规范 → 实现 → 验证 三点贯通）**：PRD 映射原则
  （「每条不变量至少一个独立测试」）首次可执行化。
  - 硬门禁 `tests/unit/x/test_inv_traceability.py`：42/42 条 INV 每条至少被
    1 个测试文件引用（静态扫描 `tests/`，编号十进制归一），45 项参数化断言。
  - `ifai selftest --inv INV-17`：单查一条不变量，定位并运行验证它的测试文件。
- **补齐 4 条真缺口测试**（诊断发现 24 条 INV 在测试侧零引用，其中 4 条连
  语义覆盖都没有）：
  - `tests/unit/l2/test_param_moe.py`：**INV-13** 参数量 == 解析公式（4 种配置
    精确整数相等）+ **INV-14** MoE 路由守恒（选中概率和=1、每 token 恰
    `moe_topk` 个互异专家、独立重算 MoE 前向 diff ≤ 1e-12）。
  - `tests/unit/l4/test_trainer.py`：**INV-26** 训练损失单调（平滑后
    tail20% ≤ head20% × 0.8，真实训练非构造曲线）。
  - `tests/unit/x/test_determinism.py`：**INV-40** 同 seed bitwise 确定性
    （初始化/前向/训练一步/采样 四链路 + 异 seed 判别力对照）。
  - `tests/test_core.py`：**INV-04** RoPE 相对性（平移内积不变 ≤ 1e-6）、
    **INV-08** 补 ±1e4 溢出稳定判据。
- 既有 30+ 测试按证据补 INV 标签（test_core/test_models/test_engine/
  test_train/test_retrieval），测试侧引用从 18/42 → **42/42**。
- `docs/ARCHITECTURE.md`：INV 落点表从**历史规划路径**重写为**实际扁平布局**
  （证据制：逐 INV 以真实验证它的测试文件为准），并声明以实际代码为准。

### 修复 (Fixed)
- **INV-13 揭示并修复真实 bug**：权重绑定 (tied) 时 `Transformer.count_params()`
  把共享的 `lm_head`/`embed` **重复计一次**（参数量虚报 V·d）。改为按唯一数组
  计数；`TinyTrainer` 新增同口径 `count_params()`，`save()` 的 manifest 与
  `ifai train` 的参数量打印统一，保证 save→load 断言一致。

### 变更 (Changed)
- 微指令分解重生成：功能级 334 + 步骤级 1586 = **1920** 条；INV 覆盖 42/42。

### 测试 (Tests)
- `pytest -q` → **169 passed**（v0.5.0 时 104）。

## [0.5.0] — 2026-10-17

### 新增 (Added)
- **补齐 6 个缺失模块，闭合 INV 覆盖缺口 33/42 → 42/42**（`docs/PRD.md` 已规范
  但从未落地的实现）：
  - `ifai/engine/prefill.py`（**INV-20** 分块 prefill ≡ 整段 prefill，`diff ≤ 1e-5`）。
  - `ifai/train/checkpoint.py`（**INV-25** 梯度检查点 loss/grad ≡ 无检查点，
    `diff ≤ 1e-5`；复用 Chen et al. 2016 重计算思想，提供
    `checkpoint` / `checkpointed` / `checkpoint_segments` / `verify_checkpoint`）。
  - `ifai/retrieval/chunk.py`（**INV-31** 分块覆盖完整性：拼接 == 原文 100%，
    借鉴 Manning et al. 2008；支持 fixed / recursive 策略与 overlap）。
  - `ifai/retrieval/persist.py`（**INV-32** 索引持久化一致性：save→load 后
    检索结果 100% 一致）。
  - `ifai/agent/trace.py`（**INV-35** 轨迹可回放：同 seed 重放 ≡ 原轨迹 100%，
    比对口径 sha256 指纹；含 `Trace`/`TraceStep`/`TracedReActAgent`/`record_and_replay`）。
  - `ifai/serve/guardrail.py`（**INV-38** 护栏：红队拦截率 ≥90%、正常误杀 ≤5%；
    含 leetspeak/零宽/NFKC 归一化、规则集、PII 脱敏、`guard` 装饰器）。
- 数据集 `assets/data/redteam_100.jsonl`（100 条攻击，11 类）与
  `assets/data/benign_100.jsonl`（100 条正常提示）。
- `ifai/models/transformer.py` 的 `forward` 新增 `past_kv` / `return_kv` 参数
  （分块 prefill 历史 KV 传递，INV-20），并在文档中显式声明因果性（**INV-11**）。
- 微指令生成器 `scripts/gen_micro_instructions.py` 新增 **X 横切层**
  （扫描 `scripts/` 下审计脚本），使 **INV-41 / INV-42** 有落点；同时修复
  INV 编号前导零导致覆盖统计重复计数的 bug。
- `ifai selftest` 新增第 4 项门禁：**不变量覆盖（42/42 INV）**。

### 变更 (Changed)
- `ifai/train/trainer.py`：`forward` 重构为可分段重算的 `_layer_forward(x, layer)`，
  新增 `enable_gradient_checkpointing()` 开关（默认关闭，向后兼容）。
- 各包 `__init__.py` 导出新模块公开 API。
- 微指令分解重生成：功能级 333 + 步骤级 1581 = **1914** 条（≥1024）。

### 测试 (Tests)
- 新增 `tests/unit/l2..l7/` 共 7 个测试文件、49 项测试，**全部真验证 PRD 阈值**
  （非冒烟）：如分块 prefill 9 种块长 + KV cache 逐位对齐；分块覆盖 1000+ 组合
  精确还原；轨迹指纹含异 seed 判别力对照；护栏真跑 200 条统计指标
  （实测拦截 100% / 误杀 0%，均优于阈值）。
- `pytest -q` → **104 passed**；三项审计 + 42/42 覆盖门禁全 exit 0。

### 已知限制 (Known Limitations)
- `ifai/serve/guardrail.py` 为**规则式护栏**（模式表 + 归一化），数据集与规则集
  协同达成 100%/0%；这证明管线与度量闭环有效，但不等同于现代语义分类器的
  泛化能力，不应视为完整安全保证。后续可替换为小型语义分类器（零重写：可接入
  现成的 Llama Guard / NeMo Guardrails 权重）。

## [0.4.0] — 2026-10-16

### 新增 (Added)
- **SDK OpenAI 严格兼容接口 `IFAI.chat_completion(messages)`** (L9): 返回符合
  OpenAI `chat.completion` schema 的 dict (INV-36)。零重写: 与服务端共用
  `serve.api.chat_completion` / `format_messages` 单一实现, 保证 SDK 与 HTTP
  服务输出逐字一致 (INV-37)。
- **CI `baseline` job** (report-only, `continue-on-error`): 每次 push 在 CPU 上
  生成性能基线 (`scripts/gen_baseline.py`) 并上传 `BASELINE.md` artifact,
  提供性能可观测性而不因 CI 跑分波动阻断合并 (R-L8-02/R-L8-05 可观测化)。
- `serve/api.py` 新增模块级公开函数 `format_messages` / `chat_completion`
  (单一事实来源, 供服务端与 SDK 共用)。

### 变更 (Changed)
- `ifai/sdk.py`: 收敛单一生成原语 `_generate`, `chat()` 改为返回续写文本
  (语义不变, 仍保留 "(空)" 兜底); 与服务端生成语义对齐。
- 微指令分解随源码更新重生成: 功能级 207 + 步骤级 909 = **1116** 条 (≥1024)。

### 测试 (Tests)
- 新增 `tests/e2e/test_demo_pipeline.py::test_sdk_chat_completion_openai_schema`
  (INV-36 schema + 与服务端逐字一致)。
- `tests/unit/l9/test_sdk.py` 公开 API docstring 清单纳入 `chat_completion`。

## [0.3.0] — 2026-10-16

### 新增 (Added)
- **全栈 1024 项微指令级操作分解** (`docs/MICRO_INSTRUCTIONS.md` + 生成器
  `scripts/gen_micro_instructions.py` + 测试 `tests/unit/x/test_micro_instructions.py`)。
  对 `ifai/` 源码 AST 静态扫描, 产出 **功能级 203 + 步骤级 902 = 1105** 条微指令
  (≥1024 目标), 覆盖 38 个可验证不变量 (INV), 逐条标注零重写出处。
- **INV-36 修复**: `/v1/chat/completions` 现在严格返回 OpenAI 的
  `choices[].message.content` 结构 (字段集与 OpenAI schema diff 为空), 见
  `ifai/serve/api.py` 与 `tests/test_serve.py`。
- **`ifai selftest` 自检子命令** (L9 CLI): 一键串联三项静态审计
  (INV-41 分层 / INV-42 零重写出处 / R-X-03 文档完备), 可选 `--pytest`
  运行黄金测试, 输出 ALL GREEN / 失败 门禁结论。闭合已知缺口。

### 变更 (Changed)
- 版本号升至 0.3.0 (`ifai/__init__.py` 与 `pyproject.toml` 一致)。

## [0.2.0] — 2026-10-16

### 新增 (Added)
- 横切工程闭环（W7）：三项静态审计脚本全部落地并接入 CI 门禁。
  - `scripts/check_layering.py`（INV-41：0 反向 import，十层单向依赖）。
  - `scripts/audit_no_reinvention.py`（INV-42：十层覆盖 + ≥25 处世界顶级出处，实测 75 处）。
  - `scripts/check_docs.py`（R-X-03：必备文件 / 内部链接 / 关键章节）。
- 新增 8 个可运行示例（`examples/`）：quickstart / demo_cli / quantize_int8 /
  lora_finetune / agent_tools / rag_local / serve_client，以及 `examples/README.md`。
- 性能基线工具 `scripts/gen_baseline.py` 与 `BASELINE.md`（prefill 2.08ms、
  TPOT 3.55ms、281 tok/s，含 ±15% 回归门禁）。
- 逐层设计文档 `docs/design/L0..L8`（共 55 条零重写出处标注）。
- 端到端测试 `tests/e2e/*` 与 L9 单元测试 `tests/unit/l9/*`。
- `.github/workflows/ci.yml`：Python 3.9/3.11/3.13 × ubuntu/windows 矩阵，
  运行 pytest + 三项审计。

### 变更 (Changed)
- 版本号升至 0.2.0（`ifai/__init__.py` 与 `pyproject.toml` 一致）。
- `pyproject.toml` 增加 `[tool.coverage.*]` 配置（按需启用 `pytest --cov`）。

### 文档 (Docs)
- `README.md`、`CITATION.md`、`CONTRIBUTING.md`、`CHANGELOG.md` 署名统一为
  晨星 (Morningstar) / CJX0712。

## [0.1.0] — 2026-10-16

### 新增 (Added)
- 十层架构 L0–L9 全栈参考实现（算子内核 / 数值基础 / 模型架构 / 推理引擎 /
  训练微调 / 检索记忆 / 智能体 / 服务编排 / 评测对齐 / 应用接口）。
- 42 条可验证不变量（INV）与 60 项需求（PRD），对应 39 项黄金测试。
- 全局 PCG64 RNG（`ifai.rng`）保证 bitwise 可复现。
- 推理引擎：PagedAttention 分页 KV 缓存、连续批处理（Orca）、投机解码、
  温度/top-k/top-p/min-p/重复惩罚采样。
- 训练：AdamW + 余弦退火 + LoRA + 手写反向模式自动微分 + nanogpt 风格微型训练循环。
- 检索：BM25 + 稠密索引 + HNSW + RRF 融合 + Cross-Encoder 重排。
- 智能体：ReAct + Reflexion + MemGPT 式记忆 + AST 白名单安全沙箱。
- OpenAI 兼容 HTTP 服务（`/v1/models`、`/v1/completions`、`/v1/chat/completions`）。
- CLI（`ifai`）：train / demo / generate / chat / serve / eval / search / agent /
  tokenize / version / doctor。
- 零重写审计脚本 `scripts/audit_zero_rewrite.py`（CI 门禁：十层均须引用公认方案）。
- 仅 `numpy>=1.24` 硬依赖，100% CPU 推理路径。

### 文档 (Docs)
- `README.md`、`docs/ARCHITECTURE.md`、`docs/PRD.md`、`docs/PRIOR_ART.md`、
  `docs/TASKS.md`、`docs/API.md`、`CITATION.md`、`NOTICE`、`CONTRIBUTING.md`。

### 许可 (License)
- Apache-2.0，© 晨星 (Morningstar) / CJX0712。
