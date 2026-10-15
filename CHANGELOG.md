# 更新日志 (Changelog)

遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

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
