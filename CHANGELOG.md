# 更新日志 (Changelog)

遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

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
