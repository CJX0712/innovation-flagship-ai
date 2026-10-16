# 创新旗舰 AI · ifai

> **世界顶级 · 零依赖 · 可运行 · 可复现 · 可验证** 的旗舰级 AI 系统参考实现。
> 作者：**晨星 (Morningstar)** · GitHub [@CJX0712](https://github.com/CJX0712)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org)
[![Zero Dependency](https://img.shields.io/badge/Hard%20Dep-numpy%20only-green.svg)](https://numpy.org)
[![Tests](https://img.shields.io/badge/tests-passing%20(263%2B)-brightgreen.svg)]()
[![Zero-Rewrite](https://img.shields.io/badge/zero--rewrite-L0--L9%20audited-success.svg)]()

`ifai` 是一套**纯 NumPy（仅 `numpy>=1.24` 硬依赖）**实现的旗舰 AI 全栈：从算子内核、
Transformer 架构、推理引擎、训练/微调、检索/记忆、智能体，到 OpenAI 兼容服务与评测对齐。
遵循 **「零重写 · 集大成」** 原则——每一层都建立在全球公认的最强成果之上（FlashAttention、
RoPE、GQA、SwiGLU、MoE、PagedAttention/vLLM、连续批处理、投机解码、BM25、HNSW、RRF、
ReAct/Reflexion、MemGPT、LoRA、AdamW、nanogpt 训练循环……），并通过 **42 条不变量（INV）**
与 **60 项需求** 进行可验证对照。

---

## ✨ 核心特性

- 🧮 **零硬依赖**：仅 `numpy`。100% CPU 推理路径，无需 GPU / CUDA / torch 即可跑通全流程。
- 🏗️ **十层架构 L0–L9**：单层聚焦、互相解耦，从算子到应用全链路打通。
- 🔬 **42 条可验证不变量**：数值/算法/系统层均有「解析对照」黄金测试（梯度 vs 有限差分、FlashAttention vs 朴素、困惑度 vs 手工公式、连续批处理 vs 串行……）。
- 🚀 **前沿推理技术**：PagedAttention 分页 KV 缓存、连续批处理（Orca）、投机解码、GQA、RoPE(NTK)。
- 🤖 **完整智能体**：ReAct + Reflexion + MemGPT 式记忆 + AST 白名单安全沙箱。
- 🔎 **检索增强**：BM25 + 稠密向量 + HNSW 近邻 + RRF 融合 + Cross-Encoder 重排。
- 🌐 **OpenAI 兼容服务**：`/v1/models`、`/v1/completions`、`/v1/chat/completions`，生态零改动接入。
- 📉 **可训练 + 量化感知**：MoE / LoRA / QAT(INT8 STE 伪量化) / gelu 开箱即用，微型 Transformer **≤60 秒 CPU** 完成训练，`ifai train && ifai demo` 离线即跑；INT8/INT4 RTN 一键部署量化。
- ✅ **三层训练保证**：五轴一致性（注意力×scaling / 激活 / MoE / INT8 / LoRA，训练↔推理 save→load ≤1e-9）+ 特征组合矩阵（参数集对齐 + 前向一致 + B=0 等价基座）+ 端到端可训性（真实语料损失单调下降），全部由黄金测试锁定。
- 📜 **零重写审计**：`scripts/audit_zero_rewrite.py` 保证十层均引用公认方案。

---

## 🏛️ 十层架构（集大成）

| 层 | 名称 | 关键复用（零重写） | 模块 |
|----|------|--------------------|------|
| **L0** | 算子 / 内核 | 在线 softmax（FlashAttention, Dao 2022）、logsumexp 稳定、int8/int4/fp16 量化 | `core/` |
| **L1** | 张量 / 数值基础 | 全局 PCG64 RNG（bitwise 可复现）、数值稳定基元 | `rng.py`, `core/` |
| **L2** | 模型架构 | Transformer + RMSNorm + RoPE(NTK) + GQA + SwiGLU + MoE + 权重绑定 | `models/` |
| **L3** | 推理引擎 | PagedAttention(vLLM) + 连续批处理(Orca) + 投机解码 + 采样 | `engine/` |
| **L4** | 训练 / 微调 | AdamW + 余弦退火 + LoRA + MoE 训练 + QAT(Jacob 2018) + SwiGLU/gelu + nanogpt 循环 + 手写反向模式自动微分 | `train/` |
| **L5** | 检索 / 记忆 | BM25 + 稠密索引 + HNSW + RRF + Cross-Encoder 重排 | `retrieval/` |
| **L6** | 智能体 | ReAct(Yao) + Reflexion(Shinn) + MemGPT(Packer) + AST 安全沙箱 | `agent/` |
| **L7** | 服务 / 编排 | OpenAI REST 兼容 + SSE 流式 | `serve/` |
| **L8** | 评估 / 对齐 | 困惑度(Jelinek) + 吞吐基准(MLPerf 法) + GSM8K-mini 管线 | `eval/` |
| **L9** | 应用 / 接口 | CLI（`ifai`）+ Python SDK | `cli/`, `sdk.py` |

详见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)、[`docs/PRD.md`](docs/PRD.md)、[`docs/PRIOR_ART.md`](docs/PRIOR_ART.md)。

---

## 🚀 快速开始

```bash
# 1) 安装（仅 numpy 硬依赖）
pip install -e .

# 2) 训练一个微型模型（默认 ≤60s CPU）
ifai train --out ./ifai_model --steps 120

# 3) 离线演示（自动训练（若缺失）后生成续写）
ifai demo --model ./ifai_model --prompt "人工智能" --max-new 24

# 4) 生成 / 对话 / 检索 / 智能体 / 评测 / 分词
ifai generate --prompt "人工智能"
ifai search   --query "注意力"
ifai agent    --question "计算 (3+5)*2"
ifai eval     --model ./ifai_model
ifai tokenize --text "人工智能 模型"

# 5) 启动 OpenAI 兼容服务
ifai serve --model ./ifai_model --port 8000
#   curl http://127.0.0.1:8000/v1/models
#   curl -X POST http://127.0.0.1:8000/v1/completions -d '{"prompt":"人工智能","max_tokens":16}'
```

> 无需联网、无需 GPU。`ifai doctor` 可自检运行环境。

---

## 🎛️ 训练特性与配置开关

`ModelConfig` 一站式驱动训练端 (`TinyTrainer`) 与推理端 (`Transformer`)，所有开关
**两端同名同语义**，save→load 往返一致（≤ 1e-9）：

| 开关 | 默认 | 说明 |
|------|------|------|
| `n_kv_heads` | `=n_heads` (MHA) | 设为 `< n_heads` 即 GQA（如 `2`） |
| `rope_scaling` | `1.0` | NTK-aware 外推缩放（`>1` 支持更长上下文，训练/推理同一频率生成） |
| `activation` | `"swiglu"` | `"swiglu"` \| `"gelu"`（dense 与 MoE 两 FFN 路径均生效） |
| `use_moe` + `n_experts` / `moe_topk` | `False` | Switch/Top-k 稀疏专家 (Fedus 2022, INV-14)，训练端可微路由 |
| `use_lora` + `lora_r` / `lora_alpha` | `False` | 低秩适配 (Hu 2022, INV-24)：B=0 初始等价基座、基座冻结仅训 A/B |
| `use_qat` | `False` | 量化感知训练 (Jacob 2018)：训练期 STE 伪量化，INT8 部署劣化 ≤ 非 QAT 基线 |
| 权重绑定 `tie_word_embeddings` | `True` | INV-12：`lm_head is embed`，量化/训练两端维持绑定语义 |

```python
from ifai.models.config import ModelConfig
from ifai.train.trainer import TinyTrainer

cfg = ModelConfig(vocab_size=512, d_model=64, n_layers=4,
                  use_moe=True, n_experts=8, moe_topk=2,   # MoE
                  use_lora=True, lora_r=8, lora_alpha=16,  # LoRA
                  use_qat=True)                            # QAT
TinyTrainer(cfg).train(corpus_ids, steps=200)              # 三特性组合训练
```

部署量化（训练后一键 INT8/INT4）：

```python
from ifai.models.transformer import Transformer

model = Transformer.load("./ifai_model")
model.quantize_int8()   # 或 model.quantize_int4() — per-channel RTN, INV-07 误差界
```

---

## 🧪 测试与审计

```bash
pytest tests/                       # 263 项黄金测试 (INV 对照)
python scripts/audit_zero_rewrite.py  # 零重写审计: 十层均引用公认方案
```

测试覆盖：算子数值（`test_core`）、架构与不变量（`test_models`）、手写自动微分 vs 有限差分（`test_train`）、推理引擎（`test_engine`）、检索（`test_retrieval`）、智能体（`test_agent`）、评测（`test_eval`）、OpenAI 服务（`test_serve`）。

---

## 📁 项目结构

```
ifai/
├── core/       L0 算子内核 (softmax/rope/swiglu/量化)
├── models/     L2 架构 (transformer/tokenizer/config)
├── engine/     L3 推理 (kvcache/sampler/scheduler/speculative)
├── train/      L4 训练 (autodiff/optim/lora/trainer)
├── retrieval/  L5 检索 (bm25/dense/hnsw/rrf/rerank)
├── agent/      L6 智能体 (tools/sandbox/memory/react)
├── serve/      L7 服务 (OpenAI 兼容)
├── eval/       L8 评测 (perplexity/benchmark/harness)
├── cli/        L9 命令行
├── sdk.py      L9 Python SDK
└── rng.py      L1 全局可复现 RNG
docs/           ARCHITECTURE / PRD / PRIOR_ART / TASKS
tests/          黄金测试 (INV 对照)
scripts/        零重写审计
```

---

## 📜 零重写原则（集大成）

本项目**严禁从零造轮子**。每个子模块的顶部 docstring 均标注其复用的世界顶级方案，
并由 `scripts/audit_zero_rewrite.py` 自动核验——任意分层缺失引用即视为违规（CI 门禁）。
详见 [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md)。

---

## 🛣️ 路线图

- [x] L0–L9 全栈可运行参考实现 + 42 不变量黄金测试
- [x] 零重写审计门禁
- [x] 量化部署示例（int8/int4 RTN 一键量化 + QAT 量化感知训练）
- [ ] 更大规模 CPU 训练配方与基线（`docs/BASELINE.md`）
- [ ] 更多检索后端（FAISS 可选依赖）与多模态接入
- [ ] 分布式部署示例

---

## 📖 引用

```bibtex
@software{ifai2026,
  title  = {ifai: 创新旗舰 AI — 零重写·集大成 全栈参考实现},
  author = {晨星 (Morningstar)},
  year   = {2026},
  url    = {https://github.com/CJX0712/innovation-flagship-ai},
  note   = {Apache-2.0}
}
```

详见 [`CITATION.md`](CITATION.md)。

---

## 📄 许可证

[Apache License 2.0](LICENSE) · © 晨星 (Morningstar) / CJX0712。
