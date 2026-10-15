# API 文档 (API Reference)

`ifai` 提供两层接口：**OpenAI 兼容 HTTP 服务**（L7，生态零改动接入）与 **Python SDK**（L9）。

---

## 一、命令行 `ifai`

| 子命令 | 说明 | 关键参数 |
|--------|------|----------|
| `train`  | 训练微型模型（≤60s CPU） | `--out --steps --vocab --d-model --layers --heads --kv-heads --ffn-mult --seq-len --lr --seed` |
| `demo`   | 离线续写演示 | `--model --prompt --max-new --greedy` |
| `generate` | 采样生成 | `--model --prompt --temp --top-p --top-k` |
| `chat`   | 简易对话 | `--model --max-new --temp` |
| `serve`  | 启动 HTTP 服务 | `--model --host --port` |
| `eval`   | 困惑度 + GSM8K-mini 管线 | `--model` |
| `search` | BM25 检索 | `--query --top-k` |
| `agent`  | ReAct 演示智能体 | `--question` |
| `tokenize` | BPE 分词 | `--text --vocab` |
| `version` | 版本信息 | — |
| `doctor` | 环境自检 | — |

---

## 二、OpenAI 兼容 HTTP 服务（L7，INV-36 / INV-37）

启动：`ifai serve --model ./ifai_model --port 8000`

### `GET /v1/models`
```json
{ "object": "list",
  "data": [ { "id": "ifai-flagship", "object": "model", "owned_by": "morningstar" } ] }
```

### `POST /v1/completions`
```jsonc
// 请求
{ "prompt": "人工智能", "max_tokens": 16, "temperature": 0.8, "top_p": 0.9, "stream": false }
// 响应 (非流式)
{ "id": "cmpl-ifai", "object": "text_completion", "model": "ifai-flagship",
  "choices": [ { "text": "…", "index": 0, "finish_reason": "stop" } ],
  "usage": { "prompt_tokens": 0, "completion_tokens": 16, "total_tokens": 16 } }
```

### `POST /v1/chat/completions`
```jsonc
{ "messages": [ { "role": "user", "content": "你好" } ], "max_tokens": 16 }
```

### 流式（SSE）
`"stream": true` 时逐 token 以 `data: {...}` 推送，**拼接结果 == 非流式结果（INV-37）**。

---

## 三、Python SDK（L9，`ifai.sdk.IFAI`）

```python
from ifai.sdk import IFAI

ai = IFAI(model_dir="./ifai_model")
print(ai.chat("人工智能是什么？"))        # 对话
print(ai.search("注意力"))               # 检索（基于内置语料 BM25）
print(ai.run("计算 (3+5)*2"))            # 智能体（ReAct + 计算器）
```

---

## 四、核心模块 API（L0–L8）

```python
from ifai.models.config import ModelConfig
from ifai.models.transformer import Transformer, load_model
from ifai.engine.sampler import Sampler
from ifai.engine.scheduler import ContinuousScheduler
from ifai.retrieval.bm25 import BM25
from ifai.agent.react import ReActAgent

cfg = ModelConfig(vocab_size=512, d_model=128, n_layers=3, n_heads=4, n_kv_heads=2)
model = Transformer(cfg)

# 推理（支持 position_ids / key_padding_mask 以支撑连续批处理 INV-16）
logits = model.forward(token_ids)                 # [B,T,V]
logits = model.forward(token_ids, position_ids=..., key_padding_mask=...)

# 采样
Sampler(seed=0).sample(logits[0, -1], temperature=0.8, top_p=0.9)

# 连续批处理（与串行贪心 100% 一致 INV-16）
ContinuousScheduler(model.forward).generate(prompts, max_new_tokens=8, greedy=True)
```

完整十层接口与参数见 `docs/ARCHITECTURE.md` 与各模块 docstring。
