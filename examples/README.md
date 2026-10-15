# 示例 (Examples)

本目录给出 **ifai** 各层能力的可运行示例。所有示例均为 CPU 推理、零外部
依赖, 每个可在 ~120s 内跑完 (R-L9-03)。

> 运行方式 (从仓库根目录):
> ```bash
> python examples/quickstart.py
> ```
> 也可在已 `pip install -e .` 后任意位置运行; 示例会自动把仓库根加入 `sys.path`。

| 文件 | 层 | 演示内容 | 零重写来源 |
| --- | --- | --- | --- |
| `quickstart.py` | L9 | 三行调用 `IFAI.chat` / `search` / `run` | HF pipeline / OpenAI SDK 范式 |
| `demo_cli.py` | L9 | 通过子进程演示 `ifai` CLI 子命令 | — |
| `quantize_int8.py` | L0/L1 | INT8 per-channel 量化 (INV-07) 与 INT4 打包往返 (R-L1-02) | LLM.int8 (Dettmers 2022) |
| `lora_finetune.py` | L4 | LoRA merge 等价 (INV-24) + 极简微调 | LoRA (Hu et al. 2022) |
| `agent_tools.py` | L6 | ReAct + 计算器工具求解算术 (INV-33/35) | ReAct (Yao 2022) |
| `rag_local.py` | L5 | BM25 + 稠密 + RRF 融合 + Cross-Encoder 重排 | BM25/RRF/Cross-Encoder |
| `serve_client.py` | L7 | 启动 OpenAI 兼容服务并用 urllib 请求 | OpenAI REST 协议 |

`common.py` 为内部工具, 负责训练一个微型模型到临时目录, 供上述示例复用
(参考 nanogpt 极简训练循环, Karpathy)。
