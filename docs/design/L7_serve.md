# L7 · 服务 / 编排 (Serving & Orchestration) — 设计说明

## 设计目标

L7 把 L2 模型 + L3 引擎封装成**对外可服务的 HTTP 接口**，并兼容 OpenAI REST 协议，
使现有生态（LangChain / OpenAI SDK）零改动接入，支持 SSE 流式输出。目标是「协议
兼容 + 流式/非流式语义一致」。

本层覆盖 `ifai/serve/`（api.py）。

## 复用世界顶级成果（零重写 · 集大成）

- **OpenAI REST 协议**（OpenAI API v1：`/v1/models`、`/v1/completions`、
  `/v1/chat/completions`）：复用其请求/响应 schema 与字段命名（`id`/`object`/
  `model`/`choices`/`usage`），使第三方客户端零适配接入；INV-36：响应字段集与
  OpenAI schema diff 为空。
- **Server-Sent Events（SSE）流式**（W3C "Server-Sent Events" 规范）：复用
  `text/event-stream` 增量推送范式实现 token 级流式；工程上「生成长度恒为
  `max_new`（或遇 `<eos>` 提前）」保证流式 == 非流式（INV-37）。
- **ThreadingHTTPServer（Python 标准库 `http.server`）**：复用标准库多线程 HTTP
  服务承载路由，零额外 Web 框架依赖；`route()` 逻辑层可在不联网情况下被直接单测。

## 关键不变量 / 验收

- **INV-36** 响应字段集与 OpenAI schema diff 为空（见 `tests`，字段/类型对齐）。
- **INV-37** 流式 `content` 拼接 == 非流式 `content`（同一生成器、恒定长度）。
- **INV-38** 服务层为「纯编排」：仅转调 L2/L3，不引入任何模型权重或采样逻辑，
  保证升级内核不影响协议兼容性。

## 工程取舍

1. **协议兼容优先**：刻意对齐 OpenAI 字段名与端点，牺牲少量「自有扩展」换取生态
   零改动接入——这是 L7 的核心价值。
2. **逻辑层与 HTTP 解耦**：`OpenAIServer._gen_tokens` / `completions` / `chat` 为纯
   Python 逻辑，可在无 socket 下测试；HTTP 仅由 `create_handler` 薄封装，便于单测
   与替换传输层。
3. **零 Web 框架依赖**：用标准库 `ThreadingHTTPServer` 而非 FastAPI/Flask，契合
   本栈「CPU-only、零依赖」定位；代价是缺少自动 OpenAPI 文档。
4. **流式靠统一生成器保证等价**：`_gen_tokens` 无论 `stream` 与否都生成相同 token
   序列，流式只是「增量消费」同一序列，从根本上保证 INV-37。
