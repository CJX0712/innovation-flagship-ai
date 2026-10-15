# L6 · 智能体 (Agent) — 设计说明

## 设计目标

L6 赋予 **ifai** 「感知—思考—行动—记忆」的智能体能力：工具调用、ReAct 推理循环、
分层记忆、以及安全代码沙箱。目标是在零依赖下复现主流智能体范式并保证「危险动作
100% 被拦截」。

本层覆盖 `ifai/agent/`（tools.py、react.py、memory.py、sandbox.py）。

## 复用世界顶级成果（零重写 · 集大成）

- **MemGPT 记忆分层**（Packer et al. 2023，"MemGPT: Towards LLMs as Operating
  Systems"）：复用「情景记忆（带时间戳事件流）+ 语义记忆（键值知识）」的分层思想
  （`MemoryStore` 的 `episodic` / `semantic`）。
- **ReAct 推理-行动循环**（Yao et al. 2022，"ReAct: Synergizing Reasoning and Acting
  in Language Models"）：复用 Thought/Action/Observation 循环——解析 `Action: name(..)`
  或 `Finish: 答案`，驱动工具调用直至完成。
- **Reflexion 自我反思**（Shinn et al. 2023，"Reflexion: Language Agents with Verbal
  Reinforcement Learning"）：复用「未达成时反思复盘并重试一次」机制（`ReActAgent.run
  (reflexion=True)`）。
- **工具调用 / Function Calling**（OpenAI Function Calling / Toolformer，Schick et al.
  2023）：复用「结构化工具调用 + JSON Schema 校验」范式——`ToolSpec.schema` 描述参数，
  `validate_args` 做 type/required/enum/minimum 校验（INV-33）。
- **安全代码沙箱**（RestrictedPython 与 LLM tool-sandbox 思想）：复用「静态 AST 白名单
  先行拦截 + 子进程隔离 + 超时终止」三层防护（跨平台、不依赖 Unix `rlimit`），INV-34：
  30 条越权用例 100% 静态拦截、超时 100% 终止。
- **AST 白名单表达式求值**（`Calculator`）：复用「仅允许算术 AST 节点」的安全求值，
  禁止函数调用/属性访问，作为工具层的安全算术原语（INV-34 协同）。

## 关键不变量 / 验收

- **INV-33** 工具参数 JSON Schema 校验 100% 命中非法输入（缺参/类型错/越界/枚举外）。
- **INV-34** 静态检查 100% 拦截越权用例（import/os/subprocess/open/eval/exec/文件/
  网络等），超时用例 100% 终止。
- **INV-35** ReAct 循环在 `max_steps` 内必然终止（不会无限行动）；`reflexion` 至多
  额外复盘一次，保证有界步数。

## 工程取舍

1. **静态先行、动态兜底**：沙箱先以 AST 白名单拒绝危险代码，再放子进程隔离执行，
   双重保险；任何一层失效仍有超时 `subprocess.TimeoutExpired` 兜底。
2. **Schema 驱动的工具市场**：`ToolRegistry` + `ToolSpec` 以 JSON Schema 为契约，使
   LLM 输出可被机器校验，非法调用在进函数前即被挡下（INV-33）。
3. **记忆与推理解耦**：`MemoryStore` 独立存在，ReAct 可选地写入「Q/A」事件，使记忆
   可跨轮复用，而不耦合于单轮循环。
4. **`reflexion` 默认关闭、有界重试**：反思只在未 `Finish` 时触发一次，避免「反思→
   仍失败→再反思」的死循环（INV-35 有界性）。
