#!/usr/bin/env python
"""智能体 + 工具演示 — ReAct 循环调用内置计算器求解算术 (INV-33/INV-35)。

零重写: 复用 Yao et al. 2022 (ReAct) 的 Thought-Action-Observation 范式,
与 AST 白名单安全求值计算器 (参考 SafeCode / AST 沙箱思想)。

运行:
    python examples/agent_tools.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ifai.agent.react import ReActAgent  # noqa: E402
from ifai.agent.tools import default_registry  # noqa: E402


def _build_llm(question: str):
    def llm(messages):
        user = messages[-1]["content"]
        if user.startswith("Observation:"):
            # 看到工具返回的观察值, 直接 Finish
            return f"Finish: {user.split('Observation:', 1)[1].strip()}"
        expr = user.split("计算", 1)[1].strip() if "计算" in user else user
        return f'Thought: 该问题需要计算, 调用计算器。\nAction: calculator(expression="{expr}")'

    return llm


def main() -> None:
    question = "请计算 (17*23+9)/4"
    agent = ReActAgent(_build_llm(question), default_registry(), max_steps=4)
    answer = agent.run(question)
    print(f"问题: {question}")
    print(f"智能体结论: {answer}")
    print("(演示结束: 智能体通过 Thought→Action→Observation→Finish 调用计算器得到结果)")


if __name__ == "__main__":
    main()
