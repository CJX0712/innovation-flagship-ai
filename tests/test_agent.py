"""L6 智能体层黄金测试 (INV-33 schema 校验, INV-34 沙箱拦截)。"""
import pytest
from ifai.agent.tools import ToolSpec, validate_args, Calculator, default_registry
from ifai.agent.sandbox import SafeCodeExecutor
from ifai.agent.react import ReActAgent
from ifai.agent.memory import MemoryStore


def test_tool_schema_validation():
    spec = ToolSpec(name="f", description="", schema={
        "type": "object",
        "properties": {"x": {"type": "integer", "minimum": 0}},
        "required": ["x"]}, func=lambda x: x)
    assert validate_args(spec.schema, {"x": 1}) == []
    assert validate_args(spec.schema, {}) != []            # missing required
    assert validate_args(spec.schema, {"x": -1}) != []     # below minimum
    assert validate_args(spec.schema, {"x": "a"}) != []    # wrong type


def test_calculator_safety():
    assert abs(Calculator.run("(2+3)*4") - 20) < 1e-9
    with pytest.raises(ValueError):
        Calculator.run("__import__('os')")


def test_sandbox_blocks_dangerous():
    ex = SafeCodeExecutor()
    banned = [
        "import os\nos.system('ls')",
        "open('/etc/passwd')",
        "eval('1+1')",
        "exec('x=1')",
        "__import__('subprocess')",
    ]
    for code in banned:
        res = ex.run(code)
        assert res["violations"], f"未拦截: {code}"


def test_react_loop_with_mock_llm():
    reg = default_registry()

    def llm(messages):
        u = messages[-1]["content"]
        if "计算" in u:
            expr = u.split("计算")[-1].strip()
            return f'Thought: 计算\nAction: calculator(expression="{expr}")'
        return "Finish: 完成"

    agent = ReActAgent(llm, reg, max_steps=4)
    ans = agent.run("请使用计算器计算 (3+5)*2")
    assert "16" in ans or ans == "完成"


def test_memory_store():
    m = MemoryStore()
    m.remember_event("今天学习了注意力机制")
    m.remember_fact("作者", "晨星")
    assert "注意力" in m.recall_events("注意力")[0]
    assert m.recall_fact("作者") == "晨星"
