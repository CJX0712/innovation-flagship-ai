"""v0.7.14 产品入口接线回归 (审计缺口 1/2: L5/L6 卖点此前"测试专属")。

锁定三件事:
1. `ifai index` 子命令真实存在并可落盘构建混合索引 (BM25+Dense+RRF);
2. `ifai search` 走混合检索且中文按字符级分词可真实命中
   (旧实现按空格分词 — 中文查询永远匹配不到任何文档, 恒空输出);
3. agent 入口: 沙箱注册为 code_exec 工具 (旧实现仅 calculator, SafeCodeExecutor
   是死导入) + ReActAgent 携带记忆与 Reflexion 真实生效。
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from ifai.agent.memory import MemoryStore
from ifai.agent.react import ReActAgent
from ifai.agent.tools import default_registry
from ifai.retrieval.persist import PersistentIndex, load_index


# ---------------- 1. 混合检索入口 ----------------

def test_default_registry_has_code_exec():
    """审计缺口 2 回归: 沙箱必须注册为 code_exec 工具 (可执行 + 危险代码被拒)。"""
    reg = default_registry()
    assert "code_exec" in reg.tools, "SafeCodeExecutor 未接入注册表 (死导入回归!)"
    out = reg.call("code_exec", {"code": "print(6 * 7)"})
    assert out["ok"] is True and "42" in out["stdout"]
    # 危险代码: 静态检查拒绝 (AST 白名单)
    blocked = reg.call("code_exec", {"code": "import os"})
    assert blocked["ok"] is False and blocked["violations"], "危险代码未被沙箱拦截"


def test_persistent_index_char_tokenization_hits():
    """中文语料字符级分词下混合检索必须真实命中 (旧实现空分词恒无结果的修复)。"""
    corpus = ["变压器网络由注意力机制构成", "模型通过学习数据来提升能力",
              "分词把文本切分成更小的单元"]
    idx = PersistentIndex().fit([list(d) for d in corpus], texts=corpus)
    hits = idx.search(list("注意力"), top_k=3)
    assert hits, "字符级混合检索零命中 — 中文分词修复回归!"
    assert hits[0].text == corpus[0], "含查询词的文档应排第一"
    # 落盘 → 加载 → 再检 (INV-32 往返)
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        path = idx.save(td)
        idx2 = load_index(path)
        hits2 = idx2.search(list("注意力"), top_k=3)
        assert [h.key() for h in hits2] == [h.key() for h in hits], (
            "落盘往返后检索结果不一致 (INV-32)")


# ---------------- 2. agent 入口: 记忆 + Reflexion ----------------

def test_react_agent_memory_and_reflexion_live():
    """ReActAgent 入口必须真实携带记忆 (finish 落库) 与 Reflexion
    (步数耗尽 → 反思提示 → 复盘后 Finish), 此前两者从不被入口传入。"""
    reg = default_registry()
    mem = MemoryStore()
    saw_reflection = {"hit": False}

    def llm(messages):
        u = messages[-1]["content"]
        if u.startswith("Reflection:"):
            saw_reflection["hit"] = True
            return "Finish: 反思后回答"
        return "Thought: 继续"          # 永不 Finish → 步数耗尽 → 触发 Reflexion

    agent = ReActAgent(llm, reg, max_steps=2, memory=mem)
    out = agent.run("演示问题", reflexion=True)
    assert saw_reflection["hit"], "Reflexion 反思提示未出现 (入口未接线回归!)"
    assert out == "反思后回答"
    assert len(mem) > 0, "MemoryStore 未记录事件 (入口未接线回归!)"


# ---------------- 3. CLI e2e: index → search ----------------

def test_cli_index_then_search_e2e(tmp_path):
    """PRD US-05 真实闭环: `python -m ifai.cli index` 落盘 → `search --index` 命中。"""
    out_dir = tmp_path / "idx"
    r1 = subprocess.run(
        [sys.executable, "-m", "ifai.cli", "index", "--out", str(out_dir)],
        capture_output=True, text=True)
    assert r1.returncode == 0, f"ifai index 失败: {r1.stderr}"
    assert "混合索引已构建" in r1.stdout and out_dir.exists()

    r2 = subprocess.run(
        [sys.executable, "-m", "ifai.cli", "search", "--query", "注意力",
         "--index", str(out_dir)],
        capture_output=True, text=True)
    assert r2.returncode == 0, f"ifai search --index 失败: {r2.stderr}"
    assert "注意力" in r2.stdout, f"落盘索引检索未命中: {r2.stdout!r}"

    r3 = subprocess.run(
        [sys.executable, "-m", "ifai.cli", "search", "--query", "注意力"],
        capture_output=True, text=True)
    assert r3.returncode == 0 and "注意力" in r3.stdout, (
        f"内置语料混合检索未命中: {r3.stdout!r}")
