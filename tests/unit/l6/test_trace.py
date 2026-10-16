"""INV-35 (L6) 轨迹可回放: 同 seed 重放 ≡ 原始轨迹 (100% 一致)。

轨迹单元 = Thought / Action / Action-args / Observation; 比对口径为
``Trace.fingerprint()`` (sha256) 与逐步 ``TraceStep.key()`` 双判据。
"""
from __future__ import annotations

import os

from ifai.agent.tools import default_registry
from ifai.agent.trace import (
    SeededLLM,
    Trace,
    TraceStep,
    TracedReActAgent,
    compare_traces,
    record_and_replay,
    replay_trace,
)

QUESTION = "请帮我算一下 (12+8)*3 的结果，并给出结论。"


def _agent(seed: int, max_steps: int = 4):
    tools = default_registry()
    llm = SeededLLM(tools, seed=seed, max_steps=max_steps)
    return TracedReActAgent(llm, tools, max_steps=max_steps), tools


def test_same_seed_replay_identical():
    """INV-35 主判据: 同 seed 两次运行得到完全相同的轨迹。"""
    a, _ = _agent(7)
    b, _ = _agent(7)
    t1 = a.run_traced(QUESTION, seed=7)
    t2 = b.run_traced(QUESTION, seed=7)
    assert t1.fingerprint() == t2.fingerprint()
    assert compare_traces(t1, t2)
    assert len(t1.steps) > 0


def test_replay_from_recorded_trace_is_identical():
    """按记录的动作重放 (不再让 LLM 重新决策) 得到相同轨迹。"""
    a, tools = _agent(11)
    original = a.run_traced(QUESTION, seed=11)
    replayed = replay_trace(original, tools)
    assert compare_traces(original, replayed), \
        "重放轨迹与原始轨迹不一致 (Observation 未能复现)"


def test_json_roundtrip_preserves_trace(tmp_path):
    """轨迹 JSON 序列化往返后指纹不变。"""
    a, _ = _agent(3)
    t = a.run_traced(QUESTION, seed=3)
    p = str(tmp_path / "trace.json")
    t.to_json(p)
    assert os.path.exists(p)
    restored = Trace.from_json(p)
    assert compare_traces(t, restored)
    assert restored.question == QUESTION
    assert len(restored.steps) == len(t.steps)


def test_trace_steps_record_thought_action_observation():
    """每一步都必须带 Thought / Action / Observation 三元组。"""
    a, _ = _agent(5, max_steps=5)
    t = a.run_traced(QUESTION, seed=5)
    assert len(t.steps) >= 2, "轨迹步数过少, 无法体现回放"
    for s in t.steps:
        assert isinstance(s, TraceStep)
        assert s.thought, "每步必须有 Thought"
        assert s.action, "每步必须有 Action"
        assert s.observation, "每步必须有 Observation"
    assert any(s.action != "finish" for s in t.steps), "应至少有一次工具调用"
    assert t.final_answer


def test_different_seeds_produce_different_traces():
    """对照: 不同 seed 应产生不同轨迹 (证明判据有判别力)。"""
    fps = set()
    for seed in range(6):
        a, _ = _agent(seed)
        fps.add(a.run_traced(QUESTION, seed=seed).fingerprint())
    assert len(fps) >= 2, "不同 seed 产生相同轨迹, 无法说明 seed 生效"


def test_record_and_replay_selfcheck_report():
    """一键自检报告: 同 seed 一致 + JSON 往返一致 + seed 敏感。"""
    tools = default_registry()
    rep = record_and_replay(QUESTION, tools, seed=9, max_steps=4)
    assert rep["identical"] is True
    assert rep["json_roundtrip"] is True
    assert rep["seed_sensitive"] is True
    assert rep["n_steps"] >= 1


def test_trace_step_excludes_timestamp_from_key():
    """时间戳不参与比对 (否则回放永远不可能一致)。"""
    s1 = TraceStep(index=0, thought="t", action="a", observation="o", ts=1.0)
    s2 = TraceStep(index=0, thought="t", action="a", observation="o", ts=2.0)
    assert s1.key() == s2.key()
    t1 = Trace(question="q", seed=0, steps=[s1])
    t2 = Trace(question="q", seed=0, steps=[s2])
    assert t1.fingerprint() == t2.fingerprint()


def test_step_key_detects_content_change():
    """内容变化时指纹必须变化 (判据有判别力)。"""
    s1 = TraceStep(index=0, thought="t", action="a", observation="o")
    s2 = TraceStep(index=0, thought="t", action="a", observation="不同观察")
    assert s1.key() != s2.key()
    assert Trace(question="q", steps=[s1]).fingerprint() != \
        Trace(question="q", steps=[s2]).fingerprint()
