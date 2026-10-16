"""INV-32 (L5) 索引持久化一致性: reload 前后检索结果 100% 一致。

落盘 = 稀疏侧 (BM25 全部状态) + 稠密侧 (向量矩阵) + sha256 manifest;
重载后对同一批 query 的 Top-K (doc_id, score) 必须逐项相同。
"""
from __future__ import annotations

import json
import os

import pytest

from ifai.retrieval.persist import (
    PersistentIndex,
    load_index,
    save_index,
    verify_persistence,
)

_DOCS = [
    "注意力机制让模型聚焦输入的关键部分。",
    "变压器网络由多头注意力堆叠而成。",
    "量化技术能显著降低模型显存占用。",
    "低秩适配 LoRA 用极少参数微调大模型。",
    "检索增强生成从外部知识库取回相关资料。",
    "分页 KV 缓存提升长上下文推理的内存效率。",
    "投机解码用小模型草稿加速大模型生成。",
    "向量数据库支撑大规模语义检索。",
    "混合检索融合稀疏与稠密两路结果。",
    "重排模型对候选文档做精排。",
] * 4                                     # 40 篇文档


def _tok(s: str) -> list[str]:
    return [c for c in s if not c.isspace()]


def _corpus() -> list[list[str]]:
    return [_tok(d) for d in _DOCS]


def _queries() -> list[list[str]]:
    qs = ["注意力 量化", "检索 向量", "LoRA 微调", "KV 缓存", "投机解码",
          "重排 精排", "变压器 多头", "显存 占用", "混合检索", "知识库 资料"]
    return [_tok(q) for q in qs]


def _index() -> PersistentIndex:
    return PersistentIndex(dim=64, seed=3).fit(_corpus(), texts=_DOCS)


def test_reload_search_results_identical(tmp_path):
    """INV-32 主判据: 落盘前后 Top-10 结果 (id + 分数) 100% 相同。"""
    idx = _index()
    path = str(tmp_path / "idx_a")
    idx.save(path)
    reloaded = PersistentIndex.load(path, verify=True)

    checked = 0
    for q in _queries():
        before = idx.search_keys(q, top_k=10)
        after = reloaded.search_keys(q, top_k=10)
        assert before == after, f"重载后结果不一致: query={q}\n{before}\n{after}"
        checked += 1
    assert checked == len(_queries())


def test_reload_hits_text_and_meta_identical(tmp_path):
    """命中结果的文本与元数据也必须一致 (可回溯)。"""
    idx = _index()
    path = str(tmp_path / "idx_b")
    idx.save(path)
    reloaded = PersistentIndex.load(path)
    for q in _queries():
        ha = idx.search(q, top_k=5)
        hb = reloaded.search(q, top_k=5)
        assert [h.key() for h in ha] == [h.key() for h in hb]
        assert all(h.text for h in hb), "重载后应能回填文档原文"


def test_verify_reload_report_is_consistent(tmp_path):
    """一致性自检报告: 0 处不一致, 最大分数差为 0。"""
    idx = _index()
    rep = verify_persistence(idx, str(tmp_path / "idx_c"), _queries(), top_k=10)
    assert rep["consistent"] is True, rep
    assert rep["n_mismatches"] == 0
    assert rep["max_score_diff"] == 0.0
    assert rep["n_queries"] == len(_queries())


def test_manifest_sha256_detects_corruption(tmp_path):
    """sha256 manifest 能检出索引被篡改 (完整性校验)。"""
    idx = _index()
    d = str(tmp_path / "idx_d")
    idx.save(d)
    p = os.path.join(d, "index.json")
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    obj["texts"][0] = "被篡改的文档"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    with pytest.raises(AssertionError):
        PersistentIndex.load(d, verify=True)
    # 关闭校验时可加载 (用于应急恢复路径)
    assert PersistentIndex.load(d, verify=False) is not None


def test_functional_api_roundtrip(tmp_path):
    """函数式 save_index / load_index 与类方法等价。"""
    idx = _index()
    d = str(tmp_path / "idx_e")
    save_index(idx, d)
    other = load_index(d)
    q = _tok("注意力 量化")
    assert idx.search_keys(q, 10) == other.search_keys(q, 10)


def test_persist_state_completeness(tmp_path):
    """落盘必须包含 BM25 的全部统计状态 (df / idf / doc_len / avgdl)。"""
    idx = _index()
    d = str(tmp_path / "idx_f")
    idx.save(d)
    with open(os.path.join(d, "index.json"), encoding="utf-8") as f:
        state = json.load(f)
    bm = state["bm25"]
    assert len(bm["docs"]) == len(_corpus())
    assert len(bm["doc_len"]) == len(_corpus())
    assert len(bm["df"]) > 0 and len(bm["idf"]) > 0
    assert bm["avgdl"] > 0
    assert len(state["texts"]) == len(_DOCS)


def test_empty_corpus_search_safe():
    """空语料检索不报错, 返回空结果。"""
    empty = PersistentIndex(dim=32).fit([])
    assert empty.search(["x"], top_k=5) == []
