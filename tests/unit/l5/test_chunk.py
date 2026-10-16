"""INV-31 (L5) 分块覆盖完整性: chunk 拼接 == 原文 (归一化后 100%)。

覆盖 100% 的口径:
1. 按 offset 去重叠拼接 (``reconstruct``) 与原文 **逐字符相等**;
2. 归一化 (空白折叠) 后相等 —— PRD 判据;
3. ``overlap=0`` 时块文本**直接串联**即原文 (零丢字、零重复);
4. 每块的 ``text`` 必须等于 ``原文[start:end]`` (边界不丢不重可核验)。
"""
from __future__ import annotations

import random

import pytest

from ifai.retrieval.chunk import (
    TextChunker,
    chunk_text,
    coverage_report,
    normalize_text,
    reconstruct,
    verify_coverage,
)

SEED = 31


def _corpus(n: int = 120) -> list[str]:
    """构造 120 篇覆盖各类边界的语料 (含 CJK / 空白 / 空串 / 短文本)。"""
    rng = random.Random(SEED)
    base = ("人工智能正在改变软件工程。检索增强生成把外部知识接入模型。"
            "Attention is all you need. 量化技术降低显存占用，LoRA 用极少参数微调。")
    words = ["模型", "推理", "训练", "索引", "向量", "Transformer", "AI", "缓存", "调度"]
    out: list[str] = ["", "短", "。", "\n\n\n", "   ", base[:7]]
    for i in range(n - len(out)):
        k = rng.randint(0, 60)
        if i % 7 == 0:
            out.append(" ".join(rng.choice(words) for _ in range(k)))
        elif i % 7 == 1:
            out.append((base * rng.randint(1, 4))[: k + 5])
        elif i % 7 == 2:
            out.append("第一段。\n\n第二段，有逗号。\n第三段！")
        elif i % 7 == 3:
            out.append("A" * rng.randint(1, 300))
        elif i % 7 == 4:
            out.append("  空白\t混合\n" + base[: k] + "  ")
        elif i % 7 == 5:
            out.append("中英混排 mixed 文本 with punctuation，用于覆盖测试。" * rng.randint(1, 3))
        else:
            out.append("".join(rng.choice("甲乙丙丁戊，。！？\n") for _ in range(k)))
    return out


def test_reconstruction_equals_source_for_all_configs():
    """120 篇语料 × 多种 (size, overlap, strategy): 拼接 == 原文 100%。"""
    corpus = _corpus()
    checked = 0
    for text in corpus:
        for size in (1, 2, 3, 5, 8, 16, 64, 512):
            for overlap in (0, 1, 2, 3):
                if overlap >= size:
                    continue
                for strategy in ("fixed", "recursive"):
                    chunks = chunk_text(text, chunk_size=size, overlap=overlap,
                                        strategy=strategy)
                    assert reconstruct(chunks) == text, \
                        f"拼接 != 原文 (size={size}, overlap={overlap}, {strategy})"
                    assert verify_coverage(text, chunks, normalized=True), \
                        f"归一化覆盖 != 100% (size={size}, overlap={overlap}, {strategy})"
                    checked += 1
    assert checked >= 1000, f"核验组合数偏少: {checked}"


def test_no_loss_no_duplicate_without_overlap():
    """overlap=0: 块文本直接串联即原文, 且块区间严格连续。"""
    for text in _corpus(40):
        chunks = chunk_text(text, chunk_size=7, overlap=0)
        assert "".join(c.text for c in chunks) == text
        for i, ch in enumerate(chunks):
            assert ch.start == (0 if i == 0 else chunks[i - 1].end)
            assert ch.end - ch.start == len(ch.text)
        if chunks:
            assert chunks[-1].end == len(text)


def test_overlap_window_boundaries_are_exact():
    """overlap>0: 相邻块起点 = 上一块终点 - overlap (重叠量精确, 不丢不重)。"""
    text = "人工智能与大规模语言模型的工程实践。" * 12
    overlap = 8
    chunks = chunk_text(text, chunk_size=40, overlap=overlap)
    assert len(chunks) >= 3
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur.start == prev.end - overlap
        assert text[cur.start:prev.end] == prev.text[-overlap:], "重叠部分内容一致"


def test_chunk_text_matches_source_slice():
    """每个块的 text 必须等于原文对应切片 (offset 元数据可信)。"""
    for text in _corpus(40):
        for size in (3, 9, 32):
            for ch in chunk_text(text, chunk_size=size, overlap=2):
                assert text[ch.start:ch.end] == ch.text
                assert ch.index >= 0


def test_edge_cases():
    """空串 / 短于块长 / 块长为 1 的边界行为。"""
    assert chunk_text("", chunk_size=8, overlap=0) == []
    one = chunk_text("你好世界", chunk_size=8, overlap=0)
    assert len(one) == 1 and one[0].text == "你好世界"
    singles = chunk_text("abcde", chunk_size=1, overlap=0)
    assert "".join(c.text for c in singles) == "abcde"
    assert all(len(c.text) == 1 for c in singles)


def test_invalid_parameters_rejected():
    """非法参数必须报错 (防止静默丢字)。"""
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=4, overlap=4)      # overlap >= chunk_size
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=4, overlap=-1)
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=4, strategy="unknown")


def test_coverage_report_fields():
    """覆盖报告字段自洽 (净覆盖字符数 == 原文长度)。"""
    text = "分块覆盖完整性验证文本。" * 20
    chunks = chunk_text(text, chunk_size=24, overlap=6)
    rep = coverage_report(text, chunks)
    assert rep["exact"] is True
    assert rep["normalized"] is True
    assert rep["offsets_valid"] is True
    assert rep["contiguous"] is True
    assert rep["covered_chars"] == rep["n_text_chars"] == len(text)


def test_text_chunker_class_api():
    """面向对象封装与函数式 API 一致。"""
    text = "检索增强生成的索引构建流程。" * 9
    c = TextChunker(chunk_size=20, overlap=5, strategy="recursive")
    chunks = c.split(text)
    assert c.reconstruct(chunks) == text
    rep = c.verify(text)
    assert rep["exact"] and rep["normalized"]
    many = c.split_many([text, "短文本", ""])
    assert len(many) == 3 and many[1][0].text == "短文本" and many[2] == []


def test_normalize_text_collapses_whitespace():
    """归一化口径: 空白折叠 + 去首尾 (INV-31 的比较口径)。"""
    assert normalize_text("  a \t b\n c  ") == "a b c"
    assert normalize_text("分块\t覆盖\n验证") == "分块 覆盖 验证"
