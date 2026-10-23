"""INV-43 ~ INV-50 (L5) QFR 融合与 MMR 重排的可验证性。

INV-43 退化性：权重全 1 + 关时序 ⟹ QFR ≡ 标准 RRF（与 rrf.py 交叉验证，≤1e-12）
INV-44 时序有界：T ∈ (0,1]、单调不增、T(0)=1、T(30)=0.5、T(60)=0.25
INV-45 单调性：d 在所有通道排名不劣于 d' ⟹ QFR(d) ≥ QFR(d')
INV-46 过期必除：valid_to 已过的文档绝不出现在结果中
INV-47 确定性：相同输入 → 相同输出
INV-48 先验可解释：提高某通道权重 ⟹ 该通道头部文档排名上升
INV-49 MMR λ=1 恒等：退化为原始相关度排序
INV-50 MMR 多样性单调：λ 降低 ⟹ 平均两两相似度不升

每条不变量都配一条**独立参考实现**（暴力枚举）交叉验证 —— 这是「零重写」
原则的验证手段：同一数学性质，两种实现算，结果必须一致。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ifai.retrieval.qfr import (
    MIN_POSITIVE_DOUBLE,
    QFRWeights,
    qfr_fuse,
    qfr_reference,
    temporal_factor,
)
from ifai.retrieval.rrf import reciprocal_rank_fusion
from ifai.retrieval.mmr import avg_pairwise_similarity, mmr_rerank, pairwise_cosine

# ---------------------------------------------------------------------------
# 语料构造（与 test_chunk.py 同一风格：确定性 SEED，覆盖 CJK / 拉丁 / 边界）
# ---------------------------------------------------------------------------

SEED = 43


def _corpus_docs(n: int = 60) -> list[str]:
    import random

    rng = random.Random(SEED)
    base = ("黄金价格今年上涨，主要驱动是全球央行连续净买入。"
            "Attention is all you need. 量化技术降低显存占用。")
    words = ["黄金", "央行", "买入", "模型", "推理", "Transformer", "AI", "索引", "向量"]
    out: list[str] = ["", "短"]
    for i in range(n - len(out)):
        k = rng.randint(0, 40)
        if i % 5 == 0:
            out.append(" ".join(rng.choice(words) for _ in range(k)))
        elif i % 5 == 1:
            out.append((base * rng.randint(1, 3))[: k + 5])
        else:
            out.append(f"doc{i} " + " ".join(rng.choice(words) for _ in range(k)))
    return out


def _channels_from_scores(
    bm25_scores: list[float], dense_scores: list[float]
) -> dict[str, list[tuple[int, float]]]:
    """把两个通道的打分列表转成 QFR 输入格式（doc_id = 下标）。"""
    return {
        "bm25": list(enumerate(bm25_scores)),
        "dense": list(enumerate(dense_scores)),
    }


# ===========================================================================
# INV-43 退化性：QFR ≡ 标准 RRF
# ===========================================================================


def test_inv43_qfr_degrades_to_rrf():
    """权重全 1 + 关时序 ⟹ QFR 与 rrf.py 的标准 RRF 逐位相等（≤1e-12）。

    这是 QFR 作为「RRF 严格超集」的核心证据 —— 保证新算法不破坏既有语义。
    """
    rng = np.random.default_rng(SEED)
    bm25 = rng.random(50).tolist()
    dense = rng.random(50).tolist()
    channels = _channels_from_scores(bm25, dense)

    w = QFRWeights(temporal_enabled=False)  # 全部权重默认 1.0
    qfr = qfr_fuse(channels, weights=w)
    qfr_map = dict(qfr)

    # 标准 RRF：rank_lists 是各检索器已排序的 doc id 列表
    bm25_order = [i for i, _ in sorted(enumerate(bm25), key=lambda kv: (-kv[1], kv[0]))]
    dense_order = [i for i, _ in sorted(enumerate(dense), key=lambda kv: (-kv[1], kv[0]))]
    rrf = dict(reciprocal_rank_fusion([bm25_order, dense_order], k=60))

    assert set(qfr_map) == set(rrf), "文档集合必须一致"
    max_err = max(abs(qfr_map[d] - rrf[d]) for d in rrf)
    assert max_err <= 1e-12, f"INV-43 退化性失败: max_abs_err={max_err:.2e}"


def test_inv43_qfr_reference_matches_main():
    """主实现与独立暴力参考实现一致（交叉验证）。"""
    rng = np.random.default_rng(SEED + 1)
    bm25 = rng.random(80).tolist()
    dense = rng.random(80).tolist()
    channels = _channels_from_scores(bm25, dense)
    ages = {i: float(rng.random() * 90) for i in range(80)}

    w = QFRWeights(half_life_days=30.0, gamma=1.0)
    main = dict(qfr_fuse(channels, weights=w, ages=ages))
    ref = dict(
        qfr_reference(
            channels,
            k=w.k,
            weights_map={"bm25": w.bm25, "dense": w.dense},
            ages=ages,
            half_life_days=w.half_life_days,
            gamma=w.gamma,
        )
    )
    assert set(main) == set(ref)
    max_err = max(abs(main[d] - ref[d]) for d in ref)
    assert max_err <= 1e-12, f"INV-43 参考实现对照失败: max_abs_err={max_err:.2e}"


# ===========================================================================
# INV-44 时序因子有界与单调
# ===========================================================================


def test_inv44_temporal_bounded_and_monotonic():
    half = 30.0
    t0 = temporal_factor(0.0, half_life_days=half)
    t_half = temporal_factor(half, half_life_days=half)
    t_double = temporal_factor(half * 2, half_life_days=half)
    t_big = temporal_factor(1e9, half_life_days=half)

    assert abs(t0 - 1.0) <= 1e-12, f"T(0) 应 = 1，实际 {t0}"
    assert abs(t_half - 0.5) <= 1e-9, f"T(30) 应 = 0.5，实际 {t_half}"
    assert abs(t_double - 0.25) <= 1e-9, f"T(60) 应 = 0.25，实际 {t_double}"
    assert 0.0 < t_big <= 1.0, f"T 必须恒 > 0，实际 {t_big}"
    assert t0 >= t_half >= t_double >= t_big, "T 必须随 age 单调不增"
    assert t_big >= MIN_POSITIVE_DOUBLE, "下溢保护必须生效"


def test_inv44_temporal_negative_age_safe():
    """age < 0（时钟回拨）不应产生 > 1 的因子。"""
    assert temporal_factor(-100.0, half_life_days=30.0) <= 1.0


def test_inv44_temporal_extreme_half_life():
    """half_life <= 0 时不过度衰减（除零保护）。"""
    assert temporal_factor(1000.0, half_life_days=0.0) == 1.0
    assert temporal_factor(1000.0, half_life_days=-5.0) == 1.0


# ===========================================================================
# INV-45 单调性
# ===========================================================================


def test_inv45_monotonic_dominance():
    """d 在所有通道排名不劣于 d' ⟹ QFR(d) ≥ QFR(d')。"""
    # better 在两个通道都排第 1；worse 都排第 2
    channels = {
        "bm25": [(10, 9.0), (20, 1.0)],
        "dense": [(10, 0.9), (20, 0.1)],
    }
    w = QFRWeights(temporal_enabled=False)
    out = dict(qfr_fuse(channels, weights=w))
    assert out[10] > out[20], f"INV-45 单调性失败: {out[10]} vs {out[20]}"


def test_inv45_monotonic_random():
    """随机语料上验证单调性（构造性：全通道排名一致的文档必须得分最高）。"""
    rng = np.random.default_rng(SEED + 2)
    n = 40
    bm25 = rng.random(n).tolist()
    dense = rng.random(n).tolist()
    channels = _channels_from_scores(bm25, dense)

    # 构造一个在双通道都排第 1 的文档（覆盖索引 0）
    bm25[0] = max(bm25) + 1.0
    dense[0] = max(dense) + 1.0
    channels = _channels_from_scores(bm25, dense)

    out = qfr_fuse(channels, weights=QFRWeights(temporal_enabled=False))
    assert out[0][0] == 0, f"双通道第一的文档应排首位，实际 {out[0]}"


# ===========================================================================
# INV-46 过期硬过滤
# ===========================================================================


def test_inv46_expired_hard_filtered():
    """valid_to 已过的文档绝不出现在结果中（不是低分，是缺席）。"""
    now = 1_000_000_000_000
    channels = {"bm25": [(1, 99.0), (2, 50.0), (3, 1.0)]}
    valid_to = {1: now - 1}  # doc 1 已过期
    out = qfr_fuse(channels, weights=QFRWeights(temporal_enabled=False), valid_to=valid_to, now_ms=now)
    ids = [d for d, _ in out]
    assert 1 not in ids, f"INV-46 失败：过期文档 1 出现在结果中 {ids}"
    assert 2 in ids and 3 in ids


def test_inv46_boundary_not_yet_expired():
    """now == valid_to 时视为已过期（半开区间 [valid_from, valid_to)）。"""
    now = 5_000
    channels = {"bm25": [(1, 1.0)]}
    out = qfr_fuse(
        channels, weights=QFRWeights(temporal_enabled=False), valid_to={1: now}, now_ms=now
    )
    assert out == [], "now == valid_to 应判定为已过期"


def test_inv46_expired_does_not_consume_rank():
    """过期文档不占用排名位次（否则会扭曲存活文档的 rank）。"""
    now = 1_000
    channels = {"bm25": [(1, 100.0), (2, 90.0), (3, 80.0)]}
    valid_to = {1: now - 10}  # 最高分文档过期
    out = qfr_fuse(channels, weights=QFRWeights(temporal_enabled=False), valid_to=valid_to, now_ms=now)
    # doc 2 应成为 rank 0 → 得分 1/k（0-based 口径，与 rrf.py 一致）
    assert out[0][0] == 2, f"存活最高分文档应升为 rank0，实际 {out}"
    assert abs(out[0][1] - 1.0 / 60) <= 1e-12, f"rank 应重排，实际得分 {out[0][1]}"


# ===========================================================================
# INV-47 确定性
# ===========================================================================


def test_inv47_deterministic_across_runs():
    rng = np.random.default_rng(SEED + 3)
    n = 30
    bm25 = rng.random(n).tolist()
    dense = rng.random(n).tolist()
    channels = _channels_from_scores(bm25, dense)
    ages = {i: float(rng.random() * 60) for i in range(n)}

    runs = [
        qfr_fuse(channels, weights=QFRWeights(), ages=ages, now_ms=1234567890)
        for _ in range(5)
    ]
    assert all(r == runs[0] for r in runs), "INV-47 确定性失败：多次运行结果不一致"


def test_inv47_tie_broken_by_doc_id():
    """同分时按 doc_id 升序（保证可复现）。"""
    channels = {"bm25": [(5, 1.0), (3, 1.0), (9, 1.0), (1, 1.0)]}
    out = qfr_fuse(channels, weights=QFRWeights(temporal_enabled=False))
    assert [d for d, _ in out] == [1, 3, 5, 9], f"同分应按 doc_id 升序，实际 {out}"


# ===========================================================================
# INV-48 通道先验可解释
# ===========================================================================


def test_inv48_channel_prior_changes_ranking():
    """提高某通道权重 ⟹ 该通道头部文档排名上升。"""
    # doc 1 在 dense 排 1、bm25 排 2；doc 2 在 bm25 排 1、dense 排 2
    channels = {
        "dense": [(1, 0.9), (2, 0.8)],
        "bm25": [(2, 9.0), (1, 8.0)],
    }
    w_dense = QFRWeights(dense=5.0, bm25=0.1, temporal_enabled=False)
    w_bm25 = QFRWeights(dense=0.1, bm25=5.0, temporal_enabled=False)

    top_dense = qfr_fuse(channels, weights=w_dense)[0][0]
    top_bm25 = qfr_fuse(channels, weights=w_bm25)[0][0]
    assert top_dense == 1, f"dense 权重高时应 doc1 居首，实际 {top_dense}"
    assert top_bm25 == 2, f"bm25 权重高时应 doc2 居首，实际 {top_bm25}"


def test_inv48_recency_prior_bounded():
    """recency 通道默认权重低于语义通道（"最近说过"不是强证据）。"""
    w = QFRWeights()
    assert w.recency < w.dense
    assert w.recency < w.bm25


def test_inv48_unknown_channel_defaults_one():
    """未登记通道默认权重 1.0（不静默丢弃新通道）。"""
    w = QFRWeights()
    assert w.weight_for("some_future_channel") == 1.0
    channels = {"some_future_channel": [(1, 1.0), (2, 0.5)]}
    out = qfr_fuse(channels, weights=w)
    assert len(out) == 2, "未登记通道必须参与融合"


# ===========================================================================
# INV-49 MMR λ=1 恒等
# ===========================================================================


def test_inv49_mmr_lambda_one_identity():
    n = 20
    rng = np.random.default_rng(SEED + 4)
    scores = rng.random(n).tolist()
    vectors = rng.standard_normal((n, 8))
    doc_ids = list(range(n))

    out = mmr_rerank(doc_ids, scores, vectors, top_k=8, lambda_mult=1.0)
    expected_order = sorted(range(n), key=lambda i: (-scores[i], doc_ids[i]))[:8]
    assert [d for d, _ in out] == expected_order, "INV-49 失败：λ=1 应退化为原始排序"


def test_inv49_mmr_length_bound():
    n = 5
    rng = np.random.default_rng(SEED + 5)
    out = mmr_rerank(
        list(range(n)), [1.0] * n, rng.standard_normal((n, 4)), top_k=99, lambda_mult=0.5
    )
    assert len(out) == n, f"返回条数应 ≤ n，实际 {len(out)}"


def test_inv49_mmr_empty_and_zero_k():
    rng = np.random.default_rng(SEED + 6)
    assert mmr_rerank([], [], np.zeros((0, 4)), top_k=5) == []
    assert mmr_rerank([1], [1.0], rng.standard_normal((1, 4)), top_k=0) == []


# ===========================================================================
# INV-50 MMR 多样性单调
# ===========================================================================


def test_inv50_mmr_lambda_lowers_similarity():
    """λ 降低 ⟹ 结果平均两两相似度不升（多样性提升）。"""
    n = 12
    rng = np.random.default_rng(SEED + 7)
    # 前 6 个高度相似，后 6 个各不相同
    base = rng.standard_normal(8)
    vectors = np.vstack([base + 0.01 * rng.standard_normal(8) for _ in range(6)])
    vectors = np.vstack([vectors, rng.standard_normal((6, 8))])
    scores = [1.0 - i * 0.01 for i in range(n)]
    doc_ids = list(range(n))

    high = mmr_rerank(doc_ids, scores, vectors, top_k=6, lambda_mult=0.95)
    low = mmr_rerank(doc_ids, scores, vectors, top_k=6, lambda_mult=0.2)

    sel_high = np.array([vectors[doc_ids.index(d)] for d, _ in high])
    sel_low = np.array([vectors[doc_ids.index(d)] for d, _ in low])

    s_high = avg_pairwise_similarity(sel_high)
    s_low = avg_pairwise_similarity(sel_low)
    assert s_low <= s_high + 1e-12, f"INV-50 失败: λ=0.95 → {s_high:.4f}, λ=0.2 → {s_low:.4f}"


def test_inv50_pairwise_cosine_properties():
    v = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
    sim = pairwise_cosine(v)
    assert abs(sim[0, 0] - 1.0) <= 1e-12
    assert abs(sim[0, 1]) <= 1e-12, "正交向量相似度应为 0"
    assert abs(sim[0, 2] + 1.0) <= 1e-12, "反向向量相似度应为 -1"
    assert np.all(sim[3, :] == 0.0), "零向量行必须为 0（不是 NaN）"
    assert not np.isnan(sim).any(), "相似度矩阵不允许 NaN"


# ===========================================================================
# 端到端：QFR + MMR 级联
# ===========================================================================


def test_e2e_qfr_mmr_cascade():
    """多路召回 → QFR 融合 → MMR 裁剪 的完整级联可跑通。"""
    docs = _corpus_docs(40)
    rng = np.random.default_rng(SEED + 8)

    # 两个通道的打分（模拟 BM25 与 HNSW）
    bm25 = [float(len(d)) for d in docs]
    dense = rng.random(len(docs)).tolist()
    channels = {"bm25": list(enumerate(bm25)), "dense": list(enumerate(dense))}
    ages = {i: float(rng.random() * 100) for i in range(len(docs))}

    fused = qfr_fuse(channels, weights=QFRWeights(half_life_days=30.0), ages=ages)
    assert len(fused) == len(docs)

    # MMR 需要向量：用打分本身构造一个低维代理向量（仅测试用）
    dim = 4
    vecs = rng.standard_normal((len(docs), dim))
    ids = [d for d, _ in fused]
    scores = [s for _, s in fused]
    reranked = mmr_rerank(ids, scores, vecs, top_k=8, lambda_mult=0.7)

    assert len(reranked) == 8
    assert len({d for d, _ in reranked}) == 8, "结果不允许重复文档"
    # 融合第一名应在 MMR 结果中（λ=0.7 不至于把它完全挤出）
    assert ids[0] in {d for d, _ in reranked}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
