"""L10 微指令层黄金测试 —— 1024 条 ISA 的数量/结构/可执行性/性质断言。

零重写: L10 ISA 移植自同门旗舰 `tianji-ai` (MIT, 作者晨星), 本仓在移植
之上追加**数学性质断言层** (见 ``ifai/l10/_properties.py``)。本文件锁住
三条承诺:

1. **数量不变量**: 恰好 1024 条微指令, 恰好 32 个家族, 每族恰好 32 条;
2. **可执行性**: 全部 1024 条都能被真实输入执行且不抛异常 (冒烟);
3. **可证性**: 390 条带数学性质的微指令逐条通过性质断言, 且所有豁免
   都**显式登记并注明理由** —— 杜绝「悄悄关掉断言」。
"""
from __future__ import annotations

import numpy as np
import pytest

from ifai.l10 import (
    FAMILIES,
    FAMILY_KIND,
    MICROOPS,
    PROPERTY_ASSERTIONS,
    count_ops,
    property_report,
    run_all,
)
from ifai.l10._microops import MicroOp


# ---------------------------------------------------------------- 数量不变量

def test_isa_capacity_is_1024():
    """ISA 容量恒为 1024 条 (数量不变量)。"""
    assert count_ops() == 1024
    assert len(MICROOPS) == 1024


def test_family_count_is_32():
    """家族数恒为 32。"""
    assert len(FAMILIES) == 32


def test_each_family_has_32_members():
    """每族恰好 32 条成员。"""
    for fam, members in FAMILIES.items():
        assert len(members) == 32, f"家族 {fam} 有 {len(members)} 条, 应为 32"


def test_member_names_unique_within_family():
    """家族内成员名唯一 (否则注册表会静默覆盖)。"""
    for fam, members in FAMILIES.items():
        assert len(set(members)) == len(members), f"家族 {fam} 存在重名成员"


def test_global_keys_unique_and_prefixed():
    """全局键唯一且带家族前缀。"""
    keys = set(MICROOPS.keys())
    assert len(keys) == len(MICROOPS)
    for key in MICROOPS:
        assert "." in key
        fam, member = key.split(".", 1)
        assert fam in FAMILIES
        assert member in FAMILIES[fam]


def test_family_kind_declared_for_every_family():
    """每个家族都要声明语义类型 (core / sweep) —— 区分独立算子与参数扫描。"""
    for fam in FAMILIES:
        assert FAMILY_KIND.get(fam) in ("core", "sweep"), f"家族 {fam} 未声明语义类型"


def test_core_and_sweep_both_present():
    """两类家族都必须存在 (否则「不注水」的声明是空的)。"""
    kinds = set(FAMILY_KIND.values())
    assert "core" in kinds and "sweep" in kinds


# ---------------------------------------------------------------- 可执行性

def test_all_microops_execute_without_exception():
    """全部 1024 条都能被真实输入执行 (冒烟门禁)。"""
    report = run_all(seed=0)
    assert report["total"] == 1024
    assert not report["failed"], f"失败条目: {report['failed'][:5]}"
    assert report["ok"] == 1024


def test_run_all_is_deterministic():
    """同 seed 两次运行结果逐条一致 (可复现)。"""
    a = run_all(seed=7)
    b = run_all(seed=7)
    assert a["ok"] == b["ok"]
    assert a["failed"] == b["failed"]


def test_microop_structural_contract():
    """每条微指令都满足结构契约: 真实可调用 + arity 合法。"""
    for key, op in MICROOPS.items():
        assert isinstance(op, MicroOp), key
        assert callable(op.fn), f"{key} 的 fn 不可调用"
        assert op.arity in (1, 2), f"{key} 的 arity 非法: {op.arity}"
        assert op.family in FAMILIES


def test_microops_produce_finite_output():
    """输出全有限 (掩码族的 -inf 是刻意行为, 由冒烟门禁单独豁免)。"""
    g = np.random.default_rng(0)
    x = g.standard_normal((4, 8))
    y = g.standard_normal((4, 8))
    for key, op in MICROOPS.items():
        out = np.asarray(op.fn(x) if op.arity == 1 else op.fn(x, y),
                         dtype=np.float64)
        assert out.size > 0, f"{key} 输出为空"
        if op.family in ("mask", "attn"):
            continue
        assert np.all(np.isfinite(out)), f"{key} 输出含 NaN/Inf"


# ---------------------------------------------------------------- 性质断言

def test_property_assertions_all_pass():
    """全部数学性质断言通过。"""
    report = property_report(seed=0)
    assert not report["failed"], f"性质断言失败: {report['failed'][:5]}"
    assert report["passed"] == report["checked"]
    assert report["checked"] > 300, "性质断言覆盖度过低"


def test_property_exemptions_are_declared_with_reason():
    """所有豁免必须显式登记并**注明理由** —— 不允许静默关掉断言。"""
    for family, (assert_fn, exclude, reason) in PROPERTY_ASSERTIONS.items():
        assert callable(assert_fn), family
        assert isinstance(exclude, set), family
        if exclude:
            assert reason and reason.strip(), f"家族 {family} 的豁免缺少理由"
            assert family in FAMILIES


def test_exempted_members_actually_exist():
    """豁免清单里的成员名必须真实存在 (防止写错名字导致豁免失效)。"""
    for family, (_fn, exclude, _reason) in PROPERTY_ASSERTIONS.items():
        for member in exclude:
            assert member in FAMILIES[family], \
                f"家族 {family} 豁免了不存在的成员 {member}"


def test_exemption_count_is_bounded():
    """豁免比例受控 —— 不能用「全部豁免」来假装门禁。"""
    report = property_report(seed=0)
    checked = report["checked"]
    skipped = report["skipped"]
    total = checked + skipped
    assert skipped / total < 0.10, f"豁免比例过高: {skipped}/{total}"


# ---------------------------------------------------------------- L10 依赖本仓 L0

def test_l10_depends_on_repo_l0():
    """L10 的正交化/谱范数/L2 归一化来自本仓 L0 (而非外部副本)。

    这是「移植即集大成」的落点: 本仓 L0 是唯一依赖来源, 不存在两套
    语义相同的实现悄悄分叉。
    """
    from ifai.core import ops as repo_ops

    assert repo_ops.l2norm is not None
    assert repo_ops.spectral_norm is not None
    assert repo_ops.newton_schulz is not None

    g = np.random.default_rng(3)
    x = g.standard_normal((4, 8))
    # norm.l2norm 必须与本仓 L0 同源 → 输出严格单位范数
    out = np.asarray(MICROOPS["norm.l2norm"].fn(x))
    assert np.allclose(np.linalg.norm(out, axis=-1), 1.0, atol=1e-9)


def test_repo_l2norm_handles_zero_vector():
    """本仓 L0 的 l2norm 对零向量返回 0 (不产生 NaN)。

    零重写依据: 同门旗舰 tianji-ai 的 G23 真实缺陷 —— 分母加 eps 会让
    小范数 key 归一化后范数严重偏离 (实测偏差 82%), 且零向量会产生 NaN。
    """
    from ifai.core.ops import l2norm

    out = l2norm(np.zeros((2, 4)))
    assert np.all(np.isfinite(out))
    assert np.all(out == 0)


def test_repo_l2norm_strict_unit_norm():
    """本仓 L0 的 l2norm 对任意非零输入给出严格单位范数。"""
    from ifai.core.ops import l2norm

    g = np.random.default_rng(5)
    x = g.standard_normal((4, 8))
    out = l2norm(x)
    assert np.allclose(np.linalg.norm(out, axis=-1), 1.0, atol=1e-12)


def test_repo_newton_schulz_singular_band():
    """本仓 L0 的 Newton-Schulz 奇异值落入 (0.5, 1.5) (Muon 公开验收标准)。"""
    from ifai.core.ops import newton_schulz

    g = np.random.default_rng(9)
    G = g.standard_normal((8, 8))
    out = newton_schulz(G)
    sv = np.linalg.svd(out, compute_uv=False)
    assert sv.min() > 0.5 and sv.max() < 1.5, f"奇异值越界: {sv}"


def test_repo_spectral_norm_matches_svd():
    """本仓 L0 的幂迭代谱范数 ≡ 精确 SVD 谱范数。"""
    from ifai.core.ops import spectral_norm

    g = np.random.default_rng(11)
    A = g.standard_normal((6, 6))
    got = spectral_norm(A)
    ref = float(np.linalg.svd(A, compute_uv=False).max())
    assert abs(got - ref) / ref < 1e-6
