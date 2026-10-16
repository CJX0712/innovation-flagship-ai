"""INV 测试可追溯性硬门禁: PRD 的 42 条不变量, 每条至少被 1 个测试文件引用。

依据 ``docs/PRD.md`` 的映射原则 (「每条不变量至少一个独立测试函数」) 的可执行化:
静态扫描 ``tests/`` 下全部测试源码, 提取 ``INV-XX`` 标记 (编号按十进制整数归一,
``INV-07`` 与 ``INV-7`` 视为同一条), 断言 INV-01..INV-42 全部有落点。

这与 ``docs/MICRO_INSTRUCTIONS.md`` 的「分解覆盖 42/42」门禁互补:
- 分解覆盖 = 源码实现侧每条 INV 有落点;
- 本门禁 = 测试验证侧每条 INV 有落点。
两者合起来才是完整的「规范 → 实现 → 验证」闭环。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parents[2]
INV_RE = re.compile(r"INV-(\d{1,2})\b")
EXPECTED = set(range(1, 43))


def _collect_inv_ids() -> dict[int, list[str]]:
    """扫描 tests/**.py, 返回 {inv_id: [引用文件相对路径...]}。"""
    found: dict[int, list[str]] = {}
    for path in sorted(TESTS_DIR.rglob("*.py")):
        if path.name == "test_inv_traceability.py":
            continue  # 本文件自身是门禁, 不作为 INV 落点
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = str(path.relative_to(TESTS_DIR)).replace("\\", "/")
        for m in INV_RE.finditer(text):
            inv = int(m.group(1))
            if 1 <= inv <= 42:
                found.setdefault(inv, []).append(rel)
    return found


def test_every_inv_has_at_least_one_test_reference():
    """42/42: 每条 INV 至少被一个测试文件引用 (缺失即失败并列出缺口)。"""
    found = _collect_inv_ids()
    missing = sorted(EXPECTED - set(found))
    assert not missing, (
        f"以下 INV 在 tests/ 下无任何测试引用 (须为每条不变量补独立测试): {missing}; "
        f"当前覆盖 {len(found)}/42")


def test_traceability_map_is_well_formed():
    """门禁自身健全性: 映射非空、无越界编号、引用路径真实存在。"""
    found = _collect_inv_ids()
    assert found, "未扫描到任何 INV 引用, 扫描逻辑失效"
    for inv, files in found.items():
        assert 1 <= inv <= 42
        for f in files:
            assert (TESTS_DIR / f).exists(), f"引用文件不存在: {f}"


@pytest.mark.parametrize("inv", sorted(EXPECTED))
def test_inv_reference_is_not_stale(inv: int):
    """逐条参数化: 每条 INV 的引用文件当前真实存在 (CI 报告可精确到条)。"""
    found = _collect_inv_ids()
    assert inv in found, f"INV-{inv:02d} 无测试引用"
