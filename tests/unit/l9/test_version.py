"""L9 版本与署名检查 (R-X-05): __version__ 与 __author__ 规范。

v0.8.0 追加: **版本号单一事实源** —— 此前 ``ifai/__init__.py`` 写 0.6.1
而 ``pyproject.toml`` 写 0.7.14, CHANGELOG 又写 0.7.14, 三处不一致。
发布出去的版本号必须是唯一可信的, 否则「我们发布了 vX」这句话本身
就没有依据。本用例锁死一致性。
"""
from __future__ import annotations

import re
from pathlib import Path

import ifai


def test_version_format():
    assert re.fullmatch(r"\d+\.\d+\.\d+", ifai.__version__), "版本号应为 semver"


def test_author_attribution():
    assert "晨星" in ifai.__author__, "__author__ 应含署名 晨星 (R-X-05)"


def test_version_matches_pyproject():
    """``__version__`` 必须与 ``pyproject.toml`` 的 version 一致。"""
    repo = Path(__file__).resolve().parents[3]
    text = (repo / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert m, "pyproject.toml 缺少 version 字段"
    assert m.group(1) == ifai.__version__, (
        f"版本号不一致: __init__={ifai.__version__} vs pyproject={m.group(1)}")


def test_version_appears_in_changelog():
    """CHANGELOG 必须记录当前版本 (发布过的版本必有条目)。"""
    repo = Path(__file__).resolve().parents[3]
    text = (repo / "CHANGELOG.md").read_text(encoding="utf-8")
    assert ifai.__version__ in text, (
        f"CHANGELOG.md 未记录当前版本 {ifai.__version__}")
