"""L9 版本与署名检查 (R-X-05): __version__ 与 __author__ 规范。"""
from __future__ import annotations

import re

import ifai


def test_version_format():
    assert re.fullmatch(r"\d+\.\d+\.\d+", ifai.__version__), "版本号应为 semver"


def test_author_attribution():
    assert "晨星" in ifai.__author__, "__author__ 应含署名 晨星 (R-X-05)"
