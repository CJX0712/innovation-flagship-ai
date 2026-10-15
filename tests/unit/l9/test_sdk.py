"""L9 SDK 公开 API 检查 (R-L9-02): 三大高阶 API 均有 docstring。"""
from __future__ import annotations

import inspect

from ifai.sdk import IFAI


def test_public_api_has_docstrings():
    for name in ("chat", "search", "run", "__init__"):
        method = getattr(IFAI, name)
        doc = inspect.getdoc(method)
        assert doc and doc.strip(), f"IFAI.{name} 缺少 docstring (R-L9-02)"


def test_sdk_methods_callable():
    for name in ("chat", "search", "run"):
        assert callable(getattr(IFAI, name)), f"IFAI.{name} 必须是可调用方法"
