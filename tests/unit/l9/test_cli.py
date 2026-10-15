"""L9 CLI 检查 (R-L9-01): 子命令齐全, version 子命令可运行。"""
from __future__ import annotations

import contextlib
import io
import sys

import ifai.cli as cli


EXPECTED_COMMANDS = [
    "train", "demo", "generate", "chat", "serve", "eval",
    "search", "agent", "tokenize", "version", "doctor", "selftest",
]


def test_all_subcommands_defined():
    for name in EXPECTED_COMMANDS:
        assert hasattr(cli, f"cmd_{name}"), f"缺少命令函数 cmd_{name}"
        assert callable(getattr(cli, f"cmd_{name}")), f"cmd_{name} 不可调用"


def test_version_subcommand_runs(capsys):
    cli.cmd_version(None)
    out = capsys.readouterr().out
    assert "ifai" in out and "晨星" in out, "version 应输出署名信息"
