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


def test_python_m_entrypoint_runs(capsys):
    """`python -m ifai.cli version` 可运行 (覆盖 cli/__main__.py 入口)。"""
    import subprocess
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[3]
    r = subprocess.run([sys.executable, "-m", "ifai.cli", "version"],
                       capture_output=True, text=True, cwd=str(repo))
    assert r.returncode == 0
    assert "ifai" in r.stdout and "晨星" in r.stdout
