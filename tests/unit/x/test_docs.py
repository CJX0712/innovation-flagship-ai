"""R-X-03 文档检查 (必备文件 / 内部链接 / 关键章节)。

调用 ``scripts/check_docs.py`` 作为子进程, 断言其退出码为 0 —
即必备文档齐全、Markdown 内部链接有效、README 含关键章节。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "check_docs.py"


def test_docs_check_passes():
    assert SCRIPT.exists(), f"找不到检查脚本: {SCRIPT}"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"R-X-03 文档检查失败 (exit={proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    )
