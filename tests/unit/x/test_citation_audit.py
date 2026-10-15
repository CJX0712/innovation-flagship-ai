"""INV-42 零重写审计 (十层覆盖 + >=25 出处)。

调用 ``scripts/audit_no_reinvention.py`` 作为子进程, 断言其退出码为 0 —
即十层架构均有"站在世界顶级成果之上"的引用, 且全代码库不同出处标记数 >= 25。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "audit_no_reinvention.py"


def test_no_reinvention_audit_passes():
    assert SCRIPT.exists(), f"找不到审计脚本: {SCRIPT}"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"INV-42 零重写审计失败 (exit={proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    )
