"""INV-41 分层依赖检查 (零反向 import)。

直接调用 ``scripts/check_layering.py`` 作为子进程, 断言其退出码为 0 —
即十层架构 (L0–L9) 不存在任何"下层 import 上层"的反向依赖。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "check_layering.py"


def test_layering_zero_reverse_imports():
    assert SCRIPT.exists(), f"找不到审计脚本: {SCRIPT}"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"INV-41 分层检查失败 (exit={proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    )
