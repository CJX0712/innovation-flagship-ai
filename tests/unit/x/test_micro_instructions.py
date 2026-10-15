"""微指令级操作分解生成器校验 (项目核心关注点: 1024 项)。

调用 ``scripts/gen_micro_instructions.py`` 作为子进程, 断言其退出码为 0 且
生成的微指令总数 >= 1024 (功能级 + 步骤级)。

用法: 该测试同时验证 ``docs/MICRO_INSTRUCTIONS.md`` 可由源码复现。
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "gen_micro_instructions.py"
DOC = REPO / "docs" / "MICRO_INSTRUCTIONS.md"


def test_micro_instructions_generated():
    assert SCRIPT.exists(), f"找不到生成脚本: {SCRIPT}"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    m = re.search(r"总计微指令:\s*(\d+)", proc.stdout)
    assert m, f"未能解析微指令总数:\n{proc.stdout}"
    total = int(m.group(1))
    assert total >= 1024, f"微指令总数 {total} < 1024 目标"
    assert DOC.exists(), "生成的 docs/MICRO_INSTRUCTIONS.md 不存在"
