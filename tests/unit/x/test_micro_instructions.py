"""微指令级操作分解生成器校验 (项目核心关注点: 1024 项)。

调用 ``scripts/gen_micro_instructions.py`` 作为子进程, 断言其退出码为 0 且
生成的微指令总数 >= 1024 (功能级 + 步骤级)。

另外校验不变量覆盖完整性: PRD 定义的 42 条 INV (INV-01..INV-42) 必须**全部**
出现在生成的分解中 —— 其中 INV-41 (分层依赖单向) 与 INV-42 (零重写审计) 是
横切审计, 其实现落在 ``scripts/``, 由生成器的 X 横切层登记。

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

# PRD 定义的全部可验证不变量
EXPECTED_INVS = set(range(1, 43))


def _run_generator() -> subprocess.CompletedProcess:
    assert SCRIPT.exists(), f"找不到生成脚本: {SCRIPT}"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return proc


def _covered_invs(stdout: str) -> set:
    m = re.search(r"覆盖 INV:\s*\[([0-9, ]+)\]", stdout)
    assert m, f"未能解析覆盖的 INV 列表:\n{stdout}"
    return {int(x) for x in m.group(1).split(",") if x.strip()}


def test_micro_instructions_generated():
    proc = _run_generator()
    m = re.search(r"总计微指令:\s*(\d+)", proc.stdout)
    assert m, f"未能解析微指令总数:\n{proc.stdout}"
    total = int(m.group(1))
    assert total >= 1024, f"微指令总数 {total} < 1024 目标"
    assert DOC.exists(), "生成的 docs/MICRO_INSTRUCTIONS.md 不存在"


def test_micro_instructions_cover_all_42_invariants():
    """42/42 覆盖: 必须包含 INV-41 / INV-42 等全部不变量。"""
    covered = _covered_invs(_run_generator().stdout)
    missing = sorted(EXPECTED_INVS - covered)
    assert not missing, f"微指令分解缺失不变量: {missing}"
    assert len(covered) >= len(EXPECTED_INVS), \
        f"覆盖 {len(covered)} 条 < 预期 {len(EXPECTED_INVS)} 条"


def test_micro_instructions_doc_mentions_cross_cutting_layer(tmp_path=None):
    """生成的文档必须含 X 横切层小节 (INV-41/42 的落点)。"""
    _run_generator()
    text = DOC.read_text(encoding="utf-8")
    assert "X 横切审计" in text, "文档中缺少 X 横切审计小节"
    assert "| X |" in text, "分层汇总表缺少 X 层行"

