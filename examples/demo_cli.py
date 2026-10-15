#!/usr/bin/env python
"""演示 CLI (`ifai`) 的常用子命令。

通过子进程调用 ``python -m ifai.cli <subcommand>`` 展示十一个子命令中的
代表性用法 (version / doctor / tokenize / search / demo)。

运行:
    python examples/demo_cli.py
"""
from __future__ import annotations

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(*args: str) -> int:
    print(f"\n$ python -m ifai.cli {' '.join(args)}")
    proc = subprocess.run(
        [sys.executable, "-m", "ifai.cli", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    out = (proc.stdout + proc.stderr).strip()
    print(out if out else "(无输出)")
    return proc.returncode


def main() -> None:
    print("=" * 60)
    print("ifai CLI 演示")
    print("=" * 60)
    rc = 0
    rc |= run("version")
    rc |= run("doctor")
    rc |= run("tokenize", "--text", "人工智能 模型")
    rc |= run("search", "--query", "注意力")
    rc |= run("demo", "--greedy", "--max-new", "8")
    print("\n" + "=" * 60)
    if rc == 0:
        print("CLI 演示完成: 所有子命令正常退出。")
    else:
        print(f"CLI 演示存在非零退出 (rc={rc})。")


if __name__ == "__main__":
    main()
