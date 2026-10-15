#!/usr/bin/env bash
# ifai 冒烟测试 (Linux/macOS) — 跑通核心 CLI 子命令, 验证端到端可运行。
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
echo "== ifai smoke (sh) =="
"$PY" -m ifai.cli version
"$PY" -m ifai.cli doctor
"$PY" -m ifai.cli tokenize --text "人工智能 模型"
"$PY" -m ifai.cli search --query "注意力"
"$PY" -m ifai.cli demo --greedy --max-new 8
"$PY" -m ifai.cli eval
echo "== smoke OK =="
