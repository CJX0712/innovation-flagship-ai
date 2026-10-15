#!/usr/bin/env python
"""分层依赖检查 (Layering Audit) — 验证单向分层 (INV-41)。

零重写 / 集大成 要求十层架构单向依赖: 低层 (算子/内核) 不得反向 import
高层 (服务/应用), 否则破坏可维护性与可复现性。本脚本用 AST 扫描 ``ifai/``
全部源码, 检测是否存在"下层 import 上层"的反向依赖, 并作为 CI 门禁。

退出码: 0 = 零反向依赖; 1 = 发现违规 (CI 应中断)。

用法:
    python scripts/check_layering.py
    python scripts/check_layering.py --root ifai
"""
from __future__ import annotations

import argparse
import ast
import os
import sys

# 十层架构的层序 (数值越大越靠"应用端"); 仅允许 低层→高层 的单向依赖,
# 禁止 高层→低层 (反向)。秩越小越基础。
PACKAGE_RANK = {
    "core": 0,        # L0 算子/内核
    "rng": 1,         # L1 数值基础
    "models": 2,      # L2 模型架构
    "engine": 3,      # L3 推理引擎
    "train": 4,       # L4 训练
    "retrieval": 5,   # L5 检索/记忆
    "agent": 6,       # L6 智能体
    "serve": 7,       # L7 服务/编排
    "eval": 8,        # L8 评测/对齐
    "cli": 9,         # L9 应用/接口
    "sdk": 9,         # L9 应用/接口
}


def rank_of(path: str) -> int:
    """返回文件所属包的层秩; 根目录 (ifai/__init__.py 等) 视为应用层 9。"""
    rel = os.path.relpath(path)
    parts = rel.split(os.sep)
    if len(parts) <= 2:           # ifai/__init__.py 或 ifai/foo.py
        return 9
    pkg = parts[1]
    return PACKAGE_RANK.get(pkg, 9)


def imports_of(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    targets = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "ifai" or node.module.startswith("ifai."):
                targets.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ifai" or alias.name.startswith("ifai."):
                    targets.append(alias.name)
    return targets


def main() -> int:
    ap = argparse.ArgumentParser(description="分层依赖检查 (INV-41)")
    ap.add_argument("--root", default="ifai")
    args = ap.parse_args()
    root = args.root
    if not os.path.isdir(root):
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.root)
    if not os.path.isdir(root):
        print(f"[ERROR] 找不到源码根目录: {args.root}")
        return 1

    violations = []
    scanned = 0
    for base, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(base, fn)
            src_rank = rank_of(path)
            for mod in imports_of(path):
                # 取 ifai.<pkg> 的 pkg
                pkg = mod.split(".")[1] if mod.startswith("ifai.") else ""
                tgt_rank = PACKAGE_RANK.get(pkg, 9)
                if tgt_rank > src_rank:   # 下层 import 上层 = 反向
                    violations.append((os.path.relpath(path, root), mod, src_rank, tgt_rank))
            scanned += 1

    print("=" * 64)
    print("分层依赖检查 · INV-41 (零反向 import)")
    print("=" * 64)
    print(f"扫描文件数: {scanned}")

    if violations:
        print(f"\n发现 {len(violations)} 处反向依赖 (下层→上层, 违规):")
        for src, mod, sr, tr in sorted(violations):
            print(f"  ❌ {src}  (rank {sr}) → import {mod}  (rank {tr})")
        print("\n审计失败: 存在反向依赖, 违反 INV-41。")
        return 1

    print("\n✅ 审计通过: 十层架构零反向依赖 (INV-41)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
