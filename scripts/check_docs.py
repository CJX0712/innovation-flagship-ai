#!/usr/bin/env python
"""文档检查 (Docs Check) — R-X-03 门禁。

三项校验:
1. 必备文件存在 (README/CITATION/NOTICE/CHANGELOG/CONTRIBUTING/LICENSE +
    docs/ 下核心文档 + docs/design/ 九层 DESIGN.md)。
2. Markdown 内部相对链接可解析 (不存在的本地链接视为损坏)。
3. README 含关键章节 (安装 / 快速开始 / 架构 / 许可 / 引用)。

退出码: 0 = 全部通过; 1 = 存在缺失/损坏 (CI 应中断)。

用法:
    python scripts/check_docs.py
    python scripts/check_docs.py --root .
"""
from __future__ import annotations

import argparse
import os
import re
import sys

# Windows 默认控制台编码 (cp1252/cp936) 无法编码中文输出, 统一强制 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "CITATION.md",
    "NOTICE",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "docs/ARCHITECTURE.md",
    "docs/PRD.md",
    "docs/PRIOR_ART.md",
    "docs/TASKS.md",
    "docs/API.md",
]

# 九层 DESIGN.md (零重写原则要求逐层文档化)
DESIGN_FILES = [
    "docs/design/L0_core.md",
    "docs/design/L1_numeric.md",
    "docs/design/L2_models.md",
    "docs/design/L3_engine.md",
    "docs/design/L4_train.md",
    "docs/design/L5_retrieval.md",
    "docs/design/L6_agent.md",
    "docs/design/L7_serve.md",
    "docs/design/L8_eval.md",
]

# README 关键章节 (匹配中文或英文, 不区分大小写)
README_SECTIONS = [
    (r"安装|install", "安装/Install"),
    (r"快速开始|快速上手|quick ?start|quickstart", "快速开始/Quickstart"),
    (r"架构|architecture", "架构/Architecture"),
    (r"许可|license", "许可/License"),
    (r"引用|致谢|citation|致已知的", "引用/Citation"),
]

_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)


def find_markdown(root: str) -> list[str]:
    out = []
    for base, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith((".md", ".markdown")):
                out.append(os.path.join(base, fn))
    return sorted(out)


def check_links(path: str, problems: list[str]) -> None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except (OSError, UnicodeDecodeError):
        problems.append(f"无法读取: {path}")
        return
    base_dir = os.path.dirname(path)
    for m in _LINK_RE.findall(text):
        target = m.strip()
        if not target:
            continue
        if target.startswith(("http://", "https://", "mailto:", "tel:")):
            continue
        # 去掉锚点
        path_part = target.split("#", 1)[0]
        if not path_part:
            continue  # 纯锚点
        if path_part.startswith("/"):
            continue  # 站点绝对路径, 不检查
        resolved = os.path.normpath(os.path.join(base_dir, path_part))
        if not os.path.exists(resolved):
            problems.append(f"{os.path.relpath(path)} → 损坏链接: {target}")


def check_readme_sections(path: str, problems: list[str]) -> None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except (OSError, UnicodeDecodeError):
        problems.append("无法读取 README.md")
        return
    headings = " ".join(_HEADING_RE.findall(text))
    headings_lower = headings.lower()
    missing = []
    for pat, label in README_SECTIONS:
        if not re.search(pat, headings_lower, re.IGNORECASE):
            missing.append(label)
    if missing:
        problems.append("README 缺少关键章节: " + ", ".join(missing))


def main() -> int:
    ap = argparse.ArgumentParser(description="文档检查 (R-X-03)")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = args.root
    if not os.path.isdir(root):
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.root)
    root = os.path.abspath(root)

    print("=" * 64)
    print("文档检查 · R-X-03 (必备文件 / 链接 / 章节)")
    print("=" * 64)

    problems: list[str] = []

    # 1) 必备文件
    print("\n[1] 必备文件")
    for f in REQUIRED_FILES + DESIGN_FILES:
        p = os.path.join(root, f)
        ok = os.path.isfile(p)
        print(f"  {'✔' if ok else '✘'} {f}")
        if not ok:
            problems.append(f"缺少必备文件: {f}")

    # 2) 内部链接 (仅扫描 repo 内 markdown)
    print("\n[2] Markdown 内部链接")
    md_files = find_markdown(root)
    link_problems: list[str] = []
    for p in md_files:
        check_links(p, link_problems)
    if link_problems:
        for lp in link_problems:
            print(f"  ❌ {lp}")
        problems.extend(link_problems)
    else:
        print(f"  ✔ 扫描 {len(md_files)} 个 markdown 文件, 内部链接全部有效")

    # 3) README 章节
    print("\n[3] README 关键章节")
    readme = os.path.join(root, "README.md")
    section_problems: list[str] = []
    check_readme_sections(readme, section_problems)
    if section_problems:
        for sp in section_problems:
            print(f"  ❌ {sp}")
        problems.extend(section_problems)
    else:
        print("  ✔ README 含全部关键章节")

    print("\n" + "=" * 64)
    if problems:
        print(f"审计失败: 发现 {len(problems)} 处文档问题。")
        return 1
    print("审计通过: 文档完整、链接有效、章节齐全 (R-X-03)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
