#!/usr/bin/env python
"""L10 微指令门禁 —— 1024 条可执行微指令逐条执行 + 性质断言。

零重写 · 集大成: L10 ISA 移植自同门旗舰 `tianji-ai` (MIT, 作者晨星),
本脚本在其冒烟断言之上追加**数学性质断言** (见 ``ifai/l10/_properties.py``)。

门禁判据 (三层, 逐层加强):
    1. 冒烟: 1024 条逐条用确定性输入执行, 不抛异常;
    2. 有限性: 输出无 NaN/±Inf (掩码族的 -inf 属刻意行为, 显式豁免);
    3. 性质: 390 条带数学不变量 (softmax 归一性 / L2 严格单位范数 /
       KL 非负 / Flash ≡ 朴素 / NS 奇异值带 / Top-K 掩码个数 / RoPE 保范数 …),
       26 条因语义特殊显式豁免并**逐条注明理由** —— 不放宽阈值蒙混过关。

退出码: 0 = 全绿; 1 = 存在失败项。

用法:
    python scripts/check_l10.py
    python scripts/check_l10.py --verbose
"""
from __future__ import annotations

import argparse
import os
import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ifai.l10 import FAMILIES, FAMILY_KIND, MICROOPS, run_all  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="L10 微指令门禁 (1024 条)")
    ap.add_argument("--verbose", action="store_true", help="打印每个家族的明细")
    ap.add_argument("--seed", type=int, default=0, help="采样种子 (固定则可复现)")
    args = ap.parse_args()

    print("=" * 68)
    print("L10 微指令门禁 · 1024 条可执行微指令 (32 家族 × 32 成员)")
    print("=" * 68)
    print(f"ISA 规模: {len(MICROOPS)} 条 / {len(FAMILIES)} 家族")
    core = sum(1 for v in FAMILY_KIND.values() if v == "core")
    sweep = sum(1 for v in FAMILY_KIND.values() if v == "sweep")
    print(f"家族语义: core(成员语义独立)={core} · sweep(参数扫描)={sweep}")
    print("-" * 68)

    report = run_all(seed=args.seed)
    props = report["properties"]

    if args.verbose:
        for fam in sorted(FAMILIES):
            kind = FAMILY_KIND.get(fam, "?")
            print(f"  {fam:<8} [{kind:<5}] {len(FAMILIES[fam])} 条")

    print(f"冒烟+有限性: {report['ok']}/{report['total']} 通过")
    print(f"数学性质断言: {props['passed']}/{props['checked']} 通过 "
          f"(豁免 {props['skipped']} 条, 已逐条注明理由)")

    if props["exemptions"]:
        print("-" * 68)
        print("显式豁免清单 (语义特殊, 不适用家族共性断言):")
        for fam, (members, reason) in sorted(props["exemptions"].items()):
            print(f"  {fam}: {len(members)} 条 — {reason}")

    failures = list(report["failed"])
    if failures:
        print("-" * 68)
        print(f"失败明细 ({len(failures)} 条):")
        for name, detail in failures[:50]:
            print(f"  ❌ {name}: {detail}")
        if len(failures) > 50:
            print(f"  ... 其余 {len(failures) - 50} 条省略")

    print("=" * 68)
    ok = not failures and report["ok"] == report["total"]
    print(f"结论: {'ALL GREEN ✅ 1024/1024' if ok else '存在失败 ❌'}")
    print("=" * 68)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
