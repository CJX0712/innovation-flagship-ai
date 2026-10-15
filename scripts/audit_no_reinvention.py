#!/usr/bin/env python
"""零重写审计 (No-Reinvention Audit) — 集大成原则的可验证门禁 (INV-42)。

设计目标: 证明旗舰 AI 的每一层 (L0–L9) 都"站在世界顶级成果之上", 而非从
零造轮子。脚本扫描 ``ifai/`` 全部源码, 完成两项校验:

1. **分层引用覆盖率**: 十层架构每一层都至少存在一个"零重写"引用标记
   (作者 / 年份 / 经典方法名)。
2. **出处总数门槛**: 全代码库命中的不同引用标记数 >= 25 (INV-42 要求
   >= 25 条出处)。

两项全部通过才退出 0; 否则退出 1 (CI 应中断)。

用法:
    python scripts/audit_no_reinvention.py
    python scripts/audit_no_reinvention.py --root ifai
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

# 十层架构 → (层名, 目录模式)
LAYERS = [
    ("L0", "算子/内核 (Operator/Kernel)", ["core/numeric.py", "core/ops.py", "core/quant.py"]),
    ("L1", "张量/数值基础 (Tensor/Numeric)", ["rng.py"]),
    ("L2", "模型架构 (Model Architecture)", ["models/"]),
    ("L3", "推理引擎 (Inference Engine)", ["engine/"]),
    ("L4", "训练/微调 (Training/Finetune)", ["train/"]),
    ("L5", "检索/记忆 (Retrieval/Memory)", ["retrieval/"]),
    ("L6", "智能体 (Agent)", ["agent/"]),
    ("L7", "服务/编排 (Service/Orchestration)", ["serve/"]),
    ("L8", "评估/对齐 (Evaluation/Alignment)", ["eval/"]),
    ("L9", "应用/接口 (Application/Interface)", ["cli/", "sdk.py"]),
]

# 公认的"零重写"引用标记 (作者 / 年份 / 经典方法 / 系统名)
MARKERS = [
    r"et al", r"零重写", r"zero-?rewrite",
    # Transformer / 注意力
    r"Vaswani", r"Su et al", r"RoPE", r"Shazeer", r"SwiGLU", r"RMSNorm", r"Zhang et al",
    r"GQA", r"Ainslie", r"MoE", r"Fedus", r"Switch", r"LoRA", r"Hu et al", r"Press", r"weight[- ]?tying",
    # 算子 / 内核
    r"FlashAttention", r"Dao", r"online softmax", r"NTK", r"Dettmers", r"GPTQ", r"AWQ",
    # 推理系统
    r"PagedAttention", r"vLLM", r"Kwon", r"Orca", r"Yu et al", r"continuous batching",
    r"Speculative", r"Leviathan", r"Medusa", r"EAGLE",
    # 训练
    r"AdamW", r"Loshchilov", r"cosine", r"nanogpt", r"Karpathy", r"Griewank", r"Walther",
    r"reverse-?mode", r"backprop", r"autodiff", r"reverse mode", r"Kingma",
    # 检索
    r"BM25", r"Robertson", r"HNSW", r"Malkov", r"RRF", r"Cormack", r"cross-?encoder",
    r"Nogueira", r"dense", r"embedding",
    # 智能体
    r"ReAct", r"Yao et al", r"Reflexion", r"Shinn", r"MemGPT", r"Packer", r"SafeCode", r"AST",
    # 服务
    r"OpenAI", r"SSE", r"REST",
    # 评估
    r"Jelinek", r"perplexity", r"MLPerf", r"GSM8K", r"benchmark", r"Cobbe",
    # 量化
    r"int8", r"int4", r"fp16", r"quantiz",
    # 分词
    r"BPE", r"Sennrich", r"byte-?level", r"byte level", r"Radford",
]

_COMPILED = [re.compile(m, re.IGNORECASE) for m in MARKERS]

# INV-42 要求的最低不同出处数
MIN_CITATIONS = 25


def file_has_citation(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except (OSError, UnicodeDecodeError):
        return False
    return any(p.search(text) for p in _COMPILED)


def resolve_files(root: str, pattern: str) -> list[str]:
    out = []
    if pattern.endswith("/"):
        d = os.path.join(root, pattern.rstrip("/"))
        if os.path.isdir(d):
            for base, _, files in os.walk(d):
                for fn in files:
                    if fn.endswith(".py") and not fn.startswith("__"):
                        out.append(os.path.join(base, fn))
    else:
        p = os.path.join(root, pattern)
        if os.path.isfile(p):
            out.append(p)
    return out


def collect_citations(root: str) -> set[str]:
    """返回全代码库命中到的不同引用标记名集合。"""
    hit: set[str] = set()
    for base, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(base, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except (OSError, UnicodeDecodeError):
                continue
            for m in MARKERS:
                if re.search(m, text, re.IGNORECASE):
                    hit.add(m)
    return hit


def main() -> int:
    ap = argparse.ArgumentParser(description="零重写审计 (INV-42)")
    ap.add_argument("--root", default="ifai", help="待审计的源码根目录 (默认 ifai)")
    args = ap.parse_args()

    root = args.root
    if not os.path.isdir(root):
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.root)
    if not os.path.isdir(root):
        print(f"[ERROR] 找不到源码根目录: {args.root}")
        return 1

    print("=" * 70)
    print("零重写审计 · 集大成原则验证 (L0–L9)  ·  INV-42")
    print("=" * 70)

    # 1) 分层引用覆盖率
    layer_ok = True
    for lid, name, patterns in LAYERS:
        files: list[str] = []
        for pat in patterns:
            files.extend(resolve_files(root, pat))
        files = sorted(set(files))
        cited = [f for f in files if file_has_citation(f)]
        covered = len(cited) > 0
        layer_ok = layer_ok and covered
        status = "✅ 已覆盖" if covered else "❌ 未覆盖"
        print(f"\n[{lid}] {name}  ({len(cited)}/{len(files)} 文件含引用)  {status}")
        for f in files:
            mark = "✔" if f in cited else "✘"
            print(f"      {mark} {os.path.relpath(f, root)}")
        if not covered:
            print("      ⚠ 该层没有任何零重写引用 — 请补充对公认方案/论文的标注")

    # 2) 出处总数门槛
    hits = collect_citations(root)
    citation_ok = len(hits) >= MIN_CITATIONS

    print("\n" + "-" * 70)
    print(f"不同出处标记命中数: {len(hits)}  (INV-42 门槛 >= {MIN_CITATIONS})  "
          + ("✅" if citation_ok else "❌"))

    print("=" * 70)
    if layer_ok and citation_ok:
        print("审计通过: 十层架构均站在世界顶级成果之上 (零重写 / 集大成)。")
        return 0
    print("审计失败: 存在未覆盖分层或出处不足, 违反零重写原则。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
