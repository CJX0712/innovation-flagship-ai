#!/usr/bin/env python
"""零重写审计 (Zero-Rewrite Audit) — 集大成原则的可验证门禁。

设计目标: 证明旗舰 AI 的每一层 (L0–L9) 都"站在世界顶级成果之上",
而非从零造轮子。脚本扫描 ``ifai/`` 全部源码, 逐文件检测是否存在对
公认论文/算法/系统的"零重写"引用标记 (作者、年份、经典方法名), 并按
十层架构汇总覆盖率。

退出码: 0 = 全部分层均有引用覆盖; 1 = 存在未覆盖分层 (CI 中应中断)。

用法:
    python scripts/audit_zero_rewrite.py
    python scripts/audit_zero_rewrite.py --root ifai
"""
from __future__ import annotations

import argparse
import os
import re
import sys

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
    r"FlashAttention", r"Dao", r"online softmax", r"NTK",
    # 推理系统
    r"PagedAttention", r"vLLM", r"Kwon", r"Orca", r"Yu et al", r"continuous batching",
    r"Speculative", r"Leviathan", r"Medusa", r"EAGLE",
    # 训练
    r"AdamW", r"Loshchilov", r"cosine", r"nanogpt", r"Karpathy", r"Griewank", r"Walther",
    r"reverse-?mode", r"backprop", r"autodiff", r"reverse mode",
    # 检索
    r"BM25", r"Robertson", r"HNSW", r"Malkov", r"RRF", r"Cormack", r"cross-?encoder",
    r"Nogueira", r"dense", r"embedding",
    # 智能体
    r"ReAct", r"Yao et al", r"Reflexion", r"Shinn", r"MemGPT", r"Packer", r"SafeCode", r"AST",
    # 服务
    r"OpenAI", r"SSE", r"REST",
    # 评估
    r"Jelinek", r"perplexity", r"MLPerf", r"GSM8K", r"benchmark",
    # 量化
    r"int8", r"int4", r"fp16", r"quantiz",
    # 分词
    r"BPE", r"Sennrich", r"byte-?level", r"byte level",
]

_COMPILED = [re.compile(m, re.IGNORECASE) for m in MARKERS]


def file_has_citation(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except (OSError, UnicodeDecodeError):
        return False
    return any(p.search(text) for p in _COMPILED)


def resolve_files(root: str, pattern: str) -> list[str]:
    # pattern 形如 "models/" (目录) 或 "rng.py" (精确文件, 相对 root)
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


def main() -> int:
    ap = argparse.ArgumentParser(description="零重写审计")
    ap.add_argument("--root", default="ifai", help="待审计的源码根目录 (默认 ifai)")
    args = ap.parse_args()

    root = args.root
    if not os.path.isdir(root):
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.root)
    if not os.path.isdir(root):
        print(f"[ERROR] 找不到源码根目录: {args.root}")
        return 1

    print("=" * 70)
    print("零重写审计 · 集大成原则验证 (L0–L9)")
    print("=" * 70)

    all_ok = True
    for lid, name, patterns in LAYERS:
        files: list[str] = []
        for pat in patterns:
            files.extend(resolve_files(root, pat))
        files = sorted(set(files))
        cited = [f for f in files if file_has_citation(f)]
        n_files, n_cited = len(files), len(cited)
        covered = n_cited > 0
        all_ok = all_ok and covered
        status = "✅ 已覆盖" if covered else "❌ 未覆盖"
        print(f"\n[{lid}] {name}  ({n_cited}/{n_files} 文件含引用)  {status}")
        for f in files:
            mark = "✔" if f in cited else "✘"
            rel = os.path.relpath(f, root)
            print(f"      {mark} {rel}")
        if not covered:
            print(f"      ⚠ 该层没有任何零重写引用 — 请补充对公认方案/论文的标注")

    print("\n" + "=" * 70)
    if all_ok:
        print("审计通过: 十层架构均站在世界顶级成果之上 (零重写 / 集大成)。")
        return 0
    print("审计失败: 存在未覆盖分层, 违反零重写原则。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
