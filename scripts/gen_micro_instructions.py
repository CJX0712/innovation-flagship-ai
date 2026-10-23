#!/usr/bin/env python
"""生成「微指令级操作分解」(Micro-Instruction Decomposition)。

零重写 · 集大成原则要求系统可被完整验证。本脚本对 ``ifai/`` 全部源码做 AST
静态扫描, 并对 ``scripts/`` 下的横切审计脚本做同样扫描 (**X 横切层**), 把
每一层 (L0–L10 与 X) 的操作分解为「微指令级」条目:

- **功能级微指令 (MI)**: 每个函数 / 方法 = 一条操作, 含层、模块、签名、
  用途 (docstring 首行)、引用的 INV 编号、复用的世界顶级来源标记。
- **步骤级微指令**: 每个函数体内的顶层语句 = 一条更细的操作步骤, 进一步
  把算法拆解到可被逐条核验的粒度。

与 ``ifai/l10/`` 可执行 ISA 的关系 (v0.8.0)
--------------------------------------------
两者**互补, 不重复**:

- 本文件产出**源码派生文档** —— 描述「代码写了什么」(每个函数被拆成多少条
  语句、引用了哪些 INV);
- ``ifai/l10/`` 产出**运行时可执行注册表** —— 描述「代码能做什么, 且每条
  都被真实执行并断言了什么」(1024 条微指令 + 390 条数学性质断言)。

即: 本文件是**静态视图**, L10 是**动态视图**。

生成结果写入 ``docs/MICRO_INSTRUCTIONS.md`` (纯源码派生, 可复现、可验证)。

X 横切层是 INV-41 (分层依赖单向) 与 INV-42 (零重写审计) 的落点: 这两条不
变量的实现本身就是 ``scripts/`` 下的静态审计脚本, 因此必须与十层一同登记,
否则 42 条不变量的覆盖统计会缺项。

用法:
    python scripts/gen_micro_instructions.py
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys

# Windows 默认控制台编码 (cp1252/cp936) 无法编码中文输出, 统一强制 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PACKAGE_RANK = {
    "core": 0, "rng": 1, "models": 2, "engine": 3, "train": 4,
    "retrieval": 5, "agent": 6, "serve": 7, "eval": 8, "cli": 9, "sdk": 9,
    "l10": 10,
}
# X 横切层: 不落在任何单一「层内模块」上的审计/门禁 (INV-41 / INV-42)
# 层号 11 —— L10 (微指令 ISA) 已占 10, X 层顺延。
CROSS_LAYER = 11

LAYER_NAMES = {
    0: "L0 算子/内核 (Operator/Kernel)",
    1: "L1 张量/数值基础 (Tensor/Numeric)",
    2: "L2 模型架构 (Model Architecture)",
    3: "L3 推理引擎 (Inference Engine)",
    4: "L4 训练/微调 (Training/Finetune)",
    5: "L5 检索/记忆 (Retrieval/Memory)",
    6: "L6 智能体 (Agent)",
    7: "L7 服务/编排 (Service/Orchestration)",
    8: "L8 评估/对齐 (Evaluation/Alignment)",
    9: "L9 应用/接口 (Application/Interface)",
    10: "L10 微指令 ISA (Micro-Instruction ISA)",
    CROSS_LAYER: "X 横切审计 (Cross-cutting Audits)",
}

INV_RE = re.compile(r"INV[ -]?(\d{1,2})")
INV_RANGE_RE = re.compile(r"INV[ -]?(\d{1,2})\s*[.。．·…\-–—]+\s*(\d{1,2})")
SRC_PATTERNS = [
    r"RoPE", r"Su et al", r"Shazeer", r"SwiGLU", r"RMSNorm", r"Zhang", r"GQA",
    r"Ainslie", r"MoE", r"Fedus", r"LoRA", r"Hu et al", r"Press", r"FlashAttention",
    r"Dao", r"NTK", r"PagedAttention", r"vLLM", r"Kwon", r"Orca", r"Yu et al",
    r"Speculative", r"Leviathan", r"AdamW", r"Loshchilov", r"Karpathy", r"nanogpt",
    r"Griewank", r"Walther", r"BM25", r"Robertson", r"HNSW", r"Malkov", r"RRF",
    r"Cormack", r"Nogueira", r"cross-?encoder", r"ReAct", r"Yao", r"Reflexion",
    r"Shinn", r"MemGPT", r"Packer", r"OpenAI", r"SSE", r"Jelinek", r"perplexity",
    r"MLPerf", r"GSM8K", r"Cobbe", r"BPE", r"Sennrich", r"Radford", r"Vaswani",
    r"dettmers", r"gptq", r"awq", r"int8", r"int4",
]
SRC_RE = [re.compile(p, re.IGNORECASE) for p in SRC_PATTERNS]


def layer_of(path: str) -> int:
    rel = os.path.relpath(path, os.path.join(REPO, "ifai"))
    parts = rel.split(os.sep)
    if len(parts) == 1:  # 顶层文件, 如 rng.py / sdk.py
        stem = parts[0][:-3] if parts[0].endswith(".py") else parts[0]
        return PACKAGE_RANK.get(stem, 9)
    pkg = parts[0]
    return PACKAGE_RANK.get(pkg, 9)


def extract_invs(text: str) -> set:
    """提取文本中的 INV 编号, 支持单条与 `INV-01..09` 区间展开。

    编号统一归一为**十进制整数字符串** (``"01"`` 与 ``"1"`` 视为同一条),
    否则覆盖统计会因前导零而重复计数。
    """
    found = set()
    for m in INV_RANGE_RE.finditer(text):
        a, b = int(m.group(1)), int(m.group(2))
        for x in range(min(a, b), max(a, b) + 1):
            found.add(str(x))
    for m in INV_RE.finditer(text):
        found.add(str(int(m.group(1))))
    return found


def args_sig(node: ast.arguments) -> str:
    names = [a.arg for a in node.posonlyargs + node.args]
    if node.vararg:
        names.append("*" + node.vararg.arg)
    names += [a.arg for a in node.kwonlyargs]
    if node.kwarg:
        names.append("**" + node.kwarg.arg)
    return ", ".join(names)


def describe_stmt(stmt) -> str:
    t = type(stmt).__name__
    if isinstance(stmt, ast.Assign):
        tgts = ", ".join(_name(n) for n in stmt.targets)
        return f"赋值 {tgts} ← {_expr(stmt.value)}"
    if isinstance(stmt, ast.AnnAssign):
        return f"声明并赋值 {_name(stmt.target)} ← {_expr(stmt.value)}"
    if isinstance(stmt, ast.AugAssign):
        return f"累算 {_name(stmt.target)} {type(stmt.op).__name__.lower()} {_expr(stmt.value)}"
    if isinstance(stmt, ast.Return):
        return f"返回 {_expr(stmt.value)}" if stmt.value else "返回"
    if isinstance(stmt, ast.If):
        return f"判断 {_expr(stmt.test)}"
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return f"遍历 {_name(stmt.target)} ∈ {_expr(stmt.iter)}"
    if isinstance(stmt, (ast.While,)):
        return f"当 {_expr(stmt.test)} 循环"
    if isinstance(stmt, ast.With):
        return "上下文管理 (with)"
    if isinstance(stmt, (ast.Try,)):
        return "异常保护 (try/except)"
    if isinstance(stmt, ast.Raise):
        return "抛出异常"
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        return f"调用 {_expr(stmt.value)}"
    if isinstance(stmt, ast.Expr):
        return f"表达式 {_expr(stmt.value)}"
    return t


def _name(n) -> str:
    if isinstance(n, ast.Name):
        return n.id
    if isinstance(n, ast.Attribute):
        return f"{_name(n.value)}.{n.attr}"
    if isinstance(n, ast.Subscript):
        return f"{_name(n.value)}[…]"
    if isinstance(n, ast.Tuple):
        return "(" + ", ".join(_name(e) for e in n.elts) + ")"
    return type(n).__name__


def _expr(n) -> str:
    if n is None:
        return "∅"
    if isinstance(n, ast.Call):
        return f"{_name(n.func)}(…)"
    if isinstance(n, ast.Name):
        return n.id
    if isinstance(n, ast.Attribute):
        return f"{_name(n.value)}.{n.attr}"
    if isinstance(n, ast.Constant):
        v = n.value
        if isinstance(v, str):
            return '"' + v[:16] + ('…' if len(v) > 16 else '') + '"'
        return repr(v)
    if isinstance(n, ast.BinOp):
        return f"{_expr(n.left)} {type(n.op).__name__.lower()} {_expr(n.right)}"
    if isinstance(n, ast.Subscript):
        return f"{_expr(n.value)}[…]"
    if isinstance(n, ast.List) or isinstance(n, ast.Tuple):
        return "[" + ", ".join(_expr(e) for e in n.elts[:3]) + "]"
    return type(n).__name__


def collect(root: str, force_layer: int | None = None):
    """返回 {layer: [entries]} 与全局统计。每个 entry 为 dict。

    Args:
        root: 源码根目录 (``ifai/`` 或 ``scripts/``)。
        force_layer: 显式指定层号 (``scripts/`` 用 ``CROSS_LAYER`` = X 横切层)。
    """
    by_layer = {i: [] for i in range(CROSS_LAYER + 1)}
    total_func = 0
    total_steps = 0
    inv_hits = set()
    for base, _, files in os.walk(root):
        for fn in sorted(files):
            if not fn.endswith(".py") or fn.startswith("__"):
                continue
            path = os.path.join(base, fn)
            try:
                src = open(path, "r", encoding="utf-8").read()
                tree = ast.parse(src, filename=path)
            except (OSError, SyntaxError):
                continue
            layer = force_layer if force_layer is not None else layer_of(path)
            rel = os.path.relpath(path, root)
            if force_layer is not None:
                # 横切层条目标注为仓库根相对路径, 与 ifai/ 内模块区分
                rel = os.path.join(os.path.basename(os.path.normpath(root)), rel)
            module_doc_inv = extract_invs(ast.get_docstring(tree) or "")

            # 模块级函数
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total_func += 1
                    e = _entry(rel, node, None, src, module_doc_inv)
                    by_layer[layer].append(e)
                    total_steps += len(e["steps"])
                    inv_hits |= e["inv"]
                elif isinstance(node, ast.ClassDef):
                    # 类: 记为一条结构微指令, 并枚举其方法
                    methods = [m for m in node.body
                               if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
                    total_func += 1
                    ce = _entry(rel, None, node, src, module_doc_inv, methods=methods)
                    by_layer[layer].append(ce)
                    total_steps += len(ce["steps"])
                    inv_hits |= ce["inv"]
                    for m in methods:
                        total_func += 1
                        me = _entry(rel, m, node, src, module_doc_inv)
                        by_layer[layer].append(me)
                        total_steps += len(me["steps"])
                        inv_hits |= me["inv"]
    return by_layer, total_func, total_steps, inv_hits


def _entry(rel, func, cls, src, module_doc_inv, methods=None):
    if func is not None:
        name = func.name
        sig = f"({args_sig(func.args)})"
        doc = ast.get_docstring(func) or ""
        body = func.body
        decorators = [ast.unparse(d) for d in func.decorator_list]
    else:
        name = cls.name
        sig = "(类)"
        doc = ast.get_docstring(cls) or ""
        body = cls.body
        decorators = []
    first_line = doc.strip().splitlines()[0] if doc.strip() else "(无 docstring)"
    func_inv = extract_invs(doc)
    inv = func_inv if func_inv else set(module_doc_inv)
    src_hits = sorted({p.pattern for p in SRC_RE if p.search(doc)})
    steps = []
    for stmt in body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) \
                and isinstance(stmt.value.value, str):
            continue  # 跳过 docstring 语句
        steps.append(describe_stmt(stmt))
    # 若没有有意义的步骤 (如纯 pass / docstring), 给出占位
    if not steps:
        steps = ["（空实现 / 占位）"]
    return {
        "layer_rel": rel,
        "qualname": (cls.name + "." if cls else "") + name,
        "sig": sig,
        "purpose": first_line,
        "inv": inv,
        "src": src_hits,
        "decorators": decorators,
        "steps": steps,
        "is_class": cls is not None and func is None,
    }


def render(by_layer, total_func, total_steps, inv_hits, out_path):
    lines = []
    lines.append("# 微指令级操作分解 (Micro-Instruction Decomposition)")
    lines.append("")
    lines.append("> 由 `scripts/gen_micro_instructions.py` 对 `ifai/` (L0–L9) 与 `scripts/` "
                 "(X 横切层) 源码 AST 静态扫描自动生成。")
    lines.append("> 零重写 · 集大成: 每条微指令标注其复用的世界顶级来源与所服务的可验证不变量 (INV)。")
    lines.append("")
    total_mi = total_func + total_steps
    lines.append(f"**总计**: 功能级微指令 {total_func} 条 + 步骤级微指令 {total_steps} 条 = "
                 f"**{total_mi}** 条; 覆盖 INV {sorted(int(x) for x in inv_hits)}。")
    lines.append("")
    lines.append("## 分层汇总")
    lines.append("")
    lines.append("| 层 | 说明 | 功能级 | 步骤级 | 合计 |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for i in range(CROSS_LAYER + 1):
        entries = by_layer[i]
        # entries 已含 class + methods + 模块函数
        fcount = len(entries)
        scount = sum(len(e["steps"]) for e in entries)
        tag = "X" if i == CROSS_LAYER else f"L{i}"
        lines.append(f"| {tag} | {LAYER_NAMES[i]} | {fcount} | {scount} | {fcount + scount} |")
    lines.append("")
    mid = 1
    for i in range(CROSS_LAYER + 1):
        entries = by_layer[i]
        lines.append(f"## {LAYER_NAMES[i]}" if i == CROSS_LAYER
                     else f"## L{i} {LAYER_NAMES[i].split(' ', 1)[1]}")
        lines.append("")
        for e in entries:
            tag = "【类】" if e["is_class"] else ""
            invs = ("  ·  INV " + ",".join(sorted(e["inv"]))) if e["inv"] else ""
            srcs = ("  ·  复用 " + ", ".join(e["src"])) if e["src"] else ""
            lines.append(f"### MI-{mid:04d} {tag}{e['qualname']}{e['sig']}")
            lines.append(f"- 模块: `{e['layer_rel']}`")
            lines.append(f"- 用途: {e['purpose']}{invs}{srcs}")
            if e["decorators"]:
                lines.append(f"- 装饰器: {', '.join(e['decorators'])}")
            lines.append(f"- 微指令步骤 ({len(e['steps'])}):")
            for s in e["steps"]:
                lines.append(f"  - {s}")
            lines.append("")
            mid += 1
    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text, total_mi


def main() -> int:
    ap = argparse.ArgumentParser(description="生成微指令级操作分解")
    ap.add_argument("--root", default=os.path.join(REPO, "ifai"))
    ap.add_argument("--scripts-root", default=os.path.join(REPO, "scripts"),
                    help="横切审计脚本根目录 (X 层, INV-41/INV-42 落点)")
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "MICRO_INSTRUCTIONS.md"))
    args = ap.parse_args()

    # 1) 十层: ifai/ 源码
    by_layer, total_func, total_steps, inv_hits = collect(args.root)
    # 2) X 横切层: scripts/ 审计脚本 (INV-41 分层依赖 / INV-42 零重写审计)
    if os.path.isdir(args.scripts_root):
        x_layer, f_x, s_x, inv_x = collect(args.scripts_root, force_layer=CROSS_LAYER)
        for k, entries in x_layer.items():
            by_layer[k].extend(entries)
        total_func += f_x
        total_steps += s_x
        inv_hits |= inv_x
    text, total_mi = render(by_layer, total_func, total_steps, inv_hits, args.out)
    print(f"功能级微指令: {total_func}")
    print(f"步骤级微指令: {total_steps}")
    print(f"总计微指令: {total_mi}  (目标 >= 1024)")
    print(f"覆盖 INV: {sorted(int(x) for x in inv_hits)}")
    print(f"覆盖 INV 数: {len(inv_hits)}")
    print(f"输出: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
