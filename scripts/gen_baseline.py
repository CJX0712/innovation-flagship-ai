#!/usr/bin/env python
"""基准生成与回归门禁 (Benchmark Baseline) — R-L8-02 / R-L8-05。

在 CPU 上对一个微型模型测量:
- prefill_ms: 对定长 prompt 做一次前向 (代表 TTFT 的主要构成)
- tpot_ms: 每个 decode 步的平均耗时 (Time Per Output Token)
- throughput: decode 吞吐 (tok/s)

并写入 ``BASELINE.md`` (人类可读) 与 ``BASELINE.json`` (机器比较)。
带 ``--check`` 时, 与既有 BASELINE.json 比较, 若 prefill/tpot 退化 >15% 则退出 1。

零重写: 吞吐/延迟基准方法论参考 MLPerf (MLCommons)。

用法:
    python scripts/gen_baseline.py                # 生成 BASELINE.md/json
    python scripts/gen_baseline.py --check         # 回归门禁
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ifai import rng  # noqa: E402
from ifai.models.config import ModelConfig  # noqa: E402
from ifai.models.tokenizer import Tokenizer  # noqa: E402
from ifai.models.transformer import load_model  # noqa: E402
from ifai.train.trainer import TinyTrainer  # noqa: E402

CORPUS = (
    "人工智能是计算机科学的一个分支。模型通过学习数据来提升能力。"
    "变压器网络由注意力机制构成。注意力让模型关注重要的信息。"
    "训练需要大量的数据和计算资源。推理是把训练好的模型用于预测。"
    "分词把文本切分成更小的单元。嵌入把符号映射到向量空间。"
    "采样从模型的输出分布中选择下一个词。检索从知识库中找到相关资料。"
)

PROMPT_LEN = 32
DECODE_STEPS = 32
REPEATS = 3
REGRESSION_TOL = 0.15  # ±15% 门禁


def build_model(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    rng.seed(0)
    cfg = ModelConfig(
        vocab_size=256, d_model=64, n_layers=3, n_heads=4,
        n_kv_heads=2, ffn_mult=4, max_seq_len=128,
    )
    tok = Tokenizer(vocab_size=256).train(CORPUS)
    tok.to_json(os.path.join(out_dir, "tokenizer.json"))
    trainer = TinyTrainer(cfg)
    trainer.train(tok.encode(CORPUS), steps=60, lr=3e-3)
    trainer.save(out_dir)
    return load_model(out_dir), tok


def bench(model, tok) -> dict:
    ids = (tok.encode("人工智能模型注意力推理检索") or [0])[:PROMPT_LEN]
    while len(ids) < PROMPT_LEN:
        ids.append(0)
    ids = ids[:PROMPT_LEN]

    prefill_ms, decode_ms = [], []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        model.forward(np.array([ids]))
        prefill_ms.append((time.perf_counter() - t0) * 1000)

        seq = list(ids)
        dts = []
        for _ in range(DECODE_STEPS):
            t1 = time.perf_counter()
            logits = model.forward(np.array([seq]))[0, -1, :]
            dts.append((time.perf_counter() - t1) * 1000)
            seq.append(int(np.argmax(logits)))
        decode_ms.extend(dts)

    prefill = statistics.mean(prefill_ms)
    tpot = statistics.mean(decode_ms)
    return {
        "prompt_len": PROMPT_LEN,
        "decode_steps": DECODE_STEPS,
        "repeats": REPEATS,
        "prefill_ms": round(prefill, 3),
        "tpot_ms": round(tpot, 3),
        "throughput_tok_s": round(1000.0 / tpot, 2),
    }


def write_baseline(metrics: dict, repo_root: str) -> None:
    md = os.path.join(repo_root, "BASELINE.md")
    js = os.path.join(repo_root, "BASELINE.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    lines = [
        "# 性能基线 (BASELINE)",
        "",
        "> 由 `scripts/gen_baseline.py` 在 CPU 上对微型模型 (d=64, 3 层) 实测生成。",
        "> 回归门禁: prefill / TPOT 退化 > 15% 视为性能回归 (R-L8-05)。",
        "",
        "## 环境",
        "",
        f"- 设备: CPU (100% numpy, 无 GPU)",
        f"- 模型: TinyTransformer d_model=64, n_layers=3, n_heads=4, n_kv_heads=2",
        "",
        "## 指标",
        "",
        "| 指标 | 值 | 说明 |",
        "| --- | --- | --- |",
        f"| Prefill (prompt_len={metrics['prompt_len']}) | {metrics['prefill_ms']} ms | 单次前向, 构成 TTFT 主体 |",
        f"| TPOT | {metrics['tpot_ms']} ms/token | 每输出 token 平均耗时 |",
        f"| Decode 吞吐 | {metrics['throughput_tok_s']} tok/s | 1000 / TPOT |",
        "",
        "## 复现",
        "",
        "```bash",
        "python scripts/gen_baseline.py            # 重新生成 BASELINE.md/json",
        "python scripts/gen_baseline.py --check     # 作为 CI 回归门禁",
        "```",
        "",
    ]
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description="基准生成 / 回归门禁")
    ap.add_argument("--check", action="store_true", help="与既有 BASELINE.json 比较, >15% 退化则失败")
    ap.add_argument("--out", default=REPO, help="输出目录 (默认仓库根)")
    args = ap.parse_args()
    root = args.out

    metrics = bench(*build_model(os.path.join(root, "_baseline_model")))

    if args.check:
        js = os.path.join(root, "BASELINE.json")
        if not os.path.isfile(js):
            print("[WARN] 未找到 BASELINE.json, 跳过门禁 (首次生成)。")
            write_baseline(metrics, root)
            return 0
        prev = json.load(open(js, encoding="utf-8"))
        bad = []
        for k in ("prefill_ms", "tpot_ms"):
            cur, old = metrics[k], prev[k]
            if old <= 0:
                continue
            delta = (cur - old) / old
            if delta > REGRESSION_TOL:
                bad.append(f"{k}: {old} -> {cur} (+{delta*100:.1f}%)")
        if bad:
            print("性能回归 (超过 +15%):")
            for b in bad:
                print("  ❌", b)
            return 1
        print("性能门禁通过: 无 >15% 退化。")
        return 0

    write_baseline(metrics, root)
    print("BASELINE 已生成:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
