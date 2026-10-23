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
REPEATS = 9
WARMUP = 2
REGRESSION_TOL = 0.15  # ±15% 门禁 (与噪声边界取大者)


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
    """测量 prefill / TPOT / 吞吐。

    零重写 · 集大成: 统计口径参考 **MLPerf Inference** 的做法 —— 报告
    **中位数** 而非均值, 并附带离散度 (CV), 因为 CPU 计时受调度抖动影响
    很大: 用均值时, 一次系统级抢占就能把基线抬高 18%, 门禁随后在
    「代码没变」的情况下误报红灯 (本机实测: 同机两次测量差 18%)。
    中位数 + CV 让基线自带噪声边界, 门禁只在真实退化时亮红。

    Args:
        model: 待测模型。
        tok: 对应分词器。

    Returns:
        指标 dict (含 prefill_ms / tpot_ms / throughput_tok_s 及离散度)。
    """
    ids = (tok.encode("人工智能模型注意力推理检索") or [0])[:PROMPT_LEN]
    while len(ids) < PROMPT_LEN:
        ids.append(0)
    ids = ids[:PROMPT_LEN]

    prefill_ms, decode_ms = [], []
    per_repeat_tpot = []
    # 预热 (v0.8.0): 前若干轮的耗时含冷启动 (内存页首次触碰、NumPy/BLAS
    # 首次分配、CPU 频率爬坡), 不剔除会把基线系统性抬高。
    for _ in range(WARMUP):
        model.forward(np.array([ids]))
        seq = list(ids)
        for _ in range(DECODE_STEPS):
            logits = model.forward(np.array([seq]))[0, -1, :]
            seq.append(int(np.argmax(logits)))
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
        # 每轮一个 TPOT 中位数样本 —— 离散度按**轮次样本**统计, 而不是
        # 把 9×32 个逐 token 耗时混池: 混池会把「个别 token 的抖动」
        # 放大成轮级离散度, 使噪声边界虚高 (实测混池 CV 44%, 轮级 CV 3%)。
        per_repeat_tpot.append(statistics.median(dts))

    prefill = statistics.median(prefill_ms)
    tpot = statistics.median(decode_ms)

    def _cv(xs):
        """相对离散度 (CV%) —— 基于轮次样本。"""
        m = statistics.mean(xs)
        return round(statistics.pstdev(xs) / m * 100.0, 2) if m > 0 and len(xs) > 1 else 0.0

    return {
        "prompt_len": PROMPT_LEN,
        "decode_steps": DECODE_STEPS,
        "repeats": REPEATS,
        "warmup": WARMUP,
        "prefill_ms": round(prefill, 3),
        "tpot_ms": round(tpot, 3),
        "throughput_tok_s": round(1000.0 / tpot, 2),
        "prefill_cv_pct": _cv(prefill_ms),
        "tpot_cv_pct": _cv(per_repeat_tpot),
        "statistic": "median",
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
        f"| Prefill 离散度 (CV) | {metrics.get('prefill_cv_pct', 0.0)}% | 测量噪声边界 |",
        f"| TPOT 离散度 (CV) | {metrics.get('tpot_cv_pct', 0.0)}% | 测量噪声边界 |",
        "",
        "## 统计口径",
        "",
        f"- 统计量: **{metrics.get('statistic', 'median')}** (中位数), 非均值 ——",
        "  CPU 计时受调度抖动影响, 均值会被一次系统级抢占抬高十几个百分点,",
        "  使门禁在「代码没变」时误报红灯。",
        f"- 回归阈值: `max(15%, 2×基线 CV)` —— 低于噪声边界的波动不判为回归。",
        "- 权威基线: **`BASELINE.json`** (机器读取)。本文件仅供人读,",
        "  不再被任何代码解析 (双轨合一, v0.8.0)。",
        "",
        "## 机器可读 (RegressionBaseline.parse_baseline 读 BASELINE.json)",
        "",
        f"tok/s mean: {metrics['throughput_tok_s']}",
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
        # 双轨合一 (v0.8.0): 门禁判据统一由 L8 `RegressionBaseline.check`
        # 提供, 本脚本不再自带一份比较逻辑 —— 此前脚本比 prefill/tpot
        # (时间, 越大越慢), 而 L8 harness 比吞吐 (tok/s, 越小越慢),
        # 两套口径各自演化, 是「双轨分裂」的根因。现统一为同一函数。
        from ifai.eval.harness import RegressionBaseline

        cur = {
            "tok_per_s_mean": metrics["throughput_tok_s"],
            "prefill_ms": metrics["prefill_ms"],
            "tpot_ms": metrics["tpot_ms"],
        }
        # 噪声边界: 基线自带 CV (离散度), 门禁阈值取 max(固定阈值, 基线 CV)。
        # 没有这一项, CPU 调度抖动会让「代码没变」也误报红灯 (实测 18%)。
        noise_pct = max(float(prev.get("tpot_cv_pct", 0.0)),
                         float(prev.get("prefill_cv_pct", 0.0)))
        tol = max(REGRESSION_TOL, noise_pct / 100.0 * 2)
        verdict = RegressionBaseline.check(cur, js, threshold=tol)
        bad = []
        # prefill / TPOT 的时间型退化判据 (吞吐口径覆盖不了这两项)
        for k in ("prefill_ms", "tpot_ms"):
            cur_v, old_v = metrics[k], prev.get(k)
            if not old_v or old_v <= 0:
                continue
            delta = (cur_v - old_v) / old_v
            if delta > tol:
                bad.append(f"{k}: {old_v} -> {cur_v} (+{delta*100:.1f}%)")
        if verdict["status"] == "RED":
            bad.append(f"throughput: 退化 {verdict['drop']*100:.1f}% "
                       f"(> {tol*100:.0f}%)")
        if bad:
            print(f"性能回归 (超过 +{tol*100:.0f}%, 噪声边界 "
                  f"{noise_pct:.1f}%):")
            for b in bad:
                print("  ❌", b)
            return 1
        print(f"性能门禁通过: 无 >{tol*100:.0f}% 退化 "
              f"(基线源 {verdict.get('source')}, 噪声边界 {noise_pct:.1f}%)。")
        return 0

    write_baseline(metrics, root)
    print("BASELINE 已生成:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
