"""TinyTrainer 训练吞吐基准 (v0.7.0 批量可微注意力)。

用法:
    python scripts/bench_trainer.py [--steps 60]

历史对照:
- v0.6.x (逐头循环 + 梯度断裂): 57.4 ms/step (60 steps, 3.44s)
- v0.7.0 (批量 GQA + autograd 贯通): 见本脚本输出

配置: d_model=128, n_layers=4, n_heads=8, n_kv_heads=2 (GQA), B=8, T=32。
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from ifai import rng
from ifai.models.config import ModelConfig
from ifai.train.trainer import TinyTrainer


def bench(steps: int = 60, d_model: int = 128, n_layers: int = 4,
          n_heads: int = 8, n_kv_heads: int = 2,
          batch_size: int = 8, seq_len: int = 32, vocab_size: int = 256) -> float:
    rng.seed(42)
    cfg = ModelConfig(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
                      n_heads=n_heads, n_kv_heads=n_kv_heads, ffn_mult=4,
                      max_seq_len=seq_len * 2)
    corpus = rng.get_rng().integers(0, vocab_size, size=seq_len * 64)
    tr = TinyTrainer(cfg)
    t0 = time.perf_counter()
    tr.train(np.asarray(corpus), steps=steps, lr=3e-3,
             batch_size=batch_size, seq_len=seq_len)
    dt = time.perf_counter() - t0
    ms = dt / steps * 1000
    print(f"{steps} steps d={d_model} L={n_layers} H={n_heads} "
          f"n_kv={n_kv_heads} B={batch_size} T={seq_len} -> {dt:.2f}s "
          f"({ms:.1f} ms/step)")
    print(f"loss head/tail: {tr._losses[0]:.4f} -> {tr._losses[-1]:.4f}"
          if hasattr(tr, "_losses") else "")
    return ms


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()
    bench(steps=args.steps)
