# 性能基线 (BASELINE)

> 由 `scripts/gen_baseline.py` 在 CPU 上对微型模型 (d=64, 3 层) 实测生成。
> 回归门禁: prefill / TPOT 退化 > 15% 视为性能回归 (R-L8-05)。

## 环境

- 设备: CPU (100% numpy, 无 GPU)
- 模型: TinyTransformer d_model=64, n_layers=3, n_heads=4, n_kv_heads=2

## 指标

| 指标 | 值 | 说明 |
| --- | --- | --- |
| Prefill (prompt_len=32) | 2.076 ms | 单次前向, 构成 TTFT 主体 |
| TPOT | 3.553 ms/token | 每输出 token 平均耗时 |
| Decode 吞吐 | 281.46 tok/s | 1000 / TPOT |

## 复现

```bash
python scripts/gen_baseline.py            # 重新生成 BASELINE.md/json
python scripts/gen_baseline.py --check     # 作为 CI 回归门禁
```
