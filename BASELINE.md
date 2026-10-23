# 性能基线 (BASELINE)

> 由 `scripts/gen_baseline.py` 在 CPU 上对微型模型 (d=64, 3 层) 实测生成。
> 回归门禁: prefill / TPOT 退化 > 15% 视为性能回归 (R-L8-05)。

## 环境

- 设备: CPU (100% numpy, 无 GPU)
- 模型: TinyTransformer d_model=64, n_layers=3, n_heads=4, n_kv_heads=2

## 指标

| 指标 | 值 | 说明 |
| --- | --- | --- |
| Prefill (prompt_len=32) | 6.699 ms | 单次前向, 构成 TTFT 主体 |
| TPOT | 9.246 ms/token | 每输出 token 平均耗时 |
| Decode 吞吐 | 108.16 tok/s | 1000 / TPOT |
| Prefill 离散度 (CV) | 22.66% | 测量噪声边界 |
| TPOT 离散度 (CV) | 23.35% | 测量噪声边界 |

## 统计口径

- 统计量: **median** (中位数), 非均值 ——
  CPU 计时受调度抖动影响, 均值会被一次系统级抢占抬高十几个百分点,
  使门禁在「代码没变」时误报红灯。
- 回归阈值: `max(15%, 2×基线 CV)` —— 低于噪声边界的波动不判为回归。
- 权威基线: **`BASELINE.json`** (机器读取)。本文件仅供人读,
  不再被任何代码解析 (双轨合一, v0.8.0)。

## 机器可读 (RegressionBaseline.parse_baseline 读 BASELINE.json)

tok/s mean: 108.16

## 复现

```bash
python scripts/gen_baseline.py            # 重新生成 BASELINE.md/json
python scripts/gen_baseline.py --check     # 作为 CI 回归门禁
```
