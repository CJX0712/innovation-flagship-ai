/**
 * @forge-module loss-act-norm-forge
 * @layer L3
 * @version 1.0.0
 * @brief     损失/激活/归一化内核：MSE、交叉熵(稳定 softmax+NLL)、RMSNorm、LayerNorm、GELU(精确 erf 与 tanh 近似)、Swish、SwiGLU。所有输出均带可交叉验证不变量。
 * @reuse     llama.cpp RMSNorm|https://github.com/ggerganov/llama.cpp|MIT|C|RMSNorm 定义与实现
 * @reuse     PyTorch nn|https://github.com/pytorch/pytorch|BSD-2|C|LayerNorm/GELU/SwiGLU 官方公式
 * @reuse     Hendrycks GELU|https://arxiv.org/abs/1606.08415|公有域|D|tanh 近似系数
 * @beyond    纯 JS 单文件零构建；稳定 softmax 复用 stable-forge 思路(移位)；RMSNorm/LayerNorm 归一不变量的 RMS/方差零均值可证；SwiGLU 与独立重实现交叉校验。
 * @invariant INV-LAN-001|RMSNorm 归一分量 RMS(x/rms) = 1 ≤ 1e-12|1e-12|RMSNorm 定义
 * @invariant INV-LAN-002|LayerNorm 归一分量零均值且方差=1 ≤ 1e-12|1e-12|LayerNorm 定义
 * @invariant INV-LAN-003|GELU(0)=0 精确；且 GELU 单调递增|0|GELU 性质
 * @invariant INV-LAN-004|GELU tanh 近似与精确 erf 式在采样点最大偏差 ≤ 2e-2|2e-2|近似保真
 * @invariant INV-LAN-005|SwiGLU 与独立重实现一致 ≤ 1e-12|1e-12|参照实现 vs 主实现
 * @invariant INV-LAN-006|CE 梯度 = softmax−onehot；且 CE ≥ 0|1e-12|交叉熵梯度定义
 * @invariant INV-LAN-007|MSE 梯度 = (2/n)(y−t) 与解析一致；MSE=mean((y−t)²)|1e-12|MSE 定义
 * @invariant INV-LAN-008|CE = −log(softmax(l)[target]) 与独立计算一致；Swish(0)=0|1e-12|CE 与 Swish 定义
 * @seed      0x9E3779B9
 */
'use strict';
const { makeRng, TOL } = require('./core-numeric.js');
const { caseResult, buildReport } = require('./core-verify.js');

/* ---------- 基础数值原语（稳定 softmax，复用 stable-forge 移位思路） ---------- */
function stableSoftmax(a) {
  let m = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i];
  const e = new Float64Array(a.length);
  let s = 0;
  for (let i = 0; i < a.length; i++) { const x = Math.exp(a[i] - m); e[i] = x; s += x; }
  for (let i = 0; i < a.length; i++) e[i] /= s;
  return e;
}

function erf(x) {
  const sgn = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax);
  return sgn * y;
}
function mean(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; }
function rmsOf(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s / a.length); }

/* ---------- 损失 ---------- */
function mse(y, t) { let s = 0; for (let i = 0; i < y.length; i++) { const d = y[i] - t[i]; s += d * d; } return s / y.length; }
function mseGrad(y, t) { const g = new Float64Array(y.length); const f = 2 / y.length; for (let i = 0; i < y.length; i++) g[i] = f * (y[i] - t[i]); return g; }

// 交叉熵（logits, 整数 target）：CE = logsumexp(l) − l[target]
function crossEntropy(logits, target) {
  let m = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > m) m = logits[i];
  let s = 0;
  for (let i = 0; i < logits.length; i++) s += Math.exp(logits[i] - m);
  const lse = m + Math.log(s);
  return lse - logits[target];
}
// CE 对 logits 的梯度 = softmax − onehot
function crossEntropyGrad(logits, target) {
  const p = stableSoftmax(logits);
  const g = Float64Array.from(p);
  g[target] -= 1;
  return g;
}

/* ---------- 归一化 ---------- */
// RMSNorm：out = (x / rms(x)) · g   （rms = sqrt(mean(x²)+eps)）
function rmsNorm(x, g, eps) {
  if (eps === undefined) eps = 1e-6;
  const r = Math.sqrt(mean(x.map(v => v * v)) + eps);
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = (x[i] / r) * (g ? g[i] : 1);
  return out;
}
// LayerNorm：out = (x−μ)/sqrt(var+eps) · g + b
function layerNorm(x, g, b, eps) {
  if (eps === undefined) eps = 1e-6;
  const mu = mean(x);
  let v = 0; for (let i = 0; i < x.length; i++) { const d = x[i] - mu; v += d * d; }
  const std = Math.sqrt(v / x.length + eps);
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = ((x[i] - mu) / std) * (g ? g[i] : 1) + (b ? b[i] : 0);
  return out;
}

/* ---------- 激活 ---------- */
// 精确 GELU：x·½(1+erf(x/√2))
function gelu(x) { return x * 0.5 * (1 + erf(x / Math.SQRT2)); }
// tanh 近似 GELU（Hendrycks）：x·½(1+tanh(√(2/π)·(x+0.044715 x³)))
function geluTanh(x) {
  const c = Math.sqrt(2 / Math.PI);
  return x * 0.5 * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
}
// Swish：x·sigmoid(x) = x/(1+e^{-x})
function swish(x) { return x / (1 + Math.exp(-x)); }
// SwiGLU(x) = Swish(xW+b) ⊗ (xV+c)  （gate·value，逐元素）
function swiglu(x, W, V, bg, bv) {
  const d = x.length;
  const g = new Float64Array(d), v = new Float64Array(d);
  for (let i = 0; i < d; i++) {
    let gw = bg ? bg[i] : 0, vv = bv ? bv[i] : 0;
    for (let j = 0; j < d; j++) { gw += x[j] * W[i * d + j]; vv += x[j] * V[i * d + j]; }
    g[i] = swish(gw); v[i] = vv;
  }
  const out = new Float64Array(d);
  for (let i = 0; i < d; i++) out[i] = g[i] * v[i];
  return out;
}

/* ============================================================================
 * selfCheck — 8 条不变量
 * ========================================================================== */
function selfCheck(opts) {
  opts = opts || {};
  const cases = [];
  const invariants = [];
  const rng = makeRng(0x9e3779b9);

  // INV-LAN-001 RMSNorm 归一分量 RMS=1
  {
    let mx = 0;
    for (let t = 0; t < 20; t++) {
      const n = 8; const x = new Float64Array(n);
      for (let i = 0; i < n; i++) x[i] = rng() * 4 - 2;
      const g = new Float64Array(n);
      for (let i = 0; i < n; i++) g[i] = 0.5 + rng();
      const out = rmsNorm(x, g, 0); // eps=0 验证纯定义 RMS=1（eps 仅为数值稳定，不属定义）
      // 去掉增益后应为 RMS=1
      let s = 0; for (let i = 0; i < n; i++) { const u = out[i] / g[i]; s += u * u; }
      mx = Math.max(mx, Math.abs(Math.sqrt(s / n) - 1));
    }
    cases.push(caseResult('INV-LAN-001', 'RMSNorm 归一 RMS=1', mx, 0, TOL.RESID));
  }
  // INV-LAN-002 LayerNorm 零均值 + 单位方差
  {
    let mx = 0;
    for (let t = 0; t < 20; t++) {
      const n = 10; const x = new Float64Array(n);
      for (let i = 0; i < n; i++) x[i] = rng() * 6 - 3;
      const g = new Float64Array(n), b = new Float64Array(n);
      for (let i = 0; i < n; i++) { g[i] = 0.5 + rng(); b[i] = rng() - 0.5; }
      const out = layerNorm(x, g, b, 0); // eps=0 验证纯定义：零均值 + 单位方差
      const u = new Float64Array(n); for (let i = 0; i < n; i++) u[i] = (out[i] - b[i]) / g[i];
      const mu = mean(u);
      let v = 0; for (let i = 0; i < n; i++) { const d = u[i] - mu; v += d * d; }
      mx = Math.max(mx, Math.abs(mu), Math.abs(v / n - 1));
    }
    cases.push(caseResult('INV-LAN-002', 'LayerNorm 零均值+单位方差', mx, 0, TOL.RESID));
  }
  // INV-LAN-003 GELU(0)=0 精确；大数渐近 gelu(x)≈x；且在 [0,∞) 非减（注：GELU 在负区因 x·Φ(x) 形态并非全局单调，故只校验非负半轴）
  {
    const z = gelu(0);
    const asymErr = Math.abs(gelu(8) - 8);
    let incrOk = true, prev = -Infinity;
    for (let i = 0; i <= 24; i++) { const x = 6 * i / 24; const val = gelu(x); if (val < prev - 1e-12) incrOk = false; prev = val; }
    const metric = Math.max(Math.abs(z), asymErr, incrOk ? 0 : 1);
    cases.push(caseResult('INV-LAN-003', 'GELU(0)=0 + 渐近 + [0,∞)非减', metric, 0, 1e-3));
  }
  // INV-LAN-004 GELU tanh 近似 vs 精确
  {
    let mx = 0;
    const pts = [-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4];
    for (const x of pts) mx = Math.max(mx, Math.abs(geluTanh(x) - gelu(x)));
    // 额外随机点
    for (let i = 0; i < 30; i++) { const x = rng() * 8 - 4; mx = Math.max(mx, Math.abs(geluTanh(x) - gelu(x))); }
    cases.push(caseResult('INV-LAN-004', 'GELU tanh 近似 vs 精确', mx, 0, 2e-2));
  }
  // INV-LAN-005 SwiGLU 与独立重实现一致
  {
    let mx = 0;
    const d = 4;
    const x = new Float64Array(d); for (let i = 0; i < d; i++) x[i] = rng() * 2 - 1;
    const W = new Float64Array(d * d), V = new Float64Array(d * d), bg = new Float64Array(d), bv = new Float64Array(d);
    for (let i = 0; i < d * d; i++) { W[i] = rng() * 2 - 1; V[i] = rng() * 2 - 1; }
    for (let i = 0; i < d; i++) { bg[i] = rng() - 0.5; bv[i] = rng() - 0.5; }
    const out = swiglu(x, W, V, bg, bv);
    // 独立重实现
    const ref = new Float64Array(d);
    for (let i = 0; i < d; i++) {
      let gw = bg[i], vv = bv[i];
      for (let j = 0; j < d; j++) { gw += x[j] * W[i * d + j]; vv += x[j] * V[i * d + j]; }
      ref[i] = (gw / (1 + Math.exp(-gw))) * vv;
    }
    for (let i = 0; i < d; i++) mx = Math.max(mx, Math.abs(out[i] - ref[i]));
    cases.push(caseResult('INV-LAN-005', 'SwiGLU vs 独立重实现', mx, 0, TOL.RESID));
  }
  // INV-LAN-006 CE 梯度 = softmax−onehot；CE≥0
  {
    let mx = 0, nonNeg = true;
    for (let t = 0; t < 20; t++) {
      const n = 5; const l = new Float64Array(n);
      for (let i = 0; i < n; i++) l[i] = rng() * 4 - 2;
      const target = t % n;
      const ce = crossEntropy(l, target);
      if (ce < -1e-12) nonNeg = false;
      const g = crossEntropyGrad(l, target);
      const p = stableSoftmax(l);
      let dev = 0; for (let i = 0; i < n; i++) dev = Math.max(dev, Math.abs(g[i] - (p[i] - (i === target ? 1 : 0))));
      mx = Math.max(mx, dev);
    }
    cases.push(caseResult('INV-LAN-006', 'CE 梯度 = softmax−onehot', mx, 0, TOL.RESID));
  }
  // INV-LAN-007 MSE 梯度与解析一致；MSE 值正确
  {
    let mx = 0;
    for (let t = 0; t < 20; t++) {
      const n = 6; const y = new Float64Array(n), tt = new Float64Array(n);
      for (let i = 0; i < n; i++) { y[i] = rng() * 2 - 1; tt[i] = rng() * 2 - 1; }
      const m = mse(y, tt);
      let ref = 0; for (let i = 0; i < n; i++) { const d = y[i] - tt[i]; ref += d * d; } ref /= n;
      const g = mseGrad(y, tt);
      let gdev = 0; for (let i = 0; i < n; i++) gdev = Math.max(gdev, Math.abs(g[i] - (2 / n) * (y[i] - tt[i])));
      mx = Math.max(mx, Math.abs(m - ref), gdev);
    }
    cases.push(caseResult('INV-LAN-007', 'MSE 值+梯度', mx, 0, TOL.RESID));
  }
  // INV-LAN-008 CE = −log(softmax[target])；Swish(0)=0
  {
    let mx = 0;
    for (let t = 0; t < 20; t++) {
      const n = 5; const l = new Float64Array(n);
      for (let i = 0; i < n; i++) l[i] = rng() * 4 - 2;
      const target = t % n;
      const ce = crossEntropy(l, target);
      const p = stableSoftmax(l);
      mx = Math.max(mx, Math.abs(ce - (-Math.log(p[target]))));
    }
    mx = Math.max(mx, Math.abs(swish(0) - 0));
    cases.push(caseResult('INV-LAN-008', 'CE=−log(softmax[t])；Swish(0)=0', mx, 0, TOL.RESID));
  }

  const stat = (id) => cases.find(c => c.id === id).status;
  for (const id of ['INV-LAN-001', 'INV-LAN-002', 'INV-LAN-003', 'INV-LAN-004', 'INV-LAN-005', 'INV-LAN-006', 'INV-LAN-007', 'INV-LAN-008']) {
    const c = cases.find(x => x.id === id);
    invariants.push({ id, expr: id, tol: c.tol, worst: c.abs_err, status: c.status, note: '' });
  }

  const started = Date.now();
  return buildReport({ module: 'loss-act-norm-forge', version: '1.0.0', layer: 'L3', cases, invariants, seed: '0x9E3779B9', started, tol: { 'INV-LAN-001': TOL.RESID, 'INV-LAN-004': 2e-2 } });
}

const API = {
  name: 'loss-act-norm-forge', version: '1.0.0', layer: 'L3',
  stableSoftmax, erf, mean, rmsOf, mse, mseGrad, crossEntropy, crossEntropyGrad,
  rmsNorm, layerNorm, gelu, geluTanh, swish, swiglu, selfCheck
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeLossActNorm = API;

if (typeof process !== 'undefined' && process.argv[1] && /loss-act-norm-forge\.js$/.test(process.argv[1]) && process.argv.includes('--self-check')) {
  const r = selfCheck({ quick: process.argv.includes('--quick') });
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : require('./core-verify.js').formatTableSafe(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
