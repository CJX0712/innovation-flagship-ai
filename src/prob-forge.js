/**
 * @forge-module prob-forge
 * @layer L5
 * @version 1.0.0
 * @brief     概率分布与信息论内核：高斯对数 PDF、伯努利 PMF/对数 PMF、高斯/伯努利 KL 散度、逆 CDF 正态采样。所有接口可交叉验证（闭式对照、参考重实现、蒙特卡洛矩收敛）。
 * @reuse     scikit-learn|https://github.com/scikit-learn/scikit-learn|BSD-3|B|多元高斯与 KL 散度参照
 * @reuse     PyTorch distributions|https://github.com/pytorch/pytorch|BSD-2|C|概率分布与 KL 实现参照
 * @beyond    纯 JS 单文件零构建；KL 散度闭式与参考重实现互验；采样用 Acklam 逆正态 CDF（core-numeric）保证双端一致；全部附可交叉验证不变量。
 * @invariant INV-PRB-001|gaussianLogPdf(x,μ,σ) 与闭式 -0.5·log(2πσ²)-0.5·((x-μ)/σ)² 一致 ≤ 1e-12|1e-12|对数高斯定义
 * @invariant INV-PRB-002|klGaussian(m,s,m,s)==0 ≤ 1e-12|1e-12|同分布 KL=0
 * @invariant INV-PRB-003|klGaussian ≥ 0（10 组随机对）≤ 1e-12|1e-12|KL 非负性
 * @invariant INV-PRB-004|klGaussian(m0,s0,m1,s1) 与内联参考 0.5·(log(s1²/s0²)+(s0²+(m0-m1)²)/s1²-1) 一致 ≤ 1e-12|1e-12|参考重实现
 * @invariant INV-PRB-005|bernoulliPmf(1,p)+bernoulliPmf(0,p)==1 ≤ 1e-12|1e-12|概率归一
 * @invariant INV-PRB-006|bernoulliLogPmf(1,p)==log(p) ≤ 1e-12|1e-12|对数 PMF 定义
 * @invariant INV-PRB-007|sampleGaussian N=20000：|样本均值-μ|≤0.05 且 |样本方差-σ²|≤0.05|0.05|MCSE 余量（约 5σ）
 * @invariant INV-PRB-008|klBernoulli(p,q) 与 p·log(p/q)+(1-p)·log((1-p)/(1-q)) 一致且 p==q 时 ==0 ≤ 1e-12|1e-12|参考重实现
 * @seed      0x9E3779B9
 */
'use strict';
const { makeRng, TOL, normInv } = require('./core-numeric.js');
const { caseResult, buildReport } = require('./core-verify.js');

/* ---------- 概率分布原语 ---------- */

// 高斯对数概率密度：μ 均值，σ 标准差（>0）
function gaussianLogPdf(x, mu, sigma) {
  if (!(typeof sigma === 'number' && isFinite(sigma) && sigma > 0)) throw new TypeError('gaussianLogPdf: sigma 须为正实数');
  if (!(typeof x === 'number' && isFinite(x)) || !(typeof mu === 'number' && isFinite(mu))) throw new TypeError('gaussianLogPdf: x,mu 须为有限实数');
  const z = (x - mu) / sigma;
  return -0.5 * Math.log(2 * Math.PI * sigma * sigma) - 0.5 * z * z;
}

// 伯努利 PMF：k∈{0,1}
function bernoulliPmf(k, p) {
  if (k !== 0 && k !== 1) throw new TypeError('bernoulliPmf: k 须为 0 或 1');
  if (!(typeof p === 'number' && p >= 0 && p <= 1)) throw new TypeError('bernoulliPmf: p 须 ∈ [0,1]');
  return k === 1 ? p : (1 - p);
}

// 伯努利对数 PMF：对数域实现，避免 p=0/1 时 log 发散（裁剪到 [1e-300,1-1e-300]）
function bernoulliLogPmf(k, p) {
  if (k !== 0 && k !== 1) throw new TypeError('bernoulliLogPmf: k 须为 0 或 1');
  if (!(typeof p === 'number' && p >= 0 && p <= 1)) throw new TypeError('bernoulliLogPmf: p 须 ∈ [0,1]');
  const pc = Math.min(Math.max(p, 1e-300), 1 - 1e-300);
  const qc = 1 - pc;
  return k * Math.log(pc) + (1 - k) * Math.log(qc);
}

// 高斯 KL 散度闭式：D(N0 || N1)
function klGaussian(m0, s0, m1, s1) {
  if (!(s0 > 0) || !(s1 > 0)) throw new TypeError('klGaussian: 标准差须为正');
  const s0sq = s0 * s0, s1sq = s1 * s1;
  return 0.5 * (Math.log(s1sq / s0sq) + (s0sq + (m0 - m1) * (m0 - m1)) / s1sq - 1);
}

// 伯努利 KL 散度：裁剪避免 log(0)
function klBernoulli(p, q) {
  if (!(p >= 0 && p <= 1 && q >= 0 && q <= 1)) throw new TypeError('klBernoulli: p,q 须 ∈ [0,1]');
  const pe = Math.min(Math.max(p, 1e-300), 1 - 1e-300);
  const qe = Math.min(Math.max(q, 1e-300), 1 - 1e-300);
  return pe * Math.log(pe / qe) + (1 - pe) * Math.log((1 - pe) / (1 - qe));
}

// 逆 CDF（Acklam）正态采样：mean + std * Φ⁻¹(u)
function sampleGaussian(mean, std, rng) {
  if (typeof rng !== 'function') throw new TypeError('sampleGaussian: rng 须为确定性 PRNG 函数');
  if (!(typeof std === 'number' && isFinite(std) && std >= 0)) throw new TypeError('sampleGaussian: std 须为非负实数');
  let u = rng();                 // [0,1)
  u = 1e-9 + (1 - 2e-9) * u;     // 映射到 (1e-9, 1-1e-9) 避免 normInv 对 0/1 抛错
  const z = normInv(u);
  return mean + std * z;
}

/* ============================================================================
 * selfCheck — 8 条不变量
 * ========================================================================== */
function selfCheck(opts) {
  opts = opts || {};
  const cases = [];
  const invariants = [];
  const rng = makeRng(0x9e3779b9);

  // INV-PRB-001 gaussianLogPdf vs 闭式（先算 exp 再 log 的独立重实现）
  {
    let maxDev = 0;
    for (let i = 0; i < 12; i++) {
      const sigma = 0.5 + rng() * 2.5;
      const mu = rng() * 6 - 3;
      const x = rng() * 10 - 5;
      const z = (x - mu) / sigma;
      const ref = Math.log(Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI)));
      const got = gaussianLogPdf(x, mu, sigma);
      maxDev = Math.max(maxDev, Math.abs(got - ref));
    }
    cases.push(caseResult('INV-PRB-001', 'gaussianLogPdf 与闭式一致', maxDev, 0, TOL.RESID));
  }
  // INV-PRB-002 同分布 KL=0
  {
    const m = rng() * 4 - 2, s = 0.5 + rng() * 2;
    const v = klGaussian(m, s, m, s);
    cases.push(caseResult('INV-PRB-002', 'klGaussian(m,s,m,s)==0', Math.abs(v), 0, TOL.RESID));
  }
  // INV-PRB-003 KL 非负性（10 组随机对）
  {
    let worst = 0;
    for (let i = 0; i < 10; i++) {
      const m0 = rng() * 4 - 2, s0 = 0.5 + rng() * 1.5;
      const m1 = rng() * 4 - 2, s1 = 0.5 + rng() * 1.5;
      const v = klGaussian(m0, s0, m1, s1);
      worst = Math.max(worst, -v); // 通过条件：-v <= 1e-12
    }
    cases.push(caseResult('INV-PRB-003', 'klGaussian ≥ 0（10 组）', worst, 0, TOL.RESID));
  }
  // INV-PRB-004 KL 主实现 vs 内联参考（用 2*log(s1/s0) 独立重推导 log(s1²/s0²)）
  {
    let maxDev = 0;
    for (let i = 0; i < 10; i++) {
      const m0 = rng() * 4 - 2, s0 = 0.5 + rng() * 1.5;
      const m1 = rng() * 4 - 2, s1 = 0.5 + rng() * 1.5;
      const got = klGaussian(m0, s0, m1, s1);
      const ref = 0.5 * (2 * Math.log(s1 / s0) + (s0 * s0 + (m0 - m1) * (m0 - m1)) / (s1 * s1) - 1);
      maxDev = Math.max(maxDev, Math.abs(got - ref));
    }
    cases.push(caseResult('INV-PRB-004', 'klGaussian 与内联参考一致', maxDev, 0, TOL.RESID));
  }
  // INV-PRB-005 伯努利归一
  {
    let maxDev = 0;
    for (let i = 0; i < 10; i++) {
      const p = 0.05 + rng() * 0.9;
      const sum = bernoulliPmf(1, p) + bernoulliPmf(0, p);
      maxDev = Math.max(maxDev, Math.abs(sum - 1));
    }
    cases.push(caseResult('INV-PRB-005', 'bernoulliPmf 归一', maxDev, 0, TOL.RESID));
  }
  // INV-PRB-006 对数 PMF 定义
  {
    let maxDev = 0;
    for (let i = 0; i < 10; i++) {
      const p = 0.05 + rng() * 0.9;
      const got = bernoulliLogPmf(1, p);
      maxDev = Math.max(maxDev, Math.abs(got - Math.log(p)));
    }
    cases.push(caseResult('INV-PRB-006', 'bernoulliLogPmf(1,p)==log(p)', maxDev, 0, TOL.RESID));
  }
  // INV-PRB-007 采样矩收敛（N=20000，参数选取使 MCSE 远小于 0.05）
  {
    const configs = [[-1, 0.8], [0, 1.0], [2, 1.0], [0.5, 0.8], [-0.5, 1.0]];
    const N = 20000;
    let worst = 0;
    for (const [mu, sigma] of configs) {
      const r = makeRng(0x1234 + Math.floor((mu + 10) * 100) + Math.floor(sigma * 7));
      let s1 = 0, s2 = 0;
      for (let i = 0; i < N; i++) {
        const x = sampleGaussian(mu, sigma, r);
        s1 += x; s2 += x * x;
      }
      const mean = s1 / N;
      const varr = s2 / N - mean * mean;
      worst = Math.max(worst, Math.abs(mean - mu), Math.abs(varr - sigma * sigma));
    }
    cases.push(caseResult('INV-PRB-007', 'sampleGaussian 矩收敛 (N=20000)', worst, 0, 0.05));
  }
  // INV-PRB-008 伯努利 KL 主实现 vs 内联参考 + 自反性
  {
    let maxDev = 0;
    for (let i = 0; i < 10; i++) {
      const p = 0.05 + rng() * 0.9, q = 0.05 + rng() * 0.9;
      const got = klBernoulli(p, q);
      const ref = p * Math.log(p / q) + (1 - p) * Math.log((1 - p) / (1 - q));
      maxDev = Math.max(maxDev, Math.abs(got - ref), Math.abs(klBernoulli(p, p)));
    }
    cases.push(caseResult('INV-PRB-008', 'klBernoulli 与内联参考一致', maxDev, 0, TOL.RESID));
  }

  const started = Date.now();
  for (const id of ['INV-PRB-001', 'INV-PRB-002', 'INV-PRB-003', 'INV-PRB-004', 'INV-PRB-005', 'INV-PRB-006', 'INV-PRB-007', 'INV-PRB-008']) {
    const c = cases.find(x => x.id === id);
    invariants.push({ id, expr: id, tol: c.tol, worst: c.abs_err, status: c.status, note: '' });
  }
  return buildReport({ module: 'prob-forge', version: '1.0.0', layer: 'L5', cases, invariants, seed: '0x9E3779B9', started, tol: { 'INV-PRB-007': 0.05 } });
}

const API = {
  name: 'prob-forge', version: '1.0.0', layer: 'L5',
  gaussianLogPdf, bernoulliPmf, bernoulliLogPmf, klGaussian, klBernoulli, sampleGaussian, selfCheck
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeProb = API;

if (typeof process !== 'undefined' && process.argv[1] && /prob-forge\.js$/.test(process.argv[1]) && process.argv.includes('--self-check')) {
  const r = selfCheck({ quick: process.argv.includes('--quick') });
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : require('./core-verify.js').formatTableSafe(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
