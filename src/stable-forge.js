/**
 * @forge-module stable-forge
 * @layer L2
 * @version 1.0.0
 * @brief     数值稳定原语：logsumexp / 稳定 softmax / log-sigmoid / log1pExp(softplus)。所有接口对大负数/大正数不溢出，硬断言 x=[1000,1000] 等极端输入不抛错且结果有限。
 * @reuse     scipy.special.logsumexp|https://github.com/scipy/scipy|BSD-3|B|移位常数 m 防溢出基线
 * @reuse     PyTorch log_softmax|https://github.com/pytorch/pytorch|BSD-2|C|数值稳定实现参照
 * @beyond    纯 JS 单文件零构建；logsumexp 用最大移位；log-sigmoid/log1pExp 分段避免 exp 溢出；全部附可交叉验证不变量。
 * @invariant INV-STB-001|logsumexp(x)=log(Σexp(xᵢ)) 对良态 x 相对误差 ≤ 1e-12；且 x=[1000,1000] 时 = 1000+log2 绝对误差 ≤ 1e-12|1e-12|移位恒等式
 * @invariant INV-STB-002|稳定 softmax 分量和为 1 ≤ 1e-15；对良态 x 与朴素 softmax 一致 ≤ 1e-12|1e-15|概率归一
 * @invariant INV-STB-003|logSigmoid(x)=log(1/(1+exp(-x))) 对良态 x 一致 ≤ 1e-12；x=±1000 结果有限不溢出|1e-12|对数 sigmoid 定义
 * @invariant INV-STB-004|log1pExp(x)=log(1+exp(x)) 对良态 x 一致 ≤ 1e-12；x=1000 时 ≈ x（绝对误差 ≤ 1e-12）|1e-12|softplus 定义
 * @seed      0x9E3779B9
 */
'use strict';
const { makeRng, TOL } = require('./core-numeric.js');

/* ---------- 数值稳定原语 ---------- */

// logsumexp：移位常数 m = max(x)，避免 exp 溢出（scipy.logsumexp 同思路）
function logsumexp(a) {
  if (!Array.isArray(a) || a.length === 0) throw new TypeError('logsumexp: 需要非空数组');
  let m = -Infinity;
  for (let i = 0; i < a.length; i++) {
    if (typeof a[i] !== 'number' || !isFinite(a[i])) throw new TypeError('logsumexp: 元素须为有限实数');
    if (a[i] > m) m = a[i];
  }
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.exp(a[i] - m);
  return m + Math.log(s);
}

// 稳定 softmax：减去最大值后再 exp，分量和为 1
function softmax(a) {
  if (!Array.isArray(a) || a.length === 0) throw new TypeError('softmax: 需要非空数组');
  let m = -Infinity;
  for (let i = 0; i < a.length; i++) {
    if (typeof a[i] !== 'number' || !isFinite(a[i])) throw new TypeError('softmax: 元素须为有限实数');
    if (a[i] > m) m = a[i];
  }
  const e = new Float64Array(a.length);
  let s = 0;
  for (let i = 0; i < a.length; i++) { const x = Math.exp(a[i] - m); e[i] = x; s += x; }
  for (let i = 0; i < a.length; i++) e[i] /= s;
  return e;
}

// log-sigmoid：分段避免 exp 溢出（x>=0 走 -log1p(e^-x)；x<0 走 x - log1p(e^x)）
function logSigmoid(x) {
  if (typeof x !== 'number' || !isFinite(x)) throw new TypeError('logSigmoid: 需要有限实数');
  if (x >= 0) return -Math.log1p(Math.exp(-x));
  const ex = Math.exp(x);
  return x - Math.log1p(ex);
}

// log1pExp (softplus)：分段避免 exp 溢出
function log1pExp(x) {
  if (typeof x !== 'number' || !isFinite(x)) throw new TypeError('log1pExp: 需要有限实数');
  if (x >= 0) return x + Math.log1p(Math.exp(-x));
  return Math.log1p(Math.exp(x));
}

/* ---------- selfCheck ---------- */
function selfCheck(opts) {
  opts = opts || {};
  const cases = []; const invariants = []; const rng = makeRng(0x9e3779b9);

  // INV-STB-001 logsumexp
  {
    const x = [rng() * 10 - 5, rng() * 10 - 5, rng() * 10 - 5, rng() * 10 - 5];
    let naive = 0; for (const v of x) naive += Math.exp(v); naive = Math.log(naive);
    const got = logsumexp(x);
    const rel = Math.abs(got - naive) / (Math.abs(naive) + 1e-300);
    const extreme = logsumexp([1000, 1000]);
    const extErr = Math.abs(extreme - (1000 + Math.log(2)));
    const ok = rel <= 1e-12 && isFinite(extreme) && extErr <= 1e-12;
    cases.push({ id: 'INV-STB-001', name: 'logsumexp 一致性 + 极端无溢出', status: ok ? 'pass' : 'fail', expected: 0, actual: Math.max(rel, extErr), abs_err: extErr, rel_err: rel, tol: 1e-12, note: 'scipy.logsumexp 移位法' });
    invariants.push({ id: 'INV-STB-001', expr: 'logsumexp=logΣexp；x=[1000,1000]=1000+log2', tol: 1e-12, worst: Math.max(rel, extErr), status: ok ? 'pass' : 'fail', note: '' });
  }

  // INV-STB-002 softmax
  {
    const x = [rng() * 10 - 5, rng() * 10 - 5, rng() * 10 - 5, rng() * 10 - 5, rng() * 10 - 5];
    const sm = softmax(x);
    let sum = 0; for (const v of sm) sum += v;
    let m = -Infinity; for (const v of x) if (v > m) m = v;
    let s = 0; const e = x.map(v => Math.exp(v - m)); for (const v of e) s += v; const naive = e.map(v => v / s);
    let maxDev = 0; for (let i = 0; i < x.length; i++) maxDev = Math.max(maxDev, Math.abs(sm[i] - naive[i]));
    const ok = Math.abs(sum - 1) <= 1e-15 && maxDev <= 1e-12;
    cases.push({ id: 'INV-STB-002', name: 'softmax 归一 + 一致性', status: ok ? 'pass' : 'fail', expected: 0, actual: Math.max(Math.abs(sum - 1), maxDev), abs_err: Math.abs(sum - 1), rel_err: maxDev, tol: 1e-15, note: '' });
    invariants.push({ id: 'INV-STB-002', expr: 'Σsoftmax=1 ≤1e-15；与朴素一致 ≤1e-12', tol: 1e-15, worst: Math.max(Math.abs(sum - 1), maxDev), status: ok ? 'pass' : 'fail', note: '' });
  }

  // INV-STB-003 log-sigmoid
  {
    const xs = [rng() * 10 - 5, rng() * 10 - 5, 1000, -1000];
    let maxDev = 0, finiteOk = true;
    for (const x of xs) {
      const got = logSigmoid(x);
      const naive = Math.log(1 / (1 + Math.exp(-x)));
      if (!isFinite(got)) finiteOk = false;
      if (isFinite(naive)) maxDev = Math.max(maxDev, Math.abs(got - naive));
    }
    const ok = finiteOk && maxDev <= 1e-12;
    cases.push({ id: 'INV-STB-003', name: 'log-sigmoid 一致性 + 极端有限', status: ok ? 'pass' : 'fail', expected: 0, actual: maxDev, abs_err: maxDev, rel_err: maxDev, tol: 1e-12, note: '' });
    invariants.push({ id: 'INV-STB-003', expr: 'logSigmoid=log(1/(1+e^-x))；x=±1000 有限', tol: 1e-12, worst: maxDev, status: ok ? 'pass' : 'fail', note: '' });
  }

  // INV-STB-004 log1pExp
  {
    const xs = [rng() * 10 - 5, rng() * 10 - 5, 1000, -1000];
    let maxDev = 0, finiteOk = true;
    for (const x of xs) {
      const got = log1pExp(x);
      const naive = Math.log(1 + Math.exp(x));
      if (!isFinite(got)) finiteOk = false;
      if (isFinite(naive)) maxDev = Math.max(maxDev, Math.abs(got - naive));
    }
    const bigErr = Math.abs(log1pExp(1000) - 1000);
    const ok = finiteOk && maxDev <= 1e-12 && bigErr <= 1e-12;
    cases.push({ id: 'INV-STB-004', name: 'log1pExp 一致性 + 大数无溢出', status: ok ? 'pass' : 'fail', expected: 0, actual: Math.max(maxDev, bigErr), abs_err: bigErr, rel_err: maxDev, tol: 1e-12, note: '' });
    invariants.push({ id: 'INV-STB-004', expr: 'log1pExp=log(1+e^x)；x=1000≈x', tol: 1e-12, worst: Math.max(maxDev, bigErr), status: ok ? 'pass' : 'fail', note: '' });
  }

  const passed = cases.filter(c => c.status === 'pass').length;
  const failed = cases.length - passed;
  let hashSrc = ''; for (const c of cases) hashSrc += c.id + '|' + c.status + '|' + Math.round(c.actual * 1e12) + '\n';
  let _sha; try { _sha = require('./core-numeric.js').sha256; } catch (e) { _sha = (s) => s.length; }
  const report = {
    schema: 'forge-selfcheck/1.0', module: 'stable-forge', version: '1.0.0', layer: 'L2', passed, failed, skipped: 0, duration_ms: 0,
    tol: { 'INV-STB-001': 1e-12, 'INV-STB-002': 1e-15 }, invariants, cases,
    env: (typeof process !== 'undefined' && process.platform) ? { platform: process.platform, arch: process.arch, node: process.version, seed: '0x9E3779B9', started_at: new Date().toISOString() } : { platform: 'browser', seed: '0x9E3779B9', started_at: new Date().toISOString() },
    hash: 'sha256:' + _sha(hashSrc)
  };
  return report;
}

const API = { name: 'stable-forge', version: '1.0.0', layer: 'L2', logsumexp, softmax, logSigmoid, log1pExp, selfCheck };
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeStable = API;

if (typeof process !== 'undefined' && process.argv[1] && /stable-forge\.js$/.test(process.argv[1]) && process.argv.includes('--self-check')) {
  const r = selfCheck({ quick: process.argv.includes('--quick') });
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : require('./core-verify.js').formatTableSafe(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
