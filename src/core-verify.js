/**
 * @forge-module core-verify
 * @layer L1
 * @version 1.0.0
 * @brief     验证内核：统一容差 TOL、assertClose、用例收集、报告构造（forge-selfcheck/1.0 schema + sha256 hash）、中心差分 gradCheck、轻量 property-based 测试器、人类可读表格
 * @reuse     fast-check|https://github.com/dubzzz/fast-check|MIT|D|property-based 范式（自写零依赖轻量版）
 * @reuse     IEEE 754|https://en.wikipedia.org/wiki/IEEE_754|公有域|D|ULP 与相对误差判定口径
 * @beyond    每个模块的统一自检出口；断言处禁止内联魔法数，一律引用 TOL；hash 用 core-numeric 自写 SHA-256 保证双端逐位一致
 * @invariant INV-VER-001|assertClose：|a-b| <= tol 或 rel <= tol 时判 pass，否则 fail|1e-12|判定口径本身
 * @invariant INV-VER-002|中心差分梯度 vs 解析梯度相对误差 <= GRAD(1e-6)，对 f(x)=x^2*sin(x) 成立|1e-6|梯度检验定义
 * @invariant INV-VER-003|元不变量：故意失败用例（2 vs 1，tol 1e-9）必须被标记为 fail（验证框架自身正确）|0|自证能力
 * @invariant INV-VER-004|buildReport 聚合：passed+failed == cases.length，hash 仅覆盖 [id,status,actual]（排除 env/duration）|0|报告构造一致性
 * @seed      0x9E3779B9
 */
'use strict';

const TOL = { EXACT: 0, ULP: 1e-16, TIGHT: 1e-15, ORTHO: 1e-14, RESID: 1e-12, SOLVE: 1e-10, GRAD: 1e-6, MONO: -1e-9, CONV: 1e-6 };

// 惰性取 sha256：优先 core-numeric 的纯 JS FIPS 180-4 实现。
// Node 端用 require；浏览器端用 globalThis.ForgeNumeric（同源纯 JS 实现）→ 双端 hash 逐位一致。
// 二者皆缺失时才降级 FNV-1a 占位（仅极端环境）。
let _sha = null;
function _sha256(s) {
  if (_sha === null) {
    let fn = null;
    if (typeof require !== 'undefined') { try { fn = require('./core-numeric.js').sha256; } catch (e) { fn = null; } }
    if (!fn && typeof globalThis !== 'undefined' && globalThis.ForgeNumeric && globalThis.ForgeNumeric.sha256) fn = globalThis.ForgeNumeric.sha256;
    _sha = fn || false;
  }
  if (_sha) return _sha(s);
  let h = 2166136261 >>> 0; // FNV-1a 32-bit 占位（非密码级，仅兜底）
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; }
  return ('00000000' + h.toString(16)).slice(-8);
}

function assertClose(actual, expected, tol, opts) {
  tol = (tol === undefined || tol === null) ? TOL.RESID : tol;
  const abs = Math.abs(actual - expected);
  const denom = Math.abs(expected) > 1e-300 ? Math.abs(expected) : 1;
  const rel = abs / denom;
  const pass = tol === 0 ? (actual === expected) : (abs <= Math.abs(tol) || rel <= Math.abs(tol));
  return { status: pass ? 'pass' : 'fail', abs_err: abs, rel_err: rel, tol };
}

function caseResult(id, name, actual, expected, tol, note) {
  const a = assertClose(actual, expected, tol);
  return { id, name, status: a.status, expected, actual, abs_err: a.abs_err, rel_err: a.rel_err, tol, note: note || '' };
}

/** 中心差分梯度检验：f 标量函数，df 解析梯度。返回最大相对误差。 */
function gradCheck(f, df, x, eps) {
  eps = eps || 1e-6;
  const fpx = (f(x + eps) - f(x - eps)) / (2 * eps);
  const ap = df(x);
  const denom = Math.abs(ap) > 1e-300 ? Math.abs(ap) : 1;
  return Math.abs(fpx - ap) / denom;
}

/** 轻量 property-based：gen(i) 生成输入，prop(input) 返回 bool，跑 n 次。 */
function forAll(gen, prop, n) {
  n = n || 200;
  let allPass = true;
  for (let i = 0; i < n; i++) { if (!prop(gen(i))) { allPass = false; break; } }
  return { allPass, trials: n };
}

/** 组装 forge-selfcheck/1.0 报告。opts: {module,version,layer,cases,invariants,seed,tol} */
function buildReport(opts) {
  const cases = opts.cases || [];
  const invariants = opts.invariants || [];
  const passed = cases.filter(c => c.status === 'pass').length;
  const failed = cases.length - passed;
  const started = opts.started || Date.now();
  let hashSrc = '';
  for (const c of cases) hashSrc += c.id + '|' + c.status + '|' + Math.round(c.actual * 1e12) + '\n';
  return {
    schema: 'forge-selfcheck/1.0', module: opts.module, version: opts.version || '1.0.0', layer: opts.layer || 'L1',
    passed, failed, skipped: opts.skipped || 0, duration_ms: Date.now() - started,
    tol: opts.tol || {}, invariants, cases,
    env: { platform: typeof process !== 'undefined' && process.platform || 'browser', arch: typeof process !== 'undefined' && process.arch || 'unknown', node: typeof process !== 'undefined' && process.version || 'browser', seed: opts.seed || '0x9E3779B9', started_at: new Date(started).toISOString() },
    hash: 'sha256:' + _sha256(hashSrc)
  };
}

function formatTableSafe(r) {
  if (!r) return 'NO REPORT';
  const lines = [];
  lines.push('=== ' + r.module + ' v' + r.version + ' (' + r.layer + ') ' + (r.failed === 0 ? 'PASS' : 'FAIL') + ' ===');
  lines.push('passed ' + r.passed + ' / failed ' + r.failed + ' / hash ' + (r.hash || '').slice(0, 24));
  for (const c of (r.cases || [])) {
    const mark = c.status === 'pass' ? '+OK' : '-XX';
    lines.push('  ' + mark + ' ' + c.id + ' ' + c.name + '  act=' + (typeof c.actual === 'number' ? c.actual.toExponential(3) : c.actual) + ' tol=' + c.tol);
  }
  return lines.join('\n');
}

function selfCheck(opts) {
  opts = opts || {};
  const cases = [];
  const invariants = [];
  // INV-VER-001 assertClose 判定口径（容差内 pass / 容差外 fail；探针不计入 cases）
  {
    const a1 = caseResult('INV-VER-001-probe-pass', '容差内 pass', 0.1 + 0.2, 0.3, TOL.RESID);
    const a2 = caseResult('INV-VER-001-probe-fail', '容差外 fail', 2, 1, TOL.ULP);
    invariants.push({ id: 'INV-VER-001', expr: '|a-b| <= tol 或 rel <= tol => pass', tol: TOL.TIGHT, worst: Math.max(a1.abs_err, a2.abs_err), status: (a1.status === 'pass' && a2.status === 'fail') ? 'pass' : 'fail', note: '容差内 pass / 容差外 fail' });
  }
  // INV-VER-002 gradCheck（固定网格，去 rng 依赖）
  {
    const f = (x) => x * x * Math.sin(x);
    const df = (x) => 2 * x * Math.sin(x) + x * x * Math.cos(x);
    let mx = 0;
    for (let i = 0; i < 40; i++) { const x = -8 + (16 * i) / 39; const e = gradCheck(f, df, x); if (e > mx) mx = e; }
    cases.push(caseResult('INV-VER-002', 'gradCheck 最大相对误差', mx, 0, TOL.GRAD));
    invariants.push({ id: 'INV-VER-002', expr: 'max|中心差分-解析|/|解析| <= 1e-6', tol: TOL.GRAD, worst: mx, status: mx <= TOL.GRAD ? 'pass' : 'fail', note: 'f=x^2*sin(x)' });
  }
  // INV-VER-003 元不变量：故意失败用例必须被标记 fail（探针不计入 failed 计数）
  {
    const bad = caseResult('INV-VER-003-probe', '故意失败用例被标记 fail', 2, 1, TOL.ULP); // 不 push 到 cases
    const ok = bad.status === 'fail';
    invariants.push({ id: 'INV-VER-003', expr: '故意失败用例 status===fail（验证框架自身）', tol: 0, worst: 0, status: ok ? 'pass' : 'fail', note: '探针 case 不计入 failed' });
  }
  // INV-VER-004 buildReport 聚合一致性
  {
    const sample = [caseResult('x', 'p', 1, 1, 1e-6), caseResult('y', 'f', 2, 1, TOL.ULP), caseResult('z', 'p', 3, 3, 1e-6)];
    const rep = buildReport({ module: 'meta', version: '1.0.0', layer: 'L1', cases: sample, seed: '0x9E3779B9' });
    const ok = (rep.passed + rep.failed === sample.length) && (rep.passed === 2) && (typeof rep.hash === 'string' && rep.hash.indexOf('sha256:') === 0);
    cases.push(caseResult('INV-VER-004', 'buildReport 聚合一致', ok ? 1 : 0, 1, 0));
    invariants.push({ id: 'INV-VER-004', expr: 'passed+failed==len 且 hash 仅覆盖 [id,status,actual]', tol: 0, worst: 0, status: ok ? 'pass' : 'fail', note: '排除 env/duration' });
  }
  const started = Date.now();
  return buildReport({ module: 'core-verify', version: '1.0.0', layer: 'L1', cases, invariants, seed: '0x9E3779B9', started });
}

const API = {
  name: 'core-verify', version: '1.0.0', layer: 'L1', TOL, assertClose, caseResult, gradCheck, forAll, buildReport, formatTableSafe, selfCheck
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeVerify = API;

if (typeof process !== 'undefined' && process.argv[1] && /core-verify\.js$/.test(process.argv[1]) && process.argv.includes('--self-check')) {
  const r = selfCheck({ quick: process.argv.includes('--quick') });
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : formatTableSafe(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
