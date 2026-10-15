/**
 * @forge-module rope-forge
 * @layer L4
 * @version 1.0.0
 * @brief     旋转位置编码 (Rotary Position Embedding, RoPE)：对词向量按维度配对做旋转，旋转角 θ_i = pos·base^{−2i/d}。旋转为正交变换，保持范数/距离并具相对位置不变性。
 * @reuse     Su et al. RoFormer/RoPE|https://arxiv.org/abs/2104.09864|公有域|D|RoPE 定义与相对位置性质
 * @reuse     rotary-embedding|https://github.com/lucidrains/rotary-embedding-torch|MIT|D|按对旋转实现
 * @beyond    纯 JS 单文件零构建；正交性（范数/距离保持）、逆变换、角度可加性、相对位置不变性等不变量可证；2D 显式旋转公式交叉校验。
 * @invariant INV-ROPE-001|范数保持：‖RoPE(x,p)‖ = ‖x‖ ≤ 1e-15|1e-15|旋转正交
 * @invariant INV-ROPE-002|逆变换：RoPE(RoPE(x,p),−p) = x ≤ 1e-12|1e-12|旋转可逆
 * @invariant INV-ROPE-003|2D 显式旋转公式交叉校验（x0cosθ−x1sinθ, x0sinθ+x1cosθ）≤ 1e-12|1e-12|定义对照
 * @invariant INV-ROPE-004|相对位置不变：⟨RoPE(q,m),RoPE(k,n)⟩ 仅依赖 (m−n)，与绝对位置无关 ≤ 1e-12|1e-12|RoPE 核心性质
 * @invariant INV-ROPE-005|任意 base 下 2D 配对角 = pos·base^{−2i/d} 显式校验 ≤ 1e-12|1e-12|角度公式
 * @invariant INV-ROPE-006|距离保持：‖RoPE(q,p)−RoPE(k,p)‖ = ‖q−k‖ ≤ 1e-12|1e-12|正交变换
 * @invariant INV-ROPE-007|角度可加：RoPE(RoPE(x,p),p) = RoPE(x,2p) ≤ 1e-12|1e-12|旋转角叠加
 * @invariant INV-ROPE-008|线性：RoPE(−x,p) = −RoPE(x,p) ≤ 1e-12|1e-12|旋转线性
 * @seed      0x9E3779B9
 */
'use strict';
const { makeRng, TOL } = require('./core-numeric.js');
const { caseResult, buildReport } = require('./core-verify.js');

// RoPE：对向量 x（长度 d，偶数）按配对旋转；pos 为位置，base 默认 10000
function rope(x, pos, base) {
  base = base || 10000;
  const d = x.length;
  const out = new Float64Array(d);
  const half = d / 2;
  for (let i = 0; i < half; i++) {
    const theta = pos * Math.pow(base, -2 * i / d);
    const c = Math.cos(theta), s = Math.sin(theta);
    const a = x[2 * i], b = x[2 * i + 1];
    out[2 * i] = a * c - b * s;
    out[2 * i + 1] = a * s + b * c;
  }
  return out;
}

function norm(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s); }
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

/* ============================================================================
 * selfCheck — 8 条不变量
 * ========================================================================== */
function selfCheck(opts) {
  opts = opts || {};
  const cases = [];
  const invariants = [];
  const rng = makeRng(0x9e3779b9);

  // INV-ROPE-001 范数保持
  {
    let mx = 0;
    for (let t = 0; t < 20; t++) {
      const d = 8; const x = new Float64Array(d); for (let i = 0; i < d; i++) x[i] = rng() * 2 - 1;
      const p = Math.floor(rng() * 20) - 10;
      mx = Math.max(mx, Math.abs(norm(rope(x, p)) - norm(x)));
    }
    cases.push(caseResult('INV-ROPE-001', '范数保持 ‖RoPE‖=‖x‖', mx, 0, 1e-15));
  }
  // INV-ROPE-002 逆变换
  {
    let mx = 0;
    for (let t = 0; t < 20; t++) {
      const d = 8; const x = new Float64Array(d); for (let i = 0; i < d; i++) x[i] = rng() * 2 - 1;
      const p = Math.floor(rng() * 20) - 10;
      const back = rope(rope(x, p), -p);
      let dev = 0; for (let i = 0; i < d; i++) dev = Math.max(dev, Math.abs(back[i] - x[i]));
      mx = Math.max(mx, dev);
    }
    cases.push(caseResult('INV-ROPE-002', '逆变换 RoPE(RoPE(x,p),−p)=x', mx, 0, TOL.RESID));
  }
  // INV-ROPE-003 2D 显式旋转公式
  {
    let mx = 0;
    for (let t = 0; t < 20; t++) {
      const a = rng() * 2 - 1, b = rng() * 2 - 1, pos = Math.floor(rng() * 12);
      const x = new Float64Array([a, b]);
      const out = rope(x, pos, 10000);
      const theta = pos * Math.pow(10000, 0); // 2i/d, i=0 → 指数 0
      const ra = a * Math.cos(theta) - b * Math.sin(theta);
      const rb = a * Math.sin(theta) + b * Math.cos(theta);
      mx = Math.max(mx, Math.abs(out[0] - ra), Math.abs(out[1] - rb));
    }
    cases.push(caseResult('INV-ROPE-003', '2D 显式旋转公式', mx, 0, TOL.RESID));
  }
  // INV-ROPE-004 相对位置不变：⟨RoPE(q,m),RoPE(k,n)⟩ 仅依赖 (m−n)
  {
    let mx = 0;
    for (let t = 0; t < 20; t++) {
      const d = 8; const q = new Float64Array(d), k = new Float64Array(d);
      for (let i = 0; i < d; i++) { q[i] = rng() * 2 - 1; k[i] = rng() * 2 - 1; }
      const m = 2, n = 7, delta = n - m;
      const m2 = 11, n2 = 11 + delta;
      const dot1 = dot(rope(q, m), rope(k, n));
      const dot2 = dot(rope(q, m2), rope(k, n2));
      mx = Math.max(mx, Math.abs(dot1 - dot2));
    }
    cases.push(caseResult('INV-ROPE-004', '相对位置不变 ⟨q_m,k_n⟩ 仅赖 m−n', mx, 0, TOL.RESID));
  }
  // INV-ROPE-005 任意 base 下 2D 配对角校验
  {
    let mx = 0;
    const base = 137; const d = 8;
    for (let t = 0; t < 20; t++) {
      const x = new Float64Array(d); for (let i = 0; i < d; i++) x[i] = rng() * 2 - 1;
      const pos = Math.floor(rng() * 10);
      const out = rope(x, pos, base);
      for (let i = 0; i < d / 2; i++) {
        const theta = pos * Math.pow(base, -2 * i / d);
        const c = Math.cos(theta), s = Math.sin(theta);
        const ra = x[2 * i] * c - x[2 * i + 1] * s;
        const rb = x[2 * i] * s + x[2 * i + 1] * c;
        mx = Math.max(mx, Math.abs(out[2 * i] - ra), Math.abs(out[2 * i + 1] - rb));
      }
    }
    cases.push(caseResult('INV-ROPE-005', '任意 base 配对角显式校验', mx, 0, TOL.RESID));
  }
  // INV-ROPE-006 距离保持
  {
    let mx = 0;
    for (let t = 0; t < 20; t++) {
      const d = 8; const q = new Float64Array(d), k = new Float64Array(d);
      for (let i = 0; i < d; i++) { q[i] = rng() * 2 - 1; k[i] = rng() * 2 - 1; }
      const p = Math.floor(rng() * 15) - 7;
      const d1 = norm(rope(q, p)) - norm(q); // 范数保持（副）
      let qk = 0; for (let i = 0; i < d; i++) qk += (q[i] - k[i]) * (q[i] - k[i]);
      const rq = rope(q, p), rk = rope(k, p); let qk2 = 0; for (let i = 0; i < d; i++) qk2 += (rq[i] - rk[i]) * (rq[i] - rk[i]);
      mx = Math.max(mx, Math.abs(d1), Math.abs(Math.sqrt(qk2) - Math.sqrt(qk)));
    }
    cases.push(caseResult('INV-ROPE-006', '距离保持 ‖RoPE(q)−RoPE(k)‖=‖q−k‖', mx, 0, TOL.RESID));
  }
  // INV-ROPE-007 角度可加
  {
    let mx = 0;
    for (let t = 0; t < 20; t++) {
      const d = 8; const x = new Float64Array(d); for (let i = 0; i < d; i++) x[i] = rng() * 2 - 1;
      const p = Math.floor(rng() * 8);
      const twice = rope(rope(x, p), p);
      const ref = rope(x, 2 * p);
      let dev = 0; for (let i = 0; i < d; i++) dev = Math.max(dev, Math.abs(twice[i] - ref[i]));
      mx = Math.max(mx, dev);
    }
    cases.push(caseResult('INV-ROPE-007', '角度可加 RoPE(RoPE(x,p),p)=RoPE(x,2p)', mx, 0, TOL.RESID));
  }
  // INV-ROPE-008 线性
  {
    let mx = 0;
    for (let t = 0; t < 20; t++) {
      const d = 8; const x = new Float64Array(d); for (let i = 0; i < d; i++) x[i] = rng() * 2 - 1;
      const p = Math.floor(rng() * 12) - 6;
      const neg = rope(x.map(v => -v), p);
      const pos = rope(x, p);
      let dev = 0; for (let i = 0; i < d; i++) dev = Math.max(dev, Math.abs(neg[i] + pos[i]));
      mx = Math.max(mx, dev);
    }
    cases.push(caseResult('INV-ROPE-008', '线性 RoPE(−x,p)=−RoPE(x,p)', mx, 0, TOL.RESID));
  }

  const stat = (id) => cases.find(c => c.id === id).status;
  for (const id of ['INV-ROPE-001', 'INV-ROPE-002', 'INV-ROPE-003', 'INV-ROPE-004', 'INV-ROPE-005', 'INV-ROPE-006', 'INV-ROPE-007', 'INV-ROPE-008']) {
    const c = cases.find(x => x.id === id);
    invariants.push({ id, expr: id, tol: c.tol, worst: c.abs_err, status: c.status, note: '' });
  }

  const started = Date.now();
  return buildReport({ module: 'rope-forge', version: '1.0.0', layer: 'L4', cases, invariants, seed: '0x9E3779B9', started, tol: { 'INV-ROPE-001': 1e-15 } });
}

const API = { name: 'rope-forge', version: '1.0.0', layer: 'L4', rope, norm, dot, selfCheck };
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeRoPE = API;

if (typeof process !== 'undefined' && process.argv[1] && /rope-forge\.js$/.test(process.argv[1]) && process.argv.includes('--self-check')) {
  const r = selfCheck({ quick: process.argv.includes('--quick') });
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : require('./core-verify.js').formatTableSafe(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
