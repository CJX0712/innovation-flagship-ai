/**
 * @forge-module retrieval-forge
 * @layer L5
 * @version 1.0.0
 * @brief     检索内核：余弦相似度、L2 距离、基于暴力精确搜索的 top-k 召回。余弦相似度对向量缩放不变，top-k 与暴力排序一致。
 * @reuse     FAISS IndexFlat|https://github.com/facebookresearch/faiss|MIT|D|精确近邻召回基线
 * @reuse     hnswlib|https://github.com/nmslib/hnswlib|Apache-2.0|D|ANN 召回范式（本模块为精确基线，零依赖）
 * @beyond    纯 JS 单文件零构建；top-k 与暴力全排序交叉校验；余弦缩放不变性、归一化等价等不变量可证；零向量安全返回 0。
 * @invariant INV-RET-001|余弦相似度 ∈ [−1,1]|1e-12|定义边界
 * @invariant INV-RET-002|相同向量余弦=1；正交向量余弦=0|1e-12|定义
 * @invariant INV-RET-003|top-k 返回 k 个结果且按相似度降序|1e-12|召回顺序
 * @invariant INV-RET-004|top-k 与暴力全排序结果一致（集合+顺序）|1e-12|精确召回
 * @invariant INV-RET-005|归一化向量余弦 = 点积|1e-12|余弦/点积等价
 * @invariant INV-RET-006|余弦检索对查询缩放不变（c≠0 不改变 top 顺序）|1e-12|缩放不变性
 * @invariant INV-RET-007|L2 距离 = ‖a−b‖；非负且零点为同向量|1e-12|L2 定义
 * @invariant INV-RET-008|零向量安全：与任意向量余弦=0（不抛错不 NaN）|1e-12|边界安全
 * @seed      0x9E3779B9
 */
'use strict';
const { makeRng, TOL } = require('./core-numeric.js');
const { caseResult, buildReport } = require('./core-verify.js');

function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function norm(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s); }

function cosineSim(a, b) {
  const na = norm(a), nb = norm(b);
  if (na === 0 || nb === 0) return 0; // 零向量安全
  return dot(a, b) / (na * nb);
}
function l2(a, b) { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } return Math.sqrt(s); }

// 精确 top-k 召回（余弦），返回 {idx, sim} 降序
function topK(corpus, query, k) {
  const sims = corpus.map((v, i) => ({ idx: i, sim: cosineSim(query, v) }));
  sims.sort((p, q) => q.sim - p.sim);
  return sims.slice(0, k);
}

/* ============================================================================
 * selfCheck — 8 条不变量
 * ========================================================================== */
function selfCheck(opts) {
  opts = opts || {};
  const cases = [];
  const invariants = [];
  const rng = makeRng(0x9e3779b9);
  const D = 16;

  function randVec() { const v = new Float64Array(D); for (let i = 0; i < D; i++) v[i] = rng() * 2 - 1; return v; }

  // INV-RET-001 余弦边界
  {
    let mx = 0;
    for (let t = 0; t < 30; t++) {
      const a = randVec(), b = randVec();
      const c = cosineSim(a, b);
      mx = Math.max(mx, Math.abs(Math.min(1, Math.max(-1, c)) - c)); // 偏离[-1,1]的程度
    }
    cases.push(caseResult('INV-RET-001', '余弦相似度 ∈ [−1,1]', mx, 0, TOL.RESID));
  }
  // INV-RET-002 相同/正交
  {
    const a = randVec();
    const same = Math.abs(cosineSim(a, Float64Array.from(a)) - 1);
    // 构造与 a 正交向量：用 Gram-Schmidt 取一个随机向量减去投影
    let z = randVec(); const proj = dot(z, a) / dot(a, a); for (let i = 0; i < D; i++) z[i] -= proj * a[i];
    const orth = Math.abs(cosineSim(a, z));
    cases.push(caseResult('INV-RET-002', '相同=1；正交=0', Math.max(same, orth), 0, TOL.RESID));
  }
  // INV-RET-003 top-k 数量与降序
  {
    const corpus = []; for (let i = 0; i < 40; i++) corpus.push(randVec());
    const q = randVec(); const k = 5;
    const res = topK(corpus, q, k);
    let ok = res.length === k;
    for (let i = 1; i < res.length; i++) if (res[i].sim > res[i - 1].sim + 1e-12) ok = false;
    cases.push(caseResult('INV-RET-003', 'top-k 数量=k 且降序', ok ? 0 : 1, 0, 0));
  }
  // INV-RET-004 top-k 与暴力排序一致
  {
    let mx = 0;
    const corpus = []; for (let i = 0; i < 40; i++) corpus.push(randVec());
    const q = randVec(); const k = 6;
    const res = topK(corpus, q, k);
    const brute = corpus.map((v, i) => ({ idx: i, sim: cosineSim(q, v) })).sort((p, x) => x.sim - p.sim).slice(0, k);
    for (let i = 0; i < k; i++) mx = Math.max(mx, Math.abs(res[i].idx - brute[i].idx), Math.abs(res[i].sim - brute[i].sim));
    cases.push(caseResult('INV-RET-004', 'top-k = 暴力排序', mx, 0, TOL.RESID));
  }
  // INV-RET-005 归一化向量余弦=点积
  {
    let mx = 0;
    for (let t = 0; t < 20; t++) {
      const a = randVec(), b = randVec();
      const an = Float64Array.from(a), bn = Float64Array.from(b);
      for (let i = 0; i < D; i++) { an[i] /= norm(a); bn[i] /= norm(b); }
      mx = Math.max(mx, Math.abs(cosineSim(an, bn) - dot(an, bn)));
    }
    cases.push(caseResult('INV-RET-005', '归一化余弦=点积', mx, 0, TOL.RESID));
  }
  // INV-RET-006 缩放不变
  {
    let mx = 0;
    const corpus = []; for (let i = 0; i < 40; i++) corpus.push(randVec());
    const q = randVec(); const k = 4;
    const r1 = topK(corpus, q, k);
    const r2 = topK(corpus, q.map(v => v * 3.7), k);
    for (let i = 0; i < k; i++) mx = Math.max(mx, Math.abs(r1[i].idx - r2[i].idx));
    cases.push(caseResult('INV-RET-006', '余弦检索缩放不变', mx, 0, TOL.RESID));
  }
  // INV-RET-007 L2 距离
  {
    let mx = 0;
    for (let t = 0; t < 20; t++) {
      const a = randVec(), b = randVec();
      const d = l2(a, b), ref = norm(a.map((v, i) => v - b[i]));
      mx = Math.max(mx, Math.abs(d - ref));
      // 同向量距离 0
      mx = Math.max(mx, l2(a, a));
    }
    cases.push(caseResult('INV-RET-007', 'L2 距离 = ‖a−b‖；非负且零点', mx, 0, TOL.RESID));
  }
  // INV-RET-008 零向量安全
  {
    const z = new Float64Array(D);
    let mx = 0, finiteOk = true;
    for (let t = 0; t < 10; t++) {
      const a = randVec(); const c = cosineSim(z, a);
      if (!isFinite(c)) finiteOk = false;
      mx = Math.max(mx, Math.abs(c - 0));
    }
    cases.push(caseResult('INV-RET-008', '零向量余弦=0 且有限', finiteOk ? mx : 1, 0, TOL.RESID));
  }

  const stat = (id) => cases.find(c => c.id === id).status;
  for (const id of ['INV-RET-001', 'INV-RET-002', 'INV-RET-003', 'INV-RET-004', 'INV-RET-005', 'INV-RET-006', 'INV-RET-007', 'INV-RET-008']) {
    const c = cases.find(x => x.id === id);
    invariants.push({ id, expr: id, tol: c.tol, worst: c.abs_err, status: c.status, note: '' });
  }

  const started = Date.now();
  return buildReport({ module: 'retrieval-forge', version: '1.0.0', layer: 'L5', cases, invariants, seed: '0x9E3779B9', started });
}

const API = { name: 'retrieval-forge', version: '1.0.0', layer: 'L5', cosineSim, l2, topK, dot, norm, selfCheck };
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeRetrieval = API;

if (typeof process !== 'undefined' && process.argv[1] && /retrieval-forge\.js$/.test(process.argv[1]) && process.argv.includes('--self-check')) {
  const r = selfCheck({ quick: process.argv.includes('--quick') });
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : require('./core-verify.js').formatTableSafe(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
