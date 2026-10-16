/**
 * 独立复核：retrieval-forge（L7 向量检索核心）
 * 不调用模块 selfCheck；独立手搓闭式/穷举黄金参照，复算关键不变量：
 *   度量 / top-k / 量化误差界 / MIPS / RRF / 确定性。
 * 全部 PASS → process.exit(0)；任一失败 → 打印诊断并 process.exit(1)。
 */
'use strict';
const M = require('../../src/retrieval-forge.js');

/* ---------- 独立黄金参照（闭式/穷举，不依赖模块检索函数） ---------- */
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function norm(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s); }
function cos(a, b) { const na = norm(a), nb = norm(b); if (!na || !nb) return 0; return dot(a, b) / (na * nb); }
function l2(a, b) { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } return Math.sqrt(s); }
function argSortDesc(arr) { const idx = arr.map((_, i) => i); idx.sort((x, y) => arr[y] - arr[x]); return idx; }
function makeRng(seed) {
  let a = (0x9E3779B9 ^ (seed >>> 0)) >>> 0, b = (0x243F6A88 ^ (seed >>> 0)) >>> 0,
      c = (0xB7E15162 ^ (seed >>> 0)) >>> 0, d = (0x85EBCA6B ^ (seed >>> 0)) >>> 0;
  for (let i = 0; i < 16; i++) { const t = (a + b) | 0; a = b ^ (b >>> 9); b = (c + (c << 3)) | 0; c = (c << 21 | c >>> 11); d = (d + 1) | 0; d = (d + t) | 0; }
  return function () { a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0; let t = (a + b) | 0; a = b ^ (b >>> 9); b = (c + (c << 3)) | 0; c = (c << 21 | c >>> 11); d = (d + 1) | 0; t = (t + d) | 0; c = (c + t) | 0; return (t >>> 0) / 4294967296; };
}

let failures = 0;
function check(name, cond, diag) {
  if (cond) { console.log('  PASS ' + name); }
  else { console.log('  FAIL ' + name + (diag ? '  | ' + diag : '')); failures++; }
}

console.log('=== 独立复核 retrieval-forge (不调用 selfCheck) ===');

/* ---------- 1) 度量 O00：余弦正交=0/平行=1（独立黄金） ---------- */
{
  const D = 8, rng = makeRng(0x9E3779B9);
  const rv = () => { const v = []; for (let i = 0; i < D; i++) v.push(rng() * 2 - 1); return v; };
  const a = rv(), b = rv();
  let proj = dot(a, b) / dot(a, a);
  const orth = a.map((x, i) => b[i] - proj * x);
  const par = a.map(x => 2 * x);
  const cOrth = cos(a, orth), cPar = cos(a, par), cRand = cos(a, b);
  const mOrth = M.cosineSim(a, orth), mPar = M.cosineSim(a, par), mRand = M.cosineSim(a, b);
  check('metric: module==golden (orth/par/rand)', Math.abs(mOrth - cOrth) <= 1e-15 && Math.abs(mPar - cPar) <= 1e-15 && Math.abs(mRand - cRand) <= 1e-15,
    'mOrth=' + mOrth + ' cOrth=' + cOrth + ' mPar=' + mPar + ' cPar=' + cPar);
  check('metric: 非恒定证伪 (rand in (-1,1) & orth~0 & par~1)', Math.abs(cOrth) <= 1e-15 && Math.abs(1 - cPar) <= 1e-15 && Math.abs(cRand) > 1e-6 && Math.abs(cRand) < 1 - 1e-9,
    'cOrth=' + cOrth + ' cPar=' + cPar + ' cRand=' + cRand);
}

/* ---------- 2) top-k O04：穷举 top-k == 独立全排序前 k（EXACT） ---------- */
{
  const D = 16, rng = makeRng(0x9E3779B9);
  const rv = () => { const v = new Array(D); for (let i = 0; i < D; i++) v[i] = rng() * 2 - 1; return v; };
  const N = 40, corpus = []; for (let i = 0; i < N; i++) corpus.push(rv());
  const q = rv(), k = 5;
  const res = M.bruteForceSearch(corpus, q, k, 'cosine');
  const golds = corpus.map(v => cos(q, v));
  const gtop = argSortDesc(golds).slice(0, k);
  let mismatch = 0; for (let i = 0; i < k; i++) if (res[i].idx !== gtop[i]) mismatch++;
  check('topk: 穷举结果与独立黄金全排序逐位一致 (mismatch=0)', mismatch === 0, 'mismatch=' + mismatch);
  const t1 = M.bruteForceSearch(corpus, q, 1, 'cosine'), t5 = M.bruteForceSearch(corpus, q, 5, 'cosine');
  check('topk: k=1 与 k=5 结果分歧（反空洞）', t1[0].idx !== t5[4].idx, 't1=' + t1[0].idx + ' t5[4]=' + t5[4].idx);
}

/* ---------- 3) 量化误差界 O05：int8 相对误差 ≤5% 且为近似 且收敛 ---------- */
{
  const D = 16, rng = makeRng(0x9E3779B9);
  const rv = () => { const v = new Array(D); for (let i = 0; i < D; i++) v[i] = rng() * 2 - 1; return v; };
  const N = 40, corpus = []; for (let i = 0; i < N; i++) corpus.push(rv());
  const q = rv();
  const qz8 = M.buildScalarQuantizer(corpus, { levels: 256 });
  const qzF = M.buildScalarQuantizer(corpus, { levels: 4096 });
  let maxRel8 = 0, maxRelF = 0, maxAbs = 0;
  for (let i = 0; i < N; i++) {
    const c8 = M.sqDecode(qz8, M.sqEncode(qz8, corpus[i])), q8 = M.sqDecode(qz8, M.sqEncode(qz8, q));
    const cF = M.sqDecode(qzF, M.sqEncode(qzF, corpus[i])), qF = M.sqDecode(qzF, M.sqEncode(qzF, q));
    const exact = l2(corpus[i], q);
    const rel8 = Math.abs(l2(c8, q8) - exact) / (exact + 1e-12);
    const relF = Math.abs(l2(cF, qF) - exact) / (exact + 1e-12);
    if (rel8 > maxRel8) maxRel8 = rel8;
    if (relF > maxRelF) maxRelF = relF;
    const ad = Math.abs(l2(c8, q8) - exact); if (ad > maxAbs) maxAbs = ad;
  }
  check('quant: int8 相对误差界 ≤5%', maxRel8 <= 0.05, 'maxRel8=' + maxRel8.toExponential(3));
  check('quant: 量化≠精确（真近似，反空洞）', maxAbs > 1e-9, 'maxAbsDiff=' + maxAbs.toExponential(3));
  check('quant: 误差随码本精度收敛 (fine<int8)', maxRelF < maxRel8, 'maxRelF=' + maxRelF.toExponential(3) + ' < maxRel8=' + maxRel8.toExponential(3));
}

/* ---------- 4) MIPS O08：穷举 MIPS == 独立内积排序前 k ---------- */
{
  const q = [1, 0];
  const corpus = [[1, 0], [3, 4], [0, 1]];
  const mips = M.mipsSearch(corpus, q, 1);
  const goldTop = argSortDesc(corpus.map(v => dot(v, q)))[0];
  check('mips: 穷举 top-1 == 独立内积排序前 k', mips[0].idx === goldTop, 'mipsTop=' + mips[0].idx + ' gold=' + goldTop);
  const nearest = M.bruteForceSearch(corpus, q, 1, 'l2');
  check('mips: 最大内积 ≠ 最近邻（证伪度量混淆）', mips[0].idx !== nearest[0].idx, 'mipsTop=' + mips[0].idx + ' nearest=' + nearest[0].idx);
}

/* ---------- 5) RRF O10：融合序 == 独立 RRF 公式且融合序≠任一输入 ---------- */
{
  // 独立 RRF 实现
  function indepRRF(lists, k) {
    const sc = new Map(), ord = [];
    for (const L of lists) for (let r = 0; r < L.length; r++) {
      const id = L[r]; if (!sc.has(id)) { sc.set(id, 0); ord.push(id); }
      sc.set(id, sc.get(id) + 1 / (k + r));
    }
    const ids = Array.from(sc.keys());
    ids.sort((a, b) => { const d = sc.get(b) - sc.get(a); if (d !== 0) return d > 0 ? 1 : -1; return ord.indexOf(a) - ord.indexOf(b); });
    return ids;
  }
  const L1 = ['A', 'B', 'C', 'D', 'E'], L2 = ['B', 'C', 'A', 'F', 'G'];
  const fused = M.rrfFuse([L1, L2], { k: 60 }).map(x => x.id);
  const indep = indepRRF([L1, L2], 60);
  check('rrf: 模块融合序 == 独立 RRF 公式', fused.join(',') === indep.join(','), 'fused=' + fused.join(',') + ' indep=' + indep.join(','));
  check('rrf: 融合 top-1 = 高频高排名项 (B)', fused[0] === 'B', 'top1=' + fused[0]);
  check('rrf: 融合序 ≠ 任一输入序（反空洞）', fused.join(',') !== L1.join(',') && fused.join(',') !== L2.join(','), 'fused=' + fused.join(','));
}

/* ---------- 6) 确定性 O13/O08：同 seed 逐位相等，异 seed 必分歧 ---------- */
{
  const D = 16;
  function buildDataset(seed) {
    const rng = makeRng(seed);
    const corpus = []; for (let i = 0; i < 30; i++) { const v = []; for (let j = 0; j < D; j++) v.push(rng() * 2 - 1); corpus.push(v); }
    const queries = []; for (let i = 0; i < 5; i++) { const v = []; for (let j = 0; j < D; j++) v.push(rng() * 2 - 1); queries.push(v); }
    return { corpus, queries };
  }
  const A1 = buildDataset(0x1111), A2 = buildDataset(0x1111), B = buildDataset(0x2222);
  const r1 = M.batchSearch(A1.corpus, A1.queries, 5, 'cosine');
  const r2 = M.batchSearch(A2.corpus, A2.queries, 5, 'cosine');
  let identical = true;
  for (let qi = 0; qi < r1.length; qi++) for (let ti = 0; ti < r1[qi].length; ti++) if (r1[qi][ti].idx !== r2[qi][ti].idx) identical = false;
  const rB = M.batchSearch(B.corpus, B.queries, 5, 'cosine');
  let diverges = false;
  for (let qi = 0; qi < r1.length; qi++) for (let ti = 0; ti < r1[qi].length; ti++) if (r1[qi][ti].idx !== rB[qi][ti].idx) diverges = true;
  check('determinism: 同 seed 两次检索逐位相等', identical, 'identical=' + identical);
  check('determinism: 异 seed 必分歧（反空洞）', diverges, 'diverges=' + diverges);
}

/* ---------- 汇总 ---------- */
if (failures === 0) {
  console.log('ALL PASS — 独立复核通过');
  process.exit(0);
} else {
  console.log('INDEPENDENT CHECK FAILED: ' + failures + ' 项未通过');
  process.exit(1);
}
