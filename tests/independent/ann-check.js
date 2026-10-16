'use strict';
/* ============================================================================
 * 独立复核 ann-forge —— 不调用模块 selfCheck，手搓穷举黄金 top-k 交叉验证。
 *
 * 复核对象（仅用模块「构建/查询」函数，绝不用 selfCheck）：
 *   O00 KD 树 / O01 球树  精确 kNN（手搓黄金对照 recall=1.0）
 *   O02/O07/O09 LSH(超平面/p稳定/旋转) 近似召回 ≥0.8 且 退化<1.0
 *   O03 Annoy 近似召回 ≥0.8 且 退化(nTrees=1) recall<1.0
 *   O05 PQ 重建误差≤20% 且 ADC≠精确 且 随 M/Ks 收敛
 *   O06 IVFPQ 召回基础（nprobe=4）
 *   O08 ScaNN 各向异性量化 MIPS top1
 *   O10 HNSW 召回随 efSearch 单调 + ef=1 明显低于大 ef + 大 ef≥0.9
 *   O11 NSW 图索引召回 ≥0.5
 *   确定性：同 seed 建索引+查询逐位相等，异 seed 必分歧
 *
 * 黄金参照：本文件手搓 dist2 / goldTopK（不调用模块任何索引与 bruteKnN）。
 * 数据：与模块 selfCheck 同 seed(0x9E3779B9) 的 12 高斯团簇生成（复用模块 makeRng/gaussian
 *       仅作确定性 PRNG，非被测算法）。
 * ========================================================================== */
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const ANN = require(path.join(__dirname, '..', '..', 'src', 'ann-forge.js'));

const SEED = 0x9E3779B9;
const N = 600, D = 16, K = 5;

/* ---------- 手搓黄金参照（完全独立，不依赖模块） ---------- */
function dist2(a, b) { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } return s; }
function norm2(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s); }
function goldTopK(points, q, k) {
  const arr = [];
  for (let i = 0; i < points.length; i++) arr.push({ i, d: dist2(points[i], q) });
  arr.sort((a, b) => a.d - b.d);
  return arr.slice(0, k).map(o => o.i);
}
function recall(res, gold, k) {
  const g = new Set(gold.slice(0, k));
  let hit = 0;
  for (const r of res) if (g.has(r)) hit++;
  return hit / Math.min(k, gold.length);
}
function avgRecall(qfn, points, queries, k) {
  let s = 0;
  for (const q of queries) { const gold = goldTopK(points, q, k); const res = qfn(q, k); s += recall(res, gold, k); }
  return s / queries.length;
}

/* ---------- 数据集（同模块 selfCheck：12 高斯团簇） ---------- */
const C = 12;
const centers = [];
const rngC = ANN.makeRng(SEED);
for (let c = 0; c < C; c++) { const vc = new Float64Array(D); for (let j = 0; j < D; j++) vc[j] = rngC() * 2 - 1; centers.push(vc); }
const rngData = ANN.makeRng(SEED ^ 0x9e3779b9);
const points = [];
for (let i = 0; i < N; i++) {
  const c = i % C;
  const v = new Float64Array(D);
  for (let j = 0; j < D; j++) v[j] = centers[c][j] + ANN.gaussian(rngData) * 0.15;
  points.push(v);
}
const rngQ = ANN.makeRng(0x1234abcd);
const queries = [];
for (let i = 0; i < 24; i++) {
  const base = points[Math.floor(rngQ() * N)];
  const v = new Float64Array(D);
  for (let j = 0; j < D; j++) v[j] = base[j] + (rngQ() * 2 - 1) * 0.04;
  queries.push(v);
}

/* ---------- 独立复核主流程 ---------- */
const results = [];
function check(name, ok, detail) { results.push({ name, ok: !!ok, detail }); console.log((ok ? '✅ ' : '❌ ') + name + ' | ' + detail); }

/* O00/O01 KD 树 & 球树：精确 kNN（黄金 recall=1.0）+ 访问节点数≪N */
{
  let recKD = 1, visitsKD = 0;
  const ti = new ANN.TreeIndex('kd'); ti.build(points, 10);
  for (const q of queries) { const r = ti.query(q, K); recKD = Math.min(recKD, recall(r.idx, goldTopK(points, q, K), K)); visitsKD += r.visits; }
  const ratioKD = visitsKD / (queries.length * N);
  check('O00 KD 树精确 kNN(recall=1.0) 且 访问节点数≪N', recKD >= 0.999 && ratioKD < 0.8,
    'recall=' + recKD.toFixed(6) + ' 访问占比=' + ratioKD.toFixed(4));

  let recBall = 1, visitsBall = 0;
  const tb = new ANN.TreeIndex('ball'); tb.build(points, 10);
  for (const q of queries) { const r = tb.query(q, K); recBall = Math.min(recBall, recall(r.idx, goldTopK(points, q, K), K)); visitsBall += r.visits; }
  const ratioBall = visitsBall / (queries.length * N);
  check('O01 球树精确 kNN(recall=1.0) 且 访问节点数≪N', recBall >= 0.999 && ratioBall < 0.8,
    'recall=' + recBall.toFixed(6) + ' 访问占比=' + ratioBall.toFixed(4));
}

/* O02/O07/O09 LSH 族：良好配置 recall≥0.8 且 退化<1.0 */
{
  const hi = new ANN.HashIndex('hyper'); hi.build(points, D, { numTables: 8, numBits: 1, seed: SEED });
  const ri = new ANN.HashIndex('rot'); ri.build(points, D, { numTables: 8, numBits: 1, seed: SEED });
  const pi = new ANN.HashIndex('pstable'); pi.build(points, D, { numTables: 12, W: 1.0, seed: SEED });
  const rH = avgRecall((q) => hi.query(q, K), points, queries, K);
  const rR = avgRecall((q) => ri.query(q, K), points, queries, K);
  const rP = avgRecall((q) => pi.query(q, K), points, queries, K);
  const good = Math.min(rH, rR, rP);
  const hi1 = new ANN.HashIndex('hyper'); hi1.build(points, D, { numTables: 1, numBits: 1, seed: SEED });
  const pi1 = new ANN.HashIndex('pstable'); pi1.build(points, D, { numTables: 1, W: 1.0, seed: SEED });
  const deg = Math.max(avgRecall((q) => hi1.query(q, K), points, queries, K), avgRecall((q) => pi1.query(q, K), points, queries, K));
  check('O02/O07/O09 LSH(超平面/p稳定/旋转) recall≥0.8 且 退化<1.0', good >= 0.8 && deg < 1.0 - 1e-9,
    'good(min)=' + good.toFixed(4) + ' (hyper=' + rH.toFixed(3) + ' rot=' + rR.toFixed(3) + ' p=' + rP.toFixed(3) + ') degraded=' + deg.toFixed(4));
}

/* O03 Annoy：良好 recall≥0.8 且 退化(nTrees=1) recall<1.0 */
{
  const goodT = ANN.buildAnnoy(points, D, { nTrees: 40, leafSize: 16, seed: SEED });
  const rGood = avgRecall((q) => ANN.annoyQuery(goodT, points, q, K), points, queries, K);
  const badT = ANN.buildAnnoy(points, D, { nTrees: 1, leafSize: 8, seed: SEED });
  const rBad = avgRecall((q) => ANN.annoyQuery(badT, points, q, K, 12), points, queries, K);
  check('O03 Annoy recall≥0.8 且 退化(nTrees=1) recall<1.0', rGood >= 0.8 && rBad < 1.0 - 1e-9,
    'good(40trees)=' + rGood.toFixed(4) + ' degraded(1tree,searchK=12)=' + rBad.toFixed(4));
}

/* O05 PQ：重建误差≤20% 且 ADC≠精确 且 随 M/Ks 收敛 */
{
  const pqMain = ANN.trainPQ(points, D, 4, 64, ANN.makeRng(SEED));
  let errMain = 0, adcDiff = 0; const nS = 80;
  for (let t = 0; t < nS; t++) {
    const p = points[(t * 7) % N];
    const codes = ANN.pqEncode(pqMain, p);
    const recon = ANN.pqDecode(pqMain, codes);
    errMain += Math.sqrt(dist2(recon, p)) / (norm2(p) || 1);
    const table = ANN.pqADCTable(pqMain, p);
    const adc = ANN.pqADCDist(table, codes);
    const exact = dist2(p, p);
    adcDiff += Math.abs(adc - exact) / (exact + 1e-12);
  }
  errMain /= nS; adcDiff /= nS;
  const errs = (M, Ks) => { const pq = ANN.trainPQ(points, D, M, Ks, ANN.makeRng(SEED)); let e = 0; for (let t = 0; t < nS; t++) { const p = points[(t * 7) % N]; e += Math.sqrt(dist2(ANN.pqDecode(pq, ANN.pqEncode(pq, p)), p)) / (norm2(p) || 1); } return e / nS; };
  const eM8 = errs(8, 64), eKs16 = errs(4, 16), eKs256 = errs(4, 256);
  check('O05 PQ 重建误差≤20% 且 ADC≠精确 且 误差随M/Ks收敛',
    errMain <= 0.2 && adcDiff > 1e-3 && eM8 < errMain && eKs256 < eKs16,
    'errMain=' + errMain.toFixed(4) + ' adcDiff=' + adcDiff.toExponential(2) + ' err(M8)=' + eM8.toFixed(4) + ' err(Ks16)=' + eKs16.toFixed(4) + ' err(Ks256)=' + eKs256.toFixed(4));
}

/* O06 IVFPQ 召回基础（nprobe=4） */
{
  const ivf = ANN.buildIVFPQ(points, D, 16, 4, 64, SEED);
  let valid = true, rec = 0;
  for (const q of queries.slice(0, 12)) {
    const r = ANN.ivfpqQuery(ivf, points, q, K, 4);
    if (r.length !== K) valid = false;
    rec += recall(r, goldTopK(points, q, K), K);
  }
  rec /= 12;
  check('O06 IVFPQ 召回基础(nprobe=4)', valid && rec >= 0.5, 'validReturn=' + valid + ' avgRecall=' + rec.toFixed(3));
}

/* O08 ScaNN 各向异性量化 MIPS top1 */
{
  const sc = ANN.buildScaNN(points, D, 64, SEED);
  let rec = 0, valid = true;
  for (const q of queries.slice(0, 12)) {
    const r = ANN.scannQuery(sc, points, q, 1, 64);
    if (r.length !== 1) valid = false; else { const gold = goldTopK(points, q, 1); if (r[0] === gold[0]) rec++; }
  }
  rec /= 12;
  check('O08 ScaNN 各向异性量化 MIPS top1', valid && rec >= 0.7, 'valid=' + valid + ' top1Recall=' + rec.toFixed(3));
}

/* O09 旋转不变 LSH 召回 */
{
  const ri = new ANN.HashIndex('rot'); ri.build(points, D, { numTables: 8, numBits: 1, seed: SEED });
  const r = avgRecall((q) => ri.query(q, K), points, queries, K);
  check('O09 旋转不变 LSH 召回', r >= 0.8, 'avgRecall=' + r.toFixed(4));
}

/* O10 HNSW：召回随 efSearch 单调 + ef=1 明显低于大 ef + 大 ef≥0.9 */
{
  const hnsw = ANN.buildHNSW(points, D, { seed: SEED, M: 16, Mmax0: 32, efConstruction: 120, mL: 1 / Math.log(16) });
  const efs = [1, 2, 4, 8, 16, 32];
  const recs = efs.map(ef => avgRecall((q) => hnsw.search(q, K, ef), points, queries, K));
  let mono = true;
  for (let i = 1; i < recs.length; i++) if (recs[i] < recs[i - 1] - 1e-9) mono = false;
  check('O10 HNSW 召回随 efSearch 单调 且 ef=1 明显低于大 ef',
    mono && recs[0] < recs[recs.length - 1] && recs[recs.length - 1] >= 0.9 && recs[0] <= 0.5,
    'recall@ef=' + efs.map((e, i) => e + ':' + recs[i].toFixed(3)).join(' '));
}

/* O11 NSW 图索引召回 */
{
  const nsw = ANN.buildNSW(points, D, 24, ANN.makeRng(SEED));
  const r = avgRecall((q) => ANN.nswSearch(nsw, points, q, K, null, 40), points, queries, K);
  check('O11 NSW 图索引召回', r >= 0.5, 'avgRecall=' + r.toFixed(4));
}

/* 确定性：同 seed 建索引+查询逐位相等；异 seed 必分歧 */
{
  const h1 = ANN.buildHNSW(points, D, { seed: SEED, M: 16, Mmax0: 32, efConstruction: 120, mL: 1 / Math.log(16) });
  const h2 = ANN.buildHNSW(points, D, { seed: SEED, M: 16, Mmax0: 32, efConstruction: 120, mL: 1 / Math.log(16) });
  const h3 = ANN.buildHNSW(points, D, { seed: 0xABCDEF01, M: 16, Mmax0: 32, efConstruction: 120, mL: 1 / Math.log(16) });
  let sameAll = true, diffAny = false;
  for (const q of queries) {
    const r1 = JSON.stringify(h1.search(q, K, 32)), r2 = JSON.stringify(h2.search(q, K, 32)), r3 = JSON.stringify(h3.search(q, K, 32));
    if (r1 !== r2) sameAll = false;
    if (r1 !== r3) diffAny = true;
  }
  const sigDiff = h1.levelSignature() !== h3.levelSignature();
  check('确定性：同seed逐位相等 异seed必分歧', sameAll && (diffAny || sigDiff),
    'sameSeed=' + sameAll + ' diffSeed=' + (diffAny || sigDiff));
}

/* ---------- 文件哈希（独立计算 sha256，供交付报告） ---------- */
const srcPath = path.join(__dirname, '..', '..', 'src', 'ann-forge.js');
const sha = crypto.createHash('sha256').update(fs.readFileSync(srcPath, 'utf8')).digest('hex');
console.log('\n[文件] ' + srcPath);
console.log('[sha256] ' + sha);

/* ---------- 汇总 ---------- */
const failed = results.filter(r => !r.ok);
const pass = failed.length === 0;
console.log('\n' + (pass ? '✅' : '❌') + ' ann-forge 独立复核：' + (results.length - failed.length) + '/' + results.length + ' 通过' +
  (pass ? '' : '；失败=' + failed.map(f => f.name).join('; ')));
process.exit(pass ? 0 : 1);
