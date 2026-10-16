'use strict';
/* ============================================================================
 * 独立复核 sparse-forge —— 不采信模块 selfCheck()，自带 mulberry32 PRNG 与
 * 手搓稠密黄金参照交叉验证（world-class 独立审计口径）。
 *
 * 覆盖：
 *   R1  格式往返（INV-SPR-001）   R2  SpMV 黄金参照 + 零元跳过（INV-SPR-002）
 *   R3  SpMM 黄金参照 + 分配律（INV-SPR-003）  R4  CSR 规范性 + 乱序等价（INV-SPR-004）
 *   R5  nnz 统计与压缩率（INV-SPR-005）        R6  幅度剪枝（INV-SPR-006）
 *   R7  稀疏向量点积（INV-SPR-007）            R8  确定性（INV-SPR-008）
 *
 * 本文件不调用 sparse-forge.selfCheck / makeRng / denseVec 等，
 * 全部随机量由本文件自带 mulberry32(seed) 生成，与模块内部轨道独立。
 * 退出码：0 = 全部独立复核通过；1 = 任一失败。
 * ========================================================================== */
const path = require('path');
const SF = require(path.join(__dirname, '..', '..', 'src', 'sparse-forge.js'));

/* ---------- 独立 PRNG：mulberry32（与本模块 sfc32 不同族） ---------- */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ---------- 独立矩阵生成（固定计数布点，与模块生成器不同实现） ---------- */
function ownSparseMatrix(nRows, nCols, count, seed) {
  const rng = mulberry32(seed);
  const total = nRows * nCols;
  const idx = [];
  for (let i = 0; i < total; i++) idx.push(i);
  for (let i = 0; i < count; i++) {
    const j = i + Math.floor(rng() * (total - i));
    const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
  }
  const M = [];
  for (let i = 0; i < nRows; i++) M.push(new Array(nCols).fill(0));
  for (let i = 0; i < count; i++) {
    const cell = idx[i];
    let v = (rng() * 2 - 1) * 4;
    if (Math.abs(v) < 1e-4) v = v >= 0 ? 5e-3 : -5e-3;
    M[Math.floor(cell / nCols)][cell % nCols] = v;
  }
  return M;
}
function ownDenseVec(n, seed) {
  const rng = mulberry32(seed);
  return Array.from({ length: n }, () => (rng() * 2 - 1) * 3 || 0.7);
}
function ownDenseMatrix(nRows, nCols, seed) {
  const rng = mulberry32(seed);
  const M = [];
  for (let i = 0; i < nRows; i++) {
    const row = [];
    for (let j = 0; j < nCols; j++) { let v = (rng() * 2 - 1) * 4; if (Math.abs(v) < 1e-4) v = 5e-3; row.push(v); }
    M.push(row);
  }
  return M;
}

/* ---------- 手搓黄金参照（零模块函数依赖） ---------- */
function handMatvec(M, x) {
  const y = [];
  for (let i = 0; i < M.length; i++) {
    let acc = 0;
    for (let j = 0; j < M[i].length; j++) acc += M[i][j] * x[j];
    y.push(acc);
  }
  return y;
}
function handMatmul(A, B) {
  const C = [];
  for (let i = 0; i < A.length; i++) {
    const row = new Array(B[0].length).fill(0);
    for (let j = 0; j < A[i].length; j++) {
      const a = A[i][j];
      if (a === 0) continue;
      for (let t = 0; t < B[0].length; t++) row[t] += a * B[j][t];
    }
    C.push(row);
  }
  return C;
}
function maxDiff(A, B) {
  let d = 0;
  for (let i = 0; i < A.length; i++) {
    if (Array.isArray(A[i])) { d = Math.max(d, maxDiff(A[i], B[i])); continue; }
    d = Math.max(d, Math.abs(A[i] - B[i]));
  }
  return d;
}
function eqArr(a, b) { return a.length === b.length && a.every((v, i) => v === b[i]); }

const results = [];
function check(id, name, ok, detail) {
  results.push({ id, ok: !!ok, detail });
  console.log((ok ? '[PASS] ' : '[FAIL] ') + id + ' ' + name + (detail ? '  — ' + detail : ''));
}

const TIGHT = 1e-15, SOLVE = 1e-10;

/* ---------- R1 格式往返（INV-SPR-001） ---------- */
{
  let worst = 0;
  for (const sd of [0x10001, 0x10002, 0x10003]) {
    const M = ownSparseMatrix(37, 53, Math.floor(0.14 * 37 * 53), sd);
    const back = SF.csrToDense(SF.cooToCSR(SF.denseToCOO(M), 53), 53);
    worst = Math.max(worst, maxDiff(M, back));
  }
  const M = ownSparseMatrix(37, 53, Math.floor(0.14 * 37 * 53), 0x10001);
  const st = SF.nnzStats(SF.cooToCSR(SF.denseToCOO(M), 53), 37, 53);
  const sparseOk = st.storageRatio < 0.5;
  const Md = ownDenseMatrix(19, 19, 0x7D11);
  const stD = SF.nnzStats(SF.cooToCSR(SF.denseToCOO(Md), 19), 19, 19);
  const denseFalsified = stD.storageRatio >= 1.0;   // 稠密对照上压缩声称必不成立
  check('R1/INV-SPR-001', '稠密→COO→CSR→稠密 往返 + 压缩双边界',
    worst <= TIGHT && sparseOk && denseFalsified,
    'max|Δ|=' + worst.toExponential(2) + ' 稀疏ratio=' + st.storageRatio.toFixed(3) + ' 稠密ratio=' + stD.storageRatio.toFixed(3));
}

/* ---------- R2 SpMV 黄金参照 + 零元跳过（INV-SPR-002） ---------- */
{
  let worst = 0, multsOk = true;
  for (const sd of [0x20001, 0x20002, 0x20003]) {
    const nR = 41, nC = 57;
    const M = ownSparseMatrix(nR, nC, Math.floor(0.16 * nR * nC), sd);
    const x = ownDenseVec(nC, sd + 7);
    const yRef = handMatvec(M, x);
    const csr = SF.cooToCSR(SF.denseToCOO(M), nC);
    const stats = {};
    const y = SF.spmv(csr, x, stats);
    worst = Math.max(worst, maxDiff(y, yRef));
    if (stats.mults !== csr.vals.length || csr.vals.length >= 0.5 * nR * nC) multsOk = false;
  }
  check('R2/INV-SPR-002', 'SpMV ≡ 手搓 matvec + mults==nnz≪n·m', worst <= TIGHT && multsOk,
    'max|Δ|=' + worst.toExponential(2) + ' mults==nnz=' + multsOk);
}

/* ---------- R3 SpMM 黄金参照 + 分配律（INV-SPR-003） ---------- */
{
  let worst = 0, worstDist = 0;
  for (const sd of [0x30001, 0x30002]) {
    const nR = 29, nC = 43, k = 5;
    const M = ownSparseMatrix(nR, nC, Math.floor(0.2 * nR * nC), sd);
    const B = ownDenseMatrix(nC, k, sd + 11), B2 = ownDenseMatrix(nC, k, sd + 23);
    const csr = SF.cooToCSR(SF.denseToCOO(M), nC);
    worst = Math.max(worst, maxDiff(SF.spmm(csr, B), handMatmul(M, B)));
    const add = (P, Q) => P.map((r, i) => r.map((v, j) => v + Q[i][j]));
    worstDist = Math.max(worstDist, maxDiff(SF.spmm(csr, add(B, B2)), add(SF.spmm(csr, B), SF.spmm(csr, B2))));
  }
  check('R3/INV-SPR-003', 'SpMM ≡ 手搓 matmul + 分配律', worst <= TIGHT && worstDist <= SOLVE,
    'SpMM max|Δ|=' + worst.toExponential(2) + ' 分配律 max|Δ|=' + worstDist.toExponential(2));
}

/* ---------- R4 CSR 规范性 + 乱序等价（INV-SPR-004） ---------- */
{
  const nR = 37, nC = 53;
  const M = ownSparseMatrix(nR, nC, Math.floor(0.14 * nR * nC), 0x40001);
  const coo = SF.denseToCOO(M);
  // 独立乱序化（mulberry32）
  const sh = coo.entries.slice();
  const rng = mulberry32(0x99001);
  for (let i = sh.length - 1; i > 0; i--) { const k = Math.floor(rng() * (i + 1)); const t = sh[i]; sh[i] = sh[k]; sh[k] = t; }
  const csrS = SF.cooToCSR({ nRows: nR, nCols: nC, entries: sh }, nC);
  const csrO = SF.cooToCSR(coo, nC);
  let structOk = csrS.rowPtr[0] === 0 && csrS.rowPtr[nR] === csrS.vals.length;
  for (let i = 0; i < nR; i++) if (csrS.rowPtr[i + 1] < csrS.rowPtr[i]) structOk = false;
  for (let i = 0; i < nR; i++) for (let p = csrS.rowPtr[i] + 1; p < csrS.rowPtr[i + 1]; p++) if (csrS.colIdx[p] <= csrS.colIdx[p - 1]) structOk = false;
  const identical = eqArr(csrS.rowPtr, csrO.rowPtr) && eqArr(csrS.colIdx, csrO.colIdx) && eqArr(csrS.vals, csrO.vals);
  check('R4/INV-SPR-004', 'rowPtr 端点/单调 + colIdx 严格递增 + 乱序≡有序', structOk && identical,
    '结构=' + structOk + ' 乱序≡有序(EXACT)=' + identical);
}

/* ---------- R5 nnz 统计与压缩率（INV-SPR-005） ---------- */
{
  let nnzOk = true, compOk = true;
  for (const sd of [0x50001, 0x50002, 0x50003]) {
    const nR = 33, nC = 47, k = Math.floor(0.18 * nR * nC);
    const M = ownSparseMatrix(nR, nC, k, sd);
    const st = SF.nnzStats(SF.cooToCSR(SF.denseToCOO(M), nC), nR, nC);
    let cnt = 0; for (const row of M) for (const v of row) if (v !== 0) cnt++;
    if (st.nnz !== cnt) nnzOk = false;
    if (!(st.compression <= k / (nR * nC) + 1e-9)) compOk = false;
  }
  const Md = ownDenseMatrix(19, 19, 0x7D11);
  const stD = SF.nnzStats(SF.cooToCSR(SF.denseToCOO(Md), 19), 19, 19);
  const denseOk = Math.abs(stD.compression - 1) <= 1e-9;
  check('R5/INV-SPR-005', 'nnz==直接计数 + 压缩率边界', nnzOk && compOk && denseOk,
    'nnz EXACT=' + nnzOk + ' 稀疏≤密度=' + compOk + ' 稠密≈1=' + denseOk);
}

/* ---------- R6 幅度剪枝（INV-SPR-006） ---------- */
{
  let countOk = true, orderOk = true, refOk = true, keep1Ok = true;
  for (const sd of [0x60001, 0x60002]) {
    const nR = 35, nC = 45, k = Math.floor(0.22 * nR * nC);
    const M = ownSparseMatrix(nR, nC, k, sd);
    const pr = SF.magnitudePrune(M, 0.5);
    if (pr.kept !== Math.floor(pr.nnz * 0.5) || pr.kept + pr.dropped !== pr.nnz) countOk = false;
    let minKept = Infinity, maxDrop = -Infinity;
    for (const e of pr.keptEntries) minKept = Math.min(minKept, Math.abs(e[2]));
    for (const e of pr.droppedEntries) maxDrop = Math.max(maxDrop, Math.abs(e[2]));
    if (pr.droppedEntries.length > 0 && minKept < maxDrop) orderOk = false;
    const x = ownDenseVec(nC, sd + 3);
    const keptDense = [];
    for (let i = 0; i < nR; i++) keptDense.push(new Array(nC).fill(0));
    for (const e of pr.keptEntries) keptDense[e[0]][e[1]] = e[2];
    if (maxDiff(SF.spmv(pr.csr, x), handMatvec(keptDense, x)) > TIGHT) refOk = false;
    const csrFull = SF.cooToCSR(SF.denseToCOO(M), nC);
    const pr1 = SF.magnitudePrune(M, 1.0);
    if (!eqArr(SF.spmv(pr1.csr, x), SF.spmv(csrFull, x))) keep1Ok = false;
  }
  check('R6/INV-SPR-006', '剪枝计数/排序 EXACT + 保留集黄金参照 + keep=1 逐位一致',
    countOk && orderOk && refOk && keep1Ok,
    '计数=' + countOk + ' 排序=' + orderOk + ' 黄金参照=' + refOk + ' keep=1=' + keep1Ok);
}

/* ---------- R7 稀疏向量点积（INV-SPR-007） ---------- */
{
  let worst = 0, multsOk = true;
  const n = 71;
  for (const sd of [0x70001, 0x70002, 0x70003]) {
    const rngA = mulberry32(sd), rngB = mulberry32(sd + 991);
    const va = Array.from({ length: n }, (_, i) => (rngA() < 0.3 ? (rngA() * 2 - 1) * 3 || 1.1 : 0));
    const vb = Array.from({ length: n }, () => (rngB() < 0.3 ? (rngB() * 2 - 1) * 3 || 0.9 : 0));
    let dotRef = 0, inter = 0;
    for (let i = 0; i < n; i++) { dotRef += va[i] * vb[i]; if (va[i] !== 0 && vb[i] !== 0) inter++; }
    const stats = {};
    const d = SF.sparseVecDot(SF.denseToSparseVec(va), SF.denseToSparseVec(vb), stats);
    worst = Math.max(worst, Math.abs(d - dotRef));
    if (stats.mults !== inter || stats.mults >= n) multsOk = false;
  }
  check('R7/INV-SPR-007', '稀疏点积 ≡ 稠密点积 + mults==交集', worst <= TIGHT && multsOk,
    'max|Δ|=' + worst.toExponential(2) + ' mults==交集=' + multsOk);
}

/* ---------- R8 确定性（INV-SPR-008，用模块生成器验证 seed 可复现契约） ---------- */
{
  const pipe = (seed) => {
    const M = SF.randomSparseMatrix(26, 34, 0.18, seed);
    const csr = SF.cooToCSR(SF.denseToCOO(M), 34);
    const x = ownDenseVec(34, 0xAB);
    const pr = SF.magnitudePrune(M, 0.4);
    return JSON.stringify([csr.rowPtr, csr.colIdx, csr.vals, SF.spmv(csr, x), pr.kept, pr.csr.vals]);
  };
  const same = pipe(0x8151515) === pipe(0x8151515);
  const diff = pipe(0x8151515) !== pipe(0x8151516);
  check('R8/INV-SPR-008', '同 seed 逐位复现 + 异 seed 分歧', same && diff,
    '同seed一致=' + same + ' 异seed分歧=' + diff);
}

/* ---------- 汇总 ---------- */
const failed = results.filter((r) => !r.ok).length;
console.log('──────────────────────────────────────────────');
console.log('独立复核 sparse-forge：' + (results.length - failed) + '/' + results.length + ' 通过');
if (failed > 0) process.exit(1);
process.exit(0);
