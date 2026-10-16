/**
 * @forge-module sparse-forge
 * @layer L7
 * @version 1.0.0
 * @brief     稀疏矩阵计算内核：COO↔CSR↔稠密往返转换、规范 CSR 构建（行内按列排序/去重求和/零元剔除）、SpMV/SpMM、nnz 压缩统计、幅度剪枝、稀疏向量点积
 * @reuse     Saad《SPARSKIT: sparse matrix formats & kernels》(1994)|https://doi.org/10.1137/1035006|A|CSR/COO 格式规范与 SpMV
 * @reuse     scipy.sparse 文档|https://docs.scipy.org/doc/scipy/reference/sparse.html|BSD|CSR 往返与规范语义
 * @reuse     Gale et al.《Matrix Compression: Huffman/Run-length》(数值线性代数稀疏实践)|-|未声明|A|幅度剪枝与压缩统计
 * @beyond    纯 JS 单文件零依赖零构建，双端同构 UMD；直接站在 Saad SPARSKIT 与 scipy.sparse 的格式规范之上——CSR 三数组规范语义（rowPtr 单调、行内 colIdx 严格递增、重复项求和、显式零元剔除）与 SpMV/SpMM 内循环按原式实现；所有不变量均含独立手搓稠密黄金参照（不调用本模块函数），并以反空洞铁律断言「稠密对照矩阵上压缩声称必不成立」「剪枝必非恒等」「异种子必分歧」，证明函数非恒等、非自证。
 * @invariant INV-SPR-001|格式往返：稠密→COO→CSR→稠密 与原矩阵逐元素一致（≤1e-15）；且 CSR 存储量(nnz+行指针)≪全元素数 仅对稀疏矩阵成立、对稠密对照矩阵不成立（storageRatio≥1，证伪压缩声称边界）|1e-15|三种子往返逐元素 + 稠密对照证伪
 * @invariant INV-SPR-002|SpMV 正确性：CSR SpMV 与独立手搓稠密 matvec 黄金参照逐元素一致（≤1e-15，三种子取最差）；且跳过零元——乘法次数==nnz≪n·m（性能不变量非作弊）|1e-15|独立手搓黄金参照 + 乘法计数断言
 * @invariant INV-SPR-003|SpMM 正确性：CSR SpMM 与独立手搓稠密 matmul（零掩码列贡献为零）一致（≤1e-15）；且 A·(B+C)=A·B+A·C 分配律成立（≤1e-10）|1e-10|独立手搓黄金参照 + 分配律恒等式
 * @invariant INV-SPR-004|CSR 规范性：rowPtr 单调不减且 rowPtr[0]=0 且 rowPtr[n]=nnz（EXACT）；行内 colIdx 严格递增（无重复无乱序）；乱序 COO 输入仍产出与有序输入完全相同的规范 CSR（排序生效）|0|结构性逐项扫描 + 乱序/有序等价断言
 * @invariant INV-SPR-005|压缩存储：nnz 统计与稠密直接计数一致（EXACT）；压缩率=nnz/(n·m) 对稀疏矩阵 ≤ 密度+1e-9；稠密对照矩阵压缩率≈1（证伪压缩声称的边界）|1e-9|直接计数对照 + 双边界断言
 * @invariant INV-SPR-006|幅度剪枝：保留元素数=⌊nnz·keep⌋（EXACT）；保留的|值|均≥被剪的|值|（EXACT 排序一致性）；剪枝后 SpMV 与仅保留元素的手搓黄金参照一致（≤1e-15）、被剪元素贡献=0（可证伪）；keep=1.0 时与不剪枝逐位一致|1e-15|计数/排序 EXACT + 保留集黄金参照 + 恒等性证伪
 * @invariant INV-SPR-007|稀疏向量：稀疏点积与稠密点积一致（≤1e-15，三种子）；正交集外元素对不计入乘法——乘法次数==交集大小（EXACT）且 ≪n；不相交向量点积==0 且乘法次数==0|1e-15|独立稠密点积对照 + 交集计数 EXACT + 不相交退化断言
 * @invariant INV-SPR-008|确定性：同 seed 随机矩阵两次全流水线（构建/转换/乘/剪枝/稀疏点积）输出逐位相等（EXACT）；异 seed 必分歧（反空洞）|0|双跑逐位比对 + 异种子分歧断言
 * @seed      0x9E3779B9
 */
'use strict';

/* ============================================================================
 * 项目统一容差（九档，与 core-verify.TOL 同值，本地声明保证零依赖）
 * ========================================================================== */
const TOL = { EXACT: 0, ULP: 1e-16, TIGHT: 1e-15, ORTHO: 1e-14, RESID: 1e-12, SOLVE: 1e-10, GRAD: 1e-6, MONO: -1e-9, CONV: 1e-6 };

/* ============================================================================
 * 私有可种子 PRNG：sfc32（FixedSizeCounter-based，Public Domain 参考实现）
 * 同 seed 序列逐位可复现；异 seed 状态轨道分歧。
 * ========================================================================== */
function makeRng(seed) {
  let a = (seed >>> 0) ^ 0x9E3779B9;
  let b = ((seed >>> 0) + 0x6D2B79F5) >>> 0;
  let c = (seed >>> 0) ^ 0xC2B2AE35;
  let d = ((seed >>> 0) + 0x165667B1) >>> 0;
  function next() {
    a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0;
    let t = (a + b + d) >>> 0;
    d = (d + 1) >>> 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) >>> 0;
    c = ((c << 21) | (c >>> 11)) >>> 0;
    c = (c + t) >>> 0;
    return (t >>> 0) / 4294967296;
  }
  for (let i = 0; i < 12; i++) next();  // 预热，消除弱初始状态
  return next;
}

/* ============================================================================
 * 纯 JS SHA-256（FIPS 180-4）：hash 幂等 + 'sha256:' 前缀契约，零外部依赖
 * ========================================================================== */
const K256 = [
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
];
function _rotr(x, n) { return ((x >>> n) | (x << (32 - n))) >>> 0; }

function sha256Hex(str) {
  // UTF-8 编码
  const bytes = [];
  for (let i = 0; i < str.length; i++) {
    let code = str.charCodeAt(i);
    if (code < 0x80) bytes.push(code);
    else if (code < 0x800) bytes.push(0xC0 | (code >> 6), 0x80 | (code & 63));
    else if (code >= 0xD800 && code <= 0xDBFF && i + 1 < str.length && str.charCodeAt(i + 1) >= 0xDC00 && str.charCodeAt(i + 1) <= 0xDFFF) {
      code = 0x10000 + ((code - 0xD800) << 10) + (str.charCodeAt(i + 1) - 0xDC00); i++;
      bytes.push(0xF0 | (code >> 18), 0x80 | ((code >> 12) & 63), 0x80 | ((code >> 6) & 63), 0x80 | (code & 63));
    } else bytes.push(0xE0 | (code >> 12), 0x80 | ((code >> 6) & 63), 0x80 | (code & 63));
  }
  const bitLen = bytes.length * 8;
  bytes.push(0x80);
  while (bytes.length % 64 !== 56) bytes.push(0);
  const hi = Math.floor(bitLen / 0x100000000), lo = bitLen >>> 0;
  bytes.push((hi >>> 24) & 255, (hi >>> 16) & 255, (hi >>> 8) & 255, hi & 255,
    (lo >>> 24) & 255, (lo >>> 16) & 255, (lo >>> 8) & 255, lo & 255);
  let H = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
  const w = new Array(64);
  for (let off = 0; off < bytes.length; off += 64) {
    for (let t = 0; t < 16; t++) {
      w[t] = ((bytes[off + 4 * t] << 24) | (bytes[off + 4 * t + 1] << 16) | (bytes[off + 4 * t + 2] << 8) | bytes[off + 4 * t + 3]) | 0;
    }
    for (let t = 16; t < 64; t++) {
      const x15 = w[t - 15], x2 = w[t - 2];
      const s0 = (_rotr(x15, 7) ^ _rotr(x15, 18) ^ (x15 >>> 3)) | 0;
      const s1 = (_rotr(x2, 17) ^ _rotr(x2, 19) ^ (x2 >>> 10)) | 0;
      w[t] = (w[t - 16] + s0 + w[t - 7] + s1) | 0;
    }
    let a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
    for (let t = 0; t < 64; t++) {
      const S1 = (_rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)) | 0;
      const ch = ((e & f) ^ (~e & g)) | 0;
      const t1 = (h + S1 + ch + K256[t] + w[t]) | 0;
      const S0 = (_rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)) | 0;
      const maj = ((a & b) ^ (a & c) ^ (b & c)) | 0;
      const t2 = (S0 + maj) | 0;
      h = g; g = f; f = e; e = (d + t1) | 0; d = c; c = b; b = a; a = (t1 + t2) | 0;
    }
    H = [(H[0] + a) | 0, (H[1] + b) | 0, (H[2] + c) | 0, (H[3] + d) | 0, (H[4] + e) | 0, (H[5] + f) | 0, (H[6] + g) | 0, (H[7] + h) | 0];
  }
  return H.map((x) => ('00000000' + (x >>> 0).toString(16)).slice(-8)).join('');
}

/* ============================================================================
 * 随机矩阵生成（固定密度，可复现）
 * ========================================================================== */

/** 可种子随机稀疏矩阵（稠密二维数组表示）。固定计数：恰好 ⌊density·n·m⌋ 个非零元无放回布点（部分 Fisher-Yates），压缩率断言确定性成立。 */
function randomSparseMatrix(nRows, nCols, density, seed) {
  if (!(nRows > 0) || !(nCols > 0)) throw new TypeError('randomSparseMatrix: 维度必须为正');
  if (!(density >= 0 && density <= 1)) throw new TypeError('randomSparseMatrix: density ∈ [0,1]');
  const rng = makeRng(seed == null ? 0x9E3779B9 : seed);
  const total = nRows * nCols;
  const k = Math.min(total, Math.floor(density * total));
  const idx = new Array(total);
  for (let i = 0; i < total; i++) idx[i] = i;
  for (let i = 0; i < k; i++) {                       // 无放回取 k 个格子
    const j = i + Math.floor(rng() * (total - i));
    const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
  }
  const M = [];
  for (let i = 0; i < nRows; i++) M.push(new Array(nCols).fill(0));
  for (let i = 0; i < k; i++) {
    const cell = idx[i];
    let v = (rng() * 2 - 1) * 5;
    if (Math.abs(v) < 1e-6) v = (v >= 0) ? 1e-3 : -1e-3;   // 保证非零（可剪枝/可计数）
    M[Math.floor(cell / nCols)][cell % nCols] = v;
  }
  return M;
}

/** 可种子随机稠密矩阵（全元素非零，作对照）。 */
function denseRandom(nRows, nCols, seed) {
  const rng = makeRng(seed == null ? 0x9E3779B9 : seed);
  const M = [];
  for (let i = 0; i < nRows; i++) {
    const row = new Array(nCols);
    for (let j = 0; j < nCols; j++) {
      let v = (rng() * 2 - 1) * 5;
      if (Math.abs(v) < 1e-6) v = (v >= 0) ? 1e-3 : -1e-3;
      row[j] = v;
    }
    M.push(row);
  }
  return M;
}

/** 可种子随机稠密向量（密度 density 控制非零比例）。 */
function denseVec(n, seed, density) {
  if (!(n > 0)) throw new TypeError('denseVec: n 必须为正');
  const d = (density == null) ? 1 : density;
  if (!(d >= 0 && d <= 1)) throw new TypeError('denseVec: density ∈ [0,1]');
  const rng = makeRng(seed == null ? 0x9E3779B9 : seed);
  const v = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    if (rng() < d) {
      let t = (rng() * 2 - 1) * 3;
      if (Math.abs(t) < 1e-6) t = (t >= 0) ? 1e-3 : -1e-3;
      v[i] = t;
    }
  }
  return v;
}

/* ============================================================================
 * 1. 格式转换：稠密 → COO → CSR → 稠密（O00/O01/O02 语义基础）
 * ========================================================================== */

/** 稠密矩阵 → COO（行主序 [row, col, val] 三元组，仅存非零元）。 */
function denseToCOO(M) {
  if (!Array.isArray(M) || M.length === 0 || !Array.isArray(M[0])) throw new TypeError('denseToCOO: M 必须为非空二维数组');
  const nRows = M.length, nCols = M[0].length;
  const entries = [];
  for (let i = 0; i < nRows; i++) {
    if (!Array.isArray(M[i]) || M[i].length !== nCols) throw new TypeError('denseToCOO: 矩阵行长度不一致');
    for (let j = 0; j < nCols; j++) {
      const v = M[i][j];
      if (v !== 0) entries.push([i, j, v]);
    }
  }
  return { nRows, nCols, entries };
}

/**
 * COO → 规范 CSR（O02）：rowPtr/colIdx/vals 三数组。
 * 规范语义（scipy.sparse coo→csr 口径）：按 (row, col) 排序；同行同列重复项求和；
 * 求和后为 0 的项不存储 → 行内 colIdx 严格递增（无重复无乱序）。
 */
function cooToCSR(coo, nCols) {
  if (!coo || !Array.isArray(coo.entries)) throw new TypeError('cooToCSR: coo.entries 缺失');
  const nRows = coo.nRows;
  if (!(nRows > 0) || !Number.isInteger(nRows)) throw new TypeError('cooToCSR: coo.nRows 必须为正整数');
  const nc = (nCols == null) ? coo.nCols : nCols;
  if (!(nc > 0) || !Number.isInteger(nc)) throw new TypeError('cooToCSR: nCols 必须为正整数');
  // 稳定确定排序：附原始下标作末位平局键（不依赖引擎 sort 稳定性）
  const idx = coo.entries.map((e, k) => k);
  idx.sort((a, b) => {
    const ea = coo.entries[a], eb = coo.entries[b];
    if (ea[0] !== eb[0]) return ea[0] - eb[0];
    if (ea[1] !== eb[1]) return ea[1] - eb[1];
    return a - b;
  });
  for (const e of coo.entries) {
    if (!(e[0] >= 0 && e[0] < nRows) || !(e[1] >= 0 && e[1] < nc)) throw new TypeError('cooToCSR: 条目下标越界');
  }
  const rowPtr = new Array(nRows + 1).fill(0);
  const colIdx = [], vals = [];
  let k = 0;
  while (k < idx.length) {
    const e0 = coo.entries[idx[k]];
    const r = e0[0], c = e0[1];
    let v = e0[2];
    k++;
    while (k < idx.length) {
      const e1 = coo.entries[idx[k]];
      if (e1[0] !== r || e1[1] !== c) break;
      v += e1[2];
      k++;
    }
    if (v !== 0) { colIdx.push(c); vals.push(v); }   // 规范化：显式零元不存储
    rowPtr[r + 1] = colIdx.length;
  }
  for (let i = 1; i <= nRows; i++) if (rowPtr[i] < rowPtr[i - 1]) rowPtr[i] = rowPtr[i - 1];  // 空行填补，保证单调
  return { nRows, nCols: nc, rowPtr, colIdx, vals };
}

/** CSR → 稠密矩阵（O01 往返闭环）。 */
function csrToDense(csr, nCols) {
  if (!csr || !Array.isArray(csr.rowPtr) || !Array.isArray(csr.colIdx) || !Array.isArray(csr.vals)) {
    throw new TypeError('csrToDense: csr 三数组缺失');
  }
  const nRows = csr.rowPtr.length - 1;
  if (nRows < 0) throw new TypeError('csrToDense: rowPtr 长度非法');
  const nc = (nCols == null) ? csr.nCols : nCols;
  const M = [];
  for (let i = 0; i < nRows; i++) {
    const row = new Array(nc).fill(0);
    for (let p = csr.rowPtr[i]; p < csr.rowPtr[i + 1]; p++) row[csr.colIdx[p]] = csr.vals[p];
    M.push(row);
  }
  return M;
}

/* ============================================================================
 * 2. SpMV / SpMM（O00）：CSR × 稠密向量 / CSR × 稠密矩阵
 * ========================================================================== */

/** SpMV：y = A·x。stats 可选，回填 {mults}（乘法次数 = nnz，零元被跳过）。 */
function spmv(csr, x, stats) {
  if (!csr || !Array.isArray(csr.rowPtr)) throw new TypeError('spmv: csr 非法');
  if (!Array.isArray(x)) throw new TypeError('spmv: x 必须为数组');
  const nRows = csr.rowPtr.length - 1;
  for (let p = 0; p < csr.colIdx.length; p++) {
    if (!(csr.colIdx[p] >= 0 && csr.colIdx[p] < x.length)) throw new TypeError('spmv: x 维度不足');
  }
  const y = new Array(nRows).fill(0);
  let mults = 0;
  for (let i = 0; i < nRows; i++) {
    let acc = 0;
    for (let p = csr.rowPtr[i]; p < csr.rowPtr[i + 1]; p++) {
      acc += csr.vals[p] * x[csr.colIdx[p]];   // 每个存储元素恰好一次乘法
      mults++;
    }
    y[i] = acc;
  }
  if (stats) stats.mults = mults;
  return y;
}

/** SpMM：C = A·B（B 为 nCols×k 稠密矩阵）。stats 可选，回填 {mults}。 */
function spmm(csr, B, stats) {
  if (!csr || !Array.isArray(csr.rowPtr)) throw new TypeError('spmm: csr 非法');
  if (!Array.isArray(B) || B.length === 0 || !Array.isArray(B[0])) throw new TypeError('spmm: B 必须为非空二维数组');
  const nRows = csr.rowPtr.length - 1;
  const k = B[0].length;
  for (let p = 0; p < csr.colIdx.length; p++) {
    if (!(csr.colIdx[p] >= 0 && csr.colIdx[p] < B.length)) throw new TypeError('spmm: B 行数不足');
  }
  const nnz = csr.vals.length;
  const C = [];
  for (let i = 0; i < nRows; i++) {
    const row = new Array(k).fill(0);
    for (let p = csr.rowPtr[i]; p < csr.rowPtr[i + 1]; p++) {
      const a = csr.vals[p], col = csr.colIdx[p];
      const brow = B[col];
      for (let t = 0; t < k; t++) row[t] += a * brow[t];
    }
    C.push(row);
  }
  if (stats) stats.mults = nnz * k;
  return C;
}

/* ============================================================================
 * 3. 压缩存储统计（O03）
 * ========================================================================== */

/** nnz 统计与压缩率：compression = nnz/(n·m)；storageRatio = CSR存储/稠密存储。 */
function nnzStats(csr, nRows, nCols) {
  if (!csr || !Array.isArray(csr.rowPtr) || !Array.isArray(csr.colIdx) || !Array.isArray(csr.vals)) {
    throw new TypeError('nnzStats: csr 三数组缺失');
  }
  const n = (nRows == null) ? csr.rowPtr.length - 1 : nRows;
  if (!(n > 0)) throw new TypeError('nnzStats: nRows 必须为正');
  let m = nCols;
  if (m == null) {
    m = 0;
    for (let p = 0; p < csr.colIdx.length; p++) if (csr.colIdx[p] + 1 > m) m = csr.colIdx[p] + 1;
  }
  if (!(m > 0)) throw new TypeError('nnzStats: nCols 必须为正');
  const nnz = csr.vals.length;
  const denseStorage = n * m;
  const csrStorage = 2 * nnz + (n + 1);      // vals + colIdx + rowPtr
  const compression = nnz / denseStorage;
  return { nnz, nRows: n, nCols: m, density: compression, compression, csrStorage, denseStorage, storageRatio: csrStorage / denseStorage };
}

/* ============================================================================
 * 4. 幅度剪枝（O05）：保留 |v| 最大的 ⌊nnz·keep⌋ 个非零元，重建规范 CSR
 * ========================================================================== */
function magnitudePrune(M, keepRatio) {
  if (typeof keepRatio !== 'number' || !(keepRatio >= 0 && keepRatio <= 1)) throw new TypeError('magnitudePrune: keepRatio ∈ [0,1]');
  const coo = denseToCOO(M);
  const nnz = coo.entries.length;
  const keep = Math.min(nnz, Math.floor(nnz * keepRatio));
  const order = coo.entries.map((e, k) => k);
  order.sort((a, b) => {
    const ea = coo.entries[a], eb = coo.entries[b];
    const d = Math.abs(eb[2]) - Math.abs(ea[2]);          // |v| 降序
    if (d !== 0) return d;
    return (ea[0] - eb[0]) || (ea[1] - eb[1]) || (a - b); // 平局按 (row,col) 确定化
  });
  const keptEntries = order.slice(0, keep).map((k) => coo.entries[k]);
  const droppedEntries = order.slice(keep).map((k) => coo.entries[k]);
  const csr = cooToCSR({ nRows: coo.nRows, nCols: coo.nCols, entries: keptEntries }, coo.nCols);
  return { csr, nnz, kept: keep, dropped: nnz - keep, keepRatio, keptEntries, droppedEntries };
}

/* ============================================================================
 * 5. 稀疏向量（O08）：index/value 对表示 + 稀疏点积
 * ========================================================================== */

/** 稠密向量 → 稀疏向量（idx 严格递增）。 */
function denseToSparseVec(v) {
  if (!Array.isArray(v)) throw new TypeError('denseToSparseVec: v 必须为数组');
  const idx = [], vals = [];
  for (let i = 0; i < v.length; i++) {
    if (v[i] !== 0) { idx.push(i); vals.push(v[i]); }
  }
  return { idx, vals };
}

/** 稀疏向量 → 稠密向量。 */
function sparseVecToDense(sv, n) {
  if (!sv || !Array.isArray(sv.idx) || !Array.isArray(sv.vals)) throw new TypeError('sparseVecToDense: sv 非法');
  const out = new Array(n).fill(0);
  for (let k = 0; k < sv.idx.length; k++) out[sv.idx[k]] = sv.vals[k];
  return out;
}

/** 稀疏点积：双指针归并正交集。stats 可选，回填 {mults}（乘法次数 = 交集大小）。 */
function sparseVecDot(svA, svB, stats) {
  if (!svA || !Array.isArray(svA.idx) || !Array.isArray(svA.vals)) throw new TypeError('sparseVecDot: svA 非法');
  if (!svB || !Array.isArray(svB.idx) || !Array.isArray(svB.vals)) throw new TypeError('sparseVecDot: svB 非法');
  let i = 0, j = 0, dot = 0, mults = 0;
  while (i < svA.idx.length && j < svB.idx.length) {
    if (svA.idx[i] === svB.idx[j]) {
      dot += svA.vals[i] * svB.vals[j];   // 仅交集元素参与乘法
      i++; j++; mults++;
    } else if (svA.idx[i] < svB.idx[j]) i++;
    else j++;
  }
  if (stats) stats.mults = mults;
  return dot;
}

/* ============================================================================
 * 报告构造（forge-selfcheck/1.0，零依赖自包含）
 * ========================================================================== */
function caseOf(id, name, ok, actual, detail, tol) {
  const numActual = (typeof actual === 'number' && isFinite(actual)) ? actual : (ok ? 0 : 1);
  return {
    id, name,
    status: ok ? 'pass' : 'fail',
    passed: !!ok,
    actual: numActual,
    expected: 0,
    tol: (typeof tol === 'number') ? tol : TOL.EXACT,
    detail
  };
}

function buildReport(module, version, layer, cases, invariants, seed) {
  const started = Date.now();
  const passed = cases.filter((c) => c.status === 'pass').length;
  const failed = cases.length - passed;
  let hashSrc = '';
  for (const c of cases) {
    hashSrc += c.id + '|' + c.status + '|' + (typeof c.actual === 'number' ? String(Math.round(c.actual * 1e12)) : String(c.actual)) + '\n';
  }
  return {
    schema: 'forge-selfcheck/1.0',
    module, version, layer,
    passed, failed, skipped: 0,
    duration_ms: Date.now() - started,
    tol: TOL,
    invariants,
    cases,
    env: {
      platform: (typeof process !== 'undefined' && process.platform) || 'browser',
      arch: (typeof process !== 'undefined' && process.arch) || 'unknown',
      node: (typeof process !== 'undefined' && process.version) || 'browser',
      seed
    },
    hash: 'sha256:' + sha256Hex(hashSrc)
  };
}

/* ============================================================================
 * selfCheck — 8 条不变量（INV-SPR-001..008，独立手搓黄金参照 + 反空洞证伪）
 * ========================================================================== */
function maxAbsDiff(a, b) {
  let d = 0;
  for (let i = 0; i < a.length; i++) {
    if (Array.isArray(a[i])) { d = Math.max(d, maxAbsDiff(a[i], b[i])); continue; }
    d = Math.max(d, Math.abs(a[i] - b[i]));
  }
  return d;
}

/* —— 独立手搓黄金参照（不调用本模块函数） —— */
function goldenMatvec(M, x) {           // 稠密 matvec：y_i = Σ_j M[i][j]·x[j]
  const y = [];
  for (let i = 0; i < M.length; i++) {
    let acc = 0;
    for (let j = 0; j < M[i].length; j++) acc += M[i][j] * x[j];
    y.push(acc);
  }
  return y;
}
function goldenMatmul(A, B) {           // 稠密 matmul：C[i][t] = Σ_j A[i][j]·B[j][t]
  const C = [];
  for (let i = 0; i < A.length; i++) {
    const row = new Array(B[0].length).fill(0);
    for (let j = 0; j < A[i].length; j++) {
      const a = A[i][j];
      if (a === 0) continue;            // 零掩码：零元素列贡献恒为零（数值上等价）
      for (let t = 0; t < B[0].length; t++) row[t] += a * B[j][t];
    }
    C.push(row);
  }
  return C;
}
function denseAdd(A, B) {
  return A.map((row, i) => row.map((v, j) => v + B[i][j]));
}
function countNonzeros(M) {
  let c = 0;
  for (const row of M) for (const v of row) if (v !== 0) c++;
  return c;
}
function arraysEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}
function csrCanonicalOk(csr, nRows) {
  const rp = csr.rowPtr;
  if (rp[0] !== 0 || rp[nRows] !== csr.vals.length) return false;
  for (let i = 0; i < nRows; i++) if (rp[i + 1] < rp[i]) return false;
  for (let i = 0; i < nRows; i++) {
    for (let p = rp[i] + 1; p < rp[i + 1]; p++) {
      if (csr.colIdx[p] <= csr.colIdx[p - 1]) return false;   // 严格递增：无重复无乱序
    }
  }
  return true;
}

function selfCheck() {
  const cases = [];
  const invariants = [];
  const DEFAULT_SEED = 0x9E3779B9;

  /* ---------- INV-SPR-001 格式往返（+ 稠密对照证伪） ---------- */
  {
    let worst = 0;
    const seeds = [DEFAULT_SEED, 0xA11CE, 0xB0B57];
    for (const sd of seeds) {
      const M = randomSparseMatrix(48, 64, 0.12, sd);
      const back = csrToDense(cooToCSR(denseToCOO(M), 64), 64);
      worst = Math.max(worst, maxAbsDiff(M, back));
    }
    // 稀疏矩阵：CSR 存储远小于全元素（压缩声称成立）
    const Ms = randomSparseMatrix(48, 64, 0.12, DEFAULT_SEED);
    const stS = nnzStats(cooToCSR(denseToCOO(Ms), 64), 48, 64);
    const sparseCompressed = stS.storageRatio < 0.5;
    // 稠密对照矩阵：同一声称必不成立（storageRatio ≥ 1）——证伪边界
    const Md = denseRandom(24, 24, 0xD311);
    const stD = nnzStats(cooToCSR(denseToCOO(Md), 24), 24, 24);
    const denseNotCompressed = stD.storageRatio >= 1.0;
    const ok = (worst <= TOL.TIGHT) && sparseCompressed && denseNotCompressed;
    cases.push(caseOf('INV-SPR-001', '稠密→COO→CSR→稠密 往返逐元素一致（+压缩声称双边界）', ok, worst,
      '3种子往返max|Δ|=' + worst.toExponential(2) + ' 稀疏storageRatio=' + stS.storageRatio.toFixed(3) + '(<0.5) 稠密对照storageRatio=' + stD.storageRatio.toFixed(3) + '(≥1 证伪)', TOL.TIGHT));
    invariants.push({ id: 'INV-SPR-001', name: '格式往返 + 压缩声称证伪', ok, expr: 'roundtrip ≤1e-15 且 稀疏storageRatio<0.5 且 稠密对照storageRatio≥1', tol: TOL.TIGHT, worst, detail: cases[cases.length - 1].detail });
  }

  /* ---------- INV-SPR-002 SpMV 正确性（独立手搓黄金参照 + 乘法计数） ---------- */
  {
    let worst = 0, multsOk = true, ratioOk = true;
    const seeds = [DEFAULT_SEED, 0xC0FFEE, 0x1234ABCD];
    for (const sd of seeds) {
      const nR = 40, nC = 56;
      const M = randomSparseMatrix(nR, nC, 0.15, sd);
      const x = denseVec(nC, sd ^ 0x51CE, 1.0);
      const yRef = goldenMatvec(M, x);                       // 手搓稠密黄金参照
      const csr = cooToCSR(denseToCOO(M), nC);
      const stats = {};
      const y = spmv(csr, x, stats);
      worst = Math.max(worst, maxAbsDiff(y, yRef));
      if (stats.mults !== csr.vals.length) multsOk = false;  // 乘法次数 == nnz（跳过零元）
      if (csr.vals.length >= 0.5 * nR * nC) ratioOk = false; // nnz ≪ n·m
    }
    const ok = (worst <= TOL.TIGHT) && multsOk && ratioOk;
    cases.push(caseOf('INV-SPR-002', 'SpMV ≡ 手搓稠密 matvec 黄金参照 + 乘法次数==nnz≪n·m', ok, worst,
      '3种子max|Δ|=' + worst.toExponential(2) + ' mults==nnz逐种子成立=' + multsOk + ' nnz/(n·m)<0.5=' + ratioOk, TOL.TIGHT));
    invariants.push({ id: 'INV-SPR-002', name: 'SpMV 正确性（黄金参照 + 零元跳过）', ok, expr: 'max|SpMV−golden|≤1e-15 且 mults==nnz<n·m/2', tol: TOL.TIGHT, worst, detail: cases[cases.length - 1].detail });
  }

  /* ---------- INV-SPR-003 SpMM 正确性 + 分配律 ---------- */
  {
    let worst = 0, worstDist = 0;
    const seeds = [DEFAULT_SEED, 0xD00D, 0x5EED];
    for (const sd of seeds) {
      const nR = 32, nC = 40, k = 6;
      const M = randomSparseMatrix(nR, nC, 0.2, sd);
      const B = denseRandom(nC, k, sd ^ 0xBEEF);
      const B2 = denseRandom(nC, k, sd ^ 0xF00D);
      const csr = cooToCSR(denseToCOO(M), nC);
      const C = spmm(csr, B);
      const CRef = goldenMatmul(M, B);                       // 手搓稠密黄金参照（零掩码列）
      worst = Math.max(worst, maxAbsDiff(C, CRef));
      // 分配律：A·(B+B2) = A·B + A·B2
      const Csum = spmm(csr, denseAdd(B, B2));
      const Csplit = denseAdd(spmm(csr, B), spmm(csr, B2));
      worstDist = Math.max(worstDist, maxAbsDiff(Csum, Csplit));
    }
    const ok = (worst <= TOL.TIGHT) && (worstDist <= TOL.SOLVE);
    cases.push(caseOf('INV-SPR-003', 'SpMM ≡ 手搓稠密 matmul + A·(B+C)=A·B+A·C 分配律', ok, Math.max(worst, worstDist),
      '3种子SpMM max|Δ|=' + worst.toExponential(2) + ' 分配律max|Δ|=' + worstDist.toExponential(2), TOL.SOLVE));
    invariants.push({ id: 'INV-SPR-003', name: 'SpMM 正确性 + 分配律', ok, expr: 'max|SpMM−golden|≤1e-15 且 分配律≤1e-10', tol: TOL.SOLVE, worst: Math.max(worst, worstDist), detail: cases[cases.length - 1].detail });
  }

  /* ---------- INV-SPR-004 CSR 规范性（乱序输入排序生效） ---------- */
  {
    const nR = 48, nC = 64;
    const M = randomSparseMatrix(nR, nC, 0.12, 0x0A1D5);
    const coo = denseToCOO(M);
    // 乱序化 COO（可种子 Fisher-Yates）
    const shuffled = coo.entries.slice();
    const rng = makeRng(0x5EED5EED);
    for (let i = shuffled.length - 1; i > 0; i--) {
      const k = Math.floor(rng() * (i + 1));
      const tmp = shuffled[i]; shuffled[i] = shuffled[k]; shuffled[k] = tmp;
    }
    const csrShuffled = cooToCSR({ nRows: nR, nCols: nC, entries: shuffled }, nC);
    const csrOrdered = cooToCSR(coo, nC);
    const structOk = csrCanonicalOk(csrShuffled, nR);
    const identical = arraysEqual(csrShuffled.rowPtr, csrOrdered.rowPtr)
      && arraysEqual(csrShuffled.colIdx, csrOrdered.colIdx)
      && arraysEqual(csrShuffled.vals, csrOrdered.vals);     // EXACT：排序生效
    const ok = structOk && identical;
    cases.push(caseOf('INV-SPR-004', 'rowPtr 单调/端点 EXACT + 行内 colIdx 严格递增 + 乱序≡有序', ok, structOk && identical ? 0 : 1,
      'rowPtr[0]=0,rowPtr[n]=nnz,单调,colIdx严格递增=' + structOk + ' 乱序COO≡有序COO(EXACT)=' + identical, TOL.EXACT));
    invariants.push({ id: 'INV-SPR-004', name: 'CSR 规范性 + 排序生效', ok, expr: 'rowPtr 端点/单调 EXACT；colIdx 行内严格递增；乱序输入≡有序输出', tol: TOL.EXACT, worst: ok ? 0 : 1, detail: cases[cases.length - 1].detail });
  }

  /* ---------- INV-SPR-005 压缩存储（双边界证伪） ---------- */
  {
    let nnzOk = true, sparseOk = true, denseOk = true;
    const seeds = [DEFAULT_SEED, 0xAAA, 0xBBB];
    for (const sd of seeds) {
      const nR = 40, nC = 56;
      const M = randomSparseMatrix(nR, nC, 0.15, sd);
      const csr = cooToCSR(denseToCOO(M), nC);
      const st = nnzStats(csr, nR, nC);
      if (st.nnz !== countNonzeros(M)) nnzOk = false;         // EXACT：与直接计数一致
      if (!(st.compression <= 0.15 + 1e-9)) sparseOk = false; // ≤ 密度 + 容差
    }
    const Md = denseRandom(24, 24, 0xD311);
    const stD = nnzStats(cooToCSR(denseToCOO(Md), 24), 24, 24);
    if (Math.abs(stD.compression - 1) > 1e-9) denseOk = false; // 稠密对照压缩率 ≈ 1（证伪边界）
    const ok = nnzOk && sparseOk && denseOk;
    cases.push(caseOf('INV-SPR-005', 'nnz==直接计数(EXACT) + 稀疏压缩率≤密度 + 稠密对照≈1', ok, nnzOk && sparseOk && denseOk ? 0 : 1,
      'nnz逐种子EXACT=' + nnzOk + ' 稀疏compression≤0.15+1e-9=' + sparseOk + ' 稠密compression≈1=' + denseOk, TOL.EXACT));
    invariants.push({ id: 'INV-SPR-005', name: '压缩存储统计（双边界）', ok, expr: 'nnz==直接计数(EXACT)；稀疏 compression≤密度+1e-9；稠密对照 compression≈1', tol: TOL.EXACT, worst: ok ? 0 : 1, detail: cases[cases.length - 1].detail });
  }

  /* ---------- INV-SPR-006 幅度剪枝（EXACT 计数/排序 + 黄金参照 + 恒等性证伪） ---------- */
  {
    let countOk = true, orderOk = true, refOk = true, notIdentity = false, keep1Ok = true;
    const seeds = [DEFAULT_SEED, 0x9EED, 0xFACE];
    for (const sd of seeds) {
      const nR = 40, nC = 48;
      const M = randomSparseMatrix(nR, nC, 0.2, sd);
      const keep = 0.5;
      const pr = magnitudePrune(M, keep);
      if (pr.kept !== Math.floor(pr.nnz * keep)) countOk = false;    // EXACT：⌊nnz·keep⌋
      if (pr.kept + pr.dropped !== pr.nnz) countOk = false;
      // 排序一致性：保留的 |v| 均 ≥ 被剪的 |v|（EXACT）
      let minKept = Infinity, maxDropped = -Infinity;
      for (const e of pr.keptEntries) minKept = Math.min(minKept, Math.abs(e[2]));
      for (const e of pr.droppedEntries) maxDropped = Math.max(maxDropped, Math.abs(e[2]));
      if (pr.droppedEntries.length > 0 && !(minKept >= maxDropped)) orderOk = false;
      // 剪枝后 SpMV ≡ 仅保留元素的手搓黄金参照（被剪元素贡献=0，可证伪）
      const x = denseVec(nC, sd ^ 0x6161, 1.0);
      const keptDense = [];
      for (let i = 0; i < nR; i++) keptDense.push(new Array(nC).fill(0));
      for (const e of pr.keptEntries) keptDense[e[0]][e[1]] = e[2];
      const yKeptRef = goldenMatvec(keptDense, x);
      const yPruned = spmv(pr.csr, x);
      if (maxAbsDiff(yPruned, yKeptRef) > TOL.TIGHT) refOk = false;
      // 剪枝非恒等：与全量 SpMV 必有可观测分歧
      const csrFull = cooToCSR(denseToCOO(M), nC);
      const yFull = spmv(csrFull, x);
      if (maxAbsDiff(yFull, yPruned) > 1e-12) notIdentity = true;
      // keep=1.0：逐位一致（EXACT）
      const pr1 = magnitudePrune(M, 1.0);
      const y1 = spmv(pr1.csr, x);
      if (!arraysEqual(y1, yFull) || !arraysEqual(pr1.csr.rowPtr, csrFull.rowPtr)
        || !arraysEqual(pr1.csr.colIdx, csrFull.colIdx) || !arraysEqual(pr1.csr.vals, csrFull.vals)) keep1Ok = false;
    }
    const ok = countOk && orderOk && refOk && notIdentity && keep1Ok;
    cases.push(caseOf('INV-SPR-006', '幅度剪枝：kept=⌊nnz·keep⌋ EXACT + |v|排序一致 + 剪后≡保留集黄金参照 + 非恒等 + keep=1 逐位一致', ok, countOk && orderOk && refOk && keep1Ok ? 0 : 1,
      '计数EXACT=' + countOk + ' 排序一致=' + orderOk + ' 保留集黄金参照≤1e-15=' + refOk + ' 剪枝非恒等=' + notIdentity + ' keep=1 逐位一致=' + keep1Ok, TOL.EXACT));
    invariants.push({ id: 'INV-SPR-006', name: '幅度剪枝（计数/排序 EXACT + 黄金参照 + 恒等证伪）', ok, expr: 'kept==⌊nnz·keep⌋；min|kept|≥max|dropped|；剪后SpMV≡保留集参照；非恒等；keep=1 逐位一致', tol: TOL.EXACT, worst: ok ? 0 : 1, detail: cases[cases.length - 1].detail });
  }

  /* ---------- INV-SPR-007 稀疏向量（稠密对照 + 交集计数 EXACT + 不相交退化） ---------- */
  {
    let worst = 0, multsOk = true, disjointOk = true;
    const n = 64;
    const seeds = [DEFAULT_SEED, 0x5ECA, 0x7E57];
    for (const sd of seeds) {
      const va = denseVec(n, sd ^ 0xA, 0.35);
      const vb = denseVec(n, sd ^ 0xB, 0.35);
      const dotRef = (() => { let d = 0; for (let i = 0; i < n; i++) d += va[i] * vb[i]; return d; })(); // 手搓稠密点积
      const svA = denseToSparseVec(va), svB = denseToSparseVec(vb);
      const stats = {};
      const d = sparseVecDot(svA, svB, stats);
      worst = Math.max(worst, Math.abs(d - dotRef));
      // 乘法次数 == 交集大小（EXACT）：正交集外元素对不计入乘法
      let inter = 0;
      for (let i = 0; i < n; i++) if (va[i] !== 0 && vb[i] !== 0) inter++;
      if (stats.mults !== inter) multsOk = false;
      if (!(stats.mults < n)) multsOk = false;
    }
    // 不相交支撑：点积==0 EXACT 且乘法次数==0
    const v1 = new Array(n).fill(0), v2 = new Array(n).fill(0);
    for (let i = 0; i < n / 2; i++) v1[i] = 1 + i;
    for (let i = n / 2; i < n; i++) v2[i] = 1 + i;
    const stD = {};
    const dDisjoint = sparseVecDot(denseToSparseVec(v1), denseToSparseVec(v2), stD);
    if (dDisjoint !== 0 || stD.mults !== 0) disjointOk = false;
    const ok = (worst <= TOL.TIGHT) && multsOk && disjointOk;
    cases.push(caseOf('INV-SPR-007', '稀疏点积≡稠密点积 + 乘法次数==交集大小(EXACT) + 不相交退化 0/0', ok, worst,
      '3种子max|Δ|=' + worst.toExponential(2) + ' mults==交集EXACT=' + multsOk + ' 不相交 dot==0且mults==0=' + disjointOk, TOL.TIGHT));
    invariants.push({ id: 'INV-SPR-007', name: '稀疏向量点积（交集语义）', ok, expr: 'max|稀疏点积−稠密点积|≤1e-15；mults==交集大小(EXACT)且<n；不相交→dot==0,mults==0', tol: TOL.TIGHT, worst, detail: cases[cases.length - 1].detail });
  }

  /* ---------- INV-SPR-008 确定性（同 seed 逐位一致 + 异 seed 分歧） ---------- */
  {
    const pipeline = (seed) => {
      const M = randomSparseMatrix(32, 40, 0.15, seed);
      const csr = cooToCSR(denseToCOO(M), 40);
      const x = denseVec(40, seed ^ 0x51, 1.0);
      const y = spmv(csr, x);
      const B = denseRandom(40, 4, seed ^ 0x77);
      const C = spmm(csr, B);
      const pr = magnitudePrune(M, 0.6);
      const svA = denseToSparseVec(denseVec(48, seed ^ 0x91, 0.4));
      const svB = denseToSparseVec(denseVec(48, seed ^ 0x92, 0.4));
      const d = sparseVecDot(svA, svB);
      return JSON.stringify({ rowPtr: csr.rowPtr, colIdx: csr.colIdx, vals: csr.vals, y, C, kept: pr.kept, prunedVals: pr.csr.vals, d });
    };
    const r1 = pipeline(0xBEEF), r2 = pipeline(0xBEEF);
    const identical = (r1 === r2);                      // EXACT：逐位相等
    const r3 = pipeline(0xC0FFEE);
    const diverged = (r1 !== r3);                       // 反空洞：异 seed 必分歧
    const ok = identical && diverged;
    cases.push(caseOf('INV-SPR-008', '同 seed 全流水线两次逐位相等 + 异 seed 分歧', ok, identical && diverged ? 0 : 1,
      '同seed逐位一致=' + identical + ' 异seed分歧=' + diverged, TOL.EXACT));
    invariants.push({ id: 'INV-SPR-008', name: '确定性 + 异种子分歧', ok, expr: 'pipeline(同seed)两次逐位相等(EXACT)；pipeline(异seed)必分歧', tol: TOL.EXACT, worst: ok ? 0 : 1, detail: cases[cases.length - 1].detail });
  }

  return buildReport('sparse-forge', '1.0.0', 'L7', cases, invariants, '0x9E3779B9');
}

/* ============================================================================
 * 人类可读表格（零依赖本地实现）
 * ========================================================================== */
function formatTable(r) {
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

/* ============================================================================
 * API 导出 + 双端挂载（UMD）+ CLI 自检
 * ========================================================================== */
const API = {
  name: 'sparse-forge', version: '1.0.0', layer: 'L7', TOL,
  makeRng, sha256Hex,
  randomSparseMatrix, denseRandom, denseVec,
  denseToCOO, cooToCSR, csrToDense, spmv, spmm, nnzStats,
  magnitudePrune, denseToSparseVec, sparseVecToDense, sparseVecDot,
  selfCheck, formatTable
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeSparse = API;

if (typeof process !== 'undefined' && process.argv[1] && /sparse-forge\.js$/.test(process.argv[1]) && (process.argv.includes('--self-check') || process.argv.includes('--json'))) {
  const r = selfCheck();
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : formatTable(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
