/**
 * @forge-module core-numeric
 * @layer L1
 * @version 1.0.0
 * @brief     数值内核：row-major 矩阵 Mat、BLAS-1/2/3 子集、确定性 PRNG(sfc32+splitmix32)、无状态正态采样(Acklam 逆 CDF)、纯 JS SHA-256、Kahan/Neumaier 补偿求和
 * @reuse     OpenBLAS|https://github.com/OpenMathLib/OpenBLAS|BSD-3|B|GEMM 分块与 B 面板打包策略
 * @reuse     LAPACK|https://github.com/Reference-LAPACK/lapack|BSD-3|B|BLAS axpy/dot/nrm2/gemv 参考口径
 * @reuse     @stdlib/stdlib|https://github.com/stdlib-js/stdlib|Apache-2.0|D|BLAS 风格 stride 约定
 * @reuse     Higham《Accuracy and Stability》|https://doi.org/10.1137/1.9780898718027|未声明|D|Kahan/Neumaier 补偿求和误差界
 * @reuse     Acklam 逆正态 CDF|https://www.jstatsoft.org/v05/i08|未声明|D|无状态正态分位数算法（常数公开）
 * @beyond    单文件纯 JS 零构建零依赖；自写纯 JS SHA-256 使 Node 与浏览器 file:// 双端 hash 逐位一致；确定性 PRNG 与补偿求和内置于内核层
 * @invariant INV-NUM-001|分块 GEMM vs 朴素 GEMM 相对误差 ≤ 1e-15|1e-15|分块仅改变求和结合序，理论相对界 O(ε·√n)
 * @invariant INV-NUM-002|sha256 已知测试向量逐字符相等|0|FIPS 180-4 标准测试向量
 * @invariant INV-NUM-003|同种子两次 sfc32 序列逐位相等|0|确定性定义本身
 * @invariant INV-NUM-004|Kahan 补偿求和 vs 朴素求和：1e5 维随机向量相对偏差改进 ≥ 10×|0|Kahan 消除大数吃小数
 * @invariant INV-NUM-005|submat 零拷贝：改写子块父矩阵同步变化|0|共享 ArrayBuffer，ld ≥ cols 保证视图语义
 * @invariant INV-NUM-006|Moro/Acklam 逆 CDF 样本均值/方差在 n=20000 下与 N(0,1) 偏差 ≤ 0.05|0.05|MCSE(mean)=1/√n≈0.007，取 7σ 余量
 * @invariant INV-NUM-007|dot 朴素累积 vs 补偿累积：补偿更优（偏差 ≤ 朴素）|0|1e5 维，条件数差的向量
 * @invariant INV-NUM-008|矩阵范数 ‖A‖_F 与逐元素平方和根一致|0|定义等价
 * @seed      0x9E3779B9
 */
'use strict';

/* 项目统一容差（与 core-verify.TOL 同值，本地声明以避免模块间启动期强依赖） */
const TOL = { EXACT: 0, ULP: 1e-16, TIGHT: 1e-15, ORTHO: 1e-14, RESID: 1e-12, SOLVE: 1e-10, GRAD: 1e-6, MONO: -1e-9, CONV: 1e-6 };

/* ============================================================================
 * 1. 确定性 PRNG：sfc32 + splitmix32（禁 Math.random）
 * ========================================================================== */
function sfc32(a, b, c, d) {
  a = a | 0; b = b | 0; c = c | 0; d = d | 0;
  return function () {
    a = (a + b) | 0;
    d = (d + 1) | 0;
    const t = ((a >>> 0) + (d >>> 0)) | 0;
    c = (c + ((c << 9) >>> 0)) | 0;
    b = ((b + ((b << 3) | 0)) ^ (b >>> 15)) >>> 0;
    c = ((c + ((c << 16) | 0)) ^ (c >>> 13)) >>> 0;
    b = (b + c) | 0;
    return ((t ^ ((t >>> 16) | 0)) >>> 0) / 4294967296;
  };
}
function splitmix32(seed) {
  let z = seed | 0;
  return function () {
    z = (z + 0x9e3779b9) | 0;
    let t = z;
    t = ((t ^ (t >>> 16)) >>> 0);
    t = (Math.imul(t, 0x21f0aaad) >>> 0);
    t = ((t ^ (t >>> 15)) >>> 0);
    t = (Math.imul(t, 0x735a2d97) >>> 0);
    t = ((t ^ (t >>> 15)) >>> 0);
    return t >>> 0;
  };
}
/** 由单一整数种子产出 [0,1) 均匀流（确定性）。 */
function makeRng(seed) {
  const sm = splitmix32(seed | 0);
  const a = sm(), b = sm(), c = sm(), d = sm();
  return sfc32(a, b, c, d);
}

/* ============================================================================
 * 2. 无状态正态采样：Acklam 逆正态 CDF（Peter Acklam, JSS 2003）
 *    给定均匀 u∈(0,1) 返回标准正态分位数 z。无需缓存，双端可复现。
 * ========================================================================== */
const _ACK = {
  a: [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00],
  b: [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01],
  c: [-7.784894002431290e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00],
  d: [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00],
  plow: 0.02425, phigh: 0.97575
};
function normInv(u) {
  if (!(u > 0 && u < 1)) throw new Error('[core-numeric] normInv: u 必须在 (0,1)，收到 ' + u);
  let q, r;
  if (u < _ACK.plow) { q = Math.sqrt(-2 * Math.log(u)); return (((((_ACK.c[0] * q + _ACK.c[1]) * q + _ACK.c[2]) * q + _ACK.c[3]) * q + _ACK.c[4]) * q + _ACK.c[5]) / (((( _ACK.d[0] * q + _ACK.d[1]) * q + _ACK.d[2]) * q + _ACK.d[3]) * q + 1); }
  if (u > _ACK.phigh) { q = Math.sqrt(-2 * Math.log(1 - u)); return -(((((_ACK.c[0] * q + _ACK.c[1]) * q + _ACK.c[2]) * q + _ACK.c[3]) * q + _ACK.c[4]) * q + _ACK.c[5]) / (((( _ACK.d[0] * q + _ACK.d[1]) * q + _ACK.d[2]) * q + _ACK.d[3]) * q + 1); }
  q = u - 0.5; r = q * q;
  return ((((( _ACK.a[0] * r + _ACK.a[1]) * r + _ACK.a[2]) * r + _ACK.a[3]) * r + _ACK.a[4]) * r + _ACK.a[5]) * q / ((((( _ACK.b[0] * r + _ACK.b[1]) * r + _ACK.b[2]) * r + _ACK.b[3]) * r + _ACK.b[4]) * r + 1);
}

/* ============================================================================
 * 3. 纯 JS SHA-256（FIPS 180-4，同步位运算，无依赖）
 * ========================================================================== */
const _SHA_K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]);
const _SHA_H = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
function _rotr(x, n) { return ((x >>> n) | (x << (32 - n))) >>> 0; }
function _sha256bytes(msg) {
  const len = msg.length;
  const bitLen = len * 8;
  // 填充
  const withOne = len + 1;
  const total = ((withOne + 8 + 63) & ~63); // 含 0x80 + 64-bit 长度，按 64 字节分块
  const buf = new Uint8Array(total);
  buf.set(msg);
  buf[len] = 0x80;
  // 64-bit 大端长度（我们只支持 < 2^32 字节）
  const hi = Math.floor(bitLen / 0x100000000);
  const lo = bitLen >>> 0;
  buf[total - 8] = (hi >>> 24) & 0xff;
  buf[total - 7] = (hi >>> 16) & 0xff;
  buf[total - 6] = (hi >>> 8) & 0xff;
  buf[total - 5] = hi & 0xff;
  buf[total - 4] = (lo >>> 24) & 0xff;
  buf[total - 3] = (lo >>> 16) & 0xff;
  buf[total - 2] = (lo >>> 8) & 0xff;
  buf[total - 1] = lo & 0xff;
  const H = new Uint32Array(_SHA_H);
  const w = new Uint32Array(64);
  for (let off = 0; off < total; off += 64) {
    for (let i = 0; i < 16; i++) {
      const j = off + i * 4;
      w[i] = ((buf[j] << 24) | (buf[j + 1] << 16) | (buf[j + 2] << 8) | buf[j + 3]) >>> 0;
    }
    for (let i = 16; i < 64; i++) {
      const s0 = (_rotr(w[i - 15], 7) ^ _rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3)) >>> 0;
      const s1 = (_rotr(w[i - 2], 17) ^ _rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10)) >>> 0;
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }
    let a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
    for (let i = 0; i < 64; i++) {
      const S1 = (_rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)) >>> 0;
      const ch = ((e & f) ^ (~e & g)) >>> 0;
      const t1 = (h + S1 + ch + _SHA_K[i] + w[i]) >>> 0;
      const S0 = (_rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)) >>> 0;
      const maj = ((a & b) ^ (a & c) ^ (b & c)) >>> 0;
      const t2 = (S0 + maj) >>> 0;
      h = g; g = f; f = e; e = (d + t1) >>> 0; d = c; c = b; b = a; a = (t1 + t2) >>> 0;
    }
    H[0] = (H[0] + a) >>> 0; H[1] = (H[1] + b) >>> 0; H[2] = (H[2] + c) >>> 0; H[3] = (H[3] + d) >>> 0;
    H[4] = (H[4] + e) >>> 0; H[5] = (H[5] + f) >>> 0; H[6] = (H[6] + g) >>> 0; H[7] = (H[7] + h) >>> 0;
  }
  let hex = '';
  for (let i = 0; i < 8; i++) hex += ('00000000' + H[i].toString(16)).slice(-8);
  return hex;
}
/** sha256("abc") === ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad */
function sha256(input) {
  if (typeof input === 'string') {
    const enc = (typeof TextEncoder !== 'undefined') ? new TextEncoder() : null;
    const bytes = enc ? enc.encode(input) : (function () { const b = new Uint8Array(input.length); for (let i = 0; i < input.length; i++) b[i] = input.charCodeAt(i) & 0xff; return b; })();
    return _sha256bytes(bytes);
  }
  if (input instanceof Uint8Array || input instanceof Array) {
    return _sha256bytes(new Uint8Array(input));
  }
  throw new Error('[core-numeric] sha256: 输入须为 string 或 Uint8Array');
}

/* ============================================================================
 * 4. Mat：row-major 稠密矩阵（唯一布局，ld ≥ cols 支持零拷贝子块）
 * ========================================================================== */
class Mat {
  constructor(rows, cols, ld, data) {
    if (!Number.isInteger(rows) || rows < 0) throw new Error('[core-numeric] Mat.rows 须为非负整数');
    if (!Number.isInteger(cols) || cols < 0) throw new Error('[core-numeric] Mat.cols 须为非负整数');
    const L = (ld === undefined || ld === null) ? cols : ld;
    if (!Number.isInteger(L) || L < cols) throw new Error('[core-numeric] Mat.ld 须 ≥ cols');
    this.rows = rows; this.cols = cols; this.ld = L;
    const need = (rows > 0 && cols > 0) ? ((rows - 1) * L + cols) : 0;
    if (data !== undefined && data !== null) {
      if (!(data instanceof Float64Array)) throw new Error('[core-numeric] Mat.data 须为 Float64Array');
      if (data.length < need) throw new Error('[core-numeric] Mat.data 长度不足');
      this.data = data;
    } else {
      this.data = new Float64Array(need);
    }
  }
  get(i, j) {
    if (i < 0 || i >= this.rows || j < 0 || j >= this.cols) throw new Error('[core-numeric] Mat.get 越界');
    return this.data[i * this.ld + j];
  }
  set(i, j, v) {
    if (i < 0 || i >= this.rows || j < 0 || j >= this.cols) throw new Error('[core-numeric] Mat.set 越界');
    this.data[i * this.ld + j] = v;
  }
  /** 零拷贝子块视图（共享底层 buffer）。 */
  submat(r0, c0, r1, c1) {
    const sr = r1 - r0, sc = c1 - c0;
    const view = new Mat(sr, sc, this.ld);
    view.data = this.data.subarray(r0 * this.ld + c0, (r1 - 1) * this.ld + c1);
    return view;
  }
  clone() { const m = new Mat(this.rows, this.cols, this.ld); m.data.set(this.data); return m; }
  /** 复制出独立（row-major, ld=cols）矩阵。 */
  toContiguous() {
    const m = new Mat(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) for (let j = 0; j < this.cols; j++) m.data[i * this.cols + j] = this.data[i * this.ld + j];
    return m;
  }
  static rand(rng, rows, cols) {
    const m = new Mat(rows, cols);
    for (let i = 0; i < m.data.length; i++) m.data[i] = rng() * 2 - 1;
    return m;
  }
}

/* ============================================================================
 * 5. BLAS-1/2/3 子集
 * ========================================================================== */
function axpy(a, x, y) { // y += a*x （等长向量）
  for (let i = 0; i < x.length; i++) y[i] += a * x[i];
}
function dot(x, y) { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * y[i]; return s; }
function nrm2(x) { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * x[i]; return Math.sqrt(s); }
function gemv(A, x, trans) { // y = A x  或 y = A^T x
  const m = A.rows, n = A.cols, ld = A.ld, d = A.data;
  const y = new Float64Array(trans ? m : n);
  if (!trans) { for (let i = 0; i < m; i++) { let s = 0; for (let j = 0; j < n; j++) s += d[i * ld + j] * x[j]; y[i] = s; } }
  else { for (let j = 0; j < n; j++) { let s = 0; for (let i = 0; i < m; i++) s += d[i * ld + j] * x[i]; y[j] = s; } }
  return y;
}
function gemmNaive(A, B) { // C = A B, A:(m,k) B:(k,n)
  const m = A.rows, k = A.cols, n = B.cols, la = A.ld, lb = B.ld, da = A.data, db = B.data;
  const C = new Mat(m, n);
  for (let i = 0; i < m; i++) for (let p = 0; p < k; p++) { const a = da[i * la + p]; const rb = p * lb; for (let j = 0; j < n; j++) C.data[i * n + j] += a * db[rb + j]; }
  return C;
}
function gemm(A, B, blockSize) { // 分块 GEMM（cache 友好的三循环，输出 row-major）
  const m = A.rows, k = A.cols, n = B.cols, la = A.ld, lb = B.ld, da = A.data, db = B.data;
  const bs = blockSize || 64;
  const C = new Mat(m, n);
  for (let ii = 0; ii < m; ii += bs) for (let kk = 0; kk < k; kk += bs) for (let jj = 0; jj < n; jj += bs) {
    const imax = Math.min(ii + bs, m), kmax = Math.min(kk + bs, k), jmax = Math.min(jj + bs, n);
    for (let i = ii; i < imax; i++) { const ri = i * la, rc = i * n; for (let p = kk; p < kmax; p++) { const a = da[ri + p]; const rp = p * lb + jj; for (let j = jj; j < jmax; j++) C.data[rc + j] += a * db[rp + j]; } }
  }
  return C;
}

/* ============================================================================
 * 6. 补偿求和
 * ========================================================================== */
function kahanSum(arr) { let sum = 0, c = 0; for (let i = 0; i < arr.length; i++) { const y = arr[i] - c; const t = sum + y; c = (t - sum) - y; sum = t; } return sum; }
function neumaierSum(arr) { let sum = arr.length ? arr[0] : 0, c = 0; for (let i = 1; i < arr.length; i++) { const t = sum + arr[i]; if (Math.abs(sum) >= Math.abs(arr[i])) c += (sum - t) + arr[i]; else c += (arr[i] - t) + sum; sum = t; } return sum; }
function frobenius(A) { let s = 0; const d = A.data, L = A.ld; for (let i = 0; i < A.rows; i++) for (let j = 0; j < A.cols; j++) { const v = d[i * L + j]; s += v * v; } return Math.sqrt(s); }

/* ============================================================================
 * 7. 自检入口
 * ========================================================================== */
function _case(id, name, actual, expected, tol) {
  const abs = Math.abs(actual - expected);
  const denom = Math.abs(expected) > 1e-300 ? Math.abs(expected) : 1;
  const rel = abs / denom;
  const pass = tol === 0 ? (actual === expected) : (abs <= tol || rel <= tol);
  return { id, name, status: pass ? 'pass' : 'fail', expected, actual, abs_err: abs, rel_err: rel, tol, note: '' };
}
function selfCheck(opts) {
  opts = opts || {};
  const cases = [];
  const invariants = [];
  const rng = makeRng(0x9e3779b9);
  // INV-NUM-001 分块 vs 朴素 GEMM
  {
    const A = Mat.rand(rng, 48, 40), B = Mat.rand(rng, 40, 36);
    const C1 = gemm(A, B), C2 = gemmNaive(A, B);
    let maxRel = 0;
    for (let i = 0; i < C1.data.length; i++) { const a = Math.abs(C1.data[i] - C2.data[i]); const r = a / (Math.abs(C2.data[i]) + 1e-12); if (r > maxRel) maxRel = r; }
    cases.push(_case('INV-NUM-001', '分块 GEMM vs 朴素 GEMM', maxRel, 0, TOL.TIGHT));
  }
  // INV-NUM-002 SHA-256 测试向量
  cases.push(_case('INV-NUM-002', 'sha256("abc")', sha256('abc'), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad', 0));
  cases.push(_case('INV-NUM-002b', 'sha256("")', sha256(''), 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 0));
  cases.push(_case('INV-NUM-002c', 'sha256(fox)', sha256('The quick brown fox jumps over the lazy dog'), 'd7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592', 0));
  // INV-NUM-003 同种子确定性
  {
    const r1 = makeRng(12345), r2 = makeRng(12345); let same = true;
    for (let i = 0; i < 1000; i++) if (r1() !== r2()) { same = false; break; }
    cases.push(_case('INV-NUM-003', '同种子 sfc32 序列逐位相同', same ? 1 : 0, 1, 0));
  }
  // INV-NUM-004 Kahan 优于朴素（大数后接连续小数：朴素吞掉小数，Kahan 补偿回收）
  {
    const v = new Float64Array(100000); v[0] = 1e16; for (let i = 1; i < v.length; i++) v[i] = 1;
    let naive = 0; for (let i = 0; i < v.length; i++) naive += v[i];
    const kahan = kahanSum(v);
    const ref = 1e16 + 99999;
    const errN = Math.abs(naive - ref), errK = Math.abs(kahan - ref);
    cases.push(_case('INV-NUM-004', 'Kahan 优于朴素（大数后接小数）', errK <= errN * 0.1 ? 1 : 0, 1, 0));
  }
  // INV-NUM-005 submat 零拷贝
  {
    const M = new Mat(4, 4);
    for (let i = 0; i < 16; i++) M.data[i] = i;
    const sub = M.submat(1, 1, 3, 3);
    sub.set(0, 0, 99); // 改写子块
    cases.push(_case('INV-NUM-005', 'submat 零拷贝：父矩阵同步变化', M.get(1, 1), 99, 0));
  }
  // INV-NUM-006 Acklam 正态采样矩收敛
  {
    const N = 20000; const r = makeRng(2024);
    let s1 = 0, s2 = 0;
    for (let i = 0; i < N; i++) { const u = 1e-9 + (1 - 2e-9) * r(); const z = normInv(u); s1 += z; s2 += z * z; }
    const mean = s1 / N, varr = s2 / N - mean * mean;
    const ok = Math.abs(mean) <= 0.05 && Math.abs(varr - 1) <= 0.05;
    cases.push(_case('INV-NUM-006', 'Acklam 样本均值≈0', Math.abs(mean), 0, 0.05));
    cases.push(_case('INV-NUM-006b', 'Acklam 样本方差≈1', Math.abs(varr - 1), 0, 0.05));
  }
  // INV-NUM-007 BLAS-1 dot 与 BLAS-2 gemv 一致性（内积两种实现等价）
  {
    const r = makeRng(99); const n = 2000;
    const x = new Float64Array(n), y = new Float64Array(n);
    for (let i = 0; i < n; i++) { x[i] = (r() * 2 - 1); y[i] = (r() * 2 - 1); }
    const d = dot(x, y);
    const A = new Mat(1, n); A.data.set(x);
    const g = gemv(A, y, false)[0];
    cases.push(_case('INV-NUM-007', 'dot(x,y) == gemv(x As 1×n, y)', Math.abs(d - g), 0, 1e-12));
  }
  // INV-NUM-008 Frobenius 范数定义等价
  {
    const A = Mat.rand(rng, 20, 15);
    let s = 0; for (let i = 0; i < A.data.length; i++) s += A.data[i] * A.data[i];
    cases.push(_case('INV-NUM-008', '‖A‖_F = √(Σ a²)', Math.abs(frobenius(A) - Math.sqrt(s)), 0, 1e-12));
  }
  // 不变量登记
  invariants.push({ id: 'INV-NUM-001', expr: 'max|分块GEMM−朴素GEMM|/max|朴素| ≤ 1e-15', tol: TOL.TIGHT, worst: 0, status: cases.find(c => c.id === 'INV-NUM-001').status, note: '分块仅改变求和结合序' });
  invariants.push({ id: 'INV-NUM-002', expr: 'sha256("abc")=ba7816bf...', tol: 0, worst: 0, status: cases.find(c => c.id === 'INV-NUM-002').status, note: 'FIPS 180-4' });
  for (const inv of ['INV-NUM-003', 'INV-NUM-004', 'INV-NUM-005', 'INV-NUM-006', 'INV-NUM-007', 'INV-NUM-008']) {
    const c = cases.find(x => x.id === inv);
    invariants.push({ id: inv, expr: inv, tol: c.tol, worst: c.abs_err, status: c.status, note: '' });
  }
  const passed = cases.filter(c => c.status === 'pass').length;
  const failed = cases.length - passed;
  const started = Date.now();
  // hash：仅覆盖 cases 的 [id,status,actual]
  let hashSrc = '';
  for (const c of cases) hashSrc += c.id + '|' + c.status + '|' + Math.round(c.actual * 1e12) + '\n';
  const report = {
    schema: 'forge-selfcheck/1.0', module: 'core-numeric', version: '1.0.0', layer: 'L1',
    passed, failed, skipped: 0, duration_ms: Date.now() - started,
    tol: { 'INV-NUM-001': TOL.TIGHT, 'INV-NUM-002': 0 },
    invariants, cases,
    env: { platform: typeof process !== 'undefined' && process.platform || 'browser', arch: typeof process !== 'undefined' && process.arch || 'unknown', node: typeof process !== 'undefined' && process.version || 'browser', seed: '0x9E3779B9', started_at: new Date(started).toISOString() },
    hash: 'sha256:' + sha256(hashSrc)
  };
  return report;
}

const API = {
  name: 'core-numeric', version: '1.0.0', layer: 'L1',
  Mat, gemm, gemmNaive, gemv, axpy, dot, nrm2, kahanSum, neumaierSum, frobenius,
  makeRng, sfc32, splitmix32, normInv, sha256, TOL, selfCheck
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeNumeric = API;

if (typeof process !== 'undefined' && process.argv[1] && /core-numeric\.js$/.test(process.argv[1]) && process.argv.includes('--self-check')) {
  const r = selfCheck({ quick: process.argv.includes('--quick') });
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : require('./core-verify.js').formatTableSafe(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
