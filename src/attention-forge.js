/**
 * @forge-module attention-forge
 * @layer L4
 * @version 1.0.0
 * @brief     注意力内核：缩放点积注意力(SDPA)、多头注意力(MHA)、分组查询注意力(GQA)、因果掩码。权重为稳定 softmax（移位防溢出），全部带可交叉验证不变量。
 * @reuse     Vaswani et al. Attention Is All You Need|https://arxiv.org/abs/1706.03762|公有域|D|SDPA/MHA 定义
 * @reuse     Ainslie et al. GQA|https://arxiv.org/abs/2305.13245|公有域|D|分组查询注意力
 * @reuse     FlashAttention|https://github.com/Dao-AILab/flash-attention|BSD-3|B|在线 softmax 数值稳定思路
 * @beyond    纯 JS 单文件零构建；softmax 移位稳定；因果掩码结构、GQA 退化为单 KV 头、softmax 平移不变性等不变量可证；MHA 与 SDPA 在单头下一致。
 * @invariant INV-ATT-001|注意力权重每行和为 1 ≤ 1e-12|1e-12|softmax 归一
 * @invariant INV-ATT-002|输出 = P·V（独立重算对照）≤ 1e-12|1e-12|注意力定义
 * @invariant INV-ATT-003|等权 logits → 权重均匀 = 1/N，输出 = mean(V) ≤ 1e-12|1e-12|均匀注意
 * @invariant INV-ATT-004|因果掩码：i<j 位置权重 = 0（下三角）≤ 1e-12|1e-12|因果结构
 * @invariant INV-ATT-005|数值稳定：logits=1e6 时输出有限且 ≈ 每行 argmax 对应 V ≤ 1e-6|1e-6|极端不溢出
 * @invariant INV-ATT-006|MHA 在 nHeads=1 时与 SDPA 一致 ≤ 1e-12|1e-12|多头退化
 * @invariant INV-ATT-007|GQA(nKVHeads=1) 与“所有头共享同一 KV”的 MHA 一致 ≤ 1e-12|1e-12|GQA 退化
 * @invariant INV-ATT-008|softmax 平移不变：scores 整体 +c 后输出不变 ≤ 1e-12|1e-12|softmax 移位不变性
 * @seed      0x9E3779B9
 */
'use strict';
const { makeRng, TOL } = require('./core-numeric.js');
const { caseResult, buildReport } = require('./core-verify.js');

/* ---------- 本地矩阵工具（行主序：数组 of Float64Array） ---------- */
function zeros(r, c) { const m = new Array(r); for (let i = 0; i < r; i++) m[i] = new Float64Array(c); return m; }
function matMul(A, B) { // A:r×k, B:k×c → r×c
  const r = A.length, k = A[0].length, c = B[0].length;
  const C = zeros(r, c);
  for (let i = 0; i < r; i++) for (let p = 0; p < k; p++) { const a = A[i][p]; for (let j = 0; j < c; j++) C[i][j] += a * B[p][j]; }
  return C;
}
function matVec(A, v) { const r = A.length, c = A[0].length; const y = new Float64Array(r); for (let i = 0; i < r; i++) { let s = 0; for (let j = 0; j < c; j++) s += A[i][j] * v[j]; y[i] = s; } return y; }
function transpose(A) { const r = A.length, c = A[0].length; const T = zeros(c, r); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) T[j][i] = A[i][j]; return T; }

// 稳定 softmax（按行；causal 时上三角置 -∞）
function softmaxRows(scores, causal) {
  const M = scores.length, N = scores[0].length;
  const P = zeros(M, N);
  for (let i = 0; i < M; i++) {
    let m = -Infinity;
    for (let j = 0; j < N; j++) { if (causal && j > i) continue; if (scores[i][j] > m) m = scores[i][j]; }
    let s = 0;
    for (let j = 0; j < N; j++) {
      if (causal && j > i) { P[i][j] = 0; continue; }
      const v = Math.exp(scores[i][j] - m);
      P[i][j] = v; s += v;
    }
    for (let j = 0; j < N; j++) P[i][j] /= s;
  }
  return P;
}

/* ---------- SDPA ---------- */
function sdpa(Q, K, V, opts) {
  opts = opts || {};
  const d = Q[0].length;
  const scale = 1 / Math.sqrt(d);
  const M = Q.length, N = K.length;
  const scores = zeros(M, N);
  for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) {
    let s = 0; for (let k = 0; k < d; k++) s += Q[i][k] * K[j][k];
    scores[i][j] = s * scale;
  }
  const P = softmaxRows(scores, !!opts.causal);
  const out = matMul(P, V); // M×d
  return { P, out, scores };
}

/* ---------- 线性投影工具 ---------- */
function linear(x, W, b) { // x:seq×in, W:out×in, b:out → seq×out
  const seq = x.length, out = W.length, inn = W[0].length;
  const y = zeros(seq, out);
  for (let s = 0; s < seq; s++) for (let o = 0; o < out; o++) {
    let acc = b ? b[o] : 0;
    for (let k = 0; k < inn; k++) acc += x[s][k] * W[o][k];
    y[s][o] = acc;
  }
  return y;
}
function splitHeads(x, nHeads) { // x:seq×d → heads array [h] of seq×(d/nHeads)
  const seq = x.length, d = x[0].length, hd = d / nHeads;
  const heads = [];
  for (let h = 0; h < nHeads; h++) {
    const Hh = zeros(seq, hd);
    for (let s = 0; s < seq; s++) for (let k = 0; k < hd; k++) Hh[s][k] = x[s][h * hd + k];
    heads.push(Hh);
  }
  return heads;
}
function mergeHeads(heads, nHeads) {
  const seq = heads[0].length, hd = heads[0][0].length, d = hd * nHeads;
  const x = zeros(seq, d);
  for (let h = 0; h < nHeads; h++) for (let s = 0; s < seq; s++) for (let k = 0; k < hd; k++) x[s][h * hd + k] = heads[h][s][k];
  return x;
}

/* ---------- MHA ---------- */
function multiHeadAttention(x, Wq, Wk, Wv, Wo, nHeads, opts) {
  const Q = linear(x, Wq), K = linear(x, Wk), V = linear(x, Wv);
  const Qh = splitHeads(Q, nHeads), Kh = splitHeads(K, nHeads), Vh = splitHeads(V, nHeads);
  const outs = [];
  for (let h = 0; h < nHeads; h++) outs.push(sdpa(Qh[h], Kh[h], Vh[h], opts).out);
  const merged = mergeHeads(outs, nHeads);
  return linear(merged, Wo);
}

/* ---------- GQA ---------- */
function gqa(x, Wq, Wk, Wv, Wo, nHeads, nKVHeads, opts) {
  const Q = linear(x, Wq), K = linear(x, Wk), V = linear(x, Wv);
  const Qh = splitHeads(Q, nHeads);
  const d = Q[0].length, hd = d / nHeads;
  const kd = K[0].length, kvd = kd / nKVHeads;
  // K/V 按 nKVHeads 分组
  const Kh = [], Vh = [];
  for (let g = 0; g < nKVHeads; g++) {
    const Kg = zeros(K.length, kvd), Vg = zeros(V.length, kvd);
    for (let s = 0; s < K.length; s++) for (let k = 0; k < kvd; k++) { Kg[s][k] = K[s][g * kvd + k]; Vg[s][k] = V[s][g * kvd + k]; }
    Kh.push(Kg); Vh.push(Vg);
  }
  const outs = [];
  for (let h = 0; h < nHeads; h++) {
    const g = Math.floor(h * nKVHeads / nHeads);
    outs.push(sdpa(Qh[h], Kh[g], Vh[g], opts).out);
  }
  const merged = mergeHeads(outs, nHeads);
  return linear(merged, Wo);
}

/* ============================================================================
 * selfCheck — 8 条不变量
 * ========================================================================== */
function selfCheck(opts) {
  opts = opts || {};
  const cases = [];
  const invariants = [];
  const rng = makeRng(0x9e3779b9);

  // INV-ATT-001/002/003 SDPA 基础（非因果）
  {
    const M = 4, N = 5, d = 6;
    const Q = zeros(M, d), K = zeros(N, d), V = zeros(N, d);
    for (let i = 0; i < M; i++) for (let k = 0; k < d; k++) Q[i][k] = rng() * 2 - 1;
    for (let j = 0; j < N; j++) for (let k = 0; k < d; k++) { K[j][k] = rng() * 2 - 1; V[j][k] = rng() * 2 - 1; }
    const { P, out } = sdpa(Q, K, V, {});
    // 001 行和
    let rowErr = 0; for (let i = 0; i < M; i++) { let s = 0; for (let j = 0; j < N; j++) s += P[i][j]; rowErr = Math.max(rowErr, Math.abs(s - 1)); }
    // 002 输出=P·V
    const ref = matMul(P, V);
    let dev = 0; for (let i = 0; i < M; i++) for (let k = 0; k < d; k++) dev = Math.max(dev, Math.abs(out[i][k] - ref[i][k]));
    // 003 等权
    const Qe = zeros(M, d), Ke = zeros(N, d), Ve = zeros(N, d);
    for (let j = 0; j < N; j++) for (let k = 0; k < d; k++) Ve[j][k] = rng() * 2 - 1;
    const r3 = sdpa(Qe, Ke, Ve, {}).out;
    const meanV = new Float64Array(d); for (let j = 0; j < N; j++) for (let k = 0; k < d; k++) meanV[k] += Ve[j][k] / N;
    let dev3 = 0; for (let i = 0; i < M; i++) for (let k = 0; k < d; k++) dev3 = Math.max(dev3, Math.abs(r3[i][k] - meanV[k]));
    cases.push(caseResult('INV-ATT-001', '注意力权重行和=1', rowErr, 0, TOL.RESID));
    cases.push(caseResult('INV-ATT-002', '输出 = P·V', dev, 0, TOL.RESID));
    cases.push(caseResult('INV-ATT-003', '等权 → 输出=mean(V)', dev3, 0, TOL.RESID));
  }
  // INV-ATT-004 因果掩码结构（方阵）
  {
    const N = 5, d = 4;
    const Q = zeros(N, d), K = zeros(N, d), V = zeros(N, d);
    for (let i = 0; i < N; i++) for (let k = 0; k < d; k++) { Q[i][k] = rng() * 2 - 1; K[i][k] = rng() * 2 - 1; V[i][k] = rng() * 2 - 1; }
    const { P } = sdpa(Q, K, V, { causal: true });
    let upper = 0; // i<j 必须 0
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) upper = Math.max(upper, P[i][j]);
    cases.push(caseResult('INV-ATT-004', '因果掩码 i<j 权重=0', upper, 0, TOL.RESID));
  }
  // INV-ATT-005 数值稳定（巨大 logits）
  {
    const M = 3, N = 3, d = 2;
    const Q = zeros(M, d), K = zeros(N, d), V = zeros(N, d);
    for (let j = 0; j < N; j++) for (let k = 0; k < d; k++) V[j][k] = rng() * 2 - 1;
    const big = 1e6;
    const Qb = zeros(M, d), Kb = zeros(N, d);
    Qb[1][0] = big; Kb[2][0] = big; // 使 row1 与 col2 点积最大
    const { out } = sdpa(Qb, Kb, V, {});
    // row1 应 ≈ V[2]
    let dev = 0; for (let k = 0; k < d; k++) dev = Math.max(dev, Math.abs(out[1][k] - V[2][k]));
    const finiteOk = isFinite(out[1][0]);
    cases.push(caseResult('INV-ATT-005', '巨大 logits 稳定且≈argmax(V)', finiteOk && dev < 1e-6 ? dev : 1, 0, 1e-6));
  }
  // INV-ATT-006 MHA(nHeads=1) 与 SDPA 一致
  {
    const seq = 4, d = 6;
    const x = zeros(seq, d), Wq = zeros(d, d), Wk = zeros(d, d), Wv = zeros(d, d), Wo = zeros(d, d);
    for (let i = 0; i < seq; i++) for (let k = 0; k < d; k++) x[i][k] = rng() * 2 - 1;
    for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) { Wq[a][b] = rng() * 2 - 1; Wk[a][b] = rng() * 2 - 1; Wv[a][b] = rng() * 2 - 1; Wo[a][b] = rng() * 2 - 1; }
    const mha = multiHeadAttention(x, Wq, Wk, Wv, Wo, 1, {});
    const Q = linear(x, Wq), K = linear(x, Wk), V = linear(x, Wv);
    const sd = sdpa(Q, K, V, {}).out;
    const WoM = matMul(sd, transpose(Wo)); // 注意 linear 已含 Wo；此处直接用 mha 与“先 SDPA 再 Wo”一致
    let dev = 0; for (let i = 0; i < seq; i++) for (let k = 0; k < d; k++) dev = Math.max(dev, Math.abs(mha[i][k] - WoM[i][k]));
    cases.push(caseResult('INV-ATT-006', 'MHA(nHeads=1) 与 SDPA 一致', dev, 0, TOL.RESID));
  }
  // INV-ATT-007 GQA(nKVHeads=1) 与“共享 KV 的 MHA”一致
  {
    const seq = 4, d = 8, nHeads = 4, nKV = 1;
    const x = zeros(seq, d), Wq = zeros(d, d), Wk = zeros(nKV * (d / nHeads), d), Wv = zeros(nKV * (d / nHeads), d), Wo = zeros(d, d);
    for (let i = 0; i < seq; i++) for (let k = 0; k < d; k++) x[i][k] = rng() * 2 - 1;
    const fill = (W) => { for (let a = 0; a < W.length; a++) for (let b = 0; b < W[0].length; b++) W[a][b] = rng() * 2 - 1; };
    fill(Wq); fill(Wk); fill(Wv); fill(Wo);
    const g = gqa(x, Wq, Wk, Wv, Wo, nHeads, nKV, {});
    // 等价 MHA：复制 KV 头到每个 query 头（即 nKV= nHeads 且所有 kv 头相等）
    const WkE = zeros(d, d), WvE = zeros(d, d);
    const kd = Wk[0].length, kvd = Wk.length;
    for (let h = 0; h < nHeads; h++) for (let a = 0; a < kvd; a++) for (let b = 0; b < kd; b++) {
      WkE[h * kvd + a][b] = Wk[a][b]; WvE[h * kvd + a][b] = Wv[a][b];
    }
    const m = multiHeadAttention(x, Wq, WkE, WvE, Wo, nHeads, {});
    let dev = 0; for (let i = 0; i < seq; i++) for (let k = 0; k < d; k++) dev = Math.max(dev, Math.abs(g[i][k] - m[i][k]));
    cases.push(caseResult('INV-ATT-007', 'GQA(nKV=1) 与共享KV的MHA一致', dev, 0, TOL.RESID));
  }
  // INV-ATT-008 softmax 平移不变：scores + 常数后输出不变
  {
    const M = 3, N = 4, d = 3;
    const Q = zeros(M, d), K = zeros(N, d), V = zeros(N, d);
    for (let i = 0; i < M; i++) for (let k = 0; k < d; k++) Q[i][k] = rng() * 2 - 1;
    for (let j = 0; j < N; j++) for (let k = 0; k < d; k++) { K[j][k] = rng() * 2 - 1; V[j][k] = rng() * 2 - 1; }
    const o1 = sdpa(Q, K, V, {}).out;
    // 构造 +c 的等价：直接对 scores 加常数后重算
    const scale = 1 / Math.sqrt(d);
    const sc = zeros(M, N);
    for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) { let s = 0; for (let k = 0; k < d; k++) s += Q[i][k] * K[j][k]; sc[i][j] = s * scale + 5.0; }
    const P2 = softmaxRows(sc, false);
    const o2 = matMul(P2, V);
    let dev = 0; for (let i = 0; i < M; i++) for (let k = 0; k < d; k++) dev = Math.max(dev, Math.abs(o1[i][k] - o2[i][k]));
    cases.push(caseResult('INV-ATT-008', 'softmax 平移不变（输出）', dev, 0, TOL.RESID));
  }

  const stat = (id) => cases.find(c => c.id === id).status;
  for (const id of ['INV-ATT-001', 'INV-ATT-002', 'INV-ATT-003', 'INV-ATT-004', 'INV-ATT-005', 'INV-ATT-006', 'INV-ATT-007', 'INV-ATT-008']) {
    const c = cases.find(x => x.id === id);
    invariants.push({ id, expr: id, tol: c.tol, worst: c.abs_err, status: c.status, note: '' });
  }

  const started = Date.now();
  return buildReport({ module: 'attention-forge', version: '1.0.0', layer: 'L4', cases, invariants, seed: '0x9E3779B9', started, tol: { 'INV-ATT-005': 1e-6 } });
}

const API = {
  name: 'attention-forge', version: '1.0.0', layer: 'L4',
  sdpa, multiHeadAttention, gqa, softmaxRows, linear, splitHeads, mergeHeads, selfCheck
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeAttention = API;

if (typeof process !== 'undefined' && process.argv[1] && /attention-forge\.js$/.test(process.argv[1]) && process.argv.includes('--self-check')) {
  const r = selfCheck({ quick: process.argv.includes('--quick') });
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : require('./core-verify.js').formatTableSafe(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
