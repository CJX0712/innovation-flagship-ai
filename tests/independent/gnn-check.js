'use strict';
/* ============================================================================
 * 独立复核 gnn-forge —— 不采信模块 selfCheck 的数学结果，手搓黄金参照独立复算。
 *
 * 覆盖关键不变量（全部使用本文件自写的参照实现与自写 LCG 随机源，与模块内部
 * sfc32/core-verify 完全独立）：
 *   INV-GNN-001  对称归一化 vs 手搓闭式；行和 ≠ 1 证伪
 *   INV-GNN-002  GCN 前向 vs 手搓循环展开；W=0 退化
 *   INV-GNN-003  GraphSAGE mean vs 手搓；三聚合器分歧
 *   INV-GNN-004  GAT α vs 手搓参照；行和=1；非均匀；LeakyReLU 斜率
 *   INV-GNN-005  MPNN ≡ GCN(无 W)；mean ≡ 手搓行均值；聚合器分歧
 *   INV-GNN-006  节点分类 300 步过拟合 acc=1.0；未训练 <1.0
 *   INV-GNN-007  采样种子复现 / 异种子分歧 / 出度约束
 *   INV-GNN-008  全流水线确定性（模块级双跑）+ selfCheck 报告契约（元校验）
 * ========================================================================== */
const path = require('path');
const GNN = require(path.join(__dirname, '..', '..', 'src', 'gnn-forge.js'));

/* ---------- 自写随机源：LCG（与模块 sfc32 无关） ---------- */
function lcg(seed) {
  let z = seed >>> 0;
  return () => { z = (Math.imul(z, 1664525) + 1013904223) >>> 0; return z / 4294967296; };
}

/* ---------- 自写数值小工具 ---------- */
function zeros(n, d) { const M = []; for (let i = 0; i < n; i++) M.push(new Array(d).fill(0)); return M; }
function randMat(rng, r, c, s) { const M = zeros(r, c); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) M[i][j] = (rng() * 2 - 1) * s; return M; }
function maxDiff(A, B) { let d = 0; for (let i = 0; i < A.length; i++) for (let j = 0; j < A[i].length; j++) d = Math.max(d, Math.abs(A[i][j] - B[i][j])); return d; }
function maxDiff3(A, B) { let d = 0; for (let t = 0; t < A.length; t++) d = Math.max(d, maxDiff(A[t], B[t])); return d; }
function handLeaky(x, s) { return x < 0 ? s * x : x; }
function handSoftmax(a) { const m = Math.max.apply(null, a); const ex = a.map(v => Math.exp(v - m)); const Z = ex.reduce((p, q) => p + q, 0); return ex.map(v => v / Z); }

/* ---------- 手搓黄金参照（独立公式转录） ---------- */
function handSymNorm(adj) {
  const n = adj.length, deg = [];
  for (let i = 0; i < n; i++) { let d = 1; for (let j = 0; j < n; j++) d += adj[i][j]; deg.push(d); }
  const An = zeros(n, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const a = i === j ? adj[i][j] + 1 : adj[i][j];
    An[i][j] = a === 0 ? 0 : a / Math.sqrt(deg[i] * deg[j]);
  }
  return An;
}
function handGCN(X, adj, W, b) {
  const An = handSymNorm(adj);
  const n = X.length, dIn = X[0].length, dOut = W[0].length;
  const AX = zeros(n, dIn);
  for (let i = 0; i < n; i++) for (let l = 0; l < dIn; l++) for (let j = 0; j < n; j++) AX[i][l] += An[i][j] * X[j][l];
  const out = zeros(n, dOut);
  for (let i = 0; i < n; i++) for (let k = 0; k < dOut; k++) {
    let s = 0;
    for (let l = 0; l < dIn; l++) s += AX[i][l] * W[l][k];
    const v = s + b[k];
    out[i][k] = v > 0 ? v : 0;
  }
  return out;
}
function handSageMean(X, adj, W) {
  const n = X.length, dIn = X[0].length, dOut = W[0].length;
  const out = zeros(n, dOut);
  for (let i = 0; i < n; i++) {
    const nbrs = [];
    for (let j = 0; j < n; j++) if (j !== i && adj[i][j] !== 0) nbrs.push(j);
    const agg = new Array(dIn).fill(0);
    for (const j of nbrs) for (let k = 0; k < dIn; k++) agg[k] += X[j][k];
    if (nbrs.length) for (let k = 0; k < dIn; k++) agg[k] /= nbrs.length;
    for (let k = 0; k < dOut; k++) {
      let s = 0;
      for (let q = 0; q < dIn; q++) s += X[i][q] * W[q][k] + agg[q] * W[dIn + q][k];
      out[i][k] = s;
    }
  }
  return out;
}
function handGATAlpha(X, adj, W, a, nHeads) {
  const n = X.length, dIn = X[0].length, dHead = W[0].length / nHeads;
  const out = [];
  for (let t = 0; t < nHeads; t++) {
    const Wh = zeros(n, dHead);
    for (let i = 0; i < n; i++) for (let k = 0; k < dHead; k++) for (let l = 0; l < dIn; l++) Wh[i][k] += X[i][l] * W[l][t * dHead + k];
    const alphaT = zeros(n, n);
    for (let i = 0; i < n; i++) {
      const nbrs = [i];
      for (let j = 0; j < n; j++) if (j !== i && adj[i][j] !== 0) nbrs.push(j);
      const scores = nbrs.map(j => {
        let s = 0;
        for (let k = 0; k < dHead; k++) s += a[t][k] * Wh[i][k] + a[t][dHead + k] * Wh[j][k];
        return handLeaky(s, 0.2);
      });
      const p = handSoftmax(scores);
      for (let idx = 0; idx < nbrs.length; idx++) alphaT[i][nbrs[idx]] = p[idx];
    }
    out.push(alphaT);
  }
  return out;
}
function handRowMean(adj, X) {
  const n = adj.length, d = X[0].length, out = [];
  for (let i = 0; i < n; i++) {
    let cnt = 1;
    for (let j = 0; j < n; j++) if (j !== i && adj[i][j] !== 0) cnt++;
    const row = [];
    for (let k = 0; k < d; k++) {
      let s = X[i][k];
      for (let j = 0; j < n; j++) if (j !== i && adj[i][j] !== 0) s += X[j][k];
      row.push(s / cnt);
    }
    out.push(row);
  }
  return out;
}

/* ---------- 测试图 ---------- */
const G1 = [
  [0, 1, 1, 0, 0, 0, 0],
  [1, 0, 1, 0, 0, 0, 0],
  [1, 1, 0, 1, 0, 0, 0],
  [0, 0, 1, 0, 1, 1, 0],
  [0, 0, 0, 1, 0, 1, 1],
  [0, 0, 0, 1, 1, 0, 1],
  [0, 0, 0, 0, 1, 1, 0]
];
const G1I = G1.map((r, i) => r.map((v, j) => (i === j ? v + 1 : v)));
const Gc = [
  [0, 1, 1, 0, 0, 0],
  [1, 0, 1, 0, 0, 0],
  [1, 1, 0, 1, 0, 0],
  [0, 0, 1, 0, 1, 1],
  [0, 0, 0, 1, 0, 1],
  [0, 0, 0, 1, 1, 0]
];

const results = [];
function report(id, name, ok, detail) {
  results.push(ok);
  console.log('[' + id + '] ' + name + ' ' + detail + (ok ? ' ✅' : ' ❌'));
}

/* ---------- INV-GNN-001 对称归一化 ---------- */
{
  const rng = lcg(0xBEEF01);
  const n = 8, GW = zeros(n, n);
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (rng() < 0.45) { const w = 0.5 + rng(); GW[i][j] = w; GW[j][i] = w; }
  const dev = maxDiff(GNN.symNormalize(GW), handSymNorm(GW));
  const An = GNN.symNormalize(GW);
  let sym = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) sym = Math.max(sym, Math.abs(An[i][j] - An[j][i]));
  const star = zeros(9, 9);
  for (let j = 1; j < 9; j++) { star[0][j] = 1; star[j][0] = 1; }
  const As = GNN.symNormalize(star);
  let rsDev = 0;
  for (let i = 0; i < 9; i++) { let s = 0; for (let j = 0; j < 9; j++) s += As[i][j]; rsDev = Math.max(rsDev, Math.abs(s - 1)); }
  report('INV-GNN-001', '对称归一化 vs 手搓闭式',
    dev < 1e-12 && sym < 1e-12 && rsDev > 0.05,
    'max|·|=' + dev.toExponential(2) + ' 对称=' + sym.toExponential(2) + ' 星形行和偏离=' + rsDev.toFixed(4));
}

/* ---------- INV-GNN-002 GCN 前向 ---------- */
{
  const rng = lcg(0xBEEF02);
  const X = randMat(rng, 7, 4, 1), W = randMat(rng, 4, 3, 1), b = randMat(rng, 1, 3, 0.5)[0];
  const dev = maxDiff(GNN.gcnLayer(X, G1, W, b, 'relu'), handGCN(X, G1, W, b));
  const W0 = zeros(4, 3);
  const H0 = GNN.gcnLayer(X, G1, W0, b, null);
  let dev0 = 0;
  for (let i = 0; i < 7; i++) for (let k = 0; k < 3; k++) dev0 = Math.max(dev0, Math.abs(H0[i][k] - b[k]));
  report('INV-GNN-002', 'GCN 前向 vs 手搓循环展开',
    dev < 1e-12 && dev0 === 0,
    'max|·|=' + dev.toExponential(2) + ' W=0 退化偏差=' + dev0.toExponential(2));
}

/* ---------- INV-GNN-003 GraphSAGE ---------- */
{
  const rng = lcg(0xBEEF03);
  const X = randMat(rng, 7, 4, 1), W = randMat(rng, 8, 2, 1);
  const dev = maxDiff(GNN.sageAggregate(X, G1, W, 'mean'), handSageMean(X, G1, W));
  const Hm = GNN.sageAggregate(X, G1, W, 'mean');
  const Hs = GNN.sageAggregate(X, G1, W, 'sum');
  const Hx = GNN.sageAggregate(X, G1, W, 'max');
  const minDiv = Math.min(maxDiff(Hm, Hs), maxDiff(Hm, Hx), maxDiff(Hs, Hx));
  report('INV-GNN-003', 'GraphSAGE mean vs 手搓 + 三聚合器分歧',
    dev < 1e-12 && minDiv > 1e-6,
    'max|·|=' + dev.toExponential(2) + ' 最小分歧=' + minDiv.toExponential(2));
}

/* ---------- INV-GNN-004 GAT ---------- */
{
  const rng = lcg(0xBEEF04);
  const X = randMat(rng, 7, 4, 1), W = randMat(rng, 4, 6, 1);
  const a = [randMat(rng, 1, 6, 1)[0], randMat(rng, 1, 6, 1)[0]];
  const alpha = GNN.gatAttention(X, G1, W, a, 2);
  const dev = maxDiff3(alpha, handGATAlpha(X, G1, W, a, 2));
  let rsDev = 0;
  for (let t = 0; t < 2; t++) for (let i = 0; i < 7; i++) { let s = 0; for (let j = 0; j < 7; j++) s += alpha[t][i][j]; rsDev = Math.max(rsDev, Math.abs(s - 1)); }
  let uniDev = 0;
  for (let t = 0; t < 2; t++) for (let i = 0; i < 7; i++) {
    let m = 1;
    for (let j = 0; j < 7; j++) if (j !== i && G1[i][j] !== 0) m++;
    for (let j = 0; j < 7; j++) if (alpha[t][i][j] > 0) uniDev = Math.max(uniDev, Math.abs(alpha[t][i][j] - 1 / m));
  }
  const ln = GNN.leakyRelu(-1.5, 0.2);
  const leakOk = Math.abs(ln - 0.2 * -1.5) < 1e-15 && ln < 0 && GNN.leakyRelu(2.5, 0.2) === 2.5;
  report('INV-GNN-004', 'GAT α vs 手搓参照 + 行和=1 + 非均匀 + 斜率',
    dev < 1e-12 && rsDev < 1e-12 && uniDev > 1e-3 && leakOk,
    'max|α−手搓|=' + dev.toExponential(2) + ' 行和偏差=' + rsDev.toExponential(2) +
    ' 均匀偏离=' + uniDev.toExponential(2) + ' 斜率探针=' + leakOk);
}

/* ---------- INV-GNN-005 MPNN ≡ GCN(无 W) ---------- */
{
  const rng = lcg(0xBEEF05);
  const X = randMat(rng, 7, 4, 1);
  const dh = [];
  for (let i = 0; i < 7; i++) { let d = 1; for (let j = 0; j < 7; j++) d += G1[i][j]; dh.push(d); }
  const mpGcn = GNN.messagePass(X, G1I,
    (hi, hj, i, j) => hj.map(v => v / Math.sqrt(dh[i] * dh[j])),
    GNN.sumMsgs, (h, m) => m);
  // 手搓 Âsym·X（独立循环）
  const An = handSymNorm(G1), AX = zeros(7, 4);
  for (let i = 0; i < 7; i++) for (let l = 0; l < 4; l++) for (let j = 0; j < 7; j++) AX[i][l] += An[i][j] * X[j][l];
  const devG = maxDiff(mpGcn, AX);
  const devM = maxDiff(GNN.messagePass(X, G1I), handRowMean(G1, X));
  const div = maxDiff(GNN.messagePass(X, G1I), GNN.messagePass(X, G1I, null, GNN.sumMsgs, null));
  report('INV-GNN-005', 'MPNN(度缩放+sum) ≡ GCN(无W) + mean ≡ 手搓行均值 + 分歧',
    devG < 1e-12 && devM < 1e-12 && div > 1e-9,
    'MPNN-GCN max|·|=' + devG.toExponential(2) + ' mean-手搓=' + devM.toExponential(2) +
    ' mean-sum 分歧=' + div.toExponential(2));
}

/* ---------- INV-GNN-006 节点分类过拟合 ---------- */
{
  const rng = lcg(0xBEEF06);
  const n = 6, X = zeros(n, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) X[i][j] = (i === j ? 1 : 0) + (rng() - 0.5) * 0.1;
  const W1 = randMat(lcg(0xCAFE1), 6, 8, 0.7), W2 = randMat(lcg(0xCAFE2), 8, 2, 0.7);
  const labels = [0, 0, 0, 1, 1, 1];
  const base = GNN.nodeClassify(X, Gc, W1, W2, { labels, steps: 0 });
  const trained = GNN.nodeClassify(X, Gc, W1, W2, { labels, lr: 0.5, steps: 300 });
  const ok = trained.accuracy === 1 && base.accuracy < 1 && trained.loss < base.loss;
  report('INV-GNN-006', '节点分类 300 步 acc=1.0 + 未训练<1.0',
    ok, '未训练 acc=' + base.accuracy.toFixed(3) + ' 训练 acc=' + trained.accuracy.toFixed(3) +
    ' loss ' + base.loss.toFixed(4) + '→' + trained.loss.toExponential(2));
}

/* ---------- INV-GNN-007 采样 ---------- */
{
  const G2 = zeros(12, 12);
  for (let j = 1; j < 12; j++) { G2[0][j] = 1; G2[j][0] = 1; }
  for (let j = 2; j <= 6; j++) { G2[1][j] = 1; G2[j][1] = 1; }
  let sameDev = 0;
  for (const sd of [0x77, 0x123, 0x9999]) {
    sameDev = Math.max(sameDev, maxDiff(GNN.sampleNeighbors(G2, 3, GNN.makeRng(sd)), GNN.sampleNeighbors(G2, 3, GNN.makeRng(sd))));
  }
  let divCount = 0, constraintOk = true;
  for (let p = 0; p < 20; p++) {
    const r1 = GNN.sampleNeighbors(G2, 3, GNN.makeRng(0xC000 + p));
    const r2 = GNN.sampleNeighbors(G2, 3, GNN.makeRng(0xD000 + p));
    if (maxDiff(r1, r2) > 0) divCount++;
    for (let i = 0; i < 12; i++) {
      let cnt = 0, deg = 0;
      for (let j = 0; j < 12; j++) {
        if (G2[i][j] !== 0 && j !== i) deg++;
        if (r1[i][j] !== 0) { cnt++; if (G2[i][j] === 0 || j === i) constraintOk = false; }
      }
      if (cnt > Math.min(3, deg)) constraintOk = false;
    }
  }
  report('INV-GNN-007', '采样同 seed 逐位相等 + 异 seed 分歧 + 约束',
    sameDev === 0 && divCount >= 18 && constraintOk,
    '同 seed max diff=' + sameDev + ' 分歧对=' + divCount + '/20 约束=' + constraintOk);
}

/* ---------- INV-GNN-008 确定性（模块级流水线） ---------- */
{
  const rngA = GNN.makeRng(0x9E3779B9), rngB = GNN.makeRng(0x9E3779B9), rngC = GNN.makeRng(0xFEEDF00D);
  // 权重在流水线外固定（与随机源无关），输出差异只随节点随机源 seed 变化
  const Wg = randMat(lcg(0xE4), 4, 3, 1), bg = randMat(lcg(0xE5), 1, 3, 0.5)[0];
  const Wga = randMat(lcg(0xE1), 4, 6, 1), aga = [randMat(lcg(0xE2), 1, 6, 1)[0], randMat(lcg(0xE3), 1, 6, 1)[0]];
  const pipe = (rng) => {
    const X = randMat(rng, 7, 4, 1);
    const sampled = GNN.sampleNeighbors(G1, 3, rng);
    const h1 = GNN.gcnLayer(X, G1, Wg, bg, 'relu');
    const h2 = GNN.gatLayer(X, G1, Wga, aga, 2);
    const mp = GNN.messagePass(X, G1I);
    return [].concat(sampled.flat(), h1.flat(), h2.flat(), mp.flat());
  };
  const pA1 = pipe(rngA), pA2 = pipe(rngB), pB = pipe(rngC);
  let sameDiff = 0, crossDiff = 0;
  for (let i = 0; i < pA1.length; i++) {
    sameDiff = Math.max(sameDiff, Math.abs(pA1[i] - pA2[i]));
    crossDiff = Math.max(crossDiff, Math.abs(pA1[i] - pB[i]));
  }
  report('INV-GNN-008', '全流水线同 seed 逐位确定 + 异 seed 分歧',
    sameDiff === 0 && crossDiff > 1e-6,
    '同 seed max diff=' + sameDiff.toExponential(2) + ' 异 seed max diff=' + crossDiff.toExponential(2) +
    ' (' + pA1.length + ' 维)');
}

/* ---------- 元校验：selfCheck 报告契约（不作为数学证据，仅校验报告形状） ---------- */
{
  const r1 = GNN.selfCheck(), r2 = GNN.selfCheck();
  const ok =
    r1.schema === 'forge-selfcheck/1.0' && r1.module === 'gnn-forge' && r1.layer === 'L7' &&
    r1.passed === r1.cases.filter(c => c.status === 'pass').length &&
    r1.passed + r1.failed === r1.cases.length &&
    r1.skipped === 0 &&
    r1.cases.every(c => c.status === 'pass' && typeof c.passed === 'boolean') &&
    typeof r1.hash === 'string' && r1.hash.startsWith('sha256:') &&
    r1.hash === r2.hash;
  report('META', 'selfCheck 报告契约（schema/passed/failed/hash 幂等/W13 status 字段）',
    ok, 'passed=' + r1.passed + '/' + r1.cases.length + ' hash=' + r1.hash.slice(0, 24) + '…');
}

const failed = results.filter(v => !v).length;
console.log('══════ 独立复核 gnn-forge：' + (results.length - failed) + '/' + results.length + ' 通过 ══════');
process.exit(failed === 0 ? 0 : 1);
