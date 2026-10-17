'use strict';
/* ============================================================================
 * 独立复核 eval-forge —— 不采信模块 selfCheck()，不调用 selfCheck/CLI，
 * 自带 mulberry32 PRNG（与模块内 sfc32 不同族）与全套手搓黄金参照交叉验证。
 *
 * 覆盖：
 *   R1  混淆矩阵 + macro/micro PR（O00 精度 / O03 召回，INV-EVL-001）
 *   R2  F1 per-class/macro/micro（O08，INV-EVL-002）
 *   R3  ROC AUC（O09，INV-EVL-003）      R4  困惑度（O10，INV-EVL-004）
 *   R5  BLEU（O11，INV-EVL-005）         R6  延迟分位数（O01，INV-EVL-006）
 *   R7  吞吐量（O02，INV-EVL-007）       R8  一致性（O04，INV-EVL-008）
 *   R9  契约字段自检（status/passed/failed/hash 前缀 + 幂等，读导出对象不调 selfCheck）
 *
 * 本文件所有随机量由自带 mulberry32(seed) 生成，与模块内部轨道独立。
 * 退出码：0 = 全部独立复核通过；1 = 任一失败。
 * ========================================================================== */
const path = require('path');
const EV = require(path.join(__dirname, '..', '..', 'src', 'eval-forge.js'));

const TIGHT = 1e-15;
const SOLVE = 1e-10;

/* ---------- 独立 PRNG：mulberry32（与模块 sfc32 不同族） ---------- */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const results = [];
function check(id, name, ok, detail) {
  results.push({ id, name, ok: !!ok, detail });
  console.log((ok ? '  +OK ' : '  -XX ') + id + ' ' + name + '  ' + detail);
}
function throws(fn) {
  try { fn(); return false; } catch (e) { return true; }
}

/* ---------- 独立数据生成 ---------- */
function ownLabels(n, weights, seed) {
  const rng = mulberry32(seed);
  const y = [];
  for (let i = 0; i < n; i++) {
    const u = rng();
    let acc = 0, k = 0;
    for (; k < weights.length - 1; k++) { acc += weights[k]; if (u < acc) break; }
    y.push(k);
  }
  return y;
}
function ownPreds(yT, K, err, seed) {
  const rng = mulberry32(seed);
  return yT.map((t) => (rng() < err) ? Math.floor(rng() * K) : t);
}
function ownLatency(n, seed) {
  const rng = mulberry32(seed);
  const out = [];
  for (let i = 0; i < n; i++) out.push(5 + 60 * Math.pow(rng(), 4) + rng() * 3);
  return out;
}
/** 独立 logits：保证 top1 与 top2 间隔 ≥ margin。 */
function ownLogits(n, K, margin, seed) {
  const rng = mulberry32(seed);
  const M = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let k = 0; k < K; k++) row.push(rng() * 2 - 1);
    let m1 = -Infinity, m2 = -Infinity, i1 = 0;
    for (let k = 0; k < K; k++) { if (row[k] > m1) { m2 = m1; m1 = row[k]; i1 = k; } else if (row[k] > m2) m2 = row[k]; }
    if (m1 - m2 < margin) row[i1] = m2 + margin;
    M.push(row);
  }
  return M;
}
function ownArgmax(row) {
  let bi = 0, bv = row[0];
  for (let i = 1; i < row.length; i++) if (row[i] > bv) { bv = row[i]; bi = i; }
  return bi;
}
function ownPerturb(logits, eps, seed) {
  const rng = mulberry32(seed);
  return logits.map((row) => ownArgmax(row.map((x) => x + (rng() * 2 - 1) * eps)));
}

/* ---------- 独立黄金参照 ---------- */
function goldConf(yT, yP, K) {
  const cm = [];
  for (let i = 0; i < K; i++) cm.push(new Array(K).fill(0));
  for (let i = 0; i < yT.length; i++) cm[yT[i]][yP[i]]++;
  return cm;
}
function goldPRF(cm, K) {
  const P = [], R = [], F = [], sup = [];
  let totTP = 0, totFP = 0, totFN = 0;
  for (let c = 0; c < K; c++) {
    let tp = 0, fp = 0, fn = 0;
    for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
      if (i === c && j === c) tp += cm[i][j];
      else if (j === c) fp += cm[i][j];
      else if (i === c) fn += cm[i][j];
    }
    const p = (tp + fp > 0) ? tp / (tp + fp) : 0;
    const r = (tp + fn > 0) ? tp / (tp + fn) : 0;
    P.push(p); R.push(r); F.push((p + r > 0) ? (2 * p * r) / (p + r) : 0);
    sup.push(tp + fn); totTP += tp; totFP += fp; totFN += fn;
  }
  return {
    P, R, F, support: sup,
    macroP: P.reduce((a, b) => a + b, 0) / K,
    macroR: R.reduce((a, b) => a + b, 0) / K,
    macroF: F.reduce((a, b) => a + b, 0) / K,
    microP: totTP / (totTP + totFP),
    microR: totTP / (totTP + totFN),
    microF: (2 * totTP) / (2 * totTP + totFP + totFN)
  };
}
/** 独立 Mann-Whitney U：暴力 O(nP·nN)，并列计 0.5。 */
function goldAucBrute(scores, labels) {
  let U = 0, nP = 0, nN = 0;
  for (let i = 0; i < scores.length; i++) {
    if (labels[i] === 1) nP++; else nN++;
  }
  for (let i = 0; i < scores.length; i++) {
    if (labels[i] !== 1) continue;
    for (let j = 0; j < scores.length; j++) {
      if (labels[j] !== 0) continue;
      if (scores[i] > scores[j]) U += 1;
      else if (scores[i] === scores[j]) U += 0.5;
    }
  }
  return U / (nP * nN);
}
/** 独立 rank-based AUC：平均秩法（与暴力法互为交叉验证）。 */
function goldAucRank(scores, labels) {
  const n = scores.length;
  const idx = Array.from({ length: n }, (_, i) => i);
  idx.sort((a, b) => (scores[a] - scores[b]) || (a - b));
  const rank = new Array(n);
  let k = 0;
  while (k < n) {
    let j = k;
    while (j + 1 < n && scores[idx[j + 1]] === scores[idx[k]]) j++;
    const avg = (k + j + 2) / 2;
    for (let t = k; t <= j; t++) rank[idx[t]] = avg;
    k = j + 1;
  }
  let nP = 0, nN = 0, sP = 0;
  for (let i = 0; i < n; i++) { if (labels[i] === 1) { nP++; sP += rank[i]; } else nN++; }
  return (sP - nP * (nP + 1) / 2) / (nP * nN);
}
/** 独立 BLEU。 */
function goldBleu(cand, refs, N) {
  const c = cand.length;
  let r = refs[0].length;
  for (const rf of refs) {
    const dn = Math.abs(rf.length - c), dc = Math.abs(r - c);
    if (dn < dc || (dn === dc && rf.length < r)) r = rf.length;
  }
  const bp = (c > r) ? 1 : Math.exp(1 - r / c);
  let logSum = 0, zero = false;
  for (let n = 1; n <= N; n++) {
    const cg = [];
    for (let i = 0; i + n <= c; i++) cg.push(cand.slice(i, i + n).join('|'));
    if (cg.length === 0) { zero = true; continue; }
    const cc = new Map();
    for (const g of cg) cc.set(g, (cc.get(g) || 0) + 1);
    let clip = 0;
    for (const [g, cnt] of cc) {
      let best = 0;
      for (const rf of refs) {
        const rc = new Map();
        for (let i = 0; i + n <= rf.length; i++) {
          const k = rf.slice(i, i + n).join('|');
          rc.set(k, (rc.get(k) || 0) + 1);
        }
        best = Math.max(best, rc.get(g) || 0);
      }
      clip += Math.min(cnt, best);
    }
    const p = clip / cg.length;
    if (p === 0) zero = true; else logSum += Math.log(p);
  }
  return { score: zero ? 0 : Math.exp(logSum / N) * bp, bp };
}
/** 独立分位数（R type-7 凸组合形式，与模块实现写法不同）。 */
function goldQ(sorted, q) {
  const n = sorted.length;
  const h = (n - 1) * q;
  const lo = Math.floor(h), fr = h - lo;
  if (lo + 1 >= n) return sorted[n - 1];
  return sorted[lo] * (1 - fr) + sorted[lo + 1] * fr;
}
function goldMean(a) { let s = 0; for (const v of a) s += v; return s / a.length; }

console.log('──────────────────────────────────────────────');
console.log('独立复核 eval-forge（不调用 selfCheck）');

/* ================= R1 混淆矩阵 + macro/micro PR ================= */
{
  const K = 3, W = [0.82, 0.13, 0.05];
  let worst = 0, rowOk = true, minGap = Infinity;
  for (const sd of [0x1001, 0x1002, 0x1003]) {
    const yT = ownLabels(700, W, sd);
    const yP = ownPreds(yT, K, 0.22, sd + 7717);
    const cm = EV.confusion(yT, yP, K);
    const g = goldPRF(goldConf(yT, yP, K), K);
    const sup = new Array(K).fill(0);
    for (const t of yT) sup[t]++;
    for (let c = 0; c < K; c++) {
      let rs = 0;
      for (let j = 0; j < K; j++) rs += cm[c][j];
      if (rs !== sup[c]) rowOk = false;                       // EXACT
    }
    const prf = EV.classPRF(cm);
    for (let c = 0; c < K; c++) {
      worst = Math.max(worst, Math.abs(prf.precision[c] - g.P[c]), Math.abs(prf.recall[c] - g.R[c]));
    }
    const mP = EV.precisionScore(yT, yP, 'macro', { numClasses: K });
    const uP = EV.precisionScore(yT, yP, 'micro', { numClasses: K });
    const mR = EV.recallScore(yT, yP, 'macro', { numClasses: K });
    const uR = EV.recallScore(yT, yP, 'micro', { numClasses: K });
    worst = Math.max(worst, Math.abs(mP - g.macroP), Math.abs(uP - g.microP),
      Math.abs(mR - g.macroR), Math.abs(uR - g.microR));
    minGap = Math.min(minGap, Math.abs(mP - uP), Math.abs(mR - uR));
  }
  check('R1/INV-EVL-001', '混淆矩阵行和 EXACT + PR ≡ 手搓 TP/(TP+FP)、TP/(TP+FN) + macro≠micro',
    rowOk && worst <= TIGHT && minGap > 1e-9,
    '行和EXACT=' + rowOk + ' max|Δ|=' + worst.toExponential(2) + ' min|macro−micro|=' + minGap.toExponential(3));
}

/* ================= R2 F1 ================= */
{
  const K = 3;
  let worst = 0, randOk = true;
  for (const sd of [0x2001, 0x2002, 0x2003]) {
    const yT = ownLabels(700, [0.65, 0.25, 0.10], sd);
    const yP = ownPreds(yT, K, 0.28, sd + 991);
    const per = EV.f1Score(yT, yP, 'none', { numClasses: K });
    const g = goldPRF(goldConf(yT, yP, K), K);
    for (let c = 0; c < K; c++) worst = Math.max(worst, Math.abs(per[c] - g.F[c]));
    worst = Math.max(worst, Math.abs(EV.f1Score(yT, yP, 'macro', { numClasses: K }) - g.macroF));
    worst = Math.max(worst, Math.abs(EV.f1Score(yT, yP, 'micro', { numClasses: K }) - g.microF));
    const m = EV.f1Score(yT, yP, 'macro', { numClasses: K });
    if (!(m > 0 && m < 1)) randOk = false;
  }
  const yP = [0, 0, 1, 1, 2, 2, 1, 0];
  const perfect = EV.f1Score(yP, yP.slice(), 'macro', { numClasses: K }) === 1;
  const yT2 = [0, 0, 0, 1, 1, 1], yW = [1, 1, 1, 0, 0, 0];
  const wrong = EV.f1Score(yT2, yW, 'macro', { numClasses: 2 }) === 0
    && EV.f1Score(yT2, yW, 'micro', { numClasses: 2 }) === 0;
  check('R2/INV-EVL-002', 'F1 ≡ 手搓 2PR/(P+R) + 全对=1/全错=0 EXACT + 随机∈(0,1)',
    worst <= TIGHT && randOk && perfect && wrong,
    'max|Δ|=' + worst.toExponential(2) + ' 随机∈(0,1)=' + randOk + ' 全对=1=' + perfect + ' 全错=0=' + wrong);
}

/* ================= R3 ROC AUC ================= */
{
  let worst = 0, ok1 = true;
  // 3.1 暴力 Mann-Whitney（小样本、大量并列）
  for (const sd of [0x3001, 0x3002, 0x3003]) {
    const rng = mulberry32(sd);
    const n = 240, sc = [], lb = [];
    for (let i = 0; i < n; i++) { sc.push(Math.floor(rng() * 8)); lb.push(rng() < 0.5 ? 1 : 0); }
    worst = Math.max(worst, Math.abs(EV.rocAuc(sc, lb) - goldAucBrute(sc, lb)));
  }
  // 3.2 独立 rank 法（大样本、连续得分）
  for (const sd of [0x3011, 0x3012]) {
    const rng = mulberry32(sd);
    const n = 1600, sc = [], lb = [];
    for (let i = 0; i < n; i++) { sc.push(rng() * 50); lb.push(rng() < 0.5 ? 1 : 0); }
    worst = Math.max(worst, Math.abs(EV.rocAuc(sc, lb) - goldAucRank(sc, lb)));
  }
  // 3.3 完美/逆序边界（EXACT）
  const n2 = 300, asc = Array.from({ length: n2 }, (_, i) => i);
  const lab = new Array(n2).fill(0);
  for (let i = n2 / 2; i < n2; i++) lab[i] = 1;
  const perfect = EV.rocAuc(asc, lab) === 1;
  const reverse = EV.rocAuc(asc, lab.slice().reverse()) === 0;
  // 3.4 随机打乱得分 ≈ 0.5（≥10σ 余量）
  let maxDev = 0;
  for (const sd of [0x3021, 0x3022, 0x3023, 0x3024]) {
    const rng = mulberry32(sd);
    const n = 2600, sc = [], lb = [];
    let nP = 0, nN = 0;
    for (let i = 0; i < n; i++) { sc.push(rng() * 1000); const l = rng() < 0.5 ? 1 : 0; lb.push(l); if (l) nP++; else nN++; }
    const sigma = Math.sqrt((n + 1) / (12 * nP * nN));
    maxDev = Math.max(maxDev, Math.abs(EV.rocAuc(sc, lb) - 0.5) / sigma);
  }
  const randOk = maxDev <= 10;
  ok1 = worst <= TIGHT;
  // 3.5 单一类别必抛错（AUC 无定义）
  const guard = throws(() => EV.rocAuc([0.1, 0.2], [1, 1])) && throws(() => EV.rocAuc([], []));
  check('R3/INV-EVL-003', 'AUC ≡ 手搓 Mann-Whitney U（暴力 + rank 双参照）+ 完美=1/逆序=0 + 随机≤10σ',
    ok1 && perfect && reverse && randOk && guard,
    'max|Δ|=' + worst.toExponential(2) + ' 完美=1=' + perfect + ' 逆序=0=' + reverse
    + ' max|AUC−0.5|/σ=' + maxDev.toFixed(2) + ' 单类抛错=' + guard);
}

/* ================= R4 困惑度 ================= */
{
  let worst = 0, closedOk = true;
  for (const sd of [0x4001, 0x4002, 0x4003]) {
    const rng = mulberry32(sd);
    const n = 500, lp = [];
    for (let i = 0; i < n; i++) lp.push(-Math.abs(Math.log(0.05 + 0.9 * rng())) - rng() * 0.3);
    const got = EV.perplexity(lp);
    let nll = 0;
    for (const v of lp) nll += -v;
    worst = Math.max(worst, Math.abs(got - Math.exp(nll / n)) / Math.exp(nll / n));
  }
  for (const p of [0.5, 0.25, 0.05]) {
    const got = EV.perplexity(new Array(96).fill(Math.log(p)));
    const rel = Math.abs(got - 1 / p) / (1 / p);
    worst = Math.max(worst, rel);
    if (rel > SOLVE) closedOk = false;
  }
  const uni = new Array(100).fill(Math.log(0.1));
  const peak = new Array(100).fill(Math.log(0.01));
  for (let i = 0; i < 97; i++) peak[i] = Math.log(0.97);
  const pU = EV.perplexity(uni), pP = EV.perplexity(peak);
  const fromProbs = Math.abs(EV.perplexityFromProbs(new Array(50).fill(0.2)) - 5) / 5 <= SOLVE;
  const guard = throws(() => EV.perplexity([])) && throws(() => EV.perplexity([1])) && throws(() => EV.perplexityFromProbs([0]));
  check('R4/INV-EVL-004', 'perplexity ≡ exp(手搓平均 NLL) + 恒定 p 闭式 1/p + 高熵>低熵 + 非法输入抛错',
    worst <= SOLVE && closedOk && pU > pP + 1 && Math.abs(pU - 10) / 10 <= SOLVE && fromProbs && guard,
    'max相对|Δ|=' + worst.toExponential(2) + ' 均匀=' + pU.toFixed(4) + ' 尖峰=' + pP.toFixed(4) + ' 1/p闭式=' + closedOk);
}

/* ================= R5 BLEU ================= */
{
  const vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
  let worst = 0;
  for (const sd of [0x5001, 0x5002, 0x5003]) {
    const rng = mulberry32(sd);
    const mk = (len) => Array.from({ length: len }, () => vocab[Math.floor(rng() * vocab.length)]);
    for (let t = 0; t < 8; t++) {
      const cand = mk(5 + Math.floor(rng() * 9));
      const refs = [mk(5 + Math.floor(rng() * 9)), mk(5 + Math.floor(rng() * 9))];
      const got = EV.bleu(cand, refs, 4);
      const g = goldBleu(cand, refs, 4);
      worst = Math.max(worst, Math.abs(got.score - g.score), Math.abs(got.bp - g.bp));
    }
  }
  const ref = 'the quick brown fox jumps over the lazy dog';
  const selfEq = EV.bleu(ref, [ref], 4).score === 1;
  const shortB = EV.bleu(['a', 'b'], [['a', 'b', 'c', 'd', 'e']], 2);
  const shortOk = shortB.score > 0 && shortB.score < 1 && shortB.bp < 1;
  const zeroOk = EV.bleu(['x', 'y', 'z', 'w'], [['a', 'b', 'c', 'd', 'e', 'f']], 4).score === 0;
  const guard = throws(() => EV.bleu([], [['a']])) && throws(() => EV.bleu(['a'], []));
  check('R5/INV-EVL-005', 'BLEU ≡ 手搓 n-gram 精度×BP + cand==ref=1 + 短候选<1 + 零匹配=0',
    worst <= SOLVE && selfEq && shortOk && zeroOk && guard,
    'max|Δ|=' + worst.toExponential(2) + ' 自等=1=' + selfEq + ' 短候选=' + shortB.score.toFixed(4)
    + '(BP=' + shortB.bp.toFixed(4) + ') 零匹配=0=' + zeroOk);
}

/* ================= R6 延迟分位数 ================= */
{
  let worst = 0, monoOk = true, p95s = [];
  const rel = (a, b) => Math.abs(a - b) / Math.max(1, Math.abs(b));
  for (const sd of [0x6001, 0x6002, 0x6003, 0x6004]) {
    const s = ownLatency(900, sd);
    const st = EV.latencyStats(s);
    const sorted = s.slice().sort((a, b) => a - b);
    worst = Math.max(worst, rel(st.p50, goldQ(sorted, 0.50)), rel(st.p95, goldQ(sorted, 0.95)),
      rel(st.p99, goldQ(sorted, 0.99)), rel(st.mean, goldMean(s)));
    if (!(st.p50 <= st.p95 && st.p95 <= st.p99)) monoOk = false;
    // 分位数必须落在最近秩 bracket 内（结构一致性的第二条独立证据）
    const lo = sorted[Math.floor(0.95 * (sorted.length - 1))], hi = sorted[Math.ceil(0.95 * (sorted.length - 1))];
    if (!(st.p95 >= lo - 1e-12 && st.p95 <= hi + 1e-12)) monoOk = false;
    p95s.push(st.p95);
  }
  let minSpread = Infinity;
  for (let i = 0; i < p95s.length; i++) for (let j = i + 1; j < p95s.length; j++) {
    minSpread = Math.min(minSpread, Math.abs(p95s[i] - p95s[j]));
  }
  const guard = throws(() => EV.latencyStats([])) && throws(() => EV.latencyStats([1, Infinity]));
  check('R6/INV-EVL-006', 'p50/p95/p99 ≡ 手搓排序插值 + 单调 EXACT + bracket 一致 + 异样本集分歧',
    worst <= TIGHT && monoOk && minSpread > 1e-9 && guard,
    'max相对|Δ|=' + worst.toExponential(2) + ' 单调&bracket=' + monoOk
    + ' min|Δp95|=' + minSpread.toExponential(3) + ' 空/Inf 抛错=' + guard);
}

/* ================= R7 吞吐量 ================= */
{
  let worst = 0, ratioOk = true;
  for (const [c, s] of [[1000, 4], [2048, 2.5], [999, 0.5], [86400, 3600]]) {
    const got = EV.throughput(c, s);
    worst = Math.max(worst, Math.abs(got - c / s));
    if (EV.throughput(2 * c, s) / got !== 2) ratioOk = false;
    if (EV.throughput(4 * c, s) / got !== 4) ratioOk = false;
  }
  const guard = throws(() => EV.throughput(10, 0)) && throws(() => EV.throughput(10, -2))
    && throws(() => EV.throughput(10, NaN)) && throws(() => EV.throughput(Infinity, 1));
  check('R7/INV-EVL-007', 'throughput ≡ count/seconds + 加倍/四倍比例 EXACT + 除零/负/非法抛错',
    worst <= TIGHT && ratioOk && guard,
    'max|Δ|=' + worst.toExponential(2) + ' 比例 EXACT=' + ratioOk + ' 0/负/NaN/Inf 抛错=' + guard);
}

/* ================= R8 一致性 ================= */
{
  const K = 4, N = 1500, MARGIN = 1e-3;
  const logits = ownLogits(N, K, MARGIN, 0x8001);
  const A = logits.map(ownArgmax);
  const A2 = logits.map(ownArgmax);
  const same = EV.consistencyRate(A, A2) === 1;                      // EXACT
  const tiny = EV.consistencyRate(A, ownPerturb(logits, 1e-6, 0x8101));
  const big = EV.consistencyRate(A, ownPerturb(logits, 0.5, 0x8201));
  const bigOk = big < 0.95 && big > 0;
  // 模块确定性：同输入重复调用，输出逐位相等（不依赖种子，纯函数性）
  const yT = ownLabels(400, [0.5, 0.3, 0.2], 0x8301);
  const yP = ownPreds(yT, 3, 0.2, 0x8302);
  const lat = ownLatency(200, 0x8303);
  const detOk = JSON.stringify(EV.confusion(yT, yP, 3)) === JSON.stringify(EV.confusion(yT, yP, 3))
    && EV.bleu(['a', 'b', 'c', 'd'], [['a', 'b', 'c', 'd', 'e']], 4).score
    === EV.bleu(['a', 'b', 'c', 'd'], [['a', 'b', 'c', 'd', 'e']], 4).score
    && EV.latencyStats(lat).p99 === EV.latencyStats(lat).p99
    && EV.perplexity([Math.log(0.5), Math.log(0.25)]) === EV.perplexity([Math.log(0.5), Math.log(0.25)]);
  const guard = throws(() => EV.consistencyRate([1, 2], [1])) && throws(() => EV.consistencyRate([], []));
  check('R8/INV-EVL-008', '同输入一致率=1(EXACT) + 微扰=1 + 大扰<0.95 + 纯函数重复调用逐位相等',
    same && tiny === 1 && bigOk && detOk && guard,
    '同输入=' + (same ? 1 : 0) + ' 微扰1e-6=' + tiny + ' 大扰0.5=' + big.toFixed(4)
    + ' 纯函数确定性=' + detOk + ' 长度/空抛错=' + guard);
}

/* ================= R9 契约字段（只读导出对象，不调 selfCheck） ================= */
{
  const need = ['confusion', 'classPRF', 'precisionScore', 'recallScore', 'f1Score', 'rocAuc',
    'perplexity', 'bleu', 'latencyStats', 'throughput', 'consistencyRate', 'selfCheck'];
  const missing = need.filter((k) => typeof EV[k] !== 'function');
  const tolOk = EV.TOL && EV.TOL.EXACT === 0 && EV.TOL.TIGHT === 1e-15 && EV.TOL.SOLVE === 1e-10
    && EV.TOL.MONO === -1e-9 && EV.TOL.CONV === 1e-6 && EV.TOL.GRAD === 1e-6
    && EV.TOL.ULP === 1e-16 && EV.TOL.ORTHO === 1e-14 && EV.TOL.RESID === 1e-12;
  const metaOk = EV.name === 'eval-forge' && EV.version === '1.0.0' && EV.layer === 'L7';
  const hashOk = typeof EV.sourceHash === 'function' && /^sha256:[0-9a-f]{64}$/.test(EV.sourceHash());
  const hashIdem = EV.sourceHash() === EV.sourceHash();
  check('R9/契约', '12 项导出齐全 + 九档容差 + 元信息 + sha256 源码 hash 幂等',
    missing.length === 0 && tolOk && metaOk && hashOk && hashIdem,
    '缺失导出=' + (missing.length ? missing.join(',') : '无') + ' 九档TOL=' + tolOk
    + ' 元信息=' + metaOk + ' hash格式=' + hashOk + ' 幂等=' + hashIdem);
}

/* ---------- 汇总 ---------- */
const failed = results.filter((r) => !r.ok).length;
console.log('──────────────────────────────────────────────');
console.log('独立复核 eval-forge：' + (results.length - failed) + '/' + results.length + ' 通过');
if (failed > 0) process.exit(1);
process.exit(0);
