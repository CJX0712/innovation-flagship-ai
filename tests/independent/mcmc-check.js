'use strict';
/* ============================================================================
 * 独立复核 mcmc-forge —— 不采信模块 selfCheck()，自带 mulberry32 PRNG、
 * Simpson 数值积分正态 CDF、独立 ESS（Geyer 初始正序列）、独立 leapfrog/Euler/
 * MH 实现（world-class 独立审计口径）。
 *
 * 覆盖：
 *   R1  MH 接受修正的独立实现三路对照：对称提议≡目标矩；非对称+修正≡目标矩；
 *       非对称−修正≡π·q 分布且必偏离目标（INV-MC-001）
 *   R2  离散两状态：独立解析核 + 细致平衡精确成立 + 恒接受必违反 + 模块链
 *       经验频率/转移矩阵 10σ（INV-MC-002）
 *   R3  RWM 大样本矩收敛（独立 ESS + 独立 Simpson-CDF KS）+ 最优窗宽 0.44 +
 *       调优收敛 + 恒接受随机游走证伪（INV-MC-003）
 *   R4  Gibbs 条件矩（独立回归公式 slope=corr·sx/sy）+ 错误条件方差证伪（INV-MC-004）
 *   R5  leapfrog 三性质：独立实现对拍 + 可逆 + 保体积 + 能量有界 vs Euler 漂移（INV-MC-005）
 *   R6  NUTS/混合MC 效率（独立 ESS 对比 + U-turn 判据单测 + 轨迹长卡方 + 超大 eps 证伪）（INV-MC-006）
 *   R7  dual averaging：用独立 HMC 步测量模块自适应出的步长，平均接受概率≈目标（INV-MC-007）
 *   R8  退火：独立调度公式对拍 + 确定性 + 8 种子全局模统计 + 冷链困陷 + 降温致锐（INV-MC-008）
 *
 * 本文件不调用 mcmc-forge.selfCheck / makeRng；所有随机量由本文件 mulberry32
 * 生成，与模块内部 sfc32 轨道独立；黄金参照均为本文件独立实现。
 * 退出码：0 = 全部独立复核通过；1 = 任一失败。
 * ========================================================================== */
const path = require('path');
const G = require(path.join(__dirname, '..', '..', 'src', 'mcmc-forge.js'));

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

/* ---------- 独立正态向量（Box-Muller，与模块成对消费约定无关） ---------- */
function ownGaussVec(rng, dim) {
  const out = [];
  for (let i = 0; i < dim; i += 2) {
    const u1 = Math.max(rng(), 1e-300), u2 = rng();
    const r = Math.sqrt(-2 * Math.log(u1));
    out.push(r * Math.cos(2 * Math.PI * u2));
    if (i + 1 < dim) out.push(r * Math.sin(2 * Math.PI * u2));
  }
  return out;
}

/* ---------- 独立正态 CDF：梯形则数值积分 pdf，单遍扫描 + 线性插值（与模块 erf 近似完全独立） ---------- */
function ownNormalCdfTable(samples, mu, sd) {
  const s = samples.slice().sort((a, b) => a - b);
  const lo = mu - 14 * sd, hi = mu + 14 * sd;
  const N = 1 << 18;                       // 262144 区间，梯形误差 ~1e-8
  const h = (hi - lo) / N;
  const pdf = (x) => Math.exp(-0.5 * ((x - mu) / sd) * ((x - mu) / sd)) / (sd * Math.sqrt(2 * Math.PI));
  const xs = new Float64Array(N + 1);
  const cum = new Float64Array(N + 1);
  let acc = 0;
  for (let i = 0; i <= N; i++) {
    xs[i] = lo + i * h;
    if (i > 0) acc += 0.5 * h * (pdf(xs[i - 1]) + pdf(xs[i]));
    cum[i] = acc;
  }
  const out = new Float64Array(s.length);
  let j = 0;
  for (let i = 0; i < N && j < s.length; i++) {
    while (j < s.length && s[j] <= xs[i + 1]) {
      const t = (s[j] - xs[i]) / h;         // [xs[i], xs[i+1]] 线性插值
      out[j] = cum[i] + t * (cum[i + 1] - cum[i]);
      j++;
    }
  }
  for (let k = 0; k < s.length; k++) {
    if (s[k] <= lo) out[k] = 0;
    if (s[k] >= hi) out[k] = 1;
  }
  return { sorted: s, cdf: out };
}

/** 独立双侧 KS 统计量 vs N(mu, sd²) */
function ownKsNormal(samples, mu, sd) {
  const { sorted, cdf } = ownNormalCdfTable(samples, mu, sd);
  const n = sorted.length;
  let d = 0;
  for (let i = 0; i < n; i++) {
    const F = cdf[i];
    const a = Math.abs(F - i / n);
    const b = Math.abs(F - (i + 1) / n);
    if (a > d) d = a;
    if (b > d) d = b;
  }
  return d;
}

/* ---------- 独立 ESS：Geyer 初始正序列（独立编码） ---------- */
function ownEss(xs) {
  const n = xs.length;
  if (n < 10) return 0;
  let m = 0;
  for (const x of xs) m += x;
  m /= n;
  let v = 0;
  for (const x of xs) v += (x - m) * (x - m);
  const rho = (lag) => {
    let num = 0;
    for (let i = 0; i + lag < n; i++) num += (xs[i] - m) * (xs[i + lag] - m);
    return num / v;
  };
  const maxLag = Math.min(n - 2, 1500);
  let tau = 1, k = 1;
  while (k + 1 <= maxLag) {
    const pair = rho(k) + rho(k + 1);
    if (pair <= 0) break;
    tau += 2 * pair;
    k += 2;
  }
  return n / Math.max(tau, 1e-9);
}

/* ---------- 独立统计量 ---------- */
function ownMean(xs) { let s = 0; for (const x of xs) s += x; return s / xs.length; }
function ownVar(xs) { const m = ownMean(xs); let s = 0; for (const x of xs) s += (x - m) * (x - m); return s / (xs.length - 1); }

/* ---------- 独立 leapfrog / Euler / HMC 步（与模块不同编码） ---------- */
function ownLeapfrog(grad, x, p, eps, L) {
  const dim = x.length;
  const xx = x.slice(), pp = p.slice();
  let g = grad(xx);
  for (let k = 0; k < dim; k++) pp[k] += 0.5 * eps * g[k];
  for (let l = 0; l < L; l++) {
    for (let k = 0; k < dim; k++) xx[k] += eps * pp[k];
    g = grad(xx);
    const c = l === L - 1 ? 0.5 : 1;
    for (let k = 0; k < dim; k++) pp[k] += c * eps * g[k];
  }
  return { x: xx, p: pp };
}
function ownEuler(grad, x, p, eps, L) {
  const dim = x.length;
  const xx = x.slice(), pp = p.slice();
  for (let l = 0; l < L; l++) {
    const g = grad(xx);
    const xn = [], pn = [];
    for (let k = 0; k < dim; k++) { xn.push(xx[k] + eps * pp[k]); pn.push(pp[k] + eps * g[k]); }
    for (let k = 0; k < dim; k++) { xx[k] = xn[k]; pp[k] = pn[k]; }
  }
  return { x: xx, p: pp };
}
function ownHmcStep(logp, grad, x, eps, L, rng) {
  const p = ownGaussVec(rng, x.length);
  const H0 = -logp(x) + 0.5 * p.reduce((s, v) => s + v * v, 0);
  const lf = ownLeapfrog(grad, x, p, eps, L);
  const H1 = -logp(lf.x) + 0.5 * lf.p.reduce((s, v) => s + v * v, 0);
  const ap = Math.min(1, Math.exp(H0 - H1));
  const u = rng();
  return { x: u < ap ? lf.x : x.slice(), acceptProb: ap };
}

/* ---------- 独立 MH（含/不含修正）—— R1 的独立实现 ---------- */
function ownMhChain(logp, dim, x0, steps, stepSize, rng, correct) {
  let x = x0.slice(), lp = logp(x), acc = 0;
  const chain = [];
  for (let i = 0; i < steps; i++) {
    const z = ownGaussVec(rng, dim);
    const y = x.map((v, k) => v + stepSize * z[k]);
    const lpy = logp(y);
    let logAlpha = lpy - lp;
    if (correct) logAlpha += 0;                 // 对称提议无需修正
    const u = rng();
    if (Math.log(u) < logAlpha) { x = y; lp = lpy; acc++; }
    chain.push(x.slice());
  }
  return { chain, acc };
}
/** 独立非对称提议 MH：独立提议 N(mu, s)，可开关 Hastings 修正项 */
function ownMhIndepChain(logp, dim, x0, steps, mu, s, rng, correct) {
  const logN = (x, m, sd) => {
    let t = 0;
    for (let k = 0; k < dim; k++) { const z = (x[k] - m[k]) / sd; t += z * z; }
    return -0.5 * dim * Math.log(2 * Math.PI) - dim * Math.log(sd) - 0.5 * t;
  };
  let x = x0.slice(), lp = logp(x), acc = 0;
  const chain = [];
  for (let i = 0; i < steps; i++) {
    const z = ownGaussVec(rng, dim);
    const y = z.map((v, k) => mu[k] + s * v);
    const lpy = logp(y);
    const logQR = logN(x, mu, s) - logN(y, mu, s);   // log q(x|y) − log q(y|x)
    const logAlpha = lpy - lp + (correct ? logQR : 0);
    const u = rng();
    if (Math.log(u) < logAlpha) { x = y; lp = lpy; acc++; }
    chain.push(x.slice());
  }
  return { chain, acc };
}

/* ---------- 结果记录 ---------- */
const results = [];
function check(id, name, ok, detail) {
  results.push({ id, ok });
  console.log((ok ? '  +OK ' : '  -XX ') + id + ' ' + name + (detail ? '  | ' + detail : ''));
}

/* ==========================================================================
 * R1  MH 接受修正独立三路对照（INV-MC-001）
 * ========================================================================== */
{
  const mu = [2], sd = [1.5];
  const logp = (x) => { let t = 0; for (let k = 0; k < 1; k++) { const z = (x[k] - mu[k]) / sd[k]; t += z * z; } return -0.5 * Math.log(2 * Math.PI) - Math.log(sd[0]) - 0.5 * t; };
  const STEPS = 60000, BURN = 5000;
  // (a) 独立对称提议 MH
  const a = ownMhChain(logp, 1, [0], STEPS, 3.57, mulberry32(0x0A011), true).chain.slice(BURN).map((v) => v[0]);
  const ea = ownEss(a), ma = ownMean(a), va = ownVar(a);
  const za = Math.max(Math.abs(ma - 2) / (1.5 / Math.sqrt(ea)), Math.abs(va - 2.25) / (2.25 * Math.sqrt(2 / ea)));
  // (b) 独立非对称提议 + Hastings 修正
  const b = ownMhIndepChain(logp, 1, [0], STEPS, [0], 3, mulberry32(0x0B022), true).chain.slice(BURN).map((v) => v[0]);
  const eb = ownEss(b), mb = ownMean(b);
  const zb = Math.abs(mb - 2) / (1.5 / Math.sqrt(eb));
  // (c) 独立非对称提议、去掉修正 → 平稳分布 π·q ∝ N(2,1.5²)·N(0,3)
  //     精度: 1/σ² = 1/2.25 + 1/9 → σ²=1.8, μ = 2·(1/2.25)/(1/2.25+1/9)=1.6
  const c = ownMhIndepChain(logp, 1, [0], STEPS, [0], 3, mulberry32(0x0C033), false).chain.slice(BURN).map((v) => v[0]);
  const ec = ownEss(c), mc = ownMean(c), vc = ownVar(c);
  const zc = Math.abs(mc - 1.6) / (Math.sqrt(1.8) / Math.sqrt(ec));
  // (d) 模块自身 Metropolis 采样器矩收敛
  const tgt = G.makeGaussTarget([2], [1.5]);
  const d = G.metropolisSample({ logTarget: tgt.logTarget, dim: 1, x0: [0], steps: STEPS, burnIn: BURN, stepSize: 3.57, rng: G.makeRng(0x0D044) });
  const dd = d.chain.map((v) => v[0]);
  const ed = ownEss(dd), md = ownMean(dd), vd = ownVar(dd);
  const zd = Math.max(Math.abs(md - 2) / (1.5 / Math.sqrt(ed)), Math.abs(vd - 2.25) / (2.25 * Math.sqrt(2 / ed)));
  const ok = za <= 10 && zb <= 10 && zc <= 10 && zd <= 10 && Math.abs(mc - 2) > 5 * (Math.sqrt(1.8) / Math.sqrt(ec));
  check('R1', '独立 MH 三路：对称≡目标 / 非对称+修正≡目标 / 非对称−修正≡π·q 且偏离目标 >5σ；模块 Metropolis 矩收敛', ok,
    'z对称=' + za.toFixed(2) + ' z非对称修正=' + zb.toFixed(2) + ' z无修正(→π·q)=' + zc.toFixed(2) + ' mean无修正=' + mc.toFixed(3) + '(目标2.0, π·q=1.6) 模块Metropolis z=' + zd.toFixed(2));
}

/* ==========================================================================
 * R2  离散两状态细致平衡（INV-MC-002）
 * ========================================================================== */
{
  // 独立解析核（本文件独立计算，不用模块任何内部量）
  const pi = [0.7, 0.3];
  const Q = [[0.2, 0.8], [0.6, 0.4]];
  const alpha = [[0, 0], [0, 0]];
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
    if (i !== j) alpha[i][j] = Math.min(1, (pi[j] * Q[j][i]) / (pi[i] * Q[i][j]));
  }
  const P = [[0, 0], [0, 0]];
  for (let i = 0; i < 2; i++) {
    let off = 0;
    for (let j = 0; j < 2; j++) if (j !== i) { P[i][j] = Q[i][j] * alpha[i][j]; off += P[i][j]; }
    P[i][i] = 1 - off;
  }
  let db = 0;
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) db = Math.max(db, Math.abs(pi[i] * P[i][j] - pi[j] * P[j][i]));
  // 恒接受核必违反
  const Pa = [[1 - Q[0][1], Q[0][1]], [Q[1][0], 1 - Q[1][0]]];
  let dbN = 0;
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) dbN = Math.max(dbN, Math.abs(pi[i] * Pa[i][j] - pi[j] * Pa[j][i]));
  // 模块链（模块自带离散提议）
  const propose = (x, rng) => {
    const s = x[0];
    const y = rng() < Q[s][0] ? 0 : 1;
    return { y: [y], logQRatio: Math.log(Q[y][s]) - Math.log(Q[s][y]) };
  };
  const N = 80000;
  const r = G.mhSample({ logTarget: (x) => Math.log(pi[x[0]]), dim: 1, x0: [0], steps: N, rng: G.makeRng(0x0212A5), propose });
  let n0 = 0, n00 = 0, n01 = 0, n10 = 0, n11 = 0;
  for (let i = 1; i < r.chain.length; i++) {
    const a0 = r.chain[i - 1][0], a1 = r.chain[i][0];
    if (a0 === 0) { n0++; if (a1 === 0) n00++; else n01++; } else { if (a1 === 0) n10++; else n11++; }
  }
  const n1 = N - 1 - n0;
  const freq0 = n0 / (N - 1);
  const emp = [[n00 / n0, n01 / n0], [n10 / n1, n11 / n1]];
  let empZ = 0;
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
    const nij = i === 0 ? n0 : n1;
    const sig = 1.5 * Math.sqrt(Math.max(P[i][j] * (1 - P[i][j]), 1e-12) / nij);
    empZ = Math.max(empZ, Math.abs(emp[i][j] - P[i][j]) / sig);
  }
  const freqSig = 1.5 * Math.sqrt(0.21 / (N - 1)) * Math.sqrt(2.1);
  const freqZ = Math.abs(freq0 - 0.7) / freqSig;
  const ok = db <= 1e-15 && dbN > 0.1 && freqZ <= 10 && empZ <= 10 / 1.5;
  check('R2', '独立解析核细致平衡 EXACT + 恒接受违反 >0.1 + 模块链频率/转移矩阵 10σ', ok,
    'dbErr=' + db.toExponential(2) + ' dbNaive=' + dbN.toFixed(3) + ' freq0=' + freq0.toFixed(4) + ' z=' + freqZ.toFixed(2) + ' maxTransZ=' + empZ.toFixed(2));
}

/* ==========================================================================
 * R3  RWM 矩收敛 + KS + 最优窗宽 + 调优（INV-MC-003）
 * ========================================================================== */
{
  const tgt = G.makeGaussTarget([3], [2]);
  let worstZ = 0, worstKS = 0, acc = 0;
  for (const sd of [0x30A001, 0x30B002, 0x30C003]) {
    const r = G.rwMetropolis({ logTarget: tgt.logTarget, dim: 1, x0: [0], steps: 50000, burnIn: 3000, stepSize: 4.76, rng: G.makeRng(sd) });
    const xs = r.chain.map((v) => v[0]);
    const e = ownEss(xs), m = ownMean(xs), v = ownVar(xs);
    worstZ = Math.max(worstZ, Math.abs(m - 3) / (2 / Math.sqrt(e)), Math.abs(v - 4) / (4 * Math.sqrt(2 / e)));
    worstKS = Math.max(worstKS, ownKsNormal(xs, 3, 2) * Math.sqrt(e));
    acc = r.acceptRate;
  }
  const tuned = G.rwMetropolis({ logTarget: tgt.logTarget, dim: 1, x0: [0], steps: 12000, tune: true, tuneTarget: 0.44, rng: G.makeRng(0x7B0E0002) });
  // 独立恒接受随机游走
  const rng = mulberry32(0xFACE0F2);
  let x = 0; const rw = [];
  for (let i = 0; i < 50000; i++) { x += 4.76 * ownGaussVec(rng, 1)[0]; if (i >= 3000) rw.push(x); }
  const rwVar = ownVar(rw);
  const ok = worstZ <= 10 && worstKS <= 1.63 && acc >= 0.38 && acc <= 0.52 &&
    Math.abs(tuned.tuneAcceptRate - 0.44) <= 0.05 && rwVar > 100;
  check('R3', '独立 ESS + 独立 Simpson-CDF KS + 最优窗宽接受率≈0.44 + 调优收敛 + 恒接受证伪', ok,
    'worstZ=' + worstZ.toFixed(2) + ' KS√n=' + worstKS.toFixed(3) + ' accept=' + acc.toFixed(4) + ' tuned=' + tuned.tuneAcceptRate.toFixed(4) + ' step=' + tuned.stepSize.toFixed(2) + ' rwVar=' + rwVar.toFixed(0));
}

/* ==========================================================================
 * R4  Gibbs 条件矩（INV-MC-004）
 * ========================================================================== */
{
  const rho = 0.6;
  const r = G.gibbsSample({ rho, x0: [0, 0], steps: 70000, burnIn: 3000, rng: G.makeRng(0x61BB07) });
  const xs = r.chain.map((c) => c[0]), ys = r.chain.map((c) => c[1]);
  const e = Math.min(ownEss(xs), ownEss(ys));
  const mx = ownMean(xs), my = ownMean(ys), vx = ownVar(xs), vy = ownVar(ys);
  // 独立回归：slope = corr·sx/sy（与模块协方差公式不同代数路径）
  let cxy = 0;
  for (let i = 0; i < xs.length; i++) cxy += (xs[i] - mx) * (ys[i] - my);
  cxy /= (xs.length - 1);
  const corr = cxy / Math.sqrt(vx * vy);
  const slope = corr * Math.sqrt(vx / vy);
  const resid = xs.map((x, i) => x - (mx + slope * (ys[i] - my)));
  const resVar = ownVar(resid);
  const zSlope = Math.abs(slope - rho) / Math.sqrt(resVar / (e * vy));
  const zRes = Math.abs(resVar - 0.64) / (0.64 * Math.sqrt(2 / e));
  const zVar = Math.abs(vx - 1) / Math.sqrt(2 / e);
  const lag1 = (() => { let n = 0, d = 0; for (let i = 1; i < xs.length; i++) { n += (xs[i] - mx) * (xs[i - 1] - mx); d += (xs[i - 1] - mx) * (xs[i - 1] - mx); } return n / d; })();
  // 错误条件方差证伪（模块注入 API）
  const rw = G.gibbsSample({ rho, condVar: 1.0, x0: [0, 0], steps: 70000, burnIn: 3000, rng: G.makeRng(0x61BB07) });
  const rwVx = ownVar(rw.chain.map((c) => c[0]));
  const ok = zSlope <= 10 && zRes <= 10 && zVar <= 10 && lag1 < 0.95 && Math.abs(rwVx - 1) > 0.3;
  check('R4', '独立回归公式(corr·sx/sy)：斜率≈ρ/残差方差≈1-ρ²/边际方差≈1（10σ）+ 错误条件方差证伪 + 链移动', ok,
    'slope=' + slope.toFixed(4) + ' z=' + zSlope.toFixed(2) + ' resVar=' + resVar.toFixed(4) + ' z=' + zRes.toFixed(2) + ' varX=' + vx.toFixed(4) + ' z=' + zVar.toFixed(2) + ' lag1=' + lag1.toFixed(3) + ' wrongCondVar=' + rwVx.toFixed(4));
}

/* ==========================================================================
 * R5  leapfrog 三性质（INV-MC-005）
 * ========================================================================== */
{
  const sd = [1, 10];
  const logp = (x) => -Math.log(2 * Math.PI) - Math.log(sd[0]) - Math.log(sd[1])
    - 0.5 * ((x[0] / sd[0]) ** 2 + (x[1] / sd[1]) ** 2);
  const grad = (x) => [-x[0] / (sd[0] * sd[0]), -x[1] / (sd[1] * sd[1])];
  // (a) 独立实现对拍
  let maxDiff = 0;
  const rng = mulberry32(0x1EAF01);
  for (let t = 0; t < 40; t++) {
    const x = [4 * (rng() - 0.5), 12 * (rng() - 0.5)], p = [2 * (rng() - 0.5), 2 * (rng() - 0.5)];
    const eps = 0.05 + 0.35 * rng(), L = 1 + Math.floor(rng() * 20);
    const m = G.leapfrog({ gradLogTarget: grad, x, p, eps, L });
    const o = ownLeapfrog(grad, x, p, eps, L);
    for (let k = 0; k < 2; k++) maxDiff = Math.max(maxDiff, Math.abs(m.x[k] - o.x[k]), Math.abs(m.p[k] - o.p[k]));
  }
  // (b) 可逆（独立实现）
  const x0 = [0.7, -1.3], p0 = [0.4, 0.9];
  const f1 = ownLeapfrog(grad, x0, p0, 0.25, 12);
  const bk = ownLeapfrog(grad, f1.x, f1.p.map((v) => -v), 0.25, 12);
  let revErr = 0;
  for (let k = 0; k < 2; k++) revErr = Math.max(revErr, Math.abs(bk.x[k] - x0[k]), Math.abs(bk.p[k] + p0[k]));
  // (c) 保体积（独立有限差分 Jacobian，4×4）
  const epsM = 0.3, LM = 3, h = 1e-6;
  const base = [0.5, -0.4, 0.3, 0.6];
  const mapSt = (st) => { const lf = ownLeapfrog(grad, [st[0], st[1]], [st[2], st[3]], epsM, LM); return [lf.x[0], lf.x[1], lf.p[0], lf.p[1]]; };
  const J = [[], [], [], []];
  for (let i = 0; i < 4; i++) {
    const sp = base.slice(), sm = base.slice();
    sp[i] += h; sm[i] -= h;
    const fp = mapSt(sp), fm = mapSt(sm);
    for (let r2 = 0; r2 < 4; r2++) J[r2][i] = (fp[r2] - fm[r2]) / (2 * h);
  }
  const det4 = (M) => {
    const mn = (r, c) => {
      const m = [];
      for (let i = 0; i < 4; i++) { if (i === r) continue; const row = []; for (let j = 0; j < 4; j++) { if (j === c) continue; row.push(M[i][j]); } m.push(row); }
      return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    };
    return M[0][0] * mn(0, 0) - M[0][1] * mn(0, 1) + M[0][2] * mn(0, 2) - M[0][3] * mn(0, 3);
  };
  const detErr = Math.abs(det4(J) - 1);
  // (d) 能量有界 vs Euler 漂移（独立实现）
  const rng2 = mulberry32(0x1EAF02);
  const H = (xx, pp) => -logp(xx) + 0.5 * (pp[0] * pp[0] + pp[1] * pp[1]);
  let lfMax = 0, euMin8 = Infinity, euMax160 = 0;
  for (let c = 0; c < 24; c++) {
    const xs = [3 * (rng2() - 0.5), 6 * (rng2() - 0.5)], ps = [2 * (rng2() - 0.5), 2 * (rng2() - 0.5)];
    const H0 = H(xs, ps);
    for (const L of [8, 40, 160]) {
      const lf = ownLeapfrog(grad, xs, ps, 0.2, L);
      lfMax = Math.max(lfMax, Math.abs(H(lf.x, lf.p) - H0));
    }
    const e8 = ownEuler(grad, xs, ps, 0.2, 8);
    const e160 = ownEuler(grad, xs, ps, 0.2, 160);
    euMin8 = Math.min(euMin8, Math.abs(H(e8.x, e8.p) - H0));
    euMax160 = Math.max(euMax160, Math.abs(H(e160.x, e160.p) - H0));
  }
  const ok = maxDiff <= 1e-12 && revErr <= 1e-12 && detErr <= 1e-6 && lfMax <= 0.05 && euMax160 > 5 * Math.max(euMin8, 1e-9) && euMax160 > 0.1;
  check('R5', '独立实现对拍≤1e-12 + 可逆≤1e-12 + |detJ−1|≤1e-6 + 能量有界≤0.05（Euler 漂移对照）', ok,
    'maxDiff=' + maxDiff.toExponential(2) + ' revErr=' + revErr.toExponential(2) + ' detErr=' + detErr.toExponential(2) + ' lfMaxΔH=' + lfMax.toExponential(2) + ' euler(8)min=' + euMin8.toExponential(2) + ' euler(160)max=' + euMax160.toExponential(2));
}

/* ==========================================================================
 * R6  NUTS / 混合MC 效率（INV-MC-006）
 * ========================================================================== */
{
  const tgt = G.makeGaussTarget([0, 0], [1, 10]);
  // U-turn 判据单测（直接调用模块导出函数）
  const utOk = G.uTurnCriterion([-1, 0], [1, 0], [1, 0], [1, 0]) === true &&
    G.uTurnCriterion([-1, 0], [1, 0], [1, 0], [-1, 0]) === false &&
    G.uTurnCriterion([-1, 0], [1, 0], [-1, 0], [-1, 0]) === false &&
    G.uTurnCriterion([0, -1], [0, 1], [0, 1], [0, 1]) === true;
  // RWM 基线（独立 ESS）
  const rwm = G.rwMetropolis({ logTarget: tgt.logTarget, dim: 2, x0: [0, 0], steps: 16000, burnIn: 2000, stepSize: 1.68, rng: G.makeRng(0x08000001) });
  const rwmEff = ownEss(rwm.chain.map((c) => c[0]));
  // NUTS（模块自带 init_stepsize + dual averaging）
  const nuts = G.nutsSample({ logTarget: tgt.logTarget, gradLogTarget: tgt.gradLogTarget, dim: 2, x0: [0, 0], steps: 16000, burnIn: 2000, rng: G.makeRng(0x0E700002), maxDepth: 6, adapt: { steps: 800, target: 0.65 } });
  const nutsEff = ownEss(nuts.chain.map((c) => c[0]));
  const effRatio = nutsEff / rwmEff;
  // 混合 MC：轨迹长卡方（独立计算）
  const hyb = G.hybridMonteCarlo({ logTarget: tgt.logTarget, gradLogTarget: tgt.gradLogTarget, dim: 2, x0: [0, 0], steps: 16000, burnIn: 2000, eps: 1.2, Lmax: 8, rng: G.makeRng(0x44B00003) });
  const hybEff = ownEss(hyb.chain.map((c) => c[0]));
  const hybRatio = hybEff / rwmEff;
  const nAll = 16000;   // trajFreq 统计全部步（含适应期前的 burn-in 亦计数）
  let chi2 = 0;
  for (let L = 1; L <= 8; L++) {
    const exp = nAll / 8;
    chi2 += (hyb.trajFreq[L - 1] * nAll - exp) ** 2 / exp;
  }
  // 超大 eps 证伪
  const big = G.nutsSample({ logTarget: tgt.logTarget, gradLogTarget: tgt.gradLogTarget, dim: 2, x0: [0, 0], steps: 800, eps: 5.0, rng: G.makeRng(0x816E0005), maxDepth: 4 });
  const ok = utOk && effRatio > 3 && hybRatio > 2 && nuts.uturnRate > 0.1 && nuts.acceptStat >= 0.5 && nuts.acceptStat <= 0.95 && big.acceptStat < 0.2 && chi2 < 30;
  check('R6', 'U-turn 判据单测 + NUTS/混合MC ESS 比 >3/>2（独立 ESS）+ 轨迹长卡方<40 + 超大 eps 接受率崩塌', ok,
    'uturn=' + utOk + ' rwmESS=' + rwmEff.toFixed(0) + ' nuts×' + effRatio.toFixed(2) + ' hyb×' + hybRatio.toFixed(2) + ' chi2=' + chi2.toFixed(2) + ' accept=' + nuts.acceptStat.toFixed(3) + ' bigEps=' + big.acceptStat.toFixed(3));
}

/* ==========================================================================
 * R7  dual averaging 收敛（INV-MC-007）——用独立 HMC 步测量模块自适应步长
 * ========================================================================== */
{
  const sd = [1, 3];
  const logp = (x) => -Math.log(2 * Math.PI) - Math.log(3) - Math.log(sd[0]) - Math.log(sd[1])
    - 0.5 * ((x[0] / sd[0]) ** 2 + (x[1] / sd[1]) ** 2);
  const grad = (x) => [-x[0] / (sd[0] * sd[0]), -x[1] / (sd[1] * sd[1])];
  const measure = (target, seed) => {
    const tgt = G.makeGaussTarget([0, 0], sd);
    const r = G.hmcWithAdaptation({ logTarget: tgt.logTarget, gradLogTarget: tgt.gradLogTarget, dim: 2, x0: [0.5, -0.5], adaptSteps: 1500, sampleSteps: 3000, L: 12, rng: G.makeRng(seed), target });
    // 独立 HMC 步在 epsBar 下复测平均接受概率
    const rng = mulberry32(seed ^ 0x5EED);
    let x = [0.5, -0.5], sum = 0;
    for (let i = 0; i < 4000; i++) {
      const s = ownHmcStep(logp, grad, x, r.epsBar, 12, rng);
      x = s.x; sum += s.acceptProb;
    }
    return { epsBar: r.epsBar, ownProb: sum / 4000, moduleProb: r.meanAcceptProb };
  };
  const r65 = measure(0.65, 0xDA65001);
  const r90 = measure(0.90, 0xDA90002);
  const e65 = Math.abs(r65.ownProb - 0.65), e90 = Math.abs(r90.ownProb - 0.90);
  const ok = e65 <= 0.05 && e90 <= 0.05 && r90.epsBar < r65.epsBar * 0.98;
  check('R7', '独立 HMC 步复测模块自适应步长：目标 0.65/0.9 平均接受概率均收敛 ±0.05 且 eps(0.9)<eps(0.65)', ok,
    'ownProb65=' + r65.ownProb.toFixed(4) + ' modProb65=' + r65.moduleProb.toFixed(4) + ' eps65=' + r65.epsBar.toFixed(4) +
    ' ownProb90=' + r90.ownProb.toFixed(4) + ' eps90=' + r90.epsBar.toFixed(4));
}

/* ==========================================================================
 * R8  退火（INV-MC-008）
 * ========================================================================== */
{
  const lb = (x) => {
    const a = Math.log(0.85) + (-0.5 * Math.log(2 * Math.PI) - 0.5 * (x - 5) ** 2);
    const b = Math.log(0.15) + (-0.5 * Math.log(2 * Math.PI) - 0.5 * (x + 5) ** 2);
    const hi = Math.max(a, b);
    return hi + Math.log(Math.exp(a - hi) + Math.exp(b - hi));
  };
  const CFG = { steps: 9000, stepSize: 1.0, T0: 8, Tmin: 0.04, hold: 3000 };
  // (a) 独立调度公式对拍（EXACT）
  const a1 = G.annealSample({ logTarget: (x) => lb(x[0]), dim: 1, x0: [-5], ...CFG, rng: G.makeRng(0x0A11E001) });
  const ownSched = (steps, T0, Tmin, hold) => {
    const h = Math.min(hold, steps - 1), denom = Math.max(1, steps - 1 - h), Ts = [];
    for (let k = 0; k < steps; k++) Ts.push(k < h ? T0 : T0 * Math.pow(Tmin / T0, (k - h) / denom));
    return Ts;
  };
  const os = ownSched(CFG.steps, CFG.T0, CFG.Tmin, CFG.hold);
  let schedExact = os.length === a1.schedule.length;
  for (let k = 0; k < os.length && schedExact; k++) if (os[k] !== a1.schedule[k]) schedExact = false;
  // (b) 确定性
  const a2 = G.annealSample({ logTarget: (x) => lb(x[0]), dim: 1, x0: [-5], ...CFG, rng: G.makeRng(0x0A11E001) });
  const a3 = G.annealSample({ logTarget: (x) => lb(x[0]), dim: 1, x0: [-5], ...CFG, rng: G.makeRng(0x0A11E002) });
  const identical = JSON.stringify(a1.chain) === JSON.stringify(a2.chain);
  const diverged = JSON.stringify(a1.chain) !== JSON.stringify(a3.chain);
  // (c) 8 种子全局模 + 末态质量
  let hits = 0, sumA = 0;
  for (const sd of [101, 202, 303, 404, 505, 606, 707, 808]) {
    const a = G.annealSample({ logTarget: (x) => lb(x[0]), dim: 1, x0: [-5], ...CFG, rng: G.makeRng(sd) });
    const f = a.chain[a.chain.length - 1][0];
    if (Math.abs(f - 5) <= 2) hits++;
    sumA += lb(f);
  }
  let sumC = 0, trapOk = true;
  for (const sd of [101, 202, 303, 404, 505, 606, 707, 808]) {
    const c = G.annealSample({ logTarget: (x) => lb(x[0]), dim: 1, x0: [-5], steps: CFG.steps, stepSize: 1.0, T0: 0.04, Tmin: 0.04, rng: G.makeRng(sd) });
    sumC += lb(c.chain[c.chain.length - 1][0]);
  }
  for (const sd of [0x0C01D001, 0x0C01D002, 0x0C01D003]) {
    const c = G.annealSample({ logTarget: (x) => lb(x[0]), dim: 1, x0: [-5], steps: CFG.steps, stepSize: 1.0, T0: 0.04, Tmin: 0.04, rng: G.makeRng(sd) });
    let maxX = -Infinity;
    for (const cc of c.chain) maxX = Math.max(maxX, cc[0]);
    if (!(maxX <= 1.5)) trapOk = false;
  }
  const gap = sumA / 8 - sumC / 8;
  // (d) 降温致锐
  const gt = G.makeGaussTarget([3], [2]);
  const sharp = G.annealSample({ logTarget: gt.logTarget, dim: 1, x0: [3], steps: 4000, stepSize: 0.9, T0: 4, Tmin: 0.25, rng: G.makeRng(0x5A2B0001) });
  const vF = ownVar(sharp.chain.slice(0, 800).map((c) => c[0]));
  const vL = ownVar(sharp.chain.slice(-800).map((c) => c[0]));
  const cT = G.annealSample({ logTarget: gt.logTarget, dim: 1, x0: [3], steps: 4000, stepSize: 0.9, T0: 4, Tmin: 4, rng: G.makeRng(0x5A2B0002) });
  const cF = ownVar(cT.chain.slice(0, 800).map((c) => c[0]));
  const cL = ownVar(cT.chain.slice(-800).map((c) => c[0]));
  const ok = schedExact && identical && diverged && hits >= 5 && gap > 0.5 && trapOk && vL < vF / 4 && cL > cF * 0.6 && cL < cF * 1.6;
  check('R8', '独立调度公式对拍 EXACT + 确定性 + ≥5/8 全局模 + 末态质量 gap>0.5 + 冷链困陷 + 降温致锐/恒温对照', ok,
    'schedExact=' + schedExact + ' identical=' + identical + ' diverged=' + diverged + ' hits=' + hits + '/8 gap=' + gap.toFixed(3) + ' trap=' + trapOk + ' varRatio=' + (vL / vF).toFixed(4) + ' constRatio=' + (cL / cF).toFixed(3));
}

/* ---------- 汇总 ---------- */
const failed = results.filter((r) => !r.ok).length;
console.log('══════ mcmc-forge 独立复核: ' + (results.length - failed) + '/' + results.length + ' 通过 ══════');
process.exit(failed === 0 ? 0 : 1);
