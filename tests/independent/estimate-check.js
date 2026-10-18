'use strict';
/* ============================================================================
 * 独立复核 estimate-forge —— 不采信模块 selfCheck()，自带 mulberry32 PRNG、
 * Simpson 数值积分（正态 CDF / χ² CDF / Student-t CDF）、全部黄金参照均为本文件
 * 独立实现（闭式解 / 手算 / 二分 / 黄金分割 / 独立 Bootstrap / 独立 Huber /
 * 独立经验似然对偶求解）。
 *
 * 覆盖：
 *   R1  MLE 闭式对拍（正态 μ̂/σ̂²_ML、伯努利 p̂、泊松 λ̂）+ Newton 得分 ≤1e-9
 *       + 矩估计与 MLE 一致性（INV-EST-001）
 *   R2  MAP / 贝叶斯共轭后验均值 == 独立解析式 (τ₀μ₀+nτx̄)/(τ₀+nτ)；平坦先验
 *       (τ₀→0) 退化为 x̄；众数==均值；黄金分割数值最大化后验独立对拍（INV-EST-002）
 *   R3  KS D 手算 EXACT（自带 Simpson 正态 CDF 双轨对拍）+ p 单调 + 边界
 *       (D=0→p=1, D 大→p→0) + 功效 >0.9 / 伪功效 <0.15（INV-EST-003）
 *   R4  χ² 拟合优度手算 EXACT + 2×2 独立性解析 25.2525（E_ij=row·col/N 自算）
 *       + 独立 Simpson χ² CDF 对拍 + df = 类别数−1−估计参数数 + 功效 >0.9（INV-EST-004）
 *   R5  Bootstrap SE 与解析 SE/√n 相对误差 ≤15%（自带 std + 独立重采样双轨）
 *       + 百分位 CI 中心 ≈ x̄ + 均值一致（INV-EST-005）
 *   R6  t / 正态 / Wilson 95% CI 蒙特卡洛覆盖 ∈[0.92,0.98]（2000 次；Wilson 用
 *       真 Binomial(100,0.5) 抽样）+ 解析公式对拍 + Wilson 边界手算 EXACT（INV-EST-006）
 *   R7  Welch t 手算 EXACT + 独立 Simpson Student-t CDF 对拍 + CDF(0,ν)=0.5
 *       + I 类错误 ∈[0.03,0.07] + d=0.8 功效 >0.8（INV-EST-007）
 *   R8  Wilks LRT（全模型同时解放 μ 与 σ ⇒ df=2）中位 p ∈[0.4,0.6] + χ²(2) 闭式
 *       p=e^{−stat/2} 对拍 + 拟似然 Poisson 退化为 MLE(φ≈1) + Huber 降权
 *       （独立二分对拍）+ 经验似然极大点在 x̄ 且离均值点与独立对偶解一致（INV-EST-008）
 *
 * 本文件不调用 estimate-forge.selfCheck / makeRng；所有随机量由本文件 mulberry32
 * 生成，与模块内部 sfc32 轨道独立；黄金参照均为本文件独立实现。
 * 退出码：0 = 全部独立复核通过；1 = 任一失败。
 * ========================================================================== */
const path = require('path');
const G = require(path.join(__dirname, '..', '..', 'src', 'estimate-forge.js'));

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

/* ---------- 独立正态向量（Box-Muller，本文件自己消费） ---------- */
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

/* ---------- 独立基础统计量 ---------- */
function ownMean(xs) { let s = 0; for (const x of xs) s += x; return s / xs.length; }
function ownVarSample(xs) { const m = ownMean(xs); let s = 0; for (const x of xs) s += (x - m) * (x - m); return s / (xs.length - 1); }
function ownVarML(xs) { const m = ownMean(xs); let s = 0; for (const x of xs) s += (x - m) * (x - m); return s / xs.length; }
function ownStdSample(xs) { return Math.sqrt(ownVarSample(xs)); }
function ownMedian(xs) {
  const s = xs.slice().sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]);
}

/* ---------- 复合 Simpson 求积（本文件唯一数值积分原语） ---------- */
function simpson(f, a, b, N) {
  if (a === b) return 0;
  const h = (b - a) / N;
  let s = f(a) + f(b);
  for (let i = 1; i < N; i++) s += (i % 2 === 1 ? 4 : 2) * f(a + i * h);
  return s * h / 3;
}

/* ---------- 独立正态 CDF：Simpson 积分 φ（与模块 erf/lowerRegGamma 路径无关） ---------- */
function ownNormalPdf(x) { return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI); }
function ownNormalCdf(x) {
  if (x === 0) return 0.5;
  const A = Math.abs(x);
  const half = simpson(ownNormalPdf, 0, A, 20000);
  const v = x > 0 ? 0.5 + half : 0.5 - half;
  return Math.max(0, Math.min(1, v));
}
/** 独立正态逆 CDF：在自有 CDF 上二分 */
function ownNormInv(p) {
  let lo = -40, hi = 40;
  for (let i = 0; i < 60; i++) { const mid = 0.5 * (lo + hi); if (ownNormalCdf(mid) < p) lo = mid; else hi = mid; }
  return 0.5 * (lo + hi);
}

/* ---------- 独立 χ² CDF：Simpson 积分密度，代换 t=u² 消除 k=1 处端点奇点 ----------
 *   F(x;k) = ∫_0^{√x} 2·u^{k-1}·e^{−u²/2} / (2^{k/2}·Γ(k/2)) du
 *   Γ 借用模块 logGamma（允许），积分与代换均为本文件独立实现。            */
function ownChiSquareCdf(x, k) {
  if (x <= 0) return 0;
  const lc = Math.log(2) * (k / 2) + G.logGamma(k / 2);
  const g = (u) => (k === 1 ? 2 : 2 * Math.pow(u, k - 1)) * Math.exp(-0.5 * u * u - lc);
  const v = simpson(g, 0, Math.sqrt(x), 20000);
  return Math.max(0, Math.min(1, v));
}
function ownChiSquareSf(x, k) { return 1 - ownChiSquareCdf(x, k); }

/* ---------- 独立 Student-t CDF：Simpson 积分 t 密度（本文件独立实现） ---------- */
function ownTCdf(t, df) {
  if (t === 0) return 0.5;
  const lc = G.logGamma((df + 1) / 2) - G.logGamma(df / 2) - 0.5 * Math.log(df * Math.PI);
  const f = (x) => Math.exp(lc - ((df + 1) / 2) * Math.log(1 + x * x / df));
  const A = Math.abs(t);
  const half = simpson(f, 0, A, 20000);
  const v = t > 0 ? 0.5 + half : 0.5 - half;
  return Math.max(0, Math.min(1, v));
}
/** 独立 t 分位数（自有 CDF 上二分，面板数略降以控时） */
function ownTQuantile(p, df) {
  let lo = -60, hi = 60;
  for (let i = 0; i < 60; i++) {
    const mid = 0.5 * (lo + hi);
    const c = ownTCdf(mid, df);
    if (c < p) lo = mid; else hi = mid;
  }
  return 0.5 * (lo + hi);
}

/* ---------- 独立 KS 统计量（D/D₊/D₋，自带排序与扫描） ---------- */
function ownKsStat(data, cdf) {
  const xs = data.slice().sort((a, b) => a - b);
  const n = xs.length;
  let D = 0, Dplus = 0, Dminus = 0;
  for (let i = 0; i < n; i++) {
    const F = cdf(xs[i]);
    const below = (i + 1) / n - F;
    const above = F - i / n;
    if (below > Dplus) Dplus = below;
    if (above > Dminus) Dminus = above;
    if (below > D) D = below;
    if (above > D) D = above;
  }
  return { D, Dplus, Dminus, n };
}

/* ---------- 独立非参数 Bootstrap 重采样 ---------- */
function ownBootstrap(data, statFn, B, rng) {
  const n = data.length;
  const out = new Array(B);
  for (let b = 0; b < B; b++) {
    const s = new Array(n);
    for (let i = 0; i < n; i++) {
      const k = Math.floor(rng() * n);
      s[i] = data[k < n ? k : n - 1];
    }
    out[b] = statFn(s);
  }
  return out;
}

/* ---------- 独立多项抽样（用于 χ² 功效） ---------- */
function ownMultinomial(trials, probs, rng) {
  const cps = []; let s = 0;
  for (const p of probs) { s += p; cps.push(s); }
  const counts = new Array(probs.length).fill(0);
  for (let i = 0; i < trials; i++) {
    const u = rng() * s;
    let c = 0;
    while (c < cps.length - 1 && u > cps[c]) c++;
    counts[c]++;
  }
  return counts;
}

/* ---------- 独立 Poisson 抽样（Knuth） ---------- */
function ownPoisson(rng, lambda) {
  const L = Math.exp(-lambda);
  let k = 0, p = 1;
  do { k++; p *= rng(); } while (p > L);
  return k - 1;
}

/* ---------- 独立黄金分割最大化（用于 MAP 数值对拍） ---------- */
function goldenMax(f, a, b, iters) {
  const phi = 0.5 * (Math.sqrt(5) - 1);
  let c = b - phi * (b - a), d = a + phi * (b - a);
  let fc = f(c), fd = f(d);
  for (let i = 0; i < iters; i++) {
    if (fc < fd) { a = c; c = d; fc = fd; d = a + phi * (b - a); fd = f(d); }
    else { b = d; d = c; fd = fc; c = b - phi * (b - a); fc = f(c); }
  }
  return 0.5 * (a + b);
}

/* ---------- 独立 Huber M 估计：二分求解 Σψ((x−μ)/s)=0 ---------- */
function ownHuberMu(data, scale, c) {
  const h = (m) => {
    let s = 0;
    for (const x of data) {
      const r = (x - m) / scale;
      s += Math.abs(r) <= c ? r : (r > 0 ? c : -c);
    }
    return s;                       // 关于 m 严格递减
  };
  let lo = data[0], hi = data[0];
  for (const x of data) { if (x < lo) lo = x; if (x > hi) hi = x; }
  lo -= 100 * scale; hi += 100 * scale;
  for (let i = 0; i < 200; i++) { const mid = 0.5 * (lo + hi); if (h(mid) > 0) lo = mid; else hi = mid; }
  return 0.5 * (lo + hi);
}

/* ---------- 独立经验似然对偶解：二分求解 Σ z_i/(1+λ z_i) = 0 ----------
 *   p_i = 1/(n(1+λ z_i))，z_i = x_i − μ₀；极大经验似然的 KKT 条件即此方程。
 *   与模块所用的 Σ1/(1+λz)=n 形式代数等价但数值路径不同（模块式含平凡根 λ=0）。
 *   ---------------------------------------------------------------- */
function ownELMinus2LogR(data, mu0) {
  const z = data.map((x) => x - mu0);
  let zmax = 0, znegAbs = 0;
  for (const v of z) { if (v > zmax) zmax = v; const av = -v; if (av > znegAbs) znegAbs = av; }
  const eps = 1e-12;
  const lo = zmax > 0 ? -(1 - eps) / zmax : -1;
  const hi = znegAbs > 0 ? (1 - eps) / znegAbs : 1;
  const h = (lam) => { let s = 0; for (const v of z) s += v / (1 + lam * v); return s; };
  let a = lo, b = hi;
  for (let i = 0; i < 200; i++) { const mid = 0.5 * (a + b); if (h(mid) > 0) a = mid; else b = mid; }
  const lam = 0.5 * (a + b);
  let m2 = 0;
  for (const v of z) m2 += 2 * Math.log(1 + lam * v);
  return { lam, minus2LogR: m2 };
}

/* ---------- 结果记录 ---------- */
const results = [];
function check(id, name, ok, detail) {
  results.push({ id, ok });
  console.log((ok ? '  +OK ' : '  -XX ') + id + ' ' + name + (detail ? '  | ' + detail : ''));
}

/* ==========================================================================
 * R1  MLE 闭式 EXACT + 牛顿收敛 + 矩估计一致（INV-EST-001）
 * ========================================================================== */
{
  const dataN = [-1.2, 0.3, 2.1, -0.7, 1.5, 0.9, -0.4, 1.8, 0.2, -1.0, 0.55, -0.15];
  const dataB = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0];
  const dataP = [3, 2, 4, 1, 5, 2, 3, 4, 6, 2];

  // 黄金参照：本文件直接算
  const muRef = ownMean(dataN);
  const s2Ref = ownVarML(dataN);
  const pRef = ownMean(dataB);
  const lamRef = ownMean(dataP);

  const mN = G.mle(dataN, 'normal');
  const mB = G.mle(dataB, 'bernoulli');
  const mP = G.mle(dataP, 'poisson');
  const eN = Math.max(Math.abs(mN.mu - muRef), Math.abs(mN.sigma2ML - s2Ref) / (Math.abs(s2Ref) + 1));
  const eB = Math.abs(mB.p - pRef);
  const eP = Math.abs(mP.lambda - lamRef);

  // 矩估计 == MLE（对称族一致）
  const momN = G.methodOfMoments(dataN, 'normal');
  const eMom = Math.max(Math.abs(momN.mu - muRef), Math.abs(momN.sigma2ML - s2Ref));

  /* Newton-1：Poisson λ。NLL(λ)=λn − Σx·logλ */
  const nP = dataP.length, sumP = dataP.reduce((s, v) => s + v, 0);
  const nllP = (th) => th[0] * nP - sumP * Math.log(th[0]);
  const gradP = (th) => [nP - sumP / th[0]];
  const hessP = (th) => [[sumP / (th[0] * th[0])]];
  const rP = G.mleNewton(nllP, gradP, hessP, [2.0], { maxIter: 25, tol: 1e-11 });
  const ePn = Math.abs(rP.theta[0] - lamRef);

  /* Newton-2：指数率 λ。NLL(λ)=n logλ − λΣx？取 f(x)=λe^{−λx} ⇒ NLL=n logλ − λΣx，λ̂=1/x̄ */
  const dataE = [0.7, 1.9, 0.4, 2.6, 1.1, 0.3, 3.2, 0.9];
  const nE = dataE.length, sumE = dataE.reduce((s, v) => s + v, 0);
  const nllE = (th) => nE * Math.log(th[0]) - th[0] * sumE;
  const gradE = (th) => [nE / th[0] - sumE];
  const hessE = (th) => [[-nE / (th[0] * th[0])]];
  const rE = G.mleNewton(nllE, gradE, hessE, [0.5], { maxIter: 25, tol: 1e-11 });
  const lamERef = 1 / ownMean(dataE);
  const eEn = Math.abs(rE.theta[0] - lamERef);

  /* Newton-3：正态二维 (μ,σ)。NLL=n logσ + Σ(x−μ)²/(2σ²) */
  const nN = dataN.length, sumN = dataN.reduce((s, v) => s + v, 0);
  const nll2 = (th) => { const mu = th[0], sg = th[1]; let S = 0; for (const x of dataN) S += (x - mu) * (x - mu); return nN * Math.log(sg) + S / (2 * sg * sg); };
  const grad2 = (th) => {
    const mu = th[0], sg = th[1]; let S = 0;
    for (const x of dataN) S += (x - mu) * (x - mu);
    return [(nN * mu - sumN) / (sg * sg), nN / sg - S / (sg * sg * sg)];
  };
  const hess2 = (th) => {
    const mu = th[0], sg = th[1]; let S = 0;
    for (const x of dataN) S += (x - mu) * (x - mu);
    return [[nN / (sg * sg), -2 * (nN * mu - sumN) / (sg * sg * sg)],
    [-2 * (nN * mu - sumN) / (sg * sg * sg), -nN / (sg * sg) + 3 * S / (sg * sg * sg * sg)]];
  };
  const r2 = G.mleNewton(nll2, grad2, hess2, [0.0, 1.0], { maxIter: 25, tol: 1e-11 });
  const e2 = Math.max(Math.abs(r2.theta[0] - muRef), Math.abs(r2.theta[1] - Math.sqrt(s2Ref)));

  const maxScore = Math.max(rP.scoreNorm, rE.scoreNorm, r2.scoreNorm);
  const maxThetaErr = Math.max(ePn, eEn, e2);
  const ok = eN <= 1e-12 && eB <= 1e-12 && eP <= 1e-12 && eMom <= 1e-12 &&
    maxScore <= 1e-9 && maxThetaErr <= 1e-9;
  check('R1', 'MLE 闭式 EXACT(正态/伯努利/泊松)+矩估计一致+Newton(Poisson/指数/正态二维) 得分≤1e-9 且 θ 命中闭式', ok,
    'eN=' + eN.toExponential(2) + ' eBern=' + eB.toExponential(2) + ' ePois=' + eP.toExponential(2) +
    ' eMom=' + eMom.toExponential(2) + ' maxScore=' + maxScore.toExponential(2) + ' maxThetaErr=' + maxThetaErr.toExponential(2) +
    ' θ2D=[' + r2.theta[0].toFixed(9) + ',' + r2.theta[1].toFixed(9) + '] 闭式=[' + muRef.toFixed(9) + ',' + Math.sqrt(s2Ref).toFixed(9) + ']');
}

/* ==========================================================================
 * R2  MAP / 贝叶斯共轭后验均值 EXACT（INV-EST-002）
 * ========================================================================== */
{
  const data = [2.1, 1.8, 2.3, 1.9, 2.0, 2.2, 1.7, 2.4, 2.05, 1.95];
  const n = data.length;
  const xbar = ownMean(data);
  const mu0 = 0.5, tau0 = 2.0, tau = 4.0;

  // 黄金参照 1：解析闭式（本文件直接算）
  const tauPostRef = tau0 + n * tau;
  const muPostRef = (tau0 * mu0 + n * tau * xbar) / tauPostRef;
  // 黄金参照 2：加权平均形式 w·μ₀+(1−w)·x̄（不同代数路径）
  const w = tau0 / (tau0 + n * tau);
  const muPostRef2 = w * mu0 + (1 - w) * xbar;
  // 黄金参照 3：数值最大化后验（黄金分割，完全不依赖共轭公式）
  const logPost = (m) => { let s = 0; for (const x of data) s += (x - m) * (x - m); return -0.5 * tau0 * (m - mu0) * (m - mu0) - 0.5 * tau * s; };
  const muNum = goldenMax(logPost, -20, 20, 200);

  const mapR = G.mapEstimate(data, mu0, tau0, tau);
  const bayR = G.bayesEstimate(data, mu0, tau0, tau);
  const eMap = Math.abs(mapR.mu - muPostRef);
  const eForm2 = Math.abs(muPostRef - muPostRef2);
  const eNum = Math.abs(muNum - muPostRef);
  const eTau = Math.abs(mapR.tauPost - tauPostRef) + Math.abs(mapR.sigma2Post - 1 / tauPostRef);
  const eBayes = Math.abs(bayR.muPost - muPostRef) + Math.abs(bayR.tauPost - tauPostRef);
  const eModeMean = Math.abs(bayR.muPost - mapR.mu);

  // 平坦先验 τ₀→0：MAP 退化为样本均值
  const mapFlat = G.mapEstimate(data, 0, 1e-9, tau);
  const eFlat = Math.abs(mapFlat.mu - xbar);

  const ok = eMap <= 1e-12 && eForm2 <= 1e-12 && eNum <= 1e-8 && eTau <= 1e-12 &&
    eBayes <= 1e-12 && eModeMean <= 1e-12 && eFlat <= 1e-9;
  check('R2', 'MAP/贝叶斯后验均值==独立解析式(三路:闭式/加权平均/黄金分割数值最大化)；平坦先验退化为 x̄；众数==均值', ok,
    'map=' + mapR.mu.toFixed(12) + ' 解析=' + muPostRef.toFixed(12) + ' 加权形式Δ=' + eForm2.toExponential(2) +
    ' 数值最大化Δ=' + eNum.toExponential(2) + ' 众数=均值Δ=' + eModeMean.toExponential(2) + ' 平坦MAP−x̄=' + eFlat.toExponential(2));
}

/* ==========================================================================
 * R3  KS D 手算 EXACT + p 单调边界 + 功效双向（INV-EST-003）
 * ========================================================================== */
{
  const data = [-1.5, -0.8, -0.3, 0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 2.0, 2.4];
  const cdfMod = G.normalCdfFactory(0, 1);
  const cdfOwn = (x) => ownNormalCdf(x);          // 本文件 Simpson 积分 CDF

  const ks = G.ksTest(data, cdfMod);
  const handMod = ownKsStat(data, cdfMod);        // 同 CDF、独立扫描 → 应 EXACT
  const handOwn = ownKsStat(data, cdfOwn);        // 独立 CDF → 应 ≤1e-9
  const eD = Math.abs(ks.D - handMod.D);
  const eDp = Math.abs(ks.Dplus - handMod.Dplus) + Math.abs(ks.Dminus - handMod.Dminus);
  const eDindep = Math.abs(ks.D - handOwn.D);

  // 模块 normalCdf vs 本文件 Simpson 正态 CDF（独立路径对拍）
  let eCdf = 0;
  for (const x of [-3.2, -1.5, -0.4, 0.0, 0.35, 1.1, 2.7]) eCdf = Math.max(eCdf, Math.abs(G.normalCdf(x) - ownNormalCdf(x)));

  // p 值：D=0→1，单调不增，D 极大→0
  let mono = true, prev = Infinity;
  for (let d = 0; d <= 1.5; d += 0.005) {
    const p = G.ksPValue(d, 11);
    if (p > prev + 1e-15) mono = false;
    prev = p;
  }
  const okBound = G.ksPValue(0, 11) === 1 && G.ksPValue(0.2, 11) > G.ksPValue(0.5, 11) &&
    G.ksPValue(0.5, 11) > G.ksPValue(1.0, 11) && G.ksPValue(5.0, 11) < 1e-12;

  // 功效 / 伪功效（双向）：本文件 mulberry32 抽样
  const rng = mulberry32(0x5A001);
  const M = 500, nS = 30;
  let rejShift = 0, rejNull = 0;
  for (let t = 0; t < M; t++) {
    const s1 = ownGaussVec(rng, nS).map((v) => v + 3);      // N(3,1)
    if (G.ksTest(s1, cdfMod).p < 0.05) rejShift++;
    const s0 = ownGaussVec(rng, nS);                        // N(0,1)
    if (G.ksTest(s0, cdfMod).p < 0.05) rejNull++;
  }
  const rateShift = rejShift / M, rateNull = rejNull / M;
  const ok = eD <= 1e-15 && eDp <= 1e-15 && eDindep <= 1e-9 && eCdf <= 1e-9 &&
    mono && okBound && rateShift > 0.9 && rateNull < 0.15;
  check('R3', 'KS D/D₊/D₋ 手算 EXACT + 独立 Simpson 正态 CDF 双轨 + p 单调边界 + 功效>0.9/伪功效<0.15', ok,
    'D=' + ks.D.toFixed(9) + ' 手算Δ=' + eD.toExponential(2) + ' 独立CDFΔD=' + eDindep.toExponential(2) +
    ' CDF对拍=' + eCdf.toExponential(2) + ' mono=' + mono + ' p(0)=' + G.ksPValue(0, 11) +
    ' p(5)=' + G.ksPValue(5.0, 11).toExponential(2) + ' 拒绝率[N(3,1)]=' + rateShift.toFixed(3) + ' [N(0,1)]=' + rateNull.toFixed(3));
}

/* ==========================================================================
 * R4  χ² 检验：统计手算 EXACT + 2×2 解析 + 独立 χ² CDF + df + 功效（INV-EST-004）
 * ========================================================================== */
{
  const obs = [18, 22, 15, 20, 25, 17, 19, 24];
  const exp = obs.map(() => 1 / 8);
  const cs = G.chiSquareTest(obs, exp, 0);

  // 黄金参照：本文件手算 Σ(O−E)²/E
  const N = obs.reduce((s, v) => s + v, 0);
  let statHand = 0;
  for (const o of obs) statHand += (o - N / 8) * (o - N / 8) / (N / 8);
  const eStat = Math.abs(cs.stat - statHand);
  const eDf = Math.abs(cs.df - (obs.length - 1));
  const cs2 = G.chiSquareTest(obs, exp, 2);
  const eDf2 = Math.abs(cs2.df - (obs.length - 1 - 2));

  // 2×2 独立性：E_ij = row·col/N 自算
  const tab = [[40, 10], [15, 35]];
  const [[a, b], [c, d]] = tab;
  const N2 = a + b + c + d;
  const r1 = a + b, r2 = c + d, c1 = a + c, c2 = b + d;
  const E11 = r1 * c1 / N2, E12 = r1 * c2 / N2, E21 = r2 * c1 / N2, E22 = r2 * c2 / N2;
  const statIndHand = (a - E11) * (a - E11) / E11 + (b - E12) * (b - E12) / E12 +
    (c - E21) * (c - E21) / E21 + (d - E22) * (d - E22) / E22;
  const ci = G.chiSquareIndependence(tab);
  const eInd = Math.abs(ci.stat - statIndHand);
  const eIndRef = Math.abs(ci.stat - 25.252525252525253);

  // p 值 vs 本文件独立 Simpson χ² CDF
  const eP1 = Math.abs(G.chiSquareSf(cs.stat, cs.df) - ownChiSquareSf(cs.stat, cs.df));
  const eP2 = Math.abs(ci.p - ownChiSquareSf(ci.stat, 1));
  const eCdfSelf = Math.abs(G.chiSquareCdf(cs.stat, cs.df) + G.chiSquareSf(cs.stat, cs.df) - 1);

  // 功效：真分布明显非均匀（偏斜），拒绝率 >0.9
  const rng = mulberry32(0x4C5A002);
  const skew = [0.25, 0.20, 0.15, 0.10, 0.10, 0.10, 0.05, 0.05];
  let rej = 0; const M = 500;
  for (let t = 0; t < M; t++) {
    const o = ownMultinomial(N, skew, rng);
    if (G.chiSquareTest(o, exp, 0).p < 0.05) rej++;
  }
  const power = rej / M;

  const ok = eStat <= 1e-12 && eDf === 0 && eDf2 === 0 && eInd <= 1e-12 && eIndRef <= 1e-3 &&
    eP1 <= 1e-9 && eP2 <= 1e-9 && eCdfSelf <= 1e-12 && power > 0.9;
  check('R4', 'χ² 拟合优度手算 EXACT + df 公式 + 2×2 独立性解析 25.2525 + 独立 Simpson χ² CDF + 功效>0.9', ok,
    'stat=' + cs.stat.toFixed(6) + ' 手算=' + statHand.toFixed(6) + ' Δ=' + eStat.toExponential(2) +
    ' ind=' + ci.stat.toFixed(9) + ' 自算E=' + statIndHand.toFixed(9) + ' df=' + ci.df +
    ' pΔ(χ²CDF)=' + eP1.toExponential(2) + '/' + eP2.toExponential(2) + ' power=' + power.toFixed(3));
}

/* ==========================================================================
 * R5  Bootstrap：SE 与解析 SE/√n ≤15% + CI 中心 ≈ x̄ + 均值一致（INV-EST-005）
 * ========================================================================== */
{
  const n = 40, B = 4000;
  const stat = (xs) => ownMean(xs);
  let worstRel = 0, worstCenter = 0, worstMean = 0, worstCross = 0;
  for (const sd of [0xB50101, 0xB50102, 0xB50103]) {
    const rng = mulberry32(sd);
    const data = ownGaussVec(rng, n).map((v) => v * 2 + 5);      // N(5, 4)
    const xb = ownMean(data);
    const seAna = ownStdSample(data) / Math.sqrt(n);             // 本文件 std

    const boot = G.bootstrap(data, stat, B, mulberry32(sd ^ 0xBEEF));
    worstRel = Math.max(worstRel, Math.abs(boot.se - seAna) / seAna);

    // 独立重采样实现（不同 rng 轨道）交叉对拍
    const ownBoots = ownBootstrap(data, stat, B, mulberry32(sd ^ 0x1234));
    const ownSe = ownStdSample(ownBoots);
    worstCross = Math.max(worstCross, Math.abs(boot.se - ownSe) / seAna);

    const ci = G.percentileCI(boot.boots, 0.05);
    const center = 0.5 * (ci[0] + ci[1]);
    worstCenter = Math.max(worstCenter, Math.abs(center - xb) / seAna);
    worstMean = Math.max(worstMean, Math.abs(boot.mean - xb) / (seAna / Math.sqrt(B)));
  }
  const ok = worstRel <= 0.15 && worstCross <= 0.10 && worstCenter <= 2 && worstMean <= 6;
  check('R5', 'Bootstrap SE 与解析 SE/√n 误差≤15% + 独立重采样交叉(≤10%) + 百分位CI中心≈x̄ + 均值一致', ok,
    'worstRelErr=' + worstRel.toFixed(4) + ' 独立重采样交叉Δ=' + worstCross.toFixed(4) +
    ' |CI中心−x̄|/SE=' + worstCenter.toFixed(3) + ' |boot均值−x̄|/(SE/√B)=' + worstMean.toFixed(3) + ' B=' + B);
}

/* ==========================================================================
 * R6  置信区间覆盖 ∈[0.92,0.98]（t / 正态 / Wilson）+ 解析对拍（INV-EST-006）
 * ========================================================================== */
{
  const NREP = 2000, n = 50, muTrue = 3, sdTrue = 2;
  const rng = mulberry32(0xC1A60001);
  let covT = 0, covN = 0, covW = 0;
  for (let t = 0; t < NREP; t++) {
    const samp = ownGaussVec(rng, n).map((v) => v * sdTrue + muTrue);
    const ciT = G.confidenceInterval(samp, { type: 't', alpha: 0.05 });
    if (ciT[0] <= muTrue && muTrue <= ciT[1]) covT++;
    const ciN = G.confidenceInterval(samp, { type: 'normal', alpha: 0.05 });
    if (ciN[0] <= muTrue && muTrue <= ciN[1]) covN++;
    // Wilson：真 Binomial(100, 0.5)——100 次伯努利求和，绝不用均匀抽 k
    let k = 0;
    for (let j = 0; j < 100; j++) if (rng() < 0.5) k++;
    const ciW = G.confidenceInterval([], { type: 'proportion', x: k, n: 100, alpha: 0.05 });
    if (ciW[0] <= 0.5 && 0.5 <= ciW[1]) covW++;
  }
  const cT = covT / NREP, cN = covN / NREP, cW = covW / NREP;

  // 解析公式对拍：本文件 std + 本文件 Simpson 导出的 t/z 分位数
  const tCrit = ownTQuantile(0.975, n - 1);
  const zCrit = ownNormInv(0.975);
  let eT = 0, eZ = 0;
  const rng2 = mulberry32(0xC1A60002);
  for (let t = 0; t < 5; t++) {
    const samp = ownGaussVec(rng2, n).map((v) => v * sdTrue + muTrue);
    const m = ownMean(samp), se = ownStdSample(samp) / Math.sqrt(n);
    const mt = G.confidenceInterval(samp, { type: 't', alpha: 0.05 });
    eT = Math.max(eT, Math.abs(mt[0] - (m - tCrit * se)), Math.abs(mt[1] - (m + tCrit * se)));
    const mz = G.confidenceInterval(samp, { type: 'normal', alpha: 0.05 });
    eZ = Math.max(eZ, Math.abs(mz[0] - (m - zCrit * se)), Math.abs(mz[1] - (m + zCrit * se)));
  }

  // Wilson 上下界手算 EXACT（两例）
  const wilsonHand = (x, nn) => {
    const ph = x / nn;
    const den = 1 + zCrit * zCrit / nn;
    const cen = (ph + zCrit * zCrit / (2 * nn)) / den;
    const half = zCrit * Math.sqrt(ph * (1 - ph) / nn + zCrit * zCrit / (4 * nn * nn)) / den;
    return [cen - half, cen + half];
  };
  let eW = 0;
  for (const [x, nn] of [[50, 100], [20, 100], [7, 60]]) {
    const hw = wilsonHand(x, nn);
    const mw = G.confidenceInterval([], { type: 'proportion', x, n: nn, alpha: 0.05 });
    eW = Math.max(eW, Math.abs(mw[0] - hw[0]), Math.abs(mw[1] - hw[1]));
  }

  // 证伪「过窄」：α=0.5 的 50% 区间覆盖必 <0.92（证明覆盖判据非空转）
  const rng3 = mulberry32(0xC1A60003);
  let covNarrow = 0; const M = 500;
  for (let t = 0; t < M; t++) {
    const samp = ownGaussVec(rng3, n).map((v) => v * sdTrue + muTrue);
    const ci = G.confidenceInterval(samp, { type: 't', alpha: 0.5 });
    if (ci[0] <= muTrue && muTrue <= ci[1]) covNarrow++;
  }
  const cNarrow = covNarrow / M;

  const ok = cT >= 0.92 && cT <= 0.98 && cN >= 0.92 && cN <= 0.98 && cW >= 0.92 && cW <= 0.98 &&
    eT <= 1e-6 && eZ <= 1e-6 && eW <= 1e-9 && cNarrow < 0.92;
  check('R6', 't/正态/Wilson 95% CI 覆盖∈[0.92,0.98](2000次,Wilson 用 Binomial(100,0.5)) + 解析公式对拍 + Wilson 边界 EXACT', ok,
    'covT=' + cT.toFixed(4) + ' covZ=' + cN.toFixed(4) + ' covWilson=' + cW.toFixed(4) +
    ' 解析Δt=' + eT.toExponential(2) + ' Δz=' + eZ.toExponential(2) + ' Δwilson=' + eW.toExponential(2) +
    ' 证伪窄区间(α=0.5)覆盖=' + cNarrow.toFixed(3));
}

/* ==========================================================================
 * R7  Welch t 手算 EXACT + 独立 Simpson Student-t CDF + 错误率/功效（INV-EST-007）
 * ========================================================================== */
{
  const a = [1, 2, 3, 4, 5], b = [2, 4, 6, 8, 10];
  const ht = G.hypothesisTest(a, b);
  // 黄金参照：本文件手算
  const n1 = a.length, n2 = b.length;
  const m1 = ownMean(a), m2 = ownMean(b);
  const v1 = ownVarSample(a), v2 = ownVarSample(b);
  const seHand = Math.sqrt(v1 / n1 + v2 / n2);
  const tHand = (m1 - m2) / seHand;
  const dfHand = Math.pow(v1 / n1 + v2 / n2, 2) /
    (Math.pow(v1 / n1, 2) / (n1 - 1) + Math.pow(v2 / n2, 2) / (n2 - 1));
  const eT = Math.abs(ht.t - tHand);
  const eDf = Math.abs(ht.df - dfHand);

  // 独立 Simpson t CDF vs 模块 studentTCdf
  let eCdf = 0, symErr = 0;
  const grid = [[-2.5, 7], [-0.4, 3], [0, 1], [0, 3], [0, 30], [1.5, 10], [-1.5, 10], [3.2, 5], [0.7, 60], [5.0, 2], [8.0, 12]];
  for (const [t, df] of grid) eCdf = Math.max(eCdf, Math.abs(G.studentTCdf(t, df) - ownTCdf(t, df)));
  for (const df of [1, 2, 3, 5, 10, 30, 49, 100]) symErr = Math.max(symErr, Math.abs(G.studentTCdf(0, df) - 0.5), Math.abs(ownTCdf(0, df) - 0.5));

  // I 类错误（H₀：两总体同分布 N(0,1)，α=0.05，2000 次）
  const rng = mulberry32(0x7A5C0001);
  let rej = 0; const NREP = 2000;
  for (let t = 0; t < NREP; t++) {
    const g1 = ownGaussVec(rng, 40), g2 = ownGaussVec(rng, 40);
    if (G.hypothesisTest(g1, g2).p < 0.05) rej++;
  }
  const iErr = rej / NREP;

  // 功效：d=0.8，n1=n2=50
  const rng2 = mulberry32(0x7A5C0002);
  let rejP = 0;
  for (let t = 0; t < NREP; t++) {
    const g1 = ownGaussVec(rng2, 50), g2 = ownGaussVec(rng2, 50).map((v) => v + 0.8);
    if (G.hypothesisTest(g1, g2).p < 0.05) rejP++;
  }
  const power = rejP / NREP;

  // 证伪：用自有 t CDF 但把显著性水平误设为 α=0.20，拒绝率应落到 [0.03,0.07] 之外
  const rng3 = mulberry32(0x7A5C0003);
  let rejWrong = 0; const M = 500;
  for (let t = 0; t < M; t++) {
    const g1 = ownGaussVec(rng3, 40), g2 = ownGaussVec(rng3, 40);
    const r = G.hypothesisTest(g1, g2);
    const pOwn = 2 * (1 - ownTCdf(Math.abs(r.t), r.df));
    if (pOwn < 0.20) rejWrong++;
  }
  const wrongRate = rejWrong / M;

  const ok = eT <= 1e-12 && eDf <= 1e-12 && eCdf <= 1e-9 && symErr <= 1e-15 &&
    iErr >= 0.03 && iErr <= 0.07 && power > 0.8 && wrongRate > 0.07;
  check('R7', 'Welch t 手算 EXACT + 独立 Simpson Student-t CDF(≤1e-9) + CDF(0,ν)=0.5 + I类错误∈[0.03,0.07] + d=0.8 功效>0.8', ok,
    't=' + ht.t.toFixed(9) + ' 手算=' + tHand.toFixed(9) + ' Δ=' + eT.toExponential(2) + ' df=' + ht.df.toFixed(4) +
    ' tCDFΔ=' + eCdf.toExponential(2) + ' 对称Δ=' + symErr.toExponential(2) +
    ' I类错误=' + iErr.toFixed(4) + ' 功效=' + power.toFixed(4) + ' 证伪(α=0.20)拒绝率=' + wrongRate.toFixed(3));
}

/* ==========================================================================
 * R8  Wilks LRT(df=2) + 拟似然 + Huber 降权 + 经验似然极大点（INV-EST-008）
 * ========================================================================== */
{
  /* --- (a) LRT：全模型同时解放 μ 与 σ ⇒ df=2，H₀ 下中位 p ∈[0.4,0.6] --- */
  const rng = mulberry32(0x81C00001);
  const NREP = 2000, n = 60;
  const ps2 = [], ps1 = [];
  let eStatClosed = 0, ePClosed = 0;
  for (let t = 0; t < NREP; t++) {
    const d = ownGaussVec(rng, n);                 // H₀: N(0,1)
    const llNull = G.normalLogLik(d, 0, 1);
    // Wilks 要求全模型取 MLE：μ̂=x̄, σ̂_ML=sqrt(1/n·Σ(x−x̄)²)（而非无偏样本 sd）
    const mh = ownMean(d), shML = Math.sqrt(ownVarML(d));
    const llFull = G.normalLogLik(d, mh, shML);
    // 独立闭式（不同代数路径）：2(ℓ_full−ℓ_null) = −2n·log σ̂_ML − n + Σx²
    let sx2 = 0; for (const x of d) sx2 += x * x;
    const statClosed = -2 * n * Math.log(shML) - n + sx2;
    const statMod = 2 * (llFull - llNull);
    if (t < 200) eStatClosed = Math.max(eStatClosed, Math.abs(statMod - statClosed) / Math.max(1, Math.abs(statClosed)));
    // χ²(2) 生存函数闭式 = e^{−stat/2}
    ePClosed = Math.max(ePClosed, Math.abs(G.chiSquareSf(statMod, 2) - Math.exp(-statMod / 2)));
    ps2.push(G.likelihoodRatioTest(llFull, llNull, 2).p);
    ps1.push(G.likelihoodRatioTest(llFull, llNull, 1).p);   // 错误 df 证伪
  }
  ps2.sort((x, y) => x - y); ps1.sort((x, y) => x - y);
  const medP2 = ps2[Math.floor(ps2.length / 2)];
  const medP1 = ps1[Math.floor(ps1.length / 2)];

  /* --- (b) 拟似然：Poisson 数据退化为 MLE，φ≈1 --- */
  const rng2 = mulberry32(0x81C00002);
  const dp = [];
  for (let i = 0; i < 300; i++) dp.push(ownPoisson(rng2, 2.9));
  const ql = G.quasiLikelihood(dp, 'poisson');
  const eQlMu = Math.abs(ql.mu - ownMean(dp));
  const eQlPhi = Math.abs(ql.phi - 1);

  /* --- (c) Huber 降权 + 独立二分对拍 --- */
  let worstHub = 0, eHub = 0;
  for (const sd of [0x81C00301, 0x81C00302, 0x81C00303]) {
    const rng3 = mulberry32(sd);
    const clean = ownGaussVec(rng3, 47);                       // N(0,1)，真值 0
    const cont = clean.concat([20, -18, 22]);                  // 污染，真值仍 0
    const hub = G.robustEstimate(cont, { c: 1.345 });
    const sm = ownMean(cont);
    worstHub = Math.max(worstHub, Math.abs(hub.mu - 0) / Math.abs(sm - 0));   // 需 <1
    eHub = Math.max(eHub, Math.abs(hub.mu - ownHuberMu(cont, hub.scale, hub.c)));
  }

  /* --- (d) 经验似然极大点在样本均值：−2logR ≈ 0（独立对偶解对拍） --- */
  const rng4 = mulberry32(0x81C00004);
  const de = ownGaussVec(rng4, 30).map((v) => v + 1);
  const muHat = ownMean(de);
  const elMod = G.empiricalLikelihood(de, muHat);
  const elOwn = ownELMinus2LogR(de, muHat);
  const eEL = Math.abs(elMod.minus2LogR - elOwn.minus2LogR);
  // 离均值点：模块 −2logR 须与独立对偶解一致，且严格为正（极大点唯一性，非恒零退化解）
  let eELOff = 0, elOffMin = Infinity, offPos = true;
  for (const off of [-0.6, -0.3, 0.3, 0.6]) {
    const mo = G.empiricalLikelihood(de, muHat + off);
    const oo = ownELMinus2LogR(de, muHat + off);
    eELOff = Math.max(eELOff, Math.abs(mo.minus2LogR - oo.minus2LogR));
    elOffMin = Math.min(elOffMin, mo.minus2LogR);
    if (!(mo.minus2LogR > 0)) offPos = false;
  }
  // 网格上 x̄ 为严格极小点
  const gridMinOk = elOffMin > elMod.minus2LogR + 1e-6;

  const ok = medP2 >= 0.4 && medP2 <= 0.6 && medP1 < 0.35 && eStatClosed <= 1e-9 && ePClosed <= 1e-12 &&
    eQlMu <= 1e-12 && eQlPhi <= 0.25 && worstHub < 1 && eHub <= 1e-8 &&
    Math.abs(elMod.minus2LogR) <= 1e-9 && eEL <= 1e-9 && eELOff <= 1e-6 && offPos && gridMinOk;
  check('R8', 'Wilks LRT(df=2) 中位p∈[0.4,0.6] + 闭式 stat/χ²(2) 对拍 + 拟似然Poisson退化(φ≈1) + Huber降权(独立二分) + EL 极大点在 x̄ 且离均值点与独立对偶解一致', ok,
    'medP(df=2)=' + medP2.toFixed(3) + ' medP(df=1)=' + medP1.toFixed(3) + ' stat闭式Δ=' + eStatClosed.toExponential(2) +
    ' χ²(2)闭式Δ=' + ePClosed.toExponential(2) + ' QL: μΔ=' + eQlMu.toExponential(2) + ' |φ−1|=' + eQlPhi.toFixed(4) +
    ' |Huber|/|均值|=' + worstHub.toFixed(4) + ' Huber独立二分Δ=' + eHub.toExponential(2) +
    ' EL: −2logR@x̄=' + elMod.minus2LogR.toExponential(2) + '(独立对偶Δ=' + eEL.toExponential(2) + ')' +
    ' 离均值点Δ=' + eELOff.toExponential(2) + ' min(−2logR离均值)=' + elOffMin.toFixed(4));
}

/* ==========================================================================
 * 审计附注（不计入 R1..R8 判定，仅为交叉验证与缺陷追踪提供证据）
 *   N1 经验似然离均值点：模块 −2logR 与独立对偶解逐点对照，并与渐近 χ²(1) 比对。
 *      （审计起始时本模块曾把对偶方程写成规范化式 Σ1/(1+λd)=n，该式在任意 μ₀ 处
 *       都被 λ=0 平凡满足，导致 −2logR 恒为 0；该缺陷已由模块方在审计过程中修复，
 *       现版本与独立对偶解一致，见下表。）
 *   N2 Wilks LRT 全模型 σ 的取法：无偏样本 sd(n−1) vs MLE sd(n)。
 * ========================================================================== */
{
  // ---- N1：经验似然离均值点逐点对照 ----
  const rng = mulberry32(0x0A0D1701);
  const de = ownGaussVec(rng, 30).map((v) => v + 1);
  const muHat = ownMean(de);
  const s2 = ownVarSample(de);
  console.log('  ..注N1 empiricalLikelihood：模块 vs 独立对偶解 vs 渐近 χ²(1)');
  for (const off of [-0.6, -0.3, 0.0, 0.3, 0.6]) {
    const mo = G.empiricalLikelihood(de, muHat + off);
    const oo = ownELMinus2LogR(de, muHat + off);
    const asym = de.length * off * off / s2;                 // 渐近 χ²(1)
    console.log('       μ₀−x̄=' + off.toFixed(1).padStart(4) +
      '  模块[λ=' + mo.lam.toFixed(6).padStart(10) + ' −2logR=' + mo.minus2LogR.toFixed(6).padStart(10) + ']' +
      '  独立对偶[λ=' + oo.lam.toFixed(6).padStart(10) + ' −2logR=' + oo.minus2LogR.toFixed(6).padStart(10) + ']' +
      '  |Δ|=' + Math.abs(mo.minus2LogR - oo.minus2LogR).toExponential(2) +
      '  渐近χ²(1)=' + asym.toFixed(3));
  }

  // ---- N2：LRT 全模型 σ 取法 ----
  const rng2 = mulberry32(0x0A0D1702);
  let negSd = 0, negMl = 0; const M = 4000, nn = 60;
  for (let t = 0; t < M; t++) {
    const d = ownGaussVec(rng2, nn);
    const llNull = G.normalLogLik(d, 0, 1);
    const mh = ownMean(d);
    if (2 * (G.normalLogLik(d, mh, ownStdSample(d)) - llNull) < 0) negSd++;   // 无偏 sd（selfCheck 口径）
    if (2 * (G.normalLogLik(d, mh, Math.sqrt(ownVarML(d))) - llNull) < 0) negMl++; // MLE sd（Wilks 口径）
  }
  console.log('  ..注N2 Wilks LRT 全模型 σ 口径: 用无偏 sd(n−1) 时 2Δℓ<0 的 H₀ 比例=' +
    (negSd / M).toFixed(4) + '；用 MLE sd(n) 时=' + (negMl / M).toFixed(4) +
    '（本复核采用后者，故 stat 恒 ≥0 且 p=e^{−stat/2} 与 χ²(2) 闭式 EXACT 吻合）');
}

/* ---------- 汇总 ---------- */
const failed = results.filter((r) => !r.ok).length;
console.log('══════ estimate-forge 独立复核: ' + (results.length - failed) + '/' + results.length + ' 通过 ══════');
process.exit(failed === 0 ? 0 : 1);
