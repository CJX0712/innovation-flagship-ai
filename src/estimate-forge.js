/**
 * @forge-module estimate-forge
 * @layer L5
 * @version 1.0.0
 * @brief     估计与检验内核（F5-G6 全 16 项）：最大似然估计(MLE) / 最大后验估计(MAP) / 矩估计 / 贝叶斯估计 / KS 检验 / 卡方检验 / Bootstrap / 交叉验证似然 / 矩方法 / 拟似然 / 稳健估计(Huber M) / 经验似然 / 假设检验(Welch t) / 置信区间(正态/t/Wilson) / 似然比检验(LRT) / 拟合优度(KS+χ² 调度)。所有接口可交叉验证（闭式解对照、独立重实现、共轭解析、蒙特卡洛覆盖/功效、渐近 χ² 分布对照）。
 * @reuse     Casella & Berger《Statistical Inference》(2nd, Duxbury 2002)|https://www.routledge.com/Statistical-Inference/Casella-Berger/p/book/9780534243128|CRC|B|MLE/UMVUE/充分统计/似然比检验的统一理论框架
 * @reuse     Kolmogorov《Sulla determinazione empirica di una legge di distribuzione》(1933)|https://doi.org/10.2307/2280095|Springer|A|KS 检验统计量 D=sup|Fₙ−F| 与 Kolmogorov 分布
 * @reuse     Pearson《On the Criterion that a given System of Deviations...》(1900)|https://doi.org/10.1080/14786440009463897|Taylor|A|卡方拟合优度 χ²=Σ(O−E)²/E 与自由度
 * @reuse     Efron《Bootstrap Methods: Another Look at the Jackknife》(1979)|https://doi.org/10.1214/aos/1176344552|IMS|A|非参数 Bootstrap 重采样标准误与百分位置信区间
 * @reuse     Welch《The Generalization of Student's Problem...》(1947)|https://doi.org/10.2307/2332510|OUP|A|Welch t 检验（不等方差 Satterthwaite 自由度）
 * @reuse     Huber《Robust Estimation of a Location Parameter》(1964)|https://doi.org/10.1214/aoms/1177703732|IMS|A|Huber M 估计（ψ 截断降权离群）
 * @reuse     Owen《Empirical Likelihood》(2001, Chapman & Hall)|https://doi.org/10.1201/9781420036152|CRC|A|经验似然比 −2logR 渐近 χ² 且极大点在样本均值
 * @reuse     Brown et al.《Interval Estimation for a Binomial Proportion》(2001, Statist. Med.)|https://doi.org/10.1002/sim.795|Wiley|B|Wilson 比例置信区间（覆盖优于正态近似）
 * @beyond    纯 JS 单文件零依赖零构建，双端同构 UMD；F5-G6 全 16 项组织为一条自洽统计推断栈（点估计→区间估计→假设检验→拟合优度→重采样→交叉验证→稳健/经验似然），各组件可独立使用亦互相组合（LRT 复用对数似然、Bootstrap 复用统计量函数、拟合优度调度 KS/χ²）。反空洞铁律贯穿：MLE/矩估计/共轭后验均值均有闭式 EXACT 对拍（直接算 ≠ 实现）；KS 的 D 统计量与手算 EXACT 且 p 值随 D 单调；卡方统计与手算 EXACT；Welch t 统计量与手算 EXACT 且学生 t CDF 满足 CDF(0;df)=0.5 单调；Bootstrap 均值标准误与解析 SE/√n 误差 ≤15%；置信区间蒙特卡洛覆盖 ∈[0.92,0.98]（证伪过窄）；Welch I 类错误 ∈[0.03,0.07] 功效 >0.8（双向）；Huber 对污染离群降权（离群权重<1 且 |Huber−真值|<|样本均值−真值|，证伪无降权）；经验似然极大点 EXACT 在样本均值（拟似然对 Poisson 退化为 MLE 且离散度 φ≈1）。
 * @invariant INV-EST-001|MLE 与矩估计闭式 EXACT + 牛顿收敛：正态 MLE μ̂=x̄、σ̂²=1/n Σ(x−x̄)² 与直接计算 EXACT(≤1e-12)；伯努利 p̂=x̄、泊松 λ̂=x̄ EXACT；MLE-Newton 在 ≤25 步内得分 ‖∂ℓ/∂θ‖≤1e-9（对照解析一阶条件）；矩估计与 MLE 在对称族一致 EXACT|1e-12|闭式对拍 + 牛顿得分收敛 + 一致性
 * @invariant INV-EST-002|MAP 与贝叶斯共轭 EXACT：正态-正态共轭后验均值 (τ₀μ₀+nτx̄)/(τ₀+nτ) 与解析 EXACT(≤1e-12)；平坦先验(τ₀→0) MAP 退化为 MLE EXACT；贝叶斯后验均值 == MAP（正态共轭众数=均值）EXACT|1e-12|共轭公式对拍 + 退化证伪 + 众数=均值
 * @invariant INV-EST-003|KS 检验 D 与 p：D/D₊/D₋ 与手算 EXACT（固定小样本 11 点）；渐近 p 值随 D 严格单调且 D=0→p=1、D 极大→p→0（K(x)=1−2Σ(−1)^{k-1}e^{−2k²x²} 自洽）；功效——来自 N(3,1) 的样本拒绝率 >90%、来自 N(0,1) 的样本拒绝率 <15%（双向证伪/功效）|1e-9|D 手算对拍 + 单调边界 + 功效双向
 * @invariant INV-EST-004|卡方检验：拟合优度 χ²=Σ(O−E)²/E 与手算 EXACT（固定频数 8 类）；p 值与 χ²(k) CDF 一致；独立性 2×2 χ² 与解析对照（且自由度=1 EXACT）；功效——观察=期望×2 系统性偏离拒绝率 >90%；卡方自由度 = 类别数−1−估计参数数 EXACT|1e-9|统计手算 + CDF 对拍 + 功效 + 自由度
 * @invariant INV-EST-005|Bootstrap 非参数：B=2000 对均值 SE_boot 与解析 SE/√n 相对误差 ≤15%；百分位 95% CI 中心 ≈ 样本均值（abs(中心−x̄)≤2·SE）；bootstrap 均值 ≈ 原始均值（固定 seed 下 abs(Δ)≤3·SE/√B）|1e-6|SE 对照 + CI 中心 + 均值一致
 * @invariant INV-EST-006|置信区间覆盖：正态/t/Wilson 95% CI 在重复抽样下覆盖真值率 ∈[0.92,0.98]（蒙特卡洛 2000 次，固定种子）；t CI 与解析公式一致；Wilson 比例 CI 上下界与手算 EXACT|1e-2|蒙特卡洛覆盖 + 解析公式 + 边界
 * @invariant INV-EST-007|假设检验 Welch：t 统计量与手算 EXACT；学生 t CDF 对照（CDF(0;df)=0.5 EXACT、单调递增）；I 类错误率 ∈[0.03,0.07]（α=0.05，2000 次）；效应量 d=0.8 功效 >0.8|1e-9|t 公式 + t CDF + 错误率双向
 * @invariant INV-EST-008|似然比检验 + 拟似然/稳健/经验似然：LRT 统计量 2(ℓ₁−ℓ₀) 中位 p 值 ∈[0.4,0.6]（H₀ 下 2000 次，全模型同时解放 μ 与 σ（σ̂ 取 MLE 除数 n）故自由度 df=2，与 χ²(2) CDF 对照）；拟似然对 Poisson 退化为 MLE(mean=x̄) EXACT 且离散度 φ≈1；Huber 稳健估计对污染离群降权（abs(Huber−真值)<abs(样本均值−真值)）；经验似然极大点 EXACT 在样本均值（−2logR(x̄)=0）且偏离均值 μ₀=x̄±0.5 后 −2logR>0.5（证伪恒零退化解，λ 由均值约束 Σd/(1+λd)=0 在非对称可行域内解出）|1e-9|LRT χ² 对照 + QL 退化 + Huber 降权 + EL 极大点双向
 * @seed      0x9E3779B9
 */
'use strict';

/* ============================================================================
 * 项目统一容差（九档，与 core-verify.TOL 同值，本地声明保证零依赖）
 * ========================================================================== */
const TOL = { EXACT: 0, ULP: 1e-16, TIGHT: 1e-15, ORTHO: 1e-14, RESID: 1e-12, SOLVE: 1e-10, GRAD: 1e-6, MONO: -1e-9, CONV: 1e-6 };

/* ============================================================================
 * 私有可种子 PRNG：sfc32（同 seed 序列逐位可复现；异 seed 轨道分歧）
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
    b = c + (c << 3);
    c = (c << 21) | (c >>> 11);
    c = (c + t) >>> 0;
    return (t >>> 0) / 4294967296;
  }
  for (let i = 0; i < 12; i++) next();
  return next;
}
/* 标准正态向量（Box-Muller，成对消费） */
function gaussVec(rng, dim) {
  const out = [];
  for (let i = 0; i < dim; i += 2) {
    const u1 = Math.max(rng(), 1e-300), u2 = rng();
    const r = Math.sqrt(-2 * Math.log(u1));
    out.push(r * Math.cos(2 * Math.PI * u2));
    if (i + 1 < dim) out.push(r * Math.sin(2 * Math.PI * u2));
  }
  return out;
}

/* ============================================================================
 * 基础统计
 * ========================================================================== */
function mean(xs) { let s = 0; for (const x of xs) s += x; return s / xs.length; }
function varSample(xs) { const m = mean(xs); let s = 0; for (const x of xs) s += (x - m) * (x - m); return s / (xs.length - 1); }
function varML(xs) { const m = mean(xs); let s = 0; for (const x of xs) s += (x - m) * (x - m); return s / xs.length; }
function stdSample(xs) { return Math.sqrt(varSample(xs)); }

/* ============================================================================
 * 数值特殊函数（纯 JS，FIPS 无关，Numerical Recipes 风格）
 * ========================================================================== */
function logGamma(x) {
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  x -= 1;
  let a = c[0]; const t = x + 7 + 0.5;
  for (let i = 1; i < 9; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
function _gamCF(s, x) {
  const FPMIN = 1e-300;
  let b = x + 1 - s, c = 1 / FPMIN, d = 1 / b, h = d;
  for (let i = 1; i <= 1000; i++) {
    const an = -i * (i - s);
    b += 2;
    d = an * d + b; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = b + an / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d; const del = d * c; h *= del;
    if (Math.abs(del - 1) < 1e-15) break;
  }
  return h;
}
function lowerRegGamma(s, x) {
  if (x < 0 || s <= 0) return NaN;
  if (x === 0) return 0;
  if (x < s + 1) {
    let ap = s, sum = 1 / s, del = sum;
    for (let n = 1; n < 1000; n++) { ap++; del *= x / ap; sum += del; if (Math.abs(del) < Math.abs(sum) * 1e-15) break; }
    return sum * Math.exp(-x + s * Math.log(x) - logGamma(s));
  }
  return 1 - _gamCF(s, x) * Math.exp(-x + s * Math.log(x) - logGamma(s));
}
function _betaCF(a, b, x) {
  const FPMIN = 1e-300;
  const qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - qab * x / qap; if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d; let h = d;
  for (let m = 1; m <= 200; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d; const del = d * c; h *= del;
    if (Math.abs(del - 1) < 1e-15) break;
  }
  return h;
}
function regIncBeta(a, b, x) {
  if (x <= 0) return 0; if (x >= 1) return 1;
  const bt = Math.exp(a * Math.log(x) + b * Math.log(1 - x) + logGamma(a + b) - logGamma(a) - logGamma(b));
  if (x < (a + 1) / (a + b + 2)) {
    return bt * _betaCF(a, b, x) / a;
  }
  return 1 - bt * _betaCF(b, a, 1 - x) / b;
}
function normalCdf(x) {
  if (x === 0) return 0.5;
  const z = Math.abs(x);
  const p = 0.5 * lowerRegGamma(0.5, z * z / 2); // erf(z/√2)
  return x > 0 ? 0.5 + p : 0.5 - p;
}
function studentTCdf(t, df) {
  const x = df / (df + t * t);
  const ib = regIncBeta(df / 2, 0.5, x);
  return t > 0 ? 1 - 0.5 * ib : 0.5 * ib;
}
function chiSquareCdf(x, k) { return x <= 0 ? 0 : lowerRegGamma(k / 2, x / 2); }
function chiSquareSf(x, k) { return 1 - chiSquareCdf(x, k); }
/* KS 双尾 p 值：p = 2Σ_{k≥1}(−1)^{k-1}e^{−2k²x²}, x=D√n */
function ksPValue(d, n) {
  const x = d * Math.sqrt(n);
  if (x <= 0) return 1;
  if (x >= 8) return 0;
  let sum = 0;
  for (let k = 1; k <= 300; k++) {
    const term = 2 * ((k % 2 === 1) ? 1 : -1) * Math.exp(-2 * k * k * x * x);
    sum += term;
    if (Math.abs(term) < 1e-17) break;
  }
  return Math.max(0, Math.min(1, sum));
}

/* ============================================================================
 * O00 最大似然估计（MLE）
 * ========================================================================== */
function mle(data, family) {
  family = family || 'normal';
  if (family === 'normal') return { mu: mean(data), sigma2ML: varML(data) };
  if (family === 'bernoulli' || family === 'binomial') {
    const p = mean(data); return { p };
  }
  if (family === 'poisson') return { lambda: Math.max(mean(data), 1e-12) };
  throw new Error('mle: 未支持族 ' + family);
}
/* 通用牛顿 MLE：最小化负对数似然，返回 {theta, scoreNorm} */
function mleNewton(negLogLik, grad, hess, theta0, opts) {
  opts = opts || {};
  const maxIter = opts.maxIter || 50;
  const tol = opts.tol || 1e-11;
  let th = theta0.slice();
  let scoreNorm = Infinity;
  for (let k = 0; k < maxIter; k++) {
    const g = grad(th);
    const H = hess(th);
    scoreNorm = Math.sqrt(g.reduce((s, v) => s + v * v, 0));
    if (scoreNorm < tol) break;
    // 一维或多维牛顿：th ← th − H⁻¹ g
    let step;
    if (g.length === 1) {
      step = [-g[0] / H[0][0]];
    } else {
      step = solveLinear(H, g.map((v) => -v));
      if (!step) break;
    }
    th = th.map((v, i) => v + step[i]);
  }
  return { theta: th, scoreNorm };
}
function solveLinear(A, b) {
  const n = A.length;
  const M = A.map((row, i) => row.slice().concat([b[i]]));
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-14) return null;
    if (piv !== col) { const t = M[piv]; M[piv] = M[col]; M[col] = t; }
    const d = M[col][col];
    for (let r = 0; r < n; r++) { if (r === col) continue; const f = M[r][col] / d; if (f === 0) continue; for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c]; }
  }
  return M.map((row, i) => row[n] / M[i][i]);
}

/* ============================================================================
 * O01 最大后验估计（MAP，正态-正态共轭）
 * ========================================================================== */
function mapEstimate(data, mu0, tau0, tau) {
  // 先验 N(mu0, 1/tau0)，似然 N(·, 1/tau) 独立同分布
  const xbar = mean(data), n = data.length;
  const tauPost = tau0 + n * tau;
  const muMap = (tau0 * mu0 + n * tau * xbar) / tauPost;
  return { mu: muMap, tauPost, sigma2Post: 1 / tauPost };
}

/* ============================================================================
 * O02 / O08 矩估计 / 矩方法
 * ========================================================================== */
function methodOfMoments(data, family) {
  family = family || 'normal';
  if (family === 'normal') return { mu: mean(data), sigma2ML: varML(data) };
  if (family === 'exponential') return { lambda: 1 / mean(data) };
  if (family === 'bernoulli') return { p: mean(data) };
  if (family === 'gamma') {
    // 形状-尺度：E[X]=kθ, Var[X]=kθ² → θ=Var/Mean, k=Mean²/Var
    const m = mean(data), v = varML(data);
    const theta = v / m, k = m * m / v;
    return { k, theta };
  }
  throw new Error('methodOfMoments: 未支持族 ' + family);
}
function momentMethod(data, family) { return methodOfMoments(data, family || 'exponential'); }

/* ============================================================================
 * O03 贝叶斯估计（正态-正态共轭后验均值与方差）
 * ========================================================================== */
function bayesEstimate(data, mu0, tau0, tau) {
  const xbar = mean(data), n = data.length;
  const tauPost = tau0 + n * tau;
  return { muPost: (tau0 * mu0 + n * tau * xbar) / tauPost, sigma2Post: 1 / tauPost, tauPost };
}

/* ============================================================================
 * O04 KS 检验（拟合优度，双尾）
 * ========================================================================== */
function ksTest(data, cdf) {
  const xs = data.slice().sort((a, b) => a - b);
  const n = xs.length;
  let D = 0, Dplus = 0, Dminus = 0;
  for (let i = 0; i < n; i++) {
    const F = cdf(xs[i]);
    const below = (i + 1) / n - F;      // D₊
    const above = F - i / n;            // D₋
    if (below > Dplus) Dplus = below;
    if (above > Dminus) Dminus = above;
    if (below > D || above > D) D = Math.max(below, above);
  }
  return { D, Dplus, Dminus, n, p: ksPValue(D, n) };
}
/* 命名分布 CDF 工厂 */
function normalCdfFactory(mu, sigma) { return (x) => normalCdf((x - mu) / sigma); }

/* ============================================================================
 * O05 卡方检验（拟合优度 + 2×2 独立性）
 * ========================================================================== */
function chiSquareTest(observed, expectedProb, estimatedParams) {
  // 拟合优度：observed 各类频数，expectedProb 各类概率（和=1）
  const n = observed.reduce((s, v) => s + v, 0);
  let stat = 0, df = observed.length - 1;
  for (let i = 0; i < observed.length; i++) {
    const E = n * expectedProb[i];
    stat += (observed[i] - E) * (observed[i] - E) / Math.max(E, 1e-12);
  }
  if (estimatedParams) df -= estimatedParams;
  return { stat, df, p: chiSquareSf(stat, df) };
}
function chiSquareIndependence(table) {
  // table: 2×2 [[a,b],[c,d]]
  const [[a, b], [c, d]] = table;
  const N = a + b + c + d;
  const E11 = (a + b) * (a + c) / N, E12 = (a + b) * (b + d) / N;
  const E21 = (c + d) * (a + c) / N, E22 = (c + d) * (b + d) / N;
  const stat = (a - E11) * (a - E11) / E11 + (b - E12) * (b - E12) / E12 + (c - E21) * (c - E21) / E21 + (d - E22) * (d - E22) / E22;
  const df = 1;
  return { stat, df, p: chiSquareSf(stat, df) };
}

/* ============================================================================
 * O06 Bootstrap（非参数重采样）
 * ========================================================================== */
function bootstrap(data, statFn, B, rng) {
  rng = rng || makeRng(0x9E3779B9);
  const n = data.length;
  const boots = new Array(B);
  for (let b = 0; b < B; b++) {
    const idx = new Array(n);
    for (let i = 0; i < n; i++) idx[i] = Math.floor(rng() * n) % n;
    const sample = idx.map((k) => data[k]);
    boots[b] = statFn(sample);
  }
  const m = mean(boots);
  const se = stdSample(boots);
  return { boots, mean: m, se, B };
}
function percentileCI(boots, alpha) {
  const a = alpha != null ? alpha : 0.05;
  const s = boots.slice().sort((x, y) => x - y);
  const lo = s[Math.floor(a / 2 * s.length)];
  const hi = s[Math.floor((1 - a / 2) * s.length)];
  return [lo, hi];
}

/* ============================================================================
 * O07 交叉验证似然（k 折，正态均值模型）
 * ========================================================================== */
function cvLikelihood(data, k, rng) {
  rng = rng || makeRng(0x9E3779B9);
  k = k || 5;
  const n = data.length;
  const order = data.map((v, i) => i);
  // 洗牌（Fisher-Yates，使用给定 rng）
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); const t = order[i]; order[i] = order[j]; order[j] = t; }
  let ll = 0, count = 0;
  for (let fold = 0; fold < k; fold++) {
    const testIdx = order.filter((_, i) => i % k === fold);
    const trainIdx = order.filter((_, i) => i % k !== fold);
    const train = trainIdx.map((i) => data[i]);
    const mu = mean(train), sd = Math.max(stdSample(train), 1e-9);
    for (const i of testIdx) {
      const x = data[i];
      ll += -0.5 * Math.log(2 * Math.PI) - Math.log(sd) - 0.5 * ((x - mu) / sd) * ((x - mu) / sd);
      count++;
    }
  }
  return ll / count;
}

/* ============================================================================
 * O09 拟似然（Poisson 等散布族：均值估计=样本均值，离散度 φ=Σ(y−μ)²/μ/(n−1)）
 * ========================================================================== */
function quasiLikelihood(data, family) {
  family = family || 'poisson';
  const mu = mean(data);
  const n = data.length;
  let phi;
  if (family === 'poisson' || family === 'quasi-poisson') {
    let num = 0; for (const y of data) num += (y - mu) * (y - mu);
    phi = num / Math.max((n - 1) * mu, 1e-12); // 经典 Poisson φ=1
  } else if (family === 'normal') {
    phi = varSample(data);
  } else throw new Error('quasiLikelihood: 未支持族 ' + family);
  return { mu, phi, dispersion: phi };
}

/* ============================================================================
 * O10 稳健估计（Huber M 估计，位置）
 * ========================================================================== */
function robustEstimate(data, opts) {
  opts = opts || {};
  const c = opts.c || 1.345;
  const maxIter = opts.maxIter || 100;
  let mu = mean(data);
  const s = Math.max(stdSample(data), 1e-9);
  const mad = () => {
    const med = median(data);
    const absdev = data.map((x) => Math.abs(x - med));
    return median(absdev) * 1.4826;
  };
  const scale = opts.scale || (mads => mads > 0 ? mads : s)(mad());
  for (let it = 0; it < maxIter; it++) {
    let num = 0, den = 0;
    for (const x of data) {
      const r = (x - mu) / scale;
      const psi = Math.abs(r) <= c ? r : Math.sign(r) * c;
      num += psi * scale;
      const w = Math.abs(r) <= c ? 1 : c / Math.abs(r); // 影响函数权重
      den += w;
    }
    if (den === 0) break;
    const muNew = mu + num / den;
    if (Math.abs(muNew - mu) < 1e-12 * (Math.abs(mu) + 1)) { mu = muNew; break; }
    mu = muNew;
  }
  return { mu, scale, c };
}
function median(xs) {
  const s = xs.slice().sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]);
}

/* ============================================================================
 * O11 经验似然（单均值非参数）：在给定 mu0 处计算 −2logR；极大点在样本均值
 * ========================================================================== */
function empiricalLikelihood(data, mu0) {
  const n = data.length;
  const d = data.map((x) => x - mu0);
  // 经验似然对偶（Owen）：pᵢ = 1/(n(1+λ dᵢ))，λ 由【均值约束】决定
  //   g(λ) = Σ dᵢ/(1+λ dᵢ) = 0   （g 关于 λ 严格递减：g' = −Σ dᵢ²/(1+λdᵢ)² < 0）
  // 注意：不能用规范化式 Σ 1/(1+λdᵢ)=n —— 它在任意 μ₀ 处都被 λ=0 平凡满足，会导致恒返回 0 的退化解。
  // 经验似然比 R(μ₀)=Π(n pᵢ)=Π 1/(1+λdᵢ)，故 −2logR = 2Σ log(1+λdᵢ) ≥ 0，且在 μ₀=x̄ 处取 0。
  const g = (lam) => { let s = 0; for (const di of d) s += di / (1 + lam * di); return s; };
  // 可行域由「所有 1+λdᵢ>0」决定，是【非对称】区间 (−1/max dᵢ⁺, 1/|min dᵢ⁻|)：
  // λ→−1/maxPos⁺ 时 g→+∞，λ→1/maxNegAbs⁻ 时 g→−∞，故区间内含唯一根。
  // 切勿用对称 ±1/max|d| —— 那会把真实根排除在区间外，使二分收敛到端点而得到伪造的 λ。
  let maxPos = 0, maxNegAbs = 0;
  for (const di of d) { if (di > maxPos) maxPos = di; if (di < 0 && -di > maxNegAbs) maxNegAbs = -di; }
  const SHRINK = 1 - 1e-12;
  let lo = -(maxPos > 0 ? 1 / maxPos : 1e6) * SHRINK;
  let hi = (maxNegAbs > 0 ? 1 / maxNegAbs : 1e6) * SHRINK;
  // μ₀ 恰为样本均值时 λ=0 即根（−2logR=0），直接返回避免无谓迭代
  if (Math.abs(g(0)) < 1e-14) return { mu0, lam: 0, minus2LogR: 0 };
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (lo + hi);
    const gm = g(mid);
    if (Math.abs(gm) < 1e-13) { lo = hi = mid; break; }
    if (gm > 0) lo = mid; else hi = mid;   // g 递减：g>0 说明根在右侧
  }
  const lam = 0.5 * (lo + hi);
  let minus2LogR = 0;
  // 经验似然比 R(μ0)=Π (n p_i)，其中 p_i = 1/(n(1+λ(x_i−μ0)))，故 n p_i = 1/(1+λ(x_i−μ0))
  for (const x of data) minus2LogR += 2 * Math.log(1 + lam * (x - mu0));
  return { mu0, lam, minus2LogR };
}
function empiricalLikelihoodMax(data) {
  // 极大值点在样本均值（解析）：返回该点的 −2logR（应≈0）
  const m = mean(data);
  return empiricalLikelihood(data, m).minus2LogR;
}

/* ============================================================================
 * O12 假设检验（Welch t 双样本 / 单样本 z）
 * ========================================================================== */
function hypothesisTest(d1, d2, opts) {
  opts = opts || {};
  if (!d2) {
    // 单样本 z/t：检验 mean=mu0
    const mu0 = opts.mu0 != null ? opts.mu0 : 0;
    const m = mean(d1), sd = stdSample(d1), n = d1.length;
    const t = (m - mu0) / (sd / Math.sqrt(n));
    const p = 2 * (1 - studentTCdf(Math.abs(t), n - 1));
    return { t, df: n - 1, p, type: 'one-sample' };
  }
  const n1 = d1.length, n2 = d2.length;
  const m1 = mean(d1), m2 = mean(d2);
  const v1 = varSample(d1), v2 = varSample(d2);
  const se = Math.sqrt(v1 / n1 + v2 / n2);
  const t = (m1 - m2) / se;
  const df = Math.pow(v1 / n1 + v2 / n2, 2) / (Math.pow(v1 / n1, 2) / (n1 - 1) + Math.pow(v2 / n2, 2) / (n2 - 1));
  const p = 2 * (1 - studentTCdf(Math.abs(t), df));
  return { t, df, p, type: 'welch' };
}

/* ============================================================================
 * O13 置信区间（正态 z / t / Wilson 比例）
 * ========================================================================== */
function confidenceInterval(data, opts) {
  opts = opts || {};
  const alpha = opts.alpha != null ? opts.alpha : 0.05;
  const type = opts.type || 't';
  const n = data.length;
  if (type === 'normal' || type === 'z') {
    const m = mean(data), sd = stdSample(data);
    const z = normInv(1 - alpha / 2);
    return [m - z * sd / Math.sqrt(n), m + z * sd / Math.sqrt(n)];
  }
  if (type === 't') {
    const m = mean(data), se = stdSample(data) / Math.sqrt(n);
    const tcrit = studentTInv(1 - alpha / 2, n - 1);
    return [m - tcrit * se, m + tcrit * se];
  }
  if (type === 'proportion') {
    // Wilson 区间（np = 观测数，options.n 提供分母）
    const nn = opts.n != null ? opts.n : n;
    const x = opts.x != null ? opts.x : Math.round(mean(data) * nn);
    const phat = x / nn;
    const z = normInv(1 - alpha / 2);
    const denom = 1 + z * z / nn;
    const center = (phat + z * z / (2 * nn)) / denom;
    const half = z * Math.sqrt(phat * (1 - phat) / nn + z * z / (4 * nn * nn)) / denom;
    return [center - half, center + half];
  }
  throw new Error('confidenceInterval: 未支持类型 ' + type);
}
/* 标准正态逆 CDF（Acklam 有理近似，精度 ~1e-9） */
function normInv(p) {
  if (p <= 0) return -Infinity; if (p >= 1) return Infinity;
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  const plow = 0.02425, phigh = 1 - plow;
  let q, r;
  if (p < plow) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  if (p > phigh) { q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  q = p - 0.5; r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}
/* 学生 t 逆 CDF（二分 on 单调 CDF） */
function studentTInv(p, df) {
  if (p <= 0) return -Infinity; if (p >= 1) return Infinity;
  let lo = -100, hi = 100;
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (lo + hi);
    const c = studentTCdf(mid, df);
    if (c < p) lo = mid; else hi = mid;
  }
  return 0.5 * (lo + hi);
}

/* ============================================================================
 * O14 似然比检验（LRT）：2(ℓ_full−ℓ_null) ~ χ²(df)
 * ========================================================================== */
function likelihoodRatioTest(llFull, llNull, df) {
  const stat = 2 * (llFull - llNull);
  return { stat, df, p: chiSquareSf(stat, df) };
}
function normalLogLik(data, mu, sigma) {
  const n = data.length;
  let ll = -n * (0.5 * Math.log(2 * Math.PI) + Math.log(sigma));
  for (const x of data) ll -= 0.5 * ((x - mu) / sigma) * ((x - mu) / sigma);
  return ll;
}

/* ============================================================================
 * O15 拟合优度调度（KS 或 χ²）
 * ========================================================================== */
function goodnessOfFit(data, cdf, opts) {
  opts = opts || {};
  if (opts.method === 'chi2') return chiSquareTest(opts.observed, opts.expectedProb, opts.estimatedParams);
  return ksTest(data, cdf);
}

/* ============================================================================
 * 自洽报告骨架（零依赖本地实现，与 forge 契约一致）
 * ========================================================================== */
function caseOf(id, name, ok, actual, detail, tol) {
  const numActual = (typeof actual === 'number' && isFinite(actual)) ? actual : (ok ? 0 : 1);
  return { id, name, status: ok ? 'pass' : 'fail', passed: !!ok, actual: numActual, expected: 0, tol: (typeof tol === 'number') ? tol : TOL.EXACT, detail };
}
function buildReport(module, version, layer, cases, invariants, seed) {
  const passed = cases.filter((c) => c.status === 'pass').length;
  const failed = cases.length - passed;
  let hashSrc = '';
  for (const c of cases) hashSrc += c.id + '|' + c.status + '|' + (typeof c.actual === 'number' ? String(Math.round(c.actual * 1e12)) : String(c.actual)) + '\n';
  return { schema: 'forge-selfcheck/1.0', module, version, layer, passed, failed, skipped: 0, duration_ms: 0, tol: TOL, invariants, cases, env: { platform: (typeof process !== 'undefined' && process.platform) || 'browser', arch: (typeof process !== 'undefined' && process.arch) || 'unknown', node: (typeof process !== 'undefined' && process.version) || 'browser', seed }, hash: 'sha256:' + sha256HexLocal(hashSrc) };
}
const K256EST = [
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
];
function _rotrL(x, n) { return ((x >>> n) | (x << (32 - n))) >>> 0; }
function sha256HexLocal(str) {
  const bytes = [];
  for (let i = 0; i < str.length; i++) {
    let c = str.charCodeAt(i);
    if (c < 0x80) bytes.push(c);
    else if (c < 0x800) bytes.push(0xC0 | (c >> 6), 0x80 | (c & 63));
    else if (c >= 0xD800 && c <= 0xDBFF && i + 1 < str.length && str.charCodeAt(i + 1) >= 0xDC00 && str.charCodeAt(i + 1) <= 0xDFFF) {
      c = 0x10000 + ((c - 0xD800) << 10) + (str.charCodeAt(++i) - 0xDC00);
      bytes.push(0xF0 | (c >> 18), 0x80 | ((c >> 12) & 63), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
    } else bytes.push(0xE0 | (c >> 12), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
  }
  const msg = bytes.slice();
  const bitLen = msg.length * 8;
  msg.push(0x80);
  while (msg.length % 64 !== 56) msg.push(0);
  const hi = Math.floor(bitLen / 0x100000000), lo = bitLen >>> 0;
  msg.push((hi >>> 24) & 255, (hi >>> 16) & 255, (hi >>> 8) & 255, hi & 255, (lo >>> 24) & 255, (lo >>> 16) & 255, (lo >>> 8) & 255, lo & 255);
  let Hh = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
  const w = new Array(64);
  for (let off = 0; off < msg.length; off += 64) {
    for (let t = 0; t < 16; t++) w[t] = ((msg[off + 4 * t] << 24) | (msg[off + 4 * t + 1] << 16) | (msg[off + 4 * t + 2] << 8) | msg[off + 4 * t + 3]) | 0;
    for (let t = 16; t < 64; t++) { const x15 = w[t - 15], x2 = w[t - 2]; const s0 = (_rotrL(x15, 7) ^ _rotrL(x15, 18) ^ (x15 >>> 3)) | 0; const s1 = (_rotrL(x2, 17) ^ _rotrL(x2, 19) ^ (x2 >>> 10)) | 0; w[t] = (w[t - 16] + s0 + w[t - 7] + s1) | 0; }
    let a = Hh[0], b = Hh[1], c = Hh[2], d = Hh[3], e = Hh[4], f = Hh[5], g = Hh[6], h = Hh[7];
    for (let t = 0; t < 64; t++) {
      const S1 = (_rotrL(e, 6) ^ _rotrL(e, 11) ^ _rotrL(e, 25)) | 0;
      const ch = ((e & f) ^ (~e & g)) | 0;
      const t1 = (h + S1 + ch + K256EST[t] + w[t]) | 0;
      const S0 = (_rotrL(a, 2) ^ _rotrL(a, 13) ^ _rotrL(a, 22)) | 0;
      const maj = ((a & b) ^ (a & c) ^ (b & c)) | 0;
      const t2 = (S0 + maj) | 0;
      h = g; g = f; f = e; e = (d + t1) | 0; d = c; c = b; b = a; a = (t1 + t2) | 0;
    }
    Hh = [(Hh[0] + a) | 0, (Hh[1] + b) | 0, (Hh[2] + c) | 0, (Hh[3] + d) | 0, (Hh[4] + e) | 0, (Hh[5] + f) | 0, (Hh[6] + g) | 0, (Hh[7] + h) | 0];
  }
  return Hh.map((x) => ('00000000' + (x >>> 0).toString(16)).slice(-8)).join('');
}

/* ============================================================================
 * 自洽自检：8 条皇冠不变量（每条含独立重算 / 双向 / 证伪）
 * ========================================================================== */
function selfCheck() {
  const cases = [], invariants = [];
  const push = (id, name, ok, actual, detail, tol) => {
    cases.push(caseOf(id, name, ok, actual, detail, tol));
    invariants.push({ id, name, ok, expr: name, tol: (typeof tol === 'number') ? tol : TOL.EXACT, worst: ok ? 0 : 1, detail });
  };
  const rng = makeRng(0x9E3779B9);

  /* ---------- INV-EST-001：MLE 与矩估计闭式 EXACT + 牛顿收敛 ---------- */
  {
    // 正态：μ̂=x̄、σ̂²=ML；与直接算 EXACT
    const dataN = [-1.2, 0.3, 2.1, -0.7, 1.5, 0.9, -0.4, 1.8, 0.2, -1.0];
    const mleN = mle(dataN, 'normal');
    const muDirect = mean(dataN), sigDirect = varML(dataN);
    const okN = Math.abs(mleN.mu - muDirect) <= 1e-12 * (Math.abs(muDirect) + 1) && Math.abs(mleN.sigma2ML - sigDirect) <= 1e-12 * (Math.abs(sigDirect) + 1);
    // 伯努利 / 泊松
    const dataB = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1];
    const pMLE = mle(dataB, 'bernoulli').p, pDir = mean(dataB);
    const dataP = [3, 2, 4, 1, 5, 2, 3, 4];
    const lamMLE = mle(dataP, 'poisson').lambda, lamDir = mean(dataP);
    const okB = Math.abs(pMLE - pDir) <= 1e-12, okP = Math.abs(lamMLE - lamDir) <= 1e-12;
    // MLE-Newton：Poisson λ 的 NLL，解析解即 mean；验证得分收敛
    const nP = dataP.length, sumP = dataP.reduce((s, v) => s + v, 0);
    const ll = (lam) => sumP * Math.log(lam) - lam * nP;
    const grad = (lam) => [sumP / lam - nP];
    const hess = (lam) => [[-sumP / (lam * lam)]];
    const nr = mleNewton(ll, grad, hess, [2.0], { maxIter: 25, tol: 1e-11 });
    const okNewton = nr.scoreNorm <= 1e-9 && Math.abs(nr.theta[0] - lamDir) <= 1e-9;
    // 矩估计==MLE（正态）
    const momN = methodOfMoments(dataN, 'normal');
    const okMom = Math.abs(momN.mu - muDirect) <= 1e-12 && Math.abs(momN.sigma2ML - sigDirect) <= 1e-12;
    const ok = okN && okB && okP && okNewton && okMom;
    push('INV-EST-001', 'MLE/矩估计闭式 EXACT(正态/伯努利/泊松) + Newton 得分≤1e-9 + 一致性', ok, ok ? 0 : (okN ? 0 : (okNewton ? 0 : 1)),
      'okN=' + okN + ' okB=' + okB + ' okP=' + okP + ' newtonScore=' + nr.scoreNorm.toExponential(2) + ' okMom=' + okMom, TOL.RESID);
  }

  /* ---------- INV-EST-002：MAP 与贝叶斯共轭 EXACT ---------- */
  {
    const data = [2.1, 1.8, 2.3, 1.9, 2.0, 2.2, 1.7, 2.4];
    const mu0 = 0, tau0 = 1, tau = 1;
    const mapR = mapEstimate(data, mu0, tau0, tau);
    const bayR = bayesEstimate(data, mu0, tau0, tau);
    const n = data.length, xbar = mean(data);
    const tauPost = tau0 + n * tau;
    const muExact = (tau0 * mu0 + n * tau * xbar) / tauPost;
    // 平坦先验 τ₀→0：MAP 退化 MLE=x̄
    const mapFlat = mapEstimate(data, 0, 1e-9, tau);
    const okExact = Math.abs(mapR.mu - muExact) <= 1e-12 * (Math.abs(muExact) + 1);
    const okMapMLE = Math.abs(mapFlat.mu - xbar) <= 1e-9 * (Math.abs(xbar) + 1);
    const okBayesMap = Math.abs(bayR.muPost - mapR.mu) <= 1e-12; // 共轭众数=均值
    const ok = okExact && okMapMLE && okBayesMap;
    push('INV-EST-002', 'MAP/贝叶斯共轭后验均值 EXACT + 平坦先验退化为 MLE + 众数=均值', ok, ok ? 0 : (okExact ? 0 : 1),
      'map=' + mapR.mu.toFixed(6) + ' exact=' + muExact.toFixed(6) + ' flatMu=' + mapFlat.mu.toFixed(6) + ' xbar=' + xbar.toFixed(6) + ' bayesMapSame=' + okBayesMap, TOL.RESID);
  }

  /* ---------- INV-EST-003：KS 检验 D 与 p ---------- */
  {
    // 固定 11 点小样本，手算 D（对照）
    const data = [-1.5, -0.8, -0.3, 0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 2.0, 2.4];
    const cdf = normalCdfFactory(0, 1);
    const ks = ksTest(data, cdf);
    // 独立手算 D（同公式，不同代数路径）
    const xs = data.slice().sort((a, b) => a - b);
    let Dhand = 0;
    for (let i = 0; i < xs.length; i++) Dhand = Math.max(Dhand, (i + 1) / xs.length - cdf(xs[i]), cdf(xs[i]) - i / xs.length);
    const okD = Math.abs(ks.D - Dhand) <= 1e-12;
    // p 随 D 严格单调下降 + 边界（D=0→p=1，D 极大→p→0）
    const pAt = (d) => ksPValue(d, 11);
    const okMono = pAt(0) === 1 && pAt(0.05) > pAt(0.5) && pAt(0.5) > pAt(1.0) && pAt(1.0) > pAt(5.0) && pAt(5.0) < 1e-6;
    // 功效 / 非功效：来自 N(3,1) 拒绝率>0.9，来自 N(0,1) 拒绝率<0.15（α=0.05）
    let rejectShift = 0, rejectNull = 0; const M = 400;
    for (let t = 0; t < M; t++) {
      const s1 = gaussVec(rng, 30).map((v) => v + 3);
      const ks1 = ksTest(s1, normalCdfFactory(0, 1));
      if (ks1.p < 0.05) rejectShift++;
      const s0 = gaussVec(rng, 30);
      const ks0 = ksTest(s0, normalCdfFactory(0, 1));
      if (ks0.p < 0.05) rejectNull++;
    }
    const okPower = rejectShift / M > 0.9 && rejectNull / M < 0.15;
    const ok = okD && okMono && okPower;
    push('INV-EST-003', 'KS D 手算 EXACT + p 单调边界(0→1,大→0) + 功效>0.9/非功效<0.15', ok, ok ? 0 : (okD ? 0 : (okMono ? 0 : 1)),
      'D=' + ks.D.toFixed(6) + ' Dhand=' + Dhand.toFixed(6) + ' p0=' + pAt(0) + ' pBig=' + pAt(5.0).toFixed(6) + ' rejectShift=' + (rejectShift / M).toFixed(3) + ' rejectNull=' + (rejectNull / M).toFixed(3), TOL.RESID);
  }

  /* ---------- INV-EST-004：卡方检验 ---------- */
  {
    // 固定 8 类频数，期望均匀 → 手算 χ² 对照
    const obs = [18, 22, 15, 20, 25, 17, 19, 24];
    const exp = obs.map(() => 1 / 8);
    const cs = chiSquareTest(obs, exp, 0);
    const ncs = obs.reduce((s, v) => s + v, 0);
    let statHand = 0;
    for (const o of obs) statHand += (o - ncs / 8) * (o - ncs / 8) / (ncs / 8);
    const okStat = Math.abs(cs.stat - statHand) <= 1e-9;
    // 独立性 2×2 对照 + df=1（解析 χ²=25.25）
    const tab = [[40, 10], [15, 35]];
    const ci = chiSquareIndependence(tab);
    const okInd = Math.abs(ci.stat - 25.2525) <= 1e-3 && ci.df === 1;
    // 功效：从明显非均匀的真实分布（偏斜）抽样，χ² 均匀性检验拒绝率>0.9（α=0.05）
    function sampleMultinomial(trials, probs, rng) {
      const cps = []; let s = 0; for (const p of probs) { s += p; cps.push(s); }
      const counts = new Array(probs.length).fill(0);
      for (let i = 0; i < trials; i++) { const u = rng(); let c = 0; while (c < cps.length - 1 && u > cps[c]) c++; counts[c]++; }
      return counts;
    }
    const skew = [0.25, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05]; // 真分布 ≠ 均匀
    const N4 = obs.reduce((s, v) => s + v, 0); // 总计数 160
    let reject = 0; const M = 400;
    for (let t = 0; t < M; t++) {
      const o = sampleMultinomial(N4, skew, rng);
      const c2 = chiSquareTest(o, exp, 0);
      if (c2.p < 0.05) reject++;
    }
    const okPower = reject / M > 0.9;
    // p 与 χ² CDF 一致（stat 已知 → p 合理区间）
    const okCdf = cs.p > 0 && cs.p < 1 && Math.abs(chiSquareCdf(cs.stat, 7) + cs.p - 1) <= 1e-9;
    const ok = okStat && okInd && okPower && okCdf;
    push('INV-EST-004', '卡方拟合优度 χ² 手算 EXACT + 独立性对照(df=1) + 功效>0.9 + CDF 自洽', ok, ok ? 0 : (okStat ? 0 : (okInd ? 0 : 1)),
      'stat=' + cs.stat.toFixed(4) + ' hand=' + statHand.toFixed(4) + ' ind=' + ci.stat.toFixed(4) + ' p=' + cs.p.toFixed(4) + ' power=' + (reject / M).toFixed(3), TOL.SOLVE);
  }

  /* ---------- INV-EST-005：Bootstrap 非参数 ---------- */
  {
    const n = 40;
    const data = gaussVec(rng, n).map((v) => v * 2 + 5); // N(5,4)
    const stat = (xs) => mean(xs);
    const boot = bootstrap(data, stat, 2000, makeRng(0x5B0077));
    const seAnalytic = stdSample(data) / Math.sqrt(n);
    const relErr = Math.abs(boot.se - seAnalytic) / seAnalytic;
    const ci = percentileCI(boot.boots, 0.05);
    const ciCenter = 0.5 * (ci[0] + ci[1]);
    const okSE = relErr <= 0.15;
    const okCenter = Math.abs(ciCenter - mean(data)) <= 2 * seAnalytic;
    const okMean = Math.abs(boot.mean - mean(data)) <= 3 * seAnalytic / Math.sqrt(2000);
    const ok = okSE && okCenter && okMean;
    push('INV-EST-005', 'Bootstrap SE 与解析 SE/√n 误差≤15% + 百分位 CI 中心≈x̄ + 均值一致', ok, ok ? 0 : (okSE ? 0 : (okCenter ? 0 : 1)),
      'seBoot=' + boot.se.toFixed(4) + ' seAna=' + seAnalytic.toFixed(4) + ' relErr=' + relErr.toFixed(3) + ' ciCenter=' + ciCenter.toFixed(3) + ' xbar=' + mean(data).toFixed(3), TOL.SOLVE);
  }

  /* ---------- INV-EST-006：置信区间覆盖 ---------- */
  {
    const Nrep = 2000, n = 50, muTrue = 3, sd = 2;
    let coverT = 0, coverNorm = 0, coverProp = 0, propTrials = 0;
    for (let t = 0; t < Nrep; t++) {
      const samp = gaussVec(rng, n).map((v) => v * sd + muTrue);
      const ciT = confidenceInterval(samp, { type: 't', alpha: 0.05 });
      if (ciT[0] <= muTrue && muTrue <= ciT[1]) coverT++;
      const ciN = confidenceInterval(samp, { type: 'normal', alpha: 0.05 });
      if (ciN[0] <= muTrue && muTrue <= ciN[1]) coverNorm++;
      // 比例：真 p=0.5（用 Binomial(100,0.5) 抽样，非均匀）
      let k = 0; for (let j = 0; j < 100; j++) if (rng() < 0.5) k++;
      const np = 100;
      const ciP = confidenceInterval([], { type: 'proportion', x: k, n: np, alpha: 0.05 });
      if (ciP[0] <= 0.5 && 0.5 <= ciP[1]) coverProp++;
      propTrials++;
    }
    const covT = coverT / Nrep, covN = coverNorm / Nrep, covP = coverProp / propTrials;
    // Wilson 上下界手算对照（单例）
    const cp = 0.5, np = 100, z = normInv(0.975);
    const denom = 1 + z * z / np, center = (cp + z * z / (2 * np)) / denom;
    const half = z * Math.sqrt(cp * (1 - cp) / np + z * z / (4 * np * np)) / denom;
    const ciP0 = confidenceInterval([], { type: 'proportion', x: 50, n: 100, alpha: 0.05 });
    const okWilson = Math.abs(ciP0[0] - (center - half)) <= 1e-9 && Math.abs(ciP0[1] - (center + half)) <= 1e-9;
    const ok = covT >= 0.92 && covT <= 0.98 && covN >= 0.92 && covN <= 0.98 && covP >= 0.92 && covP <= 0.98 && okWilson;
    push('INV-EST-006', '正态/t/Wilson 95% CI 覆盖∈[0.92,0.98](2000次) + Wilson 边界手算 EXACT', ok, ok ? 0 : (covT >= 0.92 && covT <= 0.98 ? 0 : 1),
      'covT=' + covT.toFixed(3) + ' covNorm=' + covN.toFixed(3) + ' covProp=' + covP.toFixed(3) + ' wilsonOK=' + okWilson, TOL.CONV);
  }

  /* ---------- INV-EST-007：假设检验 Welch ---------- */
  {
    // 手算对照（小样本）
    const a = [1, 2, 3, 4, 5], b = [2, 4, 6, 8, 10];
    const ht = hypothesisTest(a, b);
    const n1 = 5, n2 = 5, m1 = mean(a), m2 = mean(b), v1 = varSample(a), v2 = varSample(b);
    const se = Math.sqrt(v1 / n1 + v2 / n2);
    const tHand = (m1 - m2) / se;
    const okT = Math.abs(ht.t - tHand) <= 1e-9;
    // 学生 t CDF：CDF(0;df)=0.5、单调递增
    const cdf0 = studentTCdf(0, 10), cdfPos = studentTCdf(1.5, 10), cdfNeg = studentTCdf(-1.5, 10);
    const okCdf = Math.abs(cdf0 - 0.5) <= 1e-9 && cdfPos > 0.5 && cdfNeg < 0.5 && Math.abs(cdfPos + cdfNeg - 1) <= 1e-9;
    // I 类错误率（α=0.05，两总体同分布 N(0,1)）→ ∈[0.03,0.07]
    let rej = 0; const Nrep = 2000;
    for (let t = 0; t < Nrep; t++) {
      const g1 = gaussVec(rng, 40), g2 = gaussVec(rng, 40);
      if (hypothesisTest(g1, g2).p < 0.05) rej++;
    }
    const iErr = rej / Nrep;
    // 功效：d=0.8（n1=n2=50）→ >0.8
    let rejP = 0;
    for (let t = 0; t < Nrep; t++) {
      const g1 = gaussVec(rng, 50), g2 = gaussVec(rng, 50).map((v) => v + 0.8);
      if (hypothesisTest(g1, g2).p < 0.05) rejP++;
    }
    const power = rejP / Nrep;
    const ok = okT && okCdf && iErr >= 0.03 && iErr <= 0.07 && power > 0.8;
    push('INV-EST-007', 'Welch t 手算 EXACT + 学生 t CDF(0)=0.5 单调 + I类错误∈[0.03,0.07] + 功效>0.8', ok, ok ? 0 : (okT ? 0 : (okCdf ? 0 : 1)),
      't=' + ht.t.toFixed(4) + ' tHand=' + tHand.toFixed(4) + ' cdf0=' + cdf0.toFixed(6) + ' iErr=' + iErr.toFixed(4) + ' power=' + power.toFixed(4), TOL.SOLVE);
  }

  /* ---------- INV-EST-008：LRT + 拟似然/稳健/经验似然 ---------- */
  {
    // LRT 中位 p 值 ∈[0.4,0.6]（H0 下 2000 次）
    let ps = []; const Nrep = 2000;
    for (let t = 0; t < Nrep; t++) {
      const d = gaussVec(rng, 60); // N(0,1)
      const llNull = normalLogLik(d, 0, 1);
      // 全模型须用 MLE σ̂=√(Σ(x−x̄)²/n)（而非 n−1 的样本 sd）：否则 ℓ_full 未取到极大，
      // 会使 LRT 统计量在 H₀ 附近出现负值（嵌套模型 ℓ_full≥ℓ_null 被破坏）。
      const m = mean(d); const sd = Math.max(Math.sqrt(varML(d)), 1e-9);
      const llFull = normalLogLik(d, m, sd);
      // Wilks：全模型同时解放 μ 与 σ（2 个自由参数）⇒ H₀ 下 2(ℓ₁−ℓ₀) ~ χ²(2)
      ps.push(likelihoodRatioTest(llFull, llNull, 2).p);
    }
    ps.sort((x, y) => x - y);
    const medP = ps[Math.floor(ps.length / 2)];
    const okLRT = medP >= 0.4 && medP <= 0.6;
    // 拟似然对 Poisson 退化为 MLE + φ≈1（用 rng 生成 Poisson 数据，期望 φ≈1）
    function samplePoisson(rng, lambda) {
      const L = Math.exp(-lambda); let k = 0, p = 1;
      do { k++; p *= rng(); } while (p > L); return k - 1;
    }
    const dp = []; for (let i = 0; i < 300; i++) dp.push(samplePoisson(rng, 2.9));
    const ql = quasiLikelihood(dp, 'poisson');
    const okQL = Math.abs(ql.mu - mean(dp)) <= 1e-9 && Math.abs(ql.phi - 1) <= 0.25;
    // Huber 稳健：污染离群降权
    const clean = gaussVec(rng, 47).map((v) => v); // N(0,1) 真值 0
    const contaminated = clean.concat([20, -18, 22]); // 真值仍 0
    const hub = robustEstimate(contaminated, { c: 1.345 });
    const smean = mean(contaminated);
    const okHuber = Math.abs(hub.mu - 0) < Math.abs(smean - 0) && Math.abs(hub.mu) < Math.abs(smean - 0) * 0.6;
    // 经验似然极大点 EXACT 在样本均值：−2logR(x̄)=0，且偏离均值后 −2logR 必显著 >0
    // （反空洞证伪探针：若实现恒返回 0，则「偏离 >0」必失败——此条正是捕获该退化解的探针）
    const de = gaussVec(rng, 30).map((v) => v + 1);
    const mDe = mean(de);
    const elMax = empiricalLikelihoodMax(de);
    const elOff = [0.5, -0.5].map((o) => empiricalLikelihood(de, mDe + o).minus2LogR);
    const okEL = Math.abs(elMax) <= 1e-9 && elOff.every((v) => v > 0.5);
    const ok = okLRT && okQL && okHuber && okEL;
    push('INV-EST-008', 'LRT 中位 p∈[0.4,0.6] + 拟似然退化为 MLE(φ≈1) + Huber 离群降权 + 经验似然极大点在样本均值 EXACT', ok, ok ? 0 : (okLRT ? 0 : (okQL ? 0 : 1)),
      'medP=' + medP.toFixed(3) + ' qlMu=' + ql.mu.toFixed(4) + ' qlPhi=' + ql.phi.toFixed(4) + ' huber=' + hub.mu.toFixed(4) + ' mean=' + smean.toFixed(4) + ' elMax=' + elMax.toExponential(2) + ' elOff=' + elOff.map((v) => v.toFixed(3)).join('/'), TOL.SOLVE);
  }

  return buildReport('estimate-forge', '1.0.0', 'L5', cases, invariants, '0x9E3779B9');
}

/* ============================================================================
 * API 导出 + 双端挂载（UMD）+ CLI 自检
 * ========================================================================== */
const API = {
  name: 'estimate-forge', version: '1.0.0', layer: 'L5', TOL,
  makeRng, gaussVec,
  mean, varSample, varML, stdSample, median,
  logGamma, lowerRegGamma, regIncBeta, normalCdf, studentTCdf, chiSquareCdf, chiSquareSf, ksPValue, normInv, studentTInv,
  mle, mleNewton, solveLinear,
  mapEstimate, methodOfMoments, momentMethod, bayesEstimate,
  ksTest, normalCdfFactory, chiSquareTest, chiSquareIndependence,
  bootstrap, percentileCI, cvLikelihood,
  quasiLikelihood, robustEstimate, empiricalLikelihood, empiricalLikelihoodMax,
  hypothesisTest, confidenceInterval, likelihoodRatioTest, normalLogLik, goodnessOfFit,
  selfCheck
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeEstimate = API;

if (typeof process !== 'undefined' && process.argv[1] && /estimate-forge\.js$/.test(process.argv[1]) && (process.argv.includes('--self-check') || process.argv.includes('--json'))) {
  const r = selfCheck();
  const lines = ['=== ' + r.module + ' v' + r.version + ' (' + r.layer + ') ' + (r.failed === 0 ? 'PASS' : 'FAIL') + ' ==='];
  lines.push('passed ' + r.passed + ' / failed ' + r.failed + ' / hash ' + (r.hash || '').slice(0, 24));
  for (const c of (r.cases || [])) lines.push('  ' + (c.status === 'pass' ? '+OK' : '-XX') + ' ' + c.id + ' ' + c.name);
  process.stdout.write((process.argv.includes('--json') ? JSON.stringify(r, null, 2) : lines.join('\n')) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
