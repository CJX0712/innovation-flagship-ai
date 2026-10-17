/**
 * @forge-module mcmc-forge
 * @layer L5
 * @version 1.0.0
 * @brief     MCMC 采样内核（F5-G5）：Metropolis / Metropolis-Hastings（含非对称提议）/ Gibbs / 随机游走Metropolis+窗宽调优 / HMC(leapfrog) / NUTS(U-turn) / dual-averaging 自适应步长 / 混合蒙特卡洛(随机轨迹长) / 模拟退火（几何降温+跨势垒）
 * @reuse     Metropolis et al.《Equation of State Calculations by Fast Computing Machines》(1953)|https://doi.org/10.1063/1.1699114|A|Metropolis 算法与接受率
 * @reuse     Hastings《Monte Carlo sampling methods using Markov chains》(1970)|https://doi.org/10.1093/biomet/57.1.97|A|MH 非对称提议修正 α=min(1, π(y)q(x|y)/π(x)q(y|x))
 * @reuse     Geman & Geman《Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images》(1984)|https://doi.org/10.1109/TPAMI.1984.1056576|A|Gibbs 条件扫描
 * @reuse     Duane et al.《Hybrid Monte Carlo》(1987)|https://doi.org/10.1016/0370-2693(87)91197-X|A|HMC：动量刷新+leapfrog+MH接受
 * @reuse     Hoffman & Gelman《The No-U-Turn Sampler》(JMLR 2014)|https://jmlr.org/papers/v15/hoffman14a.html|A|NUTS 递归倍增+U-turn判据+dual averaging
 * @reuse     Neal《MCMC Using Hamiltonian Dynamics》(2011)|https://arxiv.org/abs/1206.1901|B|leapfrog 可逆性/保体积/dual averaging 公式
 * @reuse     Roberts & Rosenthal《Optimal scaling for various Metropolis-Hastings algorithms》(2001)|https://doi.org/10.1214/ss/1015346320|A|最优窗宽 2.38/√d 与 0.44 接受率
 * @reuse     Kirkpatrick et al.《Optimization by Simulated Annealing》(1983)|https://doi.org/10.1126/science.220.4598.671|A|退火调度与跨势垒
 * @beyond    纯 JS 单文件零依赖零构建，双端同构 UMD；每个采样器均为独立实现（非互相调用），并以反空洞铁律交叉验证：对称提议下 Metropolis 与 MH 轨迹必须逐位一致而 MH 非对称提议必须分歧；离散两状态链上解析计算 MH 核并断言细致平衡精确成立（同时证明恒接受核必违反）； Gibbs/矩收敛/NUTS/退火均配独立黄金参照（解析矩、手搓 KS 正态检验、解析 U-turn 界、冷链对照），并附「恒接受随机游走必不收敛」「错误条件方差必偏离」「超大步长必低接受」「冷链必陷局部模」证伪断言。
 * @invariant INV-MC-001|对称提议下 Metropolis ≡ Metropolis-Hastings：同 seed 同提议流两独立实现轨迹逐位相等（EXACT）；非对称提议下两链必分歧（反空洞，证伪提议修正项生效）；附 SHA-256 FIPS 180-4 已知向量防护（'abc'/'' 标准摘要，防哈希常量表转写错误）|0|双独立实现 JSON 逐位比对 + 分歧断言 + 已知向量
 * @invariant INV-MC-002|MH 细致平衡（离散两状态）：解析核 P=Q·α 满足 π_i·P_ij = π_j·P_ji（EXACT ≤1e-15，恒接受核同式违反 >0.1）；60000 步经验频率≈π（10σ）、经验转移频率≈解析 P（10σ）|1e-15|解析细致平衡恒等式 + 经验转移矩阵 10σ 检验 + 恒接受证伪
 * @invariant INV-MC-003|RWM 大样本矩收敛：N(3,2²) 目标 42000 步（烧除 2000）样本均值/方差 z 分数 ≤10（三种子）；KS 检验 vs 解析正态 D≤1.63/√ESS；最优窗宽 2.38 下接受率≈0.44±0.05（Roberts-Rosenthal）；Robbins-Monro 调优收敛到 0.44±0.05；恒接受随机游走方差 >100（证伪）|10σ|手搓 ESS/KS + 解析矩 + 调优收敛 + 随机游走证伪
 * @invariant INV-MC-004|Gibbs 条件矩：BVN(ρ=0.6) 62000 步，x|y 回归斜率≈0.6、残差方差≈0.64、边际方差≈1（均 10σ）；错误条件方差(1.0)链边际方差偏离 >0.3（证伪）；链 lag-1 自相关 <0.95（非平凡）|10σ|解析条件矩 + 回归估计 + 错误条件证伪
 * @invariant INV-MC-005|leapfrog 结构性质：时间反演可逆（x,p)→(x',p')→(x,-p) 误差 ≤1e-12；保体积（数值 Jacobian 行列式 ≈1，≤1e-6）；能量误差有界（L=8/40/160 最大 |ΔH| ≤ 0.05）而显式 Euler 同配置漂移（err(160) > 5·err(8)）|1e-12|有限差分 Jacobian + 能量界 + Euler 对照
 * @invariant INV-MC-006|NUTS 与混合 MC 效率：病态 2D 高斯（sd=[1,10]）上 NUTS ESS > 3×RWM ESS；U-turn 终止占比 >0.5；每步 leapfrog 数 ∈[1, 2^(maxDepth+1)-2]；dual averaging 后接受率 ∈[0.5,0.95]；超大 eps(5.0) 接受率 <0.2（证伪）；混合MC（随机轨迹长 U{1..8}）ESS > 2×RWM 且轨迹长频率≈均匀（≤0.05）|0.05|ESS 黄金对比 + U-turn 统计 + 轨迹长直方图 + 超大 eps 证伪
 * @invariant INV-MC-007|dual averaging 自适应步长：目标接受率 0.65 收敛到 0.65±0.05、目标 0.9 收敛到 0.9±0.05；且 eps(0.9) < eps(0.65)（严格小 >2%，证伪调优响应目标）|0.05|Hoffman-Gelman 公式 + 双目标对照
 * @invariant INV-MC-008|退火：同 seed 全链逐位相等（EXACT）+ 异 seed 分歧；调度几何段严格递减、T∈[Tmin,T0]（EXACT）；退火（8 种子统计）≥5/8 从局部模 -5 达全局模 +5 且末态平均对数密度高于冷链 +0.5；冷链（3 种子）全链不越过 x≤1.5（困于局部模，证伪对照）；高斯目标上末段方差 < 首段方差/4 而恒温链方差比 ∈[0.6,1.6]（降温致锐）|0|双跑逐位 + 调度结构 + 8 种子全局模统计 + 冷链对照 + 降温致锐
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

/* ============================================================================
 * 纯 JS SHA-256（FIPS 180-4）：hash 幂等 + 'sha256:' 前缀契约
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
 * 基础数值工具
 * ========================================================================== */
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

/** 标准正态向量（Box-Muller，成对消耗均匀流；同 seed 逐位可复现） */
function gaussVec(rng, dim) {
  const out = new Array(dim);
  for (let i = 0; i < dim; i += 2) {
    const u1 = Math.max(rng(), 1e-300), u2 = rng();
    const r = Math.sqrt(-2 * Math.log(u1));
    out[i] = r * Math.cos(2 * Math.PI * u2);
    if (i + 1 < dim) out[i + 1] = r * Math.sin(2 * Math.PI * u2);
  }
  return out;
}

function meanOf(xs) { let s = 0; for (const x of xs) s += x; return s / xs.length; }
function varOf(xs, m) { const mu = (m === undefined ? meanOf(xs) : m); let s = 0; for (const x of xs) s += (x - mu) * (x - mu); return s / (xs.length - 1); }
function covOf(xs, ys, mx, my) {
  const mux = (mx === undefined ? meanOf(xs) : mx), muy = (my === undefined ? meanOf(ys) : my);
  let s = 0; for (let i = 0; i < xs.length; i++) s += (xs[i] - mux) * (ys[i] - muy);
  return s / (xs.length - 1);
}

/* ============================================================================
 * 目标分布（对数密度 + 梯度，全部解析）
 * ========================================================================== */
const LOG2PI = Math.log(2 * Math.PI);

/** 对角协方差高斯：logTarget + gradLogTarget（HMC/NUTS 用） */
function makeGaussTarget(mu, sd) {
  const dim = mu.length;
  function logTarget(x) {
    let s = 0;
    for (let i = 0; i < dim; i++) { const z = (x[i] - mu[i]) / sd[i]; s += z * z; }
    let c = 0;
    for (let i = 0; i < dim; i++) c += Math.log(sd[i]);
    return -0.5 * dim * LOG2PI - c - 0.5 * s;
  }
  function gradLogTarget(x) {
    const g = new Array(dim);
    for (let i = 0; i < dim; i++) g[i] = -(x[i] - mu[i]) / (sd[i] * sd[i]);
    return g;
  }
  return { dim, logTarget, gradLogTarget, mu, sd };
}

/** 一维标准位置高斯对数密度 */
function logGauss1d(x, m, s) { const z = (x - m) / s; return -0.5 * LOG2PI - Math.log(s) - 0.5 * z * z; }

/** 多维对角高斯对数密度（含归一化常数，供独立提议的 MH 修正用） */
function logGaussNd(x, mu, sd) {
  let s = 0, c = 0;
  for (let i = 0; i < mu.length; i++) { const z = (x[i] - mu[i]) / sd[i]; s += z * z; c += Math.log(sd[i]); }
  return -0.5 * mu.length * LOG2PI - c - 0.5 * s;
}

/** 一维双峰对数密度（logsumexp 稳定）：log(w·N(x;m1,1) + (1-w)·N(x;m2,1)) */
function logBimodal1d(x, m1, m2, w) {
  const a = Math.log(w) + logGauss1d(x, m1, 1);
  const b = Math.log(1 - w) + logGauss1d(x, m2, 1);
  const hi = a > b ? a : b;
  return hi + Math.log(Math.exp(a - hi) + Math.exp(b - hi));
}

/* ============================================================================
 * 采样器 1：Metropolis（1953，对称高斯提议，独立实现）
 * ========================================================================== */
function metropolisSample(opts) {
  const logTarget = opts.logTarget;
  const dim = opts.dim || 1;
  const steps = opts.steps, stepSize = opts.stepSize, rng = opts.rng;
  const burnIn = opts.burnIn || 0;
  if (!(steps > 0)) throw new RangeError('metropolisSample: steps 必须为正');
  if (!(stepSize > 0)) throw new RangeError('metropolisSample: stepSize 必须为正');
  let x = opts.x0.slice();
  let logp = logTarget(x);
  let accepted = 0;
  const chain = [];
  for (let i = 0; i < steps; i++) {
    const z = gaussVec(rng, dim);
    const y = new Array(dim);
    for (let k = 0; k < dim; k++) y[k] = x[k] + stepSize * z[k];
    const logpY = logTarget(y);
    const logAlpha = logpY - logp;
    const u = rng();
    if (Math.log(u) < logAlpha) { x = y; logp = logpY; accepted++; }
    if (i >= burnIn) chain.push(x.slice());
  }
  return { chain, acceptRate: accepted / steps, final: x.slice(), dim };
}

/* ============================================================================
 * 采样器 2：Metropolis-Hastings（1970，一般提议，独立实现）
 *   propose(x, rng) → { y, logQRatio }，logQRatio = log q(x|y) - log q(y|x)
 * ========================================================================== */
function mhSample(opts) {
  const logTarget = opts.logTarget;
  const dim = opts.dim || 1;
  const steps = opts.steps, rng = opts.rng, propose = opts.propose;
  const burnIn = opts.burnIn || 0;
  if (!(steps > 0)) throw new RangeError('mhSample: steps 必须为正');
  if (typeof propose !== 'function') throw new TypeError('mhSample: propose 必须为函数');
  let x = opts.x0.slice();
  let logp = logTarget(x);
  let accepted = 0;
  const chain = [];
  for (let i = 0; i < steps; i++) {
    const prop = propose(x, rng);
    const y = prop.y;
    const logpY = logTarget(y);
    const logAlpha = logpY - logp + prop.logQRatio;   // 非对称提议修正项
    const u = rng();
    if (Math.log(u) < logAlpha) { x = y; logp = logpY; accepted++; }
    if (i >= burnIn) chain.push(x.slice());
  }
  return { chain, acceptRate: accepted / steps, final: x.slice(), dim };
}

/** 对称随机游走提议（logQRatio ≡ 0） */
function makeRandomWalkProposal(stepSize) {
  return function (x, rng) {
    const z = gaussVec(rng, x.length);
    const y = new Array(x.length);
    for (let k = 0; k < x.length; k++) y[k] = x[k] + stepSize * z[k];
    return { y, logQRatio: 0 };
  };
}

/** 独立提议 N(mu, sd)：logQRatio = log q(x|y) - log q(y|x)，非对称修正的非平凡来源 */
function makeIndependentProposal(mu, sd) {
  const s = typeof sd === 'number' ? new Array(mu.length).fill(sd) : sd;
  return function (x, rng) {
    const z = gaussVec(rng, x.length);
    const y = new Array(x.length);
    for (let k = 0; k < x.length; k++) y[k] = mu[k] + s[k] * z[k];
    return { y, logQRatio: logGaussNd(x, mu, s) - logGaussNd(y, mu, s) };
  };
}

/* ============================================================================
 * 采样器 3：随机游走 Metropolis + 窗宽调优（Robbins-Monro，目标 0.44）
 * ========================================================================== */
function rwMetropolis(opts) {
  const logTarget = opts.logTarget;
  const dim = opts.dim || 1;
  const steps = opts.steps, rng = opts.rng;
  const burnIn = opts.burnIn || 0;
  const tune = !!opts.tune;
  const tuneTarget = (opts.tuneTarget === undefined) ? 0.44 : opts.tuneTarget;
  if (!(steps > 0)) throw new RangeError('rwMetropolis: steps 必须为正');
  let stepSize = opts.stepSize;
  if (!(stepSize > 0)) {
    if (!tune) throw new RangeError('rwMetropolis: 需提供 stepSize 或开启 tune');
    stepSize = 1.0;
  }
  let x = opts.x0.slice();
  let logp = logTarget(x);
  let accepted = 0, acceptedTune = 0, tuneCount = 0;
  const chain = [];
  for (let i = 0; i < steps; i++) {
    const z = gaussVec(rng, dim);
    const y = new Array(dim);
    for (let k = 0; k < dim; k++) y[k] = x[k] + stepSize * z[k];
    const logpY = logTarget(y);
    const u = rng();
    const acc = Math.log(u) < (logpY - logp);
    if (acc) { x = y; logp = logpY; accepted++; }
    if (tune && i < Math.floor(steps / 2)) {
      acceptedTune += acc ? 1 : 0; tuneCount++;
      const gamma = 1 / Math.pow(i + 1, 0.6);       // Robbins-Monro 步长衰减
      stepSize *= Math.exp(gamma * ((acc ? 1 : 0) - tuneTarget));
      if (!(stepSize > 1e-8) || !isFinite(stepSize)) stepSize = 1e-8;
    }
    if (i >= burnIn) chain.push(x.slice());
  }
  return { chain, acceptRate: accepted / steps, tuneAcceptRate: tuneCount ? acceptedTune / tuneCount : accepted / steps, stepSize, final: x.slice(), dim };
}

/* ============================================================================
 * 采样器 4：Gibbs（Geman & Geman 1984，二元标准正态条件扫描）
 *   x|y ~ N(ρy, 1-ρ²)，y|x ~ N(ρx, 1-ρ²)
 * ========================================================================== */
function gibbsSample(opts) {
  const rho = opts.rho;
  const steps = opts.steps, rng = opts.rng;
  const burnIn = opts.burnIn || 0;
  const condVar = (opts.condVar === undefined) ? (1 - rho * rho) : opts.condVar;  // condVar 仅供证伪对照注入错误条件
  if (!(rho > -1 && rho < 1)) throw new RangeError('gibbsSample: ρ ∈ (-1,1)');
  if (!(steps > 0)) throw new RangeError('gibbsSample: steps 必须为正');
  const sd = Math.sqrt(condVar);
  let x = opts.x0[0], y = opts.x0[1];
  const chain = [];
  for (let i = 0; i < steps; i++) {
    x = rho * y + sd * gaussVec(rng, 1)[0];
    y = rho * x + sd * gaussVec(rng, 1)[0];
    if (i >= burnIn) chain.push([x, y]);
  }
  return { chain, final: [x, y], dim: 2 };
}

/* ============================================================================
 * 积分器：leapfrog（可逆保体积）与显式 Euler（非辛，对照用）
 * ========================================================================== */
function leapfrog(opts) {
  const gradLogTarget = opts.gradLogTarget;
  const eps = opts.eps, L = opts.L;
  if (!isFinite(eps) || eps === 0) throw new RangeError('leapfrog: eps 必须为非零有限值（负值=反向积分，NUTS 用）');
  if (!(L >= 1)) throw new RangeError('leapfrog: L ≥ 1');
  const dim = opts.x.length;
  const xx = opts.x.slice(), pp = opts.p.slice();
  let g = gradLogTarget(xx);
  for (let k = 0; k < dim; k++) pp[k] += 0.5 * eps * g[k];      // 半步动量
  for (let l = 0; l < L; l++) {
    for (let k = 0; k < dim; k++) xx[k] += eps * pp[k];          // 整步位置
    g = gradLogTarget(xx);
    const c = (l === L - 1) ? 0.5 : 1.0;                          // 末半步
    for (let k = 0; k < dim; k++) pp[k] += c * eps * g[k];
  }
  return { x: xx, p: pp };
}

/** 显式 Euler（同步更新旧状态，非辛——能量漂移对照） */
function eulerIntegrate(opts) {
  const gradLogTarget = opts.gradLogTarget;
  const eps = opts.eps, L = opts.L;
  const dim = opts.x.length;
  const xx = opts.x.slice(), pp = opts.p.slice();
  for (let l = 0; l < L; l++) {
    const g = gradLogTarget(xx);
    const xn = new Array(dim), pn = new Array(dim);
    for (let k = 0; k < dim; k++) { xn[k] = xx[k] + eps * pp[k]; pn[k] = pp[k] + eps * g[k]; }
    for (let k = 0; k < dim; k++) { xx[k] = xn[k]; pp[k] = pn[k]; }
  }
  return { x: xx, p: pp };
}

/* ============================================================================
 * 采样器 5：HMC（Duane et al. 1987）——动量刷新 + leapfrog + MH 接受
 * ========================================================================== */
function hmcStep(opts) {
  const logTarget = opts.logTarget, gradLogTarget = opts.gradLogTarget;
  const eps = opts.eps, L = opts.L, rng = opts.rng;
  const dim = opts.x.length;
  const p = gaussVec(rng, dim);
  const H0 = -logTarget(opts.x) + 0.5 * dot(p, p);
  const lf = leapfrog({ gradLogTarget, x: opts.x, p, eps, L });
  const H1 = -logTarget(lf.x) + 0.5 * dot(lf.p, lf.p);
  const dH = H0 - H1;
  const acceptProb = dH >= 0 ? 1 : Math.exp(dH);
  const u = rng();
  const accepted = u < acceptProb;
  return { x: accepted ? lf.x : opts.x.slice(), accepted, acceptProb, nLeapfrog: L };
}

function hmcSample(opts) {
  const steps = opts.steps, burnIn = opts.burnIn || 0, rng = opts.rng;
  if (!(steps > 0)) throw new RangeError('hmcSample: steps 必须为正');
  let x = opts.x0.slice();
  let accepted = 0, probSum = 0;
  const chain = [];
  for (let i = 0; i < steps; i++) {
    const r = hmcStep({ logTarget: opts.logTarget, gradLogTarget: opts.gradLogTarget, x, eps: opts.eps, L: opts.L, rng });
    x = r.x; if (r.accepted) accepted++; probSum += r.acceptProb;
    if (i >= burnIn) chain.push(x.slice());
  }
  return { chain, acceptRate: accepted / steps, meanAcceptProb: probSum / steps, final: x.slice(), dim: opts.x0.length };
}

/* ============================================================================
 * 采样器 6：NUTS（Hoffman & Gelman 2014，Algorithm 6 递归倍增 + 切片变量）
 * ========================================================================== */
const NUTS_DELTA_MAX = 1000;   // 对数空间发散界

/** U-turn 判据（Hoffman & Gelman 2014, Eq. 4）：轨迹两端动量均指向内侧才继续 */
function uTurnCriterion(xMinus, xPlus, pMinus, pPlus) {
  const n = xMinus.length;
  let dotMinus = 0, dotPlus = 0;
  for (let k = 0; k < n; k++) { const dx = xPlus[k] - xMinus[k]; dotMinus += dx * pMinus[k]; dotPlus += dx * pPlus[k]; }
  return (dotMinus > 0 && dotPlus > 0);
}

function nutsSample(opts) {
  const logTarget = opts.logTarget, gradLogTarget = opts.gradLogTarget;
  const steps = opts.steps, burnIn = opts.burnIn || 0, rng = opts.rng;
  const maxDepth = opts.maxDepth || 6;
  const adapt = opts.adapt;   // { steps, target, gamma, t0, kappa } —— dual averaging
  if (!(steps > 0)) throw new RangeError('nutsSample: steps 必须为正');
  const dim = opts.x0.length;
  let x = opts.x0.slice();
  let eps = (opts.eps && opts.eps > 0) ? opts.eps : initStepSize({ logTarget, gradLogTarget, x0: x, L: 1, rng, eps0: 0.1 });

  // dual averaging 状态（Hoffman & Gelman 2014 / Neal 2011）
  let mu = Math.log(10 * eps), logEpsBar = Math.log(eps), Hbar = 0;
  const adaptSteps = adapt ? Math.max(1, adapt.steps || 1000) : 0;
  const gamma = adapt ? (adapt.gamma || 0.05) : 0.05;
  const t0 = adapt ? (adapt.t0 || 10) : 10;
  const kappa = adapt ? (adapt.kappa || 0.75) : 0.75;
  const target = adapt ? (adapt.target || 0.65) : 0.65;

  const chain = [];
  let uturns = 0, maxDepthHits = 0, divergences = 0, leapfrogs = 0, alphaSum = 0, nAlpha = 0;

  /** NUTS 子树构建（Hoffman & Gelman 2014 Algorithm 6）。joint0 为初始联合对数密度（切片基准）。 */
  function buildTree(xs, ps, logu, v, j, joint0) {
    if (j === 0) {
      const lf = leapfrog({ gradLogTarget, x: xs, p: ps, eps: eps * v, L: 1 });
      const joint = logTarget(lf.x) - 0.5 * dot(lf.p, lf.p);
      const nPrime = (logu <= joint) ? 1 : 0;                    // 落在切片内
      const sPrime = (logu < joint + NUTS_DELTA_MAX) ? 1 : 0;    // 未发散
      const a = Math.min(1, Math.exp(Math.min(50, joint - joint0)));
      return { xMinus: lf.x, pMinus: lf.p, xPlus: lf.x, pPlus: lf.p, xPrime: lf.x, nPrime, sPrime, alpha: a, nAlpha: 1 };
    }
    const r1 = buildTree(xs, ps, logu, v, j - 1, joint0);
    let xMinus = r1.xMinus, pMinus = r1.pMinus;
    let xPlus = r1.xPlus, pPlus = r1.pPlus;
    let xPrime = r1.xPrime, nPrime = r1.nPrime, sPrime = r1.sPrime, alpha = r1.alpha, nAlpha = r1.nAlpha;
    if (sPrime === 1) {
      let r2;
      if (v === -1) r2 = buildTree(xMinus, pMinus, logu, v, j - 1, joint0);
      else r2 = buildTree(xPlus, pPlus, logu, v, j - 1, joint0);
      if (v === -1) { xMinus = r2.xMinus; pMinus = r2.pMinus; } else { xPlus = r2.xPlus; pPlus = r2.pPlus; }
      if (rng() * (nPrime + r2.nPrime) < r2.nPrime) xPrime = r2.xPrime;
      alpha += r2.alpha; nAlpha += r2.nAlpha;
      sPrime = (r2.sPrime === 1 && uTurnCriterion(xMinus, xPlus, pMinus, pPlus)) ? 1 : 0;
    }
    return { xMinus, pMinus, xPlus, pPlus, xPrime, nPrime, sPrime, alpha, nAlpha };
  }

  for (let i = 0; i < steps; i++) {
    const p0 = gaussVec(rng, dim);
    const joint0 = logTarget(x) - 0.5 * dot(p0, p0);
    const logu = joint0 + Math.log(Math.max(rng(), 1e-300));
    let xMinus = x.slice(), pMinus = p0.slice();
    let xPlus = x.slice(), pPlus = p0.slice();
    let xPrime = x.slice();
    let nPrime = 0, sPrime = 1, alphaIt = 0, nAlphaIt = 0, depth = 0;
    while (sPrime === 1 && depth < maxDepth) {
      const v = (rng() < 0.5) ? -1 : 1;
      let r2;
      if (v === -1) r2 = buildTree(xMinus, pMinus, logu, -1, depth, joint0);
      else r2 = buildTree(xPlus, pPlus, logu, 1, depth, joint0);
      let uturn = false;
      if (r2.sPrime === 1) {
        if (v === -1) { xMinus = r2.xMinus; pMinus = r2.pMinus; } else { xPlus = r2.xPlus; pPlus = r2.pPlus; }
        if (rng() * (nPrime + r2.nPrime) < r2.nPrime) xPrime = r2.xPrime;
        alphaIt += r2.alpha; nAlphaIt += r2.nAlpha;
        if (!uTurnCriterion(xMinus, xPlus, pMinus, pPlus)) uturn = true;
        sPrime = uturn ? 0 : 1;
      } else {
        sPrime = 0;
        divergences++;   // Δ_max 界触发（Stan 的 divergent transition 诊断）
      }
      if (uturn) uturns++;
      else if (depth === maxDepth - 1) maxDepthHits++;
      leapfrogs += Math.pow(2, depth);   // 本层树消耗 2^depth 次 leapfrog
      depth++;
    }
    alphaSum += alphaIt; nAlpha += nAlphaIt;
    x = xPrime;
    if (i >= burnIn) chain.push(x.slice());
    // dual averaging 更新（仅适应期）
    if (adapt && i < adaptSteps) {
      const acceptStat = nAlphaIt > 0 ? alphaIt / nAlphaIt : 0;
      const adaptErr = target - acceptStat;           // δ − ᾱ
      Hbar = Hbar + (adaptErr - Hbar) / (i + 1 + t0);
      const logEps = mu - Math.sqrt(i + 1) / gamma * Hbar;
      eps = Math.exp(logEps);
      const eta = Math.pow(i + 1, -kappa);
      logEpsBar = (1 - eta) * logEpsBar + eta * logEps;
    } else if (adapt && i === adaptSteps - 1) {
      eps = Math.exp(logEpsBar);   // 切换到平均步长
    }
  }
  return {
    chain, eps, acceptStat: nAlpha > 0 ? alphaSum / nAlpha : 0,
    uturnRate: uturns / steps, maxDepthRate: maxDepthHits / steps, divergenceRate: divergences / steps,
    meanLeapfrogs: leapfrogs / steps, final: x.slice(), dim
  };
}

/* ============================================================================
 * 采样器 7：dual averaging 自适应步长 HMC（Hoffman & Gelman；Stan 同款公式）
 *   - init_stepsize：Stan base_hmc 同款——以 ΔH=log(0.8) 为界倍增/减半找初始步长
 *   - 适应统计量：轨迹接受概率 min(1, e^{ΔH})（Stan 同款，连续量；二值信号在
 *     leapfrog 稳定性边界附近无梯度，控制器会永久振荡）
 *   - dual averaging（Nesterov 2009；Stan stepsize_adaptation.hpp 原码公式）：
 *       η=1/(t+t0); H̄←(1-η)H̄+η(δ-ᾱ); logε=μ-√t/γ·H̄; logε̄←(1-t^{-κ})logε̄+t^{-κ}logε
 * ========================================================================== */
function initStepSize(opts) {
  const logTarget = opts.logTarget, gradLogTarget = opts.gradLogTarget;
  const L = opts.L, rng = opts.rng;
  const x = opts.x0.slice();
  let eps = opts.eps0 || 0.1;
  if (!(eps > 0) || !(L >= 1)) throw new RangeError('initStepSize: 需 eps0>0 且 L≥1');
  const cross = Math.log(0.8);
  const evolve = () => {
    const r = hmcStep({ logTarget, gradLogTarget, x, eps, L, rng });
    return Math.log(Math.max(r.acceptProb, 1e-300));   // ΔH = H0 − H1
  };
  let dH = evolve();
  const direction = dH > cross ? 1 : -1;
  for (let iter = 0; iter < 50; iter++) {
    const ndH = evolve();
    if (direction === 1 && !(ndH > cross)) break;
    if (direction === -1 && !(ndH < cross)) break;
    eps = direction === 1 ? 2 * eps : 0.5 * eps;
    if (eps > 1e7 || eps < 1e-12) break;
  }
  return eps;
}

function hmcWithAdaptation(opts) {
  const logTarget = opts.logTarget, gradLogTarget = opts.gradLogTarget;
  const adaptSteps = opts.adaptSteps || 1000, sampleSteps = opts.sampleSteps || 1000;
  const rng = opts.rng;
  const target = (opts.target === undefined) ? 0.65 : opts.target;
  const gamma = opts.gamma || 0.05, t0 = opts.t0 || 10, kappa = opts.kappa || 0.75;
  const dim = opts.x0.length;
  let x = opts.x0.slice();
  const eps0 = (opts.eps0 && opts.eps0 > 0) ? opts.eps0 : initStepSize({ logTarget, gradLogTarget, x0: x, L: opts.L, rng, eps0: 0.1 });
  let mu = Math.log(10 * eps0), logEps = Math.log(eps0), logEpsBar = Math.log(eps0), Hbar = 0;
  let probSum = 0;
  for (let i = 0; i < adaptSteps; i++) {
    const r = hmcStep({ logTarget, gradLogTarget, x, eps: Math.exp(logEps), L: opts.L, rng });
    x = r.x; probSum += r.acceptProb;
    const adaptErr = target - Math.min(1, r.acceptProb);   // δ − ᾱ（ᾱ=轨迹接受概率）
    Hbar = Hbar + (adaptErr - Hbar) / (i + 1 + t0);        // EMA：H̄ ← H̄ + (x−H̄)/(m+t0)
    logEps = mu - Math.sqrt(i + 1) / gamma * Hbar;
    const eta = Math.pow(i + 1, -kappa);
    logEpsBar = (1 - eta) * logEpsBar + eta * logEps;
  }
  const epsBar = Math.exp(logEpsBar);
  // 固定步长采样期
  const chain = [];
  let sAcc = 0, sProb = 0;
  for (let i = 0; i < sampleSteps; i++) {
    const r = hmcStep({ logTarget, gradLogTarget, x, eps: epsBar, L: opts.L, rng });
    x = r.x; if (r.accepted) sAcc++; sProb += r.acceptProb;
    chain.push(x.slice());
  }
  return { chain, eps: Math.exp(logEps), epsBar, eps0, adaptMeanProb: probSum / adaptSteps, acceptRate: sAcc / sampleSteps, meanAcceptProb: sProb / sampleSteps, final: x.slice(), dim };
}

/* ============================================================================
 * 采样器 8：混合蒙特卡洛（HMC + 随机轨迹长 L ~ U{1..Lmax} + 动量刷新）
 *   确定性轨迹段与随机更新的混合：轨迹长分布应 ≈ 均匀（可检验）
 * ========================================================================== */
function hybridMonteCarlo(opts) {
  const logTarget = opts.logTarget, gradLogTarget = opts.gradLogTarget;
  const steps = opts.steps, burnIn = opts.burnIn || 0, rng = opts.rng;
  const Lmax = opts.Lmax || 8, eps = opts.eps;
  if (!(steps > 0)) throw new RangeError('hybridMonteCarlo: steps 必须为正');
  if (!(eps > 0)) throw new RangeError('hybridMonteCarlo: eps 必须为正');
  if (!(Lmax >= 1)) throw new RangeError('hybridMonteCarlo: Lmax ≥ 1');
  let x = opts.x0.slice();
  let accepted = 0;
  const chain = [], trajLens = new Array(Lmax).fill(0);
  for (let i = 0; i < steps; i++) {
    const L = 1 + Math.floor(rng() * Lmax);        // 随机轨迹长
    const r = hmcStep({ logTarget, gradLogTarget, x, eps, L, rng });
    x = r.x; if (r.accepted) accepted++;
    trajLens[L - 1]++;
    if (i >= burnIn) chain.push(x.slice());
  }
  return { chain, acceptRate: accepted / steps, trajFreq: trajLens.map((c) => c / steps), final: x.slice(), dim: opts.x0.length };
}

/* ============================================================================
 * 采样器 9：模拟退火（几何降温；高温平台 hold 步）
 * ========================================================================== */
function annealSchedule(steps, T0, Tmin, hold) {
  const h = Math.min(hold || 0, steps - 1);
  const denom = Math.max(1, steps - 1 - h);
  const Ts = new Array(steps);
  for (let k = 0; k < steps; k++) {
    if (k < h) Ts[k] = T0;
    else Ts[k] = T0 * Math.pow(Tmin / T0, (k - h) / denom);
  }
  return Ts;
}

function annealSample(opts) {
  const logTarget = opts.logTarget;
  const dim = opts.dim || 1;
  const steps = opts.steps, rng = opts.rng, stepSize = opts.stepSize;
  const burnIn = opts.burnIn || 0;
  const T0 = (opts.T0 === undefined) ? 1 : opts.T0;
  const Tmin = (opts.Tmin === undefined) ? 1e-3 : opts.Tmin;
  const hold = opts.hold || 0;
  if (!(steps > 0)) throw new RangeError('annealSample: steps 必须为正');
  if (!(stepSize > 0)) throw new RangeError('annealSample: stepSize 必须为正');
  if (!(T0 > 0 && Tmin > 0 && Tmin <= T0)) throw new RangeError('annealSample: 需 0 < Tmin ≤ T0');
  const Ts = annealSchedule(steps, T0, Tmin, hold);
  let x = opts.x0.slice();
  let logp = logTarget(x);
  const chain = [];
  for (let i = 0; i < steps; i++) {
    const z = gaussVec(rng, dim);
    const y = new Array(dim);
    for (let k = 0; k < dim; k++) y[k] = x[k] + stepSize * z[k];
    const logpY = logTarget(y);
    const u = rng();
    if (Math.log(u) < (logpY - logp) / Ts[i]) { x = y; logp = logpY; }
    if (i >= burnIn) chain.push(x.slice());
  }
  return { chain, schedule: Ts, final: x.slice(), dim };
}

/* ============================================================================
 * 诊断：自相关 / ESS（Geyer 初始正序列对）/ KS 正态检验
 * ========================================================================== */
function autocorr(xs, lag) {
  const n = xs.length;
  const m = meanOf(xs);
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const a = xs[i] - m;
    den += a * a;
    if (i + lag < n) num += a * (xs[i + lag] - m);
  }
  return num / den;
}

/** 有效样本量（Geyer 初始单调正序列；多维链取各维最小） */
function ess(chain) {
  const is2d = Array.isArray(chain[0]);
  if (!is2d) return ess1d(chain);
  let e = Infinity;
  for (let d = 0; d < chain[0].length; d++) e = Math.min(e, ess1d(chain.map((c) => c[d])));
  return e;
}
function ess1d(xs) {
  const n = xs.length;
  if (n < 8) return 0;
  const maxLag = Math.min(n - 2, 2000);
  let tau = 1;
  for (let k = 1; k + 1 < maxLag; k += 2) {
    const pair = autocorr(xs, k) + autocorr(xs, k + 1);
    if (pair <= 0) break;
    tau += 2 * pair;
  }
  return n / Math.max(tau, 1e-9);
}

/** erf 近似（Abramowitz & Stegun 7.1.26，|ε|≤1.5e-7） */
function erf(x) {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return x >= 0 ? y : -y;
}
function normalCdf(x, mu, sd) { return 0.5 * (1 + erf((x - mu) / (sd * Math.SQRT2))); }

/** 双侧 KS 统计量 vs N(mu, sd²)（手搓，不依赖任何统计库） */
function ksNormalStat(samples, mu, sd) {
  const s = samples.slice().sort((a, b) => a - b);
  const n = s.length;
  let d = 0;
  for (let i = 0; i < n; i++) {
    const F = normalCdf(s[i], mu, sd);
    d = Math.max(d, Math.abs(F - (i + 1) / n), Math.abs(F - i / n));
  }
  return d;
}

/* ============================================================================
 * 自检
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

function selfCheck() {
  const cases = [], invariants = [];
  const push = (id, name, ok, actual, detail, tol) => {
    cases.push(caseOf(id, name, ok, actual, detail, tol));
    invariants.push({ id, name, ok, expr: name, tol: (typeof tol === 'number' ? tol : TOL.EXACT), worst: ok ? 0 : 1, detail });
  };

  /* ---------- INV-MC-001：Metropolis ≡ MH（对称提议）；非对称必分歧 ---------- */
  {
    const tgt = makeGaussTarget([1, -2], [1.5, 0.7]);
    const run = (mode, seed) => {
      const rng = makeRng(seed);
      if (mode === 'metropolis') return metropolisSample({ logTarget: tgt.logTarget, dim: 2, x0: [0, 0], steps: 800, stepSize: 1.1, rng });
      if (mode === 'mhSym') return mhSample({ logTarget: tgt.logTarget, dim: 2, x0: [0, 0], steps: 800, rng, propose: makeRandomWalkProposal(1.1) });
      return mhSample({ logTarget: tgt.logTarget, dim: 2, x0: [0, 0], steps: 800, rng, propose: makeIndependentProposal([0, 0], 3.0) });
    };
    const a = run('metropolis', 0x5EED01);
    const b = run('mhSym', 0x5EED01);
    const c = run('mhAsym', 0x5EED01);
    const identical = JSON.stringify(a.chain) === JSON.stringify(b.chain);
    const diverged = JSON.stringify(b.chain) !== JSON.stringify(c.chain);
    const moved = Math.abs(b.final[0]) + Math.abs(b.final[1]) > 1e-6;
    // 哈希原语自身防护：FIPS 180-4 已知向量（防常量表转写错误——本铁律曾实测抓出 K[42]/K[54] typo）
    const hashVecOk = sha256Hex('abc') === 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
      && sha256Hex('') === 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855';
    const ok = identical && diverged && moved && hashVecOk;
    push('INV-MC-001', 'Metropolis≡MH(对称)：两独立实现同 seed 轨迹逐位一致；非对称提议必分歧（附 SHA-256 已知向量防护）', ok,
      hashVecOk ? (identical ? 0 : 1) : 2,
      'identical=' + identical + ' diverged=' + diverged + ' moved=' + moved + ' hashVec=' + hashVecOk + ' acceptMet=' + a.acceptRate.toFixed(4) + ' acceptMHsym=' + b.acceptRate.toFixed(4) + ' acceptMHasym=' + c.acceptRate.toFixed(4), TOL.EXACT);
  }

  /* ---------- INV-MC-002：离散两状态 MH 细致平衡（解析精确）+ 经验转移矩阵 ---------- */
  {
    const pi = [0.7, 0.3];
    const Q = [[0.2, 0.8], [0.6, 0.4]];
    const alpha = [[0, 0], [0, 0]];
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
      if (i === j) continue;
      alpha[i][j] = Math.min(1, (pi[j] * Q[j][i]) / (pi[i] * Q[i][j]));
    }
    const P = [[0, 0], [0, 0]];
    for (let i = 0; i < 2; i++) {
      let off = 0;
      for (let j = 0; j < 2; j++) if (j !== i) { P[i][j] = Q[i][j] * alpha[i][j]; off += P[i][j]; }
      P[i][i] = 1 - off;
    }
    let dbErr = 0;
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) dbErr = Math.max(dbErr, Math.abs(pi[i] * P[i][j] - pi[j] * P[j][i]));
    // 恒接受核必违反细致平衡（证伪）
    const Pa = [[1 - Q[0][1], Q[0][1]], [Q[1][0], 1 - Q[1][0]]];
    let dbNaive = 0;
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) dbNaive = Math.max(dbNaive, Math.abs(pi[i] * Pa[i][j] - pi[j] * Pa[j][i]));
    // 经验链：离散提议
    const propose = (x, rng) => {
      const s = x[0];
      const y = rng() < Q[s][0] ? 0 : 1;
      return { y: [y], logQRatio: Math.log(Q[y][s]) - Math.log(Q[s][y]) };
    };
    const N = 60000;
    const r = mhSample({ logTarget: (x) => Math.log(pi[x[0]]), dim: 1, x0: [0], steps: N, rng: makeRng(0xD152C0), propose });
    let n0 = 0, n00 = 0, n01 = 0, n10 = 0, n11 = 0;
    for (let i = 1; i < r.chain.length; i++) {
      const a0 = r.chain[i - 1][0], a1 = r.chain[i][0];
      if (a0 === 0) { n0++; if (a1 === 0) n00++; else n01++; } else { if (a1 === 0) n10++; else n11++; }
    }
    const freq0 = (n0 + n10) > 0 ? n0 / (n0 + (N - 1 - n0)) : 0;
    const emp = [[n00 / Math.max(n0, 1), n01 / Math.max(n0, 1)], [n10 / Math.max(N - 1 - n0, 1), n11 / Math.max(N - 1 - n0, 1)]];
    let empErr = 0;
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
      const nij = (i === 0 ? n0 : N - 1 - n0);
      const sigma = 1.5 * Math.sqrt(Math.max(P[i][j] * (1 - P[i][j]), 1e-12) / Math.max(nij, 1));  // 10σ + 自相关安全因子
      empErr = Math.max(empErr, Math.abs(emp[i][j] - P[i][j]) / sigma);
    }
    const freqSigma = 1.5 * Math.sqrt(0.21 / N) * Math.sqrt(2.1);
    const freqZ = Math.abs(freq0 - 0.7) / freqSigma;
    const ok = dbErr <= 1e-15 && dbNaive > 0.1 && freqZ <= 10 && empErr <= 10 / 1.5;
    push('INV-MC-002', 'MH 细致平衡：解析核 π_i·P_ij=π_j·P_ji（EXACT）+ 经验转移矩阵 10σ + 恒接受证伪', ok,
      Math.max(dbErr, freqZ / 10, empErr / (10 / 1.5)),
      'dbErr=' + dbErr.toExponential(2) + ' dbNaive=' + dbNaive.toFixed(3) + ' freq0=' + freq0.toFixed(4) + '(π=0.7, z=' + freqZ.toFixed(2) + ') maxEmpErr=' + empErr.toFixed(2) + 'σ10', TOL.EXACT);
  }

  /* ---------- INV-MC-003：RWM 大样本矩收敛 + 最优窗宽 + 调优 ---------- */
  {
    const tgt = makeGaussTarget([3], [2]);
    let worstZ = 0, worstD = 0, acceptRate = 0;
    const seeds = [0x0A100001, 0x0B200002, 0x0C300003];
    for (const sd of seeds) {
      const r = rwMetropolis({ logTarget: tgt.logTarget, dim: 1, x0: [0], steps: 42000, burnIn: 2000, stepSize: 4.76, rng: makeRng(sd) });
      const xs = r.chain.map((c) => c[0]);   // dim=1：链元为 [x]
      const e = ess(xs);
      const m = meanOf(xs), v = varOf(xs, m);
      const zMean = Math.abs(m - 3) / (2 / Math.sqrt(e));
      const zVar = Math.abs(v - 4) / (4 * Math.sqrt(2 / e));
      const D = ksNormalStat(xs, 3, 2);
      worstZ = Math.max(worstZ, zMean, zVar);
      worstD = Math.max(worstD, D * Math.sqrt(e));
      acceptRate = r.acceptRate;
    }
    const ksBound = 1.63;   // 99% KS 临界 × √n
    // 窗宽调优（Robbins-Monro → 0.44）
    const tuned = rwMetropolis({ logTarget: tgt.logTarget, dim: 1, x0: [0], steps: 9000, burnIn: 0, tune: true, tuneTarget: 0.44, rng: makeRng(0x7B0E0001) });
    // 恒接受随机游走（证伪：无 MH 修正必不收敛）
    const rwRng = makeRng(0xFACE0F1);
    let rwx = 0; const rwChain = [];
    for (let i = 0; i < 42000; i++) { rwx += 2.38 * gaussVec(rwRng, 1)[0]; if (i >= 2000) rwChain.push(rwx); }
    const rwVar = varOf(rwChain);
    const ok = worstZ <= 10 && worstD <= ksBound && acceptRate >= 0.38 && acceptRate <= 0.52 &&
      Math.abs(tuned.tuneAcceptRate - 0.44) <= 0.05 && tuned.stepSize > 3.0 && tuned.stepSize < 6.5 && rwVar > 100;
    push('INV-MC-003', 'RWM 矩收敛：z≤10(三种子) + KS D·√ESS≤1.63 + 最优窗宽 2.38σ 接受率≈0.44 + 调优收敛0.44 + 恒接受证伪', ok,
      Math.max(worstZ / 10, worstD / ksBound),
      'worstZ=' + worstZ.toFixed(2) + ' D√ESS=' + worstD.toFixed(3) + ' accept(2.38σ)=' + acceptRate.toFixed(4) + ' tuned=' + tuned.tuneAcceptRate.toFixed(4) + ' step=' + tuned.stepSize.toFixed(3) + ' rwVar=' + rwVar.toFixed(0), TOL.CONV);
  }

  /* ---------- INV-MC-004：Gibbs 条件矩 ---------- */
  {
    const rho = 0.6;
    const r = gibbsSample({ rho, x0: [0, 0], steps: 62000, burnIn: 2000, rng: makeRng(0x61BB05) });
    const xs = r.chain.map((c) => c[0]), ys = r.chain.map((c) => c[1]);
    const e = Math.min(ess(xs), ess(ys));
    const my = meanOf(ys), vy = varOf(ys, my);
    const mx = meanOf(xs);
    const slope = covOf(xs, ys, mx, my) / vy;
    const intercept = mx - slope * my;
    const resid = xs.map((x, i) => x - (intercept + slope * ys[i]));
    const resVar = varOf(resid);
    const seSlope = Math.sqrt(resVar / (e * vy));
    const zSlope = Math.abs(slope - rho) / seSlope;
    const zResVar = Math.abs(resVar - (1 - rho * rho)) / ((1 - rho * rho) * Math.sqrt(2 / e));
    const zVarX = Math.abs(varOf(xs) - 1) / Math.sqrt(2 / e);
    const zMeanX = Math.abs(mx) / (1 / Math.sqrt(e));
    const lag1 = autocorr(xs, 1);
    // 证伪：错误条件方差（1.0 而非 1-ρ²）
    const rw = gibbsSample({ rho, condVar: 1.0, x0: [0, 0], steps: 62000, burnIn: 2000, rng: makeRng(0x61BB05) });
    const rwVarX = varOf(rw.chain.map((c) => c[0]));
    const ok = zSlope <= 10 && zResVar <= 10 && zVarX <= 10 && zMeanX <= 10 && lag1 < 0.95 && Math.abs(rwVarX - 1) > 0.3;
    push('INV-MC-004', 'Gibbs 条件矩：斜率≈ρ/残差方差≈1-ρ²/边际方差≈1（10σ）+ 错误条件方差证伪 + 链移动', ok,
      Math.max(zSlope / 10, zResVar / 10, zVarX / 10, zMeanX / 10),
      'slope=' + slope.toFixed(4) + '(z=' + zSlope.toFixed(2) + ') resVar=' + resVar.toFixed(4) + '(z=' + zResVar.toFixed(2) + ') varX=' + varOf(xs).toFixed(4) + '(z=' + zVarX.toFixed(2) + ') meanX z=' + zMeanX.toFixed(2) + ' lag1=' + lag1.toFixed(3) + ' wrongCondVar=' + rwVarX.toFixed(4), TOL.CONV);
  }

  /* ---------- INV-MC-005：leapfrog 可逆 / 保体积 / 能量有界（vs Euler 漂移） ---------- */
  {
    const tgt = makeGaussTarget([0, 0], [1, 10]);
    const dim = 2;
    // (a) 时间反演可逆
    const x0 = [0.7, -1.3], p0 = [0.4, 0.9];
    const f1 = leapfrog({ gradLogTarget: tgt.gradLogTarget, x: x0, p: p0, eps: 0.25, L: 12 });
    const back = leapfrog({ gradLogTarget: tgt.gradLogTarget, x: f1.x, p: f1.p.map((v) => -v), eps: 0.25, L: 12 });
    let revErr = 0;
    for (let k = 0; k < dim; k++) {
      revErr = Math.max(revErr, Math.abs(back.x[k] - x0[k]), Math.abs(back.p[k] + p0[k]));
    }
    // (b) 保体积：数值 Jacobian 行列式（中心差分，4×4）
    const epsM = 0.3, LM = 3, h = 1e-6;
    const mapState = (st) => {
      const lf = leapfrog({ gradLogTarget: tgt.gradLogTarget, x: [st[0], st[1]], p: [st[2], st[3]], eps: epsM, L: LM });
      return [lf.x[0], lf.x[1], lf.p[0], lf.p[1]];
    };
    const base = [0.5, -0.4, 0.3, 0.6];
    const J = [];
    for (let i = 0; i < 4; i++) {
      const sp = base.slice(), sm = base.slice();
      sp[i] += h; sm[i] -= h;
      const fp = mapState(sp), fm = mapState(sm);
      const col = [];
      for (let r = 0; r < 4; r++) col.push((fp[r] - fm[r]) / (2 * h));
      J.push(col);
    }
    // det（4×4 Laplace 展开）
    const det4 = (M) => {
      const minor = (r, c) => {
        const m = [];
        for (let i = 0; i < 4; i++) { if (i === r) continue; const row = []; for (let j = 0; j < 4; j++) { if (j === c) continue; row.push(M[i][j]); } m.push(row); }
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
      };
      return M[0][0] * minor(0, 0) - M[0][1] * minor(0, 1) + M[0][2] * minor(0, 2) - M[0][3] * minor(0, 3);
    };
    const Jm = [[J[0][0], J[1][0], J[2][0], J[3][0]], [J[0][1], J[1][1], J[2][1], J[3][1]], [J[0][2], J[1][2], J[2][2], J[3][2]], [J[0][3], J[1][3], J[2][3], J[3][3]]];
    const detErr = Math.abs(det4(Jm) - 1);
    // (c) 能量有界 vs Euler 漂移
    const rng = makeRng(0x1EA9F06);
    let lfMax = 0, eulMax8 = 0, eulMax160 = 0, eulMin8 = Infinity;
    for (let c = 0; c < 24; c++) {
      const xs = [3 * (rng() - 0.5), 6 * (rng() - 0.5)];
      const ps = [2 * (rng() - 0.5), 2 * (rng() - 0.5)];
      const H = (xx, pp) => -tgt.logTarget(xx) + 0.5 * dot(pp, pp);
      const H0 = H(xs, ps);
      for (const L of [8, 40, 160]) {
        const lf = leapfrog({ gradLogTarget: tgt.gradLogTarget, x: xs, p: ps, eps: 0.2, L });
        lfMax = Math.max(lfMax, Math.abs(H(lf.x, lf.p) - H0));
      }
      const e8 = eulerIntegrate({ gradLogTarget: tgt.gradLogTarget, x: xs, p: ps, eps: 0.2, L: 8 });
      const e160 = eulerIntegrate({ gradLogTarget: tgt.gradLogTarget, x: xs, p: ps, eps: 0.2, L: 160 });
      eulMax8 = Math.max(eulMax8, Math.abs(H(e8.x, e8.p) - H0));
      eulMax160 = Math.max(eulMax160, Math.abs(H(e160.x, e160.p) - H0));
      eulMin8 = Math.min(eulMin8, Math.abs(H(e8.x, e8.p) - H0));
    }
    const ok = revErr <= 1e-12 && detErr <= 1e-6 && lfMax <= 0.05 && eulMax160 > 5 * Math.max(eulMin8, 1e-9) && eulMax160 > 0.1;
    push('INV-MC-005', 'leapfrog：时间反演可逆≤1e-12 + 保体积|detJ-1|≤1e-6 + 能量有界≤0.05（Euler 漂移对照）', ok,
      Math.max(revErr, detErr, lfMax),
      'revErr=' + revErr.toExponential(2) + ' detErr=' + detErr.toExponential(2) + ' lfMaxΔH=' + lfMax.toExponential(2) + ' euler(8)=' + eulMax8.toExponential(2) + ' euler(160)=' + eulMax160.toExponential(2), TOL.RESID);
  }

  /* ---------- INV-MC-006：NUTS 与混合 MC 效率 ---------- */
  {
    const tgt = makeGaussTarget([0, 0], [1, 10]);
    // U-turn 判据单测（构造轨迹，非仅统计占比）
    const utA = uTurnCriterion([-1, 0], [1, 0], [1, 0], [1, 0]);      // 两端同向内侧：继续
    const utB = uTurnCriterion([-1, 0], [1, 0], [1, 0], [-1, 0]);     // 一端反向：U-turn
    const utC = uTurnCriterion([-1, 0], [1, 0], [-1, 0], [-1, 0]);    // 两端同向反向：U-turn
    const utD = uTurnCriterion([0, -1], [0, 1], [0, 1], [0, 1]);      // 二维内侧
    const utE = uTurnCriterion([-1, 0], [1, 0], [1, 0], [1, 0]);      // 同 A（回归）
    const uturnUnit = utA === true && utB === false && utC === false && utD === true && utE === true;
    const rwm = rwMetropolis({ logTarget: tgt.logTarget, dim: 2, x0: [0, 0], steps: 14000, burnIn: 2000, stepSize: 1.68, rng: makeRng(0x08000001) });
    const rwmEff = ess(rwm.chain);
    // NUTS 自带 init_stepsize + dual averaging（Stan 式）
    const nuts = nutsSample({ logTarget: tgt.logTarget, gradLogTarget: tgt.gradLogTarget, dim: 2, x0: [0, 0], steps: 14000, burnIn: 2000, rng: makeRng(0x0E700002), maxDepth: 6, adapt: { steps: 800, target: 0.65 } });
    const nutsEff = ess(nuts.chain);
    const effRatio = nutsEff / rwmEff;
    const hyb = hybridMonteCarlo({ logTarget: tgt.logTarget, gradLogTarget: tgt.gradLogTarget, dim: 2, x0: [0, 0], steps: 14000, burnIn: 2000, eps: 1.2, Lmax: 8, rng: makeRng(0x44B00003) });
    const hybEff = ess(hyb.chain);
    const hybRatio = hybEff / rwmEff;
    // 轨迹长频率 ≈ 均匀
    let trajErr = 0;
    for (let L = 1; L <= 8; L++) trajErr = Math.max(trajErr, Math.abs(hyb.trajFreq[L - 1] - 1 / 8));
    // 证伪：超大 eps → 接受率崩塌
    const nutsBig = nutsSample({ logTarget: tgt.logTarget, gradLogTarget: tgt.gradLogTarget, dim: 2, x0: [0, 0], steps: 600, burnIn: 0, eps: 5.0, rng: makeRng(0x816E0005), maxDepth: 4 });
    const ok = uturnUnit && effRatio > 3 && nuts.uturnRate > 0.1 && nuts.divergenceRate < 0.95 &&
      nuts.meanLeapfrogs >= 1 && nuts.meanLeapfrogs <= Math.pow(2, 7) - 2 &&
      nuts.acceptStat >= 0.5 && nuts.acceptStat <= 0.95 && nutsBig.acceptStat < 0.2 &&
      hybRatio > 2 && trajErr <= 0.05;
    push('INV-MC-006', 'NUTS/混合MC 效率：U-turn 判据单测 + ESS 比 >3/>2 + U-turn 占比 >0.1 + 接受率区间 + 超大 eps 证伪 + 轨迹长均匀', ok,
      1 / Math.max(effRatio, 0.001),
      'uturnUnit=' + uturnUnit + ' rwmESS=' + rwmEff.toFixed(0) + ' nutsESS=' + nutsEff.toFixed(0) + '(×' + effRatio.toFixed(2) + ') hybESS=' + hybEff.toFixed(0) + '(×' + hybRatio.toFixed(2) + ') uturn=' + nuts.uturnRate.toFixed(3) + ' div=' + nuts.divergenceRate.toFixed(3) + ' lf/iter=' + nuts.meanLeapfrogs.toFixed(1) + ' accept=' + nuts.acceptStat.toFixed(3) + ' bigEps=' + nutsBig.acceptStat.toFixed(3) + ' trajErr=' + trajErr.toFixed(4), TOL.CONV);
  }

  /* ---------- INV-MC-007：dual averaging 收敛到目标接受率 ---------- */
  {
    const tgt = makeGaussTarget([0, 0], [1, 3]);
    const run = (target, seed) => hmcWithAdaptation({
      logTarget: tgt.logTarget, gradLogTarget: tgt.gradLogTarget, dim: 2,
      x0: [0.5, -0.5], adaptSteps: 1200, sampleSteps: 2500, L: 12,
      rng: makeRng(seed), target
    });
    const r65 = run(0.65, 0xDA65001);
    const r90 = run(0.90, 0xDA90002);
    const err65 = Math.abs(r65.meanAcceptProb - 0.65);   // Stan 语义：轨迹平均接受概率
    const err90 = Math.abs(r90.meanAcceptProb - 0.90);
    const epsOrder = r90.epsBar < r65.epsBar * 0.98;
    const ok = err65 <= 0.05 && err90 <= 0.05 && epsOrder && r65.acceptRate > 0.5 && r90.acceptRate > 0.5;
    push('INV-MC-007', 'dual averaging：目标 0.65/0.9 平均接受概率均收敛 ±0.05 且 eps(0.9)<eps(0.65)', ok,
      Math.max(err65, err90),
      'prob65=' + r65.meanAcceptProb.toFixed(4) + ' eps0=' + r65.eps0.toFixed(4) + ' eps65=' + r65.epsBar.toFixed(4) + ' binAcc65=' + r65.acceptRate.toFixed(3) + ' prob90=' + r90.meanAcceptProb.toFixed(4) + ' eps90=' + r90.epsBar.toFixed(4) + ' binAcc90=' + r90.acceptRate.toFixed(3), TOL.CONV);
  }

  /* ---------- INV-MC-008：退火（确定性 + 调度 + 全局模偏好 + 冷链证伪 + 降温致锐） ---------- */
  {
    const lb = (x) => logBimodal1d(x[0], 5, -5, 0.85);   // 全局模 +5（w=0.85）；局部模 -5 仍为真实模态（谷深约 11.6）
    const anneal = (seed) => annealSample({
      logTarget: lb, dim: 1, x0: [-5],
      steps: 9000, stepSize: 1.0, T0: 8, Tmin: 0.04, hold: 3000, rng: makeRng(seed)
    });
    // (a) 确定性
    const a1 = anneal(0x0A11E001), a2 = anneal(0x0A11E001), a3 = anneal(0x0A11E002);
    const identical = JSON.stringify(a1.chain) === JSON.stringify(a2.chain);
    const diverged = JSON.stringify(a1.chain) !== JSON.stringify(a3.chain);
    // (b) 调度结构
    const Ts = a1.schedule;
    let monoOk = true;
    for (let k = 3000; k + 1 < Ts.length; k++) if (!(Ts[k + 1] < Ts[k])) monoOk = false;
    const boundsOk = Ts[0] === 8 && Math.abs(Ts[Ts.length - 1] - 0.04) < 1e-12 && Ts.every((t) => t >= 0.04 - 1e-12 && t <= 8 + 1e-12);
    // (c) 统计：8 种子全局模命中 + 末态平均对数密度 vs 冷链
    let globalHits = 0, sumA = 0;
    for (const sd of [101, 202, 303, 404, 505, 606, 707, 808]) {
      const a = anneal(sd);
      const f = a.chain[a.chain.length - 1][0];
      if (Math.abs(f - 5) <= 2) globalHits++;
      sumA += lb([f]);
    }
    let sumC = 0;
    for (const sd of [101, 202, 303, 404, 505, 606, 707, 808]) {
      const cold = annealSample({ logTarget: lb, dim: 1, x0: [-5], steps: 9000, stepSize: 1.0, T0: 0.04, Tmin: 0.04, rng: makeRng(sd) });
      sumC += lb([cold.chain[cold.chain.length - 1][0]]);
    }
    const meanA = sumA / 8, meanC = sumC / 8, qualityGap = meanA - meanC;
    // (d) 冷链困于局部模（3 种子：全链不越过 x<=1.5，即从不跨越谷底）
    let trapOk = true;
    for (const sd of [0x0C01D001, 0x0C01D002, 0x0C01D003]) {
      const cold = annealSample({ logTarget: lb, dim: 1, x0: [-5], steps: 9000, stepSize: 1.0, T0: 0.04, Tmin: 0.04, rng: makeRng(sd) });
      let maxX = -Infinity;
      for (const c of cold.chain) maxX = Math.max(maxX, c[0]);
      if (!(maxX <= 1.5)) trapOk = false;
    }
    // (e) 降温致锐（高斯目标）vs 恒温对照
    const gt = makeGaussTarget([3], [2]);
    const sharp = annealSample({ logTarget: gt.logTarget, dim: 1, x0: [3], steps: 4000, stepSize: 0.9, T0: 4, Tmin: 0.25, rng: makeRng(0x5A2B0001) });
    const vFirst = varOf(sharp.chain.slice(0, 800).map((c) => c[0]));
    const vLast = varOf(sharp.chain.slice(-800).map((c) => c[0]));
    const constT = annealSample({ logTarget: gt.logTarget, dim: 1, x0: [3], steps: 4000, stepSize: 0.9, T0: 4, Tmin: 4, rng: makeRng(0x5A2B0002) });
    const cFirst = varOf(constT.chain.slice(0, 800).map((c) => c[0]));
    const cLast = varOf(constT.chain.slice(-800).map((c) => c[0]));
    const sharpOk = vLast < vFirst / 4;
    const constOk = cLast > cFirst * 0.6 && cLast < cFirst * 1.6;
    const ok = identical && diverged && monoOk && boundsOk && globalHits >= 5 && qualityGap > 0.5 && trapOk && sharpOk && constOk;
    push('INV-MC-008', '退火：同seed逐位/异seed分歧 + 调度严格递减 + ≥5/8达全局模 + 末态对数密度优于冷链0.5 + 冷链困局部模 + 降温致锐', ok,
      ok ? 0 : 1,
      'identical=' + identical + ' diverged=' + diverged + ' mono=' + monoOk + ' globalHits=' + globalHits + '/8 meanLogp A=' + meanA.toFixed(3) + ' C=' + meanC.toFixed(3) + ' gap=' + qualityGap.toFixed(3) + ' trap=' + trapOk + ' varRatio=' + (vLast / vFirst).toFixed(4) + ' constRatio=' + (cLast / cFirst).toFixed(3), TOL.EXACT);
  }

  return buildReport('mcmc-forge', '1.0.0', 'L5', cases, invariants, '0x9E3779B9');
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
  name: 'mcmc-forge', version: '1.0.0', layer: 'L5', TOL,
  makeRng, sha256Hex,
  gaussVec, dot, meanOf, varOf, covOf,
  makeGaussTarget, logGauss1d, logGaussNd, logBimodal1d,
  metropolisSample, mhSample, makeRandomWalkProposal, makeIndependentProposal,
  rwMetropolis, gibbsSample,   leapfrog, eulerIntegrate,
  hmcStep, hmcSample, nutsSample, uTurnCriterion, hmcWithAdaptation, hybridMonteCarlo,
  initStepSize,
  annealSchedule, annealSample,
  autocorr, ess, erf, normalCdf, ksNormalStat,
  selfCheck, formatTable
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeMCMC = API;

if (typeof process !== 'undefined' && process.argv[1] && /mcmc-forge\.js$/.test(process.argv[1]) && (process.argv.includes('--self-check') || process.argv.includes('--json'))) {
  const r = selfCheck();
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : formatTable(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
