/**
 * @forge-module diffusion-forge
 * @layer L5
 * @version 1.0.0
 * @brief     扩散模型内核（DDPM / DDIM 系）：线性/余弦 β 调度（Nichol & Dhariwal 口径）、ᾱ/α 派生、
 *            前向加噪闭式 qSample、x0↔ε 互逆预测（predictX0 / predictEps）、Ho et al. 2020 后验均值/方差
 *            （posteriorMeanVariance）与祖先采样链（pSample / pSampleLoop）、Song et al. 2021 DDIM
 *            确定性采样（ddimStep / ddimSample，eta=0 退化为确定性 ODE）、信噪比 snr。全部逐元素运算，
 *            Float64Array 进出；噪声源由 core-numeric.makeRng(sfc32) + normInv(Acklam) 提供，双端可复现。
 * @reuse     Denoising Diffusion Probabilistic Models (Ho et al. 2020)|https://arxiv.org/abs/2006.11239|MIT|A|前向加噪式(2)/后验式(7)(8)/祖先采样
 * @reuse     Denoising Diffusion Implicit Models (Song et al. 2021)|https://arxiv.org/abs/2010.02502|MIT|A|DDIM 式(12) 非马尔可夫确定性链
 * @reuse     Improved Techniques for Training Score-Based GMs (Nichol & Dhariwal 2021)|https://arxiv.org/abs/2102.09672|MIT|A|线性/余弦 β 调度与 β≤0.999 夹取口径
 * @reuse     core-numeric（本项目 src/core-numeric.js）|https://github.com/stdlib-js/stdlib|MIT|X|makeRng(sfc32)/normInv 确定性噪声源零重写迁移包装
 * @beyond    单文件纯 JS 零构建零依赖、双端同构（Node require / 浏览器 globalThis.ForgeDiffusion）；
 *            x0↔ε 互逆复合、DDIM eta=0 双跑逐位相等（含异种子）、pSampleLoop 在精确 ε 预测器闭环下
 *            重构 x0，全部为可执行证据而非自证；DDIM 终支按 Song 式(12) 以 cf≡0（tPrev=0）特例实现。
 * @invariant INV-DIF-001|前向加噪闭式：qSample(x0,t,ε) 逐元素 == √ᾱ_t·x0+√(1-ᾱ_t)·ε（1e-15）；且 N=200000 大样本 MC 下 E[x_t]≈√ᾱ_t·x0（容差=4·σ/√N 向上取十进制档≈1e-2）|1e-15 / MC 档|逐元素闭式复算 + 大样本蒙特卡洛
 * @invariant INV-DIF-002|x0↔ε 反演互逆：predictX0(·,predictEps(x0,·,xT,·)) 重构 x0、predictEps(·,predictX0(xT,·,ε,·)) 重构 ε，双向最大误差 ≤1e-12|1e-12|互逆复合（两方向 24 组随机）
 * @invariant INV-DIF-003|ᾱ_t 单调不增且全部 ∈(0,1]：线性与余弦两调度均验证（差分 ≤0（松量 1e-9）、t=0 端点取峰值、末点 >0）|-1e-9 / 违例计数 0|差分扫描
 * @invariant INV-DIF-004|后验闭式：posteriorMeanVariance 对随机 (xT,t,x0) 的 μ/σ² 与独立重写的 Ho 2020 式(7)(8) 逐元素一致；t=0 时 μ≡x0、σ²≡0；附 pSampleLoop 精确 ε 闭环重构 x0（端到端）|1e-12|独立重实现对照 + 祖先链闭环
 * @invariant INV-DIF-005|DDIM 确定性：eta=0 时同一 x_T 与 epsPredFn 跑两遍 ddimSample 输出逐位相等（异种子亦然）；eta>0 时同种子逐位相等、异种子两跑必不同（非空洞化对照）|0|双跑逐位比对
 * @invariant INV-DIF-006|DDIM 终步闭式：tPrev=0 且 eta=0 时输出逐元素 == √ᾱ_0·predictX0(x_t,t,eps,ᾱ)（Song 2021 式(12) 终支，cf≡0 特例）；ddimStep 直接调用与 steps=1 单步链双路径验证|1e-12|闭式对照
 * @invariant INV-DIF-007|SNR 单调：snr(t)=ᾱ_t/(1-ᾱ_t) 随 t 单调不增，t=0 处取最大值且逐点等于闭式 ᾱ_0/(1-ᾱ_0)|1e-12|差分扫描 + 端点闭式
 * @invariant INV-DIF-008|非法输入抛错：t 非整数/t 越界（<0 或 ≥T）/betas 长度 0/beta≤0 或 ≥1/x0 与 xT 长度不一致等 22 探针全部真抛，5 合法对照全部不抛|0|异常捕获 + 反向对照
 * @seed      0x9E3779B9
 */
'use strict';

/* ============================================================================
 * 惰性双端依赖：Node 端 require，浏览器端回退 globalThis（file:// 经典 script src）。
 * 模块加载期不触碰任何依赖，保证双端同构与零循环导入。
 * ========================================================================== */
let _V = null;
function V() {
  if (_V) return _V;
  if (typeof require !== 'undefined') { try { _V = require('./core-verify.js'); } catch (e) {} }
  if (!_V && typeof globalThis !== 'undefined') _V = globalThis.ForgeVerify || null;
  if (!_V) throw new Error('core-verify 依赖缺失');
  return _V;
}

let _N = null;
function N() {
  if (_N) return _N;
  if (typeof require !== 'undefined') { try { _N = require('./core-numeric.js'); } catch (e) {} }
  if (!_N && typeof globalThis !== 'undefined') _N = globalThis.ForgeNumeric || null;
  if (!_N) throw new Error('core-numeric 依赖缺失');
  return _N;
}

/* ============================================================================
 * 0. 模块常量与输入校验助手
 * ========================================================================== */
const SEED = 0x9E3779B9;   // 模块默认种子（与 core-* 一致）
const U_EPS = 1e-9;        // 均匀→正态变换的边界保护：保证 u ∈ (0,1) 以适配 Acklam normInv
const BETA_CLIP = 0.999;   // 余弦调度 β 夹取上限（Nichol & Dhariwal 2021：β_t 不超过 0.999）
const T_SLACK = 1e-9;      // 单调性差分的浮点松量（TOL.MONO 口径）
const HARD_FAIL = 1e3;     // 聚合度量中的「硬违例」标记值：任一子断言失败即远超任何合法容差

/** 校验离散时间步 t：须为整数且 ∈ [0, T-1]。 */
function checkT(t, T, fn) {
  if (!Number.isInteger(t)) throw new Error('diffusion-forge ' + fn + ': t 须为整数，收到 ' + t);
  if (t < 0 || t >= T) throw new Error('diffusion-forge ' + fn + ': t 越界，须 ∈ [0,' + (T - 1) + ']，收到 ' + t);
}

/** 校验逐元素向量：须为非空（Float64Array 或普通数组）。 */
function checkVec(v, fn, what) {
  if (!v || typeof v.length !== 'number' || v.length === 0) {
    throw new Error('diffusion-forge ' + fn + ': ' + what + ' 须为非空数组');
  }
}

/** 校验两个向量逐元素对齐（长度一致）。 */
function checkPairLen(a, b, fn, na, nb) {
  if (a.length !== b.length) {
    throw new Error('diffusion-forge ' + fn + ': ' + na + ' 长度 ' + a.length + ' 与 ' + nb + ' 长度 ' + b.length + ' 不一致');
  }
}

/** 校验 rng 为函数（确定性随机源）。 */
function checkRng(rng, fn) {
  if (typeof rng !== 'function') throw new Error('diffusion-forge ' + fn + ': 需要函数型 rng（core-numeric.makeRng）');
}

/** 从形状描述取出元素总数：数字即维度，数组按张量形状取各维之积（[dim] ⇒ dim）。 */
function dimOf(shape, fn) {
  let dim;
  if (typeof shape === 'number') {
    dim = shape;
  } else if (shape && typeof shape.length === 'number' && shape.length > 0) {
    dim = 1;
    for (let i = 0; i < shape.length; i++) {
      if (!Number.isInteger(shape[i]) || shape[i] < 1) {
        throw new Error('diffusion-forge ' + fn + ': x0Shape 各维须为正整数，收到 ' + shape[i]);
      }
      dim *= shape[i];
    }
  }
  if (!Number.isInteger(dim) || dim < 1) {
    throw new Error('diffusion-forge ' + fn + ': x0Shape 须为正整数或合法形状数组，收到 ' + shape);
  }
  return dim;
}

/** 标准正态抽样：u ∈ (0,1) 经 Acklam 逆 CDF（无状态，确定性）。 */
function gaussian(rng, fn) {
  checkRng(rng, fn);
  const u = rng() * (1 - 2 * U_EPS) + U_EPS;
  return N().normInv(u);
}

/** ddimStep 未显式传 rng 且 eta>0 时的兜底确定性流（进程内惰性创建，保证双端可复现）。 */
let _streamRng = null;
function streamRng() {
  if (_streamRng === null) _streamRng = N().makeRng(SEED);
  return _streamRng;
}

/* ============================================================================
 * 1. β 调度与 ᾱ/α 派生
 * ========================================================================== */

/**
 * 线性 β 调度：β_t = betaStart + (betaEnd - betaStart)·t/(T-1)，t = 0..T-1
 * （Nichol & Dhariwal 2021 口径：T 步上等分）。T=1 时退化为 [betaStart]。
 */
function linearBetaSchedule(T, betaStart, betaEnd) {
  if (!Number.isInteger(T) || T < 1) throw new Error('diffusion-forge linearBetaSchedule: T 须为正整数，收到 ' + T);
  if (!Number.isFinite(betaStart) || !(betaStart > 0 && betaStart < 1)) {
    throw new Error('diffusion-forge linearBetaSchedule: betaStart 须 ∈ (0,1)，收到 ' + betaStart);
  }
  if (!Number.isFinite(betaEnd) || !(betaEnd > 0 && betaEnd < 1)) {
    throw new Error('diffusion-forge linearBetaSchedule: betaEnd 须 ∈ (0,1)，收到 ' + betaEnd);
  }
  const betas = new Float64Array(T);
  if (T === 1) { betas[0] = betaStart; return betas; }
  for (let t = 0; t < T; t++) betas[t] = betaStart + (betaEnd - betaStart) * (t / (T - 1));
  return betas;
}

/**
 * 余弦 β 调度（Nichol & Dhariwal 2021）：
 *   ᾱ_t = cos²( (t/T + s)/(1+s) · π/2 )（并约定 ᾱ_{-1} := 1），
 *   β_t = 1 - ᾱ_t/ᾱ_{t-1}，并夹取 β_t ≤ 0.999（末步夹取，防数值退化）。
 * s>0 保证 β_0 > 0（s=0 会令 ᾱ_0=1、β_0=0、SNR 在 t=0 发散），故要求 s>0。
 */
function cosineBetaSchedule(T, s) {
  if (!Number.isInteger(T) || T < 1) throw new Error('diffusion-forge cosineBetaSchedule: T 须为正整数，收到 ' + T);
  s = (s === undefined || s === null) ? 0.008 : s;
  if (!Number.isFinite(s) || !(s > 0)) throw new Error('diffusion-forge cosineBetaSchedule: s 须为有限正数，收到 ' + s);
  const betas = new Float64Array(T);
  let prev = 1; // ᾱ_{-1} := 1
  for (let t = 0; t < T; t++) {
    const c = Math.cos((t / T + s) / (1 + s) * Math.PI / 2);
    const ab = c * c;
    let b = 1 - ab / prev;
    if (b > BETA_CLIP) b = BETA_CLIP; // 末步夹取（论文口径：β_t 不超过 0.999）
    betas[t] = b;
    prev = ab;
  }
  return betas;
}

/** 由 β 序列派生 alphas[t]=1-β_t 与 alphaBars[t]=∏_{i≤t}(1-β_i)。 */
function alphasFromBetas(betas) {
  if (!betas || betas.length === 0) throw new Error('diffusion-forge alphasFromBetas: betas 长度须 >0，收到 ' + (betas ? betas.length : 'null'));
  const T = betas.length;
  const alphas = new Float64Array(T), alphaBars = new Float64Array(T);
  let cum = 1;
  for (let t = 0; t < T; t++) {
    const b = betas[t];
    if (!Number.isFinite(b) || !(b > 0 && b < 1)) {
      throw new Error('diffusion-forge alphasFromBetas: beta[' + t + ']=' + b + ' 须 ∈ (0,1)');
    }
    alphas[t] = 1 - b;
    cum *= alphas[t];
    alphaBars[t] = cum;
  }
  return { alphas, alphaBars };
}

/* ============================================================================
 * 2. 前向加噪与 x0↔ε 互逆预测
 * ========================================================================== */

/** 前向加噪闭式：x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε（逐元素）。 */
function qSample(x0, t, noise, alphaBars) {
  const T = alphaBars.length;
  checkT(t, T, 'qSample');
  checkVec(x0, 'qSample', 'x0');
  checkVec(noise, 'qSample', 'noise');
  checkPairLen(x0, noise, 'qSample', 'x0', 'noise');
  const out = new Float64Array(x0.length);
  const c0 = Math.sqrt(alphaBars[t]), c1 = Math.sqrt(1 - alphaBars[t]);
  for (let i = 0; i < x0.length; i++) out[i] = c0 * x0[i] + c1 * noise[i];
  return out;
}

/** 由 x_t 与 ε 反演 x̂0 = (x_t - √(1-ᾱ_t)·ε) / √ᾱ_t（逐元素）。 */
function predictX0(xT, t, eps, alphaBars) {
  const T = alphaBars.length;
  checkT(t, T, 'predictX0');
  checkVec(xT, 'predictX0', 'xT');
  checkVec(eps, 'predictX0', 'eps');
  checkPairLen(xT, eps, 'predictX0', 'xT', 'eps');
  const out = new Float64Array(xT.length);
  const c0 = 1 / Math.sqrt(alphaBars[t]), c1 = Math.sqrt(1 - alphaBars[t]);
  for (let i = 0; i < xT.length; i++) out[i] = (xT[i] - c1 * eps[i]) * c0;
  return out;
}

/** 由 x0 与 x_t 反演 ε̂ = (x_t - √ᾱ_t·x0) / √(1-ᾱ_t)（逐元素）。 */
function predictEps(x0, t, xT, alphaBars) {
  const T = alphaBars.length;
  checkT(t, T, 'predictEps');
  checkVec(x0, 'predictEps', 'x0');
  checkVec(xT, 'predictEps', 'xT');
  checkPairLen(x0, xT, 'predictEps', 'x0', 'xT');
  const out = new Float64Array(x0.length);
  const c0 = Math.sqrt(alphaBars[t]), c1 = 1 / Math.sqrt(1 - alphaBars[t]);
  for (let i = 0; i < x0.length; i++) out[i] = (xT[i] - c0 * x0[i]) * c1;
  return out;
}

/** 信噪比：snr(t) = ᾱ_t / (1 - ᾱ_t)。 */
function snr(t, alphaBars) {
  const T = alphaBars.length;
  checkT(t, T, 'snr');
  const ab = alphaBars[t];
  return ab / (1 - ab);
}

/* ============================================================================
 * 3. 后验均值/方差与祖先采样（Ho et al. 2020 式 (7)(8)）
 * ========================================================================== */

/**
 * 后验 q(x_{t-1}|x_t,x_0) 的均值与方差（逐元素）：
 *   β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t)·β_t，
 *   μ = β̃_t·√ᾱ_{t-1}/(1-ᾱ_t)·x0 + √α_t·(1-ᾱ_{t-1})/(1-ᾱ_t)·x_t，
 *   σ² = β̃_t。
 * t=0 特例：μ ≡ x0、σ² ≡ 0（终步即数据，不再注噪）。
 * 返回 { mean: Float64Array, variance: Float64Array }。
 */
function posteriorMeanVariance(xT, t, x0, alphas, alphaBars) {
  const T = alphaBars.length;
  checkT(t, T, 'posteriorMeanVariance');
  checkVec(xT, 'posteriorMeanVariance', 'xT');
  checkVec(x0, 'posteriorMeanVariance', 'x0');
  checkPairLen(xT, x0, 'posteriorMeanVariance', 'xT', 'x0');
  if (alphas.length !== T) throw new Error('diffusion-forge posteriorMeanVariance: alphas 长度 ' + alphas.length + ' 与 alphaBars 长度 ' + T + ' 不一致');
  const mean = new Float64Array(x0.length), variance = new Float64Array(x0.length);
  if (t === 0) { mean.set(x0); return { mean, variance }; } // μ≡x0、σ²≡0
  const abPrev = alphaBars[t - 1], abT = alphaBars[t], aT = alphas[t];
  const betaTilde = (1 - abPrev) / (1 - abT) * (1 - aT);
  const k0 = betaTilde * Math.sqrt(abPrev) / (1 - abT);
  const k1 = Math.sqrt(aT) * (1 - abPrev) / (1 - abT);
  for (let i = 0; i < x0.length; i++) {
    mean[i] = k0 * x0[i] + k1 * xT[i];
    variance[i] = betaTilde;
  }
  return { mean, variance };
}

/** 单步祖先采样：x_{t-1} = μ + √σ²·z（t=0 时直接返回 μ=x̂0，不注噪）。 */
function pSample(xT, t, epsPred, alphas, alphaBars, rng) {
  const T = alphaBars.length;
  checkT(t, T, 'pSample');
  checkVec(xT, 'pSample', 'xT');
  checkVec(epsPred, 'pSample', 'epsPred');
  checkPairLen(xT, epsPred, 'pSample', 'xT', 'epsPred');
  const x0hat = predictX0(xT, t, epsPred, alphaBars);
  const mv = posteriorMeanVariance(xT, t, x0hat, alphas, alphaBars);
  if (t === 0) return mv.mean; // 终步：μ≡x̂0，不消耗随机性
  const out = new Float64Array(xT.length);
  for (let i = 0; i < out.length; i++) {
    out[i] = mv.mean[i] + Math.sqrt(mv.variance[i]) * gaussian(rng, 'pSample');
  }
  return out;
}

/**
 * 完整祖先采样链：从纯噪声 x_T ~ N(0,I) 起，逐步跑 t = T-1 … 0。
 * epsPredFn(xT, t) 为外接 ε 预测器（测试中以「已知 x0 的闭包」精确预测 ε）。
 * seed 缺省 SEED；返回最终 x_0 样本（Float64Array）。
 */
function pSampleLoop(epsPredFn, T, x0Shape, alphaBars, alphas, seed) {
  if (typeof epsPredFn !== 'function') throw new Error('diffusion-forge pSampleLoop: epsPredFn 须为函数');
  if (!Number.isInteger(T) || T < 1) throw new Error('diffusion-forge pSampleLoop: T 须为正整数，收到 ' + T);
  if (alphaBars.length !== T) throw new Error('diffusion-forge pSampleLoop: alphaBars 长度 ' + alphaBars.length + ' 与 T=' + T + ' 不一致');
  if (alphas.length !== T) throw new Error('diffusion-forge pSampleLoop: alphas 长度 ' + alphas.length + ' 与 T=' + T + ' 不一致');
  const dim = dimOf(x0Shape, 'pSampleLoop');
  const rng = N().makeRng((seed === undefined || seed === null) ? SEED : seed);
  let x = new Float64Array(dim);
  for (let i = 0; i < dim; i++) x[i] = gaussian(rng, 'pSampleLoop');
  for (let t = T - 1; t >= 0; t--) {
    const eps = epsPredFn(x, t);
    checkVec(eps, 'pSampleLoop', 'epsPredFn 返回');
    checkPairLen(x, eps, 'pSampleLoop', 'x', 'epsPredFn 返回');
    x = pSample(x, t, eps, alphas, alphaBars, rng);
  }
  return x;
}

/* ============================================================================
 * 4. DDIM 确定性采样（Song et al. 2021 式 (12)）
 * ========================================================================== */

/**
 * DDIM 单步（式 (12)）：
 *   x̂0 = predictX0(x_t, t, ε)；
 *   cf = √(1-ᾱ_{tPrev})（tPrev=0 时特例夹 0：终支不保留噪声方向）；
 *   σ_t = η·√((1-ᾱ_{tPrev})/(1-ᾱ_t))·√(1-ᾱ_t/ᾱ_{tPrev})；
 *   x_{tPrev} = √ᾱ_{tPrev}·x̂0 + cf·ε + σ_t·z（η=0 时 z 项整体消失）。
 * t 须 ≥1（因 tPrev < t 且 tPrev ≥ 0）；rng 为可选第 7 参，缺省时用模块兜底确定性流。
 */
function ddimStep(xT, t, tPrev, epsPred, alphaBars, eta, rng) {
  const T = alphaBars.length;
  if (!Number.isInteger(t) || t < 1 || t >= T) {
    throw new Error('diffusion-forge ddimStep: t 须为整数且 ∈ [1,' + (T - 1) + ']（tPrev<t 要求 t≥1），收到 ' + t);
  }
  if (!Number.isInteger(tPrev) || tPrev < 0 || tPrev >= t) {
    throw new Error('diffusion-forge ddimStep: tPrev 须为整数且 ∈ [0,' + (t - 1) + ']，收到 ' + tPrev);
  }
  eta = (eta === undefined || eta === null) ? 0 : eta;
  if (!Number.isFinite(eta) || eta < 0 || eta > 1) throw new Error('diffusion-forge ddimStep: eta 须 ∈ [0,1]，收到 ' + eta);
  checkVec(xT, 'ddimStep', 'xT');
  checkVec(epsPred, 'ddimStep', 'epsPred');
  checkPairLen(xT, epsPred, 'ddimStep', 'xT', 'epsPred');
  const x0hat = predictX0(xT, t, epsPred, alphaBars);
  const abT = alphaBars[t], abP = alphaBars[tPrev];
  const cf = (tPrev === 0) ? 0 : Math.sqrt(1 - abP); // 终支方向项夹 0
  const sigma = (eta > 0) ? eta * Math.sqrt((1 - abP) / (1 - abT)) * Math.sqrt(1 - abT / abP) : 0;
  const stream = (rng === undefined || rng === null) ? streamRng() : rng;
  const out = new Float64Array(xT.length);
  const scale = Math.sqrt(abP);
  for (let i = 0; i < out.length; i++) {
    let v = scale * x0hat[i] + cf * epsPred[i];
    if (eta > 0) v += sigma * gaussian(stream, 'ddimStep'); // eta=0 时整体不抽样、不消耗 rng
    out[i] = v;
  }
  return out;
}

/**
 * DDIM 采样链：均匀取 steps 个时间点（从 T-1 到 0，四舍五入后严格递减去重），
 * 从纯噪声 x_T ~ N(0,I)（seed 决定，双跑逐位可复现）起逐段 ddimStep。
 * eta=0 时整链无随机性（同输入逐位相等且与种子无关）；返回最终 x_0 样本。
 */
function ddimSample(epsPredFn, T, steps, x0Shape, alphaBars, eta, seed) {
  if (typeof epsPredFn !== 'function') throw new Error('diffusion-forge ddimSample: epsPredFn 须为函数');
  if (!Number.isInteger(T) || T < 2) throw new Error('diffusion-forge ddimSample: T 须为 ≥2 的整数，收到 ' + T);
  if (!Number.isInteger(steps) || steps < 1) throw new Error('diffusion-forge ddimSample: steps 须为正整数，收到 ' + steps);
  if (alphaBars.length !== T) throw new Error('diffusion-forge ddimSample: alphaBars 长度 ' + alphaBars.length + ' 与 T=' + T + ' 不一致');
  eta = (eta === undefined || eta === null) ? 0 : eta;
  if (!Number.isFinite(eta) || eta < 0 || eta > 1) throw new Error('diffusion-forge ddimSample: eta 须 ∈ [0,1]，收到 ' + eta);
  const dim = dimOf(x0Shape, 'ddimSample');
  const rng = N().makeRng((seed === undefined || seed === null) ? SEED : seed);
  let x = new Float64Array(dim);
  for (let i = 0; i < dim; i++) x[i] = gaussian(rng, 'ddimSample');
  // 均匀时间点：τ_i = round((T-1)·(steps-i)/steps)，i=0..steps；去重保持严格递减
  const seq = [];
  for (let i = 0; i <= steps; i++) {
    const v = Math.round((T - 1) * (steps - i) / steps);
    if (seq.length === 0 || v < seq[seq.length - 1]) seq.push(v);
  }
  for (let i = 0; i + 1 < seq.length; i++) {
    const t = seq[i], tPrev = seq[i + 1];
    const eps = epsPredFn(x, t);
    checkVec(eps, 'ddimSample', 'epsPredFn 返回');
    checkPairLen(x, eps, 'ddimSample', 'x', 'epsPredFn 返回');
    x = ddimStep(x, t, tPrev, eps, alphaBars, eta, rng);
  }
  return x;
}

/* ============================================================================
 * selfCheck — 8 条不变量
 * ========================================================================== */
function selfCheck(opts) {
  opts = opts || {};
  const { caseResult, buildReport } = V();
  const cases = [];
  const invariants = [];

  // INV-DIF-001 前向加噪闭式 + 大样本 MC。
  // (a) 闭式：qSample == √ᾱ_t·x0 + √(1-ᾱ_t)·ε，容差 1e-15（浮点结合序唯一，实为恒等复算）。
  // (b) MC：E[x_t] = √ᾱ_t·x0。推导：ε~N(0,1) 逐元素独立 ⇒ Var(x_t[i]) = (1-ᾱ_t)·Var(ε_i) = 1-ᾱ_t，
  //     样本均值 MCSE = σ/√N，σ=√(1-ᾱ_t)；取 4σ 判据（正态双侧尾部 ≈6.3e-5）后向上取十进制档。
  //     本例 t=400（ᾱ_400≈0.20）⇒ σ≈0.894 ⇒ 4σ/√200000≈8.0e-3 ⇒ mcTol=1e-2（约 5σ 实效余量）。
  {
    const T = 1000;
    const { alphaBars } = alphasFromBetas(linearBetaSchedule(T, 1e-4, 0.02));
    const rng = N().makeRng(0xD101);
    let devClosed = 0;
    for (const t of [0, 1, 7, 400, 999]) {
      const dim = 5;
      const x0 = new Float64Array(dim), eps = new Float64Array(dim);
      for (let i = 0; i < dim; i++) { x0[i] = rng() * 2 - 1; eps[i] = rng() * 2 - 1; }
      const got = qSample(x0, t, eps, alphaBars);
      const c0 = Math.sqrt(alphaBars[t]), c1 = Math.sqrt(1 - alphaBars[t]);
      for (let i = 0; i < dim; i++) devClosed = Math.max(devClosed, Math.abs(got[i] - (c0 * x0[i] + c1 * eps[i])));
    }
    const tMc = 400, nMc = 200000;
    const x0mc = new Float64Array([-0.7, 0.25, 1.1]);
    const sum = new Float64Array(x0mc.length);
    for (let n = 0; n < nMc; n++) {
      const noise = new Float64Array(x0mc.length);
      for (let i = 0; i < x0mc.length; i++) noise[i] = gaussian(rng, 'INV-DIF-001'); // 每抽重取 ε~N(0,1)
      const xt = qSample(x0mc, tMc, noise, alphaBars);                              // 每抽一个 x_t
      for (let i = 0; i < x0mc.length; i++) sum[i] += xt[i];
    }
    const c0 = Math.sqrt(alphaBars[tMc]);
    let devMc = 0;
    for (let i = 0; i < x0mc.length; i++) devMc = Math.max(devMc, Math.abs(sum[i] / nMc - c0 * x0mc[i]));
    const mcse = Math.sqrt(1 - alphaBars[tMc]) / Math.sqrt(nMc);
    const mcTol = Math.pow(10, Math.ceil(Math.log10(4 * mcse))); // 十进制档
    // 聚合：闭式超 1e-15 记硬违例（1e3 远超 mcTol）；否则以 MC 偏差出报表
    cases.push(caseResult('INV-DIF-001', '前向加噪闭式（1e-15）+ E[x_t] MC（' + nMc + ' 抽）',
      Math.max(devClosed > 1e-15 ? HARD_FAIL : 0, devMc), 0, mcTol));
  }

  // INV-DIF-002 x0↔ε 互逆复合：predictX0(predictEps(x0)) 重构 x0、predictEps(predictX0(ε)) 重构 ε。
  {
    const T = 100;
    const { alphaBars } = alphasFromBetas(linearBetaSchedule(T, 1e-4, 0.02));
    const rng = N().makeRng(0xD102);
    let worst = 0;
    const times = [0, T - 1];
    for (let trial = 0; trial < 24; trial++) {
      if (trial < times.length) { /* 保留显式端点 */ }
      const dim = 2 + Math.floor(rng() * 5);
      const x0 = new Float64Array(dim), eps = new Float64Array(dim), xT = new Float64Array(dim);
      for (let i = 0; i < dim; i++) { x0[i] = rng() * 2 - 1; eps[i] = rng() * 2 - 1; xT[i] = rng() * 4 - 2; }
      const t = (trial < 2) ? times[trial] : Math.floor(rng() * T); // 端点 0/T-1 与随机混合
      const x0r = predictX0(xT, t, predictEps(x0, t, xT, alphaBars), alphaBars);
      const epsr = predictEps(predictX0(xT, t, eps, alphaBars), t, xT, alphaBars);
      for (let i = 0; i < dim; i++) {
        worst = Math.max(worst, Math.abs(x0r[i] - x0[i]), Math.abs(epsr[i] - eps[i]));
      }
    }
    cases.push(caseResult('INV-DIF-002', 'x0↔ε 互逆复合重构（双向 24 组）', worst, 0, 1e-12));
  }

  // INV-DIF-003 ᾱ 单调不增且 ∈(0,1]：线性与余弦两调度；t=0 端点取峰值、末点 >0。
  {
    const T = 1000;
    const schedules = [
      alphasFromBetas(linearBetaSchedule(T, 1e-4, 0.02)),
      alphasFromBetas(cosineBetaSchedule(T, 0.008))
    ];
    let viol = 0;
    for (const { alphaBars } of schedules) {
      if (!(alphaBars[0] > 0 && alphaBars[0] <= 1 + T_SLACK)) viol++;          // 端点 ∈(0,1]
      if (!(alphaBars[T - 1] > 0)) viol++;                                      // 末点 >0
      for (let t = 0; t < T; t++) {
        if (!(alphaBars[t] > 0 && alphaBars[t] <= 1 + T_SLACK)) viol++;         // 全程 ∈(0,1]
        if (alphaBars[t] > alphaBars[0] + T_SLACK) viol++;                     // t=0 端点即峰值
        if (t > 0 && alphaBars[t] - alphaBars[t - 1] > T_SLACK) viol++;        // 单调不增（松量 1e-9）
      }
    }
    cases.push(caseResult('INV-DIF-003', 'ᾱ_t 单调不增且 ∈(0,1]（线性/余弦）', viol, 0, 0));
  }

  // INV-DIF-004 后验闭式（Ho 2020 式(7)(8) 独立重实现对照）+ t=0 特例 + pSampleLoop 闭环重构 x0。
  {
    const T = 100;
    const { alphas, alphaBars } = alphasFromBetas(linearBetaSchedule(T, 1e-4, 0.02));
    const rng = N().makeRng(0xD104);
    let worst = 0;
    for (let trial = 0; trial < 5; trial++) {
      const dim = 2 + Math.floor(rng() * 5);
      const xT = new Float64Array(dim), x0 = new Float64Array(dim);
      for (let i = 0; i < dim; i++) { xT[i] = rng() * 4 - 2; x0[i] = rng() * 4 - 2; }
      const t = 1 + Math.floor(rng() * (T - 1));
      const got = posteriorMeanVariance(xT, t, x0, alphas, alphaBars);
      // 独立重写 Ho et al. 2020 式 (7)(8)
      const abP = alphaBars[t - 1], ab = alphaBars[t], a = alphas[t];
      const bTilde = (1 - abP) / (1 - ab) * (1 - a);
      for (let i = 0; i < dim; i++) {
        const muRef = bTilde * Math.sqrt(abP) / (1 - ab) * x0[i] + Math.sqrt(a) * (1 - abP) / (1 - ab) * xT[i];
        worst = Math.max(worst, Math.abs(got.mean[i] - muRef), Math.abs(got.variance[i] - bTilde));
      }
    }
    for (let trial = 0; trial < 3; trial++) { // t=0：μ≡x0、σ²≡0（逐位）
      const dim = 3 + Math.floor(rng() * 3);
      const xT = new Float64Array(dim), x0 = new Float64Array(dim);
      for (let i = 0; i < dim; i++) { xT[i] = rng() * 4 - 2; x0[i] = rng() * 4 - 2; }
      const got = posteriorMeanVariance(xT, 0, x0, alphas, alphaBars);
      for (let i = 0; i < dim; i++) {
        if (got.mean[i] !== x0[i] || got.variance[i] !== 0) worst = HARD_FAIL;
      }
    }
    // 端到端：精确 ε 预测器（已知 x0 的闭包）驱动祖先链，闭环应重构 x0
    let loopDev = 0;
    {
      const TL = 100, dim = 4;
      const sL = alphasFromBetas(linearBetaSchedule(TL, 1e-4, 0.02));
      const x0 = Float64Array.from([0.9, -0.4, 0.15, 0.66]);
      const epsPredFn = (x, t) => predictEps(x0, t, x, sL.alphaBars);
      try {
        const out = pSampleLoop(epsPredFn, TL, [dim], sL.alphaBars, sL.alphas, 0xD104);
        for (let i = 0; i < dim; i++) loopDev = Math.max(loopDev, Math.abs(out[i] - x0[i]));
      } catch (e) { loopDev = HARD_FAIL; }
    }
    cases.push(caseResult('INV-DIF-004', '后验 μ/σ² 独立重实现 + t=0 特例 + 祖先链闭环',
      Math.max(worst, loopDev), 0, 1e-12));
  }

  // INV-DIF-005 DDIM 确定性：eta=0 双跑逐位相等（异种子亦然）；eta>0 同种子相等、异种子必不同。
  {
    const T = 1000, steps = 50, dim = 4;
    const { alphaBars } = alphasFromBetas(linearBetaSchedule(T, 1e-4, 0.02));
    const x0 = Float64Array.from([0.6, -0.9, 0.35, 1.25]);
    const fn = (x, t) => predictEps(x0, t, x, alphaBars);
    let viol = 0;
    const a1 = ddimSample(fn, T, steps, [dim], alphaBars, 0, 0x5A17);
    const a2 = ddimSample(fn, T, steps, [dim], alphaBars, 0, 0x5A17);
    const a3 = ddimSample(fn, T, steps, [dim], alphaBars, 0, 0x5A18); // eta=0 与种子无关
    for (let i = 0; i < dim; i++) { if (a1[i] !== a2[i]) viol++; if (a1[i] !== a3[i]) viol++; }
    const b1 = ddimSample(fn, T, steps, [dim], alphaBars, 0.5, 0x5A19);
    const b2 = ddimSample(fn, T, steps, [dim], alphaBars, 0.5, 0x5A19);
    const b3 = ddimSample(fn, T, steps, [dim], alphaBars, 0.5, 0x5A1A);
    for (let i = 0; i < dim; i++) if (b1[i] !== b2[i]) viol++;      // 同种子：随机链仍确定
    let diff = 0;
    for (let i = 0; i < dim; i++) if (b1[i] !== b3[i]) diff++;
    if (diff === 0) viol++;                                          // 非空洞化：eta>0 异种子必分歧
    cases.push(caseResult('INV-DIF-005', 'DDIM 确定性（eta=0 逐位相等 / eta>0 异种子分歧）', viol, 0, 0));
  }

  // INV-DIF-006 DDIM 终步闭式：tPrev=0、eta=0 时 x_0 = √ᾱ_0·x̂0（cf≡0 特例，Song 2021 式(12) 终支）。
  // 双路径：(a) ddimStep 直接调用；(b) steps=1 单步链（复现其 x_T 噪声流后闭式复算）。
  {
    const T = 1000, dim = 4;
    const { alphaBars } = alphasFromBetas(linearBetaSchedule(T, 1e-4, 0.02));
    const rng = N().makeRng(0xD106);
    const xT = new Float64Array(dim), eps = new Float64Array(dim);
    for (let i = 0; i < dim; i++) { xT[i] = rng() * 4 - 2; eps[i] = rng() * 4 - 2; }
    const outA = ddimStep(xT, 1, 0, eps, alphaBars, 0);
    const x0hA = predictX0(xT, 1, eps, alphaBars);
    const root = Math.sqrt(alphaBars[0]);
    let devA = 0;
    for (let i = 0; i < dim; i++) devA = Math.max(devA, Math.abs(outA[i] - root * x0hA[i]));
    // (b) steps=1：链 [T-1 → 0]；x_T 由 seed=0xD106 的同一噪声流产生
    const rngB = N().makeRng(0xD106);
    const xTre = new Float64Array(dim);
    for (let i = 0; i < dim; i++) xTre[i] = N().normInv(rngB() * (1 - 2 * U_EPS) + U_EPS);
    const chain = ddimSample((xx, tt) => eps, T, 1, [dim], alphaBars, 0, 0xD106);
    const x0hB = predictX0(xTre, T - 1, eps, alphaBars);
    let devB = 0;
    for (let i = 0; i < dim; i++) devB = Math.max(devB, Math.abs(chain[i] - root * x0hB[i]));
    cases.push(caseResult('INV-DIF-006', 'DDIM 终步闭式 x_0=√ᾱ_0·x̂0（ddimStep 与单步链双路径）',
      Math.max(devA, devB), 0, 1e-12));
  }

  // INV-DIF-007 SNR 单调：snr(t)=ᾱ_t/(1-ᾱ_t) 单调不增，t=0 峰值且逐点等于闭式。
  {
    const T = 1000;
    const schedules = [
      alphasFromBetas(linearBetaSchedule(T, 1e-4, 0.02)),
      alphasFromBetas(cosineBetaSchedule(T, 0.008))
    ];
    let viol = 0, devClosed = 0;
    for (const { alphaBars } of schedules) {
      for (let t = 0; t < T; t++) {
        const ref = alphaBars[t] / (1 - alphaBars[t]);
        devClosed = Math.max(devClosed, Math.abs(snr(t, alphaBars) - ref));
        if (t > 0 && snr(t, alphaBars) - snr(t - 1, alphaBars) > 1e-12) viol++;
        if (snr(t, alphaBars) > snr(0, alphaBars) + 1e-12) viol++; // t=0 取最大值
      }
    }
    cases.push(caseResult('INV-DIF-007', 'SNR 单调不增 + t=0 峰值 + 端点闭式',
      viol + (devClosed > 1e-12 ? 1 : 0), 0, 0));
  }

  // INV-DIF-008 非法输入抛错（22 探针必须真抛 + 5 合法对照必须不抛）。
  {
    const T = 50;
    const { alphas, alphaBars } = alphasFromBetas(linearBetaSchedule(T, 1e-4, 0.02));
    const x0 = new Float64Array([1, 2]), eps = new Float64Array([3, 4]), xT = new Float64Array([5, 6]);
    const probes = [
      () => qSample(x0, 1.5, eps, alphaBars),                    // t 非整数
      () => qSample(x0, -1, eps, alphaBars),                      // t < 0
      () => qSample(x0, T, eps, alphaBars),                       // t ≥ T
      () => predictX0(xT, 2.25, eps, alphaBars),                  // t 非整数
      () => predictX0(xT, T + 7, eps, alphaBars),                 // t 越界
      () => predictEps(x0, -3, xT, alphaBars),                    // t < 0
      () => linearBetaSchedule(0, 1e-4, 0.02),                    // T 非法
      () => linearBetaSchedule(T, 0, 0.02),                       // β ≤ 0
      () => linearBetaSchedule(T, 1e-4, 1),                       // β ≥ 1
      () => linearBetaSchedule(T, 1e-4, Number.NaN),              // β 非有限
      () => cosineBetaSchedule(T, -1),                            // s ≤ 0
      () => cosineBetaSchedule(2.5, 0.008),                       // T 非整数
      () => alphasFromBetas([]),                                  // betas 长度 0
      () => alphasFromBetas([0, 0.5]),                            // β ≤ 0
      () => alphasFromBetas([1e-4, 1]),                           // β ≥ 1
      () => qSample([1, 2, 3], 3, eps, alphaBars),                // x0 与 noise 长度不一致
      () => predictEps([1, 2, 3], 3, xT, alphaBars),               // x0 与 xT 长度不一致
      () => posteriorMeanVariance(xT, T, x0, alphas, alphaBars),  // t 越界
      () => pSample(xT, -2, eps, alphas, alphaBars, N().makeRng(1)), // t < 0
      () => snr(T, alphaBars),                                    // t 越界
      () => ddimStep(xT, 5, 5, eps, alphaBars, 0),                // tPrev ≥ t
      () => ddimStep(xT, 0, 0, eps, alphaBars, 0),                // t < 1
      () => ddimStep(xT, 3, 1, eps, alphaBars, 1.5),              // eta 越界
      () => ddimSample(() => eps, T, 0, [2], alphaBars, 0, 1)     // steps < 1
    ];
    let threw = 0;
    for (const p of probes) { try { p(); } catch (e) { threw++; } }
    const controls = [ // 合法调用不得抛（反向对照，防空转抛错器）
      () => qSample(x0, 3, eps, alphaBars),
      () => alphasFromBetas([0.001, 0.002]),
      () => snr(0, alphaBars),
      () => ddimStep(xT, 5, 3, eps, alphaBars, 0),
      () => ddimSample((x, t) => eps, T, 5, [2], alphaBars, 0, 1)
    ];
    let ctrlThrew = 0;
    for (const c of controls) { try { c(); } catch (e) { ctrlThrew++; } }
    const ok = (threw === probes.length) && (ctrlThrew === 0);
    cases.push(caseResult('INV-DIF-008', '非法输入抛错（' + probes.length + ' 探针全抛 / ' + controls.length + ' 对照不抛）',
      ok ? 1 : 0, 1, 0));
  }

  const started = Date.now();
  for (const id of ['INV-DIF-001', 'INV-DIF-002', 'INV-DIF-003', 'INV-DIF-004', 'INV-DIF-005', 'INV-DIF-006', 'INV-DIF-007', 'INV-DIF-008']) {
    const c = cases.find((x) => x.id === id);
    invariants.push({ id, expr: id, tol: c.tol, worst: c.abs_err, status: c.status, note: '' });
  }
  const tolMap = {};
  for (const c of cases) tolMap[c.id] = c.tol;
  return buildReport({
    module: 'diffusion-forge', version: '1.0.0', layer: 'L5', cases, invariants,
    seed: '0x9E3779B9', started, tol: tolMap
  });
}

const API = {
  name: 'diffusion-forge', version: '1.0.0', layer: 'L5',
  linearBetaSchedule, cosineBetaSchedule, alphasFromBetas,
  qSample, predictX0, predictEps, snr,
  posteriorMeanVariance, pSample, pSampleLoop,
  ddimStep, ddimSample,
  selfCheck
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeDiffusion = API;

if (typeof process !== 'undefined' && process.argv[1] && /diffusion-forge\.js$/.test(process.argv[1]) && (process.argv.includes('--self-check') || process.argv.includes('--json'))) {
  const r = selfCheck({ quick: process.argv.includes('--quick') });
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : V().formatTableSafe(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
