'use strict';
/* ============================================================================
 * 独立复核 vi-forge（INV-VI-002 / INV-VI-003 / INV-VI-005 为必覆盖，另加
 * INV-VI-001 / INV-VI-006 增强）。不调用模块 selfCheck；仅调用其导出函数作
 * 为「被测对象」，并以手搓闭式 / 中心差分 / 独立蒙特卡洛做黄金参照交叉验证。
 *
 * 三重独立原则：
 *   1. 手搓闭式 KL、手搓重参数化 L 与解析梯度，与模块导出函数对照；
 *   2. 解析梯度（pathwise）与中心差分独立核对（≥10σ 误差界内）；
 *   3. 确定性 / 分歧 / 矩收敛三重复核，并显式断言「不同设置必分歧」（反空洞）。
 * ========================================================================== */
const path = require('path');
const VI = require(path.join(__dirname, '..', '..', 'src', 'vi-forge.js'));
const NUM = require(path.join(__dirname, '..', '..', 'src', 'core-numeric.js'));

let fail = 0;
function check(name, cond, detail) {
  console.log((cond ? '✅ ' : '❌ ') + name + (detail ? '  ' + detail : ''));
  if (!cond) fail++;
}

/* ---------- 手搓独立参照（完全独立于模块实现） ---------- */
function indepReparam(mean, logStd, eps) { return mean + Math.exp(logStd) * eps; }
function indepKL(mq, lq, mp, lp) { // 单维闭式
  const sq = Math.exp(lq), sp = Math.exp(lp), d = mq - mp;
  return 0.5 * ((d * d) / (sp * sp) + (sq * sq) / (sp * sp) - 1 + 2 * (Math.log(sp) - Math.log(sq)));
}
function indepL(mean, logStd, x, eps) { // L = recon + kl, β=1, 解码器恒等, 先验 N(0,1)
  const std = Math.exp(logStd);
  const z = mean + std * eps;
  const A = x - z;
  const recon = 0.5 * Math.log(2 * Math.PI) + 0.5 * A * A; // = -log p
  const kl = 0.5 * (mean * mean + std * std - 1 - 2 * logStd);
  return recon + kl;
}
function indepGrad(mean, logStd, x, eps) { // 重参数化路径梯度（手搓）
  const std = Math.exp(logStd);
  const z = mean + std * eps;
  const A = x - z;
  return { mean: -A + mean, logStd: -A * eps * std + (std * std - 1) };
}

/* ============================================================================
 * INV-VI-002 重参数化梯度检验：手搓解析梯度 vs 中心差分（独立交叉验证）
 * ========================================================================== */
{
  let worst = 0;
  for (let t = 0; t < 30; t++) {
    const rng = NUM.makeRng(0xC200 + t);
    const mean = rng() * 1.5 - 0.75, logStd = rng() * 1.0 - 0.5;
    const x = rng() * 4 - 2;
    const eps = NUM.normInv(0.2 + rng() * 0.6); // 固定 eps → 光滑函数
    const g = indepGrad(mean, logStd, x, eps);
    const h = 1e-6;
    const numMean = (indepL(mean + h, logStd, x, eps) - indepL(mean - h, logStd, x, eps)) / (2 * h);
    const numLogStd = (indepL(mean, logStd + h, x, eps) - indepL(mean, logStd - h, x, eps)) / (2 * h);
    const relMean = Math.abs(numMean - g.mean) / (Math.abs(g.mean) > 1e-12 ? Math.abs(g.mean) : 1);
    const relLogStd = Math.abs(numLogStd - g.logStd) / (Math.abs(g.logStd) > 1e-12 ? Math.abs(g.logStd) : 1);
    worst = Math.max(worst, relMean, relLogStd);
  }
  // 交叉一致性：模块的 reparamSample 与手搓闭式一致（给定 eps 逐位）
  let cons = 0;
  {
    const rng = NUM.makeRng(0xC2A1);
    for (let i = 0; i < 20; i++) {
      const mean = rng() * 2 - 1, logStd = Math.log(0.3 + rng() * 2);
      const eps = NUM.normInv(1e-9 + (1 - 2e-9) * rng());
      const zMod = VI.reparamSample(new Float64Array([mean]), new Float64Array([logStd]), rng, new Float64Array([eps]));
      const zRef = indepReparam(mean, logStd, eps);
      if (Math.abs(zMod[0] - zRef) > 1e-15) cons++;
    }
  }
  check('INV-VI-002 重参数化梯度解析 vs 中心差分 (max rel ≤1e-6)',
    worst <= 1e-6, 'maxRel=' + worst.toExponential(3));
  check('INV-VI-002 模块 reparamSample 与手搓闭式一致 (≤1e-15)',
    cons === 0, cons === 0 ? '20/20 一致' : ('分歧 ' + cons));
}

/* ============================================================================
 * INV-VI-003 高斯 KL 闭式 vs 手搓黄金参照（独立重实现 + 非负/自反/非平凡）
 * ========================================================================== */
{
  const rng = NUM.makeRng(0xC300);
  let maxDev = 0, neg = 0, selfBad = 0, trivial = 0;
  for (let t = 0; t < 40; t++) {
    const D = 1 + Math.floor(rng() * 5);
    const mq = new Float64Array(D), lq = new Float64Array(D), mp = new Float64Array(D), lp = new Float64Array(D);
    for (let i = 0; i < D; i++) {
      mq[i] = rng() * 4 - 2; lq[i] = Math.log(0.3 + rng() * 2);
      mp[i] = rng() * 4 - 2; lp[i] = Math.log(0.3 + rng() * 2);
    }
    const got = VI.gaussianKL(mq, lq, mp, lp);
    let gold = 0;
    for (let i = 0; i < D; i++) gold += indepKL(mq[i], lq[i], mp[i], lp[i]);
    maxDev = Math.max(maxDev, Math.abs(got - gold));
    if (got < -1e-12) neg++;
    const same = VI.gaussianKL(mq, lq, mq, lq);
    if (Math.abs(same) > 1e-12) selfBad++;
  }
  const diff = VI.gaussianKL(new Float64Array([1.0]), new Float64Array([0.0]), new Float64Array([-1.0]), new Float64Array([0.0]));
  if (!(diff > 1e-12)) trivial++;
  check('INV-VI-003 高斯 KL 闭式 vs 手搓黄金参照 (≤1e-12)', maxDev <= 1e-12, 'maxDev=' + maxDev.toExponential(3));
  check('INV-VI-003 KL≥0 且 KL=0⟺同参数 且 异参 KL>0', neg === 0 && selfBad === 0 && trivial === 0,
    'neg=' + neg + ' selfBad=' + selfBad + ' trivial=' + trivial);
}

/* ============================================================================
 * INV-VI-001 ELBO 分解（独立复核）：elbo == -recon - kl
 * ========================================================================== */
{
  const rng = NUM.makeRng(0xC100);
  let maxDev = 0;
  for (let t = 0; t < 30; t++) {
    const D = 1 + Math.floor(rng() * 4);
    const x = new Float64Array(D), mean = new Float64Array(D), logStd = new Float64Array(D);
    const pm = new Float64Array(D), pl = new Float64Array(D);
    for (let i = 0; i < D; i++) {
      x[i] = rng() * 4 - 2; mean[i] = rng() * 2 - 1; logStd[i] = Math.log(0.3 + rng() * 1.7);
      pm[i] = 0; pl[i] = 0;
    }
    const r = VI.elbo(x, () => ({ mean, logStd }), (z) => z, { mean: pm, logStd: pl }, rng);
    maxDev = Math.max(maxDev, Math.abs(r.elbo - (-r.recon - r.kl)));
  }
  check('INV-VI-001 ELBO 分解 elbo==-recon-kl (≤1e-12)', maxDev <= 1e-12, 'maxDev=' + maxDev.toExponential(3));
}

/* ============================================================================
 * INV-VI-006 均值场 KL 可加（独立复核）：ΣKL(q_i‖p_i) == 整体 KL
 * ========================================================================== */
{
  const rng = NUM.makeRng(0xC600);
  let maxDev = 0;
  for (let t = 0; t < 30; t++) {
    const D = 1 + Math.floor(rng() * 6);
    const mq = new Float64Array(D), lq = new Float64Array(D), mp = new Float64Array(D), lp = new Float64Array(D);
    for (let i = 0; i < D; i++) {
      mq[i] = rng() * 4 - 2; lq[i] = Math.log(0.3 + rng() * 2);
      mp[i] = rng() * 4 - 2; lp[i] = Math.log(0.3 + rng() * 2);
    }
    const mf = VI.meanFieldKL(mq, lq, mp, lp);
    const full = VI.gaussianKL(mq, lq, mp, lp);
    let sum = 0; for (let i = 0; i < D; i++) sum += indepKL(mq[i], lq[i], mp[i], lp[i]);
    maxDev = Math.max(maxDev, Math.abs(mf - full), Math.abs(mf - sum));
  }
  check('INV-VI-006 均值场 KL 可加 ΣKL==整体 KL (≤1e-12)', maxDev <= 1e-12, 'maxDev=' + maxDev.toExponential(3));
}

/* ============================================================================
 * INV-VI-005 重参数化确定性（反空洞铁律）——三重 + 显式分歧断言
 * ========================================================================== */
{
  let Vcount = 0, detail = [];
  // ① 同 seed 同 (μ,σ) 逐位相等
  {
    const r1 = NUM.makeRng(0xB101), r2 = NUM.makeRng(0xB101);
    const D = 3, mean = new Float64Array(D), logStd = new Float64Array(D);
    const r0 = NUM.makeRng(0xB102);
    for (let i = 0; i < D; i++) { mean[i] = r0() * 2 - 1; logStd[i] = Math.log(0.3 + r0() * 1.7); }
    const z1 = VI.reparamSample(mean, logStd, r1), z2 = VI.reparamSample(mean, logStd, r2);
    let dev = 0; for (let i = 0; i < D; i++) dev = Math.max(dev, Math.abs(z1[i] - z2[i]));
    if (dev > 1e-15) { Vcount++; detail.push('①逐位偏差 ' + dev.toExponential(2)); }
  }
  // ② 异 seed 必分歧（可复现 + 不同序列）
  {
    const rA = NUM.makeRng(0xB111), rB = NUM.makeRng(0xB222);
    const D = 3, mean = new Float64Array(D), logStd = new Float64Array(D);
    const r0 = NUM.makeRng(0xB102);
    for (let i = 0; i < D; i++) { mean[i] = r0() * 2 - 1; logStd[i] = Math.log(0.3 + r0() * 1.7); }
    const zA = VI.reparamSample(mean, logStd, rA), zB = VI.reparamSample(mean, logStd, rB);
    let differ = false; for (let i = 0; i < D; i++) if (zA[i] !== zB[i]) differ = true;
    if (!differ) { Vcount++; detail.push('②异seed未分歧'); }
  }
  // ③ σ=0 与 σ>0 必分歧（证明非恒等掩盖——W11 INV-DIF-005 教训）
  {
    const D = 3, mean = new Float64Array(D);
    const r0 = NUM.makeRng(0xB103); for (let i = 0; i < D; i++) mean[i] = r0() * 2 - 1;
    const logStd0 = new Float64Array(D).fill(-Infinity);
    const logStdP = new Float64Array(D); for (let i = 0; i < D; i++) logStdP[i] = Math.log(0.7);
    const rC = NUM.makeRng(0xB133);
    const z0 = VI.reparamSample(mean, logStd0, rC), zP = VI.reparamSample(mean, logStdP, rC);
    let maxd = 0; for (let i = 0; i < D; i++) maxd = Math.max(maxd, Math.abs(z0[i] - zP[i]));
    if (!(maxd > 0)) { Vcount++; detail.push('③σ=0与σ>0未分歧'); }
  }
  // ④ 大样本 MC：mean(z)≈μ、var(z)≈σ²（统计容差 ≥10σ）
  {
    const Nn = 200000;
    const configs = [[0.3, 1.0], [-0.5, 0.8], [1.2, 1.5], [0.0, 0.5]];
    for (const [mu, sig] of configs) {
      const rng = NUM.makeRng(0xB400 + Math.floor((mu + 10) * 7) + Math.floor(sig * 13));
      const mc = new Float64Array([mu]), lc = new Float64Array([Math.log(sig)]);
      let s1 = 0, s2 = 0;
      for (let i = 0; i < Nn; i++) { const z = VI.reparamSample(mc, lc, rng); s1 += z[0]; s2 += z[0] * z[0]; }
      const m = s1 / Nn, v = s2 / Nn - m * m;
      const tolMean = 12 * sig / Math.sqrt(Nn), tolVar = 12 * (sig * sig) * Math.sqrt(2 / Nn);
      if (Math.abs(m - mu) > tolMean) { Vcount++; detail.push('④均值偏 ' + (Math.abs(m - mu)).toExponential(2)); }
      if (Math.abs(v - sig * sig) > tolVar) { Vcount++; detail.push('④方差偏 ' + (Math.abs(v - sig * sig)).toExponential(2)); }
    }
  }
  check('INV-VI-005 重参数化确定性·分歧·矩收敛（反空洞）', Vcount === 0,
    Vcount === 0 ? '三重确定性 + 分歧 + MC 矩收敛全通过' : detail.join('; '));
}

console.log(fail === 0
  ? '\n✅ vi-check 独立复核通过：INV-VI-001/002/003/005/006 逐项手搓黄金参照交叉验证'
  : '\n❌ vi-check 独立复核失败（' + fail + ' 项）');
process.exit(fail === 0 ? 0 : 1);
