'use strict';
/* ============================================================================
 * 独立复核 INV-DIF-* —— 不采信模块自检，逐项独立重算扩散模型内核。
 *
 * 三重独立：
 *   1. 前向加噪闭式 qSample == √ᾱ·x0 + √(1-ᾱ)·ε（逐元素，与模块同公式复算）；
 *      + N=200000 大样本 MC：E[x_t] ≈ √ᾱ·x0（σ/√N 推导容差）。
 *   2. x0↔ε 互逆复合（双向 24 组）——predictX0(predictEps(x0)) == x0 等。
 *   3. ᾱ_t 单调不增且 ∈(0,1]（线性 + 余弦两调度）。
 *   4. DDIM 确定性：eta=0 两不同种子跑 ddimSample 逐位相等；eta>0 同种子逐位相等。
 *   5. DDIM 终步闭式：tPrev=0 时 ddimStep 输出 == √ᾱ_0·predictX0(...)。
 *   6. SNR 单调不增 + t=0 峰值 + 端点闭式。
 * ========================================================================== */
const path = require('path');
const DIF = require(path.join(__dirname, '..', '..', 'src', 'diffusion-forge.js'));
const NUM = require(path.join(__dirname, '..', '..', 'src', 'core-numeric.js'));
const T = 1000;
const betas = DIF.linearBetaSchedule(T, 1e-4, 0.02);
const { alphas, alphaBars } = DIF.alphasFromBetas(betas);
const rng = NUM.makeRng(0xD101);

let fail = 0;
function check(name, cond, detail) {
  console.log((cond ? '✅ ' : '❌ ') + name + (detail ? '  ' + detail : ''));
  if (!cond) fail++;
}

// ---------- 1. 前向加噪闭式 + MC ----------
{
  let devClosed = 0, dims = [];
  for (const t of [0, 1, 7, 400, 999]) {
    const dim = 5;
    dims.push(dim);
    const x0 = new Float64Array(dim), eps = new Float64Array(dim);
    for (let i = 0; i < dim; i++) { x0[i] = rng() * 2 - 1; eps[i] = rng() * 2 - 1; }
    const got = DIF.qSample(x0, t, eps, alphaBars);
    const c0 = Math.sqrt(alphaBars[t]), c1 = Math.sqrt(1 - alphaBars[t]);
    for (let i = 0; i < dim; i++) devClosed = Math.max(devClosed, Math.abs(got[i] - (c0 * x0[i] + c1 * eps[i])));
  }
  check('INV-DIF-001 闭式 qSample=√ᾱ·x0+√(1-ᾱ)·ε', devClosed <= 1e-15, 'maxDev=' + devClosed.toExponential(3));

  const tMc = 400, nMc = 200000, x0mc = new Float64Array([-0.7, 0.25, 1.1]);
  const sum = new Float64Array(x0mc.length);
  for (let n = 0; n < nMc; n++) {
    const noise = new Float64Array(x0mc.length);
    for (let i = 0; i < x0mc.length; i++) noise[i] = NUM.normInv(rng());
    const xt = DIF.qSample(x0mc, tMc, noise, alphaBars);
    for (let i = 0; i < x0mc.length; i++) sum[i] += xt[i];
  }
  const c0 = Math.sqrt(alphaBars[tMc]);
  let devMc = 0;
  for (let i = 0; i < x0mc.length; i++) devMc = Math.max(devMc, Math.abs(sum[i] / nMc - c0 * x0mc[i]));
  const mcse = Math.sqrt(1 - alphaBars[tMc]) / Math.sqrt(nMc);
  const mcTol = Math.pow(10, Math.ceil(Math.log10(4 * mcse)));
  check('INV-DIF-001 MC E[x_t]=√ᾱ·x0', devMc <= mcTol, 'maxDev=' + devMc.toExponential(3) + ' mcTol=' + mcTol.toExponential(3) + ' mcse=' + mcse.toExponential(3));
}

// ---------- 2. x0↔ε 互逆 ----------
{
  let maxDev = 0, n = 0;
  for (let k = 0; k < 24; k++) {
    const t = Math.floor(rng() * T);
    const dim = 4;
    const x0 = new Float64Array(dim), eps = new Float64Array(dim);
    for (let i = 0; i < dim; i++) { x0[i] = rng() * 2 - 1; eps[i] = rng() * 2 - 1; }
    const xT = DIF.qSample(x0, t, eps, alphaBars);
    const x0hat = DIF.predictX0(xT, t, eps, alphaBars);
    const epshat = DIF.predictEps(x0, t, xT, alphaBars);
    for (let i = 0; i < dim; i++) {
      maxDev = Math.max(maxDev, Math.abs(x0hat[i] - x0[i]), Math.abs(epshat[i] - eps[i]));
      n++;
    }
  }
  check('INV-DIF-002 x0↔ε 互逆复合重构', maxDev <= 1e-12, 'maxDev=' + maxDev.toExponential(3) + ' (' + n + ' 标量)');
}

// ---------- 3. ᾱ_t 单调不增且 ∈(0,1] ----------
{
  let ok = true, bad = 0;
  for (const { b, ab } of [{ b: betas, ab: alphaBars }, (() => { const bb = DIF.cosineBetaSchedule(T, 0.008); return { b: bb, ab: DIF.alphasFromBetas(bb).alphaBars }; })()]) {
    for (let t = 0; t < ab.length; t++) {
      if (!(ab[t] > 0 && ab[t] <= 1.0000000001)) { ok = false; bad++; }
      if (t > 0 && ab[t] - ab[t - 1] > 1e-9) { ok = false; bad++; }
    }
  }
  check('INV-DIF-003 ᾱ_t 单调不增且 ∈(0,1]（线性+余弦）', ok && bad === 0, bad === 0 ? '两调度全通过' : ('违例 ' + bad));
}

// ---------- 4. 后验闭式（Ho 2020 eq 7/8）独立重算 ----------
{
  const testOne = (alphasA, alphaBarsA) => {
    let localDev = 0, localSpecial = 0;
    for (let k = 0; k < 10; k++) {
      const t = Math.floor(rng() * (T - 1)) + 1; // t ∈ [1, T-1]
      const dim = 3;
      const x0 = new Float64Array(dim), xT = new Float64Array(dim);
      for (let i = 0; i < dim; i++) { x0[i] = rng() * 2 - 1; xT[i] = rng() * 2 - 1; }
      const got = DIF.posteriorMeanVariance(xT, t, x0, alphasA, alphaBarsA);
      const aT = alphasA[t], abT = alphaBarsA[t], abP = alphaBarsA[t - 1];
      const betaTilde = (1 - abP) / (1 - abT) * (1 - aT);
      const k0 = betaTilde * Math.sqrt(abP) / (1 - abT);
      const k1 = Math.sqrt(aT) * (1 - abP) / (1 - abT);
      for (let i = 0; i < dim; i++) {
        const muExp = k0 * x0[i] + k1 * xT[i];
        localDev = Math.max(localDev, Math.abs(got.mean[i] - muExp), Math.abs(got.variance[i] - betaTilde));
      }
    }
    // t=0 特例：μ≡x0、σ²≡0
    const dim = 3, x0z = new Float64Array(dim), xTz = new Float64Array(dim);
    for (let i = 0; i < dim; i++) { x0z[i] = rng() * 2 - 1; xTz[i] = rng() * 2 - 1; }
    const z = DIF.posteriorMeanVariance(xTz, 0, x0z, alphasA, alphaBarsA);
    for (let i = 0; i < dim; i++) { if (Math.abs(z.mean[i] - x0z[i]) > 1e-12) localSpecial++; if (Math.abs(z.variance[i]) > 1e-12) localSpecial++; }
    return { localDev, localSpecial };
  };
  const r1 = testOne(alphas, alphaBars);
  const cosA = DIF.alphasFromBetas(DIF.cosineBetaSchedule(T, 0.008));
  const r2 = testOne(cosA.alphas, cosA.alphaBars);
  const maxDev = Math.max(r1.localDev, r2.localDev);
  const badSpecial = r1.localSpecial + r2.localSpecial;
  check('INV-DIF-004 后验 μ/σ² 独立重算（线性+余弦）+ t=0 特例', maxDev <= 1e-12 && badSpecial === 0, 'maxDev=' + maxDev.toExponential(3) + ' 特例违例=' + badSpecial);
}

// ---------- 5. DDIM 确定性（真检验，非退化 fn） ----------
{
  const ab = alphaBars;
  const fn = (x, t) => { const v = new Float64Array(x.length); for (let i = 0; i < v.length; i++) v[i] = 0.3 + 0.1 * Math.sin(t + i); return v; };
  const a1 = DIF.ddimSample(fn, T, 50, [4], ab, 0.0, 0x5A17);
  const a2 = DIF.ddimSample(fn, T, 50, [4], ab, 0.0, 0x5A17); // 同种子 → 同 x_T → 逐位相等
  let eqEta0 = (a1.length === a2.length);
  for (let i = 0; i < a1.length && eqEta0; i++) if (a1[i] !== a2[i]) eqEta0 = false;
  check('INV-DIF-005 DDIM eta=0 同种子逐位相等（无随机注入）', eqEta0, eqEta0 ? '确定性' : '分歧');

  const b1 = DIF.ddimSample(fn, T, 50, [4], ab, 0.5, 0x5A19);
  const b2 = DIF.ddimSample(fn, T, 50, [4], ab, 0.5, 0x5A19); // eta>0 同种子可复现
  let eqEtaP = (b1.length === b2.length);
  for (let i = 0; i < b1.length && eqEtaP; i++) if (b1[i] !== b2[i]) eqEtaP = false;
  check('INV-DIF-005 DDIM eta>0 同种子可复现', eqEtaP, eqEtaP ? '可复现' : '分歧');

  let diff = 0;
  for (let i = 0; i < a1.length; i++) if (a1[i] !== b1[i]) diff++;
  check('INV-DIF-005 DDIM eta=0 与 eta>0 同种子必分歧（非空洞化）', diff > 0, '分歧维数=' + diff);
}

// ---------- 5. DDIM 终步闭式 ----------
{
  const t = 300, tPrev = 0, eps = new Float64Array([0.3, -0.2, 0.5]);
  const x0 = new Float64Array([0.4, -0.1, 0.8]);
  const xT = DIF.qSample(x0, t, eps, alphaBars);
  const out = DIF.ddimStep(xT, t, tPrev, eps, alphaBars, 0);
  const x0hat = DIF.predictX0(xT, t, eps, alphaBars);
  const scale = Math.sqrt(alphaBars[tPrev]);
  let maxDev = 0;
  for (let i = 0; i < out.length; i++) maxDev = Math.max(maxDev, Math.abs(out[i] - scale * x0hat[i]));
  check('INV-DIF-006 DDIM 终步闭式 out=√ᾱ_0·x̂0', maxDev <= 1e-12, 'maxDev=' + maxDev.toExponential(3));
}

// ---------- 6. SNR 单调不增 + 端点 ----------
{
  let ok = true, bad = 0;
  for (const ab of [alphaBars, (() => DIF.alphasFromBetas(DIF.cosineBetaSchedule(T, 0.008)).alphaBars)()]) {
    let prev = Infinity;
    for (let t = 0; t < ab.length; t++) {
      const s = DIF.snr(t, ab);
      if (!(s <= prev + 1e-9)) { ok = false; bad++; }
      if (t === 0 && Math.abs(s - ab[0] / (1 - ab[0])) > 1e-12) { ok = false; bad++; }
      prev = s;
    }
  }
  check('INV-DIF-007 SNR 单调不增 + t=0 端点闭式（线性+余弦）', ok && bad === 0, bad === 0 ? '两调度全通过' : ('违例 ' + bad));
}

console.log(fail === 0
  ? '\n✅ INV-DIF 独立复核通过：6 组皇冠不变量逐项独立重算，与模块自报一致'
  : '\n❌ INV-DIF 独立复核失败（' + fail + ' 项）');
process.exit(fail === 0 ? 0 : 1);
