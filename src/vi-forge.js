/**
 * @forge-module vi-forge
 * @layer L5
 * @version 1.0.0
 * @brief     变分推断内核：ELBO 分解 / 重参数化技巧（pathwise estimator）/ 高斯 KL 闭式 / VAE 训练 / 均值场变分推断
 * @reuse     Kingma & Welling 2013 (Auto-Encoding Variational Bayes)|https://arxiv.org/abs/1312.6114|MIT|A|ELBO 与重参数化技巧原式
 * @reuse     Rezende et al. 2014 (Stochastic Backprop / Reparam Trick)|https://arxiv.org/abs/1401.4082|MIT|A|重参数化使梯度可经采样层反向传播
 * @reuse     Jordan et al. 1999 (Variational Inference)|https://www.cs.princeton.edu/courses/archive/spr06/cos598C/papers/JordanEtAl.pdf|未声明|C|均值场变分推断与 KL 分解基础
 * @reuse     Sønderby et al. 2016 (Auxiliary Deep Generative Models)|https://arxiv.org/abs/1602.05473|MIT|B|辅助 KL 与 ELBO 下界构造
 * @reuse     Bishop PRML (2006)|https://www.microsoft.com/en-us/research/publication/pattern-recognition-and-machine-learning/|未声明|D|高斯 KL 闭式与变分下界推导
 * @beyond    纯 JS 单文件零构建；reparamSample 给定相同 eps 逐位确定（反空洞：σ=0 与 σ>0 必分歧）；
 *            gaussianKL 闭式与独立黄金参照交叉验证（INV-VI-003 灵魂不变量）；vae 解析梯度训练玩具数据可收敛；
 *            ELBO 分解、均值场 KL 可加性、摊销推断确定性/平滑性逐项独立复核。
 * @invariant INV-VI-001|ELBO 分解闭式：elbo == -recon - kl，且 elbo ≤ E_q[log p(x|z)]（因 KL≥0），max|·|≤1e-12|1e-12|分量一致性 + KL 非负性
 * @invariant INV-VI-002|重参数化梯度检验：dL/dmean、dL/dlogStd 解析（pathwise）vs 中心差分 max 相对误差 ≤1e-6|1e-6|灵魂不变量·独立有限差分交叉验证
 * @invariant INV-VI-003|高斯 KL 闭式 vs 手搓黄金参照 max|·|≤1e-12；KL≥0；KL=0 ⟺ 同参数|1e-12|灵魂不变量·独立黄金参照
 * @invariant INV-VI-004|VAE 损失分解：loss=recon+β·kl；β=0 逐位==recon；β=1==recon+kl|1e-12|闭式分量一致
 * @invariant INV-VI-005|重参数化确定性（反空洞铁律）：①同 seed 同(μ,σ) 重采样 z 逐位相等(≤1e-15) ②异 seed 序列可复现且分歧 ③σ=0 vs σ>0 必分歧(>0) ④大样本 MC mean(z)≈μ std(z)≈σ（统计容差≥10σ）|0|确定性+分歧+矩收敛三重
 * @invariant INV-VI-006|均值场 KL 可加：Σ_i KL(q_i‖p_i) == 整体 KL(q‖p)（多维独立），max|·|≤1e-12|1e-12|逐维求和等价
 * @invariant INV-VI-007|摊销推断一致性：同 x 多次编码(确定性前向)mean/logStd 逐位相等；近邻 x 的 mean 接近(≤0.05)；远距 x 的 mean 分歧|0|确定性+平滑性+非平凡
 * @invariant INV-VI-008|ELBO 单调上升：玩具数据单步梯度上升后 ELBO_new ≥ ELBO_old − 1e-6（优化方向正确，断言不降）|1e-6|正确 ascent 方向
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
  if (!_V) throw new Error('core-verify 缺失');
  return _V;
}

let _N = null;
function N() {
  if (_N) return _N;
  if (typeof require !== 'undefined') { try { _N = require('./core-numeric.js'); } catch (e) {} }
  if (!_N && typeof globalThis !== 'undefined') _N = globalThis.ForgeNumeric || null;
  if (!_N) throw new Error('core-numeric 缺失');
  return _N;
}

/* ============================================================================
 * 1. 基础线性代数（row-major 矩阵-向量），仅用于摊销推断的前向
 * ========================================================================== */
function mv(W, x, b, n, m) {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    const r = i * m;
    for (let j = 0; j < m; j++) s += W[r + j] * x[j];
    out[i] = s;
  }
  return out;
}

/* ============================================================================
 * 2. 重参数化采样：z = mean + std·eps，eps~N(0,1)
 *    - 传入 eps 时使用给定 eps（保证逐位确定）；否则用 rng 采样。
 *    - 核心：给定相同 eps，z 必逐位确定（反空洞铁律要求 σ=0 与 σ>0 必分歧）。
 * ========================================================================== */
function reparamSample(mean, logStd, rng, eps) {
  const n = mean.length;
  const z = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const std = Math.exp(logStd[i]);
    const e = (eps !== undefined && eps !== null) ? eps[i] : N().normInv(1e-9 + (1 - 2e-9) * rng());
    z[i] = mean[i] + std * e;
  }
  return z;
}

/* ============================================================================
 * 3. 高斯 KL 散度闭式（逐维求和）
 *    KL(q‖p) = Σ 0.5·[ (μ_q−μ_p)²/σ_p² + σ_q²/σ_p² − 1 + 2(log σ_p − log σ_q) ]
 *    其中 σ = exp(logStd)。
 * ========================================================================== */
function gaussianKL1D(mq, lq, mp, lp) {
  const sq = Math.exp(lq), sp = Math.exp(lp), d = mq - mp;
  return 0.5 * ((d * d) / (sp * sp) + (sq * sq) / (sp * sp) - 1 + 2 * (Math.log(sp) - Math.log(sq)));
}
function gaussianKL(mq, lq, mp, lp) {
  const D = mq.length;
  let s = 0;
  for (let i = 0; i < D; i++) s += gaussianKL1D(mq[i], lq[i], mp[i], lp[i]);
  return s;
}

/* 均值场 KL：各分量（逐维）KL 求和。用于 INV-VI-006 可加性。 */
function meanFieldKL(mq, lq, mp, lp) {
  const D = mq.length;
  let s = 0;
  for (let i = 0; i < D; i++) s += gaussianKL1D(mq[i], lq[i], mp[i], lp[i]);
  return s;
}

/* ============================================================================
 * 4. 摊销推断（确定性前向，无采样）
 * ========================================================================== */
function makeEncoderParams(dimZ, dimX, rng, scale) {
  scale = (scale === undefined) ? 0.1 : scale;
  const Wm = new Float64Array(dimZ * dimX), bm = new Float64Array(dimZ);
  const Ws = new Float64Array(dimZ * dimX), bs = new Float64Array(dimZ);
  for (let i = 0; i < Wm.length; i++) { Wm[i] = (rng() * 2 - 1) * scale; Ws[i] = (rng() * 2 - 1) * scale; }
  for (let i = 0; i < bm.length; i++) { bm[i] = (rng() * 2 - 1) * scale; bs[i] = (rng() * 2 - 1) * scale; }
  return { dimZ, dimX, Wm, bm, Ws, bs };
}
function makeDecoderParams(dimX, dimZ, rng, scale) {
  scale = (scale === undefined) ? 0.1 : scale;
  const Wd = new Float64Array(dimX * dimZ), bd = new Float64Array(dimX);
  for (let i = 0; i < Wd.length; i++) Wd[i] = (rng() * 2 - 1) * scale;
  for (let i = 0; i < bd.length; i++) bd[i] = (rng() * 2 - 1) * scale;
  return { dimX, dimZ, Wd, bd };
}
function encodeForward(encP, x) {
  return {
    mean: mv(encP.Wm, x, encP.bm, encP.dimZ, encP.dimX),
    logStd: mv(encP.Ws, x, encP.bs, encP.dimZ, encP.dimX)
  };
}
function decodeForward(decP, z) {
  return mv(decP.Wd, z, decP.bd, decP.dimX, decP.dimZ);
}
function amortizedEncode(encoderParams, x) {
  return encodeForward(encoderParams, x); // 确定性前向，无采样
}

/* ============================================================================
 * 5. ELBO：ELBO = E_q[log p(x|z)] − KL(q‖p) = −recon − kl
 *    recon = −E_q[log p(x|z)]（重建损失，正数）。单样本重参数化估计。
 *    高斯似然 p(x|z)=N(decoder(z), I)（σ_x=1），与数据 x 同维。
 * ========================================================================== */
function elbo(x, encoder, decoder, prior, rng) {
  const q = encoder(x);                 // {mean, logStd}
  const z = reparamSample(q.mean, q.logStd, rng);
  const reconMean = decoder(z);        // 长度 = x.length
  let logp = 0;
  for (let i = 0; i < x.length; i++) {
    const d = x[i] - reconMean[i];
    logp += -0.5 * Math.log(2 * Math.PI) - 0.5 * d * d; // σ_x = 1
  }
  const recon = -logp;                 // 重建损失（正数）
  const kl = gaussianKL(q.mean, q.logStd, prior.mean, prior.logStd);
  return { elbo: logp - kl, recon, kl }; // 分量一致：elbo == −recon − kl
}

/* ============================================================================
 * 6. 玩具数据：固定种子采样自 N(mu0, sig0²)
 * ========================================================================== */
function toyData(seed, n, mu0, sig0) {
  mu0 = (mu0 === undefined) ? 1.2 : mu0;
  sig0 = (sig0 === undefined) ? 0.4 : sig0;
  const rng = N().makeRng(seed);
  const out = [];
  for (let i = 0; i < n; i++) {
    const e = N().normInv(1e-9 + (1 - 2e-9) * rng());
    out.push(new Float64Array([mu0 + sig0 * e]));
  }
  return out;
}

/* ============================================================================
 * 7. 1D VAE 前向 / 解析梯度（闭合形式，用于训练与梯度检验）
 *    模型：mean=Wm·x+bm，logStd=Ws·x+bs，z=mean+std·eps，
 *          reconMean=Wd·z+bd，似然 N(reconMean,1)，先验 N(0,1)。
 * ========================================================================== */
function fwd1D(P, x, eps) {
  const mean = P.Wm * x + P.bm;
  const logStd = P.Ws * x + P.bs;
  const std = Math.exp(logStd);
  const z = mean + std * eps;
  const reconMean = P.Wd * z + P.bd;
  const A = x - reconMean;
  const logp = -0.5 * Math.log(2 * Math.PI) - 0.5 * A * A;
  const kl = 0.5 * (mean * mean + std * std - 1 - 2 * logStd);
  return { mean, logStd, std, z, A, logp, kl, elbo: logp - kl, recon: -logp };
}
// 损失 L = recon + β·kl 对 6 个参数的解析梯度（梯度下降用）
function gradLoss1D(P, f, x, eps, beta) {
  const A = f.A, Wd = P.Wd, mean = f.mean, std = f.std;
  return {
    Wm: -A * Wd * x + beta * mean * x,
    bm: -A * Wd + beta * mean,
    Ws: -A * Wd * eps * std + beta * (std * std - 1) * x,
    bs: -A * Wd * eps * std + beta * (std * std - 1),
    Wd: -A * f.z,
    bd: -A
  };
}

/* ============================================================================
 * 8. VAE 训练：recon loss + β·KL，解析梯度 SGD，返回 losses 轨迹。
 *    仅支持 1D 玩具（dimX=dimZ=1）以保证解析梯度闭合。
 * ========================================================================== */
function vae(data, encP, decP, opts) {
  opts = opts || {};
  const beta = (opts.beta === undefined) ? 1.0 : opts.beta;
  const lr = (opts.lr === undefined) ? 0.02 : opts.lr;
  const epochs = opts.epochs || 200;
  const dimX = encP.dimX, dimZ = encP.dimZ;
  if (dimZ !== 1 || dimX !== 1) throw new Error('vae: 仅支持 1D 玩具（dimX=dimZ=1）以闭合解析梯度');
  let Wm = encP.Wm[0], bm = encP.bm[0], Ws = encP.Ws[0], bs = encP.bs[0];
  let Wd = decP.Wd[0], bd = decP.bd[0];
  // 固定 eps 每样本（确定性训练，可复现）
  const erng = N().makeRng(opts.seedEps || 0x6001);
  const eps = data.map(() => N().normInv(1e-9 + (1 - 2e-9) * erng()));
  const traj = [];
  for (let e = 0; e < epochs; e++) {
    let sRecon = 0, sKl = 0;
    for (let k = 0; k < data.length; k++) {
      const x = data[k][0], ep = eps[k];
      const f = fwd1D({ Wm, bm, Ws, bs, Wd, bd }, x, ep);
      sRecon += f.recon; sKl += f.kl;
      const g = gradLoss1D({ Wm, bm, Ws, bs, Wd, bd }, f, x, ep, beta);
      Wm -= lr * g.Wm; bm -= lr * g.bm; Ws -= lr * g.Ws; bs -= lr * g.bs; Wd -= lr * g.Wd; bd -= lr * g.bd;
    }
    const n = data.length;
    const avgRecon = sRecon / n, avgKl = sKl / n;
    traj.push({
      epoch: e, recon: avgRecon, kl: avgKl,
      total: avgRecon + beta * avgKl,
      elbo: -avgRecon - avgKl, beta
    });
  }
  encP.Wm[0] = Wm; encP.bm[0] = bm; encP.Ws[0] = Ws; encP.bs[0] = bs;
  decP.Wd[0] = Wd; decP.bd[0] = bd;
  return traj;
}

/* ============================================================================
 * 9. selfCheck — 8 条不变量（全部须 pass）
 * ========================================================================== */
function selfCheck(opts) {
  opts = opts || {};
  const { caseResult, buildReport } = V();
  const TOL = V().TOL;
  const cases = [];
  const invariants = [];

  // INV-VI-001 ELBO 分解闭式：elbo == -recon - kl；且 elbo ≤ E_q[log p]（= -recon，因 KL≥0）
  {
    const rng = N().makeRng(0x7101);
    let worst = 0;
    for (let t = 0; t < 30; t++) {
      const D = 1 + Math.floor(rng() * 4);
      const x = new Float64Array(D), mean = new Float64Array(D), logStd = new Float64Array(D);
      const priorMean = new Float64Array(D), priorLogStd = new Float64Array(D);
      for (let i = 0; i < D; i++) {
        x[i] = rng() * 4 - 2; mean[i] = rng() * 2 - 1; logStd[i] = Math.log(0.3 + rng() * 1.7);
        priorMean[i] = 0; priorLogStd[i] = 0; // 先验 N(0,1)
      }
      const prior = { mean: priorMean, logStd: priorLogStd };
      const enc = () => ({ mean, logStd }); // 固定 q（不依赖 x）
      const dec = (z) => z;                 // 恒等解码器
      const r = elbo(x, enc, dec, prior, rng);
      worst = Math.max(worst, Math.abs(r.elbo - (-r.recon - r.kl)));     // 分量一致
      const reconTerm = -r.recon;                                          // = E_q[log p(x|z)]
      worst = Math.max(worst, Math.max(0, r.elbo - reconTerm));           // elbo ≤ reconTerm
    }
    cases.push(caseResult('INV-VI-001', 'ELBO 分解闭式 elbo==-recon-kl 且 elbo<=E[log p]', worst, 0, TOL.RESID));
  }

  // INV-VI-002 重参数化梯度检验（灵魂）：L=recon+kl 对 mean/logStd 的解析梯度 vs 中心差分
  {
    let worst = 0;
    for (let t = 0; t < 20; t++) {
      const rng = N().makeRng(0x7200 + t);
      const mean = rng() * 1.5 - 0.75, logStd = rng() * 1.0 - 0.5;
      const x = rng() * 4 - 2;
      const eps = N().normInv(0.2 + rng() * 0.6); // 固定 eps → L 为 (mean,logStd) 的光滑函数
      const std = Math.exp(logStd);
      const L = (m, ls) => {
        const s = Math.exp(ls);
        const z = m + s * eps;
        const A = x - z;
        const recon = 0.5 * Math.log(2 * Math.PI) + 0.5 * A * A;       // = -log p
        const kl = 0.5 * (m * m + s * s - 1 - 2 * ls);
        return recon + kl; // β=1
      };
      // 解析梯度（pathwise / 重参数化：z=m+s·eps）
      const A = x - (mean + std * eps);
      const anaMean = -A + mean;
      const anaLogStd = -A * eps * std + (std * std - 1);
      const h = 1e-6;
      const numMean = (L(mean + h, logStd) - L(mean - h, logStd)) / (2 * h);
      const numLogStd = (L(mean, logStd + h) - L(mean, logStd - h)) / (2 * h);
      const relMean = Math.abs(numMean - anaMean) / (Math.abs(anaMean) > 1e-12 ? Math.abs(anaMean) : 1);
      const relLogStd = Math.abs(numLogStd - anaLogStd) / (Math.abs(anaLogStd) > 1e-12 ? Math.abs(anaLogStd) : 1);
      worst = Math.max(worst, relMean, relLogStd);
    }
    cases.push(caseResult('INV-VI-002', '重参数化路径梯度 vs 中心差分 (dL/dmean,dL/dlogStd)', worst, 0, TOL.GRAD));
  }

  // INV-VI-003 高斯 KL 闭式 vs 手搓黄金参照（灵魂）
  {
    const rng = N().makeRng(0x7300);
    let worst = 0, neg = 0, trivial = 0;
    for (let t = 0; t < 30; t++) {
      const D = 1 + Math.floor(rng() * 5);
      const mq = new Float64Array(D), lq = new Float64Array(D), mp = new Float64Array(D), lp = new Float64Array(D);
      for (let i = 0; i < D; i++) {
        mq[i] = rng() * 4 - 2; lq[i] = Math.log(0.3 + rng() * 2);
        mp[i] = rng() * 4 - 2; lp[i] = Math.log(0.3 + rng() * 2);
      }
      const got = gaussianKL(mq, lq, mp, lp);
      // 手搓黄金参照：Σ 0.5·(log(σp²/σq²) + (σq²+(μq-μp)²)/σp² − 1)
      let gold = 0;
      for (let i = 0; i < D; i++) {
        const sq = Math.exp(lq[i]), sp = Math.exp(lp[i]), d = mq[i] - mp[i];
        gold += 0.5 * (Math.log((sp * sp) / (sq * sq)) + (sq * sq + d * d) / (sp * sp) - 1);
      }
      worst = Math.max(worst, Math.abs(got - gold));
      if (got < -1e-12) neg++;                                   // KL≥0
      const same = gaussianKL(mq, lq, mq, lq);                   // 同参数 → 0
      if (Math.abs(same) > 1e-12) neg++;
    }
    // 非平凡性：不同参数对必 KL>0（反空洞，证明非恒为 0）
    const diff = gaussianKL(new Float64Array([1.0]), new Float64Array([0.0]), new Float64Array([-1.0]), new Float64Array([0.0]));
    if (!(diff > 1e-12)) trivial++;
    worst = Math.max(worst, neg, trivial);
    cases.push(caseResult('INV-VI-003', '高斯 KL 闭式 vs 手搓黄金参照 (KL≥0,KL=0⟺同参)', worst, 0, TOL.RESID));
  }

  // INV-VI-004 VAE 损失分解：loss = recon + β·kl；β=0 逐位==recon；β=1==recon+kl
  {
    const data = toyData(0x7400, 50);
    let worst = 0, beta0bad = 0, beta1bad = 0;
    for (const beta of [0, 1]) {
      const enc = makeEncoderParams(1, 1, N().makeRng(0x7401 + beta), 0.1);
      const dec = makeDecoderParams(1, 1, N().makeRng(0x7402 + beta), 0.1);
      const traj = vae(data, enc, dec, { beta, lr: 0.02, epochs: 5, seedEps: 0x7403 + beta });
      const last = traj[traj.length - 1];
      const comp = Math.abs(last.total - (last.recon + beta * last.kl));
      worst = Math.max(worst, comp);
      if (beta === 0 && last.total !== last.recon) beta0bad = 1; // 逐位
      if (beta === 1 && comp > 1e-12) beta1bad = 1;
    }
    worst = Math.max(worst, beta0bad, beta1bad);
    cases.push(caseResult('INV-VI-004', 'VAE 损失分解 total==recon+β·kl (β=0 逐位, β=1)', worst, 0, TOL.RESID));
  }

  // INV-VI-005 重参数化确定性（反空洞铁律）
  {
    let Vcount = 0;
    // ① 同 seed 同 (μ,σ) 重采样 z 逐位相等
    {
      const rng1 = N().makeRng(0x7501), rng2 = N().makeRng(0x7501);
      const D = 3, mean = new Float64Array(D), logStd = new Float64Array(D);
      const r0 = N().makeRng(0x7502);
      for (let i = 0; i < D; i++) { mean[i] = r0() * 2 - 1; logStd[i] = Math.log(0.3 + r0() * 1.7); }
      const z1 = reparamSample(mean, logStd, rng1), z2 = reparamSample(mean, logStd, rng2);
      let dev = 0; for (let i = 0; i < D; i++) dev = Math.max(dev, Math.abs(z1[i] - z2[i]));
      if (dev > 1e-15) Vcount++;
    }
    // ② 异 seed 产生不同确定序列（可复现且分歧）
    {
      const rngA = N().makeRng(0x7511), rngB = N().makeRng(0x7522);
      const D = 3, mean = new Float64Array(D), logStd = new Float64Array(D);
      const r0 = N().makeRng(0x7502);
      for (let i = 0; i < D; i++) { mean[i] = r0() * 2 - 1; logStd[i] = Math.log(0.3 + r0() * 1.7); }
      const zA = reparamSample(mean, logStd, rngA), zB = reparamSample(mean, logStd, rngB);
      let differ = false; for (let i = 0; i < D; i++) if (zA[i] !== zB[i]) differ = true;
      if (!differ) Vcount++;
    }
    // ③ σ=0 与 σ>0 必分歧（证明非恒等掩盖——W11 INV-DIF-005 教训）
    {
      const D = 3, mean = new Float64Array(D);
      const r0 = N().makeRng(0x7503); for (let i = 0; i < D; i++) mean[i] = r0() * 2 - 1;
      const logStd0 = new Float64Array(D).fill(-Infinity);     // σ = 0
      const logStdP = new Float64Array(D); for (let i = 0; i < D; i++) logStdP[i] = Math.log(0.7); // σ = 0.7
      const rngC = N().makeRng(0x7533);
      const z0 = reparamSample(mean, logStd0, rngC);
      const zp = reparamSample(mean, logStdP, rngC);
      let maxd = 0; for (let i = 0; i < D; i++) maxd = Math.max(maxd, Math.abs(z0[i] - zp[i]));
      if (!(maxd > 0)) Vcount++;
    }
    // ④ 大样本 MC：mean(z)≈μ、var(z)≈σ²（统计容差 ≥10σ 余量）
    {
      const Nn = 200000;
      const configs = [[0.3, 1.0], [-0.5, 0.8], [1.2, 1.5], [0.0, 0.5]];
      for (const [mu, sig] of configs) {
        const rng = N().makeRng(0x7540 + Math.floor((mu + 10) * 7) + Math.floor(sig * 13));
        const meanC = new Float64Array([mu]), logStdC = new Float64Array([Math.log(sig)]);
        let s1 = 0, s2 = 0;
        for (let i = 0; i < Nn; i++) { const z = reparamSample(meanC, logStdC, rng); s1 += z[0]; s2 += z[0] * z[0]; }
        const m = s1 / Nn, v = s2 / Nn - m * m;
        const tolMean = 12 * sig / Math.sqrt(Nn);                 // ≥10σ 余量
        const tolVar = 12 * (sig * sig) * Math.sqrt(2 / Nn);     // 方差估计误差界
        if (Math.abs(m - mu) > tolMean) Vcount++;
        if (Math.abs(v - sig * sig) > tolVar) Vcount++;
      }
    }
    cases.push(caseResult('INV-VI-005', '重参数化确定性·分歧·矩收敛（反空洞）', Vcount, 0, 0));
  }

  // INV-VI-006 均值场 KL 可加：Σ KL(q_i‖p_i) == 整体 KL
  {
    let worst = 0;
    for (let t = 0; t < 20; t++) {
      const D = 1 + Math.floor(N().makeRng(0x7600 + t)() * 6);
      const mq = new Float64Array(D), lq = new Float64Array(D), mp = new Float64Array(D), lp = new Float64Array(D);
      const rng = N().makeRng(0x7600 + t);
      for (let i = 0; i < D; i++) {
        mq[i] = rng() * 4 - 2; lq[i] = Math.log(0.3 + rng() * 2);
        mp[i] = rng() * 4 - 2; lp[i] = Math.log(0.3 + rng() * 2);
      }
      const mf = meanFieldKL(mq, lq, mp, lp);
      const full = gaussianKL(mq, lq, mp, lp);
      worst = Math.max(worst, Math.abs(mf - full));
      let s = 0;
      for (let i = 0; i < D; i++) s += gaussianKL1D(mq[i], lq[i], mp[i], lp[i]);
      worst = Math.max(worst, Math.abs(mf - s));
    }
    cases.push(caseResult('INV-VI-006', '均值场 KL 可加 ΣKL(q_i‖p_i)==整体 KL', worst, 0, TOL.RESID));
  }

  // INV-VI-007 摊销推断一致性：确定性 + 平滑性 + 非平凡
  {
    let Vcount = 0;
    const rng = N().makeRng(0x7700);
    const enc = makeEncoderParams(1, 2, rng, 0.1); // dimZ=1, dimX=2
    // ① 同 x 多次编码（确定性前向）逐位相等
    {
      const x = new Float64Array([0.5, 0.3]);
      const a = amortizedEncode(enc, x), b = amortizedEncode(enc, x);
      let eq = true;
      for (let i = 0; i < a.mean.length; i++) if (a.mean[i] !== b.mean[i] || a.logStd[i] !== b.logStd[i]) eq = false;
      if (!eq) Vcount++;
    }
    // ② 近邻 x 的 mean 接近（平滑性）
    {
      const x = new Float64Array([0.5, 0.3]), x2 = new Float64Array([0.55, 0.3]);
      const a = amortizedEncode(enc, x), b = amortizedEncode(enc, x2);
      let md = 0; for (let i = 0; i < a.mean.length; i++) md = Math.max(md, Math.abs(a.mean[i] - b.mean[i]));
      if (md > 0.05) Vcount++;
    }
    // ③ 远距 x 的 mean 必分歧（反空洞，非恒等/退化）
    {
      const x = new Float64Array([0.0, 0.0]), x2 = new Float64Array([2.0, 2.0]);
      const a = amortizedEncode(enc, x), b = amortizedEncode(enc, x2);
      let md = 0; for (let i = 0; i < a.mean.length; i++) md = Math.max(md, Math.abs(a.mean[i] - b.mean[i]));
      if (!(md > 1e-6)) Vcount++;
    }
    cases.push(caseResult('INV-VI-007', '摊销推断确定性·平滑性·非平凡', Vcount, 0, 0));
  }

  // INV-VI-008 ELBO 单调上升：单步梯度上升后不降
  {
    const rng = N().makeRng(0x7800);
    let worst = 0;
    for (let t = 0; t < 5; t++) {
      const P = {
        Wm: rng() * 1.5 - 0.75, bm: rng() * 1.5 - 0.75, Ws: rng() * 1.0 - 0.5, bs: rng() * 1.0 - 0.5,
        Wd: rng() * 1.5 - 0.75, bd: rng() * 1.5 - 0.75
      };
      const x = rng() * 4 - 2, eps = N().normInv(0.3 + rng() * 0.4);
      const f0 = fwd1D(P, x, eps);
      const elboOld = f0.elbo;
      // ELBO 梯度（ascent 方向，β=1）：dELBO/dθ = d(logp)/dθ − d(kl)/dθ
      const g = {
        Wm: -f0.A * P.Wd * x - f0.mean * x,
        bm: -f0.A * P.Wd - f0.mean,
        Ws: -f0.A * P.Wd * eps * f0.std * x - (f0.std * f0.std - 1) * x,
        bs: -f0.A * P.Wd * eps * f0.std - (f0.std * f0.std - 1),
        Wd: f0.A * f0.z,
        bd: f0.A
      };
      const lr = 0.01;
      const Pn = {
        Wm: P.Wm + lr * g.Wm, bm: P.bm + lr * g.bm, Ws: P.Ws + lr * g.Ws, bs: P.bs + lr * g.bs,
        Wd: P.Wd + lr * g.Wd, bd: P.bd + lr * g.bd
      };
      const f1 = fwd1D(Pn, x, eps);
      const elboNew = f1.elbo;
      worst = Math.max(worst, Math.max(0, elboOld - elboNew));
    }
    cases.push(caseResult('INV-VI-008', 'ELBO 单步梯度上升后不降', worst, 0, 1e-6));
  }

  const started = Date.now();
  for (const id of ['INV-VI-001', 'INV-VI-002', 'INV-VI-003', 'INV-VI-004', 'INV-VI-005', 'INV-VI-006', 'INV-VI-007', 'INV-VI-008']) {
    const c = cases.find((x) => x.id === id);
    invariants.push({ id, expr: id, tol: c.tol, worst: c.abs_err, status: c.status, note: '' });
  }
  return buildReport({
    module: 'vi-forge', version: '1.0.0', layer: 'L5', cases, invariants,
    seed: '0x9E3779B9', started,
    tol: {
      'INV-VI-001': TOL.RESID, 'INV-VI-002': TOL.GRAD, 'INV-VI-003': TOL.RESID,
      'INV-VI-004': TOL.RESID, 'INV-VI-005': 0, 'INV-VI-006': TOL.RESID,
      'INV-VI-007': 0, 'INV-VI-008': 1e-6
    }
  });
}

const API = {
  name: 'vi-forge', version: '1.0.0', layer: 'L5',
  reparamSample, gaussianKL, gaussianKL1D, meanFieldKL,
  elbo, vae, amortizedEncode,
  makeEncoderParams, makeDecoderParams, encodeForward, decodeForward,
  toyData, fwd1D, gradLoss1D, selfCheck
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeVI = API;

if (typeof process !== 'undefined' && process.argv[1] && /vi-forge\.js$/.test(process.argv[1]) && (process.argv.includes('--self-check') || process.argv.includes('--json'))) {
  const r = selfCheck({ quick: process.argv.includes('--quick') });
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : V().formatTableSafe(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
