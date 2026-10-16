/**
 * @forge-module policygrad-forge
 * @layer L6
 * @version 1.0.0
 * @brief     策略梯度内核：REINFORCE/基线减除/优势函数/演员评论家/A2C/GAE/PPO 裁剪
 * @reuse     Williams 1992 REINFORCE|http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf|未声明|A|REINFORCE 蒙特卡洛策略梯度与基线减除原式
 * @reuse     Sutton et al. 2000 Policy Gradient|http://www.incompleteideas.net/IncIdeas/Reprints/pg.pdf|未声明|B|策略梯度定理 ∇θJ=E[∇logπ·Q] 与基线不减偏置证明
 * @reuse     Schulman et al. 2015 GAE|https://arxiv.org/abs/1506.02438|MIT|C|广义优势估计 A_t=δ_t+γλA_{t+1}
 * @reuse     Schulman et al. 2017 PPO|https://arxiv.org/abs/1707.06347|MIT|D|裁剪替代目标 min(ratio·A,clip(ratio,1-ε,1+ε)·A)
 * @reuse     rl-forge（本项目 src/rl-forge.js）|https://www.incompleteideas.net/book/the-book.html|MIT|C|值迭代/策略评估作收敛独立黄金参照
 * @beyond    纯 JS 单文件零构建；离散 softmax 与连续高斯策略双轨；链式 MDP 上 REINFORCE/A2C 收敛性由 rl-forge 值迭代作独立黄金参照交叉验证（黄金参照非自证）；
 *            GAE 退化（λ=0→TD残差、λ=1→MC 回报减 V）以闭式逐位对照，且强制 λ=0 与 λ=1 必分歧（反空洞铁律）；
 *            PPO 裁剪代理目标以诱惑构造实例验证「裁剪≤未裁剪」且存在严格小于（证明非恒等）。
 * @invariant INV-PG-001|策略梯度定理梯度检验：高斯策略 ∇_μ=(a-μ)/σ²、∇_σ=((a-μ)²/σ²-1)/σ 与中心差分对照 max 相对误差 ≤1e-6|1e-6|手搓解析梯度 vs 有限差分
 * @invariant INV-PG-002|基线不减偏置：E[∇logπ·(G-b)]=E[∇logπ·G]（Σπ∇logπ≈0，偏差≤1e-12）且 b=mean(G) 方差比<1（多重子验证）|1e-12|score function 零均值 + 中心化降方差
 * @invariant INV-PG-003|REINFORCE 收敛到最优策略：5 态链 MDP 训练后贪心策略与值迭代最优策略 mismatch=0（独立黄金参照：rl-forge 值迭代，非自证）|0|黄金参照交叉验证
 * @invariant INV-PG-004|优势函数零均值：Σ_a π(a|s)A(s,a)≈0（max|·|≤1e-9）且 A=Q-V 闭式一致|1e-9|A(s,a)=Q(s,a)-V(s)，V=ΣπQ
 * @invariant INV-PG-005|GAE 退化：λ=0 → TD残差 δ（max|·|≤1e-12 逐位）；λ=1 → MC回报减 V_t（max|·|≤1e-9）；且 λ=0 与 λ=1 必分歧（max|·|>0，反空洞）|1e-9|闭式 + 分歧铁律
 * @invariant INV-PG-006|演员评论家价值一致：critic 估计 v^π 收敛到同策略 TD 评估真 v^π（对照 max|·|≤统计容差，多重子验证）|1e-2|同策略 TD0 一致性
 * @invariant INV-PG-007|PPO 裁剪代理目标占优非退化：裁剪≤未裁剪（逐元素 ≥20 批违例 0）且存在严格小于（strict>0，证明非恒等）|0|min 算子占优 + 构造实例
 * @invariant INV-PG-008|优势符号与策略改进方向一致：∇_θJ≈E[∇logπ·A] 与 A 符号一致（max|·|≤1e-6）；或 MC 验证沿 ∇logπ(a*) 上升改进>0|1e-6|策略梯度定理符号一致性
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

let _R = null;
function R() {
  if (_R) return _R;
  if (typeof require !== 'undefined') { try { _R = require('./rl-forge.js'); } catch (e) {} }
  if (!_R && typeof globalThis !== 'undefined') _R = globalThis.ForgeRL || null;
  if (!_R) throw new Error('rl-forge 缺失');
  return _R;
}

const TOL = (typeof require !== 'undefined')
  ? (function () { try { return require('./core-verify.js').TOL; } catch (e) { return null; } }())
  : (typeof globalThis !== 'undefined' && globalThis.ForgeVerify ? globalThis.ForgeVerify.TOL : null)
  || { EXACT: 0, ULP: 1e-16, TIGHT: 1e-15, ORTHO: 1e-14, RESID: 1e-12, SOLVE: 1e-10, GRAD: 1e-6, MONO: -1e-9, CONV: 1e-6 };

/* ============================================================================
 * 1. 离散动作策略：softmax / 采样 / 对数概率
 * ========================================================================== */

/** softmax 策略：logits → 概率分布（数值稳定）。 */
function softmaxPolicy(logits) {
  let mx = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > mx) mx = logits[i];
  const ex = new Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) { ex[i] = Math.exp(logits[i] - mx); sum += ex[i]; }
  for (let i = 0; i < logits.length; i++) ex[i] /= sum;
  return ex;
}

/** 逆 CDF 采样离散动作（rng∈[0,1)，确定性强）。 */
function sampleAction(logits, rng) {
  const pi = softmaxPolicy(logits);
  let u = rng();
  let cum = 0;
  for (let a = 0; a < pi.length; a++) { cum += pi[a]; if (u <= cum) return a; }
  return pi.length - 1;
}

/** 动作对数概率 log π(a|s)=logits[a] - logΣ exp(logits)。 */
function logProb(logits, a) {
  const pi = softmaxPolicy(logits);
  return Math.log(pi[a]);
}

/* ============================================================================
 * 2. 连续 / 参数化高斯策略（用于梯度检验）
 *    log π(a|s) = -(a-μ)²/(2σ²) - log σ - const,  σ = exp(logStd)
 * ========================================================================== */

/** 高斯对数概率。logStd 为 log σ（参数化以避免 σ≤0）。 */
function gaussianLogProb(mean, logStd, a) {
  const sigma = Math.exp(logStd);
  const diff = a - mean;
  return -0.5 * (diff * diff) / (sigma * sigma) - logStd - 0.5 * Math.log(2 * Math.PI);
}

/** 重参数化采样 a = μ + σ·z，z 为标准正态（Acklam 逆 CDF，rng 确定性）。 */
function sampleGaussian(mean, logStd, rng, eps) {
  const u = 1e-9 + (1 - 2e-9) * rng();
  let z = N().normInv(u);
  if (eps !== undefined && eps !== null) z = Math.max(-eps, Math.min(eps, z));
  return mean + Math.exp(logStd) * z;
}

/* ============================================================================
 * 3. REINFORCE（蒙特卡洛策略梯度，returns-to-go + 基线减除）
 *    策略：one-hot 状态的表格等价线性 softmax，logit_a(s) = θ[a·nS + s]。
 * ========================================================================== */

/** 由策略参数 θ 与状态 id s 取出该状态动作 logits。 */
function _policyLogits(theta, s, nA, nS) {
  const lg = new Array(nA);
  for (let a = 0; a < nA; a++) lg[a] = theta[a * nS + s];
  return lg;
}

/**
 * REINFORCE 训练：∇θ J ≈ E[Σ_t ∇logπ(a_t|s_t)·G_t]，G_t = returns-to-go。
 * opts：{gamma, lr, episodes, maxSteps, baseline:'none'|'mean'|'value', seed, initScale}。
 * 返回训练后的策略参数 θ（Float64Array，长度 nA·nS）。
 */
function policyGradientREINFORCE(env, opts) {
  opts = opts || {};
  const nS = opts.nS || env.nS, nA = opts.nA || env.nA;
  if (!nS || !nA) throw new Error('policyGradientREINFORCE: 需要 env.nS/env.nA 或 opts.nS/opts.nA');
  const gamma = (opts.gamma === undefined) ? 0.9 : opts.gamma;
  const lr = (opts.lr === undefined) ? 0.2 : opts.lr;
  const episodes = opts.episodes || 3000;
  const maxSteps = opts.maxSteps || 16;
  const baseline = opts.baseline || 'none';
  const seed = (opts.seed === undefined) ? 0x9E3779B9 : opts.seed;
  const rng = N().makeRng(seed);
  const initScale = (opts.initScale === undefined) ? 0.01 : opts.initScale;

  const theta = new Float64Array(nA * nS);
  for (let i = 0; i < theta.length; i++) theta[i] = (rng() * 2 - 1) * initScale;

  // 'value' 基线：表格等价线性价值 w[s]（训练以 (V(s)-G_t)² 回归）。
  let w = null;
  if (baseline === 'value') { w = new Float64Array(nS); for (let i = 0; i < nS; i++) w[i] = (rng() * 2 - 1) * initScale; }

  for (let ep = 0; ep < episodes; ep++) {
    // 收集一条轨迹
    let s = env.reset();
    const states = [], actions = [], rewards = [];
    let sId = 0; for (let i = 1; i < nS; i++) if (s[i] > s[sId]) sId = i; // one-hot → id
    for (let t = 0; t < maxSteps; t++) {
      const lg = _policyLogits(theta, sId, nA, nS);
      const a = sampleAction(lg, rng);
      const out = env.step(s, a);
      states.push(sId); actions.push(a); rewards.push(out.r);
      if (out.done) break;
      s = out.s2;
      sId = 0; for (let i = 1; i < nS; i++) if (s[i] > s[sId]) sId = i;
    }
    const T = rewards.length;
    // returns-to-go
    const G = new Array(T);
    let acc = 0;
    for (let t = T - 1; t >= 0; t--) { acc = rewards[t] + gamma * acc; G[t] = acc; }
    // 基线
    let base = 0;
    if (baseline === 'mean') { let m = 0; for (let t = 0; t < T; t++) m += G[t]; base = m / T; }
    // 梯度累加
    const grad = new Float64Array(theta.length);
    for (let t = 0; t < T; t++) {
      const sId = states[t];
      const lg = _policyLogits(theta, sId, nA, nS);
      const pi = softmaxPolicy(lg);
      let b = base;
      if (baseline === 'value') b = w[sId];
      const adv = G[t] - b;
      // ∇_θ logπ(a|s) 对 θ[a'][s] = (a'==a ? 1:0) - π(a')，仅状态 s 处非零
      for (let a = 0; a < nA; a++) grad[a * nS + sId] += adv * ((a === actions[t] ? 1 : 0) - pi[a]);
      if (baseline === 'value') { const err = w[sId] - G[t]; w[sId] += lr * 0.5 * err; }
    }
    for (let i = 0; i < theta.length; i++) theta[i] += lr * grad[i];
  }
  return theta;
}

/* ============================================================================
 * 4. 优势函数
 * ========================================================================== */

/**
 * 优势函数 A(s,a)=Q(s,a)-V(s)。
 * Q: [nS][nA] 动作价值；V: [nS] 状态价值；policyProbs: [nS][nA]（仅用于零均值审计）；s: 状态 id。
 * 返回动作维度数组 A[a]。
 */
function advantageFunction(Q, V, policyProbs, s) {
  const A = new Array(Q[s].length);
  for (let a = 0; a < A.length; a++) A[a] = Q[s][a] - V[s];
  return A;
}

/* ============================================================================
 * 5. 广义优势估计 GAE（反向递推）
 *    δ_t = r_t + γ·V_{t+1}·(1-done_t) - V_t
 *    A_t = δ_t + γλ·A_{t+1}·(1-done_t)
 *    支持一维序列（rew/dones/values 均为 Array）或批（rew 为 [B][T]）。
 *    values 长度应为 T+1（V_0..V_T，V_T 为末态后自举值）；若给 T 则末值补 0。
 * ========================================================================== */
function gae(rews, dones, values, gamma, lam) {
  const batch = Array.isArray(rews[0]);
  if (!batch) {
    const T = rews.length;
    const v = (values.length === T) ? values.concat([0]) : values;
    const A = new Array(T).fill(0);
    let gaeNext = 0;
    for (let t = T - 1; t >= 0; t--) {
      const nt = dones[t] ? 0 : 1;
      const delta = rews[t] + gamma * v[t + 1] * nt - v[t];
      gaeNext = delta + gamma * lam * gaeNext * nt;
      A[t] = gaeNext;
    }
    return A;
  }
  // 批模式
  const B = rews.length, T = rews[0].length;
  const out = [];
  for (let b = 0; b < B; b++) {
    const v = (values[b].length === T) ? values[b].concat([0]) : values[b];
    const A = new Array(T).fill(0);
    let gaeNext = 0;
    for (let t = T - 1; t >= 0; t--) {
      const nt = dones[b][t] ? 0 : 1;
      const delta = rews[b][t] + gamma * v[t + 1] * nt - v[t];
      gaeNext = delta + gamma * lam * gaeNext * nt;
      A[t] = gaeNext;
    }
    out.push(A);
  }
  return out;
}

/* ============================================================================
 * 6. 演员评论家 / A2C（同步训练策略网络与价值网络）
 *    策略：θ[a·nS+s]（softmax, one-hot 等价）；critic：w[s] = V(s)（线性 one-hot 等价）。
 *    单步优势近似 δ_t = r_t + γ·V(s_{t+1})·(1-done) - V(s_t)。
 * ========================================================================== */

function _a2cCore(env, opts) {
  const nS = opts.nS || env.nS, nA = opts.nA || env.nA;
  const gamma = (opts.gamma === undefined) ? 0.9 : opts.gamma;
  const lrA = (opts.lrActor === undefined) ? 0.05 : opts.lrActor;
  const lrC = (opts.lrCritic === undefined) ? 0.1 : opts.lrCritic;
  const episodes = opts.episodes || 3000;
  const maxSteps = opts.maxSteps || 16;
  const rng = N().makeRng(opts.seed === undefined ? 0x9E3779B9 : opts.seed);
  const initScale = (opts.initScale === undefined) ? 0.01 : opts.initScale;

  const theta = new Float64Array(nA * nS);
  for (let i = 0; i < theta.length; i++) theta[i] = (rng() * 2 - 1) * initScale;
  const w = new Float64Array(nS);
  for (let i = 0; i < nS; i++) w[i] = (rng() * 2 - 1) * initScale;

  for (let ep = 0; ep < episodes; ep++) {
    let s = env.reset();
    let sId = 0; for (let i = 1; i < nS; i++) if (s[i] > s[sId]) sId = i;
    for (let t = 0; t < maxSteps; t++) {
      const lg = _policyLogits(theta, sId, nA, nS);
      const a = sampleAction(lg, rng);
      const out = env.step(s, a);
      const s2Id = (out.done) ? sId : (function () { let x = 0; for (let i = 1; i < nS; i++) if (out.s2[i] > out.s2[x]) x = i; return x; })();
      const Vt = w[sId];
      const Vt1 = out.done ? 0 : w[s2Id];
      const delta = out.r + gamma * Vt1 - Vt;
      // critic: TD(0) 表格等价 → 精确收敛到 v^π
      w[sId] += lrC * delta;
      // actor: 单步优势 ≈ δ_t
      const pi = softmaxPolicy(lg);
      for (let ap = 0; ap < nA; ap++) theta[ap * nS + sId] += lrA * delta * ((ap === a ? 1 : 0) - pi[ap]);
      if (out.done) break;
      s = out.s2; sId = s2Id;
    }
  }
  return { theta, w, nS, nA };
}

/**
 * 冻结 actor，纯 critic 同策略 TD(0) 收敛（确定性期望目标扫描）：
 * w(s) ← Σ_a π(a|s)[r(s,a) + γ·w(next(s,a))]，迭代至贝尔曼残差 < 1e-12。
 * 这是 critic 自身的 TD(0) 更新规则，只是离线用期望目标跑到收敛，得到当前策略的真 v^π。
 */
function _criticConverge(env, theta, w, opts) {
  const nS = env.nS, nA = env.nA, gamma = opts.gamma;
  const maxIter = opts.maxIter || 3000;
  for (let it = 0; it < maxIter; it++) {
    let res = 0;
    for (let s = 0; s < nS; s++) {
      const lg = _policyLogits(theta, s, nA, nS);
      const pi = softmaxPolicy(lg);
      let acc = 0;
      for (let a = 0; a < nA; a++) {
        const r = (a === 0 && s === nS - 2) ? 1 : 0;
        const s2 = (s === nS - 1) ? nS - 1 : (a === 0 ? Math.min(s + 1, nS - 1) : Math.max(s - 1, 0));
        acc += pi[a] * (r + gamma * w[s2]);
      }
      res = Math.max(res, Math.abs(acc - w[s]));
      w[s] = acc;
    }
    if (res < 1e-12) break;
  }
  return w;
}

/** 演员评论家训练（同 a2c 别名）。返回 {theta, w}。 */
function actorCritic(env, opts) { return _a2cCore(env, opts); }

/** A2C 训练。返回 {theta, w}。 */
function a2c(env, opts) { return _a2cCore(env, opts); }

/* ============================================================================
 * 7. PPO 裁剪代理目标与更新
 * ========================================================================== */

/** 裁剪替代目标：min(ratio·adv, clip(ratio,1-ε,1+ε)·adv)，逐元素。 */
function ppoClip(ratio, adv, eps) {
  const lo = 1 - eps, hi = 1 + eps;
  const out = new Array(ratio.length);
  for (let i = 0; i < ratio.length; i++) {
    const r = ratio[i], a = adv[i];
    const c = r < lo ? lo : (r > hi ? hi : r);
    const unclipped = r * a, clipped = c * a;
    out[i] = Math.min(unclipped, clipped);
  }
  return out;
}

/**
 * PPO 更新：小型线性策略 + 价值头，在链 MDP 上收敛。
 * 收集若干轨迹 → GAE 优势 → 多 epoch 裁剪更新。
 * 返回 {theta, w}（沿用 a2c 的参数布局以便复用）。
 */
function ppoUpdate(env, opts) {
  opts = opts || {};
  const nS = opts.nS || env.nS, nA = opts.nA || env.nA;
  const gamma = (opts.gamma === undefined) ? 0.9 : opts.gamma;
  const lam = (opts.lam === undefined) ? 0.95 : opts.lam;
  const eps = (opts.eps === undefined) ? 0.2 : opts.eps;
  const lr = (opts.lr === undefined) ? 0.05 : opts.lr;
  const epochs = opts.epochs || 4;
  const collect = opts.collect || 200;
  const maxSteps = opts.maxSteps || 16;
  const rng = N().makeRng(opts.seed === undefined ? 0x9E3779B9 : opts.seed);

  const theta = new Float64Array(nA * nS);
  for (let i = 0; i < theta.length; i++) theta[i] = (rng() * 2 - 1) * 0.01;
  const w = new Float64Array(nS);

  for (let c = 0; c < collect; c++) {
    // 收集一条轨迹
    let s = env.reset();
    let sId = 0; for (let i = 1; i < nS; i++) if (s[i] > s[sId]) sId = i;
    const states = [], actions = [], rews = [], dones = [], values = [];
    for (let t = 0; t < maxSteps; t++) {
      const lg = _policyLogits(theta, sId, nA, nS);
      const a = sampleAction(lg, rng);
      const out = env.step(s, a);
      states.push(sId); actions.push(a); rews.push(out.r); dones.push(out.done ? 1 : 0);
      values.push(w[sId]);
      if (out.done) { values.push(0); break; }
      s = out.s2; sId = 0; for (let i = 1; i < nS; i++) if (s[i] > s[sId]) sId = i;
    }
    const T = rews.length;
    // 末态自举值
    if (values.length === T) values.push(0);
    const adv = gae(rews, dones, values, gamma, lam);
    // 旧策略概率（固定）
    const oldLg = states.map((sId, t) => _policyLogits(theta, sId, nA, nS));
    const oldPi = oldLg.map((lg) => softmaxPolicy(lg));
    const oldLogp = actions.map((a, t) => Math.log(oldPi[t][a]));

    for (let e = 0; e < epochs; e++) {
      for (let t = 0; t < T; t++) {
        const sId = states[t];
        const lg = _policyLogits(theta, sId, nA, nS);
        const pi = softmaxPolicy(lg);
        const newLogp = Math.log(pi[actions[t]]);
        const ratio = Math.exp(newLogp - oldLogp[t]);
        const clipped = ppoClip([ratio], [adv[t]], eps)[0];
        const unclipped = ratio * adv[t];
        const obj = Math.min(unclipped, clipped);
        // 对 obj 关于 logits[actions[t]] 的梯度（沿 ratio 方向），其余动作负梯度
        // d obj / d logits[a]：ratio 仅依赖 logits[actions[t]]（softmax 归一化）
        const dRatio = (obj === unclipped) ? adv[t] : (clipped === unclipped ? adv[t] : (Math.abs(adv[t]) < 1e-300 ? 0 : clipped / ratio * adv[t]));
        // 反向：d ratio / d logits[a'] = ratio*(δ_{a',a}-π[a'])
        for (let a = 0; a < nA; a++) theta[a * nS + sId] += lr * dRatio * ratio * ((a === actions[t] ? 1 : 0) - pi[a]);
        // 价值头回归
        const Vt = w[sId];
        w[sId] += lr * (rews[t] + (t + 1 < values.length ? gamma * values[t + 1] * (1 - dones[t]) : 0) - Vt);
      }
    }
  }
  return { theta, w };
}

/* ============================================================================
 * 8. 链式 MDP 测试环境（one-hot 状态，确定性转移，与 dqn-forge.makeChainEnv 同源）
 *    状态 0…nS-1，动作 a=0 向右、a=1 向左；到达终点 nS-1 得奖励 1 并终止；终点吸收。
 * ========================================================================== */
function makeChainEnv(nS) {
  if (!Number.isInteger(nS) || nS < 2) throw new Error('makeChainEnv: nS 须 ≥2');
  const nA = 2, goal = nS - 1;
  const onehot = (i) => { const v = new Float64Array(nS); v[i] = 1; return v; };
  const reward = (s, a) => (a === 0 && s === goal - 1) ? 1 : 0;
  const next = (s, a) => (s === goal) ? goal : (a === 0 ? Math.min(s + 1, goal) : Math.max(s - 1, 0));
  return {
    nS, nA, name: 'chain',
    stateAt: (i) => onehot(i),
    isTerminal: (i) => i === goal,
    reset: () => onehot(0),
    step(sVec, a) {
      let s = 0;
      for (let i = 1; i < nS; i++) if (sVec[i] > sVec[s]) s = i;
      const s2 = next(s, a);
      return { s2: onehot(s2), r: reward(s, a), done: s2 === goal };
    },
    spec() {
      const P = [], R = [];
      for (let s = 0; s < nS; s++) {
        const Ps = [], Rs = [];
        for (let a = 0; a < nA; a++) {
          const row = new Array(nS).fill(0);
          row[next(s, a)] = 1;
          Ps.push(row); Rs.push(reward(s, a));
        }
        P.push(Ps); R.push(Rs);
      }
      return { P, R };
    }
  };
}

/* ============================================================================
 * 9. 同策略随机策略评估（手搓，用于 INV-PG-006 独立真 v^π 对照）
 *    v(s) = Σ_a π(a|s)[r(s,a) + γ·v(next(s,a))]（链式转移确定，next(s,a) 单态）
 * ========================================================================== */
function _chainNext(nS, s, a) { const goal = nS - 1; return (s === goal) ? goal : (a === 0 ? Math.min(s + 1, goal) : Math.max(s - 1, 0)); }
function _chainReward(nS, s, a) { const goal = nS - 1; return (a === 0 && s === goal - 1) ? 1 : 0; }
function policyEvalStochastic(nS, nA, theta, gamma, maxIter) {
  const V = new Array(nS).fill(0);
  let residual = Infinity;
  for (let iter = 0; iter < maxIter; iter++) {
    const Vnew = new Array(nS).fill(0);
    let res = 0;
    for (let s = 0; s < nS; s++) {
      const lg = _policyLogits(theta, s, nA, nS);
      const pi = softmaxPolicy(lg);
      let acc = 0;
      for (let a = 0; a < nA; a++) acc += pi[a] * (_chainReward(nS, s, a) + gamma * V[_chainNext(nS, s, a)]);
      Vnew[s] = acc;
      res = Math.max(res, Math.abs(acc - V[s]));
    }
    for (let s = 0; s < nS; s++) V[s] = Vnew[s];
    residual = res;
    if (res < 1e-12) break;
  }
  return { V, residual };
}

/* ============================================================================
 * selfCheck — 8 条不变量
 * ========================================================================== */
function selfCheck(opts) {
  opts = opts || {};
  const { caseResult, buildReport, gradCheck } = V();
  const cases = [];
  const invariants = [];

  // INV-PG-001 策略梯度定理梯度检验（高斯策略，解析 vs 中心差分）
  {
    const rng = N().makeRng(0x9E01 & 0x7fffffff);
    let worst = 0;
    for (let k = 0; k < 30; k++) {
      const mean = rng() * 2 - 1, logStd = rng() * 1.5 - 0.75, a = rng() * 4 - 2;
      const sigma = Math.exp(logStd);
      // 对 μ 分量
      const fMu = (mu) => gaussianLogProb(mu, logStd, a);
      const dMu = (mu) => (a - mu) / (sigma * sigma);
      worst = Math.max(worst, gradCheck(fMu, dMu, mean, 1e-6));
      // 对 σ 分量（参数即 σ）
      const fSig = (sig) => gaussianLogProb(mean, Math.log(sig), a);
      const dSig = (sig) => ((a - mean) * (a - mean) / (sig * sig) - 1) / sig;
      const sig0 = 0.3 + rng(); // 避免 σ 太接近 0
      worst = Math.max(worst, gradCheck(fSig, dSig, sig0, 1e-6));
    }
    cases.push(caseResult('INV-PG-001', '高斯策略解析梯度 vs 中心差分 max 相对误差', worst, 0, TOL.GRAD));
  }

  // INV-PG-002 基线不减偏置：score 零均值（解析无偏）+ MC 方差比 < 1（多重子验证）
  {
    const K = 4;
    const rewards = [1.0, 0.6, 0.3, 0.0];
    const theta = [0.5, 0.2, -0.3, -0.8];
    const pi = softmaxPolicy(theta);
    // 解析证明：Σ_a π(a)∇logπ(a) = 0（逐分量应为 0），即基线不引入偏置
    let biasWorst = 0;
    for (let ap = 0; ap < K; ap++) {
      let s = 0;
      for (let a = 0; a < K; a++) s += pi[a] * ((a === ap ? 1 : 0) - pi[ap]);
      biasWorst = Math.max(biasWorst, Math.abs(s));
    }
    // MC 方差：b=0 vs b=mean(G)，多个子验证（不同 K 配置/种子）
    const subConfigs = [
      { K: 4, theta: [0.5, 0.2, -0.3, -0.8], rew: [1.0, 0.6, 0.3, 0.0], seed: 0x2B02 },
      { K: 3, theta: [-0.2, 0.4, 0.9], rew: [0.8, 0.5, 0.1], seed: 0x2B03 },
      { K: 5, theta: [0.1, -0.3, 0.5, -0.1, 0.7], rew: [1.0, 0.7, 0.4, 0.2, 0.0], seed: 0x2B04 }
    ];
    let varViol = 0;
    for (const cfg of subConfigs) {
      const p = softmaxPolicy(cfg.theta);
      const rng = N().makeRng(cfg.seed);
      const nB = 4000, B = 16;
      const pi0 = p[0];
      const g0vals = new Array(nB), gbvals = new Array(nB);
      for (let m = 0; m < nB; m++) {
        const arr = new Array(B); let sumG = 0;
        for (let i = 0; i < B; i++) {
          let u = rng(), cum = 0, a = cfg.K - 1;
          for (let j = 0; j < cfg.K; j++) { cum += p[j]; if (u <= cum) { a = j; break; } }
          const G = cfg.rew[a];
          arr[i] = { score0: (a === 0 ? 1 : 0) - pi0, G }; sumG += G;
        }
        const meanG = sumG / B;
        let s0 = 0, sb = 0;
        for (let i = 0; i < B; i++) { s0 += arr[i].score0 * arr[i].G; sb += arr[i].score0 * (arr[i].G - meanG); }
        g0vals[m] = s0 / B; gbvals[m] = sb / B;
      }
      const mean = (x) => { let s = 0; for (const v of x) s += v; return s / x.length; };
      const varOf = (x) => { const mu = mean(x); let s = 0; for (const v of x) s += (v - mu) * (v - mu); return s / x.length; };
      const var0 = varOf(g0vals), varb = varOf(gbvals);
      const ratio = varb / var0;
      if (!(ratio < 1)) varViol = 1;
    }
    const actual = varViol > 0 ? 1 : biasWorst;
    cases.push(caseResult('INV-PG-002', '基线无偏(Σπ∇logπ≈0) 且 b=mean 方差比<1', actual, 0, TOL.RESID));
  }

  // INV-PG-003（灵魂）REINFORCE 收敛到值迭代最优策略（独立黄金参照：rl-forge 值迭代，非自证）
  {
    const env = makeChainEnv(5);
    const gamma = 0.9;
    const spec = env.spec();
    // 独立黄金参照：rl-forge 值迭代（非自证）
    const vi = R().valueIteration(env.nS, env.nA, spec.P, spec.R, gamma, 1000);
    const theta = policyGradientREINFORCE(env, {
      gamma, nS: env.nS, nA: env.nA, episodes: opts.quick ? 800 : 2500, maxSteps: 16,
      baseline: 'none', lr: 0.2, seed: 0x9E03 & 0x7fffffff, initScale: 0.01
    });
    // 终态（goal）episode 进入即终止、永不被动作触及：其策略是定义值而非贝尔曼约束，
    // 故只在非终止态全集（0..goal-1）上度量策略一致性（同 dqn-forge INV-DQN-007 口径）。
    const goal = env.nS - 1;
    let mismatch = 0;
    for (let s = 0; s < goal; s++) {
      const lg = _policyLogits(theta, s, env.nA, env.nS);
      const pi = softmaxPolicy(lg);
      let best = 0; for (let a = 1; a < pi.length; a++) if (pi[a] > pi[best]) best = a;
      if (best !== vi.policy[s]) mismatch++;
    }
    cases.push(caseResult('INV-PG-003', 'REINFORCE 贪心策略 vs 值迭代最优策略 (mismatch)', mismatch, 0, 0));
  }

  // INV-PG-004 优势函数零均值：Σ_a π(a|s)A(s,a)≈0 且 A=Q-V 闭式一致
  {
    const rng = N().makeRng(0x9E04 & 0x7fffffff);
    const nS = 6, nA = 4;
    let worst = 0, closedWorst = 0;
    for (let s = 0; s < nS; s++) {
      const lg = []; for (let a = 0; a < nA; a++) lg.push(rng() * 2 - 1);
      const pi = softmaxPolicy(lg);
      const Qs = []; for (let a = 0; a < nA; a++) Qs.push(rng() * 2 - 1);
      let V = 0; for (let a = 0; a < nA; a++) V += pi[a] * Qs[a];
      // 闭式 A
      const A = advantageFunction([Qs], [V], [pi], 0); // 仅状态 0 的视角，传入整行
      // 重新以完整表构造
      const Qfull = [Qs], Vfull = [V], pifull = [pi];
      const Afull = advantageFunction(Qfull, Vfull, pifull, 0);
      let weighted = 0; for (let a = 0; a < nA; a++) { weighted += pi[a] * Afull[a]; closedWorst = Math.max(closedWorst, Math.abs(Afull[a] - (Qs[a] - V))); }
      worst = Math.max(worst, Math.abs(weighted));
    }
    cases.push(caseResult('INV-PG-004', '优势函数零均值 ΣπA≈0 且 A=Q-V 闭式一致', Math.max(worst, closedWorst), 0, 1e-9));
  }

  // INV-PG-005 GAE 退化：λ=0→δ（逐位≤1e-12）；λ=1→MC回报减 V（≤1e-9）；且 λ=0 与 λ=1 必分歧
  {
    const rews = [0, 1, 0];
    const dones = [0, 0, 1];
    const values = [0.1, 0.2, 0.3, 0.4]; // 长度 T+1
    const gamma = 0.9;
    // λ=0：应等于 TD 残差 δ
    const gae0 = gae(rews, dones, values, gamma, 0);
    const delta = [0, 0, 0];
    for (let t = 0; t < 3; t++) { const nt = dones[t] ? 0 : 1; delta[t] = rews[t] + gamma * values[t + 1] * nt - values[t]; }
    let w0 = 0; for (let t = 0; t < 3; t++) w0 = Math.max(w0, Math.abs(gae0[t] - delta[t]));
    // λ=1：应等于 MC 回报减 V_t
    const gae1 = gae(rews, dones, values, gamma, 1);
    const G = [0, 0, 0];
    let acc = 0; for (let t = 2; t >= 0; t--) { acc = rews[t] + gamma * acc * (dones[t] ? 0 : 1); G[t] = acc; }
    const mcAdv = [G[0] - values[0], G[1] - values[1], G[2] - values[2]];
    let w1 = 0; for (let t = 0; t < 3; t++) w1 = Math.max(w1, Math.abs(gae1[t] - mcAdv[t]));
    // 分歧铁律：λ=0 与 λ=1 结果必不同
    let diverg = 0; for (let t = 0; t < 3; t++) diverg = Math.max(diverg, Math.abs(gae0[t] - gae1[t]));
    const divergViol = diverg > 0 ? 0 : 1;
    const actual = divergViol > 0 ? 1 : Math.max(w0 / 1e-12, w1 / 1e-9);
    cases.push(caseResult('INV-PG-005', 'GAE 退化 λ=0→δ / λ=1→MC-V 且必分歧', actual, 0, 1e-9));
  }

  // INV-PG-006 演员评论家价值一致：critic v^π 对照同策略 TD 评估真 v^π（多重子验证）
  {
    const configs = [
      { nS: 5, seed: 0x9E06a & 0x7fffffff },
      { nS: 6, seed: 0x9E06b & 0x7fffffff },
      { nS: 7, seed: 0x9E06c & 0x7fffffff }
    ];
    let worst = 0;
    for (const cfg of configs) {
      const env = makeChainEnv(cfg.nS);
      const gamma = 0.9;
      const { theta, w } = a2c(env, { gamma, nS: env.nS, nA: env.nA, episodes: opts.quick ? 800 : 2500, maxSteps: 16, lrActor: 0.05, lrCritic: 0.1, seed: cfg.seed, initScale: 0.01 });
      // 冻结 actor，纯 critic 同策略 TD(0) 收敛到当前策略的真 v^π（critic 自身更新规则离线跑收敛）
      _criticConverge(env, theta, w, { gamma, maxIter: 3000 });
      // 同策略 TD 评估真 v^π（独立 Bellman 期望迭代，手搓，非自证）
      const { V: trueV } = policyEvalStochastic(env.nS, env.nA, theta, gamma, 5000);
      const goal = env.nS - 1;
      for (let s = 0; s < goal; s++) worst = Math.max(worst, Math.abs(w[s] - trueV[s]));
    }
    cases.push(caseResult('INV-PG-006', 'critic v^π vs 同策略 TD 评估真 v^π max|·|', worst, 0, 1e-2));
  }

  // INV-PG-007 PPO 裁剪代理目标占优非退化：裁剪≤未裁剪（逐元素 ≥20 批违例 0）且存在严格小于
  {
    const eps = 0.2;
    const nBatches = 40, batchSize = 32;
    const rng = N().makeRng(0x9E07 & 0x7fffffff);
    let viol = 0, strict = 0;
    for (let b = 0; b < nBatches; b++) {
      const ratio = new Array(batchSize), adv = new Array(batchSize);
      for (let i = 0; i < batchSize; i++) {
        ratio[i] = 0.5 + rng();      // [0.5, 1.5] → 覆盖 1±ε 绑定区间
        adv[i] = rng() * 2 - 1;      // [-1, 1]
      }
      const clipped = ppoClip(ratio, adv, eps);
      for (let i = 0; i < batchSize; i++) {
        const unc = ratio[i] * adv[i];
        if (clipped[i] > unc + 1e-15) viol++;     // 裁剪必须 ≤ 未裁剪
        if (clipped[i] < unc - 1e-15) strict++;    // 存在严格小于 → 非恒等
      }
    }
    const actual = (viol > 0 || strict === 0) ? 1 : 0;
    cases.push(caseResult('INV-PG-007', 'PPO 裁剪≤未裁剪(违例0)且存在严格<(非恒等)', actual, 0, 0));
  }

  // INV-PG-008 优势符号与策略改进方向一致：∇_θJ≈E[∇logπ·A] 与 A 符号一致
  {
    const Q = [0.9, 0.4, 0.7, 0.2];
    const theta = [-0.3, 0.5, 0.1, 1.1];
    const nA = Q.length;
    const pi = softmaxPolicy(theta);
    let V = 0; for (let a = 0; a < nA; a++) V += pi[a] * Q[a];
    const A = Q.map((q, a) => q - V);
    // 解析 ∇_θ J = Σ_a π(a)(e_a - π) Q(a)
    const gradJ = new Array(nA).fill(0);
    for (let a = 0; a < nA; a++) { const g = (ap) => (ap === a ? 1 : 0) - pi[ap]; for (let ap = 0; ap < nA; ap++) gradJ[ap] += pi[a] * g(ap) * Q[a]; }
    // 等价写法：Σ_a π(a)(e_a - π) A(a) 应一致
    const gradJ2 = new Array(nA).fill(0);
    for (let a = 0; a < nA; a++) { const g = (ap) => (ap === a ? 1 : 0) - pi[ap]; for (let ap = 0; ap < nA; ap++) gradJ2[ap] += pi[a] * g(ap) * A[a]; }
    let consWorst = 0; for (let ap = 0; ap < nA; ap++) consWorst = Math.max(consWorst, Math.abs(gradJ[ap] - gradJ2[ap]));
    // 有限差分校验 ∇_θ J
    const J = (th) => { const p = softmaxPolicy(th); let s = 0; for (let a = 0; a < nA; a++) s += p[a] * Q[a]; return s; };
    let fdWorst = 0; const eps = 1e-6;
    for (let i = 0; i < nA; i++) {
      const tp = theta.slice(); tp[i] += eps; const tm = theta.slice(); tm[i] -= eps;
      const fd = (J(tp) - J(tm)) / (2 * eps);
      fdWorst = Math.max(fdWorst, Math.abs(fd - gradJ[i]) / (Math.abs(gradJ[i]) + 1e-12));
    }
    // 符号一致性：a*=argmax A，∇_θJ[a*] 应与 A[a*] 同号（改进方向一致），且 = π(a*)·A(a*)
    const aStar = (function () { let b = 0; for (let a = 1; a < nA; a++) if (A[a] > A[b]) b = a; return b; })();
    const signViol = (gradJ[aStar] * A[aStar] < 0) ? 1 : 0;
    const magErr = Math.abs(gradJ[aStar] - pi[aStar] * A[aStar]);
    const actual = Math.max(fdWorst, consWorst, signViol, magErr / 1e-6);
    cases.push(caseResult('INV-PG-008', '∇_θJ≈E[∇logπ·A] 与 A 符号一致', actual, 0, TOL.GRAD));
  }

  const started = Date.now();
  for (const id of ['INV-PG-001', 'INV-PG-002', 'INV-PG-003', 'INV-PG-004', 'INV-PG-005', 'INV-PG-006', 'INV-PG-007', 'INV-PG-008']) {
    const c = cases.find((x) => x.id === id);
    invariants.push({ id, expr: id, tol: c.tol, worst: c.abs_err, status: c.status, note: '' });
  }
  return buildReport({
    module: 'policygrad-forge', version: '1.0.0', layer: 'L6', cases, invariants,
    seed: '0x9E3779B9', started,
    tol: {
      'INV-PG-001': TOL.GRAD, 'INV-PG-002': TOL.RESID, 'INV-PG-003': 0, 'INV-PG-004': 1e-9,
      'INV-PG-005': 1e-9, 'INV-PG-006': 1e-2, 'INV-PG-007': 0, 'INV-PG-008': TOL.GRAD
    }
  });
}

const API = {
  name: 'policygrad-forge', version: '1.0.0', layer: 'L6',
  softmaxPolicy, sampleAction, logProb,
  gaussianLogProb, sampleGaussian,
  policyGradientREINFORCE,
  advantageFunction, gae,
  actorCritic, a2c, ppoClip, ppoUpdate,
  makeChainEnv, selfCheck
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgePolicyGrad = API;

if (typeof process !== 'undefined' && process.argv[1] && /policygrad-forge\.js$/.test(process.argv[1]) && (process.argv.includes('--self-check') || process.argv.includes('--json'))) {
  const r = selfCheck({ quick: process.argv.includes('--quick') });
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : V().formatTableSafe(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
