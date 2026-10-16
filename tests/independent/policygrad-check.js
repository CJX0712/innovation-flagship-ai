'use strict';
/* ============================================================================
 * 独立复核脚本：policygrad-forge（不调用模块 selfCheck）
 * 方法：手搓闭式 / 中心差分 / 独立蒙特卡洛 / 独立值迭代，对关键不变量做交叉验证。
 *   覆盖：INV-PG-001（高斯策略梯度检验）、INV-PG-003（REINFORCE 收敛到最优策略）、
 *         INV-PG-005（GAE 退化 λ=0→TD残差 / λ=1→MC回报减V 且必分歧）、
 *         附 INV-PG-004（优势零均值）、INV-PG-007（PPO 裁剪占优非退化）作为独立加强。
 * 输出：✅/✗ 摘要；退出码 0 成功 / 1 失败。
 * ========================================================================== */
const path = require('path');
const PG = require(path.join(__dirname, '..', '..', 'src', 'policygrad-forge.js'));

let allPass = true;
const results = [];
function check(name, pass, detail) {
  results.push({ name, pass, detail });
  if (!pass) allPass = false;
  console.log((pass ? '✅' : '❌') + ' ' + name + (detail ? '  — ' + detail : ''));
}

/* ---------- 手搓工具 ---------- */
function softmax(logits) {
  let mx = -Infinity; for (const v of logits) if (v > mx) mx = v;
  const ex = logits.map((v) => Math.exp(v - mx));
  const s = ex.reduce((a, b) => a + b, 0);
  return ex.map((v) => v / s);
}
function centralDiff(f, x, eps) {
  return (f(x + eps) - f(x - eps)) / (2 * eps);
}

/* ============================================================================
 * INV-PG-001：高斯策略梯度检验（独立中心差分 vs 手搓解析梯度）
 *   手搓：∇_μ logπ = (a-μ)/σ² ， ∇_σ logπ = ((a-μ)²/σ² - 1)/σ
 * ========================================================================== */
(function () {
  const N = 30;
  let worst = 0;
  // 固定一组确定性采样点（不依赖模块内部 rng，手搓均匀网格）
  for (let k = 0; k < N; k++) {
    const mean = -1 + 2 * (k / (N - 1));
    const logStd = -0.75 + 1.5 * ((k * 7) % N) / (N - 1);
    const a = -2 + 4 * ((k * 3) % N) / (N - 1);
    const sigma = Math.exp(logStd);
    // 对 μ
    const fMu = (mu) => PG.gaussianLogProb(mu, logStd, a);
    const dMu = (mu) => (a - mu) / (sigma * sigma);
    const fdMu = centralDiff(fMu, mean, 1e-6);
    const relMu = Math.abs(fdMu - dMu(mean)) / (Math.abs(dMu(mean)) + 1e-12);
    worst = Math.max(worst, relMu);
    // 对 σ（参数即 σ）
    const sig0 = 0.3 + ((k % 5) + 1) * 0.2; // 避免 σ 过小
    const fSig = (sig) => PG.gaussianLogProb(mean, Math.log(sig), a);
    const dSig = (sig) => ((a - mean) * (a - mean) / (sig * sig) - 1) / sig;
    const fdSig = centralDiff(fSig, sig0, 1e-6);
    const relSig = Math.abs(fdSig - dSig(sig0)) / (Math.abs(dSig(sig0)) + 1e-12);
    worst = Math.max(worst, relSig);
  }
  check('INV-PG-001 高斯策略梯度 vs 中心差分 max 相对误差 ≤1e-6', worst <= 1e-6, 'maxRelErr=' + worst.toExponential(3));
})();

/* ============================================================================
 * INV-PG-003：REINFORCE 收敛到最优策略（独立手搓值迭代作黄金参照）
 *   5 态链 MDP：最优策略为全「向右」(action 0)。
 * ========================================================================== */
(function () {
  const nS = 5, nA = 2, goal = nS - 1, gamma = 0.9;

  // 独立手搓值迭代
  function handValueIteration() {
    let V = new Array(nS).fill(0);
    for (let iter = 0; iter < 1000; iter++) {
      const Vn = new Array(nS).fill(0);
      let res = 0;
      for (let s = 0; s < nS; s++) {
        const q = [];
        for (let a = 0; a < nA; a++) {
          const r = (a === 0 && s === goal - 1) ? 1 : 0;
          const s2 = (s === goal) ? goal : (a === 0 ? Math.min(s + 1, goal) : Math.max(s - 1, 0));
          const done = (s2 === goal) ? 1 : 0;
          q.push(r + gamma * (done ? 0 : V[s2]));
        }
        Vn[s] = Math.max(q[0], q[1]);
        res = Math.max(res, Math.abs(Vn[s] - V[s]));
      }
      V = Vn;
      if (res < 1e-12) break;
    }
    // 回溯贪心策略（终态取 action 0）
    const pi = new Array(nS).fill(0);
    for (let s = 0; s < nS; s++) {
      const q = [];
      for (let a = 0; a < nA; a++) {
        const r = (a === 0 && s === goal - 1) ? 1 : 0;
        const s2 = (s === goal) ? goal : (a === 0 ? Math.min(s + 1, goal) : Math.max(s - 1, 0));
        const done = (s2 === goal) ? 1 : 0;
        q.push(r + gamma * (done ? 0 : V[s2]));
      }
      let best = 0; for (let a = 1; a < nA; a++) if (q[a] > q[best]) best = a;
      pi[s] = best;
    }
    return { V, pi };
  }
  const vi = handValueIteration();

  // 交叉验证：模块 makeChainEnv 的 spec 与手搓一致（同源性审计）
  const env = PG.makeChainEnv(nS);
  const spec = env.spec();
  let specDiff = 0;
  for (let s = 0; s < nS; s++) for (let a = 0; a < nA; a++) {
    const r = (a === 0 && s === goal - 1) ? 1 : 0;
    const s2 = (s === goal) ? goal : (a === 0 ? Math.min(s + 1, goal) : Math.max(s - 1, 0));
    if (spec.R[s][a] !== r) specDiff++;
    if (spec.P[s][a][s2] !== 1) specDiff++;
  }

  // 模块训练 REINFORCE
  const theta = PG.policyGradientREINFORCE(env, { gamma, nS, nA, episodes: 2500, maxSteps: 16, baseline: 'none', lr: 0.2, seed: 0x9E03, initScale: 0.01 });

  // 训练后贪心策略（非终止态全集）
  let mismatch = 0;
  for (let s = 0; s < goal; s++) {
    const lg = []; for (let a = 0; a < nA; a++) lg.push(theta[a * nS + s]);
    const p = softmax(lg);
    let best = 0; for (let a = 1; a < nA; a++) if (p[a] > p[best]) best = a;
    if (best !== vi.pi[s]) mismatch++;
  }
  // 最优策略应全为向右 (action 0)
  const optimalAllRight = vi.pi.slice(0, goal).every((a) => a === 0);

  check('INV-PG-003 REINFORCE 贪心策略 vs 独立值迭代最优策略 (mismatch=0)',
    mismatch === 0 && specDiff === 0 && optimalAllRight,
    'mismatch=' + mismatch + ' specDiff=' + specDiff + ' optimalAllRight=' + optimalAllRight);
})();

/* ============================================================================
 * INV-PG-005：GAE 退化（独立闭式对照 + 必分歧铁律）
 *   λ=0 → TD 残差 δ；λ=1 → MC 回报减 V_t；且 λ=0 与 λ=1 必分歧。
 * ========================================================================== */
(function () {
  const rews = [0, 1, 0];
  const dones = [0, 0, 1];
  const values = [0.1, 0.2, 0.3, 0.4]; // 长度 T+1
  const gamma = 0.9;

  function gaeRef(rews, dones, values, gamma, lam) {
    const T = rews.length;
    const A = new Array(T).fill(0);
    let g = 0;
    for (let t = T - 1; t >= 0; t--) {
      const nt = dones[t] ? 0 : 1;
      const delta = rews[t] + gamma * values[t + 1] * nt - values[t];
      g = delta + gamma * lam * g * nt;
      A[t] = g;
    }
    return A;
  }
  // 独立 TD 残差
  const delta = rews.map((r, t) => r + gamma * values[t + 1] * (dones[t] ? 0 : 1) - values[t]);
  // 独立 MC 回报
  const G = []; let acc = 0;
  for (let t = rews.length - 1; t >= 0; t--) { acc = rews[t] + gamma * acc * (dones[t] ? 0 : 1); G[t] = acc; }
  const mcAdv = G.map((g, t) => g - values[t]);

  const g0 = PG.gae(rews, dones, values, gamma, 0);
  const g1 = PG.gae(rews, dones, values, gamma, 1);
  const d0 = gaeRef(rews, dones, values, gamma, 0);

  let w0 = 0, w1 = 0, diverg = 0;
  for (let t = 0; t < rews.length; t++) {
    w0 = Math.max(w0, Math.abs(g0[t] - delta[t]), Math.abs(g0[t] - d0[t]));
    w1 = Math.max(w1, Math.abs(g1[t] - mcAdv[t]));
    diverg = Math.max(diverg, Math.abs(g0[t] - g1[t]));
  }
  check('INV-PG-005 GAE 退化 λ=0→δ(≤1e-12) / λ=1→MC-V(≤1e-9) 且 λ=0≠λ=1',
    w0 <= 1e-12 && w1 <= 1e-9 && diverg > 0,
    'w0=' + w0.toExponential(2) + ' w1=' + w1.toExponential(2) + ' diverg=' + diverg.toExponential(2));
})();

/* ============================================================================
 * INV-PG-004（独立加强）：优势函数零均值 Σ_a π(a|s)A(s,a)≈0
 * ========================================================================== */
(function () {
  // 手搓确定性 Q/V/π（与模块无关）
  const nS = 6, nA = 4;
  let worst = 0;
  for (let s = 0; s < nS; s++) {
    const logits = []; for (let a = 0; a < nA; a++) logits.push(((s * 31 + a * 17) % 13) / 6 - 1);
    const pi = softmax(logits);
    const Q = []; for (let a = 0; a < nA; a++) Q.push(((s * 7 + a * 5) % 11) / 5 - 1);
    let V = 0; for (let a = 0; a < nA; a++) V += pi[a] * Q[a];
    // 用模块 advantageFunction 计算 A，再独立验证零均值
    const A = PG.advantageFunction([Q], [V], [pi], 0);
    let weighted = 0; for (let a = 0; a < nA; a++) weighted += pi[a] * A[a];
    worst = Math.max(worst, Math.abs(weighted));
  }
  check('INV-PG-004 优势函数零均值 ΣπA≈0 (≤1e-9)', worst <= 1e-9, 'max|ΣπA|=' + worst.toExponential(2));
})();

/* ============================================================================
 * INV-PG-007（独立加强）：PPO 裁剪代理目标占优非退化
 *   裁剪 ≤ 未裁剪（逐元素）且存在严格小于（证明非恒等）。
 * ========================================================================== */
(function () {
  const eps = 0.2, nBatches = 40, B = 32;
  // 确定性伪随机（手搓 LCG），避免依赖模块 rng
  let st = 0x1234;
  const rnd = () => { st = (1103515245 * st + 12345) >>> 0; return st / 4294967296; };
  let viol = 0, strict = 0;
  for (let b = 0; b < nBatches; b++) {
    const ratio = [], adv = [];
    for (let i = 0; i < B; i++) { ratio.push(0.5 + rnd()); adv.push(rnd() * 2 - 1); }
    const clipped = PG.ppoClip(ratio, adv, eps);
    for (let i = 0; i < B; i++) {
      const unc = ratio[i] * adv[i];
      if (clipped[i] > unc + 1e-15) viol++;
      if (clipped[i] < unc - 1e-15) strict++;
    }
  }
  check('INV-PG-007 PPO 裁剪≤未裁剪(违例0) 且存在严格<(非恒等)', viol === 0 && strict > 0, 'viol=' + viol + ' strict=' + strict);
})();

console.log('\n' + (allPass ? '✅ 独立复核全部通过' : '❌ 独立复核存在失败'));
process.exit(allPass ? 0 : 1);
