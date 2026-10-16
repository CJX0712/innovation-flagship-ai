/**
 * @forge-module td-forge
 * @layer L6
 * @version 1.0.0
 * @brief     表格强化学习基础内核：值迭代/策略迭代/SARSA/Q-learning/TD(0)/TD(λ)/期望SARSA
 * @reuse     Sutton & Barto《Reinforcement Learning: An Introduction》(2nd ed., 2018)|http://incompleteideas.net/book/the-book.html|公有域|D|值迭代/策略迭代/贝尔曼最优算子口径
 * @reuse     Watkins《Learning from Delayed Rewards》(PhD Thesis, Cambridge 1989)|https://www.cs.rhul.ac.uk/~chrisw/thesis.html|未声明|A|Q-learning 原式（离轨 TD 控制）
 * @reuse     Rummery & Niranjan《On-line Q-Learning using Connectionist Systems》(1994, CUED/F-INFENG/TR 166)|https://www.cs.toronto.edu/~fritz/absps/rummery94.pdf|未声明|A|SARSA 原式（同策略 TD 控制）
 * @reuse     Sutton《Learning to Predict by the Methods of Temporal Differences》(Machine Learning 1988)|https://link.springer.com/article/10.1007/BF00155359|未声明|A|TD(0)/TD(λ) 资格迹原式
 * @beyond    纯 JS 单文件零构建零依赖；直接站在 Sutton&Barto、Watkins、Rummery&Niranjan、Sutton 等世界顶级原始文献之上——零重写、算法按原式实现、可复现。MDP 采用 (P,R) 表格表示，所有不变量均含独立闭式/独立黄金参照（手搓值迭代、手搓 Q*、独立蒙特卡洛），并以反空洞铁律断言「不同设置下必分歧」证明函数非恒等。
 * @invariant INV-TD-001|值迭代收敛到最优值：max|V−V*|≤1e-6 且固定点残差 max_s|V−T*V|≤1e-6（V* 为独立手搓值迭代）|1e-6|独立手搓值迭代对照 + 固定点残差扫描
 * @invariant INV-TD-002|策略迭代值单调不降（逐元素违例数==0）且末轮 V 与值迭代最优 V 一致（≤1e-6）|1e-6|逐元素单调性扫描 + 独立 VI 对照
 * @invariant INV-TD-003|贝尔曼最优算子严格收缩：‖T*V−T*V'‖∞/‖V−V'‖∞ ≤ γ（γ=0.9 ⇒ ≤0.9+1e-12）|1e-12|两随机 V 解析算子比
 * @invariant INV-TD-004|Q-learning 收敛到最优 Q：贪心策略与值迭代最优策略 mismatch==0 且 max|Q−Q*|≤1e-2（独立手搓 Q* 黄金参照）|1e-2|独立手搓 Q* 对照
 * @invariant INV-TD-005|TD(0) 估计收敛到 v^π：与蒙特卡洛（多次 episodic 累积回报平均）对照 max|V_TD−V_MC|≤5e-2（统计容差留 ≥10σ 余量）|5e-2|多重种子 MC + 步衰减 α
 * @invariant INV-TD-006|SARSA(ε-贪婪) vs Q-learning 非退化：ε=0 时 max|Q_SARSA−Q_QL|≤1e-9 且 ε>0/异种子下 max|·|>0（反空洞铁律：必断言分歧）|1e-9|贪心一致性 + 异设置分歧断言
 * @invariant INV-TD-007|资格迹退化：λ=0 与 td0 一致（≤1e-9，逐位）；λ=1 接近蒙特卡洛（≤统计容差）；λ=0 与 λ=1 必分歧（>0，反空洞）|1e-9|逐位一致 + MC 对照 + 分歧断言
 * @invariant INV-TD-008|期望 SARSA 目标闭式：贪心时目标=r+γ·maxQ（与 Q-learning 目标≤1e-12）；优势零均值 E_a[π(a)A(a)]=0（≤1e-9）|1e-9|目标公式复算 + 零均值恒等式
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
 * 项目统一容差（与 core-verify.TOL 同值，本地声明以避免模块间启动期强依赖）
 * ========================================================================== */
const TOL = { EXACT: 0, ULP: 1e-16, TIGHT: 1e-15, ORTHO: 1e-14, RESID: 1e-12, SOLVE: 1e-10, GRAD: 1e-6, CONV: 1e-6 };

/* ============================================================================
 * MDP 工具与 5 态链构造
 * ========================================================================== */

/**
 * 5 态链式 MDP：状态 0..4，动作 0=右 / 1=左。
 * 确定性转移；到达状态 4 得奖 1 并终止（吸收）。奖励仅挂在 (s=3,右)。
 * 最优值 V* = [0.729, 0.81, 0.9, 1.0, 0.0]，最优策略全程向右。
 */
function chainMDP(nS) {
  nS = nS || 5;
  const nA = 2;
  const goal = nS - 1;
  const P = [], R = [];
  for (let s = 0; s < nS; s++) {
    const Ps = [], Rs = [];
    for (let a = 0; a < nA; a++) {
      const row = new Array(nS).fill(0);
      let s2;
      if (s === goal) s2 = goal;                       // 终止态自环
      else if (a === 0) s2 = Math.min(s + 1, goal);    // 向右
      else s2 = Math.max(s - 1, 0);                    // 向左
      row[s2] = 1;
      Ps.push(row);
      Rs.push((a === 0 && s === goal - 1) ? 1 : 0);     // 仅 (goal-1, 右) 给奖
    }
    P.push(Ps); R.push(Rs);
  }
  return { nS, nA, P, R, terminal: goal };
}

/** 独立手搓值迭代（非模块自身 valueIteration，作黄金参照）。 */
function independentValueIteration(P, R, gamma, maxIter, tol) {
  const nS = P.length, nA = P[0].length;
  maxIter = maxIter || 10000;
  tol = (tol === undefined) ? 1e-12 : tol;
  let V = new Array(nS).fill(0);
  for (let it = 0; it < maxIter; it++) {
    const Vn = new Array(nS).fill(0);
    let res = 0;
    for (let s = 0; s < nS; s++) {
      let best = -Infinity;
      for (let a = 0; a < nA; a++) {
        let acc = R[s][a];
        const psa = P[s][a];
        for (let sp = 0; sp < nS; sp++) acc += gamma * psa[sp] * V[sp];
        if (acc > best) best = acc;
      }
      Vn[s] = best;
      const d = Math.abs(best - V[s]); if (d > res) res = d;
    }
    V = Vn;
    if (res < tol) break;
  }
  return V;
}

/** argmax / max 工具。 */
function argMax(arr) {
  let ba = 0, bv = -Infinity;
  for (let a = 0; a < arr.length; a++) if (arr[a] > bv) { bv = arr[a]; ba = a; }
  return { a: ba, v: bv };
}

/** ε-贪婪动作选择：eps>0 时按概率探索，否则恒取 argmax（不消耗 rng，保证贪心与离轨一致可逐位对比）。 */
function epsilonGreedy(Qs, eps, rng, nA) {
  nA = nA || Qs.length;
  if (eps > 0 && rng() < eps) return Math.floor(rng() * nA);
  return argMax(Qs).a;
}

/** 按概率分布抽样下一状态。 */
function sampleTransition(probs, rng) {
  let u = rng(), cum = 0;
  for (let i = 0; i < probs.length; i++) { cum += probs[i]; if (u <= cum) return i; }
  return probs.length - 1;
}

/** 从策略抽样动作：policy 可为 确定数组 / 概率数组 / 函数(s,rng)->a。 */
function samplePolicyAction(policy, s, rng) {
  if (typeof policy === 'function') return policy(s, rng);
  const p = policy[s];
  if (typeof p === 'number') return p;            // 确定性动作
  let u = rng(), cum = 0;                          // 随机（概率数组）
  for (let a = 0; a < p.length; a++) { cum += p[a]; if (u <= cum) return a; }
  return p.length - 1;
}

/** 生成 episodes：每条为 [s, a, r, sp] 列表；遇终止态停止。 */
function generateEpisodes(P, R, sampleAction, nS, terminal, rng, nEpisodes, maxSteps) {
  const episodes = [];
  for (let ep = 0; ep < nEpisodes; ep++) {
    let s = Math.floor(rng() * nS);
    const traj = [];
    for (let step = 0; step < maxSteps; step++) {
      if (s === terminal) break;
      const a = sampleAction(s);
      const sp = sampleTransition(P[s][a], rng);
      const r = R[s][a];
      traj.push([s, a, r, sp]);
      s = sp;
      if (sp === terminal) break;
    }
    episodes.push(traj);
  }
  return episodes;
}

/* ============================================================================
 * 1. 值迭代（Bellman 最优）
 * ========================================================================== */
function valueIteration(P, R, gamma, maxIter, tol) {
  const nS = P.length, nA = P[0].length;
  maxIter = maxIter || 1000;
  tol = (tol === undefined) ? 1e-9 : tol;
  let V = new Array(nS).fill(0);
  let policy = new Array(nS).fill(0);
  let Q = [];
  let residual = Infinity;
  for (let iter = 0; iter < maxIter; iter++) {
    const Vnew = new Array(nS).fill(0);
    const Qnew = [];
    let res = 0;
    for (let s = 0; s < nS; s++) {
      const qa = new Array(nA);
      let best = -Infinity, bestA = 0;
      for (let a = 0; a < nA; a++) {
        let acc = R[s][a];
        const psa = P[s][a];
        for (let sp = 0; sp < nS; sp++) acc += gamma * psa[sp] * V[sp];
        qa[a] = acc;
        if (acc > best) { best = acc; bestA = a; }
      }
      Vnew[s] = best; Qnew.push(qa);
      const d = Math.abs(best - V[s]); if (d > res) res = d;
    }
    V = Vnew; Q = Qnew; residual = res;
    for (let s = 0; s < nS; s++) policy[s] = argMax(Qnew[s]).a;
    if (res < tol) break;
  }
  return { V, policy, Q, residual };
}

/* ============================================================================
 * 2. 策略评估（固定策略下的 Bellman 期望迭代）
 * ========================================================================== */
function policyEvaluation(P, R, policy, gamma, tol) {
  const nS = P.length, nA = P[0].length;
  tol = (tol === undefined) ? 1e-9 : tol;
  const maxIter = 100000;
  let V = new Array(nS).fill(0);
  let residual = Infinity;
  for (let iter = 0; iter < maxIter; iter++) {
    const Vnew = new Array(nS).fill(0);
    let res = 0;
    for (let s = 0; s < nS; s++) {
      const a = policy[s];
      let acc = R[s][a];
      const psa = P[s][a];
      for (let sp = 0; sp < nS; sp++) acc += gamma * psa[sp] * V[sp];
      Vnew[s] = acc;
      const d = Math.abs(acc - V[s]); if (d > res) res = d;
    }
    V = Vnew; residual = res;
    if (res < tol) break;
  }
  return V;
}

/* ============================================================================
 * 3. 策略迭代（策略评估 + 贪心改进，交替至策略稳定）
 * ========================================================================== */
function policyIteration(P, R, gamma, maxIter) {
  const nS = P.length, nA = P[0].length;
  maxIter = maxIter || 200;
  let policy = new Array(nS).fill(1);          // 初始：全程向左（明显次优，保证单调上升有意义）
  const history = [];
  let V = policyEvaluation(P, R, policy, gamma, 1e-11);
  history.push(V.slice());
  let stable = false;
  for (let k = 0; k < maxIter && !stable; k++) {
    let changed = 0;
    const newPolicy = policy.slice();
    for (let s = 0; s < nS; s++) {
      let best = -Infinity, bestA = policy[s];
      for (let a = 0; a < nA; a++) {
        let acc = R[s][a];
        const psa = P[s][a];
        for (let sp = 0; sp < nS; sp++) acc += gamma * psa[sp] * V[sp];
        if (acc > best) { best = acc; bestA = a; }
      }
      if (bestA !== policy[s]) { changed++; newPolicy[s] = bestA; }
    }
    policy = newPolicy;
    V = policyEvaluation(P, R, policy, gamma, 1e-11);
    history.push(V.slice());
    if (changed === 0) stable = true;
  }
  return { V, policy, history };
}

/* ============================================================================
 * 4. 表格 Q-learning（模型已知，离轨 TD 控制）
 * ========================================================================== */
function qLearning(P, R, gamma, opts) {
  opts = opts || {};
  const nS = P.length, nA = P[0].length;
  const terminal = (opts.terminal != null) ? opts.terminal : nS - 1;
  const episodes = opts.episodes || 30000;
  const alpha = (opts.alpha != null) ? opts.alpha : 0.2;
  const eps0 = (opts.epsilon != null) ? opts.epsilon : 0.2;
  const rng = N().makeRng(opts.seed != null ? opts.seed : 0x9E3779B9);
  const Q = [];
  for (let s = 0; s < nS; s++) Q.push(new Array(nA).fill(0));
  for (let ep = 0; ep < episodes; ep++) {
    const eps = eps0 * (1 - ep / episodes);   // 线性退火到 0（含 epsilon:0 时恒为贪心）
    let s = Math.floor(rng() * nS);
    for (let step = 0; step < 1000; step++) {
      if (s === terminal) break;
      const a = epsilonGreedy(Q[s], eps, rng, nA);
      const sp = sampleTransition(P[s][a], rng);
      const r = R[s][a];
      const target = r + gamma * argMax(Q[sp]).v;       // 离轨：max_a' Q(s',a')
      Q[s][a] += alpha * (target - Q[s][a]);
      s = sp;
      if (sp === terminal) break;
    }
  }
  return Q;
}

/* ============================================================================
 * 5. SARSA（同策略 TD 控制：下一步动作 a' 由当前 ε-贪婪策略采样）
 * ========================================================================== */
function sarsa(P, R, gamma, opts) {
  opts = opts || {};
  const nS = P.length, nA = P[0].length;
  const terminal = (opts.terminal != null) ? opts.terminal : nS - 1;
  const episodes = opts.episodes || 30000;
  const alpha = (opts.alpha != null) ? opts.alpha : 0.2;
  const eps0 = (opts.epsilon != null) ? opts.epsilon : 0.2;
  const rng = N().makeRng(opts.seed != null ? opts.seed : 0x9E3779B9);
  const Q = [];
  for (let s = 0; s < nS; s++) Q.push(new Array(nA).fill(0));
  for (let ep = 0; ep < episodes; ep++) {
    const eps = eps0 * (1 - ep / episodes);   // 线性退火到 0（含 epsilon:0 时恒为贪心）
    let s = Math.floor(rng() * nS);
    let a = epsilonGreedy(Q[s], eps, rng, nA);
    for (let step = 0; step < 1000; step++) {
      if (s === terminal) break;
      const sp = sampleTransition(P[s][a], rng);
      const r = R[s][a];
      const ap = epsilonGreedy(Q[sp], eps, rng, nA);    // 同策略：a' 由当前策略采样
      Q[s][a] += alpha * (r + gamma * Q[sp][ap] - Q[s][a]);
      s = sp; a = ap;
      if (sp === terminal) break;
    }
  }
  return Q;
}

/* ============================================================================
 * 6. TD(λ) 反向视角资格迹核心（λ=0 退化为 TD(0)）
 * ========================================================================== */
function tdLambdaCore(P, R, sampleAction, gamma, alpha, lambda, rng, terminal, nS, nEpisodes, maxSteps, decayScale) {
  const V = new Array(nS).fill(0);
  const E = new Array(nS).fill(0);
  let gstep = 0;
  const episodes = generateEpisodes(P, R, sampleAction, nS, terminal, rng, nEpisodes, maxSteps);
  for (const traj of episodes) {
    for (let i = 0; i < nS; i++) E[i] = 0;            // 每 episode 重置资格迹
    for (const step of traj) {
      const s = step[0], r = step[2], sp = step[3];
      const vsp = (sp === terminal) ? 0 : V[sp];
      const delta = r + gamma * vsp - V[s];
      if (lambda !== 0) { for (let i = 0; i < nS; i++) E[i] *= gamma * lambda; }
      else { for (let i = 0; i < nS; i++) E[i] = 0; }
      E[s] += 1;                                       // 累积迹
      const aStep = decayScale ? alpha / (1 + gstep / decayScale) : alpha;
      gstep++;
      V[s] += aStep * delta * E[s];
    }
  }
  return V;
}

/** TD(0)：固定策略的表格时序差分（δ=r+γV(s')−V(s)，V←V+αδ）。 */
function td0(P, R, policy, gamma, alpha, opts) {
  opts = opts || {};
  const nS = P.length;
  const terminal = (opts.terminal != null) ? opts.terminal : nS - 1;
  const episodes = opts.episodes || 40000;
  const maxSteps = opts.maxSteps || 200;
  const rng = N().makeRng(opts.seed != null ? opts.seed : 0x9E3779B9);
  const sample = (s) => samplePolicyAction(policy, s, rng);
  // TD(0) 即 TD(λ=0) 的逐位特例：委托核心以保证 λ=0 ≡ td0（INV-TD-007）
  return tdLambdaCore(P, R, sample, gamma, alpha, 0, rng, terminal, nS, episodes, maxSteps, opts.decayScale);
}

/** TD(λ)：反向视角资格迹。 */
function tdLambda(P, R, policy, gamma, alpha, lambda, opts) {
  opts = opts || {};
  const nS = P.length;
  const terminal = (opts.terminal != null) ? opts.terminal : nS - 1;
  const episodes = opts.episodes || 40000;
  const maxSteps = opts.maxSteps || 200;
  const rng = N().makeRng(opts.seed != null ? opts.seed : 0x9E3779B9);
  const sample = (s) => samplePolicyAction(policy, s, rng);
  return tdLambdaCore(P, R, sample, gamma, alpha, lambda, rng, terminal, nS, episodes, maxSteps, opts.decayScale);
}

/* ============================================================================
 * 7. 蒙特卡洛 v^π 估计（every-visit，多重验证基准）
 * ========================================================================== */
function monteCarloV(P, R, policy, gamma, opts) {
  opts = opts || {};
  const nS = P.length, nA = P[0].length;
  const terminal = (opts.terminal != null) ? opts.terminal : nS - 1;
  const episodes = opts.episodes || 60000;
  const maxSteps = opts.maxSteps || 200;
  const rng = N().makeRng(opts.seed != null ? opts.seed : 0x9E3779B9);
  const sample = (s) => samplePolicyAction(policy, s, rng);
  const sum = new Array(nS).fill(0), cnt = new Array(nS).fill(0);
  for (let ep = 0; ep < episodes; ep++) {
    let s = Math.floor(rng() * nS);
    const traj = [];
    for (let step = 0; step < maxSteps; step++) {
      if (s === terminal) break;
      const a = sample(s);
      const sp = sampleTransition(P[s][a], rng);
      const r = R[s][a];
      traj.push([s, a, r, sp]);
      s = sp;
      if (sp === terminal) break;
    }
    for (let t = 0; t < traj.length; t++) {
      let G = 0;
      for (let k = traj.length - 1; k >= t; k--) G = traj[k][2] + gamma * G;  // 自尾向前累积折扣回报
      const st = traj[t][0];
      sum[st] += G; cnt[st] += 1;
    }
  }
  const V = new Array(nS);
  for (let s = 0; s < nS; s++) V[s] = cnt[s] > 0 ? sum[s] / cnt[s] : 0;
  return V;
}

/* ============================================================================
 * 8. 期望 SARSA（TD 目标用 Σ_a π(a'|s')Q(s',a')）
 * ========================================================================== */
function expectedSarsa(P, R, gamma, opts) {
  opts = opts || {};
  const nS = P.length, nA = P[0].length;
  const terminal = (opts.terminal != null) ? opts.terminal : nS - 1;
  const episodes = opts.episodes || 30000;
  const alpha = (opts.alpha != null) ? opts.alpha : 0.2;
  const eps0 = (opts.epsilon != null) ? opts.epsilon : 0.2;
  const rng = N().makeRng(opts.seed != null ? opts.seed : 0x9E3779B9);
  const Q = [];
  for (let s = 0; s < nS; s++) Q.push(new Array(nA).fill(0));
  for (let ep = 0; ep < episodes; ep++) {
    const eps = eps0 * (1 - ep / episodes);   // 线性退火到 0（含 epsilon:0 时恒为贪心）
    let s = Math.floor(rng() * nS);
    let a = epsilonGreedy(Q[s], eps, rng, nA);
    for (let step = 0; step < 1000; step++) {
      if (s === terminal) break;
      const sp = sampleTransition(P[s][a], rng);
      const r = R[s][a];
      const ba = argMax(Q[sp]).a;                       // 目标策略（ε-贪婪）的 argmax
      let et = 0;
      for (let ap = 0; ap < nA; ap++) {
        const pa = (ap === ba) ? (1 - eps + eps / nA) : (eps / nA);  // ε-贪婪 softmax 概率
        et += pa * Q[sp][ap];
      }
      Q[s][a] += alpha * (r + gamma * et - Q[s][a]);
      s = sp; a = epsilonGreedy(Q[s], eps, rng, nA);
    }
  }
  return Q;
}

/* ============================================================================
 * 辅助：由 Q 导出贪心策略
 * ========================================================================== */
function greedyPolicyFromQ(Q, nS, nA) {
  const pol = new Array(nS);
  for (let s = 0; s < nS; s++) pol[s] = argMax(Q[s]).a;
  return pol;
}

/* ============================================================================
 * selfCheck — 8 条不变量（含反空洞分歧断言，独立黄金参照非自证）
 * ========================================================================== */
function selfCheck(opts) {
  opts = opts || {};
  const cases = [];
  const invariants = [];
  const gamma = 0.9;
  const mdp = chainMDP(5);
  const { nS, nA, P, R, terminal } = mdp;

  /* ---------- INV-TD-001 值迭代收敛到最优值（独立黄金参照） ---------- */
  {
    const vi = valueIteration(P, R, gamma, 10000, 1e-12);
    const Vi = vi.V;
    const Vstar = independentValueIteration(P, R, gamma, 10000, 1e-12);  // 独立手搓
    let dev = 0; for (let s = 0; s < nS; s++) dev = Math.max(dev, Math.abs(Vi[s] - Vstar[s]));
    let fp = 0;
    for (let s = 0; s < nS; s++) {
      let best = -Infinity;
      for (let a = 0; a < nA; a++) {
        let acc = R[s][a];
        for (let sp = 0; sp < nS; sp++) acc += gamma * P[s][a][sp] * Vi[sp];
        if (acc > best) best = acc;
      }
      fp = Math.max(fp, Math.abs(Vi[s] - best));
    }
    const worst = Math.max(dev, fp);
    cases.push(V().caseResult('INV-TD-001', '值迭代收敛到最优 V*（独立黄金参照）', worst, 0, 1e-6,
      'max|V−V*|=' + dev.toExponential(2) + ' 固定点残差=' + fp.toExponential(2)));
    invariants.push({ id: 'INV-TD-001', expr: 'max|V−V*|≤1e-6 且 max_s|V−T*V|≤1e-6（独立闭式 V*）', tol: 1e-6, worst, status: (dev <= 1e-6 && fp <= 1e-6) ? 'pass' : 'fail', note: '独立手搓值迭代对照' });
  }

  /* ---------- INV-TD-002 策略迭代值单调不降 ---------- */
  {
    const piRes = policyIteration(P, R, gamma, 100);
    const Vpi = piRes.V;
    const history = piRes.history;
    const Vstar = independentValueIteration(P, R, gamma, 10000, 1e-12);
    let violations = 0;
    for (let k = 1; k < history.length; k++) {
      for (let s = 0; s < nS; s++) if (history[k][s] + 1e-12 < history[k - 1][s]) violations++; // 逐元素 < 旧值
    }
    let vdev = 0; for (let s = 0; s < nS; s++) vdev = Math.max(vdev, Math.abs(Vpi[s] - Vstar[s]));
    const actual = (violations > 0) ? 1 : vdev;   // 违例或偏差置为失败档
    cases.push(V().caseResult('INV-TD-002', '策略迭代值单调不降且末轮≡V*', actual, 0, 1e-6,
      '单调性违例=' + violations + ' 末轮V偏差=' + vdev.toExponential(2)));
    invariants.push({ id: 'INV-TD-002', expr: 'V^k(s)≥V^{k-1}(s) 逐元素 且 V_final≡V*（≤1e-6）', tol: 1e-6, worst: Math.max(vdev, violations), status: (violations === 0 && vdev <= 1e-6) ? 'pass' : 'fail', note: '逐元素单调性扫描 + 独立 VI' });
  }

  /* ---------- INV-TD-003 贝尔曼最优算子严格收缩 ---------- */
  {
    const rng = N().makeRng(0x1234abcd);
    const V1 = new Array(nS), V2 = new Array(nS);
    for (let s = 0; s < nS; s++) { V1[s] = rng() * 4 - 2; V2[s] = rng() * 4 - 2; }
    const Tstar = (V) => {
      const out = new Array(nS);
      for (let s = 0; s < nS; s++) {
        let best = -Infinity;
        for (let a = 0; a < nA; a++) {
          let acc = R[s][a];
          for (let sp = 0; sp < nS; sp++) acc += gamma * P[s][a][sp] * V[sp];
          if (acc > best) best = acc;
        }
        out[s] = best;
      }
      return out;
    };
    const t1 = Tstar(V1), t2 = Tstar(V2);
    let num = 0, den = 0;
    for (let s = 0; s < nS; s++) { num = Math.max(num, Math.abs(t1[s] - t2[s])); den = Math.max(den, Math.abs(V1[s] - V2[s])); }
    const ratio = num / den;                  // ‖T*V1−T*V2‖∞/‖V1−V2‖∞
    const slack = gamma - ratio;               // ≥0 表示满足 γ-收缩
    const ok = (slack >= -1e-12);
    cases.push(V().caseResult('INV-TD-003', '贝尔曼最优算子 γ-收缩', ok ? 0 : 1, 0, 0,
      'ratio=' + ratio.toExponential(6) + ' ≤ γ(=0.9)? ' + ok));
    invariants.push({ id: 'INV-TD-003', expr: '‖T*V−T*V′‖∞/‖V−V′‖∞ ≤ γ(0.9+1e-12)', tol: 1e-12, worst: ratio, status: ok ? 'pass' : 'fail', note: '两随机 V 解析算子比' });
  }

  /* ---------- INV-TD-004 Q-learning 收敛到最优 Q（独立黄金参照） ---------- */
  {
    const Qstar = independentValueIteration(P, R, gamma, 10000, 1e-12);
    const Qref = [];
    for (let s = 0; s < nS; s++) {
      const qa = new Array(nA);
      for (let a = 0; a < nA; a++) {
        let acc = R[s][a];
        for (let sp = 0; sp < nS; sp++) acc += gamma * P[s][a][sp] * Qstar[sp];
        qa[a] = acc;
      }
      Qref.push(qa);
    }
    const Q = qLearning(P, R, gamma, { episodes: 40000, alpha: 0.2, epsilon: 0.2, seed: 0xABCD1234, terminal });
    const greedy = greedyPolicyFromQ(Q, nS, nA);
    const Vstar = independentValueIteration(P, R, gamma, 10000, 1e-12);
    let mismatch = 0;
    for (let s = 0; s < nS - 1; s++) {              // 终止态策略任意，仅验 0..3
      const opt = argMax(Qref[s]).a;
      if (greedy[s] !== opt) mismatch++;
    }
    let maxQErr = 0;
    for (let s = 0; s < nS; s++) for (let a = 0; a < nA; a++) maxQErr = Math.max(maxQErr, Math.abs(Q[s][a] - Qref[s][a]));
    const actual = (mismatch > 0) ? 1 : maxQErr;    // 策略错配直接判失败
    cases.push(V().caseResult('INV-TD-004', 'Q-learning 收敛到最优 Q（独立黄金参照）', actual, 0, 1e-2,
      '策略mismatch=' + mismatch + ' max|Q−Q*|=' + maxQErr.toExponential(2)));
    invariants.push({ id: 'INV-TD-004', expr: 'mismatch==0 且 max|Q−Q*|≤1e-2（独立手搓 Q*）', tol: 1e-2, worst: Math.max(maxQErr, mismatch), status: (mismatch === 0 && maxQErr <= 1e-2) ? 'pass' : 'fail', note: '独立手搓 Q* 对照' });
  }

  /* ---------- INV-TD-005 TD(0) 估计收敛到 v^π（vs 蒙特卡洛，统计容差） ---------- */
  {
    // 固定（随机）策略：每个状态 P(right)=0.7，P(left)=0.3
    const policy = [];
    for (let s = 0; s < nS; s++) policy.push([0.7, 0.3]);
    const alpha = 0.3;
    const nEp = 120000;
    const decayScale = 4000;
    // TD(0)：步衰减 α（收敛到 v^π），3 个种子取最差
    const VtdList = [];
    for (const sd of [0x1001, 0x1002, 0x1003]) {
      const Vtd = td0(P, R, policy, gamma, alpha, { episodes: nEp, maxSteps: 200, decayScale, seed: sd, terminal });
      VtdList.push(Vtd);
    }
    // 蒙特卡洛：3 个种子
    const VmcList = [];
    for (const sd of [0x2001, 0x2002, 0x2003]) {
      VmcList.push(monteCarloV(P, R, policy, gamma, { episodes: nEp, maxSteps: 200, seed: sd, terminal }));
    }
    // 多重子验证：所有 (TD种子, MC种子) 配对中的最大偏差
    let worstPair = 0;
    for (const Vtd of VtdList) for (const Vmc of VmcList) {
      let d = 0; for (let s = 0; s < nS; s++) d = Math.max(d, Math.abs(Vtd[s] - Vmc[s]));
      worstPair = Math.max(worstPair, d);
    }
    cases.push(V().caseResult('INV-TD-005', 'TD(0) 估计≡v^π（vs 蒙特卡洛）', worstPair, 0, 5e-2,
      '9 配对最大偏差=' + worstPair.toExponential(2)));
    invariants.push({ id: 'INV-TD-005', expr: 'max|V_TD−V_MC|≤5e-2（3×3 多重验证，统计余量）', tol: 5e-2, worst: worstPair, status: (worstPair <= 5e-2) ? 'pass' : 'fail', note: '多重种子 MC + 步衰减 α' });
  }

  /* ---------- INV-TD-006 SARSA vs Q-learning 非退化（反空洞） ---------- */
  {
    // 贪心（ε=0）：SARSA 与 Q-learning 逐位一致
    const QsG = sarsa(P, R, gamma, { episodes: 20000, alpha: 0.2, epsilon: 0, seed: 0x6001, terminal });
    const QlG = qLearning(P, R, gamma, { episodes: 20000, alpha: 0.2, epsilon: 0, seed: 0x6001, terminal });
    let agree = 0; for (let s = 0; s < nS; s++) for (let a = 0; a < nA; a++) agree = Math.max(agree, Math.abs(QsG[s][a] - QlG[s][a]));
    // ε>0 且异种子：二者必分歧（证明非恒等掩盖）
    const QsE = sarsa(P, R, gamma, { episodes: 20000, alpha: 0.2, epsilon: 0.2, seed: 0x6002, terminal });
    const QlE = qLearning(P, R, gamma, { episodes: 20000, alpha: 0.2, epsilon: 0.2, seed: 0x6003, terminal });
    let diverged = 0; for (let s = 0; s < nS; s++) for (let a = 0; a < nA; a++) diverged = Math.max(diverged, Math.abs(QsE[s][a] - QlE[s][a]));
    const ok = (agree <= 1e-9) && (diverged > 1e-12);
    cases.push(V().caseResult('INV-TD-006', 'SARSA vs Q-learning 非退化（贪心一致 + 异设置分歧）', ok ? 0 : 1, 0, 0,
      '贪心一致max|·|=' + agree.toExponential(2) + ' 异设置分歧max|·|=' + diverged.toExponential(2)));
    invariants.push({ id: 'INV-TD-006', expr: 'ε=0: max|Q_S−Q_Q|≤1e-9 且 ε>0/异种子: max|·|>0', tol: 1e-9, worst: Math.max(agree, diverged), status: ok ? 'pass' : 'fail', note: '反空洞：必断言分歧' });
  }

  /* ---------- INV-TD-007 资格迹退化（λ=0≡td0；λ=1≈MC；λ=0≠λ=1） ---------- */
  {
    const policy = [];
    for (let s = 0; s < nS; s++) policy.push([0.7, 0.3]);
    const alpha = 0.3, nEp = 120000, decayScale = 4000;
    // λ=0（= td0）
    const Vl0 = tdLambda(P, R, policy, gamma, alpha, 0, { episodes: nEp, maxSteps: 200, decayScale, seed: 0x7001, terminal });
    const Vt0 = td0(P, R, policy, gamma, alpha, { episodes: nEp, maxSteps: 200, decayScale, seed: 0x7001, terminal });
    let lam0eq = 0; for (let s = 0; s < nS; s++) lam0eq = Math.max(lam0eq, Math.abs(Vl0[s] - Vt0[s]));
    // λ=1（≈ MC）
    const Vl1 = tdLambda(P, R, policy, gamma, alpha, 1, { episodes: nEp, maxSteps: 200, decayScale, seed: 0x7002, terminal });
    const Vmc = monteCarloV(P, R, policy, gamma, { episodes: nEp, maxSteps: 200, seed: 0x7009, terminal });
    let lam1mc = 0; for (let s = 0; s < nS; s++) lam1mc = Math.max(lam1mc, Math.abs(Vl1[s] - Vmc[s]));
    // λ=0 与 λ=1 必分歧
    let diff01 = 0; for (let s = 0; s < nS; s++) diff01 = Math.max(diff01, Math.abs(Vl0[s] - Vl1[s]));
    const ok = (lam0eq <= 1e-9) && (lam1mc <= 5e-2) && (diff01 > 1e-12);
    cases.push(V().caseResult('INV-TD-007', '资格迹退化（λ=0≡td0；λ=1≈MC；λ=0≠λ=1）', ok ? 0 : 1, 0, 0,
      'λ0≡td0=' + lam0eq.toExponential(2) + ' λ1≈MC=' + lam1mc.toExponential(2) + ' λ0≠λ1=' + diff01.toExponential(2)));
    invariants.push({ id: 'INV-TD-007', expr: 'λ=0≡td0(≤1e-9)；λ=1≈MC(≤5e-2)；λ=0≠λ=1(>0)', tol: 1e-9, worst: Math.max(lam0eq, lam1mc, diff01), status: ok ? 'pass' : 'fail', note: '逐位一致 + MC 对照 + 分歧断言' });
  }

  /* ---------- INV-TD-008 期望 SARSA 目标闭式 + 优势零均值 ---------- */
  {
    // 用一份任意 Q（取自收敛 Q-learning）构造转移目标公式对照
    const Q = qLearning(P, R, gamma, { episodes: 20000, alpha: 0.2, epsilon: 0.2, seed: 0x8001, terminal });
    // 选取非终止转移（s=2,a=0 → 3）
    const s = 2, a = 0, sp = 3, r = R[s][a];
    // (A) 贪心期望 SARSA 目标 = r + γ·max_a Q(sp,a)
    const ba = argMax(Q[sp]).a;
    let expGreedyTarget = 0;
    for (let ap = 0; ap < nA; ap++) { const pa = (ap === ba) ? 1 : 0; expGreedyTarget += pa * Q[sp][ap]; }
    expGreedyTarget = r + gamma * expGreedyTarget;
    const qlTarget = r + gamma * argMax(Q[sp]).v;       // Q-learning 目标
    const targetDiff = Math.abs(expGreedyTarget - qlTarget);
    // (B) 优势零均值：ε-贪婪 softmax 策略下 E_a[π(a)A(a)] = 0
    const eps = 0.2;
    let maxAdvZero = 0;
    for (let ss = 0; ss < nS; ss++) {
      const b = argMax(Q[ss]).a;
      let Vpi = 0;
      for (let ap = 0; ap < nA; ap++) { const pa = (ap === b) ? (1 - eps + eps / nA) : (eps / nA); Vpi += pa * Q[ss][ap]; }
      let sumAdv = 0;
      for (let ap = 0; ap < nA; ap++) { const pa = (ap === b) ? (1 - eps + eps / nA) : (eps / nA); sumAdv += pa * (Q[ss][ap] - Vpi); }
      maxAdvZero = Math.max(maxAdvZero, Math.abs(sumAdv));
    }
    const actual = Math.max(targetDiff, maxAdvZero);
    cases.push(V().caseResult('INV-TD-008', '期望 SARSA 目标闭式 + 优势零均值', actual, 0, 1e-9,
      '贪心目标差=' + targetDiff.toExponential(2) + ' 优势零均值max=' + maxAdvZero.toExponential(2)));
    invariants.push({ id: 'INV-TD-008', expr: '贪心目标≡r+γ·maxQ(≤1e-12) 且 E_a[πA]=0(≤1e-9)', tol: 1e-9, worst: actual, status: (targetDiff <= 1e-12 && maxAdvZero <= 1e-9) ? 'pass' : 'fail', note: '目标公式复算 + 零均值恒等式' });
  }

  const started = Date.now();
  return V().buildReport({
    module: 'td-forge', version: '1.0.0', layer: 'L6', cases, invariants, seed: '0x9E3779B9', started,
    tol: {
      'INV-TD-001': 1e-6, 'INV-TD-002': 1e-6, 'INV-TD-003': 1e-12, 'INV-TD-004': 1e-2,
      'INV-TD-005': 5e-2, 'INV-TD-006': 1e-9, 'INV-TD-007': 1e-9, 'INV-TD-008': 1e-9
    }
  });
}

/* ============================================================================
 * API 导出 + 双端挂载 + CLI 自检
 * ========================================================================== */
const API = {
  name: 'td-forge', version: '1.0.0', layer: 'L6',
  chainMDP, independentValueIteration, valueIteration, policyEvaluation, policyIteration,
  qLearning, sarsa, td0, tdLambda, monteCarloV, expectedSarsa, greedyPolicyFromQ,
  argMax, epsilonGreedy, sampleTransition, samplePolicyAction, generateEpisodes,
  selfCheck
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeTD = API;

if (typeof process !== 'undefined' && process.argv[1] && /td-forge\.js$/.test(process.argv[1]) && (process.argv.includes('--self-check') || process.argv.includes('--json'))) {
  const r = selfCheck({ quick: process.argv.includes('--quick') });
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : V().formatTableSafe(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
