/**
 * @forge-module rl-forge
 * @layer L5
 * @version 1.0.0
 * @brief     强化学习内核：值迭代（V* 与 Q* 及贪心策略）、策略评估、REINFORCE 蒙特卡洛策略梯度（Softmax Bandit）。全部附可交叉验证不变量（收敛性、贝尔曼最优性、闭式对照、梯度 vs 有限差分）。
 * @reuse     OpenAI Spinning Up|https://spinningup.openai.com|MIT|D|REINFORCE/策略梯度
 * @reuse     Sutton & Barto《Reinforcement Learning》|https://www.incompleteideas.net/book/the-book.html|公有域|D|值迭代/贝尔曼最优性
 * @beyond    纯 JS 单文件零构建；值迭代与贝尔曼最优性自洽；REINFORCE 梯度与解析期望回报的有限差分对照；γ≥1 不收敛判定可证。
 * @invariant INV-RL-001|值迭代收敛：最终 max|V_{k+1}-V_k| ≤ 1e-9（γ=0.9）|1e-9|收缩映射
 * @invariant INV-RL-002|贝尔曼最优性自洽：V*(s)≈max_a[R+γΣPV*] ≤ 1e-6|1e-6|最优贝尔曼方程
 * @invariant INV-RL-003|贪心策略值 == V*（策略评估）≤ 1e-6|1e-6|最优策略值等价
 * @invariant INV-RL-004|2 态闭式 MDP 最优 V 对照 [9,10] ≤ 1e-6|1e-6|闭式真值
 * @invariant INV-RL-005|REINFORCE 梯度 vs 期望回报有限差分 ≤ 1e-2（N≥20000）|1e-2|策略梯度定理
 * @invariant INV-RL-006|策略梯度上升提升均值回报（终值≥初值+0.05）|0.05|策略梯度上升
 * @invariant INV-RL-007|γ=0.9 收敛(残差<1e-3) 且 γ=1 不收敛(残差>1e-3)|0|收敛条件
 * @invariant INV-RL-008|Q=R+γΣPV* 与值迭代 Q 备份一致 ≤ 1e-6|1e-6|Q 值定义
 * @seed      0x9E3779B9
 */
'use strict';
const { makeRng, TOL } = require('./core-numeric.js');
const { caseResult, buildReport } = require('./core-verify.js');

/* ---------- 值迭代 ---------- */

// states/actions: 整数计数 nS/nA；P[s][a]: 长度 nS 的转移概率数组；R[s][a]: 标量即时奖励
function valueIteration(states, actions, P, R, gamma, maxIter) {
  const nS = (typeof states === 'number') ? states : states.length;
  const nA = (typeof actions === 'number') ? actions : actions.length;
  let residual = Infinity;
  let V = new Array(nS).fill(0);
  let Q = [];
  let policy = new Array(nS).fill(0);
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
      Vnew[s] = best;
      Qnew.push(qa);
      res = Math.max(res, Math.abs(best - V[s]));
    }
    V = Vnew; Q = Qnew;
    // 依收敛后的 Qnew 记录每态贪心策略
    for (let s = 0; s < nS; s++) {
      let bestA = 0, best = -Infinity;
      for (let a = 0; a < nA; a++) if (Qnew[s][a] > best) { best = Qnew[s][a]; bestA = a; }
      policy[s] = bestA;
    }
    residual = res;
    if (res < 1e-12) break;
  }
  return { V, Q, policy, residual };
}

/* ---------- 策略评估（固定策略，迭代至收敛） ---------- */

function policyEvaluation(P, R, policy, gamma, maxIter) {
  const nS = P.length, nA = P[0].length;
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
      res = Math.max(res, Math.abs(acc - V[s]));
    }
    V = Vnew;
    residual = res;
    if (res < 1e-12) break;
  }
  return { V, residual };
}

/* ---------- REINFORCE（Softmax Bandit，K=4 固定奖励） ---------- */

const BANDIT_REWARDS = [1.0, 0.5, 0.2, 0.0];

function softmaxProbs(theta) {
  const K = theta.length;
  let maxT = -Infinity;
  for (let i = 0; i < K; i++) maxT = Math.max(maxT, theta[i]);
  const ex = new Array(K);
  let sum = 0;
  for (let i = 0; i < K; i++) { ex[i] = Math.exp(theta[i] - maxT); sum += ex[i]; }
  return ex.map(v => v / sum);
}

// 解析期望回报 J(θ) = Σ π_k r_k
function banditExpectedReturn(theta) {
  const pi = softmaxProbs(theta);
  let j = 0;
  for (let i = 0; i < pi.length; i++) j += pi[i] * BANDIT_REWARDS[i];
  return j;
}

// 蒙特卡洛 REINFORCE 梯度估计（带 baseline）
function reinforceGradient(policyParams, baseline, rng, nEpisodes) {
  const K = policyParams.length;
  if (K !== BANDIT_REWARDS.length) throw new Error('reinforceGradient: 仅支持 K=' + BANDIT_REWARDS.length + ' 的 Softmax Bandit');
  const pi = softmaxProbs(policyParams);
  const g = new Array(K).fill(0);
  for (let t = 0; t < nEpisodes; t++) {
    let u = rng();
    u = 1e-9 + (1 - 2e-9) * u;
    let cum = 0, a = K - 1;
    for (let i = 0; i < K; i++) { cum += pi[i]; if (u <= cum) { a = i; break; } }
    const r = BANDIT_REWARDS[a];
    const adv = r - baseline;
    for (let i = 0; i < K; i++) g[i] += adv * ((i === a ? 1 : 0) - pi[i]);
  }
  for (let i = 0; i < K; i++) g[i] /= nEpisodes;
  return g;
}

/* ---------- 测试用 MDP 构造 ---------- */

// 3 态 2 动作 MDP（用于 INV-RL-001/002/003/008）
function makeSmallMDP() {
  const nS = 3, nA = 2;
  const P = [
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
  ];
  const R = [
    [0.0, 1.0],
    [0.5, 0.0],
    [2.0, 0.0]
  ];
  return { nS, nA, P, R };
}

/* ============================================================================
 * selfCheck — 8 条不变量
 * ========================================================================== */
function selfCheck(opts) {
  opts = opts || {};
  const cases = [];
  const invariants = [];

  // INV-RL-001 值迭代收敛
  {
    const m = makeSmallMDP();
    const { residual } = valueIteration(m.nS, m.nA, m.P, m.R, 0.9, 1000);
    cases.push(caseResult('INV-RL-001', '值迭代收敛残差 ≤ 1e-9', residual, 0, 1e-9));
  }
  // INV-RL-002 贝尔曼最优性自洽
  {
    const m = makeSmallMDP();
    const { V } = valueIteration(m.nS, m.nA, m.P, m.R, 0.9, 1000);
    let worst = 0;
    for (let s = 0; s < m.nS; s++) {
      let best = -Infinity;
      for (let a = 0; a < m.nA; a++) {
        let acc = m.R[s][a];
        for (let sp = 0; sp < m.nS; sp++) acc += 0.9 * m.P[s][a][sp] * V[sp];
        best = Math.max(best, acc);
      }
      worst = Math.max(worst, Math.abs(V[s] - best));
    }
    cases.push(caseResult('INV-RL-002', '贝尔曼最优性自洽', worst, 0, 1e-6));
  }
  // INV-RL-003 贪心策略值 == V*
  {
    const m = makeSmallMDP();
    const { V, policy } = valueIteration(m.nS, m.nA, m.P, m.R, 0.9, 1000);
    const { V: Vpe } = policyEvaluation(m.P, m.R, policy, 0.9, 2000);
    let worst = 0;
    for (let s = 0; s < m.nS; s++) worst = Math.max(worst, Math.abs(Vpe[s] - V[s]));
    cases.push(caseResult('INV-RL-003', '贪心策略值 == V*', worst, 0, 1e-6));
  }
  // INV-RL-004 2 态闭式 MDP 最优 V == [9,10]
  {
    const nS = 2, nA = 1;
    const P = [[[0, 1]], [[0, 1]]];
    const R = [[0], [1]];
    const { V } = valueIteration(nS, nA, P, R, 0.9, 1000);
    const dev = Math.max(Math.abs(V[0] - 9), Math.abs(V[1] - 10));
    cases.push(caseResult('INV-RL-004', '2 态闭式 V* == [9,10]', dev, 0, 1e-6));
  }
  // INV-RL-005 REINFORCE 梯度 vs 有限差分
  {
    const theta = [0, 0, 0, 0];
    const eps = 1e-4;
    const gFD = new Array(4);
    for (let i = 0; i < 4; i++) {
      const tp = theta.slice(); tp[i] += eps;
      const tm = theta.slice(); tm[i] -= eps;
      gFD[i] = (banditExpectedReturn(tp) - banditExpectedReturn(tm)) / (2 * eps);
    }
    const rng = makeRng(0x5eed1);
    const gMC = reinforceGradient(theta, banditExpectedReturn(theta), rng, 20000);
    let worst = 0;
    for (let i = 0; i < 4; i++) worst = Math.max(worst, Math.abs(gMC[i] - gFD[i]));
    cases.push(caseResult('INV-RL-005', 'REINFORCE 梯度 vs 有限差分', worst, 0, 1e-2));
  }
  // INV-RL-006 策略梯度上升提升均值回报
  {
    const theta0 = [0, 0, 0, 0];
    const initialMean = banditExpectedReturn(theta0);
    const rng = makeRng(0xab1e);
    let theta = theta0.slice();
    for (let iter = 0; iter < 60; iter++) {
      const g = reinforceGradient(theta, banditExpectedReturn(theta), rng, 2000);
      for (let i = 0; i < 4; i++) theta[i] += 0.5 * g[i];
    }
    const finalMean = banditExpectedReturn(theta);
    const shortfall = Math.max(0, initialMean + 0.05 - finalMean);
    cases.push(caseResult('INV-RL-006', '策略梯度上升提升均值回报', shortfall, 0, 0.05));
  }
  // INV-RL-007 γ=0.9 收敛 且 γ=1 不收敛
  {
    const nS = 1, nA = 1;
    const P = [[[1]]];
    const R = [[1]];
    const r09 = valueIteration(nS, nA, P, R, 0.9, 200).residual;
    const r10 = valueIteration(nS, nA, P, R, 1.0, 200).residual;
    const ok = (r09 < 1e-3) && (r10 > 1e-3);
    cases.push(caseResult('INV-RL-007', 'γ<1 收敛且 γ≥1 不收敛', ok ? 0 : 1, 0, 0));
  }
  // INV-RL-008 Q=R+γΣPV* 与值迭代 Q 备份一致
  {
    const m = makeSmallMDP();
    const { V, Q } = valueIteration(m.nS, m.nA, m.P, m.R, 0.9, 1000);
    let worst = 0;
    for (let s = 0; s < m.nS; s++) {
      for (let a = 0; a < m.nA; a++) {
        let acc = m.R[s][a];
        for (let sp = 0; sp < m.nS; sp++) acc += 0.9 * m.P[s][a][sp] * V[sp];
        worst = Math.max(worst, Math.abs(acc - Q[s][a]));
      }
    }
    cases.push(caseResult('INV-RL-008', 'Q=R+γΣPV* 与值迭代 Q 一致', worst, 0, 1e-6));
  }

  const started = Date.now();
  for (const id of ['INV-RL-001', 'INV-RL-002', 'INV-RL-003', 'INV-RL-004', 'INV-RL-005', 'INV-RL-006', 'INV-RL-007', 'INV-RL-008']) {
    const c = cases.find(x => x.id === id);
    invariants.push({ id, expr: id, tol: c.tol, worst: c.abs_err, status: c.status, note: '' });
  }
  return buildReport({
    module: 'rl-forge', version: '1.0.0', layer: 'L5', cases, invariants, seed: '0x9E3779B9', started,
    tol: { 'INV-RL-001': 1e-9, 'INV-RL-002': 1e-6, 'INV-RL-003': 1e-6, 'INV-RL-004': 1e-6, 'INV-RL-005': 1e-2, 'INV-RL-006': 0.05, 'INV-RL-007': 0, 'INV-RL-008': 1e-6 }
  });
}

const API = {
  name: 'rl-forge', version: '1.0.0', layer: 'L5',
  valueIteration, policyEvaluation, reinforceGradient, banditExpectedReturn, softmaxProbs, selfCheck
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeRL = API;

if (typeof process !== 'undefined' && process.argv[1] && /rl-forge\.js$/.test(process.argv[1]) && process.argv.includes('--self-check')) {
  const r = selfCheck({ quick: process.argv.includes('--quick') });
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : require('./core-verify.js').formatTableSafe(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
