'use strict';
/* ============================================================================
 * 独立复核 INV-DQN-007「DQN 收敛到值迭代」——不采信模块自检的独立黄金参照。
 *
 * 方法（三重独立）：
 *   1. 手搓 P/R（终态两动作自环，与 makeChainEnv.next 语义一致）喂 rl-forge
 *      valueIteration，对照手工闭式解 V*=[0.729,0.81,0.9,1.0,0]；
 *   2. 交叉验证 env.spec() 导出的 P/R 与手搓 P/R 逐元素一致（同源性审计）；
 *   3. 用 3 个不同种子训练 DQN，贪心策略 vs VI 最优策略、Q 值 vs VI 的 Q*、
 *      贝尔曼残差（用 qNet 独立复算，不读 trained.bellmanResidual）。
 * ========================================================================== */
const path = require('path');
const DQN = require(path.join(__dirname, '..', '..', 'src', 'dqn-forge.js'));
const RL = require(path.join(__dirname, '..', '..', 'src', 'rl-forge.js'));

const nS = 5, nA = 2, goal = nS - 1, gamma = 0.9;

/* ---------- 1. 手搓 P/R（终态两动作自环） ---------- */
const P = [], R = [];
for (let s = 0; s < nS; s++) {
  const Ps = [], Rs = [];
  for (let a = 0; a < nA; a++) {
    const row = new Array(nS).fill(0);
    const s2 = (s === goal) ? goal : (a === 0 ? Math.min(s + 1, goal) : Math.max(s - 1, 0));
    row[s2] = 1;
    Ps.push(row);
    Rs.push((a === 0 && s === goal - 1) ? 1 : 0); // 稀疏奖励：仅 (3,前进) 得 1
  }
  P.push(Ps); R.push(Rs);
}

const vi = RL.valueIteration(nS, nA, P, R, gamma, 10000);
console.log('[VI-手搓] V* = [' + vi.V.map((v) => v.toFixed(6)).join(', ') + ']');
console.log('[VI-手搓] π* = [' + vi.policy.join(', ') + ']');

/* ---------- 2. 交叉审计：env.spec() vs 手搓 P/R ---------- */
const env = DQN.makeChainEnv(nS);
const spec = env.spec();
let specDiff = 0;
for (let s = 0; s < nS; s++) for (let a = 0; a < nA; a++) {
  if (spec.R[s][a] !== R[s][a]) specDiff++;
  for (let sp = 0; sp < nS; sp++) if (spec.P[s][a][sp] !== P[s][a][sp]) specDiff++;
}
console.log('[交叉审计] env.spec() vs 手搓 P/R 差异项 = ' + specDiff + ' ' + (specDiff === 0 ? '✅ 同源一致' : '❌ 不一致'));

/* ---------- 3. 对照手工闭式 V* ---------- */
const expect = [0.729, 0.81, 0.9, 1.0, 0.0];
let maxVErr = 0;
for (let i = 0; i < nS; i++) maxVErr = Math.max(maxVErr, Math.abs(vi.V[i] - expect[i]));
console.log('[VI-手搓] vs 手工闭式 V* 最大误差 = ' + maxVErr.toExponential(3) + (maxVErr < 1e-12 ? ' ✅' : ' ❌'));

/* ---------- 4. 多种子训练 DQN 并独立复算 ---------- */
function independentResidual(qParams) {
  // 不读 trained.bellmanResidual——用 qNet + 手搓 P/R 自己复算（非终止态全集）
  let worst = 0;
  for (let s = 0; s < nS - 1; s++) {
    const q = DQN.qNet(qParams, env.stateAt(s));
    for (let a = 0; a < nA; a++) {
      const s2 = (s === goal) ? goal : (a === 0 ? Math.min(s + 1, goal) : Math.max(s - 1, 0));
      const done = (s2 === goal) ? 1 : 0;
      const qNext = DQN.qNet(qParams, env.stateAt(s2));
      let mx = -Infinity;
      for (let ap = 0; ap < nA; ap++) mx = Math.max(mx, qNext[ap]);
      const y = R[s][a] + gamma * mx * (1 - done);
      worst = Math.max(worst, Math.abs(q[a] - y));
    }
  }
  return worst;
}

function runOnce(seed) {
  const trained = DQN.trainDQN(env, {
    gamma, episodes: 800, maxSteps: 16, capacity: 4000, batch: 32, lr: 0.05,
    targetSync: 10, initScale: 0, seed,
    eps: { mode: 'linear', epsStart: 1.0, epsEnd: 0.05, decaySteps: 400 },
  });
  const states = [];
  for (let i = 0; i < nS - 1; i++) states.push(env.stateAt(i));
  const greedy = DQN.greedyPolicy(trained.qParams, states);
  let mismatch = 0;
  for (let i = 0; i < states.length; i++) if (greedy[i] !== vi.policy[i]) mismatch++;
  let maxQErr = 0;
  for (let i = 0; i < nS - 1; i++) {
    const q = DQN.qNet(trained.qParams, env.stateAt(i));
    for (let a = 0; a < nA; a++) maxQErr = Math.max(maxQErr, Math.abs(q[a] - vi.Q[i][a]));
  }
  const res = independentResidual(trained.qParams);
  console.log('[DQN seed=0x' + seed.toString(16) + '] 策略错配=' + mismatch +
    ' Q值最大误差=' + maxQErr.toExponential(3) +
    ' 独立复算贝尔曼残差=' + res.toExponential(3) +
    ' 模块自报残差=' + trained.bellmanResidual.toExponential(3) +
    ' 贪心策略=[' + greedy.join(',') + ']');
  return { mismatch, maxQErr, res };
}

const runs = [runOnce(0xD007), runOnce(0x51EE), runOnce(0xABCD)];

const pass = specDiff === 0 && maxVErr < 1e-12 &&
  runs.every((r) => r.mismatch === 0) &&
  runs.every((r) => r.res < 1e-2) &&
  runs.every((r) => r.maxQErr < 5e-2);
console.log(pass
  ? '\n✅ INV-DQN-007 独立复核通过：手搓黄金参照 + spec 同源审计 + 3 种子稳健性 + 独立残差复算'
  : '\n❌ INV-DQN-007 独立复核失败');
process.exit(pass ? 0 : 1);
