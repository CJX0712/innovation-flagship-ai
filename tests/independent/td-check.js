'use strict';
/* ============================================================================
 * 独立复核 td-forge —— 不采信模块自检，手搓闭式 / 独立蒙特卡洛交叉验证。
 *
 * 覆盖关键不变量：
 *   INV-TD-001  手搓 V* 对照：模块 valueIteration 的 V 与手搓 V* 一致，且固定点残差 ≤1e-6
 *   INV-TD-004  Q-learning 收敛：贪心策略与最优策略一致、max|Q−Q*| ≤1e-2（Q* 手搓）
 *   INV-TD-003  收缩性：‖T*V−T*V'‖∞/‖V−V'‖∞ ≤ γ(0.9)
 *
 * 方法：手搓 P/R（与 td-forge.chainMDP 同构：5 态链，状态 4 终止给奖 1），
 *       手搓值迭代得 V* 与 Q*，再调用模块函数导出量做对照。
 * ========================================================================== */
const path = require('path');
const TD = require(path.join(__dirname, '..', '..', 'src', 'td-forge.js'));

const nS = 5, nA = 2, goal = nS - 1, gamma = 0.9;

/* ---------- 1. 手搓 P/R（与模块 chainMDP 同构） ---------- */
const P = [], R = [];
for (let s = 0; s < nS; s++) {
  const Ps = [], Rs = [];
  for (let a = 0; a < nA; a++) {
    const row = new Array(nS).fill(0);
    let s2;
    if (s === goal) s2 = goal;
    else if (a === 0) s2 = Math.min(s + 1, goal);
    else s2 = Math.max(s - 1, 0);
    row[s2] = 1;
    Ps.push(row);
    Rs.push((a === 0 && s === goal - 1) ? 1 : 0);
  }
  P.push(Ps); R.push(Rs);
}

/* ---------- 2. 手搓值迭代（独立黄金参照） ---------- */
function handValueIteration(P, R, gamma, maxIter, tol) {
  const nS = P.length, nA = P[0].length;
  tol = (tol === undefined) ? 1e-12 : tol;
  let V = new Array(nS).fill(0);
  for (let it = 0; it < maxIter; it++) {
    const Vn = new Array(nS).fill(0);
    let res = 0;
    for (let s = 0; s < nS; s++) {
      let best = -Infinity;
      for (let a = 0; a < nA; a++) {
        let acc = R[s][a];
        for (let sp = 0; sp < nS; sp++) acc += gamma * P[s][a][sp] * V[sp];
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

const Vstar = handValueIteration(P, R, gamma, 10000, 1e-12);
const Qstar = [];
for (let s = 0; s < nS; s++) {
  const qa = new Array(nA);
  for (let a = 0; a < nA; a++) {
    let acc = R[s][a];
    for (let sp = 0; sp < nS; sp++) acc += gamma * P[s][a][sp] * Vstar[sp];
    qa[a] = acc;
  }
  Qstar.push(qa);
}
const optPolicy = [];
for (let s = 0; s < nS; s++) {
  let ba = 0, bv = -Infinity;
  for (let a = 0; a < nA; a++) if (Qstar[s][a] > bv) { bv = Qstar[s][a]; ba = a; }
  optPolicy.push(ba);
}

console.log('[手搓] V* = [' + Vstar.map((v) => v.toFixed(6)).join(', ') + ']');
console.log('[手搓] π* = [' + optPolicy.join(', ') + ']');

/* ---------- 3. INV-TD-001 模块 valueIteration vs 手搓 V* ---------- */
const vi = TD.valueIteration(P, R, gamma, 10000, 1e-12);
let maxVerr = 0;
for (let s = 0; s < nS; s++) maxVerr = Math.max(maxVerr, Math.abs(vi.V[s] - Vstar[s]));
// 固定点残差
let fpRes = 0;
for (let s = 0; s < nS; s++) {
  let best = -Infinity;
  for (let a = 0; a < nA; a++) {
    let acc = R[s][a];
    for (let sp = 0; sp < nS; sp++) acc += gamma * P[s][a][sp] * vi.V[sp];
    if (acc > best) best = acc;
  }
  fpRes = Math.max(fpRes, Math.abs(vi.V[s] - best));
}
const inv001 = (maxVerr < 1e-6) && (fpRes < 1e-6);
console.log('[INV-TD-001] max|V−V*|=' + maxVerr.toExponential(3) +
  ' 固定点残差=' + fpRes.toExponential(3) + (inv001 ? ' ✅' : ' ❌'));

/* ---------- 4. INV-TD-004 模块 Q-learning vs 手搓 Q* ---------- */
const Q = TD.qLearning(P, R, gamma, { episodes: 40000, alpha: 0.2, epsilon: 0.2, seed: 0xABCD1234, terminal: goal });
const greedy = TD.greedyPolicyFromQ(Q, nS, nA);
let mismatch = 0;
for (let s = 0; s < nS - 1; s++) if (greedy[s] !== optPolicy[s]) mismatch++;
let maxQerr = 0;
for (let s = 0; s < nS; s++) for (let a = 0; a < nA; a++) maxQerr = Math.max(maxQerr, Math.abs(Q[s][a] - Qstar[s][a]));
const inv004 = (mismatch === 0) && (maxQerr < 1e-2);
console.log('[INV-TD-004] 策略mismatch=' + mismatch + ' max|Q−Q*|=' + maxQerr.toExponential(3) + (inv004 ? ' ✅' : ' ❌'));

/* ---------- 5. INV-TD-003 收缩性 ---------- */
// 用确定性伪随机（与模块无关）构造两向量
function lcg(seed) { let z = seed >>> 0; return () => { z = (Math.imul(z, 1664525) + 1013904223) >>> 0; return z / 4294967296; }; }
const rng = lcg(0x51EED);
const V1 = new Array(nS), V2 = new Array(nS);
for (let s = 0; s < nS; s++) { V1[s] = rng() * 4 - 2; V2[s] = rng() * 4 - 2; }
function Tstar(V) {
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
}
const t1 = Tstar(V1), t2 = Tstar(V2);
let num = 0, den = 0;
for (let s = 0; s < nS; s++) { num = Math.max(num, Math.abs(t1[s] - t2[s])); den = Math.max(den, Math.abs(V1[s] - V2[s])); }
const ratio = num / den;
const inv003 = (gamma - ratio) >= -1e-12;
console.log('[INV-TD-003] ‖T*V−T*V′‖∞/‖V−V′‖∞=' + ratio.toExponential(6) + ' ≤ 0.9? ' + inv003 + (inv003 ? ' ✅' : ' ❌'));

/* ---------- 汇总 ---------- */
const pass = inv001 && inv004 && inv003;
console.log(pass
  ? '\n✅ td-forge 独立复核通过：手搓 V* 对照 + 手搓 Q* 对照 + γ-收缩性'
  : '\n❌ td-forge 独立复核失败');
process.exit(pass ? 0 : 1);
