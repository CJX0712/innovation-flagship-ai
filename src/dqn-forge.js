/**
 * @forge-module dqn-forge
 * @layer L6
 * @version 1.0.0
 * @brief     深度强化学习内核（DQN 系）：经验回放（FIFO 环形缓冲）、目标网络（硬同步 / Polyak 软更新）、
 *            ε-贪婪探索与线性/指数 ε 调度、DQN 时序差分更新、Double DQN（在线网络选动作 + 目标网络估值）、
 *            完整 trainDQN 训练循环与贝尔曼残差收敛度量。线性 Q 网 q=W·s+b：one-hot 状态下与表格 Q 完全等价，
 *            可精确复算，并与 rl-forge 值迭代作独立黄金参照交叉验证收敛性。
 * @reuse     Playing Atari with DRL (Mnih et al. 2013)|https://arxiv.org/abs/1312.5602|MIT|A|DQN/经验回放/目标网络原式
 * @reuse     Human-level Control through Deep RL (Mnih et al. 2015 Nature)|https://www.nature.com/articles/nature14236|MIT|A|DQN 训练流程
 * @reuse     Double DQN (van Hasselt et al. 2016)|https://arxiv.org/abs/1509.06461|MIT|A|解耦动作选择与估值的 TD 目标
 * @reuse     Spinning Up (OpenAI)|https://spinningup.openai.com/|MIT|C|算法结构与超参口径
 * @reuse     optim-forge（本项目 src/optim-forge.js）|https://github.com/pytorch/pytorch|MIT|X|sgd/adam 参数更新器零重写迁移包装
 * @reuse     rl-forge（本项目 src/rl-forge.js）|https://www.incompleteideas.net/book/the-book.html|MIT|C|值迭代作收敛独立黄金参照与 MDP 表示约定
 * @beyond    纯 JS 单文件零构建；params 自描述布局 [nS,nA,W,b] 令 qNet(params,state) 双参即用；
 *            one-hot 链式 MDP 上 DQN 训练结果与 rl-forge 值迭代互相印证（黄金参照非自证）；
 *            Double DQN「TD 目标 ≤ 普通 DQN」逐元素有界性以诱惑构造实例验证；Polyak tau=1 与硬同步精确一致。
 * @invariant INV-DQN-001|TD 目标闭式：tdTarget==r+γ·max(qNext)·(1-done)，随机 20 组偏差 ≤1e-12；done 时精确等于 r|1e-12|内联闭式复算
 * @invariant INV-DQN-002|经验回放 FIFO：capacity=64 推入 100 条 → size==64、保留第 37..100 条（逐位断言）、最旧 36 条逐出；sample 不修改缓冲区|0|逐位断言+快照对照
 * @invariant INV-DQN-003|回放采样无偏：20000 次 sample(batch=4) 共 8 万抽，各经验被抽中频率与均匀分布最大偏差 ≤0.03（MCSE≈1.5e-3，约 20σ 余量）|0.03|蒙特卡洛频率检验
 * @invariant INV-DQN-004|ε-贪婪口径：ε=0 恒返回 argmax（随机 20 组×200 次违例 0）；ε=1 频率-均匀 ≤0.03；ε=0.5 频率 vs 0.5·均匀+0.5·argmax ≤0.05|0.05|经验频率 vs 闭式混合分布
 * @invariant INV-DQN-005|ε 调度单调：线性/指数两种模式 step∈[0,2000] 全范围单调不增且 ≥epsEnd，违例计数 0|0|单调性扫描
 * @invariant INV-DQN-006|Polyak 闭式：tau=0.01 时 target==0.01·online+0.99·target（≤1e-12）；tau=1 时 target==online（精确）|1e-12|闭式复算
 * @invariant INV-DQN-007|DQN 收敛到值迭代：5 态链式 MDP 训练后贪心策略与 rl-forge 值迭代最优策略一致，贝尔曼残差 max|Q-(r+γ·maxQ')| ≤1e-2|1e-2|独立黄金参照（非自证）
 * @invariant INV-DQN-008|Double DQN 抑制过估计：诱惑构造 MDP（目标网络含被高估动作）上 TD 目标逐元素 ≤ 普通 DQN，20 组随机批量违例 0|0|max 算子占优 + 构造实例
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

let _O = null;
function O() {
  if (_O) return _O;
  if (typeof require !== 'undefined') { try { _O = require('./optim-forge.js'); } catch (e) {} }
  if (!_O && typeof globalThis !== 'undefined') _O = globalThis.ForgeOptim || null;
  if (!_O) throw new Error('optim-forge 依赖缺失');
  return _O;
}

let _R = null;
function R() {
  if (_R) return _R;
  if (typeof require !== 'undefined') { try { _R = require('./rl-forge.js'); } catch (e) {} }
  if (!_R && typeof globalThis !== 'undefined') _R = globalThis.ForgeRL || null;
  if (!_R) throw new Error('rl-forge 依赖缺失');
  return _R;
}

/* ============================================================================
 * 1. 线性 Q 网络：q(s,a) = W[a]·s + b[a]
 *    params 为自描述 Float64Array：params[0]=nS、params[1]=nA（只读元数据，
 *    训练梯度恒 0，优化器永不移动），其后 W（nA×nS 行主序）再接 b（nA）。
 *    总长 = 2 + nA·(nS+1)。one-hot 状态 s=e_k 时 q(e_k)[a]=W[a][k]+b[a]，
 *    与表格 Q(s=k,a) 完全等价——使 DQN 更新可精确复算并与值迭代互相印证。
 * ========================================================================== */

/** 前向：返回每动作一个 Q 值的 Float64Array（长度 nA）。 */
function qNet(params, state) {
  const nS = params[0], nA = params[1];
  if (!Number.isInteger(nS) || !Number.isInteger(nA) || params.length !== 2 + nA * (nS + 1)) {
    throw new Error('dqn-forge qNet: params 布局非法（应为 [nS,nA,W,b]，长度 2+nA·(nS+1)）');
  }
  if (!state || state.length !== nS) throw new Error('dqn-forge qNet: state 维度应为 ' + nS + '，收到 ' + (state ? state.length : 'null'));
  const q = new Float64Array(nA);
  for (let a = 0; a < nA; a++) {
    let acc = params[2 + nA * nS + a]; // b[a]
    const row = 2 + a * nS;
    for (let j = 0; j < nS; j++) acc += params[row + j] * state[j];
    q[a] = acc;
  }
  return q;
}

/** 构造线性 Q 网参数（小随机初始化；scale=0 即零初始化，one-hot 下无对称死局）。 */
function makeQParams(nS, nA, rng, scale) {
  if (!Number.isInteger(nS) || nS < 1 || !Number.isInteger(nA) || nA < 1) throw new Error('makeQParams: nS/nA 须为正整数');
  scale = (scale === undefined || scale === null) ? 0.01 : scale;
  const params = new Float64Array(2 + nA * (nS + 1));
  params[0] = nS; params[1] = nA;
  for (let i = 2; i < params.length; i++) params[i] = (rng() * 2 - 1) * scale;
  return params;
}

/** 深拷贝参数（Float64Array → 独立 Float64Array）。 */
function cloneParams(p) { return Float64Array.from(p); }

/** 原地覆盖拷贝：dst ← src（源与目标同布局）。 */
function copyParams(dst, src) { dst.set(src); return dst; }

/* ============================================================================
 * 2. 经验回放缓冲区（FIFO 环形缓冲，均匀有放回采样）
 * ========================================================================== */

/**
 * 构造容量为 capacity 的回放缓冲区。
 * push(exp)：exp={s,a,r,s2,done}；超容量时逐出最旧（FIFO）。
 * sample(batch, rng)：有放回均匀抽 batch 条，返回浅拷贝（s/s2 共享引用，
 *   训练为纯函数式不会改写；缓冲区本身不被修改）。
 * toArray()：按时间顺序（旧→新）导出浅拷贝，供 FIFO 逐出断言。
 */
function ReplayBuffer(capacity) {
  if (!Number.isInteger(capacity) || capacity <= 0) throw new Error('ReplayBuffer: capacity 须为正整数');
  const buf = new Array(capacity).fill(null);
  let head = 0, count = 0;
  const copy = (e) => Object.assign({}, e);
  return {
    capacity,
    get size() { return count; },
    get isFull() { return count === capacity; },
    push(exp) {
      if (!exp || exp.s === undefined || exp.a === undefined || exp.r === undefined || exp.s2 === undefined) {
        throw new Error('ReplayBuffer.push: exp 须含 s,a,r,s2（done 可选，默认 false）');
      }
      const e = copy(exp); e.done = !!exp.done;
      buf[head] = e;
      head = (head + 1) % capacity;
      if (count < capacity) count++;
    },
    sample(batch, rng) {
      if (count === 0) throw new Error('ReplayBuffer.sample: 缓冲区为空');
      if (!Number.isInteger(batch) || batch < 1) throw new Error('ReplayBuffer.sample: batch 须为正整数');
      const out = new Array(batch);
      for (let i = 0; i < batch; i++) out[i] = copy(buf[Math.floor(rng() * count)]);
      return out;
    },
    toArray() {
      const out = new Array(count);
      const start = (head - count + capacity) % capacity; // 最旧元素下标
      for (let i = 0; i < count; i++) out[i] = copy(buf[(start + i) % capacity]);
      return out;
    }
  };
}

/* ============================================================================
 * 3. TD 目标 / ε-贪婪 / ε 调度 / Polyak
 * ========================================================================== */

/** TD 目标：y = r + γ·max_a' qNext[a']·(1-done)；done 时精确等于 r（截断自举）。 */
function tdTarget(r, qNext, done, gamma) {
  if (done) return r;
  if (!qNext || qNext.length === 0) throw new Error('tdTarget: qNext 须为非空逐动作 Q 数组');
  let best = qNext[0];
  for (let i = 1; i < qNext.length; i++) if (qNext[i] > best) best = qNext[i];
  return r + gamma * best;
}

/** ε-贪婪：概率 ε 均匀随机，否则 argmax（平局取最小索引）。 */
function epsilonGreedy(qValues, epsilon, rng) {
  const n = qValues.length;
  if (n === 0) throw new Error('epsilonGreedy: 动作集为空');
  if (!(epsilon >= 0 && epsilon <= 1)) throw new Error('epsilonGreedy: ε 须 ∈ [0,1]');
  if (rng() < epsilon) return Math.floor(rng() * n);
  let best = 0;
  for (let a = 1; a < n; a++) if (qValues[a] > qValues[best]) best = a;
  return best;
}

/**
 * ε 调度：eps = max(epsEnd, epsStart·decay^step)（指数，0<decay<1）
 *          或 eps = max(epsEnd, epsStart+(epsEnd-epsStart)·min(step,decaySteps)/decaySteps)（线性）。
 * 两种模式均单调不增且 ≥ epsEnd。
 */
function epsilonSchedule(step, opts) {
  opts = opts || {};
  const epsStart = (opts.epsStart === undefined) ? 1.0 : opts.epsStart;
  const epsEnd = (opts.epsEnd === undefined) ? 0.05 : opts.epsEnd;
  const mode = opts.mode || (opts.decay !== undefined ? 'exp' : 'linear');
  if (!Number.isFinite(step) || step < 0) throw new Error('epsilonSchedule: step 须为非负有限数');
  let eps;
  if (mode === 'exp') {
    const decay = (opts.decay === undefined) ? 0.99 : opts.decay;
    if (!(decay > 0 && decay < 1)) throw new Error('epsilonSchedule: exp 模式要求 0<decay<1');
    eps = epsStart * Math.pow(decay, step);
  } else if (mode === 'linear') {
    const decaySteps = (opts.decaySteps === undefined) ? 1000 : opts.decaySteps;
    if (!(decaySteps > 0)) throw new Error('epsilonSchedule: linear 模式要求 decaySteps>0');
    const frac = Math.min(step, decaySteps) / decaySteps;
    eps = epsStart + (epsEnd - epsStart) * frac;
  } else {
    throw new Error('epsilonSchedule: mode 仅支持 exp/linear');
  }
  return Math.max(epsEnd, eps);
}

/** Polyak 软更新（原地）：target ← tau·online + (1-tau)·target；tau=1 即硬同步。 */
function polyakUpdate(targetParams, qParams, tau) {
  if (targetParams.length !== qParams.length) throw new Error('polyakUpdate: 在线/目标参数长度不一致');
  if (!(tau >= 0 && tau <= 1)) throw new Error('polyakUpdate: tau 须 ∈ [0,1]');
  for (let i = 0; i < targetParams.length; i++) targetParams[i] = tau * qParams[i] + (1 - tau) * targetParams[i];
  return targetParams;
}

/* ============================================================================
 * 4. DQN / Double DQN 更新（复用 optim-forge 的 sgd/adam 作参数步进器）
 * ========================================================================== */

/** 普通 DQN 逐样本 TD 目标：y_i = r_i + γ·max_a q_target(s2_i,a)·(1-done_i)。 */
function dqnTargets(qParams, batch, targetParams, gamma) {
  const out = new Float64Array(batch.length);
  for (let i = 0; i < batch.length; i++) {
    const e = batch[i];
    out[i] = tdTarget(e.r, qNet(targetParams, e.s2), e.done, gamma);
  }
  return out;
}

/**
 * Double DQN 逐样本 TD 目标：a*_i = argmax_a q_online(s2_i,·)（在线网络选动作），
 * y_i = r_i + γ·q_target(s2_i, a*_i)·(1-done_i)（目标网络估值）。
 * 因 q_target(s2,a*) ≤ max_a q_target(s2,a)，逐样本必有 y ≤ 普通 DQN 目标。
 */
function doubleDqnTargets(qParams, batch, targetParams, gamma) {
  const out = new Float64Array(batch.length);
  for (let i = 0; i < batch.length; i++) {
    const e = batch[i];
    const qo = qNet(qParams, e.s2);
    const qt = qNet(targetParams, e.s2);
    let aStar = 0;
    for (let a = 1; a < qo.length; a++) if (qo[a] > qo[aStar]) aStar = a;
    out[i] = e.done ? e.r : e.r + gamma * qt[aStar];
  }
  return out;
}

/**
 * 批量 TD 梯度：L = mean_i (q_i(s,a_i) - y_i)²。
 * dL/dW[a][j] = mean_i 2(q_i(a)-y_i)·s_i[j]；dL/db[a] = mean_i 2(q_i(a)-y_i)。
 * 仅样本自身动作行非零；params[0..1] 元数据梯度恒 0。
 */
function batchTdGradient(qParams, batch, targets) {
  const nS = qParams[0], nA = qParams[1];
  const grad = new Float64Array(qParams.length); // 前两格（元数据）恒 0
  const n = batch.length;
  for (let i = 0; i < n; i++) {
    const e = batch[i];
    const err = qNet(qParams, e.s)[e.a] - targets[i];
    const g = 2 * err / n;
    for (let j = 0; j < nS; j++) grad[2 + e.a * nS + j] += g * e.s[j];
    grad[2 + nA * nS + e.a] += g;
  }
  return grad;
}

/** 内部：以给定步进器执行一步 TD 更新，返回更新前批量 loss 与梯度范数。 */
function applyTdStep(qParams, batch, targets, stepper) {
  let loss = 0;
  for (let i = 0; i < batch.length; i++) {
    const err = qNet(qParams, batch[i].s)[batch[i].a] - targets[i];
    loss += err * err;
  }
  loss /= batch.length;
  const grad = batchTdGradient(qParams, batch, targets);
  stepper.step(qParams, grad);
  let g2 = 0;
  for (let i = 2; i < grad.length; i++) g2 += grad[i] * grad[i]; // 元数据梯度恒 0
  return { loss, gradNorm: Math.sqrt(g2) };
}

/** DQN 更新：对 qParams 做一步梯度下降（复用 ForgeOptim.sgd）。返回 {loss, gradNorm}。 */
function dqnUpdate(qParams, batch, targetParams, gamma, lr) {
  if (!batch.length) throw new Error('dqnUpdate: 批量不能为空');
  return applyTdStep(qParams, batch, dqnTargets(qParams, batch, targetParams, gamma), O().sgd(lr));
}

/** Double DQN 更新（在线选动作 + 目标估值）。返回 {loss, gradNorm}。 */
function doubleDqnUpdate(qParams, batch, targetParams, gamma, lr) {
  if (!batch.length) throw new Error('doubleDqnUpdate: 批量不能为空');
  return applyTdStep(qParams, batch, doubleDqnTargets(qParams, batch, targetParams, gamma), O().sgd(lr));
}

/* ============================================================================
 * 5. 收敛度量与训练循环
 * ========================================================================== */

/** 贪心策略映射：states → argmax_a q(s,a)（平局取最小索引）。 */
function greedyPolicy(qParams, states) {
  return states.map((s) => {
    const q = qNet(qParams, s);
    let best = 0;
    for (let a = 1; a < q.length; a++) if (q[a] > q[best]) best = a;
    return best;
  });
}

/**
 * 贝尔曼残差：max_{s,a} |Q(s,a) - (r + γ·max_a' Q(s',a')·(1-done))|（在线网络自洽度量）。
 * 需要可枚举 env：env.nS / env.nA / env.stateAt(i) / env.step(sVec,a)（纯转移，step 元数 ≥2）。
 * env 提供 isTerminal(i) 时跳过终态：终态 episode 进入即终止、永不被动作触及（无训练信号），
 * 其 Q*≡0 是定义值而非贝尔曼方程约束——残差只在智能体实际决策的非终止态上度量。
 */
function bellmanResidual(env, qParams, gamma) {
  if (!env.nS || !env.nA || typeof env.stateAt !== 'function' || env.step.length < 2) {
    throw new Error('bellmanResidual: 需要可枚举 env（nS/nA/stateAt(i)/step(s,a)）');
  }
  let worst = 0;
  for (let i = 0; i < env.nS; i++) {
    if (typeof env.isTerminal === 'function' && env.isTerminal(i)) continue;
    const sVec = env.stateAt(i);
    const q = qNet(qParams, sVec);
    for (let a = 0; a < env.nA; a++) {
      const out = env.step(sVec, a);
      worst = Math.max(worst, Math.abs(q[a] - tdTarget(out.r, qNet(qParams, out.s2), out.done, gamma)));
    }
  }
  return worst;
}

/** 回退：对不可枚举 env，用缓冲区末态样本估计贝尔曼残差（蒙特卡洛口径）。 */
function bufferBellmanResidual(buf, qParams, gamma, n, rng) {
  if (buf.size === 0) return 0;
  const mb = buf.sample(n, rng);
  let worst = 0;
  for (const e of mb) {
    worst = Math.max(worst, Math.abs(qNet(qParams, e.s)[e.a] - tdTarget(e.r, qNet(qParams, e.s2), e.done, gamma)));
  }
  return worst;
}

/**
 * 完整 DQN 训练循环。
 * env 契约：{nS, nA, reset()->s, step(s,a)->{s2,r,done}}（step 元数 ≥2 为纯转移，可用作残差枚举；
 * 元数 1 的有状态 env 亦支持，此时残差退化为缓冲区 MC 估计）。
 * opts：{gamma, lr, episodes, maxSteps, capacity, batch, seed, optimizer:'sgd'|'adam',
 *        double:bool, tau(>0 每步 Polyak 软更新), targetSync(否则每 K 步硬同步),
 *        eps:{...ε 调度参数...}, initScale}。
 * 返回 {qParams, losses, episodes, bellmanResidual}。
 */
function trainDQN(env, opts) {
  opts = opts || {};
  const gamma = (opts.gamma === undefined) ? 0.9 : opts.gamma;
  const nS = opts.nS || env.nS, nA = opts.nA || env.nA;
  if (!nS || !nA) throw new Error('trainDQN: 需要 env.nS/env.nA 或 opts.nS/opts.nA');
  const episodes = opts.episodes || 500;
  const maxSteps = opts.maxSteps || 64;
  const capacity = opts.capacity || 5000;
  const batchSize = opts.batch || 32;
  const lr = (opts.lr === undefined) ? 0.05 : opts.lr;
  const seed = (opts.seed === undefined) ? 0x9E3779B9 : opts.seed;
  const rng = N().makeRng(seed);
  const pureEnv = env.step.length >= 2; // 纯转移 env：step(s,a)
  const qParams = makeQParams(nS, nA, rng, opts.initScale);
  const targetParams = cloneParams(qParams);
  const buf = new ReplayBuffer(capacity);
  const stepper = (opts.optimizer === 'adam') ? O().adam(lr, 0.9, 0.999, 1e-8) : O().sgd(lr);
  const useDouble = !!opts.double;
  const tau = opts.tau || 0;                 // tau>0：每步 Polyak 软更新
  const targetSync = opts.targetSync || 10;  // 否则每 targetSync 步硬同步目标网络
  const epsOpts = Object.assign(
    { mode: 'linear', epsStart: 1.0, epsEnd: 0.05, decaySteps: Math.max(1, Math.floor(episodes / 2)) },
    opts.eps || {}
  );
  const losses = [];
  let updates = 0;
  for (let ep = 0; ep < episodes; ep++) {
    let s = env.reset();
    const eps = epsilonSchedule(ep, epsOpts);
    for (let t = 0; t < maxSteps; t++) {
      const a = epsilonGreedy(qNet(qParams, s), eps, rng);
      const out = pureEnv ? env.step(s, a) : env.step(a);
      const s2 = out.s2, r = out.r, done = !!out.done;
      buf.push({ s, a, r, s2, done });
      if (buf.size >= batchSize) {
        const mb = buf.sample(batchSize, rng);
        const targets = useDouble
          ? doubleDqnTargets(qParams, mb, targetParams, gamma)
          : dqnTargets(qParams, mb, targetParams, gamma);
        losses.push(applyTdStep(qParams, mb, targets, stepper).loss);
        updates++;
        if (tau > 0) polyakUpdate(targetParams, qParams, tau);
        else if (updates % targetSync === 0) copyParams(targetParams, qParams);
      }
      if (done) break;
      s = s2;
    }
  }
  const residual = (pureEnv && env.nS && env.nA && typeof env.stateAt === 'function')
    ? bellmanResidual(env, qParams, gamma)
    : bufferBellmanResidual(buf, qParams, gamma, Math.min(buf.size, 512), rng);
  return { qParams, losses, episodes, bellmanResidual: residual };
}

/* ============================================================================
 * 6. 链式 MDP 测试环境（one-hot 状态；确定性转移，与 rl-forge 的 P/R 表示同源）
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
    isTerminal: (i) => i === goal, // 终态：episode 进入即终止，不被动作触及
    reset: () => onehot(0),
    step(sVec, a) {
      let s = 0;
      for (let i = 1; i < nS; i++) if (sVec[i] > sVec[s]) s = i; // one-hot → 状态 id
      const s2 = next(s, a);
      return { s2: onehot(s2), r: reward(s, a), done: s2 === goal };
    },
    /** 导出 [P,R]（rl-forge valueIteration 约定），与 env 转移同源保证黄金参照一致。 */
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
 * selfCheck — 8 条不变量
 * ========================================================================== */
function selfCheck(opts) {
  opts = opts || {};
  const { caseResult, buildReport } = V();
  const cases = [];
  const invariants = [];

  // INV-DQN-001 TD 目标闭式：与内联 r+γ·max(qNext)·(1-done) 复算；done 时精确等于 r
  {
    const rng = N().makeRng(0xD001);
    const gamma = 0.9;
    let worst = 0;
    for (let t = 0; t < 20; t++) {
      const nA = 1 + Math.floor(rng() * 5);
      const qNext = new Float64Array(nA);
      for (let a = 0; a < nA; a++) qNext[a] = rng() * 4 - 2;
      const r = rng() * 4 - 2;
      const done = (t % 2 === 0); // 半数 done 样本，检验精确截断
      const got = tdTarget(r, qNext, done, gamma);
      let mx = qNext[0];
      for (let a = 1; a < nA; a++) mx = Math.max(mx, qNext[a]);
      const ref = done ? r : r + gamma * mx;
      worst = Math.max(worst, Math.abs(got - ref));
      if (done && got !== r) worst = Math.max(worst, 1); // done 必须逐位精确
    }
    cases.push(caseResult('INV-DQN-001', 'TD 目标闭式 r+γ·max(qNext)·(1-done)', worst, 0, 1e-12));
  }

  // INV-DQN-002 经验回放 FIFO：capacity=64 推入 100 条 → 保留第 37..100 条；sample 不修改缓冲区
  {
    const capacity = 64, nPush = 100;
    const buf = new ReplayBuffer(capacity);
    for (let i = 0; i < nPush; i++) {
      buf.push({ s: new Float64Array([i]), a: i % 2, r: i, s2: new Float64Array([i + 1]), done: (i % 7 === 0), tag: i });
    }
    let viol = 0;
    if (buf.size !== capacity) viol++;
    if (!buf.isFull) viol++;
    const arr = buf.toArray();
    if (arr.length !== capacity) viol++;
    for (let i = 0; i < capacity; i++) {
      if (arr[i].tag !== nPush - capacity + i || arr[i].r !== nPush - capacity + i) viol++; // 逐位断言保留区间
    }
    const snap = JSON.stringify(arr.map((e) => [e.tag, e.a, e.r, e.done]));
    const samples = buf.sample(100, N().makeRng(0xD002));
    if (samples.length !== 100) viol++;
    if (JSON.stringify(buf.toArray().map((e) => [e.tag, e.a, e.r, e.done])) !== snap) viol++; // 快照对照
    if (buf.size !== capacity || !buf.isFull) viol++;
    cases.push(caseResult('INV-DQN-002', '经验回放 FIFO 逐出最旧 + 采样无副作用', viol, 0, 0));
  }

  // INV-DQN-003 采样分布均匀：20000 次 sample(batch=4)，频率 vs 均匀（MCSE≈1.5e-3，tol 0.03 ≈ 20σ）
  {
    const nItems = 16, calls = 20000, batchSize = 4;
    const buf = new ReplayBuffer(nItems);
    for (let i = 0; i < nItems; i++) buf.push({ s: new Float64Array([i]), a: 0, r: 1, s2: new Float64Array([i]), done: false, tag: i });
    const rng = N().makeRng(0xD003);
    const counts = new Float64Array(nItems);
    for (let c = 0; c < calls; c++) {
      const out = buf.sample(batchSize, rng);
      for (const e of out) counts[e.tag]++;
    }
    const draws = calls * batchSize;
    let maxDev = 0;
    for (let i = 0; i < nItems; i++) maxDev = Math.max(maxDev, Math.abs(counts[i] / draws - 1 / nItems));
    cases.push(caseResult('INV-DQN-003', '回放采样频率 vs 均匀分布（8 万抽）', maxDev, 0, 0.03));
  }

  // INV-DQN-004 ε-贪婪口径：ε=0 恒 argmax；ε=1 均匀；ε=0.5 为 0.5·均匀+0.5·argmax
  {
    const K = 4;
    const rng = N().makeRng(0xD004);
    let viol = 0;
    for (let t = 0; t < 20; t++) { // ε=0：随机 20 组 q，各 200 次恒等检验
      const q = new Float64Array(K);
      let am = 0;
      for (let a = 0; a < K; a++) { q[a] = rng() * 2 - 1; if (q[a] > q[am]) am = a; }
      for (let k = 0; k < 200; k++) if (epsilonGreedy(q, 0, rng) !== am) viol++;
    }
    const q1 = new Float64Array([0.3, -0.2, 0.9, 0.1]); // argmax = 2
    const nDraws = 20000;
    const freq1 = new Float64Array(K);
    for (let k = 0; k < nDraws; k++) freq1[epsilonGreedy(q1, 1, rng)]++;
    let dev1 = 0;
    for (let a = 0; a < K; a++) dev1 = Math.max(dev1, Math.abs(freq1[a] / nDraws - 1 / K));
    const freq5 = new Float64Array(K);
    for (let k = 0; k < nDraws; k++) freq5[epsilonGreedy(q1, 0.5, rng)]++;
    let dev5 = 0;
    for (let a = 0; a < K; a++) {
      const expect = 0.5 * (1 / K) + 0.5 * (a === 2 ? 1 : 0); // 闭式混合分布
      dev5 = Math.max(dev5, Math.abs(freq5[a] / nDraws - expect));
    }
    // 聚合：ε=0 任一违例记 1（必超 tol），其余为频率偏差
    cases.push(caseResult('INV-DQN-004', 'ε-贪婪口径（ε=0 违例/ε=1 均匀/ε=0.5 混合）', Math.max(viol > 0 ? 1 : 0, dev1, dev5), 0, 0.05));
  }

  // INV-DQN-005 ε 调度单调：线性/指数全 step∈[0,2000] 单调不增且 ≥epsEnd
  {
    const modes = [
      { mode: 'linear', epsStart: 1.0, epsEnd: 0.05, decaySteps: 500 },
      { mode: 'exp', epsStart: 1.0, epsEnd: 0.05, decay: 0.99 }
    ];
    let viol = 0;
    for (const m of modes) {
      let prev = epsilonSchedule(0, m);
      if (prev < m.epsEnd) viol++;
      for (let step = 1; step <= 2000; step++) {
        const cur = epsilonSchedule(step, m);
        if (cur > prev + 1e-15) viol++; // 1e-15 吸收 Math.pow 末位抖动
        if (cur < m.epsEnd) viol++;
        prev = cur;
      }
    }
    cases.push(caseResult('INV-DQN-005', 'ε 调度单调不增且 ≥epsEnd（线性/指数）', viol, 0, 0));
  }

  // INV-DQN-006 Polyak 更新：tau=0.01 闭式；tau=1 精确硬同步
  {
    const rng = N().makeRng(0xD006);
    let worst = 0;
    for (let t = 0; t < 20; t++) {
      const n = 4 + Math.floor(rng() * 20);
      const online = new Float64Array(n), target = new Float64Array(n);
      for (let i = 0; i < n; i++) { online[i] = rng() * 2 - 1; target[i] = rng() * 2 - 1; }
      const tp = Float64Array.from(target);
      polyakUpdate(tp, online, 0.01);
      for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(tp[i] - (0.01 * online[i] + 0.99 * target[i])));
      const tp1 = Float64Array.from(target);
      polyakUpdate(tp1, online, 1);
      for (let i = 0; i < n; i++) if (tp1[i] !== online[i]) worst = Math.max(worst, 1); // tau=1 必须逐位精确
    }
    cases.push(caseResult('INV-DQN-006', 'Polyak 软更新闭式与 tau=1 硬同步', worst, 0, 1e-12));
  }

  // INV-DQN-007（灵魂）：DQN 收敛到值迭代——rl-forge 值迭代作独立黄金参照（非自证）。
  // 阈值推导：one-hot 线性 Q 网 ≡ 表格 Q 学习，定点即 VI 的 Q*；SGD 稳态 TD 误差
  // ~ O(lr·σ_TD/√batch)：lr=0.05、batch=32、收敛后单样本 TD 误差 σ≈0.05 ⇒ 残差≈5e-4，
  // 取 1e-2（≈20× 余量，且远小于 Q 值尺度 γ⁴≈0.66，收敛意义不丢失）。
  // 终态（goal）episode 进入即终止、永不被动作触及：其 Q*≡0 是定义值而非贝尔曼约束，
  // 且线性近似的偏置耦合会使其未训练输出非零（结构近似而非收敛失败）——故贝尔曼残差与
  // 策略一致性均在非终止态全集（4 态 × 2 动作 = 8 对）上度量。
  {
    const env = makeChainEnv(5);
    const gamma = 0.9;
    const spec = env.spec();
    const vi = R().valueIteration(env.nS, env.nA, spec.P, spec.R, gamma, 1000); // 独立黄金参照
    const trained = trainDQN(env, {
      gamma, episodes: 800, maxSteps: 16, capacity: 4000, batch: 32, lr: 0.05,
      targetSync: 10, initScale: 0, seed: 0xD007,
      eps: { mode: 'linear', epsStart: 1.0, epsEnd: 0.05, decaySteps: 400 }
    });
    const states = [];
    for (let i = 0; i < env.nS - 1; i++) states.push(env.stateAt(i)); // 非终止态
    const greedy = greedyPolicy(trained.qParams, states);
    let mismatch = 0;
    for (let i = 0; i < states.length; i++) if (greedy[i] !== vi.policy[i]) mismatch++;
    cases.push(caseResult('INV-DQN-007', 'DQN 贪心策略 vs 值迭代最优策略 + 贝尔曼残差', Math.max(mismatch, trained.bellmanResidual), 0, 1e-2));
  }

  // INV-DQN-008 Double DQN 低估偏差有界：诱惑构造 MDP——每状态在线 argmax 动作 p_k 在目标网络中
  // 被低估(0.3)，另一动作 q_k 被高估(0.9)：普通 DQN 的 max 选中高估动作，Double DQN 不选。
  // 由 max 占优性 q_t(s,a*)≤max q_t(s,·)，逐样本 y_double ≤ y_dqn 恒成立；构造保证 240 样本全部
  // 严格诱惑（非空转检验：strict>0 否则记违例 1）。
  {
    const nS = 6, nA = 3;
    const rng = N().makeRng(0xD008);
    const online = new Float64Array(2 + nA * (nS + 1));
    const target = new Float64Array(2 + nA * (nS + 1));
    online[0] = nS; online[1] = nA; target[0] = nS; target[1] = nA;
    const pAt = new Array(nS), qAt = new Array(nS);
    for (let k = 0; k < nS; k++) {
      pAt[k] = Math.floor(rng() * nA);
      do { qAt[k] = Math.floor(rng() * nA); } while (qAt[k] === pAt[k]);
      for (let a = 0; a < nA; a++) {
        online[2 + a * nS + k] = (a === pAt[k]) ? 1.0 : 0.0;
        target[2 + a * nS + k] = (a === pAt[k]) ? 0.3 : ((a === qAt[k]) ? 0.9 : 0.0);
      }
    }
    const onehot = (i) => { const v = new Float64Array(nS); v[i] = 1; return v; };
    let viol = 0, strict = 0;
    for (let t = 0; t < 20; t++) {
      const batch = [];
      for (let i = 0; i < 12; i++) {
        batch.push({
          s: onehot(Math.floor(rng() * nS)), a: Math.floor(rng() * nA),
          s2: onehot(Math.floor(rng() * nS)), r: rng() * 2, done: rng() < 0.2
        });
      }
      const yD = dqnTargets(online, batch, target, 0.9);
      const y2 = doubleDqnTargets(online, batch, target, 0.9);
      for (let i = 0; i < batch.length; i++) {
        if (y2[i] > yD[i] + 1e-12) viol++;
        if (yD[i] - y2[i] > 1e-12) strict++;
      }
    }
    cases.push(caseResult('INV-DQN-008', 'Double DQN TD 目标 ≤ 普通 DQN（诱惑构造）', viol + (strict > 0 ? 0 : 1), 0, 0));
  }

  const started = Date.now();
  for (const id of ['INV-DQN-001', 'INV-DQN-002', 'INV-DQN-003', 'INV-DQN-004', 'INV-DQN-005', 'INV-DQN-006', 'INV-DQN-007', 'INV-DQN-008']) {
    const c = cases.find((x) => x.id === id);
    invariants.push({ id, expr: id, tol: c.tol, worst: c.abs_err, status: c.status, note: '' });
  }
  return buildReport({
    module: 'dqn-forge', version: '1.0.0', layer: 'L6', cases, invariants,
    seed: '0x9E3779B9', started,
    tol: { 'INV-DQN-001': 1e-12, 'INV-DQN-002': 0, 'INV-DQN-003': 0.03, 'INV-DQN-004': 0.05, 'INV-DQN-005': 0, 'INV-DQN-006': 1e-12, 'INV-DQN-007': 1e-2, 'INV-DQN-008': 0 }
  });
}

const API = {
  name: 'dqn-forge', version: '1.0.0', layer: 'L6',
  qNet, makeQParams, cloneParams, copyParams,
  ReplayBuffer, tdTarget, dqnTargets, doubleDqnTargets,
  epsilonGreedy, epsilonSchedule, polyakUpdate,
  dqnUpdate, doubleDqnUpdate, greedyPolicy, bellmanResidual, trainDQN,
  makeChainEnv, selfCheck
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeDQN = API;

if (typeof process !== 'undefined' && process.argv[1] && /dqn-forge\.js$/.test(process.argv[1]) && (process.argv.includes('--self-check') || process.argv.includes('--json'))) {
  const r = selfCheck({ quick: process.argv.includes('--quick') });
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : V().formatTableSafe(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
