/**
 * @forge-module scheduler-forge
 * @layer L3
 * @version 1.0.0
 * @brief     学习率调度器内核：常数、线性 warmup、线性衰减、余弦退火、余弦重启（SGDR）、阶跃衰减、
 *            指数衰减、多项式衰减、oneCycle 三段式。每个工厂返回「给定步数 t 返回 lr」的纯函数闭包，
 *            t 为从 0 开始的整数步。端点显式夹取（t=total → 末值、周期起点 → lrMax）保证位精确，
 *            warmup 从 lrMax/warmup 非零起步规避零学习率死区；全部参数构造期校验，非法输入立即抛错。
 * @reuse     SGDR: Stochastic Gradient Descent with Warm Restarts (Loshchilov & Hutter 2016)|https://arxiv.org/abs/1608.03983|MIT|A|余弦重启周期相位原式
 * @reuse     Bag of Tricks for Image Classification (He et al. 2018)|https://arxiv.org/abs/1812.01187|MIT|A|线性 warmup + 线性/余弦衰减口径
 * @reuse     A Disciplined Approach to Neural Network Hyper-Parameters (Smith 2018)|https://arxiv.org/abs/1708.07120|MIT|A|oneCycle 三段结构（warmup/余弦/线性收官）
 * @reuse     PyTorch lr_scheduler|https://github.com/pytorch/pytorch|BSD-3|C|StepLR/ExponentialLR/PolynomialLR 阶跃与指数语义
 * @beyond    纯 JS 单文件零依赖双端同构；warmup 自 lrMax/warmup 起步（t=0 非零）规避零学习率死区；
 *            余弦/oneCycle 端点与周期起点以显式分支夹取，t=warmup 恰为 lrMax、t=total 恰为末值（位精确）；
 *            8 条不变量全部以内联闭式独立复算（非自证），每条含「真实 bug 必失败」的反空洞断言。
 * @invariant INV-SCH-001|线性 warmup 闭式：t≤warmup 逐点等于 lrMax·(t+1)/warmup（t=0 恰为 lrMax/warmup 非零）|0|逐点闭式复算
 * @invariant INV-SCH-002|余弦端点：t=warmup 恰为 lrMax、t=total-1 与解析闭式误差为 0、t=total 按夹取到末值 lrMin|0|端点解析值代入
 * @invariant INV-SCH-003|余弦单调不增：warmup 之后全差分 ≤0（容差 -1e-9）且存在实质下降|1e-9|差分扫描
 * @invariant INV-SCH-004|阶跃边界：恰在 dropEvery 整数倍边界切换，与 lr·dropFactor^floor(t/dropEvery) 逐点相等|0|闭式复算
 * @invariant INV-SCH-005|线性衰减等差：相邻步差恒定且等于 (lrMin-lrMax)/(total-warmup)，终点为 lrMin|1e-12|差分闭式
 * @invariant INV-SCH-006|余弦重启周期相位：任意 t 等于「第 k 个周期相位 φ」闭式复算（周期数/相位/端点三重校验）|1e-12|闭式复算
 * @invariant INV-SCH-007|oneCycle 三段：warmup 段线性等差、峰值为 lrMax 且出现在 pctStart 边界、余弦段单调、终点=lrMax/(div·100)|1e-12|分段闭式
 * @invariant INV-SCH-008|非法输入抛错：lrMax≤0、warmup<0、total≤warmup、dropEvery≤0、power<0、t 非有限数/负数 —— 每个至少一个探针且必须真抛|0|异常捕获
 * @seed      0x5C0D01
 */
'use strict';

/* ============================================================================
 * 0. 惰性双端依赖：Node 端 require，浏览器端回退 globalThis（file:// 经典 script src）。
 *    模块加载期不触碰任何依赖，保证双端同构与零循环导入。
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

/* ============================================================================
 * 1. 参数校验（构造期 fail-fast；错误信息带模块前缀便于定位）
 * ========================================================================== */
const PREFIX = 'scheduler-forge';

/** 校验为有限数（number 且非 NaN/±Infinity），否则抛 TypeError。 */
function assertFinite(value, name) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(PREFIX + ': ' + name + ' 须为有限数，收到 ' + (typeof value === 'number' ? String(value) : typeof value));
  }
}

/** 校验为有限正数（> 0），否则抛 RangeError。 */
function assertPositive(value, name) {
  assertFinite(value, name);
  if (value <= 0) throw new RangeError(PREFIX + ': ' + name + ' 须 > 0，收到 ' + value);
}

/** 校验为有限非负数（>= 0），否则抛 RangeError。 */
function assertNonNegative(value, name) {
  assertFinite(value, name);
  if (value < 0) throw new RangeError(PREFIX + ': ' + name + ' 须 >= 0，收到 ' + value);
}

/** 步 t 校验（调度器返回函数内）：有限且非负（t 从 0 开始计）。 */
function assertStep(t) {
  assertFinite(t, 't');
  if (t < 0) throw new RangeError(PREFIX + ': t 须 >= 0（步从 0 开始），收到 ' + t);
}

/** scheduleAt 的严格步校验：在 assertStep 基础上还要求整数步。 */
function assertStepInt(t) {
  assertStep(t);
  if (!Number.isInteger(t)) throw new RangeError(PREFIX + ': t 须为整数步，收到 ' + t);
}

/** 学习率区间校验：lrMax > 0 且 lrMin ∈ [0, lrMax]。 */
function assertRange(lrMax, lrMin) {
  assertPositive(lrMax, 'lrMax');
  assertNonNegative(lrMin, 'lrMin');
  if (lrMin > lrMax) throw new RangeError(PREFIX + ': lrMin(' + lrMin + ') 不得 > lrMax(' + lrMax + ')');
}

/** warmup/total 口径：warmup >= 0 且 total > warmup（保证衰减段长度严格为正）。 */
function assertHorizon(warmup, total) {
  assertNonNegative(warmup, 'warmup');
  assertFinite(total, 'total');
  if (!(total > warmup)) throw new RangeError(PREFIX + ': total(' + total + ') 须 > warmup(' + warmup + ')');
}

/* ============================================================================
 * 2. 学习率调度器（工厂返回 lrAt(t)；t 为非负有限数；越界端点夹取到边界值）
 * ========================================================================== */

/** 常数调度：任意步返回同一 lr。 */
function constantSchedule(lr) {
  assertPositive(lr, 'lr');
  return function constantAt(t) {
    assertStep(t);
    return lr;
  };
}

/**
 * 线性 warmup：t < warmup 时 lr = lrMax·(t+1)/warmup（t=0 为 lrMax/warmup，非零起步，
 * 规避零学习率死区）；t >= warmup 时恒为 lrMax。warmup=0 时无斜坡段，恒为 lrMax。
 */
function linearWarmup(lrMax, warmup) {
  assertPositive(lrMax, 'lrMax');
  assertNonNegative(warmup, 'warmup');
  return function linearWarmupAt(t) {
    assertStep(t);
    if (t < warmup) return (lrMax * (t + 1)) / warmup;
    return lrMax;
  };
}

/**
 * 线性衰减：warmup 段同 linearWarmup；之后
 * lr = lrMax + (lrMin-lrMax)·(t-warmup)/(total-warmup)，相邻步差恒为
 * (lrMin-lrMax)/(total-warmup)；t >= total 夹取为 lrMin（位精确）。
 */
function linearDecay(lrMax, lrMin, warmup, total) {
  assertRange(lrMax, lrMin);
  assertHorizon(warmup, total);
  const span = total - warmup;
  const slope = (lrMin - lrMax) / span;
  return function linearDecayAt(t) {
    assertStep(t);
    if (t < warmup) return (lrMax * (t + 1)) / warmup;
    if (t >= total) return lrMin;
    return lrMax + slope * (t - warmup);
  };
}

/**
 * 余弦退火：warmup 段同 linearWarmup；之后
 * lr = lrMin + 0.5·(lrMax-lrMin)·(1+cos(π·(t-warmup)/(total-warmup)))。
 * t=warmup 显式返回 lrMax（cos(0)=1 的位精确夹取），t >= total 夹取为 lrMin。
 */
function cosineSchedule(lrMax, lrMin, warmup, total) {
  assertRange(lrMax, lrMin);
  assertHorizon(warmup, total);
  const span = total - warmup;
  return function cosineAt(t) {
    assertStep(t);
    if (t < warmup) return (lrMax * (t + 1)) / warmup;
    if (t === warmup) return lrMax; // 端点闭式：cos(0)=1，显式夹取保证位精确
    if (t >= total) return lrMin;   // 夹取到末值
    return lrMin + 0.5 * (lrMax - lrMin) * (1 + Math.cos(Math.PI * (t - warmup) / span));
  };
}

/**
 * 余弦重启（SGDR）：warmup 段同 linearWarmup；之后以 warmup+period·k 为周期，
 * 第 k 周期相位 φ = (t-warmup) - k·period（k = floor((t-warmup)/period)），
 * lr = lrMin + 0.5·(lrMax-lrMin)·(1+cos(π·φ/period))。
 * 周期起点（φ=0）显式返回 lrMax（位精确）；无上界夹取，周期无限重启。
 */
function cosineWithRestarts(lrMax, lrMin, warmup, period) {
  assertRange(lrMax, lrMin);
  assertNonNegative(warmup, 'warmup');
  assertPositive(period, 'period');
  return function cosineRestartAt(t) {
    assertStep(t);
    if (t < warmup) return (lrMax * (t + 1)) / warmup;
    const phase = (t - warmup) % period;
    if (phase === 0) return lrMax; // 周期起点闭式
    return lrMin + 0.5 * (lrMax - lrMin) * (1 + Math.cos(Math.PI * phase / period));
  };
}

/**
 * 阶跃衰减：每过 dropEvery 步乘一次 dropFactor，即
 * lr·dropFactor^floor(t/dropEvery)（t 从 0 计：t < dropEvery 不降，
 * t = dropEvery 才第一次降，恰在整数倍边界切换）。
 */
function stepDecay(lr, dropFactor, dropEvery) {
  assertPositive(lr, 'lr');
  assertPositive(dropFactor, 'dropFactor');
  assertPositive(dropEvery, 'dropEvery');
  return function stepDecayAt(t) {
    assertStep(t);
    return lr * Math.pow(dropFactor, Math.floor(t / dropEvery));
  };
}

/** 指数衰减：lr·decay^t（decay ∈ (0,1) 为常规衰减；>1 为增长，由调用方负责）。 */
function exponentialDecay(lr, decay) {
  assertPositive(lr, 'lr');
  assertPositive(decay, 'decay');
  return function exponentialDecayAt(t) {
    assertStep(t);
    return lr * Math.pow(decay, t);
  };
}

/**
 * 多项式衰减：lr = lrMin + (lrMax-lrMin)·(1-t/total)^power；
 * t >= total 时夹取（(1-tc/total)^power = 0，power>0 时恰为 lrMin；
 * power=0 时为常数 lrMax，属定义行为）。
 */
function polynomialDecay(lrMax, lrMin, total, power) {
  assertRange(lrMax, lrMin);
  assertPositive(total, 'total');
  assertNonNegative(power, 'power');
  return function polynomialDecayAt(t) {
    assertStep(t);
    const tc = Math.min(t, total);
    return lrMin + (lrMax - lrMin) * Math.pow(1 - tc / total, power);
  };
}

/**
 * oneCycle 三段（Smith 2018 超收敛策略）：
 *   [0, pctStart·total)             线性 warmup：lrMax/div → lrMax
 *   [pctStart·total, pctEnd·total)  余弦降到 lrMax/div
 *   [pctEnd·total, total)           线性降到 lrMax/(div·100)
 * t >= total 夹取为 lrMax/(div·100)（位精确）。
 * pctStart=0 时 warmup 段为空（余弦自 lrMax 起步）；pctEnd=1 时末段为空。
 */
function oneCycle(lrMax, div, pctStart, pctEnd, total) {
  assertPositive(lrMax, 'lrMax');
  assertPositive(div, 'div');
  assertFinite(pctStart, 'pctStart');
  assertFinite(pctEnd, 'pctEnd');
  if (!(pctStart >= 0 && pctStart < pctEnd && pctEnd <= 1)) {
    throw new RangeError(PREFIX + ': oneCycle 要求 0 <= pctStart < pctEnd <= 1，收到 pctStart=' + pctStart + '、pctEnd=' + pctEnd);
  }
  assertPositive(total, 'total');
  const warmupEnd = pctStart * total; // 第一/二段边界
  const annEnd = pctEnd * total;      // 第二/三段边界
  const lo = lrMax / div;             // warmup 起点 / 余弦段终点
  const finalLr = lrMax / (div * 100); // 全程终点
  return function oneCycleAt(t) {
    assertStep(t);
    if (t >= total) return finalLr; // 夹取到末值
    if (t < warmupEnd) {
      return warmupEnd > 0 ? lo + (lrMax - lo) * (t / warmupEnd) : lrMax;
    }
    if (t < annEnd) {
      return lo + 0.5 * (lrMax - lo) * (1 + Math.cos(Math.PI * (t - warmupEnd) / (annEnd - warmupEnd)));
    }
    return lo + (finalLr - lo) * ((t - annEnd) / (total - annEnd));
  };
}

/**
 * scheduleAt(fn, t)：fn(t) 的薄封装（含类型检查）。
 * fn 须为函数，t 须为非负整数步；fn 返回值须为有限数，否则抛 RangeError。
 */
function scheduleAt(fn, t) {
  if (typeof fn !== 'function') {
    throw new TypeError(PREFIX + ': scheduleAt 需要一个 lr(t) 函数，收到 ' + typeof fn);
  }
  assertStepInt(t);
  const lr = fn(t);
  if (typeof lr !== 'number' || !Number.isFinite(lr)) {
    throw new RangeError(PREFIX + ': 调度器返回了非有限学习率 ' + String(lr));
  }
  return lr;
}

/* ============================================================================
 * 3. selfCheck — 8 条不变量（内联闭式独立复算，非自证；含反空洞断言）
 * ========================================================================== */
function selfCheck(opts) {
  opts = opts || {};
  const { caseResult, buildReport } = V();
  const cases = [];
  const invariants = [];

  // INV-SCH-001 线性 warmup 闭式：t∈[0,warmup] 逐点等于 lrMax·(t+1)/warmup。
  // 反空洞：t=0 必须非零且恰为 lrMax/warmup——若 warmup 段误写为 t/warmup，t=0 处立即为 0 失败。
  {
    const lrMax = 0.1, warmup = 10;
    const fn = linearWarmup(lrMax, warmup);
    let viol = 0;
    if (!(fn(0) > 0) || fn(0) !== lrMax / warmup) viol++;
    for (let t = 0; t <= warmup + 5; t++) {
      const expected = t < warmup ? (lrMax * (t + 1)) / warmup : lrMax;
      if (fn(t) !== expected) viol++; // 逐点闭式复算（tol 0，位精确）
    }
    // 附带冒烟：constantSchedule 任意步恒定
    const cfn = constantSchedule(0.05);
    for (const t of [0, 1, 37, 1000]) if (cfn(t) !== 0.05) viol++;
    cases.push(caseResult('INV-SCH-001', '线性 warmup 闭式逐点相等（t=0 非零）+ 常数调度', viol, 0, 0,
      't∈[0,warmup+5] 逐点；反空洞：t=0 非零断言'));
  }

  // INV-SCH-002 余弦端点：t=warmup 恰为 lrMax（位精确）；t=total-1 与解析闭式误差为 0；
  // t=total（及更远）夹取为 lrMin。反空洞：过 warmup 必须严格下折、全程必须实质下降。
  {
    const lrMax = 0.1, lrMin = 0.001, warmup = 5, total = 105;
    const fn = cosineSchedule(lrMax, lrMin, warmup, total);
    let viol = 0;
    if (fn(warmup) !== lrMax) viol++;
    const t1 = total - 1;
    const closed = lrMin + 0.5 * (lrMax - lrMin) * (1 + Math.cos(Math.PI * (t1 - warmup) / (total - warmup)));
    if (fn(t1) !== closed) viol++;
    if (fn(total) !== lrMin) viol++;
    if (fn(total + 50) !== lrMin) viol++;
    if (!(fn(warmup + 1) < lrMax)) viol++;
    if (!(fn(total - 1) < fn(warmup))) viol++;
    cases.push(caseResult('INV-SCH-002', '余弦端点：t=warmup=lrMax、t=total-1 闭式误差 0、t=total=lrMin', viol, 0, 0,
      '端点解析值代入；反空洞：严格下折 + 实质下降'));
  }

  // INV-SCH-003 余弦单调不增：warmup 之后（含末步夹取）全差分 ≤ 1e-9 松弛。
  // 反空洞：必须存在实质下降（minDiff < -1e-6 且总降幅 ≥ 半程），否则恒平序列会空洞通过。
  {
    const lrMax = 0.1, lrMin = 0.001, warmup = 5, total = 105;
    const fn = cosineSchedule(lrMax, lrMin, warmup, total);
    let maxDiff = -Infinity, minDiff = Infinity;
    for (let t = warmup; t < total; t++) {
      const d = fn(t + 1) - fn(t);
      if (d > maxDiff) maxDiff = d;
      if (d < minDiff) minDiff = d;
    }
    const drop = fn(warmup) - fn(total - 1);
    const substantial = drop >= 0.5 * (lrMax - lrMin);
    const excess = Math.max(0, maxDiff - 1e-9) + ((minDiff < -1e-6 && substantial) ? 0 : 1);
    cases.push(caseResult('INV-SCH-003', '余弦 warmup 后单调不增（差分扫描）', excess, 0, -1e-9,
      'maxDiff=' + maxDiff.toExponential(3) + ' minDiff=' + minDiff.toExponential(3) + ' drop=' + drop.toExponential(3)));
  }

  // INV-SCH-004 阶跃边界：t∈[0,120] 逐点等于 lr·dropFactor^floor(t/dropEvery)（位精确）；
  // 恰在 dropEvery 整数倍边界切换。反空洞：边界处必须切换、边界前必须未降。
  {
    const lr = 0.1, dropFactor = 0.5, dropEvery = 7;
    const fn = stepDecay(lr, dropFactor, dropEvery);
    let viol = 0;
    for (let t = 0; t <= 120; t++) {
      const expected = lr * Math.pow(dropFactor, Math.floor(t / dropEvery));
      if (fn(t) !== expected) viol++;
    }
    if (fn(dropEvery - 1) !== lr) viol++;
    if (fn(dropEvery) !== lr * dropFactor) viol++;
    if (fn(dropEvery) === fn(dropEvery - 1)) viol++; // 反空洞：必须切换
    if (fn(2 * dropEvery) !== lr * dropFactor * dropFactor) viol++;
    if (fn(2 * dropEvery - 1) !== lr * dropFactor) viol++;
    // 附带冒烟：exponentialDecay 闭式 lr·decay^t
    const efn = exponentialDecay(0.1, 0.95);
    for (const t of [0, 1, 5, 50, 200]) if (efn(t) !== 0.1 * Math.pow(0.95, t)) viol++;
    cases.push(caseResult('INV-SCH-004', '阶跃恰在 dropEvery 整数倍边界切换（闭式逐点）', viol, 0, 0,
      't∈[0,120] 逐点；反空洞：边界处必须切换'));
  }

  // INV-SCH-005 线性衰减等差：衰减段相邻差恒定等于 (lrMin-lrMax)/(total-warmup)（≤1e-12），
  // 起点为 lrMax、终点（t=total）为 lrMin（位精确）。反空洞：衰减段必须严格递减。
  // 附带冒烟：polynomialDecay 端点/单调/闭式抽点。
  {
    const lrMax = 0.1, lrMin = 0.001, warmup = 5, total = 105;
    const fn = linearDecay(lrMax, lrMin, warmup, total);
    const slope = (lrMin - lrMax) / (total - warmup);
    let worst = 0, viol = 0;
    for (let t = warmup; t < total; t++) {
      const d = fn(t + 1) - fn(t);
      worst = Math.max(worst, Math.abs(d - slope));
      if (d >= 0) viol++; // 反空洞：衰减段严格递减
    }
    if (fn(warmup) !== lrMax) viol++;
    if (fn(total) !== lrMin) viol++;
    const pfn = polynomialDecay(0.1, 0.001, 100, 2);
    if (Math.abs(pfn(0) - 0.1) > 1e-12) viol++;
    if (pfn(100) !== 0.001) viol++;
    if (Math.abs(pfn(50) - (0.001 + (0.1 - 0.001) * Math.pow(1 - 50 / 100, 2))) > 1e-12) viol++;
    for (let t = 0; t < 100; t++) if (pfn(t + 1) > pfn(t) + 1e-12) viol++;
    cases.push(caseResult('INV-SCH-005', '线性衰减等差（差分=闭式斜率）+ 终点 lrMin', worst + viol, 0, 1e-12,
      'worstDiff=' + worst.toExponential(3) + '；附带多项式冒烟；反空洞：严格递减'));
  }

  // INV-SCH-006 余弦重启周期相位：随机 200 个 t，与「第 k 周期相位 φ」闭式复算（≤1e-12）。
  // 三重校验：周期起点（每周 k 起点恰为 lrMax）、相位推进（闭式逐点）、端点回归（重启后回到起点值）。
  // 反空洞：周期末值必须严格介于 (lrMin, lrMax)——若相位算错（如漏掉取模）会立即偏离。
  {
    const lrMax = 0.1, lrMin = 0.001, warmup = 4, period = 20;
    const fn = cosineWithRestarts(lrMax, lrMin, warmup, period);
    const rng = N().makeRng(0x5C06);
    let worst = 0, viol = 0;
    for (let i = 0; i < 200; i++) {
      const t = warmup + Math.floor(rng() * (5 * period));
      const k = Math.floor((t - warmup) / period);
      const phase = (t - warmup) - k * period;
      const expected = phase === 0
        ? lrMax
        : lrMin + 0.5 * (lrMax - lrMin) * (1 + Math.cos(Math.PI * phase / period));
      worst = Math.max(worst, Math.abs(fn(t) - expected));
    }
    for (let k = 0; k <= 4; k++) {
      if (fn(warmup + k * period) !== lrMax) viol++; // 三重校验之一：周期起点
    }
    if (fn(warmup + period) !== fn(warmup)) viol++;   // 三重校验之二：重启回归
    if (!(fn(warmup + period - 1) < lrMax && fn(warmup + period - 1) > lrMin)) viol++; // 三重校验之三：周期末介于开区间
    for (let t = 0; t < warmup; t++) {
      if (fn(t) !== (lrMax * (t + 1)) / warmup) viol++; // warmup 段同 linearWarmup
    }
    cases.push(caseResult('INV-SCH-006', '余弦重启周期相位闭式（周期数/相位/端点三重校验）', worst + viol, 0, 1e-12,
      '随机 200 点（makeRng 0x5C06）worst=' + worst.toExponential(3) + '；反空洞：周期末严格介于 (lrMin,lrMax)'));
  }

  // INV-SCH-007 oneCycle 三段：warmup 段线性等差（斜率 (lrMax-lo)/warmupEnd）、峰值为 lrMax
  // 且出现在 pctStart 边界、余弦段单调不增、末段线性等差、终点=lrMax/(div·100)（位精确）。
  // 反空洞：峰值定位断言（边界前未到峰、过峰即降）+ 触底前断言（余弦段末 > lo、末段末 > finalLr）。
  {
    const lrMax = 0.1, div = 10, pctStart = 0.3, pctEnd = 0.8, total = 100;
    const fn = oneCycle(lrMax, div, pctStart, pctEnd, total);
    const warmupEnd = pctStart * total;
    const annEnd = pctEnd * total;
    const lo = lrMax / div;
    const finalLr = lrMax / (div * 100);
    let worst = 0, viol = 0;
    // 第一段：线性等差，起点 lo
    const wSlope = (lrMax - lo) / warmupEnd;
    if (fn(0) !== lo) viol++;
    for (let t = 0; t < warmupEnd; t++) {
      worst = Math.max(worst, Math.abs(fn(t + 1) - fn(t) - wSlope));
    }
    // 第二段起点即峰值 lrMax（出现在 pctStart 边界）
    worst = Math.max(worst, Math.abs(fn(warmupEnd) - lrMax));
    if (!(fn(warmupEnd - 1) < lrMax)) viol++; // 反空洞：边界前未到峰
    if (!(fn(warmupEnd + 1) < lrMax)) viol++; // 反空洞：过峰即降
    // 第二段：余弦单调不增，终点 lo
    let maxCosDiff = -Infinity;
    for (let t = warmupEnd; t < annEnd; t++) {
      const d = fn(t + 1) - fn(t);
      if (d > maxCosDiff) maxCosDiff = d;
    }
    worst = Math.max(worst, Math.abs(fn(annEnd) - lo));
    if (maxCosDiff > 1e-12) viol++;
    if (!(fn(annEnd - 1) > lo)) viol++; // 反空洞：余弦段终点前未触 lo
    // 第三段：线性等差，终点 finalLr
    const fSlope = (finalLr - lo) / (total - annEnd);
    for (let t = annEnd; t < total; t++) {
      worst = Math.max(worst, Math.abs(fn(t + 1) - fn(t) - fSlope));
    }
    if (fn(total) !== finalLr) viol++;
    if (fn(total + 10) !== finalLr) viol++;
    if (!(fn(total - 1) > finalLr)) viol++; // 反空洞：末段递减未触底
    cases.push(caseResult('INV-SCH-007', 'oneCycle 三段闭式（等差/峰值/余弦单调/终点）', worst + viol, 0, 1e-12,
      'worst=' + worst.toExponential(3) + ' maxCosDiff=' + maxCosDiff.toExponential(3) + '；反空洞：峰值定位 + 触底前断言'));
  }

  // INV-SCH-008 非法输入抛错：每类至少一个探针且必须真抛；合法调用一律不抛（反空洞）。
  {
    const goodFn = linearWarmup(0.1, 10);
    const probes = [
      // lrMax <= 0
      () => linearWarmup(0, 10),
      () => linearWarmup(-0.1, 10),
      () => cosineSchedule(0, 0.001, 5, 50),
      () => oneCycle(0, 10, 0.3, 0.8, 100),
      // warmup < 0
      () => linearWarmup(0.1, -1),
      () => linearDecay(0.1, 0.001, -2, 50),
      () => cosineSchedule(0.1, 0.001, -1, 50),
      // total <= warmup
      () => linearDecay(0.1, 0.001, 5, 5),
      () => linearDecay(0.1, 0.001, 10, 5),
      () => cosineSchedule(0.1, 0.001, 5, 5),
      // dropEvery <= 0
      () => stepDecay(0.1, 0.5, 0),
      () => stepDecay(0.1, 0.5, -3),
      // power < 0
      () => polynomialDecay(0.1, 0.001, 100, -1),
      // 其余结构性非法（周期/总量/因子/百分比/类型）
      () => cosineWithRestarts(0.1, 0.001, 4, 0),
      () => polynomialDecay(0.1, 0.001, 0, 2),
      () => exponentialDecay(0.1, -0.5),
      () => oneCycle(0.1, 0, 0.3, 0.8, 100),
      () => oneCycle(0.1, 10, 0.8, 0.3, 100),
      () => linearWarmup('x', 10),
      // t 非有限数 / 负数 / 非整数步 / 非函数
      () => scheduleAt(goodFn, NaN),
      () => scheduleAt(goodFn, Infinity),
      () => scheduleAt(goodFn, -Infinity),
      () => scheduleAt(goodFn, -1),
      () => scheduleAt(goodFn, 1.5),
      () => scheduleAt(goodFn, '3'),
      () => scheduleAt(42, 1),
      () => goodFn(NaN),
      () => goodFn(-1)
    ];
    let viol = 0;
    for (const p of probes) {
      try { p(); viol++; } catch (e) { /* 必须真抛：未抛才记违例 */ }
    }
    // 反空洞：合法调用（含退化口径）一律不得抛
    try {
      scheduleAt(goodFn, 0); scheduleAt(goodFn, 10); scheduleAt(goodFn, 25);
      constantSchedule(0.01)(0);
      linearDecay(0.1, 0.001, 5, 105)(0); linearDecay(0.1, 0.001, 5, 105)(105);
      cosineSchedule(0.1, 0.001, 5, 105)(5); cosineSchedule(0.1, 0.001, 5, 105)(104);
      cosineWithRestarts(0.1, 0.001, 4, 20)(50);
      stepDecay(0.1, 0.5, 7)(21);
      exponentialDecay(0.1, 0.9)(5);
      polynomialDecay(0.1, 0.001, 100, 2)(100);
      oneCycle(0.1, 10, 0.3, 0.8, 100)(0);
      oneCycle(0.1, 10, 0.3, 0.8, 100)(100);
      oneCycle(0.1, 10, 0, 1, 100)(50); // 退化口径：空 warmup 段 + 空末段
    } catch (e) { viol++; }
    cases.push(caseResult('INV-SCH-008', '非法输入全部真抛 + 合法调用不抛', viol, 0, 0,
      probes.length + ' 个非法探针 + 合法调用反空洞'));
  }

  const started = Date.now();
  const tolMap = {
    'INV-SCH-001': 0, 'INV-SCH-002': 0, 'INV-SCH-003': -1e-9, 'INV-SCH-004': 0,
    'INV-SCH-005': 1e-12, 'INV-SCH-006': 1e-12, 'INV-SCH-007': 1e-12, 'INV-SCH-008': 0
  };
  const exprMap = {
    'INV-SCH-001': '线性 warmup 闭式：t≤warmup 逐点等于 lrMax·(t+1)/warmup（t=0 非零）',
    'INV-SCH-002': '余弦端点：t=warmup 恰为 lrMax、t=total-1 闭式误差 0、t=total 夹取末值',
    'INV-SCH-003': '余弦单调不增（warmup 之后全差分 ≤0，容差 -1e-9）',
    'INV-SCH-004': '阶跃恰在 dropEvery 整数倍边界切换，逐点等于 lr·dropFactor^floor(t/dropEvery)',
    'INV-SCH-005': '线性衰减等差：相邻差恒定等于 (lrMin-lrMax)/(total-warmup)，终点 lrMin',
    'INV-SCH-006': '余弦重启周期相位闭式（周期数/相位/端点三重校验）',
    'INV-SCH-007': 'oneCycle 三段：warmup 等差、峰值 lrMax@pctStart 边界、余弦段单调、终点 lrMax/(div·100)',
    'INV-SCH-008': '非法输入抛错：lrMax≤0/warmup<0/total≤warmup/dropEvery≤0/power<0/t 非有限或负'
  };
  for (const id of ['INV-SCH-001', 'INV-SCH-002', 'INV-SCH-003', 'INV-SCH-004', 'INV-SCH-005', 'INV-SCH-006', 'INV-SCH-007', 'INV-SCH-008']) {
    const c = cases.find((x) => x.id === id);
    invariants.push({ id, expr: exprMap[id], tol: c.tol, worst: c.abs_err, status: c.status, note: '' });
  }
  return buildReport({
    module: 'scheduler-forge', version: '1.0.0', layer: 'L3', cases, invariants,
    seed: '0x5C0D01', started, tol: tolMap
  });
}

const API = {
  name: 'scheduler-forge', version: '1.0.0', layer: 'L3',
  constantSchedule, linearWarmup, linearDecay, cosineSchedule, cosineWithRestarts,
  stepDecay, exponentialDecay, polynomialDecay, oneCycle, scheduleAt, selfCheck
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeScheduler = API;

if (typeof process !== 'undefined' && process.argv[1] && /scheduler-forge\.js$/.test(process.argv[1]) && (process.argv.includes('--self-check') || process.argv.includes('--json'))) {
  const r = selfCheck({ quick: process.argv.includes('--quick') });
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : V().formatTableSafe(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
