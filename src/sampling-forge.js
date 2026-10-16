/**
 * @forge-module sampling-forge
 * @layer L5
 * @version 1.0.0
 * @brief     大模型自回归生成采样内核：温度缩放 / 贪心 / top-k / top-p 核采样（nucleus）/ 重复惩罚 / 种子化抽样 / 批量采样。
 *            全部为纯函数式「logits → token 索引」管线，零 npm 依赖、零构建，Node require 与浏览器 globalThis 双端同构。
 * @reuse     Holtzman et al. 2019 (The Curious Case of Neural Text Degeneration)|https://arxiv.org/abs/1904.09751|arXiv|A|Top-p 核采样原式
 * @reuse     HuggingFace transformers|https://github.com/huggingface/transformers|Apache-2.0|B|LogitsProcessor 管线(温度/top-k/top-p/重复惩罚)与 HF 重复惩罚公式
 * @reuse     Fan et al. 2018 (Hierarchical Neural Story Generation)|https://arxiv.org/abs/1805.04833|arXiv|D|Top-k 采样经典约定
 * @beyond    纯 JS 单文件零构建；管线复用本项目 stable-forge 稳定 softmax 与 core-numeric 确定性 PRNG(sfc32)；
 *            抽样分布对 -Infinity 过滤位安全（top-k/top-p 之后仍可稳定归一化）；8 条硬不变量覆盖 T→0 收敛贪心、
 *            top-k/top-p 精确与最小性、重复惩罚公式精确性、正缩放不变、分布保真 MC 检验、确定性与批量逐行一致。
 * @invariant INV-SMP-001|温度→0 收敛贪心：唯一最大值 logits，T=1e-8 固定种子抽 500 次全部等于 greedy（不一致计数=0）|0|T→0 时 softmax 退化为 one-hot
 * @invariant INV-SMP-002|top-k 精确性：20 组随机 logits/k，过滤后非 -Inf 候选恰为 k 个且集合等于全排序前 k 名（违例计数=0）|0|定义等价
 * @invariant INV-SMP-003|top-p 最小性：20 组随机 logits/p，核集合累积概率 ≥ p 且移除最小概率成员后累积 < p（违例计数=0）|0|Holtzman 核采样最小集定义
 * @invariant INV-SMP-004|重复惩罚公式精确：正 logit→logit/coef、负 logit→logit*coef、未出现→不变，与内联参照逐元素偏差 ≤ 1e-12|1e-12|HF 约定参照实现
 * @invariant INV-SMP-005|正缩放不变：logits*3.7（c>0）后 greedy 与 top-1 全序不变（不一致计数=0）|0|正仿射单调性
 * @invariant INV-SMP-006|分布保真：[2,1,1] 偏态分布 N=20000 次抽样，经验频率与 softmax 目标最大偏差 ≤ 0.03|0.03|MCSE≈0.0035，取 8σ 余量
 * @invariant INV-SMP-007|确定性：同种子两次 sample 逐位相等；不同种子经验频率仍落在目标 ±0.05 内（违例=0）|0|sfc32 确定性 + 非塌缩
 * @invariant INV-SMP-008|批量一致：batchSample 第 i 行 == 同一 rng 种子派生对第 i 行单独 sample 的结果（不一致计数=0）|0|逐行共享连续 rng 流
 * @seed      0x9E3779B9
 */
'use strict';

/* ---------- 惰性双端依赖（Node require / 浏览器 globalThis 回退；启动期不强依赖） ---------- */
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
let _S = null;
function S() {
  if (_S) return _S;
  if (typeof require !== 'undefined') { try { _S = require('./stable-forge.js'); } catch (e) {} }
  if (!_S && typeof globalThis !== 'undefined') _S = globalThis.ForgeStable || null;
  if (!_S) throw new Error('stable-forge 依赖缺失');
  return _S;
}

/* ============================================================================
 * 采样策略内核（纯函数式）
 *   管线顺序（HuggingFace LogitsProcessor 约定）：
 *   重复惩罚 → 温度缩放 → top-k → top-p → 按 rng 从剩余分布抽样
 * ========================================================================== */

const MODULE_SEED = 0x9E3779B9;

/** 校验并拷贝 logits 为 Float64Array。allowNegInf=true 时允许 -Infinity（top-k/top-p 过滤位）。 */
function _toVector(a, allowNegInf) {
  if (a === null || a === undefined || typeof a.length !== 'number' || a.length < 1) {
    throw new TypeError('[sampling-forge] logits 须为非空数组');
  }
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    if (typeof v !== 'number' || Number.isNaN(v) || v === Infinity || (!allowNegInf && v === -Infinity)) {
      throw new RangeError('[sampling-forge] logits 元素须为有限实数' +
        (allowNegInf ? '（或 -Infinity 过滤位）' : '') + '，索引 ' + i + ' 收到 ' + v);
    }
    out[i] = v;
  }
  return out;
}

/** 校验温度：有限正数（T<=0 或非有限抛错）。 */
function _checkTemperature(T) {
  if (typeof T !== 'number' || !isFinite(T) || T <= 0) {
    throw new RangeError('[sampling-forge] temperature 须为有限正数，收到 ' + T);
  }
}

/** 降序索引排列：值降序，平局取索引升序（确定性；对 -Infinity 安全）。 */
function _orderDesc(vals) {
  const idx = new Array(vals.length);
  for (let i = 0; i < vals.length; i++) idx[i] = i;
  idx.sort((a, b) => {
    const d = vals[b] - vals[a];
    if (d > 0) return 1;
    if (d < 0) return -1;
    return a - b; // 平局：小索引优先
  });
  return idx;
}

/** 稳定 softmax（容忍 -Infinity 过滤位：exp(-Inf-m)=0；先减最大值防溢出）。全 -Infinity 抛错。 */
function _softmaxKeepNegInf(a) {
  let m = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i];
  if (m === -Infinity) throw new RangeError('[sampling-forge] 全部 logit 均为 -Infinity，无法构成分布');
  const e = new Float64Array(a.length);
  let s = 0;
  for (let i = 0; i < a.length; i++) { const x = Math.exp(a[i] - m); e[i] = x; s += x; }
  for (let i = 0; i < a.length; i++) e[i] /= s;
  return e;
}

/* ---------- 1. 温度缩放 softmax ---------- */

/**
 * 温度缩放分布：softmax(logits / T)。T<=0 抛错；内部复用 stable-forge 稳定 softmax 防溢出。
 * @param {Array<number>|Float64Array} logits 有限实数 logits（非空）
 * @param {number} T 温度（有限正数）
 * @returns {Float64Array} 概率分布（分量和=1）
 */
function softmaxTemperature(logits, T) {
  const v = _toVector(logits, false);
  _checkTemperature(T);
  const scaled = new Array(v.length);
  for (let i = 0; i < v.length; i++) scaled[i] = v[i] / T;
  return S().softmax(scaled);
}

/* ---------- 2. 贪心解码 ---------- */

/**
 * 贪心解码：argmax(logits)，平局取最小索引。允许 -Infinity（可直接作用于过滤后 logits）。
 * @returns {number} token 索引
 */
function greedy(logits) {
  const v = _toVector(logits, true);
  let best = 0;
  for (let i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
  return best;
}

/* ---------- 3. top-k 过滤 ---------- */

/**
 * top-k 过滤：保留最大的 k 个 logit，其余置 -Infinity（平局取小索引）。
 * @param {Array<number>|Float64Array} logits
 * @param {number} k 1..logits.length 的整数（越界抛错）
 * @returns {Float64Array} 过滤后 logits
 */
function topKFilter(logits, k) {
  const v = _toVector(logits, true);
  if (!Number.isInteger(k) || k <= 0) throw new RangeError('[sampling-forge] topK 须为正整数，收到 ' + k);
  if (k > v.length) throw new RangeError('[sampling-forge] topK=' + k + ' 超出词表大小 ' + v.length);
  const keep = new Set(_orderDesc(v).slice(0, k));
  const out = new Float64Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = keep.has(i) ? v[i] : -Infinity;
  return out;
}

/* ---------- 4. top-p 核采样（nucleus, Holtzman 2019） ---------- */

/**
 * top-p 过滤：按概率降序取「最小」候选集使累积概率 ≥ p，其余置 -Infinity。
 * 允许输入含 -Infinity（top-k 之后的级联）；p ∈ (0,1]，否则抛错。
 * @returns {Float64Array} 过滤后 logits
 */
function topPFilter(logits, p) {
  const v = _toVector(logits, true);
  if (typeof p !== 'number' || !isFinite(p) || p <= 0 || p > 1) {
    throw new RangeError('[sampling-forge] topP 须 ∈ (0,1]，收到 ' + p);
  }
  const probs = _softmaxKeepNegInf(v);
  const order = _orderDesc(probs);
  const keep = new Set();
  let cum = 0;
  for (let j = 0; j < order.length; j++) {
    const idx = order[j];
    keep.add(idx);
    cum += probs[idx];
    if (cum >= p) break; // 最小候选集：一旦越过 p 即停止（浮点累积不足时自然耗尽为全集）
  }
  const out = new Float64Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = keep.has(i) ? v[i] : -Infinity;
  return out;
}

/* ---------- 5. 重复惩罚（HuggingFace 约定） ---------- */

/**
 * 重复惩罚：history 中出现过的 token，正 logit 除以 coefficient，负 logit 乘以 coefficient；
 * 未出现过的 token 不变（logit=0 时两种公式同为 0）。
 * @param {Array<number>|Float64Array} logits 有限实数
 * @param {Array<number>} history 已生成 token 索引（允许重复，自动去重）
 * @param {number} coefficient 惩罚系数（有限正数；缺省 1.0 即不变）
 * @returns {Float64Array} 惩罚后 logits
 */
function repetitionPenalty(logits, history, coefficient) {
  const v = _toVector(logits, false);
  const coef = (coefficient === undefined || coefficient === null) ? 1.0 : coefficient;
  if (typeof coef !== 'number' || !isFinite(coef) || coef <= 0) {
    throw new RangeError('[sampling-forge] penalty.coefficient 须为有限正数，收到 ' + coefficient);
  }
  const seen = new Set();
  if (history) {
    for (const h of history) {
      if (!Number.isInteger(h) || h < 0 || h >= v.length) {
        throw new RangeError('[sampling-forge] penalty.history 元素须为 [0,' + v.length + ') 的整数，收到 ' + h);
      }
      seen.add(h);
    }
  }
  const out = new Float64Array(v.length);
  for (let i = 0; i < v.length; i++) {
    if (!seen.has(i)) { out[i] = v[i]; continue; }
    out[i] = v[i] < 0 ? v[i] * coef : v[i] / coef;
  }
  return out;
}

/* ---------- 6. 类别分布抽样（逆 CDF，确定性 rng） ---------- */

/** 按概率分布抽样一个索引（逆 CDF 单次扫描；每次调用恰好消耗 1 个均匀样本）。 */
function _drawCategorical(probs, rng) {
  const u = rng();
  let acc = 0, lastPositive = -1;
  for (let i = 0; i < probs.length; i++) {
    if (probs[i] > 0) lastPositive = i;
    acc += probs[i];
    if (u < acc) return i;
  }
  return lastPositive >= 0 ? lastPositive : probs.length - 1; // 浮点累积兜底：取最后一个正概率项
}

/* ---------- 7. 完整采样管线 ---------- */

/**
 * 完整自回归采样管线：重复惩罚 → 温度 → top-k → top-p → 按 rng 从剩余分布抽样。
 * @param {Array<number>|Float64Array} logits 有限实数（非空）
 * @param {Object} [opts] { temperature=1, topK, topP, rng, penalty:{history,coefficient} }
 * @returns {number} token 索引
 */
function sample(logits, opts) {
  opts = opts || {};
  let l = _toVector(logits, false);
  // 1) 重复惩罚
  const pen = opts.penalty;
  if (pen && pen.history && pen.history.length > 0) {
    l = repetitionPenalty(l, pen.history, pen.coefficient === undefined ? 1.0 : pen.coefficient);
  }
  // 2) 温度缩放
  const T = (opts.temperature === undefined || opts.temperature === null) ? 1.0 : opts.temperature;
  _checkTemperature(T);
  for (let i = 0; i < l.length; i++) l[i] /= T;
  // 3) top-k
  if (opts.topK !== undefined && opts.topK !== null) l = topKFilter(l, opts.topK);
  // 4) top-p
  if (opts.topP !== undefined && opts.topP !== null) l = topPFilter(l, opts.topP);
  // 5) 按 rng 从剩余分布抽样（rng 缺省时用固定种子派生，保持纯函数确定性）
  const rng = opts.rng || N().makeRng(MODULE_SEED);
  return _drawCategorical(_softmaxKeepNegInf(l), rng);
}

/* ---------- 8. 批量采样 ---------- */

/**
 * 批量采样：每行独立走 sample 管线，共享同一条连续 rng 流（逐行连续派生）。
 * @param {Array<Array<number>>} logits2D 二维 logits
 * @param {Object} [opts] 同 sample；rng 缺省时由 opts.seed（缺省 0x9E3779B9）派生
 * @returns {Array<number>} 每行 token 索引
 */
function batchSample(logits2D, opts) {
  opts = opts || {};
  if (!Array.isArray(logits2D)) throw new TypeError('[sampling-forge] batchSample 须传入二维数组');
  const seed = (opts.seed === undefined || opts.seed === null) ? MODULE_SEED : opts.seed;
  const rng = opts.rng || N().makeRng(seed);
  const out = new Array(logits2D.length);
  for (let i = 0; i < logits2D.length; i++) {
    out[i] = sample(logits2D[i], { ...opts, rng });
  }
  return out;
}

/* ---------- 9. 经验频率（自检辅助 / 导出） ---------- */

/**
 * 把一批抽样索引统计成频率向量（长度 vocabSize，分量和=1）。
 * @param {Array<number>} draws 抽样索引（非空，元素为 [0,vocabSize) 整数）
 * @param {number} vocabSize 词表大小（正整数）
 * @returns {Float64Array} 频率向量
 */
function empiricalFrequencies(draws, vocabSize) {
  if (!Array.isArray(draws) || draws.length === 0) {
    throw new TypeError('[sampling-forge] empiricalFrequencies 须为非空索引数组');
  }
  if (!Number.isInteger(vocabSize) || vocabSize <= 0) {
    throw new RangeError('[sampling-forge] vocabSize 须为正整数，收到 ' + vocabSize);
  }
  const freq = new Float64Array(vocabSize);
  for (const d of draws) {
    if (!Number.isInteger(d) || d < 0 || d >= vocabSize) {
      throw new RangeError('[sampling-forge] 抽样索引须为 [0,' + vocabSize + ') 的整数，收到 ' + d);
    }
    freq[d] += 1;
  }
  for (let i = 0; i < vocabSize; i++) freq[i] /= draws.length;
  return freq;
}

/* ============================================================================
 * selfCheck — 8 条硬不变量（每条可独立交叉验证）
 * ========================================================================== */
function selfCheck(opts) {
  opts = opts || {};
  const cases = [];
  const invariants = [];
  const { caseResult, buildReport } = V();
  const { makeRng, TOL } = N();

  // INV-SMP-001 温度→0 收敛贪心（唯一最大值，平局歧义消解）
  {
    const logits = [1.0, 2.5, 0.3, -1.2]; // 唯一最大值在索引 1
    const rng = makeRng(12345);
    const g = greedy(logits);
    let mismatch = 0;
    for (let i = 0; i < 500; i++) {
      if (sample(logits, { temperature: 1e-8, rng }) !== g) mismatch++;
    }
    cases.push(caseResult('INV-SMP-001', 'T=1e-8 固定种子抽 500 次全部等于 greedy', mismatch, 0, 0));
  }

  // INV-SMP-002 top-k 精确性：候选数恰为 k 且集合 = 全排序前 k 名
  {
    const rng = makeRng(777);
    let bad = 0;
    for (let g = 0; g < 20; g++) {
      const n = 6 + (g % 5);
      const logits = new Array(n);
      for (let i = 0; i < n; i++) logits[i] = rng() * 2 - 1;
      const k = 1 + Math.floor(rng() * n);
      const filtered = topKFilter(logits, k);
      const kept = [];
      for (let i = 0; i < n; i++) if (filtered[i] !== -Infinity) kept.push(i);
      if (kept.length !== k) { bad++; continue; }
      const topSet = new Set(_orderDesc(Float64Array.from(logits)).slice(0, k));
      if (kept.some(i => !topSet.has(i))) bad++;
    }
    cases.push(caseResult('INV-SMP-002', 'top-k：20 组候选数=k 且集合=全排序前 k', bad, 0, 0));
  }

  // INV-SMP-003 top-p 最小性：核集合累积 ≥ p，且移除最小概率成员后累积 < p
  {
    const rng = makeRng(2024);
    let bad = 0;
    for (let g = 0; g < 20; g++) {
      const n = 5 + (g % 6);
      const logits = new Array(n);
      for (let i = 0; i < n; i++) logits[i] = rng() * 4 - 2;
      const p = 0.2 + rng() * 0.7; // (0.2, 0.9)，避开 p=1 的浮点累积边界
      const filtered = topPFilter(logits, p);
      const probs = S().softmax(logits);
      const nucleus = [];
      for (let i = 0; i < n; i++) if (filtered[i] !== -Infinity) nucleus.push(i);
      let cum = 0, minP = Infinity;
      for (const i of nucleus) { cum += probs[i]; if (probs[i] < minP) minP = probs[i]; }
      const okCum = cum >= p - 1e-12;       // 累积概率 ≥ p
      const okMin = cum - minP < p + 1e-12; // 移除最小成员后 < p（最小性）
      if (!(okCum && okMin)) bad++;
    }
    cases.push(caseResult('INV-SMP-003', 'top-p：20 组累积≥p 且去最小成员后<p', bad, 0, 0));
  }

  // INV-SMP-004 重复惩罚公式精确：正除 / 负乘 / 未出现不变，vs 内联参照
  {
    const rng = makeRng(555);
    let maxDev = 0;
    const check = (logits, history, coef) => {
      const got = repetitionPenalty(logits, history, coef);
      const seen = new Set(history);
      for (let i = 0; i < logits.length; i++) {
        const ref = seen.has(i) ? (logits[i] < 0 ? logits[i] * coef : logits[i] / coef) : logits[i];
        maxDev = Math.max(maxDev, Math.abs(got[i] - ref));
      }
    };
    for (let g = 0; g < 20; g++) {
      const n = 8;
      const logits = new Array(n);
      for (let i = 0; i < n; i++) logits[i] = rng() * 6 - 3; // 含正 / 负 / 近零
      const history = [];
      for (let i = 0; i < n; i++) if (rng() < 0.5) history.push(i);
      if (history.length === 0) history.push(1);
      if (rng() < 0.5) history.push(history[0]); // 重复历史项（惩罚幂等）
      check(logits, history, 1.2 + rng() * 0.8); // coef ∈ (1.2, 2.0)
    }
    check([0, 1.5, -2.5, 0, 3.0], [0, 3, 4], 2.0); // 确定性边界：零 logit 两种公式同为 0
    check([1, -1], [], 1.5);                        // 空历史：全部不变
    cases.push(caseResult('INV-SMP-004', '重复惩罚 vs 内联参照（20 组随机+边界）', maxDev, 0, TOL.RESID));
  }

  // INV-SMP-005 正缩放不变：logits*3.7 后 greedy / top-1 全序 / top-k 集合不变
  {
    const rng = makeRng(31337);
    const c = 3.7;
    let bad = 0;
    for (let g = 0; g < 20; g++) {
      const n = 5 + (g % 5);
      const logits = new Array(n);
      for (let i = 0; i < n; i++) logits[i] = rng() * 4 - 2;
      const scaled = logits.map(v => v * c);
      if (greedy(logits) !== greedy(scaled)) { bad++; continue; }
      const o1 = _orderDesc(Float64Array.from(logits));
      const o2 = _orderDesc(Float64Array.from(scaled));
      for (let i = 0; i < n; i++) if (o1[i] !== o2[i]) { bad++; break; }
      const f1 = topKFilter(logits, 3), f2 = topKFilter(scaled, 3);
      for (let i = 0; i < n; i++) if ((f1[i] !== -Infinity) !== (f2[i] !== -Infinity)) { bad++; break; }
    }
    cases.push(caseResult('INV-SMP-005', 'logits*3.7 后 greedy/top-1 全序/top-k 集合不变', bad, 0, 0));
  }

  // INV-SMP-006 分布保真：[2,1,1] 偏态分布 N=20000，经验频率 vs softmax 目标
  {
    const logits = [2, 1, 1];
    const target = softmaxTemperature(logits, 1);
    const rng = makeRng(0x9E3779B9);
    const nDraws = 20000;
    const draws = new Array(nDraws);
    for (let i = 0; i < nDraws; i++) draws[i] = sample(logits, { temperature: 1, rng });
    const freq = empiricalFrequencies(draws, 3);
    let maxDev = 0;
    for (let i = 0; i < 3; i++) maxDev = Math.max(maxDev, Math.abs(freq[i] - target[i]));
    cases.push(caseResult('INV-SMP-006', '[2,1,1] N=20000 经验频率 vs softmax 目标最大偏差', maxDev, 0, 0.03));
  }

  // INV-SMP-007 确定性：同种子逐位相等；异种子不塌缩（频率在目标 ±0.05 内）
  {
    const logits = [2, 1, 1];
    const target = softmaxTemperature(logits, 1);
    const rngA = makeRng(424242), rngB = makeRng(424242);
    let detBad = 0;
    for (let i = 0; i < 200; i++) {
      if (sample(logits, { rng: rngA }) !== sample(logits, { rng: rngB })) detBad++;
    }
    const rngC = makeRng(987654321);
    const nDraws = 20000;
    const draws = new Array(nDraws);
    for (let i = 0; i < nDraws; i++) draws[i] = sample(logits, { rng: rngC });
    const freq = empiricalFrequencies(draws, 3);
    let maxDev = 0;
    for (let i = 0; i < 3; i++) maxDev = Math.max(maxDev, Math.abs(freq[i] - target[i]));
    const ok = detBad === 0 && maxDev <= 0.05;
    cases.push(caseResult('INV-SMP-007', '同种子 sample 逐位确定 + 异种子不塌缩(±0.05)',
      ok ? 0 : 1, 0, 0, 'detBad=' + detBad + ' diffSeedMaxDev=' + maxDev.toFixed(6)));
  }

  // INV-SMP-008 批量一致：batchSample 第 i 行 == 同一 rng 种子派生对第 i 行单独 sample
  {
    const rng = makeRng(24680);
    const rows = [];
    for (let i = 0; i < 8; i++) {
      const row = new Array(6);
      for (let j = 0; j < 6; j++) row[j] = rng() * 4 - 2;
      rows.push(row);
    }
    const opts = { temperature: 0.9, topK: 3, penalty: { history: [0, 2], coefficient: 1.3 } };
    const rngBatch = makeRng(13579);
    const rngRef = makeRng(13579);
    const batch = batchSample(rows, { ...opts, rng: rngBatch });
    const ref = rows.map(row => sample(row, { ...opts, rng: rngRef }));
    let bad = 0;
    for (let i = 0; i < rows.length; i++) if (batch[i] !== ref[i]) bad++;
    cases.push(caseResult('INV-SMP-008', 'batchSample 8 行逐行等于同种子单独 sample', bad, 0, 0));
  }

  // 不变量登记（8 条， worst 取实测绝对偏差）
  const INV_EXPR = {
    'INV-SMP-001': 'T=1e-8 时 500 次抽样全部等于 greedy（不一致数=0）',
    'INV-SMP-002': 'top-k 过滤后候选恰为 k 且等于全排序前 k（违例数=0）',
    'INV-SMP-003': 'top-p 核集合累积≥p 且移除最小成员后<p（违例数=0）',
    'INV-SMP-004': '重复惩罚 正除/负乘/未出现不变，与内联参照偏差≤1e-12',
    'INV-SMP-005': 'logits*3.7 正缩放不变：greedy/top-1 全序/top-k 集合（违例数=0）',
    'INV-SMP-006': '[2,1,1] N=20000 经验频率 vs softmax 目标最大偏差≤0.03',
    'INV-SMP-007': '同种子 sample 逐位相等；异种子频率在目标±0.05 内',
    'INV-SMP-008': 'batchSample 第 i 行等于同种子派生的单独 sample（违例数=0）'
  };
  for (const id of Object.keys(INV_EXPR)) {
    const c = cases.find(x => x.id === id);
    invariants.push({ id, expr: INV_EXPR[id], tol: c.tol, worst: c.abs_err, status: c.status, note: '' });
  }

  const started = Date.now();
  return buildReport({
    module: 'sampling-forge', version: '1.0.0', layer: 'L5', cases, invariants,
    seed: '0x9E3779B9', started,
    tol: {
      'INV-SMP-001': 0, 'INV-SMP-002': 0, 'INV-SMP-003': 0, 'INV-SMP-004': TOL.RESID,
      'INV-SMP-005': 0, 'INV-SMP-006': 0.03, 'INV-SMP-007': 0, 'INV-SMP-008': 0
    }
  });
}

const API = {
  name: 'sampling-forge', version: '1.0.0', layer: 'L5',
  softmaxTemperature, greedy, topKFilter, topPFilter, repetitionPenalty,
  sample, batchSample, empiricalFrequencies, selfCheck
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeSampling = API;

if (typeof process !== 'undefined' && process.argv[1] && /sampling-forge\.js$/.test(process.argv[1]) &&
    (process.argv.includes('--self-check') || process.argv.includes('--json'))) {
  const r = selfCheck({ quick: process.argv.includes('--quick') });
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : require('./core-verify.js').formatTableSafe(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
