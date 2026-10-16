/**
 * @forge-module speculative-forge
 * @layer L4
 * @version 1.0.0
 * @brief     推测解码（Speculative Decoding）内核：草稿模型自回归提议 k 个 token（含每步完整分布）→ 目标模型
 *            一次性并行验证 → Leviathan 拒绝采样 min(1, q/p) 接受 / 否则从 norm(max(0, q−p)) 重采样 →
 *            每步产出 1..k+1 个 token（含 bonus）。皇冠明珠：输出分布与目标模型单独逐 token 采样严格等价
 *            （接受-重采样无偏恒等式 min(p,q)+(q−p)₊=q）；目标调用按「并行验证趟」计，加速比存在。
 *            另含贪心路径专用验证 greedyVerifyChunk 与 k=1 退化（标准拒绝采样）。
 * @reuse     Fast Inference from Transformers via Speculative Decoding (Leviathan et al. 2023)|https://arxiv.org/abs/2211.17192|MIT|A|拒绝采样接受准则 min(1,q/p)、残差重采样 norm(max(0,q−p)) 与 bonus token
 * @reuse     Accelerating LLM Decoding with Speculative Sampling (Chen et al. 2023)|https://arxiv.org/abs/2302.01318|MIT|A|分布等价性证明路径（接受-重采样联合分布=目标分布）
 * @reuse     Medusa (Cai et al. 2024)|https://arxiv.org/abs/2401.10774|Apache-2.0|B|多草稿头并行验证结构
 * @reuse     EAGLE (Li et al. 2024)|https://arxiv.org/abs/2401.15077|Apache-2.0|C|特征级自回归草稿
 * @beyond    纯 JS 单文件零构建零依赖；targetCalls 按文献口径计「并行验证趟」（真实 Transformer 一趟并行前向产出
 *            全部 k+1 个分布；纯函数抽象逐位置求值，计趟不计次）；分布等价以 N=20000 大规模蒙特卡洛与
 *            sampling-forge 目标基线双侧交叉验证（MCSE≤√(0.25/N)≈0.0035，最大偏差 8σ≈0.028 < 0.03）；
 *            贪心 / 随机 / k=1 退化三路径与 20 组随机场景结构校验全覆盖。
 * @invariant INV-SPC-001|分布等价（皇冠明珠）：draft≠target 双分布场景 N=20000 次，speculativeDecode 与 target-only 采样的首 token 经验分布各自与目标分布最大偏差 ≤ 0.03|0.03|两侧独立大样本 MC；MCSE≈0.0035，8σ≈0.028
 * @invariant INV-SPC-002|贪心路径全接受：draft 每步 argmax==target argmax 时 greedyVerifyChunk 返回 k 且无拒绝；argmax 分歧位置 j 时返回 j（违例计数=0）|0|argmax 一致性定义（正温度缩放不变）
 * @invariant INV-SPC-003|每步产出上界：20 组随机场景 1 ≤ tokensAccepted+bonus ≤ k+1，且 ctx' 长度=ctx+产出、tokensAccepted 为草稿前缀、targetCalls=1/draftCalls=k（违例计数=0）|0|拒绝早停 + bonus 恒存在
 * @invariant INV-SPC-004|拒绝采样无偏：固定 q/p 单 token N=20000 次经验频率与 q 最大偏差 ≤ 0.03|0.03|恒等式 min(p,q)+(q−p)₊=q；MCSE≈0.0035，8σ 余量
 * @invariant INV-SPC-005|接受率单调：draft==target 时接受率==1（精确）；3 档差异增大（解析 Σmin(p,q)≈0.91/0.74/0.48）经验接受率单调不增（违例计数=0）|0|Σmin(p,q)=1−TV(p,q) 随差异单调降
 * @invariant INV-SPC-006|单 token 退化：k=1 时 speculativeDecode 与 target-only 独立重实现在 L=3 个位置的经验分布各自与目标分布一致（≤0.03）|0.03|k=1 拒绝采样边际=目标分布（位置归纳）
 * @invariant INV-SPC-007|目标调用有界：生成 n token，targetCalls ≤ n 且 ≤ draftCalls/k+n；draft==target 时 targetCalls==ceil(n/(k+1))（精确，证明确有加速）|0|每步≥1 token + 全接受每步 k+1 token
 * @invariant INV-SPC-008|确定性：同种子两次 speculativeDecode 输出 token 序列与统计量逐位相等（违例计数=0）|0|sfc32 确定性 + 纯函数式管线
 * @seed      0x9E3779B9
 */
'use strict';

/* ---------- 惰性双端依赖（Node require / 浏览器 globalThis 回退；启动期不强依赖） ---------- */
let _V = null;
function V() {
  if (_V) return _V;
  if (typeof require !== 'undefined') { try { _V = require('./core-verify.js'); } catch (e) {} }
  if (!_V && typeof globalThis !== 'undefined') _V = globalThis.ForgeVerify || null;
  if (!_V) throw new Error('[speculative-forge] core-verify 依赖缺失');
  return _V;
}
let _N = null;
function N() {
  if (_N) return _N;
  if (typeof require !== 'undefined') { try { _N = require('./core-numeric.js'); } catch (e) {} }
  if (!_N && typeof globalThis !== 'undefined') _N = globalThis.ForgeNumeric || null;
  if (!_N) throw new Error('[speculative-forge] core-numeric 依赖缺失');
  return _N;
}
let _S = null;
function S() {
  if (_S) return _S;
  if (typeof require !== 'undefined') { try { _S = require('./sampling-forge.js'); } catch (e) {} }
  if (!_S && typeof globalThis !== 'undefined') _S = globalThis.ForgeSampling || null;
  if (!_S) throw new Error('[speculative-forge] sampling-forge 依赖缺失');
  return _S;
}

const MODULE_SEED = 0x9E3779B9;
const MC_N = 20000;   // 蒙特卡洛样本量（INV-SPC-001/004/006 共用）
const MC_TOL = 0.03;  // MCSE≤√(0.25/N)≈0.0035；V≤5 时最大偏差期望≈σ·E[max|Z|]≈0.004，8σ≈0.028<0.03

/* ============================================================================
 * 输入校验与内部工具（纯函数式，零依赖）
 * ========================================================================== */

/** 校验 ctx 为 token 序列（普通数组，元素为非负整数），原样返回（拷贝由调用方负责）。 */
function _checkCtx(ctx) {
  if (!Array.isArray(ctx)) throw new TypeError('[speculative-forge] ctx 须为普通数组（token 序列）');
  for (let i = 0; i < ctx.length; i++) {
    if (!Number.isInteger(ctx[i]) || ctx[i] < 0) {
      throw new RangeError('[speculative-forge] ctx[' + i + '] 须为非负整数 token，收到 ' + ctx[i]);
    }
  }
  return ctx;
}

/** 校验模型函数 fn(ctx) -> logits。 */
function _checkModelFn(fn, name) {
  if (typeof fn !== 'function') throw new TypeError('[speculative-forge] ' + name + ' 须为函数 (ctx)->logits');
  return fn;
}

/** 校验 logits 输出：非空、有限实数；原样返回（下游 softmax/greedy 自行拷贝，不修改原对象）。 */
function _checkLogits(logits, name) {
  if (logits === null || logits === undefined || typeof logits.length !== 'number' || logits.length < 1) {
    throw new TypeError('[speculative-forge] ' + name + ' 须返回非空 logits 向量');
  }
  for (let i = 0; i < logits.length; i++) {
    const v = logits[i];
    if (typeof v !== 'number' || !isFinite(v)) {
      throw new RangeError('[speculative-forge] ' + name + ' logits[' + i + '] 须为有限实数，收到 ' + v);
    }
  }
  return logits;
}

/** 校验 k / maxTokens：正整数。 */
function _checkPositiveInt(v, name) {
  if (!Number.isInteger(v) || v < 1) throw new RangeError('[speculative-forge] ' + name + ' 须为正整数，收到 ' + v);
  return v;
}

/** 温度：有限正数（缺省 1.0）。 */
function _temperature(opts) {
  const T = (opts && opts.temperature !== undefined && opts.temperature !== null) ? opts.temperature : 1.0;
  if (typeof T !== 'number' || !isFinite(T) || T <= 0) {
    throw new RangeError('[speculative-forge] temperature 须为有限正数，收到 ' + T);
  }
  return T;
}

/** rng：缺省时由模块种子派生（保持纯函数确定性，同 sampling-forge 约定）。 */
function _rng(opts) {
  const r = (opts && opts.rng !== undefined && opts.rng !== null) ? opts.rng : N().makeRng(MODULE_SEED);
  if (typeof r !== 'function') throw new TypeError('[speculative-forge] opts.rng 须为函数 ()->[0,1)');
  return r;
}

/** 校验概率分布：非空、有限非负、分量和≈1（容差 1e-6）；返回长度。 */
function _checkDist(probs, name) {
  if (probs === null || probs === undefined || typeof probs.length !== 'number' || probs.length < 1) {
    throw new TypeError('[speculative-forge] ' + name + ' 须为非空概率向量');
  }
  let s = 0;
  for (let i = 0; i < probs.length; i++) {
    const v = probs[i];
    if (typeof v !== 'number' || !isFinite(v) || v < 0) {
      throw new RangeError('[speculative-forge] ' + name + '[' + i + '] 须为有限非负实数，收到 ' + v);
    }
    s += v;
  }
  if (Math.abs(s - 1) > 1e-6) {
    throw new RangeError('[speculative-forge] ' + name + ' 分量和须为 1（容差 1e-6），实际 ' + s);
  }
  return probs.length;
}

/**
 * 按概率向量抽样一个索引（逆 CDF 单次扫描；与 sampling-forge._drawCategorical 同构，
 * 每次调用恰好消耗 1 个均匀样本；零概率项永不命中，浮点累积不足时兜底最后一个正概率项）。
 */
function _drawFromProbs(probs, rng) {
  const u = rng();
  let acc = 0, lastPositive = -1;
  for (let i = 0; i < probs.length; i++) {
    if (probs[i] > 0) lastPositive = i;
    acc += probs[i];
    if (u < acc) return i;
  }
  return lastPositive >= 0 ? lastPositive : probs.length - 1;
}

/** short 是否为 long 的前缀（逐位相等）。 */
function _isPrefix(short, long) {
  if (short.length > long.length) return false;
  for (let i = 0; i < short.length; i++) if (short[i] !== long[i]) return false;
  return true;
}

/* ============================================================================
 * 1. 草稿提议（proposeTokens）
 * ========================================================================== */

/**
 * 草稿模型自回归提出 k 个 token 及其每步完整概率分布。
 * @param {Function} draftFn 草稿模型 (ctx) -> logits（词表分数）
 * @param {Array<number>} ctx 上下文 token 序列（非负整数数组）
 * @param {number} k 草稿长度（正整数）
 * @param {Object} [opts] { temperature=1, rng, greedy=false }；greedy=true 时每步取 argmax（概率分布仍记录）
 * @returns {{tokens: Array<number>, probs: Array<Float64Array>}} probs[i] 为第 i 个 draft token 的完整概率分布
 */
function proposeTokens(draftFn, ctx, k, opts) {
  _checkModelFn(draftFn, 'draftFn');
  _checkCtx(ctx);
  _checkPositiveInt(k, 'k');
  const T = _temperature(opts);
  const rng = _rng(opts);
  const greedyMode = !!(opts && opts.greedy);
  const tokens = new Array(k);
  const probs = new Array(k);
  const cur = ctx.slice();
  for (let i = 0; i < k; i++) {
    const logits = _checkLogits(draftFn(cur), 'draftFn');
    const p = S().softmaxTemperature(logits, T); // 复用 sampling-forge 稳定 softmax
    tokens[i] = greedyMode ? S().greedy(logits) : _drawFromProbs(p, rng);
    probs[i] = p;
    cur.push(tokens[i]);
  }
  return { tokens, probs };
}

/* ============================================================================
 * 2. Leviathan 拒绝采样（rejectionSample）
 * ========================================================================== */

/**
 * Leviathan et al. 2023 拒绝采样：以 min(1, targetProb[x]/draftProb[x]) 接受草稿 token x，
 * 否则从 norm(max(0, targetProb - draftProb)) 重采样。
 * 无偏性恒等式：P(输出=y) = min(p,q)[y] + (q−p)₊[y] = q[y]（输出边际恰为目标分布）。
 * @param {Array<number>|Float64Array} targetProb 目标分布 q（分量和=1）
 * @param {Array<number>|Float64Array} draftProb 草稿分布 p（分量和=1）
 * @param {Function} rng ()->[0,1)
 * @param {number} [x] 待验证的草稿 token（verifyChunk 传入实际草稿 token，保证拒绝判定与产出严格同一）；
 *        缺省时自 draftProb 抽一个 x（独立单 token 无偏采样路径，见 INV-SPC-004）
 * @returns {{accepted: boolean, token: number}}
 */
function rejectionSample(targetProb, draftProb, rng, x) {
  const n = _checkDist(targetProb, 'targetProb');
  if (_checkDist(draftProb, 'draftProb') !== n) {
    throw new RangeError('[speculative-forge] targetProb 与 draftProb 长度不一致');
  }
  if (typeof rng !== 'function') throw new TypeError('[speculative-forge] rng 须为函数 ()->[0,1)');
  const hasX = x !== undefined && x !== null;
  if (hasX && (!Number.isInteger(x) || x < 0 || x >= n)) {
    throw new RangeError('[speculative-forge] 草稿 token x 须为 [0,' + n + ') 的整数，收到 ' + x);
  }
  // 1) 草稿 token：缺省自 p 抽取（恰好消耗 1 个均匀样本）；验证路径由调用方给定实际草稿 token
  const tok = hasX ? x : _drawFromProbs(draftProb, rng);
  // 2) 接受概率 min(1, q[x]/p[x])；p[x]=0 时（除兜底不可达）q[x]>0 必接受
  const ratio = draftProb[tok] > 0 ? Math.min(1, targetProb[tok] / draftProb[tok]) : (targetProb[tok] > 0 ? 1 : 0);
  if (rng() < ratio) return { accepted: true, token: tok };
  // 3) 拒绝：残差分布 norm(max(0, q−p)) 重采样
  const res = new Float64Array(n);
  let s = 0;
  for (let i = 0; i < n; i++) { const d = targetProb[i] - draftProb[i]; const v = d > 0 ? d : 0; res[i] = v; s += v; }
  if (s <= 0) return { accepted: false, token: _drawFromProbs(targetProb, rng) }; // q≡p 数值兜底（理论不可达）
  for (let i = 0; i < n; i++) res[i] /= s;
  return { accepted: false, token: _drawFromProbs(res, rng) };
}

/* ============================================================================
 * 3. 目标并行验证（verifyChunk）
 * ========================================================================== */

/**
 * 目标模型对 ctx + draft.tokens 一次性并行验证（抽象计 1 趟；纯函数下逐位置求值，早停于首个拒绝）。
 * 位置 j 的验证分布 q_j = softmax(targetFn(ctx + draft[0..j-1]) / T)；拒绝位置 j 的产出为残差重采样
 * token，全部接受时从第 k 个位置的目标分布额外采一个 bonus token。
 * @param {Function} targetFn 目标模型 (ctx) -> logits
 * @param {Array<number>} ctx 上下文 token 序列
 * @param {{tokens: Array<number>, probs: Array<Array<number>>}} draft proposeTokens 的产出
 * @param {number} k 草稿长度（正整数）
 * @param {Object} [opts] { temperature=1, rng }
 * @returns {{nAccepted: number, bonusToken: number, nRejected: number}}
 *   bonusToken 恒非空（拒绝=残差重采样 token；全接受=额外采样 token）→ 每步产出 ≥ 1。
 */
function verifyChunk(targetFn, ctx, draft, k, opts) {
  _checkModelFn(targetFn, 'targetFn');
  _checkCtx(ctx);
  if (draft === null || draft === undefined || !Array.isArray(draft.tokens) || !Array.isArray(draft.probs)) {
    throw new TypeError('[speculative-forge] draft 须为 {tokens:number[], probs:Array}');
  }
  _checkPositiveInt(k, 'k');
  const T = _temperature(opts);
  const rng = _rng(opts);
  const kk = Math.min(k, draft.tokens.length, draft.probs.length);
  if (kk < 1) throw new RangeError('[speculative-forge] draft.tokens/probs 与 k 无交集');
  const base = ctx.slice();
  for (let j = 0; j < kk; j++) {
    const q = S().softmaxTemperature(_checkLogits(targetFn(base), 'targetFn'), T);
    if (q.length !== draft.probs[j].length) {
      throw new RangeError('[speculative-forge] 目标词表 ' + q.length + ' 与草稿概率长度 ' + draft.probs[j].length + ' 不一致');
    }
    const r = rejectionSample(q, draft.probs[j], rng, draft.tokens[j]); // 传入实际草稿 token
    if (r.accepted) { base.push(draft.tokens[j]); continue; }
    return { nAccepted: j, bonusToken: r.token, nRejected: kk - j };
  }
  // 全部接受：第 k 个位置的目标分布额外采一个 bonus token
  const qk = S().softmaxTemperature(_checkLogits(targetFn(base), 'targetFn'), T);
  return { nAccepted: kk, bonusToken: _drawFromProbs(qk, rng), nRejected: 0 };
}

/* ============================================================================
 * 4. 单步推测解码（speculativeStep）
 * ========================================================================== */

/**
 * 单步：propose(k 次草稿调用) → verify(1 趟目标并行验证) → 返回新上下文与调用统计。
 * @param {Function} draftFn 草稿模型 (ctx) -> logits
 * @param {Function} targetFn 目标模型 (ctx) -> logits
 * @param {Array<number>} ctx 上下文 token 序列
 * @param {number} k 草稿长度（正整数）
 * @param {Object} [opts] { temperature=1, rng, greedy }
 * @returns {{ctx: Array<number>, tokensAccepted: Array<number>, bonus: number, targetCalls: number, draftCalls: number}}
 *   ctx 为【新上下文】（输入 ctx + tokensAccepted + bonus）；targetCalls 计并行验证趟（恒 1），draftCalls=k。
 */
function speculativeStep(draftFn, targetFn, ctx, k, opts) {
  _checkModelFn(draftFn, 'draftFn');
  _checkModelFn(targetFn, 'targetFn');
  _checkCtx(ctx);
  _checkPositiveInt(k, 'k');
  const draft = proposeTokens(draftFn, ctx, k, opts);
  const v = verifyChunk(targetFn, ctx, draft, k, opts);
  const tokensAccepted = draft.tokens.slice(0, v.nAccepted);
  const next = ctx.concat(tokensAccepted);
  if (v.bonusToken !== null && v.bonusToken !== undefined) next.push(v.bonusToken);
  return { ctx: next, tokensAccepted, bonus: v.bonusToken, targetCalls: 1, draftCalls: k };
}

/* ============================================================================
 * 5. 完整推测解码循环（speculativeDecode）
 * ========================================================================== */

/**
 * 完整循环：每步产出 1..k+1 个 token，直至 maxTokens 或 EOS（EOS 本身不计入输出）。
 * @param {Function} draftFn 草稿模型 (ctx) -> logits
 * @param {Function} targetFn 目标模型 (ctx) -> logits
 * @param {Array<number>} ctx 初始上下文
 * @param {Object} [opts] { k=4, maxTokens=32, eos=-1(关闭), temperature=1, rng, greedy }
 * @returns {{tokens: Array<number>, steps: number, acceptedDraft: number, proposedDraft: number, targetCalls: number, draftCalls: number}}
 */
function speculativeDecode(draftFn, targetFn, ctx, opts) {
  _checkModelFn(draftFn, 'draftFn');
  _checkModelFn(targetFn, 'targetFn');
  _checkCtx(ctx);
  opts = opts || {};
  const k = _checkPositiveInt(opts.k === undefined || opts.k === null ? 4 : opts.k, 'opts.k');
  const maxTokens = _checkPositiveInt(opts.maxTokens === undefined || opts.maxTokens === null ? 32 : opts.maxTokens, 'opts.maxTokens');
  const eos = (opts.eos === undefined || opts.eos === null) ? -1 : opts.eos;
  const T = _temperature(opts);
  const rng = _rng(opts);
  const stepOpts = { temperature: T, rng, greedy: !!opts.greedy };
  let cur = ctx.slice();
  const tokens = [];
  let steps = 0, acceptedDraft = 0, proposedDraft = 0, targetCalls = 0, draftCalls = 0;
  while (tokens.length < maxTokens) {
    const st = speculativeStep(draftFn, targetFn, cur, k, stepOpts);
    steps++; targetCalls += st.targetCalls; draftCalls += st.draftCalls; proposedDraft += k;
    const emitted = st.tokensAccepted.length + (st.bonus === null || st.bonus === undefined ? 0 : 1);
    let take = Math.min(emitted, maxTokens - tokens.length);
    let eosHit = false;
    for (let i = 0; i < take; i++) {
      const t = i < st.tokensAccepted.length ? st.tokensAccepted[i] : st.bonus;
      if (t === eos) { take = i; eosHit = true; break; }
    }
    acceptedDraft += Math.min(take, st.tokensAccepted.length);
    for (let i = 0; i < take; i++) {
      tokens.push(i < st.tokensAccepted.length ? st.tokensAccepted[i] : st.bonus);
    }
    cur = st.ctx;
    if (eosHit) break;
  }
  return { tokens, steps, acceptedDraft, proposedDraft, targetCalls, draftCalls };
}

/* ============================================================================
 * 6. 接受率统计（acceptanceRate）
 * ========================================================================== */

/**
 * 接受率 = acceptedDraft / proposedDraft ∈ [0,1]；proposedDraft=0 时返回 0（无提议则无接受）。
 * @param {{proposedDraft: number, acceptedDraft: number}} stats speculativeDecode 的统计产出
 * @returns {number} 接受率
 */
function acceptanceRate(stats) {
  if (stats === null || stats === undefined) throw new TypeError('[speculative-forge] acceptanceRate 须传入统计对象');
  const proposed = stats.proposedDraft, accepted = stats.acceptedDraft;
  if (!Number.isInteger(proposed) || proposed < 0 || !Number.isInteger(accepted) || accepted < 0) {
    throw new RangeError('[speculative-forge] proposedDraft/acceptedDraft 须为非负整数');
  }
  if (accepted > proposed) throw new RangeError('[speculative-forge] acceptedDraft 不得超过 proposedDraft');
  return proposed === 0 ? 0 : accepted / proposed;
}

/* ============================================================================
 * 7. 贪心路径验证（greedyVerifyChunk）
 * ========================================================================== */

/**
 * 贪心路径专用：逐位置比较 draft token 与目标 argmax，返回接受前缀长度（0..draft.tokens.length）。
 * argmax 对正温度缩放不变，故无需温度参数；全一致时返回 k（无拒绝）。
 * @param {Function} targetFn 目标模型 (ctx) -> logits
 * @param {Array<number>} ctx 上下文 token 序列
 * @param {{tokens: Array<number>}} draft 草稿 token 序列（通常由 proposeTokens greedy=true 产出）
 * @returns {number} 接受前缀长度
 */
function greedyVerifyChunk(targetFn, ctx, draft) {
  _checkModelFn(targetFn, 'targetFn');
  _checkCtx(ctx);
  if (draft === null || draft === undefined || !Array.isArray(draft.tokens)) {
    throw new TypeError('[speculative-forge] draft 须为 {tokens:number[], ...}');
  }
  const base = ctx.slice();
  for (let j = 0; j < draft.tokens.length; j++) {
    const tArg = S().greedy(_checkLogits(targetFn(base), 'targetFn'));
    if (tArg !== draft.tokens[j]) return j;
    base.push(draft.tokens[j]);
  }
  return draft.tokens.length;
}

/* ============================================================================
 * selfCheck — 8 条硬不变量（每条可独立交叉验证）
 * ========================================================================== */

/** 位置循环表模型：model(ctx) 返回 table[ctx.length % table.length]（返回行为共享只读，下游不修改）。 */
function _tableModel(table) {
  return function (ctx) { return table[ctx.length % table.length]; };
}

/** 随机表模型：rows 行 × vocab 列，logits ∈ (-2,2)。 */
function _randomTableModel(rng, rows, vocab) {
  const table = new Array(rows);
  for (let r = 0; r < rows; r++) {
    const row = new Float64Array(vocab);
    for (let v = 0; v < vocab; v++) row[v] = rng() * 4 - 2;
    table[r] = row;
  }
  return _tableModel(table);
}

function selfCheck(opts) {
  opts = opts || {};
  const cases = [];
  const invariants = [];
  const { caseResult, buildReport } = V();
  const { makeRng } = N();

  /* ---- 双分布测试场景：draft 与 target 不同（拒绝/重采样路径均被高频触发） ---- */
  const V_ = 5;
  const Q_TABLE = [ // 目标 logits 表（位置 %3 轮换）
    Float64Array.from([2.0, 1.2, 0.4, -0.3, -1.1]),
    Float64Array.from([0.9, 2.1, 0.7, -0.2, -0.8]),
    Float64Array.from([1.4, 0.6, 1.9, -0.5, -0.9])
  ];
  const P_TABLE = [ // 草稿 logits 表（与目标不同）
    Float64Array.from([1.5, 1.6, 0.9, 0.1, -0.6]),
    Float64Array.from([1.2, 1.8, 1.0, 0.3, -0.4]),
    Float64Array.from([1.1, 1.0, 1.7, 0.2, -0.5])
  ];
  const targetFn = _tableModel(Q_TABLE);
  const draftFn = _tableModel(P_TABLE);

  // INV-SPC-001 分布等价（皇冠明珠）：相同 rng 种子分别跑 (a) 推测解码 (b) target-only 采样，
  // N=20000 次统计首 token 经验分布，两者与目标分布最大偏差均 ≤ 0.03。
  // MCSE≤√(0.25/20000)≈0.0035；V=5 时最大偏差期望≈σ·E[max|Z|]≈0.004，8σ≈0.028<0.03。
  {
    const ctx0 = [0];
    const n = MC_N;
    const specDraws = new Array(n);
    const tgtDraws = new Array(n);
    for (let i = 0; i < n; i++) {
      const rng = makeRng(0x5EED1A1 + i); // 相同 rng 种子
      specDraws[i] = speculativeDecode(draftFn, targetFn, ctx0, { k: 4, maxTokens: 1, rng }).tokens[0];
    }
    for (let i = 0; i < n; i++) {
      const rng = makeRng(0x5EED1A1 + i);
      tgtDraws[i] = S().sample(targetFn(ctx0), { temperature: 1, rng }); // sampling-forge 目标基线
    }
    const q0 = S().softmaxTemperature(targetFn(ctx0), 1);
    const fA = S().empiricalFrequencies(specDraws, V_);
    const fB = S().empiricalFrequencies(tgtDraws, V_);
    let devA = 0, devB = 0;
    for (let v = 0; v < V_; v++) {
      devA = Math.max(devA, Math.abs(fA[v] - q0[v]));
      devB = Math.max(devB, Math.abs(fB[v] - q0[v]));
    }
    cases.push(caseResult('INV-SPC-001', '分布等价：推测 vs 目标-only 首 token 经验分布双测最大偏差',
      Math.max(devA, devB), 0, MC_TOL,
      'spec=' + devA.toFixed(5) + ' tgt=' + devB.toFixed(5) + ' N=' + n + ' MCSE≈0.0035(8σ≈0.028)'));
  }

  // INV-SPC-002 贪心路径全接受：draft≡target / argmax 保持扰动 均全接受 k；
  // argmax 在位置 2 分歧时接受前缀恰为 2（精确，tol 0）。ctx0=[] 使表行号==token 位置号。
  {
    const ctx0 = [];
    const k = 4;
    let bad = 0;
    { // A: draft≡target → argmax 必然一致
      const draft = proposeTokens(targetFn, ctx0, k, { greedy: true, rng: makeRng(0x6EED2A2) });
      if (greedyVerifyChunk(targetFn, ctx0, draft) !== k) bad++;
    }
    { // B: draft=target+保持 argmax 的小扰动 → 仍全接受
      const perturbed = _tableModel([
        Float64Array.from([2.3, 1.0, 0.2, -0.4, -1.2]),
        Float64Array.from([0.6, 2.4, 0.5, -0.3, -0.9]),
        Float64Array.from([1.6, 0.4, 2.2, -0.6, -1.0])
      ]);
      const draft = proposeTokens(perturbed, ctx0, k, { greedy: true, rng: makeRng(0x6EED2B2) });
      if (greedyVerifyChunk(targetFn, ctx0, draft) !== k) bad++;
    }
    { // C: 位置 2 的 argmax 从 2 改到 0 → 接受前缀恰为 2
      const diverged = _tableModel([
        Float64Array.from([2.0, 1.2, 0.4, -0.3, -1.1]),
        Float64Array.from([0.9, 2.1, 0.7, -0.2, -0.8]),
        Float64Array.from([2.6, 0.6, 1.9, -0.5, -0.9])
      ]);
      const draft = proposeTokens(diverged, ctx0, k, { greedy: true, rng: makeRng(0x6EED2C2) });
      if (greedyVerifyChunk(targetFn, ctx0, draft) !== 2) bad++;
    }
    cases.push(caseResult('INV-SPC-002', '贪心路径：argmax 一致全接受 k / 分歧位置前缀=j（违例计数）',
      bad, 0, 0, 'identical→4, perturbed→4, diverged@2→2'));
  }

  // INV-SPC-003 每步产出上界：20 组随机场景，1 ≤ tokensAccepted+bonus ≤ k+1，
  // 且 ctx' 长度=ctx+产出、tokensAccepted 为草稿前缀、targetCalls=1/draftCalls=k（违例计数=0）。
  {
    const rng = makeRng(0x7EED3A3);
    let bad = 0;
    for (let g = 0; g < 20; g++) {
      const vocab = 4 + (g % 3);
      const target = _randomTableModel(rng, 3, vocab);
      const draft = _randomTableModel(rng, 3, vocab);
      const ctxLen = 1 + Math.floor(rng() * 3);
      const ctx = new Array(ctxLen);
      for (let i = 0; i < ctxLen; i++) ctx[i] = Math.floor(rng() * vocab);
      const k = 1 + Math.floor(rng() * 6);
      const T = 0.6 + rng() * 0.8;
      const stepSeed = 0xA11CE00 + g;
      const st = speculativeStep(draft, target, ctx, k, { rng: makeRng(stepSeed), temperature: T });
      const draftObj = proposeTokens(draft, ctx, k, { rng: makeRng(stepSeed), temperature: T }); // 同种子复现草稿
      const produced = st.tokensAccepted.length + (st.bonus === null || st.bonus === undefined ? 0 : 1);
      if (!(produced >= 1 && produced <= k + 1)) bad++;
      if (st.ctx.length !== ctx.length + produced) bad++;
      if (!_isPrefix(ctx, st.ctx)) bad++;
      if (!_isPrefix(st.tokensAccepted, draftObj.tokens)) bad++;
      if (st.targetCalls !== 1 || st.draftCalls !== k) bad++;
      if (st.bonus !== st.ctx[st.ctx.length - 1]) bad++;
    }
    cases.push(caseResult('INV-SPC-003', '每步产出 1..k+1 且结构一致（20 组随机场景）',
      bad, 0, 0, 'k∈[1,6] vocab∈[4,6] T∈[0.6,1.4]'));
  }

  // INV-SPC-004 拒绝采样概率正确：固定 q/p 单 token 重复 N=20000 次，经验频率与 q 最大偏差 ≤ 0.03。
  {
    const q = S().softmaxTemperature([2.0, 1.0, 0.5, 0.0, -0.5], 1);
    const p = S().softmaxTemperature([0.5, 1.5, 1.0, 0.5, -1.0], 1);
    const rng = makeRng(0x7EED4A4);
    const draws = new Array(MC_N);
    for (let i = 0; i < MC_N; i++) draws[i] = rejectionSample(q, p, rng).token;
    const freq = S().empiricalFrequencies(draws, 5);
    let dev = 0;
    for (let v = 0; v < 5; v++) dev = Math.max(dev, Math.abs(freq[v] - q[v]));
    cases.push(caseResult('INV-SPC-004', '拒绝采样无偏：N=20000 经验频率 vs q 最大偏差',
      dev, 0, MC_TOL, 'dev=' + dev.toFixed(5) + ' 恒等式 min(p,q)+(q−p)₊=q'));
  }

  // INV-SPC-005 接受率单调：draft==target 时接受率==1（精确）；
  // 3 档差异增大（解析期望接受率 Σmin(p,q)≈0.91/0.74/0.48，间距≫MCSE）经验接受率单调不增（违例计数=0）。
  {
    const same = speculativeDecode(targetFn, targetFn, [0], { k: 4, maxTokens: 40, rng: makeRng(0x7EED5B5) });
    const rateSame = acceptanceRate(same);
    const q = S().softmaxTemperature([2.0, 1.4, 1.0, 0.2, -0.6], 1);
    const pRows = [
      [1.7, 1.4, 1.1, 0.4, -0.4], // 轻差异
      [1.0, 1.2, 1.3, 0.8, 0.2], // 中差异
      [0.2, 0.4, 0.8, 1.6, 1.2]  // 强差异
    ];
    const rates = [], analytic = [];
    for (let li = 0; li < 3; li++) {
      const p = S().softmaxTemperature(pRows[li], 1);
      const rng = makeRng(0x7EED5A5 + li * 977);
      let acc = 0;
      for (let i = 0; i < MC_N; i++) if (rejectionSample(q, p, rng).accepted) acc++;
      rates.push(acc / MC_N);
      let s = 0;
      for (let v = 0; v < 5; v++) s += Math.min(p[v], q[v]);
      analytic.push(s);
    }
    let bad = 0;
    if (rateSame !== 1) bad++;
    if (!(rates[0] >= rates[1])) bad++;
    if (!(rates[1] >= rates[2])) bad++;
    cases.push(caseResult('INV-SPC-005', '接受率：draft≡target 精确 1；3 档差异单调不增',
      bad, 0, 0,
      'same=' + rateSame + ' rates=[' + rates.map(r => r.toFixed(4)).join(',') +
      '] analytic=[' + analytic.map(r => r.toFixed(4)).join(',') + ']'));
  }

  // INV-SPC-006 单 token 退化：k=1 时 speculativeDecode 与 target-only 独立重实现在 L=3 个位置的
  // 经验分布各自与目标分布一致（≤0.03）。k=1 拒绝采样边际=目标分布（对位置归纳成立）。
  // ctx0=[] 使表行号==输出位置号（qt=softmax(Q_TABLE[t])）。
  {
    const ctx0 = [];
    const L = 3, n = MC_N;
    const specCounts = new Float64Array(V_ * L);
    const tgtCounts = new Float64Array(V_ * L);
    for (let i = 0; i < n; i++) {
      const rng = makeRng(0x7EED6A6 + i);
      const out = speculativeDecode(draftFn, targetFn, ctx0, { k: 1, maxTokens: L, rng });
      for (let t = 0; t < L; t++) specCounts[t * V_ + out.tokens[t]] += 1;
    }
    for (let i = 0; i < n; i++) {
      const rng = makeRng(0x7EED6A6 + i);
      const cur = ctx0.slice();
      for (let t = 0; t < L; t++) {
        const tok = S().sample(targetFn(cur), { temperature: 1, rng });
        tgtCounts[t * V_ + tok] += 1;
        cur.push(tok);
      }
    }
    let dev = 0;
    for (let t = 0; t < L; t++) {
      const qt = S().softmaxTemperature(Q_TABLE[t % Q_TABLE.length], 1);
      for (let v = 0; v < V_; v++) {
        dev = Math.max(dev, Math.abs(specCounts[t * V_ + v] / n - qt[v]));
        dev = Math.max(dev, Math.abs(tgtCounts[t * V_ + v] / n - qt[v]));
      }
    }
    cases.push(caseResult('INV-SPC-006', 'k=1 退化：3 位置经验分布 vs 目标分布最大偏差',
      dev, 0, MC_TOL, 'dev=' + dev.toFixed(5) + ' L=' + L + ' N=' + n + ' 双侧(spec/target-only)'));
  }

  // INV-SPC-007 目标调用次数有界：生成 n token，targetCalls ≤ n 且 ≤ draftCalls/k+n；
  // draft==target 时 targetCalls==ceil(n/(k+1))（精确，每步恰好 k+1 token，证明确有加速）。
  {
    let bad = 0;
    const notes = [];
    { // A: draft≡target, k=4, n=20 → ceil(20/5)=4 趟目标调用（加速比 5×）
      const n = 20, k = 4;
      const out = speculativeDecode(targetFn, targetFn, [0], { k, maxTokens: n, rng: makeRng(0x7EED7A7) });
      notes.push('same n=20 k=4 tc=' + out.targetCalls + ' dc=' + out.draftCalls + ' (期望 ceil=4)');
      if (out.targetCalls !== Math.ceil(n / (k + 1))) bad++;
      if (!(out.targetCalls <= n)) bad++;
      if (!(out.targetCalls <= out.draftCalls / k + n)) bad++;
      if (!(out.targetCalls < n)) bad++;
    }
    { // B: draft≡target, k=4, n=23 → ceil(23/5)=5（末趟截断）
      const n = 23, k = 4;
      const out = speculativeDecode(targetFn, targetFn, [0], { k, maxTokens: n, rng: makeRng(0x7EED7B7) });
      notes.push('same n=23 tc=' + out.targetCalls + ' (期望 ceil=5)');
      if (out.targetCalls !== Math.ceil(n / (k + 1))) bad++;
    }
    { // C: 弱草稿 draft≠target, k=3, n=18 → 仅 bound 断言（每步≥1 token ⇒ tc≤n）
      const n = 18, k = 3;
      const out = speculativeDecode(draftFn, targetFn, [0], { k, maxTokens: n, rng: makeRng(0x7EED7C7) });
      notes.push('weak n=18 k=3 tc=' + out.targetCalls + ' dc=' + out.draftCalls);
      if (!(out.targetCalls <= n)) bad++;
      if (!(out.targetCalls <= out.draftCalls / k + n)) bad++;
      if (out.tokens.length !== n) bad++;
    }
    cases.push(caseResult('INV-SPC-007', '目标调用有界 + draft≡target 精确 ceil(n/(k+1))',
      bad, 0, 0, notes.join(' | ')));
  }

  // INV-SPC-008 确定性：同种子两次 speculativeDecode 输出 token 序列与统计量逐位相等（tol 0）。
  {
    const a = speculativeDecode(draftFn, targetFn, [0], { k: 4, maxTokens: 24, rng: makeRng(0x7EED8A8) });
    const b = speculativeDecode(draftFn, targetFn, [0], { k: 4, maxTokens: 24, rng: makeRng(0x7EED8A8) });
    let bad = 0;
    if (a.tokens.length !== b.tokens.length) bad++;
    else for (let i = 0; i < a.tokens.length; i++) if (a.tokens[i] !== b.tokens[i]) bad++;
    if (a.steps !== b.steps || a.targetCalls !== b.targetCalls || a.draftCalls !== b.draftCalls ||
        a.acceptedDraft !== b.acceptedDraft || a.proposedDraft !== b.proposedDraft) bad++;
    cases.push(caseResult('INV-SPC-008', '确定性：同种子两次 speculativeDecode 逐位相等',
      bad, 0, 0, 'len=' + a.tokens.length + ' steps=' + a.steps));
  }

  // 不变量登记（8 条，worst 取实测绝对偏差）
  const INV_EXPR = {
    'INV-SPC-001': '分布等价：推测解码与 target-only 的首 token 经验分布双测偏差 ≤ 0.03（N=20000）',
    'INV-SPC-002': '贪心路径 argmax 一致时全接受 k，分歧位置 j 时前缀=j（违例=0）',
    'INV-SPC-003': '每步产出 1 ≤ tokensAccepted+bonus ≤ k+1 且结构一致（20 组随机，违例=0）',
    'INV-SPC-004': '拒绝采样无偏：N=20000 经验频率与 q 偏差 ≤ 0.03（min(p,q)+(q−p)₊=q）',
    'INV-SPC-005': '接受率：draft≡target 精确 1；3 档差异单调不增（违例=0）',
    'INV-SPC-006': 'k=1 退化：3 位置经验分布与目标分布一致（≤0.03，N=20000）',
    'INV-SPC-007': '目标调用有界：targetCalls ≤ n 且 ≤ draftCalls/k+n；draft≡target 时 =ceil(n/(k+1))',
    'INV-SPC-008': '确定性：同种子两次 speculativeDecode 输出逐位相等（违例=0）'
  };
  for (const id of Object.keys(INV_EXPR)) {
    const c = cases.find(x => x.id === id);
    invariants.push({ id, expr: INV_EXPR[id], tol: c.tol, worst: c.abs_err, status: c.status, note: '' });
  }

  const started = Date.now();
  return buildReport({
    module: 'speculative-forge', version: '1.0.0', layer: 'L4', cases, invariants,
    seed: '0x9E3779B9', started,
    tol: {
      'INV-SPC-001': MC_TOL, 'INV-SPC-002': 0, 'INV-SPC-003': 0, 'INV-SPC-004': MC_TOL,
      'INV-SPC-005': 0, 'INV-SPC-006': MC_TOL, 'INV-SPC-007': 0, 'INV-SPC-008': 0
    }
  });
}

const API = {
  name: 'speculative-forge', version: '1.0.0', layer: 'L4',
  proposeTokens, rejectionSample, verifyChunk, speculativeStep, speculativeDecode,
  acceptanceRate, greedyVerifyChunk, selfCheck
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeSpeculative = API;

if (typeof process !== 'undefined' && process.argv[1] && /speculative-forge\.js$/.test(process.argv[1]) &&
    (process.argv.includes('--self-check') || process.argv.includes('--json'))) {
  const r = selfCheck({ quick: process.argv.includes('--quick') });
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : V().formatTableSafe(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
