/**
 * @forge-module generation-forge
 * @layer L4
 * @version 1.0.0
 * @brief     端到端自回归生成闭环（集成封顶模块）：KV 缓存增量解码主循环 / 流式 token 回调 / 共享前缀缓存池 /
 *            批量生成（连续批处理）/ 停止条件（EOS / maxTokens / 自定义 stopIds）/ 无缓存全序列前向黄金参照。
 *            把 kv-cache-forge、sampling-forge、attention-forge、transformer-forge 四大内核串成真实可跑的生成链路；
 *            模型以可注入 logits 函数抽象 (ctx, cache) -> {logits}，内置确定性「查表 + 位置混合」测试模型，
 *            使增量 / 全量 / 流式 / 批量四条生成路径可逐位交叉验证。8 条硬不变量。
 * @reuse     HuggingFace transformers generate|https://github.com/huggingface/transformers|Apache-2.0|C|生成循环/停止条件/EOS 接口契约
 * @reuse     vLLM (Kwon et al. 2023)|https://arxiv.org/abs/2309.06180|Apache-2.0|B|连续批处理与前缀/块共享
 * @reuse     llama.cpp|https://github.com/ggerganov/llama.cpp|MIT|B|增量解码与流式生成循环
 * @reuse     OpenAI API streaming|https://platform.openai.com/docs|MIT|D|流式增量交付语义
 * @reuse     kv-cache-forge (本仓库)|src/kv-cache-forge.js|MIT|X|createKVCache/appendKV/incrementalDecode/cacheStats 原样复用
 * @reuse     sampling-forge (本仓库)|src/sampling-forge.js|MIT|X|sample/greedy 采样管线原样复用
 * @reuse     attention-forge (本仓库)|src/attention-forge.js|MIT|X|sdpa/splitHeads/mergeHeads 原样复用
 * @reuse     transformer-forge (本仓库)|src/transformer-forge.js|MIT|X|sinusoidalPositionalEncoding 原式复用
 * @beyond    纯 JS 单文件零构建双端同构；可注入确定性查表模型使四条生成路径逐位等价可证；前缀缓存池 fork 语义
 *            （save 深快照 / get 返回独立副本）零污染；EOS 命中即停且不写入输出；批量生成按 prompt 独立派生种子流，
 *            与逐条 generate 逐位一致；增量路径每步 logits 与全序列前向黄金参照逐元素一致。
 * @invariant INV-GEN-001|流式等价批量：同模型同 prompt 同种子，generateStream 回调 token 序列与 generate 返回 tokens 逐位一致（回调抛错向上传播）|0|双路径逐元素比对
 * @invariant INV-GEN-002|增量等价全量（核心）：generate（KV 缓存增量路径）每步 logits 与 generateNoCache（每步全序列前向）对应位置逐元素一致 ≤ 1e-12|1e-12|黄金参照逐元素比对
 * @invariant INV-GEN-003|前缀缓存等价：共享长度 L 前缀的 prompts 经 batchGenerate 复用前缀缓存的结果，与各自独立 generate 逐位一致（池命中计数防空洞化）|0|批量 vs 独立逐位比对
 * @invariant INV-GEN-004|停止条件精确：eosId 为模型第 3 步（0 基）必产 token 时输出长度==3 且不含 eosId；maxTokens=5 时长度<=5；两者同时设置取先到者|0|精确整数断言
 * @invariant INV-GEN-005|确定性：同种子两次 generate 输出 tokens 逐位相等；不同种子贪心模式仍逐位相等；采样模式跨种子须有差异（防空洞化）|0|逐位比对
 * @invariant INV-GEN-006|与采样内核一致：generate（采样模式）产出的 token 序列 == sampling-forge 的 sample 对同一条 logits 序列逐次采样的结果（rng 种子派生一致）|0|独立重放逐位比对
 * @invariant INV-GEN-007|贪心确定性路径：temperature=1e-8、topK=1 时 generate 输出 == 每步 argmax 构成的确定性序列|0|独立 argmax 重实现逐位比对
 * @invariant INV-GEN-008|不良输入抛错：prompt 为空数组 / maxTokens=0 / eosId 为负数 / 缺 sampling 且缺 rng（贪心除外）等，generate 抛错且合法输入不抛|0|异常契约漏抛计数
 * @seed      0x9E3779B9
 */
'use strict';

/* 惰性双端依赖：Node require / 浏览器 globalThis 回退；加载期不强依赖，零循环导入。 */
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
let _KVC = null;
function KVC() {
  if (_KVC) return _KVC;
  if (typeof require !== 'undefined') { try { _KVC = require('./kv-cache-forge.js'); } catch (e) {} }
  if (!_KVC && typeof globalThis !== 'undefined') _KVC = globalThis.ForgeKVCache || null;
  if (!_KVC) throw new Error('kv-cache-forge 依赖缺失');
  return _KVC;
}
let _SMP = null;
function SMP() {
  if (_SMP) return _SMP;
  if (typeof require !== 'undefined') { try { _SMP = require('./sampling-forge.js'); } catch (e) {} }
  if (!_SMP && typeof globalThis !== 'undefined') _SMP = globalThis.ForgeSampling || null;
  if (!_SMP) throw new Error('sampling-forge 依赖缺失');
  return _SMP;
}
let _TR = null;
function TR() {
  if (_TR) return _TR;
  if (typeof require !== 'undefined') { try { _TR = require('./transformer-forge.js'); } catch (e) {} }
  if (!_TR && typeof globalThis !== 'undefined') _TR = globalThis.ForgeTransformer || null;
  if (!_TR) throw new Error('transformer-forge 依赖缺失');
  return _TR;
}
let _A = null;
function A() {
  if (_A) return _A;
  if (typeof require !== 'undefined') { try { _A = require('./attention-forge.js'); } catch (e) {} }
  if (!_A && typeof globalThis !== 'undefined') _A = globalThis.ForgeAttention || null;
  if (!_A) throw new Error('attention-forge 依赖缺失');
  return _A;
}

const MODULE_SEED = 0x9E3779B9;

/* ---------- 入参校验（异常契约：不良输入在生成开始前抛错） ---------- */

/** 校验模型：必须为函数 (ctx, cache) -> {logits}。 */
function validateModel(model) {
  if (typeof model !== 'function') {
    throw new TypeError('[generation-forge] model 必须为函数 (ctx, cache) -> {logits}');
  }
}

/** 校验 prompt：非空 token 数组，元素为非负整数。 */
function validatePrompt(prompt) {
  if (!Array.isArray(prompt) || prompt.length === 0) {
    throw new TypeError('[generation-forge] prompt 必须为非空 token 数组');
  }
  for (let i = 0; i < prompt.length; i++) {
    const t = prompt[i];
    if (!Number.isInteger(t) || t < 0) {
      throw new RangeError('[generation-forge] prompt[' + i + ']=' + t + ' 须为非负整数');
    }
  }
}

/** 校验 opts：maxTokens 为正整数；eosId/stopIds 若提供须为非负整数。 */
function validateOpts(opts) {
  if (!Number.isInteger(opts.maxTokens) || opts.maxTokens < 1) {
    throw new RangeError('[generation-forge] maxTokens 须为正整数，收到 ' + opts.maxTokens);
  }
  if (opts.eosId !== undefined && opts.eosId !== null) {
    if (!Number.isInteger(opts.eosId) || opts.eosId < 0) {
      throw new RangeError('[generation-forge] eosId 须为非负整数，收到 ' + opts.eosId);
    }
  }
  if (opts.stopIds !== undefined && opts.stopIds !== null) {
    if (!Array.isArray(opts.stopIds)) throw new TypeError('[generation-forge] stopIds 必须为数组');
    for (const s of opts.stopIds) {
      if (!Number.isInteger(s) || s < 0) {
        throw new RangeError('[generation-forge] stopIds 元素须为非负整数，收到 ' + s);
      }
    }
  }
}

/** 贪心模式判定：opts.greedy 或 opts.sampling.greedy 为 true。 */
function isGreedy(opts) {
  return opts.greedy === true || !!(opts.sampling && opts.sampling.greedy === true);
}

/**
 * 归一化采样配置：{greedy, temperature, topK, topP, penalty}。
 * topK/topP 缺省为 null（sampling-forge 的 sample 对 null 自动跳过对应过滤）。
 */
function normalizeSampling(opts) {
  const s = opts.sampling || {};
  const temperature = (s.temperature === undefined || s.temperature === null) ? 1.0 : s.temperature;
  if (typeof temperature !== 'number' || !isFinite(temperature) || temperature <= 0) {
    throw new RangeError('[generation-forge] sampling.temperature 须为有限正数，收到 ' + temperature);
  }
  return {
    greedy: isGreedy(opts),
    temperature,
    topK: (s.topK === undefined || s.topK === null) ? null : s.topK,
    topP: (s.topP === undefined || s.topP === null) ? null : s.topP,
    penalty: (s.penalty === undefined || s.penalty === null) ? null : s.penalty
  };
}

/**
 * 解析随机源：opts.rng 直传（批量下各 prompt 共享同一条连续流）；否则由 opts.seed 派生
 * （每个 generate 调用 / 批量中每个 prompt 独立派生同名种子流）；贪心模式返回占位流（不消耗）。
 * 非贪心且两者皆缺 → 抛错。
 */
function resolveRng(opts) {
  if (opts.rng !== undefined && opts.rng !== null) {
    if (typeof opts.rng !== 'function') throw new TypeError('[generation-forge] opts.rng 必须为函数');
    return opts.rng;
  }
  if (opts.seed !== undefined && opts.seed !== null) {
    if (!Number.isInteger(opts.seed)) throw new RangeError('[generation-forge] opts.seed 须为整数');
    return N().makeRng(opts.seed);
  }
  if (isGreedy(opts)) return N().makeRng(MODULE_SEED); // 贪心无随机性，占位流不会被消耗
  throw new Error('[generation-forge] 非贪心模式必须提供 opts.rng 或 opts.seed');
}

/**
 * 解析 KV 缓存规格：优先 model.cacheSpec，其次 opts.cacheSpec（须含 nLayers/nHeads/headDim）。
 * maxLen 取声明值与 neededLen（prompt 长度 + maxTokens）的较大者，保证增量写不越界。
 */
function resolveCacheSpec(model, opts, neededLen) {
  const base = (model && model.cacheSpec) || opts.cacheSpec || null;
  if (!base) {
    throw new Error('[generation-forge] 需要 model.cacheSpec 或 opts.cacheSpec（nLayers/nHeads/headDim）');
  }
  return {
    nLayers: base.nLayers,
    nHeads: base.nHeads,
    headDim: base.headDim,
    blockSize: (base.blockSize === undefined || base.blockSize === null) ? 16 : base.blockSize,
    maxLen: Math.max(base.maxLen || 0, neededLen)
  };
}

/** 采样下一个 token：贪心走 sampling-forge.greedy，否则走完整 sample 管线（温度/top-k/top-p/惩罚）。 */
function sampleNext(logits, sampling, rng) {
  if (sampling.greedy) return SMP().greedy(logits);
  return SMP().sample(logits, {
    temperature: sampling.temperature, topK: sampling.topK, topP: sampling.topP,
    penalty: sampling.penalty, rng
  });
}

/* ---------- 停止条件 ---------- */

/**
 * 停止条件判定（EOS / 自定义 stopIds / maxTokens 截断，按此优先级）。
 * @param {number} token 本轮采样得到的 token
 * @param {object} opts {eosId=null, stopIds=[], maxTokens=Infinity, steps=0}
 *        steps = 本次采样前已提交输出的 token 数（不含当前 token）
 * @returns {{stop:boolean, reason:('eos'|'stopId'|'maxTokens'|null)}}
 *          命中 eos/stopId 时调用方不应把该 token 写入输出（本模块语义）。
 */
function stopCheck(token, opts) {
  opts = opts || {};
  const eosId = (opts.eosId === undefined || opts.eosId === null) ? null : opts.eosId;
  const stopIds = opts.stopIds || [];
  const maxTokens = (opts.maxTokens === undefined || opts.maxTokens === null) ? Infinity : opts.maxTokens;
  const steps = opts.steps || 0;
  if (eosId !== null && token === eosId) return { stop: true, reason: 'eos' };
  for (let i = 0; i < stopIds.length; i++) {
    if (token === stopIds[i]) return { stop: true, reason: 'stopId' };
  }
  if (steps + 1 >= maxTokens) return { stop: true, reason: 'maxTokens' };
  return { stop: false, reason: null };
}

/* ---------- 核心生成循环（generate / generateStream / batchGenerate / generateNoCache 共用） ----------
 * cache 非空 → KV 缓存增量路径：model(ctx, cache) 自动预填新 token 的 K/V 并增量解码；
 * cache 为空 → 全序列前向路径：model(ctx, null) 每步对全序列做因果前向（黄金参照）。 */
function runGeneration(model, prompt, opts, rng, cache, hooks) {
  const sampling = normalizeSampling(opts);
  const eosId = (opts.eosId === undefined || opts.eosId === null) ? null : opts.eosId;
  const stopIds = opts.stopIds || [];
  const ctx = Array.prototype.slice.call(prompt); // 可增长上下文（prompt 拷贝，不修改入参）
  const tokens = [];
  while (tokens.length < opts.maxTokens) {
    const res = model(ctx, cache);
    if (!res || !res.logits || typeof res.logits.length !== 'number') {
      throw new Error('[generation-forge] model 须返回 {logits: 非空向量}');
    }
    const logits = res.logits;
    if (hooks && hooks.onLogits) hooks.onLogits(logits, tokens.length); // 观测钩子（抛错向上传播）
    const next = sampleNext(logits, sampling, rng);
    const decision = stopCheck(next, { eosId, stopIds, maxTokens: opts.maxTokens, steps: tokens.length });
    if (decision.stop && (decision.reason === 'eos' || decision.reason === 'stopId')) break; // 不写入输出
    ctx.push(next);
    tokens.push(next);
    if (hooks && hooks.onToken) hooks.onToken(next, tokens.length - 1); // 流式回调（抛错向上传播）
    if (decision.stop && decision.reason === 'maxTokens') break;
  }
  return { tokens, steps: tokens.length, cacheStats: cache ? KVC().cacheStats(cache) : null };
}

/* ---------- 公开生成 API ---------- */

/**
 * 完整生成闭环：KV 缓存预填 prompt → 循环增量解码取 logits → 采样下一 token → 写入缓存，
 * 遇 EOS / stopIds 停（不写入）或 maxTokens 截断。
 * @param {Function} model (ctx, cache) -> {logits}；可暴露 model.cacheSpec = {nLayers,nHeads,headDim[,maxLen,blockSize]}
 * @param {Array<number>} prompt 非空非负整数 token 序列
 * @param {object} [opts] {maxTokens, eosId, stopIds, sampling:{temperature,topK,topP,penalty,greedy},
 *                        rng, seed, greedy, useCache=true, cacheSpec, onToken, onLogits}
 * @returns {{tokens:Array<number>, steps:number, cacheStats:object|null}}
 *          steps = tokens.length；cacheStats = kv-cache-forge.cacheStats 快照（无缓存路径为 null）
 */
function generate(model, prompt, opts) {
  opts = opts || {};
  validateModel(model);
  validatePrompt(prompt);
  validateOpts(opts);
  normalizeSampling(opts); // 提前校验温度等配置
  const rng = resolveRng(opts);
  const useCache = opts.useCache !== false;
  const cache = useCache
    ? KVC().createKVCache(resolveCacheSpec(model, opts, prompt.length + opts.maxTokens))
    : null;
  return runGeneration(model, prompt, opts, rng, cache, { onToken: opts.onToken, onLogits: opts.onLogits });
}

/**
 * 流式生成：与 generate 完全同构，每产出一个 token 回调 onToken(token, step) 一次（step 为 0 基序号）。
 * 回调抛错直接向上传播（不做捕获）。返回结构与 generate 相同。
 */
function generateStream(model, prompt, opts, onToken) {
  opts = opts || {};
  if (onToken !== undefined && onToken !== null && typeof onToken !== 'function') {
    throw new TypeError('[generation-forge] onToken 必须为函数');
  }
  const merged = Object.assign({}, opts, { onToken: onToken || opts.onToken });
  return generate(model, prompt, merged);
}

/**
 * 无 KV 缓存的对照实现（黄金参照）：每步对全序列做因果前向（model(ctx, null)）。
 * 用于验证 INV-GEN-002：增量路径每步 logits 与逐元素一致 ≤ 1e-12。
 */
function generateNoCache(model, prompt, opts) {
  opts = opts || {};
  return generate(model, prompt, Object.assign({}, opts, { useCache: false }));
}

/** 最长公共前缀长度（批量生成共享前缀的判定）。 */
function longestCommonPrefix(prompts) {
  let L = prompts[0].length;
  for (let i = 1; i < prompts.length; i++) {
    const p = prompts[i];
    let j = 0;
    while (j < L && j < p.length && p[j] === prompts[0][j]) j++;
    L = j;
  }
  return L;
}

/** prompt 的稳定缓存键：core-numeric 纯 JS SHA-256（双端逐位一致）。 */
function stableKey(tokens) {
  return N().sha256('generation-forge/prefix/v1|' + tokens.length + '|' + Array.prototype.join.call(tokens, ','));
}

/* ---------- 前缀缓存池（fork 语义：save 深快照 / get 返回独立副本，消费者追加不污染池） ---------- */

/** 深快照：按层读出 [0, len) 的 K/V 行（getKV 拷贝语义）。 */
function snapshotCache(cache) {
  const layers = new Array(cache.nLayers);
  for (let l = 0; l < cache.nLayers; l++) {
    const upto = cache.len[l];
    layers[l] = upto > 0 ? KVC().getKV(cache, l, upto) : { K: [], V: [] };
  }
  return {
    nLayers: cache.nLayers, nHeads: cache.nHeads, headDim: cache.headDim,
    maxLen: cache.maxLen, blockSize: cache.blockSize, layers
  };
}

/** 由快照恢复独立缓存对象（新建缓冲 + 逐层预填）。 */
function restoreCache(snap) {
  const cache = KVC().createKVCache({
    nLayers: snap.nLayers, nHeads: snap.nHeads, headDim: snap.headDim,
    maxLen: snap.maxLen, blockSize: snap.blockSize
  });
  for (let l = 0; l < snap.nLayers; l++) {
    if (snap.layers[l].K.length > 0) KVC().prefillKV(cache, l, snap.layers[l].K, snap.layers[l].V);
  }
  return cache;
}

/**
 * 创建前缀缓存池。key 通常为 prompt 的稳定哈希（stableKey）。
 * @returns {{save:Function, get:Function, stats:Function}}
 *          save(key, cache) 深快照入库并返回 key；get(key) 返回独立副本（未命中返回 null）；
 *          stats() -> {entries, hits, misses, saves}
 */
function prefixCache() {
  const store = new Map();
  let hits = 0, misses = 0, saves = 0;
  return {
    save(key, cache) {
      if (typeof key !== 'string' || key.length === 0) {
        throw new TypeError('[generation-forge] prefixCache key 须为非空字符串');
      }
      if (!cache || !(cache.kBuf instanceof Float64Array) || !(cache.vBuf instanceof Float64Array)) {
        throw new TypeError('[generation-forge] prefixCache.save 须传入 createKVCache 创建的缓存');
      }
      store.set(key, snapshotCache(cache));
      saves++;
      return key;
    },
    get(key) {
      const snap = store.get(key);
      if (!snap) { misses++; return null; }
      hits++;
      return restoreCache(snap);
    },
    stats() { return { entries: store.size, hits, misses, saves }; }
  };
}

/**
 * 批量生成（连续批处理 + 前缀共享）：prompts 若共享长度 L 的最长公共前缀，则预填一份前缀缓存入池，
 * 各 prompt 从池中 fork 独立副本后续写（互不污染）；无共享前缀时各自全新缓存。
 * rng 派生与 generate 一致（seed 模式下每个 prompt 独立同名种子流 → 与逐条 generate 逐位一致）。
 * @param {Function} model 同 generate
 * @param {Array<Array<number>>} prompts 非空的 prompt 数组
 * @param {object} [opts] 同 generate；另支持 opts.prefixCache 注入外部池（可观测复用统计）
 * @returns {Array<{tokens:Array<number>, steps:number, cacheStats:object|null}>} 与 prompts 等长的结果数组
 */
function batchGenerate(model, prompts, opts) {
  opts = opts || {};
  validateModel(model);
  if (!Array.isArray(prompts) || prompts.length === 0) {
    throw new TypeError('[generation-forge] prompts 必须为非空数组');
  }
  for (const p of prompts) validatePrompt(p);
  validateOpts(opts);
  normalizeSampling(opts);
  const useCache = opts.useCache !== false;
  const pool = opts.prefixCache || prefixCache();
  const L = useCache ? longestCommonPrefix(prompts) : 0;
  let maxPromptLen = 0;
  for (const p of prompts) maxPromptLen = Math.max(maxPromptLen, p.length);
  const spec = resolveCacheSpec(model, opts, maxPromptLen + opts.maxTokens);
  let prefixKey = null;
  if (L > 0) {
    prefixKey = stableKey(prompts[0].slice(0, L));
    if (pool.get(prefixKey) === null) {
      const shared = KVC().createKVCache(spec);
      model(prompts[0].slice(0, L), shared); // 仅预填共享前缀（logits 丢弃）
      pool.save(prefixKey, shared);
    }
  }
  return prompts.map((p) => {
    const rng = resolveRng(opts); // 每 prompt 独立派生（seed 模式与逐条 generate 同流）
    const cache = useCache ? (L > 0 ? pool.get(prefixKey) : KVC().createKVCache(spec)) : null;
    return runGeneration(model, p, opts, rng, cache, { onToken: opts.onToken, onLogits: opts.onLogits });
  });
}

/* ---------- 确定性测试模型（「查表 + 位置混合」） ----------
 * tokens[i] = E[ctx[i]] + PE[i]（正弦位置编码，transformer-forge 原式）；Q/K/V = tokens·Wq/Wk/Wv；
 * 逐头因果 SDPA（attention-forge）；attn·Wo；logits = h·Wout。全部参数由单一整数种子确定性生成。
 * 增量路径：model(ctx, cache) 自动预填新 token 的 K/V，对末位 query 走 kv-cache-forge 增量解码；
 * 全量路径：model(ctx, null) 对整条序列一次性逐头因果 SDPA。两路数学同构，可逐位对照。 */
function randMatrix(rng, r, c) {
  const m = new Array(r);
  for (let i = 0; i < r; i++) {
    const row = new Float64Array(c);
    for (let k = 0; k < c; k++) row[k] = rng() * 2 - 1;
    m[i] = row;
  }
  return m;
}

/**
 * 行投影（复用 attention-forge.linear 矩阵级实现，单行包装；累加序与逐元素口径逐位一致）：
 * y[o] = Σ_k x[k]·W[o][k]，W 为 out×in。
 */
function projectRow(W, x) {
  return A().linear([x], W)[0];
}

/**
 * 创建确定性查表测试模型。
 * @param {object} [opts] {vocab=12, d=8, nHeads=2, maxLen=256, seed=0x5EED1}
 * @returns {{model:Function, vocab:number, d:number, nHeads:number, headDim:number, maxLen:number, cacheSpec:object}}
 */
function createTableModel(opts) {
  opts = opts || {};
  const vocab = (opts.vocab === undefined || opts.vocab === null) ? 12 : opts.vocab;
  const d = (opts.d === undefined || opts.d === null) ? 8 : opts.d;
  const nHeads = (opts.nHeads === undefined || opts.nHeads === null) ? 2 : opts.nHeads;
  const maxLen = (opts.maxLen === undefined || opts.maxLen === null) ? 256 : opts.maxLen;
  const seed = (opts.seed === undefined || opts.seed === null) ? 0x5EED1 : opts.seed;
  if (!Number.isInteger(vocab) || vocab < 1 || !Number.isInteger(d) || d < 2 ||
      !Number.isInteger(nHeads) || nHeads < 1 || d % nHeads !== 0) {
    throw new RangeError('[generation-forge] createTableModel 须满足 vocab>=1, d>=2, nHeads>=1 且 d%nHeads==0');
  }
  const headDim = d / nHeads;
  const rng = N().makeRng(seed);
  const E = randMatrix(rng, vocab, d);    // 嵌入表（查表）
  const Wq = randMatrix(rng, d, d);
  const Wk = randMatrix(rng, d, d);
  const Wv = randMatrix(rng, d, d);
  const Wo = randMatrix(rng, d, d);
  const Wout = randMatrix(rng, vocab, d); // 词表投影
  const PE = TR().sinusoidalPositionalEncoding(maxLen, d); // 位置混合

  /** token 行：嵌入 + 正弦位置编码（增量 / 全量两路共用，保证逐位一致）。 */
  function tokenRow(token, pos) {
    const row = new Float64Array(d);
    for (let k = 0; k < d; k++) row[k] = E[token][k] + PE[pos][k];
    return row;
  }

  /** 全序列前向：返回末位 logits（长度 vocab）。 */
  function fullLogits(ctx) {
    const T = ctx.length;
    const Q = new Array(T), K = new Array(T), Vv = new Array(T);
    for (let t = 0; t < T; t++) {
      const x = tokenRow(ctx[t], t);
      Q[t] = projectRow(Wq, x);
      K[t] = projectRow(Wk, x);
      Vv[t] = projectRow(Wv, x);
    }
    const Qh = A().splitHeads(Q, nHeads), Kh = A().splitHeads(K, nHeads), Vh = A().splitHeads(Vv, nHeads);
    const heads = new Array(nHeads);
    for (let h = 0; h < nHeads; h++) heads[h] = A().sdpa(Qh[h], Kh[h], Vh[h], { causal: true }).out;
    const attnOut = A().mergeHeads(heads, nHeads);
    return projectRow(Wout, projectRow(Wo, attnOut[T - 1]));
  }

  const model = function (ctx, cache) {
    if (cache === undefined || cache === null) return { logits: fullLogits(ctx) };
    // 增量路径：自动预填 [cached, T) 新 token 的 K/V，再对末位 query 增量解码
    const layer = 0;
    const cached = cache.len[layer];
    for (let t = cached; t < ctx.length; t++) {
      const x = tokenRow(ctx[t], t);
      KVC().appendKV(cache, layer, t, projectRow(Wk, x), projectRow(Wv, x));
    }
    const x = tokenRow(ctx[ctx.length - 1], ctx.length - 1);
    const attnOut = KVC().incrementalDecode(cache, layer, projectRow(Wq, x), null);
    return { logits: projectRow(Wout, projectRow(Wo, attnOut)) };
  };
  const cacheSpec = { nLayers: 1, nHeads, headDim, maxLen, blockSize: 16 };
  model.cacheSpec = cacheSpec;
  model.vocab = vocab;
  model.d = d;
  return { model, vocab, d, nHeads, headDim, maxLen, cacheSpec };
}

/* ---------- selfCheck — 8 条硬不变量（每条可独立交叉验证） ---------- */
function selfCheck(opts) {
  opts = opts || {};
  const { caseResult, buildReport } = V();
  const { makeRng } = N();
  const cases = [];
  const invariants = [];

  // 共享确定性测试模型与 prompt（全部不变量复用同一模型，保证可对照）
  const T = createTableModel({ seed: 0x5EED1, vocab: 12, d: 8, nHeads: 2 });
  const model = T.model;
  const prompt = [1, 3, 5, 7];
  const freshCache = () => KVC().createKVCache({
    nLayers: T.cacheSpec.nLayers, nHeads: T.cacheSpec.nHeads,
    headDim: T.cacheSpec.headDim, maxLen: T.maxLen
  });
  const sameTokens = (a, b) => {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  };

  // INV-GEN-001 流式等价批量：generateStream 回调序列 == generate 返回 tokens（逐位）
  {
    const opts = { maxTokens: 7, sampling: { temperature: 0.9, topK: 5 }, seed: 1234 };
    const streamed = [];
    const cbSteps = [];
    const rs = generateStream(model, prompt, opts, (tok, step) => { streamed.push(tok); cbSteps.push(step); });
    const rb = generate(model, prompt, opts);
    let bad = 0;
    if (!sameTokens(streamed, rb.tokens)) bad++;
    if (!sameTokens(rs.tokens, rb.tokens)) bad++;
    let stepOk = 1; // 回调步号须为 0 基连续
    for (let i = 0; i < cbSteps.length; i++) if (cbSteps[i] !== i) stepOk = 0;
    let propagated = 0; // 回调抛错须向上传播
    try { generateStream(model, prompt, opts, () => { throw new Error('stream-cb-boom'); }); }
    catch (e) { propagated = 1; }
    const nonVac = (streamed.length === 7 && rs.steps === 7) ? 0 : 1;
    cases.push(caseResult('INV-GEN-001', '流式回调序列 == generate tokens（逐位+回调传播）',
      Math.max(bad, 1 - stepOk, 1 - propagated, nonVac), 0, 0));
  }

  // INV-GEN-002 增量等价全量（核心）：generate 每步 logits vs generateNoCache 每步 logits
  {
    const opts = { maxTokens: 6, sampling: { temperature: 0.85, topK: 4 }, seed: 24680 };
    const inc = [], full = [];
    const rc = generate(model, prompt, Object.assign({}, opts, {
      onLogits: (lg) => inc.push(Float64Array.from(lg))
    }));
    const rf = generateNoCache(model, prompt, Object.assign({}, opts, {
      onLogits: (lg) => full.push(Float64Array.from(lg))
    }));
    let dev = 0, maxAbs = 0;
    const lenOk = (inc.length === full.length && inc.length === rc.tokens.length) ? 0 : 1;
    for (let k = 0; k < Math.min(inc.length, full.length); k++) {
      if (inc[k].length !== full[k].length) { dev = Infinity; break; }
      for (let v = 0; v < inc[k].length; v++) {
        dev = Math.max(dev, Math.abs(inc[k][v] - full[k][v]));
        maxAbs = Math.max(maxAbs, Math.abs(inc[k][v]));
      }
    }
    const tokOk = sameTokens(rc.tokens, rf.tokens) ? 0 : 1; // logits 一致 ⇒ 采样一致
    const nonVac = (inc.length >= 3 && maxAbs > 1e-6) ? 0 : 1;
    cases.push(caseResult('INV-GEN-002', 'KV 缓存增量 logits == 全序列前向 logits（逐元素）',
      Math.max(dev, lenOk, tokOk, nonVac), 0, 1e-12));
  }

  // INV-GEN-003 前缀缓存等价：共享前缀批量生成 == 各自独立 generate（逐位）
  {
    const p1 = [1, 3, 5, 7, 2]; // 共享前缀 [1,3,5,7]（长度 4）
    const p2 = [1, 3, 5, 7, 9];
    const opts = { maxTokens: 6, sampling: { temperature: 0.9, topK: 4 }, seed: 13579 };
    const pool = prefixCache();
    const batch = batchGenerate(model, [p1, p2], Object.assign({}, opts, { prefixCache: pool }));
    const r1 = generate(model, p1, opts);
    const r2 = generate(model, p2, opts);
    let bad = 0;
    if (!sameTokens(batch[0].tokens, r1.tokens)) bad++;
    if (!sameTokens(batch[1].tokens, r2.tokens)) bad++;
    // 防空洞化：前缀缓存确被复用（1 条目、2 次命中）且两 prompt 确有分叉
    const st = pool.stats();
    const reuseOk = (st.entries === 1 && st.hits === 2 && p1[4] !== p2[4]) ? 0 : 1;
    cases.push(caseResult('INV-GEN-003', '共享前缀批量 == 各自独立 generate（逐位+池命中）',
      Math.max(bad, reuseOk), 0, 0));
  }

  // INV-GEN-004 停止条件精确：EOS（第 3 步 0 基必产）/ maxTokens / 同时设置取先到者 / stopIds
  {
    const greedy = { maxTokens: 12, sampling: { greedy: true } };
    const ref = generate(model, prompt, greedy).tokens; // 确定性参考序列
    const eosId = ref[3]; // 模型第 3 步（0 基）必然产出
    const rEos = generate(model, prompt, { maxTokens: 12, eosId, sampling: { greedy: true } });
    let hasEos = 0;
    for (const t of rEos.tokens) if (t === eosId) hasEos = 1;
    const lenOk = (rEos.tokens.length === 3 && sameTokens(rEos.tokens, ref.slice(0, 3))) ? 0 : 1;
    const rCap = generate(model, prompt, { maxTokens: 5, sampling: { greedy: true } });
    const capOk = (rCap.tokens.length === 5) ? 0 : 1;
    const rBoth = generate(model, prompt, { maxTokens: 5, eosId, sampling: { greedy: true } });
    const bothOk = (rBoth.tokens.length === 3) ? 0 : 1;  // eos 先到（第 3 步 < 5）
    const rBoth2 = generate(model, prompt, { maxTokens: 2, eosId, sampling: { greedy: true } });
    const both2Ok = (rBoth2.tokens.length === 2) ? 0 : 1; // maxTokens 先到
    const rStop = generate(model, prompt, { maxTokens: 12, stopIds: [ref[2]], sampling: { greedy: true } });
    const stopOk = (rStop.tokens.length === 2 && sameTokens(rStop.tokens, ref.slice(0, 2))) ? 0 : 1;
    cases.push(caseResult('INV-GEN-004', 'EOS/maxTokens/stopIds 停止条件精确（整数断言）',
      Math.max(lenOk, hasEos, capOk, bothOk, both2Ok, stopOk), 0, 0));
  }

  // INV-GEN-005 确定性：同种子逐位相等；贪心跨种子仍相等；采样跨种子须有差异（防空洞化）
  {
    const opts = { maxTokens: 6, sampling: { temperature: 0.9, topK: 4 }, seed: 555 };
    const a = generate(model, prompt, opts);
    const b = generate(model, prompt, opts);
    const detOk = sameTokens(a.tokens, b.tokens) ? 0 : 1;
    const g1 = generate(model, prompt, { maxTokens: 6, sampling: { greedy: true }, seed: 1 });
    const g2 = generate(model, prompt, { maxTokens: 6, sampling: { greedy: true }, seed: 2 });
    const greedyOk = sameTokens(g1.tokens, g2.tokens) ? 0 : 1;
    const s1 = generate(model, prompt, { maxTokens: 8, sampling: { temperature: 1.3 }, seed: 11 });
    const s2 = generate(model, prompt, { maxTokens: 8, sampling: { temperature: 1.3 }, seed: 22 });
    let diff = 0;
    for (let i = 0; i < Math.min(s1.tokens.length, s2.tokens.length); i++) {
      if (s1.tokens[i] !== s2.tokens[i]) diff++;
    }
    const rngDiffer = (makeRng(11)() !== makeRng(22)()) ? 0 : 1;
    cases.push(caseResult('INV-GEN-005', '同种子确定 + 贪心跨种子确定 + 采样跨种子有差异',
      Math.max(detOk, greedyOk, diff > 0 ? 0 : 1, rngDiffer), 0, 0));
  }

  // INV-GEN-006 与采样内核一致：generate 采样序列 == sample 对同一 logits 序列逐次采样
  {
    const opts = { maxTokens: 6, sampling: { temperature: 0.85, topK: 3 }, seed: 777 };
    const r = generate(model, prompt, opts);
    const { sample } = SMP();
    const rngRef = makeRng(777); // 与 generate 的 seed 派生一致（makeRng(seed)）
    const cache = freshCache();
    const ctx = prompt.slice();
    const ref = [];
    for (let k = 0; k < opts.maxTokens; k++) {
      const lg = model(ctx, cache).logits;
      const tok = sample(lg, { temperature: 0.85, topK: 3, rng: rngRef });
      ref.push(tok);
      ctx.push(tok);
    }
    cases.push(caseResult('INV-GEN-006', 'generate 采样序列 == sampling-forge sample 逐次重放',
      sameTokens(r.tokens, ref) ? 0 : 1, 0, 0));
  }

  // INV-GEN-007 贪心确定性路径：temperature=1e-8/topK=1 输出 == 独立逐步 argmax 序列
  {
    const opts = { maxTokens: 6, sampling: { temperature: 1e-8, topK: 1 }, seed: 999 };
    const r = generate(model, prompt, opts);
    const cache = freshCache();
    const ctx = prompt.slice();
    const ref = [];
    for (let k = 0; k < opts.maxTokens; k++) {
      const lg = model(ctx, cache).logits;
      let best = 0; // argmax，平局取小索引（与 sampling-forge topKFilter 平局约定一致）
      for (let i = 1; i < lg.length; i++) if (lg[i] > lg[best]) best = i;
      ref.push(best);
      ctx.push(best);
    }
    cases.push(caseResult('INV-GEN-007', 'temperature=1e-8/topK=1 == 逐步 argmax 序列',
      sameTokens(r.tokens, ref) ? 0 : 1, 0, 0));
  }

  // INV-GEN-008 不良输入抛错：空 prompt / maxTokens=0 / 负 eosId / 缺 sampling 且缺 rng（贪心除外）等
  {
    let missed = 0;
    const expectThrow = (fn) => { try { fn(); missed++; } catch (e) { /* 预期抛错 */ } };
    const okOpts = { maxTokens: 4, sampling: { temperature: 0.9 }, seed: 1 };
    expectThrow(() => generate(model, [], okOpts));                                          // 空 prompt
    expectThrow(() => generate(model, [1, 2], { maxTokens: 0, sampling: { temperature: 0.9 }, seed: 1 })); // maxTokens=0
    expectThrow(() => generate(model, [1, 2], { maxTokens: 4, eosId: -1, sampling: { temperature: 0.9 }, seed: 1 })); // 负 eosId
    expectThrow(() => generate(model, [1, 2], { maxTokens: 4 }));                            // 缺 sampling 且缺 rng/seed
    expectThrow(() => generate(model, [1, 2], { maxTokens: 4, sampling: { temperature: 0.9 } })); // 有 sampling 仍缺 rng/seed
    expectThrow(() => generate(null, [1, 2], okOpts));                                       // model 非函数
    expectThrow(() => generate(model, [1, -2], okOpts));                                     // 负 token
    expectThrow(() => generate(model, [1, 2], { maxTokens: 2.5, seed: 1 }));                 // 非整数 maxTokens
    expectThrow(() => generate(model, [1, 2], { maxTokens: 4, eosId: 1.5, seed: 1 }));       // 非整数 eosId
    let legalOk = 0;
    try { generate(model, [1, 2], { maxTokens: 3, greedy: true }); legalOk++; } catch (e) { /* 贪心免 rng */ }
    try { generate(model, [1, 2], { maxTokens: 3, sampling: { greedy: true } }); legalOk++; } catch (e) { /* 贪心免 rng */ }
    try { generate(model, [1, 2], okOpts); legalOk++; } catch (e) { /* 合法 */ }
    try { generate(model, [1, 2], { maxTokens: 3, sampling: { temperature: 0.9 }, rng: makeRng(3) }); legalOk++; } catch (e) { /* rng 直传 */ }
    cases.push(caseResult('INV-GEN-008', '不良输入抛错 + 合法输入不抛（漏抛计数）',
      Math.max(missed, 4 - legalOk), 0, 0));
  }

  // 不变量登记（8 条，worst 取实测绝对偏差）
  const INV_EXPR = {
    'INV-GEN-001': '流式回调 token 序列与 generate 返回 tokens 逐位一致（回调抛错向上传播）',
    'INV-GEN-002': 'KV 缓存增量路径每步 logits 与全序列前向黄金参照逐元素一致 ≤ 1e-12',
    'INV-GEN-003': '共享前缀的批量生成结果与各自独立 generate 逐位一致（池命中防空洞化）',
    'INV-GEN-004': 'EOS（第 3 步 0 基必产）长度==3 且不含 eos；maxTokens 截断；同时设置取先到者',
    'INV-GEN-005': '同种子两次 generate 逐位相等；贪心跨种子逐位相等；采样跨种子有差异',
    'INV-GEN-006': 'generate 采样序列等于 sampling-forge sample 对同一 logits 序列的逐次采样',
    'INV-GEN-007': 'temperature=1e-8/topK=1 时输出等于独立逐步 argmax 序列',
    'INV-GEN-008': '空 prompt / maxTokens=0 / 负 eosId / 缺 sampling 且缺 rng（贪心除外）必抛错'
  };
  for (const id of Object.keys(INV_EXPR)) {
    const c = cases.find(x => x.id === id);
    invariants.push({ id, expr: INV_EXPR[id], tol: c.tol, worst: c.abs_err, status: c.status, note: '' });
  }

  const started = Date.now();
  return buildReport({
    module: 'generation-forge', version: '1.0.0', layer: 'L4', cases, invariants,
    seed: '0x9E3779B9', started,
    tol: {
      'INV-GEN-001': 0, 'INV-GEN-002': 1e-12, 'INV-GEN-003': 0, 'INV-GEN-004': 0,
      'INV-GEN-005': 0, 'INV-GEN-006': 0, 'INV-GEN-007': 0, 'INV-GEN-008': 0
    }
  });
}

const API = {
  name: 'generation-forge', version: '1.0.0', layer: 'L4',
  generate, generateStream, generateNoCache, batchGenerate, prefixCache, stopCheck,
  createTableModel, stableKey, longestCommonPrefix, selfCheck
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeGeneration = API;

if (typeof process !== 'undefined' && process.argv[1] && /generation-forge\.js$/.test(process.argv[1]) &&
    (process.argv.includes('--self-check') || process.argv.includes('--json'))) {
  const r = selfCheck({ quick: process.argv.includes('--quick') });
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : V().formatTableSafe(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
