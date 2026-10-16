/**
 * @forge-module kv-cache-forge
 * @layer L4
 * @version 1.0.0
 * @brief     KV 缓存与增量解码内核：预分配连续缓冲 + 分页块表（PagedAttention 布局）的多层 K/V 缓存，
 *            支持单 token 追加、批量预填充、前缀读取、增量解码（缓存注意力）、区间逐出与容量统计。
 *            增量解码与全量因果 SDPA 数值等价（逐元素 ≤ 1e-12），8 条硬不变量可独立交叉验证。
 * @reuse     vLLM PagedAttention (Kwon et al. 2023)|https://arxiv.org/abs/2309.06180|Apache-2.0|A|分页块表 KV 缓存布局
 * @reuse     llama.cpp|https://github.com/ggerganov/llama.cpp|MIT|B|KV cache 预分配与增量解码
 * @reuse     HuggingFace transformers DynamicCache|https://github.com/huggingface/transformers|Apache-2.0|C|增量缓存接口契约
 * @reuse     FlashAttention (Dao et al. 2022)|https://arxiv.org/abs/2205.14135|BSD-3|D|增量与全量注意力数值等价性口径
 * @beyond    纯 JS 单文件零构建；块表间接寻址使分页/连续两种物理布局对外读数逐位一致；增量解码复用
 *            attention-forge SDPA 而非重写注意力公式；预分配缓冲写满不重扩（utilization 精确==1）；逐出即清零。
 * @invariant INV-KVC-001|写入读出一致：乱序 append 覆盖 [0,T) 后 getKV 与写入值逐位相等|0|缓冲区 round-trip
 * @invariant INV-KVC-002|增量解码等价全量前向：T=8 逐 token append+decode 与一次性全序列因果 SDPA 对应位置输出逐元素一致 ≤ 1e-12|1e-12|attention-forge sdpa 全量参照
 * @invariant INV-KVC-003|容量边界：pos≥maxLen 或 layer≥nLayers（含负值/非整数/行宽非法/T 超限）时 appendKV/prefillKV 抛错，漏抛计数 0|0|异常契约
 * @invariant INV-KVC-004|分页一致性：同一序列走块表分页布局与连续布局，getKV / 增量解码 / 统计逐位一致|0|块表间接寻址等价
 * @invariant INV-KVC-005|多层隔离：写 layer0 后其余层 getKV 全零且有效长度不受污染|0|层间缓冲寻址隔离
 * @invariant INV-KVC-006|确定性：同序列两次完整增量解码（默认 sdpa 路径与自定义 forwardFn 路径）输出逐位相等|0|纯函数确定性
 * @invariant INV-KVC-007|预分配不重扩：写满 maxLen 时 cacheStats.utilization 精确==1 且底层 Float64Array 长度恒定|0|容量刚性
 * @invariant INV-KVC-008|逐出窗口：evictRange(from) 后有效长度==from 且已逐出位置读出头为零|0|截断语义
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

let _A = null;
function A() {
  if (_A) return _A;
  if (typeof require !== 'undefined') { try { _A = require('./attention-forge.js'); } catch (e) {} }
  if (!_A && typeof globalThis !== 'undefined') _A = globalThis.ForgeAttention || null;
  if (!_A) throw new Error('attention-forge 依赖缺失');
  return _A;
}

/* ============================================================================
 * 缓冲布局（行主序连续 Float64Array，与 attention-forge 矩阵约定一致）：
 *   逻辑索引 (layer, head, pos, k) → 物理偏移
 *   ((layer·nHeads + head)·slotsPerLayer + slot(pos))·headDim + k
 *   其中 slot(pos) 经块表间接寻址：slot = blockTable[layer][pos/blockSize]·blockSize + pos%blockSize。
 *   'contiguous' 布局块表为恒等映射；'paged' 布局按逆序分配物理块（证明间接寻址正确性）。
 * ========================================================================== */

/** 逻辑位置 → 物理槽位（块表间接寻址）。pos 须为 [0,maxLen) 整数。 */
function slotOf(cache, layer, pos) {
  const block = Math.floor(pos / cache.blockSize);
  const phys = cache.blockTables[layer][block];
  return phys * cache.blockSize + (pos - block * cache.blockSize);
}

/** (layer, head, pos, k) → 底层缓冲线性偏移。 */
function bufIndex(cache, layer, head, pos, k) {
  return ((layer * cache.nHeads + head) * cache.slotsPerLayer + slotOf(cache, layer, pos)) * cache.headDim + k;
}

/** 缓存对象形态校验（须由 createKVCache 创建）。 */
function checkCache(cache) {
  if (!cache || typeof cache !== 'object' || !(cache.kBuf instanceof Float64Array) || !(cache.vBuf instanceof Float64Array)) {
    throw new Error('cache 必须由 createKVCache 创建');
  }
}

/** 层索引校验：整数且 ∈ [0, nLayers)。 */
function checkLayer(cache, layer) {
  if (!Number.isInteger(layer) || layer < 0 || layer >= cache.nLayers) {
    throw new Error('layer 越界：' + layer + '（nLayers=' + cache.nLayers + '）');
  }
}

/** 位置校验：整数且 ∈ [0, maxLen)。 */
function checkPos(cache, pos) {
  if (!Number.isInteger(pos) || pos < 0 || pos >= cache.maxLen) {
    throw new Error('pos 越界：' + pos + '（maxLen=' + cache.maxLen + '）');
  }
}

/** 行向量校验：长度须为 nHeads·headDim，返回 Float64Array 视图（不拷贝）。 */
function checkRow(row, width, name) {
  if (!row || typeof row.length !== 'number' || row.length !== width) {
    throw new Error(name + ' 长度须为 nHeads·headDim=' + width + '，收到 ' + (row ? row.length : row) + '');
  }
  return row;
}

/* ---------- 1. 创建缓存（预分配 + 块表） ---------- */

/**
 * 创建多层 KV 缓存：一次性预分配全部连续缓冲（写满不重扩），每层持块表（分页）。
 * @param {object} opts
 * @param {number} opts.nLayers  层数（正整数）
 * @param {number} opts.nHeads   每层注意力头数（正整数；=1 即单头）
 * @param {number} opts.headDim  每头维度（正整数）
 * @param {number} opts.maxLen   每层最大序列长度（正整数）
 * @param {number} [opts.blockSize=16] 分页块大小（正整数，须 ≤ maxLen 无强制，nBlocks=ceil(maxLen/blockSize)）
 * @param {'contiguous'|'paged'} [opts.layout='contiguous'] 物理块布局：恒等映射 / 逆序分配
 * @returns {object} 缓存对象（内部字段：kBuf/vBuf 连续缓冲、blockTables 块表、len 每层有效长度）
 */
function createKVCache(opts) {
  opts = opts || {};
  const nLayers = opts.nLayers, nHeads = opts.nHeads, headDim = opts.headDim, maxLen = opts.maxLen;
  const blockSize = opts.blockSize === undefined ? 16 : opts.blockSize;
  const named = { nLayers, nHeads, headDim, maxLen, blockSize };
  for (const key of Object.keys(named)) {
    if (!Number.isInteger(named[key]) || named[key] < 1) {
      throw new Error(key + ' 必须为正整数，收到 ' + named[key]);
    }
  }
  const layout = opts.layout === undefined ? 'contiguous' : opts.layout;
  if (layout !== 'contiguous' && layout !== 'paged') {
    throw new Error("layout 仅支持 'contiguous' | 'paged'，收到 " + layout);
  }
  const nBlocks = Math.ceil(maxLen / blockSize);
  const slotsPerLayer = nBlocks * blockSize; // ≥ maxLen，物理槽位数
  const total = nLayers * nHeads * slotsPerLayer * headDim;
  const kBuf = new Float64Array(total); // 预分配，运行期 length 恒定
  const vBuf = new Float64Array(total);
  const blockTables = new Array(nLayers);
  const len = new Int32Array(nLayers);   // 每层有效长度（初始 0）
  for (let l = 0; l < nLayers; l++) {
    const bt = new Int32Array(nBlocks);
    for (let b = 0; b < nBlocks; b++) bt[b] = (layout === 'paged') ? (nBlocks - 1 - b) : b;
    blockTables[l] = bt;
  }
  return {
    nLayers, nHeads, headDim, maxLen, blockSize, nBlocks, slotsPerLayer, layout,
    kBuf, vBuf, blockTables, len
  };
}

/* ---------- 2. 单 token 追加 ---------- */

/**
 * 在指定层/位置写入单 token 的 K/V（行宽 = nHeads·headDim；nHeads=1 即单头）。
 * 越界（layer/pos）或行宽非法抛错；有效长度取 max(len, pos+1)。
 * @returns {number} 写入后该层有效长度
 */
function appendKV(cache, layer, pos, kRow, vRow) {
  checkCache(cache);
  checkLayer(cache, layer);
  checkPos(cache, pos);
  const width = cache.nHeads * cache.headDim;
  const k = checkRow(kRow, width, 'kRow');
  const v = checkRow(vRow, width, 'vRow');
  const { kBuf, vBuf, nHeads, headDim, slotsPerLayer } = cache;
  const slot = slotOf(cache, layer, pos);
  for (let h = 0; h < nHeads; h++) {
    const base = ((layer * nHeads + h) * slotsPerLayer + slot) * headDim;
    for (let d = 0; d < headDim; d++) {
      kBuf[base + d] = k[h * headDim + d];
      vBuf[base + d] = v[h * headDim + d];
    }
  }
  if (pos + 1 > cache.len[layer]) cache.len[layer] = pos + 1;
  return cache.len[layer];
}

/* ---------- 3. 批量预填充 ---------- */

/**
 * 批量写入序列 [0, T) 的 K/V（T = K.length ≤ maxLen，行宽 = nHeads·headDim）。
 * @param {Array<Float64Array>} K T × (nHeads·headDim)
 * @param {Array<Float64Array>} V T × (nHeads·headDim)
 * @returns {number} 写入后该层有效长度
 */
function prefillKV(cache, layer, K, V) {
  checkCache(cache);
  checkLayer(cache, layer);
  if (!Array.isArray(K) || !Array.isArray(V) || K.length === 0 || K.length !== V.length) {
    throw new Error('K/V 必须为同长非空二维矩阵（行主序）');
  }
  const T = K.length;
  if (T > cache.maxLen) throw new Error('预填充长度 T=' + T + ' 超过 maxLen=' + cache.maxLen);
  for (let t = 0; t < T; t++) appendKV(cache, layer, t, K[t], V[t]);
  return cache.len[layer];
}

/* ---------- 4. 前缀读取 ---------- */

/**
 * 读出指定层 [0, upto) 的 K/V 拷贝（upto 缺省 = 该层有效长度；允许 > len，未写槽为 0）。
 * @returns {{K:Array<Float64Array>, V:Array<Float64Array>}} 各 upto 行，行宽 nHeads·headDim
 */
function getKV(cache, layer, upto) {
  checkCache(cache);
  checkLayer(cache, layer);
  const end = (upto === undefined || upto === null) ? cache.len[layer] : upto;
  if (!Number.isInteger(end) || end < 0 || end > cache.maxLen) {
    throw new Error('upto 须为 [0, maxLen=' + cache.maxLen + '] 整数，收到 ' + end);
  }
  const width = cache.nHeads * cache.headDim;
  const K = new Array(end), Vv = new Array(end);
  for (let p = 0; p < end; p++) {
    const kr = new Float64Array(width), vr = new Float64Array(width);
    for (let h = 0; h < cache.nHeads; h++) {
      for (let d = 0; d < cache.headDim; d++) {
        kr[h * cache.headDim + d] = cache.kBuf[bufIndex(cache, layer, h, p, d)];
        vr[h * cache.headDim + d] = cache.vBuf[bufIndex(cache, layer, h, p, d)];
      }
    }
    K[p] = kr; Vv[p] = vr;
  }
  return { K, V: Vv };
}

/* ---------- 5. 增量解码 ---------- */

/**
 * 用缓存对单个新 query 做一步注意力：query 对应位置为该层当前末尾（len-1），
 * 注意力窗口为已缓存前缀 [0, len)。内部复用 attention-forge 的 sdpa（逐头）。
 * @param {object} cache
 * @param {number} layer
 * @param {Float64Array|number[]} q query 行（长度 nHeads·headDim）
 * @param {Function|null} [forwardFn] 自定义前向：(q, K, V, meta) → 输出行（长度 nHeads·headDim）；
 *        为 null/undefined 时用内部逐头 SDPA。meta = {layer, pos, nHeads, headDim, len}
 * @returns {Float64Array} 输出行（长度 nHeads·headDim）
 */
function incrementalDecode(cache, layer, q, forwardFn) {
  checkCache(cache);
  checkLayer(cache, layer);
  const width = cache.nHeads * cache.headDim;
  const qq = checkRow(q, width, 'q');
  if (forwardFn !== undefined && forwardFn !== null && typeof forwardFn !== 'function') {
    throw new Error('forwardFn 必须为函数或 null');
  }
  const T = cache.len[layer];
  if (T === 0) throw new Error('缓存为空（layer ' + layer + ' 无有效 K/V），无法增量解码');
  const prefix = getKV(cache, layer, T);
  if (forwardFn) {
    const raw = forwardFn(qq, prefix.K, prefix.V, { layer, pos: T - 1, nHeads: cache.nHeads, headDim: cache.headDim, len: T });
    const out = checkRow(raw, width, 'forwardFn 输出');
    return Float64Array.from(out);
  }
  // 默认路径：逐头 SDPA（复用 attention-forge），单 query 对前缀无需因果掩码
  const hd = cache.headDim;
  const out = new Float64Array(width);
  for (let h = 0; h < cache.nHeads; h++) {
    const qh = Float64Array.from(qq.subarray(h * hd, h * hd + hd));
    const Kh = new Array(T), Vh = new Array(T);
    for (let j = 0; j < T; j++) {
      Kh[j] = Float64Array.from(prefix.K[j].subarray(h * hd, h * hd + hd));
      Vh[j] = Float64Array.from(prefix.V[j].subarray(h * hd, h * hd + hd));
    }
    const oh = A().sdpa([qh], Kh, Vh, {}).out[0];
    for (let d = 0; d < hd; d++) out[h * hd + d] = oh[d];
  }
  return out;
}

/* ---------- 6. 区间逐出 ---------- */

/**
 * 从位置 from 起逐出/截断：有效长度置为 min(len, from)，并清零 [from, maxLen) 已逐出槽位。
 * @returns {number} 新的有效长度
 */
function evictRange(cache, layer, from) {
  checkCache(cache);
  checkLayer(cache, layer);
  if (!Number.isInteger(from) || from < 0 || from > cache.maxLen) {
    throw new Error('from 须为 [0, maxLen=' + cache.maxLen + '] 整数，收到 ' + from);
  }
  const newLen = Math.min(cache.len[layer], from);
  const { kBuf, vBuf, nHeads, headDim, slotsPerLayer } = cache;
  for (let pos = from; pos < cache.maxLen; pos++) {
    const slot = slotOf(cache, layer, pos);
    for (let h = 0; h < nHeads; h++) {
      const base = ((layer * nHeads + h) * slotsPerLayer + slot) * headDim;
      for (let d = 0; d < headDim; d++) { kBuf[base + d] = 0; vBuf[base + d] = 0; }
    }
  }
  cache.len[layer] = newLen;
  return newLen;
}

/* ---------- 7. 容量统计 ---------- */

/**
 * 容量统计（聚合全部层）。
 * @returns {{usedSlots:number, capacity:number, utilization:number}}
 *          usedSlots = Σ_l len[l]；capacity = nLayers·maxLen；utilization = usedSlots/capacity
 */
function cacheStats(cache) {
  checkCache(cache);
  let usedSlots = 0;
  for (let l = 0; l < cache.nLayers; l++) usedSlots += cache.len[l];
  const capacity = cache.nLayers * cache.maxLen;
  return { usedSlots, capacity, utilization: usedSlots / capacity };
}

/* ============================================================================
 * selfCheck — 8 条不变量
 * ========================================================================== */
function zerosMatrix(r, c) { const m = new Array(r); for (let i = 0; i < r; i++) m[i] = new Float64Array(c); return m; }
function randomMatrix(rng, r, c) {
  const m = new Array(r);
  for (let i = 0; i < r; i++) { const row = new Float64Array(c); for (let k = 0; k < c; k++) row[k] = rng() * 2 - 1; m[i] = row; }
  return m;
}
/** Fisher-Yates 乱序（确定性 rng）。 */
function shuffled(n, rng) {
  const a = new Array(n);
  for (let i = 0; i < n; i++) a[i] = i;
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); const t = a[i]; a[i] = a[j]; a[j] = t; }
  return a;
}

function selfCheck(opts) {
  opts = opts || {};
  const { caseResult, buildReport } = V();
  const { makeRng } = N();
  const cases = [];
  const invariants = [];
  const rng = makeRng(0x9e3779b9);

  // INV-KVC-001 写入读出一致（乱序 append 覆盖 [0,T)，round-trip 逐位相等）
  {
    const nLayers = 2, nHeads = 2, headDim = 4, maxLen = 12, blockSize = 4;
    const cache = createKVCache({ nLayers, nHeads, headDim, maxLen, blockSize });
    const T = 9, d = nHeads * headDim;
    const K = randomMatrix(rng, T, d), Vv = randomMatrix(rng, T, d);
    for (const p of shuffled(T, rng)) appendKV(cache, 1, p, K[p], Vv[p]);
    let dev = 0;
    const got = getKV(cache, 1, T), gotDefault = getKV(cache, 1);
    for (let p = 0; p < T; p++) for (let k = 0; k < d; k++) {
      dev = Math.max(dev, Math.abs(got.K[p][k] - K[p][k]), Math.abs(got.V[p][k] - Vv[p][k]));
      dev = Math.max(dev, Math.abs(gotDefault.K[p][k] - K[p][k]), Math.abs(gotDefault.V[p][k] - Vv[p][k]));
    }
    const lenOk = (cache.len[1] === T) ? 0 : 1;
    cases.push(caseResult('INV-KVC-001', '写入读出一致（乱序 append 逐位 round-trip）', Math.max(dev, lenOk), 0, 0));
  }

  // INV-KVC-002 增量解码等价全量前向（T=8；参照 = attention-forge 逐头因果 SDPA）
  {
    const nLayers = 2, nHeads = 2, headDim = 4, maxLen = 8, blockSize = 4;
    const T = 8, d = nHeads * headDim;
    const Q = randomMatrix(rng, T, d), K = randomMatrix(rng, T, d), Vv = randomMatrix(rng, T, d);
    const cache = createKVCache({ nLayers, nHeads, headDim, maxLen, blockSize });
    const outs = new Array(T);
    for (let t = 0; t < T; t++) {
      appendKV(cache, 0, t, K[t], Vv[t]);
      outs[t] = incrementalDecode(cache, 0, Q[t], null);
    }
    // 全量参照：逐头一次性因果 SDPA
    const { sdpa, splitHeads } = A();
    const Qh = splitHeads(Q, nHeads), Kh = splitHeads(K, nHeads), Vh = splitHeads(Vv, nHeads);
    let dev = 0, nzInc = 0, nzRef = 0;
    for (let h = 0; h < nHeads; h++) {
      const ref = sdpa(Qh[h], Kh[h], Vh[h], { causal: true }).out;
      for (let t = 0; t < T; t++) for (let k = 0; k < headDim; k++) {
        dev = Math.max(dev, Math.abs(outs[t][h * headDim + k] - ref[t][k]));
        nzRef = Math.max(nzRef, Math.abs(ref[t][k]));
      }
    }
    for (let t = 0; t < T; t++) for (let k = 0; k < d; k++) nzInc = Math.max(nzInc, Math.abs(outs[t][k]));
    // 防空洞化：增量与参照输出必须确实非零
    const nonVacuous = (nzInc > 1e-6 && nzRef > 1e-6) ? 0 : 1;
    cases.push(caseResult('INV-KVC-002', '增量解码 == 全量因果 SDPA（逐元素）', Math.max(dev, nonVacuous), 0, 1e-12));
  }

  // INV-KVC-003 容量边界（越界必抛，漏抛计数 0；合法边界不抛）
  {
    const cache = createKVCache({ nLayers: 2, nHeads: 1, headDim: 4, maxLen: 6, blockSize: 4 });
    const row = new Float64Array(4);
    let missed = 0;
    const expectThrow = (fn) => { try { fn(); missed++; } catch (e) { /* 预期抛错 */ } };
    expectThrow(() => appendKV(cache, 0, 6, row, row));    // pos == maxLen
    expectThrow(() => appendKV(cache, 0, 99, row, row));   // pos 远超 maxLen
    expectThrow(() => appendKV(cache, 0, -1, row, row));   // pos 负
    expectThrow(() => appendKV(cache, 0, 1.5, row, row));  // pos 非整数
    expectThrow(() => appendKV(cache, 2, 0, row, row));    // layer == nLayers
    expectThrow(() => appendKV(cache, 5, 0, row, row));    // layer 远超
    expectThrow(() => appendKV(cache, -1, 0, row, row));   // layer 负
    expectThrow(() => appendKV(cache, 0, 0, new Float64Array(3), row)); // 行宽非法
    const over = randomMatrix(rng, 7, 4);                  // T = maxLen+1 预填充
    expectThrow(() => prefillKV(cache, 0, over, over));
    let legalOk = 0;
    try { appendKV(cache, 1, 5, row, row); legalOk = 1; } catch (e) { /* 合法边界不应抛 */ }
    cases.push(caseResult('INV-KVC-003', '容量/层边界抛错（漏抛计数）', Math.max(missed, 1 - legalOk), 0, 0));
  }

  // INV-KVC-004 分页一致性（块表分页布局 vs 连续布局，读数/解码/统计逐位一致）
  {
    const base = { nLayers: 2, nHeads: 2, headDim: 4, maxLen: 12, blockSize: 4 };
    const T = 10, d = 8;
    const K = randomMatrix(rng, T, d), Vv = randomMatrix(rng, T, d), Q = randomMatrix(rng, T, d);
    const c1 = createKVCache(Object.assign({}, base, { layout: 'contiguous' }));
    const c2 = createKVCache(Object.assign({}, base, { layout: 'paged' }));
    for (let l = 0; l < base.nLayers; l++) { prefillKV(c1, l, K, Vv); prefillKV(c2, l, K, Vv); }
    let dev = 0;
    for (let l = 0; l < base.nLayers; l++) {
      const g1 = getKV(c1, l, T), g2 = getKV(c2, l, T);
      for (let p = 0; p < T; p++) for (let k = 0; k < d; k++) {
        dev = Math.max(dev, Math.abs(g1.K[p][k] - g2.K[p][k]), Math.abs(g1.V[p][k] - g2.V[p][k]));
      }
    }
    for (let t = 0; t < T; t++) {
      const o1 = incrementalDecode(c1, 0, Q[t], null), o2 = incrementalDecode(c2, 0, Q[t], null);
      for (let k = 0; k < d; k++) dev = Math.max(dev, Math.abs(o1[k] - o2[k]));
    }
    const s1 = cacheStats(c1), s2 = cacheStats(c2);
    dev = Math.max(dev, Math.abs(s1.usedSlots - s2.usedSlots), Math.abs(s1.capacity - s2.capacity), Math.abs(s1.utilization - s2.utilization));
    // 防空洞化：两种布局的块表必须确实不同（间接寻址被真正 exercise）
    let tablesDiffer = 0;
    for (let b = 0; b < c1.nBlocks; b++) if (c1.blockTables[0][b] !== c2.blockTables[0][b]) tablesDiffer = 1;
    cases.push(caseResult('INV-KVC-004', '分页 vs 连续布局：getKV/解码/统计逐位一致', Math.max(dev, 1 - tablesDiffer), 0, 0));
  }

  // INV-KVC-005 多层隔离（写 layer0 后其余层全零、有效长度不受污染）
  {
    const nLayers = 3, nHeads = 2, headDim = 4, maxLen = 8, blockSize = 4;
    const cache = createKVCache({ nLayers, nHeads, headDim, maxLen, blockSize });
    const T = 6, d = nHeads * headDim;
    const K = randomMatrix(rng, T, d), Vv = randomMatrix(rng, T, d);
    prefillKV(cache, 0, K, Vv);
    let dev = 0;
    for (const l of [1, 2]) {
      const g = getKV(cache, l, T);
      for (let p = 0; p < T; p++) for (let k = 0; k < d; k++) dev = Math.max(dev, Math.abs(g.K[p][k]), Math.abs(g.V[p][k]));
    }
    const lenOk = (cache.len[1] === 0 && cache.len[2] === 0 && cache.len[0] === T) ? 0 : 1;
    // 再写 layer2，layer1 仍须全零（交叉污染）
    prefillKV(cache, 2, K, Vv);
    const g1 = getKV(cache, 1, T);
    for (let p = 0; p < T; p++) for (let k = 0; k < d; k++) dev = Math.max(dev, Math.abs(g1.K[p][k]), Math.abs(g1.V[p][k]));
    cases.push(caseResult('INV-KVC-005', '多层隔离：未写层全零且长度隔离', Math.max(dev, lenOk), 0, 0));
  }

  // INV-KVC-006 确定性（同序列两次完整增量解码逐位相等；默认路径 + forwardFn 路径）
  {
    const nLayers = 2, nHeads = 2, headDim = 4, maxLen = 8, blockSize = 4;
    const T = 8, d = nHeads * headDim;
    const Q = randomMatrix(rng, T, d), K = randomMatrix(rng, T, d), Vv = randomMatrix(rng, T, d);
    const runOnce = (forwardFn) => {
      const cache = createKVCache({ nLayers, nHeads, headDim, maxLen, blockSize });
      const outs = new Array(T);
      for (let t = 0; t < T; t++) {
        appendKV(cache, 0, t, K[t], Vv[t]);
        outs[t] = incrementalDecode(cache, 0, Q[t], forwardFn);
      }
      return outs;
    };
    let dev = 0;
    const a1 = runOnce(null), a2 = runOnce(null);
    for (let t = 0; t < T; t++) for (let k = 0; k < d; k++) dev = Math.max(dev, Math.abs(a1[t][k] - a2[t][k]));
    // 自定义 forwardFn（确定性加权组合），验证 forwardFn 管道的确定性
    const ffn = (q, Km, Vm) => {
      const out = new Float64Array(q.length);
      for (let j = 0; j < Km.length; j++) {
        for (let k = 0; k < q.length; k++) out[k] += (q[k] + Km[j][k]) * Vm[j][k] / Km.length;
      }
      return out;
    };
    const b1 = runOnce(ffn), b2 = runOnce(ffn);
    for (let t = 0; t < T; t++) for (let k = 0; k < d; k++) dev = Math.max(dev, Math.abs(b1[t][k] - b2[t][k]));
    let nz = 0;
    for (let t = 0; t < T; t++) for (let k = 0; k < d; k++) nz = Math.max(nz, Math.abs(a1[t][k]));
    cases.push(caseResult('INV-KVC-006', '确定性：两次增量解码逐位相等', Math.max(dev, nz > 1e-6 ? 0 : 1), 0, 0));
  }

  // INV-KVC-007 预分配不重扩（写满 maxLen：utilization 精确==1，缓冲 length 恒定）
  {
    const nLayers = 2, nHeads = 2, headDim = 4, maxLen = 16, blockSize = 4;
    const cache = createKVCache({ nLayers, nHeads, headDim, maxLen, blockSize });
    const kLen0 = cache.kBuf.length, vLen0 = cache.vBuf.length;
    const T = maxLen, d = nHeads * headDim;
    const K = randomMatrix(rng, T, d), Vv = randomMatrix(rng, T, d);
    for (let l = 0; l < nLayers; l++) prefillKV(cache, l, K, Vv);
    const s = cacheStats(cache);
    const bufStable = (cache.kBuf.length === kLen0 && cache.vBuf.length === vLen0) ? 0 : 1;
    const utilExact = (s.utilization === 1 && s.usedSlots === s.capacity && s.capacity === nLayers * maxLen) ? 0 : 1;
    let missed = 0;
    try { appendKV(cache, 0, maxLen, K[0], Vv[0]); missed++; } catch (e) { /* 写满后越界必抛 */ }
    cases.push(caseResult('INV-KVC-007', '预分配不重扩：util==1 且缓冲恒定', Math.max(bufStable, utilExact, missed), 0, 0));
  }

  // INV-KVC-008 逐出窗口（evictRange(from)：有效长度==from，逐出区清零，保留区不变）
  {
    const nLayers = 2, nHeads = 2, headDim = 4, maxLen = 12, blockSize = 4;
    const cache = createKVCache({ nLayers, nHeads, headDim, maxLen, blockSize });
    const T = 10, d = nHeads * headDim;
    const K = randomMatrix(rng, T, d), Vv = randomMatrix(rng, T, d);
    prefillKV(cache, 0, K, Vv);
    const from = 6;
    const newLen = evictRange(cache, 0, from);
    const lenOk = (newLen === from && cache.len[0] === from) ? 0 : 1;
    let dev = 0;
    const g = getKV(cache, 0, from);
    for (let p = 0; p < from; p++) for (let k = 0; k < d; k++) {
      dev = Math.max(dev, Math.abs(g.K[p][k] - K[p][k]), Math.abs(g.V[p][k] - Vv[p][k]));
    }
    // 默认窗口（=len）长度须为 from；已逐出物理槽位须清零
    const gd = getKV(cache, 0);
    const windowOk = (gd.K.length === from && gd.V.length === from) ? 0 : 1;
    let evictZero = 0;
    for (let p = from; p < T; p++) {
      for (let h = 0; h < nHeads; h++) for (let k = 0; k < headDim; k++) {
        evictZero = Math.max(evictZero, Math.abs(cache.kBuf[bufIndex(cache, 0, h, p, k)]), Math.abs(cache.vBuf[bufIndex(cache, 0, h, p, k)]));
      }
    }
    cases.push(caseResult('INV-KVC-008', '逐出窗口：有效长度=from 且逐出位清零', Math.max(lenOk, dev, windowOk, evictZero), 0, 0));
  }

  // 不变量登记
  for (const id of ['INV-KVC-001', 'INV-KVC-002', 'INV-KVC-003', 'INV-KVC-004', 'INV-KVC-005', 'INV-KVC-006', 'INV-KVC-007', 'INV-KVC-008']) {
    const c = cases.find(x => x.id === id);
    invariants.push({ id, expr: id, tol: c.tol, worst: c.abs_err, status: c.status, note: '' });
  }

  const started = Date.now();
  return buildReport({
    module: 'kv-cache-forge', version: '1.0.0', layer: 'L4', cases, invariants,
    seed: '0x9E3779B9', started,
    tol: { 'INV-KVC-001': 0, 'INV-KVC-002': 1e-12, 'INV-KVC-003': 0, 'INV-KVC-004': 0, 'INV-KVC-005': 0, 'INV-KVC-006': 0, 'INV-KVC-007': 0, 'INV-KVC-008': 0 }
  });
}

const API = {
  name: 'kv-cache-forge', version: '1.0.0', layer: 'L4',
  createKVCache, appendKV, prefillKV, getKV, incrementalDecode, evictRange, cacheStats, selfCheck
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeKVCache = API;

if (typeof process !== 'undefined' && process.argv[1] && /kv-cache-forge\.js$/.test(process.argv[1]) && (process.argv.includes('--self-check') || process.argv.includes('--json'))) {
  const r = selfCheck({ quick: process.argv.includes('--quick') });
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : V().formatTableSafe(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
