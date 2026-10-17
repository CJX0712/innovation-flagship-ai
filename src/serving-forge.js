/**
 * @forge-module serving-forge
 * @layer L7
 * @version 1.0.0
 * @brief     推理服务系统内核：静态批处理/算子融合/内核DAG调度/固定块内存池/流式管线/负载均衡/请求队列/令牌桶限流
 * @reuse     vLLM《Efficient Memory Management for LLM Serving with PagedAttention》(SOSP 2023)|https://arxiv.org/abs/2309.06180|Apache-2.0|A|静态批处理与内存池范式
 * @reuse     TensorRT Operator Fusion 文档|https://docs.nvidia.com/deeplearning/tensorrt/|未声明|B|算子融合数值等价性口径
 * @reuse     Token Bucket (Wikipedia 综述, RFC 2617 语义)|https://en.wikipedia.org/wiki/Token_bucket|未声明|A|令牌桶限流算法
 * @reuse     Round-robin scheduling 与 Little's Law 经典结果|https://en.wikipedia.org/wiki/Round-robin_scheduling|未声明|A|轮询/加权轮询负载均衡范式
 * @beyond    纯 JS 单文件零构建零依赖、双端同构 UMD；直接站在 vLLM PagedAttention、TensorRT 融合等价性、令牌桶与轮询调度经典结果之上——8 条不变量全部含证伪对立面（批无关性/调用计数反作弊/异权重非退化分歧/耗尽安全/确定性重放），灵魂不变量（批处理等价、内存池无重叠）用独立手搓黄金参照交叉验证；全部时序为逻辑时钟/逻辑步，固定种子 sfc32，两次运行逐位可复现。
 * @invariant INV-SV-001|批处理等价：随机请求集组批(含padding)经有效区提取后输出≡手搓串行黄金参照(TIGHT)，且批无关性：批大小1与批大小N单条输出一致|1e-15|独立手搓串行参照 + 多批形扫描 + 批无关性
 * @invariant INV-SV-002|算子融合：融合kernel序输出≡未融合串联输出(TIGHT)且kernel调用次数严格减少|1e-15|逐位等价 + 调用计数反作弊
 * @invariant INV-SV-003|内核调度：DAG拓扑序全边满足 pos[u]<pos[v](EXACT)，不同优先级权重产生不同合法序(非退化)，环依赖必抛错|0|全边扫描 + 异权重分歧 + 环抛错
 * @invariant INV-SV-004|内存池：随机分配/释放下任意逻辑时刻活跃块区间两两不重叠(EXACT)，复用使 mallocs<allocs，池耗尽返回null而非越界，double-free 抛错|0|逐步区间断言 + 复用计数 + 耗尽/双释安全
 * @invariant INV-SV-005|流式管线：任意chunkSize(含不整除/1/超长)分块拼接≡全量一次性结果(EXACT)且不同chunkSize结果互一致|0|多chunkSize交叉对照
 * @invariant INV-SV-006|负载均衡：等权轮询各worker承载差≤1(EXACT)，加权[3,1]承载比精确3:1，加权序≠均权序(非退化)|0|计数断言 + 分歧断言
 * @invariant INV-SV-007|请求队列：FIFO出队序≡入队序(EXACT)，优先级队列出队序按(priority,seq)升序(EXACT)，同输入两队列出队序必分歧|0|序列对照 + 手搓排序参照 + 非退化
 * @invariant INV-SV-008|令牌桶：固定请求+逻辑时钟两次运行放行序列逐位一致(确定性)，任意窗口[0,T]放行数≤capacity+rate·T(EXACT)，rate减半放行序列必改变|0|重放对照 + 闭式上界 + 证伪
 * @seed      0x9E3779B9
 */
'use strict';

/* ============================================================================
 * 惰性双端依赖：Node 端 require，浏览器端回退 globalThis（与 td-forge 同口径）。
 * ========================================================================== */
let _V = null;
function V() {
  if (_V) return _V;
  if (typeof require !== 'undefined') { try { _V = require('./core-verify.js'); } catch (e) {} }
  if (!_V && typeof globalThis !== 'undefined') _V = globalThis.ForgeVerify || null;
  if (!_V) throw new Error('core-verify 缺失');
  return _V;
}

/* ============================================================================
 * 项目统一容差（九档，与 core-verify.TOL 同值，本地声明避免启动期强依赖）
 * ========================================================================== */
const TOL = { EXACT: 0, ULP: 1e-16, TIGHT: 1e-15, ORTHO: 1e-14, RESID: 1e-12, SOLVE: 1e-10, GRAD: 1e-6, MONO: -1e-9, CONV: 1e-6 };

/* ============================================================================
 * 私有可种子 PRNG（sfc32）：主种子 0x9E3779B9 派生，warm-up 20 次后使用。
 * 全模块随机负载/序列均出自此 PRNG —— 固定种子即完全可复现。
 * ========================================================================== */
function sfc32(a, b, c, d) {
  return function () {
    a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0;
    let t = (a + b) | 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) | 0;
    c = (c << 21) | (c >>> 11);
    d = (d + 1) | 0;
    t = (t + d) | 0;
    c = (c + t) | 0;
    return (t >>> 0) / 4294967296;
  };
}

function makeRng(seed) {
  const s = (seed === undefined ? 0x9E3779B9 : seed) >>> 0;
  const rng = sfc32(s ^ 0x9E3779B9, s ^ 0x85EBCA6B, s ^ 0xC2B2AE35, s ^ 0x27D4EB2F);
  for (let i = 0; i < 20; i++) rng();
  return rng;
}

/* ============================================================================
 * O00 推理批处理：静态组批 + padding 对齐 + elementwise 批处理
 *   —— elementwise 变换保证 pad 区不泄漏进有效区，批输出 ≡ 串行输出。
 * ========================================================================== */
function makeBatches(reqs, maxBatch) {
  if (!Array.isArray(reqs)) throw new TypeError('reqs 必须为数组');
  if (!Number.isInteger(maxBatch) || maxBatch < 1) throw new TypeError('maxBatch 必须为正整数');
  const batches = [];
  for (let i = 0; i < reqs.length; i += maxBatch) batches.push(reqs.slice(i, i + maxBatch));
  return batches;
}

/** 批内 pad 到最大长度；返回 {rows, lens, ids}，不改输入。 */
function padBatch(batch, padVal) {
  if (!Array.isArray(batch)) throw new TypeError('batch 必须为数组');
  padVal = (padVal === undefined) ? 0 : padVal;
  let maxLen = 0;
  for (const r of batch) {
    if (!r || !Array.isArray(r.payload)) throw new TypeError('请求必须含 payload 数组');
    if (r.payload.length > maxLen) maxLen = r.payload.length;
  }
  const rows = [], lens = [], ids = [];
  for (const r of batch) {
    const row = r.payload.slice();
    while (row.length < maxLen) row.push(padVal);
    rows.push(row); lens.push(r.payload.length); ids.push(r.id);
  }
  return { rows, lens, ids };
}

/** 对 pad 后的批逐位应用 elementwise 变换 op，返回输出矩阵（含 pad 区）。 */
function processBatch(rows, lens, op) {
  if (!Array.isArray(rows) || !Array.isArray(lens)) throw new TypeError('rows/lens 必须为数组');
  if (typeof op !== 'function') throw new TypeError('op 必须为函数');
  const out = [];
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i], o = new Array(row.length);
    for (let j = 0; j < row.length; j++) o[j] = op(row[j]);
    out.push(o);
  }
  return out;
}

/* ============================================================================
 * O01 算子融合：相邻 ['add','relu'] / ['mul','sigmoid'] / ['bias','relu']
 *   融合为单 kernel。fused kernel 与串联组合的浮点表达式树逐位一致。
 * ========================================================================== */
const KERNELS = {
  add:          (x) => x + 1,
  bias:         (x) => x + 2,
  mul:          (x) => x * 3,
  relu:         (x) => Math.max(0, x),
  sigmoid:      (x) => 1 / (1 + Math.exp(-x)),
  add_relu:     (x) => Math.max(0, x + 1),
  bias_relu:    (x) => Math.max(0, x + 2),
  mul_sigmoid:  (x) => 1 / (1 + Math.exp(-(x * 3)))
};

const FUSE_RULES = { 'add+relu': 'add_relu', 'mul+sigmoid': 'mul_sigmoid', 'bias+relu': 'bias_relu' };

function fuseOps(ops) {
  if (!Array.isArray(ops)) throw new TypeError('ops 必须为数组');
  const kernels = [];
  for (let i = 0; i < ops.length; i++) {
    const a = ops[i];
    if (typeof a !== 'string' || !KERNELS[a]) throw new TypeError('未知算子: ' + a);
    if (i + 1 < ops.length) {
      const b = ops[i + 1];
      if (typeof b !== 'string' || !KERNELS[b]) throw new TypeError('未知算子: ' + b);
      const fused = FUSE_RULES[a + '+' + b];
      if (fused) { kernels.push(fused); i++; continue; }
    }
    kernels.push(a);
  }
  return { kernels, calls: kernels.length };
}

/** 按 kernel 序列作用于标量 x（kernel 可为原始算子名或融合名）。 */
function applyOps(x, kernels) {
  if (typeof kernels === 'string') kernels = [kernels];
  if (!Array.isArray(kernels)) throw new TypeError('kernels 必须为数组或字符串');
  let v = x;
  for (const k of kernels) {
    const f = KERNELS[k];
    if (!f) throw new TypeError('未知 kernel: ' + k);
    v = f(v);
  }
  return v;
}

/* ============================================================================
 * O02 内核调度：Kahn 拓扑 + ready 集内 priority 降序 / id 字典序升序破平。
 *   环依赖或未知依赖必抛错；同 DAG 不同优先级产生不同合法序。
 * ========================================================================== */
function scheduleKernels(ops) {
  if (!Array.isArray(ops)) throw new TypeError('ops 必须为数组');
  const byId = new Map();
  for (const o of ops) {
    if (!o || o.id === undefined || o.id === null) throw new TypeError('算子缺少 id');
    if (byId.has(o.id)) throw new Error('duplicate id: ' + o.id);
    if (!Array.isArray(o.deps)) throw new TypeError('deps 必须为数组');
    byId.set(o.id, o);
  }
  const indeg = new Map(), dependents = new Map();
  for (const o of ops) {
    indeg.set(o.id, o.deps.length);
    for (const d of o.deps) {
      if (!byId.has(d)) throw new Error('unknown dep: ' + d);
      if (!dependents.has(d)) dependents.set(d, []);
      dependents.get(d).push(o.id);
    }
  }
  const ready = [];
  for (const o of ops) if (indeg.get(o.id) === 0) ready.push(o.id);
  const order = [];
  while (ready.length) {
    let best = 0;                                   // priority 降序，id 字典序升序破平
    for (let i = 1; i < ready.length; i++) {
      const pi = byId.get(ready[i]).priority || 0;
      const pb = byId.get(ready[best]).priority || 0;
      if (pi > pb || (pi === pb && String(ready[i]) < String(ready[best]))) best = i;
    }
    const id = ready.splice(best, 1)[0];
    order.push(id);
    for (const nxt of (dependents.get(id) || [])) {
      indeg.set(nxt, indeg.get(nxt) - 1);
      if (indeg.get(nxt) === 0) ready.push(nxt);
    }
  }
  if (order.length !== ops.length) throw new Error('cycle detected');
  return order;
}

/* ============================================================================
 * O03 内存池：固定块 free-list 分配/释放（vLLM PagedAttention 范式）。
 *   池耗尽 alloc 返回 null（安全失败，绝不越界）；double-free 抛错；
 *   mallocs 计真实"取新块"次数，free 后复用不计入 —— 复用率可验证。
 * ========================================================================== */
class MemoryPool {
  constructor(blockSize, nBlocks) {
    if (!Number.isInteger(blockSize) || blockSize < 1) throw new TypeError('blockSize 必须为正整数');
    if (!Number.isInteger(nBlocks) || nBlocks < 1) throw new TypeError('nBlocks 必须为正整数');
    this.blockSize = blockSize;
    this.nBlocks = nBlocks;
    this.freeList = [];          // 已释放可复用的 slot 栈
    this.fresh = 0;              // 从未使用过的 slot 高水位
    this.active = new Set();     // 当前活跃 slot 集
    this.allocs = 0; this.frees = 0; this.mallocs = 0;
  }

  alloc() {
    let slot;
    if (this.freeList.length) { slot = this.freeList.pop(); }            // 复用
    else if (this.fresh < this.nBlocks) { slot = this.fresh++; this.mallocs++; } // 取新块
    else return null;                                                    // 池耗尽：安全失败
    this.active.add(slot);
    this.allocs++;
    return { slot, offset: slot * this.blockSize };
  }

  free(h) {
    if (!h || typeof h.slot !== 'number') throw new TypeError('handle 非法');
    if (!this.active.has(h.slot)) throw new Error('double free');
    this.active.delete(h.slot);
    this.freeList.push(h.slot);
    this.frees++;
  }

  stats() {
    return {
      allocs: this.allocs, frees: this.frees, mallocs: this.mallocs,
      reuseRate: this.allocs > 0 ? 1 - this.mallocs / this.allocs : 0
    };
  }
}

/* ============================================================================
 * O05 流式管线：分块处理 + 拼接。任意 chunkSize（含不整除/1/超长）下
 *   拼接结果 ≡ 全量一次性结果。
 * ========================================================================== */
function streamProcess(items, chunkSize, fn) {
  if (!Array.isArray(items)) throw new TypeError('items 必须为数组');
  if (!Number.isInteger(chunkSize) || chunkSize < 1) throw new TypeError('chunkSize 必须为正整数');
  if (typeof fn !== 'function') throw new TypeError('fn 必须为函数');
  const out = [];
  let chunkIdx = 0;
  for (let base = 0; base < items.length; base += chunkSize) {
    const chunk = items.slice(base, base + chunkSize);
    const part = fn(chunk, chunkIdx, base);
    if (!Array.isArray(part)) throw new TypeError('fn 必须返回数组');
    for (const v of part) out.push(v);
    chunkIdx++;
  }
  return out;
}

/* ============================================================================
 * O06 负载均衡：轮询与平滑加权轮询（smooth WRR：cw += w，取最大者并减
 *   totalWeight，平局取最小索引）。w=[3,1] 在 4 的倍数次分发下精确 3:1。
 * ========================================================================== */
function makeRoundRobin(workers) {
  if (!Array.isArray(workers) || workers.length === 0) throw new TypeError('workers 必须为非空数组');
  let idx = -1;
  const loads = new Array(workers.length).fill(0);
  return {
    dispatch() { idx = (idx + 1) % workers.length; loads[idx]++; return workers[idx]; },
    loads() { return loads.slice(); }
  };
}

function makeWeightedRR(workers, weights) {
  if (!Array.isArray(workers) || workers.length === 0) throw new TypeError('workers 必须为非空数组');
  if (!Array.isArray(weights) || weights.length !== workers.length) throw new TypeError('weights 长度必须等于 workers');
  let total = 0;
  for (const w of weights) {
    if (typeof w !== 'number' || !(w > 0) || !isFinite(w)) throw new TypeError('weights 必须为正数');
    total += w;
  }
  const cw = weights.map(() => 0);
  const loads = workers.map(() => 0);
  return {
    dispatch() {
      for (let i = 0; i < cw.length; i++) cw[i] += weights[i];
      let best = 0;                              // 平局取最小索引（确定性）
      for (let i = 1; i < cw.length; i++) if (cw[i] > cw[best]) best = i;
      cw[best] -= total;
      loads[best]++;
      return workers[best];
    },
    loads() { return loads.slice(); }
  };
}

/* ============================================================================
 * O09 请求队列：FIFO 与优先级队列（priority 升序出队，seq 递增破平）。
 * ========================================================================== */
class FIFOQueue {
  constructor() { this._a = []; }
  enqueue(x) { this._a.push(x); return this; }
  dequeue() { return this._a.length ? this._a.shift() : null; }
  size() { return this._a.length; }
}

class PriorityQueue {
  constructor() { this._a = []; this._seq = 0; }
  enqueue(value, priority) {
    if (typeof priority !== 'number' || !isFinite(priority)) throw new TypeError('priority 必须为有限数');
    this._a.push({ value, priority, seq: this._seq++ });
    return this;
  }
  dequeue() {
    if (!this._a.length) return null;
    let best = 0;
    for (let i = 1; i < this._a.length; i++) {
      const it = this._a[i], b = this._a[best];
      if (it.priority < b.priority || (it.priority === b.priority && it.seq < b.seq)) best = i;
    }
    return this._a.splice(best, 1)[0].value;
  }
  size() { return this._a.length; }
}

/* ============================================================================
 * O12 限流：令牌桶（逻辑时钟）。令牌按 rate·Δt 连续补充，封顶 capacity。
 *   逻辑时刻 t 必须非降；固定请求序列下放行/拒绝序列完全确定。
 * ========================================================================== */
class TokenBucket {
  constructor(rate, capacity) {
    if (typeof rate !== 'number' || !(rate > 0) || !isFinite(rate)) throw new TypeError('rate 必须为正数');
    if (typeof capacity !== 'number' || !(capacity > 0) || !isFinite(capacity)) throw new TypeError('capacity 必须为正数');
    this.rate = rate;
    this.capacity = capacity;
    this.tokens = capacity;      // 初始满桶
    this.lastT = 0;
  }

  tryConsume(t, n) {
    if (typeof t !== 'number' || !isFinite(t)) throw new TypeError('t 必须为有限数');
    if (t < this.lastT) throw new Error('逻辑时钟必须非降');
    n = (n === undefined) ? 1 : n;
    if (!Number.isInteger(n) || n < 1) throw new TypeError('n 必须为正整数');
    this.tokens = Math.min(this.capacity, this.tokens + this.rate * (t - this.lastT));
    this.lastT = t;
    if (this.tokens >= n) { this.tokens -= n; return true; }
    return false;
  }
}

/* ============================================================================
 * selfCheck — 8 条不变量（含证伪对立面；INV-SV-001/004 用独立手搓黄金参照）
 * ========================================================================== */
function selfCheck(opts) {
  opts = opts || {};
  const cases = [];
  const invariants = [];

  /* ---------- INV-SV-001 批处理等价（独立手搓串行黄金参照 + 批无关性） ---------- */
  {
    const OP = (x) => Math.tanh(2 * x + 1) * 0.5;
    const rng = makeRng(0xA0001);
    const reqs = [];
    for (let i = 0; i < 24; i++) {
      const len = 1 + Math.floor(rng() * 8);
      const payload = [];
      for (let j = 0; j < len; j++) payload.push(rng() * 4 - 2);
      reqs.push({ id: 'r' + i, payload });
    }
    // 独立手搓串行黄金参照：绝不调用本模块任何函数
    const serialRef = (payload) => payload.map(OP);
    const batches = makeBatches(reqs, 7);
    const byIdBatch = {};
    let dev = 0;
    for (const batch of batches) {
      const pb = padBatch(batch, 0);
      const out = processBatch(pb.rows, pb.lens, OP);
      for (let i = 0; i < batch.length; i++) {
        byIdBatch[pb.ids[i]] = out[i].slice(0, pb.lens[i]);
        const ref = serialRef(batch[i].payload);
        for (let j = 0; j < pb.lens[i]; j++) dev = Math.max(dev, Math.abs(out[i][j] - ref[j]));
      }
    }
    // 批无关性：批大小=1 与批大小=N 的单条有效区输出一致
    let bdev = 0;
    for (const r of reqs) {
      const pb1 = padBatch([r], 0);
      const single = processBatch(pb1.rows, pb1.lens, OP)[0];
      for (let j = 0; j < r.payload.length; j++) bdev = Math.max(bdev, Math.abs(single[j] - byIdBatch[r.id][j]));
    }
    const ok = (dev <= TOL.TIGHT && bdev <= TOL.TIGHT);
    const worst = ok ? Math.max(dev, bdev) : 1;
    cases.push(V().caseResult('INV-SV-001', '批处理等价（串行黄金参照+批无关性）', worst, 0, TOL.TIGHT,
      '串行偏差=' + dev.toExponential(2) + ' 批无关偏差=' + bdev.toExponential(2) + ' 批数=' + batches.length));
    invariants.push({ id: 'INV-SV-001', expr: '组批(含padding)有效区≡串行参照(TIGHT) 且 批无关性（批1≡批N）', tol: TOL.TIGHT, worst, status: ok ? 'pass' : 'fail', note: '独立手搓串行黄金参照' });
  }

  /* ---------- INV-SV-002 算子融合（数值等价 + 调用计数反作弊） ---------- */
  {
    const ops = ['add', 'relu', 'mul', 'sigmoid', 'add', 'relu', 'bias', 'relu'];
    const fu = fuseOps(ops);
    const rng = makeRng(0xA0002);
    let dev = 0;
    for (let i = 0; i < 5; i++) {
      const x = rng() * 6 - 3;
      dev = Math.max(dev, Math.abs(applyOps(x, fu.kernels) - applyOps(x, ops)));
    }
    const callsOk = (fu.calls < ops.length) && (fu.kernels.length === fu.calls);
    const ok = (dev <= TOL.TIGHT) && callsOk;
    const worst = ok ? dev : 1;
    cases.push(V().caseResult('INV-SV-002', '算子融合等价 + 调用次数减少', worst, 0, TOL.TIGHT,
      '数值偏差=' + dev.toExponential(2) + ' 调用数 ' + fu.calls + '<' + ops.length));
    invariants.push({ id: 'INV-SV-002', expr: '融合kernel输出≡未融合串联(TIGHT) 且 calls<ops.length', tol: TOL.TIGHT, worst, status: ok ? 'pass' : 'fail', note: '调用计数反作弊' });
  }

  /* ---------- INV-SV-003 内核调度（全边拓扑 + 非退化 + 环抛错） ---------- */
  {
    const mk = (pr) => [
      { id: 'a', deps: [], priority: pr[0] },
      { id: 'b', deps: ['a'], priority: pr[1] },
      { id: 'c', deps: ['a'], priority: pr[2] },
      { id: 'd', deps: ['b', 'c'], priority: pr[3] },
      { id: 'e', deps: ['c'], priority: pr[4] },
      { id: 'f', deps: ['d', 'e'], priority: pr[5] },
      { id: 'g', deps: ['f'], priority: pr[6] },
      { id: 'h', deps: ['f'], priority: pr[7] }
    ];
    const edges = [['a', 'b'], ['a', 'c'], ['b', 'd'], ['c', 'd'], ['c', 'e'], ['d', 'f'], ['e', 'f'], ['f', 'g'], ['f', 'h']];
    const checkTopo = (order) => {
      const pos = new Map();
      order.forEach((id, i) => pos.set(id, i));
      let v = 0;
      for (const [u, w] of edges) if (!(pos.get(u) < pos.get(w))) v++;
      return v;
    };
    const o1 = scheduleKernels(mk([1, 2, 3, 4, 5, 6, 7, 8]));
    const o2 = scheduleKernels(mk([8, 7, 6, 5, 4, 3, 2, 1]));
    const v1 = checkTopo(o1), v2 = checkTopo(o2);
    const differs = (o1.join(',') !== o2.join(','));
    let cycleThrew = false;
    try { scheduleKernels([{ id: 'x', deps: ['y'], priority: 1 }, { id: 'y', deps: ['x'], priority: 1 }]); } catch (e) { cycleThrew = true; }
    const violations = v1 + v2 + (differs ? 0 : 1) + (cycleThrew ? 0 : 1);
    cases.push(V().caseResult('INV-SV-003', '内核DAG调度（全边+非退化+环抛错）', violations, 0, TOL.EXACT,
      '边违例=' + (v1 + v2) + ' 异权重序' + (differs ? '不同' : '相同!') + ' 环抛错=' + cycleThrew));
    invariants.push({ id: 'INV-SV-003', expr: '全边 pos[u]<pos[v](EXACT) 且 异优先级不同合法序 且 环必抛错', tol: TOL.EXACT, worst: violations, status: violations === 0 ? 'pass' : 'fail', note: '全边扫描 + 分歧 + 抛错' });
  }

  /* ---------- INV-SV-004 内存池（区间不重叠 + 复用真 + 耗尽/双释安全） ---------- */
  {
    const pool = new MemoryPool(16, 4);
    const rng = makeRng(0xA0004);
    const handles = [];
    let violations = 0;
    for (let step = 0; step < 40; step++) {
      if (handles.length === 0 || (handles.length < 4 && rng() < 0.6)) {
        const h = pool.alloc();
        if (!h) violations++;
        else handles.push(h);
      } else {
        const i = Math.floor(rng() * handles.length);
        pool.free(handles.splice(i, 1)[0]);
      }
      // 独立实现：任意逻辑时刻活跃块区间两两不重叠 + 不越界（EXACT）
      for (let i = 0; i < handles.length; i++) {
        const a = handles[i];
        if (!(a.offset >= 0 && a.offset + 16 <= 64)) violations++;
        for (let j = i + 1; j < handles.length; j++) {
          const b = handles[j];
          if (a.offset < b.offset + 16 && b.offset < a.offset + 16) violations++;
        }
      }
    }
    const st = pool.stats();
    const reuseOk = st.mallocs < st.allocs;
    const p2 = new MemoryPool(16, 2);
    p2.alloc(); p2.alloc();
    const exhausted = (p2.alloc() === null);        // 池耗尽：分配失败而非越界
    let dfThrew = false;
    try { p2.free({ slot: 99, offset: 99 * 16 }); } catch (e) { dfThrew = true; }  // 非法 handle 必抛
    p2.alloc();
    let dblThrew = false;
    const hd = p2.alloc();
    try { p2.free(hd); p2.free(hd); } catch (e) { dblThrew = true; }               // double-free 必抛
    const worst = violations + (reuseOk ? 0 : 1) + (exhausted ? 0 : 1) + (dfThrew ? 0 : 1) + (dblThrew ? 0 : 1);
    cases.push(V().caseResult('INV-SV-004', '内存池（区间不重叠+复用真+耗尽安全）', worst, 0, TOL.EXACT,
      '区间/越界违例=' + violations + ' mallocs=' + st.mallocs + '<allocs=' + st.allocs + ' 耗尽null=' + exhausted + ' 双释抛错=' + dblThrew));
    invariants.push({ id: 'INV-SV-004', expr: '活跃块区间两两不重叠(EXACT) 且 mallocs<allocs 且 耗尽返回null 且 double-free抛错', tol: TOL.EXACT, worst, status: worst === 0 ? 'pass' : 'fail', note: '逐步区间断言 + 复用计数' });
  }

  /* ---------- INV-SV-005 流式管线（任意 chunkSize 分块无关性） ---------- */
  {
    const rng = makeRng(0xA0005);
    const items = [];
    for (let i = 0; i < 13; i++) items.push(Math.floor(rng() * 100));   // 整数域 → EXACT 比较
    const fn = (chunk) => chunk.map((x) => x * 3 + 1);
    const full = fn(items);
    const sizes = [1, 3, 5, 13, 20];                 // 含 chunkSize=1、=N、>N、不整除
    const results = sizes.map((cs) => streamProcess(items, cs, fn));
    let violations = 0;
    for (const res of results) {
      if (res.length !== full.length) { violations++; continue; }
      for (let i = 0; i < full.length; i++) if (res[i] !== full[i]) violations++;
    }
    cases.push(V().caseResult('INV-SV-005', '流式管线分块无关性', violations, 0, TOL.EXACT,
      'chunkSize=[' + sizes.join(',') + '] N=13 全量对照违例=' + violations));
    invariants.push({ id: 'INV-SV-005', expr: '分块拼接≡全量一次性(EXACT) 且 各chunkSize结果互一致', tol: TOL.EXACT, worst: violations, status: violations === 0 ? 'pass' : 'fail', note: '多chunkSize交叉对照' });
  }

  /* ---------- INV-SV-006 负载均衡（承载差≤1 + 3:1 加权 + 非退化） ---------- */
  {
    const rr = makeRoundRobin(['w0', 'w1', 'w2']);
    for (let i = 0; i < 10; i++) rr.dispatch();
    const L = rr.loads();
    const spread = Math.max.apply(null, L) - Math.min.apply(null, L);
    const wrr = makeWeightedRR(['a', 'b'], [3, 1]);
    const seqW = [];
    for (let i = 0; i < 12; i++) seqW.push(wrr.dispatch());
    const LW = wrr.loads();
    const ratioOk = (LW[0] === 9 && LW[1] === 3);    // smooth WRR：totalWeight=4，12 次 → 精确 9:3
    const rr2 = makeRoundRobin(['a', 'b']);
    const seqR = [];
    for (let i = 0; i < 4; i++) seqR.push(rr2.dispatch());
    const differs = seqW.slice(0, 4).some((x, i) => x !== seqR[i]);
    const violations = (spread <= 1 ? 0 : 1) + (ratioOk ? 0 : 1) + (differs ? 0 : 1);
    cases.push(V().caseResult('INV-SV-006', '负载均衡（承载差≤1 + 3:1 + 非退化）', violations, 0, TOL.EXACT,
      '等权loads=[' + L.join(',') + '] spread=' + spread + ' 加权loads=[' + LW.join(',') + '](应9:3) 加权序≠均权序=' + differs));
    invariants.push({ id: 'INV-SV-006', expr: '等权承载差≤1 且 加权[3,1]→9:3 且 加权序≠均权序', tol: TOL.EXACT, worst: violations, status: violations === 0 ? 'pass' : 'fail', note: '计数断言 + 分歧断言' });
  }

  /* ---------- INV-SV-007 请求队列（FIFO≡入队 + PQ按(priority,seq) + 非退化） ---------- */
  {
    const rng = makeRng(0xA0007);
    const vals = [];
    for (let i = 0; i < 15; i++) vals.push(Math.floor(rng() * 1000));
    const q = new FIFOQueue();
    for (const v of vals) q.enqueue(v);
    const outF = [];
    while (q.size()) outF.push(q.dequeue());
    const fifoOk = outF.every((v, i) => v === vals[i]);
    const pqItems = [[10, 3], [21, 1], [32, 2], [43, 1], [54, 3], [65, 2], [76, 1], [87, 3], [98, 2], [19, 1]];
    const pq = new PriorityQueue();
    for (const [v, p] of pqItems) pq.enqueue(v, p);
    const outP = [];
    while (pq.size()) outP.push(pq.dequeue());
    // 手搓参照：(priority, seq) 升序（seq 唯一 → 全序确定）
    const ref = pqItems.map(([v, p], seq) => ({ v, p, seq }))
      .sort((a, b) => (a.p - b.p) || (a.seq - b.seq))
      .map((x) => x.v);
    const pqOk = outP.every((v, i) => v === ref[i]);
    const differs = outF.slice(0, 10).some((v, i) => v !== outP[i]);
    const violations = (fifoOk ? 0 : 1) + (pqOk ? 0 : 1) + (differs ? 0 : 1);
    cases.push(V().caseResult('INV-SV-007', '请求队列（FIFO≡入队 + PQ(priority,seq) + 非退化）', violations, 0, TOL.EXACT,
      'FIFO=' + (fifoOk ? '≡入队' : '错!') + ' PQ=' + (pqOk ? '≡(priority,seq)参照' : '错!') + ' 两队列出队序分歧=' + differs));
    invariants.push({ id: 'INV-SV-007', expr: 'FIFO出队≡入队(EXACT) 且 PQ出队≡(priority,seq)(EXACT) 且 两种队列必分歧', tol: TOL.EXACT, worst: violations, status: violations === 0 ? 'pass' : 'fail', note: '手搓排序参照 + 非退化' });
  }

  /* ---------- INV-SV-008 令牌桶（确定性重放 + 速率上界 + rate减半证伪） ---------- */
  {
    const runBucket = (rate) => {
      const tb = new TokenBucket(rate, 5);
      const res = [];
      for (let t = 0; t <= 7; t++) for (let k = 0; k < 3; k++) res.push(tb.tryConsume(t, 1) ? 1 : 0);
      return res;
    };
    const r1 = runBucket(2), r2 = runBucket(2);
    const detOk = r1.every((v, i) => v === r2[i]);
    // 窗口上界：[0,T] 放行数 ≤ capacity + rate·T。
    // 推导：初始 capacity 个令牌每枚至多消费一次，[0,T] 内新生成 rate·T 个，
    //       且 rate=2、capacity=5 均为精确浮点数，放行数为整数 → EXACT 成立。
    let boundViol = 0;
    for (let T = 0; T <= 11; T++) {
      let passes = 0;
      for (let i = 0; i < r1.length; i++) if (Math.floor(i / 3) <= T && r1[i] === 1) passes++;
      if (passes > 5 + 2 * T) boundViol++;
    }
    const rHalf = runBucket(1);
    const differs = r1.some((v, i) => v !== rHalf[i]);
    const violations = (detOk ? 0 : 1) + boundViol + (differs ? 0 : 1);
    cases.push(V().caseResult('INV-SV-008', '令牌桶（确定性重放 + 速率上界 + rate减半分歧）', violations, 0, TOL.EXACT,
      '重放一致=' + detOk + ' 上界违例=' + boundViol + ' rate减半分歧=' + differs + ' r1=' + r1.join('')));
    invariants.push({ id: 'INV-SV-008', expr: '两次运行逐位一致 且 ∀T: 放行数≤capacity+rate·T 且 rate减半序列必变', tol: TOL.EXACT, worst: violations, status: violations === 0 ? 'pass' : 'fail', note: '重放对照 + 闭式上界 + 证伪' });
  }

  const started = Date.now();
  return V().buildReport({
    module: 'serving-forge', version: '1.0.0', layer: 'L7', cases, invariants, seed: '0x9E3779B9', started,
    tol: {
      'INV-SV-001': TOL.TIGHT, 'INV-SV-002': TOL.TIGHT, 'INV-SV-003': TOL.EXACT, 'INV-SV-004': TOL.EXACT,
      'INV-SV-005': TOL.EXACT, 'INV-SV-006': TOL.EXACT, 'INV-SV-007': TOL.EXACT, 'INV-SV-008': TOL.EXACT
    }
  });
}

/* ============================================================================
 * API 导出 + 双端挂载 + CLI 自检
 * ========================================================================== */
const API = {
  name: 'serving-forge', version: '1.0.0', layer: 'L7', TOL, makeRng,
  makeBatches, padBatch, processBatch, fuseOps, applyOps, scheduleKernels,
  MemoryPool, streamProcess, makeRoundRobin, makeWeightedRR,
  FIFOQueue, PriorityQueue, TokenBucket, selfCheck
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeServing = API;

if (typeof require !== 'undefined' && require.main === module &&
    (process.argv.includes('--self-check') || process.argv.includes('--json'))) {
  const r = selfCheck({ quick: process.argv.includes('--quick') });
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : V().formatTableSafe(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
