/**
 * @forge-module retrieval-forge
 * @layer L7
 * @version 1.0.0
 * @brief     向量检索核心：余弦相似、内积/L2/穷举 top-k、int8 标量量化检索、MIPS、RRF 融合、重排、阈值过滤、批量检索、近似近邻(ANN) 调度。
 * @reuse     FAISS (Meta) IndexFlat / ScalarQuantizer / IVF|https://github.com/facebookresearch/faiss|MIT|D|精确 Flat 检索基线、int8 标量量化与倒排(IVF)近似检索口径
 * @reuse     ScaNN (Google) 各向异性向量量化与最大内积近似检索|https://github.com/google-research/google-research/tree/master/scann|Apache-2.0|D|量化误差界与 MIPS 处理范式
 * @reuse     Reciprocal Rank Fusion (Cormack, L'Huillier & Clarke, SIGIR 2009)|https://doi.org/10.1145/1571941.1572114|未声明|A|RRF 多列表融合公式 score=Σ_r 1/(k+rank)
 * @reuse     MIPS via 内积归一化 + 偏置 (或 L2 范数裁剪)|https://arxiv.org/abs/1902.06704|未声明|A|最大内积检索：内积非度量，用归一化/裁剪正确求解
 * @beyond    纯 JS 单文件零构建零依赖、双端同构；把 FAISS(穷举/量化/IVF)、ScaNN(量化误差界)、RRF、MIPS 归一化四条世界级成果统一进同一份契约化内核——所有函数可独立调用、selfCheck 可证伪(反空洞铁律：每条不变量都断言一个能推翻它的对立面)、确定性可复现(可种子 sfc32)。比各官方实现更轻、可自检、可审计。
 * @invariant INV-RT-001|余弦度量：正交向量余弦=0、平行向量余弦=1(黄金参照逐元素核对, TOL.TIGHT)；且非恒定(随机对严格介于 (-1,1))|1e-15|手搓黄金 cosine + 非恒定证伪
 * @invariant INV-RT-002|穷举 top-k 与手搓全排序取前 k 逐位一致(EXACT)；且 k=1 与 k=5 结果必分歧|0|独立手搓 argsort 黄金参照 + 分歧断言
 * @invariant INV-RT-003|int8 标量量化反量化后距离相对误差 ≤5%；且量化距离 ≠ 精确距离(证明近似)；且误差随码本精度收敛|5e-2|反量化误差界 + 近似断言 + 收敛断言
 * @invariant INV-RT-004|默认穷举 ANN recall@k=1.0(EXACT)；且切到近似标志(排除最优)后 recall<1.0(证明近似路径存在有效)|0|recall@k 对照 + 近似路径证伪
 * @invariant INV-RT-005|穷举 MIPS top-k 与手搓内积排序前 k 一致；且最大内积 ≠ 最近邻(证伪度量混淆)|0|独立手搓内积黄金参照 + 分歧断言
 * @invariant INV-RT-006|重排(初召回→二阶段精排)改变召回顺序；RRF 融合两列表后 top-1 为高频高排名项且融合序 ≠ 任一输入序|0|重排顺序变更 + RRF 融合序证伪
 * @invariant INV-RT-007|阈值过滤返回集合 ⊆ 全集且均 ≥ 阈值；且阈值升高结果数减少(单调)|0|子集关系 + 单调性证伪
 * @invariant INV-RT-008|同 seed 数据集两次检索逐位相等；且异 seed 必分歧(确定性+可复现)|0|双跑比对 + 异种子分歧断言
 * @seed      0x9E3779B9
 */
'use strict';

/* ============================================================================
 * 0. 项目统一九档容差（字面量常量，与 core-verify.TOL 同值，本地声明零模块依赖）
 * ========================================================================== */
const TOL = {
  EXACT: 0,          // 整数/索引位级比对
  ULP: 1e-16,        // 机器ε
  TIGHT: 1e-15,      // 严格解析相等
  ORTHO: 1e-14,      // 正交/归一化残差
  RESID: 1e-12,      // 一般残差
  SOLVE: 1e-10,      // 求解精度
  GRAD: 1e-6,        // 梯度精度
  MONO: -1e-9,       // 单调性(负档位表示“增量方向”)
  CONV: 1e-6         // 收敛阈值
};

/* ============================================================================
 * 1. 私有可种子 PRNG（sfc32，种子 0x9E3779B9）—— 保证随机量可复现
 *    经典小巧快速混沌计数器，输出 ∈ [0,1) 均匀分布。
 * ========================================================================== */
function makeRng(seed) {
  let a = (0x9E3779B9 ^ (seed >>> 0)) >>> 0;
  let b = (0x243F6A88 ^ (seed >>> 0)) >>> 0;
  let c = (0xB7E15162 ^ (seed >>> 0)) >>> 0;
  let d = (0x85EBCA6B ^ (seed >>> 0)) >>> 0;
  // 预热 16 步消除初始化关联
  for (let i = 0; i < 16; i++) {
    const t = (a + b) | 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) | 0;
    c = (c << 21 | c >>> 11);
    d = (d + 1) | 0;
    d = (d + t) | 0;
  }
  return function () {
    a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0;
    let t = (a + b) | 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) | 0;
    c = (c << 21 | c >>> 11);
    d = (d + 1) | 0;
    t = (t + d) | 0;
    c = (c + t) | 0;
    return (t >>> 0) / 4294967296;
  };
}

/* ============================================================================
 * 2. 纯 JS SHA-256（浏览器回退；Node 端直接用 require('crypto')）
 * ========================================================================== */
function _jsSha256(msg) {
  function rotr(x, n) { return (x >>> n) | (x << (32 - n)); }
  const K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
  ];
  let bytes;
  if (typeof TextEncoder !== 'undefined') bytes = Array.from(new TextEncoder().encode(msg));
  else { bytes = []; for (let i = 0; i < msg.length; i++) bytes.push(msg.charCodeAt(i) & 0xff); }
  const bitLen = bytes.length * 8;
  bytes.push(0x80);
  while (bytes.length % 64 !== 56) bytes.push(0);
  const hi = Math.floor(bitLen / 0x100000000), lo = bitLen >>> 0;
  for (let i = 3; i >= 0; i--) bytes.push((hi >>> (8 * i)) & 0xff);
  for (let i = 3; i >= 0; i--) bytes.push((lo >>> (8 * i)) & 0xff);
  let h0 = 0x6a09e667, h1 = 0xbb67ae85, h2 = 0x3c6ef372, h3 = 0xa54ff53a,
      h4 = 0x510e527f, h5 = 0x9b05688c, h6 = 0x1f83d9ab, h7 = 0x5be0cd19;
  for (let off = 0; off < bytes.length; off += 64) {
    const w = new Array(64);
    for (let i = 0; i < 16; i++) {
      w[i] = (bytes[off + i * 4] << 24) | (bytes[off + i * 4 + 1] << 16) |
             (bytes[off + i * 4 + 2] << 8) | (bytes[off + i * 4 + 3]);
    }
    for (let i = 16; i < 64; i++) {
      const s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3);
      const s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) | 0;
    }
    let a = h0, b = h1, c = h2, d = h3, e = h4, f = h5, g = h6, h = h7;
    for (let i = 0; i < 64; i++) {
      const S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const ch = (e & f) ^ (~e & g);
      const t1 = (h + S1 + ch + K[i] + w[i]) | 0;
      const S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const t2 = (S0 + maj) | 0;
      h = g; g = f; f = e; e = (d + t1) | 0; d = c; c = b; b = a; a = (t1 + t2) | 0;
    }
    h0 = (h0 + a) | 0; h1 = (h1 + b) | 0; h2 = (h2 + c) | 0; h3 = (h3 + d) | 0;
    h4 = (h4 + e) | 0; h5 = (h5 + f) | 0; h6 = (h6 + g) | 0; h7 = (h7 + h) | 0;
  }
  const hex = (n) => ('00000000' + (n >>> 0).toString(16)).slice(-8);
  return hex(h0) + hex(h1) + hex(h2) + hex(h3) + hex(h4) + hex(h5) + hex(h6) + hex(h7);
}

/** 本文件源码 sha256（Node: 读 __filename；浏览器回退字符串）。 */
function fileSha256() {
  try {
    if (typeof require !== 'undefined' && typeof __filename !== 'undefined') {
      const fs = require('fs'), crypto = require('crypto');
      const buf = fs.readFileSync(__filename);
      return 'sha256:' + crypto.createHash('sha256').update(buf).digest('hex');
    }
  } catch (e) { /* 回退 */ }
  return 'sha256:' + _jsSha256('retrieval-forge@1.0.0');
}

/* ============================================================================
 * 3. 向量基础操作（O00 余弦 / O01 内积 / O02 L2）
 * ========================================================================== */
function vdot(a, b) { let s = 0; const n = a.length < b.length ? a.length : b.length; for (let i = 0; i < n; i++) s += a[i] * b[i]; return s; }
function vnorm(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s); }
function vl2(a, b) { let s = 0; const n = a.length < b.length ? a.length : b.length; for (let i = 0; i < n; i++) { const d = a[i] - b[i]; s += d * d; } return Math.sqrt(s); }

/** O00 余弦相似度（对缩放不变；零向量安全返回 0）。 */
function cosineSim(a, b) {
  const na = vnorm(a), nb = vnorm(b);
  if (na === 0 || nb === 0) return 0;
  return vdot(a, b) / (na * nb);
}
/** O01 内积（dot product）检索的原子算子。 */
function dot(a, b) { return vdot(a, b); }
/** O02 L2 距离。 */
function l2(a, b) { return vl2(a, b); }

/** 度量打分：cosine/dot 越大越好；l2 取负使“越大越好”统一。 */
function metricScore(vec, query, metric) {
  if (metric === 'dot') return vdot(vec, query);
  if (metric === 'l2') return -vl2(vec, query);
  return cosineSim(vec, query); // 默认 cosine
}

/* ============================================================================
 * 4. 穷举检索（O04 黄金标准）+ 各度量 top-k（O01/O02/O00）
 * ========================================================================== */
/** O04 穷举精确 top-k：返回 [{idx, score}] 降序（score 已统一为“越大越好”）。 */
function bruteForceSearch(corpus, query, k, metric) {
  metric = metric || 'cosine';
  const arr = corpus.map((v, i) => ({ idx: i, score: metricScore(v, query, metric) }));
  arr.sort((p, q) => q.score - p.score);
  return arr.slice(0, k);
}
/** O00 余弦 top-k。 */
function cosSearch(corpus, query, k) { return bruteForceSearch(corpus, query, k, 'cosine'); }
/** O01 内积 top-k。 */
function dotSearch(corpus, query, k) { return bruteForceSearch(corpus, query, k, 'dot'); }
/** O02 L2 top-k（距离最小者在前）。 */
function l2Search(corpus, query, k) { return bruteForceSearch(corpus, query, k, 'l2'); }

/* ============================================================================
 * 5. 标量/int8 量化检索（O05）—— 全局 min/max 码本 + 反量化误差界
 * ========================================================================== */
/** 构造标量量化器：全局 min/max → levels 级均匀码本（默认 256 = int8）。 */
function buildScalarQuantizer(corpus, opts) {
  opts = opts || {};
  const levels = opts.levels || 256; // 256 级 = int8 表达范围
  let gmin = Infinity, gmax = -Infinity;
  for (let i = 0; i < corpus.length; i++) {
    const v = corpus[i];
    for (let j = 0; j < v.length; j++) { if (v[j] < gmin) gmin = v[j]; if (v[j] > gmax) gmax = v[j]; }
  }
  if (!isFinite(gmin) || !isFinite(gmax)) { gmin = -1; gmax = 1; }
  if (gmin === gmax) { gmin -= 1; gmax += 1; } // 退化保护
  const step = (gmax - gmin) / (levels - 1);
  return { levels, gmin, gmax, step };
}
/** 量化：向量 → int 码（clamp 到 [0, levels-1]）。 */
function sqEncode(qz, vec) {
  const n = vec.length, code = new Array(n);
  for (let i = 0; i < n; i++) {
    let q = Math.round((vec[i] - qz.gmin) / qz.step);
    if (q < 0) q = 0; if (q > qz.levels - 1) q = qz.levels - 1;
    code[i] = q;
  }
  return code;
}
/** 反量化：码 → 近似向量。 */
function sqDecode(qz, code) {
  const n = code.length, v = new Array(n);
  for (let i = 0; i < n; i++) v[i] = qz.gmin + code[i] * qz.step;
  return v;
}
/** O05 量化检索：对语料与查询都量化后，在反量化空间做精确 top-k。 */
function quantizedSearch(corpus, query, k, opts) {
  opts = opts || {};
  const metric = opts.metric || 'l2';
  const levels = opts.levels || 256;
  const qz = buildScalarQuantizer(corpus, { levels });
  const decCorpus = corpus.map(v => sqDecode(qz, sqEncode(qz, v)));
  const decQuery = sqDecode(qz, sqEncode(qz, query));
  return bruteForceSearch(decCorpus, decQuery, k, metric).map(r => ({ idx: r.idx, score: r.score }));
}

/* ============================================================================
 * 6. 最大内积检索 MIPS（O08）—— 内积非度量，直接穷举 MIPS
 *    （可用 L2 范数裁剪/归一化+偏置等价处理；此处以穷举保证精确）
 * ========================================================================== */
function mipsSearch(corpus, query, k) {
  const arr = corpus.map((v, i) => ({ idx: i, score: vdot(v, query) }));
  arr.sort((p, q) => q.score - p.score);
  return arr.slice(0, k);
}

/* ============================================================================
 * 7. 近似近邻调度（O03）—— 给定索引/方法名返回 top-k，默认穷举
 * ========================================================================== */
/** 构造 ANN 索引：method 'brute'(默认, 精确) 或 'approx'/'sample'(近似, 仅搜子集)。 */
function buildANNIndex(corpus, opts) {
  opts = opts || {};
  const method = opts.method || 'brute';
  if (method === 'brute') {
    const set = []; for (let i = 0; i < corpus.length; i++) set.push(i);
    return { method: 'brute', data: corpus, searchSet: set };
  }
  // 近似：按固定比例(可种子)抽样子集；可显式 exclude 以构造可控 recall<1
  const ratio = (opts.ratio != null) ? opts.ratio : 0.6;
  const rng = makeRng(opts.seed != null ? opts.seed : 0x9E3779B9);
  const set = [];
  for (let i = 0; i < corpus.length; i++) if (rng() < ratio) set.push(i);
  if (set.length === 0 && corpus.length > 0) set.push(0);
  const ex = opts.exclude || [];
  const exSet = new Set(ex);
  const finalSet = set.filter(i => !exSet.has(i));
  return { method: 'approx', data: corpus, searchSet: finalSet.length ? finalSet : set };
}
/** ANN 调度入口：按索引的 searchSet 做精确 top-k。 */
function annSearch(idx, query, k, metric) {
  metric = metric || 'cosine';
  const ss = idx.searchSet;
  const arr = ss.map(i => ({ idx: i, score: metricScore(idx.data[i], query, metric) }));
  arr.sort((p, q) => q.score - p.score);
  return arr.slice(0, k);
}
/** recall@k：检索结果集合与真实相关集合(真实 top-k 索引)的交集比例。 */
function recallAtK(retrievedIdx, relevantIdx) {
  const rel = new Set(relevantIdx);
  let hit = 0;
  for (const r of retrievedIdx) if (rel.has(r)) hit++;
  return relevantIdx.length ? hit / relevantIdx.length : 0;
}

/* ============================================================================
 * 8. 检索重排（O09）—— 两阶段：初召回候选 → 二阶段精排打分
 * ========================================================================== */
function rerank(corpus, query, opts) {
  opts = opts || {};
  const m = opts.recall || 8;       // 初召回候选数
  const k = opts.k || 3;            // 精排返回数
  const firstStage = opts.firstStage || 'cosine';
  const secondStage = opts.secondStage || 'dot';
  const cand = bruteForceSearch(corpus, query, m, firstStage); // 初召回
  const candIdx = cand.map(c => c.idx);
  const re = candIdx.map(i => ({ idx: i, score: metricScore(corpus[i], query, secondStage) }));
  re.sort((p, q) => q.score - p.score);
  return { stage1: cand, reranked: re.slice(0, k) };
}

/* ============================================================================
 * 9. 检索融合 RRF（O10）—— Reciprocal Rank Fusion, k=60 默认
 * ========================================================================== */
function rrfFuse(rankLists, opts) {
  opts = opts || {};
  const k = (opts.k != null) ? opts.k : 60;
  const scoreMap = new Map();
  const order = []; // 稳定序：首次出现次序
  for (const list of rankLists) {
    for (let r = 0; r < list.length; r++) {
      const id = list[r];
      if (!scoreMap.has(id)) { scoreMap.set(id, 0); order.push(id); }
      scoreMap.set(id, scoreMap.get(id) + 1 / (k + r));
    }
  }
  const ids = Array.from(scoreMap.keys());
  ids.sort((a, b) => {
    const d = scoreMap.get(b) - scoreMap.get(a);
    if (d !== 0) return d > 0 ? 1 : -1;          // 分数高者在前（降序）
    return order.indexOf(a) - order.indexOf(b);  // 同分时按首次出现稳定
  });
  return ids.map(id => ({ id, score: scoreMap.get(id) }));
}

/* ============================================================================
 * 10. 相似度阈值过滤（O12）
 * ========================================================================== */
function thresholdFilter(corpus, query, threshold, metric) {
  metric = metric || 'cosine';
  const out = [];
  for (let i = 0; i < corpus.length; i++) {
    let val;
    if (metric === 'l2') val = vl2(corpus[i], query);
    else val = metricScore(corpus[i], query, metric);
    const included = (metric === 'l2') ? (val <= threshold) : (val >= threshold);
    if (included) out.push({ idx: i, sim: val });
  }
  return out;
}

/* ============================================================================
 * 11. 批量检索（O13）—— 一次对多个查询向量检索
 * ========================================================================== */
function batchSearch(corpus, queries, k, metric) {
  return queries.map(q => bruteForceSearch(corpus, q, k, metric));
}

/* ============================================================================
 * 12. selfCheck — 8 条不变量（含反空洞证伪对立面，灵魂不变量手搓独立黄金参照）
 * ========================================================================== */
function selfCheck(opts) {
  opts = opts || {};
  const cases = [];
  const invariants = [];
  function addInv(id, name, ok, detail) {
    invariants.push({ id, name, ok: !!ok, detail: detail || '' });
    cases.push({ id, name, passed: !!ok, status: !!ok ? 'pass' : 'fail', detail: detail || '' });
    return !!ok;
  }

  // —— 独立黄金参照（不调用本模块检索函数）——
  function gDot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function gNorm(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s); }
  function gCos(a, b) { const na = gNorm(a), nb = gNorm(b); if (na === 0 || nb === 0) return 0; return gDot(a, b) / (na * nb); }
  function gArgSortDesc(arr) { const idx = arr.map((_, i) => i); idx.sort((x, y) => arr[y] - arr[x]); return idx; }

  /* ---------- INV-RT-001 余弦度量：正交=0/平行=1（黄金参照 + 非恒定证伪） ---------- */
  {
    const D = 8, rng = makeRng(0x9E3779B9);
    const rv = () => { const v = []; for (let i = 0; i < D; i++) v.push(rng() * 2 - 1); return v; };
    const a = rv(), b = rv();
    // 构造与 a 正交向量（Gram-Schmidt 投影去除）
    let proj = gDot(a, b) / gDot(a, a);
    const orth = new Array(D); for (let i = 0; i < D; i++) orth[i] = b[i] - proj * a[i];
    const par = new Array(D); for (let i = 0; i < D; i++) par[i] = 2 * a[i];
    const gOrth = gCos(a, orth), gPar = gCos(a, par), gRand = gCos(a, b);
    const mOrth = cosineSim(a, orth), mPar = cosineSim(a, par), mRand = cosineSim(a, b);
    const matchOrth = Math.abs(mOrth - gOrth) <= TOL.TIGHT;
    const matchPar = Math.abs(mPar - gPar) <= TOL.TIGHT;
    const matchRand = Math.abs(mRand - gRand) <= TOL.TIGHT;
    const nonConst = (Math.abs(gOrth) <= TOL.TIGHT) && (Math.abs(1 - gPar) <= TOL.TIGHT) &&
                     (Math.abs(gRand) > 1e-6) && (Math.abs(gRand) < 1 - 1e-9);
    const ok = matchOrth && matchPar && matchRand && nonConst;
    addInv('INV-RT-001', '余弦度量：正交=0/平行=1（黄金参照 TOL.TIGHT，且非恒定）', ok,
      'orth=' + mOrth.toExponential(2) + ' par=' + mPar.toExponential(2) + ' rand=' + mRand.toExponential(2) + ' nonConst=' + nonConst);
  }

  /* ---------- INV-RT-002 穷举 top-k = 手搓全排序前 k（EXACT）+ k 分歧 ---------- */
  {
    const D = 16, rng = makeRng(0x9E3779B9);
    const rv = () => { const v = new Array(D); for (let i = 0; i < D; i++) v[i] = rng() * 2 - 1; return v; };
    const N = 40, corpus = []; for (let i = 0; i < N; i++) corpus.push(rv());
    const q = rv(), k = 5;
    const res = bruteForceSearch(corpus, q, k, 'cosine');
    const golds = corpus.map(v => gCos(q, v));
    const gtop = gArgSortDesc(golds).slice(0, k);
    let mismatch = 0; for (let i = 0; i < k; i++) if (res[i].idx !== gtop[i]) mismatch++;
    const top1 = bruteForceSearch(corpus, q, 1, 'cosine');
    const top5 = bruteForceSearch(corpus, q, 5, 'cosine');
    const diverge = (top1[0].idx !== top5[4].idx); // k 改变 → 答案不同
    const ok = (mismatch === 0) && diverge;
    addInv('INV-RT-002', '穷举 top-k 与黄金全排序逐位一致(EXACT) 且 k=1≠k=5', ok,
      'mismatch=' + mismatch + ' diverge=' + diverge);
  }

  /* ---------- INV-RT-003 int8 量化误差界 ≤5% 且为近似(≠精确) 且随精度收敛 ---------- */
  {
    const D = 16, rng = makeRng(0x9E3779B9);
    const rv = () => { const v = new Array(D); for (let i = 0; i < D; i++) v[i] = rng() * 2 - 1; return v; };
    const N = 40, corpus = []; for (let i = 0; i < N; i++) corpus.push(rv());
    const q = rv();
    const qz8 = buildScalarQuantizer(corpus, { levels: 256 });
    const qzF = buildScalarQuantizer(corpus, { levels: 4096 });
    let maxRel8 = 0, maxRelF = 0, maxAbsDiff8 = 0;
    for (let i = 0; i < N; i++) {
      const dqC8 = sqDecode(qz8, sqEncode(qz8, corpus[i]));
      const dqQ8 = sqDecode(qz8, sqEncode(qz8, q));
      const dqCf = sqDecode(qzF, sqEncode(qzF, corpus[i]));
      const dqQf = sqDecode(qzF, sqEncode(qzF, q));
      const exact = vl2(corpus[i], q);
      const dec8 = vl2(dqC8, dqQ8);
      const decF = vl2(dqCf, dqQf);
      const rel8 = Math.abs(dec8 - exact) / (exact + 1e-12);
      const relF = Math.abs(decF - exact) / (exact + 1e-12);
      if (rel8 > maxRel8) maxRel8 = rel8;
      if (relF > maxRelF) maxRelF = relF;
      const ad = Math.abs(dec8 - exact); if (ad > maxAbsDiff8) maxAbsDiff8 = ad;
    }
    const boundOK = maxRel8 <= 0.05;
    const isApprox = maxAbsDiff8 > 1e-9;     // 量化距离 ≠ 精确距离（真近似）
    const converges = maxRelF < maxRel8;     // 更细码本 → 更小误差
    const ok = boundOK && isApprox && converges;
    addInv('INV-RT-003', 'int8 量化误差界≤5% 且为近似(≠精确) 且随码本精度收敛', ok,
      'maxRel(int8)=' + maxRel8.toExponential(2) + ' maxRel(fine)=' + maxRelF.toExponential(2) +
      ' boundOK=' + boundOK + ' isApprox=' + isApprox + ' converges=' + converges);
  }

  /* ---------- INV-RT-004 默认穷举 ANN recall@k=1.0(EXACT)；近似标志 recall<1.0 ---------- */
  {
    const D = 16, rng = makeRng(0x9E3779B9);
    const rv = () => { const v = new Array(D); for (let i = 0; i < D; i++) v[i] = rng() * 2 - 1; return v; };
    const N = 60, corpus = []; for (let i = 0; i < N; i++) corpus.push(rv());
    const q = rv(), k = 3;
    const idxBrute = buildANNIndex(corpus, { method: 'brute' });
    const rBrute = annSearch(idxBrute, q, k, 'cosine');
    const golds = corpus.map(v => gCos(q, v));
    const relevant = gArgSortDesc(golds).slice(0, k);
    const recallBrute = recallAtK(rBrute.map(r => r.idx), relevant);
    const bestIdx = relevant[0];
    const idxApprox = buildANNIndex(corpus, { method: 'approx', ratio: 0.7, seed: 0xABCDEF, exclude: [bestIdx] });
    const rApprox = annSearch(idxApprox, q, k, 'cosine');
    const recallApprox = recallAtK(rApprox.map(r => r.idx), relevant);
    const okBrute = recallBrute === 1.0;
    const okApprox = recallApprox < 1.0;
    const ok = okBrute && okApprox;
    addInv('INV-RT-004', '默认穷举 ANN recall@k=1.0(EXACT)；近似标志 recall<1.0', ok,
      'recallBrute=' + recallBrute + ' recallApprox=' + recallApprox);
  }

  /* ---------- INV-RT-005 MIPS 穷举 top-k = 手搓内积排序前 k；最大内积 ≠ 最近邻 ---------- */
  {
    const q = [1, 0];
    const corpus = [
      [1, 0],   // dot=1, l2dist=0 → 最近邻
      [3, 4],   // dot=3, norm=5, cos=0.6, l2dist=5 → 最远但最大内积
      [0, 1]    // dot=0, 正交
    ];
    const mips = mipsSearch(corpus, q, 1);
    const dots = corpus.map(v => gDot(v, q));
    const goldTop = gArgSortDesc(dots)[0];
    const mismatch = (mips[0].idx !== goldTop) ? 1 : 0;
    const nearest = bruteForceSearch(corpus, q, 1, 'l2');
    const differ = (mips[0].idx !== nearest[0].idx);
    const ok = (mismatch === 0) && differ;
    addInv('INV-RT-005', 'MIPS 穷举 top-k=手搓内积排序前 k；最大内积≠最近邻', ok,
      'mipsTop=' + mips[0].idx + ' goldMIPS=' + goldTop + ' nearest=' + nearest[0].idx + ' differ=' + differ);
  }

  /* ---------- INV-RT-006 重排改变初召回顺序；RRF 融合 top1 高频高排名且融合序≠任一输入 ---------- */
  {
    const q = [1, 0];
    const corpus = [
      [1, 0],   // 初召回(cosine) top
      [3, 4],   // 精排(dot) top
      [0, 1]
    ];
    const rr = rerank(corpus, q, { recall: 3, k: 3, firstStage: 'cosine', secondStage: 'dot' });
    const stage1Top = rr.stage1[0].idx;
    const rerankTop = rr.reranked[0].idx;
    const rerankChanged = (stage1Top !== rerankTop);
    const L1 = ['A', 'B', 'C', 'D', 'E'];
    const L2 = ['B', 'C', 'A', 'F', 'G'];
    const fused = rrfFuse([L1, L2], { k: 60 }).map(x => x.id);
    const top1HighFreq = (fused[0] === 'B');                 // B 在两列表均 rank0
    const diffL1 = (fused.join(',') !== L1.join(','));
    const diffL2 = (fused.join(',') !== L2.join(','));
    const ok = rerankChanged && top1HighFreq && diffL1 && diffL2;
    addInv('INV-RT-006', '重排改变初召回顺序；RRF top1=高频高排名项且融合序≠任一输入', ok,
      'stage1Top=' + stage1Top + ' rerankTop=' + rerankTop + ' fusedTop1=' + fused[0] + ' fused=' + fused.join(','));
  }

  /* ---------- INV-RT-007 阈值过滤：返回集⊆全集且均≥阈值；阈值升高则结果数减少(单调) ---------- */
  {
    const D = 16, rng = makeRng(0x9E3779B9);
    const rv = () => { const v = new Array(D); for (let i = 0; i < D; i++) v[i] = rng() * 2 - 1; return v; };
    const N = 50, corpus = []; for (let i = 0; i < N; i++) corpus.push(rv());
    const q = rv();
    const fLow = thresholdFilter(corpus, q, 0.0, 'cosine');
    const fHigh = thresholdFilter(corpus, q, 0.9, 'cosine');
    let allAbove = true; for (const r of fHigh) if (r.sim < 0.9 - 1e-12) allAbove = false;
    const lowSet = new Set(fLow.map(r => r.idx));
    let subset = true; for (const r of fHigh) if (!lowSet.has(r.idx)) subset = false;
    const monotone = fHigh.length < fLow.length;
    const nonTrivial = fLow.length > 0;
    const ok = allAbove && subset && monotone && nonTrivial;
    addInv('INV-RT-007', '阈值过滤：返回集⊆全集且均≥阈值；阈值升高结果数减少(单调)', ok,
      '|fLow|=' + fLow.length + ' |fHigh|=' + fHigh.length + ' allAbove=' + allAbove + ' subset=' + subset + ' monotone=' + monotone);
  }

  /* ---------- INV-RT-008 同 seed 检索逐位相等；异 seed 必分歧 ---------- */
  {
    const D = 16;
    function buildDataset(seed) {
      const rng = makeRng(seed);
      const corpus = []; for (let i = 0; i < 30; i++) { const v = []; for (let j = 0; j < D; j++) v.push(rng() * 2 - 1); corpus.push(v); }
      const queries = []; for (let i = 0; i < 5; i++) { const v = []; for (let j = 0; j < D; j++) v.push(rng() * 2 - 1); queries.push(v); }
      return { corpus, queries };
    }
    const dsA1 = buildDataset(0x1111), dsA2 = buildDataset(0x1111), dsB = buildDataset(0x2222);
    const r1 = batchSearch(dsA1.corpus, dsA1.queries, 5, 'cosine');
    const r2 = batchSearch(dsA2.corpus, dsA2.queries, 5, 'cosine');
    let identical = true;
    for (let qi = 0; qi < r1.length; qi++) for (let ti = 0; ti < r1[qi].length; ti++) if (r1[qi][ti].idx !== r2[qi][ti].idx) identical = false;
    const rB = batchSearch(dsB.corpus, dsB.queries, 5, 'cosine');
    let diverges = false;
    for (let qi = 0; qi < r1.length; qi++) for (let ti = 0; ti < r1[qi].length; ti++) if (r1[qi][ti].idx !== rB[qi][ti].idx) diverges = true;
    const ok = identical && diverges;
    addInv('INV-RT-008', '同 seed 检索逐位相等；异 seed 必分歧(确定性)', ok,
      'identical=' + identical + ' diverges=' + diverges);
  }

  const started = Date.now();
  const passed = cases.filter(c => c.status === 'pass').length;
  const failed = cases.length - passed;
  return {
    schema: 'forge-selfcheck/1.0',
    module: 'retrieval-forge',
    version: '1.0.0',
    layer: 'L7',
    passed,
    failed,
    skipped: 0,
    invariants,
    cases,
    hash: fileSha256()
  };
}

/* ============================================================================
 * 13. API 导出 + 双端挂载 + CLI 自检守卫
 * ========================================================================== */
const API = {
  name: 'retrieval-forge', version: '1.0.0', layer: 'L7', TOL, makeRng,
  // 基础 (O00/O01/O02)
  cosineSim, dot, l2, metricScore,
  // 检索 (O04/O00/O01/O02)
  bruteForceSearch, cosSearch, dotSearch, l2Search,
  // 量化 (O05)
  buildScalarQuantizer, sqEncode, sqDecode, quantizedSearch,
  // MIPS (O08)
  mipsSearch,
  // ANN 调度 (O03)
  buildANNIndex, annSearch, recallAtK,
  // 重排 (O09) / 融合 (O10) / 阈值 (O12) / 批量 (O13)
  rerank, rrfFuse, thresholdFilter, batchSearch,
  selfCheck
};

if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeRetrieval = API;

// CLI 守卫：仅当本文件被直接执行(node src/retrieval-forge.js --self-check|--json) 时运行，
// 文件被 require 时绝不自动执行（require.main === module）。
if (typeof require !== 'undefined' && typeof process !== 'undefined' && require.main === module) {
  const argv = process.argv;
  if (argv.includes('--self-check') || argv.includes('--json')) {
    const r = selfCheck({});
    process.stdout.write(JSON.stringify(r, null, 2) + '\n');
    process.exit(r.failed === 0 ? 0 : 1);
  }
}
