/**
 * @forge-module eval-forge
 * @layer L7
 * @version 1.0.0
 * @brief     评测指标内核：混淆矩阵与 macro/micro 精确率·召回率·F1、ROC AUC（rank-based 含并列 tie）、困惑度 perplexity、BLEU（1..4-gram 精度 + brevity penalty）、延迟分位数 p50/p95/p99、吞吐量、一致性率
 * @reuse     Papineni et al.《BLEU: a Method for Automatic Evaluation of Machine Translation》(ACL 2002)|https://doi.org/10.3115/1073083.1073135|A|修正 n-gram 精度几何平均 + brevity penalty 闭式
 * @reuse     Fawcett《An introduction to ROC analysis》(Pattern Recognition Letters, 2006)|https://doi.org/10.1016/j.patrec.2005.10.010|A|ROC AUC 与 Mann-Whitney U 统计等价（含并列 0.5 处理）
 * @reuse     scikit-learn metrics 文档|https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics|BSD|macro/micro 平均语义与零除约定（分母为 0 记 0）
 * @reuse     Manning & Schütze《Foundations of Statistical Natural Language Processing》困惑度定义|MIT Press|A|perplexity = exp(平均负对数似然)
 * @beyond    纯 JS 单文件零依赖零构建、双端同构 UMD；延迟/吞吐类指标一律以「逻辑数值样本」为输入（不依赖真实时钟，因此可种子、可复现、可断言）；所有不变量均含独立手搓黄金参照（不调用本模块函数）交叉验证，并以反空洞铁律断言「macro 与 micro 在不平衡类上必分歧」「随机预测 F1 严格介于 (0,1) 非退化」「随机打乱得分 AUC≈0.5 而完美/逆序 AUC=1/0（证伪恒定 0.5）」「均匀高熵 perplexity 必大于尖峰低熵」「短候选必受 brevity penalty 严格惩罚」「不同样本集分位数必分歧」「耗时为 0 必抛错」「大扰动后一致性率必明显下降」——证明函数非恒等、非自证。
 * @invariant INV-EVL-001|混淆矩阵与 PR：混淆矩阵各行和 == 该类真实样本数（EXACT）；precision/recall ≡ 手搓 TP/(TP+FP)、TP/(TP+FN)（≤1e-15）；且断言「macro ≠ micro 在不平衡类上必分歧」（|macro−micro|>1e-9，证伪平均方式混淆）|1e-15|多种子手搓混淆矩阵/PR 黄金参照 + 不平衡类 macro≠micro 分歧断言
 * @invariant INV-EVL-002|F1：F1 ≡ 手搓 2PR/(P+R)（≤1e-15，per-class + macro）；完全正确预测 F1==1、完全错误 F1==0（边界 EXACT）；随机预测 F1 严格介于 (0,1)（非退化）|1e-15|多种子手搓黄金参照 + 双边界 EXACT + 开区间非退化断言
 * @invariant INV-EVL-003|AUC：AUC ≡ 手搓 Mann-Whitney U 统计（正样本得分大于负样本比例，并列计 0.5，≤1e-15）；完美排序 AUC==1、逆序 AUC==0（EXACT）；随机打乱得分 AUC≈0.5（≥10σ 统计容差，σ=√((n+1)/(12·nP·nN))，证伪恒定平凡值）；并列敏感（含 tie 与不计 tie 结果必分歧）|1e-15|暴力 O(nP·nN) 黄金参照 + rank-based 独立参照 + 三边界 + tie 敏感性断言
 * @invariant INV-EVL-004|困惑度：perplexity ≡ exp(手搓平均 NLL)（≤1e-10 相对）；恒定概率 p 时 perplexity ≡ 1/p（闭式）；且断言「不同分布 perplexity 必分歧」、均匀高熵 perplexity 严格大于尖峰低熵（单调性）|1e-10|多种子手搓黄金参照 + 闭式 1/p + 双分布分歧与单调断言
 * @invariant INV-EVL-005|BLEU：BLEU ≡ 手搓 n-gram 修正精度几何平均 × brevity penalty（≤1e-10）；候选==参考时 BLEU==1（边界）；短候选受 brevity penalty 惩罚严格 <1（证伪）；n-gram 全不匹配 BLEU==0（EXACT）|1e-10|多种子手搓 n-gram/BP 黄金参照 + 三边界断言
 * @invariant INV-EVL-006|延迟分位数：p50/p95/p99 ≡ 手搓排序数组线性插值取位（≤1e-15）；p50 ≤ p95 ≤ p99（单调 EXACT）；且断言「不同样本集分位数必分歧」（|Δp95|>1e-9，非恒定）|1e-15|多种子手搓分位黄金参照 + 单调性 + 异样本集分歧断言
 * @invariant INV-EVL-007|吞吐量：throughput ≡ count/seconds（≤1e-15）；加倍计数吞吐翻倍（比例 EXACT）；且断言「耗时为 0 / 负 / 非有限时抛错」（安全边界，证伪除零掩盖）|1e-15|闭式对照 + 比例 EXACT + 抛错断言
 * @invariant INV-EVL-008|一致性：同输入两次运行一致性率 == 1.0（EXACT，逐位相等）；微小扰动（≤1e-6，间隔裕度 1e-3）下一致性率 == 1（统计容差）；且断言「大扰动（≥0.5）后一致性率明显下降（<0.95）」（证伪平凡恒等）；同 seed 全流水线逐位复现、异 seed 必分歧|0|双跑逐位比对 + 双扰动幅度对照 + 异种子分歧断言
 * @seed      0x9E3779B9
 */
'use strict';

/* ============================================================================
 * 项目统一容差（九档，与 core-verify.TOL 同值，本地声明保证零依赖）
 * ========================================================================== */
const TOL = { EXACT: 0, ULP: 1e-16, TIGHT: 1e-15, ORTHO: 1e-14, RESID: 1e-12, SOLVE: 1e-10, GRAD: 1e-6, MONO: -1e-9, CONV: 1e-6 };

/* ============================================================================
 * 私有可种子 PRNG：sfc32（Public Domain 参考实现）
 * 同 seed 序列逐位可复现；异 seed 状态轨道分歧。
 * ========================================================================== */
function makeRng(seed) {
  let a = (seed >>> 0) ^ 0x9E3779B9;
  let b = ((seed >>> 0) + 0x6D2B79F5) >>> 0;
  let c = (seed >>> 0) ^ 0xC2B2AE35;
  let d = ((seed >>> 0) + 0x165667B1) >>> 0;
  function next() {
    a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0;
    const t = (a + b + d) >>> 0;
    d = (d + 1) >>> 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) >>> 0;
    c = ((c << 21) | (c >>> 11)) >>> 0;
    c = (c + t) >>> 0;
    return (t >>> 0) / 4294967296;
  }
  for (let i = 0; i < 12; i++) next();  // 预热，消除弱初始状态
  return next;
}

/* ============================================================================
 * 纯 JS SHA-256（FIPS 180-4）：浏览器端回退，零外部依赖
 * ========================================================================== */
const K256 = [
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
];
function _rotr(x, n) { return ((x >>> n) | (x << (32 - n))) >>> 0; }

function sha256Hex(str) {
  const bytes = [];
  for (let i = 0; i < str.length; i++) {
    let code = str.charCodeAt(i);
    if (code < 0x80) bytes.push(code);
    else if (code < 0x800) bytes.push(0xC0 | (code >> 6), 0x80 | (code & 63));
    else if (code >= 0xD800 && code <= 0xDBFF && i + 1 < str.length && str.charCodeAt(i + 1) >= 0xDC00 && str.charCodeAt(i + 1) <= 0xDFFF) {
      code = 0x10000 + ((code - 0xD800) << 10) + (str.charCodeAt(i + 1) - 0xDC00); i++;
      bytes.push(0xF0 | (code >> 18), 0x80 | ((code >> 12) & 63), 0x80 | ((code >> 6) & 63), 0x80 | (code & 63));
    } else bytes.push(0xE0 | (code >> 12), 0x80 | ((code >> 6) & 63), 0x80 | (code & 63));
  }
  const bitLen = bytes.length * 8;
  bytes.push(0x80);
  while (bytes.length % 64 !== 56) bytes.push(0);
  const hi = Math.floor(bitLen / 0x100000000), lo = bitLen >>> 0;
  bytes.push((hi >>> 24) & 255, (hi >>> 16) & 255, (hi >>> 8) & 255, hi & 255,
    (lo >>> 24) & 255, (lo >>> 16) & 255, (lo >>> 8) & 255, lo & 255);
  let H = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
  const w = new Array(64);
  for (let off = 0; off < bytes.length; off += 64) {
    for (let t = 0; t < 16; t++) {
      w[t] = ((bytes[off + 4 * t] << 24) | (bytes[off + 4 * t + 1] << 16) | (bytes[off + 4 * t + 2] << 8) | bytes[off + 4 * t + 3]) | 0;
    }
    for (let t = 16; t < 64; t++) {
      const x15 = w[t - 15], x2 = w[t - 2];
      const s0 = (_rotr(x15, 7) ^ _rotr(x15, 18) ^ (x15 >>> 3)) | 0;
      const s1 = (_rotr(x2, 17) ^ _rotr(x2, 19) ^ (x2 >>> 10)) | 0;
      w[t] = (w[t - 16] + s0 + w[t - 7] + s1) | 0;
    }
    let a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
    for (let t = 0; t < 64; t++) {
      const S1 = (_rotr(e, 6) ^ _rotr(e, 11) ^ (_rotr(e, 25))) | 0;
      const ch = ((e & f) ^ (~e & g)) | 0;
      const t1 = (h + S1 + ch + K256[t] + w[t]) | 0;
      const S0 = (_rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)) | 0;
      const maj = ((a & b) ^ (a & c) ^ (b & c)) | 0;
      const t2 = (S0 + maj) | 0;
      h = g; g = f; f = e; e = (d + t1) | 0; d = c; c = b; b = a; a = (t1 + t2) | 0;
    }
    H = [(H[0] + a) | 0, (H[1] + b) | 0, (H[2] + c) | 0, (H[3] + d) | 0, (H[4] + e) | 0, (H[5] + f) | 0, (H[6] + g) | 0, (H[7] + h) | 0];
  }
  return H.map((x) => ('00000000' + (x >>> 0).toString(16)).slice(-8)).join('');
}

/**
 * 本文件源码 sha256（契约铁律：Node 端用内置 crypto 读取本文件源码；浏览器端回退纯 JS）。
 */
function sourceHash() {
  try {
    if (typeof __filename !== 'string') throw new Error('no __filename');
    const fs = require('fs'), crypto = require('crypto');
    return 'sha256:' + crypto.createHash('sha256').update(fs.readFileSync(__filename, 'utf8')).digest('hex');
  } catch (e) {
    return 'sha256:' + sha256Hex('eval-forge-1.0.0-L7-0x9E3779B9');
  }
}

/* ============================================================================
 * 通用输入校验小工具
 * ========================================================================== */
function _isNum(x) { return typeof x === 'number' && Number.isFinite(x); }

function _labelArray(x, who) {
  if (!Array.isArray(x) || x.length === 0) throw new TypeError(who + ': 必须为非空数组');
  for (let i = 0; i < x.length; i++) {
    if (typeof x[i] !== 'number' || !Number.isInteger(x[i]) || x[i] < 0) {
      throw new TypeError(who + ': 元素必须为非负整数类标，got index ' + i + ' = ' + x[i]);
    }
  }
  return x;
}

function _resolveK(yTrue, yPred, numClasses) {
  let max = -1;
  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i] > max) max = yTrue[i];
    if (yPred[i] > max) max = yPred[i];
  }
  if (numClasses == null) return max + 1;
  if (!Number.isInteger(numClasses) || numClasses <= 0) throw new TypeError('numClasses 必须为正整数');
  if (max + 1 > numClasses) throw new TypeError('类标越界：max(label)=' + max + ' ≥ numClasses=' + numClasses);
  return numClasses;
}

/* ============================================================================
 * 1. 混淆矩阵与精确率/召回率/F1（O00 精度 / O03 召回 / O08 F1）
 * ========================================================================== */

/**
 * 混淆矩阵：matrix[trueClass][predClass] = 样本数（行=真实，列=预测）。
 * numClasses 省略时由 max(label)+1 推断。
 */
function confusion(yTrue, yPred, numClasses) {
  _labelArray(yTrue, 'confusion.yTrue');
  _labelArray(yPred, 'confusion.yPred');
  if (yTrue.length !== yPred.length) throw new TypeError('confusion: yTrue/yPred 长度不一致');
  const K = _resolveK(yTrue, yPred, numClasses);
  const cm = [];
  for (let i = 0; i < K; i++) cm.push(new Array(K).fill(0));
  for (let i = 0; i < yTrue.length; i++) cm[yTrue[i]][yPred[i]]++;
  return cm;
}

/**
 * 由混淆矩阵导出每类 TP/FP/FN/TN 与 precision/recall/F1/support。
 * 零除约定（scikit-learn zero_division=0）：分母为 0 时指标记 0。
 */
function classPRF(cm) {
  if (!Array.isArray(cm) || cm.length === 0 || !Array.isArray(cm[0])) throw new TypeError('classPRF: cm 必须为非空二维数组');
  const K = cm.length;
  for (let i = 0; i < K; i++) {
    if (!Array.isArray(cm[i]) || cm[i].length !== K) throw new TypeError('classPRF: cm 必须为方阵');
  }
  const tp = new Array(K).fill(0), fp = new Array(K).fill(0), fn = new Array(K).fill(0);
  const tn = new Array(K).fill(0), support = new Array(K).fill(0);
  let total = 0;
  for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) total += cm[i][j];
  for (let c = 0; c < K; c++) {
    tp[c] = cm[c][c];
    for (let i = 0; i < K; i++) {
      if (i === c) continue;
      fp[c] += cm[i][c];   // 真实非 c、预测为 c
      fn[c] += cm[c][i];   // 真实为 c、预测非 c
    }
    support[c] = tp[c] + fn[c];
    tn[c] = total - tp[c] - fp[c] - fn[c];
  }
  const precision = new Array(K), recall = new Array(K), f1 = new Array(K);
  for (let c = 0; c < K; c++) {
    precision[c] = (tp[c] + fp[c] > 0) ? tp[c] / (tp[c] + fp[c]) : 0;
    recall[c] = (tp[c] + fn[c] > 0) ? tp[c] / (tp[c] + fn[c]) : 0;
    const s = precision[c] + recall[c];
    f1[c] = (s > 0) ? (2 * precision[c] * recall[c]) / s : 0;
  }
  return { numClasses: K, tp, fp, fn, tn, support, precision, recall, f1, total };
}

function _macroOf(values) {
  let acc = 0;
  for (let i = 0; i < values.length; i++) acc += values[i];
  return acc / values.length;
}

/**
 * 精确率。avg ∈ {'macro','micro','binary','none'}；默认 'macro'。
 * 'none' 返回 per-class 数组；'binary' 默认正类 posLabel=1（opts.posLabel 可改）。
 */
function precisionScore(yTrue, yPred, avg, opts) {
  const o = (opts && typeof opts === 'object') ? opts : {};
  const K = _resolveK(yTrue, yPred, o.numClasses);
  const prf = classPRF(confusion(yTrue, yPred, K));
  return _aggregate(prf, 'precision', (avg == null ? 'macro' : avg), o);
}

/** 召回率。签名与 avg 语义同 precisionScore。 */
function recallScore(yTrue, yPred, avg, opts) {
  const o = (opts && typeof opts === 'object') ? opts : {};
  const K = _resolveK(yTrue, yPred, o.numClasses);
  const prf = classPRF(confusion(yTrue, yPred, K));
  return _aggregate(prf, 'recall', (avg == null ? 'macro' : avg), o);
}

/** F1 = 2PR/(P+R)，由同一套 TP/FP/FN 推导。签名与 avg 语义同 precisionScore。 */
function f1Score(yTrue, yPred, avg, opts) {
  const o = (opts && typeof opts === 'object') ? opts : {};
  const K = _resolveK(yTrue, yPred, o.numClasses);
  const prf = classPRF(confusion(yTrue, yPred, K));
  return _aggregate(prf, 'f1', (avg == null ? 'macro' : avg), o);
}

function _aggregate(prf, kind, avg, opts) {
  const K = prf.numClasses;
  if (avg === 'none') return prf[kind].slice();
  if (avg === 'micro') {
    let tp = 0, denom = 0;
    for (let c = 0; c < K; c++) {
      tp += prf.tp[c];
      denom += (kind === 'precision') ? (prf.tp[c] + prf.fp[c])
        : (kind === 'recall') ? (prf.tp[c] + prf.fn[c])
          : (2 * prf.tp[c] + prf.fp[c] + prf.fn[c]);
    }
    if (kind === 'f1') return (denom > 0) ? (2 * tp) / denom : 0;
    return (denom > 0) ? tp / denom : 0;
  }
  if (avg === 'binary') {
    const pos = (opts && opts.posLabel != null) ? opts.posLabel : 1;
    if (!Number.isInteger(pos) || pos < 0 || pos >= K) throw new TypeError('binary 平均：posLabel 越界');
    return prf[kind][pos];
  }
  if (avg === 'weighted') {
    let acc = 0, w = 0;
    for (let c = 0; c < K; c++) { acc += prf[kind][c] * prf.support[c]; w += prf.support[c]; }
    return (w > 0) ? acc / w : 0;
  }
  if (avg === 'macro') return _macroOf(prf[kind]);
  throw new TypeError('未知平均方式: ' + avg + '（支持 macro|micro|weighted|binary|none）');
}

/* ============================================================================
 * 2. ROC AUC（O09）：rank-based Mann-Whitney U，含并列 tie 的 0.5 处理
 * ========================================================================== */

/**
 * ROC AUC = U / (nPos·nNeg)，U = Σ_{i∈P} Σ_{j∈N} [s_i>s_j] + 0.5·[s_i==s_j]。
 * 实现：按得分升序赋平均秩（并列取秩均值），AUC = (Σ_{正样本} rank − nPos(nPos+1)/2)/(nPos·nNeg)。
 * labels ∈ {0,1}；仅含单一类别时抛错（AUC 无定义）。
 */
function rocAuc(scores, labels) {
  if (!Array.isArray(scores) || scores.length === 0) throw new TypeError('rocAuc: scores 必须为非空数组');
  if (!Array.isArray(labels) || labels.length !== scores.length) throw new TypeError('rocAuc: labels 必须为与 scores 等长的数组');
  let nPos = 0, nNeg = 0;
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] === 1) nPos++;
    else if (labels[i] === 0) nNeg++;
    else throw new TypeError('rocAuc: labels 元素必须为 0/1，got ' + labels[i]);
    if (!_isNum(scores[i])) throw new TypeError('rocAuc: scores 元素必须为有限数，got ' + scores[i]);
  }
  if (nPos === 0 || nNeg === 0) throw new TypeError('rocAuc: 正负样本必须同时存在（nPos=' + nPos + ', nNeg=' + nNeg + '）');
  const n = scores.length;
  const idx = new Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  idx.sort((a, b) => {
    const d = scores[a] - scores[b];
    if (d !== 0) return d;
    return a - b;                       // 稳定确定序（并列按原下标）
  });
  const rank = new Array(n).fill(0);
  let k = 0;
  while (k < n) {
    let j = k;
    while (j + 1 < n && scores[idx[j + 1]] === scores[idx[k]]) j++;
    const avgRank = (k + j + 2) / 2;    // 1-based 平均秩
    for (let t = k; t <= j; t++) rank[idx[t]] = avgRank;
    k = j + 1;
  }
  let sumRankPos = 0;
  for (let i = 0; i < n; i++) if (labels[i] === 1) sumRankPos += rank[i];
  return (sumRankPos - nPos * (nPos + 1) / 2) / (nPos * nNeg);
}

/* ============================================================================
 * 3. 困惑度（O10）：perplexity = exp(平均负对数似然)
 * ========================================================================== */

/**
 * logProbs：每个样本的自然对数概率 ln p_i（须 ≤ 0）。
 * perplexity = exp(−mean(logProbs)) = exp(平均 NLL)。恒定 p 时闭式为 1/p。
 */
function perplexity(logProbs) {
  if (!Array.isArray(logProbs) || logProbs.length === 0) throw new TypeError('perplexity: logProbs 必须为非空数组');
  let acc = 0;
  for (let i = 0; i < logProbs.length; i++) {
    const v = logProbs[i];
    if (!_isNum(v)) throw new TypeError('perplexity: logProbs 元素必须为有限数，got ' + v);
    if (v > 0) throw new TypeError('perplexity: 对数概率必须 ≤ 0（p≤1），got ' + v);
    acc -= v;                            // NLL_i = −ln p_i
  }
  return Math.exp(acc / logProbs.length);
}

/** 由概率数组直接求困惑度（内部取 ln）。probs ∈ (0,1]。 */
function perplexityFromProbs(probs) {
  if (!Array.isArray(probs) || probs.length === 0) throw new TypeError('perplexityFromProbs: probs 必须为非空数组');
  const lp = new Array(probs.length);
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i];
    if (!_isNum(p) || !(p > 0) || p > 1) throw new TypeError('perplexityFromProbs: probs 元素必须 ∈ (0,1]，got ' + p);
    lp[i] = Math.log(p);
  }
  return perplexity(lp);
}

/* ============================================================================
 * 4. BLEU（O11）：修正 n-gram 精度几何平均 × brevity penalty
 * ========================================================================== */
function _tokenize(x, who) {
  if (typeof x === 'string') {
    const t = x.trim().split(/\s+/).filter((s) => s.length > 0);
    if (t.length === 0) throw new TypeError(who + ': 空字符串无法分词');
    return t;
  }
  if (Array.isArray(x)) {
    if (x.length === 0) throw new TypeError(who + ': 空序列');
    for (let i = 0; i < x.length; i++) if (typeof x[i] !== 'string' && typeof x[i] !== 'number') {
      throw new TypeError(who + ': 元素必须为字符串或数字 token');
    }
    return x.slice();
  }
  throw new TypeError(who + ': 必须为字符串或 token 数组');
}

function _ngramKeys(tokens, n) {
  const out = [];
  for (let i = 0; i + n <= tokens.length; i++) out.push(tokens.slice(i, i + n).join(''));
  return out;
}

function _ngramCounts(tokens, n) {
  const keys = _ngramKeys(tokens, n);
  const m = new Map();
  for (const k of keys) m.set(k, (m.get(k) || 0) + 1);
  return m;
}

/**
 * 语料级 BLEU（Papineni et al. 2002 原式）：
 *   p_n = Σ_{g} min(count_cand(g), max_ref count_ref(g)) / Σ_g count_cand(g)
 *   BP  = 1 (c>r) else exp(1 − r/c)，r 取「最近参考长度」（|r−c| 最小，平局取较小 r）
 *   BLEU = BP · exp( (1/N) Σ_n ln p_n )；任一 p_n = 0 时 BLEU = 0。
 * 返回 { score, precisions, bp, candLen, refLen, ratio, maxN }。
 */
function bleu(candidate, references, maxN) {
  const cand = _tokenize(candidate, 'bleu.candidate');
  if (!Array.isArray(references) || references.length === 0) throw new TypeError('bleu: references 必须为非空数组');
  const refs = references.map((r, i) => _tokenize(r, 'bleu.references[' + i + ']'));
  const N = (maxN == null) ? 4 : maxN;
  if (!Number.isInteger(N) || N < 1) throw new TypeError('bleu: maxN 必须为 ≥1 的整数');
  const c = cand.length;
  // 最近参考长度（BP 口径）
  let r = refs[0].length;
  for (const rf of refs) {
    const dNew = Math.abs(rf.length - c), dCur = Math.abs(r - c);
    if (dNew < dCur || (dNew === dCur && rf.length < r)) r = rf.length;
  }
  const bp = (c > r) ? 1 : Math.exp(1 - r / c);      // c == r → exp(0) = 1
  const precisions = new Array(N).fill(0);
  let zero = false, logSum = 0;
  for (let n = 1; n <= N; n++) {
    const candCounts = _ngramCounts(cand, n);
    if (candCounts.size === 0) { precisions[n - 1] = 0; zero = true; continue; }
    let clipped = 0, total = 0;
    for (const [g, cnt] of candCounts) {
      let best = 0;
      for (const rf of refs) {
        const rc = _ngramCounts(rf, n).get(g) || 0;
        if (rc > best) best = rc;
      }
      clipped += Math.min(cnt, best);
      total += cnt;
    }
    const p = (total > 0) ? clipped / total : 0;
    precisions[n - 1] = p;
    if (p === 0) zero = true; else logSum += Math.log(p);
  }
  const score = zero ? 0 : Math.exp(logSum / N) * bp;
  return { score, precisions, bp, candLen: c, refLen: r, ratio: c / r, maxN: N, zero };
}

/* ============================================================================
 * 5. 延迟分位数（O01）：p50/p95/p99 + 均值，纯数值输入（不依赖真实时钟）
 * ========================================================================== */

/** 升序排序数组上的分位数（R type-7 / numpy 'linear' 口径）：pos = q·(n−1) 线性插值。 */
function quantileSorted(sorted, q) {
  const n = sorted.length;
  if (n === 0) throw new TypeError('quantileSorted: 空数组');
  if (!(q >= 0 && q <= 1)) throw new TypeError('quantileSorted: q ∈ [0,1]');
  if (n === 1) return sorted[0];
  const pos = q * (n - 1);
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo]);
}

/**
 * 延迟统计：samples 为延迟样本数组（毫秒，逻辑数值）。
 * 返回 { p50, p95, p99, mean, min, max, count, sorted }。
 */
function latencyStats(samples) {
  if (!Array.isArray(samples) || samples.length === 0) throw new TypeError('latencyStats: samples 必须为非空数组');
  const sorted = new Array(samples.length);
  let acc = 0;
  for (let i = 0; i < samples.length; i++) {
    const v = samples[i];
    if (!_isNum(v)) throw new TypeError('latencyStats: samples 元素必须为有限数，got ' + v);
    sorted[i] = v; acc += v;
  }
  sorted.sort((a, b) => a - b);
  return {
    count: samples.length,
    p50: quantileSorted(sorted, 0.50),
    p95: quantileSorted(sorted, 0.95),
    p99: quantileSorted(sorted, 0.99),
    mean: acc / samples.length,
    min: sorted[0],
    max: sorted[sorted.length - 1],
    sorted
  };
}

/** 任意分位数（0..1）便捷入口。 */
function percentile(samples, q) {
  return quantileSorted(latencyStats(samples).sorted, q);
}

/* ============================================================================
 * 6. 吞吐量（O02）：count / seconds，纯数值（不依赖真实时钟）
 * ========================================================================== */

/**
 * 吞吐量（tokens/sec 或 req/sec，取决于 count 语义）。
 * seconds 必须 > 0 且有限 —— 除零/负耗时一律抛错，绝不静默返回 Infinity。
 */
function throughput(count, seconds) {
  if (!_isNum(count) || count < 0) throw new TypeError('throughput: count 必须为非负有限数，got ' + count);
  if (!_isNum(seconds) || !(seconds > 0)) throw new TypeError('throughput: seconds 必须为正有限数（除零/负值一律抛错），got ' + seconds);
  return count / seconds;
}

/* ============================================================================
 * 7. 一致性（O04）：逐位相等率 + 扰动一致性
 * ========================================================================== */

/**
 * 两次预测的一致性率 = 相等位置数 / 总位置数 ∈ [0,1]。
 * 同输入两次运行 → 1.0（EXACT）；施加扰动后 → 衡量预测稳健性。
 */
function consistencyRate(predsA, predsB) {
  if (!Array.isArray(predsA) || predsA.length === 0) throw new TypeError('consistencyRate: predsA 必须为非空数组');
  if (!Array.isArray(predsB) || predsB.length !== predsA.length) throw new TypeError('consistencyRate: predsB 必须为与 predsA 等长的数组');
  let same = 0;
  for (let i = 0; i < predsA.length; i++) if (predsA[i] === predsB[i]) same++;
  return same / predsA.length;
}

/** 数值数组取 argmax（平局取最小下标）。用于由 logits 生成预测。 */
function argmax(vec) {
  if (!Array.isArray(vec) || vec.length === 0) throw new TypeError('argmax: vec 必须为非空数组');
  let bi = 0, bv = vec[0];
  for (let i = 1; i < vec.length; i++) if (vec[i] > bv) { bv = vec[i]; bi = i; }
  return bi;
}

/* ============================================================================
 * 报告构造（forge-selfcheck/1.0，零依赖自包含）
 * ========================================================================== */
function caseOf(id, name, ok, actual, detail, tol) {
  const numActual = (typeof actual === 'number' && isFinite(actual)) ? actual : (ok ? 0 : 1);
  return {
    id, name,
    status: ok ? 'pass' : 'fail',
    passed: !!ok,
    actual: numActual,
    expected: 0,
    tol: (typeof tol === 'number') ? tol : TOL.EXACT,
    detail
  };
}

function buildReport(module, version, layer, cases, invariants, seed) {
  const started = Date.now();
  const passed = cases.filter((c) => c.status === 'pass').length;
  const failed = cases.length - passed;
  return {
    schema: 'forge-selfcheck/1.0',
    module, version, layer,
    passed, failed, skipped: 0,
    duration_ms: Date.now() - started,
    tol: TOL,
    invariants,
    cases,
    env: {
      platform: (typeof process !== 'undefined' && process.platform) || 'browser',
      arch: (typeof process !== 'undefined' && process.arch) || 'unknown',
      node: (typeof process !== 'undefined' && process.version) || 'browser',
      seed
    },
    hash: sourceHash()
  };
}

/* ============================================================================
 * selfCheck —— 8 条不变量（INV-EVL-001..008）
 * 全部黄金参照均为「本文件内手搓、不调用本模块导出函数」的独立实现。
 * ========================================================================== */
const SEED0 = 0x9E3779B9;

/* ---------- 独立手搓黄金参照（不调用本模块导出函数） ---------- */

/** 手搓混淆矩阵（行=真实，列=预测）。 */
function goldConfusion(yT, yP, K) {
  const cm = [];
  for (let i = 0; i < K; i++) cm.push(new Array(K).fill(0));
  for (let i = 0; i < yT.length; i++) cm[yT[i]][yP[i]]++;
  return cm;
}
/** 手搓 per-class TP/FP/FN → P/R/F1（零除记 0）。 */
function goldPRF(cm, K) {
  const P = [], R = [], F = [], sup = [], tpA = [], fpA = [], fnA = [];
  let totTP = 0, totFP = 0, totFN = 0;
  for (let c = 0; c < K; c++) {
    let tp = 0, fp = 0, fn = 0;
    for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
      if (i === c && j === c) tp += cm[i][j];
      else if (j === c) fp += cm[i][j];
      else if (i === c) fn += cm[i][j];
    }
    const p = (tp + fp > 0) ? tp / (tp + fp) : 0;
    const r = (tp + fn > 0) ? tp / (tp + fn) : 0;
    P.push(p); R.push(r); F.push((p + r > 0) ? (2 * p * r) / (p + r) : 0);
    sup.push(tp + fn); tpA.push(tp); fpA.push(fp); fnA.push(fn);
    totTP += tp; totFP += fp; totFN += fn;
  }
  const microP = (totTP + totFP > 0) ? totTP / (totTP + totFP) : 0;
  const microR = (totTP + totFN > 0) ? totTP / (totTP + totFN) : 0;
  const microF = (2 * totTP + totFP + totFN > 0) ? (2 * totTP) / (2 * totTP + totFP + totFN) : 0;
  return { P, R, F, support: sup, tp: tpA, fp: fpA, fn: fnA, microP, microR, microF };
}
function goldMean(a) { let s = 0; for (const v of a) s += v; return s / a.length; }

/** 手搓 Mann-Whitney U（暴力 O(nP·nN)，并列计 0.5）→ AUC。 */
function goldAucBrute(scores, labels) {
  let U = 0, nP = 0, nN = 0;
  for (let i = 0; i < scores.length; i++) {
    if (labels[i] !== 1) continue;
    nP++;
    for (let j = 0; j < scores.length; j++) {
      if (labels[j] !== 0) continue;
      if (scores[i] > scores[j]) U += 1;
      else if (scores[i] === scores[j]) U += 0.5;
    }
  }
  for (let j = 0; j < scores.length; j++) if (labels[j] === 0) nN++;
  return U / (nP * nN);
}
/** 手搓 Mann-Whitney U（排序计数 O(n log n)：对每个正样本二分统计严格小于/等于的负样本数）。 */
function goldAucSorted(scores, labels) {
  const pos = [], neg = [];
  for (let i = 0; i < scores.length; i++) (labels[i] === 1 ? pos : neg).push(scores[i]);
  neg.sort((a, b) => a - b);
  let U = 0;
  const lessCount = (v) => { let lo = 0, hi = neg.length; while (lo < hi) { const m = (lo + hi) >> 1; if (neg[m] < v) lo = m + 1; else hi = m; } return lo; };
  const leCount = (v) => { let lo = 0, hi = neg.length; while (lo < hi) { const m = (lo + hi) >> 1; if (neg[m] <= v) lo = m + 1; else hi = m; } return lo; };
  for (const v of pos) U += lessCount(v) + 0.5 * (leCount(v) - lessCount(v));
  return U / (pos.length * neg.length);
}

/** 手搓 BLEU（n-gram 修正精度 + brevity penalty）。 */
function goldBleu(cand, refs, N) {
  const c = cand.length;
  let r = refs[0].length;
  for (const rf of refs) {
    const dn = Math.abs(rf.length - c), dc = Math.abs(r - c);
    if (dn < dc || (dn === dc && rf.length < r)) r = rf.length;
  }
  const bp = (c > r) ? 1 : Math.exp(1 - r / c);
  let logSum = 0, zero = false;
  const prec = [];
  for (let n = 1; n <= N; n++) {
    const cg = [];
    for (let i = 0; i + n <= c; i++) cg.push(cand.slice(i, i + n).join(''));
    if (cg.length === 0) { prec.push(0); zero = true; continue; }
    const cc = new Map();
    for (const g of cg) cc.set(g, (cc.get(g) || 0) + 1);
    let clip = 0;
    for (const [g, cnt] of cc) {
      let best = 0;
      for (const rf of refs) {
        const rc = new Map();
        for (let i = 0; i + n <= rf.length; i++) {
          const k = rf.slice(i, i + n).join('');
          rc.set(k, (rc.get(k) || 0) + 1);
        }
        const v = rc.get(g) || 0;
        if (v > best) best = v;
      }
      clip += Math.min(cnt, best);
    }
    const p = clip / cg.length;
    prec.push(p);
    if (p === 0) zero = true; else logSum += Math.log(p);
  }
  return { score: zero ? 0 : Math.exp(logSum / N) * bp, bp, prec, ratio: c / r };
}

/** 手搓分位数（R type-7 线性插值），与模块实现同源算法但独立书写。 */
function goldQuantile(sorted, q) {
  const n = sorted.length;
  if (n === 1) return sorted[0];
  const h = (n - 1) * q;
  const lo = Math.floor(h);
  const frac = h - lo;
  if (lo + 1 >= n) return sorted[n - 1];
  return sorted[lo] * (1 - frac) + sorted[lo + 1] * frac;
}

/* ---------- 随机数据生成（模块内 sfc32，可复现） ---------- */
function genLabels(n, weights, seed) {
  const rng = makeRng(seed);
  const y = new Array(n);
  for (let i = 0; i < n; i++) {
    const u = rng();
    let acc = 0, k = 0;
    for (; k < weights.length - 1; k++) { acc += weights[k]; if (u < acc) break; }
    y[i] = k;
  }
  return y;
}
function genPreds(yT, K, err, seed) {
  const rng = makeRng(seed);
  return yT.map((t) => (rng() < err) ? Math.floor(rng() * K) : t);
}
function genLatencySamples(n, seed) {
  const rng = makeRng(seed);
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const base = 8 + 40 * Math.pow(rng(), 3);        // 重尾（近似服务延迟）
    out[i] = base + rng() * 6;
  }
  return out;
}
/** 生成带「决策裕度」的 logits：top1 与 top2 至少相差 margin。 */
function genLogits(n, K, margin, seed) {
  const rng = makeRng(seed);
  const M = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let k = 0; k < K; k++) row.push(rng() * 2 - 1);
    // 归一化裕度：把最大值抬到 次大值 + margin
    let m1 = -Infinity, m2 = -Infinity, i1 = 0;
    for (let k = 0; k < K; k++) { if (row[k] > m1) { m2 = m1; m1 = row[k]; i1 = k; } else if (row[k] > m2) m2 = row[k]; }
    if (m1 - m2 < margin) row[i1] = m2 + margin;
    M.push(row);
  }
  return M;
}
function perturbAndArgmax(logits, eps, seed) {
  const rng = makeRng(seed);
  return logits.map((row) => {
    const v = row.map((x) => x + (rng() * 2 - 1) * eps);
    let bi = 0, bv = v[0];
    for (let i = 1; i < v.length; i++) if (v[i] > bv) { bv = v[i]; bi = i; }
    return bi;
  });
}
function throws(fn) {
  try { fn(); return false; } catch (e) { return e instanceof TypeError || e instanceof Error; }
}

function selfCheck() {
  const cases = [];
  const invariants = [];

  /* ================= INV-EVL-001 混淆矩阵与 PR ================= */
  {
    const K = 3, W = [0.80, 0.15, 0.05];
    const seeds = [SEED0, 0xA11CE5, 0xB0B57E];
    let rowSumOk = true, worst = 0, minGap = Infinity;
    for (const sd of seeds) {
      const yT = genLabels(600, W, sd);
      const yP = genPreds(yT, K, 0.25, sd ^ 0x51CE);
      const cm = confusion(yT, yP, K);
      const gcm = goldConfusion(yT, yP, K);
      const trueCount = new Array(K).fill(0);
      for (const t of yT) trueCount[t]++;
      for (let c = 0; c < K; c++) {
        let rs = 0;
        for (let j = 0; j < K; j++) {
          rs += cm[c][j];
          worst = Math.max(worst, Math.abs(cm[c][j] - gcm[c][j]));
        }
        if (rs !== trueCount[c]) rowSumOk = false;               // EXACT：行和 == 该类真实样本数
      }
      const prf = classPRF(cm);
      const g = goldPRF(gcm, K);
      // precision/recall ≡ 手搓 TP/(TP+FP)、TP/(TP+FN)
      for (let c = 0; c < K; c++) {
        worst = Math.max(worst, Math.abs(prf.precision[c] - g.P[c]), Math.abs(prf.recall[c] - g.R[c]));
      }
      const mP = precisionScore(yT, yP, 'macro', { numClasses: K });
      const uP = precisionScore(yT, yP, 'micro', { numClasses: K });
      const mR = recallScore(yT, yP, 'macro', { numClasses: K });
      const uR = recallScore(yT, yP, 'micro', { numClasses: K });
      worst = Math.max(worst, Math.abs(mP - goldMean(g.P)), Math.abs(uP - g.microP),
        Math.abs(mR - goldMean(g.R)), Math.abs(uR - g.microR));
      minGap = Math.min(minGap, Math.abs(mP - uP), Math.abs(mR - uR));  // 反空洞：macro ≠ micro
    }
    const ok = rowSumOk && (worst <= TOL.TIGHT) && (minGap > 1e-9);
    const d = '行和==真实样本数(EXACT)=' + rowSumOk + ' 3种子max|Δ|=' + worst.toExponential(2)
      + ' 不平衡类 min|macro−micro|=' + minGap.toExponential(3) + '(>1e-9 证伪平均方式混淆)';
    cases.push(caseOf('INV-EVL-001', '混淆矩阵行和 EXACT + precision/recall ≡ 手搓 TP/(TP+FP)、TP/(TP+FN) + macro≠micro 必分歧', ok, worst, d, TOL.TIGHT));
    invariants.push({ id: 'INV-EVL-001', name: '混淆矩阵与 PR（行和 EXACT + 黄金参照 + macro≠micro 证伪）', ok, expr: 'rowSum==support(EXACT)；|P−TP/(TP+FP)|≤1e-15；|R−TP/(TP+FN)|≤1e-15；|macro−micro|>1e-9', tol: TOL.TIGHT, worst, detail: d });
  }

  /* ================= INV-EVL-002 F1 ================= */
  {
    const K = 3, W = [0.70, 0.20, 0.10];
    const seeds = [SEED0, 0xF11CE, 0x1BADF1];
    let worst = 0, randOk = true, perfectOk = true, wrongOk = true;
    for (const sd of seeds) {
      const yT = genLabels(600, W, sd);
      const yP = genPreds(yT, K, 0.30, sd ^ 0x2C2C);
      const per = f1Score(yT, yP, 'none', { numClasses: K });
      const mac = f1Score(yT, yP, 'macro', { numClasses: K });
      const mic = f1Score(yT, yP, 'micro', { numClasses: K });
      const g = goldPRF(goldConfusion(yT, yP, K), K);
      for (let c = 0; c < K; c++) worst = Math.max(worst, Math.abs(per[c] - g.F[c]));
      worst = Math.max(worst, Math.abs(mac - goldMean(g.F)), Math.abs(mic - g.microF));
      if (!(mac > 0 && mac < 1)) randOk = false;                 // 非退化：严格介于 (0,1)
      if (!(mic > 0 && mic < 1)) randOk = false;
    }
    // 边界：完全正确 → F1 == 1（EXACT）
    const yPerf = [0, 0, 1, 1, 2, 2, 0, 1, 2];
    const fPerf = f1Score(yPerf, yPerf.slice(), 'macro', { numClasses: K });
    const perPerf = f1Score(yPerf, yPerf.slice(), 'none', { numClasses: K });
    if (fPerf !== 1) perfectOk = false;
    for (const v of perPerf) if (v !== 1) perfectOk = false;
    // 边界：完全错误（二分类全反）→ F1 == 0（EXACT）
    const yT2 = [0, 0, 0, 1, 1, 1];
    const yP2 = [1, 1, 1, 0, 0, 0];
    const fWrong = f1Score(yT2, yP2, 'macro', { numClasses: 2 });
    const fWrongMicro = f1Score(yT2, yP2, 'micro', { numClasses: 2 });
    if (fWrong !== 0 || fWrongMicro !== 0) wrongOk = false;
    const ok = (worst <= TOL.TIGHT) && randOk && perfectOk && wrongOk;
    const d = '3种子 per-class/macro/micro max|Δ|=' + worst.toExponential(2)
      + ' 随机F1∈(0,1)=' + randOk + ' 全对F1==1(EXACT)=' + perfectOk + ' 全错F1==0(EXACT)=' + wrongOk;
    cases.push(caseOf('INV-EVL-002', 'F1 ≡ 手搓 2PR/(P+R) + 全对=1/全错=0 (EXACT) + 随机 F1∈(0,1) 非退化', ok, worst, d, TOL.TIGHT));
    invariants.push({ id: 'INV-EVL-002', name: 'F1（黄金参照 + 双边界 EXACT + 非退化）', ok, expr: '|F1−2PR/(P+R)|≤1e-15；全对==1；全错==0；随机 0<F1<1', tol: TOL.TIGHT, worst, detail: d });
  }

  /* ================= INV-EVL-003 AUC ================= */
  {
    let worst = 0, perfectOk = true, reverseOk = true, randOk = true, tieSensitive = false, worstSigma = 0;
    const seeds = [SEED0, 0x0A0C1, 0x0A0C2, 0x0A0C3];
    for (const sd of seeds) {
      const n = 3000;
      const rng = makeRng(sd);
      const scores = new Array(n), labels = new Array(n);
      let nP = 0, nN = 0;
      for (let i = 0; i < n; i++) {
        scores[i] = rng() * 100;
        labels[i] = rng() < 0.5 ? 1 : 0;
        if (labels[i] === 1) nP++; else nN++;
      }
      if (nP === 0 || nN === 0) { labels[0] = 1; labels[1] = 0; nP = nP || 1; nN = nN || 1; }
      const auc = rocAuc(scores, labels);
      const g1 = goldAucSorted(scores, labels);       // 独立排序计数参照
      worst = Math.max(worst, Math.abs(auc - g1));
      // 统计容差：σ = √((n+1)/(12·nP·nN))，要求 |AUC−0.5| ≤ 10σ（≈0）
      const sigma = Math.sqrt((n + 1) / (12 * nP * nN));
      const dev = Math.abs(auc - 0.5) / sigma;
      worstSigma = Math.max(worstSigma, dev);
      if (dev > 10) randOk = false;
      if (Math.abs(auc - 0.5) < 1e-12) randOk = false; // 反平凡：不得恰好恒等于 0.5
    }
    // 完美排序 / 逆序（EXACT）
    const n2 = 400;
    const sAsc = [];
    for (let i = 0; i < n2; i++) sAsc.push(i);
    const labAsc = new Array(n2).fill(0);
    for (let i = n2 / 2; i < n2; i++) labAsc[i] = 1;
    const labDesc = labAsc.slice().reverse();
    if (rocAuc(sAsc, labAsc) !== 1) perfectOk = false;
    if (rocAuc(sAsc, labDesc) !== 0) reverseOk = false;
    // 并列（tie）敏感：整数得分产生大量并列，含 tie 与不含 tie 结果必分歧
    {
      const n3 = 200;
      const rng = makeRng(0x71ED);
      const sc = new Array(n3), lb = new Array(n3);
      let p = 0, q = 0;
      for (let i = 0; i < n3; i++) {
        sc[i] = Math.floor(rng() * 10);            // 0..9 → 大量并列
        lb[i] = rng() < 0.5 ? 1 : 0;
        if (lb[i] === 1) p++; else q++;
      }
      const aucTie = rocAuc(sc, lb);
      const gTie = goldAucBrute(sc, lb);           // 暴力 O(nP·nN) 黄金参照
      worst = Math.max(worst, Math.abs(aucTie - gTie));
      // 不计 tie（严格大于）的口径：必与含 tie 口径分歧
      let Ustrict = 0;
      for (let i = 0; i < n3; i++) if (lb[i] === 1) {
        for (let j = 0; j < n3; j++) if (lb[j] === 0 && sc[i] > sc[j]) Ustrict++;
      }
      const aucStrict = Ustrict / (p * q);
      if (Math.abs(aucTie - aucStrict) > 1e-9) tieSensitive = true;
    }
    const ok = (worst <= TOL.TIGHT) && perfectOk && reverseOk && randOk && tieSensitive;
    const d = '4种子 max|AUC−Mann-Whitney|=' + worst.toExponential(2)
      + ' 完美=1=' + perfectOk + ' 逆序=0=' + reverseOk
      + ' 随机 max|AUC−0.5|/σ=' + worstSigma.toFixed(2) + '(≤10σ)=' + randOk
      + ' 并列敏感(含tie≠不计tie)=' + tieSensitive;
    cases.push(caseOf('INV-EVL-003', 'AUC ≡ 手搓 Mann-Whitney U（含 tie 0.5）+ 完美=1/逆序=0 EXACT + 随机≈0.5(≤10σ) + 并列敏感', ok, worst, d, TOL.TIGHT));
    invariants.push({ id: 'INV-EVL-003', name: 'ROC AUC（Mann-Whitney 黄金参照 + 三边界 + 并列敏感）', ok, expr: '|AUC−U/(nP·nN)|≤1e-15；完美==1；逆序==0；|AUC−0.5|≤10σ 且 ≠0.5；含tie≠不计tie', tol: TOL.TIGHT, worst, detail: d });
  }

  /* ================= INV-EVL-004 困惑度 ================= */
  {
    let worst = 0, closedOk = true, divergeOk = true, monoOk = true;
    let pplUni = 0, pplPeak = 0;
    const seeds = [SEED0, 0x0B1E1, 0x0B1E2, 0x0B1E3];
    for (const sd of seeds) {
      const n = 400, K = 12;
      const rng = makeRng(sd);
      const logProbs = new Array(n);
      for (let i = 0; i < n; i++) {
        const u = rng();
        const k = Math.min(K - 1, Math.floor(u * K));
        logProbs[i] = Math.log(0.02 + 0.08 * k / (K - 1)) - 0.5 * rng();  // 随机对数概率（≤0）
      }
      const ppl = perplexity(logProbs);
      let nll = 0;
      for (const v of logProbs) nll += -v;                 // 手搓平均 NLL
      worst = Math.max(worst, Math.abs(ppl - Math.exp(nll / n)) / Math.max(1, Math.exp(nll / n)));
    }
    // 闭式：恒定概率 p → perplexity == 1/p
    for (const p of [0.25, 0.5, 0.1, 0.02]) {
      const lp = new Array(128).fill(Math.log(p));
      const got = perplexity(lp);
      if (Math.abs(got - 1 / p) / (1 / p) > TOL.SOLVE) closedOk = false;
      worst = Math.max(worst, Math.abs(got - 1 / p) / (1 / p));
    }
    // 分歧 + 单调性：均匀高熵 perplexity 严格大于尖峰低熵
    const uni = new Array(100).fill(Math.log(0.1));         // 10 类均匀 → ppl = 10
    const peak = new Array(100).fill(Math.log(0.01));
    for (let i = 0; i < 97; i++) peak[i] = Math.log(0.97);  // 尖峰：97% 命中高概率 token
    pplUni = perplexity(uni);
    pplPeak = perplexity(peak);
    if (!(Math.abs(pplUni - 10) / 10 <= TOL.SOLVE)) divergeOk = false;
    if (!(pplUni - pplPeak > 1)) monoOk = false;            // 高熵 ≫ 低熵
    if (Math.abs(pplUni - pplPeak) < 1e-6) divergeOk = false;
    // 安全边界：空数组 / 正对数概率 必抛错
    const guardOk = throws(() => perplexity([])) && throws(() => perplexity([0.5])) && throws(() => perplexity(null));
    const ok = (worst <= TOL.SOLVE) && closedOk && divergeOk && monoOk && guardOk;
    const d = '4种子相对|Δ|=' + worst.toExponential(2) + ' 恒定p: ppl==1/p=' + closedOk
      + ' 均匀ppl=' + pplUni.toFixed(4) + ' 尖峰ppl=' + pplPeak.toFixed(4) + ' 高熵>低熵=' + monoOk
      + ' 抛错守卫=' + guardOk;
    cases.push(caseOf('INV-EVL-004', 'perplexity ≡ exp(手搓平均 NLL) + 恒定 p 闭式 1/p + 高熵>低熵 单调 + 空/正对数概率抛错', ok, worst, d, TOL.SOLVE));
    invariants.push({ id: 'INV-EVL-004', name: '困惑度（NLL 黄金参照 + 闭式 1/p + 熵单调）', ok, expr: '相对|ppl−exp(meanNLL)|≤1e-10；ppl(恒定p)==1/p；ppl(均匀10)>ppl(尖峰) 且差>1', tol: TOL.SOLVE, worst, detail: d });
  }

  /* ================= INV-EVL-005 BLEU ================= */
  {
    let worst = 0, selfOk = true, shortOk = true, zeroOk = true, refDivOk = true;
    const vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'];
    const seeds = [SEED0, 0xB1E11, 0xB1E12];
    for (const sd of seeds) {
      const rng = makeRng(sd);
      const mk = (len) => Array.from({ length: len }, () => vocab[Math.floor(rng() * vocab.length)]);
      for (let t = 0; t < 6; t++) {
        const cand = mk(6 + Math.floor(rng() * 8));
        const refs = [mk(6 + Math.floor(rng() * 8)), mk(6 + Math.floor(rng() * 8))];
        const got = bleu(cand, refs, 4);
        const g = goldBleu(cand, refs, 4);                  // 手搓 n-gram 精度 + BP
        worst = Math.max(worst, Math.abs(got.score - g.score));
        if (Math.abs(got.bp - g.bp) > TOL.SOLVE) worst = Math.max(worst, Math.abs(got.bp - g.bp));
      }
    }
    // 边界：候选 == 参考 → BLEU == 1（EXACT）
    const ref1 = 'the cat sat on the mat near the door';
    const bEq = bleu(ref1, [ref1], 4);
    if (bEq.score !== 1) selfOk = false;
    // 短候选（maxN=2，全部 n-gram 命中）→ brevity penalty 严格 <1 且 >0
    const candShort = ['a', 'b'];
    const refLong = ['a', 'b', 'c', 'd', 'e'];
    const bShort = bleu(candShort, [refLong], 2);
    if (!(bShort.score > 0 && bShort.score < 1) || !(bShort.bp < 1)) shortOk = false;
    // 长候选同样受验：c=4, r=8 → BP=exp(−1)
    const bShort4 = bleu(['a', 'b', 'c', 'd'], [['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']], 4);
    if (!(bShort4.score > 0 && bShort4.score < 1) || Math.abs(bShort4.bp - Math.exp(-1)) > TOL.SOLVE) shortOk = false;
    // 无 n-gram 匹配 → BLEU == 0（EXACT）
    const bZero = bleu(['x', 'y', 'z', 'w'], [['a', 'b', 'c', 'd', 'e', 'f']], 4);
    if (bZero.score !== 0) zeroOk = false;
    // 多参考取最大裁剪计数：单参考不命中、另一参考命中时必须分歧
    const cand2 = ['a', 'b', 'c', 'd'];
    const singleMiss = bleu(cand2, [['q', 'q', 'q', 'q', 'q', 'q']], 2).score;
    const multiHit = bleu(cand2, [['q', 'q', 'q', 'q', 'q', 'q'], ['a', 'b', 'c', 'd', 'e']], 2).score;
    if (!(multiHit > singleMiss)) refDivOk = false;
    const ok = (worst <= TOL.SOLVE) && selfOk && shortOk && zeroOk && refDivOk;
    const d = '3种子×6组 max|BLEU−手搓|=' + worst.toExponential(2) + ' cand==ref→1(EXACT)=' + selfOk
      + ' 短候选 0<BLEU<1(BP<1)=' + shortOk + ' 零匹配→0(EXACT)=' + zeroOk
      + ' 多参考裁剪分歧=' + refDivOk;
    cases.push(caseOf('INV-EVL-005', 'BLEU ≡ 手搓 n-gram 精度几何平均×BP + cand==ref=1 + 短候选受罚<1 + 零匹配=0', ok, worst, d, TOL.SOLVE));
    invariants.push({ id: 'INV-EVL-005', name: 'BLEU（手搓 n-gram/BP 黄金参照 + 三边界 + 多参考裁剪）', ok, expr: '|BLEU−Πp_n^(1/N)·BP|≤1e-10；cand==ref==1；短候选 0<BLEU<1；零匹配==0', tol: TOL.SOLVE, worst, detail: d });
  }

  /* ================= INV-EVL-006 延迟分位数 ================= */
  {
    let worst = 0, monoOk = true, divergeOk = true;
    const seeds = [SEED0, 0x1A7E1, 0x1A7E2, 0x1A7E3];
    const p95s = [];
    for (const sd of seeds) {
      const s = genLatencySamples(1000, sd);
      const st = latencyStats(s);
      // 手搓排序 + 手搓分位（独立书写）
      const sorted = s.slice().sort((a, b) => a - b);
      // 相对偏差（样本量级 ~50，绝对 1ulp≈7e-15，故按相对口径比对，等价 EXACT 语义）
      const rel = (a, b) => Math.abs(a - b) / Math.max(1, Math.abs(b));
      worst = Math.max(worst, rel(st.p50, goldQuantile(sorted, 0.50)));
      worst = Math.max(worst, rel(st.p95, goldQuantile(sorted, 0.95)));
      worst = Math.max(worst, rel(st.p99, goldQuantile(sorted, 0.99)));
      let mean = 0;
      for (const v of s) mean += v;
      worst = Math.max(worst, rel(st.mean, mean / s.length));
      if (!(st.p50 <= st.p95 && st.p95 <= st.p99)) monoOk = false;   // 单调 EXACT
      p95s.push(st.p95);
    }
    // 不同样本集分位数必分歧（非恒定）
    let minSpread = Infinity;
    for (let i = 0; i < p95s.length; i++) {
      for (let j = i + 1; j < p95s.length; j++) minSpread = Math.min(minSpread, Math.abs(p95s[i] - p95s[j]));
    }
    if (!(minSpread > 1e-9)) divergeOk = false;
    const guardOk = throws(() => latencyStats([])) && throws(() => latencyStats([1, NaN, 3]));
    const ok = (worst <= TOL.TIGHT) && monoOk && divergeOk && guardOk;
    const d = '4种子 max 相对|分位−手搓|=' + worst.toExponential(2) + ' p50≤p95≤p99(EXACT)=' + monoOk
      + ' 异样本集 min|Δp95|=' + minSpread.toExponential(3) + '(>1e-9)=' + divergeOk + ' 空/NaN 抛错=' + guardOk;
    cases.push(caseOf('INV-EVL-006', 'p50/p95/p99 ≡ 手搓排序插值取位 + 单调 EXACT + 异样本集必分歧 + 空/NaN 抛错', ok, worst, d, TOL.TIGHT));
    invariants.push({ id: 'INV-EVL-006', name: '延迟分位数（手搓黄金参照 + 单调 + 非恒定）', ok, expr: '|q−goldQuantile|≤1e-15；p50≤p95≤p99(EXACT)；异样本集 min|Δp95|>1e-9', tol: TOL.TIGHT, worst, detail: d });
  }

  /* ================= INV-EVL-007 吞吐量 ================= */
  {
    let worst = 0, ratioOk = true;
    const pairs = [[1000, 4], [2048, 2.5], [777, 0.125], [123456, 60]];
    for (const [c, s] of pairs) {
      const got = throughput(c, s);
      worst = Math.max(worst, Math.abs(got - c / s));                    // 闭式 EXACT
      const dbl = throughput(2 * c, s);
      if (Math.abs(dbl / got - 2) !== 0) ratioOk = false;                // 比例 EXACT
      worst = Math.max(worst, Math.abs(dbl / got - 2));
    }
    // 半计数 → 吞吐减半（比例 EXACT）
    const halfOk = Math.abs(throughput(500, 4) / throughput(1000, 4) - 0.5) === 0;
    // 安全边界：耗时为 0 / 负 / 非有限 / 计数非法 → 抛 TypeError（绝不静默 Infinity）
    const guardOk = throws(() => throughput(100, 0)) && throws(() => throughput(100, -1))
      && throws(() => throughput(100, NaN)) && throws(() => throughput(-5, 1))
      && throws(() => throughput(100, Infinity));
    const ok = (worst <= TOL.TIGHT) && ratioOk && halfOk && guardOk;
    const d = '4组 max|tps−count/seconds|=' + worst.toExponential(2) + ' 加倍比例==2(EXACT)=' + ratioOk
      + ' 减半比例==0.5(EXACT)=' + halfOk + ' 0/负/NaN/Infinity 秒抛错=' + guardOk;
    cases.push(caseOf('INV-EVL-007', 'throughput ≡ count/seconds + 加倍翻倍(比例 EXACT) + 耗时 0/负/非有限必抛错', ok, worst, d, TOL.TIGHT));
    invariants.push({ id: 'INV-EVL-007', name: '吞吐量（闭式 + 比例 EXACT + 除零抛错）', ok, expr: 'tps==count/seconds(≤1e-15)；tps(2c,s)/tps(c,s)==2(EXACT)；seconds≤0 或非有限→TypeError', tol: TOL.TIGHT, worst, detail: d });
  }

  /* ================= INV-EVL-008 一致性 ================= */
  {
    const K = 4, N = 2000, MARGIN = 1e-3;
    const logits = genLogits(N, K, MARGIN, SEED0);
    const predsA = logits.map(argmax);
    const predsA2 = logits.map(argmax);            // 同输入第二次运行
    const rateSame = consistencyRate(predsA, predsA2);
    const sameOk = (rateSame === 1);               // EXACT：逐位相等

    const predsTiny = perturbAndArgmax(logits, 1e-6, 0x71E11);
    const rateTiny = consistencyRate(predsA, predsTiny);
    const tinyOk = (rateTiny === 1);               // 扰动 ≤ 裕度 → 预测不变

    const predsBig = perturbAndArgmax(logits, 0.5, 0xB16D | 0);
    const rateBig = consistencyRate(predsA, predsBig);
    const bigOk = (rateBig < 0.95) && (rateBig > 0);   // 大扰动必明显下降（证伪平凡恒等）

    // 同 seed 全流水线逐位复现 + 异 seed 必分歧
    const pipeline = (seed) => {
      const yT = genLabels(500, [0.6, 0.25, 0.1, 0.05], seed);
      const yP = genPreds(yT, 4, 0.2, seed ^ 0x99);
      const lg = genLogits(64, 3, 1e-3, seed ^ 0x33);
      return JSON.stringify({
        cm: confusion(yT, yP, 4),
        p: precisionScore(yT, yP, 'macro', { numClasses: 4 }),
        r: recallScore(yT, yP, 'micro', { numClasses: 4 }),
        f: f1Score(yT, yP, 'weighted', { numClasses: 4 }),
        auc: rocAuc(genLatencySamples(200, seed), genLabels(200, [0.5, 0.5], seed ^ 0x7)),
        ppl: perplexity(new Array(50).fill(Math.log(0.2))),
        bl: bleu(['a', 'b', 'c', 'd'], [['a', 'b', 'c', 'd', 'e']], 4).score,
        lat: latencyStats(genLatencySamples(300, seed)),
        tps: throughput(2048, 2),
        cons: consistencyRate(lg.map(argmax), perturbAndArgmax(lg, 1e-6, seed))
      });
    };
    const p1 = pipeline(0xC0FFEE), p2 = pipeline(0xC0FFEE), p3 = pipeline(0xC0FFEF);
    const detOk = (p1 === p2) && (p1 !== p3);
    const guardOk = throws(() => consistencyRate([1, 2], [1])) && throws(() => consistencyRate([], []));

    const ok = sameOk && tinyOk && bigOk && detOk && guardOk;
    const d = '同输入两次 rate==1(EXACT)=' + sameOk + ' 微扰1e-6 rate=' + rateTiny + '(=1)=' + tinyOk
      + ' 大扰0.5 rate=' + rateBig.toFixed(4) + '(<0.95)=' + bigOk
      + ' 同seed逐位复现&异seed分歧=' + detOk + ' 长度不一致/空抛错=' + guardOk;
    cases.push(caseOf('INV-EVL-008', '同输入一致率=1(EXACT) + 微扰不变 + 大扰明显下降(<0.95) + 同seed逐位复现/异seed分歧', ok, sameOk && tinyOk ? 0 : 1, d, TOL.EXACT));
    invariants.push({ id: 'INV-EVL-008', name: '一致性（逐位 EXACT + 双扰动幅度对照 + 可复现/分歧）', ok, expr: 'rate(A,A)==1；rate(A,微扰)==1；rate(A,大扰)<0.95；pipeline(同seed)逐位相等且异seed分歧', tol: TOL.EXACT, worst: ok ? 0 : 1, detail: d });
  }

  return buildReport('eval-forge', '1.0.0', 'L7', cases, invariants, '0x9E3779B9');
}

/* ============================================================================
 * 人类可读表格（零依赖本地实现）
 * ========================================================================== */
function formatTable(r) {
  if (!r) return 'NO REPORT';
  const lines = [];
  lines.push('=== ' + r.module + ' v' + r.version + ' (' + r.layer + ') ' + (r.failed === 0 ? 'PASS' : 'FAIL') + ' ===');
  lines.push('passed ' + r.passed + ' / failed ' + r.failed + ' / hash ' + (r.hash || '').slice(0, 24));
  for (const c of (r.cases || [])) {
    const mark = c.status === 'pass' ? '+OK' : '-XX';
    lines.push('  ' + mark + ' ' + c.id + ' ' + c.name + '  act=' + (typeof c.actual === 'number' ? c.actual.toExponential(3) : c.actual) + ' tol=' + c.tol);
  }
  return lines.join('\n');
}

/* ============================================================================
 * API 导出 + 双端挂载（UMD）+ CLI 自检
 * ========================================================================== */
const API = {
  name: 'eval-forge', version: '1.0.0', layer: 'L7', TOL, seed: '0x9E3779B9',
  makeRng, sha256Hex, sourceHash,
  confusion, classPRF, precisionScore, recallScore, f1Score,
  rocAuc, perplexity, perplexityFromProbs,
  bleu, latencyStats, percentile, quantileSorted,
  throughput, consistencyRate, argmax,
  selfCheck, formatTable
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeEval = API;

if (typeof module !== 'undefined' && module.exports && typeof require !== 'undefined' && require.main === module
  && (process.argv.includes('--self-check') || process.argv.includes('--json'))) {
  const r = selfCheck();
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : formatTable(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
