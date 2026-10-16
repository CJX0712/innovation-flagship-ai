/**
 * @forge-module moe-forge
 * @layer L4
 * @version 1.0.0
 * @brief     混合专家（Mixture-of-Experts）内核：top-k 稀疏门控（softmax 仅在 top-k 内归一）、路由分发（加权求和）、
 *            Switch 负载均衡辅助损失、容量因子裁剪（超容量分派丢弃、残差直通）、共享专家、可选噪声门控（Shazeer 1997）。
 *            专家为「单 token 行向量 → 行向量」的纯函数数组（如 transformer-forge.ffnGlu 的 SwiGLU 前馈包装）。
 * @reuse     Outrageously Large Neural Networks (Shazeer et al. 2017)|https://arxiv.org/abs/1701.06538|arXiv|A|稀疏门控/top-k/noisy gate 原式
 * @reuse     Switch Transformer (Fedus et al. 2021)|https://arxiv.org/abs/2101.03961|Apache-2.0|A|负载均衡辅助损失与容量因子
 * @reuse     Mixtral of Experts (Mistral AI 2024)|https://arxiv.org/abs/2401.04088|Apache-2.0|B|top-k 路由 + 共享专家现代口径
 * @reuse     DeepSeekMoE (Dai et al. 2024)|https://arxiv.org/abs/2401.06066|MIT|C|细粒度专家与共享专家策略
 * @beyond    纯 JS 单文件零构建零 npm 依赖；惰性双端依赖（Node require / 浏览器 globalThis）装配 core-verify / core-numeric /
 *            transformer-forge（ffnGlu SwiGLU 专家）/ loss-act-norm（稳定 softmax）；门控/路由/损失/裁剪全部纯函数式；
 *            k=1 与 k=nExperts 退化情形用独立重实现交叉验证，向量化前向与朴素双重循环逐元素一致。
 * @invariant INV-MOE-001|top-k 门控归一：随机 gate logits 每 token top-k 权重和 == 1 且权重全非负|1e-12|softmax 定义（top-k 内归一）
 * @invariant INV-MOE-002|top-k 正确性：20 组随机 gate logits 选中集合 == logit 最大 k 个专家（违例计数 0）|0|独立排序参照 vs 主实现
 * @invariant INV-MOE-003|单专家退化：k=1 确定性门控下输出 == 被选中专家输出逐元素一致（权重=1）|1e-12|退化情形参照专家直算
 * @invariant INV-MOE-004|全专家退化：k=nExperts 时输出 == 全专家 softmax 加权平均（独立重实现）|1e-12|独立双重循环参照 vs 主实现
 * @invariant INV-MOE-005|负载均衡闭式：均匀门控 aux == 理论最小值 1.0，且随机 20 组 aux ≥ 1|1e-12|Switch 公式 + Cauchy-Schwarz 下界
 * @invariant INV-MOE-006|容量守恒：裁剪后每专家分派数 ≤ 容量上限，且总丢弃数 == 超容量数（精确整数断言）|0|独立逐专家重算 vs 主实现
 * @invariant INV-MOE-007|向量化 vs 独立重实现：moeForward 与逐 token 朴素双重循环输出逐元素一致|1e-12|参照实现 vs 主实现
 * @invariant INV-MOE-008|确定性：同输入两次 moeForward、同 rng 种子两次 noisyGate 输出逐位相等|0|纯函数确定性定义
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

let _Tf = null;
function Tf() {
  if (_Tf) return _Tf;
  if (typeof require !== 'undefined') { try { _Tf = require('./transformer-forge.js'); } catch (e) {} }
  if (!_Tf && typeof globalThis !== 'undefined') _Tf = globalThis.ForgeTransformer || null;
  if (!_Tf) throw new Error('transformer-forge 依赖缺失');
  return _Tf;
}

let _L = null;
function L() {
  if (_L) return _L;
  if (typeof require !== 'undefined') { try { _L = require('./loss-act-norm-forge.js'); } catch (e) {} }
  if (!_L && typeof globalThis !== 'undefined') _L = globalThis.ForgeLossActNorm || null;
  if (!_L) throw new Error('loss-act-norm-forge 依赖缺失');
  return _L;
}

/* ============================================================================
 * 矩阵约定（与 transformer-forge 一致）：二维矩阵 = 数组 of Float64Array（行主序），
 * 权重矩阵 W 为 out×in（linear 语义：y[o] = Σ_k x[k]·W[o][k]）。
 * ========================================================================== */

/** 校验二维矩阵形状合法性，返回 {rows, cols}；非法抛错。 */
function checkMatrix(m, name) {
  if (!Array.isArray(m) || m.length === 0) throw new Error(name + ' 必须为非空二维数组（行主序）');
  const cols = m[0].length;
  if (typeof cols !== 'number' || cols < 1) throw new Error(name + ' 首行宽度非法');
  for (let i = 1; i < m.length; i++) {
    if (!m[i] || m[i].length !== cols) throw new Error(name + ' 第 ' + i + ' 行宽度不一致（期望 ' + cols + '）');
  }
  return { rows: m.length, cols };
}

/** 逐元素相加 A+B（残差连接 / 共享专家汇入），形状不一致抛错；返回新矩阵（不修改入参）。 */
function addMatrices(a, b) {
  checkMatrix(a, '相加左操作数');
  checkMatrix(b, '相加右操作数');
  if (a.length !== b.length || a[0].length !== b[0].length) throw new Error('相加形状不一致');
  const out = new Array(a.length);
  for (let i = 0; i < a.length; i++) {
    const row = new Float64Array(a[i].length);
    for (let k = 0; k < row.length; k++) row[k] = a[i][k] + b[i][k];
    out[i] = row;
  }
  return out;
}

/** 校验门控结果 {expertIds, gateWeights}（T×k 双层数组）合法性，返回 token 数 T。 */
function checkGate(gate, name) {
  if (!gate || !Array.isArray(gate.expertIds) || !Array.isArray(gate.gateWeights)) {
    throw new Error(name + ' 须为 { expertIds, gateWeights }（T×k 双层数组）');
  }
  if (gate.expertIds.length !== gate.gateWeights.length) throw new Error(name + ' 的 expertIds 与 gateWeights 行数不一致');
  for (let t = 0; t < gate.expertIds.length; t++) {
    const ids = gate.expertIds[t], ws = gate.gateWeights[t];
    if (!Array.isArray(ids) || !Array.isArray(ws) || ids.length !== ws.length) {
      throw new Error(name + ' 第 ' + t + ' 行分派/权重长度不一致');
    }
  }
  return gate.expertIds.length;
}

/* ---------- 1. top-k 稀疏门控（Shazeer et al. 1997 口径：softmax 仅在 top-k 内归一） ---------- */

/**
 * 对每 token 的 gate logits 取 top-k 专家，softmax 仅在 top-k 内归一（非全部专家归一）。
 * 选中专家按 logit 降序排列（并列按索引升序，保证确定性）。
 * @param {Array<Float64Array|number[]>} gateLogits T × nExperts 门控 logits
 * @param {number} k 每 token 选用的专家数（1 ≤ k ≤ nExperts 的整数）
 * @returns {{expertIds:number[][], gateWeights:number[][]}} 形状均为 T×k；每行权重和 == 1
 */
function topKGating(gateLogits, k) {
  checkMatrix(gateLogits, 'gateLogits');
  const E = gateLogits[0].length;
  if (!Number.isInteger(k) || k < 1 || k > E) throw new Error('k 必须为 [1,' + E + '] 内的整数（得到 ' + k + '）');
  const T = gateLogits.length;
  const expertIds = new Array(T);
  const gateWeights = new Array(T);
  const stableSoftmax = L().stableSoftmax;
  for (let t = 0; t < T; t++) {
    const row = gateLogits[t];
    const order = new Array(E);
    for (let e = 0; e < E; e++) order[e] = e;
    // 按 logit 降序排序（并列按索引升序）——与独立参照实现同一确定性口径
    order.sort((a, b) => (row[b] - row[a]) || (a - b));
    const ids = order.slice(0, k);
    // softmax 仅在 top-k 子集内归一（Shazeer 1997：Noisy Top-K Gating 原式口径）
    const sub = new Float64Array(k);
    for (let j = 0; j < k; j++) sub[j] = row[ids[j]];
    gateWeights[t] = Array.from(stableSoftmax(sub));
    expertIds[t] = ids;
  }
  return { expertIds, gateWeights };
}

/* ---------- 2. 路由分发 ---------- */

/**
 * 按门控把每个 token 分派给其 k 个专家并加权求和：out[t] = Σ_j w[t][j] · experts[ids[t][j]](x[t])。
 * 专家为「单 token 行向量 → 行向量」纯函数（如 ffnGlu 的行包装）。空分派行（容量裁剪后）输出零行。
 * @param {{expertIds:number[][], gateWeights:number[][]}} gate topKGating / noisyGate / capacityClip 产出的门控
 * @param {Array<Float64Array>} tokens T × d 输入矩阵
 * @param {Array<(row:Float64Array)=>Float64Array>} experts 专家函数数组（索引即专家编号）
 * @returns {Array<Float64Array>} T × d 输出矩阵
 */
function routeTokens(gate, tokens, experts) {
  const ts = checkMatrix(tokens, 'tokens');
  const T = checkGate(gate, 'gate');
  if (T !== ts.rows) throw new Error('gate 行数（' + T + '）与 tokens 行数（' + ts.rows + '）不一致');
  if (!Array.isArray(experts) || experts.length === 0) throw new Error('experts 必须为非空函数数组');
  const d = ts.cols;
  const out = new Array(T);
  for (let t = 0; t < T; t++) {
    const ids = gate.expertIds[t], ws = gate.gateWeights[t];
    const row = new Float64Array(d);
    for (let s = 0; s < ids.length; s++) {
      const e = ids[s];
      if (!Number.isInteger(e) || e < 0 || e >= experts.length) {
        throw new Error('gate 第 ' + t + ' 行专家索引 ' + e + ' 越界（experts.length=' + experts.length + '）');
      }
      if (typeof experts[e] !== 'function') throw new Error('experts[' + e + '] 必须为函数');
      const y = experts[e](tokens[t]);
      if (!y || y.length !== d) throw new Error('experts[' + e + '] 输出宽度须为 d=' + d);
      for (let j = 0; j < d; j++) row[j] += ws[s] * y[j];
    }
    out[t] = row;
  }
  return out;
}

/* ---------- 3. 完整 MoE 前向 ---------- */

/**
 * 完整 MoE 前向：gateLogits = x·Wrouter → topKGating → routeTokens →（可选）加共享专家输出。
 * @param {Array<Float64Array>} x T × d 输入矩阵
 * @param {Array<Float64Array>} Wrouter nExperts × d 路由权重（out×in 约定）
 * @param {Array<(row:Float64Array)=>Float64Array>} experts 专家函数数组
 * @param {number} k top-k 专家数
 * @param {{sharedExpert?:(m:Array<Float64Array>)=>Array<Float64Array>}} [opts] 可选共享专家（矩阵级函数，Mixtral 口径）
 * @returns {{out:Array<Float64Array>, gate:{expertIds:number[][],gateWeights:number[][]}, auxLoss:number}}
 *          out = 路由输出（+共享专家）；auxLoss = Switch 负载均衡辅助损失
 */
function moeForward(x, Wrouter, experts, k, opts) {
  opts = opts || {};
  const xs = checkMatrix(x, 'x');
  const wr = checkMatrix(Wrouter, 'Wrouter');
  if (wr.cols !== xs.cols) throw new Error('Wrouter 形状须为 nExperts×d（in=' + xs.cols + '，得到 ' + wr.cols + '）');
  if (!Array.isArray(experts) || experts.length < wr.rows) {
    throw new Error('experts 数量（' + (experts ? experts.length : 0) + '）须 ≥ 专家数 ' + wr.rows);
  }
  const E = wr.rows, d = xs.cols, T = xs.rows;
  // gateLogits = x·Wrouter（向量化：logits[t][e] = Σ_j x[t][j]·Wrouter[e][j]）
  const gateLogits = new Array(T);
  for (let t = 0; t < T; t++) {
    const row = new Float64Array(E);
    for (let e = 0; e < E; e++) {
      let s = 0;
      for (let j = 0; j < d; j++) s += x[t][j] * Wrouter[e][j];
      row[e] = s;
    }
    gateLogits[t] = row;
  }
  const gate = topKGating(gateLogits, k);
  let out = routeTokens(gate, x, experts);
  if (opts.sharedExpert !== undefined && opts.sharedExpert !== null) {
    if (typeof opts.sharedExpert !== 'function') throw new Error('opts.sharedExpert 必须为矩阵级函数');
    out = addMatrices(out, opts.sharedExpert(x));
  }
  const auxLoss = loadBalancingLoss(gate, E);
  return { out, gate, auxLoss };
}

/* ---------- 4. 负载均衡辅助损失（Switch Transformer, Fedus et al. 2021） ---------- */

/**
 * Switch 负载均衡辅助损失：aux = nExperts · Σ_e f_e · P_e。
 * f_e = 分派给专家 e 的（token,槽位）比例；P_e = 专家 e 的平均门控概率（门控质量 / token 数）。
 * 由 Cauchy-Schwarz，概率向量 f 与 P（各元素和均为 1）满足 Σ f_e·P_e ≥ 1/nExperts，故 aux ≥ 1，
 * 均匀门控时取理论最小值 1.0。
 * @param {{expertIds:number[][], gateWeights:number[][]}} gate 门控结果
 * @param {number} nExperts 专家总数
 * @returns {number} 辅助损失值（≥ 1，空门控返回 0）
 */
function loadBalancingLoss(gate, nExperts) {
  if (!Number.isInteger(nExperts) || nExperts < 1) throw new Error('nExperts 必须为正整数');
  const T = checkGate(gate, 'gate');
  if (T === 0) return 0;
  const counts = new Float64Array(nExperts);
  const mass = new Float64Array(nExperts);
  let slots = 0;
  for (let t = 0; t < T; t++) {
    const ids = gate.expertIds[t], ws = gate.gateWeights[t];
    for (let s = 0; s < ids.length; s++) {
      const e = ids[s];
      if (!Number.isInteger(e) || e < 0 || e >= nExperts) throw new Error('专家索引 ' + e + ' 越界（nExperts=' + nExperts + '）');
      counts[e] += 1;
      mass[e] += ws[s];
      slots += 1;
    }
  }
  let aux = 0;
  for (let e = 0; e < nExperts; e++) {
    const f = slots > 0 ? counts[e] / slots : 0; // 分派比例
    const P = mass[e] / T;                        // 平均门控概率
    aux += f * P;
  }
  return nExperts * aux;
}

/* ---------- 5. 容量因子裁剪 ---------- */

/**
 * 容量裁剪（Switch 容量因子口径）：capacity = ceil(capacityFactor · tokensPerExpert)，
 * 按 token 顺序（槽位顺序）逐个分派计数，超出容量的分派被丢弃（残差直通，由调用方残差连接承载，
 * 本函数只做分派级裁剪，不重归一保留槽位权重）。不修改入参（拷贝语义）。
 * @param {{expertIds:number[][], gateWeights:number[][]}} gate 原始门控
 * @param {number} capacityFactor 容量因子（正数，典型 1.0~1.5）
 * @param {number} tokensPerExpert 每专家期望 token 数（如 T·k/nExperts）
 * @returns {{gate:{expertIds:number[][],gateWeights:number[][]}, dropped:number, capacity:number}}
 *          裁剪后门控（被丢弃槽位已从行中移除）、丢弃分派总数、容量上限
 */
function capacityClip(gate, capacityFactor, tokensPerExpert) {
  const T = checkGate(gate, 'gate');
  if (typeof capacityFactor !== 'number' || !(capacityFactor > 0)) throw new Error('capacityFactor 必须为正数');
  if (typeof tokensPerExpert !== 'number' || !(tokensPerExpert > 0)) throw new Error('tokensPerExpert 必须为正数');
  const capacity = Math.ceil(capacityFactor * tokensPerExpert);
  const counts = new Map();
  const expertIds = new Array(T);
  const gateWeights = new Array(T);
  let dropped = 0;
  for (let t = 0; t < T; t++) {
    const ids = gate.expertIds[t], ws = gate.gateWeights[t];
    const keepIds = [], keepWs = [];
    for (let s = 0; s < ids.length; s++) {
      const e = ids[s];
      const c = (counts.get(e) || 0) + 1;
      if (c > capacity) { dropped += 1; continue; } // 超容量：丢弃该分派（残差直通）
      counts.set(e, c);
      keepIds.push(e);
      keepWs.push(ws[s]);
    }
    expertIds[t] = keepIds;
    gateWeights[t] = keepWs;
  }
  return { gate: { expertIds, gateWeights }, dropped, capacity };
}

/* ---------- 6. 专家使用直方图 ---------- */

/**
 * 统计每个专家被选中的次数（直方图，含容量裁剪后的门控）。
 * @param {{expertIds:number[][], gateWeights:number[][]}} gate 门控结果
 * @param {number} nExperts 专家总数
 * @returns {number[]} 长度 nExperts 的计数数组
 */
function expertUsage(gate, nExperts) {
  if (!Number.isInteger(nExperts) || nExperts < 1) throw new Error('nExperts 必须为正整数');
  checkGate(gate, 'gate');
  const counts = new Array(nExperts).fill(0);
  for (let t = 0; t < gate.expertIds.length; t++) {
    for (let s = 0; s < gate.expertIds[t].length; s++) {
      const e = gate.expertIds[t][s];
      if (!Number.isInteger(e) || e < 0 || e >= nExperts) throw new Error('专家索引 ' + e + ' 越界（nExperts=' + nExperts + '）');
      counts[e] += 1;
    }
  }
  return counts;
}

/* ---------- 7. 噪声门控（Shazeer 1997 noisy top-k） ---------- */

/**
 * 噪声 top-k 门控：H(x)_e = (x·Wrouter)_e + softplus((x·Wnoise)_e)·ε_e，ε ~ N(0,1)（Acklam 逆 CDF，无状态），
 * 再加噪后的 logits 上取 top-k。softplus 用 ln1p(exp(·)) 稳定式（大值截断防溢出）。
 * @param {Array<Float64Array>} x T × d 输入
 * @param {Array<Float64Array>} Wrouter nExperts × d 干净路由权重（out×in）
 * @param {Array<Float64Array>} Wnoise  nExperts × d 噪声路由权重（out×in）
 * @param {number} k top-k 专家数
 * @param {()=>number} rng 均匀 [0,1) 随机源（确定性，如 core-numeric.makeRng 的产物）
 * @returns {{expertIds:number[][], gateWeights:number[][]}} 同 topKGating 形状
 */
function noisyGate(x, Wrouter, Wnoise, k, rng) {
  const xs = checkMatrix(x, 'x');
  const wr = checkMatrix(Wrouter, 'Wrouter');
  const wn = checkMatrix(Wnoise, 'Wnoise');
  if (wr.cols !== xs.cols || wn.cols !== xs.cols || wn.rows !== wr.rows) {
    throw new Error('Wrouter / Wnoise 形状须一致且均为 nExperts×d');
  }
  if (typeof rng !== 'function') throw new Error('rng 必须为 () => [0,1) 的函数');
  const normInv = N().normInv;
  const E = wr.rows, d = xs.cols, T = xs.rows;
  const noisy = new Array(T);
  for (let t = 0; t < T; t++) {
    const row = new Float64Array(E);
    for (let e = 0; e < E; e++) {
      let clean = 0, noisePre = 0;
      for (let j = 0; j < d; j++) { const xv = x[t][j]; clean += xv * Wrouter[e][j]; noisePre += xv * Wnoise[e][j]; }
      // softplus(v) = ln(1+e^v)，大值截断防 exp 溢出
      const sp = noisePre > 20 ? noisePre : Math.log1p(Math.exp(noisePre));
      const u = rng();
      const eps = normInv(u <= 0 ? 1e-12 : (u >= 1 ? 1 - 1e-12 : u)); // 防御 u=0/1 的边界
      row[e] = clean + sp * eps;
    }
    noisy[t] = row;
  }
  return topKGating(noisy, k);
}

/* ============================================================================
 * selfCheck — 8 条硬不变量
 * ========================================================================== */
function randMatrix(rng, r, c) {
  const m = new Array(r);
  for (let i = 0; i < r; i++) {
    const row = new Float64Array(c);
    for (let k = 0; k < c; k++) row[k] = rng() * 2 - 1;
    m[i] = row;
  }
  return m;
}

function selfCheck(opts) {
  opts = opts || {};
  const { caseResult, buildReport } = V();
  const cases = [];
  const invariants = [];
  const rng = N().makeRng(0x9e3779b9);

  // 自检用确定性专家工厂：专家 e 输出 = (e+1)·x + 0.5（互异、精确、无随机性）
  function affineExperts(E, d) {
    const experts = [];
    for (let e = 0; e < E; e++) {
      experts.push((row) => {
        const o = new Float64Array(row.length);
        for (let j = 0; j < row.length; j++) o[j] = (e + 1) * row[j] + 0.5;
        return o;
      });
    }
    return experts;
  }
  // 自检用 SwiGLU 专家工厂：复用 transformer-forge.ffnGlu（W 为 out×in 约定）
  function swigluExperts(rng, E, d, hidden) {
    const experts = [];
    for (let e = 0; e < E; e++) {
      const Wg = randMatrix(rng, hidden, d), Wu = randMatrix(rng, hidden, d), Wd = randMatrix(rng, d, hidden);
      experts.push((row) => Tf().ffnGlu([row], Wg, Wu, Wd)[0]);
    }
    return experts;
  }
  function maxDevMat(a, b) {
    let dev = 0;
    for (let i = 0; i < a.length; i++) for (let j = 0; j < a[i].length; j++) dev = Math.max(dev, Math.abs(a[i][j] - b[i][j]));
    return dev;
  }

  // INV-MOE-001 门控归一：随机 gate logits，每 token top-k 权重和 == 1 且权重全非负
  {
    let dev = 0, neg = 0;
    for (let t = 0; t < 20; t++) {
      const E = 4 + (t % 5);          // 4..8 个专家
      const Tk = 3 + (t % 4);         // 3..6 个 token
      const k = 1 + (t % E);          // 覆盖 k=1 与 k=E
      const g = topKGating(randMatrix(rng, Tk, E), k);
      for (let i = 0; i < Tk; i++) {
        let s = 0;
        for (let j = 0; j < k; j++) { s += g.gateWeights[i][j]; if (g.gateWeights[i][j] < 0) neg++; }
        dev = Math.max(dev, Math.abs(s - 1));
      }
    }
    cases.push(caseResult('INV-MOE-001', 'top-k 门控权重和==1 且非负', Math.max(dev, neg), 0, 1e-12));
  }

  // INV-MOE-002 top-k 正确性：选中集合 == logit 最大 k 个（独立排序参照，违例计数 0）
  {
    let violations = 0;
    for (let t = 0; t < 20; t++) {
      const E = 5 + (t % 4);
      const Tk = 4;
      const k = 1 + (t % E);
      const logits = randMatrix(rng, Tk, E);
      const g = topKGating(logits, k);
      for (let i = 0; i < Tk; i++) {
        // 独立参照路径：数组降序排序取前 k，再升序化为集合比较
        const ref = Array.from({ length: E }, (_, e) => e)
          .sort((a, b) => logits[i][b] - logits[i][a] || a - b)
          .slice(0, k)
          .sort((a, b) => a - b);
        const got = Array.from(g.expertIds[i]).sort((a, b) => a - b);
        if (ref.length !== got.length || ref.some((v, idx) => v !== got[idx])) violations++;
      }
    }
    cases.push(caseResult('INV-MOE-002', 'top-k 选中集合==logit 最大 k 个（违例计数）', violations, 0, 0));
  }

  // INV-MOE-003 单专家退化：k=1 确定性门控，输出 == 被选中专家输出（权重=1）
  {
    const d = 4, E = 4;
    // x[t] = one-hot(t)；Wrouter[e] = (e+1)·one-hot(e) → logits[t][e] = (e+1)·[e==t]，token t 必选专家 t
    const x = new Array(d);
    for (let t = 0; t < d; t++) { const row = new Float64Array(d); row[t] = 1; x[t] = row; }
    const Wrouter = new Array(E);
    for (let e = 0; e < E; e++) { const row = new Float64Array(d); row[e] = 10 * (e + 1); Wrouter[e] = row; }
    const experts = affineExperts(E, d);
    const r = moeForward(x, Wrouter, experts, 1);
    let dev = 0, wdev = 0;
    for (let t = 0; t < d; t++) {
      // 防空洞化：路由目标必须是专家 t
      if (r.gate.expertIds[t][0] !== t) { dev = Math.max(dev, 1); continue; }
      const y = experts[t](x[t]); // 独立参照：被选中专家直算
      for (let j = 0; j < d; j++) dev = Math.max(dev, Math.abs(r.out[t][j] - y[j]));
      wdev = Math.max(wdev, Math.abs(r.gate.gateWeights[t][0] - 1));
    }
    cases.push(caseResult('INV-MOE-003', 'k=1 退化：输出==被选中专家输出（权重=1）', Math.max(dev, wdev), 0, 1e-12));
  }

  // INV-MOE-004 全专家退化：k=nExperts 时输出 == 全专家 softmax 加权平均（独立重实现）
  {
    const Tk = 5, d = 6, E = 4;
    const x = randMatrix(rng, Tk, d);
    const Wrouter = randMatrix(rng, E, d);
    const experts = affineExperts(E, d);
    const r = moeForward(x, Wrouter, experts, E);
    // 独立重实现：逐 token 双重循环，全专家 softmax 后加权（不复用 topKGating / stableSoftmax）
    let dev = 0;
    for (let t = 0; t < Tk; t++) {
      const logits = new Array(E);
      for (let e = 0; e < E; e++) {
        let s = 0;
        for (let j = 0; j < d; j++) s += x[t][j] * Wrouter[e][j];
        logits[e] = s;
      }
      let mx = -Infinity;
      for (let e = 0; e < E; e++) if (logits[e] > mx) mx = logits[e];
      let sum = 0;
      const exps = new Array(E);
      for (let e = 0; e < E; e++) { exps[e] = Math.exp(logits[e] - mx); sum += exps[e]; }
      const row = new Float64Array(d);
      for (let e = 0; e < E; e++) {
        const p = exps[e] / sum;
        const y = experts[e](x[t]);
        for (let j = 0; j < d; j++) row[j] += p * y[j];
      }
      for (let j = 0; j < d; j++) dev = Math.max(dev, Math.abs(r.out[t][j] - row[j]));
    }
    cases.push(caseResult('INV-MOE-004', 'k=E 退化：==全专家 softmax 加权平均', dev, 0, 1e-12));
  }

  // INV-MOE-005 负载均衡闭式：均匀门控 aux == 1.0；随机 20 组 aux ≥ 1
  {
    // 均匀门控：logits[t][e] = 1 当 (e-t) mod E < k，否则 0 → 每专家被选次数相等、权重相等
    const E = 4, k = 2, Tk = 8;
    const logits = new Array(Tk);
    for (let t = 0; t < Tk; t++) {
      const row = new Float64Array(E);
      for (let e = 0; e < E; e++) row[e] = (((e - t) % E) + E) % E < k ? 1 : 0;
      logits[t] = row;
    }
    const auxUniform = loadBalancingLoss(topKGating(logits, k), E);
    let violations = 0;
    for (let t = 0; t < 20; t++) {
      const g = topKGating(randMatrix(rng, 6, 5), 2);
      if (!(loadBalancingLoss(g, 5) >= 1 - 1e-9)) violations++; // Cauchy-Schwarz 下界 aux ≥ 1
    }
    cases.push(caseResult('INV-MOE-005', '均匀门控 aux==1.0 且随机 aux≥1', Math.max(Math.abs(auxUniform - 1), violations), 0, 1e-12));
  }

  // INV-MOE-006 容量守恒：分派数 ≤ 容量且丢弃数 == 超容量数（精确整数断言）
  {
    // 构造分派：每 token 都选专家 0 与 1（T=10, E=4, k=2 → tokensPerExpert=5, cf=0.6 → capacity=3）
    const E = 4, Tk = 10, k = 2;
    const logits = new Array(Tk);
    for (let t = 0; t < Tk; t++) { const row = new Float64Array(E); row[0] = 1; row[1] = 1; logits[t] = row; }
    const gate = topKGating(logits, k);
    const tokensPerExpert = (Tk * k) / E;
    const cf = 0.6;
    const res = capacityClip(gate, cf, tokensPerExpert);
    const capacity = Math.ceil(cf * tokensPerExpert);
    // 独立参照：逐专家重算超容量丢弃数
    const counts = new Array(E).fill(0);
    let refDropped = 0;
    for (let t = 0; t < Tk; t++) {
      for (let s = 0; s < gate.expertIds[t].length; s++) {
        const e = gate.expertIds[t][s];
        counts[e] += 1;
        if (counts[e] > capacity) refDropped++;
      }
    }
    const usage = expertUsage(res.gate, E);
    let over = 0, kept = 0;
    for (let e = 0; e < E; e++) { if (usage[e] > capacity) over++; kept += usage[e]; }
    const metric = Math.max(
      over,                                                    // 每专家分派 ≤ 容量
      res.dropped === refDropped ? 0 : 1,                      // 丢弃数 == 超容量数
      res.capacity === capacity ? 0 : 1,                       // 容量公式 ceil(cf·tpe)
      kept === Tk * k - res.dropped ? 0 : 1                    // 保留+丢弃 == 总分派数
    );
    cases.push(caseResult('INV-MOE-006', '容量守恒：分派≤容量且丢弃数==超容量数', metric, 0, 0));
  }

  // INV-MOE-007 向量化 vs 独立重实现：moeForward 与逐 token 朴素双重循环逐元素一致
  {
    const Tk = 6, d = 8, E = 4, k = 2;
    const x = randMatrix(rng, Tk, d);
    const Wrouter = randMatrix(rng, E, d);
    const experts = swigluExperts(rng, E, d, 6); // SwiGLU 专家（复用 transformer-forge.ffnGlu）
    const Wsg = randMatrix(rng, 6, d), Wsu = randMatrix(rng, 6, d), Wsd = randMatrix(rng, d, 6);
    const sharedExpert = (m) => Tf().ffnGlu(m, Wsg, Wsu, Wsd); // 共享专家（Mixtral 口径）
    const r = moeForward(x, Wrouter, experts, k, { sharedExpert });
    // 朴素双重循环独立重实现（不复用本模块任何路由/门控函数）
    let dev = 0;
    for (let t = 0; t < Tk; t++) {
      const logits = new Array(E);
      for (let e = 0; e < E; e++) {
        let s = 0;
        for (let j = 0; j < d; j++) s += x[t][j] * Wrouter[e][j];
        logits[e] = s;
      }
      const idx = Array.from({ length: E }, (_, e) => e)
        .sort((a, b) => logits[b] - logits[a] || a - b)
        .slice(0, k);
      let mx = -Infinity;
      for (let s = 0; s < k; s++) if (logits[idx[s]] > mx) mx = logits[idx[s]];
      let sum = 0;
      const w = new Array(k);
      for (let s = 0; s < k; s++) { w[s] = Math.exp(logits[idx[s]] - mx); sum += w[s]; }
      for (let s = 0; s < k; s++) w[s] /= sum;
      const row = new Float64Array(d);
      for (let s = 0; s < k; s++) {
        const y = experts[idx[s]](x[t]);
        for (let j = 0; j < d; j++) row[j] += w[s] * y[j];
      }
      const sh = sharedExpert([x[t]])[0]; // ffnGlu 行独立 → 与矩阵级调用逐位一致
      for (let j = 0; j < d; j++) row[j] += sh[j];
      for (let j = 0; j < d; j++) dev = Math.max(dev, Math.abs(r.out[t][j] - row[j]));
    }
    cases.push(caseResult('INV-MOE-007', 'moeForward 向量化 vs 朴素双重循环', dev, 0, 1e-12));
  }

  // INV-MOE-008 确定性：同输入两次 moeForward、同 rng 种子两次 noisyGate 逐位相等
  {
    const Tk = 5, d = 7, E = 4, k = 2;
    const x = randMatrix(rng, Tk, d);
    const Wrouter = randMatrix(rng, E, d);
    const Wnoise = randMatrix(rng, E, d);
    const experts = affineExperts(E, d);
    const r1 = moeForward(x, Wrouter, experts, k);
    const r2 = moeForward(x, Wrouter, experts, k);
    let metric = maxDevMat(r1.out, r2.out);
    metric = Math.max(metric, Math.abs(r1.auxLoss - r2.auxLoss));
    // 噪声门控：同种子两条独立 rng 流 → 逐位相等
    const n1 = noisyGate(x, Wrouter, Wnoise, k, N().makeRng(42));
    const n2 = noisyGate(x, Wrouter, Wnoise, k, N().makeRng(42));
    let mism = 0;
    for (let t = 0; t < Tk; t++) {
      for (let s = 0; s < k; s++) {
        if (n1.expertIds[t][s] !== n2.expertIds[t][s]) mism++;
        if (n1.gateWeights[t][s] !== n2.gateWeights[t][s]) mism++;
      }
    }
    cases.push(caseResult('INV-MOE-008', '确定性：同输入两次前向+同种子噪声门控逐位相等', Math.max(metric, mism), 0, 0));
  }

  // 不变量登记
  for (const id of ['INV-MOE-001', 'INV-MOE-002', 'INV-MOE-003', 'INV-MOE-004', 'INV-MOE-005', 'INV-MOE-006', 'INV-MOE-007', 'INV-MOE-008']) {
    const c = cases.find(x => x.id === id);
    invariants.push({ id, expr: id, tol: c.tol, worst: c.abs_err, status: c.status, note: '' });
  }

  const started = Date.now();
  return buildReport({
    module: 'moe-forge', version: '1.0.0', layer: 'L4', cases, invariants,
    seed: '0x9E3779B9', started,
    tol: { 'INV-MOE-001': 1e-12, 'INV-MOE-002': 0, 'INV-MOE-003': 1e-12, 'INV-MOE-004': 1e-12, 'INV-MOE-005': 1e-12, 'INV-MOE-006': 0, 'INV-MOE-007': 1e-12, 'INV-MOE-008': 0 }
  });
}

const API = {
  name: 'moe-forge', version: '1.0.0', layer: 'L4',
  topKGating, routeTokens, moeForward, loadBalancingLoss, capacityClip, expertUsage, noisyGate, selfCheck
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeMoE = API;

if (typeof process !== 'undefined' && process.argv[1] && /moe-forge\.js$/.test(process.argv[1]) && (process.argv.includes('--self-check') || process.argv.includes('--json'))) {
  const r = selfCheck({ quick: process.argv.includes('--quick') });
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : V().formatTableSafe(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
