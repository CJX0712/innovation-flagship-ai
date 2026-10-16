/**
 * @forge-module position-encoding-forge
 * @layer L4
 * @version 1.0.0
 * @brief     位置编码族内核：ALiBi 线性偏置（Press et al. 2021，bias[h][i][j] = -slopes[h]·(j-i)，因果位上三角置 -Infinity）、
 *            T5 相对位置桶映射（Raffel et al. 2020，大小桶拆分 + 对数分桶 + 远距 clamp 到同一远桶）、
 *            位置插值（Chen et al. 2023，插值后位置 = pos/scale，可与 RoPE 组合扩展上下文）、
 *            偏置施加到注意力分数（applyBias，与 attention-forge 的 softmaxRows 对接即构成因果掩码）。
 * @reuse     ALiBi (Press et al. 2021)|https://arxiv.org/abs/2108.12409|arXiv|A|线性偏置斜率公式与 get_slopes
 * @reuse     Position Interpolation (Chen et al. 2023)|https://arxiv.org/abs/2306.15595|arXiv|A|位置插值公式
 * @reuse     T5 (Raffel et al. 2020)|https://arxiv.org/abs/1910.10683|Apache-2.0|A|相对位置分桶公式
 * @reuse     RoPE (Su et al. 2021, 本项目 rope-forge)|https://arxiv.org/abs/2104.09864|MIT|C|PI 组合复用既有 rope 接口
 * @beyond    纯 JS 单文件零构建零 npm 依赖；惰性双端依赖（core-verify / core-numeric / attention-forge / rope-forge）装配，
 *            ALiBi / T5 / 位置插值全部按原论文闭式实现并以硬不变量交叉验证；偏置施加 + softmaxRows 组合直接证明因果掩码语义；
 *            RoPE 复用既有 rope-forge 接口（PI∘RoPE == RoPE(x, pos/scale, base)）而非重写旋转；正弦位置编码已在
 *            transformer-forge 实现、RoPE 已在 rope-forge 实现，本模块均不重复造轮子。
 * @invariant INV-POS-001|ALiBi 因果结构：causal 模式所有 j>i 位 bias==-Infinity、下三角全有限（违例计数 0）|0|因果掩码定义
 * @invariant INV-POS-002|ALiBi 斜率：首项==2^(-8/n)（n 头标准式）、严格递减、末项最小；n=1 时==2^-8（≤1e-15）|1e-15|Press et al. 2021 get_slopes 原式
 * @invariant INV-POS-003|ALiBi 距离线性闭式：bias[h][i][j]==-slopes[h]·(j-i)（j≤i，20 组随机 ≤1e-12）|1e-12|线性偏置定义
 * @invariant INV-POS-004|位置插值闭式：positionInterpolation==pos/s 且 ropeWithInterpolation==rope(x,pos/s,base)（≤1e-12）|1e-12|Chen et al. 2023
 * @invariant INV-POS-005|T5 桶映射：双向模式 d 与 -d 异桶、|d|≥maxDistance clamp 到远桶、桶索引∈[0,nBuckets)（违例计数 0）|0|Raffel et al. 2020
 * @invariant INV-POS-006|偏置施加一致：applyBias==scores+bias（有限位）；-Infinity 偏置位保持 -Infinity 不 NaN（≤1e-12）|1e-12|逐元素相加定义
 * @invariant INV-POS-007|相对性：relativePositionBias 为 Toeplitz 结构，[i][j] 仅依赖 (j-i)（违例计数 0）|0|相对位置定义
 * @invariant INV-POS-008|softmaxRows 对接：ALiBi 偏置施加后每行和==1（因果下三角 ≤1e-12）且 j>i 位归一化后==0|1e-12|偏置+softmax=因果掩码
 * @seed      0x9E3779B9
 */
'use strict';

/* ============================================================================
 * 惰性双端依赖（照抄既有 L4 模块模式）：Node 端 require，浏览器端回退 globalThis
 * （file:// 经典 script src）。模块加载期不触碰任何依赖，保证双端同构与零循环导入。
 *   V() → core-verify.js    (ForgeVerify)   ：caseResult / buildReport / formatTableSafe / TOL
 *   N() → core-numeric.js   (ForgeNumeric)  ：makeRng 确定性采样（禁 Math.random）
 *   A() → attention-forge.js(ForgeAttention)：softmaxRows（偏置施加对接）
 *   R() → rope-forge.js     (ForgeRoPE)     ：rope 旋转（PI 组合复用，不重写）
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

let _R = null;
function R() {
  if (_R) return _R;
  if (typeof require !== 'undefined') { try { _R = require('./rope-forge.js'); } catch (e) {} }
  if (!_R && typeof globalThis !== 'undefined') _R = globalThis.ForgeRoPE || null;
  if (!_R) throw new Error('rope-forge 依赖缺失');
  return _R;
}

/* ============================================================================
 * 矩阵约定（与 attention-forge / rope-forge 一致）：
 *   二维矩阵 = 数组 of Float64Array（行主序）；三维张量 = 数组 of 二维矩阵。
 * ========================================================================== */

/** 单向（因果）模式下 T5 桶映射的未来位哨兵：j>i 不参与相对位置编码，置 -1（调用方施加 -∞ 掩码）。 */
const T5_MASKED = -1;

/** 校验二维矩阵（数组 of Float64Array，行主序），返回 {rows, cols}；非法抛错。 */
function checkMatrix(m, name) {
  if (!Array.isArray(m) || m.length === 0) throw new Error(name + ' 必须为非空二维矩阵（数组 of Float64Array，行主序）');
  const cols = m[0].length;
  if (typeof cols !== 'number' || cols < 1) throw new Error(name + ' 首行宽度非法');
  for (let i = 1; i < m.length; i++) {
    if (!m[i] || m[i].length !== cols) throw new Error(name + ' 第 ' + i + ' 行宽度不一致（期望 ' + cols + '）');
  }
  return { rows: m.length, cols };
}

/** 判断是否为三维张量（nHeads × nq × nk：行元素仍为矩阵）。二维矩阵的行元素为数值，据此区分。 */
function isTensor3D(t) {
  return Array.isArray(t) && t.length > 0 && Array.isArray(t[0]) && t[0].length > 0 &&
    t[0][0] !== null && typeof t[0][0] === 'object' && typeof t[0][0].length === 'number';
}

/* ============================================================================
 * 1. ALiBi 线性偏置（Press et al. 2021, Train Short, Test Long）
 * ========================================================================== */

/**
 * ALiBi 斜率（Press et al. 2021 get_slopes 原式）：
 *   2 的幂头数 n 时 start = 2^(-8/n)，slopes[i] = start·start^i = 2^(-8(i+1)/n)（i=0..n-1），
 *   即 1-indexed 意义下的 2^(-8i/n)（i=1..n）：首项 2^(-8/n)，末项 2^-8，严格递减。
 *   非 2 幂头数：原论文用「最近 2 幂斜率隔点取样」插值；本项目按约定采用等价的几何插值版
 *   slopes[i] = 2^(-8(i+1)/n)——对 2 幂头数与原式逐位一致，且保持首项/严格递减/末项最小性质。
 * @param {number} nHeads 头数（正整数）
 * @returns {Float64Array} 长度 nHeads 的斜率序列，严格递减
 */
function alibiSlopes(nHeads) {
  if (!Number.isInteger(nHeads) || nHeads < 1) throw new Error('nHeads 必须为正整数（ALiBi 按头线性偏置）');
  const slopes = new Float64Array(nHeads);
  const start = Math.pow(2, -8 / nHeads); // Press et al. 2021 原式首项 2^(-8/n)
  for (let i = 0; i < nHeads; i++) slopes[i] = Math.pow(start, i + 1); // 2^(-8(i+1)/n)
  return slopes;
}

/**
 * ALiBi 偏置矩阵：bias[h][i][j] = -slopes[h]·(j−i)（距离越远惩罚越强的线性偏置）。
 * causal=true 时未来位 j>i 置 -Infinity（与 softmax 组合即因果掩码，见 INV-POS-008）。
 * @param {number} nHeads 头数（正整数）
 * @param {number} nq query 序列长（正整数）
 * @param {number} nk key 序列长（正整数）
 * @param {boolean} [causal=false] 是否因果（遮蔽 j>i）
 * @returns {Array<Array<Float64Array>>} nHeads × nq × nk 偏置张量（行主序嵌套）
 */
function alibiBias(nHeads, nq, nk, causal) {
  const slopes = alibiSlopes(nHeads); // 复用同一斜率源（含 nHeads 校验）
  if (!Number.isInteger(nq) || nq < 1) throw new Error('nq 必须为正整数');
  if (!Number.isInteger(nk) || nk < 1) throw new Error('nk 必须为正整数');
  const bias = new Array(nHeads);
  for (let h = 0; h < nHeads; h++) {
    const head = new Array(nq);
    const s = slopes[h];
    for (let i = 0; i < nq; i++) {
      const row = new Float64Array(nk);
      for (let j = 0; j < nk; j++) {
        row[j] = (causal && j > i) ? -Infinity : -s * (j - i);
      }
      head[i] = row;
    }
    bias[h] = head;
  }
  return bias;
}

/**
 * 偏置施加：把 bias 逐元素加到 scores 上（同形），返回新矩阵（不修改入参）。
 * 支持 2D（nq×nk 单头分数）与 3D（nHeads×nq×nk，与 alibiBias 输出同形，按头施加）。
 * -Infinity 偏置位结果保持 -Infinity（有限分数 + (-Infinity) = -Infinity，不产生 NaN）。
 * @param {Array<Float64Array>|Array<Array<Float64Array>>} scores 注意力分数
 * @param {Array<Float64Array>|Array<Array<Float64Array>>} bias 偏置（与 scores 同形）
 * @returns {Array<Float64Array>|Array<Array<Float64Array>>} 与 scores 同形的新矩阵
 */
function applyBias(scores, bias) {
  if (isTensor3D(scores) || isTensor3D(bias)) {
    if (!isTensor3D(scores) || !isTensor3D(bias)) throw new Error('applyBias 三维模式下 scores 与 bias 必须同为 nHeads×nq×nk');
    if (scores.length !== bias.length) throw new Error('applyBias 头数不一致：scores=' + scores.length + ' bias=' + bias.length);
    return scores.map((s2, h) => applyBias(s2, bias[h]));
  }
  const a = checkMatrix(scores, 'scores');
  const b = checkMatrix(bias, 'bias');
  if (a.rows !== b.rows || a.cols !== b.cols) {
    throw new Error('applyBias 形状不一致：scores ' + a.rows + '×' + a.cols + ' vs bias ' + b.rows + '×' + b.cols);
  }
  const out = new Array(a.rows);
  for (let i = 0; i < a.rows; i++) {
    const row = new Float64Array(a.cols);
    for (let j = 0; j < a.cols; j++) {
      const bv = bias[i][j];
      row[j] = (bv === -Infinity) ? -Infinity : scores[i][j] + bv;
    }
    out[i] = row;
  }
  return out;
}

/* ============================================================================
 * 2. T5 相对位置桶（Raffel et al. 2020, Exploring the Limits of Transfer Learning）
 * ========================================================================== */

/**
 * T5 大小桶拆分公式（_relative_position_bucket 内部核心，Raffel et al. 2020）：
 *   前 maxExact = nBuckets/2 个距离用精确桶（distance < maxExact → bucket = distance）；
 *   其余按对数分桶：maxExact + floor(log(d/maxExact)/log(maxDistance/maxExact)·(nBuckets−maxExact))，
 *   并 clamp 到最远桶 nBuckets−1——|d| ≥ maxDistance 时全部落入同一远桶（对数分桶单调不降）。
 * @param {number} distance 非负整数相对距离（已取绝对值）
 * @param {number} nBuckets 半桶数（T5 内部 num_buckets//2，即单向可用桶数，≥2）
 * @param {number} maxDistance 对数分桶上界（正整数，须 > nBuckets/2 以保证对数分母为正）
 * @returns {number} 桶索引 ∈ [0, nBuckets)
 */
function bucketDistanceToIndex(distance, nBuckets, maxDistance) {
  if (!Number.isInteger(distance) || distance < 0) throw new Error('distance 必须为非负整数（T5 相对距离）');
  if (!Number.isInteger(nBuckets) || nBuckets < 2) throw new Error('nBuckets（半桶数）必须为 ≥2 的整数');
  if (!Number.isInteger(maxDistance) || maxDistance < 2) throw new Error('maxDistance 必须为 ≥2 的整数');
  const maxExact = Math.floor(nBuckets / 2);
  if (maxDistance <= maxExact) {
    throw new Error('maxDistance 必须大于 maxExact=' + maxExact + '（保证 log(maxDistance/maxExact) > 0）');
  }
  if (distance < maxExact) return distance; // 精确桶（小距离一一对应）
  // 对数分桶（T5 原式；正值域内 Math.floor 即 torch .to(int64) 的向零截断）
  let idx = maxExact + Math.floor(
    Math.log(distance / maxExact) / Math.log(maxDistance / maxExact) * (nBuckets - maxExact)
  );
  if (idx > nBuckets - 1) idx = nBuckets - 1; // 远距离 clamp 到同一远桶
  if (idx < 0) idx = 0;
  return idx;
}

/**
 * T5 相对位置桶映射（Raffel et al. 2020）：返回 nq × nk 整数桶索引矩阵。
 *   bidirectional=true：d = j−i；d>0 时桶偏移 +nBuckets/2 后对 |d| 分桶，d≤0 直接对 |d| 分桶
 *     → 对称距离 d 与 −d 落入不同半区（异桶），全部桶索引 ∈ [0, nBuckets)。
 *   bidirectional=false（单向/因果）：仅可见位 j≤i 计算桶（距离 i−j，桶 ∈ [0, nBuckets/2)）；
 *     未来位 j>i 置哨兵 T5_MASKED = -1（调用方据此施加 -∞ 掩码；此为对 T5 原式「未来位同桶」
 *     用法的因果适配，已文档化）。
 * @param {number} nq query 序列长（正整数）
 * @param {number} nk key 序列长（正整数）
 * @param {number} [nBuckets=32] 总桶数（T5 默认 32；内部对半拆分双向/单向）
 * @param {number} [maxDistance=128] 对数分桶上界（T5 默认 128）
 * @param {boolean} [bidirectional=true] 是否双向
 * @returns {Array<Float64Array>} nq × nk 桶索引矩阵（值为精确整数）
 */
function t5RelativeBuckets(nq, nk, nBuckets, maxDistance, bidirectional) {
  nBuckets = (nBuckets === undefined || nBuckets === null) ? 32 : nBuckets;
  maxDistance = (maxDistance === undefined || maxDistance === null) ? 128 : maxDistance;
  bidirectional = (bidirectional === undefined || bidirectional === null) ? true : !!bidirectional;
  if (!Number.isInteger(nq) || nq < 1) throw new Error('nq 必须为正整数');
  if (!Number.isInteger(nk) || nk < 1) throw new Error('nk 必须为正整数');
  if (!Number.isInteger(nBuckets) || nBuckets < 4) throw new Error('nBuckets 必须为 ≥4 的整数（T5 内部对半拆分双向/单向）');
  const half = Math.floor(nBuckets / 2);
  const B = new Array(nq);
  for (let i = 0; i < nq; i++) {
    const row = new Float64Array(nk);
    for (let j = 0; j < nk; j++) {
      const d = j - i;
      if (bidirectional) {
        const offset = d > 0 ? half : 0;
        row[j] = offset + bucketDistanceToIndex(Math.abs(d), half, maxDistance);
      } else {
        row[j] = d > 0 ? T5_MASKED : bucketDistanceToIndex(-d, half, maxDistance);
      }
    }
    B[i] = row;
  }
  return B;
}

/* ============================================================================
 * 3. 位置插值（Chen et al. 2023, Extending Context Window）与 RoPE 组合
 * ========================================================================== */

/**
 * 位置插值（Position Interpolation, Chen et al. 2023）：插值后位置 = pos / scale。
 * 将预训练上下文长度 L 扩展到 L·scale（scale>1）时按比例缩小位置下标，
 * 使全部位置落入预训练覆盖区间（而非外插），从而无需微调保持困惑度。
 * @param {number|Array<number>|Float64Array} pos 位置（标量或序列）
 * @param {number} scale 插值比例（有限正数；scale ≤ 0 或非有限抛错）
 * @returns {number|Array<number>|Float64Array} 与入参同形的缩放后位置
 *          （标量→标量；Float64Array→Float64Array；其余序列→Array）
 */
function positionInterpolation(pos, scale) {
  if (typeof scale !== 'number' || !Number.isFinite(scale) || scale <= 0) {
    throw new Error('scale 必须为有限正数（Position Interpolation 插值比例，Chen et al. 2023）');
  }
  if (typeof pos === 'number') {
    if (!Number.isFinite(pos)) throw new Error('pos 必须为有限数');
    return pos / scale;
  }
  if (!pos || typeof pos.length !== 'number' || pos.length === 0) {
    throw new Error('pos 必须为非空位置序列（或有限标量）');
  }
  const out = pos instanceof Float64Array ? new Float64Array(pos.length) : new Array(pos.length);
  for (let i = 0; i < pos.length; i++) {
    const v = pos[i];
    if (typeof v !== 'number' || !Number.isFinite(v)) throw new Error('pos[' + i + '] 必须为有限数');
    out[i] = v / scale;
  }
  return out;
}

/**
 * RoPE + 位置插值组合：先对位置做插值缩小，再施加 RoPE 旋转
 * （复用 rope-forge 的 rope，不重写旋转逻辑）：ropeWithInterpolation(x, pos, base, scale)
 * == rope(x, pos/scale, base)。证明 PI 与 RoPE 可组合：扩展上下文只需缩放旋转角，
 * RoPE 的正交性/相对位置性质不受影响（详见 rope-forge 不变量）。
 * @param {Float64Array|Array<number>} x 词向量（长度 d，偶数）
 * @param {number} pos 原始位置（非负整数）
 * @param {number} [base=10000] RoPE 底数
 * @param {number} scale 插值比例（有限正数）
 * @returns {Float64Array} 旋转后的向量
 */
function ropeWithInterpolation(x, pos, base, scale) {
  const p = positionInterpolation(pos, scale); // 标量路径：pos/scale（含 scale 校验）
  base = (base === undefined || base === null) ? 10000 : base;
  return R().rope(x, p, base);
}

/* ============================================================================
 * 4. 纯相对位置偏置
 * ========================================================================== */

/**
 * 纯相对位置偏置：bias[i][j] = f(j−i) = −|j−i|，只依赖相对距离（与绝对位置无关），
 * 矩阵为 Toeplitz 结构（沿对角线恒定）。可视为 ALiBi 线性偏置的对称化（斜率 1）：
 * 任意平移查询/键序列，偏置矩阵不变。
 * @param {number} nq query 序列长（正整数）
 * @param {number} nk key 序列长（正整数）
 * @returns {Array<Float64Array>} nq × nk 偏置矩阵
 */
function relativePositionBias(nq, nk) {
  if (!Number.isInteger(nq) || nq < 1) throw new Error('nq 必须为正整数');
  if (!Number.isInteger(nk) || nk < 1) throw new Error('nk 必须为正整数');
  const B = new Array(nq);
  for (let i = 0; i < nq; i++) {
    const row = new Float64Array(nk);
    for (let j = 0; j < nk; j++) row[j] = -Math.abs(j - i);
    B[i] = row;
  }
  return B;
}

/* ============================================================================
 * selfCheck — 8 条硬不变量（forge-selfcheck/1.0）
 * ========================================================================== */
function selfCheck(opts) {
  opts = opts || {};
  const { caseResult, TOL } = V();
  const cases = [];
  const invariants = [];
  const rng = N().makeRng(0x9e3779b9);

  // INV-POS-001 ALiBi 因果结构：j>i 位 -Infinity，下三角全有限（违例计数 0）
  {
    const nHeads = 4, nq = 6, nk = 8;
    const bias = alibiBias(nHeads, nq, nk, true);
    let viol = 0;
    for (let h = 0; h < nHeads; h++) {
      for (let i = 0; i < nq; i++) {
        for (let j = 0; j < nk; j++) {
          if (j > i) { if (bias[h][i][j] !== -Infinity) viol++; }
          else if (!Number.isFinite(bias[h][i][j])) viol++;
        }
      }
    }
    cases.push(caseResult('INV-POS-001', 'ALiBi 因果结构：j>i 置 -Infinity、下三角全有限', viol, 0, 0));
  }

  // INV-POS-002 ALiBi 斜率性质：首项 2^(-8/n)、严格递减、末项最小；n=1 → 2^-8
  {
    let worst = 0;
    const headCounts = [1, 2, 3, 4, 5, 7, 8, 12, 16, 17, 32];
    for (const n of headCounts) {
      const s = alibiSlopes(n);
      worst = Math.max(worst, Math.abs(s[0] - Math.pow(2, -8 / n))); // 首项 == 2^(-8/n)
      for (let i = 1; i < n; i++) if (!(s[i - 1] > s[i])) worst = Math.max(worst, 1); // 严格递减
      for (let i = 0; i < n; i++) if (s[i] < s[n - 1]) worst = Math.max(worst, 1); // 末项最小
    }
    worst = Math.max(worst, Math.abs(alibiSlopes(1)[0] - Math.pow(2, -8))); // n=1 → 2^-8
    cases.push(caseResult('INV-POS-002', 'ALiBi 斜率：首项 2^(-8/n)、严格递减、末项最小；n=1→2^-8', worst, 0, 1e-15));
  }

  // INV-POS-003 ALiBi 距离线性闭式：bias[h][i][j] == -slopes[h]·(j-i)（j≤i，20 组随机）
  {
    let mx = 0;
    for (let t = 0; t < 20; t++) {
      const nHeads = 1 + Math.floor(rng() * 8);
      const nq = 1 + Math.floor(rng() * 8);
      const nk = 1 + Math.floor(rng() * 8);
      const slopes = alibiSlopes(nHeads);
      const bias = alibiBias(nHeads, nq, nk, false);
      for (let h = 0; h < nHeads; h++) {
        for (let i = 0; i < nq; i++) {
          for (let j = 0; j <= i && j < nk; j++) { // 有效域：下三角 ∩ 矩阵范围（j ≤ i 且 j < nk）
            const ref = -slopes[h] * (j - i);
            mx = Math.max(mx, Math.abs(bias[h][i][j] - ref));
          }
        }
      }
    }
    cases.push(caseResult('INV-POS-003', 'ALiBi 距离线性闭式 bias=-slopes·(j−i)（j≤i，20 组随机）', mx, 0, TOL.RESID));
  }

  // INV-POS-004 位置插值闭式：PI==pos/s 且 PI∘RoPE == rope(x, pos/s, base)
  {
    let mx = 0;
    // (a) positionInterpolation(pos, s) == pos/s（逐元素）
    for (let t = 0; t < 20; t++) {
      const len = 1 + Math.floor(rng() * 16);
      const scale = 1 + rng() * 15;
      const pos = new Float64Array(len);
      for (let i = 0; i < len; i++) pos[i] = Math.floor(rng() * 512);
      const got = positionInterpolation(pos, scale);
      for (let i = 0; i < len; i++) mx = Math.max(mx, Math.abs(got[i] - pos[i] / scale));
    }
    // (b) ropeWithInterpolation == 直接对缩放后位置调 rope
    for (let t = 0; t < 20; t++) {
      const d = 8;
      const x = new Float64Array(d);
      for (let i = 0; i < d; i++) x[i] = rng() * 2 - 1;
      const pos = Math.floor(rng() * 64);
      const base = 500 + Math.floor(rng() * 10000);
      const scale = 1 + rng() * 15;
      const got = ropeWithInterpolation(x, pos, base, scale);
      const ref = R().rope(x, pos / scale, base);
      for (let i = 0; i < d; i++) mx = Math.max(mx, Math.abs(got[i] - ref[i]));
    }
    cases.push(caseResult('INV-POS-004', '位置插值闭式：PI==pos/s 且 PI∘RoPE==RoPE(x,pos/s,base)', mx, 0, TOL.RESID));
  }

  // INV-POS-005 T5 桶映射性质：d 与 -d 异桶、远距 clamp、范围 [0,nBuckets)、精确桶、单调不降、单向哨兵
  {
    const nq = 20, nk = 20, nBuckets = 32, maxDistance = 16;
    const half = nBuckets >> 1; // 16
    const maxExact = half >> 1; // 8
    const B = t5RelativeBuckets(nq, nk, nBuckets, maxDistance, true);
    let viol = 0;
    // (a) 双向模式：相对距离 d 与 -d 映射到不同桶
    for (let i = 0; i < nq; i++) {
      for (let j = i + 1; j < nk; j++) {
        if (B[i][j] === B[j][i]) viol++; // (i,j): d=j−i>0 落上半区；(j,i): d=i−j<0 落下半区
      }
    }
    // (b) |d| >= maxDistance → 同一远桶（正距 nBuckets-1，负距 half-1）
    for (let i = 0; i < nq; i++) {
      for (let j = 0; j < nk; j++) {
        const d = j - i;
        if (Math.abs(d) >= maxDistance) {
          const far = d > 0 ? nBuckets - 1 : half - 1;
          if (B[i][j] !== far) viol++;
        }
      }
    }
    // (c) 桶索引 ∈ [0, nBuckets) 且为整数
    for (let i = 0; i < nq; i++) {
      for (let j = 0; j < nk; j++) {
        if (!(B[i][j] >= 0 && B[i][j] < nBuckets && Number.isInteger(B[i][j]))) viol++;
      }
    }
    // (d) 辅助函数：小距离精确桶、远距 clamp 半区远桶、对数分桶单调不降
    let prev = -1;
    for (let d = 0; d <= 4 * maxDistance; d++) {
      const idx = bucketDistanceToIndex(d, half, maxDistance);
      if (d < maxExact && idx !== d) viol++;
      if (d >= maxDistance && idx !== half - 1) viol++;
      if (idx < prev) viol++;
      prev = idx;
    }
    // (e) 单向模式：j>i 哨兵 T5_MASKED，j≤i 有效桶 ∈ [0, nBuckets)
    const Bu = t5RelativeBuckets(nq, nk, nBuckets, maxDistance, false);
    for (let i = 0; i < nq; i++) {
      for (let j = 0; j < nk; j++) {
        if (j > i) { if (Bu[i][j] !== T5_MASKED) viol++; }
        else if (!(Bu[i][j] >= 0 && Bu[i][j] < nBuckets)) viol++;
      }
    }
    cases.push(caseResult('INV-POS-005', 'T5 桶映射：d/−d 异桶、远距 clamp、范围 [0,nBuckets)', viol, 0, 0));
  }

  // INV-POS-006 偏置施加一致性：applyBias==scores+bias（有限位）；-Infinity 位不 NaN（含 3D 按头路径）
  {
    let mx = 0;
    // (a) 有限位逐元素一致（2D）
    for (let t = 0; t < 20; t++) {
      const nq = 1 + Math.floor(rng() * 6), nk = 1 + Math.floor(rng() * 6);
      const S = new Array(nq), Bb = new Array(nq);
      for (let i = 0; i < nq; i++) {
        S[i] = new Float64Array(nk);
        Bb[i] = new Float64Array(nk);
        for (let j = 0; j < nk; j++) { S[i][j] = rng() * 4 - 2; Bb[i][j] = rng() * 4 - 2; }
      }
      const out = applyBias(S, Bb);
      for (let i = 0; i < nq; i++) {
        for (let j = 0; j < nk; j++) mx = Math.max(mx, Math.abs(out[i][j] - (S[i][j] + Bb[i][j])));
      }
    }
    // (b) ALiBi 因果偏置（3D）：-Infinity 位保持 -Infinity，有限位仍逐元素一致
    const bias3 = alibiBias(3, 5, 5, true);
    const S3 = new Array(3);
    for (let h = 0; h < 3; h++) {
      S3[h] = new Array(5);
      for (let i = 0; i < 5; i++) {
        S3[h][i] = new Float64Array(5);
        for (let j = 0; j < 5; j++) S3[h][i][j] = rng() * 2 - 1;
      }
    }
    const out3 = applyBias(S3, bias3);
    let infViol = 0;
    for (let h = 0; h < 3; h++) {
      for (let i = 0; i < 5; i++) {
        for (let j = 0; j < 5; j++) {
          const v = out3[h][i][j];
          if (j > i) { if (v !== -Infinity) infViol++; }
          else if (!Number.isFinite(v)) infViol++;
          else mx = Math.max(mx, Math.abs(v - (S3[h][i][j] + bias3[h][i][j])));
        }
      }
    }
    cases.push(caseResult('INV-POS-006', '偏置施加：applyBias==scores+bias；-Infinity 位不 NaN', Math.max(mx, infViol), 0, TOL.RESID));
  }

  // INV-POS-007 相对性：relativePositionBias 为 Toeplitz 结构（仅依赖 j−i）
  {
    const nq = 7, nk = 9;
    const B = relativePositionBias(nq, nk);
    let viol = 0;
    for (let i = 0; i < nq; i++) {
      for (let j = 0; j < nk; j++) {
        if (B[i][j] !== -Math.abs(j - i)) viol++;
        for (let i2 = 0; i2 < nq; i2++) { // 同一对角线（j−i 相同）上任意两点相等
          const j2 = j + (i2 - i);
          if (j2 >= 0 && j2 < nk && B[i][j] !== B[i2][j2]) viol++;
        }
      }
    }
    cases.push(caseResult('INV-POS-007', '相对性：relativePositionBias 为 Toeplitz（仅依赖 j−i）', viol, 0, 0));
  }

  // INV-POS-008 与 softmaxRows 对接：ALiBi 偏置 + softmax = 因果掩码（行和=1、j>i 归一化=0）
  {
    const nq = 6, nk = 6;
    const S = new Array(nq);
    for (let i = 0; i < nq; i++) {
      S[i] = new Float64Array(nk);
      for (let j = 0; j < nk; j++) S[i][j] = rng() * 4 - 2;
    }
    const bias = alibiBias(2, nq, nk, true)[0]; // 取头 0 的因果偏置（上三角 -Infinity）
    const biased = applyBias(S, bias);
    // 偏置已含 -Infinity 因果掩码，softmaxRows 不再传 causal——证明组合本身即正确因果掩码
    const P = A().softmaxRows(biased, false);
    let rowErr = 0, upper = 0;
    for (let i = 0; i < nq; i++) {
      let s = 0;
      for (let j = 0; j < nk; j++) {
        s += P[i][j];
        if (j > i) upper = Math.max(upper, P[i][j]);
      }
      rowErr = Math.max(rowErr, Math.abs(s - 1));
    }
    cases.push(caseResult('INV-POS-008', 'softmaxRows 对接：ALiBi 后行和=1 且 j>i 归一化=0', Math.max(rowErr, upper), 0, TOL.RESID));
  }

  // 不变量登记（worst = 实测绝对误差）
  for (const id of ['INV-POS-001', 'INV-POS-002', 'INV-POS-003', 'INV-POS-004', 'INV-POS-005', 'INV-POS-006', 'INV-POS-007', 'INV-POS-008']) {
    const c = cases.find(x => x.id === id);
    invariants.push({ id, expr: id, tol: c.tol, worst: c.abs_err, status: c.status, note: '' });
  }

  const started = Date.now();
  return V().buildReport({
    module: 'position-encoding-forge', version: '1.0.0', layer: 'L4', cases, invariants,
    seed: '0x9E3779B9', started,
    tol: { 'INV-POS-001': 0, 'INV-POS-002': 1e-15, 'INV-POS-003': TOL.RESID, 'INV-POS-004': TOL.RESID, 'INV-POS-005': 0, 'INV-POS-006': TOL.RESID, 'INV-POS-007': 0, 'INV-POS-008': TOL.RESID }
  });
}

const API = {
  name: 'position-encoding-forge', version: '1.0.0', layer: 'L4',
  alibiSlopes, alibiBias, applyBias, t5RelativeBuckets, bucketDistanceToIndex,
  positionInterpolation, ropeWithInterpolation, relativePositionBias, selfCheck
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgePosEnc = API;

if (typeof process !== 'undefined' && process.argv[1] && /position-encoding-forge\.js$/.test(process.argv[1]) && (process.argv.includes('--self-check') || process.argv.includes('--json'))) {
  const r = selfCheck({ quick: process.argv.includes('--quick') });
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : V().formatTableSafe(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
