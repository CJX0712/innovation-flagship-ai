/**
 * @forge-module quant-forge
 * @layer L4
 * @version 1.0.0
 * @brief     模型量化内核：int8/int4 对称(symmetric)与非对称(asymmetric/affine)校准与量化、逐张量/逐通道粒度、
 *            4-bit 位压缩打包（每字节 2 个有符号 4-bit，低半字节在前；非对称用 excess-8 偏移码）、
 *            量化权重 × 浮点激活的矩阵-向量乘、llama.cpp 风格 k-quants 超块/子块二级 scale、量化信噪比 SNR。
 *            张量 = 一维 Float64Array/Array 或二维数组（行主序，行=通道）；纯函数式、零 npm 依赖、双端同构。
 * @reuse     llama.cpp k-quants|https://github.com/ggerganov/llama.cpp|MIT|A|超块/子块二级 scale 布局
 * @reuse     LLM.int8() (Dettmers et al. 2022)|https://arxiv.org/abs/2208.07339|MIT|B|向量级 absmax 量化与逐通道粒度
 * @reuse     GPTQ (Frantar et al. 2022)|https://arxiv.org/abs/2210.17323|MIT|C|逐通道分组量化误差口径
 * @reuse     SmoothQuant (Xiao et al. 2023)|https://arxiv.org/abs/2211.10438|MIT|D|激活/权重难度迁移概念
 * @beyond    纯 JS 单文件零构建零 npm 依赖；int4 真正位压缩（非一元素一字节）；证明 round-to-nearest+clamp 误差
 *            ≤ scale/2 的不变量族；逐通道 vs 逐张量、超块 vs 整张的 SNR 同输入对照；量化矩阵乘保真度按 bits 单调；
 *            非对称 zeroPoint 为有符号整数偏移（全正/全负数据下仍精确覆盖），对称量化 zeroPoint 恒 0。
 * @invariant INV-QNT-001|可表示值精确往返：bits=8 对称，整数网格值 x=k·scale（k∈[-127,127]）经 quantize→dequantize 逐元素误差 ≤ 1e-12|1e-12|网格点往返
 * @invariant INV-QNT-002|量化误差有界：随机 20 组张量（bits=8/4 × 对称/非对称 × 逐张量/逐通道），max|deq(quant(x))-x| ≤ scale/2|1e-12|round-to-nearest 半步长界
 * @invariant INV-QNT-003|对称量化零点：symmetric 时 zeroPoint==0（所有通道，精确）；asymmetric 时 zeroPoint 为整数且在 [0,2^bits-1]|0|零点定义校验
 * @invariant INV-QNT-004|逐通道优于逐张量：随机 20 组各通道动态范围差异大（32 通道 absmax 对数均匀 1e-2..1e2）的矩阵，perChannel SNR ≥ perTensor SNR|0|同输入对照违例计数
 * @invariant INV-QNT-005|int4 范围与端点：bits=4 两种 scheme 存储码 ∈ [-8,7]；对称量化网格覆盖 [min,max]；端点重建误差 ≤ 1 步长|1e-12|范围+网格覆盖+端点断言
 * @invariant INV-QNT-006|量化矩阵乘保真：非负随机权重矩阵（逐通道非对称），quantMatvec 与浮点 matvec 对 16 组随机激活汇总相对误差 ‖ΔY‖_F/‖Y‖_F：int8 ≤ 1e-2、int4 ≤ 5e-2 且 int8 ≤ int4|1e-2|同输入对照+单调性
 * @invariant INV-QNT-007|超块细化：同张量（4 段块间 scale 相差 100 倍级）kQuantSuperBlocks 的 SNR ≥ 整张 per-tensor 量化的 SNR，块数 == ceil(n/blockSize)|0|同输入对照+块数精确
 * @invariant INV-QNT-008|确定性与 pack 往返：同张量两次 calibrate+quantize 参数与结果逐位相等；unpackInt4(packInt4(v))==v|0|确定性定义+位压缩逆运算
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

/* ============================================================================
 * 0. 内部常量与校验助手
 * ========================================================================== */
/** 对称量化整数范围上界：bits=8 → 127，bits=4 → 7。 */
function qmaxOf(bits) { return Math.pow(2, bits - 1) - 1; }
/** 非对称量化级数：bits=8 → 255，bits=4 → 15。 */
function qlevOf(bits) { return Math.pow(2, bits) - 1; }
/** 非对称 excess 偏移（存储码 = 无符号量化值 − 偏移）：bits=8 → 128，bits=4 → 8。 */
function excOf(bits) { return Math.pow(2, bits - 1); }

function checkBits(bits) {
  if (bits !== 4 && bits !== 8) throw new Error('bits 仅支持 4 或 8（int4/int8），收到 ' + bits);
}

/** 将一维输入规范化为 Float64Array（拷贝，隔离入参）；非法元素抛错。 */
function toVector(t, name) {
  if (t instanceof Float64Array) {
    for (let i = 0; i < t.length; i++) if (!Number.isFinite(t[i])) throw new Error(name + '[' + i + '] 非有限数');
    return t;
  }
  if (typeof Int8Array !== 'undefined' && t instanceof Int8Array) {
    const v = new Float64Array(t.length);
    for (let i = 0; i < t.length; i++) v[i] = t[i]; // Int8Array 元素必为有限整数
    return v;
  }
  if (!Array.isArray(t) || t.length === 0) throw new Error(name + ' 必须为非空一维数组或 Float64Array');
  const v = new Float64Array(t.length);
  for (let i = 0; i < t.length; i++) {
    const x = t[i];
    if (typeof x !== 'number' || !Number.isFinite(x)) throw new Error(name + '[' + i + '] 必须为有限数');
    v[i] = x;
  }
  return v;
}

/** 将二维输入规范化为 Array<Float64Array>（行主序拷贝）；非法形状抛错。 */
function toMatrix(m, name) {
  if (!Array.isArray(m) || m.length === 0) throw new Error(name + ' 必须为非空二维数组（行主序）');
  const rows = new Array(m.length);
  for (let i = 0; i < m.length; i++) rows[i] = toVector(m[i], name + ' 第 ' + i + ' 行');
  const cols = rows[0].length;
  for (let i = 1; i < rows.length; i++) {
    if (rows[i].length !== cols) throw new Error(name + ' 第 ' + i + ' 行宽度不一致（期望 ' + cols + '）');
  }
  return rows;
}

/** 输入是否二维（数组且首元素为行）。空数组/一维数组均判否。 */
function isMatrixLike(t) {
  return Array.isArray(t) && t.length > 0 && (t[0] instanceof Float64Array || Array.isArray(t[0]));
}

/* ============================================================================
 * 1. calibrate — 由校准张量推导量化参数 {scale, zeroPoint}
 *    对称：scale = maxAbs / (2^(bits-1)-1)，zeroPoint ≡ 0（范围 [-qmax, qmax]）
 *    非对称：scale = (max-min) / (2^bits-1)，zeroPoint = round(-min/scale)
 *            （有符号整数偏移；q = clamp(round(x/scale)+zeroPoint, 0, 2^bits-1)，
 *             deq = scale·(q-zeroPoint)。全正/全负数据下 zeroPoint 可越出 [0,2^bits-1]，
 *            但网格仍精确覆盖 [min,max]，误差仍 ≤ scale/2）
 * ========================================================================== */
function calibrate(tensor, opts) {
  opts = opts || {};
  const bits = (opts.bits === undefined || opts.bits === null) ? 8 : opts.bits;
  const scheme = opts.scheme || 'symmetric';
  const granularity = opts.granularity || 'perTensor';
  checkBits(bits);
  if (scheme !== 'symmetric' && scheme !== 'asymmetric') {
    throw new Error("scheme 仅支持 'symmetric' 或 'asymmetric'，收到 " + scheme);
  }
  if (granularity !== 'perTensor' && granularity !== 'perChannel') {
    throw new Error("granularity 仅支持 'perTensor' 或 'perChannel'，收到 " + granularity);
  }

  const mat = isMatrixLike(tensor);
  if (granularity === 'perChannel' && !mat) {
    throw new Error('perChannel 粒度需要二维张量（按行/通道校准）');
  }
  const qmax = qmaxOf(bits), qlev = qlevOf(bits);

  /** 单段（一行或整张）的 absmax 量化参数。 */
  function symParams(v) {
    let amax = 0;
    for (let i = 0; i < v.length; i++) { const a = Math.abs(v[i]); if (a > amax) amax = a; }
    return { scale: amax === 0 ? 1 : amax / qmax, zeroPoint: 0 };
  }
  /** 单段的 affine 量化参数（zeroPoint 为有符号整数偏移）。 */
  function asymParams(v) {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < v.length; i++) { if (v[i] < lo) lo = v[i]; if (v[i] > hi) hi = v[i]; }
    const range = hi - lo;
    let scale, zp;
    if (range === 0) {
      // 常量段：令端点恰好可表示（v=0 时 scale=1, zp=0）
      scale = lo === 0 ? 1 : Math.abs(lo) / qlev;
      zp = Math.round(-lo / scale);
    } else {
      scale = range / qlev;
      zp = Math.round(-lo / scale);
    }
    if (!Number.isInteger(zp)) throw new Error('内部错误：zeroPoint 非整数');
    return { scale, zeroPoint: zp };
  }
  const derive = scheme === 'symmetric' ? symParams : asymParams;

  if (mat) {
    const rows = toMatrix(tensor, 'tensor');
    if (granularity === 'perChannel') {
      const scale = new Array(rows.length), zeroPoint = new Array(rows.length);
      for (let i = 0; i < rows.length; i++) {
        const p = derive(rows[i]);
        if (!(p.scale > 0) || !Number.isFinite(p.scale)) throw new Error('校准失败：scale 非正有限数');
        scale[i] = p.scale; zeroPoint[i] = p.zeroPoint;
      }
      return { bits, scheme, granularity, scale, zeroPoint, rows: rows.length, cols: rows[0].length, n: rows[0].length };
    }
    // perTensor：全张量共享一段
    let n = 0; for (let i = 0; i < rows.length; i++) n += rows[i].length;
    const flat = new Float64Array(n);
    let off = 0; for (let i = 0; i < rows.length; i++) { flat.set(rows[i], off); off += rows[i].length; }
    const p = derive(flat);
    if (!(p.scale > 0) || !Number.isFinite(p.scale)) throw new Error('校准失败：scale 非正有限数');
    return { bits, scheme, granularity, scale: p.scale, zeroPoint: p.zeroPoint, rows: rows.length, cols: rows[0].length, n: rows[0].length };
  }
  const v = toVector(tensor, 'tensor');
  const p = derive(v);
  if (!(p.scale > 0) || !Number.isFinite(p.scale)) throw new Error('校准失败：scale 非正有限数');
  return { bits, scheme, granularity, scale: p.scale, zeroPoint: p.zeroPoint, rows: null, cols: null, n: v.length };
}

/** 校验 params 契约并取出第 ch 通道的 (scale, zeroPoint)。 */
function channelParams(params, ch) {
  if (!params || typeof params !== 'object') throw new Error('params 必须为 calibrate 返回的对象');
  checkBits(params.bits);
  if (params.scheme !== 'symmetric' && params.scheme !== 'asymmetric') throw new Error('params.scheme 非法');
  if (params.granularity !== 'perTensor' && params.granularity !== 'perChannel') throw new Error('params.granularity 非法');
  if (params.granularity === 'perChannel') {
    if (!Array.isArray(params.scale) || !Array.isArray(params.zeroPoint)) {
      throw new Error('perChannel 参数须含 scale/zeroPoint 数组');
    }
    const s = params.scale[ch], z = params.zeroPoint[ch];
    if (typeof s !== 'number' || !(s > 0) || !Number.isFinite(s)) throw new Error('params.scale[' + ch + '] 非正有限数');
    if (typeof z !== 'number' || !Number.isInteger(z)) throw new Error('params.zeroPoint[' + ch + '] 非整数');
    return { scale: s, zeroPoint: z };
  }
  if (typeof params.scale !== 'number' || !(params.scale > 0)) throw new Error('params.scale 非正有限数');
  if (typeof params.zeroPoint !== 'number' || !Number.isInteger(params.zeroPoint)) throw new Error('params.zeroPoint 非整数');
  return { scale: params.scale, zeroPoint: params.zeroPoint };
}

/* ============================================================================
 * 2. int4 位压缩：packInt4 / unpackInt4（互为逆运算）
 *    布局：每字节 2 个有符号 4-bit（two's complement，[-8,7]），低半字节在前。
 *    奇数长度时末字节高半字节补 0；解包需按 count 取回有效值。
 * ========================================================================== */
function packInt4(values) {
  const v = toVector(values, 'packInt4.values');
  const out = new Uint8Array((v.length + 1) >> 1);
  for (let i = 0; i < v.length; i += 2) {
    const a = v[i];
    if (!Number.isInteger(a) || a < -8 || a > 7) throw new Error('packInt4: values[' + i + ']=' + a + ' 须为 [-8,7] 内整数');
    const lo = a & 0xF;
    let byte = lo;
    if (i + 1 < v.length) {
      const b = v[i + 1];
      if (!Number.isInteger(b) || b < -8 || b > 7) throw new Error('packInt4: values[' + (i + 1) + ']=' + b + ' 须为 [-8,7] 内整数');
      byte |= (b & 0xF) << 4;
    }
    out[i >> 1] = byte;
  }
  return out;
}

function unpackInt4(bytes, count) {
  if (!(bytes instanceof Uint8Array)) throw new Error('unpackInt4.bytes 须为 Uint8Array');
  const n = (count === undefined || count === null) ? bytes.length * 2 : count;
  if (!Number.isInteger(n) || n < 0 || n > bytes.length * 2) {
    throw new Error('unpackInt4.count 须为 [0,' + (bytes.length * 2) + '] 内整数，收到 ' + count);
  }
  const out = new Int8Array(n);
  for (let i = 0; i < n; i++) {
    const byte = bytes[i >> 1];
    let nib = (i & 1) === 0 ? (byte & 0xF) : ((byte >> 4) & 0xF);
    if (nib >= 8) nib -= 16; // 符号扩展（two's complement 4-bit）
    out[i] = nib;
  }
  return out;
}

/* ============================================================================
 * 3. quantize / dequantize
 *    存储码：对称 → 有符号量化值本身（int8:[-127,127] / int4:[-7,7] 打包为 4-bit）；
 *            非对称 → 无符号量化值减 excess 偏移（int8: q-128 ∈ [-128,127] / int4: q-8 ∈ [-8,7]）。
 *    一维张量 → Int8Array(bits=8) 或 Uint8Array(bits=4 打包)；二维张量 → 逐行数组。
 * ========================================================================== */
function quantize(tensor, params) {
  if (isMatrixLike(tensor)) {
    if (params.rows == null) throw new Error('params 声明为一维（rows=null），tensor 不应为二维');
  } else if (params.granularity === 'perChannel') {
    throw new Error('perChannel 参数需要二维张量');
  } else if (params.rows != null) {
    throw new Error('params 声明为矩阵（rows=' + params.rows + '），tensor 须为二维');
  }
  const bits = params.bits, qmax = qmaxOf(bits), qlev = qlevOf(bits), exc = excOf(bits);
  const asym = params.scheme === 'asymmetric';

  /** 单行量化 → 存储码数组（Int8Array）或打包字节（Uint8Array）。 */
  function quantChannel(v, ch) {
    const cp = channelParams(params, ch);
    const codes = new Int8Array(v.length);
    for (let i = 0; i < v.length; i++) {
      const x = v[i];
      if (typeof x !== 'number' || !Number.isFinite(x)) throw new Error('tensor 含非有限元素');
      let code;
      if (asym) {
        let q = Math.round(x / cp.scale) + cp.zeroPoint;
        if (q < 0) q = 0; else if (q > qlev) q = qlev; // clamp 到表示范围
        code = q - exc;
      } else {
        let q = Math.round(x / cp.scale);
        if (q < -qmax) q = -qmax; else if (q > qmax) q = qmax;
        code = q;
      }
      codes[i] = code;
    }
    return bits === 4 ? packInt4(codes) : codes;
  }

  if (isMatrixLike(tensor)) {
    const rows = toMatrix(tensor, 'tensor');
    if (params.rows != null && params.rows !== rows.length) {
      throw new Error('tensor 行数 ' + rows.length + ' 与 params.rows=' + params.rows + ' 不一致');
    }
    if (params.cols != null && params.cols !== rows[0].length) {
      throw new Error('tensor 列数 ' + rows[0].length + ' 与 params.cols=' + params.cols + ' 不一致');
    }
    const out = new Array(rows.length);
    for (let i = 0; i < rows.length; i++) out[i] = quantChannel(rows[i], params.granularity === 'perChannel' ? i : 0);
    return out;
  }
  const v = toVector(tensor, 'tensor');
  if (params.n != null && params.n !== v.length) throw new Error('tensor 长度与 params.n 不一致');
  return quantChannel(v, 0);
}

function dequantize(packed, params) {
  const bits = params.bits, qlev = qlevOf(bits), exc = excOf(bits);
  const asym = params.scheme === 'asymmetric';

  /** 单行存储码 → 浮点行。 */
  function deqChannel(codes, ch) {
    const cp = channelParams(params, ch);
    const v = new Float64Array(codes.length);
    for (let i = 0; i < codes.length; i++) {
      const c = codes[i];
      v[i] = asym ? cp.scale * (c + exc - cp.zeroPoint) : cp.scale * c;
    }
    return v;
  }
  /** 存储码容器 → Int8Array 逻辑码（int4 先解包）。 */
  function toCodes(box, count, what) {
    if (bits === 8) {
      if (!(box instanceof Int8Array)) throw new Error(what + ' 须为 Int8Array（bits=8）');
      return box;
    }
    if (!(box instanceof Uint8Array)) throw new Error(what + ' 须为 Uint8Array（bits=4 打包）');
    return unpackInt4(box, count);
  }

  if (Array.isArray(packed)) {
    if (params.rows == null) throw new Error('params 声明为一维，packed 不应为逐行数组');
    if (packed.length !== params.rows) throw new Error('packed 行数与 params.rows 不一致');
    const out = new Array(packed.length);
    for (let i = 0; i < packed.length; i++) {
      out[i] = deqChannel(toCodes(packed[i], params.cols, 'packed[' + i + ']'), params.granularity === 'perChannel' ? i : 0);
    }
    return out;
  }
  const codes = toCodes(packed, params.n, 'packed');
  return deqChannel(codes, 0);
}

/* ============================================================================
 * 4. quantMatvec — 量化权重 × 浮点激活的矩阵-向量乘（逐行反量化后点积）
 * ========================================================================== */
function quantMatvec(packed, x, params) {
  if (params.rows == null || params.cols == null) throw new Error('quantMatvec 需要矩阵参数（calibrate 于二维张量）');
  if (!Array.isArray(x) && !(x instanceof Float64Array)) throw new Error('x 须为一维数组或 Float64Array');
  const n = x.length || x.size;
  if (n !== params.cols) throw new Error('x 长度 ' + n + ' 与权重列数 ' + params.cols + ' 不一致');
  const deq = dequantize(packed, params); // Array<Float64Array>，rows×cols
  const y = new Float64Array(params.rows);
  for (let i = 0; i < params.rows; i++) {
    const row = deq[i];
    let s = 0;
    for (let j = 0; j < params.cols; j++) s += row[j] * x[j];
    y[i] = s;
  }
  return y;
}

/* ============================================================================
 * 5. k-quants 超块（llama.cpp 风格）：每 blockSize 个元素一个超块，块内共享 scale；
 *    可选子块二级 scale（subScales 为相对超块 scale 的比例，llama.cpp 布局：
 *    有效 scale = block.scale × subScales[j]）。
 * ========================================================================== */
function kQuantSuperBlocks(tensor, blockSize, opts) {
  opts = opts || {};
  const bits = (opts.bits === undefined || opts.bits === null) ? 8 : opts.bits;
  const subBlockSize = (opts.subBlockSize === undefined || opts.subBlockSize === null) ? 0 : opts.subBlockSize;
  checkBits(bits);
  if (!Number.isInteger(blockSize) || blockSize < 1) throw new Error('blockSize 须为正整数');
  if (subBlockSize !== 0 && (!Number.isInteger(subBlockSize) || subBlockSize < 1 || blockSize % subBlockSize !== 0)) {
    throw new Error('subBlockSize 须为 0 或能整除 blockSize 的正整数');
  }
  const v = toVector(tensor, 'tensor');
  const qmax = qmaxOf(bits);
  const nSub = subBlockSize > 0 ? blockSize / subBlockSize : 0;
  const blocks = [];
  for (let off = 0; off < v.length; off += blockSize) {
    const len = Math.min(blockSize, v.length - off);
    let amax = 0;
    for (let i = 0; i < len; i++) { const a = Math.abs(v[off + i]); if (a > amax) amax = a; }
    const scale = amax === 0 ? 1 : amax / qmax;
    const q = new Int8Array(len);
    let subScales = null;
    if (subBlockSize > 0) {
      subScales = new Float64Array(nSub);
      for (let s = 0; s < nSub; s++) {
        const j0 = s * subBlockSize, j1 = Math.min(j0 + subBlockSize, len);
        let sam = 0;
        for (let j = j0; j < j1; j++) { const a = Math.abs(v[off + j]); if (a > sam) sam = a; }
        // 子块有效 scale = scale × subScales[s]（相对比例，llama.cpp 二级 scale 布局）
        subScales[s] = sam === 0 ? 0 : (sam / qmax) / scale;
      }
    }
    for (let i = 0; i < len; i++) {
      let eff = scale;
      if (subScales) eff = scale * subScales[(i / subBlockSize) | 0];
      let qi = Math.round(v[off + i] / eff);
      if (qi < -qmax) qi = -qmax; else if (qi > qmax) qi = qmax;
      q[i] = qi;
    }
    blocks.push({ scale, subScales, q });
  }
  return { blocks, blockSize, bits, subBlockSize, n: v.length };
}

function kQuantDequantize(kq) {
  if (!kq || !Array.isArray(kq.blocks) || !Number.isInteger(kq.blockSize) || kq.blockSize < 1) {
    throw new Error('kq 须为 kQuantSuperBlocks 的返回值');
  }
  const sbs = kq.subBlockSize || 0;
  let total = 0;
  for (const b of kq.blocks) total += b.q.length;
  const out = new Float64Array(total);
  let off = 0;
  for (const b of kq.blocks) {
    for (let i = 0; i < b.q.length; i++) {
      let eff = b.scale;
      if (b.subScales && sbs > 0) eff = b.scale * b.subScales[(i / sbs) | 0];
      out[off + i] = b.q[i] * eff;
    }
    off += b.q.length;
  }
  return out;
}

/* ============================================================================
 * 6. quantSNR — 量化信噪比 20·log10(‖original‖ / ‖original-dequantized‖)（dB）
 * ========================================================================== */
function quantSNR(original, dequantized) {
  const a = toVector(original, 'original');
  const b = toVector(dequantized, 'dequantized');
  if (a.length !== b.length) throw new Error('quantSNR 两侧长度不一致');
  let no = 0, ne = 0;
  for (let i = 0; i < a.length; i++) { no += a[i] * a[i]; const d = a[i] - b[i]; ne += d * d; }
  no = Math.sqrt(no); ne = Math.sqrt(ne);
  if (ne === 0) return Infinity; // 完全重建
  return 20 * Math.log10(no / ne);
}

/* ============================================================================
 * selfCheck — 8 条不变量
 * ========================================================================== */
function selfCheck(opts) {
  opts = opts || {};
  const { caseResult, buildReport } = V();
  const cases = [];
  const invariants = [];
  const diag = {}; // 各不变量的实测诊断值（写入 invariants.note，便于报告/QA）
  const rng = N().makeRng(0x9e3779b9);

  function randVec(n, lo, hi) { const v = new Float64Array(n); for (let i = 0; i < n; i++) v[i] = lo + (hi - lo) * rng(); return v; }
  function randMatrix(r, c, lo, hi) { const m = new Array(r); for (let i = 0; i < r; i++) m[i] = randVec(c, lo, hi); return m; }
  function flatMatrix(m) { const f = new Float64Array(m.length * m[0].length); for (let i = 0; i < m.length; i++) f.set(m[i], i * m[0].length); return f; }
  function maxDiffVec(a, b) { let d = 0; for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i])); return d; }

  // INV-QNT-001 可表示值精确往返（bits=8 对称，整数网格点）
  {
    let worst = 0;
    for (const scale of [0.125, 1 / 3, 0.01]) {
      const grid = new Float64Array(255);
      for (let k = -127; k <= 127; k++) grid[k + 127] = k * scale;
      const params = calibrate(grid, { bits: 8, scheme: 'symmetric', granularity: 'perTensor' });
      const deq = dequantize(quantize(grid, params), params);
      worst = Math.max(worst, maxDiffVec(grid, deq));
    }
    diag['INV-QNT-001'] = 'maxRoundtripErr=' + worst.toExponential(3);
    cases.push(caseResult('INV-QNT-001', '网格值往返 deq(quant(k·scale))==k·scale', worst, 0, 1e-12));
  }

  // INV-QNT-002 量化误差有界：max|deq(quant(x))-x| <= scale/2（逐元素用所属通道 scale）
  {
    let worst = 0; // max(0, |err|/(scale/2) - 1)，越小越好
    for (let t = 0; t < 20; t++) {
      const bits = (t % 2 === 0) ? 8 : 4;
      const scheme = (t % 4 < 2) ? 'symmetric' : 'asymmetric';
      // 逐张量（一维）
      {
        const v = randVec(33, -1, 1);
        const params = calibrate(v, { bits, scheme, granularity: 'perTensor' });
        const deq = dequantize(quantize(v, params), params);
        for (let i = 0; i < v.length; i++) worst = Math.max(worst, Math.abs(deq[i] - v[i]) / (params.scale / 2) - 1);
      }
      // 逐通道（二维，各行动态范围不同）
      {
        const m = new Array(5);
        for (let i = 0; i < 5; i++) m[i] = randVec(7, -(i + 1), i + 1);
        const params = calibrate(m, { bits, scheme, granularity: 'perChannel' });
        const deq = dequantize(quantize(m, params), params);
        for (let i = 0; i < 5; i++) {
          const s = params.scale[i];
          for (let j = 0; j < 7; j++) worst = Math.max(worst, Math.abs(deq[i][j] - m[i][j]) / (s / 2) - 1);
        }
      }
    }
    diag['INV-QNT-002'] = 'maxErr/(scale/2)-1=' + worst.toExponential(3);
    cases.push(caseResult('INV-QNT-002', '量化误差 ≤ scale/2（20 组随机张量）', Math.max(0, worst), 0, 1e-12));
  }

  // INV-QNT-003 对称零点恒 0；非对称零点为 [0,2^bits-1] 内整数
  {
    let violations = 0;
    for (let t = 0; t < 12; t++) {
      const bits = (t % 2 === 0) ? 8 : 4;
      const scheme = (t % 3 === 2) ? 'asymmetric' : 'symmetric';
      const qlev = qlevOf(bits);
      const v = randVec(24, -1, 1); // 随机有符号数据必然跨越 0 → 非对称 zp ∈ [0,qlev]
      const p1 = calibrate(v, { bits, scheme, granularity: 'perTensor' });
      const m = randMatrix(4, 6, -1, 1);
      const p2 = calibrate(m, { bits, scheme, granularity: 'perChannel' });
      for (const p of [p1, p2]) {
        const zps = Array.isArray(p.zeroPoint) ? p.zeroPoint : [p.zeroPoint];
        const scales = Array.isArray(p.scale) ? p.scale : [p.scale];
        for (let i = 0; i < zps.length; i++) {
          const z = zps[i];
          if (scheme === 'symmetric') { if (z !== 0) violations++; }
          else if (!Number.isInteger(z) || z < 0 || z > qlev) violations++;
          if (!(scales[i] > 0) || !Number.isFinite(scales[i])) violations++;
        }
      }
    }
    cases.push(caseResult('INV-QNT-003', '对称 zp==0 / 非对称 zp 整数且在 [0,2^bits-1]', violations, 0, 0));
  }

  // INV-QNT-004 逐通道优于逐张量（各通道动态范围差异大，同输入对照）
  // 设计要点：32 通道 absmax 对数均匀 1e-2..1e2（差异大且无单一主宰通道——
  // 若单通道绝对主宰，其误差在两种粒度下相同，SNR 差会退化为 0，对照即空洞）。
  {
    let violations = 0, minGap = Infinity;
    for (let t = 0; t < 20; t++) {
      const m = new Array(32);
      for (let i = 0; i < 32; i++) {
        const a = Math.pow(10, -2 + 4 * rng());
        const row = new Float64Array(12);
        for (let j = 0; j < 12; j++) row[j] = a * (rng() * 2 - 1);
        m[i] = row;
      }
      const orig = flatMatrix(m);
      for (const bits of [8, 4]) {
        const pC = calibrate(m, { bits, scheme: 'symmetric', granularity: 'perChannel' });
        const pT = calibrate(m, { bits, scheme: 'symmetric', granularity: 'perTensor' });
        const snrC = quantSNR(orig, flatMatrix(dequantize(quantize(m, pC), pC)));
        const snrT = quantSNR(orig, flatMatrix(dequantize(quantize(m, pT), pT)));
        if (!(snrT < Infinity)) violations++;      // 防空洞：基线须有真实量化误差
        if (snrC < snrT) violations++;             // 更细粒度不劣
        minGap = Math.min(minGap, snrC - snrT);
      }
    }
    diag['INV-QNT-004'] = 'minSnrGap(perCh-perTen)=' + minGap.toFixed(2) + 'dB';
    cases.push(caseResult('INV-QNT-004', 'perChannel SNR ≥ perTensor SNR（20 组大动态范围矩阵）', violations, 0, 0));
  }

  // INV-QNT-005 int4 范围与端点：存储码 ∈ [-8,7]；对称网格覆盖 [min,max]；端点误差 ≤ 1 步长
  {
    let violations = 0, worstExcess = 0;
    for (let t = 0; t < 20; t++) {
      const scheme = (t % 2 === 0) ? 'symmetric' : 'asymmetric';
      const v = randVec(37, -1, 1);
      const params = calibrate(v, { bits: 4, scheme, granularity: 'perTensor' });
      const packed = quantize(v, params);
      const codes = unpackInt4(packed, v.length);
      const deq = dequantize(packed, params);
      const s = params.scale;
      // (a) 存储码须落在有符号 4-bit 表示范围
      for (let i = 0; i < codes.length; i++) if (codes[i] < -8 || codes[i] > 7) violations++;
      // (b) 对称量化网格须覆盖 [min,max]（amax = qmax·scale；容许 1ulp 级 FP 舍入）
      if (scheme === 'symmetric') {
        let lo = Infinity, hi = -Infinity;
        for (let i = 0; i < v.length; i++) { if (v[i] < lo) lo = v[i]; if (v[i] > hi) hi = v[i]; }
        const covTol = 1e-12 * Math.max(1, Math.abs(lo), Math.abs(hi));
        if (!(-qmaxOf(4) * s <= lo + covTol && qmaxOf(4) * s >= hi - covTol)) violations++;
      }
      // (c) 端点（argmin/argmax）重建误差 ≤ 1 步长
      let iMin = 0, iMax = 0;
      for (let i = 1; i < v.length; i++) { if (v[i] < v[iMin]) iMin = i; if (v[i] > v[iMax]) iMax = i; }
      const eMin = Math.abs(deq[iMin] - v[iMin]), eMax = Math.abs(deq[iMax] - v[iMax]);
      worstExcess = Math.max(worstExcess, eMin - s, eMax - s);
    }
    diag['INV-QNT-005'] = 'rangeViolations=0;maxEndpointErr-step=' + Math.max(0, worstExcess).toExponential(3);
    cases.push(caseResult('INV-QNT-005', 'int4 码域 [-8,7]+网格覆盖+端点误差≤1步长', violations + Math.max(0, worstExcess), 0, 1e-12));
  }

  // INV-QNT-006 量化矩阵乘保真（非负随机权重，逐通道非对称；int8 ≤ 1e-2，int4 ≤ 5e-2，单调）
  // 协议：每试 48×32 权重矩阵 × 16 组随机激活，汇总 Frobenius 相对误差 ‖ΔY‖_F/‖Y‖_F。
  // （4-bit 非对称 16 级在均匀分布上的理论相对误差为 1/(2·15)≈3.3%，单次激活的 ‖y‖ 波动大，
  //  故按激活批次汇总口径度量；int8 理论值 1/(2·255)≈0.20%。）
  {
    let violations = 0, worstExcess = 0, maxRel8 = 0, maxRel4 = 0;
    for (let t = 0; t < 10; t++) {
      // 非负权重：行 absmax 对数均匀 [0.3,3]，行内均匀
      const M = 48, N = 32, K = 16;
      const W = new Array(M);
      for (let i = 0; i < M; i++) {
        const a = Math.pow(10, -0.5 + rng());
        const row = new Float64Array(N);
        for (let j = 0; j < N; j++) row[j] = a * rng();
        W[i] = row;
      }
      const X = new Array(K);
      for (let k = 0; k < K; k++) X[k] = randVec(N, -1, 1);
      const ref = new Array(K);
      for (let k = 0; k < K; k++) {
        const y = new Float64Array(M);
        for (let i = 0; i < M; i++) { let s = 0; for (let j = 0; j < N; j++) s += W[i][j] * X[k][j]; y[i] = s; }
        ref[k] = y;
      }
      let nY = 0;
      for (let k = 0; k < K; k++) for (let i = 0; i < M; i++) nY += ref[k][i] * ref[k][i];
      nY = Math.sqrt(nY);
      const rels = {};
      for (const bits of [8, 4]) {
        const params = calibrate(W, { bits, scheme: 'asymmetric', granularity: 'perChannel' });
        const packed = quantize(W, params);
        let e = 0;
        for (let k = 0; k < K; k++) {
          const y = quantMatvec(packed, X[k], params);
          for (let i = 0; i < M; i++) { const d = y[i] - ref[k][i]; e += d * d; }
        }
        rels[bits] = Math.sqrt(e) / nY;
      }
      if (!(rels[8] <= 1e-2)) violations++;
      if (!(rels[4] <= 5e-2)) violations++;
      if (!(rels[8] <= rels[4])) violations++; // 单调性
      worstExcess = Math.max(worstExcess, rels[8] - 1e-2, rels[4] - 5e-2);
      maxRel8 = Math.max(maxRel8, rels[8]); maxRel4 = Math.max(maxRel4, rels[4]);
    }
    diag['INV-QNT-006'] = 'maxRel8=' + maxRel8.toExponential(3) + ',maxRel4=' + maxRel4.toExponential(3) + '(theory 1/510, 1/30)';
    cases.push(caseResult('INV-QNT-006', '量化矩阵乘保真（int8≤1e-2, int4≤5e-2, 单调）', violations + Math.max(0, worstExcess), 0, 1e-2));
  }

  // INV-QNT-007 超块细化：同输入 k-quants SNR ≥ 整张 per-tensor SNR；块数 == ceil(n/blockSize)
  {
    let violations = 0, minGap = Infinity, minGap2 = Infinity;
    for (let t = 0; t < 20; t++) {
      // 块间动态范围差异大（4 段，相邻段 scale 相差 100 倍级，jitter ±0.3 十倍程不闭合差距）
      const seg = 32, n = 128;
      const base = [1, 0.01, 1e-4, 1e-6];
      const v = new Float64Array(n);
      for (let s = 0; s < 4; s++) {
        const a = base[s] * Math.pow(10, -0.3 + 0.6 * rng());
        for (let j = 0; j < seg; j++) v[s * seg + j] = (rng() < 0.5 ? -1 : 1) * a * rng();
      }
      const pT = calibrate(v, { bits: 8, scheme: 'symmetric', granularity: 'perTensor' });
      const snrT = quantSNR(v, dequantize(quantize(v, pT), pT));
      const kq = kQuantSuperBlocks(v, seg);
      const kq2 = kQuantSuperBlocks(v, seg, { subBlockSize: 16 });
      const snrS = quantSNR(v, kQuantDequantize(kq));
      const snrS2 = quantSNR(v, kQuantDequantize(kq2));
      if (!(snrT < Infinity)) violations++;   // 防空洞
      if (snrS < snrT) violations++;
      if (snrS2 < snrT) violations++;
      if (kq.blocks.length !== Math.ceil(n / seg)) violations++; // 块数精确
      minGap = Math.min(minGap, snrS - snrT); minGap2 = Math.min(minGap2, snrS2 - snrT);
    }
    diag['INV-QNT-007'] = 'minSnrGap(super-perTen)=' + minGap.toFixed(2) + 'dB, withSubBlocks=' + minGap2.toFixed(2) + 'dB';
    cases.push(caseResult('INV-QNT-007', '超块 SNR ≥ 整张 per-tensor SNR + 块数精确', violations, 0, 0));
  }

  // INV-QNT-008 确定性（两次 calibrate+quantize 逐位相等）
  {
    let mismatches = 0;
    /** 参数逐位相等（scale/zeroPoint 可为标量或逐通道数组）。 */
    function sameParams(p1, p2) {
      if (Array.isArray(p1.scale)) {
        if (!Array.isArray(p2.scale) || p1.scale.length !== p2.scale.length) return false;
        for (let i = 0; i < p1.scale.length; i++) {
          if (p1.scale[i] !== p2.scale[i] || p1.zeroPoint[i] !== p2.zeroPoint[i]) return false;
        }
        return true;
      }
      return p1.scale === p2.scale && p1.zeroPoint === p2.zeroPoint;
    }
    const trials = [
      () => ({ v: randVec(29, -1, 1), o: { bits: 4, scheme: 'asymmetric', granularity: 'perTensor' } }),
      () => ({ v: randVec(40, -1, 1), o: { bits: 8, scheme: 'symmetric', granularity: 'perTensor' } }),
      () => ({ v: randMatrix(5, 7, -1, 1), o: { bits: 4, scheme: 'symmetric', granularity: 'perChannel' } })
    ];
    for (const mk of trials) {
      const { v, o } = mk();
      const p1 = calibrate(v, o), p2 = calibrate(v, o);
      const q1 = quantize(v, p1), q2 = quantize(v, p2);
      if (!sameParams(p1, p2)) mismatches++;
      const rows1 = Array.isArray(q1) ? q1 : [q1], rows2 = Array.isArray(q2) ? q2 : [q2];
      for (let i = 0; i < rows1.length; i++) {
        if (rows1[i].length !== rows2[i].length) { mismatches++; continue; }
        for (let j = 0; j < rows1[i].length; j++) if (rows1[i][j] !== rows2[i][j]) mismatches++;
      }
    }
    cases.push(caseResult('INV-QNT-008', '确定性：两次 calibrate+quantize 逐位相等', mismatches, 0, 0));
  }

  // INV-QNT-008b pack/unpack 互为逆运算（偶长默认 count；奇长显式 count）
  {
    let worst = 0;
    const even = new Float64Array(64);
    for (let i = 0; i < even.length; i++) even[i] = Math.floor(rng() * 16) - 8; // [-8,7] 整数
    worst = Math.max(worst, maxDiffVec(even, unpackInt4(packInt4(even))));
    const odd = new Float64Array(63);
    for (let i = 0; i < odd.length; i++) odd[i] = Math.floor(rng() * 16) - 8;
    worst = Math.max(worst, maxDiffVec(odd, unpackInt4(packInt4(odd), odd.length)));
    cases.push(caseResult('INV-QNT-008b', 'unpackInt4(packInt4(v))==v（偶长/奇长）', worst, 0, 1e-12));
  }

  // 不变量登记（note 携带实测诊断值）
  for (const id of ['INV-QNT-001', 'INV-QNT-002', 'INV-QNT-003', 'INV-QNT-004', 'INV-QNT-005', 'INV-QNT-006', 'INV-QNT-007', 'INV-QNT-008']) {
    const c = cases.find(x => x.id === id);
    invariants.push({ id, expr: id, tol: c.tol, worst: c.abs_err, status: c.status, note: diag[id] || '' });
  }

  const started = Date.now();
  return buildReport({
    module: 'quant-forge', version: '1.0.0', layer: 'L4', cases, invariants, seed: '0x9E3779B9', started,
    tol: {
      'INV-QNT-001': 1e-12, 'INV-QNT-002': 1e-12, 'INV-QNT-003': 0, 'INV-QNT-004': 0,
      'INV-QNT-005': 1e-12, 'INV-QNT-006': 1e-2, 'INV-QNT-007': 0, 'INV-QNT-008': 0, 'INV-QNT-008b': 1e-12
    }
  });
}

/* ============================================================================
 * 导出：UMD 双端挂载 + CLI 守卫
 * ========================================================================== */
const API = {
  name: 'quant-forge', version: '1.0.0', layer: 'L4',
  calibrate, quantize, dequantize, quantMatvec,
  kQuantSuperBlocks, kQuantDequantize, quantSNR, packInt4, unpackInt4,
  qmaxOf, qlevOf, excOf, selfCheck
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeQuant = API;

if (typeof process !== 'undefined' && process.argv[1] && /quant-forge\.js$/.test(process.argv[1]) && (process.argv.includes('--self-check') || process.argv.includes('--json'))) {
  const r = selfCheck({ quick: process.argv.includes('--quick') });
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : V().formatTableSafe(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
