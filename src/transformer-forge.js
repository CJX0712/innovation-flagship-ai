/**
 * @forge-module transformer-forge
 * @layer L4
 * @version 1.0.0
 * @brief     Transformer 模型组件与前向传播内核：token 嵌入查找（拷贝语义）、正弦位置编码（Vaswani 2017 原式）、
 *            预归一化(pre-LN)残差块、SwiGLU 前馈、编码器堆叠与组合前向。纯函数式、零 npm 依赖、双端同构。
 * @reuse     Vaswani et al. 2017 (Attention Is All You Need)|https://arxiv.org/abs/1706.03762|arXiv|A|Transformer 架构/正弦位置编码/前馈/残差原式
 * @reuse     GPT-2 (Radford et al. 2019)|https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf|arXiv|B|预归一化(pre-LN)残差块结构
 * @reuse     llama.cpp|https://github.com/ggerganov/llama.cpp|MIT|B|RMSNorm + SwiGLU 现代 LLM 块约定
 * @beyond    纯 JS 单文件零构建；惰性双端依赖（Node require / 浏览器 globalThis）装配 core-verify / core-numeric /
 *            loss-act-norm / attention 四大内核——注意力、softmax、归一化、Swish 全部复用既有公式而非重写；
 *            嵌入拷贝语义、越界抛错、残差恒等路径、因果性、零堆叠恒等等 8 条硬不变量可独立交叉验证。
 * @invariant INV-TRA-001|嵌入查找逐行精确：tokenEmbedding(E,ids) 与 E[ids[i]] 逐元素相等且为拷贝（越界抛错）|0|查表定义
 * @invariant INV-TRA-002|正弦位置编码闭式：PE[pos][2i]=sin(pos/10000^(2i/d))、PE[pos][2i+1]=cos(...)，随机 pos/d 采样最大偏差 ≤ 1e-12|1e-12|Vaswani 2017 原式
 * @invariant INV-TRA-003|预归一化残差恒等路径：attnFn/ffnFn 恒零时 preNormBlock(x)==x 逐元素 ≤ 1e-12|1e-12|残差连接保恒等
 * @invariant INV-TRA-004|前向输出形状：transformerForward.hidden 形状 == (seq,d)，形状不合法抛错（精确断言）|0|形状契约
 * @invariant INV-TRA-005|因果性：因果 attnFn 下改动未来 token 嵌入，已生成位置 i 的输出不变 ≤ 1e-12|1e-12|因果掩码结构
 * @invariant INV-TRA-006|SwiGLU 前馈零输入→零输出（无偏置）≤ 1e-12，且形状保持 (seq,d)→(seq,d)|1e-12|swish(0)=0 门控归零
 * @invariant INV-TRA-007|0 层堆叠：transformerEncoder(x,[],finalNorm) == finalNorm(x) 逐元素 ≤ 1e-12|1e-12|空堆叠恒等
 * @invariant INV-TRA-008|确定性：同 opts 两次 transformerForward 输出逐位相等|0|纯函数确定性
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

let _L = null;
function L() {
  if (_L) return _L;
  if (typeof require !== 'undefined') { try { _L = require('./loss-act-norm-forge.js'); } catch (e) {} }
  if (!_L && typeof globalThis !== 'undefined') _L = globalThis.ForgeLossActNorm || null;
  if (!_L) throw new Error('loss-act-norm-forge 依赖缺失');
  return _L;
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
 * 矩阵约定（与 attention-forge 一致）：二维矩阵 = 数组 of Float64Array（行主序），
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

/** 逐元素相加 A+B（残差连接），形状不一致抛错；返回新矩阵（不修改入参）。 */
function addMatrices(a, b) {
  checkMatrix(a, '残差左操作数');
  checkMatrix(b, '残差右操作数');
  if (a.length !== b.length || a[0].length !== b[0].length) throw new Error('残差相加形状不一致');
  const out = new Array(a.length);
  for (let i = 0; i < a.length; i++) {
    const row = new Float64Array(a[i].length);
    for (let k = 0; k < row.length; k++) row[k] = a[i][k] + b[i][k];
    out[i] = row;
  }
  return out;
}

/* ---------- 1. token 嵌入查找 ---------- */

/**
 * 查表嵌入：out[i] = embedTable[tokenIds[i]] 的拷贝（Float64Array）。
 * 拷贝语义：改写输出不得影响嵌入表；tokenIds 越界或非整数抛错。
 * @param {Array<Float64Array|number[]>} embedTable 词表 × 维度
 * @param {Array<number>|Float64Array} tokenIds  token 索引序列
 * @returns {Array<Float64Array>} seq × d 嵌入矩阵
 */
function tokenEmbedding(embedTable, tokenIds) {
  const shape = checkMatrix(embedTable, 'embedTable');
  if (!tokenIds || typeof tokenIds.length !== 'number' || tokenIds.length === 0) {
    throw new Error('tokenIds 必须为非空索引序列');
  }
  const out = new Array(tokenIds.length);
  for (let i = 0; i < tokenIds.length; i++) {
    const t = tokenIds[i];
    if (!Number.isInteger(t) || t < 0 || t >= shape.rows) {
      throw new Error('tokenIds[' + i + ']=' + t + ' 越界（vocab=' + shape.rows + '）');
    }
    out[i] = Float64Array.from(embedTable[t]); // 深拷贝，隔离嵌入表
  }
  return out;
}

/* ---------- 2. 正弦位置编码（Vaswani et al. 2017 原式） ---------- */

/**
 * PE(pos,2i)   = sin(pos / 10000^(2i/d))
 * PE(pos,2i+1) = cos(pos / 10000^(2i/d))
 * @param {number} maxLen 最大序列长度（正整数）
 * @param {number} d      模型维度（偶数，成对出现）
 * @returns {Array<Float64Array>} maxLen × d 位置编码表
 */
function sinusoidalPositionalEncoding(maxLen, d) {
  if (!Number.isInteger(maxLen) || maxLen < 1) throw new Error('maxLen 必须为正整数');
  if (!Number.isInteger(d) || d < 2 || d % 2 !== 0) throw new Error('d 必须为不小于 2 的偶数（位置编码成对出现）');
  const PE = new Array(maxLen);
  for (let pos = 0; pos < maxLen; pos++) {
    const row = new Float64Array(d);
    for (let i = 0; i < d / 2; i++) {
      const denom = Math.pow(10000, (2 * i) / d);
      row[2 * i] = Math.sin(pos / denom);
      row[2 * i + 1] = Math.cos(pos / denom);
    }
    PE[pos] = row;
  }
  return PE;
}

/* ---------- 3. SwiGLU 前馈 ---------- */

/**
 * SwiGLU 前馈（llama 风格门控）：h = swish(x·Wgate) ⊙ (x·Wup)；out = h·Wdown。
 * 复用 loss-act-norm-forge 的 swish；无偏置时零输入→零输出（swish(0)=0）。
 * @param {Array<Float64Array>} x     seq × d 输入
 * @param {Array<Float64Array>} Wgate hidden × d
 * @param {Array<Float64Array>} Wup   hidden × d
 * @param {Array<Float64Array>} Wdown d × hidden
 * @param {{gate?:Array<number>, up?:Array<number>, down?:Array<number>}} [b] 可选偏置
 * @returns {Array<Float64Array>} seq × d 输出
 */
function ffnGlu(x, Wgate, Wup, Wdown, b) {
  const xs = checkMatrix(x, 'x');
  const gs = checkMatrix(Wgate, 'Wgate');
  const us = checkMatrix(Wup, 'Wup');
  const ds = checkMatrix(Wdown, 'Wdown');
  if (gs.cols !== xs.cols) throw new Error('Wgate 形状须为 hidden×d（in=' + xs.cols + '）');
  if (us.rows !== gs.rows || us.cols !== xs.cols) throw new Error('Wup 形状须与 Wgate 一致（hidden×d）');
  if (ds.rows !== xs.cols || ds.cols !== gs.rows) throw new Error('Wdown 形状须为 d×hidden');
  const bias = b || {};
  const bg = bias.gate || null, bu = bias.up || null, bd = bias.down || null;
  if (bg && bg.length !== gs.rows) throw new Error('bias.gate 长度须为 hidden=' + gs.rows);
  if (bu && bu.length !== us.rows) throw new Error('bias.up 长度须为 hidden=' + us.rows);
  if (bd && bd.length !== ds.rows) throw new Error('bias.down 长度须为 d=' + ds.rows);
  const swish = L().swish;
  const seq = xs.rows, d = xs.cols, hidden = gs.rows;
  const out = new Array(seq);
  for (let s = 0; s < seq; s++) {
    const h = new Float64Array(hidden);
    for (let i = 0; i < hidden; i++) {
      let gw = bg ? bg[i] : 0;
      let uv = bu ? bu[i] : 0;
      for (let j = 0; j < d; j++) { const xv = x[s][j]; gw += xv * Wgate[i][j]; uv += xv * Wup[i][j]; }
      h[i] = swish(gw) * uv; // 门控：swish(gate) ⊙ value
    }
    const row = new Float64Array(d);
    for (let k = 0; k < d; k++) {
      let acc = bd ? bd[k] : 0;
      for (let i = 0; i < hidden; i++) acc += h[i] * Wdown[k][i];
      row[k] = acc;
    }
    out[s] = row;
  }
  return out;
}

/* ---------- 4. 预归一化残差块（GPT-2 / llama 风格 pre-LN） ---------- */

/**
 * 矩阵级 RMSNorm 助手（逐行，复用 loss-act-norm-forge.rmsNorm）：
 * 同时充当 preNormBlock / transformerEncoder 的默认 normFn（矩阵 → 矩阵）。
 * @param {Array<Float64Array>} x seq × d
 * @param {Array<number>} [g] 可选增益向量（逐行共用）
 * @returns {Array<Float64Array>} seq × d
 */
function rmsNormRows(x, g) {
  checkMatrix(x, 'x');
  if (g !== undefined && g !== null && g.length !== x[0].length) throw new Error('增益长度须为 d');
  const rmsNorm = L().rmsNorm;
  const out = new Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = rmsNorm(x[i], g || null);
  return out;
}

/**
 * 预归一化残差块：x ← x + attnFn(normFn(x))；x ← x + ffnFn(normFn(x))。
 * normFn 为矩阵级函数（矩阵 → 矩阵），缺省用逐行 RMSNorm。
 * @param {Array<Float64Array>} x      seq × d
 * @param {(m:Array<Float64Array>)=>Array<Float64Array>} attnFn 注意力子层
 * @param {(m:Array<Float64Array>)=>Array<Float64Array>} ffnFn  前馈子层
 * @param {(m:Array<Float64Array>)=>Array<Float64Array>} [normFn] 归一化子层
 * @returns {Array<Float64Array>} seq × d
 */
function preNormBlock(x, attnFn, ffnFn, normFn) {
  checkMatrix(x, 'x');
  if (typeof attnFn !== 'function') throw new Error('attnFn 必须为函数');
  if (typeof ffnFn !== 'function') throw new Error('ffnFn 必须为函数');
  if (normFn !== undefined && normFn !== null && typeof normFn !== 'function') throw new Error('normFn 必须为函数');
  const norm = normFn || rmsNormRows;
  const h1 = addMatrices(x, attnFn(norm(x)));          // 残差 1：注意力
  return addMatrices(h1, ffnFn(norm(h1)));             // 残差 2：前馈
}

/* ---------- 5. 编码器堆叠 ---------- */

/**
 * 顺序堆叠 blocks（每块 {attnFn, ffnFn, normFn}），最后施加 finalNormFn。
 * 空 blocks 时输出 == finalNormFn(x)（零层堆叠恒等）。
 * @param {Array<Float64Array>} x seq × d
 * @param {Array<{attnFn:Function, ffnFn:Function, normFn?:Function}>} blocks
 * @param {(m:Array<Float64Array>)=>Array<Float64Array>} finalNormFn
 * @returns {Array<Float64Array>} seq × d
 */
function transformerEncoder(x, blocks, finalNormFn) {
  checkMatrix(x, 'x');
  if (!Array.isArray(blocks)) throw new Error('blocks 必须为数组');
  if (typeof finalNormFn !== 'function') throw new Error('finalNormFn 必须为函数');
  let h = x;
  for (let i = 0; i < blocks.length; i++) {
    const blk = blocks[i] || {};
    if (typeof blk.attnFn !== 'function' || typeof blk.ffnFn !== 'function') {
      throw new Error('blocks[' + i + '] 须含 attnFn / ffnFn 函数');
    }
    h = preNormBlock(h, blk.attnFn, blk.ffnFn, blk.normFn);
  }
  return finalNormFn(h);
}

/* ---------- 6. 组合前向 ---------- */

/**
 * 组合前向：embedding →（加正弦 PE）→ encoder 堆叠。
 * @param {object} opts
 * @param {Array<Float64Array>} opts.embedTable   词表 × d
 * @param {Array<number>} opts.tokenIds           token 序列
 * @param {Array<object>} opts.blocks             [{attnFn, ffnFn, normFn}]
 * @param {Function} opts.finalNormFn             矩阵级归一化
 * @param {'sinusoidal'|null} [opts.pe]           位置编码类型，缺省 null
 * @param {number} [opts.d]                       可选维度校验（须与嵌入一致）
 * @returns {{hidden:Array<Float64Array>, tokens:Array<Float64Array>}}
 *          hidden = 编码器输出（seq×d）；tokens = 编码器输入（嵌入[+PE]，seq×d）
 */
function transformerForward(opts) {
  opts = opts || {};
  const pe = (opts.pe === undefined || opts.pe === null) ? null : opts.pe;
  if (pe !== null && pe !== 'sinusoidal') throw new Error("pe 仅支持 'sinusoidal' 或 null");
  const emb = tokenEmbedding(opts.embedTable, opts.tokenIds); // 内含词表/越界校验
  const seq = emb.length;
  const d = emb[0].length;
  if (opts.d !== undefined && opts.d !== null && opts.d !== d) {
    throw new Error('opts.d=' + opts.d + ' 与嵌入维度 d=' + d + ' 不一致');
  }
  let tokens = emb;
  if (pe === 'sinusoidal') tokens = addMatrices(emb, sinusoidalPositionalEncoding(seq, d));
  const hidden = transformerEncoder(tokens, opts.blocks, opts.finalNormFn);
  if (!Array.isArray(hidden) || hidden.length !== seq || !hidden[0] || hidden[0].length !== d) {
    throw new Error('finalNormFn 输出形状非法（期望 seq×d）');
  }
  return { hidden, tokens };
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

function selfCheck(opts) {
  opts = opts || {};
  const { caseResult, buildReport } = V();
  const cases = [];
  const invariants = [];
  const rng = N().makeRng(0x9e3779b9);

  // 因果注意力子层（复用 attention-forge SDPA + 因果掩码）
  const causalAttnFn = (m) => A().sdpa(m, m, m, { causal: true }).out;

  // INV-TRA-001 嵌入查找逐行精确（含拷贝语义与越界抛错）
  {
    const d = 5, vocab = 7;
    const E = randomMatrix(rng, vocab, d);
    const ids = [3, 0, 6, 1, 3];
    const out = tokenEmbedding(E, ids);
    let dev = 0;
    for (let i = 0; i < ids.length; i++) {
      for (let k = 0; k < d; k++) dev = Math.max(dev, Math.abs(out[i][k] - E[ids[i]][k]));
    }
    // 拷贝语义：改写输出不得影响嵌入表
    const snapshot = E[3][0];
    out[0][0] = snapshot + 1.234;
    const copyOk = (E[3][0] === snapshot);
    // 越界抛错（正越界与负越界）
    let threwPos = false, threwNeg = false;
    try { tokenEmbedding(E, [vocab]); } catch (e) { threwPos = true; }
    try { tokenEmbedding(E, [-1]); } catch (e) { threwNeg = true; }
    const metric = Math.max(dev, copyOk ? 0 : 1, threwPos ? 0 : 1, threwNeg ? 0 : 1);
    cases.push(caseResult('INV-TRA-001', '嵌入查找逐行精确+拷贝+越界抛错', metric, 0, 0));
  }

  // INV-TRA-002 正弦位置编码闭式（随机 pos/d 采样 20 组，独立 exp/log 参照路径）
  {
    let mx = 0;
    const dims = [4, 8, 16, 32, 64];
    for (let t = 0; t < 20; t++) {
      const d = dims[t % dims.length];
      const PE = sinusoidalPositionalEncoding(128, d);
      const pos = Math.floor(rng() * 128);
      for (let i = 0; i < d / 2; i++) {
        const denom = Math.exp((2 * i / d) * Math.log(10000)); // 独立于 Math.pow 的参照路径
        mx = Math.max(mx, Math.abs(PE[pos][2 * i] - Math.sin(pos / denom)));
        mx = Math.max(mx, Math.abs(PE[pos][2 * i + 1] - Math.cos(pos / denom)));
      }
    }
    cases.push(caseResult('INV-TRA-002', '正弦PE闭式最大偏差(随机pos/d)', mx, 0, 1e-12));
  }

  // INV-TRA-003 预归一化残差恒等路径（attnFn/ffnFn 恒零 → 输出 == 输入）
  {
    const seq = 4, d = 6;
    const x = randomMatrix(rng, seq, d);
    const zeroFn = () => zerosMatrix(seq, d);
    const out = preNormBlock(x, zeroFn, zeroFn);
    let dev = 0;
    for (let i = 0; i < seq; i++) for (let k = 0; k < d; k++) dev = Math.max(dev, Math.abs(out[i][k] - x[i][k]));
    cases.push(caseResult('INV-TRA-003', '预归一化残差恒等路径', dev, 0, 1e-12));
  }

  // INV-TRA-004 前向输出形状 == (seq,d)（精确断言；形状不合法须抛错）
  {
    const seq = 5, d = 8, vocab = 10;
    const E = randomMatrix(rng, vocab, d);
    const Wgate = randomMatrix(rng, 12, d), Wup = randomMatrix(rng, 12, d), Wdown = randomMatrix(rng, d, 12);
    const ids = [0, 3, 7, 2, 9];
    const opts = {
      embedTable: E, tokenIds: ids, d,
      blocks: [{ attnFn: causalAttnFn, ffnFn: (m) => ffnGlu(m, Wgate, Wup, Wdown) }],
      finalNormFn: rmsNormRows, pe: 'sinusoidal'
    };
    const r = transformerForward(opts);
    const shapeOk = (Array.isArray(r.hidden) && r.hidden.length === seq && r.hidden[0].length === d
      && Array.isArray(r.tokens) && r.tokens.length === seq && r.tokens[0].length === d) ? 0 : 1;
    let threw = false;
    try { transformerForward(Object.assign({}, opts, { d: d + 1 })); } catch (e) { threw = true; }
    cases.push(caseResult('INV-TRA-004', '前向输出形状 hidden==(seq,d)+非法抛错', Math.max(shapeOk, threw ? 0 : 1), 0, 0));
  }

  // INV-TRA-005 因果性：改动未来 token（位置 5）的嵌入，已生成位置 0..4 输出不变
  {
    const seq = 6, d = 8, vocab = 16;
    const E = randomMatrix(rng, vocab, d);
    const Wgate = randomMatrix(rng, 12, d), Wup = randomMatrix(rng, 12, d), Wdown = randomMatrix(rng, d, 12);
    const ids = [1, 4, 9, 2, 7, 11]; // token 11 仅出现在位置 5
    const baseOpts = {
      embedTable: E, tokenIds: ids, d,
      blocks: [{ attnFn: causalAttnFn, ffnFn: (m) => ffnGlu(m, Wgate, Wup, Wdown) }],
      finalNormFn: rmsNormRows, pe: 'sinusoidal'
    };
    const r1 = transformerForward(baseOpts);
    // 仅扰动 embedTable 第 11 行（未来 token 的嵌入）；其余行逐值拷贝保持一致
    const E2 = E.map((row) => Float64Array.from(row));
    for (let k = 0; k < d; k++) E2[11][k] = E2[11][k] * 2 + 1;
    const r2 = transformerForward(Object.assign({}, baseOpts, { embedTable: E2 }));
    let dev = 0;
    for (let i = 0; i < seq - 1; i++) {
      for (let k = 0; k < d; k++) dev = Math.max(dev, Math.abs(r1.hidden[i][k] - r2.hidden[i][k]));
    }
    // 防空洞化：被扰动的未来位置 5 的输出必须确实发生变化
    let row5Delta = 0;
    for (let k = 0; k < d; k++) row5Delta = Math.max(row5Delta, Math.abs(r1.hidden[5][k] - r2.hidden[5][k]));
    const perturbedOk = (row5Delta > 1e-6) ? 0 : 1;
    cases.push(caseResult('INV-TRA-005', '因果性：未来token改动不影响已生成位置', Math.max(dev, perturbedOk), 0, 1e-12));
  }

  // INV-TRA-006 SwiGLU 零输入→零输出（无偏置）+ 形状保持
  {
    const seq = 3, d = 6, hidden = 9;
    const Wgate = randomMatrix(rng, hidden, d), Wup = randomMatrix(rng, hidden, d), Wdown = randomMatrix(rng, d, hidden);
    const out = ffnGlu(zerosMatrix(seq, d), Wgate, Wup, Wdown);
    let dev = 0;
    for (let i = 0; i < seq; i++) for (let k = 0; k < d; k++) dev = Math.max(dev, Math.abs(out[i][k]));
    const shapeOk = (out.length === seq && out[0].length === d) ? 0 : 1;
    cases.push(caseResult('INV-TRA-006', 'SwiGLU 零输入→零输出+形状保持', Math.max(dev, shapeOk), 0, 1e-12));
  }

  // INV-TRA-007 0 层堆叠：transformerEncoder(x,[],finalNorm) == finalNorm(x)
  {
    const seq = 4, d = 5;
    const x = randomMatrix(rng, seq, d);
    const out = transformerEncoder(x, [], rmsNormRows);
    const ref = rmsNormRows(x);
    let dev = 0;
    for (let i = 0; i < seq; i++) for (let k = 0; k < d; k++) dev = Math.max(dev, Math.abs(out[i][k] - ref[i][k]));
    cases.push(caseResult('INV-TRA-007', '0 层堆叠 == finalNorm(x)', dev, 0, 1e-12));
  }

  // INV-TRA-008 确定性：同 opts 两次前向逐位相等
  {
    const seq = 4, d = 8, vocab = 12;
    const E = randomMatrix(rng, vocab, d);
    const Wgate = randomMatrix(rng, 10, d), Wup = randomMatrix(rng, 10, d), Wdown = randomMatrix(rng, d, 10);
    const ids = [2, 5, 8, 1];
    const opts = {
      embedTable: E, tokenIds: ids, d,
      blocks: [{ attnFn: causalAttnFn, ffnFn: (m) => ffnGlu(m, Wgate, Wup, Wdown) }],
      finalNormFn: rmsNormRows, pe: 'sinusoidal'
    };
    const r1 = transformerForward(opts);
    const r2 = transformerForward(opts);
    let dev = 0;
    for (let i = 0; i < seq; i++) for (let k = 0; k < d; k++) dev = Math.max(dev, Math.abs(r1.hidden[i][k] - r2.hidden[i][k]));
    cases.push(caseResult('INV-TRA-008', '确定性：同 opts 两次前向逐位相等', dev, 0, 0));
  }

  // 不变量登记
  for (const id of ['INV-TRA-001', 'INV-TRA-002', 'INV-TRA-003', 'INV-TRA-004', 'INV-TRA-005', 'INV-TRA-006', 'INV-TRA-007', 'INV-TRA-008']) {
    const c = cases.find(x => x.id === id);
    invariants.push({ id, expr: id, tol: c.tol, worst: c.abs_err, status: c.status, note: '' });
  }

  const started = Date.now();
  return buildReport({
    module: 'transformer-forge', version: '1.0.0', layer: 'L4', cases, invariants,
    seed: '0x9E3779B9', started,
    tol: { 'INV-TRA-001': 0, 'INV-TRA-002': 1e-12, 'INV-TRA-003': 1e-12, 'INV-TRA-004': 0, 'INV-TRA-005': 1e-12, 'INV-TRA-006': 1e-12, 'INV-TRA-007': 1e-12, 'INV-TRA-008': 0 }
  });
}

const API = {
  name: 'transformer-forge', version: '1.0.0', layer: 'L4',
  tokenEmbedding, sinusoidalPositionalEncoding, ffnGlu, preNormBlock, transformerEncoder, transformerForward, rmsNormRows, selfCheck
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeTransformer = API;

if (typeof process !== 'undefined' && process.argv[1] && /transformer-forge\.js$/.test(process.argv[1]) && (process.argv.includes('--self-check') || process.argv.includes('--json'))) {
  const r = selfCheck({ quick: process.argv.includes('--quick') });
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : V().formatTableSafe(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
