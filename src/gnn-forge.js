/**
 * @forge-module gnn-forge
 * @layer L7
 * @version 1.0.0
 * @brief     图神经网络内核：对称归一化 GCN 传播 / GraphSAGE 邻居聚合(mean/sum/max) / GAT 边注意力(多头) / 通用 MPNN 消息传递(message/aggregate/update 三阶段可插拔) / 2 层 GCN 节点分类(全批量梯度下降可训练到过拟合) / 可种子复现邻居采样
 * @reuse     Kipf & Welling《Semi-Supervised Classification with Graph Convolutional Networks》(ICLR 2017)|https://arxiv.org/abs/1609.02907|A|对称归一化传播 H'=σ(D̂^-1/2 Â D̂^-1/2 H W)，Â=A+I
 * @reuse     Hamilton et al.《Inductive Representation Learning on Large Graphs》(NeurIPS 2017)|https://arxiv.org/abs/1706.02216|A|GraphSAGE 邻居聚合 + concat 自身特征后线性变换
 * @reuse     Veličković et al.《Graph Attention Networks》(ICLR 2018)|https://arxiv.org/abs/1710.10903|A|GAT 边注意力 α_ij = softmax_j(LeakyReLU(aᵀ[Wh_i‖Wh_j]))，多头拼接
 * @reuse     Gilmer et al.《Neural Message Passing for Quantum Chemistry》(ICML 2017)|https://arxiv.org/abs/1704.01212|A|MPNN message/aggregate/update 三阶段框架
 * @beyond    纯 JS 单文件、零 npm 依赖、双端同构（浏览器 window.ForgeGNN + Node module.exports）；把 GCN/SAGE/GAT/MPNN 统一进一个消息传递视角——GCN 是「度缩放消息 + sum 聚合 + 恒等更新」的 MPNN 特例，SAGE 是「邻居聚合 + concat 线性变换」，GAT 用可学习注意力替换聚合权重——四种算法均按原文献公式实现；8 条不变量全部携带手搓闭式/黄金参照交叉验证，并按反空洞铁律断言「不同设置下必分歧」证伪平凡恒等；私有 sfc32 可种子 PRNG（0x9E3779B9）保证全流水线逐位可复现。
 * @invariant INV-GNN-001|对称归一化 D̂^-1/2 Â D̂^-1/2 与手搓闭式逐元素一致(≤1e-15)且矩阵对称；且异质度图（星形）行和 ≠ 1（max|Σ_j-1|>0.05，证伪行归一混淆）|1e-15|手搓闭式逐元素对照 + 对称性扫描 + 星形图行和证伪
 * @invariant INV-GNN-002|GCN 前向 relu(Ânorm XW+b) 与手搓循环展开黄金参照逐元素一致(≤1e-15，参照不调用本模块函数)；且 W=0、act=null 时输出 ≡ 偏置广播（逐位精确，可证伪）|1e-15|手搓黄金参照交叉验证 + 零权重退化断言
 * @invariant INV-GNN-003|GraphSAGE mean 聚合与手搓邻居均值+concat 线性变换逐元素一致(≤1e-15)；且 mean/sum/max 三种聚合器对同一图输出两两必分歧(>1e-6，非退化)|1e-15|手搓邻居均值对照 + 三聚合器两两分歧断言
 * @invariant INV-GNN-004|GAT 每头 α 与手搓黄金参照逐元素一致(mismatch=0，≤1e-15)且每节点出边注意力权和=1(≤1e-15)；且 α 必偏离均匀(>1e-3，证伪平凡均匀)；且 LeakyReLU 负斜率生效（负输入→0.2x 严格小于 0）|1e-15|手搓注意力黄金参照 + 行和扫描 + 均匀性证伪 + 斜率探针
 * @invariant INV-GNN-005|MPNN(度缩放消息+sum聚合+恒等更新) ≡ GCN(无 W, sym-norm) 逐元素(≤1e-15)；MPNN(mean聚合+平消息) ≡ 手搓行均值(≤1e-15)；且换 sum 聚合必分歧(>1e-9)|1e-15|双框架等价交叉验证 + 聚合器切换分歧断言
 * @invariant INV-GNN-006|2 层 GCN 节点分类在小图上训练 300 步后训练准确率=1.0 且损失严格下降；且同一初始化未训练准确率<1.0（证伪平凡）|0|过拟合训练 + 未训练对照（前向+梯度联合正确性证明）
 * @invariant INV-GNN-007|邻居采样：同 seed 两次采样子图逐位相等(max diff=0)；异 seed 对 20 对种子必分歧(≥18 对)；且每节点采样出度 ≤ min(k,deg) 且为原始邻居子集|0|种子复现 + 多重种子分歧 + 约束扫描
 * @invariant INV-GNN-008|确定性：同 seed 全流水线（采样→GCN→GAT→SAGE→MPNN→节点分类训练）两次运行输出逐位相等(max diff=0)；异 seed 必分歧(>1e-6)|0|全流水线双跑对照
 * @seed      0x9E3779B9
 */
'use strict';

/* ============================================================================
 * 惰性双端依赖：core-verify（caseResult/buildReport/formatTableSafe）。
 * 模块加载期不触碰依赖；浏览器端回退 globalThis.ForgeVerify。
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
 * 项目统一容差（与 core-verify.TOL 同值，本地声明以避免模块间启动期强依赖）
 * ========================================================================== */
const TOL = { EXACT: 0, ULP: 1e-16, TIGHT: 1e-15, ORTHO: 1e-14, RESID: 1e-12, SOLVE: 1e-10, GRAD: 1e-6, MONO: -1e-9, CONV: 1e-6 };

/* ============================================================================
 * 私有可种子 PRNG：sfc32 + splitmix32（禁 Math.random，默认种子 0x9E3779B9）
 * ========================================================================== */
function sfc32(a, b, c, d) {
  a = a | 0; b = b | 0; c = c | 0; d = d | 0;
  return function () {
    a = (a + b) | 0;
    d = (d + 1) | 0;
    const t = ((a >>> 0) + (d >>> 0)) | 0;
    c = (c + ((c << 9) >>> 0)) | 0;
    b = ((b + ((b << 3) | 0)) ^ (b >>> 15)) >>> 0;
    c = ((c + ((c << 16) | 0)) ^ (c >>> 13)) >>> 0;
    b = (b + c) | 0;
    return ((t ^ ((t >>> 16) | 0)) >>> 0) / 4294967296;
  };
}
function splitmix32(seed) {
  let z = seed | 0;
  return function () {
    z = (z + 0x9e3779b9) | 0;
    let t = z;
    t = ((t ^ (t >>> 16)) >>> 0);
    t = (Math.imul(t, 0x21f0aaad) >>> 0);
    t = ((t ^ (t >>> 15)) >>> 0);
    t = (Math.imul(t, 0x735a2d97) >>> 0);
    t = ((t ^ (t >>> 15)) >>> 0);
    return t >>> 0;
  };
}
/** 由单一整数种子产出 [0,1) 均匀流（确定性）。 */
function makeRng(seed) {
  const sm = splitmix32(seed != null ? seed : 0x9E3779B9);
  const a = sm(), b = sm(), c = sm(), d = sm();
  return sfc32(a, b, c, d);
}

/* ============================================================================
 * 输入校验与小线性代数工具（row-major 二维数组）
 * ========================================================================== */
function assertMatrix(M, name, rows, cols) {
  if (!Array.isArray(M) || M.length === 0 || !Array.isArray(M[0])) {
    throw new TypeError(name + ' 必须为非空二维数组');
  }
  if (rows != null && M.length !== rows) throw new TypeError(name + ' 行数=' + M.length + '，应为 ' + rows);
  for (let i = 0; i < M.length; i++) {
    const row = M[i];
    if (!Array.isArray(row) || row.length === 0) throw new TypeError(name + '[' + i + '] 必须为非空数组');
    if (cols != null && row.length !== cols) throw new TypeError(name + '[' + i + '] 列数=' + row.length + '，应为 ' + cols);
    for (let j = 0; j < row.length; j++) {
      if (typeof row[j] !== 'number' || !isFinite(row[j])) throw new TypeError(name + '[' + i + '][' + j + '] 非有限数值');
    }
  }
  return true;
}

function validateAdj(adj) {
  if (!Array.isArray(adj) || adj.length === 0) throw new TypeError('adj 必须为非空二维数组');
  const n = adj.length;
  for (let i = 0; i < n; i++) {
    const row = adj[i];
    if (!Array.isArray(row) || row.length !== n) {
      throw new TypeError('adj 必须为方阵：第 ' + i + ' 行长度=' + (row ? row.length : 'null') + ' ≠ ' + n);
    }
    for (let j = 0; j < n; j++) {
      const v = row[j];
      if (typeof v !== 'number' || !isFinite(v)) throw new TypeError('adj[' + i + '][' + j + '] 非有限数值');
      if (v < 0) throw new TypeError('adj[' + i + '][' + j + '] 必须为非负权重');
    }
  }
  return true;
}

function zeros(n, d) {
  const M = new Array(n);
  for (let i = 0; i < n; i++) M[i] = new Array(d).fill(0);
  return M;
}
function identity(n) {
  const M = zeros(n, n);
  for (let i = 0; i < n; i++) M[i][i] = 1;
  return M;
}
function transpose(A) {
  const m = A.length, n = A[0].length;
  const T = zeros(n, m);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) T[j][i] = A[i][j];
  return T;
}
function matMul(A, B) {
  assertMatrix(A, 'A'); assertMatrix(B, 'B');
  const m = A.length, p = A[0].length, q = B.length, nb = B[0].length;
  if (p !== q) throw new TypeError('matMul: 内维不匹配 (' + p + ' vs ' + q + ')');
  const C = zeros(m, nb);
  for (let i = 0; i < m; i++) {
    for (let l = 0; l < p; l++) {
      const a = A[i][l];
      if (a !== 0) for (let j = 0; j < nb; j++) C[i][j] += a * B[l][j];
    }
  }
  return C;
}
function colSlice(W, from, to) { return W.map(r => r.slice(from, to)); }
function maxAbsDiffMat(A, B) {
  if (A.length !== B.length) throw new TypeError('maxAbsDiffMat: 行数不一致');
  let d = 0;
  for (let i = 0; i < A.length; i++) {
    if (A[i].length !== B[i].length) throw new TypeError('maxAbsDiffMat: 列数不一致');
    for (let j = 0; j < A[i].length; j++) d = Math.max(d, Math.abs(A[i][j] - B[i][j]));
  }
  return d;
}
function maxDiffNested(A, B) {
  let d = 0;
  for (let t = 0; t < A.length; t++) d = Math.max(d, maxAbsDiffMat(A[t], B[t]));
  return d;
}
function randMat(rng, rows, cols, scale) {
  const M = zeros(rows, cols);
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) M[i][j] = (rng() * 2 - 1) * scale;
  return M;
}

/* ============================================================================
 * 激活与 softmax
 * ========================================================================== */
/** LeakyReLU（GAT 默认负斜率 0.2，Veličković et al. 2018）。 */
function leakyRelu(x, slope) {
  const s = (slope != null) ? slope : 0.2;
  return x < 0 ? s * x : x;
}
const ACTS = { relu: x => (x > 0 ? x : 0), tanh: Math.tanh, sigmoid: x => 1 / (1 + Math.exp(-x)) };
function resolveAct(act) {
  if (act == null) return null;
  if (typeof act === 'function') return act;
  if (typeof act === 'string' && ACTS[act]) return ACTS[act];
  throw new TypeError('不支持的激活函数: ' + act);
}
/** 数值稳定 softmax（减最大值）。 */
function softmaxVec(arr) {
  let mx = -Infinity;
  for (const v of arr) if (v > mx) mx = v;
  let Z = 0;
  const ex = arr.map(v => { const e = Math.exp(v - mx); Z += e; return e; });
  return ex.map(e => e / Z);
}

/* ============================================================================
 * O00 图卷积 GCN：对称归一化传播 H' = σ(D̂^-1/2 Â D̂^-1/2 H W)
 * Â = A + I（内部自动加自环），D̂ 为 Â 的度矩阵；非零权重均视为边。
 * ========================================================================== */
function symNormalize(adj) {
  validateAdj(adj);
  const n = adj.length;
  const deg = new Array(n);
  for (let i = 0; i < n; i++) {
    let d = 1;                                  // Â 对角 = adj[i][i] + 1
    for (let j = 0; j < n; j++) d += adj[i][j];
    deg[i] = d;
  }
  const An = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const a = (i === j) ? adj[i][j] + 1 : adj[i][j];
      if (a !== 0) An[i][j] = a / Math.sqrt(deg[i] * deg[j]);
    }
  }
  return An;
}

/** GCN 层：H' = act(Ânorm·X·W + b)。b 为长度 dOut 的偏置向量（可为 null），act 可为 'relu'|'tanh'|'sigmoid'|函数|null。 */
function gcnLayer(X, adj, W, b, act) {
  assertMatrix(X, 'X'); validateAdj(adj); assertMatrix(W, 'W');
  const n = X.length, dIn = X[0].length;
  if (W.length !== dIn) throw new TypeError('gcnLayer: W 行数=' + W.length + '，应等于 X 特征维=' + dIn);
  const dOut = W[0].length;
  if (b != null) {
    if (!Array.isArray(b) || b.length !== dOut) throw new TypeError('gcnLayer: b 长度应为 ' + dOut);
    for (const v of b) if (typeof v !== 'number' || !isFinite(v)) throw new TypeError('gcnLayer: b 含非有限数值');
  }
  const f = resolveAct(act);
  const An = symNormalize(adj);
  const Z = matMul(matMul(An, X), W);
  const out = zeros(n, dOut);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < dOut; k++) {
      const v = Z[i][k] + (b != null ? b[k] : 0);
      out[i][k] = f ? f(v) : v;
    }
  }
  return out;
}

/* ============================================================================
 * O01 GraphSAGE：邻居聚合(mean/sum/max) + 拼接自身特征后线性变换
 * h'_i = [h_i ‖ AGG_{j∈N(i)} h_j]·W，W 形状 (2·dIn)×dOut。
 * ========================================================================== */
function sageAggregate(X, adj, W, aggregator) {
  assertMatrix(X, 'X'); validateAdj(adj); assertMatrix(W, 'W');
  aggregator = aggregator || 'mean';
  if (aggregator !== 'mean' && aggregator !== 'sum' && aggregator !== 'max') {
    throw new TypeError('sageAggregate: 不支持的聚合器 "' + aggregator + '"（可选 mean/sum/max）');
  }
  const n = X.length, dIn = X[0].length;
  if (W.length !== 2 * dIn) throw new TypeError('sageAggregate: W 行数=' + W.length + '，应为 2·dIn=' + (2 * dIn) + '（concat 自身与邻居聚合）');
  const dOut = W[0].length;
  const out = zeros(n, dOut);
  for (let i = 0; i < n; i++) {
    const nbrs = [];
    for (let j = 0; j < n; j++) if (j !== i && adj[i][j] !== 0) nbrs.push(j);
    const agg = new Array(dIn).fill(0);
    if (nbrs.length > 0) {
      if (aggregator === 'mean' || aggregator === 'sum') {
        for (const j of nbrs) for (let k = 0; k < dIn; k++) agg[k] += X[j][k];
        if (aggregator === 'mean') for (let k = 0; k < dIn; k++) agg[k] /= nbrs.length;
      } else { // max
        for (let k = 0; k < dIn; k++) {
          let m = -Infinity;
          for (const j of nbrs) if (X[j][k] > m) m = X[j][k];
          agg[k] = m;
        }
      }
    }
    for (let k = 0; k < dOut; k++) {
      let s = 0;
      for (let q = 0; q < dIn; q++) s += X[i][q] * W[q][k];
      for (let q = 0; q < dIn; q++) s += agg[q] * W[dIn + q][k];
      out[i][k] = s;
    }
  }
  return out;
}

/* ============================================================================
 * O02 GAT：边注意力 α_ij = softmax_j(LeakyReLU(aᵀ[Wh_i‖Wh_j]))，多头可选
 * W 形状 dIn×(nHeads·dHead)；a 为每头 2·dHead 向量（nHeads×2dHead 二维或共享一维）。
 * 注意力邻域 = N(i) ∪ {i}（自环）；gatLayer 输出为各头 Σ_j α_ij W_h h_j 的拼接（线性输出）。
 * ========================================================================== */
/** 返回 alpha：[nHeads][n][n]，仅邻域位置非零。 */
function gatAttention(X, adj, W, a, nHeads, opts) {
  assertMatrix(X, 'X'); validateAdj(adj); assertMatrix(W, 'W');
  nHeads = (nHeads != null) ? nHeads : 1;
  if (!Number.isInteger(nHeads) || nHeads < 1) throw new TypeError('gatAttention: nHeads 必须为正整数');
  const slope = (opts && opts.negativeSlope != null) ? opts.negativeSlope : 0.2;
  const n = X.length, dIn = X[0].length;
  const dOutAll = W[0].length;
  if (W.length !== dIn) throw new TypeError('gatAttention: W 行数=' + W.length + '，应等于 X 特征维=' + dIn);
  if (dOutAll % nHeads !== 0) throw new TypeError('gatAttention: W 列数须被 nHeads 整除');
  const dHead = dOutAll / nHeads;
  let Avec;
  if (Array.isArray(a) && typeof a[0] === 'number') {
    if (a.length !== 2 * dHead) throw new TypeError('gatAttention: a 维度应为 2·dHead=' + (2 * dHead));
    Avec = [];
    for (let t = 0; t < nHeads; t++) Avec.push(a);        // 各头共享
  } else {
    assertMatrix(a, 'a', nHeads, 2 * dHead);
    Avec = a;
  }
  const alpha = [];
  for (let t = 0; t < nHeads; t++) {
    const Wh = matMul(X, colSlice(W, t * dHead, (t + 1) * dHead));  // n×dHead
    const at = Avec[t];
    const alphaT = zeros(n, n);
    for (let i = 0; i < n; i++) {
      const nbrs = [i];
      for (let j = 0; j < n; j++) if (j !== i && adj[i][j] !== 0) nbrs.push(j);
      const scores = new Array(nbrs.length);
      for (let idx = 0; idx < nbrs.length; idx++) {
        const j = nbrs[idx];
        let s = 0;
        for (let k = 0; k < dHead; k++) s += at[k] * Wh[i][k] + at[dHead + k] * Wh[j][k];
        scores[idx] = leakyRelu(s, slope);
      }
      const p = softmaxVec(scores);
      for (let idx = 0; idx < nbrs.length; idx++) alphaT[i][nbrs[idx]] = p[idx];
    }
    alpha.push(alphaT);
  }
  return alpha;
}

/** GAT 层：各头输出 Σ_j α_ij W_h h_j 后按头拼接 → n×(nHeads·dHead)。 */
function gatLayer(X, adj, W, a, nHeads, opts) {
  const alpha = gatAttention(X, adj, W, a, nHeads, opts);
  const nHeadsEff = alpha.length;
  const n = X.length, dOutAll = W[0].length, dHead = dOutAll / nHeadsEff;
  const H = zeros(n, dOutAll);
  for (let t = 0; t < nHeadsEff; t++) {
    const Wh = matMul(X, colSlice(W, t * dHead, (t + 1) * dHead));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const aij = alpha[t][i][j];
        if (aij !== 0) for (let k = 0; k < dHead; k++) H[i][t * dHead + k] += aij * Wh[j][k];
      }
    }
  }
  return H;
}

/* ============================================================================
 * O03 通用 MPNN 消息传递（Gilmer et al. 2017）：message/aggregate/update 三阶段可插拔
 * 对每条边 adj[i][j]≠0，节点 i 收到 msgFn(h_i, h_j, i, j)；
 * 孤立节点聚合为全零向量。默认 msg=h_j / agg=mean / upd=恒等。
 * ========================================================================== */
function sumMsgs(msgs) {
  const d = msgs[0].length;
  const s = new Array(d).fill(0);
  for (const m of msgs) for (let k = 0; k < d; k++) s[k] += m[k];
  return s;
}
function meanMsgs(msgs) {
  const s = sumMsgs(msgs);
  for (let k = 0; k < s.length; k++) s[k] /= msgs.length;
  return s;
}
function maxMsgs(msgs) {
  const d = msgs[0].length;
  const s = new Array(d).fill(-Infinity);
  for (const m of msgs) for (let k = 0; k < d; k++) if (m[k] > s[k]) s[k] = m[k];
  return s;
}
function messagePass(X, adj, msgFn, aggFn, updFn) {
  assertMatrix(X, 'X'); validateAdj(adj);
  msgFn = msgFn || ((hi, hj) => hj);
  aggFn = aggFn || meanMsgs;
  updFn = updFn || ((h, m) => m);
  const n = X.length, d = X[0].length;
  const out = [];
  for (let i = 0; i < n; i++) {
    const msgs = [];
    for (let j = 0; j < n; j++) if (adj[i][j] !== 0) msgs.push(msgFn(X[i], X[j], i, j));
    const agg = msgs.length ? aggFn(msgs) : new Array(d).fill(0);
    out.push(updFn(X[i], agg, i));
  }
  return out;
}

/* ============================================================================
 * O04 节点分类：2 层 GCN 前向 + softmax；传入 labels 时全批量梯度下降训练
 * 前向：pre1=Ânorm·X·W1 → H1=relu(pre1) → Z2=Ânorm·H1·W2 → P=softmax(Z2)
 * 梯度（解析）：dZ2=(P−Y)/n；dW2=H1ᵀ·(Â·dZ2)；dH1=Â·dZ2·W2ᵀ；
 *              dPre1=dH1∘1[pre1>0]；dW1=Xᵀ·Â·dPre1（Â 对称）。
 * opts: { labels, lr=0.5, steps=300 }；无 labels 时仅前向。
 * ========================================================================== */
function nodeClassify(X, adj, W1, W2, opts) {
  assertMatrix(X, 'X'); validateAdj(adj); assertMatrix(W1, 'W1'); assertMatrix(W2, 'W2');
  const n = X.length, dIn = X[0].length, h = W1[0].length, c = W2[0].length;
  if (W1.length !== dIn) throw new TypeError('nodeClassify: W1 行数=' + W1.length + '，应等于 X 特征维=' + dIn);
  if (W2.length !== h) throw new TypeError('nodeClassify: W2 行数=' + W2.length + '，应等于隐层维=' + h);
  const o = opts || {};
  const An = symNormalize(adj);
  const copyM = M => M.map(r => r.slice());
  const w1 = copyM(W1), w2 = copyM(W2);
  const relu = x => (x > 0 ? x : 0);
  const forward = (m1, m2) => {
    const pre1 = matMul(matMul(An, X), m1);
    const H1 = pre1.map(r => r.map(relu));
    const Z2 = matMul(matMul(An, H1), m2);
    const P = Z2.map(r => softmaxVec(r));
    return { pre1, H1, Z2, P };
  };
  const labels = o.labels || null;
  if (!labels) {
    const f = forward(w1, w2);
    return { probs: f.P, logits: f.Z2, loss: null, accuracy: null, trained: false, lossHistory: [], W1: w1, W2: w2 };
  }
  if (!Array.isArray(labels) || labels.length !== n) throw new TypeError('nodeClassify: labels 长度必须等于节点数 ' + n);
  for (let i = 0; i < n; i++) {
    if (!(Number.isInteger(labels[i]) && labels[i] >= 0 && labels[i] < c)) {
      throw new TypeError('nodeClassify: labels[' + i + ']=' + labels[i] + ' 越界 [0,' + (c - 1) + ']');
    }
  }
  const lr = o.lr != null ? o.lr : 0.5;
  const steps = o.steps != null ? o.steps : 300;
  if (!(lr > 0) || !Number.isInteger(steps) || steps < 0) throw new TypeError('nodeClassify: 需 lr>0 且 steps≥0 整数');
  const Y = zeros(n, c);
  for (let i = 0; i < n; i++) Y[i][labels[i]] = 1;
  const lossOf = f => {
    let L = 0;
    for (let i = 0; i < n; i++) L -= Math.log(Math.max(f.P[i][labels[i]], 1e-300));
    return L / n;
  };
  const lossHistory = [];
  let f = forward(w1, w2);
  lossHistory.push(lossOf(f));
  for (let it = 0; it < steps; it++) {
    const dZ2 = zeros(n, c);
    for (let i = 0; i < n; i++) for (let k = 0; k < c; k++) dZ2[i][k] = (f.P[i][k] - Y[i][k]) / n;
    const G = matMul(An, dZ2);                       // n×c
    const dH1 = matMul(G, transpose(w2));            // n×h
    const dPre1 = dH1.map((r, i) => r.map((v, k) => (f.pre1[i][k] > 0 ? v : 0)));
    const dW1 = matMul(transpose(X), matMul(An, dPre1));  // dIn×h
    const dW2 = matMul(transpose(f.H1), G);               // h×c
    for (let a = 0; a < dIn; a++) for (let b = 0; b < h; b++) w1[a][b] -= lr * dW1[a][b];
    for (let a = 0; a < h; a++) for (let b = 0; b < c; b++) w2[a][b] -= lr * dW2[a][b];
    f = forward(w1, w2);
    lossHistory.push(lossOf(f));
  }
  let correct = 0;
  for (let i = 0; i < n; i++) {
    let am = 0;
    for (let k = 1; k < c; k++) if (f.P[i][k] > f.P[i][am]) am = k;
    if (am === labels[i]) correct++;
  }
  return { probs: f.P, logits: f.Z2, loss: lossHistory[lossHistory.length - 1], accuracy: correct / n, trained: steps > 0, lossHistory, W1: w1, W2: w2 };
}

/* ============================================================================
 * O07 图采样：邻居采样（每节点从 N(i)\{i} 均匀无放回采 min(k, deg) 个，
 * 部分 Fisher-Yates + 可种子 rng，固定 seed 逐位可复现）。返回 0/1 采样邻接阵。
 * ========================================================================== */
function sampleNeighbors(adj, k, rng) {
  validateAdj(adj);
  if (!Number.isInteger(k) || k < 1) throw new TypeError('sampleNeighbors: k 必须为正整数');
  if (typeof rng !== 'function') throw new TypeError('sampleNeighbors: rng 必须为函数');
  const n = adj.length;
  const out = zeros(n, n);
  for (let i = 0; i < n; i++) {
    const nbrs = [];
    for (let j = 0; j < n; j++) if (j !== i && adj[i][j] !== 0) nbrs.push(j);
    const take = Math.min(k, nbrs.length);
    for (let s = 0; s < take; s++) {
      const idx = s + Math.floor(rng() * (nbrs.length - s));
      const tmp = nbrs[s]; nbrs[s] = nbrs[idx]; nbrs[idx] = tmp;
      out[i][nbrs[s]] = 1;
    }
  }
  return out;
}

/* ============================================================================
 * selfCheck — 8 条不变量（手搓黄金参照 + 反空洞分歧断言）
 * ========================================================================== */
function selfCheck() {
  const cases = [];
  const invariants = [];

  /** caseResult 包装：补齐 W13 契约的 status/passed/detail 字段。 */
  const mkCase = (id, name, actual, expected, tol, detail) => {
    const c = V().caseResult(id, name, actual, expected, tol, detail);
    c.detail = detail || '';
    c.passed = c.status === 'pass';
    return c;
  };
  const push = (id, name, expr, tol, worst, ok, detail, note) => {
    cases.push(mkCase(id, name, ok ? worst : 1, 0, tol, detail));
    invariants.push({ id, name, expr, tol, worst, ok, status: ok ? 'pass' : 'fail', detail, note });
  };

  /* ---- 固定测试图 G1：7 节点，度异质 ---- */
  const G1 = [
    [0, 1, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 1, 1],
    [0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 0]
  ];
  const G1I = G1.map((r, i) => r.map((v, j) => (i === j ? v + 1 : v)));
  /* ---- 节点分类小图 Gc：两个 3-团 + 桥 2-3 ---- */
  const Gc = [
    [0, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 1],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 0]
  ];

  /* ---- 手搓黄金参照（独立实现，不调用本模块任何函数） ---- */
  function handSymNorm(adj) {
    const n = adj.length;
    const A = [], deg = [];
    for (let i = 0; i < n; i++) {
      let d = 1;
      const row = [];
      for (let j = 0; j < n; j++) { row.push(i === j ? adj[i][j] + 1 : adj[i][j]); d += adj[i][j]; }
      A.push(row); deg.push(d);
    }
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) A[i][j] = A[i][j] / Math.sqrt(deg[i] * deg[j]);
    return A;
  }
  function handGCN(X, adj, W, b) {                     // relu 激活
    const An = handSymNorm(adj);
    const n = X.length, dIn = X[0].length, dOut = W[0].length;
    const out = [];
    for (let i = 0; i < n; i++) {
      const row = [];
      for (let k = 0; k < dOut; k++) {
        let s = 0;
        for (let l = 0; l < dIn; l++) {
          let ax = 0;
          for (let j = 0; j < n; j++) ax += An[i][j] * X[j][l];
          s += ax * W[l][k];
        }
        const v = s + b[k];
        row.push(v > 0 ? v : 0);
      }
      out.push(row);
    }
    return out;
  }
  function handSageMean(X, adj, W) {
    const n = X.length, dIn = X[0].length, dOut = W[0].length;
    const out = [];
    for (let i = 0; i < n; i++) {
      const nbrs = [];
      for (let j = 0; j < n; j++) if (j !== i && adj[i][j] !== 0) nbrs.push(j);
      const agg = new Array(dIn).fill(0);
      for (const j of nbrs) for (let k = 0; k < dIn; k++) agg[k] += X[j][k];
      if (nbrs.length) for (let k = 0; k < dIn; k++) agg[k] /= nbrs.length;
      const row = [];
      for (let k = 0; k < dOut; k++) {
        let s = 0;
        for (let q = 0; q < dIn; q++) s += X[i][q] * W[q][k];
        for (let q = 0; q < dIn; q++) s += agg[q] * W[dIn + q][k];
        row.push(s);
      }
      out.push(row);
    }
    return out;
  }
  function handGATAlpha(X, adj, W, a, nHeads) {        // LeakyReLU(0.2) 内联手搓
    const n = X.length, dIn = X[0].length, dHead = W[0].length / nHeads;
    const out = [];
    for (let t = 0; t < nHeads; t++) {
      const Wh = [];
      for (let i = 0; i < n; i++) {
        const row = [];
        for (let k = 0; k < dHead; k++) {
          let s = 0;
          for (let l = 0; l < dIn; l++) s += X[i][l] * W[l][t * dHead + k];
          row.push(s);
        }
        Wh.push(row);
      }
      const alphaT = [];
      for (let i = 0; i < n; i++) {
        const nbrs = [i];
        for (let j = 0; j < n; j++) if (j !== i && adj[i][j] !== 0) nbrs.push(j);
        const scores = [];
        for (const j of nbrs) {
          let s = 0;
          for (let k = 0; k < dHead; k++) s += a[t][k] * Wh[i][k] + a[t][dHead + k] * Wh[j][k];
          scores.push(s < 0 ? 0.2 * s : s);
        }
        const mx = Math.max.apply(null, scores);
        const ex = scores.map(v => Math.exp(v - mx));
        let Z = 0; for (const v of ex) Z += v;
        const row = new Array(n).fill(0);
        for (let idx = 0; idx < nbrs.length; idx++) row[nbrs[idx]] = ex[idx] / Z;
        alphaT.push(row);
      }
      out.push(alphaT);
    }
    return out;
  }
  function handRowMean(adj, X) {                       // (adj+I) 行均值
    const n = adj.length, d = X[0].length;
    const out = [];
    for (let i = 0; i < n; i++) {
      let cnt = 1;
      for (let j = 0; j < n; j++) if (j !== i && adj[i][j] !== 0) cnt++;
      const row = [];
      for (let k = 0; k < d; k++) {
        let s = X[i][k];
        for (let j = 0; j < n; j++) if (j !== i && adj[i][j] !== 0) s += X[j][k];
        row.push(s / cnt);
      }
      out.push(row);
    }
    return out;
  }
  function flattenMats(mats) {
    const out = [];
    for (const M of mats) for (const row of M) for (const v of row) out.push(v);
    return out;
  }

  /* ---------- INV-GNN-001 对称归一化（手搓闭式 + 行和证伪） ---------- */
  {
    const rng = makeRng(0x51E0B);
    const n = 8;
    const GW = zeros(n, n);
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      if (rng() < 0.45) { const w = 0.5 + rng(); GW[i][j] = w; GW[j][i] = w; }
    }
    const An = symNormalize(GW);
    const ref = handSymNorm(GW);
    const dev = maxAbsDiffMat(An, ref);
    let sym = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) sym = Math.max(sym, Math.abs(An[i][j] - An[j][i]));
    // 星形图（度 1 vs 度 8）行和必 ≠ 1 → 证伪行归一混淆
    const star = zeros(9, 9);
    for (let j = 1; j < 9; j++) { star[0][j] = 1; star[j][0] = 1; }
    const As = symNormalize(star);
    let rsDev = 0;
    for (let i = 0; i < 9; i++) {
      let s = 0; for (let j = 0; j < 9; j++) s += As[i][j];
      rsDev = Math.max(rsDev, Math.abs(s - 1));
    }
    const ok = dev <= TOL.TIGHT && sym <= TOL.TIGHT && rsDev > 0.05;
    push('INV-GNN-001', '对称归一化闭式一致 + 行和≠1证伪',
      'D̂^-1/2 Â D̂^-1/2 ≡ 手搓闭式(≤1e-15) 且对称；异质度星形图行和≠1(>0.05)',
      TOL.TIGHT, Math.max(dev, sym), ok,
      'max|Ânorm−手搓|=' + dev.toExponential(2) + ' 对称偏差=' + sym.toExponential(2) +
      ' 星形行和max|Σ−1|=' + rsDev.toFixed(4) + '(应>0.05)',
      '手搓闭式逐元素对照 + 行归一混淆证伪');
  }

  /* ---------- INV-GNN-002 GCN 前向（手搓黄金参照 + W=0 退化） ---------- */
  {
    const rng = makeRng(0x52E0B);
    const X = randMat(rng, 7, 4, 1), W = randMat(rng, 4, 3, 1), b = randMat(rng, 1, 3, 0.5)[0];
    const H = gcnLayer(X, G1, W, b, 'relu');
    const Href = handGCN(X, G1, W, b);
    const dev = maxAbsDiffMat(H, Href);
    const W0 = zeros(4, 3);
    const H0 = gcnLayer(X, G1, W0, b, null);
    let dev0 = 0;
    for (let i = 0; i < 7; i++) for (let k = 0; k < 3; k++) dev0 = Math.max(dev0, Math.abs(H0[i][k] - b[k]));
    const ok = dev <= TOL.TIGHT && dev0 === 0;
    push('INV-GNN-002', 'GCN 前向黄金参照一致 + W=0 退化为偏置广播',
      'gcnLayer ≡ 手搓循环展开(≤1e-15) 且 W=0,act=null 时输出 ≡ b 广播(逐位)',
      TOL.TIGHT, Math.max(dev, dev0), ok,
      'max|H−手搓|=' + dev.toExponential(2) + ' W=0 时 max|H−b|=' + dev0.toExponential(2),
      '手搓黄金参照（不调用本模块函数）+ 零权重退化断言');
  }

  /* ---------- INV-GNN-003 GraphSAGE 聚合（手搓均值 + 三聚合器分歧） ---------- */
  {
    const rng = makeRng(0x53E0B);
    const X = randMat(rng, 7, 4, 1), W = randMat(rng, 8, 2, 1);
    const Hm = sageAggregate(X, G1, W, 'mean');
    const Href = handSageMean(X, G1, W);
    const dev = maxAbsDiffMat(Hm, Href);
    const Hs = sageAggregate(X, G1, W, 'sum');
    const Hx = sageAggregate(X, G1, W, 'max');
    const dms = maxAbsDiffMat(Hm, Hs), dmx = maxAbsDiffMat(Hm, Hx), dsx = maxAbsDiffMat(Hs, Hx);
    const minDiv = Math.min(dms, dmx, dsx);
    const ok = dev <= TOL.TIGHT && minDiv > 1e-6;
    push('INV-GNN-003', 'GraphSAGE mean 手搓一致 + 三聚合器必分歧',
      'mean 聚合 ≡ 手搓邻居均值+concat 线性(≤1e-15) 且 mean/sum/max 两两分歧(>1e-6)',
      TOL.TIGHT, Math.max(dev, 0), ok,
      'max|mean−手搓|=' + dev.toExponential(2) + ' 分歧 mean-sum=' + dms.toExponential(2) +
      ' mean-max=' + dmx.toExponential(2) + ' sum-max=' + dsx.toExponential(2),
      '手搓邻居均值对照 + 非退化分歧断言');
  }

  /* ---------- INV-GNN-004 GAT 注意力（手搓黄金参照 + 行和/均匀性/斜率） ---------- */
  {
    const rng = makeRng(0x54E0B);
    const X = randMat(rng, 7, 4, 1);
    const W = randMat(rng, 4, 6, 1);
    const a = [randMat(rng, 1, 6, 1)[0], randMat(rng, 1, 6, 1)[0]];
    const alpha = gatAttention(X, G1, W, a, 2);
    const aref = handGATAlpha(X, G1, W, a, 2);
    const dev = maxDiffNested(alpha, aref);
    let rsDev = 0;
    for (let t = 0; t < 2; t++) for (let i = 0; i < 7; i++) {
      let s = 0; for (let j = 0; j < 7; j++) s += alpha[t][i][j];
      rsDev = Math.max(rsDev, Math.abs(s - 1));
    }
    let uniDev = 0;
    for (let t = 0; t < 2; t++) for (let i = 0; i < 7; i++) {
      let m = 1;
      for (let j = 0; j < 7; j++) if (j !== i && G1[i][j] !== 0) m++;
      const u = 1 / m;
      for (let j = 0; j < 7; j++) if (alpha[t][i][j] > 0) uniDev = Math.max(uniDev, Math.abs(alpha[t][i][j] - u));
    }
    let leakOk = true, leakDev = 0;
    for (const p of [-1.5, -0.25, -3.0]) {
      const lv = leakyRelu(p, 0.2);
      leakDev = Math.max(leakDev, Math.abs(lv - 0.2 * p));
      if (!(lv < 0)) leakOk = false;                  // 负输入 → 输出严格 < 0
    }
    if (Math.abs(leakyRelu(2.5, 0.2) - 2.5) > TOL.TIGHT) leakOk = false;  // 正输入恒等
    const ok = dev <= TOL.TIGHT && rsDev <= TOL.TIGHT && uniDev > 1e-3 && leakOk && leakDev <= TOL.TIGHT;
    push('INV-GNN-004', 'GAT 注意力黄金参照一致 + 行和=1 + 非均匀 + LeakyReLU 斜率生效',
      '每头 α ≡ 手搓参照(mismatch=0,≤1e-15) 且 Σ_j α_ij=1(≤1e-15) 且 max|α−均匀|>1e-3 且负斜率生效',
      TOL.TIGHT, Math.max(dev, rsDev, leakDev), ok,
      'max|α−手搓|=' + dev.toExponential(2) + ' 行和偏差=' + rsDev.toExponential(2) +
      ' 均匀性偏离=' + uniDev.toExponential(2) + ' 斜率探针偏差=' + leakDev.toExponential(2),
      '手搓注意力黄金参照 + 均匀性证伪 + LeakyReLU 探针');
  }

  /* ---------- INV-GNN-005 MPNN ≡ GCN(无 W) + mean 手搓 + 聚合器分歧 ---------- */
  {
    const rng = makeRng(0x55E0B);
    const X = randMat(rng, 7, 4, 1);
    const dh = [];
    for (let i = 0; i < 7; i++) { let d = 1; for (let j = 0; j < 7; j++) d += G1[i][j]; dh.push(d); }
    // 度缩放消息 + sum 聚合 + 恒等更新 ≡ GCN(无 W)：Σ_j Â_ij h_j/√(d̂i·d̂j)
    const mpGcn = messagePass(X, G1I,
      (hi, hj, i, j) => hj.map(v => v / Math.sqrt(dh[i] * dh[j])),
      sumMsgs, (h, m) => m);
    const gcnRef = gcnLayer(X, G1, identity(4), null, null);
    const devG = maxAbsDiffMat(mpGcn, gcnRef);
    // 平消息 + mean 聚合 ≡ 手搓 (adj+I) 行均值
    const mpMean = messagePass(X, G1I);
    const meanRef = handRowMean(G1, X);
    const devM = maxAbsDiffMat(mpMean, meanRef);
    // mean vs sum 必分歧
    const mpSum = messagePass(X, G1I, null, sumMsgs, null);
    const div = maxAbsDiffMat(mpMean, mpSum);
    const ok = devG <= TOL.TIGHT && devM <= TOL.TIGHT && div > 1e-9;
    push('INV-GNN-005', 'MPNN ≡ GCN(无 W) + mean 手搓一致 + 聚合器切换分歧',
      '度缩放消息+sum聚合 ≡ GCN(无W)(≤1e-15) 且 mean聚合 ≡ 手搓行均值(≤1e-15) 且 mean≠sum(>1e-9)',
      TOL.TIGHT, Math.max(devG, devM), ok,
      'max|MPNN−GCN|=' + devG.toExponential(2) + ' max|mean−手搓|=' + devM.toExponential(2) +
      ' mean-sum 分歧=' + div.toExponential(2),
      '双框架等价交叉验证（Gilmer et al.：GCN 是 MPNN 特例）');
  }

  /* ---------- INV-GNN-006 节点分类可训练到过拟合 + 未训练证伪 ---------- */
  {
    const rng = makeRng(0x56E0B);
    const n = 6;
    const X = zeros(n, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) X[i][j] = (i === j ? 1 : 0) + (rng() - 0.5) * 0.1;
    const W1 = randMat(makeRng(0x56E0C), 6, 8, 0.7);
    const W2 = randMat(makeRng(0x56E0D), 8, 2, 0.7);
    const labels = [0, 0, 0, 1, 1, 1];
    const base = nodeClassify(X, Gc, W1, W2, { labels, steps: 0 });       // 未训练
    const trained = nodeClassify(X, Gc, W1, W2, { labels, lr: 0.5, steps: 300 });
    const ok = trained.accuracy === 1 && base.accuracy < 1 && trained.loss < base.loss;
    push('INV-GNN-006', '节点分类 300 步训练准确率=1.0 + 未训练<1.0',
      '训练后 acc=1.0 且损失严格下降 且 同初始化未训练 acc<1.0',
      0, ok ? 0 : 1, ok,
      '未训练 acc=' + base.accuracy.toFixed(3) + ' 训练 acc=' + trained.accuracy.toFixed(3) +
      ' loss ' + base.loss.toFixed(4) + '→' + trained.loss.toExponential(2),
      '过拟合训练证明前向+解析梯度正确（可训练性即正确性）');
  }

  /* ---------- INV-GNN-007 邻居采样：种子复现 + 分歧 + 约束 ---------- */
  {
    const G2 = zeros(12, 12);
    for (let j = 1; j < 12; j++) { G2[0][j] = 1; G2[j][0] = 1; }   // 节点 0 度 11
    for (let j = 2; j <= 6; j++) { G2[1][j] = 1; G2[j][1] = 1; }   // 节点 1 度 6
    const k = 3;
    let sameDev = 0;
    for (const sd of [0x51, 0x99, 0x1234]) {
      const s1 = sampleNeighbors(G2, k, makeRng(sd));
      const s2 = sampleNeighbors(G2, k, makeRng(sd));
      sameDev = Math.max(sameDev, maxAbsDiffMat(s1, s2));
    }
    let divCount = 0, constraintOk = true;
    for (let p = 0; p < 20; p++) {
      const r1 = sampleNeighbors(G2, k, makeRng(0xA000 + p));
      const r2 = sampleNeighbors(G2, k, makeRng(0xB000 + p));
      if (maxAbsDiffMat(r1, r2) > 0) divCount++;
      for (let i = 0; i < 12; i++) {
        let cnt = 0, deg = 0;
        for (let j = 0; j < 12; j++) {
          if (G2[i][j] !== 0 && j !== i) deg++;
          if (r1[i][j] !== 0) { cnt++; if (G2[i][j] === 0 || j === i) constraintOk = false; }
        }
        if (cnt > Math.min(k, deg)) constraintOk = false;
      }
    }
    const ok = sameDev === 0 && divCount >= 18 && constraintOk;
    push('INV-GNN-007', '邻居采样种子复现 + 异种子分歧 + 出度约束',
      '同 seed 两次采样逐位相等(=0) 且 20 对异种子 ≥18 对分歧 且 出度 ≤ min(k,deg) 且为原始邻居子集',
      0, ok ? 0 : 1, ok,
      '同 seed max diff=' + sameDev + ' 异种子分歧对数=' + divCount + '/20 约束满足=' + constraintOk,
      '部分 Fisher-Yates + sfc32 可种子复现（分歧概率上界 ~1e-4）');
  }

  /* ---------- INV-GNN-008 全流水线确定性 ---------- */
  {
    const WG = randMat(makeRng(0x58E3), 4, 3, 1), bG = randMat(makeRng(0x58E4), 1, 3, 0.5)[0];
    const WGA = randMat(makeRng(0x58E5), 4, 6, 1);
    const AGA = [randMat(makeRng(0x58E6), 1, 6, 1)[0], randMat(makeRng(0x58E7), 1, 6, 1)[0]];
    const WS = randMat(makeRng(0x58E8), 8, 2, 1);
    const Wc1 = randMat(makeRng(0x58E1), 6, 8, 0.7), Wc2 = randMat(makeRng(0x58E2), 8, 2, 0.7);
    const Lc = [0, 0, 0, 1, 1, 1];
    const pipeline = (seed) => {
      const rng = makeRng(seed);
      const X = randMat(rng, 7, 4, 1);
      const sampled = sampleNeighbors(G1, 3, rng);
      const h1 = gcnLayer(X, G1, WG, bG, 'relu');
      const h2 = gatLayer(X, G1, WGA, AGA, 2);
      const h3 = sageAggregate(X, sampled, WS, 'mean');
      const mp = messagePass(X, G1I);
      const Xc = zeros(6, 6);
      for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) Xc[i][j] = (i === j ? 1 : 0) + (rng() - 0.5) * 0.1;
      const cls = nodeClassify(Xc, Gc, Wc1, Wc2, { labels: Lc, lr: 0.5, steps: 60 });
      return flattenMats([sampled, h1, h2, h3, mp, cls.probs]).concat([cls.loss, cls.accuracy]);
    };
    const rA1 = pipeline(0x9E3779B9), rA2 = pipeline(0x9E3779B9), rB = pipeline(0xDEADBEEF);
    let sameDiff = 0, crossDiff = 0;
    for (let i = 0; i < rA1.length; i++) {
      sameDiff = Math.max(sameDiff, Math.abs(rA1[i] - rA2[i]));
      crossDiff = Math.max(crossDiff, Math.abs(rA1[i] - rB[i]));
    }
    const ok = sameDiff === 0 && crossDiff > 1e-6;
    push('INV-GNN-008', '全流水线同 seed 逐位确定 + 异 seed 分歧',
      '同 seed 两次全流水线输出逐位相等(=0) 且 异 seed 分歧(>1e-6)',
      0, ok ? 0 : 1, ok,
      '同 seed max diff=' + sameDiff.toExponential(2) + ' 异 seed max diff=' + crossDiff.toExponential(2) +
      ' (' + rA1.length + ' 维输出)',
      '采样→GCN→GAT→SAGE→MPNN→节点分类训练 全链路确定性');
  }

  const started = Date.now();
  return V().buildReport({
    module: 'gnn-forge', version: '1.0.0', layer: 'L7', cases, invariants, seed: '0x9E3779B9', started,
    tol: {
      'INV-GNN-001': 1e-15, 'INV-GNN-002': 1e-15, 'INV-GNN-003': 1e-15, 'INV-GNN-004': 1e-15,
      'INV-GNN-005': 1e-15, 'INV-GNN-006': 0, 'INV-GNN-007': 0, 'INV-GNN-008': 0
    }
  });
}

/* ============================================================================
 * API 导出 + 双端挂载 + CLI 自检
 * ========================================================================== */
const API = {
  name: 'gnn-forge', version: '1.0.0', layer: 'L7', TOL,
  makeRng, sfc32, splitmix32,
  assertMatrix, validateAdj, zeros, identity, transpose, matMul, colSlice,
  leakyRelu, resolveAct, softmaxVec,
  symNormalize, gcnLayer, sageAggregate, gatAttention, gatLayer,
  messagePass, sumMsgs, meanMsgs, maxMsgs,
  nodeClassify, sampleNeighbors,
  selfCheck
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeGNN = API;

if (typeof process !== 'undefined' && typeof require !== 'undefined' && require.main === module) {
  const argv = process.argv.slice(2);
  if (argv.includes('--self-check') || argv.includes('--json')) {
    const r = selfCheck({ quick: argv.includes('--quick') });
    const out = argv.includes('--json') ? JSON.stringify(r, null, 2) : V().formatTableSafe(r);
    process.stdout.write(out + '\n');
    process.exit(r.failed === 0 ? 0 : 1);
  }
}
