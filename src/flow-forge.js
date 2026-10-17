/**
 * @forge-module flow-forge
 * @layer L7
 * @version 1.0.0
 * @brief     网络流与图划分内核：Edmonds-Karp（BFS 最短增广路）与 Dinic（分层网络 + 阻塞流 + 当前弧优化）双实现最大流；残量可达集最小割（含割边清单）；二分图最大匹配 Hopcroft-Karp 与「二分图 → 流网络」归约双口径；标签传播 LPA 社区发现（按节点 id 升序确定性更新、固定迭代上限）；Welsh-Powell 度降序贪心图着色
 * @reuse     Ford & Fulkerson《Maximal flow through a network》(Canadian Journal of Mathematics 1956)|https://doi.org/10.4153/CJM-1956-045-5|无声明|A|增广路框架与最大流-最小割定理（maxflow ≡ mincut）
 * @reuse     Edmonds & Karp《Theoretical improvements in algorithmic efficiency for network flow problems》(JACM 1972)|https://doi.org/10.1145/321694.321699|无声明|A|BFS 最短增广路策略（O(VE²)），保证整容量网络有限步终止
 * @reuse     Dinic《Algorithm for solution of a problem of maximum flow in a network with power estimation》(Soviet Math. Doklady 1970)|https://doi.org/10.1007/BF01089429|无声明|A|分层网络 level graph + 阻塞流 + 当前弧 iter[] 指针
 * @reuse     Hopcroft & Karp《An n^2.5 algorithm for maximum matchings in bipartite graphs》(SIAM J. Comput. 1973)|https://doi.org/10.1137/0202019|无声明|A|分层 BFS + 极大不相交最短增广路集批量增广
 * @reuse     Raghavan, Albert & Kumara《Near linear time algorithm to detect community structures in large-scale networks》(Phys. Rev. E 76 036106, 2007)|https://doi.org/10.1103/PhysRevE.76.036106|无声明|A|标签传播 LPA：按邻域众数标签迭代更新、近线性社区发现
 * @reuse     Welsh & Powell《An upper bound for the chromatic number of a graph and its application to timetabling problems》(Computer Journal 1967)|无声明|A|度降序贪心着色与 Δ+1 上界
 * @beyond    纯 JS 单文件、零 npm 依赖、零构建，双端同构 UMD（window.ForgeFlow + module.exports）；直接站在 Ford&Fulkerson、Edmonds&Karp、Dinic、Hopcroft&Karp、Raghavan(LPA)、Welsh&Powell 六篇世界级原始文献之上按原式实现；每条不变量均配独立手搓黄金参照（暴力枚举全部顶点子集的最小割、Kuhn 增广路匹配、三重循环团数）而非自证；并以反空洞铁律断言「非最小割容量严格 ≥ 最小割」「零容量网络 maxflow=0」「K_{m,m} 匹配=m」「无边图匹配=0」「两个不相连团的社区数 ≥2」「无边图每点自成一社区」「K_n 着色数=n」「同 seed 逐位相等 / 异 seed 必分歧」，证明函数非恒等、非平凡、非自证。
 * @invariant INV-FLW-001|最大流双实现一致：Edmonds-Karp ≡ Dinic 最大流值（整容量 EXACT）；流守恒（除源汇外每点净出流=0，EXACT）；容量约束（−cap[v][u] ≤ f[u][v] ≤ cap[u][v]，EXACT）；且流值 ≡ 源点净出流|0|双实现逐位对照 + 全点守恒残差扫描 + 全边容量越界扫描
 * @invariant INV-FLW-002|最大流 = 最小割（核心定理）：maxflow 值 ≡ 手搓暴力枚举全部顶点子集得到的最小割容量（小图 n≤12，EXACT）；残量可达集割容量 ≡ maxflow（EXACT）；且存在容量严格大于最小割的割（证伪「所有割都等于最小割」的平凡恒等）|0|2^k 子集全枚举黄金参照 + 割容量重算 + 严格大于计数断言
 * @invariant INV-FLW-003|流值上界：maxflow ≤ 源点出边容量和 且 ≤ 汇点入边容量和（EXACT）；且存在实例严格小于上界（证伪平凡等号）；零容量网络 maxflow=0 且最小割容量=0（边界证伪）|0|上界逐种子扫描 + 严格小样本断言 + 全零容量网络断言
 * @invariant INV-FLW-004|二分图匹配：Hopcroft-Karp 匹配数 ≡ 手搓 Kuhn 增广路独立实现（EXACT）；匹配合法（左右各点至多一条匹配边、配对边真实存在、matchL/matchR 互逆，EXACT）；完全二分图 K_{m,m} 匹配数=m（闭式 EXACT，m=1..6）；无边图匹配数=0|0|独立增广路黄金参照 + 匹配合法性全扫描 + 闭式/边界断言
 * @invariant INV-FLW-005|匹配 vs 最大流归约：同一批随机二分图上「二分图 → 流网络（源/左/右/汇，单位容量）」的 Dinic maxflow ≡ Hopcroft-Karp 匹配数 ≡ 手搓 Kuhn 匹配数（EXACT）——证明 O10 与 O11 两项微指令内在一致|0|三口径逐位对照（同批随机图）+ 归约网络流守恒与整性校验
 * @invariant INV-FLW-006|标签传播社区：收敛后再迭代一轮标签逐位不变（EXACT 稳定点）；1 ≤ 社区数 ≤ 节点数；两个不相连团的社区数 ≥2（证伪「恒为单社区」）；无边图每个节点自成一社区（社区数=n）；单个团不拆分（社区数=1）|0|额外一轮稳定点复算 + 边界图断言 + 团/双团结论断言
 * @invariant INV-FLW-007|图着色合法性：贪心着色结果中任意边两端颜色不同（EXACT 全边验证）；着色数 ≥ 手搓暴力团数下界（小图 n≤12 全子集枚举，EXACT）；完全图 K_n 着色数=n（闭式 EXACT，n=1..8）；二部图断言合法即可（贪心可超 2 色）；着色数 ≤ Δ+1（Welsh-Powell 上界）|0|全边冲突扫描 + 暴力团数下界 + 闭式断言 + Δ+1 上界
 * @invariant INV-FLW-008|确定性：同 seed 随机图上全流水线（最大流 EK/Dinic + 最小割 + 匹配 + 归约流 + LPA + 着色）两次运行逐位相等；异 seed 必分歧（反空洞）|0|流水线 JSON 逐位比对 + 异种子分歧断言
 * @seed      0x9E3779B9
 */
'use strict';

/* ============================================================================
 * 项目统一容差（九档，与 core-verify.TOL 同值，本地声明保证零依赖）
 * ========================================================================== */
const TOL = { EXACT: 0, ULP: 1e-16, TIGHT: 1e-15, ORTHO: 1e-14, RESID: 1e-12, SOLVE: 1e-10, GRAD: 1e-6, MONO: -1e-9, CONV: 1e-6 };

/** 残量正性判据（整容量网络等价于 >0；对浮点容量提供 1e-12 保护带）。 */
const RESID_EPS = 1e-12;

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
    let t = (a + b + d) >>> 0;
    d = (d + 1) >>> 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) >>> 0;
    c = ((c << 21) | (c >>> 11)) >>> 0;
    c = (c + t) >>> 0;
    return (t >>> 0) / 4294967296;
  }
  for (let i = 0; i < 12; i++) next();
  return next;
}

/* ============================================================================
 * 纯 JS SHA-256（FIPS 180-4）：浏览器端 hash 回退（Node 端走 crypto 读本文件）
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
      const s1 = (_rotr(x2, 17) ^ (_rotr(x2, 19)) ^ (x2 >>> 10)) | 0;
      w[t] = (w[t - 16] + s0 + w[t - 7] + s1) | 0;
    }
    let a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
    for (let t = 0; t < 64; t++) {
      const S1 = (_rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)) | 0;
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

/* ============================================================================
 * 输入校验 + 图/网络构造（全部不良输入抛 TypeError，绝不静默返回）
 * ========================================================================== */
function isInt(x) { return typeof x === 'number' && isFinite(x) && Math.floor(x) === x; }

function zeros(n) { return Array.from({ length: n }, () => new Array(n).fill(0)); }

/** 校验容量方阵：n×n 有限非负；s≠t 且下标合法。 */
function checkNetwork(cap, s, t) {
  if (!Array.isArray(cap) || cap.length === 0) throw new TypeError('flow-forge: cap 必须为非空 n×n 二维容量数组');
  const n = cap.length;
  for (let i = 0; i < n; i++) {
    if (!Array.isArray(cap[i]) || cap[i].length !== n) throw new TypeError('flow-forge: cap 必须为 n×n 方阵');
    for (let j = 0; j < n; j++) {
      const c = cap[i][j];
      if (typeof c !== 'number' || !isFinite(c) || c < 0) throw new TypeError('flow-forge: 容量必须为有限非负数');
    }
  }
  if (!isInt(s) || !isInt(t) || s < 0 || t < 0 || s >= n || t >= n) throw new TypeError('flow-forge: 源/汇下标越界或非法');
  if (s === t) throw new TypeError('flow-forge: 源点不得等于汇点');
  return n;
}

/** 由有向边表 [u,v,c] 构造容量方阵（重边容量相加）。 */
function makeCap(n, edges) {
  if (!isInt(n) || n <= 0) throw new TypeError('makeCap: n 必须为正整数');
  if (!Array.isArray(edges)) throw new TypeError('makeCap: edges 必须为数组');
  const cap = zeros(n);
  for (const e of edges) {
    if (!Array.isArray(e) || e.length < 2) throw new TypeError('makeCap: 每条边必须为 [u, v] 或 [u, v, c]');
    const u = e[0], v = e[1], c = e.length > 2 ? e[2] : 1;
    if (!isInt(u) || !isInt(v) || u < 0 || v < 0 || u >= n || v >= n) throw new TypeError('makeCap: 顶点下标越界或非法');
    if (u === v) throw new TypeError('makeCap: 不接受自环');
    if (typeof c !== 'number' || !isFinite(c) || c < 0) throw new TypeError('makeCap: 容量必须为有限非负数');
    cap[u][v] += c;
  }
  return cap;
}

/** 无向邻接表（去重、升序）；用于 LPA / 着色。 */
function undirectedAdj(n, edges) {
  if (!isInt(n) || n <= 0) throw new TypeError('flow-forge: n 必须为正整数');
  if (!Array.isArray(edges)) throw new TypeError('flow-forge: edges 必须为数组');
  const sets = Array.from({ length: n }, () => new Set());
  for (const e of edges) {
    if (!Array.isArray(e) || e.length < 2) throw new TypeError('flow-forge: 每条边必须为 [u, v]');
    const u = e[0], v = e[1];
    if (!isInt(u) || !isInt(v) || u < 0 || v < 0 || u >= n || v >= n) throw new TypeError('flow-forge: 顶点下标越界或非法');
    if (u === v) throw new TypeError('flow-forge: 不接受自环');
    sets[u].add(v); sets[v].add(u);
  }
  return sets.map((s) => Array.from(s).sort((a, b) => a - b));
}

/* ============================================================================
 * 可种子随机输入生成器（sfc32，seed 默认 0x9E3779B9）
 * ========================================================================== */

/** 随机流网络：s=0, t=n-1；含 0→1→…→n-1 骨架路径保证正流，再补随机边。 */
function randomFlowNetwork(n, m, cMax, seed) {
  if (!isInt(n) || n < 2) throw new TypeError('randomFlowNetwork: n 必须为 ≥2 的整数');
  if (!isInt(m) || m < n - 1) throw new TypeError('randomFlowNetwork: m 必须 ≥ n−1（骨架边数）');
  if (!isInt(cMax) || cMax < 1) throw new TypeError('randomFlowNetwork: cMax 必须为 ≥1 的整数');
  const rng = makeRng(seed == null ? 0x9E3779B9 : seed);
  const total = n * (n - 1);
  if (m > total) throw new TypeError('randomFlowNetwork: m 超过简单有向图上界 n(n−1)');
  const capMap = new Map();
  const add = (u, v) => {
    const k = u * n + v;
    if (!capMap.has(k)) capMap.set(k, 1 + Math.floor(rng() * cMax));
  };
  for (let i = 0; i + 1 < n; i++) add(i, i + 1);      // 骨架：保证 value > 0（反空洞）
  let guard = 0;
  while (capMap.size < m && guard++ < 20000) {
    const u = Math.floor(rng() * n), v = Math.floor(rng() * n);
    if (u !== v) add(u, v);
  }
  const edges = Array.from(capMap, ([k, c]) => [Math.floor(k / n), k % n, c]);
  edges.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  return { n, s: 0, t: n - 1, edges, cap: makeCap(n, edges) };
}

/** 随机二分图：左 nL（0..nL−1）、右 nR，m 条互异边。 */
function randomBipartite(nL, nR, m, seed) {
  if (!isInt(nL) || !isInt(nR) || nL <= 0 || nR <= 0) throw new TypeError('randomBipartite: nL/nR 必须为正整数');
  if (!isInt(m) || m < 0 || m > nL * nR) throw new TypeError('randomBipartite: m 越界');
  const rng = makeRng(seed == null ? 0x9E3779B9 : seed);
  const pairs = [];
  for (let l = 0; l < nL; l++) for (let r = 0; r < nR; r++) pairs.push([l, r]);
  for (let i = 0; i < m; i++) {
    const j = i + Math.floor(rng() * (pairs.length - i));
    const tmp = pairs[i]; pairs[i] = pairs[j]; pairs[j] = tmp;
  }
  const edges = pairs.slice(0, m).map(([l, r]) => [l, r]);
  edges.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  return { nL, nR, edges };
}

/** 随机简单无向图：m 条互异无向边。 */
function randomUndirected(n, m, seed) {
  if (!isInt(n) || n <= 0) throw new TypeError('randomUndirected: n 必须为正整数');
  const rng = makeRng(seed == null ? 0x9E3779B9 : seed);
  const pairs = [];
  for (let u = 0; u < n; u++) for (let v = u + 1; v < n; v++) pairs.push([u, v]);
  if (!isInt(m) || m < 0 || m > pairs.length) throw new TypeError('randomUndirected: m 越界');
  for (let i = 0; i < m; i++) {
    const j = i + Math.floor(rng() * (pairs.length - i));
    const tmp = pairs[i]; pairs[i] = pairs[j]; pairs[j] = tmp;
  }
  const edges = pairs.slice(0, m).map(([u, v]) => [u, v]);
  edges.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  return { n, edges };
}

/* ============================================================================
 * O11-a 最大流：Edmonds-Karp（BFS 最短增广路）
 * 采用斜对称流矩阵 f：f[u][v] = −f[v][u]，残量 r[u][v] = cap[u][v] − f[u][v]。
 * 该表示天然支持反平行边，且容量约束为 −cap[v][u] ≤ f[u][v] ≤ cap[u][v]。
 * ========================================================================== */
function edmondsKarp(cap, s, t) {
  const n = checkNetwork(cap, s, t);
  const flow = zeros(n);
  let value = 0, augmentations = 0, bfsRounds = 0;
  const prev = new Array(n);
  for (;;) {
    bfsRounds++;
    prev.fill(-1);
    prev[s] = -2;
    const q = [s];
    let hit = false;
    for (let qi = 0; qi < q.length && !hit; qi++) {
      const u = q[qi];
      for (let v = 0; v < n; v++) {
        if (prev[v] !== -1) continue;
        if (cap[u][v] - flow[u][v] > RESID_EPS) {
          prev[v] = u;
          if (v === t) { hit = true; break; }
          q.push(v);
        }
      }
    }
    if (prev[t] === -1) break;                       // 残量网络无 s→t 路：已达最大流
    let bottleneck = Infinity;
    for (let v = t; v !== s; v = prev[v]) {
      const u = prev[v];
      const r = cap[u][v] - flow[u][v];
      if (r < bottleneck) bottleneck = r;
    }
    if (!isFinite(bottleneck) || bottleneck <= RESID_EPS) break;
    for (let v = t; v !== s; v = prev[v]) {
      const u = prev[v];
      flow[u][v] += bottleneck;
      flow[v][u] -= bottleneck;
    }
    value += bottleneck;
    augmentations++;
  }
  return finalize(cap, flow, value, s, t, { algorithm: 'edmonds-karp', augmentations: augmentations, rounds: bfsRounds });
}

/* ============================================================================
 * O11-b 最大流：Dinic（BFS 分层 + 阻塞流 DFS + 当前弧 iter[] 优化）
 * ========================================================================== */
function dinicMaxFlow(cap, s, t) {
  const n = checkNetwork(cap, s, t);
  const flow = zeros(n);
  let value = 0, phases = 0, augmentations = 0;
  const level = new Array(n);
  for (;;) {
    level.fill(-1);
    level[s] = 0;
    const q = [s];
    for (let qi = 0; qi < q.length; qi++) {
      const u = q[qi];
      for (let v = 0; v < n; v++) {
        if (level[v] === -1 && cap[u][v] - flow[u][v] > RESID_EPS) { level[v] = level[u] + 1; q.push(v); }
      }
    }
    if (level[t] === -1) break;
    phases++;
    const iter = new Array(n).fill(0);
    // 阻塞流：沿分层图反复 DFS 推送（当前弧指针保证每阶段 O(VE)）
    const dfs = (u, pushed) => {
      if (u === t) return pushed;
      for (; iter[u] < n; iter[u]++) {
        const v = iter[u];
        const r = cap[u][v] - flow[u][v];
        if (r > RESID_EPS && level[v] === level[u] + 1) {
          const tr = dfs(v, Math.min(pushed, r));
          if (tr > RESID_EPS) {
            flow[u][v] += tr;
            flow[v][u] -= tr;
            return tr;
          }
        }
      }
      return 0;
    };
    for (;;) {
      const pushed = dfs(s, Infinity);
      if (!(pushed > RESID_EPS)) break;
      value += pushed;
      augmentations++;
    }
  }
  return finalize(cap, flow, value, s, t, { algorithm: 'dinic', augmentations: augmentations, phases: phases });
}

/** 收尾：残量可达集 → 最小割；并计算流守恒/容量约束残差（供不变量直接断言）。 */
function finalize(cap, flow, value, s, t, stats) {
  const n = cap.length;
  const reach = residualReach(cap, flow, s);
  const sourceSide = [], sinkSide = [];
  for (let v = 0; v < n; v++) (reach[v] ? sourceSide : sinkSide).push(v);
  let cutCapacity = 0;
  const cutEdges = [];
  for (let i = 0; i < sourceSide.length; i++) {
    const u = sourceSide[i];
    for (let v = 0; v < n; v++) {
      if (reach[v]) continue;
      if (cap[u][v] > 0) { cutCapacity += cap[u][v]; cutEdges.push([u, v, cap[u][v]]); }
    }
  }
  // 流守恒：除源汇外每点净出流应为 0
  let conservation = 0;
  for (let v = 0; v < n; v++) {
    if (v === s || v === t) continue;
    let net = 0;
    for (let u = 0; u < n; u++) net += flow[v][u];
    conservation = Math.max(conservation, Math.abs(net));
  }
  // 容量约束：−cap[v][u] ≤ f[u][v] ≤ cap[u][v]
  let capacityViolation = -Infinity;
  for (let u = 0; u < n; u++) {
    for (let v = 0; v < n; v++) {
      const hi = flow[u][v] - cap[u][v];
      const lo = -cap[v][u] - flow[u][v];
      if (hi > capacityViolation) capacityViolation = hi;
      if (lo > capacityViolation) capacityViolation = lo;
    }
  }
  // 源点净出流 / 汇点净入流（应与 value 一致）
  let outS = 0, inT = 0;
  for (let v = 0; v < n; v++) { outS += flow[s][v]; inT += flow[v][t]; }
  return {
    value: value,
    flow: flow,
    cut: { sourceSide: sourceSide, sinkSide: sinkSide, capacity: cutCapacity, edges: cutEdges },
    conservation: conservation,
    capacityViolation: capacityViolation,
    sourceOutflow: outS,
    sinkInflow: inT,
    algorithm: stats.algorithm,
    augmentations: stats.augmentations,
    phases: stats.phases,
    rounds: stats.rounds
  };
}

function residualReach(cap, flow, s) {
  const n = cap.length;
  const seen = new Array(n).fill(false);
  seen[s] = true;
  const q = [s];
  for (let qi = 0; qi < q.length; qi++) {
    const u = q[qi];
    for (let v = 0; v < n; v++) {
      if (!seen[v] && cap[u][v] - flow[u][v] > RESID_EPS) { seen[v] = true; q.push(v); }
    }
  }
  return seen;
}

/* ============================================================================
 * 最小割：maxflow 后取残量可达集 S（含 s），割容量 = Σ_{u∈S,v∉S} cap[u][v]
 * ========================================================================== */
function minCut(cap, s, t) {
  const r = dinicMaxFlow(cap, s, t);
  return {
    capacity: r.cut.capacity,
    sourceSide: r.cut.sourceSide,
    sinkSide: r.cut.sinkSide,
    edges: r.cut.edges,
    value: r.value,
    flow: r.flow
  };
}

/* ============================================================================
 * O10-a 二分图最大匹配：Hopcroft-Karp（BFS 分层 + 极大最短增广路集）
 * 左部 0..nL−1，右部 0..nR−1，edges = [[l, r], ...]
 * ========================================================================== */
function bipartiteMatching(nL, nR, edges) {
  if (!isInt(nL) || !isInt(nR) || nL <= 0 || nR <= 0) throw new TypeError('bipartiteMatching: nL/nR 必须为正整数');
  if (!Array.isArray(edges)) throw new TypeError('bipartiteMatching: edges 必须为数组');
  const adj = Array.from({ length: nL }, () => []);
  const eset = new Set();
  for (const e of edges) {
    if (!Array.isArray(e) || e.length < 2) throw new TypeError('bipartiteMatching: 每条边必须为 [l, r]');
    const l = e[0], r = e[1];
    if (!isInt(l) || !isInt(r) || l < 0 || l >= nL || r < 0 || r >= nR) throw new TypeError('bipartiteMatching: 端点越界');
    const k = l * nR + r;
    if (eset.has(k)) continue;
    eset.add(k);
    adj[l].push(r);
  }
  for (const a of adj) a.sort((x, y) => x - y);
  const matchL = new Array(nL).fill(-1);
  const matchR = new Array(nR).fill(-1);
  const dist = new Array(nL);
  let size = 0, phases = 0;

  const tryAugment = (u) => {
    for (const v of adj[u]) {
      const pu = matchR[v];
      if (pu === -1 || (dist[pu] === dist[u] + 1 && tryAugment(pu))) {
        matchL[u] = v; matchR[v] = u; return true;
      }
    }
    dist[u] = -1;
    return false;
  };

  for (;;) {
    const q = [];
    for (let u = 0; u < nL; u++) {
      if (matchL[u] === -1) { dist[u] = 0; q.push(u); } else dist[u] = -1;
    }
    let foundFree = false;
    for (let qi = 0; qi < q.length; qi++) {
      const u = q[qi];
      for (const v of adj[u]) {
        const pu = matchR[v];
        if (pu === -1) foundFree = true;
        else if (dist[pu] === -1) { dist[pu] = dist[u] + 1; q.push(pu); }
      }
    }
    if (!foundFree) break;
    phases++;
    for (let u = 0; u < nL; u++) if (matchL[u] === -1 && tryAugment(u)) size++;
  }

  const pairs = [];
  for (let l = 0; l < nL; l++) if (matchL[l] !== -1) pairs.push([l, matchL[l]]);
  return { size: size, pairs: pairs, matchL: matchL, matchR: matchR, algorithm: 'hopcroft-karp', phases: phases };
}

/* ============================================================================
 * O10-b 二分图匹配 → 最大流归约（源 0 | 左 1..nL | 右 nL+1..nL+nR | 汇 nL+nR+1）
 * 单位容量；maxflow 值 ≡ 最大匹配数（INV-FLW-005 的内在一致性证据）
 * ========================================================================== */
function matchingAsFlow(nL, nR, edges) {
  const bm = bipartiteMatching(nL, nR, edges);       // 复用校验逻辑
  const N = nL + nR + 2;
  const S = 0, T = N - 1;
  const cap = zeros(N);
  const lId = (l) => 1 + l;
  const rId = (r) => 1 + nL + r;
  for (let l = 0; l < nL; l++) cap[S][lId(l)] = 1;
  for (let r = 0; r < nR; r++) cap[rId(r)][T] = 1;
  for (const e of edges) {                       // 原始左→右边，单位容量
    cap[lId(e[0])][rId(e[1])] = 1;
  }
  const res = dinicMaxFlow(cap, S, T);
  const pairs = [];
  for (let l = 0; l < nL; l++) {
    for (let r = 0; r < nR; r++) {
      if (res.flow[lId(l)][rId(r)] > 0.5) pairs.push([l, r]);
    }
  }
  pairs.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  return { size: res.value, value: res.value, pairs: pairs, conservation: res.conservation, capacityViolation: res.capacityViolation };
}

/* ============================================================================
 * O12 社区发现：标签传播 LPA（确定性版本）
 * 初始化 label[v]=v；每轮按 v 的 id 升序更新：取邻域标签众数，平局取最小标签 id；
 * 无邻居则保持自身标签。达到稳定点或 maxIter（默认 100）后停止，
 * 并额外执行一轮验证稳定点（INV-FLW-006）。
 * ========================================================================== */
function labelPropagation(n, edges, maxIter) {
  if (!isInt(n) || n <= 0) throw new TypeError('labelPropagation: n 必须为正整数');
  if (maxIter == null) maxIter = 100;
  if (!isInt(maxIter) || maxIter < 1) throw new TypeError('labelPropagation: maxIter 必须为 ≥1 的整数');
  const adj = undirectedAdj(n, edges);
  const labels = Array.from({ length: n }, (_, i) => i);

  const onePass = (lab) => {
    let changed = false;
    for (let v = 0; v < n; v++) {
      const cnt = new Map();
      for (const w of adj[v]) cnt.set(lab[w], (cnt.get(lab[w]) || 0) + 1);
      let best = lab[v], bestCnt = -1;
      for (const [l, c] of cnt) {
        if (c > bestCnt || (c === bestCnt && l < best)) { best = l; bestCnt = c; }
      }
      if (best !== lab[v]) { lab[v] = best; changed = true; }
    }
    return changed;
  };

  let used = 0, converged = false;
  for (let it = 0; it < maxIter; it++) {
    used = it + 1;
    if (!onePass(labels)) { converged = true; break; }
  }
  // 稳定点校验：再迭代一轮，标签必须逐位不变
  const after = labels.slice();
  const changedExtra = onePass(after);
  let stable = !changedExtra;
  for (let v = 0; v < n && stable; v++) if (after[v] !== labels[v]) stable = false;

  // 规范化：按标签（=最小成员 id）升序映射为 0..k−1
  const distinct = Array.from(new Set(labels)).sort((a, b) => a - b);
  const idx = new Map();
  distinct.forEach((l, i) => idx.set(l, i));
  const canonical = labels.map((l) => idx.get(l));
  return {
    labels: canonical,
    rawLabels: labels,
    communities: distinct.length,
    iters: used,
    converged: converged,
    stable: stable
  };
}

/* ============================================================================
 * O14 图着色：Welsh-Powell 贪心（度降序，平局取小 id；每点取最小可用色）
 * ========================================================================== */
function greedyColoring(n, edges) {
  if (!isInt(n) || n <= 0) throw new TypeError('greedyColoring: n 必须为正整数');
  const adj = undirectedAdj(n, edges);
  const deg = adj.map((a) => a.length);
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => deg[b] - deg[a] || a - b);
  const colors = new Array(n).fill(-1);
  for (const v of order) {
    const used = new Set();
    for (const w of adj[v]) if (colors[w] !== -1) used.add(colors[w]);
    let c = 0;
    while (used.has(c)) c++;
    colors[v] = c;
  }
  let count = 0;
  for (const c of colors) if (c + 1 > count) count = c + 1;
  let maxDeg = 0;
  for (const d of deg) if (d > maxDeg) maxDeg = d;
  return { colors: colors, count: count, order: order, degrees: deg, maxDegree: maxDeg };
}

/** 合法性复核：任意边两端颜色必须不同（返回冲突边数）。 */
function coloringConflicts(n, edges, colors) {
  const adj = undirectedAdj(n, edges);
  let conflicts = 0;
  for (let u = 0; u < n; u++) for (const v of adj[u]) if (u < v && colors[u] === colors[v]) conflicts++;
  return conflicts;
}

/* ============================================================================
 * 报告构造（forge-selfcheck/1.0，零依赖自包含）
 * ========================================================================== */
function caseOf(id, name, ok, actual, detail, tol) {
  const numActual = (typeof actual === 'number' && isFinite(actual)) ? actual : (ok ? 0 : 1);
  return {
    id: id,
    name: name,
    status: ok ? 'pass' : 'fail',
    passed: !!ok,
    actual: numActual,
    expected: 0,
    tol: (typeof tol === 'number') ? tol : TOL.EXACT,
    detail: detail
  };
}

function computeHash(hashSrc) {
  try {
    if (typeof require !== 'undefined' && typeof __filename !== 'undefined') {
      const crypto = require('crypto');
      const fs = require('fs');
      if (fs.existsSync(__filename)) return 'sha256:' + crypto.createHash('sha256').update(fs.readFileSync(__filename)).digest('hex');
    }
  } catch (e) { /* 浏览器端回退 */ }
  return 'sha256:' + sha256Hex(hashSrc);
}

function buildReport(module, version, layer, cases, invariants, seed) {
  const started = Date.now();
  const passed = cases.filter((c) => c.status === 'pass').length;
  const failed = cases.length - passed;
  let hashSrc = '';
  for (const c of cases) {
    hashSrc += c.id + '|' + c.status + '|' + (typeof c.actual === 'number' ? String(Math.round(c.actual * 1e12)) : String(c.actual)) + '\n';
  }
  return {
    schema: 'forge-selfcheck/1.0',
    module: module,
    version: version,
    layer: layer,
    passed: passed,
    failed: failed,
    skipped: 0,
    duration_ms: Date.now() - started,
    tol: TOL,
    invariants: invariants,
    cases: cases,
    env: {
      platform: (typeof process !== 'undefined' && process.platform) || 'browser',
      arch: (typeof process !== 'undefined' && process.arch) || 'unknown',
      node: (typeof process !== 'undefined' && process.version) || 'browser',
      seed: seed
    },
    hash: computeHash(hashSrc)
  };
}

/* ============================================================================
 * 独立手搓黄金参照（不调用本模块任何导出算法；用于交叉验证，杜绝自证）
 * ========================================================================== */

/** 黄金参照 1：暴力枚举全部顶点子集求最小 s-t 割（n≤12 ⇒ ≤2^10 子集）。 */
function goldenMinCutBrute(cap, s, t) {
  const n = cap.length;
  const others = [];
  for (let v = 0; v < n; v++) if (v !== s && v !== t) others.push(v);
  const k = others.length;
  let min = Infinity, strictlyGreater = 0, totalCuts = 0;
  for (let mask = 0; mask < (1 << k); mask++) {
    const inS = new Array(n).fill(false);
    inS[s] = true;
    for (let i = 0; i < k; i++) if (mask & (1 << i)) inS[others[i]] = true;
    let c = 0;
    for (let u = 0; u < n; u++) {
      if (!inS[u]) continue;
      for (let v = 0; v < n; v++) if (!inS[v]) c += cap[u][v];
    }
    totalCuts++;
    if (c < min) min = c;
    if (c > min) strictlyGreater++;               // 严格劣于当前最优的割（证伪「所有割都最小」）
  }
  return { min: min, strictlyGreater: strictlyGreater, totalCuts: totalCuts };
}

/** 黄金参照 2：Kuhn 增广路（DFS 匈牙利式）独立实现最大二分匹配。 */
function goldenKuhnMatching(nL, nR, edges) {
  const adj = Array.from({ length: nL }, () => []);
  for (const [l, r] of edges) adj[l].push(r);
  for (const a of adj) a.sort((x, y) => x - y);
  const matchR = new Array(nR).fill(-1);
  let size = 0;
  const tryK = (u, seen) => {
    for (const v of adj[u]) {
      if (seen[v]) continue;
      seen[v] = true;
      if (matchR[v] === -1 || tryK(matchR[v], seen)) { matchR[v] = u; return true; }
    }
    return false;
  };
  for (let u = 0; u < nL; u++) {
    const seen = new Array(nR).fill(false);
    if (tryK(u, seen)) size++;
  }
  return { size: size, matchR: matchR };
}

/** 黄金参照 3：暴力枚举全部顶点子集求团数 ω（着色数下界；n≤12）。 */
function goldenCliqueNumber(n, edges) {
  const eset = new Set();
  for (const [u, v] of edges) if (u !== v) eset.add(u < v ? u * n + v : v * n + u);
  let best = 0;
  for (let mask = 1; mask < (1 << n); mask++) {
    const verts = [];
    for (let i = 0; i < n; i++) if (mask & (1 << i)) verts.push(i);
    if (verts.length <= best) continue;
    let ok = true;
    for (let i = 0; i < verts.length && ok; i++) {
      for (let j = i + 1; j < verts.length; j++) {
        if (!eset.has(verts[i] < verts[j] ? verts[i] * n + verts[j] : verts[j] * n + verts[i])) { ok = false; break; }
      }
    }
    if (ok) best = verts.length;
  }
  return best;
}

/** 完全二分图 K_{m,m} 边表（闭式匹配数 = m）。 */
function completeBipartite(m) {
  const edges = [];
  for (let l = 0; l < m; l++) for (let r = 0; r < m; r++) edges.push([l, r]);
  return edges;
}

/** 完全图 K_n 边表（闭式着色数 = n）。 */
function completeGraph(n) {
  const edges = [];
  for (let u = 0; u < n; u++) for (let v = u + 1; v < n; v++) edges.push([u, v]);
  return edges;
}

/** 两个互不相连的团（K_a ∪ K_b，跨团零边）。 */
function twoCliques(a, b) {
  const edges = [];
  for (let u = 0; u < a; u++) for (let v = u + 1; v < a; v++) edges.push([u, v]);
  for (let u = a; u < a + b; u++) for (let v = u + 1; v < a + b; v++) edges.push([u, v]);
  return { n: a + b, edges: edges };
}

/** 匹配合法性扫描：左右各至多一次、边真实存在、matchL/matchR 互逆。 */
function matchingLegal(nL, nR, edges, res) {
  const eset = new Set();
  for (const [l, r] of edges) eset.add(l * nR + r);
  const seenL = new Array(nL).fill(0), seenR = new Array(nR).fill(0);
  for (const [l, r] of res.pairs) {
    if (l < 0 || l >= nL || r < 0 || r >= nR) return false;
    if (!eset.has(l * nR + r)) return false;
    seenL[l]++; seenR[r]++;
    if (res.matchL[l] !== r || res.matchR[r] !== l) return false;
  }
  for (const c of seenL) if (c > 1) return false;
  for (const c of seenR) if (c > 1) return false;
  if (res.pairs.length !== res.size) return false;
  for (let l = 0; l < nL; l++) if (res.matchL[l] !== -1 && seenL[l] !== 1) return false;
  return true;
}

/* ============================================================================
 * selfCheck — 8 条不变量（INV-FLW-001..008）
 * ========================================================================== */
function selfCheck() {
  const cases = [];
  const invariants = [];
  const SEEDS = [0x9E3779B9, 0x1F2E3D4C, 0x5A6B7C8D];

  /* ---------- INV-FLW-001 最大流双实现一致 + 守恒 + 容量 ---------- */
  {
    let valueGap = 0, conservation = 0, capViol = -Infinity, srcGap = 0, allPositive = true, augOk = true;
    let last = null;
    for (const sd of SEEDS) {
      const net = randomFlowNetwork(12, 26, 9, sd);
      const ek = edmondsKarp(net.cap, net.s, net.t);
      const dn = dinicMaxFlow(net.cap, net.s, net.t);
      valueGap = Math.max(valueGap, Math.abs(ek.value - dn.value));
      conservation = Math.max(conservation, ek.conservation, dn.conservation);
      capViol = Math.max(capViol, ek.capacityViolation, dn.capacityViolation);
      srcGap = Math.max(srcGap, Math.abs(ek.value - ek.sourceOutflow), Math.abs(dn.value - dn.sourceOutflow),
        Math.abs(ek.value - ek.sinkInflow), Math.abs(dn.value - dn.sinkInflow));
      if (!(ek.value > 0)) allPositive = false;
      if (!(ek.augmentations > 0 && dn.augmentations > 0)) augOk = false;
      last = { ek: ek, dn: dn, net: net };
    }
    const ok = valueGap <= TOL.EXACT && conservation <= TOL.EXACT && capViol <= TOL.EXACT &&
      srcGap <= TOL.EXACT && allPositive && augOk;
    cases.push(caseOf('INV-FLW-001', 'Edmonds-Karp ≡ Dinic(EXACT) + 流守恒 + 容量约束 + 源出/汇入≡流值', ok,
      Math.max(valueGap, conservation, srcGap, Math.max(0, capViol)),
      '3种子 |EK−Dinic|max=' + valueGap + ' 守恒残差=' + conservation + ' 容量越界=' + capViol +
      ' 源出/汇入偏差=' + srcGap + ' 全正流=' + allPositive + ' 末样本 value=' + last.dn.value +
      '(EK增广' + last.ek.augmentations + '次/Dinic分' + last.dn.phases + '阶段)', TOL.EXACT));
    invariants.push({
      id: 'INV-FLW-001', name: '最大流双实现一致 + 流守恒 + 容量约束', ok: ok,
      expr: 'EK.value≡Dinic.value；∀v∉{s,t} 净出流=0；−cap[v][u]≤f[u][v]≤cap[u][v]',
      tol: TOL.EXACT, worst: Math.max(valueGap, conservation, srcGap), detail: cases[cases.length - 1].detail
    });
  }

  /* ---------- INV-FLW-002 最大流 = 最小割（暴力全子集枚举黄金参照） ---------- */
  {
    let valueVsBrute = 0, cutVsValue = 0, strictSeen = false, cutEdgeSumOk = true;
    let info = '';
    for (const sd of SEEDS) {
      const net = randomFlowNetwork(9, 18, 7, sd);            // n=9 ≤ 12 ⇒ 2^7=128 个割全枚举
      const dn = dinicMaxFlow(net.cap, net.s, net.t);
      const bf = goldenMinCutBrute(net.cap, net.s, net.t);
      valueVsBrute = Math.max(valueVsBrute, Math.abs(dn.value - bf.min));
      cutVsValue = Math.max(cutVsValue, Math.abs(dn.cut.capacity - dn.value));
      if (bf.strictlyGreater > 0) strictSeen = true;
      let sumEdges = 0;
      for (const e of dn.cut.edges) sumEdges += e[2];
      if (Math.abs(sumEdges - dn.cut.capacity) > TOL.EXACT) cutEdgeSumOk = false;
      info = '枚举' + bf.totalCuts + '割 min=' + bf.min + ' 严格>min 的有' + bf.strictlyGreater + '个';
    }
    const ok = valueVsBrute <= TOL.EXACT && cutVsValue <= TOL.EXACT && strictSeen && cutEdgeSumOk;
    cases.push(caseOf('INV-FLW-002', 'maxflow ≡ 暴力枚举全子集最小割(EXACT) + 割容量≡流值 + 存在严格劣割', ok,
      Math.max(valueVsBrute, cutVsValue),
      '3种子 |maxflow−暴力mincut|max=' + valueVsBrute + ' |割容量−流值|max=' + cutVsValue +
      ' 存在严格>最小割的割=' + strictSeen + ' 割边和自洽=' + cutEdgeSumOk + '；' + info, TOL.EXACT));
    invariants.push({
      id: 'INV-FLW-002', name: '最大流 = 最小割（含暴力全子集黄金参照）', ok: ok,
      expr: 'maxflow ≡ min_{S∋s,t∉S} Σ_{u∈S,v∉S} cap[u][v]（EXACT）',
      tol: TOL.EXACT, worst: Math.max(valueVsBrute, cutVsValue), detail: cases[cases.length - 1].detail
    });
  }

  /* ---------- INV-FLW-003 流值上界 + 零容量边界 ---------- */
  {
    let overOut = -Infinity, overIn = -Infinity, strictSeen = false;
    for (const sd of SEEDS) {
      const net = randomFlowNetwork(11, 24, 8, sd);
      const dn = dinicMaxFlow(net.cap, net.s, net.t);
      let outSum = 0, inSum = 0;
      for (let v = 0; v < net.n; v++) { outSum += net.cap[net.s][v]; inSum += net.cap[v][net.t]; }
      overOut = Math.max(overOut, dn.value - outSum);
      overIn = Math.max(overIn, dn.value - inSum);
      if (dn.value < outSum || dn.value < inSum) strictSeen = true;
    }
    // 零容量网络（边界证伪）
    const zeroNet = makeCap(6, [[0, 1, 0], [1, 2, 0], [2, 3, 0], [0, 4, 0], [4, 5, 0], [5, 3, 0]]);
    const zeroRes = dinicMaxFlow(zeroNet, 0, 3);
    const zeroCut = minCut(zeroNet, 0, 3);
    const zeroOk = zeroRes.value === 0 && zeroCut.capacity === 0;
    // 源点无出边
    const deadNet = makeCap(5, [[1, 2, 5], [2, 3, 5], [3, 4, 5]]);
    const deadRes = edmondsKarp(deadNet, 0, 4);
    const deadOk = deadRes.value === 0;
    const ok = overOut <= TOL.EXACT && overIn <= TOL.EXACT && strictSeen && zeroOk && deadOk;
    cases.push(caseOf('INV-FLW-003', 'maxflow ≤ 源出容量和 且 ≤ 汇入容量和 + 存在严格小 + 零容量/死源=0', ok,
      Math.max(overOut, overIn, 0),
      '越界上界max(≤0)=' + Math.max(overOut, overIn) + ' 存在严格小于上界=' + strictSeen +
      ' 零容量网络 value=' + zeroRes.value + ' 割=' + zeroCut.capacity + ' 死源网络 value=' + deadRes.value, TOL.EXACT));
    invariants.push({
      id: 'INV-FLW-003', name: '流值上界 + 零容量边界', ok: ok,
      expr: 'maxflow ≤ Σcap[s][·]；maxflow ≤ Σcap[·][t]；全零容量 ⇒ 0',
      tol: TOL.EXACT, worst: Math.max(overOut, overIn, 0), detail: cases[cases.length - 1].detail
    });
  }

  /* ---------- INV-FLW-004 二分图匹配：HK ≡ 手搓 Kuhn + 合法 + 闭式 ---------- */
  {
    let sizeGap = 0, legal = true, closedOk = true, emptyOk = false;
    let info = '';
    for (const sd of SEEDS) {
      const g = randomBipartite(14, 12, 30, sd);
      const hk = bipartiteMatching(g.nL, g.nR, g.edges);
      const kuhn = goldenKuhnMatching(g.nL, g.nR, g.edges);
      sizeGap = Math.max(sizeGap, Math.abs(hk.size - kuhn.size));
      if (!matchingLegal(g.nL, g.nR, g.edges, hk)) legal = false;
      info = 'nL=' + g.nL + ' nR=' + g.nR + ' m=' + g.edges.length + ' HK=' + hk.size + ' Kuhn=' + kuhn.size;
    }
    for (let m = 1; m <= 6; m++) {
      const edges = completeBipartite(m);
      const r = bipartiteMatching(m, m, edges);
      if (r.size !== m || !matchingLegal(m, m, edges, r)) closedOk = false;
    }
    const emptyRes = bipartiteMatching(5, 5, []);
    emptyOk = emptyRes.size === 0 && emptyRes.pairs.length === 0;
    const ok = sizeGap <= TOL.EXACT && legal && closedOk && emptyOk;
    cases.push(caseOf('INV-FLW-004', 'Hopcroft-Karp ≡ 手搓 Kuhn 增广路(EXACT) + 匹配合法 + K_{m,m}=m + 无边=0', ok, sizeGap,
      '3种子 |HK−Kuhn|max=' + sizeGap + ' 合法=' + legal + ' K_{m,m}(m=1..6)=' + closedOk +
      ' 无边图=0:' + emptyOk + '；' + info, TOL.EXACT));
    invariants.push({
      id: 'INV-FLW-004', name: '二分图最大匹配正确性（独立增广路黄金参照）', ok: ok,
      expr: 'HK.size ≡ Kuhn.size；匹配合法；K_{m,m}=m；无边图=0',
      tol: TOL.EXACT, worst: sizeGap, detail: cases[cases.length - 1].detail
    });
  }

  /* ---------- INV-FLW-005 匹配 ≡ 最大流归约（同批随机图三口径） ---------- */
  {
    let gap = 0, consOk = true, info = '';
    for (const sd of SEEDS) {
      const g = randomBipartite(12, 12, 28, sd);
      const hk = bipartiteMatching(g.nL, g.nR, g.edges);
      const kuhn = goldenKuhnMatching(g.nL, g.nR, g.edges);
      const red = matchingAsFlow(g.nL, g.nR, g.edges);
      gap = Math.max(gap, Math.abs(red.size - hk.size), Math.abs(red.size - kuhn.size));
      if (!(red.conservation <= TOL.EXACT && red.capacityViolation <= TOL.EXACT)) consOk = false;
      if (!matchingLegal(g.nL, g.nR, g.edges, { pairs: red.pairs, size: red.size, matchL: buildMatchL(g.nL, red.pairs), matchR: buildMatchR(g.nR, red.pairs) })) consOk = false;
      info = '归约流=' + red.size + ' HK=' + hk.size + ' Kuhn=' + kuhn.size + ' 匹配边=' + red.pairs.length;
    }
    const ok = gap <= TOL.EXACT && consOk;
    cases.push(caseOf('INV-FLW-005', '二分图匹配数 ≡ 归约流网络 maxflow(EXACT) 三口径一致', ok, gap,
      '3种子 |归约流−HK/Kuhn|max=' + gap + ' 归约网络守恒/容量/匹配合法=' + consOk + '；' + info, TOL.EXACT));
    invariants.push({
      id: 'INV-FLW-005', name: '匹配 vs 最大流归约内在一致', ok: ok,
      expr: 'matchingAsFlow().size ≡ bipartiteMatching().size ≡ goldenKuhn().size（EXACT）',
      tol: TOL.EXACT, worst: gap, detail: cases[cases.length - 1].detail
    });
  }

  /* ---------- INV-FLW-006 标签传播社区：稳定点 + 边界断言 ---------- */
  {
    let stableAll = true, boundsOk = true, convergedAll = true, multiSeen = false, subNSeen = false, info = '';
    for (const sd of SEEDS) {
      const g = randomUndirected(24, 48, sd);
      const r = labelPropagation(g.n, g.edges, 200);
      if (!r.stable) stableAll = false;
      if (!r.converged) convergedAll = false;
      if (!(r.communities >= 1 && r.communities <= g.n)) boundsOk = false;
      if (r.communities >= 2) multiSeen = true;          // 证伪「恒为单社区」
      if (r.communities < g.n) subNSeen = true;          // 证伪「恒为每点一社区」
      info = 'n=' + g.n + ' m=' + g.edges.length + ' 社区=' + r.communities + ' 迭代=' + r.iters;
    }
    const tc = twoCliques(5, 5);
    const tcRes = labelPropagation(tc.n, tc.edges, 100);
    const twoOk = tcRes.communities >= 2;
    const emptyRes = labelPropagation(7, [], 20);
    const emptyOk = emptyRes.communities === 7 && emptyRes.stable;
    const cliqueRes = labelPropagation(8, completeGraph(8), 100);
    const oneOk = cliqueRes.communities === 1;
    const ok = stableAll && boundsOk && convergedAll && twoOk && emptyOk && oneOk && multiSeen && subNSeen;
    cases.push(caseOf('INV-FLW-006', 'LPA 稳定点(再迭代一轮不变) + 1≤社区≤n + 双团≥2 + 无边=n + 单团=1 + 多社区实例', ok, 0,
      '稳定=' + stableAll + ' 收敛=' + convergedAll + ' 界=' + boundsOk + ' 出现≥2社区=' + multiSeen +
      ' 出现<n社区=' + subNSeen + ' 双团K5∪K5社区=' + tcRes.communities +
      ' 无边图(7点)社区=' + emptyRes.communities + ' K8社区=' + cliqueRes.communities + '；' + info, TOL.EXACT));
    invariants.push({
      id: 'INV-FLW-006', name: '标签传播社区发现：稳定点与边界', ok: ok,
      expr: '收敛后再迭代一轮标签逐位不变；1≤社区数≤n；K5∪K5≥2；无边图=n；K8=1；存在≥2社区实例且存在<n社区实例',
      tol: TOL.EXACT, worst: 0, detail: cases[cases.length - 1].detail
    });
  }

  /* ---------- INV-FLW-007 图着色合法性 + 团数下界 + K_n 闭式 + Δ+1 上界 ---------- */
  {
    let conflicts = 0, lowerOk = true, closedOk = true, upperOk = true, bipLegal = true;
    let info = '';
    for (const sd of SEEDS) {
      const g = randomUndirected(20, 46, sd);
      const col = greedyColoring(g.n, g.edges);
      conflicts += coloringConflicts(g.n, g.edges, col.colors);
      if (col.count > col.maxDegree + 1) upperOk = false;
      info = 'n=' + g.n + ' m=' + g.edges.length + ' 色数=' + col.count + ' Δ=' + col.maxDegree;
    }
    // 小图团数下界（n≤12 暴力全子集）
    for (const sd of [0x2B2B, 0x3C3C]) {
      const g = randomUndirected(12, 22, sd);
      const col = greedyColoring(g.n, g.edges);
      const omega = goldenCliqueNumber(g.n, g.edges);
      if (!(col.count >= omega)) lowerOk = false;
      if (coloringConflicts(g.n, g.edges, col.colors) !== 0) conflicts++;
    }
    for (let n = 1; n <= 8; n++) {
      const col = greedyColoring(n, completeGraph(n));
      if (col.count !== n || coloringConflicts(n, completeGraph(n), col.colors) !== 0) closedOk = false;
    }
    // 二部图：只断言合法（贪心可能超过 2 色）
    for (const sd of [0x4D4D, 0x5E5E]) {
      const g = randomBipartite(9, 9, 24, sd);
      const edges = [];                                  // 二部图 → 无向边（右部编号整体右移）
      for (const [l, r] of g.edges) edges.push([l, g.nL + r]);
      const col = greedyColoring(g.nL + g.nR, edges);
      if (coloringConflicts(g.nL + g.nR, edges, col.colors) !== 0) bipLegal = false;
    }
    const ok = conflicts === 0 && lowerOk && closedOk && upperOk && bipLegal;
    cases.push(caseOf('INV-FLW-007', '贪心着色全边合法(EXACT) + 色数≥暴力团数 + K_n=n + 色数≤Δ+1 + 二部图合法', ok, conflicts,
      '冲突边=' + conflicts + ' 团数下界=' + lowerOk + ' K_n(n=1..8)闭式=' + closedOk +
      ' Δ+1上界=' + upperOk + ' 二部图合法=' + bipLegal + '；' + info, TOL.EXACT));
    invariants.push({
      id: 'INV-FLW-007', name: '图着色合法性与界', ok: ok,
      expr: '∀(u,v)∈E colors[u]≠colors[v]；色数 ≥ ω(G)；K_n 色数=n；色数 ≤ Δ+1',
      tol: TOL.EXACT, worst: conflicts, detail: cases[cases.length - 1].detail
    });
  }

  /* ---------- INV-FLW-008 确定性：同 seed 逐位 / 异 seed 分歧 ---------- */
  {
    const pipeline = (sd) => {
      const net = randomFlowNetwork(12, 26, 9, sd);
      const ek = edmondsKarp(net.cap, net.s, net.t);
      const dn = dinicMaxFlow(net.cap, net.s, net.t);
      const mc = minCut(net.cap, net.s, net.t);
      const g = randomBipartite(12, 12, 28, sd);
      const hk = bipartiteMatching(g.nL, g.nR, g.edges);
      const red = matchingAsFlow(g.nL, g.nR, g.edges);
      const ug = randomUndirected(24, 48, sd);
      const lp = labelPropagation(ug.n, ug.edges, 200);
      const col = greedyColoring(ug.n, ug.edges);
      return JSON.stringify([
        ek.value, dn.value, mc.capacity, mc.sourceSide, mc.edges,
        hk.size, hk.pairs, red.size, red.pairs,
        lp.labels, lp.communities, lp.iters, col.colors, col.count
      ]);
    };
    const a = pipeline(0x9E3779B9), b = pipeline(0x9E3779B9), c = pipeline(0xC0FFEE11);
    const identical = a === b, diverged = a !== c;
    const ok = identical && diverged;
    cases.push(caseOf('INV-FLW-008', '同 seed 全流水线两次逐位相等 + 异 seed 必分歧', ok, 0,
      '同seed逐位=' + identical + ' 异seed分歧=' + diverged + ' 指纹长度=' + a.length, TOL.EXACT));
    invariants.push({
      id: 'INV-FLW-008', name: '确定性（同种子逐位复现 / 异种子分歧）', ok: ok,
      expr: 'pipeline(seed) 两次运行逐位相等；pipeline(seed′) 必分歧',
      tol: TOL.EXACT, worst: 0, detail: cases[cases.length - 1].detail
    });
  }

  return buildReport('flow-forge', '1.0.0', 'L7', cases, invariants, '0x9E3779B9');
}

function buildMatchL(nL, pairs) {
  const m = new Array(nL).fill(-1);
  for (const [l, r] of pairs) m[l] = r;
  return m;
}
function buildMatchR(nR, pairs) {
  const m = new Array(nR).fill(-1);
  for (const [l, r] of pairs) m[r] = l;
  return m;
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
  for (const inv of (r.invariants || [])) {
    lines.push('  ' + (inv.ok ? '[INV OK] ' : '[INV FAIL] ') + inv.id + ' ' + inv.name);
  }
  return lines.join('\n');
}

/* ============================================================================
 * API 导出 + 双端挂载（UMD）+ CLI 自检
 * ========================================================================== */
const API = {
  name: 'flow-forge', version: '1.0.0', layer: 'L7', TOL,
  makeRng, sha256Hex,
  makeCap, undirectedAdj, zeros,
  randomFlowNetwork, randomBipartite, randomUndirected,
  completeGraph, completeBipartite, twoCliques,
  edmondsKarp, dinicMaxFlow, minCut,
  bipartiteMatching, matchingAsFlow,
  labelPropagation, greedyColoring, coloringConflicts,
  selfCheck, formatTable
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeFlow = API;

if (typeof require !== 'undefined' && typeof module !== 'undefined' && require.main === module &&
  (process.argv.includes('--self-check') || process.argv.includes('--json') || process.argv.includes('--table'))) {
  const r = selfCheck();
  process.stdout.write((process.argv.includes('--json') ? JSON.stringify(r, null, 2) : formatTable(r)) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
