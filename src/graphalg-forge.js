/**
 * @forge-module graphalg-forge
 * @layer L7
 * @version 1.0.0
 * @brief     图算法内核：DFS 确定序（显式栈）、A* 可采纳启发最短路（h=0 退化为 Dijkstra）、PageRank 幂迭代、连通分量（并查集）、Kruskal+Prim 双实现最小生成树、Kahn 拓扑排序（有环抛错）、Tarjan 强连通分量、三角计数（度排序定向法）、度/接近/介数（Brandes）中心性
 * @reuse     Page & Brin《The Anatomy of a Large-Scale Hypertextual Web Search Engine》(Computer Networks 1998)|https://dl.acm.org/doi/10.1016/S0169-7552(98)00110-X|未声明|A|PageRank 幂迭代原式（d=0.85，均匀传送）
 * @reuse     Tarjan《Depth-First Search and Linear Graph Algorithms》(SIAM J. Comput. 1972)|https://doi.org/10.1137/0201010|未声明|A|DFS 线性序与 SCC 线性算法
 * @reuse     Brandes《A Faster Algorithm for Betweenness Centrality》(Social Networks 2001)|https://doi.org/10.1080/0022250X.2001.9990249|未声明|A|介数中心性单源累积算法
 * @reuse     CLRS《Introduction to Algorithms》(MIT Press)|https://mitpress.mit.edu/9780262046305/|未声明|A|Kruskal/Prim（贪心 MST）、Kahn 拓扑排序、BFS 章节伪码
 * @beyond    纯 JS 单文件零 npm 依赖零构建，双端同构 UMD（window.ForgeGraphAlg + module.exports）；直接站在 Page&Brin、Tarjan、Brandes、CLRS 世界级原始文献之上——算法按原式实现；所有不变量均含独立手搓黄金参照（递归 DFS、O(n²) Dijkstra、长迭代 PageRank、Union-Find/BFS 双口径 CC、Kosaraju、三重循环三角、逐对 BFS 介数、MST 割性质证书），并以反空洞铁律断言「DFS 序≠BFS 序」「排名≠均匀」「SCC 数≠无向分量数」「不可采纳 h 允许次优」「同 seed 逐位可复现 / 异 seed 必分歧」，证明函数非恒等、非自证。
 * @invariant INV-GA-001|DFS：迭代显式栈实现与手搓递归 DFS 前序逐位一致（邻居按 id 升序，EXACT）；输出为可达集的一个排列；且 DFS 序 ≠ BFS 序（证伪遍历混淆）|0|逐位比对 + 排列校验 + DFS/BFS 分歧断言
 * @invariant INV-GA-002|A*：h=0 时 dist ≡ 手搓 O(n²) Dijkstra 黄金参照（TIGHT）；可采纳欧氏 h>0 探索节点数严格少于 Dijkstra（性能真）；h×10（不可采纳）结果恒 ≥ 最优且存在严格次优实例（证伪边界）；路径合法（首尾/连边/权和）|1e-15|黄金参照 + 探索计数 + 次优边界扫描
 * @invariant INV-GA-003|PageRank：Σrank=1（TIGHT）；收敛残差<1e-10；与手搓 30000 轮长迭代黄金参照一致（≤1e-10）；且 max|rank−1/n|>1e-3（证伪平凡均匀）|1e-10|归一化恒等式 + 长迭代对照 + 均匀证伪
 * @invariant INV-GA-004|连通分量 ≡ 手搓 Union-Find（EXACT 规范标签）；SCC(Tarjan) ≡ 手搓 Kosaraju 独立实现（EXACT 规范标签）；同一有向路径图 SCC 数=4 ≠ 无向连通分量数=1（证伪混淆）|0|双口径独立实现逐位对照 + 计数分歧断言
 * @invariant INV-GA-005|拓扑排序（Kahn）：全边 pos[u]<pos[v]（EXACT）；升序/降序两种合法实现序可不同但均合法（非唯一性断言）；有环必抛错（含自环）|0|逐边约束扫描 + 双实现合法序 + 抛错断言
 * @invariant INV-GA-006|MST：Kruskal ≡ Prim 总权（整数权 EXACT/TIGHT）；恰 n−1 条树边、无环且Spanning（EXACT）；割性质证书：删任一树边后跨割最小非树边权 ≥ 树边权（逐树边扫描，证伪平凡最优性）|1e-15|双实现对照 + 结构校验 + 割性质全扫描
 * @invariant INV-GA-007|三角计数：度排序定向法与手搓三重循环逐位一致（小图 n≤20，EXACT）；定向法配对检查操作数 ≪ n³（<n³/10，性能真）|0|三重循环黄金参照 + 操作数上界断言
 * @invariant INV-GA-008|中心性：Brandes 介数 ≡ 手搓逐对 BFS 路径计数（唯一最短路图上逐位一致，TIGHT）；度中心性 ≡ 手搓计数（EXACT）；同 seed 全流水线两次逐位相等、异 seed 必分歧（反空洞）|1e-15|逐对路径计数黄金参照 + 双跑逐位 + 异种子分歧
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
      const s1 = (_rotr(x2, 17) ^ _rotr(x2, 19) ^ (x2 >>> 10)) | 0;
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
 * 图生成器（可种子、确定性、无重边无自环）
 * ========================================================================== */

/** 有向随机图：m 条互异有向边，邻接表升序去重。 */
function randomDigraph(n, m, seed) {
  const rng = makeRng(seed == null ? 0x9E3779B9 : seed);
  const total = n * (n - 1);
  if (m > total) throw new TypeError('randomDigraph: m 超过简单有向图上界');
  const idx = new Array(total);
  for (let i = 0; i < total; i++) idx[i] = i;
  for (let i = 0; i < m; i++) {
    const j = i + Math.floor(rng() * (total - i));
    const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
  }
  // 将 [0, n(n-1)) 编码映射为 (u,v), u≠v：u=⌊code/(n−1)⌋，v=code%(n−1) 映射跳过 u 自身
  const edges = [];
  for (let i = 0; i < m; i++) {
    const code = idx[i];
    const u = Math.floor(code / (n - 1));
    const v0 = code % (n - 1);
    edges.push([u, v0 < u ? v0 : v0 + 1]);
  }
  return { n, edges, adj: buildAdj(n, edges) };
}

/** 无向随机图：m 条互异无向边，整数权 [1,wMax]。返回 edges=[u,v,w] 与加权邻接表 adjW。 */
function randomUndirected(n, m, wMax, seed) {
  const rng = makeRng(seed == null ? 0x9E3779B9 : seed);
  const pairs = [];
  for (let u = 0; u < n; u++) for (let v = u + 1; v < n; v++) pairs.push([u, v]);
  if (m > pairs.length) throw new TypeError('randomUndirected: m 超过简单无向图上界');
  for (let i = 0; i < m; i++) {
    const j = i + Math.floor(rng() * (pairs.length - i));
    const t = pairs[i]; pairs[i] = pairs[j]; pairs[j] = t;
  }
  const edges = [];
  for (let i = 0; i < m; i++) edges.push([pairs[i][0], pairs[i][1], 1 + Math.floor(rng() * wMax)]);
  const adjW = Array.from({ length: n }, () => []);
  for (const [u, v, w] of edges) { adjW[u].push([v, w]); adjW[v].push([u, w]); }
  for (const a of adjW) a.sort((x, y) => x[0] - y[0]);
  return { n, edges, adjW };
}

/** 随机 DAG：仅 u<v 的边以概率 p 出现（天然无环，保证拓扑排序可行）。 */
function randomDAG(n, p, seed) {
  const rng = makeRng(seed == null ? 0x9E3779B9 : seed);
  const edges = [];
  for (let u = 0; u < n; u++) for (let v = u + 1; v < n; v++) if (rng() < p) edges.push([u, v]);
  return { n, edges, adj: buildAdj(n, edges) };
}

/** 几何图：n 个随机点，每点连最近 k 邻（双向去重）。用于 A*（欧氏权 + 可采纳欧氏 h）。 */
function geometricGraph(n, k, seed) {
  const rng = makeRng(seed == null ? 0x9E3779B9 : seed);
  const coords = [];
  for (let i = 0; i < n; i++) coords.push([rng() * 10, rng() * 10]);
  const eset = new Set();
  for (let u = 0; u < n; u++) {
    const order = [];
    for (let v = 0; v < n; v++) if (v !== u) order.push([euclid(coords[u], coords[v]), v]);
    order.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    for (let i = 0; i < Math.min(k, order.length); i++) {
      const v = order[i][1];
      eset.add(Math.min(u, v) * n + Math.max(u, v));
    }
  }
  const adj = Array.from({ length: n }, () => []);
  for (const key of eset) {
    const u = Math.floor(key / n), v = key % n;
    adj[u].push(v); adj[v].push(u);
  }
  for (const a of adj) a.sort((x, y) => x - y);
  return { n, coords, adj };
}

function euclid(p, q) { return Math.hypot(p[0] - q[0], p[1] - q[1]); }

/** 有向邻接表构建：升序 + 去重。 */
function buildAdj(n, edges) {
  const sets = Array.from({ length: n }, () => new Set());
  for (const [u, v] of edges) sets[u].add(v);
  const adj = sets.map((s) => Array.from(s).sort((a, b) => a - b));
  return adj;
}

function ukey(u, v, n) { return (u < v ? u * n + v : v * n + u); }

/* ============================================================================
 * O01 DFS：显式栈迭代实现（迭代器栈逐位模拟递归前序；邻居按 id 升序）
 * ========================================================================== */
function dfsOrder(adj, start) {
  const visited = new Array(adj.length).fill(false);
  const order = [];
  const stack = [[start, 0]];
  visited[start] = true;
  while (stack.length) {
    const top = stack[stack.length - 1];
    const u = top[0];
    if (top[1] === 0) order.push(u);                 // 首次进入 u（前序位）
    if (top[1] < adj[u].length) {
      const v = adj[u][top[1]++];
      if (!visited[v]) { visited[v] = true; stack.push([v, 0]); }
    } else {
      stack.pop();                                   // u 的子树完成（回溯）
    }
  }
  return order;
}

/* ============================================================================
 * O03 A*：可采纳启发最短路（懒惰删除二叉堆；w=欧氏边权；h=hScale·欧氏(v,t)）
 * hScale=0 退化为 Dijkstra；hScale=1 可采纳；hScale>1 不可采纳（允许次优）。
 * ========================================================================== */
function heapPush(h, item) {
  h.push(item);
  let i = h.length - 1;
  while (i > 0) {
    const p = (i - 1) >> 1;
    if (h[p][0] <= h[i][0]) break;
    const t = h[p]; h[p] = h[i]; h[i] = t; i = p;
  }
}
function heapPop(h) {
  const top = h[0];
  const last = h.pop();
  if (h.length) {
    h[0] = last;
    let i = 0;
    for (;;) {
      const l = 2 * i + 1, r = l + 1;
      let m = i;
      if (l < h.length && h[l][0] < h[m][0]) m = l;
      if (r < h.length && h[r][0] < h[m][0]) m = r;
      if (m === i) break;
      const t = h[m]; h[m] = h[i]; h[i] = t; i = m;
    }
  }
  return top;
}

function aStar(adj, coords, s, t, opts) {
  opts = opts || {};
  const hScale = (opts.hScale !== undefined) ? opts.hScale : 1;
  const n = adj.length;
  const dist = new Array(n).fill(Infinity);
  const parent = new Array(n).fill(-1);
  const closed = new Array(n).fill(false);
  const h = (v) => hScale * euclid(coords[v], coords[t]);
  dist[s] = 0;
  const heap = [[h(s), s]];
  let explored = 0;
  while (heap.length) {
    const top = heapPop(heap);
    const u = top[1];
    if (closed[u]) continue;
    closed[u] = true;
    explored++;
    if (u === t) break;
    for (const v of adj[u]) {
      const nd = dist[u] + euclid(coords[u], coords[v]);
      if (nd < dist[v]) { dist[v] = nd; parent[v] = u; heapPush(heap, [nd + h(v), v]); }
    }
  }
  if (!closed[t]) return { dist: Infinity, path: [], explored, found: false };
  const path = [];
  for (let v = t; v !== -1; v = parent[v]) path.push(v);
  path.reverse();
  return { dist: dist[t], path, explored, found: true };
}

/* ============================================================================
 * O05 PageRank：幂迭代（d=0.85，悬挂节点均匀分摊，每轮精确归一 Σ=1）
 * ========================================================================== */
function pageRank(adj, opts) {
  opts = opts || {};
  const n = adj.length;
  const d = (opts.d !== undefined) ? opts.d : 0.85;
  const tol = (opts.tol !== undefined) ? opts.tol : 1e-12;
  const maxIter = opts.maxIter || 1000;
  const outdeg = adj.map((a) => a.length);
  let r = new Array(n).fill(1 / n);
  let residual = Infinity;
  let iters = 0;
  for (; iters < maxIter; iters++) {
    let dangling = 0;
    for (let v = 0; v < n; v++) if (outdeg[v] === 0) dangling += r[v];
    const base = (1 - d) / n + d * dangling / n;
    const rn = new Array(n).fill(base);
    for (let u = 0; u < n; u++) {
      if (outdeg[u] === 0) continue;
      const share = d * r[u] / outdeg[u];
      for (const v of adj[u]) rn[v] += share;
    }
    let ssum = 0;
    for (let v = 0; v < n; v++) ssum += rn[v];
    for (let v = 0; v < n; v++) rn[v] /= ssum;
    residual = 0;
    for (let v = 0; v < n; v++) { const dd = Math.abs(rn[v] - r[v]); if (dd > residual) residual = dd; }
    r = rn;
    if (residual < tol) { iters++; break; }
  }
  return { ranks: r, residual, iters };
}

/* ============================================================================
 * O06 连通分量（无向）：并查集（路径减半 + 按秩合并），标签规范为分量内最小 id
 * ========================================================================== */
function connectedComponents(n, edges) {
  const parent = new Array(n);
  for (let i = 0; i < n; i++) parent[i] = i;
  const rank = new Array(n).fill(0);
  function find(x) {
    while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  }
  for (const [u, v] of edges) {
    let a = find(u), b = find(v);
    if (a === b) continue;
    if (rank[a] < rank[b]) { const t = a; a = b; b = t; }
    parent[b] = a;
    if (rank[a] === rank[b]) rank[a]++;
  }
  const minOf = new Array(n).fill(-1);
  for (let v = 0; v < n; v++) {
    const r = find(v);
    if (minOf[r] === -1 || v < minOf[r]) minOf[r] = v;
  }
  const comp = new Array(n);
  let count = 0;
  const seen = new Map();
  for (let v = 0; v < n; v++) {
    const label = minOf[find(v)];
    if (!seen.has(label)) seen.set(label, count++);
    comp[v] = label;
  }
  return { comp, count };
}

/* ============================================================================
 * O07 最小生成树：Kruskal（排序 + 并查集）与 Prim（O(n²) 选择）双实现
 * ========================================================================== */
function kruskalMST(n, edges) {
  const order = edges.map((_, i) => i).sort((a, b) => (edges[a][2] - edges[b][2]) || (a - b));
  const parent = new Array(n);
  for (let i = 0; i < n; i++) parent[i] = i;
  function find(x) {
    while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  }
  const tree = [];
  let total = 0;
  for (const i of order) {
    const [u, v, w] = edges[i];
    const ru = find(u), rv = find(v);
    if (ru === rv) continue;
    parent[ru] = rv;
    tree.push([u, v, w]);
    total += w;
  }
  return { total, tree };
}

function primMST(n, adjW) {
  const inTree = new Array(n).fill(false);
  const key = new Array(n).fill(Infinity);
  const par = new Array(n).fill(-1);
  key[0] = 0;
  let total = 0;
  const tree = [];
  for (let it = 0; it < n; it++) {
    let u = -1;
    for (let v = 0; v < n; v++) if (!inTree[v] && (u === -1 || key[v] < key[u])) u = v;
    if (u === -1) break;
    inTree[u] = true;
    total += key[u];
    if (par[u] !== -1) tree.push([par[u], u, key[u]]);
    for (const [w, wt] of adjW[u]) if (!inTree[w] && wt < key[w]) { key[w] = wt; par[w] = u; }
  }
  return { total, tree };
}

/* ============================================================================
 * O08 拓扑排序：Kahn（每次取最小 id 入度零节点，确定性）；有环抛错
 * ========================================================================== */
function topoSort(n, edges) {
  const adj = Array.from({ length: n }, () => []);
  const indeg = new Array(n).fill(0);
  for (const [u, v] of edges) { adj[u].push(v); indeg[v]++; }
  const removed = new Array(n).fill(false);
  const order = [];
  for (let it = 0; it < n; it++) {
    let pick = -1;
    for (let v = 0; v < n; v++) if (!removed[v] && indeg[v] === 0) { pick = v; break; }
    if (pick === -1) throw new Error('graphalg-forge: 拓扑排序检测到环（剩余 ' + (n - order.length) + ' 个节点互达）');
    removed[pick] = true;
    order.push(pick);
    for (const w of adj[pick]) indeg[w]--;
  }
  return order;
}

/* ============================================================================
 * O09 强连通分量：迭代 Tarjan（标签规范为 SCC 内最小 id）
 * ========================================================================== */
function sccTarjan(n, edges) {
  const adj = buildAdj(n, edges);
  const idx = new Array(n).fill(-1);
  const low = new Array(n).fill(0);
  const onStack = new Array(n).fill(false);
  const comp = new Array(n).fill(-1);
  const stk = [];
  let counter = 0, count = 0;
  for (let s = 0; s < n; s++) {
    if (idx[s] !== -1) continue;
    const frame = [[s, 0]];
    while (frame.length) {
      const top = frame[frame.length - 1];
      const v = top[0];
      if (top[1] === 0) { idx[v] = low[v] = counter++; stk.push(v); onStack[v] = true; }
      let advanced = false;
      while (top[1] < adj[v].length) {
        const w = adj[v][top[1]++];
        if (idx[w] === -1) { frame.push([w, 0]); advanced = true; break; }
        else if (onStack[w] && idx[w] < low[v]) low[v] = idx[w];
      }
      if (advanced) continue;
      frame.pop();
      if (low[v] === idx[v]) {
        let minId = v;
        const members = [];
        for (;;) {
          const w = stk.pop();
          onStack[w] = false;
          members.push(w);
          if (w < minId) minId = w;
          if (w === v) break;
        }
        for (const w of members) comp[w] = minId;
        count++;
      }
      if (frame.length) {
        const p = frame[frame.length - 1][0];
        if (low[v] < low[p]) low[p] = low[v];
      }
    }
  }
  return { comp, count };
}

/* ============================================================================
 * O13 三角计数（无向）：度排序定向法（u→v 当 deg(u)<deg(v) 或相等且 u<v），
 * 对每个 u 检查出邻居对是否邻接。返回 {count, ops}（ops=配对检查次数）。
 * ========================================================================== */
function countTriangles(n, edges) {
  const eset = new Set();
  const deg = new Array(n).fill(0);
  for (const [u, v] of edges) {
    if (u === v) continue;
    eset.add(ukey(u, v, n));
    deg[u]++; deg[v]++;
  }
  const out = Array.from({ length: n }, () => []);
  for (const key of eset) {
    const u = Math.floor(key / n), v = key % n;
    if (deg[u] < deg[v] || (deg[u] === deg[v] && u < v)) out[u].push(v);
    else out[v].push(u);
  }
  let count = 0, ops = 0;
  for (let u = 0; u < n; u++) {
    const ou = out[u];
    for (let i = 0; i < ou.length; i++) {
      for (let j = i + 1; j < ou.length; j++) {
        ops++;
        if (eset.has(ukey(ou[i], ou[j], n))) count++;
      }
    }
  }
  return { count, ops };
}

/* ============================================================================
 * O15 中心性：度中心性（归一）+ 接近中心性（reach/Σdist）+ 介数中心性（Brandes）
 * ========================================================================== */
function centrality(n, edges) {
  const sets = Array.from({ length: n }, () => new Set());
  for (const [u, v] of edges) { if (u !== v) { sets[u].add(v); sets[v].add(u); } }
  const nbr = sets.map((s) => Array.from(s).sort((a, b) => a - b));
  const degree = nbr.map((a) => a.length / Math.max(1, n - 1));
  const closeness = new Array(n).fill(0);
  const betweenness = new Array(n).fill(0);
  for (let s = 0; s < n; s++) {
    // BFS：dist / σ(最短路条数) / 前驱
    const dist = new Array(n).fill(-1);
    const sigma = new Array(n).fill(0);
    const preds = Array.from({ length: n }, () => []);
    const order = [];
    dist[s] = 0; sigma[s] = 1;
    const q = [s];
    for (let qi = 0; qi < q.length; qi++) {
      const v = q[qi];
      order.push(v);
      for (const w of nbr[v]) {
        if (dist[w] === -1) { dist[w] = dist[v] + 1; q.push(w); }
        if (dist[w] === dist[v] + 1) { sigma[w] += sigma[v]; preds[w].push(v); }
      }
    }
    let sumD = 0, reach = 0;
    for (let v = 0; v < n; v++) if (dist[v] >= 0 && v !== s) { sumD += dist[v]; reach++; }
    closeness[s] = sumD > 0 ? reach / sumD : 0;
    // Brandes 依赖累积（逆 BFS 序）
    const delta = new Array(n).fill(0);
    for (let oi = order.length - 1; oi >= 0; oi--) {
      const v = order[oi];
      for (const p of preds[v]) delta[p] += sigma[p] / sigma[v] * (1 + delta[v]);
      if (v !== s) betweenness[v] += delta[v];
    }
  }
  for (let v = 0; v < n; v++) betweenness[v] /= 2;   // 无向图每对计两次
  return { degree, closeness, betweenness };
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
    hash: computeHash(hashSrc)
  };
}

/* ============================================================================
 * selfCheck — 8 条不变量（INV-GA-001..008，独立手搓黄金参照 + 反空洞证伪）
 * ========================================================================== */
function seqEq(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}
function maxAbs(a, b) {
  let d = 0;
  for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i]));
  return d;
}

/* —— 独立手搓黄金参照（不调用本模块导出算法） —— */
function goldenDFSRecursive(adj, s) {           // 递归 DFS 前序（邻居升序）
  const visited = new Array(adj.length).fill(false);
  const order = [];
  (function go(u) {
    visited[u] = true;
    order.push(u);
    for (const v of adj[u]) if (!visited[v]) go(v);
  })(s);
  return order;
}
function goldenBFS(adj, s) {                    // BFS 序（邻居升序）
  const visited = new Array(adj.length).fill(false);
  const order = [];
  const q = [s];
  visited[s] = true;
  for (let qi = 0; qi < q.length; qi++) {
    const u = q[qi];
    order.push(u);
    for (const v of adj[u]) if (!visited[v]) { visited[v] = true; q.push(v); }
  }
  return order;
}
function goldenDijkstra(adj, coords, s, t) {    // O(n²) 选择式 Dijkstra，t 出堆即停
  const n = adj.length;
  const dist = new Array(n).fill(Infinity);
  const done = new Array(n).fill(false);
  dist[s] = 0;
  let explored = 0;
  for (;;) {
    let u = -1;
    for (let v = 0; v < n; v++) if (!done[v] && (u === -1 || dist[v] < dist[u])) u = v;
    if (u === -1 || !isFinite(dist[u])) break;
    done[u] = true;
    explored++;
    if (u === t) break;
    for (const v of adj[u]) {
      const nd = dist[u] + euclid(coords[u], coords[v]);
      if (nd < dist[v]) dist[v] = nd;
    }
  }
  return { dist: dist[t], explored };
}
function goldenPageRankLong(adj, d, iters) {    // 长迭代固定轮数（无早停）
  const n = adj.length;
  const outdeg = adj.map((a) => a.length);
  let r = new Array(n).fill(1 / n);
  for (let it = 0; it < iters; it++) {
    let dangling = 0;
    for (let v = 0; v < n; v++) if (outdeg[v] === 0) dangling += r[v];
    const base = (1 - d) / n + d * dangling / n;
    const rn = new Array(n).fill(base);
    for (let u = 0; u < n; u++) {
      if (outdeg[u] === 0) continue;
      const share = d * r[u] / outdeg[u];
      for (const v of adj[u]) rn[v] += share;
    }
    let ssum = 0;
    for (let v = 0; v < n; v++) ssum += rn[v];
    for (let v = 0; v < n; v++) rn[v] /= ssum;
    r = rn;
  }
  return r;
}
function goldenUnionFindCC(n, edges) {          // 手搓并查集（根恒为分量最小 id）
  const parent = Array.from({ length: n }, (_, i) => i);
  function find(x) {
    while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  }
  for (const [u, v] of edges) {
    const ru = find(u), rv = find(v);
    if (ru === rv) continue;
    if (ru < rv) parent[rv] = ru; else parent[ru] = rv;
  }
  const comp = new Array(n);
  let count = 0;
  const rootCount = new Map();
  for (let v = 0; v < n; v++) {
    const r = find(v);
    if (!rootCount.has(r)) rootCount.set(r, count++);
    comp[v] = r;
  }
  return { comp, count };
}
function goldenKosaraju(n, edges) {             // 手搓 Kosaraju（两次 DFS）
  const adj = buildAdj(n, edges);
  const radj = Array.from({ length: n }, () => []);
  for (const [u, v] of edges) radj[v].push(u);
  for (const a of radj) a.sort((x, y) => x - y);
  const visited = new Array(n).fill(false);
  const finish = [];
  for (let s = 0; s < n; s++) {
    if (visited[s]) continue;
    const stk = [[s, 0]];
    visited[s] = true;
    while (stk.length) {
      const top = stk[stk.length - 1];
      const v = top[0];
      if (top[1] < adj[v].length) {
        const w = adj[v][top[1]++];
        if (!visited[w]) { visited[w] = true; stk.push([w, 0]); }
      } else { finish.push(v); stk.pop(); }
    }
  }
  const visited2 = new Array(n).fill(false);
  const comp = new Array(n).fill(-1);
  let count = 0;
  for (let i = finish.length - 1; i >= 0; i--) {
    const s = finish[i];
    if (visited2[s]) continue;
    const members = [];
    const stk = [s];
    visited2[s] = true;
    while (stk.length) {
      const v = stk.pop();
      members.push(v);
      for (const w of radj[v]) if (!visited2[w]) { visited2[w] = true; stk.push(w); }
    }
    let minId = members[0];
    for (const m of members) if (m < minId) minId = m;
    for (const m of members) comp[m] = minId;
    count++;
  }
  return { comp, count };
}
function goldenTopoMaxFirst(n, edges) {         // 手搓合法拓扑序：Kahn 变体，每次取最大 id 的入度零节点
  const adj = Array.from({ length: n }, () => []);
  const indeg = new Array(n).fill(0);
  for (const [u, v] of edges) { adj[u].push(v); indeg[v]++; }
  const removed = new Array(n).fill(false);
  const order = [];
  for (let it = 0; it < n; it++) {
    let pick = -1;
    for (let v = n - 1; v >= 0; v--) if (!removed[v] && indeg[v] === 0) { pick = v; break; }
    if (pick === -1) break;
    removed[pick] = true;
    order.push(pick);
    for (const w of adj[pick]) indeg[w]--;
  }
  return order;
}
function goldenTrianglesTripleLoop(n, edges) {  // 手搓三重循环
  const eset = new Set();
  for (const [u, v] of edges) { if (u !== v) eset.add(ukey(u, v, n)); }
  let c = 0;
  for (let u = 0; u < n; u++) {
    for (let v = u + 1; v < n; v++) {
      if (!eset.has(u * n + v)) continue;
      for (let w = v + 1; w < n; w++) if (eset.has(u * n + w) && eset.has(v * n + w)) c++;
    }
  }
  return c;
}
function goldenAllPairsBFS(n, nbr) {            // 全源 BFS：dist/σ（用于逐对介数黄金参照）
  const dist = [], sigma = [];
  for (let s = 0; s < n; s++) {
    const ds = new Array(n).fill(-1), ss = new Array(n).fill(0);
    ds[s] = 0; ss[s] = 1;
    const q = [s];
    for (let qi = 0; qi < q.length; qi++) {
      const v = q[qi];
      for (const w of nbr[v]) {
        if (ds[w] === -1) { ds[w] = ds[v] + 1; q.push(w); }
        if (ds[w] === ds[v] + 1) ss[w] += ss[v];
      }
    }
    dist.push(ds); sigma.push(ss);
  }
  return { dist, sigma };
}
function goldenPairwiseBetweenness(n, nbr) {    // 手搓逐对 BFS 路径计数（Brandes 之外的独立口径）
  const { dist, sigma } = goldenAllPairsBFS(n, nbr);
  const btw = new Array(n).fill(0);
  for (let s = 0; s < n; s++) {
    for (let t = 0; t < n; t++) {
      if (t === s || dist[s][t] < 0) continue;
      for (let v = 0; v < n; v++) {
        if (v === s || v === t) continue;
        if (dist[s][v] >= 0 && dist[t][v] >= 0 && dist[s][v] + dist[t][v] === dist[s][t]) {
          btw[v] += sigma[s][v] * sigma[t][v] / sigma[s][t];
        }
      }
    }
  }
  for (let v = 0; v < n; v++) btw[v] /= 2;
  return btw;
}
function mstCutCertificateOk(n, edges, tree) {  // 割性质证书：删树边 e，跨割最小非树边权 ≥ w(e)
  const tkey = new Set(tree.map((e) => ukey(e[0], e[1], n)));
  const tadj = Array.from({ length: n }, () => []);
  for (const e of tree) { tadj[e[0]].push([e[1], e[2]]); tadj[e[1]].push([e[0], e[2]]); }
  for (const e of tree) {
    const ek = ukey(e[0], e[1], n);
    const part = new Array(n).fill(false);
    const q = [e[0]];
    part[e[0]] = true;
    while (q.length) {
      const v = q.pop();
      for (const [w] of tadj[v]) {
        if (ukey(v, w, n) === ek || part[w]) continue;
        part[w] = true;
        q.push(w);
      }
    }
    let minCross = Infinity;
    for (const f of edges) {
      if (tkey.has(ukey(f[0], f[1], n))) continue;
      if (part[f[0]] !== part[f[1]] && f[2] < minCross) minCross = f[2];
    }
    if (minCross < e[2]) return false;
  }
  return true;
}
function uniqueSPGraph(n, extraList, seed0) {   // 找全对最短路唯一的连通图（保证介数逐位精确）
  for (const extra of extraList) {              // 附加边从多到少退化（extra=0 纯随机树必唯一）
    for (let sd = seed0; sd < seed0 + 200; sd++) {
      const rng = makeRng(sd);
      const edges = [];
      for (let v = 1; v < n; v++) edges.push([Math.floor(rng() * v), v]);   // 随机树
      for (let i = 0; i < extra; i++) {
        const u = Math.floor(rng() * n), v = Math.floor(rng() * n);
        if (u !== v) edges.push([Math.min(u, v), Math.max(u, v)]);
      }
      // 去重
      const seenE = new Set();
      const dedup = [];
      for (const [u, v] of edges) { const k = ukey(u, v, n); if (!seenE.has(k)) { seenE.add(k); dedup.push([u, v]); } }
      const sets = Array.from({ length: n }, () => new Set());
      for (const [u, v] of dedup) { sets[u].add(v); sets[v].add(u); }
      if (sets.some((s) => s.size === 0)) continue;
      const nbr = sets.map((s) => Array.from(s).sort((a, b) => a - b));
      const { sigma } = goldenAllPairsBFS(n, nbr);
      let unique = true;
      outer: for (let s = 0; s < n; s++) for (let t = 0; t < n; t++) if (s !== t && sigma[s][t] !== 1) { unique = false; break outer; }
      if (unique) return { n, edges: dedup, nbr, extra };
    }
  }
  return null;
}

function selfCheck() {
  const cases = [];
  const invariants = [];
  const SEED = 0x9E3779B9;

  /* ---------- INV-GA-001 DFS 逐位一致 + 排列 + ≠BFS ---------- */
  {
    let bitwise = true, permOk = true;
    for (const sd of [SEED, 0xA11CE, 0xB0B57]) {
      const g = randomDigraph(24, 60, sd);
      for (let s = 0; s < g.n; s += 5) {
        const got = dfsOrder(g.adj, s);
        const ref = goldenDFSRecursive(g.adj, s);
        if (!seqEq(got, ref)) bitwise = false;
        // 排列校验：无重复且都在可达集
        const bfs = goldenBFS(g.adj, s);
        const set = new Set(bfs);
        if (new Set(got).size !== got.length) permOk = false;
        for (const v of got) if (!set.has(v)) permOk = false;
      }
    }
    let dfsNeBfs = false;
    for (let sd = SEED; sd < SEED + 10 && !dfsNeBfs; sd++) {
      const g = randomDigraph(24, 60, sd);
      for (let s = 0; s < g.n; s++) {
        if (!seqEq(dfsOrder(g.adj, s), goldenBFS(g.adj, s))) { dfsNeBfs = true; break; }
      }
    }
    const ok = bitwise && permOk && dfsNeBfs;
    cases.push(caseOf('INV-GA-001', 'DFS 显式栈 ≡ 手搓递归前序（EXACT）+ 排列 + ≠BFS 序', ok, bitwise && permOk ? 0 : 1,
      '逐位一致=' + bitwise + ' 排列合法=' + permOk + ' DFS≠BFS=' + dfsNeBfs, TOL.EXACT));
    invariants.push({ id: 'INV-GA-001', name: 'DFS 确定序 + 遍历混淆证伪', ok, expr: '迭代栈≡递归前序(EXACT) 且 输出为可达集排列 且 DFS序≠BFS序', tol: TOL.EXACT, worst: ok ? 0 : 1, detail: cases[cases.length - 1].detail });
  }

  /* ---------- INV-GA-002 A*（黄金参照 + 探索计数 + 次优边界） ---------- */
  {
    // (a) h=0 ≡ Dijkstra（多 seed 全对最坏）
    let worst = 0;
    for (let sd = SEED; sd < SEED + 8; sd++) {
      const g = geometricGraph(30, 3, sd);
      for (let s = 0; s < 8; s++) {
        for (let t = 0; t < g.n; t++) {
          if (t === s) continue;
          const dg = goldenDijkstra(g.adj, g.coords, s, t);
          if (!isFinite(dg.dist)) continue;
          const a0 = aStar(g.adj, g.coords, s, t, { hScale: 0 });
          if (a0.found) worst = Math.max(worst, Math.abs(a0.dist - dg.dist));
        }
      }
    }
    // (b) 可采纳 h>0 探索节点严格更少（确定性扫描找到满足实例）
    let strict = null;
    for (let sd = SEED; sd < SEED + 60 && !strict; sd++) {
      const g = geometricGraph(30, 3, sd);
      const t = g.n - 1;
      const dg = goldenDijkstra(g.adj, g.coords, 0, t);
      if (!isFinite(dg.dist)) continue;
      const a1 = aStar(g.adj, g.coords, 0, t, { hScale: 1 });
      const a0 = aStar(g.adj, g.coords, 0, t, { hScale: 0 });
      if (a1.found && Math.abs(a1.dist - dg.dist) <= 1e-9 && a1.explored < dg.explored) {
        strict = { sd, a1: a1.explored, dj: dg.explored };
      }
    }
    // (c) 路径合法性（端点/连边/权和）
    const g2 = geometricGraph(30, 3, SEED);
    const t2 = g2.n - 1;
    const a2 = aStar(g2.adj, g2.coords, 0, t2, { hScale: 1 });
    let pathOk = a2.found && a2.path[0] === 0 && a2.path[a2.path.length - 1] === t2;
    let wsum = 0;
    if (pathOk) {
      for (let i = 0; i + 1 < a2.path.length; i++) {
        const u = a2.path[i], v = a2.path[i + 1];
        if (!g2.adj[u].includes(v)) { pathOk = false; break; }
        wsum += euclid(g2.coords[u], g2.coords[v]);
      }
      if (Math.abs(wsum - a2.dist) > 1e-9) pathOk = false;
    }
    // (d) h×10 不可采纳：恒 ≥ 最优，且存在严格次优实例（证伪边界）
    let allGe = true, suboptFound = false, suboptInfo = '';
    for (let sd = SEED; sd < SEED + 60; sd++) {
      const g = geometricGraph(30, 2, sd);   // 稀疏 k=2：链式结构易暴露次优
      const t = g.n - 1;
      const dg = goldenDijkstra(g.adj, g.coords, 0, t);
      if (!isFinite(dg.dist)) continue;
      const a10 = aStar(g.adj, g.coords, 0, t, { hScale: 10 });
      if (!a10.found || a10.dist < dg.dist - 1e-12) allGe = false;
      if (a10.found && a10.dist > dg.dist + 1e-9) {
        suboptFound = true;
        suboptInfo = 'seed=0x' + sd.toString(16) + ' sub=' + a10.dist.toFixed(4) + ' opt=' + dg.dist.toFixed(4);
        break;
      }
    }
    const ok = (worst <= TOL.TIGHT) && !!strict && pathOk && allGe && suboptFound;
    cases.push(caseOf('INV-GA-002', 'A*: h=0≡Dijkstra(TIGHT) + 可采纳探索更少 + ×10 恒≥最优且可次优 + 路径合法', ok, worst,
      'h=0 max|Δ|=' + worst.toExponential(2) + ' 探索=' + (strict ? strict.a1 + '<' + strict.dj : '未找到') + ' 路径合法=' + pathOk + ' ×10≥最优=' + allGe + ' 严格次优=' + suboptFound + (suboptInfo ? '(' + suboptInfo + ')' : ''), TOL.TIGHT));
    invariants.push({ id: 'INV-GA-002', name: 'A* 最短路 + 启发边界证伪', ok, expr: 'h=0≡Dijkstra(≤1e-15)；h=1 explored<Dijkstra；h=10 dist≥opt 且存在严格次优；路径合法', tol: TOL.TIGHT, worst, detail: cases[cases.length - 1].detail });
  }

  /* ---------- INV-GA-003 PageRank ---------- */
  {
    const g = randomDigraph(20, 60, SEED);
    const pr = pageRank(g.adj, { d: 0.85, tol: 1e-13, maxIter: 5000 });
    const ssum = pr.ranks.reduce((a, b) => a + b, 0);
    const golden = goldenPageRankLong(g.adj, 0.85, 30000);
    const dev = maxAbs(pr.ranks, golden);
    const uniformDev = Math.max(...pr.ranks.map((r) => Math.abs(r - 1 / g.n)));
    const ok = (Math.abs(ssum - 1) <= TOL.TIGHT) && (pr.residual < 1e-10) && (dev <= TOL.SOLVE) && (uniformDev > 1e-3);
    cases.push(caseOf('INV-GA-003', 'PageRank: Σ=1 + 残差<1e-10 + ≡长迭代黄金参照 + ≠均匀', ok, Math.max(Math.abs(ssum - 1), dev),
      'Σ−1=' + (ssum - 1).toExponential(2) + ' residual=' + pr.residual.toExponential(2) + ' vs手搓长迭代=' + dev.toExponential(2) + ' iters=' + pr.iters + ' max|rank−1/n|=' + uniformDev.toExponential(2) + '(>1e-3 证伪平凡)', TOL.SOLVE));
    invariants.push({ id: 'INV-GA-003', name: 'PageRank 幂迭代 + 平凡均匀证伪', ok, expr: '|Σ−1|≤1e-15；residual<1e-10；max|PR−golden|≤1e-10；max|rank−1/n|>1e-3', tol: TOL.SOLVE, worst: Math.max(Math.abs(ssum - 1), dev), detail: cases[cases.length - 1].detail });
  }

  /* ---------- INV-GA-004 连通分量 / SCC 双口径 ---------- */
  {
    let ccOk = true, sccOk = true;
    for (const sd of [SEED, 0xC0FFEE, 0x1234ABCD]) {
      const gu = randomUndirected(30, 45, 50, sd);
      const cc = connectedComponents(gu.n, gu.edges.map((e) => [e[0], e[1]]));
      const ref = goldenUnionFindCC(gu.n, gu.edges.map((e) => [e[0], e[1]]));
      if (!seqEq(cc.comp, ref.comp) || cc.count !== ref.count) ccOk = false;
      const gd = randomDigraph(24, 50, sd);
      const sc = sccTarjan(gd.n, gd.edges);
      const ksr = goldenKosaraju(gd.n, gd.edges);
      if (!seqEq(sc.comp, ksr.comp) || sc.count !== ksr.count) sccOk = false;
    }
    // 证伪混淆：有向路径图 SCC 数=4 ≠ 无向分量数=1
    const pathEdges = [[0, 1], [1, 2], [2, 3]];
    const sccPath = sccTarjan(4, pathEdges);
    const ccPath = connectedComponents(4, pathEdges);
    const confused = sccPath.count === 4 && ccPath.count === 1 && sccPath.count !== ccPath.count;
    const ok = ccOk && sccOk && confused;
    cases.push(caseOf('INV-GA-004', 'CC≡手搓Union-Find(EXACT) + SCC≡手搓Kosaraju(EXACT) + SCC数≠CC数', ok, ccOk && sccOk ? 0 : 1,
      'CC 3种子逐位=' + ccOk + ' SCC 3种子逐位=' + sccOk + ' 路径图 SCC=' + sccPath.count + ' vs CC=' + ccPath.count, TOL.EXACT));
    invariants.push({ id: 'INV-GA-004', name: '连通性双口径 + 有向/无向混淆证伪', ok, expr: 'CC≡Union-Find(EXACT)；SCC≡Kosaraju(EXACT)；同图 SCC数≠CC数', tol: TOL.EXACT, worst: ok ? 0 : 1, detail: cases[cases.length - 1].detail });
  }

  /* ---------- INV-GA-005 拓扑排序 ---------- */
  {
    const g = randomDAG(20, 0.15, SEED);
    const ord = topoSort(g.n, g.edges);
    const pos = new Array(g.n);
    for (let i = 0; i < ord.length; i++) pos[ord[i]] = i;
    let allEdgesOk = g.edges.every(([u, v]) => pos[u] < pos[v]);
    // 非唯一：合法降序实现序（可不同，但均须合法）
    const ord2 = goldenTopoMaxFirst(g.n, g.edges);
    const pos2 = new Array(g.n);
    for (let i = 0; i < ord2.length; i++) pos2[ord2[i]] = i;
    const allEdgesOk2 = g.edges.every(([u, v]) => pos2[u] < pos2[v]);
    const differDemo = (() => {    // 固定小图证明合法序非唯一
      const o1 = topoSort(3, [[0, 2], [1, 2]]);
      const o2 = goldenTopoMaxFirst(3, [[0, 2], [1, 2]]);
      return seqEq(o1, [0, 1, 2]) && seqEq(o2, [1, 0, 2]) && !seqEq(o1, o2);
    })();
    // 有环必抛错（含自环）
    let cycThrew = false, selfThrew = false;
    try { topoSort(3, [[0, 1], [1, 2], [2, 0]]); } catch (e) { cycThrew = true; }
    try { topoSort(2, [[0, 0]]); } catch (e) { selfThrew = true; }
    const ok = allEdgesOk && allEdgesOk2 && differDemo && cycThrew && selfThrew && ord.length === g.n;
    cases.push(caseOf('INV-GA-005', 'Kahn 拓扑: 全边 pos[u]<pos[v](EXACT) + 双合法序非唯一 + 有环必抛错', ok, allEdgesOk && allEdgesOk2 ? 0 : 1,
      '升序全边合法=' + allEdgesOk + ' 降序全边合法=' + allEdgesOk2 + ' 序非唯一演示=' + differDemo + ' 环抛错=' + cycThrew + ' 自环抛错=' + selfThrew, TOL.EXACT));
    invariants.push({ id: 'INV-GA-005', name: '拓扑排序合法性与环检测', ok, expr: '∀(u,v): pos[u]<pos[v](EXACT)；两种合法实现序均合法且可不同；有环/自环抛错', tol: TOL.EXACT, worst: ok ? 0 : 1, detail: cases[cases.length - 1].detail });
  }

  /* ---------- INV-GA-006 MST 双实现 + 割性质证书 ---------- */
  {
    let worst = 0, structOk = true, cutOk = true;
    for (const sd of [SEED, 0xBEEF, 0x5EED5]) {
      const g = randomUndirected(24, 70, 99, sd);
      const k = kruskalMST(g.n, g.edges);
      const p = primMST(g.n, g.adjW);
      worst = Math.max(worst, Math.abs(k.total - p.total));
      if (k.tree.length !== g.n - 1) structOk = false;
      // 无环 + Spanning（并查集重扫树边）
      const parent = Array.from({ length: g.n }, (_, i) => i);
      const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
      let comps = g.n;
      for (const [u, v] of k.tree) { const ru = find(u), rv = find(v); if (ru === rv) structOk = false; else { parent[ru] = rv; comps--; } }
      if (comps !== 1) structOk = false;
      if (!mstCutCertificateOk(g.n, g.edges, k.tree)) cutOk = false;
    }
    const ok = worst <= TOL.TIGHT && structOk && cutOk;
    cases.push(caseOf('INV-GA-006', 'MST: Kruskal≡Prim 总权 + n−1 边无环Spanning + 割性质证书（证伪平凡）', ok, worst,
      '3种子 |K−P|max=' + worst.toExponential(2) + ' 结构(n−1/无环/Spanning)=' + structOk + ' 全树边割性质=' + cutOk, TOL.TIGHT));
    invariants.push({ id: 'INV-GA-006', name: 'MST 双实现 + 割性质最优性证书', ok, expr: '|Kruskal−Prim|≤1e-15；n−1 边无环Spanning；∀树边 e: 跨割最小非树边权≥w(e)', tol: TOL.TIGHT, worst, detail: cases[cases.length - 1].detail });
  }

  /* ---------- INV-GA-007 三角计数 ---------- */
  {
    let exact = true, opsOk = true, lastOps = 0, lastCount = 0;
    for (const sd of [SEED, 0xA5A5, 0x7E1E]) {
      const g = randomUndirected(20, 40, 0, sd).edges.map((e) => [e[0], e[1]]);
      const r = countTriangles(20, g);
      const ref = goldenTrianglesTripleLoop(20, g);
      if (r.count !== ref) exact = false;
      if (r.ops >= 20 * 20 * 20 / 10) opsOk = false;   // ops ≪ n³
      lastOps = r.ops; lastCount = r.count;
    }
    const ok = exact && opsOk;
    cases.push(caseOf('INV-GA-007', '三角计数: 度排序定向法 ≡ 手搓三重循环(EXACT) + ops≪n³', ok, exact ? 0 : 1,
      '3种子计数逐位=' + exact + ' 最后样本 count=' + lastCount + ' ops=' + lastOps + ' <' + (8000 / 10) + '=' + opsOk, TOL.EXACT));
    invariants.push({ id: 'INV-GA-007', name: '三角计数正确性 + 复杂度下界性能真', ok, expr: 'count≡三重循环(EXACT)；ops < n³/10', tol: TOL.EXACT, worst: exact && opsOk ? 0 : 1, detail: cases[cases.length - 1].detail });
  }

  /* ---------- INV-GA-008 中心性 + 确定性 ---------- */
  {
    // (a) Brandes ≡ 手搓逐对 BFS（唯一最短路图上逐位精确）
    const ug = uniqueSPGraph(14, [6, 4, 2, 1, 0], 0x5EED);
    let bWorst = 0, degWorst = 0;
    if (ug) {
      const c = centrality(ug.n, ug.edges);
      const ref = goldenPairwiseBetweenness(ug.n, ug.nbr);
      bWorst = maxAbs(c.betweenness, ref);
      // 度中心性 ≡ 手搓计数
      const degCount = ug.nbr.map((a) => a.length / (ug.n - 1));
      degWorst = maxAbs(c.degree, degCount);
    }
    // (b) 同 seed 全流水线两次逐位相等；异 seed 必分歧
    const pipeline = (sd) => {
      const gu = randomUndirected(16, 28, 50, sd);
      const gd = randomDigraph(12, 24, sd);
      const geo = geometricGraph(14, 3, sd);
      const a = aStar(geo.adj, geo.coords, 0, geo.n - 1, { hScale: 1 });
      const m = kruskalMST(16, gu.edges);
      const s = sccTarjan(16, randomDigraph(16, 30, sd).edges);
      const c = centrality(16, gu.edges.map((e) => [e[0], e[1]]));
      const pr = pageRank(gd.adj, { d: 0.85, tol: 1e-12, maxIter: 500 });
      const dfs = dfsOrder(buildAdj(16, randomDigraph(16, 30, sd).edges), 0);
      return JSON.stringify([dfs, pr.ranks, a.dist, a.path, m.total, s.comp, c.betweenness, c.closeness]);
    };
    const r1 = pipeline(0xBEEF), r2 = pipeline(0xBEEF), r3 = pipeline(0xC0FFEE);
    const identical = r1 === r2;
    const diverged = r1 !== r3;
    const ok = !!ug && bWorst <= TOL.TIGHT && degWorst <= TOL.EXACT && identical && diverged;
    cases.push(caseOf('INV-GA-008', 'Brandes≡逐对BFS黄金参照(TIGHT) + 度≡手搓 + 同seed逐位/异seed分歧', ok, bWorst,
      '介数max|Δ|=' + bWorst.toExponential(2) + ' 度max|Δ|=' + degWorst.toExponential(2) + ' 同seed逐位=' + identical + ' 异seed分歧=' + diverged, TOL.TIGHT));
    invariants.push({ id: 'INV-GA-008', name: '中心性黄金参照 + 确定性/分歧', ok, expr: 'Brandes≡逐对BFS(≤1e-15)；degree≡手搓(EXACT)；同seed逐位相等；异seed分歧', tol: TOL.TIGHT, worst: Math.max(bWorst, degWorst), detail: cases[cases.length - 1].detail });
  }

  return buildReport('graphalg-forge', '1.0.0', 'L7', cases, invariants, '0x9E3779B9');
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
  name: 'graphalg-forge', version: '1.0.0', layer: 'L7', TOL,
  makeRng, sha256Hex,
  randomDigraph, randomUndirected, randomDAG, geometricGraph, buildAdj,
  dfsOrder, aStar, pageRank, connectedComponents,
  kruskalMST, primMST, topoSort, sccTarjan, countTriangles, centrality,
  selfCheck, formatTable
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeGraphAlg = API;

if (typeof process !== 'undefined' && process.argv[1] && /graphalg-forge\.js$/.test(process.argv[1]) && (process.argv.includes('--self-check') || process.argv.includes('--json'))) {
  const r = selfCheck();
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : formatTable(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
