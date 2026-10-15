/**
 * @forge-module graph-forge
 * @layer L5
 * @version 1.0.0
 * @brief     图算法内核：边表建图、BFS 最短路径（跳数）、Dijkstra 最短路（非负权）、路径重建、三角不等式与确定性验证。全部附可交叉验证不变量（手算对照、独立重实现、暴力枚举）。
 * @reuse     NetworkX|https://github.com/networkx/networkx|BSD-3|B|BFS/Dijkstra/最短路径参照
 * @reuse     Cormen et al. Introduction to Algorithms|https://mitpress.mit.edu/9780262046305/|公有域|D|Dijkstra/BFS 正确性
 * @beyond    纯 JS 单文件零构建；Dijkstra 双独立实现互验；含 0 权边的暴力全简单路径枚举做黄金参照；三角不等式/确定性等结构性不变量可证。
 * @invariant INV-GRAPH-001|BFS 在无向无权图上的跳数距离与手算真值一致（精确整数）|0|手算对照
 * @invariant INV-GRAPH-002|同图 Dijkstra(权=1) == BFS 跳数 ≤ 1e-12|1e-12|BFS/Dijkstra 等价
 * @invariant INV-GRAPH-003|Dijkstra 主实现 vs 独立第二实现距离一致 ≤ 1e-12|1e-12|参考重实现
 * @invariant INV-GRAPH-004|BFS 父链重建路径长度 == 报告距离（精确）|0|路径重建
 * @invariant INV-GRAPH-005|不连通分量节点距离返回 Infinity（精确）|0|连通性
 * @invariant INV-GRAPH-006|含 0 权边图：Dijkstra == 暴力枚举全简单路径最小权 ≤ 1e-12|1e-12|暴力黄金参照
 * @invariant INV-GRAPH-007|Dijkstra 同图两次运行距离完全一致（精确）|0|确定性
 * @invariant INV-GRAPH-008|三角不等式 dist(u,w) ≤ dist(u,v)+dist(v,w) ≤ 1e-12|1e-12|度量性质
 * @seed      0x9E3779B9
 */
'use strict';
const { makeRng, TOL } = require('./core-numeric.js');
const { caseResult, buildReport } = require('./core-verify.js');

/* ---------- 建图 ---------- */

// edges: [[from, to, weight], ...]；opts.directed 默认 false（无向则双向加边）
function buildGraph(edges, opts) {
  opts = opts || {};
  const directed = !!opts.directed;
  const idToIdx = new Map();
  const nodes = [];
  function indexOf(id) {
    let i = idToIdx.get(id);
    if (i === undefined) { i = nodes.length; nodes.push(id); idToIdx.set(id, i); }
    return i;
  }
  const adj = [];
  for (const e of edges) {
    const a = indexOf(e[0]);
    const b = indexOf(e[1]);
    const w = e[2];
    if (!adj[a]) adj[a] = [];
    adj[a].push({ to: b, w: w });
    if (!directed) {
      if (!adj[b]) adj[b] = [];
      adj[b].push({ to: a, w: w });
    }
  }
  for (let i = 0; i < nodes.length; i++) if (!adj[i]) adj[i] = [];
  return { nodes, idToIdx, adj, n: nodes.length };
}

/* ---------- BFS（按跳数，忽略边权） ---------- */

function bfsShortest(graph, src) {
  const srcIdx = graph.idToIdx.get(src);
  if (srcIdx === undefined) throw new TypeError('bfsShortest: 源点不存在');
  const n = graph.n;
  const dist = new Array(n).fill(Infinity);
  const parent = new Array(n).fill(null);
  const visited = new Array(n).fill(false);
  const queue = [srcIdx];
  visited[srcIdx] = true;
  dist[srcIdx] = 0;
  let head = 0;
  while (head < queue.length) {
    const u = queue[head++];
    for (const e of graph.adj[u]) {
      const v = e.to;
      if (!visited[v]) {
        visited[v] = true;
        dist[v] = dist[u] + 1;
        parent[v] = u;
        queue.push(v);
      }
    }
  }
  const distMap = new Map(), parentMap = new Map();
  for (let i = 0; i < n; i++) {
    distMap.set(graph.nodes[i], dist[i]);
    parentMap.set(graph.nodes[i], parent[i] === null ? null : graph.nodes[parent[i]]);
  }
  return { dist: distMap, parent: parentMap };
}

/* ---------- Dijkstra（主实现：O(V²) 提取最小） ---------- */

function dijkstra(graph, src) {
  const srcIdx = graph.idToIdx.get(src);
  if (srcIdx === undefined) throw new TypeError('dijkstra: 源点不存在');
  const n = graph.n;
  const dist = new Array(n).fill(Infinity);
  const parent = new Array(n).fill(null);
  const visited = new Array(n).fill(false);
  dist[srcIdx] = 0;
  for (let iter = 0; iter < n; iter++) {
    let u = -1, best = Infinity;
    for (let i = 0; i < n; i++) if (!visited[i] && dist[i] < best) { best = dist[i]; u = i; }
    if (u === -1) break;
    visited[u] = true;
    for (const e of graph.adj[u]) {
      const v = e.to, w = e.w;
      if (dist[u] + w < dist[v]) { dist[v] = dist[u] + w; parent[v] = u; }
    }
  }
  const distMap = new Map(), parentMap = new Map();
  for (let i = 0; i < n; i++) {
    distMap.set(graph.nodes[i], dist[i]);
    parentMap.set(graph.nodes[i], parent[i] === null ? null : graph.nodes[parent[i]]);
  }
  return { dist: distMap, parent: parentMap };
}

/* ---------- Dijkstra（第二独立实现：重复松弛，非负权下正确） ---------- */

function dijkstraRef(graph, src) {
  const srcIdx = graph.idToIdx.get(src);
  if (srcIdx === undefined) throw new TypeError('dijkstraRef: 源点不存在');
  const n = graph.n;
  const dist = new Array(n).fill(Infinity);
  const parent = new Array(n).fill(null);
  dist[srcIdx] = 0;
  for (let pass = 0; pass < n; pass++) {
    let changed = false;
    for (let u = 0; u < n; u++) {
      if (dist[u] === Infinity) continue;
      for (const e of graph.adj[u]) {
        const v = e.to, w = e.w;
        if (dist[u] + w < dist[v]) { dist[v] = dist[u] + w; parent[v] = u; changed = true; }
      }
    }
    if (!changed) break;
  }
  const distMap = new Map(), parentMap = new Map();
  for (let i = 0; i < n; i++) {
    distMap.set(graph.nodes[i], dist[i]);
    parentMap.set(graph.nodes[i], parent[i] === null ? null : graph.nodes[parent[i]]);
  }
  return { dist: distMap, parent: parentMap };
}

/* ---------- 路径重建 ---------- */

function reconstructPath(parents, src, t) {
  if (!parents.has(src) || !parents.has(t)) return [];
  if (src === t) return [src];
  const path = [];
  let cur = t;
  let guard = 0;
  while (cur !== null && cur !== undefined && guard <= parents.size + 1) {
    path.push(cur);
    if (cur === src) break;
    cur = parents.get(cur);
    guard++;
  }
  if (path[path.length - 1] !== src) return [];
  path.reverse();
  return path;
}

/* ---------- 暴力枚举全部简单路径（非负权下的最短 = 某简单路径） ---------- */

function bruteForceShortest(graph, src) {
  const srcIdx = graph.idToIdx.get(src);
  const dist = {};
  for (const nid of graph.nodes) dist[nid] = Infinity;
  dist[graph.nodes[srcIdx]] = 0;
  const visited = new Set();
  function dfs(u, acc) {
    const uid = graph.nodes[u];
    for (const e of graph.adj[u]) {
      const vid = graph.nodes[e.to];
      if (visited.has(vid)) continue;
      const nc = acc + e.w;
      if (nc < dist[vid]) { dist[vid] = nc; visited.add(vid); dfs(e.to, nc); visited.delete(vid); }
    }
  }
  visited.add(graph.nodes[srcIdx]);
  dfs(srcIdx, 0);
  return dist;
}

/* ============================================================================
 * selfCheck — 8 条不变量
 * ========================================================================== */
function selfCheck(opts) {
  opts = opts || {};
  const cases = [];
  const invariants = [];

  // INV-GRAPH-001 BFS 跳数 vs 手算真值（无向无权图，全权=1）
  {
    const g = buildGraph([[0, 1, 1], [1, 2, 1], [2, 3, 1], [0, 4, 1], [4, 5, 1], [5, 2, 1]], { directed: false });
    const { dist } = bfsShortest(g, 0);
    const truth = { 0: 0, 1: 1, 4: 1, 2: 2, 5: 2, 3: 3 };
    let maxDev = 0;
    for (const k in truth) maxDev = Math.max(maxDev, Math.abs(dist.get(Number(k)) - truth[k]));
    cases.push(caseResult('INV-GRAPH-001', 'BFS 跳数 vs 手算真值', maxDev, 0, 0));
  }
  // INV-GRAPH-002 同图 Dijkstra(权=1) == BFS
  {
    const g = buildGraph([[0, 1, 1], [1, 2, 1], [2, 3, 1], [0, 4, 1], [4, 5, 1], [5, 2, 1]], { directed: false });
    const bfs = bfsShortest(g, 0).dist;
    const dij = dijkstra(g, 0).dist;
    let maxDev = 0;
    for (const nid of g.nodes) maxDev = Math.max(maxDev, Math.abs(bfs.get(nid) - dij.get(nid)));
    cases.push(caseResult('INV-GRAPH-002', 'Dijkstra(权=1) == BFS', maxDev, 0, TOL.RESID));
  }
  // INV-GRAPH-003 Dijkstra 主实现 vs 独立重实现
  {
    const g = buildGraph([[0, 1, 2], [1, 2, 1], [2, 3, 3], [0, 3, 0], [1, 3, 4], [0, 4, 5]], { directed: false });
    const a = dijkstra(g, 0).dist;
    const b = dijkstraRef(g, 0).dist;
    let maxDev = 0;
    for (const nid of g.nodes) maxDev = Math.max(maxDev, Math.abs(a.get(nid) - b.get(nid)));
    cases.push(caseResult('INV-GRAPH-003', 'Dijkstra 主实现 vs 独立重实现', maxDev, 0, TOL.RESID));
  }
  // INV-GRAPH-004 BFS 父链重建长度 == 报告距离
  {
    const g = buildGraph([[0, 1, 1], [1, 2, 1], [2, 3, 1], [0, 4, 1], [4, 5, 1], [5, 2, 1]], { directed: false });
    const { dist, parent } = bfsShortest(g, 0);
    const pairs = [[0, 3], [0, 5], [0, 1], [0, 2]];
    let maxDev = 0;
    for (const [s, t] of pairs) {
      const p = reconstructPath(parent, s, t);
      const len = (p.length === 0 ? Infinity : p.length - 1);
      maxDev = Math.max(maxDev, Math.abs(len - dist.get(t)));
    }
    cases.push(caseResult('INV-GRAPH-004', 'BFS 父链重建长度 == 距离', maxDev, 0, 0));
  }
  // INV-GRAPH-005 不连通分量返回 Infinity
  {
    const g = buildGraph([[0, 1, 1], [1, 2, 1], [3, 4, 1]], { directed: false });
    const { dist } = bfsShortest(g, 0);
    const ok = (dist.get(3) === Infinity) && (dist.get(4) === Infinity) && (dist.get(0) === 0) && (dist.get(2) === 2);
    cases.push(caseResult('INV-GRAPH-005', '不连通分量距离=Infinity', ok ? 0 : 1, 0, 0));
  }
  // INV-GRAPH-006 含 0 权边：Dijkstra == 暴力枚举
  {
    const g = buildGraph([[0, 1, 2], [1, 2, 1], [2, 3, 3], [0, 3, 0], [1, 3, 4], [0, 4, 5]], { directed: false });
    const dij = dijkstra(g, 0).dist;
    const bf = bruteForceShortest(g, 0);
    let maxDev = 0;
    for (const nid of g.nodes) maxDev = Math.max(maxDev, Math.abs(dij.get(nid) - bf[nid]));
    cases.push(caseResult('INV-GRAPH-006', 'Dijkstra == 暴力全路径枚举（含 0 权边）', maxDev, 0, TOL.RESID));
  }
  // INV-GRAPH-007 Dijkstra 确定性（两次完全一致）
  {
    const g = buildGraph([[0, 1, 2], [1, 2, 1], [2, 3, 3], [0, 3, 0], [1, 3, 4], [0, 4, 5]], { directed: false });
    const a = dijkstra(g, 0).dist;
    const b = dijkstra(g, 0).dist;
    let identical = true;
    for (const nid of g.nodes) if (a.get(nid) !== b.get(nid)) { identical = false; break; }
    cases.push(caseResult('INV-GRAPH-007', 'Dijkstra 确定性（两次一致）', identical ? 0 : 1, 0, 0));
  }
  // INV-GRAPH-008 三角不等式
  {
    const g = buildGraph([[0, 1, 2], [1, 2, 1], [2, 3, 3], [0, 3, 0], [1, 3, 4], [0, 4, 5]], { directed: false });
    const nodes = g.nodes;
    const D = {};
    for (const s of nodes) { const d = dijkstra(g, s).dist; D[s] = d; }
    let worst = 0;
    for (const u of nodes) for (const v of nodes) for (const w of nodes) {
      const duw = D[u].get(w), duv = D[u].get(v), dvw = D[v].get(w);
      if (!isFinite(duw) || !isFinite(duv) || !isFinite(dvw)) continue;
      worst = Math.max(worst, duw - (duv + dvw));
    }
    cases.push(caseResult('INV-GRAPH-008', '三角不等式', worst, 0, TOL.RESID));
  }

  const started = Date.now();
  for (const id of ['INV-GRAPH-001', 'INV-GRAPH-002', 'INV-GRAPH-003', 'INV-GRAPH-004', 'INV-GRAPH-005', 'INV-GRAPH-006', 'INV-GRAPH-007', 'INV-GRAPH-008']) {
    const c = cases.find(x => x.id === id);
    invariants.push({ id, expr: id, tol: c.tol, worst: c.abs_err, status: c.status, note: '' });
  }
  return buildReport({
    module: 'graph-forge', version: '1.0.0', layer: 'L5', cases, invariants, seed: '0x9E3779B9', started,
    tol: { 'INV-GRAPH-001': 0, 'INV-GRAPH-004': 0, 'INV-GRAPH-005': 0, 'INV-GRAPH-007': 0 }
  });
}

const API = {
  name: 'graph-forge', version: '1.0.0', layer: 'L5',
  buildGraph, bfsShortest, dijkstra, dijkstraRef, reconstructPath, bruteForceShortest, selfCheck
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeGraph = API;

if (typeof process !== 'undefined' && process.argv[1] && /graph-forge\.js$/.test(process.argv[1]) && process.argv.includes('--self-check')) {
  const r = selfCheck({ quick: process.argv.includes('--quick') });
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : require('./core-verify.js').formatTableSafe(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
