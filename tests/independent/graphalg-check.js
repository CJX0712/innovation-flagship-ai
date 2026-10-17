'use strict';
/* ============================================================================
 * 独立复核 graphalg-forge —— 不采信模块 selfCheck()，自带 mulberry32 PRNG 与
 * 手搓黄金参照交叉验证（world-class 独立审计口径）。
 *
 * 覆盖：
 *   R1  DFS 确定序 + ≠BFS（INV-GA-001）      R2  A* ≡ Dijkstra + 启发边界（INV-GA-002）
 *   R3  PageRank 长迭代对照（INV-GA-003）     R4  CC/SCC 双口径（INV-GA-004）
 *   R5  拓扑合法 + 环抛错（INV-GA-005）       R6  MST 双实现 + 割性质（INV-GA-006）
 *   R7  三角计数三重循环（INV-GA-007）        R8  Brandes 逐对 BFS + 确定性（INV-GA-008）
 *
 * 本文件不调用 graphalg-forge.selfCheck / makeRng 等内部件，
 * 全部随机量由本文件自带 mulberry32(seed) 生成，与模块内部轨道独立。
 * 退出码：0 = 全部独立复核通过；1 = 任一失败。
 * ========================================================================== */
const path = require('path');
const G = require(path.join(__dirname, '..', '..', 'src', 'graphalg-forge.js'));

/* ---------- 独立 PRNG：mulberry32（与模块 sfc32 不同族） ---------- */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ---------- 独立图生成（与模块生成器不同实现） ---------- */
function ownDigraph(n, m, seed) {
  const rng = mulberry32(seed);
  const set = new Set();
  while (set.size < m) {
    const u = Math.floor(rng() * n), v = Math.floor(rng() * n);
    if (u !== v) set.add(u * n + v);
  }
  const edges = Array.from(set, (k) => [Math.floor(k / n), k % n]);
  return { n, edges };
}
function ownUndirected(n, m, seed, wMax) {
  const rng = mulberry32(seed);
  const set = new Set();
  while (set.size < m) {
    const u = Math.floor(rng() * n), v = Math.floor(rng() * n);
    if (u !== v) set.add(u < v ? u * n + v : v * n + u);
  }
  const edges = Array.from(set, (k) => {
    const u = Math.floor(k / n), v = k % n;
    return wMax ? [u, v, 1 + Math.floor(rng() * wMax)] : [u, v];
  });
  return { n, edges };
}
function ownGeomGraph(n, k, seed) {
  const rng = mulberry32(seed);
  const coords = [];
  for (let i = 0; i < n; i++) coords.push([rng() * 10, rng() * 10]);
  const set = new Set();
  for (let u = 0; u < n; u++) {
    const ds = [];
    for (let v = 0; v < n; v++) if (v !== u) ds.push([Math.hypot(coords[u][0] - coords[v][0], coords[u][1] - coords[v][1]), v]);
    ds.sort((a, b) => a[0] - b[0]);
    for (let i = 0; i < Math.min(k, ds.length); i++) {
      const v = ds[i][1];
      set.add(u < v ? u * n + v : v * n + u);
    }
  }
  const adj = Array.from({ length: n }, () => []);
  for (const key of set) { const u = Math.floor(key / n), v = key % n; adj[u].push(v); adj[v].push(u); }
  for (const a of adj) a.sort((x, y) => x - y);
  return { n, coords, adj };
}
function buildAdj(n, edges) {
  const s = Array.from({ length: n }, () => new Set());
  for (const [u, v] of edges) s[u].add(v);
  return s.map((x) => Array.from(x).sort((a, b) => a - b));
}

/* ---------- 手搓黄金参照（零模块函数依赖） ---------- */
function handDFS(adj, s) {
  const vis = new Array(adj.length).fill(false), order = [];
  (function go(u) { vis[u] = true; order.push(u); for (const v of adj[u]) if (!vis[v]) go(v); })(s);
  return order;
}
function handBFS(adj, s) {
  const vis = new Array(adj.length).fill(false), order = [], q = [s];
  vis[s] = true;
  for (let qi = 0; qi < q.length; qi++) {
    const u = q[qi]; order.push(u);
    for (const v of adj[u]) if (!vis[v]) { vis[v] = true; q.push(v); }
  }
  return order;
}
function handDijkstra(adj, coords, s, t) {   // O(n²) 选择式
  const n = adj.length;
  const dist = new Array(n).fill(Infinity), done = new Array(n).fill(false);
  dist[s] = 0;
  let explored = 0;
  for (;;) {
    let u = -1;
    for (let v = 0; v < n; v++) if (!done[v] && (u === -1 || dist[v] < dist[u])) u = v;
    if (u === -1 || !isFinite(dist[u])) break;
    done[u] = true; explored++;
    if (u === t) break;
    for (const v of adj[u]) {
      const nd = dist[u] + Math.hypot(coords[u][0] - coords[v][0], coords[u][1] - coords[v][1]);
      if (nd < dist[v]) dist[v] = nd;
    }
  }
  return { dist: dist[t], explored };
}
function handPageRank(adj, d, iters) {
  const n = adj.length;
  const outdeg = adj.map((a) => a.length);
  let r = new Array(n).fill(1 / n);
  for (let it = 0; it < iters; it++) {
    let dang = 0;
    for (let v = 0; v < n; v++) if (outdeg[v] === 0) dang += r[v];
    const rn = new Array(n).fill((1 - d) / n + d * dang / n);
    for (let u = 0; u < n; u++) {
      if (outdeg[u] === 0) continue;
      const sh = d * r[u] / outdeg[u];
      for (const v of adj[u]) rn[v] += sh;
    }
    let ssum = 0; for (const x of rn) ssum += x;
    for (let v = 0; v < n; v++) rn[v] /= ssum;
    r = rn;
  }
  return r;
}
function handUnionFindCC(n, edges) {          // 根恒为最小 id
  const p = Array.from({ length: n }, (_, i) => i);
  const find = (x) => { while (p[x] !== x) { p[x] = p[p[x]]; x = p[x]; } return x; };
  for (const [u, v] of edges) {
    const ru = find(u), rv = find(v);
    if (ru !== rv) { if (ru < rv) p[rv] = ru; else p[ru] = rv; }
  }
  const comp = new Array(n); let count = 0; const seen = new Map();
  for (let v = 0; v < n; v++) { const r = find(v); if (!seen.has(r)) seen.set(r, count++); comp[v] = r; }
  return { comp, count };
}
function handBFS_CC(n, edges) {               // 泛洪标记（与并查集完全不同族；无向图双向邻接）
  const adj = Array.from({ length: n }, () => []);
  for (const [u, v] of edges) { adj[u].push(v); adj[v].push(u); }
  const comp = new Array(n).fill(-1);
  let count = 0;
  for (let s = 0; s < n; s++) {
    if (comp[s] !== -1) continue;
    const q = [s]; comp[s] = s;
    while (q.length) { const v = q.pop(); for (const w of adj[v]) if (comp[w] === -1) { comp[w] = s; q.push(w); } }
    count++;
  }
  return { comp, count };
}
function handKosaraju(n, edges) {
  const adj = buildAdj(n, edges);
  const radj = Array.from({ length: n }, () => []);
  for (const [u, v] of edges) radj[v].push(u);
  for (const a of radj) a.sort((x, y) => x - y);
  const vis = new Array(n).fill(false), fin = [];
  for (let s = 0; s < n; s++) {
    if (vis[s]) continue;
    const stk = [[s, 0]]; vis[s] = true;
    while (stk.length) {
      const top = stk[stk.length - 1], v = top[0];
      if (top[1] < adj[v].length) { const w = adj[v][top[1]++]; if (!vis[w]) { vis[w] = true; stk.push([w, 0]); } }
      else { fin.push(v); stk.pop(); }
    }
  }
  const vis2 = new Array(n).fill(false), comp = new Array(n).fill(-1);
  let count = 0;
  for (let i = fin.length - 1; i >= 0; i--) {
    const s = fin[i];
    if (vis2[s]) continue;
    const mem = [], stk = [s]; vis2[s] = true;
    while (stk.length) { const v = stk.pop(); mem.push(v); for (const w of radj[v]) if (!vis2[w]) { vis2[w] = true; stk.push(w); } }
    const minId = Math.min(...mem);
    for (const m of mem) comp[m] = minId;
    count++;
  }
  return { comp, count };
}
function handTopoCheck(n, edges, order) {     // 合法性：全边 pos[u]<pos[v] 且为排列
  if (order.length !== n) return false;
  const pos = new Array(n);
  for (let i = 0; i < order.length; i++) pos[order[i]] = i;
  if (new Set(order).size !== n) return false;
  return edges.every(([u, v]) => pos[u] < pos[v]);
}
function handMSTCutOk(n, edges, tree) {       // 割性质证书
  const tset = new Set(tree.map((e) => (e[0] < e[1] ? e[0] * n + e[1] : e[1] * n + e[0])));
  const tadj = Array.from({ length: n }, () => []);
  for (const e of tree) { tadj[e[0]].push(e[1]); tadj[e[1]].push(e[0]); }
  for (const e of tree) {
    const ek = e[0] < e[1] ? e[0] * n + e[1] : e[1] * n + e[0];
    const part = new Array(n).fill(false), q = [e[0]];
    part[e[0]] = true;
    while (q.length) {
      const v = q.pop();
      for (const w of tadj[v]) {
        const wk = v < w ? v * n + w : w * n + v;
        if (wk === ek || part[w]) continue;
        part[w] = true; q.push(w);
      }
    }
    let minCross = Infinity;
    for (const f of edges) {
      const fk = f[0] < f[1] ? f[0] * n + f[1] : f[1] * n + f[0];
      if (tset.has(fk)) continue;
      if (part[f[0]] !== part[f[1]] && f[2] < minCross) minCross = f[2];
    }
    if (minCross < e[2]) return false;
  }
  return true;
}
function handTripleTriangles(n, edges) {
  const eset = new Set();
  for (const [u, v] of edges) { if (u !== v) eset.add(u < v ? u * n + v : v * n + u); }
  let c = 0;
  for (let u = 0; u < n; u++) for (let v = u + 1; v < n; v++) {
    if (!eset.has(u * n + v)) continue;
    for (let w = v + 1; w < n; w++) if (eset.has(u * n + w) && eset.has(v * n + w)) c++;
  }
  return c;
}
function handAllPairs(n, nbr) {
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
function handPairwiseBetweenness(n, nbr) {
  const { dist, sigma } = handAllPairs(n, nbr);
  const btw = new Array(n).fill(0);
  for (let s = 0; s < n; s++) for (let t = 0; t < n; t++) {
    if (t === s || dist[s][t] < 0) continue;
    for (let v = 0; v < n; v++) {
      if (v === s || v === t) continue;
      if (dist[s][v] >= 0 && dist[t][v] >= 0 && dist[s][v] + dist[t][v] === dist[s][t]) {
        btw[v] += sigma[s][v] * sigma[t][v] / sigma[s][t];
      }
    }
  }
  for (let v = 0; v < n; v++) btw[v] /= 2;
  return btw;
}

/* ---------- 审计框架 ---------- */
const results = [];
function check(id, name, ok, detail) {
  results.push({ id, name, ok: !!ok, detail });
  console.log((ok ? '  [R-PASS] ' : '  [R-FAIL] ') + id + ' ' + name + ' — ' + detail);
}
function seqEq(a, b) { return a.length === b.length && a.every((x, i) => x === b[i]); }
function maxAbs(a, b) { let d = 0; for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i])); return d; }

/* ---------- R1 DFS（INV-GA-001） ---------- */
{
  let bitwise = true, permOk = true;
  for (const sd of [0x1234, 0x9E3779B9 ^ 0x51CE, 0xDEADBEEF]) {
    const g = ownDigraph(26, 70, sd);
    const adj = buildAdj(g.n, g.edges);
    for (let s = 0; s < g.n; s += 4) {
      const got = G.dfsOrder(adj, s);
      const ref = handDFS(adj, s);
      if (!seqEq(got, ref)) bitwise = false;
      const reach = new Set(handBFS(adj, s));
      if (new Set(got).size !== got.length || !got.every((v) => reach.has(v))) permOk = false;
    }
  }
  let dfsNeBfs = false;
  for (let sd = 0x1234; sd < 0x1234 + 12 && !dfsNeBfs; sd++) {
    const g = ownDigraph(26, 70, sd);
    const adj = buildAdj(g.n, g.edges);
    for (let s = 0; s < g.n; s++) if (!seqEq(G.dfsOrder(adj, s), handBFS(adj, s))) { dfsNeBfs = true; break; }
  }
  check('R1', 'DFS≡递归前序(EXACT)+排列+≠BFS', bitwise && permOk && dfsNeBfs,
    '逐位=' + bitwise + ' 排列=' + permOk + ' DFS≠BFS=' + dfsNeBfs);
}

/* ---------- R2 A*（INV-GA-002） ---------- */
{
  let worst = 0;
  for (let sd = 0x51CE; sd < 0x51CE + 6; sd++) {
    const g = ownGeomGraph(28, 3, sd);
    for (let s = 0; s < 6; s++) for (let t = 0; t < g.n; t++) {
      if (t === s) continue;
      const dg = handDijkstra(g.adj, g.coords, s, t);
      if (!isFinite(dg.dist)) continue;
      const a0 = G.aStar(g.adj, g.coords, s, t, { hScale: 0 });
      if (a0.found) worst = Math.max(worst, Math.abs(a0.dist - dg.dist));
    }
  }
  // 可采纳 h 探索更少 + ×10 次优边界 + 路径合法
  let strict = false, allGe = true, subopt = false, pathOk = true;
  for (let sd = 0x51CE; sd < 0x51CE + 80; sd++) {
    const g = ownGeomGraph(28, 3, sd);
    const t = g.n - 1;
    const dg = handDijkstra(g.adj, g.coords, 0, t);
    if (!isFinite(dg.dist)) continue;
    const a1 = G.aStar(g.adj, g.coords, 0, t, { hScale: 1 });
    if (a1.found && a1.explored < dg.explored) strict = true;
    if (a1.found && a1.path[0] === 0 && a1.path[a1.path.length - 1] === t) {
      let wsum = 0;
      let okP = true;
      for (let i = 0; i + 1 < a1.path.length; i++) {
        const u = a1.path[i], v = a1.path[i + 1];
        if (!g.adj[u].includes(v)) { okP = false; break; }
        wsum += Math.hypot(g.coords[u][0] - g.coords[v][0], g.coords[u][1] - g.coords[v][1]);
      }
      if (okP && Math.abs(wsum - a1.dist) > 1e-9) okP = false;
      if (!okP) pathOk = false;
    }
    const g2 = ownGeomGraph(28, 2, sd);
    const t2 = g2.n - 1;
    const dg2 = handDijkstra(g2.adj, g2.coords, 0, t2);
    if (!isFinite(dg2.dist)) continue;
    const a10 = G.aStar(g2.adj, g2.coords, 0, t2, { hScale: 10 });
    if (!a10.found || a10.dist < dg2.dist - 1e-12) allGe = false;
    if (a10.found && a10.dist > dg2.dist + 1e-9) subopt = true;
  }
  check('R2', 'A*: h=0≡Dijkstra + 探索更少 + ×10 恒≥最优可次优 + 路径合法', worst <= 1e-15 && strict && allGe && subopt && pathOk,
    'h=0 max|Δ|=' + worst.toExponential(2) + ' 探索更少=' + strict + ' ≥最优=' + allGe + ' 严格次优=' + subopt + ' 路径=' + pathOk);
}

/* ---------- R3 PageRank（INV-GA-003） ---------- */
{
  let okAll = true, det = '';
  for (const sd of [0x9E37, 0xC0DE, 0x777]) {
    const g = ownDigraph(20, 55, sd);
    const adj = buildAdj(g.n, g.edges);
    const pr = G.pageRank(adj, { d: 0.85, tol: 1e-13, maxIter: 5000 });
    const ssum = pr.ranks.reduce((a, b) => a + b, 0);
    const ref = handPageRank(adj, 0.85, 30000);
    const dev = maxAbs(pr.ranks, ref);
    const uni = Math.max(...pr.ranks.map((r) => Math.abs(r - 1 / g.n)));
    const ok = Math.abs(ssum - 1) <= 1e-15 && pr.residual < 1e-10 && dev <= 1e-10 && uni > 1e-3;
    if (!ok) okAll = false;
    det = 'Σ−1=' + (ssum - 1).toExponential(1) + ' resid=' + pr.residual.toExponential(1) + ' Δgolden=' + dev.toExponential(1) + ' ≠均匀=' + uni.toExponential(1);
  }
  check('R3', 'PageRank Σ=1/残差/长迭代对照/≠均匀', okAll, det);
}

/* ---------- R4 CC / SCC（INV-GA-004） ---------- */
{
  let ccUf = true, ccBfs = true, sccOk = true;
  for (const sd of [0xAA11, 0xBB22, 0xCC33]) {
    const gu = ownUndirected(30, 42, sd, 0);
    const cc = G.connectedComponents(gu.n, gu.edges);
    const r1 = handUnionFindCC(gu.n, gu.edges);
    const r2 = handBFS_CC(gu.n, gu.edges);
    if (!seqEq(cc.comp, r1.comp) || cc.count !== r1.count) ccUf = false;
    if (!seqEq(cc.comp, r2.comp) || cc.count !== r2.count) ccBfs = false;
    const gd = ownDigraph(24, 48, sd);
    const sc = G.sccTarjan(gd.n, gd.edges);
    const ks = handKosaraju(gd.n, gd.edges);
    if (!seqEq(sc.comp, ks.comp) || sc.count !== ks.count) sccOk = false;
  }
  const sccP = G.sccTarjan(4, [[0, 1], [1, 2], [2, 3]]);
  const ccP = G.connectedComponents(4, [[0, 1], [1, 2], [2, 3]]);
  const confused = sccP.count === 4 && ccP.count === 1;
  check('R4', 'CC≡并查集≡泛洪(EXACT) + SCC≡Kosaraju(EXACT) + SCC≠CC', ccUf && ccBfs && sccOk && confused,
    'CC-UF=' + ccUf + ' CC-BFS=' + ccBfs + ' SCC-Kosaraju=' + sccOk + ' 路径图 SCC=' + sccP.count + '/CC=' + ccP.count);
}

/* ---------- R5 拓扑（INV-GA-005） ---------- */
{
  const rng = mulberry32(0x7070);
  let allOk = true, nG = 0;
  for (let i = 0; i < 6; i++) {
    const n = 18;
    const edges = [];
    for (let u = 0; u < n; u++) for (let v = u + 1; v < n; v++) if (rng() < 0.12) edges.push([u, v]);
    const ord = G.topoSort(n, edges);
    if (!handTopoCheck(n, edges, ord)) allOk = false;
    nG++;
  }
  let cyc = false, self = false;
  try { G.topoSort(3, [[0, 1], [1, 2], [2, 0]]); } catch (e) { cyc = true; }
  try { G.topoSort(2, [[0, 0]]); } catch (e) { self = true; }
  check('R5', '拓扑全边合法(' + nG + '图) + 环/自环抛错', allOk && cyc && self, '合法=' + allOk + ' 环抛错=' + cyc + ' 自环抛错=' + self);
}

/* ---------- R6 MST（INV-GA-006） ---------- */
{
  let worst = 0, structOk = true, cutOk = true;
  for (const sd of [0x6A6A, 0x6B6B, 0x6C6C]) {
    const g = ownUndirected(22, 60, sd, 97);
    const k = G.kruskalMST(g.n, g.edges);
    const adjW = Array.from({ length: g.n }, () => []);
    for (const [u, v, w] of g.edges) { adjW[u].push([v, w]); adjW[v].push([u, w]); }
    for (const a of adjW) a.sort((x, y) => x[0] - y[0]);
    const p = G.primMST(g.n, adjW);
    worst = Math.max(worst, Math.abs(k.total - p.total));
    if (k.tree.length !== g.n - 1) structOk = false;
    const par = Array.from({ length: g.n }, (_, i) => i);
    const find = (x) => { while (par[x] !== x) { par[x] = par[par[x]]; x = par[x]; } return x; };
    let comps = g.n;
    for (const [u, v] of k.tree) { const ru = find(u), rv = find(v); if (ru === rv) structOk = false; else { par[ru] = rv; comps--; } }
    if (comps !== 1) structOk = false;
    if (!handMSTCutOk(g.n, g.edges, k.tree)) cutOk = false;
  }
  check('R6', 'MST Kruskal≡Prim + n−1 边无环Spanning + 割性质证书', worst <= 1e-15 && structOk && cutOk,
    '|K−P|max=' + worst.toExponential(2) + ' 结构=' + structOk + ' 割性质=' + cutOk);
}

/* ---------- R7 三角计数（INV-GA-007） ---------- */
{
  let exact = true, opsOk = true, info = '';
  for (const sd of [0x7A7A, 0x7B7B, 0x7C7C]) {
    const g = ownUndirected(20, 42, sd, 0);
    const r = G.countTriangles(g.n, g.edges);
    const ref = handTripleTriangles(g.n, g.edges);
    if (r.count !== ref) exact = false;
    if (r.ops >= (g.n * g.n * g.n) / 10) opsOk = false;
    info = 'count=' + r.count + ' ops=' + r.ops;
  }
  check('R7', '三角计数≡三重循环(EXACT) + ops≪n³', exact && opsOk, info);
}

/* ---------- R8 中心性 + 确定性（INV-GA-008） ---------- */
{
  // 唯一最短路图（随机树 + 逐 seed 校验 σ≡1），保证逐对介数逐位精确
  let ug = null;
  for (let sd = 0x8A8A; sd < 0x8A8A + 300 && !ug; sd++) {
    const n = 14, rng = mulberry32(sd);
    const edges = [];
    for (let v = 1; v < n; v++) edges.push([Math.floor(rng() * v), v]);
    const set = new Set();
    for (const [u, v] of edges) set.add(u < v ? u * n + v : v * n + u);
    const nbr = Array.from({ length: n }, () => []);
    for (const key of set) { const u = Math.floor(key / n), v = key % n; nbr[u].push(v); nbr[v].push(u); }
    for (const a of nbr) a.sort((x, y) => x - y);
    const { sigma } = handAllPairs(n, nbr);
    let uniq = true;
    outer: for (let s = 0; s < n; s++) for (let t = 0; t < n; t++) if (s !== t && sigma[s][t] !== 1) { uniq = false; break outer; }
    if (uniq) ug = { n, edges: Array.from(set, (k) => [Math.floor(k / n), k % n]), nbr };
  }
  let bWorst = Infinity, degWorst = Infinity;
  if (ug) {
    const c = G.centrality(ug.n, ug.edges);
    bWorst = maxAbs(c.betweenness, handPairwiseBetweenness(ug.n, ug.nbr));
    degWorst = maxAbs(c.degree, ug.nbr.map((a) => a.length / (ug.n - 1)));
  }
  // 确定性：同 seed 逐位 / 异 seed 分歧（自建输入直接喂模块 API）
  const pipeline = (sd) => {
    const g1 = ownUndirected(16, 26, sd, 40);
    const g2 = ownDigraph(12, 22, sd);
    const geo = ownGeomGraph(14, 3, sd);
    const a = G.aStar(geo.adj, geo.coords, 0, geo.n - 1, { hScale: 1 });
    const m = G.kruskalMST(16, g1.edges);
    const s = G.sccTarjan(16, ownDigraph(16, 28, sd).edges);
    const c = G.centrality(16, ownUndirected(16, 26, sd, 0).edges);
    const pr = G.pageRank(buildAdj(12, g2.edges), { d: 0.85, tol: 1e-12, maxIter: 500 });
    const dfs = G.dfsOrder(buildAdj(16, ownDigraph(16, 28, sd).edges), 0);
    return JSON.stringify([dfs, pr.ranks, a.dist, a.path, m.total, s.comp, c.betweenness]);
  };
  const r1 = pipeline(0xB00B), r2 = pipeline(0xB00B), r3 = pipeline(0xC0DE);
  const identical = r1 === r2, diverged = r1 !== r3;
  check('R8', 'Brandes≡逐对BFS(TIGHT) + 度≡手搓 + 同seed逐位/异seed分歧', bWorst <= 1e-15 && degWorst <= 0 && identical && diverged,
    '介数max|Δ|=' + bWorst.toExponential(2) + ' 度max|Δ|=' + degWorst.toExponential(2) + ' 同seed=' + identical + ' 异seed分歧=' + diverged);
}

/* ---------- 汇总 ---------- */
const failed = results.filter((r) => !r.ok).length;
console.log('══════ graphalg-forge 独立复核: ' + (results.length - failed) + '/' + results.length + ' 通过 ══════');
process.exit(failed === 0 ? 0 : 1);
