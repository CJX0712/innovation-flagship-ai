'use strict';
/* ============================================================================
 * 独立复核 serving-forge —— 不采信模块自检，手搓参照交叉验证。
 *
 * 覆盖全部 8 条不变量（INV-SV-001..008），每条至少一个独立验证角度：
 *   INV-SV-001  手搓串行黄金参照 + 手搓组批参照 + 批无关性
 *   INV-SV-002  手搓算子实现三方对照（模块融合 vs 模块串联 vs 手搓融合）
 *   INV-SV-003  手搓 Kahn 重实现对照 + 全边扫描 + 环抛错
 *   INV-SV-004  手搓区间重叠/越界扫描 + 复用计数 + 耗尽/双释安全
 *   INV-SV-005  手搓分块循环参照 + 多 chunkSize 交叉
 *   INV-SV-006  手搓计数断言（承载差≤1、3:1、加权序≠均权序）
 *   INV-SV-007  手搓排序参照（FIFO / (priority,seq)）
 *   INV-SV-008  重放确定性 + 闭式上界推导 + rate 减半证伪
 *
 * 确定性：固定种子 LCG + 逻辑时钟，连跑 3 次结果逐位相同且 exit 0。
 * ========================================================================== */
const path = require('path');
const SV = require(path.join(__dirname, '..', '..', 'src', 'serving-forge.js'));

/* ---------- 独立 PRNG（LCG，与模块 sfc32 完全无关） ---------- */
function lcg(seed) {
  let z = seed >>> 0;
  return () => { z = (Math.imul(z, 1664525) + 1013904223) >>> 0; return z / 4294967296; };
}

const results = [];
function record(id, ok, detail) {
  results.push(ok);
  console.log('[' + id + '] ' + detail + (ok ? ' ✅' : ' ❌'));
}

/* ---------- INV-SV-001 批处理等价：手搓串行 + 手搓组批双参照 ---------- */
{
  const rng = lcg(0x51001);
  const OP = (x) => Math.tanh(2 * x + 1) * 0.5;
  const reqs = [];
  for (let i = 0; i < 20; i++) {
    const len = 1 + Math.floor(rng() * 8);
    const payload = [];
    for (let j = 0; j < len; j++) payload.push(rng() * 4 - 2);
    reqs.push({ id: 'q' + i, payload });
  }
  // 手搓串行参照
  const serial = reqs.map((r) => r.payload.map(OP));
  // 手搓组批参照（独立实现 pad + elementwise）
  const handBatch = (batch) => {
    const maxLen = Math.max.apply(null, batch.map((r) => r.payload.length));
    return batch.map((r) => {
      const row = r.payload.slice();
      while (row.length < maxLen) row.push(0);
      return row.map(OP);
    });
  };
  const batches = SV.makeBatches(reqs, 6);
  let dev = 0, bdev = 0;
  const validOf = {};
  for (const batch of batches) {
    const pb = SV.padBatch(batch, 0);
    const out = SV.processBatch(pb.rows, pb.lens, OP);
    const hand = handBatch(batch);
    for (let i = 0; i < batch.length; i++) {
      const valid = out[i].slice(0, pb.lens[i]);
      validOf[pb.ids[i]] = valid;
      for (let j = 0; j < pb.lens[i]; j++) {
        dev = Math.max(dev, Math.abs(valid[j] - serial[reqs.indexOf(batch[i])][j]));
        bdev = Math.max(bdev, Math.abs(valid[j] - hand[i][j]));
      }
    }
  }
  // 批无关性：批大小=1 vs 批大小=N
  for (const r of reqs) {
    const pb1 = SV.padBatch([r], 0);
    const single = SV.processBatch(pb1.rows, pb1.lens, OP)[0].slice(0, r.payload.length);
    for (let j = 0; j < r.payload.length; j++) bdev = Math.max(bdev, Math.abs(single[j] - validOf[r.id][j]));
  }
  record('INV-SV-001', dev <= 1e-15 && bdev <= 1e-15,
    '手搓串行偏差=' + dev.toExponential(2) + ' 手搓组批/批无关偏差=' + bdev.toExponential(2) + ' 批数=' + batches.length);
}

/* ---------- INV-SV-002 算子融合：手搓三方对照 + 调用计数 ---------- */
{
  const myOps = {
    add: (x) => x + 1, bias: (x) => x + 2, mul: (x) => x * 3,
    relu: (x) => Math.max(0, x), sigmoid: (x) => 1 / (1 + Math.exp(-x))
  };
  const myFused = {
    add_relu: (x) => Math.max(0, x + 1),
    bias_relu: (x) => Math.max(0, x + 2),
    mul_sigmoid: (x) => 1 / (1 + Math.exp(-(x * 3)))
  };
  const ops = ['add', 'relu', 'mul', 'sigmoid', 'add', 'relu', 'bias', 'relu'];
  const fu = SV.fuseOps(ops);
  const rng = lcg(0x51002);
  let devMod = 0, devHand = 0;
  for (let i = 0; i < 8; i++) {
    const x = rng() * 6 - 3;
    let seq = x;
    for (const op of ops) seq = myOps[op](seq);                 // 手搓未融合
    const mRaw = SV.applyOps(x, ops);                           // 模块未融合
    const mFus = SV.applyOps(x, fu.kernels);                    // 模块融合
    let handFus = x;                                            // 手搓融合（按 fused kernels 映射）
    for (const k of fu.kernels) handFus = (myFused[k] || myOps[k])(handFus);
    devMod = Math.max(devMod, Math.abs(mFus - mRaw));
    devHand = Math.max(devHand, Math.abs(mFus - handFus), Math.abs(mFus - seq));
  }
  const callsOk = fu.calls < ops.length;
  record('INV-SV-002', devMod <= 1e-15 && devHand <= 1e-15 && callsOk,
    '模块融合vs串联=' + devMod.toExponential(2) + ' vs手搓=' + devHand.toExponential(2) + ' 调用数 ' + fu.calls + '<' + ops.length);
}

/* ---------- INV-SV-003 内核调度：手搓 Kahn 对照 + 全边 + 环抛错 ---------- */
function myKahn(ops) {
  const byId = new Map(ops.map((o) => [o.id, o]));
  const indeg = new Map(ops.map((o) => [o.id, o.deps.length]));
  const out = new Map();
  ops.forEach((o) => out.set(o.id, []));
  ops.forEach((o) => o.deps.forEach((d) => out.get(d).push(o.id)));
  const ready = ops.filter((o) => o.deps.length === 0).map((o) => o.id);
  const order = [];
  while (ready.length) {
    ready.sort((a, b) => (byId.get(b).priority - byId.get(a).priority) || (String(a) < String(b) ? -1 : 1));
    const id = ready.shift();
    order.push(id);
    for (const n of out.get(id)) {
      indeg.set(n, indeg.get(n) - 1);
      if (indeg.get(n) === 0) ready.push(n);
    }
  }
  return order;
}
{
  const mk = (pr) => [
    { id: 'a', deps: [], priority: pr[0] },
    { id: 'b', deps: ['a'], priority: pr[1] },
    { id: 'c', deps: ['a'], priority: pr[2] },
    { id: 'd', deps: ['b', 'c'], priority: pr[3] },
    { id: 'e', deps: ['c'], priority: pr[4] },
    { id: 'f', deps: ['d', 'e'], priority: pr[5] },
    { id: 'g', deps: ['f'], priority: pr[6] },
    { id: 'h', deps: ['f'], priority: pr[7] }
  ];
  const edges = [['a', 'b'], ['a', 'c'], ['b', 'd'], ['c', 'd'], ['c', 'e'], ['d', 'f'], ['e', 'f'], ['f', 'g'], ['f', 'h']];
  const p1 = [1, 2, 3, 4, 5, 6, 7, 8], p2 = [8, 7, 6, 5, 4, 3, 2, 1];
  const o1 = SV.scheduleKernels(mk(p1)), o2 = SV.scheduleKernels(mk(p2));
  const checkTopo = (order) => {
    const pos = new Map();
    order.forEach((id, i) => pos.set(id, i));
    let v = 0;
    for (const [u, w] of edges) if (!(pos.get(u) < pos.get(w))) v++;
    return v;
  };
  const edgeV = checkTopo(o1) + checkTopo(o2);
  const sameAsHand = (o1.join(',') === myKahn(mk(p1)).join(',')) && (o2.join(',') === myKahn(mk(p2)).join(','));
  const differs = o1.join(',') !== o2.join(',');
  let cycleThrew = false;
  try { SV.scheduleKernels([{ id: 'x', deps: ['y'], priority: 1 }, { id: 'y', deps: ['x'], priority: 1 }]); } catch (e) { cycleThrew = true; }
  record('INV-SV-003', edgeV === 0 && sameAsHand && differs && cycleThrew,
    '边违例=' + edgeV + ' ≡手搓Kahn=' + sameAsHand + ' 异权重不同序=' + differs + ' 环抛错=' + cycleThrew);
}

/* ---------- INV-SV-004 内存池：手搓区间扫描 + 复用/耗尽/双释 ---------- */
{
  const pool = new SV.MemoryPool(16, 4);
  const rng = lcg(0x51004);
  const handles = [];
  let viol = 0;
  for (let step = 0; step < 60; step++) {
    if (handles.length === 0 || (handles.length < 4 && rng() < 0.6)) {
      const h = pool.alloc();
      if (!h) viol++;
      else handles.push(h);
    } else {
      const i = Math.floor(rng() * handles.length);
      pool.free(handles.splice(i, 1)[0]);
    }
    for (let i = 0; i < handles.length; i++) {
      const a = handles[i];
      if (!(a.offset >= 0 && a.offset + 16 <= 64)) viol++;
      for (let j = i + 1; j < handles.length; j++) {
        const b = handles[j];
        if (a.offset < b.offset + 16 && b.offset < a.offset + 16) viol++;
      }
    }
  }
  const st = pool.stats();
  const reuseOk = st.mallocs < st.allocs;
  const p2 = new SV.MemoryPool(16, 2);
  p2.alloc(); p2.alloc();
  const exhausted = p2.alloc() === null;
  let dblThrew = false;
  const hd = p2.alloc();
  try { p2.free(hd); p2.free(hd); } catch (e) { dblThrew = true; }
  record('INV-SV-004', viol === 0 && reuseOk && exhausted && dblThrew,
    '区间/越界违例=' + viol + ' mallocs=' + st.mallocs + '<allocs=' + st.allocs + ' 耗尽null=' + exhausted + ' 双释抛错=' + dblThrew);
}

/* ---------- INV-SV-005 流式管线：手搓分块循环参照 ---------- */
{
  const rng = lcg(0x51005);
  const items = [];
  for (let i = 0; i < 13; i++) items.push(Math.floor(rng() * 100));
  const fn = (chunk) => chunk.map((x) => x * 3 + 1);
  const handStream = (cs) => {
    const out = [];
    for (let b = 0; b < items.length; b += cs) for (const v of fn(items.slice(b, b + cs))) out.push(v);
    return out;
  };
  const full = fn(items);
  let viol = 0;
  for (const cs of [1, 2, 3, 5, 7, 13, 20]) {
    const m = SV.streamProcess(items, cs, fn);
    const h = handStream(cs);
    if (m.length !== full.length) { viol++; continue; }
    for (let i = 0; i < full.length; i++) if (m[i] !== full[i] || m[i] !== h[i]) viol++;
  }
  record('INV-SV-005', viol === 0, 'chunkSize=[1,2,3,5,7,13,20] 手搓参照违例=' + viol);
}

/* ---------- INV-SV-006 负载均衡：手搓计数断言 ---------- */
{
  const rr = SV.makeRoundRobin(['w0', 'w1', 'w2']);
  for (let i = 0; i < 10; i++) rr.dispatch();
  const L = rr.loads();
  const spread = Math.max.apply(null, L) - Math.min.apply(null, L);
  const wrr = SV.makeWeightedRR(['a', 'b'], [3, 1]);
  const seqW = [];
  for (let i = 0; i < 12; i++) seqW.push(wrr.dispatch());
  const LW = wrr.loads();
  const ratioOk = LW[0] === 9 && LW[1] === 3;
  const rr2 = SV.makeRoundRobin(['a', 'b']);
  const seqR = [];
  for (let i = 0; i < 4; i++) seqR.push(rr2.dispatch());
  const differs = seqW.slice(0, 4).some((x, i) => x !== seqR[i]);
  record('INV-SV-006', spread <= 1 && ratioOk && differs,
    '等权loads=[' + L.join(',') + '] spread=' + spread + ' 加权loads=[' + LW.join(',') + '](应9:3) 加权序≠均权序=' + differs);
}

/* ---------- INV-SV-007 请求队列：手搓排序参照 ---------- */
{
  const rng = lcg(0x51007);
  const vals = [];
  for (let i = 0; i < 15; i++) vals.push(Math.floor(rng() * 1000));
  const q = new SV.FIFOQueue();
  for (const v of vals) q.enqueue(v);
  const outF = [];
  while (q.size()) outF.push(q.dequeue());
  const fifoOk = outF.every((v, i) => v === vals[i]);
  const pqItems = [[10, 3], [21, 1], [32, 2], [43, 1], [54, 3], [65, 2], [76, 1], [87, 3], [98, 2], [19, 1]];
  const pq = new SV.PriorityQueue();
  for (const [v, p] of pqItems) pq.enqueue(v, p);
  const outP = [];
  while (pq.size()) outP.push(pq.dequeue());
  const ref = pqItems.map(([v, p], seq) => ({ v, p, seq }))
    .sort((a, b) => (a.p - b.p) || (a.seq - b.seq))
    .map((x) => x.v);
  const pqOk = outP.every((v, i) => v === ref[i]);
  const differs = outF.slice(0, 10).some((v, i) => v !== outP[i]);
  record('INV-SV-007', fifoOk && pqOk && differs,
    'FIFO≡入队=' + fifoOk + ' PQ≡(priority,seq)手搓参照=' + pqOk + ' 两队列出队序分歧=' + differs);
}

/* ---------- INV-SV-008 令牌桶：重放确定性 + 闭式上界 + rate减半证伪 ---------- */
{
  const runBucket = (rate) => {
    const tb = new SV.TokenBucket(rate, 5);
    const res = [];
    for (let t = 0; t <= 7; t++) for (let k = 0; k < 3; k++) res.push(tb.tryConsume(t, 1) ? 1 : 0);
    return res;
  };
  const r1 = runBucket(2), r2 = runBucket(2);
  const detOk = r1.every((v, i) => v === r2[i]);
  // 闭式上界：初始 5 枚令牌各至多消费一次，[0,T] 内新生 2T 枚 → 放行数 ≤ 5+2T（EXACT）
  let boundViol = 0;
  for (let T = 0; T <= 11; T++) {
    let passes = 0;
    for (let i = 0; i < r1.length; i++) if (Math.floor(i / 3) <= T && r1[i] === 1) passes++;
    if (passes > 5 + 2 * T) boundViol++;
  }
  const rHalf = runBucket(1);
  const differs = r1.some((v, i) => v !== rHalf[i]);
  record('INV-SV-008', detOk && boundViol === 0 && differs,
    '重放一致=' + detOk + ' 上界违例=' + boundViol + ' rate减半分歧=' + differs + ' r1=' + r1.join(''));
}

/* ---------- 汇总 ---------- */
const pass = results.every(Boolean);
console.log(pass
  ? '\n✅ serving-forge 独立复核通过：8 条不变量全部经手搓参照交叉验证'
  : '\n❌ serving-forge 独立复核失败');
process.exit(pass ? 0 : 1);
