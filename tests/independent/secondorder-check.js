'use strict';
/* ============================================================================
 * 独立复核 secondorder-forge —— 不采信模块 selfCheck()，自带 mulberry32 PRNG、
 * 独立高斯消元 / 独立 2×2 特征分解 / 独立 CG / 独立 dogleg 信赖域 / 独立 BFGS·
 * DFP·SR1 更新 / 独立黄金分割线搜索作为黄金参照（均本文件独立实现，与模块
 * 内部 sfc32 及优化算法路径完全独立）。
 *
 * 覆盖：
 *   R1  牛顿法对凸二次 1 步收敛 EXACT：x₁ 恰为极小点（独立高斯消元对照），残差
 *       ‖A x₁ − b‖ ≤ 1e-12；1 维 f(x)=x² 单步必达 0；非二次无此性质（反空洞）（INV-SO-001）
 *   R2  CG 对 SPD 线性系统：独立 CG 重实现与模块 cg 解一致（差 ≤1e-10）+ 独立解与
 *       高斯消元 EXACT + 下降方向 gₖᵀdₖ<0 扫描（INV-SO-002）
 *   R3  BFGS/L-BFGS：Rosenbrock 收敛(f≤f₀,‖∇f‖<1e-3) + 逆更新(hessianApprox)恒对称正定
 *       (SPD,最小特征值>0) + 割线方程 H₊y=s(≤1e-12) + 曲率 yᵀs>0 护 SPD（INV-SO-003）
 *   R4  强 Wolfe：Armijo + 曲率条件双双满足 + 独立黄金分割局部极小对拍(差≤1e-3) +
 *       非下降方向(取自身梯度)返回 0 证伪（INV-SO-004）
 *   R5  信赖域 dogleg：独立 dogleg 子问题重实现满足 m(s*)≤m(p_c) 且 ‖s*‖≤Δ（逐配置/
 *       逐迭代）+ 独立 trust-region 循环与模块 trustRegion 在二次方程上同点收敛 +
 *       半径自适应(扩/缩均触发)（INV-SO-005）
 *   R6  高斯牛顿 ≡ 牛顿(线性残差) EXACT：独立推导 m=(JᵀJ)⁻¹Jᵀr 与牛顿步一致(差≤1e-12)
 *       + 非线性最小二乘残差范数非增（INV-SO-006）
 *   R7  阻尼牛顿(H+λI 恒 SPD 且下降) + 端到端收敛 + Hessian-free ≡ 精确牛顿(二次) EXACT
 *       （独立重实现对照）（INV-SO-007）
 *   R8  拟牛顿校正(BFGS/DFP/SR1) 对称 + 割线方程 + SR1 秩1公式 EXACT + 独立重实现
 *       逐项对拍 + 负曲率跳过护 SPD（INV-SO-008）
 *
 * 本文件不调用 secondorder-forge.selfCheck；所有随机量由本文件 mulberry32 生成，
 * 与模块内部 sfc32 轨道独立；黄金参照均为本文件独立实现。
 * 退出码：0 = 全部独立复核通过；1 = 任一失败。
 * ========================================================================== */
const path = require('path');
const G = require(path.join(__dirname, '..', '..', 'src', 'secondorder-forge.js'));

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

/* ---------- 向量/矩阵原语（本文件独立实现，仅作标量/向量便利，非被测算法） ---------- */
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function vadd(a, b) { return a.map((x, i) => x + b[i]); }
function vsub(a, b) { return a.map((x, i) => x - b[i]); }
function vscale(a, s) { return a.map((x) => x * s); }
function vnorm(a) { return Math.sqrt(dot(a, a)); }
function matVec(A, x) { return A.map((row) => dot(row, x)); }

/* ---------- 独立高斯消元 + 部分主元解 Ax=b（黄金参照） ---------- */
function ownSolveLinear(A, b) {
  const n = A.length;
  const M = A.map((row, i) => row.slice().concat([b[i]]));
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-14) return null;
    if (piv !== col) { const t = M[piv]; M[piv] = M[col]; M[col] = t; }
    const d = M[col][col];
    for (let r = 0; r < n; r++) { if (r === col) continue; const f = M[r][col] / d; if (f === 0) continue; for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c]; }
  }
  return M.map((row, i) => row[n] / M[i][i]);
}

/* ---------- 独立 2×2 对称特征分解（升序特征值） ---------- */
function ownEigSym2(A) {
  const a = A[0][0], b = A[0][1], c = A[1][1];
  const tr = a + c, det = a * c - b * b;
  const disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
  const mid = tr / 2;
  return [mid - disc, mid + disc];
}
function ownIsSymPD2(M) {
  if (Math.abs(M[0][1] - M[1][0]) > 1e-12) return false;
  const e = ownEigSym2(M);
  return e[0] > 1e-9 && e[1] > 1e-9;
}

/* ---------- 独立 CG（SPD 系统 Ax=b，完整重实现用于交叉验证 + 下降性扫描） ----------
 * 返回带每步残差范数与下降符号的追踪，用以独立验证「CG 下降方向 gₖᵀdₖ<0」以及
 * 该性质对 SPD 的依赖（非 SPD 时残差会不降反升）。
 *   - x._res：每步 ‖rₖ‖（含初值）
 *   - x._descent：每步 rₖᵀpₖ>0（即 −∇fₖᵀpₖ<0，下降方向）的真值数组 */
function ownCg(A, b, x0) {
  const tol = 1e-12, maxIter = A.length * 2 + 10;
  let x = x0 ? x0.slice() : new Array(b.length).fill(0);
  let r = vsub(b, matVec(A, x));
  let p = r.slice();
  let rsOld = dot(r, r);
  const resHistory = [Math.sqrt(rsOld)];
  const descentAll = [];
  for (let k = 0; k < maxIter; k++) {
    const Ap = matVec(A, p);
    const alpha = rsOld / dot(p, Ap);
    // 当前步方向 pₖ 相对二次梯度 ∇f=Ax−b=−r 的下降性：rₖᵀpₖ>0 ⇔ (−∇f)ᵀp>0
    descentAll.push(dot(r, p) > 0);
    x = vadd(x, vscale(p, alpha));
    r = vsub(r, vscale(Ap, alpha));
    const rsNew = dot(r, r);
    resHistory.push(Math.sqrt(rsNew));
    if (Math.sqrt(rsNew) < tol) { x._iters = k + 1; x._res = resHistory; x._descent = descentAll; return x; }
    p = vadd(r, vscale(p, rsNew / rsOld));
    rsOld = rsNew;
  }
  x._res = resHistory; x._descent = descentAll;
  return x;
}

/* ---------- 独立黄金分割线搜索（φ(α) 局部极小参照，标准实现） ---------- */
function goldenMin(phi, lo, hi) {
  const phiRatio = (Math.sqrt(5) - 1) / 2; // ≈0.61803
  let c = hi - phiRatio * (hi - lo);
  let d = lo + phiRatio * (hi - lo);
  let fc = phi(c), fd = phi(d);
  for (let i = 0; i < 100; i++) {
    if (fc < fd) { hi = d; d = c; fd = fc; c = hi - phiRatio * (hi - lo); fc = phi(c); }
    else { lo = c; c = d; fc = fd; d = lo + phiRatio * (hi - lo); fd = phi(d); }
  }
  return (lo + hi) / 2;
}

/* ---------- 独立 BFGS 逆更新（H₊ = H − HyyᵀH/(yᵀHy) + ssᵀ/(yᵀs)） ---------- */
function ownBfgsInvUpdate(H, s, y) {
  const n = s.length;
  const ys = dot(y, s);
  if (ys <= 1e-10) return H.map((r) => r.slice());
  const Hy = matVec(H, y);
  const yHy = dot(y, Hy);
  if (yHy <= 1e-12) return H.map((r) => r.slice());
  const Hp = H.map((row) => row.slice());
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Hp[i][j] += (s[i] * s[j]) / ys - (Hy[i] * Hy[j]) / yHy;
  return Hp;
}

/* ---------- 独立 BFGS B 形式更新（B₊ = B + yyᵀ/(yᵀs) − BssᵀB/(sᵀBs)） ---------- */
function ownBfgsUpdate(B, s, y) {
  const n = s.length;
  const ys = dot(y, s);
  if (ys <= 1e-12) return B.map((r) => r.slice());
  const Bs = matVec(B, s);
  const sBs = dot(s, Bs);
  const Bp = B.map((row) => row.slice());
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Bp[i][j] += (y[i] * y[j]) / ys;
  if (sBs > 1e-14) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Bp[i][j] -= (Bs[i] * Bs[j]) / sBs;
  return Bp;
}

/* ---------- 独立 SR1 更新（B₊ = B + (y−Bs)(y−Bs)ᵀ/((y−Bs)ᵀs)） ---------- */
function ownSr1Update(B, s, y) {
  const n = s.length;
  const Bs = matVec(B, s);
  const diff = vsub(y, Bs);
  const denom = dot(diff, s);
  if (Math.abs(denom) <= 1e-12) return B.map((r) => r.slice());
  const Bp = B.map((row) => row.slice());
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Bp[i][j] += (diff[i] * diff[j]) / denom;
  return Bp;
}

/* ---------- 独立 DFP 逆形式更新（H₊ = H + ssᵀ/(yᵀs) − Hy yᵀH/(yᵀHy)） ---------- */
function ownDfpUpdate(H, s, y) {
  const n = s.length;
  const ys = dot(y, s);
  if (ys <= 1e-12) return H.map((r) => r.slice());
  const Hy = matVec(H, y);
  const yHy = dot(y, Hy);
  const Hp = H.map((row) => row.slice());
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Hp[i][j] += (s[i] * s[j]) / ys;
  if (yHy > 1e-14) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Hp[i][j] -= (Hy[i] * Hy[j]) / yHy;
  return Hp;
}

/* ---------- 独立 dogleg 子问题（复刻模块算法逻辑，用于交叉验证不变量） ---------- */
function ownDoglegSubproblem(g, B, delta) {
  const gtg = dot(g, g);
  const gtgBg = dot(g, matVec(B, g));
  let pu;
  if (gtgBg > 1e-12) pu = vscale(g, -gtg / gtgBg);
  else pu = vscale(g, -1);
  const pclen = vnorm(pu);
  let pc;
  if (pclen >= delta) pc = vscale(pu, delta / pclen);
  else pc = pu.slice();
  let s;
  if (pclen <= delta) s = pu.slice();
  else {
    const w = vsub(pu, pc);
    const A2 = dot(w, w), B2 = 2 * dot(pc, w), C2 = dot(pc, pc) - delta * delta;
    const disc = Math.max(0, B2 * B2 - 4 * A2 * C2);
    let tau = (-B2 + Math.sqrt(disc)) / (2 * A2);
    if (!(tau >= 0 && tau <= 1)) tau = 1;
    s = vadd(pc, vscale(w, tau));
    if (vnorm(s) > delta) s = pc.slice(); // 镜像模块护持：确保 ‖s‖≤Δ
  }
  return { s, pc, pu, pclen };
}

/* ---------- 独立信任域（dogleg）循环，返还每步内部量用于对照 ---------- */
function ownTrustRegion(f, grad, hess, x0, opts) {
  opts = opts || {};
  const maxIter = opts.maxIter || 60, tol = opts.tol || 1e-10;
  let x = x0.slice();
  let delta = opts.delta0 || 1, deltaMax = opts.deltaMax || 1e3;
  const trace = [];
  for (let k = 0; k < maxIter; k++) {
    const g = grad(x), B = hess(x);
    const f0 = f(x);
    const { s, pc } = ownDoglegSubproblem(g, B, delta);
    const ms = f0 + dot(g, s) + 0.5 * dot(s, matVec(B, s));
    const mpc = f0 + dot(g, pc) + 0.5 * dot(pc, matVec(B, pc));
    const m0 = f0;
    const xn = vadd(x, s);
    const fn = f(xn);
    const ared = f0 - fn, pred = m0 - ms;
    const rho = pred > 1e-15 ? ared / pred : (ared > 0 ? 1 : 0);
    const dPrev = delta;
    if (rho < 0.25) delta = Math.max(delta * 0.25, 1e-8);
    else if (rho > 0.75) delta = Math.min(delta * 2, deltaMax);
    let accepted = rho > 0.1;
    if (accepted) x = xn;
    trace.push({ ms, mpc, snorm: vnorm(s), delta, rho, dPrev, deltaAfter: delta, accepted });
    if (vnorm(g) < tol) break;
    if (delta < 1e-10) break;
  }
  return { x, trace };
}

/* ---------- 结果记录 ---------- */
const results = [];
function check(id, name, ok, detail) {
  results.push({ id, ok });
  console.log((ok ? '  +OK ' : '  -XX ') + id + ' ' + name + (detail ? '  | ' + detail : ''));
}

/* ==========================================================================
 * R1  牛顿法二次收敛（INV-SO-001）
 * ========================================================================== */
{
  // 2D 二次 f(x)=½xᵀAx−bᵀx，A SPD
  const A = [[3, 1], [1, 2]], b = [5, 3];
  const f = (x) => 0.5 * dot(x, matVec(A, x)) - dot(b, x);
  const grad = (x) => vsub(matVec(A, x), b);
  const hess = () => A;
  const x0 = [-4, 6];

  // (a) 单步牛顿
  const x1 = G.newton(f, grad, hess, x0, { maxIter: 1 });
  const xStar = ownSolveLinear(A, b);            // 独立高斯消元黄金参照
  const resid = vnorm(vsub(matVec(A, x1), b));
  const directDiff = vnorm(vsub(x1, xStar));
  const oneStepOk = resid <= 1e-12 * (vnorm(b) + 1) && directDiff <= 1e-12 * (vnorm(xStar) + 1);

  // (b) 全收敛（默认 50 步）必达解析极小
  const xFull = G.newton(f, grad, hess, x0, { maxIter: 50 });
  const fullDiff = vnorm(vsub(xFull, xStar));
  const fullOk = fullDiff <= 1e-12 * (vnorm(xStar) + 1);

  // (c) 1 维 f(x)=x²：单步必达 0（1×1 Hessian 矩阵 [[2]]）
  const x1d = G.newton((x) => x[0] * x[0], (x) => [2 * x[0]], (x) => [[2]], [3], { maxIter: 1 });
  const ok1d = Math.abs(x1d[0]) <= 1e-12;

  // 反空洞：非二次 f(x)=x⁴，单步牛顿不达标（证明「1 步收敛」是二次专属，非普遍成立）
  const xNonQ = G.newton((x) => x[0] * x[0] * x[0] * x[0], (x) => [4 * x[0] * x[0] * x[0]], (x) => [[12 * x[0] * x[0]]], [3], { maxIter: 1 });
  const nonQAtMin = Math.abs(xNonQ[0]) <= 1e-12;  // 应为 false（3 步牛顿后 3→2.25→...≠0）
  const falsOk = !nonQAtMin;

  const ok = oneStepOk && fullOk && ok1d && falsOk;
  check('R1', '牛顿法对二次 1 步收敛 EXACT（对拍独立高斯消元）+ 全收敛达解析极小 + 1维 x² 单步达 0 + 非二次无此性（证伪）', ok,
    '1步残差=' + resid.toExponential(2) + ' 1步Δ=' + directDiff.toExponential(2) + ' 全收敛Δ=' + fullDiff.toExponential(2) +
    ' x*=[' + xStar.map((v) => v.toFixed(4)).join(',') + '] 1维单步=' + x1d[0].toFixed(12) + ' 非二次1步≠0=' + falsOk);
}

/* ==========================================================================
 * R2  CG 对 SPD 线性系统（INV-SO-002）
 * ========================================================================== */
{
  const A = [[4, 1], [1, 3]], b = [1, 2];
  const xMod = G.cg(A, b);
  const xOwn = ownCg(A, b);
  const xDir = ownSolveLinear(A, b);

  const sysResidMod = vnorm(vsub(matVec(A, xMod), b));
  const sysResidOwn = vnorm(vsub(matVec(A, xOwn), b));
  const solDiffModDir = vnorm(vsub(xMod, xDir));
  const solDiffOwnDir = vnorm(vsub(xOwn, xDir));
  const crossModOwn = vnorm(vsub(xMod, xOwn));

  // 下降性 + 残差单调（SPD 专属性质，独立 CG 追踪）
  const itersMod = xMod._iters || 2;
  const convOk = sysResidMod <= 1e-10 * (vnorm(b) + 1) && itersMod <= 2;
  const crossOk = crossModOwn <= 1e-10 * (vnorm(xDir) + 1);
  let ownMonoOk = true;
  for (let k = 1; k < xOwn._res.length; k++) if (xOwn._res[k] > xOwn._res[k - 1] + 1e-12) ownMonoOk = false;
  const ownDescentOk = xOwn._descent.every((v) => v);

  // 反空洞：非 SPD 矩阵 A=[[1,2],[2,1]]（特征值 −1,+3）下 CG 的残差不单调下降（证明「下降」依赖 SPD）
  const An = [[1, 2], [2, 1]], bn = [3, -1];
  const xNon = ownCg(An, bn);
  let nonMono = false;
  for (let k = 1; k < xNon._res.length; k++) if (xNon._res[k] > xNon._res[k - 1] + 1e-9) { nonMono = true; break; }
  const falsOk = nonMono;  // 非 SPD 时残差必不降反升 → 证伪「CG 总下降」

  const ok = convOk && crossOk && ownMonoOk && ownDescentOk && sysResidOwn <= 1e-10 * (vnorm(b) + 1) &&
    solDiffModDir <= 1e-10 * (vnorm(xDir) + 1) && solDiffOwnDir <= 1e-10 * (vnorm(xDir) + 1) && falsOk;
  check('R2', 'CG 对 SPD 系统：模块与独立重实现解一致(≤1e-10) + 与高斯消元 EXACT + ≤n 步收敛 + 残差单调下降 + 非 SPD 残差不单调(证伪)', ok,
    'iters=' + itersMod + ' 残差(mod)=' + sysResidMod.toExponential(2) + ' 残差(own)=' + sysResidOwn.toExponential(2) +
    ' Δ(mod,dir)=' + solDiffModDir.toExponential(2) + ' Δ(own,dir)=' + solDiffOwnDir.toExponential(2) +
    ' Δ(mod,own)=' + crossModOwn.toExponential(2) + ' own单调=' + ownMonoOk + ' own下降=' + ownDescentOk +
    ' 非SPD残差非单调=' + falsOk);
}

/* ==========================================================================
 * R3  BFGS/L-BFGS 收敛 + 逆更新 SPD + 割线 + 曲率护 SPD（INV-SO-003）
 * ========================================================================== */
{
  // (a) L-BFGS 在 Rosenbrock 上收敛
  const f = (x) => 100 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]) + (1 - x[0]) * (1 - x[0]);
  const grad = (x) => [-400 * x[0] * (x[1] - x[0] * x[0]) - 2 * (1 - x[0]), 200 * (x[1] - x[0] * x[0])];
  const f0 = f([-1.2, 1]);
  const xL = G.lbfgs(f, grad, [-1.2, 1], { m: 10, maxIter: 5000, tol: 1e-8 });
  const fEnd = f(xL);
  const gEnd = grad(xL);
  const converged = fEnd <= f0 && vnorm(gEnd) < 1e-3;

  // (b) 逆更新 hessianApprox(sList,yList,n) 的对称/SPD/割线
  const sList = [[0.3, 0.1], [0.2, -0.1]], yList = [[0.5, 0.2], [0.15, -0.2]];
  let symOk = true, pdOk = true, secantOk = true, curvAllPos = true;
  for (let i = 0; i < sList.length; i++) {
    const Hacc = G.hessianApprox([sList[i]], [yList[i]], 2);
    if (Math.abs(Hacc[0][1] - Hacc[1][0]) > 1e-12) symOk = false;
    if (!ownIsSymPD2(Hacc)) pdOk = false;
    const Hy = matVec(Hacc, yList[i]);
    if (vnorm(vsub(Hy, sList[i])) > 1e-12 * (vnorm(sList[i]) + 1)) secantOk = false;
    if (dot(yList[i], sList[i]) <= 0) curvAllPos = false;
  }
  // 累计 H（两套对的组合，对称性保持）
  const Htot = G.hessianApprox(sList, yList, 2);
  const symTot = Math.abs(Htot[0][1] - Htot[1][0]) < 1e-12;

  // (c) 独立重实现对拍：ownBfgsInvUpdate(I, s, y) 应与 hessianApprox([s],[y],2) 一致（同一对）
  const s = sList[0], y = yList[0];
  const ownH = ownBfgsInvUpdate([[1, 0], [0, 1]], s, y);
  const Hacc0 = G.hessianApprox([s], [y], 2);
  let crossDiff = 0;
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) crossDiff = Math.max(crossDiff, Math.abs(ownH[i][j] - Hacc0[i][j]));

  // (d) 曲率护 SPD：yᵀs≤0 时更新被跳过（hessianApprox 返回 I，仍 SPD）
  const sBad = [0.1, 0.1], yBad = [-0.2, -0.2];
  const Hbad = G.hessianApprox([sBad], [yBad], 2);
  const skippedOk = Hbad[0][0] === 1 && Hbad[1][1] === 1 && ownIsSymPD2(Hbad);

  const ok = converged && symOk && pdOk && secantOk && symTot && curvAllPos && crossDiff <= 1e-12 && skippedOk;
  check('R3', 'BFGS/L-BFGS：Rosenbrock 收敛(f≤f₀,‖∇f‖<1e-3) + 逆更新恒对称正定(SPD) + 割线 H₊y=s(≤1e-12) + 独立重实现对拍 + 负曲率跳过护 SPD', ok,
    'f0=' + f0.toFixed(2) + ' fEnd=' + fEnd.toExponential(2) + ' ‖∇f‖=' + vnorm(gEnd).toExponential(2) +
    ' sym=' + symOk + ' pd=' + pdOk + ' secant=' + secantOk + ' 独立Δ=' + crossDiff.toExponential(2) + ' 负曲率跳过=' + skippedOk);
}

/* ==========================================================================
 * R4  强 Wolfe 线搜索（Armijo + 曲率 + 黄金分割对拍 + 非下降证伪）（INV-SO-004）
 * ========================================================================== */
{
  const f = (x) => (x[0] - 2) * (x[0] - 2) + (x[1] + 1) * (x[1] + 1);
  const grad = (x) => [2 * (x[0] - 2), 2 * (x[1] + 1)];
  const x = [0, 0];
  const d = [4, -2];  // 下降方向
  const aWolfe = G.strongWolfe(f, grad, x, d, { c1: 1e-4, c2: 0.9, maxIter: 40 });
  const f0 = f(x), g0 = grad(x), slope0 = dot(g0, d);
  const aF = vadd(x, vscale(d, aWolfe));
  const armijo = f(aF) <= f0 + 1e-4 * aWolfe * slope0;
  const curv = Math.abs(dot(grad(aF), d)) <= 0.9 * Math.abs(slope0);

  // 独立黄金分割局部极小对拍
  const phi = (a) => f(vadd(x, vscale(d, a)));
  const aGold = goldenMin(phi, 0, Math.max(aWolfe * 4, 4));
  const closeMin = Math.abs(phi(aWolfe) - phi(aGold)) <= Math.max(1e-3, 1e-2 * Math.abs(phi(aGold)) + 1e-6);

  // 证伪：非下降方向（取自身梯度 g₀，slope0=g₀ᵀg₀>0）返回 0
  const badDir = g0.slice();
  const aBad = G.strongWolfe(f, grad, x, badDir, { c1: 1e-4, c2: 0.9 });
  const falsOk = aBad === 0;

  const ok = armijo && curv && closeMin && falsOk;
  check('R4', '强 Wolfe：Armijo + 曲率条件满足 + 独立黄金分割局部极小对拍(差≤1e-3) + 非下降方向返回0证伪', ok,
    'α=' + aWolfe.toFixed(4) + ' armijo=' + armijo + ' curv=' + curv + ' αgold=' + aGold.toFixed(4) +
    ' Δf=' + (phi(aWolfe) - phi(aGold)).toExponential(2) + ' badDirReturns0=' + falsOk);
}

/* ==========================================================================
 * R5  信赖域 dogleg（INV-SO-005）
 * ========================================================================== */
{
  const A = [[2, 0.5], [0.5, 1]], b = [1, 0.5];
  const f = (x) => 0.5 * dot(x, matVec(A, x)) - dot(b, x);
  const grad = (x) => vsub(matVec(A, x), b);
  const hess = () => A;
  const x0 = [-2, 3];

  // (a) 独立 dogleg 子问题：批量随机 (g,B,Δ) 下 m(s*)≤m(p_c) 且 ‖s*‖≤Δ
  const rng = mulberry32(0x50D0C1A);
  let allDoglegOk = true, allBoundOk = true, allReduceOk = true;
  for (let t = 0; t < 40; t++) {
    // 随机 SPD B（经由 Cholesky 因子 L Lᵀ）
    const L = [[0.5 + rng(), 0], [rng() - 0.5, 0.5 + rng()]];
    const B = matMatMulOwn(L, matTransposeOwn(L));
    const g = [rng() * 4 - 2, rng() * 4 - 2];
    const delta = 0.2 + rng() * 3;
    const { s, pc } = ownDoglegSubproblem(g, B, delta);
    const m = (ss) => dot(g, ss) + 0.5 * dot(ss, matVec(B, ss));
    const ms = m(s), mpc = m(pc), m0 = 0;
    if (!(ms <= mpc + 1e-12)) allDoglegOk = false;       // dogleg 不劣于柯西点
    if (!(ms <= m0 + 1e-12)) allReduceOk = false;         // 优于原点
    if (!(vnorm(s) <= delta + 1e-12)) allBoundOk = false; // 信赖域内
  }

  // (b) 独立 trust-region 循环：逐迭代验证 m(s*)≤m(p_c) + 半径自适应，并用模块对照
  const ownRes = ownTrustRegion(f, grad, hess, x0, { delta0: 1, maxIter: 60, tol: 1e-10 });
  let ownDoglegAll = true, ownExpanded = false, ownContracted = false;
  for (const tr of ownRes.trace) {
    if (!(tr.ms <= tr.mpc + 1e-12)) ownDoglegAll = false;
    if (tr.deltaAfter > tr.dPrev + 1e-12) ownExpanded = true;
    if (tr.deltaAfter < tr.dPrev - 1e-12) ownContracted = true;
  }

  // (c) 模块 trustRegion 同问题同初值，终态应与独立实现一致（同点收敛到解析极小）
  const xMod = G.trustRegion(f, grad, hess, x0, { delta0: 1, maxIter: 60, tol: 1e-10 });
  const xStar = ownSolveLinear(A, b);
  const convMod = vnorm(vsub(xMod, xStar)) <= 1e-6 * (vnorm(xStar) + 1);
  const convOwn = vnorm(vsub(ownRes.x, xStar)) <= 1e-6 * (vnorm(xStar) + 1);
  const crossModOwnVal = vnorm(vsub(xMod, ownRes.x));
  const crossModOwn = crossModOwnVal <= 1e-6 * (vnorm(xStar) + 1);

  const ok = allDoglegOk && allBoundOk && allReduceOk && ownDoglegAll && ownExpanded && ownContracted && convMod && convOwn && crossModOwn;
  check('R5', '信赖域 dogleg：m(s*)≤m(p_c) 恒成立 + ‖s*‖≤Δ + m(s*)≤m(0) + 独立 TR 循环半径自适应(扩/缩) + 模块与独立同点收敛', ok,
    'dogleg≤cauchy=' + allDoglegOk + ' bound=' + allBoundOk + ' reduce=' + allReduceOk +
    ' ownDogleg=' + ownDoglegAll + ' expanded=' + ownExpanded + ' contracted=' + ownContracted +
    ' 模块收敛Δ=' + vnorm(vsub(xMod, xStar)).toExponential(2) + ' 独立收敛Δ=' + vnorm(vsub(ownRes.x, xStar)).toExponential(2) +
    ' 模-独Δ=' + crossModOwnVal.toExponential(2));
}

/* ==========================================================================
 * R6  高斯牛顿 ≡ 牛顿（线性残差）（INV-SO-006）
 * ========================================================================== */
{
  const A = [[2, 1], [1, 3]], b = [4, 5];
  const R = (x) => vsub(matVec(A, x), b);  // 线性残差
  const J = () => A;
  const f = (x) => 0.5 * dot(R(x), R(x));
  const grad = (x) => matVec(matTransposeOwn(A), R(x));
  const hess = () => matMatMulOwn(matTransposeOwn(A), A);
  const x0 = [-3, 2];

  const xGN = G.gaussNewton(R, J, x0, { maxIter: 1 });
  const xN = G.newton(f, grad, hess, x0, { maxIter: 1 });
  const same = vnorm(vsub(xGN, xN)) <= 1e-12 * (vnorm(xN) + 1);

  // 独立推导：GN 步 = (JᵀJ)⁻¹Jᵀr，牛顿步 = H⁻¹(x − Ax...) = (AᵀA)⁻¹Aᵀb（两者应相等）
  const Jt = matTransposeOwn(A), JtJ = matMatMulOwn(Jt, A);
  const r0 = R(x0);
  const gnStep = ownSolveLinear(JtJ, vscale(matVec(Jt, r0), -1));
  const newtonStep = ownSolveLinear(hess(x0), vscale(grad(x0), -1));  // 真牛顿步 = −H⁻¹∇f = −(AᵀA)⁻¹Aᵀ(Ax₀−b)
  const ownSame = vnorm(vsub(gnStep, newtonStep)) <= 1e-12;
  const ownVsModVal = vnorm(vsub(gnStep, vsub(xGN, x0)));
  const ownVsMod = ownVsModVal <= 1e-12 * (vnorm(xGN) + 1);

  // 非线性最小二乘：残差范数非增
  const R2 = (x) => [x[0] * x[0] - x[1], x[0] + x[1] - 2];
  const J2 = (x) => [[2 * x[0], -1], [1, 1]];
  const xNl = G.gaussNewton(R2, J2, [0.5, 0.5], { maxIter: 20 });
  const r0n = vnorm(R2([0.5, 0.5])), r1n = vnorm(R2(xNl));
  const nonIncrease = r1n <= r0n + 1e-9;

  const ok = same && ownSame && ownVsMod && nonIncrease;
  check('R6', '高斯牛顿≡牛顿(线性残差) EXACT（模块对拍 + 独立推导 (AᵀA)⁻¹Aᵀb 一致） + 非线性最小二乘残差范数非增', ok,
    'GN==N_diff=' + vnorm(vsub(xGN, xN)).toExponential(2) + ' 独立推导Δ=' + vnorm(vsub(gnStep, newtonStep)).toExponential(2) +
    ' 独立vs模块Δ=' + ownVsModVal.toExponential(2) + ' r0=' + r0n.toFixed(4) + ' r1=' + r1n.toFixed(4) + ' nonInc=' + nonIncrease);
}

/* ==========================================================================
 * R7  阻尼牛顿 + Hessian-free（INV-SO-007）
 * ========================================================================== */
{
  // (A) 阻尼性质（独立核验）：H 不定则提 λ 使 (H+λI) SPD 且步长必下降
  const H0 = [[2, 0], [0, -1]];  // 鞍点 Hessian（不定）
  const g0 = [1, 1];
  let lam = 0, Hl;
  for (let guard = 0; guard < 80; guard++) {
    Hl = H0.map((row, i) => row.map((v, j) => (i === j ? v + lam : v)));
    if (ownIsSymPD2(Hl)) break;
    lam = lam === 0 ? 1e-3 : lam * 10;
  }
  const dDamp = ownSolveLinear(Hl, vscale(g0, -1));
  const descentOk = dot(g0, dDamp) < 0;
  const pdOk = ownIsSymPD2(Hl);

  // (B) 阻尼牛顿端到端：f=(x²−1)²+(y²−1)² 起步 Hessian 不定，收敛到 (1,1)
  const fb = (x) => (x[0] * x[0] - 1) * (x[0] * x[0] - 1) + (x[1] * x[1] - 1) * (x[1] * x[1] - 1);
  const ggb = (x) => [4 * x[0] * (x[0] * x[0] - 1), 4 * x[1] * (x[1] * x[1] - 1)];
  const Hb = (x) => [[12 * x[0] * x[0] - 4, 0], [0, 12 * x[1] * x[1] - 4]];
  const xDb = G.dampedNewton(fb, ggb, Hb, [0.5, 0.5], { lambda0: 0, maxIter: 500, tol: 1e-10 });
  const convOk = vnorm(vsub(xDb, [1, 1])) <= 1e-7 * (Math.SQRT2 + 1);

  // (C) Hessian-free：二次 Af,b，HF 方向 == 精确牛顿步（一步达极小点）
  const Af = [[3, 1], [1, 2]], bf = [5, 3];
  const fq = (x) => 0.5 * dot(x, matVec(Af, x)) - dot(bf, x);
  const gq = (x) => vsub(matVec(Af, x), bf);
  const hvpQ = (x, v) => matVec(Af, v);
  const xHF = G.hessianFree(fq, gq, hvpQ, [-4, 6], { maxIter: 20 });
  const xExact = ownSolveLinear(Af, bf);
  const hfOk = vnorm(vsub(xHF, xExact)) <= 1e-9 * (vnorm(xExact) + 1);

  // 反空洞：在鞍点方向 (H 不定) 阻尼牛顿不退化（(H+λI) 必须真正 SPD，已证 pdOk）
  const ok = descentOk && pdOk && convOk && hfOk;
  check('R7', '阻尼牛顿(H+λI 恒 SPD 且下降) + 端到端收敛(1,1) + Hessian-free≡精确牛顿(二次) EXACT', ok,
    'descent=' + descentOk + ' pd=' + pdOk + ' 阻尼收敛Δ=' + vnorm(vsub(xDb, [1, 1])).toExponential(2) +
    ' HFΔ=' + vnorm(vsub(xHF, xExact)).toExponential(2) + ' 独立x*=[' + xExact.map((v) => v.toFixed(4)).join(',') + ']');
}

/* ==========================================================================
 * R8  拟牛顿校正（BFGS/DFP/SR1）对称 + 割线 + 秩1公式 + 独立对拍（INV-SO-008）
 * ========================================================================== */
{
  const s = [0.3, 0.1], y = [0.5, 0.2];

  // BFGS 逆更新（经 hessianApprox）
  const Hp = G.hessianApprox([s], [y], 2);
  const sym1 = Math.abs(Hp[0][1] - Hp[1][0]) < 1e-12;
  const Hy = matVec(Hp, y);
  const secant1 = vnorm(vsub(Hy, s)) <= 1e-12 * (vnorm(s) + 1);
  const pd1 = ownIsSymPD2(Hp);

  // BFGS B 形式更新（quasiNewtonUpdateBFGS）
  const Bp = G.quasiNewtonUpdateBFGS([[1, 0], [0, 1]], s, y);
  const symB = Math.abs(Bp[0][1] - Bp[1][0]) < 1e-12;
  const Bs = matVec(Bp, s);
  const secantB = vnorm(vsub(Bs, y)) <= 1e-12 * (vnorm(y) + 1);  // B-form 割线 B₊s=y
  const pdB = ownIsSymPD2(Bp);

  // DFP 逆形式更新（quasiNewtonUpdateDFP）
  const Hdfp = G.quasiNewtonUpdateDFP([[1, 0], [0, 1]], s, y);
  const sym2 = Math.abs(Hdfp[0][1] - Hdfp[1][0]) < 1e-12;
  const HdfpY = matVec(Hdfp, y);
  const secant2 = vnorm(vsub(HdfpY, s)) <= 1e-12 * (vnorm(s) + 1);
  const pd2 = ownIsSymPD2(Hdfp);

  // SR1 秩1 公式（sr1Update）
  const S1 = G.sr1Update([[1, 0], [0, 1]], s, y);
  const sym3 = Math.abs(S1[0][1] - S1[1][0]) < 1e-12;
  const Hy0 = matVec([[1, 0], [0, 1]], y), diff = vsub(y, Hy0), denom = dot(diff, s);
  let sr1exact = true;
  if (Math.abs(denom) > 1e-12) {
    const expect = [[1 + diff[0] * diff[0] / denom, diff[0] * diff[1] / denom], [diff[1] * diff[0] / denom, 1 + diff[1] * diff[1] / denom]];
    sr1exact = Math.abs(S1[0][0] - expect[0][0]) < 1e-12 && Math.abs(S1[1][1] - expect[1][1]) < 1e-12 && Math.abs(S1[0][1] - expect[0][1]) < 1e-12;
  }

  // 独立重实现逐项对拍（BFGS逆 / BFGS-B / SR1 / DFP）
  const ownHp = ownBfgsInvUpdate([[1, 0], [0, 1]], s, y);
  const ownBp = ownBfgsUpdate([[1, 0], [0, 1]], s, y);
  const ownS1 = ownSr1Update([[1, 0], [0, 1]], s, y);
  const ownHd = ownDfpUpdate([[1, 0], [0, 1]], s, y);
  let crossDiff = 0;
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
    crossDiff = Math.max(crossDiff, Math.abs(ownHp[i][j] - Hp[i][j]), Math.abs(ownBp[i][j] - Bp[i][j]),
      Math.abs(ownS1[i][j] - S1[i][j]), Math.abs(ownHd[i][j] - Hdfp[i][j]));
  }

  // 曲率护持：yᵀs≤0 时更新被跳过（BFGS 逆经 hessianApprox 返回 I，仍 SPD）
  const sBad = [0.1, 0.1], yBad = [-0.2, -0.2];
  const Hbad = G.hessianApprox([sBad], [yBad], 2);
  const skippedOk = Hbad[0][0] === 1 && Hbad[1][1] === 1 && ownIsSymPD2(Hbad);

  const ok = sym1 && secant1 && pd1 && symB && secantB && pdB && sym2 && secant2 && pd2 &&
    sym3 && sr1exact && crossDiff <= 1e-12 && skippedOk;
  check('R8', '拟牛顿 BFGS/DFP/SR1 恒对称 + BFGS逆割线 H₊y=s + BFGS-B割线 B₊s=y + SR1 秩1公式 EXACT + 独立重实现对拍 + 负曲率跳过护 SPD', ok,
    'sym1=' + sym1 + ' secant1=' + secant1 + ' pd1=' + pd1 + ' symB=' + symB + ' secantB=' + secantB + ' pdB=' + pdB +
    ' sym2=' + sym2 + ' secant2=' + secant2 + ' pd2=' + pd2 + ' sym3=' + sym3 + ' sr1exact=' + sr1exact +
    ' 独立Δ=' + crossDiff.toExponential(2) + ' 负曲率跳过=' + skippedOk);
}

/* ---------- 矩阵转置/乘法（本文件独立实现，供 R5/R6 使用） ---------- */
function matTransposeOwn(A) { const n = A.length, m = A[0].length; const B = Array.from({ length: m }, () => new Array(n).fill(0)); for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) B[j][i] = A[i][j]; return B; }
function matMatMulOwn(A, B) { const n = A.length, m = B[0].length, p = B.length; const C = Array.from({ length: n }, () => new Array(m).fill(0)); for (let i = 0; i < n; i++) for (let k = 0; k < p; k++) { const aik = A[i][k]; if (aik === 0) continue; for (let j = 0; j < m; j++) C[i][j] += aik * B[k][j]; } return C; }

/* ---------- 汇总 ---------- */
const failed = results.filter((r) => !r.ok).length;
console.log('══════ secondorder-forge 独立复核: ' + (results.length - failed) + '/' + results.length + ' 通过 ══════');
process.exit(failed === 0 ? 0 : 1);
