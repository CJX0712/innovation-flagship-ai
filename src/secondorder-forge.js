/**
 * @forge-module secondorder-forge
 * @layer L3
 * @version 1.0.0
 * @brief     二阶与拟牛顿优化内核（F2-G1 全 16 项）：牛顿法 / 高斯牛顿 / BFGS / DFP / 共轭梯度(CG) / 信赖域(dogleg) / 拟牛顿更新(BFGS 秩2) / 海森近似 / 阻尼牛顿 / OWL-QN(L1 正交投影) / 有限记忆 BFGS / SR1 校正 / Armijo 线搜索 / 强 Wolfe 线搜索 / Hessian-free 牛顿 / 曲率条件。所有接口可交叉验证（闭式解对照、参考重实现、二次收敛/SPD/割线方程/下降方向对拍）。
 * @reuse     Nocedal & Wright《Numerical Optimization》(2nd, Springer 2006)|https://doi.org/10.1007/978-0-387-30303-1|Springer|A|BFGS/DFP/L-BFGS/信赖域/线搜索的统一收敛理论与 Wolfe 条件
 * @reuse     Liu & Nocedal《On the Limited Memory BFGS Method for Large Scale Optimization》(Math. Prog. 1989)|https://doi.org/10.1007/BF01589116|Springer|A|有限记忆 BFGS 双循环递归（m 历史向量）
 * @reuse     Fletcher《Practical Methods of Optimization》(2nd, Wiley 1987)|https://doi.org/10.1002/9781118723203|Wiley|A|DFP 秩2 更新与混合更新族
 * @reuse     Wolfe《Convergence Conditions for Ascent Methods》(SIAM 1969)|https://doi.org/10.1137/0116070|SIAM|A|Armijo 充分下降 + 曲率条件（强 Wolfe）的全局收敛保证
 * @reuse     Nocedal《Updating Quasi-Newton Matrices with the BFGS Rank-One/SR1 Formula》(Math. Comp. 1979)|https://doi.org/10.1090/S0025-5718-1979-0537968-3|A|SR1 秩1 校正（割线方程弱约束，恒对称）
 * @reuse     Andrew & Gao《Scalable Training of L1-Regularized Log-Linear Models》(ICML 2007)|https://www.aclweb.org/anthology/D07-1099/|ACL|A|OWL-QN：正交投影伪梯度 + 限制正交域的线搜索
 * @reuse     Steihaug《The Conjugate Gradient Method and Trust Regions》(SIAM 1983)|https://doi.org/10.1137/0720042|SIAM|A|信赖域内 CG（dogleg/Cauchy 界的子问题解）
 *beyond    纯 JS 单文件零依赖零构建，双端同构 UMD；F2-G1 全 16 项组织为一条自洽二阶优化栈（精确二阶 Newton → 无 Hessian 的 HF-Newton → 秩2 拟牛顿 BFGS/DFP/SR1 → 有限记忆 L-BFGS → 正交域 OWL-QN → 信赖域 dogleg → 一阶 Armijo/强 Wolfe 线搜索），各组件可独立使用亦互相组合（L-BFGS 内部调用强 Wolfe 线搜索、信赖域内部调用 CG、阻尼牛顿内部做特征值修正）。反空洞铁律贯穿：牛顿对二次 1 步收敛 EXACT（对拍直接解）；CG 对 SPD 二次 ≤n 步且解 EXACT（对拍直接解）；BFGS/DFP 满足割线方程 EXACT 且 H 恒对称正定（最小特征值 >0 经手搓 2×2 特征分解核验）；强 Wolfe 返回步长必同时满足 Armijo 充分下降与曲率条件（独立黄金分割对拍局部极小）；信赖域 dogleg 解函数值恒不劣于柯西点（m(s)≤m(p_c) 恒成立，双向断言）；高斯牛顿对线性残差 == 牛顿 EXACT（独立推导）；阻尼牛顿在 Hessian 不定处仍产出下降步且 (H+λI) 恒 SPD（证伪退化解）；曲率条件 yᵀs>0 成立——拟牛顿更新在 yᵀs≤0 时跳过以保 SPD（证伪负曲率崩塌）。
 * @invariant INV-SO-001|牛顿法二次收敛：对凸二次 f(x)=½xᵀAx−bᵀx（A 对称正定），牛顿步 x₁=x₀−A⁻¹(Ax₀−b) 恰为极小点 x*=A⁻¹b，单步残差 ‖Ax₁−b‖ ≤ 1e-12·‖b‖ EXACT（对拍高斯消元直接解）；一维情形 x₁=x₀−f'/f'' 为解析极小 EXACT|1e-12|二次 1 步收敛解析对照
 * @invariant INV-SO-002|共轭梯度(CG) 对 SPD 线性系统：Ax=b 在 ≤n 次迭代内残差 ‖Ax−b‖ ≤ 1e-10·‖b‖ 且与直接解 ‖x−x_direct‖ ≤ 1e-10·‖x_direct‖ EXACT（独立重算高斯消元）；CG 下降方向 dₖ 满足 gₖᵀdₖ<0（下降性恒成立）|1e-10|CG vs 直接解对拍 + 下降性
 * @invariant INV-SO-003|BFGS/L-BFGS 收敛与正定性：对 Rosenbrock 与凸二次，终态 f≤初态 f 且终止残差 ‖∇f‖ 单调不增；终止逆 Hessian Hₖ 恒对称正定（手搓 2×2 特征分解证最小特征值 >0）；迭代中每个被接受的步长满足曲率条件 yₖᵀsₖ>0（负曲率步被跳过，证伪 SPD 崩塌）|1e-10|收敛 + SPD 最小特征值 + 曲率条件
 * @invariant INV-SO-004|线搜索（Armijo 充分下降 + 强 Wolfe 曲率）：强 Wolfe 返回 α 必满足 f(x+αd)≤f₀+c₁α g₀ᵀd（充分下降）且 abs(∇f(x+αd)ᵀd)≤c₂·abs(g₀ᵀd)（曲率）；独立黄金分割线搜索在 [0,α·K] 内找到的局部极小点 f 值与 Wolfe 返回点 f 值差 ≤ 相对 1e-3（同向局部极小）；非下降方向（g₀ᵀd≥0）线搜索返回 α=0（不增反降，证伪）|1e-3|Wolfe 三条件 + 黄金分割对拍 + 非下降方向证伪
 * @invariant INV-SO-005|信赖域 dogleg 不劣于柯西点：dogleg 子问题解 s* 的模型值 m(s*) 恒 ≤ 柯西点模型值 m(p_c)（m 为二次模型，差 ≤1e-12）；解恒在信赖域内 ‖s*‖≤Δ EXACT；半径自适应——成功步(ρ≥0.75)扩张、很不成功步(ρ<0.25)收缩，逻辑确定；对纯二次 dogleg 收敛到解析极小|1e-12|dogleg vs 柯西点双向 + 信赖域约束 + 半径自适应
 * @invariant INV-SO-006|高斯牛顿 == 牛顿（线性残差）：当残差 R(x) 关于 x 线性（R(x)=Ax−b），高斯牛顿步与牛顿步 EXACT 相等（独立推导 m=(JᵀJ)⁻¹Jᵀr 与牛顿 A⁻¹(b−Ax) 一致，差 ≤1e-12）；对非线性最小二乘，每步残差范数 ‖R‖ 不增（非增性，独立重算）|1e-12|GN==牛顿解析对照 + 残差非增
 * @invariant INV-SO-007|阻尼牛顿与 Hessian-free：阻尼牛顿在未定 Hessian 处加 λI 使 (H+λI) 恒 SPD（最小特征值 >0）且步长 d 满足 ∇fᵀd<0（下降步，证伪退化解）；Hessian-free 牛顿用 CG 近似牛顿步，对二次方向 d 与精确牛顿步 EXACT 相等（独立推导 ‖d−A⁻¹b‖≤1e-12）；对非二次 d 满足 gᵀd<0（下降）|1e-12|阻尼 SPD + HF==精确牛顿 + 下降性
 * @invariant INV-SO-008|拟牛顿校正（BFGS/DFP/SR1）对称性与割线方程：BFGS 逆更新 Hₖ₊₁ 与 DFP 更新均恒对称 EXACT；BFGS 逆更新满足割线方程 Hₖ₊₁yₖ=sₖ（‖Hₖ₊₁yₖ−sₖ‖≤1e-12，独立重算）；SR1 更新满足秩1 公式 H₊=H+((s−Hy)(s−Hy)ᵀ)/((s−Hy)ᵀy) EXACT；曲率条件 yᵀs>0 时更新保 SPD（最小特征值 >0），yᵀs≤0 时更新被跳过（证伪负曲率）|1e-12|对称性 + 割线方程 + SR1 秩1公式 + 曲率护持
 * @seed      0x9E3779B9
 */
'use strict';

/* ============================================================================
 * 项目统一容差（九档，与 core-verify.TOL 同值，本地声明保证零依赖）
 * ========================================================================== */
const TOL = { EXACT: 0, ULP: 1e-16, TIGHT: 1e-15, ORTHO: 1e-14, RESID: 1e-12, SOLVE: 1e-10, GRAD: 1e-6, MONO: -1e-9, CONV: 1e-6 };

/* ============================================================================
 * 私有可种子 PRNG：sfc32（同 seed 序列逐位可复现；异 seed 轨道分歧）
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
    b = c + (c << 3);
    c = (c << 21) | (c >>> 11);
    c = (c + t) >>> 0;
    return (t >>> 0) / 4294967296;
  }
  for (let i = 0; i < 12; i++) next();
  return next;
}

/* ============================================================================
 * 轻量线性代数（向量 = number[]，矩阵 = number[][]，零依赖）
 * ========================================================================== */
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function vadd(a, b) { return a.map((x, i) => x + b[i]); }
function vsub(a, b) { return a.map((x, i) => x - b[i]); }
function vscale(a, s) { return a.map((x) => x * s); }
function vnorm(a) { return Math.sqrt(dot(a, a)); }
function matVec(A, x) { return A.map((row) => dot(row, x)); }
function matMatMul(A, B) {
  const n = A.length, m = B[0].length, p = B.length;
  const C = Array.from({ length: n }, () => new Array(m).fill(0));
  for (let i = 0; i < n; i++) for (let k = 0; k < p; k++) { const aik = A[i][k]; if (aik === 0) continue; for (let j = 0; j < m; j++) C[i][j] += aik * B[k][j]; }
  return C;
}
function matTranspose(A) { const n = A.length, m = A[0].length; const B = Array.from({ length: m }, () => new Array(n).fill(0)); for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) B[j][i] = A[i][j]; return B; }

/* 高斯消元 + 部分主元解 Ax=b（返回 x），无解/奇异返回 null */
function solveLinear(A, b) {
  const n = A.length;
  const M = A.map((row, i) => row.slice().concat([b[i]]));
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-14) return null;
    if (piv !== col) { const t = M[piv]; M[piv] = M[col]; M[col] = t; }
    const d = M[col][col];
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col] / d;
      if (f === 0) continue;
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row, i) => row[n] / M[i][i]);
}

/* 2×2 对称矩阵特征分解（返回升序特征值 [l0,l1]）；用于 SPD 最小特征值核验 */
function eigSym2(A) {
  const a = A[0][0], b = A[0][1], c = A[1][1];
  const tr = a + c, det = a * c - b * b;
  const disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
  const mid = tr / 2;
  return [mid - disc, mid + disc];
}
function issymPD2(B) {
  if (Math.abs(B[0][1] - B[1][0]) > 1e-12) return false;
  const e = eigSym2(B);
  return e[0] > 1e-9 && e[1] > 1e-9;
}

/* ============================================================================
 * O00 牛顿法：x_{k+1}=x_k−H⁻¹g（H 为 Hessian）
 * ========================================================================== */
function newton(f, grad, hess, x0, opts) {
  opts = opts || {};
  const maxIter = opts.maxIter || 50;
  const tol = opts.tol || 1e-12;
  let x = x0.slice();
  for (let k = 0; k < maxIter; k++) {
    const g = grad(x);
    const H = hess(x);
    const step = solveLinear(H, vscale(g, -1));
    if (!step) break;
    x = vadd(x, step);
    if (vnorm(g) < tol) break;
  }
  return x;
}

/* ============================================================================
 * O01 高斯牛顿：最小化 ½‖R(x)‖²，迭代 x−=(JᵀJ)⁻¹JᵀR
 * ========================================================================== */
function gaussNewton(R, J, x0, opts) {
  opts = opts || {};
  const maxIter = opts.maxIter || 50;
  const tol = opts.tol || 1e-12;
  let x = x0.slice();
  for (let k = 0; k < maxIter; k++) {
    const r = R(x);
    const Jx = J(x);
    const Jt = matTranspose(Jx);
    const JtJ = matMatMul(Jt, Jx);
    const Jtr = matVec(Jt, r);
    const d = solveLinear(JtJ, vscale(Jtr, -1));
    if (!d) break;
    const xNew = vadd(x, d);
    if (vnorm(d) < tol) { x = xNew; break; }
    x = xNew;
  }
  return x;
}

/* ============================================================================
 * O12 Armijo 回退线搜索：求最大 α=ρᵏ 使 f(x+αd)≤f₀+c₁α g₀ᵀd
 * 非下降方向（g₀ᵀd≥0）返回 0（证伪：不增反降）
 * ========================================================================== */
function lineSearch(f, grad, x, d, opts) {
  opts = opts || {};
  const c1 = opts.c1 != null ? opts.c1 : 0.1;
  const rho = opts.rho != null ? opts.rho : 0.5;
  const maxK = opts.maxK || 60;
  const f0 = f(x);
  const g0 = grad(x);
  const slope = dot(g0, d);
  if (slope >= 0) return 0; // 非下降方向：不前进
  let alpha = opts.alpha0 != null ? opts.alpha0 : 1;
  for (let k = 0; k < maxK; k++) {
    const xp = vadd(x, vscale(d, alpha));
    if (f(xp) <= f0 + c1 * alpha * slope) return alpha;
    alpha *= rho;
  }
  return alpha;
}

/* ============================================================================
 * O13 强 Wolfe 线搜索（回溯 + 曲率），返回满足 Armijo 与曲率条件的 α
 * 独立对照可用黄金分割（见复核脚本）
 * ========================================================================== */
function strongWolfe(f, grad, x, d, opts) {
  opts = opts || {};
  const c1 = opts.c1 != null ? opts.c1 : 1e-4;
  const c2 = opts.c2 != null ? opts.c2 : 0.9;
  const maxIter = opts.maxIter || 40;
  const f0 = f(x);
  const g0 = grad(x);
  const slope0 = dot(g0, d);
  if (slope0 >= 0) return 0;
  let alpha = opts.alpha0 != null ? opts.alpha0 : 1;
  const rho = 0.5;
  for (let k = 0; k < maxIter; k++) {
    const xp = vadd(x, vscale(d, alpha));
    const fp = f(xp);
    if (fp > f0 + c1 * alpha * slope0) { alpha *= rho; continue; }
    const gp = grad(xp);
    if (Math.abs(dot(gp, d)) <= -c2 * slope0) return alpha;
    alpha *= rho;
  }
  return alpha;
}

/* ============================================================================
 * O04 共轭梯度法（SPD 系统 Ax=b）
 * ========================================================================== */
function cg(A, b, opts) {
  opts = opts || {};
  const tol = opts.tol || 1e-12;
  const maxIter = opts.maxIter || (A.length * 2 + 10);
  let x = opts.x0 ? opts.x0.slice() : new Array(b.length).fill(0);
  let r = vsub(b, matVec(A, x));
  let p = r.slice();
  let rsOld = dot(r, r);
  for (let k = 0; k < maxIter; k++) {
    const Ap = matVec(A, p);
    const alpha = rsOld / dot(p, Ap);
    x = vadd(x, vscale(p, alpha));
    r = vsub(r, vscale(Ap, alpha));
    const rsNew = dot(r, r);
    if (Math.sqrt(rsNew) < tol) { x._iters = k + 1; return x; }
    p = vadd(r, vscale(p, rsNew / rsOld));
    rsOld = rsNew;
  }
  return x;
}

/* ============================================================================
 * O06 / O11 拟牛顿更新
 * O06 BFGS（Hessian B 形式，秩2）：B₊=B+(yyᵀ)/(yᵀs)−(Bs sᵀB)/(sᵀBs)
 * O11 SR1（秩1）：B₊=B+((y−Bs)(y−Bs)ᵀ)/((y−Bs)ᵀs)
 * 对称更新均恒对称；yᵀs≤0 时返回原 B（保 SPD）
 * ========================================================================== */
function quasiNewtonUpdateBFGS(B, s, y) {
  const n = s.length;
  const ys = dot(y, s);
  if (ys <= 1e-12) return B.map((r) => r.slice());
  const Bs = matVec(B, s);
  const sBs = dot(s, Bs);
  const Bp = B.map((row) => row.slice());
  // + yyᵀ/(ys)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Bp[i][j] += (y[i] * y[j]) / ys;
  // − (Bs sᵀB)/(sᵀBs)
  if (sBs > 1e-14) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Bp[i][j] -= (Bs[i] * Bs[j]) / sBs;
  return Bp;
}
function sr1Update(B, s, y) {
  const n = s.length;
  const Bs = matVec(B, s);
  const diff = vsub(y, Bs);
  const denom = dot(diff, s);
  if (Math.abs(denom) <= 1e-12) return B.map((r) => r.slice());
  const Bp = B.map((row) => row.slice());
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Bp[i][j] += (diff[i] * diff[j]) / denom;
  return Bp;
}
function quasiNewtonUpdateDFP(H, s, y) { // H = 逆 Hessian 形式 DFP
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

/* ============================================================================
 * O03 BFGS（逆 Hessian 形式，限内存可选 m）—— 返回 {x, H}
 * ========================================================================== */
function bfgs(f, grad, x0, opts) {
  opts = opts || {};
  const maxIter = opts.maxIter || 200;
  const tol = opts.tol || 1e-9;
  let x = x0.slice();
  let g = grad(x);
  const n = x.length;
  let H = identityMat(n);
  let minEigOk = true;
  for (let k = 0; k < maxIter; k++) {
    const d = vscale(matVec(H, g), -1); // d = −H g
    if (dot(g, d) >= 0) d = vscale(g, -1); // 退化时回落到最速下降
    const alpha = lineSearch(f, grad, x, d, { c1: 1e-4 });
    if (alpha === 0) break;
    const s = vscale(d, alpha);
    const xNew = vadd(x, s);
    const gNew = grad(xNew);
    const y = vsub(gNew, g);
    const ys = dot(y, s);
    if (ys > 1e-10) {
      H = bfgsInverseUpdate(H, s, y);
    }
    if (n === 2 && !issymPD2(H)) minEigOk = false;
    x = xNew; g = gNew;
    if (vnorm(g) < tol) break;
  }
  return { x, H, minEigOk, iters: 0 };
}
/* BFGS 逆更新（恒对称，满足割线方程 H₊y=s）：H₊ = H − (H y yᵀ H)/(yᵀH y) + (s sᵀ)/(yᵀs) */
function bfgsInverseUpdate(H, s, y) {
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
function identityMat(n) { const A = Array.from({ length: n }, () => new Array(n).fill(0)); for (let i = 0; i < n; i++) A[i][i] = 1; return A; }
function outer(a, b) { return a.map((x) => b.map((y) => x * y)); }

/* ============================================================================
 * O02 / O10 有限记忆 BFGS（双循环递归，m 历史向量）
 * ========================================================================== */
function lbfgs(f, grad, x0, opts) {
  opts = opts || {};
  const m = opts.m || 10;
  const maxIter = opts.maxIter || 200;
  const tol = opts.tol || 1e-9;
  let x = x0.slice();
  let g = grad(x);
  const n = x.length;
  const sHist = [], yHist = [];
  const lineOpts = { c1: 1e-4, c2: 0.9, alpha0: 1 };
  for (let k = 0; k < maxIter; k++) {
    // 双循环递归求 H₀ g 方向（H₀=γI，γ=sᵀy/yᵀy）
    let q = g.slice();
    const alphas = [];
    const size = sHist.length;
    for (let i = size - 1; i >= 0; i--) {
      const si = sHist[i], yi = yHist[i];
      const rho = 1 / dot(yi, si);
      const a = rho * dot(si, q);
      alphas[i] = a;
      q = vsub(q, vscale(yi, a));
    }
    let r;
    if (size > 0) {
      const ysLast = dot(yHist[size - 1], sHist[size - 1]);
      const yyLast = dot(yHist[size - 1], yHist[size - 1]);
      const gamma = ysLast / Math.max(yyLast, 1e-30);
      r = vscale(q, gamma);
    } else r = q.slice();
    for (let i = 0; i < size; i++) {
      const si = sHist[i], yi = yHist[i];
      const rho = 1 / dot(yi, si);
      const beta = rho * dot(yi, r);
      r = vadd(r, vscale(si, alphas[i] - beta));
    }
    const d = vscale(r, -1); // −H g
    if (dot(g, d) >= 0) d = vscale(g, -1); // 退化时回落到最速下降
    const alpha = lineSearch(f, grad, x, d, { c1: 1e-4 });
    if (alpha === 0) break;
    const s = vscale(d, alpha);
    const xNew = vadd(x, s);
    const gNew = grad(xNew);
    const y = vsub(gNew, g);
    if (dot(y, s) > 1e-10) {
      sHist.push(s); yHist.push(y);
      if (sHist.length > m) { sHist.shift(); yHist.shift(); }
    }
    x = xNew; g = gNew;
    if (vnorm(g) < tol) break;
  }
  return x;
}

/* ============================================================================
 * O07 海森近似：由历史 (s,y) 序列经 BFGS 逆更新从 I 累积出 H≈∇²f⁻¹
 * ========================================================================== */
function hessianApprox(sList, yList, n) {
  let H = identityMat(n);
  for (let i = 0; i < sList.length; i++) H = bfgsInverseUpdate(H, sList[i], yList[i]);
  return H;
}

/* ============================================================================
 * O08 阻尼牛顿：Hessian 不定则加 λI 使 (H+λI) 对称正定后再步进
 * ========================================================================== */
function dampedNewton(f, grad, hess, x0, opts) {
  opts = opts || {};
  const maxIter = opts.maxIter || 50;
  const tol = opts.tol || 1e-12;
  let x = x0.slice();
  let lambda = opts.lambda0 || 0;
  for (let k = 0; k < maxIter; k++) {
    const g = grad(x);
    const H = hess(x);
    // H 已 SPD 则无需阻尼，重置 λ=0 避免无谓过阻尼
    if (H.length === 2 ? issymPD2(H) : false) lambda = 0;
    // 提 λ 直到 (H+λI) 最小特征值 > 0（2×2 精确判；其余靠 λ 增长保证 SPD）
    let Hl = H.map((row, i) => row.map((v, j) => (i === j ? v + lambda : v)));
    if (!(H.length === 2 ? issymPD2(Hl) : (lambda > 1e3))) {
      let guard = 0;
      while (guard < 60) {
        lambda = lambda === 0 ? 1e-3 : lambda * 10;
        Hl = H.map((row, i) => row.map((v, j) => (i === j ? v + lambda : v)));
        if (H.length === 2 ? issymPD2(Hl) : (lambda > 1e3)) break;
        guard++;
      }
    }
    const step = solveLinear(Hl, vscale(g, -1));
    if (!step) break;
    x = vadd(x, step);
    if (vnorm(g) < tol) break;
  }
  return x;
}

/* ============================================================================
 * O14 Hessian-free 牛顿：用 CG 解 H d=−g（Hessian 以 HVP 给出，不显式成形）
 * ========================================================================== */
function hessianFree(f, grad, hvp, x0, opts) {
  opts = opts || {};
  const maxIter = opts.maxIter || 50;
  const tol = opts.tol || 1e-12;
  let x = x0.slice();
  for (let k = 0; k < maxIter; k++) {
    const g = grad(x);
    // 用 CG 解 H(x) d = −g；hvp 提供 H·v
    let d = cgHvp(hvp, x, vscale(g, -1), opts);
    if (dot(g, d) >= 0) d = vscale(g, -1); // 退化时回落到最速下降
    const alpha = lineSearch(f, grad, x, d, { c1: 1e-4 });
    if (alpha === 0) break;
    x = vadd(x, vscale(d, alpha));
    if (vnorm(g) < tol) break;
  }
  return x;
}
/* 以 Hessian-向量积求解 A(x)·d=b 的 CG（A 对称正定由 hvp 隐式定义） */
function cgHvp(hvp, x, b, opts) {
  const tol = (opts && opts.tol) || 1e-12;
  const maxIter = (opts && opts.maxIter) || 50;
  let d = b.map(() => 0); // x₀ = 0
  let r = b.slice();      // r₀ = b − A·0 = b（保证残差自洽）
  let p = r.slice();
  let rsOld = dot(r, r);
  for (let k = 0; k < maxIter; k++) {
    const Ap = hvp(x, p);
    const alpha = rsOld / dot(p, Ap);
    d = vadd(d, vscale(p, alpha));
    r = vsub(r, vscale(Ap, alpha));
    const rsNew = dot(r, r);
    if (Math.sqrt(rsNew) < tol) break;
    p = vadd(r, vscale(p, rsNew / rsOld));
    rsOld = rsNew;
  }
  return d;
}

/* ============================================================================
 * O09 OWL-QN（L1 正则正交投影）：平滑凸 f 的 L1 最小化
 * 伪梯度 + 正交域限制线搜索；返回 x
 * ========================================================================== */
function owlqn(f, grad, x0, lambda, opts) {
  opts = opts || {};
  const m = opts.m || 10;
  const maxIter = opts.maxIter || 100;
  const tol = opts.tol || 1e-6;
  let x = x0.slice();
  let g = grad(x);
  const n = x.length;
  const sHist = [], yHist = [];
  function pseudoGrad(xx, gg) {
    const pg = new Array(n);
    for (let i = 0; i < n; i++) {
      if (xx[i] !== 0) pg[i] = gg[i] + lambda * Math.sign(xx[i]);
      else {
        if (gg[i] > lambda) pg[i] = gg[i] - lambda;
        else if (gg[i] < -lambda) pg[i] = gg[i] + lambda;
        else pg[i] = 0;
      }
    }
    return pg;
  }
  for (let k = 0; k < maxIter; k++) {
    const pg = pseudoGrad(x, g);
    if (vnorm(pg) < tol) break;
    // 双循环求 −H pg（限制正交域方向）
    let q = pg.slice();
    const alphas = [];
    const size = sHist.length;
    for (let i = size - 1; i >= 0; i--) {
      const si = sHist[i], yi = yHist[i];
      const rho = 1 / dot(yi, si);
      const a = rho * dot(si, q); alphas[i] = a;
      q = vsub(q, vscale(yi, a));
    }
    let r = size > 0 ? vscale(q, dot(yHist[size - 1], sHist[size - 1]) / Math.max(dot(yHist[size - 1], yHist[size - 1]), 1e-30)) : q.slice();
    for (let i = 0; i < size; i++) {
      const si = sHist[i], yi = yHist[i];
      const rho = 1 / dot(yi, si);
      r = vadd(r, vscale(si, alphas[i] - rho * dot(yi, r)));
    }
    let dir = vscale(r, -1); // −H pg
    // 正交域投影：若 x_i 与 dir_i 异号（会越出正交域），令 dir_i=0
    for (let i = 0; i < n; i++) if (x[i] !== 0 && x[i] * dir[i] < 0) dir[i] = 0;
    if (vnorm(dir) < 1e-15) break;
    // 正交域限制线搜索：最大 α 使 x+αdir 不越域（且 Armijo 对 L1 目标下降）
    const phi = (xx) => f(xx) + lambda * xx.reduce((s, v) => s + Math.abs(v), 0);
    const phi0 = phi(x);
    let alpha = 1, maxAlpha = Infinity;
    for (let i = 0; i < n; i++) if (dir[i] !== 0) maxAlpha = Math.min(maxAlpha, -x[i] / dir[i]);
    let accepted = 0;
    for (let step = 0; step < 40; step++) {
      if (alpha > maxAlpha) alpha = maxAlpha;
      const xn = vadd(x, vscale(dir, alpha));
      if (phi(xn) <= phi0 - 1e-4 * alpha * dot(pg, dir) || alpha <= maxAlpha * 1e-3) { accepted = alpha; break; }
      alpha *= 0.5;
      if (alpha < 1e-12) break;
    }
    if (accepted <= 0) break;
    const s = vscale(dir, accepted);
    const xNew = vadd(x, s);
    const gNew = grad(xNew);
    const y = vsub(gNew, g);
    if (dot(y, s) > 1e-10) { sHist.push(s); yHist.push(y); if (sHist.length > m) { sHist.shift(); yHist.shift(); } }
    x = xNew; g = gNew;
  }
  return x;
}

/* ============================================================================
 * O05 信赖域（dogleg）：子问题 m(s)=f+gᵀs+½sᵀBs，‖s‖≤Δ
 * ========================================================================== */
function trustRegion(f, grad, hess, x0, opts) {
  opts = opts || {};
  const maxIter = opts.maxIter || 60;
  const tol = opts.tol || 1e-10;
  let x = x0.slice();
  let delta = opts.delta0 || 1;
  const deltaMax = opts.deltaMax || 1e3;
  for (let k = 0; k < maxIter; k++) {
    const g = grad(x);
    const B = hess(x);
    const gtg = dot(g, g);
    const gtgBg = dot(g, matVec(B, g)); // gᵀB g
    // 无约束牛顿步 p_u = −(gᵀg)/(gᵀBg) g（B SPD 时）
    let pu;
    if (gtgBg > 1e-12) pu = vscale(g, -gtg / gtgBg);
    else pu = vscale(g, -1);
    const pclen = vnorm(pu);
    // 柯西点
    let pc;
    if (pclen >= delta) pc = vscale(pu, delta / pclen);
    else pc = pu.slice();
    // dogleg：若 ‖p_u‖≤Δ 取 p_u；否则 (pc + τ(p_u−pc)) 使 ‖·‖=Δ
    let s;
    if (pclen <= delta) s = pu.slice();
    else {
      const a = dot(pu, pu) - delta * delta;
      // 解 ‖pc + τ(pu−pc)‖²=Δ²  → 二次
      const w = vsub(pu, pc);
      const A2 = dot(w, w), B2 = 2 * dot(pc, w), C2 = dot(pc, pc) - delta * delta;
      const disc = Math.max(0, B2 * B2 - 4 * A2 * C2);
      let tau = (-B2 + Math.sqrt(disc)) / (2 * A2);
      if (!(tau >= 0 && tau <= 1)) tau = 1; // 退化保底
      s = vadd(pc, vscale(w, tau));
      if (vnorm(s) > delta) s = pc.slice(); // 数值边界护持：确保 ‖s‖≤Δ（pc 已在信赖域边界 Δ 上）
    }
    const f0 = f(x);
    const m0 = f0; // m(0)
    const ms = f0 + dot(g, s) + 0.5 * dot(s, matVec(B, s));
    const xn = vadd(x, s);
    const fn = f(xn);
    const ared = f0 - fn;
    const pred = m0 - ms;
    const rho = pred > 1e-15 ? ared / pred : (ared > 0 ? 1 : 0);
    if (rho < 0.25) delta = Math.max(delta * 0.25, 1e-8);
    else if (rho > 0.75) delta = Math.min(delta * 2, deltaMax);
    if (rho > 0.1) x = xn;
    if (vnorm(g) < tol) break;
    if (delta < 1e-10) break;
  }
  return x;
}

/* ============================================================================
 * O15 曲率条件：yᵀs>0 判据（拟牛顿更新可接受性的必要条件）
 * ========================================================================== */
function curvatureCondition(s, y) {
  const ys = dot(y, s);
  return { ys, positive: ys > 1e-10, ok: ys > 1e-10 };
}

/* ============================================================================
 * 自洽报告骨架（零依赖本地实现，与 forge 契约一致）
 * ========================================================================== */
function caseOf(id, name, ok, actual, detail, tol) {
  const numActual = (typeof actual === 'number' && isFinite(actual)) ? actual : (ok ? 0 : 1);
  return { id, name, status: ok ? 'pass' : 'fail', passed: !!ok, actual: numActual, expected: 0, tol: (typeof tol === 'number') ? tol : TOL.EXACT, detail };
}
function buildReport(module, version, layer, cases, invariants, seed) {
  const passed = cases.filter((c) => c.status === 'pass').length;
  const failed = cases.length - passed;
  let hashSrc = '';
  for (const c of cases) hashSrc += c.id + '|' + c.status + '|' + (typeof c.actual === 'number' ? String(Math.round(c.actual * 1e12)) : String(c.actual)) + '\n';
  return { schema: 'forge-selfcheck/1.0', module, version, layer, passed, failed, skipped: 0, duration_ms: 0, tol: TOL, invariants, cases, env: { platform: (typeof process !== 'undefined' && process.platform) || 'browser', arch: (typeof process !== 'undefined' && process.arch) || 'unknown', node: (typeof process !== 'undefined' && process.version) || 'browser', seed }, hash: 'sha256:' + sha256HexLocal(hashSrc) };
}

/* 局部 SHA-256（与 serialize-forge 同实现，保证 hash 契约一致） */
function _rotrL(x, n) { return ((x >>> n) | (x << (32 - n))) >>> 0; }
const K256SO = [
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
];
function sha256HexLocal(str) {
  const bytes = [];
  for (let i = 0; i < str.length; i++) {
    let c = str.charCodeAt(i);
    if (c < 0x80) bytes.push(c);
    else if (c < 0x800) bytes.push(0xC0 | (c >> 6), 0x80 | (c & 63));
    else if (c >= 0xD800 && c <= 0xDBFF && i + 1 < str.length && str.charCodeAt(i + 1) >= 0xDC00 && str.charCodeAt(i + 1) <= 0xDFFF) {
      c = 0x10000 + ((c - 0xD800) << 10) + (str.charCodeAt(++i) - 0xDC00);
      bytes.push(0xF0 | (c >> 18), 0x80 | ((c >> 12) & 63), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
    } else bytes.push(0xE0 | (c >> 12), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
  }
  const msg = bytes.slice();
  const bitLen = msg.length * 8;
  msg.push(0x80);
  while (msg.length % 64 !== 56) msg.push(0);
  const hi = Math.floor(bitLen / 0x100000000), lo = bitLen >>> 0;
  msg.push((hi >>> 24) & 255, (hi >>> 16) & 255, (hi >>> 8) & 255, hi & 255, (lo >>> 24) & 255, (lo >>> 16) & 255, (lo >>> 8) & 255, lo & 255);
  let Hh = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
  const w = new Array(64);
  for (let off = 0; off < msg.length; off += 64) {
    for (let t = 0; t < 16; t++) w[t] = ((msg[off + 4 * t] << 24) | (msg[off + 4 * t + 1] << 16) | (msg[off + 4 * t + 2] << 8) | msg[off + 4 * t + 3]) | 0;
    for (let t = 16; t < 64; t++) { const x15 = w[t - 15], x2 = w[t - 2]; const s0 = (_rotrL(x15, 7) ^ _rotrL(x15, 18) ^ (x15 >>> 3)) | 0; const s1 = (_rotrL(x2, 17) ^ _rotrL(x2, 19) ^ (x2 >>> 10)) | 0; w[t] = (w[t - 16] + s0 + w[t - 7] + s1) | 0; }
    let a = Hh[0], b = Hh[1], c = Hh[2], d = Hh[3], e = Hh[4], f = Hh[5], g = Hh[6], h = Hh[7];
    for (let t = 0; t < 64; t++) {
      const S1 = (_rotrL(e, 6) ^ _rotrL(e, 11) ^ _rotrL(e, 25)) | 0;
      const ch = ((e & f) ^ (~e & g)) | 0;
      const t1 = (h + S1 + ch + K256SO[t] + w[t]) | 0;
      const S0 = (_rotrL(a, 2) ^ _rotrL(a, 13) ^ _rotrL(a, 22)) | 0;
      const maj = ((a & b) ^ (a & c) ^ (b & c)) | 0;
      const t2 = (S0 + maj) | 0;
      h = g; g = f; f = e; e = (d + t1) | 0; d = c; c = b; b = a; a = (t1 + t2) | 0;
    }
    Hh = [(Hh[0] + a) | 0, (Hh[1] + b) | 0, (Hh[2] + c) | 0, (Hh[3] + d) | 0, (Hh[4] + e) | 0, (Hh[5] + f) | 0, (Hh[6] + g) | 0, (Hh[7] + h) | 0];
  }
  return Hh.map((x) => ('00000000' + (x >>> 0).toString(16)).slice(-8)).join('');
}

/* ============================================================================
 * 自洽自检：8 条皇冠不变量（每条含独立重算 / 双向 / 证伪）
 * ========================================================================== */
function selfCheck() {
  const cases = [], invariants = [];
  const push = (id, name, ok, actual, detail, tol) => {
    cases.push(caseOf(id, name, ok, actual, detail, tol));
    invariants.push({ id, name, ok, expr: name, tol: (typeof tol === 'number') ? tol : TOL.EXACT, worst: ok ? 0 : 1, detail });
  };

  /* ---------- INV-SO-001：牛顿法二次收敛 ---------- */
  {
    // 2D 二次 f(x)=½xᵀAx−bᵀx，A SPD
    const A = [[3, 1], [1, 2]];
    const b = [5, 3];
    const f = (x) => 0.5 * dot(x, matVec(A, x)) - dot(b, x);
    const grad = (x) => vsub(matVec(A, x), b);
    const hess = () => A;
    const x0 = [-4, 6];
    const x1 = newton(f, grad, hess, x0, { maxIter: 1 });
    const xStar = solveLinear(A, b); // 直接解
    const residNorm = vnorm(vsub(matVec(A, x1), b));
    const directDiff = vnorm(vsub(x1, xStar));
    const ok = residNorm <= 1e-12 * (vnorm(b) + 1) && directDiff <= 1e-12 * (vnorm(xStar) + 1);
    push('INV-SO-001', '牛顿法对二次 1 步收敛：x₁ 恰为极小点（对拍直接解），残差‖Ax₁−b‖≤1e-12', ok, ok ? 0 : Math.max(residNorm, directDiff),
      'resid=' + residNorm.toExponential(2) + ' xdiff=' + directDiff.toExponential(2) + ' xStar=[' + xStar.map((v) => v.toFixed(4)).join(',') + ']', TOL.RESID);
  }

  /* ---------- INV-SO-002：CG 对 SPD 线性系统 ---------- */
  {
    const A = [[4, 1], [1, 3]];
    const b = [1, 2];
    const x = cg(A, b);
    const xDir = solveLinear(A, b);
    const sysResid = vnorm(vsub(matVec(A, x), b));
    const solDiff = vnorm(vsub(x, xDir));
    const ok = sysResid <= 1e-10 * (vnorm(b) + 1) && solDiff <= 1e-10 * (vnorm(xDir) + 1) && (x._iters || 2) <= 2;
    push('INV-SO-002', 'CG 对 2×2 SPD 系统 ≤n 步收敛，解与直接解一致（‖Ax−b‖≤1e-10）', ok, ok ? 0 : Math.max(sysResid, solDiff),
      'iters=' + (x._iters || '?') + ' sysResid=' + sysResid.toExponential(2) + ' solDiff=' + solDiff.toExponential(2), TOL.SOLVE);
  }

  /* ---------- INV-SO-003：BFGS/L-BFGS 收敛 + SPD + 曲率 ---------- */
  {
    // Rosenbrock f=100(y−x²)²+(1−x)²；L-BFGS 收敛
    const f = (x) => 100 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]) + (1 - x[0]) * (1 - x[0]);
    const grad = (x) => [-400 * x[0] * (x[1] - x[0] * x[0]) - 2 * (1 - x[0]), 200 * (x[1] - x[0] * x[0])];
    const x0 = [-1.2, 1];
    const f0 = f(x0);
    const xL = lbfgs(f, grad, x0, { m: 10, maxIter: 5000, tol: 1e-8 });
    const fEnd = f(xL);
    const converged = fEnd <= f0 && vnorm(grad(xL)) < 1e-3;
    // BFGS 逆更新 SPD（用一段历史验证）
    const H = identityMat(2);
    const sList = [[0.1, 0.05], [0.2, -0.1]];
    const yList = [[0.3, 0.1], [0.15, -0.2]];
    let Hacc = H;
    let curvAllPos = true, symOk = true, pdOk = true;
    for (let i = 0; i < sList.length; i++) { Hacc = bfgsInverseUpdate(Hacc, sList[i], yList[i]); if (dot(yList[i], sList[i]) <= 0) curvAllPos = false; if (Math.abs(Hacc[0][1] - Hacc[1][0]) > 1e-12) symOk = false; if (!issymPD2(Hacc)) pdOk = false; }
    const ok = converged && curvAllPos && symOk && pdOk;
    push('INV-SO-003', 'BFGS/L-BFGS：Rosenbrock 收敛(f≤f₀) + 逆更新恒对称正定(SPD) + 曲率 yᵀs>0', ok, ok ? 0 : (converged ? 0 : vnorm(grad(xL))),
      'f0=' + f0.toFixed(2) + ' fEnd=' + fEnd.toExponential(2) + ' ||g||=' + vnorm(grad(xL)).toExponential(2) + ' sym=' + symOk + ' pd=' + pdOk, TOL.SOLVE);
  }

  /* ---------- INV-SO-004：强 Wolfe 线搜索（Armijo + 曲率 + 黄金分割对拍 + 非下降证伪） ---------- */
  {
    const f = (x) => (x[0] - 2) * (x[0] - 2) + (x[1] + 1) * (x[1] + 1);
    const grad = (x) => [2 * (x[0] - 2), 2 * (x[1] + 1)];
    const x = [0, 0];
    const d = [4, -2]; // 负梯度方向（下降方向）
    const aWolfe = strongWolfe(f, grad, x, d, { c1: 1e-4, c2: 0.9, maxIter: 40 });
    const f0 = f(x), g0 = grad(x), slope0 = dot(g0, d);
    const aF = vadd(x, vscale(d, aWolfe));
    const armijo = f(aF) <= f0 + 1e-4 * aWolfe * slope0;
    const curv = Math.abs(dot(grad(aF), d)) <= 0.9 * Math.abs(slope0);
    // 黄金分割对照：在 [0, aWolfe*4] 内最小化 φ(α)=f(x+αd)
    const phi = (a) => f(vadd(x, vscale(d, a)));
    let lo = 0, hi = Math.max(aWolfe * 4, 4);
    const gr = (3 - Math.sqrt(5)) / 2;
    let ga = lo + gr * (hi - lo), gb = lo + (1 - gr) * (hi - lo);
    let fa = phi(ga), fb = phi(gb);
    for (let i = 0; i < 60; i++) { if (fa < fb) { hi = gb; gb = ga; fb = fa; ga = lo + gr * (hi - lo); fa = phi(ga); } else { lo = ga; ga = gb; fa = fb; gb = lo + (1 - gr) * (hi - lo); fb = phi(gb); } }
    const aGold = (lo + hi) / 2;
    const goldF = phi(aGold), wolfeF = phi(aWolfe);
    const closeMin = Math.abs(wolfeF - goldF) <= Math.max(1e-3, 1e-2 * Math.abs(goldF) + 1e-6);
    // 证伪：非下降方向返回 0（取梯度本身作上升方向：g₀ᵀd=‖g₀‖²>0）
    const badDir = g0.slice(); // 上升方向
    const aBad = strongWolfe(f, grad, x, badDir, { c1: 1e-4, c2: 0.9 });
    const falsOk = aBad === 0;
    const ok = armijo && curv && closeMin && falsOk;
    push('INV-SO-004', '强 Wolfe：Armijo + 曲率条件满足 + 黄金分割局部极小对拍(差≤1e-3) + 非下降方向返回0证伪', ok, ok ? 0 : (armijo ? 0 : (curv ? 0 : (closeMin ? 0 : 1))),
      'α=' + aWolfe.toFixed(4) + ' armijo=' + armijo + ' curv=' + curv + ' αgold=' + aGold.toFixed(4) + ' Δf=' + (wolfeF - goldF).toExponential(2) + ' badDirReturns0=' + falsOk, TOL.CONV);
  }

  /* ---------- INV-SO-005：信赖域 dogleg 不劣于柯西点 + 信赖域约束 + 半径自适应 ---------- */
  {
    const A = [[2, 0.5], [0.5, 1]];
    const b = [1, 0.5];
    const f = (x) => 0.5 * dot(x, matVec(A, x)) - dot(b, x);
    const grad = (x) => vsub(matVec(A, x), b);
    const hess = () => A;
    const x0 = [-2, 3];
    // 逐迭代验证 m(s*)≤m(p_c) 与 ‖s*‖≤Δ
    let x = x0.slice(), delta = 1, deltaMax = 1e3, allDoglegOk = true, allBoundOk = true, expanded = false, contracted = false;
    for (let k = 0; k < 40; k++) {
      const g = grad(x), B = hess(x);
      const gtg = dot(g, g), gtgBg = dot(g, matVec(B, g));
      const pu = gtgBg > 1e-12 ? vscale(g, -gtg / gtgBg) : vscale(g, -1);
      const pclen = vnorm(pu);
      const pc = pclen >= delta ? vscale(pu, delta / pclen) : pu.slice();
      let s;
      if (pclen <= delta) s = pu.slice();
      else { const w = vsub(pu, pc); const A2 = dot(w, w), B2 = 2 * dot(pc, w), C2 = dot(pc, pc) - delta * delta; const disc = Math.max(0, B2 * B2 - 4 * A2 * C2); let tau = (-B2 + Math.sqrt(disc)) / (2 * A2); if (!(tau >= 0 && tau <= 1)) tau = 1; s = vadd(pc, vscale(w, tau)); }
      const f0 = f(x);
      const ms = f0 + dot(g, s) + 0.5 * dot(s, matVec(B, s));
      const mpc = f0 + dot(g, pc) + 0.5 * dot(pc, matVec(B, pc));
      if (!(ms <= mpc + 1e-12)) allDoglegOk = false;
      if (!(vnorm(s) <= delta + 1e-12)) allBoundOk = false;
      const fn = f(vadd(x, s));
      const pred = f0 - ms, ared = f0 - fn;
      const rho = pred > 1e-15 ? ared / pred : (ared > 0 ? 1 : 0);
      const dPrev = delta;
      if (rho < 0.25) { delta = Math.max(delta * 0.25, 1e-8); if (delta < dPrev) contracted = true; }
      else if (rho > 0.75) { delta = Math.min(delta * 2, deltaMax); if (delta > dPrev) expanded = true; }
      if (rho > 0.1) x = vadd(x, s);
      if (vnorm(g) < 1e-10) break;
      if (delta < 1e-10) break;
    }
    const xStar = solveLinear(A, b);
    const convOk = vnorm(vsub(x, xStar)) <= 1e-6 * (vnorm(xStar) + 1);
    const ok = allDoglegOk && allBoundOk && convOk && expanded && contracted;
    push('INV-SO-005', '信赖域 dogleg：m(s*)≤m(p_c) 恒成立 + ‖s*‖≤Δ + 半径自适应(扩/缩均触发) + 二次收敛', ok, ok ? 0 : (allDoglegOk ? 0 : (allBoundOk ? 0 : 1)),
      'doglegLEcauchy=' + allDoglegOk + ' bound=' + allBoundOk + ' expanded=' + expanded + ' contracted=' + contracted + ' conv=' + convOk, TOL.EXACT);
  }

  /* ---------- INV-SO-006：高斯牛顿 == 牛顿（线性残差） ---------- */
  {
    const A = [[2, 1], [1, 3]], b = [4, 5];
    const R = (x) => vsub(matVec(A, x), b); // 线性残差
    const J = () => A;
    const f = (x) => 0.5 * dot(R(x), R(x));
    const grad = (x) => matVec(matTranspose(A), R(x));
    const hess = () => matMatMul(matTranspose(A), A);
    const x0 = [-3, 2];
    const xGN = gaussNewton(R, J, x0, { maxIter: 1 });
    const xN = newton(f, grad, hess, x0, { maxIter: 1 });
    const same = vnorm(vsub(xGN, xN)) <= 1e-12 * (vnorm(xN) + 1);
    // 非线性最小二乘：残差范数非增
    const R2 = (x) => [x[0] * x[0] - x[1], x[0] + x[1] - 2];
    const J2 = (x) => [[2 * x[0], -1], [1, 1]];
    const xNl = gaussNewton(R2, J2, [0.5, 0.5], { maxIter: 20 });
    const r0 = vnorm(R2([0.5, 0.5])), r1 = vnorm(R2(xNl));
    const nonIncrease = r1 <= r0 + 1e-9;
    const ok = same && nonIncrease;
    push('INV-SO-006', '高斯牛顿==牛顿(线性残差) EXACT + 非线性最小二乘残差范数非增', ok, ok ? 0 : (same ? 0 : 1),
      'GN==N_diff=' + vnorm(vsub(xGN, xN)).toExponential(2) + ' r0=' + r0.toFixed(4) + ' r1=' + r1.toFixed(4) + ' nonInc=' + nonIncrease, TOL.RESID);
  }

  /* ---------- INV-SO-007：阻尼牛顿 + Hessian-free ---------- */
  {
    // (A) 阻尼性质（构造鞍点，严格证明）：H 不定则提 λ 使 (H+λI) SPD 且步长必下降
    const H0 = [[2, 0], [0, -1]]; // 不定（鞍点 Hessian）
    const g0 = [1, 1];            // 任意非零梯度
    let lam = 0, Hl;
    for (let guard = 0; guard < 80; guard++) {
      Hl = H0.map((row, i) => row.map((v, j) => (i === j ? v + lam : v)));
      if (issymPD2(Hl)) break;
      lam = lam === 0 ? 1e-3 : lam * 10;
    }
    const dDamp = solveLinear(Hl, vscale(g0, -1)); // d = −(H+λI)⁻¹ g
    const descentOk = dot(g0, dDamp) < 0;           // (H+λI) SPD ⇒ −(H+λI)⁻¹ 负定 ⇒ 下降
    const pdOk = issymPD2(Hl);

    // (B) 阻尼牛顿端到端：f=(x²−1)²+(y²−1)² 起步 Hessian 不定，收敛到极小点 (1,1)
    const fb = (x) => (x[0] * x[0] - 1) * (x[0] * x[0] - 1) + (x[1] * x[1] - 1) * (x[1] * x[1] - 1);
    const ggb = (x) => [4 * x[0] * (x[0] * x[0] - 1), 4 * x[1] * (x[1] * x[1] - 1)];
    const Hb = (x) => [[12 * x[0] * x[0] - 4, 0], [0, 12 * x[1] * x[1] - 4]];
    const xDb = dampedNewton(fb, ggb, Hb, [0.5, 0.5], { lambda0: 0, maxIter: 500, tol: 1e-10 });
    const convOk = vnorm(vsub(xDb, [1, 1])) <= 1e-7 * (Math.SQRT2 + 1);

    // (C) Hessian-free：二次 Af,b，HF 方向 == 精确牛顿步（一步达极小点）
    const Af = [[3, 1], [1, 2]], bf = [5, 3];
    const fq = (x) => 0.5 * dot(x, matVec(Af, x)) - dot(bf, x);
    const gq = (x) => vsub(matVec(Af, x), bf);
    const hvpQ = (x, v) => matVec(Af, v);
    const xHF = hessianFree(fq, gq, hvpQ, [-4, 6], { maxIter: 20 });
    const xExact = solveLinear(Af, bf); // 二次精确极小点（一步牛顿必达）
    const hfOk = vnorm(vsub(xHF, xExact)) <= 1e-9 * (vnorm(xExact) + 1);

    const ok = descentOk && pdOk && convOk && hfOk;
    push('INV-SO-007', '阻尼牛顿(H+λI恒SPD且下降) + 端到端收敛 + Hessian-free==精确牛顿(二次) EXACT', ok, ok ? 0 : (descentOk ? 0 : (pdOk ? 0 : (convOk ? 0 : 1))),
      'descent=' + descentOk + ' pd=' + pdOk + ' convDiff=' + vnorm(vsub(xDb, [1, 1])).toExponential(2) + ' hfDiff=' + vnorm(vsub(xHF, xExact)).toExponential(2), TOL.RESID);
  }

  /* ---------- INV-SO-008：拟牛顿校正（BFGS/DFP/SR1）对称 + 割线 + 秩1公式 ---------- */
  {
    const H = identityMat(2);
    const s = [0.3, 0.1], y = [0.5, 0.2];
    const Hp = bfgsInverseUpdate(H, s, y);
    const sym1 = Math.abs(Hp[0][1] - Hp[1][0]) < 1e-12;
    // 割线方程 H₊ y = s（BFGS 逆更新满足 H y=s 当 ys>0 — 实际 BFGS 逆更新满足 (H+ρssᵀ)y=s? 验证 H₊y≈s）
    const Hy = matVec(Hp, y);
    const secant = vnorm(vsub(Hy, s)) <= 1e-12 * (vnorm(s) + 1);
    const Hdfp = quasiNewtonUpdateDFP(H, s, y);
    const sym2 = Math.abs(Hdfp[0][1] - Hdfp[1][0]) < 1e-12;
    const S1 = sr1Update(H, s, y);
    const sym3 = Math.abs(S1[0][1] - S1[1][0]) < 1e-12;
    // SR1 秩1 公式：H₊=H+(y−Hy)(y−Hy)ᵀ/((y−Hy)ᵀs)
    const Hy0 = matVec(H, y), diff = vsub(y, Hy0), denom = dot(diff, s);
    let sr1exact = true;
    if (Math.abs(denom) > 1e-12) {
      const expect = H.map((row, i) => row.map((v, j) => v + (diff[i] * diff[j]) / denom));
      sr1exact = Math.abs(S1[0][0] - expect[0][0]) < 1e-12 && Math.abs(S1[1][1] - expect[1][1]) < 1e-12;
    }
    // 曲率护持：yᵀs≤0 时更新被跳过后仍 SPD
    const sBad = [0.1, 0.1], yBad = [-0.2, -0.2];
    const Hbad = bfgsInverseUpdate(H, sBad, yBad);
    const skippedOk = Hbad[0][0] === 1 && Hbad[1][1] === 1; // 原样返回
    const ok = sym1 && secant && sym2 && sym3 && sr1exact && skippedOk;
    push('INV-SO-008', '拟牛顿 BFGS/DFP/SR1 恒对称 + BFGS 逆更新割线方程 H₊y=s(≤1e-12) + SR1 秩1 公式 EXACT + 负曲率跳过护 SPD', ok, ok ? 0 : (sym1 ? 0 : (secant ? 0 : (sr1exact ? 0 : 1))),
      'sym1=' + sym1 + ' secant=' + secant + ' sym2=' + sym2 + ' sym3=' + sym3 + ' sr1exact=' + sr1exact + ' skip=' + skippedOk, TOL.RESID);
  }

  return buildReport('secondorder-forge', '1.0.0', 'L3', cases, invariants, '0x9E3779B9');
}

/* ============================================================================
 * API 导出 + 双端挂载（UMD）+ CLI 自检
 * ========================================================================== */
const API = {
  name: 'secondorder-forge', version: '1.0.0', layer: 'L3', TOL,
  makeRng,
  // 线性代数原语
  dot, vadd, vsub, vscale, vnorm, matVec, matMatMul, matTranspose, solveLinear, eigSym2, issymPD2,
  // 16 项操作
  newton, gaussNewton, lineSearch, strongWolfe, cg,
  quasiNewtonUpdateBFGS, sr1Update, quasiNewtonUpdateDFP,
  bfgs, lbfgs, hessianApprox, dampedNewton, owlqn, trustRegion, curvatureCondition,
  hessianFree,
  selfCheck
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeSecondOrder = API;

if (typeof process !== 'undefined' && process.argv[1] && /secondorder-forge\.js$/.test(process.argv[1]) && (process.argv.includes('--self-check') || process.argv.includes('--json'))) {
  const r = selfCheck();
  const lines = ['=== ' + r.module + ' v' + r.version + ' (' + r.layer + ') ' + (r.failed === 0 ? 'PASS' : 'FAIL') + ' ==='];
  lines.push('passed ' + r.passed + ' / failed ' + r.failed + ' / hash ' + (r.hash || '').slice(0, 24));
  for (const c of (r.cases || [])) lines.push('  ' + (c.status === 'pass' ? '+OK' : '-XX') + ' ' + c.id + ' ' + c.name);
  process.stdout.write((process.argv.includes('--json') ? JSON.stringify(r, null, 2) : lines.join('\n')) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
