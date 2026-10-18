# 不变量总览 · INVARIANTS

> 创新旗舰 AI 的**硬不变量（hard invariant）清单**——每条不变量都有 ≥1 条可被独立交叉验证的金标准。
> 当前基线：**25 模块 · 194 用例 · 0 失败**（`npm run verify` 实测）。

---

## 不变量设计原则（S2 门槛）

每个模块必须声明 ≥1 条不变量，且至少一条满足以下**可独立交叉验证**之一：

| 验证方法 | 说明 | 典型 `TOL` |
|----------|------|-----------|
| 解析解 vs 数值解 | 闭式解与迭代/数值实现互比 | `RESID=1e-12` |
| 暴力枚举 vs 贪心 | 暴力黄金参照对照 | `RESID=1e-12` |
| 参照实现 vs 主实现 | 独立重写的第二实现互比 | `RESID=1e-12` |
| 梯度检验 | 中心差分 vs 解析梯度 | `GRAD=1e-6` |
| 正交 / 归一 | `QᵀQ=I`、`‖v‖=1` | `ORTHO=1e-14` |
| 残差界 | `‖A−重构‖/‖A‖` | `RESID=1e-12` |
| 对数似然单调 | EM / 优化单调不降 | `MONO=-1e-9` |
| 确定性 / 幂等 | 同种子同输出、hash 幂等 | `EXACT=0` |

`TOL` 九档：`EXACT=0` / `ULP=1e-16` / `TIGHT=1e-15` / `ORTHO=1e-14` / `RESID=1e-12` / `SOLVE=1e-10` / `GRAD=1e-6` / `MONO=-1e-9` / `CONV=1e-6`。

---

## L1 — 基础设施 / 数值内核

### `core-numeric.js`（INV-NUM-001..008，8 条）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-NUM-001 | 分块 GEMM vs 朴素 GEMM 相对误差 ≤ 1e-15 | 1e-15 | 分块仅改变求和结合序，理论相对界 O(ε·√n) |
| INV-NUM-002 | sha256 已知测试向量逐字符相等（`"abc"` / `""` / fox） | 0 | FIPS 180-4 标准测试向量 |
| INV-NUM-003 | 同种子两次 sfc32 序列逐位相等 | 0 | 确定性定义本身 |
| INV-NUM-004 | Kahan 补偿求和 vs 朴素求和：1e5 维随机向量相对偏差改进 ≥ 10× | 0 | Kahan 消除大数吃小数 |
| INV-NUM-005 | submat 零拷贝：改写子块父矩阵同步变化 | 0 | 共享 ArrayBuffer，ld ≥ cols 保证视图语义 |
| INV-NUM-006 | Moro/Acklam 逆 CDF 样本均值/方差在 n=20000 下与 N(0,1) 偏差 ≤ 0.05 | 0.05 | MCSE(mean)=1/√n≈0.007，取 7σ 余量 |
| INV-NUM-007 | dot 朴素累积 vs 补偿累积：补偿更优（偏差 ≤ 朴素） | 0 | 1e5 维条件数差的向量 |
| INV-NUM-008 | 矩阵范数 ‖A‖_F 与逐元素平方和根一致 | 0 | 定义等价 |

### `core-verify.js`（INV-VER-001..004，4 条）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-VER-001 | assertClose：`\|a−b\| ≤ tol` 或 `rel ≤ tol` 时判 pass，否则 fail | 1e-12 | 判定口径本身（含探针 pass/fail） |
| INV-VER-002 | 中心差分梯度 vs 解析梯度相对误差 ≤ GRAD(1e-6)，对 f(x)=x²·sin(x) | 1e-6 | 梯度检验定义 |
| INV-VER-003 | 元不变量：故意失败用例（2 vs 1，tol 1e-9）必须被标记为 fail（验证框架自身正确） | 0 | 自证能力 |
| INV-VER-004 | buildReport 聚合：passed+failed == cases.length，hash 仅覆盖 `[id,status,actual]`（排除 env/duration） | 0 | 报告构造一致性 |

---

## L2 — 线性代数 / 稳定数值 / 概率图基础

### `stable-forge.js`（INV-STB-001..004，4 条）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-STB-001 | logsumexp(x)=log(Σexp(xᵢ)) 对良态 x 相对误差 ≤ 1e-12；x=[1000,1000] 时 = 1000+log2 绝对误差 ≤ 1e-12 | 1e-12 | 移位恒等式 |
| INV-STB-002 | 稳定 softmax 分量和为 1 ≤ 1e-15；对良态 x 与朴素 softmax 一致 ≤ 1e-12 | 1e-15 | 概率归一 |
| INV-STB-003 | logSigmoid(x)=log(1/(1+exp(−x))) 对良态 x 一致 ≤ 1e-12；x=±1000 结果有限不溢出 | 1e-12 | 对数 sigmoid 定义 |
| INV-STB-004 | log1pExp(x)=log(1+exp(x)) 对良态 x 一致 ≤ 1e-12；x=1000 时 ≈ x（绝对误差 ≤ 1e-12） | 1e-12 | softplus 定义 |

### `linalg-forge.js`（INV-LIN-001..008，8 条）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-LIN-001 | ‖A−PᵀLU‖_F ≤ 1e-12·‖A‖_F | 1e-12 | LAPACK dgetrf 后向误差界（本实现约定 A=PᵀLU，等价 PA=LU） |
| INV-LIN-002 | ‖A−LLᵀ‖_F ≤ 1e-12·‖A‖_F（对称正定） | 1e-12 | Cholesky 定义残差 |
| INV-LIN-003 | ‖A−QR‖_F ≤ 1e-12 且 ‖QᵀQ−I‖ ≤ 1e-14 | 1e-14 | Householder QR 正交性 |
| INV-LIN-004 | ‖A−UΣVᵀ‖_F ≤ 5e-12·‖A‖_F；U/V 列正交 ≤ 1e-14；奇异值 = √eig(AᵀA) ± 1e-10 | 5e-12 | one-sided Jacobi SVD |
| INV-LIN-005 | Jacobi 对称 EVD：max‖Av−λv‖_2 ≤ 1e-12 | 1e-12 | 对齐存量 pca-forge 实测 1.09e-15 |
| INV-LIN-006 | 特征向量矩阵 VᵀV = I（正交归一）≤ 2e-14 | 2e-14 | 对齐实测 1.54e-14 |
| INV-LIN-007 | 最小二乘解 vs 正规方程(Cholesky)解在良态问题上一致 ≤ 1e-10 | 1e-10 | 两种独立实现互比 |
| INV-LIN-008 | LU 求解 vs 朴素高斯消元求解（独立实现）一致 ≤ 1e-10 | 1e-10 | 求解器双路互比 |

### `pca-forge.js`（INV-PCA-001..008，8 条）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-PCA-001 | 协方差矩阵对称：max\|A−Aᵀ\| < 1e-12 | 1e-12 | 协方差对称性 |
| INV-PCA-002 | 协方差半正定：λmin > −1e-10 | 1e-10 | 特征值下界 |
| INV-PCA-003 | 特征向量正交：max off-diag(VᵀV) < 1e-10 | 1e-10 | 正交性 |
| INV-PCA-004 | 特征向量归一：max\|‖v‖−1\| < 1e-12 | 1e-12 | 归一性 |
| INV-PCA-005 | 特征分解重构：max\|A−VΛVᵀ\| < 1e-9 | 1e-9 | EVD 重构残差 |
| INV-PCA-006 | Jacobi 残差：max‖Av−λv‖ < 1e-10 | 1e-10 | Jacobi EVD 残差 |
| INV-PCA-007 | 方差解释率降序且和 ≤ 1+1e-9 | 1e-9 | EVR 单调性 |
| INV-PCA-008 | 全分量重构无损：max\|X−Xr\| < 1e-9 | 1e-9 | 重构无损 |

### `gmm-forge.js`（INV-GMM-001..008，8 条）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-GMM-001 | 混合权重 Σπ = 1 | — | 权重归一 |
| INV-GMM-002 | 责任度每行 Σ = 1 | — | 责任度归一 |
| INV-GMM-003 | 协方差对称 | — | 对称性 |
| INV-GMM-004 | 协方差半正定 | — | 半正定 |
| INV-GMM-005 | 对数似然单调不降（EM 迭代） | — | 对数似然单调性 |
| INV-GMM-006 | 单高斯拟合接近真值 | — | 闭式真值对照 |
| INV-GMM-007 | logsumexp 稳定（不溢出） | — | 数值稳定 |
| INV-GMM-008 | 双簇聚类一致率 > 0.90 | — | 聚类一致性 |

---

## L3 — 优化器 / 自动微分 / 损失激活归一化 / 基础网络

### `nn-forge.js`（INV-NN-001..008，8 条）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-NN-001 | 前向输出 shape 正确 | — | shape 定义 |
| INV-NN-002 | 训练后损失下降 | — | 训练收敛 |
| INV-NN-003 | 数值梯度 maxRelErr < 1e-5 | 1e-5 | 梯度检验 |
| INV-NN-004 | 梯度检验 ≈ 9.23e-7 量级 | 1e-5 | 中心差分 vs 解析 |
| INV-NN-005 | 单步 SGD 损失下降 | — | SGD 单调 |
| INV-NN-006 | XOR 可解 acc > 0.9 | — | 非线性可分 |
| INV-NN-007 | softmax 和 = 1 且非负 | — | 概率定义 |
| INV-NN-008 | 反向梯度有限 | — | 梯度有限性 |

### `attn-forge.js`（INV-ATTN-001..008，8 条，经典注意力零重写迁移包装）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-ATTN-001 | 输出 shape = (nq, dv) | — | shape 定义 |
| INV-ATTN-002 | 权重每行和 = 1 | — | softmax 归一 |
| INV-ATTN-003 | 权重非负 | — | softmax 非负 |
| INV-ATTN-004 | 数值梯度 maxRelErr < 1e-5 | 1e-5 | 梯度检验 |
| INV-ATTN-005 | 梯度检验 ≈ 5.86e-7 量级 | 1e-5 | 中心差分 vs 解析 |
| INV-ATTN-006 | 因果掩码上三角 ≈ 0 | — | 因果结构 |
| INV-ATTN-007 | 加权求和平验 O = ΣP·V | — | 注意力定义 |
| INV-ATTN-008 | 梯度有限 | — | 梯度有限性 |

### `optim-forge.js`（INV-OPT-001..008，8 条）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-OPT-001 | SGD 在强凸 f(θ)=θ² 上收敛：θ0=1,lr=0.1,200步 → \|θ\| ≤ 1e-3（闭式 (1−2lr)^200≈1.7e-20） | 1e-3 | 闭式对照 |
| INV-OPT-002 | Adam 在 f(θ)=θ² 上收敛：lr=0.1,β=(0.9,0.999),300步 → \|θ\| ≤ 1e-3 | 1e-3 | Adam 收敛性经验边界 |
| INV-OPT-003 | Adam 每步更新与独立重实现公式一致（mhat/vhat→更新量），最大偏差 ≤ 1e-12 | 1e-12 | 参照实现 vs 主实现 |
| INV-OPT-004 | AdaGrad 在 f(θ)=θ² 上收敛：lr=0.3,800步 → \|θ\| ≤ 1e-2 | 1e-2 | 累积二阶矩单调性 |
| INV-OPT-005 | RMSProp 在 f(θ)=θ² 上收敛：lr=0.1,β=0.99,400步 → \|θ\| ≤ 1e-3 | 1e-3 | 指数滑动二阶矩收敛 |
| INV-OPT-006 | Nesterov 动量在 f(θ)=θ² 上收敛：lr=0.1,μ=0.9,300步 → \|θ\| ≤ 1e-3 | 1e-3 | Nesterov 收敛性 |
| INV-OPT-007 | AdamW 解耦权重衰减：零梯度下 θ_k = θ0·(1−lr·wd)^k，与优化器逐元素偏差 ≤ 1e-12 | 1e-12 | 解耦衰减闭式 |
| INV-OPT-008 | SGD+Momentum 在 2D 二次 f=½(x²+y²) 收敛：lr=0.1,μ=0.9,200步 → ‖θ‖ ≤ 1e-3 | 1e-3 | 梯度下降收敛 |

### `autodiff-forge.js`（INV-AUT-001..008，8 条）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-AUT-001 | 前向 Dual 求导 f(x)=x²·sin(x) 与解析 2x·sin+ x²·cos 一致 | 1e-12 | 乘积/正弦链式求导 |
| INV-AUT-002 | 反向梯度 f(x,y)=e^{xy}+sin(x) 与解析偏导及 gradCheck(≤1e-6) 一致 | 1e-6 | 多变量反向 VJP |
| INV-AUT-003 | 前向 vs 反向单变量梯度一致（同 f(x)=x²·sin(x)） | 1e-12 | 双向模式一致性 |
| INV-AUT-004 | 链式 f(x)=sin(e^{x²}) 反向梯度与解析 f'=cos(e^{x²})·e^{x²}·2x 一致 | 1e-12 | 多层复合链式法则 |
| INV-AUT-005 | 二阶前向 Dual2 对 f(x)=sin(x) 给出 f''=−sin(x) 一致 | 1e-12 | 高阶前向 |
| INV-AUT-006 | 反向向量梯度 f(v)=Σvᵢ² → ∇=2v，与 gradCheck(≤1e-6) 一致 | 1e-6 | Rⁿ→R 梯度 |
| INV-AUT-007 | 反向 f=x·y·z → (yz,xz,xy) 与解析一致 | 1e-12 | 多输入乘积偏导 |
| INV-AUT-008 | 反向 5 维函数 gradCheck 全分量 ≤ 1e-6 | 1e-6 | 数值梯度对照 |

### `loss-act-norm-forge.js`（INV-LAN-001..008，8 条）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-LAN-001 | RMSNorm 归一分量 RMS(x/rms) = 1 | 1e-12 | RMSNorm 定义 |
| INV-LAN-002 | LayerNorm 归一分量零均值且方差 = 1 | 1e-12 | LayerNorm 定义 |
| INV-LAN-003 | GELU(0)=0 精确；且 GELU 在 [0,∞) 非减 | 0 | GELU 性质（负区导数可负，非全局单调） |
| INV-LAN-004 | GELU tanh 近似与精确 erf 式在采样点最大偏差 ≤ 2e-2 | 2e-2 | 近似保真 |
| INV-LAN-005 | SwiGLU 与独立重实现一致 | 1e-12 | 参照实现 vs 主实现 |
| INV-LAN-006 | CE 梯度 = softmax−onehot；且 CE ≥ 0 | 1e-12 | 交叉熵梯度定义 |
| INV-LAN-007 | MSE 梯度 = (2/n)(y−t) 与解析一致；MSE=mean((y−t)²) | 1e-12 | MSE 定义 |
| INV-LAN-008 | CE = −log(softmax(l)[target]) 与独立计算一致；Swish(0)=0 | 1e-12 | CE 与 Swish 定义 |

### `scheduler-forge.js`（INV-SCH-001..008，8 条，学习率调度器）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-SCH-001 | 线性 warmup 闭式：t≤warmup 逐点等于 lrMax·(t+1)/warmup（t=0 非零，规避零学习率死区） | 0 | 逐点闭式复算 |
| INV-SCH-002 | 余弦端点：cosineSchedule 在 t=warmup 恰为 lrMax、t=total−1 闭式误差 0、t=total 夹取为 lrMin | 0 | 端点解析值代入 |
| INV-SCH-003 | 余弦 warmup 后单调不增（全差分 ≤ 0，且要求实质下降） | −1e-9 | 差分扫描 |
| INV-SCH-004 | 阶跃边界：stepDecay 恰在 dropEvery 整数倍边界切换，与 lr·dropFactor^floor(t/dropEvery) 逐点相等 | 0 | 闭式复算 |
| INV-SCH-005 | 线性衰减等差：相邻步差恒定且等于 (lrMin−lrMax)/(total−warmup)，终点为 lrMin | 1e-12 | 差分闭式 |
| INV-SCH-006 | 余弦重启周期相位：任意 t 的 lr 等于「周期内相位」闭式复算（周期数/相位/端点三重校验） | 1e-12 | 闭式复算 |
| INV-SCH-007 | oneCycle 三段：warmup 段线性等差、峰值为 lrMax 且出现在 pctStart 边界、终点 = lrMax/(div·100)、余弦段单调 | 1e-12 | 分段闭式 |
| INV-SCH-008 | 非法输入全部真抛（lrMax≤0 / warmup<0 / total≤warmup / dropEvery≤0 / power<0 / t 非有限数 / 非函数 fn 等 26 探针）+ 合法调用不抛 | 0 | 异常捕获 + 反向对照 |

> 阈值档位：闭式等价取 EXACT=0；单调取 MONO=−1e-9（要求实质非增，浮点容差）；等差/相位取 RESID=1e-12。所有断言含反空洞化项（如 INV-SCH-001 锁定 t=0 非零、INV-SCH-003 要求峰后确有下降、INV-SCH-007 峰值定位与触底前断言）。

### `secondorder-forge.js`（INV-SO-001..008，8 条，L3 二阶优化 / 拟牛顿 / 信赖域 / 高斯牛顿）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-SO-001 | 牛顿法二次收敛：对凸二次 f=½xᵀAx−bᵀx（A 对称正定），牛顿步 x₁=x₀−A⁻¹(Ax₀−b) 恰为极小点 x*=A⁻¹b，单步残差 ‖Ax₁−b‖ ≤ 1e-12·‖b‖（对拍高斯消元直接解）；一维 x₁=x₀−f'/f'' 为解析极小 EXACT | 1e-12 | 二次 1 步收敛解析对照 |
| INV-SO-002 | 共轭梯度(CG) 对 SPD 线性系统：Ax=b 在 ≤n 次迭代内残差 ‖Ax−b‖ ≤ 1e-10·‖b‖ 且与直接解一致；CG 下降方向 dₖ 满足 gₖᵀdₖ<0（下降性恒成立） | 1e-10 | CG vs 直接解对拍 + 下降性 |
| INV-SO-003 | BFGS/L-BFGS 收敛与正定性：终态 f≤初态 f、终止 ‖∇f‖ 单调不增；终止逆 Hessian 恒对称正定（手搓 2×2 特征分解证 λmin>0）；被接受步长满足曲率条件 yₖᵀsₖ>0（负曲率步跳过，证伪 SPD 崩塌） | 1e-10 | 收敛 + SPD 最小特征值 + 曲率条件 |
| INV-SO-004 | 强 Wolfe 线搜索：返回 α 必满足 Armijo f(x+αd)≤f₀+c₁αg₀ᵀd 且 abs(∇f(x+αd)ᵀd)≤c₂·abs(g₀ᵀd)；独立黄金分割局部极小与 Wolfe 返回点 f 值差 ≤1e-3；非下降方向(g₀ᵀd≥0)返回 α=0（证伪） | 1e-3 | Wolfe 三条件 + 黄金分割对拍 + 非下降方向证伪 |
| INV-SO-005 | 信赖域 dogleg 不劣于柯西点：m(s*)≤m(p_c)（差 ≤1e-12）；解恒在信赖域内 ‖s*‖≤Δ EXACT（经数值边界护持）；半径自适应（ρ≥0.75 扩张、ρ<0.25 收缩）；对纯二次收敛到解析极小 | 1e-12 | dogleg vs 柯西点双向 + 信赖域约束 + 半径自适应 |
| INV-SO-006 | 高斯牛顿 == 牛顿（线性残差）：R(x)=Ax−b 时高斯牛顿步与牛顿步 EXACT 相等（差 ≤1e-12）；非线性最小二乘每步残差范数 ‖R‖ 不增（独立重算） | 1e-12 | GN==牛顿解析对照 + 残差非增 |
| INV-SO-007 | 阻尼牛顿与 Hessian-free：加 λI 使 (H+λI) 恒 SPD（λmin>0）且 ∇fᵀd<0（下降，证伪退化解）；HF 对二次 d 与精确牛顿步 EXACT 相等（‖d−A⁻¹b‖≤1e-12）；非二次 d 满足 gᵀd<0 | 1e-12 | 阻尼 SPD + HF==精确牛顿 + 下降性 |
| INV-SO-008 | 拟牛顿校正对称性与割线：BFGS 逆更新与 DFP 恒对称 EXACT；BFGS 逆更新满足割线 H₊y=s（‖H₊y−s‖≤1e-12）；SR1 满足秩1 公式 EXACT；yᵀs>0 保 SPD、yᵀs≤0 跳过（证伪负曲率） | 1e-12 | 对称性 + 割线方程 + SR1 秩1公式 + 曲率护持 |

> 反空洞铁律落实（INV-SO-004/005）：非下降方向强 Wolfe 返回 α=0（证伪"总有步长"）、dogleg 经数值边界护持确保 ‖s*‖≤Δ 恒成立。灵魂不变量 INV-SO-001..008 经 `tests/independent/secondorder-check.js` 独立复核 8/8 通过（`node tests/independent/secondorder-check.js`，exit 0）。

---

## L4 — 注意力 / 位置编码

### `attention-forge.js`（INV-ATT-001..008，8 条，SDPA/MHA/GQA/因果）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-ATT-001 | 注意力权重每行和为 1 | 1e-12 | softmax 归一 |
| INV-ATT-002 | 输出 = P·V（独立重算对照） | 1e-12 | 注意力定义 |
| INV-ATT-003 | 等权 logits → 权重均匀 = 1/N，输出 = mean(V) | 1e-12 | 均匀注意 |
| INV-ATT-004 | 因果掩码：i<j 位置权重 = 0（下三角） | 1e-12 | 因果结构 |
| INV-ATT-005 | 数值稳定：logits=1e6 时输出有限且 ≈ 每行 argmax 对应 V | 1e-6 | 极端不溢出 |
| INV-ATT-006 | MHA 在 nHeads=1 时与 SDPA 一致 | 1e-12 | 多头退化 |
| INV-ATT-007 | GQA(nKVHeads=1) 与「所有头共享同一 KV」的 MHA 一致 | 1e-12 | GQA 退化 |
| INV-ATT-008 | softmax 平移不变：scores 整体 +c 后输出不变 | 1e-12 | softmax 移位不变性 |

### `rope-forge.js`（INV-ROPE-001..008，8 条，Rotary Position Embedding）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-ROPE-001 | 范数保持：‖RoPE(x,p)‖ = ‖x‖ | 1e-15 | 旋转正交 |
| INV-ROPE-002 | 逆变换：RoPE(RoPE(x,p),−p) = x | 1e-12 | 旋转可逆 |
| INV-ROPE-003 | 2D 显式旋转公式交叉校验（x0cosθ−x1sinθ, x0sinθ+x1cosθ） | 1e-12 | 定义对照 |
| INV-ROPE-004 | 相对位置不变：⟨RoPE(q,m),RoPE(k,n)⟩ 仅依赖 (m−n) | 1e-12 | RoPE 核心性质 |
| INV-ROPE-005 | 任意 base 下 2D 配对角 = pos·base^{−2i/d} 显式校验 | 1e-12 | 角度公式 |
| INV-ROPE-006 | 距离保持：‖RoPE(q,p)−RoPE(k,p)‖ = ‖q−k‖ | 1e-12 | 正交变换 |
| INV-ROPE-007 | 角度可加：RoPE(RoPE(x,p),p) = RoPE(x,2p) | 1e-12 | 旋转角叠加 |
| INV-ROPE-008 | 线性：RoPE(−x,p) = −RoPE(x,p) | 1e-12 | 旋转线性 |

### `transformer-forge.js`（INV-TRA-001..008，8 条，Transformer 模型组件）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-TRA-001 | 嵌入查找逐行精确：`tokenEmbedding(E,ids)` 与 `E[ids[i]]` 逐元素相等且为拷贝（越界抛错） | 0 | 查表定义 |
| INV-TRA-002 | 正弦位置编码闭式：`PE[pos][2i]=sin(pos/10000^(2i/d))`、`PE[pos][2i+1]=cos(...)`，随机 pos/d 采样最大偏差 ≤ 1e-12 | 1e-12 | Vaswani 2017 原式 |
| INV-TRA-003 | 预归一化残差恒等路径：attnFn/ffnFn 恒零时 `preNormBlock(x)==x` 逐元素 ≤ 1e-12 | 1e-12 | 残差连接保恒等 |
| INV-TRA-004 | 前向输出形状：`transformerForward.hidden` 形状 == (seq,d)，形状不合法抛错 | 0 | 形状契约 |
| INV-TRA-005 | 因果性：因果 attnFn 下改动未来 token 嵌入，已生成位置 i 的输出不变（含防空洞化断言：被扰动位置必须确实变化） | 1e-12 | 因果掩码结构 |
| INV-TRA-006 | SwiGLU 前馈零输入→零输出（无偏置）≤ 1e-12，且形状保持 (seq,d)→(seq,d) | 1e-12 | swish(0)=0 门控归零 |
| INV-TRA-007 | 0 层堆叠：`transformerEncoder(x,[],finalNorm) == finalNorm(x)` 逐元素 ≤ 1e-12 | 1e-12 | 空堆叠恒等 |
| INV-TRA-008 | 确定性：同 opts 两次 `transformerForward` 输出逐位相等 | 0 | 纯函数确定性 |

### `kv-cache-forge.js`（INV-KVC-001..008，8 条，KV 缓存与增量解码）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-KVC-001 | 写入读出一致：乱序 append 覆盖 [0,T) 后 getKV 与写入值逐位相等 | 0 | 缓冲区 round-trip |
| INV-KVC-002 | **增量解码等价全量前向**：T=8 逐 token append+decode 与一次性全序列因果 SDPA 对应位置输出逐元素一致（含非零断言） | 1e-12 | attention-forge sdpa 全量参照 |
| INV-KVC-003 | 容量边界：pos≥maxLen 或 layer≥nLayers（含负值/非整数/行宽非法/T 超限）时抛错，漏抛计数 0 | 0 | 异常契约 |
| INV-KVC-004 | 分页一致性：同一序列走块表分页布局与连续布局，getKV / 增量解码 / 统计逐位一致 | 0 | 块表间接寻址等价 |
| INV-KVC-005 | 多层隔离：写 layer0 后其余层 getKV 全零且有效长度不受污染 | 0 | 层间缓冲寻址隔离 |
| INV-KVC-006 | 确定性：同序列两次完整增量解码（默认 sdpa 路径与自定义 forwardFn 路径）输出逐位相等 | 0 | 纯函数确定性 |
| INV-KVC-007 | 预分配不重扩：写满 maxLen 时 utilization 精确 ==1 且底层 Float64Array 长度恒定 | 0 | 容量刚性 |
| INV-KVC-008 | 逐出窗口：evictRange(from) 后有效长度 ==from 且已逐出位置读出头为零 | 0 | 截断语义 |

### `position-encoding-forge.js`（INV-POS-001..008，8 条，位置编码族）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-POS-001 | ALiBi 因果结构：causal 模式所有 j>i 位 bias==-Infinity、下三角全有限（违例计数 0） | 0 | 因果掩码定义 |
| INV-POS-002 | ALiBi 斜率：首项==2^(-8/n)（n 头标准式）、严格递减、末项最小；n=1 时==2^-8（8 头与 Press 原式 [2^-1..2^-8] 逐位相等） | 1e-15 | Press et al. 2021 get_slopes 原式 |
| INV-POS-003 | ALiBi 距离线性闭式：`bias[h][i][j]==-slopes[h]·(j-i)`（j≤i，20 组随机） | 1e-12 | 线性偏置定义 |
| INV-POS-004 | 位置插值闭式：`positionInterpolation==pos/s` 且 `ropeWithInterpolation==rope(x,pos/s,base)` | 1e-12 | Chen et al. 2023 |
| INV-POS-005 | T5 桶映射：双向模式 d 与 -d 异桶、\|d\|≥maxDistance clamp 到远桶、桶索引∈[0,nBuckets)（违例计数 0） | 0 | Raffel et al. 2020 |
| INV-POS-006 | 偏置施加一致：`applyBias==scores+bias`（有限位）；-Infinity 偏置位保持 -Infinity 不 NaN | 1e-12 | 逐元素相加定义 |
| INV-POS-007 | 相对性：`relativePositionBias` 为 Toeplitz 结构，[i][j] 仅依赖 (j-i)（违例计数 0） | 0 | 相对位置定义 |
| INV-POS-008 | softmaxRows 对接：ALiBi 偏置施加后每行和==1（因果下三角）且 j>i 位归一化后==0 | 1e-12 | 偏置+softmax=因果掩码 |

### `moe-forge.js`（INV-MOE-001..008，8 条，混合专家）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-MOE-001 | top-k 门控归一：随机 gate logits 每 token top-k 权重和 == 1 且权重全非负 | 1e-12 | softmax 定义（top-k 内归一） |
| INV-MOE-002 | top-k 正确性：20 组随机 gate logits 选中集合 == logit 最大 k 个专家（违例计数 0） | 0 | 独立排序参照 vs 主实现 |
| INV-MOE-003 | 单专家退化：k=1 确定性门控下输出 == 被选中专家输出逐元素一致（权重=1） | 1e-12 | 退化情形参照专家直算 |
| INV-MOE-004 | 全专家退化：k=nExperts 时输出 == 全专家 softmax 加权平均（独立重实现） | 1e-12 | 独立双重循环参照 vs 主实现 |
| INV-MOE-005 | 负载均衡闭式：均匀门控 aux == 理论最小值 1.0，且随机 20 组 aux ≥ 1 | 1e-12 | Switch 公式 + Cauchy-Schwarz 下界 |
| INV-MOE-006 | 容量守恒：裁剪后每专家分派数 ≤ 容量上限，且总丢弃数 == 超容量数（精确整数断言） | 0 | 独立逐专家重算 vs 主实现 |
| INV-MOE-007 | 向量化 vs 独立重实现：moeForward 与逐 token 朴素双重循环输出逐元素一致 | 1e-12 | 参照实现 vs 主实现 |
| INV-MOE-008 | 确定性：同输入两次 moeForward、同 rng 种子两次 noisyGate 输出逐位相等 | 0 | 纯函数确定性定义 |

### `quant-forge.js`（INV-QNT-001..008，9 用例，模型量化）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-QNT-001 | 可表示值精确往返：bits=8 对称，整数网格值 x=k·scale（k∈[-127,127]）经 quantize→dequantize 逐元素误差 ≤ 1e-12 | 1e-12 | 网格点往返 |
| INV-QNT-002 | 量化误差有界：随机 20 组张量（bits=8/4 × 对称/非对称 × 逐张量/逐通道），max\|deq(quant(x))−x\| ≤ scale/2 | 1e-12 | round-to-nearest 半步长界 |
| INV-QNT-003 | 对称量化零点：symmetric 时 zeroPoint==0（所有通道，精确）；asymmetric 时 zeroPoint 为整数且在 [0,2^bits−1] | 0 | 零点定义校验 |
| INV-QNT-004 | 逐通道优于逐张量：32 通道 absmax 对数均匀 1e-2..1e2 的矩阵，perChannel SNR ≥ perTensor SNR | 0 | 同输入对照违例计数 |
| INV-QNT-005 | int4 范围与端点：bits=4 两种 scheme 存储码 ∈ [-8,7]；对称量化网格覆盖 [min,max]；端点重建误差 ≤ 1 步长 | 1e-12 | 范围+网格覆盖+端点断言 |
| INV-QNT-006 | 量化矩阵乘保真：非负随机权重矩阵（逐通道非对称），`quantMatvec` 与浮点 matvec 对 16 组激活汇总相对误差：int8 ≤ 1e-2、int4 ≤ 5e-2 且 int8 ≤ int4 | 1e-2 | 同输入对照+单调性 |
| INV-QNT-007 | 超块细化：4 段块间 scale 相差 100 倍级的张量，kQuantSuperBlocks 的 SNR ≥ 整张 per-tensor 量化的 SNR，块数 == ceil(n/blockSize) | 0 | 同输入对照+块数精确 |
| INV-QNT-008 | 确定性与 pack 往返：同张量两次 calibrate+quantize 参数与结果逐位相等；`unpackInt4(packInt4(v))==v` | 0 | 确定性定义+位压缩逆运算 |

> 独立复算：1024 元素 int4 打包恰为 **512 字节**（真位压缩，每字节 2 个）；误差界两种位宽下**恰好等于** scale/2（紧界）；逐通道 SNR 高出逐张量 **+5.44dB / +5.59dB**。

### `speculative-forge.js`（INV-SPC-001..008，8 条，推测解码）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-SPC-001 | **分布等价（皇冠明珠）**：draft≠target 双分布场景 N=20000 次，speculativeDecode 与 target-only 采样的首 token 经验分布各自与目标分布最大偏差 ≤ 0.03 | 0.03 | 两侧独立大样本 MC；MCSE≈0.0035，8σ≈0.028 |
| INV-SPC-002 | 贪心路径全接受：draft 每步 argmax==target argmax 时 greedyVerifyChunk 返回 k 且无拒绝；argmax 分歧位置 j 时返回 j | 0 | argmax 一致性定义（正温度缩放不变） |
| INV-SPC-003 | 每步产出上界：20 组随机场景 1 ≤ tokensAccepted+bonus ≤ k+1，且 ctx' 长度=ctx+产出、tokensAccepted 为草稿前缀、targetCalls=1/draftCalls=k | 0 | 拒绝早停 + bonus 恒存在 |
| INV-SPC-004 | 拒绝采样无偏：固定 q/p 单 token N=20000 次经验频率与 q 最大偏差 ≤ 0.03 | 0.03 | 恒等式 min(p,q)+(q−p)₊=q |
| INV-SPC-005 | 接受率单调：draft==target 时接受率==1（精确）；3 档差异增大（解析 Σmin(p,q)≈0.91/0.74/0.48）经验接受率单调不增 | 0 | Σmin(p,q)=1−TV(p,q) 随差异单调降 |
| INV-SPC-006 | 单 token 退化：k=1 时 speculativeDecode 与 target-only 独立重实现在 L=3 个位置的经验分布各自与目标分布一致 | 0.03 | k=1 拒绝采样边际=目标分布 |
| INV-SPC-007 | 目标调用有界：生成 n token，targetCalls ≤ n 且 ≤ draftCalls/k+n；draft==target 时 targetCalls==ceil(n/(k+1))（精确，证明确有加速） | 0 | 每步≥1 token + 全接受每步 k+1 token |
| INV-SPC-008 | 确定性：同种子两次 speculativeDecode 输出 token 序列与统计量逐位相等 | 0 | sfc32 确定性 + 纯函数式管线 |

> 独立复算（主代理自建测试）：目标 q=[0.5,0.25,0.15,0.10] vs 均匀草稿，N=20000，推测解码首 token 经验分布偏差 **0.0062**、目标单独采样 **0.0032**，两路径互差 0.0093（MCSE≈0.0033 的 2σ 内）→ 端到端分布等价成立。

### `generation-forge.js`（INV-GEN-001..008，8 条，端到端生成闭环 · 集成封顶）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-GEN-001 | 流式等价批量：同模型同 prompt 同种子，generateStream 回调 token 序列与 generate 返回 tokens 逐位一致（回调抛错向上传播） | 0 | 双路径逐元素比对 |
| INV-GEN-002 | **增量等价全量（核心）**：generate（KV 缓存增量路径）每步 logits 与 generateNoCache（每步全序列前向）对应位置逐元素一致 ≤ 1e-12 | 1e-12 | 黄金参照逐元素比对 |
| INV-GEN-003 | 前缀缓存等价：共享长度 L 前缀的 prompts 经 batchGenerate 复用前缀缓存的结果，与各自独立 generate 逐位一致（池命中计数防空洞化） | 0 | 批量 vs 独立逐位比对 |
| INV-GEN-004 | 停止条件精确：eosId 为模型第 3 步（0 基）必产 token 时输出长度==3 且不含 eosId；maxTokens=5 时长度<=5；两者同时设置取先到者 | 0 | 精确整数断言 |
| INV-GEN-005 | 确定性：同种子两次 generate 输出 tokens 逐位相等；不同种子贪心模式仍逐位相等；采样模式跨种子须有差异（防空洞化） | 0 | 逐位比对 |
| INV-GEN-006 | 与采样内核一致：generate（采样模式）产出的 token 序列 == sampling-forge 的 sample 对同一条 logits 序列逐次采样的结果（rng 种子派生一致） | 0 | 独立重放逐位比对 |
| INV-GEN-007 | 贪心确定性路径：temperature=1e-8、topK=1 时 generate 输出 == 每步 argmax 构成的确定性序列 | 0 | 独立 argmax 重实现逐位比对 |
| INV-GEN-008 | 不良输入抛错：prompt 为空数组 / maxTokens=0 / eosId 为负数 / 缺 sampling 且缺 rng（贪心除外）等，generate 抛错且合法输入不抛 | 0 | 异常契约漏抛计数 |

---

## L5 — 概率分布 / 图算法 / 强化学习 / 生成采样

### `prob-forge.js`（INV-PRB-001..008，8 条）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-PRB-001 | gaussianLogPdf(x,μ,σ) 与闭式 −0.5·log(2πσ²)−0.5·((x−μ)/σ)² 一致 | 1e-12 | 对数高斯定义 |
| INV-PRB-002 | klGaussian(m,s,m,s)==0 | 1e-12 | 同分布 KL=0 |
| INV-PRB-003 | klGaussian ≥ 0（10 组随机对） | 1e-12 | KL 非负性 |
| INV-PRB-004 | klGaussian(m0,s0,m1,s1) 与内联参考 0.5·(log(s1²/s0²)+(s0²+(m0−m1)²)/s1²−1) 一致 | 1e-12 | 参考重实现 |
| INV-PRB-005 | bernoulliPmf(1,p)+bernoulliPmf(0,p)==1 | 1e-12 | 概率归一 |
| INV-PRB-006 | bernoulliLogPmf(1,p)==log(p) | 1e-12 | 对数 PMF 定义 |
| INV-PRB-007 | sampleGaussian N=20000：\|样本均值−μ\|≤0.05 且 \|样本方差−σ²\|≤0.05 | 0.05 | MCSE 余量（约 5σ） |
| INV-PRB-008 | klBernoulli(p,q) 与 p·log(p/q)+(1−p)·log((1−p)/(1−q)) 一致且 p==q 时 ==0 | 1e-12 | 参考重实现 |

### `graph-forge.js`（INV-GRAPH-001..008，8 条）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-GRAPH-001 | BFS 在无向无权图上的跳数距离与手算真值一致（精确整数） | 0 | 手算对照 |
| INV-GRAPH-002 | 同图 Dijkstra(权=1) == BFS 跳数 | 1e-12 | BFS/Dijkstra 等价 |
| INV-GRAPH-003 | Dijkstra 主实现 vs 独立第二实现距离一致 | 1e-12 | 参考重实现 |
| INV-GRAPH-004 | BFS 父链重建路径长度 == 报告距离（精确） | 0 | 路径重建 |
| INV-GRAPH-005 | 不连通分量节点距离返回 Infinity（精确） | 0 | 连通性 |
| INV-GRAPH-006 | 含 0 权边图：Dijkstra == 暴力枚举全简单路径最小权 | 1e-12 | 暴力黄金参照 |
| INV-GRAPH-007 | 同图两次运行距离完全一致（精确） | 0 | 确定性 |
| INV-GRAPH-008 | 三角不等式 dist(u,w) ≤ dist(u,v)+dist(v,w) | 1e-12 | 度量性质 |

### `rl-forge.js`（INV-RL-001..008，8 条）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-RL-001 | 值迭代收敛：最终 max\|V_{k+1}−V_k\| ≤ 1e-9（γ=0.9） | 1e-9 | 收缩映射 |
| INV-RL-002 | 贝尔曼最优性自洽：V*(s)≈max_a[R+γΣPV*] | 1e-6 | 最优贝尔曼方程 |
| INV-RL-003 | 贪心策略值 == V*（策略评估） | 1e-6 | 最优策略值等价 |
| INV-RL-004 | 2 态闭式 MDP 最优 V 对照 [9,10] | 1e-6 | 闭式真值 |
| INV-RL-005 | REINFORCE 梯度 vs 期望回报有限差分 ≤ 1e-2（N≥20000） | 1e-2 | 策略梯度定理 |
| INV-RL-006 | 策略梯度上升提升均值回报（终值 ≥ 初值+0.05） | 0.05 | 策略梯度上升 |
| INV-RL-007 | γ=0.9 收敛(残差<1e-3) 且 γ=1 不收敛(残差>1e-3) | 0 | 收敛条件 |
| INV-RL-008 | Q=R+γΣPV* 与值迭代 Q 备份一致 | 1e-6 | Q 值定义 |

### `sampling-forge.js`（INV-SMP-001..008，8 条，生成采样内核）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-SMP-001 | 温度→0 收敛贪心：唯一最大值 logits，T=1e-8 固定种子抽 500 次全部等于 greedy（不一致计数=0） | 0 | T→0 时 softmax 退化为 one-hot |
| INV-SMP-002 | top-k 精确性：20 组随机 logits/k，过滤后非 −Inf 候选恰为 k 个且集合等于全排序前 k 名（违例计数=0） | 0 | 定义等价 |
| INV-SMP-003 | top-p 最小性：20 组随机 logits/p，核集合累积概率 ≥ p 且移除最小概率成员后累积 < p（违例计数=0） | 0 | Holtzman 核采样最小集定义 |
| INV-SMP-004 | 重复惩罚公式精确：正 logit→logit/coef、负 logit→logit*coef、未出现→不变，与内联参照逐元素偏差 ≤ 1e-12 | 1e-12 | HF 约定参照实现 |
| INV-SMP-005 | 正缩放不变：logits×3.7（c>0）后 greedy 与 top-1 全序不变（不一致计数=0） | 0 | 正仿射单调性 |
| INV-SMP-006 | 分布保真：[2,1,1] 偏态分布 N=20000 次抽样，经验频率与 softmax 目标最大偏差 ≤ 0.03（实测 2.78e-3） | 0.03 | MCSE≈0.0035，取 8σ 余量 |
| INV-SMP-007 | 确定性：同种子两次 sample 逐位相等；不同种子经验频率仍落在目标 ±0.05 内（违例=0） | 0 | sfc32 确定性 + 非塌缩 |
| INV-SMP-008 | 批量一致：batchSample 第 i 行 == 同一 rng 种子派生对第 i 行单独 sample 的结果（不一致计数=0） | 0 | 逐行共享连续 rng 流 |

### `diffusion-forge.js`（INV-DIF-001..008，8 条，扩散模型 DDPM/DDIM）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-DIF-001 | 前向加噪闭式：qSample == √ᾱ_t·x0+√(1−ᾱ_t)·ε（逐元素，实测 0）；N=200000 大样本 MC 下 E[x_t]≈√ᾱ_t·x0（容差=4σ/√N 向上取十进制档≈1e-2，实测 2.04e-3） | 1e-15 / 1e-2 | 逐元素闭式 + 大样本 MC |
| INV-DIF-002 | x0↔ε 互逆复合：predictX0(predictEps(x0)) 重构 x0、predictEps(predictX0(ε)) 重构 ε，双向 24 组最大误差 ≤ 1e-12（实测 2.33e-15） | 1e-12 | 互逆复合 |
| INV-DIF-003 | ᾱ_t 单调不增且全部 ∈(0,1]：线性与余弦两调度均验证（差分 ≤ 0、t=0 峰值、末点 > 0） | −1e-9 | 差分扫描 |
| INV-DIF-004 | 后验闭式：posteriorMeanVariance 的 μ/σ² 与独立重写的 Ho 2020 式(7)(8) 逐元素一致（实测 0）；t=0 时 μ≡x0、σ²≡0 | 1e-12 | 独立重实现对照 |
| INV-DIF-005 | **DDIM 确定性（真检验，非退化）**：eta=0 时 DDIM 步不注入随机噪声（零 rng 消耗），同种子生成的同一 x_T 与 epsPredFn 两遍 ddimSample 逐位相等；eta>0 同种子仍可复现、但与 eta=0 必分歧（epsPredFn 取与 x_T 无关的确定性向量，令 x_T 差异真实传导，杜绝退化恒等掩盖） | 0 | 双跑逐位比对 + eta=0/eta>0 分歧对照 |
| INV-DIF-006 | DDIM 终步闭式：tPrev=0 且 eta=0 时输出逐元素 == √ᾱ_0·predictX0(x_t,t,ε)（Song 2021 式(12) 终支 cf≡0 特例，实测 0） | 1e-12 | 闭式对照 |
| INV-DIF-007 | SNR 单调：snr(t)=ᾱ_t/(1−ᾱ_t) 随 t 单调不增，t=0 处取最大值且逐点等于 ᾱ_0/(1−ᾱ_0) | 1e-12 | 差分扫描 + 端点闭式 |
| INV-DIF-008 | 非法输入全部真抛（t 非整数/t 越界/betas 长度 0/beta≤0 或 ≥1/形状非法等 24 探针）+ 5 合法对照全部不抛 | 0 | 异常捕获 + 反向对照 |

> 阈值推导（INV-DIF-005）：原始模块自检曾用 `fn=(x,t)⇒predictEps(x0,t,x,ᾱ)` 的退化 epsPredFn，使 DDIM 首步即解出 x0、与 x_T 完全无关，"异种子亦相等"仅是恒等掩盖——该不变量空洞。已改为与 x_T 无关的确定性 epsPredFn，并新增"eta=0 与 eta>0 同种子必分歧"的非空洞化对照，确证 eta=0 确实消除了随机注入。修复后经 `tests/independent/INV-DIF-check.js` 独立复核：10 组皇冠不变量逐项独立重算全绿。

### `vi-forge.js`（INV-VI-001..008，8 条，变分推断 / ELBO / 重参数化 / VAE）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-VI-001 | ELBO 分解闭式：elbo == −recon − kl，且 elbo ≤ E_q[log p(x|z)]（因 KL≥0，max|·|≤1e-12） | 1e-12 | 分量一致性 + KL 非负性 |
| INV-VI-002 | 重参数化梯度检验：dL/dmean、dL/dlogStd 解析（pathwise）vs 中心差分 max 相对误差 ≤1e-6（灵魂不变量） | 1e-6 | 独立有限差分交叉验证 |
| INV-VI-003 | 高斯 KL 闭式 vs 手搓黄金参照 max|·|≤1e-12；KL≥0；KL=0 ⟺ 同参数（灵魂不变量） | 1e-12 | 独立黄金参照 |
| INV-VI-004 | VAE 损失分解：loss=recon+β·kl；β=0 逐位 == recon；β=1 == recon+kl | 1e-12 | 闭式分量一致 |
| INV-VI-005 | **重参数化确定性（反空洞铁律）**：①同 seed 同(μ,σ) 重采样 z 逐位相等(≤1e-15) ②异 seed 序列可复现且分歧 ③σ=0 vs σ>0 必分歧(>0) ④大样本 MC mean(z)≈μ、std(z)≈σ（统计容差 ≥10σ） | 0 | 确定性 + 分歧 + 矩收敛三重 |
| INV-VI-006 | 均值场 KL 可加：Σ_i KL(q_i‖p_i) == 整体 KL(q‖p)，max|·|≤1e-12 | 1e-12 | 逐维求和等价 |
| INV-VI-007 | 摊销推断一致性：同 x 多次编码(确定性前向) mean/logStd 逐位相等；近邻 x 的 mean 接近(≤0.05)；远距 x 的 mean 分歧 | 0 | 确定性 + 平滑性 + 非平凡 |
| INV-VI-008 | ELBO 单调上升：玩具数据单步梯度上升后 ELBO_new ≥ ELBO_old − 1e-6（优化方向正确，断言不降） | 1e-6 | 正确 ascent 方向 |

> 反空洞铁律落实（INV-VI-005）：原易退化为「同 seed 相等」的恒等掩盖——已新增「σ=0 与 σ>0 同种子必分歧」对照，确证采样式 z 的差异真实传导；并经 `tests/independent/vi-check.js` 独立复核：手搓高斯 KL 黄金参照、闭式 ELBO 分量、重参数化确定性三重逐项确认全绿。

### `mcmc-forge.js`（INV-MC-001..008，8 条，L5 MCMC 采样 / Metropolis / MH / Gibbs / NUTS / HMC / 退火）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-MC-001 | 对称提议下 Metropolis ≡ Metropolis-Hastings：同 seed 同提议流两独立实现轨迹逐位相等；非对称提议下两链必分歧（证伪提议修正项生效）；SHA-256 FIPS 180-4 已知向量防护（'abc'/''） | 0 | 双独立实现 JSON 逐位比对 + 分歧断言 + 已知向量 |
| INV-MC-002 | MH 细致平衡（离散两状态）：解析核 P=Q·α 满足 π_i·P_ij = π_j·P_ji（EXACT ≤1e-15，恒接受核同式违反 >0.1）；60000 步经验频率≈π（10σ）、经验转移频率≈解析 P（10σ） | 1e-15 | 解析细致平衡恒等式 + 经验转移矩阵 10σ 检验 + 恒接受证伪 |
| INV-MC-003 | RWM 大样本矩收敛：N(3,2²) 目标 42000 步（烧除 2000）样本均值/方差 z 分数 ≤10（三种子）；KS 检验 vs 解析正态 D≤1.63/√ESS；最优窗宽 2.38 下接受率≈0.44±0.05；Robbins-Monro 调优收敛到 0.44±0.05；恒接受随机游走方差 >100（证伪） | 10σ | 手搓 ESS/KS + 解析矩 + 调优收敛 + 随机游走证伪 |
| INV-MC-004 | Gibbs 条件矩：BVN(ρ=0.6) 62000 步，x|y 回归斜率≈0.6、残差方差≈0.64、边际方差≈1（均 10σ）；错误条件方差(1.0)链边际方差偏离 >0.3（证伪）；lag-1 自相关 <0.95（非平凡） | 10σ | 解析条件矩 + 回归估计 + 错误条件证伪 |
| INV-MC-005 | leapfrog 结构性质：时间反演可逆 (x,p)→(x',p')→(x,-p) 误差 ≤1e-12；保体积（数值 Jacobian 行列式 ≈1，≤1e-6）；能量误差有界（L=8/40/160 最大 |ΔH| ≤ 0.05）而显式 Euler 同配置漂移（err(160) > 5·err(8)） | 1e-12 | 有限差分 Jacobian + 能量界 + Euler 对照 |
| INV-MC-006 | NUTS 与混合 MC 效率：病态 2D 高斯（sd=[1,10]）上 NUTS ESS > 3×RWM ESS；U-turn 终止占比 >0.5；dual averaging 后接受率 ∈[0.5,0.95]；超大 eps(5.0) 接受率 <0.2（证伪）；混合 MC（随机轨迹长 U{1..8}）ESS > 2×RWM 且轨迹长频率≈均匀 | 0.05 | ESS 黄金对比 + U-turn 统计 + 轨迹长直方图 + 超大 eps 证伪 |
| INV-MC-007 | dual averaging 自适应步长：目标接受率 0.65 收敛到 0.65±0.05、目标 0.9 收敛到 0.9±0.05；且 eps(0.9) < eps(0.65)（严格小 >2%，证伪调优响应目标） | 0.05 | Hoffman-Gelman 公式 + 双目标对照 |
| INV-MC-008 | 退火：同 seed 全链逐位相等（EXACT）+ 异 seed 分歧；调度几何段严格递减、T∈[Tmin,T0]；退火 8 种子 ≥5/8 从局部模 -5 达全局模 +5 且末态平均对数密度高于冷链 +0.5；冷链 3 种子全链不越过 x≤1.5（困于局部模，证伪对照）；降温致锐（末段方差 < 首段方差/4） | 0 | 双跑逐位 + 调度结构 + 8 种子全局模统计 + 冷链对照 |

> 反空洞铁律落实（INV-MC-002/003/005/008）：恒接受核必违反细致平衡（证伪"接受即正确"）、恒接受随机游走方差 >100（证伪收敛）、显式 Euler 漂移劣于 leapfrog（证伪积分器无关）、冷链困于局部模（证伪退火无效）。灵魂不变量 INV-MC-001/002/005 经手搓黄金参照（解析细致平衡 + 有限差分 Jacobian）Δ=0，并经 `tests/independent/mcmc-check.js` 独立复核（3 连跑确定性验证通过）。

### `estimate-forge.js`（INV-EST-001..008，8 条，L5 参数估计 / 假设检验 / 拟合优度 / 区间 / 似然）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-EST-001 | MLE 与矩估计闭式 EXACT + 牛顿收敛：正态 μ̂=x̄、σ̂²=1/n Σ(x−x̄)² 与直接计算 EXACT；伯努利 p̂=x̄、泊松 λ̂=x̄ EXACT；MLE-Newton ≤25 步得分 ‖∂ℓ/∂θ‖≤1e-9；矩估计与 MLE 在对称族一致 | 1e-12 | 闭式对拍 + 牛顿得分收敛 + 一致性 |
| INV-EST-002 | MAP 与贝叶斯共轭 EXACT：正态-正态后验均值 (τ₀μ₀+nτx̄)/(τ₀+nτ) 与解析 EXACT；平坦先验(τ₀→0) MAP 退化为 MLE EXACT；后验均值 == MAP（正态共轭众数=均值） | 1e-12 | 共轭公式对拍 + 退化证伪 + 众数=均值 |
| INV-EST-003 | KS 检验 D 与手算 EXACT（固定 11 点小样本）；渐近 p 随 D 严格单调且 D=0→p=1、D 极大→p→0（K(x)=1−2Σ(−1)^{k-1}e^{−2k²x²} 自洽）；功效 N(3,1) 拒绝率 >90%、N(0,1) 拒绝率 <15% | 1e-9 | D 手算对拍 + 单调边界 + 功效双向 |
| INV-EST-004 | 卡方检验：拟合优度 χ²=Σ(O−E)²/E 与手算 EXACT（固定 8 类频数）；p 与 χ²(k) CDF 一致；2×2 独立性 χ² 与解析对照且 df=1 EXACT；功效 观察=期望×2 拒绝率 >90%；df=类别数−1−估计参数数 EXACT | 1e-9 | 统计手算 + CDF 对拍 + 功效 + 自由度 |
| INV-EST-005 | Bootstrap 非参数：B=2000 均值 SE_boot 与解析 SE/√n 相对误差 ≤15%；百分位 95% CI 中心 ≈ 样本均值（abs(中心−x̄)≤2·SE）；bootstrap 均值 ≈ 原始均值 | 1e-6 | SE 对照 + CI 中心 + 均值一致 |
| INV-EST-006 | 置信区间覆盖：正态/t/Wilson 95% CI 重复抽样覆盖真值率 ∈[0.92,0.98]（蒙特卡洛 2000 次，固定种子）；t CI 与解析一致；Wilson 上下界与手算 EXACT | 1e-2 | 蒙特卡洛覆盖 + 解析公式 + 边界 |
| INV-EST-007 | Welch t 检验：t 统计量与手算 EXACT；学生 t CDF 对照（CDF(0;df)=0.5 EXACT、单调递增）；I 类错误率 ∈[0.03,0.07]（α=0.05，2000 次）；效应量 d=0.8 功效 >0.8 | 1e-9 | t 公式 + t CDF + 错误率双向 |
| INV-EST-008 | 似然比/LRT + 拟似然/稳健/经验似然：LRT 2(ℓ₁−ℓ₀) 中位 p ∈[0.4,0.6]（H₀ 下 2000 次，df=2，χ²(2) 对照）；拟似然对 Poisson 退化为 MLE(mean=x̄) EXACT 且 φ≈1；Huber 对污染离群降权（abs(Huber−真值)<abs(均值−真值)）；经验似然极大点 EXACT 在样本均值（−2logR(x̄)=0）且偏离 >0.5 后 −2logR>0.5（证伪恒零退化解） | 1e-9 | LRT χ² 对照 + QL 退化 + Huber 降权 + EL 极大点双向 |

> 反空洞铁律落实（INV-EST-003/006/007/008）：KS 双向功效（N(3,1) 高拒、N(0,1) 低拒，证伪单侧偏倚）、CI 覆盖双向边界 [0.92,0.98]（证伪过窄/过宽）、I 类错误 ∈[0.03,0.07]（证伪膨胀/失效）、经验似然偏离均值后 −2logR>0.5（证伪极大点恒零退化）。灵魂不变量 INV-EST-001..008 经 `tests/independent/estimate-check.js` 独立复核 8/8 通过（`node tests/independent/estimate-check.js`，exit 0）。

---

## L6 — 深度强化学习

### `dqn-forge.js`（INV-DQN-001..008，8 条，DQN 系算法族）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-DQN-001 | TD 目标闭式：tdTarget == r+γ·max(qNext)·(1−done)，随机 20 组偏差 ≤ 1e-12（实测 0）；done 时精确等于 r（截断自举） | 1e-12 | 内联闭式复算 |
| INV-DQN-002 | 经验回放 FIFO：capacity=64 推入 100 条 → size==64、保留第 37..100 条（逐位断言）、最旧 36 条逐出；sample 不修改缓冲区 | 0 | 逐位断言 + 快照对照 |
| INV-DQN-003 | 回放采样无偏：20000 次 sample(batch=4) 共 8 万抽，各经验被抽中频率与均匀分布最大偏差 ≤ 0.03（实测 1.85e-3，约 20σ 余量） | 0.03 | 蒙特卡洛频率检验 |
| INV-DQN-004 | ε-贪婪口径：ε=0 恒返回 argmax（20 组×200 次违例 0）；ε=1 频率−均匀 ≤ 0.03；ε=0.5 频率 vs 0.5·均匀+0.5·argmax ≤ 0.05（实测 1.45e-3） | 0.05 | 经验频率 vs 闭式混合分布 |
| INV-DQN-005 | ε 调度单调：线性/指数两种模式 step∈[0,2000] 全范围单调不增且 ≥ epsEnd（违例 0） | 0 | 单调性扫描 |
| INV-DQN-006 | Polyak 闭式：tau=0.01 时 target == 0.01·online+0.99·target（≤1e-12）；tau=1 时 target == online（精确，硬同步特例） | 1e-12 | 闭式复算 |
| INV-DQN-007 | **DQN 收敛到值迭代**（皇冠）：5 态链式 MDP 训练后贪心策略与 rl-forge 值迭代最优策略一致，贝尔曼残差 ≤ 1e-2（实测 2.62e-12） | 1e-2 | 独立黄金参照（非自证）；另经 `tests/independent/INV-DQN-007-check.js` 三重独立复核：手搓 P/R 值迭代 V\*=[0.729,0.81,0.9,1.0,0] 对闭式误差 1.1e-16、env.spec() 同源审计 0 差异、3 种子 Q 值收敛至 Q\* ~1e-11、独立复算残差与模块自报逐位相等 |
| INV-DQN-008 | Double DQN 抑制过估计：诱惑构造 MDP（目标网络含被高估动作）上 TD 目标逐元素 ≤ 普通 DQN，20 组随机批量违例 0 | 0 | max 算子占优 + 构造实例 |

> 阈值推导（INV-DQN-007）：one-hot 线性 Q 网 ≡ 表格 Q 学习，定点即 VI 的 Q\*；SGD 稳态 TD 误差 ~ O(lr·σ_TD/√batch)：lr=0.05、batch=32、收敛后单样本 TD 误差 σ≈0.05 ⇒ 残差≈5e-4，取 1e-2（≈20× 余量，且远小于 Q 值尺度 γ⁴≈0.66，收敛意义不丢失）。终态（goal）episode 进入即终止、永不被动作触及，其 Q\*≡0 是定义值而非贝尔曼约束，残差与策略一致性均在非终止态全集（4 态 × 2 动作 = 8 对）上度量。

### `td-forge.js`（INV-TD-001..008，8 条，表格强化学习基础 / 值迭代 / SARSA / Q-learning / TD(λ)）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-TD-001 | 值迭代收敛到最优值：max\|V−V*\|≤1e-6 且固定点残差 max_s\|V−T*V\|≤1e-6（V* 为独立手搓值迭代） | 1e-6 | 独立手搓值迭代对照 + 固定点残差扫描 |
| INV-TD-002 | 策略迭代值单调不降（逐元素违例数==0）且末轮 V 与值迭代最优 V 一致（≤1e-6） | 1e-6 | 逐元素单调性扫描 + 独立 VI 对照 |
| INV-TD-003 | 贝尔曼最优算子严格收缩：‖T*V−T*V'‖∞/‖V−V'‖∞ ≤ γ（γ=0.9 ⇒ ≤0.9+1e-12） | 1e-12 | 两随机 V 解析算子比 |
| INV-TD-004 | **Q-learning 收敛到最优 Q**（皇冠）：贪心策略与值迭代最优策略 mismatch==0 且 max\|Q−Q*\|≤1e-2（独立手搓 Q* 黄金参照） | 1e-2 | 独立手搓 Q* 对照 |
| INV-TD-005 | TD(0) 估计收敛到 v^π：与蒙特卡洛（多次 episodic 累积回报平均）对照 max\|V_TD−V_MC\|≤5e-2（统计容差留 ≥10σ 余量） | 5e-2 | 多重种子 MC + 步衰减 α |
| INV-TD-006 | SARSA(ε-贪婪) vs Q-learning 非退化：ε=0 时 max\|Q_SARSA−Q_QL\|≤1e-9 且 ε>0/异种子下 max\|·\|>0（反空洞铁律：必断言分歧） | 1e-9 | 贪心一致性 + 异设置分歧断言 |
| INV-TD-007 | 资格迹退化：λ=0 与 td0 一致（≤1e-9 逐位）；λ=1 接近蒙特卡洛（≤统计容差）；λ=0 与 λ=1 必分歧（>0，反空洞） | 1e-9 | 逐位一致 + MC 对照 + 分歧断言 |
| INV-TD-008 | 期望 SARSA 目标闭式：贪心时目标=r+γ·maxQ（与 Q-learning 目标≤1e-12）；优势零均值 E_a[π(a)A(a)]=0（≤1e-9） | 1e-9 | 目标公式复算 + 零均值恒等式 |

> 反空洞铁律落实（INV-TD-006/007）：凡「退化/一致性」必同时断言「不同设置必分歧」，杜绝恒等掩盖；灵魂不变量 INV-TD-004 经 `tests/independent/td-check.js` 手搓 Q* 黄金参照独立复核（mismatch=0、max\|Q−Q*\|≈1e-15）。

### `policygrad-forge.js`（INV-PG-001..008，8 条，策略梯度 / REINFORCE / PPO / GAE）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-PG-001 | 策略梯度定理梯度检验：高斯策略 ∇_μ=(a−μ)/σ²、∇_σ=((a−μ)²/σ²−1)/σ 与中心差分对照 max 相对误差 ≤1e-6 | 1e-6 | 手搓解析梯度 vs 有限差分 |
| INV-PG-002 | 基线不减偏置：E[∇logπ·(G−b)]=E[∇logπ·G]（Σπ∇logπ≈0，偏差≤1e-12）且 b=mean(G) 方差比<1（多重子验证） | 1e-12 | score function 零均值 + 中心化降方差 |
| INV-PG-003 | **REINFORCE 收敛到最优策略**（皇冠）：5 态链 MDP 训练后贪心策略与值迭代最优策略 mismatch=0（独立黄金参照：rl-forge 值迭代，非自证） | 0 | 黄金参照交叉验证 |
| INV-PG-004 | 优势函数零均值：Σ_a π(a|s)A(s,a)≈0（max\|·\|≤1e-9）且 A=Q−V 闭式一致 | 1e-9 | A(s,a)=Q(s,a)−V(s)，V=ΣπQ |
| INV-PG-005 | GAE 退化：λ=0 → TD 残差 δ（≤1e-12 逐位）；λ=1 → MC 回报减 V_t（≤1e-9）；且 λ=0 与 λ=1 必分歧（>0，反空洞） | 1e-9 | 闭式 + 分歧铁律 |
| INV-PG-006 | 演员评论家价值一致：critic 估计 v^π 收敛到同策略 TD 评估真 v^π（对照 max\|·\|≤统计容差，多重子验证） | 1e-2 | 同策略 TD0 一致性 |
| INV-PG-007 | PPO 裁剪代理目标占优非退化：裁剪≤未裁剪（逐元素 ≥20 批违例 0）且存在严格小于（strict>0，证明非恒等） | 0 | min 算子占优 + 构造实例 |
| INV-PG-008 | 优势符号与策略改进方向一致：∇_θJ≈E[∇logπ·A] 与 A 符号一致（≤1e-6）；或 MC 验证沿 ∇logπ(a*) 上升改进>0 | 1e-6 | 策略梯度定理符号一致性 |

> 反空洞铁律落实（INV-PG-005/007）：GAE 的 λ=0 与 λ=1 强制分歧、PPO 裁剪必存在严格小于（403 组）以证明非恒等；灵魂不变量 INV-PG-003 经 rl-forge 值迭代独立黄金参照交叉验证，并经 `tests/independent/policygrad-check.js` 复核。

---

## L7 — 向量检索 / ANN 索引（检索 / 图 / 系统族 F7）

### `retrieval-forge.js`（INV-RT-001..008，8 条，L7 向量检索核心 / 余弦 / 穷举 top-k / MIPS / RRF）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-RT-001 | 余弦度量：正交向量余弦=0、平行向量余弦=1（黄金参照逐元素核对）；且随机对严格 ∈(−1,1) 非恒定 | 1e-15 | 手搓黄金 cosine + 非恒定证伪 |
| INV-RT-002 | 穷举 top-k 与手搓全排序取前 k 逐位一致(EXACT)；且 k=1 与 k=5 结果必分歧 | 0 | 独立手搓 argsort 黄金参照 + 分歧断言 |
| INV-RT-003 | int8 标量量化反量化后距离相对误差 ≤5%；且量化距离 ≠ 精确距离(近似)；误差随码本精度收敛 | 5e-2 | 反量化误差界 + 近似断言 + 收敛断言 |
| INV-RT-004 | 默认穷举 ANN recall@k=1.0(EXACT)；切到近似标志后 recall<1.0(证明近似路径有效) | 0 | recall@k 对照 + 近似路径证伪 |
| INV-RT-005 | 穷举 MIPS top-k 与手搓内积排序前 k 一致；最大内积 ≠ 最近邻(证伪度量混淆) | 0 | 独立手搓内积黄金参照 + 分歧断言 |
| INV-RT-006 | 重排改变初召回顺序；RRF 融合后 top-1 为高频高排名项且融合序 ≠ 任一输入序 | 0 | 重排顺序变更 + RRF 融合序证伪 |
| INV-RT-007 | 阈值过滤返回集 ⊆ 全集且均 ≥ 阈值；阈值升高结果数减少(单调) | 0 | 子集关系 + 单调性证伪 |
| INV-RT-008 | 同 seed 数据集两次检索逐位相等；异 seed 必分歧(确定性+可复现) | 0 | 双跑比对 + 异种子分歧断言 |

> 反空洞铁律落实（INV-RT-004/008）：ANN 默认穷举 recall=1.0（精确）必断言近似路径 recall<1.0 证伪"作弊全量"；确定性同 seed 逐位相等必断言异 seed 分歧。灵魂不变量 INV-RT-002 经手搓 argsort 黄金参照交叉验证 mismatch=0，并经 `tests/independent/retrieval-check.js` 复核。

### `ann-forge.js`（INV-ANN-001..008，8 条，L7 ANN 索引结构全集 / KD树 / 球树 / LSH / Annoy / PQ / IVFPQ / ScaNN / HNSW）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-ANN-001 | KD 树精确 kNN：recall@k=1.0(=手搓穷举黄金) 且平均访问节点数 ≪N | 1e-12 | 手搓穷举黄金 top-k 对照 + 访问计数扫描 |
| INV-ANN-002 | 球树精确 kNN：recall@k=1.0(=手搓穷举黄金) 且平均访问节点数 ≪N | 1e-12 | 手搓穷举黄金 top-k 对照 + 访问计数扫描 |
| INV-ANN-003 | LSH 族(超平面/p稳定/旋转) 良好配置 recall≥0.8(=手搓黄金)；退化配置 recall<1.0(证伪作弊全量) | 0 | 手搓穷举黄金 top-k 对照 + 退化配置证伪 |
| INV-ANN-004 | Annoy 良好配置 recall≥0.8；退化(nTrees=1) recall<1.0 | 0 | 手搓穷举黄金 top-k 对照 + 退化配置证伪 |
| INV-ANN-005 | PQ/IVFPQ 重建相对误差 ≤20%；量化(ADC)距离 ≠ 精确 L2 距离(近似)；误差随子空间数/码本增大收敛 | 0.2 | 重建误差比 + ADC vs 精确距离 + M/Ks 收敛扫描 |
| INV-ANN-006 | HNSW 召回随 efSearch 单调不减；efSearch=1 召回明显低于大 ef(证伪平凡)；大 ef recall≥0.9 | 1e-9 | efSearch 扫描单调 + ef=1 对比如下 |
| INV-ANN-007 | 粗/细量化：同 seed 两次运行中心 id / 码本逐位相等；异 seed 分歧 | 0 | 同 seed 双跑一致性 + 异 seed 分歧 |
| INV-ANN-008 | 确定性：同 seed 数据集建索引+查询逐位相等；异 seed 必分歧 | 0 | 双跑比对 + 异种子分歧断言 |

> 反空洞铁律落实（INV-ANN-003/004/006）：各近似索引必断言退化配置 recall<1.0 证伪"作弊返回全量"；HNSW ef=1 必低于大 ef。灵魂不变量经手搓穷举黄金 top-k 交叉验证（mismatch=0），并经 `tests/independent/ann-check.js` 复核。

### `gnn-forge.js`（INV-GNN-001..008，8 条，L7 图神经网络 / GCN / GraphSAGE / GAT / MPNN）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-GNN-001 | 对称归一化 D̂^-1/2 Â D̂^-1/2 与手搓闭式逐元素一致；且星形图行和偏离 1(≠1，证伪行归一混淆) | 0 | 手搓闭式对照 + 行和证伪 |
| INV-GNN-002 | GCN 前向 ≡ 手搓循环展开黄金参照；W=0 时输出逐位 ≡ 偏置广播 | 0 | 手搓黄金参照(不调用模块) + 退化对照 |
| INV-GNN-003 | GraphSAGE mean 聚合 ≡ 手搓邻居均值；mean/sum/max 三聚合器两两必分歧(≥1.86) | 2.2e-16 | 手搓均值对照 + 三聚合器分歧断言 |
| INV-GNN-004 | GAT 边注意力 ≡ 手搓参照、每节点出边权重和=1；非均匀(偏离 0.298)且 LeakyReLU 负斜率生效 | 2.2e-16 | 手搓注意力对照 + 行和 + 非均匀 + 斜率探针 |
| INV-GNN-005 | MPNN(度缩放+sum) ≡ GCN(无W) 逐元素等价(消息传递统一视角)；mean ≡ 手搓行均值；mean/sum 分歧 | 1.1e-16 | GCN 等价实证 + 聚合器分歧 |
| INV-GNN-006 | 2 层 GCN 节点分类训练 300 步 acc=1.0(loss 0.747→3.7e-3)；未训练 acc=0.5<1.0(证伪平凡) | 0 | 训练收敛 + 未训练对照 |
| INV-GNN-007 | 邻居采样同 seed 逐位相等；20 对异 seed 全部分歧；出度 ≤ min(k,deg) 约束成立 | 0 | 双跑比对 + 异 seed 分歧 + 约束断言 |
| INV-GNN-008 | 全流水线(采样→GCN→GAT→SAGE→MPNN→训练)同 seed 140 维输出逐位相等；异 seed 分歧 | 0 | 双跑比对 + 异 seed 分歧 |

> 反空洞铁律落实（INV-GNN-003/004/006）：三聚合器必分歧、注意力必非均匀、未训练必低于训练——证伪"恒等/平凡"掩盖。灵魂不变量 INV-GNN-002/004 经手搓黄金参照 mismatch=0，并经 `tests/independent/gnn-check.js` 复核（5 连跑确定性验证通过）。

### `sparse-forge.js`（INV-SPR-001..008，8 条，L7 稀疏计算 / CSR / SpMV / SpMM / 剪枝）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-SPR-001 | 稠密→COO→CSR→稠密 往返逐元素一致；稀疏 storageRatio<0.5 且稠密对照≥1(证伪压缩边界) | 0 | 手搓稠密黄金往返 + 双边界断言 |
| INV-SPR-002 | SpMV ≡ 手搓稠密 matvec；乘法次数=nnz≪n·m(证伪全扫作弊) | 0 | 手搓稠密黄金对照 + 乘法计数 |
| INV-SPR-003 | SpMM ≡ 手搓 matmul；分配律 A(B+C)=AB+AC 成立 | 1e-10 | 手搓黄金对照 + 分配律 |
| INV-SPR-004 | CSR 规范性：rowPtr[0]=0、单调不减、rowPtr[n]=nnz；行内 colIdx 严格递增；乱序 COO 输入仍产出规范 CSR | 0 | 规范断言 + 乱序输入等价 |
| INV-SPR-005 | nnz 统计 ≡ 直接计数(EXACT)；稀疏 compression≤密度 且稠密对照≈1(证伪边界) | 0 | 双计数对照 + 边界断言 |
| INV-SPR-006 | 幅度剪枝 kept=⌊nnz·keep⌋、min&#124;kept&#124;≥max&#124;dropped&#124;、剪后 ≡ 保留集黄金参照、keep=1.0 逐位一致 | 0 | 排序一致性 + 黄金参照 + 非恒等断言 |
| INV-SPR-007 | 稀疏向量点积 ≡ 稠密点积；乘法次数=正交集大小；不相交 → dot=0 | 0 | 稠密黄金对照 + 交集计数 |
| INV-SPR-008 | 同 seed 全流水线(生成/转换/乘/剪枝)逐位相等；异 seed 分歧 | 0 | 双跑比对 + 异 seed 分歧 |

> 反空洞铁律落实（INV-SPR-002/005/006）：乘法次数=nnz 证伪"稠密全扫作弊"、稠密对照证伪压缩率声称、剪枝 keep=1.0 恒等对照。灵魂不变量 INV-SPR-001/002 经手搓稠密黄金参照 mismatch=0，并经 `tests/independent/sparse-check.js` 复核（3 连跑确定性验证通过）。

### `graphalg-forge.js`（INV-GA-001..008，8 条，L7 图算法 / DFS / A* / PageRank / MST / SCC / 中心性）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-GA-001 | DFS 访问序 ≡ 手搓显式栈实现；且 DFS 序 ≠ BFS 序（证伪遍历混淆） | 0 | 手搓栈实现对照 + 遍历序分歧断言 |
| INV-GA-002 | A*：h=0 时 dist ≡ 手搓 Dijkstra 黄金参照；可采纳 h>0 探索节点数严格少于 Dijkstra；不可采纳 h(×10) 允许次优（证伪边界） | 1e-15 | 手搓 Dijkstra 对照 + 探索计数 + 次优捕获 |
| INV-GA-003 | PageRank：Σrank=1、残差<1e-10、≡ 30000 轮长迭代黄金参照；且排名 ≠ 均匀 1/n（证伪平凡） | 1e-10 | 闭式归一 + 长迭代对照 + 非均匀断言 |
| INV-GA-004 | 连通分量 ≡ 手搓 Union-Find；SCC ≡ 手搓 Kosaraju 独立实现；同一有向图 SCC 数 ≠ 无向连通分量数（证伪混淆） | 0 | 手搓并查集/Kosaraju 双对照 |
| INV-GA-005 | 拓扑排序：全边 pos[u]<pos[v]；升/降序两种合法实现序可不同但均合法（非唯一）；有环/自环必抛错 | 0 | 全边序验证 + 双合法序 + 环抛错 |
| INV-GA-006 | MST：Kruskal ≡ Prim 总权；恰 n−1 边无环；任意换树边为非树边后总权 ≥ MST（证伪平凡最优） | 1e-15 | 双实现互校 + 并查集无环 + 割性质证书 |
| INV-GA-007 | 三角计数：小图(n≤20) ≡ 手搓三重循环；排序法操作数 ≪ n³（性能真） | 0 | 手搓暴力对照 + 操作数比例 |
| INV-GA-008 | Brandes 介数中心性 ≡ 手搓逐对 BFS 路径计数（唯一最短路图上逐位精确）；同 seed 全流水线逐位相等、异 seed 分歧 | 0 | 手搓逐对 BFS 对照 + 双跑/异种子 |

> 反空洞铁律落实（INV-GA-002/003/005/006）：不可采纳启发必捕获次优、PageRank 必非均匀、拓扑排序非唯一且环必抛错、MST 换边必不优——证伪"恒等/平凡"掩盖。灵魂不变量 INV-GA-002/006/008 经手搓黄金参照 Δ=0，并经 `tests/independent/graphalg-check.js` 复核（3 连跑确定性验证通过）。

### `serving-forge.js`（INV-SV-001..008，8 条，L7 推理服务 / 批处理 / 融合 / 调度 / 内存池 / 限流）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-SV-001 | 批处理（含 padding）有效区输出 ≡ 逐条串行输出；批大小 1 与 N 单条输出一致（批无关性） | 1e-15 | 手搓串行黄金参照 + 批无关断言 |
| INV-SV-002 | 算子融合 ≡ 未融合串联输出；且融合后 kernel 调用次数 < 未融合（性能真，非数值作弊） | 1e-15 | 数值等价 + 调用计数 |
| INV-SV-003 | 内核调度序满足全部依赖 pos[u]<pos[v]；不同优先级权重产生不同合法序（非退化）；含环依赖必抛错 | 0 | 全边序验证 + 异权重分歧 + 环抛错 |
| INV-SV-004 | 内存池：随机分配/释放序列下任意时刻活跃块区间两两不重叠；复用后 malloc 次数 < 请求块次数；池耗尽返回 null 而非越界 | 0 | 手搓区间扫描 + 复用计数 + 安全边界 |
| INV-SV-005 | 流式管线：任意 chunkSize（含 1/不整除/超长）分块拼接 ≡ 全量结果 | 0 | 5 种 chunkSize 全量对照 |
| INV-SV-006 | 负载均衡：等权 N 请求 M worker 承载差 ≤1；加权 w=[3,1] 承载比精确 9:3；加权序 ≠ 均权序（非退化） | 0 | 计数断言 + 加权比例 + 序分歧 |
| INV-SV-007 | FIFO 出队序 ≡ 入队序；优先级队列 ≡ 手搓 (priority,seq) 排序参照；同一请求集两队列出队序必分歧 | 0 | 手搓参照 + 序分歧断言 |
| INV-SV-008 | 令牌桶：固定请求序列+逻辑时钟下放行/拒绝序列两次运行逐位一致；窗口 [0,T] 放行数 ≤ capacity+rate·T；rate 减半必分歧 | 0 | 重放确定性 + 闭式上界 + 速率分歧 |

> 反空洞铁律落实（INV-SV-002/003/006/008）：融合必减少调用数、调度异权重必异序、加权必异于均权、限流速率减半必变化——证伪"恒等/平凡"掩盖。所有时序基于**逻辑时钟**（非真实时间）保证确定性。灵魂不变量 INV-SV-001/004 经手搓黄金参照 mismatch=0，并经 `tests/independent/serving-check.js` 复核（3 连跑确定性验证通过）。

### `eval-forge.js`（INV-EVL-001..008，8 条，L7 评测指标 / precision / F1 / AUC / perplexity / BLEU）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-EVL-001 | 混淆矩阵各行和 = 该类真实样本数；precision/recall ≡ 手搓 TP/(TP+FP)·TP/(TP+FN)；不平衡类上 macro ≠ micro（证伪平均方式混淆） | 1e-15 | 手搓 PR 对照 + macro/micro 分歧断言 |
| INV-EVL-002 | F1 ≡ 手搓 2PR/(P+R)；全对=1、全错=0（边界）；随机预测 F1 ∈(0,1)（非退化） | 1e-15 | 手搓 F1 对照 + 双边界 + 非退化 |
| INV-EVL-003 | AUC ≡ 手搓 Mann-Whitney U（暴力 O(nP·nN) + 平均秩双路交叉）；完美=1、逆序=0；随机打乱 ≈0.5（10σ 余量） | 1e-15 | 双路黄金参照 + 边界 + 统计容差 |
| INV-EVL-004 | perplexity ≡ exp(手搓平均 NLL)；恒定概率 p 时 ≡ 1/p（闭式）；均匀高熵 perplexity > 尖峰低熵（单调性） | 1e-10 | 手搓 NLL 对照 + 闭式 + 熵单调 |
| INV-EVL-005 | BLEU ≡ 手搓 n-gram 精度几何平均 × brevity penalty；cand=ref→1；短候选 BP<1 受罚（证伪）；零匹配=0 | 1e-10 | 手搓 n-gram/BP 独立实现(分隔符不同) + 边界 |
| INV-EVL-006 | p50/p95/p99 ≡ 手搓排序取位；p50 ≤ p95 ≤ p99（单调）；不同样本集分位数必分歧 | 1e-15 | 手搓排序对照 + 单调性 + 分歧 |
| INV-EVL-007 | throughput ≡ count/seconds；加倍计数吞吐翻倍（比例）；seconds=0/负/NaN/Inf 抛 TypeError（安全边界） | 0 | 闭式对照 + 比例 + 除零抛错 |
| INV-EVL-008 | 同输入两次预测一致性率=1.0；微扰(≤1e-6)率≈1；大扰(≥0.5)率明显下降（证伪恒等）；同 seed 逐位复现、异 seed 分歧 | 0 | 双跑/扰动对照 + 种子分歧 |

> 反空洞铁律落实（INV-EVL-002/003/005/007/008）：随机预测 F1 必非 0/1、随机打乱 AUC 必≈0.5 而非 1、短候选 BLEU 必 <1、除零必抛错、大扰动一致性必下降——证伪"恒等/平凡"掩盖。灵魂不变量 INV-EVL-003/005 经手搓黄金参照 Δ=0（AUC 双路交叉），并经 `tests/independent/eval-check.js` 复核（3 连跑逐字节一致）。

### `flow-forge.js`（INV-FLW-001..008，8 条，L7 网络流 / 最大流=最小割 / 二分匹配 / 社区 / 着色）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-FLW-001 | Edmonds-Karp ≡ Dinic 最大流值；流守恒（除源汇外入流=出流）；容量约束 0≤f≤c | 0 | 双实现互校 + 守恒/容量扫描 |
| INV-FLW-002 | **最大流 = 最小割**（核心定理）：maxflow ≡ 暴力枚举所有顶点子集的最小割容量；非最小割严格 > 最小割 | 0 | 暴力枚举 2^n 割对照(n=8，58 个割严格更大) |
| INV-FLW-003 | maxflow ≤ 源点出边容量和 且 ≤ 汇点入边容量和（存在严格小实例）；零容量网络 value=0 | 0 | 上界断言 + 零容量边界 |
| INV-FLW-004 | Hopcroft-Karp 匹配数 ≡ 手搓 Kuhn 独立实现；匹配合法（每点至多一条）；K_{m,m}=m（闭式 m=1..6）；无边=0 | 0 | 手搓 Kuhn 对照 + 合法性 + 闭式 |
| INV-FLW-005 | 二分图匹配数 ≡ 归约流网络后的 maxflow（同一批随机图）——两项微指令内在一致 | 0 | 自建归约网络 ≡ HK ≡ Kuhn 三方互校 |
| INV-FLW-006 | 标签传播收敛后标签稳定（再迭代一轮不变）；K5∪K5 社区数=2、无边图每点自成一社区、K8=1 | 0 | 稳定点复算 + 结构断言（证伪单社区平凡） |
| INV-FLW-007 | 贪心着色任意边两端颜色不同（全边零冲突）；着色数 ≥ 手搓暴力团数；K_n=n（n=1..8）；≤Δ+1 | 0 | 全边冲突扫描 + 团数下界 + 闭式 |
| INV-FLW-008 | 同 seed 随机图全流水线（最大流+最小割+匹配+LPA+着色）两次逐位相等；异 seed 分歧 | 0 | 双跑比对 + 异种子分歧 |

> 反空洞铁律落实（INV-FLW-002/003/006）：最小割经**暴力枚举顶点子集**独立验证（非自证）、上界存在严格小实例（非平凡恒等）、社区发现断言 K5∪K5=2 与无边图 n（证伪单社区）。灵魂不变量 INV-FLW-002/004 经手搓黄金参照（暴力割枚举 + Kuhn 独立实现）Δ=0，并经 `tests/independent/flow-check.js` 复核（3 连跑逐位一致）。

### `serialize-forge.js`（INV-SER-001..008，8 条，L7 序列化与存储 / 四编解码器 / 内存映射 / 增量·快照 / 版本·WAL·灾恢 / 分区·对象 / 缓存 / 校验和）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-SER-001 | 四编解码器 roundtrip 保真与规范确定性：JSON(规范键序)/二进制(TLV)/列式(字典+位包)/protobuf(varint+zigzag) 对随机对象图（3 种子）decode 后 deepEqual EXACT；encodeJson 与 encodeBinary 键序无关（同值不同键序 → 字节级 EXACT 相等）；紧凑性：数值负载 binary ≤0.75·json、小整数 protobuf ≤0.6·binary；损坏流（魔数翻转/截断/超长 varint/尾随垃圾）必抛错 | 0 | deepEqual EXACT + 字节级对拍 + zigzag 已知向量 + 损坏必抛 |
| INV-SER-002 | 列式 vs 行式无免费午餐：低基数列（每列 4 distinct）列式 ≤0.5·直存 且 ≤0.7·行式（字典+位包致胜）；高基数列（每列全 distinct int）列式 ≥1.15·直存（字典开销致亏）；两种负载 decode roundtrip EXACT 且同表编码字节确定 EXACT | 0 | 字典+位打包尺寸双向对比 + roundtrip + 确定性对拍 |
| INV-SER-003 | 内存映射分页语义：脚本访问序列下 LRU 逐出序 EXACT；缺页计数 == 首次触页数 EXACT；脏页 flush 后 backing 与映射视图逐字节一致 EXACT 且 flush 前 backing 不含该写；越界读/写均抛错 | 0 | 页表 LRU 对拍 + 写回前后一致性 + OOB 证伪 |
| INV-SER-004 | 增量编码与快照隔离：applyDelta(base, encodeDelta(base,target))==target EXACT（3 种子删/插/换段变异）；同输入 delta 操作序列与字节确定 EXACT；空变异仅剩单 copy 且 ≤16 字节；快照深冻结隔离——快照后写入不影响快照、restore 后状态==快照、版本历史不灭、快照对象 Object.isFrozen 且嵌套全冻 | 0 | diff-apply 对拍 + COW 隔离 + 深冻结 + 历史保全 |
| INV-SER-005 | 版本化与事务持久性：版本 id 严格递增且旧版本内容不变 EXACT；WAL 交错事务（两事务+未提交+中止+同键覆写）恢复态 EXACT（仅已提交按 lsn 序）；每个截断前缀 p∈[0,N] 恢复态 == p 前最后提交态 EXACT（原子性+持久性全前缀证明）；抽去提交记录的日志中对应写入不可见（证伪）；灾恢 backup→restore EXACT、备份翻转一字节必抛、PITR 到任一提交点状态 EXACT | 0 | 两阶段恢复 + 全前缀崩溃模拟 + 备份双校验 + 提交抽离证伪 |
| INV-SER-006 | 校验和家族：已知向量 EXACT（CRC32("123456789")=0xCBF43926、CRC32("")=0、FNV-1a-64("")=0xcbf29ce484222325、FNV-1a-64("a")=0xaf63dc4c8601ec8c、SHA256("abc")=ba7816bf…、SHA256("")=e3b0c442…）；单比特翻转雪崩均值 ≥0.3 且 ≤0.75；1000 随机输入无碰撞 EXACT；块级校验（16 块）定位损坏块 EXACT；记录级翻转仅污染对应记录 | 0 | 标准测试向量 + 雪崩统计 + 定位对拍 + 选择性污染 |
| INV-SER-007 | 分区与对象存储：哈希分区确定（同 key 同分区，100 次 EXACT）且 8 桶近似均匀（最大桶偏差 ≤50%）；一致性哈希环（4 节点×50 虚节点）：同 key 同节点 EXACT、增加 1 节点移动键比例 ≤2/(n+1) 且 ≥0.02、删除 1 节点 ≤0.5；范围分区路由 EXACT；对象存储内容寻址去重、同键覆写产生新版本且旧版本内容 EXACT、head 元数据 EXACT、删除后 get 抛错、异内容 etag 必异 | 0 | 确定性+均匀性+移动界双向+去重与版本对拍 |
| INV-SER-008 | 缓存语义：LRU（容量 3）脚本序列逐出序 EXACT、size≤capacity 恒成立 EXACT、hits+misses==gets EXACT、capacity=1 连续 put 必逐出前值（证伪）；TTL 可注入时钟：过期前 hit / 过期 miss EXACT、ttl=0 即 miss、stale-while-revalidate 过期仍返回旧值且置 revalidate、refresh 后新值生效 EXACT、过期项被惰性清除 | 0 | Map 插入序对拍 + 注入时钟 + SWR 语义 + 边界证伪 |

> 反空洞铁律落实（INV-SER-002/003/005/008）：列式无免费午餐双向断言（低基数必赢、高基数必亏）、缺页计数必等于首次触页数、WAL 每个截断前缀必恢复到最后提交态（原子性+持久性全前缀证明而非单点自证）、capacity=1 连续 put 必逐出前值（证伪 LRU 空转）。灵魂不变量 INV-SER-001/003/005/006 经手搓黄金参照（独立 SHA-256 + 独立 LRU 仿真 + 独立 ARIES 重放 + 独立 CRC/FNV/雪崩统计）交叉验证，并经 `tests/independent/serialize-check.js` 独立复核（R1–R8 全绿）。

---

## 统计汇总

| 层 | 模块数 | 不变量条数 | 失败 |
|----|--------|-----------|------|
| L1 | 2 | 12 | 0 |
| L2 | 4 | 28 | 0 |
| L3 | 7 | 56 | 0 |
| L4 | 9 | 72 | 0 |
| L5 | 8 | 64 | 0 |
| L6 | 3 | 24 | 0 |
| L7 | 9 | 72 | 0 |
| **合计** | **42** | **328** | **0** |

> 注：本表按**不变量 ID 数**统计，合计 **328 条**。运行时 case 数为 **335**（`npm run verify` 实测），差异来自多条子用例：
> core-numeric 11 个 case 对应 8 条 ID（INV-NUM-002/006 含 a/b/c 子用例，+3）；core-verify 2 个 case 对应 4 条 ID（INV-VER-001/003 为不计数探针，−2）；quant-forge 9 个 case 对应 8 条 ID（INV-QNT-008b 为 pack 往返辅助用例，+1）；ann-forge 13 个 case 对应 8 条 ID（O04/O06/O08/O09/O11 为覆盖用例，+5）。净 +7。
> W17 新增 mcmc-forge（L5，8 条 ID / 8 个 case，1:1）与 serialize-forge（L7，8 条 ID / 8 个 case，1:1），合计 +16 ID / +16 case，净差不变。
> W18 新增 secondorder-forge（L3，8 条 ID / 8 个 case，1:1）与 estimate-forge（L5，8 条 ID / 8 个 case，1:1），合计 +16 ID / +16 case，净差不变。
> 分层归属以源码 `selfCheck` 声明的 `layer` 为准；`attn-forge.js` 声明 L4，但本表沿其文档归位列于 L3 段，总数不变；retrieval-forge / ann-forge / gnn-forge / sparse-forge 声明 L7（F7 检索·图·系统族），W13 由 L5 规划位升格至 L7。

---

## 如何复现

```bash
npm run verify             # 42 模块 · 335 用例 · 0 失败
node tests/independent/INV-DQN-007-check.js  # INV-DQN-007 三重独立复核（手搓黄金参照）
node tests/independent/INV-DIF-check.js      # INV-DIF 皇冠不变量逐项独立重算
node tests/independent/td-check.js          # INV-TD 表格RL 皇冠不变量独立复核
node tests/independent/policygrad-check.js  # INV-PG 策略梯度 皇冠不变量独立复核
node tests/independent/vi-check.js          # INV-VI 变分推断 皇冠不变量独立复核
node tests/independent/retrieval-check.js  # INV-RT 向量检索 皇冠不变量独立复核
node tests/independent/ann-check.js        # INV-ANN ANN索引 皇冠不变量独立复核
node tests/independent/gnn-check.js        # INV-GNN 图神经网络 皇冠不变量独立复核
node tests/independent/sparse-check.js     # INV-SPR 稀疏计算 皇冠不变量独立复核
node tests/independent/graphalg-check.js  # INV-GA 图算法 皇冠不变量独立复核
node tests/independent/serving-check.js   # INV-SV 推理服务 皇冠不变量独立复核
node tests/independent/eval-check.js     # INV-EVL 评测指标 皇冠不变量独立复核
node tests/independent/flow-check.js     # INV-FLW 网络流 皇冠不变量独立复核
node tests/independent/mcmc-check.js     # INV-MC  MCMC 皇冠不变量独立复核（R1–R8 全绿）
node tests/independent/serialize-check.js # INV-SER 序列化·存储 皇冠不变量独立复核（R1–R8 全绿）
node tests/independent/secondorder-check.js # INV-SO 二阶·拟牛顿 皇冠不变量独立复核（R1–R8 全绿）
node tests/independent/estimate-check.js  # INV-EST 估计·检验 皇冠不变量独立复核（R1–R8 全绿）
node tools/verify-l10.js   # L10 1024 项 / done 246 / planned 778
node tools/check-reuse.js  # 复用合规 0 ERROR
node tools/verify-l10.js   # L10 1024 项 / done 214 / planned 810
```

每条不变量均在对应模块的 `selfCheck()` 中以 `caseResult(...)` / `cases.push(...)` 注册，并在文件头 `@invariant ID|描述|容差|验证方法` 一行声明，供 `tests/harness.js` 与文档双重校验。
