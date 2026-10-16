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

## L5 — 检索 / 概率分布 / 图算法 / 强化学习 / 生成采样

### `retrieval-forge.js`（INV-RET-001..008，8 条）

| ID | 不变量 | 容差 | 验证方法 |
|----|--------|------|----------|
| INV-RET-001 | 余弦相似度 ∈ [−1,1] | 1e-12 | 定义边界 |
| INV-RET-002 | 相同向量余弦=1；正交向量余弦=0 | 1e-12 | 定义 |
| INV-RET-003 | top-k 返回 k 个结果且按相似度降序 | 1e-12 | 召回顺序 |
| INV-RET-004 | top-k 与暴力全排序结果一致（集合+顺序） | 1e-12 | 精确召回 |
| INV-RET-005 | 归一化向量余弦 = 点积 | 1e-12 | 余弦/点积等价 |
| INV-RET-006 | 余弦检索对查询缩放不变（c≠0 不改变 top 顺序） | 1e-12 | 缩放不变性 |
| INV-RET-007 | L2 距离 = ‖a−b‖；非负且零点为同向量 | 1e-12 | L2 定义 |
| INV-RET-008 | 零向量安全：与任意向量余弦=0（不抛错不 NaN） | 1e-12 | 边界安全 |

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

---

## 统计汇总

| 层 | 模块数 | 不变量条数 | 失败 |
|----|--------|-----------|------|
| L1 | 2 | 12 | 0 |
| L2 | 4 | 28 | 0 |
| L3 | 6 | 48 | 0 |
| L4 | 9 | 72 | 0 |
| L5 | 6 | 48 | 0 |
| L6 | 1 | 8 | 0 |
| **合计** | **28** | **216** | **0** |

> 注：本表按**不变量 ID 数**统计，合计 **216 条**。运行时 case 数为 **218**（`npm run verify` 实测），差异来自多条子用例：
> core-numeric 11 个 case 对应 8 条 ID（INV-NUM-002/006 含 a/b/c 子用例，+3）；core-verify 2 个 case 对应 4 条 ID（INV-VER-001/003 为不计数探针，−2）；quant-forge 9 个 case 对应 8 条 ID（INV-QNT-008b 为 pack 往返辅助用例，+1）。净 +2。
> 分层归属以源码 `selfCheck` 声明的 `layer` 为准；`attn-forge.js` 声明 L4，但本表沿其文档归位列于 L3 段，总数不变。

---

## 如何复现

```bash
npm run verify             # 28 模块 · 218 用例 · 0 失败
node tests/independent/INV-DQN-007-check.js  # INV-DQN-007 三重独立复核（手搓黄金参照）
node tests/independent/INV-DIF-check.js      # INV-DIF 皇冠不变量逐项独立重算
node tools/check-reuse.js  # 复用合规 0 ERROR
node tools/verify-l10.js   # L10 1024 项 / done 85 / planned 939
```

每条不变量均在对应模块的 `selfCheck()` 中以 `caseResult(...)` / `cases.push(...)` 注册，并在文件头 `@invariant ID|描述|容差|验证方法` 一行声明，供 `tests/harness.js` 与文档双重校验。
