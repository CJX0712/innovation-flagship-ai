# 不变量总览 · INVARIANTS

> 创新旗舰 AI 的**硬不变量（hard invariant）清单**——每条不变量都有 ≥1 条可被独立交叉验证的金标准。
> 当前基线：**19 模块 · 145 条不变量 · 0 失败**（`npm run verify` 实测）。

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

---

## 统计汇总

| 层 | 模块数 | 不变量条数 | 失败 |
|----|--------|-----------|------|
| L1 | 2 | 12 | 0 |
| L2 | 4 | 28 | 0 |
| L3 | 5 | 40 | 0 |
| L4 | 3 | 24 | 0 |
| L5 | 5 | 40 | 0 |
| **合计** | **19** | **144** | **0** |

> 注：本表按**不变量 ID 数**统计，合计 **144 条**。运行时 case 数为 **145**（`npm run verify` 实测），差异来自多条子用例：
> core-numeric 11 个 case 对应 8 条 ID（INV-NUM-002/006 含 a/b/c 子用例，+3）；core-verify 2 个 case 对应 4 条 ID（INV-VER-001/003 为不计数探针，−2）。净 +1。
> 分层归属以源码 `selfCheck` 声明的 `layer` 为准；`attn-forge.js` 声明 L4，但本表沿其文档归位列于 L3 段，总数不变。

---

## 如何复现

```bash
npm run verify             # 19 模块 · 145 用例 · 0 失败
node tools/check-reuse.js  # 复用合规 0 ERROR
node tools/verify-l10.js   # L10 1024 项 / done 42 / planned 982
```

每条不变量均在对应模块的 `selfCheck()` 中以 `caseResult(...)` / `cases.push(...)` 注册，并在文件头 `@invariant ID|描述|容差|验证方法` 一行声明，供 `tests/harness.js` 与文档双重校验。
