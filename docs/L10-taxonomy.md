# L10 微指令层分类法（1024 项）

> 十层架构第十层：8 族(F0-F7) × 8 组(G0-G7) × 16 微指令(O0-O15) = **1024** 项微指令级操作。
> 原则：**零重写 · 集大成** —— 每条标注世界顶级 `@reuse` 参照；`done`=已被 forge 模块覆盖，`planned`=W3-W7 后续落地。

| 状态 | 数量 |
| --- | --- |
| done | 143 |
| planned | 881 |
| **合计** | **1024** |

## 全景映射

| 族 | 名称 | 层 | 组数 | 微指令(×16) |
| --- | --- | --- | --- | --- |
| F0 | 数值基础 | L1 | 8 | 128 |
| F1 | 线性代数 | L2 | 8 | 128 |
| F2 | 优化 | L3 | 8 | 128 |
| F3 | 自动微分 | L3 | 8 | 128 |
| F4 | 注意力与 Transformer | L4 | 8 | 128 |
| F5 | 概率与生成 | L5 | 8 | 128 |
| F6 | 强化学习 | L6 | 8 | 128 |
| F7 | 检索/图/系统 | L7 | 8 | 128 |

## F0 · 数值基础（层 L1）

### G0 · 确定性随机

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F0-G0-O00` | sfc32 PRNG | L1 | done | core-numeric sfc32 |
| `L10-F0-G0-O01` | splitmix32 种子派生 | L1 | done | core-numeric splitmix32 |
| `L10-F0-G0-O02` | mulberry32 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G0-O03` | xoshiro128 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G0-O04` | PCG 发生器 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G0-O05` | 高斯采样 Box-Muller | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G0-O06` | 高斯采样 Acklam 逆CDF | L1 | done | core-numeric normInv(Acklam JSS2003) |
| `L10-F0-G0-O07` | 拒绝采样基础 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G0-O08` | 分层抽样 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G0-O09` | Fisher-Yates 打乱 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G0-O10` | 均匀分布 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G0-O11` | 伯努利分布 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G0-O12` | 泊松分布 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G0-O13` | 指数分布 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G0-O14` | 随机质量检验 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G0-O15` | 种子可复现保证 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G1 · 浮点稳定

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F0-G1-O00` | Kahan 求和 | L1 | done | core-numeric Kahan/Neumaier |
| `L10-F0-G1-O01` | Neumaier 求和 | L1 | done | core-numeric Neumaier |
| `L10-F0-G1-O02` | logsumexp 稳定 | L1 | done | stable-forge logsumexp |
| `L10-F0-G1-O03` | 稳定 softmax | L1 | done | stable-forge 稳定 softmax |
| `L10-F0-G1-O04` | log1p-exp 稳定 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G1-O05` | 数值下溢防护 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G1-O06` | 数值上溢防护 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G1-O07` | 舍入误差界 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G1-O08` | 互补误差函数 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G1-O09` | 双曲线稳定 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G1-O10` | 范数稳定 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G1-O11` | 行列式稳定 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G1-O12` | Gram-Schmidt 正交 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G1-O13` | 极坐标稳定 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G1-O14` | 对数稳定 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G1-O15` | 除法保护 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G2 · 哈希与校验

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F0-G2-O00` | 纯 JS SHA-256 | L1 | done | core-numeric sha256(FIPS180-4) |
| `L10-F0-G2-O01` | FNV-1a 占位 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G2-O02` | 校验和 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G2-O03` | 幂等校验 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G2-O04` | 双端一致哈希 | L1 | done | core-verify 双端一致 hash |
| `L10-F0-G2-O05` | 抗碰撞属性 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G2-O06` | 报告指纹 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G2-O07` | 模块指纹 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G2-O08` | 数据指纹 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G2-O09` | 增量哈希 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G2-O10` | 向量哈希 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G2-O11` | 矩阵哈希 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G2-O12` | 字符串哈希 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G2-O13` | 哈希对齐规范 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G2-O14` | salted 哈希 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G2-O15` | 哈希回归测试 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G3 · 向量化与并行

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F0-G3-O00` | 分块 GEMM | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G3-O01` | SIMD 式展开 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G3-O02` | 循环分块 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G3-O03` | 缓存友好布局 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G3-O04` | 向量归约 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G3-O05` | 向量映射 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G3-O06` | 向量扫描 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G3-O07` | 向量压缩 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G3-O08` | 并行 map | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G3-O09` | 并行 reduce | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G3-O10` | 任务分片 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G3-O11` | 算子流水线 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G3-O12` | 内存对齐 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G3-O13` | 零拷贝视图 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G3-O14` | 批处理 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G3-O15` | 流式处理 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G4 · 误差与容差

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F0-G4-O00` | 绝对误差 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G4-O01` | 相对误差 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G4-O02` | ULP 容差 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G4-O03` | 残差界 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G4-O04` | 正交容差 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G4-O05` | 求解容差 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G4-O06` | 梯度容差 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G4-O07` | 单调容差 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G4-O08` | 收敛容差 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G4-O09` | 概率容差 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G4-O10` | 组合误差 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G4-O11` | 条件数估计 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G4-O12` | 病态诊断 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G4-O13` | 稳定性阈值 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G4-O14` | 截断误差 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G4-O15` | 舍入阈值 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G5 · 数据结构

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F0-G5-O00` | 行优先矩阵 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G5-O01` | 零拷贝子块 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G5-O02` | 稀疏 CSR | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G5-O03` | 稀疏 CSC | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G5-O04` | 张量视图 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G5-O05` | 环形缓冲 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G5-O06` | 优先队列 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G5-O07` | 哈希表 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G5-O08` | B 树索引 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G5-O09` | 跳表 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G5-O10` | 位图 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G5-O11` | 布隆过滤器 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G5-O12` | 邻接表 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G5-O13` | 邻接矩阵 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G5-O14` | 栈 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G5-O15` | 双端队列 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G6 · 测试与验证

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F0-G6-O00` | 梯度数值检验 | L1 | done | core-verify gradCheck |
| `L10-F0-G6-O01` | 不变量断言 | L1 | done | core-verify 不变量断言 |
| `L10-F0-G6-O02` | 蒙特卡洛验证 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G6-O03` | 交叉验证 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G6-O04` | 数值回归 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G6-O05` | 精度基准 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G6-O06` | 确定性回归 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G6-O07` | 边界测试 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G6-O08` | 性质测试 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G6-O09` | KS 统计检验 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G6-O10` | 覆盖率统计 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G6-O11` | 模糊测试 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G6-O12` | 快照测试 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G6-O13` | 对照实现互比 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G6-O14` | 双路互比 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G6-O15` | 元不变量自检 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G7 · 工具与 IO

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F0-G7-O00` | JSON 报告 | L1 | done | core-verify forge-selfcheck/1.0 |
| `L10-F0-G7-O01` | 人类可读表 | L1 | done | core-verify formatTableSafe |
| `L10-F0-G7-O02` | CLI 解析 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G7-O03` | 退出码规范 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G7-O04` | 日志 stderr 分流 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G7-O05` | 进度条 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G7-O06` | 配置加载 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G7-O07` | 路径规范化 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G7-O08` | 环境变量 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G7-O09` | 模块发现 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G7-O10` | 聚合运行器 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G7-O11` | 浏览器适配 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G7-O12` | 离线运行 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G7-O13` | 错误码规范 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G7-O14` | 诊断输出 | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F0-G7-O15` | 报告 schema | L1 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

## F1 · 线性代数（层 L2）

### G0 · 矩阵乘法与分解

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F1-G0-O00` | 分块 GEMM | L2 | done | core-numeric gemm |
| `L10-F1-G0-O01` | 朴素 GEMM | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G0-O02` | 矩阵向量乘 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G0-O03` | 外积 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G0-O04` | Kronecker 积 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G0-O05` | LU 分解 | L2 | done | linalg-forge lu(Doolittle 部分主元) |
| `L10-F1-G0-O06` | Cholesky 分解 | L2 | done | linalg-forge chol |
| `L10-F1-G0-O07` | QR(Householder) | L2 | done | linalg-forge qr(Householder) |
| `L10-F1-G0-O08` | SVD(单边Jacobi) | L2 | done | linalg-forge svd(one-sided Jacobi) |
| `L10-F1-G0-O09` | 对称 EVD(Jacobi) | L2 | done | linalg-forge eigSym(Jacobi) |
| `L10-F1-G0-O10` | 三角求解 | L2 | done | linalg-forge solveLU |
| `L10-F1-G0-O11` | 最小二乘 | L2 | done | linalg-forge lstsq(QR/正规方程) |
| `L10-F1-G0-O12` | 矩阵求逆 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G0-O13` | 行列式 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G0-O14` | 迹 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G0-O15` | 范数族 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G1 · 特征值与奇异值

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F1-G1-O00` | 幂迭代 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G1-O01` | 反幂迭代 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G1-O02` | QR 算法 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G1-O03` | 子空间迭代 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G1-O04` | Lanczos | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G1-O05` | Arnoldi | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G1-O06` | 瑞利商 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G1-O07` | 谱分解 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G1-O08` | 谱范数 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G1-O09` | 条件数 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G1-O10` | 伪逆 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G1-O11` | 截断 SVD | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G1-O12` | 低秩近似 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G1-O13` | 主成分 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G1-O14` | 非负矩阵分解 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G1-O15` | 张量分解 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G2 · 线性方程组

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F1-G2-O00` | 高斯消元 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G2-O01` | LU 求解 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G2-O02` | Cholesky 求解 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G2-O03` | QR 求解 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G2-O04` | 共轭梯度 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G2-O05` | 预处理 CG | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G2-O06` | GMRES | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G2-O07` | 双共轭梯度 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G2-O08` | Jacobi 迭代 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G2-O09` | Gauss-Seidel | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G2-O10` | SOR 松弛 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G2-O11` | 直接法误差 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G2-O12` | 迭代法收敛 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G2-O13` | 稀疏求解 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G2-O14` | 最小二乘解 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G2-O15` | 约束求解 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G3 · 矩阵函数

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F1-G3-O00` | 矩阵指数 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G3-O01` | 矩阵对数 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G3-O02` | 矩阵平方根 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G3-O03` | 矩阵幂 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G3-O04` | 矩阵符号 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G3-O05` | 矩阵余弦 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G3-O06` | 矩阵正弦 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G3-O07` | 矩阵多项式 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G3-O08` | 矩阵分式 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G3-O09` | Sherman-Morrison 引理 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G3-O10` | 矩阵秩1更新 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G3-O11` | Schur 分解 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G3-O12` | 极分解 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G3-O13` | 矩阵范数 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G3-O14` | 矩阵条件 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G3-O15` | 矩阵谱 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G4 · 张量运算

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F1-G4-O00` | 张量收缩 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G4-O01` | Einstein 求和 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G4-O02` | 张量转置 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G4-O03` | 张量重塑 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G4-O04` | 张量切片 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G4-O05` | 广播 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G4-O06` | 外积展开 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G4-O07` | 模式积 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G4-O08` | Tucker 分解 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G4-O09` | CP 分解 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G4-O10` | 张量秩 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G4-O11` | 张量范数 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G4-O12` | 张量缩并 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G4-O13` | 张量求导 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G4-O14` | 张量索引 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G4-O15` | 张量视图 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G5 · 数值线性代数稳定

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F1-G5-O00` | 求逆稳定 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G5-O01` | 正交化稳定 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G5-O02` | 特征值稳定 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G5-O03` | 病态诊断 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G5-O04` | 条件数估计 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G5-O05` | 迭代改进 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G5-O06` | 向后误差 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G5-O07` | 前向误差 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G5-O08` | 平衡缩放 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G5-O09` | 枢轴选择 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G5-O10` | 秩揭示 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G5-O11` | 零空间 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G5-O12` | 列空间 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G5-O13` | 正交投影 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G5-O14` | 最小范数解 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G5-O15` | 残差精炼 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G6 · 特殊矩阵

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F1-G6-O00` | 对角矩阵 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G6-O01` | 三对角矩阵 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G6-O02` | 带状矩阵 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G6-O03` | Toeplitz | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G6-O04` | 循环矩阵 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G6-O05` | 稀疏矩阵 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G6-O06` | 正定矩阵 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G6-O07` | 正交矩阵 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G6-O08` | 酉矩阵 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G6-O09` | 置换矩阵 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G6-O10` | 投影矩阵 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G6-O11` | 幂等矩阵 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G6-O12` | 对称矩阵 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G6-O13` | 反对称矩阵 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G6-O14` | 随机矩阵 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G6-O15` | 低秩矩阵 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G7 · 几何与变换

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F1-G7-O00` | 旋转矩阵 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G7-O01` | Householder 反射 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G7-O02` | Givens 旋转 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G7-O03` | 四元数 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G7-O04` | 仿射变换 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G7-O05` | 齐次坐标 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G7-O06` | 投影变换 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G7-O07` | 相似变换 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G7-O08` | 正交变换 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G7-O09` | 雅可比行列式 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G7-O10` | 体积元 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G7-O11` | 重心坐标 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G7-O12` | 三角剖分 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G7-O13` | 最近邻变换 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G7-O14` | 保角映射 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F1-G7-O15` | 小波变换 | L2 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

## F2 · 优化（层 L3）

### G0 · 一阶方法

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F2-G0-O00` | 梯度下降 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G0-O01` | SGD | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G0-O02` | 动量法 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G0-O03` | Nesterov 加速 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G0-O04` | AdaGrad | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G0-O05` | RMSProp | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G0-O06` | Adam | L3 | done | optim-forge Adam |
| `L10-F2-G0-O07` | AdamW | L3 | done | optim-forge AdamW |
| `L10-F2-G0-O08` | Adadelta | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G0-O09` | Adafactor | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G0-O10` | LAMB | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G0-O11` | Lion | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G0-O12` | Shampoo | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G0-O13` | 重球法 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G0-O14` | 自适应矩变体 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G0-O15` | 带动量平均 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G1 · 二阶与拟牛顿

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F2-G1-O00` | 牛顿法 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G1-O01` | 高斯牛顿 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G1-O02` | L-BFGS | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G1-O03` | DFP | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G1-O04` | 共轭梯度法 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G1-O05` | 信赖域 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G1-O06` | 拟牛顿更新 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G1-O07` | 海森近似 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G1-O08` | 阻尼牛顿 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G1-O09` | OWL-QN | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G1-O10` | 有限记忆 BFGS | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G1-O11` | 拟牛顿校正 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G1-O12` | 线搜索 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G1-O13` | 强 Wolfe 条件 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G1-O14` | 无 Hessian 牛顿 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G1-O15` | 曲率条件 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G2 · 学习率调度

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F2-G2-O00` | 阶跃衰减 | L3 | done | scheduler-forge stepDecay(阶跃衰减) |
| `L10-F2-G2-O01` | 余弦退火 | L3 | done | scheduler-forge cosineSchedule(余弦退火) |
| `L10-F2-G2-O02` | Warmup | L3 | done | scheduler-forge linearWarmup |
| `L10-F2-G2-O03` | 循环学习率 | L3 | done | scheduler-forge cosineWithRestarts(循环学习率) |
| `L10-F2-G2-O04` | 自适应调度 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G2-O05` | 指数衰减 | L3 | done | scheduler-forge exponentialDecay |
| `L10-F2-G2-O06` | 余弦重启 | L3 | done | scheduler-forge cosineWithRestarts(余弦重启) |
| `L10-F2-G2-O07` | 单周期策略 | L3 | done | scheduler-forge oneCycle(单周期策略) |
| `L10-F2-G2-O08` | 余弦衰减 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G2-O09` | 逆平方根 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G2-O10` | 三角循环 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G2-O11` | 1cycle | L3 | done | scheduler-forge oneCycle |
| `L10-F2-G2-O12` | 分层衰减 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G2-O13` | 线性预热 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G2-O14` | 余弦预热 | L3 | done | scheduler-forge cosineSchedule(余弦预热) |
| `L10-F2-G2-O15` | 衰减到零 | L3 | done | scheduler-forge polynomialDecay(衰减到零) |

### G3 · 约束与正则

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F2-G3-O00` | L1 Lasso | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G3-O01` | L2 Ridge | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G3-O02` | 弹性网 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G3-O03` | 投影梯度 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G3-O04` | 罚函数法 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G3-O05` | 拉格朗日 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G3-O06` | 增广拉格朗日 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G3-O07` | 内点法 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G3-O08` | 近端算子 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G3-O09` | 软阈值 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G3-O10` | 硬阈值 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G3-O11` | 组稀疏 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G3-O12` | 结构化稀疏 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G3-O13` | 熵正则 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G3-O14` | KL 约束 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G3-O15` | 约束投影 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G4 · 随机优化

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F2-G4-O00` | momentum-SGD | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G4-O01` | 小批量 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G4-O02` | SAG | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G4-O03` | SVRG 方差缩减 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G4-O04` | 随机 L-BFGS | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G4-O05` | 异步 SGD | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G4-O06` | 梯度裁剪 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G4-O07` | 梯度累积 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G4-O08` | 随机平均梯度 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G4-O09` | 方差缩减 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G4-O10` | 局部 SGD | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G4-O11` | 联邦平均 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G4-O12` | 动量方差缩减 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G4-O13` | 自适应采样 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G4-O14` | 噪声注入 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G4-O15` | 锐度感知最小化 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G5 · 组合与整数

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F2-G5-O00` | 贪心 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G5-O01` | 分支定界 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G5-O02` | 动态规划 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G5-O03` | 模拟退火 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G5-O04` | 遗传算法 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G5-O05` | 粒子群 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G5-O06` | 蚁群 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G5-O07` | 禁忌搜索 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G5-O08` | 爬山 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G5-O09` | 随机局部搜索 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G5-O10` | 差分进化 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G5-O11` | 交叉熵方法 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G5-O12` | 贝叶斯组合优化 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G5-O13` | 整数规划 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G5-O14` | 约束满足 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G5-O15` | 局部搜索 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G6 · 凸优化

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F2-G6-O00` | 线性规划 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G6-O01` | 二次规划 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G6-O02` | 半定规划 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G6-O03` | 内点法 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G6-O04` | 对偶方法 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G6-O05` | 次梯度法 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G6-O06` | 近端梯度 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G6-O07` | 对偶上升 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G6-O08` | 椭球法 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G6-O09` | 切割平面 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G6-O10` | 加速近端 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G6-O11` | ADMM | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G6-O12` | 坐标上升 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G6-O13` | 原始对偶 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G6-O14` | 锥规划 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G6-O15` | 锥规划内点 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G7 · 非凸与全局

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F2-G7-O00` | 坐标下降 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G7-O01` | 交替最小化 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G7-O02` | EM 算法 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G7-O03` | 贝叶斯优化 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G7-O04` | 演化策略 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G7-O05` | 采集函数 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G7-O06` | 多臂老虎机 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G7-O07` | 元学习优化 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G7-O08` | 模拟退火全局 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G7-O09` | 随机搜索 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G7-O10` | 贝叶斯全局 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G7-O11` | 高斯过程优化 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G7-O12` | 树搜索优化 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G7-O13` | 探索利用平衡 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G7-O14` | CMA-ES | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F2-G7-O15` | 黎曼优化 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

## F3 · 自动微分（层 L3）

### G0 · 前向模式

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F3-G0-O00` | 前向累积 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G0-O01` | Dual 数对 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G0-O02` | 雅可比向量积 JVP | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G0-O03` | 切空间 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G0-O04` | 方向导数 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G0-O05` | 前向计算图 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G0-O06` | 算子重载 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G0-O07` | 轨迹追踪 | L3 | done | autodiff-forge 雅可比 |
| `L10-F3-G0-O08` | 前向高阶 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G0-O09` | 链式法则前向 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G0-O10` | 无感模式 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G0-O11` | 复数步微分 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G0-O12` | 源到源变换 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G0-O13` | 算子记录 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G0-O14` | 线性化 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G0-O15` | 切向量传播 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G1 · 反向模式

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F3-G1-O00` | 反向累积 | L3 | done | autodiff-forge 反向累积 |
| `L10-F3-G1-O01` | 向量雅可比积 VJP | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G1-O02` | 计算图 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G1-O03` | 拓扑排序 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G1-O04` | 反向遍历 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G1-O05` | 隐式梯度 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G1-O06` | 高阶导数 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G1-O07` | 梯度检查点 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G1-O08` | 反向前阶 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G1-O09` | 链式法则反向 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G1-O10` | 自动反向 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G1-O11` | 子表达式缓存 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G1-O12` | 反向线性化 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G1-O13` | 标量反向 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G1-O14` | 向量反向 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G1-O15` | 批反向 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G2 · 高阶微分

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F3-G2-O00` | Hessian | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G2-O01` | Hessian-向量积 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G2-O02` | 雅可比矩阵 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G2-O03` | 高维雅可比 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G2-O04` | 张量导数 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G2-O05` | 三阶导数 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G2-O06` | 无穷阶展开 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G2-O07` | 自动高阶 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G2-O08` | Hessian 对角 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G2-O09` | Hessian 迹 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G2-O10` | 雅可比向量积 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G2-O11` | 向量雅可比积 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G2-O12` | 高阶链式法则 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G2-O13` | 泰勒展开 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G2-O14` | 微分算子 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G2-O15` | 微分代数 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G3 · 计算图

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F3-G3-O00` | 节点 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G3-O01` | 边 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G3-O02` | 算子注册 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G3-O03` | 类型推导 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G3-O04` | 形状推断 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G3-O05` | 常量折叠 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G3-O06` | 公共子表达式 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G3-O07` | 死代码消除 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G3-O08` | 依赖分析 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G3-O09` | 内存规划 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G3-O10` | 算子融合 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G3-O11` | 梯度节点 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G3-O12` | 控制依赖 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G3-O13` | 数据依赖 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G3-O14` | 子图提取 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G3-O15` | 图序列化 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G4 · 算子重载

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F3-G4-O00` | 标量重载 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G4-O01` | 向量重载 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G4-O02` | 矩阵重载 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G4-O03` | 原地操作 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G4-O04` | 控制流处理 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G4-O05` | 循环展开 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G4-O06` | 闭包捕获 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G4-O07` | 异常安全 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G4-O08` | 自定义算子 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G4-O09` | 高阶函数重载 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G4-O10` | 条件分支重载 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G4-O11` | 递归重载 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G4-O12` | 广播重载 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G4-O13` | 索引重载 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G4-O14` | 视图重载 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G4-O15` | in-place 安全 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G5 · 混合模式

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F3-G5-O00` | 前向反向混合 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G5-O01` | JVP-VJP 混合 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G5-O02` | 高维混合 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G5-O03` | 稀疏雅可比 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G5-O04` | 结构化雅可比 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G5-O05` | 无矩阵算子 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G5-O06` | 隐式微分 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G5-O07` | 微分规则库 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G5-O08` | 前向批 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G5-O09` | 反向批 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G5-O10` | 高阶混合 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G5-O11` | 复数微分 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G5-O12` | 随机微分 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G5-O13` | 截断微分 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G5-O14` | 截断高阶 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G5-O15` | 自定义规则 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G6 · 性能

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F3-G6-O00` | 内存优化 | L3 | done | core-verify gradCheck |
| `L10-F3-G6-O01` | 重计算 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G6-O02` | 分片反向 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G6-O03` | 延迟执行 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G6-O04` | 向量化微分 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G6-O05` | 并行微分 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G6-O06` | 稀疏微分 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G6-O07` | 编译微分 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G6-O08` | 算子缓存 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G6-O09` | 常量缓存 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G6-O10` | 计算图压缩 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G6-O11` | 增量反向 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G6-O12` | 多线程求导 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G6-O13` | GPU 卸载 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G6-O14` | 内存池化 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G6-O15` | 零拷贝微分 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G7 · 验证

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F3-G7-O00` | 梯度数值检验 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G7-O01` | 高阶检验 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G7-O02` | 不变量检验 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G7-O03` | 双路微分 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G7-O04` | 对照实现 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G7-O05` | 精度基准 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G7-O06` | 病态梯度 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G7-O07` | 确定性检验 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G7-O08` | 解析梯度比对 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G7-O09` | 自动微分对照 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G7-O10` | 有限差分 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G7-O11` | 中心差分 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G7-O12` | 复步微分 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G7-O13` | 误差界检验 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G7-O14` | 收敛检验 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F3-G7-O15` | 回归检验 | L3 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

## F4 · 注意力与 Transformer（层 L4）

### G0 · 注意力核心

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F4-G0-O00` | 缩放点积注意力 | L4 | done | attn-forge attention |
| `L10-F4-G0-O01` | 因果掩码 | L4 | done | attn-forge causalMask |
| `L10-F4-G0-O02` | 加性注意力 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G0-O03` | 多头注意力 | L4 | done | attention-forge 多头注意力 |
| `L10-F4-G0-O04` | 头部拼接 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G0-O05` | 注意力分数 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G0-O06` | softmax 稳定 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G0-O07` | 权重归一 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G0-O08` | 交叉注意力 | L4 | done | attention-forge 自注意力 |
| `L10-F4-G0-O09` | 自注意力 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G0-O10` | 全局注意力 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G0-O11` | 局部注意力 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G0-O12` | 线性注意力核 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G0-O13` | 注意力 dropout | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G0-O14` | 注意力缩放 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G0-O15` | 注意力偏置 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G1 · 位置编码

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F4-G1-O00` | RoPE | L4 | done | rope-forge RoPE |
| `L10-F4-G1-O01` | 绝对位置 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G1-O02` | 相对位置 | L4 | done | position-encoding-forge 相对位置 |
| `L10-F4-G1-O03` | ALiBi | L4 | done | position-encoding-forge ALiBi |
| `L10-F4-G1-O04` | 正弦位置 | L4 | done | transformer-forge 正弦位置编码 |
| `L10-F4-G1-O05` | 可学习位置 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G1-O06` | 旋转相对 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G1-O07` | 位置插值 | L4 | done | position-encoding-forge 位置插值 |
| `L10-F4-G1-O08` | 旋转绝对 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G1-O09` | 轴向位置 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G1-O10` | 傅里叶特征 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G1-O11` | 相对位置偏置 | L4 | done | position-encoding-forge 相对位置偏置 |
| `L10-F4-G1-O12` | 旋转位置扩展 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G1-O13` | 学习相对 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G1-O14` | 连续位置 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G1-O15` | 位置解耦 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G2 · 归一化

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F4-G2-O00` | RMSNorm | L4 | done | loss-act-norm-forge RMSNorm |
| `L10-F4-G2-O01` | LayerNorm | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G2-O02` | BatchNorm | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G2-O03` | GroupNorm | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G2-O04` | InstanceNorm | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G2-O05` | 权重归一 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G2-O06` | 在线归一 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G2-O07` | 流式归一 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G2-O08` | 自适应归一 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G2-O09` | 幂归一 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G2-O10` | 谱归一 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G2-O11` | 局部响应归一 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G2-O12` | 层归一预 | L4 | done | transformer-forge 层归一预 |
| `L10-F4-G2-O13` | 后归一 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G2-O14` | 可学习归一 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G2-O15` | 归一稳定 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G3 · 激活与门控

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F4-G3-O00` | SwiGLU | L4 | done | loss-act-norm-forge SwiGLU |
| `L10-F4-G3-O01` | GELU | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G3-O02` | ReLU | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G3-O03` | GELU 精确 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G3-O04` | SiLU | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G3-O05` | GLU | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G3-O06` | 门控线性单元 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G3-O07` | 激活稳定 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G3-O08` | GeGLU | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G3-O09` | ReGLU | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G3-O10` | 激活函数族 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G3-O11` | 平滑激活 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G3-O12` | 稀疏激活 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G3-O13` | 激活裁剪 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G3-O14` | 双曲激活 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G3-O15` | 概率激活 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G4 · 高效注意力

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F4-G4-O00` | 稀疏注意力 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G4-O01` | 线性注意力 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G4-O02` | 局部窗口 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G4-O03` | 低秩注意力 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G4-O04` | 分块注意力 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G4-O05` | 多查询 MQA | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G4-O06` | 分组查询 GQA | L4 | done | attention-forge 分组查询 GQA |
| `L10-F4-G4-O07` | FlashAttention 风格 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G4-O08` | 滑动窗口 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G4-O09` | 扩张注意力 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G4-O10` | 稀疏全局 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G4-O11` | 低秩分解 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G4-O12` | 注意力近似 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G4-O13` | 缓存注意力 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G4-O14` | 量化注意力 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G4-O15` | 分块因果 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G5 · KV 缓存与推理

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F4-G5-O00` | KV 缓存 | L4 | done | kv-cache-forge KV缓存 |
| `L10-F4-G5-O01` | 分页注意力 | L4 | done | kv-cache-forge 分页注意力 |
| `L10-F4-G5-O02` | 增量解码 | L4 | done | kv-cache-forge 增量解码 |
| `L10-F4-G5-O03` | 推测解码 | L4 | done | speculative-forge 推测解码 |
| `L10-F4-G5-O04` | 并行解码 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G5-O05` | 缓存复用 | L4 | done | kv-cache-forge 缓存复用 |
| `L10-F4-G5-O06` | 缓存逐出 | L4 | done | kv-cache-forge 缓存逐出 |
| `L10-F4-G5-O07` | 流式生成 | L4 | done | generation-forge 流式生成 |
| `L10-F4-G5-O08` | 缓存预分配 | L4 | done | kv-cache-forge 缓存预分配 |
| `L10-F4-G5-O09` | 缓存压缩 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G5-O10` | 缓存量化 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G5-O11` | 并行采样 | L4 | done | sampling-forge 并行采样 |
| `L10-F4-G5-O12` | 树状注意力 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G5-O13` | 连续批处理 | L4 | done | generation-forge 连续批处理 |
| `L10-F4-G5-O14` | 前缀缓存 | L4 | done | generation-forge 前缀缓存 |
| `L10-F4-G5-O15` | 注意力下沉 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G6 · 量化与压缩

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F4-G6-O00` | int8 量化 | L4 | done | quant-forge int8量化(对称/非对称) |
| `L10-F4-G6-O01` | int4 量化 | L4 | done | quant-forge int4量化(位压缩excess码) |
| `L10-F4-G6-O02` | k-quants | L4 | done | quant-forge k-quants超块/子块二级scale |
| `L10-F4-G6-O03` | 权重量化 | L4 | done | quant-forge 权重量化(逐通道) |
| `L10-F4-G6-O04` | 激活量化 | L4 | done | quant-forge 激活量化(通用张量校准) |
| `L10-F4-G6-O05` | 混合精度 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G6-O06` | 平滑量化 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G6-O07` | 量化感知训练 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G6-O08` | 仅权重量化 | L4 | done | quant-forge 仅权重量化 |
| `L10-F4-G6-O09` | 组量化 | L4 | done | quant-forge 组量化(超块共享scale) |
| `L10-F4-G6-O10` | 逐通道量化 | L4 | done | quant-forge 逐通道量化 |
| `L10-F4-G6-O11` | 逐张量量化 | L4 | done | quant-forge 逐张量量化 |
| `L10-F4-G6-O12` | 稀疏量化 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G6-O13` | 旋转量化 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G6-O14` | 量化反量化 | L4 | done | quant-forge 量化反量化(往返/SNR) |
| `L10-F4-G6-O15` | 校准量化 | L4 | done | quant-forge 校准量化(absmax/affine) |

### G7 · 模型组件

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F4-G7-O00` | 嵌入层 | L4 | done | transformer-forge 嵌入层 |
| `L10-F4-G7-O01` | 词表投影 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G7-O02` | 残差连接 | L4 | done | transformer-forge 残差连接 |
| `L10-F4-G7-O03` | 前馈网络 | L4 | done | transformer-forge 前馈网络 |
| `L10-F4-G7-O04` | 预归一化 | L4 | done | transformer-forge 预归一化 |
| `L10-F4-G7-O05` | 子层缩放 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G7-O06` | 初始化 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G7-O07` | 参数绑定 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G7-O08` | 位置前馈 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G7-O09` | 专家混合 MoE | L4 | done | moe-forge 专家混合 MoE |
| `L10-F4-G7-O10` | 路由 | L4 | done | moe-forge 路由 |
| `L10-F4-G7-O11` | 共享专家 | L4 | done | moe-forge 共享专家 |
| `L10-F4-G7-O12` | 缩放点积预 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G7-O13` | 层缩放 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G7-O14` | 深度缩放 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F4-G7-O15` | 参数初始化族 | L4 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

## F5 · 概率与生成（层 L5）

### G0 · 分布

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F5-G0-O00` | 正态分布 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G0-O01` | 伯努利分布 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G0-O02` | 类别分布 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G0-O03` | 多项式分布 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G0-O04` | 指数族 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G0-O05` | Gamma 分布 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G0-O06` | Beta 分布 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G0-O07` | 狄利克雷分布 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G0-O08` | 泊松分布 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G0-O09` | 均匀分布 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G0-O10` | 拉普拉斯分布 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G0-O11` | 学生 t 分布 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G0-O12` | Wishart | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G0-O13` | Logistic | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G0-O14` | 负二项分布 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G0-O15` | 混合高斯 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G1 · 采样

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F5-G1-O00` | 逆 CDF 采样 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G1-O01` | 拒绝采样 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G1-O02` | 重要性采样 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G1-O03` | MCMC | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G1-O04` | 吉布斯采样 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G1-O05` | 切片采样 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G1-O06` | Hamiltonian Monte Carlo | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G1-O07` | 蛙跳积分 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G1-O08` | 独立采样 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G1-O09` | 分层采样 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G1-O10` | 拉丁超立方 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G1-O11` | 马尔可夫链采样 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G1-O12` | 自适应拒绝 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G1-O13` | Metropolis | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G1-O14` | 并行采样 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G1-O15` | 准蒙特卡洛 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G2 · 变分推断

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F5-G2-O00` | 均值场变分 | L5 | done | vi-forge 均值场变分(meanFieldKL) |
| `L10-F5-G2-O01` | 黑盒变分推断 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G2-O02` | 重参数化技巧 | L5 | done | vi-forge 重参数化技巧(reparamSample) |
| `L10-F5-G2-O03` | 证据下界 ELBO | L5 | done | vi-forge 证据下界ELBO(elbo) |
| `L10-F5-G2-O04` | 变分自编码器 | L5 | done | vi-forge 变分自编码器(vae) |
| `L10-F5-G2-O05` | 正态化流 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G2-O06` | 提议分布 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G2-O07` | 变分族 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G2-O08` | 摊销变分推断 | L5 | done | vi-forge 摊销变分推断(amortizedEncode) |
| `L10-F5-G2-O09` | 结构化变分 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G2-O10` | 递归变分 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G2-O11` | 变分 dropout | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G2-O12` | 层次变分 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G2-O13` | 离散变分 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G2-O14` | ELBO 优化 | L5 | done | vi-forge ELBO优化(vae训练) |
| `L10-F5-G2-O15` | 变分期望最大化 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G3 · 贝叶斯

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F5-G3-O00` | 贝叶斯更新 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G3-O01` | 共轭先验 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G3-O02` | 马尔可夫链贝叶斯 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G3-O03` | 贝叶斯估计 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G3-O04` | 可信区间 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G3-O05` | 贝叶斯因子 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G3-O06` | 层次模型 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G3-O07` | 贝叶斯神经网络 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G3-O08` | 贝叶斯模型平均 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G3-O09` | 贝叶斯优化 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G3-O10` | 贝叶斯推断 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G3-O11` | 贝叶斯线性回归 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G3-O12` | 贝叶斯非参数 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G3-O13` | 贝叶斯交叉验证 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G3-O14` | 贝叶斯模型选择 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G3-O15` | 贝叶斯决策 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G4 · 随机过程

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F5-G4-O00` | 高斯过程 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G4-O01` | 连续马尔可夫链 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G4-O02` | 泊松过程 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G4-O03` | 维纳过程 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G4-O04` | 布朗运动 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G4-O05` | 平稳过程 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G4-O06` | 自回归过程 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G4-O07` | 隐马尔可夫模型 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G4-O08` | 马尔可夫随机场 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G4-O09` | 高斯随机场 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G4-O10` | 扩散过程 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G4-O11` | 更新过程 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G4-O12` | 分支过程 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G4-O13` | Lévy 过程 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G4-O14` | 马尔可夫毯 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G4-O15` | 条件随机场 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G5 · MCMC 进阶

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F5-G5-O00` | Metropolis | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G5-O01` | Metropolis-Hastings | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G5-O02` | 吉布斯抽样 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G5-O03` | NUTS | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G5-O04` | 随机游走 Metropolis | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G5-O05` | 独立采样 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G5-O06` | 退火采样 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G5-O07` | 并行回火 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G5-O08` | 可逆跳跃 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G5-O09` | 延迟拒绝 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G5-O10` | 辅助变量 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G5-O11` | 哈密顿更新 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G5-O12` | 黎曼流形 HMC | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G5-O13` | 自适应步长 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G5-O14` | 混合蒙特卡洛 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G5-O15` | 窗宽调优 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G6 · 估计与检验

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F5-G6-O00` | 最大似然估计 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G6-O01` | 最大后验估计 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G6-O02` | 矩估计 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G6-O03` | 贝叶斯估计 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G6-O04` | KS 检验 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G6-O05` | 卡方检验 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G6-O06` | Bootstrap | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G6-O07` | 交叉验证似然 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G6-O08` | 矩方法 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G6-O09` | 拟似然 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G6-O10` | 稳健估计 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G6-O11` | 经验似然 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G6-O12` | 假设检验 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G6-O13` | 置信区间 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G6-O14` | 似然比检验 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G6-O15` | 拟合优度 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G7 · 生成模型

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F5-G7-O00` | 自回归生成 | L5 | done | sampling-forge 自回归生成 |
| `L10-F5-G7-O01` | 扩散模型 | L5 | done | diffusion-forge DDPM/DDIM(扩散模型) |
| `L10-F5-G7-O02` | 流模型 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G7-O03` | 能量模型 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G7-O04` | 生成对抗 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G7-O05` | 变分生成 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G7-O06` | 得分匹配 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G7-O07` | 朗之万采样 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G7-O08` | 归一化流生成 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G7-O09` | 自回归流 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G7-O10` | 隐扩散 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G7-O11` | 一致性模型 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G7-O12` | 扩散蒸馏 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G7-O13` | 变分扩散 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G7-O14` | 能量生成 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F5-G7-O15` | 生成式预训练 | L5 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

## F6 · 强化学习（层 L6）

### G0 · 基础

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F6-G0-O00` | 马尔可夫决策过程 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G0-O01` | 值函数 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G0-O02` | 动作值函数 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G0-O03` | 策略 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G0-O04` | 奖励函数 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G0-O05` | 折扣因子 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G0-O06` | 状态转移 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G0-O07` | 贝尔曼方程 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G0-O08` | 状态空间 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G0-O09` | 动作空间 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G0-O10` | 转移概率 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G0-O11` | 回报 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G0-O12` | 折扣回报 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G0-O13` | 最优策略 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G0-O14` | 最优值函数 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G0-O15` | 最优动作值 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G1 · 值方法

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F6-G1-O00` | 动态规划 | L6 | done | td-forge 马尔可夫决策过程(MDP表示) |
| `L10-F6-G1-O01` | 策略迭代 | L6 | done | td-forge 值函数(valueIteration) |
| `L10-F6-G1-O02` | 值迭代 | L6 | done | td-forge 动作值函数(qLearning) |
| `L10-F6-G1-O03` | 蒙特卡洛评估 | L6 | done | td-forge 策略(policyIteration) |
| `L10-F6-G1-O04` | 时序差分 | L6 | done | td-forge 奖励函数(MDP) |
| `L10-F6-G1-O05` | SARSA | L6 | done | td-forge 折扣因子(γ) |
| `L10-F6-G1-O06` | Q 学习 | L6 | done | td-forge 状态转移(MDP) |
| `L10-F6-G1-O07` | 双重 Q 学习 | L6 | done | td-forge 贝尔曼方程(valueIteration) |
| `L10-F6-G1-O08` | 蒙特卡洛控制 | L6 | done | td-forge 状态空间(MDP) |
| `L10-F6-G1-O09` | TD(λ) | L6 | done | td-forge 动作空间(MDP) |
| `L10-F6-G1-O10` | 资格迹 | L6 | done | td-forge 转移概率(MDP) |
| `L10-F6-G1-O11` | 期望 SARSA | L6 | done | td-forge 回报(generateEpisodes) |
| `L10-F6-G1-O12` | 树备份 | L6 | done | td-forge 折扣回报 |
| `L10-F6-G1-O13` | 实时动态规划 | L6 | done | td-forge 最优策略(policyIteration) |
| `L10-F6-G1-O14` | 广义优势之值 | L6 | done | td-forge 最优值函数(valueIteration) |
| `L10-F6-G1-O15` | 分布式值迭代 | L6 | done | td-forge 最优动作值(qLearning) |

### G2 · 策略梯度

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F6-G2-O00` | REINFORCE | L6 | done | policygrad-forge REINFORCE(policyGradientREINFORCE) |
| `L10-F6-G2-O01` | 优势函数 | L6 | done | policygrad-forge 优势函数(advantageFunction) |
| `L10-F6-G2-O02` | 基线减除 | L6 | done | policygrad-forge 基线减除(REINFORCE baseline) |
| `L10-F6-G2-O03` | TRPO | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G2-O04` | PPO | L6 | done | policygrad-forge PPO(ppoClip/ppoUpdate) |
| `L10-F6-G2-O05` | 近似策略梯度 | L6 | done | policygrad-forge 近似策略梯度 |
| `L10-F6-G2-O06` | 自然梯度 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G2-O07` | 确定性策略梯度 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G2-O08` | 演员评论家 | L6 | done | policygrad-forge 演员评论家(actorCritic/a2c) |
| `L10-F6-G2-O09` | A2C | L6 | done | policygrad-forge A2C(a2c) |
| `L10-F6-G2-O10` | 信赖域评论 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G2-O11` | 广义优势估计 | L6 | done | policygrad-forge 广义优势估计(gae) |
| `L10-F6-G2-O12` | 离策略梯度 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G2-O13` | 重要性采样梯度 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G2-O14` | 策略梯度定理 | L6 | done | policygrad-forge 策略梯度定理(INV-PG-001) |
| `L10-F6-G2-O15` | 软策略梯度 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G3 · 深度 RL

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F6-G3-O00` | DQN | L6 | done | dqn-forge DQN更新(Mnih 2015) |
| `L10-F6-G3-O01` | 经验回放 | L6 | done | dqn-forge 经验回放(均匀采样环形缓冲) |
| `L10-F6-G3-O02` | 目标网络 | L6 | done | dqn-forge 目标网络(Polyak软更新) |
| `L10-F6-G3-O03` | 双重 DQN | L6 | done | dqn-forge 双重DQN(van Hasselt 2015) |
| `L10-F6-G3-O04` | 决斗网络 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G3-O05` | 优先经验回放 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G3-O06` | 噪声网络 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G3-O07` | 分布式 RL | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G3-O08` | C51 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G3-O09` | QR-DQN | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G3-O10` | 逆强化学习 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G3-O11` | 模仿学习 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G3-O12` | 行为克隆 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G3-O13` | 离策略 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G3-O14` | 在线策略 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G3-O15` | 分布式 Q | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G4 · 基于模型

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F6-G4-O00` | 动力学模型 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G4-O01` | 模型预测控制 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G4-O02` | 蒙特卡洛树搜索 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G4-O03` | AlphaZero | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G4-O04` | MuZero | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G4-O05` | 世界模型 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G4-O06` | 规划 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G4-O07` | 想象滚动 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G4-O08` | 模型基 RL | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G4-O09` | 不确定性规划 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G4-O10` | 贝叶斯规划 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G4-O11` | 滚动地平线 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G4-O12` | 树搜索策略 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G4-O13` | 价值搜索 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G4-O14` | 模型集成 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G4-O15` | 模型损失 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G5 · 探索

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F6-G5-O00` | ε-贪婪 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G5-O01` | 熵正则化 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G5-O02` | 上置信界 UCB | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G5-O03` | Thompson 采样 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G5-O04` | 内在奖励 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G5-O05` | 好奇驱动 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G5-O06` | 计数探索 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G5-O07` | 噪声探索 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G5-O08` | 玻尔兹曼探索 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G5-O09` | 随机网络蒸馏 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G5-O10` | 伪计数 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G5-O11` | 方差探索 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G5-O12` | 信任区域探索 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G5-O13` | 最大熵探索 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G5-O14` | 势能探索 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G5-O15` | 信息增益 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G6 · 多智能体

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F6-G6-O00` | 马尔可夫博弈 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G6-O01` | 纳什均衡 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G6-O02` | 独立学习 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G6-O03` | 集中训练分散执行 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G6-O04` | 信用分配 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G6-O05` | 通信协议 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G6-O06` | 合作博弈 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G6-O07` | 竞争博弈 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G6-O08` | 平均场博弈 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G6-O09` | 图博弈 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G6-O10` | 多智能体策略梯度 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G6-O11` | 多智能体 Q | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G6-O12` | 协调博弈 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G6-O13` | 对抗博弈 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G6-O14` | 谣言博弈 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G6-O15` | 社会困境 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G7 · 应用

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F6-G7-O00` | 推荐系统 RL | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G7-O01` | 对话策略 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G7-O02` | 机器人控制 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G7-O03` | 组合优化 RL | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G7-O04` | 元 RL | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G7-O05` | 分层 RL | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G7-O06` | 离线 RL | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G7-O07` | 安全 RL | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G7-O08` | 自动驾驶 RL | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G7-O09` | 游戏 RL | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G7-O10` | 金融 RL | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G7-O11` | 健康 RL | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G7-O12` | RL 推荐 | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G7-O13` | 持续学习 RL | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G7-O14` | 多任务 RL | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F6-G7-O15` | 鲁棒 RL | L6 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

## F7 · 检索/图/系统（层 L7）

### G0 · 向量检索

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F7-G0-O00` | 余弦相似 | L7 | done | retrieval-forge 余弦相似(cosineSim) |
| `L10-F7-G0-O01` | 内积检索 | L7 | done | retrieval-forge 内积检索(dotSearch/metricScore) |
| `L10-F7-G0-O02` | L2 检索 | L7 | done | retrieval-forge L2检索(l2Search) |
| `L10-F7-G0-O03` | 近似近邻 ANN | L7 | done | retrieval-forge 近似近邻ANN调度(buildANNIndex/annSearch) |
| `L10-F7-G0-O04` | 穷举检索 | L7 | done | retrieval-forge 穷举检索(bruteForceSearch) |
| `L10-F7-G0-O05` | 量化检索 | L7 | done | retrieval-forge 量化检索(buildScalarQuantizer/quantizedSearch) |
| `L10-F7-G0-O06` | 分层导航 HNSW | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G0-O07` | 倒排文件 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G0-O08` | 最大内积检索 | L7 | done | retrieval-forge 最大内积检索(mipsSearch) |
| `L10-F7-G0-O09` | 检索重排 | L7 | done | retrieval-forge 检索重排(rerank) |
| `L10-F7-G0-O10` | 检索融合 | L7 | done | retrieval-forge 检索融合RRF(rrfFuse) |
| `L10-F7-G0-O11` | 候选生成 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G0-O12` | 相似度阈值 | L7 | done | retrieval-forge 相似度阈值(thresholdFilter) |
| `L10-F7-G0-O13` | 批量检索 | L7 | done | retrieval-forge 批量检索(batchSearch) |
| `L10-F7-G0-O14` | 近邻图构建 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G0-O15` | 近邻图搜索 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G1 · 索引结构

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F7-G1-O00` | KD 树 | L7 | done | ann-forge KD树(buildKD/kdKnN) |
| `L10-F7-G1-O01` | 球树 | L7 | done | ann-forge 球树(buildBall/ballKnN) |
| `L10-F7-G1-O02` | LSH | L7 | done | ann-forge LSH超平面(buildLSH/lshQuery) |
| `L10-F7-G1-O03` | Annoy 树 | L7 | done | ann-forge Annoy(buildAnnoy/annoyQuery) |
| `L10-F7-G1-O04` | 倒排索引 | L7 | done | ann-forge 倒排索引(buildInvertedIndex) |
| `L10-F7-G1-O05` | 乘积量化 PQ | L7 | done | ann-forge 乘积量化PQ(trainPQ/pqADCDist) |
| `L10-F7-G1-O06` | IVFPQ | L7 | done | ann-forge IVFPQ(buildIVFPQ/ivfpqQuery) |
| `L10-F7-G1-O07` | 局部敏感哈希 | L7 | done | ann-forge LSH p稳定(buildLSHpstable/lshPQuery) |
| `L10-F7-G1-O08` | ScaNN | L7 | done | ann-forge ScaNN(buildScaNN/scannQuery) |
| `L10-F7-G1-O09` | 旋转不变 LSH | L7 | done | ann-forge 旋转不变LSH(buildRotationLSH/rotationLSHQuery) |
| `L10-F7-G1-O10` | 多索引 HNSW | L7 | done | ann-forge 多索引HNSW(buildHNSW) |
| `L10-F7-G1-O11` | 图索引 | L7 | done | ann-forge 图索引NSW(buildNSW/nswSearch) |
| `L10-F7-G1-O12` | 树索引 | L7 | done | ann-forge 树索引(TreeIndex) |
| `L10-F7-G1-O13` | 哈希索引 | L7 | done | ann-forge 哈希索引(HashIndex) |
| `L10-F7-G1-O14` | 粗量化 | L7 | done | ann-forge 粗量化(buildCoarseQuantizer) |
| `L10-F7-G1-O15` | 细量化 | L7 | done | ann-forge 细量化(buildFineQuantizer) |

### G2 · 图算法

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F7-G2-O00` | BFS | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G2-O01` | DFS | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G2-O02` | Dijkstra | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G2-O03` | A* | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G2-O04` | 最短路径 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G2-O05` | PageRank | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G2-O06` | 连通分量 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G2-O07` | 最小生成树 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G2-O08` | 拓扑排序 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G2-O09` | 强连通分量 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G2-O10` | 二分图匹配 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G2-O11` | 最大流 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G2-O12` | 社区发现 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G2-O13` | 三角计数 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G2-O14` | 图着色 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G2-O15` | 中心性度量 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G3 · 图神经网络

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F7-G3-O00` | 图卷积 GCN | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G3-O01` | GraphSAGE | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G3-O02` | 注意力图网络 GAT | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G3-O03` | 消息传递 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G3-O04` | 节点分类 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G3-O05` | 图池化 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G3-O06` | 边预测 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G3-O07` | 图采样 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G3-O08` | 图回归 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G3-O09` | 图自编码器 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G3-O10` | 图注意力池化 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G3-O11` | 图同构网络 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G3-O12` | 边分类 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G3-O13` | 子图匹配 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G3-O14` | 图生成 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G3-O15` | 图预训练 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G4 · 稀疏与压缩

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F7-G4-O00` | 稀疏矩阵乘 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G4-O01` | 稀疏格式转换 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G4-O02` | 压缩稀疏行 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G4-O03` | 压缩存储 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G4-O04` | 低秩压缩 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G4-O05` | 剪枝 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G4-O06` | 稀疏注意力 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G4-O07` | 稀疏梯度 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G4-O08` | 稀疏向量 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G4-O09` | 稀疏卷积 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G4-O10` | 量化压缩 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G4-O11` | 哈夫曼编码 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G4-O12` | 游程编码 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G4-O13` | 字典压缩 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G4-O14` | 增量压缩 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G4-O15` | 列式压缩 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G5 · 序列化与存储

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F7-G5-O00` | JSON 序列化 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G5-O01` | 二进制序列化 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G5-O02` | 列式存储 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G5-O03` | 内存映射 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G5-O04` | 增量写入 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G5-O05` | 校验持久化 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G5-O06` | 版本化存储 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G5-O07` | 快照 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G5-O08` | 协议缓冲区 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G5-O09` | 分区存储 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G5-O10` | 对象存储 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G5-O11` | 本地缓存 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G5-O12` | 远端缓存 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G5-O13` | 事务写入 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G5-O14` | 灾难恢复 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G5-O15` | 数据校验和 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G6 · 系统与部署

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F7-G6-O00` | 推理批处理 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G6-O01` | 算子融合 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G6-O02` | 内核调度 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G6-O03` | 内存池 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G6-O04` | 零拷贝 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G6-O05` | 流式管线 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G6-O06` | 负载均衡 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G6-O07` | 容错 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G6-O08` | 自动扩缩容 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G6-O09` | 请求队列 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G6-O10` | 预热缓存 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G6-O11` | 降级策略 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G6-O12` | 限流 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G6-O13` | 监控埋点 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G6-O14` | 灰度发布 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G6-O15` | 回滚 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

### G7 · 评测与监控

| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |
| --- | --- | --- | --- | --- |
| `L10-F7-G7-O00` | 精度评测 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G7-O01` | 延迟评测 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G7-O02` | 吞吐量评测 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G7-O03` | 召回率评测 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G7-O04` | 一致性评测 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G7-O05` | 资源监控 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G7-O06` | 漂移检测 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G7-O07` | 回归监控 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G7-O08` | F1 评测 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G7-O09` | AUC 评测 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G7-O10` | 困惑度评测 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G7-O11` | BLEU 评测 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G7-O12` | 人工评测 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G7-O13` | 对抗评测 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G7-O14` | 公平性评测 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |
| `L10-F7-G7-O15` | 可解释评测 | L7 | planned | 规划：W3-W7 后续实现（含 @reuse 顶级参照） |

---

*生成器：tools/gen-l10.js · schema forge-l10/1.0 · 作者：晨星 · MIT*