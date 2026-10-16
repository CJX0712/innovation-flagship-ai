/**
 * tools/gen-l10.js — L10 微指令层（1024 项）分类法生成器
 *
 * 十层架构：L1(2)+L2(4)+L3(8)+L4(16)+L5(32)+L6(64)+L7(128)+L8(256)+L9(512)+L10(1024)=2046
 * L10 = 8 族(F0-F7) × 8 组(G0-G7) × 16 微指令(O0-O15) = 1024
 *
 * 设计原则（零重写 · 集大成）：REUSE 表把每条已覆盖的微指令映射到其实现模块
 *（@reuse 世界顶级参照）；status 由「被引用的模块文件是否真实存在于 src/」推导：
 * done=对应模块已实现；planned=W3-W7 后续实现（模块尚未落地）。
 *
 * 用法：node tools/gen-l10.js  →  生成 docs/L10-taxonomy.md + docs/L10-index.json
 * 作者：晨星 · MIT
 */
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const OUT_MD = path.join(ROOT, 'docs', 'L10-taxonomy.md');
const OUT_JSON = path.join(ROOT, 'docs', 'L10-index.json');
const SRC_DIR = path.join(ROOT, 'src');

/**
 * 由 REUSE 引用串解析实现模块名（首 token），判断对应 src/<module>.js 是否真实存在。
 * 用于推导 L10 微指令 status：被引用模块已实现 => done，否则 planned。
 * 这样 status 永远与 src/ 中真实落地的模块保持一致，避免「硬编码 done 但模块未实现」的漂移。
 */
function moduleImplemented(ref) {
  const mod = String(ref || '').trim().split(/\s+/)[0];
  if (!mod) return false;
  return fs.existsSync(path.join(SRC_DIR, mod + '.js'));
}

const REUSE = {
  // F0 数值基础
  'L10-F0-G1-O00': 'core-numeric Kahan/Neumaier',
  'L10-F0-G1-O01': 'core-numeric Neumaier',
  'L10-F0-G1-O02': 'stable-forge logsumexp',
  'L10-F0-G1-O03': 'stable-forge 稳定 softmax',
  'L10-F0-G0-O00': 'core-numeric sfc32',
  'L10-F0-G0-O01': 'core-numeric splitmix32',
  'L10-F0-G0-O06': 'core-numeric normInv(Acklam JSS2003)',
  'L10-F0-G2-O00': 'core-numeric sha256(FIPS180-4)',
  'L10-F0-G2-O04': 'core-verify 双端一致 hash',
  'L10-F0-G6-O00': 'core-verify gradCheck',
  'L10-F0-G6-O01': 'core-verify 不变量断言',
  'L10-F0-G7-O00': 'core-verify forge-selfcheck/1.0',
  'L10-F0-G7-O01': 'core-verify formatTableSafe',
  // F1 线性代数
  'L10-F1-G0-O00': 'core-numeric gemm',
  'L10-F1-G0-O05': 'linalg-forge lu(Doolittle 部分主元)',
  'L10-F1-G0-O06': 'linalg-forge chol',
  'L10-F1-G0-O07': 'linalg-forge qr(Householder)',
  'L10-F1-G0-O08': 'linalg-forge svd(one-sided Jacobi)',
  'L10-F1-G0-O09': 'linalg-forge eigSym(Jacobi)',
  'L10-F1-G0-O10': 'linalg-forge solveLU',
  'L10-F1-G0-O11': 'linalg-forge lstsq(QR/正规方程)',
  // F2 优化
  'L10-F2-G0-O06': 'optim-forge Adam',
  'L10-F2-G0-O07': 'optim-forge AdamW',
  // F2 学习率调度（scheduler-forge）
  'L10-F2-G2-O00': 'scheduler-forge stepDecay(阶跃衰减)',
  'L10-F2-G2-O01': 'scheduler-forge cosineSchedule(余弦退火)',
  'L10-F2-G2-O02': 'scheduler-forge linearWarmup',
  'L10-F2-G2-O03': 'scheduler-forge cosineWithRestarts(循环学习率)',
  'L10-F2-G2-O05': 'scheduler-forge exponentialDecay',
  'L10-F2-G2-O06': 'scheduler-forge cosineWithRestarts(余弦重启)',
  'L10-F2-G2-O07': 'scheduler-forge oneCycle(单周期策略)',
  'L10-F2-G2-O11': 'scheduler-forge oneCycle',
  'L10-F2-G2-O14': 'scheduler-forge cosineSchedule(余弦预热)',
  'L10-F2-G2-O15': 'scheduler-forge polynomialDecay(衰减到零)',
  // F3 自动微分
  'L10-F3-G0-O07': 'autodiff-forge 雅可比',
  'L10-F3-G1-O00': 'autodiff-forge 反向累积',
  'L10-F3-G6-O00': 'core-verify gradCheck',
  // F4 注意力与 Transformer
  'L10-F4-G0-O00': 'attn-forge attention',
  'L10-F4-G0-O01': 'attn-forge causalMask',
  'L10-F4-G0-O03': 'attention-forge 多头注意力',
  'L10-F4-G0-O08': 'attention-forge 自注意力',
  'L10-F4-G1-O00': 'rope-forge RoPE',
  'L10-F4-G1-O04': 'transformer-forge 正弦位置编码',
  'L10-F4-G2-O00': 'loss-act-norm-forge RMSNorm',
  'L10-F4-G2-O12': 'transformer-forge 层归一预',
  'L10-F4-G3-O00': 'loss-act-norm-forge SwiGLU',
  'L10-F4-G4-O06': 'attention-forge 分组查询 GQA',
  'L10-F4-G7-O00': 'transformer-forge 嵌入层',
  'L10-F4-G7-O02': 'transformer-forge 残差连接',
  'L10-F4-G7-O03': 'transformer-forge 前馈网络',
  'L10-F4-G7-O04': 'transformer-forge 预归一化',
  'L10-F4-G5-O11': 'sampling-forge 并行采样',
  'L10-F4-G5-O00': 'kv-cache-forge KV缓存',
  'L10-F4-G5-O01': 'kv-cache-forge 分页注意力',
  'L10-F4-G5-O02': 'kv-cache-forge 增量解码',
  'L10-F4-G5-O05': 'kv-cache-forge 缓存复用',
  'L10-F4-G5-O06': 'kv-cache-forge 缓存逐出',
  'L10-F4-G5-O08': 'kv-cache-forge 缓存预分配',
  'L10-F4-G1-O02': 'position-encoding-forge 相对位置',
  'L10-F4-G1-O03': 'position-encoding-forge ALiBi',
  'L10-F4-G1-O07': 'position-encoding-forge 位置插值',
  'L10-F4-G1-O11': 'position-encoding-forge 相对位置偏置',
  'L10-F4-G7-O09': 'moe-forge 专家混合 MoE',
  'L10-F4-G7-O10': 'moe-forge 路由',
  'L10-F4-G7-O11': 'moe-forge 共享专家',
  'L10-F4-G5-O03': 'speculative-forge 推测解码',
  'L10-F4-G5-O07': 'generation-forge 流式生成',
  'L10-F4-G5-O13': 'generation-forge 连续批处理',
  'L10-F4-G5-O14': 'generation-forge 前缀缓存',
  'L10-F4-G6-O00': 'quant-forge int8量化(对称/非对称)',
  'L10-F4-G6-O01': 'quant-forge int4量化(位压缩excess码)',
  'L10-F4-G6-O02': 'quant-forge k-quants超块/子块二级scale',
  'L10-F4-G6-O03': 'quant-forge 权重量化(逐通道)',
  'L10-F4-G6-O04': 'quant-forge 激活量化(通用张量校准)',
  'L10-F4-G6-O08': 'quant-forge 仅权重量化',
  'L10-F4-G6-O09': 'quant-forge 组量化(超块共享scale)',
  'L10-F4-G6-O10': 'quant-forge 逐通道量化',
  'L10-F4-G6-O11': 'quant-forge 逐张量量化',
  'L10-F4-G6-O14': 'quant-forge 量化反量化(往返/SNR)',
  'L10-F4-G6-O15': 'quant-forge 校准量化(absmax/affine)',
  'L10-F5-G7-O00': 'sampling-forge 自回归生成',
  'L10-F5-G7-O01': 'diffusion-forge DDPM/DDIM(扩散模型)',
  // F6 强化学习
  'L10-F6-G3-O00': 'dqn-forge DQN更新(Mnih 2015)',
  'L10-F6-G3-O01': 'dqn-forge 经验回放(均匀采样环形缓冲)',
  'L10-F6-G3-O02': 'dqn-forge 目标网络(Polyak软更新)',
  'L10-F6-G3-O03': 'dqn-forge 双重DQN(van Hasselt 2015)',
};

const FAM = [
  { f: 'F0', name: '数值基础', layer: 'L1', groups: [
    { g: 'G0', name: '确定性随机', ops: ['sfc32 PRNG', 'splitmix32 种子派生', 'mulberry32', 'xoshiro128', 'PCG 发生器', '高斯采样 Box-Muller', '高斯采样 Acklam 逆CDF', '拒绝采样基础', '分层抽样', 'Fisher-Yates 打乱', '均匀分布', '伯努利分布', '泊松分布', '指数分布', '随机质量检验', '种子可复现保证'] },
    { g: 'G1', name: '浮点稳定', ops: ['Kahan 求和', 'Neumaier 求和', 'logsumexp 稳定', '稳定 softmax', 'log1p-exp 稳定', '数值下溢防护', '数值上溢防护', '舍入误差界', '互补误差函数', '双曲线稳定', '范数稳定', '行列式稳定', 'Gram-Schmidt 正交', '极坐标稳定', '对数稳定', '除法保护'] },
    { g: 'G2', name: '哈希与校验', ops: ['纯 JS SHA-256', 'FNV-1a 占位', '校验和', '幂等校验', '双端一致哈希', '抗碰撞属性', '报告指纹', '模块指纹', '数据指纹', '增量哈希', '向量哈希', '矩阵哈希', '字符串哈希', '哈希对齐规范', 'salted 哈希', '哈希回归测试'] },
    { g: 'G3', name: '向量化与并行', ops: ['分块 GEMM', 'SIMD 式展开', '循环分块', '缓存友好布局', '向量归约', '向量映射', '向量扫描', '向量压缩', '并行 map', '并行 reduce', '任务分片', '算子流水线', '内存对齐', '零拷贝视图', '批处理', '流式处理'] },
    { g: 'G4', name: '误差与容差', ops: ['绝对误差', '相对误差', 'ULP 容差', '残差界', '正交容差', '求解容差', '梯度容差', '单调容差', '收敛容差', '概率容差', '组合误差', '条件数估计', '病态诊断', '稳定性阈值', '截断误差', '舍入阈值'] },
    { g: 'G5', name: '数据结构', ops: ['行优先矩阵', '零拷贝子块', '稀疏 CSR', '稀疏 CSC', '张量视图', '环形缓冲', '优先队列', '哈希表', 'B 树索引', '跳表', '位图', '布隆过滤器', '邻接表', '邻接矩阵', '栈', '双端队列'] },
    { g: 'G6', name: '测试与验证', ops: ['梯度数值检验', '不变量断言', '蒙特卡洛验证', '交叉验证', '数值回归', '精度基准', '确定性回归', '边界测试', '性质测试', 'KS 统计检验', '覆盖率统计', '模糊测试', '快照测试', '对照实现互比', '双路互比', '元不变量自检'] },
    { g: 'G7', name: '工具与 IO', ops: ['JSON 报告', '人类可读表', 'CLI 解析', '退出码规范', '日志 stderr 分流', '进度条', '配置加载', '路径规范化', '环境变量', '模块发现', '聚合运行器', '浏览器适配', '离线运行', '错误码规范', '诊断输出', '报告 schema'] },
  ]},
  { f: 'F1', name: '线性代数', layer: 'L2', groups: [
    { g: 'G0', name: '矩阵乘法与分解', ops: ['分块 GEMM', '朴素 GEMM', '矩阵向量乘', '外积', 'Kronecker 积', 'LU 分解', 'Cholesky 分解', 'QR(Householder)', 'SVD(单边Jacobi)', '对称 EVD(Jacobi)', '三角求解', '最小二乘', '矩阵求逆', '行列式', '迹', '范数族'] },
    { g: 'G1', name: '特征值与奇异值', ops: ['幂迭代', '反幂迭代', 'QR 算法', '子空间迭代', 'Lanczos', 'Arnoldi', '瑞利商', '谱分解', '谱范数', '条件数', '伪逆', '截断 SVD', '低秩近似', '主成分', '非负矩阵分解', '张量分解'] },
    { g: 'G2', name: '线性方程组', ops: ['高斯消元', 'LU 求解', 'Cholesky 求解', 'QR 求解', '共轭梯度', '预处理 CG', 'GMRES', '双共轭梯度', 'Jacobi 迭代', 'Gauss-Seidel', 'SOR 松弛', '直接法误差', '迭代法收敛', '稀疏求解', '最小二乘解', '约束求解'] },
    { g: 'G3', name: '矩阵函数', ops: ['矩阵指数', '矩阵对数', '矩阵平方根', '矩阵幂', '矩阵符号', '矩阵余弦', '矩阵正弦', '矩阵多项式', '矩阵分式', 'Sherman-Morrison 引理', '矩阵秩1更新', 'Schur 分解', '极分解', '矩阵范数', '矩阵条件', '矩阵谱'] },
    { g: 'G4', name: '张量运算', ops: ['张量收缩', 'Einstein 求和', '张量转置', '张量重塑', '张量切片', '广播', '外积展开', '模式积', 'Tucker 分解', 'CP 分解', '张量秩', '张量范数', '张量缩并', '张量求导', '张量索引', '张量视图'] },
    { g: 'G5', name: '数值线性代数稳定', ops: ['求逆稳定', '正交化稳定', '特征值稳定', '病态诊断', '条件数估计', '迭代改进', '向后误差', '前向误差', '平衡缩放', '枢轴选择', '秩揭示', '零空间', '列空间', '正交投影', '最小范数解', '残差精炼'] },
    { g: 'G6', name: '特殊矩阵', ops: ['对角矩阵', '三对角矩阵', '带状矩阵', 'Toeplitz', '循环矩阵', '稀疏矩阵', '正定矩阵', '正交矩阵', '酉矩阵', '置换矩阵', '投影矩阵', '幂等矩阵', '对称矩阵', '反对称矩阵', '随机矩阵', '低秩矩阵'] },
    { g: 'G7', name: '几何与变换', ops: ['旋转矩阵', 'Householder 反射', 'Givens 旋转', '四元数', '仿射变换', '齐次坐标', '投影变换', '相似变换', '正交变换', '雅可比行列式', '体积元', '重心坐标', '三角剖分', '最近邻变换', '保角映射', '小波变换'] },
  ]},
  { f: 'F2', name: '优化', layer: 'L3', groups: [
    { g: 'G0', name: '一阶方法', ops: ['梯度下降', 'SGD', '动量法', 'Nesterov 加速', 'AdaGrad', 'RMSProp', 'Adam', 'AdamW', 'Adadelta', 'Adafactor', 'LAMB', 'Lion', 'Shampoo', '重球法', '自适应矩变体', '带动量平均'] },
    { g: 'G1', name: '二阶与拟牛顿', ops: ['牛顿法', '高斯牛顿', 'L-BFGS', 'DFP', '共轭梯度法', '信赖域', '拟牛顿更新', '海森近似', '阻尼牛顿', 'OWL-QN', '有限记忆 BFGS', '拟牛顿校正', '线搜索', '强 Wolfe 条件', '无 Hessian 牛顿', '曲率条件'] },
    { g: 'G2', name: '学习率调度', ops: ['阶跃衰减', '余弦退火', 'Warmup', '循环学习率', '自适应调度', '指数衰减', '余弦重启', '单周期策略', '余弦衰减', '逆平方根', '三角循环', '1cycle', '分层衰减', '线性预热', '余弦预热', '衰减到零'] },
    { g: 'G3', name: '约束与正则', ops: ['L1 Lasso', 'L2 Ridge', '弹性网', '投影梯度', '罚函数法', '拉格朗日', '增广拉格朗日', '内点法', '近端算子', '软阈值', '硬阈值', '组稀疏', '结构化稀疏', '熵正则', 'KL 约束', '约束投影'] },
    { g: 'G4', name: '随机优化', ops: ['momentum-SGD', '小批量', 'SAG', 'SVRG 方差缩减', '随机 L-BFGS', '异步 SGD', '梯度裁剪', '梯度累积', '随机平均梯度', '方差缩减', '局部 SGD', '联邦平均', '动量方差缩减', '自适应采样', '噪声注入', '锐度感知最小化'] },
    { g: 'G5', name: '组合与整数', ops: ['贪心', '分支定界', '动态规划', '模拟退火', '遗传算法', '粒子群', '蚁群', '禁忌搜索', '爬山', '随机局部搜索', '差分进化', '交叉熵方法', '贝叶斯组合优化', '整数规划', '约束满足', '局部搜索'] },
    { g: 'G6', name: '凸优化', ops: ['线性规划', '二次规划', '半定规划', '内点法', '对偶方法', '次梯度法', '近端梯度', '对偶上升', '椭球法', '切割平面', '加速近端', 'ADMM', '坐标上升', '原始对偶', '锥规划', '锥规划内点'] },
    { g: 'G7', name: '非凸与全局', ops: ['坐标下降', '交替最小化', 'EM 算法', '贝叶斯优化', '演化策略', '采集函数', '多臂老虎机', '元学习优化', '模拟退火全局', '随机搜索', '贝叶斯全局', '高斯过程优化', '树搜索优化', '探索利用平衡', 'CMA-ES', '黎曼优化'] },
  ]},
  { f: 'F3', name: '自动微分', layer: 'L3', groups: [
    { g: 'G0', name: '前向模式', ops: ['前向累积', 'Dual 数对', '雅可比向量积 JVP', '切空间', '方向导数', '前向计算图', '算子重载', '轨迹追踪', '前向高阶', '链式法则前向', '无感模式', '复数步微分', '源到源变换', '算子记录', '线性化', '切向量传播'] },
    { g: 'G1', name: '反向模式', ops: ['反向累积', '向量雅可比积 VJP', '计算图', '拓扑排序', '反向遍历', '隐式梯度', '高阶导数', '梯度检查点', '反向前阶', '链式法则反向', '自动反向', '子表达式缓存', '反向线性化', '标量反向', '向量反向', '批反向'] },
    { g: 'G2', name: '高阶微分', ops: ['Hessian', 'Hessian-向量积', '雅可比矩阵', '高维雅可比', '张量导数', '三阶导数', '无穷阶展开', '自动高阶', 'Hessian 对角', 'Hessian 迹', '雅可比向量积', '向量雅可比积', '高阶链式法则', '泰勒展开', '微分算子', '微分代数'] },
    { g: 'G3', name: '计算图', ops: ['节点', '边', '算子注册', '类型推导', '形状推断', '常量折叠', '公共子表达式', '死代码消除', '依赖分析', '内存规划', '算子融合', '梯度节点', '控制依赖', '数据依赖', '子图提取', '图序列化'] },
    { g: 'G4', name: '算子重载', ops: ['标量重载', '向量重载', '矩阵重载', '原地操作', '控制流处理', '循环展开', '闭包捕获', '异常安全', '自定义算子', '高阶函数重载', '条件分支重载', '递归重载', '广播重载', '索引重载', '视图重载', 'in-place 安全'] },
    { g: 'G5', name: '混合模式', ops: ['前向反向混合', 'JVP-VJP 混合', '高维混合', '稀疏雅可比', '结构化雅可比', '无矩阵算子', '隐式微分', '微分规则库', '前向批', '反向批', '高阶混合', '复数微分', '随机微分', '截断微分', '截断高阶', '自定义规则'] },
    { g: 'G6', name: '性能', ops: ['内存优化', '重计算', '分片反向', '延迟执行', '向量化微分', '并行微分', '稀疏微分', '编译微分', '算子缓存', '常量缓存', '计算图压缩', '增量反向', '多线程求导', 'GPU 卸载', '内存池化', '零拷贝微分'] },
    { g: 'G7', name: '验证', ops: ['梯度数值检验', '高阶检验', '不变量检验', '双路微分', '对照实现', '精度基准', '病态梯度', '确定性检验', '解析梯度比对', '自动微分对照', '有限差分', '中心差分', '复步微分', '误差界检验', '收敛检验', '回归检验'] },
  ]},
  { f: 'F4', name: '注意力与 Transformer', layer: 'L4', groups: [
    { g: 'G0', name: '注意力核心', ops: ['缩放点积注意力', '因果掩码', '加性注意力', '多头注意力', '头部拼接', '注意力分数', 'softmax 稳定', '权重归一', '交叉注意力', '自注意力', '全局注意力', '局部注意力', '线性注意力核', '注意力 dropout', '注意力缩放', '注意力偏置'] },
    { g: 'G1', name: '位置编码', ops: ['RoPE', '绝对位置', '相对位置', 'ALiBi', '正弦位置', '可学习位置', '旋转相对', '位置插值', '旋转绝对', '轴向位置', '傅里叶特征', '相对位置偏置', '旋转位置扩展', '学习相对', '连续位置', '位置解耦'] },
    { g: 'G2', name: '归一化', ops: ['RMSNorm', 'LayerNorm', 'BatchNorm', 'GroupNorm', 'InstanceNorm', '权重归一', '在线归一', '流式归一', '自适应归一', '幂归一', '谱归一', '局部响应归一', '层归一预', '后归一', '可学习归一', '归一稳定'] },
    { g: 'G3', name: '激活与门控', ops: ['SwiGLU', 'GELU', 'ReLU', 'GELU 精确', 'SiLU', 'GLU', '门控线性单元', '激活稳定', 'GeGLU', 'ReGLU', '激活函数族', '平滑激活', '稀疏激活', '激活裁剪', '双曲激活', '概率激活'] },
    { g: 'G4', name: '高效注意力', ops: ['稀疏注意力', '线性注意力', '局部窗口', '低秩注意力', '分块注意力', '多查询 MQA', '分组查询 GQA', 'FlashAttention 风格', '滑动窗口', '扩张注意力', '稀疏全局', '低秩分解', '注意力近似', '缓存注意力', '量化注意力', '分块因果'] },
    { g: 'G5', name: 'KV 缓存与推理', ops: ['KV 缓存', '分页注意力', '增量解码', '推测解码', '并行解码', '缓存复用', '缓存逐出', '流式生成', '缓存预分配', '缓存压缩', '缓存量化', '并行采样', '树状注意力', '连续批处理', '前缀缓存', '注意力下沉'] },
    { g: 'G6', name: '量化与压缩', ops: ['int8 量化', 'int4 量化', 'k-quants', '权重量化', '激活量化', '混合精度', '平滑量化', '量化感知训练', '仅权重量化', '组量化', '逐通道量化', '逐张量量化', '稀疏量化', '旋转量化', '量化反量化', '校准量化'] },
    { g: 'G7', name: '模型组件', ops: ['嵌入层', '词表投影', '残差连接', '前馈网络', '预归一化', '子层缩放', '初始化', '参数绑定', '位置前馈', '专家混合 MoE', '路由', '共享专家', '缩放点积预', '层缩放', '深度缩放', '参数初始化族'] },
  ]},
  { f: 'F5', name: '概率与生成', layer: 'L5', groups: [
    { g: 'G0', name: '分布', ops: ['正态分布', '伯努利分布', '类别分布', '多项式分布', '指数族', 'Gamma 分布', 'Beta 分布', '狄利克雷分布', '泊松分布', '均匀分布', '拉普拉斯分布', '学生 t 分布', 'Wishart', 'Logistic', '负二项分布', '混合高斯'] },
    { g: 'G1', name: '采样', ops: ['逆 CDF 采样', '拒绝采样', '重要性采样', 'MCMC', '吉布斯采样', '切片采样', 'Hamiltonian Monte Carlo', '蛙跳积分', '独立采样', '分层采样', '拉丁超立方', '马尔可夫链采样', '自适应拒绝', 'Metropolis', '并行采样', '准蒙特卡洛'] },
    { g: 'G2', name: '变分推断', ops: ['均值场变分', '黑盒变分推断', '重参数化技巧', '证据下界 ELBO', '变分自编码器', '正态化流', '提议分布', '变分族', '摊销变分推断', '结构化变分', '递归变分', '变分 dropout', '层次变分', '离散变分', 'ELBO 优化', '变分期望最大化'] },
    { g: 'G3', name: '贝叶斯', ops: ['贝叶斯更新', '共轭先验', '马尔可夫链贝叶斯', '贝叶斯估计', '可信区间', '贝叶斯因子', '层次模型', '贝叶斯神经网络', '贝叶斯模型平均', '贝叶斯优化', '贝叶斯推断', '贝叶斯线性回归', '贝叶斯非参数', '贝叶斯交叉验证', '贝叶斯模型选择', '贝叶斯决策'] },
    { g: 'G4', name: '随机过程', ops: ['高斯过程', '连续马尔可夫链', '泊松过程', '维纳过程', '布朗运动', '平稳过程', '自回归过程', '隐马尔可夫模型', '马尔可夫随机场', '高斯随机场', '扩散过程', '更新过程', '分支过程', 'Lévy 过程', '马尔可夫毯', '条件随机场'] },
    { g: 'G5', name: 'MCMC 进阶', ops: ['Metropolis', 'Metropolis-Hastings', '吉布斯抽样', 'NUTS', '随机游走 Metropolis', '独立采样', '退火采样', '并行回火', '可逆跳跃', '延迟拒绝', '辅助变量', '哈密顿更新', '黎曼流形 HMC', '自适应步长', '混合蒙特卡洛', '窗宽调优'] },
    { g: 'G6', name: '估计与检验', ops: ['最大似然估计', '最大后验估计', '矩估计', '贝叶斯估计', 'KS 检验', '卡方检验', 'Bootstrap', '交叉验证似然', '矩方法', '拟似然', '稳健估计', '经验似然', '假设检验', '置信区间', '似然比检验', '拟合优度'] },
    { g: 'G7', name: '生成模型', ops: ['自回归生成', '扩散模型', '流模型', '能量模型', '生成对抗', '变分生成', '得分匹配', '朗之万采样', '归一化流生成', '自回归流', '隐扩散', '一致性模型', '扩散蒸馏', '变分扩散', '能量生成', '生成式预训练'] },
  ]},
  { f: 'F6', name: '强化学习', layer: 'L6', groups: [
    { g: 'G0', name: '基础', ops: ['马尔可夫决策过程', '值函数', '动作值函数', '策略', '奖励函数', '折扣因子', '状态转移', '贝尔曼方程', '状态空间', '动作空间', '转移概率', '回报', '折扣回报', '最优策略', '最优值函数', '最优动作值'] },
    { g: 'G1', name: '值方法', ops: ['动态规划', '策略迭代', '值迭代', '蒙特卡洛评估', '时序差分', 'SARSA', 'Q 学习', '双重 Q 学习', '蒙特卡洛控制', 'TD(λ)', '资格迹', '期望 SARSA', '树备份', '实时动态规划', '广义优势之值', '分布式值迭代'] },
    { g: 'G2', name: '策略梯度', ops: ['REINFORCE', '优势函数', '基线减除', 'TRPO', 'PPO', '近似策略梯度', '自然梯度', '确定性策略梯度', '演员评论家', 'A2C', '信赖域评论', '广义优势估计', '离策略梯度', '重要性采样梯度', '策略梯度定理', '软策略梯度'] },
    { g: 'G3', name: '深度 RL', ops: ['DQN', '经验回放', '目标网络', '双重 DQN', '决斗网络', '优先经验回放', '噪声网络', '分布式 RL', 'C51', 'QR-DQN', '逆强化学习', '模仿学习', '行为克隆', '离策略', '在线策略', '分布式 Q'] },
    { g: 'G4', name: '基于模型', ops: ['动力学模型', '模型预测控制', '蒙特卡洛树搜索', 'AlphaZero', 'MuZero', '世界模型', '规划', '想象滚动', '模型基 RL', '不确定性规划', '贝叶斯规划', '滚动地平线', '树搜索策略', '价值搜索', '模型集成', '模型损失'] },
    { g: 'G5', name: '探索', ops: ['ε-贪婪', '熵正则化', '上置信界 UCB', 'Thompson 采样', '内在奖励', '好奇驱动', '计数探索', '噪声探索', '玻尔兹曼探索', '随机网络蒸馏', '伪计数', '方差探索', '信任区域探索', '最大熵探索', '势能探索', '信息增益'] },
    { g: 'G6', name: '多智能体', ops: ['马尔可夫博弈', '纳什均衡', '独立学习', '集中训练分散执行', '信用分配', '通信协议', '合作博弈', '竞争博弈', '平均场博弈', '图博弈', '多智能体策略梯度', '多智能体 Q', '协调博弈', '对抗博弈', '谣言博弈', '社会困境'] },
    { g: 'G7', name: '应用', ops: ['推荐系统 RL', '对话策略', '机器人控制', '组合优化 RL', '元 RL', '分层 RL', '离线 RL', '安全 RL', '自动驾驶 RL', '游戏 RL', '金融 RL', '健康 RL', 'RL 推荐', '持续学习 RL', '多任务 RL', '鲁棒 RL'] },
  ]},
  { f: 'F7', name: '检索/图/系统', layer: 'L7', groups: [
    { g: 'G0', name: '向量检索', ops: ['余弦相似', '内积检索', 'L2 检索', '近似近邻 ANN', '穷举检索', '量化检索', '分层导航 HNSW', '倒排文件', '最大内积检索', '检索重排', '检索融合', '候选生成', '相似度阈值', '批量检索', '近邻图构建', '近邻图搜索'] },
    { g: 'G1', name: '索引结构', ops: ['KD 树', '球树', 'LSH', 'Annoy 树', '倒排索引', '乘积量化 PQ', 'IVFPQ', '局部敏感哈希', 'ScaNN', '旋转不变 LSH', '多索引 HNSW', '图索引', '树索引', '哈希索引', '粗量化', '细量化'] },
    { g: 'G2', name: '图算法', ops: ['BFS', 'DFS', 'Dijkstra', 'A*', '最短路径', 'PageRank', '连通分量', '最小生成树', '拓扑排序', '强连通分量', '二分图匹配', '最大流', '社区发现', '三角计数', '图着色', '中心性度量'] },
    { g: 'G3', name: '图神经网络', ops: ['图卷积 GCN', 'GraphSAGE', '注意力图网络 GAT', '消息传递', '节点分类', '图池化', '边预测', '图采样', '图回归', '图自编码器', '图注意力池化', '图同构网络', '边分类', '子图匹配', '图生成', '图预训练'] },
    { g: 'G4', name: '稀疏与压缩', ops: ['稀疏矩阵乘', '稀疏格式转换', '压缩稀疏行', '压缩存储', '低秩压缩', '剪枝', '稀疏注意力', '稀疏梯度', '稀疏向量', '稀疏卷积', '量化压缩', '哈夫曼编码', '游程编码', '字典压缩', '增量压缩', '列式压缩'] },
    { g: 'G5', name: '序列化与存储', ops: ['JSON 序列化', '二进制序列化', '列式存储', '内存映射', '增量写入', '校验持久化', '版本化存储', '快照', '协议缓冲区', '分区存储', '对象存储', '本地缓存', '远端缓存', '事务写入', '灾难恢复', '数据校验和'] },
    { g: 'G6', name: '系统与部署', ops: ['推理批处理', '算子融合', '内核调度', '内存池', '零拷贝', '流式管线', '负载均衡', '容错', '自动扩缩容', '请求队列', '预热缓存', '降级策略', '限流', '监控埋点', '灰度发布', '回滚'] },
    { g: 'G7', name: '评测与监控', ops: ['精度评测', '延迟评测', '吞吐量评测', '召回率评测', '一致性评测', '资源监控', '漂移检测', '回归监控', 'F1 评测', 'AUC 评测', '困惑度评测', 'BLEU 评测', '人工评测', '对抗评测', '公平性评测', '可解释评测'] },
  ]},
];

const FAM_LAYER = {};
FAM.forEach(fa => { FAM_LAYER[fa.f] = fa.layer; });

function main() {
  const leaves = [];
  for (const fa of FAM) {
    for (const gp of fa.groups) {
      if (gp.ops.length !== 16) throw new Error(fa.f + '/' + gp.g + ' 期望 16 微指令，实际 ' + gp.ops.length);
      gp.ops.forEach((name, i) => {
        const o = 'O' + ('0' + i).slice(-2);
        const id = 'L10-' + fa.f + '-' + gp.g + '-' + o;
        const reuse = REUSE[id] || null;
        const ref = reuse ? reuse : '规划：W3-W7 后续实现（含 @reuse 顶级参照）';
        const status = reuse ? (moduleImplemented(reuse) ? 'done' : 'planned') : 'planned';
        leaves.push({ id, family: fa.f, group: gp.g, op: o, name, layer: fa.layer, status, reuse: ref });
      });
    }
  }
  if (leaves.length !== 1024) throw new Error('L10 条目应为 1024，实际 ' + leaves.length);

  fs.mkdirSync(path.dirname(OUT_JSON), { recursive: true });
  const idx = {
    schema: 'forge-l10/1.0', generated_at: new Date().toISOString(), total: leaves.length,
    families: FAM.length, groups: FAM.reduce((a, f) => a + f.groups.length, 0), ops_per_group: 16,
    by_status: leaves.reduce((a, l) => { a[l.status] = (a[l.status] || 0) + 1; return a; }, {}),
    leaves
  };
  fs.writeFileSync(OUT_JSON, JSON.stringify(idx, null, 2));

  const L = [];
  L.push('# L10 微指令层分类法（1024 项）');
  L.push('');
  L.push('> 十层架构第十层：8 族(F0-F7) × 8 组(G0-G7) × 16 微指令(O0-O15) = **1024** 项微指令级操作。');
  L.push('> 原则：**零重写 · 集大成** —— 每条标注世界顶级 `@reuse` 参照；`done`=已被 forge 模块覆盖，`planned`=W3-W7 后续落地。');
  L.push('');
  L.push('| 状态 | 数量 |');
  L.push('| --- | --- |');
  L.push('| done | ' + idx.by_status.done + ' |');
  L.push('| planned | ' + idx.by_status.planned + ' |');
  L.push('| **合计** | **' + leaves.length + '** |');
  L.push('');
  L.push('## 全景映射');
  L.push('');
  L.push('| 族 | 名称 | 层 | 组数 | 微指令(×16) |');
  L.push('| --- | --- | --- | --- | --- |');
  for (const fa of FAM) {
    L.push('| ' + fa.f + ' | ' + fa.name + ' | ' + fa.layer + ' | ' + fa.groups.length + ' | ' + (fa.groups.length * 16) + ' |');
  }
  L.push('');
  for (const fa of FAM) {
    L.push('## ' + fa.f + ' · ' + fa.name + '（层 ' + fa.layer + '）');
    L.push('');
    for (const gp of fa.groups) {
      L.push('### ' + gp.g + ' · ' + gp.name);
      L.push('');
      L.push('| 微指令 ID | 名称 | 层 | 状态 | @reuse 顶级参照 |');
      L.push('| --- | --- | --- | --- | --- |');
      gp.ops.forEach((name, i) => {
        const o = 'O' + ('0' + i).slice(-2);
        const id = 'L10-' + fa.f + '-' + gp.g + '-' + o;
        const leaf = leaves.find(l => l.id === id);
        L.push('| `' + id + '` | ' + name + ' | ' + leaf.layer + ' | ' + leaf.status + ' | ' + leaf.reuse + ' |');
      });
      L.push('');
    }
  }
  L.push('---');
  L.push('');
  L.push('*生成器：tools/gen-l10.js · schema forge-l10/1.0 · 作者：晨星 · MIT*');
  fs.writeFileSync(OUT_MD, L.join('\n'));

  console.log('L10 分类法生成完成：' + leaves.length + ' 项（done=' + idx.by_status.done + ', planned=' + idx.by_status.planned + '）');
  console.log('  → ' + path.relative(ROOT, OUT_MD));
  console.log('  → ' + path.relative(ROOT, OUT_JSON));
}

main();
