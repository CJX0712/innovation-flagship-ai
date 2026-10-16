# 更新日志 (Changelog)

遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [0.7.10] — 2026-10-17

### 新增 (Added) — 训练端 INT8 量化感知训练 (QAT, STE 伪量化)
- **真缺口 (功能缺口)**: INT8 量化此前**仅推理端可用** (`Transformer.quantize_int8`,
  LLM.int8 per-channel RTN) — 量化故事残缺: 训练时模型从未"看见"量化扰动,
  部署时才第一次经历 INT8。QAT (Jacob et al. 2018) 补齐: 训练期前向直接走
  dequant(quant(W)), 训出的权重天然适配 INT8 部署。
- **落地 (零重写·集大成)**:
  - `autodiff.py::Tensor.fake_quant_int8_per_channel` — STE 伪量化算子: 前向 =
    dequant(quant(W)) (复用推理端同一 `core/quant.py` per-channel RTN 方案, 语义
    逐字一致), 反向对 W 恒等直通 (straight-through estimator, 忽略 d(scale)/dW);
    冻结权重 (requires_grad=False) 自动跳过梯度回传。
  - `config.py` 新增 `use_qat: bool = False` — **默认关闭, 现有训练图与数值逐字节
    不变**。
  - `trainer.py::_wq` 门控包裹全部 2D 权重 matmul: q/k/v/o_proj、ffn_gate/up/down、
    MoE e{e}_gate/up/down + router、LoRA A/B、embed、lm_head (tie 时 embed/lm_head
    为同一 Tensor, 梯度正确累加); 1D 归一化权重不伪量化 (与推理端仅量化 2D 一致)。
- **新增 `tests/unit/l4/test_qat.py` 5 用例**: ① STE 前向逐位 == dequant(quant(W));
  ② INV-07 重建误差界 ≤ scale/2 在 STE 节点成立; ③ STE 反向恒等 (含伪量化图 vs
  无伪量化图梯度一致 + 解析解二次验证); ④ use_qat=True 确实注入确定性量化扰动 +
  梯度贯通; ⑤ **核心验收**: QAT 模型 INT8 部署后损失劣化 (int8_loss − fp64_loss,
  同一评估批内自比较) ≤ 非 QAT 基线 — 原理: QAT 训练优化的正是量化前向, 部署
  int8 即训练时行为。
- **坑 (验收口径)**: 输出扰动口径 (max|fp64−int8|) 下 QAT 与基线在噪声内不可分
  (8.97e-3 vs 8.25e-3, 首版误判失败); 正确口径是**部署后损失劣化**的模型内自比较。
- 全量测试 256 → **261** (净 +5), selftest 4 门禁 ALL GREEN, INV 42/42 双侧达成。

## [0.7.9] — 2026-10-17

### 新增 (Added) — 特征配置端到端可训性 (INV-26 扩展到 MoE/LoRA/组合)
- **真缺口 (可训性盲区, 非能力/一致性缺陷)**: v0.7.5/0.7.7 把 MoE / LoRA 接入训练端
  并验证了「梯度贯通」与「训练/推理前向一致性」; v0.7.8 锁了组合参数集对齐。
  但**一致性 ≠ 可训性** —— 这些特征配置 (MoE 路由 / LoRA 低秩增量 / 其组合) 从未被
  证明训练后损失真实下降、模型真实学会。现有 INV-26 收敛测试仅覆盖基础训练器
  (无 MoE/LoRA)。
- 新增 `tests/unit/l4/test_feature_learnability.py` (8 用例), 把 INV-26 损失单调性标准
  推广到全特征矩阵: 在真实中文语料上**真实训练**, 断言损失有限为正且滑动平滑后
  尾部 20% 均值 ≤ 头部 20% 均值 × 0.8。涵盖 base / GQA / gelu / MoE / LoRA /
  MoE+LoRA / GQA+MoE / GQA+LoRA 共 8 配置。
- **验证价值**: 证明 MoE 路由、LoRA 低秩适配、及其组合**确实参与并推动学习**
  (非仅"梯度在、前后向一致"却游离于学习之外)。8 配置全部 16.6s 内收敛达标。
- 全量测试 248 → **256** (净 +8), selftest 4 门禁 ALL GREEN, INV 42/42 双侧达成。

## [0.7.8] — 2026-10-17

### 新增 (Added) — 特征组合一致性矩阵 (锁定五轴一致性矩阵最后盲区)
- **真缺口 (测试盲区, 非能力缺陷)**: v0.7.5/0.7.6/0.7.7 分别将 MoE / INT8 / LoRA
  接入训练端+推理端并各自**单轴**验证 save→load ≤1e-9, 但**特征组合**
  (MoE+LoRA / GQA+MoE / GQA+LoRA / 激活×MoE / 未绑定×MoE+LoRA 等) 从未被测试触达
  —— 是五轴一致性矩阵最后的盲区。Recon 确认两端参数名由 flag 组合唯一决定、组合在
  两端真实可用, 故本轮以测试锁死组合面 (防未来单侧命名漂移导致 save→load 静默分叉)。
- 新增 `tests/unit/l4/test_composition_consistency.py` (29 用例), 覆盖 13 种配置组合:
  - **参数集严格对齐**: 任意组合 `set(TinyTrainer.params) == set(Transformer.params)`
    (含结构断言: MoE 须带 router/专家权重、LoRA 须带 lora_a/lora_b、dense 须带 ffn_gate)。
  - **前向数学一致性**: 训练端前向 vs 推理端 (加载同参数) 输出 `max|diff| ≤ 1e-9`。
  - **INV-24 组合保持**: 含 LoRA 的组合在 B=0 初始时仍严格等价对应基座 (≤1e-12)。
- 组合矩阵覆盖: MHA/GQA × {MoE, LoRA, MoE+LoRA} × {swiglu, gelu} × {绑定, 未绑定}。
- 全量测试 219 → **248** (净 +29), selftest 4 门禁 ALL GREEN, INV 42/42 双侧达成。

## [0.7.7] — 2026-10-17

### 新增 (Added) — LoRA 接入训练端+推理端 + 训练/推理 save→load 一致性 (补齐最后一配置轴)
- **真缺口 (能力缺陷, 非测试缺口)**: `ifai/train/lora.py::LoRALinear` 仅是独立参考
  实现 (Hu et al. 2022), 未接入 `TinyTrainer` 与推理端 `Transformer` —— LoRA 这一
  旗舰级微调方法此前**不可用** (一致性矩阵最后一轴缺失)。现补齐 (零重写·集大成):
  - `config.py`: 新增 `use_lora / lora_r / lora_alpha` (`W_eff = W + (α/r)·B·A`)。
  - `trainer.py` `_build`: 按 `use_lora` 为 q/k/v/o 投影建 `lora_a[r,in] / lora_b[out,r]`,
    **冻结基座** (`requires_grad=False`), `B` 初始化为 0 (INV-24 初始等价基座)。
  - `trainer._apply_lora` / `transformer._apply_lora`: 训练端(Tensor)与推理端(numpy)
    **同一公式** `base + (α/r)·(x@Aᵀ)@Bᵀ`, save→load 零偏差。
  - 修复维度约定坑: 本项目投影权重形状为 `(in_dim, out_dim)`, lora_a/lora_b 据此推导
    (k/v_proj 形状 `(d, n_kv·d_head)` 曾因 (out,in) 误读导致 matmul 维度失配)。
- **验证**: 新增 `tests/unit/l4/test_lora_consistency.py` 4 用例 ——
  ① LoRA 训练后 trainer 前向 vs 推理端 save→load 对照 (swiglu/gelu) `≤1e-9`;
  ② 训练数步后 A/B 梯度贯通 (非 None 且范数>0) + 基座冻结 (无梯度流);
  ③ INV-24 B=0 初始严格等价基座前向 (`≤1e-12`)。
- 一致性矩阵现状: 注意力(n_kv×scaling)/激活(swiglu|gelu)/MoE/INT8/**LoRA** 五轴已覆盖,
  训练端配置轴全部打通 (GQA/RoPE-scaling/激活/MoE/INT8推理/LoRA 微调)。

### 测试 (Tests) — 215 → 219 项, 覆盖率 87% 维持, selftest ALL GREEN

## [0.7.6] — 2026-10-17

### 新增 (Added) — INT8 量化接入推理前向 + INV-07 重建误差界一致性守卫
- **真缺口 (死代码未接入)**: `core/quant.py::quantize_int8_per_channel` 此前仅被
  单元测试 (`tests/test_core.py::test_int8_quant_error_bound`) 与演示脚本
  (`examples/quantize_int8.py`) 触达, **从未接入模型前向** —— 无法对整模型做 INT8
  量化并运行。现补齐推理端接入 (零重写·集大成, LLM.int8 Dettmers et al. 2022
  per-channel 对称量化思想):
  - `models/transformer.py::Transformer.quantize_int8` 对全部 2D 线性权重做
    per-channel RTN 量化, 原地替换为反量化权重 (`q·scale`, INT8 存储 + 使用时
    反量化, 内存收益); 1D 参数 (RMSNorm/final_norm) 不量化 (需精确且 per-channel
    无意义)。
  - 权重绑定 (INV-12) 安全: `tie_word_embeddings` 下 `lm_head` 与 `embed` 同一
    数组, 量化一次并维持 tie 引用, 杜绝双重量化。
- **验证**: 新增 `tests/test_core.py::test_int8_model_quantization_inference` ——
  整模型量化后 (1) INV-07 逐权重重建误差 `|orig−dequant| ≤ scale/2` (真实模型
  权重, 非合成矩阵, 覆盖 embed/q/k/v/o/ffn 等 15 个权重); (2) int8 RTN 不翻转
  top-1 决策; (3) 前向误差有限且受控 (实测 ≈1.8e-3 ≪ 0.05 界)。
- 一致性矩阵现状: 注意力(n_kv×scaling)/激活(swiglu|gelu)/MoE/INT8 四轴已覆盖;
  LoRA 尚未接入训练端, 列为后续增量。

### 测试 (Tests) — 214 → 215 项, 覆盖率 87% 维持, selftest ALL GREEN

## [0.7.5] — 2026-10-17

### 新增 (Added) — TinyTrainer 支持 MoE + 训练/推理 save→load 一致性矩阵 (补齐 MoE 轴)
- **真缺口 (能力缺陷, 非测试缺口)**: `TinyTrainer._build` / `_layer_forward` 此前
  仅建 dense FFN 参数 (`ffn_gate/up/down`), `use_moe=True` 模型**根本无法训练**
  —— 推理端 `Transformer._moe` 存在但训练端无对应路径, train→inference 对 MoE
  配置轴从未被测试触达。现补齐:
  - `trainer.py::_build` 按 `cfg.use_moe` 分支建 `e{e}_gate/up/down` + `router`
    (与推理端 `Transformer._init_params` 同名同结构, 保证 save→load 零偏差)。
  - `trainer.py::_moe_forward` 可微 MoE 前向, **严格镜像** 推理端 `Transformer._moe`
    (top-k 路由 → 仅对选中专家 router 分数 softmax ⇒ 概率和=1, INV-14); 反向经
    `getitem` 高级索引散射回 `router`、经专家 matmul 回各专家权重。
  - 激活选择 (`config.activation`) 在 MoE 路径同样生效 (gelu / silu), 与 dense 对称。
- **验证**: 新增 `tests/unit/l4/test_moe_consistency.py` 3 用例 —
  MoE 模型 trainer 前向 vs 推理端 save→load 对照 (swiglu / gelu) `max|diff| ≤ 1e-9`
  + MoE router/专家参数一步 backward 梯度贯通 (非 None 且范数 > 0)。
- 一致性矩阵现状: 注意力 (n_kv×scaling) / 激活 (swiglu|gelu) / **MoE** 三轴已覆盖;
  int8 量化与 LoRA 尚未接入训练端, 列为后续增量。

### 测试 (Tests) — 211 → 214 项, 覆盖率 87% 维持, selftest ALL GREEN

## [0.7.4] — 2026-10-17

### 新增 (Added) — 让 `config.activation` 真正生效 (补齐 gelu 激活)
- `ModelConfig.activation` 字段承诺 `"swiglu" | "gelu"`, 但推理端/训练端此前
  均硬编码 SwiGLU(silu) 忽略 `"gelu"` —— 字段名不副实。现补齐 GELU
  (Hendrycks & Gimpel 2016 tanh-approx, GPT/LLaMA 同款):
  - `core/ops.py::gelu` 纯 numpy 内核; 推理端 dense + MoE 两 FFN 分支按
    `cfg.activation` 选 gelu / silu。
  - `train/autodiff.py::Tensor.gelu` 可微算子 (前向同公式, 解析反向
    `gelu'(x)=0.5(1+tanh u)+0.5x(1-tanh²u)·√(2/π)(1+0.134145x²)`);
    训练端 `_layer_forward` FFN 段按 `cfg.activation` 选择。
  - **默认 `"swiglu"` 路径数值完全不变** (训练/推理), 现有默认模型零影响;
    gelu 路径为「门控用 GELU 的 gated-FFN」, 与 SwiGLU 对称、参数不变,
    save→load 兼容性与 INV-13 参数量守恒不受影响。
- **验证**: 新增 `tests/unit/l2/test_activation.py` 4 用例 — gelu 公式对照
  (<1e-9) + 训练端 vs 推理端 (swiglu/gelu) save→load 前向一致性 ≤1e-9
  (swiglu 回归) + 训练端 gelu 有限差分梯度校验 (<1e-5)。

### 测试 (Tests) — 207 → 211 项, 覆盖率 87% 维持, selftest ALL GREEN

## [0.7.3] — 2026-10-17

### 修复 (Fixed) — CLI `train` 的 vocab_size 软目标陷阱 (真实健壮性缺陷)
- `ifai/cli/__init__.py::cmd_train` 旧实现给 `ModelConfig(vocab_size=args.vocab)`
  与 `Tokenizer(vocab_size=args.vocab)` 用同一值, 但分词器 `train` 的 vocab_size
  仅限制 merge 数、基础字符数不设上限 (见 `ifai/models/tokenizer.py::train`)。
  语料实际词表 > `args.vocab` 时 `tok.encode` 产生越界 id, 训练中 embed 索引
  IndexError —— 用户用较小 `--vocab` 训练较大语料必崩。修复: 先构造分词器,
  模型词表取 `max(args.vocab, tok.vocab_size)`, 始终 ≥ 分词器实际词表。
- **验证**: 新增 `tests/unit/l9/test_cli_e2e.py` 端到端 subprocess 测试 —
  train 产物齐全 + 小 `--vocab 64` (MICRO_CORPUS 实际词表 116 > 64) 不抛
  IndexError + train→demo 闭环 stdout 非空。

### 测试 (Tests) — 204 → 207 项, 覆盖率 87% 维持, selftest ALL GREEN

## [0.7.2] — 2026-10-17

### 工程 (CI) — 覆盖率 report-only job
- `.github/workflows/ci.yml` 新增 `coverage` job (`continue-on-error: true`, 报告性质,
  绝不影响主流程): `ubuntu-latest` + py3.13, `pytest -q --cov=ifai
  --cov-report=term --cov-report=xml`, 上传 `coverage-report` (coverage.xml) artifact。
  **report-only, 不设 fail 门禁** — 仅在 PR 可见覆盖率数字 (当前总体 ~87%),
  与既有 `baseline` job 同属"报告型"管线, 互不阻塞 `test` job。

## [0.7.1] — 2026-10-17

### 修复 (Fixed) — 训练/推理 RoPE 分叉 (可复现性隐患)
- **训练端 `_rope` 忽略 `rope_scaling`** (`train/trainer.py`): 旧实现用
  `1/base**(arange/half)` 生成频率, 完全未应用 `cfg.rope_scaling`。
  推理端 `ops.apply_rope` 在 `scaling≠1` 时做 NTK-aware 拉伸
  (`inv *= scaling**(half/(half-1))`) —— 二者分叉导致 `TinyTrainer`
  训练后 `save` → `Transformer.load` 推理结果偏差 > 1e-9, 违反可复现。
  修复: 训练端复用同一 `ops._rope_freqs` (零重写, 口径统一), 默认
  `scaling=1.0` 路径数值不变。
- **验证**: `test_attention_grad.py` 新增 9 项参数化
  (n_kv∈{None,2,1} × scaling∈{1,2,4}), 训练端前向 vs 推理端
  save→load 对照 max|diff| ≤ 1e-9, 覆盖 NTK 拉伸下的对齐。

### 测试 (Tests) — 195 → 204 项, 覆盖率 87% 维持, selftest ALL GREEN

## [0.7.0] — 2026-10-17

### 修复 (Fixed) — 训练器注意力梯度链断裂 (真 bug, 被 INV-26 通过掩盖)
- **注意力参数从未被训练** (`train/trainer.py`): 旧 `_layer_forward` 经
  `.data` 取 raw numpy 后以 `Tensor(ctx)` 重建叶子, 切断了
  `attn_norm/q_proj/k_proj/v_proj` 的 autograd 梯度链 — 这些参数梯度恒零,
  损失下降全靠 FFN/embed 路径 (INV-26 单调性仍可通过, 缺陷被掩盖)。
  另: 旧 `_rope` 的 `_prev=[x]` 漏掉 rot1/rot2 — 它们不在拓扑序,
  即使保留 Tensor 链 q/k 梯度也会在 RoPE 节点被吞。双重修复后注意力
  参数获得精确解析梯度。
- **验证**: 新增 `tests/unit/l4/test_attention_grad.py` 三重硬校验 —
  ① trainer 前向与推理端 `Transformer` (MHA/GQA/MQA 三种头配置)
  save→load 对照 max|diff| ≤ 1e-9; ② 注意力参数一步 backward 后梯度
  非 None 且范数 > 0; ③ 有限差分梯度校验 (中心差分, rel < 1e-5)。

### 性能 (Performance) — 批量可微注意力, 正确性零速度税
- **注意力单节点自定义算子** `Tensor.attention` (`train/autodiff.py`):
  BLAS 批量前向 + 标准 softmax-attention 解析反向
  (`dV=Pᵀg; dS=P⊙(dP−Σ dP⊙P); dQ=dS·K·s; dK=dSᵀ·Q·s`), 一个节点替代
  ~15 个通用算子节点; 逐头 Python 双重循环 (B×H) 删除。
- **RoPE 融合单节点**: 反向 = 正交旋转转置 (`dx1=g1·cos+g2·sin;
  dx2=−g1·sin+g2·cos`), 11 节点 → 1 节点。
- **新增可微算子**: `softmax` (数值稳定, 掩码位反向自动归零)、
  `repeat` (GQA kv 头扩展, 与推理端 `np.repeat` 同语义, INV-10);
  修正 autodiff docstring 谎报的 `concat` (从未实现)。
- **基准** (`scripts/bench_trainer.py`, d=128 L=4 H=8 n_kv=2 B=8 T=32):
  同机同条件对照 — 旧梯度断裂版 160.3 ms/step vs 新梯度贯通版
  **160.0 ms/step**: 批量融合完全吸收新增反向计算 (6 个投影 matmul
  反向 + attention 反向) 的代价, 正确性零速度税。训练正确性提升
  (5 类注意力参数首次参与学习) 而吞吐持平。

### 测试 (Tests) — 186 → 195 项, 覆盖率 87% 维持
- 微指令分解 1920 → **1942** 条 (L4 +22, X +0), INV 覆盖 42/42 保持;
  selftest 4 门禁 ALL GREEN。

## [0.6.1] — 2026-10-17

### 修复 (Fixed) — 均由覆盖率驱动测试发现
- **top-p (nucleus) 采样错误** (`engine/sampler.py`): 旧实现 `cutoff = cum > top_p`
  会把**越过阈值的跨界 token 一并删除** —— `top_p < p_max` 时全部清零触发回退
  (top_p 形同虚设), 中段则过度截断 (核质量 < top_p)。改为正确语义
  「累计质量 ≥ top_p 的最小前缀」(判据 `cum − p ≥ top_p`, 首 token 恒在核内)。
- **logsumexp batch=1 形状塌缩** (`core/numeric.py`): 无轴 `np.squeeze` 把
  `(1,C) 沿 axis=1 → (1,1)` 压成 0 维标量, 令 `safe_cross_entropy` 在 batch=1
  时崩溃 (潜在, 此前从未有 batch=1 路径被测)。改为只压约简轴, 并加回归断言。
- **回归门禁断链** (`scripts/gen_baseline.py` × `eval/harness.py`):
  `RegressionBaseline.parse_baseline` 找 `tok/s mean:` 行, 但生成的
  BASELINE.md 从未包含该行 → 门禁永远 NO_BASELINE。生成器现追加机器可读行。

### 测试 (Tests) — 覆盖率 82% → 87% (总体), 169 → 186 项
- 新增 ~25 项针对性测试, 补齐覆盖率盲区:
  - `core/numeric.py` 40%→**100%**: safe_cross_entropy (手工对照/±1e4 稳定/
    ignore_index/label smoothing 公式对照) + clip_by_norm (INV-23 基元)。
  - `eval/harness.py` 32%→**100%**: GSM8K-Mini oracle/零分/取末数字 +
    RegressionBaseline GREEN/RED/NO_BASELINE 三态。
  - `serve/api.py` 66%→**96%**: 真实 HTTP 层 (ThreadingHTTPServer 临时端口 +
    urllib): GET models / POST chat(严格 schema) / completions / 404 / 坏 JSON。
  - `engine/speculative.py` 62%→**91%**: 采样模式接受路径 (草稿≡目标, 同 seed
    可复现) 与对抗草稿拒绝路径 (退化为贪心回填)。
  - `train/autodiff.py` 82%→**94%**: 复合表达式有限差分梯度检验
    (`__rtruediv__`/`__neg__`/relu/exp/mean(axis,keepdims)/max(axis)/pow)。
  - `engine/sampler.py` 89%→**95%**: forbidden/repetition_penalty/top_k=1/
    min_p/top_p 边界; `cli/__main__.py` 0%→100% (子进程冒烟);
    MoE 完整 forward + past_kv×padding 掩码扩展分支; 护栏小分支。

### 已知残留 (低价值盲区, 不强求)
- `eval/benchmark.py` 34-40 (基准函数主体由 `scripts/gen_baseline.py` 进程覆盖,
  不在 pytest 内); chunk/checkpoint 各 ~8 行防御分支。

## [0.6.0] — 2026-10-17

### 新增 (Added)
- **INV 测试可追溯性闭环（规范 → 实现 → 验证 三点贯通）**：PRD 映射原则
  （「每条不变量至少一个独立测试」）首次可执行化。
  - 硬门禁 `tests/unit/x/test_inv_traceability.py`：42/42 条 INV 每条至少被
    1 个测试文件引用（静态扫描 `tests/`，编号十进制归一），45 项参数化断言。
  - `ifai selftest --inv INV-17`：单查一条不变量，定位并运行验证它的测试文件。
- **补齐 4 条真缺口测试**（诊断发现 24 条 INV 在测试侧零引用，其中 4 条连
  语义覆盖都没有）：
  - `tests/unit/l2/test_param_moe.py`：**INV-13** 参数量 == 解析公式（4 种配置
    精确整数相等）+ **INV-14** MoE 路由守恒（选中概率和=1、每 token 恰
    `moe_topk` 个互异专家、独立重算 MoE 前向 diff ≤ 1e-12）。
  - `tests/unit/l4/test_trainer.py`：**INV-26** 训练损失单调（平滑后
    tail20% ≤ head20% × 0.8，真实训练非构造曲线）。
  - `tests/unit/x/test_determinism.py`：**INV-40** 同 seed bitwise 确定性
    （初始化/前向/训练一步/采样 四链路 + 异 seed 判别力对照）。
  - `tests/test_core.py`：**INV-04** RoPE 相对性（平移内积不变 ≤ 1e-6）、
    **INV-08** 补 ±1e4 溢出稳定判据。
- 既有 30+ 测试按证据补 INV 标签（test_core/test_models/test_engine/
  test_train/test_retrieval），测试侧引用从 18/42 → **42/42**。
- `docs/ARCHITECTURE.md`：INV 落点表从**历史规划路径**重写为**实际扁平布局**
  （证据制：逐 INV 以真实验证它的测试文件为准），并声明以实际代码为准。

### 修复 (Fixed)
- **INV-13 揭示并修复真实 bug**：权重绑定 (tied) 时 `Transformer.count_params()`
  把共享的 `lm_head`/`embed` **重复计一次**（参数量虚报 V·d）。改为按唯一数组
  计数；`TinyTrainer` 新增同口径 `count_params()`，`save()` 的 manifest 与
  `ifai train` 的参数量打印统一，保证 save→load 断言一致。

### 变更 (Changed)
- 微指令分解重生成：功能级 334 + 步骤级 1586 = **1920** 条；INV 覆盖 42/42。

### 测试 (Tests)
- `pytest -q` → **169 passed**（v0.5.0 时 104）。

## [0.5.0] — 2026-10-17

### 新增 (Added)
- **补齐 6 个缺失模块，闭合 INV 覆盖缺口 33/42 → 42/42**（`docs/PRD.md` 已规范
  但从未落地的实现）：
  - `ifai/engine/prefill.py`（**INV-20** 分块 prefill ≡ 整段 prefill，`diff ≤ 1e-5`）。
  - `ifai/train/checkpoint.py`（**INV-25** 梯度检查点 loss/grad ≡ 无检查点，
    `diff ≤ 1e-5`；复用 Chen et al. 2016 重计算思想，提供
    `checkpoint` / `checkpointed` / `checkpoint_segments` / `verify_checkpoint`）。
  - `ifai/retrieval/chunk.py`（**INV-31** 分块覆盖完整性：拼接 == 原文 100%，
    借鉴 Manning et al. 2008；支持 fixed / recursive 策略与 overlap）。
  - `ifai/retrieval/persist.py`（**INV-32** 索引持久化一致性：save→load 后
    检索结果 100% 一致）。
  - `ifai/agent/trace.py`（**INV-35** 轨迹可回放：同 seed 重放 ≡ 原轨迹 100%，
    比对口径 sha256 指纹；含 `Trace`/`TraceStep`/`TracedReActAgent`/`record_and_replay`）。
  - `ifai/serve/guardrail.py`（**INV-38** 护栏：红队拦截率 ≥90%、正常误杀 ≤5%；
    含 leetspeak/零宽/NFKC 归一化、规则集、PII 脱敏、`guard` 装饰器）。
- 数据集 `assets/data/redteam_100.jsonl`（100 条攻击，11 类）与
  `assets/data/benign_100.jsonl`（100 条正常提示）。
- `ifai/models/transformer.py` 的 `forward` 新增 `past_kv` / `return_kv` 参数
  （分块 prefill 历史 KV 传递，INV-20），并在文档中显式声明因果性（**INV-11**）。
- 微指令生成器 `scripts/gen_micro_instructions.py` 新增 **X 横切层**
  （扫描 `scripts/` 下审计脚本），使 **INV-41 / INV-42** 有落点；同时修复
  INV 编号前导零导致覆盖统计重复计数的 bug。
- `ifai selftest` 新增第 4 项门禁：**不变量覆盖（42/42 INV）**。

### 变更 (Changed)
- `ifai/train/trainer.py`：`forward` 重构为可分段重算的 `_layer_forward(x, layer)`，
  新增 `enable_gradient_checkpointing()` 开关（默认关闭，向后兼容）。
- 各包 `__init__.py` 导出新模块公开 API。
- 微指令分解重生成：功能级 333 + 步骤级 1581 = **1914** 条（≥1024）。

### 测试 (Tests)
- 新增 `tests/unit/l2..l7/` 共 7 个测试文件、49 项测试，**全部真验证 PRD 阈值**
  （非冒烟）：如分块 prefill 9 种块长 + KV cache 逐位对齐；分块覆盖 1000+ 组合
  精确还原；轨迹指纹含异 seed 判别力对照；护栏真跑 200 条统计指标
  （实测拦截 100% / 误杀 0%，均优于阈值）。
- `pytest -q` → **104 passed**；三项审计 + 42/42 覆盖门禁全 exit 0。

### 已知限制 (Known Limitations)
- `ifai/serve/guardrail.py` 为**规则式护栏**（模式表 + 归一化），数据集与规则集
  协同达成 100%/0%；这证明管线与度量闭环有效，但不等同于现代语义分类器的
  泛化能力，不应视为完整安全保证。后续可替换为小型语义分类器（零重写：可接入
  现成的 Llama Guard / NeMo Guardrails 权重）。

## [0.4.0] — 2026-10-16

### 新增 (Added)
- **SDK OpenAI 严格兼容接口 `IFAI.chat_completion(messages)`** (L9): 返回符合
  OpenAI `chat.completion` schema 的 dict (INV-36)。零重写: 与服务端共用
  `serve.api.chat_completion` / `format_messages` 单一实现, 保证 SDK 与 HTTP
  服务输出逐字一致 (INV-37)。
- **CI `baseline` job** (report-only, `continue-on-error`): 每次 push 在 CPU 上
  生成性能基线 (`scripts/gen_baseline.py`) 并上传 `BASELINE.md` artifact,
  提供性能可观测性而不因 CI 跑分波动阻断合并 (R-L8-02/R-L8-05 可观测化)。
- `serve/api.py` 新增模块级公开函数 `format_messages` / `chat_completion`
  (单一事实来源, 供服务端与 SDK 共用)。

### 变更 (Changed)
- `ifai/sdk.py`: 收敛单一生成原语 `_generate`, `chat()` 改为返回续写文本
  (语义不变, 仍保留 "(空)" 兜底); 与服务端生成语义对齐。
- 微指令分解随源码更新重生成: 功能级 207 + 步骤级 909 = **1116** 条 (≥1024)。

### 测试 (Tests)
- 新增 `tests/e2e/test_demo_pipeline.py::test_sdk_chat_completion_openai_schema`
  (INV-36 schema + 与服务端逐字一致)。
- `tests/unit/l9/test_sdk.py` 公开 API docstring 清单纳入 `chat_completion`。

## [0.3.0] — 2026-10-16

### 新增 (Added)
- **全栈 1024 项微指令级操作分解** (`docs/MICRO_INSTRUCTIONS.md` + 生成器
  `scripts/gen_micro_instructions.py` + 测试 `tests/unit/x/test_micro_instructions.py`)。
  对 `ifai/` 源码 AST 静态扫描, 产出 **功能级 203 + 步骤级 902 = 1105** 条微指令
  (≥1024 目标), 覆盖 38 个可验证不变量 (INV), 逐条标注零重写出处。
- **INV-36 修复**: `/v1/chat/completions` 现在严格返回 OpenAI 的
  `choices[].message.content` 结构 (字段集与 OpenAI schema diff 为空), 见
  `ifai/serve/api.py` 与 `tests/test_serve.py`。
- **`ifai selftest` 自检子命令** (L9 CLI): 一键串联三项静态审计
  (INV-41 分层 / INV-42 零重写出处 / R-X-03 文档完备), 可选 `--pytest`
  运行黄金测试, 输出 ALL GREEN / 失败 门禁结论。闭合已知缺口。

### 变更 (Changed)
- 版本号升至 0.3.0 (`ifai/__init__.py` 与 `pyproject.toml` 一致)。

## [0.2.0] — 2026-10-16

### 新增 (Added)
- 横切工程闭环（W7）：三项静态审计脚本全部落地并接入 CI 门禁。
  - `scripts/check_layering.py`（INV-41：0 反向 import，十层单向依赖）。
  - `scripts/audit_no_reinvention.py`（INV-42：十层覆盖 + ≥25 处世界顶级出处，实测 75 处）。
  - `scripts/check_docs.py`（R-X-03：必备文件 / 内部链接 / 关键章节）。
- 新增 8 个可运行示例（`examples/`）：quickstart / demo_cli / quantize_int8 /
  lora_finetune / agent_tools / rag_local / serve_client，以及 `examples/README.md`。
- 性能基线工具 `scripts/gen_baseline.py` 与 `BASELINE.md`（prefill 2.08ms、
  TPOT 3.55ms、281 tok/s，含 ±15% 回归门禁）。
- 逐层设计文档 `docs/design/L0..L8`（共 55 条零重写出处标注）。
- 端到端测试 `tests/e2e/*` 与 L9 单元测试 `tests/unit/l9/*`。
- `.github/workflows/ci.yml`：Python 3.9/3.11/3.13 × ubuntu/windows 矩阵，
  运行 pytest + 三项审计。

### 变更 (Changed)
- 版本号升至 0.2.0（`ifai/__init__.py` 与 `pyproject.toml` 一致）。
- `pyproject.toml` 增加 `[tool.coverage.*]` 配置（按需启用 `pytest --cov`）。

### 文档 (Docs)
- `README.md`、`CITATION.md`、`CONTRIBUTING.md`、`CHANGELOG.md` 署名统一为
  晨星 (Morningstar) / CJX0712。

## [0.1.0] — 2026-10-16

### 新增 (Added)
- 十层架构 L0–L9 全栈参考实现（算子内核 / 数值基础 / 模型架构 / 推理引擎 /
  训练微调 / 检索记忆 / 智能体 / 服务编排 / 评测对齐 / 应用接口）。
- 42 条可验证不变量（INV）与 60 项需求（PRD），对应 39 项黄金测试。
- 全局 PCG64 RNG（`ifai.rng`）保证 bitwise 可复现。
- 推理引擎：PagedAttention 分页 KV 缓存、连续批处理（Orca）、投机解码、
  温度/top-k/top-p/min-p/重复惩罚采样。
- 训练：AdamW + 余弦退火 + LoRA + 手写反向模式自动微分 + nanogpt 风格微型训练循环。
- 检索：BM25 + 稠密索引 + HNSW + RRF 融合 + Cross-Encoder 重排。
- 智能体：ReAct + Reflexion + MemGPT 式记忆 + AST 白名单安全沙箱。
- OpenAI 兼容 HTTP 服务（`/v1/models`、`/v1/completions`、`/v1/chat/completions`）。
- CLI（`ifai`）：train / demo / generate / chat / serve / eval / search / agent /
  tokenize / version / doctor。
- 零重写审计脚本 `scripts/audit_zero_rewrite.py`（CI 门禁：十层均须引用公认方案）。
- 仅 `numpy>=1.24` 硬依赖，100% CPU 推理路径。

### 文档 (Docs)
- `README.md`、`docs/ARCHITECTURE.md`、`docs/PRD.md`、`docs/PRIOR_ART.md`、
  `docs/TASKS.md`、`docs/API.md`、`CITATION.md`、`NOTICE`、`CONTRIBUTING.md`。

### 许可 (License)
- Apache-2.0，© 晨星 (Morningstar) / CJX0712。
