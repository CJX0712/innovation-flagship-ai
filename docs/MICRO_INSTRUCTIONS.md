# 微指令级操作分解 (Micro-Instruction Decomposition)

> 由 `scripts/gen_micro_instructions.py` 对 `ifai/` (L0–L9) 与 `scripts/` (X 横切层) 源码 AST 静态扫描自动生成。
> 零重写 · 集大成: 每条微指令标注其复用的世界顶级来源与所服务的可验证不变量 (INV)。

**总计**: 功能级微指令 469 条 + 步骤级微指令 2221 条 = **2690** 条; 覆盖 INV [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]。

## 分层汇总

| 层 | 说明 | 功能级 | 步骤级 | 合计 |
| --- | --- | ---: | ---: | ---: |
| L0 | L0 算子/内核 (Operator/Kernel) | 22 | 135 | 157 |
| L1 | L1 张量/数值基础 (Tensor/Numeric) | 4 | 9 | 13 |
| L2 | L2 模型架构 (Model Architecture) | 36 | 202 | 238 |
| L3 | L3 推理引擎 (Inference Engine) | 42 | 175 | 217 |
| L4 | L4 训练/微调 (Training/Finetune) | 72 | 405 | 477 |
| L5 | L5 检索/记忆 (Retrieval/Memory) | 70 | 335 | 405 |
| L6 | L6 智能体 (Agent) | 57 | 190 | 247 |
| L7 | L7 服务/编排 (Service/Orchestration) | 42 | 151 | 193 |
| L8 | L8 评估/对齐 (Evaluation/Alignment) | 10 | 40 | 50 |
| L9 | L9 应用/接口 (Application/Interface) | 7 | 23 | 30 |
| L10 | L10 微指令 ISA (Micro-Instruction ISA) | 80 | 302 | 382 |
| X | X 横切审计 (Cross-cutting Audits) | 27 | 254 | 281 |

## L0 算子/内核 (Operator/Kernel)

### MI-0001 logsumexp(x, axis, keepdims)
- 模块: `core\numeric.py`
- 用途: 数值稳定的 log(sum(exp(x)))。  ·  INV 8
- 微指令步骤 (6):
  - 赋值 x ← np.asarray(…)
  - 赋值 xmax ← np.max(…)
  - 赋值 xmax_safe ← np.where(…)
  - 赋值 out ← xmax_safe add np.log(…)
  - 判断 UnaryOp
  - 返回 out.astype(…)

### MI-0002 stable_softmax(logits, axis)
- 模块: `core\numeric.py`
- 用途: 平移不变的 softmax (INV-09)。  ·  INV 9
- 微指令步骤 (4):
  - 赋值 logits ← np.asarray(…)
  - 赋值 z ← logits sub np.max(…)
  - 赋值 e ← np.exp(…)
  - 返回 e div np.sum(…)

### MI-0003 safe_cross_entropy(logits, targets, ignore_index, label_smoothing)
- 模块: `core\numeric.py`
- 用途: 返回 (loss 标量, 对 logits 的梯度)。  ·  INV 8
- 微指令步骤 (17):
  - 赋值 logits ← np.asarray(…)
  - 赋值 targets ← np.asarray(…)
  - 判断 Compare
  - 赋值 (N, C) ← logits.shape
  - 赋值 lse ← logsumexp(…)
  - 赋值 log_probs ← logits sub lse[…]
  - 赋值 mask ← Compare
  - 赋值 rows ← np.where(…)
  - 赋值 q ← np.zeros(…)
  - 赋值 q[…] ← 1.0
  - 判断 Compare
  - 赋值 ce ← UnaryOp
  - 赋值 loss ← np.sum(…) div max(…)
  - 赋值 p ← np.exp(…)
  - 赋值 grad ← p sub q div max(…)
  - 赋值 grad ← grad mult mask[…]
  - 返回 [float(…), grad]

### MI-0004 clip_by_norm(grad, max_norm, norm_type)
- 模块: `core\numeric.py`
- 用途: 全局范数裁剪，返回 (裁剪后梯度, 是否触发裁剪)。INV-23 由 train 层实现全局版。  ·  INV 23
- 微指令步骤 (4):
  - 赋值 g ← np.asarray(…)
  - 赋值 norm ← float(…)
  - 判断 Compare
  - 返回 [g, False]

### MI-0005 rms_norm(x, weight, eps)
- 模块: `core\ops.py`
- 用途: RMSNorm: x / sqrt(mean(x^2)+eps) * weight。INV-06 范数归一。  ·  INV 6  ·  复用 RMSNorm
- 微指令步骤 (3):
  - 赋值 x ← np.asarray(…)
  - 赋值 ms ← np.mean(…)
  - 返回 x div np.sqrt(…) mult np.asarray(…)

### MI-0006 _rope_freqs(dim, base, scaling)
- 模块: `core\ops.py`
- 用途: 生成旋转频率 (NTK-aware: 当 scaling≠1 拉伸 base)。  ·  INV 1,2,3,4,5,6,7  ·  复用 NTK
- 微指令步骤 (4):
  - 赋值 half ← dim floordiv 2
  - 赋值 inv ← 1.0 div base pow np.arange(…) div half
  - 判断 Compare
  - 返回 inv

### MI-0007 apply_rope(x, positions, dim, base, scaling)
- 模块: `core\ops.py`
- 用途: 对最后一维 (隐维) 施加旋转位置编码。  ·  INV 16,3
- 微指令步骤 (18):
  - 赋值 x ← np.asarray(…)
  - 判断 Compare
  - 赋值 T ← x.shape[…]
  - 赋值 half ← dim floordiv 2
  - 赋值 freqs ← _rope_freqs(…)
  - 赋值 pos ← np.asarray(…)
  - 判断 Compare
  - 赋值 cos ← np.cos(…)
  - 赋值 sin ← np.sin(…)
  - 判断 Compare
  - 赋值 x1 ← x[…]
  - 赋值 x2 ← x[…]
  - 赋值 rot1 ← x1 mult cos sub x2 mult sin
  - 赋值 rot2 ← x1 mult sin add x2 mult cos
  - 赋值 out ← np.empty_like(…)
  - 赋值 out[…] ← rot1
  - 赋值 out[…] ← rot2
  - 返回 out

### MI-0008 swiglu(x, w1, w2, b1, b2)
- 模块: `core\ops.py`
- 用途: SwiGLU(x) = (x·W1 + b1) ⊙ silu(x·W2 + b2) · W? — 这里实现 SwiGLU FFN 内核。  ·  INV 5  ·  复用 SwiGLU
- 微指令步骤 (6):
  - 赋值 x ← np.asarray(…)
  - 赋值 (w1, w2) ← [np.asarray(…), np.asarray(…)]
  - 赋值 g ← IfExp
  - 赋值 u ← IfExp
  - 赋值 silu ← g div 1.0 add np.exp(…)
  - 返回 silu mult u

### MI-0009 gelu(x)
- 模块: `core\ops.py`
- 用途: GELU (Hendrycks & Gimpel 2016, tanh-approx, GPT/LLaMA 同款)。  ·  INV 1,2,3,4,5,6,7  ·  复用 SwiGLU
- 微指令步骤 (3):
  - 赋值 x ← np.asarray(…)
  - 赋值 c ← np.sqrt(…)
  - 返回 0.5 mult x mult 1.0 add np.tanh(…)

### MI-0010 tiled_matmul(a, b, block)
- 模块: `core\ops.py`
- 用途: 分块矩阵乘 (INV-01 == a@b)。分块提升 cache 命中，复刻 BLAS 思想。  ·  INV 1
- 微指令步骤 (7):
  - 赋值 a ← np.asarray(…)
  - 赋值 b ← np.asarray(…)
  - 赋值 (m, k) ← a.shape
  - 赋值 n ← b.shape[…]
  - 赋值 out ← np.zeros(…)
  - 遍历 i ∈ range(…)
  - 返回 out

### MI-0011 attention_flash(q, k, v, scale, causal)
- 模块: `core\ops.py`
- 用途: 在线 softmax 的因果注意力 (INV-02 == 朴素)。  ·  INV 2  ·  复用 Dao
- 微指令步骤 (8):
  - 赋值 q ← np.asarray(…)
  - 赋值 k ← np.asarray(…)
  - 赋值 v ← np.asarray(…)
  - 赋值 (Sq, d) ← q.shape
  - 赋值 Sk ← k.shape[…]
  - 赋值 out ← np.zeros(…)
  - 遍历 i ∈ range(…)
  - 返回 out

### MI-0012 attention_naive(q, k, v, scale, causal)
- 模块: `core\ops.py`
- 用途: (无 docstring)  ·  INV 1,2,3,4,5,6,7
- 微指令步骤 (7):
  - 赋值 q ← np.asarray(…)
  - 赋值 k ← np.asarray(…)
  - 赋值 v ← np.asarray(…)
  - 赋值 scores ← q matmult k.T mult scale
  - 判断 causal
  - 赋值 p ← numeric.stable_softmax(…)
  - 返回 p matmult v

### MI-0013 l2norm(x)
- 模块: `core\ops.py`
- 用途: 沿最后一维做 L2 归一化 → **严格**单位范数; 零向量行为 0。  ·  INV 1,2,3,4,5,6,7
- 微指令步骤 (3):
  - 赋值 x ← np.asarray(…)
  - 赋值 n ← np.linalg.norm(…)
  - 返回 np.where(…)

### MI-0014 spectral_norm(A, iters, eps)
- 模块: `core\ops.py`
- 用途: 幂迭代求谱范数 (最大奇异值)。  ·  INV 1,2,3,4,5,6,7
- 微指令步骤 (6):
  - 赋值 A ← np.asarray(…)
  - 判断 Compare
  - 赋值 (_, n) ← A.shape
  - 赋值 b ← np.ones(…) div max(…) pow 0.5
  - 遍历 _ ∈ range(…)
  - 返回 float(…)

### MI-0015 newton_schulz(G, steps, eps, coeffs)
- 模块: `core\ops.py`
- 用途: Newton-Schulz 正交化: 把矩阵推向最近的半正交矩阵 UVᵀ。  ·  INV 1,2,3,4,5,6,7  ·  复用 nanogpt
- 微指令步骤 (10):
  - 赋值 G ← np.asarray(…)
  - 判断 Compare
  - 赋值 (a, b, c) ← coeffs
  - 赋值 X ← G.copy(…)
  - 赋值 transposed ← Compare
  - 判断 transposed
  - 赋值 X ← X div spectral_norm(…) add eps
  - 遍历 _ ∈ range(…)
  - 判断 transposed
  - 返回 X

### MI-0016 quantize_int8_per_channel(x, axis)
- 模块: `core\quant.py`
- 用途: 对称 per-channel RTN 量化到 INT8。  ·  INV 3,7  ·  复用 int8
- 微指令步骤 (5):
  - 赋值 x ← np.asarray(…)
  - 赋值 max_abs ← np.max(…)
  - 赋值 scale ← np.where(…) div 127.0
  - 赋值 q ← Call.astype(…)
  - 返回 [q, scale.astype(…)]

### MI-0017 dequantize_int8(q, scale)
- 模块: `core\quant.py`
- 用途: (无 docstring)  ·  INV 3,7
- 微指令步骤 (1):
  - 返回 np.asarray(…) mult np.asarray(…)

### MI-0018 quantize_int4_per_channel(x, axis)
- 模块: `core\quant.py`
- 用途: 对称 per-channel RTN 量化到 INT4 (范围 [-7, 7], 15 电平)。  ·  INV 7  ·  复用 int4, int8
- 微指令步骤 (5):
  - 赋值 x ← np.asarray(…)
  - 赋值 max_abs ← np.max(…)
  - 赋值 scale ← np.where(…) div 7.0
  - 赋值 q ← Call.astype(…)
  - 返回 [q, scale.astype(…)]

### MI-0019 dequantize_int4(q, scale)
- 模块: `core\quant.py`
- 用途: (无 docstring)  ·  INV 3,7
- 微指令步骤 (1):
  - 返回 np.asarray(…) mult np.asarray(…)

### MI-0020 pack_int4(x)
- 模块: `core\quant.py`
- 用途: 把 int8 范围 [-8,7] 的数组打包成 uint8 (每字节两个数, 低4位先)。  ·  INV 3,7  ·  复用 int8
- 微指令步骤 (7):
  - 赋值 x ← Call.ravel(…)
  - 赋值 x ← Call.astype(…)
  - 赋值 pad ← len(…) add 1 floordiv 2 mult 2 sub len(…)
  - 判断 pad
  - 赋值 lo ← x[…].astype(…) bitand 15
  - 赋值 hi ← x[…].astype(…) bitand 15
  - 返回 BinOp.astype(…)

### MI-0021 unpack_int4(packed, n)
- 模块: `core\quant.py`
- 用途: 解包 int4 (符号扩展回 int8 范围 [-8,7])。  ·  INV 3,7  ·  复用 int4, int8
- 微指令步骤 (8):
  - 赋值 p ← np.asarray(…)
  - 赋值 lo ← BinOp.astype(…)
  - 赋值 hi ← BinOp.astype(…)
  - 赋值 x ← np.empty(…)
  - 赋值 x[…] ← lo
  - 赋值 x[…] ← hi
  - 赋值 x ← Call.astype(…)
  - 返回 IfExp

### MI-0022 quantize_fp16(x)
- 模块: `core\quant.py`
- 用途: 模拟 fp16 往返 (numpy 无纯 fp16 计算，这里用 float16 dtype 模拟量化误差)。  ·  INV 3,7
- 微指令步骤 (2):
  - 赋值 x ← np.asarray(…)
  - 返回 Call.astype(…)

## L1 张量/数值基础 (Tensor/Numeric)

### MI-0023 seed(value)
- 模块: `rng.py`
- 用途: 设置全局随机种子并复位所有派生源。  ·  INV 40
- 微指令步骤 (3):
  - Global
  - 赋值 _SEED ← int(…)
  - 赋值 _RNG ← np.random.default_rng(…)

### MI-0024 get_rng()
- 模块: `rng.py`
- 用途: 返回全局 ``Generator``；未初始化时默认 seed(0)。  ·  INV 40
- 微指令步骤 (4):
  - Global
  - 判断 Compare
  - Assert
  - 返回 _RNG

### MI-0025 get_seed()
- 模块: `rng.py`
- 用途: 返回当前种子 (None 表示未设置)。  ·  INV 40
- 微指令步骤 (1):
  - 返回 _SEED

### MI-0026 spawn(child_seed)
- 模块: `rng.py`
- 用途: 派生一个独立的子随机源 (用于并行/隔离场景)，不影响全局源。  ·  INV 40
- 微指令步骤 (1):
  - 返回 np.random.default_rng(…)

## L2 模型架构 (Model Architecture)

### MI-0027 【类】ModelConfig.ModelConfig(类)
- 模块: `models\config.py`
- 用途: Transformer 解码器配置。所有字段可 JSON 往返。  ·  INV 13
- 微指令步骤 (27):
  - 声明并赋值 vocab_size ← 32000
  - 声明并赋值 d_model ← 512
  - 声明并赋值 n_layers ← 6
  - 声明并赋值 n_heads ← 8
  - 声明并赋值 n_kv_heads ← None
  - 声明并赋值 ffn_mult ← 4
  - 声明并赋值 rope_base ← 10000.0
  - 声明并赋值 rope_scaling ← 1.0
  - 声明并赋值 norm_eps ← 1e-06
  - 声明并赋值 use_moe ← False
  - 声明并赋值 n_experts ← 0
  - 声明并赋值 moe_topk ← 1
  - 声明并赋值 tie_word_embeddings ← True
  - 声明并赋值 max_seq_len ← 1024
  - 声明并赋值 activation ← "swiglu"
  - 声明并赋值 use_lora ← False
  - 声明并赋值 lora_r ← 4
  - 声明并赋值 lora_alpha ← 8
  - 声明并赋值 use_qat ← False
  - 声明并赋值 initializer_range ← 0.02
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0028 ModelConfig.__post_init__(self)
- 模块: `models\config.py`
- 用途: (无 docstring)  ·  INV 13
- 微指令步骤 (4):
  - 判断 Compare
  - Assert
  - 判断 self.use_moe
  - 判断 self.use_lora

### MI-0029 ModelConfig.d_head(self)
- 模块: `models\config.py`
- 用途: (无 docstring)  ·  INV 13
- 装饰器: property
- 微指令步骤 (1):
  - 返回 self.d_model floordiv self.n_heads

### MI-0030 ModelConfig.d_ff(self)
- 模块: `models\config.py`
- 用途: (无 docstring)  ·  INV 13
- 装饰器: property
- 微指令步骤 (1):
  - 返回 self.d_model mult self.ffn_mult

### MI-0031 ModelConfig.n_repeats(self)
- 模块: `models\config.py`
- 用途: (无 docstring)  ·  INV 13
- 装饰器: property
- 微指令步骤 (1):
  - 返回 self.n_heads floordiv self.n_kv_heads

### MI-0032 ModelConfig.to_json(self, path)
- 模块: `models\config.py`
- 用途: (无 docstring)  ·  INV 13
- 微指令步骤 (3):
  - 赋值 s ← json.dumps(…)
  - 判断 path
  - 返回 s

### MI-0033 ModelConfig.from_json(cls, source)
- 模块: `models\config.py`
- 用途: (无 docstring)  ·  INV 13
- 装饰器: classmethod
- 微指令步骤 (4):
  - 判断 isinstance(…)
  - 赋值 allowed ← SetComp
  - 赋值 obj ← DictComp
  - 返回 cls(…)

### MI-0034 ModelConfig.clone(self)
- 模块: `models\config.py`
- 用途: (无 docstring)  ·  INV 13
- 微指令步骤 (1):
  - 返回 ModelConfig.from_json(…)

### MI-0035 【类】Tokenizer.Tokenizer(类)
- 模块: `models\tokenizer.py`
- 用途: (无 docstring)  ·  INV 40
- 微指令步骤 (8):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0036 Tokenizer.__init__(self, vocab_size, specials)
- 模块: `models\tokenizer.py`
- 用途: (无 docstring)  ·  INV 40
- 微指令步骤 (6):
  - 赋值 self.vocab_size_target ← vocab_size
  - 赋值 self.specials ← BoolOp add ListComp
  - 声明并赋值 self.merges ← []
  - 声明并赋值 self.merge_ranks ← Dict
  - 声明并赋值 self.token2id ← Dict
  - 声明并赋值 self.id2token ← Dict

### MI-0037 Tokenizer.train(self, corpus, min_freq)
- 模块: `models\tokenizer.py`
- 用途: (无 docstring)  ·  INV 40
- 微指令步骤 (10):
  - 赋值 chars ← sorted(…)
  - 赋值 symbols ← list(…) add chars
  - 赋值 self.token2id ← DictComp
  - 赋值 self.id2token ← DictComp
  - 赋值 words ← re.findall(…)
  - FunctionDef
  - 赋值 word_syms ← ListComp
  - 赋值 max_merges ← max(…)
  - 遍历 _ ∈ range(…)
  - 返回 self

### MI-0038 Tokenizer._apply_merges(self, syms)
- 模块: `models\tokenizer.py`
- 用途: (无 docstring)  ·  INV 40
- 微指令步骤 (2):
  - 当 True 循环
  - 返回 syms

### MI-0039 Tokenizer.encode(self, text)
- 模块: `models\tokenizer.py`
- 用途: (无 docstring)  ·  INV 40
- 微指令步骤 (3):
  - 声明并赋值 ids ← []
  - 遍历 w ∈ re.findall(…)
  - 返回 ids

### MI-0040 Tokenizer.decode(self, ids)
- 模块: `models\tokenizer.py`
- 用途: (无 docstring)  ·  INV 40
- 微指令步骤 (1):
  - 返回 Constant.join(…)

### MI-0041 Tokenizer.vocab_size(self)
- 模块: `models\tokenizer.py`
- 用途: (无 docstring)  ·  INV 40
- 装饰器: property
- 微指令步骤 (1):
  - 返回 len(…)

### MI-0042 Tokenizer.to_json(self, path)
- 模块: `models\tokenizer.py`
- 用途: (无 docstring)  ·  INV 40
- 微指令步骤 (4):
  - 赋值 obj ← Dict
  - 赋值 s ← json.dumps(…)
  - 判断 path
  - 返回 s

### MI-0043 Tokenizer.from_json(cls, source)
- 模块: `models\tokenizer.py`
- 用途: (无 docstring)  ·  INV 40
- 装饰器: classmethod
- 微指令步骤 (7):
  - 赋值 obj ← IfExp
  - 赋值 t ← cls(…)
  - 赋值 t.merges ← ListComp
  - 赋值 t.merge_ranks ← DictComp
  - 赋值 t.token2id ← DictComp
  - 赋值 t.id2token ← DictComp
  - 返回 t

### MI-0044 【类】Transformer.Transformer(类)
- 模块: `models\transformer.py`
- 用途: (无 docstring)  ·  INV 10,11,12,13,14,20
- 微指令步骤 (15):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0045 Transformer.__init__(self, config)
- 模块: `models\transformer.py`
- 用途: (无 docstring)  ·  INV 10,11,12,13,14,20
- 微指令步骤 (2):
  - 赋值 self.cfg ← config
  - 调用 self._init_params(…)

### MI-0046 Transformer._normal(self, *shape)
- 模块: `models\transformer.py`
- 用途: (无 docstring)  ·  INV 10,11,12,13,14,20
- 微指令步骤 (2):
  - 赋值 r ← rng.get_rng(…)
  - 返回 BinOp.astype(…)

### MI-0047 Transformer._init_params(self)
- 模块: `models\transformer.py`
- 用途: (无 docstring)  ·  INV 10,11,12,13,14,20
- 微指令步骤 (10):
  - 赋值 c ← self.cfg
  - 赋值 (V, d) ← [c.vocab_size, c.d_model]
  - 声明并赋值 p ← Dict
  - 赋值 p[…] ← self._normal(…)
  - 判断 c.tie_word_embeddings
  - 赋值 p[…] ← np.ones(…)
  - 赋值 n_kv ← c.n_kv_heads
  - 遍历 l ∈ range(…)
  - 赋值 self.params ← p
  - 声明并赋值 self._cache ← Dict

### MI-0048 Transformer.tie_word_embeddings(self)
- 模块: `models\transformer.py`
- 用途: (无 docstring)  ·  INV 10,11,12,13,14,20
- 装饰器: property
- 微指令步骤 (1):
  - 返回 self.cfg.tie_word_embeddings

### MI-0049 Transformer.count_params(self)
- 模块: `models\transformer.py`
- 用途: 可训练参数量 (INV-13: 与按配置解析计算值精确相等)。  ·  INV 12,13
- 微指令步骤 (3):
  - 声明并赋值 seen ← Dict
  - 遍历 v ∈ self.params.values(…)
  - 返回 int(…)

### MI-0050 Transformer.forward(self, token_ids, return_hidden, position_ids, key_padding_mask, past_kv, return_kv)
- 模块: `models\transformer.py`
- 用途: 前向。  ·  INV 11,16,20
- 微指令步骤 (14):
  - 赋值 c ← self.cfg
  - 赋值 ids ← np.asarray(…)
  - 判断 Compare
  - 赋值 (B, T) ← ids.shape
  - 赋值 (d, dh, n_h, n_kv) ← [c.d_model, c.d_head, c.n_heads]
  - 赋值 scale ← 1.0 div dh pow 0.5
  - 判断 Compare
  - 赋值 x ← self.params[…][…]
  - 声明并赋值 present ← []
  - 遍历 l ∈ range(…)
  - 赋值 x ← ops.rms_norm(…)
  - 赋值 out ← IfExp
  - 判断 return_kv
  - 返回 out

### MI-0051 Transformer._allow_mask(positions, B, T, T_k)
- 模块: `models\transformer.py`
- 用途: 构造 ``[B|1, 1, T, T_k]`` 的可见性掩码: key 绝对位置 <= query 绝对位置。  ·  INV 11
- 装饰器: staticmethod
- 微指令步骤 (7):
  - 赋值 pos ← np.asarray(…)
  - 赋值 P ← T_k sub T
  - 判断 Compare
  - 赋值 allow_cur ← Compare
  - 赋值 allow_past ← IfExp
  - 赋值 allow ← np.concatenate(…)
  - 返回 allow[…]

### MI-0052 Transformer._moe(self, h2, pre, c)
- 模块: `models\transformer.py`
- 用途: (无 docstring)  ·  INV 10,11,12,13,14,20
- 微指令步骤 (8):
  - 赋值 (B, T, d) ← h2.shape
  - 赋值 router_logits ← h2 matmult self.params[…]
  - 赋值 topk_idx ← np.argsort(…)[…]
  - 赋值 sel ← np.take_along_axis(…)
  - 赋值 w ← numeric.stable_softmax(…)
  - 赋值 out ← np.zeros(…)
  - 遍历 e ∈ range(…)
  - 返回 out

### MI-0053 Transformer._apply_lora(self, x, pre, tgt, base_out)
- 模块: `models\transformer.py`
- 用途: 推理端 LoRA 增量 (numpy 同公式, Hu et al. 2022, INV-24):  ·  INV 24  ·  复用 Hu et al, LoRA
- 微指令步骤 (7):
  - 赋值 c ← self.cfg
  - 判断 UnaryOp
  - 赋值 A ← self.params[…]
  - 赋值 B ← self.params[…]
  - 赋值 scale ← c.lora_alpha div c.lora_r
  - 赋值 lora ← x matmult A.T matmult B.T mult scale
  - 返回 base_out add lora

### MI-0054 Transformer.logits_next(self, token_ids)
- 模块: `models\transformer.py`
- 用途: 给定已生成序列, 返回最后一个位置的 logits [V]。  ·  INV 10,11,12,13,14,20
- 微指令步骤 (3):
  - 赋值 logits ← self.forward(…)
  - 赋值 last ← logits[…]
  - 返回 last

### MI-0055 Transformer.save(self, dir_path)
- 模块: `models\transformer.py`
- 用途: (无 docstring)  ·  INV 10,11,12,13,14,20
- 微指令步骤 (7):
  - 调用 os.makedirs(…)
  - 调用 np.savez(…)
  - 赋值 manifest ← Dict
  - 上下文管理 (with)
  - 赋值 manifest[…][…] ← Call.hexdigest(…)
  - 上下文管理 (with)
  - 返回 dir_path

### MI-0056 Transformer.load(cls, dir_path, verify)
- 模块: `models\transformer.py`
- 用途: (无 docstring)  ·  INV 10,11,12,13,14,20
- 装饰器: classmethod
- 微指令步骤 (12):
  - 上下文管理 (with)
  - 赋值 cfg ← ModelConfig.from_json(…)
  - 上下文管理 (with)
  - 判断 verify
  - 赋值 data ← np.load(…)
  - 赋值 model ← cls.__new__(…)
  - 赋值 model.cfg ← cfg
  - 赋值 model.params ← DictComp
  - 赋值 model._cache ← Dict
  - 判断 cfg.tie_word_embeddings
  - Assert
  - 返回 model

### MI-0057 Transformer.quantize_int8(self, axis)
- 模块: `models\transformer.py`
- 用途: 对全部线性权重做 per-channel INT8 RTN 量化 (Dettmers et al. 2022  ·  INV 12,7  ·  复用 RMSNorm, dettmers, int8
- 微指令步骤 (4):
  - 声明并赋值 info ← Dict
  - 遍历 k ∈ ListComp
  - 判断 self.cfg.tie_word_embeddings
  - 返回 info

### MI-0058 Transformer.quantize_fp16(self)
- 模块: `models\transformer.py`
- 用途: 对全部线性权重做 FP16 往返量化 (IEEE 754 半精度模拟), 与  ·  INV 10,11,12,13,14,20  ·  复用 int4, int8
- 微指令步骤 (4):
  - 声明并赋值 info ← Dict
  - 遍历 k ∈ ListComp
  - 判断 self.cfg.tie_word_embeddings
  - 返回 info

### MI-0059 Transformer.quantize_int4(self, axis)
- 模块: `models\transformer.py`
- 用途: 对全部线性权重做 per-channel INT4 RTN 量化 (对称 [-7,7], 15 电平),  ·  INV 7  ·  复用 awq, gptq, int4, int8
- 微指令步骤 (4):
  - 声明并赋值 info ← Dict
  - 遍历 k ∈ ListComp
  - 判断 self.cfg.tie_word_embeddings
  - 返回 info

### MI-0060 verify_causal_mask(model, token_ids, position)
- 模块: `models\transformer.py`
- 用途: INV-11 自检: 扰动 ``position`` 之后的 token, 之前位置的输出必须 bitwise 不变。  ·  INV 11  ·  复用 Radford, Vaswani
- 微指令步骤 (13):
  - 赋值 ids ← np.asarray(…)
  - 判断 Compare
  - 赋值 T ← ids.shape[…]
  - 赋值 pos ← IfExp
  - Assert
  - 赋值 base ← np.asarray(…)
  - 赋值 vocab ← int(…)
  - 赋值 perturbed ← ids.copy(…)
  - 遍历 j ∈ range(…)
  - 赋值 after ← np.asarray(…)
  - 赋值 prefix_equal ← bool(…)
  - 赋值 n_changed ← int(…)
  - 返回 [prefix_equal, n_changed]

### MI-0061 save_model(model, dir_path)
- 模块: `models\transformer.py`
- 用途: (无 docstring)  ·  INV 10,11,12,13,14,20
- 微指令步骤 (1):
  - 返回 model.save(…)

### MI-0062 load_model(dir_path, verify)
- 模块: `models\transformer.py`
- 用途: (无 docstring)  ·  INV 10,11,12,13,14,20
- 微指令步骤 (1):
  - 返回 Transformer.load(…)

## L3 推理引擎 (Inference Engine)

### MI-0063 【类】PagedKVCache.PagedKVCache(类)
- 模块: `engine\kvcache.py`
- 用途: 按序列分块存储 K/V。K 与 V 各用独立 block 缓冲。  ·  INV 15
- 微指令步骤 (6):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0064 PagedKVCache.__init__(self, block_size, kv_dim, max_blocks)
- 模块: `engine\kvcache.py`
- 用途: (无 docstring)  ·  INV 15
- 微指令步骤 (9):
  - 赋值 self.block_size ← block_size
  - 赋值 self.kv_dim ← kv_dim
  - 赋值 self.max_blocks ← max_blocks
  - 赋值 self.k_blocks ← np.zeros(…)
  - 赋值 self.v_blocks ← np.zeros(…)
  - 声明并赋值 self.block_table ← Dict
  - 声明并赋值 self.seq_len ← Dict
  - 赋值 self._free ← list(…)
  - 赋值 self._next_seq ← 0

### MI-0065 PagedKVCache.new_sequence(self)
- 模块: `engine\kvcache.py`
- 用途: (无 docstring)  ·  INV 15
- 微指令步骤 (5):
  - 赋值 sid ← self._next_seq
  - 累算 self._next_seq add 1
  - 赋值 self.block_table[…] ← []
  - 赋值 self.seq_len[…] ← 0
  - 返回 sid

### MI-0066 PagedKVCache._ensure_block(self, sid)
- 模块: `engine\kvcache.py`
- 用途: (无 docstring)  ·  INV 15
- 微指令步骤 (1):
  - 判断 Compare

### MI-0067 PagedKVCache.append(self, sid, k, v)
- 模块: `engine\kvcache.py`
- 用途: 追加单个 token 的 K,V (各 [kv_dim])。  ·  INV 15
- 微指令步骤 (8):
  - 赋值 k ← np.asarray(…)
  - 赋值 v ← np.asarray(…)
  - 调用 self._ensure_block(…)
  - 赋值 bi ← self.block_table[…][…]
  - 赋值 p ← self.seq_len[…] mod self.block_size
  - 赋值 self.k_blocks[…] ← k
  - 赋值 self.v_blocks[…] ← v
  - 累算 self.seq_len[…] add 1

### MI-0068 PagedKVCache.get_kv(self, sid)
- 模块: `engine\kvcache.py`
- 用途: 返回该序列截至当前的稠密 K,V [L, kv_dim] (INV-15 等价性来源)。  ·  INV 15
- 微指令步骤 (5):
  - 赋值 L ← self.seq_len[…]
  - 赋值 K ← np.zeros(…)
  - 赋值 V ← np.zeros(…)
  - 遍历 l ∈ range(…)
  - 返回 [K, V]

### MI-0069 PagedKVCache.memory_blocks_used(self)
- 模块: `engine\kvcache.py`
- 用途: (无 docstring)  ·  INV 15
- 装饰器: property
- 微指令步骤 (1):
  - 返回 self.max_blocks sub len(…)

### MI-0070 【类】PrefillResult.PrefillResult(类)
- 模块: `engine\prefill.py`
- 用途: 一次 prefill 的产物。  ·  INV 11,15,20
- 微指令步骤 (6):
  - 声明并赋值 logits ← ∅
  - 声明并赋值 kv_cache ← field(…)
  - 声明并赋值 n_chunks ← 1
  - 声明并赋值 chunk_size ← 0
  - FunctionDef
  - FunctionDef

### MI-0071 PrefillResult.n_tokens(self)
- 模块: `engine\prefill.py`
- 用途: prefill 的 token 数 (不含批维)。  ·  INV 11,15,20
- 装饰器: property
- 微指令步骤 (1):
  - 返回 int(…)

### MI-0072 PrefillResult.last_logits(self)
- 模块: `engine\prefill.py`
- 用途: 返回最后一个位置的 logits ``[B, V]`` (decode 的起点)。  ·  INV 11,15,20
- 微指令步骤 (1):
  - 返回 self.logits[…]

### MI-0073 【类】ChunkedPrefill.ChunkedPrefill(类)
- 模块: `engine\prefill.py`
- 用途: 分块 prefill 与整段 prefill 的统一入口 (INV-20)。  ·  INV 20
- 微指令步骤 (6):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0074 ChunkedPrefill.__init__(self, model, chunk_size, tol)
- 模块: `engine\prefill.py`
- 用途: (无 docstring)  ·  INV 11,15,20
- 微指令步骤 (4):
  - 判断 Compare
  - 赋值 self.model ← model
  - 赋值 self.chunk_size ← int(…)
  - 赋值 self.tol ← float(…)

### MI-0075 ChunkedPrefill.run(self, token_ids, chunk_size, return_hidden)
- 模块: `engine\prefill.py`
- 用途: 分块累积前向: 每块只看历史 KV, 峰值激活与块长成正比。  ·  INV 11,15,20
- 微指令步骤 (11):
  - 赋值 ids ← np.asarray(…)
  - 判断 Compare
  - 赋值 (B, T) ← ids.shape
  - 赋值 cs ← IfExp
  - 赋值 cs ← max(…)
  - 声明并赋值 kv ← None
  - 声明并赋值 parts ← []
  - 赋值 n_chunks ← 0
  - 遍历 start ∈ range(…)
  - 赋值 logits ← IfExp
  - 返回 PrefillResult(…)

### MI-0076 ChunkedPrefill.run_whole(self, token_ids, return_hidden)
- 模块: `engine\prefill.py`
- 用途: 整段一次性前向 (INV-20 的对照基准)。  ·  INV 20
- 微指令步骤 (5):
  - 赋值 ids ← np.asarray(…)
  - 判断 Compare
  - 赋值 T ← ids.shape[…]
  - 赋值 (out, present) ← self.model.forward(…)
  - 返回 PrefillResult(…)

### MI-0077 ChunkedPrefill.last_logits(self, token_ids, chunk_size)
- 模块: `engine\prefill.py`
- 用途: 分块 prefill 后取末位 logits ``[B, V]`` (供 decode 使用)。  ·  INV 11,15,20
- 微指令步骤 (1):
  - 返回 Call.last_logits(…)

### MI-0078 ChunkedPrefill.verify(self, token_ids, chunk_size, tol)
- 模块: `engine\prefill.py`
- 用途: 对比分块 prefill 与整段 prefill, 返回可断言的报告。  ·  INV 11,15,20
- 微指令步骤 (5):
  - 赋值 tol ← IfExp
  - 赋值 chunked ← self.run(…)
  - 赋值 whole ← self.run_whole(…)
  - 赋值 diff ← max_abs_diff(…)
  - 返回 Dict

### MI-0079 ChunkedPrefill.verify_kv(self, token_ids, chunk_size, tol)
- 模块: `engine\prefill.py`
- 用途: 分块与整段两条路径的 KV cache 是否逐位等价 (INV-20 的更强判据)。  ·  INV 16,20
- 微指令步骤 (7):
  - 赋值 tol ← IfExp
  - 赋值 chunked ← self.run(…)
  - 赋值 whole ← self.run_whole(…)
  - 声明并赋值 diffs ← []
  - 遍历 ((kc, vc), (kw, vw)) ∈ zip(…)
  - 赋值 worst ← IfExp
  - 返回 Dict

### MI-0080 max_abs_diff(a, b)
- 模块: `engine\prefill.py`
- 用途: 两个数组的最大绝对差 (形状不一致时返回 ``inf``)。  ·  INV 11,15,20
- 微指令步骤 (5):
  - 赋值 a ← np.asarray(…)
  - 赋值 b ← np.asarray(…)
  - 判断 Compare
  - 判断 Compare
  - 返回 float(…)

### MI-0081 chunked_prefill(model, token_ids, chunk_size)
- 模块: `engine\prefill.py`
- 用途: 函数式入口: 一次分块 prefill。  ·  INV 11,15,20
- 微指令步骤 (1):
  - 返回 Call.run(…)

### MI-0082 full_prefill(model, token_ids)
- 模块: `engine\prefill.py`
- 用途: 函数式入口: 整段 prefill (基准)。  ·  INV 11,15,20
- 微指令步骤 (1):
  - 返回 Call.run_whole(…)

### MI-0083 verify_prefill(model, token_ids, chunk_size, tol)
- 模块: `engine\prefill.py`
- 用途: 函数式入口: INV-20 等价性报告。  ·  INV 20
- 微指令步骤 (1):
  - 返回 Call.verify(…)

### MI-0084 【类】Sampler.Sampler(类)
- 模块: `engine\sampler.py`
- 用途: (无 docstring)  ·  INV 18,19
- 微指令步骤 (4):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0085 Sampler.__init__(self, seed)
- 模块: `engine\sampler.py`
- 用途: (无 docstring)  ·  INV 18,19
- 微指令步骤 (1):
  - 赋值 self._rng ← IfExp

### MI-0086 Sampler.probs(self, logits, temperature, top_k, top_p, min_p, repetition_penalty, forbidden)
- 模块: `engine\sampler.py`
- 用途: (无 docstring)  ·  INV 18,19
- 微指令步骤 (11):
  - 赋值 logits ← Call.copy(…)
  - 判断 forbidden
  - 判断 Compare
  - 判断 Compare
  - 赋值 logits ← logits div max(…)
  - 判断 BoolOp
  - 判断 Compare
  - 赋值 probs ← numeric.stable_softmax(…)
  - 判断 Compare
  - 赋值 s ← probs.sum(…)
  - 返回 probs div s

### MI-0087 Sampler.greedy(self, logits, forbidden)
- 模块: `engine\sampler.py`
- 用途: (无 docstring)  ·  INV 18,19
- 微指令步骤 (3):
  - 赋值 logits ← Call.copy(…)
  - 判断 forbidden
  - 返回 int(…)

### MI-0088 Sampler.sample(self, logits, temperature, top_k, top_p, min_p, repetition_penalty, forbidden)
- 模块: `engine\sampler.py`
- 用途: (无 docstring)  ·  INV 18,19
- 微指令步骤 (3):
  - 赋值 p ← self.probs(…)
  - 判断 Compare
  - 返回 int(…)

### MI-0089 【类】ContinuousScheduler.ContinuousScheduler(类)
- 模块: `engine\scheduler.py`
- 用途: 对同一批 prompt 做并行贪心/采样生成。  ·  INV 16
- 微指令步骤 (3):
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0090 ContinuousScheduler.__init__(self, model_forward, sampler)
- 模块: `engine\scheduler.py`
- 用途: (无 docstring)  ·  INV 16
- 微指令步骤 (2):
  - 赋值 self.fwd ← model_forward
  - 赋值 self.sampler ← sampler

### MI-0091 ContinuousScheduler._sample(self, logits_1d)
- 模块: `engine\scheduler.py`
- 用途: (无 docstring)  ·  INV 16
- 微指令步骤 (2):
  - 判断 Compare
  - 返回 int(…)

### MI-0092 ContinuousScheduler.generate(self, prompts, max_new_tokens, eos_id, greedy, pad_id)
- 模块: `engine\scheduler.py`
- 用途: (无 docstring)  ·  INV 16
- 微指令步骤 (5):
  - 赋值 seqs ← ListComp
  - 赋值 finished ← [False] mult len(…)
  - 赋值 produced ← [0] mult len(…)
  - 遍历 _ ∈ range(…)
  - 返回 ListComp

### MI-0093 【类】EngineBackend.EngineBackend(类)
- 模块: `engine\session.py`
- 用途: 推理后端选择。  ·  INV 15,16,17,20  ·  复用 Speculative
- 微指令步骤 (1):
  - 赋值 BACKENDS ← ["naive", "paged", "chunked"]

### MI-0094 【类】EngineSession.EngineSession(类)
- 模块: `engine\session.py`
- 用途: 一次生成会话 —— 持有 KV 缓存并支持多种推理后端。  ·  INV 15,16,17,20  ·  复用 OpenAI
- 微指令步骤 (8):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0095 EngineSession.__init__(self, model, sampler, backend, block_size, chunk_size, draft_model)
- 模块: `engine\session.py`
- 用途: (无 docstring)  ·  INV 15,16,17,20
- 微指令步骤 (8):
  - 判断 Compare
  - 赋值 self.model ← model
  - 赋值 self.sampler ← BoolOp
  - 赋值 self.backend ← backend
  - 赋值 self.chunk_size ← int(…)
  - 赋值 self.draft_model ← draft_model
  - 赋值 self._kv_dim ← model.cfg.n_kv_heads mult model.cfg.d_head
  - 赋值 self.cache ← PagedKVCache(…)

### MI-0096 EngineSession.prefill(self, token_ids)
- 模块: `engine\session.py`
- 用途: 对整段 prompt 做 (可选分块) prefill, 返回 ``(logits, kv_cache)``。  ·  INV 20
- 微指令步骤 (4):
  - 赋值 ids ← np.asarray(…)
  - 判断 Compare
  - 赋值 (out, present) ← self.model.forward(…)
  - 返回 [out[…], present]

### MI-0097 EngineSession.generate(self, prompt_ids, max_new_tokens, eos_id, temperature, top_p)
- 模块: `engine\session.py`
- 用途: 生成新 token 列表 (不含 prompt)。  ·  INV 15,16,17,20
- 微指令步骤 (5):
  - 赋值 ids ← list(…)
  - 判断 Compare
  - 判断 Compare
  - 遍历 _ ∈ range(…)
  - 返回 ids[…]

### MI-0098 EngineSession._generate_paged(self, ids, max_new_tokens, eos_id, temperature, top_p)
- 模块: `engine\session.py`
- 用途: 分页 KV 增量 decode: 每步只前向新 token, 历史 KV 复用。  ·  INV 15,16,17,20
- 微指令步骤 (4):
  - 赋值 prompt_len ← len(…)
  - 赋值 (_, present) ← self.model.forward(…)
  - 遍历 _ ∈ range(…)
  - 返回 ids[…]

### MI-0099 EngineSession._generate_speculative(self, ids, max_new_tokens, eos_id, temperature)
- 模块: `engine\session.py`
- 用途: 投机解码: 草稿模型提候选, 目标模型批量验证 (INV-17)。  ·  INV 17
- 微指令步骤 (3):
  - 判断 Compare
  - FunctionDef
  - 返回 speculative_decode(…)

### MI-0100 EngineSession._pick(self, logits, temperature, top_p)
- 模块: `engine\session.py`
- 用途: (无 docstring)  ·  INV 15,16,17,20
- 微指令步骤 (2):
  - 判断 Compare
  - 返回 int(…)

### MI-0101 EngineSession.generate_batch(self, prompts, max_new_tokens, eos_id, temperature)
- 模块: `engine\session.py`
- 用途: 连续批处理: 多条 prompt 共享前向步, 每步只算「仍在生成」的序列。  ·  INV 16  ·  复用 Orca, Yu et al
- 微指令步骤 (8):
  - 赋值 seqs ← ListComp
  - 判断 UnaryOp
  - 赋值 max_len ← max(…)
  - 赋值 model ← self.model
  - FunctionDef
  - 赋值 sched ← ContinuousScheduler(…)
  - 赋值 out ← sched.generate(…)
  - 返回 out

### MI-0102 EngineSession.kv_stats(self)
- 模块: `engine\session.py`
- 用途: 返回分页 KV 缓存的占用统计 (可观测性, 供 /metrics 使用)。  ·  INV 15,16,17,20
- 微指令步骤 (1):
  - 返回 Dict

### MI-0103 speculative_decode(target_next, draft_next, prompt, gamma, max_new_tokens, eos_id, greedy, temperature)
- 模块: `engine\speculative.py`
- 用途: 返回新生成的 token 列表 (不含 prompt)。  ·  INV 17
- 微指令步骤 (4):
  - 赋值 r ← rng.get_rng(…)
  - 赋值 seq ← list(…)
  - 当 Compare 循环
  - 返回 seq[…]

### MI-0104 _softmax(x)
- 模块: `engine\speculative.py`
- 用途: (无 docstring)  ·  INV 17
- 微指令步骤 (3):
  - 赋值 m ← x.max(…)
  - 赋值 e ← np.exp(…)
  - 返回 e div e.sum(…)

## L4 训练/微调 (Training/Finetune)

### MI-0105 _unbroadcast(grad, shape)
- 模块: `train\autodiff.py`
- 用途: 把梯度累加值收缩回 ``shape`` (处理 numpy 广播的反向, INV-22)。  ·  INV 22
- 微指令步骤 (4):
  - 赋值 grad ← np.asarray(…)
  - 判断 Compare
  - 遍历 i ∈ range(…)
  - 返回 grad

### MI-0106 【类】Tensor.Tensor(类)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (30):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0107 Tensor.__init__(self, data, _children, _op, requires_grad)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (5):
  - 赋值 self.data ← np.asarray(…)
  - 赋值 self.requires_grad ← requires_grad
  - 赋值 self.grad ← IfExp
  - 赋值 self._backward ← Lambda
  - 赋值 self._prev ← list(…)

### MI-0108 Tensor.shape(self)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 装饰器: property
- 微指令步骤 (1):
  - 返回 self.data.shape

### MI-0109 Tensor._set(self, val)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (1):
  - 赋值 self.data ← np.asarray(…)

### MI-0110 Tensor.__add__(self, other)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (5):
  - 赋值 other ← IfExp
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0111 Tensor.__mul__(self, other)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (5):
  - 赋值 other ← IfExp
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0112 Tensor.__sub__(self, other)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (2):
  - 赋值 other ← IfExp
  - 返回 self add UnaryOp

### MI-0113 Tensor.__truediv__(self, other)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (5):
  - 赋值 other ← IfExp
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0114 Tensor.__rtruediv__(self, other)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (5):
  - 赋值 other ← IfExp
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0115 Tensor.__neg__(self)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (4):
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0116 Tensor.matmul(self, other)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (4):
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0117 Tensor.fake_quant_int8_per_channel(self, axis)
- 模块: `train\autodiff.py`
- 用途: STE 伪量化 (QAT, Jacob et al. 2018): 前向 = dequant(quant(W)),  ·  INV 7  ·  复用 dettmers, int8
- 微指令步骤 (6):
  - 赋值 (q, scale) ← quant.quantize_int8_per_channel(…)
  - 赋值 dq ← quant.dequantize_int8(…)
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0118 Tensor.transpose(self, axes)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (4):
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0119 Tensor.reshape(self, *shape)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (4):
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0120 Tensor.sigmoid(self)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (5):
  - 赋值 s ← 1.0 div 1.0 add np.exp(…)
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0121 Tensor.tanh(self)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (5):
  - 赋值 t ← np.tanh(…)
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0122 Tensor.gelu(self)
- 模块: `train\autodiff.py`
- 用途: 可微 GELU (Hendrycks & Gimpel 2016, tanh-approx, GPT/LLaMA 同款)。  ·  INV 22  ·  复用 SwiGLU
- 微指令步骤 (9):
  - 赋值 x ← self.data
  - 赋值 c ← np.sqrt(…)
  - 赋值 u ← c mult x add 0.044715 mult x pow 3
  - 赋值 t ← np.tanh(…)
  - 赋值 out_data ← 0.5 mult x mult 1.0 add t
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0123 Tensor.relu(self)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (5):
  - 赋值 m ← Compare
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0124 Tensor.exp(self)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (5):
  - 赋值 e ← np.exp(…)
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0125 Tensor.log(self)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (4):
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0126 Tensor.mean(self, axis, keepdims)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (4):
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0127 Tensor.sum(self, axis, keepdims)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (4):
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0128 Tensor.max(self, axis, keepdims)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (4):
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0129 Tensor.pow(self, p)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (4):
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0130 Tensor.__getitem__(self, idx)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (4):
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0131 Tensor.softmax(self, axis)
- 模块: `train\autodiff.py`
- 用途: 可微 softmax (数值稳定: max 作为常数 detach)。  ·  INV 22
- 微指令步骤 (7):
  - 赋值 m ← self.data.max(…)
  - 赋值 e ← np.exp(…)
  - 赋值 p ← e div e.sum(…)
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0132 Tensor.repeat(self, repeats, axis)
- 模块: `train\autodiff.py`
- 用途: 沿 ``axis`` 把每个元素连续重复 ``repeats`` 次 (np.repeat 语义)。  ·  INV 10  ·  复用 GQA
- 微指令步骤 (4):
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0133 Tensor.attention(q, k, v, scale, bias)
- 模块: `train\autodiff.py`
- 用途: 批量缩放点积注意力 (自定义单节点: BLAS 前向 + 标准解析反向)。  ·  INV 22  ·  复用 Vaswani
- 装饰器: staticmethod
- 微指令步骤 (11):
  - 赋值 (qd, kd, vd) ← [q.data, k.data, v.data]
  - 赋值 s ← qd matmult kd.swapaxes(…) mult scale
  - 判断 Compare
  - 赋值 m ← s.max(…)
  - 赋值 e ← np.exp(…)
  - 赋值 p ← e div e.sum(…)
  - 赋值 ctx ← p matmult vd
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0134 Tensor.backward(self)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (6):
  - 赋值 topo ← []
  - 赋值 visited ← set(…)
  - FunctionDef
  - 调用 build(…)
  - 赋值 self.grad ← np.ones_like(…)
  - 遍历 v ∈ reversed(…)

### MI-0135 Tensor.zeros(*shape)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 装饰器: staticmethod
- 微指令步骤 (1):
  - 返回 Tensor(…)

### MI-0136 Tensor.parameter(data)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 装饰器: staticmethod
- 微指令步骤 (1):
  - 返回 Tensor(…)

### MI-0137 _topo_from(root)
- 模块: `train\checkpoint.py`
- 用途: 拓扑排序: 保证子节点先于父节点被反向调用。  ·  INV 25
- 微指令步骤 (5):
  - 声明并赋值 topo ← []
  - 赋值 visited ← set(…)
  - FunctionDef
  - 调用 build(…)
  - 返回 topo

### MI-0138 backward_from(root, grad)
- 模块: `train\checkpoint.py`
- 用途: 从 ``root`` 以给定上游梯度做**局部**反向 (不重置全局图)。  ·  INV 25
- 微指令步骤 (3):
  - 赋值 grad ← np.asarray(…)
  - 赋值 root.grad ← Call.copy(…)
  - 遍历 node ∈ reversed(…)

### MI-0139 checkpoint(fn, *args, enabled)
- 模块: `train\checkpoint.py`
- 用途: 以检查点方式执行 ``fn(*args)``: 前向丢弃段内激活, 反向时重算。  ·  INV 25
- 微指令步骤 (8):
  - 判断 UnaryOp
  - 赋值 out ← fn(…)
  - 判断 UnaryOp
  - 赋值 y ← Tensor(…)
  - 赋值 y._prev ← []
  - FunctionDef
  - 赋值 y._backward ← _recompute_backward
  - 返回 y

### MI-0140 checkpointed(fn, enabled)
- 模块: `train\checkpoint.py`
- 用途: 装饰器版本: ``@checkpointed`` 令整个函数体走检查点路径。  ·  INV 25
- 微指令步骤 (3):
  - FunctionDef
  - 判断 Compare
  - 返回 wrap

### MI-0141 grad_diff(a, b)
- 模块: `train\checkpoint.py`
- 用途: 两组同名参数梯度的最大绝对差 (INV-25 判据之一)。  ·  INV 25
- 微指令步骤 (3):
  - 赋值 worst ← 0.0
  - 遍历 key ∈ sorted(…)
  - 返回 worst

### MI-0142 verify_checkpoint(loss_fn, params_fn, tol)
- 模块: `train\checkpoint.py`
- 用途: INV-25 一键自检: 开/关检查点两次前向反向, 对比 loss 与梯度。  ·  INV 25
- 微指令步骤 (18):
  - FunctionDef
  - 赋值 plain_params ← params_fn(…)
  - 调用 zero_grad(…)
  - 赋值 loss_plain_t ← loss_fn(…)
  - 调用 loss_plain_t.backward(…)
  - 赋值 loss_plain ← float(…)
  - 赋值 plain_snapshot ← DictComp
  - 赋值 ckpt_params ← params_fn(…)
  - 调用 zero_grad(…)
  - 赋值 loss_ckpt_t ← loss_fn(…)
  - 调用 loss_ckpt_t.backward(…)
  - 赋值 loss_ckpt ← float(…)
  - 赋值 ckpt_snapshot ← DictComp
  - 赋值 gd ← 0.0
  - 遍历 key ∈ sorted(…)
  - 赋值 ld ← abs(…)
  - 赋值 ok ← bool(…)
  - 返回 [ok, Dict]

### MI-0143 checkpoint_segments(n_items, n_segments)
- 模块: `train\checkpoint.py`
- 用途: 把 ``n_items`` 项 (如层) 均分为 ``n_segments`` 个检查点段, 返回区间列表。  ·  INV 25
- 微指令步骤 (4):
  - 判断 Compare
  - 赋值 n_segments ← max(…)
  - 赋值 bounds ← ListComp
  - 返回 ListComp

### MI-0144 【类】LoRALinear.LoRALinear(类)
- 模块: `train\lora.py`
- 用途: (无 docstring)  ·  INV 24
- 微指令步骤 (5):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0145 LoRALinear.__init__(self, W, r, alpha)
- 模块: `train\lora.py`
- 用途: (无 docstring)  ·  INV 24
- 微指令步骤 (8):
  - 赋值 self.W ← Tensor(…)
  - 赋值 (d_out, d_in) ← W.shape
  - 赋值 self.r ← r
  - 赋值 self.scale ← alpha div r
  - 赋值 self.A ← Tensor(…)
  - 赋值 self.B ← Tensor(…)
  - 赋值 rng ← np.random.default_rng(…)
  - 赋值 self.A.data ← rng.standard_normal(…) mult 0.01

### MI-0146 LoRALinear.trainable_params(self)
- 模块: `train\lora.py`
- 用途: (无 docstring)  ·  INV 24
- 装饰器: property
- 微指令步骤 (1):
  - 返回 [self.A, self.B]

### MI-0147 LoRALinear.forward(self, x)
- 模块: `train\lora.py`
- 用途: (无 docstring)  ·  INV 24
- 微指令步骤 (3):
  - 赋值 base ← x.matmul(…)
  - 赋值 lora ← Call.matmul(…) mult self.scale
  - 返回 base add lora

### MI-0148 LoRALinear.merge(self)
- 模块: `train\lora.py`
- 用途: 合并低秩增量到原权重, 返回新 W (INV-24)。  ·  INV 24
- 微指令步骤 (1):
  - 返回 self.W.data add self.scale mult self.B.data matmult self.A.data

### MI-0149 LoRALinear.merged_linear(self, x)
- 模块: `train\lora.py`
- 用途: (无 docstring)  ·  INV 24
- 微指令步骤 (1):
  - 返回 x.matmul(…)

### MI-0150 【类】GlobalStep.GlobalStep(类)
- 模块: `train\optim.py`
- 用途: 学习率调度状态机 (线性预热 + 余弦退火)。  ·  INV 21,23
- 微指令步骤 (2):
  - FunctionDef
  - FunctionDef

### MI-0151 GlobalStep.__init__(self, lr, total_steps, warmup_steps, min_lr)
- 模块: `train\optim.py`
- 用途: (无 docstring)  ·  INV 21,23
- 微指令步骤 (5):
  - 赋值 self.lr0 ← lr
  - 赋值 self.total ← max(…)
  - 赋值 self.warmup ← warmup_steps
  - 赋值 self.min_lr ← min_lr
  - 赋值 self.t ← 0

### MI-0152 GlobalStep.step(self)
- 模块: `train\optim.py`
- 用途: (无 docstring)  ·  INV 21,23
- 微指令步骤 (3):
  - 赋值 lr ← cosine_schedule(…)
  - 累算 self.t add 1
  - 返回 lr

### MI-0153 cosine_schedule(lr0, step, total, warmup, min_lr)
- 模块: `train\optim.py`
- 用途: 线性预热 + 余弦退火。解析公式, 供 INV(R-L4-02) 对照。  ·  INV 21,23
- 微指令步骤 (4):
  - 判断 BoolOp
  - 判断 Compare
  - 赋值 progress ← step sub warmup div max(…)
  - 返回 min_lr add 0.5 mult lr0 sub min_lr mult 1.0 add np.cos(…)

### MI-0154 【类】AdamW.AdamW(类)
- 模块: `train\optim.py`
- 用途: 解耦权重衰减 AdamW。  ·  INV 21,23  ·  复用 AdamW
- 微指令步骤 (3):
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0155 AdamW.__init__(self, params, lr, betas, eps, weight_decay)
- 模块: `train\optim.py`
- 用途: (无 docstring)  ·  INV 21,23
- 微指令步骤 (8):
  - 赋值 self.params ← ListComp
  - 赋值 self.lr ← lr
  - 赋值 (self.b1, self.b2) ← betas
  - 赋值 self.eps ← eps
  - 赋值 self.wd ← weight_decay
  - 赋值 self.m ← ListComp
  - 赋值 self.v ← ListComp
  - 赋值 self.step_count ← 0

### MI-0156 AdamW.step(self)
- 模块: `train\optim.py`
- 用途: (无 docstring)  ·  INV 21,23
- 微指令步骤 (4):
  - 累算 self.step_count add 1
  - 赋值 bc1 ← 1 sub self.b1 pow self.step_count
  - 赋值 bc2 ← 1 sub self.b2 pow self.step_count
  - 遍历 (i, p) ∈ enumerate(…)

### MI-0157 AdamW.zero_grad(self)
- 模块: `train\optim.py`
- 用途: (无 docstring)  ·  INV 21,23
- 微指令步骤 (1):
  - 遍历 p ∈ self.params

### MI-0158 clip_global_norm(params, max_norm, norm_type)
- 模块: `train\optim.py`
- 用途: 全局范数裁剪。返回 (total_norm, triggered)。INV-23。  ·  INV 23
- 微指令步骤 (4):
  - 赋值 norms ← ListComp
  - 赋值 total ← float(…)
  - 判断 BoolOp
  - 返回 [total, False]

### MI-0159 【类】TinyTrainer.TinyTrainer(类)
- 模块: `train\trainer.py`
- 用途: (无 docstring)  ·  INV 25,26
- 微指令步骤 (17):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0160 TinyTrainer.__init__(self, config)
- 模块: `train\trainer.py`
- 用途: (无 docstring)  ·  INV 25,26
- 微指令步骤 (4):
  - 赋值 self.cfg ← config
  - 声明并赋值 self.params ← Dict
  - 赋值 self.gradient_checkpointing ← False
  - 调用 self._build(…)

### MI-0161 TinyTrainer._P(self, name, shape)
- 模块: `train\trainer.py`
- 用途: (无 docstring)  ·  INV 25,26
- 微指令步骤 (3):
  - 赋值 r ← rng.get_rng(…)
  - 赋值 data ← BinOp.astype(…)
  - 赋值 self.params[…] ← Tensor(…)

### MI-0162 TinyTrainer._build(self)
- 模块: `train\trainer.py`
- 用途: (无 docstring)  ·  INV 25,26
- 微指令步骤 (7):
  - 赋值 c ← self.cfg
  - 赋值 (V, d) ← [c.vocab_size, c.d_model]
  - 调用 self._P(…)
  - 判断 c.tie_word_embeddings
  - 调用 self._P(…)
  - 赋值 n_kv ← c.n_kv_heads
  - 遍历 l ∈ range(…)

### MI-0163 TinyTrainer._rope(self, x, pos, dim)
- 模块: `train\trainer.py`
- 用途: RoPE 旋转 (单节点自定义算子, 反向 = 正交旋转转置)。  ·  INV 25,26  ·  复用 RoPE, Su et al
- 微指令步骤 (14):
  - 赋值 half ← dim floordiv 2
  - 赋值 freqs ← ops._rope_freqs(…)
  - 赋值 angles ← np.outer(…)
  - 赋值 cos ← np.cos(…)[…]
  - 赋值 sin ← np.sin(…)[…]
  - 赋值 x1 ← x.data[…]
  - 赋值 x2 ← x.data[…]
  - 赋值 out ← np.empty_like(…)
  - 赋值 out[…] ← x1 mult cos sub x2 mult sin
  - 赋值 out[…] ← x1 mult sin add x2 mult cos
  - 赋值 out_t ← Tensor(…)
  - FunctionDef
  - 赋值 out_t._backward ← _b
  - 返回 out_t

### MI-0164 TinyTrainer.enable_gradient_checkpointing(self, enabled)
- 模块: `train\trainer.py`
- 用途: 开启/关闭梯度检查点 (INV-25)。  ·  INV 25
- 微指令步骤 (1):
  - 赋值 self.gradient_checkpointing ← bool(…)

### MI-0165 TinyTrainer._causal_bias(T)
- 模块: `train\trainer.py`
- 用途: 因果掩码偏置 [T,T]: 上三角 -1e30 (经 softmax 后概率严格为 0)。  ·  INV 25,26
- 装饰器: staticmethod
- 微指令步骤 (2):
  - 赋值 mask ← np.triu(…)
  - 返回 np.where(…)

### MI-0166 TinyTrainer._bias(self, T)
- 模块: `train\trainer.py`
- 用途: 因果偏置缓存 (同 T 复用, 免每层每步重建)。  ·  INV 25,26
- 微指令步骤 (5):
  - 赋值 cache ← getattr(…)
  - 判断 Compare
  - 赋值 b ← cache.get(…)
  - 判断 Compare
  - 返回 b

### MI-0167 TinyTrainer._layer_forward(self, x, layer)
- 模块: `train\trainer.py`
- 用途: 单层 DecoderBlock 前向 (供梯度检查点按层分段重算, INV-25)。  ·  INV 25  ·  复用 GQA
- 微指令步骤 (25):
  - 赋值 c ← self.cfg
  - 赋值 (B, T) ← [x.data.shape[…], x.data.shape[…]]
  - 赋值 (d, dh, n_h, n_kv) ← [c.d_model, c.d_head, c.n_heads]
  - 赋值 pos ← np.arange(…)
  - 赋值 p ← JoinedStr
  - 赋值 ms ← BinOp.mean(…)
  - 赋值 h ← x div BinOp.pow(…) mult self.params[…]
  - 赋值 q ← self._apply_lora(…)
  - 赋值 k ← self._apply_lora(…)
  - 赋值 v ← self._apply_lora(…)
  - 赋值 q ← Call.transpose(…)
  - 赋值 k ← Call.transpose(…)
  - 赋值 v ← Call.transpose(…)
  - 赋值 k ← k.repeat(…)
  - 赋值 v ← v.repeat(…)
  - 赋值 scale ← 1.0 div dh pow 0.5
  - 赋值 ctx ← Tensor.attention(…)
  - 赋值 ctx ← Call.reshape(…)
  - 赋值 o ← ctx.matmul(…)
  - 赋值 o ← self._apply_lora(…)
  - 赋值 x ← x add o
  - 赋值 ms2 ← BinOp.mean(…)
  - 赋值 h2 ← x div BinOp.pow(…) mult self.params[…]
  - 判断 c.use_moe
  - 返回 x

### MI-0168 TinyTrainer._moe_forward(self, h2, layer)
- 模块: `train\trainer.py`
- 用途: MoE 前向 (可微, 严格镜像推理端 ``Transformer._moe`` 以保证  ·  INV 14  ·  复用 Fedus, MoE
- 微指令步骤 (14):
  - 赋值 c ← self.cfg
  - 赋值 (B, T, d) ← h2.data.shape
  - 赋值 (E, K) ← [c.n_experts, c.moe_topk]
  - 赋值 router ← self._wq(…)
  - 赋值 router_logits ← h2.matmul(…)
  - 赋值 rl ← router_logits.data
  - 赋值 topk_idx ← np.argsort(…)[…]
  - 赋值 bi ← np.arange(…)[…]
  - 赋值 ti ← np.arange(…)[…]
  - 赋值 sel ← router_logits[…]
  - 赋值 w ← sel.softmax(…)
  - 赋值 out ← Tensor.zeros(…)
  - 遍历 e ∈ range(…)
  - 返回 out

### MI-0169 TinyTrainer._wq(self, name)
- 模块: `train\trainer.py`
- 用途: QAT 门控 (Jacob et al. 2018): ``use_qat`` 时返回权重的 STE 伪量化  ·  INV 25,26
- 微指令步骤 (2):
  - 赋值 w ← self.params[…]
  - 返回 IfExp

### MI-0170 TinyTrainer._apply_lora(self, x, pre, tgt, base_out)
- 模块: `train\trainer.py`
- 用途: LoRA 低秩增量 (Hu et al. 2022, INV-24): ``base + (α/r)·(x@Aᵀ)@Bᵀ``。  ·  INV 24  ·  复用 Hu et al, LoRA
- 微指令步骤 (7):
  - 赋值 c ← self.cfg
  - 判断 UnaryOp
  - 赋值 A ← self._wq(…)
  - 赋值 B ← self._wq(…)
  - 赋值 scale ← c.lora_alpha div c.lora_r
  - 赋值 lora ← Call.matmul(…) mult scale
  - 返回 base_out add lora

### MI-0171 TinyTrainer.forward(self, ids)
- 模块: `train\trainer.py`
- 用途: (无 docstring)  ·  INV 25,26
- 微指令步骤 (9):
  - 赋值 c ← self.cfg
  - 赋值 ids ← np.asarray(…)
  - 判断 Compare
  - 赋值 x ← self._wq(…)[…]
  - 遍历 l ∈ range(…)
  - 赋值 ms3 ← BinOp.mean(…)
  - 赋值 x ← x div BinOp.pow(…) mult self.params[…]
  - 赋值 logits ← x.matmul(…)
  - 返回 logits

### MI-0172 TinyTrainer.loss(self, ids, targets)
- 模块: `train\trainer.py`
- 用途: (无 docstring)  ·  INV 25,26
- 微指令步骤 (10):
  - 赋值 logits ← self.forward(…)
  - 赋值 logits_flat ← logits.reshape(…)
  - 赋值 tgt ← targets.reshape(…)
  - 赋值 m ← logits_flat.data.max(…)
  - 赋值 z ← logits_flat sub Tensor(…)
  - 赋值 logsum ← Call.log(…)
  - 赋值 lse ← logsum add Tensor(…)
  - 赋值 logp ← logits_flat sub lse
  - 赋值 nll ← UnaryOp
  - 返回 nll

### MI-0173 TinyTrainer.train(self, ids_corpus, steps, lr, batch_size, seq_len, grad_clip, warmup)
- 模块: `train\trainer.py`
- 用途: (无 docstring)  ·  INV 25,26
- 微指令步骤 (7):
  - 赋值 params ← ListComp
  - 赋值 opt ← AdamW(…)
  - 赋值 sched ← GlobalStep(…)
  - 赋值 losses ← []
  - 赋值 N ← len(…) sub seq_len
  - 遍历 s ∈ range(…)
  - 返回 losses

### MI-0174 TinyTrainer.to_numpy_params(self)
- 模块: `train\trainer.py`
- 用途: (无 docstring)  ·  INV 25,26
- 微指令步骤 (4):
  - 赋值 out ← Dict
  - 遍历 (k, v) ∈ self.params.items(…)
  - 判断 self.cfg.tie_word_embeddings
  - 返回 out

### MI-0175 TinyTrainer.count_params(self)
- 模块: `train\trainer.py`
- 用途: 可训练参数量 (与 ``Transformer.count_params`` 同口径, INV-13)。  ·  INV 12,13
- 微指令步骤 (2):
  - 赋值 skip ← IfExp
  - 返回 int(…)

### MI-0176 TinyTrainer.save(self, dir_path)
- 模块: `train\trainer.py`
- 用途: (无 docstring)  ·  INV 25,26
- 微指令步骤 (7):
  - 调用 os.makedirs(…)
  - 赋值 params ← self.to_numpy_params(…)
  - 调用 np.savez(…)
  - 上下文管理 (with)
  - 赋值 manifest ← Dict
  - 上下文管理 (with)
  - 返回 dir_path

## L5 检索/记忆 (Retrieval/Memory)

### MI-0177 【类】BM25.BM25(类)
- 模块: `retrieval\bm25.py`
- 用途: (无 docstring)  ·  INV 27
- 微指令步骤 (5):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0178 BM25.__init__(self, k1, b)
- 模块: `retrieval\bm25.py`
- 用途: (无 docstring)  ·  INV 27
- 微指令步骤 (7):
  - 赋值 self.k1 ← k1
  - 赋值 self.b ← b
  - 声明并赋值 self.docs ← []
  - 赋值 self.df ← Counter(…)
  - 赋值 self.idf ← Dict
  - 赋值 self.avgdl ← 0.0
  - 赋值 self.doc_len ← []

### MI-0179 BM25.fit(self, corpus)
- 模块: `retrieval\bm25.py`
- 用途: (无 docstring)  ·  INV 27
- 微指令步骤 (7):
  - 赋值 self.docs ← ListComp
  - 赋值 self.doc_len ← ListComp
  - 赋值 self.avgdl ← IfExp
  - 遍历 d ∈ self.docs
  - 赋值 n ← len(…)
  - 遍历 (w, f) ∈ self.df.items(…)
  - 返回 self

### MI-0180 BM25._score(self, query, doc_idx)
- 模块: `retrieval\bm25.py`
- 用途: (无 docstring)  ·  INV 27
- 微指令步骤 (5):
  - 赋值 tf ← Counter(…)
  - 赋值 dl ← self.doc_len[…]
  - 赋值 score ← 0.0
  - 遍历 w ∈ query
  - 返回 score

### MI-0181 BM25.search(self, query, top_k)
- 模块: `retrieval\bm25.py`
- 用途: (无 docstring)  ·  INV 27
- 微指令步骤 (3):
  - 赋值 scores ← ListComp
  - 调用 scores.sort(…)
  - 返回 scores[…]

### MI-0182 BM25.reference(corpus, query, k1, b)
- 模块: `retrieval\bm25.py`
- 用途: (无 docstring)  ·  INV 27
- 装饰器: staticmethod
- 微指令步骤 (9):
  - 赋值 n ← len(…)
  - 赋值 df ← Counter(…)
  - 遍历 d ∈ corpus
  - 赋值 idf ← DictComp
  - 赋值 dl ← ListComp
  - 赋值 avgdl ← sum(…) div n
  - 赋值 out ← []
  - 遍历 (i, d) ∈ enumerate(…)
  - 返回 sorted(…)

### MI-0183 【类】Chunk.Chunk(类)
- 模块: `retrieval\chunk.py`
- 用途: 一个文本块。  ·  INV 31
- 微指令步骤 (9):
  - 声明并赋值 text ← ∅
  - 声明并赋值 start ← ∅
  - 声明并赋值 end ← ∅
  - 声明并赋值 index ← ∅
  - 声明并赋值 meta ← field(…)
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0184 Chunk.__len__(self)
- 模块: `retrieval\chunk.py`
- 用途: (无 docstring)  ·  INV 31
- 微指令步骤 (1):
  - 返回 len(…)

### MI-0185 Chunk.length(self)
- 模块: `retrieval\chunk.py`
- 用途: 块长度 (字符数)。  ·  INV 31
- 装饰器: property
- 微指令步骤 (1):
  - 返回 len(…)

### MI-0186 Chunk.to_dict(self)
- 模块: `retrieval\chunk.py`
- 用途: (无 docstring)  ·  INV 31
- 微指令步骤 (1):
  - 返回 Dict

### MI-0187 Chunk.from_dict(cls, d)
- 模块: `retrieval\chunk.py`
- 用途: (无 docstring)  ·  INV 31
- 装饰器: classmethod
- 微指令步骤 (1):
  - 返回 cls(…)

### MI-0188 normalize_text(text)
- 模块: `retrieval\chunk.py`
- 用途: 归一化: 折叠所有空白为单空格并去首尾 (INV-31 判据用的比较口径)。  ·  INV 31
- 微指令步骤 (1):
  - 返回 Call.strip(…)

### MI-0189 reconstruct(chunks)
- 模块: `retrieval\chunk.py`
- 用途: 按 offset 去重叠拼接块文本, 还原原文 (INV-31)。  ·  INV 31
- 微指令步骤 (4):
  - 声明并赋值 parts ← []
  - 声明并赋值 prev_end ← None
  - 遍历 ch ∈ chunks
  - 返回 Constant.join(…)

### MI-0190 _snap_end(text, start, end, separators, min_keep)
- 模块: `retrieval\chunk.py`
- 用途: 把切分点 ``end`` 回退到最近的语义分隔符之后 (递归字符切分的核心)。  ·  INV 31
- 微指令步骤 (7):
  - 判断 Compare
  - 赋值 (best, best_len) ← [UnaryOp, 0]
  - 遍历 sep ∈ separators
  - 判断 Compare
  - 赋值 cand ← best add best_len
  - 判断 Compare
  - 返回 cand

### MI-0191 chunk_text(text, chunk_size, overlap, strategy, separators)
- 模块: `retrieval\chunk.py`
- 用途: 把 ``text`` 切成覆盖完整的块 (INV-31)。  ·  INV 31
- 微指令步骤 (12):
  - 判断 Compare
  - 判断 UnaryOp
  - 判断 Compare
  - 赋值 text ← str(…)
  - 赋值 seps ← IfExp
  - 判断 UnaryOp
  - 赋值 min_keep ← IfExp
  - 声明并赋值 chunks ← []
  - 赋值 start ← 0
  - 赋值 n ← len(…)
  - 当 Compare 循环
  - 返回 chunks

### MI-0192 verify_coverage(text, chunks, normalized)
- 模块: `retrieval\chunk.py`
- 用途: INV-31 判据: 拼接后的文本是否 == 原文。  ·  INV 31
- 微指令步骤 (3):
  - 赋值 joined ← reconstruct(…)
  - 判断 normalized
  - 返回 Compare

### MI-0193 coverage_report(text, chunks)
- 模块: `retrieval\chunk.py`
- 用途: 返回覆盖完整性报告, 便于断言与调试。  ·  INV 31
- 微指令步骤 (7):
  - 赋值 n ← len(…)
  - 赋值 covered ← 0
  - 赋值 prev_end ← 0
  - 赋值 contiguous ← True
  - 赋值 offsets_valid ← True
  - 遍历 ch ∈ chunks
  - 返回 Dict

### MI-0194 【类】TextChunker.TextChunker(类)
- 模块: `retrieval\chunk.py`
- 用途: 面向对象封装: 固定配置复用, 并提供覆盖自检。  ·  INV 31
- 微指令步骤 (5):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0195 TextChunker.__init__(self, chunk_size, overlap, strategy, separators)
- 模块: `retrieval\chunk.py`
- 用途: (无 docstring)  ·  INV 31
- 微指令步骤 (4):
  - 赋值 self.chunk_size ← int(…)
  - 赋值 self.overlap ← int(…)
  - 赋值 self.strategy ← strategy
  - 赋值 self.separators ← separators

### MI-0196 TextChunker.split(self, text)
- 模块: `retrieval\chunk.py`
- 用途: 切分文本为覆盖完整的块 (INV-31)。  ·  INV 31
- 微指令步骤 (1):
  - 返回 chunk_text(…)

### MI-0197 TextChunker.reconstruct(self, chunks)
- 模块: `retrieval\chunk.py`
- 用途: 去重叠拼接还原原文。  ·  INV 31
- 微指令步骤 (1):
  - 返回 reconstruct(…)

### MI-0198 TextChunker.verify(self, text, chunks)
- 模块: `retrieval\chunk.py`
- 用途: 覆盖完整性报告 (块未提供时内部先切分)。  ·  INV 31
- 微指令步骤 (2):
  - 赋值 cs ← IfExp
  - 返回 coverage_report(…)

### MI-0199 TextChunker.split_many(self, texts)
- 模块: `retrieval\chunk.py`
- 用途: 批量切分 (便于构建语料级索引)。  ·  INV 31
- 微指令步骤 (1):
  - 返回 ListComp

### MI-0200 stable_hash(t)
- 模块: `retrieval\dense.py`
- 用途: 跨进程稳定的词项哈希 (sha256 派生)。  ·  INV 29
- 微指令步骤 (1):
  - 返回 int.from_bytes(…)

### MI-0201 【类】DenseIndex.DenseIndex(类)
- 模块: `retrieval\dense.py`
- 用途: (无 docstring)  ·  INV 29
- 微指令步骤 (5):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0202 DenseIndex.__init__(self, dim, seed)
- 模块: `retrieval\dense.py`
- 用途: (无 docstring)  ·  INV 29
- 微指令步骤 (2):
  - 赋值 self.dim ← dim
  - 赋值 self.seed ← seed

### MI-0203 DenseIndex._embed(self, tokens)
- 模块: `retrieval\dense.py`
- 用途: (无 docstring)  ·  INV 29
- 微指令步骤 (4):
  - 赋值 vec ← np.zeros(…)
  - 遍历 t ∈ tokens
  - 赋值 n ← np.linalg.norm(…)
  - 返回 IfExp

### MI-0204 DenseIndex.fit(self, corpus)
- 模块: `retrieval\dense.py`
- 用途: (无 docstring)  ·  INV 29
- 微指令步骤 (3):
  - 赋值 self.docs ← ListComp
  - 赋值 self.matrix ← IfExp
  - 返回 self

### MI-0205 DenseIndex.search(self, query, top_k)
- 模块: `retrieval\dense.py`
- 用途: (无 docstring)  ·  INV 29
- 微指令步骤 (5):
  - 赋值 q ← self._embed(…)
  - 判断 Compare
  - 赋值 sims ← self.matrix matmult q
  - 赋值 idx ← np.argsort(…)[…]
  - 返回 ListComp

### MI-0206 DenseIndex.brute(matrix, q, top_k)
- 模块: `retrieval\dense.py`
- 用途: (无 docstring)  ·  INV 29
- 装饰器: staticmethod
- 微指令步骤 (3):
  - 赋值 sims ← matrix matmult q
  - 赋值 idx ← np.argsort(…)[…]
  - 返回 ListComp

### MI-0207 【类】HNSW.HNSW(类)
- 模块: `retrieval\hnsw.py`
- 用途: (无 docstring)  ·  INV 28
- 微指令步骤 (9):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0208 HNSW.__init__(self, M, ef_construction, ef_search, ml, seed)
- 模块: `retrieval\hnsw.py`
- 用途: (无 docstring)  ·  INV 28
- 微指令步骤 (9):
  - 赋值 self.M ← M
  - 赋值 self.efC ← ef_construction
  - 赋值 self.efS ← ef_search
  - 赋值 self.ml ← BoolOp
  - 赋值 self.rng ← random.Random(…)
  - 声明并赋值 self.data ← []
  - 声明并赋值 self.layers ← Dict
  - 赋值 self.max_level ← UnaryOp
  - 赋值 self.enter ← None

### MI-0209 HNSW._dist(self, a, b)
- 模块: `retrieval\hnsw.py`
- 用途: (无 docstring)  ·  INV 28
- 微指令步骤 (1):
  - 返回 float(…)

### MI-0210 HNSW._level(self)
- 模块: `retrieval\hnsw.py`
- 用途: (无 docstring)  ·  INV 28
- 微指令步骤 (1):
  - 返回 int(…)

### MI-0211 HNSW.add(self, vec)
- 模块: `retrieval\hnsw.py`
- 用途: (无 docstring)  ·  INV 28
- 微指令步骤 (9):
  - 赋值 idx ← len(…)
  - 调用 self.data.append(…)
  - 赋值 level ← self._level(…)
  - 赋值 self.layers[…] ← []
  - 赋值 prev_enter ← self.enter
  - 判断 BoolOp
  - 赋值 ep ← IfExp
  - 遍历 lc ∈ range(…)
  - 返回 idx

### MI-0212 HNSW._connect(self, a, b, layer)
- 模块: `retrieval\hnsw.py`
- 用途: (无 docstring)  ·  INV 28
- 微指令步骤 (5):
  - 判断 Compare
  - 调用 self.layers.setdefault(…)
  - 调用 self.layers.setdefault(…)
  - 判断 Compare
  - 判断 Compare

### MI-0213 HNSW._search_layer(self, q, entry, ef, layer)
- 模块: `retrieval\hnsw.py`
- 用途: (无 docstring)  ·  INV 28
- 微指令步骤 (7):
  - 赋值 visited ← set(…)
  - 赋值 candidates ← ListComp
  - 赋值 results ← ListComp
  - 调用 heapq.heapify(…)
  - 调用 heapq.heapify(…)
  - 当 candidates 循环
  - 返回 ListComp

### MI-0214 HNSW._dist_id(self, q, i)
- 模块: `retrieval\hnsw.py`
- 用途: (无 docstring)  ·  INV 28
- 微指令步骤 (1):
  - 返回 float(…)

### MI-0215 HNSW.build(self, data)
- 模块: `retrieval\hnsw.py`
- 用途: (无 docstring)  ·  INV 28
- 微指令步骤 (2):
  - 遍历 v ∈ data
  - 返回 self

### MI-0216 HNSW.search(self, q, k)
- 模块: `retrieval\hnsw.py`
- 用途: (无 docstring)  ·  INV 28
- 微指令步骤 (4):
  - 判断 Compare
  - 赋值 res ← self._search_layer(…)
  - 赋值 out ← ListComp
  - 返回 sorted(…)[…]

### MI-0217 pairwise_cosine(vectors)
- 模块: `retrieval\mmr.py`
- 用途: 计算向量集合的两两余弦相似度矩阵。  ·  INV 49,50
- 微指令步骤 (7):
  - 赋值 norms ← np.linalg.norm(…)
  - 赋值 safe ← np.where(…)
  - 赋值 unit ← vectors div safe
  - 赋值 sim ← unit matmult unit.T
  - 赋值 zero_mask ← Compare.ravel(…)
  - 判断 zero_mask.any(…)
  - 返回 sim

### MI-0218 mmr_rerank(doc_ids, scores, vectors, top_k, lambda_mult, candidate_k)
- 模块: `retrieval\mmr.py`
- 用途: 对候选文档做 MMR 重排。  ·  INV 49,50  ·  复用 RRF
- 微指令步骤 (15):
  - 赋值 n ← len(…)
  - 判断 BoolOp
  - 判断 Compare
  - 赋值 pre_order ← sorted(…)[…]
  - 赋值 m ← len(…)
  - 赋值 k ← min(…)
  - 判断 Compare
  - 赋值 cand_ids ← ListComp
  - 赋值 cand_scores ← ListComp
  - 赋值 cand_vecs ← vectors[…]
  - 赋值 sim ← pairwise_cosine(…)
  - 声明并赋值 selected ← []
  - 赋值 remaining ← list(…)
  - 当 BoolOp 循环
  - 返回 ListComp

### MI-0219 avg_pairwise_similarity(vectors)
- 模块: `retrieval\mmr.py`
- 用途: 一组向量的平均两两余弦相似度（多样性度量，越小越多样）。  ·  INV 49,50
- 微指令步骤 (6):
  - 赋值 m ← len(…)
  - 判断 Compare
  - 赋值 sim ← pairwise_cosine(…)
  - 赋值 total ← float(…)
  - 赋值 pairs ← m mult m sub 1
  - 返回 IfExp

### MI-0220 【类】Hit.Hit(类)
- 模块: `retrieval\persist.py`
- 用途: 一条检索结果。  ·  INV 32  ·  复用 Cormack, RRF
- 微指令步骤 (6):
  - 声明并赋值 doc_id ← ∅
  - 声明并赋值 score ← ∅
  - 声明并赋值 text ← ""
  - 声明并赋值 meta ← field(…)
  - FunctionDef
  - FunctionDef

### MI-0221 Hit.key(self)
- 模块: `retrieval\persist.py`
- 用途: 用于一致性比对的键 (INV-32: id + 分数逐项相同)。  ·  INV 32
- 微指令步骤 (1):
  - 返回 [self.doc_id, float(…), self.text]

### MI-0222 Hit.to_dict(self)
- 模块: `retrieval\persist.py`
- 用途: (无 docstring)  ·  INV 32
- 微指令步骤 (1):
  - 返回 Dict

### MI-0223 【类】PersistentIndex.PersistentIndex(类)
- 模块: `retrieval\persist.py`
- 用途: BM25 + 稠密双路 RRF 融合索引, 支持落盘与重载 (INV-32)。  ·  INV 32  ·  复用 BM25, RRF, SSE
- 微指令步骤 (9):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0224 PersistentIndex.__init__(self, dim, seed, k1, b, rrf_k)
- 模块: `retrieval\persist.py`
- 用途: (无 docstring)  ·  INV 32
- 微指令步骤 (10):
  - 赋值 self.dim ← int(…)
  - 赋值 self.seed ← int(…)
  - 赋值 self.k1 ← float(…)
  - 赋值 self.b ← float(…)
  - 赋值 self.rrf_k ← int(…)
  - 声明并赋值 self.bm25 ← None
  - 声明并赋值 self.dense ← None
  - 声明并赋值 self.corpus ← []
  - 声明并赋值 self.texts ← []
  - 声明并赋值 self.metas ← []

### MI-0225 PersistentIndex.fit(self, corpus, texts, metas)
- 模块: `retrieval\persist.py`
- 用途: 用分词后的语料构建索引。  ·  INV 32
- 微指令步骤 (6):
  - 赋值 self.corpus ← ListComp
  - 赋值 self.texts ← IfExp
  - 赋值 self.metas ← IfExp
  - 赋值 self.bm25 ← Call.fit(…)
  - 赋值 self.dense ← Call.fit(…)
  - 返回 self

### MI-0226 PersistentIndex.search(self, query, top_k)
- 模块: `retrieval\persist.py`
- 用途: 双路检索 + RRF 融合, 返回 Top-K (INV-32 的比对对象)。  ·  INV 32  ·  复用 RRF
- 微指令步骤 (7):
  - 判断 BoolOp
  - 判断 UnaryOp
  - 赋值 pool ← max(…)
  - 赋值 bm ← self.bm25.search(…)
  - 赋值 dn ← self.dense.search(…)
  - 赋值 fused ← reciprocal_rank_fusion(…)
  - 返回 ListComp

### MI-0227 PersistentIndex.search_keys(self, query, top_k)
- 模块: `retrieval\persist.py`
- 用途: 检索结果的紧凑键 ``[(doc_id, score), ...]`` (便于一致性断言)。  ·  INV 32
- 微指令步骤 (1):
  - 返回 ListComp

### MI-0228 PersistentIndex.save(self, dir_path)
- 模块: `retrieval\persist.py`
- 用途: 落盘: ``index.json`` (稀疏侧状态 + 原文) + ``vectors.npz`` (稠密侧)。  ·  INV 32
- 微指令步骤 (11):
  - 判断 BoolOp
  - 调用 os.makedirs(…)
  - 赋值 state ← Dict
  - 赋值 idx_path ← os.path.join(…)
  - 上下文管理 (with)
  - 赋值 vec_path ← os.path.join(…)
  - 调用 np.savez(…)
  - 赋值 manifest ← Dict
  - 遍历 name ∈ ["index.json", "vectors.npz"]
  - 上下文管理 (with)
  - 返回 dir_path

### MI-0229 PersistentIndex._bm25_state(self)
- 模块: `retrieval\persist.py`
- 用途: 导出 BM25 的完整状态 (docs / df / idf / doc_len / avgdl)。  ·  INV 32  ·  复用 BM25
- 微指令步骤 (2):
  - Assert
  - 返回 Dict

### MI-0230 PersistentIndex.load(cls, dir_path, verify)
- 模块: `retrieval\persist.py`
- 用途: 重载索引; 重载后检索结果与落盘前 100% 一致 (INV-32)。  ·  INV 32  ·  复用 SSE
- 装饰器: classmethod
- 微指令步骤 (14):
  - 上下文管理 (with)
  - 判断 verify
  - 上下文管理 (with)
  - 赋值 obj ← cls(…)
  - 赋值 obj.corpus ← ListComp
  - 赋值 obj.texts ← list(…)
  - 赋值 obj.metas ← ListComp
  - 赋值 obj.bm25 ← cls._bm25_from_state(…)
  - 赋值 data ← np.load(…)
  - 赋值 dense ← DenseIndex(…)
  - 赋值 dense.docs ← ListComp
  - 赋值 dense.matrix ← np.asarray(…)
  - 赋值 obj.dense ← dense
  - 返回 obj

### MI-0231 PersistentIndex._bm25_from_state(st)
- 模块: `retrieval\persist.py`
- 用途: 按保存的状态逐字段重建 BM25 (不重新 fit, 保证分数逐位相同)。  ·  INV 32  ·  复用 BM25
- 装饰器: staticmethod
- 微指令步骤 (7):
  - 赋值 m ← BM25(…)
  - 赋值 m.docs ← ListComp
  - 赋值 m.doc_len ← ListComp
  - 赋值 m.avgdl ← float(…)
  - 赋值 m.df ← Counter(…)
  - 赋值 m.idf ← DictComp
  - 返回 m

### MI-0232 PersistentIndex.verify_reload(self, dir_path, queries, top_k)
- 模块: `retrieval\persist.py`
- 用途: 落盘 → 重载 → 逐 query 比对 Top-K, 返回一致性报告。  ·  INV 32
- 微指令步骤 (6):
  - 调用 self.save(…)
  - 赋值 reloaded ← PersistentIndex.load(…)
  - 声明并赋值 mismatches ← []
  - 赋值 worst ← 0.0
  - 遍历 (qi, q) ∈ enumerate(…)
  - 返回 Dict

### MI-0233 save_index(index, dir_path)
- 模块: `retrieval\persist.py`
- 用途: 函数式入口: 落盘。  ·  INV 32
- 微指令步骤 (1):
  - 返回 index.save(…)

### MI-0234 load_index(dir_path, verify)
- 模块: `retrieval\persist.py`
- 用途: 函数式入口: 重载 (INV-32)。  ·  INV 32
- 微指令步骤 (1):
  - 返回 PersistentIndex.load(…)

### MI-0235 verify_persistence(index, dir_path, queries, top_k)
- 模块: `retrieval\persist.py`
- 用途: 函数式入口: INV-32 一致性报告。  ·  INV 32
- 微指令步骤 (1):
  - 返回 index.verify_reload(…)

### MI-0236 【类】QFRWeights.QFRWeights(类)
- 模块: `retrieval\qfr.py`
- 用途: 各召回通道的可靠性先验 + 融合参数。  ·  INV 30,43,44,45,46,47,48  ·  复用 BM25, HNSW
- 微指令步骤 (9):
  - 声明并赋值 bm25 ← 1.0
  - 声明并赋值 dense ← 1.0
  - 声明并赋值 hnsw ← 1.0
  - 声明并赋值 recency ← 0.6
  - 声明并赋值 k ← 60
  - 声明并赋值 half_life_days ← 30.0
  - 声明并赋值 gamma ← 1.0
  - 声明并赋值 temporal_enabled ← True
  - FunctionDef

### MI-0237 QFRWeights.weight_for(self, channel)
- 模块: `retrieval\qfr.py`
- 用途: 按通道名取权重。未登记通道默认 1.0。  ·  INV 30,43,44,45,46,47,48
- 微指令步骤 (1):
  - 返回 float(…)

### MI-0238 temporal_factor(age_days, half_life_days, gamma)
- 模块: `retrieval\qfr.py`
- 用途: 时序衰减因子 T。  ·  INV 44
- 微指令步骤 (6):
  - 判断 Compare
  - 判断 Compare
  - 判断 Compare
  - 赋值 ratio ← age_days div half_life_days pow gamma
  - 判断 Compare
  - 返回 float(…)

### MI-0239 qfr_fuse(channels, weights, ages, valid_to, now_ms)
- 模块: `retrieval\qfr.py`
- 用途: 多通道 QFR 融合。  ·  INV 46,47  ·  复用 RRF
- 微指令步骤 (13):
  - Import
  - 赋值 w ← BoolOp
  - 赋值 ages ← BoolOp
  - 赋值 valid_to ← BoolOp
  - 赋值 now ← IfExp
  - 声明并赋值 expired ← set(…)
  - 遍历 (doc_id, vt) ∈ valid_to.items(…)
  - 声明并赋值 ranks ← Dict
  - 遍历 (channel, ranked) ∈ channels.items(…)
  - 声明并赋值 out ← []
  - 遍历 (doc_id, ch_ranks) ∈ ranks.items(…)
  - 调用 out.sort(…)
  - 返回 out

### MI-0240 qfr_reference(channels, k, weights_map, ages, half_life_days, gamma)
- 模块: `retrieval\qfr.py`
- 用途: QFR 的独立参考实现（暴力版，无剪枝、无去重优化）。  ·  INV 43  ·  复用 BM25
- 微指令步骤 (5):
  - 赋值 weights_map ← BoolOp
  - 赋值 ages ← BoolOp
  - 声明并赋值 scores ← Dict
  - 遍历 (channel, ranked) ∈ channels.items(…)
  - 返回 sorted(…)

### MI-0241 【类】CrossEncoderReranker.CrossEncoderReranker(类)
- 模块: `retrieval\rerank.py`
- 用途: (无 docstring)
- 微指令步骤 (4):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0242 CrossEncoderReranker.__init__(self, dim, seed)
- 模块: `retrieval\rerank.py`
- 用途: (无 docstring)
- 微指令步骤 (2):
  - 赋值 self.dim ← dim
  - 赋值 self.seed ← seed

### MI-0243 CrossEncoderReranker._token_vec(self, t)
- 模块: `retrieval\rerank.py`
- 用途: (无 docstring)
- 微指令步骤 (4):
  - 赋值 h ← stable_hash(…) bitand 4294967295 bitxor self.seed bitand 4294967295
  - 赋值 v ← Call.standard_normal(…)
  - 赋值 n ← np.linalg.norm(…)
  - 返回 IfExp

### MI-0244 CrossEncoderReranker._feat(self, tokens)
- 模块: `retrieval\rerank.py`
- 用途: (无 docstring)
- 微指令步骤 (4):
  - 赋值 v ← np.zeros(…)
  - 遍历 t ∈ tokens
  - 赋值 n ← np.linalg.norm(…)
  - 返回 IfExp

### MI-0245 CrossEncoderReranker.rerank(self, query, docs, top_k)
- 模块: `retrieval\rerank.py`
- 用途: (无 docstring)
- 微指令步骤 (4):
  - 赋值 qv ← self._feat(…)
  - 赋值 scored ← ListComp
  - 调用 scored.sort(…)
  - 返回 scored[…]

### MI-0246 reciprocal_rank_fusion(rank_lists, k)
- 模块: `retrieval\rrf.py`
- 用途: rank_lists: 多个检索器各自的(已排序)文档 id 列表。返回融合排序。  ·  INV 30
- 微指令步骤 (3):
  - 声明并赋值 scores ← Dict
  - 遍历 rl ∈ rank_lists
  - 返回 sorted(…)

## L6 智能体 (Agent)

### MI-0247 【类】Episode.Episode(类)
- 模块: `agent\memory.py`
- 用途: (无 docstring)
- 微指令步骤 (2):
  - 声明并赋值 ts ← ∅
  - 声明并赋值 text ← ∅

### MI-0248 【类】MemoryStore.MemoryStore(类)
- 模块: `agent\memory.py`
- 用途: (无 docstring)
- 微指令步骤 (6):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0249 MemoryStore.__init__(self)
- 模块: `agent\memory.py`
- 用途: (无 docstring)
- 微指令步骤 (2):
  - 声明并赋值 self.episodic ← []
  - 声明并赋值 self.semantic ← Dict

### MI-0250 MemoryStore.remember_event(self, text)
- 模块: `agent\memory.py`
- 用途: (无 docstring)
- 微指令步骤 (1):
  - 调用 self.episodic.append(…)

### MI-0251 MemoryStore.remember_fact(self, key, value)
- 模块: `agent\memory.py`
- 用途: (无 docstring)
- 微指令步骤 (1):
  - 赋值 self.semantic[…] ← value

### MI-0252 MemoryStore.recall_events(self, query, limit)
- 模块: `agent\memory.py`
- 用途: (无 docstring)
- 微指令步骤 (2):
  - 赋值 out ← ListComp
  - 返回 out[…]

### MI-0253 MemoryStore.recall_fact(self, key)
- 模块: `agent\memory.py`
- 用途: (无 docstring)
- 微指令步骤 (1):
  - 返回 self.semantic.get(…)

### MI-0254 MemoryStore.__len__(self)
- 模块: `agent\memory.py`
- 用途: (无 docstring)
- 微指令步骤 (1):
  - 返回 len(…) add len(…)

### MI-0255 【类】ReActAgent.ReActAgent(类)
- 模块: `agent\react.py`
- 用途: (无 docstring)
- 微指令步骤 (5):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0256 ReActAgent.__init__(self, llm_fn, tools, max_steps, memory)
- 模块: `agent\react.py`
- 用途: (无 docstring)
- 微指令步骤 (5):
  - 赋值 self.llm ← llm_fn
  - 赋值 self.tools ← tools
  - 赋值 self.max_steps ← max_steps
  - 赋值 self.memory ← memory
  - 声明并赋值 self.history ← []

### MI-0257 ReActAgent._system(self)
- 模块: `agent\react.py`
- 用途: (无 docstring)
- 微指令步骤 (4):
  - 赋值 lines ← ["你是一个使用工具的 ReAct …"]
  - 遍历 t ∈ self.tools.describe(…)
  - 调用 lines.append(…)
  - 返回 Constant.join(…)

### MI-0258 ReActAgent._parse_action(self, text)
- 模块: `agent\react.py`
- 用途: (无 docstring)
- 微指令步骤 (5):
  - 赋值 m ← re.search(…)
  - 判断 m
  - 赋值 m ← re.search(…)
  - 判断 m
  - 返回 ["none", None, text.strip(…)]

### MI-0259 ReActAgent._parse_args(raw)
- 模块: `agent\react.py`
- 用途: (无 docstring)
- 装饰器: staticmethod
- 微指令步骤 (4):
  - 赋值 args ← Dict
  - 判断 UnaryOp
  - 遍历 kv ∈ re.findall(…)
  - 返回 args

### MI-0260 ReActAgent.run(self, question, reflexion)
- 模块: `agent\react.py`
- 用途: (无 docstring)
- 微指令步骤 (4):
  - 赋值 self.history ← [Dict, Dict]
  - 遍历 step ∈ range(…)
  - 判断 reflexion
  - 返回 "未能在步数内完成。"

### MI-0261 【类】SafeCodeExecutor.SafeCodeExecutor(类)
- 模块: `agent\sandbox.py`
- 用途: (无 docstring)  ·  INV 34
- 微指令步骤 (3):
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0262 SafeCodeExecutor.__init__(self, timeout, workdir)
- 模块: `agent\sandbox.py`
- 用途: (无 docstring)  ·  INV 34
- 微指令步骤 (2):
  - 赋值 self.timeout ← timeout
  - 赋值 self.workdir ← BoolOp

### MI-0263 SafeCodeExecutor.static_check(self, code)
- 模块: `agent\sandbox.py`
- 用途: 返回违规列表 (非空 ⇒ 拒绝执行)。  ·  INV 34
- 微指令步骤 (4):
  - 声明并赋值 violations ← []
  - 异常保护 (try/except)
  - 遍历 node ∈ ast.walk(…)
  - 返回 violations

### MI-0264 SafeCodeExecutor.run(self, code)
- 模块: `agent\sandbox.py`
- 用途: (无 docstring)  ·  INV 34
- 微指令步骤 (4):
  - 赋值 v ← self.static_check(…)
  - 判断 v
  - 赋值 runner ← "import sys, json…"
  - 异常保护 (try/except)

### MI-0265 【类】ToolSpec.ToolSpec(类)
- 模块: `agent\tools.py`
- 用途: (无 docstring)  ·  INV 33
- 微指令步骤 (6):
  - 声明并赋值 name ← ∅
  - 声明并赋值 description ← ∅
  - 声明并赋值 schema ← ∅
  - 声明并赋值 func ← field(…)
  - FunctionDef
  - FunctionDef

### MI-0266 ToolSpec.validate(self, args)
- 模块: `agent\tools.py`
- 用途: (无 docstring)  ·  INV 33
- 微指令步骤 (1):
  - 返回 validate_args(…)

### MI-0267 ToolSpec.__call__(self, args)
- 模块: `agent\tools.py`
- 用途: (无 docstring)  ·  INV 33
- 微指令步骤 (3):
  - 赋值 errs ← self.validate(…)
  - 判断 errs
  - 返回 self.func(…)

### MI-0268 validate_args(schema, args)
- 模块: `agent\tools.py`
- 用途: 极简 JSON Schema 校验 (type / required / enum / minimum)。INV-33。  ·  INV 33
- 微指令步骤 (6):
  - 声明并赋值 errs ← []
  - 赋值 props ← schema.get(…)
  - 赋值 required ← schema.get(…)
  - 遍历 k ∈ required
  - 遍历 (k, v) ∈ args.items(…)
  - 返回 errs

### MI-0269 【类】Calculator.Calculator(类)
- 模块: `agent\tools.py`
- 用途: 安全算术计算器 (仅允许四则 + 括号, 经 AST 白名单)。  ·  INV 33
- 微指令步骤 (3):
  - 赋值 ALLOWED ← [ast.Expression, ast.BinOp, ast.UnaryOp]
  - FunctionDef
  - FunctionDef

### MI-0270 Calculator._eval(node, names)
- 模块: `agent\tools.py`
- 用途: (无 docstring)  ·  INV 33
- 装饰器: staticmethod
- 微指令步骤 (5):
  - 判断 isinstance(…)
  - 判断 isinstance(…)
  - 判断 isinstance(…)
  - 判断 isinstance(…)
  - 抛出异常

### MI-0271 Calculator.run(cls, expression)
- 模块: `agent\tools.py`
- 用途: (无 docstring)  ·  INV 33
- 装饰器: classmethod
- 微指令步骤 (3):
  - 赋值 tree ← ast.parse(…)
  - 遍历 node ∈ ast.walk(…)
  - 返回 float(…)

### MI-0272 【类】ToolRegistry.ToolRegistry(类)
- 模块: `agent\tools.py`
- 用途: (无 docstring)  ·  INV 33
- 微指令步骤 (5):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0273 ToolRegistry.__init__(self)
- 模块: `agent\tools.py`
- 用途: (无 docstring)  ·  INV 33
- 微指令步骤 (1):
  - 声明并赋值 self.tools ← Dict

### MI-0274 ToolRegistry.register(self, spec)
- 模块: `agent\tools.py`
- 用途: (无 docstring)  ·  INV 33
- 微指令步骤 (1):
  - 赋值 self.tools[…] ← spec

### MI-0275 ToolRegistry.get(self, name)
- 模块: `agent\tools.py`
- 用途: (无 docstring)  ·  INV 33
- 微指令步骤 (1):
  - 返回 self.tools.get(…)

### MI-0276 ToolRegistry.call(self, name, args)
- 模块: `agent\tools.py`
- 用途: (无 docstring)  ·  INV 33
- 微指令步骤 (2):
  - 赋值 spec ← self.tools[…]
  - 返回 spec(…)

### MI-0277 ToolRegistry.describe(self)
- 模块: `agent\tools.py`
- 用途: (无 docstring)  ·  INV 33
- 微指令步骤 (1):
  - 返回 ListComp

### MI-0278 default_registry()
- 模块: `agent\tools.py`
- 用途: (无 docstring)  ·  INV 33
- 微指令步骤 (5):
  - 赋值 reg ← ToolRegistry(…)
  - 调用 reg.register(…)
  - 赋值 sandbox ← SafeCodeExecutor(…)
  - 调用 reg.register(…)
  - 返回 reg

### MI-0279 _stable_hash(text)
- 模块: `agent\trace.py`
- 用途: 跨进程稳定的字符串哈希 (避免内置 ``hash()`` 的 PYTHONHASHSEED 随机化)。  ·  INV 35,40
- 微指令步骤 (1):
  - 返回 int(…)

### MI-0280 _extract_thought(raw)
- 模块: `agent\trace.py`
- 用途: 从 LLM 原始输出中抽取 ``Thought:`` 内容 (缺失时返回空串)。  ·  INV 35,40
- 微指令步骤 (2):
  - 赋值 m ← _THOUGHT_RE.search(…)
  - 返回 IfExp

### MI-0281 【类】TraceStep.TraceStep(类)
- 模块: `agent\trace.py`
- 用途: 一步 ReAct 记录: Thought / Action / Observation。  ·  INV 35,40  ·  复用 ReAct
- 微指令步骤 (10):
  - 声明并赋值 index ← ∅
  - 声明并赋值 thought ← ∅
  - 声明并赋值 action ← ∅
  - 声明并赋值 action_args ← field(…)
  - 声明并赋值 observation ← ""
  - 声明并赋值 raw ← ""
  - 声明并赋值 ts ← None
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0282 TraceStep.key(self)
- 模块: `agent\trace.py`
- 用途: 语义键 (排除时间戳), 用于逐步比对。  ·  INV 35,40
- 微指令步骤 (1):
  - 返回 [int(…), self.thought, self.action]

### MI-0283 TraceStep.to_dict(self)
- 模块: `agent\trace.py`
- 用途: (无 docstring)  ·  INV 35,40
- 微指令步骤 (1):
  - 返回 Dict

### MI-0284 TraceStep.from_dict(cls, d)
- 模块: `agent\trace.py`
- 用途: (无 docstring)  ·  INV 35,40
- 装饰器: classmethod
- 微指令步骤 (1):
  - 返回 cls(…)

### MI-0285 【类】Trace.Trace(类)
- 模块: `agent\trace.py`
- 用途: 一次 ReAct 运行的完整轨迹。  ·  INV 35,40  ·  复用 ReAct
- 微指令步骤 (12):
  - 声明并赋值 question ← ∅
  - 声明并赋值 seed ← 0
  - 声明并赋值 steps ← field(…)
  - 声明并赋值 final_answer ← ""
  - 声明并赋值 trace_id ← ""
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0286 Trace.__len__(self)
- 模块: `agent\trace.py`
- 用途: (无 docstring)  ·  INV 35,40
- 微指令步骤 (1):
  - 返回 len(…)

### MI-0287 Trace.fingerprint(self)
- 模块: `agent\trace.py`
- 用途: 轨迹指纹: 逐步语义键的 sha256 (INV-35 的主判据)。  ·  INV 35
- 微指令步骤 (2):
  - 赋值 payload ← json.dumps(…)
  - 返回 Call.hexdigest(…)

### MI-0288 Trace.same_steps(self, other)
- 模块: `agent\trace.py`
- 用途: 逐步比对 (排除时间戳), 与 :meth:`fingerprint` 互为双保险。  ·  INV 35,40
- 微指令步骤 (2):
  - 判断 Compare
  - 返回 all(…)

### MI-0289 Trace.to_dict(self)
- 模块: `agent\trace.py`
- 用途: (无 docstring)  ·  INV 35,40
- 微指令步骤 (1):
  - 返回 Dict

### MI-0290 Trace.from_dict(cls, d)
- 模块: `agent\trace.py`
- 用途: (无 docstring)  ·  INV 35,40
- 装饰器: classmethod
- 微指令步骤 (1):
  - 返回 cls(…)

### MI-0291 Trace.to_json(self, path)
- 模块: `agent\trace.py`
- 用途: 序列化为 JSON (``path`` 给出时同时落盘)。  ·  INV 35,40
- 微指令步骤 (3):
  - 赋值 s ← json.dumps(…)
  - 判断 path
  - 返回 s

### MI-0292 Trace.from_json(cls, source)
- 模块: `agent\trace.py`
- 用途: 从 JSON 字符串或文件路径重建轨迹。  ·  INV 35,40
- 装饰器: classmethod
- 微指令步骤 (2):
  - 判断 BoolOp
  - 返回 cls.from_dict(…)

### MI-0293 【类】SeededLLM.SeededLLM(类)
- 模块: `agent\trace.py`
- 用途: 确定性 LLM 存根: 由 ``(seed, 步序, 上下文)`` 经 PCG64 决定 Thought/Action。  ·  INV 35,40
- 微指令步骤 (4):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0294 SeededLLM.__init__(self, tools, seed, max_steps, finish_prob)
- 模块: `agent\trace.py`
- 用途: (无 docstring)  ·  INV 35,40
- 微指令步骤 (4):
  - 赋值 self.tools ← tools
  - 赋值 self.seed ← int(…)
  - 赋值 self.max_steps ← int(…)
  - 赋值 self.finish_prob ← float(…)

### MI-0295 SeededLLM.__call__(self, messages)
- 模块: `agent\trace.py`
- 用途: (无 docstring)  ·  INV 35,40
- 微指令步骤 (13):
  - 赋值 step ← sum(…)
  - 赋值 ctx ← IfExp
  - 赋值 r ← np.random.default_rng(…)
  - 赋值 specs ← self.tools.describe(…)
  - 赋值 name ← IfExp
  - 赋值 thought ← JoinedStr
  - 赋值 last_obs ← ""
  - 遍历 m ∈ reversed(…)
  - 判断 BoolOp
  - 赋值 spec ← self.tools.get(…)
  - 赋值 args ← IfExp
  - 赋值 args_str ← Constant.join(…)
  - 返回 JoinedStr

### MI-0296 SeededLLM._answer(last_obs)
- 模块: `agent\trace.py`
- 用途: 由最近一次 Observation 生成确定性最终答案。  ·  INV 35,40
- 装饰器: staticmethod
- 微指令步骤 (2):
  - 赋值 m ← re.search(…)
  - 返回 BoolOp

### MI-0297 SeededLLM._plan_args(schema, r, ctx)
- 模块: `agent\trace.py`
- 用途: 按 JSON Schema 生成确定性参数 (INV-33 保证可被校验通过)。  ·  INV 33
- 装饰器: staticmethod
- 微指令步骤 (4):
  - 声明并赋值 args ← Dict
  - 赋值 props ← schema.get(…)
  - 遍历 (key, spec) ∈ props.items(…)
  - 返回 args

### MI-0298 【类】TracedReActAgent.TracedReActAgent(类)
- 模块: `agent\trace.py`
- 用途: ReAct 智能体的"带轨迹"版本: 逐步记录 Thought/Action/Observation (INV-35)。  ·  INV 35  ·  复用 ReAct
- 微指令步骤 (2):
  - FunctionDef
  - FunctionDef

### MI-0299 TracedReActAgent.__init__(self, llm_fn, tools, max_steps, memory)
- 模块: `agent\trace.py`
- 用途: (无 docstring)  ·  INV 35,40
- 微指令步骤 (2):
  - 调用 Call.__init__(…)
  - 声明并赋值 self.trace ← None

### MI-0300 TracedReActAgent.run_traced(self, question, seed, reflexion)
- 模块: `agent\trace.py`
- 用途: 运行并逐步记录轨迹。  ·  INV 35,40  ·  复用 Reflexion, Shinn
- 微指令步骤 (8):
  - 赋值 self.history ← [Dict, Dict]
  - 声明并赋值 steps ← []
  - 赋值 final ← ""
  - 遍历 step ∈ range(…)
  - 判断 BoolOp
  - 赋值 trace ← Trace(…)
  - 赋值 self.trace ← trace
  - 返回 trace

### MI-0301 replay_trace(trace, tools, max_steps)
- 模块: `agent\trace.py`
- 用途: 按轨迹记录的 Action 重放 (不再让 LLM 重新决策), 返回重放轨迹。  ·  INV 35
- 微指令步骤 (4):
  - 声明并赋值 script ← DictComp
  - FunctionDef
  - 赋值 agent ← TracedReActAgent(…)
  - 返回 agent.run_traced(…)

### MI-0302 compare_traces(a, b)
- 模块: `agent\trace.py`
- 用途: INV-35 判据: 两条轨迹是否 100% 一致 (指纹 + 逐步双判据)。  ·  INV 35
- 微指令步骤 (1):
  - 返回 BoolOp

### MI-0303 record_and_replay(question, tools, seed, max_steps)
- 模块: `agent\trace.py`
- 用途: 一键自检: 同 seed 录制 → 重放 → JSON 往返, 返回一致性报告。  ·  INV 35,40
- 微指令步骤 (7):
  - 赋值 agent ← TracedReActAgent(…)
  - 赋值 original ← agent.run_traced(…)
  - 赋值 again ← Call.run_traced(…)
  - 赋值 replayed ← replay_trace(…)
  - 赋值 restored ← Trace.from_json(…)
  - 赋值 fps ← SetComp
  - 返回 Dict

## L7 服务/编排 (Service/Orchestration)

### MI-0304 format_messages(messages)
- 模块: `serve\api.py`
- 用途: 将 OpenAI chat messages 拼接为单条 prompt (逐行 "role: content")。  ·  INV 36,37,38  ·  复用 OpenAI
- 微指令步骤 (1):
  - 返回 Constant.join(…)

### MI-0305 chat_completion(text, model_name)
- 模块: `serve\api.py`
- 用途: 构造 OpenAI 严格的 ``chat.completion`` 响应体 (INV-36)。  ·  INV 36,37  ·  复用 OpenAI
- 微指令步骤 (2):
  - 赋值 n ← len(…)
  - 返回 Dict

### MI-0306 _guard_error(verdict)
- 模块: `serve\api.py`
- 用途: 把护栏裁决转成 OpenAI 风格错误体。  ·  INV 36,37,38  ·  复用 OpenAI
- 微指令步骤 (1):
  - 返回 Dict

### MI-0307 【类】OpenAIServer.OpenAIServer(类)
- 模块: `serve\api.py`
- 用途: (无 docstring)  ·  INV 36,37,38
- 微指令步骤 (11):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0308 OpenAIServer.__init__(self, model, tokenizer, max_new_tokens, seed, guardrail, engine_backend)
- 模块: `serve\api.py`
- 用途: (无 docstring)  ·  INV 36,37,38
- 微指令步骤 (8):
  - 赋值 self.model ← model
  - 赋值 self.tok ← tokenizer
  - 赋值 self.max_new_tokens ← max_new_tokens
  - 赋值 self.sampler ← Sampler(…)
  - 赋值 self.model_name ← "ifai-flagship"
  - 赋值 self.guardrail ← guardrail
  - 赋值 self.engine_backend ← engine_backend
  - 赋值 self._engine ← None

### MI-0309 OpenAIServer.engine(self)
- 模块: `serve\api.py`
- 用途: 惰性构造 :class:`ifai.engine.session.EngineSession` (L3 五件套入口)。  ·  INV 36,37,38
- 装饰器: property
- 微指令步骤 (2):
  - 判断 Compare
  - 返回 self._engine

### MI-0310 OpenAIServer._gen_tokens(self, prompt, max_new, temperature, top_p, stream)
- 模块: `serve\api.py`
- 用途: 生成 token 迭代器。  ·  INV 37  ·  复用 SSE
- 微指令步骤 (5):
  - 赋值 ids ← self.tok.encode(…)
  - 判断 UnaryOp
  - 赋值 eos_id ← self.tok.token2id.get(…)
  - 赋值 new_ids ← self.engine.generate(…)
  - 遍历 nxt ∈ new_ids

### MI-0311 OpenAIServer._guard_input(self, prompt)
- 模块: `serve\api.py`
- 用途: 输入护栏: 命中则返回 OpenAI 风格错误体, 否则 None。  ·  INV 36,37,38  ·  复用 OpenAI
- 微指令步骤 (4):
  - 判断 Compare
  - 赋值 verdict ← self.guardrail.check_input(…)
  - 判断 verdict.blocked
  - 返回 None

### MI-0312 OpenAIServer._guard_output(self, text)
- 模块: `serve\api.py`
- 用途: 输出护栏: 命中则返回 OpenAI 风格错误体, 否则 None。  ·  INV 36,37,38  ·  复用 OpenAI
- 微指令步骤 (4):
  - 判断 Compare
  - 赋值 verdict ← self.guardrail.check_output(…)
  - 判断 verdict.blocked
  - 返回 None

### MI-0313 OpenAIServer.completions(self, prompt, temperature, top_p, max_tokens, stream)
- 模块: `serve\api.py`
- 用途: (无 docstring)  ·  INV 36,37,38
- 微指令步骤 (4):
  - 赋值 max_new ← BoolOp
  - 赋值 text ← Constant.join(…)
  - 赋值 obj ← Dict
  - 返回 obj

### MI-0314 OpenAIServer.chat(self, messages, **kw)
- 模块: `serve\api.py`
- 用途: (无 docstring)  ·  INV 36,37,38
- 微指令步骤 (3):
  - 赋值 prompt ← format_messages(…)
  - 赋值 comp ← self.completions(…)
  - 返回 chat_completion(…)

### MI-0315 OpenAIServer.stream_chat(self, messages, temperature, top_p, max_tokens)
- 模块: `serve\api.py`
- 用途: 以 SSE 事件流的形式产出 chat 增量 (v0.8.0 闭合 G1)。  ·  INV 36,37,38  ·  复用 OpenAI, SSE
- 微指令步骤 (4):
  - 赋值 prompt ← format_messages(…)
  - 赋值 max_new ← BoolOp
  - 赋值 tokens ← self._gen_tokens(…)
  - 表达式 YieldFrom

### MI-0316 OpenAIServer.stream_completions(self, prompt, temperature, top_p, max_tokens)
- 模块: `serve\api.py`
- 用途: 以 SSE 事件流的形式产出 completions 增量 (v0.8.0 闭合 G1)。  ·  INV 36,37,38  ·  复用 SSE
- 微指令步骤 (3):
  - 赋值 max_new ← BoolOp
  - 赋值 tokens ← self._gen_tokens(…)
  - 表达式 YieldFrom

### MI-0317 OpenAIServer.models(self)
- 模块: `serve\api.py`
- 用途: (无 docstring)  ·  INV 36,37,38
- 微指令步骤 (1):
  - 返回 Dict

### MI-0318 OpenAIServer.route(self, method, path, body)
- 模块: `serve\api.py`
- 用途: (无 docstring)  ·  INV 36,37,38
- 微指令步骤 (4):
  - 判断 BoolOp
  - 判断 BoolOp
  - 判断 BoolOp
  - 返回 [404, Dict]

### MI-0319 create_handler(server)
- 模块: `serve\api.py`
- 用途: (无 docstring)  ·  INV 36,37,38
- 微指令步骤 (2):
  - ClassDef
  - 返回 Handler

### MI-0320 serve(server, host, port)
- 模块: `serve\api.py`
- 用途: (无 docstring)  ·  INV 36,37,38
- 微指令步骤 (2):
  - 赋值 httpd ← ThreadingHTTPServer(…)
  - 调用 httpd.serve_forever(…)

### MI-0321 normalize(text, leet)
- 模块: `serve\guardrail.py`
- 用途: 归一化: 去零宽字符 → NFKC → 小写 → leetspeak 还原 → 折叠空白。  ·  INV 38
- 微指令步骤 (5):
  - 赋值 t ← str(…)
  - 遍历 ch ∈ _ZERO_WIDTH
  - 赋值 t ← Call.lower(…)
  - 判断 leet
  - 返回 Call.strip(…)

### MI-0322 【类】Rule.Rule(类)
- 模块: `serve\guardrail.py`
- 用途: 一条护栏规则。  ·  INV 38
- 微指令步骤 (5):
  - 声明并赋值 category ← ∅
  - 声明并赋值 severity ← ∅
  - 声明并赋值 pattern ← ∅
  - 声明并赋值 reason ← ∅
  - 声明并赋值 on_raw ← False

### MI-0323 _rule(category, severity, pattern, reason, on_raw)
- 模块: `serve\guardrail.py`
- 用途: (无 docstring)  ·  INV 38
- 微指令步骤 (1):
  - 返回 Rule(…)

### MI-0324 【类】GuardrailVerdict.GuardrailVerdict(类)
- 模块: `serve\guardrail.py`
- 用途: 护栏判定结果 (结构化, 便于审计与分级处置)。  ·  INV 38
- 微指令步骤 (8):
  - 声明并赋值 allowed ← ∅
  - 声明并赋值 risk ← 0.0
  - 声明并赋值 categories ← field(…)
  - 声明并赋值 reasons ← field(…)
  - 声明并赋值 sanitized_text ← ""
  - 声明并赋值 stage ← "input"
  - FunctionDef
  - FunctionDef

### MI-0325 GuardrailVerdict.blocked(self)
- 模块: `serve\guardrail.py`
- 用途: 是否拦截 (``allowed`` 的反义, 便于断言)。  ·  INV 38
- 装饰器: property
- 微指令步骤 (1):
  - 返回 UnaryOp

### MI-0326 GuardrailVerdict.to_dict(self)
- 模块: `serve\guardrail.py`
- 用途: (无 docstring)  ·  INV 38
- 微指令步骤 (1):
  - 返回 Dict

### MI-0327 【类】Guardrail.Guardrail(类)
- 模块: `serve\guardrail.py`
- 用途: 规则护栏: 输入注入/有害检测 + 输出泄露/PII 过滤 (INV-38)。  ·  INV 38
- 微指令步骤 (5):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0328 Guardrail.__init__(self, max_prompt_chars, min_block_severity, enable_leet, extra_rules)
- 模块: `serve\guardrail.py`
- 用途: (无 docstring)  ·  INV 38
- 微指令步骤 (5):
  - 赋值 self.max_prompt_chars ← int(…)
  - 赋值 self.min_block_severity ← float(…)
  - 赋值 self.enable_leet ← bool(…)
  - 声明并赋值 self.input_rules ← list(…) add list(…)
  - 声明并赋值 self.output_rules ← list(…)

### MI-0329 Guardrail.check_input(self, text)
- 模块: `serve\guardrail.py`
- 用途: 对用户输入做护栏检查 (INV-38 的红队/误杀统计口径)。  ·  INV 38
- 微指令步骤 (9):
  - 赋值 raw ← str(…)
  - 赋值 norms ← IfExp
  - 声明并赋值 cats ← []
  - 声明并赋值 reasons ← []
  - 赋值 worst ← 0.0
  - 遍历 r ∈ self.input_rules
  - 判断 Compare
  - 赋值 risk ← min(…)
  - 返回 GuardrailVerdict(…)

### MI-0330 Guardrail.check_output(self, text)
- 模块: `serve\guardrail.py`
- 用途: 对模型输出做泄露/PII 过滤, 并给出脱敏后的文本。  ·  INV 38
- 微指令步骤 (10):
  - 赋值 raw ← str(…)
  - 赋值 norms ← IfExp
  - 声明并赋值 cats ← []
  - 声明并赋值 reasons ← []
  - 赋值 worst ← 0.0
  - 遍历 r ∈ self.output_rules
  - 赋值 masked ← raw
  - 遍历 (pat, repl) ∈ _PII_MASK
  - 赋值 risk ← min(…)
  - 返回 GuardrailVerdict(…)

### MI-0331 Guardrail.check(self, text)
- 模块: `serve\guardrail.py`
- 用途: :meth:`check_input` 的别名 (统一入口)。  ·  INV 38
- 微指令步骤 (1):
  - 返回 self.check_input(…)

### MI-0332 Guardrail.guard(self, fn)
- 模块: `serve\guardrail.py`
- 用途: 装饰器: 包裹生成函数, 返回 ``(allowed, text_or_reason)``。  ·  INV 38
- 微指令步骤 (2):
  - FunctionDef
  - 返回 wrapped

### MI-0333 _repetition_flood(text, threshold)
- 模块: `serve\guardrail.py`
- 用途: 检测重复字符洪水 (同一字符连续出现 ``threshold`` 次以上)。  ·  INV 38
- 微指令步骤 (3):
  - 判断 UnaryOp
  - 赋值 m ← re.search(…)
  - 返回 Compare

### MI-0334 load_cases(path)
- 模块: `serve\guardrail.py`
- 用途: 加载 jsonl 用例集 (每行一个 JSON 对象, 含 ``text`` 字段)。  ·  INV 38
- 微指令步骤 (3):
  - 声明并赋值 cases ← []
  - 上下文管理 (with)
  - 返回 cases

### MI-0335 evaluate(guardrail, redteam, benign)
- 模块: `serve\guardrail.py`
- 用途: INV-38 统计: 红队拦截率与正常提示误杀率。  ·  INV 38  ·  复用 SSE
- 微指令步骤 (7):
  - 赋值 red ← IfExp
  - 赋值 ben ← IfExp
  - 声明并赋值 missed ← []
  - 遍历 c ∈ red
  - 声明并赋值 fps ← []
  - 遍历 c ∈ ben
  - 返回 Dict

### MI-0336 default_guardrail()
- 模块: `serve\guardrail.py`
- 用途: 返回默认配置护栏 (供 L7 服务与 L9 CLI 直接复用)。  ·  INV 38
- 微指令步骤 (1):
  - 返回 Guardrail(…)

### MI-0337 redteam_path()
- 模块: `serve\guardrail.py`
- 用途: 内置红队数据集路径 (100 条)。  ·  INV 38
- 微指令步骤 (1):
  - 返回 os.path.join(…)

### MI-0338 benign_path()
- 模块: `serve\guardrail.py`
- 用途: 内置正常提示数据集路径 (100 条)。  ·  INV 38
- 微指令步骤 (1):
  - 返回 os.path.join(…)

### MI-0339 sse_headers()
- 模块: `serve\sse.py`
- 用途: 返回标准 SSE 响应头。  ·  INV 37  ·  复用 SSE
- 微指令步骤 (1):
  - 返回 Dict

### MI-0340 encode_sse_event(data, event)
- 模块: `serve\sse.py`
- 用途: 把一条负载编码为 SSE 事件文本 (以空行结尾)。  ·  INV 37  ·  复用 SSE
- 微指令步骤 (2):
  - 赋值 prefix ← IfExp
  - 返回 JoinedStr

### MI-0341 sse_chat_chunk(content, model, chunk_id, created, finish_reason)
- 模块: `serve\sse.py`
- 用途: 构造一个 OpenAI 兼容的流式分块负载 (JSON 字符串)。  ·  INV 37  ·  复用 OpenAI, SSE
- 微指令步骤 (3):
  - Import
  - 赋值 payload ← Dict
  - 返回 json.dumps(…)

### MI-0342 sse_chunks(tokens, model, chunk_id, created)
- 模块: `serve\sse.py`
- 用途: 把 token 迭代器编码为完整 SSE 事件流。  ·  INV 37  ·  复用 SSE
- 微指令步骤 (4):
  - 赋值 ts ← IfExp
  - 遍历 tok ∈ tokens
  - 表达式 Yield
  - 表达式 Yield

### MI-0343 iter_sse_lines(stream)
- 模块: `serve\sse.py`
- 用途: 把 SSE 字节流拆成逐行 (按 ``\n`` 切分, 去掉尾部 ``\r``)。  ·  INV 37  ·  复用 SSE
- 微指令步骤 (2):
  - 判断 isinstance(…)
  - 遍历 line ∈ text.split(…)

### MI-0344 parse_sse_stream(stream)
- 模块: `serve\sse.py`
- 用途: 解析 SSE 字节流为事件字典列表。  ·  INV 37  ·  复用 SSE
- 微指令步骤 (6):
  - 声明并赋值 events ← []
  - 声明并赋值 name ← None
  - 声明并赋值 data_lines ← []
  - 遍历 line ∈ iter_sse_lines(…)
  - 判断 data_lines
  - 返回 events

### MI-0345 collect_sse_content(stream)
- 模块: `serve\sse.py`
- 用途: 从 SSE 字节流中收集全部增量文本并拼接。  ·  INV 37  ·  复用 SSE
- 微指令步骤 (3):
  - 声明并赋值 parts ← []
  - 遍历 ev ∈ parse_sse_stream(…)
  - 返回 Constant.join(…)

## L8 评估/对齐 (Evaluation/Alignment)

### MI-0346 benchmark_generation(generate_fn, prompt, max_new_tokens, runs)
- 模块: `eval\benchmark.py`
- 用途: generate_fn(prompt, n) -> 生成的 token 数; 返回吞吐统计。
- 微指令步骤 (6):
  - 赋值 (times, toks) ← [[], []]
  - 遍历 _ ∈ range(…)
  - 赋值 tps ← ListComp
  - 赋值 arr ← np.array(…)
  - 赋值 cv ← IfExp
  - 返回 Dict

### MI-0347 write_baseline(path, stats)
- 模块: `eval\benchmark.py`
- 用途: (无 docstring)
- 微指令步骤 (2):
  - 赋值 lines ← ["# 性能基线 (BASELINE…", "", JoinedStr]
  - 上下文管理 (with)

### MI-0348 【类】GSM8K_Mini.GSM8K_Mini(类)
- 模块: `eval\harness.py`
- 用途: 内置 20 道极简算术题 (离线, 不随仓下载)。
- 微指令步骤 (3):
  - 赋值 DATA ← [["小明有 3 个苹果, 又买了 5…", 8], ["7 减去 2 等于多少?", 5], ["一个盒子有 4 排, 每排 6 …", 24]]
  - FunctionDef
  - FunctionDef

### MI-0349 GSM8K_Mini.extract_number(text)
- 模块: `eval\harness.py`
- 用途: (无 docstring)
- 装饰器: staticmethod
- 微指令步骤 (2):
  - 赋值 nums ← re.findall(…)
  - 返回 IfExp

### MI-0350 GSM8K_Mini.evaluate(self, answer_fn)
- 模块: `eval\harness.py`
- 用途: (无 docstring)
- 微指令步骤 (4):
  - 赋值 correct ← 0
  - 赋值 samples ← []
  - 遍历 (q, gold) ∈ self.DATA
  - 返回 Dict

### MI-0351 【类】RegressionBaseline.RegressionBaseline(类)
- 模块: `eval\harness.py`
- 用途: 回归门禁: 当前吞吐相对基线掉落 >15% ⇒ 红灯。
- 微指令步骤 (2):
  - FunctionDef
  - FunctionDef

### MI-0352 RegressionBaseline.parse_baseline(path)
- 模块: `eval\harness.py`
- 用途: 读取基线。
- 装饰器: staticmethod
- 微指令步骤 (5):
  - 声明并赋值 out ← Dict
  - 赋值 json_path ← IfExp
  - 判断 os.path.isfile(…)
  - 异常保护 (try/except)
  - 返回 out

### MI-0353 RegressionBaseline.check(current, baseline_path, threshold)
- 模块: `eval\harness.py`
- 用途: 比对当前指标与基线。
- 装饰器: staticmethod
- 微指令步骤 (5):
  - 赋值 base ← RegressionBaseline.parse_baseline(…)
  - 判断 Compare
  - 赋值 drop ← base[…] sub current[…] div base[…]
  - 赋值 status ← IfExp
  - 返回 Dict

### MI-0354 perplexity(model, token_ids)
- 模块: `eval\perplexity.py`
- 用途: 给定 token 序列计算困惑度 (逐 token 用 logits_next, 因果)。  ·  INV 39
- 微指令步骤 (6):
  - 赋值 ids ← list(…)
  - 赋值 total_nll ← 0.0
  - 赋值 n ← 0
  - 遍历 i ∈ range(…)
  - 赋值 mean_nll ← total_nll div max(…)
  - 返回 float(…)

### MI-0355 perplexity_manual(model, token_ids)
- 模块: `eval\perplexity.py`
- 用途: 纯手工 exp(mean NLL) 参考 (INV-39 对照)。  ·  INV 39
- 微指令步骤 (5):
  - 赋值 ids ← list(…)
  - 赋值 nll ← 0.0
  - 赋值 n ← 0
  - 遍历 i ∈ range(…)
  - 返回 float(…)

## L9 应用/接口 (Application/Interface)

### MI-0356 【类】IFAI.IFAI(类)
- 模块: `sdk.py`
- 用途: 旗舰 AI 高层客户端。
- 微指令步骤 (6):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0357 IFAI.__init__(self, model_dir, seed)
- 模块: `sdk.py`
- 用途: 载入模型与分词器。
- 微指令步骤 (5):
  - 调用 rng.seed(…)
  - 赋值 tok_path ← os.path.join(…)
  - 赋值 self.tok ← Tokenizer.from_json(…)
  - 赋值 self.model ← load_model(…)
  - 赋值 self.sampler ← Sampler(…)

### MI-0358 IFAI._generate(self, prompt, max_new, temperature)
- 模块: `sdk.py`
- 用途: 内部单一生成原语: 逐 token 前向 + 贪心/采样, 返回续写文本。  ·  复用 OpenAI
- 微指令步骤 (4):
  - 赋值 ids ← BoolOp
  - 赋值 out ← []
  - 遍历 _ ∈ range(…)
  - 返回 Constant.join(…)

### MI-0359 IFAI.chat(self, prompt, max_new, temperature)
- 模块: `sdk.py`
- 用途: 生成续写文本 (贪心/采样)。temperature=0 走贪心 (可复现)。
- 微指令步骤 (1):
  - 返回 BoolOp

### MI-0360 IFAI.chat_completion(self, messages, max_new, temperature)
- 模块: `sdk.py`
- 用途: OpenAI 严格兼容的 chat 接口, 返回 ``chat.completion`` dict (INV-36)。  ·  INV 36,37  ·  复用 OpenAI
- 微指令步骤 (2):
  - 赋值 text ← self._generate(…)
  - 返回 chat_completion(…)

### MI-0361 IFAI.search(self, query, corpus, top_k)
- 模块: `sdk.py`
- 用途: 混合检索 (BM25 + 稠密向量 + RRF 融合, Cormack et al. 2009)。  ·  复用 BM25, Cormack, RRF
- 微指令步骤 (2):
  - 赋值 idx ← Call.fit(…)
  - 返回 ListComp

### MI-0362 IFAI.run(self, question, max_steps)
- 模块: `sdk.py`
- 用途: 运行 ReAct 智能体 (内置计算器 + 安全沙箱工具, 记忆 + Reflexion)。  ·  复用 ReAct, Reflexion
- 微指令步骤 (3):
  - FunctionDef
  - 赋值 agent ← ReActAgent(…)
  - 返回 agent.run(…)

## L10 微指令 ISA (Micro-Instruction ISA)

### MI-0363 【类】MicroOp.MicroOp(类)
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (5):
  - 声明并赋值 family ← ∅
  - 声明并赋值 name ← ∅
  - 声明并赋值 arity ← ∅
  - 声明并赋值 fn ← ∅
  - 声明并赋值 note ← ""

### MI-0364 _reg(family, arity, table, note, flat)
- 模块: `l10\_microops.py`
- 用途: flat=True 的家族：成员语义上是「一维序列算子」，  ·  INV 8,9
- 微指令步骤 (3):
  - 赋值 out ← Dict
  - 遍历 (name, fn) ∈ table.items(…)
  - 返回 out

### MI-0365 _uact()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (3):
  - 赋值 e ← np.exp
  - 赋值 s ← Lambda
  - 返回 _reg(…)

### MI-0366 _bact()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (1):
  - 返回 _reg(…)

### MI-0367 _ract()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (1):
  - 返回 _reg(…)

### MI-0368 _norm()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (2):
  - FunctionDef
  - 返回 _reg(…)

### MI-0369 _smax()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (7):
  - FunctionDef
  - 赋值 tbl ← Dict
  - 遍历 (i, t) ∈ enumerate(…)
  - 遍历 k ∈ range(…)
  - 赋值 tbl[…] ← Lambda
  - 赋值 tbl[…] ← Lambda
  - 返回 _reg(…)

### MI-0370 _rope()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (13):
  - 赋值 tbl ← Dict
  - FunctionDef
  - 遍历 b ∈ [100, 500, 1000]
  - 赋值 tbl[…] ← Lambda
  - 赋值 tbl[…] ← Lambda
  - 赋值 tbl[…] ← Lambda
  - 赋值 tbl[…] ← Lambda
  - 赋值 tbl[…] ← Lambda
  - 赋值 tbl[…] ← Lambda
  - 赋值 tbl[…] ← Lambda
  - 遍历 k ∈ range(…)
  - 遍历 k ∈ range(…)
  - 返回 _reg(…)

### MI-0371 _mask()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (3):
  - 赋值 tbl ← Dict
  - 遍历 k ∈ range(…)
  - 返回 _reg(…)

### MI-0372 _attn()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (4):
  - FunctionDef
  - 赋值 tbl ← Dict
  - 遍历 k ∈ range(…)
  - 返回 _reg(…)

### MI-0373 _gate()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (3):
  - 赋值 s ← Lambda
  - 赋值 tbl ← Dict
  - 返回 _reg(…)

### MI-0374 _moer()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (3):
  - 赋值 tbl ← Dict
  - 遍历 k ∈ range(…)
  - 返回 _reg(…)

### MI-0375 _loss()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (3):
  - 赋值 tbl ← Dict
  - 遍历 p ∈ [1.5, 2.5, 3]
  - 返回 _reg(…)

### MI-0376 _metr()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (3):
  - 赋值 tbl ← Dict
  - 遍历 k ∈ range(…)
  - 返回 _reg(…)

### MI-0377 _initf()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (5):
  - 赋值 tbl ← Dict
  - FunctionDef
  - 遍历 (i, k) ∈ enumerate(…)
  - 遍历 s ∈ [0.001, 0.002, 0.005]
  - 返回 _reg(…)

### MI-0378 _scan()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (5):
  - 赋值 tbl ← Dict
  - 遍历 a ∈ [0.1, 0.2, 0.3]
  - 遍历 a ∈ [0.1, 0.2, 0.3]
  - 当 Compare 循环
  - 返回 _reg(…)

### MI-0379 _quant()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (5):
  - 赋值 tbl ← Dict
  - 遍历 b ∈ [2, 3, 5]
  - 遍历 g ∈ [4, 8, 16]
  - 当 Compare 循环
  - 返回 _reg(…)

### MI-0380 _smpl()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (3):
  - 赋值 tbl ← Dict
  - 遍历 t ∈ [0.1, 0.2, 0.3]
  - 返回 _reg(…)

### MI-0381 _optm()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (5):
  - 赋值 tbl ← Dict
  - 遍历 lr ∈ [0.0001, 0.0003, 0.001]
  - 遍历 wd ∈ [0.0001, 0.001, 0.01]
  - 当 Compare 循环
  - 返回 _reg(…)

### MI-0382 _grad()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (4):
  - 赋值 s ← Lambda
  - 赋值 tbl ← Dict
  - 遍历 k ∈ range(…)
  - 返回 _reg(…)

### MI-0383 _dist()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (15):
  - FunctionDef
  - 赋值 tbl ← Dict
  - 遍历 p ∈ [0.25, 0.5, 0.75]
  - 赋值 tbl[…] ← Lambda
  - 赋值 tbl[…] ← Lambda
  - 赋值 tbl[…] ← Lambda
  - 赋值 tbl[…] ← Lambda
  - 赋值 tbl[…] ← Lambda
  - 赋值 tbl[…] ← Lambda
  - 赋值 tbl[…] ← Lambda
  - 赋值 tbl[…] ← Lambda
  - 赋值 tbl[…] ← Lambda
  - 赋值 tbl[…] ← Lambda
  - 当 Compare 循环
  - 返回 _reg(…)

### MI-0384 _prob()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (5):
  - FunctionDef
  - 赋值 tbl ← Dict
  - 遍历 b ∈ [0.1, 0.25, 0.5]
  - 遍历 t ∈ [0.1, 0.5, 1.0]
  - 返回 _reg(…)

### MI-0385 _spec()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (3):
  - 赋值 tbl ← Dict
  - 遍历 k ∈ range(…)
  - 返回 _reg(…)

### MI-0386 _misc()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (3):
  - 赋值 tbl ← Dict
  - 遍历 k ∈ range(…)
  - 返回 _reg(…)

### MI-0387 _conv()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (4):
  - FunctionDef
  - 赋值 tbl ← Dict
  - 遍历 k ∈ range(…)
  - 返回 _reg(…)

### MI-0388 _pool()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (3):
  - 赋值 tbl ← Dict
  - 遍历 k ∈ range(…)
  - 返回 _reg(…)

### MI-0389 _shap()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (3):
  - 赋值 tbl ← Dict
  - 遍历 k ∈ range(…)
  - 返回 _reg(…)

### MI-0390 _indx()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (3):
  - 赋值 tbl ← Dict
  - 遍历 k ∈ range(…)
  - 返回 _reg(…)

### MI-0391 _cast()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (3):
  - 赋值 tbl ← Dict
  - 遍历 b ∈ [2, 4, 6]
  - 返回 _reg(…)

### MI-0392 _rand()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (5):
  - 赋值 tbl ← Dict
  - FunctionDef
  - 遍历 k ∈ ["normal", "uniform", "bernoulli"]
  - 遍历 s ∈ range(…)
  - 返回 _reg(…)

### MI-0393 _seqm()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (4):
  - 赋值 s ← Lambda
  - 赋值 tbl ← Dict
  - 遍历 k ∈ range(…)
  - 返回 _reg(…)

### MI-0394 _cache()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (3):
  - 赋值 tbl ← Dict
  - 遍历 k ∈ range(…)
  - 返回 _reg(…)

### MI-0395 _specf()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (3):
  - 赋值 tbl ← Dict
  - 遍历 n ∈ range(…)
  - 返回 _reg(…)

### MI-0396 _topk()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (6):
  - 赋值 tbl ← Dict
  - FunctionDef
  - 遍历 k ∈ range(…)
  - 遍历 k ∈ range(…)
  - 遍历 k ∈ range(…)
  - 返回 _reg(…)

### MI-0397 _build()
- 模块: `l10\_microops.py`
- 用途: 组装 32 家族 × 32 成员 = 1024 条微指令。  ·  INV 8,9
- 微指令步骤 (6):
  - 声明并赋值 ops ← Dict
  - 声明并赋值 fams ← Dict
  - 遍历 b ∈ _BUILDERS
  - Assert
  - Assert
  - 返回 [ops, fams]

### MI-0398 count_ops()
- 模块: `l10\_microops.py`
- 用途: (无 docstring)  ·  INV 8,9
- 微指令步骤 (1):
  - 返回 len(…)

### MI-0399 run_all(seed, shape)
- 模块: `l10\_microops.py`
- 用途: 对全部微指令执行冒烟，返回报告。  ·  INV 8,9
- 微指令步骤 (6):
  - 赋值 g ← np.random.default_rng(…)
  - 赋值 x ← g.standard_normal(…)
  - 赋值 y ← g.standard_normal(…)
  - 赋值 (ok, bad) ← [0, []]
  - 遍历 (name, op) ∈ MICROOPS.items(…)
  - 返回 Dict

### MI-0400 _assert_softmax_unit_sum(family, member, call)
- 模块: `l10\_properties.py`
- 用途: smax 家族中「确实是分布」的成员: 非负 + 末维和为 1。  ·  INV 9
- 微指令步骤 (5):
  - 赋值 out ← np.asarray(…)
  - 判断 BoolOp
  - 判断 np.any(…)
  - 赋值 s ← out.sum(…)
  - 判断 UnaryOp

### MI-0401 _assert_softmax_shift_invariant(family, member, call)
- 模块: `l10\_properties.py`
- 用途: smax 家族: softmax(x + c) == softmax(x) —— 平移不变 (INV-09)。  ·  INV 9
- 微指令步骤 (5):
  - 赋值 x ← Call.standard_normal(…)
  - 赋值 a ← np.asarray(…)
  - 赋值 b ← np.asarray(…)
  - 判断 BoolOp
  - 判断 UnaryOp

### MI-0402 _assert_logsumexp_identity(family, member, call)
- 模块: `l10\_properties.py`
- 用途: logsumexp 恒等式 (INV-08): logsumexp(x) == log(Σexp(x)) 且大数不溢出。  ·  INV 8
- 微指令步骤 (6):
  - 赋值 x ← Call.standard_normal(…) mult 50.0
  - 赋值 out ← np.asarray(…)
  - 赋值 m ← np.max(…)
  - 赋值 ref ← float(…)
  - 赋值 got ← float(…)
  - 判断 Compare

### MI-0403 _assert_l2norm_strict_unit(family, member, call)
- 模块: `l10\_properties.py`
- 用途: norm 家族: ``l2norm`` 输出严格单位范数 (零向量返回 0)。
- 微指令步骤 (5):
  - 赋值 x ← Call.standard_normal(…)
  - 赋值 out ← np.asarray(…)
  - 判断 Compare
  - 赋值 n ← np.linalg.norm(…)
  - 判断 UnaryOp

### MI-0404 _assert_norm_family_contract(family, member, call)
- 模块: `l10\_properties.py`
- 用途: norm 家族各成员的真实不变量 (逐成员判据, 不套用统一模板)。  ·  复用 RMSNorm
- 微指令步骤 (4):
  - 赋值 x ← Call.standard_normal(…)
  - 赋值 out ← np.asarray(…)
  - 判断 UnaryOp
  - 判断 Compare

### MI-0405 _assert_kl_nonneg(family, member, call)
- 模块: `l10\_properties.py`
- 用途: prob 家族: KL 散度非负 (Shannon 1948): KL(p‖q) ≥ 0。
- 微指令步骤 (6):
  - 赋值 g ← np.random.default_rng(…)
  - 赋值 p ← g.random(…) add 0.05
  - 赋值 p ← p div p.sum(…)
  - 赋值 out ← np.asarray(…)
  - 判断 Compare
  - 判断 np.any(…)

### MI-0406 _assert_entropy_bound(family, member, call)
- 模块: `l10\_properties.py`
- 用途: prob 家族: 熵非负, 且 ≤ log|V| (均匀分布取等)。
- 微指令步骤 (7):
  - 赋值 g ← np.random.default_rng(…)
  - 赋值 p ← g.random(…) add 0.05
  - 赋值 p ← p div p.sum(…)
  - 赋值 out ← np.asarray(…)
  - 判断 Compare
  - 判断 np.any(…)
  - 判断 np.any(…)

### MI-0407 _assert_loss_nonneg(family, member, call)
- 模块: `l10\_properties.py`
- 用途: loss 家族: 损失非负 (MSE/MAE/Huber/CE/NLL 的共性)。
- 微指令步骤 (4):
  - 赋值 x ← Call.standard_normal(…)
  - 赋值 out ← np.asarray(…)
  - 判断 BoolOp
  - 判断 np.any(…)

### MI-0408 _assert_attn_flash_equiv(family, member, call)
- 模块: `l10\_properties.py`
- 用途: attn 家族: Flash 风格在线 softmax ≡ 朴素注意力 (INV-02)。  ·  INV 2
- 微指令步骤 (7):
  - 赋值 x ← Call.standard_normal(…)
  - 赋值 got ← np.asarray(…)
  - 判断 Compare
  - 赋值 d ← x.shape[…]
  - 判断 Compare
  - 赋值 ref ← _ops.attention_naive(…)
  - 判断 UnaryOp

### MI-0409 _assert_ns_singular_band(family, member, call)
- 模块: `l10\_properties.py`
- 用途: Newton-Schulz 奇异值落入 (0.5, 1.5) —— Muon 的公开验收标准。  ·  复用 nanogpt
- 微指令步骤 (5):
  - 赋值 G ← Call.standard_normal(…)
  - 赋值 out ← np.asarray(…)
  - 判断 Compare
  - 赋值 sv ← np.linalg.svd(…)
  - 判断 BoolOp

### MI-0410 _assert_optm_contract(family, member, call)
- 模块: `l10\_properties.py`
- 用途: optm 家族各成员的真实不变量 (逐成员判据)。  ·  复用 AdamW
- 微指令步骤 (5):
  - 赋值 x ← Call.standard_normal(…)
  - 判断 Compare
  - 赋值 out ← np.asarray(…)
  - 判断 UnaryOp
  - 判断 Compare

### MI-0411 _assert_spec_contract(family, member, call)
- 模块: `l10\_properties.py`
- 用途: spec 家族各成员的真实不变量 (逐成员判据)。
- 微指令步骤 (9):
  - 判断 Compare
  - 判断 member.startswith(…)
  - 判断 Compare
  - 赋值 A ← Call.standard_normal(…)
  - 赋值 out ← np.asarray(…)
  - 判断 UnaryOp
  - 判断 BoolOp
  - 判断 BoolOp
  - 判断 BoolOp

### MI-0412 _assert_power_iteration_convergence(call)
- 模块: `l10\_properties.py`
- 用途: 幂迭代收敛性: 估计值从下方逼近真值, 且迭代数越多误差越小。
- 微指令步骤 (6):
  - 赋值 A ← Call.standard_normal(…)
  - 赋值 ref ← float(…)
  - 赋值 got ← float(…)
  - 判断 Compare
  - 判断 Compare
  - 返回

### MI-0413 _assert_topk_contract(family, member, call)
- 模块: `l10\_properties.py`
- 用途: topk 家族: 掩码型每行恰好 min(k, 维数) 个 1; 均值型落在 [min, max] 内。
- 微指令步骤 (4):
  - 赋值 x ← Call.standard_normal(…)
  - 赋值 out ← np.asarray(…)
  - 判断 Compare
  - 判断 Compare

### MI-0414 _assert_nonneg(family, member, call)
- 模块: `l10\_properties.py`
- 用途: 通用: 输出非负 (距离/范数/计数类)。
- 微指令步骤 (3):
  - 赋值 x ← Call.standard_normal(…)
  - 赋值 out ← np.asarray(…)
  - 判断 np.any(…)

### MI-0415 _assert_finite(family, member, call)
- 模块: `l10\_properties.py`
- 用途: 通用: 输出全有限 (量化族, INV-07 误差界前提)。  ·  INV 7
- 微指令步骤 (3):
  - 赋值 x ← Call.standard_normal(…)
  - 赋值 out ← np.asarray(…)
  - 判断 UnaryOp

### MI-0416 _assert_rope_norm_preserved(family, member, call)
- 模块: `l10\_properties.py`
- 用途: rope 家族: **真正的旋转成员**保持向量范数 (正交变换性质, INV-03)。  ·  INV 3  ·  复用 NTK, RoPE
- 微指令步骤 (6):
  - 赋值 x ← Call.standard_normal(…)
  - 赋值 out ← np.asarray(…)
  - 判断 Compare
  - 赋值 n0 ← np.linalg.norm(…)
  - 赋值 n1 ← np.linalg.norm(…)
  - 判断 UnaryOp

### MI-0417 _assert_ract_stat_valid(family, member, call)
- 模块: `l10\_properties.py`
- 用途: ract 家族: 统计量输出有限; 非负量 (范数/矩/平方均值) 不为负。
- 微指令步骤 (4):
  - 赋值 x ← Call.standard_normal(…)
  - 赋值 out ← np.asarray(…)
  - 判断 UnaryOp
  - 判断 Compare

### MI-0418 _assert_metr_valid(family, member, call)
- 模块: `l10\_properties.py`
- 用途: metr 家族: 指标输出有限; 比例型成员必须在 [0,1]。
- 微指令步骤 (4):
  - 赋值 x ← Call.standard_normal(…)
  - 赋值 out ← np.asarray(…)
  - 判断 UnaryOp
  - 判断 BoolOp

### MI-0419 _member_name(family, key)
- 模块: `l10\_properties.py`
- 用途: 从全局键 ``family.member`` 取出成员名。
- 微指令步骤 (1):
  - 返回 IfExp

### MI-0420 _make_call(op, x, y)
- 模块: `l10\_properties.py`
- 用途: 构造该微指令的调用器。
- 微指令步骤 (2):
  - FunctionDef
  - 返回 call

### MI-0421 property_report(seed, shape)
- 模块: `l10\_properties.py`
- 用途: 对注册了性质断言的家族逐条执行断言。  ·  复用 SSE
- 微指令步骤 (9):
  - 赋值 g ← np.random.default_rng(…)
  - 赋值 x ← g.standard_normal(…)
  - 赋值 y ← g.standard_normal(…)
  - 赋值 checked, passed ← 0
  - 声明并赋值 failed ← []
  - 声明并赋值 exemptions ← Dict
  - 遍历 (family, (assert_fn, exclude, reason)) ∈ PROPERTY_ASSERTIONS.items(…)
  - 赋值 skipped ← sum(…)
  - 返回 Dict

### MI-0422 rng()
- 模块: `l10\_support.py`
- 用途: 返回固定种子的独立随机源 (不触碰全局状态)。
- 微指令步骤 (1):
  - 返回 np.random.default_rng(…)

### MI-0423 vec(n)
- 模块: `l10\_support.py`
- 用途: 标准正态向量。
- 微指令步骤 (1):
  - 返回 Call.standard_normal(…)

### MI-0424 mat(n)
- 模块: `l10\_support.py`
- 用途: 标准正态方阵。
- 微指令步骤 (1):
  - 返回 Call.standard_normal(…)

### MI-0425 spd(n)
- 模块: `l10\_support.py`
- 用途: 对称正定矩阵 (Cholesky / 二次型 / 谱类算子的前提)。
- 微指令步骤 (2):
  - 赋值 a ← Call.standard_normal(…)
  - 返回 a matmult a.T add n mult np.eye(…)

### MI-0426 pos_mat(n)
- 模块: `l10\_support.py`
- 用途: 元素严格为正的矩阵 (log / 概率类逐元素算子的前提)。
- 微指令步骤 (1):
  - 返回 np.abs(…) add 0.5

### MI-0427 dist(n)
- 模块: `l10\_support.py`
- 用途: 非负且和为 1 的概率分布。
- 微指令步骤 (2):
  - 赋值 p ← Call.random(…) add 0.05
  - 返回 p div p.sum(…)

### MI-0428 ints(n, hi)
- 模块: `l10\_support.py`
- 用途: 整数向量 (索引 / 计数类算子的前提)。
- 微指令步骤 (1):
  - 返回 Call.integers(…)

### MI-0429 check_close(out, ref, tol)
- 模块: `l10\_support.py`
- 用途: 断言标量输出等于参考值 (容差内)。
- 微指令步骤 (2):
  - 赋值 got ← float(…)
  - 判断 Compare

### MI-0430 check_nonneg(out, inputs)
- 模块: `l10\_support.py`
- 用途: 断言输出非负 (熵 / 散度 / 范数的基本性质)。
- 微指令步骤 (1):
  - 判断 np.any(…)

### MI-0431 check_unit_norm(out, inputs)
- 模块: `l10\_support.py`
- 用途: 断言输出末维 L2 范数为 1。
- 微指令步骤 (2):
  - 赋值 n ← np.linalg.norm(…)
  - 判断 UnaryOp

### MI-0432 check_unit_sum(out, inputs)
- 模块: `l10\_support.py`
- 用途: 断言输出末维和为 1 (概率分布族)。
- 微指令步骤 (2):
  - 赋值 s ← Call.sum(…)
  - 判断 UnaryOp

### MI-0433 check_softmax(out, inputs)
- 模块: `l10\_support.py`
- 用途: 断言 softmax 族: 非负且行和为 1。
- 微指令步骤 (3):
  - 赋值 o ← np.asarray(…)
  - 判断 np.any(…)
  - 调用 check_unit_sum(…)

### MI-0434 check_symmetric(out, inputs)
- 模块: `l10\_support.py`
- 用途: 断言输出矩阵对称。
- 微指令步骤 (2):
  - 赋值 o ← np.asarray(…)
  - 判断 BoolOp

### MI-0435 check_psd(out, inputs)
- 模块: `l10\_support.py`
- 用途: 断言输出矩阵半正定 (最小特征值 ≥ -tol)。
- 微指令步骤 (2):
  - 赋值 o ← np.asarray(…)
  - 判断 BoolOp

### MI-0436 check_posdef(out, inputs)
- 模块: `l10\_support.py`
- 用途: 断言输出矩阵正定 (Cholesky 可分解)。
- 微指令步骤 (2):
  - 赋值 o ← np.asarray(…)
  - 判断 BoolOp

### MI-0437 check_shape_kept(out, inputs)
- 模块: `l10\_support.py`
- 用途: 断言输出与首个输入同形。
- 微指令步骤 (1):
  - 判断 Compare

### MI-0438 check_identity(out, inputs)
- 模块: `l10\_support.py`
- 用途: 断言输出与输入逐位一致 (幂等 / 自反族)。
- 微指令步骤 (1):
  - 判断 UnaryOp

### MI-0439 check_sorted(out, inputs)
- 模块: `l10\_support.py`
- 用途: 断言输出单调不减 (排序族)。
- 微指令步骤 (2):
  - 赋值 o ← np.asarray(…)
  - 判断 BoolOp

### MI-0440 check_integer(out, inputs)
- 模块: `l10\_support.py`
- 用途: 断言输出为整数值 (索引 / 计数族)。
- 微指令步骤 (2):
  - 赋值 o ← np.asarray(…)
  - 判断 UnaryOp

### MI-0441 check_in_range(out, inputs)
- 模块: `l10\_support.py`
- 用途: 断言输出落在 [0, 1]。
- 微指令步骤 (2):
  - 赋值 o ← np.asarray(…)
  - 判断 BoolOp

### MI-0442 check_corr(out, inputs)
- 模块: `l10\_support.py`
- 用途: 断言输出为合法相关系数 (|r| ≤ 1)。
- 微指令步骤 (2):
  - 赋值 o ← np.asarray(…)
  - 判断 np.any(…)

## X 横切审计 (Cross-cutting Audits)

### MI-0443 file_has_citation(path)
- 模块: `scripts\audit_no_reinvention.py`
- 用途: (无 docstring)  ·  INV 42
- 微指令步骤 (2):
  - 异常保护 (try/except)
  - 返回 any(…)

### MI-0444 resolve_files(root, pattern)
- 模块: `scripts\audit_no_reinvention.py`
- 用途: (无 docstring)  ·  INV 42
- 微指令步骤 (3):
  - 赋值 out ← []
  - 判断 pattern.endswith(…)
  - 返回 out

### MI-0445 collect_citations(root)
- 模块: `scripts\audit_no_reinvention.py`
- 用途: 返回全代码库命中到的不同引用标记名集合。  ·  INV 42
- 微指令步骤 (3):
  - 声明并赋值 hit ← set(…)
  - 遍历 (base, _, files) ∈ os.walk(…)
  - 返回 hit

### MI-0446 main()
- 模块: `scripts\audit_no_reinvention.py`
- 用途: (无 docstring)  ·  INV 42
- 微指令步骤 (19):
  - 赋值 ap ← argparse.ArgumentParser(…)
  - 调用 ap.add_argument(…)
  - 赋值 args ← ap.parse_args(…)
  - 赋值 root ← args.root
  - 判断 UnaryOp
  - 判断 UnaryOp
  - 调用 print(…)
  - 调用 print(…)
  - 调用 print(…)
  - 赋值 layer_ok ← True
  - 遍历 (lid, name, patterns) ∈ LAYERS
  - 赋值 hits ← collect_citations(…)
  - 赋值 citation_ok ← Compare
  - 调用 print(…)
  - 调用 print(…)
  - 调用 print(…)
  - 判断 BoolOp
  - 调用 print(…)
  - 返回 1

### MI-0447 bench(steps, d_model, n_layers, n_heads, n_kv_heads, batch_size, seq_len, vocab_size)
- 模块: `scripts\bench_trainer.py`
- 用途: (无 docstring)
- 微指令步骤 (11):
  - 调用 rng.seed(…)
  - 赋值 cfg ← ModelConfig(…)
  - 赋值 corpus ← Call.integers(…)
  - 赋值 tr ← TinyTrainer(…)
  - 赋值 t0 ← time.perf_counter(…)
  - 调用 tr.train(…)
  - 赋值 dt ← time.perf_counter(…) sub t0
  - 赋值 ms ← dt div steps mult 1000
  - 调用 print(…)
  - 调用 print(…)
  - 返回 ms

### MI-0448 find_markdown(root)
- 模块: `scripts\check_docs.py`
- 用途: (无 docstring)
- 微指令步骤 (3):
  - 赋值 out ← []
  - 遍历 (base, _, files) ∈ os.walk(…)
  - 返回 sorted(…)

### MI-0449 check_links(path, problems)
- 模块: `scripts\check_docs.py`
- 用途: (无 docstring)
- 微指令步骤 (3):
  - 异常保护 (try/except)
  - 赋值 base_dir ← os.path.dirname(…)
  - 遍历 m ∈ _LINK_RE.findall(…)

### MI-0450 check_readme_sections(path, problems)
- 模块: `scripts\check_docs.py`
- 用途: (无 docstring)
- 微指令步骤 (6):
  - 异常保护 (try/except)
  - 赋值 headings ← Constant.join(…)
  - 赋值 headings_lower ← headings.lower(…)
  - 赋值 missing ← []
  - 遍历 (pat, label) ∈ README_SECTIONS
  - 判断 missing

### MI-0451 main()
- 模块: `scripts\check_docs.py`
- 用途: (无 docstring)
- 微指令步骤 (26):
  - 赋值 ap ← argparse.ArgumentParser(…)
  - 调用 ap.add_argument(…)
  - 赋值 args ← ap.parse_args(…)
  - 赋值 root ← args.root
  - 判断 UnaryOp
  - 赋值 root ← os.path.abspath(…)
  - 调用 print(…)
  - 调用 print(…)
  - 调用 print(…)
  - 声明并赋值 problems ← []
  - 调用 print(…)
  - 遍历 f ∈ REQUIRED_FILES add DESIGN_FILES
  - 调用 print(…)
  - 赋值 md_files ← find_markdown(…)
  - 声明并赋值 link_problems ← []
  - 遍历 p ∈ md_files
  - 判断 link_problems
  - 调用 print(…)
  - 赋值 readme ← os.path.join(…)
  - 声明并赋值 section_problems ← []
  - 调用 check_readme_sections(…)
  - 判断 section_problems
  - 调用 print(…)
  - 判断 problems
  - 调用 print(…)
  - 返回 0

### MI-0452 main()
- 模块: `scripts\check_l10.py`
- 用途: (无 docstring)
- 微指令步骤 (25):
  - 赋值 ap ← argparse.ArgumentParser(…)
  - 调用 ap.add_argument(…)
  - 调用 ap.add_argument(…)
  - 赋值 args ← ap.parse_args(…)
  - 调用 print(…)
  - 调用 print(…)
  - 调用 print(…)
  - 调用 print(…)
  - 赋值 core ← sum(…)
  - 赋值 sweep ← sum(…)
  - 调用 print(…)
  - 调用 print(…)
  - 赋值 report ← run_all(…)
  - 赋值 props ← report[…]
  - 判断 args.verbose
  - 调用 print(…)
  - 调用 print(…)
  - 判断 props[…]
  - 赋值 failures ← list(…)
  - 判断 failures
  - 调用 print(…)
  - 赋值 ok ← BoolOp
  - 调用 print(…)
  - 调用 print(…)
  - 返回 IfExp

### MI-0453 rank_of(path)
- 模块: `scripts\check_layering.py`
- 用途: 返回文件所属包的层秩; 根目录 (ifai/__init__.py 等) 视为应用层 9。  ·  INV 41
- 微指令步骤 (5):
  - 赋值 rel ← os.path.relpath(…)
  - 赋值 parts ← rel.split(…)
  - 判断 Compare
  - 赋值 pkg ← parts[…]
  - 返回 PACKAGE_RANK.get(…)

### MI-0454 imports_of(path)
- 模块: `scripts\check_layering.py`
- 用途: (无 docstring)  ·  INV 41
- 微指令步骤 (4):
  - 上下文管理 (with)
  - 赋值 targets ← []
  - 遍历 node ∈ ast.walk(…)
  - 返回 targets

### MI-0455 main()
- 模块: `scripts\check_layering.py`
- 用途: (无 docstring)  ·  INV 41
- 微指令步骤 (16):
  - 赋值 ap ← argparse.ArgumentParser(…)
  - 调用 ap.add_argument(…)
  - 赋值 args ← ap.parse_args(…)
  - 赋值 root ← args.root
  - 判断 UnaryOp
  - 判断 UnaryOp
  - 赋值 violations ← []
  - 赋值 scanned ← 0
  - 遍历 (base, _, files) ∈ os.walk(…)
  - 调用 print(…)
  - 调用 print(…)
  - 调用 print(…)
  - 调用 print(…)
  - 判断 violations
  - 调用 print(…)
  - 返回 0

### MI-0456 build_model(out_dir)
- 模块: `scripts\gen_baseline.py`
- 用途: (无 docstring)
- 微指令步骤 (9):
  - 调用 os.makedirs(…)
  - 调用 rng.seed(…)
  - 赋值 cfg ← ModelConfig(…)
  - 赋值 tok ← Call.train(…)
  - 调用 tok.to_json(…)
  - 赋值 trainer ← TinyTrainer(…)
  - 调用 trainer.train(…)
  - 调用 trainer.save(…)
  - 返回 [load_model(…), tok]

### MI-0457 bench(model, tok)
- 模块: `scripts\gen_baseline.py`
- 用途: 测量 prefill / TPOT / 吞吐。  ·  复用 MLPerf
- 微指令步骤 (11):
  - 赋值 ids ← BoolOp[…]
  - 当 Compare 循环
  - 赋值 ids ← ids[…]
  - 赋值 (prefill_ms, decode_ms) ← [[], []]
  - 赋值 per_repeat_tpot ← []
  - 遍历 _ ∈ range(…)
  - 遍历 _ ∈ range(…)
  - 赋值 prefill ← statistics.median(…)
  - 赋值 tpot ← statistics.median(…)
  - FunctionDef
  - 返回 Dict

### MI-0458 write_baseline(metrics, repo_root)
- 模块: `scripts\gen_baseline.py`
- 用途: (无 docstring)
- 微指令步骤 (5):
  - 赋值 md ← os.path.join(…)
  - 赋值 js ← os.path.join(…)
  - 上下文管理 (with)
  - 赋值 lines ← ["# 性能基线 (BASELINE…", "", "> 由 `scripts/gen…"]
  - 上下文管理 (with)

### MI-0459 main()
- 模块: `scripts\gen_baseline.py`
- 用途: (无 docstring)
- 微指令步骤 (11):
  - 赋值 ap ← argparse.ArgumentParser(…)
  - 调用 ap.add_argument(…)
  - 调用 ap.add_argument(…)
  - 赋值 args ← ap.parse_args(…)
  - 赋值 root ← args.out
  - 赋值 metrics ← bench(…)
  - 判断 args.check
  - 调用 write_baseline(…)
  - 调用 print(…)
  - 遍历 (k, v) ∈ metrics.items(…)
  - 返回 0

### MI-0460 layer_of(path)
- 模块: `scripts\gen_micro_instructions.py`
- 用途: (无 docstring)  ·  INV 41,42
- 微指令步骤 (5):
  - 赋值 rel ← os.path.relpath(…)
  - 赋值 parts ← rel.split(…)
  - 判断 Compare
  - 赋值 pkg ← parts[…]
  - 返回 PACKAGE_RANK.get(…)

### MI-0461 extract_invs(text)
- 模块: `scripts\gen_micro_instructions.py`
- 用途: 提取文本中的 INV 编号, 支持单条与 `INV-01..09` 区间展开。  ·  INV 1,2,3,4,5,6,7,8,9
- 微指令步骤 (4):
  - 赋值 found ← set(…)
  - 遍历 m ∈ INV_RANGE_RE.finditer(…)
  - 遍历 m ∈ INV_RE.finditer(…)
  - 返回 found

### MI-0462 args_sig(node)
- 模块: `scripts\gen_micro_instructions.py`
- 用途: (无 docstring)  ·  INV 41,42
- 微指令步骤 (5):
  - 赋值 names ← ListComp
  - 判断 node.vararg
  - 累算 names add ListComp
  - 判断 node.kwarg
  - 返回 Constant.join(…)

### MI-0463 describe_stmt(stmt)
- 模块: `scripts\gen_micro_instructions.py`
- 用途: (无 docstring)  ·  INV 41,42
- 微指令步骤 (14):
  - 赋值 t ← Call.__name__
  - 判断 isinstance(…)
  - 判断 isinstance(…)
  - 判断 isinstance(…)
  - 判断 isinstance(…)
  - 判断 isinstance(…)
  - 判断 isinstance(…)
  - 判断 isinstance(…)
  - 判断 isinstance(…)
  - 判断 isinstance(…)
  - 判断 isinstance(…)
  - 判断 BoolOp
  - 判断 isinstance(…)
  - 返回 t

### MI-0464 _name(n)
- 模块: `scripts\gen_micro_instructions.py`
- 用途: (无 docstring)  ·  INV 41,42
- 微指令步骤 (5):
  - 判断 isinstance(…)
  - 判断 isinstance(…)
  - 判断 isinstance(…)
  - 判断 isinstance(…)
  - 返回 Call.__name__

### MI-0465 _expr(n)
- 模块: `scripts\gen_micro_instructions.py`
- 用途: (无 docstring)  ·  INV 41,42
- 微指令步骤 (9):
  - 判断 Compare
  - 判断 isinstance(…)
  - 判断 isinstance(…)
  - 判断 isinstance(…)
  - 判断 isinstance(…)
  - 判断 isinstance(…)
  - 判断 isinstance(…)
  - 判断 BoolOp
  - 返回 Call.__name__

### MI-0466 collect(root, force_layer)
- 模块: `scripts\gen_micro_instructions.py`
- 用途: 返回 {layer: [entries]} 与全局统计。每个 entry 为 dict。  ·  INV 41,42
- 微指令步骤 (6):
  - 赋值 by_layer ← DictComp
  - 赋值 total_func ← 0
  - 赋值 total_steps ← 0
  - 赋值 inv_hits ← set(…)
  - 遍历 (base, _, files) ∈ os.walk(…)
  - 返回 [by_layer, total_func, total_steps]

### MI-0467 _entry(rel, func, cls, src, module_doc_inv, methods)
- 模块: `scripts\gen_micro_instructions.py`
- 用途: (无 docstring)  ·  INV 41,42
- 微指令步骤 (9):
  - 判断 Compare
  - 赋值 first_line ← IfExp
  - 赋值 func_inv ← extract_invs(…)
  - 赋值 inv ← IfExp
  - 赋值 src_hits ← sorted(…)
  - 赋值 steps ← []
  - 遍历 stmt ∈ body
  - 判断 UnaryOp
  - 返回 Dict

### MI-0468 render(by_layer, total_func, total_steps, inv_hits, out_path)
- 模块: `scripts\gen_micro_instructions.py`
- 用途: (无 docstring)  ·  INV 41,42
- 微指令步骤 (20):
  - 赋值 lines ← []
  - 调用 lines.append(…)
  - 调用 lines.append(…)
  - 调用 lines.append(…)
  - 调用 lines.append(…)
  - 调用 lines.append(…)
  - 赋值 total_mi ← total_func add total_steps
  - 调用 lines.append(…)
  - 调用 lines.append(…)
  - 调用 lines.append(…)
  - 调用 lines.append(…)
  - 调用 lines.append(…)
  - 调用 lines.append(…)
  - 遍历 i ∈ range(…)
  - 调用 lines.append(…)
  - 赋值 mid ← 1
  - 遍历 i ∈ range(…)
  - 赋值 text ← Constant.join(…)
  - 上下文管理 (with)
  - 返回 [text, total_mi]

### MI-0469 main()
- 模块: `scripts\gen_micro_instructions.py`
- 用途: (无 docstring)  ·  INV 41,42
- 微指令步骤 (15):
  - 赋值 ap ← argparse.ArgumentParser(…)
  - 调用 ap.add_argument(…)
  - 调用 ap.add_argument(…)
  - 调用 ap.add_argument(…)
  - 赋值 args ← ap.parse_args(…)
  - 赋值 (by_layer, total_func, total_steps, inv_hits) ← collect(…)
  - 判断 os.path.isdir(…)
  - 赋值 (text, total_mi) ← render(…)
  - 调用 print(…)
  - 调用 print(…)
  - 调用 print(…)
  - 调用 print(…)
  - 调用 print(…)
  - 调用 print(…)
  - 返回 0
