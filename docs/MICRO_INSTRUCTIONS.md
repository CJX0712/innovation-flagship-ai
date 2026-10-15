# 微指令级操作分解 (Micro-Instruction Decomposition)

> 由 `scripts/gen_micro_instructions.py` 对 `ifai/` 源码 AST 静态扫描自动生成。
> 零重写 · 集大成: 每条微指令标注其复用的世界顶级来源与所服务的可验证不变量 (INV)。

**总计**: 功能级微指令 203 条 + 步骤级微指令 902 条 = **1105** 条; 覆盖 INV [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 30, 33, 34, 36, 37, 39, 40]。

## 分层汇总

| 层 | 说明 | 功能级 | 步骤级 | 合计 |
| --- | --- | ---: | ---: | ---: |
| L0 | L0 算子/内核 (Operator/Kernel) | 16 | 107 | 123 |
| L1 | L1 张量/数值基础 (Tensor/Numeric) | 4 | 9 | 13 |
| L2 | L2 模型架构 (Model Architecture) | 30 | 150 | 180 |
| L3 | L3 推理引擎 (Inference Engine) | 17 | 72 | 89 |
| L4 | L4 训练/微调 (Training/Finetune) | 52 | 254 | 306 |
| L5 | L5 检索/记忆 (Retrieval/Memory) | 28 | 126 | 154 |
| L6 | L6 智能体 (Agent) | 32 | 97 | 129 |
| L7 | L7 服务/编排 (Service/Orchestration) | 9 | 32 | 41 |
| L8 | L8 评估/对齐 (Evaluation/Alignment) | 10 | 38 | 48 |
| L9 | L9 应用/接口 (Application/Interface) | 5 | 17 | 22 |

## L0 算子/内核 (Operator/Kernel)

### MI-0001 logsumexp(x, axis, keepdims)
- 模块: `core\numeric.py`
- 用途: 数值稳定的 log(sum(exp(x)))。  ·  INV 08
- 微指令步骤 (6):
  - 赋值 x ← np.asarray(…)
  - 赋值 xmax ← np.max(…)
  - 赋值 xmax_safe ← np.where(…)
  - 赋值 out ← xmax_safe add np.log(…)
  - 判断 UnaryOp
  - 返回 out.astype(…)

### MI-0002 stable_softmax(logits, axis)
- 模块: `core\numeric.py`
- 用途: 平移不变的 softmax (INV-09)。  ·  INV 09
- 微指令步骤 (4):
  - 赋值 logits ← np.asarray(…)
  - 赋值 z ← logits sub np.max(…)
  - 赋值 e ← np.exp(…)
  - 返回 e div np.sum(…)

### MI-0003 safe_cross_entropy(logits, targets, ignore_index, label_smoothing)
- 模块: `core\numeric.py`
- 用途: 返回 (loss 标量, 对 logits 的梯度)。  ·  INV 08
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
- 用途: RMSNorm: x / sqrt(mean(x^2)+eps) * weight。INV-06 范数归一。  ·  INV 06  ·  复用 RMSNorm
- 微指令步骤 (3):
  - 赋值 x ← np.asarray(…)
  - 赋值 ms ← np.mean(…)
  - 返回 x div np.sqrt(…) mult np.asarray(…)

### MI-0006 _rope_freqs(dim, base, scaling)
- 模块: `core\ops.py`
- 用途: 生成旋转频率 (NTK-aware: 当 scaling≠1 拉伸 base)。  ·  INV 01,02,03,04,05,06,07  ·  复用 NTK
- 微指令步骤 (4):
  - 赋值 half ← dim floordiv 2
  - 赋值 inv ← 1.0 div base pow np.arange(…) div half
  - 判断 Compare
  - 返回 inv

### MI-0007 apply_rope(x, positions, dim, base, scaling)
- 模块: `core\ops.py`
- 用途: 对最后一维 (隐维) 施加旋转位置编码。  ·  INV 03,16
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
- 用途: SwiGLU(x) = (x·W1 + b1) ⊙ silu(x·W2 + b2) · W? — 这里实现 SwiGLU FFN 内核。  ·  INV 05  ·  复用 SwiGLU
- 微指令步骤 (6):
  - 赋值 x ← np.asarray(…)
  - 赋值 (w1, w2) ← [np.asarray(…), np.asarray(…)]
  - 赋值 g ← IfExp
  - 赋值 u ← IfExp
  - 赋值 silu ← g div 1.0 add np.exp(…)
  - 返回 silu mult u

### MI-0009 tiled_matmul(a, b, block)
- 模块: `core\ops.py`
- 用途: 分块矩阵乘 (INV-01 == a@b)。分块提升 cache 命中，复刻 BLAS 思想。  ·  INV 01
- 微指令步骤 (7):
  - 赋值 a ← np.asarray(…)
  - 赋值 b ← np.asarray(…)
  - 赋值 (m, k) ← a.shape
  - 赋值 n ← b.shape[…]
  - 赋值 out ← np.zeros(…)
  - 遍历 i ∈ range(…)
  - 返回 out

### MI-0010 attention_flash(q, k, v, scale, causal)
- 模块: `core\ops.py`
- 用途: 在线 softmax 的因果注意力 (INV-02 == 朴素)。  ·  INV 02  ·  复用 Dao
- 微指令步骤 (8):
  - 赋值 q ← np.asarray(…)
  - 赋值 k ← np.asarray(…)
  - 赋值 v ← np.asarray(…)
  - 赋值 (Sq, d) ← q.shape
  - 赋值 Sk ← k.shape[…]
  - 赋值 out ← np.zeros(…)
  - 遍历 i ∈ range(…)
  - 返回 out

### MI-0011 attention_naive(q, k, v, scale, causal)
- 模块: `core\ops.py`
- 用途: (无 docstring)  ·  INV 01,02,03,04,05,06,07
- 微指令步骤 (7):
  - 赋值 q ← np.asarray(…)
  - 赋值 k ← np.asarray(…)
  - 赋值 v ← np.asarray(…)
  - 赋值 scores ← q matmult k.T mult scale
  - 判断 causal
  - 赋值 p ← numeric.stable_softmax(…)
  - 返回 p matmult v

### MI-0012 quantize_int8_per_channel(x, axis)
- 模块: `core\quant.py`
- 用途: 对称 per-channel RTN 量化到 INT8。  ·  INV 03,07  ·  复用 int8
- 微指令步骤 (5):
  - 赋值 x ← np.asarray(…)
  - 赋值 max_abs ← np.max(…)
  - 赋值 scale ← np.where(…) div 127.0
  - 赋值 q ← Call.astype(…)
  - 返回 [q, scale.astype(…)]

### MI-0013 dequantize_int8(q, scale)
- 模块: `core\quant.py`
- 用途: (无 docstring)  ·  INV 03,07
- 微指令步骤 (1):
  - 返回 np.asarray(…) mult np.asarray(…)

### MI-0014 pack_int4(x)
- 模块: `core\quant.py`
- 用途: 把 int8 范围 [-8,7] 的数组打包成 uint8 (每字节两个数, 低4位先)。  ·  INV 03,07  ·  复用 int8
- 微指令步骤 (7):
  - 赋值 x ← Call.ravel(…)
  - 赋值 x ← Call.astype(…)
  - 赋值 pad ← len(…) add 1 floordiv 2 mult 2 sub len(…)
  - 判断 pad
  - 赋值 lo ← x[…].astype(…) bitand 15
  - 赋值 hi ← x[…].astype(…) bitand 15
  - 返回 BinOp.astype(…)

### MI-0015 unpack_int4(packed, n)
- 模块: `core\quant.py`
- 用途: 解包 int4 (符号扩展回 int8 范围 [-8,7])。  ·  INV 03,07  ·  复用 int4, int8
- 微指令步骤 (8):
  - 赋值 p ← np.asarray(…)
  - 赋值 lo ← BinOp.astype(…)
  - 赋值 hi ← BinOp.astype(…)
  - 赋值 x ← np.empty(…)
  - 赋值 x[…] ← lo
  - 赋值 x[…] ← hi
  - 赋值 x ← Call.astype(…)
  - 返回 IfExp

### MI-0016 quantize_fp16(x)
- 模块: `core\quant.py`
- 用途: 模拟 fp16 往返 (numpy 无纯 fp16 计算，这里用 float16 dtype 模拟量化误差)。  ·  INV 03,07
- 微指令步骤 (2):
  - 赋值 x ← np.asarray(…)
  - 返回 Call.astype(…)

## L1 张量/数值基础 (Tensor/Numeric)

### MI-0017 seed(value)
- 模块: `rng.py`
- 用途: 设置全局随机种子并复位所有派生源。  ·  INV 40
- 微指令步骤 (3):
  - Global
  - 赋值 _SEED ← int(…)
  - 赋值 _RNG ← np.random.default_rng(…)

### MI-0018 get_rng()
- 模块: `rng.py`
- 用途: 返回全局 ``Generator``；未初始化时默认 seed(0)。  ·  INV 40
- 微指令步骤 (4):
  - Global
  - 判断 Compare
  - Assert
  - 返回 _RNG

### MI-0019 get_seed()
- 模块: `rng.py`
- 用途: 返回当前种子 (None 表示未设置)。  ·  INV 40
- 微指令步骤 (1):
  - 返回 _SEED

### MI-0020 spawn(child_seed)
- 模块: `rng.py`
- 用途: 派生一个独立的子随机源 (用于并行/隔离场景)，不影响全局源。  ·  INV 40
- 微指令步骤 (1):
  - 返回 np.random.default_rng(…)

## L2 模型架构 (Model Architecture)

### MI-0021 【类】ModelConfig.ModelConfig(类)
- 模块: `models\config.py`
- 用途: Transformer 解码器配置。所有字段可 JSON 往返。  ·  INV 13
- 微指令步骤 (23):
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
  - 声明并赋值 initializer_range ← 0.02
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0022 ModelConfig.__post_init__(self)
- 模块: `models\config.py`
- 用途: (无 docstring)  ·  INV 13
- 微指令步骤 (3):
  - 判断 Compare
  - Assert
  - 判断 self.use_moe

### MI-0023 ModelConfig.d_head(self)
- 模块: `models\config.py`
- 用途: (无 docstring)  ·  INV 13
- 装饰器: property
- 微指令步骤 (1):
  - 返回 self.d_model floordiv self.n_heads

### MI-0024 ModelConfig.d_ff(self)
- 模块: `models\config.py`
- 用途: (无 docstring)  ·  INV 13
- 装饰器: property
- 微指令步骤 (1):
  - 返回 self.d_model mult self.ffn_mult

### MI-0025 ModelConfig.n_repeats(self)
- 模块: `models\config.py`
- 用途: (无 docstring)  ·  INV 13
- 装饰器: property
- 微指令步骤 (1):
  - 返回 self.n_heads floordiv self.n_kv_heads

### MI-0026 ModelConfig.to_json(self, path)
- 模块: `models\config.py`
- 用途: (无 docstring)  ·  INV 13
- 微指令步骤 (3):
  - 赋值 s ← json.dumps(…)
  - 判断 path
  - 返回 s

### MI-0027 ModelConfig.from_json(cls, source)
- 模块: `models\config.py`
- 用途: (无 docstring)  ·  INV 13
- 装饰器: classmethod
- 微指令步骤 (4):
  - 判断 isinstance(…)
  - 赋值 allowed ← SetComp
  - 赋值 obj ← DictComp
  - 返回 cls(…)

### MI-0028 ModelConfig.clone(self)
- 模块: `models\config.py`
- 用途: (无 docstring)  ·  INV 13
- 微指令步骤 (1):
  - 返回 ModelConfig.from_json(…)

### MI-0029 【类】Tokenizer.Tokenizer(类)
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

### MI-0030 Tokenizer.__init__(self, vocab_size, specials)
- 模块: `models\tokenizer.py`
- 用途: (无 docstring)  ·  INV 40
- 微指令步骤 (6):
  - 赋值 self.vocab_size_target ← vocab_size
  - 赋值 self.specials ← BoolOp add ListComp
  - 声明并赋值 self.merges ← []
  - 声明并赋值 self.merge_ranks ← Dict
  - 声明并赋值 self.token2id ← Dict
  - 声明并赋值 self.id2token ← Dict

### MI-0031 Tokenizer.train(self, corpus, min_freq)
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

### MI-0032 Tokenizer._apply_merges(self, syms)
- 模块: `models\tokenizer.py`
- 用途: (无 docstring)  ·  INV 40
- 微指令步骤 (2):
  - 当 True 循环
  - 返回 syms

### MI-0033 Tokenizer.encode(self, text)
- 模块: `models\tokenizer.py`
- 用途: (无 docstring)  ·  INV 40
- 微指令步骤 (3):
  - 声明并赋值 ids ← []
  - 遍历 w ∈ re.findall(…)
  - 返回 ids

### MI-0034 Tokenizer.decode(self, ids)
- 模块: `models\tokenizer.py`
- 用途: (无 docstring)  ·  INV 40
- 微指令步骤 (1):
  - 返回 Constant.join(…)

### MI-0035 Tokenizer.vocab_size(self)
- 模块: `models\tokenizer.py`
- 用途: (无 docstring)  ·  INV 40
- 装饰器: property
- 微指令步骤 (1):
  - 返回 len(…)

### MI-0036 Tokenizer.to_json(self, path)
- 模块: `models\tokenizer.py`
- 用途: (无 docstring)  ·  INV 40
- 微指令步骤 (4):
  - 赋值 obj ← Dict
  - 赋值 s ← json.dumps(…)
  - 判断 path
  - 返回 s

### MI-0037 Tokenizer.from_json(cls, source)
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

### MI-0038 【类】Transformer.Transformer(类)
- 模块: `models\transformer.py`
- 用途: (无 docstring)  ·  INV 10,12,13,14
- 微指令步骤 (10):
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

### MI-0039 Transformer.__init__(self, config)
- 模块: `models\transformer.py`
- 用途: (无 docstring)  ·  INV 10,12,13,14
- 微指令步骤 (2):
  - 赋值 self.cfg ← config
  - 调用 self._init_params(…)

### MI-0040 Transformer._normal(self, *shape)
- 模块: `models\transformer.py`
- 用途: (无 docstring)  ·  INV 10,12,13,14
- 微指令步骤 (2):
  - 赋值 r ← rng.get_rng(…)
  - 返回 BinOp.astype(…)

### MI-0041 Transformer._init_params(self)
- 模块: `models\transformer.py`
- 用途: (无 docstring)  ·  INV 10,12,13,14
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

### MI-0042 Transformer.tie_word_embeddings(self)
- 模块: `models\transformer.py`
- 用途: (无 docstring)  ·  INV 10,12,13,14
- 装饰器: property
- 微指令步骤 (1):
  - 返回 self.cfg.tie_word_embeddings

### MI-0043 Transformer.count_params(self)
- 模块: `models\transformer.py`
- 用途: (无 docstring)  ·  INV 10,12,13,14
- 微指令步骤 (1):
  - 返回 int(…)

### MI-0044 Transformer.forward(self, token_ids, return_hidden, position_ids, key_padding_mask)
- 模块: `models\transformer.py`
- 用途: 前向。  ·  INV 16
- 微指令步骤 (13):
  - 赋值 c ← self.cfg
  - 赋值 ids ← np.asarray(…)
  - 判断 Compare
  - 赋值 (B, T) ← ids.shape
  - 赋值 (d, dh, n_h, n_kv) ← [c.d_model, c.d_head, c.n_heads]
  - 赋值 scale ← 1.0 div dh pow 0.5
  - 判断 Compare
  - 赋值 x ← self.params[…][…]
  - 遍历 l ∈ range(…)
  - 赋值 x ← ops.rms_norm(…)
  - 判断 return_hidden
  - 赋值 logits ← x matmult self.params[…].T
  - 返回 logits

### MI-0045 Transformer._moe(self, h2, pre, c)
- 模块: `models\transformer.py`
- 用途: (无 docstring)  ·  INV 10,12,13,14
- 微指令步骤 (8):
  - 赋值 (B, T, d) ← h2.shape
  - 赋值 router_logits ← h2 matmult self.params[…]
  - 赋值 topk_idx ← np.argsort(…)[…]
  - 赋值 sel ← np.take_along_axis(…)
  - 赋值 w ← numeric.stable_softmax(…)
  - 赋值 out ← np.zeros(…)
  - 遍历 e ∈ range(…)
  - 返回 out

### MI-0046 Transformer.logits_next(self, token_ids)
- 模块: `models\transformer.py`
- 用途: 给定已生成序列, 返回最后一个位置的 logits [V]。  ·  INV 10,12,13,14
- 微指令步骤 (3):
  - 赋值 logits ← self.forward(…)
  - 赋值 last ← logits[…]
  - 返回 last

### MI-0047 Transformer.save(self, dir_path)
- 模块: `models\transformer.py`
- 用途: (无 docstring)  ·  INV 10,12,13,14
- 微指令步骤 (7):
  - 调用 os.makedirs(…)
  - 调用 np.savez(…)
  - 赋值 manifest ← Dict
  - 上下文管理 (with)
  - 赋值 manifest[…][…] ← Call.hexdigest(…)
  - 上下文管理 (with)
  - 返回 dir_path

### MI-0048 Transformer.load(cls, dir_path, verify)
- 模块: `models\transformer.py`
- 用途: (无 docstring)  ·  INV 10,12,13,14
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

### MI-0049 save_model(model, dir_path)
- 模块: `models\transformer.py`
- 用途: (无 docstring)  ·  INV 10,12,13,14
- 微指令步骤 (1):
  - 返回 model.save(…)

### MI-0050 load_model(dir_path, verify)
- 模块: `models\transformer.py`
- 用途: (无 docstring)  ·  INV 10,12,13,14
- 微指令步骤 (1):
  - 返回 Transformer.load(…)

## L3 推理引擎 (Inference Engine)

### MI-0051 【类】PagedKVCache.PagedKVCache(类)
- 模块: `engine\kvcache.py`
- 用途: 按序列分块存储 K/V。K 与 V 各用独立 block 缓冲。  ·  INV 15
- 微指令步骤 (6):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0052 PagedKVCache.__init__(self, block_size, kv_dim, max_blocks)
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

### MI-0053 PagedKVCache.new_sequence(self)
- 模块: `engine\kvcache.py`
- 用途: (无 docstring)  ·  INV 15
- 微指令步骤 (5):
  - 赋值 sid ← self._next_seq
  - 累算 self._next_seq add 1
  - 赋值 self.block_table[…] ← []
  - 赋值 self.seq_len[…] ← 0
  - 返回 sid

### MI-0054 PagedKVCache._ensure_block(self, sid)
- 模块: `engine\kvcache.py`
- 用途: (无 docstring)  ·  INV 15
- 微指令步骤 (1):
  - 判断 Compare

### MI-0055 PagedKVCache.append(self, sid, k, v)
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

### MI-0056 PagedKVCache.get_kv(self, sid)
- 模块: `engine\kvcache.py`
- 用途: 返回该序列截至当前的稠密 K,V [L, kv_dim] (INV-15 等价性来源)。  ·  INV 15
- 微指令步骤 (5):
  - 赋值 L ← self.seq_len[…]
  - 赋值 K ← np.zeros(…)
  - 赋值 V ← np.zeros(…)
  - 遍历 l ∈ range(…)
  - 返回 [K, V]

### MI-0057 PagedKVCache.memory_blocks_used(self)
- 模块: `engine\kvcache.py`
- 用途: (无 docstring)  ·  INV 15
- 装饰器: property
- 微指令步骤 (1):
  - 返回 self.max_blocks sub len(…)

### MI-0058 【类】Sampler.Sampler(类)
- 模块: `engine\sampler.py`
- 用途: (无 docstring)  ·  INV 18,19
- 微指令步骤 (4):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0059 Sampler.__init__(self, seed)
- 模块: `engine\sampler.py`
- 用途: (无 docstring)  ·  INV 18,19
- 微指令步骤 (1):
  - 赋值 self._rng ← IfExp

### MI-0060 Sampler.probs(self, logits, temperature, top_k, top_p, min_p, repetition_penalty, forbidden)
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

### MI-0061 Sampler.greedy(self, logits, forbidden)
- 模块: `engine\sampler.py`
- 用途: (无 docstring)  ·  INV 18,19
- 微指令步骤 (3):
  - 赋值 logits ← Call.copy(…)
  - 判断 forbidden
  - 返回 int(…)

### MI-0062 Sampler.sample(self, logits, temperature, top_k, top_p, min_p, repetition_penalty, forbidden)
- 模块: `engine\sampler.py`
- 用途: (无 docstring)  ·  INV 18,19
- 微指令步骤 (3):
  - 赋值 p ← self.probs(…)
  - 判断 Compare
  - 返回 int(…)

### MI-0063 【类】ContinuousScheduler.ContinuousScheduler(类)
- 模块: `engine\scheduler.py`
- 用途: 对同一批 prompt 做并行贪心生成。  ·  INV 16
- 微指令步骤 (2):
  - FunctionDef
  - FunctionDef

### MI-0064 ContinuousScheduler.__init__(self, model_forward)
- 模块: `engine\scheduler.py`
- 用途: (无 docstring)  ·  INV 16
- 微指令步骤 (1):
  - 赋值 self.fwd ← model_forward

### MI-0065 ContinuousScheduler.generate(self, prompts, max_new_tokens, eos_id, greedy, pad_id)
- 模块: `engine\scheduler.py`
- 用途: (无 docstring)  ·  INV 16
- 微指令步骤 (5):
  - 赋值 seqs ← ListComp
  - 赋值 finished ← [False] mult len(…)
  - 赋值 produced ← [0] mult len(…)
  - 遍历 _ ∈ range(…)
  - 返回 ListComp

### MI-0066 speculative_decode(target_next, draft_next, prompt, gamma, max_new_tokens, eos_id, greedy, temperature)
- 模块: `engine\speculative.py`
- 用途: 返回新生成的 token 列表 (不含 prompt)。  ·  INV 17
- 微指令步骤 (4):
  - 赋值 r ← rng.get_rng(…)
  - 赋值 seq ← list(…)
  - 当 Compare 循环
  - 返回 seq[…]

### MI-0067 _softmax(x)
- 模块: `engine\speculative.py`
- 用途: (无 docstring)  ·  INV 17
- 微指令步骤 (3):
  - 赋值 m ← x.max(…)
  - 赋值 e ← np.exp(…)
  - 返回 e div e.sum(…)

## L4 训练/微调 (Training/Finetune)

### MI-0068 _unbroadcast(grad, shape)
- 模块: `train\autodiff.py`
- 用途: 把梯度累加值收缩回 ``shape`` (处理 numpy 广播的反向, INV-22)。  ·  INV 22
- 微指令步骤 (4):
  - 赋值 grad ← np.asarray(…)
  - 判断 Compare
  - 遍历 i ∈ range(…)
  - 返回 grad

### MI-0069 【类】Tensor.Tensor(类)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (25):
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

### MI-0070 Tensor.__init__(self, data, _children, _op, requires_grad)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (5):
  - 赋值 self.data ← np.asarray(…)
  - 赋值 self.requires_grad ← requires_grad
  - 赋值 self.grad ← IfExp
  - 赋值 self._backward ← Lambda
  - 赋值 self._prev ← list(…)

### MI-0071 Tensor.shape(self)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 装饰器: property
- 微指令步骤 (1):
  - 返回 self.data.shape

### MI-0072 Tensor._set(self, val)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (1):
  - 赋值 self.data ← np.asarray(…)

### MI-0073 Tensor.__add__(self, other)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (5):
  - 赋值 other ← IfExp
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0074 Tensor.__mul__(self, other)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (5):
  - 赋值 other ← IfExp
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0075 Tensor.__sub__(self, other)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (2):
  - 赋值 other ← IfExp
  - 返回 self add UnaryOp

### MI-0076 Tensor.__truediv__(self, other)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (5):
  - 赋值 other ← IfExp
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0077 Tensor.__rtruediv__(self, other)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (5):
  - 赋值 other ← IfExp
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0078 Tensor.__neg__(self)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (4):
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0079 Tensor.matmul(self, other)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (4):
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0080 Tensor.transpose(self, axes)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (4):
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0081 Tensor.reshape(self, *shape)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (4):
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0082 Tensor.sigmoid(self)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (5):
  - 赋值 s ← 1.0 div 1.0 add np.exp(…)
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0083 Tensor.tanh(self)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (5):
  - 赋值 t ← np.tanh(…)
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0084 Tensor.relu(self)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (5):
  - 赋值 m ← Compare
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0085 Tensor.exp(self)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (5):
  - 赋值 e ← np.exp(…)
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0086 Tensor.log(self)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (4):
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0087 Tensor.mean(self, axis, keepdims)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (4):
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0088 Tensor.sum(self, axis, keepdims)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (4):
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0089 Tensor.max(self, axis, keepdims)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (4):
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0090 Tensor.pow(self, p)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (4):
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0091 Tensor.__getitem__(self, idx)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (4):
  - 赋值 out ← Tensor(…)
  - FunctionDef
  - 赋值 out._backward ← _b
  - 返回 out

### MI-0092 Tensor.backward(self)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 微指令步骤 (6):
  - 赋值 topo ← []
  - 赋值 visited ← set(…)
  - FunctionDef
  - 调用 build(…)
  - 赋值 self.grad ← np.ones_like(…)
  - 遍历 v ∈ reversed(…)

### MI-0093 Tensor.zeros(*shape)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 装饰器: staticmethod
- 微指令步骤 (1):
  - 返回 Tensor(…)

### MI-0094 Tensor.parameter(data)
- 模块: `train\autodiff.py`
- 用途: (无 docstring)  ·  INV 22
- 装饰器: staticmethod
- 微指令步骤 (1):
  - 返回 Tensor(…)

### MI-0095 【类】LoRALinear.LoRALinear(类)
- 模块: `train\lora.py`
- 用途: (无 docstring)  ·  INV 24
- 微指令步骤 (5):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0096 LoRALinear.__init__(self, W, r, alpha)
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

### MI-0097 LoRALinear.trainable_params(self)
- 模块: `train\lora.py`
- 用途: (无 docstring)  ·  INV 24
- 装饰器: property
- 微指令步骤 (1):
  - 返回 [self.A, self.B]

### MI-0098 LoRALinear.forward(self, x)
- 模块: `train\lora.py`
- 用途: (无 docstring)  ·  INV 24
- 微指令步骤 (3):
  - 赋值 base ← x.matmul(…)
  - 赋值 lora ← Call.matmul(…) mult self.scale
  - 返回 base add lora

### MI-0099 LoRALinear.merge(self)
- 模块: `train\lora.py`
- 用途: 合并低秩增量到原权重, 返回新 W (INV-24)。  ·  INV 24
- 微指令步骤 (1):
  - 返回 self.W.data add self.scale mult self.B.data matmult self.A.data

### MI-0100 LoRALinear.merged_linear(self, x)
- 模块: `train\lora.py`
- 用途: (无 docstring)  ·  INV 24
- 微指令步骤 (1):
  - 返回 x.matmul(…)

### MI-0101 【类】GlobalStep.GlobalStep(类)
- 模块: `train\optim.py`
- 用途: 学习率调度状态机 (线性预热 + 余弦退火)。  ·  INV 21,23
- 微指令步骤 (2):
  - FunctionDef
  - FunctionDef

### MI-0102 GlobalStep.__init__(self, lr, total_steps, warmup_steps, min_lr)
- 模块: `train\optim.py`
- 用途: (无 docstring)  ·  INV 21,23
- 微指令步骤 (5):
  - 赋值 self.lr0 ← lr
  - 赋值 self.total ← max(…)
  - 赋值 self.warmup ← warmup_steps
  - 赋值 self.min_lr ← min_lr
  - 赋值 self.t ← 0

### MI-0103 GlobalStep.step(self)
- 模块: `train\optim.py`
- 用途: (无 docstring)  ·  INV 21,23
- 微指令步骤 (3):
  - 赋值 lr ← cosine_schedule(…)
  - 累算 self.t add 1
  - 返回 lr

### MI-0104 cosine_schedule(lr0, step, total, warmup, min_lr)
- 模块: `train\optim.py`
- 用途: 线性预热 + 余弦退火。解析公式, 供 INV(R-L4-02) 对照。  ·  INV 21,23
- 微指令步骤 (4):
  - 判断 BoolOp
  - 判断 Compare
  - 赋值 progress ← step sub warmup div max(…)
  - 返回 min_lr add 0.5 mult lr0 sub min_lr mult 1.0 add np.cos(…)

### MI-0105 【类】AdamW.AdamW(类)
- 模块: `train\optim.py`
- 用途: 解耦权重衰减 AdamW。  ·  INV 21,23  ·  复用 AdamW
- 微指令步骤 (3):
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0106 AdamW.__init__(self, params, lr, betas, eps, weight_decay)
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

### MI-0107 AdamW.step(self)
- 模块: `train\optim.py`
- 用途: (无 docstring)  ·  INV 21,23
- 微指令步骤 (4):
  - 累算 self.step_count add 1
  - 赋值 bc1 ← 1 sub self.b1 pow self.step_count
  - 赋值 bc2 ← 1 sub self.b2 pow self.step_count
  - 遍历 (i, p) ∈ enumerate(…)

### MI-0108 AdamW.zero_grad(self)
- 模块: `train\optim.py`
- 用途: (无 docstring)  ·  INV 21,23
- 微指令步骤 (1):
  - 遍历 p ∈ self.params

### MI-0109 clip_global_norm(params, max_norm, norm_type)
- 模块: `train\optim.py`
- 用途: 全局范数裁剪。返回 (total_norm, triggered)。INV-23。  ·  INV 23
- 微指令步骤 (4):
  - 赋值 norms ← ListComp
  - 赋值 total ← float(…)
  - 判断 BoolOp
  - 返回 [total, False]

### MI-0110 【类】TinyTrainer.TinyTrainer(类)
- 模块: `train\trainer.py`
- 用途: (无 docstring)  ·  INV 26
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

### MI-0111 TinyTrainer.__init__(self, config)
- 模块: `train\trainer.py`
- 用途: (无 docstring)  ·  INV 26
- 微指令步骤 (3):
  - 赋值 self.cfg ← config
  - 声明并赋值 self.params ← Dict
  - 调用 self._build(…)

### MI-0112 TinyTrainer._P(self, name, shape)
- 模块: `train\trainer.py`
- 用途: (无 docstring)  ·  INV 26
- 微指令步骤 (3):
  - 赋值 r ← rng.get_rng(…)
  - 赋值 data ← BinOp.astype(…)
  - 赋值 self.params[…] ← Tensor(…)

### MI-0113 TinyTrainer._build(self)
- 模块: `train\trainer.py`
- 用途: (无 docstring)  ·  INV 26
- 微指令步骤 (7):
  - 赋值 c ← self.cfg
  - 赋值 (V, d) ← [c.vocab_size, c.d_model]
  - 调用 self._P(…)
  - 判断 c.tie_word_embeddings
  - 调用 self._P(…)
  - 赋值 n_kv ← c.n_kv_heads
  - 遍历 l ∈ range(…)

### MI-0114 TinyTrainer._rope(self, x, pos, dim)
- 模块: `train\trainer.py`
- 用途: RoPE 旋转 (偶数维旋转, 支持反向传播)。  ·  INV 26  ·  复用 RoPE
- 微指令步骤 (15):
  - 赋值 freqs ← 1.0 div self.cfg.rope_base pow np.arange(…) div dim floordiv 2
  - 赋值 angles ← np.outer(…)
  - 赋值 cos ← Tensor(…)
  - 赋值 sin ← Tensor(…)
  - 赋值 x1 ← x[…]
  - 赋值 x2 ← x[…]
  - 赋值 rot1 ← x1 mult cos sub x2 mult sin
  - 赋值 rot2 ← x1 mult sin add x2 mult cos
  - 赋值 out ← Tensor(…)
  - 赋值 out.data[…] ← rot1.data
  - 赋值 out.data[…] ← rot2.data
  - FunctionDef
  - 赋值 out._backward ← _b
  - 赋值 out._prev ← [x]
  - 返回 out

### MI-0115 TinyTrainer.forward(self, ids)
- 模块: `train\trainer.py`
- 用途: (无 docstring)  ·  INV 26
- 微指令步骤 (10):
  - 赋值 c ← self.cfg
  - 赋值 (B, T) ← ids.shape
  - 赋值 (d, dh, n_h, n_kv) ← [c.d_model, c.d_head, c.n_heads]
  - 赋值 pos ← np.arange(…)
  - 赋值 x ← self.params[…][…]
  - 遍历 l ∈ range(…)
  - 赋值 ms3 ← BinOp.mean(…)
  - 赋值 x ← x div BinOp.pow(…) mult self.params[…]
  - 赋值 logits ← x.matmul(…)
  - 返回 logits

### MI-0116 TinyTrainer.loss(self, ids, targets)
- 模块: `train\trainer.py`
- 用途: (无 docstring)  ·  INV 26
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

### MI-0117 TinyTrainer.train(self, ids_corpus, steps, lr, batch_size, seq_len, grad_clip, warmup)
- 模块: `train\trainer.py`
- 用途: (无 docstring)  ·  INV 26
- 微指令步骤 (7):
  - 赋值 params ← ListComp
  - 赋值 opt ← AdamW(…)
  - 赋值 sched ← GlobalStep(…)
  - 赋值 losses ← []
  - 赋值 N ← len(…) sub seq_len
  - 遍历 s ∈ range(…)
  - 返回 losses

### MI-0118 TinyTrainer.to_numpy_params(self)
- 模块: `train\trainer.py`
- 用途: (无 docstring)  ·  INV 26
- 微指令步骤 (4):
  - 赋值 out ← Dict
  - 遍历 (k, v) ∈ self.params.items(…)
  - 判断 self.cfg.tie_word_embeddings
  - 返回 out

### MI-0119 TinyTrainer.save(self, dir_path)
- 模块: `train\trainer.py`
- 用途: (无 docstring)  ·  INV 26
- 微指令步骤 (7):
  - 调用 os.makedirs(…)
  - 赋值 params ← self.to_numpy_params(…)
  - 调用 np.savez(…)
  - 上下文管理 (with)
  - 赋值 manifest ← Dict
  - 上下文管理 (with)
  - 返回 dir_path

## L5 检索/记忆 (Retrieval/Memory)

### MI-0120 【类】BM25.BM25(类)
- 模块: `retrieval\bm25.py`
- 用途: (无 docstring)  ·  INV 27
- 微指令步骤 (5):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0121 BM25.__init__(self, k1, b)
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

### MI-0122 BM25.fit(self, corpus)
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

### MI-0123 BM25._score(self, query, doc_idx)
- 模块: `retrieval\bm25.py`
- 用途: (无 docstring)  ·  INV 27
- 微指令步骤 (5):
  - 赋值 tf ← Counter(…)
  - 赋值 dl ← self.doc_len[…]
  - 赋值 score ← 0.0
  - 遍历 w ∈ query
  - 返回 score

### MI-0124 BM25.search(self, query, top_k)
- 模块: `retrieval\bm25.py`
- 用途: (无 docstring)  ·  INV 27
- 微指令步骤 (3):
  - 赋值 scores ← ListComp
  - 调用 scores.sort(…)
  - 返回 scores[…]

### MI-0125 BM25.reference(corpus, query, k1, b)
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

### MI-0126 【类】DenseIndex.DenseIndex(类)
- 模块: `retrieval\dense.py`
- 用途: (无 docstring)  ·  INV 29
- 微指令步骤 (5):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0127 DenseIndex.__init__(self, dim, seed)
- 模块: `retrieval\dense.py`
- 用途: (无 docstring)  ·  INV 29
- 微指令步骤 (3):
  - 赋值 self.dim ← dim
  - 赋值 self.rng ← np.random.default_rng(…)
  - 赋值 self.proj ← self.rng.standard_normal(…)

### MI-0128 DenseIndex._embed(self, tokens)
- 模块: `retrieval\dense.py`
- 用途: (无 docstring)  ·  INV 29
- 微指令步骤 (4):
  - 赋值 vec ← np.zeros(…)
  - 遍历 t ∈ tokens
  - 赋值 n ← np.linalg.norm(…)
  - 返回 IfExp

### MI-0129 DenseIndex.fit(self, corpus)
- 模块: `retrieval\dense.py`
- 用途: (无 docstring)  ·  INV 29
- 微指令步骤 (3):
  - 赋值 self.docs ← ListComp
  - 赋值 self.matrix ← IfExp
  - 返回 self

### MI-0130 DenseIndex.search(self, query, top_k)
- 模块: `retrieval\dense.py`
- 用途: (无 docstring)  ·  INV 29
- 微指令步骤 (5):
  - 赋值 q ← self._embed(…)
  - 判断 Compare
  - 赋值 sims ← self.matrix matmult q
  - 赋值 idx ← np.argsort(…)[…]
  - 返回 ListComp

### MI-0131 DenseIndex.brute(matrix, q, top_k)
- 模块: `retrieval\dense.py`
- 用途: (无 docstring)  ·  INV 29
- 装饰器: staticmethod
- 微指令步骤 (3):
  - 赋值 sims ← matrix matmult q
  - 赋值 idx ← np.argsort(…)[…]
  - 返回 ListComp

### MI-0132 【类】HNSW.HNSW(类)
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

### MI-0133 HNSW.__init__(self, M, ef_construction, ef_search, ml, seed)
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

### MI-0134 HNSW._dist(self, a, b)
- 模块: `retrieval\hnsw.py`
- 用途: (无 docstring)  ·  INV 28
- 微指令步骤 (1):
  - 返回 float(…)

### MI-0135 HNSW._level(self)
- 模块: `retrieval\hnsw.py`
- 用途: (无 docstring)  ·  INV 28
- 微指令步骤 (1):
  - 返回 int(…)

### MI-0136 HNSW.add(self, vec)
- 模块: `retrieval\hnsw.py`
- 用途: (无 docstring)  ·  INV 28
- 微指令步骤 (8):
  - 赋值 idx ← len(…)
  - 调用 self.data.append(…)
  - 赋值 level ← self._level(…)
  - 赋值 self.layers[…] ← []
  - 判断 BoolOp
  - 赋值 ep ← IfExp
  - 遍历 lc ∈ range(…)
  - 返回 idx

### MI-0137 HNSW._connect(self, a, b, layer)
- 模块: `retrieval\hnsw.py`
- 用途: (无 docstring)  ·  INV 28
- 微指令步骤 (4):
  - 调用 self.layers.setdefault(…)
  - 调用 self.layers.setdefault(…)
  - 判断 Compare
  - 判断 Compare

### MI-0138 HNSW._search_layer(self, q, entry, ef, layer)
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

### MI-0139 HNSW._dist_id(self, q, i)
- 模块: `retrieval\hnsw.py`
- 用途: (无 docstring)  ·  INV 28
- 微指令步骤 (1):
  - 返回 float(…)

### MI-0140 HNSW.build(self, data)
- 模块: `retrieval\hnsw.py`
- 用途: (无 docstring)  ·  INV 28
- 微指令步骤 (2):
  - 遍历 v ∈ data
  - 返回 self

### MI-0141 HNSW.search(self, q, k)
- 模块: `retrieval\hnsw.py`
- 用途: (无 docstring)  ·  INV 28
- 微指令步骤 (4):
  - 判断 Compare
  - 赋值 res ← self._search_layer(…)
  - 赋值 out ← ListComp
  - 返回 sorted(…)[…]

### MI-0142 【类】CrossEncoderReranker.CrossEncoderReranker(类)
- 模块: `retrieval\rerank.py`
- 用途: (无 docstring)
- 微指令步骤 (4):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0143 CrossEncoderReranker.__init__(self, dim, seed)
- 模块: `retrieval\rerank.py`
- 用途: (无 docstring)
- 微指令步骤 (2):
  - 赋值 self.dim ← dim
  - 赋值 self.seed ← seed

### MI-0144 CrossEncoderReranker._token_vec(self, t)
- 模块: `retrieval\rerank.py`
- 用途: (无 docstring)
- 微指令步骤 (4):
  - 赋值 h ← hash(…) bitand 4294967295 bitxor self.seed bitand 4294967295
  - 赋值 v ← Call.standard_normal(…)
  - 赋值 n ← np.linalg.norm(…)
  - 返回 IfExp

### MI-0145 CrossEncoderReranker._feat(self, tokens)
- 模块: `retrieval\rerank.py`
- 用途: (无 docstring)
- 微指令步骤 (4):
  - 赋值 v ← np.zeros(…)
  - 遍历 t ∈ tokens
  - 赋值 n ← np.linalg.norm(…)
  - 返回 IfExp

### MI-0146 CrossEncoderReranker.rerank(self, query, docs, top_k)
- 模块: `retrieval\rerank.py`
- 用途: (无 docstring)
- 微指令步骤 (4):
  - 赋值 qv ← self._feat(…)
  - 赋值 scored ← ListComp
  - 调用 scored.sort(…)
  - 返回 scored[…]

### MI-0147 reciprocal_rank_fusion(rank_lists, k)
- 模块: `retrieval\rrf.py`
- 用途: rank_lists: 多个检索器各自的(已排序)文档 id 列表。返回融合排序。  ·  INV 30
- 微指令步骤 (3):
  - 声明并赋值 scores ← Dict
  - 遍历 rl ∈ rank_lists
  - 返回 sorted(…)

## L6 智能体 (Agent)

### MI-0148 【类】Episode.Episode(类)
- 模块: `agent\memory.py`
- 用途: (无 docstring)
- 微指令步骤 (2):
  - 声明并赋值 ts ← ∅
  - 声明并赋值 text ← ∅

### MI-0149 【类】MemoryStore.MemoryStore(类)
- 模块: `agent\memory.py`
- 用途: (无 docstring)
- 微指令步骤 (6):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0150 MemoryStore.__init__(self)
- 模块: `agent\memory.py`
- 用途: (无 docstring)
- 微指令步骤 (2):
  - 声明并赋值 self.episodic ← []
  - 声明并赋值 self.semantic ← Dict

### MI-0151 MemoryStore.remember_event(self, text)
- 模块: `agent\memory.py`
- 用途: (无 docstring)
- 微指令步骤 (1):
  - 调用 self.episodic.append(…)

### MI-0152 MemoryStore.remember_fact(self, key, value)
- 模块: `agent\memory.py`
- 用途: (无 docstring)
- 微指令步骤 (1):
  - 赋值 self.semantic[…] ← value

### MI-0153 MemoryStore.recall_events(self, query, limit)
- 模块: `agent\memory.py`
- 用途: (无 docstring)
- 微指令步骤 (2):
  - 赋值 out ← ListComp
  - 返回 out[…]

### MI-0154 MemoryStore.recall_fact(self, key)
- 模块: `agent\memory.py`
- 用途: (无 docstring)
- 微指令步骤 (1):
  - 返回 self.semantic.get(…)

### MI-0155 MemoryStore.__len__(self)
- 模块: `agent\memory.py`
- 用途: (无 docstring)
- 微指令步骤 (1):
  - 返回 len(…) add len(…)

### MI-0156 【类】ReActAgent.ReActAgent(类)
- 模块: `agent\react.py`
- 用途: (无 docstring)
- 微指令步骤 (5):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0157 ReActAgent.__init__(self, llm_fn, tools, max_steps, memory)
- 模块: `agent\react.py`
- 用途: (无 docstring)
- 微指令步骤 (5):
  - 赋值 self.llm ← llm_fn
  - 赋值 self.tools ← tools
  - 赋值 self.max_steps ← max_steps
  - 赋值 self.memory ← memory
  - 声明并赋值 self.history ← []

### MI-0158 ReActAgent._system(self)
- 模块: `agent\react.py`
- 用途: (无 docstring)
- 微指令步骤 (4):
  - 赋值 lines ← ["你是一个使用工具的 ReAct …"]
  - 遍历 t ∈ self.tools.describe(…)
  - 调用 lines.append(…)
  - 返回 Constant.join(…)

### MI-0159 ReActAgent._parse_action(self, text)
- 模块: `agent\react.py`
- 用途: (无 docstring)
- 微指令步骤 (5):
  - 赋值 m ← re.search(…)
  - 判断 m
  - 赋值 m ← re.search(…)
  - 判断 m
  - 返回 ["none", None, text.strip(…)]

### MI-0160 ReActAgent._parse_args(raw)
- 模块: `agent\react.py`
- 用途: (无 docstring)
- 装饰器: staticmethod
- 微指令步骤 (4):
  - 赋值 args ← Dict
  - 判断 UnaryOp
  - 遍历 kv ∈ re.findall(…)
  - 返回 args

### MI-0161 ReActAgent.run(self, question, reflexion)
- 模块: `agent\react.py`
- 用途: (无 docstring)
- 微指令步骤 (4):
  - 赋值 self.history ← [Dict, Dict]
  - 遍历 step ∈ range(…)
  - 判断 reflexion
  - 返回 "未能在步数内完成。"

### MI-0162 【类】SafeCodeExecutor.SafeCodeExecutor(类)
- 模块: `agent\sandbox.py`
- 用途: (无 docstring)  ·  INV 34
- 微指令步骤 (3):
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0163 SafeCodeExecutor.__init__(self, timeout, workdir)
- 模块: `agent\sandbox.py`
- 用途: (无 docstring)  ·  INV 34
- 微指令步骤 (2):
  - 赋值 self.timeout ← timeout
  - 赋值 self.workdir ← BoolOp

### MI-0164 SafeCodeExecutor.static_check(self, code)
- 模块: `agent\sandbox.py`
- 用途: 返回违规列表 (非空 ⇒ 拒绝执行)。  ·  INV 34
- 微指令步骤 (4):
  - 声明并赋值 violations ← []
  - 异常保护 (try/except)
  - 遍历 node ∈ ast.walk(…)
  - 返回 violations

### MI-0165 SafeCodeExecutor.run(self, code)
- 模块: `agent\sandbox.py`
- 用途: (无 docstring)  ·  INV 34
- 微指令步骤 (4):
  - 赋值 v ← self.static_check(…)
  - 判断 v
  - 赋值 runner ← "import sys, json…"
  - 异常保护 (try/except)

### MI-0166 【类】ToolSpec.ToolSpec(类)
- 模块: `agent\tools.py`
- 用途: (无 docstring)  ·  INV 33
- 微指令步骤 (6):
  - 声明并赋值 name ← ∅
  - 声明并赋值 description ← ∅
  - 声明并赋值 schema ← ∅
  - 声明并赋值 func ← field(…)
  - FunctionDef
  - FunctionDef

### MI-0167 ToolSpec.validate(self, args)
- 模块: `agent\tools.py`
- 用途: (无 docstring)  ·  INV 33
- 微指令步骤 (1):
  - 返回 validate_args(…)

### MI-0168 ToolSpec.__call__(self, args)
- 模块: `agent\tools.py`
- 用途: (无 docstring)  ·  INV 33
- 微指令步骤 (3):
  - 赋值 errs ← self.validate(…)
  - 判断 errs
  - 返回 self.func(…)

### MI-0169 validate_args(schema, args)
- 模块: `agent\tools.py`
- 用途: 极简 JSON Schema 校验 (type / required / enum / minimum)。INV-33。  ·  INV 33
- 微指令步骤 (6):
  - 声明并赋值 errs ← []
  - 赋值 props ← schema.get(…)
  - 赋值 required ← schema.get(…)
  - 遍历 k ∈ required
  - 遍历 (k, v) ∈ args.items(…)
  - 返回 errs

### MI-0170 【类】Calculator.Calculator(类)
- 模块: `agent\tools.py`
- 用途: 安全算术计算器 (仅允许四则 + 括号, 经 AST 白名单)。  ·  INV 33
- 微指令步骤 (3):
  - 赋值 ALLOWED ← [ast.Expression, ast.BinOp, ast.UnaryOp]
  - FunctionDef
  - FunctionDef

### MI-0171 Calculator._eval(node, names)
- 模块: `agent\tools.py`
- 用途: (无 docstring)  ·  INV 33
- 装饰器: staticmethod
- 微指令步骤 (5):
  - 判断 isinstance(…)
  - 判断 isinstance(…)
  - 判断 isinstance(…)
  - 判断 isinstance(…)
  - 抛出异常

### MI-0172 Calculator.run(cls, expression)
- 模块: `agent\tools.py`
- 用途: (无 docstring)  ·  INV 33
- 装饰器: classmethod
- 微指令步骤 (3):
  - 赋值 tree ← ast.parse(…)
  - 遍历 node ∈ ast.walk(…)
  - 返回 float(…)

### MI-0173 【类】ToolRegistry.ToolRegistry(类)
- 模块: `agent\tools.py`
- 用途: (无 docstring)  ·  INV 33
- 微指令步骤 (5):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0174 ToolRegistry.__init__(self)
- 模块: `agent\tools.py`
- 用途: (无 docstring)  ·  INV 33
- 微指令步骤 (1):
  - 声明并赋值 self.tools ← Dict

### MI-0175 ToolRegistry.register(self, spec)
- 模块: `agent\tools.py`
- 用途: (无 docstring)  ·  INV 33
- 微指令步骤 (1):
  - 赋值 self.tools[…] ← spec

### MI-0176 ToolRegistry.get(self, name)
- 模块: `agent\tools.py`
- 用途: (无 docstring)  ·  INV 33
- 微指令步骤 (1):
  - 返回 self.tools.get(…)

### MI-0177 ToolRegistry.call(self, name, args)
- 模块: `agent\tools.py`
- 用途: (无 docstring)  ·  INV 33
- 微指令步骤 (2):
  - 赋值 spec ← self.tools[…]
  - 返回 spec(…)

### MI-0178 ToolRegistry.describe(self)
- 模块: `agent\tools.py`
- 用途: (无 docstring)  ·  INV 33
- 微指令步骤 (1):
  - 返回 ListComp

### MI-0179 default_registry()
- 模块: `agent\tools.py`
- 用途: (无 docstring)  ·  INV 33
- 微指令步骤 (3):
  - 赋值 reg ← ToolRegistry(…)
  - 调用 reg.register(…)
  - 返回 reg

## L7 服务/编排 (Service/Orchestration)

### MI-0180 【类】OpenAIServer.OpenAIServer(类)
- 模块: `serve\api.py`
- 用途: (无 docstring)  ·  INV 36,37
- 微指令步骤 (6):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0181 OpenAIServer.__init__(self, model, tokenizer, max_new_tokens, seed)
- 模块: `serve\api.py`
- 用途: (无 docstring)  ·  INV 36,37
- 微指令步骤 (5):
  - 赋值 self.model ← model
  - 赋值 self.tok ← tokenizer
  - 赋值 self.max_new_tokens ← max_new_tokens
  - 赋值 self.sampler ← Sampler(…)
  - 赋值 self.model_name ← "ifai-flagship"

### MI-0182 OpenAIServer._gen_tokens(self, prompt, max_new, temperature, top_p, stream)
- 模块: `serve\api.py`
- 用途: 生成 token 迭代器。stream 仅决定调用方是否增量消费;  ·  INV 37
- 微指令步骤 (3):
  - 赋值 ids ← self.tok.encode(…)
  - 判断 UnaryOp
  - 遍历 _ ∈ range(…)

### MI-0183 OpenAIServer.completions(self, prompt, temperature, top_p, max_tokens, stream)
- 模块: `serve\api.py`
- 用途: (无 docstring)  ·  INV 36,37
- 微指令步骤 (4):
  - 赋值 max_new ← BoolOp
  - 赋值 text ← Constant.join(…)
  - 赋值 obj ← Dict
  - 返回 obj

### MI-0184 OpenAIServer.chat(self, messages, **kw)
- 模块: `serve\api.py`
- 用途: (无 docstring)  ·  INV 36,37
- 微指令步骤 (5):
  - 赋值 prompt ← Constant.join(…)
  - 赋值 comp ← self.completions(…)
  - 赋值 text ← comp[…][…][…]
  - 赋值 n ← len(…)
  - 返回 Dict

### MI-0185 OpenAIServer.models(self)
- 模块: `serve\api.py`
- 用途: (无 docstring)  ·  INV 36,37
- 微指令步骤 (1):
  - 返回 Dict

### MI-0186 OpenAIServer.route(self, method, path, body)
- 模块: `serve\api.py`
- 用途: (无 docstring)  ·  INV 36,37
- 微指令步骤 (4):
  - 判断 BoolOp
  - 判断 BoolOp
  - 判断 BoolOp
  - 返回 [404, Dict]

### MI-0187 create_handler(server)
- 模块: `serve\api.py`
- 用途: (无 docstring)  ·  INV 36,37
- 微指令步骤 (2):
  - ClassDef
  - 返回 Handler

### MI-0188 serve(server, host, port)
- 模块: `serve\api.py`
- 用途: (无 docstring)  ·  INV 36,37
- 微指令步骤 (2):
  - 赋值 httpd ← ThreadingHTTPServer(…)
  - 调用 httpd.serve_forever(…)

## L8 评估/对齐 (Evaluation/Alignment)

### MI-0189 benchmark_generation(generate_fn, prompt, max_new_tokens, runs)
- 模块: `eval\benchmark.py`
- 用途: generate_fn(prompt, n) -> 生成的 token 数; 返回吞吐统计。
- 微指令步骤 (6):
  - 赋值 (times, toks) ← [[], []]
  - 遍历 _ ∈ range(…)
  - 赋值 tps ← ListComp
  - 赋值 arr ← np.array(…)
  - 赋值 cv ← IfExp
  - 返回 Dict

### MI-0190 write_baseline(path, stats)
- 模块: `eval\benchmark.py`
- 用途: (无 docstring)
- 微指令步骤 (2):
  - 赋值 lines ← ["# 性能基线 (BASELINE…", "", JoinedStr]
  - 上下文管理 (with)

### MI-0191 【类】GSM8K_Mini.GSM8K_Mini(类)
- 模块: `eval\harness.py`
- 用途: 内置 20 道极简算术题 (离线, 不随仓下载)。
- 微指令步骤 (3):
  - 赋值 DATA ← [["小明有 3 个苹果, 又买了 5…", 8], ["7 减去 2 等于多少?", 5], ["一个盒子有 4 排, 每排 6 …", 24]]
  - FunctionDef
  - FunctionDef

### MI-0192 GSM8K_Mini.extract_number(text)
- 模块: `eval\harness.py`
- 用途: (无 docstring)
- 装饰器: staticmethod
- 微指令步骤 (2):
  - 赋值 nums ← re.findall(…)
  - 返回 IfExp

### MI-0193 GSM8K_Mini.evaluate(self, answer_fn)
- 模块: `eval\harness.py`
- 用途: (无 docstring)
- 微指令步骤 (4):
  - 赋值 correct ← 0
  - 赋值 samples ← []
  - 遍历 (q, gold) ∈ self.DATA
  - 返回 Dict

### MI-0194 【类】RegressionBaseline.RegressionBaseline(类)
- 模块: `eval\harness.py`
- 用途: 回归门禁: 当前吞吐相对 BASELINE.md 掉落 >15% ⇒ 红灯。
- 微指令步骤 (2):
  - FunctionDef
  - FunctionDef

### MI-0195 RegressionBaseline.parse_baseline(path)
- 模块: `eval\harness.py`
- 用途: (无 docstring)
- 装饰器: staticmethod
- 微指令步骤 (3):
  - 赋值 out ← Dict
  - 异常保护 (try/except)
  - 返回 out

### MI-0196 RegressionBaseline.check(current, baseline_path, threshold)
- 模块: `eval\harness.py`
- 用途: (无 docstring)
- 装饰器: staticmethod
- 微指令步骤 (5):
  - 赋值 base ← RegressionBaseline.parse_baseline(…)
  - 判断 Compare
  - 赋值 drop ← base[…] sub current[…] div base[…]
  - 赋值 status ← IfExp
  - 返回 Dict

### MI-0197 perplexity(model, token_ids)
- 模块: `eval\perplexity.py`
- 用途: 给定 token 序列计算困惑度 (逐 token 用 logits_next, 因果)。  ·  INV 39
- 微指令步骤 (6):
  - 赋值 ids ← list(…)
  - 赋值 total_nll ← 0.0
  - 赋值 n ← 0
  - 遍历 i ∈ range(…)
  - 赋值 mean_nll ← total_nll div max(…)
  - 返回 float(…)

### MI-0198 perplexity_manual(model, token_ids)
- 模块: `eval\perplexity.py`
- 用途: 纯手工 exp(mean NLL) 参考 (INV-39 对照)。  ·  INV 39
- 微指令步骤 (5):
  - 赋值 ids ← list(…)
  - 赋值 nll ← 0.0
  - 赋值 n ← 0
  - 遍历 i ∈ range(…)
  - 返回 float(…)

## L9 应用/接口 (Application/Interface)

### MI-0199 【类】IFAI.IFAI(类)
- 模块: `sdk.py`
- 用途: 旗舰 AI 高层客户端。
- 微指令步骤 (4):
  - FunctionDef
  - FunctionDef
  - FunctionDef
  - FunctionDef

### MI-0200 IFAI.__init__(self, model_dir, seed)
- 模块: `sdk.py`
- 用途: 载入模型与分词器。
- 微指令步骤 (5):
  - 调用 rng.seed(…)
  - 赋值 tok_path ← os.path.join(…)
  - 赋值 self.tok ← Tokenizer.from_json(…)
  - 赋值 self.model ← load_model(…)
  - 赋值 self.sampler ← Sampler(…)

### MI-0201 IFAI.chat(self, prompt, max_new, temperature)
- 模块: `sdk.py`
- 用途: 生成续写文本 (贪心/采样)。
- 微指令步骤 (3):
  - 赋值 ids ← BoolOp
  - 遍历 _ ∈ range(…)
  - 返回 BoolOp

### MI-0202 IFAI.search(self, query, corpus, top_k)
- 模块: `sdk.py`
- 用途: BM25 检索, 返回 [(doc_idx, score)...]。  ·  复用 BM25
- 微指令步骤 (2):
  - 赋值 bm25 ← Call.fit(…)
  - 返回 bm25.search(…)

### MI-0203 IFAI.run(self, question, max_steps)
- 模块: `sdk.py`
- 用途: 运行 ReAct 智能体 (内置计算器工具)。  ·  复用 ReAct
- 微指令步骤 (3):
  - FunctionDef
  - 赋值 agent ← ReActAgent(…)
  - 返回 agent.run(…)
