# L2 · 模型架构 (Model Architecture) — 设计说明

## 设计目标

L2 定义 **ifai** 的解码器-only Transformer 架构与分词器：从配置、分词到完整前向
（`Transformer.forward`）与参数序列化。该层把 L0 的算子「组装」成标准 LLM 主干，
并严格复用 Transformer 家族的成熟组件，而非自创结构。

本层覆盖 `ifai/models/`（config.py、tokenizer.py、transformer.py）。

## 复用世界顶级成果（零重写 · 集大成）

- **Transformer 解码器**（Vaswani et al. 2017，"Attention Is All You Need"）：复用
  多头因果自注意力 + 残差前馈的主干，作为整体骨架。
- **Pre-LN / RMSNorm**（Zhang & Sennrich 2019）：复用在注意力与 FFN 前的 RMSNorm
  归一化（配合 L0 的 `rms_norm`），而非 Post-LN，提升训练稳定性。
- **RoPE 旋转位置编码**（Su et al. 2024）+ **NTK-aware 插值**：复用 L0 的旋转编码；
  工程上由 `ModelConfig.rope_base / rope_scaling` 驱动，支持上下文外推。
- **GQA 分组查询注意力**（Ainslie et al. 2023，"GQA: Training Generalized Multi-Query
  Transformer Models"）：复用「多 Q 头共享少量 KV 头」结构；`n_kv_heads` 整除约束下，
  `n_kv==n_q` 时严格退化为 MHA（INV-10）。
- **SwiGLU FFN**（Shazeer 2020）：复用 L0 的融合 SwiGLU 作为前馈激活（可切 `gelu`）。
- **MoE / Switch 稀疏专家**（Fedus et al. 2022，"Switch Transformers"）：复用 top-k
  路由 + load 均衡思想；工程上对选中专家分数做 softmax 使「路由概率和=1、token 分配
  守恒」（INV-14），`n_experts/moe_topk` 由配置约束。
- **权重绑定 Weight Tying**（Press & Wolf 2017，"Using the Output Embedding to Improve
  Language Models"）：复用「输入 embedding 与 lm_head 共享权重」；工程上令二者为同一
  对象（INV-12），`tie_word_embeddings` 可关。
- **BPE 字节对编码**（Sennrich et al. 2016，"Neural Machine Translation of Rare Words
  with Subword Units"）：复用以合并频率最高相邻符号为核心的 BPE 训练算法（`train()`）。
- **Byte-level BPE**（Radford et al. 2019，GPT-2）：复用「字符级符号空间规避 UTF-8
  边界」思路，工程上以字符为初始符号、`encode→decode` 保真。
- **配置驱动范式**（HuggingFace `transformers.PretrainedConfig`）：复用「dataclass +
  JSON 往返」配置范式，`to_json/from_json` 保证两次加载 bitwise 一致（R-L2-05）。

## 关键不变量 / 验收

- **INV-10** GQA 在 `n_kv_heads == n_heads` 时严格 == MHA（K/V 经 `np.repeat` 无变化）。
- **INV-12** 输出 embedding 与输入 embedding 是同一对象（tie）；`load` 时重建该引用。
- **INV-13** 参数量精确守恒：`count_params()` 与 `save` 时记录的 `total_params` 一致，
  由配置完全决定。
- **INV-14** MoE 路由概率和=1 且 token 分配守恒（仅对选中专家 softmax）。
- **R-L2-05** 配置 JSON 两次加载 bitwise 一致。
- **R-L2-06** 权重 `np.savez` + `sha256` manifest 往返误差=0，保证完整性可校验。

## 工程取舍

1. **解码器-only + Pre-LN**：只实现自回归解码器（无交叉注意力），Pre-LN 顺序，契合
   现代 LLM 主流且数值更稳。
2. **GQA 默认复用 MHA 路径**：`n_kv_heads=None` 时等于 `n_heads`，单一代码路径同时
   覆盖 MHA 与 GQA，避免重复实现（INV-10 即此路径的验收）。
3. **权重 sha256 manifest**：`save` 写入 `manifest.json`（含 `params_sha256`），`load`
   可选校验，使模型文件具备防篡改与可审计性，契合本栈「可验证」哲学。
4. **CPU 友好的微型默认配置**：默认 `d_model=512, n_layers=6, vocab=32000`，使
   `ifai demo` 可在数十秒内跑通推理/训练，而非追求大模型规模。
