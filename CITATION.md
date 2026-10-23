# 引用 (Citation)

如果您在研究中使用了 **ifai**，请引用：

```bibtex
@software{ifai2026,
  title  = {ifai: 创新旗舰 AI — 零重写·集大成 全栈参考实现},
  author = {晨星 (Morningstar)},
  year   = {2026},
  url    = {https://github.com/CJX0712/innovation-flagship-ai},
  license = {Apache-2.0}
}
```

## 致已知的奠基性工作（零重写 · 集大成）

本项目在每一层复用并标注了以下公认成果（详见 `docs/PRIOR_ART.md` 与各模块 docstring）：

- **Transformer / 注意力**：Vaswani et al. 2017; Su et al. 2024 (RoPE); Shazeer 2020 (GLU/SwiGLU); Zhang & Sennrich 2019 (RMSNorm); Ainslie et al. 2023 (GQA); Fedus et al. 2022 (MoE/Switch); Press & Wolf 2017 (权重绑定).
- **算子内核**：Dao et al. 2022 (FlashAttention 在线 softmax).
- **推理系统**：Kwon et al. 2023 (PagedAttention/vLLM); Yu et al. 2022 (Orca 连续批处理); Leviathan et al. 2023 / Chen et al. 2023 (投机解码 Medusa/EAGLE).
- **训练**：Loshchilov & Hutter 2019 (AdamW); Kingma & Ba 2015 (Adam); Hu et al. 2022 (LoRA); Karpathy (nanogpt 极简训练循环); Griewank & Walther 2008 (反向模式自动微分).
- **检索**：Robertson & Zaragoza 2009 (BM25); Malkov & Yashunin 2018 (HNSW); Cormack et al. 2009 (RRF); Nogueira & Cho 2019 (Cross-Encoder 重排).
- **检索（续）**：Carbonell & Goldstein 1998 (MMR 多样性重排); Karpukhin et al. 2020 (DPR 双塔); Zep/Graphiti 2024–2026 (bi-temporal 事实有效期); Letta/MemGPT — Packer et al. 2023 (arXiv:2310.08560, OS 分层记忆); Hindsight — Vectorize.io 2026 (MIT, LongMemEval 91.4%, 多策略并行检索 + reflect 合成).
- **智能体**：Yao et al. 2022 (ReAct); Shinn et al. 2023 (Reflexion); Packer et al. 2023 (MemGPT 记忆).
- **评测**：Jelinek et al. 1977 (困惑度); MLPerf (吞吐/延迟基准方法论); Cobbe et al. 2021 (GSM8K).
- **分词**：Sennrich et al. 2016 (BPE); Radford et al. 2019 (byte-level BPE).
- **服务**：OpenAI REST / SSE 协议。

本软件依 Apache-2.0 发布；上述工作的版权与许可归各自作者所有，本项目仅作工程复用与教学性集成。
