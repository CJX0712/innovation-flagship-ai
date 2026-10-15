# L5 · 检索 / 记忆 (Retrieval & Memory) — 设计说明

## 设计目标

L5 提供混合检索与重排能力：稀疏（BM25）、稠密（双塔）、图索引（HNSW）、交互式
重排（Cross-Encoder）与多路融合（RRF）。目标是在**零依赖、可复现**前提下，复现
现代 RAG 检索管线的核心算法，并用暴力基准严格验收。

本层覆盖 `ifai/retrieval/`（bm25.py、dense.py、hnsw.py、rerank.py、rrf.py）。

## 复用世界顶级成果（零重写 · 集大成）

- **BM25 / Okapi BM25**（Robertson & Zaragoza 2009，"The Probabilistic Relevance
  Framework: BM25 and Beyond"）：复用 `idf = log(1+(n-f+0.5)/(f+0.5))` 与
  `score = idf·f·(k1+1)/(f+k1·(1-b+b·dl/avgdl))`，默认 `k1=1.5, b=0.75`。INV-27：
  与朴素参考实现 Top-10 重合度 ≥ 95%（`BM25.reference` 作对照）。
- **稠密双塔检索 DPR**（Karpukhin et al. 2020，"Dense Passage Retrieval"）：复用
  「查询/文档在共享嵌入空间做余弦相似度」思想；工程上以「哈希-加权嵌入
  （hashing trick + TF 加权）」作为零依赖可复现的稠密编码器，TopK 与暴力一致
  （INV-29）。
- **HNSW 分层可导航小世界图**（Malkov & Yashunin 2018，"Efficient and robust
  approximate nearest neighbor search using HNSW"）：复用多层图 + 随机 level（ln
  分布）+ 贪心搜索 + `ef` 候选集；INV-28：与暴力 TopK 相比 `recall@10 ≥ 0.95`，
  QPS ≥ 暴力 5×。
- **Cross-Encoder 重排**（Nogueira & Cho 2019，"Passage Re-ranking with BERT"）：复用
  「查询与文档交互打分」范式；工程上以「词→稳定单位随机向量、文档=词向量和、点积」
  的随机特征核实现确定性可复现的相关性打分，凸现词项重叠。
- **RRF 倒数排名融合**（Cormack et al. 2009，"Reciprocal Rank Fusion outperforms
  Condorcet and individual Rank Learning Methods"）：复用 `score = Σ 1/(k+rank)`
  （k=60）融合多路排序；INV-30：与解析公式一致（≤1e-12）。
- **Hashing Trick 哈希技巧**（Weinberger et al. 2009，"Feature Hashing for Large Scale
  Multitask Learning"）：复用「词哈希到固定维、可加性累加」构造零依赖稠密向量，
  保证可复现且无需预训练 encoder 权重。

## 关键不变量 / 验收

- **INV-27** BM25 与参考实现 Top-10 重合度 ≥ 95%。
- **INV-28** HNSW 相比暴力 TopK `recall@10 ≥ 0.95`，QPS ≥ 暴力 5×。
- **INV-29** 稠密索引 TopK 与暴力余弦排序 100% 一致（零近似）。
- **INV-30** RRF `fused = Σ 1/(k+rank)` 与解析公式一致（≤1e-12）。
- **INV-31** 多路检索器（BM25 + 稠密）输出 id 空间一致，可被 RRF 正确融合。
- **INV-32** 重排器将「含查询词项」的文档排在无关文档之前（`test_rerank_top1`）。

## 工程取舍

1. **零依赖的可复现编码器**：稠密/重排均用「固定种子随机投影/词哈希向量」而非预训练
   模型权重，使检索在离线、无网络、无大型权重下仍可运行与复现（契合 INV 确定性）。
2. **每个检索器都配暴力对照**：`BM25.reference`、`DenseIndex.brute` 与 HNSW 的暴力
   召回基准并存，使 INV-27/28/29 可被严格证明。
3. **混合检索 + RRF 融合**：稀疏（BM25，精确词项）与稠密（语义）互补，RRF 以无监督
   方式融合，避免为融合学习额外权重。
4. **可加性哈希嵌入**：文档向量为词向量之和，支持增量更新与 O(1) 拼接，适配边检索
   边索引的场景。
