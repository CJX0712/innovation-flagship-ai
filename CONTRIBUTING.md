# 贡献指南 (Contributing)

感谢关注 **ifai**！本项目的核心原则是 **「零重写 · 集大成」**：任何新增或修改
都必须站在全球公认的最强成果之上，不得从零造轮子。

## 🧰 开发环境

```bash
python -m venv .venv && source .venv/bin/activate   # 或复用已有 venv
pip install -e .[dev]                                 # numpy + pytest 等
```

## 🧪 本地校验（提交前必跑）

```bash
pytest tests/                        # 39 项黄金测试 (INV 对照, 须全绿)
python scripts/audit_zero_rewrite.py  # 零重写审计 (十层须有引用, 否则中断)
python -m ifai.cli doctor            # 运行环境自检
```

## 📐 编码规范

1. **零重写优先**：新增算法前，先在 docstring 标注复用的论文/系统（作者 + 年份 +
   方法名），并通过 `audit_zero_rewrite.py` 校验。无标注视为违规。
2. **可验证不变量（INV）**：每个关键算法都至少对应一条可独立复现的不变量
   （解析解 vs 数值解、朴素 vs 优化、参照 vs 主实现、有限差分 vs 解析梯度……），
   配套黄金测试写入 `tests/`。
3. **依赖纪律**：仅 `numpy` 为硬依赖；torch 等仅放 `[dev]` 可选；不引入破坏
   CPU 推理路径的库。
4. **可复现**：所有随机性走 `ifai.rng`（PCG64），新增随机逻辑须复用而非自建 RNG。
5. **文档同步**：接口或架构变更须同步更新 `docs/` 与 `README.md`。

## 🔀 提交流程

1. Fork → 自建分支 → 提交（清晰的中文/英文 commit message）。
2. 确保 pytest 与零重写审计全绿。
3. 发起 Pull Request，描述「复用了什么 / 验证了哪条 INV / 为何优于从零实现」。

## 🏷️ 署名

本仓库版权归 **晨星 (Morningstar) / CJX0712** 所有（Apache-2.0）。合并的贡献者
将在 PR 与 CHANGELOG 中致谢。
