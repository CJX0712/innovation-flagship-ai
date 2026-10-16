"""L9 CLI 端到端集成测试 (R-L9-01): train → save → demo 闭环。

用 subprocess 以仓库 venv 解释器绝对路径跑真实的 `python -m ifai.cli`,
覆盖「训练产出、vocab_size 软目标陷阱修复、demo 推理」三条关键路径。

关键回归: `--vocab 64` 配合默认 MICRO_CORPUS (实际中文字符数 > 64) 时,
修复前 tok.encode 会产生 ≥ 64 的 id, 而 ModelConfig(vocab_size=64) 的 embed
矩阵只有 64 行 → IndexError; 修复后模型词表取 max(args.vocab, tok.vocab_size),
训练必须不抛异常。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# 仓库 venv 解释器 (已装 pytest/pytest-cov)。若该绝对路径不存在 (如换机/重装),
# 回退到当前 pytest 运行所用的解释器, 保证测试仍可执行。
VENV_PY = r"C:\Users\Administrator\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
PY = VENV_PY if os.path.exists(VENV_PY) else sys.executable

REPO = Path(__file__).resolve().parents[3]


def _run_cli(args: list, cwd: str | None = None) -> subprocess.CompletedProcess:
    """用 venv 解释器运行 `python -m ifai.cli <args>`, 返回 CompletedProcess。"""
    return subprocess.run(
        [PY, "-m", "ifai.cli", *args],
        capture_output=True, text=True,
        cwd=str(cwd or REPO),
    )


TRAIN_ARGS = [
    "--vocab", "64", "--steps", "5", "--d-model", "32",
    "--layers", "1", "--heads", "2", "--kv-heads", "2",
    "--ffn-mult", "2", "--seq-len", "32",
]


def test_train_produces_artifacts(tmp_path):
    """train 子命令退出码 0, 且生成 manifest.json 与 params.npz。"""
    out = tmp_path / "model"
    r = _run_cli(["train", "--out", str(out), *TRAIN_ARGS])
    assert r.returncode == 0, f"train 应成功: {r.stderr}"
    assert (out / "manifest.json").exists(), "缺少 manifest.json"
    assert (out / "params.npz").exists(), "缺少 params.npz"
    assert (out / "tokenizer.json").exists(), "缺少 tokenizer.json"


def test_train_vocab_trap_no_index_error(tmp_path):
    """回归: 小 --vocab 配合大语料不得触发 IndexError (vocab_size 软目标陷阱)。"""
    out = tmp_path / "model"
    r = _run_cli(["train", "--out", str(out), *TRAIN_ARGS])
    assert r.returncode == 0, (
        f"小 vocab 训练必须不抛 IndexError (实际 stderr: {r.stderr})"
    )
    # 进一步确认: 退出码 0 且产物齐全, 即训练闭环完成。
    assert (out / "params.npz").exists()


def test_demo_after_train_runs(tmp_path):
    """train 之后 demo 推理退出码 0 且 stdout 非空。"""
    out = tmp_path / "model"
    # 先训练 (默认 MICRO_CORPUS, 不依赖外部文件)
    r_train = _run_cli(["train", "--out", str(out), *TRAIN_ARGS])
    assert r_train.returncode == 0, f"train 应成功: {r_train.stderr}"

    # 再 demo
    r_demo = _run_cli(
        ["demo", "--model", str(out), "--max-new", "8", "--greedy"],
    )
    assert r_demo.returncode == 0, f"demo 应成功: {r_demo.stderr}"
    assert r_demo.stdout.strip(), "demo 应输出非空结果"
