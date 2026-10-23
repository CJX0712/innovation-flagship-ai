"""L8 回归门禁双轨合一回归 (v0.8.0)。

背景
----
v0.7.13 全栈审计记录: 回归门禁**双轨分裂** ——

- ``scripts/gen_baseline.py`` 的机器比较读 **BASELINE.json**;
- ``ifai/eval/harness.py`` 的 ``RegressionBaseline.parse_baseline`` 却去
  **解析 BASELINE.md 的人类可读行** (``tok/s mean: …``)。

同一份基线两套读法: 基线文件被重新生成时, MD 行格式一旦变动, harness
就**静默返回 NO_BASELINE** —— 门禁形同虚设且不报错。这正是 ifai 自己
审计里反复出现的「同型缺口」。

v0.8.0 合���: **BASELINE.json 为唯一权威源**, MD 降级为纯人类可读产物。
"""
from __future__ import annotations

import json
import os

import pytest

from ifai.eval.harness import RegressionBaseline


@pytest.fixture()
def baseline_dir(tmp_path):
    """构造一个含 BASELINE.json + BASELINE.md 的目录。"""
    metrics = {
        "prompt_len": 32,
        "decode_steps": 32,
        "repeats": 9,
        "warmup": 2,
        "prefill_ms": 7.0,
        "tpot_ms": 9.0,
        "throughput_tok_s": 111.11,
        "prefill_cv_pct": 3.0,
        "tpot_cv_pct": 4.0,
        "statistic": "median",
    }
    (tmp_path / "BASELINE.json").write_text(
        json.dumps(metrics, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "BASELINE.md").write_text(
        "# 性能基线\n\ntok/s mean: 999.99\n", encoding="utf-8")
    return tmp_path


def test_json_is_authoritative_source(baseline_dir):
    """JSON 优先: 即使 MD 里的数字不同, 也必须以 JSON 为准。"""
    md = str(baseline_dir / "BASELINE.md")
    base = RegressionBaseline.parse_baseline(md)
    assert base["tok_per_s_mean"] == pytest.approx(111.11)
    assert base["source"].endswith("BASELINE.json")


def test_parse_baseline_accepts_json_path_directly(baseline_dir):
    """直接给 .json 路径也能读。"""
    js = str(baseline_dir / "BASELINE.json")
    base = RegressionBaseline.parse_baseline(js)
    assert base["tok_per_s_mean"] == pytest.approx(111.11)
    assert base["tpot_ms"] == pytest.approx(9.0)
    assert base["prefill_ms"] == pytest.approx(7.0)


def test_md_only_fallback_still_works(tmp_path):
    """兼容旧基线: 只有 MD 时仍能读 (但来源标记为 MD)。"""
    md = tmp_path / "BASELINE.md"
    md.write_text("# 性能基线\n\ntok/s mean: 88.8\n", encoding="utf-8")
    base = RegressionBaseline.parse_baseline(str(md))
    assert base["tok_per_s_mean"] == pytest.approx(88.8)
    assert base["source"].endswith("BASELINE.md")


def test_missing_baseline_returns_empty(tmp_path):
    """基线不存在时返回空 dict (不抛异常)。"""
    base = RegressionBaseline.parse_baseline(str(tmp_path / "nope.md"))
    assert base == {}


def test_check_reports_green_when_stable(baseline_dir):
    """无退化 → GREEN。"""
    md = str(baseline_dir / "BASELINE.md")
    verdict = RegressionBaseline.check({"tok_per_s_mean": 111.0}, md)
    assert verdict["status"] == "GREEN"
    assert verdict["drop"] < 0.01


def test_check_reports_red_on_regression(baseline_dir):
    """退化超阈值 → RED。"""
    md = str(baseline_dir / "BASELINE.md")
    verdict = RegressionBaseline.check({"tok_per_s_mean": 50.0}, md,
                                       threshold=0.15)
    assert verdict["status"] == "RED"
    assert verdict["drop"] > 0.5


def test_check_reports_no_baseline(tmp_path):
    """无基线 → NO_BASELINE (显式, 不静默当绿灯)。"""
    verdict = RegressionBaseline.check({"tok_per_s_mean": 100.0},
                                       str(tmp_path / "missing.md"))
    assert verdict["status"] == "NO_BASELINE"
    assert verdict["source"] is None


def test_gen_baseline_and_harness_share_one_authority(baseline_dir):
    """双轨合一的核心断言: 脚本与 harness 对同一基线的读数完全一致。

    若二者再次分裂 (脚本读 JSON / harness 读 MD), 本用例必然失败。
    """
    md = str(baseline_dir / "BASELINE.md")
    js = str(baseline_dir / "BASELINE.json")
    from_json = RegressionBaseline.parse_baseline(js)
    from_md = RegressionBaseline.parse_baseline(md)
    assert from_json["tok_per_s_mean"] == from_md["tok_per_s_mean"]
    assert from_json["tpot_ms"] == from_md["tpot_ms"]


def test_repo_baseline_json_exists():
    """仓库内必须有权威基线 JSON (否则门禁在 CI 上无源可比)。"""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    js = os.path.join(repo, "BASELINE.json")
    assert os.path.isfile(js), "缺少权威基线 BASELINE.json"
    with open(js, encoding="utf-8") as f:
        data = json.load(f)
    assert "throughput_tok_s" in data
    # 基线必须自带噪声边界, 否则 CPU 抖动会让门禁误报
    assert "tpot_cv_pct" in data
    assert data.get("statistic") == "median"
