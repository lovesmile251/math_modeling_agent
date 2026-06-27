from __future__ import annotations

from tools.paper_quality import clean_paper_text, check_traceability, evaluate_paper_quality


def test_clean_paper_text_removes_llm_chatter_and_normalizes_title():
    raw = """好的，请稍等。我将撰写一篇论文。

---

### **校园社交平台建模**

#### **摘 要**

本文建立模型，得到 1、2、3 个结果。
"""

    cleaned = clean_paper_text(raw)

    assert "好的，请稍等" not in cleaned
    assert "我将撰写" not in cleaned
    assert cleaned.startswith("# 校园社交平台建模")
    assert "#### **摘 要**" not in cleaned
    assert "摘要" in cleaned


def test_evaluate_paper_quality_flags_thin_draft():
    thin = "# 论文\n\n## 摘要\n本文建立模型。\n\n## 问题重述\n略。"

    report = evaluate_paper_quality(thin)

    assert report.score < 82
    assert any("缺少" in issue for issue in report.issues)
    assert any("摘要" in issue for issue in report.issues)


def test_check_traceability_flags_missing_reference_entry():
    text = """# Paper

Body cites method [2].

## 参考文献
[1] Author. Title. Journal, 2024.
"""

    issues, suggestions = check_traceability(text)

    assert any("Reference citations without matching entries" in item for item in issues)
    assert suggestions


def test_check_traceability_flags_claimed_model_without_table(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "tables").mkdir(parents=True)
    (workspace / "tables" / "sample_describe.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    issues, _suggestions = check_traceability(
        "The paper reports a topsis_rank model result.",
        workspace_root=workspace,
    )

    assert any("Model claims without generated result tables" in item for item in issues)


def test_check_traceability_flags_reference_number_mismatch():
    text = """# 论文

正文引用了已有文献[1]，也错误引用了不存在的文献[3]。

## 参考文献
[1] Zhang. Model validation. Journal, 2024.
[2] Li. Data consistency. Journal, 2023.
"""

    issues, suggestions = check_traceability(text)

    assert any("[3]" in issue for issue in issues)
    assert any("[2]" in suggestion for suggestion in suggestions)


def test_check_traceability_allows_matching_table_and_flags_missing_model(tmp_path):
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    (tables_dir / "baseline_trend_forecast.csv").write_text("x,y\n1,2\n", encoding="utf-8")

    text = "本文使用 trend forecast 和 grey gm11 进行预测。"

    issues, suggestions = check_traceability(text, workspace_root=tmp_path)

    assert any("grey_gm11" in issue for issue in issues)
    assert not any("trend_forecast" in issue for issue in issues)
    assert suggestions == ["Remove unsupported model claims or run the corresponding models first."]


def test_check_traceability_ignores_model_phrase_inside_references(tmp_path):
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    (tables_dir / "sample_describe.csv").write_text("x,y\n1,2\n", encoding="utf-8")

    text = """# Paper

The body does not claim an optimization model result.

## 参考文献
[1] Saaty T L. The analytic hierarchy process: planning, priority setting, resource allocation. McGraw-Hill, 1980.
"""

    issues, _suggestions = check_traceability(text, workspace_root=tmp_path)

    assert not any("resource_allocation" in issue for issue in issues)
