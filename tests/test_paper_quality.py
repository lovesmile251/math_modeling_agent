from __future__ import annotations

from tools.paper_quality import clean_paper_text, check_keywords, check_traceability, evaluate_paper_quality


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


def test_submission_blockers_are_hard_quality_issues():
    text = _paper_fixture(
        keyword_line="关键词：优化模型；灵敏度分析；结果验证",
        table_block="| 指标 | 数值 |\n| --- | --- |\n| 最优值 | 12.5 |",
        extra_result="当前未产出优化数值解，后续计算。"
    )

    report = evaluate_paper_quality(text)

    assert any("Submission blocker" in issue for issue in report.issues)
    assert report.score < 82


def test_reference_entries_not_cited_are_quality_issues():
    text = _paper_fixture(
        keyword_line="关键词：优化模型；灵敏度分析；结果验证",
        table_block="| 指标 | 数值 |\n| --- | --- |\n| 最优值 | 12.5 |",
        references="[1] Zhang. Model validation. Journal, 2024.\n[2] Li. Data consistency. Journal, 2023.",
    )

    report = evaluate_paper_quality(text)

    assert any("Reference entries not cited in body" in issue and "[2]" in issue for issue in report.issues)


def test_keyword_count_is_enforced_as_issue():
    issues, suggestions, count = check_keywords("## 关键词\n优化；验证")

    assert count == 2
    assert any("关键词数量" in issue for issue in issues)
    assert suggestions


def test_core_result_table_missing_blocks_high_score():
    text = _paper_fixture(
        keyword_line="关键词：优化模型；灵敏度分析；结果验证",
        table_block="",
    )

    report = evaluate_paper_quality(text)

    assert any("Core result table missing" in issue for issue in report.issues)
    assert report.score < 82


def _paper_fixture(
    *,
    keyword_line: str,
    table_block: str,
    references: str = "[1] Zhang. Model validation. Journal, 2024.",
    extra_result: str = "",
) -> str:
    equations = "\n".join(r"\(x_{%d}=1\)" % i for i in range(8))
    table_section = f"\n{table_block}\n" if table_block else ""
    return f"""# 测试论文

## 摘要
本文围绕问题一、问题二和问题三建立优化模型，得到目标值 12.5、误差 0.31、权重 0.42、稳定性 0.91、覆盖率 0.83、排名 1、2、3、4。正文引用文献[1]并结合结果表验证模型有效性。

## 关键词
{keyword_line}

## 问题重述
问题一要求建立评价模型。问题二要求完成优化求解。问题三要求进行检验对比。

## 问题分析
问题一、问题二和问题三分别对应评价、优化和检验闭环。

## 模型假设
假设输入数据经过清洗，符号定义如下。

## 符号说明
| 符号 | 含义 |
| --- | --- |
| x | 决策变量 |

## 模型建立
{equations}

## 结果分析
{table_section}
![图1](fig1.png)
![图2](fig2.png)
{extra_result}

## 模型检验与误差分析
本文进行误差、灵敏度、检验、对比和基准分析，验证结果可信。

## 模型评价与推广
模型评价说明适用范围和推广条件。

## 参考文献
{references}

## 附录
代码和数据处理过程见附录。
"""
