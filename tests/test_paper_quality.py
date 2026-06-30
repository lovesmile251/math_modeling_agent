from __future__ import annotations

import json

import pandas as pd

from tools.paper_evidence_audit import audit_paper_evidence_density
from tools.paper_quality import (
    check_keywords,
    check_traceability,
    clean_paper_text,
    evaluate_paper_quality,
    submission_blocking_issues,
)


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


def test_clean_paper_text_removes_review_report_residue_and_placeholders():
    raw = """# Paper

## Results
The result table is complete. 具体数值待补充。

## 发现
- 论文未达到国奖质量门禁。

## 修改建议
- 删除不可提交表述。
"""

    cleaned = clean_paper_text(raw)

    assert "待补充" not in cleaned
    assert "## 发现" not in cleaned
    assert "## 修改建议" not in cleaned
    assert "论文未达到国奖质量门禁" not in cleaned
    assert "由本次运行产物支撑" in cleaned


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


def test_structured_paper_audit_flags_symbol_formula_and_citation_issues():
    text = """# Paper

## Abstract
This paper reports problem one, problem two, and problem three with values 12.5, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, and 0.9.

## Keywords
optimization; sensitivity; validation

## Symbol table
| symbol | meaning |
| --- | --- |
| x | decision variable |

## Model
\\[y=a+b+c\\]
\\[z=a+c+d\\]
\\[q=b+c+d\\]

## Results
| metric | value |
| --- | --- |
| objective | 12.5 |
Ref. 2 reports a comparison.

## References
[1] Zhang. Model validation. Journal, 2024.
"""

    report = evaluate_paper_quality(text)

    assert report.metrics["display_equations"] >= 3
    assert any("Formula numbering inconsistent" in issue for issue in report.issues)
    assert any("Symbol definition mismatch" in issue for issue in report.issues)
    assert any("Citation format issue" in issue for issue in report.issues)


def test_evaluate_paper_quality_blocks_weak_national_award_structure():
    text = """# Paper

## Abstract
Problem 1 and Problem 2 are solved with objective 12.5, score 88.0, error 0.31, stability 0.91, and coverage 0.83.

## Keywords
optimization; validation; sensitivity

## Model
The model is described in prose.

## Results
| metric | value |
| --- | --- |
| objective | 12.5 |
Problem 1 obtains score 88.0.

## References
[1] Zhang. Model validation. Journal, 2024.
"""

    report = evaluate_paper_quality(text)

    assert any("Award structure weak" in issue for issue in report.issues)
    assert any("Problem-answer closure weak" in issue for issue in report.issues)
    assert any("Model formulation weak" in issue for issue in report.issues)
    assert any("Award structure weak" in issue for issue in submission_blocking_issues(report))
    assert report.score < 82


def test_paper_evidence_audit_flags_cvar_claim_without_tail_metrics(tmp_path):
    paper = """# Paper

## Abstract
The CVaR tail risk model improves the solution and gives reliable decisions.

## Results
| metric | value |
| --- | --- |
| objective | 12.5 |
"""

    audit = audit_paper_evidence_density(paper, workspace_root=tmp_path)

    assert any("Risk model evidence weak: cvar_optimization" in issue for issue in audit.issues)
    assert audit.metrics["claimed_risk_models"] == 1


def test_paper_evidence_audit_accepts_cvar_with_table_and_metrics(tmp_path):
    tables = tmp_path / "tables"
    logs = tmp_path / "logs"
    tables.mkdir()
    logs.mkdir()
    pd.DataFrame(
        {
            "var_loss": [25.0],
            "cvar_loss": [45.0],
            "tail_scenario_count": [1],
            "risk_adjusted_score": [30.0],
        }
    ).to_csv(tables / "result_cvar_optimization.csv", index=False)
    (logs / "model_selection_report.json").write_text(
        json.dumps({"selected_model_ids": ["cvar_optimization"]}),
        encoding="utf-8",
    )
    paper = """# Paper

## Abstract
The CVaR model obtains var_loss 25.0, cvar_loss 45.0, tail_scenario_count 1, risk_adjusted_score 30.0, and objective 12.5.

## Results
| metric | value |
| --- | --- |
| var_loss | 25.0 |
| cvar_loss | 45.0 |
| tail_scenario_count | 1 |
| risk_adjusted_score | 30.0 |
"""

    audit = audit_paper_evidence_density(paper, workspace_root=tmp_path)

    assert not any("Risk model evidence weak" in issue for issue in audit.issues)
    assert not any("Claimed high-level model has no matching" in issue for issue in audit.issues)
    assert audit.metrics["cvar_optimization_metric_hits"] >= 2
    assert audit.metrics["cvar_optimization_table_metric_hits"] >= 2


def test_paper_evidence_audit_flags_high_level_table_without_model_metrics(tmp_path):
    tables = tmp_path / "tables"
    logs = tmp_path / "logs"
    tables.mkdir()
    logs.mkdir()
    pd.DataFrame({"objective": [12.5], "score": [88.0]}).to_csv(
        tables / "result_cvar_optimization.csv",
        index=False,
    )
    (logs / "model_selection_report.json").write_text(
        json.dumps({"selected_model_ids": ["cvar_optimization"]}),
        encoding="utf-8",
    )
    paper = """# Paper

## Abstract
The CVaR model obtains var_loss 25.0, cvar_loss 45.0, tail_scenario_count 1, risk_adjusted_score 30.0, and objective 12.5.

## Results
| metric | value |
| --- | --- |
| var_loss | 25.0 |
| cvar_loss | 45.0 |
| risk_adjusted_score | 30.0 |
"""

    audit = audit_paper_evidence_density(paper, workspace_root=tmp_path)

    assert any(
        "High-level model table lacks model-specific metrics: cvar_optimization" in issue
        for issue in audit.issues
    )
    assert audit.metrics["cvar_optimization_table_metric_hits"] == 0


def test_paper_evidence_audit_flags_selected_high_level_model_missing_from_paper(tmp_path):
    tables = tmp_path / "tables"
    logs = tmp_path / "logs"
    tables.mkdir()
    logs.mkdir()
    pd.DataFrame({"cvar_loss": [45.0], "risk_adjusted_score": [30.0]}).to_csv(
        tables / "result_cvar_optimization.csv",
        index=False,
    )
    (logs / "model_selection_report.json").write_text(
        json.dumps({"selected_model_ids": ["cvar_optimization"]}),
        encoding="utf-8",
    )
    paper = """# Paper

## Abstract
The optimization model obtains objective 12.5, stability 0.91, error 0.31, coverage 0.83, and score 88.0.

## Results
| metric | value |
| --- | --- |
| objective | 12.5 |
| stability | 0.91 |
"""

    audit = audit_paper_evidence_density(paper, workspace_root=tmp_path)

    assert any(
        "Selected high-level model missing from paper narrative: cvar_optimization" in issue
        for issue in audit.issues
    )
    assert audit.metrics["selected_risk_models_missing_narrative"] == 1


def test_evaluate_paper_quality_includes_p2_evidence_density_gate():
    text = _paper_fixture(
        keyword_line="关键词：CVaR；tail risk；robust optimization",
        table_block="| metric | value |\n| --- | --- |\n| objective | 12.5 |",
        extra_result="本文采用 CVaR tail risk model 控制极端损失。",
    )

    report = evaluate_paper_quality(text)

    assert any("Risk model evidence weak: cvar_optimization" in issue for issue in report.issues)
    assert report.metrics["claimed_risk_models"] >= 1


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
