from __future__ import annotations

import pandas as pd

from agents.base import ClaimEvidence, ClaimEvidenceMap
from tools.traceability import evaluate_numeric_traceability


def test_numeric_traceability_maps_values_to_result_table(temp_workspace):
    table = temp_workspace.tables_dir / "forecast.csv"
    pd.DataFrame({"rmse": [2.5], "forecast": [120.0]}).to_csv(table, index=False)
    paper = "模型预测值达到 120.0，RMSE 误差为 2.5。"

    report = evaluate_numeric_traceability(
        paper, temp_workspace, ClaimEvidenceMap(), threshold_pct=100
    )

    assert report.passed
    assert report.coverage_pct == 100
    assert report.claims[0].matched_sources == ["forecast.csv"]


def test_numeric_traceability_rejects_unmapped_claim(temp_workspace):
    paper = "结果表明准确率达到 99.9%。"

    report = evaluate_numeric_traceability(
        paper, temp_workspace, ClaimEvidenceMap(), threshold_pct=70
    )

    assert not report.passed
    assert report.coverage_pct == 0
    assert report.issues


def test_numeric_traceability_accepts_claim_id_value(temp_workspace):
    claim = ClaimEvidence(
        claim_id="C-001",
        claim="平均得分为 88.5",
        calculation="mean(score)=88.5",
        source_file="scores.csv",
    )
    paper = "结果显示平均得分为 88.5 [C-001]。"

    report = evaluate_numeric_traceability(
        paper,
        temp_workspace,
        ClaimEvidenceMap(claims=[claim]),
        threshold_pct=100,
    )

    assert report.passed


def test_numeric_traceability_maps_derived_column_statistics(temp_workspace):
    table = temp_workspace.tables_dir / "model.csv"
    pd.DataFrame({"rmse": [1.0, 2.0, 3.0]}).to_csv(table, index=False)
    paper = "- rmse：最小值 1，最大值 3，均值 2"

    report = evaluate_numeric_traceability(
        paper,
        temp_workspace,
        ClaimEvidenceMap(),
        threshold_pct=100,
    )

    assert report.passed
    assert report.coverage_pct == 100
    assert report.claims[0].matched_sources == ["model.csv"]


def test_numeric_traceability_ignores_markdown_stat_table_header(temp_workspace):
    paper = "| Unnamed: 0 | count | unique | top | freq | mean | std | min | 25% | 50% |"

    report = evaluate_numeric_traceability(
        paper,
        temp_workspace,
        ClaimEvidenceMap(),
        threshold_pct=100,
    )

    assert report.total_numeric_claims == 0
    assert report.passed


def test_numeric_traceability_handles_boolean_table_columns(temp_workspace):
    table = temp_workspace.tables_dir / "comparison.csv"
    pd.DataFrame({"is_best_by_rmse": [True, False, True]}).to_csv(table, index=False)
    paper = "- is_best_by_rmse：最小值 0，最大值 1，均值 0.6667"

    report = evaluate_numeric_traceability(
        paper,
        temp_workspace,
        ClaimEvidenceMap(),
        threshold_pct=100,
    )

    assert report.passed
    assert report.coverage_pct == 100
