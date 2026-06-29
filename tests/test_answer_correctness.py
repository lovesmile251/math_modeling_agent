from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tools.answer_correctness import (
    audit_workspace_correctness,
    load_gold_expectations,
    summarize_correctness,
)


def test_load_gold_expectations_accepts_cases_wrapper(tmp_path: Path):
    path = tmp_path / "gold.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "case-a",
                        "expected_numeric_ranges": [{"metric": "profit", "min": 1, "max": 2}],
                    },
                    {"case_id": "case-b"},
                ]
            }
        ),
        encoding="utf-8",
    )

    expectations = load_gold_expectations(path)

    assert sorted(expectations) == ["case-a"]


def test_audit_workspace_correctness_checks_numeric_and_decision_expectations(tmp_path: Path):
    workspace = tmp_path / "case-a"
    logs = workspace / "logs"
    tables = workspace / "tables"
    paper = workspace / "paper"
    for path in (logs, tables, paper):
        path.mkdir(parents=True)

    table_path = tables / "result.csv"
    pd.DataFrame(
        {
            "metric": ["profit", "service_rate"],
            "value": [42.0, 0.94],
            "decision": ["open hub A", "acceptable"],
        }
    ).to_csv(table_path, index=False)
    (logs / "result_registry.json").write_text(
        json.dumps({"entries": [{"type": "table", "path": str(table_path)}]}),
        encoding="utf-8",
    )
    (paper / "paper_draft.md").write_text("The final plan is to open hub A.", encoding="utf-8")

    audit = audit_workspace_correctness(
        workspace,
        case_id="case-a",
        expectation={
            "expected_numeric_ranges": [
                {"metric": "profit", "min": 40.0, "max": 45.0},
                {"column": "value", "min": 0.9, "max": 1.0},
            ],
            "expected_decisions": [{"acceptable_values": ["open hub A"]}],
        },
    )

    assert audit["applicable"] is True
    assert audit["passed"] is True
    assert audit["pass_rate"] == 1.0
    summary = summarize_correctness([{**audit, "case_id": "case-a"}])
    assert summary["applicable_case_count"] == 1
    assert summary["pass_rate"] == 1.0


def test_audit_workspace_correctness_flags_failures(tmp_path: Path):
    workspace = tmp_path / "case-b"
    logs = workspace / "logs"
    tables = workspace / "tables"
    paper = workspace / "paper"
    for path in (logs, tables, paper):
        path.mkdir(parents=True)

    table_path = tables / "result.csv"
    pd.DataFrame({"metric": ["profit"], "value": [12.0]}).to_csv(table_path, index=False)
    (logs / "result_registry.json").write_text(
        json.dumps({"entries": [{"type": "table", "path": str(table_path)}]}),
        encoding="utf-8",
    )
    (paper / "paper_draft.md").write_text("The final plan keeps the current hub.", encoding="utf-8")

    audit = audit_workspace_correctness(
        workspace,
        case_id="case-b",
        expectation={
            "expected_numeric_ranges": [{"metric": "profit", "min": 40.0, "max": 45.0}],
            "expected_decisions": [{"acceptable_values": ["open hub A"]}],
        },
    )

    assert audit["passed"] is False
    assert len(audit["failures"]) == 2
