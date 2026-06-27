from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from agents.review_agent import ReviewAgent
from tools.answer_reproduction_audit import audit_workspace, summarize_audits


def test_answer_reproduction_audit_verifies_claims_hashes_and_model_count(tmp_path: Path):
    workspace = tmp_path / "case-a"
    logs = workspace / "logs"
    tables = workspace / "tables"
    figures = workspace / "figures"
    paper = workspace / "paper"
    for path in (logs, tables, figures, paper):
        path.mkdir(parents=True)

    table_path = tables / "result.csv"
    pd.DataFrame({"score": [1.0, 2.0, 3.0]}).to_csv(table_path, index=False)
    digest = hashlib.sha256(table_path.read_bytes()).hexdigest()
    (logs / "result_registry.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "type": "table",
                        "name": "result",
                        "path": str(table_path),
                        "sha256": digest,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (logs / "claim_evidence_map.json").write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claim_id": "C-001",
                        "source_file": str(table_path),
                        "calculation": "mean(score) = 2.0000, std(score) = 1.0000",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (logs / "traceability_report.json").write_text(
        json.dumps({"passed": True, "coverage_pct": 90.0}, ensure_ascii=False),
        encoding="utf-8",
    )
    (logs / "run_summary.json").write_text(
        json.dumps(
            [
                {
                    "model_runs": [
                        {"model_id": "demo", "status": "success", "table": str(table_path)}
                    ]
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (paper / "review_report.md").write_text("- 成功产出结果的模型数：1", encoding="utf-8")
    (paper / "paper_draft.md").write_text(_paper_text(), encoding="utf-8")

    audit = audit_workspace(workspace, case_id="case-a")

    assert audit["numeric_claim_audit"]["verified_rate"] == 1.0
    assert audit["registry_hash_audit"]["hash_pass_rate"] == 1.0
    assert audit["model_count_audit"]["consistent"] is True
    assert audit["audit_score"] > 80

    summary = summarize_audits([audit])
    assert summary["case_count"] == 1
    assert summary["verified_claim_rate"] == 1.0


def test_answer_reproduction_audit_flags_review_model_count_mismatch(tmp_path: Path):
    workspace = tmp_path / "case-b"
    logs = workspace / "logs"
    paper = workspace / "paper"
    logs.mkdir(parents=True)
    paper.mkdir(parents=True)
    (logs / "run_summary.json").write_text(
        json.dumps([{"model_runs": [{"status": "success", "table": "x.csv"}]}]),
        encoding="utf-8",
    )
    (paper / "review_report.md").write_text("- 成功产出结果的模型数：0", encoding="utf-8")

    audit = audit_workspace(workspace, case_id="case-b")

    assert audit["model_count_audit"]["consistent"] is False
    assert any("不一致" in risk for risk in audit["risks"])


def test_review_agent_counts_model_runs_format(tmp_path: Path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "run_summary.json").write_text(
        json.dumps(
            [
                {
                    "model_runs": [
                        {"status": "success", "table": "a.csv"},
                        {"status": "skipped", "table": None},
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )
    state = SimpleNamespace(workspace=SimpleNamespace(logs_dir=logs))

    assert ReviewAgent()._count_model_outputs(state) == 1


def _paper_text() -> str:
    equations = "\n".join(r"$$x_{%d}=1$$" % index for index in range(25))
    references = "\n".join(f"[{index}] A. Title. Journal, 2024." for index in range(1, 11))
    return (
        "# 测试论文\n\n"
        "## 摘要\n"
        + "本文给出模型、结果和结论。" * 200
        + "\n\n## 结果分析\n"
        + equations
        + "\n\n"
        + "\n".join(f"![图{index}](fig{index}.png)" for index in range(8))
        + "\n\n"
        + "\n".join("| a | b |\n|---|---|\n| 1 | 2 |" for _ in range(12))
        + "\n\n## 参考文献\n"
        + references
    )
