from __future__ import annotations

from pathlib import Path


def test_national_award_optimization_doc_is_linked_and_covers_gates():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    doc_path = root / "docs" / "national_award_agent_optimization.md"
    doc = doc_path.read_text(encoding="utf-8")

    assert "docs/national_award_agent_optimization.md" in readme
    assert "# 国奖论文智能体优化说明" in doc
    assert "## 关键门禁" in doc
    assert "## 后续优化路线图" in doc
    for gate in (
        "export_quality_gate",
        "task_traceability_gate",
        "strong_baseline_gate",
        "innovation_evidence_gate",
    ):
        assert gate in doc
