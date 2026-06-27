from __future__ import annotations

import pandas as pd

from agents.base import ResultRegistry
from agents.evidence_agent import EvidenceAgent


def test_evidence_agent_skips_non_numeric_values_in_numeric_registry_column(tmp_path):
    table_path = tmp_path / "mixed_numeric.csv"
    pd.DataFrame({"score": ["1.0", "2.5", "bad", "4.0"]}).to_csv(table_path, index=False)
    registry = ResultRegistry(
        entries=[
            {
                "type": "table",
                "name": "mixed_numeric",
                "path": str(table_path),
                "numeric_columns": ["score"],
            }
        ]
    )

    claim_map = EvidenceAgent()._build_claim_map(state=None, registry=registry)  # type: ignore[arg-type]

    assert claim_map.coverage_pct == 100.0
    assert any("mean(score)" in claim.calculation for claim in claim_map.claims)
