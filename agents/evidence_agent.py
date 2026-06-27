from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

from agents.base import (
    Agent,
    ClaimEvidence,
    ClaimEvidenceMap,
    ResultRegistry,
    WorkflowState,
)
from tools.file_tool import write_text

log = logging.getLogger("mma.evidence_agent")

_NUMERIC_PATTERN = re.compile(r"[-+]?\d+\.?\d*")


class EvidenceAgent(Agent):
    """Scans execution outputs and builds a structured evidence map.

    Produces two artifacts stored on ``state``:

    - **ResultRegistry** — catalogue of every result table, figure, and log
    - **ClaimEvidenceMap** — binding between paper claims and their source data

    The evidence map enforces that every numerical claim in the paper must
    have a ``claim_id`` traceable to a specific file, row, and calculation.
    """

    name = "evidence_agent"

    def run(self, state: WorkflowState) -> WorkflowState:
        # 1. Build result registry from filesystem
        registry = self._build_registry(state)
        state.result_registry = registry
        registry_path = write_text(
            state.workspace.logs_dir / "result_registry.json",
            json.dumps(
                {
                    "schema_version": registry.schema_version,
                    "entries": registry.entries,
                    "source_path": registry.source_path,
                    "evidence_records": registry.evidence_records,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        state.artifacts["result_registry"] = registry_path

        # 2. Extract claims from result files
        claim_map = self._build_claim_map(state, registry)
        state.claim_evidence_map = claim_map
        claim_path = write_text(
            state.workspace.logs_dir / "claim_evidence_map.json",
            self._claim_map_to_json(claim_map),
        )
        state.artifacts["claim_evidence_map"] = claim_path

        log.info("EvidenceAgent: %d results, %d claims mapped (%.0f%% coverage)",
                 len(registry.entries), len(claim_map.claims), claim_map.coverage_pct)
        return state

    # ── result registry ──────────────────────────────────────────────────
    def _build_registry(self, state: WorkflowState) -> ResultRegistry:
        entries: list[dict] = []
        evidence_records: list[dict] = []

        # scan tables
        for table_path in sorted(state.workspace.tables_dir.glob("*.csv")):
            try:
                import pandas as pd
                df = pd.read_csv(table_path, nrows=100)
                profile = self._table_profile(df)
                table_evidence = self._table_evidence_records(table_path, df, len(evidence_records))
                evidence_records.extend(table_evidence)
                entries.append({
                    "type": "table",
                    "name": table_path.stem,
                    "path": str(table_path),
                    "rows": len(df),
                    "columns": list(df.columns),
                    "numeric_columns": list(df.select_dtypes(include="number").columns),
                    "profile": profile,
                    "evidence_ids": [item["evidence_id"] for item in table_evidence],
                    "sha256": self._hash_file(table_path),
                })
            except Exception:
                entries.append({
                    "type": "table",
                    "name": table_path.stem,
                    "path": str(table_path),
                    "bytes": table_path.stat().st_size,
                })

        # scan figures
        for fig_path in sorted(state.workspace.figures_dir.glob("*.png")):
            entries.append({
                "type": "figure",
                "name": fig_path.stem,
                "path": str(fig_path),
                "bytes": fig_path.stat().st_size,
            })

        # scan run_summary
        summary_path = state.workspace.logs_dir / "run_summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                entries.append({
                    "type": "run_summary",
                    "name": "run_summary",
                    "path": str(summary_path),
                    "model_count": len(summary) if isinstance(summary, list) else 1,
                })
            except (json.JSONDecodeError, OSError):
                pass

        # scan execution manifest
        manifest_path = state.workspace.logs_dir / "execution_manifest.json"
        if manifest_path.exists():
            entries.append({
                "type": "execution_manifest",
                "name": "execution_manifest",
                "path": str(manifest_path),
                "bytes": manifest_path.stat().st_size,
            })

        return ResultRegistry(
            entries=entries,
            source_path=str(state.workspace.root),
            evidence_records=evidence_records,
        )

    # ── claim evidence map ───────────────────────────────────────────────
    def _build_claim_map(self, state: WorkflowState, registry: ResultRegistry) -> ClaimEvidenceMap:
        claims: list[ClaimEvidence] = []
        counter = 0

        evidence_records = registry.evidence_records or self._records_from_legacy_entries(registry)
        if not evidence_records:
            return ClaimEvidenceMap(claims=[], coverage_pct=0.0, unmapped_claims=[])

        for record in evidence_records:
            if record.get("kind") != "column_stat":
                continue
            if record.get("statistic") != "mean_std":
                continue
            counter += 1
            claims.append(ClaimEvidence(
                claim_id=f"C-{counter:03d}",
                claim=record["claim"],
                model_id=self._guess_model_id(str(record.get("table", ""))),
                source_file=str(record.get("source_file", "")),
                source_rows=list(record.get("source_rows", [])),
                calculation=str(record.get("calculation", "")),
                paper_sections=["结果分析"],
            ))

        for record in evidence_records:
            if record.get("kind") != "rank":
                continue
            counter += 1
            claims.append(ClaimEvidence(
                claim_id=f"C-{counter:03d}",
                claim=record["claim"],
                model_id=self._guess_model_id(str(record.get("table", ""))),
                source_file=str(record.get("source_file", "")),
                source_rows=list(record.get("source_rows", [])),
                calculation=str(record.get("calculation", "")),
                paper_sections=["模型对比", "结果分析"],
            ))

        # compute coverage
        total_claims = len(claims)
        mapped_claims = sum(1 for c in claims if c.source_file and c.claim)
        coverage = (mapped_claims / total_claims * 100) if total_claims > 0 else 0.0

        return ClaimEvidenceMap(
            claims=claims,
            coverage_pct=round(coverage, 1),
            unmapped_claims=[],
        )

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _hash_file(path: Path) -> str:
        try:
            digest = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except Exception:
            return ""

    def _table_profile(self, df) -> dict:
        numeric = df.select_dtypes(include="number")
        return {
            "row_count_sampled": int(len(df)),
            "column_count": int(len(df.columns)),
            "numeric_column_count": int(len(numeric.columns)),
            "missing_values": {str(column): int(df[column].isna().sum()) for column in df.columns},
        }

    def _table_evidence_records(self, path: Path, df, offset: int) -> list[dict]:
        import pandas as pd

        records: list[dict] = []
        table_name = path.stem
        numeric_cols = [
            column
            for column in df.columns
            if pd.to_numeric(df[column], errors="coerce").notna().any()
        ]
        for col in numeric_cols[:8]:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                continue
            mean_val = float(series.mean())
            std_val = float(series.std())
            best_row = int(series.idxmax()) if pd.notna(series.idxmax()) else 0
            worst_row = int(series.idxmin()) if pd.notna(series.idxmin()) else 0
            records.append(
                {
                    "evidence_id": f"E-{offset + len(records) + 1:04d}",
                    "kind": "column_stat",
                    "table": table_name,
                    "column": str(col),
                    "statistic": "mean_std",
                    "value": {"mean": round(mean_val, 6), "std": round(std_val, 6)},
                    "source_file": str(path),
                    "source_rows": [best_row, worst_row],
                    "calculation": f"mean({col}) = {mean_val:.4f}, std({col}) = {std_val:.4f}",
                    "claim": f"{table_name} 中 {col} 的均值为 {mean_val:.4f}，标准差为 {std_val:.4f}",
                }
            )

        if numeric_cols:
            primary_col = numeric_cols[0]
            sortable = pd.to_numeric(df[primary_col], errors="coerce")
            valid_rows = df.loc[sortable.notna()].copy()
            if not valid_rows.empty:
                valid_rows["_evidence_sort_value"] = sortable.loc[valid_rows.index]
                df_sorted = valid_rows.sort_values("_evidence_sort_value", ascending=False).head(3)
                top_rows = [int(item) for item in list(df_sorted.index[:3])]
                top_values = [
                    f"{df_sorted.iloc[i]['_evidence_sort_value']:.4g}"
                    for i in range(min(3, len(df_sorted)))
                ]
                records.append(
                    {
                        "evidence_id": f"E-{offset + len(records) + 1:04d}",
                        "kind": "rank",
                        "table": table_name,
                        "column": str(primary_col),
                        "statistic": "top3_desc",
                        "value": top_values,
                        "source_file": str(path),
                        "source_rows": top_rows,
                        "calculation": f"sort({primary_col}, descending).head(3)",
                        "claim": f"{table_name} 按 {primary_col} 排序，前3名取值：{', '.join(top_values)}",
                    }
                )
        return records

    def _records_from_legacy_entries(self, registry: ResultRegistry) -> list[dict]:
        records: list[dict] = []
        for entry in registry.entries:
            if entry.get("type") != "table":
                continue
            path = Path(str(entry.get("path", "")))
            if not path.exists():
                continue
            try:
                import pandas as pd
                df = pd.read_csv(path)
            except Exception:
                continue
            records.extend(self._table_evidence_records(path, df, len(records)))
        return records

    @staticmethod
    def _guess_model_id(table_name: str) -> str:
        """Guess which model produced this table from its filename."""
        table_lower = table_name.lower()
        model_keywords = {
            "ridge": "ridge_regression",
            "lasso": "lasso_regression",
            "linear": "linear_regression",
            "logistic": "logistic_classifier",
            "random_forest": "random_forest",
            "svm": "svm_classifier",
            "kmeans": "kmeans_clustering",
            "pca": "pca_reduction",
            "entropy": "entropy_weights",
            "topsis": "topsis_rank",
            "capacity": "capacity_gap",
            "trend": "trend_forecast",
            "describe": "describe_stats",
            "comparison": "model_comparison",
            "error": "error_analysis",
            "sensitivity": "sensitivity_analysis",
            "community": "community_detection",
            "friend": "friend_recommendation",
            "propagation": "information_propagation",
            "influence": "influence_maximization",
            "esp": "cement_esp_optimization",
            "cement": "cement_esp_optimization",
        }
        for keyword, model_id in model_keywords.items():
            if keyword in table_lower:
                return model_id
        return "unknown"

    @staticmethod
    def _claim_map_to_json(cm: ClaimEvidenceMap) -> str:
        import dataclasses
        claims_data = [dataclasses.asdict(c) for c in cm.claims]
        payload = {
            "claims": claims_data,
            "coverage_pct": cm.coverage_pct,
            "unmapped_claims": cm.unmapped_claims,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
