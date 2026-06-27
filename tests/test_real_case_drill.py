from __future__ import annotations

import json
from pathlib import Path

from tools.real_case_drill import _load_model_feedback, run_real_case_drill


def test_real_case_drill_runs_full_workflow(tmp_path: Path, sample_dataframe):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    statement = corpus_root / "case.md"
    statement.write_text(
        "A题 根据附件数据预测需求趋势，并对容量缺口进行综合评价。",
        encoding="utf-8",
    )
    data_path = corpus_root / "data.csv"
    sample_dataframe.to_csv(data_path, index=False)

    index_path = tmp_path / "corpus_index.json"
    index_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "toy-2025-a",
                    "year": 2025,
                    "problem": "A",
                    "title": "需求预测与容量评价",
                    "statement_path": "case.md",
                    "attachment_paths": ["data.csv"],
                    "statement_chars": statement.stat().st_size,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "results"
    runs_root = tmp_path / "runs"
    summary = run_real_case_drill(
        corpus_index_path=index_path,
        corpus_root=corpus_root,
        output_dir=output_dir,
        runs_root=runs_root,
        case_ids=["toy-2025-a"],
        limit=None,
        use_llm=False,
    )

    assert summary["case_count"] == 1
    assert summary["execution_success_rate"] == 1.0
    result = summary["results"][0]
    assert result["case_id"] == "toy-2025-a"
    assert result["execution_status"] == "success"
    assert result["score"] > 0
    assert result["table_count"] > 0
    assert result["figure_count"] > 0
    assert Path(result["artifacts"]["code"]).exists()
    assert Path(result["artifacts"]["paper"]).exists()
    assert (output_dir / "real_case_drill.json").exists()
    assert (output_dir / "real_case_drill.md").exists()


def test_real_case_drill_rejects_unknown_case(tmp_path: Path):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    index_path = tmp_path / "corpus_index.json"
    index_path.write_text("[]", encoding="utf-8")

    try:
        run_real_case_drill(
            corpus_index_path=index_path,
            corpus_root=corpus_root,
            output_dir=tmp_path / "results",
            runs_root=tmp_path / "runs",
            case_ids=["missing-case"],
            limit=None,
        )
    except ValueError as exc:
        assert "missing-case" in str(exc)
    else:
        raise AssertionError("Expected an unknown case id to fail.")


def test_real_case_drill_records_failed_case_and_continues(tmp_path: Path, sample_dataframe):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    good_statement = corpus_root / "good.md"
    good_statement.write_text("根据附件数据预测需求趋势。", encoding="utf-8")
    (corpus_root / "good.csv").write_text(sample_dataframe.to_csv(index=False), encoding="utf-8")

    index_path = tmp_path / "corpus_index.json"
    index_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "bad-case",
                    "year": 2025,
                    "problem": "A",
                    "title": "缺失题面",
                    "statement_path": "missing.md",
                    "attachment_paths": [],
                    "statement_chars": 0,
                },
                {
                    "case_id": "good-case",
                    "year": 2025,
                    "problem": "B",
                    "title": "正常题面",
                    "statement_path": "good.md",
                    "attachment_paths": ["good.csv"],
                    "statement_chars": good_statement.stat().st_size,
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = run_real_case_drill(
        corpus_index_path=index_path,
        corpus_root=corpus_root,
        output_dir=tmp_path / "results",
        runs_root=tmp_path / "runs",
        limit=None,
    )

    assert summary["case_count"] == 2
    assert summary["results"][0]["execution_status"] == "failed"
    assert summary["results"][1]["execution_status"] == "success"
    assert (tmp_path / "results" / "real_case_drill.json").exists()


def test_real_case_drill_feedback_counts_case_level_coverage(tmp_path: Path):
    feedback_path = tmp_path / "model_execution_feedback.json"
    feedback_path.write_text(
        json.dumps(
            {
                "summary": {
                    "produced_models": [{"model_id": "inventory_policy"}],
                    "empty_models": [],
                    "missing_models": [
                        {"model_id": "inventory_policy"},
                        {"model_id": "forecast_model"},
                        {"model_id": "error_analysis"},
                    ],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    feedback = _load_model_feedback(feedback_path)

    assert feedback["produced_models"] == ["inventory_policy"]
    assert feedback["missing_models"] == ["forecast_model"]


def test_real_case_drill_feedback_ignores_graph_case_scheduling_mismatch(tmp_path: Path):
    feedback_path = tmp_path / "model_execution_feedback.json"
    feedback_path.write_text(
        json.dumps(
            {
                "summary": {
                    "produced_models": [{"model_id": "graph_shortest_paths"}],
                    "empty_models": [{"model_id": "scheduling_plan"}],
                    "missing_models": [{"model_id": "scheduling_plan"}],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    feedback = _load_model_feedback(feedback_path)

    assert feedback["produced_models"] == ["graph_shortest_paths"]
    assert feedback["empty_models"] == []
    assert feedback["missing_models"] == []
