from __future__ import annotations

import json

import pandas as pd

from tools.paper_templates.general import GeneralPaperTemplate


def test_general_template_handles_model_runs_and_nan_describe(temp_workspace):
    describe_path = temp_workspace.tables_dir / "sample_describe.csv"
    model_path = temp_workspace.tables_dir / "sample_inventory_policy.csv"
    figure_path = temp_workspace.figures_dir / "sample_hist_value.png"
    pd.DataFrame({"metric": ["count", "mean"], "value": [3.0, float("nan")]}).to_csv(
        describe_path,
        index=False,
    )
    pd.DataFrame({"item_key": ["A"], "suggested_replenishment": [12.5]}).to_csv(
        model_path,
        index=False,
    )
    figure_path.write_text("not a real png; path presence is enough for markdown", encoding="utf-8")
    (temp_workspace.logs_dir / "run_summary.json").write_text(
        json.dumps(
            [
                {
                    "source": "sample.csv",
                    "rows": 3,
                    "columns": 2,
                    "column_names": ["item", "value"],
                    "numeric_columns": ["value"],
                    "missing_values": {},
                    "describe_table": str(describe_path),
                    "charts": [str(figure_path)],
                    "model_runs": [
                        {
                            "model_id": "inventory_policy",
                            "status": "success",
                            "table": str(model_path),
                            "elapsed_seconds": 0.01,
                            "error": None,
                        }
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    paper = GeneralPaperTemplate(temp_workspace, "补货与定价问题").build()

    assert "核心数学表达" in paper
    assert "inventory_policy" in paper
    assert "suggested_replenishment" in paper
    assert paper.count("\\[") >= 8
