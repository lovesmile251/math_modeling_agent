from __future__ import annotations

import json

import pandas as pd

from agents.base import WorkflowState
from agents.model_selection_agent import ModelSelectionAgent
from agents.model_selection_crew import TaskDecompositionAgent
from models.catalog import ALGORITHM_CATALOG, EXECUTABLE_MODEL_LABELS, catalog_size, check_applicable, executable_model_ids


class MockLLM:
    enabled = True

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def complete(self, instructions, user_input):
        self.calls.append((instructions, user_input))
        if self.error:
            raise self.error
        return self.response


class StructuredMockLLM:
    enabled = True

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def complete_json(self, instructions, user_input, schema=None):
        self.calls.append((instructions, user_input, schema))
        return self.payload


def _run_selection(problem_text, workspace, data_files, llm=None):
    state = WorkflowState(problem_text=problem_text, data_files=data_files, workspace=workspace, llm=llm)
    return ModelSelectionAgent().run(state)


def test_catalog_executable_ids_are_registered():
    """Every executable id referenced in the catalog must have a label."""
    labels = executable_model_ids()
    for entry in ALGORITHM_CATALOG:
        for model_id in entry.executable_model_ids:
            assert model_id in labels, f"{model_id} 缺少标签"


def test_catalog_size_matches():
    assert catalog_size() == len(ALGORITHM_CATALOG)
    assert catalog_size() > 0


def test_task_decomposition_only_splits_explicit_question_markers():
    text = """
    题目背景中包含 1.4 mm、10 ms 和 2.4 ms 等数值。
    问题1 建立运动模型并分析误差。
    (1) 给出参数值；
    (2) 讨论局部条件。
    问题2 给出最优控制策略。
    """

    segments = TaskDecompositionAgent()._segments(text)

    assert [item[0] for item in segments] == ["Q1", "Q2"]


def test_generic_relationship_is_not_misclassified_as_network():
    tasks = TaskDecompositionAgent().run(
        "分析温度、压力和产量之间的关系，并建立回归校准模型。",
        [],
    )

    assert "statistics" in {task.task_type for task in tasks}
    assert "network" not in {task.task_type for task in tasks}


def test_ascii_keyword_requires_word_boundaries():
    tasks = TaskDecompositionAgent().run(
        "Use computed tomography to calibrate the imaging system.",
        [],
    )

    assert "network" not in {task.task_type for task in tasks}


def test_classification_indicator_phrase_is_not_a_classification_task():
    tasks = TaskDecompositionAgent().run(
        "评酒员对葡萄酒的分类指标打分，并分析评分显著性。",
        [],
    )

    assert "statistics" in {task.task_type for task in tasks}
    assert "classification" not in {task.task_type for task in tasks}


def test_check_applicable_returns_structured_result():
    profile = {
        "columns": ("demand", "capacity", "cost"),
        "rows": 5,
        "numeric_columns": ("demand", "capacity", "cost"),
        "categorical_columns": (),
        "datetime_columns": (),
        "id_like_columns": (),
        "demand_columns": ("demand",),
        "capacity_columns": ("capacity",),
        "cost_columns": ("cost",),
        "benefit_columns": (),
        "relation_columns": (),
        "target_columns": (),
        "constraint_columns": (),
    }

    result = check_applicable("capacity_gap", profile, "evaluation")

    assert result.can_run is True
    assert result.required_fields == ()
    assert {"can_run", "reasons", "warnings", "required_fields"}.issubset(result.as_dict())


def test_check_applicable_reports_missing_required_fields():
    profile = {
        "columns": ("demand", "cost"),
        "rows": 5,
        "numeric_columns": ("demand", "cost"),
        "categorical_columns": (),
        "datetime_columns": (),
        "id_like_columns": (),
        "demand_columns": ("demand",),
        "capacity_columns": (),
        "cost_columns": ("cost",),
        "benefit_columns": (),
        "relation_columns": (),
        "target_columns": (),
        "constraint_columns": (),
    }

    result = check_applicable("capacity_gap", profile, "evaluation")

    assert result.can_run is False
    assert "capacity_columns" in result.required_fields
    assert result.warnings


def test_selection_writes_catalog_and_notes(temp_workspace, sample_csv):
    state = _run_selection("请根据需求与容量数据预测未来趋势并做综合评价。", temp_workspace, [sample_csv])

    assert "algorithm_catalog" in state.artifacts
    assert state.artifacts["algorithm_catalog"].exists()
    assert "model_selection_report" in state.artifacts
    assert state.artifacts["model_selection_report"].exists()
    assert "selected_model_ids" in state.notes

    selected = json.loads(state.notes["selected_model_ids"])
    assert isinstance(selected, list)
    # all selected ids must be runnable
    assert set(selected).issubset(executable_model_ids())


def test_selection_report_json_contains_five_agent_outputs(temp_workspace, sample_csv):
    state = _run_selection("forecast future demand and evaluate capacity gap", temp_workspace, [sample_csv])
    payload = json.loads(state.artifacts["model_selection_report"].read_text(encoding="utf-8"))

    assert payload["tasks"]
    assert payload["data_profile"]["numeric_columns"]
    assert payload["selected_models"]
    assert {"model_id", "total_score", "role", "reasons", "applicability"}.issubset(payload["selected_models"][0])
    assert "can_run" in payload["selected_models"][0]["applicability"]


def test_selection_uses_llm_task_decomposition_json_without_final_model_choice(temp_workspace, sample_csv):
    llm = MockLLM(
        response=json.dumps(
            {
                "subproblems": [
                    {
                        "id": "Q1",
                        "task_type": "forecast",
                        "objective": "forecast future demand",
                        "variables": ["year", "demand"],
                        "constraints": ["use available observations"],
                        "metrics": ["MAE"],
                        "possible_model_types": ["classification"],
                        "evidence": ["future demand"],
                        "source_text": "forecast future demand",
                    }
                ]
            }
        )
    )
    state = _run_selection("ambiguous wording with future demand", temp_workspace, [sample_csv], llm=llm)
    payload = json.loads(state.artifacts["model_selection_report"].read_text(encoding="utf-8"))
    selected = set(json.loads(state.notes["selected_model_ids"]))

    assert state.notes["model_selection_task_parser_mode"] == "llm"
    assert llm.calls
    assert payload["tasks"][0]["variables"] == ["year", "demand"]
    assert payload["tasks"][0]["metrics"] == ["MAE"]
    assert payload["tasks"][0]["possible_model_types"] == ["classification"]
    assert "trend_forecast" in selected
    assert not selected.intersection({"logistic_classifier", "naive_bayes_classifier", "knn_classifier"})


def test_selection_prefers_structured_llm_json_interface(temp_workspace, sample_csv):
    llm = StructuredMockLLM(
        {
            "subproblems": [
                {
                    "id": "Q1",
                    "task_type": "forecast",
                    "objective": "forecast future demand",
                    "variables": ["year", "demand"],
                    "metrics": ["MAE"],
                }
            ]
        }
    )

    state = _run_selection("forecast future demand", temp_workspace, [sample_csv], llm=llm)
    payload = json.loads(state.artifacts["model_selection_report"].read_text(encoding="utf-8"))

    assert state.notes["model_selection_task_parser_mode"] == "llm"
    assert llm.calls
    assert llm.calls[0][2] == {"type": "object"}
    assert payload["tasks"][0]["task_type"] == "forecast"


def test_selection_falls_back_to_rules_when_llm_task_decomposition_fails(temp_workspace, sample_csv):
    llm = MockLLM(error=RuntimeError("mock LLM failure"))
    state = _run_selection("forecast future demand and evaluate capacity gap", temp_workspace, [sample_csv], llm=llm)
    payload = json.loads(state.artifacts["model_selection_report"].read_text(encoding="utf-8"))
    selected = set(json.loads(state.notes["selected_model_ids"]))

    assert state.notes["model_selection_task_parser_mode"] == "fallback"
    assert "mock LLM failure" in state.notes["model_selection_task_parser_error"]
    assert payload["tasks"]
    assert "trend_forecast" in selected


def test_selection_covers_multiple_subquestions(temp_workspace, sample_csv):
    problem = "问题一：对各对象进行综合评价。问题二：预测未来需求趋势。问题三：在成本和容量约束下优化资源分配。"
    state = _run_selection(problem, temp_workspace, [sample_csv])
    selected = set(json.loads(state.notes["selected_model_ids"]))
    payload = json.loads(state.artifacts["model_selection_report"].read_text(encoding="utf-8"))
    task_ids = {task["task_id"] for task in payload["tasks"]}
    report_selected_ids = {item["model_id"] for item in payload["selected_models"]}

    assert {"Q1", "Q2", "Q3"}.issubset(task_ids)
    assert set(payload["selected_model_ids"]) == report_selected_ids
    assert selected == report_selected_ids
    assert {"Q1", "Q2", "Q3"}.issubset(payload["subproblem_models"])
    for task_id in ("Q1", "Q2", "Q3"):
        assert {"primary", "comparison", "validation"}.issubset(payload["subproblem_models"][task_id])
        assert payload["subproblem_models"][task_id]["primary"]
    assert selected.intersection({"entropy_weights", "topsis_rank"})
    assert "trend_forecast" in selected
    assert selected.intersection({"resource_allocation", "capacity_gap"})


def test_selection_uses_label_column_for_classification(temp_workspace):
    path = temp_workspace.data_dir / "classification.csv"
    pd.DataFrame(
        {
            "feature_a": [1.0, 1.2, 2.1, 2.4, 3.1, 3.4, 4.0, 4.3],
            "feature_b": [5.0, 4.8, 4.1, 3.9, 3.0, 2.8, 2.1, 1.9],
            "label": ["A", "A", "A", "B", "B", "B", "B", "A"],
        }
    ).to_csv(path, index=False)

    state = _run_selection("请根据特征进行分类识别，并给出分类模型。", temp_workspace, [path])
    selected = set(json.loads(state.notes["selected_model_ids"]))
    payload = json.loads(state.artifacts["model_selection_report"].read_text(encoding="utf-8"))

    assert selected.intersection({"logistic_classifier", "naive_bayes_classifier", "knn_classifier"})
    assert "label" in payload["data_profile"]["target_columns"]


def test_selection_profiles_nonstandard_semantic_columns(temp_workspace):
    path = temp_workspace.data_dir / "nonstandard_semantics.csv"
    pd.DataFrame(
        {
            "period_idx": [1, 2, 3, 4, 5, 6, 7, 8],
            "u": ["A", "A", "B", "C", "D", "B", "C", "D"],
            "v": ["B", "C", "C", "D", "A", "D", "A", "B"],
            "need_units": [10, 12, 14, 13, 15, 16, 18, 17],
            "limit_units": [16, 16, 18, 18, 20, 20, 21, 22],
            "cost_amt": [4.0, 4.5, 5.0, 4.8, 5.2, 5.4, 5.9, 6.1],
            "profit_amt": [9.0, 9.5, 10.2, 10.0, 10.8, 11.1, 11.5, 11.9],
            "labor_hours": [3, 4, 5, 4, 6, 6, 7, 7],
            "outcome_code": ["ok", "ok", "fail", "ok", "fail", "fail", "ok", "fail"],
        }
    ).to_csv(path, index=False)

    problem = "classify outcomes, analyze edge routes, forecast periods, and optimize resources under limits"
    state = _run_selection(problem, temp_workspace, [path])
    selected = set(json.loads(state.notes["selected_model_ids"]))
    payload = json.loads(state.artifacts["model_selection_report"].read_text(encoding="utf-8"))
    profile = payload["data_profile"]

    assert profile["unique_value_ratios"]["outcome_code"] == 0.25
    assert "period_idx" in profile["monotonic_time_columns"]
    assert "outcome_code" in profile["binary_label_columns"]
    assert profile["has_edge_table"] is True
    assert ["u", "v"] in profile["source_target_pairs"]
    assert ["need_units", "limit_units"] in profile["demand_capacity_pairs"]
    assert profile["has_objective_constraint_combo"] is True
    assert {"cost_amt", "profit_amt", "labor_hours"}.issubset(set(profile["objective_constraint_columns"]))
    assert selected.intersection({"logistic_classifier", "knn_classifier"})
    assert selected.intersection({"graph_shortest_paths", "graph_centrality"})
    assert "capacity_gap" in selected
    assert "resource_allocation" in selected


def test_selection_promotes_tail_risk_optimization_narrative(temp_workspace):
    path = temp_workspace.data_dir / "tail_risk_optimization.csv"
    pd.DataFrame(
        {
            "project": ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
            "scenario": ["normal", "normal", "normal", "stress", "stress", "stress", "tail", "tail", "tail"],
            "resource": [20.0, 25.0, 18.0, 20.0, 25.0, 18.0, 20.0, 25.0, 18.0],
            "profit": [70.0, 64.0, 45.0, 55.0, 42.0, 40.0, 36.0, 20.0, 32.0],
            "loss": [3.0, 5.0, 2.0, 10.0, 18.0, 9.0, 25.0, 45.0, 20.0],
            "budget": [45.0] * 9,
            "alpha": [0.9] * 9,
        }
    ).to_csv(path, index=False)

    problem = "optimize project resources under uncertainty, scenario stress, CVaR tail risk, and downside loss"
    state = _run_selection(problem, temp_workspace, [path])
    selected = set(json.loads(state.notes["selected_model_ids"]))
    payload = json.loads(state.artifacts["model_selection_report"].read_text(encoding="utf-8"))

    assert "cvar_optimization" in selected
    assert selected.intersection({"robust_optimization", "scenario_optimization", "chance_constrained_optimization"})
    narrative_ids = {item["model_id"] for item in payload["paper_model_narrative"]}
    assert "cvar_optimization" in narrative_ids
    cvar_plan = next(item for item in payload["model_comparison_plan"] if item["model_id"] == "cvar_optimization")
    assert {"var_loss", "cvar_loss", "risk_adjusted_score"}.issubset(set(cvar_plan["metrics"]))


def test_selection_picks_forecast_for_trend_problem(temp_workspace, sample_csv):
    state = _run_selection("对时间序列进行趋势预测，预测未来数值。", temp_workspace, [sample_csv])
    selected = json.loads(state.notes["selected_model_ids"])
    assert "trend_forecast" in selected


def test_selection_capacity_gap_on_demand_capacity(temp_workspace, sample_csv):
    state = _run_selection("分析需求与容量之间的缺口。", temp_workspace, [sample_csv])
    selected = json.loads(state.notes["selected_model_ids"])
    assert "capacity_gap" in selected


def test_selection_default_exploration_without_match(temp_workspace, sample_csv):
    # A problem with no catalog keyword still yields default exploration models
    state = _run_selection("xyzzy 无关题目内容。", temp_workspace, [sample_csv])
    selected = json.loads(state.notes["selected_model_ids"])
    assert selected, "应至少返回默认探索模型"


def test_selection_report_format(temp_workspace, sample_csv):
    state = _run_selection("综合评价与趋势预测。", temp_workspace, [sample_csv])
    report = state.notes["model_selection"]
    assert "# 模型选择" in report
    assert "可执行模型" in report
