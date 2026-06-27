from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from agents.base import (
    A_MODEL_EXECUTION_FEEDBACK,
    K_RESULT_ANALYSIS,
    Agent,
    WorkflowState,
)
from models.catalog import EXECUTABLE_MODEL_LABELS
from tools.file_tool import write_text

log = logging.getLogger("mma.analysis_agent")


class AnalysisAgent(Agent):
    name = "analysis_agent"

    def run(self, state: WorkflowState) -> WorkflowState:
        summary_path = state.workspace.logs_dir / "run_summary.json"
        if not summary_path.exists():
            state.notes["result_analysis"] = "# 结果分析\n\n- 未找到运行结果摘要，需要先修复代码执行问题。"
            return state

        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        # run_summary.json can be a list of per-file summaries or a single dict
        # (e.g. {"message": "No data files provided."}) — treat both as empty-results.
        if isinstance(payload, dict):
            state.notes[K_RESULT_ANALYSIS] = (
                "# 结果分析\n\n- 当前未提供数据，因此只完成了工作流骨架运行。"
            )
            return state
        if not isinstance(payload, list) or len(payload) == 0:
            state.notes[K_RESULT_ANALYSIS] = (
                "# 结果分析\n\n- 运行结果摘要为空，可能代码未正确生成输出。"
            )
            return state

        lines = ["# 结果分析", ""]
        feedback_path = self._write_execution_feedback(state, payload)
        state.artifacts[A_MODEL_EXECUTION_FEEDBACK] = feedback_path

        for item in payload:
            lines.extend(
                [
                    f"## {item['source']}",
                    f"- 样本量：{item['rows']}",
                    f"- 字段数：{item['columns']}",
                    f"- 数值字段：{', '.join(item['numeric_columns']) or '无'}",
                    f"- 已选择模型：{', '.join(item.get('selected_models', [])) or '无'}",
                    f"- 描述统计表：{item['describe_table']}",
                ]
            )
            if item.get("charts"):
                lines.append(f"- 生成图表：{', '.join(item['charts'])}")

            model_outputs = self._model_outputs(item)
            if model_outputs:
                lines.append("- 模型输出：")
                for name, path in model_outputs.items():
                    lines.append(f"  - {self._label_model_output(name)}：{path}")

            lines.extend(self._diagnostics_lines(model_outputs))

            missing = {key: value for key, value in item["missing_values"].items() if value}
            lines.append(f"- 缺失值字段：{missing if missing else '未发现明显缺失'}")
            lines.append("")
        state.notes[K_RESULT_ANALYSIS] = "\n".join(lines)
        return state

    def _write_execution_feedback(self, state: WorkflowState, payload: list) -> Path:
        sources: list[dict] = []
        aggregate: dict[str, dict] = {
            "produced": {"ids": set(), "details": {}},
            "empty": {"ids": set(), "details": {}},
            "missing": {"ids": set(), "details": {}},
        }

        for item in payload:
            if not isinstance(item, dict):
                continue

            selected_models = [str(model_id) for model_id in item.get("selected_models") or []]
            model_outputs = self._model_outputs(item)

            source_feedback = {
                "source": item.get("source", ""),
                "selected_models": selected_models,
                "produced_models": [],
                "empty_models": [],
                "missing_models": [],
            }

            for model_id in selected_models:
                output_path = model_outputs.get(model_id)
                if not output_path:
                    detail = {"model_id": model_id, "reason": "not_found_in_model_outputs"}
                    source_feedback["missing_models"].append(detail)
                    self._add_feedback_detail(aggregate["missing"], model_id, detail)
                    continue

                table_status = self._table_status(output_path)
                detail = {"model_id": model_id, "path": str(output_path), **table_status}
                if table_status["status"] == "produced":
                    source_feedback["produced_models"].append(detail)
                    self._add_feedback_detail(aggregate["produced"], model_id, detail)
                elif table_status["status"] == "empty":
                    source_feedback["empty_models"].append(detail)
                    self._add_feedback_detail(aggregate["empty"], model_id, detail)
                else:
                    source_feedback["missing_models"].append(detail)
                    self._add_feedback_detail(aggregate["missing"], model_id, detail)

            sources.append(source_feedback)

        payload_out = {
            "summary": {
                "produced_models": self._aggregate_models(aggregate["produced"]),
                "empty_models": self._aggregate_models(aggregate["empty"]),
                "missing_models": self._aggregate_models(aggregate["missing"]),
            },
            "sources": sources,
        }
        return write_text(
            state.workspace.logs_dir / "model_execution_feedback.json",
            json.dumps(payload_out, ensure_ascii=False, indent=2),
        )

    def _model_outputs(self, item: dict) -> dict[str, str]:
        """Return model_id -> table path for legacy and current run summaries."""
        model_outputs = item.get("model_outputs") or {}
        if isinstance(model_outputs, dict) and model_outputs:
            return {str(key): str(value) for key, value in model_outputs.items() if value}

        outputs: dict[str, str] = {}
        model_runs = item.get("model_runs") or []
        if not isinstance(model_runs, list):
            return outputs
        for run in model_runs:
            if not isinstance(run, dict):
                continue
            if run.get("status") != "success" or not run.get("table"):
                continue
            outputs[str(run.get("model_id", ""))] = str(run["table"])
        return {key: value for key, value in outputs.items() if key}

    def _add_feedback_detail(self, bucket: dict, model_id: str, detail: dict) -> None:
        bucket["ids"].add(model_id)
        bucket["details"].setdefault(model_id, detail)

    def _aggregate_models(self, bucket: dict) -> list[dict]:
        return [bucket["details"][model_id] for model_id in sorted(bucket["ids"])]

    def _table_status(self, output_path) -> dict:
        path = Path(output_path)
        if not path.exists():
            return {"status": "missing", "reason": "output_file_missing", "rows": 0}
        rows = self._count_csv_rows(path)
        if rows > 0:
            return {"status": "produced", "rows": rows}
        return {"status": "empty", "reason": "empty_table", "rows": 0}

    def _count_csv_rows(self, path: Path) -> int:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    return 0
                return sum(1 for row in reader if any(str(value).strip() for value in row.values()))
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            log.debug("Could not count rows in %s: %s", path.name, exc)
            return 0

    def _diagnostics_lines(self, model_outputs: dict) -> list[str]:
        lines: list[str] = []

        esp_rows = self._read_rows(model_outputs.get("esp_optimization"))
        if esp_rows:
            lines.extend(self._esp_optimization_lines(esp_rows))

        error_rows = self._read_rows(model_outputs.get("error_analysis"))
        if error_rows:
            metrics = self._error_metrics(error_rows)
            lines.append(
                "- 误差分析："
                + f"目标 {metrics.get('target', '')}，"
                + f"RMSE={self._fmt(metrics.get('rmse'))}，"
                + f"MAE={self._fmt(metrics.get('mae'))}，"
                + f"MAPE={self._fmt(metrics.get('mape_percent'))}%，"
                + f"R²={self._fmt(metrics.get('r_squared'))}"
                + f"（调整 R²={self._fmt(metrics.get('adj_r_squared'))}）。"
            )

        sensitivity_rows = self._read_rows(model_outputs.get("sensitivity_analysis"))
        if sensitivity_rows:
            top = sensitivity_rows[0]
            oat_key = next((k for k in top if str(k).startswith("oat_response_percent")), None)
            oat = f"，OAT 响应={self._fmt(top.get(oat_key))}%" if oat_key else ""
            lines.append(
                "- 灵敏度分析："
                + f"对目标 {top.get('target', '')} 影响最大的因素为 {top.get('feature', '')}"
                + f"（弹性={self._fmt(top.get('elasticity'))}，"
                + f"标准化灵敏度={self._fmt(top.get('standardized_sensitivity'))}{oat}）。"
            )

        comparison_rows = self._read_rows(model_outputs.get("model_comparison"))
        if comparison_rows:
            best = next(
                (row for row in comparison_rows if str(row.get("is_best_by_rmse", "")).lower() == "true"),
                None,
            )
            if best is not None:
                lines.append(
                    "- 模型对比："
                    + f"交叉验证下最优模型为 {best.get('model', '')}"
                    + f"（CV-RMSE={self._fmt(best.get('cv_rmse'))}，"
                    + f"CV-R²={self._fmt(best.get('cv_r_squared'))}）。"
                )
        return lines

    def _esp_optimization_lines(self, rows: list[dict]) -> list[str]:
        lines: list[str] = []
        summary = next((row for row in rows if row.get("section") == "standard_tightening_summary"), None)
        if summary:
            lines.append(
                "- ESP optimization: "
                + f"tightening {self._fmt(summary.get('baseline_standard_mgNm3'))}mg to {self._fmt(summary.get('standard_mgNm3'))}mg "
                + f"raises average power by {self._fmt(summary.get('energy_increment_pct'))}% "
                + f"({self._fmt(summary.get('baseline_P_total_kW'))} -> {self._fmt(summary.get('predicted_P_total_kW'))} kW)."
            )

        optima = [row for row in rows if row.get("section") == "typical_condition_optimum"]
        for row in optima[:6]:
            lines.append(
                "- ESP optimum "
                + f"{row.get('condition', '')} @ {self._fmt(row.get('standard_mgNm3'))}mg: "
                + f"U=({self._fmt(row.get('U1_kV'))}, {self._fmt(row.get('U2_kV'))}, "
                + f"{self._fmt(row.get('U3_kV'))}, {self._fmt(row.get('U4_kV'))}) kV, "
                + f"T=({self._fmt(row.get('T1_s'))}, {self._fmt(row.get('T2_s'))}, "
                + f"{self._fmt(row.get('T3_s'))}, {self._fmt(row.get('T4_s'))}) s, "
                + f"P={self._fmt(row.get('predicted_P_total_kW'))} kW, "
                + f"Cout={self._fmt(row.get('predicted_C_out_mgNm3'))} mg/Nm3."
            )

        strategies = [row for row in rows if row.get("section") == "differential_strategy"]
        if strategies:
            labels = [
                f"{row.get('condition', '')}: {row.get('priority_rule', '')}"
                for row in strategies[:2]
            ]
            lines.append("- ESP differential strategies: " + " | ".join(labels))
        return lines

    def _error_metrics(self, rows: list[dict]) -> dict:
        """Support both the wide single-row schema and a long metric/value schema."""
        first = rows[0]
        if "rmse" in first or "r_squared" in first:
            return first
        long_map = {
            "RMSE 均方根误差": "rmse",
            "MAE 平均绝对误差": "mae",
            "MAPE 平均绝对百分比误差(%)": "mape_percent",
            "R² 决定系数": "r_squared",
            "调整 R²": "adj_r_squared",
        }
        metrics: dict = {"target": first.get("target", "")}
        for row in rows:
            key = long_map.get(str(row.get("metric", "")))
            if key:
                metrics[key] = row.get("value")
        return metrics

    def _read_rows(self, path) -> list[dict]:
        if not path:
            return []
        try:
            file_path = Path(path)
            if not file_path.exists():
                return []
            with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
                return list(csv.DictReader(handle))
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            log.debug("Could not read rows from %s: %s", Path(path).name, exc)
            return []

    def _fmt(self, value) -> str:
        if value is None or value == "":
            return "—"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if number != number:  # NaN
            return "—"
        return f"{number:.4g}"

    def _label_model_output(self, name: str) -> str:
        extra_labels = {
            "esp_optimization": "ESP operating optimization",
            "capacity_gap": "需求容量缺口分析",
            "community_detection": "社群发现",
            "top5_communities": "5 大高密度社群",
            "community_relation": "社群间关系与重叠",
            "friend_recommendation": "好友推荐候选",
            "recommendation_reason": "Top-3 推荐及原因",
            "network_properties": "网络结构指标",
            "key_user_candidates": "关键用户候选",
            "key_user_summary": "关键用户与传播范围",
            "propagation_curve": "48 小时传播曲线",
            "push_schedule": "推送优化方案",
            "push_strategy_comparison": "推送策略对比",
        }
        return EXECUTABLE_MODEL_LABELS.get(name, extra_labels.get(name, name))
