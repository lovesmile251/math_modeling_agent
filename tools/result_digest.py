from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import WorkspaceConfig
from models.catalog import EXECUTABLE_MODEL_LABELS

_EXTRA_LABELS = {
    "capacity_gap": "需求容量缺口分析",
    "describe": "描述统计",
    "community_detection": "社群发现（各社群规模/内部密度/核心成员）",
    "top5_communities": "内部连接密度最大的 5 个社群",
    "community_relation": "5 大社群间关系强度与重叠分析",
    "friend_recommendation": "好友推荐候选（链路预测得分）",
    "recommendation_reason": "Top-3 好友推荐及未成好友原因",
    "network_properties": "好友网络整体结构指标",
    "key_user_candidates": "关键用户候选（中心性 + 传播影响力）",
    "key_user_summary": "关键用户与 48 小时传播范围",
    "propagation_curve": "关键用户 48 小时传播曲线",
    "push_schedule": "推送名额优化方案（贪心影响力最大化）",
    "push_strategy_comparison": "推送策略传播范围对比",
}


def _label(name: str) -> str:
    return EXECUTABLE_MODEL_LABELS.get(name, _EXTRA_LABELS.get(name, name))


def _read_csv(path: Path):
    import pandas as pd

    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError, OSError):
            return None
    return None


def _frame_to_text(df, max_rows: int, max_cols: int) -> str:
    import pandas as pd

    notes: list[str] = []
    if df.shape[1] > max_cols:
        df = df.iloc[:, :max_cols]
        notes.append(f"列已截断至前 {max_cols} 列")
    total_rows = df.shape[0]
    if total_rows > max_rows:
        df = df.head(max_rows)
        notes.append(f"共 {total_rows} 行，仅展示前 {max_rows} 行")

    df = df.copy()
    for column in df.columns:
        if pd.api.types.is_float_dtype(df[column]):
            df[column] = df[column].map(lambda value: f"{value:.4g}" if pd.notna(value) else "")

    headers = [str(column) for column in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in df.iterrows():
        cells = [str(row[column]).replace("\n", " ").replace("|", "/") for column in df.columns]
        lines.append("| " + " | ".join(cells) + " |")

    text = "\n".join(lines)
    if notes:
        text += "\n（" + "；".join(notes) + "）"
    return text


def _compute_stats(df: pd.DataFrame) -> dict:
    """Extract statistical summaries from a DataFrame.

    Returns a dict with:
      - ``numeric``: per-column {col: {min, max, mean, std, top3, bottom3}}.
      - ``shape``: {rows, columns}.
      - ``notable``: from ``_find_notable_patterns``.
    """
    stats: dict[str, Any] = {"numeric": {}, "shape": {"rows": len(df), "columns": len(df.columns)}}

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        col_stats: dict[str, Any] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)) if len(series) > 1 else 0.0,
        }
        # Top-3 and bottom-3 with row context
        sorted_asc = series.sort_values()
        bottom3 = sorted_asc.head(3)
        top3 = sorted_asc.tail(3)[::-1]

        def _ctx(vals: pd.Series) -> list[dict]:
            result: list[dict] = []
            for idx, val in vals.items():
                entry: dict = {"value": float(val), "row": int(idx) if isinstance(idx, (int, np.integer)) else str(idx)}
                # Include a neighbour column value for context when available
                other_cols = [c for c in df.columns if c != col]
                if other_cols:
                    neighbour = other_cols[0]
                    entry["context"] = {neighbour: str(df.at[idx, neighbour])}
                result.append(entry)
            return result

        col_stats["bottom3"] = _ctx(bottom3)
        col_stats["top3"] = _ctx(top3)
        stats["numeric"][col] = col_stats

    stats["notable"] = _find_notable_patterns(df)
    return stats


def _find_notable_patterns(df: pd.DataFrame) -> dict:
    """Flag columns with high variance, extreme ratios, and outlier rows."""
    notable: dict[str, Any] = {"high_variance_cols": [], "extreme_ratio_cols": [], "outlier_rows": {}}

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return notable

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty or len(series) < 3:
            continue
        mean = series.mean()
        std = series.std(ddof=0)
        col_min = series.min()
        col_max = series.max()

        # High std/mean ratio (> 1.0)
        if mean != 0 and abs(std / mean) > 1.0:
            notable["high_variance_cols"].append(col)

        # Max/min ratio > 10
        if col_min != 0 and abs(col_max / col_min) > 10:
            notable["extreme_ratio_cols"].append(col)

        # Outlier rows (> 3 std from mean)
        if std > 0:
            outlier_mask = (series - mean).abs() > 3 * std
            for idx in series[outlier_mask].index:
                notable["outlier_rows"].setdefault(str(idx), []).append(col)

    return notable


def build_result_digest(workspace: WorkspaceConfig, max_rows: int = 12, max_cols: int = 10) -> str:
    """Read run_summary.json and the produced CSV tables, return a markdown
    digest containing the *actual numbers* so the writer cites real results
    instead of fabricating placeholders."""
    summary_path = workspace.logs_dir / "run_summary.json"
    if not summary_path.exists():
        return "（未找到运行结果摘要 run_summary.json，没有任何模型结果，禁止编造模型结论与数值。）"

    try:
        payload: Any = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return "（运行结果摘要解析失败，禁止编造模型结论与数值。）"

    if isinstance(payload, dict):
        return "（本次运行未提供数据，没有任何模型结果，禁止编造模型结论、数值、排名或图表。）"

    sections: list[str] = []
    any_output = False
    figure_names: list[str] = []

    # ------------------------------------------------------------------
    # Statistics collected for the "Results at a Glance" section
    # ------------------------------------------------------------------
    total_models = 0
    successful_models = 0
    empty_models = 0
    model_stats: list[dict[str, Any]] = []  # {label, key_col, stats, notable}

    for item in payload:
        source = Path(str(item.get("source", ""))).name or "未知数据源"
        block = [
            f"### 数据源：{source}",
            f"- 样本量：{item.get('rows', '未知')}，字段数：{item.get('columns', '未知')}",
            f"- 字段：{', '.join(str(c) for c in item.get('column_names', [])) or '无'}",
            f"- 数值字段：{', '.join(str(c) for c in item.get('numeric_columns', [])) or '无'}",
        ]

        missing = {k: v for k, v in (item.get("missing_values") or {}).items() if v}
        block.append(f"- 缺失值字段：{missing if missing else '未发现明显缺失'}")

        charts = item.get("charts") or []
        if charts:
            # Store absolute paths for inline embedding: ![caption](abs_path)
            abs_paths = [str(Path(str(c)).resolve()) for c in charts]
            figure_names.extend(abs_paths)
            block.append(f"- 已生成图表：{', '.join(Path(p).name for p in abs_paths)}")

        describe_path = item.get("describe_table")
        if describe_path and Path(describe_path).exists():
            frame = _read_csv(Path(describe_path))
            if frame is not None and not frame.empty:
                block.append("\n**描述统计：**\n" + _frame_to_text(frame, max_rows, max_cols))

        outputs = item.get("model_outputs") or {}
        if outputs:
            for name, path in outputs.items():
                total_models += 1
                file_path = Path(str(path))
                if not file_path.exists():
                    empty_models += 1
                    continue
                frame = _read_csv(file_path)
                if frame is None or frame.empty:
                    empty_models += 1
                    continue
                any_output = True
                successful_models += 1
                block.append(f"\n**{_label(name)}（文件：{file_path.name}）：**\n" + _frame_to_text(frame, max_rows, max_cols))

                # Collect stats for the glance section
                try:
                    stats = _compute_stats(frame)
                    numeric = stats.get("numeric", {})
                    key_col = list(numeric.keys())[0] if numeric else None
                    model_stats.append({
                        "label": _label(name),
                        "file": file_path.name,
                        "key_col": key_col,
                        "stats": numeric,
                        "notable": stats.get("notable", {}),
                        "shape": stats.get("shape", {}),
                    })
                except Exception:
                    # Stats collection is best-effort; never break the digest
                    pass
        else:
            block.append("\n> 注意：该数据源没有任何模型成功产出结果表。")

        sections.append("\n".join(block))

    # ------------------------------------------------------------------
    # Build "Results at a Glance" section
    # ------------------------------------------------------------------
    glance_lines: list[str] = []
    if model_stats:
        glance_lines.append("## Results at a Glance")
        glance_lines.append("")
        glance_lines.append(f"- **Total models run**: {total_models}, **successful**: {successful_models}, **empty**: {empty_models}")
        glance_lines.append("")

        for ms in model_stats:
            label = ms["label"]
            key_col = ms["key_col"]
            numeric = ms["stats"]
            notable = ms["notable"]
            glance_lines.append(f"### {label}")
            glance_lines.append("")

            if key_col and key_col in numeric:
                col_info = numeric[key_col]
                glance_lines.append(
                    f"- **Key column `{key_col}`**: "
                    f"min={col_info['min']:.4g}, max={col_info['max']:.4g}, "
                    f"mean={col_info['mean']:.4g}, std={col_info['std']:.4g}"
                )

            # Per-column top-3 / bottom-3
            for col, col_info in numeric.items():
                top3 = col_info.get("top3", [])
                bottom3 = col_info.get("bottom3", [])
                if top3:
                    vals = ", ".join(f"{e['value']:.4g}" for e in top3)
                    glance_lines.append(f"- `{col}` top-3: {vals}")
                if bottom3:
                    vals = ", ".join(f"{e['value']:.4g}" for e in bottom3)
                    glance_lines.append(f"- `{col}` bottom-3: {vals}")

            # Notable patterns
            flags: list[str] = []
            if notable.get("high_variance_cols"):
                flags.append(f"high-variance columns: {', '.join(notable['high_variance_cols'])}")
            if notable.get("extreme_ratio_cols"):
                flags.append(f"extreme-ratio columns (max/min > 10): {', '.join(notable['extreme_ratio_cols'])}")
            outlier_rows = notable.get("outlier_rows") or {}
            if outlier_rows:
                row_list = ", ".join(
                    f"row {r} ({', '.join(cols)})" for r, cols in list(outlier_rows.items())[:5]
                )
                if len(outlier_rows) > 5:
                    row_list += f" … (+{len(outlier_rows) - 5} more)"
                flags.append(f"outlier rows: {row_list}")
            if flags:
                glance_lines.append("- ⚠️ " + "; ".join(flags))

            glance_lines.append("")

    # ------------------------------------------------------------------
    # Assemble final digest
    # ------------------------------------------------------------------
    digest_body = "\n\n".join(sections)
    if glance_lines:
        digest = "\n".join(glance_lines) + "\n\n---\n\n" + digest_body
    else:
        digest = digest_body

    if figure_names:
        digest += (
            "\n\n### 可在正文引用的图表（使用绝对路径内联嵌入）\n"
            + "\n".join(f"- ![]({p})" for p in dict.fromkeys(figure_names))
        )

    if not any_output:
        digest = (
            "【重要约束】本次运行没有任何模型成功产出结果数据表。"
            "论文只能基于下面的描述统计与题目本身展开，"
            "禁止编造任何模型计算结果、具体数值结论、排名或图表引用。\n\n" + digest
        )
    return digest
