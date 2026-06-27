"""Abstract base class for problem-type-specific paper generators.

Each subclass produces a complete competition-grade paper in Markdown, injecting
real numbers from run_summary.json and result tables — never fabricating data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Shared helpers (usable by all subclasses)
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> pd.DataFrame | None:
    """Try reading a CSV with common encodings, returning None on failure."""
    if not path or not Path(path).exists():
        return None
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return None


def _md_table(df: pd.DataFrame, max_rows: int = 12, max_cols: int = 12) -> str:
    """Render a DataFrame as a Markdown table, truncating if needed."""
    if df is None or df.empty:
        return "_（无数据）_"
    if df.shape[1] > max_cols:
        df = df.iloc[:, :max_cols]
    truncated = df.shape[0] > max_rows
    if truncated:
        df = df.head(max_rows)

    def fmt(v):
        if pd.isna(v):
            return ""
        if isinstance(v, float):
            return str(int(v)) if v == int(v) else f"{v:.4g}"
        return str(v)

    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in df.columns) + " |")
    text = "\n".join(lines)
    if truncated:
        text += "\n\n（注：表格仅展示前若干行，完整结果见结果数据表附录。）"
    return text


def _val(df: pd.DataFrame | None, row: int, col: str, default: Any = "—") -> Any:
    """Safely extract a cell value from a DataFrame."""
    try:
        return df.iloc[row][col]
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class PaperTemplate:
    """Base for problem-type-specific paper generators.

    Subclasses override section methods to inject problem-specific narrative,
    formulas, and data extraction logic.  The ``build()`` method orchestrates
    everything by calling ``_section_order()`` and assembling the result.
    """

    problem_type: str = "general"

    def __init__(self, workspace: Any, problem_text: str, notes: dict[str, str] | None = None) -> None:
        self.workspace = workspace
        self.problem_text = problem_text
        self.notes = notes or {}
        self.logs_dir = Path(workspace.logs_dir)
        self.tables_dir = Path(workspace.tables_dir)
        self.figures_dir = Path(workspace.figures_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> str:
        """Build the complete paper Markdown.

        Calls every method listed in ``_section_order()``, joins non-empty
        results with double-newlines, and returns the final string.
        """
        return "\n\n".join(s for s in self._assemble_sections() if s)

    # ------------------------------------------------------------------
    # Section orchestration (override _section_order to reorder)
    # ------------------------------------------------------------------

    def _assemble_sections(self) -> list[str]:
        """Return the ordered list of section strings."""
        sections: list[str] = []
        for name in self._section_order():
            method = getattr(self, name, None)
            if method is None:
                continue
            result = method()
            if result:
                sections.append(result)
        return sections

    def _section_order(self) -> list[str]:
        """Override to add, remove, or reorder sections.

        Each string is a method name on this class (e.g. ``"build_abstract"``).
        """
        return [
            "build_title",
            "build_abstract",
            "build_problem_restatement",
            "build_problem_analysis",
            "build_model_assumptions",
            "build_notation",
            "build_data_preprocessing",
            "build_models",
            "build_results",
            "build_validation",
            "build_sensitivity",
            "build_model_evaluation",
            "build_conclusion",
            "build_references",
            "build_appendix",
        ]

    # ------------------------------------------------------------------
    # Overridable section methods (return a Markdown string each)
    # ------------------------------------------------------------------

    def build_title(self) -> str:
        """Return the document title / heading block."""
        return "# 数学建模论文"

    def build_abstract(self) -> str:
        """Return the abstract section."""
        return "## 摘要\n\n（待补充）"

    def build_problem_restatement(self) -> str:
        """Return the problem-restatement section."""
        return "## 一、问题重述\n\n" + self.problem_text.strip()

    def build_problem_analysis(self) -> str:
        """Return the problem-analysis section."""
        content = self.notes.get("problem_analysis", "")
        if not content:
            return ""
        return "## 二、问题分析\n\n" + content

    def build_model_assumptions(self) -> str:
        """Return the model-assumptions section."""
        return "## 三、模型假设\n\n（根据具体问题补充假设列表）"

    def build_notation(self) -> str:
        """Return the notation / symbol-table section."""
        return "## 四、符号说明\n\n（根据具体问题补充符号表）"

    def build_data_preprocessing(self) -> str:
        """Return the data-preprocessing section."""
        return "## 五、数据预处理\n\n（根据具体数据描述预处理步骤）"

    def build_models(self) -> str:
        """Return the model-setup section (possibly multiple sub-sections)."""
        plan = self.notes.get("modeling_plan", "")
        if plan:
            return "## 六、模型建立与求解\n\n" + plan
        return "## 六、模型建立与求解\n\n（根据建模方案填充）"

    def build_results(self) -> str:
        """Return the results section."""
        analysis = self.notes.get("result_analysis", "")
        if analysis:
            return "## 七、结果分析\n\n" + analysis
        return "## 七、结果分析\n\n（根据运行结果填充）"

    def build_validation(self) -> str:
        """Return the model-validation section."""
        review = self.notes.get("review_report", "")
        if review:
            return "## 八、模型检验与审稿\n\n" + review
        return ""

    def build_sensitivity(self) -> str:
        """Return the sensitivity / error analysis section."""
        return "## 九、灵敏度分析与误差分析\n\n（待补充灵敏度与误差分析）"

    def build_model_evaluation(self) -> str:
        """Return the model evaluation / generalization section."""
        return (
            "## 十、模型评价与推广\n\n"
            "**优点**：模型基于真实数据运行，结论可复现、可验证。\n\n"
            "**缺点**：部分参数依赖假设或代理变量，需补充实际数据标定。\n\n"
            "**改进方向**：补充实际行为数据后可进一步校准模型参数。\n\n"
            "**推广**：本框架可推广至同类问题的建模与分析场景。"
        )

    def build_conclusion(self) -> str:
        """Return the conclusion section."""
        return "## 十一、结论\n\n（总结全文核心发现与建议）"

    def build_references(self) -> str:
        """Return the references section."""
        return "## 十二、参考文献\n\n（根据实际引用补充参考文献）"

    def build_appendix(self) -> str:
        """Return the appendix section."""
        return (
            "## 附录\n\n"
            "本文全部结果由自动生成的分析代码在真实数据上运行产出，"
            "完整结果数据表与高清图表见下方附录（由系统自动嵌入）。"
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def _load_summary(self) -> list[dict[str, Any]] | None:
        """Load run_summary.json and return the payload list, or None."""
        summary_path = self.logs_dir / "run_summary.json"
        if not summary_path.exists():
            return None
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return None
        if isinstance(payload, list) and payload:
            return payload
        return None

    def _build_figure_map(self, item: dict[str, Any]) -> dict[str, str]:
        """Return {filename: absolute_path} for charts in one summary item."""
        charts: dict[str, str] = {}
        for c in item.get("charts", []):
            name = Path(str(c)).name
            charts[name] = str(Path(str(c)).resolve())
        return charts

    def _fig(self, charts: dict[str, str], suffix: str, caption: str) -> str:
        """Return a Markdown figure block if a chart whose name ends with
        *suffix* exists in *charts*, otherwise ""."""
        for name, path in charts.items():
            if name.endswith(suffix):
                return f"\n![{caption}]({path})\n\n*{caption}*\n"
        return ""
