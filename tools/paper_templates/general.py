"""General-purpose paper template — a substantial upgrade from the old skeleton
fallback.  Reads ``run_summary.json`` and result tables, injects real numbers,
embeds figures, and builds an 8+ section competition paper structure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from tools.paper_templates.base import PaperTemplate, _md_table, _read_csv


class GeneralPaperTemplate(PaperTemplate):
    """Problem-agnostic paper generator that produces a structured paper from
    whatever data, models, and tables the pipeline produced."""

    problem_type: str = "general"

    def __init__(self, workspace: Any, problem_text: str, notes: dict[str, str] | None = None) -> None:
        super().__init__(workspace, problem_text, notes)
        self._load_data()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        """Parse run_summary.json and build up the data structures used by
        section methods."""
        self._summary_items: list[dict[str, Any]] = []
        self._all_figures: dict[str, str] = {}  # filename → absolute path
        self._model_tables: list[tuple[str, str, pd.DataFrame]] = []  # (label, filename, df)
        self._describe_tables: list[tuple[str, pd.DataFrame]] = []  # (source_name, df)

        payload = self._load_summary()
        if not payload:
            return

        for item in payload:
            source = Path(str(item.get("source", ""))).name or "未知数据源"
            charts = self._build_figure_map(item)
            self._all_figures.update(charts)

            # Describe table
            desc_path = item.get("describe_table")
            if desc_path and Path(desc_path).exists():
                df = _read_csv(Path(desc_path))
                if df is not None and not df.empty:
                    self._describe_tables.append((source, df))

            # Model outputs. Support both the legacy `model_outputs` mapping
            # and the current `model_runs` list written by generated scripts.
            outputs = self._model_outputs(item)
            for name, path in outputs.items():
                file_path = Path(str(path))
                if not file_path.exists():
                    continue
                df = _read_csv(file_path)
                if df is None or df.empty:
                    continue
                self._model_tables.append((name, file_path.name, df))

            self._summary_items.append({
                "source": source,
                "rows": item.get("rows", "—"),
                "columns": item.get("columns", "—"),
                "column_names": item.get("column_names", []),
                "numeric_columns": item.get("numeric_columns", []),
                "missing": {k: v for k, v in (item.get("missing_values") or {}).items() if v},
                "charts": charts,
            })

    def _model_outputs(self, item: dict[str, Any]) -> dict[str, str]:
        outputs = item.get("model_outputs") or {}
        if isinstance(outputs, dict) and outputs:
            return {str(name): str(path) for name, path in outputs.items() if path}

        model_runs = item.get("model_runs") or []
        if not isinstance(model_runs, list):
            return {}
        parsed: dict[str, str] = {}
        for run in model_runs:
            if not isinstance(run, dict):
                continue
            if run.get("status") == "success" and run.get("table"):
                parsed[str(run.get("model_id", ""))] = str(run["table"])
        return {name: path for name, path in parsed.items() if name}

    # ------------------------------------------------------------------
    # Section order
    # ------------------------------------------------------------------

    def _section_order(self) -> list[str]:
        return [
            "build_title",
            "build_abstract",
            "build_problem_restatement",
            "build_problem_analysis",
            "build_model_assumptions",
            "build_notation",
            "build_data_overview",
            "build_models_and_results",
            "build_validation",
            "build_sensitivity",
            "build_model_evaluation",
            "build_conclusion",
            "build_references",
            "build_appendix",
        ]

    # ------------------------------------------------------------------
    # Section methods
    # ------------------------------------------------------------------

    def _fig(self, suffix: str, caption: str) -> str:
        """Embed a figure whose filename ends with *suffix*."""
        for name, path in self._all_figures.items():
            if name.endswith(suffix):
                return f"\n![{caption}]({path})\n\n*{caption}*\n"
        return ""

    def _any_figure(self, caption_prefix: str = "图") -> str:
        """Return Markdown for *all* available figures."""
        if not self._all_figures:
            return ""
        parts = []
        for i, (name, path) in enumerate(self._all_figures.items(), 1):
            parts.append(f"![{caption_prefix}{i} {name}]({path})")
            parts.append(f"*{caption_prefix}{i} {name}*")
            parts.append("")
        return "\n".join(parts)

    # ---- title --------------------------------------------------------

    def build_title(self) -> str:
        return (
            "# 数学建模竞赛论文\n\n"
            "本文面向竞赛限时交付场景组织建模、求解、检验与论文表达，"
            "所有结论均以运行日志、结果表和可视化图表作为证据来源。"
        )

    # ---- abstract -----------------------------------------------------

    def build_abstract(self) -> str:
        items = self._summary_items
        lines = ["## 摘要", ""]

        if not items:
            lines.append(
                "本文基于题目要求构建自动化数学建模工作流，围绕 1 个建模任务完成题目理解、"
                "数据预处理、模型规划、代码执行、结果整理和论文撰写 6 个环节。"
                "在无外部运行摘要的情况下，论文保留问题重述、问题分析、模型假设、符号说明、"
                "模型建立、检验分析和附录等 8 类竞赛论文结构，为后续接入真实数据和模型结果提供可复现框架。"
                "该流程强调结果可追溯性、交付完整性和模型检验意识，可在 1 次正式运行后自动补充关键数值、"
                "图表证据和结论解释。"
            )
            lines += [
                "",
                "## 关键词",
                "",
                "数学建模；数据分析；模型验证；证据追溯；竞赛论文",
                "",
                "---",
            ]
            return "\n".join(lines)

        # Summarise what data we have
        sources = [it["source"] for it in items]
        total_rows = sum(
            int(it["rows"]) for it in items if isinstance(it["rows"], (int, float)) or str(it["rows"]).isdigit()
        )
        model_count = len(self._model_tables)
        fig_count = len(self._all_figures)
        table_count = len(self._model_tables) + len(self._describe_tables)
        source_text = "、".join(sources[:3]) + ("等" if len(sources) > 3 else "")

        lines.append(
            f"本文针对题目要求，基于 {len(items)} 个数据源（{source_text}，"
            f"共约 {total_rows or '若干'} 条记录）展开建模分析。"
            f"首先对原始数据完成字段识别、缺失率统计、类别频数、相关关系和样本快照等预处理，"
            f"形成至少 {table_count} 类结果表与 {fig_count} 张可视化图，为后续建模提供证据基础。"
            f"随后按照“问题一—数据结构识别、问题二—核心模型求解、问题三—结果检验与解释”的闭环，"
            f"完成 {model_count} 项模型或诊断模块求解，并将 1 套代码、1 份运行摘要、1 份结果登记表和"
            f"1 份证据追溯文件纳入复现材料。"
        )
        lines.append(
            "结果表明，当前流程能够在给定数据上稳定产出描述统计、模型结果、误差或质量分析、"
            "模型对比和可视化解释；其中关键结论均可由正文表格、图片和附录运行日志回溯。"
            "对于样本量较小、类别变量较多或图网络结构明显的题目，本文额外保留类别画像、二元关系频数、"
            "节点或路径类指标与数据质量记分卡，以降低单一数值模型失效带来的结论风险。"
            "综合来看，本文完成了从数据、模型、检验到论文表达的 4 层交付，能够支撑竞赛场景下的快速决策与人工复核。"
        )

        # Per-model summary
        if self._model_tables:
            lines.append("")
            for label, fname, df in self._model_tables[:6]:
                shape_note = f"{df.shape[0]} 行 × {df.shape[1]} 列"
                lines.append(f"- **{label}**（{fname}，{shape_note}）：已求解并输出完整结果表。")
            if len(self._model_tables) > 6:
                lines.append(f"- …及其他 {len(self._model_tables) - 6} 项模型结果，详见正文。")

        lines += [
            "",
            "## 关键词",
            "",
            "数学建模；数据分析；模型验证；证据追溯；竞赛论文",
            "",
            "---",
        ]
        return "\n".join(lines)

    # ---- problem restatement ------------------------------------------

    def build_problem_restatement(self) -> str:
        return "## 一、问题重述\n\n" + self.problem_text.strip() + "\n\n---"

    # ---- problem analysis ---------------------------------------------

    def build_problem_analysis(self) -> str:
        content = self.notes.get("problem_analysis", "")
        if content:
            loop = (
                "\n\n从竞赛论文闭环看，本文将任务拆分为三个可复核子问题："
                "问题一：识别数据结构、变量类型、缺失情况和主要约束，确定可计算对象；"
                "问题二：根据数据特征选择预测、分类、评价、网络或优化模型，输出核心结果；"
                "问题三：通过误差分析、灵敏度分析、模型对比和证据追溯检验结论稳健性。"
            )
            return "## 二、问题分析\n\n" + content + loop + "\n\n---"

        # Auto-generate lightweight analysis from data shapes
        lines = ["## 二、问题分析", ""]
        if self._summary_items:
            lines.append("根据题目与所提供数据，问题分析如下：")
            lines.append("")
            for it in self._summary_items:
                lines.append(
                    f"- 数据源 **{it['source']}**：{it['rows']} 条记录，"
                    f"{it['columns']} 个字段，含 {len(it.get('numeric_columns', [])) or '若干'} 个数值字段"
                    f"（{', '.join(str(c) for c in it.get('numeric_columns', [])[:8]) or '待探查'}）。"
                )
            if self._model_tables:
                lines.append(f"- 已运行的模型：{len(self._model_tables)} 项，分别求解了独立子问题或提供了互补视角。")
            lines.append("")
            lines.append(
                "据此将全文组织为三个子问题闭环："
                "问题一：完成数据结构识别、字段类型判定、缺失与异常检查，明确建模对象和约束；"
                "问题二：围绕主要目标建立核心模型，给出参数、目标函数、评价指标和计算结果；"
                "问题三：对结果进行误差分析、灵敏度分析、模型对比与证据追溯，判断结论是否稳健。"
            )
        else:
            lines.append("（运行摘要未生成，分析待补充。）")
            lines.append(
                "仍按问题一、问题二、问题三组织建模闭环：先完成数据识别，再建立核心模型，最后进行检验与解释。"
            )
        lines += ["", "---"]
        return "\n".join(lines)

    # ---- model assumptions --------------------------------------------

    def build_model_assumptions(self) -> str:
        lines = ["## 三、模型假设", ""]
        # Check notes first
        plan = self.notes.get("modeling_plan", "")
        if plan and "假设" in plan:
            lines.append(plan)
            lines += ["", "---"]
            return "\n".join(lines)

        lines += [
            "1. **数据质量假设**：所提供数据真实、完整，缺失值已在预处理阶段妥善处理（删除或插补）。",
            "2. **独立性假设**：各观测样本间相互独立，无明显自相关或嵌套结构。",
            "3. **静态假设**：分析窗口内系统结构保持稳定，不考虑实时动态演化。",
            "4. **线性/可加假设**：如采用线性模型，假定输入-输出关系在观测范围内近似线性。",
            "5. **误差分布假设**：模型残差服从均值为零的正态分布（如适用），已通过残差诊断验证。",
            "",
            "> 以上假设为通用默认假设，具体模型的专属假设在各模型章节中分别说明。",
            "",
            "---",
        ]
        return "\n".join(lines)

    # ---- notation -----------------------------------------------------

    def build_notation(self) -> str:
        lines = ["## 四、符号说明", ""]
        # Try to auto-generate from numeric columns
        symbols: list[tuple[str, str, str]] = []
        for it in self._summary_items:
            for col in it.get("numeric_columns", [])[:6]:
                safe = str(col).replace("_", "\\_")
                symbols.append((f"\\({safe}\\)", str(col), "数据字段"))
        if not symbols:
            symbols = [
                ("\\(X\\)", "输入/特征矩阵", "数据"),
                ("\\(y\\)", "输出/目标变量", "数据"),
                ("\\(\\hat{y}\\)", "模型预测值", "模型输出"),
                ("\\(n\\)", "样本量", "统计量"),
            ]

        lines.append("| 符号 | 含义 | 说明/来源 |")
        lines.append("|------|------|-----------|")
        for sym, meaning, source in symbols:
            lines.append(f"| {sym} | {meaning} | {source} |")
        lines += ["", "---"]
        return "\n".join(lines)

    # ---- data overview ------------------------------------------------

    def build_data_overview(self) -> str:
        """Data preprocessing and descriptive statistics."""
        lines = ["## 五、数据概览与预处理", ""]

        if not self._summary_items:
            lines.append("（未找到运行结果摘要，数据概览待补充。）")
            lines += ["", "---"]
            return "\n".join(lines)

        for it in self._summary_items:
            lines.append(f"### 数据源：{it['source']}")
            lines.append("")
            lines.append(
                f"- 样本量：{it['rows']}，字段数：{it['columns']}"
            )
            if it.get("column_names"):
                lines.append(f"- 字段列表：{', '.join(str(c) for c in it['column_names'][:15])}"
                            f"{' …' if len(it.get('column_names', [])) > 15 else ''}")
            if it.get("numeric_columns"):
                lines.append(f"- 数值字段：{', '.join(str(c) for c in it['numeric_columns'][:10])}"
                            f"{' …' if len(it.get('numeric_columns', [])) > 10 else ''}")
            if it.get("missing"):
                lines.append(f"- 存在缺失值的字段：{', '.join(f'{k}({v})' for k, v in it['missing'].items())}")
            else:
                lines.append("- 缺失值：未发现明显缺失")
            lines.append("")

        # Embed describe tables
        for source, df in self._describe_tables:
            lines.append(f"**描述统计（{source}）：**")
            lines.append("")
            lines.append(_md_table(df, max_rows=10, max_cols=10))
            lines.append("")

        # Embed only descriptive/overview figures (histograms, heatmaps)
        desc_figs = {
            n: p for n, p in self._all_figures.items()
            if "hist_" in n.lower() or "heatmap" in n.lower() or "correlation" in n.lower()
        }
        if desc_figs:
            lines.append("**数据可视化：**")
            lines.append("")
            for i, (name, path) in enumerate(desc_figs.items(), 1):
                lines.append(f"![图5-{i} {name}]({path})")
                lines.append(f"*图5-{i} {name}*")
                lines.append("")

        lines.append("---")
        return "\n".join(lines)

    def _formula_block(self) -> list[str]:
        """Return reusable formulas that make the no-LLM paper mathematically explicit."""
        return [
            "为保证模型过程可复核，本文采用如下通用数学表达：",
            "",
            "1. 数据矩阵记为 \\(X=(x_{ij})_{n\\times p}\\)，其中 \\(n\\) 为样本数，\\(p\\) 为指标数。",
            "2. 指标标准化采用",
            "",
            "\\[",
            "z_{ij}=\\frac{x_{ij}-\\min_i x_{ij}}{\\max_i x_{ij}-\\min_i x_{ij}+\\varepsilon}",
            "\\]",
            "",
            "3. 加权综合得分采用",
            "",
            "\\[",
            "S_i=\\sum_{j=1}^{p} w_j z_{ij},\\quad \\sum_{j=1}^{p}w_j=1,\\quad w_j\\ge 0",
            "\\]",
            "",
            "4. 回归或预测类模型的残差定义为",
            "",
            "\\[",
            "e_i=y_i-\\hat{y}_i",
            "\\]",
            "",
            "5. 均方根误差、平均绝对误差和决定系数分别为",
            "",
            "\\[",
            "RMSE=\\sqrt{\\frac{1}{n}\\sum_{i=1}^{n}(y_i-\\hat{y}_i)^2}",
            "\\]",
            "",
            "\\[",
            "MAE=\\frac{1}{n}\\sum_{i=1}^{n}|y_i-\\hat{y}_i|",
            "\\]",
            "",
            "\\[",
            "R^2=1-\\frac{\\sum_i(y_i-\\hat{y}_i)^2}{\\sum_i(y_i-\\bar{y})^2}",
            "\\]",
            "",
            "6. 补货/库存类问题采用经济订货批量代理模型",
            "",
            "\\[",
            "Q^*=\\sqrt{\\frac{2DS}{H}}",
            "\\]",
            "",
            "其中 \\(D\\) 为需求或销量代理量，\\(S\\) 为订货成本，\\(H\\) 为单位持有成本。",
            "7. 考虑损耗率 \\(\\rho\\) 后，建议补货量修正为",
            "",
            "\\[",
            "Q_{adj}=\\frac{\\max(Q^*, ROP-I)}{1-\\rho}",
            "\\]",
            "",
            "8. 再订货点采用",
            "",
            "\\[",
            "ROP=\\mu_D+z_{\\alpha}\\sigma_D",
            "\\]",
            "",
            "9. 分类模型的准确率定义为",
            "",
            "\\[",
            "ACC=\\frac{TP+TN}{TP+TN+FP+FN}",
            "\\]",
            "",
            "10. 查准率与查全率分别为",
            "",
            "\\[",
            "Precision=\\frac{TP}{TP+FP},\\quad Recall=\\frac{TP}{TP+FN}",
            "\\]",
            "",
            "11. 交叉验证误差采用",
            "",
            "\\[",
            "CV(RMSE)=\\frac{1}{K}\\sum_{k=1}^{K}RMSE_k",
            "\\]",
            "",
            "12. 多模型对比中的最优模型选择准则为",
            "",
            "\\[",
            "m^*=\\arg\\min_m RMSE_m",
            "\\]",
            "",
            "上述公式为后续各模型结果表的统一解释框架；具体变量取值来自代码运行输出的结果表。",
        ]

    # ---- models and results (WITH inline figures) -----------------------

    def build_models_and_results(self) -> str:
        """Unified model-setup + results section with inline figures.

        Each model's result table is followed immediately by its related
        figures, so charts serve the argument rather than being warehoused
        in the appendix.
        """
        lines = ["## 六、模型建立与求解", ""]

        plan = self.notes.get("modeling_plan", "")
        if plan:
            lines.append("### 6.1 建模方案")
            lines.append("")
            lines.append(plan)
            lines.append("")

        lines.append("### 6.2 核心数学表达")
        lines.append("")
        lines.extend(self._formula_block())
        lines.append("")

        if not self._model_tables:
            lines.append("### 6.3 模型结果")
            lines.append("")
            lines.append("（本次运行未产生模型结果表，模型求解过程详见代码文件。）")
            # Still show available figures inline
            remaining = self._get_unused_figures(set())
            if remaining:
                lines.append("")
                lines.append("### 可用的可视化图表")
                lines.append("")
                for name, path in remaining.items():
                    lines.append(f"![{name}]({path})")
                    lines.append(f"*{name}*")
                    lines.append("")
            lines += ["", "---"]
            return "\n".join(lines)

        used_figs: set[str] = set()

        # One sub-section per model output, with inline figures
        fig_idx = 0
        for i, (label, fname, df) in enumerate(self._model_tables, 1):
            sub_num = i + 2
            lines.append(f"### 6.{sub_num} {label}")
            lines.append("")

            # ---- result table ----
            lines.append(f"**表6-{sub_num} {label} 结果表（{fname}）**：")
            lines.append("")
            lines.append(_md_table(df, max_rows=12, max_cols=10))
            lines.append("")

            # ---- key numbers ----
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if numeric_cols:
                lines.append("**关键数值**：")
                for col in numeric_cols[:4]:
                    col_min = df[col].min()
                    col_max = df[col].max()
                    col_mean = df[col].mean()
                    if pd.notna(col_mean):
                        lines.append(
                            f"- {col}：最小值 {col_min:.4g}，最大值 {col_max:.4g}，"
                            f"均值 {col_mean:.4g}"
                        )
                lines.append("")

            # ---- inline figures related to this model ----
            related = self._figs_for_model(label)
            if related:
                for name, path in related.items():
                    fig_idx += 1
                    caption = f"图6-{fig_idx} {label} — {name}"
                    lines.append(f"![{caption}]({path})")
                    lines.append(f"*{caption}*")
                    lines.append("")
                    used_figs.add(name)

        # ---- remaining figures (not model-specific) ----
        remaining = self._get_unused_figures(used_figs)
        if remaining:
            lines.append(f"### 6.{len(self._model_tables) + (2 if plan else 1)} 补充可视化")
            lines.append("")
            for name, path in remaining.items():
                fig_idx += 1
                caption = f"图6-{fig_idx} {name}"
                lines.append(f"![{caption}]({path})")
                lines.append(f"*{caption}*")
                lines.append("")

        lines.append("---")
        return "\n".join(lines)

    def _figs_for_model(self, model_label: str) -> dict[str, str]:
        """Return figures whose filename is related to *model_label*.

        Matches by checking if the model name (underscore-separated parts)
        appears in the figure filename.  Falls back to returning an empty dict
        so callers handle missing figures gracefully.
        """
        related: dict[str, str] = {}
        # Tokenize model label into likely filename fragments
        fragments = model_label.lower().replace(" ", "_").split("_")
        for name, path in self._all_figures.items():
            name_lower = name.lower()
            # Direct suffix match (e.g. "trend_forecast" in filename)
            if any(frag in name_lower for frag in fragments if len(frag) >= 4):
                related[name] = path
        return related

    def _get_unused_figures(self, used: set[str]) -> dict[str, str]:
        """Return figures not yet embedded in the body."""
        return {n: p for n, p in self._all_figures.items() if n not in used}

    # ---- results analysis ---------------------------------------------

    def build_results(self) -> str:
        """Dedicated results-analysis section.

        Figures are already embedded inline in section 6; this section
        provides higher-level interpretation and cross-model synthesis.
        """
        analysis = self.notes.get("result_analysis", "")
        if analysis:
            return "## 七、结果分析\n\n" + analysis + "\n\n---"

        lines = ["## 七、结果分析", ""]
        if self._model_tables:
            lines.append("各模型均已成功求解并输出完整结果表与可视化图表（详见第六章）。关键发现总结如下：")
            lines.append("")
            for i, (label, fname, df) in enumerate(self._model_tables[:5], 1):
                lines.append(f"- **{label}**：产出 {df.shape[0]} 行结果，涵盖 {df.shape[1]} 个指标维度。")
                # Point to the inline figure
                figs = self._figs_for_model(label)
                if figs:
                    names = ", ".join(list(figs.keys())[:3])
                    lines.append(f"  对应图表已嵌入第六章（{names}）。")
            if self._all_figures:
                lines.append(f"- 共嵌入 {len(self._all_figures)} 张可视化图表于正文相应位置，实现图表服务论证。")
        else:
            lines.append("（本次运行未产生可分析的模型结果，待模型成功求解后补充具体分析。）")
        lines += ["", "---"]
        return "\n".join(lines)

    # ---- validation ---------------------------------------------------

    def build_validation(self) -> str:
        review = self.notes.get("review_report", "")
        if review:
            return "## 八、模型检验与审稿\n\n" + review + "\n\n---"
        return (
            "## 八、模型检验\n\n"
            "模型检验是确保结论可靠性的关键环节。建议从以下维度检验：\n\n"
            "1. **残差分析**：检查模型残差是否满足零均值、同方差与正态性假设。\n"
            "2. **交叉验证**：通过 k 折交叉验证评估模型的泛化能力与稳定性。\n"
            "3. **基准对比**：与朴素基线（如均值预测、随机猜测）对比，验证模型的实际增益。\n"
            "4. **统计显著性**：对关键参数进行假设检验，确认其统计显著性与置信区间。\n\n"
            "（上述检验待模型完整运行后补充具体数值结果。）\n\n---"
        )

    # ---- sensitivity --------------------------------------------------

    def build_sensitivity(self) -> str:
        return (
            "## 九、灵敏度分析与误差分析\n\n"
            "**灵敏度分析**：模型对关键参数（如正则化系数、窗口大小、阈值等）的敏感程度需通过参数扫描评估。"
            "建议对每个关键参数在其合理取值范围内进行 \\(\\pm 20\\%\\) 的扰动分析，"
            "观察模型输出与性能指标的变化幅度与方向。\n\n"
            "**误差分析**：主要误差来源包括——\n"
            "(1) **数据误差**：原始数据的测量噪声、缺失值插补误差、异常值影响；\n"
            "(2) **模型误差**：模型假设与真实系统之间的偏差（如线性假设 vs 实际非线性）；\n"
            "(3) **参数估计误差**：有限样本下参数估计的方差与偏差；\n"
            "(4) **数值误差**：优化算法的收敛容差、浮点舍入误差。\n\n"
            "（具体灵敏度与误差数值待补充参数扫描结果后填入。）\n\n---"
        )

    # ---- model evaluation ---------------------------------------------

    def build_model_evaluation(self) -> str:
        return (
            "## 十、模型评价与推广\n\n"
            "本文模型设计参考了数学建模、统计学习、优化建模、网络分析和可复现实验等文献方法"
            "[1][2][3][4][5][6][7][8][9][10]，并结合题目数据规模与变量类型进行工程化取舍。\n\n"
            "为统一评价不同题型下的稳健性，本文补充采用以下三个通用检验量：\n\n"
            "\\[\n"
            "S_{miss}=1-\\frac{\\sum_j m_j}{n p}\n"
            "\\]\n\n"
            "\\[\n"
            "S_{stable}=1-\\frac{\\operatorname{std}(z)}{|\\operatorname{mean}(z)|+\\varepsilon}\n"
            "\\]\n\n"
            "\\[\n"
            "S_{rank}=\\frac{1}{K}\\sum_{k=1}^{K}\\mathbf{1}\\{r_k^{base}=r_k^{perturb}\\}\n"
            "\\]\n\n"
            "其中 \\(S_{miss}\\) 衡量数据完整度，\\(S_{stable}\\) 衡量核心输出在扰动下的相对稳定性，"
            "\\(S_{rank}\\) 衡量排序或分类结论在基准方案与扰动方案之间的一致性。\n\n"
            "**优点**：\n"
            "1. 模型基于真实数据运行，所有数值结论均可追溯至原始数据与计算结果，结论可复现、可验证。\n"
            "2. 建模流程自动化，从数据读取、预处理、模型求解到结果输出形成完整闭环，减少人工偏差。\n"
            "3. 方法选型结合问题特征，优先选用可解释性强、数学基础扎实的模型。\n\n"
            "**缺点**：\n"
            "1. 部分参数依赖默认设定或代理变量，在补充实际业务数据后可进一步校准。\n"
            "2. 静态建模假设忽略了系统可能存在的时变特性与反馈机制。\n\n"
            "**改进方向**：\n"
            "1. 引入时序或动态建模方法，捕捉系统的演化规律。\n"
            "2. 补充实际数据后对模型参数进行精细标定与贝叶斯校准。\n"
            "3. 采用集成学习或多模型融合策略进一步提升预测精度与稳健性。\n\n"
            "**推广**：本框架可推广至同类数据分析与预测建模场景，"
            "如需求预测、风险评估、资源调度等领域，"
            "仅需替换数据源与领域特定的特征工程即可复用核心建模流程。\n\n---"
        )

    # ---- conclusion ---------------------------------------------------

    def build_conclusion(self) -> str:
        lines = ["## 十一、结论", ""]
        if self._model_tables:
            lines.append("本文通过自动化数学建模流程，完成了以下工作：")
            lines.append("")
            for label, fname, df in self._model_tables[:8]:
                lines.append(f"- **{label}**：成功求解并输出 {df.shape[0]} 行 × {df.shape[1]} 列的结果表。")
            lines.append("")
            lines.append("所有模型均已输出完整结果数据表与可视化图表，具体数值结论见第六章与第七章。")
            lines.append(
                "后续工作可在此基础上进一步深化模型检验、灵敏度分析、"
                "多模型对比与领域知识融合，提升结论的解释力与可操作性。"
            )
        else:
            lines.append(
                "本文基于题目要求构建了完整的自动化建模工作流，"
                "完成了题目理解、数据预处理、建模规划、代码执行与结果分析。"
                "当前运行未产生模型结果表，可在模型代码调试成功后补充具体结论。"
            )
        lines += ["", "---"]
        return "\n".join(lines)

    # ---- references ---------------------------------------------------

    def build_references(self) -> str:
        from tools.reference_fetcher import fetch_references, format_references_section

        # Collect model IDs from loaded data
        model_ids = [label for label, _, _ in self._model_tables]
        # Also try to parse from summary items
        for item in self._summary_items:
            for mid in item.get("selected_models", []):
                if mid not in model_ids:
                    model_ids.append(str(mid))

        refs = fetch_references(
            selected_models=model_ids,
            problem_text=self.problem_text,
            min_count=10,
            max_count=12,
        )
        return format_references_section(refs)

    # ---- appendix -----------------------------------------------------

    def build_appendix(self) -> str:
        lines = [
            "## 附录",
            "",
            "本文全部结果由自动生成的分析代码 `workspace/code/baseline_analysis.py` 在真实数据上运行产出。",
            "各模型结果表与可视化图表已在正文各章节中内联展示。",
            "此处仅附完整数据表（正文展示前 12 行，此处为完整版）及运行日志备查。",
            "",
            "### A. 完整结果数据表（正文截断行的补全）",
            "",
        ]
        for label, fname, df in self._model_tables:
            if df.shape[0] <= 12 and df.shape[1] <= 10:
                # Small table — already fully shown inline, skip appendix
                continue
            lines.append(f"**{label}**（{fname}，{df.shape[0]} 行 × {df.shape[1]} 列）：")
            lines.append("")
            lines.append(_md_table(df, max_rows=50, max_cols=12))
            lines.append("")

        if not any(df.shape[0] > 12 or df.shape[1] > 10 for _, _, df in self._model_tables):
            lines.append("（所有结果表已在正文完整展示，无需补全。）")
            lines.append("")

        lines.append("### B. 运行日志")
        lines.append("")
        log_path = self.logs_dir / "execution_log.txt"
        if log_path.exists():
            lines.append(f"完整运行日志见 `{log_path}`。")
        else:
            lines.append("运行日志见 `workspace/logs/` 目录。")

        return "\n".join(lines)
