from __future__ import annotations

import json
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

# Allow `streamlit run app/streamlit_app.py` from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from agents.base import (
    PhaseStatus,
    WorkflowPhase,
    WorkflowState,
    format_error_for_display,
)
from agents.export_agent import export_paper
from app.config import WORKSPACE
from tools.file_tool import read_docx_text, read_pdf_text, validate_data_file
from workflows.modeling_workflow import ModelingWorkflow

EXPORT_LABELS = {"docx": "Word (.docx)", "pdf": "PDF (.pdf)", "latex": "LaTeX (.tex)"}
MAX_DATA_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_PROBLEM_UPLOAD_BYTES = 10 * 1024 * 1024
MIME_TYPES = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pdf": "application/pdf",
    "latex": "text/x-tex",
}
AGENT_LABELS = {
    "problem_agent": "题目解析",
    "modeling_agent": "建模方案",
    "model_selection_agent": "模型选择",
    "modeling_critic_agent": "模型批评",
    "decision_agent": "模型决策",
    "coding_agent": "代码生成",
    "execution_agent": "执行与修复",
    "analysis_agent": "结果分析",
    "evidence_agent": "证据映射",
    "writing_agent": "论文写作",
    "math_reviewer": "数学审稿",
    "fact_reviewer": "事实审稿",
    "structure_reviewer": "结构审稿",
    "language_reviewer": "语言审稿",
    "review_agent": "质量审阅",
    "export_agent": "文档导出",
}
STATUS_LABELS = {
    "pending": "等待",
    "started": "运行中",
    "completed": "完成",
    "skipped": "跳过",
    "failed": "异常",
    "waiting_for_user": "⏸ 待确认",
    "approved": "✅ 已确认",
    "rejected": "❌ 已拒绝",
    "needs_revision": "🔄 需返工",
}

PHASE_ICONS = {
    "problem_analysis": "🔍",
    "model_proposal": "💡",
    "model_critique": "🔬",
    "model_decision": "✅",
    "experiment_plan": "🧪",
    "code_plan": "📋",
    "code_generation": "⚙️",
    "execution": "▶️",
    "result_analysis": "📊",
    "evidence_mapping": "🔗",
    "paper_outline": "📝",
    "section_writing": "✍️",
    "fact_review": "📋",
    "math_review": "🧮",
    "structure_review": "🏗️",
    "language_review": "📖",
    "export": "📦",
    "complete": "🏁",
}

FEEDBACK_TEMPLATES: dict[str, dict[str, str]] = {
    "model_decision": {
        "强调可解释性": "请优先选择可解释、便于论文表述的模型，并补充模型适用条件、失效场景和与基线模型的对比理由。",
        "增强鲁棒性": "请重新检查主模型对缺失值、异常值、小样本和参数扰动的鲁棒性，并在方案中加入敏感性分析。",
        "贴合题目任务": "请逐个子问题核对模型是否覆盖题目要求，避免只给通用模型；每个模型需要对应具体输入、输出和评价指标。",
    },
    "experiment_plan": {
        "补充消融实验": "请加入消融实验，说明去掉关键变量、约束或模型模块后结果如何变化。",
        "补充基线对比": "请加入至少一个简单基线和一个竞争模型，并明确评价指标、随机种子和复现实验设置。",
        "控制数据泄漏": "请重新检查训练/验证/测试划分，避免时间泄漏、目标泄漏和同源样本重复出现在不同集合。",
    },
    "code_plan": {
        "增加可复现性": "请在代码计划中明确随机种子、输入输出文件、日志位置和失败重试策略。",
        "细化文件职责": "请把数据读取、建模、评估和制图拆分为清晰函数，并说明每个函数的输入输出契约。",
        "加入诊断输出": "请增加数据质量检查、模型指标表、关键中间结果和异常处理说明。",
    },
    "result_analysis": {
        "强化结论证据": "请把每条核心结论绑定到具体结果表、图表或指标，并说明该结论的局限性。",
        "补充异常解释": "请分析异常值、反常趋势和模型表现较差的子问题，避免只描述正向结果。",
        "增加竞赛表达": "请把结果分析改写成论文可直接使用的表述，突出可复现数值、对比提升和现实解释。",
    },
    "paper_outline": {
        "突出解题主线": "请重排提纲，使问题分析、模型建立、求解、结果验证和结论建议之间的逻辑链更清晰。",
        "绑定证据编号": "请为每个结果章节补充证据 ID 或结果表来源，避免后续写作出现无证据结论。",
        "压缩弱章节": "请合并重复或空泛章节，把篇幅留给模型推导、结果分析、灵敏度分析和优缺点。",
    },
    "language_review": {
        "消除口语化": "请删除聊天式语气、占位符和泛泛承诺，改成正式竞赛论文语言。",
        "统一术语": "请统一变量、模型、图表和章节术语，确保摘要、正文、公式和结论命名一致。",
        "提高摘要密度": "请增强摘要的信息密度，补足方法、关键数值结果、验证方式和主要结论。",
    },
}

st.set_page_config(page_title="数学建模智能体 · 阶段工作台", page_icon="📐", layout="wide")


# ── helpers ──────────────────────────────────────────────────────────────
def save_uploaded_data(files) -> list[Path]:
    WORKSPACE.ensure_dirs()
    saved: list[Path] = []
    for uploaded in files or []:
        payload = uploaded.getvalue()
        if len(payload) > MAX_DATA_UPLOAD_BYTES:
            raise ValueError(f"上传文件 {uploaded.name!r} 超过 50 MB 限制。")
        safe_name = _safe_upload_name(uploaded.name)
        if Path(safe_name).suffix.lower() not in {".csv", ".tsv", ".xlsx", ".xls"}:
            raise ValueError(f"不支持的数据文件类型：{safe_name}")
        target = (WORKSPACE.data_dir / safe_name).resolve()
        if target.parent != WORKSPACE.data_dir.resolve():
            raise ValueError("上传文件名包含非法路径。")
        target.write_bytes(payload)
        validate_data_file(target)
        saved.append(target)
    return saved


def render_data_upload_precheck(files) -> None:
    if not files:
        return
    rows: list[dict[str, Any]] = []
    for uploaded in files:
        payload = uploaded.getvalue()
        suffix = Path(uploaded.name).suffix.lower()
        row: dict[str, Any] = {
            "文件": uploaded.name,
            "大小(MB)": round(len(payload) / (1024 * 1024), 2),
            "类型": suffix.lstrip("."),
            "状态": "待校验",
            "预览行": "",
            "列数": "",
            "字段预览": "",
        }
        try:
            if len(payload) > MAX_DATA_UPLOAD_BYTES:
                raise ValueError("超过 50 MB 限制")
            if suffix == ".csv":
                df = pd.read_csv(BytesIO(payload), nrows=100)
            elif suffix == ".tsv":
                df = pd.read_csv(BytesIO(payload), sep="\t", nrows=100)
            elif suffix in {".xlsx", ".xls"}:
                df = pd.read_excel(BytesIO(payload), nrows=100)
            else:
                raise ValueError("不支持的文件类型")
            row.update({
                "状态": "可读取",
                "预览行": len(df),
                "列数": len(df.columns),
                "字段预览": ", ".join(str(col) for col in list(df.columns)[:8]),
            })
        except Exception as exc:
            row["状态"] = f"预检失败：{exc}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _safe_upload_name(name: str) -> str:
    basename = name.replace("\\", "/").rsplit("/", 1)[-1].strip()
    basename = re.sub(r"[\x00-\x1f<>:\"|?*]", "_", basename)
    if basename in {"", ".", ".."}:
        raise ValueError("上传文件名无效。")
    return basename[:180]


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}m {rest:.0f}s"


def format_agent_label(agent_name: str) -> str:
    if agent_name.startswith("writing_agent_retry_"):
        return f"论文写作重试 {agent_name.rsplit('_', 1)[-1]}"
    if agent_name.startswith("review_agent_retry_"):
        return f"质量审阅重试 {agent_name.rsplit('_', 1)[-1]}"
    if agent_name.startswith("writing_agent_revise_"):
        return f"写作修订: {agent_name.replace('writing_agent_revise_', '')}"
    return AGENT_LABELS.get(agent_name, agent_name)


def read_log_tail(workspace, max_lines: int = 80) -> str:
    log_path = workspace.logs_dir / "agent.log"
    if not log_path.exists():
        return "日志文件尚未生成。"
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"无法读取日志：{exc}"
    return "\n".join(lines[-max_lines:])


# ── progress tracker ─────────────────────────────────────────────────────
def make_progress_tracker(agent_names: list[str]) -> dict[str, Any]:
    return {
        "started_at": time.perf_counter(),
        "elapsed": 0.0,
        "current": "准备启动",
        "errors": 0,
        "workspace": WORKSPACE,
        "rows": {
            name: {
                "agent": name,
                "stage": format_agent_label(name),
                "status": "pending",
                "duration": None,
                "note": "",
            }
            for name in agent_names
        },
    }


def render_runtime_progress(container, tracker: dict[str, Any]) -> None:
    rows = list(tracker["rows"].values())
    total = max(len(rows), 1)
    finished = sum(1 for row in rows if row["status"] in {"completed", "skipped", "failed"})
    progress = min(finished / total, 1.0)
    workspace = tracker.get("workspace") or WORKSPACE

    with container.container():
        st.subheader("运行进度")
        st.progress(progress, text=f"{int(progress * 100)}% · {tracker['current']}")
        cols = st.columns(4)
        cols[0].metric("当前阶段", tracker["current"])
        cols[1].metric("阶段完成", f"{finished}/{total}")
        cols[2].metric("累计耗时", format_duration(tracker["elapsed"]))
        cols[3].metric("错误数", tracker.get("errors", 0))
        st.dataframe(
            pd.DataFrame([
                {"阶段": row["stage"], "状态": STATUS_LABELS.get(row["status"], row["status"]),
                 "耗时": format_duration(row["duration"]), "说明": row["note"]}
                for row in rows
            ]), hide_index=True, use_container_width=True,
        )
        with st.expander("日志与诊断入口", expanded=False):
            st.caption(f"日志：{workspace.logs_dir / 'agent.log'}")
            st.code(read_log_tail(workspace), language="text")


def make_progress_callback(container, tracker: dict[str, Any]) -> Callable:
    def progress_callback(agent_name: str, status: str, state) -> None:
        now = time.perf_counter()
        tracker["elapsed"] = now - tracker["started_at"]
        if state is not None:
            tracker["errors"] = len(getattr(state, "errors", []))
            tracker["workspace"] = getattr(state, "workspace", WORKSPACE)
        row = tracker["rows"].setdefault(agent_name, {
            "agent": agent_name, "stage": format_agent_label(agent_name),
            "status": "pending", "duration": None, "note": "",
        })
        row["status"] = status
        if status == "started":
            row["started_at"] = now; row["note"] = "正在处理"
            tracker["current"] = row["stage"]
        elif status == "completed":
            duration = getattr(state, "durations", {}).get(agent_name) if state else None
            row["duration"] = duration if duration is not None else now - row.get("started_at", now)
            row["note"] = "已完成"; tracker["current"] = f"{row['stage']}完成"
        elif status == "skipped":
            row["duration"] = 0.0; row["note"] = "按条件跳过"
            tracker["current"] = f"{row['stage']}跳过"
        elif status == "failed":
            row["duration"] = now - row.get("started_at", now)
            row["note"] = "出现异常"; tracker["current"] = f"{row['stage']}异常"
        else:
            row["note"] = status; tracker["current"] = row["stage"]
        render_runtime_progress(container, tracker)
    return progress_callback


def run_with_progress(wf: ModelingWorkflow, operation: Callable[[], WorkflowState], message: str) -> WorkflowState:
    """Run a workflow operation with the shared progress panel attached."""
    progress_panel = st.empty()
    tracker = make_progress_tracker([a.name for a in wf.agents])
    render_runtime_progress(progress_panel, tracker)
    previous_callback = wf.progress_callback
    wf.progress_callback = make_progress_callback(progress_panel, tracker)
    try:
        with st.spinner(message):
            state = operation()
        tracker["elapsed"] = time.perf_counter() - tracker["started_at"]
        tracker["current"] = "运行完成"
        tracker["workspace"] = getattr(state, "workspace", tracker.get("workspace") or WORKSPACE)
        render_runtime_progress(progress_panel, tracker)
        return state
    finally:
        wf.progress_callback = previous_callback


# ── phase sidebar ─────────────────────────────────────────────────────────
def _find_paused_phase(state: WorkflowState | None) -> WorkflowPhase | None:
    if state is None:
        return None
    for phase in WorkflowPhase:
        if state.get_phase_status(phase) == PhaseStatus.WAITING_FOR_USER:
            return phase
    return None


def _phase_status_marker(status: PhaseStatus) -> str:
    if status == PhaseStatus.COMPLETED:
        return "✅"
    if status == PhaseStatus.APPROVED:
        return "☑️"
    if status == PhaseStatus.WAITING_FOR_USER:
        return "⏸"
    if status == PhaseStatus.RUNNING:
        return "⏳"
    if status == PhaseStatus.NEEDS_REVISION:
        return "🔄"
    if status == PhaseStatus.FAILED:
        return "❌"
    if status == PhaseStatus.SKIPPED:
        return "⏭"
    return "⚪"


def _default_focus_phase(state: WorkflowState) -> WorkflowPhase:
    paused = _find_paused_phase(state)
    if paused:
        return paused
    selected_value = st.session_state.get("selected_phase")
    if selected_value:
        try:
            return WorkflowPhase(selected_value)
        except ValueError:
            pass
    return state.phase


def _phase_status_counts(state: WorkflowState) -> dict[str, int]:
    counts = {status.value: 0 for status in PhaseStatus}
    for phase in WorkflowPhase:
        status = state.get_phase_status(phase)
        counts[status.value] = counts.get(status.value, 0) + 1
    return counts


def _phase_completion_ratio(state: WorkflowState) -> float:
    counts = _phase_status_counts(state)
    done = counts.get(PhaseStatus.COMPLETED.value, 0) + counts.get(PhaseStatus.APPROVED.value, 0)
    return min(done / max(len(list(WorkflowPhase)), 1), 1.0)


def _next_user_action(state: WorkflowState) -> str:
    paused = _find_paused_phase(state)
    if paused:
        return f"确认或返工：{paused.label}"
    if state.errors:
        return "查看最近错误并决定是否从相关阶段重跑"
    if state.phase == WorkflowPhase.COMPLETE or state.get_phase_status(WorkflowPhase.COMPLETE) == PhaseStatus.COMPLETED:
        return "检查产物并下载导出文件"
    return f"继续执行：{state.phase.label}"


def _feedback_templates_for_phase(phase: WorkflowPhase) -> dict[str, str]:
    return FEEDBACK_TEMPLATES.get(phase.value, {})


def _split_csv_items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _revision_target_options(state: WorkflowState) -> dict[str, str]:
    options = {
        "paper_outline": "论文提纲",
        "paper_draft": "论文全文",
        "model_decision": "模型决策",
    }
    if state.paper_outline:
        for index, section in enumerate(state.paper_outline.sections, start=1):
            section_id = str(section.get("id") or f"section_{index}").strip()
            title = str(section.get("title") or section_id).strip()
            if section_id:
                options[f"paper_section:{section_id}"] = f"论文章节：{title}"
    if state.review_findings and state.review_findings.issues:
        for index, issue in enumerate(state.review_findings.issues, start=1):
            issue_id = str(issue.get("id") or issue.get("title") or index).strip()
            label = str(issue.get("title") or issue.get("issue") or issue.get("message") or issue_id).strip()
            options[f"review_finding:{issue_id}"] = f"审稿问题：{label[:36]}"
    return options


def render_workbench_overview(state: WorkflowState) -> None:
    ratio = _phase_completion_ratio(state)
    counts = _phase_status_counts(state)
    st.subheader("工作台总览")
    st.progress(ratio, text=f"{int(ratio * 100)}% · {_next_user_action(state)}")
    overview = st.columns(3)
    overview[0].metric("已完成", counts.get(PhaseStatus.COMPLETED.value, 0))
    overview[1].metric("待确认", counts.get(PhaseStatus.WAITING_FOR_USER.value, 0))
    overview[2].metric("需返工", counts.get(PhaseStatus.NEEDS_REVISION.value, 0))


def render_phase_nav(state: WorkflowState | None) -> WorkflowPhase | None:
    """Render phase navigation sidebar. Returns the phase the user clicked."""
    with st.sidebar:
        st.header("📐 阶段导航")
        if state is None:
            st.info("尚未开始运行")
            return None

        render_workbench_overview(state)
        st.divider()

        phase_options: list[str] = []
        phase_map: dict[str, WorkflowPhase] = {}
        for phase in WorkflowPhase:
            status = state.get_phase_status(phase)
            icon = PHASE_ICONS.get(phase.value, "⚪")
            marker = _phase_status_marker(status)
            label = f"{marker} {icon} {phase.label}"
            phase_options.append(label)
            phase_map[label] = phase

        current_paused = _find_paused_phase(state)
        if current_paused:
            st.warning(f"⏸ 待确认：**{current_paused.label}**")

        focus_phase = _default_focus_phase(state)
        default_label = next(
            (label for label in phase_options if phase_map[label] == focus_phase),
            phase_options[0],
        )
        selected_label = st.selectbox(
            "跳转到阶段",
            options=phase_options,
            index=phase_options.index(default_label) if phase_options else 0,
            label_visibility="collapsed",
            key="phase_nav_select",
        )
        selected_phase = phase_map[selected_label]
        st.session_state["selected_phase"] = selected_phase.value
        st.caption(f"当前：{state.phase.label} | 共{len(list(WorkflowPhase))}阶段")

        st.divider()
        if state.decisions:
            with st.expander(f"📋 决策历史 ({len(state.decisions)}条)", expanded=False):
                for d in state.decisions[-10:]:
                    st.caption(f"{d.get('phase','')} · {d.get('action','')} · {d.get('timestamp','')[:19]}")
        return selected_phase


# ── content tabs ──────────────────────────────────────────────────────────
def _render_list_block(title: str, items: list[Any], limit: int = 8) -> None:
    clean_items = [str(item) for item in items if str(item).strip()]
    if not clean_items:
        return
    st.caption(title)
    for item in clean_items[:limit]:
        st.markdown(f"- {item}")
    if len(clean_items) > limit:
        st.caption(f"另有 {len(clean_items) - limit} 项未展开。")


def _compact_json_expander(title: str, payload: Any) -> None:
    with st.expander(title, expanded=False):
        st.json(json.loads(json.dumps(payload, ensure_ascii=False, default=str)))


def tab_problem(state: WorkflowState | None) -> None:
    st.subheader("题意分析")
    if state:
        st.text_area("题目文本", value=state.problem_text, height=120, disabled=True)
        if state.problem_spec:
            spec = state.problem_spec
            cols = st.columns(4)
            cols[0].metric("子问题", len(spec.sub_questions or spec.subproblems))
            cols[1].metric("变量", len(spec.observed_variables) + len(spec.decision_variables) + len(spec.state_variables))
            cols[2].metric("约束", len(spec.constraints))
            cols[3].metric("指标", len(spec.metrics))
            _render_list_block("子问题", spec.sub_questions)
            if spec.subproblems:
                st.dataframe(pd.DataFrame(spec.subproblems), use_container_width=True, hide_index=True)
            variable_rows = []
            for role, values in (
                ("观测变量", spec.observed_variables),
                ("决策变量", spec.decision_variables),
                ("状态变量", spec.state_variables),
                ("参数", spec.parameters),
            ):
                variable_rows.extend({"角色": role, "名称": value} for value in values)
            if variable_rows:
                st.dataframe(pd.DataFrame(variable_rows), use_container_width=True, hide_index=True)
            _render_list_block("目标输出", spec.outputs)
            _render_list_block("约束条件", spec.constraints)
            _render_list_block("数据需求", spec.data_requirements)
            _compact_json_expander("结构化题意原文", {k: v for k, v in spec.__dict__.items() if v})
        analysis = state.notes.get("problem_analysis", "")
        if analysis:
            with st.expander("Agent 分析原文", expanded=False):
                st.markdown(analysis)
    else:
        st.info("尚未运行工作流")


def tab_models(state: WorkflowState | None) -> None:
    st.subheader("模型选择与决策")
    if state:
        selection = state.notes.get("model_selection", "")
        if selection:
            with st.expander("模型选择报告", expanded=False):
                st.markdown(selection)
        if state.model_decision:
            d = state.model_decision
            cols = st.columns(3)
            cols[0].metric("主模型", d.primary_model_id or "未定")
            cols[1].metric("基线模型", d.baseline_model_id or "未定")
            cols[2].metric("入选模型", len(d.selected_model_ids))
            if d.selected_model_ids:
                st.dataframe(
                    pd.DataFrame({"模型ID": d.selected_model_ids}),
                    use_container_width=True,
                    hide_index=True,
                )
            if d.rationale:
                st.markdown("**选择理由**")
                st.markdown(d.rationale)
            if d.comparison_plan:
                with st.expander("对比计划", expanded=False):
                    st.dataframe(pd.DataFrame(d.comparison_plan), use_container_width=True, hide_index=True)
        if state.model_critique:
            c = state.model_critique
            if c.issues:
                st.warning(f"批评意见: {len(c.issues)} 条问题")
                st.dataframe(pd.DataFrame(c.issues), use_container_width=True, hide_index=True)
            if c.risk_assessment:
                st.markdown("**风险评估**")
                st.markdown(c.risk_assessment)
    else:
        st.info("尚未运行工作流")


def tab_experiment(state: WorkflowState | None) -> None:
    st.subheader("实验方案")
    if state and state.experiment_plan:
        plan = state.experiment_plan
        cols = st.columns(4)
        cols[0].metric("指标数", len(plan.metrics))
        cols[1].metric("测试集比例", f"{plan.test_size:.0%}")
        cols[2].metric("交叉验证折数", plan.cv_folds)
        cols[3].metric("随机种子", len(plan.random_seeds))
        _render_list_block("评价指标", plan.metrics)
        summary_rows = [
            {"项目": "数据划分", "内容": plan.data_split},
            {"项目": "验证策略", "内容": plan.validation_strategy},
            {"项目": "敏感性分析", "内容": plan.sensitivity_plan},
            {"项目": "消融实验", "内容": plan.ablation_plan},
        ]
        st.dataframe(pd.DataFrame([row for row in summary_rows if row["内容"]]), use_container_width=True, hide_index=True)
        if plan.parameter_grid:
            with st.expander("参数网格", expanded=False):
                st.json(json.loads(json.dumps(plan.parameter_grid, ensure_ascii=False, default=str)))
        if plan.raw_plan:
            with st.expander("实验方案原文", expanded=False):
                st.markdown(plan.raw_plan)
    elif state:
        st.info("实验方案尚未生成（在模型确认后由实验设计Agent生成）")
    else:
        st.info("尚未运行工作流")


def tab_evidence(state: WorkflowState | None) -> None:
    st.subheader("证据映射")
    if state and state.claim_evidence_map:
        cem = state.claim_evidence_map
        cols = st.columns(3)
        cols[0].metric("证据覆盖率", f"{cem.coverage_pct:.0f}%")
        cols[1].metric("已映射声明", len(cem.claims))
        cols[2].metric("未映射声明", len(cem.unmapped_claims))
        if cem.coverage_pct < 90:
            st.warning("证据覆盖率偏低，正式导出前建议先补齐未映射声明。")
        if cem.claims:
            df = pd.DataFrame([
                {
                    "ID": c.claim_id,
                    "声明": c.claim[:120],
                    "来源": Path(c.source_file).name,
                    "模型": c.model_id,
                    "章节": ", ".join(c.paper_sections),
                    "计算": c.calculation[:80],
                }
                for c in cem.claims
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
        _render_list_block("未映射声明", cem.unmapped_claims)
    elif state and state.result_registry:
        reg = state.result_registry
        st.metric("结果条目", len(reg.entries))
    else:
        st.info("尚未生成证据映射")


def tab_paper(workspace) -> None:
    paper_path = workspace.paper_dir / "paper_draft.md"
    if paper_path.exists():
        st.markdown(paper_path.read_text(encoding="utf-8"))
    else:
        st.info("尚未生成论文草稿。")


def tab_tables(workspace) -> None:
    tables = sorted(workspace.tables_dir.glob("*.csv"))
    if not tables:
        st.info("尚未生成结果表。")
        return
    table = st.selectbox("选择结果表", options=tables, format_func=lambda p: p.name)
    try:
        df = pd.read_csv(table)
    except Exception as exc:
        st.error(f"无法读取 {table.name}: {exc}")
        return
    cols = st.columns(3)
    cols[0].metric("行数", len(df))
    cols[1].metric("列数", len(df.columns))
    cols[2].metric("缺失值", int(df.isna().sum().sum()))
    query = st.text_input("筛选文本", key=f"table_filter_{table.stem}")
    shown = df
    if query.strip():
        mask = df.astype(str).apply(lambda col: col.str.contains(query.strip(), case=False, na=False)).any(axis=1)
        shown = df[mask]
    st.dataframe(shown, use_container_width=True)
    st.download_button(
        "⬇️ 下载当前表",
        data=table.read_bytes(),
        file_name=table.name,
        mime="text/csv",
        key=f"download_table_{table.stem}",
    )


def tab_figures(workspace) -> None:
    figures = sorted(workspace.figures_dir.glob("*.png"))
    if not figures:
        st.info("尚未生成图表。")
        return
    for row_start in range(0, len(figures), 2):
        cols = st.columns(2)
        for col, figure in zip(cols, figures[row_start:row_start + 2]):
            with col:
                st.image(str(figure), caption=figure.stem, use_container_width=True)


def tab_review(workspace) -> None:
    review_path = workspace.paper_dir / "review_report.md"
    if review_path.exists():
        st.markdown(review_path.read_text(encoding="utf-8"))
    else:
        st.info("尚未生成审稿报告。")


def tab_export_page(formats: list[str], workspace, key_prefix: str = "export") -> None:
    st.caption("点击下方按钮下载已生成的文档。")
    if st.button("🔁 重新生成导出文件", use_container_width=False, key=f"{key_prefix}_regenerate"):
        results = export_paper(workspace, formats or list(EXPORT_LABELS.keys()))
        results.pop("_errors", None)
        st.success(f"已生成：{', '.join(results.keys()) or '无'}")
    md_path = workspace.paper_dir / "paper_draft.md"
    if md_path.exists():
        st.download_button("⬇️ 下载 Markdown (.md)", data=md_path.read_bytes(),
                           file_name="paper_draft.md", mime="text/markdown", key=f"{key_prefix}_dl_md")
    for fmt, suffix in (("docx", ".docx"), ("pdf", ".pdf"), ("latex", ".tex")):
        path = workspace.paper_dir / f"paper{suffix}"
        if path.exists():
            st.download_button(f"⬇️ 下载 {EXPORT_LABELS[fmt]}", data=path.read_bytes(),
                               file_name=path.name, mime=MIME_TYPES[fmt], key=f"{key_prefix}_dl_{fmt}")


def _format_phase_status(state: WorkflowState, phase: WorkflowPhase) -> str:
    status = state.get_phase_status(phase)
    return STATUS_LABELS.get(status.value, status.value)


def _downstream_phase_labels(phase: WorkflowPhase) -> str:
    labels = [p.label for p in WorkflowPhase if p.order > phase.order and p != WorkflowPhase.COMPLETE]
    if not labels:
        return "无后续阶段"
    if len(labels) > 6:
        return "、".join(labels[:6]) + f" 等 {len(labels)} 个阶段"
    return "、".join(labels)


def _render_code_plan(state: WorkflowState) -> None:
    if not state.code_plan:
        st.info("代码计划尚未生成。")
        return
    plan = state.code_plan
    if plan.files:
        st.dataframe(pd.DataFrame(plan.files), use_container_width=True, hide_index=True)
    if plan.function_specs:
        with st.expander("函数计划", expanded=False):
            st.json(json.loads(json.dumps(plan.function_specs, ensure_ascii=False, default=str)))
    if plan.model_calls:
        st.caption("模型调用")
        st.write("、".join(plan.model_calls))


def _render_paper_outline(state: WorkflowState) -> None:
    if not state.paper_outline:
        st.info("论文提纲尚未生成。")
        return
    rows = []
    for section in state.paper_outline.sections:
        rows.append({
            "章节": section.get("title", ""),
            "ID": section.get("id", ""),
            "证据": ", ".join(section.get("evidence_ids", []) or []),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.json(json.loads(json.dumps(state.paper_outline.__dict__, ensure_ascii=False, default=str)))


def _render_result_summary(state: WorkflowState) -> None:
    workspace = state.workspace
    cols = st.columns(3)
    cols[0].metric("结果表", len(list(workspace.tables_dir.glob("*.csv"))))
    cols[1].metric("图表", len(list(workspace.figures_dir.glob("*.png"))))
    cols[2].metric("错误数", len(state.errors))
    if state.result_registry and state.result_registry.entries:
        st.dataframe(pd.DataFrame(state.result_registry.entries), use_container_width=True, hide_index=True)
    else:
        st.info("结果登记尚未生成。")


def render_phase_focus(phase: WorkflowPhase, state: WorkflowState, formats: list[str]) -> None:
    """Render the content most relevant to the selected workflow phase."""
    icon = PHASE_ICONS.get(phase.value, "⚪")
    st.subheader(f"{icon} {phase.label}")
    cols = st.columns(3)
    cols[0].metric("阶段状态", _format_phase_status(state, phase))
    cols[1].metric("当前阶段", state.phase.label)
    cols[2].metric("运行目录", Path(getattr(state, "workspace", WORKSPACE).root).name)

    if state.get_phase_status(phase) == PhaseStatus.WAITING_FOR_USER:
        st.warning(f"该阶段正在等待确认。继续后会影响：{_downstream_phase_labels(phase)}。")

    phase_decisions = [d for d in state.decisions if d.get("phase") == phase.value]
    stored_feedback = state.notes.get(f"user_feedback_{phase.value}", "")
    if stored_feedback or phase_decisions:
        with st.expander("本阶段交互记录", expanded=False):
            if stored_feedback:
                st.markdown("**最近反馈**")
                st.write(stored_feedback)
            if phase_decisions:
                decision_rows = [
                    {
                        "时间": d.get("timestamp", "")[:19],
                        "操作": d.get("action", ""),
                        "操作者": d.get("operator", ""),
                        "备注": d.get("notes", ""),
                    }
                    for d in phase_decisions[-8:]
                ]
                st.dataframe(pd.DataFrame(decision_rows), use_container_width=True, hide_index=True)

    if phase == WorkflowPhase.PROBLEM_ANALYSIS:
        tab_problem(state)
    elif phase in {WorkflowPhase.MODEL_PROPOSAL, WorkflowPhase.MODEL_CRITIQUE, WorkflowPhase.MODEL_DECISION}:
        tab_models(state)
    elif phase == WorkflowPhase.EXPERIMENT_PLAN:
        tab_experiment(state)
    elif phase == WorkflowPhase.CODE_PLAN:
        _render_code_plan(state)
    elif phase in {WorkflowPhase.CODE_GENERATION, WorkflowPhase.EXECUTION, WorkflowPhase.RESULT_ANALYSIS}:
        _render_result_summary(state)
    elif phase == WorkflowPhase.EVIDENCE_MAPPING:
        tab_evidence(state)
    elif phase == WorkflowPhase.PAPER_OUTLINE:
        _render_paper_outline(state)
    elif phase == WorkflowPhase.SECTION_WRITING:
        tab_paper(state.workspace)
    elif phase in {
        WorkflowPhase.FACT_REVIEW,
        WorkflowPhase.MATH_REVIEW,
        WorkflowPhase.STRUCTURE_REVIEW,
        WorkflowPhase.LANGUAGE_REVIEW,
    }:
        tab_review(state.workspace)
    elif phase == WorkflowPhase.EXPORT:
        tab_export_page(formats, state.workspace, key_prefix=f"focus_{phase.value}")
    elif phase == WorkflowPhase.COMPLETE:
        st.success("工作流已完成，可在产物浏览中查看论文、结果表、图表和导出文件。")
    else:
        st.info("该阶段暂无专属视图。")


# ── checkpoint action bar ─────────────────────────────────────────────────
def render_action_bar(phase: WorkflowPhase, state: WorkflowState, wf: ModelingWorkflow) -> WorkflowState | None:
    """Render accept/modify/redo buttons for a checkpoint phase."""
    st.divider()
    st.subheader(f"📌 确认节点：{phase.label}")
    st.caption(f"继续后会影响：{_downstream_phase_labels(phase)}。")

    action = st.radio(
        "处理方式",
        ["接受并继续", "编辑后继续", "带反馈重跑", "创建备选分支"],
        horizontal=True,
        key=f"checkpoint_action_{phase.value}",
    )

    if action == "接受并继续":
        st.info("确认当前阶段结论，并从下一阶段继续执行。")
        if st.button("✅ 接受并继续", type="primary", use_container_width=True, key=f"btn_accept_{phase.value}"):
            wf.approve(phase)
            return run_with_progress(wf, wf.resume, "继续执行工作流...")

    elif action == "编辑后继续":
        st.info("进入该阶段的可编辑表单，保存后自动继续执行。")
        if st.button("✏️ 打开编辑表单", use_container_width=True, key=f"btn_modify_{phase.value}"):
            st.session_state["editing_phase"] = phase.value
            st.rerun()

    elif action == "带反馈重跑":
        feedback_key = f"redo_feedback_{phase.value}"
        templates = _feedback_templates_for_phase(phase)
        if templates:
            template_names = ["不使用模板", *templates.keys()]
            selected_template = st.selectbox(
                "反馈模板",
                options=template_names,
                key=f"feedback_template_{phase.value}",
            )
            if selected_template != "不使用模板":
                st.caption(templates[selected_template])
                if st.button("插入该模板", key=f"insert_feedback_template_{phase.value}"):
                    current = st.session_state.get(feedback_key, "").strip()
                    addition = templates[selected_template]
                    st.session_state[feedback_key] = f"{current}\n{addition}".strip() if current else addition
                    st.rerun()
        feedback = st.text_area(
            "反馈意见",
            key=feedback_key,
            placeholder="请描述需要调整的结论、约束、指标或写作要求...",
        )
        can_submit = bool(feedback.strip())
        if st.button(
            "🔄 记录反馈并从本阶段重跑",
            use_container_width=True,
            disabled=not can_submit,
            key=f"confirm_redo_{phase.value}",
        ):
            state.record_decision(phase, "feedback_rerun", operator="user", notes=feedback.strip())
            state.notes[f"user_feedback_{phase.value}"] = feedback.strip()
            return run_with_progress(wf, lambda: wf.rerun_from(phase), f"从{phase.label}重跑...")

    else:
        st.info("备选分支需要独立保存多套状态，当前版本暂未启用。")
        st.button("🔀 创建备选分支", use_container_width=True, disabled=True, key=f"btn_branch_{phase.value}")

    return None


def render_edit_form(phase: WorkflowPhase, state: WorkflowState) -> dict | None:
    """Render editable form for the given phase. Returns edits dict on submit."""
    edits = {}
    if phase == WorkflowPhase.MODEL_DECISION:
        st.subheader("编辑模型决策")
        primary = st.text_input("主模型ID", value=state.model_decision.primary_model_id if state.model_decision else "")
        baseline = st.text_input("基线模型ID", value=state.model_decision.baseline_model_id if state.model_decision else "")
        selected = st.text_input(
            "入选模型ID（逗号分隔）",
            value=", ".join(state.model_decision.selected_model_ids) if state.model_decision else "",
        )
        if primary:
            edits["primary_model_id"] = primary
        if baseline:
            edits["baseline_model_id"] = baseline
        if selected.strip():
            edits["selected_model_ids"] = [item.strip() for item in selected.split(",") if item.strip()]
        if st.button("💾 保存修改", key="save_model_edits"):
            return edits
    elif phase == WorkflowPhase.EXPERIMENT_PLAN:
        st.subheader("编辑实验方案")
        current_metrics = ", ".join(state.experiment_plan.metrics) if state.experiment_plan else ""
        metrics = st.text_input("评价指标（逗号分隔）", value=current_metrics)
        if metrics.strip():
            edits["metrics"] = [item.strip() for item in metrics.split(",") if item.strip()]
        note = st.text_area("补充说明", key="experiment_edit_note")
        if note.strip():
            edits["experiment_note"] = note.strip()
        if st.button("💾 保存实验方案", key="save_experiment_edits"):
            return edits
    elif phase == WorkflowPhase.PAPER_OUTLINE:
        st.subheader("编辑论文提纲")
        revised_sections: list[dict[str, Any]] = []
        if state.paper_outline:
            st.caption("取消勾选可移除章节；证据 ID 用英文逗号分隔。")
            for index, sec in enumerate(state.paper_outline.sections):
                section_id = str(sec.get("id") or f"section_{index + 1}")
                with st.expander(f"{index + 1}. {sec.get('title', section_id)}", expanded=False):
                    enabled = st.checkbox("保留该章节", value=True, key=f"outline_keep_{index}_{section_id}")
                    title = st.text_input(
                        "章节标题",
                        value=str(sec.get("title", "")),
                        key=f"outline_title_{index}_{section_id}",
                    )
                    evidence_ids = st.text_input(
                        "证据 ID",
                        value=", ".join(sec.get("evidence_ids", []) or []),
                        key=f"outline_evidence_{index}_{section_id}",
                    )
                    if enabled:
                        updated = dict(sec)
                        updated["id"] = section_id
                        updated["title"] = title.strip() or section_id
                        updated["evidence_ids"] = _split_csv_items(evidence_ids)
                        revised_sections.append(updated)
        else:
            st.info("当前尚无可编辑提纲，可先记录修改意见并继续。")

        new_count = st.number_input("追加空白章节数", min_value=0, max_value=5, value=0, step=1)
        for index in range(int(new_count)):
            section_id = st.text_input("新增章节 ID", value=f"custom_{index + 1}", key=f"outline_new_id_{index}")
            title = st.text_input("新增章节标题", key=f"outline_new_title_{index}")
            evidence_ids = st.text_input("新增章节证据 ID", key=f"outline_new_evidence_{index}")
            if section_id.strip() and title.strip():
                revised_sections.append({
                    "id": section_id.strip(),
                    "title": title.strip(),
                    "evidence_ids": _split_csv_items(evidence_ids),
                })
        if st.button("💾 保存提纲", key="save_outline"):
            if revised_sections:
                edits["sections"] = revised_sections
            return edits
    else:
        st.subheader("编辑确认意见")
        note = st.text_area(
            "修改说明",
            key=f"generic_edit_note_{phase.value}",
            placeholder="记录对当前阶段的调整要求，后续阶段会带着该意见继续执行。",
        )
        if note.strip():
            edits[f"{phase.value}_note"] = note.strip()
        if st.button("💾 保存意见并继续", key=f"save_generic_edits_{phase.value}"):
            return edits
    return None


# ── main ──────────────────────────────────────────────────────────────────
def main() -> None:
    st.title("📐 数学建模智能体 · 阶段工作台")
    st.caption("分阶段协作：建模 → 代码 → 论文，每个关键节点可暂停确认与修改。")

    # ── session state ──
    if "workflow" not in st.session_state:
        st.session_state["workflow"] = None
    if "state" not in st.session_state:
        st.session_state["state"] = None
    if "ran_legacy" not in st.session_state:
        st.session_state["ran_legacy"] = False
    if "editing_phase" not in st.session_state:
        st.session_state["editing_phase"] = None
    if "running_staged" not in st.session_state:
        st.session_state["running_staged"] = False

    # ── sidebar settings ──
    with st.sidebar:
        st.header("⚙️ 运行设置")
        use_llm = st.toggle("使用大模型（需配置 API Key）", value=False)
        st.caption("关闭时使用规则/模板驱动。")
        st.divider()
        st.subheader("📄 导出格式")
        formats = [fmt for fmt in ("docx", "pdf", "latex") if st.checkbox(EXPORT_LABELS[fmt], value=(fmt != "latex"))]
        st.divider()

        st.subheader("🚀 运行模式")
        mode = st.radio("选择模式", ["分阶段工作台（推荐）", "一键运行（传统）"], index=0)
        staged_mode = mode.startswith("分阶段")

        if staged_mode:
            if st.button("▶️ 开始分阶段运行", type="primary", use_container_width=True):
                st.session_state["running_staged"] = True
                st.rerun()

    # ── problem input (always visible) ──
    st.subheader("1. 题目与数据")
    problem_file = st.file_uploader(
        "上传题目文件（.txt/.md/.docx/.pdf，可选）",
        type=["txt", "md", "docx", "pdf"],
    )

    # derive problem_text from file or session state
    if "problem_text_value" not in st.session_state:
        st.session_state["problem_text_value"] = ""
    if "problem_text_from_file" not in st.session_state:
        st.session_state["problem_text_from_file"] = False
    if "problem_text_file_snapshot" not in st.session_state:
        st.session_state["problem_text_file_snapshot"] = ""
    if "problem_file_signature" not in st.session_state:
        st.session_state["problem_file_signature"] = ""
    if "problem_file_ignored_signature" not in st.session_state:
        st.session_state["problem_file_ignored_signature"] = ""

    if problem_file is not None:
        problem_payload = problem_file.getvalue()
        problem_signature = f"{problem_file.name}:{len(problem_payload)}"
        if len(problem_payload) > MAX_PROBLEM_UPLOAD_BYTES:
            st.error("题目文件超过 10 MB 限制。")
            st.stop()
        if st.session_state["problem_file_ignored_signature"] == problem_signature:
            pass
        elif st.session_state["problem_file_signature"] != problem_signature:
            if problem_file.name.lower().endswith(".docx"):
                try:
                    st.session_state["problem_text_value"] = read_docx_text(BytesIO(problem_payload))
                except Exception as exc:
                    st.error(f"无法解析 Word 文件：{exc}")
            elif problem_file.name.lower().endswith(".pdf"):
                try:
                    st.session_state["problem_text_value"] = read_pdf_text(BytesIO(problem_payload))
                except Exception as exc:
                    st.error(f"无法解析 PDF 文件：{exc}")
            else:
                st.session_state["problem_text_value"] = problem_payload.decode("utf-8", errors="ignore")
            st.session_state["problem_text_file_snapshot"] = st.session_state["problem_text_value"]
            st.session_state["problem_text_from_file"] = True
            st.session_state["problem_file_signature"] = problem_signature
            st.session_state["problem_file_ignored_signature"] = ""

    if st.session_state["problem_text_from_file"]:
        st.caption("✅ 已从文件读取题目内容，下方文本为可编辑副本。")
        col_clear, col_restore = st.columns(2)
        if col_clear.button("🗑 清除文件内容，改为手动输入", key="clear_problem_file"):
            st.session_state["problem_file_ignored_signature"] = st.session_state["problem_file_signature"]
            st.session_state["problem_text_value"] = ""
            st.session_state["problem_text_from_file"] = False
            st.session_state["problem_text_file_snapshot"] = ""
            st.rerun()
        if col_restore.button("↩️ 恢复文件原文", key="restore_problem_file"):
            st.session_state["problem_text_value"] = st.session_state["problem_text_file_snapshot"]
            st.rerun()
    problem_text = st.text_area(
        "题目文本",
        value=st.session_state["problem_text_value"],
        height=120,
        placeholder="请粘贴或输入数学建模题目……",
        key="problem_text_input",
    )
    st.session_state["problem_text_value"] = problem_text

    data_files = st.file_uploader(
        "上传数据文件（csv/tsv/xlsx/xls，可多选）",
        type=["csv", "tsv", "xlsx", "xls"],
        accept_multiple_files=True,
    )
    render_data_upload_precheck(data_files)

    # ── legacy mode ──
    if not staged_mode:
        if st.button("🚀 运行工作流", type="primary", use_container_width=True, key="btn_legacy_run"):
            if not problem_text.strip():
                st.error("请先输入或上传题目文本。")
            else:
                data_paths = save_uploaded_data(data_files)
                wf = ModelingWorkflow(use_llm=use_llm, export_formats=formats or None)
                progress_panel = st.empty()
                tracker = make_progress_tracker([a.name for a in wf.agents])
                render_runtime_progress(progress_panel, tracker)
                cb = make_progress_callback(progress_panel, tracker)
                with st.spinner("工作流运行中……"):
                    try:
                        state = wf.run(problem_text, data_paths)
                        tracker["elapsed"] = time.perf_counter() - tracker["started_at"]
                        tracker["current"] = "运行完成"
                        render_runtime_progress(progress_panel, tracker)
                        st.session_state["ran_legacy"] = True
                        st.session_state["state"] = state
                        st.success("工作流运行完成。")
                        show_status(state)
                    except Exception as exc:
                        st.session_state["ran_legacy"] = False
                        st.exception(exc)

        if st.session_state.get("ran_legacy"):
            st.divider()
            st.subheader("3. 结果")
            tabs = st.tabs(["📝 论文", "📊 结果表", "📈 图表", "🔍 审稿", "📥 导出下载"])
            with tabs[0]: tab_paper(WORKSPACE)
            with tabs[1]: tab_tables(WORKSPACE)
            with tabs[2]: tab_figures(WORKSPACE)
            with tabs[3]: tab_review(WORKSPACE)
            with tabs[4]: tab_export_page(formats, WORKSPACE, key_prefix="legacy_export")

    # ── staged mode ──
    else:
        state = st.session_state.get("state")
        wf = st.session_state.get("workflow")

        # phase navigation
        selected_phase = render_phase_nav(state)

        if st.session_state.get("running_staged") and state is None:
            if not problem_text.strip():
                st.error("请先输入或上传题目文本。")
                st.session_state["running_staged"] = False
            else:
                data_paths = save_uploaded_data(data_files)
                wf = ModelingWorkflow(
                    use_llm=use_llm,
                    export_formats=formats or None,
                    run_workspace=True,
                    progress_callback=None,
                )
                st.session_state["workflow"] = wf

                try:
                    state = run_with_progress(
                        wf,
                        lambda: wf.run_until(WorkflowPhase.MODEL_DECISION, problem_text, data_paths, auto_approve=False),
                        "初始化工作流...",
                    )
                    st.session_state["state"] = state
                    st.session_state["selected_phase"] = (_find_paused_phase(state) or state.phase).value
                    st.rerun()
                except Exception as exc:
                    st.exception(exc)
                    st.session_state["running_staged"] = False

        # main content area for staged mode
        if state is not None:
            col_main, col_side = st.columns([2, 1])

            with col_main:
                focus_phase = selected_phase or _default_focus_phase(state)
                render_phase_focus(focus_phase, state, formats)

                with st.expander("完整产物浏览", expanded=False):
                    tabs = st.tabs(["📄 论文", "📊 结果表", "📈 图表", "🔍 审稿", "📥 导出"])
                    with tabs[0]: tab_paper(state.workspace)
                    with tabs[1]: tab_tables(state.workspace)
                    with tabs[2]: tab_figures(state.workspace)
                    with tabs[3]: tab_review(state.workspace)
                    with tabs[4]: tab_export_page(formats, state.workspace, key_prefix="staged_browser_export")

                # ── checkpoint action bar ──
                paused_phase = None
                for p in WorkflowPhase:
                    if state.get_phase_status(p) == PhaseStatus.WAITING_FOR_USER:
                        paused_phase = p
                        break

                if paused_phase and wf:
                    # editing mode
                    if st.session_state.get("editing_phase") == paused_phase.value:
                        edits = render_edit_form(paused_phase, state)
                        if edits:
                            wf.approve(paused_phase, edits)
                            state = run_with_progress(wf, wf.resume, "保存修改并继续执行...")
                            st.session_state["state"] = state
                            st.session_state["editing_phase"] = None
                            st.session_state["selected_phase"] = (_find_paused_phase(state) or state.phase).value
                            st.rerun()
                        if st.button("取消编辑", key="cancel_edit"):
                            st.session_state["editing_phase"] = None
                            st.rerun()
                    else:
                        next_state = render_action_bar(paused_phase, state, wf)
                        if next_state is not None:
                            st.session_state["state"] = next_state
                            st.session_state["selected_phase"] = (_find_paused_phase(next_state) or next_state.phase).value
                            st.rerun()

                # continue button for non-paused states
                elif state.phase != WorkflowPhase.COMPLETE and wf and not paused_phase:
                    st.divider()
                    if st.button("▶️ 继续执行", type="primary", key="btn_continue"):
                        try:
                            state = run_with_progress(
                                wf,
                                lambda: wf.run_until(WorkflowPhase.COMPLETE, auto_approve=False),
                                "执行中...",
                            )
                            st.session_state["state"] = state
                            st.session_state["selected_phase"] = (_find_paused_phase(state) or state.phase).value
                            st.rerun()
                        except Exception as exc:
                            st.exception(exc)

                # completion
                if state.phase == WorkflowPhase.COMPLETE or state.get_phase_status(WorkflowPhase.COMPLETE) == PhaseStatus.COMPLETED:
                    st.success("🎉 工作流已完成！可在各标签页查看结果。")

            with col_side:
                show_status(state)

                # decisions log
                if state.decisions:
                    st.subheader("📋 决策日志")
                    for d in state.decisions[-5:]:
                        phase_label = WorkflowPhase(d.get("phase", "")).label if d.get("phase") else ""
                        st.caption(f"{d.get('timestamp','')[:19]} | {phase_label} | {d.get('action','')}")

                # rerun controls
                with st.expander("🔧 高级：返工控制", expanded=False):
                    rerun_target = st.selectbox(
                        "从哪个阶段重新执行？",
                        options=[p.value for p in WorkflowPhase if p != WorkflowPhase.COMPLETE],
                        format_func=lambda v: WorkflowPhase(v).label,
                    )
                    if st.button("🔄 从该阶段重跑", key="btn_rerun"):
                        if wf:
                            state = run_with_progress(
                                wf,
                                lambda: wf.rerun_from(WorkflowPhase(rerun_target)),
                                f"从{WorkflowPhase(rerun_target).label}重跑...",
                            )
                            st.session_state["state"] = state
                            st.session_state["selected_phase"] = (_find_paused_phase(state) or state.phase).value
                            st.rerun()

                    revision_targets = _revision_target_options(state)
                    artifact_target = st.selectbox(
                        "修订产物",
                        options=list(revision_targets.keys()),
                        format_func=lambda value: revision_targets.get(value, value),
                    )
                    artifact_feedback = st.text_area("修订意见", key="artifact_feedback")
                    if st.button("🔧 修订产物", key="btn_revise_artifact"):
                        if wf and artifact_feedback:
                            state = run_with_progress(
                                wf,
                                lambda: wf.revise_artifact(artifact_target, artifact_feedback),
                                "修订产物...",
                            )
                            st.session_state["state"] = state
                            st.session_state["selected_phase"] = (_find_paused_phase(state) or state.phase).value
                            st.rerun()


def show_status(state) -> None:
    cols = st.columns(4)
    exec_status = state.notes.get("execution_status", "未知")
    cols[0].metric("执行状态", "成功" if exec_status == "success" else exec_status)
    cols[1].metric("数据文件", len(state.data_files))
    workspace = getattr(state, "workspace", WORKSPACE)
    table_count = len(list(workspace.tables_dir.glob("*.csv")))
    figure_count = len(list(workspace.figures_dir.glob("*.png")))
    cols[2].metric("结果表", table_count)
    cols[3].metric("图表", figure_count)
    st.caption(f"运行目录：{workspace.root}")
    llm_status = state.notes.get("llm_status", "")
    if llm_status and llm_status != "enabled":
        st.info(f"LLM 状态：{llm_status}")
    if state.notes.get("export_errors"):
        st.warning(f"部分导出失败：{state.notes['export_errors']}")
    if state.errors:
        st.warning(format_error_for_display(state.errors[-1]))
        with st.expander("错误明细", expanded=False):
            error_rows = [{
                "agent": e.get("agent", ""),
                "category": e.get("category", "unknown"),
                "exception_type": e.get("exception_type", "RecordedError"),
                "recoverable": e.get("recoverable", ""),
                "message": e.get("message", ""),
            } for e in state.errors]
            st.dataframe(pd.DataFrame(error_rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
