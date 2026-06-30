from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from collections.abc import Mapping
from typing import Any
from pathlib import Path

from app.config import WorkspaceConfig

# ---- well-known state keys (use these constants instead of raw strings) ----
# notes keys
K_SELECTED_MODEL_IDS = "selected_model_ids"
K_PROBLEM_ANALYSIS = "problem_analysis"
K_MODELING_PLAN = "modeling_plan"
K_MODEL_SELECTION = "model_selection"
K_EXECUTION_STATUS = "execution_status"
K_EXECUTION_ATTEMPTS = "execution_attempts"
K_EXECUTION_ERROR = "execution_error"
K_RESULT_ANALYSIS = "result_analysis"
K_REVIEW_REPORT = "review_report"
K_LLM_STATUS = "llm_status"
K_PROBLEM_TYPE = "problem_type"
K_LAST_REPAIR_APPLIED = "last_repair_applied"
K_LAST_REPAIR_NOTE = "last_repair_note"
K_WRITING_MODE = "writing_agent_mode"
K_PAPER_QUALITY_SCORE = "paper_quality_score"
K_PAPER_QUALITY_REPORT = "paper_quality_report"
K_PAPER_SOLUTION_SCORE = "paper_solution_score"
K_PAPER_EVIDENCE_SCORE = "paper_evidence_score"
K_PAPER_STRUCTURE_SCORE = "paper_structure_score"
K_PAPER_EXPORT_SCORE = "paper_export_score"
K_PREWRITING_GATE_STATUS = "prewriting_gate_status"
K_PREWRITING_GATE_REPORT = "prewriting_gate_report"
K_LLM_FAILURE_KIND = "llm_failure_kind"
K_EXPORT_QUALITY_GATE = "export_quality_gate"
K_EXPORT_BLOCKING_ISSUES = "export_blocking_issues"
K_EXPORT_ERRORS = "export_errors"
K_EXPORT_DOCX_LAYOUT_GATE = "export_docx_layout_gate"
K_EXPORT_PDF_LAYOUT_GATE = "export_pdf_layout_gate"
K_TASK_TRACEABILITY_GATE = "task_traceability_gate"
K_TASK_TRACEABILITY_COVERAGE_PCT = "task_traceability_coverage_pct"
K_TASK_TRACEABILITY_BLOCKING_ISSUES = "task_traceability_blocking_issues"
K_STRONG_BASELINE_GATE = "strong_baseline_gate"
K_STRONG_BASELINE_ISSUES = "strong_baseline_issues"
K_INNOVATION_EVIDENCE_GATE = "innovation_evidence_gate"
K_INNOVATION_EVIDENCE_ISSUES = "innovation_evidence_issues"
K_PAPER_EVIDENCE_GATE = "paper_evidence_gate"
K_PAPER_EVIDENCE_ISSUES = "paper_evidence_issues"
K_AUTO_REWORK_STATUS = "auto_rework_status"
K_AUTO_REWORK_RERUN_FROM_PHASE = "auto_rework_rerun_from_phase"
K_AUTO_REWORK_REPAIR_HINTS = "auto_rework_repair_hints"
K_AUTO_REWORK_REPAIR_BRIEF = "auto_rework_repair_brief"
QUALITY_GATE_NOTE_KEYS = (
    K_EXPORT_QUALITY_GATE,
    K_EXPORT_DOCX_LAYOUT_GATE,
    K_TASK_TRACEABILITY_GATE,
    K_STRONG_BASELINE_GATE,
    K_INNOVATION_EVIDENCE_GATE,
    K_PAPER_EVIDENCE_GATE,
    K_EXPORT_PDF_LAYOUT_GATE,
    K_PREWRITING_GATE_STATUS,
)
# artifacts keys
A_CODE = "code"
A_EXECUTION_LOG = "execution_log"
A_PAPER = "paper"
A_PAPER_QUALITY = "paper_quality"
A_REVIEW = "review"
A_MODEL_EXECUTION_FEEDBACK = "model_execution_feedback"
# structured artifact keys (new)
A_PROBLEM_SPEC = "problem_spec"
A_DATA_PROFILE = "data_profile"
A_MODEL_PROPOSAL = "model_proposal"
A_MODEL_CRITIQUE = "model_critique"
A_MODEL_DECISION = "model_decision"
A_FORMULATION_SPEC = "formulation_spec"
A_TASK_DELIVERABLE_SPEC = "task_deliverable_spec"
A_EXPERIMENT_PLAN = "experiment_plan"
A_EXPERIMENT_REPORT = "experiment_report"
A_CODE_PLAN = "code_plan"
A_RESULT_REGISTRY = "result_registry"
A_CLAIM_EVIDENCE_MAP = "claim_evidence_map"
A_TRACEABILITY_REPORT = "traceability_report"
A_TASK_TRACEABILITY_REPORT = "task_traceability_report"
A_INNOVATION_EVIDENCE_REPORT = "innovation_evidence_report"
A_PAPER_EVIDENCE_AUDIT = "paper_evidence_audit"
A_AUTO_REWORK_PLAN = "auto_rework_plan"
A_AUTO_REWORK_REPORT = "auto_rework_report"
A_AUTO_REWORK_REPORT_MD = "auto_rework_report_md"
A_PAPER_DOCX_LAYOUT_REPORT = "paper_docx_layout_report"
A_PAPER_PDF_LAYOUT_REPORT = "paper_pdf_layout_report"
A_WORKFLOW_GATE_SUMMARY = "workflow_gate_summary"
A_WORKFLOW_GATE_SUMMARY_MD = "workflow_gate_summary_md"
A_PAPER_OUTLINE = "paper_outline"
A_SECTION_DRAFT = "section_draft"
A_REVIEW_FINDINGS = "review_findings"
A_DECISIONS_LOG = "decisions_log"


# ---- workflow phases ----
class WorkflowPhase(str, Enum):
    """Ordered phases of the modelling → code → paper pipeline."""
    PROBLEM_ANALYSIS = "problem_analysis"
    MODEL_PROPOSAL = "model_proposal"
    MODEL_CRITIQUE = "model_critique"
    MODEL_DECISION = "model_decision"
    EXPERIMENT_PLAN = "experiment_plan"
    CODE_PLAN = "code_plan"
    CODE_GENERATION = "code_generation"
    EXECUTION = "execution"
    RESULT_ANALYSIS = "result_analysis"
    EVIDENCE_MAPPING = "evidence_mapping"
    PAPER_OUTLINE = "paper_outline"
    SECTION_WRITING = "section_writing"
    FACT_REVIEW = "fact_review"
    MATH_REVIEW = "math_review"
    STRUCTURE_REVIEW = "structure_review"
    LANGUAGE_REVIEW = "language_review"
    EXPORT = "export"
    COMPLETE = "complete"

    @property
    def label(self) -> str:
        _LABELS: dict[str, str] = {
            "problem_analysis": "题意分析",
            "model_proposal": "候选模型讨论",
            "model_critique": "模型批评",
            "model_decision": "模型决策",
            "experiment_plan": "实验方案",
            "code_plan": "代码计划",
            "code_generation": "代码生成",
            "execution": "执行与修复",
            "result_analysis": "结果分析",
            "evidence_mapping": "证据映射",
            "paper_outline": "论文提纲",
            "section_writing": "分章节写作",
            "fact_review": "事实审稿",
            "math_review": "数学审稿",
            "structure_review": "结构审稿",
            "language_review": "语言审稿",
            "export": "文档导出",
            "complete": "完成",
        }
        return _LABELS.get(self.value, self.value)

    @property
    def order(self) -> int:
        return list(WorkflowPhase).index(self)

    @classmethod
    def from_label(cls, label: str) -> "WorkflowPhase | None":
        for phase in cls:
            if phase.label == label:
                return phase
        return None


class PhaseStatus(str, Enum):
    RUNNING = "running"
    WAITING_FOR_USER = "waiting_for_user"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


# ---- structured artifact dataclasses ----
@dataclass
class ProblemSpec:
    """Structured problem analysis result."""
    sub_questions: list[str] = field(default_factory=list)
    subproblems: list[dict[str, Any]] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    observed_variables: list[str] = field(default_factory=list)
    decision_variables: list[str] = field(default_factory=list)
    state_variables: list[str] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    time_scale: str = ""
    spatial_scale: str = ""
    uncertainty_sources: list[str] = field(default_factory=list)
    data_requirements: list[str] = field(default_factory=list)
    task_dependencies: list[dict[str, str]] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    raw_analysis: str = ""


@dataclass
class TaskDeliverableSpec:
    """Expected outputs and evidence contract for one decomposed task."""

    task_id: str = ""
    task_type: str = "exploration"
    objective: str = ""
    required_outputs: list[str] = field(default_factory=list)
    required_models: list[str] = field(default_factory=list)
    required_tables: list[str] = field(default_factory=list)
    required_figures: list[str] = field(default_factory=list)
    success_metrics: list[str] = field(default_factory=list)
    evidence_requirements: list[str] = field(default_factory=list)
    blocking_conditions: list[str] = field(default_factory=list)
    status: str = "planned"


@dataclass
class DataProfile:
    """Structured data summary."""
    file_count: int = 0
    total_rows: int = 0
    columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    missing_summary: dict[str, int] = field(default_factory=dict)


@dataclass
class ModelProposal:
    """Candidate models proposed by ModelingAgent."""
    candidates: list[dict[str, Any]] = field(default_factory=list)
    raw_plan: str = ""


@dataclass
class ModelCritique:
    """Critique of model proposals."""
    issues: list[dict[str, Any]] = field(default_factory=list)
    risk_assessment: str = ""
    data_condition_checks: dict[str, bool] = field(default_factory=dict)


@dataclass
class ModelDecision:
    """Final model selection after critique."""
    primary_model_id: str = ""
    baseline_model_id: str = ""
    selected_model_ids: list[str] = field(default_factory=list)
    rationale: str = ""
    comparison_plan: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class FormulationSpec:
    """Machine-readable mathematical formulation and multi-stage pipeline."""

    variables: list[dict[str, Any]] = field(default_factory=list)
    parameters: list[dict[str, Any]] = field(default_factory=list)
    objectives: list[dict[str, Any]] = field(default_factory=list)
    constraints: list[dict[str, Any]] = field(default_factory=list)
    stages: list[dict[str, Any]] = field(default_factory=list)
    dependencies: list[dict[str, str]] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    validation_issues: list[str] = field(default_factory=list)


@dataclass
class ExperimentPlan:
    """Experiment design — metrics, splits, sensitivity."""
    metrics: list[str] = field(default_factory=list)
    data_split: str = ""
    validation_strategy: str = ""
    test_size: float = 0.2
    cv_folds: int = 5
    random_seeds: list[int] = field(default_factory=lambda: [42])
    parameter_grid: dict[str, list[Any]] = field(default_factory=dict)
    sensitivity_plan: str = ""
    ablation_plan: str = ""
    raw_plan: str = ""


@dataclass
class CodePlan:
    """File/function plan before code generation."""
    files: list[dict[str, Any]] = field(default_factory=list)
    function_specs: list[dict[str, Any]] = field(default_factory=list)
    model_calls: list[str] = field(default_factory=list)


@dataclass
class ResultRegistry:
    """Catalogue of every output file produced by execution."""
    entries: list[dict[str, Any]] = field(default_factory=list)
    source_path: str = ""
    evidence_records: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = "2.0"


@dataclass
class ClaimEvidence:
    """One claim-to-data binding."""
    claim_id: str = ""
    claim: str = ""
    model_id: str = ""
    source_file: str = ""
    source_rows: list[int] = field(default_factory=list)
    calculation: str = ""
    paper_sections: list[str] = field(default_factory=list)


@dataclass
class ClaimEvidenceMap:
    """Full evidence map for the paper."""
    claims: list[ClaimEvidence] = field(default_factory=list)
    coverage_pct: float = 0.0
    unmapped_claims: list[str] = field(default_factory=list)

    def find_by_section(self, section: str) -> list[ClaimEvidence]:
        return [c for c in self.claims if section in c.paper_sections]


@dataclass
class PaperOutline:
    """Structured paper outline with evidence assignment."""
    sections: list[dict[str, Any]] = field(default_factory=list)
    total_sections: int = 0


@dataclass
class ReviewFindings:
    """Multi-dimensional review result."""
    reviewer: str = ""
    score: int = 0
    issues: list[dict[str, Any]] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    raw_report: str = ""


class ErrorCategory(str, Enum):
    INPUT = "input"
    DATA = "data"
    IO = "io"
    TIMEOUT = "timeout"
    DEPENDENCY = "dependency"
    VALIDATION = "validation"
    AGENT = "agent"
    UNKNOWN = "unknown"


class WorkflowError(Exception):
    """Workflow-level exception with stable category metadata."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory | str = ErrorCategory.UNKNOWN,
        recoverable: bool = True,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.category = normalize_error_category(category)
        self.recoverable = recoverable
        self.details = dict(details or {})


_DATA_EXCEPTION_NAMES = {"ParserError", "EmptyDataError", "UnicodeDecodeError"}


def normalize_error_category(category: ErrorCategory | str | None) -> str:
    if category is None:
        return ErrorCategory.UNKNOWN.value
    if isinstance(category, ErrorCategory):
        return category.value
    value = str(category).strip().lower()
    known = {item.value for item in ErrorCategory}
    return value if value in known else ErrorCategory.UNKNOWN.value


def classify_exception(exc: BaseException | None) -> str:
    """Map common exception types to stable workflow error categories."""
    if exc is None:
        return ErrorCategory.UNKNOWN.value
    if isinstance(exc, WorkflowError):
        return exc.category

    exc_name = type(exc).__name__
    if exc_name in _DATA_EXCEPTION_NAMES:
        return ErrorCategory.DATA.value
    if isinstance(exc, TimeoutError):
        return ErrorCategory.TIMEOUT.value
    if isinstance(exc, (FileNotFoundError, PermissionError, OSError)):
        return ErrorCategory.IO.value
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return ErrorCategory.DEPENDENCY.value
    if isinstance(exc, (ValueError, TypeError, KeyError, IndexError, AssertionError)):
        return ErrorCategory.VALIDATION.value
    if isinstance(exc, RuntimeError):
        return ErrorCategory.AGENT.value
    return ErrorCategory.UNKNOWN.value


def _json_safe_details(details: Mapping[str, Any] | None) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in (details or {}).items():
        if value is None or isinstance(value, (str, int, float, bool)):
            safe[str(key)] = value
        else:
            safe[str(key)] = str(value)
    return safe


def format_error_for_display(error: Mapping[str, Any]) -> str:
    category = str(error.get("category") or ErrorCategory.UNKNOWN.value)
    exception_type = str(error.get("exception_type") or "RecordedError")
    agent = str(error.get("agent") or "agent")
    message = str(error.get("message") or "")
    return f"[{category}/{exception_type}] {agent}: {message}"


def _restore_structured(state: WorkflowState, data: dict[str, Any], field_name: str, cls: type) -> None:
    """Populate a structured field on `state` from `data` dict, if present."""
    import dataclasses
    raw = data.get(field_name)
    if raw is None or not isinstance(raw, dict):
        return
    try:
        field_types = {f.name: f.type for f in dataclasses.fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            if key in field_types:
                kwargs[key] = value
        setattr(state, field_name, cls(**kwargs))
    except Exception:
        pass


@dataclass
class WorkflowState:
    problem_text: str
    data_files: list[Path]
    workspace: WorkspaceConfig
    llm: Any | None = None
    artifacts: dict[str, Path] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    durations: dict[str, float] = field(default_factory=dict)
    # ---- new structured fields ----
    phase: WorkflowPhase = WorkflowPhase.PROBLEM_ANALYSIS
    phase_status: dict[str, str] = field(default_factory=dict)  # phase.value → PhaseStatus.value
    run_id: str = ""
    # structured artifacts
    problem_spec: ProblemSpec | None = None
    task_deliverable_specs: list[TaskDeliverableSpec] = field(default_factory=list)
    data_profile: DataProfile | None = None
    model_proposal: ModelProposal | None = None
    model_critique: ModelCritique | None = None
    model_decision: ModelDecision | None = None
    formulation_spec: FormulationSpec | None = None
    experiment_plan: ExperimentPlan | None = None
    code_plan: CodePlan | None = None
    result_registry: ResultRegistry | None = None
    claim_evidence_map: ClaimEvidenceMap | None = None
    paper_outline: PaperOutline | None = None
    review_findings: ReviewFindings | None = None
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.phase_status:
            self.phase_status = {}
        if not self.run_id:
            self.run_id = self._derive_run_id()

    def _derive_run_id(self) -> str:
        from datetime import datetime, timezone
        from uuid import uuid4
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{ts}_{uuid4().hex[:8]}"

    def set_phase_status(self, phase: WorkflowPhase, status: PhaseStatus) -> None:
        self.phase_status[phase.value] = status.value

    def get_phase_status(self, phase: WorkflowPhase) -> PhaseStatus:
        raw = self.phase_status.get(phase.value, "")
        try:
            return PhaseStatus(raw)
        except ValueError:
            return PhaseStatus.RUNNING

    def is_phase_completed(self, phase: WorkflowPhase) -> bool:
        return self.get_phase_status(phase) == PhaseStatus.COMPLETED

    def next_pending_phase(self) -> WorkflowPhase | None:
        for p in WorkflowPhase:
            status = self.get_phase_status(p)
            if status not in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED, PhaseStatus.APPROVED):
                return p
        return WorkflowPhase.COMPLETE

    def record_decision(
        self,
        phase: WorkflowPhase,
        action: str,
        operator: str = "user",
        notes: str = "",
    ) -> None:
        from datetime import datetime, timezone
        self.decisions.append({
            "phase": phase.value,
            "action": action,
            "operator": operator,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
            "version": len(self.decisions) + 1,
        })

    def to_json(self) -> dict[str, Any]:
        """Serialise to JSON-safe dict for persistence."""
        import dataclasses
        payload: dict[str, Any] = {
            "problem_text": self.problem_text,
            "data_files": [str(p) for p in self.data_files],
            "workspace_root": str(self.workspace.root),
            "phase": self.phase.value,
            "phase_status": dict(self.phase_status),
            "run_id": self.run_id,
            "artifacts": {k: str(v) for k, v in self.artifacts.items()},
            "notes": dict(self.notes),
            "errors": list(self.errors),
            "durations": dict(self.durations),
            "decisions": list(self.decisions),
        }
        for field_name in (
            "problem_spec", "data_profile", "model_proposal", "model_critique",
            "model_decision", "formulation_spec", "experiment_plan", "code_plan",
            "result_registry", "claim_evidence_map", "paper_outline", "review_findings",
        ):
            obj = getattr(self, field_name, None)
            if obj is not None:
                payload[field_name] = dataclasses.asdict(obj)  # type: ignore[arg-type]
        if self.task_deliverable_specs:
            payload["task_deliverable_specs"] = [
                dataclasses.asdict(item)
                for item in self.task_deliverable_specs
            ]
        return payload

    @classmethod
    def from_json(cls, data: dict[str, Any], workspace: WorkspaceConfig | None = None) -> "WorkflowState":
        """Deserialise from a dict saved by to_json."""
        from pathlib import Path as _Path
        ws = workspace or WorkspaceConfig.from_root(_Path(data["workspace_root"]))
        state = cls(
            problem_text=data.get("problem_text", ""),
            data_files=[_Path(p) for p in data.get("data_files", [])],
            workspace=ws,
            phase=WorkflowPhase(data.get("phase", "problem_analysis")),
            run_id=data.get("run_id", ""),
        )
        state.phase_status = dict(data.get("phase_status", {}))
        state.artifacts = {k: _Path(v) for k, v in data.get("artifacts", {}).items()}
        state.notes = dict(data.get("notes", {}))
        state.errors = list(data.get("errors", []))
        state.durations = dict(data.get("durations", {}))
        state.decisions = list(data.get("decisions", []))
        _restore_structured(state, data, "problem_spec", ProblemSpec)
        raw_deliverables = data.get("task_deliverable_specs")
        if isinstance(raw_deliverables, list):
            state.task_deliverable_specs = [
                TaskDeliverableSpec(**item)
                for item in raw_deliverables
                if isinstance(item, dict)
            ]
        _restore_structured(state, data, "data_profile", DataProfile)
        _restore_structured(state, data, "model_proposal", ModelProposal)
        _restore_structured(state, data, "model_critique", ModelCritique)
        _restore_structured(state, data, "model_decision", ModelDecision)
        _restore_structured(state, data, "formulation_spec", FormulationSpec)
        _restore_structured(state, data, "experiment_plan", ExperimentPlan)
        _restore_structured(state, data, "code_plan", CodePlan)
        _restore_structured(state, data, "result_registry", ResultRegistry)
        _restore_structured(state, data, "claim_evidence_map", ClaimEvidenceMap)
        _restore_structured(state, data, "paper_outline", PaperOutline)
        _restore_structured(state, data, "review_findings", ReviewFindings)
        return state

    def add_error(
        self,
        agent: str,
        message: str | BaseException,
        recoverable: bool | None = None,
        category: ErrorCategory | str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a non-fatal error for later diagnostics.

        Existing string callers are supported. Passing an exception adds a
        stable category and exception type for logs, diagnostics, and UI.
        """
        exc = message if isinstance(message, BaseException) else None
        error_category = normalize_error_category(category) if category else classify_exception(exc)
        exception_type = type(exc).__name__ if exc else "RecordedError"
        recoverable_value = recoverable
        if recoverable_value is None:
            recoverable_value = exc.recoverable if isinstance(exc, WorkflowError) else True

        error_details = details
        if error_details is None and isinstance(exc, WorkflowError):
            error_details = exc.details

        record: dict[str, Any] = {
            "agent": agent,
            "message": str(message),
            "recoverable": str(recoverable_value),
            "category": error_category,
            "exception_type": exception_type,
        }
        safe_details = _json_safe_details(error_details)
        if safe_details:
            record["details"] = safe_details
        self.errors.append(record)
        logging.getLogger("mma").warning(
            "[%s] %s (category=%s, type=%s, recoverable=%s)",
            agent,
            message,
            error_category,
            exception_type,
            recoverable_value,
        )
        return record

    def set_duration(self, agent: str, seconds: float) -> None:
        self.durations[agent] = round(seconds, 4)


class Agent:
    name = "agent"

    def run(self, state: WorkflowState) -> WorkflowState:
        raise NotImplementedError
