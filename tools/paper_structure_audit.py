from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PaperStructureAudit:
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    metrics: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "abstract": ("abstract", "摘要"),
    "keywords": ("keywords", "key words", "关键词", "關鍵詞"),
    "problem_restatement": ("problem restatement", "problem statement", "问题重述", "問題重述"),
    "problem_analysis": ("problem analysis", "问题分析", "問題分析"),
    "assumptions": ("assumptions", "model assumptions", "模型假设", "模型假設"),
    "notation": ("notation", "symbols", "symbol table", "符号说明", "符號說明"),
    "model": ("model", "model formulation", "模型建立", "模型构建", "模型求解"),
    "results": ("results", "result analysis", "结果分析", "結果分析"),
    "validation": ("validation", "verification", "model validation", "模型检验", "模型檢驗"),
    "sensitivity": ("sensitivity", "error analysis", "robustness", "灵敏度", "敏感性", "误差分析"),
    "evaluation": ("evaluation", "discussion", "generalization", "模型评价", "模型評價", "推广"),
    "conclusion": ("conclusion", "conclusions", "结论", "結論"),
    "references": ("references", "bibliography", "参考文献", "參考文獻"),
    "appendix": ("appendix", "附录", "附錄"),
}

HIGH_VALUE_SECTIONS = (
    "abstract",
    "keywords",
    "problem_restatement",
    "problem_analysis",
    "assumptions",
    "notation",
    "model",
    "results",
    "validation",
    "sensitivity",
    "evaluation",
    "conclusion",
    "references",
    "appendix",
)

MODEL_TERMS = (
    "objective",
    "constraint",
    "decision variable",
    "目标函数",
    "約束",
    "约束",
    "决策变量",
)
VALIDATION_TERMS = (
    "error",
    "baseline",
    "validation",
    "ablation",
    "sensitivity",
    "误差",
    "检验",
    "檢驗",
    "基线",
    "基準",
    "对比",
)
EVALUATION_TERMS = (
    "advantage",
    "limitation",
    "improvement",
    "generalization",
    "优点",
    "優點",
    "缺点",
    "缺點",
    "局限",
    "推广",
)


def audit_national_award_structure(text: str) -> PaperStructureAudit:
    issues: list[str] = []
    suggestions: list[str] = []
    headings = _headings(text)
    section_map = _section_map(text)
    present = {key for key in HIGH_VALUE_SECTIONS if _section_text(section_map, key)}
    missing = [key for key in HIGH_VALUE_SECTIONS if key not in present]

    metrics: dict[str, int] = {
        "heading_count": len(headings),
        "award_sections_present": len(present),
        "award_sections_missing": len(missing),
    }

    if missing:
        issues.append("Award structure weak: missing high-value sections: " + ", ".join(missing))
        suggestions.append(
            "Use the national contest structure: abstract, keywords, assumptions, notation, model, results, validation, sensitivity, evaluation, conclusion, references, and appendix."
        )

    abstract = _section_text(section_map, "abstract")
    conclusion = _section_text(section_map, "conclusion")
    result_text = _section_text(section_map, "results")
    problem_ids = _problem_ids(text)
    metrics["problem_answer_targets"] = len(problem_ids)
    missing_answer_closure = [
        problem_id
        for problem_id in problem_ids
        if not _mentions_problem(result_text + "\n" + conclusion, problem_id)
    ]
    metrics["problem_answer_closure_missing"] = len(missing_answer_closure)
    if problem_ids and missing_answer_closure:
        issues.append(
            "Problem-answer closure weak: missing explicit answer thread for "
            + ", ".join(missing_answer_closure[:6])
        )
        suggestions.append(
            "For every subproblem, write the method, final answer, table reference, and conclusion in the result or conclusion sections."
        )

    model_text = _section_text(section_map, "model")
    model_math_signals = _count_model_math_signals(model_text)
    metrics["model_math_signals"] = model_math_signals
    if model_text and model_math_signals < 2:
        issues.append(
            "Model formulation weak: model section lacks objective/constraint/equation signals."
        )
        suggestions.append(
            "Add decision variables, objective functions, constraints, and numbered equations for core models."
        )

    notation = _section_text(section_map, "notation")
    notation_rows = _count_table_rows(notation)
    metrics["notation_table_rows"] = notation_rows
    if notation and notation_rows < 3:
        issues.append("Notation section weak: symbol table has fewer than 3 Markdown rows.")
        suggestions.append("Add a symbol table with variables, meanings, and units.")

    validation = _section_text(section_map, "validation")
    validation_hits = _term_hits(validation, VALIDATION_TERMS)
    metrics["validation_signal_hits"] = validation_hits
    if validation and validation_hits < 2:
        issues.append(
            "Model validation section weak: lacks error, baseline, ablation, or robustness evidence."
        )
        suggestions.append(
            "Add error analysis, baseline comparison, ablation, or sensitivity-backed robustness checks."
        )

    evaluation = _section_text(section_map, "evaluation")
    evaluation_hits = _term_hits(evaluation, EVALUATION_TERMS)
    metrics["evaluation_signal_hits"] = evaluation_hits
    if evaluation and evaluation_hits < 2:
        issues.append(
            "Model evaluation section weak: lacks advantages, limitations, improvements, or generalization."
        )
        suggestions.append(
            "Write explicit model advantages, limitations, improvement directions, and applicable scenarios."
        )

    conclusion_numbers = _substantive_number_count(conclusion)
    metrics["conclusion_substantive_numbers"] = conclusion_numbers
    if conclusion and conclusion_numbers < max(1, min(len(problem_ids), 3)):
        issues.append("Conclusion answer density weak: conclusion lacks numeric final answers.")
        suggestions.append("Restate the final task answers with quantitative values in the conclusion.")

    return PaperStructureAudit(
        issues=_dedupe(issues),
        suggestions=_dedupe(suggestions),
        metrics=metrics,
    )


def _headings(text: str) -> list[tuple[int, str, int, int]]:
    items: list[tuple[int, str, int, int]] = []
    for match in re.finditer(r"^(#{1,6})\s+(.+?)\s*$", text, flags=re.MULTILINE):
        items.append((len(match.group(1)), match.group(2).strip(), match.start(), match.end()))
    return items


def _section_map(text: str) -> dict[str, str]:
    headings = _headings(text)
    sections: dict[str, str] = {}
    for index, (_level, title, _start, end) in enumerate(headings):
        canonical = _canonical_section(title)
        if not canonical:
            continue
        next_start = len(text)
        for next_level, _next_title, candidate_start, _candidate_end in headings[index + 1:]:
            if next_level <= _level:
                next_start = candidate_start
                break
        sections.setdefault(canonical, text[end:next_start].strip())
    return sections


def _canonical_section(title: str) -> str | None:
    normalized = title.lower()
    normalized = re.sub(r"^[一二三四五六七八九十\d]+[.、\s-]*", "", normalized)
    for key, aliases in SECTION_ALIASES.items():
        if any(alias.lower() in normalized for alias in aliases):
            return key
    return None


def _section_text(section_map: dict[str, str], key: str) -> str:
    return section_map.get(key, "")


def _problem_ids(text: str) -> list[str]:
    ids: list[str] = []
    patterns = (
        r"\bQ\s*([1-9]\d*)\b",
        r"\bProblem\s+([1-9]\d*)\b",
        r"问题\s*([一二三四五六七八九十]|[1-9]\d*)",
        r"問題\s*([一二三四五六七八九十]|[1-9]\d*)",
    )
    for pattern in patterns:
        for raw in re.findall(pattern, text, flags=re.IGNORECASE):
            parsed = _normalize_problem_id(str(raw))
            if parsed and parsed not in ids:
                ids.append(parsed)
    return ids


def _normalize_problem_id(raw: str) -> str:
    raw = raw.strip()
    chinese = {
        "一": "1",
        "二": "2",
        "三": "3",
        "四": "4",
        "五": "5",
        "六": "6",
        "七": "7",
        "八": "8",
        "九": "9",
        "十": "10",
    }
    return "Q" + chinese.get(raw, raw) if raw else ""


def _mentions_problem(text: str, problem_id: str) -> bool:
    number = problem_id.removeprefix("Q")
    return bool(
        re.search(rf"\bQ\s*{re.escape(number)}\b", text, flags=re.IGNORECASE)
        or re.search(rf"\bProblem\s+{re.escape(number)}\b", text, flags=re.IGNORECASE)
        or f"问题{number}" in text
        or f"問題{number}" in text
    )


def _count_model_math_signals(text: str) -> int:
    if not text:
        return 0
    equation_count = len(re.findall(r"\\\[|\\\(|\$\$", text))
    return equation_count + _term_hits(text, MODEL_TERMS)


def _count_table_rows(text: str) -> int:
    return len(re.findall(r"^\s*\|.+\|\s*$", text, flags=re.MULTILINE))


def _term_hits(text: str, terms: tuple[str, ...]) -> int:
    lowered = text.lower()
    return sum(1 for term in terms if term.lower() in lowered)


def _substantive_number_count(text: str) -> int:
    values = []
    for raw in re.findall(r"(?<![\w.])[-+]?\d+(?:\.\d+)?%?", text):
        try:
            value = float(raw.rstrip("%"))
        except ValueError:
            continue
        if raw.isdigit() and 1900 <= value <= 2100:
            continue
        values.append(raw)
    return len(values)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
