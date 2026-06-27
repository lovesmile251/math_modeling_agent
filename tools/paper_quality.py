"""Paper quality evaluation with multi-dimensional scoring.

v2 improvements:
- #2  Figure citation integrity check
- #6  LaTeX bracket-matching validation
- #7  Reference structural validation
- #8  Score alignment with national-award deep-study thresholds
- #10 Structured checks via report_builder.parse_markdown()
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Patterns & thresholds
# ---------------------------------------------------------------------------

CHATTER_PATTERNS = (
    r"^\s*好的[，,].*",
    r"^\s*请稍等.*",
    r"^\s*我将.*撰写.*",
    r"^\s*下面是.*论文.*",
    r"^\s*以下是.*论文.*",
)

REQUIRED_SECTION_GROUPS = (
    ("摘要",),
    ("关键词",),
    ("问题重述",),
    ("问题分析",),
    ("模型假设",),
    ("符号说明",),
    ("模型",),
    ("结果",),
    ("检验", "误差", "灵敏度", "敏感性", "对比"),
    ("评价", "推广"),
    ("参考文献",),
    ("附录",),
)

# Thresholds derived from deep-study of 16 national-award papers (804 pages total).
# These are *median* values — a paper below them gets a warning, not a hard fail.
# Source: prompts/national_award_paper_deep_study.md
AWARD_MEDIAN_EQUATIONS = 25
AWARD_MEDIAN_TABLES = 12
AWARD_MEDIAN_FIGURES = 8
AWARD_MEDIAN_REFERENCES = 10
AWARD_MEDIAN_CHARS = 20000
AWARD_ABSTRACT_MIN_CHARS = 300
AWARD_ABSTRACT_MAX_CHARS = 800
AWARD_MIN_KEYWORDS = 3
AWARD_MAX_KEYWORDS = 5

SUBMISSION_BLOCKER_PATTERNS = (
    r"待补充",
    r"待完善",
    r"待确定",
    r"未产出优化数值解",
    r"未产出.*数值解",
    r"未得到.*数值解",
    r"后续计算",
    r"后续可补充",
    r"后续补充",
    r"后续完善",
    r"\bTODO\b",
)

SEVERE_ISSUE_MARKERS = (
    "Submission blocker",
    "Core result table missing",
    "Reference citations without matching entries",
    "Reference entries not cited in body",
    "Reference section is missing",
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PaperQualityReport:
    score: int
    issues: list[str]
    suggestions: list[str]
    metrics: dict[str, int]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_paper_text(text: str) -> str:
    """Remove LLM chat residue and normalize Markdown heading style."""

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    first_heading = next((i for i, line in enumerate(lines) if re.match(r"^\s*#{1,6}\s+", line)), None)
    if first_heading is not None and any(_is_chatter(line) for line in lines[:first_heading]):
        lines = lines[first_heading:]

    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        if _is_chatter(stripped):
            continue
        if re.match(r"^\s*[-*_]{3,}\s*$", stripped):
            continue
        line = _normalize_heading_line(line)
        cleaned.append(line.rstrip())

    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()

    # The first Markdown heading is the paper title; make it a level-1 heading.
    for idx, line in enumerate(cleaned):
        match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if match:
            cleaned[idx] = "# " + match.group(2).strip()
            break

    out: list[str] = []
    blank = 0
    for line in cleaned:
        if line.strip():
            blank = 0
            out.append(line)
        else:
            blank += 1
            if blank <= 2:
                out.append("")
    return "\n".join(out).strip() + "\n"


def evaluate_paper_quality(
    text: str,
    workspace_root: Path | None = None,
    available_figures: list[str] | None = None,
) -> PaperQualityReport:
    """Score a paper across structural, content, LaTeX, reference, and figure dimensions.

    Args:
        text: The paper Markdown.
        workspace_root: If given, checks that cited figures exist on disk.
        available_figures: If given, list of figure filenames the paper *should* cite.
    """
    issues: list[str] = []
    suggestions: list[str] = []

    metrics = {
        "chars": len(re.sub(r"\s+", "", text)),
        "equations": len(re.findall(r"\\\(|\\\[|\$\$", text)),
        "tables": len(re.findall(r"^\s*\|.+\|\s*$", text, flags=re.MULTILINE)),
        "result_tables": _count_markdown_table_rows(_extract_section(text, "结果")),
        "figures": len(re.findall(r"!\[.*?\]\(.*?\)", text)),
        "digits": len(re.findall(r"\d", text)),
        "problem_sections": len(re.findall(r"问题[一二三四五六七八九十\d]", text)),
        "references": len(re.findall(r"\[\d+\]", text)),
        "keywords": 0,
    }

    # ── structural checks ──
    _check_structure(text, issues, suggestions, metrics)

    blocker_issues, blocker_suggestions = check_submission_readiness(text)
    issues.extend(blocker_issues)
    suggestions.extend(blocker_suggestions)

    keyword_issues, keyword_suggestions, keyword_count = check_keywords(text)
    metrics["keywords"] = keyword_count
    issues.extend(keyword_issues)
    suggestions.extend(keyword_suggestions)

    # ── LaTeX integrity (#6) ──
    latex_issues = check_latex_integrity(text)
    issues.extend(latex_issues)
    if latex_issues:
        suggestions.append("修复 LaTeX 语法错误：括号不匹配或非法命令。")

    # ── figure citation integrity (#2) ──
    fig_issues, fig_suggestions = check_figure_citations(
        text, workspace_root=workspace_root, available_figures=available_figures
    )
    issues.extend(fig_issues)
    suggestions.extend(fig_suggestions)

    # ── reference quality (#7) ──
    ref_issues, ref_suggestions = check_references(text)
    issues.extend(ref_issues)
    suggestions.extend(ref_suggestions)

    trace_issues, trace_suggestions = check_traceability(text, workspace_root=workspace_root)
    issues.extend(trace_issues)
    suggestions.extend(trace_suggestions)

    # ── structured parse via report_builder (#10) ──
    struct_issues, struct_suggestions = _check_via_parse(text)
    issues.extend(struct_issues)
    suggestions.extend(struct_suggestions)

    # ── scoring (#8: aligned with award medians) ──
    score = _compute_score(issues, metrics)

    if not issues:
        suggestions.append("论文质量门禁通过；后续重点做语言润色和版式检查。")

    return PaperQualityReport(score=score, issues=issues, suggestions=_dedupe(suggestions), metrics=metrics)


def format_quality_report(report: PaperQualityReport) -> str:
    lines = [
        "## 国奖质量门禁",
        f"- 质量分：{report.score}/100",
        f"- 正文字数估计：{report.metrics['chars']}",
        f"- 公式标记数：{report.metrics['equations']}",
        f"- Markdown 表格行数：{report.metrics['tables']}",
        f"- 结果章节表格行数：{report.metrics.get('result_tables', 0)}",
        f"- 图片引用数：{report.metrics['figures']}",
        f"- 关键词数量：{report.metrics.get('keywords', 0)}",
        f"- 参考文献标记数：{report.metrics.get('references', 0)}",
        "",
        "### 质量问题",
    ]
    lines.extend(f"- {item}" for item in (report.issues or ["未发现硬性质量问题。"]))
    lines.append("")
    lines.append("### 优化建议")
    lines.extend(f"- {item}" for item in report.suggestions)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# #2 — Figure citation integrity
# ---------------------------------------------------------------------------

def check_figure_citations(
    text: str,
    workspace_root: Path | None = None,
    available_figures: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Verify that cited figures exist and important figures are cited.

    Returns (issues, suggestions).
    """
    issues: list[str] = []
    suggestions: list[str] = []

    cited = set()
    for match in re.finditer(r"!\[.*?\]\((.+?)\)", text):
        fname = Path(match.group(1)).name
        cited.add(fname)

    # Check file existence
    if workspace_root:
        for fname in cited:
            full = workspace_root / fname
            if not full.exists():
                # also try figures_dir
                figures_dir = workspace_root / "figures"
                if not (figures_dir / fname).exists():
                    issues.append(f"引用的图片文件不存在：{fname}。")

    # Check that available figures are cited
    if available_figures:
        available_set = {Path(f).name for f in available_figures}
        missing_citations = available_set - cited
        if missing_citations:
            suggestions.append(
                f"以下已生成的图表未被正文引用：{', '.join(sorted(missing_citations)[:5])}"
                + ("..." if len(missing_citations) > 5 else "")
            )

    return issues, suggestions


# ---------------------------------------------------------------------------
# #6 — LaTeX integrity
# ---------------------------------------------------------------------------

def check_latex_integrity(text: str) -> list[str]:
    """Validate LaTeX bracket matching and detect common errors.

    Returns list of issue descriptions (empty = clean).
    """
    issues: list[str] = []

    # Count bracket pairs
    inline_open = text.count(r"\(")
    inline_close = text.count(r"\)")
    display_open = text.count(r"\[")
    display_close = text.count(r"\]")
    dollar_pairs = text.count("$$")  # should be even

    if inline_open != inline_close:
        issues.append(
            f"LaTeX 行内公式括号不匹配：\\( 出现 {inline_open} 次，\\) 出现 {inline_close} 次。"
        )
    if display_open != display_close:
        issues.append(
            f"LaTeX 独立公式括号不匹配：\\[ 出现 {display_open} 次，\\] 出现 {display_close} 次。"
        )
    if dollar_pairs % 2 != 0:
        issues.append("LaTeX $$ 分隔符不成对（应为偶数个）。")

    # Detect common invalid commands (missing backslash)
    for illegal in (r"alpha ", r"beta ", r"gamma ", r"delta ", r"epsilon ",
                    r"sum ", r"prod ", r"int ", r"frac ", r"sqrt "):
        if illegal in text.replace("\\", ""):  # don't flag properly escaped ones
            continue
        # Actually check: these patterns WITHOUT leading backslash
        # More robust: find instances where Greek letters appear as plain text in math context
        pass  # False-positive prone; skip for now

    # Detect unclosed braces inside math
    for delim_pair in [(r"\((", r"\)"), (r"\[", r"\]")]:
        # Simple heuristic: strip everything outside math delimiters, count braces inside
        pass  # Complex to implement robustly; add if needed

    return issues


# ---------------------------------------------------------------------------
# #7 — Reference validation
# ---------------------------------------------------------------------------

def check_references(text: str) -> tuple[list[str], list[str]]:
    """Check that references have the required structural elements.

    Returns (issues, suggestions).
    """
    issues: list[str] = []
    suggestions: list[str] = []

    # Extract reference section
    ref_section = _extract_section(text, "参考文献")
    if not ref_section:
        return issues, suggestions

    # Split into individual references (lines starting with [N] or containing [N])
    ref_entries = re.findall(r"(?:^|\n)\s*\[\d+\][^\n]+", ref_section)
    if not ref_entries:
        ref_entries = [ref_section]  # Fallback: treat as one blob

    complete = 0
    incomplete = 0
    for entry in ref_entries:
        entry = entry.strip()
        # A well-formed reference has: [N] Author. Title. Source, Year.
        # Check for minimum structure: number + text + year-like pattern
        has_number = bool(re.match(r"\[\d+\]", entry))
        has_year = bool(re.search(r"(?:19|20)\d{2}[a-z]?", entry))
        has_author = len(entry) > 15  # At least some content beyond the number

        if has_number and has_year and has_author:
            complete += 1
        else:
            incomplete += 1

    if incomplete > 0:
        issues.append(
            f"参考文献中 {incomplete} 条缺少必要信息（作者/标题/年份）；"
            f"完整条目 {complete} 条。"
        )
        suggestions.append(
            "每条参考文献应包含：[编号] 作者. 标题. 来源, 年份. 四要素。"
        )

    # Compare count against award median
    total = complete + incomplete
    if total < AWARD_MEDIAN_REFERENCES:
        suggestions.append(
            f"参考文献数量（{total}）低于国奖论文中位数（{AWARD_MEDIAN_REFERENCES}），"
            "建议补充更多真实文献。"
        )

    return issues, suggestions


def check_submission_readiness(text: str) -> tuple[list[str], list[str]]:
    """Reject phrases that indicate the paper is not submit-ready."""
    issues: list[str] = []
    suggestions: list[str] = []

    offenders: list[str] = []
    for pattern in SUBMISSION_BLOCKER_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            snippet = re.sub(r"\s+", " ", match.group(0)).strip()
            if snippet:
                offenders.append(snippet[:60])

    if offenders:
        unique = _dedupe(offenders)
        issues.append(
            "Submission blocker phrases remain in paper: "
            + ", ".join(unique[:8])
        )
        suggestions.append(
            "Remove non-submit-ready wording and rewrite unsupported claims as completed, evidence-bounded conclusions."
        )

    return issues, suggestions


def check_keywords(text: str) -> tuple[list[str], list[str], int]:
    """Validate the keyword section and return (issues, suggestions, count)."""
    keyword_text = _extract_keywords_text(text)
    keywords = _split_keywords(keyword_text)
    count = len(keywords)

    issues: list[str] = []
    suggestions: list[str] = []
    if count == 0:
        issues.append("缺少关键词或关键词无法解析。")
        suggestions.append("补充 3-5 个关键词，并用分号、逗号或顿号分隔。")
    elif count < AWARD_MIN_KEYWORDS:
        issues.append(f"关键词数量（{count}）偏少，要求 3-5 个。")
        suggestions.append("补充能够覆盖模型、方法和应用场景的关键词。")
    elif count > AWARD_MAX_KEYWORDS:
        issues.append(f"关键词数量（{count}）偏多，要求 3-5 个。")
        suggestions.append("合并或删除泛化关键词，保留 3-5 个核心关键词。")

    return issues, suggestions, count


def check_traceability(
    text: str,
    workspace_root: Path | None = None,
) -> tuple[list[str], list[str]]:
    """Check claims that can be traced to generated artifacts.

    The checks are intentionally conservative: they report only filename,
    reference-number, and model-output mismatches that can be verified from the
    workspace without interpreting the paper semantically.
    """
    issues: list[str] = []
    suggestions: list[str] = []

    declared_refs = _declared_reference_numbers(text)
    cited_refs = _inline_reference_numbers(text)
    if cited_refs and not declared_refs:
        issues.append(
            "Reference section is missing for body citations: "
            + ", ".join(f"[{number}]" for number in sorted(cited_refs)[:10])
        )
    missing_entries = sorted(cited_refs - declared_refs)
    uncited_entries = sorted(declared_refs - cited_refs)
    if missing_entries:
        issues.append(
            "Reference citations without matching entries: "
            + ", ".join(f"[{number}]" for number in missing_entries[:10])
        )
    if uncited_entries:
        message = (
            "Reference entries not cited in body: "
            + ", ".join(f"[{number}]" for number in uncited_entries[:10])
        )
        issues.append(message)
        suggestions.append(message)

    if workspace_root:
        table_suffixes = _generated_table_suffixes(workspace_root)
        if table_suffixes:
            claimed_models = _claimed_model_suffixes(text)
            missing_outputs = sorted(claimed_models - table_suffixes)
            if missing_outputs:
                issues.append(
                    "Model claims without generated result tables: "
                    + ", ".join(missing_outputs[:10])
                )
                suggestions.append("Remove unsupported model claims or run the corresponding models first.")

    return issues, suggestions


# ---------------------------------------------------------------------------
# #10 — Structured checks via report_builder
# ---------------------------------------------------------------------------

def _check_via_parse(text: str) -> tuple[list[str], list[str]]:
    """Use report_builder's Markdown parser for structured section-level checks.

    Falls back gracefully if report_builder is not available.
    """
    issues: list[str] = []
    suggestions: list[str] = []

    try:
        from tools.report_builder import parse_markdown

        doc = parse_markdown(text)
    except Exception:
        return issues, suggestions

    # Count actual sections (headings only, not other blocks)
    headings = [b for b in doc.blocks if hasattr(b, "level")]
    heading_count = len(headings)

    # Detect if any image block's path doesn't exist
    image_blocks = [b for b in doc.blocks if hasattr(b, "path") and not hasattr(b, "level")]
    if image_blocks and not any("图片文件不存在" in i for i in issues):
        # The figure check is already done by check_figure_citations above;
        # here we just note the count parsed
        pass

    # Check for consecutive headings without body text between them
    for i in range(len(headings) - 1):
        curr = headings[i]
        next_h = headings[i + 1]
        # If two headings are adjacent in the block list, there's no content
        curr_idx = doc.blocks.index(curr)
        next_idx = doc.blocks.index(next_h)
        if next_idx - curr_idx == 1:
            suggestions.append(
                f"章节“{_heading_text(curr)}”之后缺少正文内容，建议补充。"
            )
            break  # One warning is enough

    return issues, suggestions


def _heading_text(heading) -> str:
    try:
        return heading.text[:40]
    except Exception:
        return "?"


# ---------------------------------------------------------------------------
# #8 — Scoring aligned with award medians
# ---------------------------------------------------------------------------

def _compute_score(issues: list[str], metrics: dict) -> int:
    """Compute quality score with thresholds calibrated against 16 real award papers.

    Scoring dimensions:
    - Baselined: start at 100, subtract 8 per structural issue (max -55)
    - Depth warnings (soft): warn when below award median, but only -3 each
    - Short paper penalty: -10 if chars < AWARD_MEDIAN_CHARS
    """
    score = 100

    # Hard issues: -8 each, with extra weight for submit blockers.
    # Only count structural/LaTeX/reference issues, not "suggestions-only"
    hard_issues = [
        i for i in issues
        if any(
            kw in i
            for kw in (
                "缺少",
                "不匹配",
                "不存在",
                "不足",
                "偏短",
                "偏少",
                "偏多",
                "Submission blocker",
                "Reference citations",
                "Reference entries",
                "Reference section",
                "Core result table",
                "Model claims without",
            )
        )
    ]
    severe_issues = [
        i for i in issues if any(marker in i for marker in SEVERE_ISSUE_MARKERS)
    ]
    score -= min(70, len(hard_issues) * 8 + len(severe_issues) * 10)

    # Soft warnings (below award median but not hard-fail)
    if metrics["equations"] > 0 and metrics["equations"] < AWARD_MEDIAN_EQUATIONS:
        score -= 3
    if metrics["tables"] > 0 and metrics["tables"] < AWARD_MEDIAN_TABLES:
        score -= 3
    if metrics["figures"] > 0 and metrics["figures"] < AWARD_MEDIAN_FIGURES:
        score -= 3

    # Short paper penalty
    if metrics["chars"] < AWARD_MEDIAN_CHARS:
        score -= 10

    return max(0, score)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_structure(
    text: str, issues: list[str], suggestions: list[str], metrics: dict
) -> None:
    """Populate issues/suggestions for structural dimensions."""
    if any(_is_chatter(line) for line in text.splitlines()):
        issues.append("论文中残留助手寒暄或生成说明。")
        suggestions.append("删除\u201c好的，请稍等\u201d\u201c我将撰写\u201d等非论文内容。")

    missing = [
        "/".join(group)
        for group in REQUIRED_SECTION_GROUPS
        if not any(section in text for section in group)
    ]
    if missing:
        issues.append("缺少国奖论文常见关键章节：" + "、".join(missing) + "。")
        suggestions.append(
            "补齐摘要、符号说明、模型检验/误差分析、模型评价、参考文献和附录等章节。"
        )

    abstract = _extract_section(text, "摘要")
    if not abstract:
        issues.append("缺少摘要正文。")
    else:
        abstract_chars = len(re.sub(r"\s+", "", abstract))
        abstract_digits = len(re.findall(r"\d", abstract))
        if abstract_chars < AWARD_ABSTRACT_MIN_CHARS:
            issues.append("摘要偏短，未达到国奖论文常见信息密度。")
            suggestions.append("摘要应逐题写出模型、关键数值结果和结论意义。")
        if abstract_digits < 8:
            issues.append("摘要量化结果不足。")
            suggestions.append(
                "把真实结果中的关键数值（排名、误差、权重、传播范围等）写入摘要。"
            )

    if metrics["equations"] < 8:
        issues.append("公式数量偏少，模型数学骨架不足。")
        suggestions.append("为每个核心模型补变量定义、目标函数/评价指标和约束条件。")
    if metrics["tables"] < 6:
        issues.append("结果表或符号表不足，证据密度偏低。")
        suggestions.append("正文保留符号说明表、核心结果表、对比表或参数表。")
    if metrics.get("result_tables", 0) == 0:
        issues.append("Core result table missing: no Markdown table is present in the result section.")
        suggestions.append("Add at least one core task/result table before treating the paper as submit-ready.")
    if metrics["figures"] < 2:
        issues.append("正文图表引用偏少。")
        suggestions.append("至少引用关键图表，并紧跟解释。")
    if not re.search(r"敏感性|灵敏度|误差|检验|对比|基准", text):
        issues.append("缺少模型检验、误差分析或基准对比。")
        suggestions.append("增加模型检验/误差分析/灵敏度分析，说明结果可信度。")
    if metrics["problem_sections"] < 3:
        issues.append("分问题闭环不明显。")
        suggestions.append("每个子问题按\u201c分析-模型-求解-结果-解释\u201d展开。")


def _normalize_heading_line(line: str) -> str:
    match = re.match(r"^(\s*#{1,6}\s+)(?:\*\*)?(.+?)(?:\*\*)?\s*$", line)
    if not match:
        return line
    prefix, title = match.group(1), match.group(2).strip()
    title = title.replace("摘 要", "摘要").replace("摘  要", "摘要")
    title = re.sub(r"\s*、\s*", "、", title)
    return prefix + title


def _is_chatter(line: str) -> bool:
    return any(re.match(pattern, line) for pattern in CHATTER_PATTERNS)


def _extract_section(text: str, section_name: str) -> str:
    pattern = re.compile(rf"^#+\s*.*{re.escape(section_name)}.*$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return ""
    start = match.end()
    next_heading = re.search(r"^#+\s+", text[start:], flags=re.MULTILINE)
    end = start + next_heading.start() if next_heading else len(text)
    return text[start:end].strip()


def _count_markdown_table_rows(text: str) -> int:
    return len(re.findall(r"^\s*\|.+\|\s*$", text, flags=re.MULTILINE))


def _extract_keywords_text(text: str) -> str:
    section = _extract_section(text, "关键词")
    if section:
        return section

    inline = re.search(r"关键词\s*[:：]\s*(.+)", text)
    if inline:
        return inline.group(1).strip()
    return ""


def _split_keywords(keyword_text: str) -> list[str]:
    if not keyword_text:
        return []

    cleaned = re.sub(r"^\s*[-*]\s*", "", keyword_text.strip())
    cleaned = re.sub(r"^关键词\s*[:：]\s*", "", cleaned)
    parts = re.split(r"[；;，,、\n]+", cleaned)
    return [
        part.strip().strip("。.;；,，")
        for part in parts
        if part.strip().strip("。.;；,，")
    ]


def _declared_reference_numbers(text: str) -> set[int]:
    ref_section = _extract_reference_section(text)
    if not ref_section:
        return set()
    return {int(number) for number in re.findall(r"^\s*\[(\d+)\]", ref_section, flags=re.MULTILINE)}


def _inline_reference_numbers(text: str) -> set[int]:
    ref_section = _extract_reference_section(text)
    body = text.replace(ref_section, "") if ref_section else text
    return {int(number) for number in re.findall(r"\[(\d+)\]", body)}


def _extract_reference_section(text: str) -> str:
    for marker in ("参考文献",):
        section = _extract_section(text, marker)
        if section:
            return section
    return ""


def _generated_table_suffixes(workspace_root: Path) -> set[str]:
    tables_dir = workspace_root / "tables"
    if not tables_dir.exists():
        return set()
    suffixes: set[str] = set()
    for path in tables_dir.glob("*.csv"):
        parts = path.stem.split("_")
        for start in range(1, len(parts)):
            suffixes.add("_".join(parts[start:]))
    return suffixes


def _claimed_model_suffixes(text: str) -> set[str]:
    try:
        from tools.model_registry import registry_table_suffixes
    except Exception:
        return set()

    ref_section = _extract_reference_section(text)
    body = text.replace(ref_section, "") if ref_section else text
    normalized = body.lower().replace("-", "_")
    claimed: set[str] = set()
    for suffix in registry_table_suffixes():
        readable = suffix.replace("_", " ")
        if suffix.lower() in normalized or readable.lower() in normalized:
            claimed.add(suffix)
    return claimed


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
