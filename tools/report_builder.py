from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is a hard dependency in practice
    pd = None  # type: ignore


# ---------------------------------------------------------------------------
# Document model
# ---------------------------------------------------------------------------


@dataclass
class Heading:
    level: int
    text: str
    kind: Literal["heading"] = "heading"


@dataclass
class Paragraph:
    text: str
    kind: Literal["paragraph"] = "paragraph"


@dataclass
class ListBlock:
    items: list[str]
    ordered: bool = False
    kind: Literal["list"] = "list"


@dataclass
class CodeBlock:
    text: str
    kind: Literal["code"] = "code"


@dataclass
class ImageBlock:
    path: Path
    caption: str = ""
    kind: Literal["image"] = "image"


@dataclass
class TableBlock:
    headers: list[str]
    rows: list[list[str]]
    caption: str = ""
    kind: Literal["table"] = "table"


@dataclass
class MathBlock:
    """A standalone display equation, stored as raw LaTeX without delimiters."""

    latex: str
    kind: Literal["math"] = "math"


@dataclass
class InlineRun:
    """A segment of inline content within a paragraph or list item."""

    kind: Literal["text", "bold", "math"]
    content: str


Block = Heading | Paragraph | ListBlock | CodeBlock | ImageBlock | TableBlock | MathBlock


@dataclass
class Document:
    title: str = "数学建模论文"
    blocks: list[Block] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Markdown parsing (tailored to the paper drafts produced by WritingAgent)
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET_RE = re.compile(r"^[-*]\s+(.*)$")
_ORDERED_RE = re.compile(r"^\d+[.)]\s+(.*)$")
_IMAGE_RE = re.compile(r"^!\[(.*?)\]\((.*?)\)\s*$")
_FENCE_RE = re.compile(r"^```")
_HRULE_RE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")
_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:\-|]+\|?\s*$")
_INLINE_RE = re.compile(
    r"\\\[(?P<dmath>.+?)\\\]"
    r"|\\\((?P<math>.+?)\\\)"
    r"|\$\$(?P<dollar_dmath>.+?)\$\$"
    r"|\$(?P<dollar_math>[^$]+?)\$"
    r"|\*\*(?P<bold>.+?)\*\*",
    re.DOTALL,
)
_DISPLAY_MATH_RE = re.compile(r"^\s*(?:\\\[(?P<bracket>.+?)\\\]|\$\$(?P<dollar>.+?)\$\$)\s*$", re.DOTALL)


def iter_inline_runs(text: str) -> list[InlineRun]:
    """Split text into plain, bold and inline-math runs."""

    runs: list[InlineRun] = []
    pos = 0
    for match in _INLINE_RE.finditer(text):
        if match.start() > pos:
            runs.append(InlineRun("text", text[pos : match.start()]))
        if match.group("bold") is not None:
            runs.append(InlineRun("bold", match.group("bold")))
        else:
            latex = (
                match.group("dmath")
                or match.group("math")
                or match.group("dollar_dmath")
                or match.group("dollar_math")
                or ""
            )
            runs.append(InlineRun("math", latex.strip()))
        pos = match.end()
    if pos < len(text):
        runs.append(InlineRun("text", text[pos:]))
    if not runs:
        runs.append(InlineRun("text", text))
    return runs


def parse_markdown(text: str) -> Document:
    """Parse a (simple) Markdown string into the internal document model.

    Supports headings, paragraphs, bullet/ordered lists, fenced code blocks and
    image references. This is intentionally lightweight; it only needs to cover
    the structure emitted by the writing/review agents.
    """

    lines = text.replace("\r\n", "\n").split("\n")
    doc = Document()
    i = 0
    n = len(lines)
    pending_paragraph: list[str] = []

    def flush_paragraph() -> None:
        if pending_paragraph:
            joined = " ".join(part.strip() for part in pending_paragraph).strip()
            if joined:
                math = _DISPLAY_MATH_RE.match(joined)
                if math:
                    doc.blocks.append(MathBlock(latex=(math.group("bracket") or math.group("dollar") or "").strip()))
                else:
                    doc.blocks.append(Paragraph(text=joined))
            pending_paragraph.clear()

    while i < n:
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            i += 1
            continue

        if _HRULE_RE.match(stripped):
            flush_paragraph()
            i += 1
            continue

        # Fenced code block
        if _FENCE_RE.match(stripped):
            flush_paragraph()
            code_lines: list[str] = []
            i += 1
            while i < n and not _FENCE_RE.match(lines[i].strip()):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            doc.blocks.append(CodeBlock(text="\n".join(code_lines)))
            continue

        heading = _HEADING_RE.match(stripped)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            doc.blocks.append(Heading(level=level, text=heading.group(2).strip()))
            i += 1
            continue

        image = _IMAGE_RE.match(stripped)
        if image:
            flush_paragraph()
            caption, src = image.group(1), image.group(2)
            doc.blocks.append(ImageBlock(path=Path(src), caption=caption.strip()))
            i += 1
            continue

        if (
            _TABLE_ROW_RE.match(stripped)
            and i + 1 < n
            and _TABLE_SEP_RE.match(lines[i + 1].strip())
            and "-" in lines[i + 1]
        ):
            flush_paragraph()
            table_lines = [stripped]
            i += 2
            while i < n and _TABLE_ROW_RE.match(lines[i].strip()):
                table_lines.append(lines[i].strip())
                i += 1
            table = _parse_pipe_table(table_lines)
            if table is not None:
                doc.blocks.append(table)
            continue

        # Lists: collect consecutive list items (including nested indented items)
        if _BULLET_RE.match(stripped) or _ORDERED_RE.match(stripped):
            flush_paragraph()
            ordered = bool(_ORDERED_RE.match(stripped))
            items: list[str] = []
            while i < n:
                current = lines[i]
                cstrip = current.strip()
                if not cstrip:
                    break
                bullet = _BULLET_RE.match(cstrip)
                order = _ORDERED_RE.match(cstrip)
                if bullet:
                    items.append(bullet.group(1).strip())
                elif order:
                    items.append(order.group(1).strip())
                elif current.startswith((" ", "\t")):
                    # continuation / nested item -> append to previous entry
                    if items:
                        items[-1] = f"{items[-1]} {cstrip}"
                    else:
                        items.append(cstrip)
                else:
                    break
                i += 1
            doc.blocks.append(ListBlock(items=items, ordered=ordered))
            continue

        pending_paragraph.append(stripped)
        i += 1

    flush_paragraph()
    return doc


# ---------------------------------------------------------------------------
# Enrichment: embed generated figures and result tables as an appendix
# ---------------------------------------------------------------------------


def _csv_to_table(path: Path, caption: str, max_rows: int = 30, max_cols: int = 12) -> TableBlock | None:
    if pd is None or not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    if frame.empty:
        return None
    if frame.shape[1] > max_cols:
        frame = frame.iloc[:, :max_cols]
    truncated = frame.shape[0] > max_rows
    if truncated:
        frame = frame.head(max_rows)
    headers = [str(col) for col in frame.columns]
    rows = [[_fmt_cell(value) for value in record] for record in frame.itertuples(index=False, name=None)]
    note = caption
    if truncated:
        note = f"{caption}（仅展示前 {max_rows} 行）"
    return TableBlock(headers=headers, rows=rows, caption=note)


def _fmt_cell(value: object) -> str:
    if pd is not None:
        try:
            if pd.isna(value):  # type: ignore[arg-type]
                return ""
        except (TypeError, ValueError):
            pass
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return f"{value:.4g}"
    return str(value)


def _parse_pipe_table(rows: list[str]) -> TableBlock | None:
    def cells(line: str) -> list[str]:
        inner = _TABLE_ROW_RE.match(line)
        if not inner:
            return []
        return [c.strip() for c in inner.group(1).split("|")]

    if not rows:
        return None
    headers = cells(rows[0])
    if not headers:
        return None
    body = [cells(line) for line in rows[1:]]
    body = [row for row in body if any(cell for cell in row)]
    width = len(headers)
    return TableBlock(headers=headers, rows=[(row + [""] * width)[:width] for row in body])


def build_appendix_blocks(run_summary_path: Path, model_labels: dict[str, str] | None = None) -> list[Block]:
    """Build figure/table appendix blocks from a run_summary.json file."""

    if not run_summary_path.exists():
        return []
    try:
        payload = json.loads(run_summary_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list) or not payload:
        return []

    labels = model_labels or {}
    blocks: list[Block] = []

    figure_blocks: list[Block] = []
    table_blocks: list[Block] = []

    for item in payload:
        source = Path(str(item.get("source", "data"))).name

        for chart in item.get("charts", []) or []:
            chart_path = Path(chart)
            if chart_path.exists():
                figure_blocks.append(ImageBlock(path=chart_path, caption=f"{source} - {chart_path.stem}"))

        describe = item.get("describe_table")
        if describe:
            table = _csv_to_table(Path(describe), caption=f"{source} 描述统计")
            if table:
                table_blocks.append(table)

        for name, out_path in (item.get("model_outputs", {}) or {}).items():
            label = labels.get(name, name)
            table = _csv_to_table(Path(out_path), caption=f"{source} - {label}")
            if table:
                table_blocks.append(table)

    if figure_blocks:
        blocks.append(Heading(level=2, text="附录A 关键图表"))
        blocks.extend(figure_blocks)
    if table_blocks:
        blocks.append(Heading(level=2, text="附录B 结果数据表"))
        blocks.extend(table_blocks)
    return blocks


def build_document_from_paper(
    paper_path: Path,
    run_summary_path: Path | None = None,
    model_labels: dict[str, str] | None = None,
    title: str = "数学建模论文",
) -> Document:
    """Parse a paper markdown file and (optionally) append a figures/tables appendix."""

    text = paper_path.read_text(encoding="utf-8") if paper_path.exists() else ""
    doc = parse_markdown(text)
    doc.title = title
    if run_summary_path is not None:
        doc.blocks.extend(build_appendix_blocks(run_summary_path, model_labels))
    return doc
