from __future__ import annotations

from pathlib import Path

from tools.report_builder import (
    Block,
    CodeBlock,
    Document,
    Heading,
    ImageBlock,
    ListBlock,
    MathBlock,
    Paragraph,
    TableBlock,
    iter_inline_runs,
)
from tools.math_render import render_latex

# Default CJK-capable font names per backend. reportlab ships the Adobe
# STSong-Light CID font so Chinese renders without any external font file.
_DOCX_CJK_FONT = "SimSun"
_PDF_CJK_FONT = "STSong-Light"


def _equation_dir(output_path: Path) -> Path:
    return output_path.parent / "_equations"


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------


def export_docx(doc: Document, output_path: Path) -> Path:
    from docx import Document as DocxDocument
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt

    docx = DocxDocument()

    normal = docx.styles["Normal"]
    normal.font.name = _DOCX_CJK_FONT
    normal.font.size = Pt(11)
    rpr = normal.element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    rfonts.set(qn("w:eastAsia"), _DOCX_CJK_FONT)

    eq_dir = _equation_dir(output_path)

    docx.add_heading(doc.title, level=0)

    for block in _content_blocks(doc):
        if isinstance(block, Heading):
            heading = docx.add_heading("", level=min(max(block.level, 1), 4))
            _docx_add_runs(heading, block.text, eq_dir)
        elif isinstance(block, Paragraph):
            _docx_add_runs(docx.add_paragraph(), block.text, eq_dir)
        elif isinstance(block, MathBlock):
            # Render display math as image (works in all viewers, no OMML leak)
            image = render_latex(block.latex, eq_dir, display=True)
            if image is not None and image.path.exists():
                para = docx.add_paragraph()
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                try:
                    para.add_run().add_picture(
                        str(image.path),
                        width=Inches(min(image.width_pt / 72, 5.8)),
                    )
                except Exception:
                    para.add_run(_math_fallback_text(block.latex))
            else:
                para = docx.add_paragraph()
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                para.add_run(_math_fallback_text(block.latex))
        elif isinstance(block, ListBlock):
            style = "List Number" if block.ordered else "List Bullet"
            for item in block.items:
                _docx_add_runs(docx.add_paragraph(style=style), item, eq_dir)
        elif isinstance(block, CodeBlock):
            para = docx.add_paragraph()
            run = para.add_run(block.text)
            run.font.name = "Consolas"
            run.font.size = Pt(9)
        elif isinstance(block, ImageBlock):
            if block.path.exists():
                try:
                    docx.add_picture(str(block.path), width=Inches(5.8))
                except Exception:
                    docx.add_paragraph(f"[图片无法嵌入：{block.path.name}]")
                if block.caption:
                    cap = docx.add_paragraph(block.caption)
                    cap.alignment = 1  # centered
        elif isinstance(block, TableBlock):
            if block.caption:
                docx.add_paragraph(block.caption)
            table = docx.add_table(rows=1, cols=len(block.headers))
            table.style = "Table Grid"
            for idx, header in enumerate(block.headers):
                table.rows[0].cells[idx].text = ""
                _docx_add_runs(table.rows[0].cells[idx].paragraphs[0], header, eq_dir)
            for row in block.rows:
                cells = table.add_row().cells
                for idx, value in enumerate(row[: len(block.headers)]):
                    cells[idx].text = ""
                    _docx_add_runs(cells[idx].paragraphs[0], value, eq_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    docx.save(str(output_path))
    return output_path


def _docx_add_runs(paragraph, text: str, eq_dir: Path) -> None:
    """Add inline runs to a paragraph, rendering math as small embedded images.

    No OMML/Office Math is used — all equations become PNG images via
    matplotlib, so they render correctly in Word, LibreOffice, WPS, and
    browser-based viewers without leaking raw OOXML tags.
    """
    from docx.shared import Inches

    for run in iter_inline_runs(text):
        if run.kind == "text":
            if run.content:
                paragraph.add_run(run.content)
        elif run.kind == "bold":
            r = paragraph.add_run(run.content)
            r.bold = True
        else:  # inline math
            image = render_latex(run.content, eq_dir, display=False)
            if image is not None and image.path.exists():
                try:
                    run_obj = paragraph.add_run()
                    run_obj.add_picture(
                        str(image.path),
                        width=Inches(min(image.width_pt / 72, 5.5)),
                    )
                except Exception:
                    paragraph.add_run(_math_fallback_text(run.content))
            else:
                paragraph.add_run(_math_fallback_text(run.content))


def _math_fallback_text(latex: str) -> str:
    """Minimal Unicode fallback when math rendering fails.

    Strips LaTeX commands and keeps only readable characters so the paper
    is still legible even without rendered equation images.
    """
    import re

    # Remove common LaTeX commands, keep Greek letters and math symbols
    cleaned = re.sub(r"\\[a-zA-Z]+", "", latex)
    cleaned = cleaned.replace("{", "").replace("}", "")
    cleaned = cleaned.replace("$", "").replace("\\", "")
    cleaned = cleaned.replace("_", "").replace("^", "")
    return cleaned.strip() or latex[:40]


# ---------------------------------------------------------------------------
# PDF (reportlab + Adobe STSong-Light for CJK)
# ---------------------------------------------------------------------------


def export_pdf(doc: Document, output_path: Path) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import (
        Image as RLImage,
        ListFlowable,
        ListItem,
        Paragraph as RLParagraph,
        Preformatted,
        SimpleDocTemplate,
        Spacer,
        Table as RLTable,
        TableStyle,
    )

    pdfmetrics.registerFont(UnicodeCIDFont(_PDF_CJK_FONT))

    base = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=base["Normal"], fontName=_PDF_CJK_FONT, fontSize=10.5, leading=16)
    title_style = ParagraphStyle("DocTitle", parent=body, fontSize=20, leading=26, spaceAfter=14, alignment=TA_CENTER)
    caption_style = ParagraphStyle("Caption", parent=body, fontSize=9, textColor=colors.grey, alignment=TA_CENTER)
    code_style = ParagraphStyle("Code", parent=base["Code"], fontSize=8, leading=11)
    heading_styles = {
        level: ParagraphStyle(
            f"H{level}",
            parent=body,
            fontSize=max(18 - 2 * level, 11),
            leading=max(22 - 2 * level, 15),
            spaceBefore=10,
            spaceAfter=6,
        )
        for level in range(1, 5)
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    eq_dir = _equation_dir(output_path)
    story: list = [RLParagraph(_esc(doc.title), title_style), Spacer(1, 0.3 * cm)]
    max_width = A4[0] - 4 * cm

    for block in _content_blocks(doc):
        if isinstance(block, Heading):
            level = min(max(block.level, 1), 4)
            story.append(RLParagraph(_rl_inline(block.text, eq_dir), heading_styles[level]))
        elif isinstance(block, Paragraph):
            story.append(RLParagraph(_rl_inline(block.text, eq_dir), body))
        elif isinstance(block, MathBlock):
            image = render_latex(block.latex, eq_dir, display=True)
            if image is not None and image.path.exists():
                width, height = image.width_pt, image.height_pt
                if width > max_width:
                    height *= max_width / width
                    width = max_width
                flowable = RLImage(str(image.path), width=width, height=height)
                flowable.hAlign = "CENTER"
                story.append(flowable)
            else:
                story.append(RLParagraph(_esc(block.latex), body))
        elif isinstance(block, ListBlock):
            items = [ListItem(RLParagraph(_rl_inline(item, eq_dir), body)) for item in block.items]
            story.append(
                ListFlowable(items, bulletType="1" if block.ordered else "bullet", start="1" if block.ordered else None)
            )
        elif isinstance(block, CodeBlock):
            story.append(Preformatted(block.text, code_style))
        elif isinstance(block, ImageBlock):
            flowable = _pdf_image(block.path, max_width, RLImage, ImageReader)
            if flowable is not None:
                story.append(flowable)
                if block.caption:
                    story.append(RLParagraph(_rl_inline(block.caption, eq_dir), caption_style))
                story.append(Spacer(1, 0.2 * cm))
        elif isinstance(block, TableBlock):
            if block.caption:
                story.append(RLParagraph(_rl_inline(block.caption, eq_dir), caption_style))
            story.append(_pdf_table(block, max_width, body, RLTable, TableStyle, colors, eq_dir))
            story.append(Spacer(1, 0.2 * cm))

    SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=doc.title,
    ).build(story)
    return output_path


def _pdf_image(path: Path, max_width: float, RLImage, ImageReader):
    if not path.exists():
        return None
    try:
        reader = ImageReader(str(path))
        iw, ih = reader.getSize()
        if iw <= 0 or ih <= 0:
            return None
        scale = min(max_width / iw, 1.0)
        return RLImage(str(path), width=iw * scale, height=ih * scale)
    except Exception:
        return None


def _rl_inline(text: str, eq_dir: Path) -> str:
    parts: list[str] = []
    for run in iter_inline_runs(text):
        if run.kind == "text":
            parts.append(_esc(run.content))
        elif run.kind == "bold":
            parts.append("<b>" + _esc(run.content) + "</b>")
        else:
            image = render_latex(run.content, eq_dir, display=False)
            if image is not None:
                parts.append(
                    f'<img src="{image.path.as_posix()}" width="{image.width_pt:.2f}" '
                    f'height="{image.height_pt:.2f}" valign="-2"/>'
                )
            else:
                parts.append(_esc(run.content))
    return "".join(parts)


def _pdf_table(block: TableBlock, max_width: float, body_style, RLTable, TableStyle, colors, eq_dir: Path):
    from reportlab.platypus import Paragraph as RLParagraph

    cell_style = body_style.clone("Cell")
    cell_style.fontSize = 8.5
    cell_style.leading = 11
    header_cells = [RLParagraph(_rl_inline(h, eq_dir), cell_style) for h in block.headers]
    data = [header_cells]
    for row in block.rows:
        padded = list(row[: len(block.headers)])
        padded += [""] * (len(block.headers) - len(padded))
        data.append([RLParagraph(_rl_inline(value, eq_dir), cell_style) for value in padded])

    col_count = max(len(block.headers), 1)
    col_width = max_width / col_count
    table = RLTable(data, colWidths=[col_width] * col_count, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3b6ea5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


# ---------------------------------------------------------------------------
# LaTeX (.tex, compile with xelatex for Chinese via ctex)
# ---------------------------------------------------------------------------

_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _latex_escape(text: str) -> str:
    out: list[str] = []
    for ch in text:
        out.append(_LATEX_SPECIAL.get(ch, ch))
    return "".join(out)


def _latex_inline(text: str) -> str:
    parts: list[str] = []
    for run in iter_inline_runs(text):
        if run.kind == "text":
            parts.append(_latex_escape(run.content))
        elif run.kind == "bold":
            parts.append(r"\textbf{" + _latex_escape(run.content) + "}")
        else:
            parts.append("$" + run.content + "$")
    return "".join(parts)


def export_latex(doc: Document, output_path: Path) -> Path:
    section_cmds = {1: "section", 2: "subsection", 3: "subsubsection", 4: "paragraph"}
    lines: list[str] = [
        r"\documentclass[12pt]{article}",
        r"\usepackage{ctex}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{geometry}",
        r"\usepackage{float}",
        r"\geometry{a4paper,margin=2.5cm}",
        r"\title{" + _latex_escape(doc.title) + "}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
    ]

    for block in _content_blocks(doc):
        if isinstance(block, Heading):
            cmd = section_cmds.get(min(max(block.level, 1), 4), "paragraph")
            lines.append(f"\\{cmd}*{{{_latex_inline(block.text)}}}")
        elif isinstance(block, Paragraph):
            lines.append(_latex_inline(block.text))
            lines.append("")
        elif isinstance(block, MathBlock):
            lines.append(r"\[")
            lines.append(block.latex)
            lines.append(r"\]")
            lines.append("")
        elif isinstance(block, ListBlock):
            env = "enumerate" if block.ordered else "itemize"
            lines.append(f"\\begin{{{env}}}")
            for item in block.items:
                lines.append(r"  \item " + _latex_inline(item))
            lines.append(f"\\end{{{env}}}")
        elif isinstance(block, CodeBlock):
            lines.append(r"\begin{verbatim}")
            lines.append(block.text)
            lines.append(r"\end{verbatim}")
        elif isinstance(block, ImageBlock):
            if block.path.exists():
                path_str = block.path.as_posix()
                lines.append(r"\begin{figure}[H]")
                lines.append(r"  \centering")
                lines.append(r"  \includegraphics[width=0.85\textwidth]{" + path_str + "}")
                if block.caption:
                    lines.append(r"  \caption{" + _latex_inline(block.caption) + "}")
                lines.append(r"\end{figure}")
        elif isinstance(block, TableBlock):
            lines.extend(_latex_table(block))

    lines.append(r"\end{document}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _latex_table(block: TableBlock) -> list[str]:
    col_count = max(len(block.headers), 1)
    col_spec = "|" + "c|" * col_count
    out = [r"\begin{table}[H]", r"  \centering"]
    if block.caption:
        out.append(r"  \caption{" + _latex_escape(block.caption) + "}")
    out.append(r"  \begin{tabular}{" + col_spec + "}")
    out.append(r"    \hline")
    out.append("    " + " & ".join(_latex_inline(h) for h in block.headers) + r" \\")
    out.append(r"    \hline")
    for row in block.rows:
        padded = list(row[:col_count]) + [""] * (col_count - len(row))
        out.append("    " + " & ".join(_latex_inline(v) for v in padded) + r" \\")
    out.append(r"    \hline")
    out.append(r"  \end{tabular}")
    out.append(r"\end{table}")
    return out


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------

_EXPORTERS = {
    "docx": (export_docx, ".docx"),
    "pdf": (export_pdf, ".pdf"),
    "latex": (export_latex, ".tex"),
    "tex": (export_latex, ".tex"),
}

SUPPORTED_FORMATS = ("docx", "pdf", "latex")


def _esc(text: str) -> str:
    """Escape XML-sensitive characters for reportlab paragraphs."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _content_blocks(doc: Document):
    skipped_title = False
    for block in doc.blocks:
        if (
            not skipped_title
            and isinstance(block, Heading)
            and block.level == 1
            and _normalize_title(block.text) == _normalize_title(doc.title)
        ):
            skipped_title = True
            continue
        skipped_title = True
        yield block


def _normalize_title(text: str) -> str:
    return "".join(str(text).split())


def export_document(doc: Document, fmt: str, output_dir: Path, stem: str = "paper") -> Path:
    key = fmt.lower()
    if key not in _EXPORTERS:
        raise ValueError(f"Unsupported export format: {fmt}. Choose from {sorted(set(SUPPORTED_FORMATS))}.")
    func, suffix = _EXPORTERS[key]
    output_path = output_dir / f"{stem}{suffix}"
    return func(doc, output_path)
