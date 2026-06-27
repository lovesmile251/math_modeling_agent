from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PdfPageCheck:
    page: int
    screenshot: str
    width: int
    height: int
    ink_ratio: float
    dark_ratio: float
    blank: bool


@dataclass(frozen=True)
class PdfLayoutCheck:
    passed: bool
    renderer: str
    pages_checked: int
    screenshots_dir: str
    pages: list[PdfPageCheck]
    warnings: list[str]

    def to_dict(self) -> dict:
        payload = asdict(self)
        return payload


def check_pdf_render_layout(
    pdf_path: Path,
    output_dir: Path,
    *,
    max_pages: int = 5,
    zoom: float = 1.2,
) -> PdfLayoutCheck:
    """Render PDF pages to screenshots and detect obvious layout failures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        return _check_with_fitz(pdf_path, output_dir, max_pages=max_pages, zoom=zoom)
    except Exception as exc:
        return PdfLayoutCheck(
            passed=False,
            renderer="fitz",
            pages_checked=0,
            screenshots_dir=str(output_dir),
            pages=[],
            warnings=[f"PDF render check failed: {exc}"],
        )


def write_pdf_layout_report(report: PdfLayoutCheck, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _check_with_fitz(pdf_path: Path, output_dir: Path, *, max_pages: int, zoom: float) -> PdfLayoutCheck:
    import fitz
    from PIL import Image

    doc = fitz.open(str(pdf_path))
    warnings: list[str] = []
    pages: list[PdfPageCheck] = []
    matrix = fitz.Matrix(zoom, zoom)
    for page_index in range(min(len(doc), max_pages)):
        page = doc.load_page(page_index)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        screenshot = output_dir / f"{pdf_path.stem}_page_{page_index + 1:03d}.png"
        pixmap.save(str(screenshot))
        with Image.open(screenshot) as image:
            width, height = image.size
            gray = image.convert("L")
            pixels = list(gray.getdata())
        total = max(len(pixels), 1)
        ink_ratio = sum(1 for value in pixels if value < 245) / total
        dark_ratio = sum(1 for value in pixels if value < 35) / total
        blank = ink_ratio < 0.001
        if blank:
            warnings.append(f"page {page_index + 1}: rendered screenshot is nearly blank")
        if dark_ratio > 0.55:
            warnings.append(f"page {page_index + 1}: rendered screenshot is excessively dark")
        pages.append(
            PdfPageCheck(
                page=page_index + 1,
                screenshot=str(screenshot),
                width=width,
                height=height,
                ink_ratio=round(ink_ratio, 6),
                dark_ratio=round(dark_ratio, 6),
                blank=blank,
            )
        )
    if len(doc) == 0:
        warnings.append("PDF has no pages")
    return PdfLayoutCheck(
        passed=not warnings,
        renderer="fitz",
        pages_checked=len(pages),
        screenshots_dir=str(output_dir),
        pages=pages,
        warnings=warnings,
    )
