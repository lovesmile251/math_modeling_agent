from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PdfPageCheck:
    page: int
    screenshot: str
    width: int
    height: int
    ink_ratio: float
    dark_ratio: float
    edge_ink_ratio: float
    content_bbox: dict[str, int]
    content_margin_px: int
    edge_contact: bool
    large_dark_block_ratio: float
    blank: bool


@dataclass(frozen=True)
class PdfLayoutCheck:
    passed: bool
    renderer: str
    pages_checked: int
    screenshots_dir: str
    pages: list[PdfPageCheck]
    warnings: list[str]
    screenshot_manifest: list[str]

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
            screenshot_manifest=[],
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
        metrics = _image_layout_metrics(screenshot)
        width = metrics["width"]
        height = metrics["height"]
        ink_ratio = metrics["ink_ratio"]
        dark_ratio = metrics["dark_ratio"]
        blank = ink_ratio < 0.001
        if blank:
            warnings.append(f"page {page_index + 1}: rendered screenshot is nearly blank")
        if dark_ratio > 0.55:
            warnings.append(f"page {page_index + 1}: rendered screenshot is excessively dark")
        if metrics["edge_contact"] and metrics["edge_ink_ratio"] > 0.006:
            warnings.append(f"page {page_index + 1}: content touches the rendered page boundary")
        if metrics["large_dark_block_ratio"] > 0.35:
            warnings.append(f"page {page_index + 1}: large dark block may hide text or figures")
        pages.append(
            PdfPageCheck(
                page=page_index + 1,
                screenshot=str(screenshot),
                width=width,
                height=height,
                ink_ratio=round(ink_ratio, 6),
                dark_ratio=round(dark_ratio, 6),
                edge_ink_ratio=round(metrics["edge_ink_ratio"], 6),
                content_bbox=metrics["content_bbox"],
                content_margin_px=int(metrics["content_margin_px"]),
                edge_contact=bool(metrics["edge_contact"]),
                large_dark_block_ratio=round(metrics["large_dark_block_ratio"], 6),
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
        screenshot_manifest=[page.screenshot for page in pages],
    )


def _image_layout_metrics(screenshot: Path) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    with Image.open(screenshot) as image:
        width, height = image.size
        gray = image.convert("L")
        arr = np.asarray(gray)

    total = max(int(arr.size), 1)
    ink_mask = arr < 245
    dark_mask = arr < 35
    ink_ratio = float(ink_mask.sum() / total)
    dark_ratio = float(dark_mask.sum() / total)
    bbox, content_margin_px = _content_bbox(ink_mask, width, height)
    edge_band = max(4, min(width, height) // 50)
    edge_mask = np.zeros_like(ink_mask, dtype=bool)
    edge_mask[:edge_band, :] = True
    edge_mask[-edge_band:, :] = True
    edge_mask[:, :edge_band] = True
    edge_mask[:, -edge_band:] = True
    edge_pixels = max(int(edge_mask.sum()), 1)
    edge_ink_ratio = float((ink_mask & edge_mask).sum() / edge_pixels)
    edge_contact = content_margin_px <= edge_band and ink_ratio >= 0.01
    large_dark_block_ratio = _largest_dark_bbox_ratio(dark_mask, width, height)
    return {
        "width": width,
        "height": height,
        "ink_ratio": ink_ratio,
        "dark_ratio": dark_ratio,
        "edge_ink_ratio": edge_ink_ratio,
        "content_bbox": bbox,
        "content_margin_px": content_margin_px,
        "edge_contact": edge_contact,
        "large_dark_block_ratio": large_dark_block_ratio,
    }


def _content_bbox(mask, width: int, height: int) -> tuple[dict[str, int], int]:
    import numpy as np

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        bbox = {"x0": 0, "y0": 0, "x1": 0, "y1": 0}
        return bbox, min(width, height)
    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())
    bbox = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
    margin = min(x0, y0, max(width - 1 - x1, 0), max(height - 1 - y1, 0))
    return bbox, int(margin)


def _largest_dark_bbox_ratio(mask, width: int, height: int) -> float:
    import numpy as np

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return 0.0
    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())
    bbox_area = max((x1 - x0 + 1) * (y1 - y0 + 1), 0)
    return float(bbox_area / max(width * height, 1))
