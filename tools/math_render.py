from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RenderedEquation:
    path: Path
    width_pt: float
    height_pt: float


def render_latex(latex: str, output_dir: Path, *, display: bool = False) -> RenderedEquation | None:
    """Render a LaTeX math fragment to a transparent PNG for PDF embedding.

    This intentionally uses matplotlib's built-in mathtext engine instead of a
    system LaTeX installation, so PDF export works on a clean Python setup.
    """

    expr = _normalise_latex(latex)
    if not expr:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(f"{display}:{expr}".encode("utf-8")).hexdigest()[:16]
    path = output_dir / f"eq_{digest}.png"
    dpi = 220
    fontsize = 13 if display else 10.5

    if not path.exists():
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return None

        fig = plt.figure(figsize=(0.01, 0.01), dpi=dpi)
        fig.patch.set_alpha(0)
        try:
            fig.text(0, 0, f"${expr}$", fontsize=fontsize, color="black")
            fig.savefig(path, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.02)
        except Exception:
            plt.close(fig)
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            return None
        finally:
            plt.close(fig)

    try:
        from PIL import Image

        with Image.open(path) as image:
            width_px, height_px = image.size
    except Exception:
        return None
    return RenderedEquation(path=path, width_pt=width_px * 72 / dpi, height_pt=height_px * 72 / dpi)


def _normalise_latex(latex: str) -> str:
    expr = latex.strip()
    for left, right in (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$")):
        if expr.startswith(left) and expr.endswith(right):
            expr = expr[len(left) : -len(right)].strip()
            break
    expr = expr.replace(r"\left", "").replace(r"\right", "")
    return expr.strip()
