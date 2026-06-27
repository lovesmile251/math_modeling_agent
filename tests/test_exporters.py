from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

from tools.exporters import export_document, export_docx, export_latex, export_pdf
from tools.report_builder import (
    CodeBlock,
    Document,
    Heading,
    ImageBlock,
    ListBlock,
    MathBlock,
    Paragraph,
    TableBlock,
)


@pytest.fixture
def sample_png(tmp_path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = tmp_path / "fig.png"
    fig, ax = plt.subplots(figsize=(2, 1.5))
    ax.plot([0, 1, 2], [1, 3, 2])
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path


@pytest.fixture
def sample_document(sample_png) -> Document:
    return Document(
        title="测试论文",
        blocks=[
            Heading(level=1, text="一、引言"),
            Paragraph(text="这是一段包含特殊字符 & < > % _ 的中文正文，并包含公式 $a_{ij}=1$。"),
            MathBlock(latex=r"\rho(S)=\frac{2|E(S)|}{|S|(|S|-1)}"),
            ListBlock(items=["要点一", "要点二"], ordered=False),
            ListBlock(items=["步骤一", "步骤二"], ordered=True),
            CodeBlock(text="print('hi')"),
            ImageBlock(path=sample_png, caption="图1 示例曲线"),
            TableBlock(headers=["指标", "数值"], rows=[["均值", "1.5"], ["方差", "0.7"]], caption="表1 统计"),
        ],
    )


def test_export_docx(sample_document, tmp_path):
    out = export_docx(sample_document, tmp_path / "paper.docx")
    assert out.exists() and out.stat().st_size > 0

    from docx import Document as DocxDocument

    docx = DocxDocument(str(out))
    text = "\n".join(p.text for p in docx.paragraphs)
    assert "测试论文" in text
    assert "一、引言" in text
    assert len(docx.tables) == 1
    assert docx.tables[0].rows[0].cells[0].text == "指标"

    with ZipFile(out) as package:
        document_xml = package.read("word/document.xml").decode("utf-8")
        # Math is rendered as embedded PNG images (no OMML leaks)
        assert not any("<m:oMath" in name.decode() if isinstance(name, bytes) else "<m:oMath" in name for name in package.namelist())
        assert "<m:oMath" not in document_xml  # no raw OMML tags leak
        assert any(name.startswith("word/media/") for name in package.namelist())  # images embedded


def test_export_pdf(sample_document, tmp_path):
    out = export_pdf(sample_document, tmp_path / "paper.pdf")
    assert out.exists()
    header = out.read_bytes()[:5]
    assert header == b"%PDF-"
    assert list((tmp_path / "_equations").glob("eq_*.png"))


def test_export_latex(sample_document, tmp_path):
    out = export_latex(sample_document, tmp_path / "paper.tex")
    content = out.read_text(encoding="utf-8")
    assert r"\documentclass" in content
    assert r"\usepackage{ctex}" in content
    assert r"\section*{一、引言}" in content
    # special characters must be escaped
    assert r"\&" in content and r"\%" in content and r"\_" in content
    assert r"\includegraphics" in content
    assert r"\begin{tabular}" in content


def test_export_latex_preserves_markdown_math(tmp_path):
    doc = Document(
        title="公式测试",
        blocks=[Paragraph(text=r"构建无向社交网络 \(G=(V,E)\)，模块度 \(Q=0.2576\)。")],
    )

    out = export_latex(doc, tmp_path / "paper.tex")
    content = out.read_text(encoding="utf-8")

    assert "$G=(V,E)$" in content
    assert "$Q=0.2576$" in content
    assert r"\textbackslash{}(" not in content


def test_export_docx_converts_markdown_inline_math(tmp_path):
    doc = Document(
        title="公式测试",
        blocks=[Paragraph(text=r"内部连接密度 \(\rho=\frac{2|E_C|}{|C|(|C|-1)}\)。")],
    )

    out = export_docx(doc, tmp_path / "paper.docx")
    with ZipFile(out) as package:
        document_xml = package.read("word/document.xml").decode("utf-8")

    assert r"\rho" not in document_xml
    assert r"\(" not in document_xml
    # Math rendered as image, no OMML tags leak
    assert "<m:oMath" not in document_xml
    assert any(name.startswith("word/media/") for name in package.namelist())


def test_export_document_dispatch(sample_document, tmp_path):
    for fmt, suffix in (("docx", ".docx"), ("pdf", ".pdf"), ("latex", ".tex")):
        out = export_document(sample_document, fmt, tmp_path, stem="report")
        assert out.name == f"report{suffix}"
        assert out.exists()


def test_export_document_invalid_format(sample_document, tmp_path):
    with pytest.raises(ValueError):
        export_document(sample_document, "rtf", tmp_path)


def test_export_latex_skips_duplicate_markdown_title(tmp_path):
    doc = Document(title="同名标题", blocks=[Heading(level=1, text="同名标题"), Paragraph(text="正文。")])

    out = export_latex(doc, tmp_path / "paper.tex")
    content = out.read_text(encoding="utf-8")

    assert content.count("同名标题") == 1
    assert r"\section*{同名标题}" not in content


def test_export_handles_missing_image(tmp_path):
    doc = Document(title="无图", blocks=[ImageBlock(path=tmp_path / "nope.png", caption="缺失")])
    # None of the backends should raise when the referenced image is absent.
    assert export_docx(doc, tmp_path / "a.docx").exists()
    assert export_pdf(doc, tmp_path / "a.pdf").exists()
    assert export_latex(doc, tmp_path / "a.tex").exists()
