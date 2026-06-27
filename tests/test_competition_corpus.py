from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfWriter
from reportlab.pdfgen import canvas

from tools.competition_corpus import (
    CorpusCase,
    _extract_title,
    discover_competition_cases,
    load_corpus_index,
    write_corpus_index,
)
from tools.file_tool import SUPPORTED_PROBLEM_SUFFIXES, read_pdf_text


def test_pdf_is_supported_problem_format():
    assert ".pdf" in SUPPORTED_PROBLEM_SUFFIXES


def test_pdf_text_can_be_read_from_memory():
    payload = BytesIO()
    document = canvas.Canvas(payload)
    document.drawString(72, 720, "Competition problem statement")
    document.save()
    payload.seek(0)

    assert "Competition problem statement" in read_pdf_text(payload)


def test_corpus_index_round_trip(tmp_path):
    cases = [
        CorpusCase(
            case_id="cumcm-2024-a",
            year=2024,
            problem="A",
            title="测试题",
            statement_path="2024/A题/A题.pdf",
            attachment_paths=("2024/A题/附件.xlsx",),
            statement_chars=1234,
        )
    ]
    path = write_corpus_index(cases, tmp_path / "index.json")

    assert load_corpus_index(path) == cases


def test_discovery_ignores_empty_pdf(tmp_path):
    case_dir = tmp_path / "2024" / "A题"
    case_dir.mkdir(parents=True)
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with (case_dir / "A题.pdf").open("wb") as stream:
        writer.write(stream)

    assert discover_competition_cases(tmp_path) == []


def test_extract_title_handles_pdf_line_breaks():
    text = "2024 年竞赛题目\nA 题 “板凳龙” 闹元宵\n请建立数学模型"

    assert _extract_title(text, "A") == "“板凳龙” 闹元宵"


def test_extract_title_handles_problem_letter_prefix():
    text = "2019 年竞赛题目\n问题C 机场的出租车问题\n大多数乘客下飞机后"

    assert _extract_title(text, "C") == "机场的出租车问题"
