from __future__ import annotations

from tools.paper_templates.national_contest import (
    NATIONAL_CONTEST_REQUIRED_SECTIONS,
    build_national_contest_document,
    build_national_contest_markdown,
    export_national_contest_template,
)
from tools.report_builder import Heading, TableBlock


def test_national_contest_markdown_contains_required_sections():
    markdown = build_national_contest_markdown("contest paper")

    assert markdown.startswith("# contest paper")
    for section in NATIONAL_CONTEST_REQUIRED_SECTIONS:
        assert section in markdown
    assert "| 符号 | 含义 | 单位/说明 |" in markdown
    assert "[1] 作者. 文献题名[J]. 期刊名, 年份, 卷(期): 起止页码." in markdown
    assert "附录A 主要程序" in markdown


def test_national_contest_document_keeps_symbol_table_and_headings():
    doc = build_national_contest_document("contest paper")

    headings = [block.text for block in doc.blocks if isinstance(block, Heading)]
    tables = [block for block in doc.blocks if isinstance(block, TableBlock)]

    assert doc.title == "contest paper"
    assert "摘要" in headings
    assert "四、符号说明" in headings
    assert "参考文献" in headings
    assert tables
    assert tables[0].headers == ["符号", "含义", "单位/说明"]


def test_export_national_contest_template_writes_markdown_and_latex(tmp_path):
    outputs = export_national_contest_template(tmp_path, formats=["latex"], title="contest paper")

    assert outputs["markdown"].name == "national_contest_template.md"
    assert outputs["latex"].name == "national_contest_template.tex"
    assert outputs["markdown"].exists()
    tex = outputs["latex"].read_text(encoding="utf-8")
    assert r"\usepackage{ctex}" in tex
    assert "contest paper" in tex
