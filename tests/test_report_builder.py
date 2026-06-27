from __future__ import annotations

import json

from tools.report_builder import (
    CodeBlock,
    Heading,
    ImageBlock,
    ListBlock,
    Paragraph,
    TableBlock,
    build_appendix_blocks,
    build_document_from_paper,
    parse_markdown,
)

MARKDOWN = """# 标题一

这是一个段落，
跨越两行。

## 二级标题

- 项目一
- 项目二

1. 步骤一
2. 步骤二

```python
print("hello")
```

![图注](figure.png)
"""


def test_parse_markdown_block_types():
    doc = parse_markdown(MARKDOWN)
    kinds = [type(block) for block in doc.blocks]

    assert Heading in kinds
    assert Paragraph in kinds
    assert ListBlock in kinds
    assert CodeBlock in kinds
    assert ImageBlock in kinds


def test_parse_markdown_heading_levels_and_text():
    doc = parse_markdown(MARKDOWN)
    headings = [b for b in doc.blocks if isinstance(b, Heading)]
    assert headings[0].level == 1
    assert headings[0].text == "标题一"
    assert any(h.level == 2 and h.text == "二级标题" for h in headings)


def test_parse_markdown_paragraph_joins_wrapped_lines():
    doc = parse_markdown(MARKDOWN)
    paragraphs = [b for b in doc.blocks if isinstance(b, Paragraph)]
    assert any("跨越两行" in p.text for p in paragraphs)


def test_parse_markdown_lists_ordered_flag():
    doc = parse_markdown(MARKDOWN)
    lists = [b for b in doc.blocks if isinstance(b, ListBlock)]
    assert any(not lst.ordered and "项目一" in lst.items for lst in lists)
    assert any(lst.ordered and "步骤一" in lst.items for lst in lists)


def test_parse_markdown_code_block_content():
    doc = parse_markdown(MARKDOWN)
    code = next(b for b in doc.blocks if isinstance(b, CodeBlock))
    assert 'print("hello")' in code.text


def test_parse_empty_markdown():
    doc = parse_markdown("")
    assert doc.blocks == []


def test_build_appendix_blocks_embeds_figures_and_tables(tmp_path):
    figure = tmp_path / "chart.png"
    figure.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal placeholder bytes
    table = tmp_path / "describe.csv"
    table.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    summary = tmp_path / "run_summary.json"
    summary.write_text(
        json.dumps(
            [
                {
                    "source": str(tmp_path / "data.csv"),
                    "charts": [str(figure)],
                    "describe_table": str(table),
                    "model_outputs": {"trend_forecast": str(table)},
                }
            ]
        ),
        encoding="utf-8",
    )

    blocks = build_appendix_blocks(summary, model_labels={"trend_forecast": "趋势预测"})

    assert any(isinstance(b, ImageBlock) for b in blocks)
    assert any(isinstance(b, TableBlock) for b in blocks)
    # The custom label should be applied to the model output caption.
    table_captions = [b.caption for b in blocks if isinstance(b, TableBlock)]
    assert any("趋势预测" in caption for caption in table_captions)


def test_build_appendix_blocks_missing_summary(tmp_path):
    assert build_appendix_blocks(tmp_path / "missing.json") == []


def test_build_document_from_paper(tmp_path):
    paper = tmp_path / "paper_draft.md"
    paper.write_text("# 论文\n\n正文内容。", encoding="utf-8")
    doc = build_document_from_paper(paper, title="测试论文")
    assert doc.title == "测试论文"
    assert any(isinstance(b, Heading) for b in doc.blocks)
