from __future__ import annotations

from agents.language_reviewer import LanguageReviewerAgent


def test_language_reviewer_placeholder_examples_are_sliceable():
    issues = LanguageReviewerAgent._check_language("待补充，待补充，TODO，TODO")

    assert issues
    assert "发现占位符" in issues[0]["description"]
