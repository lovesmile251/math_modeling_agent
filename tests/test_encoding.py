from __future__ import annotations

from pathlib import Path

import pytest

from tools.encoding import (
    decode_strict_utf8,
    find_mojibake_markers,
    scan_project_text_encoding,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_text_files_are_plain_utf8_without_mojibake():
    problems = scan_project_text_encoding(PROJECT_ROOT)

    assert not problems, "Encoding problems found:\n" + "\n".join(problems)


def test_mojibake_detector_catches_replacement_and_common_markers():
    text = "\ufffd \u93c1\u677f \u00e4\u00b8\u00ad\u00e6\u2013\u2021"

    markers = find_mojibake_markers(text)

    assert "\ufffd" in markers
    assert "\u93c1" in markers
    assert "\u00e4\u00b8" in markers


def test_decode_strict_utf8_rejects_bom(tmp_path: Path):
    path = tmp_path / "bom.txt"
    path.write_bytes(b"\xef\xbb\xbfhello")

    with pytest.raises(UnicodeError, match="BOM"):
        decode_strict_utf8(path)
