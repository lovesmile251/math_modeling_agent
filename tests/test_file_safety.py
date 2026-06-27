from __future__ import annotations

import zipfile

import pytest

from tools.file_tool import validate_data_file


def test_xlsx_zip_bomb_is_rejected(tmp_path):
    path = tmp_path / "bomb.xlsx"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/worksheets/sheet1.xml", "0" * 1_000_000)

    with pytest.raises(ValueError, match="compression ratio"):
        validate_data_file(path)
