from __future__ import annotations

from app.streamlit_app import _safe_upload_name


def test_upload_name_strips_path_components():
    assert _safe_upload_name("../../secret.csv") == "secret.csv"
    assert _safe_upload_name(r"..\..\secret.csv") == "secret.csv"


def test_upload_name_rejects_empty_basename():
    try:
        _safe_upload_name("..")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid upload name was accepted")
