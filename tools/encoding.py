from __future__ import annotations

import sys
from pathlib import Path


TEXT_ROOTS = ("agents", "app", "models", "prompts", "tests", "tools", "workflows")
TEXT_FILES = ("README.md", "pyproject.toml", "requirements.txt", "requirements-dev.txt")
TEXT_SUFFIXES = {".py", ".md", ".toml", ".txt", ".json", ".yaml", ".yml"}

UTF8_BOM = b"\xef\xbb\xbf"

MOJIBAKE_MARKERS = (
    "\ufffd",  # Unicode replacement character.
    "\ue000",
    "\ue4b7",
    "\ue5b9",
    "\u93c1",
    "\u7027",
    "\u59ab",
    "\u93c9",
    "\u68e3",
    "\u95b3",
    "\u951b",
    "\u9286",
    "\u9422",
    "\u8b81",
    "\u9359",
    "\u6d63",
    "\u5bee",
    "\u0101",
    "\u00c3",
    "\u00c2",
    "\u00e2\u20ac",
    "\u00e4\u00b8",
    "\u00e5\u203a",
    "\u00e6\u2022",
    "\u00e7\u0161",
)


def configure_utf8_stdio() -> None:
    """Prefer UTF-8 for CLI output when stdout/stderr are redirected or captured."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            # Some embedded or test streams do not allow reconfiguration.
            continue


def iter_project_text_files(project_root: Path) -> list[Path]:
    """Return repository text files that should be strict UTF-8."""

    files: list[Path] = []
    for root in TEXT_ROOTS:
        root_path = project_root / root
        if not root_path.exists():
            continue
        files.extend(
            path
            for path in root_path.rglob("*")
            if path.is_file()
            and path.suffix.lower() in TEXT_SUFFIXES
            and "__pycache__" not in path.parts
        )

    files.extend(project_root / name for name in TEXT_FILES)
    return sorted({path for path in files if path.exists()})


def decode_strict_utf8(path: Path) -> str:
    """Read a text file as UTF-8 and reject BOM-prefixed files."""

    data = path.read_bytes()
    if data.startswith(UTF8_BOM):
        raise UnicodeError("UTF-8 BOM is not allowed; save as plain UTF-8")
    return data.decode("utf-8")


def find_mojibake_markers(text: str) -> list[str]:
    """Find common markers from UTF-8/GBK/Latin-1 mojibake."""

    markers = [marker for marker in MOJIBAKE_MARKERS if marker in text]
    markers.extend(
        f"private-use:{ord(char):04x}"
        for char in text
        if "\ue000" <= char <= "\uf8ff" and char not in MOJIBAKE_MARKERS
    )
    return sorted(set(markers))


def scan_project_text_encoding(project_root: Path) -> list[str]:
    """Return human-readable encoding problems for project text files."""

    problems: list[str] = []
    for path in iter_project_text_files(project_root):
        try:
            text = decode_strict_utf8(path)
        except UnicodeError as exc:
            problems.append(f"{path}: {exc}")
            continue

        markers = find_mojibake_markers(text)
        if markers:
            escaped = ", ".join(marker.encode("unicode_escape").decode("ascii") for marker in markers)
            problems.append(f"{path}: mojibake markers: {escaped}")
    return problems
