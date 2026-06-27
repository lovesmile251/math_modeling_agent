from __future__ import annotations

import re
from pathlib import Path


def test_all_literal_prompt_references_exist():
    project_root = Path(__file__).resolve().parents[1]
    referenced: set[str] = set()
    pattern = re.compile(r'load_prompt\("([^"]+)"\)')
    for root_name in ("agents", "tools"):
        for path in (project_root / root_name).rglob("*.py"):
            referenced.update(pattern.findall(path.read_text(encoding="utf-8")))

    missing = sorted(
        name for name in referenced if not (project_root / "prompts" / name).exists()
    )
    assert missing == []
