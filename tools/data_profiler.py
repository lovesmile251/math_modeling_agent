from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataFileSummary:
    path: Path
    file_type: str
    note: str


def summarize_data_files(paths: list[Path]) -> list[DataFileSummary]:
    summaries: list[DataFileSummary] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            note = "tabular text data"
        elif suffix in {".xlsx", ".xls"}:
            note = "spreadsheet data"
        else:
            note = "unsupported or auxiliary file"
        summaries.append(DataFileSummary(path=path, file_type=suffix.lstrip("."), note=note))
    return summaries
