from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from tools.file_tool import SUPPORTED_DATA_SUFFIXES, read_problem_file, write_text


_STATEMENT_SUFFIXES = {".pdf", ".docx"}
_EXCLUDED_NAME_TERMS = (
    "format",
    "论文格式",
    "附件",
    "appendix",
    "承诺书",
)


@dataclass(frozen=True)
class CorpusCase:
    case_id: str
    year: int
    problem: str
    title: str
    statement_path: str
    attachment_paths: tuple[str, ...]
    statement_chars: int

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["attachment_paths"] = list(self.attachment_paths)
        return payload


def discover_competition_cases(root: Path) -> list[CorpusCase]:
    """Discover CUMCM problem statements and their local tabular attachments."""
    root = root.resolve()
    candidates: dict[tuple[int, str], list[tuple[Path, str]]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _STATEMENT_SUFFIXES:
            continue
        if _is_excluded_statement(path):
            continue
        try:
            text = read_problem_file(path)
        except (OSError, RuntimeError, ValueError):
            continue
        year = _detect_year(path, text)
        problem = _detect_problem_letter(path, text)
        if year is None or problem is None:
            continue
        candidates.setdefault((year, problem), []).append((path, text))

    cases: list[CorpusCase] = []
    for (year, problem), options in sorted(candidates.items()):
        statement_path, text = max(options, key=_statement_preference)
        case_dir = statement_path.parent
        attachments = tuple(
            str(path.resolve().relative_to(root))
            for path in sorted(case_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in SUPPORTED_DATA_SUFFIXES
        )
        cases.append(
            CorpusCase(
                case_id=f"cumcm-{year}-{problem.lower()}",
                year=year,
                problem=problem,
                title=_extract_title(text, problem),
                statement_path=str(statement_path.resolve().relative_to(root)),
                attachment_paths=attachments,
                statement_chars=len(text),
            )
        )
    return cases


def write_corpus_index(cases: Iterable[CorpusCase], output_path: Path) -> Path:
    payload = [case.to_dict() for case in cases]
    return write_text(
        output_path,
        json.dumps(payload, ensure_ascii=False, indent=2),
    )


def load_corpus_index(path: Path) -> list[CorpusCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        CorpusCase(
            case_id=str(item["case_id"]),
            year=int(item["year"]),
            problem=str(item["problem"]),
            title=str(item["title"]),
            statement_path=str(item["statement_path"]),
            attachment_paths=tuple(item.get("attachment_paths", [])),
            statement_chars=int(item.get("statement_chars", 0)),
        )
        for item in payload
    ]


def _is_excluded_statement(path: Path) -> bool:
    normalized = str(path).lower()
    return any(term in normalized for term in _EXCLUDED_NAME_TERMS)


def _detect_year(path: Path, text: str) -> int | None:
    header_match = re.search(
        r"(20(?:0\d|1\d|2\d))\s*年.{0,30}(?:数学建模|竞赛题目)",
        text[:1500],
    )
    if header_match:
        return int(header_match.group(1))

    for parent in path.parents[:3]:
        try:
            format_files = parent.glob("format20*")
        except OSError:
            continue
        for format_path in format_files:
            match = re.search(r"(20(?:0\d|1\d|2\d))", format_path.name)
            if match:
                return int(match.group(1))

    path_match = re.search(r"(20(?:0\d|1\d|2\d))", str(path))
    if path_match:
        return int(path_match.group(1))
    text_match = re.search(r"(20(?:0\d|1\d|2\d))", text[:1000])
    return int(text_match.group(1)) if text_match else None


def _detect_problem_letter(path: Path, text: str) -> str | None:
    path_patterns = (
        r"(?:^|[\\/])([A-E])题(?:[\\/]|\.|$)",
        r"(?:^|[\\/])([A-E])-\d{4}",
        r"problem[-_ ]?([A-E])",
        r"cumcm\d{4}[-_ ]?([A-E])",
        r"\d{4}([A-E])(?:[-_.]|$)",
    )
    normalized_path = str(path)
    for pattern in path_patterns:
        match = re.search(pattern, normalized_path, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    match = re.search(r"(?:问题\s*)?([A-E])\s*题", text[:2000], re.IGNORECASE)
    return match.group(1).upper() if match else None


def _extract_title(text: str, problem: str) -> str:
    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.replace("\r", "\n").splitlines()
    ]
    lines = [line for line in lines if line]
    for index, line in enumerate(lines[:40]):
        match = re.search(
            rf"^(?:(?:问题\s*)?{problem}\s*题|问题\s*{problem})\s*(.*)$",
            line,
            re.IGNORECASE,
        )
        if match:
            title = match.group(1).strip(" ：:　")
            if not title and index + 1 < len(lines):
                title = lines[index + 1].strip(" ：:　")
            title = re.split(r"（请先阅读|请建立数学模型|问题\s*1", title)[0]
            if title:
                return title[:80]
    return f"{problem}题"


def _statement_preference(item: tuple[Path, str]) -> tuple[int, int]:
    path, text = item
    suffix_score = {".pdf": 3, ".docx": 2}.get(path.suffix.lower(), 0)
    return suffix_score, len(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an index of real competition cases.")
    parser.add_argument("root", type=Path, help="Root directory containing extracted cases.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/real_competition_corpus.json"),
    )
    args = parser.parse_args()
    cases = discover_competition_cases(args.root)
    write_corpus_index(cases, args.output)
    print(f"Indexed {len(cases)} cases into {args.output}")


if __name__ == "__main__":
    main()
