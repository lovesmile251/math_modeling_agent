from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
import re
from uuid import uuid4


@dataclass(frozen=True)
class WorkspaceConfig:
    root: Path
    input_dir: Path
    code_dir: Path
    data_dir: Path
    figures_dir: Path
    tables_dir: Path
    paper_dir: Path
    logs_dir: Path
    project_root: Path | None = None

    @classmethod
    def from_root(cls, root: Path, project_root: Path | None = None) -> "WorkspaceConfig":
        return cls(
            root=root,
            input_dir=root / "input",
            code_dir=root / "code",
            data_dir=root / "data",
            figures_dir=root / "figures",
            tables_dir=root / "tables",
            paper_dir=root / "paper",
            logs_dir=root / "logs",
            project_root=project_root,
        )

    @classmethod
    def from_project_root(cls, project_root: Path) -> "WorkspaceConfig":
        return cls.from_root(project_root / "workspace", project_root=project_root)

    @classmethod
    def isolated_run(cls, project_root: Path, run_id: str | None = None) -> "WorkspaceConfig":
        """Create a per-run workspace under ``workspace/runs``.

        The project root is carried explicitly because generated scripts may
        live more than one directory below the repository root.
        """
        safe_run_id = _safe_run_id(run_id or _timestamp_run_id())
        workspace = project_root / "workspace" / "runs" / safe_run_id
        return cls.from_root(workspace, project_root=project_root)

    def ensure_dirs(self) -> None:
        for path in (
            self.root,
            self.input_dir,
            self.code_dir,
            self.data_dir,
            self.figures_dir,
            self.tables_dir,
            self.paper_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def effective_project_root(self) -> Path:
        return self.project_root or self.root.parent


def _timestamp_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid4().hex[:8]}"


def _safe_run_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    cleaned = cleaned.strip(".-")
    if not cleaned:
        return _timestamp_run_id()
    return cleaned[:80]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = WorkspaceConfig.from_project_root(PROJECT_ROOT)
