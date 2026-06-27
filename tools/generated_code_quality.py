"""Generated code quality gate — pre-flight checks before execution.

Runs py_compile + ruff check (+ optional import check) on the generated
``baseline_analysis.py`` so syntax and lint errors surface before the
execution agent runs the script.  If the gate fails, the CodeRepairAgent
gets a chance to fix the code instead of waiting for a subprocess crash.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


_ALLOWED_IMPORT_ROOTS = {
    "__future__",
    "json",
    "matplotlib",
    "models",
    "numpy",
    "pandas",
    "pathlib",
    "random",
    "sys",
    "time",
}
_BLOCKED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "input",
}
_BLOCKED_ATTRIBUTES = {
    "chmod",
    "connect",
    "hardlink_to",
    "home",
    "modules",
    "open",
    "read_bytes",
    "read_html",
    "read_json",
    "read_pickle",
    "read_text",
    "read_xml",
    "rename",
    "replace",
    "rmdir",
    "symlink_to",
    "to_pickle",
    "urlopen",
    "write_bytes",
}


@dataclass(frozen=True)
class GeneratedCodeQuality:
    """Result of running the quality gate on a single generated script."""

    syntax_ok: bool
    security_ok: bool = True
    ruff_ok: bool | None = None   # None = ruff not installed / skipped
    issues: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.syntax_ok and self.security_ok and (self.ruff_ok is not False)

    def summary(self) -> str:
        parts = [
            f"syntax={'OK' if self.syntax_ok else 'FAIL'}",
            f"security={'OK' if self.security_ok else 'FAIL'}",
        ]
        if self.ruff_ok is not None:
            parts.append(f"ruff={'OK' if self.ruff_ok else 'FAIL'}")
        if self.issues:
            parts.append(f"issues={len(self.issues)}")
        return "; ".join(parts)


def check_generated_code(script_path: Path) -> GeneratedCodeQuality:
    """Run py_compile and ruff check on *script_path*.

    Returns a ``GeneratedCodeQuality`` with findings.  This is designed to
    be called from ``ExecutionAgent`` *before* attempting execution.
    """
    issues: list[str] = []
    syntax_ok = _check_py_compile(script_path, issues)
    security_ok = _check_security(script_path, issues) if syntax_ok else False
    ruff_ok = _check_ruff(script_path, issues)
    return GeneratedCodeQuality(
        syntax_ok=syntax_ok,
        security_ok=security_ok,
        ruff_ok=ruff_ok,
        issues=issues,
    )


# ── internal helpers ────────────────────────────────────────────────────────

def _check_py_compile(path: Path, issues: list[str]) -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        issues.append(f"py_compile: {result.stderr.strip()}")
        return False
    return True


def _check_ruff(path: Path, issues: list[str]) -> bool | None:
    """Run ``ruff check``.  Returns None if ruff is not available."""
    executable = shutil.which("ruff")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "check", str(path)],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        return None  # ruff not installed; not a failure

    if result.returncode != 0:
        # Only report first 5 lint issues to keep output manageable
        lines = result.stdout.strip().splitlines()
        issues.extend(lines[:5])
        if len(lines) > 5:
            issues.append(f"... and {len(lines) - 5} more ruff issues")
        return False
    return True


def _check_security(path: Path, issues: list[str]) -> bool:
    """Reject generated code that can launch commands or access the network.

    Generated analysis scripts are intentionally limited to deterministic
    data processing and project model imports.  This is a security gate, not
    a replacement for an OS sandbox.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        issues.append(f"security scan: {exc}")
        return False

    security_issues: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".", 1)[0] for alias in node.names]
            for root in roots:
                if root not in _ALLOWED_IMPORT_ROOTS:
                    security_issues.append(
                        f"security: import {root!r} is not allowed (line {node.lineno})"
                    )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in _ALLOWED_IMPORT_ROOTS:
                security_issues.append(
                    f"security: import from {root!r} is not allowed (line {node.lineno})"
                )
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _BLOCKED_CALLS:
                security_issues.append(
                    f"security: call to {node.func.id}() is not allowed (line {node.lineno})"
                )
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in _BLOCKED_ATTRIBUTES:
                security_issues.append(
                    f"security: call to .{node.func.attr}() is not allowed (line {node.lineno})"
                )
        elif isinstance(node, ast.Attribute) and node.attr == "modules":
            security_issues.append(
                f"security: access to .modules is not allowed (line {node.lineno})"
            )

    issues.extend(security_issues[:10])
    if len(security_issues) > 10:
        issues.append(f"... and {len(security_issues) - 10} more security issues")
    return not security_issues
