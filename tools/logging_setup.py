"""Centralized logging and observability for the math modeling agent."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable


def setup_logging(log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure a root logger that writes to both console and a rotating file.

    Returns the project-level logger ``mma`` so agents can do
    ``logging.getLogger("mma")`` or ``logging.getLogger("mma.coding_agent")``.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("mma")
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls (e.g. Streamlit reruns).
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S"
    )

    # File handler
    file_handler = logging.FileHandler(log_dir / "agent.log", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # Console handler (quieter)
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(fmt)
    logger.addHandler(console)

    return logger


@contextmanager
def timed(log: logging.Logger, label: str):
    """Context manager that logs the wall-clock duration of a block.

    Usage::

        with timed(log, "model_selection"):
            state = model_selection_agent.run(state)

    Logs ``label completed in X.XXs`` at INFO level.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.info("%s completed in %.2fs", label, elapsed)


def timed_decorator(label: str) -> Callable:
    """Decorator that times a function call via ``timed``.

    Usage::

        @timed_decorator("coding_agent.run")
        def run(self, state):
            ...
    """
    import functools

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log = logging.getLogger("mma")
            with timed(log, label):
                return func(*args, **kwargs)

        return wrapper

    return decorator
