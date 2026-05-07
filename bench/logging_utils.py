"""Structured logging and per-run log file."""

import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_state = threading.local()


def _get_run_log_path() -> Optional[str]:
    return getattr(_state, "run_log_path", None)


def _set_run_log_path(path: Optional[str]) -> None:
    _state.run_log_path = path


def _get_run_log_file() -> Optional[Any]:
    return getattr(_state, "run_log_file", None)


def _set_run_log_file(handle: Optional[Any]) -> None:
    _state.run_log_file = handle


def configure_root_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a consistent format."""
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root.addHandler(h)


def set_run_log_path(path: str) -> None:
    """Set the path for the current run's log file. Call at start of a run."""
    current_file = _get_run_log_file()
    if current_file is not None:
        try:
            current_file.close()
        except Exception:
            pass
        _set_run_log_file(None)
    _set_run_log_path(path)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        _set_run_log_file(open(path, "a", encoding="utf-8"))


def clear_run_log_path() -> None:
    """Clear run log path and close file."""
    current_file = _get_run_log_file()
    if current_file is not None:
        try:
            current_file.close()
        except Exception:
            pass
        _set_run_log_file(None)
    _set_run_log_path(None)


def run_log(
    event: str,
    level: str = "info",
    run_id: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Write a structured log line to the run log file (and optionally stdout)."""
    payload = {
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "event": event,
        "level": level,
        **kwargs,
    }
    if run_id is not None:
        payload["run_id"] = run_id
    line = json.dumps(payload, default=str) + "\n"
    current_file = _get_run_log_file()
    if current_file is not None:
        try:
            current_file.write(line)
            current_file.flush()
        except Exception:
            pass
    logger = logging.getLogger("bench.run")
    log_fn = getattr(logger, level.lower(), logger.info)
    log_fn("%s %s", event, kwargs)


def run_log_path() -> Optional[str]:
    """Return current run log file path, if set."""
    return _get_run_log_path()
