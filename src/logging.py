from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import Any

_LOG_PATH: Path | None = None
_ACTIVE_LOG_FILENAME = ".active_log"


def _build_file_handler(path: Path) -> logging.FileHandler:
    """Create the single run-scoped file handler."""

    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname).1s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    return handler


def _active_log_path(logs_directory: Path) -> Path:
    """Return the pointer file used to reuse a log across processes."""

    return logs_directory / _ACTIVE_LOG_FILENAME


def _resolve_log_path(
    logs_directory: Path,
    reset: bool,
) -> Path:
    """Pick the log file for this process, reusing the active run when needed."""

    active_log_path = _active_log_path(logs_directory)
    if not reset and active_log_path.exists():
        candidate = Path(active_log_path.read_text(encoding="utf-8").strip())
        if candidate.exists():
            return candidate

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    log_path = logs_directory / f"{timestamp}.log"
    active_log_path.write_text(str(log_path), encoding="utf-8")
    return log_path


def _log_unhandled_exception(
    exception_type: type[BaseException],
    exception_value: BaseException,
    exception_traceback: Any,
) -> None:
    """Write uncaught exceptions to the log file."""

    if issubclass(exception_type, KeyboardInterrupt):
        sys.__excepthook__(exception_type, exception_value, exception_traceback)
        return

    get_logger(__name__).error(
        "unhandled exception",
        exc_info=(exception_type, exception_value, exception_traceback),
    )


def initialize_logging(
    logs_directory: Path,
    reset: bool = False,
) -> Path:
    """Configure centralized run-scoped logging once per process."""

    global _LOG_PATH

    if _LOG_PATH is not None:
        return _LOG_PATH

    logs_directory.mkdir(parents=True, exist_ok=True)
    log_path = _resolve_log_path(logs_directory, reset=reset)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(_build_file_handler(log_path))

    logging.captureWarnings(True)
    sys.excepthook = _log_unhandled_exception

    _LOG_PATH = log_path
    get_logger(__name__).info("log file: %s", log_path)
    return log_path


def clear_active_log(logs_directory: Path) -> None:
    """Clear the active log pointer after a full pipeline run completes."""

    active_log_path = _active_log_path(logs_directory)
    if active_log_path.exists():
        active_log_path.unlink()


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a module logger."""

    return logging.getLogger(name if name is not None else "pipeline")
