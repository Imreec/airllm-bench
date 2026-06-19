"""A rotating JSONL logger with FIFO file retention (ported from HW2).

Each event is one JSON object per line; the file rolls over every ``max_lines``
lines and at most ``max_files`` are kept (oldest dropped first). Used to record
every benchmark run as one structured line, which the analysis layer reads back.
Secret-keyed fields are redacted so a token can never land in a log.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

#: Substrings that mark a field as a secret. Deliberately specific so legitimate
#: metrics (``prompt_tokens``, ``cache_read_tokens``) are never scrubbed.
_SECRET_HINTS = ("api_key", "apikey", "secret", "password", "authorization", "access_key", "token")
_REDACTED = "***redacted***"


def _is_secret_key(key: str) -> bool:
    """Return whether ``key`` names a secret value that must not be logged."""
    lowered = key.lower()
    return any(hint in lowered for hint in _SECRET_HINTS)


def _redact(event: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively replace secret-keyed values so no credential is ever written."""
    cleaned: dict[str, Any] = {}
    for key, value in event.items():
        if _is_secret_key(key):
            cleaned[key] = _REDACTED
        elif isinstance(value, Mapping):
            cleaned[key] = _redact(value)
        else:
            cleaned[key] = value
    return cleaned


class JsonlFormatter(logging.Formatter):
    """Render a log record as a single-line JSON object, redacting secret fields."""

    def format(self, record: logging.LogRecord) -> str:
        event = record.msg if isinstance(record.msg, Mapping) else {"message": record.getMessage()}
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "event": _redact(event),
        }
        return json.dumps(payload, ensure_ascii=False)


class LineCountRotatingHandler(RotatingFileHandler):
    """A ``RotatingFileHandler`` that rolls over by line count, not bytes."""

    def __init__(self, filename: Path, *, max_files: int, max_lines: int) -> None:
        super().__init__(filename, maxBytes=0, backupCount=max(max_files - 1, 0), encoding="utf-8")
        self.max_lines = max_lines
        self._lines = 0

    def shouldRollover(self, record: logging.LogRecord) -> int:  # noqa: N802
        if self.max_lines <= 0:
            return 0
        do_roll = self._lines >= self.max_lines
        self._lines = 1 if do_roll else self._lines + 1
        return int(do_roll)


def build_logger(
    log_file: Path, *, max_files: int = 5, max_lines: int = 10_000, name: str = "airllm_bench"
) -> logging.Logger:
    """Create a JSONL logger writing to ``log_file`` with FIFO rotation."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    handler = LineCountRotatingHandler(log_file, max_files=max_files, max_lines=max_lines)
    handler.setFormatter(JsonlFormatter())
    logger.addHandler(handler)
    return logger


def log_event(logger: logging.Logger, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` (e.g. a RunResult dict) as one JSONL line."""
    logger.info(payload)
