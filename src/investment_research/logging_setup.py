"""Structured JSON logging with mandatory secret redaction (requirements 14/20)."""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_TOKEN_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"(?i)\b(?:api[_-]?key|token|secret|password)\b\s*[:=]\s*[^\s,;)}\]]+"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{12,}"),
)

REDACTED = "***REDACTED***"


class SecretRedactingFilter(logging.Filter):
    """Scrubs known secret values and secret-looking tokens from every record."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets = [s for s in secrets if s]

    def add_secret(self, value: str) -> None:
        if value and value not in self._secrets:
            self._secrets.append(value)

    def scrub(self, text: str) -> str:
        for secret in self._secrets:
            if secret and secret in text:
                text = text.replace(secret, REDACTED)
        for pattern in _TOKEN_PATTERNS:
            text = pattern.sub(REDACTED, text)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = self.scrub(str(record.msg))
        except Exception:  # pragma: no cover - never let logging break a run
            record.msg = "<unloggable>"
        if record.args:
            record.args = tuple(self.scrub(str(a)) for a in record.args)
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            record.extra_fields = {k: self.scrub(str(v)) for k, v in extra.items()}
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in ("run_id", "agent_id", "ticker", "stage"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


_redactor: SecretRedactingFilter | None = None


def setup_logging(
    log_dir: str | Path,
    run_id: str,
    *,
    secrets: Iterable[str] = (),
    level: int = logging.INFO,
    console: bool = True,
) -> Path:
    """Configure root logging.  Returns the JSONL log file path."""
    global _redactor
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{run_id}.jsonl"

    _redactor = SecretRedactingFilter(secrets)

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    file_handler.addFilter(_redactor)
    root.addHandler(file_handler)

    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(logging.Formatter("%(levelname)-7s %(name)s :: %(message)s"))
        stream.addFilter(_redactor)
        stream.setLevel(max(level, logging.INFO))
        root.addHandler(stream)

    return log_path


def get_redactor() -> SecretRedactingFilter | None:
    return _redactor


def log_event(logger: logging.Logger, level: int, msg: str, **fields: Any) -> None:
    logger.log(level, msg, extra={"extra_fields": fields})
