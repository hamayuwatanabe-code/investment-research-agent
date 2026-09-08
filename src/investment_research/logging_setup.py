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
    """Scrubs known secret values and secret-looking tokens from every record.

    Type-preserving: a %-style logging call such as
    ``logger.info('%s %d', 'GET', 200)`` still works after this filter runs.
    Only ``str`` values (and strings nested in a list/tuple/dict) are ever
    scrubbed; every other type -- int, float, bool, None, or anything else --
    passes through completely untouched, so ``record.getMessage() %
    record.args`` never sees an int-shaped ``%d`` slot fed a string. A secret
    can only ever be a string in the first place, so this loses no coverage.
    """

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

    def _scrub_value(self, value: Any) -> Any:
        """Redact ``value`` if it is (or contains) a string; else return it as-is."""
        if isinstance(value, str):
            return self.scrub(value)
        if isinstance(value, tuple):
            return tuple(self._scrub_value(v) for v in value)
        if isinstance(value, list):
            return [self._scrub_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._scrub_value(v) for k, v in value.items()}
        # int, float, bool, None, or any other non-string type: never
        # stringified here. A %-format arg must keep its original type or
        # %d/%f-style specifiers crash on the next getMessage() call.
        return value

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            # The format string itself is always coerced to a scrubbed str,
            # exactly as before -- only the %-format *arguments* below need
            # type preservation, since those are what %d/%f substitute into.
            record.msg = self.scrub(str(record.msg))
        except Exception:  # pragma: no cover - never let logging break a run
            record.msg = "<unloggable>"
        if record.args:
            try:
                if isinstance(record.args, dict):
                    record.args = {k: self._scrub_value(v) for k, v in record.args.items()}
                else:
                    record.args = tuple(self._scrub_value(a) for a in record.args)
            except Exception:  # pragma: no cover - never let logging break a run
                record.args = ()
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            record.extra_fields = {k: self._scrub_value(v) for k, v in extra.items()}
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
