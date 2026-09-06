"""Runtime configuration.

Requirement 20: API keys are read from the environment only.  Nothing here ever
hard-codes a secret, and :func:`Settings.secret_values` feeds the log redactor
so a key cannot reach a log file even by accident.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


def load_dotenv(path: str | Path | None = None) -> None:
    """Minimal .env loader (no dependency).  Existing env vars win."""
    dotenv = Path(path) if path else REPO_ROOT / ".env"
    if not dotenv.is_file():
        return
    for line in dotenv.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass
class Settings:
    sec_user_agent: str = field(
        default_factory=lambda: _env(
            "IRA_SEC_USER_AGENT", "investment-research-agent contact@example.com"
        )
    )
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    search_provider: str = field(default_factory=lambda: _env("IRA_SEARCH_PROVIDER", "none"))
    tavily_api_key: str = field(default_factory=lambda: _env("TAVILY_API_KEY"))
    brave_api_key: str = field(default_factory=lambda: _env("BRAVE_API_KEY"))
    market_provider: str = field(default_factory=lambda: _env("IRA_MARKET_PROVIDER", "none"))
    polygon_api_key: str = field(default_factory=lambda: _env("POLYGON_API_KEY"))

    db_path: Path = field(
        default_factory=lambda: REPO_ROOT / _env("IRA_DB_PATH", "data/research.db")
    )
    cache_dir: Path = field(
        default_factory=lambda: REPO_ROOT / _env("IRA_CACHE_DIR", "data/cache")
    )
    log_dir: Path = field(default_factory=lambda: REPO_ROOT / _env("IRA_LOG_DIR", "logs"))
    fixture_dir: Path = field(default_factory=lambda: REPO_ROOT / "data/fixtures")

    http_timeout: float = field(default_factory=lambda: _env_float("IRA_HTTP_TIMEOUT", 30.0))
    http_max_retries: int = field(default_factory=lambda: _env_int("IRA_HTTP_MAX_RETRIES", 4))
    rate_limit_rps: float = field(default_factory=lambda: _env_float("IRA_RATE_LIMIT_RPS", 5.0))

    #: Staleness threshold: facts whose event date is older than this are
    #: flagged ``stale`` rather than silently presented as current.
    stale_after_days: int = 400

    def secret_values(self) -> list[str]:
        """Every secret this process knows about, for log redaction."""
        return [
            v
            for v in (
                self.anthropic_api_key,
                self.tavily_api_key,
                self.brave_api_key,
                self.polygon_api_key,
            )
            if v and len(v) >= 8
        ]

    def ensure_dirs(self) -> None:
        for d in (self.db_path.parent, self.cache_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)

    def describe(self) -> dict[str, object]:
        """Non-secret description, safe to log and to print in reports."""
        return {
            "db_path": str(self.db_path),
            "cache_dir": str(self.cache_dir),
            "search_provider": self.search_provider,
            "market_provider": self.market_provider,
            "llm_configured": bool(self.anthropic_api_key),
            "http_timeout": self.http_timeout,
            "http_max_retries": self.http_max_retries,
            "rate_limit_rps": self.rate_limit_rps,
        }


_SETTINGS: Settings | None = None


def get_settings(reload: bool = False) -> Settings:
    global _SETTINGS
    if _SETTINGS is None or reload:
        load_dotenv()
        _SETTINGS = Settings()
    return _SETTINGS
