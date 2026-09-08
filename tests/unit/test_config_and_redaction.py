"""Settings credential source and secret-redaction tests (requirement 20).

Covers the switch of the Anthropic credential from the global
``ANTHROPIC_API_KEY`` to the app-scoped ``IRA_ANTHROPIC_API_KEY``: the global
variable must never be required, and the credential must never survive log
redaction regardless of which environment variable it came from.
"""

from __future__ import annotations

import logging

from investment_research.config import Settings
from investment_research.logging_setup import SecretRedactingFilter


def test_settings_reads_anthropic_key_from_ira_prefixed_var(monkeypatch):
    monkeypatch.setenv("IRA_ANTHROPIC_API_KEY", "sk-ant-scoped-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = Settings()
    assert settings.anthropic_api_key == "sk-ant-scoped-key"


def test_settings_does_not_read_global_anthropic_api_key(monkeypatch):
    """The global var must never be required or substituted for the app key."""
    monkeypatch.delenv("IRA_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-global-should-be-ignored")
    settings = Settings()
    assert settings.anthropic_api_key == ""


def test_settings_anthropic_key_defaults_empty_without_any_env(monkeypatch):
    monkeypatch.delenv("IRA_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = Settings()
    assert settings.anthropic_api_key == ""


def test_secret_values_includes_ira_anthropic_key(monkeypatch):
    monkeypatch.setenv("IRA_ANTHROPIC_API_KEY", "sk-ant-scoped-secret-value")
    settings = Settings()
    assert "sk-ant-scoped-secret-value" in settings.secret_values()


def test_redactor_scrubs_ira_anthropic_key_value():
    """A logged message containing the credential must come back redacted."""
    secret = "sk-ant-scoped-secret-value-0123456789"
    redactor = SecretRedactingFilter([secret])
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=f"calling Anthropic with key {secret}",
        args=(),
        exc_info=None,
    )
    redactor.filter(record)
    assert secret not in record.msg
    assert "***REDACTED***" in record.msg


def test_redactor_scrubs_ira_anthropic_key_by_shape_even_if_unknown():
    """Even without being registered as a known secret, an sk-... value is caught."""
    redactor = SecretRedactingFilter([])
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="IRA_ANTHROPIC_API_KEY=sk-ant-not-registered-abcdefghijklmnop",
        args=(),
        exc_info=None,
    )
    redactor.filter(record)
    assert "sk-ant-not-registered" not in record.msg
    assert "***REDACTED***" in record.msg
