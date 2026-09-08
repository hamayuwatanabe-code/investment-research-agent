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


# --- type-preserving redaction (production defect: httpx2 %-style logging) --
def test_redactor_preserves_types_in_httpx_style_percent_logging():
    """The exact pattern that crashed production: %d fed a str raises TypeError.

    httpx2 logs 'HTTP Request: %s %s "%s %d %s"' with a real int status code.
    Stringifying every arg (the old behavior) turned that int into a str, so
    the eventual `record.getMessage()` -- which does `msg % args` -- crashed
    with `TypeError: %d format: a real number is required, not str`.
    """
    redactor = SecretRedactingFilter(["sk-ant-super-secret-value-0123456789"])
    record = logging.LogRecord(
        name="httpx",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='HTTP Request: %s %s "%s %d %s"',
        args=("POST", "https://api.anthropic.com/v1/messages", "HTTP/1.1", 200, "OK"),
        exc_info=None,
    )
    redactor.filter(record)

    # Types preserved -- this is what would have raised before the fix.
    assert record.getMessage() == 'HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"'
    assert isinstance(record.args[3], int)
    assert record.args[3] == 200


def test_redactor_redacts_a_string_arg_alongside_preserved_numeric_args():
    secret = "sk-ant-super-secret-value-0123456789"
    redactor = SecretRedactingFilter([secret])
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="key=%s status=%d ratio=%.1f",
        args=(secret, 200, 0.5),
        exc_info=None,
    )
    redactor.filter(record)

    assert record.args[0] == "***REDACTED***"
    assert record.args[1] == 200 and isinstance(record.args[1], int)
    assert record.args[2] == 0.5 and isinstance(record.args[2], float)
    message = record.getMessage()
    assert secret not in message
    assert "status=200" in message
    assert "ratio=0.5" in message


def test_redactor_preserves_percent_d_and_percent_f_formatting():
    redactor = SecretRedactingFilter([])
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="count=%d avg=%.1f flag=%s value=%s",
        args=(42, 3.75, True, None),
        exc_info=None,
    )
    redactor.filter(record)
    # Must not raise -- this is exactly the TypeError observed in production.
    assert record.getMessage() == "count=42 avg=3.8 flag=True value=None"
    assert isinstance(record.args[0], int)
    assert isinstance(record.args[1], float)
    assert record.args[2] is True
    assert record.args[3] is None


def test_redactor_redacts_strings_nested_in_a_dict_arg():
    """A lone mapping arg is logging's %(key)s form -- LogRecord unwraps the
    1-tuple into a bare dict on `record.args` itself (stdlib behavior)."""
    secret = "sk-ant-super-secret-value-0123456789"
    redactor = SecretRedactingFilter([secret])
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="headers=%(Authorization)s",
        args=({"Authorization": f"Bearer {secret}", "count": 3},),
        exc_info=None,
    )
    redactor.filter(record)
    assert isinstance(record.args, dict)
    assert secret not in record.args["Authorization"]
    assert record.args["count"] == 3 and isinstance(record.args["count"], int)


def test_redactor_redacts_strings_nested_in_a_list_within_a_tuple_arg():
    secret = "sk-ant-super-secret-value-0123456789"
    redactor = SecretRedactingFilter([secret])
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="values=%s count=%d",
        args=([f"Bearer {secret}", "ok"], 3),
        exc_info=None,
    )
    redactor.filter(record)
    values, count = record.args
    assert secret not in values[0]
    assert values[1] == "ok"
    assert count == 3 and isinstance(count, int)


def test_redactor_never_reveals_the_anthropic_api_key_via_percent_logging():
    """End-to-end: the exact credential must never survive a %-style log call."""
    secret = "sk-ant-do-not-leak-0123456789abcdef"
    redactor = SecretRedactingFilter([secret])
    record = logging.LogRecord(
        name="httpx",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='HTTP Request: %s %s "%s %d %s"',
        args=("POST", f"https://api.anthropic.com/v1/messages?key={secret}", "HTTP/1.1", 200, "OK"),
        exc_info=None,
    )
    redactor.filter(record)
    rendered = record.getMessage()
    assert secret not in rendered
    assert "***REDACTED***" in rendered
