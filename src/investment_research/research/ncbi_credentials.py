"""NCBI E-utilities / Europe PMC courtesy-identification credential
resolution -- shared between Literature Live Smoke (Phase 3F.1) and the
Phase 4.2A production Literature Pipeline Integration, so the two never
carry two independently-maintained copies of the same refusal rule.

Deliberately NOT ``config.Settings``, whose ``ncbi_tool`` carries a silent
non-empty default (``"investment-research-agent"``) that is exactly wrong
for a caller that must refuse outright on an unconfigured environment
rather than quietly substitute a placeholder. ``IRA_NCBI_TOOL``/
``IRA_NCBI_EMAIL`` are read straight from the process environment here
(or an injected ``env`` mapping, for offline testing) and are REQUIRED;
``IRA_NCBI_API_KEY`` is always optional. Values are never printed, logged,
or placed in a plan/report/exception/Document/Capture Manifest by any
caller of this module -- only whether each is CONFIGURED (a plain boolean,
via ``CredentialStatus``) is ever meant to be displayed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

NCBI_TOOL_ENV_VAR = "IRA_NCBI_TOOL"
NCBI_EMAIL_ENV_VAR = "IRA_NCBI_EMAIL"
NCBI_API_KEY_ENV_VAR = "IRA_NCBI_API_KEY"


@dataclass(frozen=True)
class NcbiCredentials:
    """Duck-typed to match ``literature_acquisition_adapter.py``'s own
    ``_eutils_params``/``_secret_values`` expectations (``ncbi_tool``/
    ``ncbi_email``/``ncbi_api_key`` attributes). Never logged, printed, or
    placed in a plan/report/Document/Capture Manifest by any caller; the
    adapter it is handed to uses these values for exactly one thing --
    building a wire URL -- per that module's own wire_url/public_url
    discipline."""

    ncbi_tool: str
    ncbi_email: str
    ncbi_api_key: str


@dataclass(frozen=True)
class CredentialStatus:
    """Configured/not-configured booleans ONLY -- never the values
    themselves."""

    tool_configured: bool
    email_configured: bool
    #: Optional; never contributes to a refusal on its own.
    api_key_configured: bool


def resolve_ncbi_credentials(
    env: dict[str, str] | None = None,
) -> tuple[NcbiCredentials, CredentialStatus, list[str]]:
    """Reads the three NCBI courtesy-identification env vars directly from
    the environment. Returns ``(credentials, status,
    missing_required_env_var_names)`` -- ``IRA_NCBI_API_KEY`` is always
    optional and never appears in the missing list. ``env`` is injectable
    for offline testing only; a production caller passes ``None`` to read
    the real process environment.
    """
    source = env if env is not None else os.environ
    tool = (source.get(NCBI_TOOL_ENV_VAR) or "").strip()
    email = (source.get(NCBI_EMAIL_ENV_VAR) or "").strip()
    api_key = (source.get(NCBI_API_KEY_ENV_VAR) or "").strip()
    missing = [
        name for name, value in ((NCBI_TOOL_ENV_VAR, tool), (NCBI_EMAIL_ENV_VAR, email)) if not value
    ]
    credentials = NcbiCredentials(ncbi_tool=tool, ncbi_email=email, ncbi_api_key=api_key)
    status = CredentialStatus(
        tool_configured=bool(tool), email_configured=bool(email), api_key_configured=bool(api_key)
    )
    return credentials, status, missing


__all__ = [
    "NCBI_API_KEY_ENV_VAR",
    "NCBI_EMAIL_ENV_VAR",
    "NCBI_TOOL_ENV_VAR",
    "CredentialStatus",
    "NcbiCredentials",
    "resolve_ncbi_credentials",
]
