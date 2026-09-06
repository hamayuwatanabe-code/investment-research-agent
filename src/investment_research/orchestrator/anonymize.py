"""Blind-evaluation pack construction (requirements 6 and 14).

The Blind Judge must evaluate facts, not a name.  Two things have to disappear:

* the obvious identity strings (ticker, company name, aliases), and
* the *implicit* identity carried by source URLs and titles -- a
  ``sec.gov/.../corbus-pharmaceuticals-8k.htm`` link de-anonymizes instantly.

So sources are replaced by opaque references (``S-001``) that retain everything
the Judge legitimately needs -- tier, dates, document type -- and nothing that
names the company.  The mapping is kept by the orchestrator and restored after
the verdict is fixed.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Mapping, Sequence

from ..schemas.agent_io import Evaluation
from ..schemas.fact import Fact

ANON_LABEL = "Company X"

#: Fields that would identify the company and are therefore rewritten.
_IDENTIFYING_FACT_FIELDS = ("claim", "notes", "source_title", "value")


def _redact(text: str, patterns: Sequence[re.Pattern[str]]) -> str:
    for pattern in patterns:
        text = pattern.sub(ANON_LABEL, text)
    return text


def _identity_patterns(
    ticker: str, company_name: str, aliases: Sequence[str]
) -> list[re.Pattern[str]]:
    from .isolation import identity_markers

    return [
        re.compile(rf"(?<![A-Za-z0-9]){re.escape(m)}(?![A-Za-z0-9])", re.I)
        for m in identity_markers(ticker, company_name, aliases)
    ]


class SourceRefMap:
    """Bidirectional map between real sources and opaque blind references."""

    def __init__(self) -> None:
        self.to_ref: dict[str, str] = {}
        self.to_source: dict[str, tuple[str, str]] = {}

    def ref_for(self, source_id: str, url: str, title: str) -> str:
        if source_id not in self.to_ref:
            ref = f"S-{len(self.to_ref) + 1:03d}"
            self.to_ref[source_id] = ref
            self.to_source[ref] = (url, title)
        return self.to_ref[source_id]

    def restore(self, ref: str) -> tuple[str, str]:
        return self.to_source.get(ref, ("UNKNOWN", "UNKNOWN"))


def anonymize_facts(
    facts: Sequence[Fact],
    ticker: str,
    company_name: str,
    aliases: Sequence[str] = (),
    ref_map: SourceRefMap | None = None,
) -> tuple[tuple[Fact, ...], SourceRefMap]:
    ref_map = ref_map or SourceRefMap()
    patterns = _identity_patterns(ticker, company_name, aliases)
    out: list[Fact] = []
    for fact in facts:
        ref = ref_map.ref_for(fact.source_id, fact.source_url, fact.source_title)
        changes: dict[str, Any] = {
            "ticker": ANON_LABEL,
            "source_url": f"blindref://{ref}",
            "source_id": ref,
            "source_title": _redact(fact.source_title, patterns) or "(document)",
        }
        for field_name in _IDENTIFYING_FACT_FIELDS:
            if field_name in ("source_title",):
                continue
            value = getattr(fact, field_name)
            if isinstance(value, str):
                changes[field_name] = _redact(value, patterns)
        out.append(replace(fact, **changes))
    return tuple(out), ref_map


def anonymize_evaluation(
    evaluation: Evaluation, ticker: str, company_name: str, aliases: Sequence[str] = ()
) -> Evaluation:
    patterns = _identity_patterns(ticker, company_name, aliases)

    def scrub(value: Any) -> Any:
        if isinstance(value, str):
            return _redact(value, patterns)
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(scrub(v) for v in value)
        return value

    return Evaluation(
        author=evaluation.author,
        channel=evaluation.channel,
        summary=scrub(evaluation.summary),
        points=tuple(scrub(list(evaluation.points))),
        payload=scrub(evaluation.payload),
        fingerprint_tokens=evaluation.fingerprint_tokens,
    )


def anonymize_risk_flags(flags, ticker, company_name, aliases=()):
    """Redact identity from risk flags handed to a blind agent."""
    from dataclasses import replace as _replace

    patterns = _identity_patterns(ticker, company_name, aliases)
    return tuple(
        _replace(flag, title=_redact(flag.title, patterns), detail=_redact(flag.detail, patterns))
        for flag in flags
    )


def anonymize_contradictions(items, ticker, company_name, aliases=()):
    from dataclasses import replace as _replace

    patterns = _identity_patterns(ticker, company_name, aliases)
    return tuple(
        _replace(
            item,
            ticker=ANON_LABEL,
            description=_redact(item.description, patterns),
            left_summary=_redact(item.left_summary, patterns),
            right_summary=_redact(item.right_summary, patterns),
            resolution_note=_redact(item.resolution_note, patterns),
        )
        for item in items
    )


def anonymize_unresolved(items, ticker, company_name, aliases=()):
    from dataclasses import replace as _replace

    patterns = _identity_patterns(ticker, company_name, aliases)
    return tuple(
        _replace(
            item,
            question=_redact(item.question, patterns),
            why_it_matters=_redact(item.why_it_matters, patterns),
        )
        for item in items
    )


#: Params that must never reach the Blind Judge (requirement 6/10/14):
#: prior evaluations anchor, holdings bias, and rankings re-import a view.
_BLIND_FORBIDDEN_PARAMS = frozenset(
    {
        "ticker",
        "company_name",
        "aliases",
        "user_preferences",
        "prior_scores",
        "prior_rank",
        "prior_action",
        "previous_verdict",
        "holdings",
        "cost_basis",
        "watchlist_rank",
        "user_opinion",
    }
)


def anonymize_pack(
    facts: Sequence[Fact],
    channels: Mapping[str, Evaluation],
    params: Mapping[str, Any],
    ticker: str,
    company_name: str,
    aliases: Sequence[str] = (),
) -> tuple[tuple[Fact, ...], dict[str, Evaluation], dict[str, Any], SourceRefMap]:
    """Return (facts, channels, params, ref_map).

    The ref map is returned **out of band** and deliberately NOT placed in
    ``params``: it holds the real URLs and titles, so handing it to the Judge
    would de-anonymize the pack through the back door.  The orchestrator keeps
    it and uses it only after the verdict is fixed.
    """
    anon_facts, ref_map = anonymize_facts(facts, ticker, company_name, aliases)
    anon_channels = {
        name: anonymize_evaluation(ev, ticker, company_name, aliases)
        for name, ev in channels.items()
    }
    patterns = _identity_patterns(ticker, company_name, aliases)
    def scrub(value: Any) -> Any:
        """Redact recursively: a company name hides just as well inside a list."""
        if isinstance(value, str):
            return _redact(value, patterns)
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(scrub(v) for v in value)
        return value

    anon_params: dict[str, Any] = {}
    for key, value in params.items():
        if key in _BLIND_FORBIDDEN_PARAMS:
            continue
        anon_params[key] = scrub(value)
    anon_params["anonymized_label"] = ANON_LABEL
    return anon_facts, anon_channels, anon_params, ref_map
