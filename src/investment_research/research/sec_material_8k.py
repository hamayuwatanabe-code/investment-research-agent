"""Phase 4.3L: read the bodies of material 8-K filings and their EX-99
exhibits, so a ticker-only live run cannot miss an issuer's own disclosure
that a trial failed or that its strategy changed.

Before this phase the SEC collector read only the submissions listing: an
8-K became the metadata fact "8-K filed <date> (items ...)" and nothing in
its body -- a primary-endpoint miss, a strategic review -- ever became
evidence.

Selection (8-K ``items`` are used ONLY to choose what to read; a fact is
only ever created from a fetched, validated body):

* forms ``8-K``/``8-K/A`` whose items include 1.02 (termination of a
  material agreement), 2.05 (exit/restructuring costs), 7.01 (Reg FD --
  where press releases are furnished) or 8.01 (other events -- where
  clinical results are commonly filed); an 8-K whose items field is empty
  is selected too (its items cannot be judged, so it is not excluded);
* filed within the last ``LOOKBACK_DAYS`` (365: one full annual reporting
  cycle);
* newest first, at most ``MAX_FILINGS`` (8) read per run.

What this deliberately does NOT cover, and says so in the run's notes:
8-Ks outside the look-back window, other item codes (notably 2.02 earnings
releases), and 10-K/10-Q bodies. A candidate that IS in scope but is not
fully read -- beyond the cap, out of GET budget, a failed or invalid body,
a missing SEC User-Agent, an unclassifiable outcome sentence, or
contradictory primary-endpoint statements -- is never dropped silently: it
is recorded as an ``UNVERIFIED MATERIAL 8-K`` error, which the pipeline
turns into a blocking reason (no action label). The cap therefore trades
blocking frequency, never safety.

Traffic: per filing at most ``GETS_PER_FILING`` (4) GETs -- the accession's
``index.json``, the primary document, and up to ``MAX_EXHIBITS_PER_FILING``
(2) EX-99 exhibits -- so the run budget is ``MAX_OUTBOUND_GETS`` (32 = 8 x 4)
ACTUAL outbound GETs. The caller supplies an ``AllowlistedHttpClient``
(www.sec.gov only) whose ``max_requests`` counts retries and followed
redirects as GETs too (Phase 4.3K), rate-limits every GET, validates
redirect targets before connecting, and strips secrets from every URL it
returns. This module never builds a client of its own.

Extraction is deterministic and conservative (no LLM): a sentence becomes a
fact only when it states, in the past tense and without a conditional or
forward-looking qualifier, that a PRIMARY endpoint was or was not met, or
that the company began a strategic review / discontinued a programme /
lost pivotal status. Secondary-endpoint, future-tense and hypothetical
wording ("may not meet", "if the trial fails to meet") is never read as a
result. Every fact is a ``COMPANY_CLAIM`` (the issuer's own words, however
primary the venue) -- never decision-grade on its own.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from ..collectors.base import CollectionResult
from ..schemas.enums import (
    UNKNOWN,
    ContentKind,
    DocumentAuthority,
    FactCategory,
    FetchOutcome,
    Provenance,
    SourceTier,
)
from ..schemas.fact import RawFact, Source, make_source_id
from .sec_acquisition_adapters import (
    ExhibitCategory,
    _content_is_valid_body,
    _directory_index_json_url,
    _html_to_text,
    _resolve_document_url,
    _validate_accession,
    _validate_filename,
    classify_exhibit_entry,
)
from .sec_live_smoke import MaxRequestsExceededError

MATERIAL_8K_COLLECTOR = "sec_material_8k"
MATERIAL_8K_FORMS = frozenset({"8-K", "8-K/A"})
MATERIAL_8K_ITEMS = frozenset({"1.02", "2.05", "7.01", "8.01"})
LOOKBACK_DAYS = 365
MAX_FILINGS = 8
MAX_EXHIBITS_PER_FILING = 2
GETS_PER_FILING = 2 + MAX_EXHIBITS_PER_FILING
MAX_OUTBOUND_GETS = MAX_FILINGS * GETS_PER_FILING
ALLOWED_HOSTS = frozenset({"www.sec.gov"})
RATE_LIMIT_RPS = 2.0  # SEC's fair-access ceiling is 10 requests/second
MAX_RETRIES = 1
TIMEOUT_SECONDS = 20.0
#: Every error line the pipeline must turn into a blocking reason starts
#: with this prefix (fresh run and resume alike -- it is persisted in the
#: collect snapshot's per-collector ``errors``).
UNVERIFIED_PREFIX = "UNVERIFIED MATERIAL 8-K:"
_MAX_CLAIM_CHARS = 1200


@dataclass(frozen=True)
class Material8KCandidate:
    accession: str
    form: str
    filing_date: str
    items: tuple[str, ...]
    primary_document: str


def _parse_iso(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def select_material_8k_candidates(
    recent: Mapping[str, Any], *, today: date,
) -> tuple[list[Material8KCandidate], list[Material8KCandidate], date]:
    """``(selected, beyond_cap, window_start)`` from a submissions
    payload's ``filings.recent`` block, newest first. Selection only --
    nothing here is evidence."""
    window_start = today - timedelta(days=LOOKBACK_DAYS)
    forms = recent.get("form") or []
    columns = {
        key: recent.get(key) or []
        for key in ("accessionNumber", "filingDate", "items", "primaryDocument")
    }

    def at(key: str, index: int) -> str:
        values = columns[key]
        return str(values[index]) if index < len(values) and values[index] is not None else ""

    in_scope: list[Material8KCandidate] = []
    for index, form in enumerate(forms):
        if form not in MATERIAL_8K_FORMS:
            continue
        filed = _parse_iso(at("filingDate", index))
        # An undatable 8-K cannot be placed outside the window: kept.
        if filed is not None and not (window_start <= filed <= today):
            continue
        items = tuple(i.strip() for i in at("items", index).split(",") if i.strip())
        if items and not (set(items) & MATERIAL_8K_ITEMS):
            continue
        in_scope.append(
            Material8KCandidate(
                accession=at("accessionNumber", index) or UNKNOWN,
                form=str(form),
                filing_date=filed.isoformat() if filed else UNKNOWN,
                items=items,
                primary_document=at("primaryDocument", index),
            )
        )
    # Newest dated first; an undatable filing last (it cannot be shown to be
    # recent, but it is still in scope -- beyond the cap it is UNVERIFIED).
    in_scope.sort(key=lambda c: c.filing_date if c.filing_date != UNKNOWN else "0000", reverse=True)
    return in_scope[:MAX_FILINGS], in_scope[MAX_FILINGS:], window_start


# --- statement extraction ------------------------------------------------------
_PRIMARY = r"primary\s+(?:efficacy\s+)?end\s?-?point"
_NOT_MET_RE = re.compile(
    r"(?i)"
    rf"\b(?:did\s+not|didn't|failed\s+to|fails\s+to)\s+(?:meet|achieve|reach|demonstrate)\b[^.;]{{0,80}}\b{_PRIMARY}"
    rf"|\bmiss(?:ed|es)\b[^.;]{{0,60}}\b{_PRIMARY}"
    rf"|\b{_PRIMARY}\b[^.;]{{0,80}}\b(?:was|were)\s+not\s+(?:met|achieved|statistically\s+significant)"
    rf"|\b{_PRIMARY}\b[^.;]{{0,80}}\bnot\s+statistically\s+significant"
)
_MET_RE = re.compile(
    r"(?i)"
    rf"(?<!not\s)\b(?:met|achieved)\b[^.;]{{0,40}}\b{_PRIMARY}"
    rf"|\b{_PRIMARY}\b[^.;]{{0,80}}\b(?:was|were)\s+(?:met|achieved)\b"
)
#: Strategic change / programme status disclosures (kept in step with
#: ``scoring.program_resolution._SUPERSEDING_DISCLOSURE_RE``, which must
#: recognise every fact this module emits -- a test asserts it does).
_STRATEGY_RE = re.compile(
    r"(?i)"
    r"\bstrategic\s+(?:review|alternatives|options|re-?prioriti[sz]ation)\b"
    r"|\bexplor\w*\s+(?:all\s+)?(?:strategic\s+)?(?:options|alternatives)\b"
    r"|\b(?:discontinu\w*|terminat\w*|halt\w*|suspend\w*|wind(?:ing)?\s+down|deprioriti[sz]\w*)\b"
    r"[^.;]{0,60}\b(?:program|programme|trial|study|development)\b"
    r"|\bno\s+longer\b[^.;]{0,60}\b(?:pivotal|registrational|lead)\b"
)
#: A qualifier in the clause BEFORE an outcome phrase turns it into a
#: hypothetical/forward-looking statement, never a reported result.
_CONDITIONAL_RE = re.compile(
    r"(?i)\b(?:if|whether|unless|should|assum\w*|risks?|possibility|possible|potential(?:ly)?"
    r"|may|might|could|would|will|can|cannot|expect\w*|anticipat\w*|intend\w*|plan\w*"
    r"|hope\w*|aim\w*|seek\w*|assurance|uncertain\w*|forward-looking)\b"
)
#: Outcome-like wording around a primary endpoint that neither pattern above
#: classifies -- surfaced as unverified rather than silently ignored.
_OUTCOME_HINT_RE = re.compile(
    r"(?i)\b(?:not|no|never|fail\w*|miss\w*|short|fell|numerical\w*|trend\w*|nominal\w*"
    r"|significan\w*|result\w*|outcome\w*|topline|top-line)\b"
)
_PRIMARY_RE = re.compile(rf"(?i)\b{_PRIMARY}\b")
#: The issuer reporting what a REGULATOR said about the primary endpoint
#: ("FDA advised that ... is not sufficient to demonstrate efficacy"). This
#: is a different kind of fact from a trial result: it is a COMPANY_CLAIM
#: about a regulatory consultation, in category REGULATORY, and it stays a
#: company claim until the regulator's own document is retrieved.
_REGULATOR_REF_RE = re.compile(
    r"(?i)\b(?:FDA|Food\s+and\s+Drug\s+Administration|the\s+Agency|EMA|PMDA|MHRA|regulators?)\b"
)
#: Wording inside (not before) a regulator-view match that makes it
#: hypothetical; the match itself is one of the Kill Gate's own regulator
#: rules, so every fact emitted here is one the gate can read.
_INNER_HYPOTHETICAL_RE = re.compile(
    r"(?i)\b(?:if|whether|unless|assum\w*|expect\w*|anticipat\w*|may|might|could|will)\b"
)
_REGULATOR_VIEW_RULES = ("endpoint_insufficient_for_efficacy", "endpoint_not_accepted")
_REGULATOR_STATUS_RULES = ("registrational_status_withdrawn",)
_REGULATOR_CONSULTATION_RULES = ("endpoint_agreement_not_reached",)
#: A financing whose later tranche / closing depends on a regulatory step
#: (an FDA agreement, the primary endpoint). A contractual condition, not a
#: result, so the forward-looking qualifier list does not apply -- only the
#: risk-factor vocabulary does.
_FINANCING_RE = re.compile(
    r"(?i)\b(?:tranche|financing|private\s+placement|purchase\s+agreement|warrants?"
    r"|funding|gross\s+proceeds)\b"
)
_FINANCING_CONDITION_RE = re.compile(
    r"(?i)\b(?:condition(?:ed|al)?|contingent|milestone|subject\s+to)\b"
)
_FINANCING_REGULATORY_RE = re.compile(r"(?i)\b(?:FDA|primary\s+(?:efficacy\s+)?end\s?-?point)\b")
_RISK_TEXT_RE = re.compile(
    r"(?i)\b(?:risks?|possibility|possible|may|might|could|assurance|uncertain\w*|forward-looking)\b"
)
#: The company reporting how a consultation with the regulator ended (it
#: agreed, no agreement was reached, an alternative endpoint was discussed).
#: That is neither the regulator's own document nor a trial result, and it
#: is not classified further: it is surfaced as unverified.
_CONSULTATION_RE = re.compile(
    r"(?i)\b(?:unable|not\s+able|failed|did\s+not|could\s+not|have\s+not|has\s+not)\b"
    r"[^.;]{0,40}\bagree(?:ment|d)?\b|\bno\s+agreement\b|\bdisagree\w*"
    r"|\bagree(?:s|d)?\s+with\s+the\s+(?:FDA|Agency|regulators?)\b"
)


def _rule_patterns(rule_ids: tuple[str, ...]) -> list[re.Pattern[str]]:
    # Imported lazily: scoring imports are not needed by the other helpers.
    from ..scoring.kill_gate import KILL_RULES

    return [r.pattern for r in KILL_RULES if r.rule_id in rule_ids]


def _reports_regulator(sentence: str, rule_ids: tuple[str, ...]) -> bool:
    """True when an unqualified sentence reports a regulator saying what one
    of the named Kill Gate regulator rules reads. The rules' own patterns
    are used, so a sentence classified here is one the gate can see."""
    ref = _REGULATOR_REF_RE.search(sentence)
    if ref is None:
        return False
    for pattern in _rule_patterns(rule_ids):
        # Scan from the regulator mention onward so an earlier "endpoint ..."
        # run cannot swallow the regulator's own wording.
        for match in pattern.finditer(sentence, ref.start()):
            text = match.group(0)
            if text.lower().startswith("would") or _INNER_HYPOTHETICAL_RE.search(text):
                continue
            start = min(ref.start(), match.start())
            if _CONDITIONAL_RE.search(sentence[max(0, start - 80):start]):
                continue
            between = sentence[ref.end():match.start()]
            if _INNER_HYPOTHETICAL_RE.search(between):
                continue
            return True
    return False
_ABBREVIATIONS = (
    "Inc.", "Corp.", "Co.", "Ltd.", "L.L.C.", "U.S.", "No.", "Nos.", "Dr.", "Mr.", "Ms.",
    "vs.", "e.g.", "i.e.", "approx.", "Ph.D.", "Jan.", "Feb.", "Mar.", "Apr.", "Jun.",
    "Jul.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec.", "St.",
)


def split_sentences(text: str) -> list[str]:
    protected = re.sub(r"\s+", " ", text)
    for abbreviation in _ABBREVIATIONS:
        protected = protected.replace(abbreviation, abbreviation.replace(".", "\x00"))
    protected = re.sub(r"(\d)\.(\d)", "\\1\x00\\2", protected)
    parts = re.split(r"(?<=[.!?])\s+(?=[\"'(\u201c]?[A-Z0-9])", protected)
    return [p.replace("\x00", ".").strip() for p in parts if p.strip()]


def _reports_regulator_view(sentence: str) -> bool:
    return _reports_regulator(sentence, _REGULATOR_VIEW_RULES)


def _reports_regulator_status(sentence: str) -> bool:
    return _reports_regulator(sentence, _REGULATOR_STATUS_RULES)


def _reports_consultation_outcome(sentence: str) -> bool:
    return _reports_regulator(sentence, _REGULATOR_CONSULTATION_RULES)


def _financing_contingent_on_regulatory(sentence: str) -> bool:
    return bool(
        _FINANCING_RE.search(sentence)
        and _FINANCING_CONDITION_RE.search(sentence)
        and _FINANCING_REGULATORY_RE.search(sentence)
        and not _RISK_TEXT_RE.search(sentence)
    )


def _qualified(sentence: str, match: re.Match[str]) -> bool:
    """True when the clause before the outcome phrase is conditional or
    forward-looking (same sentence, nearest 80 characters)."""
    return bool(_CONDITIONAL_RE.search(sentence[max(0, match.start() - 80):match.start()]))


@dataclass(frozen=True)
class DisclosureStatement:
    #: PRIMARY_ENDPOINT_NOT_MET / PRIMARY_ENDPOINT_MET / STRATEGIC_CHANGE /
    #: REGULATOR_CONSULTATION_OUTCOME_REPORTED (issuer's report that it and
    #: the regulator did not agree on an endpoint; a REGULATORY company
    #: claim) / FINANCING_CONTINGENT_ON_REGULATORY_MILESTONE (a financing
    #: condition tied to an FDA / endpoint step) /
    #: REGULATOR_STATUS_CHANGE_REPORTED (issuer's report that the regulator no
    #: longer treats the trial as pivotal/registrational) /
    #: AMBIGUOUS_REGULATOR_STATEMENT (a consultation outcome that is not
    #: classified; never becomes a fact) /
    #: REGULATOR_VIEW_REPORTED (issuer's report of a regulator's view; a
    #: REGULATORY company claim, never a trial result) /
    #: AMBIGUOUS_PRIMARY_ENDPOINT (never becomes a fact)
    kind: str
    sentence: str


def _consultation_outcome(sentence: str) -> bool:
    """An unqualified report of how a consultation with a regulator ended."""
    if _REGULATOR_REF_RE.search(sentence) is None:
        return False
    return any(not _qualified(sentence, m) for m in _CONSULTATION_RE.finditer(sentence))


def extract_statements(text: str) -> list[DisclosureStatement]:
    statements: list[DisclosureStatement] = []
    for sentence in split_sentences(text):
        if "forward-looking" in sentence.lower():
            continue
        not_met = [m for m in _NOT_MET_RE.finditer(sentence) if not _qualified(sentence, m)]
        met = [m for m in _MET_RE.finditer(sentence) if not _qualified(sentence, m)]
        regulator_view = _reports_regulator_view(sentence)
        regulator_status = _reports_regulator_status(sentence)
        consultation = _consultation_outcome(sentence)
        consultation_outcome = _reports_consultation_outcome(sentence)
        overlong = len(sentence) > _MAX_CLAIM_CHARS and bool(
            not_met or met or regulator_view or regulator_status
        )
        if overlong or (not_met and met) or (
            (regulator_view or regulator_status) and (not_met or met)
        ):
            # Both readings in one sentence, or a run-on too long to quote
            # whole: never settled either way.
            statements.append(DisclosureStatement("AMBIGUOUS_PRIMARY_ENDPOINT", sentence))
        elif regulator_view:
            statements.append(DisclosureStatement("REGULATOR_VIEW_REPORTED", sentence))
        elif regulator_status:
            statements.append(DisclosureStatement("REGULATOR_STATUS_CHANGE_REPORTED", sentence))
        elif not_met:
            statements.append(DisclosureStatement("PRIMARY_ENDPOINT_NOT_MET", sentence))
        elif met:
            statements.append(DisclosureStatement("PRIMARY_ENDPOINT_MET", sentence))
        elif (
            not consultation
            and _PRIMARY_RE.search(sentence)
            and _OUTCOME_HINT_RE.search(sentence)
            and not any(_qualified(sentence, m) for m in _PRIMARY_RE.finditer(sentence))
        ):
            statements.append(DisclosureStatement("AMBIGUOUS_PRIMARY_ENDPOINT", sentence))
        if consultation_outcome:
            statements.append(
                DisclosureStatement("REGULATOR_CONSULTATION_OUTCOME_REPORTED", sentence)
            )
        elif consultation:
            statements.append(DisclosureStatement("AMBIGUOUS_REGULATOR_STATEMENT", sentence))
        if (
            len(sentence) <= _MAX_CLAIM_CHARS
            and "forward-looking" not in sentence.lower()
            and _financing_contingent_on_regulatory(sentence)
        ):
            statements.append(
                DisclosureStatement("FINANCING_CONTINGENT_ON_REGULATORY_MILESTONE", sentence)
            )
        if (
            not regulator_status
            and len(sentence) <= _MAX_CLAIM_CHARS
            and any(not _qualified(sentence, m) for m in _STRATEGY_RE.finditer(sentence))
        ):
            statements.append(DisclosureStatement("STRATEGIC_CHANGE", sentence))
    return statements


#: The canonical value carried by each fact kind. ``claim`` stays the
#: verbatim sentence; ``value`` states the classifier's reading in fixed
#: words the Kill Gate (``trial_failed``) and programme resolution read.
_FACT_VALUE = {
    "PRIMARY_ENDPOINT_NOT_MET": "primary endpoint was not met (issuer 8-K disclosure)",
    "PRIMARY_ENDPOINT_MET": "primary endpoint was met (issuer 8-K disclosure)",
    "STRATEGIC_CHANGE": "strategic review or programme change (issuer 8-K disclosure)",
    # Deliberately neutral: the Kill Gate must fire on the verbatim sentence
    # (the issuer's own words), never on this label.
    "REGULATOR_VIEW_REPORTED": (
        "regulatory view reported by the issuer (FDA document not retrieved)"
    ),
    "REGULATOR_STATUS_CHANGE_REPORTED": (
        "regulatory status change reported by the issuer (FDA document not retrieved)"
    ),
    "REGULATOR_CONSULTATION_OUTCOME_REPORTED": (
        "regulatory consultation outcome reported by the issuer (FDA document not retrieved)"
    ),
    "FINANCING_CONTINGENT_ON_REGULATORY_MILESTONE": (
        "financing condition tied to a regulatory step (issuer 8-K disclosure)"
    ),
}
_FACT_CATEGORY = {
    "STRATEGIC_CHANGE": FactCategory.OTHER,
    "REGULATOR_VIEW_REPORTED": FactCategory.REGULATORY,
    "REGULATOR_STATUS_CHANGE_REPORTED": FactCategory.REGULATORY,
    "REGULATOR_CONSULTATION_OUTCOME_REPORTED": FactCategory.REGULATORY,
    "FINANCING_CONTINGENT_ON_REGULATORY_MILESTONE": FactCategory.LIQUIDITY,
}
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_DATE_RE = (
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?"
    r"|Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+(\d{1,2}),\s+(\d{4})"
)
_EVENT_DATE_8K_RE = re.compile(
    rf"(?i)date\s+of\s+report\s*\(\s*date\s+of\s+earliest\s+event\s+reported\s*\)\s*:?\s*{_DATE_RE}"
)
_DATELINE_RE = re.compile(rf"(?i){_DATE_RE}")


def _to_iso(match: re.Match[str] | None) -> str:
    if match is None:
        return UNKNOWN
    month = _MONTHS.get(match.group(1).lower()[:3])
    try:
        return date(int(match.group(3)), month or 0, int(match.group(2))).isoformat()
    except (TypeError, ValueError):
        return UNKNOWN


def body_event_date(text: str, *, is_primary_8k: bool) -> str:
    """The event date the BODY itself states -- the 8-K cover's "Date of
    earliest event reported", or a press release's dateline in its first
    600 characters. UNKNOWN when the body does not state one: never the
    filing date substituted under another name."""
    if is_primary_8k:
        return _to_iso(_EVENT_DATE_8K_RE.search(text[:4000]))
    return _to_iso(_DATELINE_RE.search(text[:600]))


# --- reading ---------------------------------------------------------------------
@dataclass
class _Document:
    name: str
    doc_type: str
    url: str
    authority: DocumentAuthority
    is_primary: bool


def _unverified(candidate: Material8KCandidate, reason: str) -> str:
    items = ",".join(candidate.items) or "items not stated"
    return (
        f"{UNVERIFIED_PREFIX} {candidate.form} accession {candidate.accession} filed "
        f"{candidate.filing_date} ({items}) -- {reason}; its content is UNVERIFIED and no "
        "action label can rest on the evidence without it"
    )


def _get(http: Any, url: str) -> tuple[Any | None, str]:
    try:
        result = http.get(url)
    except MaxRequestsExceededError:
        return None, f"the run's budget of {MAX_OUTBOUND_GETS} outbound SEC GETs was exhausted"
    if not result.ok:
        return None, f"fetch failed ({result.outcome})"
    return result, ""


def _read_filing(
    ticker: str, cik: int, candidate: Material8KCandidate, http: Any,
) -> tuple[list[RawFact], list[str], list[str]]:
    """``(raw_facts, unverified_reasons, notes)`` for one candidate."""
    if not _validate_accession(candidate.accession):
        return [], [_unverified(candidate, "the accession number is malformed")], []
    if not _validate_filename(candidate.primary_document):
        return [], [_unverified(candidate, "the primary document name is missing or unsafe")], []
    accession_nodash = candidate.accession.replace("-", "")

    index, failure = _get(http, _directory_index_json_url(cik, accession_nodash))
    if index is None:
        return [], [_unverified(candidate, f"the filing index could not be read: {failure}")], []
    listing = ((index.json() or {}).get("directory") or {}).get("item") or []
    names = {str(item.get("name", "")) for item in listing if isinstance(item, Mapping)}
    if candidate.primary_document not in names:
        return [], [_unverified(candidate, "the filing index does not list the primary document")], []

    exhibits = sorted(
        (
            str(item.get("name", "")), str(item.get("type", ""))
        )
        for item in listing
        if isinstance(item, Mapping)
        and str(item.get("type", "")).upper().startswith("EX-99")
        and classify_exhibit_entry(
            filename=str(item.get("name", "")), exhibit_type=str(item.get("type", "")),
            description="", primary_document=candidate.primary_document,
        ) is ExhibitCategory.PRESS_RELEASE
    )
    reasons: list[str] = []
    if len(exhibits) > MAX_EXHIBITS_PER_FILING:
        reasons.append(_unverified(
            candidate,
            f"{len(exhibits) - MAX_EXHIBITS_PER_FILING} further EX-99 exhibit(s) beyond the "
            f"per-filing cap of {MAX_EXHIBITS_PER_FILING} were not read",
        ))
    documents = [
        _Document(candidate.primary_document, candidate.form, "", DocumentAuthority.STATUTORY_FILING, True),
        *(
            _Document(name, doc_type, "", DocumentAuthority.COMPANY_IR, False)
            for name, doc_type in exhibits[:MAX_EXHIBITS_PER_FILING]
        ),
    ]

    facts: list[RawFact] = []
    kinds: set[str] = set()
    read_names: list[str] = []
    for document in documents:
        url = _resolve_document_url(cik, accession_nodash, document.name)
        if url is None:
            reasons.append(_unverified(candidate, f"{document.doc_type} {document.name!r} has no safe URL"))
            continue
        fetched, failure = _get(http, url)
        if fetched is None:
            reasons.append(_unverified(candidate, f"{document.doc_type} {document.name} was not read: {failure}"))
            continue
        text = _html_to_text(fetched.text)
        if not _content_is_valid_body(text):
            reasons.append(_unverified(
                candidate,
                f"{document.doc_type} {document.name} failed body validation (too short, an index "
                "page, or an access-denial/rate-limit page)",
            ))
            continue
        read_names.append(f"{document.doc_type} {document.name}")
        event_date = body_event_date(text, is_primary_8k=document.is_primary)
        source = Source(
            source_id=make_source_id(fetched.url, f"{candidate.accession} {document.name}"),
            url=fetched.url,
            title=f"{candidate.form} {candidate.accession} {document.doc_type} ({document.name})",
            tier=SourceTier.TIER_1,
            publisher="SEC EDGAR",
            filing_date=candidate.filing_date,
            published_date=candidate.filing_date,
            event_date=event_date,
            accession=candidate.accession,
            provenance=Provenance.LIVE,
        )
        for statement in extract_statements(text):
            if statement.kind == "AMBIGUOUS_REGULATOR_STATEMENT":
                reasons.append(_unverified(
                    candidate,
                    f"{document.doc_type} {document.name} contains a regulator-consultation "
                    "statement that cannot be classified (it is neither the regulator's own "
                    "document nor a trial result, and no fact is recorded for it): "
                    f"\"{statement.sentence[:300]}\"",
                ))
                continue
            if statement.kind == "AMBIGUOUS_PRIMARY_ENDPOINT":
                reasons.append(_unverified(
                    candidate,
                    f"{document.doc_type} {document.name} contains a primary-endpoint statement "
                    f"that cannot be classified unambiguously: \"{statement.sentence[:300]}\"",
                ))
                continue
            kinds.add(statement.kind)
            facts.append(
                RawFact(
                    ticker=ticker.upper(),
                    category=_FACT_CATEGORY.get(statement.kind, FactCategory.CLINICAL),
                    claim=statement.sentence,
                    source=source,
                    value=_FACT_VALUE[statement.kind],
                    unit="8k_disclosure_statement",
                    company_claim=True,
                    collector=MATERIAL_8K_COLLECTOR,
                    content_kind=ContentKind.FULL_DOCUMENT,
                    source_authority=document.authority,
                )
            )
    if {"PRIMARY_ENDPOINT_NOT_MET", "PRIMARY_ENDPOINT_MET"} <= kinds:
        reasons.append(_unverified(
            candidate,
            "its documents contain contradictory primary-endpoint statements (both met and not "
            "met); both are reported as facts and neither is treated as settled",
        ))
    notes = [
        f"{candidate.form} {candidate.accession} filed {candidate.filing_date}: read "
        f"{', '.join(read_names) or 'nothing'}; {len(facts)} outcome/strategy statement(s) extracted"
    ]
    return facts, reasons, notes


def collect_material_8k_disclosures(
    ticker: str,
    *,
    cik: int | None,
    recent: Mapping[str, Any] | None,
    http: Any | None,
    today: date,
    unavailable_reason: str = "",
) -> CollectionResult:
    """One ``CollectionResult`` for this run's material 8-K reading. ``http``
    is ``None`` exactly when no request may be made (``unavailable_reason``
    says why: offline run, no valid SEC User-Agent) -- every in-scope
    candidate is then UNVERIFIED, and a submissions listing that was never
    obtained makes the whole scope UNVERIFIED (it cannot be known whether a
    material 8-K exists)."""
    out = CollectionResult(collector=MATERIAL_8K_COLLECTOR, provenance=Provenance.LIVE)
    if cik is None or recent is None:
        out.outcome = FetchOutcome.ERROR
        out.errors.append(
            f"{UNVERIFIED_PREFIX} the SEC filing list was not obtained, so whether a material "
            f"8-K was filed in the last {LOOKBACK_DAYS} days cannot be determined"
            + (f" ({unavailable_reason})" if unavailable_reason else "")
        )
        return out

    selected, beyond_cap, window_start = select_material_8k_candidates(recent, today=today)
    out.notes.append(
        f"material 8-K scope: forms {'/'.join(sorted(MATERIAL_8K_FORMS))} with items "
        f"{', '.join(sorted(MATERIAL_8K_ITEMS))} (or no items stated) filed {window_start} to "
        f"{today}; {len(selected)} selected for reading, {len(beyond_cap)} beyond the per-run "
        f"cap of {MAX_FILINGS}. NOT covered: earlier 8-Ks, other item codes (e.g. 2.02), "
        "and 10-K/10-Q bodies"
    )
    for candidate in beyond_cap:
        out.errors.append(_unverified(candidate, f"it is beyond the per-run cap of {MAX_FILINGS} filings"))
    for candidate in selected:
        if http is None:
            out.errors.append(_unverified(candidate, unavailable_reason or "no SEC client available"))
            continue
        facts, reasons, notes = _read_filing(ticker, cik, candidate, http)
        out.raw_facts.extend(facts)
        out.sources.extend({f.source.source_id: f.source for f in facts}.values())
        out.errors.extend(reasons)
        out.notes.extend(notes)
    if out.errors:
        out.outcome = FetchOutcome.ERROR
    return out


def unverified_material_8k_reasons(collector_summaries: Sequence[Mapping[str, Any]]) -> list[str]:
    """The blocking reasons a run's collector summaries carry -- read from
    THIS invocation's CollectionResults on a fresh run, or from the collect
    snapshot's persisted summaries on a resume (same shape:
    ``{"collector": ..., "errors": [...]}``)."""
    return [
        str(error)
        for summary in collector_summaries
        if summary.get("collector") == MATERIAL_8K_COLLECTOR
        for error in (summary.get("errors") or [])
        if str(error).startswith(UNVERIFIED_PREFIX)
    ]


__all__ = [
    "GETS_PER_FILING",
    "LOOKBACK_DAYS",
    "MATERIAL_8K_COLLECTOR",
    "MATERIAL_8K_FORMS",
    "MATERIAL_8K_ITEMS",
    "MAX_EXHIBITS_PER_FILING",
    "MAX_FILINGS",
    "MAX_OUTBOUND_GETS",
    "UNVERIFIED_PREFIX",
    "DisclosureStatement",
    "Material8KCandidate",
    "body_event_date",
    "collect_material_8k_disclosures",
    "extract_statements",
    "select_material_8k_candidates",
    "split_sentences",
    "unverified_material_8k_reasons",
]
