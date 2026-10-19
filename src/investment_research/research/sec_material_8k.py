"""Phase 4.3L: read the bodies of material 8-K filings and their EX-99
exhibits, so a ticker-only live run cannot miss an issuer's own disclosure
that a trial failed or that its strategy changed.

Before this phase the SEC collector read only the submissions listing: an
8-K became the metadata fact "8-K filed <date> (items ...)" and nothing in
its body -- a primary-endpoint miss, a strategic review -- ever became
evidence.

Selection (8-K ``items`` are used ONLY to choose what to read; a fact is
only ever created from a fetched, validated body):

* forms ``8-K``/``8-K/A`` whose items include 1.01 (material definitive
  agreement), 1.02 (termination of a material agreement), 2.03 (direct
  financial obligation), 2.05 (exit/restructuring costs), 3.02
  (unregistered sales of equity), 7.01 (Reg FD -- where press releases are
  furnished) or 8.01 (other events -- where clinical results are commonly
  filed); an 8-K whose items field is empty is selected too (its items
  cannot be judged, so it is not excluded). 8-Ks whose items are ALL outside
  that set are counted in a run note (diagnostic only, never a blocking
  reason) and are not read;
* filed within the last ``LOOKBACK_DAYS`` (365: one full annual reporting
  cycle);
* newest first (same-day filings by accession, descending), at most
  ``MAX_FILINGS`` (8) read per run.

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
from ..scoring.regulator_language import ENDPOINT_TERM
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
from .sec_coverage import (
    ADMINISTRATIVE_ITEMS,
    FINANCING_RELEASE_CONDITIONS,
    MATERIAL_ITEMS,
    PERIODIC_RESULTS_ITEMS,
    PERIODIC_RESULTS_READ,
    CoverageEntry,
    CoverageStatus,
    PolicyClass,
    classify_items,
)
from .sec_live_smoke import MaxRequestsExceededError

MATERIAL_8K_COLLECTOR = "sec_material_8k"
MATERIAL_8K_FORMS = frozenset({"8-K", "8-K/A"})
#: Items whose filing is READ (see ``sec_coverage`` for the full item policy):
#: the interpretable MATERIAL items plus 2.02 earnings releases (newest
#: ``PERIODIC_RESULTS_READ`` only). 1.01 / 2.03 / 3.02 carry the terms of a
#: financing; 1.02, 2.05, 7.01, 8.01 terminations, restructurings, Reg FD
#: press releases and "other events" (clinical results); 3.01, 4.01, 4.02
#: listing notices, auditor changes and non-reliance on financial statements.
MATERIAL_8K_ITEMS = MATERIAL_ITEMS | PERIODIC_RESULTS_ITEMS
LOOKBACK_DAYS = 365
#: 12 = the 7 filings a real LGVN run selected under the narrower 4.3L scope
#: (observed, not assumed) + up to 2 newly in-scope 3.01 notices + the newest
#: 2 earnings releases + 1 spare. Beyond it a filing is UNVERIFIED, not dropped.
MAX_FILINGS = 12
MAX_EXHIBITS_PER_FILING = 2
GETS_PER_FILING = 2 + MAX_EXHIBITS_PER_FILING
#: The 8-K share of the run's GET budget: every filing may use its full
#: ``GETS_PER_FILING``; retries and redirects count inside the same budget.
MAX_OUTBOUND_GETS = MAX_FILINGS * GETS_PER_FILING
ALLOWED_HOSTS = frozenset({"www.sec.gov"})
RATE_LIMIT_RPS = 2.0  # SEC's fair-access ceiling is 10 requests/second
MAX_RETRIES = 1
TIMEOUT_SECONDS = 20.0
#: Every error line the pipeline must turn into a blocking reason starts
#: with this prefix (fresh run and resume alike -- it is persisted in the
#: collect snapshot's per-collector ``errors``).
UNVERIFIED_PREFIX = "UNVERIFIED MATERIAL 8-K:"
#: Prefix of the blocking reason raised for a company-reported financing
#: condition tied to a regulatory step (see :func:`financing_condition_blocking_reasons`).
FINANCING_CONDITION_PREFIX = "FINANCING CONDITION REQUIRES VERIFICATION:"
#: Version of the selection RULE itself (not just its parameters): bump when
#: the ordering, the tie-break or the meaning of an item changes.
MATERIAL_8K_SCOPE_RULE = "material-8k-scope/v3"
_MAX_CLAIM_CHARS = 1200


def _scope_id() -> str:
    return (
        f"{MATERIAL_8K_SCOPE_RULE}|forms={','.join(sorted(MATERIAL_8K_FORMS))}"
        f"|read_items={','.join(sorted(MATERIAL_8K_ITEMS))}"
        f"|administrative_items={','.join(sorted(ADMINISTRATIVE_ITEMS))}"
        f"|periodic_results_read={PERIODIC_RESULTS_READ}|lookback_days={LOOKBACK_DAYS}"
        f"|max_filings={MAX_FILINGS}|unsupported=blocks"
        "|order=class,filing_date_desc,accession_desc"
    )


#: Identifies the selection rule and its parameters. Saved in the collect
#: snapshot; a resume compares it with this value and refuses a record
#: written under any other (or no) scope.
MATERIAL_8K_SCOPE_ID = _scope_id()


def material_8k_scope() -> dict[str, Any]:
    """The structured scope a material-8-K collection ran under."""
    return {
        "id": MATERIAL_8K_SCOPE_ID,
        "rule": MATERIAL_8K_SCOPE_RULE,
        "forms": sorted(MATERIAL_8K_FORMS),
        "items": sorted(MATERIAL_8K_ITEMS),
        "administrative_items": sorted(ADMINISTRATIVE_ITEMS),
        "periodic_results_read": PERIODIC_RESULTS_READ,
        "lookback_days": LOOKBACK_DAYS,
        "max_filings": MAX_FILINGS,
        "max_outbound_gets": MAX_OUTBOUND_GETS,
        "order": "class,filing_date_desc,accession_desc",
        # stated, not searched: what this scope does NOT protect
        "not_covered": [
            "8-K items outside the read set are not read: ADMINISTRATIVE items are reported "
            "as not examined; any other item blocks as UNSUPPORTED",
            "exhibits other than EX-99 press releases (agreements, instruments) are not read",
            "contracts or conditions older than lookback_days",
            "filings absent from the submissions `filings.recent` block",
        ],
    }


@dataclass(frozen=True)
class Material8KCandidate:
    accession: str
    form: str
    filing_date: str
    items: tuple[str, ...]
    primary_document: str
    policy: PolicyClass = PolicyClass.MATERIAL
    rank: int = 0


def _parse_iso(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


@dataclass(frozen=True)
class Material8KSelection:
    selected: list[Material8KCandidate]
    beyond_cap: list[Material8KCandidate]
    window_start: date
    #: Dispositions decided WITHOUT reading anything: ADMINISTRATIVE and older
    #: earnings releases (NOT_READ_POLICY), UNSUPPORTED items (NOT_READ_UNSUPPORTED,
    #: blocking). One per accession -- nothing in the window goes unrecorded.
    pre_entries: list[CoverageEntry]


def _unsupported_items(items: tuple[str, ...]) -> list[str]:
    known = MATERIAL_ITEMS | PERIODIC_RESULTS_ITEMS | ADMINISTRATIVE_ITEMS
    return sorted(i for i in items if i not in known)


def _select(recent: Mapping[str, Any], *, today: date) -> Material8KSelection:
    window_start = today - timedelta(days=LOOKBACK_DAYS)
    forms = recent.get("form") or []
    columns = {
        key: recent.get(key) or []
        for key in ("accessionNumber", "filingDate", "items", "primaryDocument")
    }

    def at(key: str, index: int) -> str:
        values = columns[key]
        return str(values[index]) if index < len(values) and values[index] is not None else ""

    material: list[Material8KCandidate] = []
    periodic: list[Material8KCandidate] = []
    pre: list[CoverageEntry] = []
    for index, form in enumerate(forms):
        if form not in MATERIAL_8K_FORMS:
            continue
        filed = _parse_iso(at("filingDate", index))
        # An undatable 8-K cannot be placed outside the window: kept.
        if filed is not None and not (window_start <= filed <= today):
            continue
        items = tuple(i.strip() for i in at("items", index).split(",") if i.strip())
        policy = classify_items(items)
        candidate = Material8KCandidate(
            accession=at("accessionNumber", index) or UNKNOWN,
            form=str(form),
            filing_date=filed.isoformat() if filed else UNKNOWN,
            items=items,
            primary_document=at("primaryDocument", index),
            policy=policy,
        )

        def entry(status: CoverageStatus, reason: str, c: Material8KCandidate = candidate) -> CoverageEntry:
            return CoverageEntry(
                accession=c.accession, form=c.form, filing_date=c.filing_date, items=c.items,
                policy=c.policy, status=status, reason=reason,
            )

        if policy is PolicyClass.ADMINISTRATIVE:
            pre.append(entry(
                CoverageStatus.NOT_READ_POLICY,
                "administrative items only (officers/directors, bylaws, votes, exhibits): not read "
                "and NOT EXAMINED",
            ))
        elif policy is PolicyClass.UNSUPPORTED:
            pre.append(entry(
                CoverageStatus.NOT_READ_UNSUPPORTED,
                f"item(s) {', '.join(_unsupported_items(items))} have no extractor in this "
                "system, so the filing was not interpreted",
            ))
        elif policy is PolicyClass.PERIODIC_RESULTS:
            periodic.append(candidate)
        else:
            material.append(candidate)

    def newest_first(rows: list[Material8KCandidate]) -> list[Material8KCandidate]:
        # Same-day filings by accession (descending) so the order never depends
        # on how the listing happened to be arranged; an undatable filing last
        # (it cannot be shown to be recent, but it is still in scope).
        return sorted(
            rows,
            key=lambda c: (c.filing_date if c.filing_date != UNKNOWN else "0000", c.accession),
            reverse=True,
        )

    periodic = newest_first(periodic)
    for older in periodic[PERIODIC_RESULTS_READ:]:
        pre.append(CoverageEntry(
            accession=older.accession, form=older.form, filing_date=older.filing_date,
            items=older.items, policy=older.policy, status=CoverageStatus.NOT_READ_POLICY,
            reason=f"older than the newest {PERIODIC_RESULTS_READ} earnings releases: not read "
            "and NOT EXAMINED",
        ))
    # Read order: interpretable material items first, then the newest earnings
    # releases; within a class newest first. A filing past the cap is UNVERIFIED.
    ordered = [*newest_first(material), *periodic[:PERIODIC_RESULTS_READ]]
    ranked = [
        Material8KCandidate(
            c.accession, c.form, c.filing_date, c.items, c.primary_document, c.policy, rank,
        )
        for rank, c in enumerate(ordered, start=1)
    ]
    return Material8KSelection(
        ranked[:MAX_FILINGS], ranked[MAX_FILINGS:], window_start,
        sorted(pre, key=lambda e: (e.filing_date, e.accession), reverse=True),
    )


def select_material_8k_candidates(
    recent: Mapping[str, Any], *, today: date,
) -> tuple[list[Material8KCandidate], list[Material8KCandidate], date]:
    """``(selected, beyond_cap, window_start)`` from a submissions
    payload's ``filings.recent`` block, in read-priority order. Selection
    only -- nothing here is evidence."""
    selection = _select(recent, today=today)
    return selection.selected, selection.beyond_cap, selection.window_start


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
#: Inside a contractual financing condition ("... a statistically significant
#: improvement in the primary endpoint ...") "significant", "result" and a
#: bare "shall not" are contract vocabulary, not a reported trial outcome.
#: Only wording that NARRATES a result keeps a financing sentence ambiguous.
_OUTCOME_HINT_NARRATIVE_RE = re.compile(
    r"(?i)\b(?:fail\w*|miss\w*|fell\s+short|numerical\w*|trend\w*|nominal\w*"
    r"|(?:did|was|were|has|have|had)\s+not|no\s+(?:significant|statistically|difference|effect)"
    r"|showed|demonstrated\s+no)\b"
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
    r"|funding|gross\s+proceeds|closing)\b"
)
_FINANCING_CONDITION_RE = re.compile(
    r"(?i)\b(?:condition(?:ed|al)?|contingent|milestone|subject\s+to"
    r"|upon\s+(?:the\s+)?(?:satisfaction|achievement|occurrence|receipt))\b"
)
_FINANCING_REGULATORY_RE = re.compile(r"(?i)\b(?:FDA|primary\s+(?:efficacy\s+)?end\s?-?point)\b")
#: Risk-factor vocabulary. Contract modals ("which condition MAY be waived")
#: are not risk wording; a modal matters only BEFORE the condition, where it
#: turns the condition itself into a possibility (see below).
_RISK_TEXT_RE = re.compile(
    r"(?i)\b(?:risks?|possibility|possible|assurance|uncertain\w*|forward-looking)\b"
)
_FINANCING_HYPOTHETICAL_RE = re.compile(r"(?i)\b(?:if|may|might|could|would|should|whether|unless)\b")
#: "<endpoint> ... could not be agreed" with no regulator named in the
#: sentence: with whom it was not agreed is unknown, so it is surfaced.
_UNATTRIBUTED_NO_AGREEMENT_RE = re.compile(
    rf"(?i)\b{ENDPOINT_TERM}[^.;]{{0,160}}?\bcould\s+not\s+be\s+agreed\b"
)
_COULD_NOT_RE = re.compile(r"(?i)\bcould\s+not\b")
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


#: Does a conditional word govern the OUTCOME, or only something else in the
#: sentence? ("Following discussions of potentially appropriate endpoints, the
#: FDA indicated that ... could not be agreed to ..." -- the "potentially"
#: describes the endpoints under discussion, not the reported result.)
#: * STRONG: opens a hypothetical / expresses doubt; governs everything after it.
_STRONG_QUALIFIER_RE = re.compile(
    r"(?i)\b(?:if|whether|unless|should(?!\s+(?:no\s+longer|not)\s+be\s+relied)|assum\w*|risks?"
    r"|possibility|assurance|uncertain\w*"
    r"|forward-looking|in\s+the\s+event)\b"
)
#: * LOCAL: modal / intention wording governs only the clause it sits in ...
_LOCAL_QUALIFIER_RE = re.compile(
    r"(?i)\b(?:may|might|would|will|can|cannot|could|expect\w*|anticipat\w*|intend\w*|plan\w*"
    r"|hope\w*|aim\w*|seek\w*)\b"
)
#: ... unless it introduces a clause ("we expect THAT, ..., the FDA will ...").
_COMPLEMENT_QUALIFIER_RE = re.compile(
    r"(?i)\b(?:expect\w*|anticipat\w*|intend\w*|plan\w*|hope\w*|aim\w*|believe\w*|seek\w*)"
    r"\s+that\b"
)
#: * "potentially" / "possible" count only when they hedge the verb group itself.
_HEDGED_VERB_GROUP_RE = re.compile(
    r"(?i)\b(?:potential(?:ly)?|possibl[ey])\s+(?:\w+\s+)?"
    r"(?:could|cannot|can|would|will|may|might|not|be|was|were|is|are|unable|fail\w*)\b"
    r"|\bit\s+is\s+(?:potentially\s+)?possible\b"
)
_CLAUSE_BREAK_RE = re.compile(r"[,;:\u2014]")


def _governing_qualifier(sentence: str, start: int, end: int) -> bool:
    """True when conditional / forward-looking wording governs the clause
    ``sentence[start:end]`` (the reported outcome), as opposed to qualifying
    some other noun or an unrelated earlier clause."""
    prefix = sentence[:start]
    # matched over the whole sentence (a lookahead may need the words that
    # follow) but only a qualifier that BEGINS before the outcome counts
    for regex in (_STRONG_QUALIFIER_RE, _COMPLEMENT_QUALIFIER_RE):
        if any(m.start() < start for m in regex.finditer(sentence)):
            return True
    clause_start = max((m.end() for m in _CLAUSE_BREAK_RE.finditer(prefix)), default=0)
    if _LOCAL_QUALIFIER_RE.search(prefix[clause_start:]):
        return True
    return bool(_HEDGED_VERB_GROUP_RE.search(sentence[clause_start:end]))


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
        # A passive consultation report ("<endpoint> could not be agreed to
        # with the Agency ...") names the endpoint BEFORE the regulator, so
        # that rule is also scanned from the start -- but only where the
        # regulator is inside the matched clause.
        starts = [ref.start()]
        if rule_ids == _REGULATOR_CONSULTATION_RULES:
            starts.append(0)
        matches = (m for origin in starts for m in pattern.finditer(sentence, origin))
        for match in matches:
            if match.start() < ref.start() and not match.start() <= ref.start() < match.end():
                continue
            text = match.group(0)
            # "could not" is the reported outcome in "... could not be agreed to
            # while the trial is still ongoing" (the pattern demands that
            # anchor); it is not itself a hypothetical marker.
            inner = _COULD_NOT_RE.sub("", text)
            if text.lower().startswith("would") or _INNER_HYPOTHETICAL_RE.search(inner):
                continue
            if rule_ids == _REGULATOR_CONSULTATION_RULES:
                # Scoped: which words govern the OUTCOME, not just appear nearby.
                if _governing_qualifier(sentence, match.start(), match.end()):
                    continue
                return True
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
    if (
        not _FINANCING_RE.search(sentence)
        or not _FINANCING_REGULATORY_RE.search(sentence)
        or _RISK_TEXT_RE.search(sentence)
    ):
        return False
    condition = _FINANCING_CONDITION_RE.search(sentence)
    # A modal / "if" ahead of the condition makes the condition itself
    # hypothetical ("could be conditioned on ..."); after it, it is contract text.
    return condition is not None and not _FINANCING_HYPOTHETICAL_RE.search(
        sentence[:condition.start()]
    )


def _qualified(sentence: str, match: re.Match[str]) -> bool:
    """True when the clause before the outcome phrase is conditional or
    forward-looking (same sentence, nearest 80 characters)."""
    return bool(_CONDITIONAL_RE.search(sentence[max(0, match.start() - 80):match.start()]))


@dataclass(frozen=True)
class DisclosureStatement:
    #: PRIMARY_ENDPOINT_NOT_MET / PRIMARY_ENDPOINT_MET / STRATEGIC_CHANGE /
    #: AMBIGUOUS_FINANCING_STATEMENT (a financing condition too long to quote;
    #: never becomes a fact) /
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
    """An unqualified report of how a consultation with a regulator ended,
    or an endpoint "could not be agreed" with nobody named."""
    def reported(match: re.Match[str]) -> bool:
        return not _governing_qualifier(sentence, match.start(), match.end())

    if _REGULATOR_REF_RE.search(sentence) is None:
        return any(reported(m) for m in _UNATTRIBUTED_NO_AGREEMENT_RE.finditer(sentence))
    return any(reported(m) for m in _CONSULTATION_RE.finditer(sentence))


# --- issuer-reported events the Kill Gate already has rules for ---------------------
#: listing notice / going-concern doubt / accounting reliability / auditor
#: change. Each kind is tied to ONE Kill Gate rule and a sentence becomes a
#: fact only if that rule's own pattern matches it (the gate can therefore
#: always see what was extracted), it is not hypothetical, and it is not a
#: negation or a resolved matter. The fact's value is a neutral label: the
#: gate fires on the issuer's verbatim sentence, never on this classifier.
_LISTING_CONTEXT_RE = re.compile(
    r"(?i)\b(?:received|notif\w*|notice|letter|deficien\w*|not\s+in\s+compliance"
    r"|failed\s+to\s+(?:maintain|satisfy|meet)|fell\s+below|non-?compliance)\b"
)
_LISTING_VENUE_RE = re.compile(
    r"(?i)\b(?:nasdaq|nyse|new\s+york\s+stock\s+exchange|stock\s+exchange|listing)\b"
)
_GOING_CONCERN_NEGATION_RE = re.compile(
    r"(?i)\b(?:alleviat\w*|dispel\w*|no\s+substantial\s+doubt|(?:did|does|do)\s+not\s+raise"
    r"|not\s+(?:raise|raised|result)|without\s+substantial)\b"
)
_RELIABILITY_NEGATION_RE = re.compile(
    r"(?i)\b(?:no|not|none|never|without|nor|absence\s+of|remediat\w*)\s+(?:\w+\s+){0,2}"
    r"(?:material\s+weakness\w*|restate\w*|non-?reliance)\b"
    r"|\bmaterial\s+weakness\w*[^.]{0,60}\b(?:remediat\w*|resolved|corrected)\b"
)
_AUDITOR_CONTEXT_RE = re.compile(r"(?i)\b(?:auditor|accounting\s+firm|certifying\s+accountant)\b")

_EVENT_SPECS: tuple[tuple[str, str], ...] = (
    ("LISTING_DEFICIENCY_REPORTED", "delisting"),
    ("GOING_CONCERN_DOUBT_REPORTED", "going_concern"),
    ("ACCOUNTING_RELIABILITY_REPORTED", "restatement"),
    ("AUDITOR_CHANGE_REPORTED", "auditor_change"),
)


def _event_context_ok(kind: str, sentence: str) -> bool:
    if kind == "LISTING_DEFICIENCY_REPORTED":
        return bool(_LISTING_CONTEXT_RE.search(sentence) and _LISTING_VENUE_RE.search(sentence))
    if kind == "GOING_CONCERN_DOUBT_REPORTED":
        return "substantial doubt" in sentence.lower() and not _GOING_CONCERN_NEGATION_RE.search(
            sentence
        )
    if kind == "ACCOUNTING_RELIABILITY_REPORTED":
        return not _RELIABILITY_NEGATION_RE.search(sentence)
    return bool(_AUDITOR_CONTEXT_RE.search(sentence))


def _event_kind(sentence: str) -> str | None:
    if "forward-looking" in sentence.lower():
        return None
    for kind, rule_id in _EVENT_SPECS:
        for pattern in _rule_patterns((rule_id,)):
            for match in pattern.finditer(sentence):
                if _governing_qualifier(sentence, match.start(), match.end()):
                    continue
                if _event_context_ok(kind, sentence):
                    return kind
    return None


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
        financing = "forward-looking" not in sentence.lower() and _financing_contingent_on_regulatory(
            sentence
        )
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
            and (_OUTCOME_HINT_NARRATIVE_RE if financing else _OUTCOME_HINT_RE).search(sentence)
            and not any(_qualified(sentence, m) for m in _PRIMARY_RE.finditer(sentence))
        ):
            statements.append(DisclosureStatement("AMBIGUOUS_PRIMARY_ENDPOINT", sentence))
        if consultation_outcome:
            statements.append(
                DisclosureStatement("REGULATOR_CONSULTATION_OUTCOME_REPORTED", sentence)
            )
        elif consultation:
            statements.append(DisclosureStatement("AMBIGUOUS_REGULATOR_STATEMENT", sentence))
        if financing:
            statements.append(DisclosureStatement(
                "FINANCING_CONTINGENT_ON_REGULATORY_MILESTONE"
                if len(sentence) <= _MAX_CLAIM_CHARS else "AMBIGUOUS_FINANCING_STATEMENT",
                sentence,
            ))
        if (
            not regulator_status
            and len(sentence) <= _MAX_CLAIM_CHARS
            and any(not _qualified(sentence, m) for m in _STRATEGY_RE.finditer(sentence))
        ):
            statements.append(DisclosureStatement("STRATEGIC_CHANGE", sentence))
        event = _event_kind(sentence)
        if event is not None:
            statements.append(DisclosureStatement(
                event if len(sentence) <= _MAX_CLAIM_CHARS else "AMBIGUOUS_EVENT_STATEMENT",
                sentence,
            ))
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
    "LISTING_DEFICIENCY_REPORTED": "exchange listing notice reported by the issuer (issuer SEC disclosure)",
    "GOING_CONCERN_DOUBT_REPORTED": "continuing-operations doubt reported by the issuer (issuer SEC disclosure)",
    "ACCOUNTING_RELIABILITY_REPORTED": "financial-statement reliability matter reported by the issuer (issuer SEC disclosure)",
    "AUDITOR_CHANGE_REPORTED": "auditor change reported by the issuer (issuer SEC disclosure)",
}
_FACT_CATEGORY = {
    "LISTING_DEFICIENCY_REPORTED": FactCategory.LISTING,
    "GOING_CONCERN_DOUBT_REPORTED": FactCategory.LIQUIDITY,
    "ACCOUNTING_RELIABILITY_REPORTED": FactCategory.ACCOUNTING,
    "AUDITOR_CHANGE_REPORTED": FactCategory.ACCOUNTING,
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


#: ``_get`` failure text for an exhausted GET budget (a cause, not just a message).
_BUDGET_FAILURE = "the run's budget of {cap} outbound SEC GETs was exhausted"

FACT_UNIT_8K = "8k_disclosure_statement"
FACT_UNIT_PERIODIC = "periodic_report_statement"
_FACT_UNITS = frozenset({FACT_UNIT_8K, FACT_UNIT_PERIODIC})

#: Statement kinds that are "found a sentence, could not classify it".
AMBIGUOUS_KINDS = frozenset({
    "AMBIGUOUS_PRIMARY_ENDPOINT", "AMBIGUOUS_REGULATOR_STATEMENT",
    "AMBIGUOUS_FINANCING_STATEMENT", "AMBIGUOUS_EVENT_STATEMENT",
})
_AMBIGUOUS_TEXT = {
    "AMBIGUOUS_PRIMARY_ENDPOINT": (
        "{doc} contains a primary-endpoint statement that cannot be classified unambiguously: "
        "\"{sentence}\""
    ),
    "AMBIGUOUS_REGULATOR_STATEMENT": (
        "{doc} contains a regulator-consultation statement that cannot be classified (it is "
        "neither the regulator's own document nor a trial result, and no fact is recorded for "
        "it): \"{sentence}\""
    ),
    "AMBIGUOUS_FINANCING_STATEMENT": (
        "{doc} contains a financing condition tied to a regulatory step in a sentence too long "
        "to quote whole, so it cannot be classified and no fact is recorded for it: "
        "\"{sentence}\""
    ),
    "AMBIGUOUS_EVENT_STATEMENT": (
        "{doc} contains a listing / going-concern / accounting-reliability / auditor statement "
        "in a sentence too long to quote whole, so it cannot be classified and no fact is "
        "recorded for it: \"{sentence}\""
    ),
}


def ambiguous_statement_text(kind: str, doc: str, sentence: str) -> str:
    return _AMBIGUOUS_TEXT[kind].format(doc=doc, sentence=sentence[:300])


def filing_source(
    *, form: str, accession: str, filing_date: str, doc_type: str, name: str, url: str,
    event_date: str,
) -> Source:
    return Source(
        source_id=make_source_id(url, f"{accession} {name}"),
        url=url,
        title=f"{form} {accession} {doc_type} ({name})",
        tier=SourceTier.TIER_1,
        publisher="SEC EDGAR",
        filing_date=filing_date,
        published_date=filing_date,
        event_date=event_date,
        accession=accession,
        provenance=Provenance.LIVE,
    )


def statement_fact(
    ticker: str, statement: DisclosureStatement, source: Source,
    authority: DocumentAuthority, *, collector: str, unit: str,
) -> RawFact:
    """The issuer's verbatim sentence as a COMPANY_CLAIM -- never decision-grade."""
    return RawFact(
        ticker=ticker.upper(),
        category=_FACT_CATEGORY.get(statement.kind, FactCategory.CLINICAL),
        claim=statement.sentence,
        source=source,
        value=_FACT_VALUE[statement.kind],
        unit=unit,
        company_claim=True,
        collector=collector,
        content_kind=ContentKind.FULL_DOCUMENT,
        source_authority=authority,
    )


def _unverified(candidate: Material8KCandidate, reason: str) -> str:
    items = ",".join(candidate.items) or "items not stated"
    return (
        f"{UNVERIFIED_PREFIX} {candidate.form} accession {candidate.accession} filed "
        f"{candidate.filing_date} ({items}) -- {reason}; its content is UNVERIFIED and no "
        "action label can rest on the evidence without it"
    )


class SubBudgetHttp:
    """A view of one shared ``AllowlistedHttpClient`` that stops starting new
    logical requests once ``limit`` ACTUAL outbound GETs (as counted by the
    client: retries and redirect hops included) have been spent THROUGH THIS
    VIEW. The client's own ``max_requests`` stays the hard total for the run,
    so the sum of the views can never exceed it; a view only keeps one
    collector from starving the other. A single logical request that is
    already running may overshoot its view by its own retries/redirects."""

    def __init__(self, client: Any, limit: int) -> None:
        self._client = client
        self._limit = limit
        self._base = getattr(client, "attempts_made", 0)

    @property
    def limit(self) -> int:
        return self._limit

    def get(self, url: str, **kwargs: Any) -> Any:
        if getattr(self._client, "attempts_made", 0) - self._base >= self._limit:
            raise MaxRequestsExceededError(f"sub-budget of {self._limit} outbound GETs exhausted")
        return self._client.get(url, **kwargs)


def _get(http: Any, url: str, *, cap: int = MAX_OUTBOUND_GETS) -> tuple[Any | None, str]:
    try:
        result = http.get(url)
    except MaxRequestsExceededError:
        return None, _BUDGET_FAILURE.format(cap=getattr(http, "limit", cap))
    if not result.ok:
        return None, f"fetch failed ({result.outcome})"
    return result, ""


def get_failure_status(failure: str) -> CoverageStatus:
    """Budget exhaustion and a failed fetch are different causes in the ledger."""
    return (
        CoverageStatus.UNREAD_BUDGET if failure.startswith("the run's budget of")
        else CoverageStatus.UNREAD_FETCH_FAILED
    )


#: Non-press-release exhibit types listed in a filing's index that this system
#: does not read (agreements, instruments, plans). Recorded per accession so
#: the gap is visible; they do not block (the 8-K body must itself describe
#: the material terms) -- stated as a residual risk, see the README.
_UNSUPPORTED_EXHIBIT_RE = re.compile(r"(?i)^EX-(?:1|2|3|4|10)(?:\.|$)")


@dataclass
class FilingRead:
    facts: list[RawFact]
    reasons: list[str]
    notes: list[str]
    status: CoverageStatus
    cause: str
    documents_read: list[str]
    documents_not_read: list[str]
    #: statements found but not turned into facts (reported, not blocking, for
    #: long periodic reports -- see ``sec_periodic_reports``)
    unclassified: int = 0


def _read_filing(
    ticker: str, cik: int, candidate: Material8KCandidate, http: Any,
) -> FilingRead:
    """Read one candidate's primary document and (up to the cap) its EX-99
    press releases, classifying every statement."""
    causes: list[str] = []

    def fail(
        status: CoverageStatus, why: str, *, facts: list[RawFact] | None = None,
    ) -> FilingRead:
        return FilingRead(
            facts or [], [_unverified(candidate, why)], [], status, why, [], [],
        )

    if not _validate_accession(candidate.accession):
        return fail(CoverageStatus.UNREAD_INVALID, "the accession number is malformed")
    if not _validate_filename(candidate.primary_document):
        return fail(CoverageStatus.UNREAD_INVALID, "the primary document name is missing or unsafe")
    accession_nodash = candidate.accession.replace("-", "")

    index, failure = _get(http, _directory_index_json_url(cik, accession_nodash))
    if index is None:
        return fail(get_failure_status(failure), f"the filing index could not be read: {failure}")
    listing = ((index.json() or {}).get("directory") or {}).get("item") or []
    names = {str(item.get("name", "")) for item in listing if isinstance(item, Mapping)}
    if candidate.primary_document not in names:
        return fail(CoverageStatus.UNREAD_INVALID, "the filing index does not list the primary document")

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
    not_read = [
        f"{item.get('type')} {item.get('name')}: not read (agreement / instrument exhibits "
        "are not supported)"
        for item in listing
        if isinstance(item, Mapping) and _UNSUPPORTED_EXHIBIT_RE.match(str(item.get("type", "")))
    ]
    reasons: list[str] = []
    if len(exhibits) > MAX_EXHIBITS_PER_FILING:
        why = (
            f"{len(exhibits) - MAX_EXHIBITS_PER_FILING} further EX-99 exhibit(s) beyond the "
            f"per-filing cap of {MAX_EXHIBITS_PER_FILING} were not read"
        )
        reasons.append(_unverified(candidate, why))
        causes.append(why)
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
    first_unread: list[CoverageStatus] = []

    def note_failure(why: str, status: CoverageStatus = CoverageStatus.UNREAD_INVALID) -> None:
        reasons.append(_unverified(candidate, why))
        causes.append(why)
        first_unread.append(status)

    for document in documents:
        url = _resolve_document_url(cik, accession_nodash, document.name)
        if url is None:
            note_failure(f"{document.doc_type} {document.name!r} has no safe URL")
            continue
        fetched, failure = _get(http, url)
        if fetched is None:
            note_failure(
                f"{document.doc_type} {document.name} was not read: {failure}",
                get_failure_status(failure),
            )
            continue
        text = _html_to_text(fetched.text)
        if not _content_is_valid_body(text):
            note_failure(
                f"{document.doc_type} {document.name} failed body validation (too short, an index "
                "page, or an access-denial/rate-limit page)"
            )
            continue
        read_names.append(f"{document.doc_type} {document.name}")
        source = filing_source(
            form=candidate.form, accession=candidate.accession, filing_date=candidate.filing_date,
            doc_type=document.doc_type, name=document.name, url=fetched.url,
            event_date=body_event_date(text, is_primary_8k=document.is_primary),
        )
        for statement in extract_statements(text):
            if statement.kind in AMBIGUOUS_KINDS:
                note_failure(ambiguous_statement_text(
                    statement.kind, f"{document.doc_type} {document.name}", statement.sentence,
                ))
                continue
            kinds.add(statement.kind)
            facts.append(statement_fact(
                ticker, statement, source, document.authority,
                collector=MATERIAL_8K_COLLECTOR, unit=FACT_UNIT_8K,
            ))
    if {"PRIMARY_ENDPOINT_NOT_MET", "PRIMARY_ENDPOINT_MET"} <= kinds:
        why = (
            "its documents contain contradictory primary-endpoint statements (both met and not "
            "met); both are reported as facts and neither is treated as settled"
        )
        reasons.append(_unverified(candidate, why))
        causes.append(why)
    notes = [
        f"{candidate.form} {candidate.accession} filed {candidate.filing_date}: read "
        f"{', '.join(read_names) or 'nothing'}; {len(facts)} outcome/strategy statement(s) extracted"
    ]
    if not reasons:
        status = CoverageStatus.READ
    elif read_names:
        status = CoverageStatus.READ_PARTIAL
    else:
        status = first_unread[0] if first_unread else CoverageStatus.UNREAD_FETCH_FAILED
    return FilingRead(
        facts, reasons, notes, status, "; ".join(causes), read_names, not_read,
    )


def _ledger_note(entries: Sequence[CoverageEntry]) -> list[str]:
    """Run notes summarising the NOT-read dispositions (the entries
    themselves are the authority; these only make them easy to see)."""
    notes: list[str] = []
    by_reason: dict[tuple[str, str], int] = {}
    for entry in entries:
        if entry.status in (CoverageStatus.NOT_READ_POLICY, CoverageStatus.NOT_READ_UNSUPPORTED):
            key = (entry.status.value, ",".join(entry.items) or "items not stated")
            by_reason[key] = by_reason.get(key, 0) + 1
    for (status, items), count in sorted(by_reason.items()):
        label = (
            "NOT EXAMINED by policy (does not block)" if status == "NOT_READ_POLICY"
            else "NOT INTERPRETED, no extractor for these items (BLOCKS)"
        )
        notes.append(f"{count} 8-K filing(s) with items {items}: {label}")
    return notes


def collect_material_8k_disclosures(
    ticker: str,
    *,
    cik: int | None,
    recent: Mapping[str, Any] | None,
    http: Any | None,
    today: date,
    unavailable_reason: str = "",
    older_listing_pages: int = 0,
) -> CollectionResult:
    """One ``CollectionResult`` for this run's material 8-K reading. ``http``
    is ``None`` exactly when no request may be made (``unavailable_reason``
    says why: offline run, no valid SEC User-Agent) -- every in-scope
    candidate is then UNVERIFIED, and a submissions listing that was never
    obtained makes the whole scope UNVERIFIED (it cannot be known whether a
    material 8-K exists). ``older_listing_pages`` is the number of further
    submissions pages the SEC lists beyond ``filings.recent`` (not fetched)."""
    out = CollectionResult(
        collector=MATERIAL_8K_COLLECTOR, provenance=Provenance.LIVE, scope=material_8k_scope(),
    )
    if cik is None or recent is None:
        out.outcome = FetchOutcome.ERROR
        out.errors.append(
            f"{UNVERIFIED_PREFIX} the SEC filing list was not obtained, so whether a material "
            f"8-K was filed in the last {LOOKBACK_DAYS} days cannot be determined"
            + (f" ({unavailable_reason})" if unavailable_reason else "")
        )
        return out

    selection = _select(recent, today=today)
    selected, beyond_cap, window_start = (
        selection.selected, selection.beyond_cap, selection.window_start,
    )
    out.notes.append(
        f"material 8-K scope: forms {'/'.join(sorted(MATERIAL_8K_FORMS))} filed {window_start} to "
        f"{today}; read-set items {', '.join(sorted(MATERIAL_8K_ITEMS))} (or no items stated); "
        f"{len(selected)} selected for reading, {len(beyond_cap)} beyond the per-run cap of "
        f"{MAX_FILINGS}. NOT covered: earlier 8-Ks, agreement/instrument exhibits, and 10-K/10-Q "
        "bodies (see the periodic-report collector)"
    )
    entries: list[CoverageEntry] = list(selection.pre_entries)
    out.notes.extend(_ledger_note(selection.pre_entries))
    for entry in selection.pre_entries:
        if entry.status is CoverageStatus.NOT_READ_UNSUPPORTED:
            out.errors.append(
                f"{UNVERIFIED_PREFIX} {entry.form} accession {entry.accession} filed "
                f"{entry.filing_date} ({','.join(entry.items)}) -- {entry.reason}; its content is "
                "UNVERIFIED and no action label can rest on the evidence without it"
            )
    if older_listing_pages > 0:
        dates = [d for d in (_parse_iso(x) for x in (recent.get("filingDate") or [])) if d]
        if dates and min(dates) > window_start:
            out.errors.append(
                f"{UNVERIFIED_PREFIX} the SEC filing list reaches back only to {min(dates)} but "
                f"the {LOOKBACK_DAYS}-day window starts {window_start}; {older_listing_pages} "
                "older listing page(s) exist and are not fetched, so 8-Ks in the earlier part of "
                "the window cannot be accounted for"
            )
    for candidate in beyond_cap:
        why = f"it is beyond the per-run cap of {MAX_FILINGS} filings"
        out.errors.append(_unverified(candidate, why))
        entries.append(CoverageEntry(
            candidate.accession, candidate.form, candidate.filing_date, candidate.items,
            candidate.policy, CoverageStatus.UNREAD_BEYOND_CAP, why, rank=candidate.rank,
        ))
    for candidate in selected:
        if http is None:
            why = unavailable_reason or "no SEC client available"
            out.errors.append(_unverified(candidate, why))
            entries.append(CoverageEntry(
                candidate.accession, candidate.form, candidate.filing_date, candidate.items,
                candidate.policy, CoverageStatus.UNREAD_NO_CLIENT, why, rank=candidate.rank,
            ))
            continue
        read = _read_filing(ticker, cik, candidate, http)
        out.raw_facts.extend(read.facts)
        out.sources.extend({f.source.source_id: f.source for f in read.facts}.values())
        out.errors.extend(read.reasons)
        out.notes.extend(read.notes)
        entries.append(CoverageEntry(
            candidate.accession, candidate.form, candidate.filing_date, candidate.items,
            candidate.policy, read.status, read.cause,
            tuple(read.documents_read), tuple(read.documents_not_read), len(read.facts),
            rank=candidate.rank,
        ))
    out.coverage = [e.to_dict() for e in sorted(
        entries, key=lambda e: (e.filing_date, e.accession), reverse=True,
    )]
    if out.errors:
        out.outcome = FetchOutcome.ERROR
    return out


_FINANCING_KIND = "FINANCING_CONTINGENT_ON_REGULATORY_MILESTONE"


def is_financing_condition_fact(fact: Any) -> bool:
    """True for a fact this module emitted as a financing condition tied to a
    regulatory step -- recognised by the unit and the fixed value phrase the
    extractor writes (both survive collection, verification and a snapshot
    round-trip), never by re-reading the claim."""
    return (
        getattr(fact, "unit", None) in _FACT_UNITS
        and getattr(fact, "value", None) == _FACT_VALUE[_FINANCING_KIND]
    )


def financing_condition_blocking_reasons(facts: Sequence[Any]) -> list[str]:
    """One explicit blocking reason per distinct company-reported financing
    condition tied to a regulatory step. It is NOT a kill level: no rule maps
    "a tranche depends on the FDA" to a severity, and inventing one would be
    arithmetic without evidence. It is a verification requirement -- the
    agreement itself and the regulator's position are unread -- so the Action
    is withheld until a person has checked it. Built from the fact's own
    claim, source and dates, so none of them is lost."""
    reasons: list[str] = []
    seen: set[tuple[str, str]] = set()
    for fact in facts:
        if not is_financing_condition_fact(fact):
            continue
        key = (str(fact.source_url), str(fact.claim))
        if key in seen:
            continue
        seen.add(key)
        reasons.append(
            f"{FINANCING_CONDITION_PREFIX} the issuer reports a financing condition tied to a "
            f"regulatory step (COMPANY_CLAIM, not verified against the agreement or the "
            f"regulator): \"{str(fact.claim)[:400]}\" -- source {fact.source_title}, "
            f"{fact.source_url}, filed {fact.filing_date}, event {fact.event_date}. "
            f"{FINANCING_RELEASE_CONDITIONS}"
        )
    return reasons


def material_8k_scope_blocking_reasons(
    collector_summaries: Sequence[Mapping[str, Any]],
) -> list[str]:
    """For a RESTORED collect snapshot: its material-8-K record is only
    usable if it was written under the CURRENT selection scope. A record with
    no scope, an unreadable scope or another scope's id is refused -- never
    reused, and never repaired by guessing or by re-fetching -- because what
    it did not select (a financing 8-K, say) cannot be told from what did not
    exist."""
    reasons: list[str] = []
    for summary in collector_summaries:
        if summary.get("collector") != MATERIAL_8K_COLLECTOR:
            continue
        scope = summary.get("scope")
        recorded = scope.get("id") if isinstance(scope, Mapping) else None
        if recorded == MATERIAL_8K_SCOPE_ID:
            continue
        if not isinstance(scope, Mapping) or not scope:
            found = "no scope recorded"
        else:
            found = f"scope {recorded!r}" if isinstance(recorded, str) else "an unreadable scope"
        reasons.append(
            f"{UNVERIFIED_PREFIX} the restored collection's material 8-K record was written "
            f"under a different selection scope ({found}; current scope "
            f"{MATERIAL_8K_SCOPE_ID}), so which 8-Ks it covered cannot be established and it is "
            "not reused; no request was made to fill the gap"
        )
    return reasons


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
