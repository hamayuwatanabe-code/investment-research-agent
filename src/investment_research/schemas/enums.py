"""Controlled vocabularies.

Every enum here is part of the system's contract.  Free-text substitutes are
rejected by the validators in :mod:`investment_research.schemas.validation`,
because the whole point of this system is that an agent cannot smuggle an
evaluation through a string field.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """`str`-valued enum (3.10-compatible replacement for `enum.StrEnum`)."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


class EvidenceClass(StrEnum):
    """Fact vs. opinion separation (requirement 1B).

    A claim MUST be classified before it can be stored.  Nothing downstream is
    permitted to treat COMPANY_CLAIM or ANALYST_OPINION as if it were a
    VERIFIED_FACT.
    """

    VERIFIED_FACT = "VERIFIED_FACT"
    COMPANY_CLAIM = "COMPANY_CLAIM"
    INDEPENDENT_EVIDENCE = "INDEPENDENT_EVIDENCE"
    ANALYST_OPINION = "ANALYST_OPINION"
    MARKET_INFERENCE = "MARKET_INFERENCE"
    MODEL_INFERENCE = "MODEL_INFERENCE"
    UNVERIFIED_CLAIM = "UNVERIFIED_CLAIM"
    #: Phase 3E.4: a REPORTING PERSON's own statutory assertion (a Form
    #: 4/4-A transaction or relationship claim) -- distinct from
    #: COMPANY_CLAIM (the issuer's own statement) and from
    #: INDEPENDENT_EVIDENCE. The filer is neither the issuer speaking about
    #: itself nor an independent third party confirming anything; they are
    #: an interested party making a statutorily-compelled assertion about
    #: their own transaction. Deliberately excluded from
    #: DECISION_GRADE_CLASSES (see below) -- never VERIFIED_FACT and never
    #: promotable to it by anything downstream.
    REPORTING_PERSON_STATUTORY_ASSERTION = "REPORTING_PERSON_STATUTORY_ASSERTION"
    #: Phase 3F, renamed and corrected in Phase 3F.0.1: a claim asserted BY
    #: a PubMed/Europe PMC-indexed document's own authors (a reported
    #: result, an endpoint definition, a study-design statement) --
    #: distinct from INDEPENDENT_EVIDENCE, which this system's existing
    #: escalation mapping treats as carrying ``independent_confirmation=
    #: True``. That is exactly wrong here: PubMed/Europe PMC indexing is
    #: not itself proof of peer review (a preprint, an online book chapter,
    #: an editorial, or a letter can be indexed too -- see
    #: ``PublicationStage``/``PeerReviewStatus``), and even genuine peer
    #: review is editorial review of methodology, not independent
    #: confirmation that a reported result is true; the authors themselves
    #: may be sponsor employees or investigators paid by the company under
    #: study. A document existing, and even being indexed by these APIs, is
    #: never itself "efficacy confirmed" or "safety confirmed" -- see
    #: requirement 6 of the Phase 3F literature adapters. Deliberately
    #: excluded from DECISION_GRADE_CLASSES: never VERIFIED_FACT and never
    #: promotable to it by anything downstream, regardless of
    #: ``PeerReviewStatus``. Independence of the specific authors/funding/
    #: COI is a SEPARATE, later-stage assessment this class does not
    #: perform or imply.
    BIOMEDICAL_PUBLICATION_ASSERTION = "BIOMEDICAL_PUBLICATION_ASSERTION"


#: Evidence classes that may, on their own, support a material investment
#: conclusion.  See :func:`investment_research.scoring.evidence_confidence`.
#: REPORTING_PERSON_STATUTORY_ASSERTION is deliberately NOT a member (Phase
#: 3E.4 requirement 11) -- a reporting person's own statutory assertion is
#: never decision-grade on its own, however primary the venue it was filed
#: through.
#: BIOMEDICAL_PUBLICATION_ASSERTION (Phase 3F) is likewise deliberately
#: NOT a member: see its own docstring above.
DECISION_GRADE_CLASSES = frozenset(
    {EvidenceClass.VERIFIED_FACT, EvidenceClass.INDEPENDENT_EVIDENCE}
)

#: Statuses that bar a fact from settling a material question, however good its
#: source tier looks. A Tier 1 URL found through a search engine is still only a
#: pointer until somebody reads the document behind it.
NON_DECISIVE_STATUSES = frozenset(
    {
        "UNVERIFIED_MATERIAL_CLAIM",
        "NOT_VERIFIED",
        "NOT_FOUND",
        "INSUFFICIENT_EVIDENCE",
        "CONTRADICTED",
        "SEARCH_EVIDENCE",
        "PRIMARY_SOURCE_IDENTIFIED_BUT_NOT_FETCHED",
    }
)


class QueryPurpose(StrEnum):
    """Why a search was issued (requirement M3).

    Purpose is not a label for reporting: it is an isolation boundary. A result
    retrieved to disconfirm a thesis must not reach the agent building the case
    for it, and vice versa. Pooling them would rebuild the rebuttal loop that
    requirement 1C exists to prevent, one search result at a time.
    """

    BULL = "BULL"
    BEAR = "BEAR"
    NEUTRAL = "NEUTRAL"
    CONTRADICTION = "CONTRADICTION"
    #: Primary-source escalation. Verification is stance-free: it asks whether a
    #: document says a thing, not whether that thing is good or bad.
    VERIFICATION = "VERIFICATION"

    @property
    def is_stance_bound(self) -> bool:
        """Whether results from this purpose are restricted to one side."""
        return self in (QueryPurpose.BULL, QueryPurpose.BEAR)


#: Which purposes each agent may read. NEUTRAL, CONTRADICTION and VERIFICATION
#: results are shared; BULL and BEAR results are not.
AGENT_QUERY_PURPOSES: dict[str, frozenset[QueryPurpose]] = {
    "bull_agent": frozenset(
        {QueryPurpose.BULL, QueryPurpose.NEUTRAL, QueryPurpose.VERIFICATION}
    ),
    "bear_agent": frozenset(
        {QueryPurpose.BEAR, QueryPurpose.NEUTRAL, QueryPurpose.VERIFICATION}
    ),
    "kill_agent": frozenset(
        {QueryPurpose.BEAR, QueryPurpose.NEUTRAL, QueryPurpose.CONTRADICTION,
         QueryPurpose.VERIFICATION}
    ),
    "contradiction": frozenset(
        {QueryPurpose.NEUTRAL, QueryPurpose.CONTRADICTION, QueryPurpose.VERIFICATION}
    ),
}
#: Everyone not listed above sees the stance-free purposes only.
DEFAULT_QUERY_PURPOSES: frozenset[QueryPurpose] = frozenset(
    {QueryPurpose.NEUTRAL, QueryPurpose.CONTRADICTION, QueryPurpose.VERIFICATION}
)


def purposes_for_agent(agent_id: str) -> frozenset[QueryPurpose]:
    return AGENT_QUERY_PURPOSES.get(agent_id, DEFAULT_QUERY_PURPOSES)


class SourceTier(StrEnum):
    """Source hierarchy (requirement 2).  Lower tier number == more primary."""

    TIER_1 = "TIER_1"  # regulators, exchanges, statutory filings
    TIER_2 = "TIER_2"  # peer-reviewed, company IR / transcripts, gov research
    TIER_3 = "TIER_3"  # major financial press, trade press, expert interviews
    TIER_4 = "TIER_4"  # sell-side analysts, broker research, price targets
    TIER_5 = "TIER_5"  # social media, retail blogs, message boards
    UNKNOWN = "UNKNOWN"

    @property
    def rank(self) -> int:
        return {"TIER_1": 1, "TIER_2": 2, "TIER_3": 3, "TIER_4": 4, "TIER_5": 5}.get(self.value, 9)

    @property
    def is_primary(self) -> bool:
        return self.rank <= 2


#: Tiers that may NOT, alone, settle a material investment question (req. 2).
NON_DECISIVE_TIERS = frozenset({SourceTier.TIER_4, SourceTier.TIER_5, SourceTier.UNKNOWN})


class DocumentAuthority(StrEnum):
    """Who actually authored/issued a document -- distinct from SourceTier.

    SourceTier is a confidence/priority ranking; ContentKind is how much of
    the document is in hand; this is about WHO IS SPEAKING in it. A statutory
    filing hosted at sec.gov, or a registry entry hosted at
    clinicaltrials.gov, is still issuer/sponsor-authored: the regulator or
    exchange lends the filing its hosting, not its voice. Hosting location
    alone must never be read as who is speaking -- that conflation is exactly
    how a fetched issuer filing can end up mislabeled as independent
    regulator confirmation.
    """

    #: A regulator/agency itself authored this document (an FDA letter,
    #: meeting minutes, a warning letter, an EMA opinion, ...).
    REGULATOR = "REGULATOR"
    #: The issuer's own required statutory disclosure (10-K/10-Q/8-K and
    #: equivalents), however primary the domain that hosts it.
    STATUTORY_FILING = "STATUTORY_FILING"
    #: A government or clinical registry entry -- typically sponsor-authored
    #: (e.g. a ClinicalTrials.gov record), read directly rather than via a
    #: search summary.
    REGISTRY = "REGISTRY"
    #: Company investor relations material or a press release.
    COMPANY_IR = "COMPANY_IR"
    #: An independent publication: wire/trade press, peer-reviewed journal,
    #: or similar -- authored by neither the issuer nor a regulator.
    INDEPENDENT = "INDEPENDENT"
    #: Phase 3E.4: a REPORTING PERSON's own statutory filing (e.g. a Form
    #: 4/4-A) -- filed THROUGH the issuer's own SEC EDGAR filing
    #: infrastructure, but authored by the individual insider, not the
    #: issuer. Kept structurally distinct from STATUTORY_FILING (the
    #: issuer's OWN required disclosure): conflating the two would let a
    #: reporting person's unconfirmed assertion inherit an issuer
    #: statement's evidentiary weight merely because SEC hosts both.
    REPORTING_PERSON_FILING = "REPORTING_PERSON_FILING"
    #: Phase 3F, corrected in Phase 3F.0.1: a document indexed by PubMed or
    #: Europe PMC -- authored by the study's investigators, not by the
    #: issuer, a regulator, or a registry. Deliberately NOT named
    #: "peer-reviewed": PubMed indexes far more than MEDLINE-reviewed
    #: journal articles (online books, editorials, letters, and -- via NIH's
    #: Preprint Pilot -- preprints), and Europe PMC's own search explicitly
    #: covers preprint servers. Being retrievable through either API is
    #: never itself proof that peer review happened; see ``PublicationStage``/
    #: ``PeerReviewStatus`` below for the fields that actually carry that
    #: distinction, always defaulting to UNKNOWN rather than assumed true.
    #: Kept structurally distinct from INDEPENDENT (whose escalation mapping
    #: is company_claim=False, independent_confirmation=True): a document's
    #: authors may be company employees, paid investigators, or otherwise
    #: non-independent of the subject under study, and even genuine peer
    #: review checks methodology, not truth. Never conflated with REGISTRY
    #: (a ClinicalTrials.gov record is the sponsor's own structured entry,
    #: not a published document) or with COMPANY_IR (indexing is not the
    #: issuer's own investor-relations channel merely because a company
    #: employee co-authored the piece).
    BIOMEDICAL_LITERATURE = "BIOMEDICAL_LITERATURE"
    UNKNOWN = "UNKNOWN"


class PublicationStage(StrEnum):
    """What KIND of document this is, as declared by the source's own
    structured metadata (a PubMed ``PublicationType``, an Europe PMC
    ``source`` code) -- never inferred from prose, and never itself a claim
    about peer review (see ``PeerReviewStatus``). A journal that has since
    retracted or corrected an article is tracked by
    ``ParsedPubmedArticle.is_retracted``/``is_correction_or_erratum``
    (``collectors/literature.py``) when the ORIGINAL article merely
    references its own correction/retraction; the CORRECTION/ERRATUM/
    RETRACTION members here are for the rarer case where the fetched
    document IS itself the correction/erratum/retraction notice.
    """

    JOURNAL_ARTICLE = "JOURNAL_ARTICLE"
    PREPRINT = "PREPRINT"
    BOOK_OR_CHAPTER = "BOOK_OR_CHAPTER"
    EDITORIAL = "EDITORIAL"
    LETTER = "LETTER"
    CORRECTION = "CORRECTION"
    ERRATUM = "ERRATUM"
    RETRACTION = "RETRACTION"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class PeerReviewStatus(StrEnum):
    """Whether a document actually underwent peer review -- structurally
    separate from ``PublicationStage`` and from mere PubMed/Europe PMC
    indexing (``DocumentAuthority.BIOMEDICAL_LITERATURE``'s own docstring).

    Phase 3F.0.1's central correction: nothing in ``collectors/literature.py``
    may ever set this to CONFIRMED merely because a document is indexed by
    PubMed, or merely because its ``PublicationType`` says "Journal
    Article" -- neither is evidence that review actually happened.
    NOT_PEER_REVIEWED is set only from an explicit negative signal (a
    PubMed "Preprint" publication type, or a Europe PMC preprint-server
    ``source`` code). Absent either signal, this stays UNKNOWN -- never
    filled with a guess (CLAUDE.md rule 7).
    """

    CONFIRMED = "CONFIRMED"
    NOT_PEER_REVIEWED = "NOT_PEER_REVIEWED"
    UNKNOWN = "UNKNOWN"


class VerifiedStatus(StrEnum):
    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    NOT_VERIFIED = "NOT_VERIFIED"
    CONTRADICTED = "CONTRADICTED"
    NOT_FOUND = "NOT_FOUND"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    #: A material claim found only in Tier 3-5 reporting whose primary-source
    #: confirmation was attempted and failed. Distinct from NOT_VERIFIED: the
    #: escalation ran, and the primary source does not (yet) corroborate it.
    #: Barred from decision-grade use (requirement P4).
    UNVERIFIED_MATERIAL_CLAIM = "UNVERIFIED_MATERIAL_CLAIM"
    #: Derived from a search result -- a title, snippet or engine summary -- and
    #: NOT from the body of the document it describes. Discovery evidence: it
    #: tells you where to look, and it is never a verified finding on its own.
    SEARCH_EVIDENCE = "SEARCH_EVIDENCE"
    #: The primary document that would settle this claim has been located (its
    #: URL is known) but its body could not be retrieved. Stronger than
    #: SEARCH_EVIDENCE, because the next step is identified; still not verified,
    #: because nobody has read the document.
    PRIMARY_SOURCE_IDENTIFIED_BUT_NOT_FETCHED = "PRIMARY_SOURCE_IDENTIFIED_BUT_NOT_FETCHED"


#: The single sentinel used everywhere a value is unknown.  Requirement 1G:
#: missing data is never imputed.
UNKNOWN = "UNKNOWN"


class Materiality(StrEnum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFORMATIONAL = "INFORMATIONAL"

    @property
    def rank(self) -> int:
        return {
            "CRITICAL": 5,
            "HIGH": 4,
            "MEDIUM": 3,
            "LOW": 2,
            "INFORMATIONAL": 1,
        }[self.value]


class FactCategory(StrEnum):
    REGULATORY = "REGULATORY"
    CLINICAL = "CLINICAL"
    SCIENCE = "SCIENCE"
    TECHNOLOGY = "TECHNOLOGY"
    FINANCIAL = "FINANCIAL"
    CAPITAL_STRUCTURE = "CAPITAL_STRUCTURE"
    LIQUIDITY = "LIQUIDITY"
    GOVERNANCE = "GOVERNANCE"
    INSIDER = "INSIDER"
    COMMERCIAL = "COMMERCIAL"
    COMPETITION = "COMPETITION"
    MARKET_SIZE = "MARKET_SIZE"
    CONTRACTS = "CONTRACTS"
    LEGAL = "LEGAL"
    ACCOUNTING = "ACCOUNTING"
    LISTING = "LISTING"
    CATALYST = "CATALYST"
    MICROSTRUCTURE = "MICROSTRUCTURE"
    MANAGEMENT = "MANAGEMENT"
    OTHER = "OTHER"


class KillCategory(StrEnum):
    REGULATORY_KILL = "REGULATORY_KILL"
    CLINICAL_KILL = "CLINICAL_KILL"
    SCIENCE_KILL = "SCIENCE_KILL"
    CAPITAL_KILL = "CAPITAL_KILL"
    COMMERCIAL_KILL = "COMMERCIAL_KILL"
    GOVERNANCE_KILL = "GOVERNANCE_KILL"
    ACCOUNTING_KILL = "ACCOUNTING_KILL"
    LIQUIDITY_KILL = "LIQUIDITY_KILL"


class KillSearchFailureReason(StrEnum):
    """Why one mandatory kill query did not produce results (requirement B).

    A live run reported "no search provider configured" for every
    unexecuted mandatory kill query even when a real, credentialed provider
    was wired up -- collapsing budget starvation, transient provider
    unavailability, and a genuine absence of any search capability into one
    misleading message. This distinguishes them, so the report never asserts
    "no provider" merely because a query happened not to execute.
    """

    #: No research provider AND no legacy SearchProvider were configured for
    #: this run at all -- the only case that may say "no search provider".
    NO_PROVIDER = "NO_PROVIDER"
    #: A provider object exists but reported itself unusable right now (no
    #: credentials, disabled, ...) -- distinct from it never existing.
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    #: The provider was usable; the stage/global LLM token budget refused
    #: this call before it ever reached the network.
    SKIPPED_DUE_TO_BUDGET = "SKIPPED_DUE_TO_BUDGET"
    #: The provider was usable, budget was not the issue, and the call
    #: itself failed (network error, malformed response, ...).
    SEARCH_ERROR = "SEARCH_ERROR"
    #: The query executed successfully and returned zero documents.
    EXECUTED_ZERO_RESULTS = "EXECUTED_ZERO_RESULTS"
    #: The query executed successfully and returned at least one document.
    EXECUTED_WITH_RESULTS = "EXECUTED_WITH_RESULTS"


#: Categories that must always be reported, even when the assessment is K0
#: (requirement 5: "最低限 ... を出す").
MANDATORY_KILL_CATEGORIES = (
    KillCategory.REGULATORY_KILL,
    KillCategory.CLINICAL_KILL,
    KillCategory.CAPITAL_KILL,
    KillCategory.COMMERCIAL_KILL,
    KillCategory.GOVERNANCE_KILL,
    KillCategory.LIQUIDITY_KILL,
)


class KillLevel(StrEnum):
    """K0-K5 severity ladder (requirement 5)."""

    K0 = "K0"  # no material concern found
    K1 = "K1"  # minor
    K2 = "K2"  # meaningful but manageable
    K3 = "K3"  # major red flag
    K4 = "K4"  # severe / normally avoid
    K5 = "K5"  # investment disqualifier

    @property
    def level(self) -> int:
        return int(self.value[1])

    @classmethod
    def from_level(cls, n: int) -> KillLevel:
        return cls(f"K{max(0, min(5, int(n)))}")


class Action(StrEnum):
    """The only permitted final action labels (requirement 23)."""

    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    BUY_ON_PULLBACK = "BUY_ON_PULLBACK"
    WAIT_FOR_EVENT = "WAIT_FOR_EVENT"
    HOLD = "HOLD"
    PARTIAL_TAKE_PROFIT = "PARTIAL_TAKE_PROFIT"
    CONSIDER_SELL = "CONSIDER_SELL"
    AVOID = "AVOID"


class Horizon(StrEnum):
    """Catalyst horizons (requirement 11)."""

    T0 = "T0"  # intraday
    T1 = "T1"  # 1-10 trading days
    T2 = "T2"  # 1-3 months
    T3 = "T3"  # 6-18 months
    T4 = "T4"  # 3-10 years

    @property
    def label(self) -> str:
        return {
            "T0": "intraday",
            "T1": "1-10 trading days",
            "T2": "1-3 months",
            "T3": "6-18 months",
            "T4": "3-10 years",
        }[self.value]


class ScenarioName(StrEnum):
    DISASTER = "DISASTER"
    BEAR = "BEAR"
    BASE = "BASE"
    BULL = "BULL"
    EXTREME_BULL = "EXTREME_BULL"


class DateKind(StrEnum):
    """Date-integrity discrimination (requirement 3)."""

    PUBLISHED_DATE = "published_date"
    EVENT_DATE = "event_date"
    EFFECTIVE_DATE = "effective_date"
    FILING_DATE = "filing_date"


class RunStatus(StrEnum):
    COMPLETE = "COMPLETE"
    INCOMPLETE_RESEARCH = "INCOMPLETE_RESEARCH"
    FAILED = "FAILED"


class FetchOutcome(StrEnum):
    OK = "OK"
    NOT_FOUND = "NOT_FOUND"
    BLOCKED = "BLOCKED"  # egress policy / 403 / 407
    RATE_LIMITED = "RATE_LIMITED"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"
    DISABLED = "DISABLED"  # collector switched off (no credentials, offline mode)


class Provenance(StrEnum):
    """Hard separation between production, captured and mock data (req. 24)."""

    LIVE = "LIVE"
    CACHE = "CACHE"
    FIXTURE = "FIXTURE"  # mock; may never be presented as real research
    #: Real documents about a real issuer, captured at a stated time through a
    #: stated channel, and replayed. Not synthetic -- but also not fetched live
    #: at run time, so the report says when it was captured.
    CAPTURED = "CAPTURED"


class ContentKind(StrEnum):
    """How much of the source document is actually in hand.

    This distinction is load-bearing. A search engine's summary *about* a press
    release is not the press release: it is a third party's paraphrase, and a
    quotation drawn from it has not been checked against the original. Treating
    the two as equivalent is how "the filing says X" ends up meaning "a search
    result said the filing says X".
    """

    FULL_DOCUMENT = "FULL_DOCUMENT"
    EXCERPT = "EXCERPT"
    SEARCH_SUMMARY = "SEARCH_SUMMARY"
    METADATA_ONLY = "METADATA_ONLY"

    @property
    def is_primary_text(self) -> bool:
        """Whether the actual source text was read."""
        return self in (ContentKind.FULL_DOCUMENT, ContentKind.EXCERPT)

    @property
    def confidence_multiplier(self) -> float:
        return {
            "FULL_DOCUMENT": 1.0,
            "EXCERPT": 0.9,
            "SEARCH_SUMMARY": 0.6,
            "METADATA_ONLY": 0.3,
        }[self.value]


class ResearchPath(StrEnum):
    """Which channel actually served a piece of evidence (ADR 0005)."""

    #: Anthropic server-side web_search / web_fetch tools.
    ANTHROPIC_WEB = "ANTHROPIC_WEB"
    #: Direct structured API (SEC EDGAR, ClinicalTrials.gov, openFDA).
    DIRECT_API = "DIRECT_API"
    #: Third-party search API (Tavily / Brave / MCP provider).
    SEARCH_API = "SEARCH_API"
    #: Replayed from a captured corpus.
    CORPUS = "CORPUS"
    #: Synthetic fixture.
    FIXTURE = "FIXTURE"
    NONE = "NONE"


class ResearchDomain(StrEnum):
    """The domains the Search Completeness Gate requires (requirement P6)."""

    REGULATORY = "REGULATORY"
    CAPITAL_STRUCTURE = "CAPITAL_STRUCTURE"
    SCIENCE_TECHNOLOGY = "SCIENCE_TECHNOLOGY"
    COMPETITION = "COMPETITION"
    CATALYST = "CATALYST"
    CONTRADICTION = "CONTRADICTION"


#: Every one of these must be SEARCHED before a final verdict may be issued.
REQUIRED_RESEARCH_DOMAINS: tuple[ResearchDomain, ...] = tuple(ResearchDomain)


class SearchStatus(StrEnum):
    #: A web/discovery-path query executed for this domain (requirement F:
    #: this is what "WEB_SEARCHED" means in this system -- the query record
    #: itself is always search-path, never inferred from facts existing).
    SEARCHED = "SEARCHED"
    #: A structured collector (SEC EDGAR / ClinicalTrials.gov / openFDA, ...)
    #: with a genuine, auditable execution record directly covers this
    #: domain -- never inferred merely because facts happen to exist for it.
    DIRECTLY_RESEARCHED = "DIRECTLY_RESEARCHED"
    PARTIAL = "PARTIAL"
    UNSEARCHED = "UNSEARCHED"
    FAILED = "FAILED"


class IntentStatus(StrEnum):
    """Per-research-intent outcome inside a batched discovery call.

    A single Anthropic server-tool call may carry evidence for several
    ResearchIntents at once (cost control), but that must never mean every
    included intent is silently marked "done" -- each one's own outcome is
    tracked exactly as if it had been issued as its own, separate call. An
    intent the model's response never addressed reads INCOMPLETE_RESPONSE,
    never merged into a neighboring intent's evidence and never quietly
    treated as searched.
    """

    PENDING = "PENDING"
    EXECUTED_WITH_EVIDENCE = "EXECUTED_WITH_EVIDENCE"
    EXECUTED_ZERO_RESULTS = "EXECUTED_ZERO_RESULTS"
    INCOMPLETE_RESPONSE = "INCOMPLETE_RESPONSE"
    SKIPPED_DUE_TO_BUDGET = "SKIPPED_DUE_TO_BUDGET"
    ERROR = "ERROR"


class ResearchStatus(StrEnum):
    """Whether the research behind a run is complete enough to bear an Action.

    This is a distinct axis from :class:`RunStatus` (did the pipeline finish
    without agent/collector failures?) and from the Search Completeness Gate
    (was every domain searched?). A run can be COMPLETE by both of those and
    still have found nothing decision-grade -- searching a domain and
    confirming a claim from it are different achievements. This status is the
    single gate that decides whether ``Verdict.action`` may be non-``None``.
    """

    #: The Evidence Sufficiency Matrix is satisfied: every required domain has
    #: decision-grade backing where material, and no unresolved MATERIAL or
    #: CRITICAL claim remains. An Action may be emitted.
    COMPLETE = "COMPLETE"
    #: The pipeline ran to the end but one or more agents/collectors failed or
    #: degraded. Distinct from BLOCKED_PENDING_VERIFICATION: this is about the
    #: run's own health, not about the evidence it produced.
    INCOMPLETE = "INCOMPLETE"
    #: Research ran and domains were searched, but the evidence that surfaced
    #: is not decision-grade (search summaries, unfetched primary sources,
    #: unverified material claims) or a material finding is still provisional.
    #: No Action may be emitted -- not even WAIT_FOR_EVENT, which is itself an
    #: Action and must never be used as a stand-in for insufficient evidence.
    BLOCKED_PENDING_VERIFICATION = "BLOCKED_PENDING_VERIFICATION"


class KillConfirmation(StrEnum):
    """Whether a Kill finding rests on decision-grade evidence.

    Distinct from :class:`KillLevel`: severity (how bad, if true) and
    confirmation (whether it has actually been verified) are two different
    questions, and collapsing them let an unfetched search snippet drive an
    AVOID exactly as if a filing had been read.
    """

    #: Raised from evidence that is not (yet) decision-grade: a search
    #: snippet, an identified-but-unfetched primary source, an unverified
    #: material claim, or an LLM-proposed flag not traced to a decision-grade
    #: fact. Reportable and must trigger further verification; must never by
    #: itself justify a final Action.
    PROVISIONAL = "PROVISIONAL"
    #: Backed by at least one decision-grade fact at this finding's severity.
    CONFIRMED = "CONFIRMED"


class EvidenceSufficiencyStatus(StrEnum):
    """Per-domain outcome of the Evidence Sufficiency Matrix.

    Deliberately separate from :class:`SearchStatus`: a domain can be fully
    SEARCHED and still be evidence-INSUFFICIENT, because searching a domain
    and confirming what it found are different achievements.
    """

    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"
    UNSEARCHED = "UNSEARCHED"
