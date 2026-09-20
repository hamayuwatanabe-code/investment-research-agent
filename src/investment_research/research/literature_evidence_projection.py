"""Literature Evidence Projection Bridge (Phase 4.1A).

Offline-only, pure, and never called from ``Pipeline.run()``/``cli.py`` --
see the repository's Phase 4.0 audit and CLAUDE.md's standing prohibition
on connecting the new Acquisition layer to production without a separate,
explicit approval (Phase 4.2+ territory, not this module's).

What this module does: projects the output of one ``AcquisitionExecutor``
run against ``PubMedLiteratureAdapter`` -- a run's
``TargetExecutionReport``s (each carrying its own ``StepExecutionResult``s
and their payloads) plus the SAME run's ``DocumentStore`` -- into the
EXISTING, canonical evidence path, unchanged:

    source-specific parser (collectors/literature.py)
    -> RawFact
    -> CollectionResult (this module's own output)
    -> FactCollectorAgent
    -> EvidenceIntegrityAgent
    -> verified Fact

This module is the FIRST of those arrows' caller, nothing more. It reuses
``collectors/literature.py``'s already-tested, independently-reviewed
``raw_facts_from_pubmed_article()``/``raw_facts_from_europepmc_fulltext()``
functions verbatim -- the same claim-generation logic is never duplicated
here.

Hard boundaries (Phase 4.1A -- do not weaken these to "simplify" a future
change; raise the question instead, per CLAUDE.md):

* Never constructs a ``Fact`` directly -- only ``Source``, ``RawFact``, and
  ``collectors.base.CollectionResult``. Fact construction (and the initial,
  unverified ``EvidenceClass`` it starts life at) stays
  ``FactCollectorAgent._to_fact``'s job.
* Never calls ``EvidenceIntegrityAgent._classify`` (a private method) or
  any other agent internal, and never defines a second, competing
  ``EvidenceClass``-assignment rule. Every ``RawFact`` this module's
  helpers produce carries ``source_authority=DocumentAuthority.
  BIOMEDICAL_LITERATURE`` exactly as ``collectors/literature.py``'s
  ``_make_raw_fact`` already, unconditionally, sets it -- classification
  into ``EvidenceClass.BIOMEDICAL_PUBLICATION_ASSERTION`` (excluded from
  ``DECISION_GRADE_CLASSES``, never independently confirmed) is decided
  downstream, by the real ``EvidenceIntegrityAgent``, exactly as it already
  does for every other literature RawFact in this repository.
* Never issues, retries, or waits on an HTTP/API request. Every value this
  module reads was already produced by a prior, real (or fixture-driven,
  in tests) ``AcquisitionExecutor.run()`` call; this module makes zero
  network calls of its own, by construction -- it does not even import
  anything HTTP-shaped.
* Never infers ``SourceTier`` from ``DocumentAuthority`` -- ``Source``
  projection uses one fixed, explicit, documented constant
  (``LITERATURE_SOURCE_TIER``) for every literature source, matching every
  existing literature test fixture's own choice and ``SourceTier``'s own
  docstring ("TIER_2 = peer-reviewed, ..."). It is never computed from
  ``document.authority`` or any other field.
* Never copies a document's full abstract/full-text body into
  ``Source.excerpt`` -- ``Source.excerpt`` is left at its default (empty);
  the relevant excerpted text already lives in each ``RawFact.claim``,
  produced by the existing extractor functions this module calls
  unchanged.
* Never treats a 404/timeout/BLOCKED/RATE_LIMITED/budget-exclusion/
  malformed-response/missing-Document/corrupted-version-chain outcome as a
  clean, empty result. ``CollectionResult.zero_results`` is set True only
  when a target's own LOCATE step genuinely reached
  ``StepStatus.ZERO_RESULTS`` (a completed search that found nothing) and
  nothing else went wrong anywhere in the projected set -- every other
  failure mode instead becomes ``CollectionResult.degraded=True`` plus an
  explicit, human-readable entry in ``LiteratureEvidenceProjection.
  unresolved_reasons``, never silently dropped.
* Never touches ``collectors/documents.py::chunk_document``/
  ``chunk_corpus``, and ``LiteratureEvidenceProjection`` never carries a
  ``chunks`` field. A Document's raw XML (the PubMed article's stored
  ``Document.text``) is not the same shape as its section-joined,
  already-flattened plain text (the Europe PMC full-text ``Document.text``
  -- see ``EuropePmcFullTextAdapter.fetch_fulltext``'s own docstring), and
  this phase does not decide how -- or whether -- either should be
  chunked. Chunk projection is Phase 4.1B's separate, later concern.

Europe PMC metadata gap this module's own Phase 4.1A change closed
(documented here, not just in the commit message, since it explains why
``research/literature_acquisition_adapter.py`` gained two small additive
fields): ``collectors.literature.raw_facts_from_europepmc_fulltext()``
needs a full ``EuropePmcSearchResult`` (six scalar fields:
``pmcid``/``doi``/``title``/``journal_title``/``pub_year``/``source`` --
the last of which is the ONLY signal ``EuropePmcSearchResult.
peer_review_status``/``publication_stage`` read) and the genuine,
structured ``ParsedEuropePmcFullText`` the adapter already computed
internally. Before Phase 4.1A, ``PubMedLiteratureAdapter``'s own PARSE
payload exposed only four of those six scalars (never ``pmcid``/``doi``/
``title``/``journal_title``/``pub_year``/``source``) and never the parsed
full-text structure at all. Rather than re-deriving either from
``DocumentStore`` (impossible for the parsed structure: the stored
Europe PMC full-text ``Document.text`` is already ``parsed.full_text``,
the section-joined PLAIN TEXT, not the raw XML -- ``parse_europepmc_
fulltext_xml()`` cannot be re-run over it), this module's companion change
in ``literature_acquisition_adapter.py`` makes ``EuropePmcFullTextAdapter.
fetch_fulltext()`` additionally return the ``ParsedEuropePmcFullText`` it
already computes (a 5th, additive tuple element -- ``None`` on every
failure path), and threads it plus the missing scalar fields into the
SAME per-pmid dict this module already reads from
``StepExecutionResult.payload["europepmc"]``. Zero new HTTP requests, zero
re-fetching of any body from a different URL, and
``literature_live_smoke.py`` is untouched -- it never read that dict's
pre-existing keys either (confirmed by inspection before this change was
made), so it cannot regress from new ones being added beside them.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..collectors.base import CollectionResult
from ..collectors.documents import Document
from ..collectors.literature import (
    EuropePmcSearchResult,
    ParsedPubmedArticle,
    raw_facts_from_europepmc_fulltext,
    raw_facts_from_pubmed_article,
)
from ..schemas.enums import FetchOutcome, SourceTier
from ..schemas.fact import RawFact, Source, make_source_id
from .acquisition_executor import StepExecutionResult, TargetExecutionReport
from .document_store import CorruptVersionChainError, DocumentStore, StoredDocument
from .source_routing import StepStatus

log = logging.getLogger(__name__)

#: Deliberate, fixed, and documented -- never derived from
#: ``DocumentAuthority.BIOMEDICAL_LITERATURE`` or any other field (see
#: module docstring). Matches ``SourceTier``'s own docstring ("TIER_2 =
#: peer-reviewed, company IR / transcripts, gov research") and every
#: existing literature test fixture's own choice
#: (``tests/unit/test_literature_evidence_boundary.py``,
#: ``tests/unit/test_literature_collector.py``).
LITERATURE_SOURCE_TIER = SourceTier.TIER_2

#: The whole-``CollectionResult``-level collector label this bridge
#: stamps -- distinct from the PER-RAWFACT ``collector`` values
#: (``"literature_pubmed"``/``"literature_europepmc"``), which are left at
#: ``collectors/literature.py``'s own existing defaults, unchanged, so
#: ``notes="collected_by=literature_pubmed"`` (set by
#: ``FactCollectorAgent._to_fact``) keeps meaning exactly what it already
#: means everywhere else in this repository.
BRIDGE_COLLECTOR_LABEL = "literature_evidence_projection"

#: EuropePmcSearchResult scalar fields ``literature_acquisition_adapter.py``
#: (Phase 4.1A) now exposes on ``StepExecutionResult.payload["europepmc"]
#: [pmid]``, beyond the pre-existing ``is_open_access``/``in_epmc``/
#: ``license``. All six are required to rebuild a faithful
#: ``EuropePmcSearchResult`` -- see ``_rebuild_europepmc_search_result``.
_EUROPEPMC_REQUIRED_SCALAR_FIELDS = ("pmcid", "doi", "title", "journal_title", "pub_year", "source")


@dataclass(frozen=True)
class LiteratureEvidenceProjection:
    """Everything one bridge call produced, plus the diagnostics needed to
    audit it -- never more than ``collectors.base.CollectionResult`` itself
    plus the handful of fields ``CollectionResult`` has no place for.

    ``document_ids`` is every ``DocumentStore`` identity this call actually
    resolved a ``Source``/``RawFact`` pair from -- the "投影対象
    document_id" set, in encounter order, never containing a duplicate
    (mirrors the dedup ``duplicate_source_count`` reports on).
    """

    collection_result: CollectionResult
    document_ids: tuple[str, ...]
    coverage_complete: bool
    unresolved_reasons: tuple[str, ...]
    #: How many times a PMID this call had already projected a Source/
    #: RawFact pair for was encountered again (from a second/third
    #: semantic target referencing the same PMID) and was correctly
    #: skipped rather than re-projected. Zero on a run with no overlap.
    duplicate_source_count: int = 0


def project_literature_target_reports(
    ticker: str,
    target_reports: Sequence[TargetExecutionReport],
    document_store: DocumentStore,
) -> LiteratureEvidenceProjection:
    """Project one or more literature ``TargetExecutionReport``s (e.g.
    ``execution_report.target_reports``, pre-filtered by the caller to the
    literature ones -- this module never inspects a ``SourceRoutingGraph``
    to decide that itself, since ``TargetExecutionReport`` alone carries no
    ``target_kind``) into one aggregate ``CollectionResult``.

    Never raises on a single bad target/PMID/document: every failure mode
    this module can detect (missing DocumentStore entry, a corrupted
    version chain, an incomplete Europe PMC payload, a FAILED/
    SKIPPED_DUE_TO_BUDGET step) is recorded as an unresolved reason and
    processing continues with whatever else in the batch is still sound --
    mirroring ``agents/base.py::Agent.execute``'s own partial-failure
    philosophy, never CLAUDE.md rule 7's forbidden alternative (silently
    filling the gap).
    """
    raw_facts: list[RawFact] = []
    sources_by_id: dict[str, Source] = {}
    document_ids: list[str] = []
    unresolved_reasons: list[str] = []
    notes: list[str] = []

    seen_pmids: set[str] = set()
    duplicate_source_count = 0
    acquired_targets = 0
    problem_targets = 0
    clean_zero_targets = 0
    raw_fact_count_before_dedup = 0

    for target_report in target_reports:
        parse_result = _find_literature_parse_result(target_report)
        if parse_result is None:
            reason, is_clean_zero = _describe_unresolved_target(target_report)
            if is_clean_zero:
                clean_zero_targets += 1
                notes.append(reason)
            else:
                problem_targets += 1
                unresolved_reasons.append(reason)
            continue

        payload = parse_result.payload or {}
        parsed_documents = payload.get("parsed_documents") or []
        europepmc_by_pmid = payload.get("europepmc") or {}
        target_had_problem = False

        if not payload.get("coverage_complete", True):
            target_had_problem = True
        for pmid in payload.get("budget_excluded_pmids") or ():
            target_had_problem = True
            unresolved_reasons.append(
                f"target {target_report.target_id}: PMID {pmid} excluded from EFetch by the "
                "article budget"
            )
        for pmid in payload.get("budget_skipped_europepmc_search_pmids") or ():
            target_had_problem = True
            unresolved_reasons.append(
                f"target {target_report.target_id}: Europe PMC search for PMID {pmid} skipped "
                "due to budget"
            )
        for pmcid in payload.get("budget_skipped_fulltext_pmcids") or ():
            target_had_problem = True
            unresolved_reasons.append(
                f"target {target_report.target_id}: Europe PMC full-text fetch for PMCID {pmcid} "
                "skipped due to budget"
            )
        for pmid, info in (payload.get("europepmc_search_failures") or {}).items():
            target_had_problem = True
            unresolved_reasons.append(
                f"target {target_report.target_id}: Europe PMC search failed for PMID {pmid}: {info}"
            )
        for pmid, info in (payload.get("europepmc_fulltext_failures") or {}).items():
            target_had_problem = True
            unresolved_reasons.append(
                f"target {target_report.target_id}: Europe PMC full-text fetch failed for PMID "
                f"{pmid}: {info}"
            )

        target_acquired_anything = False
        for entry in parsed_documents:
            pmid = str(entry.get("pmid", ""))
            article: ParsedPubmedArticle = entry["article"]
            document_id = entry.get("document_id")

            if pmid in seen_pmids:
                duplicate_source_count += 1
                continue

            if not document_id:
                target_had_problem = True
                unresolved_reasons.append(
                    f"PMID {pmid}: parsed article carries no document_id -- excluded from projection"
                )
                continue

            stored = _resolve_document(document_store, document_id, pmid, unresolved_reasons)
            if stored is None:
                target_had_problem = True
                continue

            seen_pmids.add(pmid)
            target_acquired_anything = True
            source = _project_source(stored.document)
            sources_by_id.setdefault(source.source_id, source)
            document_ids.append(document_id)

            pubmed_facts = raw_facts_from_pubmed_article(
                ticker, article, source, document_id=document_id,
            )
            raw_fact_count_before_dedup += len(pubmed_facts)
            raw_facts.extend(pubmed_facts)

            epmc_entry = europepmc_by_pmid.get(pmid) or {}
            if not epmc_entry.get("search_succeeded"):
                continue

            search_result = _rebuild_europepmc_search_result(pmid, epmc_entry)
            if search_result is None:
                target_had_problem = True
                unresolved_reasons.append(
                    f"PMID {pmid}: Europe PMC search succeeded but the parse payload lacked one "
                    "or more of "
                    f"{_EUROPEPMC_REQUIRED_SCALAR_FIELDS} -- excluded from Europe PMC RawFact "
                    "generation (PubMed facts for this PMID are unaffected)"
                )
                continue

            epmc_source, epmc_document_id = source, document_id
            fulltext_document_id = epmc_entry.get("fulltext_document_id")
            if fulltext_document_id:
                fulltext_stored = _resolve_document(
                    document_store, fulltext_document_id, pmid, unresolved_reasons
                )
                if fulltext_stored is not None:
                    epmc_source = _project_source(fulltext_stored.document)
                    sources_by_id.setdefault(epmc_source.source_id, epmc_source)
                    epmc_document_id = fulltext_document_id
                    if fulltext_document_id not in document_ids:
                        document_ids.append(fulltext_document_id)
                else:
                    target_had_problem = True
                    # Fall back to the PubMed abstract's own source/document_id
                    # below -- the Europe PMC metadata facts are still worth
                    # projecting even though the fulltext Document itself
                    # could not be resolved.

            parsed_fulltext = epmc_entry.get("parsed_fulltext")
            epmc_facts = raw_facts_from_europepmc_fulltext(
                ticker, pmid, search_result, parsed_fulltext, epmc_source,
                document_id=epmc_document_id,
            )
            raw_fact_count_before_dedup += len(epmc_facts)
            raw_facts.extend(epmc_facts)

        if target_acquired_anything:
            acquired_targets += 1
        if target_had_problem:
            problem_targets += 1

    # -- dedup: RawFact.fact_id() is the deterministic identity ------------
    deduped_facts: list[RawFact] = []
    seen_fact_ids: set[str] = set()
    for raw in raw_facts:
        fact_id = raw.fact_id()
        if fact_id in seen_fact_ids:
            continue
        seen_fact_ids.add(fact_id)
        deduped_facts.append(raw)

    outcome = FetchOutcome.OK if problem_targets == 0 else FetchOutcome.ERROR
    zero_results = outcome is FetchOutcome.OK and acquired_targets == 0 and clean_zero_targets > 0
    coverage_complete = problem_targets == 0 and not unresolved_reasons

    collection_result = CollectionResult(
        collector=BRIDGE_COLLECTOR_LABEL,
        raw_facts=deduped_facts,
        sources=list(sources_by_id.values()),
        outcome=outcome,
        errors=list(unresolved_reasons),
        attempted_urls=[s.url for s in sources_by_id.values()],
        notes=notes,
        zero_results=zero_results,
        raw_fact_count_before_dedup=raw_fact_count_before_dedup,
    )
    return LiteratureEvidenceProjection(
        collection_result=collection_result,
        document_ids=tuple(dict.fromkeys(document_ids)),
        coverage_complete=coverage_complete,
        unresolved_reasons=tuple(unresolved_reasons),
        duplicate_source_count=duplicate_source_count,
    )


# -- helpers -----------------------------------------------------------------
def _find_literature_parse_result(target_report: TargetExecutionReport) -> StepExecutionResult | None:
    """The PARSE step's own result, identified by SHAPE (a
    ``"parsed_documents"`` payload whose every entry's ``"article"`` is a
    real ``ParsedPubmedArticle``), never by a guessed ``step_id`` string --
    a graph the caller built is free to name its steps however it likes
    (see ``tests/unit/test_literature_acquisition_adapter.py``'s own
    ``_literature_graph`` helper). ``"parsed_documents"`` is not a unique
    payload key across every adapter in this repository (``form4_
    acquisition_adapter.py`` uses the same key for its own, differently-
    shaped PARSE payload) -- the ``ParsedPubmedArticle`` type check below
    is what actually disambiguates, defense in depth against silently
    misinterpreting a different adapter's output as literature's."""
    for step_result in target_report.step_results:
        payload = step_result.payload or {}
        parsed_documents = payload.get("parsed_documents")
        if not parsed_documents:
            continue
        if all(isinstance(entry.get("article"), ParsedPubmedArticle) for entry in parsed_documents):
            return step_result
    return None


def _describe_unresolved_target(target_report: TargetExecutionReport) -> tuple[str, bool]:
    """``(reason, is_clean_zero_result)`` for a target this bridge could
    not find a literature PARSE result for. A genuine, completed
    zero-result search (``StepStatus.ZERO_RESULTS``) is reported ONLY when
    nothing else in this target's own step results indicates a real
    problem -- a FAILED or SKIPPED_DUE_TO_BUDGET step anywhere in the same
    target always wins, even if a later step also reports ZERO_RESULTS,
    because CLAUDE.md rule 8 (\"a category that was never searched is
    UNSEARCHED, not K0\") applies here just as much as it does to the Kill
    Gate: the honest reading of a mixed outcome is the less comfortable
    one, never the more comfortable one."""
    target_id = target_report.target_id
    if not target_report.step_results:
        return f"target {target_id}: no acquisition steps were executed", False
    for step_result in target_report.step_results:
        if step_result.status is StepStatus.FAILED:
            return (
                f"target {target_id}: step {step_result.step_id} FAILED: "
                f"{step_result.failure_reason or 'no reason recorded'}",
                False,
            )
    for step_result in target_report.step_results:
        if step_result.status is StepStatus.SKIPPED_DUE_TO_BUDGET:
            return (
                f"target {target_id}: step {step_result.step_id} skipped due to budget: "
                f"{step_result.failure_reason or 'no reason recorded'}",
                False,
            )
    for step_result in target_report.step_results:
        if step_result.status is StepStatus.ZERO_RESULTS:
            return f"target {target_id}: step {step_result.step_id} completed with zero results", True
    last = target_report.step_results[-1]
    return (
        f"target {target_id}: step {last.step_id} ended at {last.status} without producing a "
        "parsed literature payload",
        False,
    )


def _resolve_document(
    document_store: DocumentStore, document_id: str, pmid: str, unresolved_reasons: list[str]
) -> StoredDocument | None:
    """``DocumentStore.get`` plus a defensive ``version_history`` integrity
    check, both failure modes recorded (never raised past this module) --
    a corrupted version chain or a missing entry excludes only the ONE
    document it applies to, never the whole projection call."""
    try:
        document_store.version_history(document_id)
    except CorruptVersionChainError as exc:
        unresolved_reasons.append(f"PMID {pmid}: document {document_id} version chain corrupted: {exc}")
        return None
    stored = document_store.get(document_id)
    if stored is None:
        unresolved_reasons.append(f"PMID {pmid}: document_id {document_id} not found in DocumentStore")
        return None
    return stored


def _project_source(document: Document) -> Source:
    """Deterministic, lossless-within-bounds ``Source`` projection from a
    ``DocumentStore``-held ``Document``. Every date/identity field maps
    1:1 from the SAME-NAMED ``Document`` field -- never cross-assigned
    (``retrieved_at`` never becomes ``event_date``/``published_date``),
    never partially filled, and ``tier`` is the fixed
    ``LITERATURE_SOURCE_TIER`` constant, never derived from
    ``document.authority``. ``excerpt`` is left at its default (empty):
    the document body is never copied into it (see module docstring)."""
    return Source(
        source_id=make_source_id(document.url, document.title),
        url=document.url,
        title=document.title,
        tier=LITERATURE_SOURCE_TIER,
        publisher=document.publisher,
        published_date=document.published_date,
        event_date=document.event_date,
        effective_date=document.effective_date,
        filing_date=document.filing_date,
        accession=document.accession,
        retrieved_at=document.retrieved_at,
        provenance=document.provenance,
        content_hash=document.content_hash(),
        content_kind=document.content_kind,
    )


def _rebuild_europepmc_search_result(pmid: str, epmc_entry: dict[str, Any]) -> EuropePmcSearchResult | None:
    """Faithfully rebuilds the SAME ``EuropePmcSearchResult``
    ``PubMedLiteratureAdapter`` already computed, from the scalar fields it
    now exposes on its PARSE payload (Phase 4.1A additive change to
    ``literature_acquisition_adapter.py``) -- never inferred, never
    guessed. Returns ``None`` when any required scalar is absent (e.g. an
    ``ExecutionReport`` produced before that additive change existed), so
    the caller records an explicit unresolved reason instead of
    fabricating a result."""
    if any(key not in epmc_entry for key in _EUROPEPMC_REQUIRED_SCALAR_FIELDS):
        return None
    return EuropePmcSearchResult(
        pmid=pmid,
        pmcid=epmc_entry["pmcid"],
        doi=epmc_entry["doi"],
        title=epmc_entry["title"],
        is_open_access=bool(epmc_entry.get("is_open_access", False)),
        in_epmc=bool(epmc_entry.get("in_epmc", False)),
        license=epmc_entry.get("license", "") or "",
        journal_title=epmc_entry["journal_title"],
        pub_year=epmc_entry["pub_year"],
        source=epmc_entry["source"],
    )


__all__ = [
    "BRIDGE_COLLECTOR_LABEL",
    "LITERATURE_SOURCE_TIER",
    "LiteratureEvidenceProjection",
    "project_literature_target_reports",
]
