"""Append-only persistence layer.

Requirement 11/12: facts are never overwritten.  ``save_fact`` looks up the
highest stored version of a ``fact_id``; if the incoming content differs it
writes ``version + 1`` and marks the old row ``superseded_by`` the new version
key.  Identical content is a no-op (duplicate detection).
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..schemas.agent_io import AgentRunRecord, RunContext
from ..schemas.enums import (
    ContentKind,
    DocumentAuthority,
    EvidenceClass,
    FactCategory,
    Materiality,
    Provenance,
    SourceTier,
    VerifiedStatus,
)
from ..schemas.evaluation import (
    CatalystEvent,
    KillGateResult,
    Scenario,
    ScoreCard,
    Verdict,
)
from ..schemas.fact import Contradiction, Fact, Source
from ..schemas.validation import QuarantinedSource, SchemaError, validate_fact, validate_source
from ..scoring.identifier_validation import validate_strict_nct_id, validate_strict_pmid
from ..scoring.program_evidence import (
    CompanyIdentityEvidence,
    LiteratureCandidateEvidence,
    ProgramCandidateEvidence,
)

log = logging.getLogger(__name__)


def _split_csv(value: str | None) -> tuple[str, ...]:
    return tuple(value.split(",")) if value else ()


def _fact_from_row(row: sqlite3.Row | dict) -> Fact:
    """Reconstructs a Fact from a raw `facts` row, every column
    ``Fact.to_row()`` writes (Phase 4.3C correction 3/4) -- used by
    :meth:`Repository.facts_for_resume` and, since ``Fact.to_row()``'s own
    key/value shape survives a JSON round-trip unchanged (enums and ints
    are already stringified/int-ified by ``to_row()`` itself, and a plain
    ``dict`` supports the same ``row["key"]`` access a ``sqlite3.Row``
    does), also by :meth:`Repository.collect_snapshot_for_resume` and
    :meth:`Repository.verify_snapshot_for_resume` (Phase 4.3C correction
    5) to reconstruct a Fact from its own JSON snapshot payload -- one
    reconstruction implementation, two sources of the same row shape."""
    return Fact(
        fact_id=row["fact_id"],
        ticker=row["ticker"],
        category=FactCategory(row["category"]),
        claim=row["claim"],
        evidence_class=EvidenceClass(row["evidence_class"]),
        source_id=row["source_id"],
        source_url=row["source_url"],
        source_title=row["source_title"] or "",
        source_tier=SourceTier(row["source_tier"]),
        publication_date=row["publication_date"],
        event_date=row["event_date"],
        effective_date=row["effective_date"],
        filing_date=row["filing_date"],
        verified_status=VerifiedStatus(row["verified_status"]),
        confidence=float(row["confidence"]),
        company_claim=bool(row["company_claim"]),
        independent_confirmation=bool(row["independent_confirmation"]),
        corroborating_source_ids=_split_csv(row["corroborating_source_ids"]),
        contradicting_evidence=_split_csv(row["contradicting_evidence"]),
        materiality=Materiality(row["materiality"]),
        value=row["value"],
        unit=row["unit"],
        provenance=Provenance(row["provenance"]),
        stale=bool(row["stale"]),
        version=int(row["version"]),
        run_id=row["run_id"],
        notes=row["notes"] or "",
        tags=_split_csv(row["tags"]),
        content_kind=ContentKind(row["content_kind"]),
        primary_source_url=row["primary_source_url"],
        document_id=row["document_id"],
        source_authority=DocumentAuthority(row["source_authority"]),
    )


def _source_to_snapshot_dict(source: Source) -> dict:
    """Every ``Source`` field, losslessly -- deliberately NOT
    ``Source.to_row()``, which (confirmed by reading it directly) omits
    ``content_kind`` entirely; the ``sources`` table has no column for it
    either, so ``to_row()``/the ``sources`` table cannot round-trip a
    Source losslessly even in isolation. A resume snapshot must not
    inherit that gap, so this is a separate, complete serialization used
    only for :meth:`Repository.save_collect_snapshot`. Excerpt is kept in
    full (``to_row()`` truncates to 2000 chars for the regular ``sources``
    table row-size convention; a lossless snapshot does not truncate)."""
    return {
        "source_id": source.source_id,
        "url": source.url,
        "title": source.title,
        "tier": str(source.tier),
        "publisher": source.publisher,
        "published_date": source.published_date,
        "event_date": source.event_date,
        "effective_date": source.effective_date,
        "filing_date": source.filing_date,
        "accession": source.accession,
        "retrieved_at": source.retrieved_at,
        "provenance": str(source.provenance),
        "content_hash": source.content_hash,
        "syndicated_from": source.syndicated_from,
        "excerpt": source.excerpt,
        "content_kind": str(source.content_kind),
    }


def _source_from_snapshot_dict(d: dict) -> Source:
    return Source(
        source_id=d["source_id"],
        url=d["url"],
        title=d["title"],
        tier=SourceTier(d["tier"]),
        publisher=d["publisher"],
        published_date=d["published_date"],
        event_date=d["event_date"],
        effective_date=d["effective_date"],
        filing_date=d["filing_date"],
        accession=d["accession"],
        retrieved_at=d["retrieved_at"],
        provenance=Provenance(d["provenance"]),
        content_hash=d["content_hash"],
        syndicated_from=d["syndicated_from"],
        excerpt=d["excerpt"],
        content_kind=ContentKind(d["content_kind"]),
    )


def _quarantined_source_to_dict(q: QuarantinedSource) -> dict:
    return {"source_id": q.source_id, "url": q.url, "field": q.field, "value": q.value, "error": q.error}


def _quarantined_source_from_dict(d: dict) -> QuarantinedSource:
    return QuarantinedSource(
        source_id=d["source_id"], url=d["url"], field=d["field"], value=d["value"], error=d["error"],
    )


class _SnapshotFieldError(ValueError):
    """Phase 4.3F correction 2 Finding 5: raised by the strict field
    validators below on an EXPLICIT type/format/enum/identifier failure --
    a ``ValueError`` subclass, so it is already caught by the same
    ``except (..., ValueError)`` clause :meth:`Repository.
    collect_snapshot_for_resume` uses to raise ``ResumeSnapshotCorrupted``
    for the INCIDENTAL cases (a missing key raising ``KeyError``, an
    outright wrong-shaped value raising ``TypeError``). One exception
    hierarchy covers both."""


def _snap_str(d: dict, key: str) -> str:
    """A required string field -- never an int/bool/list/dict/None
    silently accepted where a string was expected."""
    value = d[key]
    if not isinstance(value, str):
        raise _SnapshotFieldError(f"{key!r} must be a string, got {type(value).__name__}")
    return value


def _snap_str_list(d: dict, key: str) -> tuple[str, ...]:
    """A JSON array of strings, as a tuple -- NEVER a bare string (which a
    bare ``tuple(value)`` would silently explode into one character per
    element) and never any other non-list value."""
    value = d[key]
    if not isinstance(value, list):
        raise _SnapshotFieldError(f"{key!r} must be a list, got {type(value).__name__}")
    for item in value:
        if not isinstance(item, str):
            raise _SnapshotFieldError(f"{key!r} entries must all be strings")
    return tuple(value)


def _snap_cik(d: dict, key: str) -> int | None:
    """``int`` or ``None`` only -- ``bool`` is explicitly rejected (Python
    treats ``bool`` as an ``int`` subclass, so an un-guarded ``isinstance
    (value, int)`` check alone would silently accept ``True``/``False`` as
    CIK 1/0)."""
    value = d[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise _SnapshotFieldError(f"{key!r} must be an int or null, got {type(value).__name__}")
    return value


def _snap_source_tier(d: dict, key: str) -> SourceTier:
    value = _snap_str(d, key)
    try:
        return SourceTier(value)
    except ValueError as exc:
        raise _SnapshotFieldError(f"{key!r} is not a valid SourceTier: {value!r}") from exc


def _snap_nct_id(d: dict, key: str) -> str:
    """Strict-valid NCT id only -- reuses ``scoring.identifier_validation.
    validate_strict_nct_id`` (the single source of truth for this format,
    per Phase 4.3F correction 1's own "never duplicate this regex"
    precedent), never a locally re-implemented pattern. The canonical
    (upper-cased, stripped) form is what is stored, mirroring the
    collector's own candidate-construction gate."""
    value = _snap_str(d, key)
    canonical = validate_strict_nct_id(value)
    if canonical is None:
        raise _SnapshotFieldError(f"{key!r} is not a strict-valid NCT id: {value!r}")
    return canonical


def _snap_nct_id_list(d: dict, key: str) -> tuple[str, ...]:
    values = _snap_str_list(d, key)
    canonical_ids: list[str] = []
    for value in values:
        canonical = validate_strict_nct_id(value)
        if canonical is None:
            raise _SnapshotFieldError(f"{key!r} entry is not a strict-valid NCT id: {value!r}")
        canonical_ids.append(canonical)
    return tuple(canonical_ids)


def _snap_pmid(d: dict, key: str) -> str:
    """Strict-valid PMID only -- reuses ``scoring.identifier_validation.
    validate_strict_pmid``, the same single source of truth as
    ``_snap_nct_id`` above."""
    value = _snap_str(d, key)
    canonical = validate_strict_pmid(value)
    if canonical is None:
        raise _SnapshotFieldError(f"{key!r} is not a strict-valid PMID: {value!r}")
    return canonical


#: Mirrors ``scoring.program_evidence._FACT_ID_RE`` exactly (``fact_`` +
#: 20 lowercase hex characters) -- duplicated, not imported (that name is
#: module-private there), following this codebase's own established
#: precedent for small format regexes shared across layers (see
#: ``scoring/identifier_validation.py``'s own module docstring on
#: ``NCT_ID_RE``/``PMID_RE``).
_SNAPSHOT_FACT_ID_RE = re.compile(r"^fact_[0-9a-f]{20}$")


def _snap_fact_ids(d: dict, key: str) -> tuple[str, ...]:
    """``supporting_fact_ids`` in its FORMAT-valid form only (``fact_`` +
    20 hex characters) -- this is a structural snapshot-decode check, not
    the referential one ``scoring.program_evidence.
    validate_program_candidate_evidence`` performs separately against an
    actual verified-facts context; that stays a later caller's job."""
    ids = _snap_str_list(d, key)
    for fact_id in ids:
        if not _SNAPSHOT_FACT_ID_RE.match(fact_id):
            raise _SnapshotFieldError(f"{key!r} entry is not a well-formed fact id: {fact_id!r}")
    return ids


def _snap_collection(payload: dict, key: str) -> list:
    """One of the three Phase 4.3F structured-evidence collections inside
    a collect-snapshot payload: a JSON array, or absent entirely (a
    snapshot saved before Phase 4.3F -- an empty list, never corrupted
    merely for lacking a key that did not exist yet when it was written).
    Present but NOT a list (a dict, a string, a number, ``null``) is
    corrupted -- Phase 4.3F correction 2 Finding 5: a bare string here
    would otherwise iterate character-by-character below, and a dict
    would iterate its keys -- neither is ever silently accepted as "zero
    or more records"."""
    if key not in payload:
        return []
    value = payload[key]
    if not isinstance(value, list):
        raise _SnapshotFieldError(f"{key!r} must be a JSON array, got {type(value).__name__}")
    return value


def _snap_record(value: object) -> dict:
    """One element of a structured-evidence collection: must itself be a
    JSON object -- never a bare string/number/list/``null`` masquerading
    as a record. Phase 4.3F correction 2 Finding 5: even one non-dict
    element makes the WHOLE collection (and so the whole snapshot)
    corrupted -- never silently skipped while keeping the others."""
    if not isinstance(value, dict):
        raise _SnapshotFieldError(
            f"structured-evidence record must be a JSON object, got {type(value).__name__}"
        )
    return value


# -- Phase 4.3F: structured primary-source evidence, snapshotted losslessly
# alongside Stage 1's own Facts/Sources -- see CollectSnapshot's own
# docstring. Every field of every type in scoring/program_evidence.py is
# covered explicitly; none of these three types is re-derived, guessed, or
# partially reconstructed on read-back.
#
# Phase 4.3F correction 2 Finding 5: every ``_..._from_dict`` below uses
# the strict ``_snap_*`` validators above, never a bare ``d["key"]``/
# ``tuple(d["key"])`` -- a value that is present but the WRONG shape (a
# string where a list was expected, a bool where an int-or-None was, an
# invalid enum member, a non-strict-valid identifier) is now an EXPLICIT
# ``_SnapshotFieldError`` (a ``ValueError``), not a value silently
# accepted or a failure that merely happens to raise incidentally.
def _company_identity_evidence_to_dict(evidence: CompanyIdentityEvidence) -> dict:
    return {
        "ticker": evidence.ticker,
        "cik": evidence.cik,
        "sec_official_name": evidence.sec_official_name,
        "explicitly_verified_aliases": list(evidence.explicitly_verified_aliases),
        "source_id": evidence.source_id,
        "source_tier": str(evidence.source_tier),
        "retrieved_at": evidence.retrieved_at,
        "content_hash": evidence.content_hash,
    }


def _company_identity_evidence_from_dict(d: dict) -> CompanyIdentityEvidence:
    return CompanyIdentityEvidence(
        ticker=_snap_str(d, "ticker"),
        cik=_snap_cik(d, "cik"),
        sec_official_name=_snap_str(d, "sec_official_name"),
        explicitly_verified_aliases=_snap_str_list(d, "explicitly_verified_aliases"),
        source_id=_snap_str(d, "source_id"),
        source_tier=_snap_source_tier(d, "source_tier"),
        retrieved_at=_snap_str(d, "retrieved_at"),
        content_hash=_snap_str(d, "content_hash"),
    )


def _program_candidate_evidence_to_dict(evidence: ProgramCandidateEvidence) -> dict:
    return {
        "nct_id": evidence.nct_id,
        "lead_sponsor": evidence.lead_sponsor,
        "collaborators": list(evidence.collaborators),
        "interventions": list(evidence.interventions),
        "conditions": list(evidence.conditions),
        "overall_status": evidence.overall_status,
        "phases": list(evidence.phases),
        "primary_completion_date": evidence.primary_completion_date,
        "completion_date": evidence.completion_date,
        "first_posted_date": evidence.first_posted_date,
        "source_id": evidence.source_id,
        "source_tier": str(evidence.source_tier),
        "retrieved_at": evidence.retrieved_at,
        "content_hash": evidence.content_hash,
        "supporting_fact_ids": list(evidence.supporting_fact_ids),
    }


def _program_candidate_evidence_from_dict(d: dict) -> ProgramCandidateEvidence:
    return ProgramCandidateEvidence(
        nct_id=_snap_nct_id(d, "nct_id"),
        lead_sponsor=_snap_str(d, "lead_sponsor"),
        collaborators=_snap_str_list(d, "collaborators"),
        interventions=_snap_str_list(d, "interventions"),
        conditions=_snap_str_list(d, "conditions"),
        overall_status=_snap_str(d, "overall_status"),
        phases=_snap_str_list(d, "phases"),
        primary_completion_date=_snap_str(d, "primary_completion_date"),
        completion_date=_snap_str(d, "completion_date"),
        first_posted_date=_snap_str(d, "first_posted_date"),
        source_id=_snap_str(d, "source_id"),
        source_tier=_snap_source_tier(d, "source_tier"),
        retrieved_at=_snap_str(d, "retrieved_at"),
        content_hash=_snap_str(d, "content_hash"),
        supporting_fact_ids=_snap_fact_ids(d, "supporting_fact_ids"),
    )


def _literature_candidate_evidence_to_dict(evidence: LiteratureCandidateEvidence) -> dict:
    return {
        "pmid": evidence.pmid,
        "nct_ids": list(evidence.nct_ids),
        "source_id": evidence.source_id,
        "source_tier": str(evidence.source_tier),
        "retrieved_at": evidence.retrieved_at,
        "content_hash": evidence.content_hash,
    }


def _literature_candidate_evidence_from_dict(d: dict) -> LiteratureCandidateEvidence:
    return LiteratureCandidateEvidence(
        pmid=_snap_pmid(d, "pmid"),
        nct_ids=_snap_nct_id_list(d, "nct_ids"),
        source_id=_snap_str(d, "source_id"),
        source_tier=_snap_source_tier(d, "source_tier"),
        retrieved_at=_snap_str(d, "retrieved_at"),
        content_hash=_snap_str(d, "content_hash"),
    )


class ResumeSnapshotCorrupted(Exception):
    """Phase 4.3C correction 5: raised by
    :meth:`Repository.collect_snapshot_for_resume`/
    :meth:`Repository.verify_snapshot_for_resume` when a snapshot row
    exists but cannot be parsed back losslessly. Deliberately never
    swallowed into ``None`` (which would read as "no snapshot was ever
    saved", a materially different, less alarming condition) or into an
    empty result (which would silently discard whatever the snapshot
    actually held) -- ``Pipeline.run()`` catches this specifically and
    fails closed."""


@dataclass
class CollectSnapshot:
    """A run-scoped, lossless snapshot of Stage 1's own output: the
    pre-Integrity ``Fact``s :meth:`Repository.save_collect_snapshot` was
    given, every ``Source`` they reference, and per-collector metadata
    (never the raw per-collector ``RawFact``/``Source`` breakdown, which
    Stage 1 has already reduced into the two lists above by the time this
    is saved).

    Phase 4.3F: also carries the structured, typed primary-source evidence
    (``scoring/program_evidence.py``) any collector this run built --
    flattened across every ``CollectionResult``, exactly like ``facts``/
    ``sources`` already are, rather than left nested inside the opaque
    ``collectors`` diagnostic dicts below (which are never reconstructed
    into typed objects on read-back). A resumed run that restores this
    snapshot gets these back as real, usable ``CompanyIdentityEvidence``/
    ``ProgramCandidateEvidence``/``LiteratureCandidateEvidence`` objects,
    not opaque dicts -- this phase does not yet validate or resolve them,
    but a later phase that does must never have to re-derive them from a
    diagnostic string."""

    facts: list[Fact] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    #: One dict per original CollectionResult: collector, outcome,
    #: provenance, errors, attempted_urls, notes, zero_results,
    #: raw_fact_count_before_dedup -- the "failure/degraded/zero-results
    #: semantics" Phase 4.3C correction 5 requires preserved losslessly.
    collectors: list[dict] = field(default_factory=list)
    #: Phase 4.3F: flattened across every CollectionResult this run had,
    #: exactly like facts/sources above.
    company_identity_evidence: list[CompanyIdentityEvidence] = field(default_factory=list)
    program_candidate_evidence: list[ProgramCandidateEvidence] = field(default_factory=list)
    literature_candidate_evidence: list[LiteratureCandidateEvidence] = field(default_factory=list)


@dataclass
class VerifySnapshot:
    """A run-scoped, lossless snapshot of Stage 2's own output: the
    post-Integrity verified ``Fact``s, the ``QuarantinedSource``s Source
    validation rejected, and the (possibly Literature-consistency-
    corrected) ``direct_acquisition_info`` mapping."""

    verified_facts: list[Fact] = field(default_factory=list)
    quarantined_sources: list[QuarantinedSource] = field(default_factory=list)
    direct_acquisition_info: dict[str, Any] = field(default_factory=dict)


_COLLECT_SNAPSHOT_STAGE = "_collect_snapshot"
_VERIFY_SNAPSHOT_STAGE = "_verify_snapshot"
#: Sentinel stage_index for the two synthetic stages above: never a real
#: value from orchestrator.resume.STAGES (which only ever indexes
#: non-negative), and orchestrator.resume.stage_index()/resume_point()
#: only ever look at stages that are IN STAGES -- these two names never
#: are, by construction, so this value is never read as meaningful, only
#: stored to satisfy run_checkpoints.stage_index's NOT NULL constraint.
_SNAPSHOT_STAGE_INDEX = -1

_CONTENT_FIELDS = (
    "claim",
    "value",
    "evidence_class",
    "verified_status",
    "confidence",
    "materiality",
    "independent_confirmation",
    "contradicting_evidence",
    "source_url",
    "event_date",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Repository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # -- companies ---------------------------------------------------------
    def upsert_company(
        self,
        ticker: str,
        company_name: str = "UNKNOWN",
        *,
        cik: str = "UNKNOWN",
        exchange: str = "UNKNOWN",
        country: str = "UNKNOWN",
        sector: str = "UNKNOWN",
        aliases: Sequence[str] = (),
    ) -> None:
        now = _now()
        self.conn.execute(
            """
            INSERT INTO companies(ticker, company_name, cik, exchange, country, sector,
                                  aliases, first_seen, last_seen)
            VALUES(?,?,?,?,?,?,?,?,?)
            ON CONFLICT(ticker) DO UPDATE SET
                company_name=excluded.company_name,
                cik=excluded.cik,
                exchange=excluded.exchange,
                country=excluded.country,
                sector=excluded.sector,
                aliases=excluded.aliases,
                last_seen=excluded.last_seen
            """,
            (
                ticker.upper(),
                company_name,
                cik,
                exchange,
                country,
                sector,
                ",".join(aliases),
                now,
                now,
            ),
        )
        self.conn.commit()

    def get_company(self, ticker: str) -> sqlite3.Row | None:
        cur = self.conn.execute("SELECT * FROM companies WHERE ticker = ?", (ticker.upper(),))
        return cur.fetchone()

    # -- runs --------------------------------------------------------------
    def start_run(self, ctx: RunContext) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO runs(run_id, ticker, mode, status, offline,
                                           used_fixtures, started_at, finished_at, notes)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                ctx.run_id,
                ctx.ticker,
                ctx.mode,
                str(ctx.status),
                int(ctx.offline),
                int(ctx.use_fixtures),
                ctx.started_at or _now(),
                "",
                "\n".join(ctx.notes),
            ),
        )
        self.conn.commit()

    def finish_run(self, ctx: RunContext) -> None:
        self.conn.execute(
            "UPDATE runs SET status = ?, finished_at = ?, notes = ? WHERE run_id = ?",
            (str(ctx.status), _now(), "\n".join(ctx.notes), ctx.run_id),
        )
        self.conn.commit()

    def previous_run(self, ticker: str, exclude_run_id: str) -> sqlite3.Row | None:
        cur = self.conn.execute(
            """SELECT * FROM runs WHERE ticker = ? AND run_id != ?
               ORDER BY started_at DESC LIMIT 1""",
            (ticker.upper(), exclude_run_id),
        )
        return cur.fetchone()

    # -- sources -----------------------------------------------------------
    def save_source(self, source: Source) -> None:
        validate_source(source)
        row = source.to_row()
        cols = ",".join(row)
        marks = ",".join("?" * len(row))
        self.conn.execute(
            f"INSERT OR REPLACE INTO sources({cols}) VALUES({marks})", tuple(row.values())
        )

    def save_sources(self, sources: Iterable[Source]) -> list[QuarantinedSource]:
        """Persist every valid source; quarantine, never crash on, a bad one.

        Requirement 14/24: schema integrity stays strict -- a malformed
        source is never silently persisted -- but one bad record must not
        abort the whole run. Returns the sources that were rejected, each
        with enough detail (source_id, url, field, offending value, error)
        for the run to report exactly what was dropped and why.
        """
        quarantined: list[QuarantinedSource] = []
        for s in sources:
            try:
                self.save_source(s)
            except SchemaError as exc:
                log.error("quarantined malformed source %s (%s): %s", s.source_id, s.url, exc)
                quarantined.append(
                    QuarantinedSource(
                        source_id=s.source_id,
                        url=s.url,
                        field=exc.field,
                        value=exc.value,
                        error=str(exc),
                    )
                )
        self.conn.commit()
        return quarantined

    def find_source_by_hash(self, content_hash: str) -> sqlite3.Row | None:
        if not content_hash or content_hash == "UNKNOWN":
            return None
        cur = self.conn.execute(
            "SELECT * FROM sources WHERE content_hash = ? LIMIT 1", (content_hash,)
        )
        return cur.fetchone()

    # -- facts -------------------------------------------------------------
    def latest_fact_row(self, fact_id: str) -> sqlite3.Row | None:
        cur = self.conn.execute(
            "SELECT * FROM facts WHERE fact_id = ? ORDER BY version DESC LIMIT 1", (fact_id,)
        )
        return cur.fetchone()

    def save_fact(self, fact: Fact) -> tuple[str, int]:
        """Persist a fact append-only.  Returns (fact_id, stored_version)."""
        validate_fact(fact)
        existing = self.latest_fact_row(fact.fact_id)
        row = fact.to_row()
        if existing is not None:
            unchanged = all(str(existing[f]) == str(row[f]) for f in _CONTENT_FIELDS)
            if unchanged:
                return fact.fact_id, int(existing["version"])  # duplicate: no-op
            new_version = int(existing["version"]) + 1
            row["version"] = new_version
            # mark predecessor superseded (allowed: not a content column)
            self.conn.execute(
                "UPDATE facts SET superseded_by = ? WHERE fact_id = ? AND version = ?",
                (f"{fact.fact_id}#v{new_version}", fact.fact_id, existing["version"]),
            )
        row["created_at"] = _now()
        cols = ",".join(row)
        marks = ",".join("?" * len(row))
        self.conn.execute(f"INSERT INTO facts({cols}) VALUES({marks})", tuple(row.values()))
        self.conn.commit()
        return fact.fact_id, int(row["version"])

    def save_facts(self, facts: Iterable[Fact]) -> int:
        count = 0
        for f in facts:
            self.save_fact(f)
            count += 1
        return count

    def latest_facts(self, ticker: str) -> list[sqlite3.Row]:
        cur = self.conn.execute(
            """SELECT f.* FROM facts f
               JOIN (SELECT fact_id, MAX(version) AS v FROM facts WHERE ticker = ?
                     GROUP BY fact_id) m
               ON f.fact_id = m.fact_id AND f.version = m.v
               ORDER BY f.category, f.fact_id""",
            (ticker.upper(),),
        )
        return list(cur.fetchall())

    def fact_versions(self, fact_id: str) -> list[sqlite3.Row]:
        cur = self.conn.execute(
            "SELECT * FROM facts WHERE fact_id = ? ORDER BY version", (fact_id,)
        )
        return list(cur.fetchall())

    def facts_for_run(self, run_id: str) -> list[sqlite3.Row]:
        cur = self.conn.execute("SELECT * FROM facts WHERE run_id = ?", (run_id,))
        return list(cur.fetchall())

    # -- contradictions ----------------------------------------------------
    def save_contradictions(self, items: Iterable[Contradiction]) -> int:
        n = 0
        for c in items:
            row = c.to_row()
            row["created_at"] = _now()
            cols = ",".join(row)
            marks = ",".join("?" * len(row))
            self.conn.execute(
                f"INSERT OR REPLACE INTO contradictions({cols}) VALUES({marks})",
                tuple(row.values()),
            )
            n += 1
        self.conn.commit()
        return n

    # -- structured domain tables -----------------------------------------
    def save_capital_structure(self, run_id: str, ticker: str, data: dict[str, Any]) -> None:
        payload = {
            "run_id": run_id,
            "ticker": ticker.upper(),
            "created_at": _now(),
            **{k: v for k, v in data.items() if k not in ("run_id", "ticker", "created_at")},
        }
        cols = ",".join(payload)
        marks = ",".join("?" * len(payload))
        self.conn.execute(
            f"INSERT OR REPLACE INTO capital_structure({cols}) VALUES({marks})",
            tuple(payload.values()),
        )
        self.conn.commit()

    def save_clinical_trials(self, run_id: str, ticker: str, trials: Iterable[dict]) -> int:
        n = 0
        for t in trials:
            payload = {"run_id": run_id, "ticker": ticker.upper(), "created_at": _now(), **t}
            cols = ",".join(payload)
            marks = ",".join("?" * len(payload))
            self.conn.execute(
                f"INSERT OR REPLACE INTO clinical_trials({cols}) VALUES({marks})",
                tuple(payload.values()),
            )
            n += 1
        self.conn.commit()
        return n

    def save_regulatory_events(self, run_id: str, ticker: str, events: Iterable[dict]) -> int:
        n = 0
        for e in events:
            payload = {"run_id": run_id, "ticker": ticker.upper(), "created_at": _now(), **e}
            cols = ",".join(payload)
            marks = ",".join("?" * len(payload))
            self.conn.execute(
                f"INSERT OR REPLACE INTO regulatory_events({cols}) VALUES({marks})",
                tuple(payload.values()),
            )
            n += 1
        self.conn.commit()
        return n

    def save_insider_trades(self, run_id: str, ticker: str, trades: Iterable[dict]) -> int:
        n = 0
        for t in trades:
            payload = {"run_id": run_id, "ticker": ticker.upper(), "created_at": _now(), **t}
            cols = ",".join(payload)
            marks = ",".join("?" * len(payload))
            self.conn.execute(
                f"INSERT OR REPLACE INTO insider_trades({cols}) VALUES({marks})",
                tuple(payload.values()),
            )
            n += 1
        self.conn.commit()
        return n

    def save_catalysts(self, run_id: str, ticker: str, events: Iterable[CatalystEvent]) -> int:
        n = 0
        for e in events:
            payload = {
                "run_id": run_id,
                "ticker": ticker.upper(),
                "created_at": _now(),
                **e.to_row(),
            }
            cols = ",".join(payload)
            marks = ",".join("?" * len(payload))
            self.conn.execute(
                f"INSERT OR REPLACE INTO catalysts({cols}) VALUES({marks})",
                tuple(payload.values()),
            )
            n += 1
        self.conn.commit()
        return n

    # -- evaluation outputs ------------------------------------------------
    def save_scores(self, card: ScoreCard) -> int:
        n = 0
        capped = set(card.capped_by_kill_gate)
        for dim, value in card.scores.items():
            self.conn.execute(
                """INSERT OR REPLACE INTO scores(run_id, ticker, dimension, score,
                                                 confidence, rationale, capped_by_kill_gate,
                                                 created_at)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (
                    card.run_id,
                    card.ticker,
                    dim,
                    value,
                    card.per_score_confidence.get(dim, card.evidence_confidence),
                    card.rationale.get(dim, ""),
                    int(dim in capped),
                    _now(),
                ),
            )
            n += 1
        self.conn.commit()
        return n

    def save_kill_gate(self, run_id: str, ticker: str, gate: KillGateResult) -> int:
        n = 0
        for a in gate.assessments:
            self.conn.execute(
                """INSERT OR REPLACE INTO kill_assessments(run_id, ticker, category, level,
                                                           rationale, evidence_confidence,
                                                           confirmation, created_at)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    ticker.upper(),
                    str(a.category),
                    str(a.level),
                    a.rationale,
                    a.evidence_confidence,
                    str(a.confirmation),
                    _now(),
                ),
            )
            n += 1
        self.conn.commit()
        return n

    def save_evidence_sufficiency(self, run_id: str, ticker: str, matrix) -> int:
        """Persist the Evidence Sufficiency Matrix (requirement DG5/DG10)."""
        n = 0
        for domain, entry in matrix.domains.items():
            self.conn.execute(
                """INSERT OR REPLACE INTO evidence_sufficiency(run_id, ticker, domain,
                       search_status, evidence_sufficiency_status, decision_grade_fact_count,
                       total_fact_count, reason, created_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    ticker.upper(),
                    str(domain),
                    str(entry.search_status),
                    str(entry.evidence_sufficiency_status),
                    entry.decision_grade_fact_count,
                    entry.total_fact_count,
                    entry.reason,
                    _now(),
                ),
            )
            n += 1
        self.conn.commit()
        return n

    def save_scenarios(self, run_id: str, ticker: str, scenarios: Iterable[Scenario]) -> int:
        n = 0
        for s in scenarios:
            payload = {
                "run_id": run_id,
                "ticker": ticker.upper(),
                "created_at": _now(),
                **s.to_row(),
            }
            cols = ",".join(payload)
            marks = ",".join("?" * len(payload))
            self.conn.execute(
                f"INSERT OR REPLACE INTO scenarios({cols}) VALUES({marks})",
                tuple(payload.values()),
            )
            n += 1
        self.conn.commit()
        return n

    def save_agent_run(self, record: AgentRunRecord) -> None:
        row = record.to_row()
        cols = ",".join(row)
        marks = ",".join("?" * len(row))
        self.conn.execute(
            f"INSERT OR REPLACE INTO agent_runs({cols}) VALUES({marks})", tuple(row.values())
        )
        self.conn.commit()

    # -- phase 2: checkpoints, coverage, llm calls, escalations -----------
    def save_checkpoint(self, checkpoint) -> None:
        from ..orchestrator.resume import serialize_payload

        self.conn.execute(
            """INSERT OR REPLACE INTO run_checkpoints(run_id, stage, stage_index, status,
                                                      payload, fact_count, created_at)
               VALUES(?,?,?,?,?,?,?)""",
            (
                checkpoint.run_id,
                checkpoint.stage,
                checkpoint.stage_index,
                checkpoint.status,
                serialize_payload(checkpoint.payload),
                checkpoint.fact_count,
                _now(),
            ),
        )
        self.conn.commit()

    def checkpoints(self, run_id: str) -> list:
        from ..orchestrator.resume import Checkpoint

        rows = self.conn.execute(
            "SELECT * FROM run_checkpoints WHERE run_id = ? ORDER BY stage_index", (run_id,)
        ).fetchall()
        out = []
        for row in rows:
            try:
                payload = json.loads(row["payload"] or "{}")
            except json.JSONDecodeError:
                payload = {}
            out.append(
                Checkpoint(
                    run_id=row["run_id"],
                    stage=row["stage"],
                    stage_index=int(row["stage_index"]),
                    status=row["status"],
                    payload=payload if isinstance(payload, dict) else {},
                    fact_count=int(row["fact_count"]),
                )
            )
        return out

    def facts_for_resume(self, run_id: str) -> list:
        """Rehydrate the LATEST version of every fact recorded for a run --
        i.e. whatever this run's own persistence has most recently written
        for each fact_id: the verified/post-Integrity version once "verify"
        has completed, but possibly still the pristine pre-Integrity
        version (or a genuine mix, if a crash landed mid-persist-loop) when
        it has not. Phase 4.3C correction 4: callers deciding whether to
        feed Evidence Integrity an already-processed fact must NOT use
        this method for that -- see :meth:`pristine_facts_for_resume`,
        which answers a different question ("what did THIS run's own
        collect stage actually produce, before Evidence Integrity touched
        it at all") that this method cannot answer once even one fact has
        advanced past version 1.

        Phase 4.3C correction 3: this reconstructs EVERY column
        ``Fact.to_row()`` writes, not a subset -- a prior version of this
        method silently dropped ``corroborating_source_ids``,
        ``contradicting_evidence``, ``tags``, ``content_kind``,
        ``primary_source_url``, ``document_id`` and ``source_authority``
        back to their dataclass defaults on every restore, even though all
        seven are real, persisted columns (confirmed by reading
        ``storage/schema.sql`` and ``Fact.to_row()`` directly). A restored
        fact whose real stored ``content_kind`` was ``SEARCH_SUMMARY``, or
        whose real ``source_authority``/``document_id`` were set, would
        silently read back as ``FULL_DOCUMENT``/``UNKNOWN``/``None`` after
        a resume -- a genuine semantic corruption a fact-id-only comparison
        can never catch. ``superseded_by`` is deliberately NOT restored:
        the query already selects the MAX(version) row for each fact_id,
        so a genuinely-latest row's own ``superseded_by`` is always NULL.
        """
        rows = self.conn.execute(
            """SELECT f.* FROM facts f
               JOIN (SELECT fact_id, MAX(version) AS v FROM facts WHERE run_id = ?
                     GROUP BY fact_id) m
               ON f.fact_id = m.fact_id AND f.version = m.v""",
            (run_id,),
        ).fetchall()
        return [_fact_from_row(row) for row in rows]

    # -- run-scoped resume snapshots (Phase 4.3C correction 5) -------------
    #
    # Replaces Phase 4.3C correction 4's MIN(version)/MAX(version) queries
    # against the shared `facts` table entirely. Both were proven, by
    # direct empirical reproduction (this correction's own completion
    # report), to silently return NOTHING for a run_id whose Stage-1 (or
    # verified) content happened to be byte-identical to what an earlier,
    # different run_id had already saved -- `Repository.save_fact` looks
    # up the latest row for a fact_id GLOBALLY, never scoped by run_id,
    # and a content-identical write is a no-op that writes no new row at
    # all. `run_checkpoints`, by contrast, has `run_id` IN its own PRIMARY
    # KEY (run_id, stage) -- genuinely, structurally immune to this
    # collision -- so a full, lossless snapshot is stored there instead,
    # via a dedicated stage name outside orchestrator.resume.STAGES (never
    # interfering with the ordinary stage-index resume-point calculation)
    # and its own JSON serialization (never
    # orchestrator.resume.serialize_payload's 100_000-character
    # truncation, which is safe for the small bookkeeping dicts every
    # OTHER checkpoint stores but would silently corrupt a real fact/
    # source snapshot). `sources` cannot be the snapshot's own source of
    # truth either: confirmed by reading storage/schema.sql directly, that
    # table has no run_id column AT ALL, and `save_source` is an
    # unconditional `INSERT OR REPLACE` keyed only by source_id -- no
    # versioning, no history, so a later run (or the SAME run re-collecting
    # on --resume) silently overwrites an earlier one's row with no way to
    # tell whose content is currently there.
    def save_collect_snapshot(
        self,
        run_id: str,
        facts: Sequence[Fact],
        sources: Sequence[Source],
        collectors: Sequence[dict],
        *,
        company_identity_evidence: Sequence[CompanyIdentityEvidence] = (),
        program_candidate_evidence: Sequence[ProgramCandidateEvidence] = (),
        literature_candidate_evidence: Sequence[LiteratureCandidateEvidence] = (),
    ) -> None:
        """Persists Stage 1's own pristine (pre-Integrity) output, in
        full, keyed only by this run_id. Called once, right after Stage 1
        computes its own facts/sources -- never merged with, or replaced
        by, ANY other run_id's data.

        Phase 4.3F: the three structured-evidence sequences default to
        empty so this stays callable exactly as before for any run that
        built none -- unchanged behavior, unchanged snapshot shape, for
        every collector that does not populate them.
        """
        payload = {
            "facts": [f.to_row() for f in facts],
            "company_identity_evidence": [
                _company_identity_evidence_to_dict(e) for e in company_identity_evidence
            ],
            "program_candidate_evidence": [
                _program_candidate_evidence_to_dict(e) for e in program_candidate_evidence
            ],
            "literature_candidate_evidence": [
                _literature_candidate_evidence_to_dict(e) for e in literature_candidate_evidence
            ],
            "sources": [_source_to_snapshot_dict(s) for s in sources],
            "collectors": list(collectors),
        }
        self.conn.execute(
            """INSERT OR REPLACE INTO run_checkpoints(run_id, stage, stage_index, status,
                                                      payload, fact_count, created_at)
               VALUES(?,?,?,?,?,?,?)""",
            (
                run_id, _COLLECT_SNAPSHOT_STAGE, _SNAPSHOT_STAGE_INDEX, "OK",
                json.dumps(payload, default=str), len(facts), _now(),
            ),
        )
        self.conn.commit()

    def collect_snapshot_for_resume(self, run_id: str) -> CollectSnapshot | None:
        """``None`` when no collect snapshot was ever saved for this
        run_id (a run predating Phase 4.3C correction 5, or one that never
        reached Stage 1's own persist step at all) -- never a guess.
        Raises :class:`ResumeSnapshotCorrupted` when a row exists but
        cannot be parsed back losslessly; never silently returns an empty
        or partial snapshot."""
        row = self.conn.execute(
            "SELECT payload FROM run_checkpoints WHERE run_id = ? AND stage = ?",
            (run_id, _COLLECT_SNAPSHOT_STAGE),
        ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row["payload"])
            facts = [_fact_from_row(r) for r in payload["facts"]]
            sources = [_source_from_snapshot_dict(r) for r in payload["sources"]]
            collectors = list(payload["collectors"])
            # Phase 4.3F: a snapshot saved BEFORE this phase (a run that
            # predates these keys entirely) is a genuinely valid,
            # un-corrupted snapshot that simply has nothing structured to
            # restore, never a corrupted one merely for lacking a key that
            # did not exist yet when it was written -- _snap_collection
            # returns [] for a genuinely absent key. Phase 4.3F correction
            # 2 Finding 5: a key that IS present but not a JSON array
            # (_snap_collection), or an element of it that is not itself a
            # JSON object (_snap_record), or a record whose own fields
            # fail their strict per-type/format/enum/identifier check
            # (the ``_..._from_dict`` functions, via the ``_snap_*``
            # validators) -- ANY of these makes the WHOLE snapshot
            # corrupted, never a partial record silently dropped while
            # the rest of the collection is kept.
            company_identity_evidence = [
                _company_identity_evidence_from_dict(_snap_record(r))
                for r in _snap_collection(payload, "company_identity_evidence")
            ]
            program_candidate_evidence = [
                _program_candidate_evidence_from_dict(_snap_record(r))
                for r in _snap_collection(payload, "program_candidate_evidence")
            ]
            literature_candidate_evidence = [
                _literature_candidate_evidence_from_dict(_snap_record(r))
                for r in _snap_collection(payload, "literature_candidate_evidence")
            ]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ResumeSnapshotCorrupted(
                f"collect snapshot for run_id={run_id!r} is corrupted: {exc}"
            ) from exc
        return CollectSnapshot(
            facts=facts,
            sources=sources,
            collectors=collectors,
            company_identity_evidence=company_identity_evidence,
            program_candidate_evidence=program_candidate_evidence,
            literature_candidate_evidence=literature_candidate_evidence,
        )

    def save_verify_snapshot(
        self,
        run_id: str,
        verified_facts: Sequence[Fact],
        quarantined_sources: Sequence[QuarantinedSource],
        direct_acquisition_info: dict[str, Any],
    ) -> None:
        """Persists Stage 2's own verified output, in full, keyed only by
        this run_id. Called once, right after the full Evidence Integrity
        pass computes its own verified_facts/quarantined_sources -- never
        merged with, or replaced by, ANY other run_id's data."""
        payload = {
            "verified_facts": [f.to_row() for f in verified_facts],
            "quarantined_sources": [_quarantined_source_to_dict(q) for q in quarantined_sources],
            "direct_acquisition_info": dict(direct_acquisition_info),
        }
        self.conn.execute(
            """INSERT OR REPLACE INTO run_checkpoints(run_id, stage, stage_index, status,
                                                      payload, fact_count, created_at)
               VALUES(?,?,?,?,?,?,?)""",
            (
                run_id, _VERIFY_SNAPSHOT_STAGE, _SNAPSHOT_STAGE_INDEX, "OK",
                json.dumps(payload, default=str), len(verified_facts), _now(),
            ),
        )
        self.conn.commit()

    def verify_snapshot_for_resume(self, run_id: str) -> VerifySnapshot | None:
        """``None`` when no verify snapshot was ever saved for this
        run_id -- never a guess. Raises :class:`ResumeSnapshotCorrupted`
        when a row exists but cannot be parsed back losslessly."""
        row = self.conn.execute(
            "SELECT payload FROM run_checkpoints WHERE run_id = ? AND stage = ?",
            (run_id, _VERIFY_SNAPSHOT_STAGE),
        ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row["payload"])
            verified_facts = [_fact_from_row(r) for r in payload["verified_facts"]]
            quarantined = [_quarantined_source_from_dict(r) for r in payload["quarantined_sources"]]
            direct_acquisition_info = dict(payload["direct_acquisition_info"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ResumeSnapshotCorrupted(
                f"verify snapshot for run_id={run_id!r} is corrupted: {exc}"
            ) from exc
        return VerifySnapshot(
            verified_facts=verified_facts,
            quarantined_sources=quarantined,
            direct_acquisition_info=direct_acquisition_info,
        )

    def save_research_coverage(self, run_id: str, ticker: str, coverage) -> int:
        n = 0
        for domain, entry in coverage.items():
            self.conn.execute(
                """INSERT OR REPLACE INTO research_coverage(run_id, ticker, domain, status,
                       queries_attempted, queries_executed, documents_found, paths, detail,
                       created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    ticker.upper(),
                    str(domain),
                    str(entry.status),
                    entry.queries_attempted,
                    entry.queries_executed,
                    entry.documents_found,
                    ",".join(str(p) for p in entry.paths),
                    entry.detail,
                    _now(),
                ),
            )
            n += 1
        self.conn.commit()
        return n

    def save_llm_calls(self, run_id: str, calls) -> int:
        n = 0
        for call in calls:
            self.conn.execute(
                """INSERT INTO llm_calls(run_id, agent_id, model, input_tokens, output_tokens,
                       cache_read_tokens, duration_ms, ok, stop_reason, error, created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    call.agent_id,
                    call.model,
                    call.input_tokens,
                    call.output_tokens,
                    call.cache_read_tokens,
                    call.duration_ms,
                    int(call.ok),
                    call.stop_reason,
                    call.error[:500],
                    _now(),
                ),
            )
            n += 1
        self.conn.commit()
        return n

    def save_escalations(self, run_id: str, attempts) -> int:
        n = 0
        for attempt in attempts:
            self.conn.execute(
                """INSERT OR REPLACE INTO escalations(run_id, fact_id, reason, searched,
                       confirmed, confirming_url, queries, note, created_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    attempt.fact_id,
                    attempt.reason,
                    int(attempt.searched),
                    int(attempt.confirmed),
                    attempt.confirming_url,
                    " | ".join(attempt.queries)[:1000],
                    attempt.note,
                    _now(),
                ),
            )
            n += 1
        self.conn.commit()
        return n

    # -- search discovery log (requirements M2/M3) ------------------------
    def save_discovery_log(self, discovery) -> tuple[int, int]:
        """Persist a DiscoveryLog: queries and hits, kept apart from facts."""
        n_queries = 0
        for query in discovery.queries:
            row = query.to_row()
            cols = ",".join(row)
            marks = ",".join("?" * len(row))
            self.conn.execute(
                f"INSERT OR REPLACE INTO search_queries({cols}) VALUES({marks})",
                tuple(row.values()),
            )
            n_queries += 1
        n_hits = 0
        for hit in discovery.hits:
            row = hit.to_row()
            cols = ",".join(row)
            marks = ",".join("?" * len(row))
            self.conn.execute(
                f"INSERT OR REPLACE INTO search_hits({cols}) VALUES({marks})",
                tuple(row.values()),
            )
            n_hits += 1
        self.conn.commit()
        return n_queries, n_hits

    def discovery_queries(self, run_id: str, purposes: tuple[str, ...] | None = None):
        if purposes:
            placeholders = ",".join("?" * len(purposes))
            return self.conn.execute(
                f"SELECT * FROM search_queries WHERE run_id = ? "
                f"AND query_purpose IN ({placeholders}) ORDER BY created_at",
                (run_id, *purposes),
            ).fetchall()
        return self.conn.execute(
            "SELECT * FROM search_queries WHERE run_id = ? ORDER BY created_at", (run_id,)
        ).fetchall()

    def discovery_hits(self, run_id: str, purposes: tuple[str, ...] | None = None):
        if purposes:
            placeholders = ",".join("?" * len(purposes))
            return self.conn.execute(
                f"SELECT * FROM search_hits WHERE run_id = ? "
                f"AND query_purpose IN ({placeholders}) ORDER BY rank",
                (run_id, *purposes),
            ).fetchall()
        return self.conn.execute(
            "SELECT * FROM search_hits WHERE run_id = ? ORDER BY rank", (run_id,)
        ).fetchall()

    def log_fetch(
        self,
        run_id: str,
        url: str,
        collector: str,
        outcome: str,
        http_status: int | None = None,
        attempts: int = 1,
        detail: str = "",
    ) -> None:
        self.conn.execute(
            """INSERT INTO fetch_log(run_id, url, collector, outcome, http_status,
                                     attempts, detail, created_at)
               VALUES(?,?,?,?,?,?,?,?)""",
            (run_id, url, collector, outcome, http_status, attempts, detail[:500], _now()),
        )
        self.conn.commit()

    # -- thesis versioning -------------------------------------------------
    def latest_thesis(self, ticker: str) -> sqlite3.Row | None:
        cur = self.conn.execute(
            "SELECT * FROM thesis_versions WHERE ticker = ? ORDER BY version DESC LIMIT 1",
            (ticker.upper(),),
        )
        return cur.fetchone()

    def save_thesis_version(
        self,
        ticker: str,
        run_id: str,
        verdict: Verdict,
        card: ScoreCard,
        diff: dict[str, Any],
        snapshot: dict[str, Any],
    ) -> int:
        prev = self.latest_thesis(ticker)
        version = 1 if prev is None else int(prev["version"]) + 1
        self.conn.execute(
            """INSERT INTO thesis_versions(ticker, version, run_id, action, evidence_confidence,
                                           headline, max_kill_level, what_changed, why_changed,
                                           new_facts, removed_assumptions, score_change,
                                           snapshot, research_status,
                                           blocking_verification_required, created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                ticker.upper(),
                version,
                run_id,
                str(verdict.action),
                verdict.evidence_confidence,
                verdict.headline,
                str(verdict.kill_gate.max_level),
                json.dumps(diff.get("WHAT_CHANGED", []), ensure_ascii=False),
                json.dumps(diff.get("WHY_CHANGED", []), ensure_ascii=False),
                json.dumps(diff.get("NEW_FACT", []), ensure_ascii=False),
                json.dumps(diff.get("REMOVED_ASSUMPTION", []), ensure_ascii=False),
                json.dumps(diff.get("SCORE_CHANGE", {}), ensure_ascii=False),
                json.dumps(snapshot, ensure_ascii=False, default=str),
                str(getattr(verdict, "research_status", "COMPLETE")),
                json.dumps(
                    list(getattr(verdict, "blocking_verification_required", ())),
                    ensure_ascii=False,
                ),
                _now(),
            ),
        )
        self.conn.commit()
        return version
