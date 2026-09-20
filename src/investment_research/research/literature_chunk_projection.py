"""Literature Chunk Projection Bridge (Phase 4.1B).

Offline-only and pure -- this module itself still calls no network code
and imports nothing from ``cli.py``/``pipeline.py``. Production
reachability: Phase 4.2A explicitly authorized
``research/literature_pipeline_integration.py`` to call
``project_literature_chunks`` as one step of a larger, flag-gated
production path -- see ``literature_evidence_projection.py``'s own module
docstring for the exact same note, which applies here identically. This
module is never called directly by ``cli.py``/``pipeline.py`` themselves.

What this module does: takes the ``document_ids`` a Phase 4.1A
``LiteratureEvidenceProjection`` call already resolved a ``Source``/
``RawFact`` pair from, plus the SAME run's ``DocumentStore``, and projects
each eligible ``Document`` into the EXISTING, canonical chunk corpus shape
-- ``collectors.documents.Chunk``, built exclusively by ``chunk_document()``
-- so a future interpretive agent has something to read besides raw
``Fact`` claims. It invents no new chunk type, no new chunk-id scheme, and
no PubMed-XML-specific chunker: every ``Chunk`` this module produces is
exactly what ``chunk_document()`` would build from the same ``Document``,
nothing more.

Hard boundaries (do not weaken these to "simplify" a future change; raise
the question instead, per CLAUDE.md):

* Never invents a second ``Chunk`` type or a second chunk-id scheme.
  ``chunk_document()`` (``collectors/documents.py``) is the only chunker
  called here, exactly as it already is by ``collectors/extraction.py``'s
  ``DocumentCollector`` -- this module's only job is choosing WHICH stored
  Documents are eligible to be chunked and recording what happened.
* ``Chunk.document`` is always the ORIGINAL ``Document`` this module read
  out of ``DocumentStore`` -- never a copy, never a rebuilt/summarized
  stand-in. ``Chunk.doc_id`` always equals that ``Document.doc_id``
  (``chunk_document()``'s own construction already guarantees this; this
  module additionally checks it, defense in depth, per the explicit
  integrity-failure requirement below).
* Chunk-eligibility is decided by exactly one discriminator:
  ``document.content_kind.is_primary_text`` (``True`` for
  ``FULL_DOCUMENT``/``EXCERPT``, ``False`` for ``SEARCH_SUMMARY``/
  ``METADATA_ONLY``). A ``METADATA_ONLY`` Europe PMC search-response
  document -- Phase 4.1A's own addition -- is excluded here exactly as it
  is excluded from ``Fact.is_decision_grade``, for the same reason: its
  body was never actually read, so there is no primary text to chunk.
  Exclusion is recorded explicitly (``excluded_non_primary_document_ids``),
  never silently dropped, and never treated as if the body-fetch had
  succeeded.
* Never summarizes, supplements, rewrites, or re-derives ``DocumentStore``
  body text. ``chunk_document()`` slices the SAME ``document.text`` this
  module read from the store; nothing here transforms it.
* Never duplicates a document's body text into an unresolved reason, a log
  line, or this module's own diagnostics -- every message below names the
  document/chunk id and the failure mode, never the text itself.
* ``document_ids`` input is deduplicated while preserving encounter order,
  and no ``Document`` is ever chunked twice -- mirrors
  ``LiteratureEvidenceProjection``'s own ``seen_pmids``/
  ``duplicate_source_count`` pattern exactly (see
  ``literature_evidence_projection.py``), here as ``seen_document_ids``/
  ``duplicate_document_count``.
* Same input (the same ``document_ids`` tuple against the same
  ``DocumentStore`` state) always produces the same ``Chunk`` ids in the
  same order -- ``chunk_document()`` is itself deterministic, and this
  module never reorders or randomizes its own iteration.

Integrity failures -- never silently skipped, never falling back to a
different Document, each recorded as an explicit unresolved reason with
``coverage_complete`` set ``False``:

* ``document_id`` not present in ``DocumentStore`` (``.get()`` returns
  ``None``).
* A corrupted version chain (``DocumentStore.version_history`` raises
  ``CorruptVersionChainError`` -- the exact same check
  ``literature_evidence_projection.py``'s own ``_resolve_document`` helper
  performs, in the same order: the chain is walked BEFORE ``.get()`` is
  trusted, so a self-reference/cycle is caught here as well, since
  ``version_history`` is precisely where that cycle would surface).
* A primary-text (``is_primary_text`` True) ``Document`` whose body is
  empty. ``chunk_document()`` itself silently returns ``[]`` for empty
  text -- indistinguishable, from its own return value alone, from "this
  Document legitimately produced zero chunks because it is entirely
  boilerplate". This module does NOT rely on that silent behavior: it
  checks ``document.text`` for emptiness BEFORE calling ``chunk_document``
  and records the gap explicitly, rather than letting a body-fetch that
  produced nothing pass as if it had never been attempted.
* ``Chunk.doc_id`` / ``Document.doc_id`` mismatch. Structurally impossible
  given ``chunk_document()``'s own construction (it always sets
  ``chunk_id``/``doc_id`` from the SAME ``document.doc_id`` argument it was
  called with) -- checked anyway, per the explicit requirement that this
  never be assumed rather than verified.
* A duplicate chunk id. Also structurally impossible per document (each
  chunk's ``index`` is unique within one ``chunk_document()`` call) and
  across documents once ``document_ids`` dedup has run (each Document is
  chunked at most once, and ``chunk_id`` is prefixed by the Document's own
  ``doc_id``) -- checked anyway, defense in depth, never assumed.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from ..collectors.documents import Chunk, chunk_document
from .document_store import CorruptVersionChainError, DocumentStore

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiteratureChunkProjection:
    """Everything one chunk-projection call produced, plus the diagnostics
    needed to audit it -- mirrors ``LiteratureEvidenceProjection``'s own
    shape (Phase 4.1A) as closely as the different responsibility allows.

    ``projected_document_ids`` is every ``DocumentStore`` identity this call
    actually chunked (successfully, primary-text, non-empty) -- in
    encounter order, never containing a duplicate.

    ``excluded_non_primary_document_ids`` is every ``document_id`` this call
    deliberately did NOT chunk because ``content_kind.is_primary_text`` was
    ``False`` (``SEARCH_SUMMARY``/``METADATA_ONLY``) -- an expected,
    non-error exclusion, distinct from an integrity failure. It never
    overlaps with ``projected_document_ids`` or with any document_id named
    in ``unresolved_reasons``.
    """

    chunks: tuple[Chunk, ...]
    projected_document_ids: tuple[str, ...]
    excluded_non_primary_document_ids: tuple[str, ...]
    unresolved_reasons: tuple[str, ...]
    coverage_complete: bool
    #: How many times a document_id this call had already chunked (from a
    #: duplicate entry in the input ``document_ids``) was encountered again
    #: and correctly skipped rather than re-chunked. Zero on a call whose
    #: input carried no duplicates.
    duplicate_document_count: int = 0


def project_literature_chunks(
    document_ids: Sequence[str],
    document_store: DocumentStore,
    *,
    target_tokens: int = 400,
    overlap_sentences: int = 1,
) -> LiteratureChunkProjection:
    """Project ``document_ids`` (typically a Phase 4.1A
    ``LiteratureEvidenceProjection.document_ids``) into a canonical
    ``Chunk`` corpus, reading each ``Document`` from ``document_store``.

    Never raises on a single bad document_id: every failure mode this
    module can detect is recorded as an unresolved reason and processing
    continues with whatever else in the batch is still sound -- mirroring
    ``project_literature_target_reports``'s own partial-failure philosophy
    (see ``literature_evidence_projection.py``), never CLAUDE.md rule 7's
    forbidden alternative (silently filling the gap).
    """
    chunks: list[Chunk] = []
    projected_document_ids: list[str] = []
    excluded_non_primary_document_ids: list[str] = []
    unresolved_reasons: list[str] = []

    seen_document_ids: set[str] = set()
    seen_chunk_ids: set[str] = set()
    duplicate_document_count = 0
    coverage_complete = True

    for document_id in document_ids:
        if document_id in seen_document_ids:
            duplicate_document_count += 1
            continue
        seen_document_ids.add(document_id)

        try:
            document_store.version_history(document_id)
        except CorruptVersionChainError as exc:
            coverage_complete = False
            unresolved_reasons.append(
                f"document {document_id}: version chain corrupted: {exc}"
            )
            continue

        stored = document_store.get(document_id)
        if stored is None:
            coverage_complete = False
            unresolved_reasons.append(
                f"document_id {document_id} not found in DocumentStore"
            )
            continue

        document = stored.document
        if document.doc_id != document_id:
            # Structurally should never happen -- StoredDocument.document is
            # the same Document object put() was given for this identity --
            # but never trusted silently; see module docstring.
            coverage_complete = False
            unresolved_reasons.append(
                f"document {document_id}: StoredDocument.document.doc_id "
                f"({document.doc_id!r}) does not match the requested document_id -- "
                "excluded, never chunked under a mismatched identity"
            )
            continue

        if not document.content_kind.is_primary_text:
            excluded_non_primary_document_ids.append(document_id)
            continue

        if not (document.text or "").strip():
            coverage_complete = False
            unresolved_reasons.append(
                f"document {document_id}: content_kind={document.content_kind} is primary "
                "text but its body is empty -- excluded, never silently treated as zero "
                "legitimate chunks"
            )
            continue

        document_chunks = chunk_document(
            document, target_tokens=target_tokens, overlap_sentences=overlap_sentences
        )

        document_had_chunk_problem = False
        accepted_chunks: list[Chunk] = []
        for chunk in document_chunks:
            if chunk.doc_id != document_id:
                coverage_complete = False
                document_had_chunk_problem = True
                unresolved_reasons.append(
                    f"document {document_id}: chunk {chunk.chunk_id} carries doc_id "
                    f"{chunk.doc_id!r} -- excluded, never trusted"
                )
                continue
            if chunk.chunk_id in seen_chunk_ids:
                coverage_complete = False
                document_had_chunk_problem = True
                unresolved_reasons.append(
                    f"document {document_id}: duplicate chunk_id {chunk.chunk_id} -- excluded"
                )
                continue
            seen_chunk_ids.add(chunk.chunk_id)
            accepted_chunks.append(chunk)

        chunks.extend(accepted_chunks)
        projected_document_ids.append(document_id)
        if document_had_chunk_problem:
            log.warning(
                "literature_chunk_projection: document %s produced at least one rejected "
                "chunk; %d of %d chunks accepted",
                document_id,
                len(accepted_chunks),
                len(document_chunks),
            )

    return LiteratureChunkProjection(
        chunks=tuple(chunks),
        projected_document_ids=tuple(projected_document_ids),
        excluded_non_primary_document_ids=tuple(excluded_non_primary_document_ids),
        unresolved_reasons=tuple(unresolved_reasons),
        coverage_complete=coverage_complete,
        duplicate_document_count=duplicate_document_count,
    )


__all__ = [
    "LiteratureChunkProjection",
    "project_literature_chunks",
]
