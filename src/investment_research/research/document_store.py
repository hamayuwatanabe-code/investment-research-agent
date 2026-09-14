"""Document identity, versioning and duplicate-content tracking.

Phase 1 scope only: this module is pure, offline, in-memory bookkeeping. It
performs no network I/O and is not wired into the pipeline, any collector, or
any live/corpus execution path this turn.

The problem this solves: an SEC accession number identifies a *filing
submission*, not a document. One accession commonly contains a filing-index
page, a primary document, one or more exhibits, and an XBRL instance -- each
retrieved separately, each with its own text, and each capable of settling (or
failing to settle) a different question. Treating "accession" as if it were a
document's identity is how a filing-index listing page and the 10-K text it
points at end up conflated as "the same document" merely because they share an
accession number.

So identity here is deliberately layered: (accession, filename) narrows a
document within one submission; ``document_role`` distinguishes what kind of
thing was retrieved (an index page is not a primary document, even before
looking at ``ContentKind``); ``canonical_url`` (or the document's own URL)
drives *versioning* -- a body change at the same URL produces a new version,
never an in-place overwrite, so a Fact that already cited the old body keeps
resolving to exactly what it cited.

Deviation from the Phase 0B design note (reported per the Phase 1 review
requirement): :class:`~investment_research.collectors.documents.Document` has
no ``filename``/``document_role``/``sequence``/``canonical_url``/``version``/
``previous_version_id`` fields, and ``collectors/documents.py`` is out of
scope for this change. Rather than extend ``Document`` itself, this module
wraps it in :class:`StoredDocument`, which carries a ``Document`` plus the
identity/versioning metadata the store needs. ``StoredDocument.authority``
delegates to the wrapped ``Document.authority`` -- there is exactly one
canonical place authority lives.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum

from ..collectors.documents import Document
from ..schemas.enums import DocumentAuthority


def derive_document_id(prefix: str, url: str, text: str) -> str:
    """The single source of truth for computing a ``document_id`` from a
    document's identity (URL) AND its content. Every adapter that stores a
    ``Document`` here -- SEC primary/exhibit, ClinicalTrials, and any future
    one -- must call this rather than hand-rolling its own hash, so the
    content-dependence invariant below can never be silently dropped by a
    new adapter (Phase 3D.1 requirement 2/5: no adapter-specific id rule).

    URL-only derivation is what caused a real bug during Phase 3D: two
    fetches of the SAME URL whose CONTENT differs -- a genuine registry
    update -- produced the SAME id, since the id depended only on the
    unchanged URL. :meth:`DocumentStore.put` then tried to create a "new
    version" whose ``previous_version_id`` pointed at that SAME id: the old
    and new ``StoredDocument`` shared one ``document_id``, an immediate
    self-reference that made :meth:`DocumentStore.version_history` loop
    forever. Content dependence fixes this at the source: identical content
    naturally reproduces the identical id, which ``put()`` already treats
    as idempotent (harmless, no version is created), while different
    content always produces a different one, which ``put()`` can now also
    verify is not already anywhere in the URL's own version chain (see
    ``put()``'s ``DocumentIdentityError`` checks).

    ``prefix`` is a short, adapter-chosen label purely for human
    readability when an id shows up in logs/tests (e.g. ``"sec_primary"``,
    ``"clinicaltrials"``) -- it carries no semantic weight and is never
    used for lookup, comparison, or versioning logic.
    """
    digest = hashlib.sha256(f"{url}\x00{text}".encode()).hexdigest()[:24]
    return f"doc_{prefix}_{digest}"


class DocumentIdentityError(Exception):
    """Raised when a caller-supplied ``document_id`` would corrupt
    ``DocumentStore``'s version-chain invariants -- a self-referencing or
    cyclic ``previous_version_id`` chain is never silently allowed through
    (Phase 3D.1 requirement 3). The fix is almost always to derive
    ``document_id`` from content as well as URL -- see
    :func:`derive_document_id`."""


class CorruptVersionChainError(Exception):
    """Raised by :meth:`DocumentStore.version_history` if a document's
    ``previous_version_id`` chain contains a cycle -- never silently
    truncated or treated as a complete, valid history, and never read as
    if the chain's target were ACQUIRED (Phase 3D.1 requirement 3)."""


class DocumentRole(str, Enum):
    """What kind of thing was retrieved -- distinct from ``ContentKind``
    (how much of it is in hand) and from ``SourceTier`` (how much it is
    trusted). An index page and the primary document it points at can both be
    ``FULL_DOCUMENT`` content-kind while being wholly different roles.
    """

    FILING_INDEX = "FILING_INDEX"
    PRIMARY_DOCUMENT = "PRIMARY_DOCUMENT"
    EXHIBIT = "EXHIBIT"
    XBRL_INSTANCE = "XBRL_INSTANCE"
    STRUCTURED_API_RECORD = "STRUCTURED_API_RECORD"
    PRESS_RELEASE = "PRESS_RELEASE"
    JOURNAL_ARTICLE = "JOURNAL_ARTICLE"
    UNKNOWN = "UNKNOWN"


def _normalize_filename(filename: str) -> str:
    return (filename or "").strip().lower()


@dataclass(frozen=True)
class StoredDocument:
    """One identity-resolved, versioned entry in the :class:`DocumentStore`."""

    document_id: str
    document: Document
    accession: str
    filename: str
    document_role: DocumentRole
    sequence: int = 0
    canonical_url: str = ""
    version: int = 1
    #: The document_id of the version this one replaced, or ``None`` for the
    #: first version. The predecessor stays fully retrievable via
    #: ``DocumentStore.get`` under its own id -- it is never overwritten.
    previous_version_id: str | None = None

    @property
    def authority(self) -> DocumentAuthority:
        return self.document.authority

    def content_hash(self) -> str:
        return self.document.content_hash()


@dataclass(frozen=True)
class DuplicateContentRelation:
    """Two or more documents share a content hash. Recorded, never acted on.

    Identical bytes do not imply identical identity: a press release mirrored
    verbatim on the issuer's own site and on a wire service is the same text
    from two different speakers. Collapsing them into one Document on hash
    alone would silently launder away exactly the distinction
    ``DocumentAuthority`` exists to preserve. ``same_authority`` records
    whether that risk is actually present for this particular relation; it is
    informational only and never triggers a merge.
    """

    content_hash: str
    document_ids: tuple[str, ...]
    same_authority: bool


class DocumentStore:
    """In-memory, offline document identity/version registry.

    Five lookup indexes, each answering a different question that "just use
    the accession number" cannot:

    * ``_by_id`` -- the only true identity: one row per ``document_id``,
      including every superseded version.
    * ``_by_accession_file`` -- ``(accession, normalized filename)`` to the
      *latest* document_id at that key. Distinguishes the filing-index page
      from the primary document from an exhibit within one accession, since
      each has its own filename.
    * ``_by_canonical_url`` -- URL to every document_id ever registered at
      it, oldest first, so a version history can be walked without a
      separate structure.
    * ``_by_content_hash`` -- content hash to the *set* of document_ids
      sharing it (never a single winner), feeding
      :meth:`duplicate_content_relations`.
    * ``_by_accession`` -- accession to the set of every document_id filed
      under it. Deliberately a set, and :meth:`resolve_by_accession` returns
      a list, never a single Document: an accession is a filing submission,
      not a document (requirement 1).
    """

    def __init__(self) -> None:
        self._by_id: dict[str, StoredDocument] = {}
        self._by_accession_file: dict[tuple[str, str], str] = {}
        self._by_canonical_url: dict[str, list[str]] = {}
        self._by_content_hash: dict[str, set[str]] = {}
        self._by_accession: dict[str, set[str]] = {}
        #: Bookkeeping only -- not one of the five identity indexes above.
        #: Tracks the latest document_id registered at a given URL so `put`
        #: can decide whether new content at that URL is a new version.
        self._latest_by_url: dict[str, str] = {}
        self._duplicate_relations: dict[str, DuplicateContentRelation] = {}

    def put(
        self,
        document: Document,
        *,
        document_id: str,
        accession: str = "UNKNOWN",
        filename: str = "",
        document_role: DocumentRole = DocumentRole.UNKNOWN,
        sequence: int = 0,
        canonical_url: str = "",
    ) -> StoredDocument:
        """Register a document, versioning it if new content arrives at a
        URL this store has already seen.

        Versioning is keyed on URL (``canonical_url`` if given, else
        ``document.url``), not on ``(accession, filename)``: a re-fetch of
        the same address with a changed body is the case requirement 3
        describes ("same-URL body change"), and using accession/filename
        for this instead would wrongly chain together every document that
        happens to share an empty filename under an unknown accession (the
        common case for non-SEC documents).

        * Same URL, identical content hash: idempotent -- the existing
          :class:`StoredDocument` is returned unchanged, no new version.
        * Same URL, different content hash: a new ``document_id``/version is
          created with ``previous_version_id`` pointing at the prior one,
          which remains registered under its own id and fully retrievable.
        * A new URL: a first version (``version=1``, no predecessor).

        SEC amendments (e.g. a 10-K/A) are, in principle, filed under their
        *own* accession number -- they are not modeled as a new version of
        the original filing's accession here; that falls out naturally from
        versioning being URL-keyed rather than accession-keyed. A different
        filename within the same accession (a filing index vs. its primary
        document vs. an exhibit) is likewise never treated as a version of
        one another -- each gets its own URL and thus its own identity.
        """
        url_key = canonical_url or document.url
        file_key = (accession, _normalize_filename(filename))
        new_hash = document.content_hash()

        existing_id = self._latest_by_url.get(url_key)
        if existing_id is not None:
            existing = self._by_id[existing_id]
            if existing.content_hash() == new_hash:
                return existing
            if document_id == existing.document_id:
                raise DocumentIdentityError(
                    f"refusing to create a new version of {url_key!r}: the supplied "
                    f"document_id ({document_id!r}) is identical to the version it would "
                    "replace, which would make previous_version_id self-reference the same "
                    "document. Derive document_id from CONTENT as well as URL -- see "
                    "derive_document_id()."
                )
            # Defense in depth: the new id must not already appear anywhere
            # in the existing version chain, not merely differ from the
            # immediate predecessor -- a non-adjacent collision would still
            # eventually create a cycle.
            ancestor_ids = {v.document_id for v in self.version_history(existing.document_id)}
            if document_id in ancestor_ids:
                raise DocumentIdentityError(
                    f"document_id {document_id!r} already exists earlier in {url_key!r}'s "
                    "own version chain -- refusing to create a cycle."
                )
            stored = StoredDocument(
                document_id=document_id,
                document=document,
                accession=accession,
                filename=filename,
                document_role=document_role,
                sequence=sequence,
                canonical_url=canonical_url,
                version=existing.version + 1,
                previous_version_id=existing.document_id,
            )
        else:
            stored = StoredDocument(
                document_id=document_id,
                document=document,
                accession=accession,
                filename=filename,
                document_role=document_role,
                sequence=sequence,
                canonical_url=canonical_url,
            )

        self._index(stored, file_key, url_key)
        return stored

    def _index(self, stored: StoredDocument, file_key: tuple[str, str], url_key: str) -> None:
        self._by_id[stored.document_id] = stored
        self._by_accession_file[file_key] = stored.document_id
        self._by_canonical_url.setdefault(url_key, []).append(stored.document_id)
        self._latest_by_url[url_key] = stored.document_id

        content_hash = stored.content_hash()
        bucket = self._by_content_hash.setdefault(content_hash, set())
        if bucket:
            self._record_duplicate_relation(content_hash, bucket, stored)
        bucket.add(stored.document_id)

        if stored.accession and stored.accession != "UNKNOWN":
            self._by_accession.setdefault(stored.accession, set()).add(stored.document_id)

    def _record_duplicate_relation(
        self, content_hash: str, existing_ids: set[str], new_doc: StoredDocument
    ) -> None:
        authorities = {self._by_id[doc_id].authority for doc_id in existing_ids}
        authorities.add(new_doc.authority)
        all_ids = tuple(sorted({*existing_ids, new_doc.document_id}))
        self._duplicate_relations[content_hash] = DuplicateContentRelation(
            content_hash=content_hash,
            document_ids=all_ids,
            same_authority=len(authorities) == 1,
        )

    # -- lookups -------------------------------------------------------
    def get(self, document_id: str) -> StoredDocument | None:
        """Resolve one exact document_id, whatever version it is."""
        return self._by_id.get(document_id)

    def resolve_by_accession_and_file(
        self, accession: str, filename: str
    ) -> StoredDocument | None:
        """The latest version at one (accession, filename) key."""
        doc_id = self._by_accession_file.get((accession, _normalize_filename(filename)))
        return self._by_id.get(doc_id) if doc_id else None

    def resolve_by_accession(self, accession: str) -> list[StoredDocument]:
        """Every document filed under one accession.

        Always a list, never a single Document (requirement 1): a filing
        index page, its primary document, and any exhibits all share one
        accession number while being distinct documents with distinct roles.
        """
        return [self._by_id[doc_id] for doc_id in sorted(self._by_accession.get(accession, ()))]

    def resolve_by_canonical_url(self, url: str) -> list[StoredDocument]:
        """Every version ever registered at one URL, oldest first."""
        return [self._by_id[doc_id] for doc_id in self._by_canonical_url.get(url, [])]

    def version_history(self, document_id: str) -> list[StoredDocument]:
        """The chain ending at ``document_id``, oldest first.

        ``document_id`` need not be the latest version -- a superseded
        version is still fully retrievable and still produces its own
        correct (shorter) history.

        Raises :class:`CorruptVersionChainError` if the chain ever revisits
        a ``document_id`` already seen -- ``put()``'s own checks should
        make this impossible to construct through this store's public API,
        but this walk never silently truncates or treats a corrupt chain as
        complete (Phase 3D.1 requirement 3): a broken chain is reported as
        broken, never quietly read as if its target were valid/ACQUIRED.
        """
        chain: list[StoredDocument] = []
        seen: set[str] = set()
        current = self._by_id.get(document_id)
        while current is not None:
            if current.document_id in seen:
                raise CorruptVersionChainError(
                    f"version chain for {document_id!r} revisits {current.document_id!r} -- "
                    "a cycle exists in previous_version_id links; never silently truncated."
                )
            seen.add(current.document_id)
            chain.append(current)
            current = (
                self._by_id.get(current.previous_version_id)
                if current.previous_version_id
                else None
            )
        return list(reversed(chain))

    def duplicate_content_relations(self) -> list[DuplicateContentRelation]:
        return list(self._duplicate_relations.values())
