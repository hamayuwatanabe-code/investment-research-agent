"""Phase 3D.4: a shared Capture Manifest -- the ONE place that persists the
ACTUAL UTC timestamp a live HTTP capture was retrieved at, so an offline
``analyze_capture()`` re-run never has to guess a retrieval time from a
directory name, a file's mtime, or an operator-supplied ``--as-of``.

Used by BOTH ``sec_live_smoke.py`` and ``clinicaltrials_live_smoke.py``
through the one shared ``AllowlistedHttpClient._save_response`` -- one
manifest type, one persistence rule, for every live capture this repository
ever makes (Phase 3D.4 requirement 4). Deliberately excludes anything
secret or identifying: no User-Agent, no API key, no email address, no
request/response headers (Phase 3D.4 requirement 3) -- only what an
offline re-analysis genuinely needs: which URL, what came back, when, and a
hash to catch a saved body that no longer matches what was captured.

Phase 3D.4.1: a manifest can fail to be trustworthy in several DIFFERENT
ways -- absent, corrupt, an unrecognized schema, or present-but-disagreeing
with the body actually on disk -- and treating all of them as one opaque
``None`` erased exactly the distinction Evidence Integrity auditing needs
(a missing manifest is an unremarkable pre-3D.4 capture; a manifest whose
hash disagrees with its own body is a live integrity failure). This module
now returns a typed ``ManifestReadResult`` naming exactly which case
applies, instead of collapsing them all to ``None``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

#: Bumped only if the manifest's own field set changes shape -- read_manifest
#: never guesses at a mismatched/legacy schema (Phase 3D.4 requirement 11:
#: an existing capture with no manifest at all, or a manifest of an
#: unrecognized schema, is backward-compatible by degrading to UNKNOWN, not
#: by crashing or half-parsing).
CAPTURE_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CaptureManifest:
    """Everything -- and ONLY everything -- an offline re-analysis needs to
    know about how one saved response body was actually captured."""

    schema_version: int
    source: str  # e.g. "sec" | "clinicaltrials" -- never a secret value
    requested_url: str
    final_url: str
    http_status: int | None
    #: UTC ISO 8601, e.g. "2026-09-14T12:34:56Z" -- the REAL time this body
    #: was retrieved, recorded at the moment of capture. Never derived from
    #: a directory name, a file's mtime, or any caller-supplied "as of"
    #: value (Phase 3D.4 requirements 1/7).
    capture_retrieved_at: str
    content_hash: str  # sha256 hex digest of the saved body bytes, exactly
    content_length: int


def compute_content_hash(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def manifest_path_for(capture_dir: Path, digest: str) -> Path:
    return capture_dir / f"{digest}.manifest.json"


def write_manifest(capture_dir: Path, digest: str, manifest: CaptureManifest) -> None:
    """Atomic write: a reader must never observe a partially-written
    manifest (Phase 3D.4 requirement 10). Writes to a temp file in the same
    directory (so the final ``os.replace`` stays on one filesystem, making
    it atomic) and only then renames it into place; a crash or concurrent
    read mid-write either sees the old file (absent, on first write) or the
    complete new one -- never a truncated one."""
    capture_dir.mkdir(parents=True, exist_ok=True)
    final_path = manifest_path_for(capture_dir, digest)
    tmp_path = capture_dir / f".{digest}.manifest.json.tmp"
    tmp_path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    os.replace(tmp_path, final_path)


class ManifestReadStatus(str, Enum):
    """Every distinguishable reason a Capture Manifest is, or is not, usable
    evidence -- deliberately NOT collapsed into a single ``None`` (Phase
    3D.4.1 requirement 1). Only ``VERIFIED`` licenses using the manifest's
    own ``capture_retrieved_at``; every other value means ``UNKNOWN``."""

    #: The manifest file exists, parses, matches the supported schema, AND
    #: its content_hash/content_length agree with the body actually on
    #: disk right now. The only status a caller may treat as trustworthy.
    VERIFIED = "VERIFIED"
    #: No manifest file at all -- an unremarkable, pre-Phase-3D.4 capture
    #: (or anything else that only ever saved the raw body). NOT an
    #: Evidence Integrity failure; explicitly a legacy capture (requirement
    #: 5) as long as the body itself is present.
    MISSING = "MISSING"
    #: The manifest file exists but is not valid JSON, or its JSON is not
    #: an object. An Evidence Integrity failure (requirement 5) -- never
    #: treated the same as MISSING (requirement 6).
    MALFORMED = "MALFORMED"
    #: The manifest is valid JSON but its fields (or schema_version) do not
    #: match a schema this code understands. An Evidence Integrity failure.
    UNSUPPORTED_SCHEMA = "UNSUPPORTED_SCHEMA"
    #: The manifest's own content_hash disagrees with the body's actual
    #: hash (and content_length agrees, or was not the deciding factor --
    #: see verify_manifest_against_body's priority rule). An Evidence
    #: Integrity failure.
    HASH_MISMATCH = "HASH_MISMATCH"
    #: The manifest's own content_length disagrees with the body's actual
    #: length -- checked, and reported, before the hash (Phase 3D.4.1
    #: requirement 3). An Evidence Integrity failure.
    CONTENT_LENGTH_MISMATCH = "CONTENT_LENGTH_MISMATCH"
    #: A manifest exists (and is itself well-formed) but no captured body
    #: file exists to verify it against, or to analyze at all (Phase
    #: 3D.4.1 requirement 7). Distinct from MISSING (which is "no manifest
    #: at all") and from a normal "nothing captured for this category"
    #: (both manifest and body absent, which callers report via their own
    #: existing "not found"/"missing category" semantics, never via this
    #: status at all). An Evidence Integrity failure (Phase 3D.4.1.1
    #: correction: the manifest asserts a capture was made, but the
    #: artifact it describes cannot be re-verified and is not on disk to
    #: analyze at all -- unlike MISSING, this is not "an ordinary capture
    #: from before manifests existed", it is a claimed capture that is now
    #: unsubstantiated).
    BODY_MISSING = "BODY_MISSING"
    #: The manifest file exists but could not be READ (a filesystem-level
    #: error, e.g. a permissions problem) -- distinct from MALFORMED
    #: (readable but not valid JSON/schema). An Evidence Integrity failure.
    IO_ERROR = "IO_ERROR"


#: Statuses that represent a genuine Evidence Integrity problem -- as
#: opposed to MISSING (the one status that is an unremarkable legacy
#: capture, never a failure) or VERIFIED (Phase 3D.4.1 requirement 5,
#: corrected in Phase 3D.4.1.1: BODY_MISSING belongs here too -- a
#: manifest that claims a capture was made, for an artifact that cannot
#: actually be found or re-verified, is exactly the kind of unsubstantiated
#: evidence claim this classification exists to catch, not something to
#: wave through as harmless). The single source of truth for "is this
#: status a failure", so every caller (and every LIVE_VERIFIED-exclusion
#: check) agrees, rather than each re-deriving its own list.
EVIDENCE_INTEGRITY_FAILURE_STATUSES: frozenset[ManifestReadStatus] = frozenset({
    ManifestReadStatus.MALFORMED,
    ManifestReadStatus.UNSUPPORTED_SCHEMA,
    ManifestReadStatus.HASH_MISMATCH,
    ManifestReadStatus.CONTENT_LENGTH_MISMATCH,
    ManifestReadStatus.BODY_MISSING,
    ManifestReadStatus.IO_ERROR,
})


def is_evidence_integrity_failure(status: ManifestReadStatus) -> bool:
    return status in EVIDENCE_INTEGRITY_FAILURE_STATUSES


class VerificationStatus(str, Enum):
    """The outcome of comparing a manifest to the ACTUAL body bytes on
    disk, in isolation from every manifest-file-level concern (missing,
    malformed, unsupported schema) that ``ManifestReadStatus`` also
    covers (Phase 3D.4.1 requirement 3)."""

    VERIFIED = "VERIFIED"
    HASH_MISMATCH = "HASH_MISMATCH"
    CONTENT_LENGTH_MISMATCH = "CONTENT_LENGTH_MISMATCH"


def verify_manifest_against_body(manifest: CaptureManifest, body: bytes) -> VerificationStatus:
    """Compares the manifest's own recorded ``content_length`` and
    ``content_hash`` against the ACTUAL bytes on disk right now. A caller
    must never treat anything but ``VERIFIED`` as a normal, verified
    capture (Phase 3D.4 requirement 9) -- the body file could have been
    edited, truncated, or swapped after the manifest was written, and a
    manifest that no longer describes its own body is not trustworthy
    evidence of anything, including its own ``capture_retrieved_at``.

    Explicit priority when BOTH disagree (Phase 3D.4.1 requirement 3):
    ``content_length`` is checked FIRST and reported alone if it disagrees
    -- a length mismatch is already sufficient proof the body does not
    match what the manifest describes, is cheaper to compute, and is
    logically the more primitive fact (a hash mismatch is a strict
    consequence of a length mismatch in every practical case, never the
    other way around), so it is never worth also reporting a hash
    disagreement on top of it.
    """
    if manifest.content_length != len(body):
        return VerificationStatus.CONTENT_LENGTH_MISMATCH
    if manifest.content_hash != compute_content_hash(body):
        return VerificationStatus.HASH_MISMATCH
    return VerificationStatus.VERIFIED


@dataclass(frozen=True)
class ManifestReadResult:
    """What ``read_manifest`` actually found -- never just
    ``CaptureManifest | None`` (Phase 3D.4.1 requirement 2). ``error_reason``
    is always a short, static, factual description of WHAT went wrong
    (never a copy of file content, never a header, User-Agent, API key, or
    email address -- there is nothing in this module's own code that could
    even put one there, since no such value is ever read in the first
    place)."""

    status: ManifestReadStatus
    manifest: CaptureManifest | None
    error_reason: str | None


def read_manifest(capture_dir: Path, digest: str, *, body: bytes | None) -> ManifestReadResult:
    """Reads the manifest for ``digest`` in ``capture_dir`` and, when
    ``body`` is given (the actual bytes of the captured body currently on
    disk, or ``None`` if no body file was found), resolves the FINAL
    combined status against it in one call -- there is exactly one place
    a caller needs to look to learn whether a capture's
    ``capture_retrieved_at`` may be trusted.

    ``body=None`` means "no captured body file exists for this digest" --
    NOT "don't bother checking". A manifest that exists despite that is
    reported as ``BODY_MISSING`` (Phase 3D.4.1 requirement 7), never
    silently ignored and never conflated with ``MISSING`` (no manifest at
    all).
    """
    path = manifest_path_for(capture_dir, digest)
    if not path.is_file():
        return ManifestReadResult(status=ManifestReadStatus.MISSING, manifest=None, error_reason=None)
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return ManifestReadResult(
            status=ManifestReadStatus.IO_ERROR, manifest=None,
            error_reason=f"{type(exc).__name__} reading manifest file",
        )
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return ManifestReadResult(
            status=ManifestReadStatus.MALFORMED, manifest=None,
            error_reason=f"invalid JSON in manifest file: {exc.msg} (line {exc.lineno}, column {exc.colno})",
        )
    if not isinstance(payload, dict):
        return ManifestReadResult(
            status=ManifestReadStatus.MALFORMED, manifest=None,
            error_reason="manifest JSON is not an object",
        )
    try:
        manifest = CaptureManifest(**payload)
    except TypeError:
        return ManifestReadResult(
            status=ManifestReadStatus.UNSUPPORTED_SCHEMA, manifest=None,
            error_reason="manifest fields do not match a supported CaptureManifest schema",
        )
    if manifest.schema_version != CAPTURE_MANIFEST_SCHEMA_VERSION:
        return ManifestReadResult(
            status=ManifestReadStatus.UNSUPPORTED_SCHEMA, manifest=manifest,
            error_reason=f"unsupported manifest schema_version: {manifest.schema_version!r}",
        )

    if body is None:
        return ManifestReadResult(
            status=ManifestReadStatus.BODY_MISSING, manifest=manifest,
            error_reason="manifest present but no captured body file was found",
        )

    verification = verify_manifest_against_body(manifest, body)
    if verification is VerificationStatus.VERIFIED:
        return ManifestReadResult(status=ManifestReadStatus.VERIFIED, manifest=manifest, error_reason=None)
    if verification is VerificationStatus.CONTENT_LENGTH_MISMATCH:
        return ManifestReadResult(
            status=ManifestReadStatus.CONTENT_LENGTH_MISMATCH, manifest=manifest,
            error_reason="manifest content_length does not match the body actually on disk",
        )
    return ManifestReadResult(
        status=ManifestReadStatus.HASH_MISMATCH, manifest=manifest,
        error_reason="manifest content_hash does not match the body actually on disk",
    )
