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
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
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


def read_manifest(capture_dir: Path, digest: str) -> CaptureManifest | None:
    """``None`` for a missing file, unreadable file, malformed JSON, or a
    JSON object whose keys don't match this schema -- never a guess, and
    never an exception a caller must remember to catch (Phase 3D.4
    requirement 11: an existing capture directory with no manifest at all
    is exactly this case, and must degrade cleanly, not crash)."""
    path = manifest_path_for(capture_dir, digest)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return CaptureManifest(**payload)
    except TypeError:
        return None


def verify_manifest_against_body(manifest: CaptureManifest, body: bytes) -> bool:
    """Whether the manifest's own recorded ``content_hash`` matches the
    ACTUAL bytes on disk right now. A caller must never treat a mismatch as
    a normal, verified capture (Phase 3D.4 requirement 9) -- the body file
    could have been edited, truncated, or swapped after the manifest was
    written, and a manifest that no longer describes its own body is not
    trustworthy evidence of anything, including its own
    ``capture_retrieved_at``."""
    return manifest.content_hash == compute_content_hash(body)
