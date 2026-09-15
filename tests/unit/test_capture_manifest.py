"""Phase 3D.4.1: capture_manifest.py exercised directly and exhaustively --
the ONE shared module both sec_live_smoke.py and clinicaltrials_live_smoke.py
consume, so proving each ManifestReadStatus's behavior here proves it for
both callers at once (requirement 10: no duplicated implementation).

No real network call anywhere in this file -- pure local filesystem
fixtures under ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from investment_research.research.capture_manifest import (
    CAPTURE_MANIFEST_SCHEMA_VERSION,
    CaptureManifest,
    ManifestReadStatus,
    VerificationStatus,
    compute_content_hash,
    is_evidence_integrity_failure,
    manifest_path_for,
    read_manifest,
    verify_manifest_against_body,
    write_manifest,
)

_DIGEST = "abc123def456abc123def456"


def _manifest(*, content: bytes, capture_retrieved_at: str = "2026-08-01T00:00:00Z") -> CaptureManifest:
    return CaptureManifest(
        schema_version=CAPTURE_MANIFEST_SCHEMA_VERSION,
        source="sec",
        requested_url="https://data.sec.gov/submissions/CIK0000320193.json",
        final_url="https://data.sec.gov/submissions/CIK0000320193.json",
        http_status=200,
        capture_retrieved_at=capture_retrieved_at,
        content_hash=compute_content_hash(content),
        content_length=len(content),
    )


# --- VERIFIED -----------------------------------------------------------
def test_read_manifest_verified_when_hash_and_length_agree(tmp_path):
    content = b'{"ok": true}'
    write_manifest(tmp_path, _DIGEST, _manifest(content=content))
    result = read_manifest(tmp_path, _DIGEST, body=content)
    assert result.status is ManifestReadStatus.VERIFIED
    assert result.manifest is not None
    assert result.manifest.capture_retrieved_at == "2026-08-01T00:00:00Z"
    assert result.error_reason is None
    assert not is_evidence_integrity_failure(result.status)


# --- MISSING --------------------------------------------------------------
def test_read_manifest_missing_when_no_manifest_file_exists(tmp_path):
    result = read_manifest(tmp_path, _DIGEST, body=b"whatever the body is")
    assert result.status is ManifestReadStatus.MISSING
    assert result.manifest is None
    assert result.error_reason is None
    # MISSING is explicitly NOT an Evidence Integrity failure -- an
    # unremarkable pre-Phase-3D.4 capture (requirement 5).
    assert not is_evidence_integrity_failure(result.status)


# --- MALFORMED --------------------------------------------------------------
def test_read_manifest_malformed_for_invalid_json(tmp_path):
    manifest_path_for(tmp_path, _DIGEST).write_text("{not valid json at all", encoding="utf-8")
    result = read_manifest(tmp_path, _DIGEST, body=b"whatever")
    assert result.status is ManifestReadStatus.MALFORMED
    assert result.manifest is None
    assert result.error_reason is not None
    assert is_evidence_integrity_failure(result.status)
    # MALFORMED must never be conflated with MISSING (requirement 6).
    assert result.status is not ManifestReadStatus.MISSING


def test_read_manifest_malformed_for_a_json_array_not_an_object(tmp_path):
    manifest_path_for(tmp_path, _DIGEST).write_text("[1, 2, 3]", encoding="utf-8")
    result = read_manifest(tmp_path, _DIGEST, body=b"whatever")
    assert result.status is ManifestReadStatus.MALFORMED


# --- UNSUPPORTED_SCHEMA ------------------------------------------------------
def test_read_manifest_unsupported_schema_for_unrecognized_fields(tmp_path):
    manifest_path_for(tmp_path, _DIGEST).write_text(
        json.dumps({"totally": "different", "shape": True}), encoding="utf-8",
    )
    result = read_manifest(tmp_path, _DIGEST, body=b"whatever")
    assert result.status is ManifestReadStatus.UNSUPPORTED_SCHEMA
    assert result.manifest is None
    assert is_evidence_integrity_failure(result.status)


def test_read_manifest_unsupported_schema_for_a_future_schema_version(tmp_path):
    content = b"the body"
    manifest = _manifest(content=content)
    payload = {
        "schema_version": CAPTURE_MANIFEST_SCHEMA_VERSION + 999,
        "source": manifest.source,
        "requested_url": manifest.requested_url,
        "final_url": manifest.final_url,
        "http_status": manifest.http_status,
        "capture_retrieved_at": manifest.capture_retrieved_at,
        "content_hash": manifest.content_hash,
        "content_length": manifest.content_length,
    }
    manifest_path_for(tmp_path, _DIGEST).write_text(json.dumps(payload), encoding="utf-8")
    result = read_manifest(tmp_path, _DIGEST, body=content)
    assert result.status is ManifestReadStatus.UNSUPPORTED_SCHEMA
    # The manifest object itself is still returned (schema recognized
    # enough to parse, just an unsupported version) -- but its status,
    # never VERIFIED, is what a caller must act on.
    assert result.manifest is not None
    assert is_evidence_integrity_failure(result.status)


# --- HASH_MISMATCH / CONTENT_LENGTH_MISMATCH -------------------------------
def test_verify_manifest_against_body_hash_mismatch_same_length():
    actual = b"0123456789"
    corrupted = b"0123456789"[:-1] + b"X"
    manifest = _manifest(content=actual)
    assert len(corrupted) == len(actual)
    assert verify_manifest_against_body(manifest, corrupted) is VerificationStatus.HASH_MISMATCH


def test_verify_manifest_against_body_content_length_mismatch():
    actual = b"0123456789"
    shorter = b"01234"
    manifest = _manifest(content=actual)
    assert verify_manifest_against_body(manifest, shorter) is VerificationStatus.CONTENT_LENGTH_MISMATCH


def test_verify_manifest_against_body_content_length_priority_when_both_disagree():
    """When length AND hash both disagree, content_length is reported
    (Phase 3D.4.1 requirement 3's explicit priority)."""
    actual = b"0123456789"
    totally_different = b"completely different body, different length too"
    manifest = _manifest(content=actual)
    assert verify_manifest_against_body(manifest, totally_different) is VerificationStatus.CONTENT_LENGTH_MISMATCH


def test_verify_manifest_against_body_verified_when_bytes_match():
    content = b"exact match"
    manifest = _manifest(content=content)
    assert verify_manifest_against_body(manifest, content) is VerificationStatus.VERIFIED


def test_read_manifest_hash_mismatch_via_read_manifest(tmp_path):
    actual = b"0123456789"
    corrupted = actual[:-1] + b"X"
    write_manifest(tmp_path, _DIGEST, _manifest(content=corrupted))
    result = read_manifest(tmp_path, _DIGEST, body=actual)
    assert result.status is ManifestReadStatus.HASH_MISMATCH
    assert is_evidence_integrity_failure(result.status)
    assert result.manifest is not None  # the manifest is still returned...
    # ...but never used as if it were verified.


def test_read_manifest_content_length_mismatch_via_read_manifest(tmp_path):
    write_manifest(tmp_path, _DIGEST, _manifest(content=b"a much longer original body"))
    result = read_manifest(tmp_path, _DIGEST, body=b"short")
    assert result.status is ManifestReadStatus.CONTENT_LENGTH_MISMATCH
    assert is_evidence_integrity_failure(result.status)


# --- BODY_MISSING -------------------------------------------------------
def test_read_manifest_body_missing_when_manifest_exists_but_body_is_none(tmp_path):
    """A manifest that claims a capture was made, for an artifact that
    cannot be found to re-verify, is an Evidence Integrity failure -- NOT
    a harmless legacy case like MISSING (Phase 3D.4.1.1 correction)."""
    write_manifest(tmp_path, _DIGEST, _manifest(content=b"whatever was captured"))
    result = read_manifest(tmp_path, _DIGEST, body=None)
    assert result.status is ManifestReadStatus.BODY_MISSING
    assert result.manifest is not None
    assert result.error_reason is not None
    assert is_evidence_integrity_failure(result.status)
    assert result.status is not ManifestReadStatus.VERIFIED
    # BODY_MISSING is distinct from MISSING (no manifest at all) --
    # different status, different classification (MISSING is not a
    # failure; BODY_MISSING is).
    assert result.status is not ManifestReadStatus.MISSING
    assert not is_evidence_integrity_failure(ManifestReadStatus.MISSING)


# --- IO_ERROR -------------------------------------------------------------
def test_read_manifest_io_error_when_the_file_cannot_be_read(tmp_path, monkeypatch):
    path = manifest_path_for(tmp_path, _DIGEST)
    path.write_text("{}", encoding="utf-8")

    original_read_text = Path.read_text

    def boom(self, *args, **kwargs):
        if self == path:
            raise OSError("simulated IO error")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", boom)

    result = read_manifest(tmp_path, _DIGEST, body=b"whatever")
    assert result.status is ManifestReadStatus.IO_ERROR
    assert result.manifest is None
    assert is_evidence_integrity_failure(result.status)


# --- error_reason never contains secrets/headers/body content -------------
@pytest.mark.parametrize(
    "forbidden",
    ["user_agent", "User-Agent", "api_key", "Authorization", "email", "@example.com", "secret-contact"],
)
def test_error_reason_never_contains_secret_shaped_values(tmp_path, forbidden):
    manifest_path_for(tmp_path, _DIGEST).write_text("{not valid json", encoding="utf-8")
    result = read_manifest(tmp_path, _DIGEST, body=b"whatever")
    assert result.error_reason is not None
    assert forbidden not in result.error_reason


def test_manifest_dataclass_has_no_secret_shaped_fields():
    field_names = set(CaptureManifest.__dataclass_fields__)
    assert field_names.isdisjoint({"user_agent", "headers", "api_key", "email", "authorization"})


# --- status is a plain string in JSON output --------------------------------
def test_manifest_read_status_serializes_as_a_plain_string():
    import json as json_module

    assert json_module.dumps(ManifestReadStatus.VERIFIED.value) == '"VERIFIED"'
    for status in ManifestReadStatus:
        assert isinstance(status.value, str)
        # Round-trips cleanly through json.dumps/json.loads as a string.
        assert json_module.loads(json_module.dumps(status.value)) == status.value


# --- evidence integrity classification is exhaustive and non-overlapping ---
def test_every_status_is_classified_exactly_once():
    non_failures = {ManifestReadStatus.VERIFIED, ManifestReadStatus.MISSING}
    for status in ManifestReadStatus:
        if status in non_failures:
            assert not is_evidence_integrity_failure(status)
        else:
            assert is_evidence_integrity_failure(status)
