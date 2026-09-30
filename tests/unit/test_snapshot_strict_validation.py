"""Phase 4.3F correction 2 Finding 5: the collect-snapshot decoder applies
STRICT, explicit type/format/enum/identifier validation to the three
structured-evidence collections (company_identity_evidence/
program_candidate_evidence/literature_candidate_evidence) -- any failure
anywhere in one of these collections corrupts the WHOLE snapshot
(``ResumeSnapshotCorrupted``), never a partial/best-effort acceptance.

Mirrors ``tests/integration/test_pipeline_resume.py::
test_corrupted_collect_snapshot_fails_closed``'s own low-level pattern: a
valid snapshot is saved first, then its stored JSON payload is mutated
directly to simulate a specific corruption, and
``collect_snapshot_for_resume`` is asserted to raise.
"""

from __future__ import annotations

import json

import pytest

from investment_research.storage.db import open_db
from investment_research.storage.repository import Repository, ResumeSnapshotCorrupted

RUN_ID = "snapfield-test"


def _repo_with_base_snapshot() -> Repository:
    repo = Repository(open_db(":memory:"))
    repo.upsert_company("SNAPFLD", "Snapshot Field Test Inc")
    repo.save_collect_snapshot(
        RUN_ID,
        facts=[],
        sources=[],
        collectors=[
            {
                "collector": "x", "outcome": "OK", "provenance": "LIVE", "errors": [],
                "attempted_urls": [], "notes": [], "zero_results": False,
                "raw_fact_count_before_dedup": 0,
            }
        ],
    )
    return repo


def _set_payload_key(repo: Repository, key: str, value: object) -> None:
    row = repo.conn.execute(
        "SELECT payload FROM run_checkpoints WHERE run_id = ? AND stage = '_collect_snapshot'",
        (RUN_ID,),
    ).fetchone()
    payload = json.loads(row["payload"])
    payload[key] = value
    repo.conn.execute(
        "UPDATE run_checkpoints SET payload = ? WHERE run_id = ? AND stage = '_collect_snapshot'",
        (json.dumps(payload), RUN_ID),
    )
    repo.conn.commit()


_VALID_COMPANY_IDENTITY = {
    "ticker": "SNAPFLD",
    "cik": 1234567,
    "sec_official_name": "Snapshot Field Test Inc",
    "explicitly_verified_aliases": [],
    "source_id": "src_x",
    "source_tier": "TIER_1",
    "retrieved_at": "2026-09-01T00:00:00+00:00",
    "content_hash": "h",
}

_VALID_PROGRAM_CANDIDATE = {
    "nct_id": "NCT01234567",
    "lead_sponsor": "Snapshot Field Test Inc",
    "collaborators": [],
    "interventions": [],
    "conditions": [],
    "overall_status": "RECRUITING",
    "phases": [],
    "primary_completion_date": "UNKNOWN",
    "completion_date": "UNKNOWN",
    "first_posted_date": "UNKNOWN",
    "source_id": "src_ct",
    "source_tier": "TIER_1",
    "retrieved_at": "2026-09-01T00:00:00+00:00",
    "content_hash": "h",
    "supporting_fact_ids": [],
}

_VALID_LITERATURE_CANDIDATE = {
    "pmid": "12345678",
    "nct_ids": ["NCT01234567"],
    "source_id": "src_lit",
    "source_tier": "UNKNOWN",
    "retrieved_at": "2026-09-01T00:00:00+00:00",
    "content_hash": "h",
}


@pytest.mark.parametrize(
    "collection_key,malformed_value",
    [
        ("company_identity_evidence", {"not": "a list"}),
        ("company_identity_evidence", "not-a-list-either"),
        ("company_identity_evidence", None),
        ("program_candidate_evidence", {"not": "a list"}),
        ("program_candidate_evidence", "not-a-list-either"),
        ("program_candidate_evidence", None),
        ("literature_candidate_evidence", {"not": "a list"}),
        ("literature_candidate_evidence", "not-a-list-either"),
        ("literature_candidate_evidence", None),
    ],
)
def test_collection_not_a_list_is_corrupted(collection_key, malformed_value):
    repo = _repo_with_base_snapshot()
    _set_payload_key(repo, collection_key, malformed_value)
    with pytest.raises(ResumeSnapshotCorrupted):
        repo.collect_snapshot_for_resume(RUN_ID)


@pytest.mark.parametrize(
    "collection_key,bad_record",
    [
        ("company_identity_evidence", "not-a-dict"),
        ("company_identity_evidence", 42),
        ("program_candidate_evidence", ["nested", "list"]),
        ("literature_candidate_evidence", None),
    ],
)
def test_record_not_a_dict_is_corrupted(collection_key, bad_record):
    repo = _repo_with_base_snapshot()
    _set_payload_key(repo, collection_key, [bad_record])
    with pytest.raises(ResumeSnapshotCorrupted):
        repo.collect_snapshot_for_resume(RUN_ID)


def test_missing_required_key_is_corrupted():
    repo = _repo_with_base_snapshot()
    record = dict(_VALID_COMPANY_IDENTITY)
    del record["sec_official_name"]
    _set_payload_key(repo, "company_identity_evidence", [record])
    with pytest.raises(ResumeSnapshotCorrupted):
        repo.collect_snapshot_for_resume(RUN_ID)


def test_scalar_field_wrong_type_is_corrupted():
    repo = _repo_with_base_snapshot()
    record = dict(_VALID_COMPANY_IDENTITY)
    record["sec_official_name"] = 12345  # must be a string
    _set_payload_key(repo, "company_identity_evidence", [record])
    with pytest.raises(ResumeSnapshotCorrupted):
        repo.collect_snapshot_for_resume(RUN_ID)


def test_cik_bool_is_corrupted():
    """bool is an int subclass in Python -- a naive isinstance(value, int)
    check alone would silently accept True/False as CIK 1/0."""
    repo = _repo_with_base_snapshot()
    record = dict(_VALID_COMPANY_IDENTITY)
    record["cik"] = True
    _set_payload_key(repo, "company_identity_evidence", [record])
    with pytest.raises(ResumeSnapshotCorrupted):
        repo.collect_snapshot_for_resume(RUN_ID)


def test_cik_none_is_accepted():
    """cik=None is explicitly VALID (an identity record can legitimately
    have no resolved CIK) -- only bool/non-int is rejected."""
    repo = _repo_with_base_snapshot()
    record = dict(_VALID_COMPANY_IDENTITY)
    record["cik"] = None
    _set_payload_key(repo, "company_identity_evidence", [record])
    snapshot = repo.collect_snapshot_for_resume(RUN_ID)
    assert snapshot.company_identity_evidence[0].cik is None


def test_list_field_given_a_string_is_corrupted_not_split_into_characters():
    """Phase 4.3F correction 2 Finding 5's own explicit requirement: a
    string in a list field must never be silently exploded into one
    character per element by a bare tuple(value)."""
    repo = _repo_with_base_snapshot()
    record = dict(_VALID_PROGRAM_CANDIDATE)
    record["collaborators"] = "Some University"  # should be a list
    _set_payload_key(repo, "program_candidate_evidence", [record])
    with pytest.raises(ResumeSnapshotCorrupted):
        repo.collect_snapshot_for_resume(RUN_ID)


def test_invalid_source_tier_enum_is_corrupted():
    repo = _repo_with_base_snapshot()
    record = dict(_VALID_COMPANY_IDENTITY)
    record["source_tier"] = "NOT_A_REAL_TIER"
    _set_payload_key(repo, "company_identity_evidence", [record])
    with pytest.raises(ResumeSnapshotCorrupted):
        repo.collect_snapshot_for_resume(RUN_ID)


def test_invalid_nct_id_is_corrupted():
    repo = _repo_with_base_snapshot()
    record = dict(_VALID_PROGRAM_CANDIDATE)
    record["nct_id"] = "NCT123"  # not 8 digits
    _set_payload_key(repo, "program_candidate_evidence", [record])
    with pytest.raises(ResumeSnapshotCorrupted):
        repo.collect_snapshot_for_resume(RUN_ID)


def test_invalid_pmid_is_corrupted():
    repo = _repo_with_base_snapshot()
    record = dict(_VALID_LITERATURE_CANDIDATE)
    record["pmid"] = "not-numeric"
    _set_payload_key(repo, "literature_candidate_evidence", [record])
    with pytest.raises(ResumeSnapshotCorrupted):
        repo.collect_snapshot_for_resume(RUN_ID)


def test_invalid_nct_id_inside_literature_nct_ids_is_corrupted():
    repo = _repo_with_base_snapshot()
    record = dict(_VALID_LITERATURE_CANDIDATE)
    record["nct_ids"] = ["NCT123"]
    _set_payload_key(repo, "literature_candidate_evidence", [record])
    with pytest.raises(ResumeSnapshotCorrupted):
        repo.collect_snapshot_for_resume(RUN_ID)


def test_invalid_fact_id_format_is_corrupted():
    repo = _repo_with_base_snapshot()
    record = dict(_VALID_PROGRAM_CANDIDATE)
    record["supporting_fact_ids"] = ["not-a-fact-id"]
    _set_payload_key(repo, "program_candidate_evidence", [record])
    with pytest.raises(ResumeSnapshotCorrupted):
        repo.collect_snapshot_for_resume(RUN_ID)


def test_mixed_list_one_bad_record_rejects_the_whole_snapshot():
    """Phase 4.3F correction 2 Finding 5's own explicit requirement: even
    one invalid record in an otherwise-valid list corrupts the WHOLE
    snapshot -- never silently dropped while the good ones are kept."""
    repo = _repo_with_base_snapshot()
    good = dict(_VALID_PROGRAM_CANDIDATE)
    bad = dict(_VALID_PROGRAM_CANDIDATE)
    bad["nct_id"] = "NOT-VALID"
    _set_payload_key(repo, "program_candidate_evidence", [good, bad])
    with pytest.raises(ResumeSnapshotCorrupted):
        repo.collect_snapshot_for_resume(RUN_ID)


def test_legacy_snapshot_missing_the_three_keys_entirely_is_not_corrupted():
    """Backward compatibility: a snapshot saved before Phase 4.3F (no
    company_identity_evidence/program_candidate_evidence/
    literature_candidate_evidence keys at all) is still valid, with empty
    lists -- never corrupted merely for lacking keys that did not exist
    yet."""
    repo = _repo_with_base_snapshot()
    row = repo.conn.execute(
        "SELECT payload FROM run_checkpoints WHERE run_id = ? AND stage = '_collect_snapshot'",
        (RUN_ID,),
    ).fetchone()
    payload = json.loads(row["payload"])
    for key in (
        "company_identity_evidence", "program_candidate_evidence", "literature_candidate_evidence",
    ):
        payload.pop(key, None)
    repo.conn.execute(
        "UPDATE run_checkpoints SET payload = ? WHERE run_id = ? AND stage = '_collect_snapshot'",
        (json.dumps(payload), RUN_ID),
    )
    repo.conn.commit()
    snapshot = repo.collect_snapshot_for_resume(RUN_ID)
    assert snapshot.company_identity_evidence == []
    assert snapshot.program_candidate_evidence == []
    assert snapshot.literature_candidate_evidence == []


def test_present_but_empty_list_is_valid():
    repo = _repo_with_base_snapshot()
    _set_payload_key(repo, "program_candidate_evidence", [])
    snapshot = repo.collect_snapshot_for_resume(RUN_ID)
    assert snapshot.program_candidate_evidence == []


def test_well_formed_records_still_round_trip_cleanly():
    """The strict validators must not reject genuinely well-formed
    records -- a positive control alongside all the negative cases
    above."""
    repo = _repo_with_base_snapshot()
    _set_payload_key(repo, "company_identity_evidence", [_VALID_COMPANY_IDENTITY])
    _set_payload_key(repo, "program_candidate_evidence", [_VALID_PROGRAM_CANDIDATE])
    _set_payload_key(repo, "literature_candidate_evidence", [_VALID_LITERATURE_CANDIDATE])
    snapshot = repo.collect_snapshot_for_resume(RUN_ID)
    assert len(snapshot.company_identity_evidence) == 1
    assert snapshot.company_identity_evidence[0].ticker == "SNAPFLD"
    assert len(snapshot.program_candidate_evidence) == 1
    assert snapshot.program_candidate_evidence[0].nct_id == "NCT01234567"
    assert len(snapshot.literature_candidate_evidence) == 1
    assert snapshot.literature_candidate_evidence[0].pmid == "12345678"
