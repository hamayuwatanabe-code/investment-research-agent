"""Phase 3D: clinicaltrials_live_smoke.py exercised entirely offline.

No real network call anywhere in this file. The orchestration tests inject
a ``FakeHttpClient`` (reusing ``_clinicaltrials_fixture_support``'s
real-API-v2-format fixtures) in place of ``AllowlistedHttpClient``.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from investment_research.research import clinicaltrials_live_smoke as live
from investment_research.research.capture_manifest import (
    CAPTURE_MANIFEST_SCHEMA_VERSION,
    CaptureManifest,
    ManifestReadStatus,
    compute_content_hash,
    is_evidence_integrity_failure,
    write_manifest,
)
from investment_research.schemas.enums import UNKNOWN

from . import _clinicaltrials_fixture_support as fx
from ._clinicaltrials_fixture_support import FakeHttpClient
from ._network_guard import forbid_external_network_autouse  # noqa: F401

_REAL_LOOKING_AGENT = "clinicaltrials-live-smoke-test-agent/9.9"


class _PoisonHttpClient:
    def get(self, url, **kwargs):
        raise AssertionError(f"must never be called -- attempted GET {url}")


# --- User-Agent: optional, never a refusal gate, never logged --------------
def test_resolve_user_agent_returns_default_when_unset():
    assert live.resolve_user_agent(env={}) == live.DEFAULT_CT_USER_AGENT


def test_resolve_user_agent_returns_configured_value_when_set():
    assert live.resolve_user_agent(env={live.CT_USER_AGENT_ENV_VAR: _REAL_LOOKING_AGENT}) == _REAL_LOOKING_AGENT


# --- NCT ID validation: zero network on malformed input ---------------------
def test_run_live_smoke_refuses_malformed_nct_id_with_zero_requests():
    report = live.run_live_smoke("not-an-nct-id", http_client=_PoisonHttpClient())
    assert report.refused
    assert report.request_count == 0
    assert report.requested_urls == []


def test_main_refuses_without_nct_id():
    assert live.main([]) == 1


# --- NCT ID normalization: only whitespace/case, still strictly validated --
def test_normalize_nct_id_trims_and_uppercases():
    assert live.normalize_nct_id(f"  {fx.NCT_ID.lower()}  \n") == fx.NCT_ID


def test_run_live_smoke_accepts_lowercase_and_whitespace_after_normalization():
    http = FakeHttpClient(responses=fx.default_responses())
    report = live.run_live_smoke(f"  {fx.NCT_ID.lower()}\n", http_client=http)
    assert not report.refused
    assert report.plan.nct_id == fx.NCT_ID  # displayed plan shows the NORMALIZED value


def test_normalization_never_widens_what_counts_as_valid():
    """Normalizing a genuinely malformed id must still refuse -- trimming/
    uppercasing never turns garbage into a valid-looking NCT ID."""
    report = live.run_live_smoke("  not-an-nct-id  ", http_client=_PoisonHttpClient())
    assert report.refused
    assert report.request_count == 0


def test_analyze_capture_normalizes_nct_id_too(tmp_path):
    import hashlib

    url = fx.study_url(fx.NCT_ID)
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    (tmp_path / f"{digest}.json").write_text(fx.fixture_text("study_recruiting_interventional.json"), encoding="utf-8")
    result = live.analyze_capture(tmp_path, nct_id=f"  {fx.NCT_ID.lower()}  ")
    assert result["found"] is True


def test_analyze_capture_refuses_malformed_nct_id_never_guesses_a_file():
    result = live.analyze_capture(Path("/tmp"), nct_id="not-an-nct-id")
    assert result["found"] is False
    assert "error" in result


# --- full offline orchestration (FakeHttpClient, real-format fixtures) -----
def test_full_orchestration_against_fake_transport():
    http = FakeHttpClient(responses=fx.default_responses())
    report = live.run_live_smoke(fx.NCT_ID, http_client=http)

    assert not report.refused
    assert report.errors == []
    assert report.request_count == 1
    assert report.document_id is not None
    assert report.parsed_fields.get("status") == "RECRUITING"
    assert report.document_diagnostics["authority"] == "REGISTRY"
    # ClinicalTrials.gov is a government-operated REGISTRY, not a
    # company IR channel.
    assert report.document_diagnostics["is_company_ir"] is False
    # zero external LLM/AI activity, always
    assert report.external_llm_tokens == 0
    assert report.anthropic_api_calls == 0
    assert report.web_search_calls == 0
    assert report.execution_report is not None
    assert report.execution_report.diagnostics.external_llm_tokens == 0
    assert report.live_verified_candidates  # structurally completed this run (diagnostic only)


def test_404_is_reported_never_acquired():
    http = FakeHttpClient(responses={fx.study_url(fx.NCT_ID_404): fx.not_found()})
    report = live.run_live_smoke(fx.NCT_ID_404, http_client=http)
    assert not report.refused
    assert report.errors
    assert report.document_id is None
    assert report.live_verified_candidates == []


def test_malformed_json_is_reported_never_acquired():
    http = FakeHttpClient(responses={fx.study_url(fx.NCT_ID_MALFORMED): fx.ok(fx.fixture_text("study_malformed.txt"))})
    report = live.run_live_smoke(fx.NCT_ID_MALFORMED, http_client=http)
    assert not report.refused
    assert report.errors
    assert report.document_id is None


def test_fixture_diff_reports_structural_comparison():
    http = FakeHttpClient(responses=fx.default_responses())
    report = live.run_live_smoke(fx.NCT_ID, http_client=http)
    assert "keys_in_both" in report.fixture_diff
    assert report.fixture_diff["keys_in_both"] > 0


def test_no_user_agent_value_ever_appears_in_printed_report():
    http = FakeHttpClient(responses=fx.default_responses())
    report = live.run_live_smoke(fx.NCT_ID, user_agent=_REAL_LOOKING_AGENT, http_client=http)
    printed = live.format_report_for_print(report, secrets=[_REAL_LOOKING_AGENT])
    assert _REAL_LOOKING_AGENT not in printed


def test_printed_report_documents_independent_confirmation_is_a_fact_level_field():
    http = FakeHttpClient(responses=fx.default_responses())
    report = live.run_live_smoke(fx.NCT_ID, http_client=http)
    printed = live.format_report_for_print(report, secrets=[])
    assert "independent_confirmation" in printed
    assert "company_claim" in printed


# --- one-time marker guard (main()) -----------------------------------------
def test_main_marker_guard_blocks_a_second_invocation(tmp_path):
    marker = tmp_path / "marker.json"
    marker.write_text(json.dumps({"attempted_at": "2020-01-01T00:00:00Z", "refused": False}))
    exit_code = live.main(["--nct-id", fx.NCT_ID, "--marker-path", str(marker)])
    assert exit_code == 2


def test_main_never_writes_marker_on_malformed_nct_id(tmp_path):
    marker = tmp_path / "marker.json"
    exit_code = live.main(["--nct-id", "garbage", "--marker-path", str(marker)])
    assert exit_code == 1
    assert not marker.exists()


# --- analyze_capture: offline re-analysis of already-saved bodies -----------
def test_analyze_capture_reads_saved_body_offline(tmp_path):
    import hashlib

    url = fx.study_url(fx.NCT_ID)
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    (tmp_path / f"{digest}.json").write_text(fx.fixture_text("study_recruiting_interventional.json"), encoding="utf-8")

    result = live.analyze_capture(tmp_path, nct_id=fx.NCT_ID)
    assert result["found"] is True
    assert result["schema_ok"] is True
    assert result["parsed_fields"]["status"] == "RECRUITING"
    assert result["fixture_diff"]["keys_in_both"] > 0


def test_analyze_capture_reports_not_found_never_guesses(tmp_path):
    result = live.analyze_capture(tmp_path, nct_id=fx.NCT_ID)
    assert result["found"] is False


def test_analyze_capture_never_touches_the_one_time_marker(tmp_path):
    marker = tmp_path / "marker.json"
    live.analyze_capture(tmp_path, nct_id=fx.NCT_ID)
    assert not marker.exists()


def test_main_analyze_capture_mode_requires_no_marker_or_network(tmp_path):
    exit_code = live.main(["--nct-id", fx.NCT_ID, "--analyze-capture", str(tmp_path)])
    assert exit_code == 1  # nothing captured in an empty tmp_path -- reported, not a crash


# --- transport wiring: allowlisted to clinicaltrials.gov only --------------
def test_allowed_hosts_is_scoped_to_clinicaltrials_gov_only():
    assert frozenset({"clinicaltrials.gov"}) == live.ALLOWED_HOSTS


def test_build_plan_shows_max_two_requests_never_makes_one():
    plan = live.build_plan(fx.NCT_ID)
    assert plan.max_requests == live.MAX_LIVE_GETS
    assert live.MAX_LIVE_GETS <= 2
    printed = live.format_plan_for_print(plan)
    assert "nothing sent yet" in printed


def test_no_nct_id_is_hardcoded_as_a_default_anywhere():
    """An NCT ID may appear in --help example text (documentation), but
    never as a default value/constant this module would use if the caller
    didn't supply one -- run_live_smoke/build_plan/main all require an
    explicit nct_id with no fallback."""
    import inspect

    assert "nct_id" in inspect.signature(live.run_live_smoke).parameters
    assert inspect.signature(live.run_live_smoke).parameters["nct_id"].default is inspect.Parameter.empty
    assert inspect.signature(live.build_plan).parameters["nct_id"].default is inspect.Parameter.empty
    assert live.main([]) == 1  # no --nct-id supplied -> refuses, never guesses one


# --- Phase 3D.2: registry freshness diagnostics -----------------------------
def test_stale_after_days_threshold_matches_config_settings():
    """One staleness threshold for the whole system -- never a second,
    independently-chosen number living only in this module."""
    from investment_research.config import Settings

    assert Settings.__dataclass_fields__["stale_after_days"].default == live.REGISTRY_STALE_AFTER_DAYS


def test_as_of_date_is_never_confused_with_a_study_date():
    parsed = {
        "last_update_post": "2026-02-15", "last_update_submit": "2026-02-10",
        "primary_completion": "2026-06-30", "primary_completion_type": "ESTIMATED",
        "completion_date": "2026-12-31", "completion_date_type": "ESTIMATED",
    }
    freshness = live._build_freshness_diagnostics(parsed, "2026-09-14T00:00:00+00:00")
    assert freshness.as_of_date == "2026-09-14T00:00:00+00:00"
    assert freshness.last_update_post == "2026-02-15"
    assert freshness.primary_completion == "2026-06-30"
    assert freshness.completion_date == "2026-12-31"
    # All four kept as four distinct values -- never collapsed into one.
    assert len({freshness.as_of_date[:10], freshness.last_update_post,
                freshness.primary_completion, freshness.completion_date}) == 4


def test_capture_retrieved_at_defaults_to_unknown_never_fabricated_from_as_of():
    """_build_freshness_diagnostics's caller decides capture_retrieved_at
    explicitly; when it doesn't (the default), it is UNKNOWN -- never
    silently filled in from as_of_date (Phase 3D.3 requirement 2)."""
    parsed = {"last_update_post": "2026-02-15"}
    freshness = live._build_freshness_diagnostics(parsed, "2026-09-14T00:00:00+00:00")
    assert freshness.capture_retrieved_at == UNKNOWN
    assert freshness.capture_retrieved_at != freshness.as_of_date


def test_estimated_date_in_the_past_is_flagged_factually_never_interpreted():
    parsed = {
        "last_update_post": "2026-02-15", "last_update_submit": "2026-02-10",
        "primary_completion": "2026-06-30", "primary_completion_type": "ESTIMATED",
        "completion_date": "2027-01-01", "completion_date_type": "ESTIMATED",
    }
    freshness = live._build_freshness_diagnostics(parsed, "2026-09-14T00:00:00+00:00")
    assert freshness.estimated_primary_completion_date_passed is True  # 2026-06-30 < 2026-09-14
    assert freshness.estimated_completion_date_passed is False  # 2027-01-01 is still in the future
    # The dataclass carries no field named anything like "delayed"/"failed"/"completed".
    field_names = {f.name for f in fields(freshness)}
    assert field_names.isdisjoint({"delayed", "failed", "trial_completed", "success"})


def test_actual_typed_date_is_never_flagged_as_passed():
    """An ACTUAL-typed date having "passed" is not an informative fact (it
    already happened by definition) -- only ESTIMATED dates are flagged."""
    parsed = {
        "last_update_post": "2026-02-15",
        "primary_completion": "2020-01-01", "primary_completion_type": "ACTUAL",
        "completion_date": UNKNOWN, "completion_date_type": UNKNOWN,
    }
    freshness = live._build_freshness_diagnostics(parsed, "2026-09-14T00:00:00+00:00")
    assert freshness.estimated_primary_completion_date_passed is None
    assert freshness.estimated_completion_date_passed is None


def test_partial_dates_never_padded_before_comparison():
    parsed = {
        "last_update_post": "2026-02-15",
        "primary_completion": "2026", "primary_completion_type": "ESTIMATED",
        "completion_date": "2026-03", "completion_date_type": "ESTIMATED",
    }
    freshness = live._build_freshness_diagnostics(parsed, "2026-09-14T00:00:00+00:00")
    # "2026" compared only at year precision against "2026" (today's year) -- equal, not "passed".
    assert freshness.estimated_primary_completion_date_passed is False
    # "2026-03" compared at month precision against "2026-09" -- March < September -> passed.
    assert freshness.estimated_completion_date_passed is True


def test_missing_dates_never_guessed():
    parsed = {"last_update_post": UNKNOWN, "primary_completion": UNKNOWN,
              "primary_completion_type": UNKNOWN, "completion_date": UNKNOWN,
              "completion_date_type": UNKNOWN}
    freshness = live._build_freshness_diagnostics(parsed, "2026-09-14T00:00:00+00:00")
    assert freshness.days_since_last_update_post is None
    assert freshness.stale_registry_record is None
    assert freshness.estimated_primary_completion_date_passed is None
    assert freshness.estimated_completion_date_passed is None


def test_stale_registry_record_flag_is_purely_a_day_count():
    fresh = live._build_freshness_diagnostics(
        {"last_update_post": "2026-08-01"}, "2026-09-14T00:00:00+00:00",
    )
    assert fresh.days_since_last_update_post == 44
    assert fresh.stale_registry_record is False

    old = live._build_freshness_diagnostics(
        {"last_update_post": "2024-01-01"}, "2026-09-14T00:00:00+00:00",
    )
    assert old.days_since_last_update_post is not None
    assert old.days_since_last_update_post > live.REGISTRY_STALE_AFTER_DAYS
    assert old.stale_registry_record is True


def test_full_orchestration_populates_freshness_from_the_real_document():
    http = FakeHttpClient(responses=fx.default_responses())
    report = live.run_live_smoke(fx.NCT_ID, http_client=http)
    assert report.freshness is not None
    # A genuine live run: both the as-of point AND the recorded capture
    # time are the real Document.retrieved_at -- never fabricated, and
    # equal to each other only because this IS the moment of capture.
    assert report.freshness.as_of_date == report.document_diagnostics["retrieved_at"]
    assert report.freshness.capture_retrieved_at == report.document_diagnostics["retrieved_at"]
    assert report.freshness.last_update_post == report.parsed_fields["last_update_post"]


def test_analyze_capture_freshness_omitted_without_as_of(tmp_path):
    import hashlib

    url = fx.study_url(fx.NCT_ID)
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    (tmp_path / f"{digest}.json").write_text(fx.fixture_text("study_recruiting_interventional.json"), encoding="utf-8")
    result = live.analyze_capture(tmp_path, nct_id=fx.NCT_ID)
    assert result["freshness"] is None  # never guessed from file mtime or today's real date


def test_analyze_capture_freshness_populated_with_explicit_as_of(tmp_path):
    import hashlib

    url = fx.study_url(fx.NCT_ID)
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    (tmp_path / f"{digest}.json").write_text(fx.fixture_text("study_recruiting_interventional.json"), encoding="utf-8")
    result = live.analyze_capture(tmp_path, nct_id=fx.NCT_ID, as_of="2026-09-14T00:00:00+00:00")
    assert result["freshness"] is not None
    assert result["freshness"].as_of_date == "2026-09-14T00:00:00+00:00"
    # --as-of is a re-analysis reference date, never a fabricated capture
    # time -- analyze_capture() has no recorded capture metadata to draw
    # one from (Phase 3D.3 requirement 2).
    assert result["freshness"].capture_retrieved_at == UNKNOWN


def test_analyze_capture_freshness_capture_retrieved_at_never_changes_with_as_of(tmp_path):
    """--as-of only ever moves as_of_date; capture_retrieved_at must stay
    UNKNOWN regardless of what --as-of is passed (Phase 3D.3 requirement 3)."""
    import hashlib

    url = fx.study_url(fx.NCT_ID)
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    (tmp_path / f"{digest}.json").write_text(fx.fixture_text("study_recruiting_interventional.json"), encoding="utf-8")

    first = live.analyze_capture(tmp_path, nct_id=fx.NCT_ID, as_of="2026-01-01T00:00:00+00:00")
    second = live.analyze_capture(tmp_path, nct_id=fx.NCT_ID, as_of="2026-09-14T00:00:00+00:00")

    assert first["freshness"].as_of_date != second["freshness"].as_of_date
    assert first["freshness"].capture_retrieved_at == UNKNOWN
    assert second["freshness"].capture_retrieved_at == UNKNOWN


def test_freshness_to_jsonable_is_a_plain_dict_with_real_types():
    freshness = live._build_freshness_diagnostics(
        {"last_update_post": "2026-02-15"}, "2026-09-14T00:00:00+00:00",
    )
    jsonable = live.freshness_to_jsonable(freshness)
    assert isinstance(jsonable, dict)
    assert jsonable["as_of_date"] == "2026-09-14T00:00:00+00:00"
    assert jsonable["capture_retrieved_at"] == UNKNOWN
    assert jsonable["days_since_last_update_post"] == 211
    assert jsonable["stale_registry_record"] is False
    # Never stringified -- real Python/JSON bool and int, not "True"/"211".
    assert isinstance(jsonable["stale_registry_record"], bool)
    assert isinstance(jsonable["days_since_last_update_post"], int)
    assert not isinstance(jsonable["days_since_last_update_post"], bool)


def test_freshness_to_jsonable_none_stays_none():
    assert live.freshness_to_jsonable(None) is None


def test_analyze_capture_freshness_serializes_as_a_json_object_not_a_repr_string(tmp_path, capsys):
    """End-to-end: the printed --analyze-capture JSON output must round-trip
    with freshness as a nested object whose bools/ints keep their real JSON
    types -- reproduces the exact bug the real Mac re-analysis run
    surfaced: "freshness": "RegistryFreshnessDiagnostics(...)" as one
    opaque string (Phase 3D.3 requirement 1)."""
    import hashlib
    import json as json_module

    url = fx.study_url(fx.NCT_ID)
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    (tmp_path / f"{digest}.json").write_text(fx.fixture_text("study_recruiting_interventional.json"), encoding="utf-8")

    exit_code = live.main([
        "--nct-id", fx.NCT_ID, "--analyze-capture", str(tmp_path), "--as-of", "2026-09-14T00:00:00+00:00",
    ])
    assert exit_code == 0
    printed = capsys.readouterr().out
    parsed_output = json_module.loads(printed)

    freshness_obj = parsed_output["freshness"]
    assert isinstance(freshness_obj, dict)
    assert freshness_obj["as_of_date"] == "2026-09-14T00:00:00+00:00"
    assert freshness_obj["capture_retrieved_at"] == UNKNOWN
    assert isinstance(freshness_obj["stale_registry_record"], bool)
    assert isinstance(freshness_obj["days_since_last_update_post"], int)
    # The dataclass repr must never leak through as an opaque string.
    assert "RegistryFreshnessDiagnostics" not in printed


# --- Phase 3D.2 item 4: a report-shaped capture re-analyzes fully offline ---
#
# study_recruiting_interventional_v2.json is structurally the same shape a
# real ACTIVE_NOT_RECRUITING / ACTUAL-enrollment / has_results=False study
# report takes (see MANIFEST.md) -- it is a synthetic, neutral fixture, not
# any real issuer's data. analyze_capture() itself never accepts an
# http_client parameter at all, so there is no live-transport code path it
# could reach even in error; this test locks that structural guarantee down
# against a capture shaped like a real Live Smoke run's own output.
def test_analyze_capture_reanalyzes_a_report_shaped_capture_fully_offline(tmp_path):
    import hashlib
    import inspect

    assert "http_client" not in inspect.signature(live.analyze_capture).parameters
    url = fx.study_url(fx.NCT_ID)
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    (tmp_path / f"{digest}.json").write_text(
        fx.fixture_text("study_recruiting_interventional_v2.json"), encoding="utf-8",
    )

    result = live.analyze_capture(tmp_path, nct_id=fx.NCT_ID, as_of="2026-09-14T00:00:00+00:00")

    assert result["found"] is True
    assert result["schema_ok"] is True
    assert result["parsed_fields"]["status"] == "ACTIVE_NOT_RECRUITING"
    assert result["parsed_fields"]["enrollment"] == 150
    assert result["parsed_fields"]["enrollment_type"] == "ACTUAL"
    assert result["parsed_fields"]["has_results"] is False
    assert result["freshness"] is not None
    # A later lastUpdatePost than the offline as_of would be a fixture-data
    # bug, not something this test should silently tolerate.
    assert result["freshness"].days_since_last_update_post is not None
    assert result["freshness"].days_since_last_update_post >= 0


# --- Phase 3D.2 item 5: LIVE_VERIFIED is scoped per target, never global ---
def test_live_verified_candidates_are_diagnostic_only_never_a_code_change():
    """Mirrors sec_live_smoke's own regression: running this smoke test must
    never touch source_routing_catalog.py or promote any step's
    ImplementationStatus in code -- confirmed by the fact that the real
    catalog's build function is never even imported by this module."""
    import investment_research.research.clinicaltrials_live_smoke as module

    assert "source_routing_catalog" not in module.__dict__
    assert not hasattr(module, "build_source_routing_graph")


def test_live_verified_candidates_scoped_to_this_one_targets_own_steps():
    """A single-NCT-ID smoke run's live_verified_candidates must never imply
    success of the broader ClinicalTrials.gov schema or of any other
    Evidence Requirement -- only the LOCATE/FETCH/PARSE steps of THIS one
    target, for THIS one study, actually ran."""
    http = FakeHttpClient(responses=fx.default_responses())
    report = live.run_live_smoke(fx.NCT_ID, http_client=http)

    assert report.live_verified_candidates
    for candidate in report.live_verified_candidates:
        target_id, _, step_id = candidate.partition(":")
        assert target_id == "target_ct_smoke"
        assert step_id in {"l1", "f", "p"}
    # Confirms this diagnostic never widens beyond the one target this run
    # actually executed.
    assert len({c.partition(":")[0] for c in report.live_verified_candidates}) == 1


# --- Phase 3D.4: analyze_capture() consumes the shared Capture Manifest ----
def _write_ct_manifest(tmp_path: Path, *, capture_retrieved_at: str, content: bytes) -> str:
    """Writes a manifest matching a study_recruiting_interventional.json
    capture already saved at tmp_path -- returns the digest used, purely a
    test helper, never a production code path."""
    import hashlib

    url = fx.study_url(fx.NCT_ID)
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    manifest = CaptureManifest(
        schema_version=CAPTURE_MANIFEST_SCHEMA_VERSION,
        source="clinicaltrials",
        requested_url=url,
        final_url=url,
        http_status=200,
        capture_retrieved_at=capture_retrieved_at,
        content_hash=compute_content_hash(content),
        content_length=len(content),
    )
    write_manifest(tmp_path, digest, manifest)
    return digest


def test_analyze_capture_uses_manifest_capture_retrieved_at_when_hash_verified(tmp_path):
    import hashlib

    url = fx.study_url(fx.NCT_ID)
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    content = fx.fixture_text("study_recruiting_interventional.json").encode("utf-8")
    (tmp_path / f"{digest}.json").write_bytes(content)
    _write_ct_manifest(tmp_path, capture_retrieved_at="2026-08-01T00:00:00Z", content=content)

    result = live.analyze_capture(tmp_path, nct_id=fx.NCT_ID, as_of="2026-09-14T00:00:00+00:00")
    assert result["capture_manifest_status"] == "VERIFIED"
    assert result["capture_manifest_schema_version"] == CAPTURE_MANIFEST_SCHEMA_VERSION
    assert result["capture_manifest_error"] is None
    assert result["capture_retrieved_at"] == "2026-08-01T00:00:00Z"
    assert result["freshness"] is not None
    # capture_retrieved_at comes from the manifest, never from --as-of.
    assert result["freshness"].capture_retrieved_at == "2026-08-01T00:00:00Z"
    assert result["freshness"].as_of_date == "2026-09-14T00:00:00+00:00"
    assert result["freshness"].capture_retrieved_at != result["freshness"].as_of_date


def test_analyze_capture_manifest_hash_mismatch_never_treated_as_verified(tmp_path):
    """The manifest's content_hash was computed against DIFFERENT bytes of
    the SAME length as what's on disk now (a same-size corruption, so
    content_length alone cannot catch it and the hash comparison is what
    matters) -- must never be trusted as a normal, verified capture (Phase
    3D.4 requirement 9): capture_retrieved_at stays UNKNOWN even though a
    manifest is present."""
    import hashlib

    url = fx.study_url(fx.NCT_ID)
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    actual_content = fx.fixture_text("study_recruiting_interventional.json").encode("utf-8")
    (tmp_path / f"{digest}.json").write_bytes(actual_content)
    # Manifest describes DIFFERENT bytes of the SAME length as what is
    # actually on disk -- isolates the hash check from the length check.
    corrupted_same_length = actual_content[:-1] + (b"X" if actual_content[-1:] != b"X" else b"Y")
    _write_ct_manifest(tmp_path, capture_retrieved_at="2026-08-01T00:00:00Z", content=corrupted_same_length)

    result = live.analyze_capture(tmp_path, nct_id=fx.NCT_ID, as_of="2026-09-14T00:00:00+00:00")
    assert result["capture_manifest_status"] == "HASH_MISMATCH"
    assert result["capture_manifest_error"] is not None
    assert result["capture_retrieved_at"] == UNKNOWN
    assert result["freshness"] is not None
    assert result["freshness"].capture_retrieved_at == UNKNOWN


def test_analyze_capture_manifest_content_length_mismatch_takes_priority(tmp_path):
    """When BOTH content_length and content_hash disagree, content_length
    is reported (Phase 3D.4.1 requirement 3's explicit priority) -- never
    treated as verified either way."""
    import hashlib

    url = fx.study_url(fx.NCT_ID)
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    actual_content = fx.fixture_text("study_recruiting_interventional.json").encode("utf-8")
    (tmp_path / f"{digest}.json").write_bytes(actual_content)
    # Manifest describes bytes of a DIFFERENT length AND different content.
    _write_ct_manifest(tmp_path, capture_retrieved_at="2026-08-01T00:00:00Z", content=b"not the real body")

    result = live.analyze_capture(tmp_path, nct_id=fx.NCT_ID, as_of="2026-09-14T00:00:00+00:00")
    assert result["capture_manifest_status"] == "CONTENT_LENGTH_MISMATCH"
    assert result["capture_manifest_error"] is not None
    assert result["capture_retrieved_at"] == UNKNOWN


def test_analyze_capture_manifest_absent_is_backward_compatible(tmp_path):
    """A capture directory with no manifest at all (e.g. from before Phase
    3D.4) must never crash and must report capture_retrieved_at UNKNOWN,
    never guessed from mtime or --as-of (Phase 3D.4 requirements 6/7)."""
    import hashlib

    url = fx.study_url(fx.NCT_ID)
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    (tmp_path / f"{digest}.json").write_text(fx.fixture_text("study_recruiting_interventional.json"), encoding="utf-8")

    result = live.analyze_capture(tmp_path, nct_id=fx.NCT_ID, as_of="2026-09-14T00:00:00+00:00")
    assert result["capture_manifest_status"] == "MISSING"
    assert result["capture_manifest_schema_version"] is None
    assert result["capture_manifest_error"] is None
    assert result["capture_retrieved_at"] == UNKNOWN
    assert result["freshness"] is not None
    assert result["freshness"].capture_retrieved_at == UNKNOWN


def test_analyze_capture_manifest_never_referenced_via_mtime_or_directory_name(tmp_path):
    """No code path in analyze_capture() ever reads the capture file's
    mtime or the capture_dir's own name for a timestamp -- capture_
    retrieved_at is sourced exclusively from the manifest object (Phase
    3D.4 requirement 7). Checked against the compiled code object's own
    referenced names, not the source text, so a docstring that merely
    explains this guarantee never causes a false failure."""
    code = live.analyze_capture.__code__
    referenced_names = set(code.co_names) | set(code.co_varnames)
    assert "mtime" not in referenced_names
    assert "st_mtime" not in referenced_names


def test_analyze_capture_manifest_malformed_json_is_an_integrity_failure_not_missing(tmp_path):
    import hashlib

    url = fx.study_url(fx.NCT_ID)
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    (tmp_path / f"{digest}.json").write_text(fx.fixture_text("study_recruiting_interventional.json"), encoding="utf-8")
    (tmp_path / f"{digest}.manifest.json").write_text("{not valid json", encoding="utf-8")

    result = live.analyze_capture(tmp_path, nct_id=fx.NCT_ID, as_of="2026-09-14T00:00:00+00:00")
    assert result["capture_manifest_status"] == "MALFORMED"
    assert result["capture_manifest_status"] != "MISSING"
    assert result["capture_manifest_error"] is not None
    assert result["capture_retrieved_at"] == UNKNOWN


def test_analyze_capture_manifest_unsupported_schema_is_an_integrity_failure(tmp_path):
    import hashlib

    url = fx.study_url(fx.NCT_ID)
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    (tmp_path / f"{digest}.json").write_text(fx.fixture_text("study_recruiting_interventional.json"), encoding="utf-8")
    (tmp_path / f"{digest}.manifest.json").write_text(
        json.dumps({"totally": "unrecognized", "shape": True}), encoding="utf-8",
    )

    result = live.analyze_capture(tmp_path, nct_id=fx.NCT_ID, as_of="2026-09-14T00:00:00+00:00")
    assert result["capture_manifest_status"] == "UNSUPPORTED_SCHEMA"
    assert result["capture_retrieved_at"] == UNKNOWN


def test_analyze_capture_manifest_io_error_is_an_integrity_failure(tmp_path, monkeypatch):
    import hashlib

    url = fx.study_url(fx.NCT_ID)
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    (tmp_path / f"{digest}.json").write_text(fx.fixture_text("study_recruiting_interventional.json"), encoding="utf-8")
    manifest_path = tmp_path / f"{digest}.manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    original_read_text = Path.read_text

    def boom(self, *args, **kwargs):
        if self == manifest_path:
            raise OSError("simulated IO error")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", boom)

    result = live.analyze_capture(tmp_path, nct_id=fx.NCT_ID, as_of="2026-09-14T00:00:00+00:00")
    assert result["capture_manifest_status"] == "IO_ERROR"
    assert result["capture_retrieved_at"] == UNKNOWN


def test_analyze_capture_manifest_body_missing_is_an_integrity_failure_distinct_from_missing(tmp_path):
    """A manifest exists but no body file was ever saved for this NCT ID --
    an Evidence Integrity failure (Phase 3D.4.1.1 correction: the manifest
    claims a capture was made for an artifact that cannot be found to
    re-verify or analyze, which is unsubstantiated evidence, not a
    harmless legacy case), and explicitly distinct from MISSING (no
    manifest at all, which stays a non-failure legacy capture)."""
    import hashlib

    url = fx.study_url(fx.NCT_ID)
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    manifest = CaptureManifest(
        schema_version=CAPTURE_MANIFEST_SCHEMA_VERSION,
        source="clinicaltrials",
        requested_url=url,
        final_url=url,
        http_status=200,
        capture_retrieved_at="2026-08-01T00:00:00Z",
        content_hash=compute_content_hash(b"whatever was captured"),
        content_length=len(b"whatever was captured"),
    )
    write_manifest(tmp_path, digest, manifest)
    # No body file written at all.

    result = live.analyze_capture(tmp_path, nct_id=fx.NCT_ID, as_of="2026-09-14T00:00:00+00:00")
    assert result["found"] is False
    assert result["capture_manifest_status"] == "BODY_MISSING"
    assert result["capture_manifest_status"] != "MISSING"
    assert result["capture_manifest_status"] != "VERIFIED"
    assert is_evidence_integrity_failure(ManifestReadStatus.BODY_MISSING)
    assert not is_evidence_integrity_failure(ManifestReadStatus.MISSING)
    assert result["capture_manifest_error"] is not None
    assert "user_agent" not in result["capture_manifest_error"].lower()
    assert "@" not in result["capture_manifest_error"]
    assert result["capture_retrieved_at"] == UNKNOWN
    assert "live_verified_candidates" not in result


def test_analyze_capture_manifest_status_serializes_as_a_plain_json_string(tmp_path, capsys):
    """End-to-end through main()'s printed --analyze-capture JSON output --
    capture_manifest_status must round-trip as a plain JSON string, never
    an Enum repr or anything else opaque (Phase 3D.4.1 requirement 12)."""
    import hashlib
    import json as json_module

    url = fx.study_url(fx.NCT_ID)
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    content = fx.fixture_text("study_recruiting_interventional.json").encode("utf-8")
    (tmp_path / f"{digest}.json").write_bytes(content)
    _write_ct_manifest(tmp_path, capture_retrieved_at="2026-08-01T00:00:00Z", content=content)

    exit_code = live.main(["--nct-id", fx.NCT_ID, "--analyze-capture", str(tmp_path)])
    assert exit_code == 0
    printed = capsys.readouterr().out
    parsed_output = json_module.loads(printed)

    assert parsed_output["capture_manifest_status"] == "VERIFIED"
    assert isinstance(parsed_output["capture_manifest_status"], str)
    assert "ManifestReadStatus" not in printed


def test_analyze_capture_never_returns_live_verified_candidates():
    """analyze_capture() never sets/returns a live_verified_candidates key
    at all -- offline re-analysis must never be conflated with a live
    run's own LIVE_VERIFIED diagnostic, regardless of manifest status
    (Phase 3D.4.1 requirement 5)."""
    result = live.analyze_capture(Path("/nonexistent"), nct_id=fx.NCT_ID)
    assert "live_verified_candidates" not in result
