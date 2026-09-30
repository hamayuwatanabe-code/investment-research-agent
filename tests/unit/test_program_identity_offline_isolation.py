"""Phase 4.3B requirement 8 (as narrowed by Phase 4.3F): of the three
offline contract-layer modules (``scoring/program_evidence.py``,
``scoring/identifier_validation.py``, ``scoring/program_identity_
resolution.py``), the RESOLVER-side two --
``identifier_validation``/``program_identity_resolution`` -- must still
never be imported by ``cli.py``, ``orchestrator/pipeline.py``, or any
``agents/*.py`` module: no Program Identity Resolution execution, no
Adaptive Acquisition, no Agent delivery, no Action-gate involvement --
exactly the exclusions Phase 4.3F's own task scope names explicitly.

``scoring/program_evidence.py`` (the frozen EVIDENCE TYPES themselves,
never the resolver) is the one deliberate exception, starting Phase 4.3F:
that phase's whole purpose is producing these types as typed Production
collector output (``collectors/base.py``/``collectors/sec_edgar.py``/
``collectors/clinicaltrials.py``) and snapshotting them losslessly
(``storage/repository.py``, threaded through ``orchestrator/pipeline.py``).
``orchestrator/pipeline.py`` referencing ``program_evidence`` is therefore
now REQUIRED, not forbidden -- checked by a dedicated positive test below,
so a future accidental removal of that wiring is caught just as loudly as
an accidental resolver import would be. ``cli.py`` and every ``agents/*.py``
module must still never reference ANY of the three names, including
``program_evidence`` -- Phase 4.3F's own scope is collectors/storage/
Pipeline-internal only, never Agent-visible.

Checked by reading each target file's own SOURCE TEXT (never merely
``sys.modules``/``__dict__`` after import) -- a stronger guarantee than a
runtime check, since it also catches an import hidden inside a function
body that only executes conditionally. Mirrors the existing precedent in
``tests/unit/test_literature_pipeline_integration.py::
test_pipeline_never_calls_literature_acquisition_code_directly``.
"""

from __future__ import annotations

from pathlib import Path

REPO_SRC = Path(__file__).resolve().parent.parent.parent / "src" / "investment_research"

#: All three -- used by checks that must hold regardless of Phase 4.3F
#: (cli.py, agents/*.py, isolation.py, agent_io.py; the modules' own
#: mutual-isolation checks).
NEW_MODULE_NAMES = (
    "program_evidence",
    "identifier_validation",
    "program_identity_resolution",
)

#: The two that must STILL never reach pipeline.py after Phase 4.3F --
#: the resolver/validation half, never the frozen evidence types.
RESOLVER_MODULE_NAMES = ("identifier_validation", "program_identity_resolution")

TARGET_FILES = (
    REPO_SRC / "cli.py",
    *sorted((REPO_SRC / "agents").glob("*.py")),
)

PIPELINE_FILE = REPO_SRC / "orchestrator" / "pipeline.py"


def test_target_files_exist_so_this_check_is_non_vacuous():
    assert (REPO_SRC / "cli.py").is_file()
    assert (REPO_SRC / "orchestrator" / "pipeline.py").is_file()
    agent_files = list((REPO_SRC / "agents").glob("*.py"))
    assert len(agent_files) >= 10  # sanity: the real agents/ directory, not an empty one


def test_new_modules_never_referenced_in_cli_or_any_agent_source():
    """cli.py and every agents/*.py module: all three names forbidden,
    program_evidence included -- Phase 4.3F never makes structured evidence
    Agent-visible or CLI-visible."""
    for path in TARGET_FILES:
        text = path.read_text(encoding="utf-8")
        for module_name in NEW_MODULE_NAMES:
            assert module_name not in text, f"{path} references {module_name!r}"


def test_pipeline_never_references_the_resolver_modules():
    """pipeline.py: the resolver/validation half stays forbidden -- Phase
    4.3F connects only the frozen evidence TYPES, never
    resolve_program_identity()/resolve_literature_link() or their
    identifier-validation helpers."""
    text = PIPELINE_FILE.read_text(encoding="utf-8")
    for module_name in RESOLVER_MODULE_NAMES:
        assert module_name not in text, f"{PIPELINE_FILE} references {module_name!r}"


def test_pipeline_does_reference_program_evidence_types():
    """The positive counterpart: Phase 4.3F's whole point is that
    pipeline.py DOES now carry scoring.program_evidence's typed evidence
    (aggregating/restoring CollectionResult's new fields for the collect
    snapshot) -- this failing would mean that wiring silently regressed."""
    text = PIPELINE_FILE.read_text(encoding="utf-8")
    assert "program_evidence" in text


def test_new_modules_never_referenced_in_isolation_policy_or_agent_io():
    """The isolation core itself (Phase 4.3B requirement 8 implies: no
    IsolationPolicy/Channel change either) must be untouched by this
    phase."""
    extra_targets = (
        REPO_SRC / "orchestrator" / "isolation.py",
        REPO_SRC / "schemas" / "agent_io.py",
    )
    for path in extra_targets:
        text = path.read_text(encoding="utf-8")
        for module_name in NEW_MODULE_NAMES:
            assert module_name not in text, f"{path} references {module_name!r}"


def test_new_modules_import_cleanly_standalone_with_no_pipeline_agent_dependency():
    """The reverse direction: these new modules must themselves be
    importable without pulling in Pipeline/Agent/HTTP-capable code, so a
    future test/caller can use them in true isolation."""
    import investment_research.scoring.identifier_validation as identifier_validation
    import investment_research.scoring.program_evidence as program_evidence
    import investment_research.scoring.program_identity_resolution as program_identity_resolution

    for module in (program_evidence, identifier_validation, program_identity_resolution):
        assert "pipeline" not in module.__dict__
        assert "cli" not in module.__dict__
        assert not any(name.startswith("LLM") for name in module.__dict__)


def test_new_modules_never_import_http_client_or_acquisition_executor():
    for module_name in NEW_MODULE_NAMES:
        path = REPO_SRC / "scoring" / f"{module_name}.py"
        text = path.read_text(encoding="utf-8")
        assert "HttpClient" not in text
        assert "AcquisitionExecutor" not in text
        assert "ExecutionContext" not in text


def test_existing_program_resolution_module_is_untouched_by_this_phase():
    """Phase 4.3B requirement 8: existing ProgramResolution call sites
    (agents/kill_agent.py, agents/science.py) and scoring/program_resolution.py
    itself must not reference the new modules either -- this phase adds new,
    parallel, disconnected code, never a rewire of the existing one."""
    existing = REPO_SRC / "scoring" / "program_resolution.py"
    text = existing.read_text(encoding="utf-8")
    for module_name in NEW_MODULE_NAMES:
        assert module_name not in text, f"program_resolution.py references {module_name!r}"
