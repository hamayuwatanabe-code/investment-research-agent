"""Phase 4.3B requirement 8: the three new offline contract-layer modules
(``scoring/program_evidence.py``, ``scoring/identifier_validation.py``,
``scoring/program_identity_resolution.py``) must never be imported by
``cli.py``, ``orchestrator/pipeline.py``, or any ``agents/*.py`` module --
this phase implements ONLY the offline contract layer, with zero Pipeline/
CLI/Agent wiring.

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

NEW_MODULE_NAMES = (
    "program_evidence",
    "identifier_validation",
    "program_identity_resolution",
)

TARGET_FILES = (
    REPO_SRC / "cli.py",
    REPO_SRC / "orchestrator" / "pipeline.py",
    *sorted((REPO_SRC / "agents").glob("*.py")),
)


def test_target_files_exist_so_this_check_is_non_vacuous():
    assert (REPO_SRC / "cli.py").is_file()
    assert (REPO_SRC / "orchestrator" / "pipeline.py").is_file()
    agent_files = list((REPO_SRC / "agents").glob("*.py"))
    assert len(agent_files) >= 10  # sanity: the real agents/ directory, not an empty one


def test_new_modules_never_referenced_in_cli_pipeline_or_any_agent_source():
    for path in TARGET_FILES:
        text = path.read_text(encoding="utf-8")
        for module_name in NEW_MODULE_NAMES:
            assert module_name not in text, f"{path} references {module_name!r}"


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
