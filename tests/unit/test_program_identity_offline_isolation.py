"""Phase 4.3B requirement 8, as narrowed by Phase 4.3F and Phase 4.3G.

Four offline contract-layer modules now exist: ``scoring/program_evidence.py``,
``scoring/identifier_validation.py``, ``scoring/program_identity_
resolution.py``, and ``scoring/adaptive_acquisition_plan.py``.

ALL FOUR must still never be imported by ``cli.py`` or any ``agents/*.py``
module: no Program Identity Resolution execution, no Adaptive Acquisition
Plan, no Agent delivery, no Action-gate involvement, ever reaches an Agent
or the CLI directly.

``orchestrator/pipeline.py``'s own relationship to these four is narrower,
not blanket-forbidden, and has changed TWICE:

* Phase 4.3F connected ``program_evidence`` (the frozen EVIDENCE TYPES
  only) to ``pipeline.py`` -- producing/restoring/snapshotting typed
  Production collector output. ``identifier_validation``/
  ``program_identity_resolution`` stayed forbidden there.
* Phase 4.3G connected ``program_identity_resolution`` (the RESOLVER
  functions themselves: ``resolve_program_identity``/
  ``resolve_literature_link``) and the new ``adaptive_acquisition_plan``
  to ``pipeline.py`` too -- a bootstrap resolution computed once, after
  Stage 2's full Evidence Integrity pass and before Stage 2b escalation,
  stored as plain internal diagnostic fields on ``ResearchResult`` (never
  reaching an ``AgentInput``, never affecting ``RunStatus``/
  ``Verdict.action``/``blocking_reasons`` -- see
  ``tests/unit/test_adaptive_acquisition_plan.py`` and
  ``tests/integration/test_phase_4_3g_pipeline_integration.py`` for those
  guarantees). ``identifier_validation`` alone stays forbidden in
  ``pipeline.py`` even now: the one place that module's validators are
  needed from Pipeline-reachable code is inside
  ``adaptive_acquisition_plan.py`` itself (which pipeline.py calls into,
  never ``identifier_validation`` directly).

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

#: All four -- used by checks that must hold regardless of Phase 4.3F/4.3G
#: (cli.py, agents/*.py, isolation.py, agent_io.py; the modules' own
#: mutual-isolation checks).
NEW_MODULE_NAMES = (
    "program_evidence",
    "identifier_validation",
    "program_identity_resolution",
    "adaptive_acquisition_plan",
)

#: The one that must STILL never reach pipeline.py even after Phase
#: 4.3G -- see this file's own module docstring for why.
RESOLVER_MODULE_NAMES_STILL_FORBIDDEN_IN_PIPELINE = ("identifier_validation",)

#: The two Phase 4.3G intentionally connects to pipeline.py.
PHASE_4_3G_PIPELINE_MODULE_NAMES = ("program_identity_resolution", "adaptive_acquisition_plan")

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
    """cli.py and every agents/*.py module: all four names forbidden --
    neither Phase 4.3F's structured evidence nor Phase 4.3G's resolver/
    plan connection ever makes these Agent-visible or CLI-visible."""
    for path in TARGET_FILES:
        text = path.read_text(encoding="utf-8")
        for module_name in NEW_MODULE_NAMES:
            assert module_name not in text, f"{path} references {module_name!r}"


def test_pipeline_never_references_identifier_validation_directly():
    """pipeline.py: ``identifier_validation`` stays forbidden even after
    Phase 4.3G -- the one place its validators are needed from
    Pipeline-reachable code is inside ``adaptive_acquisition_plan.py``
    itself, never pipeline.py directly."""
    text = PIPELINE_FILE.read_text(encoding="utf-8")
    for module_name in RESOLVER_MODULE_NAMES_STILL_FORBIDDEN_IN_PIPELINE:
        assert module_name not in text, f"{PIPELINE_FILE} references {module_name!r}"


def test_pipeline_does_reference_program_evidence_types():
    """The positive counterpart: Phase 4.3F's whole point is that
    pipeline.py DOES now carry scoring.program_evidence's typed evidence
    (aggregating/restoring CollectionResult's new fields for the collect
    snapshot) -- this failing would mean that wiring silently regressed."""
    text = PIPELINE_FILE.read_text(encoding="utf-8")
    assert "program_evidence" in text


def test_pipeline_does_reference_the_phase_4_3g_resolver_and_plan_modules():
    """The positive counterpart for Phase 4.3G: pipeline.py DOES now call
    into program_identity_resolution (resolve_program_identity/
    resolve_literature_link, for the bootstrap resolution) and
    adaptive_acquisition_plan (build_adaptive_acquisition_plan) -- this
    failing would mean that wiring silently regressed."""
    text = PIPELINE_FILE.read_text(encoding="utf-8")
    for module_name in PHASE_4_3G_PIPELINE_MODULE_NAMES:
        assert module_name in text, f"{PIPELINE_FILE} does not reference {module_name!r}"


def test_new_modules_never_referenced_in_isolation_policy_or_agent_io():
    """The isolation core itself (Phase 4.3B requirement 8 implies: no
    IsolationPolicy/Channel change either) must be untouched by Phase
    4.3F or Phase 4.3G."""
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
    import investment_research.scoring.adaptive_acquisition_plan as adaptive_acquisition_plan
    import investment_research.scoring.identifier_validation as identifier_validation
    import investment_research.scoring.program_evidence as program_evidence
    import investment_research.scoring.program_identity_resolution as program_identity_resolution

    for module in (
        program_evidence, identifier_validation, program_identity_resolution,
        adaptive_acquisition_plan,
    ):
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
    itself must not reference the new modules either -- Phase 4.3F/4.3G add
    new, parallel, disconnected code, never a rewire of the existing one."""
    existing = REPO_SRC / "scoring" / "program_resolution.py"
    text = existing.read_text(encoding="utf-8")
    for module_name in NEW_MODULE_NAMES:
        assert module_name not in text, f"program_resolution.py references {module_name!r}"
