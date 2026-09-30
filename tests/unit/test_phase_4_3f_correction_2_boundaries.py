"""Phase 4.3F correction 2: the files this correction touches
(``collectors/clinicaltrials.py``, ``collectors/sec_edgar.py``,
``storage/repository.py``) must stay disconnected from the Program
Identity RESOLVER (``scoring/program_identity_resolution.py``), Adaptive
Acquisition, ``AgentInput``, and the Action Gate (``scoring/
decision_gate_consistency.py``) -- this correction adds strict collector-
and snapshot-level validation, never a new wiring path for any of those.

Also verifies ``scoring.program_resolution.resolve_current_program`` (the
EXISTING, pre-Phase-4.3F program resolution used for Kill/Science/Bull/Bear
evidence routing -- unrelated to the NEW ``program_identity_resolution``
resolver) still has exactly one production call site, unchanged by this
correction.

Checked by reading each target file's own SOURCE TEXT, mirroring
``tests/unit/test_program_identity_offline_isolation.py``'s own
established pattern.
"""

from __future__ import annotations

from pathlib import Path

REPO_SRC = Path(__file__).resolve().parent.parent.parent / "src" / "investment_research"

#: The three files Phase 4.3F correction 2 actually edits.
CORRECTION_2_FILES = (
    REPO_SRC / "collectors" / "clinicaltrials.py",
    REPO_SRC / "collectors" / "sec_edgar.py",
    REPO_SRC / "storage" / "repository.py",
)

#: Names that must never appear in any of the three files above -- the
#: RESOLVER half of the offline contract layer, Adaptive Acquisition's own
#: executor/context types, AgentInput, and the Action Gate module.
FORBIDDEN_NAMES = (
    "program_identity_resolution",
    "AcquisitionExecutor",
    "ExecutionContext",
    "AgentInput",
    "decision_gate_consistency",
    "enforce_complete_only_action",
)


def test_target_files_exist_so_this_check_is_non_vacuous():
    for path in CORRECTION_2_FILES:
        assert path.is_file(), path


def test_correction_2_files_never_reference_forbidden_names():
    for path in CORRECTION_2_FILES:
        text = path.read_text(encoding="utf-8")
        for name in FORBIDDEN_NAMES:
            assert name not in text, f"{path} references forbidden name {name!r}"


def test_repository_may_import_identifier_validation_but_not_the_resolver():
    """storage/repository.py's Finding 5 fix DOES import
    scoring.identifier_validation (the strict NCT/PMID validators -- not
    forbidden, see test_program_identity_offline_isolation.py's own scope,
    which never lists storage/repository.py) -- but must still never
    import the resolver half."""
    text = (REPO_SRC / "storage" / "repository.py").read_text(encoding="utf-8")
    assert "identifier_validation" in text
    assert "program_identity_resolution" not in text


def test_resolve_current_program_has_exactly_one_production_call_site():
    """scoring.program_resolution.resolve_current_program (the EXISTING
    resolver used by Kill/Science/Bull/Bear -- unrelated to the NEW
    program_identity_resolution module) must still be called from exactly
    one place in src/: orchestrator/pipeline.py. Phase 4.3F correction 2
    adds no second call site anywhere, including in the three files it
    touches."""
    call_sites = []
    for path in sorted(REPO_SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "resolve_current_program(" in stripped and not stripped.startswith(
                "def resolve_current_program"
            ):
                call_sites.append((path, line_no, stripped))
    assert len(call_sites) == 1, f"expected exactly one call site, found: {call_sites}"
    path, _line_no, _line = call_sites[0]
    assert path == REPO_SRC / "orchestrator" / "pipeline.py"
