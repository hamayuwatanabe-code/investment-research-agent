#!/usr/bin/env bash
# Post-edit quality gate (requirement 16).
#
# Runs formatter, linter, type check and the fast test suites after a code
# change. Exits non-zero on failure so a completion cannot be reported over a
# red suite.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

status=0
run() {
    local label="$1"; shift
    if ! "$@"; then
        echo "[hook] FAILED: ${label}" >&2
        status=1
    fi
}

if command -v ruff >/dev/null 2>&1; then
    run "ruff format" ruff format --check src tests main.py
    run "ruff lint" ruff check src tests main.py
else
    echo "[hook] ruff not installed; skipping format and lint" >&2
fi

if command -v mypy >/dev/null 2>&1; then
    run "mypy" mypy src
else
    echo "[hook] mypy not installed; skipping type check" >&2
fi

run "unit tests" python3 -m pytest tests/unit -q
run "integration tests" python3 -m pytest tests/integration -q
run "regression tests" python3 -m pytest tests/regression -q

if [ "${status}" -ne 0 ]; then
    echo "[hook] checks failed -- do not report this work as complete" >&2
fi
exit "${status}"
