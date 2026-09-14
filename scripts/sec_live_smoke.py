#!/usr/bin/env python3
"""Phase 3C: SEC-only live smoke test entry point. Run this on a machine
where ``IRA_SEC_USER_AGENT`` is already set in the shell environment (this
tool never creates or requires a ``.env`` file, and never prints the value):

    python3 scripts/sec_live_smoke.py
    python3 scripts/sec_live_smoke.py --cik 320193 --out-dir /tmp/sec_smoke
    python3 scripts/sec_live_smoke.py --force-rerun

To re-analyze an already-saved capture directory OFFLINE (no network,
no marker, safe to re-run any number of times):

    python3 scripts/sec_live_smoke.py --analyze-capture data/live_smoke/sec_edgar/<TIMESTAMP> \
        --cik 320193 --accession 0000320193-25-000079 \
        --primary-document aapl-20250927.htm \
        --exhibit-filename a10-kexhibit31109272025.htm --exhibit-type EX-31.1

See ``python3 scripts/sec_live_smoke.py --help`` for every option, and
``src/investment_research/research/sec_live_smoke.py``'s module docstring
for exactly what this does and does not do:

* Read-only GET requests to data.sec.gov / www.sec.gov ONLY, hard-capped at
  6 real requests per run, redirect targets re-validated against the same
  host allowlist.
* Zero Anthropic API calls, zero LLM calls, zero Web Search calls, no Full
  DD, and this NEVER connects to Pipeline.run() -- it is a standalone
  diagnostic script, never imported by production code.
* Refuses outright, before any network call, if IRA_SEC_USER_AGENT is
  unset, blank, or still the repository's built-in placeholder value.
* Meant to run ONCE -- a marker file records the attempt and a second
  invocation refuses unless ``--force-rerun`` is passed deliberately.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from investment_research.research.sec_live_smoke import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
