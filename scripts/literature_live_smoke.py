#!/usr/bin/env python3
"""Phase 3F.1: Literature (PubMed/Europe PMC) Live Smoke entry point --
OFFLINE this phase. No real NCBI/Europe PMC network call is made or
intended; this validates the plan/CLI/credential-gating/Capture-Manifest-
replay surface ahead of a future live-communication phase:

    python3 scripts/literature_live_smoke.py --nct-id NCT01234567
    python3 scripts/literature_live_smoke.py --pmid 12345678
    python3 scripts/literature_live_smoke.py --nct-id NCT01234567 --max-articles 5 --max-fulltext-fetches 2
    python3 scripts/literature_live_smoke.py --nct-id NCT01234567 --force-rerun

To re-analyze an already-saved capture directory OFFLINE (no network, no
marker, no credentials required, safe to re-run any number of times):

    python3 scripts/literature_live_smoke.py \
        --analyze-capture data/live_smoke/literature/<TIMESTAMP>

See ``python3 scripts/literature_live_smoke.py --help`` for every option,
and ``src/investment_research/research/literature_live_smoke.py``'s module
docstring for exactly what this does and does not do:

* Refuses outright, before any request, if IRA_NCBI_TOOL or IRA_NCBI_EMAIL
  is unset/blank (IRA_NCBI_API_KEY is always optional). Values are never
  printed -- only whether each is configured.
* Read-only GET requests to NCBI E-utilities / Europe PMC hosts ONLY, with
  an explicit, printed request-count formula and Phase 3F.0.4's pre-connect
  redirect validation.
* Zero Anthropic API calls, zero LLM calls, zero Web Search calls, no Full
  DD, and this NEVER connects to Pipeline.run() -- it is a standalone
  diagnostic script, never imported by production code.
* Never prints an abstract or full-text body -- only structural metadata
  (PMID, publication stage, peer-review status, publication types, NCT-ID
  match, OA/full-text-acquired status, Document authority/content_kind).
* Meant to run ONCE per invocation -- a marker file records the attempt
  (only on a real, non-refused run) and a second invocation refuses unless
  --force-rerun is passed deliberately.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from investment_research.research.literature_live_smoke import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
