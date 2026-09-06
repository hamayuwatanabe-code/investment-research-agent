# Agent prompt specifications

These Markdown files are the **prompt contracts** for the fourteen agents. They
are the specification an LLM-backed implementation must follow, and they mirror
the Python implementations in `src/investment_research/agents/`.

Two things are true of every agent here and are enforced in code, not by
politeness:

1. An agent only ever sees the input the orchestrator projected for it through
   `orchestrator/isolation.py`. Anything its policy does not name is absent.
2. Anything it cannot establish is reported as `UNKNOWN`, `NOT FOUND` or
   `INSUFFICIENT EVIDENCE`. Filling a gap with a plausible guess is the single
   most damaging thing any agent here can do.

If you use these as Claude Code subagent definitions, keep the "MUST NOT see"
section intact: it is the machine-checkable part of the contract.
