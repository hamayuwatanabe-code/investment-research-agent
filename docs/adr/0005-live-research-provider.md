# ADR 0005: Anthropic server-side web tools as the primary research path

## Status
Accepted.

## Context

Phase 1 shipped collectors for SEC EDGAR, ClinicalTrials.gov and openFDA, and
they worked — but in the environment the system actually runs in, all three
hosts are blocked by an egress policy (403 at the proxy CONNECT). So was every
news and IR domain. A research system that can only research when the network
is unrestricted is not a research system.

Four options were on the table:

1. Claude/Anthropic's own web search and fetch
2. An MCP web-search provider
3. A commercial search API (Tavily, Brave)
4. Direct API access from the user's own machine

## Decision

**A dual path, with Anthropic's server-side tools as primary.**

### Primary: `web_search_20260209` / `web_fetch_20260209`

Declared as tools on the Messages API. Both execute on Anthropic's
infrastructure, not in the client. That property is the reason this option wins
rather than merely being convenient:

- **It does not circumvent anything.** The program is not reaching a blocked
  host and being denied; it is not reaching that host at all. The distinction
  matters, because working *around* an egress policy would be a different act
  with a different meaning, and one this project will not perform.
- **One credential, one channel.** The same client serves both the research and
  the LLM agents. Two integrations would mean two failure modes and two ways to
  be half-configured.
- **Citations come back natively**, which feeds the traceability requirement
  directly rather than being reconstructed.

Two constraints are recorded rather than discovered later: `web_fetch` only
fetches URLs already present in the conversation, so a search must precede a
fetch; and web search is US-only, which matters for the Japanese-equity ambition
and is listed under limitations.

### Complement: direct structured APIs

SEC EDGAR, ClinicalTrials.gov v2 and openFDA are free, keyless, and *structured*.
For a share count or a registered primary outcome measure, parsing a JSON field
beats reading a search result about it. They stay in the system and are used
wherever the environment can reach them.

### Options 2 and 3

Both drop into the existing `SearchProvider` / `ResearchProvider` interfaces as
configuration. They are supported, not architectural.

### What was rejected

Reusing the host's own Claude Code credentials from inside the program. They are
granted to the agent session, not to an arbitrary Python process, and taking
them would be exactly the circumvention this ADR is written to avoid.

## Consequences

- `ResearchPath` is recorded on every fact, and the report states which channels
  served the run. A reader can tell a filing from a search summary from a replay.
- The `ContentKind` distinction became necessary and is arguably the most
  valuable thing to come out of this decision: a search engine's summary *about*
  a filing is not the filing. Treating them as the same is how "the filing says
  X" quietly becomes "a search result said the filing says X". Summaries are
  confidence-capped and their material claims are escalated.
- In an environment with no credentials and no egress — the one this was built
  in — the honest outcome is a `CAPTURED` corpus replay plus a report that says
  so. That is worse than live research and better than a fabrication.
