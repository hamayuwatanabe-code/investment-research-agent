"""Per-agent LLM reasoning-effort policy (requirement G).

A live run's interpretive stage consumed 55,846 of its 60,000-token quota
before Science/Competitive ever got a turn. Raising the quota further only
delays the same failure -- the fix is to spend effort where it matters most.

Regulatory, Science, Contradiction, Kill and the Blind Judge falsify the
thesis or render the final judgement; degrading their reasoning to save cost
is exactly what CLAUDE.md's non-negotiable rules forbid. Competitive, Bear
and Bull construct a case FROM evidence those agents (and the deterministic
core) already gathered -- a real place to spend less without weakening
evidence integrity.

This is DATA, not agent logic: no agent decides its own effort, and the
policy can be replaced wholesale by a caller (e.g. a future CLI flag) without
touching a single agent class.
"""

from __future__ import annotations

#: Effort levels, weakest to strongest -- matches LLMClient's `effort`
#: choices (see cli.py's --llm-effort / --research-effort).
EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

#: Default per-agent effort policy (requirement G). An agent_id absent from
#: this mapping simply runs at the run's configured ceiling (--llm-effort).
DEFAULT_AGENT_EFFORT_POLICY: dict[str, str] = {
    "regulatory": "high",
    "science": "high",
    "contradiction": "high",
    "kill_agent": "high",
    "blind_judge": "high",
    "competitive": "medium",
    "bear_agent": "medium",
    "bull_agent": "medium",
}


def _effort_rank(effort: str) -> int:
    try:
        return EFFORT_LEVELS.index(effort)
    except ValueError:
        # An unrecognized level cannot be safely compared -- treat it as
        # already at the ceiling rather than guessing a rank for it.
        return len(EFFORT_LEVELS) - 1


def resolve_effort(
    agent_id: str,
    *,
    ceiling: str,
    policy: dict[str, str] | None = None,
    allow_exceed_ceiling: bool = False,
) -> str:
    """The effort level to use for ``agent_id``'s LLM calls this run.

    ``ceiling`` is ``--llm-effort`` -- the run's configured default/maximum.
    The per-agent ``policy`` (``DEFAULT_AGENT_EFFORT_POLICY`` unless a
    caller supplies its own) may only LOWER an agent's effort below the
    ceiling, never raise it above what the run was invoked with -- unless
    ``allow_exceed_ceiling=True`` is explicitly passed, the one documented
    escape hatch ("unless explicitly configured otherwise").
    """
    active_policy = DEFAULT_AGENT_EFFORT_POLICY if policy is None else policy
    requested = active_policy.get(agent_id)
    if requested is None:
        return ceiling
    if not allow_exceed_ceiling and _effort_rank(requested) > _effort_rank(ceiling):
        return ceiling
    return requested
