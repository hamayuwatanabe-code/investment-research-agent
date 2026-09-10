"""The 49 legacy adversarial-search templates as individually-addressable
``LegacyResearchNeed`` rows, plus their exact-match deduplication into groups.

Source of the 49 templates (unchanged, not modified by this module):

* ``research.adversarial.BEAR_TEMPLATES`` -- 17 rows
* ``research.adversarial.BULL_TEMPLATES`` -- 6 rows
* ``collectors.search.KILL_QUERY_TEMPLATES`` -- 18 rows, plus its own
  self-duplication of its first 8 entries under a company-name substitution
  (``collectors.search.kill_queries``) -- 8 more rows -- for 26 kill rows
  total.

17 + 6 + 26 = 49.

Deduplication in Phase 1 is exact-match only, on ``(equivalence_key, domain)``
-- never fuzzy, never token-overlap, never LLM-assisted (that is
``research.adversarial._dedupe_semantically``, a different mechanism, used for
a different purpose, and explicitly out of scope for this dedup). Two legacy
needs collapse into one group if and only if their ``equivalence_key`` string
is byte-identical AND their ``domain`` enum member is identical.

Running ``group_legacy_needs()`` over ``LEGACY_CATALOG`` produces exactly 31
groups, collapsing 18 of the 49 rows (49 - 31 = 18). This is locked by
``tests/unit/test_legacy_catalog.py`` -- see that file for the full row-by-row
accounting of which of the 49 collapse into which of the 31 groups and why
(e.g. why ``bear_2`` "{d} endpoint concern" and ``kill_3`` "{t} endpoint" do
NOT merge despite both falling under ``ResearchDomain.REGULATORY``: their
equivalence keys, "entity endpoint concern" and "entity endpoint", are not
identical strings).
"""

from __future__ import annotations

import re
from collections import defaultdict

from ..schemas.enums import ResearchDomain
from .checks import AcquisitionOrigin, LegacyResearchNeed

_PLACEHOLDER_RE = re.compile(r"\{[tcd]\}")
_WS_RE = re.compile(r"\s+")


def equivalence_key(template: str) -> str:
    """Normalize a raw template to its exact-match dedup key.

    Every ``{t}``/``{c}``/``{d}`` placeholder becomes the literal token
    ``ENTITY``; the result is lowercased and whitespace-normalized. This is a
    string-identity operation, not a similarity measure: "{c} going concern"
    and "{t} going concern" produce the identical key "entity going concern",
    while "{d} endpoint concern" and "{t} endpoint" do not, because their
    non-placeholder words differ.
    """
    normalized = _PLACEHOLDER_RE.sub("ENTITY", template)
    return _WS_RE.sub(" ", normalized).strip().lower()


#: (legacy_need_id, original_template, domain) for research.adversarial.BEAR_TEMPLATES,
#: in its exact source order.
_BEAR_ROWS: tuple[tuple[str, str, ResearchDomain], ...] = (
    ("bear_0", "{t} FDA concern", ResearchDomain.REGULATORY),
    ("bear_1", "{c} regulatory risk", ResearchDomain.REGULATORY),
    ("bear_2", "{d} endpoint concern", ResearchDomain.REGULATORY),
    ("bear_3", "{c} delayed catalyst", ResearchDomain.CATALYST),
    ("bear_4", "{c} failed trial", ResearchDomain.SCIENCE_TECHNOLOGY),
    ("bear_5", "{c} dilution", ResearchDomain.CAPITAL_STRUCTURE),
    ("bear_6", "{c} going concern", ResearchDomain.CAPITAL_STRUCTURE),
    ("bear_7", "{c} warrant", ResearchDomain.CAPITAL_STRUCTURE),
    ("bear_8", "{c} reverse split", ResearchDomain.CAPITAL_STRUCTURE),
    ("bear_9", "{c} delisting", ResearchDomain.CAPITAL_STRUCTURE),
    ("bear_10", "{c} lawsuit", ResearchDomain.CONTRADICTION),
    ("bear_11", "{c} auditor", ResearchDomain.CONTRADICTION),
    ("bear_12", "{c} insider selling", ResearchDomain.CONTRADICTION),
    ("bear_13", "{c} criticism", ResearchDomain.CONTRADICTION),
    ("bear_14", "{c} short thesis", ResearchDomain.COMPETITION),
    ("bear_15", "{d} competitor superiority", ResearchDomain.COMPETITION),
    ("bear_16", "{d} safety concern", ResearchDomain.SCIENCE_TECHNOLOGY),
)

#: research.adversarial.BULL_TEMPLATES, exact source order.
_BULL_ROWS: tuple[tuple[str, str, ResearchDomain], ...] = (
    ("bull_0", "{c} clinical data results", ResearchDomain.SCIENCE_TECHNOLOGY),
    ("bull_1", "{c} partnership agreement", ResearchDomain.COMPETITION),
    ("bull_2", "{c} FDA designation", ResearchDomain.REGULATORY),
    ("bull_3", "{c} cash runway financing", ResearchDomain.CAPITAL_STRUCTURE),
    ("bull_4", "{d} mechanism efficacy", ResearchDomain.SCIENCE_TECHNOLOGY),
    ("bull_5", "{c} upcoming catalyst readout", ResearchDomain.CATALYST),
)

#: collectors.search.KILL_QUERY_TEMPLATES, exact source order (kill_0..kill_17),
#: domain assigned via the same routing agents.kill_agent.QUERY_CATEGORY_MAP /
#: agents.kill_agent._KILL_CATEGORY_TO_RESEARCH_DOMAIN use for these exact
#: strings (agents.kill_agent._domain_for_kill_query).
_KILL_TEMPLATE_ROWS: tuple[tuple[str, str, ResearchDomain], ...] = (
    ("kill_0", "{t} FDA concern", ResearchDomain.REGULATORY),
    ("kill_1", "{t} regulatory risk", ResearchDomain.REGULATORY),
    ("kill_2", "{t} failed", ResearchDomain.SCIENCE_TECHNOLOGY),
    ("kill_3", "{t} endpoint", ResearchDomain.REGULATORY),
    ("kill_4", "{t} dilution", ResearchDomain.CAPITAL_STRUCTURE),
    ("kill_5", "{t} going concern", ResearchDomain.CAPITAL_STRUCTURE),
    ("kill_6", "{t} reverse split", ResearchDomain.CAPITAL_STRUCTURE),
    ("kill_7", "{t} delisting", ResearchDomain.CAPITAL_STRUCTURE),
    ("kill_8", "{t} lawsuit", ResearchDomain.CONTRADICTION),
    ("kill_9", "{t} accounting", ResearchDomain.CONTRADICTION),
    ("kill_10", "{t} auditor", ResearchDomain.CONTRADICTION),
    ("kill_11", "{t} clinical hold", ResearchDomain.SCIENCE_TECHNOLOGY),
    ("kill_12", "{t} insider selling", ResearchDomain.CONTRADICTION),
    ("kill_13", "{t} short thesis", ResearchDomain.COMPETITION),
    ("kill_14", "{t} Complete Response Letter", ResearchDomain.REGULATORY),
    ("kill_15", "{t} SEC investigation", ResearchDomain.CONTRADICTION),
    ("kill_16", "{t} restatement", ResearchDomain.CONTRADICTION),
    ("kill_17", "{t} offering priced", ResearchDomain.CAPITAL_STRUCTURE),
)

#: collectors.search.kill_queries()'s own self-duplication: when a company
#: name is supplied, KILL_QUERY_TEMPLATES[:8] is re-formatted against the
#: company name and appended -- the SAME template text as kill_0..kill_7,
#: just substituted differently at format time, hence the same
#: equivalence_key and domain as their kill_0..kill_7 counterparts.
_KILL_DUPLICATE_ROWS: tuple[tuple[str, str, ResearchDomain], ...] = tuple(
    (f"kill_{18 + i}", template, domain) for i, (_, template, domain) in enumerate(_KILL_TEMPLATE_ROWS[:8])
)

assert len(_BEAR_ROWS) == 17
assert len(_BULL_ROWS) == 6
assert len(_KILL_TEMPLATE_ROWS) == 18
assert len(_KILL_DUPLICATE_ROWS) == 8


def _build_catalog() -> tuple[LegacyResearchNeed, ...]:
    rows: list[LegacyResearchNeed] = []
    for need_id, template, domain in _BEAR_ROWS:
        rows.append(
            LegacyResearchNeed(
                legacy_need_id=need_id,
                origin=AcquisitionOrigin.BEAR,
                original_template=template,
                domain=domain,
                equivalence_key=equivalence_key(template),
            )
        )
    for need_id, template, domain in _BULL_ROWS:
        rows.append(
            LegacyResearchNeed(
                legacy_need_id=need_id,
                origin=AcquisitionOrigin.BULL,
                original_template=template,
                domain=domain,
                equivalence_key=equivalence_key(template),
            )
        )
    for need_id, template, domain in (*_KILL_TEMPLATE_ROWS, *_KILL_DUPLICATE_ROWS):
        rows.append(
            LegacyResearchNeed(
                legacy_need_id=need_id,
                origin=AcquisitionOrigin.KILL,
                original_template=template,
                domain=domain,
                equivalence_key=equivalence_key(template),
            )
        )
    return tuple(rows)


#: All 49 legacy needs, individually addressable by ``legacy_need_id``.
LEGACY_CATALOG: tuple[LegacyResearchNeed, ...] = _build_catalog()

assert len(LEGACY_CATALOG) == 49
assert len({need.legacy_need_id for need in LEGACY_CATALOG}) == 49


def group_legacy_needs(
    needs: tuple[LegacyResearchNeed, ...] = LEGACY_CATALOG,
) -> dict[tuple[str, ResearchDomain], tuple[str, ...]]:
    """Group needs by exact ``(equivalence_key, domain)`` match.

    Returns a mapping from the group's key to the ``legacy_need_id``s in it,
    in catalog order. Over ``LEGACY_CATALOG`` this produces exactly 31 groups.
    No fuzzy or token-overlap comparison is performed anywhere in this
    function -- grouping is a plain dict keyed on an exact string+enum pair.
    """
    groups: dict[tuple[str, ResearchDomain], list[str]] = defaultdict(list)
    for need in needs:
        groups[(need.equivalence_key, need.domain)].append(need.legacy_need_id)
    return {key: tuple(ids) for key, ids in groups.items()}


def collapsed_count(needs: tuple[LegacyResearchNeed, ...] = LEGACY_CATALOG) -> int:
    """How many of ``needs`` disappear into a group with another (49 - 31 = 18
    over the full catalog): for every group of size n, n - 1 rows collapse.
    """
    groups = group_legacy_needs(needs)
    return len(needs) - len(groups)
