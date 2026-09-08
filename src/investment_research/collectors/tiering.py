"""Source-tier classification and syndication detection (requirements 2, 3, 13)."""

from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse

from ..schemas.enums import DocumentAuthority, SourceTier

#: Host suffix -> tier.  Ordered specific-to-general at lookup time.
_TIER_1_HOSTS = {
    "sec.gov",
    "efts.sec.gov",
    "data.sec.gov",
    "fda.gov",
    "accessdata.fda.gov",
    "clinicaltrials.gov",
    "nih.gov",
    "clinicalregulatory.nih.gov",
    "nasa.gov",
    "defense.gov",
    "dod.gov",
    "dhs.gov",
    "sam.gov",
    "usaspending.gov",
    "fpds.gov",
    "gsa.gov",
    "federalregister.gov",
    "regulations.gov",
    "ema.europa.eu",
    "pmda.go.jp",
    "mhlw.go.jp",
    "fsa.go.jp",
    "gov.uk",
    "mhra.gov.uk",
    "hc-sc.gc.ca",
    "nasdaq.com",
    "nyse.com",
    "jpx.co.jp",
    "release.tdnet.info",
    "disclosure.edinet-fsa.go.jp",
    "edinet-fsa.go.jp",
    "kanpou.npb.go.jp",
}

_TIER_2_HOSTS = {
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "nejm.org",
    "thelancet.com",
    "jamanetwork.com",
    "nature.com",
    "science.org",
    "cell.com",
    "bmj.com",
    "annalsofoncology.org",
    "ascopubs.org",
    "aacrjournals.org",
    "biorxiv.org",
    "medrxiv.org",
    "arxiv.org",
    "doi.org",
    "europepmc.org",
    "nist.gov",
    "energy.gov",
    "osti.gov",
}

_TIER_3_HOSTS = {
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "nikkei.com",
    "apnews.com",
    "economist.com",
    "barrons.com",
    "statnews.com",
    "endpts.com",
    "fiercebiotech.com",
    "fiercepharma.com",
    "biopharmadive.com",
    "scrip.pharmaintelligence.informa.com",
    "aviationweek.com",
    "spacenews.com",
    "eetimes.com",
    "semianalysis.com",
    "theinformation.com",
    "cnbc.com",
}

_TIER_4_HOSTS = {
    "morganstanley.com",
    "goldmansachs.com",
    "jpmorgan.com",
    "bofa.com",
    "citi.com",
    "jefferies.com",
    "cantor.com",
    "hcwresearch.com",
    "zacks.com",
    "tipranks.com",
    "marketbeat.com",
    "benzinga.com",
    "analyst.ai",
}

_TIER_5_HOSTS = {
    "seekingalpha.com",
    "x.com",
    "twitter.com",
    "reddit.com",
    "stocktwits.com",
    "medium.com",
    "substack.com",
    "投資家.com",
    "5ch.net",
    "yahoo.co.jp",
    "finance.yahoo.com",
    "motleyfool.com",
    "fool.com",
    "investorplace.com",
    "wallstreetbets.com",
}

#: IR / company-controlled hosts and wire services carrying company releases.
#: A press release is a COMPANY_CLAIM regardless of which wire distributed it.
_COMPANY_WIRE_HOSTS = {
    "globenewswire.com",
    "prnewswire.com",
    "businesswire.com",
    "accesswire.com",
    "newsfilecorp.com",
    "prtimes.jp",
}


#: Hosts where a regulator/agency itself is the author of what is published
#: (requirement B3). Distinct from a statutory-filing or registry host below,
#: where the SITE is government/exchange-operated but the CONTENT is
#: issuer/sponsor-authored -- hosting location is never read as who speaks.
_REGULATOR_HOSTS = {
    "fda.gov",
    "accessdata.fda.gov",
    "ema.europa.eu",
    "pmda.go.jp",
    "mhlw.go.jp",
    "fsa.go.jp",
    "gov.uk",
    "mhra.gov.uk",
    "hc-sc.gc.ca",
    "federalregister.gov",
    "regulations.gov",
    "nih.gov",
    "nasa.gov",
    "defense.gov",
    "dod.gov",
    "dhs.gov",
    "sam.gov",
    "usaspending.gov",
    "fpds.gov",
    "gsa.gov",
    "nist.gov",
    "energy.gov",
    "osti.gov",
}

#: Hosts where the CONTENT is the issuer's own required statutory disclosure,
#: even though the exchange/regulator hosts or receives it.
_STATUTORY_FILING_HOSTS = {
    "sec.gov",
    "efts.sec.gov",
    "data.sec.gov",
    "nasdaq.com",
    "nyse.com",
    "jpx.co.jp",
    "release.tdnet.info",
    "disclosure.edinet-fsa.go.jp",
    "edinet-fsa.go.jp",
    "kanpou.npb.go.jp",
}

#: Government/clinical registries -- typically sponsor-authored entries, read
#: directly rather than via a search summary. ``nih.gov`` itself is NOT
#: here: an NIH grant record or agency page is government-authored (see
#: _REGULATOR_HOSTS), not a sponsor-authored registry entry -- only the
#: actual trial registry is.
_REGISTRY_HOSTS = {
    "clinicaltrials.gov",
    "clinicalregulatory.nih.gov",
}

#: More specific than ``nih.gov`` (see _REGULATOR_HOSTS) and must win by
#: longest-suffix match: PubMed/NCBI is peer-reviewed independent literature,
#: not an NIH agency communication, even though it sits under nih.gov.
_INDEPENDENT_OVERRIDE_HOSTS = {
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
}


def classify_authority(url: str, *, is_company_ir: bool = False) -> DocumentAuthority:
    """Classify WHO authored/issued a document (requirement B3).

    Longest-suffix matching, same discipline as :func:`classify_tier`, and
    deliberately a SEPARATE axis from it: a host being Tier 1 (high
    confidence/priority) says nothing about whether the regulator or the
    issuer is speaking at that host. ``is_company_ir`` lets a caller assert
    company authorship the host tables would not otherwise catch (e.g. the
    company's own website). Unknown hosts get ``UNKNOWN`` -- never an
    optimistic default.
    """
    host = _host(url)
    if not host:
        return DocumentAuthority.UNKNOWN
    if is_company_ir or _matches(host, _COMPANY_WIRE_HOSTS):
        return DocumentAuthority.COMPANY_IR
    candidates = (
        (_match_len(host, _REGULATOR_HOSTS), DocumentAuthority.REGULATOR),
        (_match_len(host, _STATUTORY_FILING_HOSTS), DocumentAuthority.STATUTORY_FILING),
        (_match_len(host, _REGISTRY_HOSTS), DocumentAuthority.REGISTRY),
        (_match_len(host, _INDEPENDENT_OVERRIDE_HOSTS), DocumentAuthority.INDEPENDENT),
    )
    length, authority = max(candidates, key=lambda pair: pair[0])
    if length:
        return authority
    # Any other host this system already recognises as a source at all (news,
    # trade press, peer-reviewed literature, broker research, social media)
    # is authored by neither the issuer nor a regulator.
    if classify_tier(url) is not SourceTier.UNKNOWN:
        return DocumentAuthority.INDEPENDENT
    return DocumentAuthority.UNKNOWN


def _host(url: str) -> str:
    """Hostname, lowercased, with a leading ``www.`` label removed.

    Note: ``lstrip("www.")`` would strip *characters*, turning ``www.wsj.com``
    into ``sj.com`` and silently misclassifying the source.
    """
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _match_len(host: str, table: set[str]) -> int:
    """Length of the longest entry in ``table`` that matches ``host``, else 0."""
    best = 0
    for entry in table:
        if host == entry or host.endswith("." + entry):
            best = max(best, len(entry))
    return best


def _matches(host: str, table: set[str]) -> bool:
    return _match_len(host, table) > 0


def classify_tier(url: str, *, is_company_ir: bool = False) -> SourceTier:
    """Classify a URL into the source hierarchy (requirement 2).

    Matching is by **longest suffix**, not by table order: ``pubmed.ncbi.nlm.nih.gov``
    is peer-reviewed literature (Tier 2) even though it sits under ``nih.gov``
    (Tier 1).  Unknown hosts get ``UNKNOWN`` -- never an optimistic default, since
    an unknown source cannot settle a material question.
    """
    host = _host(url)
    if not host:
        return SourceTier.UNKNOWN
    if is_company_ir or _matches(host, _COMPANY_WIRE_HOSTS):
        # Company IR / press release: a real, citable primary document, but its
        # *content* is a company claim.  Tier 2 by document, never independent.
        return SourceTier.TIER_2
    candidates = (
        (_match_len(host, _TIER_1_HOSTS), SourceTier.TIER_1),
        (_match_len(host, _TIER_2_HOSTS), SourceTier.TIER_2),
        (_match_len(host, _TIER_3_HOSTS), SourceTier.TIER_3),
        (_match_len(host, _TIER_4_HOSTS), SourceTier.TIER_4),
        (_match_len(host, _TIER_5_HOSTS), SourceTier.TIER_5),
    )
    length, tier = max(candidates, key=lambda pair: (pair[0], -pair[1].rank))
    return tier if length else SourceTier.UNKNOWN


def is_company_controlled(url: str, *, is_company_ir: bool = False) -> bool:
    """Whether the publisher is the company itself (or its wire)."""
    return is_company_ir or _matches(_host(url), _COMPANY_WIRE_HOSTS)


# --- syndication / reprint detection ---------------------------------------
_BOILERPLATE = re.compile(
    r"(?i)(forward[- ]looking statements.*$|about\s+\w+\s+(?:inc|corp|ltd).*$|"
    r"(?:source|出典)\s*[:：].*$|copyright.*$|©.*$)",
    re.S,
)
_WS = re.compile(r"\s+")


def _normalize_words(text: str) -> list[str]:
    cleaned = _BOILERPLATE.sub("", text or "")
    return _WS.sub(" ", cleaned).strip().lower().split()


def content_signature(text: str, *, head_words: int = 120) -> str:
    """Exact-copy signature of an article's substantive opening."""
    words = _normalize_words(text)
    if not words:
        return "UNKNOWN"
    return hashlib.sha256(" ".join(words[:head_words]).encode()).hexdigest()[:32]


def _shingles(text: str, n: int = 5, head_words: int = 200) -> set[str]:
    words = _normalize_words(text)[:head_words]
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def similarity(text_a: str, text_b: str) -> float:
    """Jaccard similarity over 5-word shingles."""
    a, b = _shingles(text_a), _shingles(text_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


#: Above this shingle overlap, two articles are the same story reprinted.
REPRINT_THRESHOLD = 0.6


def is_reprint(text_a: str, text_b: str, threshold: float = REPRINT_THRESHOLD) -> bool:
    """Whether two articles are the same text redistributed.

    Requirement 13: a wire story carried by five outlets is one source, not
    five.  Exact hashing is not enough -- outlets prepend their own headline and
    byline -- so near-duplicate detection is done by shingle overlap.
    """
    return similarity(text_a, text_b) >= threshold


def independent_source_count(
    sources: list[tuple[str, str, str]], threshold: float = REPRINT_THRESHOLD
) -> tuple[int, list[str]]:
    """Count genuinely independent sources.

    ``sources`` is a list of ``(source_id, url, body_text)``.  Sources are
    collapsed when they share a host or when their text is a near-duplicate of
    an already-counted source.  Returns ``(count, representative_source_ids)``.
    """
    seen_host: set[str] = set()
    kept: list[tuple[str, str]] = []  # (source_id, text)
    representatives: list[str] = []
    for source_id, url, text in sources:
        host = _host(url)
        if host and host in seen_host:
            continue
        if any(
            is_reprint(text, prior_text, threshold) for _, prior_text in kept if text and prior_text
        ):
            continue
        if host:
            seen_host.add(host)
        kept.append((source_id, text))
        representatives.append(source_id)
    return len(representatives), representatives
