"""Finance-specific source reliability, adapted from Daily-Learn's trust.js.

A transparent rule of thumb for ranking evidence, not a fact-check. Edit freely.
Scores are in [0, 1]; anything under MIN_TRUST is dropped from the evidence.
"""
from __future__ import annotations

from urllib.parse import urlparse

DEFAULT_TRUST = 0.5
MIN_TRUST = 0.3

# Matched against the hostname (exact or as a parent domain).
HOST_TRUST: dict[str, float] = {
    # primary / official
    "sec.gov": 0.97, "federalreserve.gov": 0.96, "treasury.gov": 0.95, "bls.gov": 0.95,
    "bea.gov": 0.95, "census.gov": 0.92, "ftc.gov": 0.9, "justice.gov": 0.9,
    "fda.gov": 0.92, "ecb.europa.eu": 0.93, "imf.org": 0.9, "worldbank.org": 0.88,
    ".gov": 0.88, ".edu": 0.8,
    # wires and top-tier financial press
    "reuters.com": 0.9, "apnews.com": 0.88, "bloomberg.com": 0.87, "wsj.com": 0.87,
    "ft.com": 0.87, "economist.com": 0.8, "barrons.com": 0.8, "cnbc.com": 0.78,
    "nytimes.com": 0.75, "bbc.com": 0.75, "bbc.co.uk": 0.75, "theguardian.com": 0.72,
    "npr.org": 0.75, "marketwatch.com": 0.72, "investors.com": 0.7, "nasdaq.com": 0.7,
    "finance.yahoo.com": 0.68, "morningstar.com": 0.75, "spglobal.com": 0.8,
    "wikipedia.org": 0.65,
    # company press releases: primary, but promotional
    "businesswire.com": 0.72, "prnewswire.com": 0.72, "globenewswire.com": 0.72,
    # opinion / aggregators
    "seekingalpha.com": 0.5, "fool.com": 0.5, "benzinga.com": 0.52, "investopedia.com": 0.6,
    "zacks.com": 0.5, "investorplace.com": 0.42, "thestreet.com": 0.5,
}

# Matched (substring, lowercase) against the publisher name. Needed because Google News and
# GDELT often give redirect URLs, so the hostname says nothing about who published it.
NAME_TRUST: dict[str, float] = {
    "reuters": 0.9, "associated press": 0.88, "bloomberg": 0.87, "wall street journal": 0.87,
    "financial times": 0.87, "barron": 0.8, "cnbc": 0.78, "the economist": 0.8,
    "marketwatch": 0.72, "investor's business daily": 0.7, "yahoo finance": 0.68,
    "business wire": 0.72, "pr newswire": 0.72, "globenewswire": 0.72, "morningstar": 0.75,
    "new york times": 0.75, "bbc": 0.75, "guardian": 0.72, "nasdaq": 0.7,
    "seeking alpha": 0.5, "motley fool": 0.5, "benzinga": 0.52, "zacks": 0.5,
}

# Dropped: unmoderated chatter and pump-prone sources. Overrides any other match.
LOW_QUALITY = [
    "stocktwits.com", "reddit.com", "x.com", "twitter.com", "facebook.com", "tiktok.com",
    "t.me", "telegram.org", "pinterest.com", "quora.com", "medium.com", "substack.com",
    "tipranks.com", "pennystocks.com", "scribd.com", "slideshare.net",
]
LOW_QUALITY_NAMES = ["stocktwits", "reddit", "tipranks", "penny stock"]
LOW_SCORE = 0.2


def host_of(url: str) -> str:
    try:
        h = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    return h[4:] if h.startswith("www.") else h


def _match(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain) or (
        domain.startswith(".") and host.endswith(domain))


def trust_score(url: str = "", name: str = "") -> float:
    host = host_of(url)
    lname = (name or "").lower()
    if host and any(_match(host, d) for d in LOW_QUALITY):
        return LOW_SCORE
    if any(n in lname for n in LOW_QUALITY_NAMES):
        return LOW_SCORE
    best = 0.0
    if host:
        best = max((s for d, s in HOST_TRUST.items() if _match(host, d)), default=0.0)
        # company investor-relations pages: a primary source for that company's own news
        if host.startswith(("investor.", "investors.", "ir.")):
            best = max(best, 0.85)
    if lname:
        best = max(best, max((s for n, s in NAME_TRUST.items() if n in lname), default=0.0))
    return best or DEFAULT_TRUST


def is_usable(score: float) -> bool:
    return score >= MIN_TRUST
