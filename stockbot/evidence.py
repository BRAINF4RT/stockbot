"""Turn FreeSearcher's raw evidence into trust-ranked, citation-tagged sources, and verify
that an analyst's cited claims are actually supported by them.

Adapted from Daily-Learn: trust ranking (trust.js) and the lexical fact-check (pipeline.js).
Pure Python with no dependency on forecasting-tools, so it is fully unit-testable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from stockbot import trust

PER_SOURCE_CHARS = 1400
MAX_SOURCES = 14
RECENCY_HALF_LIFE_DAYS = 10.0


@dataclass
class Source:
    id: str
    kind: str          # news | web | wiki | market
    title: str
    url: str
    publisher: str
    date: datetime | None
    text: str
    trust: float


# --------------------------------------------------------------------------- building
_WEB_BLOCK = re.compile(r"^###\s*(?P<title>.*?)\s*\nSource:\s*(?P<url>\S+)[^\n]*\n(?P<body>.*)$", re.S)


def parse_web_blocks(blob: str) -> list[dict[str, str]]:
    """FreeSearcher's web evidence is markdown blocks joined by '---'. Split them back out."""
    out = []
    for block in re.split(r"\n---\n", blob or ""):
        m = _WEB_BLOCK.match(block.strip())
        if not m:
            continue
        body = re.sub(r"^\(query:[^\n]*\)\s*", "", m["body"].strip())
        out.append({"title": m["title"], "url": m["url"], "text": body.strip()})
    return out


def _aware(d: datetime | None) -> datetime | None:
    if d is not None and d.tzinfo is None:
        return d.replace(tzinfo=timezone.utc)
    return d


def recency_weight(date: datetime | None, now: datetime) -> float:
    """1.0 for today, halving every RECENCY_HALF_LIFE_DAYS. Undated items get a neutral 0.5."""
    date = _aware(date)
    if date is None:
        return 0.5
    age_days = max((now - date).total_seconds() / 86400, 0.0)
    return 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)


def _norm_title(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


def build_sources(evidence: Any, now: datetime | None = None, max_sources: int = MAX_SOURCES,
                  per_source_chars: int = PER_SOURCE_CHARS) -> list[Source]:
    """Rank by trust x recency, drop untrusted/duplicate items, tag S1..Sn.

    ``evidence`` is duck-typed to FreeSearcher's ``Evidence`` (news/markets/wiki/web).
    """
    now = _aware(now) or datetime.now(timezone.utc)
    raw: list[Source] = []

    def add(kind: str, title: str, url: str, publisher: str, date: datetime | None, text: str):
        score = trust.trust_score(url, publisher)
        if not trust.is_usable(score) or not (title or text).strip():
            return
        raw.append(Source("", kind, title.strip(), url, publisher, _aware(date),
                          (text or title).strip()[:per_source_chars], score))

    for n in getattr(evidence, "news", []):
        add("news", n.title, n.url, n.source, n.date, n.snippet or n.title)
    for w in getattr(evidence, "wiki", []):
        add("wiki", w.title, w.url, "Wikipedia", None, w.snippet)
    for m in getattr(evidence, "markets", []):
        p = f"{m.probability:.0%} Yes" if m.probability is not None else m.snippet
        add("market", m.title, m.url, m.source, None, f"Prediction market {m.source}: {p}")
    for blob in getattr(evidence, "web", []):
        for b in parse_web_blocks(blob):
            add("web", b["title"], b["url"], "", None, b["text"])

    seen: set[str] = set()
    unique: list[Source] = []
    for s in raw:
        key = _norm_title(s.title) or s.url
        url_key = re.sub(r"[?#].*$", "", s.url).rstrip("/").lower()
        if key in seen or (url_key and url_key in seen):
            continue
        seen.add(key)
        if url_key:
            seen.add(url_key)
        unique.append(s)

    unique.sort(key=lambda s: -(s.trust * (0.4 + 0.6 * recency_weight(s.date, now))))
    top = unique[:max_sources]
    for i, s in enumerate(top, 1):
        s.id = f"S{i}"
    return top


def format_block(sources: list[Source]) -> str:
    if not sources:
        return "NO_RESEARCH_AVAILABLE: no sufficiently trustworthy sources were found."
    lines = []
    for s in sources:
        when = s.date.strftime("%Y-%m-%d") if s.date else "date unknown"
        who = s.publisher or (s.url.split("/")[2] if s.url.count("/") >= 2 else "web")
        lines.append(f"[{s.id}] {s.kind} | {who} | {when} | reliability {s.trust:.2f}\n"
                     f"{s.title}\n{s.url}\n{s.text}")
    return "\n\n".join(lines)


# --------------------------------------------------------------------------- verification
_STOP = set("""a about after again all also an and any are as at be been before but by can could did do
does for from had has have he her his how if in into is it its more most not of on or our over she
should so some such than that the their them then there these they this those to up us was we were
what when where which while who will with would you your""".split())
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _stems(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOP}


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUM.findall(text)}


def claim_supported(claim: str, source: Source | None, min_overlap: float = 0.6) -> bool:
    """Lexical check: enough of the claim's words appear in the cited source, and every
    number the claim states appears there too. Catches invented figures and misattribution;
    it does not prove the claim true."""
    if source is None:
        return False
    words = _stems(claim)
    if len(words) < 2:
        return False
    haystack = f"{source.title} {source.text}"
    overlap = len(words & _stems(haystack)) / len(words)
    nums = _numbers(claim)
    return overlap >= min_overlap and nums <= _numbers(haystack)


def verify_claims(claims: list[dict[str, Any]], sources: list[Source]) -> tuple[int, int]:
    """Return (supported, total). Each claim is {"claim": str, "source": "S3"}."""
    by_id = {s.id: s for s in sources}
    supported = total = 0
    for c in claims or []:
        if not isinstance(c, dict):
            continue
        text = str(c.get("claim", "")).strip()
        if not text:
            continue
        total += 1
        m = re.search(r"S\d+", str(c.get("source", "")).upper())
        supported += claim_supported(text, by_id.get(m.group(0)) if m else None)
    return supported, total


def confidence_multiplier(supported: int, total: int) -> float:
    """How much of the LLM's lean to keep: 1.0 if every claim is grounded, 0.5 if none are
    (or it cited nothing). A forecast resting on unverifiable claims moves the position less."""
    grounded = supported / total if total else 0.0
    return 0.5 + 0.5 * grounded
