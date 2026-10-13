from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest

from stockbot import analyst, evidence as ev, llm, trust

NOW = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)


def news(title, source, url, days_ago=1, snippet=""):
    return NS(kind="news", source=source, title=title, url=url,
              date=NOW - timedelta(days=days_ago), snippet=snippet)


# ---- trust ---------------------------------------------------------------
def test_primary_and_wire_sources_outrank_blogs():
    assert trust.trust_score("https://www.sec.gov/ix?doc=abc") > 0.95
    assert trust.trust_score("https://www.reuters.com/x") > trust.trust_score("https://random-blog.com/x")
    assert trust.trust_score("https://www.reuters.com/x") > trust.trust_score("https://seekingalpha.com/x")


def test_chatter_is_unusable():
    for url in ("https://stocktwits.com/symbol/NVDA", "https://www.reddit.com/r/wsb/1", "https://x.com/a"):
        assert not trust.is_usable(trust.trust_score(url))


def test_publisher_name_used_when_url_is_a_redirect():
    # Google News links are redirects; the publisher name must carry the signal.
    assert trust.trust_score("https://news.google.com/rss/articles/CBMi", "Reuters") == pytest.approx(0.9)
    assert not trust.is_usable(trust.trust_score("https://news.google.com/rss/articles/x", "StockTwits"))


def test_investor_relations_hosts_count_as_primary():
    assert trust.trust_score("https://investor.nvidia.com/news/1") >= 0.85
    assert trust.trust_score("https://ir.example.com/x") >= 0.85


def test_unknown_domain_gets_default():
    assert trust.trust_score("https://some-site.example/x") == trust.DEFAULT_TRUST


# ---- building sources -----------------------------------------------------
WEB = ("### Nvidia raises guidance\nSource: https://www.reuters.com/nvda-guide\n"
       "(query: 'x'; backend: google; extraction: trafilatura; relevance: 3.00)\n\n"
       "Nvidia raised its quarterly revenue outlook to $54 billion on strong data center demand.\n"
       "\n---\n### Meme chatter\nSource: https://www.reddit.com/r/wsb/1\n(query: 'x')\n\nTo the moon")


def test_parse_web_blocks():
    blocks = ev.parse_web_blocks(WEB)
    assert [b["url"] for b in blocks] == ["https://www.reuters.com/nvda-guide", "https://www.reddit.com/r/wsb/1"]
    assert blocks[0]["text"].startswith("Nvidia raised")
    assert "query:" not in blocks[0]["text"]


def test_build_sources_filters_ranks_dedupes_and_tags():
    e = NS(
        news=[news("Nvidia raises guidance", "Reuters", "https://news.google.com/a", 1, "Reuters copy"),
              news("Nvidia raises guidance", "Benzinga", "https://news.google.com/b", 0, "dup title"),
              news("Old but trusted", "Reuters", "https://news.google.com/c", 60, "stale"),
              news("Meme thread", "StockTwits", "https://news.google.com/d", 0, "moon")],
        wiki=[NS(title="Nvidia", url="https://en.wikipedia.org/wiki/Nvidia", snippet="GPU maker")],
        markets=[], web=[WEB])
    out = ev.build_sources(e, now=NOW)
    titles = [s.title for s in out]
    assert "Meme thread" not in titles and "Meme chatter" not in titles
    assert not any("reddit" in s.url for s in out)                    # chatter dropped
    assert titles.count("Nvidia raises guidance") == 1          # same story from news + web collapsed
    assert [s.id for s in out] == [f"S{i}" for i in range(1, len(out) + 1)]
    fresh = next(s for s in out if s.title == "Nvidia raises guidance" and s.publisher == "Reuters")
    stale = next(s for s in out if s.title == "Old but trusted")
    assert out.index(fresh) < out.index(stale)                         # recency matters at equal trust


def test_recency_halves_every_half_life():
    assert ev.recency_weight(NOW, NOW) == pytest.approx(1.0)
    assert ev.recency_weight(NOW - timedelta(days=ev.RECENCY_HALF_LIFE_DAYS), NOW) == pytest.approx(0.5)
    assert ev.recency_weight(None, NOW) == 0.5


def test_empty_evidence_is_reported_not_silent():
    assert "NO_RESEARCH_AVAILABLE" in ev.format_block([])


# ---- claim verification -----------------------------------------------------
def src(i, text, title=""):
    return ev.Source(f"S{i}", "news", title, f"https://x/{i}", "Reuters", NOW, text, 0.9)


SOURCES = [src(1, "Nvidia raised its quarterly revenue outlook to $54 billion on strong data center demand."),
           src(2, "Apple announced a new iPhone with a faster camera sensor.")]


def test_supported_claim_passes():
    assert ev.claim_supported("Nvidia raised its revenue outlook to $54 billion", SOURCES[0])


def test_invented_number_fails():
    assert not ev.claim_supported("Nvidia raised its revenue outlook to $64 billion", SOURCES[0])


def test_wrong_source_fails():
    assert not ev.claim_supported("Nvidia raised its revenue outlook to $54 billion", SOURCES[1])


def test_missing_source_and_trivial_claim_fail():
    assert not ev.claim_supported("Nvidia revenue outlook", None)
    assert not ev.claim_supported("Up", SOURCES[0])


def test_verify_claims_counts_and_tolerates_junk():
    claims = [{"claim": "Nvidia raised its revenue outlook to $54 billion", "source": "S1"},
              {"claim": "Nvidia raised its revenue outlook to $99 billion", "source": "[S1]"},
              {"claim": "Apple announced a new iPhone", "source": "s2"},
              {"claim": "Nothing cited", "source": ""}, "garbage", {"claim": ""}]
    assert ev.verify_claims(claims, SOURCES) == (2, 4)


def test_confidence_multiplier_bounds():
    assert ev.confidence_multiplier(3, 3) == 1.0
    assert ev.confidence_multiplier(0, 3) == 0.5
    assert ev.confidence_multiplier(0, 0) == 0.5   # cited nothing -> half the lean
    assert ev.confidence_multiplier(1, 2) == 0.75


# ---- analyst integration ------------------------------------------------------
def test_parse_nested_claims_json():
    text = ('reasoning...\n{"probability": 0.6, "reasoning": "r", '
            '"claims": [{"claim": "a b c", "source": "S1"}]}')
    assert analyst.parse_forecast(text)[0] == pytest.approx(0.6)
    assert analyst.parse_claims(text) == [{"claim": "a b c", "source": "S1"}]


async def _run(monkeypatch, reply, sources):
    async def fake(*a, **k):
        return reply
    monkeypatch.setattr(llm, "generate", fake)
    return await analyst.forecast("q", "t", "r", sources=sources, samples=1, shrink_factor=1.0)


async def test_grounded_forecast_keeps_full_lean(monkeypatch):
    reply = ('{"probability": 0.62, "reasoning": "x", "claims": '
             '[{"claim": "Nvidia raised its revenue outlook to $54 billion", "source": "S1"}]}')
    fc = await _run(monkeypatch, reply, SOURCES)
    assert (fc.supported, fc.total, fc.multiplier) == (1, 1, 1.0)
    assert fc.p_up == pytest.approx(0.62)


async def test_fabricated_claims_halve_the_lean(monkeypatch):
    reply = ('{"probability": 0.62, "reasoning": "x", "claims": '
             '[{"claim": "Nvidia raised its revenue outlook to $99 billion", "source": "S1"}]}')
    fc = await _run(monkeypatch, reply, SOURCES)
    assert fc.multiplier == 0.5 and fc.p_up == pytest.approx(0.56)


async def test_no_sources_argument_skips_verification(monkeypatch):
    async def fake(*a, **k):
        return '{"probability": 0.6}'
    monkeypatch.setattr(llm, "generate", fake)
    fc = await analyst.forecast("q", "t", "r", samples=1, shrink_factor=1.0)
    assert fc.multiplier == 1.0 and fc.p_up == pytest.approx(0.6)
