"""The bot must not touch forecasting-tools, and the vendored research code must work under
a plain asyncio.run (the fork's nest_asyncio patching broke asyncio.wait_for on some setups)."""
import ast
import asyncio
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from stockbot import llm
from stockbot.free_research import sources as fs
from stockbot.free_research.searcher_base import FreeSearcher
from stockbot.searcher import TrustedSearcher

ROOT = Path(__file__).resolve().parent.parent


def test_no_file_imports_forecasting_tools():
    offenders = []
    for path in list((ROOT / "stockbot").rglob("*.py")) + list((ROOT / "tests").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            if any(n.split(".")[0] in ("forecasting_tools", "nest_asyncio") for n in names):
                offenders.append(f"{path.relative_to(ROOT)}: {names}")
    assert offenders == []


def test_importing_the_bot_loads_neither_forecasting_tools_nor_nest_asyncio():
    code = ("import sys, stockbot.cli, stockbot.searcher, stockbot.free_research.web_scraper;"
            "bad=[m for m in sys.modules if m.split('.')[0] in ('forecasting_tools','nest_asyncio')];"
            "print('BAD', bad) if bad else print('CLEAN')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert out.stdout.strip() == "CLEAN", out.stdout + out.stderr


NOW = datetime.now(timezone.utc)
WEB = ("### Guidance\nSource: https://www.sec.gov/f\n(query: 'q')\n\nRevenue outlook $54 billion\n"
       "---\n### Chatter\nSource: https://reddit.com/r/x\n(query: 'q')\n\nlol")


@pytest.fixture
def stub_sources(monkeypatch):
    item = fs.SourceItem
    monkeypatch.setattr(fs, "google_news_rss", lambda *a, **k: [
        item("news", "Reuters", "Nvidia raises outlook", "https://news.google.com/a", NOW - timedelta(days=1), "Up to $54 billion"),
        item("news", "StockTwits", "NVDA moon", "https://news.google.com/b", NOW, "rocket")])
    monkeypatch.setattr(fs, "gdelt_articles", lambda *a, **k: [
        item("news", "bloomberg.com", "Chip demand strong", "https://www.bloomberg.com/x", NOW - timedelta(days=3), "Demand up")])
    monkeypatch.setattr(fs, "ddgs_news", lambda *a, **k: [])
    monkeypatch.setattr(fs, "wikipedia_extracts", lambda *a, **k: [
        item("wiki", "Wikipedia", "Nvidia", "https://en.wikipedia.org/wiki/Nvidia", None, "GPU maker")])
    monkeypatch.setattr(fs, "polymarket_markets", lambda *a, **k: [])
    monkeypatch.setattr(fs, "manifold_markets", lambda *a, **k: [])


async def _planner(prompt):
    return "1. nvidia earnings\n2. nvidia guidance\n3. nvidia news"


def test_gather_has_no_errors_under_plain_asyncio_run(stub_sources):
    """Regression: every source used to fail with 'Timeout should be used inside a task'."""
    s = FreeSearcher(llm=_planner, planner=_planner, num_queries=3,
                     web_search_fn=lambda q, n, scrape, rel: WEB)

    async def go():
        queries = await s.plan_queries("Will NVDA close higher?")
        return await s.gather("Will NVDA close higher?", queries)

    evidence = asyncio.run(go())
    assert evidence.errors == [], evidence.errors
    assert evidence.news and evidence.wiki and evidence.web


def test_trusted_searcher_end_to_end_drops_chatter_and_tags_sources(stub_sources):
    s = TrustedSearcher(llm=_planner, planner=_planner, web_search_fn=lambda q, n, sc, r: WEB)
    srcs = asyncio.run(s.research_sources("Will NVDA close higher?"))
    assert [x.id for x in srcs] == [f"S{i}" for i in range(1, len(srcs) + 1)]
    assert any("sec.gov" in x.url for x in srcs)
    assert not any("reddit" in x.url or x.publisher == "StockTwits" for x in srcs)


def test_planner_defaults_to_the_free_only_client(monkeypatch, stub_sources):
    seen = []

    async def fake_generate(prompt, **kw):
        seen.append(kw)
        return "1. a query\n2. another query"

    monkeypatch.setattr(llm, "generate", fake_generate)
    s = FreeSearcher(web_search_fn=lambda q, n, sc, r: WEB)  # no llm/planner given
    queries = asyncio.run(s.plan_queries("Will NVDA close higher?"))
    assert seen, "planner did not go through stockbot.llm"
    assert queries[0] == "Will NVDA close higher?" and len(queries) >= 2


def test_string_model_specs_are_rejected_not_silently_sent_to_a_paid_default():
    s = FreeSearcher(llm="openrouter/openai/gpt-4.1-mini", web_search_fn=lambda *a: "")
    with pytest.raises(TypeError):
        asyncio.run(s._call_llm("hi"))
