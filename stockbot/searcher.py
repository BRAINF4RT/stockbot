"""FreeSearcher subclass that returns trust-ranked, citation-tagged sources.

Reuses the vendored (standalone) query planner and concurrent multi-source ``gather()`` (Google News,
GDELT, DDGS news, Wikipedia, scraped web pages, prediction markets), but skips the fork's
LLM condensing step: the analyst reads the tagged sources directly, which saves a rate-limited
free-model call per ticker and keeps [S#] tags intact for claim verification.
"""
from __future__ import annotations

from stockbot.free_research.searcher_base import FreeSearcher

from stockbot import evidence as ev


class TrustedSearcher(FreeSearcher):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("max_news", 30)  # wider pool: the trust filter does the narrowing
        super().__init__(*args, **kwargs)

    async def research_sources(self, question_text: str, resolution_criteria: str = "",
                               background: str = "") -> list[ev.Source]:
        queries = await self.plan_queries(question_text, resolution_criteria, background)
        evidence = await self.gather(question_text, queries)
        return ev.build_sources(evidence)
