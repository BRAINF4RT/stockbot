"""Evidence gathering, using the vendored ``FreeSearcher`` machinery (stockbot/free_research).

The stock is framed as a forecasting question, exactly like the Metaculus bot. Evidence is
then trust-ranked with finance-specific source scores and tagged [S1]..[Sn] so the analyst's
claims can be verified (see evidence.py).

All LLM calls go through ``stockbot.llm``, which only allows free OpenRouter models.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from stockbot import evidence as ev
from stockbot import llm

logger = logging.getLogger(__name__)


@dataclass
class ResearchResult:
    text: str
    sources: list[ev.Source] = field(default_factory=list)


def make_question(ticker: str, name: str, price: float, horizon_days: int) -> str:
    return (
        f"Will {name} ({ticker}) stock close higher in {horizon_days} trading days "
        f"than its latest price of ${price:.2f}?"
    )


def build_searcher() -> Any:
    try:
        from stockbot.searcher import TrustedSearcher  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            f"Research dependencies missing ({exc}). Run: pip install -e . "
            "(or use --no-research for technical-only mode)."
        ) from exc

    async def free_llm(prompt: str) -> str:
        return await llm.generate(prompt, temperature=0.3, max_tokens=800)

    return TrustedSearcher(llm=free_llm, planner=free_llm, num_queries=3, sites_per_query=3)


async def research_ticker(ticker: str, name: str, price: float, horizon_days: int,
                          technical_note: str = "", searcher: Any | None = None) -> ResearchResult:
    searcher = searcher or build_searcher()
    sources = await searcher.research_sources(
        make_question(ticker, name, price, horizon_days),
        f"Resolves YES if {ticker}'s adjusted close {horizon_days} trading days from now "
        f"is above ${price:.2f}.",
        technical_note,
    )
    return ResearchResult(ev.format_block(sources), sources)
