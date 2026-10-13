"""LLM probability forecast for 'will this stock be higher in N days?', with cited,
verified claims (adapted from Daily-Learn's citation tags + lexical fact-check)."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import statistics
from dataclasses import dataclass
from typing import Any

from stockbot import evidence as ev
from stockbot import llm

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are a calibrated superforecaster. Short-horizon stock moves are close to a coin "
    "flip, so most forecasts should sit between 0.40 and 0.60. Only move far from 0.5 for "
    "strong, specific, recent evidence (earnings, guidance, regulatory action, macro shocks). "
    "Base rate: roughly 53% of stocks close higher over any 5-day window. You never invent "
    "facts, numbers or sources."
)

PROMPT = """\
Question: {question}

Technical context (not a citable source): {technical}

EVIDENCE. Each source has a tag like [S3] and a reliability score; prefer higher-reliability
sources and recent dates. Absence of news is not evidence.
{research}

Think step by step about base rates, then the strongest evidence for and against.
Every factual claim you rely on must be listed with the tag of the source that states it,
using only facts that appear in that source. If the evidence is thin, say so and stay near 0.5.
Finish with ONLY a JSON object, e.g.
{{"probability": 0.54, "reasoning": "one or two sentences",
 "claims": [{{"claim": "short factual statement", "source": "S2"}}]}}"""


@dataclass
class Forecast:
    p_up: float
    reasoning: str
    raw_p: float
    supported: int = 0
    total: int = 0
    multiplier: float = 1.0


def _extract_json(text: str) -> dict[str, Any] | None:
    """Last JSON object in the text that has a 'probability' key (handles nested claims)."""
    dec = json.JSONDecoder()
    for m in reversed(list(re.finditer(r"\{", text))):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except ValueError:
            continue
        if isinstance(obj, dict) and "probability" in obj:
            return obj
    return None


def _prob(value: Any) -> float | None:
    try:
        p = float(value)
    except (TypeError, ValueError):
        return None
    if p > 1:
        p /= 100
    return p if 0 <= p <= 1 else None


def parse_forecast(text: str) -> tuple[float, str] | None:
    """Pull the probability from model output; tolerate prose and percentages."""
    obj = _extract_json(text)
    if obj is not None:
        p = _prob(obj.get("probability"))
        if p is not None:
            return p, str(obj.get("reasoning", ""))[:500]
    m = re.search(r"probability\D{0,10}(\d{1,3}(?:\.\d+)?)\s*%", text, flags=re.I)
    if m and 0 <= float(m.group(1)) <= 100:
        return float(m.group(1)) / 100, ""
    return None


def parse_claims(text: str) -> list[dict[str, Any]]:
    obj = _extract_json(text)
    claims = obj.get("claims") if obj else None
    return [c for c in claims if isinstance(c, dict)] if isinstance(claims, list) else []


def shrink(p: float, factor: float) -> float:
    """Pull toward 0.5 and clip: free LLMs are overconfident on markets."""
    p = 0.5 + factor * (p - 0.5)
    return min(0.65, max(0.35, p))


async def forecast(question: str, technical: str, research: str, *,
                   sources: list[ev.Source] | None = None, samples: int = 1,
                   shrink_factor: float = 0.5) -> Forecast | None:
    """``sources`` enables claim verification; without it the LLM's lean is used as-is."""
    prompt = PROMPT.format(question=question, technical=technical, research=research[:24000])
    draws: list[tuple[float, str]] = []
    supported = total = 0
    for _ in range(max(1, samples)):
        try:
            text = await llm.generate(prompt, system=SYSTEM, temperature=0.4, max_tokens=1800)
        except Exception as exc:  # noqa: BLE001
            logger.warning("forecast call failed: %s", exc)
            continue
        parsed = parse_forecast(text)
        if parsed:
            draws.append(parsed)
            if sources is not None:
                s, t = ev.verify_claims(parse_claims(text), sources)
                supported, total = supported + s, total + t
        await asyncio.sleep(1)
    if not draws:
        return None
    raw = statistics.median(p for p, _ in draws)
    mult = ev.confidence_multiplier(supported, total) if sources is not None else 1.0
    p_up = 0.5 + mult * (shrink(raw, shrink_factor) - 0.5)
    return Forecast(p_up, draws[-1][1], raw, supported, total, mult)
