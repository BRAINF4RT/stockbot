"""Combine the technical score and LLM probability into target portfolio weights."""
from __future__ import annotations

from dataclasses import dataclass, field

from stockbot.config import Settings


@dataclass
class Signal:
    ticker: str
    price: float
    quant: float
    vol: float
    p_up: float | None = None
    score: float = 0.0
    weight: float = 0.0
    notes: list[str] = field(default_factory=list)


def combined_score(quant: float, p_up: float | None, cfg: Settings) -> float:
    if p_up is None:  # LLM unavailable -> technical-only (same as the backtest)
        return quant
    llm_view = 2 * p_up - 1
    return cfg.w_quant * quant + cfg.w_llm * llm_view


def target_weight(score: float, vol: float, held: bool, cfg: Settings) -> float:
    """Long-only, volatility-scaled. Hysteresis stops buy/sell churn near the threshold."""
    threshold = cfg.exit_score if held else cfg.enter_score
    if score <= threshold:
        return 0.0
    conviction = min(1.0, score)
    vol_scalar = min(1.0, cfg.target_vol / max(vol, 0.05))
    return cfg.max_position_pct * conviction * vol_scalar


def build_targets(signals: list[Signal], held: set[str], cfg: Settings) -> dict[str, float]:
    for s in signals:
        s.score = combined_score(s.quant, s.p_up, cfg)
        s.weight = target_weight(s.score, s.vol, s.ticker in held, cfg)
    return {s.ticker: s.weight for s in signals}
