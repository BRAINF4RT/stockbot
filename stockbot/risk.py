"""Hard risk limits. These run after the model and can only reduce risk."""
from __future__ import annotations

import math
from dataclasses import dataclass

from stockbot.broker import Order, Position
from stockbot.config import Settings


def cap_weights(weights: dict[str, float], cfg: Settings) -> dict[str, float]:
    """Per-name cap, then scale everything down if gross exposure is too high."""
    capped = {t: min(max(w, 0.0), cfg.max_position_pct) for t, w in weights.items()}
    gross = sum(capped.values())
    if gross > cfg.max_gross_pct:
        k = cfg.max_gross_pct / gross
        capped = {t: w * k for t, w in capped.items()}
    return capped


def stop_loss_hits(positions: dict[str, Position], prices: dict[str, float],
                   cfg: Settings) -> set[str]:
    hits = set()
    for t, pos in positions.items():
        px = prices.get(t)
        if px and pos.avg_price > 0 and px <= pos.avg_price * (1 - cfg.stop_loss_pct):
            hits.add(t)
    return hits


def kill_switch(equity: float, last_equity: float, cfg: Settings) -> bool:
    """True when today's loss exceeds the limit: no new buys, only exits."""
    return last_equity > 0 and (equity / last_equity - 1) <= -cfg.max_daily_loss_pct


def make_orders(weights: dict[str, float], positions: dict[str, Position],
                prices: dict[str, float], equity: float, cash: float,
                cfg: Settings, allow_buys: bool = True) -> list[Order]:
    """Turn target weights into whole-share orders. Sells first, buys limited to cash."""
    sells: list[Order] = []
    buys: list[Order] = []
    for t in set(weights) | set(positions):
        px = prices.get(t)
        if not px or px < cfg.min_price and t not in positions:
            continue
        have = positions[t].qty if t in positions else 0.0
        want = math.floor(weights.get(t, 0.0) * equity / px)
        diff = want - have
        if abs(diff) * px < cfg.min_trade_value and want != 0:
            continue
        if diff < 0:
            sells.append(Order("sell", t, min(have, abs(diff))))
        elif diff > 0 and allow_buys:
            buys.append(Order("buy", t, diff))
    # fund buys from available cash plus sale proceeds, largest conviction first
    budget = cash + sum(o.qty * prices[o.ticker] for o in sells)
    funded: list[Order] = []
    for o in sorted(buys, key=lambda o: -weights.get(o.ticker, 0)):
        cost = o.qty * prices[o.ticker]
        if cost <= budget:
            funded.append(o)
            budget -= cost
    return sells + funded
