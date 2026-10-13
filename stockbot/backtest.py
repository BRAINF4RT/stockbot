"""Technical-only backtest with costs. The LLM/news layer cannot be backtested honestly:
free news sources are not point-in-time, so any 'backtest' of it would leak the future."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockbot.config import Settings
from stockbot.risk import cap_weights
from stockbot.signals import quant_score
from stockbot.strategy import build_targets, Signal


@dataclass
class Result:
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    turnover_per_year: float
    bench_total_return: float
    bench_max_drawdown: float
    equity: pd.Series


def _stats(rets: pd.Series) -> tuple[float, float, float, float]:
    eq = (1 + rets).cumprod()
    years = max(len(rets) / 252, 1e-9)
    cagr = eq.iloc[-1] ** (1 / years) - 1
    sharpe = rets.mean() / rets.std() * math.sqrt(252) if rets.std() > 0 else 0.0
    dd = (eq / eq.cummax() - 1).min()
    return float(eq.iloc[-1] - 1), float(cagr), float(sharpe), float(dd)


def run_backtest(history: dict[str, pd.DataFrame], cfg: Settings,
                 rebalance_every: int = 5, cost_bps: float = 5.0) -> Result:
    closes = pd.DataFrame({t: df["close"] for t, df in history.items()}).dropna()
    rets = closes.pct_change().fillna(0.0)
    n = len(closes)
    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    cur = pd.Series(0.0, index=closes.columns)
    for i in range(210, n):
        if (i - 210) % rebalance_every == 0:
            sigs = []
            for t in closes.columns:
                q = quant_score(closes[t].iloc[: i + 1])  # data up to and including day i
                sigs.append(Signal(t, float(closes[t].iloc[i]), q.score, q.vol))
            held = {t for t in closes.columns if cur[t] > 0}
            cur = pd.Series(cap_weights(build_targets(sigs, held, cfg), cfg)).reindex(
                closes.columns).fillna(0.0)
        weights.iloc[i] = cur.values

    # decide at close of day i, earn the return of day i+1 (no lookahead)
    held_w = weights.shift(1).fillna(0.0)
    turnover = held_w.diff().abs().sum(axis=1).fillna(0.0)
    port = (held_w * rets).sum(axis=1) - turnover * cost_bps / 10_000
    port = port.iloc[210:]
    bench = rets.mean(axis=1).iloc[210:]  # equal-weight buy & hold of the same universe

    tr, cagr, sh, dd = _stats(port)
    btr, _, _, bdd = _stats(bench)
    return Result(tr, cagr, sh, dd, float(turnover.iloc[210:].sum() / (len(port) / 252)),
                  btr, bdd, (1 + port).cumprod())
