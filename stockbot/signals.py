"""Transparent technical score in [-1, 1]. A heuristic, not a proven edge."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def rsi(close: pd.Series, n: int = 14) -> float:
    delta = close.diff().dropna()
    up = delta.clip(lower=0).rolling(n).mean().iloc[-1]
    down = (-delta.clip(upper=0)).rolling(n).mean().iloc[-1]
    if down == 0:
        return 100.0
    return float(100 - 100 / (1 + up / down))


def annual_vol(close: pd.Series, n: int = 60) -> float:
    r = np.log(close).diff().dropna().tail(n)
    return float(r.std() * math.sqrt(252)) if len(r) > 5 else 0.3


@dataclass
class Quant:
    score: float
    momentum: float
    above_200: bool
    golden: bool
    rsi: float
    vol: float


def quant_score(close: pd.Series) -> Quant:
    """Score using only data up to and including the last bar (no lookahead)."""
    if len(close) < 210:
        raise ValueError("need >= 210 bars")
    px = float(close.iloc[-1])
    sma50 = float(close.tail(50).mean())
    sma200 = float(close.tail(200).mean())
    mom = float(close.iloc[-22] / close.iloc[-127] - 1)  # 6-1 month momentum
    above = px > sma200
    golden = sma50 > sma200
    r = rsi(close)

    score = (
        0.5 * math.tanh(mom / 0.25)
        + 0.3 * (1 if above else -1)
        + 0.2 * (1 if golden else -1)
    )
    if r > 85:  # parabolic: don't chase
        score = min(score, 0.3)
    return Quant(max(-1.0, min(1.0, score)), mom, above, golden, r, annual_vol(close))
