"""Free price data: yfinance first, Stooq CSV as a fallback."""
from __future__ import annotations

import io
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _from_yfinance(ticker: str, days: int) -> pd.DataFrame:
    import yfinance as yf  # noqa: PLC0415

    df = yf.Ticker(ticker).history(period=f"{max(days, 30)}d", auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.lower)[["close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df


def _from_stooq(ticker: str, days: int) -> pd.DataFrame:
    import httpx  # noqa: PLC0415

    symbol = f"{ticker.lower()}.us"
    r = httpx.get(f"https://stooq.com/q/d/l/?s={symbol}&i=d", timeout=20, follow_redirects=True)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty or "Close" not in df:
        return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").rename(columns=str.lower)[["close", "volume"]]
    return df.tail(days)


def get_history(ticker: str, days: int = 400) -> pd.DataFrame:
    """Daily close/volume, oldest first. Empty frame if every source fails."""
    for name, fn in (("yfinance", _from_yfinance), ("stooq", _from_stooq)):
        try:
            df = fn(ticker, days)
            if not df.empty:
                return df.dropna(subset=["close"]).sort_index()
        except Exception as exc:  # noqa: BLE001 - sources fail independently
            logger.warning("%s failed for %s: %s", name, ticker, exc)
    return pd.DataFrame()


def get_universe_history(tickers: list[str], days: int = 400) -> dict[str, pd.DataFrame]:
    out = {}
    for t in tickers:
        df = get_history(t, days)
        if len(df) >= 210:  # need enough bars for the 200d average
            out[t] = df
        else:
            logger.warning("Skipping %s: only %d bars", t, len(df))
    return out
