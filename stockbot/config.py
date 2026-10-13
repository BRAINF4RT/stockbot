"""Settings, read from environment variables (see .env.template)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_UNIVERSE = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN",
    "GOOGL", "META", "JPM", "XOM", "UNH", "COST",
]

COMPANY_NAMES = {
    "SPY": "SPDR S&P 500 ETF", "QQQ": "Invesco QQQ Nasdaq-100 ETF",
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "AMZN": "Amazon",
    "GOOGL": "Alphabet", "META": "Meta Platforms", "JPM": "JPMorgan Chase",
    "XOM": "Exxon Mobil", "UNH": "UnitedHealth", "COST": "Costco",
}


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    universe: list[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))

    # --- forecasting layer -------------------------------------------------
    horizon_days: int = 5            # "will it close higher in N trading days?"
    llm_samples: int = 1             # free models are rate limited; keep low
    llm_shrink: float = 0.5          # pull LLM probability toward 0.5 (LLMs are overconfident)
    w_llm: float = 0.4               # weight of LLM view in final score
    w_quant: float = 0.6             # weight of technical score in final score

    # --- strategy ----------------------------------------------------------
    enter_score: float = 0.25        # open a position above this score
    exit_score: float = 0.0          # close a held position below this (hysteresis)
    target_vol: float = 0.25         # annualised vol a position is sized to

    # --- hard risk limits (never overridden by the model) -------------------
    max_position_pct: float = 0.10   # per-name cap, fraction of equity
    max_gross_pct: float = 0.70      # total invested cap
    stop_loss_pct: float = 0.08      # exit if down this much from entry
    max_daily_loss_pct: float = 0.03 # kill switch: no new buys if equity is down this much today
    min_price: float = 5.0
    min_trade_value: float = 50.0

    # --- infrastructure ----------------------------------------------------
    alpaca_key: str = ""
    alpaca_secret: str = ""
    sim_start_cash: float = 10_000.0
    state_dir: str = "state"

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        uni = os.getenv("UNIVERSE", "").replace(" ", "")
        if uni:
            s.universe = [t.upper() for t in uni.split(",") if t]
        s.horizon_days = int(_f("HORIZON_DAYS", s.horizon_days))
        s.llm_samples = int(_f("LLM_SAMPLES", s.llm_samples))
        s.max_position_pct = _f("MAX_POSITION_PCT", s.max_position_pct)
        s.max_gross_pct = _f("MAX_GROSS_PCT", s.max_gross_pct)
        s.stop_loss_pct = _f("STOP_LOSS_PCT", s.stop_loss_pct)
        s.max_daily_loss_pct = _f("MAX_DAILY_LOSS_PCT", s.max_daily_loss_pct)
        s.sim_start_cash = _f("SIM_START_CASH", s.sim_start_cash)
        s.alpaca_key = os.getenv("ALPACA_KEY_ID", "")
        s.alpaca_secret = os.getenv("ALPACA_SECRET_KEY", "")
        s.state_dir = os.getenv("STATE_DIR", s.state_dir)
        return s
