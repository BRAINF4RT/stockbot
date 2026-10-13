"""Brokers. Only simulated and Alpaca *paper* trading exist; there is no live path."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)

ALPACA_PAPER_URL = "https://paper-api.alpaca.markets/v2"  # deliberately hard-coded


@dataclass
class Position:
    qty: float
    avg_price: float


@dataclass
class Order:
    side: str  # "buy" | "sell"
    ticker: str
    qty: float


@dataclass
class Account:
    equity: float
    cash: float
    last_equity: float  # equity at previous close, for the daily-loss kill switch


class Broker(Protocol):
    name: str
    def account(self, prices: dict[str, float]) -> Account: ...
    def positions(self) -> dict[str, Position]: ...
    def submit(self, order: Order, price: float) -> None: ...
    def market_open(self) -> bool: ...


class SimBroker:
    """Offline paper broker persisted to JSON. Fills at the given price plus slippage."""
    name = "sim"

    def __init__(self, state_path: str | Path, start_cash: float = 10_000.0,
                 slippage_bps: float = 5.0) -> None:
        self.path = Path(state_path)
        self.slip = slippage_bps / 10_000
        if self.path.exists():
            s = json.loads(self.path.read_text())
        else:
            s = {"cash": start_cash, "positions": {}, "last_equity": start_cash, "day": ""}
        self.cash: float = s["cash"]
        self._pos = {t: Position(**p) for t, p in s["positions"].items()}
        self.last_equity: float = s["last_equity"]
        self.day: str = s["day"]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "cash": self.cash, "last_equity": self.last_equity, "day": self.day,
            "positions": {t: asdict(p) for t, p in self._pos.items()},
        }, indent=2))

    def positions(self) -> dict[str, Position]:
        return dict(self._pos)

    def account(self, prices: dict[str, float]) -> Account:
        today = date.today().isoformat()
        eq = self.cash + sum(p.qty * prices.get(t, p.avg_price) for t, p in self._pos.items())
        if self.day != today:  # roll the daily baseline once per day
            self.last_equity, self.day = eq, today
            self._save()
        return Account(eq, self.cash, self.last_equity)

    def submit(self, order: Order, price: float) -> None:
        if order.side == "buy":
            fill = price * (1 + self.slip)
            cost = fill * order.qty
            if cost > self.cash + 1e-6:
                raise ValueError("insufficient cash")
            self.cash -= cost
            old = self._pos.get(order.ticker)
            if old:
                tot = old.qty + order.qty
                self._pos[order.ticker] = Position(tot, (old.qty * old.avg_price + cost) / tot)
            else:
                self._pos[order.ticker] = Position(order.qty, fill)
        else:
            old = self._pos.get(order.ticker)
            if not old or order.qty > old.qty + 1e-9:
                raise ValueError("cannot sell more than held (no shorting)")
            self.cash += price * (1 - self.slip) * order.qty
            left = old.qty - order.qty
            if left <= 1e-9:
                del self._pos[order.ticker]
            else:
                self._pos[order.ticker] = Position(left, old.avg_price)
        self._save()

    def market_open(self) -> bool:
        return True  # simulator has no clock


class AlpacaPaperBroker:
    """Alpaca paper trading over REST (free account, no real money)."""
    name = "alpaca-paper"

    def __init__(self, key: str, secret: str) -> None:
        if not key or not secret:
            raise RuntimeError("Set ALPACA_KEY_ID and ALPACA_SECRET_KEY (paper keys).")
        self.http = httpx.Client(
            base_url=ALPACA_PAPER_URL, timeout=30,
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        )

    def account(self, prices: dict[str, float]) -> Account:
        r = self.http.get("/account")
        r.raise_for_status()
        a = r.json()
        return Account(float(a["equity"]), float(a["cash"]), float(a["last_equity"]))

    def positions(self) -> dict[str, Position]:
        r = self.http.get("/positions")
        r.raise_for_status()
        return {p["symbol"]: Position(float(p["qty"]), float(p["avg_entry_price"]))
                for p in r.json()}

    def submit(self, order: Order, price: float) -> None:
        r = self.http.post("/orders", json={
            "symbol": order.ticker, "qty": str(int(order.qty)), "side": order.side,
            "type": "market", "time_in_force": "day",
        })
        r.raise_for_status()

    def market_open(self) -> bool:
        r = self.http.get("/clock")
        r.raise_for_status()
        return bool(r.json()["is_open"])
