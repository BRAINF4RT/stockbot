import numpy as np
import pandas as pd
import pytest

from stockbot import analyst, llm
from stockbot.backtest import run_backtest
from stockbot.broker import Order, Position, SimBroker
from stockbot.config import Settings
from stockbot.risk import cap_weights, kill_switch, make_orders, stop_loss_hits
from stockbot.signals import quant_score
from stockbot.strategy import Signal, build_targets, target_weight


def prices(n=400, drift=0.0005, vol=0.01, seed=0, start=100.0):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({"close": start * np.exp(np.cumsum(r)), "volume": 1e6}, index=idx)


cfg = Settings()


# ---- signals -------------------------------------------------------------
def test_uptrend_scores_higher_than_downtrend():
    up = quant_score(prices(drift=0.002, vol=0.012)["close"])
    down = quant_score(prices(drift=-0.002, vol=0.012)["close"])
    assert up.score > 0.3 and down.score < -0.3 and up.score > down.score


def test_parabolic_moves_are_not_chased():
    q = quant_score(prices(drift=0.003, vol=0.005)["close"])  # perfectly smooth, RSI ~ 87
    assert q.rsi > 85 and q.score <= 0.3


def test_quant_needs_enough_bars():
    with pytest.raises(ValueError):
        quant_score(prices(100)["close"])


def test_quant_ignores_future_bars():
    df = prices()
    a = quant_score(df["close"].iloc[:300])
    b = quant_score(pd.concat([df["close"].iloc[:300], df["close"].iloc[300:] * 5]).iloc[:300])
    assert a.score == b.score


# ---- strategy ------------------------------------------------------------
def test_hysteresis_holds_between_exit_and_enter():
    s = (cfg.enter_score + cfg.exit_score) / 2
    assert target_weight(s, 0.2, held=False, cfg=cfg) == 0
    assert target_weight(s, 0.2, held=True, cfg=cfg) > 0


def test_high_vol_gets_smaller_weight():
    assert target_weight(0.8, 0.8, False, cfg) < target_weight(0.8, 0.2, False, cfg)


def test_llm_bearish_view_can_veto_marginal_trade():
    sig_bull = [Signal("X", 10, 0.4, 0.2, p_up=0.6)]
    sig_bear = [Signal("X", 10, 0.4, 0.2, p_up=0.35)]
    assert build_targets(sig_bull, set(), cfg)["X"] > 0
    assert build_targets(sig_bear, set(), cfg)["X"] == 0


# ---- risk ----------------------------------------------------------------
def test_caps_per_name_and_gross():
    w = cap_weights({f"T{i}": 0.5 for i in range(20)}, cfg)
    assert max(w.values()) <= cfg.max_position_pct + 1e-12
    assert sum(w.values()) <= cfg.max_gross_pct + 1e-9


def test_stop_loss_and_kill_switch():
    pos = {"A": Position(10, 100.0), "B": Position(10, 100.0)}
    assert stop_loss_hits(pos, {"A": 91.0, "B": 93.0}, cfg) == {"A"}
    assert kill_switch(96_000, 100_000, cfg)
    assert not kill_switch(99_000, 100_000, cfg)


def test_sale_proceeds_fund_buys_and_sells_come_first():
    pos = {"OLD": Position(10, 50.0)}
    px = {"OLD": 50.0, "NEW": 100.0}
    # equity = 100 cash + 10 * 50 = 600; target 50% NEW = 3 shares ($300)
    orders = make_orders({"NEW": 0.50}, pos, px, equity=600, cash=100.0, cfg=cfg)
    assert [(o.side, o.ticker, o.qty) for o in orders] == [("sell", "OLD", 10), ("buy", "NEW", 3)]


def test_unaffordable_buy_is_dropped_not_overdrawn():
    px = {"NEW": 100.0}
    orders = make_orders({"NEW": 0.10}, {}, px, equity=10_000, cash=100.0, cfg=cfg)
    assert orders == []  # wants $1,000 but only $100 cash


def test_halted_means_no_buys():
    orders = make_orders({"NEW": 0.10}, {}, {"NEW": 100.0}, 10_000, 10_000, cfg, allow_buys=False)
    assert orders == []


def test_penny_stocks_are_ignored():
    assert make_orders({"JUNK": 0.10}, {}, {"JUNK": 1.0}, 10_000, 10_000, cfg) == []


# ---- broker --------------------------------------------------------------
def test_sim_broker_roundtrip_and_persistence(tmp_path):
    b = SimBroker(tmp_path / "s.json", 10_000, slippage_bps=0)
    b.submit(Order("buy", "A", 10), 100.0)
    b.submit(Order("buy", "A", 10), 110.0)
    assert b.positions()["A"].qty == 20 and b.positions()["A"].avg_price == pytest.approx(105)
    assert b.cash == pytest.approx(10_000 - 2_100)
    b2 = SimBroker(tmp_path / "s.json", slippage_bps=0)  # reload from disk
    assert b2.positions()["A"].qty == 20
    b2.submit(Order("sell", "A", 20), 120.0)
    assert "A" not in b2.positions() and b2.cash == pytest.approx(10_000 + 300)


def test_sim_broker_rejects_shorting_and_overspend(tmp_path):
    b = SimBroker(tmp_path / "s.json", 1_000, slippage_bps=0)
    with pytest.raises(ValueError):
        b.submit(Order("sell", "A", 1), 10.0)
    with pytest.raises(ValueError):
        b.submit(Order("buy", "A", 1000), 10.0)


# ---- backtest ------------------------------------------------------------
def test_backtest_runs_and_respects_exposure():
    hist = {f"S{i}": prices(drift=0.0006, seed=i) for i in range(6)}
    r = run_backtest(hist, cfg)
    assert r.max_drawdown <= 0 and np.isfinite(r.sharpe)
    assert len(r.equity) == 400 - 210


def test_backtest_has_no_lookahead():
    """A signal built from tomorrow's price would make a random walk look brilliant."""
    hist = {f"S{i}": prices(drift=0.0, seed=10 + i) for i in range(8)}
    r = run_backtest(hist, cfg)
    assert r.sharpe < 2.5  # on pure noise, a clean backtest cannot show a huge Sharpe


# ---- llm / analyst -------------------------------------------------------
def test_refuses_paid_models(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-opus")
    with pytest.raises(llm.LLMError):
        llm._chain()


def test_free_models_allowed(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "some/model:free")
    assert llm._chain()[0] == "some/model:free"


@pytest.mark.parametrize("text,expected", [
    ('blah\n{"probability": 0.57, "reasoning": "ok"}', 0.57),
    ('{"probability": 62, "reasoning": "pct"}', 0.62),
    ("I estimate the probability is 45%.", 0.45),
])
def test_parse_forecast(text, expected):
    assert analyst.parse_forecast(text)[0] == pytest.approx(expected)


def test_parse_forecast_garbage():
    assert analyst.parse_forecast("no idea") is None


def test_shrink_clips_overconfidence():
    assert analyst.shrink(0.99, 0.5) == 0.65
    assert analyst.shrink(0.01, 0.5) == 0.35
    assert analyst.shrink(0.5, 0.5) == 0.5


async def test_forecast_median_and_shrink(monkeypatch):
    replies = iter(['{"probability": 0.9}', '{"probability": 0.6}', '{"probability": 0.7}'])

    async def fake(*a, **k):
        return next(replies)
    monkeypatch.setattr(llm, "generate", fake)
    fc = await analyst.forecast("q", "t", "r", samples=3, shrink_factor=0.5)
    assert fc.raw_p == pytest.approx(0.7) and fc.p_up == pytest.approx(0.6)


async def test_forecast_survives_llm_outage(monkeypatch):
    async def boom(*a, **k):
        raise llm.LLMError("down")
    monkeypatch.setattr(llm, "generate", boom)
    assert await analyst.forecast("q", "t", "r") is None
