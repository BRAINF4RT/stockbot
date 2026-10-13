"""Command line: scan | run | backtest."""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from stockbot import analyst, journal, research
from stockbot.backtest import run_backtest
from stockbot.broker import AlpacaPaperBroker, Broker, SimBroker
from stockbot.config import COMPANY_NAMES, Settings
from stockbot.data import get_universe_history
from stockbot.risk import cap_weights, kill_switch, make_orders, stop_loss_hits
from stockbot.signals import quant_score
from stockbot.strategy import Signal, build_targets

log = logging.getLogger("stockbot")


def _broker(kind: str, cfg: Settings) -> Broker:
    if kind == "alpaca":
        return AlpacaPaperBroker(cfg.alpaca_key, cfg.alpaca_secret)
    return SimBroker(Path(cfg.state_dir) / "sim.json", cfg.sim_start_cash)


async def build_signals(cfg: Settings, history: dict, held: set[str],
                        use_research: bool) -> list[Signal]:
    signals: list[Signal] = []
    for t, df in history.items():
        q = quant_score(df["close"])
        price = float(df["close"].iloc[-1])
        sig = Signal(t, price, q.score, q.vol)
        sig.notes.append(f"mom6-1={q.momentum:+.1%} rsi={q.rsi:.0f} vol={q.vol:.0%} "
                         f"{'>' if q.above_200 else '<'}200d")
        # Spend scarce free-tier LLM calls only on plausible candidates or current holdings.
        if use_research and (q.score > -0.2 or t in held):
            name = COMPANY_NAMES.get(t, t)
            try:
                res = await research.research_ticker(
                    t, name, price, cfg.horizon_days, technical_note="; ".join(sig.notes))
                fc = await analyst.forecast(
                    research.make_question(t, name, price, cfg.horizon_days),
                    "; ".join(sig.notes), res.text, sources=res.sources,
                    samples=cfg.llm_samples, shrink_factor=cfg.llm_shrink)
                if fc:
                    sig.p_up = fc.p_up
                    sig.notes.append(
                        f"P(up)={fc.p_up:.2f} (raw {fc.raw_p:.2f}, "
                        f"{fc.supported}/{fc.total} claims verified, {len(res.sources)} sources) "
                        f"{fc.reasoning}")
                    journal.log(cfg.state_dir, "forecast", ticker=t, price=price,
                                horizon=cfg.horizon_days, p_up=fc.p_up, raw=fc.raw_p,
                                claims_supported=fc.supported, claims_total=fc.total,
                                sources=[{"id": x.id, "url": x.url, "trust": x.trust}
                                         for x in res.sources],
                                reasoning=fc.reasoning)
            except Exception as exc:  # noqa: BLE001 - fall back to technical-only
                log.warning("research failed for %s: %s", t, exc)
        signals.append(sig)
    return signals


def _print_table(signals: list[Signal]) -> None:
    print(f"{'TICKER':7}{'PRICE':>9}{'QUANT':>7}{'P(UP)':>7}{'SCORE':>7}{'WEIGHT':>8}  NOTES")
    for s in sorted(signals, key=lambda s: -s.score):
        p = f"{s.p_up:.2f}" if s.p_up is not None else "  -"
        print(f"{s.ticker:7}{s.price:9.2f}{s.quant:7.2f}{p:>7}{s.score:7.2f}{s.weight:8.1%}  "
              f"{s.notes[0] if s.notes else ''}")


async def cmd_scan(args, cfg: Settings) -> None:
    history = get_universe_history(cfg.universe)
    sigs = await build_signals(cfg, history, set(), not args.no_research)
    build_targets(sigs, set(), cfg)
    _print_table(sigs)


async def cmd_run(args, cfg: Settings) -> None:
    broker = _broker(args.broker, cfg)
    if not args.force and not broker.market_open():
        print("Market is closed; nothing to do (use --force to run anyway).")
        return
    history = get_universe_history(cfg.universe)
    prices = {t: float(df["close"].iloc[-1]) for t, df in history.items()}
    positions = broker.positions()
    acct = broker.account(prices)
    held = set(positions)

    sigs = await build_signals(cfg, history, held, not args.no_research)
    weights = build_targets(sigs, held, cfg)
    stopped = stop_loss_hits(positions, prices, cfg)
    for t in stopped:
        weights[t] = 0.0
        print(f"STOP-LOSS: exiting {t}")
    halted = kill_switch(acct.equity, acct.last_equity, cfg)
    if halted:
        print("KILL SWITCH: daily loss limit hit; sells only.")
    weights = cap_weights(weights, cfg)
    for s in sigs:
        s.weight = weights.get(s.ticker, 0.0)
    _print_table(sigs)

    for t in set(positions) - set(prices):  # holdings we have no price for: leave alone
        log.warning("No price for held %s; not touching it", t)
    orders = make_orders(weights, {t: p for t, p in positions.items() if t in prices},
                         prices, acct.equity, acct.cash, cfg, allow_buys=not halted)
    print(f"\nEquity ${acct.equity:,.2f}  cash ${acct.cash:,.2f}  broker={broker.name}")
    if not orders:
        print("No orders.")
    for o in orders:
        tag = "EXECUTE" if args.execute else "DRY-RUN"
        print(f"{tag}: {o.side.upper():4} {o.qty:g} {o.ticker} @ ~{prices[o.ticker]:.2f}")
        if args.execute:
            try:
                broker.submit(o, prices[o.ticker])
                journal.log(cfg.state_dir, "order", side=o.side, ticker=o.ticker,
                            qty=o.qty, price=prices[o.ticker], broker=broker.name)
            except Exception as exc:  # noqa: BLE001
                log.error("order failed %s %s: %s", o.side, o.ticker, exc)
    if not args.execute and orders:
        print("\n(dry run: add --execute to place these orders)")


async def cmd_backtest(args, cfg: Settings) -> None:
    history = get_universe_history(cfg.universe, days=1500)
    if not history:
        print("No data.")
        return
    r = run_backtest(history, cfg)
    print("Technical-only backtest (LLM/news layer is not backtestable):")
    print(f"  strategy : return {r.total_return:+.1%}  CAGR {r.cagr:+.1%}  "
          f"Sharpe {r.sharpe:.2f}  maxDD {r.max_drawdown:.1%}  turnover {r.turnover_per_year:.1f}x/yr")
    print(f"  benchmark: return {r.bench_total_return:+.1%}  maxDD {r.bench_max_drawdown:.1%} "
          "(equal-weight buy & hold, fully invested)")
    print("  Past performance does not predict future results; the strategy is capped at "
          "partial exposure, so compare risk-adjusted numbers, not just return.")


def main() -> None:
    try:
        from dotenv import load_dotenv  # noqa: PLC0415
        load_dotenv()
    except ImportError:
        pass
    p = argparse.ArgumentParser(prog="stockbot", description="Free paper-trading stock bot")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("scan", "run", "backtest"):
        sp = sub.add_parser(name)
        sp.add_argument("--no-research", action="store_true",
                        help="technical-only; skips news research and LLM calls")
        sp.add_argument("-v", "--verbose", action="store_true")
        if name == "run":
            sp.add_argument("--broker", choices=["sim", "alpaca"], default="sim")
            sp.add_argument("--execute", action="store_true",
                            help="actually place (paper) orders; default is dry-run")
            sp.add_argument("--force", action="store_true", help="run even if market is closed")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = Settings.from_env()
    asyncio.run({"scan": cmd_scan, "run": cmd_run, "backtest": cmd_backtest}[args.cmd](args, cfg))


if __name__ == "__main__":
    main()
