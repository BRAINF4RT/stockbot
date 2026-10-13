# stockbot

A free-only **paper-trading** stock bot. It is **fully standalone**: the research code is adapted from
[`forecasting-tools`](https://github.com/BRAINF4RT/forecasting-tools) and
[`forecastingbot`](https://github.com/BRAINF4RT/forecastingbot) and copied into `stockbot/free_research/`,
so nothing here imports `forecasting_tools` (a test enforces this).

Each stock is treated like a Metaculus question — *"Will NVDA close higher in 5 trading days than
$X?"* — and gets the same treatment your forecasting bot gives questions:

```
price history (yfinance → Stooq fallback)
        │
        ├── technical score  (6-1 momentum, 200d trend, 50/200 cross, RSI cap)      [signals.py]
        │
        └── TrustedSearcher (extends vendored FreeSearcher):                           [searcher.py]
            Google News, GDELT, DDGS news, Wikipedia, scraped web, Polymarket/Manifold
                    │
                    ├── finance trust scores + recency → drop chatter, rank, tag [S1]..  [trust.py, evidence.py]
                    │
                    └── LLM probability P(up) with cited claims, median of N,             [analyst.py]
                        shrunk toward 0.5, clipped to [0.35, 0.65]
                        └── claims lexically verified against their cited source;
                            unverified claims shrink the LLM's lean by up to half
        │
        ▼
 score = 0.6·technical + 0.4·(2·P(up) − 1)   → vol-scaled, long-only target weights  [strategy.py]
        │
        ▼
 HARD RISK LIMITS (code, not the model): per-name cap, gross cap, stop-loss,          [risk.py]
 daily-loss kill switch, no shorting, no leverage, whole shares, cash-limited buys
        │
        ▼
 SimBroker (offline JSON)  or  Alpaca PAPER trading                                   [broker.py]
        │
        └── every forecast and order appended to state/journal.jsonl (for calibration) [journal.py]
```

## Everything is free

| Need | Service | Cost |
|---|---|---|
| Prices | yfinance, Stooq fallback | free (yfinance is unofficial and can break; Stooq backs it up) |
| News/web research | vendored `FreeSearcher` in `stockbot/free_research/` (DDGS, Google News RSS, GDELT, Wikipedia, prediction markets) | free |
| LLM | OpenRouter `:free` models (Nemotron → Laguna → Qwen chain, same as forecastingbot) | free, rate-limited |
| Execution | local `SimBroker`, or Alpaca **paper** account | free |
| Scheduling | GitHub Actions (`.github/workflows/trade.yml`) | free on public repos; 2,000 min/mo on private |

Cost guard built in: `llm.py` **refuses any model id that doesn't end in `:free`**, and every LLM call
(including the research query planner) goes through it. The original fork defaulted its planner to
`openrouter/auto`, which can route to paid models; that default was removed in the vendored copy.

## Evidence quality (adapted from Daily-Learn)

- **Finance trust scores** (`trust.py`): SEC/Fed/BLS and company investor-relations pages score highest, then wires and
  financial press (Reuters, AP, Bloomberg, WSJ, FT, CNBC), then opinion/aggregator sites; Reddit, StockTwits, X, TipRanks and similar
  are dropped. Google News/GDELT links are redirects, so the *publisher name* is scored too. Edit the dicts to taste.
- **Ranking**: `trust × recency` (recency halves every 10 days), duplicates collapsed, top 14 kept, each tagged `[S1]`...
- **Claim verification** (`evidence.py`): the model must list each claim with the source tag it relies on. A claim counts as
  verified only if ≥60% of its content words appear in that source *and* every number it states appears there too. The LLM's lean is
  then scaled by `0.5 + 0.5 × (verified / total)`; citing nothing counts as 0 verified. This catches invented figures and
  misattribution. It does **not** prove a claim true, and a claim can be faithful to a wrong article.
- The vendored searcher's LLM "condense" step is skipped (the analyst reads the tagged sources directly), so each ticker costs 2 free-LLM
  calls per sample (query planner + forecast) instead of 3. Verification outcomes and source lists are written to `journal.jsonl`.

## Safety

- **Paper only.** The Alpaca URL is hard-coded to `paper-api.alpaca.markets`; there is no live-trading code path.
- **Dry-run by default.** `run` prints the orders it *would* place; nothing is sent without `--execute`.
- Risk limits live in `risk.py`/`config.py` and run *after* the model; the LLM can't override them.
- If the LLM or news sources fail, the bot falls back to technical-only for that name and keeps going.

## Setup

```bash
git clone <your new repo> && cd stockbot
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'               # everything needed, including research libraries
cp .env.template .env                  # add OPENROUTER_API_KEY (and Alpaca paper keys if wanted)
pytest                                 # no network needed
```

## Use

```bash
python -m stockbot backtest --no-research          # technical-only backtest vs equal-weight buy & hold
python -m stockbot scan                            # score the universe, place nothing
python -m stockbot run                             # dry run against the offline simulator
python -m stockbot run --execute                   # simulator, places simulated orders
python -m stockbot run --broker alpaca --execute   # Alpaca paper account
python -m stockbot scan --no-research              # fast, no LLM calls
```

To run on a schedule, add `OPENROUTER_API_KEY`, `ALPACA_KEY_ID`, `ALPACA_SECRET_KEY` as repo secrets;
the included workflow runs on weekdays and only trades if the market is open.

## Tuning (all env vars, see `.env.template`)

`UNIVERSE`, `HORIZON_DAYS`, `MAX_POSITION_PCT` (10%), `MAX_GROSS_PCT` (70%), `STOP_LOSS_PCT` (8%),
`MAX_DAILY_LOSS_PCT` (3%). Default sizing is deliberately small: per-name weight =
`max_position × conviction × vol_scalar`, so a typical position is ~3–6% of equity.

## Limits you should know about

- **No proven edge.** The technical score is a standard momentum/trend heuristic, and 5-day stock direction is close to a coin flip.
  The LLM view is shrunk hard toward 0.5 for that reason. Expect modest, noisy results.
- **The news/LLM layer can't be backtested honestly.** Free news sources aren't point-in-time, so any backtest of it would leak the
  future. `backtest` is technical-only. The way to evaluate the LLM layer is forward: let it paper trade, then score
  `journal.jsonl` (`forecast` entries have `p_up`, price and horizon) with Brier/log score against what actually happened.
- Free LLM tiers are rate-limited, so research runs one ticker at a time and only for candidates/holdings. Large universes will be slow.
- Free data has gaps (no point-in-time fundamentals, occasional yfinance outages, adjusted prices only).
- Run it on paper for months and compare to simply holding SPY before trusting any of it with real money.
  This is software, not financial advice.
