# Zipfolio
<p align="center">
  <a href="https://pypi.python.org/pypi/zipfolio"><img src="https://img.shields.io/badge/python-3.12+-blue.svg?style=flat" alt="Python"></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue.svg?style=flat" alt="License: AGPL-3.0"></a>
  <a href="https://pypi.python.org/pypi/zipfolio"><img src="https://img.shields.io/pypi/v/zipfolio?maxAge=60" alt="PyPI"></a>
  <a href="https://pypi.python.org/pypi/zipfolio"><img src="https://img.shields.io/pypi/dm/zipfolio.svg?maxAge=2592000&label=installs&color=%2327B1FF" alt="Installs"></a>
  <a href="https://github.com/Limex-com/zipfolio"><img src="https://img.shields.io/github/stars/Limex-com/zipfolio.svg?style=social&label=Star&maxAge=60" alt="Stars"></a>
</p>

**Backtest tearsheets that read the book, not just the returns.**

Zipfolio turns a backtest into one self-contained HTML report about *how the
strategy behaves position by position*: what it held and how concentrated it
was, what each instrument and sector earned, which factors it was really
exposed to, where its risk sat, how its orders were filled -- and how far those
fills can be believed -- and what trading costs took. It is part of the
[ZipLime](https://github.com/Limex-com) ecosystem, where it is built for every
backtest next to the QuantStats returns tearsheet.

```python
from zipfolio import build_tearsheet

build_tearsheet("report.html", returns, benchmark,
                position_records=positions, cash=ending_cash, transactions=fills,
                orders=orders, fill_bars=fill_bars, decision_prices=decisions,
                market_data=daily_bars, cost_settings=settings)
```

---

## Why a backtester-aware report

A returns tearsheet -- QuantStats, or pyfolio fed only a return series -- sees
one number per day. That is enough for Sharpe ratios and drawdowns, and nothing
else: it cannot tell you whether a fill was possible, what slippage cost, or
which position carried the risk, because none of that is in the returns.

A backtester knows all of it while it runs. Every order has a type, a creation
time, a limit or a stop; every fill belongs to an order and happens on a bar
with an open, high, low, close and volume; every cost comes from a model with
known settings. Most of this is thrown away once the simulation ends.
**Zipfolio is built to be fed by the backtester itself**, while that detail
still exists, and that is what makes its metrics possible:

| Question the report answers | What it needs | Where a ziplime run has it |
| --- | --- | --- |
| Was each fill priced inside the bar it happened on? | the fill's own bar (OHLC) | the market-data bundle, joined as of the fill time |
| Did a fill take more of the bar's volume than could trade? | the fill's bar volume | the same bundle |
| Did the strategy trade at a price it had only just seen? | execution timing settings | `same_bar_execution`, `price_used_in_order_execution` |
| What did a fill cost over the price the strategy decided at? | the close at the order's creation | the order's `created` time, joined to the bundle |
| How long did orders wait; were any left unfilled, and why? | order lifecycle | orders: `created`, `dt`, `filled`, `status`, `reason` |
| Which order types did the strategy use? | execution style | `MarketOrder` / `LimitOrder` / `StopOrder` / `StopLimitOrder` |
| What did commission cost? | commission per order | `order.commission` (ziplime charges it to the order, not the fill) |
| What did slippage cost? | the slippage model | `FixedBasisPointsSlippage`: reconstructed exactly from the fill prices |
| What did each instrument earn? | positions and fills per day | position snapshots, fills with prices and quantities |
| Which names made each drawdown? | the same, between a peak and its valley | (same) |
| Was the risk each day what the positions said? | that day's weights and the names' history | position snapshots, the bundle's daily bars |
| How much capital could the strategy take? | daily volume per traded name | the bundle, folded into sessions |
| How is the cost model set, in words? | the settings themselves | `PerShare` / `PerDollar`, minimum per order, slippage bps, volume limit |

The result reads like an execution and risk desk's review of the backtest,
not a restatement of its equity curve.

## What is in a report

Charts on the left and narrow tables on the right in a 960px column, in the
QuantStats Pro look (white ground, light grid, the same palette: strategy
`#6B9E00`, benchmark `#5C5C5C`), so the two reports read as one product.
Every chart is SVG; the page has no external requests. The header and footer
name the Zipfolio version that built the page and link back to this repository.
The default title is "Zipfolio Tearsheet: Portfolio, Execution & Cost Analysis"
(`title=` changes it).

The page opens with a **summary** in the look of QuantStats Pro's health
summary: a score (checks passed out of checks made), counts of checks that hold
up, need a look or are critical, and one table of verdicts -- measure, value,
badge -- under the part of the report each belongs to, each row linked to its
section with the reasoning on hover. Facts that are not verdicts (alpha that
is not significant, turnover, the longest losing streak) go on one line under it.
The checks and their thresholds are constants at the top of each section's module. The sections
follow in four parts, each the question a reader brings:

**I. Where the return came from**

| Section | Shown when | Contents |
| --- | --- | --- |
| **Performance** | always | key metrics, cumulative returns, underwater curve -- deliberately short; returns are QuantStats' job |
| **Profit Concentration** | 50+ sessions | the return without the best 5, 10 and 20 days; the share of net PnL in the best trades; a luck test -- the same trades redrawn at random 5,000 times -- and its chance of a net loss |
| **Instruments** | traded | each name's PnL, contribution, weight, days held, trade stats; contribution by year; daily and round-trip return distributions per name; cumulative PnL by name |
| **Long and Short** | the book went short | PnL, contribution, exposure, winning days and return on exposure of each side; cumulative and yearly PnL by side |
| **Sector Analysis** | a held name has a sector | allocation over time, average and peak, profit attribution by GICS sector |
| **Factor Analysis** | a factor model is set | Fama-French 5 + momentum (US) or a market model against the benchmark over RUONIA (Moscow): betas with Newey-West t-stats, alpha, fit, attribution (overall and by year), rolling betas, return decomposition |

**II. What it risked**

| Section | Shown when | Contents |
| --- | --- | --- |
| **Drawdowns** | traded | each of the five worst drawdowns split, exactly, across the names and sectors that made it |
| **Market Regimes** | a benchmark is given | up and down capture, beta on up and down days, the average month in rising/falling x calm/volatile markets, monthly returns against the benchmark |
| **Portfolio Risk** | always (more when traded) | VaR/CVaR/EVaR, drawdown-at-risk/CDaR/EDaR, Ulcer index, Gini mean difference; rolling tail risk; risk contribution per holding; clustered correlations; the same names under four allocation rules, walk-forward |
| **Risk Over Time** | traded | predicted volatility (each day's weights x trailing covariance) against what the next month delivered, each name's share of risk by month, effective number of names, top-3 and largest weight |

**III. How it holds and trades**

| Section | Shown when | Contents |
| --- | --- | --- |
| **Positions** | traded | exposure, top positions, long/short concentration, holdings, gross leverage |
| **Transactions** | traded | trading frequency, fill sizes, buys and sells, turnover, volume, fill times on the market's own clock |
| **Closed Trades** | 5+ trades | what counts as a trade, with examples (pyfolio's round trips: every sale with the shares it closes, oldest first); counts, PnL, holding time and return on cost, long and short; share of profit by symbol; lifetimes, distributions |

**IV. Can it be traded**

| Section | Shown when | Contents |
| --- | --- | --- |
| **Next-Bar Fills** | orders fill on their signal bar and `fill_bars` carry the next open | every fill priced again at the open of the instrument's next bar, same shares and slippage: returns, Sharpe and drawdown both ways, the move per fill, what the wait cost over time |
| **Execution** | orders are given | how the backtest fills orders, realism checks, orders by type, fill statistics, fill price within the bar and against its typical price, share of bar volume, implementation shortfall, waits |
| **Trading Costs** | traded | the cost model in plain words, cost summary and per symbol, costs over time, returns before and after costs, sensitivity to extra slippage |
| **Capacity** | daily bars are given | days to liquidate, fills over 5% of daily volume, Sharpe across capital bases |

A section that cannot be computed becomes a short note, a chart that fails is
left out, and both are logged; the page fails only if nothing at all can be
reported.

## US and Moscow

Zipfolio reports a US book and a Moscow book each in its own terms:

| | US | Moscow |
| --- | --- | --- |
| Currency (`currency=`) | `"USD"`: `$1,234.50`, capacity at a $1m base | `"RUB"`: `1,234.50 ₽`, capacity at a 100m ₽ base |
| Factor analysis (`factor_model=`) | `"ff5_mom"`: Fama-French 5 + momentum | `"market"`: a market model against the benchmark -- IMOEX -- over RUONIA |
| Risk-free rate | Fama-French `RF` (one-month T-bill) | RUONIA, the Bank of Russia's overnight rate (`risk_free="ruonia"`) |
| Sectors | GICS, from a table you supply (`us_sector_mappings`) | the Moscow Exchange sector indices -- MOEXOG, MOEXFN, MOEXMM... (`moex_sector_mappings`) |
| Fill-time chart | New York clock | Moscow clock (`market_timezone="Europe/Moscow"`) |

There are no published Fama-French factors for Moscow, so the factor analysis
there is a one-factor market model: the strategy's return over RUONIA
regressed on IMOEX's, with the same exposures, alpha, attribution, rolling beta
and decomposition. The risk-free rate matters more there than anywhere: RUONIA
has run between 4% and 21% a year, and a regression without it would report
the rate as alpha. RUONIA is read from the Bank of Russia's `DailyInfo` web
service and cached for a day; each session earns the rate published the day
before, over the calendar days since the previous session (actual/365).

Moscow sectors come from the exchange's own sector indices, read from its ISS
service when a report is built and cached for a week: Oil & Gas (MOEXOG),
Electric Utilities (MOEXEU), Telecommunications (MOEXTL), Metals & Mining
(MOEXMM), Financials (MOEXFN), Consumer (MOEXCN), Chemicals (MOEXCH),
Transportation (MOEXTN), Information Technology (MOEXIT), Real Estate (MOEXRE).
Membership is today's, not the backtest's.

```python
from zipfolio import build_tearsheet, moex_sector_mappings

build_tearsheet("report.html", returns, imoex.rename("IMOEX"), ...,
                factor_model="market", risk_free="ruonia", currency="RUB",
                market_timezone="Europe/Moscow",
                sector_mappings=moex_sector_mappings(["SBER@MISX", "GAZP@MISX", ...]))
```

## Zipfolio and pyfolio

Zipfolio stands on [pyfolio-reloaded](https://github.com/stefan-jansen/pyfolio-reloaded)
and owes it most of its analytics. What it takes, what it changes, and what it
adds:

**Uses pyfolio for the computation.** Performance statistics, drawdown tables,
exposures and concentration, top positions, turnover, round-trip extraction and
statistics, profit attribution, sector exposure, liquidity and the capacity
sweep, slippage sensitivity -- all are pyfolio's functions (`timeseries`,
`pos`, `txn`, `round_trips`, `capacity`, `plotting.plot_*`).

**Replaces the presentation.** pyfolio is written for a notebook: tables go to
IPython's `display`, and each tear sheet is one tall seaborn figure of a dozen
panels. Zipfolio does not call the `create_*_tear_sheet` functions. It calls the
individual `plotting.plot_*` functions, one chart per figure on an axis it
owns, restyles each (palette, grid, percent axes, legends) and renders it to
SVG; it catches every table at `utils.print_table` as a *frame* and renders it
itself, with formatting pyfolio applies one-size-fits-all repaired (counts as
`146`, money as `-$5,067.13`, three-decimal percentages as two).

**Fixes, locally.** pyfolio-reloaded 0.9.9 is used as released; where it
misbehaves, Zipfolio works around it for the one call that needs it, never by
patching pyfolio globally:

- the capacity analysis fails whenever two names trade on the same day
  (`daily_txns_with_bar_data` calls `asfreq` on duplicate dates) -- a repaired
  copy is swapped in for that call;
- the transaction-time chart is fixed to New York's 09:30-16:00 -- Zipfolio
  draws its own, on the market's clock and over the hours fills actually fall in;
- the exposure panel disappears when pandas draws a regular-frequency index in
  period ordinals -- charts are drawn with matplotlib dates;
- labels and heatmap numbers that collide on dense axes are thinned and fitted
  after drawing.

**Leaves out what QuantStats already shows.** Rolling statistics, return
distributions, the monthly heatmap and stress events are pyfolio's too, but a
ZipLime backtest already has them in its QuantStats tearsheet.

**Adds what pyfolio does not have.** The Instruments, Execution, Trading Costs
and Factor Analysis sections, the Portfolio Risk section (computed with
[skfolio](https://github.com/skfolio/skfolio)), sector analysis from a GICS
table you supply, and the cost model described in words.

## Zipfolio and ziplime

Zipfolio is the reporting layer of ZipLime: it is designed around what a
ziplime 1.20 run produces, and the `zipfolio.ziplime` module reads the parts a
return series loses.

What it relies on from ziplime:

- **Orders as objects** -- `execution_style` (market, limit, stop, stop-limit),
  `created` and `dt`, `amount` and `filled`, `limit` and `stop`, `status` and
  `reason`, and `commission`, which ziplime charges to the order rather than the
  fill.
- **Fills with their order** -- each transaction carries its `order_id`,
  `amount`, `price` and simulation time.
- **The market-data bundle** -- a polars frame of OHLCV bars per instrument,
  daily or intraday, that the simulation read; `zipfolio.ziplime.execution_bars`
  joins every fill and every order to it *as of* the simulation time -- the fill
  with the open of the bar after it too, for the next-bar repricing -- and
  `daily_bars` folds it into sessions.
- **The cost and execution models** -- `PerShare` / `PerDollar` commission with
  a per-order minimum, `FixedBasisPointsSlippage` with a volume limit,
  `same_bar_execution` and `price_used_in_order_execution`. Zipfolio describes
  them in words and checks every fill against them.
- **Positions and cash per bar** -- position snapshots taken during
  `run_simulation_iter` (ziplime mutates position objects later, so they are
  copied as they are emitted) and `ending_cash` from the performance frame.

```python
from zipfolio.ziplime import daily_bars, execution_bars, order_type

# while the run's bundle is still loaded:
bars = daily_bars(bundle)                                  # symbol, day, price, volume
found = execution_bars(bundle, fills, orders)              # fills: id, symbol, dt; orders: id, symbol, created
fill_bars, decisions = found["fill_bars"], found["decision_prices"]
kind = order_type(order)                                   # "market", "limit", "stop", "stop_limit"
```

The ZipLime backtester calls exactly this after every simulation and passes the
frames to `build_tearsheet`. Nothing in the report requires ziplime itself:
the inputs are plain pandas frames, and any backtester that can produce them
gets the same report.

## Installation

```bash
pip install zipfolio
```

With the optional parts:

```bash
pip install "zipfolio[all]"      # everything
pip install "zipfolio[risk]"     # + skfolio, for the Portfolio Risk section
pip install "zipfolio[ziplime]"  # + polars, for zipfolio.ziplime
```

Without skfolio the Portfolio Risk section is left out; nothing else changes.
Requires Python 3.10+.

Until the first release reaches PyPI, or for the latest unreleased code,
install from GitHub:

```bash
pip install "zipfolio[all] @ git+https://github.com/Limex-com/zipfolio"
```

## Releases

Zipfolio is published on GitHub first and then on PyPI, under the same
version: a version is tagged here (`v0.1.0`), and the wheel built from that tag
is uploaded to PyPI as `zipfolio`. Every report says which version built it --
in the header ("Generated by ZipLime Zipfolio v0.1.0") and the footer, both
linking back to this repository -- so a report can always be traced to the code
that made it. Changes are listed in [CHANGELOG.md](CHANGELOG.md).

To publish a release:

```bash
poetry version 0.1.1             # or: patch / minor / major
git commit -am "Zipfolio 0.1.1" && git tag v0.1.1 && git push --tags
poetry build && poetry publish   # PyPI token configured with `poetry config pypi-token.pypi ...`
```

## Inputs

All frames are plain pandas. Only `returns` is required; each input switches
on the sections that need it.

| Argument | Shape | Used by |
| --- | --- | --- |
| `returns` | per-period strategy returns, datetime index | everything |
| `benchmark` | benchmark returns, same index, or `None` | Performance, Portfolio Risk |
| `position_records` | rows of `date, symbol, quantity, last_sale_price` per open position per day | Positions, Instruments, Sectors, Risk, Round Trips |
| `cash` | cash at the end of each period | (same) |
| `transactions` | rows of `id, order_id, dt, symbol, quantity, price` | Instruments, Transactions, Execution, Costs, Round Trips |
| `orders` | rows of `id, created, dt, type, symbol, quantity, filled, commission, limit, stop, status, reason` | Execution, Trading Costs, Instruments |
| `fill_bars` | `open, high, low, close, volume`, optionally `next_open, next_date`, indexed by fill `id` | Execution, Next-Bar Fills |
| `decision_prices` | `decision_price` indexed by order `id` | Execution (shortfall) |
| `market_data` | rows of `symbol, day, price, volume` | Capacity, Portfolio Risk |
| `cost_settings` | `commission_model, commission_per_share, commission_rate, commission_min_trade_cost, slippage_basis_points, slippage_volume_limit, price_used_in_order_execution, same_bar_execution, capital` | Trading Costs, Execution |
| `sector_mappings` | `{symbol: sector}`; `us_sector_mappings(["AAPL@XNGS", ...], table=...)` gives GICS sectors for US listings from your table | Sector Analysis |
| `factor_model` | `"ff5_mom"` (default), `"ff5"`, `"ff3"`, `"ff3_mom"`, `"market"` or `None` | Factor Analysis |
| `risk_free` | for `"market"`: `"ruonia"` (default) or a series of a rate in percent a year | Factor Analysis |
| `currency` | `"USD"` (default), `"RUB"`, `"EUR"` | every amount, the capacity base |
| `factor_data` | your own factor returns (decimals, with `RF`), instead of the download | Factor Analysis |
| `title` | the page title; defaults to `zipfolio.DEFAULT_TITLE` | header |
| `market_timezone` | e.g. `"Europe/Moscow"`; defaults to `ZIPFOLIO_MARKET_TIMEZONE`, then New York | fill-time chart |

`build_tearsheet` returns the titles of the sections that made it into the page,
in page order.

## Data the package fetches or ships

- **RUONIA** is read from the Bank of Russia's `DailyInfo` web service (cached
  a day) and the **Moscow sector indices** from the exchange's ISS service
  (cached a week), both under the same cache directory; unreachable with
  nothing cached, the section that needs them says so or is left out.
- **Fama-French factors** are downloaded from Kenneth French's
  [data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
  -- the same files `pandas_datareader.famafrench` reads, without the
  dependency -- and cached for a week under `~/.cache/zipfolio`
  (`ZIPFOLIO_CACHE_DIR`). A failed download falls back to a stale cache, then to
  a note in the report; the timeout is 8 seconds. The library publishes monthly
  with a lag, so the last weeks of a fresh backtest may fall outside it. The
  factors describe US stocks: pass `factor_model=None` for other markets.
- **GICS sectors are not shipped.** The classification is licensed by MSCI
  and S&P, so the codes are yours to bring: a CSV of `symbol,gics_sector`
  (two-digit sector code) passed as `us_sector_mappings(listings, table=...)`
  or found through `ZIPFOLIO_GICS_TABLE`. `scripts/build_us_sector_map.py`
  builds one from exports that carry GICS codes, one row per ticker from its
  most recent classification. `us_sector_mappings` looks up only listings on
  US venues, so a Moscow `T` is not filed as AT&T; a reused ticker gets its
  current holder's sector. Or skip the table and pass `sector_mappings`
  yourself -- any `{symbol: sector}` works, GICS or not.

## Limitations

- Futures exposure is understated: position values are quantity x price, and
  the records carry no contract multiplier.
- Slippage is reconstructed for the fixed-basis-point model; other slippage
  models would need their own reconstruction.
- Instrument contribution adds daily PnL over the start-of-day portfolio value,
  so dollars and percent can disagree in sign for a name that earned early and
  gave back later -- the report says so where it shows them.
- Factor analysis needs at least 60 sessions inside the factor data.

## Development

```bash
pip install -e ".[all]" pytest     # or: poetry install --all-extras
pytest
```

Tests never touch the network: factor downloads are replaced by synthetic
factors. Release notes are in [CHANGELOG.md](CHANGELOG.md).

## License

Apache License 2.0 -- see [LICENSE](LICENSE). Zipfolio builds on
pyfolio-reloaded (Apache-2.0) and skfolio (BSD-3-Clause). No market or
classification data is distributed with it.
