# Zipfolio reference

Everything about a Zipfolio report: what each section and check contains and when it appears, the
accounting integrity check, what Zipfolio takes from pyfolio and from ziplime, every input, the data
it downloads, and how releases are made. For the overview and a quick start, see the
[README](../README.md).

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
The header and footer name the Zipfolio version that built the page and link
back to this repository. The default title is "Zipfolio Tearsheet: Portfolio,
Execution & Cost Analysis" (`title=` changes it).

**Charts are interactive**: hover for the values, click a legend entry to hide a
series, zoom any dated chart with its slider or the zoom tool. The icons in a
chart's corner open it larger, with zoom and pan, and copy it as an image or save it
as PNG, JPG or SVG; the button
over a table copies it for Excel or Google Sheets. A term in a table -- CVaR,
Treynor, M² -- explains itself on hover, or on tap on a phone (`zipfolio.glossary`).
On a phone the charts lay themselves out for the width they get. "Copy report for
an AI" in the header puts the report's numbers -- every metric and table, without
its verdicts or reasoning -- on the clipboard as Markdown, to talk the strategy
over with any assistant; it carries what the page shows and nothing more.
Each chart is drawn with matplotlib -- most through pyfolio's own plotting -- and
then read back, artist by artist, into an [Apache ECharts](https://echarts.apache.org)
chart with the same limits, ticks and colours (`zipfolio.interactive`). ECharts
5.6 (Apache-2.0) is inlined in the page, which still makes no external requests.
A chart the converter cannot read stays a static SVG; `interactive=False` keeps
them all static.

The page opens with a **summary** in the look of QuantStats Pro's health
summary: the main issues in one sentence, the facts at a glance (time in the
market, style, how far below its high the strategy ended, the extra cost that
would erase the profit, capacity), the score, and one table of **16 checks, four per
part** (`page.SCORECARD`), the same for every backtest. A row shows the check, its
value and a verdict; tapped, it opens the reasoning, the threshold, what to try,
and a link to the numbers.

Checks that measure one fact are a **family** -- one row, with the worst of its
members' verdicts -- so a strategy on one stock is not marked down four times
for it. A check that does not apply (the top two names of a book that traded
one) is N/A and out of the count; the score is out of the checks that could be
judged. Thresholds are constants at the top of each section's module. The
scorecard is also embedded in the page as JSON (`<script id="zipfolio-scorecard">`),
for an agent reading the report.

| Part | The four rows |
| --- | --- |
| I | annual return vs the benchmark; against buying and holding the same names; **more than luck** (the Sharpe ratio's probabilistic -- or, with `trials`, deflated -- significance, the luck test, alpha after the factors); **rests on a few days or trades** (the return without the best 5 days, profit from the best 5 trades) |
| II | max drawdown vs the benchmark (at the strategy's own exposure when it is partly in cash); up / down capture; longest time below a peak; **concentration in one name** (largest position, profit from the top 2 names, the riskiest name's share of predicted volatility, the largest sector) |
| III | winning trades and profit factor; enough closed trades to judge (30); profit still open at the end; max gross exposure |
| IV | fill timing (the return filled a bar later, when orders fill on their signal bar); fill realism; costs' share of the profit before costs; liquidity |

The parts start open; "Collapse all" folds them down to their verdicts, and a
link into a part opens it again. In each section, the paragraphs that explain how
to read it wait behind a "How to read this section" toggle. A strategy that
traded one name gets no breakdown by name or sector -- each would read "100%"
of that name.

The sections follow in four parts, each the question a reader brings:

**I. Where the return came from**

| Section | Shown when | Contents |
| --- | --- | --- |
| **Performance** | always | key metrics for the strategy, buying and holding its names, and the benchmark, side by side; the Sharpe ratio over the risk-free rate and at 0%; whether it is more than luck after N tries; first half against second; cumulative returns, underwater curve -- returns in depth are QuantStats' job |
| **Profit Concentration** | 50+ sessions | the return without the best 5, 10 and 20 days; the share of net PnL in the best trades; a luck test -- the same trades redrawn at random 5,000 times -- and its chance of a net loss |
| **Instruments** | two or more names traded | each name's PnL, contribution, weight, days held, trade stats; contribution by year; daily and round-trip return distributions per name; cumulative PnL by name |
| **Long and Short** | the book went short | PnL, contribution, exposure, winning days and return on exposure of each side; cumulative and yearly PnL by side |
| **Sector Analysis** | two or more names, one with a sector | a **treemap** in the look of finviz's market map -- a tile per name inside its sector, sized by the capital it took and coloured by what it added to the return; allocation over time, average and peak, net PnL and contribution by sector (GICS from a table you supply, or any classification you pass) |
| **Factor Analysis** | a factor model is set | Fama-French 5 + momentum (US stocks) or a market model against the benchmark over the risk-free rate (any other market): betas with Newey-West t-stats, alpha, fit, attribution (overall and by year), rolling betas, return decomposition; with SMB and HML, a **style box** -- Morningstar's large/mid/small by value/blend/growth grid, the strategy placed by its betas with its 95% interval, its rolling path and the benchmark |

**II. What it risked**

| Section | Shown when | Contents |
| --- | --- | --- |
| **Drawdowns** | traded | each of the five worst drawdowns split, exactly, across the names and sectors that made it |
| **Market Regimes** | a benchmark is given | up and down capture, beta on up and down days, the average month in rising/falling x calm/volatile markets, monthly returns against the benchmark |
| **Portfolio Risk** | always (more when traded) | VaR/CVaR/EVaR, drawdown-at-risk/CDaR/EDaR, Ulcer index, Gini mean difference; rolling tail risk; risk contribution per holding; clustered correlations; the same names under four allocation rules, walk-forward |
| **CAPM and the Efficient Frontier** | a benchmark, or bars of two held names | not a check: the security market line (strategy and names by beta, Jensen's alpha), the capital market line, CAPM measures (Treynor, M², systematic and specific volatility, tracking error, information ratio); with bars, the long-only efficient frontier of the names held most with the max-Sharpe and minimum-variance mixes -- hindsight, drawn as a ceiling |
| **Risk Over Time** | traded | predicted volatility (each day's weights x trailing covariance) against what the next month delivered, each name's share of risk by month, effective number of names, top-3 and largest weight |

**III. How it holds and trades**

| Section | Shown when | Contents |
| --- | --- | --- |
| **Positions** | traded | exposure, top positions, long/short concentration, holdings, gross leverage |
| **Transactions** | traded | trading frequency, fill sizes, buys and sells, turnover (traded value over portfolio value, and one-way), volume, fill times on the market's own clock |
| **Closed Trades** | 5+ trades | what counts as a trade, with examples (pyfolio's round trips: every sale with the shares it closes, oldest first); counts, PnL, holding time and return on cost, long and short; share of profit by symbol; lifetimes, distributions |

**IV. Can it be traded**

| Section | Shown when | Contents |
| --- | --- | --- |
| **Next-Bar Fills** | orders fill on their signal bar and `fill_bars` carry the next open | every fill priced again at the open of the instrument's next bar, same shares and slippage: returns, Sharpe and drawdown both ways, the move per fill, what the wait did to PnL over time, reconciled: the difference in final value is the sum over the fills |
| **Execution** | orders are given | how the backtest fills orders, realism checks, orders by type, fill statistics, fill price within the bar and against its typical price, share of bar volume, implementation shortfall, waits |
| **Trading Costs** | traded | the cost model in plain words, cost summary and per symbol, costs over time, returns before and after costs, sensitivity to extra slippage |
| **Capacity** | daily bars are given | the starting capital at which the largest fill reaches 1% of its day's volume, and the run's volume limit; days to liquidate at the strategy's own capital; fills over 1% of daily volume; Sharpe across capital bases |

A section that cannot be computed becomes a short note, a chart that fails is
left out, and both are logged; the page fails only if nothing at all can be
reported.

## Accounting integrity

Before a reader trusts what the page says about a strategy, the numbers have to add up to the book they
came from. The panel at the top of every page says whether they do: `zipfolio.integrity` replays the
book -- fills, fees, cash, positions, events -- independently of the report (no pyfolio), and checks every
profit figure the page publishes against it. It certifies the arithmetic of the book, nothing more: not
the market data, not the absence of look-ahead, not the realism of the fills, not an edge.

```python
from zipfolio.integrity import AccountingInputs, AccountingPolicy, check_integrity
from zipfolio.ziplime import contract_specs, opening_state

report = check_integrity(AccountingInputs(
    returns=returns, position_records=positions, cash=cash, transactions=fills, orders=orders,
    cost_settings=settings,
    opening_state=opening_state(capital, returns.index[0]),   # the book before the first session
    contract_specs=contract_specs(["AAPL@XNGS", "JPM@XNYS"]),  # what each instrument is
    portfolio_events=events,                                  # dividends, fees, flows, splits, if any
), AccountingPolicy(mode="strict"))
report.to_json("accounting_integrity.json")                   # deterministic, accounting-integrity/v1
report.raise_on_failure()                                     # strict: a failed invariant is fatal
```

`build_tearsheet` takes the same `opening_state`, `contract_specs` and `portfolio_events`, runs the
check itself and embeds the JSON (`#zipfolio-integrity`) beside the panel; `integrity_report=` shows one
made beforehand. From a shell or CI: `python -m zipfolio.integrity book.pkl --mode ci --json out.json`.

Each check has a stable id and one outcome -- `PASS`, `FAIL` (an invariant broken beyond tolerance),
`WARN` (a convention or label in doubt, the arithmetic intact), `SKIPPED` (not possible with the inputs
given, with what is missing) or `NOT_APPLICABLE`. The report is `FAIL` if a required check fails,
`INCOMPLETE` if one could not be made, `WARN` if a warning remains, `PASS` only when nothing is left open;
its assurance says how much was checked: `summary` (the published figures against each other), `reconciled`
(recomputed from the period snapshots) or `ledger` (every event replayed from the opening state). Missing
evidence is never a pass: a cash or share difference with no event ledger is `SKIPPED` -- a dividend or a
split could explain it -- and an instrument without a contract spec is not valued.

| Check | What it proves |
| --- | --- |
| `INPUT.FILL_UNIQUENESS`, `INPUT.TIMESTAMP_ALIGNMENT` | every fill once, readable; returns, snapshots and fills on the same sessions |
| `NAV.SNAPSHOT_RECONCILIATION` | cash plus positions at their marks equals the opening value compounded by the returns, every session |
| `RETURNS.NAV_RECONCILIATION` | each return is the change in cash plus positions, external flows out |
| `CASH.LEDGER_REPLAY`, `POSITIONS.LEDGER_REPLAY` | cash and shares replayed from the opening state, fills, fees and events equal the record, every session |
| `PNL.REPORTED_SUMMARY_IDENTITY` | the published realized + unrealized - commission + other cash equals the profit the returns imply (report figures only) |
| `PNL.PORTFOLIO_RECONCILIATION` | each name's change in value less its fills, less fees, plus income, equals the change in NAV net of flows |
| `PNL.REALIZED_UNREALIZED` | the report's split matches first-in-first-out lots from the fills |
| `COST.COMMISSION_RECONCILIATION` | the commission is the orders' total, each order once, and what the fee model gives |
| `COST.SLIPPAGE_SINGLE_COUNT` | slippage is in the fill prices and taken off nowhere else |
| `ATTRIBUTION.INSTRUMENT_SUM`, `ATTRIBUTION.SECTOR_SUM` | the names, and the sectors with an explicit Unclassified bucket, add up to the book |
| `ROUND_TRIP.SYNTHETIC_CLOSURES` | every closed trade closes on a real fill; open positions are unrealized, not trades |
| `ROUND_TRIP.GROSS_NET_CONVENTION`, `REPORT.DISPLAY_VS_CALCULATION` | nothing gross is called net; every shown figure is the calculated one |

v1 values cash equities and funds in one currency. Futures, options, bonds and foreign-currency
instruments are skipped, never valued with a guessed multiplier. The tests replay the real Momentum
Rotation backtest from its primary book (`tests/fixtures/momentum_rotation`) and check that a one-fill,
one-fee or one-mark corruption of random books always fails.

## Any market

Zipfolio reports a book of US stocks and a book from any other market, each in its own terms:

| | US stocks | Any other market |
| --- | --- | --- |
| Currency (`currency=`) | `"USD"`: `$1,234.50` | `"EUR"`: `€1,234.50` |
| Factor analysis (`factor_model=`) | `"ff5_mom"`: Fama-French 5 + momentum | `"market"`: a market model against the benchmark, over the risk-free rate |
| Risk-free rate | Fama-French `RF` (one-month T-bill) | `risk_free=`: your rate as a series in percent a year, such as €STR or SONIA |
| Sectors | GICS, from a table you supply (`us_sector_mappings`) | any `{symbol: sector}` |
| Fill-time chart | New York clock | the venue's own clock (`market_timezone="Europe/Berlin"`) |

The Fama-French factors describe US stocks. Elsewhere the factor analysis is a one-factor market model:
the strategy's return over the risk-free rate regressed on the benchmark's, with the same exposures,
alpha, attribution, rolling beta and decomposition. Daily factors of your own can replace it through
`factor_data=` (decimals, with an `RF` column). The risk-free rate matters most where it is high: a
regression without it reports the rate as alpha. Each session earns the rate known the day before,
over the calendar days since the previous session (actual/365).

```python
build_tearsheet("report.html", returns, dax.rename("DAX"), ...,
                factor_model="market", risk_free=estr.rename("€STR"),
                currency="EUR", market_timezone="Europe/Berlin",
                sector_mappings={"SAP": "Information Technology", "ALV": "Financials"})
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
ziplime run produces, and the `zipfolio.ziplime` module reads the parts a
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

## Releases

Zipfolio is published on GitHub first and then on PyPI, under the same
version: a version is tagged here (`v0.1.0`), and the wheel built from that tag
is uploaded to PyPI as `zipfolio`. Every report says which version built it --
in the header ("Generated by ZipLime Zipfolio v0.1.0") and the footer, both
linking back to this repository -- so a report can always be traced to the code
that made it. Changes are listed in [CHANGELOG.md](../CHANGELOG.md).

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
| `benchmark` | benchmark returns, same index, or `None` | Performance, Portfolio Risk, CAPM |
| `position_records` | rows of `date, symbol, quantity, last_sale_price` per open position per day | Positions, Instruments, Sectors, Risk, Round Trips |
| `cash` | cash at the end of each period | (same) |
| `transactions` | rows of `id, order_id, dt, symbol, quantity, price` | Instruments, Transactions, Execution, Costs, Round Trips |
| `orders` | rows of `id, created, dt, type, symbol, quantity, filled, commission, limit, stop, status, reason` | Execution, Trading Costs, Instruments |
| `fill_bars` | `open, high, low, close, volume`, optionally `next_open, next_date`, indexed by fill `id` | Execution, Next-Bar Fills |
| `decision_prices` | `decision_price` indexed by order `id` | Execution (shortfall) |
| `market_data` | rows of `symbol, day, price, volume` | Capacity, Portfolio Risk, Efficient Frontier |
| `cost_settings` | `commission_model, commission_per_share, commission_rate, commission_min_trade_cost, slippage_basis_points, slippage_volume_limit, price_used_in_order_execution, same_bar_execution, capital` | Trading Costs, Execution |
| `sector_mappings` | `{symbol: sector}`; `us_sector_mappings(["AAPL@XNGS", ...], table=...)` gives GICS sectors for US listings from your table | Sector Analysis |
| `factor_model` | `"ff5_mom"` (default), `"ff5"`, `"ff3"`, `"ff3_mom"`, `"market"` or `None` | Factor Analysis |
| `risk_free` | for `"market"`: the risk-free rate as a series in percent a year, or `None` for 0% | Factor Analysis |
| `currency` | `"USD"` (default) or `"EUR"` | how every amount is written |
| `run_info` | `{label: value}` | what the header says about the run before what the page knows: strategy and version, the benchmark's ticker and whether it is a price or a total return, universe |
| `trials` | int | versions of the strategy backtested; the Sharpe ratio is then judged against the best of that many no-edge strategies (deflated Sharpe ratio) |
| `interactive` | `True` (default), `False` | ECharts charts with tooltips and zoom, or static SVG |
| `opening_state` | `OpeningState` -- the book before the first session (`zipfolio.ziplime.opening_state(capital, start)`) | Accounting integrity (ledger) |
| `contract_specs` | `{symbol: ContractSpec}` (`zipfolio.ziplime.contract_specs(listings)`) | Accounting integrity |
| `portfolio_events` | rows of `dt, kind, symbol, cash, quantity, ratio` -- dividends, interest, fees, deposits, withdrawals, splits | Accounting integrity, Profit Reconciliation |
| `integrity`, `integrity_report`, `integrity_policy` | `False` to leave the panel out; a report made beforehand; an `AccountingPolicy` | Accounting integrity |
| `factor_data` | your own factor returns (decimals, with `RF`), instead of the download | Factor Analysis |
| `title` | the page title; defaults to `zipfolio.DEFAULT_TITLE` | header |
| `market_timezone` | e.g. `"Europe/Berlin"`; defaults to `ZIPFOLIO_MARKET_TIMEZONE`, then New York | fill-time chart |

`build_tearsheet` returns the titles of the sections that made it into the page,
in page order.

## Data the package fetches or ships

- **Fama-French factors** are downloaded from Kenneth French's
  [data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
  -- the same files `pandas_datareader.famafrench` reads, without the
  dependency -- and cached for a week under `~/.cache/zipfolio`
  (`ZIPFOLIO_CACHE_DIR`). A failed download falls back to a stale cache, then to
  a note in the report; the timeout is 8 seconds. The library publishes monthly
  with a lag, so the last weeks of a fresh backtest may fall outside it. The
  factors describe US stocks: pass `factor_model="market"` or `None` for other markets.
- **GICS sectors are not shipped.** The classification is licensed by MSCI
  and S&P, so the codes are yours to bring: a CSV of `symbol,gics_sector`
  (two-digit sector code) passed as `us_sector_mappings(listings, table=...)`
  or found through `ZIPFOLIO_GICS_TABLE`. `scripts/build_us_sector_map.py`
  builds one from exports that carry GICS codes, one row per ticker from its
  most recent classification. `us_sector_mappings` looks up only listings on
  US venues, so a `T` listed elsewhere is not filed as AT&T; a reused ticker gets its
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
- The efficient frontier is measured with the returns the names went on to
  make: hindsight, a ceiling no strategy could have held. It is long only.

## Development

```bash
pip install -e ".[all]" pytest     # or: poetry install --all-extras
pytest
```

Tests never touch the network: factor downloads are replaced by synthetic
factors. Release notes are in [CHANGELOG.md](../CHANGELOG.md).
