# Changelog

## 0.2.1 -- 2026-10-14

A report that says what it found, and three sections that ask whether the
result is real.

- **Scorecard.** The page opens with a score out of 16 -- the same sixteen
  checks for every backtest, four per part, each against a fixed threshold
  (`page.SCORECARD`) -- the counts that hold up, need a look or are critical,
  one table of verdicts linked to the sections, and a line of context. A
  check that cannot be made reads N/A and does not pass.
- **Four parts**: where the return came from, what it risked, how it holds
  and trades, whether it can be traded; a sticky bar jumps between them.
- **Next-Bar Fills**: every fill on its signal bar priced again at the next
  bar's open, same shares and slippage. `zipfolio.ziplime.execution_bars`
  now returns `next_open` and `next_date` with each fill's bar.
- **Profit Concentration**: the return without its best 5, 10 and 20 days;
  the share of profit in the best trades; a luck test -- the same trades
  redrawn 5,000 times -- with its chance of a net loss.
- **Long and Short**: PnL, exposure and return on exposure of each side.
- **Closed Trades** (was Round Trips) opens with what counts as a trade, with
  examples: pyfolio makes one per closing sale, not one per position.
- New checks: alpha significance, the riskiest name's share of predicted
  volatility, the largest sector's share of the book, drawdown against the
  benchmark, time below a peak, the worst 5% of days, profit still open at
  the end, leverage, the longest losing streak.
- Fix: a trade's return is its PnL over what its shares cost (`rt_returns`);
  Instruments read pyfolio's PnL over the whole portfolio.
- `Section.finding(status, label, value, note, key)` for sections of your own.

## 0.1.0 -- 2026-10-13

First release, extracted from the ZipLime backtester, where it builds the
report next to the QuantStats tearsheet of every backtest.

- `build_tearsheet`: one self-contained HTML page in the QuantStats Pro look --
  Performance, Positions, Instruments, Sector Analysis, Factor Analysis,
  Portfolio Risk, Transactions, Execution, Trading Costs, Round Trips, Capacity.
- Drawdowns split exactly across the names and sectors that made them; market
  regimes (up/down capture, asymmetric beta, months by regime); risk over
  time (predicted against realized volatility, risk shares, concentration).
- Moscow books: amounts in roubles (`currency="RUB"`), a market model against
  IMOEX over RUONIA from the Bank of Russia (`factor_model="market"`), and
  sectors from the Moscow Exchange sector indices (`moex_sector_mappings`).
- `zipfolio.ziplime`: daily bars, per-fill bars and decision prices, and order
  types read from a ziplime run.
- Fama-French 3/5 factors and momentum, daily, from Kenneth French's data
  library, cached for a week.
- GICS sectors for US tickers from a table the user supplies (`table=` or
  `ZIPFOLIO_GICS_TABLE`); `scripts/build_us_sector_map.py` builds one. No
  sector data ships with the package -- GICS is licensed by MSCI and S&P.
- Every report links to https://github.com/Limex-com/zipfolio in its header
  and footer, next to the Zipfolio version that built it, under the title
  "Zipfolio Tearsheet: Portfolio, Execution & Cost Analysis".
- Fixes to pyfolio-reloaded 0.9.9 behaviour, applied without patching it
  globally: the capacity join on same-day fills, the New-York-only fill-time
  chart, the exposure panel lost to pandas period axes, overlapping labels.
- Allocation charts (top positions, sectors) draw a weight only while it is
  held: an unheld name is a gap, not a zero, so exits and entries are no
  longer vertical drops to the axis; one-session holdings are dotted, and the
  axis keeps zero.
