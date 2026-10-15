# Changelog

## 0.3.0 -- unreleased

Interactive charts, a scorecard that counts problems rather than rows, and numbers
behind it that add up.

- **Interactive charts.** Every chart is drawn with matplotlib as before, then
  read back into an Apache ECharts chart (`zipfolio.interactive`): tooltips,
  legend toggles, zoom on dates, the same limits, ticks and colours. ECharts 5.6 (Apache-2.0) is inlined; the page makes no external
  requests. A chart the converter cannot read stays SVG; `interactive=False`
  keeps all of them static. Charts are drawn as their part is opened.
- **CAPM and the Efficient Frontier**, a section in part II and not a check:
  the security market line (the strategy and each traded name by beta, Jensen's
  alpha as the gap to the line), the capital market line, and the long-only
  efficient frontier of the names held most (skfolio, at most 30, and no more
  than a third of the sessions) with the max-Sharpe and minimum-variance mixes,
  equal weight and the strategy's average weights held fixed. Tables: CAPM
  measures (beta, R², systematic and specific volatility, the return CAPM asks,
  Jensen's alpha, Treynor, M², tracking error, information ratio) and the fixed
  mixes. The frontier is said to be hindsight. Two facts go to the JSON context,
  and the buy-and-hold check quotes the frontier -- the strategy, equal weight
  and the best fixed mix of the same names -- with a link to the section.
- **Save or copy any chart** from a menu in its corner: copy the image, or save
  it as PNG, JPG or SVG. Every format is made from the chart's SVG, so the file
  is what is on screen, zoom and legend toggles included. ECharts' own save
  button is gone (it saved SVG under a PNG label).
- **Open a chart larger**: an icon beside the save menu opens the chart over
  the page, as wide as the window, laid out again for that width, with zoom by
  wheel or pinch and panning by drag on both axes, and a reset; Esc closes it.
- **Copy any table** for a spreadsheet: an icon over every table, and over the
  scorecard, puts both tab-separated text and an HTML table on the clipboard;
  numbers carry `x:num` and `mso-number-format`, so Excel stores 23.07% as
  0.2307 whatever its regional settings.
- **Hints on terms**: a table's row or column label that is in the glossary
  (`zipfolio.glossary`, 76 terms -- Sharpe, CVaR, CDaR, Treynor, M²,
  tracking error, PSR, capture...) explains itself on hover or tap. The facts at
  a glance do too, where they had a `title` a phone could not reach.
- **Copy report for an AI.** A button in the header copies the report's numbers
  as Markdown -- the run, every metric and its value, every table, the titles
  of the charts, and definitions of the less common measures -- to paste into
  the reader's own assistant; another saves it as a `.md` file. No opinions:
  no verdicts, score, reasoning, thresholds or advice, so the assistant judges
  the numbers itself; checks phrased as questions go by their measure
  (`page.METRIC_NAMES`, `data-metric` on check lines). Put together in the
  browser from what the page shows: no holdings, fills or chart data beyond it.
- **Titles explain themselves**: every section heading and table title has a
  one-line hint (`glossary.SECTIONS`, `glossary.TABLES`); a factor's bar says
  which way its beta leans -- "leans to growth: expensive companies, priced
  high for their book value" -- with its interval and whether it counts.
- **No tables of nothing.** Capacity's four lists -- over a day to liquidate,
  fills over 1% of volume, each for the whole backtest and the last 126
  sessions -- appear only with rows in them; when a list is empty, one line
  with a check mark says so.
- **Moscow**: in a period when the benchmark made less than the rate, the CAPM
  section says why its lines slope down, and why there is no max-Sharpe mix
  when no mix of the names beat cash.
- **Phones.** Charts lay themselves out for the width they get, on every
  resize: legends wrap instead of paging, titles shrink to fit, a wide gutter of
  names is cut to a third of the chart, crowded labels turn 45 degrees, the
  treemap grows taller. The scorecard becomes cards, so an opened check reads
  across the screen. Wide tables keep their first column in place and shade
  the side there is more to scroll to; numbers and dates do not break. The jump
  bar keeps "Expand all" in reach; tap targets are larger.
- **Ticks.** Percent ticks carry the decimals their spacing needs (-2.5%, not
  -3%); large plain numbers have thousands separators; the correlation heatmap
  names every column.
- **Treemap of sectors and names** in Sector Analysis, after finviz's market
  map: a tile per name inside its sector, sized by its average share of the
  portfolio, coloured by its contribution to the return, the numbers on hover.
  Drawn natively in ECharts; squarified in matplotlib when charts are static.
- **Style box** in Factor Analysis: Morningstar's 3x3 grid of size by style, the
  strategy placed by its SMB and HML betas, with its 95% interval, its rolling
  path a point a month, and the benchmark; the style is a fact at a glance.
- **16 checks, four per part, families as one row.** Checks that measure one
  fact are one row with the worst of their verdicts, opened by a tap: *more
  than luck* (PSR or deflated Sharpe, luck test, alpha), *rests on a few days
  or trades* (best days, best trades), *concentration in one name* (largest
  position, top two names, risk from the top name, largest sector). A check
  that does not apply is N/A and out of the count. A row opens to its
  reasoning, threshold, what to try and a link to the numbers.
- **Less to read.** The summary is a one-sentence verdict, four facts at a
  glance (time in the market, style, how far below its high it ended, the
  extra cost that would erase the profit, capacity), the score and the table. In the
  sections, the explanations wait behind "How to read this section".
- **New checks**: against buying and holding the same names; whether the Sharpe
  ratio is more than luck (probabilistic Sharpe ratio, deflated when `trials`
  -- the versions tried -- is given); enough closed trades (30); the longest time
  below a peak; profit still open; max gross exposure.
- **Severity by size**: five trades or days whose removal turns the result into
  a loss are critical; fills outside their bar are judged by the money and by
  whom they favour -- slippage past a bar that closed at its high only costs the
  strategy.
- **Max drawdown against the benchmark at the same exposure** for a strategy
  that is in the market part of the time.
- **Accounting integrity check** (`zipfolio.integrity`, schema `accounting-integrity/v1`): an
  independent ledger -- no pyfolio -- replays the book from its opening state (fills, fees booked by
  the run's own fee model and matched to every order's charge, cash, positions, dividends, flows,
  splits) and checks every profit figure the page publishes against it: 16 checks with stable ids,
  `PASS`/`FAIL`/`WARN`/`SKIPPED`/`NOT_APPLICABLE`, an overall `PASS`/`WARN`/`FAIL`/`INCOMPLETE` and
  an assurance level (`summary`, `reconciled`, `ledger`). Deterministic JSON, `strict` and `ci`
  policies, `python -m zipfolio.integrity`. A panel above the summary shows it with the evidence;
  `zipfolio.ziplime.opening_state` and `contract_specs` give a ziplime run what the replay needs.
  Missing evidence is `SKIPPED`, never a pass. The momentum, AAPL and Moscow examples pass all 16 at
  ledger level. Found on the way: Profit by Sector dropped names with no sector and names held only
  within a day (now an Unclassified bucket and every traded name); the Profit Reconciliation counted
  a deposit as profit (external flows are now taken out when `portfolio_events` gives them).
- **Accounting audited and reconciled** (`zipfolio.accounting`). Closed trades are
  the closing fills only: pyfolio's tear sheet adds a sale of everything still held
  at the last price and counts it as a trade, and dated it one second after the
  last position snapshot -- here midnight before the last session's own fills, so a
  name bought that day came out as a closed short. What is still held is now
  unrealized PnL, counted apart. A **Profit Reconciliation** table in Performance
  splits the profit into realized, unrealized, commission and other cash, equal to
  the change in cash plus positions, and checks it against the start value
  compounded by the daily returns; a gap or unexplained cash is shown on the page,
  never absorbed. Trade PnL is labelled before commission (it was "net");
  pyfolio's "Total profit" and "Avg. trade net profit" read "Total PnL, before
  commission" and "Avg. trade PnL"; Net PnL and Contribution say what they add up
  to. Momentum example: 79 "closed trades" were 76 closed and 3 still held.
- **Fixes.** Profit by sector is counted as the treemap counts it -- net PnL and
  contribution, open positions marked -- instead of pyfolio's share of the closed
  trades' PnL, which turned a sector that made money negative whenever the total
  was a loss; the same share of a loss is left out of Instruments and Long and
  Short. Next-Bar Fills adds the fills' differences to the book instead of
  compounding them, so the totals reconcile, and signs a gain as a gain.
  Turnover is traded value over portfolio value (pyfolio's average gross book
  counted a purchase from cash twice). One historical VaR. The Sharpe ratio
  over the risk-free rate the factor analysis uses, beside the 0% one, and
  interest on idle cash said to be not credited. Market regimes leave out
  partial months. Realized volatility in Forecast against Outcome counts only
  invested days. Capacity at the strategy's own capital and the run's volume
  limit, with one number for it.
- **Key Performance Metrics** are Zipfolio's own: strategy, buy and hold, and
  benchmark side by side, each method named; first half against second half.
- **One name**: no breakdown by name or sector.
- **The page.** Parts start open under their verdicts (charts still draw as
  they come into view), a button floats back to the summary once it is out of
  sight; the header lists the
  run (`run_info`, then capital, fills, commission); one date format; static
  charts carry a title and no Matplotlib metadata, with stable ids; the
  scorecard is embedded as JSON (`#zipfolio-scorecard`) for agents.
- `build_tearsheet` takes `run_info`, `trials` and `interactive`.
  `Section.finding` takes `threshold`, `advice` and `issue`.
- **Documentation.** A new README: a live example report (a US momentum rotation, and the same
  backtest in QuantStats) served from `docs/` on GitHub Pages, screenshots, a comparison with
  QuantStats, and which inputs turn on which sections. The detail moved to `docs/REFERENCE.md`.
  `examples/quickstart.py` builds a report from the Momentum Rotation book in one command.

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
