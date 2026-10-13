# Changelog

## 0.1.0 -- 2026-10-13

First release, extracted from the ZipLime backtester, where it builds the
report next to the QuantStats tearsheet of every backtest.

- `build_tearsheet`: one self-contained HTML page in the QuantStats Pro look --
  Performance, Positions, Instruments, Sector Analysis, Factor Analysis,
  Portfolio Risk, Transactions, Execution, Trading Costs, Round Trips, Capacity.
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
