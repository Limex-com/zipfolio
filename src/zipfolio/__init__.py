"""Zipfolio: what a strategy does with its portfolio, position by position.

Part of the ZipLime ecosystem: the report ZipLime builds beside the QuantStats
tearsheet for every backtest, starting where a returns tearsheet stops. QuantStats and
its kind describe the return series; Zipfolio describes the book behind it --
positions and exposure, instruments and sectors, factor exposures, risk by
holding, how orders were executed and how far to believe those fills, trading
costs and the cost model, round trips, capacity -- as one self-contained HTML
page in the QuantStats Pro look.

It computes with pyfolio-reloaded, skfolio and statsmodels, reads the
Fama-French factors from Kenneth French's data library, and draws every chart
and table itself. Its inputs are the frames a ziplime run produces, as plain
pandas::

    from zipfolio import build_tearsheet

    build_tearsheet(
        "report.html",
        returns,                      # per-period strategy returns
        benchmark,                    # benchmark returns, or None
        position_records=positions,   # date, symbol, quantity, last_sale_price
        cash=ending_cash,             # cash at the end of each period
        transactions=fills,           # dt, symbol, quantity, price
        market_data=daily_bars,       # symbol, day, price, volume (optional)
        orders=orders,                # id, dt, symbol, commission (optional)
        cost_settings={...},          # commission / slippage settings (optional)
        fill_bars=fill_bars,          # OHLCV of each fill's bar, by fill id (optional)
        decision_prices=decisions,    # close at each order's creation, by order id (optional)
        factor_model="ff5_mom",       # or "ff5", "ff3", "ff3_mom", None
        sector_mappings={"AAPL": "Information Technology"},  # optional
    )

pyfolio and skfolio are imported when a page is built, not on import: a host
without skfolio loses that one section, not the report.
"""

__version__ = "0.1.0"

from .inputs import daily_returns, market_data_frame, positions_in_dollars, transactions_frame  # noqa: E402
from .page import DEFAULT_TITLE, REPOSITORY_URL  # noqa: E402
from .sectors import GICS_SECTORS, UNCLASSIFIED, US_MICS, us_sector_mappings  # noqa: E402
from .tearsheet import build_tearsheet  # noqa: E402

__all__ = [
    "DEFAULT_TITLE",
    "GICS_SECTORS",
    "REPOSITORY_URL",
    "UNCLASSIFIED",
    "US_MICS",
    "__version__",
    "build_tearsheet",
    "daily_returns",
    "market_data_frame",
    "positions_in_dollars",
    "transactions_frame",
    "us_sector_mappings",
]
