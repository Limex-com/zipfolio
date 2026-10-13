"""Shaping a backtest into the frames pyfolio and skfolio read.

Each function takes the plain frames a backtester already has -- per-period returns,
one row per open position per day, one row per fill, daily bars -- and returns the
shape the analytics expect: one row per session at midnight UTC, positions in
dollars with a ``cash`` column, fills indexed by time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns(returns: pd.Series) -> pd.Series:
    """Return ``returns`` as one compounded return per calendar day, indexed in UTC.

    pyfolio annualises with 252 periods and aligns returns, positions and
    transactions by normalised date, so it needs exactly one row per session.
    ziplime emits one already, but at the session close rather than at
    midnight, and an intraday emission would emit several -- compounding them
    is the only way that keeps the curve the same.

    UTC because the fills pyfolio lines up against these dates carry a zone, and
    pyfolio compares them as they are.
    """
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if returns.empty:
        return returns.astype(float)

    dates = utc_dates(returns.index)
    daily = (1.0 + returns.to_numpy()).astype(float)
    compounded = pd.Series(daily, index=dates).groupby(level=0).prod() - 1.0
    return compounded.rename(returns.name)


def positions_in_dollars(
    position_records: pd.DataFrame,
    cash: pd.Series,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame | None:
    """Build pyfolio's positions frame: one column per symbol in dollars, plus ``cash``.

    ``position_records`` is one row per open position per day, with ``date``,
    ``symbol``, ``quantity`` and ``last_sale_price`` -- what a ziplime run
    yields when its positions are snapshotted bar by bar. A day with no open
    position has no row, so it comes back as zeros rather than missing.

    The value is quantity times price. For a futures contract that leaves out
    the multiplier, which understates its exposure; the records carry no
    multiplier to correct it with.
    """
    if cash is None or cash.empty or len(dates) == 0:
        return None

    cash = pd.to_numeric(cash, errors="coerce")
    cash.index = utc_dates(cash.index)
    cash = cash[~cash.index.duplicated(keep="last")].reindex(dates).ffill()

    if position_records is None or position_records.empty:
        holdings = pd.DataFrame(index=dates)
    else:
        records = position_records.dropna(subset=["symbol"]).copy()
        records["date"] = utc_dates(records["date"])
        records["value"] = (
            pd.to_numeric(records["quantity"], errors="coerce")
            * pd.to_numeric(records["last_sale_price"], errors="coerce")
        )
        holdings = records.pivot_table(
            index="date", columns="symbol", values="value", aggfunc="sum",
        )
        holdings = holdings.reindex(dates).fillna(0.0)
        holdings.columns = [str(column) for column in holdings.columns]
        holdings.columns.name = None

    holdings["cash"] = cash.fillna(0.0)
    return holdings


def transactions_frame(transactions: pd.DataFrame) -> pd.DataFrame | None:
    """Build pyfolio's transactions frame from the flattened ziplime transactions.

    pyfolio wants the fill time as the index and ``amount``, ``price`` and
    ``symbol`` as columns; the rest of what ziplime records is dropped.
    """
    if transactions is None or transactions.empty:
        return None

    frame = pd.DataFrame({
        "amount": pd.to_numeric(transactions["quantity"], errors="coerce"),
        "price": pd.to_numeric(transactions["price"], errors="coerce"),
        "symbol": transactions["symbol"].astype(str),
    })
    frame.index = pd.DatetimeIndex(pd.to_datetime(transactions["dt"], errors="coerce", utc=True))
    frame = frame[frame.index.notna()].dropna(subset=["amount", "price"])
    frame = frame[frame["amount"] != 0].sort_index()
    return frame if not frame.empty else None


def market_data_frame(
    bars: pd.DataFrame | None,
    symbols,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame | None:
    """Build pyfolio's ``market_data``: rows ``(date, "price" | "volume")``, a column per symbol.

    ``bars`` is one row per instrument per session with ``symbol``, ``day``,
    ``price`` and ``volume`` -- what ``backtesting`` reads out of the market
    bundle the simulation ran on. Only the traded symbols are kept: the bundle
    also holds the benchmark and, for a universe, hundreds of names the
    strategy never bought.
    """
    if bars is None or bars.empty or not len(symbols):
        return None
    bars = bars[bars["symbol"].astype(str).isin(set(symbols))]
    if bars.empty:
        return None

    bars = bars.assign(date=utc_dates(bars["day"]))
    wide = bars.pivot_table(index="date", columns="symbol", values=["price", "volume"], aggfunc="last")
    wide = wide.reindex(index=dates)
    frame = wide.stack(level=0, future_stack=True)
    frame.index.names = ["date", "field"]
    frame.columns.name = None
    return frame.reindex(columns=sorted(set(symbols)))


def utc_dates(values) -> pd.DatetimeIndex:
    """Normalise timestamps to midnight UTC of their own calendar date.

    The date is taken in the timestamp's own zone first: a Moscow session
    closing at 18:50 local is that day, and so is a New York one at 16:00.
    """
    index = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))
    if index.tz is not None:
        index = index.tz_localize(None)
    return index.normalize().tz_localize("UTC")


def usable_benchmark(benchmark: pd.Series | None, dates: pd.DatetimeIndex) -> pd.Series | None:
    """Return the benchmark aligned to ``dates``, or ``None`` when there is none.

    A backtester often hands a run without a benchmark a series of zeros, so
    its own metrics have a column to fill. pyfolio would divide by its zero
    variance for beta and draw a flat line, so that series counts as no
    benchmark here.
    """
    if benchmark is None:
        return None
    benchmark = daily_returns(benchmark)
    if benchmark.empty or np.allclose(benchmark.to_numpy(), 0.0):
        return None
    return benchmark.reindex(dates).fillna(0.0)
