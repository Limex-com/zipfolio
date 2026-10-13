"""What Zipfolio reads from a ziplime run that a return series cannot tell it.

A returns tearsheet sees one number per day. A ziplime run has much more, and
most of it is gone once the simulation ends: the market-data bundle it read,
each order with its execution style and creation time, each fill with the
order it belongs to and the bar it happened on. These helpers take what the
report needs while the run is still in memory, as plain frames:

* :func:`daily_bars` -- price and volume per instrument per session, for the
  capacity analysis;
* :func:`execution_bars` -- the OHLCV bar each fill happened on and the open of
  the bar after it, and the close of the bar each order was placed on, for the
  execution analysis and the next-bar repricing;
* :func:`order_type` -- ``market``, ``limit``, ``stop`` or ``stop_limit``, from
  an order's ziplime execution style.

They need ``polars`` (ziplime's frame library) only when called; the rest of
Zipfolio does not import this module.
"""

from __future__ import annotations

import traceback

import pandas as pd

ORDER_TYPES = {
    "MarketOrder": "market",
    "LimitOrder": "limit",
    "StopOrder": "stop",
    "StopLimitOrder": "stop_limit",
}

BAR_COLUMNS = ("open", "high", "low", "close", "volume")


def order_type(order) -> str:
    """``market``, ``limit``, ``stop`` or ``stop_limit``, from the order's ziplime execution style."""
    style = type(getattr(order, "execution_style", None)).__name__
    return ORDER_TYPES.get(style, style.lower() or "unknown")


def daily_bars(bundle) -> pd.DataFrame:
    """Price and volume per instrument per session, from the bundle the simulation read.

    Intraday bars are folded into sessions -- the last close, the summed
    volume -- so the frame is sessions x instruments whatever the emission
    rate, and the bundle itself need not outlive the simulation. Columns:
    ``symbol``, ``day``, ``price``, ``volume``.

    Empty, never raising, when the bundle does not carry what is needed; the
    tearsheet then leaves the capacity section out.
    """
    import polars as pl

    data = getattr(bundle, "data", None)
    required = {"symbol", "date", "close", "volume"}
    if data is None or not required <= set(data.columns):
        return pd.DataFrame()
    try:
        return (
            data.select(sorted(required))
            .sort("date")
            .group_by(["symbol", pl.col("date").dt.date().alias("day")])
            .agg(pl.col("close").last().alias("price"), pl.col("volume").sum())
            .to_pandas()
        )
    except Exception:
        print(f"Warning: no daily bars for the capacity analysis:\n{traceback.format_exc()}")
        return pd.DataFrame()


def execution_bars(bundle, fills: pd.DataFrame, orders: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """The bar each fill happened on, and the close of the bar each order was placed on.

    ``fills`` needs ``id``, ``symbol`` and ``dt``; ``orders`` needs ``id``,
    ``symbol`` and ``created``. Both are joined as of the simulation time --
    the latest bar at or before the fill, or the order's creation -- so a
    daily and an intraday bundle answer the same way. Returns
    ``{"fill_bars": OHLCV by fill id, "decision_prices": decision_price by order id}``;
    ``fill_bars`` also carries ``next_open`` and ``next_date``, the open and the
    time of the instrument's following bar -- where the fill would have landed
    had the order waited a bar. Both are empty on the last bar of the data.

    A fill priced outside its own bar's high-low range, or for more of the
    bar's volume than traded, is a fill the market would not have given; that
    is what the execution section checks with these. Empty frames, never
    raising, when the bundle does not carry the columns.
    """
    import polars as pl

    empty = {"fill_bars": pd.DataFrame(), "decision_prices": pd.DataFrame()}
    data = getattr(bundle, "data", None)
    required = {"symbol", "date", *BAR_COLUMNS}
    if data is None or not required <= set(data.columns) or fills is None or fills.empty:
        return empty
    try:
        symbols = sorted(set(fills["symbol"].dropna().astype(str)))
        bars = (data.select(sorted(required))
                .filter(pl.col("symbol").is_in(symbols))
                .with_columns(pl.col("date").dt.convert_time_zone("UTC").dt.cast_time_unit("us"))
                .sort("date")
                .with_columns(pl.col("open").shift(-1).over("symbol").alias("next_open"),
                              pl.col("date").shift(-1).over("symbol").alias("next_date")))

        def as_of(frame: pd.DataFrame, when: str) -> pd.DataFrame:
            keys = pl.from_pandas(pd.DataFrame({
                "key": frame["id"].astype(str),
                "symbol": frame["symbol"].astype(str),
                "at": pd.to_datetime(frame[when], utc=True),
            }).dropna(subset=["at"])).with_columns(pl.col("at").dt.cast_time_unit("us")).sort("at")
            joined = keys.join_asof(bars, left_on="at", right_on="date", by="symbol", strategy="backward")
            return joined.to_pandas().set_index("key")

        fill_bars = as_of(fills, "dt")[[*BAR_COLUMNS, "next_open", "next_date"]]
        decision = pd.DataFrame()
        if orders is not None and not orders.empty and "created" in orders.columns:
            first = orders.drop_duplicates("id", keep="first")
            decision = as_of(first, "created")[["close"]].rename(columns={"close": "decision_price"})
        return {"fill_bars": fill_bars, "decision_prices": decision}
    except Exception:
        print(f"Warning: no execution bars for the Zipfolio tearsheet:\n{traceback.format_exc()}")
        return empty
