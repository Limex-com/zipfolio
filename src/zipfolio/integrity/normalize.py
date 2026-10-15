"""The inputs in one shape: sessions as dates, fills in time order, one row per order.

Nothing is coerced silently. A value that cannot be read -- a price that is not a number, a fill with no
symbol, a cash balance missing for a session -- is kept out of the arithmetic and written down as an
issue, which the input checks report. The book is never repaired.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..inputs import utc_dates
from .models import EVENT_KINDS


@dataclass
class Book:
    """The normalized book. ``issues`` maps a check id to the problems found in its inputs."""

    sessions: pd.DatetimeIndex
    returns: pd.Series
    cash: pd.Series | None = None
    quantities: pd.DataFrame | None = None
    prices: pd.DataFrame | None = None
    fills: pd.DataFrame | None = None
    orders: pd.DataFrame | None = None
    order_rows: int = 0
    events: pd.DataFrame | None = None
    issues: dict = field(default_factory=dict)

    def problem(self, check: str, kind: str, **detail) -> None:
        self.issues.setdefault(check, []).append({"problem": kind, **detail})

    @property
    def symbols(self) -> list:
        names = set()
        if self.quantities is not None:
            names |= set(self.quantities.columns)
        if self.fills is not None:
            names |= set(self.fills["symbol"])
        return sorted(names)


def _session(values) -> pd.DatetimeIndex:
    return utc_dates(values)


def normalize(inputs) -> Book:
    from ..inputs import daily_returns

    returns = daily_returns(inputs.returns)
    book = Book(sessions=returns.index, returns=returns)
    raw = pd.to_numeric(inputs.returns, errors="coerce")
    if raw.isna().any() or not np.isfinite(raw.dropna()).all():
        book.problem("INPUT.TIMESTAMP_ALIGNMENT", "returns_not_numeric", rows=int(raw.isna().sum()))
    if not returns.index.is_monotonic_increasing:
        book.problem("INPUT.TIMESTAMP_ALIGNMENT", "returns_out_of_order")

    if inputs.cash is not None and len(inputs.cash):
        cash = pd.to_numeric(inputs.cash, errors="coerce")
        cash.index = _session(inputs.cash.index)
        repeated = cash.index.duplicated(keep=False)
        if repeated.any():
            disagree = cash[repeated].groupby(level=0).nunique()
            for day in disagree[disagree > 1].index:
                book.problem("INPUT.TIMESTAMP_ALIGNMENT", "cash_snapshot_repeated_with_different_values",
                             session=day.date().isoformat())
        cash = cash[~cash.index.duplicated(keep="last")]
        outside = cash.index.difference(book.sessions)
        if len(outside):
            book.problem("INPUT.TIMESTAMP_ALIGNMENT", "cash_on_a_day_with_no_return",
                         sessions=[day.date().isoformat() for day in outside[:5]], count=len(outside))
        book.cash = cash.reindex(book.sessions)
        missing = book.cash.index[book.cash.isna()]
        if len(missing):
            book.problem("INPUT.TIMESTAMP_ALIGNMENT", "cash_snapshot_missing",
                         sessions=[day.date().isoformat() for day in missing[:5]], count=len(missing))

    records = inputs.position_records
    if records is not None and book.cash is not None:
        if len(records):
            frame = records.dropna(subset=["symbol"]).reset_index(drop=True)
            frame = frame.assign(session=_session(frame["date"]), symbol=frame["symbol"].astype(str),
                                 quantity=pd.to_numeric(frame["quantity"], errors="coerce").astype(float),
                                 price=pd.to_numeric(frame["last_sale_price"], errors="coerce").astype(float))
            bad = ~np.isfinite(frame["quantity"]) | ~np.isfinite(frame["price"])
            if bad.any():
                book.problem("INPUT.TIMESTAMP_ALIGNMENT", "position_snapshot_not_numeric", rows=int(bad.sum()))
            outside = pd.DatetimeIndex(frame["session"].unique()).difference(book.sessions)
            if len(outside):
                book.problem("INPUT.TIMESTAMP_ALIGNMENT", "positions_on_a_day_with_no_return",
                             sessions=[day.date().isoformat() for day in outside[:5]], count=len(outside))
            frame = frame[~bad]
            book.quantities = (frame.pivot_table(index="session", columns="symbol", values="quantity", aggfunc="sum")
                               .reindex(book.sessions).fillna(0.0))
            book.prices = frame.pivot_table(index="session", columns="symbol", values="price",
                                            aggfunc="last").reindex(book.sessions)
        else:
            book.quantities = pd.DataFrame(index=book.sessions, dtype=float)
            book.prices = pd.DataFrame(index=book.sessions, dtype=float)

    fills = inputs.transactions
    if fills is not None and len(fills):
        frame = fills.copy()
        if "quantity" not in frame and "amount" in frame:
            frame["quantity"] = frame["amount"]
        if "dt" not in frame:
            frame["dt"] = frame.index
        frame = frame.reset_index(drop=True)
        frame["dt"] = pd.to_datetime(frame["dt"], utc=True, errors="coerce")
        frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce").astype(float)
        frame["price"] = pd.to_numeric(frame["price"], errors="coerce").astype(float)
        frame["symbol"] = frame["symbol"].astype("string")
        frame["fill_id"] = frame["id"].astype(str) if "id" in frame else frame.index.astype(str)
        frame["order_id"] = frame["order_id"].astype("string") if "order_id" in frame else pd.NA
        if "id" not in fills:
            book.problem("INPUT.FILL_UNIQUENESS", "fills_have_no_id")
        duplicated = frame["fill_id"].duplicated(keep=False) & ("id" in fills)
        for fill_id in frame.loc[duplicated, "fill_id"].unique()[:10]:
            book.problem("INPUT.FILL_UNIQUENESS", "duplicate_fill_id", fill_id=str(fill_id))
        invalid = (frame["dt"].isna() | frame["symbol"].isna() | ~np.isfinite(frame["quantity"])
                   | ~np.isfinite(frame["price"]) | ~(frame["price"] > 0))
        for row in frame[invalid].head(10).itertuples():
            book.problem("INPUT.FILL_UNIQUENESS", "invalid_fill", fill_id=str(row.fill_id),
                         quantity=row.quantity, price=row.price)
        zero = ~invalid & (frame["quantity"] == 0)
        if zero.any():
            book.problem("INPUT.FILL_UNIQUENESS", "zero_quantity_fill", count=int(zero.sum()),
                         fill_ids=[str(f) for f in frame.loc[zero, "fill_id"].head(5)])
        frame = frame[~invalid & ~zero]
        frame["session"] = _session(frame["dt"])
        frame["symbol"] = frame["symbol"].astype(str)
        outside = pd.DatetimeIndex(frame["session"].unique()).difference(book.sessions)
        if len(outside):
            book.problem("INPUT.TIMESTAMP_ALIGNMENT", "fill_on_a_day_with_no_return",
                         sessions=[day.date().isoformat() for day in outside[:5]], count=len(outside))
        book.fills = frame.sort_values(["dt", "fill_id"], kind="stable").reset_index(drop=True)[
            ["fill_id", "order_id", "dt", "session", "symbol", "quantity", "price"]]

    orders = inputs.orders
    if orders is not None and len(orders) and "commission" in orders.columns:
        frame = orders.copy()
        frame["commission"] = pd.to_numeric(frame["commission"], errors="coerce").fillna(0.0)
        book.order_rows = len(frame)
        if "id" in frame:
            conflicts = frame.groupby(frame["id"].astype(str))["commission"].nunique()
            for order_id in conflicts[conflicts > 1].index[:10]:
                book.problem("COST.COMMISSION_RECONCILIATION", "order_rows_disagree_on_commission",
                             order_id=str(order_id))
            frame = frame.drop_duplicates("id", keep="last")
            frame["id"] = frame["id"].astype(str)
        else:
            frame["id"] = frame.index.astype(str)
        frame["symbol"] = frame["symbol"].astype(str) if "symbol" in frame else ""
        frame["session"] = _session(frame["dt"]) if "dt" in frame else pd.NaT
        book.orders = frame[["id", "symbol", "session", "commission"]].reset_index(drop=True)

    events = inputs.portfolio_events
    if events is not None and len(events):
        frame = events.copy()
        frame["dt"] = pd.to_datetime(frame["dt"], utc=True, errors="coerce")
        frame["kind"] = frame["kind"].astype(str).str.lower()
        for column in ("cash", "quantity", "ratio"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce") if column in frame else np.nan
        frame["symbol"] = frame["symbol"].astype("string") if "symbol" in frame else pd.NA
        unknown = ~frame["kind"].isin(EVENT_KINDS)
        if unknown.any():
            book.problem("INPUT.TIMESTAMP_ALIGNMENT", "unknown_event_kind",
                         kinds=sorted(frame.loc[unknown, "kind"].unique()))
        frame["session"] = _session(frame["dt"])
        book.events = frame.sort_values("dt", kind="stable").reset_index(drop=True)
    return book
