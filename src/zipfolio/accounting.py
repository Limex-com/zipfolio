"""Where the profit is: realized, unrealized, commission and other cash, reconciled with the portfolio's value.

Every profit figure on the page is one of these, with one definition each:

* **trading PnL**, before commission: what the fills and the final book add up to,
  ``sum(final position values) - sum(quantity x fill price)``. Slippage is inside it,
  because it is inside the fill prices -- it is never taken off a second time;
* **realized PnL**: the part of it closed trades made -- pyfolio's round trips, built from
  the fills alone, first in, first out. A position still held at the end is not a trade;
* **unrealized PnL**: the rest, ``trading - realized`` -- the shares still held, at their
  last price, over what they cost, first in, first out;
* **commission**: what the orders were charged (ziplime puts it on the order);
* **other cash**: the change in the book that neither the fills nor the commissions nor the
  external flows explain -- dividends, interest, fees. Zero when the engine pays none;
* **external flows**: money put in or taken out (deposits, withdrawals), which are not profit;
* **net profit**: ``realized + unrealized - commission + other cash``, which is the change in
  the book's value (cash plus positions) less the external flows, and is checked against the
  value the daily returns imply -- the start value compounded by them, each flow added at the
  end of its day -- less the same flows; the two come from different records of the engine.

The book is taken to start in cash, as a backtest does.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ProfitBreakdown:
    """The profit of a backtest, split by where it came from; amounts in the report's currency."""

    realized: float
    unrealized: float
    commission: float | None
    other_cash: float
    start_value: float
    flows: float
    end_value: float
    compounded_end: float
    closed_trades: int
    open_names: int

    @property
    def trading(self) -> float:
        """Realized plus unrealized: the PnL before commission."""
        return self.realized + self.unrealized

    @property
    def net(self) -> float:
        return self.trading - (self.commission or 0.0) + self.other_cash

    @property
    def from_returns(self) -> float:
        """The net profit the daily returns imply: the value they compound to, less the start and the flows."""
        return self.compounded_end - self.start_value - self.flows

    @property
    def returns_gap(self) -> float:
        """The book's net profit less the one the daily returns imply: zero when the two records agree."""
        return self.net - self.from_returns

    @property
    def tolerance(self) -> float:
        """A cent, or one part in a hundred million of the capital: what compounding a few thousand returns
        can leave over in floating point."""
        return max(0.01, abs(self.start_value) * 1e-8)


def order_commission(orders: pd.DataFrame | None) -> float | None:
    """What the orders were charged, each order once; ``None`` when the orders are not given."""
    if orders is None or orders.empty or "commission" not in orders.columns:
        return None
    charged = orders.drop_duplicates("id", keep="last") if "id" in orders.columns else orders
    return float(pd.to_numeric(charged["commission"], errors="coerce").fillna(0.0).sum())


def profit_breakdown(returns: pd.Series, holdings: pd.DataFrame, transactions: pd.DataFrame,
                     orders: pd.DataFrame | None, closed: pd.DataFrame | None,
                     flows: pd.Series | None = None) -> ProfitBreakdown:
    """The profit split into realized, unrealized, commission and other cash, see the module.

    ``holdings`` is the book as recorded -- money per symbol and ``cash`` per session;
    ``transactions`` the fills (``amount``, ``price``); ``closed`` the closed trades
    (``round_trips.extract_round_trips``), or ``None`` when there are none; ``flows`` the external
    cash per session -- in positive, out negative -- when there was any (``external_flows``).
    """
    book = holdings.drop(columns="cash", errors="ignore")
    final_values = float(book.iloc[-1].sum())
    spent = float((transactions["amount"] * transactions["price"]).sum()) if len(transactions) else 0.0
    trading = final_values - spent
    realized = float(closed["pnl"].sum()) if closed is not None and len(closed) else 0.0
    commission = order_commission(orders)
    value = holdings.sum(axis="columns")
    flows = (flows if flows is not None else pd.Series(dtype=float)).reindex(returns.index, fill_value=0.0)
    start_value = float((value.iloc[0] - flows.iloc[0]) / (1 + returns.iloc[0]))
    end_value = float(value.iloc[-1])
    other = (end_value - start_value - float(flows.sum())) - (trading - (commission or 0.0))
    implied = start_value
    for r, flow in zip(returns.to_numpy(), flows.to_numpy()):
        implied = implied * (1 + r) + flow
    return ProfitBreakdown(
        realized=realized, unrealized=trading - realized, commission=commission, other_cash=other,
        start_value=start_value, flows=float(flows.sum()), end_value=end_value, compounded_end=implied,
        closed_trades=0 if closed is None else len(closed),
        open_names=int((book.iloc[-1].abs() > 0).sum()))


#: Event kinds that move money in or out of the account rather than make or lose it.
EXTERNAL = ("deposit", "withdrawal", "transfer_in", "transfer_out")


def external_flows(events: pd.DataFrame | None, dates: pd.DatetimeIndex) -> pd.Series | None:
    """External cash per session from an event ledger (``dt``, ``kind``, ``cash``); ``None`` when there is none."""
    if events is None or not len(events) or "kind" not in events:
        return None
    from .inputs import utc_dates

    own = events[events["kind"].astype(str).str.lower().isin(EXTERNAL)]
    if own.empty:
        return None
    cash = pd.to_numeric(own["cash"], errors="coerce").fillna(0.0)
    return cash.groupby(utc_dates(pd.to_datetime(own["dt"], utc=True))).sum().reindex(dates, fill_value=0.0)
