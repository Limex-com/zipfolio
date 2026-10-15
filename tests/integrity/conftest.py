"""Books small enough to check by hand, recorded the way ziplime records one.

``ledger_book`` plays a list of fills and events through a cash account session by session, as the engine
does: fills move cash at their price, each fill's fee leaves cash on its own session, the order keeps a
running total of its charges -- repeated on every session the order was touched, as the engine's order
frame repeats it -- positions are snapshotted at the close and the return is the book's own, time-weighted
around external flows at the end of the day.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pytest

from zipfolio.integrity import AccountingInputs, ContractSpec, OpeningState


@dataclass
class Fill:
    session: int
    symbol: str
    quantity: float
    price: float
    order: str | None = None
    fee: float = 0.0


@dataclass
class Event:
    session: int
    kind: str
    symbol: str | None = None
    cash: float = 0.0
    quantity: float = 0.0
    ratio: float | None = None


@dataclass
class Book:
    prices: dict
    fills: list = field(default_factory=list)
    events: list = field(default_factory=list)
    capital: float = 10_000.0
    opening: dict = field(default_factory=dict)
    opening_basis: dict = field(default_factory=dict)


def sessions(count: int) -> pd.DatetimeIndex:
    """New York session closes, as the engine stamps its daily records."""
    return pd.bdate_range("2024-03-04", periods=count) + pd.Timedelta(hours=21)


def ledger_book(spec: Book, declare_events: bool | None = None, settings: dict | None = None,
                currency: str = "USD", specs: dict | None = None, opening: bool = True) -> AccountingInputs:
    """The inputs the engine would hand over for ``spec``; ``declare_events`` gives the event ledger
    (default: when there are events), ``specs`` replaces the cash-equity contract specs."""
    count = len(next(iter(spec.prices.values())))
    closes = sessions(count)
    held = {symbol: float(quantity) for symbol, quantity in spec.opening.items()}
    first_prices = {symbol: spec.prices[symbol][0] for symbol in held}
    cash = spec.capital
    start_value = cash + sum(q * spec.prices[s][0] for s, q in held.items())
    last = start_value
    charged, order_rows, fill_rows, records, cash_rows, returns, event_rows = {}, [], [], [], [], [], []
    for day, close in enumerate(closes):
        stamp = (close - pd.Timedelta(minutes=1)).tz_localize("UTC")
        flow = 0.0
        for event in [e for e in spec.events if e.session == day]:
            if event.kind == "split":
                held[event.symbol] = held.get(event.symbol, 0.0) * event.ratio
            elif event.symbol and event.quantity:
                held[event.symbol] = held.get(event.symbol, 0.0) + event.quantity
            cash += event.cash
            if event.kind in ("deposit", "withdrawal", "transfer_in", "transfer_out"):
                flow += event.cash
            event_rows.append({"dt": close.tz_localize("UTC") - pd.Timedelta(hours=8), "kind": event.kind,
                               "symbol": event.symbol, "cash": event.cash, "quantity": event.quantity,
                               "ratio": event.ratio})
        touched = []
        for number, fill in enumerate(spec.fills):
            if fill.session != day:
                continue
            cash -= fill.quantity * fill.price + fill.fee
            held[fill.symbol] = held.get(fill.symbol, 0.0) + fill.quantity
            order = fill.order or f"o{number}"
            charged[order] = charged.get(order, 0.0) + fill.fee
            touched.append((order, fill.symbol))
            fill_rows.append({"id": f"f{number}", "order_id": order, "dt": stamp, "symbol": fill.symbol,
                              "quantity": fill.quantity, "price": fill.price, "commission": None})
        for order, symbol in dict.fromkeys(touched):
            size = sum(f.quantity for f in spec.fills if (f.order or f"o{spec.fills.index(f)}") == order)
            order_rows.append({"id": order, "dt": stamp, "created": stamp, "type": "market", "symbol": symbol,
                               "quantity": size, "filled": size, "status": "filled", "commission": None})
        value = cash + sum(q * spec.prices[s][day] for s, q in held.items())
        returns.append((value - flow) / last - 1)
        last = value
        records += [{"date": close.strftime("%Y-%m-%d"), "symbol": s, "quantity": q, "cost_basis": 0.0,
                     "last_sale_price": spec.prices[s][day]} for s, q in held.items() if q]
        cash_rows.append(cash)
    orders = pd.DataFrame(order_rows, columns=["id", "dt", "created", "type", "symbol", "quantity", "filled",
                                              "status", "commission"])
    orders["commission"] = orders["id"].map(charged)  # every row carries the final total, as the engine's do
    symbols = set(spec.prices)
    events = None
    if declare_events or (declare_events is None and spec.events):
        events = pd.DataFrame(event_rows, columns=["dt", "kind", "symbol", "cash", "quantity", "ratio"])
    return AccountingInputs(
        returns=pd.Series(returns, index=closes), position_records=pd.DataFrame(
            records, columns=["date", "symbol", "quantity", "cost_basis", "last_sale_price"]),
        cash=pd.Series(cash_rows, index=closes),
        transactions=pd.DataFrame(fill_rows, columns=["id", "order_id", "dt", "symbol", "quantity", "price",
                                                      "commission"]),
        orders=orders, cost_settings=settings,
        opening_state=OpeningState(closes[0].normalize().tz_localize("UTC"), spec.capital, currency,
                                   dict(spec.opening), first_prices, dict(spec.opening_basis)) if opening else None,
        portfolio_events=events,
        contract_specs=specs if specs is not None else {s: ContractSpec("equity", 1.0, currency) for s in symbols},
        currency=currency, run_id="test")


@pytest.fixture
def pyfolio():
    return pytest.importorskip("pyfolio", reason="pyfolio-reloaded is not installed here")


def statuses(report) -> dict:
    return {check.id: check.status.value for check in report.checks}
