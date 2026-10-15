"""An independent ledger: the book replayed from its opening state, fill by fill and event by event.

It uses no pyfolio and no part of the report it checks. Fees are booked when they were charged: the
engine's fee model is run over the fills of each order (``model_fees``) and the result must match what
the order was charged; where it does, each fill carries its own fee. Where it does not, or no fee
model is given, an order's charge is booked on its session when all its fills fall in one, and the
sessions an order filled across are left out of the day-by-day cash comparison -- its cumulative
charge says nothing about which day paid what.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .models import EXTERNAL_EVENTS


@dataclass
class FeeBooking:
    """When each fee was paid, and how that is known."""

    by_session: pd.Series
    by_symbol: pd.Series
    order_totals: pd.Series
    method: str
    modeled_totals: pd.Series | None = None
    mismatches: list = field(default_factory=list)
    ambiguous_sessions: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([], tz="UTC"))


def model_fees(fills: pd.DataFrame, settings: dict | None) -> pd.Series | None:
    """Each fill's fee under the run's fee model, as ziplime computes it; ``None`` for an unknown model.

    Per dollar: the fill's value times the rate. Per share: the first fill of an order pays the larger of
    the minimum and its shares times the rate; later fills pay only once the order's running total passes
    the minimum, and then the difference (``ziplime.finance.commission.utils.calculate_per_unit_commission``).
    A fill without an order -- the engine closing a position itself -- pays nothing.
    """
    if not settings or fills is None:
        return None
    model = settings.get("commission_model")
    if model == "per_dollar":
        rate = settings.get("commission_rate")
        rate = settings.get("commission_per_share") if rate is None else rate
        if rate is None:
            return None
        fees = (fills["quantity"].abs() * fills["price"] * float(rate)).where(fills["order_id"].notna(), 0.0)
        return pd.Series(fees.to_numpy(dtype=float), index=fills["fill_id"])
    if model in (None, "per_share") and settings.get("commission_per_share") is not None:
        cost = float(settings["commission_per_share"])
        minimum = float(settings.get("commission_min_trade_cost") or 0.0)
        fees = pd.Series(0.0, index=fills["fill_id"])
        for order_id, own in fills[fills["order_id"].notna()].groupby("order_id", sort=False):
            filled, charged = 0.0, 0.0
            for fill in own.itertuples():
                extra = abs(fill.quantity * cost)
                if charged == 0:
                    fee = max(minimum, extra)
                else:
                    total = abs(filled * cost) + extra
                    fee = 0.0 if total < minimum else total - charged
                fees[fill.fill_id] = fee
                charged += fee
                filled += fill.quantity
        return fees
    return None


def book_fees(book, settings: dict | None, tolerance: float) -> FeeBooking | None:
    """When each order's charge was paid, by session and by symbol; ``None`` when no orders are given."""
    if book.orders is None:
        return None
    orders = book.orders.set_index("id")
    totals = orders["commission"]
    fills = book.fills if book.fills is not None else pd.DataFrame(
        columns=["fill_id", "order_id", "dt", "session", "symbol", "quantity", "price"])
    modeled = model_fees(fills, settings) if len(fills) else None
    by_session = pd.Series(0.0, index=book.sessions)
    by_symbol = pd.Series(dtype=float)
    mismatches = []
    modeled_totals = None
    if modeled is not None:
        modeled_totals = modeled.groupby(fills.set_index("fill_id")["order_id"]).sum()
        for order_id in totals.index.union(modeled_totals.index):
            charged = float(totals.get(order_id, 0.0))
            expected = float(modeled_totals.get(order_id, 0.0))
            if abs(charged - expected) > tolerance:
                mismatches.append({"order_id": str(order_id), "charged": charged, "fee_model": expected,
                                   "difference": charged - expected})
    if modeled is not None and not mismatches:
        per_fill = fills.assign(fee=modeled.to_numpy())
        by_session = by_session.add(per_fill.groupby("session")["fee"].sum(), fill_value=0.0).reindex(book.sessions,
                                                                                                    fill_value=0.0)
        by_symbol = per_fill.groupby("symbol")["fee"].sum()
        # Charges on orders that never filled are not in the fills; they are a mismatch above, so none here.
        return FeeBooking(by_session, by_symbol, totals, "fee_model", modeled_totals, mismatches)

    ambiguous = []
    sessions = fills.groupby("order_id")["session"].agg(["min", "max"]) if len(fills) else pd.DataFrame()
    for order_id, charge in totals.items():
        if not charge:
            continue
        symbol = orders.at[order_id, "symbol"]
        if order_id in sessions.index:
            first, last = sessions.loc[order_id, "min"], sessions.loc[order_id, "max"]
            if first != last:
                # Before its last fill some of the charge was paid and some not -- which, no record says.
                ambiguous += list(book.sessions[(book.sessions >= first) & (book.sessions < last)])
            day = last
        else:
            day = orders.at[order_id, "session"]
        if day in by_session.index:
            by_session[day] += charge
        by_symbol[symbol] = by_symbol.get(symbol, 0.0) + charge
    method = "order_session" if not ambiguous else "order_session_partly_ambiguous"
    return FeeBooking(by_session, by_symbol, totals, method, modeled_totals, mismatches,
                      pd.DatetimeIndex(sorted(set(ambiguous))))


@dataclass
class Replay:
    """The book replayed: cash and shares per session, and the PnL split first in, first out."""

    cash: pd.Series
    quantities: pd.DataFrame
    realized: float
    realized_by_symbol: dict
    unrealized_by_symbol: dict
    open_lots: int
    closing_fills: int
    income: float
    expenses: float
    external: pd.Series
    basis_unknown: set
    cash_events: pd.Series


def replay(book, opening, fees: FeeBooking | None, multipliers: dict) -> Replay:
    """Replay the fills, fees and events from ``opening``, session by session."""
    sessions = book.sessions
    fills = book.fills if book.fills is not None else pd.DataFrame(
        columns=["fill_id", "order_id", "dt", "session", "symbol", "quantity", "price"])
    events = book.events if book.events is not None else pd.DataFrame(
        columns=["dt", "session", "kind", "symbol", "cash", "quantity", "ratio"])
    symbols = sorted(set(book.symbols) | set(opening.positions) | set(events["symbol"].dropna().astype(str)))

    lots = {symbol: deque() for symbol in symbols}
    basis_unknown = set()
    for symbol, quantity in opening.positions.items():
        if not quantity:
            continue
        price = opening.cost_basis.get(symbol)
        if price is None:
            basis_unknown.add(symbol)
            price = opening.prices.get(symbol, np.nan)
        lots[symbol].append([float(quantity), float(price)])

    held = {symbol: float(opening.positions.get(symbol, 0.0)) for symbol in symbols}
    cash = float(opening.cash)
    realized, realized_by_symbol, closing_fills = 0.0, {}, 0
    income = expenses = 0.0
    external = pd.Series(0.0, index=sessions)
    cash_events = pd.Series(0.0, index=sessions)
    cash_path, held_path = [], []

    # By session, events first -- a split applies from the open of its day -- then the fills, in time order.
    stream = [(f.session, 1, f.dt, f) for f in fills.itertuples()] + [(e.session, 0, e.dt, e)
                                                                       for e in events.itertuples()]
    stream.sort(key=lambda item: (item[0], item[1], item[2]))
    position = 0
    fee_by_session = fees.by_session if fees is not None else pd.Series(0.0, index=sessions)
    for day in sessions:
        while position < len(stream) and stream[position][0] <= day:
            _, is_fill, _, item = stream[position]
            position += 1
            if is_fill:
                multiplier = multipliers.get(item.symbol, 1.0)
                cash -= item.quantity * item.price * multiplier
                held[item.symbol] = held.get(item.symbol, 0.0) + item.quantity
                quantity, closed_any = item.quantity, False
                book_lots = lots.setdefault(item.symbol, deque())
                while quantity and book_lots and np.sign(book_lots[0][0]) != np.sign(quantity):
                    lot = book_lots[0]
                    take = min(abs(quantity), abs(lot[0])) * np.sign(lot[0])
                    pnl = take * (item.price - lot[1]) * multiplier
                    realized += pnl
                    realized_by_symbol[item.symbol] = realized_by_symbol.get(item.symbol, 0.0) + pnl
                    lot[0] -= take
                    quantity += take
                    closed_any = True
                    if abs(lot[0]) < 1e-12:
                        book_lots.popleft()
                if abs(quantity) > 1e-12:
                    book_lots.append([quantity, item.price])
                closing_fills += closed_any
                continue
            kind, symbol = item.kind, (None if pd.isna(item.symbol) else str(item.symbol))
            amount = 0.0 if pd.isna(item.cash) else float(item.cash)
            cash += amount
            cash_events[day] += amount
            if kind in EXTERNAL_EVENTS:
                external[day] += amount
            elif amount > 0:
                income += amount
            else:
                expenses -= amount
            if kind == "split" and symbol is not None and not pd.isna(item.ratio):
                ratio = float(item.ratio)
                for lot in lots.setdefault(symbol, deque()):
                    lot[0] *= ratio
                    lot[1] /= ratio
                held[symbol] = held.get(symbol, 0.0) * ratio
            elif symbol is not None and not pd.isna(item.quantity) and item.quantity:
                held[symbol] = held.get(symbol, 0.0) + float(item.quantity)
                lots.setdefault(symbol, deque()).append([float(item.quantity), np.nan])
                basis_unknown.add(symbol)
        cash -= float(fee_by_session.get(day, 0.0))
        cash_path.append(cash)
        held_path.append(dict(held))

    final_prices = book.prices.iloc[-1] if book.prices is not None and len(book.prices) else pd.Series(dtype=float)
    unrealized_by_symbol = {}
    for symbol, symbol_lots in lots.items():
        if not symbol_lots:
            continue
        price = final_prices.get(symbol, np.nan)
        multiplier = multipliers.get(symbol, 1.0)
        unrealized_by_symbol[symbol] = float(sum(q * (price - p) * multiplier for q, p in symbol_lots))
    quantities = pd.DataFrame(held_path, index=sessions).reindex(columns=symbols).fillna(0.0)
    return Replay(pd.Series(cash_path, index=sessions), quantities, realized, realized_by_symbol,
                  unrealized_by_symbol, sum(len(v) for v in lots.values()), closing_fills, income, expenses,
                  external, basis_unknown, cash_events)
