"""The report's currency: how money is written.

A US backtest is in dollars and a Moscow one in roubles, and every amount on the
page -- PnL, costs, fill sizes, capacity -- has
to say which. The currency is set for the duration of one page
(:func:`using_currency`) and read wherever money is written (:func:`money`).
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class Currency:
    code: str
    symbol: str
    #: Whether the symbol follows the amount, as roubles do ("1,250 ₽").
    symbol_after: bool
    #: About a million dollars in this currency. Kept for callers; the capacity
    #: analysis now measures at the strategy's own capital.
    capacity_base: float


CURRENCIES = {
    "USD": Currency("USD", "$", False, 1e6),
    "RUB": Currency("RUB", "₽", True, 1e8),
    "EUR": Currency("EUR", "€", False, 1e6),
}

_current: ContextVar[Currency] = ContextVar("zipfolio_currency", default=CURRENCIES["USD"])


def currency() -> Currency:
    return _current.get()


@contextmanager
def using_currency(code: str | None):
    """Write money in ``code`` (``"USD"``, ``"RUB"``, ``"EUR"``) while the block runs."""
    chosen = CURRENCIES.get((code or "USD").upper())
    if chosen is None:
        raise ValueError(f"unknown currency {code!r}; one of {sorted(CURRENCIES)}")
    token = _current.set(chosen)
    try:
        yield chosen
    finally:
        _current.reset(token)


def money(value: float, decimals: int = 2) -> str:
    """``$1,234.50`` / ``-$5.00`` in dollars, ``1,234.50 ₽`` / ``-5.00 ₽`` in roubles."""
    current = currency()
    sign = "-" if value < 0 else ""
    body = f"{abs(value):,.{decimals}f}"
    return f"{sign}{body} {current.symbol}" if current.symbol_after else f"{sign}{current.symbol}{body}"


def compact(value: float) -> str:
    """A short amount for the summary: ``$109.6k``, ``-2.7m ₽``, ``$950``."""
    current = currency()
    sign = "-" if value < 0 else ""
    size = abs(value)
    body = (f"{size / 1e9:,.1f}bn" if size >= 1e9 else f"{size / 1e6:,.1f}m" if size >= 1e6
            else f"{size / 1e3:,.1f}k" if size >= 1e3 else f"{size:,.0f}")
    return f"{sign}{body} {current.symbol}" if current.symbol_after else f"{sign}{current.symbol}{body}"


def money_formatter(decimals: int = 2):
    """:func:`money` as a one-argument formatter, for a table column."""
    return lambda value: money(value, decimals)


def unit() -> str:
    """The symbol alone, for an axis label: ``"Cumulative PnL, $"``."""
    return currency().symbol


def millions(value: float) -> str:
    """A round amount in millions: ``$1m``, ``100m ₽``."""
    current = currency()
    body = f"{value / 1e6:,.0f}m"
    return f"{body} {current.symbol}" if current.symbol_after else f"{current.symbol}{body}"
