"""The report's currency: how money is written.

A backtest in New York is in dollars, one in London in pounds and one in Tokyo in
yen, and every amount on the page -- PnL, costs, fill sizes, capacity -- has
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
    #: Digits after the point in an ordinary amount: none for the yen or the won.
    minor_digits: int = 2

    def write(self, sign: str, body: str) -> str:
        """``$1,234.50``, ``CHF 1,234.50``, ``1,234.50 ₽``: the symbol where this currency puts it.

        A symbol made of letters -- an ISO code standing in for one -- is kept
        apart from the number by a space; a sign like ``$`` or ``£`` is not.
        """
        if self.symbol_after:
            return f"{sign}{body} {self.symbol}"
        gap = " " if self.symbol[-1].isalpha() and len(self.symbol) > 1 else ""
        return f"{sign}{self.symbol}{gap}{body}"


#: Every currency a page can be written in, by ISO 4217 code. Where several
#: share a sign, all but the dollar carry a prefix (C$, A$, HK$...), so a
#: report never leaves the reader to guess.
CURRENCIES = {currency.code: currency for currency in (
    Currency("USD", "$", False, 1e6),
    Currency("EUR", "€", False, 1e6),
    Currency("GBP", "£", False, 8e5),
    Currency("JPY", "¥", False, 1.5e8, 0),
    Currency("CHF", "CHF", False, 9e5),
    Currency("CAD", "C$", False, 1.4e6),
    Currency("AUD", "A$", False, 1.5e6),
    Currency("NZD", "NZ$", False, 1.7e6),
    Currency("HKD", "HK$", False, 8e6),
    Currency("SGD", "S$", False, 1.3e6),
    Currency("CNY", "CN¥", False, 7e6),
    Currency("INR", "₹", False, 9e7),
    Currency("KRW", "₩", False, 1.4e9, 0),
    Currency("TWD", "NT$", False, 3e7),
    Currency("SEK", "SEK", False, 1e7),
    Currency("NOK", "NOK", False, 1e7),
    Currency("DKK", "DKK", False, 7e6),
    Currency("PLN", "PLN", False, 4e6),
    Currency("BRL", "R$", False, 5e6),
    Currency("MXN", "MX$", False, 2e7),
    Currency("ZAR", "R", False, 2e7),
    Currency("ILS", "₪", False, 4e6),
    Currency("TRY", "₺", False, 4e7),
    Currency("RUB", "₽", True, 1e8),
)}

_current: ContextVar[Currency] = ContextVar("zipfolio_currency", default=CURRENCIES["USD"])


def currency() -> Currency:
    return _current.get()


@contextmanager
def using_currency(code: str | None):
    """Write money in ``code`` -- an ISO code in :data:`CURRENCIES` -- while the block runs."""
    chosen = CURRENCIES.get((code or "USD").upper())
    if chosen is None:
        raise ValueError(f"unknown currency {code!r}; one of {', '.join(sorted(CURRENCIES))}")
    token = _current.set(chosen)
    try:
        yield chosen
    finally:
        _current.reset(token)


def money(value: float, decimals: int | None = None) -> str:
    """``$1,234.50`` / ``-$5.00`` in dollars, ``¥1,235`` in yen, ``1,234.50 ₽`` in roubles.

    ``decimals`` defaults to the currency's own (two, none for the yen); pass
    one for a rounded amount (``0``) or a price per share (``4``).
    """
    current = currency()
    decimals = current.minor_digits if decimals is None else decimals
    sign = "-" if value < 0 else ""
    return current.write(sign, f"{abs(value):,.{decimals}f}")


def compact(value: float) -> str:
    """A short amount for the summary: ``$109.6k``, ``-2.7m ₽``, ``$950``."""
    sign = "-" if value < 0 else ""
    size = abs(value)
    body = (f"{size / 1e9:,.1f}bn" if size >= 1e9 else f"{size / 1e6:,.1f}m" if size >= 1e6
            else f"{size / 1e3:,.1f}k" if size >= 1e3 else f"{size:,.0f}")
    return currency().write(sign, body)


def money_formatter(decimals: int | None = None):
    """:func:`money` as a one-argument formatter, for a table column."""
    return lambda value: money(value, decimals)


def unit() -> str:
    """The symbol alone, for an axis label: ``"Cumulative PnL, $"``."""
    return currency().symbol


def millions(value: float) -> str:
    """A round amount in millions: ``$1m``, ``100m ₽``."""
    return currency().write("", f"{value / 1e6:,.0f}m")
