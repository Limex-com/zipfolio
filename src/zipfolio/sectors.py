"""Sectors for the sector analysis: GICS for US tickers, MOEX sector indices for Moscow.

* **US** -- GICS, from a table the user supplies (:func:`us_sector_mappings`):
  the classification is licensed by MSCI and S&P, so no codes ship here.
* **Moscow** -- the Moscow Exchange's own sector indices, MOEXOG for oil and
  gas, MOEXFN for financials and so on (:func:`moex_sector_mappings`). Their
  constituents are public; they are read from the exchange's ISS service when
  a report is built and cached for a week.
"""

from __future__ import annotations

import csv
import json
import os
import time
import urllib.request
from functools import lru_cache
from pathlib import Path

import pandas as pd

#: GICS sectors by the first two digits of a GICS code.
GICS_SECTORS = {
    "10": "Energy",
    "15": "Materials",
    "20": "Industrials",
    "25": "Consumer Discretionary",
    "30": "Consumer Staples",
    "35": "Health Care",
    "40": "Financials",
    "45": "Information Technology",
    "50": "Communication Services",
    "55": "Utilities",
    "60": "Real Estate",
}


#: The venues a US GICS table describes. A ticker is looked up there only
#: when it trades on one of them: the table is keyed by ticker, and Moscow's
#: `T` or `MAGN` would otherwise be filed under an American company's sector.
US_MICS = frozenset({"XNYS", "XNGS", "XNAS", "XNMS", "XNCM", "ARCX", "XASE", "BATS", "IEXG"})


UNCLASSIFIED = "Unclassified"


def us_sector_mappings(ticker_mics, table: str | Path | None = None) -> dict[str, str]:
    """Return ``{ticker: GICS sector}`` for the US listings among ``ticker_mics``.

    Keyed by the plain ticker because that is how positions and transactions
    name an instrument by the time they reach pyfolio. A listing outside the US
    venues, or one the table does not know -- an ETF, a recent IPO -- is left
    out; the sector section files it as unclassified.

    Zipfolio ships no sector data: GICS is licensed by MSCI and S&P, and the
    codes are the user's to bring. ``table`` is a CSV of ``symbol,gics_sector``
    (the two-digit GICS sector code), found in this order: the argument, the
    ``ZIPFOLIO_GICS_TABLE`` environment variable, a ``data/us_gics_sectors.csv``
    inside the package for private builds. With none, the mapping is empty and
    the report has no sector section. ``scripts/build_us_sector_map.py`` builds
    a table from exports that carry GICS codes.
    """
    table = _gics_table(_table_path(table))
    mappings = {}
    for ticker_mic in ticker_mics or []:
        ticker, _, mic = str(ticker_mic).partition("@")
        if mic in US_MICS and ticker in table:
            mappings[ticker] = table[ticker]
    return mappings


def _table_path(table: str | Path | None) -> Path | None:
    for candidate in (table, os.getenv("ZIPFOLIO_GICS_TABLE"), Path(__file__).parent / "data" / "us_gics_sectors.csv"):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


@lru_cache(maxsize=4)
def _gics_table(path: Path | None) -> dict[str, str]:
    """``{ticker: sector name}`` from a ``symbol,gics_sector`` CSV, or nothing without one."""
    if path is None:
        print("No GICS sector table (see zipfolio.us_sector_mappings); the report has no sector section.")
        return {}
    try:
        with path.open(newline="") as handle:
            return {
                row["symbol"]: GICS_SECTORS.get(row["gics_sector"], UNCLASSIFIED)
                for row in csv.DictReader(handle)
            }
    except (OSError, KeyError) as error:
        print(f"Warning: could not read the sector table {path}: {error}")
        return {}


def sectors_of(positions: pd.DataFrame, sector_mappings: dict[str, str] | None) -> dict[str, str]:
    """Map every held symbol to a sector, or return nothing when none of them has one.

    A held name the table does not know is filed as unclassified rather than
    left out: pyfolio drops an unmapped symbol from the allocation altogether,
    and the sectors would then add up to less than the book.
    """
    held = [column for column in positions.columns if column != "cash"]
    if not sector_mappings or not any(symbol in sector_mappings for symbol in held):
        return {}
    return {symbol: sector_mappings.get(symbol, UNCLASSIFIED) for symbol in held}


#: The Moscow Exchange's sector indices and the sector each stands for.
MOEX_SECTOR_INDICES = {
    "MOEXOG": "Oil & Gas",
    "MOEXEU": "Electric Utilities",
    "MOEXTL": "Telecommunications",
    "MOEXMM": "Metals & Mining",
    "MOEXFN": "Financials",
    "MOEXCN": "Consumer",
    "MOEXCH": "Chemicals",
    "MOEXTN": "Transportation",
    "MOEXIT": "Information Technology",
    "MOEXRE": "Real Estate",
}
MOEX_MICS = frozenset({"MISX"})
MOEX_ISS_URL = "https://iss.moex.com/iss/statistics/engines/stock/markets/index/analytics/{index}.json"
MOEX_SECTORS_MAX_AGE = 7 * 24 * 3600


def moex_sector_mappings(ticker_mics, cache_dir: str | Path | None = None, timeout: float = 8.0) -> dict[str, str]:
    """Return ``{ticker: sector}`` for the Moscow listings among ``ticker_mics``.

    The sector is the Moscow Exchange sector index the ticker belongs to today:
    membership is current, not as of the backtest, so a company that changed
    sector, or left every sector index, is filed by where it is now -- or, if
    nowhere, as unclassified. With the exchange unreachable and nothing cached,
    the mapping is empty and the report has no sector section.
    """
    table = _moex_table(cache_dir, timeout)
    mappings = {}
    for ticker_mic in ticker_mics or []:
        ticker, _, mic = str(ticker_mic).partition("@")
        if mic in MOEX_MICS and ticker in table:
            mappings[ticker] = table[ticker]
    return mappings


def _moex_table(cache_dir, timeout: float) -> dict[str, str]:
    from .factors import _cache_dir, _store

    path = _cache_dir(cache_dir) / "moex_sectors.json"
    if path.exists() and time.time() - path.stat().st_mtime < MOEX_SECTORS_MAX_AGE:
        return json.loads(path.read_text())
    table: dict[str, str] = {}
    try:
        for index, sector in MOEX_SECTOR_INDICES.items():
            url = MOEX_ISS_URL.format(index=index) + "?iss.meta=off&limit=100"
            with urllib.request.urlopen(url, timeout=timeout) as response:
                analytics = json.load(response)["analytics"]
            column = analytics["columns"].index("ticker")
            for row in analytics["data"]:
                table.setdefault(str(row[column]), sector)
    except Exception as error:
        if path.exists():
            print(f"Warning: could not refresh the MOEX sector indices ({error}); using the cached copy.")
            return json.loads(path.read_text())
        print(f"Warning: no MOEX sector indices ({error}); the report has no sector section.")
        return {}
    _store(path, json.dumps(table, ensure_ascii=False).encode("utf-8"))
    return table
