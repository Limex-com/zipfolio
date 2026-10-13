"""Factor returns for the factor analysis: Fama-French for US books, a market model elsewhere.

* **Fama-French** daily factors come from Kenneth French's data library, which
  publishes each set as a zipped CSV of percentages, updated monthly from
  CRSP. This is what ``pandas_datareader.famafrench`` reads, without the
  dependency: one request per file, cached for a week.
* **A market model** -- the strategy against its own benchmark over a
  risk-free rate -- is what a market without published factors gets: Moscow
  against IMOEX over RUONIA, for one. :func:`market_factors` builds it;
  :func:`ruonia` reads the rate from the Bank of Russia.
"""

from __future__ import annotations

import io
import os
import re
import time
import urllib.request
import xml.etree.ElementTree as ElementTree
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

LIBRARY_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"

FIVE_FACTORS = "F-F_Research_Data_5_Factors_2x3_daily"
THREE_FACTORS = "F-F_Research_Data_Factors_daily"
MOMENTUM = "F-F_Momentum_Factor_daily"

#: Model name -> (description, the library files it is made of).
MODELS = {
    "ff3": ("Fama-French 3 factors", (THREE_FACTORS,)),
    "ff3_mom": ("Carhart 4 factors (Fama-French 3 + momentum)", (THREE_FACTORS, MOMENTUM)),
    "ff5": ("Fama-French 5 factors", (FIVE_FACTORS,)),
    "ff5_mom": ("Fama-French 5 factors + momentum", (FIVE_FACTORS, MOMENTUM)),
}

#: What each factor is long and short of, for the page's glossary.
FACTOR_DESCRIPTIONS = {
    "Mkt-RF": "the US stock market over the risk-free rate",
    "SMB": "size: small companies minus big ones",
    "HML": "value: high book-to-market (cheap) minus low (expensive)",
    "RMW": "profitability: robust operating profitability minus weak",
    "CMA": "investment: companies that invest conservatively minus aggressively",
    "MOM": "momentum: the past year's winners minus its losers",
}

CACHE_MAX_AGE = 7 * 24 * 3600

#: The Bank of Russia's web service; RUONIA is its overnight rouble rate.
CBR_SERVICE = "https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx"
RUONIA_SINCE = date(2010, 1, 1)
#: RUONIA is published every business day, so a day-old copy is fresh enough.
RUONIA_MAX_AGE = 24 * 3600


def fama_french_daily(model: str = "ff5_mom", cache_dir: str | Path | None = None,
                      timeout: float = 8.0) -> pd.DataFrame:
    """Daily factor returns as decimals, one row per day at midnight UTC, plus ``RF``.

    Raises when a file can be neither downloaded nor found in the cache; a
    stale cached copy is used when the download fails.
    """
    if model not in MODELS:
        raise ValueError(f"unknown factor model {model!r}; one of {sorted(MODELS)}")
    cache = _cache_dir(cache_dir)
    frames = [_parse(_fetch(name, cache, timeout)) for name in MODELS[model][1]]
    factors = pd.concat(frames, axis="columns", join="inner")
    return factors.loc[:, ~factors.columns.duplicated()] / 100.0


def model_description(model: str) -> str:
    return MODELS[model][0]


def ruonia(cache_dir: str | Path | None = None, timeout: float = 8.0) -> pd.Series:
    """RUONIA, the Bank of Russia's overnight rouble rate, in percent a year, by day (UTC).

    Read from the Bank's ``DailyInfo`` web service since 2010 and cached for a
    day under the same directory as the Fama-French files. Raises when it can
    be neither downloaded nor found in the cache.
    """
    cache = _cache_dir(cache_dir)
    path = cache / "ruonia.xml"
    if path.exists() and time.time() - path.stat().st_mtime < RUONIA_MAX_AGE:
        payload = path.read_bytes()
    else:
        envelope = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/"><soap:Body>'
            '<Ruonia xmlns="http://web.cbr.ru/">'
            f"<fromDate>{RUONIA_SINCE.isoformat()}</fromDate><ToDate>{date.today().isoformat()}</ToDate>"
            "</Ruonia></soap:Body></soap:Envelope>"
        ).encode("utf-8")
        request = urllib.request.Request(CBR_SERVICE, data=envelope, headers={
            "Content-Type": "text/xml; charset=utf-8", "SOAPAction": "http://web.cbr.ru/Ruonia"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            _store(path, payload)
        except Exception:
            if not path.exists():
                raise
            print("Warning: could not refresh RUONIA; using the cached copy.")
            payload = path.read_bytes()

    rates = {}
    for row in ElementTree.fromstring(payload).iter():
        if row.tag.split("}")[-1] != "ro":
            continue
        fields = {child.tag.split("}")[-1]: child.text for child in row}
        if fields.get("D0") and fields.get("ruo"):
            rates[fields["D0"][:10]] = float(fields["ruo"])
    if not rates:
        raise ValueError("the Bank of Russia returned no RUONIA rates")
    series = pd.Series(rates, name="RUONIA").sort_index()
    series.index = pd.DatetimeIndex(pd.to_datetime(series.index)).tz_localize("UTC")
    return series


def risk_free_returns(rate: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Per-session returns of a rate in percent a year, accrued over the calendar days since the last session.

    Each session earns the rate known the day before -- so no session uses a
    rate published after it -- for the days between the two sessions, on an
    actual/365 basis: a Monday earns Friday's rate for three days.
    """
    rate = rate.sort_index()
    known = rate.reindex(rate.index.union(dates)).ffill().shift(1).reindex(dates)
    known = known.fillna(rate.iloc[0])
    days = pd.Series(dates, index=dates).diff().dt.days.fillna(1).clip(lower=1)
    return (known / 100.0 * days / 365.0).rename("RF")


def market_factors(benchmark: pd.Series, risk_free: pd.Series | None = None) -> pd.DataFrame:
    """A one-factor model: the benchmark over the risk-free rate, plus ``RF``.

    ``benchmark`` is daily returns; ``risk_free`` per-session returns (see
    :func:`risk_free_returns`), zero when not given.
    """
    rf = (risk_free if risk_free is not None else pd.Series(0.0, index=benchmark.index)).reindex(
        benchmark.index).fillna(0.0)
    return pd.DataFrame({"Mkt-RF": benchmark - rf, "RF": rf})


def _cache_dir(cache_dir: str | Path | None) -> Path:
    return Path(cache_dir or os.getenv("ZIPFOLIO_CACHE_DIR") or Path.home() / ".cache" / "zipfolio")


def _store(path: Path, payload: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    except OSError as error:
        print(f"Warning: could not cache {path.name}: {error}")


def _fetch(name: str, cache: Path, timeout: float) -> bytes:
    path = cache / f"{name}.zip"
    if path.exists() and time.time() - path.stat().st_mtime < CACHE_MAX_AGE:
        return path.read_bytes()
    try:
        with urllib.request.urlopen(LIBRARY_URL.format(name=name), timeout=timeout) as response:
            payload = response.read()
    except Exception:
        if path.exists():
            print(f"Warning: could not refresh {name}; using the cached copy from "
                  f"{time.strftime('%Y-%m-%d', time.gmtime(path.stat().st_mtime))}.")
            return path.read_bytes()
        raise
    _store(path, payload)
    return payload


def _parse(payload: bytes) -> pd.DataFrame:
    """The daily table of one library file: a few lines of notes, a header, ``YYYYMMDD,...`` rows."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        text = archive.read(archive.namelist()[0]).decode("latin-1")
    lines = [line.strip() for line in text.splitlines()]
    first = next(index for index, line in enumerate(lines) if re.match(r"^\d{8}\s*,", line))
    header = next(lines[index] for index in range(first - 1, -1, -1) if lines[index].startswith(","))
    columns = [column.strip() for column in header.split(",")[1:]]
    rows = []
    for line in lines[first:]:
        if not re.match(r"^\d{8}\s*,", line):
            break
        cells = [cell.strip() for cell in line.split(",")]
        rows.append([cells[0], *map(float, cells[1:len(columns) + 1])])
    frame = pd.DataFrame(rows, columns=["date", *columns])
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("date"), format="%Y%m%d")).tz_localize("UTC")
    return frame.rename(columns={"Mom": "MOM", "Mom   ": "MOM"})
