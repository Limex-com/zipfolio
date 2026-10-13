"""Daily Fama-French factors, from Kenneth French's data library.

The library publishes each factor set as a zipped CSV of percentages, updated
monthly from CRSP. This is what ``pandas_datareader.famafrench`` reads, without
the dependency: one request per file, cached for a week -- the files change
once a month, and a backtest service builds many reports a day.
"""

from __future__ import annotations

import io
import os
import re
import time
import urllib.request
import zipfile
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


def fama_french_daily(model: str = "ff5_mom", cache_dir: str | Path | None = None,
                      timeout: float = 8.0) -> pd.DataFrame:
    """Daily factor returns as decimals, one row per day at midnight UTC, plus ``RF``.

    Raises when a file can be neither downloaded nor found in the cache; a
    stale cached copy is used when the download fails.
    """
    if model not in MODELS:
        raise ValueError(f"unknown factor model {model!r}; one of {sorted(MODELS)}")
    cache = Path(cache_dir or os.getenv("ZIPFOLIO_CACHE_DIR") or Path.home() / ".cache" / "zipfolio")
    frames = [_parse(_fetch(name, cache, timeout)) for name in MODELS[model][1]]
    factors = pd.concat(frames, axis="columns", join="inner")
    return factors.loc[:, ~factors.columns.duplicated()] / 100.0


def model_description(model: str) -> str:
    return MODELS[model][0]


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
    try:
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    except OSError as error:
        print(f"Warning: could not cache {name}: {error}")
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
