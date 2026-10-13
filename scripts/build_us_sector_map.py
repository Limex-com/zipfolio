"""Build the ticker-to-sector table `zipfolio.us_sector_mappings` reads.

Zipfolio ships no sector data -- GICS is licensed by MSCI and S&P -- so the
table is built from exports you are entitled to use: CSV files with
`symbol`, `gics`, `start_date` and `end_date` columns, one row per stretch of
membership. Point Zipfolio at the result with `table=` or `ZIPFOLIO_GICS_TABLE`.

The `gics` column holds the eight-digit GICS code; the sector is its first
two digits.

One row per ticker, from its most recent membership row: a current member
(`end_date` empty) wins, then the latest `start_date`. That is the right answer
for the two ways a ticker's code changes across rows -- GICS reclassifying a
company (META and GOOGL moved from 45 to 50 in 2018) and a ticker passing to
another company (`S` was Sears, and is SentinelOne now) -- as long as the
backtest is recent. For an old window over a reused ticker it is the new
company's sector; the positions carry only the ticker, so there is nothing to
tell the two apart by.

    python scripts/build_us_sector_map.py <exports dir> [<output csv>]
"""

import csv
import sys
from pathlib import Path

OUTPUT = Path("us_gics_sectors.csv")


def main(exports: Path, output: Path = OUTPUT) -> None:
    latest: dict[str, tuple[tuple, str]] = {}
    for path in sorted(exports.glob("*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                symbol, gics = row["symbol"].strip(), row["gics"].strip()
                if not symbol or len(gics) < 2 or not gics[:2].isdigit():
                    continue
                rank = (row["end_date"] == "", row["end_date"], row["start_date"])
                if symbol not in latest or rank > latest[symbol][0]:
                    latest[symbol] = (rank, gics[:2])

    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["symbol", "gics_sector"])
        for symbol in sorted(latest):
            writer.writerow([symbol, latest[symbol][1]])
    print(f"{len(latest)} tickers -> {output}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "exports"),
         Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT)
