"""Your first Zipfolio tearsheet, from a real backtest's book, in one command.

    pip install "zipfolio[all]"
    python examples/quickstart.py            # writes momentum_rotation.html and opens it

The book is the Momentum Rotation backtest the tests replay (tests/fixtures/momentum_rotation):
three years of a monthly rotation into the top 3 of 10 US large caps, $1m, with ziplime's fills,
orders and commissions. From a clone the files are read from disk; anywhere else, from GitHub.

There is no benchmark or daily market data in this book, so the sections that need them (market
regimes, CAPM, capacity, fill realism within the bar) are left out. The live examples in the README
have everything.
"""

import json
import sys
import webbrowser
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from zipfolio import build_tearsheet
from zipfolio.ziplime import contract_specs, opening_state

LOCAL = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "momentum_rotation"
REMOTE = "https://raw.githubusercontent.com/Limex-com/zipfolio/main/tests/fixtures/momentum_rotation/"
SOURCE = f"{LOCAL}/" if LOCAL.exists() else REMOTE


def read(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(SOURCE + name, **kwargs)


book = json.loads((LOCAL / "book.json").read_text() if LOCAL.exists() else urlopen(REMOTE + "book.json").read())

returns = read("returns.csv", index_col=0, parse_dates=True)["return"]        # one return per session
cash = read("cash.csv", index_col=0)["cash"]                                   # cash at each session's end
cash.index = pd.to_datetime(cash.index, utc=True)
positions = read("positions.csv")                                              # date, symbol, quantity, last_sale_price
fills = read("fills.csv").assign(dt=lambda f: pd.to_datetime(f["dt"], utc=True))    # id, order_id, dt, symbol, quantity, price
orders = read("orders.csv").assign(dt=lambda f: pd.to_datetime(f["dt"], utc=True))  # id, dt, symbol, quantity, filled, commission

output = Path(sys.argv[1] if len(sys.argv) > 1 else "momentum_rotation.html").resolve()
sections = build_tearsheet(
    output,
    returns,
    position_records=positions,
    cash=cash,
    transactions=fills,
    orders=orders,
    cost_settings=book["cost_settings"],
    sector_mappings={
        "AAPL": "Information Technology", "MSFT": "Information Technology", "NVDA": "Information Technology",
        "GOOGL": "Communication Services", "META": "Communication Services", "AMZN": "Consumer Discretionary",
        "JPM": "Financials", "XOM": "Energy", "JNJ": "Health Care", "WMT": "Consumer Staples",
    },
    opening_state=opening_state(book["cost_settings"]["capital"], returns.index[0], book["currency"]),
    contract_specs=contract_specs(book["listings"], book["currency"]),
    run_info={"Strategy": "Momentum rotation, top 3 of 10 US large caps, monthly (example)"},
)
print(f"{output}\n{len(sections)} sections: {', '.join(sections)}")
webbrowser.open(output.as_uri())
