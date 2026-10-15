"""Random books: every honest one reconciles; every single corruption is caught by a required check."""

import numpy as np
import pytest

from conftest import Book, Fill, ledger_book, statuses
from zipfolio.integrity import check_integrity

PER_SHARE = {"commission_model": "per_share", "commission_per_share": 0.005, "commission_min_trade_cost": 1.0}


def random_book(seed: int) -> Book:
    rng = np.random.default_rng(seed)
    count = 12
    prices = {s: list(np.round(rng.uniform(20, 200) * np.cumprod(1 + rng.normal(0, 0.02, count)), 2))
              for s in ("A", "B", "C")}
    fills = []
    for day in range(count):
        for symbol in prices:
            if rng.random() < 0.35:
                quantity = int(rng.integers(-30, 40))
                if quantity == 0:
                    continue
                if rng.random() < 0.3:  # an order filled over two sessions, as the volume limit splits one
                    first = quantity // 2 or quantity
                    rest = quantity - first
                    fills.append(Fill(day, symbol, first, prices[symbol][day], f"o{seed}-{day}-{symbol}"))
                    if rest and day + 1 < count:
                        fills.append(Fill(day + 1, symbol, rest, prices[symbol][day + 1], f"o{seed}-{day}-{symbol}"))
                else:
                    fills.append(Fill(day, symbol, quantity, prices[symbol][day]))
    # fees as the per-share model charges them, order by order
    charged, filled = {}, {}
    for fill in fills:
        order = fill.order or id(fill)
        extra = abs(fill.quantity) * PER_SHARE["commission_per_share"]
        if not charged.get(order):
            fee = max(PER_SHARE["commission_min_trade_cost"], extra)
        else:
            total = abs(filled.get(order, 0)) * PER_SHARE["commission_per_share"] + extra
            fee = 0.0 if total < PER_SHARE["commission_min_trade_cost"] else total - charged[order]
        fill.fee = round(fee, 10)
        charged[order] = charged.get(order, 0.0) + fee
        filled[order] = filled.get(order, 0) + fill.quantity
    return Book(prices, fills, capital=50_000.0)


@pytest.mark.parametrize("seed", range(12))
def test_every_honest_book_reconciles(pyfolio, seed):
    report = check_integrity(ledger_book(random_book(seed), settings=PER_SHARE, declare_events=True))
    failing = {k: v for k, v in statuses(report).items() if v in ("FAIL", "SKIPPED")}
    assert not failing, failing
    assert report.assurance_level.value == "ledger"


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("corruption", ["price", "quantity", "fee", "valuation"])
def test_13_every_single_corruption_is_caught(seed, corruption):
    inputs = ledger_book(random_book(seed), settings=PER_SHARE, declare_events=True)
    if inputs.transactions.empty:
        pytest.skip("no fills in this book")
    rng = np.random.default_rng(1000 + seed)
    if corruption == "price":
        row = int(rng.integers(len(inputs.transactions)))
        inputs.transactions.loc[row, "price"] += 0.5
    elif corruption == "quantity":
        row = int(rng.integers(len(inputs.transactions)))
        inputs.transactions.loc[row, "quantity"] += 1
    elif corruption == "fee":
        inputs.orders.loc[inputs.orders["id"] == inputs.orders["id"].iloc[0], "commission"] += 0.25
    else:
        if inputs.position_records.empty:
            pytest.skip("nothing held")
        row = int(rng.integers(len(inputs.position_records)))
        inputs.position_records.loc[row, "last_sale_price"] += 0.5
    report = check_integrity(inputs)
    assert report.status.value == "FAIL", (corruption, statuses(report))
