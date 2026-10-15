"""Commissions against orders and the fee model; slippage counted once."""

import pandas as pd

from conftest import Book, Fill, ledger_book, statuses
from zipfolio.integrity import check_integrity

PER_SHARE = {"commission_model": "per_share", "commission_per_share": 0.01, "commission_min_trade_cost": 1.0}


def test_4_fees_with_a_minimum_on_the_buy_and_the_sale(pyfolio):
    book = Book({"A": [100.0, 110.0, 120.0]}, [Fill(0, "A", 10, 100.0, fee=1.0), Fill(1, "A", -4, 110.0, fee=1.0)])
    report = check_integrity(ledger_book(book, settings=PER_SHARE))

    commission = report.check("COST.COMMISSION_RECONCILIATION")
    assert commission.status.value == "PASS" and "fee model" in commission.explanation
    assert report.status.value == "PASS", statuses(report)


def test_a_charge_the_fee_model_does_not_give_fails():
    book = Book({"A": [100.0, 110.0]}, [Fill(0, "A", 10, 100.0, fee=3.0)])
    report = check_integrity(ledger_book(book, settings=PER_SHARE))

    commission = report.check("COST.COMMISSION_RECONCILIATION")
    assert commission.status.value == "FAIL" and commission.details[0]["fee_model"] == 1.0


def test_repeated_order_rows_count_once_and_conflicting_ones_fail():
    inputs = ledger_book(Book({"A": [100.0, 110.0]}, [Fill(0, "A", 10, 100.0, fee=1.0)]))
    inputs.orders = pd.concat([inputs.orders, inputs.orders])  # the engine repeats an order on each session
    report = check_integrity(inputs)
    assert report.check("COST.COMMISSION_RECONCILIATION").status.value == "PASS"
    assert float(report.check("COST.COMMISSION_RECONCILIATION").to_dict("USD")["observed"]) == 1.0

    conflicting = ledger_book(Book({"A": [100.0, 110.0]}, [Fill(0, "A", 10, 100.0, fee=1.0)]))
    conflicting.orders = pd.concat([conflicting.orders, conflicting.orders.assign(commission=2.0)])
    assert check_integrity(conflicting).check("COST.COMMISSION_RECONCILIATION").status.value == "FAIL"


def test_6_slippage_at_an_unchanged_price_is_a_loss_counted_once(pyfolio):
    bps = 0.0005
    book = Book({"A": [100.0, 100.0, 100.0]},
                [Fill(0, "A", 10, 100.0 * (1 + bps), fee=1.0), Fill(1, "A", -10, 100.0 * (1 - bps), fee=1.0)])
    inputs = ledger_book(book, settings={"commission_model": "per_share", "commission_per_share": 0.1,
                                         "commission_min_trade_cost": 1.0, "slippage_basis_points": 5.0})
    inputs.fill_bars = pd.DataFrame({"close": [100.0, 100.0]}, index=["f0", "f1"])
    report = check_integrity(inputs)

    slippage = report.check("COST.SLIPPAGE_SINGLE_COUNT")
    assert slippage.status.value == "PASS", slippage.explanation
    assert float(slippage.to_dict("USD")["observed"]) == -3.0, "one dollar of slippage and two of fees, once each"
    assert report.status.value == "PASS", statuses(report)


def test_slippage_taken_off_twice_fails(pyfolio, monkeypatch):
    import zipfolio.integrity.checks as checks

    bps = 0.0005
    book = Book({"A": [100.0, 100.0]}, [Fill(0, "A", 10, 100.0 * (1 + bps)), Fill(1, "A", -10, 100.0 * (1 - bps))])
    inputs = ledger_book(book, settings={"commission_model": "per_share", "commission_per_share": 0.0,
                                         "slippage_basis_points": 5.0})
    honest = checks._report_figures

    def deducting(inputs):
        figures = honest(inputs)
        figures["instrument_net"] = figures["instrument_net"] - 1.0 / len(figures["instrument_net"])
        return figures

    monkeypatch.setattr(checks, "_report_figures", deducting)
    slippage = check_integrity(inputs).check("COST.SLIPPAGE_SINGLE_COUNT")
    assert slippage.status.value == "FAIL" and slippage.reason_code == "SLIPPAGE_DEDUCTED_TWICE"
