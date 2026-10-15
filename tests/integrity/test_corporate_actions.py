"""Splits, dividends, interest and external flows: replayed when declared, never assumed away."""

from conftest import Book, Event, Fill, ledger_book, statuses
from zipfolio.integrity import check_integrity


def test_9_a_split_is_replayed_when_declared_and_skipped_when_not(pyfolio):
    book = Book({"A": [100.0, 50.0, 55.0]}, [Fill(0, "A", 10, 100.0, fee=1.0)], [Event(1, "split", "A", ratio=2.0)])
    report = check_integrity(ledger_book(book))
    assert report.status.value == "PASS", statuses(report)
    split = {row["figure"]: row for row in report.check("PNL.REALIZED_UNREALIZED").details}
    assert split["unrealized"]["ledger_fifo"] == 100.0, "20 shares at 55 over a cost of 50 each"

    undeclared = check_integrity(ledger_book(book, declare_events=False))
    positions = undeclared.check("POSITIONS.LEDGER_REPLAY")
    assert positions.status.value == "SKIPPED" and positions.affected_symbols == ["A"]
    assert undeclared.status.value == "INCOMPLETE", "never FAIL for a missing event ledger, never PASS either"


def test_9_dividends_and_interest_are_income(pyfolio):
    book = Book({"A": [100.0, 100.0, 101.0]}, [Fill(0, "A", 10, 100.0)],
                [Event(1, "dividend", "A", cash=5.0), Event(2, "interest", None, cash=0.8)])
    report = check_integrity(ledger_book(book))
    assert report.status.value == "PASS", statuses(report)
    components = {row["component"]: row["amount"] for row in report.check("PNL.PORTFOLIO_RECONCILIATION").details}
    assert components["income events"] == 5.8
    instruments = report.check("ATTRIBUTION.INSTRUMENT_SUM")
    assert instruments.status.value == "PASS", "the income is not any name's"

    missing = check_integrity(ledger_book(book, declare_events=False))
    assert missing.check("CASH.LEDGER_REPLAY").status.value == "SKIPPED"
    assert missing.check("CASH.LEDGER_REPLAY").reason_code == "UNEXPLAINED_WITHOUT_EVENT_LEDGER"


def test_10_deposits_and_withdrawals_are_flows_not_profit(pyfolio):
    book = Book({"A": [100.0, 110.0, 120.0]}, [Fill(0, "A", 10, 100.0)],
                [Event(1, "deposit", None, cash=5_000.0), Event(2, "withdrawal", None, cash=-1_000.0)])
    report = check_integrity(ledger_book(book))

    assert report.status.value == "PASS", statuses(report)
    profit = report.check("PNL.PORTFOLIO_RECONCILIATION")
    assert float(profit.to_dict("USD")["expected"]) == 200.0, "the change in NAV less 4,000 put in"
    assert report.check("RETURNS.NAV_RECONCILIATION").status.value == "PASS", "time-weighted around the flows"
