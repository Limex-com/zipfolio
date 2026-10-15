"""Cash and positions replayed from the opening state; realized and unrealized first in, first out."""

from conftest import Book, Fill, ledger_book, statuses
from zipfolio.integrity import check_integrity

PER_SHARE = {"commission_model": "per_share", "commission_per_share": 0.01, "commission_min_trade_cost": 1.0}


def test_3_partial_close_splits_realized_and_unrealized(pyfolio):
    report = check_integrity(ledger_book(Book({"A": [100.0, 110.0, 120.0]},
                                              [Fill(0, "A", 10, 100.0), Fill(1, "A", -4, 110.0)])))

    assert report.status.value == "PASS", statuses(report)
    details = {row["figure"]: row for row in report.check("PNL.REALIZED_UNREALIZED").details}
    assert details["realized"]["ledger_fifo"] == 40.0 and details["unrealized"]["ledger_fifo"] == 120.0


def test_5_an_order_filled_on_two_days_books_its_fees_by_the_fee_model(pyfolio):
    """The order's charge is cumulative; the fee model says which day paid what."""
    fills = [Fill(0, "A", 150, 100.0, "o1", fee=1.5), Fill(1, "A", 50, 101.0, "o1", fee=0.5)]
    with_model = check_integrity(ledger_book(Book({"A": [100.0, 101.0, 102.0]}, fills), settings=PER_SHARE))
    assert with_model.check("CASH.LEDGER_REPLAY").status.value == "PASS"
    assert with_model.status.value == "PASS", statuses(with_model)

    without = check_integrity(ledger_book(Book({"A": [100.0, 101.0, 102.0]}, fills)))
    cash = without.check("CASH.LEDGER_REPLAY")
    assert cash.status.value == "WARN" and cash.reason_code == "FEE_BOOKING_AMBIGUOUS", (
        "the cumulative charge cannot be pinned to the last day without a fee model")


def test_7_a_short_is_covered_in_part_and_a_borrow_fee_is_booked(pyfolio):
    from conftest import Event

    book = Book({"A": [100.0, 90.0, 95.0]}, [Fill(0, "A", -10, 100.0, fee=1.0), Fill(1, "A", 4, 90.0, fee=1.0)],
                [Event(2, "borrow_fee", "A", cash=-2.5)])
    report = check_integrity(ledger_book(book))

    assert report.status.value == "PASS", statuses(report)
    details = {row["figure"]: row for row in report.check("PNL.REALIZED_UNREALIZED").details}
    assert details["realized"]["ledger_fifo"] == 40.0, "4 shares covered 10 below the sale"
    assert details["unrealized"]["ledger_fifo"] == 30.0, "6 still short, 5 below the sale"
    components = {row["component"]: row["amount"] for row in report.check("PNL.PORTFOLIO_RECONCILIATION").details}
    assert components["expense events"] == -2.5


def test_13_a_corrupted_fill_fails_the_replay_that_sees_it():
    book = Book({"A": [100.0, 110.0, 120.0]}, [Fill(0, "A", 10, 100.0, fee=1.0)])
    for column, value, check in (("price", 100.5, "CASH.LEDGER_REPLAY"), ("quantity", 11, "POSITIONS.LEDGER_REPLAY")):
        inputs = ledger_book(book, declare_events=True)
        inputs.transactions.loc[0, column] = value
        report = check_integrity(inputs)
        result = report.check(check)
        assert result.status.value == "FAIL", (column, result.explanation)
        assert result.first_failure_at == "2024-03-04"
        assert report.status.value == "FAIL"

        undeclared = ledger_book(book)
        undeclared.transactions.loc[0, column] = value
        result = check_integrity(undeclared).check(check)
        assert result.status.value == "SKIPPED" and result.reason_code == "UNEXPLAINED_WITHOUT_EVENT_LEDGER", (
            "without an event ledger a gap could be a dividend: not checkable, never a pass")


def test_8_several_names_first_and_last_session(pyfolio):
    book = Book({"A": [10.0, 11.0, 12.0, 13.0], "B": [50.0, 49.0, 48.0, 52.0], "C": [5.0, 5.0, 5.0, 6.0]},
                [Fill(0, "A", 100, 10.0, fee=1.0), Fill(0, "B", 20, 50.0, fee=1.0), Fill(2, "A", -100, 12.0, fee=1.0),
                 Fill(3, "C", 200, 5.0, fee=1.0)])
    report = check_integrity(ledger_book(book))

    assert report.status.value == "PASS", statuses(report)
    assert report.check("ROUND_TRIP.SYNTHETIC_CLOSURES").status.value == "PASS", "C bought on the last session"
