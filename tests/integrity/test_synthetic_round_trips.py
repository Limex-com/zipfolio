"""Trades are real closes; open positions are unrealized; the page calls gross gross."""

from conftest import Book, Fill, ledger_book, statuses
from zipfolio.integrity import check_integrity


def test_12_open_positions_are_not_closed_trades_and_realized_is_not_inflated(pyfolio):
    book = Book({"A": [100.0, 120.0, 130.0], "B": [200.0, 190.0, 180.0], "C": [50.0, 50.0, 52.0]},
                [Fill(0, "A", 10, 100.0, fee=1.0), Fill(0, "B", 5, 200.0, fee=1.0), Fill(1, "A", -4, 120.0, fee=1.0),
                 Fill(2, "C", 3, 50.0, fee=1.0)])
    report = check_integrity(ledger_book(book))

    trades = report.check("ROUND_TRIP.SYNTHETIC_CLOSURES")
    assert trades.status.value == "PASS" and trades.details[0]["count"] == 1, "one sale, one trade"
    split = {row["figure"]: row for row in report.check("PNL.REALIZED_UNREALIZED").details}
    assert split["realized"]["report"] == split["realized"]["ledger_fifo"] == 80.0
    assert report.status.value == "PASS", statuses(report)


def test_a_report_that_sells_on_paper_is_caught(pyfolio, monkeypatch):
    """pyfolio's own closing sale, counted as a trade, as the report did before 0.3."""
    from pyfolio import round_trips

    import zipfolio.sections.round_trips as section

    def legacy(returns, positions, transactions):
        closed = round_trips.add_closing_transactions(positions, transactions)
        return round_trips.extract_round_trips(closed, portfolio_value=positions.sum(axis="columns") / (1 + returns))

    monkeypatch.setattr(section, "extract_round_trips", legacy)
    book = Book({"A": [100.0, 120.0, 130.0]}, [Fill(0, "A", 10, 100.0), Fill(1, "A", -4, 120.0)])
    report = check_integrity(ledger_book(book))

    trades = report.check("ROUND_TRIP.SYNTHETIC_CLOSURES")
    assert trades.status.value == "WARN" and trades.reason_code == "SYNTHETIC_CLOSE_COUNTED"
    assert report.check("PNL.REALIZED_UNREALIZED").status.value == "WARN", "realized inflated by the paper sale"


def _sections(inputs):
    """The page's sections for ``inputs``, built the way build_tearsheet builds them."""
    import tempfile
    from pathlib import Path

    import zipfolio.tearsheet as tearsheet

    captured = {}
    original = tearsheet.page_html

    def keep(title, returns, benchmark, analytics, sections, run=None, integrity=""):
        captured["sections"] = sections
        return original(title, returns, benchmark, analytics, sections, run, integrity)

    tearsheet.page_html = keep
    try:
        with tempfile.TemporaryDirectory() as folder:
            tearsheet.build_tearsheet(Path(folder) / "page.html", inputs.returns, None, inputs.position_records,
                                      inputs.cash, inputs.transactions, orders=inputs.orders, factor_model=None,
                                      opening_state=inputs.opening_state, contract_specs=inputs.contract_specs,
                                      integrity=False)
    finally:
        tearsheet.page_html = original
    return captured["sections"]


def _busy_book():
    prices = {"A": [100.0 + i for i in range(40)], "B": [50.0 + (i % 5) for i in range(40)]}
    fills = [Fill(i, "A", 10 if i % 2 == 0 else -10, 100.0 + i, fee=1.0) for i in range(0, 30, 3)]
    fills += [Fill(i, "B", 20 if i % 2 == 0 else -20, 50.0 + (i % 5), fee=1.0) for i in range(1, 31, 3)]
    return ledger_book(Book(prices, fills))


def test_14_a_gross_figure_called_net_is_a_warning_and_changes_no_number(pyfolio):
    inputs = _busy_book()
    sections = _sections(inputs)
    honest = check_integrity(inputs, sections=sections)
    assert honest.check("ROUND_TRIP.GROSS_NET_CONVENTION").status.value == "PASS"
    assert honest.check("REPORT.DISPLAY_VS_CALCULATION").status.value == "PASS"

    for section in sections:
        section.side = [part.replace("Total PnL, before commission", "Total net profit") for part in section.side]
        section.wide = [part.replace("Total PnL, before commission", "Total net profit") for part in section.wide]
    mislabeled = check_integrity(inputs, sections=sections)
    convention = mislabeled.check("ROUND_TRIP.GROSS_NET_CONVENTION")
    assert convention.status.value == "WARN" and convention.reason_code == "GROSS_PNL_LABELED_NET"
    assert mislabeled.check("PNL.PORTFOLIO_RECONCILIATION").to_dict("USD") == \
        honest.check("PNL.PORTFOLIO_RECONCILIATION").to_dict("USD"), "the ledger is untouched"
    assert mislabeled.status.value == "WARN"


def test_a_shown_figure_that_is_not_the_calculated_one_fails(pyfolio):
    inputs = _busy_book()
    sections = _sections(inputs)
    performance = next(s for s in sections if s.title == "Performance")
    import re

    performance.side = [re.sub(r"(Realized:[^$]*?</td><td[^>]*>)\$", r"\1$1", part) for part in performance.side]
    display = check_integrity(inputs, sections=sections).check("REPORT.DISPLAY_VS_CALCULATION")
    assert display.status.value == "FAIL"
