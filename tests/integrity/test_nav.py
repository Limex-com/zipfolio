"""NAV, returns and the status model: what passes, what fails, and what is only incomplete."""

import json

from conftest import Book, Fill, ledger_book, statuses
from zipfolio.integrity import AccountingPolicy, ContractSpec, IntegrityFailure, check_integrity


def test_1_no_trades_cash_only_is_a_ledger_pass(pyfolio):
    report = check_integrity(ledger_book(Book({"A": [100.0, 101.0, 102.0]})))

    s = statuses(report)
    assert s["INPUT.FILL_UNIQUENESS"] == "NOT_APPLICABLE" and s["COST.SLIPPAGE_SINGLE_COUNT"] == "NOT_APPLICABLE"
    assert s["NAV.SNAPSHOT_RECONCILIATION"] == s["CASH.LEDGER_REPLAY"] == s["POSITIONS.LEDGER_REPLAY"] == "PASS"
    assert report.status.value == "PASS" and report.assurance_level.value == "ledger"


def test_2_an_open_long_is_unrealized_and_reconciles(pyfolio):
    report = check_integrity(ledger_book(Book({"A": [100.0, 110.0]}, [Fill(0, "A", 10, 100.0)])))

    assert report.status.value == "PASS", statuses(report)
    split = report.check("PNL.REALIZED_UNREALIZED")
    assert float(split.to_dict("USD")["observed"]) == 0.0, "nothing was sold"
    assert "unrealized PnL (100.00)" in report.check("ROUND_TRIP.SYNTHETIC_CLOSURES").explanation


def test_a_corrupted_valuation_fails_nav_on_its_session():
    inputs = ledger_book(Book({"A": [100.0, 110.0, 120.0]}, [Fill(0, "A", 10, 100.0)]))
    records = inputs.position_records
    records.loc[1, "last_sale_price"] = 111.0  # the second session's mark, by a dollar
    report = check_integrity(inputs)

    nav = report.check("NAV.SNAPSHOT_RECONCILIATION")
    assert nav.status.value == "FAIL" and nav.first_failure_at == "2024-03-05"
    assert float(nav.to_dict("USD")["difference"]) == 10.0, "ten shares a dollar too high"
    assert report.status.value == "FAIL"


def test_a_changed_return_fails_the_returns_check():
    inputs = ledger_book(Book({"A": [100.0, 110.0, 120.0]}, [Fill(0, "A", 10, 100.0)]))
    inputs.returns.iloc[-1] = 0.000001  # what the service's nudge for quiet strategies did
    report = check_integrity(inputs)

    assert report.check("RETURNS.NAV_RECONCILIATION").status.value == "FAIL"
    assert report.check("RETURNS.NAV_RECONCILIATION").first_failure_at == "2024-03-06"


def test_without_an_opening_state_the_book_is_reconciled_not_replayed(pyfolio):
    inputs = ledger_book(Book({"A": [100.0, 110.0, 120.0]}, [Fill(0, "A", 10, 100.0)]), opening=False)
    report = check_integrity(inputs)

    s = statuses(report)
    assert s["CASH.LEDGER_REPLAY"] == s["POSITIONS.LEDGER_REPLAY"] == s["NAV.SNAPSHOT_RECONCILIATION"] == "SKIPPED"
    assert s["RETURNS.NAV_RECONCILIATION"] == s["PNL.PORTFOLIO_RECONCILIATION"] == "PASS"
    assert report.status.value == "INCOMPLETE" and report.assurance_level.value == "reconciled"
    assert "opening_state" in report.check("CASH.LEDGER_REPLAY").missing_inputs


def test_returns_alone_are_a_summary_never_a_pass():
    inputs = ledger_book(Book({"A": [100.0, 110.0]}))
    inputs.position_records = inputs.cash = inputs.transactions = inputs.orders = None
    inputs.opening_state = None
    report = check_integrity(inputs)

    assert report.status.value == "INCOMPLETE" and report.assurance_level.value == "summary"


def test_11_unknown_or_unsupported_instruments_are_skipped_not_valued():
    book = Book({"NG": [3.0, 3.1]}, [Fill(0, "NG", 1, 3.0)])
    for specs, reason in (({}, "MISSING_CONTRACT_SPECS"),
                          ({"NG": ContractSpec("future", 10_000.0, "USD")}, "UNSUPPORTED_INSTRUMENT"),
                          ({"NG": ContractSpec("equity", 1.0, "EUR")}, "UNSUPPORTED_MULTICURRENCY")):
        report = check_integrity(ledger_book(book, specs=specs))
        nav = report.check("NAV.SNAPSHOT_RECONCILIATION")
        assert nav.status.value == "SKIPPED" and nav.reason_code == reason, reason
        assert report.status.value == "INCOMPLETE"


def test_strict_and_ci_policies():
    good = ledger_book(Book({"A": [100.0, 110.0]}, [Fill(0, "A", 10, 100.0)]))
    bad = ledger_book(Book({"A": [100.0, 110.0]}, [Fill(0, "A", 10, 100.0)]))
    bad.cash.iloc[-1] += 5.0
    check_integrity(good, AccountingPolicy(mode="strict")).raise_on_failure()
    failing = check_integrity(bad, AccountingPolicy(mode="strict"))
    try:
        failing.raise_on_failure()
        raise AssertionError("a failed required check is fatal under strict")
    except IntegrityFailure as error:
        assert "NAV.SNAPSHOT_RECONCILIATION" in str(error), "cash plus positions no longer match the returns"
    incomplete = ledger_book(Book({"A": [100.0, 110.0]}), opening=False)
    assert check_integrity(incomplete, AccountingPolicy(mode="report")).exit_code() == 0
    assert check_integrity(incomplete, AccountingPolicy(mode="ci")).exit_code() == 1
    assert check_integrity(bad, AccountingPolicy(mode="report")).exit_code() == 1


def test_15_the_json_is_deterministic_and_plain(pyfolio):
    first = check_integrity(ledger_book(Book({"A": [100.0, 110.0, 105.0]},
                                             [Fill(0, "A", 10, 100.0, fee=1.0), Fill(1, "A", -4, 110.0, fee=1.0)])))
    second = check_integrity(ledger_book(Book({"A": [100.0, 110.0, 105.0]},
                                              [Fill(0, "A", 10, 100.0, fee=1.0), Fill(1, "A", -4, 110.0, fee=1.0)])))
    text = first.to_json()
    assert text == second.to_json(), "same inputs, same bytes"
    assert "NaN" not in text and "Infinity" not in text
    data = json.loads(text)
    assert data["schema_version"] == "accounting-integrity/v1" and data["input_hash"] == first.input_hash
    nav = next(c for c in data["checks"] if c["id"] == "NAV.SNAPSHOT_RECONCILIATION")
    assert isinstance(nav["observed"], str) and nav["currency"] == "USD", "money as decimal strings"
    assert [c["id"] for c in data["checks"]][:2] == ["INPUT.FILL_UNIQUENESS", "INPUT.TIMESTAMP_ALIGNMENT"]


def test_inputs_are_not_modified():
    inputs = ledger_book(Book({"A": [100.0, 110.0]}, [Fill(0, "A", 10, 100.0, fee=1.0)]))
    before = {name: getattr(inputs, name).copy() for name in ("returns", "position_records", "cash",
                                                                "transactions", "orders")}
    check_integrity(inputs)
    for name, frame in before.items():
        assert frame.equals(getattr(inputs, name)), name


def test_the_command_line_exits_non_zero_when_ci_rejects(tmp_path, capsys):
    import dataclasses
    import pickle

    from zipfolio.integrity.__main__ import main

    good = ledger_book(Book({"A": [100.0, 110.0]}, [Fill(0, "A", 10, 100.0)]))
    path = tmp_path / "book.pkl"
    path.write_bytes(pickle.dumps({f.name: getattr(good, f.name) for f in dataclasses.fields(good)}))
    assert main([str(path), "--json", str(tmp_path / "report.json")]) == 0
    assert '"status": "PASS"' in (tmp_path / "report.json").read_text()

    incomplete = dataclasses.replace(good, opening_state=None)
    path.write_bytes(pickle.dumps({f.name: getattr(incomplete, f.name) for f in dataclasses.fields(incomplete)}))
    assert main([str(path)]) == 1, "ci: a required check that could not be made fails the step"
    assert "INCOMPLETE" in capsys.readouterr().out
