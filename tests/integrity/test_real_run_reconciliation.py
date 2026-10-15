"""The actual Momentum Rotation backtest -- its primary book, not its report -- reconciled to the cent.

Ten US large caps, the top three by three-month momentum, rebalanced monthly; ziplime 2.11, 752 sessions
from 11 Oct 2023, $1,000,000, 0.0354% of value in commission, 5 bps of slippage. The fixture holds what the
service hands Zipfolio: returns, position snapshots, cash, fills, orders, the fills' bars, the settings.
The sectors are neutral groups, not GICS -- that classification is licensed and stays out of the repository.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from zipfolio.integrity import AccountingInputs, check_integrity
from zipfolio.ziplime import contract_specs, opening_state

FIXTURE = Path(__file__).parent.parent / "fixtures" / "momentum_rotation"


def momentum_book() -> AccountingInputs:
    book = json.loads((FIXTURE / "book.json").read_text())
    returns = pd.read_csv(FIXTURE / "returns.csv", index_col=0, parse_dates=True)["return"]
    cash = pd.read_csv(FIXTURE / "cash.csv", index_col=0)["cash"]
    cash.index = pd.to_datetime(cash.index, utc=True).tz_convert("America/New_York")
    # The times carry New York's offset, -04:00 or -05:00 across the clock change: read them as instants.
    fills = pd.read_csv(FIXTURE / "fills.csv").assign(dt=lambda f: pd.to_datetime(f["dt"], utc=True))
    orders = pd.read_csv(FIXTURE / "orders.csv").assign(dt=lambda f: pd.to_datetime(f["dt"], utc=True))
    bars = pd.read_csv(FIXTURE / "fill_bars.csv", index_col=0)
    settings = book["cost_settings"]
    return AccountingInputs(
        returns=returns, position_records=pd.read_csv(FIXTURE / "positions.csv"), cash=cash, transactions=fills,
        orders=orders, cost_settings=settings, fill_bars=bars, sector_mappings=book["sector_mappings"],
        opening_state=opening_state(settings["capital"], returns.index[0], "USD"),
        contract_specs=contract_specs(book["listings"], "USD"), currency="USD", run_id="momentum-rotation")


def test_the_momentum_book_reconciles_at_ledger_level(pyfolio):
    report = check_integrity(momentum_book())

    not_passed = {c.id: (c.status.value, c.explanation) for c in report.checks
                  if c.status.value not in ("PASS", "NOT_APPLICABLE")}
    assert not not_passed, not_passed
    assert report.status.value == "PASS" and report.assurance_level.value == "ledger"


def test_the_reported_figures_are_the_ledger_figures(pyfolio):
    report = check_integrity(momentum_book())
    reported = json.loads((FIXTURE / "book.json").read_text())["reported"]

    nav = report.check("NAV.SNAPSHOT_RECONCILIATION").to_dict("USD")
    assert nav["observed"] == nav["expected"] == reported["final_nav"]
    split = {row["figure"]: row for row in report.check("PNL.REALIZED_UNREALIZED").details}
    assert f"{split['realized']['ledger_fifo']:.2f}" == reported["realized"]
    assert f"{split['unrealized']['ledger_fifo']:.2f}" == reported["unrealized"]
    assert report.check("COST.COMMISSION_RECONCILIATION").to_dict("USD")["observed"] == reported["commission"]
    slippage = {row["figure"]: row["amount"] for row in report.check("COST.SLIPPAGE_SINGLE_COUNT").details}
    assert f"{slippage['slippage reconstructed by the report']:.2f}" == reported["slippage_reconstructed"]
    assert f"{slippage['slippage in the fill prices (|price - bar close| x shares)']:.2f}" == \
        reported["slippage_reconstructed"]
    assert report.check("ROUND_TRIP.SYNTHETIC_CLOSURES").details[0]["count"] == reported["closed_trades"]
    sectors = report.check("ATTRIBUTION.SECTOR_SUM")
    assert sectors.status.value == "PASS" and sectors.affected_symbols == ["WMT"], "in the Unclassified bucket"


@pytest.mark.parametrize("column, change, check", [
    ("price", 0.01, "CASH.LEDGER_REPLAY"), ("quantity", 1, "POSITIONS.LEDGER_REPLAY")])
def test_a_one_fill_corruption_of_the_real_book_fails(pyfolio, column, change, check):
    inputs = momentum_book()
    inputs.portfolio_events = pd.DataFrame(columns=["dt", "kind", "symbol", "cash", "quantity", "ratio"])
    inputs.transactions.loc[40, column] += change
    report = check_integrity(inputs)

    assert report.check(check).status.value == "FAIL"
    session = pd.Timestamp(inputs.transactions.loc[40, "dt"]).tz_convert("UTC").date().isoformat()
    assert report.check(check).first_failure_at == session, "on the session of the corrupted fill"
    assert report.status.value == "FAIL"


def test_the_report_is_deterministic(pyfolio):
    assert check_integrity(momentum_book()).to_json() == check_integrity(momentum_book()).to_json()
