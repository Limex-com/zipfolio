"""The checks, by stable id, in the order every report lists them.

Two kinds of evidence meet here. The independent one is this package's own: the normalized book and the
ledger replayed from it (``ledger``). The other is the report's: the figures Zipfolio publishes, computed
the way the page computes them -- pyfolio's round trips, ``accounting.profit_breakdown``, the instrument
and sector tables. Those are the items under test, never the source of truth.
"""

from __future__ import annotations

import re
import traceback
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import __version__
from .ledger import book_fees, replay
from .models import (
    ASSURANCE_RANK,
    Assurance,
    CheckResult,
    IntegrityReport,
    Status,
)
from .normalize import normalize

#: Every check, in report order.
CHECK_IDS = (
    "INPUT.FILL_UNIQUENESS", "INPUT.TIMESTAMP_ALIGNMENT", "NAV.SNAPSHOT_RECONCILIATION",
    "RETURNS.NAV_RECONCILIATION", "CASH.LEDGER_REPLAY", "POSITIONS.LEDGER_REPLAY",
    "PNL.REPORTED_SUMMARY_IDENTITY", "PNL.PORTFOLIO_RECONCILIATION", "PNL.REALIZED_UNREALIZED",
    "COST.COMMISSION_RECONCILIATION", "COST.SLIPPAGE_SINGLE_COUNT", "ATTRIBUTION.INSTRUMENT_SUM",
    "ATTRIBUTION.SECTOR_SUM", "ROUND_TRIP.SYNTHETIC_CLOSURES", "ROUND_TRIP.GROSS_NET_CONVENTION",
    "REPORT.DISPLAY_VS_CALCULATION",
)
#: The invariants an accounting pass needs; the others are conventions and presentation.
REQUIRED = frozenset(CHECK_IDS[:12])

#: Shares compare to this; a fraction of a share is a different holding.
SHARE_TOLERANCE = 1e-9


@dataclass
class Context:
    """What the checks share: the book, the ledger and the report's own figures."""

    inputs: object
    policy: object
    book: object
    valuation_problem: tuple | None
    multipliers: dict
    fees: object
    ledger: object
    snapshot_nav: pd.Series | None
    report: dict | None
    sections: list | None
    results: dict

    def ledger_holds(self) -> bool:
        """Whether the replayed book matched the record: only then are its lots the book's lots."""
        positions = self.results.get("POSITIONS.LEDGER_REPLAY")
        cash = self.results.get("CASH.LEDGER_REPLAY")
        return (self.ledger is not None and positions is not None and positions.status == Status.PASS
                and cash is not None and cash.status in (Status.PASS, Status.WARN))

    @property
    def events_declared(self) -> bool:
        """Whether the caller gave an event ledger -- an empty one included, which says there were none."""
        return self.inputs.portfolio_events is not None

    def tol(self, scale: float = 0.0) -> float:
        return self.policy.tolerance(scale)


def check_integrity(inputs, policy=None, sections=None) -> IntegrityReport:
    """Run every check on ``inputs`` (``AccountingInputs``) and return the report.

    ``sections`` -- the page's built sections, when there is a page -- lets the presentation checks read
    what the page shows. The inputs are not modified.
    """
    from .models import AccountingPolicy

    policy = policy or AccountingPolicy()
    book = normalize(inputs)
    valuation_problem, multipliers = _valuation(inputs, book)
    fees = book_fees(book, inputs.cost_settings, policy.abs_tolerance) if book.orders is not None else None
    ledger = None
    if inputs.opening_state is not None and valuation_problem is None and book.cash is not None:
        ledger = replay(book, inputs.opening_state, fees, multipliers)
    snapshot_nav = _snapshot_nav(book, multipliers) if valuation_problem is None else None
    report = _report_figures(inputs)
    context = Context(inputs, policy, book, valuation_problem, multipliers, fees, ledger, snapshot_nav, report,
                      sections, {})
    checks = []
    for check_id in CHECK_IDS:
        function = _CHECKS[check_id]
        try:
            result = function(context)
        except Exception as error:  # a check that breaks is a check that could not be made, never a pass
            result = CheckResult(check_id, check_id in REQUIRED, Status.SKIPPED,
                                 f"The check could not run: {type(error).__name__}: {error}",
                                 reason_code="CHECK_ERROR", severity="warning")
            print(f"Warning: integrity check {check_id} failed to run:\n{traceback.format_exc()}")
        result.required = check_id in REQUIRED and result.status != Status.NOT_APPLICABLE
        if result.status == Status.FAIL:
            result.severity = "error" if result.required else "warning"
        elif result.status == Status.WARN:
            result.severity = "warning"
        checks.append(result)
        context.results[check_id] = result
    return IntegrityReport(checks=checks, assurance_level=_assurance(checks), policy=policy,
                           currency=inputs.currency, input_hash=inputs.digest(), zipfolio_version=__version__,
                           run_id=inputs.run_id, engine=inputs.engine)


def _assurance(checks) -> Assurance:
    status = {check.id: check.status for check in checks}
    ran = {Status.PASS, Status.FAIL, Status.WARN}
    if status["CASH.LEDGER_REPLAY"] in ran and status["POSITIONS.LEDGER_REPLAY"] in ran:
        return Assurance.LEDGER
    if status["RETURNS.NAV_RECONCILIATION"] in ran and status["PNL.PORTFOLIO_RECONCILIATION"] in ran:
        return Assurance.RECONCILED
    return Assurance.SUMMARY


assert set(ASSURANCE_RANK) == set(Assurance)


# ----------------------------------------------------------------------------
# Shared pieces.
# ----------------------------------------------------------------------------

def _valuation(inputs, book) -> tuple[tuple | None, dict]:
    """Whether every instrument can be valued as quantity x price x multiplier, and the multipliers.

    Returns ``(problem, multipliers)``; ``problem`` is ``(reason_code, missing_inputs, symbols)`` or ``None``.
    """
    symbols = set(book.symbols)
    if inputs.opening_state is not None:
        symbols |= {s for s, q in inputs.opening_state.positions.items() if q}
    specs = inputs.contract_specs
    if specs is None:
        return ("MISSING_CONTRACT_SPECS", ["contract_specs"], sorted(symbols)), {}
    missing = sorted(symbol for symbol in symbols if symbol not in specs)
    if missing:
        return ("MISSING_CONTRACT_SPECS", ["contract_specs"], missing), {}
    unsupported = sorted(symbol for symbol in symbols if not specs[symbol].supported)
    if unsupported:
        return ("UNSUPPORTED_INSTRUMENT", [], unsupported), {}
    foreign = sorted(symbol for symbol in symbols
                     if specs[symbol].currency not in (None, inputs.currency))
    if foreign:
        return ("UNSUPPORTED_MULTICURRENCY", ["fx_rates"], foreign), {}
    return None, {symbol: float(specs[symbol].multiplier) for symbol in symbols}


def _snapshot_nav(book, multipliers) -> pd.Series | None:
    """Cash plus every position at its valuation price, per session; ``None`` without the snapshots."""
    if book.cash is None or book.quantities is None:
        return None
    quantities, prices = book.quantities, book.prices.reindex(columns=book.quantities.columns)
    held_without_price = (quantities != 0) & prices.isna()
    if held_without_price.any().any():
        day = held_without_price.any(axis="columns").idxmax()
        book.problem("INPUT.TIMESTAMP_ALIGNMENT", "position_without_valuation_price",
                     session=day.date().isoformat())
    scale = pd.Series({symbol: multipliers.get(symbol, 1.0) for symbol in quantities.columns}, dtype=float)
    values = (quantities * prices.fillna(0.0)).mul(scale, axis="columns") if len(scale) else quantities
    return book.cash + values.sum(axis="columns")


def _report_figures(inputs) -> dict | None:
    """The figures the report publishes, computed the way the page computes them; ``None`` without a book."""
    if inputs.cash is None or inputs.position_records is None:
        return None
    try:
        from ..accounting import external_flows, profit_breakdown
        from ..inputs import daily_returns, positions_in_dollars, transactions_frame
        from ..sections.instruments import commission_by_day, instrument_pnl
        from ..sections.round_trips import extract_round_trips
    except ImportError:
        return None
    returns = daily_returns(inputs.returns)
    holdings = positions_in_dollars(inputs.position_records, inputs.cash, returns.index)
    if holdings is None:
        return None
    fills = transactions_frame(inputs.transactions) if inputs.transactions is not None else None
    if fills is None:
        fills = pd.DataFrame({"amount": pd.Series(dtype=float), "price": pd.Series(dtype=float),
                              "symbol": pd.Series(dtype=str)}, index=pd.DatetimeIndex([], tz="UTC"))
    closed = extract_round_trips(returns, holdings, fills) if len(fills) else None
    breakdown = profit_breakdown(returns, holdings, fills, inputs.orders, closed,
                                 external_flows(inputs.portfolio_events, returns.index))
    gross = instrument_pnl(holdings, fills)
    net = gross - commission_by_day(inputs.orders, gross.index, gross.columns)
    sectors = None
    if inputs.sector_mappings is not None:
        from ..sections.sector_analysis import sector_profit

        sectors = sector_profit(returns, holdings, fills, inputs.orders, inputs.sector_mappings)
    return {"breakdown": breakdown, "closed": closed, "instrument_net": net.sum(), "fills": fills,
            "sectors": sectors, "returns": returns, "holdings": holdings}


def _skip(check_id, reason, missing=(), explanation=None, symbols=()) -> CheckResult:
    return CheckResult(check_id, True, Status.SKIPPED,
                       explanation or f"Not checked: {', '.join(missing) or reason} not given.",
                       reason_code=reason, missing_inputs=list(missing), affected_symbols=list(symbols))


def _valuation_skip(check_id, context) -> CheckResult | None:
    if context.valuation_problem is None:
        return None
    reason, missing, symbols = context.valuation_problem
    words = {"MISSING_CONTRACT_SPECS": "without each instrument's kind and multiplier, positions cannot be valued",
             "UNSUPPORTED_INSTRUMENT": "instruments other than cash equities are outside v1",
             "UNSUPPORTED_MULTICURRENCY": "instruments in another currency need FX rates, outside v1"}[reason]
    return _skip(check_id, reason, missing, f"Not checked: {words}.", symbols)


def _rows(frame: pd.DataFrame, limit: int) -> list:
    return [{key: (value.date().isoformat() if isinstance(value, pd.Timestamp) else value)
             for key, value in row.items()} for row in frame.head(limit).to_dict("records")]


def _sessions_label(day) -> str:
    return day.date().isoformat() if isinstance(day, pd.Timestamp) else str(day)


# ----------------------------------------------------------------------------
# The checks.
# ----------------------------------------------------------------------------

def _fill_uniqueness(c: Context) -> CheckResult:
    check = "INPUT.FILL_UNIQUENESS"
    if c.inputs.transactions is None or not len(c.inputs.transactions):
        return CheckResult(check, True, Status.NOT_APPLICABLE, "No fills: the backtest never traded.",
                           unit="count")
    issues = c.book.issues.get(check, [])
    fatal = [i for i in issues if i["problem"] in ("duplicate_fill_id", "invalid_fill")]
    status = Status.FAIL if fatal else Status.WARN if issues else Status.PASS
    count = 0 if c.book.fills is None else len(c.book.fills)
    words = {Status.PASS: f"All {count:,} fills have a unique id, a signed non-zero quantity and a positive price.",
             Status.WARN: "Every fill is usable, but some cannot be told apart or traded nothing.",
             Status.FAIL: "Some fills are duplicated or unreadable; they are left out of every replay."}
    return CheckResult(check, True, status, words[status], unit="count", observed=count,
                       sample_size=len(c.inputs.transactions), inputs_used=["transactions"],
                       reason_code=(fatal or issues or [{"problem": None}])[0]["problem"],
                       details=issues[: c.policy.max_failing_rows],
                       suggested_action=None if status == Status.PASS else
                       "Export each fill once, with its own id, signed quantity and execution price.")


def _timestamp_alignment(c: Context) -> CheckResult:
    check = "INPUT.TIMESTAMP_ALIGNMENT"
    if c.book.returns.empty:
        return CheckResult(check, True, Status.FAIL, "There are no returns to align anything with.",
                           reason_code="NO_RETURNS", unit="count", observed=0)
    if c.inputs.cash is None:
        return _skip(check, "MISSING_SNAPSHOTS", ["cash"],
                     "Not checked: without the cash snapshots the sessions of the book cannot be matched.")
    issues = c.book.issues.get(check, [])
    fatal = {"returns_not_numeric", "returns_out_of_order", "cash_snapshot_missing",
             "cash_snapshot_repeated_with_different_values", "position_snapshot_not_numeric",
             "fill_on_a_day_with_no_return", "position_without_valuation_price"}
    failing = [i for i in issues if i["problem"] in fatal]
    status = Status.FAIL if failing else Status.WARN if issues else Status.PASS
    words = {Status.PASS: f"Returns, cash and positions describe the same {len(c.book.sessions):,} sessions, "
                          "each once; every fill falls in one of them.",
             Status.WARN: "The series line up, with extra snapshots on days that have no return.",
             Status.FAIL: "The returns, the snapshots and the fills do not describe the same sessions."}
    first = next((i.get("session") or (i.get("sessions") or [None])[0] for i in failing), None)
    return CheckResult(check, True, status, words[status], unit="count", observed=len(c.book.sessions),
                       sample_size=len(c.book.sessions), first_failure_at=first,
                       reason_code=(failing or issues or [{"problem": None}])[0]["problem"],
                       inputs_used=["returns", "cash", "position_records", "transactions"],
                       details=issues[: c.policy.max_failing_rows])


def _flows(c: Context) -> pd.Series:
    """External cash in (+) or out (-) per session, end of day, from the event ledger."""
    flows = pd.Series(0.0, index=c.book.sessions)
    events = c.book.events
    if events is not None:
        from .models import EXTERNAL_EVENTS

        external = events[events["kind"].isin(EXTERNAL_EVENTS)]
        flows = flows.add(external.groupby("session")["cash"].sum(), fill_value=0.0).reindex(c.book.sessions,
                                                                                                fill_value=0.0)
    return flows


def _nav_snapshot(c: Context) -> CheckResult:
    check = "NAV.SNAPSHOT_RECONCILIATION"
    skip = _valuation_skip(check, c)
    if skip:
        return skip
    if c.snapshot_nav is None:
        return _skip(check, "MISSING_SNAPSHOTS", [n for n in ("cash", "position_records")
                                                  if getattr(c.inputs, n) is None])
    opening = c.inputs.opening_state
    if opening is None or opening.value is None:
        return _skip(check, "MISSING_OPENING_STATE", ["opening_state"],
                     "Not checked: the NAV the returns imply needs the value the book started from.")
    flows = _flows(c)
    implied, value = [], float(opening.value)
    for day, r in c.book.returns.items():
        value = value * (1 + r) + flows[day]
        implied.append(value)
    implied = pd.Series(implied, index=c.book.sessions)
    gap = c.snapshot_nav - implied
    tolerance = implied.abs().map(c.tol)
    bad = gap.abs() > tolerance
    worst = gap.abs().idxmax()
    status = Status.FAIL if bad.any() else Status.PASS
    rows = pd.DataFrame({"session": gap.index[bad], "snapshot_nav": c.snapshot_nav[bad].round(2),
                         "nav_from_returns": implied[bad].round(2), "difference": gap[bad].round(6)})
    return CheckResult(
        check, True, status,
        f"Cash plus every position at its valuation price, against the opening value compounded by the "
        f"engine's returns, on each of {len(gap):,} sessions." + (
            f" They part on {int(bad.sum()):,} sessions, first on {_sessions_label(gap.index[bad][0])}."
            if bad.any() else " They agree on every one."),
        observed=float(c.snapshot_nav.iloc[-1]), expected=float(implied.iloc[-1]), difference=float(gap[worst]),
        tolerance=float(tolerance[worst]), sample_size=len(gap),
        first_failure_at=_sessions_label(gap.index[bad][0]) if bad.any() else None,
        inputs_used=["cash", "position_records", "returns", "opening_state", "contract_specs"],
        reason_code="NAV_MISMATCH" if bad.any() else None, details=_rows(rows, c.policy.max_failing_rows),
        suggested_action=None if status == Status.PASS else
        "Compare the engine's portfolio value with cash plus marked positions on the first failing session.")


def _returns_nav(c: Context) -> CheckResult:
    check = "RETURNS.NAV_RECONCILIATION"
    skip = _valuation_skip(check, c)
    if skip:
        return skip
    if c.snapshot_nav is None:
        return _skip(check, "MISSING_SNAPSHOTS", [n for n in ("cash", "position_records")
                                                  if getattr(c.inputs, n) is None])
    nav, flows = c.snapshot_nav, _flows(c)
    previous = nav.shift(1)
    opening = c.inputs.opening_state
    if opening is not None and opening.value is not None:
        previous.iloc[0] = opening.value
    usable = previous.notna() & (previous > 0)
    from_nav = (nav - flows) / previous - 1
    gap = (c.book.returns - from_nav)[usable]
    if gap.empty:
        return _skip(check, "TOO_FEW_SESSIONS", [], "Not checked: fewer than two sessions with a positive NAV.")
    tolerance = previous[usable].map(lambda value: max(c.policy.return_tolerance, c.tol(value) / value))
    bad = gap.abs() > tolerance
    worst = gap.abs().idxmax()
    status = Status.FAIL if bad.any() else Status.PASS
    start = "" if opening is not None and opening.value is not None else " from the second session (no opening state)"
    rows = pd.DataFrame({"session": gap.index[bad], "engine_return": c.book.returns[gap.index[bad]],
                         "return_from_nav": from_nav[gap.index[bad]], "difference": gap[bad]})
    return CheckResult(
        check, True, status,
        f"Each session's return against the change in cash plus positions{start}, external flows taken out at "
        f"the end of the day." + (f" {int(bad.sum()):,} sessions disagree." if bad.any() else
                                  f" All {len(gap):,} agree."),
        unit="return", observed=float(c.book.returns[worst]), expected=float(from_nav[worst]),
        difference=float(gap[worst]), tolerance=float(tolerance[worst]), sample_size=len(gap),
        first_failure_at=_sessions_label(gap.index[bad][0]) if bad.any() else None,
        inputs_used=["returns", "cash", "position_records", "contract_specs"] + (
            ["portfolio_events"] if c.events_declared else []),
        reason_code="RETURN_MISMATCH" if bad.any() else None, details=_rows(rows, c.policy.max_failing_rows),
        suggested_action=None if status == Status.PASS else
        "A return that does not match the book was changed after the engine computed it, or the book was.")


def _unexplained(check, c: Context, difference: float, text: str, **fields) -> CheckResult:
    """A gap the fills and fees do not explain: a failure with a complete event ledger, otherwise a gap that
    dividends, interest or a split could explain -- not checkable, never a pass."""
    if c.events_declared:
        return CheckResult(check, True, Status.FAIL, text, difference=difference, **fields)
    fields.pop("reason_code", None)
    return CheckResult(check, True, Status.SKIPPED,
                       text + " No event ledger was given, so dividends, interest, fees or a split could explain it.",
                       difference=difference, reason_code="UNEXPLAINED_WITHOUT_EVENT_LEDGER",
                       missing_inputs=["portfolio_events"], **fields)


def _cash_replay(c: Context) -> CheckResult:
    check = "CASH.LEDGER_REPLAY"
    skip = _valuation_skip(check, c)
    if skip:
        return skip
    missing = [name for name, value in (("opening_state", c.inputs.opening_state), ("cash", c.inputs.cash))
               if value is None]
    if c.book.fills is not None and len(c.book.fills) and c.fees is None:
        missing.append("orders")
    if missing:
        return _skip(check, "INSUFFICIENT_EVENT_LEDGER", missing)
    recorded, replayed = c.book.cash, c.ledger.cash
    ambiguous = c.fees.ambiguous_sessions if c.fees is not None else pd.DatetimeIndex([], tz="UTC")
    compared = recorded.index.difference(ambiguous)
    gap = (recorded - replayed)[compared]
    tolerance = recorded[compared].abs().map(c.tol)
    bad = gap.abs() > tolerance
    worst = gap.abs().idxmax()
    fields = dict(observed=float(recorded.iloc[-1]), expected=float(replayed.iloc[-1]),
                  tolerance=float(tolerance[worst]), sample_size=len(compared),
                  first_failure_at=_sessions_label(gap.index[bad][0]) if bad.any() else None,
                  inputs_used=["opening_state", "transactions", "orders", "cost_settings", "cash"]
                  + (["portfolio_events"] if c.events_declared else []),
                  details=_rows(pd.DataFrame({"session": gap.index[bad], "recorded": recorded[gap.index[bad]],
                                              "replayed": replayed[gap.index[bad]], "difference": gap[bad]}),
                                c.policy.max_failing_rows),
                  reason_code="CASH_MISMATCH")
    booking = {"fee_model": "each fill's fee from the run's fee model, which matches every order's charge",
               "order_session": "each order's charge on the session it filled in",
               "order_session_partly_ambiguous": "each order's charge on the session of its last fill"}
    method = booking.get(c.fees.method if c.fees is not None else "", "no fees")
    if bad.any():
        return _unexplained(check, c, float(gap[worst]),
                            f"Cash replayed from the opening state, the fills and the fees ({method}) differs from "
                            f"the cash recorded on {int(bad.sum()):,} sessions.", **fields)
    fields["reason_code"] = None
    if len(ambiguous):
        return CheckResult(check, True, Status.WARN,
                           f"Cash replays to the recorded balance on every session it can be compared on; "
                           f"{len(ambiguous):,} sessions are left out because an order that filled across them "
                           "carries one cumulative charge, with no record of what each day paid.",
                           difference=float(gap[worst]), **{**fields, "reason_code": "FEE_BOOKING_AMBIGUOUS"})
    return CheckResult(check, True, Status.PASS,
                       f"Cash replayed from the opening state, the fills and the fees ({method}) equals the cash "
                       f"recorded on all {len(compared):,} sessions.", difference=float(gap[worst]), **fields)


def _position_replay(c: Context) -> CheckResult:
    check = "POSITIONS.LEDGER_REPLAY"
    skip = _valuation_skip(check, c)
    if skip:
        return skip
    missing = [name for name, value in (("opening_state", c.inputs.opening_state),
                                        ("position_records", c.inputs.position_records), ("cash", c.inputs.cash))
               if value is None]
    if missing:
        return _skip(check, "MISSING_OPENING_STATE" if "opening_state" in missing else "MISSING_SNAPSHOTS", missing)
    columns = c.book.quantities.columns.union(c.ledger.quantities.columns)
    recorded = c.book.quantities.reindex(columns=columns, fill_value=0.0)
    replayed = c.ledger.quantities.reindex(columns=columns, fill_value=0.0)
    gap = recorded - replayed
    bad = gap.abs() > SHARE_TOLERANCE
    sessions_bad = bad.any(axis="columns")
    symbols = sorted(columns[bad.any(axis="index")])
    rows = gap.where(bad).stack().reset_index()
    rows.columns = ["session", "symbol", "difference"]
    fields = dict(unit="shares", observed=float(recorded.iloc[-1].abs().sum()),
                  expected=float(replayed.iloc[-1].abs().sum()), tolerance=SHARE_TOLERANCE,
                  sample_size=int(recorded.size), affected_symbols=symbols,
                  first_failure_at=_sessions_label(sessions_bad.idxmax()) if sessions_bad.any() else None,
                  inputs_used=["opening_state", "transactions", "position_records"]
                  + (["portfolio_events"] if c.events_declared else []),
                  details=_rows(rows, c.policy.max_failing_rows), reason_code="INVENTORY_MISMATCH")
    if bad.any().any():
        return _unexplained(check, c, float(gap.abs().to_numpy().max()),
                            f"Shares replayed from the opening state and the fills differ from the recorded "
                            f"positions in {len(symbols):,} names, on {int(sessions_bad.sum()):,} sessions.", **fields)
    fields["reason_code"] = None
    return CheckResult(check, True, Status.PASS,
                       f"Every name's shares, replayed from the opening state and the fills, equal the recorded "
                       f"position on all {len(recorded):,} sessions; the figures are the shares held at the end.",
                       difference=0.0, **fields)


def _economic(c: Context) -> dict | None:
    """Profit from the book itself: every name's change in value less what its fills cost, events, fees."""
    if c.snapshot_nav is None or c.book.quantities is None:
        return None
    opening = c.inputs.opening_state
    fills = c.book.fills if c.book.fills is not None else pd.DataFrame(columns=["session", "symbol", "quantity",
                                                                                 "price"])
    if opening is not None and opening.value is not None:
        start_value, window = float(opening.value), c.book.sessions
        start_positions = {s: q * opening.prices.get(s, 0.0) * c.multipliers.get(s, 1.0)
                           for s, q in opening.positions.items() if q}
        basis = "the opening state"
    else:
        if len(c.book.sessions) < 2:
            return None
        start_value, window = float(c.snapshot_nav.iloc[0]), c.book.sessions[1:]
        first = c.book.sessions[0]
        start_positions = {s: float(c.book.quantities.at[first, s] * c.book.prices.at[first, s]
                                    * c.multipliers.get(s, 1.0))
                           for s in c.book.quantities.columns if c.book.quantities.at[first, s]}
        basis = "the first session's snapshot (no opening state)"
    last = c.book.sessions[-1]
    end_positions = {s: float(c.book.quantities.at[last, s] * c.book.prices.at[last, s] * c.multipliers.get(s, 1.0))
                     for s in c.book.quantities.columns if c.book.quantities.at[last, s]}
    in_window = fills[fills["session"].isin(window)]
    spent = (in_window["quantity"] * in_window["price"]
             * in_window["symbol"].map(lambda s: c.multipliers.get(s, 1.0))).groupby(in_window["symbol"]).sum()
    names = set(end_positions) | set(start_positions) | set(spent.index)
    gross = {s: end_positions.get(s, 0.0) - start_positions.get(s, 0.0) - float(spent.get(s, 0.0)) for s in names}
    fees = 0.0
    if c.fees is not None:
        fees = float(c.fees.by_session.reindex(window, fill_value=0.0).sum())
    events = c.book.events
    income = expenses = 0.0
    if events is not None:
        from .models import EXTERNAL_EVENTS

        own = events[events["session"].isin(window) & ~events["kind"].isin(EXTERNAL_EVENTS)]["cash"].fillna(0.0)
        income, expenses = float(own[own > 0].sum()), float(-own[own < 0].sum())
    flows = float(_flows(c).reindex(window, fill_value=0.0).sum())
    end_value = float(c.snapshot_nav.iloc[-1])
    return {"gross": gross, "fees": fees, "income": income, "expenses": expenses, "flows": flows,
            "start_value": start_value, "end_value": end_value, "basis": basis,
            "economic": sum(gross.values()) - fees + income - expenses,
            "nav_change": end_value - start_value - flows}


def _summary_identity(c: Context) -> CheckResult:
    check = "PNL.REPORTED_SUMMARY_IDENTITY"
    if c.report is None:
        return _skip(check, "REPORT_FIGURES_UNAVAILABLE", [n for n in ("cash", "position_records")
                                                           if getattr(c.inputs, n) is None],
                     "Not checked: the report has no book-based profit to check without cash and positions.")
    p = c.report["breakdown"]
    implied = p.from_returns
    status = Status.PASS if abs(p.returns_gap) <= p.tolerance else Status.FAIL
    return CheckResult(
        check, True, status,
        "The published realized plus unrealized PnL, less commission, plus other cash, against the profit the "
        "published returns imply, external flows taken out. This proves the published figures agree with each "
        "other; it replays no book.",
        observed=p.net, expected=implied, difference=p.returns_gap, tolerance=p.tolerance,
        evidence_level="report_only", inputs_used=["published report figures"],
        reason_code=None if status == Status.PASS else "REPORTED_TOTALS_DISAGREE")


def _portfolio_reconciliation(c: Context) -> CheckResult:
    check = "PNL.PORTFOLIO_RECONCILIATION"
    skip = _valuation_skip(check, c)
    if skip:
        return skip
    if c.book.fills is not None and len(c.book.fills) and c.fees is None:
        return _skip(check, "MISSING_FEE_LEDGER", ["orders"])
    e = _economic(c)
    if e is None:
        return _skip(check, "MISSING_SNAPSHOTS", [n for n in ("cash", "position_records")
                                                  if getattr(c.inputs, n) is None])
    difference = e["economic"] - e["nav_change"]
    tolerance = c.tol(e["start_value"])
    fields = dict(observed=e["economic"], expected=e["nav_change"], tolerance=tolerance,
                  inputs_used=["position_records", "cash", "transactions", "orders", "contract_specs"]
                  + (["opening_state"] if c.inputs.opening_state is not None else [])
                  + (["portfolio_events"] if c.events_declared else []),
                  reason_code="PROFIT_MISMATCH",
                  details=[{"component": "instruments, before fees", "amount": round(sum(e["gross"].values()), 6)},
                           {"component": "commissions", "amount": round(-e["fees"], 6)},
                           {"component": "income events", "amount": round(e["income"], 6)},
                           {"component": "expense events", "amount": round(-e["expenses"], 6)},
                           {"component": "external flows (excluded)", "amount": round(e["flows"], 6)}])
    text = (f"Each name's change in value, less what its fills cost, less commissions, plus income and expense "
            f"events, against the change in cash plus positions net of external flows, measured from {e['basis']}.")
    if abs(difference) > tolerance:
        return _unexplained(check, c, difference, text + f" They differ by {difference:,.2f}.", **fields)
    fields["reason_code"] = None
    return CheckResult(check, True, Status.PASS, text + " They agree.", difference=difference, **fields)


def _realized_unrealized(c: Context) -> CheckResult:
    check = "PNL.REALIZED_UNREALIZED"
    skip = _valuation_skip(check, c)
    if skip:
        return skip
    if c.ledger is None:
        return _skip(check, "MISSING_OPENING_STATE", ["opening_state"] + (["cash"] if c.inputs.cash is None else []))
    if not c.ledger_holds():
        return _skip(check, "LEDGER_NOT_RECONCILED", [],
                     "Not checked: the replayed positions or cash do not match the record, so the replayed lots "
                     "are not the book's.")
    if c.ledger.basis_unknown:
        return _skip(check, "MISSING_COST_BASIS", ["opening_state.cost_basis"],
                     "Not checked: some shares entered the book without a cost, so their profit cannot be "
                     "split into realized and unrealized.", symbols=sorted(c.ledger.basis_unknown))
    if c.report is None:
        return _skip(check, "REPORT_FIGURES_UNAVAILABLE", [])
    p = c.report["breakdown"]
    ledger_unrealized = float(sum(c.ledger.unrealized_by_symbol.values()))
    ledger_total = c.ledger.realized + ledger_unrealized
    total_gap = p.trading - ledger_total
    split_gap = p.realized - c.ledger.realized
    tolerance = c.tol(abs(ledger_total))
    details = [{"figure": "realized", "report": round(p.realized, 6), "ledger_fifo": round(c.ledger.realized, 6)},
               {"figure": "unrealized", "report": round(p.unrealized, 6), "ledger_fifo": round(ledger_unrealized, 6)}]
    fields = dict(observed=p.realized, expected=c.ledger.realized, tolerance=tolerance, details=details,
                  inputs_used=["transactions", "position_records", "opening_state"])
    if abs(total_gap) > tolerance:
        return CheckResult(check, True, Status.FAIL,
                           f"The report's realized plus unrealized PnL differs from the ledger's by {total_gap:,.2f}: "
                           "profit is missing or counted twice.", difference=total_gap,
                           reason_code="PNL_TOTAL_MISMATCH", **fields)
    if abs(split_gap) > tolerance:
        return CheckResult(check, True, Status.WARN,
                           f"Realized plus unrealized agree, but the split differs by {split_gap:,.2f}: the report's "
                           "trades (pyfolio) merge same-side fills within 8 hours at their average price and count "
                           "whole shares, where the ledger matches fill by fill, first in, first out.",
                           difference=split_gap, reason_code="LOT_CONVENTION_DIFFERS", **fields)
    return CheckResult(check, True, Status.PASS,
                       f"Realized ({c.ledger.realized:,.2f}) and unrealized ({ledger_unrealized:,.2f}) PnL, before "
                       "commission, match the ledger's first-in-first-out lots.", difference=split_gap, **fields)


def _commission(c: Context) -> CheckResult:
    check = "COST.COMMISSION_RECONCILIATION"
    has_fills = c.book.fills is not None and len(c.book.fills)
    if c.fees is None:
        if not has_fills:
            return CheckResult(check, True, Status.NOT_APPLICABLE, "No fills and no orders: nothing was charged.")
        return _skip(check, "MISSING_FEE_LEDGER", ["orders"])
    total = float(c.fees.order_totals.sum())
    issues = list(c.book.issues.get(check, []))
    reported = c.report["breakdown"].commission if c.report is not None else None
    problems = []
    if reported is not None and abs(reported - total) > c.tol(total):
        problems.append("the report's commission is not the orders' total")
    if c.fees.mismatches:
        problems.append(f"{len(c.fees.mismatches):,} orders were charged other than the fee model says")
    if any(i["problem"] == "order_rows_disagree_on_commission" for i in issues):
        problems.append("repeated rows of one order disagree on its charge")
    unfilled = [order_id for order_id, charge in c.fees.order_totals.items()
                if charge and (not has_fills or order_id not in set(c.book.fills["order_id"].dropna()))]
    if unfilled and c.fees.modeled_totals is None:
        problems.append(f"{len(unfilled):,} orders were charged without a fill")
    modeled = float(c.fees.modeled_totals.sum()) if c.fees.modeled_totals is not None else None
    repeated = c.book.order_rows - len(c.fees.order_totals)
    note = (f" The orders came as {c.book.order_rows:,} rows for {len(c.fees.order_totals):,} orders; each is "
            "counted once." if repeated else "")
    details = c.fees.mismatches[: c.policy.max_failing_rows] + issues[: c.policy.max_failing_rows]
    fields = dict(observed=total, expected=modeled if modeled is not None else reported,
                  difference=(total - modeled) if modeled is not None else
                  (None if reported is None else reported - total),
                  tolerance=c.tol(total), inputs_used=["orders", "transactions", "cost_settings"],
                  details=details, missing_inputs=[] if c.inputs.cost_settings else ["cost_settings"])
    if problems:
        return CheckResult(check, True, Status.FAIL, "; ".join(problems).capitalize() + "." + note,
                           reason_code="FEE_MISMATCH", **fields)
    model = ("and every order's charge is what the run's fee model gives for its fills" if modeled is not None else
             "(no fee model given, so the charges themselves are not recomputed)")
    return CheckResult(check, True, Status.PASS,
                       f"The report's commission, {total:,.2f}, is the orders' total, each order once, {model}." + note,
                       **fields)


def _slippage(c: Context) -> CheckResult:
    check = "COST.SLIPPAGE_SINGLE_COUNT"
    if c.book.fills is None or not len(c.book.fills):
        return CheckResult(check, True, Status.NOT_APPLICABLE, "No fills: no slippage was paid.")
    skip = _valuation_skip(check, c)
    if skip:
        return skip
    if c.report is None or c.fees is None:
        return _skip(check, "REPORT_FIGURES_UNAVAILABLE", ["orders"] if c.fees is None else [])
    e = _economic(c)
    if e is None:
        return _skip(check, "MISSING_SNAPSHOTS", [])
    settings = c.inputs.cost_settings or {}
    bps = float(settings.get("slippage_basis_points") or 0.0) / 1e4
    fills = c.book.fills
    buys = fills["quantity"] > 0
    reconstructed = float(((fills["quantity"] * fills["price"]).abs() * bps / np.where(buys, 1 + bps, 1 - bps)).sum())
    published = float(c.report["instrument_net"].sum())
    expected = sum(e["gross"].values()) - e["fees"] if c.inputs.opening_state is not None else None
    if expected is None:
        # Without an opening state the instrument table covers the first session too; compare with the book.
        expected = c.report["breakdown"].trading - c.report["breakdown"].commission
    gap = published - expected
    tolerance = c.tol(abs(expected))
    details = [{"figure": "slippage reconstructed by the report", "amount": round(reconstructed, 6)}]
    embedded = None
    bars = c.inputs.fill_bars
    if bars is not None and "close" in bars and bps:
        reference = bars.reindex(fills["fill_id"])["close"].to_numpy(dtype=float)
        if np.isfinite(reference).all():
            embedded = float((np.abs(fills["price"].to_numpy() - reference) * fills["quantity"].abs().to_numpy()).sum())
            details.append({"figure": "slippage in the fill prices (|price - bar close| x shares)",
                            "amount": round(embedded, 6)})
    fields = dict(observed=published, expected=expected, difference=gap, tolerance=tolerance, details=details,
                  inputs_used=["transactions", "orders", "position_records", "cost_settings"]
                  + (["fill_bars"] if embedded is not None else []))
    if reconstructed > tolerance and abs(gap + reconstructed) <= tolerance:
        return CheckResult(check, True, Status.FAIL,
                           "The published net PnL is the book's less the reconstructed slippage: slippage is "
                           "already in the fill prices and was taken off a second time.",
                           reason_code="SLIPPAGE_DEDUCTED_TWICE", **fields)
    if abs(gap) > tolerance:
        return CheckResult(check, True, Status.FAIL,
                           f"The published net PnL differs from the book's by {gap:,.2f}.",
                           reason_code="NET_PNL_MISMATCH", **fields)
    if embedded is not None and abs(embedded - reconstructed) > c.tol(reconstructed):
        return CheckResult(check, True, Status.WARN,
                           "Slippage is counted once -- in the fill prices -- but the report's reconstruction of it "
                           f"({reconstructed:,.2f}) is not what the prices hold ({embedded:,.2f}).",
                           reason_code="SLIPPAGE_RECONSTRUCTION_DIFFERS", **fields)
    return CheckResult(check, True, Status.PASS,
                       "Slippage is counted once: it is in the fill prices, the published net PnL is the book's "
                       f"to the cent, and the reconstructed {reconstructed:,.2f} is shown as an estimate, never "
                       "taken off." + (" The reconstruction equals what the prices hold against their bars."
                                       if embedded is not None else ""), **fields)


def _instrument_sum(c: Context) -> CheckResult:
    check = "ATTRIBUTION.INSTRUMENT_SUM"
    if c.report is None:
        return _skip(check, "REPORT_FIGURES_UNAVAILABLE", [n for n in ("cash", "position_records")
                                                           if getattr(c.inputs, n) is None])
    skip = _valuation_skip(check, c)
    if skip:
        return skip
    published = c.report["instrument_net"]
    p = c.report["breakdown"]
    # The instruments carry their fills and fees; events not tied to a name are left out of them.
    expected = p.trading - (p.commission or 0.0)
    gap = float(published.sum()) - expected
    tolerance = c.tol(abs(expected))
    affected, details = [], []
    if c.ledger_holds() and c.fees is not None and not c.ledger.basis_unknown:
        for symbol, value in published.items():
            own = (c.ledger.realized_by_symbol.get(symbol, 0.0) + c.ledger.unrealized_by_symbol.get(symbol, 0.0)
                   - float(c.fees.by_symbol.get(symbol, 0.0)))
            if abs(value - own) > c.tol(abs(own)):
                affected.append(symbol)
                details.append({"symbol": symbol, "report": round(float(value), 6), "ledger": round(own, 6)})
    status = Status.FAIL if abs(gap) > tolerance or affected else Status.PASS
    return CheckResult(
        check, True, status,
        f"The instruments' net PnL adds up to {float(published.sum()):,.2f}, against the book's PnL after "
        f"commission, {expected:,.2f}" + (", and each name matches its ledger lots less its fees."
                                          if c.ledger_holds() and not affected else ".")
        + (f" {len(affected):,} names do not match their ledger." if affected else "")
        + ("" if c.ledger_holds() or c.ledger is None else
           " The names are not compared with the ledger: its replay does not match the record."),
        observed=float(published.sum()), expected=expected, difference=gap, tolerance=tolerance,
        affected_symbols=affected, details=details[: c.policy.max_failing_rows], sample_size=len(published),
        inputs_used=["position_records", "cash", "transactions", "orders"],
        reason_code="ATTRIBUTION_MISMATCH" if status == Status.FAIL else None)


def _sector_sum(c: Context) -> CheckResult:
    check = "ATTRIBUTION.SECTOR_SUM"
    if c.inputs.sector_mappings is None:
        return CheckResult(check, False, Status.NOT_APPLICABLE, "No sector mappings: no sector table to check.")
    if c.report is None or c.report["sectors"] is None:
        return _skip(check, "REPORT_FIGURES_UNAVAILABLE", [])
    from ..sectors import UNCLASSIFIED

    sectors, instruments = c.report["sectors"], c.report["instrument_net"]
    unmapped = sorted(s for s in instruments.index if s not in c.inputs.sector_mappings)
    gap = float(sectors["pnl"].sum() - instruments.sum())
    tolerance = c.tol(abs(float(instruments.sum())))
    bucket = UNCLASSIFIED in sectors.index
    status = Status.FAIL if abs(gap) > tolerance or (unmapped and not bucket) else Status.PASS
    return CheckResult(
        check, False, status,
        "The sectors add up to the instruments' net PnL" + (
            f", with {len(unmapped):,} names in an explicit '{UNCLASSIFIED}' bucket." if unmapped else ".")
        if status == Status.PASS else f"The sectors miss {gap:,.2f} of the instruments' net PnL.",
        observed=float(sectors["pnl"].sum()), expected=float(instruments.sum()), difference=gap, tolerance=tolerance,
        affected_symbols=unmapped, inputs_used=["sector_mappings"], sample_size=len(sectors),
        reason_code=None if status == Status.PASS else "SECTOR_DROPS_PNL")


def _synthetic_closures(c: Context) -> CheckResult:
    check = "ROUND_TRIP.SYNTHETIC_CLOSURES"
    if c.book.fills is None or not len(c.book.fills):
        return CheckResult(check, False, Status.NOT_APPLICABLE, "No fills: no trades.", unit="count")
    if c.report is None:
        return _skip(check, "REPORT_FIGURES_UNAVAILABLE", [])
    closed = c.report["closed"]
    count = 0 if closed is None else len(closed)
    fills = c.report["fills"]
    synthetic = 0
    if count:
        stamps = {(str(symbol), pd.Timestamp(dt)) for dt, symbol in zip(fills.index, fills["symbol"])}
        synthetic = sum((str(row.symbol), pd.Timestamp(row.close_dt)) not in stamps for row in closed.itertuples())
    open_lots = c.ledger.open_lots if c.ledger is not None else None
    unrealized = c.report["breakdown"].unrealized
    if not synthetic and c.ledger_holds() and not c.ledger.basis_unknown:
        # pyfolio merges a paper sale into a real one within 8 hours: no row without a fill, but the realized
        # PnL then carries the open positions' profit, and what is left unrealized is not what is held.
        ledger_open = float(sum(c.ledger.unrealized_by_symbol.values()))
        tolerance = c.tol(abs(ledger_open))
        if abs(ledger_open) > tolerance and abs(unrealized) <= tolerance and \
                abs(c.report["breakdown"].realized - (c.ledger.realized + ledger_open)) <= tolerance:
            synthetic = c.report["breakdown"].open_names
    details = [{"view": "closed trades in the report", "count": count, "backed_by_a_fill": count - synthetic},
               {"view": "positions open at the end (unrealized, not trades)",
                "count": c.report["breakdown"].open_names, "gross_pnl": round(unrealized, 6)}]
    if synthetic:
        return CheckResult(check, False, Status.WARN,
                           f"The closed trades include {synthetic:,} positions still held, sold on paper at the end "
                           "and counted as executed trades: their profit is in the realized PnL.",
                           unit="count", observed=synthetic, expected=0, reason_code="SYNTHETIC_CLOSE_COUNTED",
                           details=details, inputs_used=["transactions"])
    return CheckResult(check, False, Status.PASS,
                       f"All {count:,} closed trades close on an actual fill. The {c.report['breakdown'].open_names:,} "
                       f"positions held at the end are unrealized PnL ({unrealized:,.2f}), not trades."
                       + (f" The ledger holds {open_lots:,} open lots." if open_lots is not None else ""),
                       unit="count", observed=0, expected=0, details=details, inputs_used=["transactions"])


#: Where the page shows trade PnL -- before commission -- and so must not call it net.
TRADE_SECTIONS = ("Closed Trades", "Profit Concentration")
_NET = re.compile(r"\bnet\b", re.I)


def _labels(html_text: str) -> list:
    text = re.sub(r'<span class="term"[^>]*>(.*?)</span>', r"\1", html_text)
    cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", text, re.S) + re.findall(r"<h3>(.*?)</h3>", text, re.S)
    return [re.sub(r"<[^>]+>", "", cell).strip() for cell in cells]


def _gross_net(c: Context) -> CheckResult:
    check = "ROUND_TRIP.GROSS_NET_CONVENTION"
    if c.sections is None:
        return CheckResult(check, False, Status.NOT_APPLICABLE, "No rendered report to read.")
    trade = [s for s in c.sections if s.title in TRADE_SECTIONS and not s.failed]
    if not trade:
        return CheckResult(check, False, Status.NOT_APPLICABLE, "The report shows no trade PnL.")
    labels = [label for s in trade for label in _labels("".join(s.side + s.wide))]
    wrong = sorted({label for label in labels if _NET.search(label) and "before commission" not in label.lower()})
    stated = any("before commission" in label.lower() for label in labels)
    if wrong:
        return CheckResult(check, False, Status.WARN,
                           "Trade PnL is before commission, but the report calls it net: " + "; ".join(wrong) + ".",
                           unit="count", observed=len(wrong), expected=0, reason_code="GROSS_PNL_LABELED_NET",
                           details=[{"label": label} for label in wrong])
    if not stated:
        return CheckResult(check, False, Status.WARN,
                           "The report shows trade PnL without saying whether commission is in it.",
                           unit="count", reason_code="CONVENTION_NOT_STATED")
    return CheckResult(check, False, Status.PASS,
                       "Trade PnL is labelled before commission wherever the report shows it, and nothing gross is "
                       "called net; commissions are taken off once, in the instruments' net PnL and the "
                       "reconciliation.", unit="count", observed=0, expected=0)


def _money(text: str) -> float | None:
    cleaned = re.sub(r"[^\d.\-−]", "", text.replace("−", "-"))
    try:
        return float(cleaned)
    except ValueError:
        return None


def _display(c: Context) -> CheckResult:
    check = "REPORT.DISPLAY_VS_CALCULATION"
    if c.sections is None:
        return CheckResult(check, False, Status.NOT_APPLICABLE, "No rendered report to read.")
    if c.report is None:
        return _skip(check, "REPORT_FIGURES_UNAVAILABLE", [])
    performance = next((s for s in c.sections if s.title == "Performance" and not s.failed), None)
    html_text = "".join(performance.side + performance.wide) if performance is not None else ""
    match = re.search(r"Profit Reconciliation.*?<table>(.*?)</table>", html_text, re.S)
    if not match:
        return CheckResult(check, False, Status.NOT_APPLICABLE, "The report shows no profit reconciliation.")
    shown = {}
    for row in re.findall(r"<tr>(.*?)</tr>", match.group(1), re.S):
        cells = [re.sub(r"<[^>]+>", "", cell).strip() for cell in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
        if len(cells) == 2:
            shown[cells[0]] = cells[1]
    p = c.report["breakdown"]
    expected = {"Realized": p.realized, "Unrealized": p.unrealized, "Commission": -(p.commission or 0.0),
                "Net profit: change": p.net, "Net profit from the daily returns": p.from_returns}
    wrong, details = [], []
    for prefix, value in expected.items():
        label = next((label for label in shown if label.startswith(prefix)), None)
        if label is None:
            if prefix == "Commission" and p.commission is None:
                continue
            wrong.append(prefix)
            details.append({"row": prefix, "shown": None, "calculated": round(value, 2)})
            continue
        number = _money(shown[label])
        if number is None or abs(number - value) > 0.005 + c.tol(abs(value)):
            wrong.append(label)
        details.append({"row": label, "shown": shown[label], "calculated": round(value, 2)})
    status = Status.FAIL if wrong else Status.PASS
    return CheckResult(check, False, status,
                       "Every row of the published profit reconciliation is the calculated figure, rounded to the "
                       "cent." if not wrong else f"Shown and calculated differ: {'; '.join(wrong)}.",
                       unit="count", observed=len(wrong), expected=0, details=details,
                       reason_code="DISPLAY_MISMATCH" if wrong else None)


_CHECKS = {
    "INPUT.FILL_UNIQUENESS": _fill_uniqueness,
    "INPUT.TIMESTAMP_ALIGNMENT": _timestamp_alignment,
    "NAV.SNAPSHOT_RECONCILIATION": _nav_snapshot,
    "RETURNS.NAV_RECONCILIATION": _returns_nav,
    "CASH.LEDGER_REPLAY": _cash_replay,
    "POSITIONS.LEDGER_REPLAY": _position_replay,
    "PNL.REPORTED_SUMMARY_IDENTITY": _summary_identity,
    "PNL.PORTFOLIO_RECONCILIATION": _portfolio_reconciliation,
    "PNL.REALIZED_UNREALIZED": _realized_unrealized,
    "COST.COMMISSION_RECONCILIATION": _commission,
    "COST.SLIPPAGE_SINGLE_COUNT": _slippage,
    "ATTRIBUTION.INSTRUMENT_SUM": _instrument_sum,
    "ATTRIBUTION.SECTOR_SUM": _sector_sum,
    "ROUND_TRIP.SYNTHETIC_CLOSURES": _synthetic_closures,
    "ROUND_TRIP.GROSS_NET_CONVENTION": _gross_net,
    "REPORT.DISPLAY_VS_CALCULATION": _display,
}
assert tuple(_CHECKS) == CHECK_IDS
