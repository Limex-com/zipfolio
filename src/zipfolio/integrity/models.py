"""The integrity check's data: what goes in, how it is judged, and the report that comes out.

The report is deterministic: the same inputs and policy give the same JSON, byte for byte -- stable
check order, ISO-8601 timestamps, money as decimal strings with its currency, no NaN or Infinity,
and no generation time. It names the Zipfolio version and hashes the inputs and the policy instead.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum

import pandas as pd

SCHEMA_VERSION = "accounting-integrity/v1"


class Status(str, Enum):
    """One check's outcome."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIPPED = "SKIPPED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Overall(str, Enum):
    """The report's outcome: FAIL if a required invariant fails, INCOMPLETE if one could not be checked,
    WARN if all held but a definition or convention is in doubt, PASS only when nothing is left open."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    INCOMPLETE = "INCOMPLETE"


class Assurance(str, Enum):
    """How much the report could check, independently of what it found.

    ``summary``: the published figures agree with each other -- no book replayed.
    ``reconciled``: NAV, positions and costs recomputed from the period snapshots.
    ``ledger``: every book event replayed from an explicit opening state.
    """

    SUMMARY = "summary"
    RECONCILED = "reconciled"
    LEDGER = "ledger"


ASSURANCE_RANK = {Assurance.SUMMARY: 0, Assurance.RECONCILED: 1, Assurance.LEDGER: 2}

#: Instrument kinds valued as quantity x price x multiplier, fully paid, with no accruals or margin.
#: Anything else -- futures, options, bonds, FX -- is outside v1, and checks that need its value are skipped.
SUPPORTED_KINDS = frozenset({"equity", "etf", "fund"})

#: Book events: cash moved (``cash``, signed, to the account) and shares moved (``quantity``, signed).
#: External ones -- money or securities in or out -- are flows, not profit.
EVENT_KINDS = frozenset({"dividend", "coupon", "interest", "fee", "borrow_fee", "financing", "deposit",
                         "withdrawal", "transfer_in", "transfer_out", "split", "quantity_adjustment", "other"})
EXTERNAL_EVENTS = frozenset({"deposit", "withdrawal", "transfer_in", "transfer_out"})


@dataclass(frozen=True)
class AccountingPolicy:
    """How strictly to judge.

    ``mode``: ``report`` never raises; ``strict`` makes ``raise_on_failure`` fatal for a failed required
    check; ``ci`` also treats an incomplete required scope as failure (``exit_code``). Money compares to
    ``max(abs_tolerance, rel_tolerance x scale)``; returns to ``return_tolerance`` per session.
    ``flow_timing`` places an external flow in the return of its session: ``end_of_day`` means it arrives
    after the session is valued.
    """

    mode: str = "report"
    abs_tolerance: float = 0.01
    rel_tolerance: float = 1e-10
    return_tolerance: float = 1e-9
    flow_timing: str = "end_of_day"
    max_failing_rows: int = 20

    def __post_init__(self):
        if self.mode not in ("report", "strict", "ci"):
            raise ValueError(f"unknown mode {self.mode!r}; report, strict or ci")
        if self.flow_timing != "end_of_day":
            raise ValueError("only end_of_day flow timing is supported in v1")

    def tolerance(self, scale: float = 0.0) -> float:
        return max(self.abs_tolerance, self.rel_tolerance * abs(scale))

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class OpeningState:
    """The book immediately before the first evaluated session: cash, and any positions with their prices.

    A backtest starts flat, with its capital in cash (``zipfolio.ziplime.opening_state``). ``cost_basis``
    is per share, needed to split realized from unrealized PnL on positions held from the start.
    """

    timestamp: pd.Timestamp
    cash: float
    currency: str = "USD"
    positions: dict = field(default_factory=dict)
    prices: dict = field(default_factory=dict)
    cost_basis: dict = field(default_factory=dict)

    @property
    def value(self) -> float | None:
        missing = [symbol for symbol, quantity in self.positions.items() if quantity and symbol not in self.prices]
        if missing:
            return None
        return self.cash + sum(quantity * self.prices[symbol] for symbol, quantity in self.positions.items())


@dataclass(frozen=True)
class ContractSpec:
    """What an instrument is: its kind, contract multiplier and currency."""

    kind: str = "equity"
    multiplier: float = 1.0
    currency: str | None = None

    @property
    def supported(self) -> bool:
        return self.kind in SUPPORTED_KINDS


@dataclass
class AccountingInputs:
    """The book of one backtest, as the engine recorded it.

    ``returns``: the engine's return per session (time-weighted, flows per ``AccountingPolicy.flow_timing``).
    ``position_records``: end-of-session snapshots, one row per open position -- ``date``, ``symbol``,
    ``quantity`` (signed), ``last_sale_price``. ``cash``: end-of-session cash. ``transactions``: the fills --
    ``id``, ``order_id``, ``dt``, ``symbol``, ``quantity`` (signed, buy > 0), ``price`` (the actual
    execution price, slippage inside). ``orders``: ``id``, ``dt``, ``symbol``, ``commission`` (cumulative
    per order, as ziplime records it; repeated rows of one order are fine). ``cost_settings``: the fee and
    slippage model the run used. ``opening_state``, ``portfolio_events`` (``dt``, ``kind``, ``symbol``,
    ``cash``, ``quantity``, ``ratio``) and ``contract_specs`` (symbol -> ``ContractSpec``) are what an
    independent ledger needs on top; without them the checks that need them are skipped, never assumed.
    ``fill_bars``: each fill's bar (``close``), indexed by fill id, to see the slippage in the prices.
    """

    returns: pd.Series
    position_records: pd.DataFrame | None = None
    cash: pd.Series | None = None
    transactions: pd.DataFrame | None = None
    orders: pd.DataFrame | None = None
    cost_settings: dict | None = None
    opening_state: OpeningState | None = None
    portfolio_events: pd.DataFrame | None = None
    contract_specs: dict | None = None
    sector_mappings: dict | None = None
    fill_bars: pd.DataFrame | None = None
    currency: str = "USD"
    run_id: str | None = None
    engine: str | None = None

    def digest(self) -> str:
        """A hash of every input, so a stored report can be matched to the book it judged."""
        digest = hashlib.sha256()
        for name in ("returns", "position_records", "cash", "transactions", "orders", "portfolio_events",
                     "fill_bars"):
            value = getattr(self, name)
            digest.update(name.encode())
            if value is None:
                digest.update(b"none")
                continue
            try:
                digest.update(pd.util.hash_pandas_object(value, index=True).to_numpy().tobytes())
            except TypeError:
                digest.update(value.astype(str).to_csv().encode())
        extras = {"cost_settings": self.cost_settings, "currency": self.currency,
                  "opening_state": None if self.opening_state is None else {
                      **asdict(self.opening_state), "timestamp": str(self.opening_state.timestamp)},
                  "contract_specs": {k: asdict(v) for k, v in sorted((self.contract_specs or {}).items())},
                  "sector_mappings": dict(sorted((self.sector_mappings or {}).items()))}
        digest.update(json.dumps(extras, sort_keys=True, default=str).encode())
        return digest.hexdigest()[:16]


#: How each unit is written in the JSON: money to the cent, returns to 1e-12, shares to 1e-6.
_FORMATS = {"money": "{:.2f}", "return": "{:.12f}", "shares": "{:.6f}", "count": "{:.0f}", "ratio": "{:.6f}"}


def _number(value, unit: str):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    value = float(value)
    if not math.isfinite(value):
        return None
    text = _FORMATS.get(unit, "{:.10g}").format(value)
    return text.lstrip("-") if float(text) == 0 else text


@dataclass
class CheckResult:
    """One check: what it tested, what it found, and what evidence it had."""

    id: str
    required: bool
    status: Status
    explanation: str
    severity: str = "info"
    reason_code: str | None = None
    unit: str = "money"
    observed: float | str | None = None
    expected: float | str | None = None
    difference: float | str | None = None
    tolerance: float | str | None = None
    sample_size: int | None = None
    first_failure_at: str | None = None
    affected_symbols: list = field(default_factory=list)
    inputs_used: list = field(default_factory=list)
    missing_inputs: list = field(default_factory=list)
    evidence_level: str = "independent"
    suggested_action: str | None = None
    details: list = field(default_factory=list)

    def to_dict(self, currency: str) -> dict:
        entry = {
            "id": self.id, "status": self.status.value, "required": self.required, "severity": self.severity,
            "reason_code": self.reason_code, "evidence_level": self.evidence_level, "unit": self.unit,
            "currency": currency if self.unit == "money" else None,
            "observed": _number(self.observed, self.unit), "expected": _number(self.expected, self.unit),
            "difference": _number(self.difference, self.unit), "tolerance": _number(self.tolerance, self.unit),
            "sample_size": self.sample_size, "first_failure_at": self.first_failure_at,
            "affected_symbols": sorted(map(str, self.affected_symbols)),
            "inputs_used": list(self.inputs_used), "missing_inputs": sorted(self.missing_inputs),
            "explanation": self.explanation, "suggested_action": self.suggested_action,
            "details": [{key: _plain(value) for key, value in row.items()} for row in self.details],
        }
        return entry


def _plain(value):
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, 10)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return _plain(value.item())
    return value


class IntegrityFailure(RuntimeError):
    """Raised by ``IntegrityReport.raise_on_failure`` when a required accounting check fails."""


@dataclass
class IntegrityReport:
    """Every check's result, the overall status and how much could be checked."""

    checks: list
    assurance_level: Assurance
    policy: AccountingPolicy
    currency: str
    input_hash: str
    zipfolio_version: str
    run_id: str | None = None
    engine: str | None = None

    @property
    def status(self) -> Overall:
        applicable = [check for check in self.checks if check.status != Status.NOT_APPLICABLE]
        if any(check.required and check.status == Status.FAIL for check in applicable):
            return Overall.FAIL
        if any(check.required and check.status == Status.SKIPPED for check in applicable):
            return Overall.INCOMPLETE
        if any(check.status in (Status.WARN, Status.FAIL) for check in applicable):
            return Overall.WARN
        return Overall.PASS

    @property
    def coverage(self) -> dict:
        counts = {status: sum(check.status == status for check in self.checks) for status in Status}
        applicable = len(self.checks) - counts[Status.NOT_APPLICABLE]
        checked = applicable - counts[Status.SKIPPED]
        return {"applicable": applicable, "passed": counts[Status.PASS], "failed": counts[Status.FAIL],
                "warned": counts[Status.WARN], "skipped": counts[Status.SKIPPED],
                "not_applicable": counts[Status.NOT_APPLICABLE],
                "coverage_ratio": f"{checked / applicable:.4f}" if applicable else "0.0000"}

    def check(self, check_id: str) -> CheckResult:
        return next(check for check in self.checks if check.id == check_id)

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "zipfolio_version": self.zipfolio_version,
            "engine": self.engine,
            "currency": self.currency,
            "status": self.status.value,
            "assurance_level": self.assurance_level.value,
            "coverage": self.coverage,
            "policy": {**asdict(self.policy), "hash": self.policy.digest()},
            "input_hash": self.input_hash,
            "checks": [check.to_dict(self.currency) for check in self.checks],
        }

    def to_json(self, path=None) -> str:
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
        if path is not None:
            from pathlib import Path

            Path(path).write_text(text + "\n", encoding="utf-8")
        return text

    def raise_on_failure(self) -> None:
        """Raise ``IntegrityFailure`` when the policy makes the outcome fatal: a failed required check under
        ``strict``; under ``ci``, also a required check that could not be made."""
        fatal = {Overall.FAIL} | ({Overall.INCOMPLETE} if self.policy.mode == "ci" else set())
        if self.policy.mode in ("strict", "ci") and self.status in fatal:
            failing = [check.id for check in self.checks if check.required
                       and check.status in (Status.FAIL, Status.SKIPPED)]
            raise IntegrityFailure(f"accounting integrity {self.status.value}: {', '.join(failing)}")

    def exit_code(self) -> int:
        """0 when the policy accepts the outcome, 1 when it does not: what a CI step should return."""
        if self.status == Overall.FAIL:
            return 1
        if self.status == Overall.INCOMPLETE and self.policy.mode == "ci":
            return 1
        return 0
