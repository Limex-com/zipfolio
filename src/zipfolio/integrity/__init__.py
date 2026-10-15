"""Accounting integrity: whether a backtest's reported numbers reconcile to the book they came from.

``check_integrity(AccountingInputs(...))`` replays the book -- fills, fees, cash, positions, events -- and
checks every profit figure Zipfolio publishes against it, returning an ``IntegrityReport``: one result per
check (stable ids, ``checks.CHECK_IDS``), an overall status (``PASS``, ``WARN``, ``FAIL``, ``INCOMPLETE``)
and how much could be checked (``summary``, ``reconciled``, ``ledger``). It serialises to deterministic JSON
(``accounting-integrity/v1``) for an API, an agent or CI.

It certifies the arithmetic of the book, nothing more: not the market data, not the absence of look-ahead,
not the realism of the fills, not an edge. Missing evidence is reported as ``SKIPPED`` and leaves the
status ``INCOMPLETE`` -- never a pass. Nothing it is given is modified.
"""

from .checks import CHECK_IDS, REQUIRED, check_integrity
from .models import (
    SCHEMA_VERSION,
    AccountingInputs,
    AccountingPolicy,
    Assurance,
    CheckResult,
    ContractSpec,
    IntegrityFailure,
    IntegrityReport,
    OpeningState,
    Overall,
    Status,
)

__all__ = [
    "CHECK_IDS", "REQUIRED", "SCHEMA_VERSION", "AccountingInputs", "AccountingPolicy", "Assurance",
    "CheckResult", "ContractSpec", "IntegrityFailure", "IntegrityReport", "OpeningState", "Overall", "Status",
    "check_integrity",
]
