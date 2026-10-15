"""The accounting integrity panel: a report the check has already made (``zipfolio.integrity``), shown.

It sits above the summary and apart from the sixteen checks of the strategy: those judge the strategy,
this one whether the numbers judging it add up. It renders what it is given and computes nothing; the
JSON goes into the page beside it for an agent. It never calls a backtest verified for having rendered.
"""

from __future__ import annotations

import html

from ..integrity import Assurance, IntegrityReport, Overall, Status

#: The badge colour of each outcome, after the page's own verdict badges.
TONES = {Overall.PASS: "ok", Overall.WARN: "warn", Overall.FAIL: "bad", Overall.INCOMPLETE: "na",
         Status.PASS: "ok", Status.WARN: "warn", Status.FAIL: "bad", Status.SKIPPED: "na",
         Status.NOT_APPLICABLE: "na"}
#: The line under the status, by assurance: what was actually checked.
ASSURANCE_WORDS = {
    Assurance.LEDGER: "Every fill, fee and position replayed from the opening state",
    Assurance.RECONCILED: "Recomputed from the period snapshots; the book was not replayed from its start",
    Assurance.SUMMARY: "The published figures checked against each other only; no book replayed",
}
#: What each check is called on the page.
NAMES = {
    "INPUT.FILL_UNIQUENESS": "Fills", "INPUT.TIMESTAMP_ALIGNMENT": "Sessions",
    "NAV.SNAPSHOT_RECONCILIATION": "NAV from cash and positions", "RETURNS.NAV_RECONCILIATION": "Returns vs NAV",
    "CASH.LEDGER_REPLAY": "Cash replay", "POSITIONS.LEDGER_REPLAY": "Position replay",
    "PNL.REPORTED_SUMMARY_IDENTITY": "Published totals", "PNL.PORTFOLIO_RECONCILIATION": "Profit vs NAV",
    "PNL.REALIZED_UNREALIZED": "Realized / unrealized", "COST.COMMISSION_RECONCILIATION": "Commissions",
    "COST.SLIPPAGE_SINGLE_COUNT": "Slippage counted once", "ATTRIBUTION.INSTRUMENT_SUM": "Instruments add up",
    "ATTRIBUTION.SECTOR_SUM": "Sectors add up", "ROUND_TRIP.SYNTHETIC_CLOSURES": "Trades are real closes",
    "ROUND_TRIP.GROSS_NET_CONVENTION": "Gross vs net labels", "REPORT.DISPLAY_VS_CALCULATION": "Shown vs computed",
}


def panel_html(report: IntegrityReport | None, error: str | None = None) -> str:
    """The panel, and the report's JSON for an agent; a short note when the check could not run."""
    if report is None:
        if error is None:
            return ""
        return ('<section id="accounting-integrity" class="integrity"><div class="integrity-head">'
                '<span class="badge na">Accounting: not checked</span>'
                f'<span class="integrity-line">The integrity check could not run: {html.escape(error)}</span>'
                '</div></section>')
    status, coverage = report.status, report.coverage
    applicable = [check for check in report.checks if check.status != Status.NOT_APPLICABLE]
    line = (f"Assurance: {report.assurance_level.value} &middot; {coverage['passed']} of {coverage['applicable']} "
            f"checks hold" + (f" &middot; {coverage['failed']} failed" if coverage['failed'] else "")
            + (f" &middot; {coverage['warned']} warning{'s' if coverage['warned'] != 1 else ''}"
               if coverage['warned'] else "")
            + (f" &middot; {coverage['skipped']} not checked" if coverage['skipped'] else ""))
    notes = "".join(
        f'<li><span class="badge {TONES[check.status]}">{check.status.value.replace("_", " ")}</span> '
        f'<b>{html.escape(NAMES.get(check.id, check.id))}</b>: {html.escape(check.explanation)}'
        + (f' <span class="integrity-missing">Missing: {html.escape(", ".join(check.missing_inputs))}.</span>'
           if check.missing_inputs and check.status == Status.SKIPPED else "") + "</li>"
        for check in applicable if check.status != Status.PASS)
    rows = "".join(_row(check, report.currency) for check in report.checks)
    label = ("Accounting reconciles" if status == Overall.PASS and report.assurance_level == Assurance.LEDGER
             else f"Accounting: {status.value.lower()}")
    data = report.to_json().replace("</", "<\\/")
    return (
        f'<section id="accounting-integrity" class="integrity {TONES[status]}">'
        f'<div class="integrity-head"><span class="badge {TONES[status]}">{html.escape(label)}</span>'
        f'<span class="integrity-line">{line}</span></div>'
        f'<p class="integrity-scope">{html.escape(ASSURANCE_WORDS[report.assurance_level])}. It checks that the '
        "numbers add up to the book -- not the data, the fills' realism or the strategy.</p>"
        + (f'<ul class="integrity-notes">{notes}</ul>' if notes else "")
        + '<details class="integrity-evidence"><summary>View evidence</summary>'
        '<div class="table"><table><thead><tr><th>Check</th><th>Status</th><th>Observed</th><th>Expected</th>'
        f'<th>Difference</th><th class="integrity-why">What was checked</th></tr></thead><tbody>{rows}</tbody></table></div>'
        f'<p class="integrity-scope">{html.escape(report.policy.mode)} policy &middot; tolerance '
        f'{report.policy.abs_tolerance:g} {html.escape(report.currency)} or {report.policy.rel_tolerance:g} of the '
        f'amount &middot; inputs {report.input_hash} &middot; {html.escape("accounting-integrity/v1")}</p>'
        '</details></section>'
        f'<script type="application/json" id="zipfolio-integrity">{data}</script>')


def _row(check, currency: str) -> str:
    entry = check.to_dict(currency)

    def cell(key):
        value = entry[key]
        if value is None:
            return "&mdash;"
        suffix = f" {currency}" if entry["unit"] == "money" and key != "tolerance" else ""
        return html.escape(f"{value}{suffix}")

    return (f'<tr><td>{html.escape(NAMES.get(check.id, check.id))}<div class="integrity-id">{check.id}</div></td>'
            f'<td><span class="badge {TONES[check.status]}">{check.status.value.replace("_", " ")}</span></td>'
            f'<td class="n">{cell("observed")}</td><td class="n">{cell("expected")}</td>'
            f'<td class="n">{cell("difference")}</td><td class="integrity-why">{html.escape(check.explanation)}</td></tr>')
