"""The page: sections of charts and tables in QuantStats Pro's layout."""

from __future__ import annotations

import html
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import pandas as pd

from . import __version__
from .interactive import RUNTIME, library
from .glossary import SECTIONS
from .tables import COPY_BUTTON, Table, table_html, term

#: Tables wider than this many columns go under the charts rather than beside them.
SIDE_TABLE_MAX_COLUMNS = 3

#: Where Zipfolio lives; every report links to it, in the header and the footer,
#: next to the version that built it.
REPOSITORY_URL = "https://github.com/Limex-com/zipfolio"

#: The report's title unless the caller names it.
DEFAULT_TITLE = "Zipfolio Tearsheet: Portfolio, Execution & Cost Analysis"

#: How every full date on the page is written: 11 Oct 2023.
DATE_FORMAT = "%-d %b %Y"


#: How a finding reads: fine, worth a look, wrong, just a fact, or a check that does not apply.
OK, WARN, BAD, INFO, NA = "ok", "warn", "bad", "info", "na"
MARKS = {OK: "&#10003;", WARN: "!", BAD: "&#10007;", INFO: "i"}
#: The summary lists what needs attention first.
SEVERITY = {BAD: 0, WARN: 1, INFO: 2, OK: 3, NA: 4}

#: The page's four parts, each the question a reader brings to it, and the
#: sections that answer it, in page order. A section missing here goes last.
GROUPS: list[tuple[str, list[str]]] = [
    ("Where the Return Came From", [
        "Performance", "Profit Concentration", "Instruments", "Long and Short", "Sector Analysis",
        "Factor Analysis"]),
    ("What It Risked", ["Drawdowns", "Market Regimes", "Portfolio Risk", "CAPM and the Efficient Frontier",
                        "Risk Over Time"]),
    ("How It Holds and Trades", ["Positions", "Transactions", "Closed Trades"]),
    ("Can It Be Traded", ["Next-Bar Fills", "Execution", "Trading Costs", "Capacity"]),
]


#: Checks that measure one fact several ways. A strategy on one stock fails every
#: concentration check at once, and a result made in a few sessions fails the best-days
#: and the best-trades check together; counted row by row that is one problem scored
#: several times. A family is one row of the scorecard: its verdict is the worst of its
#: members', and the members open beneath it.
FAMILIES = {
    "edge": "More than luck",
    "few_events": "Rests on a few days or trades",
    "concentration": "Concentration in one name",
}

#: The scorecard: four rows per part, sixteen in all, the same for every backtest. A row is a
#: check, ``(key, label shown when it could not be made)``, or a family, ``(family key, [checks])``.
#: Sections fill the slots by key (``Section.finding``).
SCORECARD: list[tuple[str, list]] = [
    ("Where the Return Came From", [
        ("return", "Annual return vs the benchmark"),
        ("buy_hold", "Against holding the same names"),
        ("edge", [("significance", "Sharpe ratio beyond luck"),
                  ("luck", "Luck test: chance of a loss"),
                  ("alpha", "Alpha after factors")]),
        ("few_events", [("best_days", "Total return without the best 5 days"),
                        ("best_trades", "Profit from the best 5 trades")]),
    ]),
    ("What It Risked", [
        ("max_dd", "Max drawdown"),
        ("capture", "Up / down capture"),
        ("underwater", "Longest time below a peak"),
        ("concentration", [("largest", "Largest position, typical day"),
                           ("top_names", "Profit from the top 2 names"),
                           ("risk_conc", "Risk from the top name, typical day"),
                           ("sector", "Largest sector, typical day")]),
    ]),
    ("How It Holds and Trades", [
        ("win_pf", "Winning trades · profit factor"),
        ("sample", "Closed trades: enough to judge"),
        ("open_profit", "Profit still open at the end"),
        ("leverage", "Max gross exposure"),
    ]),
    ("Can It Be Traded", [
        ("timing", "Fill timing"),
        ("realism", "Fill realism"),
        ("costs", "Costs' share of profit before costs"),
        ("liquidity", "Liquidity"),
    ]),
]

#: Which issues the verdict names first, among those of one severity: what most often decides
#: whether a backtest is worth anything, before what only qualifies it.
VERDICT_ORDER = ("few_events", "edge", "concentration", "buy_hold", "sample", "timing", "realism",
                 "costs", "liquidity", "win_pf", "max_dd", "underwater", "capture", "return",
                 "open_profit", "leverage")

#: The measure behind a check whose label is a question, for the copy of the report an assistant reads:
#: "Closed trades: enough to judge" is a verdict waiting to happen, "Closed trades" a number.
METRIC_NAMES = {
    "significance": "Sharpe ratio significance",
    "luck": "Luck test: share of draws with a loss",
    "sample": "Closed trades",
    # Its value can be "all checks passed": the Execution section's measures stand in for it.
    "realism": "",
}

#: Facts the summary shows above the scorecard, in this order: not verdicts, but what a
#: reader about to run the strategy wants to know first.
GLANCE = ("exposure", "style", "now", "breakeven", "capacity")
#: Facts that are kept out of the summary's context line: they belong to the glance row or the JSON.
QUIET = ("one_name", "cash", "turnover")


@dataclass
class Finding:
    """One row of the summary: a measure, its value, and how much it matters.

    ``label`` and ``value`` are what the row shows -- "Without best 5 days",
    "30.9% → 5.2%"; ``note`` is the sentence behind it and ``threshold`` the line
    it is judged against, both shown when the row is opened. ``issue`` is the
    finding as a clause of the summary's verdict, when it does not hold up, and
    ``advice`` what to try next.
    """

    status: str
    label: str
    value: str
    note: str = ""
    section: str = ""
    key: str = ""
    threshold: str = ""
    advice: str = ""
    issue: str = ""
    #: Another section that says more about it, linked beside the numbers.
    see: str = ""


@dataclass
class Section:
    """One block of the page: charts on the left, narrow tables on the right, wide ones under both.

    ``findings`` are what the section has to say in a sentence; the summary at
    the top of the page collects them, each linked back here.
    """

    title: str
    charts: list[str] = field(default_factory=list)
    side: list[str] = field(default_factory=list)
    wide: list[str] = field(default_factory=list)
    failed: bool = False
    findings: list[Finding] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.charts or self.side or self.wide)

    @property
    def anchor(self) -> str:
        return anchor(self.title)

    def finding(self, status: str, label: str, value: str, note: str = "", key: str = "",
                threshold: str = "", advice: str = "", issue: str = "") -> None:
        """Record a finding; ``key`` puts it in its slot of the scorecard (``SCORECARD``) or the glance row."""
        self.findings.append(Finding(status, label, value, note, self.title, key, threshold, advice, issue))

    def chart(self, svg: str | None) -> None:
        if svg:
            self.charts.append(svg)

    def table(self, table: "Table", wide: bool | None = None, **kwargs) -> None:
        """Add a table: beside the charts when narrow, under them when wide -- or where ``wide`` says."""
        rendered = table_html(table, **kwargs)
        if wide is None:
            wide = len(table.frame.columns) > SIDE_TABLE_MAX_COLUMNS
        (self.wide if wide else self.side).append(rendered)


def anchor(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def ordered(sections: list[Section]) -> list[tuple[str | None, list[Section]]]:
    """The sections under their groups, in page order; groups with nothing in them are left out."""
    by_title = {section.title: section for section in sections}
    placed: set[str] = set()
    groups = []
    for name, titles in GROUPS:
        members = [by_title[title] for title in titles if title in by_title]
        placed.update(section.title for section in members)
        if members:
            groups.append((name, members))
    rest = [section for section in sections if section.title not in placed]
    if rest:
        groups.append((None, rest))
    return groups


def _numeral(number: int) -> str:
    return ["I", "II", "III", "IV", "V", "VI"][number] if number < 6 else str(number + 1)


def _jump_bar_html(groups, has_summary: bool) -> str:
    links = ['<a href="#summary">Summary</a>'] if has_summary else []
    links += [f'<a href="#part-{number + 1}"><span class="num">{_numeral(number)}</span> '
              f'{html.escape(name or "More")}</a>' for number, (name, _) in enumerate(groups)]
    links.append('<button type="button" class="expand" data-expand>Collapse all</button>')
    return f'<nav class="jump">{"".join(links)}</nav>'


#: The verdict badges, after QuantStats Pro's health summary, and the tiles that count them.
BADGES = {OK: "Holds up", WARN: "Needs a look", BAD: "Critical", NA: "N/A"}
TILE_NOTES = {OK: "within a safe range", WARN: "check before trusting the result", BAD: "likely to mislead"}


@dataclass
class Row:
    """One row of the scorecard: a check, or a family of checks, with the verdict it counts as."""

    part: str
    key: str
    label: str
    finding: Finding | None = None
    members: list["Row"] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.members:
            judged = [member.status for member in self.members if member.status != NA]
            return min(judged, key=SEVERITY.get) if judged else NA
        if self.finding is None or self.finding.status not in (OK, WARN, BAD):
            return NA
        return self.finding.status

    @property
    def worst(self) -> "Row":
        """The row whose finding speaks for this one: itself, or its family's worst member."""
        if not self.members:
            return self
        return min(self.members, key=lambda member: SEVERITY[member.status])


def scorecard(groups) -> tuple[list[Row], list[Finding], list[Finding]]:
    """The scorecard's rows, the glance facts and the context line, from the sections' findings.

    A slot's first finding fills it; a later one for the same slot, and every
    finding outside the scorecard, is context -- unless its key is a glance fact.
    """
    findings = [finding for _, members in groups for section in members if not section.failed
                for finding in section.findings]
    order = {title: rank for rank, title in enumerate(title for _, titles in GROUPS for title in titles)}
    findings.sort(key=lambda f: order.get(f.section, 99))
    keys = set()
    for _, checks in SCORECARD:
        for key, label in checks:
            if isinstance(label, list):
                keys.update(member for member, _ in label)
            else:
                keys.add(key)
    slots: dict[str, Finding] = {}
    glance: dict[str, Finding] = {}
    context = []
    for finding in findings:
        if finding.key in GLANCE:
            glance.setdefault(finding.key, finding)
        elif finding.key in keys and finding.key not in slots:
            slots[finding.key] = finding
        elif finding.key in QUIET:
            continue
        elif finding.key not in keys or finding.status == INFO:
            context.append(finding)

    rows = []
    for part, checks in SCORECARD:
        for key, label in checks:
            if isinstance(label, list):
                members = [Row(part, member, fallback, slots.get(member)) for member, fallback in label]
                rows.append(Row(part, key, FAMILIES[key], members=members))
            else:
                rows.append(Row(part, key, label, slots.get(key)))
    return rows, [glance[key] for key in GLANCE if key in glance], context


def verdict_text(rows: list[Row]) -> str:
    """The scorecard in a sentence: the worst issues, at most three."""
    rank = {key: position for position, key in enumerate(VERDICT_ORDER)}
    flagged = sorted((row for row in rows if row.status in (BAD, WARN)),
                     key=lambda row: (SEVERITY[row.status], rank.get(row.key, len(rank))))
    if not flagged:
        judged = [row for row in rows if row.status != NA]
        return ("Every check that could be made holds up." if judged else
                "Too little in this backtest to judge it.")
    issues = []
    for row in flagged:
        finding = row.worst.finding
        issue = finding.issue or f"{finding.label[0].lower()}{finding.label[1:]}: {finding.value}"
        if issue not in issues:
            issues.append(issue)
    return "Main issues: " + "; ".join(issues[:3]) + "."


def _details(finding: Finding, threshold: bool = True, advice: bool = True) -> str:
    """What a row says when opened: the reasoning, the threshold, what to do, and where the numbers are."""
    parts = []
    if finding.note:
        parts.append(f"<p>{html.escape(finding.note)}</p>")
    if threshold and finding.threshold:
        parts.append(f"<p><b>Threshold:</b> {html.escape(finding.threshold)}</p>")
    if advice and finding.advice and finding.status in (WARN, BAD):
        parts.append(f"<p><b>What to try:</b> {html.escape(finding.advice)}</p>")
    parts.append(f'<p><a href="#{anchor(finding.section)}">The numbers: {html.escape(finding.section)} &rarr;</a>'
                 + (f' &middot; <a href="#{anchor(finding.see)}">{html.escape(finding.see)} &rarr;</a>'
                    if finding.see else "") + '</p>')
    return "".join(parts)


def _badge(status: str) -> str:
    return f'<span class="badge {status}">{BADGES[status]}</span>'


def _member_html(member: Row) -> str:
    """One check inside an opened family: its verdict, value and reasoning."""
    found = member.finding
    if found is None or member.status == NA:
        note = found.note if found is not None and found.note else "Not available for this backtest."
        label = found.label if found is not None else member.label
        return (f'<li>{_badge(NA)} <b>{html.escape(label)}</b>'
                f'<div class="why"><p>{html.escape(note)}</p></div></li>')
    return (f'<li>{_badge(member.status)} <b>{html.escape(found.label)}</b>: {html.escape(found.value)}'
            f'<div class="why">{_details(found, advice=False)}</div></li>')


def _row_html(row: Row) -> str:
    status = row.status
    if row.members:
        worst = row.worst.finding
        value = (f'{html.escape(worst.value)}<div class="sub">{html.escape(worst.label)}</div>'
                 if status != NA else "&mdash;")
        advice = (f'<p class="why"><b>What to try:</b> {html.escape(worst.advice)}</p>'
                  if worst is not None and worst.advice and status in (WARN, BAD) else "")
        return (f'<tr class="family {status}"><td><details><summary>{html.escape(row.label)}'
                f'<span class="count">{len(row.members)}</span></summary>'
                f'<ul class="members">{"".join(_member_html(member) for member in row.members)}</ul>{advice}'
                f'</details></td><td class="v">{value}</td><td class="b">{_badge(status)}</td></tr>')
    found = row.finding
    if status == NA:
        note = found.note if found is not None and found.note else "Not available for this backtest."
        label = found.label if found is not None else row.label
        value = html.escape(found.value) if found is not None and found.value else "&mdash;"
        return (f'<tr class="na"><td><details><summary>{html.escape(label)}</summary>'
                f'<div class="why"><p>{html.escape(note)}</p></div></details></td>'
                f'<td class="v">{value}</td><td class="b">{_badge(NA)}</td></tr>')
    return (f'<tr class="{status}"><td><details><summary>{html.escape(found.label)}</summary>'
            f'<div class="why">{_details(found)}</div></details></td>'
            f'<td class="v">{html.escape(found.value)}</td><td class="b">{_badge(status)}</td></tr>')


def summary_html(groups) -> str:
    """The summary: a verdict, the facts at a glance, the score, the checks, one line of context.

    The score counts what could be judged: a check that does not apply to this
    backtest -- the top two names of a strategy that traded one -- is N/A and
    out of the count, and a family of checks that measure one fact counts once.
    """
    rows, glance, context = scorecard(groups)
    if all(row.status == NA and not (row.finding and row.finding.note) for row in rows) and not glance \
            and not context:
        return ""
    part_of = {name: number for number, (name, _) in enumerate(groups)}

    body = []
    for number, (part, _) in enumerate(SCORECARD):
        link = f'#part-{part_of[part] + 1}' if part in part_of else "#summary"
        body.append(f'<tr class="grp"><td colspan="3"><a href="{link}">{_numeral(number)} &middot; '
                    f'{html.escape(part)}</a></td></tr>')
        body.extend(_row_html(row) for row in rows if row.part == part)

    statuses = [row.status for row in rows]
    counted = [status for status in statuses if status != NA]
    counts = {status: statuses.count(status) for status in (OK, WARN, BAD, NA)}
    share = counts[OK] / len(counted) if counted else 0.0
    tone = OK if share >= 0.75 else WARN if share >= 0.5 else BAD
    missing = counts[NA]
    tiles = [f'<div class="tile {tone}"><div class="k">Score</div><div class="n">{counts[OK]}/{len(counted)}</div>'
             + (f'<div class="s">{missing} not applicable</div>' if missing else "") + '</div>']
    tiles += [f'<div class="tile {status}"><div class="k">{BADGES[status]}</div><div class="n">{counts[status]}</div>'
              '</div>' for status in (BAD, WARN, OK)]
    facts = "".join(
        f'<div class="fact"><div class="k">' + (
            f'<span class="term" tabindex="0" data-tip="{html.escape(f.note)}">{html.escape(f.label)}</span>'
            if f.note else html.escape(f.label))
        + f'</div><div class="fv {f.status}">{html.escape(f.value)}</div></div>' for f in glance)
    return ('<section id="summary" class="summary"><h2>Summary</h2>'
            f'<p class="verdict">{html.escape(verdict_text(rows))}</p>'
            + (f'<div class="glance">{facts}</div>' if facts else "")
            + f'<div class="tiles">{"".join(tiles)}</div>'
            f'<div class="tbl-head">{COPY_BUTTON}</div><table class="checks"><thead><tr><th>Check <span class="hint">tap a row for details</span></th>'
            f'<th>Value</th><th>Verdict</th></tr></thead><tbody>{"".join(body)}</tbody></table>'
            '</section>')


def scorecard_json(groups, run: dict[str, str], returns: pd.Series) -> str:
    """The scorecard as data, for an agent reading the page: every check with its value, threshold and verdict."""
    rows, glance, context = scorecard(groups)
    counted = [row for row in rows if row.status != NA]

    def check(row: Row) -> dict:
        found = row.finding
        entry = {"key": row.key, "part": row.part, "status": row.status,
                 "label": found.label if found else row.label}
        if row.key in METRIC_NAMES:
            entry["metric"] = METRIC_NAMES[row.key]
        if found is not None:
            entry.update({name: value for name, value in asdict(found).items()
                          if name in ("value", "threshold", "note", "advice", "section") and value})
        return entry

    payload = {
        "generator": f"Zipfolio {__version__}",
        "period": [f"{returns.index[0]:%Y-%m-%d}", f"{returns.index[-1]:%Y-%m-%d}"],
        "sessions": len(returns),
        "run": run,
        "verdict": verdict_text(rows),
        "score": {"holds_up": sum(row.status == OK for row in counted), "counted": len(counted),
                  "needs_a_look": sum(row.status == WARN for row in counted),
                  "critical": sum(row.status == BAD for row in counted),
                  "not_applicable": len(rows) - len(counted)},
        "checks": [
            {**check(row), "family": [check(member) for member in row.members]} if row.members else check(row)
            for row in rows],
        "facts": [{"key": f.key, "label": f.label, "value": f.value, "note": f.note} for f in glance],
        "context": [{"label": f.label, "value": f.value, "status": f.status} for f in context],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=1)
    return text.replace("</", "<\\/")


def part_status(groups, part: str) -> str:
    """The part header's verdict: how many of its checks are critical or need a look."""
    rows, _, _ = scorecard(groups)
    statuses = [row.status for row in rows if row.part == part]
    bad, warn = statuses.count(BAD), statuses.count(WARN)
    if not bad and not warn:
        return f'<span class="badge ok">{"All hold up" if statuses.count(OK) else "Nothing to judge"}</span>'
    parts = []
    if bad:
        parts.append(f'<span class="badge bad">{bad} critical</span>')
    if warn:
        parts.append(f'<span class="badge warn">{warn} need{"s" if warn == 1 else ""} a look</span>')
    return " ".join(parts)


#: Paragraphs that explain how to read a section -- with the heading over them, if they have
#: one of their own -- rather than report what the backtest did.
_EXPLANATION = re.compile(r'(?:<h3>[^<]*</h3>\s*)?(?:<(p|ul) class="(?:explain|caption)">.*?</\1>\s*)+', re.S)


def explanations(side: str) -> tuple[str, str]:
    """``side`` split into what stays in view and the explanations, which go under a toggle."""
    moved = [match.group(0) for match in _EXPLANATION.finditer(side)]
    return _EXPLANATION.sub("", side), "".join(moved)


def _section_html(section: Section) -> str:
    heading = term(section.title, SECTIONS[section.title]) if section.title in SECTIONS else html.escape(section.title)
    parts = [f'<section id="{section.anchor}"><h2>{heading}</h2>']
    if section.failed:
        parts.append('<p class="note">This section could not be computed for this backtest.</p>')
    else:
        charts = "\n".join(f'<div class="chart">{svg}</div>' for svg in section.charts)
        side, notes = explanations("\n".join(section.side))
        if notes:
            side = f'<details class="howto"><summary>How to read this section</summary>{notes}</details>' + side
        parts.append(f'<div class="row"><div class="left">{charts}</div><div class="right">{side}</div></div>')
        if section.wide:
            parts.append(f'<div class="wide">{"".join(section.wide)}</div>')
    parts.append("</section>")
    return "\n".join(parts)


#: Opens the part a link points into -- a reader may have closed it -- and every part before printing; shows
#: the way back to the summary once the reader has scrolled past it.
_SCRIPT = """
(function () {
  function reveal(id) {
    var target = document.getElementById(id);
    if (!target) return false;
    for (var node = target; node; node = node.parentElement) {
      if (node.tagName === "DETAILS") node.open = true;
    }
    target.scrollIntoView();
    return true;
  }
  document.addEventListener("click", function (event) {
    var expand = event.target.closest("[data-expand]");
    if (expand) {
      var parts = document.querySelectorAll("details.part");
      var open = Array.prototype.some.call(parts, function (part) { return !part.open; });
      parts.forEach(function (part) { part.open = open; });
      expand.textContent = open ? "Collapse all" : "Expand all";
      return;
    }
    var link = event.target.closest('a[href^="#"]');
    if (!link) return;
    var id = decodeURIComponent(link.getAttribute("href").slice(1));
    if (id && reveal(id)) {
      event.preventDefault();
      history.replaceState(null, "", "#" + id);
    }
  });
  if (location.hash) reveal(decodeURIComponent(location.hash.slice(1)));
  var back = document.querySelector(".totop"), summary = document.getElementById("summary");
  if (back) {
    var place = function () {
      back.classList.toggle("shown", summary ? summary.getBoundingClientRect().bottom < 0 : window.scrollY > 800);
    };
    window.addEventListener("scroll", place, {passive: true});
    place();
  }
  window.addEventListener("beforeprint", function () {
    document.querySelectorAll("details").forEach(function (node) { node.open = true; });
  });
})();
"""

#: The tools on the page: copying a table for a spreadsheet, saving or copying a chart as an
#: image, and the hints behind the terms. Every chart -- drawn by ECharts or a static SVG -- is
#: exported from its SVG, so what is saved is what is on screen, legend toggles and zoom included.
_TOOLS = r"""
(function () {
  var SAVE_ICON = '<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><path d="M8 1.5v8.5M4.5 6.8 8 10.2l3.5-3.4M2.5 12v2h11v-2" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  var ENLARGE_ICON = '<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><path d="M2.5 6V2.5H6M10 2.5h3.5V6M13.5 10v3.5H10M6 13.5H2.5V10"/></svg>';
  var canCopyImage = !!(window.ClipboardItem && navigator.clipboard && navigator.clipboard.write && window.isSecureContext);

  function flash(button, text, ok) {
    if (!button.querySelector(".label") && button.querySelector("svg")) {
      // An icon: the check for a moment, and the words in the hint.
      button.classList.toggle("done", ok !== false);
      button.classList.toggle("failed", ok === false);
      showTip(button, text);
      clearTimeout(button.zfTimer);
      button.zfTimer = setTimeout(function () { button.classList.remove("done", "failed"); }, 1600);
      return;
    }
    var label = button.querySelector(".label") || button;
    if (button.zfLabel == null) button.zfLabel = label.textContent;
    label.textContent = text;
    button.classList.toggle("done", ok !== false);
    button.classList.toggle("failed", ok === false);
    clearTimeout(button.zfTimer);
    button.zfTimer = setTimeout(function () {
      label.textContent = button.zfLabel; button.classList.remove("done", "failed");
    }, 1600);
  }
  function escape(text) { return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }

  // -- Tables: tab-separated text and an HTML table, so a spreadsheet gets cells, not one line.
  function cellText(cell) {
    var copy = cell.cloneNode(true);
    copy.querySelectorAll(".why, .members, .count, .hint, details > :not(summary)").forEach(function (node) { node.remove(); });
    copy.querySelectorAll(".sub").forEach(function (node) { node.textContent = " (" + node.textContent.trim() + ")"; });
    return copy.textContent.replace(/\s+/g, " ").trim();
  }
  var NUMBER = /^([+\-]?)([$\u20ac\u20bd]?)(\d[\d,]*(?:\.\d+)?|\.\d+)(%?)(?:\s?[\u20bd\u20ac])?$/;
  function number(text) {
    // A number as Excel should store it -- x:num, in any regional setting -- and how to show it.
    var match = text.replace(/\u2212/g, "-").match(NUMBER);
    if (!match) return null;
    var digits = match[3], value = parseFloat(digits.replace(/,/g, "")) * (match[1] === "-" ? -1 : 1);
    if (!isFinite(value)) return null;
    var decimals = (digits.split(".")[1] || "").length;
    var format = (digits.indexOf(",") >= 0 ? "#,##0" : "0") + (decimals ? "." + new Array(decimals + 1).join("0") : "");
    if (match[4]) { value = value / 100; format += "%"; }
    return {value: +value.toPrecision(12), format: format};
  }
  function tableRows(table) {
    var rows = [];
    table.querySelectorAll("tr").forEach(function (tr) {
      if (tr.classList.contains("sep")) return;
      var head = tr.parentNode.tagName === "THEAD" || tr.classList.contains("grp");
      rows.push(Array.prototype.map.call(tr.children, function (cell) {
        return {text: cellText(cell), head: head || cell.tagName === "TH", span: +cell.getAttribute("colspan") || 1};
      }));
    });
    return rows;
  }
  function copyTable(table, button) {
    var rows = tableRows(table);
    var plain = rows.map(function (cells) {
      return cells.map(function (c) { return c.text.replace(/\t/g, " ") + new Array(c.span).join("\t"); }).join("\t");
    }).join("\n");
    var body = rows.map(function (cells) {
      return "<tr>" + cells.map(function (c) {
        var tag = c.head ? "th" : "td", n = c.head ? null : number(c.text);
        var style = c.head ? "background:#2d2d2d;color:#ffffff;font-weight:bold;" : "";
        var extra = (c.span > 1 ? ' colspan="' + c.span + '"' : "") + (n ? ' x:num="' + n.value + '"' : "");
        if (n) style += "mso-number-format:'" + n.format + "';text-align:right;";
        return "<" + tag + extra + (style ? ' style="' + style + '"' : "") + ">" + escape(c.text) + "</" + tag + ">";
      }).join("") + "</tr>";
    }).join("");
    var markup = '<table style="border-collapse:collapse;font-family:Arial;font-size:10pt">' + body + "</table>";
    write(markup, plain).then(function () { flash(button, "Copied: paste it into Excel or Google Sheets"); },
                              function () { flash(button, "Could not copy: the browser refused", false); });
  }
  function write(markup, plain) {
    // The copy event first: it works in every browser and keeps both formats; the async
    // clipboard where the event is refused (iOS without a selection).
    var done = false;
    function fill(event) {
      if (markup) event.clipboardData.setData("text/html", markup);
      event.clipboardData.setData("text/plain", plain);
      event.preventDefault();
      done = true;
    }
    document.addEventListener("copy", fill);
    try { document.execCommand("copy"); } catch (error) { done = false; }
    document.removeEventListener("copy", fill);
    if (done) return Promise.resolve();
    if (markup && window.ClipboardItem && navigator.clipboard && navigator.clipboard.write) {
      return navigator.clipboard.write([new ClipboardItem({
        "text/html": new Blob([markup], {type: "text/html"}),
        "text/plain": new Blob([plain], {type: "text/plain"})})]);
    }
    return navigator.clipboard ? navigator.clipboard.writeText(plain) : Promise.reject();
  }

  // -- The report as numbers, for the reader's own assistant: the run, every metric, every table, the
  //    titles of the charts. None of the report's opinions -- no verdicts, scores, reasoning or thresholds --
  //    so the assistant reads the strategy rather than Zipfolio's view of it; only definitions of the
  //    measures it may not know.
  var DEFINITIONS = [
    "Returns are daily; annual figures count 252 sessions a year. An annual return is compounded unless it says mean.",
    "Sharpe ratio: the mean daily return over its standard deviation, times the square root of 252; at a 0% rate unless a risk-free rate is named.",
    "PSR (probabilistic Sharpe ratio): the probability that the true Sharpe ratio is above zero, given the number of sessions and the skew and kurtosis of daily returns (Bailey and Lopez de Prado).",
    "Closed trade: one per closing fill, matched to the opening fills first in, first out (pyfolio's round trips); a position still held at the end is not a trade. Trade PnL is before commission; trade return is its PnL over its cost.",
    "Realized PnL: the closed trades' PnL. Unrealized PnL: the shares still held, at the last price, over what they cost. Net profit = realized + unrealized - commission + other cash = the change in cash plus positions. Slippage is inside the fill prices, so inside both PnL figures; the reconstructed slippage is not subtracted again.",
    "Net PnL of a name: realized plus unrealized, after commission. Contribution of a name: its daily PnL after commission over the portfolio's value at the start of each day, summed; the names' contributions add up to the sum of daily returns, not to the compounded total return.",
    "Luck test: the closed trades' PnL drawn again at random with replacement, as many times as the run's settings say; the share of draws whose total is a loss, and the percentiles of the totals.",
    "Gross exposure: longs plus shorts over the portfolio's value. Effective names: one over the sum of the squared weights of the book.",
    "Up and down capture: the strategy's geometric mean monthly return over the benchmark's, in the months the benchmark rose, and in those it fell.",
    "Factor betas and alpha: least squares of the daily return over the risk-free rate on the factors, with Newey-West standard errors (5 lags); alpha times 252.",
    "CAPM and efficient frontier: means of daily returns times 252 and daily volatility times the square root of 252. The frontier is long only and fully invested, computed from the names' returns over the whole backtest.",
    "Next-bar fills: every fill priced again at the instrument's next bar open instead of the signal bar's close, the difference added to the book.",
    "Profit gone at: the extra cost on every fill, in basis points of its value, at which the net profit would be zero.",
    "Capacity: the starting capital at which the largest fill would be 1% of its day's volume."
  ];
  function markdownTable(table) {
    var rows = tableRows(table);
    if (!rows.length) return "";
    var width = Math.max.apply(null, rows.map(function (cells) {
      return cells.reduce(function (sum, c) { return sum + c.span; }, 0); }));
    var lines = rows.map(function (cells) {
      var out = [];
      cells.forEach(function (c) { out.push(c.text.replace(/\|/g, "/")); for (var i = 1; i < c.span; i++) out.push(""); });
      while (out.length < width) out.push("");
      return "| " + out.join(" | ") + " |";
    });
    lines.splice(1, 0, "|" + new Array(width + 1).join(" --- |"));
    return lines.join("\n");
  }
  function clean(node) { return node ? node.textContent.replace(/\s+/g, " ").trim() : ""; }
  function reportText() {
    var out = ["# " + clean(document.querySelector("h1")), clean(document.querySelector(".period"))];
    document.querySelectorAll("dl.run dt").forEach(function (dt) { out.push("- " + clean(dt) + ": " + clean(dt.nextElementSibling)); });
    var generator = (document.querySelector('meta[name="generator"]') || {}).content || "Zipfolio";
    out.push("", "The numbers of a backtest report made by " + generator + ": the run's settings, its metrics and " +
             "every table. The report's own assessments are left out, and so are the charts; their titles are listed.");
    var integrity = document.getElementById("zipfolio-integrity");
    if (integrity) {
      var check = JSON.parse(integrity.textContent);
      out.push("", "Accounting integrity (" + check.schema_version + "): " + check.status + ", assurance " +
               check.assurance_level + "; " + check.coverage.passed + " of " + check.coverage.applicable +
               " checks hold, " + check.coverage.failed + " failed, " + check.coverage.skipped + " not checked.");
    }
    var card = document.getElementById("zipfolio-scorecard");
    if (card) {
      var data = JSON.parse(card.textContent), part = null;
      out.push("", "## Metrics");
      data.facts.forEach(function (f) { out.push("- " + f.label + ": " + f.value); });
      function metric(c) {
        if (c.family && c.family.length) { c.family.forEach(metric); return; }
        if (c.value && c.metric !== "") out.push("- " + (c.metric || c.label) + ": " + c.value);
      }
      data.checks.forEach(function (c) {
        if (c.part !== part) { part = c.part; out.push("", "### " + part); }
        metric(c);
      });
      if (data.context.length) {
        out.push("", "### Other");
        data.context.forEach(function (f) { out.push("- " + f.label + ": " + f.value); });
      }
    }
    document.querySelectorAll("details.part").forEach(function (partNode) {
      out.push("", "## " + clean(partNode.querySelector(".part-name")));
      partNode.querySelectorAll("section").forEach(function (section) {
        var heading = section.querySelector("h2"), about = heading.querySelector("[data-tip]");
        out.push("", "### " + clean(heading));
        if (about) out.push(about.getAttribute("data-tip"));
        var titles = Array.prototype.map.call(section.querySelectorAll(".ichart[aria-label], .chart > svg > title"),
          function (node) { return node.getAttribute("aria-label") || clean(node); });
        if (titles.length) out.push("Charts on the page: " + titles.join("; ") + ".");
        section.querySelectorAll(".tbl-head, .check, p.note").forEach(function (node) {
          if (node.classList.contains("tbl-head")) {
            var next = node.nextElementSibling, table = next && next.querySelector("table");
            if (table) out.push("", "#### " + clean(node.querySelector("h3")), markdownTable(table));
          } else if (node.classList.contains("check")) {
            // The bare measure, not the line's reasoning; a line without one is an opinion and stays out.
            if (node.hasAttribute("data-metric"))
              out.push("- " + node.getAttribute("data-metric") + ": " + node.getAttribute("data-value"));
          } else {
            out.push(clean(node));
          }
        });
      });
    });
    out.push("", "## Definitions");
    DEFINITIONS.forEach(function (line) { out.push("- " + line); });
    return out.join("\n").replace(/\n{3,}/g, "\n\n") + "\n";
  }
  function reportName() {
    var period = clean(document.querySelector(".period")).split(/\s/).slice(0, 3).join("-").toLowerCase();
    return "zipfolio-report-" + period.replace(/[^a-z0-9-]+/g, "") + ".md";
  }

  // -- Charts: the menu, and the SVG every format is made from.
  function chartSvg(holder) {
    var box = holder.querySelector(".ichart");
    if (box) {
      var chart = box.zfChart || (window.zipfolioDraw && window.zipfolioDraw(box));
      if (!chart) return null;
      var toolbox = (chart.getOption().toolbox || [])[0], shown = toolbox && toolbox.show !== false;
      if (shown) chart.setOption({toolbox: {show: false}});
      var text = chart.renderToSVGString();
      if (shown) chart.setOption({toolbox: {show: true}});
      return {text: text, width: box.offsetWidth, height: box.offsetHeight};
    }
    var svg = holder.querySelector("svg");
    if (!svg) return null;
    var copy = svg.cloneNode(true), size = svg.getBoundingClientRect();
    if (!copy.getAttribute("xmlns")) copy.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    return {text: new XMLSerializer().serializeToString(copy), width: size.width, height: size.height};
  }
  function raster(svg, type) {
    return new Promise(function (resolve, reject) {
      var image = new Image();
      image.onload = function () {
        var scale = 2, canvas = document.createElement("canvas");
        canvas.width = Math.round(svg.width * scale);
        canvas.height = Math.round(svg.height * scale);
        var context = canvas.getContext("2d");
        context.fillStyle = "#ffffff";
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.drawImage(image, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(function (blob) { if (blob) resolve(blob); else reject(); }, type, 0.92);
      };
      image.onerror = reject;
      image.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg.text);
    });
  }
  function download(blob, name) {
    var link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = name;
    document.body.appendChild(link);
    link.click();
    setTimeout(function () { URL.revokeObjectURL(link.href); link.remove(); }, 2000);
  }
  function fileName(holder) {
    var named = holder.querySelector(".ichart[aria-label]"), title = holder.querySelector("svg > title");
    var text = named ? named.getAttribute("aria-label") : title ? title.textContent : "chart";
    return text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "chart";
  }
  function exportChart(holder, kind, button) {
    var svg = chartSvg(holder);
    if (!svg) return;
    var name = fileName(holder);
    if (kind === "svg") return download(new Blob([svg.text], {type: "image/svg+xml"}), name + ".svg");
    if (kind === "copy") {
      // Safari wants the item made while the click is still being handled, with the image to come.
      navigator.clipboard.write([new ClipboardItem({"image/png": raster(svg, "image/png")})]).then(
        function () { flash(button, "Copied"); }, function () { flash(button, "Not copied", false); });
      return;
    }
    raster(svg, kind === "jpg" ? "image/jpeg" : "image/png").then(function (blob) { download(blob, name + "." + kind); });
  }
  function closeMenus(except) {
    document.querySelectorAll(".export.open").forEach(function (menu) {
      if (menu !== except) { menu.classList.remove("open"); menu.firstChild.setAttribute("aria-expanded", "false"); }
    });
  }
  document.querySelectorAll(".chart").forEach(function (holder) {
    var menu = document.createElement("div");
    menu.className = "export";
    menu.innerHTML = '<button type="button" class="tool icon" data-enlarge aria-label="Open the chart larger" ' +
      'data-tip="Open larger: zoom with the wheel or a pinch, drag to move">' + ENLARGE_ICON + '</button>' +
      '<button type="button" class="tool icon" aria-haspopup="menu" aria-expanded="false" ' +
      'data-tip="Save or copy this chart" aria-label="Save or copy this chart">' + SAVE_ICON + '</button>' +
      '<div class="menu" role="menu">' +
      (canCopyImage ? '<button type="button" role="menuitem" data-export="copy"><span class="label">Copy image</span></button>' : "") +
      '<button type="button" role="menuitem" data-export="png">Save as PNG</button>' +
      '<button type="button" role="menuitem" data-export="jpg">Save as JPG</button>' +
      '<button type="button" role="menuitem" data-export="svg">Save as SVG</button></div>';
    holder.appendChild(menu);
  });

  // -- Wide tables: a shadow on the side there is more to scroll to.
  function edges(scroller) {
    var room = scroller.scrollWidth - scroller.clientWidth;
    scroller.classList.toggle("more-left", room > 2 && scroller.scrollLeft > 2);
    scroller.classList.toggle("more-right", room > 2 && scroller.scrollLeft < room - 2);
  }
  var scrollers = Array.prototype.slice.call(document.querySelectorAll(".table"));
  scrollers.forEach(function (scroller) {
    scroller.addEventListener("scroll", function () { edges(scroller); }, {passive: true});
  });
  function allEdges() { scrollers.forEach(function (scroller) { if (scroller.offsetWidth) edges(scroller); }); }
  allEdges();
  document.querySelectorAll("details").forEach(function (d) { d.addEventListener("toggle", allEdges); });
  window.addEventListener("resize", allEdges);

  // -- A chart, enlarged: over the page, as wide as the window, with zoom and pan when it is drawn by ECharts.
  var zoom = document.createElement("div"), zoomChart = null;
  zoom.className = "zoombox";
  zoom.hidden = true;
  zoom.setAttribute("role", "dialog");
  zoom.setAttribute("aria-modal", "true");
  zoom.innerHTML = '<div class="zoompanel"><div class="zoomhead"><span class="zoomhint"></span>' +
    '<button type="button" class="tool icon" data-zoom-close aria-label="Close">' +
    '<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><path d="M3.5 3.5l9 9M12.5 3.5l-9 9"/></svg>' +
    '</button></div><div class="zoomchart"></div></div>';
  document.body.appendChild(zoom);
  function enlarge(holder) {
    var target = zoom.querySelector(".zoomchart"), box = holder.querySelector(".ichart"), svg = holder.querySelector(":scope > svg");
    zoom.hidden = false;
    document.documentElement.classList.add("zooming");
    target.innerHTML = "";
    target.style.height = "";
    zoom.setAttribute("aria-label", (box && box.getAttribute("aria-label")) || "Chart");
    if (box && window.zipfolioEnlarge) {
      zoomChart = window.zipfolioEnlarge(box, target);
      zoom.querySelector(".zoomhint").textContent = "Wheel or pinch to zoom, drag to move; the corner button resets.";
    } else if (svg) {
      var copy = svg.cloneNode(true);
      copy.removeAttribute("width");
      copy.removeAttribute("height");
      target.appendChild(copy);
      zoom.querySelector(".zoomhint").textContent = "";
    }
    zoom.querySelector("[data-zoom-close]").focus();
  }
  function shrink() {
    if (zoom.hidden) return;
    if (zoomChart) { zoomChart.dispose(); zoomChart = null; }
    zoom.querySelector(".zoomchart").innerHTML = "";
    zoom.hidden = true;
    document.documentElement.classList.remove("zooming");
  }
  zoom.addEventListener("click", function (event) { if (event.target === zoom) shrink(); });
  window.addEventListener("resize", function () { if (zoomChart) zoomChart.resize(); });

  // -- Hints: one floating box for every term, so a table's scroller never clips it.
  var tip = document.createElement("div"), current = null;
  tip.className = "tip";
  tip.setAttribute("role", "tooltip");
  document.body.appendChild(tip);
  function show(term) { if (current !== term) showTip(term, term.getAttribute("data-tip")); }
  function showTip(term, text) {
    current = term;
    tip.textContent = text;
    tip.style.display = "block";
    var at = term.getBoundingClientRect(), width = tip.offsetWidth, height = tip.offsetHeight;
    var left = Math.min(Math.max(8, at.left), window.innerWidth - width - 8);
    var top = at.bottom + 6 + height > window.innerHeight - 8 ? at.top - height - 6 : at.bottom + 6;
    tip.style.left = left + "px";
    tip.style.top = Math.max(8, top) + "px";
  }
  function hide() { current = null; tip.style.display = "none"; }
  document.addEventListener("mouseover", function (event) {
    var term = event.target.closest && event.target.closest(".term, [data-tip]");
    if (term) show(term); else if (current && document.activeElement !== current) hide();
  });
  document.addEventListener("focusin", function (event) {
    var term = event.target.closest && event.target.closest(".term, [data-tip]");
    if (term) show(term);
  });
  document.addEventListener("focusout", function (event) { if (event.target === current) hide(); });
  window.addEventListener("scroll", hide, {passive: true});
  document.addEventListener("keydown", function (event) { if (event.key === "Escape") { hide(); closeMenus(); shrink(); } });

  document.addEventListener("click", function (event) {
    var button = event.target.closest("button");
    var term = event.target.closest(".term");
    if (term) { show(term); return; }
    if (!button) { if (!event.target.closest(".totop")) hide(); closeMenus(); return; }
    if (button.hasAttribute("data-enlarge")) {
      hide();
      closeMenus();
      enlarge(button.closest(".chart"));
      return;
    }
    if (button.hasAttribute("data-zoom-close")) { shrink(); return; }
    if (button.hasAttribute("data-copy-report")) {
      write(null, reportText()).then(function () { flash(button, "Copied"); },
                                     function () { flash(button, "Not copied", false); });
      return;
    }
    if (button.hasAttribute("data-save-report")) {
      download(new Blob([reportText()], {type: "text/markdown;charset=utf-8"}), reportName());
      flash(button, "Saved");
      return;
    }
    if (button.hasAttribute("data-copy-table")) {
      var next = button.parentElement.nextElementSibling;
      var table = next && (next.tagName === "TABLE" ? next : next.querySelector("table"));
      if (table) copyTable(table, button);
      return;
    }
    var menu = button.parentElement;
    if (menu.classList.contains("export")) {
      var open = !menu.classList.contains("open");
      hide();
      closeMenus(menu);
      menu.classList.toggle("open", open);
      button.setAttribute("aria-expanded", String(open));
      return;
    }
    var kind = button.getAttribute("data-export");
    if (kind) {
      var holder = button.closest(".chart");
      exportChart(holder, kind, button);
      if (kind !== "copy") closeMenus();
      else setTimeout(closeMenus, 900);
    }
  });
})();
"""


#: The header's buttons: the report as text for the reader's own assistant, copied or saved. The text is
#: put together in the browser from what the page shows (``_TOOLS``): no raw book, no chart data.
_REPORT_BUTTONS = (
    '<div class="exports"><button type="button" class="tool" data-copy-report data-tip="The report\'s numbers as '
    'text -- the run, every metric and table, without the report\'s own assessments -- to paste into ChatGPT, Claude '
    'or any other assistant and talk the strategy over.">'
    '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><rect x="5.25" y="5.25" width="8.5" '
    'height="8.5" rx="1.75"/><path d="M10.75 5.25V4A1.75 1.75 0 0 0 9 2.25H4A1.75 1.75 0 0 0 2.25 4v5A1.75 1.75 0 0 '
    '0 4 10.75h1.25"/></svg><span class="label">Copy report for an AI</span></button>'
    '<button type="button" class="tool icon" data-save-report aria-label="Save the report as a Markdown file" '
    'data-tip="Save the same text as a Markdown file, to attach instead of pasting">'
    '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M8 1.5v8.5M4.5 6.8 8 10.2l3.5-3.4'
    'M2.5 12v2h11v-2"/></svg></button></div>')


def page_html(title: str, returns: pd.Series, benchmark, analytics: str, sections: list[Section],
              run: dict[str, str] | None = None, integrity: str = "") -> str:
    """The whole page. ``analytics`` names the libraries the numbers came from, with versions.

    ``run`` is what the header says about the backtest -- strategy, benchmark,
    capital, how orders fill -- as label and value, in order.
    """
    period = f"{returns.index[0]:{DATE_FORMAT}} &ndash; {returns.index[-1]:{DATE_FORMAT}}"
    generated = f"{datetime.now(timezone.utc):{DATE_FORMAT} %H:%M} UTC"
    run = dict(run or {})
    if "Benchmark" not in run:
        run["Benchmark"] = (str(benchmark.name) if benchmark is not None and benchmark.name else
                            "Benchmark" if benchmark is not None else "none")
    groups = ordered(sections)
    summary = summary_html(groups)
    parts = []
    for number, (name, members) in enumerate(groups):
        verdict = part_status(groups, name) if name else ""
        parts.append(f'<details class="part" id="part-{number + 1}" open><summary class="part-head">'
                     f'<span class="num">{_numeral(number)}</span>'
                     f'<span class="part-name">{html.escape(name or "More")}</span>'
                     f'<span class="part-verdict">{verdict}</span></summary>')
        parts.append("\n<hr>\n".join(_section_html(section) for section in members))
        parts.append('<p class="back"><a href="#summary">Summary &uarr;</a></p></details>')
    body = "\n".join(parts)
    jump = _jump_bar_html(groups, bool(summary))
    facts = "".join(f"<dt>{html.escape(label)}</dt><dd>{html.escape(str(value))}</dd>"
                    for label, value in run.items())
    data = scorecard_json(groups, run, returns) if summary else ""
    interactive = any('class="ichart"' in chart for section in sections for chart in section.charts)
    charts = (f"<script>{library()}</script>\n<script>{RUNTIME}</script>" if interactive else "")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="Zipfolio {__version__}">
<meta name="robots" content="noindex, nofollow">
<title>{html.escape(title)}</title>
<style>
  body {{ -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
         margin: 30px; background: #fff; color: #000; }}
  body, p, table, td, th {{ font: 13px/1.4 Arial, sans-serif; }}
  .container {{ max-width: 960px; margin: auto; }}
  h1, h2, h3, h4 {{ font-weight: 400; margin: 0; }}
  h1 {{ font-size: 26px; }}
  h2 {{ font-size: 18px; font-weight: 700; margin-bottom: 18px; }}
  h3 {{ margin-bottom: 10px; font-weight: 700; }}
  h4 {{ color: grey; margin-top: 4px; }}
  h4 strong {{ color: #333; }}
  h4 a, h4 a strong, footer a {{ color: #09c; text-decoration: none; white-space: nowrap; }}
  h4 a:hover, footer a:hover {{ color: #069; text-decoration: underline; }}
  .period {{ font-size: 15px; color: #333; margin: 4px 0 10px; }}
  dl.run {{ display: grid; grid-template-columns: max-content minmax(0, 1fr) max-content minmax(0, 1fr);
           gap: 3px 12px; margin: 0 0 10px; font-size: 12px; }}
  dl.run dt {{ color: grey; white-space: nowrap; }}
  dl.run dd {{ margin: 0 12px 0 0; color: #222; overflow-wrap: anywhere; }}
  hr {{ margin: 25px 0 40px; height: 0; border: 0; border-top: 1px solid #ccc; }}
  .row {{ display: flex; gap: 20px; align-items: flex-start; }}
  .left {{ flex: 0 0 620px; min-width: 0; }}
  .right {{ flex: 1 1 320px; min-width: 0; }}
  .chart {{ margin: 0 0 22px; }}
  .chart > svg {{ display: block; width: 100%; height: auto; }}
  .ichart {{ width: 100%; }}
  table {{ margin: 0 0 30px; border: 0; border-spacing: 0; width: 100%; }}
  table td, table th {{ text-align: right; padding: 4px 5px 3px 5px; }}
  table th {{ padding: 6px 5px 5px 5px; }}
  table td:first-of-type, table th:first-of-type {{ text-align: left; padding-left: 2px; }}
  table td:last-of-type, table th:last-of-type {{ padding-right: 2px; }}
  table thead th {{ font-weight: 700; background: #2d2d2d; color: #fff; }}
  tr.sep td {{ padding: 0; }}
  tr.sep hr {{ margin: 5px 0; }}
  .table, .wide {{ overflow-x: auto; }}
  .empty {{ color: grey; font-style: italic; margin: -4px 0 26px; }}
  .caption {{ color: grey; margin: -14px 0 26px; }}
  .explain {{ margin: 0 0 10px; }}
  .explain:last-of-type {{ margin-bottom: 26px; }}
  code {{ font: 12px Menlo, Consolas, monospace; background: #f3f3f3; padding: 0 3px; }}
  .check {{ display: flex; gap: 10px; margin: 0 0 12px; align-items: flex-start; }}
  .check:last-child {{ margin-bottom: 26px; }}
  .mark {{ flex: 0 0 20px; height: 20px; border-radius: 50%; color: #fff; font-weight: 700;
          font-size: 12px; line-height: 20px; text-align: center; }}
  .mark.ok {{ background: #6B9E00; }}
  .mark.warn {{ background: #D9A400; }}
  .mark.bad {{ background: #BA516B; }}
  .mark.info {{ background: #8A8A8A; font-style: italic; font-family: Georgia, serif; }}
  .note {{ color: #9a6700; }}
  details.howto {{ margin: 0 0 18px; border: 1px solid #e5e5e5; border-radius: 6px; padding: 6px 10px; }}
  details.howto > summary {{ cursor: pointer; color: #09c; font-size: 12px; font-weight: 700; }}
  details.howto[open] > summary {{ margin-bottom: 8px; }}
  details.howto .caption {{ margin: 0 0 10px; }}
  details.howto h3 {{ margin-top: 6px; }}
  table.checks .sub {{ font-weight: 400; color: #888; font-size: 11px; }}
  a {{ color: #09c; text-decoration: none; }}
  a:hover {{ color: #069; text-decoration: underline; }}
  section {{ scroll-margin-top: 48px; }}
  .jump {{ position: sticky; top: 0; z-index: 5; background: #fff; display: flex; gap: 18px; align-items: center;
          overflow-x: auto; white-space: nowrap; padding: 10px 0 8px; margin: 0 0 6px;
          border-bottom: 1px solid #e5e5e5; scrollbar-width: none; }}
  .jump::-webkit-scrollbar {{ display: none; }}
  .jump a {{ color: #333; font-weight: 700; font-size: 12px; }}
  .jump .expand {{ margin-left: auto; font: 700 12px Arial, sans-serif; color: #09c; background: none;
                   border: 1px solid #cfe7f2; border-radius: 4px; padding: 3px 8px; cursor: pointer; }}
  .num {{ display: inline-block; min-width: 18px; color: #6B9E00; font-weight: 700; }}
  .summary {{ margin-top: 22px; }}
  .verdict {{ font-size: 15px; line-height: 1.5; margin: -6px 0 16px; padding: 10px 14px; background: #f7f7f2;
             border-left: 4px solid #2d2d2d; border-radius: 0 6px 6px 0; }}
  .glance {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 10px 16px;
            margin: 0 0 16px; }}
  .fact .k {{ font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: .04em; }}
  .fact .fv {{ font-size: 14px; font-weight: 700; color: #222; }}
  .fact .fv.warn {{ color: #a87700; }} .fact .fv.bad {{ color: #a3405a; }}
  .lead {{ margin: -6px 0 12px; }}
  .tiles {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 0 0 18px; }}
  .tile {{ border: 1px solid #e3e3e3; border-left: 4px solid #8A8A8A; border-radius: 6px; padding: 10px 12px; }}
  .tile .k {{ font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: #555; }}
  .tile .n {{ font-size: 26px; font-weight: 700; margin: 2px 0; }}
  .tile .s {{ font-size: 11px; color: grey; }}
  .tile.ok {{ border-left-color: #6B9E00; }} .tile.ok .n {{ color: #4f7a00; }}
  .tile.warn {{ border-left-color: #D9A400; }} .tile.warn .n {{ color: #b07f00; }}
  .tile.bad {{ border-left-color: #BA516B; }} .tile.bad .n {{ color: #BA516B; }}
  table.checks {{ margin: 0 0 10px; }}
  table.checks td {{ padding: 6px 5px; border-bottom: 1px solid #eee; vertical-align: top; }}
  table.checks td.v {{ font-weight: 700; }}
  table.checks td.b {{ width: 1%; white-space: nowrap; }}
  table.checks tr.grp td {{ background: #f5f5f5; font-weight: 700; font-size: 12px; padding: 5px; }}
  table.checks tr.grp a {{ color: #333; }}
  table.checks summary {{ cursor: pointer; color: #222; list-style: none; }}
  table.checks summary::-webkit-details-marker {{ display: none; }}
  table.checks summary::before {{ content: "\\25B8"; color: #999; display: inline-block; width: 14px; }}
  table.checks details[open] > summary::before {{ content: "\\25BE"; }}
  table.checks .why {{ color: #444; font-size: 12px; margin: 4px 0 2px 14px; }}
  table.checks .why p {{ margin: 0 0 4px; }}
  table.checks .count {{ color: #888; font-size: 11px; font-weight: 700; margin-left: 6px; padding: 0 5px;
                         border: 1px solid #ddd; border-radius: 8px; }}
  table.checks th .hint {{ font-weight: 400; color: #bbb; font-size: 11px; margin-left: 6px; }}
  table.checks tr.family > td:first-child {{ font-weight: 700; }}
  table.checks ul.members {{ list-style: none; padding: 0; margin: 6px 0 0 14px; font-weight: 400; }}
  table.checks ul.members li {{ margin: 0 0 8px; }}
  table.checks ul.members .badge {{ font-size: 10px; padding: 1px 5px; }}
  table.checks ul.members .why {{ margin-left: 0; }}
  .badge {{ display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: .04em;
           text-transform: uppercase; padding: 2px 8px; border-radius: 4px; white-space: nowrap; }}
  .badge.ok {{ background: #eaf3dc; color: #4f7a00; }}
  .badge.warn {{ background: #fdf1d6; color: #a87700; }}
  .badge.bad {{ background: #f6dfe5; color: #a3405a; }}
  .badge.na {{ background: #eee; color: #888; }}
  table.checks tr.na td {{ color: #999; }}
  table.checks tr.na summary {{ color: #999; }}
  .context b.warn, .context b.bad {{ color: #a87700; }}
  .context {{ color: #555; font-size: 12px; line-height: 1.8; margin: 8px 0 0; }}
  .context span {{ color: grey; }}
  .context a {{ color: #555; white-space: nowrap; }}
  details.part {{ margin: 0; scroll-margin-top: 48px; }}
  .part-head {{ display: flex; align-items: baseline; gap: 8px; margin: 34px 0 22px; padding: 0 0 8px;
               border-bottom: 3px solid #2d2d2d; cursor: pointer; list-style: none; }}
  .part-head::-webkit-details-marker {{ display: none; }}
  .part-head::after {{ content: "Show"; font-size: 12px; color: #09c; white-space: nowrap; }}
  details[open] > .part-head::after {{ content: "Hide"; }}
  .part-head .num {{ font-size: 22px; min-width: 30px; }}
  .part-name {{ font-size: 22px; font-weight: 700; flex: 1 1 auto; }}
  .part-verdict {{ white-space: nowrap; }}
  details.part:not([open]) .part-head {{ margin-bottom: 0; }}
  .back {{ text-align: right; font-size: 12px; margin: 0 0 10px; }}
  footer {{ color: grey; font-size: 12px; margin-top: 10px; }}
  .tool {{ display: inline-flex; align-items: center; gap: 5px; font: 700 11px/14px Arial, sans-serif;
          letter-spacing: .04em; text-transform: uppercase; color: #555; background: #fff;
          border: 1px solid #d6d6d6; border-radius: 4px; padding: 3px 8px; cursor: pointer; }}
  .tool:hover, .tool:focus-visible, .export.open > .tool[aria-haspopup] {{ background: #2d2d2d; border-color: #2d2d2d; color: #fff; }}
  .tool.done {{ background: #6B9E00; border-color: #6B9E00; color: #fff; }}
  .tool svg, .totop svg {{ fill: none; stroke: currentColor; stroke-width: 1.5; stroke-linecap: round;
                          stroke-linejoin: round; display: block; }}
  .tool.icon {{ padding: 4px 5px; color: #777; border-color: #e0e0e0; }}
  .tool.icon:hover, .tool.icon:focus-visible, .export.open > .tool.icon[aria-haspopup] {{ color: #fff; border-color: #2d2d2d; }}
  .tool.icon.done {{ color: #fff; border-color: #6B9E00; }}
  .tool.icon.failed {{ color: #fff; border-color: #BA516B; }}
  .tool .i-done, .tool.done .i-copy {{ display: none; }}
  .tool.done .i-done {{ display: block; stroke-width: 2; }}
  .totop {{ position: fixed; right: 22px; bottom: 22px; z-index: 6; width: 42px; height: 42px; border-radius: 50%;
           display: flex; align-items: center; justify-content: center; background: #fff; color: #333;
           border: 1px solid #d6d6d6; box-shadow: 0 4px 14px rgba(0,0,0,.14); opacity: 0; visibility: hidden;
           transform: translateY(8px); transition: opacity .15s, transform .15s, visibility .15s; }}
  .totop.shown {{ opacity: 1; visibility: visible; transform: none; }}
  .totop:hover, .totop:focus-visible {{ background: #2d2d2d; border-color: #2d2d2d; color: #fff; text-decoration: none; }}
  .totop svg {{ stroke-width: 1.8; }}
  .tool.failed {{ background: #BA516B; border-color: #BA516B; color: #fff; }}
  .exports {{ display: flex; gap: 6px; align-items: center; margin: 10px 0 4px; }}
  .exports .tool {{ text-transform: none; letter-spacing: 0; font-size: 12px; }}
  .integrity {{ margin: 18px 0 4px; padding: 10px 14px; border: 1px solid #e3e3e3; border-left: 4px solid #8A8A8A;
               border-radius: 6px; }}
  .integrity.ok {{ border-left-color: #6B9E00; }} .integrity.warn {{ border-left-color: #D9A400; }}
  .integrity.bad {{ border-left-color: #BA516B; }}
  .integrity-head {{ display: flex; flex-wrap: wrap; align-items: center; gap: 6px 12px; }}
  .integrity-line {{ font-size: 12px; color: #444; }}
  .integrity-scope {{ font-size: 12px; color: #777; margin: 6px 0 0; }}
  .integrity-notes {{ list-style: none; padding: 0; margin: 8px 0 0; font-size: 12px; color: #333; }}
  .integrity-notes li {{ margin: 0 0 5px; }}
  .integrity-notes .badge {{ font-size: 9px; padding: 1px 5px; }}
  .integrity-missing {{ color: #888; }}
  details.integrity-evidence {{ margin: 8px 0 0; }}
  details.integrity-evidence > summary {{ cursor: pointer; color: #09c; font-size: 12px; font-weight: 700; }}
  details.integrity-evidence table {{ margin: 8px 0 4px; font-size: 12px; }}
  details.integrity-evidence td {{ vertical-align: top; }}
  .integrity-id {{ color: #999; font: 10px Menlo, Consolas, monospace; }}
  details.integrity-evidence .integrity-why {{ text-align: left; color: #555; min-width: 260px; }}
  details.integrity-evidence th.integrity-why {{ color: #fff; }}
  .tbl-head {{ display: flex; align-items: flex-end; gap: 10px; margin: 0 0 10px; }}
  .tbl-head h3 {{ margin: 0; flex: 1 1 auto; }}
  .tbl-head .tool {{ margin-left: auto; flex: none; }}
  .summary .tbl-head {{ margin-bottom: 6px; }}
  .chart {{ position: relative; }}
  .export {{ position: absolute; top: 0; right: 0; z-index: 3; display: flex; gap: 4px; }}
  .zoombox {{ position: fixed; inset: 0; z-index: 40; display: flex; align-items: center; justify-content: center;
             padding: 24px; background: rgba(20,20,20,.55); }}
  .zoombox[hidden] {{ display: none; }}
  .zoompanel {{ width: min(1400px, 100%); max-height: 100%; overflow: auto; background: #fff; border-radius: 10px;
               padding: 10px 16px 16px; box-shadow: 0 12px 40px rgba(0,0,0,.3); }}
  .zoomhead {{ display: flex; align-items: center; gap: 10px; margin: 0 0 6px; }}
  .zoomhint {{ flex: 1 1 auto; color: #888; font-size: 12px; }}
  .zoomchart {{ width: 100%; }}
  .zoomchart > svg {{ display: block; width: 100%; height: auto; max-height: calc(100vh - 110px); }}
  html.zooming {{ overflow: hidden; }}
  .export > .tool {{ opacity: .55; }}
  .chart:hover .export > .tool, .export.open > .tool, .export > .tool:focus-visible {{ opacity: 1; }}
  .zoomhead .tool.icon {{ opacity: 1; }}
  .export .menu {{ display: none; position: absolute; right: 0; top: calc(100% + 4px); min-width: 140px; padding: 4px;
                  background: #fff; border: 1px solid #ddd; border-radius: 6px; box-shadow: 0 6px 18px rgba(0,0,0,.12); }}
  .export.open .menu {{ display: block; }}
  .export .menu button {{ display: block; width: 100%; text-align: left; font: 12px/1.3 Arial, sans-serif; color: #222;
                         background: none; border: 0; border-radius: 4px; padding: 7px 10px; cursor: pointer; }}
  .export .menu button:hover, .export .menu button:focus-visible {{ background: #f1f1f1; }}
  .export .menu button.done {{ color: #4f7a00; font-weight: 700; }}
  .export .menu button.failed {{ color: #a3405a; font-weight: 700; }}
  .term {{ border-bottom: 1px dotted #9a9a9a; cursor: help; }}
  .term:focus {{ outline: none; border-bottom-style: solid; }}
  thead .term {{ border-bottom-color: rgba(255,255,255,.55); }}
  h2 .term, h3 .term {{ border-bottom-color: #cfcfcf; }}
  .tip {{ position: fixed; z-index: 30; display: none; max-width: 300px; padding: 7px 10px; border-radius: 6px;
         background: #2d2d2d; color: #fff; font: 12px/1.45 Arial, sans-serif; text-align: left;
         box-shadow: 0 4px 14px rgba(0,0,0,.18); pointer-events: none; }}
  @media (hover: none) {{ .export > .tool {{ opacity: .85; }} }}
  .table td.n {{ white-space: nowrap; }}
  .table td:first-child, .table th:first-child {{ position: sticky; left: 0; z-index: 1; }}
  .table td:first-child {{ background: #fff; }}
  .table.more-left td:first-child, .table.more-left th:first-child {{ box-shadow: 9px 0 8px -6px rgba(0,0,0,.32); }}
  .table.more-right {{ box-shadow: inset -12px 0 10px -10px rgba(0,0,0,.35); }}
  @media (max-width: 1000px) {{
    body {{ margin: 16px; }}
    .row {{ flex-direction: column; }}
    .left, .right {{ flex: 1 1 auto; width: 100%; }}
    .tiles {{ grid-template-columns: 1fr 1fr; }}
  }}
  @media (max-width: 640px) {{
    dl.run {{ grid-template-columns: max-content minmax(0, 1fr); }}
    .part-head .num, .part-name {{ font-size: 18px; }}
    .part-head {{ flex-wrap: wrap; }}
    /* The scorecard as cards: the check across the width, its value and verdict under it, and an
       opened check's reasoning across the full width rather than down a narrow first column. */
    table.checks, table.checks tbody {{ display: block; }}
    table.checks thead {{ display: none; }}
    table.checks tr {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; column-gap: 10px;
                      border-bottom: 1px solid #eee; }}
    table.checks tr.grp {{ display: block; border-bottom: 0; }}
    table.checks tr.grp td {{ display: block; }}
    table.checks td {{ border-bottom: 0; }}
    table.checks td:first-child {{ grid-column: 1 / -1; padding-bottom: 0; }}
    table.checks td.v {{ text-align: left; padding-top: 2px; }}
    table.checks td.b {{ padding-top: 2px; }}
    table.checks summary {{ padding: 4px 0; }}
    table.checks tr.grp a {{ display: block; padding: 6px 0; }}
    .jump {{ padding: 0; gap: 16px; }}
    .jump a {{ padding: 12px 0 11px; }}
    .jump .expand {{ position: sticky; right: 0; flex: none; background: #fff; padding: 6px 10px;
                     box-shadow: -18px 0 12px 2px #fff; }}
    .back a {{ display: inline-block; padding: 10px 0; }}
    details.howto > summary {{ padding: 6px 0; }}
    .totop {{ right: 14px; bottom: 14px; }}
    .zoombox {{ padding: 8px; }}
    .zoompanel {{ padding: 8px 8px 10px; }}
  }}
  @media (hover: none) {{
    .tool {{ padding: 6px 10px; }}
    .tool.icon {{ padding: 6px 7px; }}
  }}
  @media print {{
    body {{ margin: 0; }}
    .container {{ max-width: 100%; }}
    section {{ break-inside: avoid-page; }}
    .jump, .back, .tool, .export, .tip, .totop, .exports, .zoombox {{ display: none !important; }}
  }}
</style>
</head>
<body>
<div class="container">
<h1>{html.escape(title)}</h1>
<p class="period">{period} &middot; {len(returns):,} sessions</p>
<dl class="run">{facts}</dl>
<h4>Generated by <a href="{REPOSITORY_URL}" target="_blank" rel="noopener" title="{REPOSITORY_URL}"><strong>ZipLime Zipfolio v{__version__}</strong></a> on {generated}</h4>
{_REPORT_BUTTONS}
{jump}
{integrity}
{summary}
{body}
<hr>
<footer>Built with <a href="{REPOSITORY_URL}" target="_blank" rel="noopener">Zipfolio v{__version__}</a>, part of the ZipLime ecosystem &middot; generated on {generated} &middot; {analytics} &middot; <a href="{REPOSITORY_URL}" target="_blank" rel="noopener">github.com/Limex-com/zipfolio</a></footer>
</div>
<a class="totop" href="{'#summary' if summary else '#'}" aria-label="Back to the summary" data-tip="Back to the summary"><svg viewBox="0 0 16 16" width="17" height="17" aria-hidden="true"><path d="M8 13.5V3M3.5 7.5 8 3l4.5 4.5"/></svg></a>
{f'<script type="application/json" id="zipfolio-scorecard">{data}</script>' if data else ""}
<script>{_SCRIPT}</script>
{charts}
<script>{_TOOLS}</script>
</body>
</html>
"""
