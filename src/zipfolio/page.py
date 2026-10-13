"""The page: sections of charts and tables in QuantStats Pro's layout."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from . import __version__
from .tables import Table, table_html

#: Tables wider than this many columns go under the charts rather than beside them.
SIDE_TABLE_MAX_COLUMNS = 3

#: Where Zipfolio lives; every report links to it, in the header and the footer,
#: next to the version that built it.
REPOSITORY_URL = "https://github.com/Limex-com/zipfolio"

#: The report's title unless the caller names it.
DEFAULT_TITLE = "Zipfolio Tearsheet: Portfolio, Execution & Cost Analysis"


#: How a finding reads: fine, worth a look, wrong, or just a fact.
OK, WARN, BAD, INFO = "ok", "warn", "bad", "info"
MARKS = {OK: "&#10003;", WARN: "!", BAD: "&#10007;", INFO: "i"}
#: The summary lists what needs attention first.
SEVERITY = {BAD: 0, WARN: 1, INFO: 2, OK: 3}

#: The page's four parts, each the question a reader brings to it, and the
#: sections that answer it, in page order. A section missing here goes last.
GROUPS: list[tuple[str, list[str]]] = [
    ("Where the Return Came From", [
        "Performance", "Profit Concentration", "Instruments", "Long and Short", "Sector Analysis",
        "Factor Analysis"]),
    ("What It Risked", ["Drawdowns", "Market Regimes", "Portfolio Risk", "Risk Over Time"]),
    ("How It Holds and Trades", ["Positions", "Transactions", "Round Trips"]),
    ("Can It Be Traded", ["Next-Bar Fills", "Execution", "Trading Costs", "Capacity"]),
]


@dataclass
class Finding:
    """One row of the summary: a measure, its value, and how much it matters.

    ``label`` and ``value`` are what the row shows -- "Without best 5 days",
    "30.9% → 5.2%"; ``note`` is the sentence behind it, shown on hover.
    """

    status: str
    label: str
    value: str
    note: str = ""
    section: str = ""


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

    def finding(self, status: str, label: str, value: str, note: str = "") -> None:
        self.findings.append(Finding(status, label, value, note, self.title))

    def chart(self, svg: str | None) -> None:
        if svg:
            self.charts.append(svg)

    def table(self, table: "Table", **kwargs) -> None:
        rendered = table_html(table, **kwargs)
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
    return f'<nav class="jump">{"".join(links)}</nav>'


#: How a card names its worst finding, and what the dots mean.
VERDICTS = {BAD: "needs work", WARN: "needs a look", INFO: "for context", OK: "holds up"}
LEGEND = {OK: "holds up", INFO: "for context", WARN: "needs a look", BAD: "wrong or misleading"}


def summary_html(groups) -> str:
    """One card per part of the page: its findings as measure, value and a coloured dot.

    The card's colour is its worst finding, so a glance at the four says where
    the problems are; each row links to its section, and its full sentence
    is on hover. The cards double as the contents: each lists its sections.
    """
    order = {title: rank for rank, title in enumerate(title for _, titles in GROUPS for title in titles)}
    cards = []
    for number, (name, members) in enumerate(groups):
        findings = [finding for section in members if not section.failed for finding in section.findings]
        findings.sort(key=lambda f: (order.get(f.section, 99), SEVERITY.get(f.status, 9)))
        worst = min((f.status for f in findings), key=lambda status: SEVERITY[status], default=INFO)
        flagged = sum(f.status in (BAD, WARN) for f in findings)
        verdict = f"{flagged} to check" if flagged else VERDICTS[worst]
        rows = "".join(
            f'<tr title="{html.escape(f.note)}"><td class="d"><span class="dot {f.status}"></span></td>'
            f'<td><a href="#{anchor(f.section)}">{html.escape(f.label)}</a></td>'
            f'<td class="v">{html.escape(f.value)}</td></tr>'
            for f in findings)
        table = f'<table class="score">{rows}</table>' if rows else ""
        links = " &middot; ".join(f'<a href="#{section.anchor}">{html.escape(section.title)}</a>'
                                  for section in members)
        cards.append(
            f'<div class="card {worst}"><a class="card-head" href="#part-{number + 1}">'
            f'<span class="num">{_numeral(number)}</span><span class="card-name">{html.escape(name or "More")}</span>'
            f'<span class="verdict {BAD if worst == BAD else WARN if flagged else worst}">{verdict}</span></a>'
            f'{table}<div class="card-foot">{links}</div></div>')
    if not any("<table" in card for card in cards):
        return ""
    legend = "".join(f'<span><span class="dot {status}"></span>{text}</span>' for status, text in LEGEND.items())
    return ('<section id="summary" class="summary"><h2>Summary</h2>'
            f'<div class="cards">{"".join(cards)}</div>'
            f'<p class="legend">{legend}<span>Hover a row for the reasoning; click it for the numbers.</span></p>'
            '</section>')


def _section_html(section: Section) -> str:
    parts = [f'<section id="{section.anchor}"><h2>{html.escape(section.title)}</h2>']
    if section.failed:
        parts.append('<p class="note">This section could not be computed for this backtest.</p>')
    else:
        charts = "\n".join(f'<div class="chart">{svg}</div>' for svg in section.charts)
        side = "\n".join(section.side)
        parts.append(f'<div class="row"><div class="left">{charts}</div><div class="right">{side}</div></div>')
        if section.wide:
            parts.append(f'<div class="wide">{"".join(section.wide)}</div>')
    parts.append("</section>")
    return "\n".join(parts)


def page_html(title: str, returns: pd.Series, benchmark, analytics: str, sections: list[Section]) -> str:
    """The whole page. ``analytics`` names the libraries the numbers came from, with versions."""
    period = f"{returns.index[0]:%-d %b, %Y} - {returns.index[-1]:%-d %b, %Y}"
    generated = f"{datetime.now(timezone.utc):%-d %b, %Y %H:%M} UTC"
    benchmark_name = html.escape(str(benchmark.name)) if benchmark is not None and benchmark.name else (
        "Benchmark" if benchmark is not None else "none")
    groups = ordered(sections)
    summary = summary_html(groups)
    parts = []
    for number, (name, members) in enumerate(groups):
        parts.append(f'<div class="part" id="part-{number + 1}"><span class="num">{_numeral(number)}</span>'
                     f'<span class="part-name">{html.escape(name or "More")}</span>'
                     '<a class="top" href="#summary">Summary &uarr;</a></div>')
        parts.append("\n<hr>\n".join(_section_html(section) for section in members))
    body = "\n".join(parts)
    jump = _jump_bar_html(groups, bool(summary))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="Zipfolio">
<meta name="robots" content="noindex, nofollow">
<title>{html.escape(title)}</title>
<style>
  body {{ -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
         margin: 30px; background: #fff; color: #000; }}
  body, p, table, td, th {{ font: 13px/1.4 Arial, sans-serif; }}
  .container {{ max-width: 960px; margin: auto; }}
  h1, h2, h3, h4 {{ font-weight: 400; margin: 0; }}
  h1 {{ font-size: 26px; }}
  h1 dt {{ display: inline; margin-left: 10px; font-size: 14px; }}
  h2 {{ font-size: 18px; font-weight: 700; margin-bottom: 18px; }}
  h3 {{ margin-bottom: 10px; font-weight: 700; }}
  h4 {{ color: grey; margin-top: 4px; }}
  h4 strong {{ color: #333; }}
  h4 a, h4 a strong, footer a {{ color: #09c; text-decoration: none; white-space: nowrap; }}
  h4 a:hover, footer a:hover {{ color: #069; text-decoration: underline; }}
  hr {{ margin: 25px 0 40px; height: 0; border: 0; border-top: 1px solid #ccc; }}
  .row {{ display: flex; gap: 20px; align-items: flex-start; }}
  .left {{ flex: 0 0 620px; min-width: 0; }}
  .right {{ flex: 1 1 320px; min-width: 0; }}
  .chart {{ margin: 0 0 22px; }}
  .chart svg {{ display: block; width: 100%; height: auto; }}
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
  a {{ color: #09c; text-decoration: none; }}
  a:hover {{ color: #069; text-decoration: underline; }}
  section {{ scroll-margin-top: 48px; }}
  .jump {{ position: sticky; top: 0; z-index: 5; background: #fff; display: flex; gap: 18px;
          overflow-x: auto; white-space: nowrap; padding: 10px 0 8px; margin: 0 0 6px;
          border-bottom: 1px solid #e5e5e5; scrollbar-width: none; }}
  .jump::-webkit-scrollbar {{ display: none; }}
  .jump a {{ color: #333; font-weight: 700; font-size: 12px; }}
  .num {{ display: inline-block; min-width: 18px; color: #6B9E00; font-weight: 700; }}
  .summary {{ margin-top: 22px; }}
  .cards {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .card {{ border: 1px solid #e3e3e3; border-top: 4px solid #8A8A8A; border-radius: 6px;
          padding: 12px 14px 10px; min-width: 0; }}
  .card.ok {{ border-top-color: #6B9E00; }}
  .card.warn {{ border-top-color: #D9A400; }}
  .card.bad {{ border-top-color: #BA516B; }}
  .card-head {{ display: flex; align-items: baseline; gap: 6px; color: #000; font-weight: 700;
               font-size: 14px; margin: 0 0 8px; }}
  .card-head:hover {{ color: #000; text-decoration: none; }}
  .card-name {{ flex: 1 1 auto; }}
  .verdict {{ font-size: 11px; padding: 2px 8px; border-radius: 10px; color: #fff; white-space: nowrap;
             background: #8A8A8A; }}
  .verdict.ok {{ background: #6B9E00; }}
  .verdict.warn {{ background: #D9A400; }}
  .verdict.bad {{ background: #BA516B; }}
  table.score {{ margin: 0 0 8px; }}
  table.score td {{ padding: 5px 4px; border-bottom: 1px solid #f0f0f0; text-align: left; vertical-align: top; }}
  table.score td.d {{ width: 12px; padding-right: 2px; }}
  table.score td.v {{ text-align: right; font-weight: 700; }}
  table.score a {{ color: #222; }}
  table.score tr:hover td {{ background: #fafafa; }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #8A8A8A;
         margin: 4px 6px 0 0; }}
  .dot.ok {{ background: #6B9E00; }}
  .dot.warn {{ background: #D9A400; }}
  .dot.bad {{ background: #BA516B; }}
  .card-foot {{ font-size: 11px; color: grey; line-height: 1.6; }}
  .card-foot a {{ color: #888; }}
  .legend {{ color: grey; font-size: 12px; margin: 12px 0 0; display: flex; flex-wrap: wrap; gap: 4px 16px; }}
  .legend .dot {{ margin-top: 0; vertical-align: -1px; }}
  .part {{ display: flex; align-items: baseline; gap: 8px; margin: 46px 0 22px; padding: 0 0 8px;
          border-bottom: 3px solid #2d2d2d; scroll-margin-top: 48px; }}
  .part .num {{ font-size: 22px; min-width: 30px; }}
  .part-name {{ font-size: 22px; font-weight: 700; flex: 1 1 auto; }}
  .part .top {{ font-size: 12px; white-space: nowrap; }}
  footer {{ color: grey; font-size: 12px; margin-top: 10px; }}
  @media (max-width: 1000px) {{
    body {{ margin: 16px; }}
    .row {{ flex-direction: column; }}
    .left, .right {{ flex: 1 1 auto; width: 100%; }}
    .cards {{ grid-template-columns: 1fr; }}
  }}
  @media (max-width: 560px) {{
    .part .num, .part-name {{ font-size: 18px; }}
  }}
  @media print {{
    body {{ margin: 0; }}
    .container {{ max-width: 100%; }}
    section {{ break-inside: avoid-page; }}
    .jump, .part .top {{ display: none; }}
  }}
</style>
</head>
<body>
<div class="container">
<h1>{html.escape(title)} <dt>{period}</dt></h1>
<h4>Benchmark: {benchmark_name} &bull; Sessions: {len(returns):,} &bull; Generated by <a href="{REPOSITORY_URL}" target="_blank" rel="noopener" title="{REPOSITORY_URL}"><strong>ZipLime Zipfolio v{__version__}</strong></a> on {generated}</h4>
{jump}
{summary}
{body}
<hr>
<footer>Built with <a href="{REPOSITORY_URL}" target="_blank" rel="noopener">Zipfolio v{__version__}</a>, part of the ZipLime ecosystem &middot; generated on {generated} &middot; {analytics} &middot; <a href="{REPOSITORY_URL}" target="_blank" rel="noopener">github.com/Limex-com/zipfolio</a></footer>
</div>
</body>
</html>
"""
