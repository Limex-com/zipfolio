"""The page: sections of charts and tables in QuantStats Pro's layout."""

from __future__ import annotations

import html
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


@dataclass
class Section:
    """One block of the page: charts on the left, narrow tables on the right, wide ones under both."""

    title: str
    charts: list[str] = field(default_factory=list)
    side: list[str] = field(default_factory=list)
    wide: list[str] = field(default_factory=list)
    failed: bool = False

    @property
    def empty(self) -> bool:
        return not (self.charts or self.side or self.wide)

    def chart(self, svg: str | None) -> None:
        if svg:
            self.charts.append(svg)

    def table(self, table: "Table", **kwargs) -> None:
        rendered = table_html(table, **kwargs)
        wide = len(table.frame.columns) > SIDE_TABLE_MAX_COLUMNS
        (self.wide if wide else self.side).append(rendered)


def _section_html(section: Section) -> str:
    parts = [f'<section><h2>{html.escape(section.title)}</h2>']
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
    body = "\n<hr>\n".join(_section_html(section) for section in sections)
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
  footer {{ color: grey; font-size: 12px; margin-top: 10px; }}
  @media (max-width: 1000px) {{
    body {{ margin: 16px; }}
    .row {{ flex-direction: column; }}
    .left, .right {{ flex: 1 1 auto; width: 100%; }}
  }}
  @media print {{
    body {{ margin: 0; }}
    .container {{ max-width: 100%; }}
    section {{ break-inside: avoid-page; }}
  }}
</style>
</head>
<body>
<div class="container">
<h1>{html.escape(title)} <dt>{period}</dt></h1>
<h4>Benchmark: {benchmark_name} &bull; Sessions: {len(returns):,} &bull; Generated by <a href="{REPOSITORY_URL}" target="_blank" rel="noopener" title="{REPOSITORY_URL}"><strong>ZipLime Zipfolio v{__version__}</strong></a> on {generated}</h4>
<hr>
{body}
<hr>
<footer>Built with <a href="{REPOSITORY_URL}" target="_blank" rel="noopener">Zipfolio v{__version__}</a>, part of the ZipLime ecosystem &middot; generated on {generated} &middot; {analytics} &middot; <a href="{REPOSITORY_URL}" target="_blank" rel="noopener">github.com/Limex-com/zipfolio</a></footer>
</div>
</body>
</html>
"""
