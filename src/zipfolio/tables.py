"""Tables: caught from pyfolio as frames, rendered the way QuantStats Pro renders its own."""

from __future__ import annotations

import html
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .glossary import hint, title_hint


@dataclass
class Table:
    """A table pyfolio meant to print, with the formatting it meant to print it with."""

    frame: pd.DataFrame
    name: str | None
    float_format: Callable | None = None
    formatters: dict | None = None
    header_rows: dict | None = None


@contextmanager
def captured_tables():
    """Collect every table pyfolio prints while the block runs, as frames.

    Every table in pyfolio goes through ``utils.print_table`` -- the round-trip
    module calls it under a name of its own, imported at load time, so that one
    is replaced too. Catching the frame rather than the HTML pyfolio would have
    made is what lets the page render it in its own style.
    """
    from pyfolio import round_trips, utils

    tables: list[Table] = []

    def capture(table, name=None, float_format=None, formatters=None, header_rows=None):
        frame = table.to_frame() if isinstance(table, pd.Series) else table.copy()
        tables.append(Table(frame, name, float_format, formatters, header_rows))

    originals = (utils.print_table, round_trips.print_table)
    utils.print_table = round_trips.print_table = capture
    try:
        yield tables
    finally:
        utils.print_table, round_trips.print_table = originals


_SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "per", "the", "to", "vs"}


def _title_case(text: str) -> str:
    words = str(text).split()
    return " ".join(
        word if (index and word.lower() in _SMALL_WORDS) or not word[:1].islower() else word[:1].upper() + word[1:]
        for index, word in enumerate(words)
    )


def _cell(value, column, table: Table) -> str:
    if table.formatters and column in table.formatters and present(value):
        return table.formatters[column](value)
    if not present(value):
        return "-"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%-d %b %Y")  # page.DATE_FORMAT: one way to write a date on the page
    if isinstance(value, pd.Timedelta):
        return f"{value.days} days"
    if isinstance(value, (float, np.floating)) and table.float_format:
        return table.float_format(value)
    if isinstance(value, (float, np.floating)):
        return f"{value:,.2f}"
    text = str(value)
    # pyfolio writes its percentages with three decimals; two is what QuantStats
    # shows. Only from 1% up: below that the third decimal is the number.
    match = re.fullmatch(r"(-?\d+\.\d{3})%", text)
    if match and abs(float(match.group(1))) >= 1:
        return f"{float(match.group(1)):.2f}%"
    return text


#: A cell that reads as one token -- a number, an amount, a date -- and should not break across lines.
_COMPACT = re.compile(r"^(?:[-+−]?[$€₽]?[\d,.]+%?(?: ?(?:₽|€|x|bps|days?|sessions?))?|\d{1,2} [A-Z][a-z]{2} \d{4}|-)$")


def _td(text: str) -> str:
    return f'<td class="n">{html.escape(text)}</td>' if _COMPACT.match(text) else f"<td>{html.escape(text)}</td>"


def present(value) -> bool:
    if value is None or value is pd.NaT:
        return False
    try:
        return not pd.isna(value)
    except (TypeError, ValueError):
        return True


#: The button over a table that copies it for a spreadsheet -- two sheets, and a check once copied;
#: the page's script does the copying and shows the hint (``data-tip``).
COPY_BUTTON = (
    '<button type="button" class="tool icon" data-copy-table aria-label="Copy the table" '
    'data-tip="Copy the table, to paste into Excel or Google Sheets">'
    '<svg class="i-copy" viewBox="0 0 16 16" width="15" height="15" aria-hidden="true">'
    '<rect x="5.25" y="5.25" width="8.5" height="8.5" rx="1.75"/>'
    '<path d="M10.75 5.25V4A1.75 1.75 0 0 0 9 2.25H4A1.75 1.75 0 0 0 2.25 4v5A1.75 1.75 0 0 0 4 10.75h1.25"/></svg>'
    '<svg class="i-done" viewBox="0 0 16 16" width="15" height="15" aria-hidden="true">'
    '<path d="M3 8.5l3.25 3.25L13 4.75"/></svg></button>')


def term(label: str, explanation: str | None = None) -> str:
    """A label, escaped, with its hint attached: ``explanation``, or the glossary's (``glossary.hint``)."""
    text = html.escape(label)
    explanation = explanation or hint(label)
    if not explanation:
        return text
    return f'<span class="term" tabindex="0" data-tip="{html.escape(explanation)}">{text}</span>'


def table_html(
    table: Table,
    title: str | None = None,
    first_header: str | None = None,
    rename: dict[str, str] | None = None,
) -> str:
    """Render a pyfolio table the way QuantStats Pro renders its own, with a button that copies it."""
    frame = table.frame
    rename = rename or {}
    title = title or (_title_case(table.name) if table.name else None)
    heading = f"<h3>{term(title, title_hint(title)) if title_hint(title) else html.escape(title)}</h3>" if title else ""
    if frame.empty:
        return "\n".join(part for part in (heading, '<p class="empty">None.</p>') if part)
    parts = [f'<div class="tbl-head">{heading}{COPY_BUTTON}</div>']

    index_is_rank = isinstance(frame.index, pd.RangeIndex)
    headers = [first_header or _title_case(frame.index.name or "")]
    headers += [rename.get(str(column), _title_case(str(column))) for column in frame.columns]
    rows = []
    for label, value in (table.header_rows or {}).items():
        rows.append(f"<tr><td>{term(str(label))}</td>"
                    f'<td colspan="{len(frame.columns)}">{html.escape(_cell(value, None, table))}</td></tr>')
    if rows:
        rows.append(f'<tr class="sep"><td colspan="{len(headers)}"><hr></td></tr>')
    for position, (label, values) in enumerate(frame.iterrows()):
        label = position + 1 if index_is_rank else label
        if isinstance(label, str):
            label = label.replace("_", " ")
        cells = "".join(_td(_cell(value, column, table)) for column, value in values.items())
        rows.append(f"<tr><td>{term(_cell(label, None, table))}</td>{cells}</tr>")
    head = "".join(f"<th>{term(header)}</th>" for header in headers)
    parts.append(f'<div class="table"><table><thead><tr>{head}</tr></thead>'
                 f'<tbody>{"".join(rows)}</tbody></table></div>')
    return "\n".join(parts)
