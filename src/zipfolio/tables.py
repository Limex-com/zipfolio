"""Tables: caught from pyfolio as frames, rendered the way QuantStats Pro renders its own."""

from __future__ import annotations

import html
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


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
        return value.strftime("%Y-%m-%d")
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


def present(value) -> bool:
    if value is None or value is pd.NaT:
        return False
    try:
        return not pd.isna(value)
    except (TypeError, ValueError):
        return True


def table_html(
    table: Table,
    title: str | None = None,
    first_header: str | None = None,
    rename: dict[str, str] | None = None,
) -> str:
    """Render a pyfolio table the way QuantStats Pro renders its own."""
    frame = table.frame
    rename = rename or {}
    title = title or (_title_case(table.name) if table.name else None)
    parts = [f"<h3>{html.escape(title)}</h3>"] if title else []
    if frame.empty:
        parts.append('<p class="empty">None.</p>')
        return "\n".join(parts)

    index_is_rank = isinstance(frame.index, pd.RangeIndex)
    headers = [first_header or _title_case(frame.index.name or "")]
    headers += [rename.get(str(column), _title_case(str(column))) for column in frame.columns]
    rows = []
    for label, value in (table.header_rows or {}).items():
        rows.append(f"<tr><td>{html.escape(str(label))}</td>"
                    f'<td colspan="{len(frame.columns)}">{html.escape(_cell(value, None, table))}</td></tr>')
    if rows:
        rows.append(f'<tr class="sep"><td colspan="{len(headers)}"><hr></td></tr>')
    for position, (label, values) in enumerate(frame.iterrows()):
        label = position + 1 if index_is_rank else label
        if isinstance(label, str):
            label = label.replace("_", " ")
        cells = "".join(f"<td>{html.escape(_cell(value, column, table))}</td>"
                        for column, value in values.items())
        rows.append(f"<tr><td>{html.escape(_cell(label, None, table))}</td>{cells}</tr>")
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    parts.append(f'<div class="table"><table><thead><tr>{head}</tr></thead>'
                 f'<tbody>{"".join(rows)}</tbody></table></div>')
    return "\n".join(parts)
