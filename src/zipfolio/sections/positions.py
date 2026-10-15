"""Exposure, concentration, holdings and leverage over the backtest."""

from __future__ import annotations

from ..page import BAD, OK, WARN, Section
from ..style import anchor_zero, chart, held_only, mark_isolated
from ..tables import captured_tables
from .instruments import CONCENTRATION_ADVICE

#: Gross leverage above the first needs a look; above the second the result misleads -- borrowing is not modeled.
LEVERAGE_NOTE, LEVERAGE_WARN = 1.05, 1.5
#: A typical day's largest position above this share of the portfolio needs a look.
LARGEST_POSITION = 0.25


def build_positions(returns, positions) -> Section:
    from pyfolio import plotting, pos

    section = Section("Positions")
    allocation = pos.get_percent_alloc(positions)
    names = positions.drop(columns="cash", errors="ignore")
    held_names = [column for column in names.columns if names[column].abs().sum() > 0]
    shorts = bool((names < 0).any().any())
    section.chart(chart("Exposure", lambda ax: plotting.plot_exposures(returns, positions, ax=ax),
                        y="pct"))
    def top_positions(ax):
        # Lines only while a position is open -- see style.held_only. The
        # tables pyfolio prints from the same frame skip the gaps.
        plotting.show_and_plot_top_positions(returns, held_only(allocation), show_and_plot=2, ax=ax)
        mark_isolated(ax)
        anchor_zero(ax)

    with captured_tables() as tables:
        section.chart(chart("Top 10 Positions Allocation", top_positions, height=4.0, y="pct"))
    for table in tables:
        # A long-only book: the long list is the full list, and the short one is empty.
        if not shorts and table.name != "Top 10 positions of all time":
            continue
        section.table(table, first_header="Symbol", rename={"max": "Max Allocation"})
    if len(held_names) > 1:
        section.chart(chart("Long/Short Max and Median Position Concentration",
                            lambda ax: plotting.plot_max_median_position_concentration(positions, ax=ax),
                            y="pct"))
        section.chart(chart("Total Holdings", lambda ax: plotting.plot_holdings(returns, allocation, ax=ax)))
        if shorts:
            section.chart(chart("Long and Short Holdings", lambda ax: plotting.plot_long_short_holdings(
                returns, allocation, ax=ax)))
    section.chart(chart("Gross Leverage", lambda ax: plotting.plot_gross_leverage(returns, positions, ax=ax)))

    largest = allocation.drop(columns="cash", errors="ignore").abs().max(axis="columns")
    held = largest[largest > 0]
    if len(held):
        only = f" ({held_names[0]}, the only name)" if len(held_names) == 1 else ""
        section.finding(OK if held.median() <= LARGEST_POSITION else WARN, "Largest position, typical day",
                        f"{held.median():.0%}{only}" if only else f"{held.median():.0%} (max {held.max():.0%})",
                        f"The largest position is {held.median():.0%} of the portfolio on a typical day it holds "
                        f"anything, {held.max():.0%} at its largest. Over {LARGEST_POSITION:.0%} means one name can "
                        "swing the result.", key="largest", threshold=f"at most {LARGEST_POSITION:.0%}",
                        issue=(f"the whole book is one stock, {held_names[0]}" if only else
                               f"one position is {held.median():.0%} of the book on a typical day"),
                        advice=CONCENTRATION_ADVICE)
    leverage = names.abs().sum(axis="columns") / positions.sum(axis="columns")
    peak = float(leverage.max())
    common = dict(key="leverage", threshold=f"at most {LEVERAGE_NOTE:.2f}x",
                  advice="Keep the gross exposure at or under the capital: borrowing and shorting are not charged.")
    if peak > LEVERAGE_NOTE:
        section.finding(BAD if peak > LEVERAGE_WARN else WARN, "Max gross exposure", f"{peak:.2f}x",
                        f"{leverage.median():.2f}x on a typical day. Above 1x the strategy borrows or shorts, and "
                        "neither borrowing nor shorting costs are modeled.",
                        issue=f"it borrows or shorts up to {peak:.2f}x, at no modeled cost", **common)
    else:
        section.finding(OK, "Max gross exposure", f"{peak:.2f}x",
                        "The book never held more than its capital: nothing borrowed, nothing to charge for.",
                        **common)
    return section
