"""Exposure, concentration, holdings and leverage over the backtest."""

from __future__ import annotations

from ..page import INFO, WARN, Section
from ..style import anchor_zero, chart, held_only, mark_isolated
from ..tables import captured_tables

#: Gross leverage above this is worth a line in the summary; above the second, a warning.
LEVERAGE_NOTE, LEVERAGE_WARN = 1.05, 1.5


def build_positions(returns, positions) -> Section:
    from pyfolio import plotting, pos

    section = Section("Positions")
    allocation = pos.get_percent_alloc(positions)
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
        section.table(table, first_header="Symbol", rename={"max": "Max Allocation"})
    section.chart(chart("Long/Short Max and Median Position Concentration",
                        lambda ax: plotting.plot_max_median_position_concentration(positions, ax=ax),
                         y="pct"))
    section.chart(chart("Total Holdings", lambda ax: plotting.plot_holdings(returns, allocation, ax=ax)))
    section.chart(chart("Long and Short Holdings", lambda ax: plotting.plot_long_short_holdings(
        returns, allocation, ax=ax)))
    section.chart(chart("Gross Leverage", lambda ax: plotting.plot_gross_leverage(returns, positions, ax=ax)))

    largest = allocation.drop(columns="cash", errors="ignore").abs().max(axis="columns")
    held = largest[largest > 0]
    if len(held):
        section.finding(INFO, f"The largest position is {held.median():.0%} of the portfolio on a typical day",
                        f"{held.max():.0%} at its largest.")
    leverage = positions.drop(columns="cash", errors="ignore").abs().sum(axis="columns") / positions.sum(axis="columns")
    if leverage.max() > LEVERAGE_NOTE:
        section.finding(WARN if leverage.max() > LEVERAGE_WARN else INFO,
                        f"Gross leverage reached {leverage.max():.2f}x",
                        f"{leverage.median():.2f}x on a typical day. Borrowing and shorting costs are not "
                        "modeled.")
    return section
