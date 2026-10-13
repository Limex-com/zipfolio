"""Exposure, concentration, holdings and leverage over the backtest."""

from __future__ import annotations

from ..page import Section
from ..style import anchor_zero, chart, held_only, mark_isolated
from ..tables import captured_tables


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
    return section
