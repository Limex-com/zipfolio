"""Allocation and profit by GICS sector."""

from __future__ import annotations

import pandas as pd

from ..page import OK, WARN, Section
from ..style import anchor_zero, chart, held_only, mark_isolated
from ..sectors import UNCLASSIFIED
from ..tables import Table, captured_tables
from .round_trips import extract_round_trips

#: One sector holding more than this share of the gross book on a typical day needs a look.
SECTOR_SHARE = 0.4


def sector_concentration(positions: pd.DataFrame, sectors: dict[str, str]) -> pd.DataFrame:
    """Each sector's share of the gross book (longs and shorts both counted), per session held."""
    values = positions.drop(columns="cash", errors="ignore").abs()
    by_sector = values.T.groupby(lambda symbol: sectors.get(symbol, UNCLASSIFIED)).sum().T
    gross = by_sector.sum(axis="columns")
    return by_sector.div(gross.where(gross > 0), axis="index").dropna(how="all")


def build_sector_analysis(returns, positions, transactions, sectors: dict[str, str]) -> Section:
    """Allocation by sector over time, its average and peak, and profit attribution by sector.

    Built from the pieces ``create_position_tear_sheet`` and
    ``create_round_trip_tear_sheet`` draw when handed ``sector_mappings``, so
    the sector view has a section of its own rather than a last panel in two.
    """
    from pyfolio import plotting, pos, round_trips

    section = Section("Sector Analysis")
    shares = sector_concentration(positions, sectors).drop(columns=UNCLASSIFIED, errors="ignore")
    if not shares.empty and shares.notna().any().any():
        top = shares.max(axis="columns")
        leader = shares.idxmax(axis="columns").dropna().mode().iloc[0]
        typical = float(top.median())
        section.finding(OK if typical <= SECTOR_SHARE else WARN, "Largest sector, typical day",
                        f"{typical:.0%} ({leader})",
                        f"On a typical day the largest sector held {typical:.0%} of the gross book, {leader} more "
                        f"often than any other. Names in one sector fall together; over {SECTOR_SHARE:.0%} needs a "
                        "look.", key="sector")
    exposures = pos.get_sector_exposures(positions, sectors)
    allocation = pos.get_percent_alloc(exposures).drop("cash", axis="columns")
    average = allocation.mean().sort_values(ascending=False)
    section.table(
        Table(pd.DataFrame({"Average": average, "Max": allocation.max()[average.index]}),
              None, "{:.2%}".format),
        title="Sector Allocation", first_header="Sector")

    trades = extract_round_trips(returns, positions, transactions)
    if len(trades):
        with captured_tables() as tables:
            plotting.show_profit_attribution(round_trips.apply_sector_mappings_to_round_trips(trades, sectors))
        for table in tables:
            section.table(table, title="Profit Attribution by Sector", first_header="Sector",
                          rename={"pnl": "Share of PnL"})

    def sector_allocation(ax):
        # A sector with nothing in it is a gap, not a zero -- see style.held_only.
        plotting.plot_sector_allocations(returns, held_only(allocation), ax=ax)
        mark_isolated(ax)
        anchor_zero(ax)
        # pyfolio puts the legend beside the plot, which makes the picture twice
        # as wide and everything in it half the size once it fits the column.
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)

    section.chart(chart("Sector Allocation Over Time", sector_allocation, height=4.0, y="pct"))
    return section
