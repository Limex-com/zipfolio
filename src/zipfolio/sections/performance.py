"""Performance, kept short: the QuantStats tearsheet beside this one covers returns."""

from __future__ import annotations

from ..page import Section
from ..style import chart
from ..tables import captured_tables


def build_performance(returns, benchmark, positions, transactions) -> Section:
    """Key metrics, cumulative returns and the underwater curve -- nothing more.

    The QuantStats tearsheet a ZipLime backtest gets beside this report already has
    the rolling statistics, the monthly heatmap and the return distributions;
    repeating them here would only make this page longer.
    """
    from pyfolio import plotting

    section = Section("Performance")
    with captured_tables() as tables:
        plotting.show_perf_stats(returns, benchmark, positions=positions, transactions=transactions)
    for table in tables:
        section.table(table, title="Key Performance Metrics", first_header="Metric",
                      rename={"Backtest": "Strategy"})

    versus = " vs Benchmark" if benchmark is not None else ""
    section.chart(chart(f"Cumulative Returns{versus}", lambda ax: plotting.plot_rolling_returns(
        returns, factor_returns=benchmark, ax=ax), height=4.5, y="growth"))

    section.chart(chart("Underwater Plot", lambda ax: plotting.plot_drawdown_underwater(
        returns=returns, ax=ax)))
    return section
