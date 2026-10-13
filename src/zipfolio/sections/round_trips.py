"""Round trips: every position from open to close."""

from __future__ import annotations

import pandas as pd

from ..currency import money, unit
from ..page import INFO, WARN, Section
from ..style import LOSS_COLOR, STRATEGY_COLOR, chart
from ..tables import Table, captured_tables, present


def extract_round_trips(returns, positions, transactions) -> pd.DataFrame:
    from pyfolio import round_trips

    closed = round_trips.add_closing_transactions(positions, transactions)
    # extract_round_trips wants the beginning-of-day portfolio value.
    return round_trips.extract_round_trips(
        closed, portfolio_value=positions.sum(axis="columns") / (1 + returns))


def build_round_trips(returns, positions, transactions) -> Section | None:
    import seaborn as sns
    from pyfolio import plotting, round_trips

    trades = extract_round_trips(returns, positions, transactions)
    if len(trades) < 5:
        # pyfolio's own threshold: fewer round trips than this say nothing.
        return None

    section = Section("Round Trips")
    with captured_tables() as tables:
        round_trips.print_round_trip_stats(trades)
        plotting.show_profit_attribution(trades)
    for table in tables:
        if "Profitability" in (table.name or ""):
            section.table(table, title="Profit Attribution by Symbol", first_header="Symbol",
                          rename={"pnl": "Share of PnL"})
        else:
            section.table(_tidy_round_trip_table(table), first_header="Statistic")

    section.chart(chart("Round Trip Lifetimes", lambda ax: plotting.plot_round_trip_lifetimes(
        trades, ax=ax), height=max(3.0, 0.32 * min(trades["symbol"].nunique(), 16) + 1.4),
        recolor={"#0000ff": STRATEGY_COLOR, "#ff0000": LOSS_COLOR}))
    section.chart(chart("Probability of Making a Profitable Decision",
                        lambda ax: plotting.plot_prob_profit_trade(trades, ax=ax),
                         recolor={"#0000ff": STRATEGY_COLOR, "#1f77b4": STRATEGY_COLOR}))

    def histogram(values, xlabel):
        def draw(ax):
            sns.histplot(values, ax=ax, color=STRATEGY_COLOR)
            ax.set(xlabel=xlabel)
        return draw

    section.chart(chart("Holding Time", histogram([x.days for x in trades["duration"]], "Days")))
    section.chart(chart("PnL per Round Trip", histogram(trades["pnl"], f"PnL, {unit()}")))
    section.chart(chart("Round Trip Returns", histogram(trades["returns"].dropna() * 100, "Return, %")))

    pnl = trades["pnl"].astype(float)
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    factor = wins / losses if losses else float("inf")
    shown = f"{factor:.2f}" if losses else "no losing trades"
    section.finding(WARN if factor < 1 else INFO,
                    f"{len(pnl):,} round trips, {(pnl > 0).mean():.0%} profitable, profit factor {shown}",
                    f"Median holding time {trades['duration'].median().days} days.")
    return section


def _tidy_round_trip_table(table: "Table") -> "Table":
    """Counts as counts and money as money in pyfolio's round-trip tables.

    pyfolio formats a whole table one way: the trade counts come out as
    "146.00", every row of the PnL table gets a dollar sign -- the profit factor
    and the win/loss ratio included -- and a loss reads "$-5067.13", in
    dollars whatever the currency.
    """
    frame = table.frame.astype(object)
    for row in frame.index:
        label = str(row).lower()
        if "percent" in label:
            frame.loc[row] = frame.loc[row].map(lambda v: f"{v:.0%}" if present(v) else v)
        elif "number" in label or label.startswith(("winning", "losing", "even")):
            frame.loc[row] = frame.loc[row].map(lambda v: f"{int(v):,}" if present(v) else v)
        elif table.name == "PnL stats":
            if "factor" in label or "ratio" in label:
                frame.loc[row] = frame.loc[row].map(lambda v: f"{v:.2f}" if present(v) else v)
            else:
                frame.loc[row] = frame.loc[row].map(
                    lambda v: money(v) if present(v) else v)
    return Table(frame, table.name, table.float_format, table.formatters, table.header_rows)
