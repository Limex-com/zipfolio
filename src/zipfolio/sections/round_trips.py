"""Closed trades -- pyfolio's round trips -- in words a reader can follow.

pyfolio counts a "round trip" for every sale that closes shares bought
earlier, matched oldest first, not one per position; a strategy that trims
at each rebalance shows several per holding. The section says so up front,
with examples, and calls them trades.
"""

from __future__ import annotations

import pandas as pd

from ..currency import money, unit
from ..page import INFO, OK, WARN, Section
from ..style import LOSS_COLOR, STRATEGY_COLOR, chart
from ..tables import Table, captured_tables, present

#: What a trade is, for the page: the matching rule pyfolio applies, with examples.
TRADE_EXPLAINED = (
    "<h3>What Counts as a Trade</h3>"
    '<p class="explain">A <b>trade</b> is shares bought and later sold -- or sold short and later '
    "bought back -- and what they earned in between. Every sale closes the shares bought earliest "
    "first, and every sale is a trade of its own:</p>"
    '<ul class="explain">'
    "<li>buy 100, later sell 100: <b>one trade</b>;</li>"
    "<li>buy 100; sell 30 at the next rebalance and the other 70 a month later: <b>two trades</b>, "
    "of 30 and of 70 shares, each held from the purchase to its own sale;</li>"
    "<li>buy 100, buy 50 more, sell all 150: <b>one trade</b>, at the average of the two purchases.</li>"
    "</ul>"
    '<p class="explain">So a strategy that trims its holdings at every rebalance shows many more trades '
    "than positions. Fills on the same side within 8 hours count as one; what is still held at the end "
    "is counted as sold at the last price. <b>PnL</b> is before commission; a trade's <b>return</b> is "
    "its PnL over what its shares cost.</p>"
)

#: pyfolio's table and row names, said plainly.
TABLE_TITLES = {"Summary stats": "How Many Trades", "PnL stats": "Profit and Loss per Trade",
                "Duration stats": "How Long Trades Were Held", "Return stats": "Return per Trade"}
COLUMN_NAMES = {"All trades": "All", "Long trades": "Long", "Short trades": "Short"}


def extract_round_trips(returns, positions, transactions) -> pd.DataFrame:
    """pyfolio's round trips: one row per closing sale (or buy-back), see ``TRADE_EXPLAINED``.

    ``returns`` in the frame is pyfolio's -- PnL over the whole portfolio's value;
    ``rt_returns`` is PnL over what the trade's shares cost, which is what this
    report calls a trade's return.
    """
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
        # pyfolio's own threshold: fewer trades than this say nothing.
        return None

    section = Section("Closed Trades")
    section.side.append(TRADE_EXPLAINED)
    # pyfolio's return tables read ``returns``; a trade's return is on its own cost.
    on_cost = trades.assign(returns=trades["rt_returns"])
    with captured_tables() as tables:
        round_trips.print_round_trip_stats(on_cost)
        plotting.show_profit_attribution(trades)
    for table in tables:
        if "Profitability" in (table.name or ""):
            section.table(table, title="Share of Profit by Symbol", first_header="Symbol",
                          rename={"pnl": "Share of PnL"})
        else:
            section.table(_tidy_round_trip_table(table), title=TABLE_TITLES.get(table.name),
                          first_header="", rename=COLUMN_NAMES)

    section.chart(chart("When Each Trade Was Open", lambda ax: plotting.plot_round_trip_lifetimes(
        trades, ax=ax), height=max(3.0, 0.32 * min(trades["symbol"].nunique(), 16) + 1.4),
        recolor={"#0000ff": STRATEGY_COLOR, "#ff0000": LOSS_COLOR}))
    section.chart(chart("How Likely a Trade Is to Make Money",
                        lambda ax: plotting.plot_prob_profit_trade(trades, ax=ax),
                         recolor={"#0000ff": STRATEGY_COLOR, "#1f77b4": STRATEGY_COLOR}))

    def histogram(values, xlabel):
        def draw(ax):
            sns.histplot(values, ax=ax, color=STRATEGY_COLOR)
            ax.set(xlabel=xlabel, ylabel="Trades")
        return draw

    section.chart(chart("Holding Time", histogram([x.days for x in trades["duration"]], "Days")))
    section.chart(chart("PnL per Trade", histogram(trades["pnl"], f"PnL, {unit()}")))
    section.chart(chart("Return per Trade", histogram(trades["rt_returns"].dropna() * 100,
                                                      "Return on what the shares cost, %")))

    _open_and_streak_findings(section, trades, positions)
    pnl = trades["pnl"].astype(float)
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    factor = wins / losses if losses else float("inf")
    shown = f"{factor:.2f}" if losses else "no losers"
    section.finding(WARN if factor < 1 else OK, "Winning trades · profit factor",
                    f"{(pnl > 0).mean():.0%} · {shown}",
                    f"{len(pnl):,} closed trades; median holding time {trades['duration'].median().days} days. "
                    "The profit factor is what the winning trades made over what the losing ones lost: "
                    "under 1, the trades lose money on balance.")
    return section


#: Profit still on paper at the end -- positions counted as sold at the last price -- above this share needs a look.
OPEN_SHARE = 0.3


def losing_streak(pnl: pd.Series) -> int:
    """The most losing trades in a row, in the order they closed."""
    losing = (pnl < 0).astype(int)
    return int(losing.groupby((losing == 0).cumsum()).sum().max()) if len(losing) else 0


def _open_and_streak_findings(section: Section, trades: pd.DataFrame, positions: pd.DataFrame) -> None:
    closing = positions.index[-1] + pd.Timedelta(seconds=1)  # where pyfolio closes what is still held
    still_open = trades["close_dt"] == closing
    total = float(trades["pnl"].sum())
    paper = float(trades.loc[still_open, "pnl"].sum())
    if total > 0:
        share = max(paper, 0.0) / total
        section.finding(OK if share <= OPEN_SHARE else WARN, "Profit still open at the end", f"{share:.0%}",
                        f"{int(still_open.sum()):,} positions were still held when the backtest ended; they are "
                        f"counted as sold at the last price, for {money(paper, 0)} of the {money(total, 0)} net. "
                        f"Over {OPEN_SHARE:.0%} means much of the result is a paper gain.")
    streak = losing_streak(trades.sort_values("close_dt")["pnl"].astype(float))
    section.finding(INFO, "Longest losing streak", f"{streak} trades",
                    "The most losing trades in a row, in the order they closed.")


def _tidy_round_trip_table(table: "Table") -> "Table":
    """Counts as counts and money as money in pyfolio's round-trip tables.

    pyfolio formats a whole table one way: the trade counts come out as
    "146.00", every row of the PnL table gets a dollar sign -- the profit factor
    and the win/loss ratio included -- and a loss reads "$-5067.13", in
    dollars whatever the currency.
    """
    frame = table.frame.astype(object)
    frame.index = [_plain(row) for row in frame.index]
    for row in frame.index:
        label = str(row).lower()
        if "percent" in label:
            frame.loc[row] = frame.loc[row].map(lambda v: f"{v:.0%}" if present(v) else v)
        elif "number" in label or label.startswith(("winning", "losing", "even", "trades")):
            frame.loc[row] = frame.loc[row].map(lambda v: f"{int(v):,}" if present(v) else v)
        elif table.name == "Return stats":
            frame.loc[row] = frame.loc[row].map(lambda v: f"{v / 100:.2%}" if present(v) else v)
        elif table.name == "PnL stats":
            if "factor" in label or "ratio" in label:
                frame.loc[row] = frame.loc[row].map(lambda v: f"{v:.2f}" if present(v) else v)
            else:
                frame.loc[row] = frame.loc[row].map(
                    lambda v: money(v) if present(v) else v)
    return Table(frame, table.name, table.float_format, table.formatters, table.header_rows)


def _plain(row) -> str:
    """pyfolio's row names without its ``round_trips``: "Total number of round_trips" -> "Trades"."""
    text = str(row)
    replacements = {"Total number of round_trips": "Trades", "Percent profitable": "Percent profitable",
                    "Avg returns all round_trips": "Average, all trades",
                    "Median returns all round_trips": "Median, all trades",
                    "Avg returns winning": "Average, winners", "Avg returns losing": "Average, losers",
                    "Median returns winning": "Median, winners", "Median returns losing": "Median, losers"}
    return replacements.get(text, text.replace("round_trips", "trades"))
