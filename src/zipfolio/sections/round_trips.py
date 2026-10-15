"""Closed trades -- pyfolio's round trips -- in words a reader can follow.

pyfolio counts a "round trip" for every sale that closes shares bought
earlier, matched oldest first, not one per position; a strategy that trims
at each rebalance shows several per holding. The section says so up front,
with examples, and calls them trades.
"""

from __future__ import annotations

import pandas as pd

from ..accounting import profit_breakdown
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
    "than positions. Fills on the same side within 8 hours count as one. What is still held at the end is "
    "not a trade -- it was never sold -- and its paper profit is counted apart, as unrealized PnL (Profit "
    "Reconciliation, in Performance). A trade's <b>PnL</b> is before commission; its <b>return</b> is its "
    "PnL over what its shares cost.</p>"
)

#: pyfolio's table and row names, said plainly.
TABLE_TITLES = {"Summary stats": "How Many Trades", "PnL stats": "Profit and Loss per Trade",
                "Duration stats": "How Long Trades Were Held", "Return stats": "Return per Trade",
                "Symbol stats": "Return per Trade, by Symbol"}
#: Fewer closed trades than this are too few to judge a strategy by.
MIN_TRADES = 30
COLUMN_NAMES = {"All trades": "All", "Long trades": "Long", "Short trades": "Short"}


def extract_round_trips(returns, positions, transactions) -> pd.DataFrame:
    """The closed trades: pyfolio's round trips, one row per closing sale (or buy-back), see ``TRADE_EXPLAINED``.

    Built from the fills alone. pyfolio's own tear sheet first adds a sale of
    everything still held, at the last price (``add_closing_transactions``), and
    counts it as a trade; it is not one -- nothing was sold -- and pyfolio dates
    it one second after the last position snapshot, which in this report is
    midnight before the last session's own fills: a name bought that day came
    out as a short sold at midnight and bought back at the close. What is still
    held is unrealized PnL, which ``accounting.profit_breakdown`` counts apart.

    ``returns`` in the frame is pyfolio's -- PnL over the whole portfolio's value;
    ``rt_returns`` is PnL over what the trade's shares cost, which is what this
    report calls a trade's return. ``pnl`` is before commission.
    """
    from pyfolio import round_trips

    empty = pd.DataFrame(columns=["symbol", "pnl", "open_dt", "close_dt", "long", "rt_returns", "returns",
                                  "duration"])
    fills = transactions[["symbol", "amount", "price"]]
    fills = fills[fills["amount"] != 0]
    if fills.empty or not _closes_something(fills):
        return empty  # pyfolio fails on a book that never sold: there is nothing to match
    # extract_round_trips wants the beginning-of-day portfolio value.
    return round_trips.extract_round_trips(fills, portfolio_value=positions.sum(axis="columns") / (1 + returns))


def _closes_something(fills: pd.DataFrame) -> bool:
    """Whether any fill trades against a position held at the time -- a sale of shares held, a buy-back."""
    for _, own in fills.sort_index().groupby("symbol"):
        held = own["amount"].cumsum().shift(fill_value=0)
        if ((held != 0) & (held * own["amount"] < 0)).any():
            return True
    return False


def build_round_trips(returns, positions, transactions, holdings=None, orders=None) -> Section | None:
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
    shorts = bool((trades["long"] == False).any()) if "long" in trades else True  # noqa: E712
    names = trades["symbol"].nunique()
    for table in tables:
        if table.name == "Symbol stats" and names < 2:
            continue  # one name: the same numbers as Return per Trade
        tidy = _tidy_round_trip_table(table)
        if not shorts:
            # A long-only book: the "Long" column repeats "All" and "Short" is empty.
            tidy.frame = tidy.frame.drop(columns=[c for c in ("Long trades", "Short trades") if c in tidy.frame])
        section.table(tidy, title=TABLE_TITLES.get(table.name), first_header="", rename=COLUMN_NAMES)

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

    _open_and_streak_findings(section, trades, returns, positions if holdings is None else holdings, transactions,
                              orders)
    pnl = trades["pnl"].astype(float)
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    factor = wins / losses if losses else float("inf")
    shown = f"{factor:.2f}" if losses else "no losers"
    section.finding(WARN if factor < 1 else OK, "Winning trades · profit factor",
                    f"{(pnl > 0).mean():.0%} · {shown}",
                    f"{len(pnl):,} closed trades; median holding time {trades['duration'].median().days} days. "
                    "The profit factor is what the winning trades made over what the losing ones lost: "
                    "under 1, the trades lose money on balance.", key="win_pf", threshold="profit factor above 1",
                    issue=f"the trades lose money on balance (profit factor {shown})",
                    advice="Look at the losing trades: an exit that cuts them earlier, or an entry that skips them.")
    enough = len(pnl) >= MIN_TRADES
    section.finding(OK if enough else WARN, "Closed trades: enough to judge", f"{len(pnl):,}",
                    f"{len(pnl):,} closed trades over {len(returns):,} sessions. Under {MIN_TRADES}, a win rate, a "
                    "profit factor or a luck test moves a lot with one trade more or less.", key="sample",
                    threshold=f"at least {MIN_TRADES}",
                    issue=f"only {len(pnl):,} closed trades -- too few to judge by",
                    advice="Run the rule over a longer history or more names, until it has made at least "
                           f"{MIN_TRADES} trades.")
    return section


#: Profit still on paper at the end -- unrealized PnL of what is still held -- above this share needs a look.
OPEN_SHARE = 0.3


def losing_streak(pnl: pd.Series) -> int:
    """The most losing trades in a row, in the order they closed."""
    losing = (pnl < 0).astype(int)
    return int(losing.groupby((losing == 0).cumsum()).sum().max()) if len(losing) else 0


def _open_and_streak_findings(section: Section, trades: pd.DataFrame, returns, holdings: pd.DataFrame,
                              transactions: pd.DataFrame, orders) -> None:
    profit = profit_breakdown(returns, holdings, transactions, orders, trades)
    total, paper = profit.trading, profit.unrealized
    if total > 0:
        share = max(paper, 0.0) / total
        section.finding(OK if share <= OPEN_SHARE else WARN, "Profit still open at the end", f"{share:.0%}",
                        f"Unrealized PnL over all PnL, both before commission: {profit.open_names:,} positions "
                        f"were still held when the backtest ended, {money(paper, 0)} above what their shares cost at "
                        f"the last price, of {money(total, 0)} in all ({money(profit.realized, 0)} realized by the "
                        f"{profit.closed_trades:,} closed trades). Over {OPEN_SHARE:.0%} means much of the result "
                        "is a paper gain.", key="open_profit",
                        threshold=f"at most {OPEN_SHARE:.0%} of the profit",
                        issue=f"{share:.0%} of the profit is a paper gain on positions still open",
                        advice="Run the backtest past the point the open positions are closed.")
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
                    # pyfolio's PnL rows are before commission; "net" there means winners net of losers.
                    "Total profit": "Total PnL, before commission", "Gross profit": "Gross profit (winning trades)",
                    "Gross loss": "Gross loss (losing trades)", "Avg. trade net profit": "Avg. trade PnL",
                    "Avg returns all round_trips": "Average, all trades",
                    "Median returns all round_trips": "Median, all trades",
                    "Avg returns winning": "Average, winners", "Avg returns losing": "Average, losers",
                    "Median returns winning": "Median, winners", "Median returns losing": "Median, losers"}
    return replacements.get(text, text.replace("round_trips", "trades"))
