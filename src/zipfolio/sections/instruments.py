"""Returns by instrument: what each name contributed, and how its returns were distributed."""

from __future__ import annotations

import traceback

import numpy as np
import pandas as pd

from ..currency import money, money_formatter, unit
from ..inputs import utc_dates
from ..page import INFO, OK, WARN, Section
from ..style import LOSS_COLOR, PALETTE, STRATEGY_COLOR, chart
from ..tables import Table
from .round_trips import extract_round_trips

#: How many instruments the charts name; the table lists all of them.
CHART_INSTRUMENTS = 15
CUMULATIVE_LINES = 8


def instrument_pnl(holdings: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Daily mark-to-market PnL per instrument, before commission.

    For instrument *i* on day *t*::

        PnL(i, t) = V(i, t) - V(i, t-1) - spent(i, t)

    where ``V`` is the end-of-day value of the position (quantity x last price,
    negative for a short) and ``spent`` the signed value of that day's fills --
    a buy spends cash, a sell brings it back. Nothing is estimated: summed over
    the instruments it is the portfolio's PnL before commission, and slippage
    is inside it, as it is inside the fill prices.
    """
    values = holdings.drop(columns="cash", errors="ignore")
    spent = (transactions["amount"] * transactions["price"]).groupby(
        [utc_dates(transactions.index), transactions["symbol"].astype(str)]).sum().unstack(fill_value=0.0)
    symbols = values.columns.union(spent.columns)
    values = values.reindex(columns=symbols, fill_value=0.0).fillna(0.0)
    spent = spent.reindex(index=values.index, columns=symbols, fill_value=0.0).fillna(0.0)
    return values - values.shift(1, fill_value=0.0) - spent


def commission_by_day(orders: pd.DataFrame | None, index: pd.DatetimeIndex, columns) -> pd.DataFrame:
    """Commission per instrument per day, from the orders that paid it."""
    if orders is None or orders.empty or "commission" not in orders.columns:
        return pd.DataFrame(0.0, index=index, columns=columns)
    charged = orders.drop_duplicates("id", keep="last") if "id" in orders.columns else orders
    amounts = pd.to_numeric(charged["commission"], errors="coerce").fillna(0.0)
    by_day = amounts.groupby([utc_dates(charged["dt"]), charged["symbol"].astype(str)]).sum().unstack(fill_value=0.0)
    return by_day.reindex(index=index, columns=columns, fill_value=0.0).fillna(0.0)


def _concentration_finding(section: Section, summary: pd.DataFrame) -> None:
    traded = summary[summary["Days Held"] > 0]
    if len(traded) < 3:
        return
    total = float(traded["Net PnL"].sum())
    if total > 0:
        top = traded["Net PnL"].nlargest(2)
        share = float(top.sum()) / total
        section.finding(WARN if share >= 0.75 else OK, "Profit from the top 2 names",
                        f"{share:.0%} ({top.index[0]}, {top.index[1]})",
                        f"{top.index[0]} and {top.index[1]} made {share:.0%} of the net PnL; "
                        f"{len(traded):,} instruments held in all. Over 100% means the rest lost money.")
    else:
        worst = traded["Net PnL"].nsmallest(2)
        section.finding(INFO, "Biggest losers", f"{worst.index[0]}, {worst.index[1]}",
                        f"A net loss of {money(-total, 0)}; {worst.index[0]} and {worst.index[1]} lost the most "
                        f"of {len(traded):,} instruments held.")


def build_instruments(returns, holdings, positions, transactions, orders=None) -> Section:
    """What each instrument earned, its share of the result, and the spread of its returns.

    ``holdings`` is the book as recorded (dollars per symbol and ``cash``);
    ``positions`` the same after pyfolio's intraday adjustment, which is what
    its round trips are built from.
    """
    section = Section("Instruments")
    gross = instrument_pnl(holdings, transactions)
    commission = commission_by_day(orders, gross.index, gross.columns)
    pnl = gross - commission
    start_value = holdings.sum(axis="columns") / (1 + returns)
    contribution = pnl.div(start_value, axis="index")
    values = holdings.drop(columns="cash", errors="ignore").reindex(columns=pnl.columns, fill_value=0.0)
    weights = values.div(holdings.sum(axis="columns"), axis="index")
    held = values.ne(0)

    previous = values.shift(1)
    daily_returns = (gross / previous.abs()).where(previous.ne(0) & previous.notna())

    trades = None
    try:
        trades = extract_round_trips(returns, positions, transactions)
    except Exception:
        print(f"Warning: no trades for the instrument table:\n{traceback.format_exc()}")

    total_pnl = pnl.sum()
    summary = pd.DataFrame({
        "Net PnL": total_pnl,
        "Contribution": contribution.sum(),
        "Share of PnL": total_pnl / total_pnl.sum() if total_pnl.sum() else np.nan,
        "Avg. Weight": weights.where(held).mean(),
        "Days Held": held.sum(),
        "Commission": commission.sum(),
    })
    if trades is not None and len(trades):
        by_symbol = trades.groupby(trades["symbol"].astype(str))["rt_returns"]
        summary["Trades"] = by_symbol.size()
        summary["Win Rate"] = by_symbol.apply(lambda r: (r > 0).mean())
        summary["Avg. Trade"] = by_symbol.mean()
        summary["Best Trade"] = by_symbol.max()
        summary["Worst Trade"] = by_symbol.min()
    summary = summary.sort_values("Net PnL", ascending=False)
    _concentration_finding(section, summary)

    money = money_formatter()
    percent = "{:.2%}".format
    section.table(
        Table(summary, None, None, {
            "Net PnL": money, "Contribution": percent, "Share of PnL": "{:.1%}".format,
            "Avg. Weight": "{:.1%}".format, "Days Held": "{:,.0f}".format, "Commission": money,
            "Trades": "{:,.0f}".format, "Win Rate": "{:.0%}".format, "Avg. Trade": percent,
            "Best Trade": percent, "Worst Trade": percent,
        }),
        title="Returns by Instrument", first_header="Symbol")
    section.side.append(
        '<p class="explain"><b>Net PnL</b> is each instrument\'s mark-to-market result: the change in '
        'the value of its position, less what was paid for it, less its commission. Slippage is '
        'inside it, as it is in the fill prices. <b>Contribution</b> adds up its daily PnL over the '
        'portfolio\'s value at the start of each day, so the contributions add up to the strategy\'s '
        'return before compounding. <b>Avg. weight</b> is over the days the position was held.</p>'
        '<p class="explain">The two can disagree, even in sign. A dollar earned when the portfolio '
        'was small is a larger share of it than a dollar earned after it grew: a name that made '
        'money early and gave back a similar amount later shows a positive contribution and a '
        'small or negative PnL, and the reverse.</p>'
        '<p class="explain">The trade columns count closed trades, as in <a href="#closed-trades">Closed '
        'Trades</a> -- each sale with the shares it closed -- and a trade\'s return is its PnL over what '
        'those shares cost.</p>')

    ranked = summary["Contribution"].reindex(
        summary["Contribution"].abs().sort_values(ascending=False).index).head(CHART_INSTRUMENTS)

    def contributions(ax):
        bars = ranked.sort_values()
        ax.barh(bars.index, bars.to_numpy(), color=np.where(bars >= 0, STRATEGY_COLOR, LOSS_COLOR))
        ax.axvline(0, color="black", lw=1)
        from matplotlib.ticker import FuncFormatter

        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))

    section.chart(chart("Contribution to Return by Instrument", contributions,
                        height=max(2.6, 0.32 * len(ranked) + 1.2)))

    top = summary["Net PnL"].abs().sort_values(ascending=False).index[:CUMULATIVE_LINES]

    def cumulative(ax):
        running = pnl.cumsum()
        for symbol, color in zip(top, PALETTE * 2):
            ax.plot(running.index, running[symbol], color=color, lw=1.4, label=symbol)
        others = [symbol for symbol in pnl.columns if symbol not in top]
        if others:
            ax.plot(running.index, running[others].sum(axis="columns"), color="#999999", lw=1.2,
                    ls="--", label=f"Other {len(others)}")
        ax.axhline(0, ls="--", lw=1, color="black", zorder=1)
        ax.legend(loc="upper left", ncol=2)
        ax.set_ylabel(f"Cumulative PnL, {unit()}")

    section.chart(chart("Cumulative PnL by Instrument", cumulative, height=4.0))

    years = sorted(set(pnl.index.year))
    if len(years) > 1:
        yearly = contribution.groupby(contribution.index.year).sum().transpose().loc[ranked.index] * 100
        yearly = yearly.round(1) + 0.0  # no "-0.0" in a cell

        def by_year(ax):
            import seaborn as sns

            sns.heatmap(yearly, annot=True, fmt=".1f", annot_kws={"size": 9}, cmap="RdYlGn", center=0.0,
                        cbar=False, ax=ax, linewidths=0.5, linecolor="white")
            ax.set_xlabel("")
            ax.set_ylabel("")

        section.chart(chart("Contribution by Instrument and Year (%)", by_year,
                            height=max(2.4, 0.34 * len(yearly) + 1.0)))

    order = daily_returns[ranked.index].median().sort_values().index
    section.chart(chart("Daily Returns by Instrument (While Held)", lambda ax: _boxes(
        ax, [daily_returns[symbol].dropna().to_numpy() for symbol in order], order),
        height=max(2.6, 0.32 * len(order) + 1.2)))

    if trades is not None and len(trades):
        grouped = {str(symbol): group["rt_returns"].dropna().to_numpy()
                   for symbol, group in trades.groupby(trades["symbol"].astype(str))}
        order = [symbol for symbol in ranked.index if symbol in grouped]
        order = sorted(order, key=lambda symbol: np.median(grouped[symbol]) if len(grouped[symbol]) else 0.0)
        section.chart(chart("Trade Returns by Instrument", lambda ax: _boxes(
            ax, [grouped[symbol] for symbol in order], order, points=True),
            height=max(2.6, 0.32 * len(order) + 1.2)))
    return section


def _boxes(ax, samples, labels, points: bool = False) -> None:
    """Horizontal box plots, one per instrument, on a percent axis."""
    from matplotlib.ticker import FuncFormatter

    kept = [(sample, label) for sample, label in zip(samples, labels) if len(sample)]
    if not kept:
        return
    samples, labels = zip(*kept)
    ax.boxplot(samples, vert=False, tick_labels=list(labels), widths=0.6, patch_artist=True,
               showfliers=not points,
               boxprops={"facecolor": STRATEGY_COLOR, "alpha": 0.55, "edgecolor": "#333333"},
               medianprops={"color": "#333333", "lw": 1.4},
               whiskerprops={"color": "#666666"}, capprops={"color": "#666666"},
               flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": "#999999",
                           "markeredgecolor": "none", "alpha": 0.6})
    if points:
        rng = np.random.default_rng(0)
        for position, sample in enumerate(samples, start=1):
            ax.scatter(sample, position + rng.uniform(-0.18, 0.18, len(sample)), s=9,
                       color=np.where(np.asarray(sample) >= 0, STRATEGY_COLOR, LOSS_COLOR), alpha=0.7,
                       zorder=3, linewidths=0)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
