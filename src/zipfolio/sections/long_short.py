"""Long and short: what each side of the book earned.

The positions section shows how much was long and how much short; this one
shows what each side made of it. For a long-short strategy it is the first
question: is the edge in both legs, or is one of them paying for the other?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..currency import compact, money, money_formatter, unit
from ..inputs import utc_dates
from ..page import OK, WARN, Section
from ..style import BENCHMARK_COLOR, LOSS_COLOR, STRATEGY_COLOR, chart
from ..tables import Table
from .instruments import commission_by_day, instrument_pnl

#: A side that gives back more than this share of what the other made is worth a warning.
GIVE_BACK = 0.5


def side_of_day(holdings: pd.DataFrame, transactions: pd.DataFrame, columns) -> pd.DataFrame:
    """+1 long, -1 short, per instrument per day: the side the day's PnL belongs to.

    The side held at the previous close, since that position earned the day;
    for a position opened that day, the side it was opened on; for one opened
    and closed within the day, the side of its first fill. A day that turns a
    long into a short is filed under the side it started on.
    """
    values = holdings.drop(columns="cash", errors="ignore").reindex(columns=columns, fill_value=0.0).fillna(0.0)
    side = np.sign(values.shift(1, fill_value=0.0))
    side = side.where(side != 0, np.sign(values))
    first = (transactions["amount"].groupby([utc_dates(transactions.index), transactions["symbol"].astype(str)])
             .first().unstack().reindex(index=values.index, columns=columns))
    side = side.where(side != 0, np.sign(first.fillna(0.0)))
    return side


def build_long_short(returns, holdings, transactions, orders=None) -> Section | None:
    from matplotlib.ticker import FuncFormatter

    values = holdings.drop(columns="cash", errors="ignore")
    if not (values < 0).any().any():
        return None  # long only: the positions section already says all there is

    gross = instrument_pnl(holdings, transactions)
    pnl = gross - commission_by_day(orders, gross.index, gross.columns)
    side = side_of_day(holdings, transactions, pnl.columns)
    values = values.reindex(columns=pnl.columns, fill_value=0.0).fillna(0.0)
    legs = pd.DataFrame({
        "Long": pnl.where(side > 0, 0.0).sum(axis="columns"),
        "Short": pnl.where(side < 0, 0.0).sum(axis="columns"),
    })
    # A day with no position on either side: commission on an order that never
    # filled, nothing else. Filed as long so the legs add up to the whole.
    legs["Long"] += pnl.where(side == 0, 0.0).sum(axis="columns")
    legs["Total"] = legs["Long"] + legs["Short"]

    portfolio = holdings.sum(axis="columns")
    start_value = portfolio / (1 + returns)
    exposure = pd.DataFrame({
        "Long": values.clip(lower=0).sum(axis="columns"),
        "Short": -values.clip(upper=0).sum(axis="columns"),
    })
    exposure["Total"] = exposure["Long"] + exposure["Short"]
    years = len(returns) / 252
    total = float(legs["Total"].sum())

    rows = {}
    for leg in ("Long", "Short", "Total"):
        net = float(legs[leg].sum())
        active = exposure[leg] > 0
        capital = float(exposure.loc[active, leg].mean()) if active.any() else 0.0
        rows[leg] = {
            "Net PnL": net,
            **({"Share of PnL": net / total} if total > 0 else {}),  # a share of a loss reads backwards
            "Contribution": float((legs[leg] / start_value).sum()),
            "Avg. Exposure": float((exposure[leg] / portfolio).where(active).mean()) if active.any() else np.nan,
            "Days Held": int(active.sum()),
            "Winning Days": float((legs.loc[active, leg] > 0).mean()) if active.any() else np.nan,
            "Return on Exposure, a Year": net / capital / years if capital and years else np.nan,
        }
    summary = pd.DataFrame(rows).T
    money_column = money_formatter()
    section = Section("Long and Short")
    section.table(Table(summary, None, None, {
        "Net PnL": money_column, "Share of PnL": "{:.0%}".format, "Contribution": "{:.2%}".format,
        "Avg. Exposure": "{:.0%}".format, "Days Held": "{:,.0f}".format, "Winning Days": "{:.0%}".format,
        "Return on Exposure, a Year": "{:.1%}".format,
    }), title="Each Side of the Book", first_header="Side")
    section.side.append(
        '<p class="explain">Each instrument\'s daily mark-to-market PnL, after commission, filed under '
        'the side it was held on that day -- long or short at the previous close, or the side a new '
        'position was opened on. <b>Avg. exposure</b> is the side\'s gross value over the portfolio\'s, on '
        'the days it held anything; <b>return on exposure</b> is its PnL over the average value it held, '
        'a year. A short side that loses money can still earn its place by cutting the drawdowns: '
        'compare with the Drawdowns and Market Regimes sections.</p>')

    def cumulative(ax):
        running = legs.cumsum()
        ax.plot(running.index, running["Long"], color=STRATEGY_COLOR, lw=1.4, label="Long")
        ax.plot(running.index, running["Short"], color=LOSS_COLOR, lw=1.4, label="Short")
        ax.plot(running.index, running["Total"], color=BENCHMARK_COLOR, lw=1.2, ls="--", label="Total")
        ax.axhline(0, color="black", lw=0.8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        ax.set_ylabel(f"Cumulative PnL, {unit()}")
        ax.legend(loc="upper left")

    section.chart(chart("Cumulative PnL by Side", cumulative, height=3.8))

    yearly = legs[["Long", "Short"]].groupby(legs.index.year).sum()
    if len(yearly) > 1:
        def by_year(ax):
            positions = np.arange(len(yearly))
            ax.bar(positions - 0.2, yearly["Long"], width=0.4, color=STRATEGY_COLOR, label="Long")
            ax.bar(positions + 0.2, yearly["Short"], width=0.4, color=LOSS_COLOR, label="Short")
            ax.set_xticks(positions)
            ax.set_xticklabels([str(year) for year in yearly.index])
            ax.axhline(0, color="black", lw=0.8)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
            ax.set_ylabel(f"PnL, {unit()}")
            ax.legend(loc="upper left")

        section.chart(chart("PnL by Side and Year", by_year, height=3.0))

    long_pnl, short_pnl = float(legs["Long"].sum()), float(legs["Short"].sum())
    value = f"{compact(long_pnl)} / {compact(short_pnl)}"
    winner, loser = (long_pnl, short_pnl) if long_pnl >= short_pnl else (short_pnl, long_pnl)
    names = ("short", "long") if long_pnl >= short_pnl else ("long", "short")
    if winner > 0 > loser:
        note = (f"The {names[0]} side lost {money(-loser, 0)}, giving back {-loser / winner:.0%} of what the "
                f"{names[1]} side made ({money(winner, 0)}).")
        section.finding(WARN if -loser >= GIVE_BACK * winner else OK, "Long / short PnL", value, note)
    else:
        section.finding(OK if loser >= 0 else WARN, "Long / short PnL", value,
                        "Both sides made money." if loser >= 0 else "Both sides lost money.")
    return section
