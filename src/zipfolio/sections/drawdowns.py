"""Who made the drawdowns: each of the worst drawdowns split across the names that caused it."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..page import Section
from ..style import BENCHMARK_COLOR, PALETTE, chart
from ..tables import Table
from .instruments import commission_by_day, instrument_pnl

TOP_DRAWDOWNS = 5
#: Names drawn in their own colour; the rest are one "other names" segment.
NAMED = 8
OTHER = "Other names"
RESIDUAL = "Dividends, cash, rounding"


def drawdown_attribution(returns: pd.Series, holdings: pd.DataFrame, transactions: pd.DataFrame,
                         orders: pd.DataFrame | None = None, top: int = TOP_DRAWDOWNS,
                         sectors: dict[str, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The worst drawdowns, and how many points of each every name (or sector) accounts for.

    Exact, not estimated: from the peak's close to the valley's close the book
    lost ``V(valley) - V(peak)``, and that loss is the sum of each name's
    mark-to-market PnL over the same days (see ``instrument_pnl``), less
    commission. Divided by ``V(peak)`` each name's share is in points of the
    drawdown and the shares add up to its depth; what is left -- dividends
    paid into cash, rounding -- is shown as its own residual.

    Returns ``(drawdowns, shares)``: one row per drawdown with its dates and
    depth, and one row per drawdown with a column per name (or sector).
    """
    from pyfolio import timeseries

    table = timeseries.gen_drawdown_table(returns, top=top).dropna(subset=["Peak date", "Valley date"])
    pnl = instrument_pnl(holdings, transactions)
    pnl = pnl - commission_by_day(orders, pnl.index, pnl.columns)
    if sectors:
        pnl = pnl.T.groupby(lambda symbol: sectors.get(symbol, "Unclassified")).sum().T
    value = holdings.sum(axis="columns")

    rows, shares = [], []
    for _, drawdown in table.iterrows():
        peak, valley = (pd.Timestamp(drawdown[column]).tz_localize("UTC")
                        for column in ("Peak date", "Valley date"))
        if peak not in value.index or valley not in value.index or value[peak] <= 0:
            continue
        window = pnl.loc[(pnl.index > peak) & (pnl.index <= valley)].sum() / value[peak]
        depth = value[valley] / value[peak] - 1
        window[RESIDUAL] = depth - window.sum()
        recovered = drawdown["Recovery date"]
        rows.append({
            "Peak": peak, "Valley": valley,
            "Recovered": pd.Timestamp(recovered).tz_localize("UTC") if pd.notna(recovered) else pd.NaT,
            "Depth": depth,
            "Days": (valley - peak).days,
        })
        shares.append(window)
    return pd.DataFrame(rows), pd.DataFrame(shares).fillna(0.0)


def build_drawdowns(returns, holdings, transactions, orders=None, sectors=None) -> Section | None:
    drawdowns, shares = drawdown_attribution(returns, holdings, transactions, orders)
    if drawdowns.empty:
        return None
    section = Section("Drawdowns")
    labels = [f"#{n + 1}  {row.Peak:%b %Y} - {row.Valley:%b %Y}  ({row.Depth:.1%})"
              for n, row in enumerate(drawdowns.itertuples())]

    names = shares.drop(columns=RESIDUAL)
    summary = drawdowns.copy()
    for rank, title in ((0, "Largest Drag"), (1, "Second"), (2, "Third")):
        summary[title] = [_nth(row, rank, ascending=True) for _, row in names.iterrows()]
    summary["Best Offset"] = [_nth(row, 0, ascending=False) for _, row in names.iterrows()]
    summary.index = [f"#{n + 1}" for n in range(len(summary))]
    section.table(Table(summary, None, None, {"Depth": "{:.2%}".format, "Days": "{:,.0f}".format}),
                  title="Worst Drawdowns and Who Made Them", first_header="#")

    section.side.append(
        '<p class="explain">Each drawdown -- from the close of its peak to the close of its '
        'valley -- split across the names that made it. A name\'s share is its mark-to-market '
        'PnL over those days, after commission, over the portfolio\'s value at the peak, so the '
        'shares are points of the drawdown and add up to its depth. What remains, mostly '
        'dividends paid into cash, is shown on its own.</p>')
    worst = shares.iloc[0].drop(RESIDUAL)
    worst = worst[worst.abs() > 1e-9].sort_values()  # names not held then contributed nothing
    section.table(Table(worst.head(10).to_frame("Points of the Drawdown"), None, "{:.2%}".format),
                  title=f"Drawdown #1, by Name ({drawdowns.iloc[0]['Depth']:.1%})", first_header="Symbol")

    section.chart(chart("Who Made the Drawdowns", lambda ax: _stacked(ax, shares, labels),
                        height=max(2.6, 0.55 * len(labels) + 1.6), y=None))

    if sectors:
        _, by_sector = drawdown_attribution(returns, holdings, transactions, orders, sectors=sectors)
        section.chart(chart("Drawdowns by Sector", lambda ax: _stacked(ax, by_sector, labels),
                            height=max(2.6, 0.55 * len(labels) + 1.6)))
        section.side.append(
            '<p class="caption">The same split by sector: whether a drawdown was one bet held '
            'several times, or losses spread across the book.</p>')
    return section


def _nth(row: pd.Series, rank: int, ascending: bool) -> str:
    ordered = row.sort_values(ascending=ascending)
    if len(ordered) <= rank:
        return "-"
    name, value = ordered.index[rank], ordered.iloc[rank]
    if (ascending and value >= 0) or (not ascending and value <= 0):
        return "-"
    return f"{name} {value * 100:+.1f} pt"


def _stacked(ax, shares: pd.DataFrame, labels: list[str]) -> None:
    """One bar per drawdown: losses stacked left of zero, offsets right of it, depth marked."""
    from matplotlib.ticker import FuncFormatter

    names = shares.drop(columns=RESIDUAL)
    named = names.abs().sum().sort_values(ascending=False).index[:NAMED]
    parts = names[named].copy()
    rest = [column for column in names.columns if column not in named]
    if rest:
        parts[OTHER] = names[rest].sum(axis="columns")
    parts[RESIDUAL] = shares[RESIDUAL]
    colors = dict(zip(named, PALETTE[2:] + PALETTE[:2]))
    colors[OTHER] = "#bbbbbb"
    colors[RESIDUAL] = "#e5e5e5"

    rows = np.arange(len(labels))[::-1]
    left_neg = np.zeros(len(labels))
    left_pos = np.zeros(len(labels))
    for column in parts.columns:
        values = parts[column].to_numpy()
        start = np.where(values < 0, left_neg, left_pos)
        ax.barh(rows, values, left=start, height=0.6, color=colors.get(column, BENCHMARK_COLOR),
                label=column, edgecolor="white", linewidth=0.5)
        left_neg += np.where(values < 0, values, 0)
        left_pos += np.where(values >= 0, values, 0)
    ax.scatter(shares.sum(axis="columns"), rows, marker="D", color="black", zorder=3, s=26, label="Depth")
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(rows)
    ax.set_yticklabels(labels)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.set_xlabel("Points of the drawdown (losses left, offsets right)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)
