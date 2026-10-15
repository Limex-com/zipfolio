"""Allocation and profit by GICS sector, and a treemap of where the capital went and what it made."""

from __future__ import annotations

import html

import numpy as np
import pandas as pd

from ..currency import money, money_formatter
from ..page import OK, WARN, Section
from ..style import anchor_zero, chart, color_mix, held_only, mark_isolated
from ..sectors import UNCLASSIFIED
from ..tables import Table
from ..interactive import INTERACTIVE, PX_PER_INCH, native_chart
from .instruments import CONCENTRATION_ADVICE, commission_by_day, instrument_pnl

#: One sector holding more than this share of the gross book on a typical day needs a look.
SECTOR_SHARE = 0.4
#: The treemap's colours, after finviz's market map: a loss red, a gain green, about nothing dark grey.
#: Each is the colour at that share of the scale's bound (``tile_color``).
TILE_SCALE = [(-1.0, "#f63538"), (-0.66, "#bf4045"), (-0.33, "#8b444e"), (0.0, "#414554"),
              (0.33, "#35764e"), (0.66, "#2f9e4f"), (1.0, "#30cc5a")]
#: The darkest red and green are for this share of the names' contributions and beyond.
TILE_BOUND_QUANTILE = 80


def treemap_frame(returns: pd.Series, holdings: pd.DataFrame, transactions: pd.DataFrame,
                  orders: pd.DataFrame | None, sectors: dict[str, str]) -> pd.DataFrame:
    """One row per name the strategy held: its sector, how much capital it took, what it made.

    ``weight`` is the name's absolute share of the portfolio averaged over every
    session -- days it was not held count as nothing -- so the weights of all
    names add up to the average gross exposure. ``contribution`` is its PnL
    after commission over the portfolio's value at the start of each day, added
    up, as in Instruments.
    """
    pnl = instrument_pnl(holdings, transactions)
    pnl = pnl - commission_by_day(orders, pnl.index, pnl.columns)
    start_value = holdings.sum(axis="columns") / (1 + returns)
    book = holdings.sum(axis="columns")
    values = holdings.drop(columns="cash", errors="ignore").reindex(columns=pnl.columns, fill_value=0.0)
    fills = transactions.groupby(transactions["symbol"].astype(str)).size()
    frame = pd.DataFrame({
        "sector": [sectors.get(symbol, UNCLASSIFIED) for symbol in pnl.columns],
        "weight": values.abs().div(book.where(book > 0), axis="index").fillna(0.0).mean(),
        "contribution": pnl.div(start_value, axis="index").sum(),
        "pnl": pnl.sum(),
        "days": values.ne(0).sum(),
        "fills": fills.reindex(pnl.columns, fill_value=0),
    }, index=pnl.columns)
    return frame[frame["weight"] > 0]


def tile_color(value: float, bound: float) -> str:
    """The treemap colour of a contribution, on ``TILE_SCALE`` stretched to +-``bound``."""
    share = float(np.clip(value / bound, -1.0, 1.0)) if bound > 0 else 0.0
    for (low, low_color), (high, high_color) in zip(TILE_SCALE, TILE_SCALE[1:]):
        if share <= high:
            return color_mix(low_color, high_color, (share - low) / (high - low))
    return TILE_SCALE[-1][1]


def squarify(sizes: list[float], x: float, y: float, width: float, height: float) -> list[tuple]:
    """Rectangles for ``sizes`` (largest first) tiling ``(x, y, width, height)``, as square as can be.

    Bruls, Huizing and van Wijk's squarified treemap: rows are laid along the
    shorter side, and a row takes the next size only while that makes its
    worst aspect ratio no worse.
    """
    total = float(sum(sizes))
    if total <= 0:
        return []
    areas = [size * width * height / total for size in sizes]
    rectangles = []

    def worst(row, side):
        filled = sum(row)
        return max(max(side * side * area / (filled * filled), filled * filled / (side * side * area)) for area in row)

    while areas:
        side = min(width, height)
        row = [areas[0]]
        while len(row) < len(areas) and worst(row + [areas[len(row)]], side) <= worst(row, side):
            row.append(areas[len(row)])
        areas = areas[len(row):]
        filled = sum(row)
        if width >= height:
            column = filled / height
            top = y
            for area in row:
                rectangles.append((x, top, column, area / column))
                top += area / column
            x, width = x + column, width - column
        else:
            band = filled / width
            left = x
            for area in row:
                rectangles.append((left, y, area / band, band))
                left += area / band
            y, height = y + band, height - band
    return rectangles


def _tile_tip(symbol: str, row) -> str:
    return (f"<b>{html.escape(symbol)}</b> &middot; {html.escape(row.sector)}<br>"
            f"Contribution: <b>{row.contribution:+.2%}</b> of the return<br>"
            f"Net PnL: {html.escape(money(row.pnl, 0))}<br>"
            f"Average weight: {row.weight:.1%} of the portfolio<br>"
            f"Held {int(row.days):,} sessions &middot; {int(row.fills):,} fills")


def _treemap(section: Section, returns, holdings, transactions, orders, sectors) -> None:
    frame = treemap_frame(returns, holdings, transactions, orders, sectors)
    if len(frame) < 2:
        return
    magnitudes = frame["contribution"].abs()
    bound = float(np.percentile(magnitudes, TILE_BOUND_QUANTILE)) if len(frame) >= 5 else float(magnitudes.max())
    bound = max(bound, 1e-4)
    by_sector = frame.groupby("sector").agg(weight=("weight", "sum"), contribution=("contribution", "sum"))
    by_sector = by_sector.sort_values("weight", ascending=False)
    title = "Where the Capital Went and What It Made"
    height = 4.6

    if INTERACTIVE.get():
        data = []
        for sector, totals in by_sector.iterrows():
            names = frame[frame["sector"] == sector].sort_values("weight", ascending=False)
            data.append({
                "name": f"{sector}  {totals.contribution:+.1%}", "value": float(totals.weight),
                "tip": (f"<b>{html.escape(sector)}</b><br>Contribution: <b>{totals.contribution:+.2%}</b><br>"
                        f"Average weight: {totals.weight:.1%} &middot; {len(names)} "
                        f"name{'s' if len(names) != 1 else ''}"),
                "children": [{
                    "name": symbol, "value": float(row.weight), "tip": _tile_tip(symbol, row),
                    "itemStyle": {"color": tile_color(row.contribution, bound)},
                    "label": {"formatter": f"{symbol}\n{row.contribution:+.1%}"},
                } for symbol, row in names.iterrows()],
            })
        option = {"series": [{
            "type": "treemap", "name": "All sectors", "data": data, "top": 32, "left": 2, "right": 2, "bottom": 4,
            # No zoom on click: in a 620px column ECharts pans the map instead of opening the sector.
            "roam": False, "nodeClick": False, "squareRatio": 1.2, "breadcrumb": {"show": False},
            "label": {"show": True, "color": "#ffffff", "fontFamily": "Arial", "fontSize": 13, "fontWeight": "bold",
                      "lineHeight": 16, "overflow": "truncate"},
            "upperLabel": {"show": False, "height": 20, "color": "#ffffff", "fontFamily": "Arial", "fontSize": 11,
                           "fontWeight": "bold"},
            "itemStyle": {"borderColor": "#2d2d2d"},
            "levels": [
                {"itemStyle": {"borderColor": "#2d2d2d", "borderWidth": 0, "gapWidth": 3}},
                {"itemStyle": {"borderColor": "#2d2d2d", "color": "#2d2d2d", "borderWidth": 3, "gapWidth": 1},
                 "upperLabel": {"show": True}},
                {"itemStyle": {"borderColor": "#2d2d2d", "borderWidth": 0, "gapWidth": 1}},
            ],
        }]}
        section.chart(native_chart(option, int(height * PX_PER_INCH), title))
    else:
        section.chart(chart(title, lambda ax: _static_treemap(ax, frame, by_sector, bound), height=height))
    section.side.append(
        '<p class="caption">Each tile is a name, inside its sector. Its <b>size</b> is the capital it took -- its '
        'share of the portfolio averaged over the whole backtest -- and its <b>colour</b> what it added to the '
        f'return: red a loss, green a gain, darkest at &plusmn;{bound:.1%} and beyond. Hover a tile for its '
        'numbers.</p>')


def _static_treemap(ax, frame: pd.DataFrame, by_sector: pd.DataFrame, bound: float) -> None:
    """The same treemap in matplotlib, for a page with static charts."""
    from matplotlib.patches import Rectangle

    width, height = 100.0, 58.0
    for (sector, totals), (x, y, w, h) in zip(by_sector.iterrows(),
                                               squarify(list(by_sector["weight"]), 0, 0, width, height)):
        header = min(3.0, h * 0.25)
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#2d2d2d", edgecolor="#ffffff", linewidth=1.5))
        if w > 12:
            ax.text(x + 0.8, y + header / 2, f"{sector}  {totals.contribution:+.1%}", color="white", fontsize=8,
                    fontweight="bold", va="center", ha="left", clip_on=True)
        names = frame[frame["sector"] == sector].sort_values("weight", ascending=False)
        inner = squarify(list(names["weight"]), x + 0.4, y + header, w - 0.8, h - header - 0.4)
        for (symbol, row), (tx, ty, tw, th) in zip(names.iterrows(), inner):
            ax.add_patch(Rectangle((tx, ty), tw, th, facecolor=tile_color(row.contribution, bound),
                                   edgecolor="#2d2d2d", linewidth=0.8))
            if tw > 5 and th > 4:
                ax.text(tx + tw / 2, ty + th / 2, f"{symbol}\n{row.contribution:+.1%}", color="white",
                        fontsize=8 if tw < 12 else 10, fontweight="bold", ha="center", va="center")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")


def sector_concentration(positions: pd.DataFrame, sectors: dict[str, str]) -> pd.DataFrame:
    """Each sector's share of the gross book (longs and shorts both counted), per session held."""
    values = positions.drop(columns="cash", errors="ignore").abs()
    by_sector = values.T.groupby(lambda symbol: sectors.get(symbol, UNCLASSIFIED)).sum().T
    gross = by_sector.sum(axis="columns")
    return by_sector.div(gross.where(gross > 0), axis="index").dropna(how="all")


def sector_profit(returns, holdings, transactions, orders, sectors) -> pd.DataFrame:
    """Net PnL and contribution per sector, every traded name in one -- ``UNCLASSIFIED`` when it has none.

    Counted as Instruments counts it: daily mark-to-market PnL after commission, open positions at the last
    price; the sectors add up to the instruments, so a name held only within a day, or one with no sector,
    is not dropped.
    """
    pnl = instrument_pnl(holdings, transactions)
    pnl = pnl - commission_by_day(orders, pnl.index, pnl.columns)
    start_value = holdings.sum(axis="columns") / (1 + returns)
    frame = pd.DataFrame({"pnl": pnl.sum(), "contribution": pnl.div(start_value, axis="index").sum()})
    frame["sector"] = [sectors.get(symbol, UNCLASSIFIED) for symbol in frame.index]
    return frame.groupby("sector")[["pnl", "contribution"]].sum().sort_values("contribution", ascending=False)


def _profit_by_sector(section: Section, returns, holdings, transactions, orders, sectors) -> None:
    """What each sector made, the way Instruments counts it (``sector_profit``), names with no sector included.

    pyfolio's table is each sector's share of the closed trades' PnL; when the
    total is a loss, a sector that made money shows a negative share.
    """
    by_sector = sector_profit(returns, holdings, transactions, orders, sectors)
    if by_sector.empty:
        return
    table = pd.DataFrame({"Net PnL": by_sector["pnl"], "Contribution": by_sector["contribution"]})
    total = float(by_sector["pnl"].sum())
    if total > 0:
        table["Share of PnL"] = by_sector["pnl"] / total
    section.table(Table(table, None, None, {"Net PnL": money_formatter(), "Contribution": "{:.2%}".format,
                                            "Share of PnL": "{:.1%}".format}),
                  title="Profit by Sector", first_header="Sector")


def build_sector_analysis(returns, positions, transactions, sectors: dict[str, str], holdings=None,
                          orders=None) -> Section:
    """Allocation by sector over time, its average and peak, the treemap, and what each sector made.

    Built from the pieces ``create_position_tear_sheet`` draws when handed
    ``sector_mappings``, so the sector view has a section of its own rather
    than a last panel in two; the profit by sector is counted as the treemap
    counts it, not from pyfolio's round trips.
    """
    from pyfolio import plotting, pos

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
                        "look.", key="sector", threshold=f"at most {SECTOR_SHARE:.0%} of the book",
                        issue=f"{typical:.0%} of the book sits in one sector, {leader}", advice=CONCENTRATION_ADVICE)
    _treemap(section, returns, holdings if holdings is not None else positions, transactions, orders, sectors)
    held = positions.drop(columns="cash", errors="ignore")
    held = [symbol for symbol in held.columns if held[symbol].abs().sum() > 0]
    classified = {sectors.get(symbol, UNCLASSIFIED) for symbol in held} - {UNCLASSIFIED}
    if len(classified) < 2:
        only = next(iter(classified), UNCLASSIFIED)
        section.side.append(
            f'<p class="explain">Every name the strategy held is in one sector, {only}: there is no split by '
            'sector to show. Names in one sector tend to fall together, so the book is one bet held '
            f'{len(held)} times.</p>')
        return section
    exposures = pos.get_sector_exposures(positions, sectors)
    allocation = pos.get_percent_alloc(exposures).drop("cash", axis="columns")
    average = allocation.mean().sort_values(ascending=False)
    section.table(
        Table(pd.DataFrame({"Average": average, "Max": allocation.max()[average.index]}),
              None, "{:.2%}".format),
        title="Sector Allocation", first_header="Sector")

    if holdings is not None:
        _profit_by_sector(section, returns, holdings, transactions, orders, sectors)

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
