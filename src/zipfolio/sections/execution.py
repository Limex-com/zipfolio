"""Execution: how the backtest filled its orders, and how far to believe those fills.

A backtest is only as good as its fills. This section checks every fill
against the bar it happened on -- was the price inside the bar's range, how
much of the bar's volume did it take -- and every order against the moment it
was placed: how long it waited, whether it filled in full, what it cost over
the price the strategy saw when it decided.
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd

from ..page import Section
from ..style import BENCHMARK_COLOR, LOSS_COLOR, STRATEGY_COLOR, chart
from ..tables import Table

#: A fill taking more than this share of its bar's volume would move the price
#: in a real market; the fixed-basis-point model charges the same however large.
IMPACT_PARTICIPATION = 0.01

ORDER_TYPE_NAMES = {"market": "Market", "limit": "Limit", "stop": "Stop", "stop_limit": "Stop-limit"}

OK, WARN, BAD, INFO = "ok", "warn", "bad", "info"
_MARKS = {OK: "&#10003;", WARN: "!", BAD: "&#10007;", INFO: "i"}


def fill_quality(transactions: pd.DataFrame, fill_bars: pd.DataFrame | None,
                 decision_prices: pd.DataFrame | None, orders: pd.DataFrame | None,
                 reference: str = "close") -> pd.DataFrame:
    """One row per fill: its bar, where it priced inside it, its share of volume, its shortfall.

    ``shortfall_bps`` is what the fill cost over the close of the bar the order
    was placed on -- the price the strategy saw when it decided -- signed so
    that a cost is positive for buys and sells alike. It splits into
    ``delay_bps``, the move from that close to the fill bar's ``reference``
    price, and ``slippage_bps``, the fill's distance from that price.
    """
    fills = pd.DataFrame({
        "id": transactions["id"].astype(str),
        "order_id": transactions["order_id"].astype(str),
        "dt": pd.to_datetime(transactions["dt"], utc=True),
        "symbol": transactions["symbol"].astype(str),
        "amount": pd.to_numeric(transactions["quantity"], errors="coerce"),
        "price": pd.to_numeric(transactions["price"], errors="coerce"),
    }).dropna(subset=["amount", "price"])
    fills = fills[fills["amount"] != 0]
    fills["side"] = np.sign(fills["amount"])
    fills["value"] = (fills["amount"] * fills["price"]).abs()

    if fill_bars is not None and not fill_bars.empty:
        bars = fill_bars.copy()
        bars.index = bars.index.astype(str)
        fills = fills.join(bars.add_prefix("bar_"), on="id")
        span = fills["bar_high"] - fills["bar_low"]
        tolerance = 1e-9 * fills["price"].abs()
        fills["above_high"] = fills["price"] > fills["bar_high"] + tolerance
        fills["below_low"] = fills["price"] < fills["bar_low"] - tolerance
        fills["in_range"] = (fills["price"] - fills["bar_low"]) / span.where(span > 0)
        volume = fills["bar_volume"].where(fills["bar_volume"] > 0)
        fills["participation"] = fills["amount"].abs() / volume
        fills["zero_volume"] = fills["bar_volume"].fillna(0) <= 0
        typical = (fills["bar_high"] + fills["bar_low"] + fills["bar_close"]) / 3
        fills["typical_bps"] = fills["side"] * (fills["price"] - typical) / typical * 1e4

    if orders is not None and not orders.empty and decision_prices is not None and not decision_prices.empty:
        decided = decision_prices["decision_price"].copy()
        decided.index = decided.index.astype(str)
        fills["decision_price"] = fills["order_id"].map(decided)
        basis = fills["decision_price"]
        fills["shortfall_bps"] = fills["side"] * (fills["price"] - basis) / basis * 1e4
        reference_price = fills.get(f"bar_{reference}")
        if reference_price is not None:
            fills["delay_bps"] = fills["side"] * (reference_price - basis) / basis * 1e4
            fills["slippage_bps"] = fills["shortfall_bps"] - fills["delay_bps"]
    return fills


def order_book(orders: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
    """One row per order: type, size, what filled, over how many fills, and how long it waited."""
    final = orders.drop_duplicates("id", keep="last").copy()
    first = orders.drop_duplicates("id", keep="first").set_index("id")
    final["created"] = pd.to_datetime(final["id"].map(first["created"] if "created" in first else first["dt"]),
                                      utc=True)
    final["id"] = final["id"].astype(str)
    per_order = fills.groupby("order_id").agg(fills=("id", "size"), first_fill=("dt", "min"),
                                               last_fill=("dt", "max"))
    final = final.join(per_order, on="id")
    final["fills"] = final["fills"].fillna(0).astype(int)
    final["quantity"] = pd.to_numeric(final.get("quantity", final.get("filled")), errors="coerce")
    final["filled"] = pd.to_numeric(final["filled"], errors="coerce").fillna(0)
    final["completion"] = (final["filled"].abs() / final["quantity"].abs()).where(final["quantity"] != 0)
    final["delay"] = final["first_fill"] - final["created"]
    final["type"] = final.get("type", pd.Series("market", index=final.index)).fillna("market")
    final["state"] = np.select(
        [final["completion"] >= 1 - 1e-9, final["filled"].abs() > 0],
        ["filled", "partial"], default="unfilled")
    return final


def build_execution(transactions: pd.DataFrame, orders: pd.DataFrame | None,
                    fill_bars: pd.DataFrame | None, decision_prices: pd.DataFrame | None,
                    settings: dict | None) -> Section | None:
    if orders is None or orders.empty or transactions is None or transactions.empty:
        return None
    settings = settings or {}
    reference = settings.get("price_used_in_order_execution") or "close"
    fills = fill_quality(transactions, fill_bars, decision_prices, orders, reference)
    book = order_book(orders, fills)
    # An order for zero shares -- a rebalance to the weight already held --
    # is recorded as filled with nothing to fill. It is neither a fill nor a
    # failure, so it is counted apart.
    empty = int((book["quantity"] == 0).sum())
    book = book[book["quantity"] != 0]
    section = Section("Execution")

    section.side.append("<h3>How the Backtest Fills Orders</h3>" + "".join(
        f'<p class="explain">{line}</p>' for line in _mechanics_text(settings)))
    checks = _checks(fills, book, settings)
    if empty:
        checks.append((INFO, "Orders for zero shares",
                       f"{empty:,} orders asked for no shares -- a rebalance to a weight already held. "
                       "They are left out of every count here."))
    section.side.append("<h3>Realism Checks</h3>" + _checks_html(checks))

    types = book.groupby("type")["state"].value_counts().unstack(fill_value=0)
    for state in ("filled", "partial", "unfilled"):
        if state not in types.columns:
            types[state] = 0
    types = types[["filled", "partial", "unfilled"]]
    types.insert(0, "orders", types.sum(axis="columns"))
    types.index = [ORDER_TYPE_NAMES.get(name, str(name).title()) for name in types.index]
    section.table(Table(types.rename(columns=str.title), None, None,
                        {column: "{:,.0f}".format for column in ("orders", "filled", "partial", "unfilled")}),
                  title="Orders by Type", first_header="Type")

    section.table(Table(_statistics(fills, book).to_frame("Value"), None), title="Fill Statistics",
                  first_header="Metric")

    failed = book[book["state"] != "filled"]
    if len(failed):
        reasons = failed.assign(
            status=failed.get("status", pd.Series("-", index=failed.index)).astype(str)
            .str.replace("OrderStatus.", "", regex=False).str.title(),
            reason=failed.get("reason", pd.Series(None, index=failed.index)).fillna("-").astype(str),
        ).groupby(["status", "reason"]).size().rename("Orders").reset_index()
        reasons.index = reasons.pop("status")
        section.table(Table(reasons, None, None, {"Orders": "{:,.0f}".format}),
                      title="Orders Not Filled in Full", first_header="Status")

    if "in_range" in fills:
        section.chart(chart("Where Fills Priced Within Their Bar", lambda ax: _range_histogram(ax, fills),
                            height=3.2))
    if "typical_bps" in fills and fills["typical_bps"].notna().any():
        section.chart(chart("Fill Price vs the Bar's Typical Price", lambda ax: _typical_histogram(ax, fills),
                            height=3.2))
    if "participation" in fills and fills["participation"].notna().any():
        limit = settings.get("slippage_volume_limit")
        section.chart(chart("Fill Size as a Share of Bar Volume", lambda ax: _participation_histogram(
            ax, fills, limit), height=3.2))
        section.chart(chart("Participation Over Time", lambda ax: _participation_over_time(ax, fills, limit),
                            height=3.2))
    shortfall = fills["shortfall_bps"].dropna() if "shortfall_bps" in fills else pd.Series(dtype=float)
    if len(shortfall) and shortfall.max() - shortfall.min() > 0.5:
        # Every fill at the same cost -- same-bar fills at a fixed slippage --
        # makes a histogram of one bar; the statistics table says it better.
        section.chart(chart("Implementation Shortfall per Fill", lambda ax: _shortfall_histogram(ax, fills),
                            height=3.2))
    waited = book["delay"].dropna()
    if len(waited) and waited.max() > pd.Timedelta(0):
        section.chart(chart("Order-to-Fill Delay", lambda ax: _delay_histogram(ax, waited), height=3.0))
    return section


def _mechanics_text(settings: dict) -> list[str]:
    reference = settings.get("price_used_in_order_execution") or "close"
    bps = float(settings.get("slippage_basis_points") or 0.0)
    limit = settings.get("slippage_volume_limit")
    same_bar = settings.get("same_bar_execution", True)
    timing = (f"on the bar it is placed in, at that bar's {reference}" if same_bar
              else f"on the next bar, at that bar's {reference}")
    lines = [
        f"A <b>market order</b> fills {timing}, moved {bps:g} bps against the strategy for slippage. "
        + (f"A single bar fills at most {float(limit):.0%} of its volume; the rest of the order waits "
           "for the next bars, at their prices." if limit else ""),
        "A <b>limit order</b> fills only on a bar that trades through its limit, and never at a worse "
        "price; a <b>stop</b> becomes a market order once a bar trades through the stop. Orders stay "
        "open until filled or cancelled.",
        "Not simulated: the bid-ask spread (other than inside the fixed basis points), the order "
        "book and queue position, partial liquidity inside a bar, and the price impact of the "
        "strategy's own size -- which is why the checks below compare every fill with its bar.",
    ]
    return lines


def _checks(fills: pd.DataFrame, book: pd.DataFrame, settings: dict) -> list[tuple[str, str, str]]:
    checks = []
    reference = settings.get("price_used_in_order_execution") or "close"
    same_bar = settings.get("same_bar_execution", True)
    if same_bar and reference == "close":
        checks.append((WARN, "Fills at the close of the signal bar",
                       "Each order fills at the close of the bar whose data produced it, so the strategy "
                       "trades at a price it has only just seen. That holds only for market-on-close "
                       "orders sent before the close; next-bar execution is the conservative setting."))
    elif same_bar:
        checks.append((BAD, f"Fills at the {reference} of the signal bar",
                       f"An order placed on a bar's data fills at that bar's {reference}, which came "
                       "before the data the order was based on: look-ahead. Use next-bar execution."))
    else:
        checks.append((OK, "Fills on the bar after the signal",
                       f"Orders fill on the next bar, at its {reference}: no price is used before it existed."))
    if reference in ("high", "low"):
        checks.append((BAD, f"Fill price is the bar's {reference}",
                       f"Every fill takes the bar's {reference} -- the best price of the bar for one side "
                       "and the worst for the other. No order can choose that price."))

    if "above_high" in fills:
        outside = int((fills["above_high"] | fills["below_low"]).sum())
        share = outside / len(fills) if len(fills) else 0.0
        above, below = int(fills["above_high"].sum()), int(fills["below_low"].sum())
        detail = (f"{outside:,} of {len(fills):,} fills ({share:.1%}) priced outside their bar's "
                  f"high-low range ({above:,} above the high, {below:,} below the low). "
                  if outside else f"All {len(fills):,} fills priced inside their bar's high-low range. ")
        if outside:
            detail += ("Typically slippage pushing a buy past a bar that closed at its high: the "
                       "market never traded at that price, so neither could the strategy.")
        checks.append((OK if not outside else WARN if share < 0.05 else BAD, "Fill prices inside the bar", detail))

    if "participation" in fills and fills["participation"].notna().any():
        participation = fills["participation"].dropna()
        limit = float(settings.get("slippage_volume_limit") or 1.0)
        zero = int(fills["zero_volume"].sum())
        worst, typical = participation.max(), participation.median()
        heavy = int((participation > IMPACT_PARTICIPATION).sum())
        status = OK if worst <= IMPACT_PARTICIPATION and not zero else BAD if worst > limit or zero else WARN
        detail = (f"The median fill took {typical:.3%} of its bar's volume, the largest {worst:.2%}; "
                  f"{heavy:,} fills took over {IMPACT_PARTICIPATION:.0%}. ")
        if zero:
            detail += f"{zero:,} fills happened on bars with no volume at all. "
        detail += ("Above about 1% a real order moves the price; the model charges the same fixed "
                   "basis points whatever the size." if heavy else "Small enough not to move the price.")
        checks.append((status, "Size against liquidity", detail))

    incomplete = book[book["state"] != "filled"]
    if len(incomplete):
        partial = int((incomplete["state"] == "partial").sum())
        checks.append((WARN, "Orders not filled in full",
                       f"{len(incomplete):,} of {len(book):,} orders did not fill in full ({partial:,} "
                       "partly) -- cancelled, rejected, or still open at the end. The strategy's intended "
                       "positions and its actual ones differ by these."))
    else:
        checks.append((OK, "Every order filled in full", f"All {len(book):,} orders filled completely."))

    slow = int((book["fills"] > 1).sum())
    if slow:
        checks.append((INFO, "Orders filled over several bars",
                       f"{slow:,} orders filled in more than one piece, each at its own bar's price -- "
                       "the volume limit at work."))
    types = sorted(set(book["type"]))
    if types == ["market"]:
        checks.append((INFO, "Market orders only",
                       "Every order was a market order: the backtest assumes immediate execution at the "
                       "bar's price plus fixed slippage, with no protection from a bad print."))
    if "limit" in book.columns and (book["type"] == "limit").any() and "decision_price" in fills:
        limits = book.set_index("id")["limit"]
        priced = fills.assign(limit=fills["order_id"].map(limits)).dropna(subset=["limit"])
        breach = priced[((priced["side"] > 0) & (priced["price"] > priced["limit"] * (1 + 1e-9)))
                        | ((priced["side"] < 0) & (priced["price"] < priced["limit"] * (1 - 1e-9)))]
        checks.append((OK if breach.empty else BAD, "Limit prices respected",
                       "No limit order filled at a worse price than its limit." if breach.empty else
                       f"{len(breach):,} limit-order fills priced through their limit."))
    return checks


def _checks_html(checks) -> str:
    rows = []
    for status, title, detail in checks:
        rows.append(f'<div class="check"><span class="mark {status}">{_MARKS[status]}</span>'
                    f'<div><b>{html.escape(title)}</b><br>{html.escape(detail)}</div></div>')
    return "".join(rows)


def _statistics(fills: pd.DataFrame, book: pd.DataFrame) -> pd.Series:
    rows = {
        "Orders": f"{len(book):,}",
        "Fills": f"{len(fills):,}",
        "Fills per order": f"{len(fills) / len(book):.2f}" if len(book) else "-",
        "Traded value": f"${fills['value'].sum():,.0f}",
    }
    waited = book["delay"].dropna()
    if len(waited):
        rows["Median wait, order to first fill"] = _duration(waited.median())
        rows["Longest wait"] = _duration(waited.max())
    if "participation" in fills:
        participation = fills["participation"].dropna()
        if len(participation):
            rows["Median share of bar volume"] = f"{participation.median():.3%}"
            rows["95th percentile share"] = f"{participation.quantile(0.95):.3%}"
            rows["Largest share"] = f"{participation.max():.2%}"
    if "shortfall_bps" in fills:
        weighted = fills.dropna(subset=["shortfall_bps"])
        if len(weighted):
            weights = weighted["value"]

            def average(column):
                return float(np.average(weighted[column], weights=weights))

            rows["Implementation shortfall (value-weighted)"] = f"{average('shortfall_bps'):.2f} bps"
            if "delay_bps" in weighted:
                rows["  of which delay"] = f"{average('delay_bps'):.2f} bps"
                rows["  of which slippage"] = f"{average('slippage_bps'):.2f} bps"
            if "typical_bps" in weighted:
                rows["Cost vs bar typical price (value-weighted)"] = f"{average('typical_bps'):.2f} bps"
    return pd.Series(rows)


def _duration(delta: pd.Timedelta) -> str:
    if delta <= pd.Timedelta(0):
        return "same bar"
    if delta < pd.Timedelta(hours=1):
        return f"{delta.total_seconds() / 60:.0f} min"
    if delta < pd.Timedelta(days=1):
        return f"{delta.total_seconds() / 3600:.1f} h"
    return f"{delta.total_seconds() / 86400:.1f} days"


def _range_histogram(ax, fills: pd.DataFrame) -> None:
    clipped = fills["in_range"].clip(-0.25, 1.25)
    bins = np.linspace(-0.25, 1.25, 31)
    ax.axvspan(-0.25, 0, color=LOSS_COLOR, alpha=0.08)
    ax.axvspan(1, 1.25, color=LOSS_COLOR, alpha=0.08)
    for side, label, color in ((1, "Buys", STRATEGY_COLOR), (-1, "Sells", BENCHMARK_COLOR)):
        values = clipped[fills["side"] == side].dropna()
        if len(values):
            ax.hist(values, bins=bins, color=color, alpha=0.7, label=label)
    for edge in (0, 1):
        ax.axvline(edge, color="black", lw=1, ls="--")
    ax.set_xticks([-0.25, 0, 0.25, 0.5, 0.75, 1, 1.25])
    ax.set_xticklabels(["below low", "low", "25%", "50%", "75%", "high", "above high"])
    ax.set_ylabel("Fills")
    ax.legend(loc="upper left")


def _typical_histogram(ax, fills: pd.DataFrame) -> None:
    """Each fill against its bar's typical price (high + low + close) / 3, signed as a cost.

    The typical price stands in for the bar's VWAP, the usual yardstick of an
    execution desk: a buy above it, or a sell below it, did worse than the
    bar's average trade.
    """
    data = fills["typical_bps"].dropna()
    low, high = np.nanpercentile(data, [1, 99]) if len(data) > 20 else (data.min(), data.max())
    bins = np.linspace(low, high, 40) if high > low else 20
    for side, label, color in ((1, "Buys", STRATEGY_COLOR), (-1, "Sells", BENCHMARK_COLOR)):
        values = fills.loc[fills["side"] == side, "typical_bps"].dropna().clip(low, high)
        if len(values):
            ax.hist(values, bins=bins, color=color, alpha=0.7, label=label)
            ax.axvline(values.mean(), color=color, lw=1.6)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("bps over (high + low + close) / 3; positive is worse")
    ax.set_ylabel("Fills")
    ax.legend(loc="upper right")


def _participation_histogram(ax, fills: pd.DataFrame, limit) -> None:
    from matplotlib.ticker import FuncFormatter

    share = fills["participation"].dropna()
    share = share[share > 0]
    if share.empty:
        return
    bins = np.logspace(np.log10(share.min()), np.log10(max(share.max(), share.min() * 10)), 40)
    ax.hist(share, bins=bins, color=STRATEGY_COLOR, alpha=0.8)
    ax.set_xscale("log")
    ax.axvline(IMPACT_PARTICIPATION, color=BENCHMARK_COLOR, ls="--", lw=1.2, label="1%: price impact")
    if limit:
        ax.axvline(float(limit), color=LOSS_COLOR, ls="--", lw=1.2, label=f"{float(limit):.0%}: volume limit")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.3g}%"))
    ax.set_ylabel("Fills")
    ax.legend(loc="upper right")


def _participation_over_time(ax, fills: pd.DataFrame, limit) -> None:
    from matplotlib.ticker import FuncFormatter

    data = fills.dropna(subset=["participation"])
    data = data[data["participation"] > 0]
    for side, label, color in ((1, "Buys", STRATEGY_COLOR), (-1, "Sells", LOSS_COLOR)):
        part = data[data["side"] == side]
        ax.scatter(part["dt"].dt.tz_convert(None), part["participation"], s=10, color=color, alpha=0.6,
                   label=label, linewidths=0)
    ax.set_yscale("log")
    ax.axhline(IMPACT_PARTICIPATION, color=BENCHMARK_COLOR, ls="--", lw=1.2)
    if limit:
        ax.axhline(float(limit), color=LOSS_COLOR, ls="--", lw=1.2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.3g}%"))
    ax.set_ylabel("Share of bar volume")
    ax.legend(loc="upper left")


def _shortfall_histogram(ax, fills: pd.DataFrame) -> None:
    data = fills["shortfall_bps"].dropna()
    low, high = np.nanpercentile(data, [1, 99]) if len(data) > 20 else (data.min(), data.max())
    bins = np.linspace(low, high, 40) if high > low else 20
    for side, label, color in ((1, "Buys", STRATEGY_COLOR), (-1, "Sells", BENCHMARK_COLOR)):
        values = fills.loc[fills["side"] == side, "shortfall_bps"].dropna().clip(low, high)
        if len(values):
            ax.hist(values, bins=bins, color=color, alpha=0.7, label=label)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("Cost over the decision price, bps (positive = paid more than it saw)")
    ax.set_ylabel("Fills")
    ax.legend(loc="upper right")


def _delay_histogram(ax, waited: pd.Series) -> None:
    hours = waited.dt.total_seconds() / 3600
    unit, values = ("days", hours / 24) if hours.max() > 48 else ("hours", hours)
    ax.hist(values, bins=30, color=STRATEGY_COLOR, alpha=0.8)
    ax.set_xlabel(f"Wait from order to first fill, {unit}")
    ax.set_ylabel("Orders")
