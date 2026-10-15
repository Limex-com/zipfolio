"""Next-bar fills: the same trades, had every order waited for the next bar's open.

A backtest that fills an order on the bar whose data produced it trades at a
price it has only just seen -- the platform's default, and the most flattering
setting there is. This section prices every such fill again at the open of the
instrument's following bar, the earliest a real order sent on that data could
trade, and shows what is left of the return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..currency import money, unit
from ..inputs import utc_dates
from ..page import BAD, OK, WARN, Section
from ..style import BENCHMARK_COLOR, LOSS_COLOR, STRATEGY_COLOR, chart
from ..tables import Table

#: Losing this share of the annual return to a bar's wait is worth a warning; this much more, a failure.
WARN_SHARE, BAD_SHARE = 0.2, 0.5


def repriced_fills(transactions: pd.DataFrame, fill_bars: pd.DataFrame, reference: str = "close") -> pd.DataFrame:
    """One row per fill, priced again at the open of the bar after its own.

    The fill keeps its quantity and its slippage: a buy at ``reference`` x
    (1 + b) becomes one at ``next_open`` x (1 + b). After that open the two
    books hold the same shares, so the whole difference is the price paid::

        impact = -quantity x price x (next_open / reference - 1)

    in money, positive when waiting would have paid off. ``move_bps`` is the
    same move signed as a cost: positive when the next open was worse for the
    fill's side -- higher for a buy, lower for a sell.
    """
    fills = pd.DataFrame({
        "id": transactions["id"].astype(str),
        "dt": pd.to_datetime(transactions["dt"], utc=True),
        "symbol": transactions["symbol"].astype(str),
        "amount": pd.to_numeric(transactions["quantity"], errors="coerce"),
        "price": pd.to_numeric(transactions["price"], errors="coerce"),
    }).dropna(subset=["amount", "price"])
    fills = fills[fills["amount"] != 0]
    bars = fill_bars.copy()
    bars.index = bars.index.astype(str)
    fills["reference"] = fills["id"].map(pd.to_numeric(bars[reference], errors="coerce"))
    fills["next_open"] = fills["id"].map(pd.to_numeric(bars["next_open"], errors="coerce"))
    fills["next_date"] = pd.to_datetime(fills["id"].map(bars["next_date"]), utc=True)
    change = fills["next_open"] / fills["reference"].where(fills["reference"] > 0) - 1
    fills["move_bps"] = np.sign(fills["amount"]) * change * 1e4
    fills["impact"] = -fills["amount"] * fills["price"] * change
    fills["value"] = (fills["amount"] * fills["price"]).abs()
    return fills


def next_bar_returns(returns: pd.Series, holdings: pd.DataFrame, fills: pd.DataFrame) -> pd.Series:
    """The strategy's returns had every fill been priced at its next bar's open.

    From that open on the two books hold the same shares, so the alternative
    book is worth the backtest's book plus every ``impact`` booked so far --
    added up, not compounded: the money a better price saved stays in cash, as
    the later fills buy the same shares either way. Its final value is the
    backtest's plus the sum of the fills' differences, to the cent.
    """
    priced = fills.dropna(subset=["impact", "next_date"])
    impact = priced["impact"].groupby(utc_dates(priced["next_date"])).sum()
    impact = impact.reindex(returns.index, fill_value=0.0)
    value = holdings.sum(axis="columns").reindex(returns.index)
    start = float(value.iloc[0] / (1 + returns.iloc[0]))
    alternative = value + impact.cumsum()
    return (alternative / alternative.shift(1).fillna(start) - 1).rename(returns.name)


def signed_money(value: float) -> str:
    """``+$1,234.50`` for a gain, ``-$1,234.50`` for a loss."""
    return ("+" if value > 0 else "") + money(value)


def build_next_bar_fills(returns, holdings, transactions, fill_bars, settings) -> Section | None:
    import empyrical as ep

    settings = settings or {}
    if not settings.get("same_bar_execution", True):
        # Orders already wait for the next bar: there is nothing to reprice.
        return None
    reference = settings.get("price_used_in_order_execution") or "close"
    if (transactions is None or transactions.empty or fill_bars is None or fill_bars.empty
            or not {"next_open", "next_date", reference} <= set(fill_bars.columns)):
        return None

    fills = repriced_fills(transactions, fill_bars, reference)
    priced = fills.dropna(subset=["impact", "next_date"])
    priced = priced[utc_dates(priced["next_date"]).isin(returns.index)]
    if priced.empty:
        return None
    alternative = next_bar_returns(returns, holdings, priced)

    section = Section("Next-Bar Fills")
    signal, waited = f"Signal Bar's {reference.title()}", "Next Bar's Open"

    def stats(series):
        return {
            "Total return": f"{ep.cum_returns_final(series):.2%}",
            "Annual return": f"{ep.annual_return(series):.2%}",
            "Annual volatility": f"{ep.annual_volatility(series):.2%}",
            "Sharpe ratio": f"{ep.sharpe_ratio(series):.2f}",
            "Max drawdown": f"{ep.max_drawdown(series):.2%}",
        }

    compared = pd.DataFrame({signal: stats(returns), waited: stats(alternative)})
    section.table(Table(compared, None), title="The Same Trades, a Bar Later", first_header="Metric")

    weights = priced["value"]
    buys, sells = priced[priced["amount"] > 0], priced[priced["amount"] < 0]

    def weighted(frame):
        return f"{np.average(frame['move_bps'], weights=frame['value']):+.1f} bps" if len(frame) else "-"

    start_capital = float(holdings.sum(axis="columns").iloc[0] / (1 + returns.iloc[0]))
    total_impact = float(priced["impact"].sum())
    final = float(holdings.sum(axis="columns").iloc[-1])
    waited_final = start_capital * float((1 + alternative).prod())
    moves = pd.Series({
        "Fills repriced": f"{len(priced):,} of {len(fills):,}",
        "Move to the next open (value-weighted)": f"{np.average(priced['move_bps'], weights=weights):+.1f} bps",
        "  buys": weighted(buys),
        "  sells": weighted(sells),
        "Fills the wait made worse": f"{(priced['move_bps'] > 0).mean():.0%}",
        "Fills the wait made better": f"{(priced['move_bps'] < 0).mean():.0%}",
        "Effect of waiting on PnL": signed_money(total_impact) + (" (a gain)" if total_impact > 0 else
                                                                " (a loss)" if total_impact < 0 else ""),
        "  vs starting capital": f"{total_impact / start_capital:+.2%}" if start_capital else "-",
        f"Final value, {signal.lower()}": money(final),
        f"Final value, {waited.lower()}": money(waited_final),
        "Difference in final value": signed_money(waited_final - final) + (
            " = the sum over the fills" if abs(waited_final - final - total_impact) < 0.01 else
            f" (the fills add up to {signed_money(total_impact)})"),
    })
    section.table(Table(moves.to_frame("Value"), None), title="What the Wait Did", first_header="Metric")
    section.side.append(
        '<p class="caption">Signs: a move in bps is positive when the next open was worse for the fill -- '
        'higher for a buy, lower for a sell. The effect on PnL is positive when waiting would have paid '
        'more, negative when it would have cost.</p>')
    section.side.append(
        f'<p class="explain">This backtest fills each order on the bar whose data produced it, at that '
        f'bar\'s {reference}. A real order sent on that data reaches the market on the bar after. Here '
        'every fill is priced again at the open of the instrument\'s next bar in the data -- the next '
        'session for daily bars -- with the same quantity and the same slippage. From that open on the '
        'two books hold the same shares, so the difference is exactly the price paid, booked on the day '
        'of the next bar and kept in cash rather than reinvested.</p>'
        '<p class="explain">A strategy that buys what has just risen and sells what has just fallen '
        'tends to pay for the wait; one whose edge is real survives it. The trades themselves are not '
        're-decided: the same shares, a bar later.</p>')
    if len(priced) < len(fills):
        section.side.append(
            f'<p class="caption">{len(fills) - len(priced):,} fills on the last bar of the data have no '
            'next bar and keep their price.</p>')

    def cumulative(ax):
        ax.plot(ep.cum_returns(returns, 1.0), color=STRATEGY_COLOR, lw=1.6, label=signal.lower().capitalize())
        ax.plot(ep.cum_returns(alternative, 1.0), color=LOSS_COLOR, lw=1.4, label=waited.lower().capitalize())
        ax.axhline(1.0, color="black", lw=0.8)
        ax.legend(loc="upper left")

    section.chart(chart("Cumulative Returns, Filled Now or a Bar Later", cumulative, height=4.0, y="growth"))

    data = priced["move_bps"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) and data.max() - data.min() > 0:
        def histogram(ax):
            low, high = np.nanpercentile(data, [1, 99]) if len(data) > 20 else (data.min(), data.max())
            bins = np.linspace(low, high, 40) if high > low else 20
            for side, label, color in ((1, "Buys", STRATEGY_COLOR), (-1, "Sells", BENCHMARK_COLOR)):
                values = priced.loc[np.sign(priced["amount"]) == side, "move_bps"].dropna().clip(low, high)
                if len(values):
                    ax.hist(values, bins=bins, color=color, alpha=0.7, label=label)
            ax.axvline(0, color="black", lw=1, ls="--")
            ax.set_xlabel("Move to the next open, bps (positive = waiting cost more)")
            ax.set_ylabel("Fills")
            ax.legend(loc="upper right")

        section.chart(chart("Move from the Fill to the Next Open", histogram, height=3.2))

    running = priced["impact"].groupby(utc_dates(priced["next_date"])).sum().reindex(
        returns.index, fill_value=0.0).cumsum()

    def impact_over_time(ax):
        values = running.to_numpy()
        ax.fill_between(running.index, values, 0, where=values >= 0, color=STRATEGY_COLOR, alpha=0.35, lw=0,
                        interpolate=True)
        ax.fill_between(running.index, values, 0, where=values < 0, color=LOSS_COLOR, alpha=0.35, lw=0,
                        interpolate=True)
        ax.plot(running.index, values, color="#333333", lw=1.1)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel(f"Effect of waiting, {unit()} (+ = gain)")

    section.chart(chart("What the Wait Did to PnL Over Time", impact_over_time, height=3.0))

    base, after = ep.annual_return(returns), ep.annual_return(alternative)
    detail = (f"Annual return {base:.1%} with fills at the signal bar's {reference}, "
              f"{after:.1%} at the next bar's open; the wait's effect on PnL is {signed_money(total_impact)}.")
    label, value = "Annual return, filled a bar later", f"{base:.1%} → {after:.1%}"
    common = dict(key="timing", threshold=f"waiting a bar costs under {WARN_SHARE:.0%} of the return",
                  advice="Run it again with next-bar execution: that is the result a real order would get.")
    if after >= base:
        section.finding(OK, label, value, "Waiting one bar does not hurt this strategy. " + detail, **common)
    elif base > 0:
        lost = (base - after) / base
        if after <= 0:
            section.finding(BAD, label, value, "Waiting one bar wipes out the return. " + detail,
                            issue="waiting one bar for the fills wipes out the return", **common)
        else:
            status = BAD if lost >= BAD_SHARE else WARN if lost >= WARN_SHARE else OK
            section.finding(status, label, value, f"Waiting one bar costs {lost:.0%} of the return. " + detail,
                            issue=f"waiting one bar for the fills costs {lost:.0%} of the return", **common)
    else:
        section.finding(WARN, label, value, "The backtest lost money; waiting one bar makes it worse. " + detail,
                        issue="waiting one bar for the fills deepens the loss", **common)
    return section
