"""The cost model in words, and what commission and slippage took."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..inputs import utc_dates
from ..page import Section
from ..style import BENCHMARK_COLOR, LOSS_COLOR, STRATEGY_COLOR, chart, color_mix
from ..tables import Table


def build_trading_costs(returns, positions, transactions, orders, settings) -> Section:
    """What the cost model was, in words, and what it took, in numbers.

    Commission is read from the orders: ziplime charges it to the order as it
    fills, and the transactions carry none. Slippage is not recorded anywhere,
    but the fixed-basis-point model makes it exact to reconstruct: a buy fills
    at reference x (1 + b), so it paid price x b / (1 + b) a share over the
    reference, and a sell price x b / (1 - b).
    """
    import empyrical as ep

    settings = settings or {}
    section = Section("Trading Costs")

    commissions = pd.Series(dtype=float)
    if orders is not None and not orders.empty and "commission" in orders.columns:
        charged = orders.drop_duplicates("id", keep="last") if "id" in orders.columns else orders
        charged = charged.assign(
            commission=pd.to_numeric(charged["commission"], errors="coerce").fillna(0.0),
            date=utc_dates(charged["dt"]),
        )
        commissions = charged.groupby("date")["commission"].sum()
        commission_by_symbol = charged.groupby(charged["symbol"].astype(str))["commission"].sum()
        order_count = int((charged["commission"] > 0).sum()) or len(charged)
    else:
        charged = None
        commission_by_symbol = pd.Series(dtype=float)
        order_count = 0

    rate = float(settings.get("slippage_basis_points") or 0.0) / 1e4
    traded = (transactions["amount"] * transactions["price"]).abs()
    buys = transactions["amount"] > 0
    slippage_per_fill = traded * rate / np.where(buys, 1 + rate, 1 - rate)
    fill_dates = utc_dates(transactions.index)
    slippage = pd.Series(slippage_per_fill.to_numpy(), index=fill_dates).groupby(level=0).sum()
    slippage_by_symbol = pd.Series(slippage_per_fill.to_numpy(), index=transactions["symbol"]).groupby(level=0).sum()

    daily = pd.DataFrame({"Commission": commissions, "Slippage": slippage}).reindex(returns.index).fillna(0.0)
    total_commission = float(daily["Commission"].sum())
    total_slippage = float(daily["Slippage"].sum())
    total_costs = total_commission + total_slippage
    traded_value = float(traded.sum())

    start_value = positions.sum(axis="columns") / (1 + returns)
    gross_returns = returns + daily.sum(axis="columns") / start_value
    starting_capital = float(settings.get("capital") or start_value.iloc[0])
    net_profit = float(positions.sum(axis="columns").iloc[-1] - start_value.iloc[0])
    gross_profit = net_profit + total_costs
    net_annual, gross_annual = ep.annual_return(returns), ep.annual_return(gross_returns)

    section.side.append("<h3>How Costs Are Modeled</h3>" + "".join(
        f'<p class="explain">{line}</p>' for line in cost_model_text(settings)))
    section.table(Table(_cost_settings_table(settings).to_frame("Setting"), None), title="Cost Settings",
                  first_header="Parameter")

    summary = {
        "Commissions": f"${total_commission:,.2f}",
        "Slippage (reconstructed)": f"${total_slippage:,.2f}",
        "Total trading costs": f"${total_costs:,.2f}",
        "Traded value": f"${traded_value:,.0f}",
        "Costs per traded value": f"{total_costs / traded_value * 1e4:,.2f} bps" if traded_value else "-",
        "Costs vs starting capital": f"{total_costs / starting_capital:.2%}" if starting_capital else "-",
        "Costs vs gross profit": f"{total_costs / gross_profit:.2%}" if gross_profit > 0 else "-",
        "Annual return before costs": f"{gross_annual:.2%}",
        "Annual return after costs": f"{net_annual:.2%}",
        "Annual drag from costs": f"{(gross_annual - net_annual) * 100:.2f} pp",
        "Orders charged": f"{order_count:,}",
        "Fills": f"{len(transactions):,}",
        "Avg. commission per order": f"${total_commission / order_count:,.2f}" if order_count else "-",
        "Avg. commission per share": (f"${total_commission / transactions['amount'].abs().sum():,.4f}"
                                      if total_commission else "-"),
    }
    minimum = float(settings.get("commission_min_trade_cost") or 0.0)
    if minimum > 0 and charged is not None and len(charged):
        at_minimum = int(np.isclose(charged["commission"], minimum).sum())
        summary["Orders paying the minimum"] = f"{at_minimum:,} ({at_minimum / len(charged):.0%})"
    section.table(Table(pd.Series(summary).to_frame("Value"), None), title="Cost Summary", first_header="Metric")

    by_symbol = pd.DataFrame({"Commission": commission_by_symbol, "Slippage": slippage_by_symbol}).fillna(0.0)
    by_symbol["Total"] = by_symbol.sum(axis="columns")
    by_symbol["Share"] = by_symbol["Total"] / by_symbol["Total"].sum() if by_symbol["Total"].sum() else 0.0
    by_symbol = by_symbol.sort_values("Total", ascending=False).head(15)
    money = "${:,.2f}".format
    section.table(Table(by_symbol, None, None, {"Commission": money, "Slippage": money, "Total": money,
                                                 "Share": "{:.1%}".format}),
                  title="Costs by Symbol", first_header="Symbol")

    from pyfolio import plotting

    def cumulative(ax):
        stacked = daily.cumsum()
        ax.stackplot(stacked.index, stacked["Commission"], stacked["Slippage"],
                     labels=["Commission", "Slippage"], colors=[BENCHMARK_COLOR, LOSS_COLOR], alpha=0.75)
        ax.legend(loc="upper left")
        ax.set_ylabel("Cumulative cost, $")

    def monthly(ax):
        months = daily.resample("ME").sum()
        ax.bar(months.index, months["Commission"], width=20, color=BENCHMARK_COLOR, label="Commission")
        ax.bar(months.index, months["Slippage"], width=20, bottom=months["Commission"], color=LOSS_COLOR,
               label="Slippage")
        ax.legend(loc="upper left")
        ax.set_ylabel("Cost per month, $")

    def before_after(ax):
        ax.plot(ep.cum_returns(gross_returns, 1.0), color=BENCHMARK_COLOR, lw=1.2, ls="--", label="Before costs")
        ax.plot(ep.cum_returns(returns, 1.0), color=STRATEGY_COLOR, lw=1.8, label="After costs")
        ax.axhline(1.0, ls="--", lw=1, color="black", zorder=1)
        ax.legend(loc="upper left")
        ax.set_ylabel("Cumulative returns")

    section.chart(chart("Cumulative Trading Costs", cumulative))
    section.chart(chart("Trading Costs by Month", monthly))
    section.chart(chart("Returns Before and After Costs", before_after, height=3.5, y="growth"))
    section.chart(chart("Sensitivity to Additional Slippage", lambda ax: plotting.plot_slippage_sensitivity(
        returns, positions, transactions, ax=ax), y="pct"))
    def sweep(ax):
        plotting.plot_slippage_sweep(returns, positions, transactions, ax=ax)
        # pyfolio draws the sweep at half a point in the default cycle; seven
        # hairlines read as one. A sequential ramp says "more slippage" instead.
        lines = ax.get_lines()
        for line, shade in zip(lines, np.linspace(0.25, 1.0, len(lines))):
            line.set_linewidth(1.2)
            line.set_color(color_mix(STRATEGY_COLOR, LOSS_COLOR, shade))
        ax.legend(loc="upper left", ncol=2)

    section.chart(chart("Cumulative Returns with Additional Slippage", sweep, height=3.5, y="growth"))
    section.side.append(
        '<p class="caption">The last two charts charge every dollar traded an extra 3 to 50 bps on '
        'top of the costs above -- how much worse execution the strategy can take before its edge '
        'is gone.</p>')
    return section


def cost_model_text(settings: dict) -> list[str]:
    """The cost model in sentences, from the run's own settings."""
    if not settings:
        return ["The cost settings of this backtest were not passed to the report; the figures "
                "below are measured from its orders and fills."]
    lines = []
    if settings.get("commission_model") == "per_dollar":
        rate = settings.get("commission_rate")
        rate = settings.get("commission_per_share") if rate is None else rate
        lines.append(
            f"<b>Commission</b> is {float(rate):.4%} of the value of each fill (ziplime's "
            "<code>PerDollar</code> model). It is charged to the order as it fills and comes out of "
            "cash; it is not part of the fill price.")
    else:
        per_share = float(settings.get("commission_per_share") or 0.0)
        minimum = float(settings.get("commission_min_trade_cost") or 0.0)
        floor = (f", and never less than ${minimum:,.2f} per order -- so a small order pays the "
                 "minimum, not its per-share amount" if minimum > 0 else ", with no minimum per order")
        lines.append(
            f"<b>Commission</b> is ${per_share:,.4f} per share{floor} (ziplime's <code>PerShare</code> "
            "model). It is charged to the order as it fills and comes out of cash; it is not part "
            "of the fill price.")
    bps = float(settings.get("slippage_basis_points") or 0.0)
    reference = settings.get("price_used_in_order_execution") or "close"
    if bps > 0:
        lines.append(
            f"<b>Slippage</b>: every fill is priced {bps:g} bps worse than the bar's {reference} -- "
            f"a buy at {reference} &times; {1 + bps / 1e4:.4f}, a sell at {reference} &times; "
            f"{1 - bps / 1e4:.4f} (ziplime's <code>FixedBasisPointsSlippage</code>). It stands in for "
            "the bid-ask spread and market impact. It is in the fill price, so the backtest never "
            "records it as a cost; the report reconstructs it from the fills.")
    else:
        lines.append(f"<b>Slippage</b>: none -- fills are priced at the bar's {reference}.")
    volume = settings.get("slippage_volume_limit")
    if volume:
        lines.append(
            f"<b>Volume limit</b>: a fill takes at most {float(volume):.0%} of the bar's volume. The "
            "rest of the order waits for later bars, so a large order can fill over several bars, at "
            "several prices, and pay commission on each fill.")
    if settings.get("same_bar_execution", True):
        lines.append(f"<b>Execution</b>: an order fills on the bar it was placed in, at that bar's "
                     f"{reference}.")
    else:
        lines.append(f"<b>Execution</b>: an order fills on the next bar, at that bar's {reference} -- "
                     "a bar later than the data that triggered it.")
    lines.append(
        "Both costs are already inside every return, metric and chart in this report. Not modeled "
        "by these settings: borrowing costs of short positions, financing of leverage, taxes, and "
        "market impact beyond the fixed basis points.")
    return lines


def _cost_settings_table(settings: dict) -> pd.Series:
    if not settings:
        return pd.Series(dtype=object)
    per_dollar = settings.get("commission_model") == "per_dollar"
    rows = {"Commission model": "Per dollar" if per_dollar else "Per share"}
    if per_dollar:
        rate = settings.get("commission_rate")
        rate = settings.get("commission_per_share") if rate is None else rate
        rows["Commission rate"] = f"{float(rate):.4%} of value"
    else:
        rows["Commission per share"] = f"${float(settings.get('commission_per_share') or 0):,.4f}"
        rows["Minimum per order"] = f"${float(settings.get('commission_min_trade_cost') or 0):,.2f}"
    rows["Slippage"] = f"{float(settings.get('slippage_basis_points') or 0):g} bps"
    if settings.get("slippage_volume_limit") is not None:
        rows["Volume limit"] = f"{float(settings['slippage_volume_limit']):.0%} of bar volume"
    rows["Fill price"] = f"Bar {settings.get('price_used_in_order_execution') or 'close'}"
    rows["Fills on"] = "Same bar" if settings.get("same_bar_execution", True) else "Next bar"
    if settings.get("capital"):
        rows["Starting capital"] = f"${float(settings['capital']):,.0f}"
    return pd.Series(rows)
