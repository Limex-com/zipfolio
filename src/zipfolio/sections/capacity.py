"""How much capital the strategy could take before liquidity binds."""

from __future__ import annotations

import pandas as pd

from ..page import Section
from ..style import chart
from ..tables import Table


def build_capacity(returns, positions, transactions, market_data) -> Section:
    """Days to liquidate, low-liquidity fills and the capital-base sweep.

    The same analysis as ``create_capacity_tear_sheet``, with its daily-volume
    join repaired: ``capacity.daily_txns_with_bar_data`` ends in
    ``.set_index("date").asfreq("D")``, and the index it sets holds one row per
    symbol per day -- so any strategy that trades two names on the same day
    fails with "cannot reindex on an axis with duplicate labels". The ``asfreq``
    only inserts empty days, which none of its callers need: the liquidity
    tables group by symbol and the slippage sweep resamples by day. The repaired
    copy is in place for this one call.
    """
    from pyfolio import capacity, plotting

    section = Section("Capacity")
    original = capacity.daily_txns_with_bar_data
    capacity.daily_txns_with_bar_data = _daily_txns_with_bar_data
    try:
        last_n_days = 126
        section.side.append(
            '<p class="explain">At a constant $1m capital base, taking at most 20% of '
            'a day\'s volume, with the trailing 5-day mean as that volume.</p>')
        for label, window in (("Whole Backtest", None), (f"Last {last_n_days} Sessions", last_n_days)):
            days = capacity.get_max_days_to_liquidate_by_ticker(
                positions, market_data, max_bar_consumption=0.2, capital_base=1e6,
                mean_volume_window=5, last_n_days=window)
            section.table(
                Table(days[days["days_to_liquidate"] > 1], None, None,
                      {"days_to_liquidate": "{:.1f}".format, "pos_alloc_pct": "{:.2f}%".format}),
                title=f"Over a Day to Liquidate: {label}", first_header="Symbol",
                rename={"date": "Date", "days_to_liquidate": "Days", "pos_alloc_pct": "Allocation"})
        for label, window in (("Whole Backtest", None), (f"Last {last_n_days} Sessions", last_n_days)):
            fills = capacity.get_low_liquidity_transactions(transactions, market_data, last_n_days=window)
            section.table(
                Table(fills[fills["max_pct_bar_consumed"] > 5], None, None,
                      {"max_pct_bar_consumed": "{:.2f}%".format}),
                title=f"Fills Over 5% of Daily Volume: {label}", first_header="Symbol",
                rename={"date": "Date", "max_pct_bar_consumed": "Of Volume"})

        starting_capital = positions.iloc[0].sum() / (1 + returns.iloc[0])
        section.chart(chart("Capital Base Performance Sweep", lambda ax: plotting.plot_capacity_sweep(
            returns, transactions, market_data, starting_capital,
            min_pv=100000, max_pv=300000000, step_size=1000000, ax=ax), height=3.5))
    finally:
        capacity.daily_txns_with_bar_data = original
    return section


def _daily_txns_with_bar_data(transactions, market_data):
    """``pyfolio.capacity.daily_txns_with_bar_data`` without the ``asfreq`` -- see ``build_capacity``."""
    transactions.index.name = "date"
    txn_daily = pd.DataFrame(
        transactions.assign(amount=abs(transactions.amount))
        .groupby(["symbol", pd.Grouper(freq="D")])
        .sum()["amount"]
    )
    txn_daily["price"] = market_data.xs("price", level=1).unstack()
    txn_daily["volume"] = market_data.xs("volume", level=1).unstack()
    return txn_daily.reset_index().set_index("date").sort_index()
