"""Turnover, volume and when the strategy trades."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..currency import money
from ..page import Section
from ..style import STRATEGY_COLOR, chart
from ..tables import Table


def build_transactions(returns, positions, transactions, market_timezone: str) -> Section:
    from pyfolio import plotting, txn

    section = Section("Transactions")
    turnover = txn.get_turnover(positions, transactions)
    section.table(Table(trading_activity(returns, transactions, turnover).to_frame("Value"), None, None),
                  title="Trading Activity", first_header="Metric")

    section.chart(chart("Daily Turnover", lambda ax: plotting.plot_turnover(
        returns, transactions, positions, ax=ax), y="pct"))
    section.chart(chart("Daily Trading Volume", lambda ax: plotting.plot_daily_volume(
        returns, transactions, ax=ax)))
    section.chart(chart("Distribution of Daily Turnover Rates", lambda ax: plotting.plot_daily_turnover_hist(
        transactions, positions, ax=ax, color=STRATEGY_COLOR)))
    section.chart(chart(f"Transaction Time Distribution ({market_timezone})", lambda ax: plot_fill_times(
        transactions, market_timezone, ax), y="pct"))
    return section


def trading_activity(returns: pd.Series, transactions: pd.DataFrame, turnover: pd.Series) -> pd.Series:
    """How often, how much and how the strategy trades, in one column."""
    value = (transactions["amount"] * transactions["price"]).abs()
    buys = transactions["amount"] > 0
    days = transactions.index.normalize()
    per_day = transactions.groupby(days)
    trading_days = per_day.size()
    sessions = pd.Series(np.arange(len(returns)), index=returns.index)
    positions_of_days = sessions.reindex(trading_days.index.tz_convert("UTC").normalize()).dropna()
    gaps = np.diff(positions_of_days.to_numpy())
    rows = {
        "Fills": f"{len(transactions):,}",
        "Symbols traded": f"{transactions['symbol'].nunique():,}",
        "Trading days": f"{len(trading_days):,} of {len(returns):,} ({len(trading_days) / len(returns):.1%})",
        "Median sessions between trading days": f"{np.median(gaps):.0f}" if len(gaps) else "-",
        "Avg. fills per trading day": f"{trading_days.mean():.1f}",
        "Most fills in a day": f"{trading_days.max():,}",
        "Avg. symbols per trading day": f"{per_day['symbol'].nunique().mean():.1f}",
        "Buys": f"{int(buys.sum()):,} fills, {money(value[buys].sum(), 0)}",
        "Sells": f"{int((~buys).sum()):,} fills, {money(value[~buys].sum(), 0)}",
        "Traded value": money(value.sum(), 0),
        "Avg. fill value": money(value.mean(), 0),
        "Median fill value": money(value.median(), 0),
        "Largest fill": money(value.max(), 0),
        "Avg. shares per fill": f"{transactions['amount'].abs().mean():,.0f}",
        "Avg. daily turnover": f"{turnover.mean():.2%}",
        "Annual turnover": f"{turnover.mean() * 252:.0%}",
        "Max daily turnover": f"{turnover.max():.2%}",
    }
    return pd.Series(rows)


def plot_fill_times(transactions, market_timezone: str, ax) -> None:
    """Share of traded value by time of day, on the market's own clock.

    pyfolio's ``plot_txn_time_hist`` is fixed to New York's 09:30-16:00, which
    would cut a Moscow session (10:00-18:45, and evening trading after it) off at
    four o'clock. This spans the hours the fills actually fall in.
    """
    local = transactions.index.tz_convert(market_timezone)
    minutes = np.asarray(local.hour * 60 + local.minute)
    value = (transactions["amount"] * transactions["price"]).abs().to_numpy()
    width = 15
    first = int(minutes.min() // 60 * 60)
    last = int(math.ceil((minutes.max() + 1) / 60) * 60)
    if last - first < 120:
        first, last = max(0, first - 60), min(24 * 60, last + 60)
    share, edges = np.histogram(minutes, bins=np.arange(first, last + width, width), weights=value)
    total = share.sum()
    share = share / total if total else share
    ax.bar(edges[:-1], share, width=width * 0.9, align="edge", color=STRATEGY_COLOR)
    hours = np.arange(first, last + 1, 60)
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h // 60:02d}:00" for h in hours])
    ax.set_xlim(first, last)
    ax.set_ylabel("Share of traded value")
