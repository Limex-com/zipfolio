"""Turnover, volume and when the strategy trades."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..currency import money
from ..inputs import utc_dates
from ..page import INFO, Section
from ..style import BENCHMARK_COLOR, STRATEGY_COLOR, chart, percent_ticks
from ..tables import Table


def daily_turnover(transactions: pd.DataFrame, positions: pd.DataFrame) -> pd.Series:
    """Each session's traded value over the portfolio's value at its close.

    Not pyfolio's default, which divides by the average gross book of the day
    and the day before: a strategy that goes from all cash to fully invested
    has half a book on average that day, so buying the whole portfolio once
    counted as 200%, and its annual turnover came out twice what was traded.
    """
    traded = (transactions["amount"] * transactions["price"]).abs().groupby(utc_dates(transactions.index)).sum()
    value = positions.sum(axis="columns")
    return (traded.reindex(value.index, fill_value=0.0) / value.where(value > 0)).fillna(0.0)


def build_transactions(returns, positions, transactions, market_timezone: str) -> Section:
    from pyfolio import plotting

    section = Section("Transactions")
    turnover = daily_turnover(transactions, positions)
    section.table(Table(trading_activity(returns, transactions, turnover, positions).to_frame("Value"), None, None),
                  title="Trading Activity", first_header="Metric")
    section.side.append(
        '<p class="caption"><b>Turnover</b> is the value traded, buys and sells together, over the portfolio\'s '
        'value: buying the whole portfolio and selling it again is 2&times;. <b>One-way</b> counts only the '
        'smaller of the two sides -- how many times the portfolio was replaced.</p>')

    traded = turnover[turnover > 0]

    def daily(ax):
        ax.bar(traded.index, traded.to_numpy(), width=3, color=STRATEGY_COLOR, label="Trading day")
        smooth = turnover.rolling(63, min_periods=20).mean()
        ax.plot(smooth.index, smooth.to_numpy(), color=BENCHMARK_COLOR, lw=1.4, label="63-session average")
        ax.set_ylabel("Traded value / portfolio")
        ax.legend(loc="upper left")

    section.chart(chart("Daily Turnover", daily, y="pct"))
    section.chart(chart("Daily Trading Volume", lambda ax: plotting.plot_daily_volume(
        returns, transactions, ax=ax)))
    # Every trading day the same turnover -- all in, all out -- is one bar, said better by the table.
    if len(traded) > 1 and traded.max() - traded.min() > 0.1:
        def distribution(ax):
            ax.hist(traded.to_numpy(), bins=min(30, max(5, len(traded) // 2)), color=STRATEGY_COLOR, alpha=0.8)
            ax.set_xlabel("Traded value / portfolio, on days with a fill")
            ax.set_ylabel("Days")
            ax.xaxis.set_major_formatter(percent_ticks())

        section.chart(chart("Turnover on Trading Days", distribution))
    section.chart(chart(f"Transaction Time Distribution ({market_timezone})", lambda ax: plot_fill_times(
        transactions, market_timezone, ax), y="pct"))
    annual = float(turnover.mean() * 252)
    if np.isfinite(annual) and annual > 0:
        section.finding(INFO, "Turnover", f"{annual:.1f}× a year",
                        f"Traded value over portfolio value, buys and sells together; trades on "
                        f"{transactions.index.normalize().nunique():,} of {len(returns):,} sessions.", key="turnover")
    return section


def trading_activity(returns: pd.Series, transactions: pd.DataFrame, turnover: pd.Series,
                     positions: pd.DataFrame | None = None) -> pd.Series:
    """How often, how much and how the strategy trades, in one column."""
    value = (transactions["amount"] * transactions["price"]).abs()
    buys = transactions["amount"] > 0
    days = transactions.index.normalize()
    per_day = transactions.groupby(days)
    trading_days = per_day.size()
    sessions = pd.Series(np.arange(len(returns)), index=returns.index)
    positions_of_days = sessions.reindex(trading_days.index.tz_convert("UTC").normalize()).dropna()
    gaps = np.diff(positions_of_days.to_numpy())
    years = len(returns) / 252
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
        "Turnover, a year (traded / portfolio)": f"{turnover.mean() * 252:.1f}×",
    }
    if positions is not None and years > 0:
        book = float(positions.sum(axis="columns").mean())
        one_way = min(value[buys].sum(), value[~buys].sum()) / book / years if book > 0 else float("nan")
        rows["One-way turnover, a year"] = f"{one_way:.1f}×"
    rows["Turnover on an average trading day"] = f"{turnover[turnover > 0].mean():.0%}" if (turnover > 0).any() else "-"
    rows["Largest daily turnover"] = f"{turnover.max():.0%}"
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
