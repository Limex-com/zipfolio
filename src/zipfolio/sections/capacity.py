"""How much capital the strategy could take before liquidity binds."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..currency import compact, currency
from ..inputs import utc_dates
from ..page import INFO, OK, WARN, Section
from ..style import chart
from ..tables import Table
from .execution import IMPACT_PARTICIPATION

#: The share of a day's volume a position may take to sell when the run sets no volume limit.
DEFAULT_LIMIT = 0.2


def fill_participation(transactions: pd.DataFrame, market_data: pd.DataFrame) -> pd.Series:
    """Each fill's shares as a share of its instrument's volume that day."""
    volume = market_data.xs("volume", level=1).astype(float)
    dates = utc_dates(transactions.index)
    day_volume = [volume.at[date, symbol] if date in volume.index and symbol in volume.columns else np.nan
                  for date, symbol in zip(dates, transactions["symbol"].astype(str))]
    day_volume = pd.Series(day_volume, index=transactions.index, dtype=float)
    return (transactions["amount"].abs() / day_volume.where(day_volume > 0)).dropna()


def capacity_at(participation: pd.Series, starting_capital: float, share: float) -> float:
    """The starting capital at which the largest fill would take ``share`` of its day's volume.

    The strategy sizes its orders from its capital, so scaling the capital by k
    scales every fill by k; the first to reach ``share`` sets the limit.
    """
    largest = float(participation.max()) if len(participation) else 0.0
    return starting_capital * share / largest if largest > 0 else float("inf")


def build_capacity(returns, positions, transactions, market_data, settings: dict | None = None) -> Section:
    """Days to liquidate, heavy fills, the capital the strategy could take, and the capital-base sweep.

    Measured at the strategy's own capital at the end of the backtest, with the
    run's volume limit -- the share of a bar's volume the engine lets a fill
    take -- as the most a position may take of a day's volume to sell, and 1%
    of it as where a fill starts to move the price (as in Execution).

    The pyfolio analysis has its daily-volume join repaired:
    ``capacity.daily_txns_with_bar_data`` ends in ``.set_index("date").asfreq("D")``,
    and the index it sets holds one row per symbol per day -- so any strategy
    that trades two names on the same day fails with "cannot reindex on an axis
    with duplicate labels". The ``asfreq`` only inserts empty days, which none
    of its callers need. The repaired copy is in place for this one call.
    """
    from pyfolio import capacity, plotting

    settings = settings or {}
    limit = float(settings.get("slippage_volume_limit") or DEFAULT_LIMIT)
    section = Section("Capacity")
    base = float(positions.sum(axis="columns").iloc[-1])
    starting_capital = float(positions.iloc[0].sum() / (1 + returns.iloc[0]))
    participation = fill_participation(transactions, market_data)
    impact_capital = capacity_at(participation, starting_capital, IMPACT_PARTICIPATION)
    limit_capital = capacity_at(participation, starting_capital, limit)

    original = capacity.daily_txns_with_bar_data
    capacity.daily_txns_with_bar_data = _daily_txns_with_bar_data
    slow, heavy = 0, 0
    try:
        last_n_days = 126
        if len(participation):
            facts = pd.Series({
                "Capital measured at": f"{compact(base)} (the end of the backtest)",
                "Largest fill, share of its day's volume": f"{participation.max():.3%}",
                "Median fill, share of its day's volume": f"{participation.median():.4%}",
                f"Starting capital with fills up to {IMPACT_PARTICIPATION:.0%} of volume": (
                    compact(impact_capital) if np.isfinite(impact_capital) else "-"),
                f"Starting capital with fills up to {limit:.0%} of volume": (
                    compact(limit_capital) if np.isfinite(limit_capital) else "-"),
            })
            section.table(Table(facts.to_frame("Value"), None), title="How Much Capital It Could Take",
                          first_header="Measure")
        section.side.append(
            f'<p class="explain">The strategy sizes its orders from its capital, so more capital means larger '
            f'fills. Up to {IMPACT_PARTICIPATION:.0%} of a day\'s volume a fill barely moves the price; '
            f'{limit:.0%} is the most the engine lets a fill take of a bar. The checks below sell every '
            f'position at {compact(base)}, taking at most {limit:.0%} of a day\'s volume, with the trailing '
            f'5-day mean as that volume.</p>')
        windows = (("the whole backtest", None), (f"the last {last_n_days} sessions", last_n_days))
        slow_rows, heavy_rows = {}, {}
        for label, window in windows:
            days = capacity.get_max_days_to_liquidate_by_ticker(
                positions, market_data, max_bar_consumption=limit, capital_base=base,
                mean_volume_window=5, last_n_days=window)
            slow_rows[label] = days[days["days_to_liquidate"] > 1]
        threshold = IMPACT_PARTICIPATION * 100
        for label, window in windows:
            fills = capacity.get_low_liquidity_transactions(transactions, market_data, last_n_days=window)
            heavy_rows[label] = fills[fills["max_pct_bar_consumed"] > threshold]
        slow, heavy = len(slow_rows[windows[0][0]]), len(heavy_rows[windows[0][0]])
        _listed(section, slow_rows, "Over a Day to Liquidate", "Every position sellable within a day",
                f"Taking at most {limit:.0%} of its day's volume",
                {"days_to_liquidate": "{:.1f}".format, "pos_alloc_pct": "{:.2f}%".format},
                {"date": "Date", "days_to_liquidate": "Days", "pos_alloc_pct": "Allocation"})
        _listed(section, heavy_rows, f"Fills Over {IMPACT_PARTICIPATION:.0%} of Daily Volume",
                f"No fill over {IMPACT_PARTICIPATION:.0%} of daily volume",
                "No name's fills in a day took more of its volume",
                {"max_pct_bar_consumed": "{:.2f}%".format}, {"date": "Date", "max_pct_bar_consumed": "Of Volume"})

        step = max(int(base), 1)

        def sweep(ax):
            # pyfolio steps through range(): whole units of the currency.
            plotting.plot_capacity_sweep(
                returns, transactions, market_data, starting_capital,
                min_pv=max(int(base / 10), 1), max_pv=int(base * 300), step_size=step, ax=ax)
            ax.set_xlabel(f"Capital base, millions of {currency().code}")

        section.chart(chart("Capital Base Performance Sweep", sweep, height=3.5))
    finally:
        capacity.daily_txns_with_bar_data = original

    common = dict(key="liquidity", threshold=f"every position sellable within a day at ≤ {limit:.0%} of volume",
                  advice="Trade more liquid names, or smaller positions in the thin ones.")
    if slow or heavy:
        section.finding(WARN, f"Liquidity at {compact(base)}",
                        f"{slow:,} names over a day to sell" if slow else
                        f"{heavy:,} names over {IMPACT_PARTICIPATION:.0%} of volume",
                        f"At {compact(base)}, taking at most {limit:.0%} of a day's volume, {slow:,} names would "
                        f"take over a day to sell; {heavy:,} had a day's fills over {IMPACT_PARTICIPATION:.0%} of "
                        "its volume.", issue=f"at {compact(base)} some positions are hard to sell", **common)
    else:
        section.finding(OK, f"Liquidity at {compact(base)}", "all sellable within a day",
                        f"At {compact(base)}, taking at most {limit:.0%} of a day's volume.", **common)
    if np.isfinite(impact_capital):
        section.finding(INFO, "Capacity", f"{compact(impact_capital)} at {IMPACT_PARTICIPATION:.0%} of volume",
                        f"The starting capital at which the strategy's largest fill would take "
                        f"{IMPACT_PARTICIPATION:.0%} of its day's volume -- where its own orders start to move the "
                        f"price. At {limit:.0%}, the engine's volume limit: {compact(limit_capital)}.", key="capacity")
    return section


def _listed(section: Section, rows: dict[str, pd.DataFrame], title: str, clear: str, detail: str,
            formatters: dict, rename: dict) -> None:
    """A table for each window that has rows; for the windows that have none, one line saying so.

    Four tables reading "None." said one thing four times; the reader wants the
    names when there are some, and a sentence when there are not.
    """
    import html

    from ..page import MARKS

    empty = [label for label, frame in rows.items() if frame.empty]
    for label, frame in rows.items():
        if not frame.empty:
            section.table(Table(frame, None, None, formatters), title=f"{title}: {label[4:].title()}",
                          first_header="Symbol", rename=rename)
    if empty:
        whole = len(empty) == len(rows)
        heading = clear if whole else f"{clear} in {' or '.join(empty)}"
        text = f"{detail}, over {' and '.join(empty)}." if whole else f"{detail}."
        measure = f"{title}, {' and '.join(empty)}"
        section.side.append(f'<div class="check" data-metric="{html.escape(measure)}" data-value="0">'
                            f'<span class="mark {OK}">{MARKS[OK]}</span>'
                            f'<div><b>{html.escape(heading)}</b><br>{html.escape(text)}</div></div>')


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
