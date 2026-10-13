"""Profit concentration: how much of the result rests on a few days and a few trades.

A backtest's total is one number, and one number hides whether it was earned
steadily or in a handful of lucky sessions. This section takes the best days
and the best closed trades away and looks at what remains, and redraws the
trades at random to see how often the same trades, in another draw, lose money.
"""

from __future__ import annotations

import traceback

import numpy as np
import pandas as pd

from ..currency import money, unit
from ..page import BAD, OK, WARN, Section
from ..style import BENCHMARK_COLOR, LOSS_COLOR, PALETTE, STRATEGY_COLOR, chart
from ..tables import Table
from .round_trips import extract_round_trips

#: Best days taken away, in turn; a count is shown only when the backtest has many times more sessions.
BEST_DAYS = (5, 10, 20)
BEST_TRADES = 5
BOOTSTRAP_SAMPLES = 5_000
#: A bootstrap that loses money this often is worth a warning; this often, a failure.
WARN_LOSS, BAD_LOSS = 0.10, 0.25


def without_best_days(returns: pd.Series, count: int) -> pd.Series:
    """``returns`` with its ``count`` best sessions spent in cash."""
    trimmed = returns.copy()
    trimmed[returns.nlargest(count).index] = 0.0
    return trimmed


def bootstrap_totals(pnl: np.ndarray, samples: int = BOOTSTRAP_SAMPLES, seed: int = 0) -> np.ndarray:
    """Net PnL of ``samples`` draws of as many trades, with replacement, from ``pnl``.

    Seeded, so the same backtest gets the same page. Drawn in blocks to keep a
    long trade list from asking for gigabytes at once.
    """
    rng = np.random.default_rng(seed)
    block = max(1, 2_000_000 // max(len(pnl), 1))
    totals = []
    for start in range(0, samples, block):
        size = min(block, samples - start)
        totals.append(rng.choice(pnl, size=(size, len(pnl)), replace=True).sum(axis=1))
    return np.concatenate(totals)


def build_profit_concentration(returns, positions, transactions) -> Section | None:
    import empyrical as ep

    section = Section("Profit Concentration")
    counts = [count for count in BEST_DAYS if len(returns) >= 10 * count]
    if not counts:
        return None

    curves = {"All sessions": returns}
    for count in counts:
        curves[f"Without the best {count}"] = without_best_days(returns, count)
    table = pd.DataFrame({
        name: {
            "Total return": f"{ep.cum_returns_final(series):.2%}",
            "Annual return": f"{ep.annual_return(series):.2%}",
            "Sharpe ratio": f"{ep.sharpe_ratio(series):.2f}",
        }
        for name, series in curves.items()
    }).T
    section.table(Table(table, None), title="Without the Best Days", first_header="Sessions")
    section.side.append(
        f'<p class="explain">The same backtest with its best sessions spent in cash -- the {", ".join(map(str, counts))} '
        f'days with the highest returns, out of {len(returns):,}. A strategy whose edge is spread over '
        'time loses some return; one that rests on a few days loses all of it.</p>')

    def cumulative(ax):
        for (name, series), color in zip(curves.items(), [STRATEGY_COLOR, BENCHMARK_COLOR, LOSS_COLOR, PALETTE[3]]):
            ax.plot(ep.cum_returns(series, 1.0), color=color, lw=1.6 if name == "All sessions" else 1.2,
                    label=name)
        ax.axhline(1.0, color="black", lw=0.8)
        ax.legend(loc="upper left")

    section.chart(chart("Cumulative Returns Without the Best Days", cumulative, height=4.0, y="growth"))

    base = ep.cum_returns_final(returns)
    if base > 0:
        fewest = next((count for count in counts if ep.cum_returns_final(curves[f"Without the best {count}"]) <= 0),
                      None)
        smallest = counts[0]
        shown = fewest or smallest
        rest = ep.cum_returns_final(curves[f"Without the best {shown}"])
        detail = (f"Total return {base:.1%}; without its {shown} best days of {len(returns):,}, "
                  f"{rest:.1%}.")
        status = BAD if fewest == smallest else WARN if fewest else OK
        note = (f"{shown} sessions made all of the return. " if fewest else
                "The return does not hang on a few days. ") + detail
        section.finding(status, f"Total return without best {shown} days", f"{base:.1%} → {rest:.1%}", note)

    trades = None
    if positions is not None and transactions is not None:
        try:
            trades = extract_round_trips(returns, positions, transactions)
        except Exception:
            print(f"Warning: no trades for the profit concentration:\n{traceback.format_exc()}")
    if trades is None or len(trades) < 5:
        return section

    pnl = trades["pnl"].astype(float).to_numpy()
    total = float(pnl.sum())
    ranked = np.sort(pnl)[::-1]
    top = float(ranked[:BEST_TRADES].sum())
    top_tenth = float(ranked[:max(1, len(ranked) // 10)].sum())
    totals = bootstrap_totals(pnl)
    loss = float((totals <= 0).mean())
    low, middle, high = np.percentile(totals, [5, 50, 95])
    facts = {
        "Closed trades": f"{len(pnl):,}",
        "Their net PnL": money(total),
        f"Best {BEST_TRADES} trades": money(top),
        "Best 10% of trades": money(top_tenth),
    }
    if total > 0:
        facts[f"  best {BEST_TRADES}, share of net PnL"] = f"{top / total:.0%}"
        facts["  best 10%, share of net PnL"] = f"{top_tenth / total:.0%}"
    facts[f"Net PnL without the best {BEST_TRADES}"] = money(total - top)
    facts["Luck test: chance of a net loss"] = f"{loss:.1%}"
    facts["Luck test: bad case (5th percentile)"] = money(low)
    facts["Luck test: typical (median)"] = money(middle)
    facts["Luck test: good case (95th percentile)"] = money(high)
    section.table(Table(pd.Series(facts).to_frame("Value"), None), title="The Best Trades and the Luck Test",
                  first_header="Measure")
    section.side.append(
        '<p class="explain">Trades are counted as in <a href="#closed-trades">Closed Trades</a>: each sale '
        'with the shares it closed, PnL before commission.</p>'
        f'<p class="explain"><b>The luck test</b> takes the {len(pnl):,} trades and draws {len(pnl):,} of '
        'them again at random -- some twice, some not at all -- and adds them up; '
        f'{BOOTSTRAP_SAMPLES:,} times. If many of those totals are losses, the result depended on which '
        'trades happened to come up, not on an edge most of them share. It treats trades as independent '
        'and keeps their sizes.</p>')

    if total > 0:
        def profit_curve(ax):
            share = np.arange(1, len(ranked) + 1) / len(ranked)
            ax.plot(share * 100, np.cumsum(ranked) / total, color=STRATEGY_COLOR, lw=1.6)
            ax.axhline(1.0, color="black", lw=0.8, ls="--")
            ax.set_xlabel("Trades, best first, %")
            ax.set_ylabel("Share of net PnL")

        section.chart(chart("Net PnL Added Up Trade by Trade, Best First", profit_curve, height=3.2, y="pct"))
        section.side.append(
            '<p class="explain">The <b>trade-by-trade</b> curve climbs through the winners and falls through the losers to '
            '100%. The higher its peak, the more the losers took back from a few large wins.</p>')

    def bootstrap(ax):
        ax.hist(totals, bins=50, color=STRATEGY_COLOR, alpha=0.75)
        ax.axvline(0, color=LOSS_COLOR, lw=1.4, ls="--", label="Break-even")
        ax.axvline(total, color="black", lw=1.4, label="This backtest")
        from matplotlib.ticker import FuncFormatter

        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        ax.set_xlabel(f"Net PnL of the redrawn trades, {unit()}")
        ax.set_ylabel("Draws")
        ax.legend(loc="upper left")

    section.chart(chart("Luck Test: 5,000 Redraws of the Same Trades", bootstrap, height=3.2))

    if total > 0:
        share = top / total
        section.finding(WARN if share >= 0.5 else OK, f"Profit from the best {BEST_TRADES} trades", f"{share:.0%}",
                        f"{len(pnl):,} closed trades in all; without the best {BEST_TRADES}, {money(total - top)}. "
                        "Over 100% means the rest lost money together.")
    status = BAD if loss >= BAD_LOSS else WARN if loss >= WARN_LOSS else OK
    section.finding(status, "Luck test: chance of a loss", f"{loss:.0%}",
                    f"Redrawing the same trades at random, the total is a loss {loss:.0%} of the time; a bad "
                    f"case (5th percentile) {money(low, 0)}, a good one {money(high, 0)}.")
    return section
