"""Market regimes: what the strategy does when its market rises, falls, or turns volatile."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..page import OK, WARN, Section
from ..style import BENCHMARK_COLOR, LOSS_COLOR, STRATEGY_COLOR, chart
from ..tables import Table

#: The four regimes, in the order they are shown.
REGIMES = ("Rising, calm", "Rising, volatile", "Falling, calm", "Falling, volatile")
MIN_MONTHS = 6


def monthly_regimes(returns: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """One row per month: strategy and benchmark returns, and the benchmark's regime.

    A month is *rising* or *falling* by the benchmark's return over it, and
    *volatile* when the benchmark's daily volatility in it is above its median
    month's -- so half the months are volatile by construction, and the split
    is the strategy's own market's, not an outside index's.
    """
    daily = pd.DataFrame({"strategy": returns, "benchmark": benchmark}).dropna()
    grouped = daily.groupby(daily.index.to_period("M") if daily.index.tz is None
                            else daily.index.tz_convert(None).to_period("M"))
    monthly = grouped.apply(lambda month: (1 + month).prod() - 1)
    monthly["volatility"] = grouped["benchmark"].std()
    volatile = monthly["volatility"] > monthly["volatility"].median()
    rising = monthly["benchmark"] > 0
    monthly["regime"] = np.select(
        [rising & ~volatile, rising & volatile, ~rising & ~volatile],
        [REGIMES[0], REGIMES[1], REGIMES[2]], default=REGIMES[3])
    return monthly


def capture(monthly: pd.DataFrame, rising: bool) -> float:
    """Up or down capture: the strategy's geometric mean monthly return over the benchmark's, in its months."""
    months = monthly[monthly["benchmark"] > 0] if rising else monthly[monthly["benchmark"] <= 0]
    if months.empty:
        return float("nan")

    def geometric(series):
        return (1 + series).prod() ** (1 / len(series)) - 1

    base = geometric(months["benchmark"])
    return geometric(months["strategy"]) / base if base else float("nan")


def asymmetric_beta(returns: pd.Series, benchmark: pd.Series, rising: bool) -> float:
    """Beta to the benchmark on its up days only, or its down days only."""
    daily = pd.DataFrame({"strategy": returns, "benchmark": benchmark}).dropna()
    days = daily[daily["benchmark"] > 0] if rising else daily[daily["benchmark"] < 0]
    if len(days) < 20 or days["benchmark"].var() == 0:
        return float("nan")
    return float(np.polyfit(days["benchmark"], days["strategy"], 1)[0])


#: Worst days this many times as bad as the benchmark's need a look.
BAD_DAYS_RATIO = 1.25


def expected_shortfall(returns: pd.Series, level: float = 0.05) -> float:
    """The average of the worst ``level`` share of daily returns."""
    cut = returns.quantile(level)
    return float(returns[returns <= cut].mean())


def build_market_regimes(returns: pd.Series, benchmark: pd.Series | None) -> Section | None:
    if benchmark is None:
        return None
    monthly = monthly_regimes(returns, benchmark)
    if len(monthly) < MIN_MONTHS:
        return None
    name = benchmark.name or "Benchmark"
    section = Section("Market Regimes")

    up, down = capture(monthly, True), capture(monthly, False)
    beta_up, beta_down = asymmetric_beta(returns, benchmark, True), asymmetric_beta(returns, benchmark, False)
    facts = pd.Series({
        "Up capture": f"{up:.0%}",
        "Down capture": f"{down:.0%}",
        "Capture ratio (up / down)": f"{up / down:.2f}" if down and np.isfinite(down) else "-",
        "Beta on up days": f"{beta_up:.2f}",
        "Beta on down days": f"{beta_down:.2f}",
        f"Months {name} rose": f"{int((monthly['benchmark'] > 0).sum()):,} of {len(monthly):,}",
    })
    section.table(Table(facts.to_frame("Value"), None), title=f"Against {name}", first_header="Measure")
    tail, market_tail = expected_shortfall(returns), expected_shortfall(benchmark)
    if market_tail < 0:
        section.finding(OK if tail >= BAD_DAYS_RATIO * market_tail else WARN, "Bad days: average of the worst 5%",
                        f"{tail:.1%} vs {market_tail:.1%}",
                        f"The average loss on the strategy's worst 5% of days (expected shortfall) against "
                        f"{name}'s. More than {BAD_DAYS_RATIO:g}x {name}'s needs a look.")
    if np.isfinite(up) and np.isfinite(down):
        better = down <= 0 or up / down >= 1
        section.finding(OK if better else WARN, "Up / down capture", f"{up:.0%} / {down:.0%}",
                        f"Against {name}: it takes more of the market's rises than of its falls." if better else
                        f"Against {name}: it takes more of the market's falls than of its rises.")
    section.side.append(
        f'<p class="explain"><b>Up capture</b> is how much of {name}\'s rise the strategy took in '
        f'the months {name} rose -- 120% is a fifth more than the market, 60% a bit over half -- and '
        f'<b>down capture</b> how much of its fall it took when it fell; under 100% is protection. '
        'A ratio above 1 is the asymmetry a strategy wants. The two <b>betas</b> are the same '
        'question day by day: a strategy that falls with the market but lags it up has a down beta '
        'above its up beta.</p>')

    by_regime = monthly.groupby("regime").agg(
        months=("strategy", "size"), strategy=("strategy", "mean"), benchmark=("benchmark", "mean"),
        beat=("strategy", lambda series: (series > monthly.loc[series.index, "benchmark"]).mean()),
    ).reindex(REGIMES).dropna(subset=["months"])
    shown = by_regime.rename(columns={"months": "Months", "strategy": "Strategy", "benchmark": name,
                                      "beat": f"Beat {name}"})
    section.table(Table(shown, None, None, {"Months": "{:,.0f}".format, "Strategy": "{:.2%}".format,
                                            name: "{:.2%}".format, f"Beat {name}": "{:.0%}".format}),
                  title="Average Month, by Regime", first_header="Regime")
    section.side.append(
        f'<p class="caption">A month is rising or falling by {name}\'s return, and volatile when '
        f'{name}\'s daily volatility in it is above the median month\'s.</p>')

    def scatter(ax):
        from matplotlib.ticker import FuncFormatter

        for regime, color, marker in zip(REGIMES, (STRATEGY_COLOR, "#4FA487", BENCHMARK_COLOR, LOSS_COLOR),
                                         ("o", "s", "o", "s")):
            months = monthly[monthly["regime"] == regime]
            ax.scatter(months["benchmark"], months["strategy"], s=26, color=color, marker=marker,
                       label=regime, alpha=0.85, linewidths=0)
        limits = np.array([monthly["benchmark"].min(), monthly["benchmark"].max()])
        ax.plot(limits, limits, ls="--", lw=1, color="#999999", label=f"Same as {name}")
        for rising in (True, False):
            months = monthly[monthly["benchmark"] > 0] if rising else monthly[monthly["benchmark"] <= 0]
            if len(months) >= 3:
                slope, intercept = np.polyfit(months["benchmark"], months["strategy"], 1)
                xs = np.linspace(0, limits[1] if rising else limits[0], 20)
                ax.plot(xs, slope * xs + intercept, lw=1.8, color="black",
                        ls="-" if rising else ":", label=f"{'Up' if rising else 'Down'} months fit ({slope:.2f})")
        ax.axhline(0, color="#666666", lw=0.8)
        ax.axvline(0, color="#666666", lw=0.8)
        percent = FuncFormatter(lambda value, _: f"{value:.0%}")
        ax.xaxis.set_major_formatter(percent)
        ax.yaxis.set_major_formatter(percent)
        ax.set_xlabel(f"{name}, month")
        ax.set_ylabel("Strategy, month")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, fontsize=9)

    section.chart(chart(f"Monthly Returns against {name}", scatter, height=4.2))

    def bars(ax):
        positions = np.arange(len(by_regime))
        ax.bar(positions - 0.2, by_regime["strategy"], 0.4, color=STRATEGY_COLOR, label="Strategy")
        ax.bar(positions + 0.2, by_regime["benchmark"], 0.4, color=BENCHMARK_COLOR, label=name)
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(positions)
        ax.set_xticklabels([regime.replace(", ", "\n") for regime in by_regime.index])
        ax.legend(loc="upper right")
        ax.set_ylabel("Average month")

    section.chart(chart("Average Month, by Regime", bars, height=3.2, y="pct"))

    def timeline(ax):
        cumulative = pd.DataFrame({"Strategy": (1 + returns).cumprod(), name: (1 + benchmark).cumprod()})
        cumulative.index = cumulative.index.tz_convert(None) if cumulative.index.tz is not None else cumulative.index
        for period, regime in monthly["regime"].items():
            if regime.endswith("volatile"):
                start, end = period.start_time, period.end_time
                ax.axvspan(start, end, color=LOSS_COLOR if regime.startswith("Falling") else "#4FA487",
                           alpha=0.12, linewidth=0)
        ax.plot(cumulative.index, cumulative["Strategy"], color=STRATEGY_COLOR, lw=1.8, label="Strategy")
        ax.plot(cumulative.index, cumulative[name], color=BENCHMARK_COLOR, lw=1.4, label=name)
        ax.axhline(1.0, ls="--", lw=1, color="black", zorder=1)
        ax.legend(loc="upper left")

    section.chart(chart("Cumulative Returns through the Regimes", timeline, height=3.6, y="growth"))
    section.side.append(
        '<p class="caption">On the last chart, shaded months are volatile ones: red when the '
        'market fell, green when it rose.</p>')
    return section
