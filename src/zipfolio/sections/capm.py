"""CAPM and the efficient frontier: the strategy among the risk and return that were on offer.

Not a check. The capital asset pricing model and Markowitz's frontier are the
two classic maps of risk against return, and a backtest reads differently once
it is placed on them: how much of its return its beta alone would have paid,
whether it beat the market per unit of risk, and how it compares with the
best fixed mix of its own names -- a mix only hindsight could have chosen.
"""

from __future__ import annotations

import html
import math
import traceback
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..interactive import ITEM_TIPS, TIPS
from ..page import INFO, Section
from ..style import BENCHMARK_COLOR, LOSS_COLOR, STRATEGY_COLOR, chart, percent_ticks
from ..tables import Table
from .portfolio_risk import daily_asset_returns

TITLE = "CAPM and the Efficient Frontier"
#: The fact the frontier leaves for the summary; the buy-and-hold check quotes it (``tearsheet``).
FRONTIER_FACT = "Against the efficient frontier"
#: Sessions a year, as everywhere on the page.
YEAR = 252
#: Fewer sessions than this and a beta or a covariance is noise.
MIN_SESSIONS = 60
#: The frontier is drawn over the names held most, at most this many -- and at most a third as many as
#: there are sessions: with more names than that, a sample covariance finds mixes that look riskless.
FRONTIER_NAMES = 30
FRONTIER_POINTS = 40
#: Names are written beside their dots up to this many; past it, only the tooltips name them.
LABELLED_NAMES = 12
#: The heaviest names of the max-Sharpe mix listed beside the chart.
MIX_ROWS = 10

FRONTIER_COLOR = "#4A5899"
TANGENCY_COLOR = "#9B59B6"
MIN_RISK_COLOR = "#4FA487"
EQUAL_COLOR = "#559CAD"
AVERAGE_COLOR = "#DC136C"
NAME_COLOR = "#b8b8b8"


@dataclass
class Point:
    """A portfolio on the risk-return map: annual mean return and volatility, and its beta when known."""

    label: str
    mean: float
    vol: float
    beta: float | None = None

    def sharpe(self, rate: float) -> float:
        return (self.mean - rate) / self.vol if self.vol > 0 else math.nan


@dataclass
class Frontier:
    """The long-only efficient frontier of a set of names, and the mixes on and inside it."""

    curve: list[Point]
    tangency: Point | None
    tangency_weights: pd.Series | None
    minimum: Point
    equal: Point
    average: Point | None


def build_capm(returns: pd.Series, benchmark: pd.Series | None = None, risk_free: pd.Series | None = None,
               risk_free_name: str = "", positions: pd.DataFrame | None = None, bars=None) -> Section | None:
    """The security market line, the capital market line and the efficient frontier of the traded names.

    ``risk_free`` is each session's risk-free return (``risk_free_for``); without it the rate is 0%.
    The CAPM half needs a benchmark, the frontier half daily bars of at least two held names
    (``market_data_frame``); with neither there is no section.
    """
    rf = (risk_free.reindex(returns.index).fillna(0.0) if risk_free is not None
          else pd.Series(0.0, index=returns.index))
    rate = float(rf.mean() * YEAR)
    bench_name = str(benchmark.name) if benchmark is not None and benchmark.name else "Benchmark"

    market = strategy = None
    if benchmark is not None:
        joined = pd.concat([returns.rename("s"), benchmark.rename("m"), rf.rename("rf")], axis="columns",
                           join="inner").dropna()
        if len(joined) >= MIN_SESSIONS:
            strategy = _point("Strategy", joined["s"], joined["m"], joined["rf"])
            market = _point(bench_name, joined["m"], joined["m"], joined["rf"])
    if strategy is None and len(returns) >= MIN_SESSIONS:
        strategy = _point("Strategy", returns)

    names, average = held_names(positions, bars, returns.index)
    dots: list[Point] = []
    if names is not None:
        market_series = benchmark.reindex(names.index) if market is not None else None
        for name in names.columns:
            dots.append(_point(_short(name), names[name], market_series, rf.reindex(names.index).fillna(0.0)))

    frontier = None
    if names is not None and names.shape[1] >= 2:
        try:
            frontier = efficient_frontier(names, rate, average)
        except Exception:
            print(f"Warning: the efficient frontier could not be computed:\n{traceback.format_exc()}")

    if strategy is None or (market is None and frontier is None):
        return None

    section = Section(TITLE)
    if market is not None:
        _capm_table(section, strategy, market, rate, risk_free_name, returns, benchmark)
        section.chart(_security_market_line(strategy, market, dots, rate, risk_free_name))
    section.chart(_capital_market_line(strategy, market, dots, frontier, rate))
    if frontier is not None:
        _frontier_tables(section, strategy, market, frontier, rate, risk_free_name, names.shape[1])
    _explain(section, market is not None, frontier is not None, risk_free_name, rate,
             shorted=average is not None and bool((average < 0).any()))
    _odd_period(section, market, frontier, risk_free_name, rate)
    return section


# ----------------------------------------------------------------------------
# The numbers.
# ----------------------------------------------------------------------------

def _point(label: str, series: pd.Series, market: pd.Series | None = None, rf: pd.Series | None = None) -> Point:
    """Annual mean and volatility of daily returns, and the beta of their excess over ``rf`` on the market's."""
    series = series.dropna()
    point = Point(label, float(series.mean() * YEAR), float(series.std() * math.sqrt(YEAR)))
    if market is not None:
        rf = rf.reindex(series.index).fillna(0.0) if rf is not None else 0.0
        both = pd.concat([series - rf, market.reindex(series.index) - rf], axis="columns").dropna()
        variance = float(both.iloc[:, 1].var())
        if len(both) >= MIN_SESSIONS and variance > 0:
            point.beta = float(both.iloc[:, 0].cov(both.iloc[:, 1]) / variance)
    return point


def held_names(positions: pd.DataFrame | None, bars, dates: pd.DatetimeIndex) -> tuple[pd.DataFrame | None,
                                                                                         pd.Series | None]:
    """Daily returns of the names held most, and their average weights over the book's gross value.

    At most ``FRONTIER_NAMES`` names, and no more than a third of the sessions.
    """
    asset_returns = daily_asset_returns(bars)
    if asset_returns is None or positions is None:
        return None, None
    book = positions.drop(columns="cash", errors="ignore")
    gross = book.abs().sum(axis="columns")
    weights = book.div(gross.where(gross > 0), axis="index").mean()
    weights = weights[weights.abs() > 0].reindex(asset_returns.columns).dropna()
    if weights.empty or len(asset_returns) < MIN_SESSIONS:
        return None, None
    keep = min(FRONTIER_NAMES, max(1, len(asset_returns) // 3))
    weights = weights.reindex(weights.abs().sort_values(ascending=False).index[:keep])
    return asset_returns[weights.index], weights / weights.abs().sum()


def efficient_frontier(names: pd.DataFrame, rate: float, average: pd.Series | None = None) -> Frontier:
    """Markowitz's frontier of ``names``, long only and fully invested, from their own daily returns.

    The curve runs from the minimum-variance mix to the best single name; the
    tangency mix has the highest Sharpe ratio over ``rate`` (a year). All of it
    is measured with the returns the names went on to make.
    """
    from skfolio.optimization import MeanRisk, ObjectiveFunction

    mu = names.mean().to_numpy() * YEAR
    cov = names.cov().to_numpy() * YEAR

    def point(label: str, weights) -> Point:
        weights = np.asarray(weights, dtype=float)
        return Point(label, float(weights @ mu), float(math.sqrt(max(weights @ cov @ weights, 0.0))))

    swept = np.atleast_2d(MeanRisk(efficient_frontier_size=FRONTIER_POINTS).fit(names).weights_)
    curve = sorted((point("Efficient frontier", weights) for weights in swept), key=lambda p: p.vol)
    minimum = point("Minimum variance", swept[0]) if len(swept) else curve[0]
    minimum = min(curve + [minimum], key=lambda p: p.vol)
    minimum.label = "Minimum variance"

    tangency = weights = None
    if mu.max() > rate:
        try:
            model = MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RATIO, risk_free_rate=rate / YEAR)
            weights = np.asarray(model.fit(names).weights_, dtype=float)
        except Exception:
            print(f"Note: no max-Sharpe mix from the optimiser ({traceback.format_exc(limit=1).strip()}); "
                  "the best point of the frontier stands in.")
            best = max(range(len(swept)), key=lambda i: point("", swept[i]).sharpe(rate))
            weights = swept[best]
        weights = np.clip(weights, 0.0, None)
        weights = weights / weights.sum()
        tangency = point("Max Sharpe", weights)
    equal = point("Equal weight", np.full(len(mu), 1.0 / len(mu)))
    held = point("Average weights", average.reindex(names.columns).fillna(0.0)) if average is not None else None
    return Frontier(curve, tangency, pd.Series(weights, index=names.columns) if weights is not None else None,
                    minimum, equal, held)


def _short(symbol: str) -> str:
    return str(symbol).split("@")[0]


def _pct(value: float, signed: bool = False) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:+.2%}" if signed else f"{value:.2%}"


def _ratio(value: float) -> str:
    return f"{value:.2f}" if value is not None and math.isfinite(value) else "-"


# ----------------------------------------------------------------------------
# Tables.
# ----------------------------------------------------------------------------

def _capm_table(section: Section, strategy: Point, market: Point, rate: float, rate_name: str,
                returns: pd.Series, benchmark: pd.Series) -> None:
    beta = strategy.beta if strategy.beta is not None else math.nan
    premium = market.mean - rate
    expected = rate + beta * premium
    alpha = strategy.mean - expected
    active = pd.concat([returns, benchmark], axis="columns", join="inner").dropna()
    difference = active.iloc[:, 0] - active.iloc[:, 1]
    tracking = float(difference.std() * math.sqrt(YEAR))
    correlation = float(active.iloc[:, 0].corr(active.iloc[:, 1]))
    systematic = abs(beta) * market.vol
    specific = math.sqrt(max(strategy.vol ** 2 - systematic ** 2, 0.0))
    treynor = (strategy.mean - rate) / beta if abs(beta) >= 0.05 else math.nan
    m_squared = rate + strategy.sharpe(rate) * market.vol

    rows = {
        "Annual return (mean)": (_pct(strategy.mean), _pct(market.mean)),
        "Annual volatility": (_pct(strategy.vol), _pct(market.vol)),
        "Beta": (_ratio(beta), "1.00"),
        "Correlation": (_ratio(correlation), "1.00"),
        "Variance the market explains (R²)": (f"{correlation ** 2:.0%}", "100%"),
        "Systematic volatility": (_pct(systematic), _pct(market.vol)),
        "Specific volatility": (_pct(specific), "0.00%"),
        "Return CAPM asks for its beta": (_pct(expected), _pct(market.mean)),
        "Jensen's alpha, a year": (_pct(alpha, signed=True), "0.00%"),
        "Treynor ratio": (_pct(treynor), _pct(premium)),
        "Sharpe ratio (mean, over the rate)": (_ratio(strategy.sharpe(rate)), _ratio(market.sharpe(rate))),
        "M² (at the benchmark's volatility)": (_pct(m_squared), _pct(market.mean)),
        "Tracking error": (_pct(tracking), "-"),
        "Information ratio": (_ratio((strategy.mean - market.mean) / tracking if tracking else math.nan), "-"),
    }
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=["Strategy", market.label])
    label = f"Risk-free rate ({rate_name}), a year" if rate_name else "Risk-free rate"
    section.table(Table(frame, None, header_rows={label: _pct(rate) if rate_name else "0% (none to be had)"}),
                  title="CAPM Measures", first_header="Measure")
    section.finding(INFO, "CAPM", f"beta {beta:.2f}, Jensen's alpha {alpha:+.1%} a year",
                    f"Against {market.label}{f' over {rate_name}' if rate_name else ''}: the return the beta "
                    f"asks for is {expected:.1%} a year and the strategy made {strategy.mean:.1%} (means of "
                    f"daily returns). Sharpe {strategy.sharpe(rate):.2f} against the market's "
                    f"{market.sharpe(rate):.2f}.")


def _frontier_tables(section: Section, strategy: Point, market: Point | None, frontier: Frontier, rate: float,
                     rate_name: str, count: int) -> None:
    mixes = [strategy] + ([market] if market is not None else []) + [
        frontier.equal, *([frontier.average] if frontier.average is not None else []), frontier.minimum,
        *([frontier.tangency] if frontier.tangency is not None else [])]
    labels = {"Equal weight": "Equal weight, held fixed", "Average weights": "Strategy's average weights, held fixed",
              "Max Sharpe": "Max Sharpe (hindsight)", "Minimum variance": "Minimum variance (hindsight)"}
    frame = pd.DataFrame(
        {"Return": [_pct(p.mean) for p in mixes], "Volatility": [_pct(p.vol) for p in mixes],
         "Sharpe": [_ratio(p.sharpe(rate)) for p in mixes]},
        index=[labels.get(p.label, p.label) for p in mixes])
    section.table(Table(frame, None), title="Fixed Mixes of the Same Names", first_header="Portfolio")
    if frontier.tangency_weights is not None:
        mix = frontier.tangency_weights[frontier.tangency_weights >= 0.005].sort_values(ascending=False)
        shown = mix.head(MIX_ROWS)
        table = shown.rename(index=_short).to_frame("Weight")
        if len(mix) > len(shown):
            table.loc[f"{len(mix) - len(shown)} more"] = mix.iloc[len(shown):].sum()
        section.table(Table(table, None, "{:.1%}".format), title="The Max Sharpe Mix, in Hindsight",
                      first_header="Name")

    best = frontier.tangency.sharpe(rate) if frontier.tangency is not None else math.nan
    mine = strategy.sharpe(rate)
    on_curve = _frontier_return_at(frontier.curve, strategy.vol)
    where = ("above the frontier: its timing beat every fixed mix of its names"
             if on_curve is not None and strategy.mean > on_curve else
             f"{on_curve - strategy.mean:.1%} a year below the frontier at its volatility"
             if on_curve is not None else "outside the frontier's range of volatility")
    basis = f"over {rate_name}" if rate_name else "at 0%"
    best_text = (f"the best fixed mix, chosen in hindsight, {best:.2f}" if math.isfinite(best) else
                 "no fixed mix of them beat the rate")
    section.finding(INFO, FRONTIER_FACT,
                    f"Sharpe {mine:.2f} vs {best:.2f} for the best fixed mix" if math.isfinite(best) else
                    f"Sharpe {mine:.2f}; no mix of these names beat the risk-free rate",
                    f"On the efficient frontier of the same {count} names (Sharpe {basis}, as the frontier measures "
                    f"it): the strategy {mine:.2f}, equal weight {frontier.equal.sharpe(rate):.2f}, {best_text}. "
                    f"The strategy sits {where}.")


def _frontier_return_at(curve: list[Point], vol: float) -> float | None:
    """The frontier's return at a volatility, or ``None`` outside the range it spans."""
    vols = [p.vol for p in curve]
    if not curve or vol < min(vols) or vol > max(vols):
        return None
    return float(np.interp(vol, vols, [p.mean for p in curve]))


# ----------------------------------------------------------------------------
# Charts.
# ----------------------------------------------------------------------------

def _tip(point: Point, rate: float, extra: str = "") -> str:
    beta = f"<br>Beta {point.beta:.2f}" if point.beta is not None else ""
    return (f"<b>{point.label}</b><br>Return {point.mean:.1%} a year<br>Volatility {point.vol:.1%}<br>"
            f"Sharpe {point.sharpe(rate):.2f}{beta}{extra}")


def _mark(ax, point: Point, rate: float, marker: str, color: str, size: float, extra: str = "",
          zorder: int = 4, x: str = "vol"):
    """One portfolio as a marker, placed by its volatility or its beta (``x``), with its own tooltip."""
    line = ax.plot([point.vol if x == "vol" else point.beta], [point.mean], ls="none", marker=marker,
                   ms=size, color=color, label=point.label, zorder=zorder)[0]
    setattr(line, TIPS, [_tip(point, rate, extra)])
    return line


def _names_on(ax, dots: list[Point], rate: float, x: str) -> None:
    """The traded names as light dots, written beside them when there are few."""
    shown = [dot for dot in dots if x == "vol" or dot.beta is not None]
    if not shown:
        return
    xs = [dot.vol if x == "vol" else dot.beta for dot in shown]
    line = ax.plot(xs, [dot.mean for dot in shown], ls="none", marker="o", ms=5, color=NAME_COLOR,
                   label="Traded names", zorder=3)[0]
    setattr(line, TIPS, [_tip(dot, rate) for dot in shown])
    if len(shown) <= LABELLED_NAMES:
        for dot, at in zip(shown, xs):
            ax.text(at, dot.mean, f" {dot.label}", ha="left", va="center", fontsize=8, color="#888888")


def _label_room(dots: list[Point], span: float) -> float:
    """Room on the right for the names written beside the dots furthest out."""
    return 0.07 * span if 0 < len(dots) <= LABELLED_NAMES else 0.0


def _limits(values: list[float], floor: float | None = None, pad: float = 0.08) -> tuple[float, float]:
    values = [v for v in values if v is not None and math.isfinite(v)]
    low, high = min(values), max(values)
    if floor is not None:
        low = min(low, floor)
    span = (high - low) or abs(high) or 0.1
    return low - pad * span, high + pad * span


def _security_market_line(strategy: Point, market: Point, dots: list[Point], rate: float,
                          rate_name: str) -> str | None:
    from matplotlib.ticker import FuncFormatter

    from ..style import percent_axis

    beta = strategy.beta if strategy.beta is not None else math.nan
    if not math.isfinite(beta):
        return None
    premium = market.mean - rate
    expected = rate + beta * premium
    alpha = strategy.mean - expected

    def draw(ax):
        setattr(ax, ITEM_TIPS, True)
        betas = [beta, 1.0, 0.0] + [dot.beta for dot in dots if dot.beta is not None]
        x_low, x_high = _limits(betas, pad=0.12)
        x_high += _label_room(dots, x_high - x_low)
        means = [strategy.mean, market.mean, rate, expected] + [dot.mean for dot in dots if dot.beta is not None]
        y_low, y_high = _limits(means + [rate + x_low * premium, rate + x_high * premium])
        xs = np.array([x_low, x_high])
        ax.plot(xs, rate + xs * premium, color=BENCHMARK_COLOR, lw=1.4, label="Security market line", zorder=2)
        _names_on(ax, dots, rate, "beta")
        free = Point(f"Risk-free ({rate_name})" if rate_name else "Risk-free (0%)", rate, 0.0, 0.0)
        line = ax.plot([0.0], [rate], ls="none", marker="o", ms=6, color="#999999", zorder=4)[0]
        setattr(line, TIPS, [f"<b>{free.label}</b><br>{rate:.2%} a year, beta 0"])
        _mark(ax, market, rate, "s", BENCHMARK_COLOR, 9, x="beta")
        color = STRATEGY_COLOR if alpha >= 0 else LOSS_COLOR
        ax.plot([beta, beta], [expected, strategy.mean], ls="--", lw=1.3, color=color, zorder=3)
        # Left of the gap: the names' labels run to the right of their dots.
        ax.text(beta, (expected + strategy.mean) / 2, f"alpha {alpha:+.1%} ", ha="right", va="center",
                fontsize=9, color=color, fontweight="bold")
        _mark(ax, strategy, rate, "D", STRATEGY_COLOR, 11, x="beta", zorder=5,
              extra=f"<br>CAPM asks {expected:.1%} for this beta<br>Jensen's alpha <b>{alpha:+.1%}</b>")
        ax.set_xlim(x_low, x_high)
        ax.set_ylim(y_low, y_high)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}"))
        percent_axis(ax)
        ax.set_xlabel(f"Beta to {market.label}")
        ax.set_ylabel("Annual return (mean)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False, fontsize=9)

    return chart("Security Market Line", draw, height=4.2)


def _capital_market_line(strategy: Point, market: Point | None, dots: list[Point], frontier: Frontier | None,
                         rate: float) -> str | None:

    from ..style import percent_axis

    mixes = [] if frontier is None else [frontier.equal, frontier.minimum] + [
        p for p in (frontier.average, frontier.tangency) if p is not None]
    curve = frontier.curve if frontier is not None else []
    points = [strategy] + ([market] if market is not None else []) + dots + mixes + curve

    def draw(ax):
        setattr(ax, ITEM_TIPS, True)
        _, x_high = _limits([p.vol for p in points] + [0.0], pad=0.1)
        x_high += _label_room(dots, x_high)
        y_low, y_high = _limits([p.mean for p in points] + [rate], floor=min(0.0, rate))

        def ray(through: Point, color: str, style: str, label: str) -> None:
            slope = through.sharpe(rate)
            if not math.isfinite(slope):
                return
            end = x_high if slope <= 0 else min(x_high, (y_high - rate) / slope)
            ax.plot([0.0, end], [rate, rate + slope * end], color=color, ls=style, lw=1.4, label=label, zorder=2)

        if market is not None:
            ray(market, BENCHMARK_COLOR, ":", "Capital market line")
        if frontier is not None and frontier.tangency is not None:
            ray(frontier.tangency, TANGENCY_COLOR, "--", "Best mix + cash")
        if curve:
            line = ax.plot([p.vol for p in curve], [p.mean for p in curve], color=FRONTIER_COLOR, lw=2.2,
                           label="Efficient frontier", zorder=2)[0]
            setattr(line, TIPS, [_tip(p, rate) for p in curve])
        _names_on(ax, dots, rate, "vol")
        if frontier is not None:
            _mark(ax, frontier.equal, rate, "o", EQUAL_COLOR, 8)
            if frontier.average is not None:
                _mark(ax, frontier.average, rate, "p", AVERAGE_COLOR, 9)
            _mark(ax, frontier.minimum, rate, "^", MIN_RISK_COLOR, 9)
            if frontier.tangency is not None:
                _mark(ax, frontier.tangency, rate, "^", TANGENCY_COLOR, 10)
        if market is not None:
            _mark(ax, market, rate, "s", BENCHMARK_COLOR, 9)
        _mark(ax, strategy, rate, "D", STRATEGY_COLOR, 11, zorder=5)
        ax.set_xlim(0.0, x_high)
        ax.set_ylim(y_low, y_high)
        ax.xaxis.set_major_formatter(percent_ticks())
        percent_axis(ax)
        ax.set_xlabel("Annual volatility")
        ax.set_ylabel("Annual return (mean)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, fontsize=9)

    title = "Capital Market Line and Efficient Frontier" if frontier is not None else "Capital Market Line"
    return chart(title, draw, height=4.8 if frontier is not None else 4.2)


# ----------------------------------------------------------------------------
# Words.
# ----------------------------------------------------------------------------

def _odd_period(section: Section, market: Point | None, frontier: Frontier | None, rate_name: str,
                rate: float) -> None:
    """What the maps mean in a period when holding the market, or any mix of the names, lost to cash."""
    rate_text = f"{rate_name} ({rate:.2%} a year)" if rate_name else f"the risk-free rate ({rate:.2%})"
    if market is not None and market.mean < rate:
        section.side.append(
            f'<p class="caption">Over this backtest {html.escape(market.label)} made {market.mean:.2%} a year, less '
            f'than {html.escape(rate_text)}, so the security market line slopes down: the market paid for being '
            'out of it, and every unit of beta cost return. The capital market line slopes down for the same '
            'reason.</p>')
    if frontier is not None and frontier.tangency is None:
        section.side.append(
            f'<p class="caption">No fixed mix of these names made more than {html.escape(rate_text)}, so there is '
            'no max-Sharpe mix and no line from the rate touching the frontier: cash at the rate beat every '
            'mix of them.</p>')


def _explain(section: Section, capm: bool, frontier: bool, rate_name: str, rate: float, shorted: bool) -> None:
    if capm:
        section.side.append(
            '<p class="explain">The <b>capital asset pricing model</b> (CAPM) holds that the market pays for one '
            'risk only: the part that moves with the market, measured by <b>beta</b>. Risk a portfolio could have '
            'diversified away earns nothing. The <b>security market line</b> is the return the model asks for each '
            'beta -- the risk-free rate at 0, the benchmark\'s return at 1. A point above the line made more than '
            'its beta asks for; the gap is <b>Jensen\'s alpha</b>.</p>'
            '<p class="explain">The <b>capital market line</b> runs from the risk-free rate through the benchmark: '
            'every mix of the market and cash sits on it. Above it is a better Sharpe ratio than the market\'s. '
            '<b>M&sup2;</b> is the strategy\'s return scaled to the benchmark\'s volatility, so the two compare in '
            'percent; the <b>Treynor ratio</b> is excess return per unit of beta rather than of volatility. '
            '<b>Specific volatility</b> is the risk the market does not explain, the part CAPM says goes '
            'unpaid.</p>')
    if frontier:
        section.side.append(
            '<p class="explain">The <b>efficient frontier</b> is the highest return any fixed mix of the traded '
            'names offered at each volatility -- long only, fully invested -- and the line from the risk-free rate '
            'that touches it marks the mix with the best Sharpe ratio. It is measured with the returns the names '
            'went on to make, so it is <b>hindsight</b>: no one could have held it, and it is a ceiling, not a '
            'target. A strategy that changes its holdings can still land above it -- that is what timing is for. '
            'One well inside the equal-weight point did worse than the simplest fixed mix of its own names.</p>')
    rate_text = (f"The risk-free rate is {rate_name}, {rate:.2%} a year on average." if rate_name else
                 "No risk-free rate could be had, so it is 0%.")
    section.side.append(
        '<p class="caption">Returns here are means of daily returns times 252, and volatility the daily one '
        'times &radic;252, as mean-variance theory has them -- not the compounded returns of Key Performance '
        f'Metrics, whose alpha is at a 0% rate, as QuantStats has it. {rate_text}'
        + (' The book also went short; the frontier is long only, so a long-short book can sit beyond it '
           'without any hindsight.' if shorted else "") + '</p>')
