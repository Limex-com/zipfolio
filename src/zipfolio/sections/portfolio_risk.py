"""skfolio's risk measures, risk by holding, and other allocations of the same names."""

from __future__ import annotations

import traceback

import numpy as np
import pandas as pd

from ..page import Section
from ..style import BENCHMARK_COLOR, LOSS_COLOR, STRATEGY_COLOR, chart
from ..tables import Table

# Risk measures from skfolio, as (label, attribute, kind). "loss" measures are
# reported by skfolio as positive numbers and shown here as negative, the way
# pyfolio and QuantStats show a drawdown.
_SKFOLIO_MEASURES = [
    ("Annualized mean", "annualized_mean", "pct"),
    ("Annualized volatility", "annualized_standard_deviation", "pct"),
    ("Annualized semi-deviation", "annualized_semi_deviation", "pct"),
    ("Mean absolute deviation (daily)", "mean_absolute_deviation", "pct"),
    ("Gini mean difference (daily)", "gini_mean_difference", "pct"),
    ("VaR 95% (daily)", "value_at_risk", "loss"),
    ("CVaR 95% (daily)", "cvar", "loss"),
    ("EVaR 95% (daily)", "evar", "loss"),
    ("Worst day", "worst_realization", "loss"),
    ("Max drawdown", "max_drawdown", "loss"),
    ("Average drawdown", "average_drawdown", "loss"),
    ("Drawdown at risk 95%", "drawdown_at_risk", "loss"),
    ("CDaR 95%", "cdar", "loss"),
    ("EDaR 95%", "edar", "loss"),
    ("Ulcer index", "ulcer_index", "number"),
    ("Sharpe (annualized)", "annualized_sharpe_ratio", "number"),
    ("Sortino (annualized)", "annualized_sortino_ratio", "number"),
]


_RISK_GLOSSARY = (
    "<b>VaR 95%</b> is the daily loss exceeded on one day in twenty; <b>CVaR</b> is the "
    "average loss on those days. <b>EVaR</b> (entropic VaR) is an upper bound on both, "
    "derived from the whole distribution rather than its tail alone, so it reacts more "
    "to fat tails. <b>Drawdown at risk</b>, <b>CDaR</b> and <b>EDaR</b> are the same three "
    "ideas applied to drawdowns: the depth exceeded on the 5% of days furthest below a "
    "peak, the average depth on those days, and its entropic bound. The <b>Ulcer "
    "index</b> is the root mean square of all drawdowns, so it grows with both depth and "
    "duration. The <b>Gini mean difference</b> is the average gap between any two daily "
    "returns -- a dispersion measure that, unlike volatility, is not dominated by a few "
    "extreme days."
)


def build_portfolio_risk(returns, benchmark, positions, bars) -> Section | None:
    """skfolio's view of the backtest: risk measures, risk by holding, other allocations of the same names."""
    try:
        import skfolio  # noqa: F401
    except ImportError:
        print("skfolio is not installed; the portfolio risk section is left out.")
        return None

    section = Section("Portfolio Risk")
    columns = {"Strategy": returns}
    if benchmark is not None:
        columns[benchmark.name or "Benchmark"] = benchmark
    measured = {name: _skfolio_portfolio(series.to_frame(name), [1.0], compounded=True, name=name)
                for name, series in columns.items()}
    rows = {}
    for label, attribute, kind in _SKFOLIO_MEASURES:
        values = []
        for portfolio in measured.values():
            value = float(getattr(portfolio, attribute))
            values.append(f"{-value:.2%}" if kind == "loss" else f"{value:.2%}" if kind == "pct" else f"{value:.2f}")
        rows[label] = values
    table = pd.DataFrame.from_dict(rows, orient="index", columns=list(measured))
    section.table(Table(table, None), title="Risk Measures", first_header="Measure")
    section.side.append(f'<p class="caption">{_RISK_GLOSSARY}</p>')
    section.chart(_rolling_tail_risk(columns))

    asset_returns = _asset_returns(bars)
    if positions is None or asset_returns is None:
        return section

    weights = (positions.drop(columns="cash").div(positions.sum(axis="columns"), axis="index")
               .reindex(columns=asset_returns.columns).fillna(0.0).mean())
    _risk_contribution(section, asset_returns, weights)
    if asset_returns.shape[1] >= 3:
        _holding_correlations(section, asset_returns, weights)
    if asset_returns.shape[1] >= 2 and len(asset_returns) >= _WALK_FORWARD_TRAIN + 2 * _WALK_FORWARD_TEST:
        _alternative_allocations(section, returns, asset_returns)
    return section


def _skfolio_portfolio(returns: pd.DataFrame, weights, **kwargs):
    """A skfolio ``Portfolio`` annualised over 252 sessions, under either spelling of the argument.

    The images pin skfolio 0.15.2, which calls it ``annualized_factor``; 1.x
    renamed it ``annualization_factor`` and deprecates the old name.
    """
    import inspect

    from skfolio import Portfolio

    parameters = inspect.signature(Portfolio.__init__).parameters
    factor = "annualization_factor" if "annualization_factor" in parameters else "annualized_factor"
    return Portfolio(X=returns, weights=weights, **{factor: 252}, **kwargs)


def _asset_returns(bars) -> pd.DataFrame | None:
    """Daily returns of the traded names from the bundle's closes; ``None`` when there is too little."""
    if bars is None:
        return None
    prices = bars.xs("price", level=1).astype(float)
    returns = prices.pct_change(fill_method=None).iloc[1:]
    # A name with no bars for most of the window would be all zeros, not a risk.
    returns = returns.loc[:, returns.notna().mean() >= 0.5].fillna(0.0)
    return returns if returns.shape[1] and len(returns) > 20 else None


def _risk_contribution(section: Section, asset_returns: pd.DataFrame, weights: pd.Series) -> None:
    from skfolio import RiskMeasure

    portfolio = _skfolio_portfolio(asset_returns, weights.to_numpy())
    shares = pd.DataFrame({
        "Avg. Weight": weights / weights.sum(),
        "Volatility Share": _shares(portfolio.contribution(measure=RiskMeasure.STANDARD_DEVIATION)),
        "CVaR Share": _shares(portfolio.contribution(measure=RiskMeasure.CVAR)),
    }, index=asset_returns.columns).sort_values("Volatility Share", ascending=False)
    shares = shares[shares["Avg. Weight"] > 0]
    section.table(Table(shares.head(15), None, "{:.1%}".format), title="Risk Contribution", first_header="Symbol")
    section.side.append(
        '<p class="caption">Each holding\'s share of the portfolio\'s volatility and of its CVaR, '
        'for the average weights held over the backtest. A name whose risk share is well above '
        'its weight is where the risk is concentrated.</p>')

    top = shares.head(12).iloc[::-1]

    def draw(ax):
        positions_y = np.arange(len(top))
        height = 0.27
        ax.barh(positions_y + height, top["Avg. Weight"], height, color=BENCHMARK_COLOR, label="Avg. weight")
        ax.barh(positions_y, top["Volatility Share"], height, color=STRATEGY_COLOR, label="Volatility share")
        ax.barh(positions_y - height, top["CVaR Share"], height, color=LOSS_COLOR, label="CVaR share")
        ax.set_yticks(positions_y)
        ax.set_yticklabels(top.index)
        ax.legend(loc="lower right")
        from matplotlib.ticker import FuncFormatter

        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))

    section.chart(chart("Risk Contribution by Holding", draw, height=max(3.0, 0.42 * len(top) + 1.0)))


ROLLING_RISK_WINDOW = 126
CORRELATION_NAMES = 20


def _rolling_tail_risk(columns: dict[str, pd.Series]) -> str | None:
    """Daily VaR and CVaR at 95% over a moving half-year, strategy against benchmark."""
    strategy = next(iter(columns.values()))
    if len(strategy) <= ROLLING_RISK_WINDOW + 20:
        return None

    def cvar(window: np.ndarray) -> float:
        cutoff = np.quantile(window, 0.05)
        return window[window <= cutoff].mean()

    def draw(ax):
        for (name, series), color in zip(columns.items(), (STRATEGY_COLOR, BENCHMARK_COLOR)):
            rolling = series.rolling(ROLLING_RISK_WINDOW)
            ax.plot(rolling.apply(cvar, raw=True), color=color, lw=1.6, label=f"{name} CVaR 95%")
            ax.plot(rolling.quantile(0.05), color=color, lw=1.0, ls="--", label=f"{name} VaR 95%")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
        ax.set_ylabel("Daily loss")

    return chart(f"Rolling Tail Risk ({ROLLING_RISK_WINDOW} Sessions)", draw, height=3.2, y="pct")


def _holding_correlations(section: Section, asset_returns: pd.DataFrame, weights: pd.Series) -> None:
    """Correlations of the most-held names, ordered so that clusters sit together."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    names = weights[weights > 0].sort_values(ascending=False).index[:CORRELATION_NAMES]
    correlation = asset_returns[names].corr().fillna(0.0)
    distance = np.sqrt(np.clip((1 - correlation.to_numpy()) / 2, 0, None))
    np.fill_diagonal(distance, 0.0)
    order = leaves_list(linkage(squareform(distance, checks=False), method="average"))
    correlation = correlation.iloc[order, order]
    pairs = correlation.to_numpy()[np.triu_indices(len(correlation), 1)]

    def draw(ax):
        import seaborn as sns

        sns.heatmap(correlation, ax=ax, cmap="RdYlGn_r", vmin=-1, vmax=1, center=0,
                    annot=len(correlation) <= 12, fmt=".2f", annot_kws={"size": 8}, cbar=True,
                    cbar_kws={"shrink": 0.7}, linewidths=0.5, linecolor="white")
        ax.set_xlabel("")
        ax.set_ylabel("")

    section.chart(chart("Correlation Between Holdings (Clustered)", draw,
                        height=max(3.4, 0.3 * len(correlation) + 1.4)))
    section.side.append(
        f'<p class="caption">Daily-return correlations of the {len(correlation)} most-held names, '
        f'average pair {pairs.mean():.2f}. Clustering puts names that move together side by side: '
        'a block of high correlation is one bet held several times.</p>')


def _shares(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    total = values.sum()
    return values / total if total else values


_WALK_FORWARD_TRAIN = 252


_WALK_FORWARD_TEST = 21


def _alternative_allocations(section: Section, returns: pd.Series, asset_returns: pd.DataFrame) -> None:
    """The same names under four standard allocation rules, refitted walk-forward."""
    import empyrical as ep
    from skfolio.model_selection import WalkForward, cross_val_predict
    from skfolio.optimization import (
        EqualWeighted,
        HierarchicalRiskParity,
        InverseVolatility,
        MeanRisk,
        ObjectiveFunction,
    )

    cv = WalkForward(train_size=_WALK_FORWARD_TRAIN, test_size=_WALK_FORWARD_TEST)
    rules = {
        "Equal Weighted": EqualWeighted(),
        "Inverse Volatility": InverseVolatility(),
        "Hierarchical Risk Parity": HierarchicalRiskParity(),
        "Max Sharpe (Mean-Variance)": MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RATIO),
    }
    curves = {}
    for name, rule in rules.items():
        try:
            predicted = cross_val_predict(rule, asset_returns, cv=cv)
            curves[name] = predicted.returns_df.squeeze().rename(name)
        except Exception:
            print(f"Warning: the {name} allocation could not be fitted:\n{traceback.format_exc()}")
    if not curves:
        return
    window = curves[next(iter(curves))].index
    compared = {"Strategy": returns.reindex(window).fillna(0.0), **curves}

    stats = pd.DataFrame({
        name: {"Annual Return": f"{ep.annual_return(series):.2%}",
               "Sharpe": f"{ep.sharpe_ratio(series):.2f}",
               "Max Drawdown": f"{ep.max_drawdown(series):.2%}"}
        for name, series in compared.items()
    }).transpose()
    section.table(Table(stats, None), title="Same Names, Other Allocations", first_header="Allocation")
    section.side.append(
        f'<p class="caption">Each rule is fitted on the previous {_WALK_FORWARD_TRAIN} sessions and '
        f'held for the next {_WALK_FORWARD_TEST}, over the names this strategy traded, from '
        f'{window[0]:%-d %b, %Y}, without trading costs. The names themselves come from the '
        'strategy, so this compares its timing and sizing with simple rules -- not its choice '
        'of stocks.</p>')

    colors = [STRATEGY_COLOR, BENCHMARK_COLOR, "#4FA487", "#559CAD", "#9B59B6"]

    def draw(ax):
        for (name, series), color in zip(compared.items(), colors):
            ax.plot(ep.cum_returns(series, 1.0), label=name, color=color,
                    lw=2.0 if name == "Strategy" else 1.2)
        ax.axhline(1.0, ls="--", lw=1, color="black", zorder=1)
        ax.legend(loc="upper left")
        ax.set_ylabel("Cumulative returns")

    section.chart(chart("Same Names, Other Allocations (Walk-Forward)", draw, height=4.0, y="growth"))
