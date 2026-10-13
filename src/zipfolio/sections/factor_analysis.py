"""Factor analysis: how much of the strategy is known factors, and what is left."""

from __future__ import annotations

import traceback

import numpy as np
import pandas as pd

from ..factors import (
    FACTOR_DESCRIPTIONS,
    MODELS,
    fama_french_daily,
    market_factors,
    model_description,
    risk_free_returns,
    ruonia,
)
from ..page import Section
from ..style import BENCHMARK_COLOR, LOSS_COLOR, PALETTE, STRATEGY_COLOR, chart
from ..tables import Table

#: Fewer daily observations than this and the betas are noise.
MIN_OBSERVATIONS = 60
ROLLING_WINDOW = 126
#: Newey-West lags: daily returns carry a few days of autocorrelation, which
#: plain OLS standard errors ignore and so overstate significance.
HAC_LAGS = 5


def build_market_model_analysis(returns: pd.Series, benchmark: pd.Series | None,
                                risk_free: str | pd.Series | None = "ruonia") -> Section:
    """The factor analysis of a market without published factors: the strategy against its benchmark.

    One factor, the benchmark over the risk-free rate -- IMOEX over RUONIA for
    a Moscow book. ``risk_free`` is ``"ruonia"`` (read from the Bank of
    Russia), a series of the rate in percent a year, or ``None`` for none. In a
    market where the rate has moved between 4% and 21% a year, leaving it out
    would hand the rate to alpha, so a rate that cannot be had is said so.
    """
    section = Section("Factor Analysis")
    if benchmark is None:
        section.side.append('<p class="explain">The market model measures the strategy against its '
                            'benchmark, and this backtest has none.</p>')
        return section
    name = benchmark.name or "the benchmark"
    rate, rate_name, rate_source = None, None, ""
    if isinstance(risk_free, str):
        if risk_free.lower() != "ruonia":
            raise ValueError(f"unknown risk-free rate {risk_free!r}; 'ruonia' or a series")
        try:
            rate, rate_name, rate_source = ruonia(), "RUONIA", " from the Bank of Russia"
        except Exception:
            print(f"Warning: no RUONIA for the Zipfolio tearsheet:\n{traceback.format_exc()}")
            section.side.append(
                '<p class="explain">RUONIA could not be downloaded from the Bank of Russia, so this '
                'backtest has no factor analysis: without the risk-free rate the alpha would carry '
                'it.</p>')
            return section
    elif risk_free is not None:
        rate, rate_name = risk_free, risk_free.name or "the risk-free rate"

    rf = risk_free_returns(rate, returns.index) if rate is not None else None
    factors = market_factors(benchmark.reindex(returns.index).fillna(0.0), rf)
    over = f" over {rate_name}" if rate_name else ""
    return build_factor_analysis(
        returns, "market", factors,
        description=f"Market model: {name}{over}",
        source=f", with {rate_name} read{rate_source}" if rate_name and rate_source else "",
        glossary={"Mkt-RF": f"{name}{over or ' (no risk-free rate)'}, the market this strategy trades in"},
    )


def build_factor_analysis(returns: pd.Series, model: str = "ff5_mom",
                          factors: pd.DataFrame | None = None, description: str | None = None,
                          source: str | None = None, glossary: dict[str, str] | None = None) -> Section | None:
    """Regress the strategy's excess returns on a factor model and decompose them.

    ``factors`` -- decimal daily returns with an ``RF`` column -- is downloaded
    from Kenneth French's library when not given. ``description``, ``source``
    and ``glossary`` (``{factor: what it is}``) name a model of the caller's
    own; see :func:`build_market_model_analysis`. A section that cannot get its
    data says so rather than disappearing, so a missing analysis is not
    mistaken for one that was never asked for.
    """
    import statsmodels.api as sm

    section = Section("Factor Analysis")
    description = description or (model_description(model) if model in MODELS else "factor model")
    source = " from Kenneth French's data library" if source is None else source
    glossary = {**FACTOR_DESCRIPTIONS, **(glossary or {})}
    if factors is None:
        try:
            factors = fama_french_daily(model)
        except Exception:
            print(f"Warning: no factor data for the Zipfolio tearsheet:\n{traceback.format_exc()}")
            section.side.append(
                f'<p class="explain">The {description} could not be downloaded from Kenneth French\'s '
                'data library, so this backtest has no factor analysis.</p>')
            return section

    data = pd.concat([returns.rename("strategy"), factors], axis="columns", join="inner").dropna()
    if len(data) < MIN_OBSERVATIONS:
        section.side.append(
            f'<p class="explain">The {description} cover {len(data)} of this backtest\'s sessions -- '
            f'fewer than {MIN_OBSERVATIONS}, too few to estimate exposures. The library publishes '
            'with a lag of a month or two.</p>')
        return section

    names = [column for column in data.columns if column not in ("strategy", "RF")]
    excess = data["strategy"] - data["RF"]
    design = sm.add_constant(data[names])
    fit = sm.OLS(excess, design).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})

    estimates = pd.DataFrame({
        "Estimate": fit.params,
        "t-stat": fit.tvalues,
        "p-value": fit.pvalues,
    }).rename(index={"const": "Alpha (annualized)"})
    estimates.loc["Alpha (annualized)", "Estimate"] *= 252
    shown = estimates.astype(object)
    for row in shown.index:
        value = estimates.loc[row, "Estimate"]
        shown.loc[row, "Estimate"] = f"{value:.2%}" if row.startswith("Alpha") else f"{value:.2f}"
        shown.loc[row, "t-stat"] = f"{estimates.loc[row, 't-stat']:.2f}"
        shown.loc[row, "p-value"] = _p_value(estimates.loc[row, "p-value"])
    section.table(Table(shown, None), title=f"Factor Exposures ({description})", first_header="Factor")

    residual_vol = float(np.std(fit.resid, ddof=len(fit.params)) * np.sqrt(252))
    fitted = pd.Series({
        "R-squared": f"{fit.rsquared:.2f}",
        "Adjusted R-squared": f"{fit.rsquared_adj:.2f}",
        "Residual volatility (annualized)": f"{residual_vol:.2%}",
        "Information ratio (alpha / residual vol)": (
            f"{fit.params['const'] * 252 / residual_vol:.2f}" if residual_vol else "-"),
        "Sessions": f"{len(data):,}",
        "Period": f"{data.index[0]:%-d %b %Y} - {data.index[-1]:%-d %b %Y}",
    })
    section.table(Table(fitted.to_frame("Value"), None), title="Model Fit", first_header="Metric")

    # Annualized attribution: mean return = RF + alpha + sum(beta x mean factor) + 0.
    attribution = {"Risk-free rate": data["RF"].mean() * 252}
    for name in names:
        attribution[name] = fit.params[name] * data[name].mean() * 252
    attribution["Alpha"] = fit.params["const"] * 252
    attribution["Strategy (arithmetic mean)"] = data["strategy"].mean() * 252
    section.table(Table(pd.Series(attribution).to_frame("Annual Return"), None, "{:.2%}".format),
                  title="Return Attribution", first_header="Source")

    explained = "".join(f"<b>{name}</b> -- {glossary.get(name, name)}. " for name in names)
    value = ", a positive HML leans to value" if "HML" in names else ""
    section.side.append(
        '<p class="explain">The strategy\'s daily return over the risk-free rate, regressed on the '
        f'{description[0].lower() + description[1:]}{source}. {explained}</p>'
        '<p class="explain">A <b>beta</b> is how much of a factor the strategy carries: 1.0 on '
        f'Mkt-RF moves with the market{value}. '
        '<b>Alpha</b> is the return '
        'the factors do not explain; it is worth something only with a t-stat beyond about 2 '
        '(p &lt; 0.05). Standard errors are Newey-West, robust to the autocorrelation of daily '
        'returns. The attribution splits the average annual return into what each factor paid '
        'at its average beta, and the alpha.</p>')

    def exposures(ax):
        betas = fit.params[names]
        errors = 1.96 * fit.bse[names]
        significant = fit.pvalues[names] < 0.05
        colors = [STRATEGY_COLOR if beta >= 0 else LOSS_COLOR for beta in betas]
        positions = np.arange(len(names))[::-1]
        ax.barh(positions, betas, xerr=errors, color=colors, ecolor="#333333", capsize=4,
                alpha=1.0, height=0.6)
        for position, beta, strong in zip(positions, betas, significant):
            if not strong:
                ax.barh(position, beta, color="white", alpha=0.55, height=0.6)
        ax.set_yticks(positions)
        ax.set_yticklabels(names)
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("Beta, with 95% interval (faded: not significant)")

    section.chart(chart("Factor Exposures", exposures, height=max(1.8, 0.45 * len(names) + 1.0)))

    if len(data) > ROLLING_WINDOW + 20:
        from statsmodels.regression.rolling import RollingOLS

        rolling = RollingOLS(excess, design, window=ROLLING_WINDOW).fit().params[names].dropna()

        def rolling_betas(ax):
            for name, color in zip(names, PALETTE):
                ax.plot(rolling.index, rolling[name], color=color, lw=1.3, label=name)
            ax.axhline(0, ls="--", lw=1, color="black", zorder=1)
            ax.legend(loc="upper left", ncol=3)
            ax.set_ylabel("Beta")

        section.chart(chart(f"Rolling Factor Exposures ({ROLLING_WINDOW} Sessions)", rolling_betas, height=3.5))

    contributions = data[names].mul(fit.params[names], axis="columns")
    unexplained = excess - contributions.sum(axis="columns")

    def decomposition(ax):
        ax.plot(excess.cumsum(), color="black", lw=2.0, label="Strategy over risk-free")
        for name, color in zip(names, PALETTE[1:] + PALETTE[:1]):
            ax.plot(contributions[name].cumsum(), color=color, lw=1.2, label=name)
        ax.plot(unexplained.cumsum(), color=STRATEGY_COLOR, lw=2.0, ls="--", label="Alpha + residual")
        ax.axhline(0, ls="--", lw=1, color=BENCHMARK_COLOR, zorder=1)
        ax.legend(loc="upper left", ncol=2)
        ax.set_ylabel("Cumulative return (sum)")

    section.chart(chart("Return Decomposition: Factors vs Alpha", decomposition, height=4.0, y="pct"))

    components = contributions.assign(**{"Alpha + residual": unexplained})
    yearly = components.groupby(components.index.year).sum()
    if len(yearly) > 1:
        def by_year(ax):
            positions = np.arange(len(yearly))
            above = np.zeros(len(yearly))
            below = np.zeros(len(yearly))
            colors = dict(zip(names, PALETTE[1:] + PALETTE[:1]))
            colors["Alpha + residual"] = STRATEGY_COLOR
            for name in yearly.columns:
                values = yearly[name].to_numpy()
                base = np.where(values >= 0, above, below)
                ax.bar(positions, values, 0.65, bottom=base, color=colors.get(name, BENCHMARK_COLOR),
                       label=name, edgecolor="white", linewidth=0.5)
                above += np.where(values >= 0, values, 0)
                below += np.where(values < 0, values, 0)
            ax.scatter(positions, yearly.sum(axis="columns"), marker="D", color="black", zorder=3,
                       s=28, label="Strategy over risk-free")
            ax.axhline(0, color="black", lw=1)
            ax.set_xticks(positions)
            ax.set_xticklabels([str(year) for year in yearly.index])
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)

        section.chart(chart("Return Attribution by Year", by_year, height=4.2, y="pct"))
    section.side.append(
        '<p class="caption">The decomposition and the attribution by year use the full-period '
        'betas, and add daily returns up rather than compounding them, so the parts sum exactly '
        'to the strategy\'s return over the risk-free rate (the black line and diamonds).</p>')
    return section


def _p_value(value: float) -> str:
    return "< 0.001" if value < 0.001 else f"{value:.3f}"
