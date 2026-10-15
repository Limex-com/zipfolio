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
from ..page import INFO, OK, WARN, Section
from ..style import BENCHMARK_COLOR, LOSS_COLOR, PALETTE, STRATEGY_COLOR, chart, color_mix
from ..tables import Table

#: Fewer daily observations than this and the betas are noise.
MIN_OBSERVATIONS = 60
ROLLING_WINDOW = 126
#: Newey-West lags: daily returns carry a few days of autocorrelation, which
#: plain OLS standard errors ignore and so overstate significance.
HAC_LAGS = 5


def risk_free_for(dates: pd.DatetimeIndex, factor_model: str | None, risk_free: str | pd.Series | None = "ruonia",
                  factor_data: pd.DataFrame | None = None) -> tuple[pd.Series | None, str]:
    """The risk-free return of each session in ``dates``, and what the rate is, for the Sharpe ratio and idle cash.

    The same rate the factor analysis uses: the one-month T-bill from Kenneth
    French's library under a Fama-French model, RUONIA under the market model
    (``risk_free`` names it, or is a series of a rate in percent a year).
    French's library publishes a month or two late; its last rate is carried
    to the end of the backtest. ``(None, "")`` when there is no rate to have.
    """
    if isinstance(risk_free, pd.Series):
        return risk_free_returns(risk_free, dates), str(risk_free.name or "the risk-free rate")
    if factor_model == "market":
        if not risk_free:
            return None, ""
        try:
            return risk_free_returns(ruonia(), dates), "RUONIA"
        except Exception:
            print(f"Warning: no RUONIA for the Sharpe ratio:\n{traceback.format_exc()}")
            return None, ""
    if not factor_model:
        return None, ""
    try:
        data = factor_data if factor_data is not None else fama_french_daily(factor_model)
    except Exception:
        print(f"Warning: no risk-free rate for the Sharpe ratio:\n{traceback.format_exc()}")
        return None, ""
    rate = data["RF"].astype(float)
    rf = rate.reindex(rate.index.union(dates)).ffill().reindex(dates).fillna(float(rate.iloc[0]))
    return rf.rename("RF"), "the one-month T-bill"


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
                          source: str | None = None, glossary: dict[str, str] | None = None,
                          benchmark: pd.Series | None = None) -> Section | None:
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
    _alpha_finding(section, fit, names, model, description)

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
    attribution[f"Strategy (arithmetic mean, to {data.index[-1]:%-d %b %Y})"] = data["strategy"].mean() * 252
    section.table(Table(pd.Series(attribution).to_frame("Annual Return"), None, "{:.2%}".format),
                  title="Return Attribution", first_header="Source")
    section.side.append(
        '<p class="caption">The attribution adds up the arithmetic mean return over the sessions the '
        'factors cover -- not the compounded annual return of Key Performance Metrics, and not past the '
        'last month the library has published.</p>')

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
        from ..interactive import TIPS

        betas = fit.params[names]
        errors = 1.96 * fit.bse[names]
        significant = fit.pvalues[names] < 0.05
        colors = [STRATEGY_COLOR if beta >= 0 else LOSS_COLOR for beta in betas]
        positions = np.arange(len(names))[::-1]
        tips = [factor_tip(name, float(betas[name]), float(errors[name]), float(fit.pvalues[name]),
                           glossary.get(name, name)) for name in names]
        bars = ax.barh(positions, betas, xerr=errors, color=colors, ecolor="#333333", capsize=4,
                       alpha=1.0, height=0.6)
        setattr(bars, TIPS, tips)
        for position, beta, strong, tip in zip(positions, betas, significant, tips):
            if not strong:
                faded = ax.barh(position, beta, color="white", alpha=0.55, height=0.6)
                setattr(faded, TIPS, [tip])
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
    else:
        rolling = None

    if {"SMB", "HML"} <= set(names):
        _style_box(section, data, names, fit, rolling, benchmark)

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


#: What a factor beta says, by its sign: (positive, negative). A reader hovering a bar learns what
#: kind of company the strategy behaved like, not only the factor's name.
LEANS = {
    "Mkt-RF": ("Moves with the market", "Moves against the market"),
    "SMB": ("Leans to small companies: it moved like a book of small caps held against large ones",
            "Leans to large companies: it moved like a book of large caps held against small ones, as the "
            "S&P 500 does"),
    "HML": ("Leans to value: cheap companies, priced low for their book value -- banks, energy, industrials",
            "Leans to growth: expensive companies, priced high for their book value -- technology and other "
            "high-multiple names"),
    "RMW": ("Leans to profitable companies: robust operating profitability",
            "Leans to weakly profitable companies, often young or loss-making"),
    "CMA": ("Leans to conservative companies: they grow their assets slowly",
            "Leans to aggressive companies: they expand their assets fast"),
    "MOM": ("Follows momentum: it moved like the past year's winners",
            "Leans against momentum: it moved like the past year's losers"),
}


def factor_tip(name: str, beta: float, error: float, p_value: float, description: str) -> str:
    """The tooltip of one factor's bar: the beta, which way it leans, what the factor is, whether it counts."""
    import html

    positive, negative = LEANS.get(name, ("A positive exposure", "A negative exposure"))
    lean = positive if beta >= 0 else negative
    if name == "Mkt-RF":
        lean += f": a 1% move of the market came with {abs(beta):.2f}% of the strategy"
    elif abs(beta) < 0.05:
        lean = "No lean either way"
    verdict = ("Significant." if p_value < 0.05 else
               f"Not significant (p = {p_value:.2f}): it may well be zero.")
    return (f"<b>{html.escape(name)} {beta:+.2f}</b> &nbsp;95%: {beta - error:+.2f} to {beta + error:+.2f}<br>"
            f"{html.escape(lean)}.<br><span style=\"color:#888\">{html.escape(description[:1].upper() + description[1:])}."
            f"</span><br>{verdict}")


def _alpha_finding(section: Section, fit, names: list[str], model: str, description: str) -> None:
    """Alpha and the exposures that matter, in a sentence."""
    alpha, p = fit.params["const"] * 252, fit.pvalues["const"]
    significant = p < 0.05
    verdict = "significant" if significant else f"not significant (p = {p:.2f})"
    value = f"{alpha:+.1%} a year" + ("" if significant else ", n.s.")
    status = OK if significant and alpha > 0 else WARN
    common = dict(key="alpha", threshold="positive, with p < 0.05",
                  issue=f"no return beyond what the factors pay (alpha {value})",
                  advice="Check what the factors explain: a strategy that is mostly market beta is cheaper bought as an index.")
    why = ("A return the factors do not explain, and too steady to be chance." if status == OK else
           "No evidence of a return beyond what the factors pay: an alpha that is not significant may be "
           "luck, and a negative one is a cost.")
    if model == "market":
        market = description.split(": ", 1)[-1]
        beta = fit.params[names[0]]
        section.finding(INFO, "Beta to the market", f"{beta:.2f}", f"Market model: {market}.")
        section.finding(status, "Alpha over the market", value, f"Over {market}: {verdict}. {why}", **common)
        return
    exposures = [f"{name} {fit.params[name]:.2f}" for name in names if fit.pvalues[name] < 0.05]
    section.finding(status, "Alpha after factors", value,
                    f"After {description}: {verdict}. {why} " + (
                        f"Significant exposures: {', '.join(exposures)}." if exposures else
                        "No factor exposure is significant."), **common)


def _p_value(value: float) -> str:
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


#: The style box's lines, in betas: HML beyond +-``STYLE_TILT`` leans value or growth; SMB under
#: ``MID_SIZE`` is large companies, over ``SMALL_SIZE`` small ones. For scale, the S&P 500 sits near
#: SMB -0.15, the Russell 2000 near +0.8; the Russell 1000 Value and Growth near HML +0.35 and -0.35.
STYLE_TILT, MID_SIZE, SMALL_SIZE = 0.15, 0.15, 0.55
#: The rolling path is drawn a point a month.
STYLE_STEP = 21


def style_of(smb: float, hml: float) -> str:
    """The style box cell of a pair of betas: "Large Growth", "Small Value", "Mid Blend"."""
    size = "Large" if smb < MID_SIZE else "Mid" if smb < SMALL_SIZE else "Small"
    style = "Value" if hml > STYLE_TILT else "Growth" if hml < -STYLE_TILT else "Blend"
    return f"{size} {style}"


def _style_box(section: Section, data: pd.DataFrame, names: list[str], fit, rolling, benchmark) -> None:
    """Morningstar's style box, from returns: size by the SMB beta, value or growth by the HML beta.

    Holdings-based style needs every company's size and book-to-market; the
    betas say how the strategy's returns moved with small against big
    companies and cheap against expensive ones, which is the same question
    asked of the returns. The whole backtest is one point, with its 95%
    interval; the rolling betas are its path, a point a month.
    """
    import statsmodels.api as sm
    from matplotlib.patches import Rectangle

    from ..interactive import BACKGROUND, TIPS

    smb, hml = float(fit.params["SMB"]), float(fit.params["HML"])
    smb_error, hml_error = 1.96 * float(fit.bse["SMB"]), 1.96 * float(fit.bse["HML"])
    style = style_of(smb, hml)
    market = None
    if benchmark is not None:
        joined = pd.concat([benchmark.rename("benchmark"), data[names + ["RF"]]], axis="columns", join="inner").dropna()
        if len(joined) >= MIN_OBSERVATIONS:
            market_fit = sm.OLS(joined["benchmark"] - joined["RF"], sm.add_constant(joined[names])).fit()
            market = (float(market_fit.params["SMB"]), float(market_fit.params["HML"]), benchmark.name or "Benchmark")

    path = rolling[["SMB", "HML"]].iloc[::STYLE_STEP] if rolling is not None and len(rolling) else None
    xs = [hml - hml_error, hml + hml_error] + ([market[1]] if market else []) + (
        list(path["HML"]) if path is not None else [])
    ys = [smb - smb_error, smb + smb_error] + ([market[0]] if market else []) + (
        list(path["SMB"]) if path is not None else [])
    x_low, x_high = min(min(xs) - 0.15, -0.6), max(max(xs) + 0.15, 0.6)
    y_low, y_high = min(min(ys) - 0.15, -0.5), max(max(ys) + 0.15, 1.0)

    def draw(ax):
        columns = [(x_high, STYLE_TILT, "Value"), (STYLE_TILT, -STYLE_TILT, "Blend"), (-STYLE_TILT, x_low, "Growth")]
        rows = [(y_low, MID_SIZE, "Large"), (MID_SIZE, SMALL_SIZE, "Mid"), (SMALL_SIZE, y_high, "Small")]
        for row, (bottom, top, size) in enumerate(rows):
            for column, (left, right, kind) in enumerate(columns):
                mine = f"{size} {kind}" == style
                cell = Rectangle((right, bottom), left - right, top - bottom, gid=BACKGROUND,
                                 facecolor="#e8f0d8" if mine else ("#f6f6f6" if (row + column) % 2 else "#fbfbfb"),
                                 edgecolor="white", linewidth=1.5, zorder=0)
                ax.add_patch(cell)
                ax.text((left + right) / 2, (bottom + top) / 2, f"{size}\n{kind}", ha="center", va="center",
                        color="#4f7a00" if mine else "#b0b0b0", fontsize=10, fontweight="bold" if mine else "normal",
                        zorder=1)
        if path is not None and len(path) > 1:
            ax.plot(path["HML"], path["SMB"], color="#bbbbbb", lw=1.0, zorder=2)
            shades = [color_mix("#d7e6b8", STRATEGY_COLOR, share) for share in np.linspace(0, 1, len(path))]
            dots = ax.scatter(path["HML"], path["SMB"], s=22, c=shades, zorder=3, linewidths=0,
                              label=f"Rolling {ROLLING_WINDOW} sessions, older to newer")
            setattr(dots, TIPS, [f"{day:%b %Y}: <b>{style_of(row.SMB, row.HML)}</b><br>SMB {row.SMB:+.2f}, "
                                 f"HML {row.HML:+.2f}" for day, row in path.iterrows()])
        if market:
            point = ax.scatter([market[1]], [market[0]], s=60, marker="s", color=BENCHMARK_COLOR, zorder=4,
                               label=market[2])
            setattr(point, TIPS, [f"{market[2]}: <b>{style_of(market[0], market[1])}</b><br>SMB "
                                  f"{market[0]:+.2f}, HML {market[1]:+.2f}"])
        ax.errorbar([hml], [smb], xerr=[hml_error], yerr=[smb_error], fmt="none", ecolor="#333333", capsize=4,
                    lw=1.2, zorder=4)
        point = ax.scatter([hml], [smb], s=90, marker="D", color="#000000", zorder=5,
                           label="Whole backtest, 95% interval")
        setattr(point, TIPS, [f"Whole backtest: <b>{style}</b><br>SMB {smb:+.2f} &plusmn; {smb_error:.2f}, "
                              f"HML {hml:+.2f} &plusmn; {hml_error:.2f}"])
        ax.set_xlim(x_high, x_low)  # value on the left, as Morningstar draws it
        ax.set_ylim(y_high, y_low)  # large companies at the top
        ax.set_xlabel("HML beta: value (left) to growth (right)")
        ax.set_ylabel("SMB beta: large (top) to small")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, fontsize=9)

    section.chart(chart("Style Box", draw, height=4.6))
    tilt = "no lean to value or growth" if style.endswith("Blend") else f"a lean to {style.split()[1].lower()}"
    section.finding(INFO, "Style", style,
                    f"From the factor betas: SMB {smb:+.2f} (size), HML {hml:+.2f} (value against growth) -- "
                    f"{style.split()[0].lower()} companies, {tilt}.", key="style")
    section.side.append(
        '<p class="caption">The <b>style box</b> places the strategy by how its returns moved: with small '
        'against big companies (SMB) and cheap against expensive ones (HML). The lines are at SMB '
        f'{MID_SIZE:g} and {SMALL_SIZE:g} and HML &plusmn;{STYLE_TILT:g}; the S&amp;P 500 sits near SMB -0.15, '
        'the Russell 2000 near 0.8. The path is the 126-session betas a month apart, darker as they get '
        'recent.</p>')
