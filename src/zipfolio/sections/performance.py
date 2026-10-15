"""Performance, kept short: the QuantStats tearsheet beside this one covers returns.

What is here is what a returns tearsheet cannot say: the strategy against simply
holding the names it traded, how much of the time it was in the market and what its
idle cash would have earned, whether its Sharpe ratio is more than luck, and whether
the first half of the backtest looks like the second.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..page import BAD, DATE_FORMAT, INFO, OK, WARN, Section
from ..style import chart
from ..tables import Table

#: The verdict on the Sharpe ratio: the chance its true value is above zero -- or, once the
#: number of versions tried is known, above the best a field of pure-noise strategies would show.
SIGNIFICANT = 0.95
#: Versions of a strategy tried before this one, for the table of what noise alone would reach.
TRIES = (5, 10, 20, 50)
EULER_GAMMA = 0.5772156649015329
#: Halves are compared only when each is at least half a year long.
MIN_HALF = 126


def exposure_of(positions: pd.DataFrame | None, index: pd.DatetimeIndex) -> pd.Series | None:
    """Each session's gross exposure at its close: the absolute value of the positions over the book."""
    if positions is None:
        return None
    book = positions.sum(axis="columns")
    gross = positions.drop(columns="cash", errors="ignore").abs().sum(axis="columns")
    return (gross / book.where(book > 0)).reindex(index).fillna(0.0)


def buy_and_hold(bars: pd.DataFrame | None, symbols, index: pd.DatetimeIndex) -> tuple[pd.Series | None, str]:
    """Daily returns of buying the traded names in equal amounts on the first session and holding them.

    Prices only, as the bundle has them: dividends are not in it. A name with
    no price on the first session is left out -- it could not have been bought.
    """
    if bars is None:
        return None, ""
    prices = bars.xs("price", level=1).astype(float).reindex(index).ffill()
    prices = prices[[symbol for symbol in symbols if symbol in prices.columns]]
    held = [symbol for symbol in prices.columns if pd.notna(prices[symbol].iloc[0]) and prices[symbol].iloc[0] > 0]
    if not held:
        return None, ""
    growth = (prices[held] / prices[held].iloc[0]).mean(axis="columns")
    returns = growth.pct_change().fillna(0.0)
    name = f"Buy & hold {held[0]}" if len(held) == 1 else f"Buy & hold, {len(held)} names"
    return returns.rename(name), name


def sharpe_inference(returns: pd.Series) -> dict:
    """The Sharpe ratio's standard error, the probabilistic Sharpe ratio, and the deflated one for a few tries.

    Daily Sharpe ``SR`` at a 0% rate. Its standard error allows for skew and fat
    tails (Lo 2002; Mertens 2002)::

        se = sqrt((1 - skew * SR + (kurtosis - 1) / 4 * SR^2) / (n - 1))

    The probabilistic Sharpe ratio (Bailey and Lopez de Prado, 2012) is the
    chance the true Sharpe is above a benchmark ``SR*``: ``PSR = Phi((SR - SR*) / se)``.
    With ``SR* = 0`` it asks whether the strategy makes money at all. After N
    tries, the best of N strategies with no edge reaches about::

        SR*(N) = se0 * ((1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N e)))

    with ``se0 = 1 / sqrt(n - 1)``, the spread of a zero Sharpe's estimate,
    and ``PSR`` against that is the deflated Sharpe ratio (2014). The tries are
    taken as independent, which makes the bar a little higher than for
    versions that resemble each other.
    """
    from scipy.stats import norm

    values = returns.dropna().astype(float)
    n = len(values)
    deviation = values.std()
    sharpe = float(values.mean() / deviation) if deviation > 0 else 0.0
    skew, kurtosis = float(values.skew()), float(values.kurt()) + 3.0
    variance = max(1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe ** 2, 1e-12) / max(n - 1, 1)
    error = math.sqrt(variance)
    null_error = 1 / math.sqrt(max(n - 1, 1))
    annual = math.sqrt(252)

    def noise_best(tries: int) -> float:
        return null_error * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / tries)
                             + EULER_GAMMA * norm.ppf(1 - 1 / (tries * math.e)))

    return {
        "sharpe": sharpe * annual,
        "low": (sharpe - 1.96 * error) * annual,
        "high": (sharpe + 1.96 * error) * annual,
        "psr": float(norm.cdf(sharpe / error)),
        "skew": skew,
        "excess_kurtosis": kurtosis - 3,
        "noise": {tries: noise_best(tries) * annual for tries in TRIES},
        "deflated": {tries: float(norm.cdf((sharpe - noise_best(tries)) / error)) for tries in TRIES},
        "deflated_at": lambda tries: (noise_best(tries) * annual, float(norm.cdf((sharpe - noise_best(tries)) / error))),
    }


def current_drawdown(returns: pd.Series) -> tuple[float, pd.Timestamp | None, int]:
    """How far below its high the strategy ended, the day of that high, and the sessions since."""
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    depth = float(wealth.iloc[-1] / peak.iloc[-1] - 1)
    if depth > -1e-9:
        return 0.0, None, 0
    high = wealth.idxmax()
    return depth, high, int((returns.index > high).sum())


def build_performance(returns, benchmark, positions, transactions, bars=None, risk_free=None,
                      risk_free_name: str = "", trials: int | None = None, holdings=None, orders=None,
                      flows=None) -> Section:
    """Key metrics against buy and hold and the benchmark, significance, and the two halves.

    ``risk_free`` is each session's risk-free return (``risk_free_for``) and
    ``risk_free_name`` what it is; without one the Sharpe ratio is at a 0% rate.
    ``trials`` is how many versions of the strategy were backtested, when known.
    """
    import empyrical as ep
    from pyfolio import plotting

    section = Section("Performance")
    exposure = exposure_of(positions, returns.index)
    symbols = sorted(transactions["symbol"].unique()) if transactions is not None else []
    holding, holding_name = buy_and_hold(bars, symbols, returns.index)
    rf = risk_free.reindex(returns.index).fillna(0.0) if risk_free is not None else None
    rate = float(rf.mean() * 252) if rf is not None else 0.0
    bench_name = benchmark.name if benchmark is not None and benchmark.name else "Benchmark"

    columns = {"Strategy": returns}
    if holding is not None:
        columns[holding_name] = holding
    if benchmark is not None:
        columns[bench_name] = benchmark

    def metrics(name: str, series: pd.Series) -> dict:
        row = {
            "Total return": f"{ep.cum_returns_final(series):.2%}",
            "Annual return (compounded)": f"{ep.annual_return(series):.2%}",
            "Annual volatility": f"{ep.annual_volatility(series):.2%}",
        }
        if rf is not None:
            row[f"Sharpe ratio over {rate:.2%} risk-free"] = f"{ep.sharpe_ratio(series, risk_free=rf):.2f}"
        row["Sharpe ratio at 0% (as QuantStats)"] = f"{ep.sharpe_ratio(series):.2f}"
        row["Sortino ratio at 0%"] = f"{ep.sortino_ratio(series):.2f}"
        row["Max drawdown"] = f"{ep.max_drawdown(series):.2%}"
        row["Calmar ratio"] = f"{ep.calmar_ratio(series):.2f}"
        row["Worst day"] = f"{series.min():.2%}"
        row["Skew"] = f"{series.skew():.2f}"
        row["Excess kurtosis"] = f"{series.kurt():.2f}"
        if benchmark is not None and name != bench_name:
            alpha, beta = ep.alpha_beta(series, benchmark)
            row[f"Beta to {bench_name}"] = f"{beta:.2f}"
            row[f"Alpha to {bench_name} (CAPM, a year)"] = f"{alpha:.2%}"
        if name == "Strategy" and exposure is not None:
            row["Time in the market"] = f"{(exposure > 0).mean():.0%}"
            row["Average exposure (gross)"] = f"{exposure.mean():.2f}"
        elif exposure is not None:
            row["Time in the market"] = "100%"
            row["Average exposure (gross)"] = "1.00"
        return row

    table = pd.DataFrame({name: metrics(name, series) for name, series in columns.items()})
    section.table(Table(table.fillna("-"), None), title="Key Performance Metrics", first_header="Metric",
                  wide=len(table.columns) > 2)

    cash_line = ""
    extra = None
    if exposure is not None and rf is not None and positions is not None:
        book = positions.sum(axis="columns")
        idle = (positions["cash"].clip(lower=0) / book.where(book > 0)).clip(0, 1).shift(1).fillna(1.0)
        credited = returns + idle.reindex(returns.index).fillna(0.0) * rf
        extra = ep.annual_return(credited) - ep.annual_return(returns)
        cash_line = (f" Had it earned the rate, the strategy's idle cash would have added about {extra:.2%} a year, "
                     f"and its Sharpe ratio over the rate would be {ep.sharpe_ratio(credited, risk_free=rf):.2f}.")
    rate_text = (f"The Sharpe ratio over the risk-free rate uses {risk_free_name}, {rate:.2%} a year on average "
                 "over the backtest -- the rate the factor analysis uses. QuantStats, beside this report, uses 0%; "
                 "both are shown." if rf is not None else
                 "No risk-free rate could be had, so the Sharpe ratio is at 0%, as QuantStats has it.")
    section.side.append(
        f'<p class="caption">Annual return is compounded (CAGR); volatility, Sharpe and Sortino count 252 '
        f'sessions a year. {rate_text}</p>'
        f'<p class="caption"><b>Interest on idle cash is not credited</b>: the backtest earns nothing on cash.'
        f'{cash_line}</p>'
        + (f'<p class="caption"><b>{holding_name}</b> buys the traded names in equal amounts on the first session '
           'and holds them to the end, on prices alone -- dividends are in neither it nor the strategy. The '
           'names were picked by the strategy, so this measures its timing, not its choice of names.</p>'
           if holding is not None else ""))

    if transactions is not None and holdings is not None:
        _reconciliation(section, returns, holdings, positions, transactions, orders, flows)

    versus = " vs Benchmark" if benchmark is not None else ""
    section.chart(chart(f"Cumulative Returns{versus}", lambda ax: plotting.plot_rolling_returns(
        returns, factor_returns=benchmark, ax=ax), height=4.5, y="growth"))
    section.chart(chart("Underwater Plot", lambda ax: plotting.plot_drawdown_underwater(
        returns=returns, ax=ax)))

    _glance(section, returns, exposure, extra, rate, risk_free_name)
    _return_findings(section, returns, benchmark, holding, holding_name)
    _significance(section, returns, trials)
    _halves(section, returns, exposure)
    return section


def _reconciliation(section: Section, returns, holdings, positions, transactions, orders, flows=None) -> None:
    """The profit split into realized, unrealized, commission and other cash, and checked (``accounting``).

    Nothing is adjusted to make it add up: a gap between the book and the daily
    returns, or cash no fill explains, is shown as it is.
    """
    from ..accounting import profit_breakdown
    from ..currency import money
    from .round_trips import extract_round_trips

    closed = extract_round_trips(returns, positions, transactions)
    profit = profit_breakdown(returns, holdings, transactions, orders, closed, flows)
    from_returns = profit.from_returns
    rows = {
        f"Realized: {profit.closed_trades:,} closed trades, before commission": money(profit.realized),
        f"Unrealized: {profit.open_names:,} positions still held, at the last price": money(profit.unrealized),
    }
    if profit.commission is not None:
        rows["Commission"] = money(-profit.commission)
        rows["Other cash: dividends, interest, fees"] = money(profit.other_cash)
    else:
        rows["Commission and other cash (orders not given)"] = money(profit.other_cash)
    if profit.flows:
        rows["External flows: deposits less withdrawals, not profit"] = money(profit.flows)
    rows["Net profit: change in cash plus positions"] = money(profit.net)
    rows["Net profit from the daily returns"] = money(from_returns)
    rows["Difference"] = money(profit.returns_gap)
    section.table(Table(pd.Series(rows).to_frame("Amount"), None), title="Profit Reconciliation",
                  first_header="Source")
    section.side.append(
        '<p class="caption">Every profit on this page is one of these. <b>Realized</b> is what the closed trades '
        'made -- each sale with the shares it closed, first in, first out; <b>unrealized</b>, what the shares still '
        'held are worth at the last price over what they cost. Both are before commission, and slippage is inside '
        'both, because it is inside the fill prices: Trading Costs reconstructs it and nothing takes it off again. '
        'The net profit is the change in the book, cash plus positions, and is checked against the start value '
        'compounded by the daily returns, which the engine records apart.</p>')
    if abs(profit.returns_gap) > profit.tolerance:
        section.side.append(
            f'<p class="note">The book and the daily returns disagree by {money(profit.returns_gap)}: one of the '
            "engine's records does not match the other, and every figure built on the returns -- annual return, "
            "Sharpe ratio, drawdowns -- differs from the book by as much.</p>")
        section.finding(WARN, "Book against daily returns", money(profit.returns_gap),
                        "The change in cash plus positions differs from the start value compounded by the daily "
                        "returns.")
    if profit.commission is not None and abs(profit.other_cash) > profit.tolerance:
        section.side.append(
            f'<p class="note">{money(profit.other_cash)} of the change in the book is explained by no fill and no '
            "commission -- dividends, interest or fees the engine booked. It is in the net profit and in no PnL "
            "above.</p>")


def _glance(section: Section, returns: pd.Series, exposure, extra, rate: float, rate_name: str) -> None:
    if exposure is not None:
        held = float((exposure > 0).mean())
        section.finding(INFO, "In the market", f"{held:.0%} of sessions · exposure {exposure.mean():.2f}",
                        f"Holding positions on {int((exposure > 0).sum()):,} of {len(exposure):,} sessions; the "
                        "average gross exposure counts the sessions in cash as zero.", key="exposure")
    depth, high, since = current_drawdown(returns)
    if high is None:
        section.finding(INFO, "Now", "at a new high", "The backtest ended at its highest value.", key="now")
    else:
        section.finding(WARN if depth <= -0.05 else INFO, "Now", f"{depth:.1%} from the {high:{DATE_FORMAT}} high",
                        f"At the end of the backtest the strategy was {abs(depth):.1%} below its high of "
                        f"{high:{DATE_FORMAT}}, {since:,} sessions earlier, and had not recovered. Whoever starts "
                        "it now starts in a drawdown.", key="now")
    if extra is not None:
        section.finding(INFO, "Interest on idle cash", f"not credited (+{extra:.1%}/yr at the rate)",
                        f"The backtest earns nothing on cash. At {rate_name}, {rate:.2%} a year on average, its idle "
                        f"cash would have earned about {extra:.2%} a year more.", key="cash")


def _return_findings(section: Section, returns, benchmark, holding, holding_name: str) -> None:
    import empyrical as ep

    annual = ep.annual_return(returns)
    if benchmark is not None:
        name = benchmark.name or "the benchmark"
        market = ep.annual_return(benchmark)
        section.finding(OK if annual >= market else WARN, f"Annual return vs {name}", f"{annual:.1%} vs {market:.1%}",
                        f"Compounded a year, over the same sessions. Below {name}'s, holding {name} would have "
                        "done better.", key="return", threshold=f"at least {name}'s",
                        issue=f"it earned less than {name} ({annual:.1%} against {market:.1%} a year)",
                        advice=f"Compare it with simply holding {name}: a strategy has to beat that to be worth running.")
    else:
        section.finding(OK if annual > 0 else WARN, "Annual return", f"{annual:.1%}", "Compounded a year.",
                        key="return", threshold="above zero", issue=f"it lost money ({annual:.1%} a year)")
    if holding is None:
        return
    mine, theirs = ep.sharpe_ratio(returns), ep.sharpe_ratio(holding)
    held = ep.annual_return(holding)
    better = mine >= theirs
    section.finding(OK if better else WARN, f"Against {holding_name[0].lower()}{holding_name[1:]}",
                    f"Sharpe {mine:.2f} vs {theirs:.2f}",
                    f"The strategy's Sharpe ratio against buying the same names on day one and holding them "
                    f"(both at 0%); annual return {annual:.1%} against {held:.1%}. A timing rule that does worse "
                    "per unit of risk than holding its own names adds nothing but trading.",
                    key="buy_hold", threshold="Sharpe at least buy & hold's",
                    issue=f"holding the same names did better per unit of risk (Sharpe {theirs:.2f} against {mine:.2f})",
                    advice="Check whether the entries and exits add anything over simply holding the names.")


def _significance(section: Section, returns: pd.Series, trials: int | None) -> None:
    if len(returns) < 60:
        return
    stats = sharpe_inference(returns)
    rows = {"Tried once: above 0": (0.0, stats["psr"])}
    for tries in TRIES:
        rows[f"Best of {tries} tries"] = (stats["noise"][tries], stats["deflated"][tries])
    if trials and trials > 1 and trials not in TRIES:
        rows[f"Best of {trials} tries (this strategy)"] = stats["deflated_at"](trials)
    table = pd.DataFrame({
        "Noise reaches": {label: f"{bar:.2f}" for label, (bar, _) in rows.items()},
        "Chance it beats that": {label: f"{chance:.0%}" for label, (_, chance) in rows.items()},
    })
    facts = pd.Series({
        "Sharpe ratio at 0%": f"{stats['sharpe']:.2f}",
        "95% interval": f"{stats['low']:.2f} to {stats['high']:.2f}",
        "Chance the true Sharpe is above 0 (PSR)": f"{stats['psr']:.0%}",
        "Skew · excess kurtosis of daily returns": f"{stats['skew']:.2f} · {stats['excess_kurtosis']:.2f}",
        "Sessions": f"{len(returns):,}",
    })
    section.table(Table(facts.to_frame("Value"), None), title="Is the Sharpe Ratio More than Luck?",
                  first_header="Measure")
    section.table(Table(table, None), title="Against What Noise Reaches after Many Tries", first_header="Versions tried")
    section.side.append(
        '<p class="caption">The interval and the <b>probabilistic Sharpe ratio</b> (PSR) allow for the skew '
        'and fat tails of the daily returns. Every version of a strategy that was backtested is a try: '
        'among N strategies with no edge at all, the best one shows a Sharpe ratio of about the second column '
        'by chance alone, and the third is the chance this one beats it -- the <b>deflated Sharpe ratio</b>. '
        'Tries are taken as independent. 95% is the usual bar.</p>')

    known = bool(trials and trials > 1)
    if known:
        bar, chance = stats["deflated_at"](trials)
        value = f"{chance:.0%} after {trials} tries"
        label = "Sharpe ratio beyond luck (deflated)"
        threshold = f"≥ {SIGNIFICANT:.0%} chance of beating the best of {trials} noise strategies"
        status = OK if chance >= SIGNIFICANT else BAD if chance < 0.5 else WARN
    else:
        chance = stats["psr"]
        value = f"PSR {chance:.0%}"
        label = "Sharpe ratio beyond luck (PSR)"
        threshold = f"≥ {SIGNIFICANT:.0%} chance the true Sharpe is above 0"
        status = OK if chance >= SIGNIFICANT else WARN
    after = f"{stats['noise'][10]:.2f} after 10 tries"
    section.finding(
        status, label, value,
        f"Sharpe {stats['sharpe']:.2f}, 95% interval {stats['low']:.2f} to {stats['high']:.2f} over "
        f"{len(returns):,} sessions. The best of strategies with no edge reaches about {after}, by chance.",
        key="significance", threshold=threshold,
        issue=f"the Sharpe ratio cannot yet be told from luck ({value})",
        advice="Test it on years it was not tuned on, and count every version you tried: each one raises the bar.")


def _halves(section: Section, returns: pd.Series, exposure) -> None:
    import empyrical as ep

    if len(returns) < 2 * MIN_HALF:
        return
    middle = len(returns) // 2
    halves = {f"{part.index[0]:%b %Y} – {part.index[-1]:%b %Y}": (part, None if exposure is None else
                                                                   exposure.reindex(part.index))
              for part in (returns.iloc[:middle], returns.iloc[middle:])}
    table = pd.DataFrame({
        name: {
            "Annual return": f"{ep.annual_return(part):.2%}",
            "Sharpe ratio at 0%": f"{ep.sharpe_ratio(part):.2f}",
            "Max drawdown": f"{ep.max_drawdown(part):.2%}",
            **({"Time in the market": f"{(held > 0).mean():.0%}"} if held is not None else {}),
        }
        for name, (part, held) in halves.items()
    })
    section.table(Table(table, None), title="First Half and Second Half", first_header="Metric")
    section.side.append(
        '<p class="caption">The backtest cut in two at its middle session. A rule whose parameters were '
        'tuned on the whole period often works in one half only.</p>')
