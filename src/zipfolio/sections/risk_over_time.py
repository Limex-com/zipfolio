"""Risk over time: what the book's risk was each day, where it sat, and how concentrated it was."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..page import DATE_FORMAT, NA, OK, WARN, Section
from ..style import BENCHMARK_COLOR, LOSS_COLOR, PALETTE, STRATEGY_COLOR, chart
from ..tables import Table
from .instruments import CONCENTRATION_ADVICE
from .portfolio_risk import daily_asset_returns

#: Sessions of history behind each day's covariance, and ahead of it for the realized check.
COVARIANCE_WINDOW = 63
REALIZED_WINDOW = 21
#: A day whose next month ran this much hotter than forecast is a risk surprise.
SURPRISE = 1.5
NAMED = 8
#: One name carrying more than this share of the book's risk on a typical day needs a look;
#: without the bars to measure risk, fewer effective names than the second does.
TOP_RISK_SHARE, MIN_EFFECTIVE_NAMES = 0.4, 5
#: Charts are drawn weekly: a point a day for years of daily shares is megabytes of SVG.
SAMPLE_EVERY = 5


def risk_path(holdings: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Each session's predicted volatility, and each held name's share of it.

    The prediction is what a risk manager would have had that morning: the
    day's weights times the covariance of the held names over the previous
    ``COVARIANCE_WINDOW`` sessions -- nothing from the day itself or after it.
    A name's share is its marginal contribution, ``w_i (S w)_i / w'S w``, so
    the shares add up to one.

    Returns a frame indexed by session: ``predicted`` (annualised volatility)
    and one column per name with its share.
    """
    weights = holdings.drop(columns="cash", errors="ignore").div(holdings.sum(axis="columns"), axis="index")
    weights = weights.reindex(columns=asset_returns.columns).fillna(0.0)
    returns = asset_returns.reindex(weights.index)
    rows = {}
    for position in range(COVARIANCE_WINDOW, len(weights)):
        day = weights.index[position]
        held = weights.columns[weights.iloc[position].to_numpy() != 0]
        if not len(held):
            continue
        history = returns[held].iloc[position - COVARIANCE_WINDOW:position].dropna(how="all").fillna(0.0)
        if len(history) < COVARIANCE_WINDOW // 2:
            continue
        w = weights.loc[day, held].to_numpy()
        covariance = np.cov(history.to_numpy(), rowvar=False).reshape(len(held), len(held))
        variance = float(w @ covariance @ w)
        if variance <= 0:
            continue
        row = dict(zip(held, w * (covariance @ w) / variance))
        row["predicted"] = np.sqrt(variance * 252)
        rows[day] = row
    return pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)


def concentration(holdings: pd.DataFrame) -> pd.DataFrame:
    """Per session: names held, effective number of names, top-3 share and the largest weight, of the gross book."""
    values = holdings.drop(columns="cash", errors="ignore").abs()
    gross = values.sum(axis="columns")
    shares = values.div(gross.where(gross > 0), axis="index")
    ordered = np.sort(shares.fillna(0.0).to_numpy(), axis=1)[:, ::-1]
    return pd.DataFrame({
        "held": (values > 0).sum(axis="columns"),
        "effective": 1.0 / (shares ** 2).sum(axis="columns").replace(0, np.nan),
        "top3": ordered[:, :3].sum(axis=1),
        "largest": ordered[:, 0],
    }, index=holdings.index).where(gross > 0)


def build_risk_over_time(returns: pd.Series, holdings: pd.DataFrame, bars) -> Section | None:
    asset_returns = daily_asset_returns(bars)
    section = Section("Risk Over Time")
    crowding = concentration(holdings)
    values = holdings.drop(columns="cash", errors="ignore")
    single = int((values.abs().sum() > 0).sum()) < 2

    path = pd.DataFrame()
    if asset_returns is not None and len(asset_returns) > COVARIANCE_WINDOW + REALIZED_WINDOW:
        path = risk_path(holdings, asset_returns)
        if not path.empty:
            _forecast(section, returns, path, holdings)
            if not single:
                _risk_shares(section, path.drop(columns="predicted"))
    if single:
        section.finding(NA, "Risk from the top name, typical day", "one name",
                        "One name carries all of the book's risk; the largest position says so.", key="risk_conc")
        return section if not section.empty else None
    _concentration_finding(section, path, crowding)

    held = crowding.dropna()
    if not held.empty:
        facts = pd.Series({
            "Average names held": f"{held['held'].mean():.1f}",
            "Average effective names": f"{held['effective'].mean():.1f}",
            "Lowest effective names": f"{held['effective'].min():.1f} ({held['effective'].idxmin():{DATE_FORMAT}})",
            "Average top-3 share": f"{held['top3'].mean():.0%}",
            "Largest single weight": f"{held['largest'].max():.0%} ({held['largest'].idxmax():{DATE_FORMAT}})",
        })
        section.table(Table(facts.to_frame("Value"), None), title="Concentration", first_header="Measure")
        section.side.append(
            '<p class="explain">The <b>effective number of names</b> is 1 / &Sigma; w&sup2; over the '
            'gross book: ten equal positions count as ten, one 90% position and nine small ones as '
            'barely more than one. Below the names held, the book is concentrated in a few of '
            'them.</p>')

        def names(ax):
            sampled = crowding.iloc[::SAMPLE_EVERY]  # all-cash days stay gaps
            ax.plot(sampled.index, sampled["held"], color=BENCHMARK_COLOR, lw=1.2, drawstyle="steps-post",
                    label="Names held")
            ax.plot(sampled.index, sampled["effective"], color=STRATEGY_COLOR, lw=1.8, label="Effective names")
            ax.set_ylim(bottom=0)
            ax.legend(loc="upper left")
            ax.set_ylabel("Names")

        def top(ax):
            sampled = crowding.iloc[::SAMPLE_EVERY]  # all-cash days stay gaps
            ax.plot(sampled.index, sampled["top3"], color=STRATEGY_COLOR, lw=1.8, label="Top 3 names")
            ax.plot(sampled.index, sampled["largest"], color=LOSS_COLOR, lw=1.2, label="Largest name")
            ax.set_ylim(0, 1.05)
            ax.legend(loc="lower left")
            ax.set_ylabel("Share of the gross book")

        section.chart(chart("Effective Number of Names", names, height=3.0))
        section.chart(chart("Top-3 and Largest Position", top, height=3.0, y="pct"))
    return section if not section.empty else None


def _concentration_finding(section: Section, path: pd.DataFrame, crowding: pd.DataFrame) -> None:
    """How much of the book's risk sits in one name on a typical day -- or, without risk, how few names it is."""
    if not path.empty:
        shares = path.drop(columns="predicted").clip(lower=0)
        shares = shares.div(shares.sum(axis="columns").replace(0, np.nan), axis="index").dropna(how="all")
        if not shares.empty:
            top = shares.max(axis="columns")
            leader = shares.idxmax(axis="columns").mode().iloc[0]
            typical = float(top.median())
            section.finding(OK if typical <= TOP_RISK_SHARE else WARN, "Risk from the top name, typical day",
                            f"{typical:.0%} (most often {leader})",
                            f"On a typical day the riskiest name carried {typical:.0%} of the book's predicted "
                            f"volatility, {leader} more often than any other. A position can be a modest weight "
                            f"and still most of the risk. Over {TOP_RISK_SHARE:.0%} needs a look.", key="risk_conc",
                            threshold=f"at most {TOP_RISK_SHARE:.0%} of the risk",
                            issue=f"one name, most often {leader}, carries {typical:.0%} of the risk",
                            advice=CONCENTRATION_ADVICE)
            return
    effective = crowding["effective"].dropna()
    if len(effective):
        typical = float(effective.median())
        section.finding(OK if typical >= MIN_EFFECTIVE_NAMES else WARN, "Effective number of names, typical day",
                        f"{typical:.1f}", "Ten equal positions count as ten, one large and nine small as barely more "
                        f"than one. Without the bars to measure risk, fewer than {MIN_EFFECTIVE_NAMES} needs a look.",
                        key="risk_conc", threshold=f"at least {MIN_EFFECTIVE_NAMES}",
                        issue=f"the book is {typical:.1f} effective names", advice=CONCENTRATION_ADVICE)


def realized_while_invested(returns: pd.Series, holdings: pd.DataFrame, window: int = REALIZED_WINDOW,
                            minimum: int = REALIZED_WINDOW // 2) -> pd.Series:
    """For each session, the strategy's annualised volatility over the next ``window`` sessions it was invested.

    A session counts when the book held positions at the close before it --
    the days its return came from positions, not from cash. The prediction is
    the risk of the positions held; counting the days in cash after an exit
    would compare it with a volatility diluted by zeros, and read as a
    forecast twice too high.
    """
    book = holdings.sum(axis="columns")
    gross = holdings.drop(columns="cash", errors="ignore").abs().sum(axis="columns")
    invested = ((gross / book.where(book > 0)).fillna(0.0).shift(1).fillna(0.0) > 0).reindex(returns.index,
                                                                                               fill_value=False)
    values, mask = returns.to_numpy(dtype=float), invested.to_numpy()
    realized = np.full(len(values), np.nan)
    for position in range(len(values)):
        ahead = values[position + 1:position + 1 + window][mask[position + 1:position + 1 + window]]
        if len(ahead) >= minimum:
            realized[position] = ahead.std(ddof=1) * np.sqrt(252)
    return pd.Series(realized, index=returns.index)


def _forecast(section: Section, returns: pd.Series, path: pd.DataFrame, holdings: pd.DataFrame) -> None:
    """Predicted volatility against what the next month delivered."""
    realized = realized_while_invested(returns, holdings)
    compared = pd.DataFrame({"predicted": path["predicted"], "realized": realized}).dropna()
    if compared.empty:
        return
    ratio = compared["realized"] / compared["predicted"]
    facts = pd.Series({
        "Days compared": f"{len(compared):,}",
        "Average predicted volatility": f"{compared['predicted'].mean():.1%}",
        "Average realized volatility": f"{compared['realized'].mean():.1%}",
        "Bias (realized / predicted)": f"{ratio.mean():.2f}",
        "Correlation": f"{compared['predicted'].corr(compared['realized']):.2f}",
        f"Days the next month ran {SURPRISE:g}x hotter": f"{(ratio > SURPRISE).mean():.0%}",
    })
    section.table(Table(facts.to_frame("Value"), None), title="Forecast against Outcome", first_header="Measure")
    section.side.append(
        f'<p class="explain"><b>Predicted</b> is the volatility the book carried each morning: that '
        f'day\'s weights times the covariance of its names over the previous {COVARIANCE_WINDOW} '
        f'sessions. <b>Realized</b> is the strategy\'s volatility over the next {REALIZED_WINDOW} sessions '
        f'it held positions -- days in cash are left out, as the prediction is the risk of the positions. '
        'A bias near 1 and a high correlation mean the strategy\'s risk was what its positions said it '
        'was. Well above 1, risk arrived that the positions did not show -- crowded names, regime changes, '
        'gaps. Well below 1, the positions were calmer than their history: the strategy tends to hold '
        'them after a volatile stretch, as a breakout entry does.</p>')

    def draw(ax):
        # On every session, so a stretch the book spent in cash is a gap, not a straight line.
        sampled = compared.reindex(returns.index).iloc[::SAMPLE_EVERY]
        ax.plot(sampled.index, sampled["predicted"], color=BENCHMARK_COLOR, lw=1.6,
                label=f"Predicted (weights x {COVARIANCE_WINDOW}-session covariance)")
        ax.plot(sampled.index, sampled["realized"], color=STRATEGY_COLOR, lw=1.6,
                label=f"Realized over the next {REALIZED_WINDOW} sessions")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left")
        ax.set_ylabel("Annualized volatility")

    section.chart(chart("Predicted against Realized Volatility", draw, height=3.4, y="pct"))


def _risk_shares(section: Section, shares: pd.DataFrame) -> None:
    """Each name's share of the book's risk, through time."""
    named = shares.mean().sort_values(ascending=False).index[:NAMED]
    parts = shares[named].clip(lower=0)
    rest = [column for column in shares.columns if column not in named]
    if rest:
        parts["Other names"] = shares[rest].clip(lower=0).sum(axis="columns")
    parts = parts.div(parts.sum(axis="columns").replace(0, np.nan), axis="index").fillna(0.0)
    # One stacked bar a month, of the month's average shares: a rotating book
    # changes names between days, and lines drawn through the change cross.
    index = parts.index.tz_convert(None) if parts.index.tz is not None else parts.index
    monthly = parts.set_axis(index).resample("MS").mean().dropna(how="all")
    monthly = monthly.div(monthly.sum(axis="columns").replace(0, np.nan), axis="index").fillna(0.0)
    colors = list(PALETTE[2:] + PALETTE[:2])[:len(named)] + (["#bbbbbb"] if rest else [])

    def draw(ax):
        bottom = np.zeros(len(monthly))
        for column, color in zip(monthly.columns, colors):
            values = monthly[column].to_numpy()
            ax.bar(monthly.index, values, width=26, bottom=bottom, color=color, label=column,
                   align="edge", linewidth=0)
            bottom += values
        ax.set_ylim(0, 1)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=5, frameon=False)
        ax.set_ylabel("Share of risk")

    section.chart(chart("Share of Risk by Name", draw, height=3.8, y="pct"))
    section.side.append(
        '<p class="caption">Shares of risk are marginal contributions to the predicted volatility, '
        'averaged over each month; a hedge, whose share is negative, is drawn as zero.</p>')
