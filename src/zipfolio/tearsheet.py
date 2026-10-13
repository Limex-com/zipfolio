"""The Zipfolio tearsheet: one self-contained HTML page per backtest."""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Callable

import pandas as pd

from .currency import using_currency
from .inputs import (
    daily_returns,
    market_data_frame,
    positions_in_dollars,
    transactions_frame,
    usable_benchmark,
)
from .page import DEFAULT_TITLE, Section, ordered, page_html
from .sections.capacity import build_capacity
from .sections.drawdowns import build_drawdowns
from .sections.execution import build_execution
from .sections.factor_analysis import build_factor_analysis, build_market_model_analysis
from .sections.instruments import build_instruments
from .sections.long_short import build_long_short
from .sections.market_regimes import build_market_regimes
from .sections.next_bar_fills import build_next_bar_fills
from .sections.performance import build_performance
from .sections.portfolio_risk import build_portfolio_risk
from .sections.positions import build_positions
from .sections.profit_concentration import build_profit_concentration
from .sections.risk_over_time import build_risk_over_time
from .sections.round_trips import build_round_trips
from .sections.sector_analysis import build_sector_analysis
from .sections.trading_costs import build_trading_costs
from .sections.transactions import build_transactions
from .sectors import sectors_of
from .style import CHART_RC, PALETTE


def build_tearsheet(
    output: Path | str,
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    position_records: pd.DataFrame | None = None,
    cash: pd.Series | None = None,
    transactions: pd.DataFrame | None = None,
    market_data: pd.DataFrame | None = None,
    sector_mappings: dict[str, str] | None = None,
    title: str = DEFAULT_TITLE,
    market_timezone: str | None = None,
    orders: pd.DataFrame | None = None,
    cost_settings: dict | None = None,
    fill_bars: pd.DataFrame | None = None,
    decision_prices: pd.DataFrame | None = None,
    factor_model: str | None = "ff5_mom",
    factor_data: pd.DataFrame | None = None,
    risk_free: str | pd.Series | None = "ruonia",
    currency: str = "USD",
) -> list[str]:
    """Write the Zipfolio tearsheet for one backtest to ``output``.

    The page opens with contents and a summary of what the sections found
    (``Section.finding``), most pressing first, then the sections in four parts
    (``page.GROUPS``), each charts on the left and tables on the right:

    **Where the return came from**

    * **Performance** -- key metrics, cumulative returns, the top drawdowns and
      the underwater curve, always. Deliberately short: the QuantStats tearsheet
      beside this one covers returns in depth;
    * **Profit Concentration** -- the return without its best days; with round
      trips, their share of the net PnL in the best few, and a bootstrap;
    * **Instruments** -- when the strategy traded: each name's PnL and
      contribution, read mark-to-market from the book and the fills, and the
      spread of its daily and round-trip returns;
    * **Long and Short** -- when the book went short: what each side earned;
    * **Sector Analysis** -- when ``sector_mappings`` classifies a held name;
    * **Factor Analysis** -- when ``factor_model`` names one: ``"ff5_mom"`` (the
      default), ``"ff5"``, ``"ff3"`` or ``"ff3_mom"`` for Fama-French, which
      describe US stocks; ``"market"`` for a market model against the benchmark
      over ``risk_free`` (``"ruonia"`` -- the Bank of Russia's rate -- or a
      series of a rate in percent a year), which is what a Moscow book gets;
      ``None`` for none. Exposures, alpha, attribution, rolling betas.
      ``factor_data`` replaces the Fama-French download;

    **What it risked**

    * **Drawdowns** -- when the strategy traded: each of the worst drawdowns
      split, exactly, across the names (and sectors) that made it;
    * **Market Regimes** -- when there is a benchmark: up and down capture,
      beta on up and down days, the average month in rising, falling, calm
      and volatile markets;
    * **Portfolio Risk** -- skfolio's risk measures, always; risk contribution
      and alternative allocations when the strategy traded and ``market_data``
      covers it;
    * **Risk Over Time** -- when the strategy traded: predicted against
      realized volatility, each name's share of risk, concentration;

    **How it holds and trades** -- **Positions**, **Transactions**, **Round
    Trips**, when the strategy traded;

    **Can it be traded**

    * **Next-Bar Fills** -- when orders fill on the bar that produced them and
      ``fill_bars`` carry ``next_open``: every fill priced again at the open of
      the instrument's next bar, and what is left of the return;
    * **Execution** -- when the strategy traded and ``orders`` are given: how
      the backtest fills orders, realism checks of every fill against its bar
      (``fill_bars``) and of every order against the price it was decided at
      (``decision_prices``), order types, waits, participation, shortfall;
    * **Trading Costs** -- when the strategy traded: the cost model from
      ``cost_settings`` (the run's commission and slippage settings) in
      words, commissions from ``orders`` (ziplime charges them to the order,
      not the fill), slippage reconstructed from the fills;
    * **Capacity** -- when ``market_data`` (daily bars, see
      ``market_data_frame``) covers the traded names.

    A chart that cannot be drawn is left out and logged; a section that cannot
    be computed is replaced by a note and logged. The page fails as a whole only
    when there is nothing to report at all. ``market_timezone`` places fills on
    the clock of the transaction-time chart; it defaults to
    ``ZIPFOLIO_MARKET_TIMEZONE``, then New York.

    ``currency`` (``"USD"``, ``"RUB"``, ``"EUR"``) is how every amount is
    written, and sets the capital base of the capacity analysis.

    Returns the titles of the sections that made it into the page, in page order.
    """
    import matplotlib.pyplot as plt
    import pyfolio as pf
    from cycler import cycler
    from pyfolio import utils

    returns = daily_returns(returns)
    if len(returns) < 2:
        raise ValueError(f"pyfolio needs at least two daily returns, got {len(returns)}")
    returns = returns.rename("Strategy")

    benchmark = usable_benchmark(benchmark, returns.index)
    raw_fills = transactions
    transactions = transactions_frame(transactions)
    positions = holdings = None
    if transactions is not None:
        holdings = positions = positions_in_dollars(position_records, cash, returns.index)
    if positions is not None:
        # pyfolio's own guess at whether the strategy trades intraday, and the
        # reconstruction of peak exposure it uses when it does -- exactly what
        # create_full_tear_sheet does before its sections.
        positions = utils.check_intraday("infer", returns, positions, transactions)
    market_timezone = market_timezone or os.getenv("ZIPFOLIO_MARKET_TIMEZONE") or "America/New_York"

    bars = None
    if positions is not None:
        bars = market_data_frame(market_data, transactions["symbol"].unique(), returns.index)

    # Built in this order -- Portfolio Risk last of all, after pyfolio's styles
    # are no longer needed -- and laid out on the page by group (page.GROUPS).
    builders: list[tuple[str, Callable[[], Section | None]]] = [
        ("Performance", lambda: build_performance(returns, benchmark, positions, transactions)),
        ("Profit Concentration", lambda: build_profit_concentration(returns, positions, transactions)),
        ("Market Regimes", lambda: build_market_regimes(returns, benchmark)),
    ]
    sectors = sectors_of(positions, sector_mappings) if positions is not None else {}
    if positions is not None:
        builders += [
            ("Positions", lambda: build_positions(returns, positions)),
            ("Instruments", lambda: build_instruments(returns, holdings, positions, transactions, orders)),
            ("Long and Short", lambda: build_long_short(returns, holdings, transactions, orders)),
            ("Drawdowns", lambda: build_drawdowns(returns, holdings, transactions, orders, sectors or None)),
        ]
        if sectors:
            builders.append(("Sector Analysis", lambda: build_sector_analysis(
                returns, positions, transactions, sectors)))
    if factor_model == "market":
        builders.append(("Factor Analysis", lambda: build_market_model_analysis(returns, benchmark, risk_free)))
    elif factor_model:
        builders.append(("Factor Analysis", lambda: build_factor_analysis(returns, factor_model, factor_data)))
    builders.append(("Portfolio Risk", lambda: build_portfolio_risk(returns, benchmark, positions, bars)))
    if positions is not None:
        builders += [
            ("Risk Over Time", lambda: build_risk_over_time(returns, holdings, bars)),
            ("Transactions", lambda: build_transactions(returns, positions, transactions, market_timezone)),
            ("Next-Bar Fills", lambda: build_next_bar_fills(returns, holdings, raw_fills, fill_bars, cost_settings)),
            ("Execution", lambda: build_execution(raw_fills, orders, fill_bars, decision_prices, cost_settings)),
            ("Trading Costs", lambda: build_trading_costs(returns, positions, transactions, orders, cost_settings)),
            ("Round Trips", lambda: build_round_trips(returns, positions, transactions)),
        ]
        if bars is not None:
            builders.append(("Capacity", lambda: build_capacity(returns, positions, transactions, bars)))

    sections: list[Section] = []
    built: list[str] = []
    rc = {**CHART_RC, "axes.prop_cycle": cycler(color=PALETTE)}
    # Matplotlib dates for every plot. Left to itself, pandas draws a series
    # whose frequency it can infer -- a window with no holiday in it -- in
    # period ordinals, and anything pyfolio adds with matplotlib directly on
    # the same axis lands off-screen: the exposure panel came out empty that way.
    with plt.rc_context(rc), pd.plotting.plot_params.use("x_compat", True), using_currency(currency):
        for name, build in builders:
            try:
                section = build()
            except Exception:
                print(f"Warning: the Zipfolio {name.lower()} section could not be built. Continuing without it.")
                print(traceback.format_exc())
                sections.append(Section(name, failed=True))
                continue
            if section is None or section.empty:
                # Nothing to say -- too few round trips, no skfolio.
                continue
            sections.append(section)
            built.append(name)

    if not built:
        raise RuntimeError("no Zipfolio section could be built")

    versions = f"analytics by pyfolio-reloaded {getattr(pf, '__version__', '')}"
    if "Portfolio Risk" in built:
        import skfolio

        versions += f" and skfolio {skfolio.__version__}"
    Path(output).write_text(
        page_html(title, returns, benchmark, versions, sections),
        encoding="utf-8",
    )
    # In page order, which is the order a reader meets them in.
    return [section.title for _, members in ordered(sections) for section in members if section.title in built]
