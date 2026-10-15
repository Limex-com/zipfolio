"""The Zipfolio tearsheet: one self-contained HTML page per backtest."""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Callable

import pandas as pd

from .accounting import external_flows
from .currency import using_currency
from .inputs import (
    daily_returns,
    market_data_frame,
    positions_in_dollars,
    transactions_frame,
    usable_benchmark,
)
from .currency import money
from .interactive import INTERACTIVE
from .page import DEFAULT_TITLE, INFO, NA, Section, ordered, page_html
from .sections.capacity import build_capacity
from .sections.capm import FRONTIER_FACT, TITLE as CAPM_TITLE, build_capm
from .sections.drawdowns import average_exposure, build_drawdowns
from .sections.execution import build_execution
from .sections.factor_analysis import build_factor_analysis, build_market_model_analysis, risk_free_for
from .sections.instruments import build_instruments
from .sections.long_short import build_long_short
from .sections.market_regimes import build_market_regimes
from .sections.next_bar_fills import build_next_bar_fills
from .sections.performance import build_performance
from .sections.portfolio_risk import build_portfolio_risk
from .sections.positions import build_positions
from .sections.profit_concentration import BOOTSTRAP_SAMPLES, build_profit_concentration
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
    risk_free: str | pd.Series | None = None,
    currency: str = "USD",
    run_info: dict[str, str] | None = None,
    trials: int | None = None,
    interactive: bool = True,
    opening_state=None,
    portfolio_events: pd.DataFrame | None = None,
    contract_specs: dict | None = None,
    integrity: bool = True,
    integrity_report=None,
    integrity_policy=None,
    run_id: str | None = None,
) -> list[str]:
    """Write the Zipfolio tearsheet for one backtest to ``output``.

    The page opens with a summary -- one card per part, each row a finding
    (``Section.finding``: measure, value, status) -- then the sections in four
    parts (``page.GROUPS``), each charts on the left and tables on the right:

    **Where the return came from**

    * **Performance** -- key metrics, cumulative returns, the top drawdowns and
      the underwater curve, always. Deliberately short: the QuantStats tearsheet
      beside this one covers returns in depth;
    * **Profit Concentration** -- the return without its best days; with round
      trips, their share of the net PnL in the best few, and a bootstrap;
    * **Instruments** -- when the strategy traded: each name's PnL and
      contribution, read mark-to-market from the book and the fills, and the
      spread of its daily and trade returns;
    * **Long and Short** -- when the book went short: what each side earned;
    * **Sector Analysis** -- when ``sector_mappings`` classifies a held name;
    * **Factor Analysis** -- when ``factor_model`` names one: ``"ff5_mom"`` (the
      default), ``"ff5"``, ``"ff3"`` or ``"ff3_mom"`` for Fama-French, which
      describe US stocks; ``"market"`` for a market model against the benchmark
      over ``risk_free`` -- a series of the rate in percent a year, such as
      €STR or SONIA, or ``"ruonia"`` to have it read -- which is what a book
      outside the US gets; ``None`` for none. With no ``risk_free`` the market
      model runs on raw returns and says so: nothing is downloaded for a rate
      the caller did not name. Exposures, alpha, attribution, rolling betas.
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
    * **CAPM and the Efficient Frontier** -- not a check, a map: the security
      market line and the capital market line against the benchmark, and the
      long-only efficient frontier of the names held most (from ``market_data``),
      which is hindsight and drawn as a ceiling;
    * **Risk Over Time** -- when the strategy traded: predicted against
      realized volatility, each name's share of risk, concentration;

    **How it holds and trades** -- **Positions**, **Transactions**, **Closed
    Trades** (pyfolio's round trips: one per closing sale), when the strategy traded;

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

    ``currency`` -- an ISO code in ``zipfolio.CURRENCIES``: ``"USD"``,
    ``"EUR"``, ``"GBP"``, ``"JPY"``, ``"CHF"``... -- is how every amount is
    written.

    ``run_info`` is what the header says about the run, label to value, in
    order -- the strategy and its version, the benchmark's ticker and whether
    it is a price or a total return, the universe -- before what the
    page knows itself (capital, how orders fill, costs). ``trials`` is how many
    versions of the strategy were backtested, when the caller knows: the
    Sharpe ratio is then judged against the best of that many strategies with
    no edge (the deflated Sharpe ratio), not against zero.

    The page opens with an **accounting integrity** panel (``zipfolio.integrity``):
    whether the numbers on it reconcile to the book -- fills, fees, cash,
    positions -- with the evidence. ``opening_state`` (the book before the
    first session; ``zipfolio.ziplime.opening_state`` for a ziplime run),
    ``contract_specs`` (what each instrument is; ``zipfolio.ziplime.contract_specs``)
    and ``portfolio_events`` (dividends, fees, flows, splits) let it replay the
    book from its start; without them it says what it could not check.
    ``integrity_report`` shows a report made beforehand instead, and
    ``integrity=False`` leaves the panel out.

    ``interactive`` draws the charts with Apache ECharts -- tooltips, legend
    toggles, zoom on dates -- with the library inlined, so the page still
    needs no network; ``False`` keeps every chart a static SVG.

    A strategy that traded one name gets no breakdown by name or sector --
    Instruments, Sector Analysis, risk contribution, concentration -- since
    each would say "100%" of that name; the summary says so once.

    Returns the titles of the sections that made it into the page, in page order.
    """
    import matplotlib.pyplot as plt
    import pyfolio as pf
    from cycler import cycler
    from pyfolio import utils

    raw_returns = returns
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
    names = sorted(transactions["symbol"].unique()) if transactions is not None else []
    if positions is not None:
        bars = market_data_frame(market_data, names, returns.index)
    rf, rf_name = risk_free_for(returns.index, factor_model, risk_free, factor_data)
    exposure = average_exposure(holdings)

    # Built in this order -- Portfolio Risk last of all, after pyfolio's styles
    # are no longer needed -- and laid out on the page by group (page.GROUPS).
    builders: list[tuple[str, Callable[[], Section | None]]] = [
        ("Performance", lambda: build_performance(returns, benchmark, positions, transactions, bars, rf, rf_name,
                                                  trials, holdings, orders,
                                                  external_flows(portfolio_events, returns.index))),
        ("Profit Concentration", lambda: build_profit_concentration(returns, positions, transactions)),
        ("Market Regimes", lambda: build_market_regimes(returns, benchmark, exposure)),
    ]
    sectors = sectors_of(positions, sector_mappings) if positions is not None else {}
    several = len(names) > 1
    if positions is not None:
        builders.append(("Positions", lambda: build_positions(returns, positions)))
        if several:
            builders.append(("Instruments", lambda: build_instruments(returns, holdings, positions, transactions,
                                                                      orders)))
        builders += [
            ("Long and Short", lambda: build_long_short(returns, holdings, transactions, orders)),
            ("Drawdowns", lambda: build_drawdowns(returns, holdings, transactions, orders, sectors or None,
                                                         benchmark)),
        ]
        if sectors and several:
            builders.append(("Sector Analysis", lambda: build_sector_analysis(
                returns, positions, transactions, sectors, holdings, orders)))
    if factor_model == "market":
        builders.append(("Factor Analysis", lambda: build_market_model_analysis(returns, benchmark, risk_free)))
    elif factor_model:
        builders.append(("Factor Analysis", lambda: build_factor_analysis(returns, factor_model, factor_data,
                                                                          benchmark=benchmark)))
    builders.append(("Portfolio Risk", lambda: build_portfolio_risk(returns, benchmark, positions, bars)))
    builders.append((CAPM_TITLE, lambda: build_capm(returns, benchmark, rf, rf_name, positions, bars)))
    if positions is not None:
        builders += [
            ("Risk Over Time", lambda: build_risk_over_time(returns, holdings, bars)),
            ("Transactions", lambda: build_transactions(returns, positions, transactions, market_timezone)),
            ("Next-Bar Fills", lambda: build_next_bar_fills(returns, holdings, raw_fills, fill_bars, cost_settings)),
            ("Execution", lambda: build_execution(raw_fills, orders, fill_bars, decision_prices, cost_settings)),
            ("Trading Costs", lambda: build_trading_costs(returns, positions, transactions, orders, cost_settings)),
            ("Closed Trades", lambda: build_round_trips(returns, positions, transactions, holdings, orders)),
        ]
        if bars is not None:
            builders.append(("Capacity", lambda: build_capacity(returns, positions, transactions, bars,
                                                                cost_settings)))

    sections: list[Section] = []
    built: list[str] = []
    rc = {**CHART_RC, "axes.prop_cycle": cycler(color=PALETTE)}
    # Matplotlib dates for every plot. Left to itself, pandas draws a series
    # whose frequency it can infer -- a window with no holiday in it -- in
    # period ordinals, and anything pyfolio adds with matplotlib directly on
    # the same axis lands off-screen: the exposure panel came out empty that way.
    token = INTERACTIVE.set(interactive)
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

    INTERACTIVE.reset(token)
    if not built:
        raise RuntimeError("no Zipfolio section could be built")
    if len(names) == 1:
        first = next(section for section in sections if not section.failed)
        first.finding(INFO, "One name traded", f"{names[0]}: no split by name or sector",
                      f"Every trade was in {names[0]}, so the breakdowns by name and by sector -- each 100% "
                      f"{names[0]} -- are left out. A stock picked knowing its history flatters any rule run on "
                      "it: the same rule on a group of stocks says whether the rule made the money.",
                      key="one_name")
        for key, label in (("top_names", "Profit from the top 2 names"), ("sector", "Largest sector, typical day")):
            first.finding(NA, label, "one name", f"One name traded, {names[0]}: there is nothing to compare it "
                          "with. The largest position says how concentrated the book was.", key=key)
    _frontier_beside_buy_and_hold(sections)
    with using_currency(currency):
        run = run_details(run_info, benchmark, cost_settings)

    panel = ""
    if integrity:
        from .sections.accounting_integrity import panel_html

        error = None
        if integrity_report is None:
            try:
                from .integrity import AccountingInputs, check_integrity

                with using_currency(currency):
                    integrity_report = check_integrity(AccountingInputs(
                        returns=raw_returns, position_records=position_records, cash=cash, transactions=raw_fills,
                        orders=orders, cost_settings=cost_settings, opening_state=opening_state,
                        portfolio_events=portfolio_events, contract_specs=contract_specs,
                        sector_mappings=sector_mappings, fill_bars=fill_bars, currency=currency.upper(),
                        run_id=run_id), integrity_policy, sections=sections)
            except Exception as failure:
                print(f"Warning: the accounting integrity check could not run:\n{traceback.format_exc()}")
                error = f"{type(failure).__name__}: {failure}"
        panel = panel_html(integrity_report, error)

    versions = f"analytics by pyfolio-reloaded {getattr(pf, '__version__', '')}"
    if "Portfolio Risk" in built:
        import skfolio

        versions += f" and skfolio {skfolio.__version__}"
    Path(output).write_text(
        page_html(title, returns, benchmark, versions, sections, run, panel),
        encoding="utf-8",
    )
    # In page order, which is the order a reader meets them in.
    return [section.title for _, members in ordered(sections) for section in members if section.title in built]


def _frontier_beside_buy_and_hold(sections: list[Section]) -> None:
    """The buy-and-hold check says where the efficient frontier of the same names puts the strategy.

    Both ask whether the strategy did more with its names than a passive holding
    of them: buy and hold is the mix a reader could have held, the frontier the
    best one hindsight could pick. The frontier stays out of the score.
    """
    found = [finding for section in sections if not section.failed for finding in section.findings]
    frontier = next((finding for finding in found if finding.section == CAPM_TITLE and finding.label == FRONTIER_FACT),
                    None)
    holding = next((finding for finding in found if finding.key == "buy_hold"), None)
    if frontier is not None and holding is not None:
        holding.note = f"{holding.note} {frontier.note}"
        holding.see = CAPM_TITLE


def run_details(run_info: dict[str, str] | None, benchmark: pd.Series | None, settings: dict | None) -> dict[str, str]:
    """What the header says about the run: the caller's facts first, then what the page knows itself."""
    run = {str(label): str(value) for label, value in (run_info or {}).items()}
    if "Benchmark" not in run:
        run["Benchmark"] = str(benchmark.name) if benchmark is not None and benchmark.name else (
            "Benchmark" if benchmark is not None else "none")
    settings = settings or {}
    if settings.get("capital"):
        run.setdefault("Capital", money(float(settings["capital"]), 0))
    if settings:
        reference = settings.get("price_used_in_order_execution") or "close"
        when = "the signal bar" if settings.get("same_bar_execution", True) else "the next bar"
        limit = settings.get("slippage_volume_limit")
        run.setdefault("Fills", f"at {when}'s {reference}, {float(settings.get('slippage_basis_points') or 0):g} bps "
                                "slippage" + (f", up to {float(limit):.0%} of a bar's volume" if limit else ""))
        if settings.get("commission_model") == "per_dollar":
            rate = settings.get("commission_rate")
            rate = settings.get("commission_per_share") if rate is None else rate
            run.setdefault("Commission", f"{float(rate):.4%} of value")
        elif settings.get("commission_per_share") is not None:
            run.setdefault("Commission", f"{money(float(settings['commission_per_share']), 4)} a share")
    run.setdefault("Luck test", f"{BOOTSTRAP_SAMPLES:,} redraws, seed 0")
    return run
