"""Zipfolio: what it is fed, and the page it writes.

Zipfolio reshapes a backtest into the frames pyfolio and skfolio expect and
draws one HTML page from what they compute. The shaping is tested without
pyfolio; the page needs it installed, and its risk section needs skfolio.
"""

import types

import numpy as np
import pandas as pd
import pytest

from zipfolio import (  # noqa: E402
    UNCLASSIFIED,
    build_tearsheet,
    daily_returns,
    market_data_frame,
    positions_in_dollars,
    transactions_frame,
    us_sector_mappings,
)


def synthetic_factors(start="2015-01-01", end="2027-12-31", seed=21):
    """Fama-French-shaped daily factors, decimals, at midnight UTC -- no download in tests."""
    days = pd.bdate_range(start, end, tz="UTC")
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(0, 0.006, (len(days), 6)), index=days,
                         columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"])
    frame["RF"] = 0.0001
    return frame


@pytest.fixture(autouse=True)
def no_factor_download(monkeypatch):
    """The page would fetch Kenneth French's files; tests use synthetic factors instead."""
    try:
        import zipfolio.sections.factor_analysis as factor_analysis
    except ImportError:
        return
    monkeypatch.setattr(factor_analysis, "fama_french_daily", lambda model="ff5_mom": synthetic_factors())


def session_closes(start="2023-01-03", end="2023-12-29"):
    """New York session closes, tz removed as ``run_backtest`` removes it, holidays included."""
    days = pd.bdate_range(start, end).difference(pd.to_datetime(["2023-07-04", "2023-11-23"]))
    return days + pd.Timedelta(hours=21)


def a_traded_book(closes, seed=7):
    """Two symbols rebalanced weekly: the backend positions, cash and transactions frames."""
    rng = np.random.default_rng(seed)
    prices = {"AAPL@XNAS": 150.0, "MSFT@XNAS": 250.0}
    held = dict.fromkeys(prices, 0)
    cash, cash_by_day, records, fills = 100_000.0, [], [], []
    for i, close in enumerate(closes):
        for symbol in prices:
            prices[symbol] *= 1 + rng.normal(0, 0.01)
        if i % 5 == 0:
            for symbol in prices:
                target = int(rng.integers(-50, 150))
                amount = target - held[symbol]
                if amount:
                    fills.append({
                        "id": f"{i}-{symbol}", "order_id": None,
                        "dt": (close - pd.Timedelta(hours=2)).tz_localize("UTC"),
                        "symbol": symbol, "quantity": amount, "price": prices[symbol],
                        "commission": 1.0,
                    })
                    cash -= amount * prices[symbol]
                    held[symbol] = target
        for symbol, quantity in held.items():
            if quantity:
                records.append({
                    "date": close.strftime("%Y-%m-%d"), "symbol": symbol,
                    "quantity": float(quantity), "cost_basis": prices[symbol],
                    "last_sale_price": prices[symbol],
                })
        cash_by_day.append(cash)
    returns = pd.Series(rng.normal(0.0004, 0.01, len(closes)), index=closes)
    return returns, pd.DataFrame(records), pd.Series(cash_by_day, index=closes), pd.DataFrame(fills)


# ----------------------------------------------------------------------------
# Shaping: one row per session, at midnight UTC.
# ----------------------------------------------------------------------------

def test_returns_become_one_row_per_session_at_midnight_utc():
    closes = session_closes("2023-03-01", "2023-03-10")
    returns = daily_returns(pd.Series(0.01, index=closes))

    assert str(returns.index.tz) == "UTC"
    assert (returns.index == returns.index.normalize()).all()
    assert len(returns) == len(closes)


def test_intraday_returns_are_compounded_not_dropped():
    bars = pd.to_datetime(["2023-03-01 15:00", "2023-03-01 16:00", "2023-03-02 15:00"])
    returns = daily_returns(pd.Series([0.10, 0.10, 0.05], index=bars))

    assert returns.iloc[0] == pytest.approx(0.21)
    assert returns.iloc[1] == pytest.approx(0.05)


def test_positions_are_dollars_per_symbol_with_cash_and_zeros_on_flat_days():
    closes = session_closes("2023-03-01", "2023-03-03")
    dates = daily_returns(pd.Series(0.0, index=closes)).index
    records = pd.DataFrame([
        {"date": "2023-03-01", "symbol": "AAPL@XNAS", "quantity": 10.0, "last_sale_price": 150.0},
        {"date": "2023-03-01", "symbol": "MSFT@XNAS", "quantity": -2.0, "last_sale_price": 250.0},
        {"date": "2023-03-03", "symbol": "AAPL@XNAS", "quantity": 5.0, "last_sale_price": 160.0},
    ])
    cash = pd.Series([1000.0, 2500.0, 1700.0], index=closes)

    positions = positions_in_dollars(records, cash, dates)

    assert list(positions.columns) == ["AAPL@XNAS", "MSFT@XNAS", "cash"]
    assert positions.loc[dates[0]].tolist() == [1500.0, -500.0, 1000.0]
    assert positions.loc[dates[1]].tolist() == [0.0, 0.0, 2500.0], "a flat day is zeros, not missing"
    assert positions.loc[dates[2]].tolist() == [800.0, 0.0, 1700.0]


def test_no_cash_means_no_positions_frame():
    """Without cash pyfolio would read leverage off a portfolio with no capital in it."""
    dates = daily_returns(pd.Series(0.0, index=session_closes("2023-03-01", "2023-03-03"))).index
    assert positions_in_dollars(pd.DataFrame(), pd.Series(dtype=float), dates) is None


def test_transactions_are_indexed_by_fill_time_and_drop_empty_fills():
    fills = pd.DataFrame([
        {"dt": pd.Timestamp("2023-03-01 19:00", tz="UTC"), "symbol": "AAPL@XNAS", "quantity": 10, "price": 150.0},
        {"dt": pd.Timestamp("2023-03-02 19:00", tz="UTC"), "symbol": "AAPL@XNAS", "quantity": 0, "price": 151.0},
    ])

    frame = transactions_frame(fills)

    assert list(frame.columns) == ["amount", "price", "symbol"]
    assert len(frame) == 1
    assert transactions_frame(pd.DataFrame()) is None


def daily_bars_for(closes, symbols, volume=5_000_000.0):
    """What ``backtesting._daily_bars`` hands over: one row per symbol per session."""
    rng = np.random.default_rng(11)
    return pd.DataFrame([
        {"symbol": symbol, "day": close.date(), "price": 100 + rng.normal(0, 1), "volume": volume}
        for close in closes for symbol in symbols
    ])


def test_sectors_come_only_for_us_listings(tmp_path):
    """The table is keyed by ticker; Moscow's T is not AT&T."""
    table = tmp_path / "gics.csv"
    table.write_text("symbol,gics_sector\nAAPL,45\nJPM,40\nT,50\n")

    sectors = us_sector_mappings(["AAPL@XNGS", "JPM@XNYS", "T@MISX", "SPY@ARCX", "SBER@MISX"], table=table)

    assert sectors == {"AAPL": "Information Technology", "JPM": "Financials"}


def test_no_sector_table_means_no_sectors(monkeypatch):
    """The package ships no GICS data; without a table there is simply no sector section."""
    import zipfolio.sectors as sectors

    monkeypatch.setattr(sectors, "_table_path", lambda table: None)
    assert us_sector_mappings(["AAPL@XNGS"]) == {}


def test_the_sector_table_can_come_from_the_environment(monkeypatch, tmp_path):
    table = tmp_path / "gics.csv"
    table.write_text("symbol,gics_sector\nXOM,10\n")
    monkeypatch.setenv("ZIPFOLIO_GICS_TABLE", str(table))

    assert us_sector_mappings(["XOM@XNYS"]) == {"XOM": "Energy"}


def test_market_data_is_price_and_volume_rows_for_the_traded_symbols_only():
    closes = session_closes("2023-03-01", "2023-03-03")
    dates = daily_returns(pd.Series(0.0, index=closes)).index
    bars = daily_bars_for(closes, ["AAPL", "MSFT", "SPY"])

    frame = market_data_frame(bars, ["AAPL", "MSFT"], dates)

    assert list(frame.columns) == ["AAPL", "MSFT"], "the benchmark is not something the strategy traded"
    assert frame.index.names == ["date", "field"]
    assert len(frame.xs("price", level=1)) == len(dates)
    assert (frame.xs("volume", level=1) == 5_000_000.0).all().all()
    assert market_data_frame(pd.DataFrame(), ["AAPL"], dates) is None


# ----------------------------------------------------------------------------
# The page: needs pyfolio.
# ----------------------------------------------------------------------------

@pytest.fixture
def pyfolio():
    return pytest.importorskip("pyfolio", reason="pyfolio-reloaded is not installed here")


def test_a_traded_backtest_gets_every_section(pyfolio, tmp_path):
    closes = session_closes()
    returns, records, cash, fills = a_traded_book(closes)
    benchmark = pd.Series(np.random.default_rng(1).normal(0.0003, 0.009, len(closes)), index=closes)
    output = tmp_path / "zipfolio.html"

    built = build_tearsheet(output, returns, benchmark, records, cash, fills)

    assert [name for name in built if name != "Portfolio Risk"] == [
        "Performance", "Profit Concentration", "Instruments", "Long and Short", "Factor Analysis",
        "Drawdowns", "Market Regimes", "Risk Over Time",
        "Positions", "Transactions", "Closed Trades",
        "Trading Costs"], "page order: returns, risk, the book, then whether it can be traded"
    page = output.read_text(encoding="utf-8")
    assert "Key Performance Metrics" in page and "Annual return" in page, (
        "the perf stats table was caught at print_table and rendered in the page's own style")
    assert page.count("<svg") >= 15, "every chart is a figure of its own"
    assert "could not be computed" not in page
    assert "#6b9e00" in page, "the strategy is drawn in QuantStats Pro's colour"


def test_a_backtest_that_never_traded_still_gets_its_returns(pyfolio, tmp_path):
    closes = session_closes()
    returns = pd.Series(np.random.default_rng(3).normal(0, 0.01, len(closes)), index=closes)
    output = tmp_path / "zipfolio.html"

    built = build_tearsheet(
        output, returns, pd.Series(0.0, index=closes),
        pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(),
    )

    assert [name for name in built if name != "Portfolio Risk"] == [
        "Performance", "Profit Concentration", "Factor Analysis"]
    assert "Benchmark: none" in output.read_text(encoding="utf-8"), (
        "the zeros a run without a benchmark is handed are not a benchmark"
    )


def test_performance_is_only_what_quantstats_does_not_already_show(pyfolio, tmp_path):
    """The platform shows the QuantStats tearsheet beside this one; returns are not repeated here."""
    closes = session_closes()
    returns = pd.Series(np.random.default_rng(5).normal(0, 0.01, len(closes)), index=closes)
    benchmark = pd.Series(np.random.default_rng(6).normal(0, 0.01, len(closes)), index=closes)
    output = tmp_path / "zipfolio.html"

    built = build_tearsheet(output, returns, benchmark)

    assert "Stress Events" not in built
    page = output.read_text(encoding="utf-8")
    performance = page.split("<h2>Performance</h2>")[1].split("</section>")[0]
    assert performance.count("<svg") == 2, "cumulative returns and the underwater curve"
    assert "Key Performance Metrics" in performance
    for repeated in ("Rolling Sharpe", "Rolling Volatility", "Rolling Beta", "Monthly Returns (%)",
                     "EOY Returns", "Return Quantiles", "Volatility Matched", "Log Scaled",
                     "Worst 5 Drawdowns", "Top 5 Drawdown Periods"):
        assert repeated not in page, repeated


def test_one_session_is_not_a_tearsheet(pyfolio, tmp_path):
    closes = session_closes("2023-03-01", "2023-03-01")
    with pytest.raises(ValueError):
        build_tearsheet(tmp_path / "zipfolio.html", pd.Series([0.01], index=closes))


def test_sector_analysis_and_capacity_when_sectors_and_bars_are_given(pyfolio, tmp_path):
    """Both names trade on the same days -- the case pyfolio's own capacity join fails on."""
    closes = session_closes()
    returns, records, cash, fills = a_traded_book(closes)
    output = tmp_path / "zipfolio.html"

    built = build_tearsheet(
        output, returns, None, records, cash, fills,
        market_data=daily_bars_for(closes, ["AAPL@XNAS", "MSFT@XNAS"], volume=1_000.0),
        sector_mappings={"AAPL@XNAS": "Information Technology"},
    )

    assert "Sector Analysis" in built
    assert "Capacity" in built
    page = output.read_text(encoding="utf-8")
    assert "Information Technology" in page
    assert UNCLASSIFIED in page, "a held name with no sector is filed, not dropped"
    assert "Fills Over 5% of Daily Volume" in page, (
        "two names filled on the same day: pyfolio's own join fails on exactly that")
    assert "could not be computed" not in page


def test_no_sector_section_when_no_held_name_has_a_sector(pyfolio, tmp_path):
    closes = session_closes()
    returns, records, cash, fills = a_traded_book(closes)

    built = build_tearsheet(
        tmp_path / "zipfolio.html", returns, None, records, cash, fills,
        sector_mappings={"SBER": "Financials"},
    )

    assert "Sector Analysis" not in built


def test_a_failing_section_leaves_a_note_and_no_traceback_on_the_page(pyfolio, monkeypatch, tmp_path):
    from pyfolio import txn

    def broken(*args, **kwargs):
        raise RuntimeError("pyfolio fell over")

    monkeypatch.setattr(txn, "get_turnover", broken)
    closes = session_closes()
    returns, records, cash, fills = a_traded_book(closes)
    output = tmp_path / "zipfolio.html"

    built = build_tearsheet(output, returns, None, records, cash, fills)

    assert "Transactions" not in built
    page = output.read_text(encoding="utf-8")
    assert "could not be computed" in page
    assert "Traceback" not in page and "fell over" not in page, "the warning goes to the log, not the page"


def test_the_page_says_who_made_it_and_when(pyfolio, tmp_path):
    closes = session_closes("2023-01-03", "2023-03-31")
    output = tmp_path / "zipfolio.html"
    build_tearsheet(output, pd.Series(0.001, index=closes))

    page = output.read_text(encoding="utf-8")
    from zipfolio import __version__

    assert f"<strong>ZipLime Zipfolio v{__version__}</strong></a> on " in page
    assert f">Zipfolio v{__version__}</a>" in page, "the footer names the version that built the page"
    assert "<title>Zipfolio Tearsheet: Portfolio, Execution &amp; Cost Analysis</title>" in page
    assert "part of the ZipLime ecosystem" in page
    assert page.count('href="https://github.com/Limex-com/zipfolio"') >= 2, "the header and the footer link home"
    assert page.count(" UTC") >= 2, "the date is in the header and the footer"


def test_heatmap_numbers_are_made_to_fit_their_cells(pyfolio):
    import matplotlib.pyplot as plt
    import seaborn as sns

    from zipfolio.style import declutter

    values = np.random.default_rng(2).normal(0, 8, (6, 12))
    figure, axis = plt.subplots(figsize=(4, 3))
    sns.heatmap(values, annot=True, annot_kws={"size": 9}, ax=axis, cbar=False)

    declutter(figure)

    renderer = figure.canvas.get_renderer()
    cell = axis.get_window_extent(renderer).width / 12
    assert max(text.get_window_extent(renderer).width for text in axis.texts) <= cell
    plt.close(figure)


def test_dense_axis_labels_are_thinned(pyfolio):
    import matplotlib.pyplot as plt

    from zipfolio.style import declutter

    figure, axis = plt.subplots(figsize=(4, 2))
    axis.set_xticks(range(14))
    axis.set_xticklabels([f"{9 + i // 2:02d}:{30 * (i % 2):02d}" for i in range(14)])

    declutter(figure)

    visible = [label.get_text() for label in axis.get_xticklabels() if label.get_visible()]
    assert 1 < len(visible) < 14
    assert visible[0] == "09:00"
    plt.close(figure)


def test_print_table_is_restored_after_the_page_is_built(pyfolio, tmp_path):
    from pyfolio import round_trips, utils

    originals = (utils.print_table, round_trips.print_table)
    closes = session_closes()
    returns, records, cash, fills = a_traded_book(closes)
    build_tearsheet(tmp_path / "zipfolio.html", returns, None, records, cash, fills)

    assert (utils.print_table, round_trips.print_table) == originals


def test_a_chart_that_fails_is_left_out_and_the_section_stays(pyfolio, monkeypatch, tmp_path):
    from pyfolio import plotting

    def broken(*args, **kwargs):
        raise RuntimeError("no gross leverage today")

    monkeypatch.setattr(plotting, "plot_gross_leverage", broken)
    closes = session_closes()
    returns, records, cash, fills = a_traded_book(closes)
    output = tmp_path / "zipfolio.html"

    built = build_tearsheet(output, returns, None, records, cash, fills)

    assert "Positions" in built
    assert "Gross Leverage" not in output.read_text(encoding="utf-8")


def test_fill_times_follow_the_market_clock(pyfolio, tmp_path):
    """pyfolio's own chart is fixed to New York's 09:30-16:00; a Moscow evening fill falls off it."""
    import matplotlib.pyplot as plt

    from zipfolio.sections.transactions import plot_fill_times

    fills = pd.DataFrame(
        {"amount": [10, 10], "price": [100.0, 100.0], "symbol": ["SBER", "SBER"]},
        index=pd.DatetimeIndex(["2024-03-01 07:30", "2024-03-01 18:00"], tz="UTC"),
    )
    figure, axis = plt.subplots()
    plot_fill_times(fills, "Europe/Moscow", axis)

    labels = [label.get_text() for label in axis.get_xticklabels()]
    assert labels[0] <= "10:00" and labels[-1] >= "21:00"
    assert sum(patch.get_height() for patch in axis.patches) == pytest.approx(1.0)
    plt.close(figure)


# ----------------------------------------------------------------------------
# Wiring: announced as its own report kind, deleted once S3 has it.
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Trading costs: the model in words, the costs measured.
# ----------------------------------------------------------------------------

COST_SETTINGS = {
    "commission_model": "per_share", "commission_per_share": 0.005, "commission_rate": None,
    "commission_min_trade_cost": 1.0, "slippage_basis_points": 10.0, "slippage_volume_limit": 0.1,
    "price_used_in_order_execution": "close", "same_bar_execution": False, "capital": 100000,
}


def orders_for(fills, per_share=0.005, minimum=1.0):
    """The orders ziplime would record for ``fills``: one per fill, commission on the order."""
    return pd.DataFrame({
        "id": fills["id"], "dt": fills["dt"], "created": fills["dt"], "type": "market",
        "symbol": fills["symbol"], "quantity": fills["quantity"], "filled": fills["quantity"],
        "status": "OrderStatus.FILLED",
        "commission": np.maximum(fills["quantity"].abs() * per_share, minimum),
    })


def test_the_cost_model_is_described_from_the_settings():
    from zipfolio.sections.trading_costs import cost_model_text

    text = " ".join(cost_model_text(COST_SETTINGS))

    assert "$0.0050 per share" in text and "$1.00 per order" in text
    assert "10 bps worse" in text and "close &times; 1.0010" in text
    assert "10% of the bar's volume" in text
    assert "next bar" in text
    assert "Not modeled" in text

    per_dollar = " ".join(cost_model_text({**COST_SETTINGS, "commission_model": "per_dollar",
                                             "commission_rate": 0.0002}))
    assert "0.0200% of the value of each fill" in per_dollar


def test_trading_costs_read_commission_from_orders_and_rebuild_slippage(pyfolio, tmp_path):
    closes = session_closes()
    returns, records, cash, fills = a_traded_book(closes)
    orders = orders_for(fills)
    output = tmp_path / "zipfolio.html"

    built = build_tearsheet(output, returns, None, records, cash, fills,
                                    orders=orders, cost_settings=COST_SETTINGS)

    assert "Trading Costs" in built
    page = output.read_text(encoding="utf-8")
    assert f"${orders['commission'].sum():,.2f}" in page, "commission is the orders' own, summed"
    value = (fills["quantity"] * fills["price"]).abs()
    rate = 10 / 1e4
    slippage = (value * rate / np.where(fills["quantity"] > 0, 1 + rate, 1 - rate)).sum()
    assert f"${slippage:,.2f}" in page, "slippage is what the fixed-bps model put into the fill prices"
    assert "Orders paying the minimum" in page
    assert "How Costs Are Modeled" in page


def test_trading_costs_without_settings_still_measure_commission(pyfolio, tmp_path):
    closes = session_closes()
    returns, records, cash, fills = a_traded_book(closes)
    output = tmp_path / "zipfolio.html"

    build_tearsheet(output, returns, None, records, cash, fills, orders=orders_for(fills))

    page = output.read_text(encoding="utf-8")
    assert "were not passed to the report" in page


# ----------------------------------------------------------------------------
# Portfolio risk: skfolio.
# ----------------------------------------------------------------------------

@pytest.fixture
def skfolio():
    return pytest.importorskip("skfolio", reason="skfolio is not installed here")


def test_portfolio_risk_measures_contribution_and_alternatives(pyfolio, skfolio, tmp_path):
    closes = session_closes("2021-01-04", "2023-12-29")
    returns, records, cash, fills = a_traded_book(closes)
    output = tmp_path / "zipfolio.html"

    built = build_tearsheet(
        output, returns, None, records, cash, fills,
        market_data=daily_bars_for(closes, ["AAPL@XNAS", "MSFT@XNAS"]))

    assert "Portfolio Risk" in built
    page = output.read_text(encoding="utf-8")
    for heading in ("Risk Measures", "CVaR 95% (daily)", "CDaR 95%", "Ulcer index", "Rolling Tail Risk",
                    "Risk Contribution", "Same Names, Other Allocations", "Equal Weighted"):
        assert heading in page, heading
    assert "skfolio" in page, "the footer credits the library"


def test_portfolio_risk_needs_only_returns(pyfolio, skfolio, tmp_path):
    closes = session_closes()
    output = tmp_path / "zipfolio.html"
    built = build_tearsheet(output, pd.Series(np.random.default_rng(4).normal(0, 0.01, len(closes)),
                                                      index=closes))

    assert "Portfolio Risk" in built
    page = output.read_text(encoding="utf-8")
    assert "Risk Measures" in page and "Risk Contribution" not in page


# ----------------------------------------------------------------------------
# Instruments: each name's PnL, read from the book and the fills.
# ----------------------------------------------------------------------------

def test_instrument_pnl_adds_up_to_the_portfolio_pnl():
    """Mark-to-market per name, nothing estimated: the names add up to the book, day by day."""
    from zipfolio.sections.instruments import instrument_pnl

    closes = session_closes()
    returns, records, cash, fills = a_traded_book(closes)
    dates = daily_returns(returns).index
    holdings = positions_in_dollars(records, cash, dates)

    pnl = instrument_pnl(holdings, transactions_frame(fills))

    book = holdings.sum(axis="columns")
    expected = book.diff()
    expected.iloc[0] = book.iloc[0] - 100_000.0
    assert pnl.sum(axis="columns").to_numpy() == pytest.approx(expected.to_numpy(), abs=1e-6)
    assert set(pnl.columns) == {"AAPL@XNAS", "MSFT@XNAS"}


def test_instrument_commission_comes_off_its_own_pnl():
    from zipfolio.sections.instruments import commission_by_day

    closes = session_closes("2023-03-01", "2023-03-03")
    dates = daily_returns(pd.Series(0.0, index=closes)).index
    orders = pd.DataFrame({
        "id": ["a", "a", "b"], "symbol": ["AAPL", "AAPL", "MSFT"],
        "dt": pd.to_datetime(["2023-03-01 20:00", "2023-03-01 20:00", "2023-03-02 20:00"], utc=True),
        "commission": [0.5, 1.0, 2.0],
    })

    charged = commission_by_day(orders, dates, ["AAPL", "MSFT"])

    assert charged["AAPL"].sum() == 1.0, "an order's commission is cumulative; the last record counts"
    assert charged.loc[dates[1], "MSFT"] == 2.0


def test_the_instruments_section_breaks_returns_down_by_name(pyfolio, tmp_path):
    closes = session_closes("2022-01-03", "2023-12-29")
    returns, records, cash, fills = a_traded_book(closes)
    output = tmp_path / "zipfolio.html"

    built = build_tearsheet(output, returns, None, records, cash, fills, orders=orders_for(fills))

    assert built.index("Instruments") < built.index("Positions"), "what each name earned comes with the returns"
    page = output.read_text(encoding="utf-8")
    section = page.split("<h2>Instruments</h2>")[1].split("</section>")[0]
    for heading in ("Returns by Instrument", "Contribution to Return by Instrument",
                    "Cumulative PnL by Instrument", "Contribution by Instrument and Year",
                    "Daily Returns by Instrument", "Trade Returns by Instrument", "Win Rate"):
        assert heading in section, heading
    assert "could not be computed" not in page


# ----------------------------------------------------------------------------
# Execution: every fill against its bar, every order against its decision.
# ----------------------------------------------------------------------------

def execution_inputs():
    """Two orders, three fills: one fill priced above its bar's high, one taking 20% of its bar."""
    fills = pd.DataFrame({
        "id": ["f1", "f2", "f3"], "order_id": ["o1", "o1", "o2"],
        "dt": pd.to_datetime(["2024-03-01 21:00", "2024-03-04 21:00", "2024-03-01 21:00"], utc=True),
        "symbol": ["AAPL", "AAPL", "MSFT"], "quantity": [100, 50, -40], "price": [101.0, 99.0, 50.0],
        "commission": [None, None, None],
    })
    bars = pd.DataFrame({
        "open": [99.0, 98.5, 50.5], "high": [100.5, 99.5, 51.0], "low": [98.0, 98.0, 49.5],
        "close": [100.4, 99.0, 50.0], "volume": [1_000_000, 250.0, 100_000],
    }, index=["f1", "f2", "f3"])
    orders = pd.DataFrame({
        "id": ["o1", "o2", "o3", "o4"],
        "created": pd.to_datetime(["2024-03-01 21:00", "2024-03-01 21:00", "2024-03-01 21:00",
                                   "2024-03-01 21:00"], utc=True),
        "dt": pd.to_datetime(["2024-03-04 21:00", "2024-03-01 21:00", "2024-03-01 21:00",
                              "2024-03-01 21:00"], utc=True),
        "type": ["market", "market", "limit", "market"], "reason": [None, None, "price never reached", None],
        "symbol": ["AAPL", "MSFT", "NVDA", "JPM"], "quantity": [150, -40, 10, 0], "filled": [150, -40, 0, 0],
        "commission": [0.15, 0.04, 0.0, 0.0], "stop": [None] * 4, "limit": [None, None, 90.0, None],
        "status": ["OrderStatus.FILLED", "OrderStatus.FILLED", "OrderStatus.CANCELLED", "OrderStatus.FILLED"],
    })
    decisions = pd.DataFrame({"decision_price": [100.0, 50.0, 95.0, 120.0]}, index=["o1", "o2", "o3", "o4"])
    return fills, bars, orders, decisions


def test_fills_are_checked_against_their_bar():
    from zipfolio.sections.execution import fill_quality

    fills, bars, orders, decisions = execution_inputs()
    quality = fill_quality(fills, bars, decisions, orders).set_index("id")

    assert quality.loc["f1", "above_high"], "101 is above a bar whose high was 100.5"
    assert not quality.loc["f3", "above_high"] and not quality.loc["f3", "below_low"]
    assert quality.loc["f2", "participation"] == pytest.approx(0.2)
    assert quality.loc["f1", "shortfall_bps"] == pytest.approx(100.0), "bought at 101 having seen 100"
    assert quality.loc["f3", "shortfall_bps"] == pytest.approx(0.0), "a sell at the price it saw costs nothing"
    assert quality.loc["f1", "delay_bps"] + quality.loc["f1", "slippage_bps"] == pytest.approx(100.0)


def test_orders_are_followed_from_creation_to_their_last_fill():
    from zipfolio.sections.execution import fill_quality, order_book

    fills, bars, orders, decisions = execution_inputs()
    book = order_book(orders, fill_quality(fills, bars, decisions, orders)).set_index("id")

    assert book.loc["o1", "fills"] == 2 and book.loc["o1", "state"] == "filled"
    assert book.loc["o1", "delay"] == pd.Timedelta(0)
    assert book.loc["o3", "state"] == "unfilled"


def test_the_execution_section_flags_what_the_market_would_not_have_given(pyfolio):
    from zipfolio.sections.execution import build_execution

    fills, bars, orders, decisions = execution_inputs()
    settings = {"same_bar_execution": True, "price_used_in_order_execution": "close",
                "slippage_basis_points": 5, "slippage_volume_limit": 0.1}

    section = build_execution(fills, orders, bars, decisions, settings)

    side = " ".join(section.side)
    assert "Fills at the close of the signal bar" in side, "same-bar close fills are flagged"
    assert "priced outside their bar" in side
    assert "Size against liquidity" in side and "mark bad" in side, "20% of a bar is over the 10% limit"
    assert "Orders for zero shares" in side, "a zero-share rebalance order is neither a fill nor a failure"
    assert "price never reached" in "".join(section.wide + section.side)


def test_next_bar_execution_passes_the_timing_check(pyfolio):
    from zipfolio.sections.execution import build_execution

    fills, bars, orders, decisions = execution_inputs()
    section = build_execution(fills, orders, bars, decisions,
                              {"same_bar_execution": False, "price_used_in_order_execution": "open"})

    assert "Fills on the bar after the signal" in " ".join(section.side)


# ----------------------------------------------------------------------------
# Factor analysis.
# ----------------------------------------------------------------------------

def test_factor_betas_are_recovered(pyfolio):
    from zipfolio.sections.factor_analysis import build_factor_analysis

    factors = synthetic_factors("2021-01-01", "2023-12-31")
    rng = np.random.default_rng(3)
    returns = (factors["RF"] + 1.2 * factors["Mkt-RF"] + 0.5 * factors["SMB"]
               + rng.normal(0, 0.002, len(factors))).rename("Strategy")

    section = build_factor_analysis(returns, "ff5_mom", factors)

    import re

    table = section.side[0]
    assert "Fama-French 5 factors + momentum" in table

    def beta(name):
        return float(re.search(rf"<td>{re.escape(name)}</td><td>(-?[\d.]+)</td>", table).group(1))

    assert beta("Mkt-RF") == pytest.approx(1.2, abs=0.03)
    assert beta("SMB") == pytest.approx(0.5, abs=0.03)
    assert abs(beta("HML")) < 0.05
    assert len(section.charts) == 4, "exposures, rolling betas, decomposition, attribution by year"


def test_factor_analysis_says_so_when_the_data_cannot_be_had(pyfolio, monkeypatch):
    import zipfolio.sections.factor_analysis as factor_analysis

    def offline(model="ff5_mom"):
        raise OSError("no network")

    monkeypatch.setattr(factor_analysis, "fama_french_daily", offline)
    closes = session_closes()
    section = factor_analysis.build_factor_analysis(daily_returns(pd.Series(0.001, index=closes)))

    assert "could not be downloaded" in section.side[0]


def test_the_french_library_csv_is_parsed(tmp_path):
    """The library's own layout: notes, a header starting with a comma, YYYYMMDD rows in percent."""
    import io
    import zipfile

    from zipfolio.factors import _parse

    text = ("This file was created by using the 202608 CRSP database.\r\n\r\n"
            ",Mkt-RF,SMB,HML,RMW,CMA,RF\r\n"
            "20260828,   -0.50,    0.10,   -0.20,    0.30,    0.00,    0.01\r\n"
            "20260831,   -0.33,   -0.29,   -0.39,   -1.09,    0.03,    0.01\r\n\r\n"
            "Copyright 2026 Eugene F. Fama and Kenneth R. French\r\n")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("F-F_Research_Data_5_Factors_2x3_daily.csv", text)

    frame = _parse(buffer.getvalue())

    assert list(frame.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert len(frame) == 2 and str(frame.index.tz) == "UTC"
    assert frame.loc["2026-08-31", "RMW"].item() == -1.09, "still in percent; the loader divides"


# ----------------------------------------------------------------------------
# zipfolio.ziplime: what is read from a ziplime run.
# ----------------------------------------------------------------------------

def test_each_fill_is_matched_to_the_bar_it_happened_on():
    pl = pytest.importorskip("polars")

    from zipfolio.ziplime import execution_bars

    fills = pd.DataFrame({"id": ["t1"], "symbol": ["AAPL"],
                          "dt": [pd.Timestamp("2024-03-04 21:00", tz="UTC")]})
    orders = pd.DataFrame({"id": ["o1"], "symbol": ["AAPL"],
                           "created": [pd.Timestamp("2024-03-01 21:00", tz="UTC")]})
    bars = pl.DataFrame({
        "symbol": ["AAPL", "AAPL", "AAPL"],
        "date": pd.to_datetime(["2024-03-01 05:00", "2024-03-04 05:00", "2024-03-05 05:00"], utc=True),
        "open": [99.0, 100.0, 103.0], "high": [100.0, 102.0, 104.0], "low": [98.0, 99.5, 102.0],
        "close": [99.5, 101.5, 103.5], "volume": [1e6, 2e6, 3e6],
    })

    found = execution_bars(types.SimpleNamespace(data=bars), fills, orders)

    assert found["fill_bars"].loc["t1", "high"] == 102.0, "the fill's own bar, not the next one"
    assert found["decision_prices"].loc["o1", "decision_price"] == 99.5, "the close the order was placed on"


def test_order_types_come_from_the_execution_style():

    from zipfolio.ziplime import order_type

    class LimitOrder:
        pass

    assert order_type(types.SimpleNamespace(execution_style=LimitOrder())) == "limit"


def test_trading_activity_describes_how_the_strategy_trades():
    from zipfolio.sections.transactions import trading_activity

    closes = session_closes()
    returns, records, cash, fills = a_traded_book(closes)
    frame = transactions_frame(fills)
    turnover = pd.Series(0.05, index=daily_returns(returns).index)

    activity = trading_activity(daily_returns(returns), frame, turnover)

    assert activity["Median sessions between trading days"] == "5", "the book rebalances every fifth session"
    assert activity["Annual turnover"] == "1260%"
    for row in ("Buys", "Sells", "Avg. fill value", "Largest fill", "Avg. symbols per trading day"):
        assert row in activity.index


def test_an_unheld_name_is_a_gap_not_a_zero():
    """Zero weight means "not held"; drawn as a value it is a vertical drop at every exit."""
    from zipfolio.style import held_only

    days = pd.bdate_range("2024-01-01", periods=5, tz="UTC")
    allocation = pd.DataFrame({"AAPL": [0.3, 0.3, 0.0, 0.0, 0.3], "cash": [0.7, 0.7, 1.0, 1.0, 0.7]}, index=days)

    masked = held_only(allocation)

    assert masked["AAPL"].isna().tolist() == [False, False, True, True, False]
    assert masked["cash"].tolist() == allocation["cash"].tolist(), "cash is not a holding"


def test_a_one_session_holding_still_shows(pyfolio):
    import matplotlib.pyplot as plt

    from zipfolio.style import mark_isolated

    figure, ax = plt.subplots()
    ax.plot(range(5), [np.nan, 0.3, np.nan, 0.2, 0.25])
    mark_isolated(ax)

    offsets = ax.collections[0].get_offsets()
    assert offsets.tolist() == [[1.0, 0.3]], "only the point with no neighbour gets a dot"
    plt.close(figure)



# ----------------------------------------------------------------------------
# Moscow: roubles, MOEX sectors, a market model against IMOEX over RUONIA.
# ----------------------------------------------------------------------------

def test_money_is_written_in_the_report_currency():
    from zipfolio.currency import millions, money, using_currency

    assert money(-5067.13) == "-$5,067.13"
    with using_currency("RUB"):
        assert money(1250.5) == "1,250.50 \u20bd"
        assert money(-3, 0) == "-3 \u20bd"
        assert millions(1e8) == "100m \u20bd"
    assert money(1) == "$1.00", "the currency is the page's, and only for as long as it is built"


def test_risk_free_accrues_the_day_before_rate_over_calendar_days():
    from zipfolio.factors import risk_free_returns

    rate = pd.Series([10.0, 20.0], index=pd.DatetimeIndex(["2024-03-01", "2024-03-04"], tz="UTC"))
    sessions = pd.DatetimeIndex(["2024-03-01", "2024-03-04", "2024-03-05"], tz="UTC")

    rf = risk_free_returns(rate, sessions)

    assert rf.loc["2024-03-04"] == pytest.approx(0.10 * 3 / 365), "Monday earns Friday's rate for three days"
    assert rf.loc["2024-03-05"] == pytest.approx(0.20 * 1 / 365), "not a rate published after the day"


def test_ruonia_is_read_from_the_bank_of_russia_reply(monkeypatch, tmp_path):
    import zipfolio.factors as factors

    reply = (b'<?xml version="1.0" encoding="utf-8"?><soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">'
             b'<soap:Body><RuoniaResponse xmlns="http://web.cbr.ru/"><RuoniaResult><diffgr:diffgram '
             b'xmlns:diffgr="urn:schemas-microsoft-com:xml-diffgram-v1"><Ruonia xmlns="">'
             b'<ro><D0>2026-08-03T00:00:00+03:00</D0><ruo>13.80</ruo><vol>500</vol></ro>'
             b'<ro><D0>2026-08-04T00:00:00+03:00</D0><ruo>13.75</ruo><vol>510</vol></ro>'
             b'</Ruonia></diffgr:diffgram></RuoniaResult></RuoniaResponse></soap:Body></soap:Envelope>')

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return reply

    monkeypatch.setattr(factors.urllib.request, "urlopen", lambda request, timeout=None: Reply())
    rate = factors.ruonia(cache_dir=tmp_path)

    assert rate.tolist() == [13.80, 13.75]
    assert str(rate.index[0].date()) == "2026-08-03"
    assert (tmp_path / "ruonia.xml").exists(), "cached for the next report"


def test_the_market_model_keeps_the_rate_out_of_alpha(pyfolio):
    from zipfolio.sections.factor_analysis import build_market_model_analysis

    days = pd.bdate_range("2023-01-02", "2024-12-31", tz="UTC")
    rng = np.random.default_rng(9)
    imoex = pd.Series(rng.normal(0.0004, 0.012, len(days)), index=days, name="IMOEX")
    rate = pd.Series(16.0, index=days, name="RUONIA")
    from zipfolio.factors import risk_free_returns

    rf = risk_free_returns(rate, days)
    strategy = rf + 0.8 * (imoex - rf) + rng.normal(0, 0.002, len(days))

    section = build_market_model_analysis(strategy, imoex, rate)

    import re

    table = section.side[0]
    assert "Market model: IMOEX over RUONIA" in table
    beta = float(re.search(r"<td>Mkt-RF</td><td>(-?[\d.]+)</td>", table).group(1))
    alpha = float(re.search(r"<td>Alpha \(annualized\)</td><td>(-?[\d.]+)%</td>", table).group(1))
    assert beta == pytest.approx(0.8, abs=0.03)
    assert abs(alpha) < 3, "a 16% rate left in the regression would show up here as alpha"


def test_no_benchmark_no_market_model(pyfolio):
    from zipfolio.sections.factor_analysis import build_market_model_analysis

    days = pd.bdate_range("2023-01-02", "2023-06-30", tz="UTC")
    section = build_market_model_analysis(pd.Series(0.001, index=days), None, None)
    assert "has none" in section.side[0]


def test_moex_sectors_come_from_the_sector_indices(monkeypatch, tmp_path):
    import json

    import zipfolio.sectors as sectors

    members = {"MOEXFN": ["SBER", "VTBR"], "MOEXOG": ["GAZP"]}

    class Reply:
        def __init__(self, url):
            self.index = next((name for name in sectors.MOEX_SECTOR_INDICES if f"/{name}.json" in url), None)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            rows = [["x", "d", ticker] for ticker in members.get(self.index, [])]
            return json.dumps({"analytics": {"columns": ["indexid", "tradedate", "ticker"], "data": rows}}).encode()

    monkeypatch.setattr(sectors.urllib.request, "urlopen", lambda url, timeout=None: Reply(url))
    found = sectors.moex_sector_mappings(["SBER@MISX", "GAZP@MISX", "AAPL@XNGS", "IMOEX@RTSX"], cache_dir=tmp_path)

    assert found == {"SBER": "Financials", "GAZP": "Oil & Gas"}
    assert (tmp_path / "moex_sectors.json").exists()


def test_a_rouble_report(pyfolio, tmp_path):
    closes = session_closes()
    returns, records, cash, fills = a_traded_book(closes)
    benchmark = pd.Series(np.random.default_rng(1).normal(0.0003, 0.009, len(closes)), index=closes, name="IMOEX")
    rate = pd.Series(16.0, index=daily_returns(returns).index, name="RUONIA")
    output = tmp_path / "zipfolio.html"

    built = build_tearsheet(output, returns, benchmark, records, cash, fills, orders=orders_for(fills),
                            market_data=daily_bars_for(closes, ["AAPL@XNAS", "MSFT@XNAS"]),
                            factor_model="market", risk_free=rate, currency="RUB")

    assert "Factor Analysis" in built and "Capacity" in built
    page = output.read_text(encoding="utf-8")
    assert "Market model: IMOEX over RUONIA" in page
    assert "\u20bd" in page and "Traded value" in page
    assert "100m \u20bd capital base" in page


# ----------------------------------------------------------------------------
# Drawdowns, market regimes, risk over time.
# ----------------------------------------------------------------------------

def test_a_drawdown_is_split_exactly_across_the_names_that_made_it():
    pytest.importorskip("pyfolio")
    from zipfolio.sections.drawdowns import RESIDUAL, drawdown_attribution

    closes = session_closes("2022-01-03", "2023-12-29")
    returns, records, cash, fills = a_traded_book(closes)
    dates = daily_returns(returns).index
    holdings = positions_in_dollars(records, cash, dates)
    book_returns = holdings.sum(axis="columns").pct_change().fillna(0.0)

    drawdowns, shares = drawdown_attribution(book_returns, holdings, transactions_frame(fills))

    assert len(drawdowns)
    assert shares.sum(axis="columns").to_numpy() == pytest.approx(drawdowns["Depth"].to_numpy())
    assert shares[RESIDUAL].abs().max() < 1e-9, "no dividends in this book: the names account for all of it"


def test_capture_and_betas_of_half_the_market():
    from zipfolio.sections.market_regimes import asymmetric_beta, capture, monthly_regimes

    days = pd.bdate_range("2021-01-01", "2023-12-29", tz="UTC")
    market = pd.Series(np.random.default_rng(12).normal(0.0004, 0.012, len(days)), index=days)
    half = market * 0.5

    monthly = monthly_regimes(half, market)

    assert set(monthly["regime"]) <= {"Rising, calm", "Rising, volatile", "Falling, calm", "Falling, volatile"}
    assert capture(monthly, True) == pytest.approx(0.5, abs=0.03)
    assert capture(monthly, False) == pytest.approx(0.5, abs=0.03)
    assert asymmetric_beta(half, market, True) == pytest.approx(0.5)
    assert asymmetric_beta(half, market, False) == pytest.approx(0.5)


def test_predicted_risk_is_the_weights_times_the_trailing_covariance():
    from zipfolio.sections.risk_over_time import COVARIANCE_WINDOW, risk_path

    days = pd.bdate_range("2023-01-02", periods=COVARIANCE_WINDOW + 5, tz="UTC")
    rng = np.random.default_rng(4)
    assets = pd.DataFrame(rng.normal(0, 0.01, (len(days), 2)), index=days, columns=["A", "B"])
    holdings = pd.DataFrame({"A": 600.0, "B": 300.0, "cash": 100.0}, index=days)

    path = risk_path(holdings, assets)

    first = path.index[0]
    w = np.array([0.6, 0.3])
    history = assets.iloc[:COVARIANCE_WINDOW].to_numpy()
    expected = np.sqrt(w @ np.cov(history, rowvar=False) @ w * 252)
    assert path.loc[first, "predicted"] == pytest.approx(expected)
    assert path[["A", "B"]].sum(axis="columns").to_numpy() == pytest.approx(1.0)


def test_effective_number_of_names():
    from zipfolio.sections.risk_over_time import concentration

    days = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    holdings = pd.DataFrame({"A": [100.0, 90.0, 0.0], "B": [100.0, 5.0, 0.0], "C": [100.0, 5.0, 0.0],
                             "cash": [0.0, 0.0, 100.0]}, index=days)

    crowding = concentration(holdings)

    assert crowding["effective"].iloc[0] == pytest.approx(3.0)
    assert crowding["effective"].iloc[1] == pytest.approx(1 / (0.9 ** 2 + 0.05 ** 2 * 2))
    assert crowding.iloc[2].isna().all(), "an all-cash day has no concentration to speak of"



# ----------------------------------------------------------------------------
# The page's structure: four parts, contents, and a summary of the findings.
# ----------------------------------------------------------------------------

def test_the_page_is_four_parts_with_contents_and_a_summary(pyfolio, tmp_path):
    import re

    closes = session_closes()
    returns, records, cash, fills = a_traded_book(closes)
    benchmark = pd.Series(np.random.default_rng(1).normal(0.0003, 0.009, len(closes)), index=closes)
    output = tmp_path / "zipfolio.html"

    build_tearsheet(output, returns, benchmark, records, cash, fills, orders=orders_for(fills))

    page = output.read_text(encoding="utf-8")
    parts = re.findall(r'<span class="part-name">([^<]+)</span>', page)
    assert parts == ["Where the Return Came From", "What It Risked", "How It Holds and Trades",
                     "Can It Be Traded"]
    assert page.index('id="summary"') < page.index('id="part-1"'), "the summary comes before the parts"
    ids = set(re.findall(r'id="([^"]+)"', page))
    links = set(re.findall(r'href="#([^"]+)"', page))
    assert links <= ids, f"every link lands somewhere: {links - ids}"
    summary = page.split('id="summary"')[1].split("</section>")[0]
    assert "Checks Passed" in summary
    rows = re.findall(r'<span class="badge (\w+)">', summary)
    assert len(rows) >= 5, "the sections say what they found"
    passed = re.search(r'<div class="n">(\d+)/(\d+)</div>', summary)
    assert (int(passed.group(1)), int(passed.group(2))) == (rows.count("ok"), len(rows)), "the score counts the rows"
    assert 'href="#trading-costs"' in summary and 'href="#closed-trades"' in summary
    assert "For context:" in summary, "facts that are not verdicts go on one line"
    trades = page.split('id="closed-trades"')[1].split("</section>")[0]
    assert "What Counts as a Trade" in trades and "two trades" in trades, "a round trip is explained, with examples"
    assert "round_trips" not in trades and "Round Trip" not in trades, "pyfolio's jargon is not shown"


def test_a_failed_section_says_nothing_in_the_summary(pyfolio):
    from zipfolio.page import BAD, Section, ordered, summary_html

    broken = Section("Execution", failed=True)
    broken.finding(BAD, "should not show", "-")
    fine = Section("Trading Costs")
    fine.finding(BAD, "Costs ate it", "90%")
    html = summary_html(ordered([broken, fine]))
    assert "Costs ate it" in html and "should not show" not in html
    assert "Checks Passed" in html and ">0/1<" in html


# ----------------------------------------------------------------------------
# Next-bar fills: the same trades, a bar later.
# ----------------------------------------------------------------------------

def test_a_fill_repriced_at_the_next_open_keeps_its_slippage():
    from zipfolio.sections.next_bar_fills import repriced_fills

    fills = pd.DataFrame({
        "id": ["b", "s"], "dt": pd.to_datetime(["2024-03-01 21:00"] * 2, utc=True), "symbol": ["A", "B"],
        "quantity": [10, -10], "price": [100.1, 49.95],
    })
    bars = pd.DataFrame({"close": [100.0, 50.0], "next_open": [102.0, 49.0],
                         "next_date": pd.to_datetime(["2024-03-04 14:30"] * 2, utc=True)}, index=["b", "s"])

    priced = repriced_fills(fills, bars).set_index("id")

    assert priced.loc["b", "impact"] == pytest.approx(-10 * 100.1 * 0.02), "the buy pays 2% more, slippage included"
    assert priced.loc["b", "move_bps"] == pytest.approx(200.0)
    assert priced.loc["s", "impact"] == pytest.approx(-10 * 49.95 * 0.02), "the sell gets 2% less"
    assert priced.loc["s", "move_bps"] == pytest.approx(200.0), "worse for the side is positive for both"


def next_bar_inputs(gap):
    """A position bought at each close and sold at the next; every next open is ``gap`` above the close."""
    closes = session_closes("2023-01-03", "2023-03-31")
    days = pd.DatetimeIndex(closes).tz_localize("UTC")
    prices = 100.0 * np.cumprod(np.full(len(days), 1.001))
    fills, bars, records, cash = [], {}, [], []
    for i, day in enumerate(days[:-1]):
        fills.append({"id": f"f{i}", "dt": day, "symbol": "A", "quantity": 10, "price": prices[i]})
        bars[f"f{i}"] = {"close": prices[i], "next_open": prices[i] * (1 + gap), "next_date": days[i + 1]}
        records.append({"date": day.strftime("%Y-%m-%d"), "symbol": "A", "quantity": 10.0 * (i + 1),
                        "last_sale_price": prices[i]})
        cash.append(100_000.0 - 10 * prices[: i + 1].sum())
    records.append({"date": days[-1].strftime("%Y-%m-%d"), "symbol": "A", "quantity": 10.0 * (len(days) - 1),
                    "last_sale_price": prices[-1]})
    cash.append(cash[-1])
    holdings = positions_in_dollars(pd.DataFrame(records), pd.Series(cash, index=closes),
                                    daily_returns(pd.Series(0.0, index=closes)).index)
    returns = holdings.sum(axis="columns").pct_change().fillna(0.0)
    return returns, holdings, pd.DataFrame(fills), pd.DataFrame(bars).T


def test_a_bar_that_wipes_out_the_return_is_flagged(pyfolio):
    from zipfolio.page import BAD
    from zipfolio.sections.next_bar_fills import build_next_bar_fills

    returns, holdings, fills, bars = next_bar_inputs(gap=0.05)
    section = build_next_bar_fills(returns, holdings, fills, bars, {"same_bar_execution": True})

    assert section is not None and section.findings[0].status == BAD
    assert "wipes out" in section.findings[0].note
    assert "The Same Trades, a Bar Later" in "".join(section.side + section.wide)


def test_no_next_bar_section_when_orders_already_wait(pyfolio):
    from zipfolio.sections.next_bar_fills import build_next_bar_fills

    returns, holdings, fills, bars = next_bar_inputs(gap=0.0)
    assert build_next_bar_fills(returns, holdings, fills, bars, {"same_bar_execution": False}) is None
    assert build_next_bar_fills(returns, holdings, fills, bars.drop(columns="next_open"), {}) is None


def test_the_next_bar_comes_with_the_fill_bar():
    pl = pytest.importorskip("polars")

    from zipfolio.ziplime import execution_bars

    fills = pd.DataFrame({"id": ["t1", "t2"], "symbol": ["AAPL", "AAPL"],
                          "dt": [pd.Timestamp("2024-03-04 21:00", tz="UTC"),
                                 pd.Timestamp("2024-03-05 21:00", tz="UTC")]})
    bars = pl.DataFrame({
        "symbol": ["AAPL", "MSFT", "AAPL", "AAPL"],
        "date": pd.to_datetime(["2024-03-01 05:00", "2024-03-04 05:00", "2024-03-04 05:00", "2024-03-05 05:00"],
                               utc=True),
        "open": [99.0, 400.0, 100.0, 103.0], "high": [100.0, 401.0, 102.0, 104.0],
        "low": [98.0, 399.0, 99.5, 102.0], "close": [99.5, 400.5, 101.5, 103.5], "volume": [1e6] * 4,
    })

    found = execution_bars(types.SimpleNamespace(data=bars), fills)["fill_bars"]

    assert found.loc["t1", "next_open"] == 103.0, "the same instrument's next bar, not another's"
    assert found.loc["t1", "next_date"] == pd.Timestamp("2024-03-05 05:00", tz="UTC")
    assert pd.isna(found.loc["t2", "next_open"]), "the last bar has no next"


# ----------------------------------------------------------------------------
# Profit concentration.
# ----------------------------------------------------------------------------

def test_the_best_days_are_spent_in_cash():
    from zipfolio.sections.profit_concentration import without_best_days

    returns = pd.Series([0.01, 0.05, -0.02, 0.03, 0.0])
    assert without_best_days(returns, 2).tolist() == [0.01, 0.0, -0.02, 0.0, 0.0]


def test_the_bootstrap_is_repeatable_and_centred_on_the_total():
    from zipfolio.sections.profit_concentration import bootstrap_totals

    pnl = np.array([100.0, -40.0, 25.0, -10.0, 60.0])
    first, second = bootstrap_totals(pnl, samples=4_000), bootstrap_totals(pnl, samples=4_000)
    assert np.array_equal(first, second), "the same backtest gets the same page"
    assert first.mean() == pytest.approx(pnl.sum(), rel=0.05)
    assert len(bootstrap_totals(pnl, samples=7)) == 7


def test_a_return_made_in_five_days_is_flagged(pyfolio):
    from zipfolio.page import BAD
    from zipfolio.sections.profit_concentration import build_profit_concentration

    closes = session_closes()
    returns = pd.Series(-0.0005, index=daily_returns(pd.Series(0.0, index=closes)).index)
    returns.iloc[[20, 60, 100, 140, 180]] = 0.05
    section = build_profit_concentration(returns, None, None)

    assert section.findings[0].status == BAD
    assert section.findings[0].label == "Total return without best 5 days"
    assert section.findings[0].note.startswith("5 sessions made all of the return")


# ----------------------------------------------------------------------------
# Long and short.
# ----------------------------------------------------------------------------

def test_long_and_short_legs_add_up_to_the_book(pyfolio):
    from zipfolio.sections.instruments import instrument_pnl
    from zipfolio.sections.long_short import build_long_short

    closes = session_closes()
    returns, records, cash, fills = a_traded_book(closes)
    dates = daily_returns(returns).index
    holdings = positions_in_dollars(records, cash, dates)
    txns = transactions_frame(fills)

    section = build_long_short(daily_returns(returns), holdings, txns)

    assert section is not None, "the book shorts MSFT and AAPL at times"
    table = section.side[0] + "".join(section.wide)
    assert "Each Side of the Book" in table
    assert section.findings and section.findings[0].section == "Long and Short"
    total = instrument_pnl(holdings, txns).to_numpy().sum()
    from zipfolio.sections.long_short import side_of_day

    pnl = instrument_pnl(holdings, txns)
    side = side_of_day(holdings, txns, pnl.columns)
    assert pnl.where(side > 0, 0).to_numpy().sum() + pnl.where(side < 0, 0).to_numpy().sum() \
        + pnl.where(side == 0, 0).to_numpy().sum() == pytest.approx(total)


def test_a_day_belongs_to_the_side_held_at_the_previous_close():
    from zipfolio.sections.long_short import side_of_day

    days = pd.date_range("2024-03-01", periods=4, tz="UTC")
    holdings = pd.DataFrame({"A": [100.0, -50.0, 0.0, 0.0], "cash": 1000.0}, index=days)
    txns = pd.DataFrame({"amount": [1, -3, 1, 2, -2], "price": 50.0, "symbol": "A"},
                        index=pd.DatetimeIndex([days[0], days[1], days[2], days[3], days[3]]))

    side = side_of_day(holdings, txns, ["A"])["A"].tolist()
    assert side == [1, 1, -1, 1], "opened long; flipped short on a long day; closed a short; in and out long"


def test_a_long_only_book_has_no_long_short_section(pyfolio):
    from zipfolio.sections.long_short import build_long_short

    days = pd.date_range("2024-03-01", periods=5, tz="UTC")
    holdings = pd.DataFrame({"A": [100.0, 110.0, 120.0, 0.0, 0.0], "cash": 1000.0}, index=days)
    txns = pd.DataFrame({"amount": [1, -1], "price": 100.0, "symbol": "A"},
                        index=pd.DatetimeIndex([days[0], days[3]]))
    assert build_long_short(pd.Series(0.0, index=days), holdings, txns) is None


def test_a_trade_is_each_sale_with_the_shares_it_closes_and_its_return_is_on_their_cost(pyfolio):
    """Buy 100 at 10, sell 30 at 12, sell 70 at 9: two trades, as the page explains."""
    from zipfolio.sections.round_trips import extract_round_trips

    days = pd.date_range("2024-03-01", periods=4, tz="UTC")
    txns = pd.DataFrame({"amount": [100, -30, -70], "price": [10.0, 12.0, 9.0], "symbol": "A"},
                        index=pd.DatetimeIndex([days[0], days[1], days[3]]))
    positions = pd.DataFrame({"A": [1000.0, 840.0, 700.0, 0.0], "cash": [9000.0, 9360.0, 9360.0, 9990.0]},
                             index=days)
    returns = positions.sum(axis="columns").pct_change().fillna(0.0)

    trades = extract_round_trips(returns, positions, txns).sort_values("close_dt")

    assert trades["pnl"].tolist() == pytest.approx([60.0, -70.0])
    assert trades["rt_returns"].tolist() == pytest.approx([0.2, -0.1]), "PnL over what the shares cost"


def test_every_finding_branch_has_its_status_names(pyfolio):
    """Both sides of a long-short book winning, and a spread-out instrument book: the OK branches."""
    from zipfolio.page import OK
    from zipfolio.sections.instruments import _concentration_finding
    from zipfolio.sections.long_short import build_long_short
    from zipfolio.page import Section

    days = pd.date_range("2024-03-01", periods=5, tz="UTC")
    holdings = pd.DataFrame({"A": [100.0, 110.0, 120.0, 130.0, 0.0], "B": [-100.0, -90.0, -80.0, -70.0, 0.0],
                             "cash": 1000.0}, index=days)
    txns = pd.DataFrame({"amount": [1, -1, -1, 1], "price": [100.0, 100.0, 130.0, 70.0], "symbol": ["A", "B", "A", "B"]},
                        index=pd.DatetimeIndex([days[0], days[0], days[4], days[4]]))
    section = build_long_short(pd.Series(0.0, index=days), holdings, txns)
    assert section.findings[0].status == OK and section.findings[0].note == "Both sides made money."

    section = Section("Instruments")
    summary = pd.DataFrame({"Net PnL": [40.0, 35.0, 30.0, 25.0], "Days Held": [5, 5, 5, 5]},
                           index=["A", "B", "C", "D"])
    _concentration_finding(section, summary)
    assert section.findings[0].status == OK
