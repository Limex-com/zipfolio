"""What the page's terms mean, a sentence each, shown when a reader points at one.

The page keeps its text short and its tables dense; a reader who does not know
what CDaR or a Treynor ratio is can hover over the name -- or tap it on a
phone -- instead of reading a paragraph that every other reader skips. A table
row or column whose label matches an entry gets the hint (``tables.table_html``).
"""

from __future__ import annotations

import re

#: ``(pattern, hint)``: the first pattern that matches the start of a label, ignoring case, gives its
#: hint. More specific patterns come before the general ones they would otherwise lose to.
TERMS: list[tuple[str, str]] = [
    # Profit, as accounting.profit_breakdown defines it
    (r"realized:", "What the closed trades made: each sale with the shares it closed, first in, first out; before "
                   "commission, slippage inside (it is in the fill prices)."),
    (r"unrealized:", "The shares still held at the end, at the last price, over what they cost, first in, first "
                     "out; before commission. Not a trade: nothing was sold."),
    (r"net profit: change", "Realized plus unrealized, less commission, plus any other cash: the change in cash plus "
                            "positions over the backtest."),
    (r"net profit from the daily returns", "The start value compounded by the daily returns, less the start value: "
                                           "the same profit from the engine's other record."),
    (r"net pnl", "After commission: realized on closed trades plus unrealized on what is still held, at the last "
                 "price; slippage is inside it, in the fill prices. Summed over the names it is the change in the "
                 "portfolio's value."),
    (r"contribution", "Daily PnL after commission over the portfolio's value at the start of each day, added up. "
                      "Summed over the names it is the sum of the daily returns, not the compounded total return, so "
                      "a name can show a dollar loss and a positive contribution, or the reverse."),
    (r"(their pnl|total pnl), before commission", "The closed trades' PnL added up: realized, before commission. "
                                                  "Positions still held are not in it."),
    # Return
    (r"total return", "Growth of the account from the first session to the last."),
    (r"annual return \(compounded\)|annual return$",
     "CAGR: the constant yearly return that compounds to the same total return."),
    (r"annual (mean )?return \((mean|arithmetic)\)|annual mean return",
     "The mean daily return times 252. Mean-variance theory uses it; it runs above the compounded return "
     "by about half the variance."),
    (r"annual(ized)? volatility", "The standard deviation of daily returns times √252."),
    # Ratios
    (r"sharpe ratio at 0%", "Return per unit of volatility, a year, at a 0% risk-free rate, as QuantStats "
                            "computes it."),
    (r"sharpe( ratio)?\b", "Return over the risk-free rate per unit of volatility, a year. A broad stock index "
                           "has averaged about 0.4-0.5 over the long run."),
    (r"sortino", "Like the Sharpe ratio, but divided by the downside volatility only: falls count, rises do not."),
    (r"calmar", "The annual return divided by the maximum drawdown."),
    (r"treynor", "Return over the risk-free rate per unit of beta. The benchmark's is its own premium over "
                 "the rate."),
    (r"m²|m2\b", "Modigliani's measure: the strategy's return scaled to the benchmark's volatility, at its own "
                 "Sharpe ratio. It compares with the benchmark's return directly."),
    (r"information ratio \(alpha", "Alpha per unit of residual volatility: return the factors do not explain, "
                                   "per unit of risk they do not explain."),
    (r"information ratio", "Return over the benchmark per unit of tracking error. Above 0.5 is good; above 1 "
                           "is rare."),
    (r"tracking error", "The volatility of the strategy's return minus the benchmark's, a year."),
    (r"profit factor", "Gross profit over gross loss. Above 1 makes money; 1.5 and up is solid."),
    (r"percent profitable|win rate|winning trades", "The share of closed trades that made money."),
    (r"ratio avg\. win", "The average winning trade over the average losing trade."),
    # Market
    (r"beta on up days", "Beta measured only on the days the benchmark rose."),
    (r"beta on down days", "Beta measured only on the days the benchmark fell. Lower than on up days is the "
                           "asymmetry worth having."),
    (r"beta\b", "How much the strategy moves when the benchmark moves 1%: 1.0 moves with it, 0 not at all, "
                "a negative beta against it."),
    (r"correlation", "How closely daily returns move together, from -1 to 1."),
    (r"variance the market explains|r-squared|r²",
     "The share of the daily variance the market (or the factors) explains. The rest is the strategy's own."),
    (r"adjusted r-squared", "R-squared penalised for the number of factors."),
    (r"systematic volatility", "The part of volatility that comes from the market: beta times the benchmark's "
                               "volatility."),
    (r"specific volatility|residual volatility",
     "Volatility the market (or the factors) does not explain. CAPM says it earns nothing: a portfolio "
     "could have diversified it away."),
    (r"return capm asks", "The risk-free rate plus beta times the benchmark's premium over it: what the market "
                          "pays for this much beta."),
    (r"jensen", "The return beyond what CAPM asks for the strategy's beta."),
    (r"alpha to .*capm", "The return beta to the benchmark does not explain, a year, at a 0% risk-free rate, as "
                         "QuantStats computes it."),
    (r"alpha\b", "The return the factors do not explain, a year. It counts only with a t-stat beyond about 2."),
    (r"up capture", "The strategy's average month in the months the benchmark rose, as a share of the "
                    "benchmark's. Above 100% gains more."),
    (r"down capture", "The same for the months the benchmark fell. Below 100% falls less."),
    (r"capture ratio", "Up capture over down capture. Above 1, the strategy gains more with the market than it "
                       "loses with it."),
    (r"(rising|falling), (calm|volatile)", "Months grouped by the benchmark's direction, and by whether its daily "
                                           "volatility was above the median month's."),
    (r"mkt-rf", "The market's return over the risk-free rate."),
    (r"smb\b", "Size: small companies minus big ones. A positive beta leans to small caps."),
    (r"hml\b", "Value: cheap companies (high book-to-market) minus expensive ones. Positive leans to value, "
               "negative to growth."),
    (r"rmw\b", "Profitability: companies with robust operating profitability minus weak ones."),
    (r"cma\b", "Investment: companies that invest conservatively minus those that invest aggressively."),
    (r"mom\b", "Momentum: the past year's winners minus its losers."),
    (r"t-stat", "The estimate over its standard error. Beyond about ±2 it is unlikely to be zero."),
    (r"p-value", "The chance of an estimate at least this far from zero if the true value were zero. Below 0.05 "
                 "is significant."),
    # Tail and drawdown risk
    (r"var 95%", "The daily loss exceeded on one day in twenty, read from the backtest's own days."),
    (r"cvar 95%|cvar share", "The average loss on the worst 5% of days (expected shortfall)."),
    (r"evar", "Entropic VaR: an upper bound on VaR and CVaR from the whole distribution. It reacts more to fat "
              "tails."),
    (r"max(imum)? drawdown", "The deepest fall from a peak to a later low."),
    (r"average drawdown", "The average depth below the last peak, over every session."),
    (r"drawdown at risk", "The drawdown depth exceeded on the 5% of sessions furthest below a peak."),
    (r"cdar", "Conditional drawdown at risk: the average depth over the 5% of sessions furthest below a peak."),
    (r"edar", "Entropic drawdown at risk: an upper bound on CDaR."),
    (r"ulcer index", "The root mean square of all drawdowns. It grows with their depth and with their length."),
    (r"gini mean difference", "The average gap between any two daily returns: a spread that a few extreme days "
                              "do not dominate."),
    (r"mean absolute deviation", "The average distance of a daily return from the mean."),
    (r"annualized semi-deviation", "The volatility of returns below the mean only, a year."),
    (r"worst day", "The largest one-day loss."),
    (r"skew", "The asymmetry of daily returns. Negative: frequent small gains and rare large losses."),
    (r"excess kurtosis", "Fat tails. Above 0, extreme days come more often than a normal distribution allows."),
    # Significance and luck
    (r"chance the true sharpe", "The probabilistic Sharpe ratio: allowing for the length of the backtest, skew "
                                "and fat tails, the chance the true Sharpe ratio is above zero."),
    (r"best of \d+ tries", "The Sharpe ratio the best of that many strategies with no edge would show by chance "
                           "alone."),
    (r"chance it beats that", "The deflated Sharpe ratio: the chance this Sharpe ratio beats the best of that "
                              "many strategies with no edge."),
    (r"luck test", "The closed trades drawn again at random, with replacement, 10,000 times: how the total "
                   "could have come out with trades like these."),
    # The book
    (r"time in the market", "The share of sessions with a position open."),
    (r"average exposure \(gross\)", "Longs plus shorts over the portfolio's value, on average. 1.00 is fully "
                                    "invested without leverage."),
    (r"(average|lowest) effective names", "One over the sum of squared weights: how many equal positions the "
                                          "book's concentration is worth."),
    (r"one-way turnover|turnover, a year", "Value traded over the portfolio's value, a year: 100% replaces the "
                                           "book once."),
    (r"bias \(realized", "Realized volatility over predicted. Above 1, the trailing covariance underestimated "
                         "the risk."),
    (r"volatility share", "The holding's share of the portfolio's volatility at the average weights."),
    # Allocations
    (r"equal weight", "Every name the same share, rebalanced back to it."),
    (r"inverse volatility", "Weights in proportion to one over each name's volatility."),
    (r"hierarchical risk parity", "Clusters the names by correlation and splits the risk between the clusters "
                                  "(López de Prado)."),
    (r"max sharpe \(mean-variance\)", "Markowitz's mix with the best Sharpe ratio, fitted on the previous "
                                      "year's returns."),
    (r"max sharpe \(hindsight\)", "The mix of these names with the best Sharpe ratio, measured with the returns "
                                  "they went on to make."),
    (r"minimum variance", "The mix of these names with the lowest volatility."),
    (r"strategy's average weights", "The strategy's average weight in each name over the backtest, held "
                                    "throughout."),
    # Execution and costs
    (r"implementation shortfall", "The fill price against the price when the order was decided, weighted by "
                                  "value: what getting into the trade cost."),
    (r"slippage \(reconstructed\)|slippage$", "The gap between the fill price and the bar's price: the "
                                              "backtest's model of market impact."),
    (r"next bar's open|final value, next bar's open",
     "Every fill priced again at the instrument's next open: what trading after seeing the bar would have got."),
    (r"median share of bar volume|(largest|median) fill, share of its day's volume",
     "The fill's size against the volume traded in its bar or day. Above a few percent, the price would move."),
]

_COMPILED = [(re.compile(pattern, re.I), hint) for pattern, hint in TERMS]


def hint(label: str) -> str | None:
    """The hint for a table's row or column label, or ``None``."""
    text = str(label).strip()
    for pattern, explanation in _COMPILED:
        if pattern.match(text):
            return explanation
    return None


#: What each table is for, a sentence, shown on its title: ``(pattern, hint)`` as for ``TERMS``.
TABLES: list[tuple[str, str]] = [
    (r"key performance metrics", "The headline numbers side by side: the strategy, holding its own names from day "
                                 "one, and the benchmark."),
    (r"is the sharpe ratio more than luck", "Whether the Sharpe ratio could be chance: its 95% interval and the "
                                            "probability that the true one is above zero."),
    (r"against what noise reaches", "How high a Sharpe ratio strategies with no edge reach when many versions are "
                                    "tried, and the chance this one beats that."),
    (r"first half and second half", "The backtest cut in two: a rule that worked in only one half is fragile."),
    (r"without the best days", "The return with the best sessions taken out: how much of it rests on a few days."),
    (r"the best trades and the luck test", "How much of the profit the best trades made, and the closed trades "
                                           "redrawn at random to see how often they lose."),
    (r"returns by instrument", "What each name made, how much of the book it took and how its trades went."),
    (r"sector allocation", "Each sector's share of the book: on average and at its peak."),
    (r"profit by sector", "What each sector made: net PnL after commission and its contribution to the return."),
    (r"factor exposures", "The regression of the strategy's excess return on the factors: each beta, the alpha, and "
                          "whether they could be zero."),
    (r"model fit", "How much of the strategy the factor model explains, and the risk it leaves unexplained."),
    (r"return attribution", "The average annual return split into what each factor paid and the alpha."),
    (r"drawdown #\d+, by name", "Who made this drawdown: each name's points of the fall, losses and offsets."),
    (r"worst drawdowns and who made them", "The five deepest falls from a peak: when, how deep, how long, and the "
                                           "names that made them."),
    (r"worst drawdowns", "The five deepest falls from a peak, with their dates, depth and length."),
    (r"against ", "How the strategy moved with the benchmark: capture in up and down months, beta on up and down "
                  "days."),
    (r"average month, by regime", "The average month in rising and falling, calm and volatile markets, against the "
                                  "benchmark."),
    (r"risk measures", "Tail and drawdown risk, measured several ways, for the strategy and the benchmark."),
    (r"risk contribution", "Each holding's share of the portfolio's volatility and of its tail loss, against its "
                           "weight."),
    (r"same names, other allocations", "The strategy's names under four standard allocation rules, refitted "
                                       "walk-forward: does its sizing beat them?"),
    (r"capm measures", "The strategy against the capital asset pricing model: its beta, the return that beta asks "
                       "for, and what it made beyond it."),
    (r"fixed mixes of the same names", "Return, volatility and Sharpe of fixed mixes of the strategy's names, "
                                       "beside the strategy itself."),
    (r"the max sharpe mix", "The weights of the best fixed mix in hindsight: what the frontier would have held."),
    (r"forecast against outcome", "The volatility predicted from the positions each day against what the next "
                                  "month delivered."),
    (r"concentration", "How many names the book really held: names held, effective names, top-3 share, largest "
                       "weight."),
    (r"top 10 positions", "The largest weight each name ever reached."),
    (r"trading activity", "How often the strategy traded: fills, names, trading days and turnover."),
    (r"how many trades", "Closed trades counted: winners, losers and the share that made money."),
    (r"profit and loss per trade", "Profit and loss per closed trade: totals, profit factor, average and extremes."),
    (r"how long trades were held", "How long closed trades lasted, from the opening fill to the closing one."),
    (r"return per trade, by symbol", "The return of closed trades for each name: average and median, winners and "
                                     "losers."),
    (r"return per trade", "The return of closed trades: average and median, winners and losers."),
    (r"each side of the book", "What the long side and the short side each made."),
    (r"the same trades, a bar later", "The backtest replayed with every fill at the next bar's open instead of the "
                                      "signal bar's close."),
    (r"what the wait did", "How prices moved between the signal bar's close and the next open, for buys and sells."),
    (r"fill statistics", "Orders and fills: how many, how big, and how long orders waited."),
    (r"orders by type", "Orders by type, and how many filled in full, in part, or not at all."),
    (r"orders not filled in full", "The orders that were cut short, and why."),
    (r"cost settings", "The commission and slippage the backtest charged."),
    (r"cost summary", "What trading cost in total: commission, slippage, and their share of the traded value."),
    (r"costs by symbol", "Trading costs by name."),
    (r"how much capital it could take", "How large the strategy could grow before its fills take too much of the "
                                        "names' daily volume."),
]

#: What each section answers, shown on its heading.
SECTIONS: dict[str, str] = {
    "Performance": "The headline numbers, against holding the same names and against the benchmark.",
    "Profit Concentration": "Whether the result rests on a few days or a few trades.",
    "Instruments": "What each name made and how its trades went.",
    "Long and Short": "What each side of the book made.",
    "Sector Analysis": "Where the capital went by sector, and what each sector made.",
    "Factor Analysis": "How much of the return is known factors -- the market, size, value, momentum -- and what is "
                       "left (alpha).",
    "Drawdowns": "The worst falls from a peak, and the names that made them.",
    "Market Regimes": "How the strategy behaves when the market rises, falls, is calm or volatile.",
    "Portfolio Risk": "Tail and drawdown risk, risk by holding, and the same names under other allocations.",
    "CAPM and the Efficient Frontier": "Where the strategy sits on the classic maps of risk against return: the "
                                       "market lines of CAPM and the frontier of its own names. A map, not a check.",
    "Risk Over Time": "The risk the positions predicted each day against what came, and how concentrated the book "
                      "was.",
    "Positions": "What the book held, and how much of it.",
    "Transactions": "How much and when the strategy traded.",
    "Closed Trades": "Every round trip from opening to closing, counted and measured.",
    "Next-Bar Fills": "What is left of the return when every fill waits for the next bar's open.",
    "Execution": "Whether each fill could have happened at its price, and how orders were filled.",
    "Trading Costs": "What commission and slippage took.",
    "Capacity": "How much capital the strategy could run before its trades move prices.",
}

_TITLES = [(re.compile(pattern, re.I), hint) for pattern, hint in TABLES]


def title_hint(title: str) -> str | None:
    """What a table is for, from its title, or ``None``."""
    text = str(title).strip()
    for pattern, explanation in _TITLES:
        if pattern.match(text):
            return explanation
    return None
