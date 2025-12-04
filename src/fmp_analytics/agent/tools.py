"""Financial analysis tools for Agno agent.

This module provides tool functions that can be used by the Agno agent
to perform financial analysis tasks.
"""

import asyncio
from datetime import date, datetime, timedelta
from typing import Any

from agno.tools import tool

from fmp_analytics.api.client import FMPClient
from fmp_analytics.pipeline.data_pipeline import DataPipeline
from fmp_analytics.pipeline.analysis_pipeline import AnalysisPipeline


def _get_pipeline() -> tuple[DataPipeline, AnalysisPipeline]:
    """Get data and analysis pipelines."""
    client = FMPClient()
    data_pipeline = DataPipeline(client)
    analysis_pipeline = AnalysisPipeline(data_pipeline)
    return data_pipeline, analysis_pipeline


def _run_async(coro: Any) -> Any:
    """Run async function synchronously."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@tool
def get_quote_tool(symbols: str) -> str:
    """Get current stock quotes for one or more symbols.

    Use this tool to get real-time price information including current price,
    change, volume, market cap, P/E ratio, and 52-week range.

    Args:
        symbols: Comma-separated stock symbols (e.g., "AAPL" or "AAPL,MSFT,GOOGL")

    Returns:
        String with quote information for each symbol.
    """
    data_pipeline, _ = _get_pipeline()

    async def fetch():
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        quotes = await data_pipeline.get_quotes_batch(symbol_list)
        await data_pipeline._client.close()
        return quotes

    quotes = _run_async(fetch())

    results = []
    for q in quotes:
        results.append(f"""
**{q.symbol}** ({q.name})
- Price: ${q.price:.2f}
- Change: {q.change:+.2f} ({q.changes_percentage:+.2f}%)
- Day Range: ${q.day_low:.2f} - ${q.day_high:.2f}
- 52W Range: ${q.year_low:.2f} - ${q.year_high:.2f}
- Volume: {q.volume:,}
- Market Cap: ${q.market_cap:,.0f}
- P/E Ratio: {q.pe:.2f if q.pe else 'N/A'}
""")

    return "\n".join(results)


@tool
def get_company_profile_tool(symbol: str) -> str:
    """Get detailed company profile information.

    Use this tool to get company details including description, sector, industry,
    CEO, number of employees, headquarters location, and website.

    Args:
        symbol: Stock symbol (e.g., "AAPL")

    Returns:
        String with company profile information.
    """
    data_pipeline, _ = _get_pipeline()

    async def fetch():
        profile = await data_pipeline.get_company_profile(symbol.upper())
        await data_pipeline._client.close()
        return profile

    p = _run_async(fetch())

    return f"""
**{p.company_name}** ({p.symbol})

**Overview:**
- Sector: {p.sector}
- Industry: {p.industry}
- CEO: {p.ceo}
- Employees: {p.full_time_employees:,}
- Website: {p.website}

**Location:**
- {p.address}, {p.city}, {p.state} {p.zip}
- Country: {p.country}

**Market Data:**
- Price: ${p.price:.2f}
- Market Cap: ${p.mkt_cap:,.0f}
- Beta: {p.beta:.2f}
- Dividend: ${p.last_div:.2f}
- DCF Value: ${p.dcf:.2f}

**Description:**
{p.description[:500]}...
"""


@tool
def get_historical_prices_tool(
    symbol: str,
    days: int = 30,
) -> str:
    """Get historical stock prices.

    Use this tool to get historical OHLCV (Open, High, Low, Close, Volume) data
    for technical analysis or performance tracking.

    Args:
        symbol: Stock symbol (e.g., "AAPL")
        days: Number of days of history (default: 30)

    Returns:
        String with historical price summary and recent prices.
    """
    data_pipeline, _ = _get_pipeline()

    async def fetch():
        to_date = date.today()
        from_date = to_date - timedelta(days=days)
        df = await data_pipeline.get_historical_prices(
            symbol.upper(), str(from_date), str(to_date)
        )
        await data_pipeline._client.close()
        return df

    df = _run_async(fetch())

    if df.empty:
        return f"No historical data found for {symbol}"

    # Calculate summary statistics
    latest = df.iloc[-1]
    earliest = df.iloc[0]
    period_return = (latest["adjClose"] / earliest["adjClose"] - 1) * 100
    high = df["high"].max()
    low = df["low"].min()
    avg_volume = df["volume"].mean()

    result = f"""
**{symbol.upper()} - {days} Day Historical Summary**

**Period Performance:**
- Start Price: ${earliest['adjClose']:.2f} ({earliest['date'].strftime('%Y-%m-%d')})
- End Price: ${latest['adjClose']:.2f} ({latest['date'].strftime('%Y-%m-%d')})
- Return: {period_return:+.2f}%

**Range:**
- High: ${high:.2f}
- Low: ${low:.2f}
- Avg Volume: {avg_volume:,.0f}

**Last 5 Days:**
"""

    for _, row in df.tail(5).iterrows():
        result += f"- {row['date'].strftime('%Y-%m-%d')}: O=${row['open']:.2f} H=${row['high']:.2f} L=${row['low']:.2f} C=${row['close']:.2f}\n"

    return result


@tool
def get_financial_ratios_tool(
    symbol: str,
    period: str = "annual",
) -> str:
    """Get financial ratios for a company.

    Use this tool to analyze profitability, liquidity, leverage, and efficiency
    ratios for fundamental analysis.

    Args:
        symbol: Stock symbol (e.g., "AAPL")
        period: "annual" or "quarter"

    Returns:
        String with key financial ratios.
    """
    data_pipeline, _ = _get_pipeline()

    async def fetch():
        df = await data_pipeline.get_financial_ratios(symbol.upper(), period, limit=3)
        await data_pipeline._client.close()
        return df

    df = _run_async(fetch())

    if df.empty:
        return f"No financial ratios found for {symbol}"

    latest = df.iloc[0]

    return f"""
**{symbol.upper()} - Financial Ratios ({period.capitalize()})**

**Profitability:**
- Gross Profit Margin: {latest.get('grossProfitMargin', 0)*100:.2f}%
- Operating Margin: {latest.get('operatingProfitMargin', 0)*100:.2f}%
- Net Profit Margin: {latest.get('netProfitMargin', 0)*100:.2f}%
- ROE: {latest.get('returnOnEquity', 0)*100:.2f}%
- ROA: {latest.get('returnOnAssets', 0)*100:.2f}%

**Liquidity:**
- Current Ratio: {latest.get('currentRatio', 0):.2f}
- Quick Ratio: {latest.get('quickRatio', 0):.2f}
- Cash Ratio: {latest.get('cashRatio', 0):.2f}

**Leverage:**
- Debt/Equity: {latest.get('debtEquityRatio', 0):.2f}
- Debt Ratio: {latest.get('debtRatio', 0):.2f}
- Interest Coverage: {latest.get('interestCoverage', 0):.2f}

**Valuation:**
- P/E Ratio: {latest.get('priceEarningsRatio', 0):.2f}
- P/B Ratio: {latest.get('priceToBookRatio', 0):.2f}
- P/S Ratio: {latest.get('priceToSalesRatio', 0):.2f}
- EV/EBITDA: {latest.get('enterpriseValueMultiple', 0):.2f}

**Efficiency:**
- Asset Turnover: {latest.get('assetTurnover', 0):.2f}
- Inventory Turnover: {latest.get('inventoryTurnover', 0):.2f}
- Receivables Turnover: {latest.get('receivablesTurnover', 0):.2f}
"""


@tool
def get_market_movers_tool() -> str:
    """Get today's market movers (gainers, losers, most active).

    Use this tool to see which stocks are moving the most today.

    Returns:
        String with top gainers, losers, and most active stocks.
    """
    data_pipeline, _ = _get_pipeline()

    async def fetch():
        movers = await data_pipeline.get_market_movers()
        await data_pipeline._client.close()
        return movers

    movers = _run_async(fetch())

    result = "**Today's Market Movers**\n\n"

    result += "**Top Gainers:**\n"
    for _, row in movers["gainers"].head(5).iterrows():
        result += f"- {row['symbol']}: ${row['price']:.2f} ({row['changesPercentage']:+.2f}%)\n"

    result += "\n**Top Losers:**\n"
    for _, row in movers["losers"].head(5).iterrows():
        result += f"- {row['symbol']}: ${row['price']:.2f} ({row['changesPercentage']:+.2f}%)\n"

    result += "\n**Most Active:**\n"
    for _, row in movers["most_active"].head(5).iterrows():
        result += f"- {row['symbol']}: ${row['price']:.2f} ({row['changesPercentage']:+.2f}%)\n"

    return result


@tool
def analyze_stock_tool(symbol: str) -> str:
    """Perform comprehensive stock analysis.

    Use this tool for in-depth analysis of a single stock including valuation,
    fundamentals, technicals, and risk metrics.

    Args:
        symbol: Stock symbol (e.g., "AAPL")

    Returns:
        String with comprehensive stock analysis.
    """
    data_pipeline, analysis_pipeline = _get_pipeline()

    async def fetch():
        result = await analysis_pipeline.analyze_stock(symbol.upper())
        await data_pipeline._client.close()
        return result

    r = _run_async(fetch())

    rating_emoji = "🟢" if r.overall_score >= 70 else "🟡" if r.overall_score >= 50 else "🔴"

    return f"""
**{r.company_name}** ({r.symbol}) {rating_emoji}
Sector: {r.sector} | Industry: {r.industry}

**Valuation:**
- Current Price: ${r.current_price:.2f}
- DCF Fair Value: ${r.dcf_value:.2f}
- Upside Potential: {r.upside_potential:+.1f}%
- P/E Ratio: {r.pe_ratio:.2f}
- P/B Ratio: {r.pb_ratio:.2f}
- EV/EBITDA: {r.ev_ebitda:.2f}

**Profitability:**
- Profit Margin: {r.profit_margin*100:.1f}%
- ROE: {r.roe*100:.1f}%
- ROIC: {r.roic*100:.1f}%

**Financial Health:**
- Debt/Equity: {r.debt_to_equity:.2f}

**Technicals:**
- 50-Day SMA: ${r.sma_50:.2f} ({r.price_vs_sma_50:+.1f}% from price)
- 200-Day SMA: ${r.sma_200:.2f} ({r.price_vs_sma_200:+.1f}% from price)

**Risk:**
- Beta: {r.beta:.2f}
- Volatility: {r.volatility*100:.1f}%
- 95% VaR: {r.var_95*100:.2f}%

**Overall Score: {r.overall_score:.0f}/100**
"""


@tool
def analyze_portfolio_tool(
    holdings: str,
    benchmark: str = "SPY",
) -> str:
    """Analyze a portfolio of stocks.

    Use this tool to analyze portfolio performance, risk metrics, and composition.

    Args:
        holdings: Comma-separated holdings in format "SYMBOL:WEIGHT" (e.g., "AAPL:0.3,MSFT:0.3,GOOGL:0.4")
        benchmark: Benchmark symbol (default: SPY)

    Returns:
        String with portfolio analysis results.
    """
    # Parse holdings
    holdings_dict = {}
    for item in holdings.split(","):
        parts = item.strip().split(":")
        if len(parts) == 2:
            symbol = parts[0].strip().upper()
            weight = float(parts[1].strip())
            holdings_dict[symbol] = weight

    if not holdings_dict:
        return "Error: Invalid holdings format. Use 'SYMBOL:WEIGHT,SYMBOL:WEIGHT'"

    data_pipeline, analysis_pipeline = _get_pipeline()

    async def fetch():
        result = await analysis_pipeline.analyze_portfolio(
            holdings_dict, benchmark.upper()
        )
        await data_pipeline._client.close()
        return result

    r = _run_async(fetch())

    weights_str = "\n".join([f"- {sym}: {w*100:.1f}%" for sym, w in r.weights.items()])

    return f"""
**Portfolio Analysis** (Benchmark: {benchmark})

**Allocation:**
{weights_str}

**Performance:**
- Total Return: {r.total_return*100:.2f}%
- Annualized Return: {r.annualized_return*100:.2f}%
- Volatility: {r.volatility*100:.2f}%

**Risk-Adjusted Returns:**
- Sharpe Ratio: {r.sharpe_ratio:.2f}
- Sortino Ratio: {r.sortino_ratio:.2f}
- Calmar Ratio: {r.calmar_ratio:.2f}

**Risk Metrics:**
- 95% VaR: {r.var_95*100:.2f}%
- 99% VaR: {r.var_99*100:.2f}%
- Max Drawdown: {r.max_drawdown*100:.2f}%

**Relative to Benchmark:**
- Beta: {r.beta:.2f}
- Alpha: {r.alpha*100:.2f}%
- Tracking Error: {r.tracking_error*100:.2f}%
- Information Ratio: {r.information_ratio:.2f}
- R-Squared: {r.r_squared:.2f}
"""


@tool
def optimize_portfolio_tool(
    symbols: str,
    risk_free_rate: float = 0.05,
) -> str:
    """Optimize portfolio weights using mean-variance optimization.

    Use this tool to find optimal portfolio weights that maximize the Sharpe ratio.

    Args:
        symbols: Comma-separated stock symbols (e.g., "AAPL,MSFT,GOOGL,AMZN")
        risk_free_rate: Risk-free rate (default: 0.05 = 5%)

    Returns:
        String with optimal portfolio weights and expected metrics.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")]

    data_pipeline, analysis_pipeline = _get_pipeline()

    async def fetch():
        result = await analysis_pipeline.optimize_portfolio(
            symbol_list, risk_free_rate=risk_free_rate
        )
        await data_pipeline._client.close()
        return result

    r = _run_async(fetch())

    weights_str = "\n".join([
        f"- {sym}: {w*100:.1f}%"
        for sym, w in sorted(r["optimal_weights"].items(), key=lambda x: -x[1])
    ])

    return f"""
**Optimal Portfolio** (Maximizing Sharpe Ratio)

**Optimal Weights:**
{weights_str}

**Expected Metrics:**
- Expected Return: {r['expected_return']*100:.2f}%
- Volatility: {r['volatility']*100:.2f}%
- Sharpe Ratio: {r['sharpe_ratio']:.2f}

*Based on historical returns. Past performance does not guarantee future results.*
"""


@tool
def risk_analysis_tool(
    holdings: str,
    portfolio_value: float = 1000000,
) -> str:
    """Perform detailed risk analysis on a portfolio.

    Use this tool to analyze VaR, stress tests, and risk contributions.

    Args:
        holdings: Comma-separated holdings in format "SYMBOL:WEIGHT"
        portfolio_value: Total portfolio value (default: $1,000,000)

    Returns:
        String with comprehensive risk analysis.
    """
    # Parse holdings
    holdings_dict = {}
    for item in holdings.split(","):
        parts = item.strip().split(":")
        if len(parts) == 2:
            symbol = parts[0].strip().upper()
            weight = float(parts[1].strip())
            holdings_dict[symbol] = weight

    if not holdings_dict:
        return "Error: Invalid holdings format. Use 'SYMBOL:WEIGHT,SYMBOL:WEIGHT'"

    data_pipeline, analysis_pipeline = _get_pipeline()

    async def fetch():
        result = await analysis_pipeline.analyze_risk(
            holdings_dict, portfolio_value
        )
        await data_pipeline._client.close()
        return result

    r = _run_async(fetch())

    risk_contrib_str = "\n".join([
        f"- {sym}: {contrib:.1f}%"
        for sym, contrib in sorted(r.risk_contribution.items(), key=lambda x: -x[1])
    ])

    stress_str = "\n".join([
        f"- {scenario}: ${loss:+,.0f}"
        for scenario, loss in r.stress_test_results.items()
    ])

    return f"""
**Risk Analysis** (Portfolio Value: ${portfolio_value:,.0f})

**Value at Risk (VaR):**
- 1-Day 95% VaR: ${r.var_95_1d:,.0f}
- 1-Day 99% VaR: ${r.var_99_1d:,.0f}
- 10-Day 95% VaR: ${r.var_95_10d:,.0f}
- 10-Day 99% VaR: ${r.var_99_10d:,.0f}

**Expected Shortfall (CVaR):**
- 95% CVaR: ${r.cvar_95:,.0f}
- 99% CVaR: ${r.cvar_99:,.0f}

**Volatility:**
- Daily: {r.volatility_daily*100:.2f}%
- Annual: {r.volatility_annual*100:.2f}%
- Downside: {r.volatility_downside*100:.2f}%

**Drawdown:**
- Maximum: {r.max_drawdown*100:.2f}%
- Average: {r.avg_drawdown*100:.2f}%

**Risk Contributions:**
{risk_contrib_str}

**Stress Test Results:**
{stress_str}
"""


@tool
def dcf_valuation_tool(
    symbol: str,
    growth_rate: float = 0.05,
    terminal_growth: float = 0.02,
    discount_rate: float = 0.10,
) -> str:
    """Perform DCF (Discounted Cash Flow) valuation.

    Use this tool to calculate the intrinsic value of a stock using DCF model.

    Args:
        symbol: Stock symbol (e.g., "AAPL")
        growth_rate: FCF growth rate (default: 5%)
        terminal_growth: Terminal growth rate (default: 2%)
        discount_rate: Discount rate/WACC (default: 10%)

    Returns:
        String with DCF valuation results.
    """
    data_pipeline, analysis_pipeline = _get_pipeline()

    async def fetch():
        result = await analysis_pipeline.dcf_analysis(
            symbol.upper(),
            growth_rate,
            terminal_growth,
            discount_rate,
        )
        await data_pipeline._client.close()
        return result

    r = _run_async(fetch())

    fcf_str = "\n".join([
        f"- Year {i+1}: ${fcf:,.0f}"
        for i, fcf in enumerate(r["projected_fcfs"])
    ])

    return f"""
**DCF Valuation: {r['symbol']}**

**Result:**
- Current Price: ${r['current_price']:.2f}
- Intrinsic Value: ${r['intrinsic_value']:.2f}
- Upside Potential: {r['upside_potential']:+.1f}%

**Assumptions:**
- Latest FCF: ${r['assumptions']['latest_fcf']:,.0f}
- Growth Rate: {r['assumptions']['growth_rate']*100:.1f}%
- Terminal Growth: {r['assumptions']['terminal_growth']*100:.1f}%
- Discount Rate (WACC): {r['assumptions']['discount_rate']*100:.1f}%

**Projected Free Cash Flows:**
{fcf_str}

*Note: DCF valuations are sensitive to assumptions. Adjust inputs for sensitivity analysis.*
"""


@tool
def options_analysis_tool(
    symbol: str,
    strike: float,
    expiry_days: int,
    risk_free_rate: float = 0.05,
) -> str:
    """Analyze options pricing using Black-Scholes model.

    Use this tool to calculate theoretical option prices and Greeks.

    Args:
        symbol: Stock symbol (e.g., "AAPL")
        strike: Strike price
        expiry_days: Days until expiration
        risk_free_rate: Risk-free rate (default: 5%)

    Returns:
        String with options analysis including prices and Greeks.
    """
    data_pipeline, analysis_pipeline = _get_pipeline()

    async def fetch():
        result = await analysis_pipeline.options_analysis(
            symbol.upper(),
            strike,
            expiry_days,
            risk_free_rate,
        )
        await data_pipeline._client.close()
        return result

    r = _run_async(fetch())

    return f"""
**Options Analysis: {r['symbol']}**

**Parameters:**
- Spot Price: ${r['spot_price']:.2f}
- Strike Price: ${r['strike_price']:.2f}
- Time to Expiry: {r['time_to_expiry']*365:.0f} days
- Volatility: {r['volatility']*100:.1f}%
- Risk-Free Rate: {r['risk_free_rate']*100:.1f}%

**Theoretical Prices (Black-Scholes):**
- Call Price: ${r['call_price']:.2f}
- Put Price: ${r['put_price']:.2f}

**Greeks:**
- Delta: {r['greeks']['delta']:.4f}
- Gamma: {r['greeks']['gamma']:.4f}
- Theta: ${r['greeks']['theta']:.4f}/day
- Vega: ${r['greeks']['vega']:.4f}/1% vol
- Rho: ${r['greeks']['rho']:.4f}/1% rate

*Prices based on Black-Scholes model. Actual market prices may differ.*
"""
