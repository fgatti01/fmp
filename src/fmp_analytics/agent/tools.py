"""Financial analysis tools for Agno agent.

This module provides tool functions that can be used by the Agno agent
to perform financial analysis tasks.
"""

import asyncio
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
from agno.tools import tool

from fmp_analytics.api.client import FMPClient
from fmp_analytics.metrics.portfolio import PortfolioManagement
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


@tool
def sector_analysis_tool(sector: str) -> str:
    """Analyze a sector and identify stocks with unusual behavior or distortions.

    Use this tool to find anomalies, outliers, and distortions within a sector.
    This includes stocks with unusual price movements, valuation discrepancies,
    earnings surprises, or divergence from sector trends.

    Args:
        sector: Sector name (e.g., "Technology", "Healthcare", "Financial Services",
                "Consumer Cyclical", "Industrials", "Energy", "Basic Materials",
                "Communication Services", "Consumer Defensive", "Utilities", "Real Estate")

    Returns:
        String with sector analysis including identified distortions and anomalies.
    """
    data_pipeline, _ = _get_pipeline()

    async def fetch():
        # Get stocks in the sector
        stocks = await data_pipeline._stock_list.get_stock_screener(
            sector=sector,
            is_actively_trading=True,
            limit=50,
        )

        if not stocks:
            return None, None, None, None

        symbols = [s["symbol"] for s in stocks[:30]]  # Top 30 by market cap

        # Get quotes for all stocks
        quotes = await data_pipeline.get_quotes_batch(symbols)

        # Get sector performance
        sector_perf = await data_pipeline.get_sector_performance()

        # Get earnings surprises for top stocks
        earnings_data = []
        for sym in symbols[:10]:
            try:
                surprises = await data_pipeline._financials.get_earnings_surprises(sym)
                if surprises:
                    earnings_data.append({"symbol": sym, "data": surprises[0] if surprises else None})
            except Exception:
                pass

        await data_pipeline._client.close()
        return quotes, sector_perf, stocks, earnings_data

    quotes, sector_perf, stocks, earnings_data = _run_async(fetch())

    if not quotes:
        return f"No data found for sector: {sector}"

    # Analyze for distortions
    result = f"**Sector Analysis: {sector}**\n\n"

    # Sector performance
    sector_change = None
    for _, row in sector_perf.iterrows():
        if sector.lower() in row.get("sector", "").lower():
            sector_change = float(row.get("changesPercentage", "0").replace("%", ""))
            break

    if sector_change is not None:
        change_style = "📈" if sector_change >= 0 else "📉"
        result += f"**Sector Performance Today:** {change_style} {sector_change:+.2f}%\n\n"

    # Find outliers (stocks moving significantly different from sector)
    outliers_up = []
    outliers_down = []
    high_volume = []
    valuation_outliers = []

    avg_change = sum(q.changes_percentage or 0 for q in quotes) / len(quotes) if quotes else 0

    for q in quotes:
        change = q.changes_percentage or 0
        deviation = change - avg_change

        # Price movement outliers (>2x sector move or opposite direction)
        if deviation > 5:
            outliers_up.append((q.symbol, q.name, change, q.price, q.volume))
        elif deviation < -5:
            outliers_down.append((q.symbol, q.name, change, q.price, q.volume))

        # High volume (potential unusual activity)
        if q.volume and q.avg_volume and q.volume > q.avg_volume * 2:
            high_volume.append((q.symbol, q.name, q.volume / q.avg_volume, change))

        # Valuation outliers
        if q.pe and q.pe > 0:
            if q.pe > 50:
                valuation_outliers.append((q.symbol, q.name, q.pe, "High P/E"))
            elif q.pe < 5:
                valuation_outliers.append((q.symbol, q.name, q.pe, "Low P/E"))

    # Report outliers
    if outliers_up:
        result += "**🔺 Positive Outliers (Outperforming Sector):**\n"
        for sym, name, change, price, vol in sorted(outliers_up, key=lambda x: -x[2])[:5]:
            result += f"- **{sym}** ({name}): {change:+.2f}% @ ${price:.2f}\n"
            result += f"  *Deviation: +{change - avg_change:.1f}% from sector avg*\n"
        result += "\n"

    if outliers_down:
        result += "**🔻 Negative Outliers (Underperforming Sector):**\n"
        for sym, name, change, price, vol in sorted(outliers_down, key=lambda x: x[2])[:5]:
            result += f"- **{sym}** ({name}): {change:+.2f}% @ ${price:.2f}\n"
            result += f"  *Deviation: {change - avg_change:.1f}% from sector avg*\n"
        result += "\n"

    if high_volume:
        result += "**📊 Unusual Volume (>2x Average):**\n"
        for sym, name, vol_ratio, change in sorted(high_volume, key=lambda x: -x[2])[:5]:
            result += f"- **{sym}** ({name}): {vol_ratio:.1f}x avg volume, {change:+.2f}%\n"
            result += f"  *Potential catalyst or institutional activity*\n"
        result += "\n"

    if valuation_outliers:
        result += "**💰 Valuation Anomalies:**\n"
        for sym, name, pe, reason in valuation_outliers[:5]:
            result += f"- **{sym}** ({name}): P/E = {pe:.1f} ({reason})\n"
        result += "\n"

    # Earnings surprises
    if earnings_data:
        result += "**📋 Recent Earnings Surprises:**\n"
        for item in earnings_data:
            if item["data"]:
                d = item["data"]
                actual = d.get("actualEarningResult", 0)
                estimated = d.get("estimatedEarning", 0)
                if estimated and actual:
                    surprise_pct = ((actual - estimated) / abs(estimated)) * 100
                    emoji = "✅" if surprise_pct > 0 else "❌"
                    result += f"- **{item['symbol']}**: {emoji} {surprise_pct:+.1f}% surprise (${actual:.2f} vs ${estimated:.2f} est)\n"

    result += "\n**Analysis Summary:**\n"
    result += f"- Sector average change: {avg_change:+.2f}%\n"
    result += f"- Stocks outperforming: {len(outliers_up)}\n"
    result += f"- Stocks underperforming: {len(outliers_down)}\n"
    result += f"- Unusual volume activity: {len(high_volume)}\n"

    return result


@tool
def find_sector_distortions_tool(min_deviation: float = 3.0) -> str:
    """Find distortions and anomalies across all sectors.

    Use this tool to identify stocks that are behaving unusually compared to their
    sector peers. This helps find potential opportunities or risks from mispricing,
    earnings surprises, or market dislocations.

    Args:
        min_deviation: Minimum deviation from sector average to flag (default: 3%)

    Returns:
        String with all identified distortions across market sectors.
    """
    data_pipeline, _ = _get_pipeline()

    async def fetch():
        # Get market movers to identify unusual activity
        movers = await data_pipeline.get_market_movers()

        # Get sector performance
        sector_perf = await data_pipeline.get_sector_performance()

        # Get S&P 500 constituents for high-quality analysis
        sp500 = await data_pipeline._stock_list.get_sp500_constituents()
        sp500_symbols = [s["symbol"] for s in sp500[:100]]

        # Get quotes
        quotes = await data_pipeline.get_quotes_batch(sp500_symbols)

        # Get profiles for sector info
        profiles = await data_pipeline.get_company_profiles_batch(sp500_symbols[:50])

        await data_pipeline._client.close()
        return movers, sector_perf, quotes, profiles

    movers, sector_perf, quotes, profiles = _run_async(fetch())

    result = "**Market-Wide Distortion Analysis**\n\n"

    # Sector performance summary
    result += "**Sector Performance Today:**\n"
    sector_changes = {}
    for _, row in sector_perf.iterrows():
        sector_name = row.get("sector", "Unknown")
        change = float(row.get("changesPercentage", "0").replace("%", ""))
        sector_changes[sector_name] = change
        emoji = "📈" if change >= 0 else "📉"
        result += f"- {emoji} {sector_name}: {change:+.2f}%\n"
    result += "\n"

    # Group quotes by sector
    sector_stocks = {}
    profile_lookup = {p.symbol: p for p in profiles}

    for q in quotes:
        profile = profile_lookup.get(q.symbol)
        if profile and profile.sector:
            if profile.sector not in sector_stocks:
                sector_stocks[profile.sector] = []
            sector_stocks[profile.sector].append({
                "symbol": q.symbol,
                "name": q.name,
                "change": q.changes_percentage or 0,
                "price": q.price,
                "pe": q.pe,
                "volume_ratio": (q.volume / q.avg_volume) if q.avg_volume else 1,
            })

    # Find distortions in each sector
    distortions = []

    for sector, stocks in sector_stocks.items():
        if len(stocks) < 3:
            continue

        avg_change = sum(s["change"] for s in stocks) / len(stocks)

        for stock in stocks:
            deviation = stock["change"] - avg_change

            if abs(deviation) >= min_deviation:
                distortions.append({
                    "symbol": stock["symbol"],
                    "name": stock["name"],
                    "sector": sector,
                    "change": stock["change"],
                    "sector_avg": avg_change,
                    "deviation": deviation,
                    "volume_ratio": stock["volume_ratio"],
                    "pe": stock["pe"],
                })

    # Sort by absolute deviation
    distortions.sort(key=lambda x: abs(x["deviation"]), reverse=True)

    if distortions:
        result += f"**Identified Distortions (>{min_deviation}% deviation):**\n\n"

        # Group by direction
        positive_distortions = [d for d in distortions if d["deviation"] > 0]
        negative_distortions = [d for d in distortions if d["deviation"] < 0]

        if positive_distortions:
            result += "**🟢 Outperforming (Potential Catalysts):**\n"
            for d in positive_distortions[:7]:
                result += f"\n**{d['symbol']}** - {d['name']}\n"
                result += f"- Sector: {d['sector']}\n"
                result += f"- Stock Change: {d['change']:+.2f}%\n"
                result += f"- Sector Avg: {d['sector_avg']:+.2f}%\n"
                result += f"- **Deviation: +{d['deviation']:.2f}%**\n"
                if d['volume_ratio'] > 1.5:
                    result += f"- ⚠️ Volume {d['volume_ratio']:.1f}x average\n"
                result += f"- *Possible reasons: Positive earnings, upgrade, M&A news, product launch*\n"
            result += "\n"

        if negative_distortions:
            result += "**🔴 Underperforming (Potential Concerns):**\n"
            for d in negative_distortions[:7]:
                result += f"\n**{d['symbol']}** - {d['name']}\n"
                result += f"- Sector: {d['sector']}\n"
                result += f"- Stock Change: {d['change']:+.2f}%\n"
                result += f"- Sector Avg: {d['sector_avg']:+.2f}%\n"
                result += f"- **Deviation: {d['deviation']:.2f}%**\n"
                if d['volume_ratio'] > 1.5:
                    result += f"- ⚠️ Volume {d['volume_ratio']:.1f}x average\n"
                result += f"- *Possible reasons: Earnings miss, downgrade, legal issues, guidance cut*\n"

    else:
        result += "*No significant distortions found at the specified threshold.*\n"

    # Top movers summary
    result += "\n**Top Market Movers:**\n"
    result += "\nGainers:\n"
    for _, row in movers["gainers"].head(3).iterrows():
        result += f"- {row['symbol']}: {row['changesPercentage']:+.2f}%\n"
    result += "\nLosers:\n"
    for _, row in movers["losers"].head(3).iterrows():
        result += f"- {row['symbol']}: {row['changesPercentage']:+.2f}%\n"

    return result


@tool
def explain_market_movement_tool(
    symbol: str,
    days: int = 5,
) -> str:
    """Explain why a stock, index, or ETF has moved.

    Use this tool to understand and explain the reasons behind price movements
    for any security. Analyzes technical factors, fundamental data, news catalysts,
    and market context.

    Args:
        symbol: Stock/ETF/Index symbol (e.g., "AAPL", "SPY", "QQQ", "^GSPC")
        days: Number of days to analyze (default: 5)

    Returns:
        String with detailed explanation of the price movement.
    """
    data_pipeline, _ = _get_pipeline()

    async def fetch():
        to_date = date.today()
        from_date = to_date - timedelta(days=days + 30)  # Extra for moving averages

        # Get historical prices
        hist = await data_pipeline.get_historical_prices(symbol.upper(), str(from_date), str(to_date))

        # Get quote
        quote_data = await data_pipeline._market_data.get_quote(symbol.upper())
        quote = quote_data[0] if quote_data else None

        # Get profile (for stocks)
        try:
            profile_data = await data_pipeline._company_info.get_profile(symbol.upper())
            profile = profile_data[0] if profile_data else None
        except Exception:
            profile = None

        # Get sector performance
        sector_perf = await data_pipeline.get_sector_performance()

        # Get earnings surprises (for stocks)
        try:
            earnings = await data_pipeline._financials.get_earnings_surprises(symbol.upper())
        except Exception:
            earnings = []

        # Get analyst recommendations
        try:
            recommendations = await data_pipeline._company_info.get_analyst_recommendations(symbol.upper())
        except Exception:
            recommendations = []

        # Get news/press releases
        try:
            news = await data_pipeline._company_info.get_press_releases(symbol.upper(), limit=5)
        except Exception:
            news = []

        await data_pipeline._client.close()
        return hist, quote, profile, sector_perf, earnings, recommendations, news

    hist, quote, profile, sector_perf, earnings, recommendations, news = _run_async(fetch())

    if hist.empty:
        return f"No data found for {symbol}"

    result = f"**Market Movement Analysis: {symbol.upper()}**\n\n"

    # Current status
    if quote:
        change_emoji = "📈" if (quote.get("change", 0) or 0) >= 0 else "📉"
        result += f"**Current Status:** {change_emoji}\n"
        result += f"- Price: ${quote.get('price', 0):.2f}\n"
        result += f"- Today's Change: {quote.get('change', 0):+.2f} ({quote.get('changesPercentage', 0):+.2f}%)\n"
        result += f"- Volume: {quote.get('volume', 0):,} ({quote.get('volume', 0) / quote.get('avgVolume', 1):.1f}x avg)\n\n"

    # Period performance
    if len(hist) >= days:
        recent = hist.tail(days)
        period_start = recent.iloc[0]["adjClose"]
        period_end = recent.iloc[-1]["adjClose"]
        period_return = ((period_end / period_start) - 1) * 100

        result += f"**{days}-Day Performance:**\n"
        result += f"- Return: {period_return:+.2f}%\n"
        result += f"- High: ${recent['high'].max():.2f}\n"
        result += f"- Low: ${recent['low'].min():.2f}\n"
        result += f"- Avg Volume: {recent['volume'].mean():,.0f}\n\n"

    # Technical analysis
    result += "**Technical Factors:**\n"

    if len(hist) >= 50:
        sma_20 = hist["adjClose"].tail(20).mean()
        sma_50 = hist["adjClose"].tail(50).mean()
        current_price = hist.iloc[-1]["adjClose"]

        # Trend analysis
        if current_price > sma_20 > sma_50:
            result += "- 📈 **Uptrend**: Price > 20-day SMA > 50-day SMA\n"
            result += "  *Bullish momentum with moving average support*\n"
        elif current_price < sma_20 < sma_50:
            result += "- 📉 **Downtrend**: Price < 20-day SMA < 50-day SMA\n"
            result += "  *Bearish momentum with moving average resistance*\n"
        elif sma_20 > sma_50 and current_price < sma_20:
            result += "- ⚠️ **Pullback in Uptrend**: Testing 20-day support\n"
        else:
            result += "- ↔️ **Consolidation**: Mixed signals, range-bound\n"

        # RSI approximation
        changes = hist["adjClose"].diff().tail(14)
        gains = changes[changes > 0].sum()
        losses = abs(changes[changes < 0].sum())
        if losses > 0:
            rs = gains / losses
            rsi = 100 - (100 / (1 + rs))
            if rsi > 70:
                result += f"- 🔴 **RSI: {rsi:.0f}** (Overbought - potential pullback)\n"
            elif rsi < 30:
                result += f"- 🟢 **RSI: {rsi:.0f}** (Oversold - potential bounce)\n"
            else:
                result += f"- ⚪ **RSI: {rsi:.0f}** (Neutral)\n"

        # Volatility
        volatility = hist["adjClose"].tail(20).pct_change().std() * (252 ** 0.5) * 100
        result += f"- Volatility (20d): {volatility:.1f}% annualized\n"

    result += "\n"

    # Sector context
    if profile:
        sector = profile.get("sector", "Unknown")
        result += f"**Sector Context ({sector}):**\n"

        sector_change = None
        for _, row in sector_perf.iterrows():
            if sector.lower() in row.get("sector", "").lower():
                sector_change = float(row.get("changesPercentage", "0").replace("%", ""))
                break

        if sector_change is not None:
            result += f"- Sector Performance Today: {sector_change:+.2f}%\n"

            stock_change = quote.get("changesPercentage", 0) if quote else 0
            relative = stock_change - sector_change
            if abs(relative) > 2:
                if relative > 0:
                    result += f"- ✨ **Outperforming sector by {relative:+.1f}%** (stock-specific catalyst likely)\n"
                else:
                    result += f"- ⚠️ **Underperforming sector by {relative:.1f}%** (company-specific issue likely)\n"
            else:
                result += f"- Moving in-line with sector ({relative:+.1f}% relative)\n"

        result += "\n"

    # Fundamental factors
    result += "**Potential Fundamental Drivers:**\n"

    # Earnings
    if earnings:
        latest = earnings[0]
        actual = latest.get("actualEarningResult")
        estimated = latest.get("estimatedEarning")
        report_date = latest.get("date", "Unknown")

        if actual and estimated:
            surprise = ((actual - estimated) / abs(estimated)) * 100
            result += f"- 📋 **Last Earnings ({report_date})**: "
            if surprise > 5:
                result += f"Beat by {surprise:.1f}% ✅\n"
                result += "  *Positive earnings surprise typically drives appreciation*\n"
            elif surprise < -5:
                result += f"Missed by {abs(surprise):.1f}% ❌\n"
                result += "  *Negative earnings surprise typically causes decline*\n"
            else:
                result += f"In-line ({surprise:+.1f}%)\n"

    # Analyst sentiment
    if recommendations:
        recent_recs = recommendations[:3]
        upgrades = sum(1 for r in recent_recs if "upgrade" in str(r.get("newGrade", "")).lower())
        downgrades = sum(1 for r in recent_recs if "downgrade" in str(r.get("newGrade", "")).lower())

        if upgrades > 0:
            result += f"- 📈 **{upgrades} recent analyst upgrade(s)**\n"
        if downgrades > 0:
            result += f"- 📉 **{downgrades} recent analyst downgrade(s)**\n"

    # News
    if news:
        result += "- 📰 **Recent News:**\n"
        for n in news[:3]:
            title = n.get("title", "")[:60]
            result += f"  - {title}...\n"

    result += "\n**Movement Explanation Summary:**\n"

    # Generate explanation
    explanations = []

    stock_change = quote.get("changesPercentage", 0) if quote else 0
    is_positive = stock_change >= 0

    if abs(stock_change) < 1:
        explanations.append("Minor movement within normal trading range")
    elif abs(stock_change) < 3:
        explanations.append("Moderate movement, likely following broader market trends")
    else:
        explanations.append("Significant movement suggesting specific catalyst")

    if earnings and abs(((earnings[0].get("actualEarningResult", 0) or 0) - (earnings[0].get("estimatedEarning", 0) or 1)) / abs(earnings[0].get("estimatedEarning", 1) or 1)) > 0.05:
        explanations.append("Recent earnings results impacting sentiment")

    if quote and quote.get("volume", 0) > quote.get("avgVolume", 1) * 2:
        explanations.append("Unusually high volume indicates institutional activity or news catalyst")

    for exp in explanations:
        result += f"- {exp}\n"

    return result


@tool
def explain_indicator_tool(
    indicator: str,
    symbol: str = "SPY",
) -> str:
    """Explain a technical indicator's current reading and what it means.

    Use this tool to understand technical indicators like RSI, MACD, moving averages,
    Bollinger Bands, and what their current values indicate for a security.

    Args:
        indicator: Indicator name ("RSI", "MACD", "SMA", "EMA", "BB", "ATR", "OBV")
        symbol: Symbol to analyze (default: SPY for market overview)

    Returns:
        String with indicator explanation and interpretation.
    """
    data_pipeline, _ = _get_pipeline()

    async def fetch():
        to_date = date.today()
        from_date = to_date - timedelta(days=100)

        hist = await data_pipeline.get_historical_prices(symbol.upper(), str(from_date), str(to_date))
        quote = await data_pipeline._market_data.get_quote(symbol.upper())

        await data_pipeline._client.close()
        return hist, quote[0] if quote else None

    hist, quote = _run_async(fetch())

    if hist.empty:
        return f"No data found for {symbol}"

    import numpy as np

    result = f"**{indicator.upper()} Analysis for {symbol.upper()}**\n\n"

    prices = hist["adjClose"].values
    highs = hist["high"].values
    lows = hist["low"].values
    closes = hist["close"].values
    volumes = hist["volume"].values

    indicator = indicator.upper()

    if indicator == "RSI":
        # Calculate RSI
        period = 14
        changes = np.diff(prices)
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 100

        result += f"**RSI (Relative Strength Index) - 14 Period**\n\n"
        result += f"Current Value: **{rsi:.1f}**\n\n"

        result += "**What RSI Measures:**\n"
        result += "RSI measures the speed and magnitude of recent price changes to evaluate "
        result += "overbought or oversold conditions. It oscillates between 0 and 100.\n\n"

        result += "**Interpretation:**\n"
        if rsi > 70:
            result += f"🔴 **OVERBOUGHT** (RSI > 70)\n"
            result += "- The security may be overvalued and due for a pullback\n"
            result += "- Buyers have been aggressive; momentum may be exhausting\n"
            result += "- Consider taking profits or waiting for better entry\n"
            result += "- However, strong trends can remain overbought for extended periods\n"
        elif rsi < 30:
            result += f"🟢 **OVERSOLD** (RSI < 30)\n"
            result += "- The security may be undervalued and due for a bounce\n"
            result += "- Sellers have been aggressive; selling pressure may be exhausting\n"
            result += "- Potential buying opportunity for contrarian traders\n"
            result += "- However, downtrends can remain oversold for extended periods\n"
        elif rsi > 50:
            result += f"⬆️ **BULLISH MOMENTUM** (RSI 50-70)\n"
            result += "- Buying pressure exceeds selling pressure\n"
            result += "- Uptrend is intact with room to run\n"
            result += "- Watch for RSI to stay above 50 for trend confirmation\n"
        else:
            result += f"⬇️ **BEARISH MOMENTUM** (RSI 30-50)\n"
            result += "- Selling pressure exceeds buying pressure\n"
            result += "- Downtrend or consolidation phase\n"
            result += "- Watch for RSI to reclaim 50 for bullish reversal\n"

        result += f"\n**Trading Signals:**\n"
        result += "- Bullish divergence: Price makes lower low, RSI makes higher low\n"
        result += "- Bearish divergence: Price makes higher high, RSI makes lower high\n"

    elif indicator in ["SMA", "EMA"]:
        # Calculate moving averages
        sma_20 = np.mean(prices[-20:])
        sma_50 = np.mean(prices[-50:])
        sma_200 = np.mean(prices[-200:]) if len(prices) >= 200 else np.mean(prices)

        # EMA calculation
        def ema(data, period):
            multiplier = 2 / (period + 1)
            ema_values = [data[0]]
            for price in data[1:]:
                ema_values.append((price * multiplier) + (ema_values[-1] * (1 - multiplier)))
            return ema_values[-1]

        ema_12 = ema(prices, 12)
        ema_26 = ema(prices, 26)

        current_price = prices[-1]

        result += f"**Moving Averages Analysis**\n\n"
        result += f"Current Price: ${current_price:.2f}\n\n"

        result += "**Simple Moving Averages (SMA):**\n"
        result += f"- 20-day SMA: ${sma_20:.2f} ({((current_price/sma_20)-1)*100:+.1f}% from price)\n"
        result += f"- 50-day SMA: ${sma_50:.2f} ({((current_price/sma_50)-1)*100:+.1f}% from price)\n"
        result += f"- 200-day SMA: ${sma_200:.2f} ({((current_price/sma_200)-1)*100:+.1f}% from price)\n\n"

        result += "**Exponential Moving Averages (EMA):**\n"
        result += f"- 12-day EMA: ${ema_12:.2f}\n"
        result += f"- 26-day EMA: ${ema_26:.2f}\n\n"

        result += "**What Moving Averages Tell Us:**\n"
        result += "- MAs smooth price data to identify trend direction\n"
        result += "- Shorter MAs react faster; longer MAs show major trends\n"
        result += "- Price above MA = bullish; below = bearish\n\n"

        result += "**Current Interpretation:**\n"
        if current_price > sma_20 > sma_50 > sma_200:
            result += "📈 **STRONG UPTREND**\n"
            result += "- All MAs aligned bullishly (Golden arrangement)\n"
            result += "- Price has strong support from rising averages\n"
        elif current_price < sma_20 < sma_50 < sma_200:
            result += "📉 **STRONG DOWNTREND**\n"
            result += "- All MAs aligned bearishly (Death arrangement)\n"
            result += "- Price faces resistance from falling averages\n"
        elif current_price > sma_200 and sma_50 > sma_200:
            result += "⬆️ **BULLISH** (Above 200-day, 50 > 200)\n"
            result += "- Long-term trend is up\n"
            result += "- 200-day acts as support\n"
        elif current_price < sma_200:
            result += "⬇️ **BEARISH** (Below 200-day)\n"
            result += "- Long-term trend is down or transitioning\n"
            result += "- 200-day acts as resistance\n"
        else:
            result += "↔️ **TRANSITIONING**\n"
            result += "- Mixed signals, trend may be changing\n"

        # Golden/Death cross
        if sma_50 > sma_200 and np.mean(prices[-55:-50]) < np.mean(prices[-205:-200]):
            result += "\n🌟 **Recent Golden Cross** (50 crossed above 200)\n"
            result += "- Historically bullish long-term signal\n"
        elif sma_50 < sma_200 and np.mean(prices[-55:-50]) > np.mean(prices[-205:-200]):
            result += "\n💀 **Recent Death Cross** (50 crossed below 200)\n"
            result += "- Historically bearish long-term signal\n"

    elif indicator == "MACD":
        # Calculate MACD
        def ema(data, period):
            multiplier = 2 / (period + 1)
            ema_values = [data[0]]
            for price in data[1:]:
                ema_values.append((price * multiplier) + (ema_values[-1] * (1 - multiplier)))
            return np.array(ema_values)

        ema_12 = ema(prices, 12)
        ema_26 = ema(prices, 26)
        macd_line = ema_12 - ema_26
        signal_line = ema(macd_line, 9)
        histogram = macd_line - signal_line

        result += f"**MACD (Moving Average Convergence Divergence)**\n\n"
        result += f"- MACD Line: {macd_line[-1]:.2f}\n"
        result += f"- Signal Line: {signal_line[-1]:.2f}\n"
        result += f"- Histogram: {histogram[-1]:.2f}\n\n"

        result += "**What MACD Measures:**\n"
        result += "MACD shows the relationship between two EMAs. It helps identify:\n"
        result += "- Trend direction and strength\n"
        result += "- Momentum changes\n"
        result += "- Potential buy/sell signals\n\n"

        result += "**Current Interpretation:**\n"
        if macd_line[-1] > signal_line[-1]:
            result += "📈 **BULLISH** (MACD above Signal)\n"
            if macd_line[-1] > 0:
                result += "- Strong bullish momentum (MACD positive)\n"
            else:
                result += "- Bullish crossover but still negative territory\n"
            if histogram[-1] > histogram[-2]:
                result += "- Histogram expanding: momentum increasing ✨\n"
            else:
                result += "- Histogram contracting: momentum may be weakening\n"
        else:
            result += "📉 **BEARISH** (MACD below Signal)\n"
            if macd_line[-1] < 0:
                result += "- Strong bearish momentum (MACD negative)\n"
            else:
                result += "- Bearish crossover but still positive territory\n"
            if histogram[-1] < histogram[-2]:
                result += "- Histogram expanding down: selling pressure increasing\n"
            else:
                result += "- Histogram contracting: selling may be slowing\n"

        result += "\n**Trading Signals:**\n"
        result += "- Bullish: MACD crosses above signal line\n"
        result += "- Bearish: MACD crosses below signal line\n"
        result += "- Zero line crossover confirms trend change\n"

    elif indicator == "BB":
        # Bollinger Bands
        period = 20
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        current_price = prices[-1]

        bandwidth = (upper_band - lower_band) / sma * 100
        percent_b = (current_price - lower_band) / (upper_band - lower_band) * 100

        result += f"**Bollinger Bands (20, 2)**\n\n"
        result += f"- Upper Band: ${upper_band:.2f}\n"
        result += f"- Middle (SMA): ${sma:.2f}\n"
        result += f"- Lower Band: ${lower_band:.2f}\n"
        result += f"- Current Price: ${current_price:.2f}\n"
        result += f"- %B: {percent_b:.1f}%\n"
        result += f"- Bandwidth: {bandwidth:.1f}%\n\n"

        result += "**What Bollinger Bands Tell Us:**\n"
        result += "- Bands expand during high volatility, contract during low volatility\n"
        result += "- Price tends to stay within bands 95% of the time\n"
        result += "- %B shows where price is relative to bands\n\n"

        result += "**Current Interpretation:**\n"
        if percent_b > 100:
            result += "🔴 **ABOVE UPPER BAND** (Overbought)\n"
            result += "- Price is extended beyond normal range\n"
            result += "- Potential for mean reversion pullback\n"
            result += "- Or could indicate breakout if volume confirms\n"
        elif percent_b < 0:
            result += "🟢 **BELOW LOWER BAND** (Oversold)\n"
            result += "- Price is depressed below normal range\n"
            result += "- Potential for mean reversion bounce\n"
            result += "- Or could indicate breakdown if volume confirms\n"
        elif percent_b > 80:
            result += "⚠️ **NEAR UPPER BAND** (Getting extended)\n"
            result += "- Approaching resistance zone\n"
        elif percent_b < 20:
            result += "⚠️ **NEAR LOWER BAND** (Getting oversold)\n"
            result += "- Approaching support zone\n"
        else:
            result += "↔️ **MID-RANGE** (Neutral)\n"
            result += "- Price is within normal trading range\n"

        if bandwidth < 5:
            result += "\n🔔 **SQUEEZE ALERT**: Low bandwidth indicates potential breakout incoming\n"

    else:
        result += f"Indicator '{indicator}' not recognized.\n"
        result += "Available indicators: RSI, SMA, EMA, MACD, BB (Bollinger Bands)\n"

    return result


@tool
def earnings_surprise_analysis_tool(
    sector: str = "",
    limit: int = 20,
) -> str:
    """Analyze recent earnings surprises and their market impact.

    Use this tool to find stocks with significant earnings beats or misses
    and understand how the market reacted.

    Args:
        sector: Optional sector filter (leave empty for all sectors)
        limit: Number of companies to analyze (default: 20)

    Returns:
        String with earnings surprise analysis and market reactions.
    """
    data_pipeline, _ = _get_pipeline()

    async def fetch():
        # Get earnings calendar (recent)
        to_date = date.today()
        from_date = to_date - timedelta(days=30)

        earnings = await data_pipeline.get_earnings_calendar(str(from_date), str(to_date))

        # Get quotes for companies with earnings
        if not earnings.empty:
            symbols = earnings["symbol"].head(limit).tolist()
            quotes = await data_pipeline.get_quotes_batch(symbols)
        else:
            quotes = []

        await data_pipeline._client.close()
        return earnings, quotes

    earnings_df, quotes = _run_async(fetch())

    if earnings_df.empty:
        return "No recent earnings data available."

    result = "**Recent Earnings Surprises Analysis**\n\n"

    quote_lookup = {q.symbol: q for q in quotes}

    beats = []
    misses = []
    in_line = []

    for _, row in earnings_df.head(limit).iterrows():
        symbol = row.get("symbol", "")
        actual = row.get("eps", 0) or 0
        estimate = row.get("epsEstimated", 0) or 0
        report_date = row.get("date", "")

        if not estimate or estimate == 0:
            continue

        surprise_pct = ((actual - estimate) / abs(estimate)) * 100

        quote = quote_lookup.get(symbol)
        price_change = quote.changes_percentage if quote else 0

        entry = {
            "symbol": symbol,
            "actual": actual,
            "estimate": estimate,
            "surprise": surprise_pct,
            "date": report_date,
            "price_change": price_change,
            "price": quote.price if quote else 0,
        }

        if surprise_pct > 5:
            beats.append(entry)
        elif surprise_pct < -5:
            misses.append(entry)
        else:
            in_line.append(entry)

    # Sort by surprise magnitude
    beats.sort(key=lambda x: -x["surprise"])
    misses.sort(key=lambda x: x["surprise"])

    if beats:
        result += "**📈 Significant Earnings Beats (>5% surprise):**\n\n"
        for b in beats[:7]:
            emoji = "✅" if b["price_change"] > 0 else "⚠️"
            result += f"**{b['symbol']}** - {b['date']}\n"
            result += f"- EPS: ${b['actual']:.2f} vs ${b['estimate']:.2f} est\n"
            result += f"- Surprise: **+{b['surprise']:.1f}%**\n"
            result += f"- Stock Reaction: {emoji} {b['price_change']:+.1f}%\n"
            if b["price_change"] < 0 and b["surprise"] > 10:
                result += f"  *⚠️ Beat but sold off - possible guidance concerns or 'sell the news'*\n"
            elif b["price_change"] > b["surprise"]:
                result += f"  *Stock reaction exceeded surprise - positive forward guidance likely*\n"
            result += "\n"

    if misses:
        result += "**📉 Significant Earnings Misses (<-5% surprise):**\n\n"
        for m in misses[:7]:
            emoji = "❌" if m["price_change"] < 0 else "🤔"
            result += f"**{m['symbol']}** - {m['date']}\n"
            result += f"- EPS: ${m['actual']:.2f} vs ${m['estimate']:.2f} est\n"
            result += f"- Surprise: **{m['surprise']:.1f}%**\n"
            result += f"- Stock Reaction: {emoji} {m['price_change']:+.1f}%\n"
            if m["price_change"] > 0 and m["surprise"] < -10:
                result += f"  *🤔 Missed but rallied - likely positive guidance or low expectations*\n"
            elif m["price_change"] < m["surprise"]:
                result += f"  *Stock fell more than miss - additional concerns beyond EPS*\n"
            result += "\n"

    # Summary statistics
    all_surprises = beats + misses + in_line
    if all_surprises:
        avg_surprise = sum(e["surprise"] for e in all_surprises) / len(all_surprises)
        beat_rate = len(beats) / len(all_surprises) * 100

        result += "**Summary Statistics:**\n"
        result += f"- Companies analyzed: {len(all_surprises)}\n"
        result += f"- Beat rate: {beat_rate:.0f}%\n"
        result += f"- Average surprise: {avg_surprise:+.1f}%\n"
        result += f"- Significant beats: {len(beats)}\n"
        result += f"- Significant misses: {len(misses)}\n"

    return result


@tool
def comprehensive_equity_analysis_tool(
    symbol: str,
) -> str:
    """Perform complete equity analysis with fundamental and quantitative metrics.

    This tool gathers ALL available data for an equity and performs:
    - Fundamental analysis (DCF, DDM, multiples, Graham number)
    - Quantitative analysis (risk metrics, technical indicators, momentum)
    - Fair value calculation using multiple methods
    - Comparison of current price vs calculated fair values
    - Buy/Sell/Hold recommendation based on the analysis

    Use this tool when you need a complete picture of an equity's value and prospects.

    Args:
        symbol: Stock symbol (e.g., "AAPL")

    Returns:
        String with comprehensive equity analysis including fair value vs market price.
    """
    data_pipeline, analysis_pipeline = _get_pipeline()

    async def fetch():
        sym = symbol.upper()
        to_date = date.today()
        from_date = to_date - timedelta(days=365)

        # Gather ALL data
        quote_data = await data_pipeline._market_data.get_quote(sym)
        quote = quote_data[0] if quote_data else None

        profile_data = await data_pipeline._company_info.get_profile(sym)
        profile = profile_data[0] if profile_data else None

        # Financial statements
        income_stmt = await data_pipeline._financials.get_income_statement(sym, "annual", limit=5)
        balance_sheet = await data_pipeline._financials.get_balance_sheet(sym, "annual", limit=5)
        cash_flow = await data_pipeline._financials.get_cash_flow_statement(sym, "annual", limit=5)

        # Ratios and metrics
        ratios_df = await data_pipeline.get_financial_ratios(sym, "annual", limit=3)
        key_metrics = await data_pipeline._financials.get_key_metrics(sym, "annual", limit=3)

        # Historical prices for technical analysis
        hist = await data_pipeline.get_historical_prices(sym, str(from_date), str(to_date))

        # DCF value from FMP
        dcf_data = await data_pipeline._company_info.get_dcf(sym)

        # Earnings surprises
        earnings = await data_pipeline._financials.get_earnings_surprises(sym)

        # Analyst estimates
        try:
            estimates = await data_pipeline._financials.get_analyst_estimates(sym, "annual")
        except Exception:
            estimates = []

        # Analyst recommendations
        try:
            recommendations = await data_pipeline._company_info.get_analyst_recommendations(sym)
        except Exception:
            recommendations = []

        # Sector peers for comparison
        if profile:
            peers_data = await data_pipeline._stock_list.get_stock_screener(
                sector=profile.get("sector"),
                is_actively_trading=True,
                limit=20,
            )
            peer_symbols = [p["symbol"] for p in peers_data if p["symbol"] != sym][:10]
            peer_quotes = await data_pipeline.get_quotes_batch(peer_symbols)
        else:
            peer_quotes = []

        await data_pipeline._client.close()
        return (quote, profile, income_stmt, balance_sheet, cash_flow, ratios_df,
                key_metrics, hist, dcf_data, earnings, estimates, recommendations, peer_quotes)

    (quote, profile, income_stmt, balance_sheet, cash_flow, ratios_df,
     key_metrics, hist, dcf_data, earnings, estimates, recommendations, peer_quotes) = _run_async(fetch())

    if not quote or not profile:
        return f"Unable to fetch data for {symbol}"

    import numpy as np

    result = f"# Comprehensive Equity Analysis: {symbol.upper()}\n\n"

    # ===== COMPANY OVERVIEW =====
    result += "## 1. Company Overview\n\n"
    result += f"**{profile.get('companyName', symbol)}** ({symbol.upper()})\n"
    result += f"- Sector: {profile.get('sector', 'N/A')}\n"
    result += f"- Industry: {profile.get('industry', 'N/A')}\n"
    result += f"- Market Cap: ${profile.get('mktCap', 0):,.0f}\n"
    result += f"- Employees: {profile.get('fullTimeEmployees', 'N/A'):,}\n"
    result += f"- Exchange: {profile.get('exchange', 'N/A')}\n\n"

    current_price = quote.get("price", 0)
    result += f"**Current Price: ${current_price:.2f}**\n"
    result += f"- Change Today: {quote.get('change', 0):+.2f} ({quote.get('changesPercentage', 0):+.2f}%)\n"
    result += f"- 52-Week Range: ${quote.get('yearLow', 0):.2f} - ${quote.get('yearHigh', 0):.2f}\n"
    result += f"- Volume: {quote.get('volume', 0):,} (Avg: {quote.get('avgVolume', 0):,})\n\n"

    # ===== FUNDAMENTAL ANALYSIS =====
    result += "## 2. Fundamental Analysis\n\n"

    # Financial Statement Analysis
    result += "### 2.1 Financial Performance\n\n"

    if income_stmt:
        latest_income = income_stmt[0] if income_stmt else {}
        prev_income = income_stmt[1] if len(income_stmt) > 1 else {}

        revenue = latest_income.get("revenue", 0)
        net_income = latest_income.get("netIncome", 0)
        gross_profit = latest_income.get("grossProfit", 0)
        operating_income = latest_income.get("operatingIncome", 0)
        eps = latest_income.get("eps", 0)

        prev_revenue = prev_income.get("revenue", 1)
        revenue_growth = ((revenue / prev_revenue) - 1) * 100 if prev_revenue else 0

        result += "**Income Statement (Latest Year):**\n"
        result += f"- Revenue: ${revenue:,.0f}\n"
        result += f"- Revenue Growth: {revenue_growth:+.1f}% YoY\n"
        result += f"- Gross Profit: ${gross_profit:,.0f} ({gross_profit/revenue*100:.1f}% margin)\n"
        result += f"- Operating Income: ${operating_income:,.0f} ({operating_income/revenue*100:.1f}% margin)\n"
        result += f"- Net Income: ${net_income:,.0f} ({net_income/revenue*100:.1f}% margin)\n"
        result += f"- EPS: ${eps:.2f}\n\n"

    if balance_sheet:
        latest_bs = balance_sheet[0] if balance_sheet else {}

        total_assets = latest_bs.get("totalAssets", 0)
        total_debt = latest_bs.get("totalDebt", 0)
        total_equity = latest_bs.get("totalStockholdersEquity", 0)
        cash = latest_bs.get("cashAndCashEquivalents", 0)

        result += "**Balance Sheet:**\n"
        result += f"- Total Assets: ${total_assets:,.0f}\n"
        result += f"- Cash & Equivalents: ${cash:,.0f}\n"
        result += f"- Total Debt: ${total_debt:,.0f}\n"
        result += f"- Shareholders' Equity: ${total_equity:,.0f}\n"
        result += f"- Debt/Equity Ratio: {total_debt/total_equity:.2f}\n\n" if total_equity else ""

    if cash_flow:
        latest_cf = cash_flow[0] if cash_flow else {}

        operating_cf = latest_cf.get("operatingCashFlow", 0)
        fcf = latest_cf.get("freeCashFlow", 0)
        capex = latest_cf.get("capitalExpenditure", 0)
        dividends = latest_cf.get("dividendsPaid", 0)

        result += "**Cash Flow:**\n"
        result += f"- Operating Cash Flow: ${operating_cf:,.0f}\n"
        result += f"- Free Cash Flow: ${fcf:,.0f}\n"
        result += f"- CapEx: ${capex:,.0f}\n"
        result += f"- Dividends Paid: ${abs(dividends):,.0f}\n\n"

    # Key Ratios
    result += "### 2.2 Key Financial Ratios\n\n"

    if not ratios_df.empty:
        r = ratios_df.iloc[0]

        result += "**Profitability:**\n"
        result += f"- Gross Margin: {r.get('grossProfitMargin', 0)*100:.1f}%\n"
        result += f"- Operating Margin: {r.get('operatingProfitMargin', 0)*100:.1f}%\n"
        result += f"- Net Margin: {r.get('netProfitMargin', 0)*100:.1f}%\n"
        result += f"- ROE: {r.get('returnOnEquity', 0)*100:.1f}%\n"
        result += f"- ROA: {r.get('returnOnAssets', 0)*100:.1f}%\n"
        result += f"- ROIC: {r.get('returnOnCapitalEmployed', 0)*100:.1f}%\n\n"

        result += "**Liquidity:**\n"
        result += f"- Current Ratio: {r.get('currentRatio', 0):.2f}\n"
        result += f"- Quick Ratio: {r.get('quickRatio', 0):.2f}\n"
        result += f"- Cash Ratio: {r.get('cashRatio', 0):.2f}\n\n"

        result += "**Valuation Multiples:**\n"
        pe_ratio = r.get('priceEarningsRatio', 0) or quote.get('pe', 0)
        pb_ratio = r.get('priceToBookRatio', 0)
        ps_ratio = r.get('priceToSalesRatio', 0)
        ev_ebitda = r.get('enterpriseValueMultiple', 0)
        peg_ratio = r.get('priceEarningsToGrowthRatio', 0)

        result += f"- P/E Ratio: {pe_ratio:.2f}\n"
        result += f"- P/B Ratio: {pb_ratio:.2f}\n"
        result += f"- P/S Ratio: {ps_ratio:.2f}\n"
        result += f"- EV/EBITDA: {ev_ebitda:.2f}\n"
        result += f"- PEG Ratio: {peg_ratio:.2f}\n\n"

    # ===== VALUATION ANALYSIS =====
    result += "## 3. Valuation Analysis (Fair Value Calculation)\n\n"
    result += "This section calculates fair value using multiple methodologies, showing the complete logic and calculations for each approach.\n\n"

    fair_values = {}

    # Calculate shares outstanding from market cap
    mkt_cap = profile.get("mktCap", 0)
    shares_outstanding = mkt_cap / current_price if current_price and mkt_cap else 0

    # Method 1: FMP DCF Value
    if dcf_data:
        dcf_value = dcf_data.get("dcf", 0)
        stock_price = dcf_data.get("Stock Price", current_price)
        fair_values["DCF (FMP)"] = dcf_value

        result += "### 3.1 Discounted Cash Flow (DCF) Valuation\n\n"
        result += "**Methodology:** DCF values a company based on the present value of its expected future cash flows.\n\n"
        result += "**Formula:**\n"
        result += "```\n"
        result += "Intrinsic Value = Σ (FCFt / (1 + WACC)^t) + Terminal Value / (1 + WACC)^n\n"
        result += "```\n\n"
        result += "**FMP Calculated DCF:**\n"
        result += f"- DCF Fair Value: **${dcf_value:.2f}**\n"
        result += f"- Current Market Price: ${current_price:.2f}\n"
        upside = ((dcf_value / current_price) - 1) * 100 if current_price else 0
        result += f"- Implied Upside/Downside: **{upside:+.1f}%**\n\n"

        result += "**Interpretation:**\n"
        if upside > 20:
            result += f"- The DCF model suggests the stock is **significantly undervalued** by {upside:.0f}%\n"
            result += "- The market may be underestimating future cash flow potential\n"
        elif upside > 0:
            result += f"- The DCF model suggests the stock is **modestly undervalued** by {upside:.0f}%\n"
        elif upside > -20:
            result += f"- The DCF model suggests the stock is **fairly valued** or slightly overvalued\n"
        else:
            result += f"- The DCF model suggests the stock is **overvalued** by {abs(upside):.0f}%\n"
        result += "\n"

    # Method 2: Graham Number (Detailed)
    if balance_sheet and income_stmt:
        latest_bs = balance_sheet[0]
        latest_income = income_stmt[0]
        eps = latest_income.get("eps", 0) or 0
        total_equity = latest_bs.get("totalStockholdersEquity", 0)

        # Calculate book value per share properly
        if shares_outstanding > 0:
            book_value_per_share = total_equity / shares_outstanding
        else:
            book_value_per_share = 0

        result += "### 3.2 Graham Number (Value Investing)\n\n"
        result += "**Methodology:** Developed by Benjamin Graham, the father of value investing. It calculates the maximum price a defensive investor should pay for a stock.\n\n"
        result += "**Formula:**\n"
        result += "```\n"
        result += "Graham Number = √(22.5 × EPS × Book Value Per Share)\n"
        result += "```\n\n"
        result += "**Where:**\n"
        result += "- 22.5 = Graham's constant (derived from P/E of 15 × P/B of 1.5)\n"
        result += "- EPS = Earnings Per Share (trailing twelve months)\n"
        result += "- BVPS = Total Shareholders' Equity / Shares Outstanding\n\n"

        result += "**Step-by-Step Calculation:**\n"
        result += f"1. **EPS (Earnings Per Share):** ${eps:.2f}\n"
        result += f"   - Net Income: ${latest_income.get('netIncome', 0):,.0f}\n"
        result += f"   - Shares Outstanding: {shares_outstanding:,.0f}\n\n"

        result += f"2. **Book Value Per Share:** ${book_value_per_share:.2f}\n"
        result += f"   - Total Shareholders' Equity: ${total_equity:,.0f}\n"
        result += f"   - Shares Outstanding: {shares_outstanding:,.0f}\n"
        result += f"   - BVPS = ${total_equity:,.0f} / {shares_outstanding:,.0f} = ${book_value_per_share:.2f}\n\n"

        if eps > 0 and book_value_per_share > 0:
            graham_number = (22.5 * eps * book_value_per_share) ** 0.5
            fair_values["Graham Number"] = graham_number

            result += f"3. **Graham Number Calculation:**\n"
            result += f"   - Graham Number = √(22.5 × ${eps:.2f} × ${book_value_per_share:.2f})\n"
            result += f"   - Graham Number = √({22.5 * eps * book_value_per_share:,.2f})\n"
            result += f"   - **Graham Number = ${graham_number:.2f}**\n\n"

            upside = ((graham_number / current_price) - 1) * 100 if current_price else 0
            result += f"**Result:**\n"
            result += f"- Graham Fair Value: **${graham_number:.2f}**\n"
            result += f"- Current Price: ${current_price:.2f}\n"
            result += f"- Implied Upside/Downside: **{upside:+.1f}%**\n\n"

            result += "**Interpretation:**\n"
            if current_price < graham_number:
                result += f"- Current price is **below** the Graham Number, suggesting a margin of safety\n"
                result += f"- Graham would consider this potentially attractive for value investors\n"
            else:
                result += f"- Current price is **above** the Graham Number\n"
                result += f"- Graham would suggest waiting for a lower entry point\n"
            result += "\n"
        else:
            result += "- *Cannot calculate Graham Number: EPS or Book Value is negative/zero*\n\n"

    # Method 3: Relative Valuation (Peer Comparison) - Detailed
    if peer_quotes and not ratios_df.empty:
        result += "### 3.3 Relative Valuation (Peer P/E Comparison)\n\n"
        result += "**Methodology:** Compare the company's valuation multiples to sector peers to determine if it's trading at a premium or discount.\n\n"
        result += "**Formula:**\n"
        result += "```\n"
        result += "Fair Value = Company EPS × Sector Average P/E\n"
        result += "```\n\n"

        peer_data = [(q.symbol, q.pe, q.price) for q in peer_quotes if q.pe and q.pe > 0 and q.pe < 100]

        if peer_data:
            result += "**Peer P/E Ratios:**\n"
            result += "| Peer | P/E Ratio | Price |\n"
            result += "|------|-----------|-------|\n"
            for sym, pe, price in sorted(peer_data, key=lambda x: x[1])[:10]:
                result += f"| {sym} | {pe:.1f}x | ${price:.2f} |\n"
            result += "\n"

            peer_pes = [p[1] for p in peer_data]
            avg_peer_pe = sum(peer_pes) / len(peer_pes)
            median_peer_pe = sorted(peer_pes)[len(peer_pes)//2]
            min_peer_pe = min(peer_pes)
            max_peer_pe = max(peer_pes)

            eps = income_stmt[0].get("eps", 0) if income_stmt else 0
            company_pe = current_price / eps if eps > 0 else 0

            result += "**Peer Statistics:**\n"
            result += f"- Minimum P/E: {min_peer_pe:.1f}x\n"
            result += f"- Average P/E: {avg_peer_pe:.1f}x\n"
            result += f"- Median P/E: {median_peer_pe:.1f}x\n"
            result += f"- Maximum P/E: {max_peer_pe:.1f}x\n\n"

            result += f"**Company's Current P/E:** {company_pe:.1f}x\n\n"

            if eps > 0:
                fair_value_avg_pe = eps * avg_peer_pe
                fair_value_median_pe = eps * median_peer_pe
                fair_values["Peer Avg P/E"] = fair_value_avg_pe
                fair_values["Peer Median P/E"] = fair_value_median_pe

                result += "**Step-by-Step Calculation:**\n"
                result += f"1. Company EPS: ${eps:.2f}\n"
                result += f"2. Sector Average P/E: {avg_peer_pe:.1f}x\n"
                result += f"3. Sector Median P/E: {median_peer_pe:.1f}x\n\n"

                result += f"**Fair Value (Average P/E):**\n"
                result += f"- Fair Value = ${eps:.2f} × {avg_peer_pe:.1f} = **${fair_value_avg_pe:.2f}**\n"
                upside_avg = ((fair_value_avg_pe / current_price) - 1) * 100 if current_price else 0
                result += f"- Implied Upside/Downside: **{upside_avg:+.1f}%**\n\n"

                result += f"**Fair Value (Median P/E):**\n"
                result += f"- Fair Value = ${eps:.2f} × {median_peer_pe:.1f} = **${fair_value_median_pe:.2f}**\n"
                upside_median = ((fair_value_median_pe / current_price) - 1) * 100 if current_price else 0
                result += f"- Implied Upside/Downside: **{upside_median:+.1f}%**\n\n"

                result += "**Interpretation:**\n"
                if company_pe < avg_peer_pe:
                    discount = ((avg_peer_pe - company_pe) / avg_peer_pe) * 100
                    result += f"- Stock is trading at a **{discount:.0f}% discount** to sector average\n"
                    result += "- May indicate undervaluation or company-specific concerns\n"
                else:
                    premium = ((company_pe - avg_peer_pe) / avg_peer_pe) * 100
                    result += f"- Stock is trading at a **{premium:.0f}% premium** to sector average\n"
                    result += "- Premium may be justified by superior growth or quality\n"
                result += "\n"

    # Method 4: Custom DCF Model (Detailed)
    if cash_flow and balance_sheet:
        fcf = cash_flow[0].get("freeCashFlow", 0)

        if fcf > 0 and shares_outstanding > 0:
            result += "### 3.4 Custom DCF Model (5-Year Projection)\n\n"
            result += "**Methodology:** Project future free cash flows, discount to present value, add terminal value.\n\n"
            result += "**Formula:**\n"
            result += "```\n"
            result += "Enterprise Value = Σ (FCF × (1+g)^t / (1+r)^t) + Terminal Value\n"
            result += "Equity Value = Enterprise Value - Net Debt + Cash\n"
            result += "Fair Value Per Share = Equity Value / Shares Outstanding\n"
            result += "```\n\n"

            # Assumptions
            growth_rate = 0.10  # 10% growth
            terminal_growth = 0.025  # 2.5% terminal
            discount_rate = 0.10  # 10% WACC

            result += "**Assumptions:**\n"
            result += f"- Latest Free Cash Flow (FCF): ${fcf:,.0f}\n"
            result += f"- FCF Growth Rate (Years 1-5): {growth_rate*100:.0f}% annually\n"
            result += f"- Terminal Growth Rate: {terminal_growth*100:.1f}% (perpetuity)\n"
            result += f"- Discount Rate (WACC): {discount_rate*100:.0f}%\n\n"

            # Calculate projected FCFs
            result += "**Step 1: Project Future Cash Flows**\n"
            result += "| Year | FCF | Discount Factor | Present Value |\n"
            result += "|------|-----|-----------------|---------------|\n"

            projected_fcfs = []
            pv_fcfs = []
            for year in range(1, 6):
                projected_fcf = fcf * ((1 + growth_rate) ** year)
                discount_factor = 1 / ((1 + discount_rate) ** year)
                pv = projected_fcf * discount_factor
                projected_fcfs.append(projected_fcf)
                pv_fcfs.append(pv)
                result += f"| {year} | ${projected_fcf:,.0f} | {discount_factor:.4f} | ${pv:,.0f} |\n"

            sum_pv_fcf = sum(pv_fcfs)
            result += f"| **Total** | | | **${sum_pv_fcf:,.0f}** |\n\n"

            # Terminal Value
            result += "**Step 2: Calculate Terminal Value**\n"
            terminal_fcf = projected_fcfs[-1] * (1 + terminal_growth)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth)
            discount_factor_5 = 1 / ((1 + discount_rate) ** 5)
            pv_terminal = terminal_value * discount_factor_5

            result += f"- Year 5 FCF: ${projected_fcfs[-1]:,.0f}\n"
            result += f"- Terminal FCF (Year 6): ${projected_fcfs[-1]:,.0f} × (1 + {terminal_growth*100:.1f}%) = ${terminal_fcf:,.0f}\n"
            result += f"- Terminal Value = ${terminal_fcf:,.0f} / ({discount_rate*100:.0f}% - {terminal_growth*100:.1f}%) = ${terminal_value:,.0f}\n"
            result += f"- PV of Terminal Value = ${terminal_value:,.0f} × {discount_factor_5:.4f} = ${pv_terminal:,.0f}\n\n"

            # Enterprise Value
            result += "**Step 3: Calculate Enterprise Value**\n"
            enterprise_value = sum_pv_fcf + pv_terminal
            result += f"- PV of FCFs (Years 1-5): ${sum_pv_fcf:,.0f}\n"
            result += f"- PV of Terminal Value: ${pv_terminal:,.0f}\n"
            result += f"- **Enterprise Value: ${enterprise_value:,.0f}**\n\n"

            # Equity Value
            result += "**Step 4: Calculate Equity Value**\n"
            total_debt = balance_sheet[0].get("totalDebt", 0) if balance_sheet else 0
            cash = balance_sheet[0].get("cashAndCashEquivalents", 0) if balance_sheet else 0
            equity_value = enterprise_value - total_debt + cash

            result += f"- Enterprise Value: ${enterprise_value:,.0f}\n"
            result += f"- Less: Total Debt: (${total_debt:,.0f})\n"
            result += f"- Plus: Cash & Equivalents: ${cash:,.0f}\n"
            result += f"- **Equity Value: ${equity_value:,.0f}**\n\n"

            # Fair Value Per Share
            result += "**Step 5: Calculate Fair Value Per Share**\n"
            fair_value_dcf = equity_value / shares_outstanding
            fair_values["Custom DCF"] = fair_value_dcf

            result += f"- Equity Value: ${equity_value:,.0f}\n"
            result += f"- Shares Outstanding: {shares_outstanding:,.0f}\n"
            result += f"- **Fair Value Per Share: ${equity_value:,.0f} / {shares_outstanding:,.0f} = ${fair_value_dcf:.2f}**\n\n"

            upside = ((fair_value_dcf / current_price) - 1) * 100 if current_price else 0
            result += f"**Result:**\n"
            result += f"- DCF Fair Value: **${fair_value_dcf:.2f}**\n"
            result += f"- Current Price: ${current_price:.2f}\n"
            result += f"- Implied Upside/Downside: **{upside:+.1f}%**\n\n"

            result += "**Sensitivity Note:** DCF is sensitive to assumptions. A 1% change in:\n"
            result += "- Growth rate changes fair value by ~5-10%\n"
            result += "- Discount rate changes fair value by ~10-15%\n\n"

    # Method 5: Earnings Power Value (EPV)
    if income_stmt and not ratios_df.empty:
        result += "### 3.5 Earnings Power Value (EPV)\n\n"
        result += "**Methodology:** Values the company based on its current normalized earnings, assuming no growth.\n\n"
        result += "**Formula:**\n"
        result += "```\n"
        result += "EPV = Adjusted Earnings / Cost of Capital\n"
        result += "```\n\n"

        operating_income = income_stmt[0].get("operatingIncome", 0)
        tax_rate = 0.25  # Assume 25% tax rate
        adjusted_earnings = operating_income * (1 - tax_rate)
        cost_of_capital = 0.10  # 10% WACC

        result += "**Calculation:**\n"
        result += f"1. Operating Income (EBIT): ${operating_income:,.0f}\n"
        result += f"2. Tax Rate (assumed): {tax_rate*100:.0f}%\n"
        result += f"3. After-Tax Earnings: ${operating_income:,.0f} × (1 - {tax_rate*100:.0f}%) = ${adjusted_earnings:,.0f}\n"
        result += f"4. Cost of Capital (WACC): {cost_of_capital*100:.0f}%\n\n"

        if adjusted_earnings > 0:
            epv_enterprise = adjusted_earnings / cost_of_capital
            total_debt = balance_sheet[0].get("totalDebt", 0) if balance_sheet else 0
            cash = balance_sheet[0].get("cashAndCashEquivalents", 0) if balance_sheet else 0
            epv_equity = epv_enterprise - total_debt + cash

            result += f"5. EPV (Enterprise): ${adjusted_earnings:,.0f} / {cost_of_capital*100:.0f}% = ${epv_enterprise:,.0f}\n"
            result += f"6. Less Debt, Plus Cash: ${epv_enterprise:,.0f} - ${total_debt:,.0f} + ${cash:,.0f} = ${epv_equity:,.0f}\n"

            if shares_outstanding > 0:
                epv_per_share = epv_equity / shares_outstanding
                fair_values["EPV"] = epv_per_share
                upside = ((epv_per_share / current_price) - 1) * 100 if current_price else 0

                result += f"7. **EPV Per Share: ${epv_equity:,.0f} / {shares_outstanding:,.0f} = ${epv_per_share:.2f}**\n\n"
                result += f"**Result:** EPV Fair Value = **${epv_per_share:.2f}** ({upside:+.1f}% vs current)\n\n"

    # Fair Value Summary
    result += "### 3.6 Fair Value Summary\n\n"
    result += "| Method | Fair Value | vs Current Price | Assessment |\n"
    result += "|--------|------------|------------------|------------|\n"

    for method, fv in fair_values.items():
        diff = ((fv / current_price) - 1) * 100 if current_price else 0
        if diff > 20:
            status = "🟢 Significantly Undervalued"
        elif diff > 5:
            status = "🟢 Undervalued"
        elif diff > -5:
            status = "🟡 Fairly Valued"
        elif diff > -20:
            status = "🟠 Overvalued"
        else:
            status = "🔴 Significantly Overvalued"
        result += f"| {method} | ${fv:.2f} | {diff:+.1f}% | {status} |\n"

    if fair_values:
        avg_fair_value = sum(fair_values.values()) / len(fair_values)
        min_fair_value = min(fair_values.values())
        max_fair_value = max(fair_values.values())
        avg_diff = ((avg_fair_value / current_price) - 1) * 100 if current_price else 0

        result += f"\n**Valuation Range:**\n"
        result += f"- Low Estimate: ${min_fair_value:.2f}\n"
        result += f"- Average Estimate: **${avg_fair_value:.2f}**\n"
        result += f"- High Estimate: ${max_fair_value:.2f}\n"
        result += f"- Current Price: ${current_price:.2f}\n"
        result += f"- Average Implied Upside/Downside: **{avg_diff:+.1f}%**\n\n"

        result += "**Valuation Conclusion:**\n"
        methods_above = sum(1 for fv in fair_values.values() if fv > current_price)
        methods_below = len(fair_values) - methods_above

        if methods_above > methods_below:
            result += f"- {methods_above} of {len(fair_values)} valuation methods suggest the stock is **undervalued**\n"
        elif methods_below > methods_above:
            result += f"- {methods_below} of {len(fair_values)} valuation methods suggest the stock is **overvalued**\n"
        else:
            result += f"- Valuation methods are split, suggesting stock is **fairly valued**\n"
        result += "\n"

    # ===== QUANTITATIVE ANALYSIS =====
    result += "## 4. Quantitative Analysis\n\n"

    if not hist.empty:
        prices = hist["adjClose"].values

        # Technical Indicators
        result += "### 4.1 Technical Indicators\n\n"

        # Moving Averages
        sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
        sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else prices[-1]
        sma_200 = np.mean(prices[-200:]) if len(prices) >= 200 else prices[-1]

        result += "**Moving Averages:**\n"
        result += f"- 20-day SMA: ${sma_20:.2f} ({((current_price/sma_20)-1)*100:+.1f}%)\n"
        result += f"- 50-day SMA: ${sma_50:.2f} ({((current_price/sma_50)-1)*100:+.1f}%)\n"
        result += f"- 200-day SMA: ${sma_200:.2f} ({((current_price/sma_200)-1)*100:+.1f}%)\n"

        # Trend assessment
        if current_price > sma_20 > sma_50 > sma_200:
            result += "- Trend: 📈 **STRONG UPTREND** (Price > 20 > 50 > 200 SMA)\n"
        elif current_price < sma_20 < sma_50 < sma_200:
            result += "- Trend: 📉 **STRONG DOWNTREND** (Price < 20 < 50 < 200 SMA)\n"
        elif current_price > sma_200:
            result += "- Trend: ⬆️ **BULLISH** (Above 200 SMA)\n"
        else:
            result += "- Trend: ⬇️ **BEARISH** (Below 200 SMA)\n"

        # RSI
        changes = np.diff(prices)
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi = 100 - (100 / (1 + rs))

        result += f"\n**RSI (14): {rsi:.1f}**\n"
        if rsi > 70:
            result += "- Status: 🔴 OVERBOUGHT\n"
        elif rsi < 30:
            result += "- Status: 🟢 OVERSOLD\n"
        else:
            result += "- Status: ⚪ NEUTRAL\n"

        # Volatility
        returns = np.diff(prices) / prices[:-1]
        volatility_daily = np.std(returns)
        volatility_annual = volatility_daily * np.sqrt(252) * 100

        result += f"\n**Volatility:**\n"
        result += f"- Daily: {volatility_daily*100:.2f}%\n"
        result += f"- Annualized: {volatility_annual:.1f}%\n"

        # Risk Metrics
        result += "\n### 4.2 Risk Metrics\n\n"

        # VaR
        var_95 = np.percentile(returns, 5) * 100
        var_99 = np.percentile(returns, 1) * 100
        result += f"- 95% VaR (Daily): {var_95:.2f}%\n"
        result += f"- 99% VaR (Daily): {var_99:.2f}%\n"

        # Max Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns) * 100

        result += f"- Max Drawdown (1Y): {max_drawdown:.1f}%\n"
        result += f"- Beta: {profile.get('beta', 1.0):.2f}\n"

        # Sharpe approximation
        avg_return = np.mean(returns) * 252
        sharpe = (avg_return - 0.05) / (volatility_annual / 100) if volatility_annual > 0 else 0
        result += f"- Sharpe Ratio (Est): {sharpe:.2f}\n\n"

    # ===== ANALYST SENTIMENT =====
    result += "## 5. Analyst Sentiment\n\n"

    if estimates:
        result += "### 5.1 Earnings Estimates\n\n"
        for est in estimates[:2]:
            result += f"**{est.get('date', 'N/A')}:**\n"
            result += f"- Revenue Est: ${est.get('estimatedRevenueAvg', 0):,.0f}\n"
            result += f"- EPS Est: ${est.get('estimatedEpsAvg', 0):.2f}\n"

    if earnings:
        latest_earnings = earnings[0]
        actual = latest_earnings.get("actualEarningResult", 0)
        estimated = latest_earnings.get("estimatedEarning", 0)
        if estimated:
            surprise = ((actual - estimated) / abs(estimated)) * 100
            result += f"\n### 5.2 Latest Earnings\n"
            result += f"- Actual EPS: ${actual:.2f}\n"
            result += f"- Estimated EPS: ${estimated:.2f}\n"
            result += f"- Surprise: {surprise:+.1f}%\n"

    if recommendations:
        result += "\n### 5.3 Analyst Recommendations\n\n"
        for rec in recommendations[:3]:
            result += f"- {rec.get('date', 'N/A')}: {rec.get('analystRatingsStrongBuy', 0)} Strong Buy, "
            result += f"{rec.get('analystRatingsBuy', 0)} Buy, {rec.get('analystRatingsHold', 0)} Hold, "
            result += f"{rec.get('analystRatingsSell', 0)} Sell\n"

    # ===== FINAL RECOMMENDATION =====
    result += "\n## 6. Investment Recommendation\n\n"

    # Score calculation
    scores = []

    # Valuation score
    if fair_values:
        avg_fair_value = sum(fair_values.values()) / len(fair_values)
        valuation_score = min(100, max(0, 50 + (avg_fair_value / current_price - 1) * 100)) if current_price else 50
        scores.append(("Valuation", valuation_score))

    # Profitability score
    if not ratios_df.empty:
        r = ratios_df.iloc[0]
        roe = r.get('returnOnEquity', 0) * 100
        margin = r.get('netProfitMargin', 0) * 100
        prof_score = min(100, (roe * 2 + margin * 2))
        scores.append(("Profitability", prof_score))

    # Technical score
    if not hist.empty:
        tech_score = 50
        if current_price > sma_50:
            tech_score += 15
        if current_price > sma_200:
            tech_score += 15
        if 30 < rsi < 70:
            tech_score += 10
        elif rsi < 30:
            tech_score += 20  # Oversold opportunity
        scores.append(("Technical", min(100, tech_score)))

    # Risk score (inverse - lower risk = higher score)
    if not hist.empty:
        risk_score = max(0, 100 - volatility_annual)
        scores.append(("Risk-Adjusted", risk_score))

    if scores:
        overall_score = sum(s[1] for s in scores) / len(scores)

        result += "**Scoring Summary:**\n"
        for name, score in scores:
            bar = "█" * int(score/10) + "░" * (10 - int(score/10))
            result += f"- {name}: {bar} {score:.0f}/100\n"

        result += f"\n**Overall Score: {overall_score:.0f}/100**\n\n"

        # Recommendation
        result += "**RECOMMENDATION:** "
        if overall_score >= 70 and avg_fair_value > current_price * 1.1:
            result += "🟢 **STRONG BUY**\n"
            result += "- Fundamentals are strong\n"
            result += "- Stock appears undervalued\n"
            result += "- Technical setup is favorable\n"
        elif overall_score >= 60 and avg_fair_value > current_price:
            result += "🟢 **BUY**\n"
            result += "- Decent fundamentals\n"
            result += "- Trading below fair value\n"
        elif overall_score >= 50 or (avg_fair_value > current_price * 0.9 and avg_fair_value < current_price * 1.1):
            result += "🟡 **HOLD**\n"
            result += "- Trading near fair value\n"
            result += "- Wait for better entry or exit point\n"
        elif overall_score >= 40:
            result += "🟠 **REDUCE**\n"
            result += "- Some concerns with valuation or fundamentals\n"
            result += "- Consider taking partial profits\n"
        else:
            result += "🔴 **SELL**\n"
            result += "- Stock appears overvalued\n"
            result += "- Fundamental or technical concerns\n"

    result += "\n---\n*Disclaimer: This analysis is for informational purposes only and does not constitute investment advice.*\n"

    return result


@tool
def comprehensive_sector_analysis_tool(
    sector: str,
) -> str:
    """Perform complete sector analysis comparing all equities fundamentally and quantitatively.

    This tool analyzes an entire sector by:
    - Gathering fundamental data for all major companies in the sector
    - Calculating fair values for each company using multiple methods
    - Comparing valuations across the sector
    - Identifying the most undervalued and overvalued stocks
    - Providing sector-level insights and recommendations

    Use this tool when you need a complete picture of a sector and its best investment opportunities.

    Args:
        sector: Sector name (e.g., "Technology", "Healthcare", "Financial Services",
                "Consumer Cyclical", "Industrials", "Energy", "Basic Materials",
                "Communication Services", "Consumer Defensive", "Utilities", "Real Estate")

    Returns:
        String with comprehensive sector analysis including best opportunities.
    """
    data_pipeline, _ = _get_pipeline()

    async def fetch():
        # Get all stocks in sector
        stocks = await data_pipeline._stock_list.get_stock_screener(
            sector=sector,
            is_actively_trading=True,
            limit=50,
        )

        if not stocks:
            return None, None, None, None, None

        # Get top stocks by market cap
        symbols = [s["symbol"] for s in stocks[:30]]

        # Get quotes
        quotes = await data_pipeline.get_quotes_batch(symbols)

        # Get profiles
        profiles = await data_pipeline.get_company_profiles_batch(symbols[:25])

        # Get ratios for top companies
        ratios_data = {}
        for sym in symbols[:15]:
            try:
                ratios = await data_pipeline.get_financial_ratios(sym, "annual", limit=1)
                if not ratios.empty:
                    ratios_data[sym] = ratios.iloc[0].to_dict()
            except Exception:
                pass

        # Get DCF values
        dcf_data = {}
        for sym in symbols[:15]:
            try:
                dcf = await data_pipeline._company_info.get_dcf(sym)
                if dcf:
                    dcf_data[sym] = dcf
            except Exception:
                pass

        # Get sector performance
        sector_perf = await data_pipeline.get_sector_performance()

        await data_pipeline._client.close()
        return quotes, profiles, ratios_data, dcf_data, sector_perf

    quotes, profiles, ratios_data, dcf_data, sector_perf = _run_async(fetch())

    if not quotes:
        return f"No data found for sector: {sector}"

    import numpy as np

    result = f"# Comprehensive Sector Analysis: {sector}\n\n"

    # ===== SECTOR OVERVIEW =====
    result += "## 1. Sector Overview\n\n"

    # Sector performance
    sector_change = None
    for _, row in sector_perf.iterrows():
        if sector.lower() in row.get("sector", "").lower():
            sector_change = float(row.get("changesPercentage", "0").replace("%", ""))
            break

    if sector_change is not None:
        emoji = "📈" if sector_change >= 0 else "📉"
        result += f"**Sector Performance Today:** {emoji} {sector_change:+.2f}%\n\n"

    # Summary statistics
    total_mkt_cap = sum(q.market_cap or 0 for q in quotes)
    avg_pe = np.mean([q.pe for q in quotes if q.pe and 0 < q.pe < 100])
    avg_change = np.mean([q.changes_percentage or 0 for q in quotes])

    result += f"**Sector Statistics:**\n"
    result += f"- Total Market Cap: ${total_mkt_cap:,.0f}\n"
    result += f"- Average P/E: {avg_pe:.1f}x\n"
    result += f"- Average Stock Change: {avg_change:+.2f}%\n"
    result += f"- Companies Analyzed: {len(quotes)}\n\n"

    # ===== COMPANY COMPARISON =====
    result += "## 2. Company Comparison\n\n"

    # Build comparison table
    companies = []
    profile_lookup = {p.symbol: p for p in profiles}

    for q in quotes:
        company = {
            "symbol": q.symbol,
            "name": q.name,
            "price": q.price,
            "change": q.changes_percentage or 0,
            "mkt_cap": q.market_cap or 0,
            "pe": q.pe if q.pe and 0 < q.pe < 200 else None,
            "volume_ratio": (q.volume / q.avg_volume) if q.avg_volume else 1,
        }

        # Add profile data
        profile = profile_lookup.get(q.symbol)
        if profile:
            company["industry"] = profile.industry
            company["beta"] = profile.beta

        # Add ratios
        ratios = ratios_data.get(q.symbol, {})
        company["roe"] = ratios.get("returnOnEquity", 0)
        company["margin"] = ratios.get("netProfitMargin", 0)
        company["debt_equity"] = ratios.get("debtEquityRatio", 0)
        company["pb"] = ratios.get("priceToBookRatio", 0)
        company["ps"] = ratios.get("priceToSalesRatio", 0)

        # Add DCF data
        dcf = dcf_data.get(q.symbol, {})
        company["dcf_value"] = dcf.get("dcf", 0) if dcf else 0
        if company["dcf_value"] and company["price"]:
            company["dcf_upside"] = ((company["dcf_value"] / company["price"]) - 1) * 100
        else:
            company["dcf_upside"] = 0

        companies.append(company)

    # Sort by market cap
    companies.sort(key=lambda x: -x["mkt_cap"])

    # Top Companies Table
    result += "### 2.1 Top Companies by Market Cap\n\n"
    result += "| Company | Price | Change | P/E | ROE | Margin | DCF Value | Upside |\n"
    result += "|---------|-------|--------|-----|-----|--------|-----------|--------|\n"

    for c in companies[:15]:
        pe_str = f"{c['pe']:.1f}x" if c['pe'] else "N/A"
        roe_str = f"{c['roe']*100:.1f}%" if c['roe'] else "N/A"
        margin_str = f"{c['margin']*100:.1f}%" if c['margin'] else "N/A"
        dcf_str = f"${c['dcf_value']:.0f}" if c['dcf_value'] else "N/A"
        upside_str = f"{c['dcf_upside']:+.0f}%" if c['dcf_upside'] else "N/A"

        result += f"| **{c['symbol']}** | ${c['price']:.2f} | {c['change']:+.1f}% | {pe_str} | {roe_str} | {margin_str} | {dcf_str} | {upside_str} |\n"

    # ===== VALUATION ANALYSIS =====
    result += "\n## 3. Valuation Analysis\n\n"

    # Find undervalued stocks (DCF upside > 20%)
    undervalued = [c for c in companies if c["dcf_upside"] > 20]
    undervalued.sort(key=lambda x: -x["dcf_upside"])

    if undervalued:
        result += "### 3.1 Most Undervalued (DCF Upside > 20%)\n\n"
        for c in undervalued[:7]:
            result += f"**{c['symbol']}** - {c['name']}\n"
            result += f"- Current Price: ${c['price']:.2f}\n"
            result += f"- DCF Fair Value: ${c['dcf_value']:.2f}\n"
            result += f"- **Upside Potential: {c['dcf_upside']:+.1f}%**\n"
            if c['pe']:
                pe_vs_sector = ((c['pe'] / avg_pe) - 1) * 100
                result += f"- P/E: {c['pe']:.1f}x ({pe_vs_sector:+.1f}% vs sector avg)\n"
            result += "\n"

    # Find overvalued stocks (DCF upside < -20%)
    overvalued = [c for c in companies if c["dcf_upside"] < -20]
    overvalued.sort(key=lambda x: x["dcf_upside"])

    if overvalued:
        result += "### 3.2 Most Overvalued (DCF Downside > 20%)\n\n"
        for c in overvalued[:5]:
            result += f"**{c['symbol']}** - {c['name']}\n"
            result += f"- Current Price: ${c['price']:.2f}\n"
            result += f"- DCF Fair Value: ${c['dcf_value']:.2f}\n"
            result += f"- **Downside Risk: {c['dcf_upside']:.1f}%**\n\n"

    # ===== FUNDAMENTAL SCREENING =====
    result += "## 4. Fundamental Screening\n\n"

    # High Quality (High ROE, Good Margins)
    quality = [c for c in companies if c['roe'] and c['roe'] > 0.15 and c['margin'] and c['margin'] > 0.10]
    quality.sort(key=lambda x: -x['roe'])

    if quality:
        result += "### 4.1 Quality Leaders (ROE > 15%, Margin > 10%)\n\n"
        for c in quality[:5]:
            result += f"- **{c['symbol']}**: ROE {c['roe']*100:.1f}%, Margin {c['margin']*100:.1f}%\n"
        result += "\n"

    # Value Opportunities (Low P/E, Positive ROE)
    value = [c for c in companies if c['pe'] and c['pe'] < avg_pe * 0.7 and c['roe'] and c['roe'] > 0.05]
    value.sort(key=lambda x: x['pe'])

    if value:
        result += "### 4.2 Value Opportunities (Low P/E, Profitable)\n\n"
        for c in value[:5]:
            result += f"- **{c['symbol']}**: P/E {c['pe']:.1f}x (vs sector {avg_pe:.1f}x), ROE {c['roe']*100:.1f}%\n"
        result += "\n"

    # Momentum (Top Performers Today)
    momentum = sorted(companies, key=lambda x: -x['change'])[:5]

    result += "### 4.3 Momentum Leaders (Top Performers Today)\n\n"
    for c in momentum:
        vol_str = f"({c['volume_ratio']:.1f}x vol)" if c['volume_ratio'] > 1.5 else ""
        result += f"- **{c['symbol']}**: {c['change']:+.2f}% {vol_str}\n"
    result += "\n"

    # Laggards
    laggards = sorted(companies, key=lambda x: x['change'])[:5]

    result += "### 4.4 Today's Laggards (Potential Opportunities?)\n\n"
    for c in laggards:
        result += f"- **{c['symbol']}**: {c['change']:+.2f}%"
        if c['dcf_upside'] > 10:
            result += " *(Still undervalued per DCF)*"
        result += "\n"

    # ===== SECTOR METRICS =====
    result += "\n## 5. Sector Metrics Distribution\n\n"

    # P/E distribution
    pe_values = [c['pe'] for c in companies if c['pe']]
    if pe_values:
        result += "**P/E Ratio:**\n"
        result += f"- Low: {min(pe_values):.1f}x\n"
        result += f"- Median: {sorted(pe_values)[len(pe_values)//2]:.1f}x\n"
        result += f"- High: {max(pe_values):.1f}x\n"
        result += f"- Average: {np.mean(pe_values):.1f}x\n\n"

    # ROE distribution
    roe_values = [c['roe']*100 for c in companies if c['roe']]
    if roe_values:
        result += "**ROE (Return on Equity):**\n"
        result += f"- Low: {min(roe_values):.1f}%\n"
        result += f"- Median: {sorted(roe_values)[len(roe_values)//2]:.1f}%\n"
        result += f"- High: {max(roe_values):.1f}%\n"
        result += f"- Average: {np.mean(roe_values):.1f}%\n\n"

    # ===== TOP PICKS =====
    result += "## 6. Sector Top Picks\n\n"

    # Score each company
    scored_companies = []
    for c in companies:
        score = 0

        # Valuation score (DCF upside)
        if c['dcf_upside'] > 30:
            score += 30
        elif c['dcf_upside'] > 15:
            score += 20
        elif c['dcf_upside'] > 0:
            score += 10

        # Quality score (ROE)
        if c['roe'] and c['roe'] > 0.20:
            score += 25
        elif c['roe'] and c['roe'] > 0.15:
            score += 20
        elif c['roe'] and c['roe'] > 0.10:
            score += 15
        elif c['roe'] and c['roe'] > 0.05:
            score += 10

        # Margin score
        if c['margin'] and c['margin'] > 0.20:
            score += 20
        elif c['margin'] and c['margin'] > 0.10:
            score += 15
        elif c['margin'] and c['margin'] > 0.05:
            score += 10

        # Value score (P/E below sector)
        if c['pe'] and avg_pe:
            if c['pe'] < avg_pe * 0.7:
                score += 15
            elif c['pe'] < avg_pe * 0.9:
                score += 10

        # Low debt score
        if c['debt_equity'] and c['debt_equity'] < 0.5:
            score += 10

        c['score'] = score
        if score > 0:
            scored_companies.append(c)

    scored_companies.sort(key=lambda x: -x['score'])

    result += "### Best Investment Opportunities (Composite Score)\n\n"
    for i, c in enumerate(scored_companies[:10], 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        result += f"**{medal} {c['symbol']}** - Score: {c['score']}/100\n"
        result += f"   - Price: ${c['price']:.2f} | "
        result += f"DCF Upside: {c['dcf_upside']:+.0f}% | " if c['dcf_upside'] else ""
        result += f"P/E: {c['pe']:.1f}x | " if c['pe'] else ""
        result += f"ROE: {c['roe']*100:.0f}%\n" if c['roe'] else "\n"

    result += "\n---\n*Disclaimer: This analysis is for informational purposes only and does not constitute investment advice.*\n"

    return result


@tool
def correlation_analysis_tool(
    symbols: str,
    days: int = 60,
) -> str:
    """Analyze correlations between multiple securities.

    Use this tool to understand how different stocks, sectors, or asset classes
    move together. Helps identify diversification opportunities and hidden relationships.

    Args:
        symbols: Comma-separated symbols (e.g., "AAPL,MSFT,GOOGL,SPY,QQQ")
        days: Number of days for correlation calculation (default: 60)

    Returns:
        String with correlation matrix and analysis.
    """
    data_pipeline, _ = _get_pipeline()

    async def fetch():
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        to_date = date.today()
        from_date = to_date - timedelta(days=days + 10)

        returns_df = await data_pipeline.get_returns_matrix(symbol_list, str(from_date), str(to_date))
        await data_pipeline._client.close()
        return returns_df, symbol_list

    returns_df, symbol_list = _run_async(fetch())

    if returns_df.empty:
        return "Unable to fetch data for the specified symbols."

    import numpy as np

    result = f"**Correlation Analysis ({days} Days)**\n\n"
    result += f"Symbols: {', '.join(symbol_list)}\n\n"

    # Calculate correlation matrix
    corr_matrix = returns_df.corr()

    result += "**Correlation Matrix:**\n```\n"

    # Header
    result += " " * 8
    for sym in corr_matrix.columns:
        result += f"{sym:>8}"
    result += "\n"

    # Matrix
    for sym in corr_matrix.index:
        result += f"{sym:<8}"
        for col in corr_matrix.columns:
            val = corr_matrix.loc[sym, col]
            result += f"{val:>8.2f}"
        result += "\n"

    result += "```\n\n"

    # Find notable correlations
    high_corr = []
    low_corr = []
    negative_corr = []

    for i, sym1 in enumerate(corr_matrix.columns):
        for j, sym2 in enumerate(corr_matrix.columns):
            if i < j:  # Only upper triangle
                corr = corr_matrix.loc[sym1, sym2]
                pair = (sym1, sym2, corr)

                if corr > 0.8:
                    high_corr.append(pair)
                elif corr < 0.3 and corr > 0:
                    low_corr.append(pair)
                elif corr < 0:
                    negative_corr.append(pair)

    result += "**Key Findings:**\n\n"

    if high_corr:
        result += "**🔗 Highly Correlated Pairs (>0.8):**\n"
        for sym1, sym2, corr in sorted(high_corr, key=lambda x: -x[2]):
            result += f"- {sym1} & {sym2}: {corr:.2f}\n"
            result += f"  *These move together - limited diversification benefit*\n"
        result += "\n"

    if low_corr:
        result += "**✨ Low Correlation Pairs (<0.3):**\n"
        for sym1, sym2, corr in sorted(low_corr, key=lambda x: x[2]):
            result += f"- {sym1} & {sym2}: {corr:.2f}\n"
            result += f"  *Good diversification - different risk drivers*\n"
        result += "\n"

    if negative_corr:
        result += "**🔄 Negatively Correlated Pairs:**\n"
        for sym1, sym2, corr in sorted(negative_corr, key=lambda x: x[2]):
            result += f"- {sym1} & {sym2}: {corr:.2f}\n"
            result += f"  *Natural hedge - tend to move opposite*\n"
        result += "\n"

    # Volatility comparison
    result += "**Volatility Comparison (Annualized):**\n"
    for sym in symbol_list:
        if sym in returns_df.columns:
            vol = returns_df[sym].std() * np.sqrt(252) * 100
            result += f"- {sym}: {vol:.1f}%\n"

    # Portfolio suggestions
    result += "\n**Diversification Insights:**\n"
    avg_corr = corr_matrix.values[np.triu_indices(len(corr_matrix), k=1)].mean()
    result += f"- Average pairwise correlation: {avg_corr:.2f}\n"

    if avg_corr > 0.7:
        result += "- ⚠️ High average correlation - consider adding uncorrelated assets\n"
    elif avg_corr < 0.4:
        result += "- ✅ Good diversification - assets have distinct risk profiles\n"
    else:
        result += "- Moderate diversification - some shared risk factors\n"

    return result


# =============================================================================
# Advanced Portfolio Management Tools (Norte Asset Management Quant Finance Guide)
# =============================================================================


@tool
def risk_parity_portfolio_tool(
    symbols: str,
    risk_budgets: str = "",
) -> str:
    """Optimize portfolio using Risk Parity methodology.

    Risk Parity allocates weights so each asset contributes equally to portfolio risk.
    This approach provides better diversification than equal-weight or mean-variance.

    Args:
        symbols: Comma-separated stock symbols (e.g., "AAPL,MSFT,GOOGL,BND,GLD")
        risk_budgets: Optional comma-separated risk budgets (e.g., "0.25,0.25,0.25,0.25")
                     If empty, uses equal risk contribution.

    Returns:
        String with risk parity portfolio weights and analysis.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")]

    # Parse risk budgets if provided
    budgets = None
    if risk_budgets.strip():
        try:
            budgets = np.array([float(b.strip()) for b in risk_budgets.split(",")])
            budgets = budgets / budgets.sum()  # Normalize
        except ValueError:
            budgets = None

    data_pipeline, _ = _get_pipeline()

    async def fetch():
        prices = await data_pipeline.get_historical_prices_batch(symbol_list, limit=252)
        await data_pipeline._client.close()
        return prices

    prices_dict = _run_async(fetch())

    # Build returns dataframe
    import pandas as pd
    returns_data = {}
    for symbol, prices in prices_dict.items():
        if prices:
            closes = [p.close for p in sorted(prices, key=lambda x: x.date)]
            returns = np.diff(np.log(closes))
            returns_data[symbol] = returns

    if len(returns_data) < 2:
        return "Error: Need at least 2 symbols with valid data for optimization."

    # Align all returns to same length
    min_len = min(len(r) for r in returns_data.values())
    for sym in returns_data:
        returns_data[sym] = returns_data[sym][-min_len:]

    returns_df = pd.DataFrame(returns_data)
    cov_matrix = returns_df.cov().values * 252  # Annualize

    # Calculate risk parity portfolio
    result_data = PortfolioManagement.risk_parity_portfolio(
        cov_matrix=cov_matrix,
        risk_budgets=budgets,
    )

    # Build result string
    result = "**Risk Parity Portfolio Optimization**\n\n"
    result += "**Methodology:**\n"
    result += "Risk Parity allocates portfolio weights so that each asset contributes\n"
    result += "equally (or according to specified budgets) to total portfolio risk.\n\n"

    result += "**Formula:**\n"
    result += "```\n"
    result += "Minimize: Σᵢ(RCᵢ - bᵢ)²\n"
    result += "Where: RCᵢ = wᵢ × (Σw)ᵢ / σₚ (Risk Contribution of asset i)\n"
    result += "       bᵢ = target risk budget for asset i\n"
    result += "       σₚ = portfolio volatility\n"
    result += "```\n\n"

    result += "**Optimal Weights:**\n"
    for i, sym in enumerate(returns_df.columns):
        weight = result_data.weights[i]
        rc = result_data.risk_contributions[i]
        result += f"- {sym}: {weight*100:.1f}% (Risk Contribution: {rc*100:.1f}%)\n"

    result += f"\n**Portfolio Metrics:**\n"
    result += f"- Portfolio Volatility: {result_data.portfolio_volatility*100:.2f}%\n"

    # Show individual asset volatilities for comparison
    result += "\n**Individual Asset Volatilities:**\n"
    asset_vols = np.sqrt(np.diag(cov_matrix))
    for i, sym in enumerate(returns_df.columns):
        result += f"- {sym}: {asset_vols[i]*100:.1f}%\n"

    result += "\n**Why Risk Parity?**\n"
    result += "- Equal risk contribution provides true diversification\n"
    result += "- Avoids concentration in high-volatility assets\n"
    result += "- More stable in different market regimes\n"
    result += "- Bridgewater's All Weather strategy uses this approach\n"

    return result


@tool
def black_litterman_portfolio_tool(
    symbols: str,
    views: str,
    tau: float = 0.025,
) -> str:
    """Optimize portfolio using Black-Litterman model with investor views.

    Black-Litterman combines market equilibrium returns with investor views
    to create more stable and intuitive portfolio allocations.

    Args:
        symbols: Comma-separated stock symbols (e.g., "AAPL,MSFT,GOOGL")
        views: Investor views in format "SYMBOL:VIEW_RETURN:CONFIDENCE" (e.g.,
              "AAPL:0.15:0.8,MSFT:0.10:0.6" means AAPL expected to return 15%
              with 80% confidence, MSFT 10% with 60% confidence)
        tau: Scaling factor for uncertainty (default: 0.025)

    Returns:
        String with Black-Litterman portfolio weights and expected returns.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")]

    # Parse views
    view_dict = {}
    for item in views.split(","):
        parts = item.strip().split(":")
        if len(parts) >= 2:
            sym = parts[0].strip().upper()
            expected_return = float(parts[1].strip())
            confidence = float(parts[2].strip()) if len(parts) > 2 else 0.5
            view_dict[sym] = {"return": expected_return, "confidence": confidence}

    if not view_dict:
        return "Error: Invalid views format. Use 'SYMBOL:RETURN:CONFIDENCE'"

    data_pipeline, _ = _get_pipeline()

    async def fetch():
        prices = await data_pipeline.get_historical_prices_batch(symbol_list, limit=252)
        profiles = await data_pipeline.get_company_profiles_batch(symbol_list)
        await data_pipeline._client.close()
        return prices, profiles

    prices_dict, profiles = _run_async(fetch())

    # Build returns and covariance matrix
    import pandas as pd
    returns_data = {}
    for symbol, prices in prices_dict.items():
        if prices:
            closes = [p.close for p in sorted(prices, key=lambda x: x.date)]
            returns = np.diff(np.log(closes))
            returns_data[symbol] = returns

    if len(returns_data) < 2:
        return "Error: Need at least 2 symbols with valid data."

    # Align returns
    min_len = min(len(r) for r in returns_data.values())
    for sym in returns_data:
        returns_data[sym] = returns_data[sym][-min_len:]

    returns_df = pd.DataFrame(returns_data)
    cov_matrix = returns_df.cov().values * 252

    # Get market caps for equilibrium weights
    market_caps = {}
    for p in profiles:
        if p.symbol in returns_df.columns:
            market_caps[p.symbol] = p.mkt_cap or 1e10

    # Calculate market weights
    total_cap = sum(market_caps.values())
    market_weights = np.array([
        market_caps.get(sym, total_cap / len(symbol_list)) / total_cap
        for sym in returns_df.columns
    ])

    # Build P matrix (picking matrix) and Q vector (views)
    view_symbols = [s for s in view_dict.keys() if s in returns_df.columns]
    n_assets = len(returns_df.columns)
    n_views = len(view_symbols)

    if n_views == 0:
        return "Error: No valid views found for the provided symbols."

    P = np.zeros((n_views, n_assets))
    Q = np.zeros(n_views)
    omega_diag = []

    for i, sym in enumerate(view_symbols):
        col_idx = list(returns_df.columns).index(sym)
        P[i, col_idx] = 1.0
        Q[i] = view_dict[sym]["return"]
        # Uncertainty inversely proportional to confidence
        confidence = view_dict[sym]["confidence"]
        uncertainty = (1 - confidence) * 0.1  # Scale uncertainty
        omega_diag.append(max(uncertainty, 0.001))

    # Run Black-Litterman
    result_data = PortfolioManagement.black_litterman(
        market_weights=market_weights,
        cov_matrix=cov_matrix,
        P=P,
        Q=Q,
        tau=tau,
        omega=np.diag(omega_diag),
    )

    # Build result string
    result = "**Black-Litterman Portfolio Optimization**\n\n"
    result += "**Methodology:**\n"
    result += "Black-Litterman starts with market equilibrium returns and adjusts them\n"
    result += "based on your views, weighted by your confidence level.\n\n"

    result += "**Formula:**\n"
    result += "```\n"
    result += "E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [(τΣ)⁻¹π + P'Ω⁻¹Q]\n"
    result += "\n"
    result += "Where:\n"
    result += "  π = equilibrium excess returns (from CAPM)\n"
    result += "  Σ = covariance matrix\n"
    result += "  P = picking matrix (which assets have views)\n"
    result += "  Q = view returns\n"
    result += "  Ω = uncertainty in views\n"
    result += "  τ = scaling factor (typically 0.025)\n"
    result += "```\n\n"

    result += "**Your Views:**\n"
    for sym in view_symbols:
        v = view_dict[sym]
        result += f"- {sym}: {v['return']*100:.1f}% expected return ({v['confidence']*100:.0f}% confidence)\n"

    result += "\n**Equilibrium vs Black-Litterman Returns:**\n"
    result += "```\n"
    result += f"{'Symbol':<8} {'Market Eq.':<12} {'BL Adjusted':<12} {'Difference':<12}\n"
    result += "-" * 44 + "\n"
    for i, sym in enumerate(returns_df.columns):
        eq_ret = result_data.equilibrium_returns[i] * 100
        bl_ret = result_data.posterior_returns[i] * 100
        diff = bl_ret - eq_ret
        result += f"{sym:<8} {eq_ret:>10.2f}% {bl_ret:>10.2f}% {diff:>+10.2f}%\n"
    result += "```\n\n"

    result += "**Optimal Weights:**\n"
    result += "```\n"
    result += f"{'Symbol':<8} {'Market Wt.':<12} {'BL Weight':<12} {'Change':<12}\n"
    result += "-" * 44 + "\n"
    for i, sym in enumerate(returns_df.columns):
        mkt_wt = market_weights[i] * 100
        bl_wt = result_data.optimal_weights[i] * 100
        change = bl_wt - mkt_wt
        result += f"{sym:<8} {mkt_wt:>10.1f}% {bl_wt:>10.1f}% {change:>+10.1f}%\n"
    result += "```\n\n"

    result += "**Why Black-Litterman?**\n"
    result += "- Combines market consensus with your own views\n"
    result += "- Produces more intuitive, diversified portfolios\n"
    result += "- Handles uncertainty in forecasts explicitly\n"
    result += "- Views with low confidence have less impact\n"

    return result


@tool
def kelly_criterion_tool(
    symbols: str,
    fraction: float = 0.5,
    risk_free_rate: float = 0.05,
) -> str:
    """Calculate optimal position sizes using Kelly Criterion.

    Kelly Criterion determines the optimal fraction of capital to bet to maximize
    long-term growth rate. Full Kelly is often too aggressive, so a fractional
    Kelly (e.g., 0.5) is commonly used.

    Args:
        symbols: Comma-separated stock symbols (e.g., "AAPL,MSFT,GOOGL")
        fraction: Kelly fraction to use (0.5 = half-Kelly, recommended for stability)
        risk_free_rate: Risk-free rate (default: 0.05 = 5%)

    Returns:
        String with Kelly optimal weights and growth rate analysis.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")]

    data_pipeline, _ = _get_pipeline()

    async def fetch():
        prices = await data_pipeline.get_historical_prices_batch(symbol_list, limit=252)
        await data_pipeline._client.close()
        return prices

    prices_dict = _run_async(fetch())

    # Build returns
    import pandas as pd
    returns_data = {}
    for symbol, prices in prices_dict.items():
        if prices:
            closes = [p.close for p in sorted(prices, key=lambda x: x.date)]
            returns = np.diff(np.log(closes))
            returns_data[symbol] = returns

    if len(returns_data) < 1:
        return "Error: Need at least 1 symbol with valid data."

    min_len = min(len(r) for r in returns_data.values())
    for sym in returns_data:
        returns_data[sym] = returns_data[sym][-min_len:]

    returns_df = pd.DataFrame(returns_data)

    # Calculate expected returns and covariance
    expected_returns = returns_df.mean().values * 252
    cov_matrix = returns_df.cov().values * 252

    # Run Kelly Criterion
    result_data = PortfolioManagement.kelly_criterion(
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        risk_free_rate=risk_free_rate,
        fraction=fraction,
    )

    # Build result
    result = "**Kelly Criterion Portfolio Sizing**\n\n"
    result += "**Methodology:**\n"
    result += "Kelly Criterion maximizes the expected logarithmic growth rate of capital.\n"
    result += f"Using {fraction:.0%} Kelly (fractional Kelly for stability).\n\n"

    result += "**Formula:**\n"
    result += "```\n"
    result += "f* = Σ⁻¹ × (μ - rᶠ)\n"
    result += "\n"
    result += "Where:\n"
    result += "  f* = optimal fraction of capital for each asset\n"
    result += "  Σ⁻¹ = inverse covariance matrix\n"
    result += "  μ = expected returns vector\n"
    result += "  rᶠ = risk-free rate\n"
    result += "\n"
    result += "Kelly Fraction Applied:\n"
    result += f"  f_actual = {fraction} × f*\n"
    result += "```\n\n"

    result += "**Full Kelly vs Fractional Kelly Weights:**\n"
    result += "```\n"
    result += f"{'Symbol':<8} {'Full Kelly':<14} {f'{fraction:.0%} Kelly':<14} {'Exp.Return':<12}\n"
    result += "-" * 48 + "\n"
    for i, sym in enumerate(returns_df.columns):
        full_kelly = result_data["full_kelly_weights"][i] * 100
        frac_kelly = result_data["fractional_kelly_weights"][i] * 100
        exp_ret = expected_returns[i] * 100
        result += f"{sym:<8} {full_kelly:>12.1f}% {frac_kelly:>12.1f}% {exp_ret:>10.1f}%\n"
    result += "```\n\n"

    result += "**Growth Rate Analysis:**\n"
    result += f"- Expected Growth Rate (Full Kelly): {result_data['expected_growth_rate']*100:.2f}%\n"
    result += f"- Estimated Volatility: {np.sqrt(result_data['expected_growth_rate'])*100:.2f}% (approx)\n\n"

    # Cash allocation if weights < 100%
    total_weight = sum(result_data["fractional_kelly_weights"])
    if total_weight < 1.0:
        result += f"**Cash Allocation:** {(1-total_weight)*100:.1f}%\n"
        result += "(Remaining capital held in risk-free asset)\n\n"
    elif total_weight > 1.0:
        result += f"**Leverage Required:** {total_weight:.1f}x\n"
        result += "(Full Kelly often requires leverage - use with caution)\n\n"

    result += "**Why Use Fractional Kelly?**\n"
    result += "- Full Kelly maximizes growth but has high volatility\n"
    result += f"- {fraction:.0%} Kelly reduces volatility significantly\n"
    result += "- Only sacrifices a small amount of expected growth\n"
    result += "- More robust to estimation errors in returns\n"
    result += "- Warren Buffett: 'Rule #1: Never lose money'\n"

    return result


@tool
def monte_carlo_wealth_tool(
    initial_wealth: float,
    expected_return: float,
    volatility: float,
    years: int = 30,
    annual_contribution: float = 0,
    target_wealth: float = 0,
) -> str:
    """Run Monte Carlo simulation for wealth projection.

    Simulate thousands of potential wealth paths to understand the range of
    possible outcomes and probability of reaching financial goals.

    Args:
        initial_wealth: Starting portfolio value (e.g., 100000)
        expected_return: Expected annual return (e.g., 0.08 for 8%)
        volatility: Annual volatility (e.g., 0.15 for 15%)
        years: Investment horizon in years (default: 30)
        annual_contribution: Amount added each year (default: 0)
        target_wealth: Target wealth to calculate probability of reaching

    Returns:
        String with wealth simulation results and probabilities.
    """
    # Run simulation
    result_data = PortfolioManagement.monte_carlo_wealth_simulation(
        initial_wealth=initial_wealth,
        expected_return=expected_return,
        volatility=volatility,
        years=years,
        annual_contribution=annual_contribution,
        n_simulations=10000,
    )

    # Build result
    result = "**Monte Carlo Wealth Simulation**\n\n"
    result += "**Parameters:**\n"
    result += f"- Initial Wealth: ${initial_wealth:,.0f}\n"
    result += f"- Expected Annual Return: {expected_return*100:.1f}%\n"
    result += f"- Annual Volatility: {volatility*100:.1f}%\n"
    result += f"- Investment Horizon: {years} years\n"
    if annual_contribution > 0:
        result += f"- Annual Contribution: ${annual_contribution:,.0f}\n"
    result += f"- Simulations: 10,000\n\n"

    result += "**Methodology:**\n"
    result += "```\n"
    result += "W(t+1) = W(t) × exp[(μ - σ²/2)Δt + σ√Δt × Z]\n"
    result += "\n"
    result += "Where:\n"
    result += "  W(t) = wealth at time t\n"
    result += "  μ = expected return\n"
    result += "  σ = volatility\n"
    result += "  Z = standard normal random variable\n"
    result += "  Δt = time step (1 year)\n"
    result += "```\n\n"

    result += "**Wealth Projection (End of Period):**\n"
    result += "```\n"
    result += f"{'Percentile':<15} {'Wealth':<20} {'Total Return':<15}\n"
    result += "-" * 50 + "\n"

    percentiles = [5, 10, 25, 50, 75, 90, 95]
    for p in percentiles:
        key = f"p{p}"
        if key in result_data["final_wealth_percentiles"]:
            wealth = result_data["final_wealth_percentiles"][key]
            total_return = (wealth / initial_wealth - 1) * 100
            result += f"{p}th percentile  ${wealth:>15,.0f}  {total_return:>+12.1f}%\n"

    result += "```\n\n"

    result += "**Key Statistics:**\n"
    result += f"- Mean Final Wealth: ${result_data['mean_final_wealth']:,.0f}\n"
    result += f"- Median Final Wealth: ${result_data['median_final_wealth']:,.0f}\n"
    result += f"- Std Dev of Final Wealth: ${result_data['std_final_wealth']:,.0f}\n"
    result += f"- Probability of Loss: {result_data['probability_of_loss']*100:.1f}%\n\n"

    if target_wealth > 0:
        # Calculate probability of reaching target
        prob_target = result_data.get("probability_above_target", 0)
        if prob_target == 0:
            # Estimate from percentiles
            final_wealths = result_data.get("final_wealths", [])
            if final_wealths:
                prob_target = np.mean(np.array(final_wealths) >= target_wealth)

        result += f"**Goal Analysis:**\n"
        result += f"- Target Wealth: ${target_wealth:,.0f}\n"
        result += f"- Probability of Reaching Target: ~{prob_target*100:.1f}%\n\n"

    result += "**Interpretation:**\n"
    median = result_data['median_final_wealth']
    result += f"- 50% chance of having more than ${median:,.0f}\n"
    p10 = result_data['final_wealth_percentiles'].get('p10', median * 0.5)
    p90 = result_data['final_wealth_percentiles'].get('p90', median * 2)
    result += f"- 80% confidence range: ${p10:,.0f} to ${p90:,.0f}\n"
    loss_prob = result_data['probability_of_loss'] * 100
    if loss_prob > 10:
        result += f"- ⚠️ {loss_prob:.1f}% chance of losing money - consider more conservative allocation\n"
    else:
        result += f"- ✅ Only {loss_prob:.1f}% chance of losing money over {years} years\n"

    return result


@tool
def advanced_portfolio_optimization_tool(
    symbols: str,
    method: str = "all",
    risk_free_rate: float = 0.05,
) -> str:
    """Compare multiple portfolio optimization methods side-by-side.

    Runs Mean-Variance, Risk Parity, Maximum Diversification, and Kelly Criterion
    optimizations to compare different allocation strategies.

    Args:
        symbols: Comma-separated stock symbols (e.g., "AAPL,MSFT,GOOGL,BND")
        method: Optimization method - "all", "mean_variance", "risk_parity",
               "max_diversification", or "kelly" (default: "all")
        risk_free_rate: Risk-free rate for Sharpe calculation (default: 0.05)

    Returns:
        String comparing different optimization approaches.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")]

    data_pipeline, _ = _get_pipeline()

    async def fetch():
        prices = await data_pipeline.get_historical_prices_batch(symbol_list, limit=252)
        await data_pipeline._client.close()
        return prices

    prices_dict = _run_async(fetch())

    # Build returns
    import pandas as pd
    returns_data = {}
    for symbol, prices in prices_dict.items():
        if prices:
            closes = [p.close for p in sorted(prices, key=lambda x: x.date)]
            returns = np.diff(np.log(closes))
            returns_data[symbol] = returns

    if len(returns_data) < 2:
        return "Error: Need at least 2 symbols with valid data."

    min_len = min(len(r) for r in returns_data.values())
    for sym in returns_data:
        returns_data[sym] = returns_data[sym][-min_len:]

    returns_df = pd.DataFrame(returns_data)
    expected_returns = returns_df.mean().values * 252
    cov_matrix = returns_df.cov().values * 252
    n_assets = len(returns_df.columns)

    results = {}

    # Equal Weight (baseline)
    equal_weights = np.ones(n_assets) / n_assets
    eq_ret = np.dot(equal_weights, expected_returns)
    eq_vol = np.sqrt(np.dot(equal_weights, np.dot(cov_matrix, equal_weights)))
    eq_sharpe = (eq_ret - risk_free_rate) / eq_vol if eq_vol > 0 else 0
    results["Equal Weight"] = {
        "weights": equal_weights,
        "return": eq_ret,
        "volatility": eq_vol,
        "sharpe": eq_sharpe,
    }

    # Mean-Variance (Max Sharpe)
    if method in ["all", "mean_variance"]:
        try:
            mv_result = PortfolioManagement.mean_variance_optimization(
                expected_returns=expected_returns,
                cov_matrix=cov_matrix,
                risk_free_rate=risk_free_rate,
            )
            # Find max sharpe portfolio
            max_sharpe_idx = np.argmax(mv_result.sharpe_ratios)
            mv_weights = mv_result.efficient_weights[max_sharpe_idx]
            mv_ret = mv_result.efficient_returns[max_sharpe_idx]
            mv_vol = mv_result.efficient_volatilities[max_sharpe_idx]
            mv_sharpe = mv_result.sharpe_ratios[max_sharpe_idx]
            results["Mean-Variance (Max Sharpe)"] = {
                "weights": mv_weights,
                "return": mv_ret,
                "volatility": mv_vol,
                "sharpe": mv_sharpe,
            }
        except Exception:
            pass

    # Risk Parity
    if method in ["all", "risk_parity"]:
        try:
            rp_result = PortfolioManagement.risk_parity_portfolio(cov_matrix=cov_matrix)
            rp_weights = rp_result.weights
            rp_ret = np.dot(rp_weights, expected_returns)
            rp_vol = rp_result.portfolio_volatility
            rp_sharpe = (rp_ret - risk_free_rate) / rp_vol if rp_vol > 0 else 0
            results["Risk Parity"] = {
                "weights": rp_weights,
                "return": rp_ret,
                "volatility": rp_vol,
                "sharpe": rp_sharpe,
            }
        except Exception:
            pass

    # Maximum Diversification
    if method in ["all", "max_diversification"]:
        try:
            md_result = PortfolioManagement.maximum_diversification_portfolio(
                cov_matrix=cov_matrix
            )
            md_weights = md_result["weights"]
            md_ret = np.dot(md_weights, expected_returns)
            md_vol = md_result["portfolio_volatility"]
            md_sharpe = (md_ret - risk_free_rate) / md_vol if md_vol > 0 else 0
            results["Max Diversification"] = {
                "weights": md_weights,
                "return": md_ret,
                "volatility": md_vol,
                "sharpe": md_sharpe,
                "div_ratio": md_result["diversification_ratio"],
            }
        except Exception:
            pass

    # Kelly Criterion (Half Kelly)
    if method in ["all", "kelly"]:
        try:
            kelly_result = PortfolioManagement.kelly_criterion(
                expected_returns=expected_returns,
                cov_matrix=cov_matrix,
                risk_free_rate=risk_free_rate,
                fraction=0.5,
            )
            kelly_weights = np.array(kelly_result["fractional_kelly_weights"])
            # Normalize if sum > 1 for comparison
            if kelly_weights.sum() > 1:
                kelly_weights_norm = kelly_weights / kelly_weights.sum()
            else:
                kelly_weights_norm = kelly_weights
            kelly_ret = np.dot(kelly_weights_norm, expected_returns)
            kelly_vol = np.sqrt(np.dot(kelly_weights_norm, np.dot(cov_matrix, kelly_weights_norm)))
            kelly_sharpe = (kelly_ret - risk_free_rate) / kelly_vol if kelly_vol > 0 else 0
            results["Half-Kelly"] = {
                "weights": kelly_weights_norm,
                "return": kelly_ret,
                "volatility": kelly_vol,
                "sharpe": kelly_sharpe,
            }
        except Exception:
            pass

    # Build result string
    result = "**Advanced Portfolio Optimization Comparison**\n\n"
    result += f"Symbols: {', '.join(returns_df.columns)}\n"
    result += f"Risk-Free Rate: {risk_free_rate*100:.1f}%\n\n"

    # Weights comparison
    result += "**Optimal Weights by Method:**\n"
    result += "```\n"
    header = f"{'Symbol':<8}"
    for method_name in results.keys():
        short_name = method_name[:12]
        header += f"{short_name:>14}"
    result += header + "\n"
    result += "-" * (8 + 14 * len(results)) + "\n"

    for i, sym in enumerate(returns_df.columns):
        row = f"{sym:<8}"
        for method_name, data in results.items():
            weight = data["weights"][i] * 100
            row += f"{weight:>13.1f}%"
        result += row + "\n"
    result += "```\n\n"

    # Performance comparison
    result += "**Expected Performance:**\n"
    result += "```\n"
    result += f"{'Method':<24} {'Return':<10} {'Vol':<10} {'Sharpe':<10}\n"
    result += "-" * 54 + "\n"
    for method_name, data in results.items():
        ret = data["return"] * 100
        vol = data["volatility"] * 100
        sharpe = data["sharpe"]
        result += f"{method_name:<24} {ret:>8.1f}% {vol:>8.1f}% {sharpe:>8.2f}\n"
    result += "```\n\n"

    # Find best
    best_sharpe = max(results.items(), key=lambda x: x[1]["sharpe"])
    lowest_vol = min(results.items(), key=lambda x: x[1]["volatility"])
    highest_ret = max(results.items(), key=lambda x: x[1]["return"])

    result += "**Recommendations:**\n"
    result += f"- **Best Risk-Adjusted Return:** {best_sharpe[0]} (Sharpe: {best_sharpe[1]['sharpe']:.2f})\n"
    result += f"- **Lowest Volatility:** {lowest_vol[0]} ({lowest_vol[1]['volatility']*100:.1f}%)\n"
    result += f"- **Highest Expected Return:** {highest_ret[0]} ({highest_ret[1]['return']*100:.1f}%)\n\n"

    result += "**Method Descriptions:**\n"
    result += "- **Equal Weight:** Simple 1/N allocation - surprisingly robust\n"
    result += "- **Mean-Variance:** Markowitz optimization maximizing Sharpe ratio\n"
    result += "- **Risk Parity:** Equal risk contribution from each asset\n"
    result += "- **Max Diversification:** Maximizes diversification ratio\n"
    result += "- **Half-Kelly:** Growth-optimal sizing with reduced volatility\n"

    return result


@tool
def glide_path_tool(
    current_age: int,
    retirement_age: int = 65,
    risk_profile: str = "moderate",
) -> str:
    """Generate a lifecycle glide path for retirement investing.

    Creates an age-appropriate asset allocation that shifts from stocks to bonds
    as you approach retirement.

    Args:
        current_age: Current age of the investor
        retirement_age: Target retirement age (default: 65)
        risk_profile: "conservative", "moderate", or "aggressive"

    Returns:
        String with glide path allocations by age.
    """
    # Generate glide path
    glide_path = PortfolioManagement.generate_glide_path(
        current_age=current_age,
        retirement_age=retirement_age,
        risk_profile=risk_profile,
    )

    result = "**Lifecycle Glide Path**\n\n"
    result += f"**Profile:** {risk_profile.title()}\n"
    result += f"**Current Age:** {current_age}\n"
    result += f"**Retirement Age:** {retirement_age}\n"
    result += f"**Years to Retirement:** {max(0, retirement_age - current_age)}\n\n"

    result += "**Methodology:**\n"
    result += "Glide paths gradually shift from growth assets (stocks) to income assets\n"
    result += "(bonds) as you approach and enter retirement. This reduces sequence-of-returns\n"
    result += "risk near and during retirement.\n\n"

    result += "**Your Allocation Over Time:**\n"
    result += "```\n"
    result += f"{'Age':<6} {'Stocks':<10} {'Bonds':<10} {'Cash':<10}\n"
    result += "-" * 36 + "\n"

    # Show key ages
    ages_to_show = list(range(current_age, min(current_age + 50, 95), 5))
    if retirement_age not in ages_to_show:
        ages_to_show.append(retirement_age)
    ages_to_show = sorted(set(ages_to_show))

    for age in ages_to_show:
        if age in glide_path:
            alloc = glide_path[age]
            stocks = alloc.get("equity", 0) * 100
            bonds = alloc.get("fixed_income", 0) * 100
            cash = alloc.get("cash", 0) * 100
            marker = " <-- YOU" if age == current_age else (" <-- RETIRE" if age == retirement_age else "")
            result += f"{age:<6} {stocks:>8.0f}% {bonds:>8.0f}% {cash:>8.0f}%{marker}\n"

    result += "```\n\n"

    # Current allocation
    if current_age in glide_path:
        current_alloc = glide_path[current_age]
        result += "**Your Current Recommended Allocation:**\n"
        result += f"- Stocks/Equity: {current_alloc.get('equity', 0)*100:.0f}%\n"
        result += f"- Bonds/Fixed Income: {current_alloc.get('fixed_income', 0)*100:.0f}%\n"
        result += f"- Cash/Money Market: {current_alloc.get('cash', 0)*100:.0f}%\n\n"

    result += "**Risk Profile Differences:**\n"
    result += "- **Conservative:** Higher bond allocation, earlier de-risking\n"
    result += "- **Moderate:** Balanced approach (most common)\n"
    result += "- **Aggressive:** More equity, later de-risking for higher growth potential\n\n"

    result += "**Implementation Suggestions:**\n"
    result += "- **Stocks:** Total market index, S&P 500 (VTI, SPY, VOO)\n"
    result += "- **Bonds:** Aggregate bond index (BND, AGG)\n"
    result += "- **Cash:** Money market, Treasury bills\n"
    result += "- Consider target-date funds that automate this glide path\n"

    return result
