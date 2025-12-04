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
