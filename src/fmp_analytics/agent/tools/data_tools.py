"""Data retrieval tools for financial information.

Tools for fetching quotes, company profiles, historical prices,
financial ratios, and market movers.
"""

from fmp_analytics.agent.tools._common import (
    date,
    timedelta,
    tool,
    get_pipeline,
    run_async,
)


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
    data_pipeline, _ = get_pipeline()

    async def fetch():
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        quotes = await data_pipeline.get_quotes_batch(symbol_list)
        await data_pipeline._client.close()
        return quotes

    quotes = run_async(fetch())

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
    data_pipeline, _ = get_pipeline()

    async def fetch():
        profile = await data_pipeline.get_company_profile(symbol.upper())
        await data_pipeline._client.close()
        return profile

    p = run_async(fetch())

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
    data_pipeline, _ = get_pipeline()

    async def fetch():
        to_date = date.today()
        from_date = to_date - timedelta(days=days)
        df = await data_pipeline.get_historical_prices(
            symbol.upper(), str(from_date), str(to_date)
        )
        await data_pipeline._client.close()
        return df

    df = run_async(fetch())

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
    data_pipeline, _ = get_pipeline()

    async def fetch():
        df = await data_pipeline.get_financial_ratios(symbol.upper(), period, limit=3)
        await data_pipeline._client.close()
        return df

    df = run_async(fetch())

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
    data_pipeline, _ = get_pipeline()

    async def fetch():
        movers = await data_pipeline.get_market_movers()
        await data_pipeline._client.close()
        return movers

    movers = run_async(fetch())

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
