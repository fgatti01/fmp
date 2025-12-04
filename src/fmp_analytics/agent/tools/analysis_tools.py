"""Stock and company analysis tools.

Tools for analyzing individual stocks including fundamentals,
technicals, and valuation.
"""

from fmp_analytics.agent.tools._common import (
    tool,
    get_pipeline,
    run_async,
)


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
    data_pipeline, analysis_pipeline = get_pipeline()

    async def fetch():
        result = await analysis_pipeline.analyze_stock(symbol.upper())
        await data_pipeline._client.close()
        return result

    r = run_async(fetch())

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
