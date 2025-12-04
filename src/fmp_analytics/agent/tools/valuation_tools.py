"""Valuation and pricing tools.

Tools for DCF valuation and options pricing analysis.
"""

from fmp_analytics.agent.tools._common import (
    tool,
    get_pipeline,
    run_async,
)


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
    data_pipeline, analysis_pipeline = get_pipeline()

    async def fetch():
        result = await analysis_pipeline.dcf_analysis(
            symbol.upper(),
            growth_rate,
            terminal_growth,
            discount_rate,
        )
        await data_pipeline._client.close()
        return result

    r = run_async(fetch())

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
    data_pipeline, analysis_pipeline = get_pipeline()

    async def fetch():
        result = await analysis_pipeline.options_analysis(
            symbol.upper(),
            strike,
            expiry_days,
            risk_free_rate,
        )
        await data_pipeline._client.close()
        return result

    r = run_async(fetch())

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
